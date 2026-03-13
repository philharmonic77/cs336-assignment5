import random
from pathlib import Path
import torch
import wandb
import typer
from torch.utils.data import Dataset, DataLoader
from typing import Callable, Literal
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.sft import load_jsonl, load_prompt_template, run_eval, get_ground_truth, format_prompts,\
    init_vllm, load_policy_into_vllm_instance, tokenize_prompt_and_output, get_response_log_probs, masked_normalize

def setup_grpo_environment(
    run_name,
    seed,
):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_device = "cuda:0"
    eval_device = "cuda:1"

    wandb.init(project="a5-grpo", name=run_name)
    wandb.define_metric("grpo_step")
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("grpo/*", step_metric="grpo_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    return train_device, eval_device

def setup_grpo_models_and_tokenizer(
    model_load_path,
    train_device,
    eval_device,
    seed,
    gpu_memory_utilization,
    learning_rate,
    sampling_temperature,
    sampling_min_tokens,
    sampling_max_tokens,
    group_size,
):
    policy = AutoModelForCausalLM.from_pretrained(
        model_load_path, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2",
        local_files_only=True
        ).to(train_device)
    policy.config.use_cache = False
    policy.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(model_load_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy.config.pad_token_id = tokenizer.pad_token_id

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        )

    rollout_llm = init_vllm(str(model_load_path), eval_device, seed, gpu_memory_utilization)
    train_sampling_params = SamplingParams(
        temperature=sampling_temperature,
        top_p=1.0,
        min_tokens=sampling_min_tokens,
        max_tokens=sampling_max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        n=group_size
    )

    eval_sampling_params = SamplingParams(
        temperature=sampling_temperature,
        top_p=1.0,
        min_tokens=sampling_min_tokens,
        max_tokens=sampling_max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        n=1
    )

    return policy, tokenizer, optimizer, rollout_llm, train_sampling_params, eval_sampling_params

def setup_grpo_data(
    prompt_path,
    train_jsonl_path,
    eval_json_path,
    eval_sample_size,
    n_prompts_per_rollout_batch,
):
    prompt_template = load_prompt_template(prompt_path)
    dataset = GrpoDataset(train_jsonl_path)
    dataloader = DataLoader(
        dataset,
        n_prompts_per_rollout_batch,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    dataloader_iter = iter(dataloader)

    eval_data = random.sample(load_jsonl(eval_json_path), eval_sample_size)
    eval_prompt_template = load_prompt_template(prompt_path)
    eval_prompts = format_prompts(eval_data, eval_prompt_template)
    eval_ground_truths = [get_ground_truth(ed) for ed in eval_data]

    return prompt_template, dataloader, dataloader_iter, eval_prompts, eval_ground_truths

def collect_rollouts_and_rewards(
    dataloader_iter,
    dataloader,
    prompt_template,
    rollout_llm,
    train_sampling_params,
    group_size,
    advantage_eps,
    normalize_by_std,
):
    try:
        batch = next(dataloader_iter)
    except StopIteration:
        dataloader_iter = iter(dataloader)
        batch = next(dataloader_iter)
    prompts = [prompt_template.format(question=p) for p in batch["problem"]]
    answers = batch["answer"]

    # 1. old policy rollout
    raw_outputs = rollout_llm.generate(prompts, train_sampling_params)
    rollout_responses = [
        candidate.text
        for output in raw_outputs
        for candidate in output.outputs
    ]

    # 2. repeat answers to match group_size   
    repeated_prompts = [p for p in prompts for _ in range(group_size)]    
    repeated_answers = [a for a in answers for _ in range(group_size)]

    # 3. compute rewards and advantages
    advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
        reward_fn=r1_zero_reward_fn,
        rollout_responses=rollout_responses,
        repeated_ground_truths=repeated_answers,
        group_size=group_size,
        advantage_eps=advantage_eps,
        normalize_by_std=normalize_by_std,
    )

    return dataloader_iter, repeated_prompts, rollout_responses, advantages, raw_rewards, reward_metadata

def prepare_training_tensors(
    repeated_prompts,
    rollout_responses,
    tokenizer,
    policy,
    train_device,
):
    tokenized = tokenize_prompt_and_output(
        repeated_prompts,
        rollout_responses,
        tokenizer,
    )
    input_ids = tokenized["input_ids"].to(train_device)
    labels = tokenized["labels"].to(train_device)
    response_mask = tokenized["response_mask"].to(train_device)

    with torch.no_grad():
        old_log_probs = get_response_log_probs(policy, input_ids, labels)["log_probs"].detach()

    return input_ids, labels, response_mask, old_log_probs

def train_on_rollout_batch(
    policy,
    optimizer,
    input_ids,
    labels,
    response_mask,
    advantages,
    raw_rewards,
    old_log_probs,
    train_batch_size,
    gradient_accumulation_steps,
    micro_train_batch_size,
    n_train_batches_per_rollout_batch,
    epochs_per_rollout_batch,
    loss_type,
    cliprange,
    max_grad_norm,
    grpo_step,
):
    train_device = input_ids.device
    for epoch_idx in range(epochs_per_rollout_batch):
        policy.train()

        for train_batch_idx in range(n_train_batches_per_rollout_batch):
            optimizer.zero_grad(set_to_none=True)
            train_batch_start = train_batch_idx * train_batch_size

            running_loss = 0.0
            all_is_clipped = []
            for micro_step in range(gradient_accumulation_steps):
                start = train_batch_start + micro_step * micro_train_batch_size
                end = start + micro_train_batch_size

                mb_input_ids = input_ids[start:end]
                mb_labels = labels[start:end]
                mb_response_mask = response_mask[start:end]
                mb_policy_log_probs = get_response_log_probs(policy, mb_input_ids, mb_labels)["log_probs"]

                mb_advantages = advantages[start:end].unsqueeze(1).to(train_device)
                mb_raw_rewards = raw_rewards[start:end].unsqueeze(1).to(train_device)
                mb_old_log_probs = old_log_probs[start:end]

                microbatch_loss, metadata = grpo_microbatch_train_step(
                    mb_policy_log_probs,
                    mb_response_mask,
                    gradient_accumulation_steps,
                    loss_type,
                    mb_raw_rewards,
                    mb_advantages,
                    mb_old_log_probs,
                    cliprange
                )
                running_loss += microbatch_loss.detach().item()
                if loss_type == "grpo_clip":
                    all_is_clipped.append(metadata["is_clipped"].float())

            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step() 

            train_step = grpo_step * epochs_per_rollout_batch * n_train_batches_per_rollout_batch \
                + epoch_idx * n_train_batches_per_rollout_batch \
                + train_batch_idx
            
            train_batch_start = train_batch_idx * train_batch_size
            train_batch_end = train_batch_start + train_batch_size

            train_batch_entropy = masked_mean(
                get_response_log_probs(
                    policy,
                    input_ids[train_batch_start:train_batch_end],
                    labels[train_batch_start:train_batch_end],
                    True,
                )["token_entropy"],
                response_mask[train_batch_start:train_batch_end],
            ).item()
            
            wandb.log({
            "train_step": train_step,
            "train/loss": running_loss,
            "train/grad_norm": grad_norm.item(),
            "train/token_entropy": train_batch_entropy
        })
            if loss_type == "grpo_clip":
                wandb.log({
                    "train_step": train_step,
                    "train/clip_fraction": torch.cat(
                            [x.reshape(-1) for x in all_is_clipped]
                        ).mean().item(),
                })

    return train_step

def grpo_train_loop(
    run_name: str,
    prompt_path: Path = Path("cs336_alignment/prompts/r1_zero.prompt"),
    model_load_path: Path = Path("models/Qwen2.5-Math-1.5B"),
    model_save_path: Path = Path("outputs/grpo"),
    train_jsonl_path: Path = Path("data/math/sft_format_correction.jsonl"),
    group_size: int = 8,
    rollout_batch_size: int = 256,
    train_batch_size: int = 256,
    gradient_accumulation_steps: int = 256,
    n_grpo_steps: int = 200,
    epochs_per_rollout_batch: int = 1,
    learning_rate: float = 1e-5,
    max_grad_norm: float = 1.0,
    advantage_eps: float = 1e-6,
    normalize_by_std: bool = True,
    loss_type: str = "reinforce_with_baseline",
    cliprange: float = 0.2,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,
    sampling_max_tokens: int = 1024,
    gpu_memory_utilization: float = 0.85,
    eval_json_path: Path = Path("data/math/validation.jsonl"),
    eval_interval: int = 5,
    eval_sample_size: int = 1024,
    seed: int = 0,
):
    assert loss_type in {
        "no_baseline",
        "reinforce_with_baseline",
        "grpo_clip",
    }, "loss_type must be one of: no_baseline, reinforce_with_baseline, grpo_clip"
    assert train_batch_size % gradient_accumulation_steps == 0, (
    "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps

    assert rollout_batch_size % group_size == 0, (
    "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = rollout_batch_size // group_size

    assert train_batch_size >= group_size, (
    "train_batch_size must be greater than or equal to group_size"
    )
    assert rollout_batch_size % train_batch_size == 0, (
    "rollout_batch_size must be divisible by train_batch_size"
    )
    n_train_batches_per_rollout_batch = rollout_batch_size // train_batch_size

    train_device, eval_device = setup_grpo_environment(
        run_name,
        seed,
    )

    policy, tokenizer, optimizer, rollout_llm, train_sampling_params, eval_sampling_params = setup_grpo_models_and_tokenizer(
        model_load_path,
        train_device,
        eval_device,
        seed,
        gpu_memory_utilization,
        learning_rate,
        sampling_temperature,
        sampling_min_tokens,
        sampling_max_tokens,
        group_size,
    )

    prompt_template, dataloader, dataloader_iter, eval_prompts, eval_ground_truths = setup_grpo_data(
        prompt_path,
        train_jsonl_path,
        eval_json_path,
        eval_sample_size,
        n_prompts_per_rollout_batch,
    )

    eval_step = 0
    for grpo_step in range(n_grpo_steps):

        dataloader_iter, repeated_prompts, rollout_responses, advantages, raw_rewards, reward_metadata = collect_rollouts_and_rewards(
            dataloader_iter,
            dataloader,
            prompt_template,
            rollout_llm,
            train_sampling_params,
            group_size,
            advantage_eps,
            normalize_by_std,
        )
        wandb.log({
            "grpo_step": grpo_step,
            "grpo/mean_raw_reward": reward_metadata["mean_raw_reward"],
            "grpo/mean_format_reward": reward_metadata["mean_format_reward"],
            "grpo/mean_answer_reward": reward_metadata["mean_answer_reward"],
        })

        input_ids, labels, response_mask, old_log_probs = prepare_training_tensors(
            repeated_prompts,
            rollout_responses,
            tokenizer,
            policy,
            train_device,
        )

        train_on_rollout_batch(
            policy,
            optimizer,
            input_ids,
            labels,
            response_mask,
            advantages,
            raw_rewards,
            old_log_probs,
            train_batch_size,
            gradient_accumulation_steps,
            micro_train_batch_size,
            n_train_batches_per_rollout_batch,
            epochs_per_rollout_batch,
            loss_type,
            cliprange,
            max_grad_norm,
            grpo_step,
        )

        # 7. reload rollout model              
        load_policy_into_vllm_instance(policy, rollout_llm)
        
        if (1 + grpo_step) % eval_interval == 0:
            eval_result = run_eval(rollout_llm, eval_prompts, eval_sampling_params, eval_ground_truths)
            eval_step += 1
            wandb.log({
                "eval_step": eval_step,
                **{f"eval/{k}": v for k, v in eval_result.items()}
            })

    model_save_path = Path(model_save_path)
    model_save_path.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

class GrpoDataset(Dataset):
    def __init__(self, path: str | Path):
        self.examples = load_jsonl(path)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "problem": ex["problem"],
            "answer": ex["answer"],
        }  
    
def collate_fn(batch):
    return {
        "problem": [x["problem"] for x in batch],
        "answer": [x["answer"] for x in batch],
    }  

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None= None,
    advantages: torch.Tensor | None= None,
    old_log_probs: torch.Tensor | None= None,
    cliprange: float | None= None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    
    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange
    )   
    per_example_loss = masked_mean(
        per_token_loss,
        response_mask,
        dim=-1
    ) # shape: (batch,)
    loss = per_example_loss.mean()
    loss = loss / gradient_accumulation_steps
    loss.backward()

    return loss, metadata

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None= None,
) -> torch.Tensor:
    masked_sum = (tensor * mask).sum(dim=dim)
    mask_count = mask.sum(dim=dim)
    return masked_sum / mask_count

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None= None,
    advantages: torch.Tensor | None= None,
    old_log_probs: torch.Tensor | None= None,
    cliprange: float | None= None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    
    if loss_type == "no_baseline": 
        assert raw_rewards is not None
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
    if loss_type == "reinforce_with_baseline":
        assert advantages is not None
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}
    if loss_type == "grpo_clip":
        assert advantages is not None
        assert old_log_probs is not None
        assert cliprange is not None
        return compute_grpo_clip_loss(
            advantages,
            policy_log_probs,
            old_log_probs,
            cliprange
        )

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    return -raw_rewards_or_advantages * policy_log_probs

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    
    assert policy_log_probs.shape == old_log_probs.shape 

    metadata = {}  
    
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)

    lhs = ratio * advantages
    rhs = clipped_ratio * advantages
    loss = -torch.minimum(lhs, rhs)

    metadata["is_clipped"] = rhs < lhs

    return loss, metadata

    
def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]] ,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    assert len(rollout_responses) == len(repeated_ground_truths)
    assert len(rollout_responses) % group_size == 0
    
    rollout_train_batch_size = len(rollout_responses)    
    raw_rewards = torch.zeros(rollout_train_batch_size, dtype=torch.float32)
    raw_format_rewards = torch.zeros(rollout_train_batch_size, dtype=torch.float32)
    raw_answer_rewards = torch.zeros(rollout_train_batch_size, dtype=torch.float32)

    for i, (re, gt) in enumerate(zip(rollout_responses, repeated_ground_truths)):
        reward_dict = reward_fn(re, gt)
        raw_rewards[i] = reward_dict["reward"]
        raw_format_rewards[i] = reward_dict["format_reward"]
        raw_answer_rewards[i] = reward_dict["answer_reward"]

    grouped = raw_rewards.view(-1, group_size)                    
    group_means = grouped.mean(dim=1, keepdim=True)            
    advantages = grouped - group_means 

    if normalize_by_std:
       group_stds = advantages.std(dim=1, keepdim=True) + advantage_eps
       advantages /= group_stds
    else:
        group_stds = None

    advantages = advantages.reshape(-1)
    metadata = {
    "mean_raw_reward": raw_rewards.mean().item(),
    "mean_format_reward": raw_format_rewards.mean().item(),
    "mean_answer_reward": raw_answer_rewards.mean().item(),
    }

    if group_stds is not None:
        metadata["mean_group_std"] = group_stds.mean().item()

    return advantages, raw_rewards, metadata

if __name__ == "__main__":
    typer.run(grpo_train_loop)
