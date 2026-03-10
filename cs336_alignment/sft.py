import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase, PreTrainedModel,\
      AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from unittest.mock import patch
import json
import wandb
from pathlib import Path
from xopen import xopen
import random

def run_sft(
        model_path,
        train_jsonl_path,
        eval_data_path,
        prompt_path,
        batch_size,
        gradient_accumulation_steps,
        eval_intervals,
        lr,
        train_sample_size,
        output_dir,
        normalize_constant=1.0,
        max_grad_norm=1.0,
        only_use_correct=False,
        seed=0,
):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    num_micro_steps = train_sample_size // batch_size
    assert train_sample_size % batch_size == 0
    assert num_micro_steps % gradient_accumulation_steps == 0

    train_device = "cuda:0"
    eval_device = "cuda:1"

    wandb.init(project="a5-sft")
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    train_data = load_jsonl(train_jsonl_path)
    if only_use_correct:
        train_data = [d for d in train_data if r1_zero_reward_fn(d["response"], d["answer"])["reward"] == 1.0]
    print(f"Original data size: {len(train_data)}")

    subset_ids = random.choices(range(len(train_data)), k=train_sample_size)
    train_data = [train_data[i] for i in subset_ids]
    print(f"Sampled data size: {len(train_data)}")

    eval_data = load_jsonl(eval_data_path)

    policy = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2",
        local_files_only=True
        ).to(train_device)
    policy.config.use_cache = False
    policy.gradient_checkpointing_enable()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy.config.pad_token_id = tokenizer.pad_token_id

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=lr
    )

    model_path = str(model_path)
    llm = init_vllm(model_path, eval_device, seed, gpu_memory_utilization=0.7)
    eval_prompt_template = load_prompt_template(prompt_path)
    eval_prompts = format_prompts(eval_data, eval_prompt_template)

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    ground_truths = [get_ground_truth(ed) for ed in eval_data]


    train_step = 0
    eval_step = 0
    running_loss = 0.0

    optimizer.zero_grad(set_to_none=True)

    for micro_step in range(0, num_micro_steps):
        policy.train()

        batch = train_data[micro_step*batch_size:(micro_step+1)*batch_size]
        prompts = [x["prompt"] for x in batch]
        outputs = [x["response"] for x in batch]

        tokenized = tokenize_prompt_and_output(prompts, outputs, tokenizer)

        input_ids = tokenized["input_ids"].to(train_device)
        labels = tokenized["labels"].to(train_device)
        response_mask = tokenized["response_mask"].to(train_device)

        logprobs = get_response_log_probs(policy, input_ids, labels)["log_probs"]
        microbatch_loss, metadata = sft_microbatch_train_step(
            logprobs, response_mask, gradient_accumulation_steps, normalize_constant
        )

        running_loss += microbatch_loss.detach().item()

        if (micro_step + 1) % gradient_accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            train_step += 1

            wandb.log({
                "train_step": train_step,
                "train/loss": running_loss,
                "train/grad_norm": grad_norm.item()
            })

            if train_step % eval_intervals == 0:
                policy.eval()
                with torch.inference_mode():
                    load_policy_into_vllm_instance(policy, llm)
                eval_result = run_eval(llm, eval_prompts, sampling_params, ground_truths)
                eval_step += 1
                wandb.log({
                    "eval_step": eval_step,
                    **{f"eval/{k}": v for k, v in eval_result.items()}
                })
            running_loss = 0.0

    policy.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def run_eval(
    vllm_model: LLM,
    eval_prompts: list[str],
    sampling_params: SamplingParams,
    ground_truths: list,
) -> dict[str, float]:
    
    all_metrics: list[dict[str, float]] = []
    finish_reasons = []
    num_tokens = []

    chunk_size = 256
    for start in range(0, len(eval_prompts), chunk_size):
        end = start + chunk_size
        prompt_chunk = eval_prompts[start:end]
        gt_chunk = ground_truths[start:end]
        raw_outputs = vllm_model.generate(prompt_chunk, sampling_params)
        gen_chunk = [output.outputs[0].text for output in raw_outputs]

        for gen, gt, raw_output in zip(
                gen_chunk, gt_chunk, raw_outputs
            ):
            output0 = raw_output.outputs[0]
            token_ids = getattr(output0, "token_ids", None)

            finish_reasons.append(getattr(output0, "finish_reason", None))
            num_tokens.append(len(token_ids) if token_ids is not None else None)
            all_metrics.append(r1_zero_reward_fn(gen, gt))

    N = len(all_metrics)     
    frac_correct = sum(1 for m in all_metrics
        if m.get("format_reward") == 1.0 and m.get("answer_reward") == 1.0) / N
    frac_format_only = sum(1 for m in all_metrics
        if m.get("format_reward") == 1.0 and m.get("answer_reward") == 0.0) / N
    frac_zero = sum(1 for m in all_metrics
        if m.get("format_reward") == 0.0 and m.get("answer_reward") == 0.0) / N
    frac_answer_only = sum(1 for m in all_metrics if m.get("format_reward") == 0.0 and m.get("answer_reward") == 1.0) / N
    frac_stop = sum(1 for r in finish_reasons if r == "stop") / N
    frac_cut = sum(1 for r in finish_reasons if r != "stop") / N
    valid_num_tokens = [x for x in num_tokens if x is not None]
    avg_response_tokens = sum(valid_num_tokens) / len(valid_num_tokens) if valid_num_tokens else 0.0

    return {
        "frac_correct": frac_correct,
        "frac_format_only": frac_format_only,
        "frac_zero": frac_zero, 
        "frac_answer_only": frac_answer_only,
        "frac_stop": frac_stop,
        "frac_cut": frac_cut,
        "avg_response_tokens": avg_response_tokens
    }    

def load_jsonl(path):
    examples = []
    with xopen(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples

def load_prompt_template(path: str | Path) -> str:
    with xopen(path) as f:
        return f.read()


def get_question(example: dict) -> str:
    if "problem" in example and example["problem"] is not None:
        return str(example["problem"])
    raise KeyError("Could not find 'problem' field in example")


def get_ground_truth(example: dict):
    if "answer" in example and example["answer"] is not None:
        return example["answer"]
    raise KeyError("Could not find 'answer' field in example")


def format_prompts(examples: list[dict], prompt_template: str) -> list[str]:
    prompts = []
    for ex in examples:
        question = get_question(ex)
        prompts.append(prompt_template.format(question=question))
    return prompts


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )
    
def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    
    pad_id = tokenizer.pad_token_id

    combined = []
    prompt_lens = []
    for p, o in zip(prompt_strs, output_strs):
        p_ids = tokenizer.encode(p, add_special_tokens=False)
        o_ids = tokenizer.encode(o, add_special_tokens=False)
        combined.append(p_ids + o_ids)
        prompt_lens.append(len(p_ids))

    max_len = max(len(x) for x in combined)

    input_ids, labels, response_mask = [], [], []
    for ids, p_len in zip(combined, prompt_lens):
        total_len = len(ids)
        ids = ids + [pad_id] * (max_len - total_len)

        input_ids.append(ids[:-1])
        labels.append(ids[1:])
        response_mask.append([
            1 if p_len - 1 <= i < total_len - 1 else 0
            for i in range(max_len - 1)
        ])

    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "response_mask": torch.tensor(response_mask),
    }

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)   # log ∑ exp(z)
    log_probs = logits - log_z
    probs = torch.exp(log_probs)
    return -(probs * log_probs).sum(dim=-1)

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    
    logits = model(input_ids).logits # (batch_size, sequence_length, vocab_size)
    log_probs = F.log_softmax(logits, dim=-1)
    selected_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    if not return_token_entropy:
        return {"log_probs": selected_log_probs}
    
    token_entropy = compute_entropy(logits)
    return {"log_probs": selected_log_probs, "token_entropy": token_entropy}

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None= None,
) -> torch.Tensor:
    return (tensor * mask).sum(dim=dim) / normalize_constant

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    
    print("grad_acc_steps =", gradient_accumulation_steps)
    raw_loss = -masked_normalize(policy_log_probs, response_mask, normalize_constant)
    print("raw_loss =", raw_loss.item())
    loss = raw_loss / gradient_accumulation_steps
    print("returned_loss =", loss.item())
    loss.backward()

    metadata = {
        "num_response_tokens": response_mask.sum().detach(),
    }
    return loss, metadata

def log_generations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,  
):
    raise NotImplementedError

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run SFT training.")
    parser.add_argument("--model-path", type=Path, default=Path("models/Qwen2.5-Math-1.5B"))
    parser.add_argument("--train-jsonl-path", type=Path, default=Path("data/math/sft_format_correction.jsonl"))
    parser.add_argument("--eval-data-path", type=Path, default=Path("data/math/validation.jsonl"))
    parser.add_argument("--prompt-path", type=Path, default=Path("cs336_alignment/prompts/r1_zero.prompt"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--eval-intervals", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--train-sample-size", type=int, default=4096)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sft"))
    parser.add_argument("--normalize-constant", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--only-use-correct", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_sft(
        model_path=args.model_path,
        train_jsonl_path=args.train_jsonl_path,
        eval_data_path=args.eval_data_path,
        prompt_path=args.prompt_path,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_intervals=args.eval_intervals,
        lr=args.lr,
        train_sample_size=args.train_sample_size,
        output_dir=args.output_dir,
        normalize_constant=args.normalize_constant,
        max_grad_norm=args.max_grad_norm,
        only_use_correct=args.only_use_correct,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
