import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase, PreTrainedModel,\
      AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch
import json
import wandb
from pathlib import Path
from xopen import xopen
import random

def train_sft(
        model_path,
        train_jsonl_path,
        eval_data_path,
        batch_size,
        micro_batch_steps,
        gradient_accumulation_steps,
        normalize_constant,
        lr,
        max_grad_norm,
):
    train_device = "cuda:0"
    eval_device = "cuda:1"

    wandb.init(project="a5-sft")
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    policy = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2",
        local_files_only=True
        ).to(train_device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=lr
    )

    train_data = load_jsonl(train_jsonl_path)

    train_step = 0
    running_loss = 0.0
    while train_step < micro_batch_steps:
        policy.train()
        optimizer.zero_grad()

        prompts, outputs = get_batch(train_data, batch_size)    
        tokenized = tokenize_prompt_and_output(prompts, outputs, tokenizer)

        input_ids = tokenized["input_ids"].to(train_device)
        labels = tokenized["labels"].to(train_device)
        response_mask = tokenized["response_mask"].to(train_device)

        logprobs = get_response_log_probs(policy, input_ids, labels)["log_probs"]
        microbatch_loss, metadata = sft_microbatch_train_step(logprobs, response_mask, gradient_accumulation_steps, normalize_constant)

        running_loss += microbatch_loss.detach().item()

        if (train_step + 1) % gradient_accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            wandb.log({
                "train_step": (train_step + 1) // gradient_accumulation_steps,
                "train/loss": running_loss,
                "train/grad_norm": grad_norm.item()
            })
            running_loss = 0.0

        train_step += 1

    

def run_eval():

    raise NotImplementedError

def get_batch(dataset: list[dict], batch_size: int) -> tuple[list[str], list[str]]:
    n = len(dataset)
    sample_ids = [random.randint(0, n-1) for _ in range(batch_size)]
    samples = [dataset[i] for i in sample_ids]
    prompts, outputs = [x["prompt"] for x in samples], [x["response"] for x in samples]
    return prompts, outputs

def load_jsonl(path):
    examples = []
    with xopen(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


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
    
    loss = -masked_normalize(policy_log_probs, response_mask, normalize_constant)
    loss /= gradient_accumulation_steps
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


