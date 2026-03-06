import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase, PreTrainedModel

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
    selected = F.log_softmax(logits, dim=-1)
    log_probs = selected.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    if not return_token_entropy:
        return {"log_probs": log_probs}
    
    token_entropy = compute_entropy(logits)
    return {"log_probs": log_probs, "token_entropy": token_entropy}

