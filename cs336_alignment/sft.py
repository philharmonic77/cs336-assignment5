import torch
from transformers import PreTrainedTokenizerBase

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
            1 if p_len <= i < total_len - 1 else 0
            for i in range(max_len - 1)
        ])

    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "response_mask": torch.tensor(response_mask),
    }