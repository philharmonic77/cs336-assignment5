import torch
from typing import Callable




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
    
    rollout_batch_size = len(rollout_responses)    
    raw_rewards = torch.zeros(rollout_batch_size, dtype=torch.float32) 

    for i, (re, gt) in enumerate(zip(rollout_responses, repeated_ground_truths)):
        raw_rewards[i] = reward_fn(re, gt)["reward"]

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
    "frac_nonzero_reward": (raw_rewards != 0).float().mean().item(),
    }

    if group_stds is not None:
        metadata["mean_group_std"] = group_stds.mean().item()

    return advantages, raw_rewards, metadata