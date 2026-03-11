import torch
from typing import Callable, Literal
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.sft import compute_entropy, get_response_log_probs

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