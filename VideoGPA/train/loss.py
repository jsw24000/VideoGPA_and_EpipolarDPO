"""
DPO (Direct Preference Optimization) Loss for Diffusion Models

This file implements the DPO loss function for video diffusion models.
Core Idea: The model learns preferences by comparing the prediction error of winner/loser samples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LossOutput:
    """Output structure for the loss function"""
    loss: torch.Tensor                      # Total loss       
    reward_margin: torch.Tensor             # Reward margin (winner - loser)
    winner_reward: torch.Tensor             # Implicit reward for the winner
    loser_reward: torch.Tensor              # Implicit reward for the loser
    accuracy: torch.Tensor                  # Preference accuracy (ratio where winner reward > loser reward)
    metrics: Optional[Dict[str, torch.Tensor]] = None


class DPOLoss(nn.Module):
    """
    DPO Loss for Video Diffusion Models

    Based on velocity prediction error as an implicit reward signal.
    Core DPO Formula:
        L = -log(σ(β * (r_win - r_lose)))

    Where r = -(model_error - ref_error)
    """

    def __init__(
        self,
        beta: float = 500.0,
        label_smoothing: float = 0.0,
        loss_type: str = "sigmoid",
    ):
        """
        Args:
            beta: DPO temperature parameter, controls preference strength (larger is stronger).
            label_smoothing: Label smoothing to prevent overfitting (0-1).
            loss_type: Loss type, either "sigmoid" or "hinge".
        """
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type

    def forward(
        self,
        v_win: torch.Tensor,           # [B, F, C, H, W] - Winner prediction
        v_lose: torch.Tensor,          # [B, F, C, H, W] - Loser prediction
        v_win_ref: torch.Tensor,       # [B, F, C, H, W] - Winner reference prediction
        v_lose_ref: torch.Tensor,      # [B, F, C, H, W] - Loser reference prediction
        v_win_target: torch.Tensor,    # [B, F, C, H, W] - Winner ground truth
        v_lose_target: torch.Tensor,   # [B, F, C, H, W] - Loser ground truth
    ) -> LossOutput:
        """
        Compute DPO Loss.

        Core Logic:
        1. Calculate MSE as implicit reward (negative value, smaller error means higher reward).
        2. Compare the reward difference between the current model vs. reference model.
        3. Use DPO loss to enforce that the winner's reward is better than the loser's.
        """

        # 1. Calculate MSE Error (Averaged over all spatial and temporal dimensions)
        # Shape: [B, F, C, H, W] -> [B]
        model_win_err = (v_win - v_win_target).pow(2).mean(dim=[1, 2, 3, 4])
        model_lose_err = (v_lose - v_lose_target).pow(2).mean(dim=[1, 2, 3, 4])

        ref_win_err = (v_win_ref - v_win_target).pow(2).mean(dim=[1, 2, 3, 4])
        ref_lose_err = (v_lose_ref - v_lose_target).pow(2).mean(dim=[1, 2, 3, 4])

        # 2. Calculate improvement relative to the reference model
        # Positive value = Model is better than reference
        # Negative value = Model is worse than reference
        win_diff = ref_win_err - model_win_err      # Improvement on Winner
        lose_diff = ref_lose_err - model_lose_err   # Improvement on Loser

        # 3. Calculate Implicit Reward (Negative MSE)
        winner_reward = -model_win_err
        loser_reward = -model_lose_err
        reward_margin = winner_reward - loser_reward  # Expected to be positive

        # 4. DPO Loss
        # We want: win_diff > lose_diff (i.e., the winner improves more)
        # logits = β * (win_diff - lose_diff)
        logits = self.beta * (win_diff - lose_diff)

        if self.loss_type == "sigmoid":
            # Standard DPO: -log(σ(logits))
            if self.label_smoothing > 0:
                # Label smoothing: target = 1 - ε
                target = 1.0 - self.label_smoothing
                dpo_loss = F.binary_cross_entropy_with_logits(
                    logits,
                    torch.full_like(logits, target)
                )
            else:
                dpo_loss = -F.logsigmoid(logits).mean()
        elif self.loss_type == "hinge":
            # Hinge loss: max(0, 1 - logits)
            dpo_loss = F.relu(1.0 - logits).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # 5. Calculate Accuracy (Percentage of samples where winner reward > loser reward)
        accuracy = (winner_reward > loser_reward).float().mean()

        return LossOutput(
            loss=dpo_loss,
            reward_margin=reward_margin.mean(),
            winner_reward=winner_reward.mean(),
            loser_reward=loser_reward.mean(),
            accuracy=accuracy,
            metrics={
                "dpo_logits": logits.mean(),
                "dpo_beta": torch.as_tensor(self.beta, device=logits.device, dtype=logits.dtype),
            },
        )


class EpipolarDPOLoss(nn.Module):
    """
    Flow-DPO objective used by the upstream Epipolar-DPO reward LoRA trainer.

    The existing VideoGPA ``dpo`` strategy keeps its historical beta scaling.
    This strategy preserves the upstream 0.5 factor and exposes the Epipolar
    error terms in metrics so run metadata can audit the exact objective.
    """

    def __init__(self, beta: float = 500.0):
        super().__init__()
        self.beta = float(beta)

    def forward(
        self,
        v_win: torch.Tensor,
        v_lose: torch.Tensor,
        v_win_ref: torch.Tensor,
        v_lose_ref: torch.Tensor,
        v_win_target: torch.Tensor,
        v_lose_target: torch.Tensor,
    ) -> LossOutput:
        model_win_err = (v_win - v_win_target).pow(2).mean(dim=[1, 2, 3, 4])
        model_lose_err = (v_lose - v_lose_target).pow(2).mean(dim=[1, 2, 3, 4])
        ref_win_err = (v_win_ref - v_win_target).pow(2).mean(dim=[1, 2, 3, 4])
        ref_lose_err = (v_lose_ref - v_lose_target).pow(2).mean(dim=[1, 2, 3, 4])

        win_diff = model_win_err - ref_win_err
        lose_diff = model_lose_err - ref_lose_err
        inside_term = -0.5 * self.beta * (win_diff - lose_diff)
        dpo_loss = -F.logsigmoid(inside_term).mean()

        winner_reward = -win_diff
        loser_reward = -lose_diff
        reward_margin = winner_reward - loser_reward
        accuracy = (reward_margin > 0).float().mean()
        metric_dtype = inside_term.dtype
        metric_device = inside_term.device
        return LossOutput(
            loss=dpo_loss,
            reward_margin=reward_margin.mean(),
            winner_reward=winner_reward.mean(),
            loser_reward=loser_reward.mean(),
            accuracy=accuracy,
            metrics={
                "model_win_err": model_win_err.mean(),
                "model_lose_err": model_lose_err.mean(),
                "ref_win_err": ref_win_err.mean(),
                "ref_lose_err": ref_lose_err.mean(),
                "win_diff": win_diff.mean(),
                "lose_diff": lose_diff.mean(),
                "inside_term": inside_term.mean(),
                "beta": torch.as_tensor(self.beta, device=metric_device, dtype=metric_dtype),
                "half_factor": torch.as_tensor(0.5, device=metric_device, dtype=metric_dtype),
                "sign": torch.as_tensor(-1.0, device=metric_device, dtype=metric_dtype),
            },
        )


def create_loss_strategy(
    strategy: str = "dpo",
    beta: float = 1.0,
    label_smoothing: float = 0.0,
) -> nn.Module:
    """
    Factory function to create loss modules.

    Args:
        strategy: "dpo" or "sft"
        Other args: Same as DPOLoss

    Returns:
        Loss function module
    """
    if strategy == "dpo":
        return DPOLoss(beta=beta, label_smoothing=label_smoothing)
    elif strategy == "epipolar_dpo":
        return EpipolarDPOLoss(beta=beta)
    elif strategy == "sft":
        # SFT uses simple MSE loss
        class SFTLoss(nn.Module):
            def forward(self, v_pred, v_target, **kwargs):
                mse_loss = F.mse_loss(v_pred, v_target)
                return LossOutput(
                    loss=mse_loss,
                    reward_margin=torch.tensor(0.0),
                    winner_reward=torch.tensor(0.0),
                    loser_reward=torch.tensor(0.0),
                    accuracy=torch.tensor(0.0),
                    metrics={},
                )
        return SFTLoss()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


if __name__ == "__main__":
    # Test Loss Function
    print("Testing DPO Loss...")

    batch_size = 2
    num_frames = 49
    channels = 16
    height = 60
    width = 90

    # Create random test data
    v_win = torch.randn(batch_size, num_frames, channels, height, width)
    v_lose = torch.randn(batch_size, num_frames, channels, height, width)
    v_win_ref = torch.randn(batch_size, num_frames, channels, height, width)
    v_lose_ref = torch.randn(batch_size, num_frames, channels, height, width)
    v_target_win = torch.randn(batch_size, num_frames, channels, height, width)
    v_target_lose = torch.randn(batch_size, num_frames, channels, height, width)

    # Test DPO loss
    loss_fn = create_loss_strategy("dpo", beta=500.0)
    output = loss_fn(
        v_win, v_lose,
        v_win_ref, v_lose_ref,
        v_target_win, v_target_lose
    )

    print(f"DPO Loss: {output.loss.item():.4f}") # Changed from dpo_loss to loss to match dataclass
    print(f"Reward Margin: {output.reward_margin.item():.4f}")
    print(f"Accuracy: {output.accuracy.item():.4f}")
    print(f"Winner Reward: {output.winner_reward.item():.4f}")
    print(f"Loser Reward: {output.loser_reward.item():.4f}")
    print("\nTest passed!")
