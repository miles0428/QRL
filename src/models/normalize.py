"""Shared CartPole observation normalizer.

Both the VQC and the MLP baseline use THIS, so "same normalization pipeline" is
literally true and the comparison isolates the approximator. Bounded dims (cart
position, pole angle) are divided by their termination bounds; the two unbounded
velocity dims are squashed with arctan/tanh (never raw division). Operates on raw
observations; the replay buffer stores raw obs.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CartPoleNormalizer(nn.Module):
    """[B, 4] raw CartPole obs -> [B, 4] normalized. No trainable parameters."""

    def __init__(self, norm_cfg: dict):
        super().__init__()
        self.cart_pos_bound = float(norm_cfg["cart_position_bound"])
        self.pole_angle_bound = float(norm_cfg["pole_angle_bound"])
        vt = str(norm_cfg["velocity_transform"]).lower()
        if vt not in ("arctan", "tanh"):
            raise ValueError(f"velocity_transform must be 'arctan' or 'tanh', got {vt!r}")
        self._transform = torch.arctan if vt == "arctan" else torch.tanh

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # CartPole obs order: [cart x, cart v, pole angle, pole angular v]
        return torch.stack(
            [
                s[:, 0] / self.cart_pos_bound,   # cart position (bounded)
                self._transform(s[:, 1]),        # cart velocity (unbounded)
                s[:, 2] / self.pole_angle_bound, # pole angle (bounded)
                self._transform(s[:, 3]),        # pole angular velocity (unbounded)
            ],
            dim=1,
        )
