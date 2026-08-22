"""BaseVQC — VQC 模型超類別，含張量形狀廣播防禦（YC 約束四）

BLUEPRINT §9.1 — BaseVQC 重構
- 張量形狀廣播防禦（YC 約束四）：1D 張量自動 unsqueeze
- 電路外縮放（Failure Mode 2 防禦）：lam * normalized 在電路外執行
- preprocess_observations / head_transform 可 override
"""

from __future__ import annotations

import torch
import torch.nn as nn
from abc import abstractmethod
from typing import Callable

from qrl.models.base import normalize_observation


class BaseVQC(nn.Module):
    """
    VQC 身體：電路 + backend + lam（input scaling）。

    Subclass 只需 override：
    1. head_transform() — 把 expectation values 轉成輸出
    2. preprocess_observations() — 編碼器（可 override）

    張量形狀廣播防禦（YC 約束四）：
    - 1D 輸入自動 unsqueeze(0) 擴展為 [1, n_qubits]
    - 斷言：Encoder 輸出維度必須與 VQC 輸入維度（lam.shape[0]）匹配
    - 縮放在電路外執行（Failure Mode 2 防禦）
    """

    def __init__(
        self,
        vqc: nn.Module,
        lam: nn.Parameter,
        preprocess_observations: Callable[[torch.Tensor], torch.Tensor] | None = None,
        head_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        """
        Initialize BaseVQC.

        Args:
            vqc: The VQC backend module (e.g., _TorchStatevectorBackend).
            lam: Trainable input scaling parameter [n_encoding].
            preprocess_observations: Observation encoder. Defaults to normalize_observation (arctan).
            head_transform: Head transformation. Defaults to identity.
        """
        super().__init__()
        self.vqc = vqc
        self.lam = lam
        self.preprocess_observations = preprocess_observations or self._default_preprocess
        self.head_transform = head_transform or (lambda x: x)

    def _default_preprocess(self, states: torch.Tensor) -> torch.Tensor:
        """Default: normalize to arctan bounded angles (LOCKED normalize_observation)."""
        return normalize_observation(states)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with tensor shape broadcasting defense.

        Args:
            states: Raw observation tensor. Can be [n_qubits] (1D) or [B, n_qubits] (2D).

        Returns:
            Transformed expectation values [B, n_obs] or [n_obs] for 1D input.
        """
        # ----------------------------------------------------------
        # 張量形狀廣播防禦（YC 約束四）
        # 1D → 自動 unsqueeze(0)，防止 shape mismatch
        # ----------------------------------------------------------
        if states.dim() == 1:
            states = states.unsqueeze(0)

        normalized = self.preprocess_observations(states)

        # ----------------------------------------------------------
        # 斷言：Encoder 輸出維度必須與 VQC 輸入維度匹配
        # ----------------------------------------------------------
        assert normalized.shape[-1] == self.lam.shape[0], (
            f"Encoder 輸出維度 ({normalized.shape[-1]}) 與 "
            f"VQC 輸入維度 ({self.lam.shape[0]}) 不匹配"
        )

        # ----------------------------------------------------------
        # 電路外縮放（Failure Mode 2 防禦）
        # lam * normalized 必须在电路外执行，否则破坏梯度
        # ----------------------------------------------------------
        scaled = self.lam * normalized

        # ----------------------------------------------------------
        # 通過 VQC 電路
        # ----------------------------------------------------------
        expvals = self.vqc(scaled)

        return self.head_transform(expvals)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def build_base_vqc(
    vqc: nn.Module,
    lam: nn.Parameter,
    preprocess: Callable[[torch.Tensor], torch.Tensor] | None = None,
    head_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> BaseVQC:
    """
    Build a BaseVQC instance.

    Args:
        vqc: The VQC backend module.
        lam: Trainable input scaling parameter.
        preprocess: Optional observation encoder override.
        head_transform: Optional head transformation override.

    Returns:
        BaseVQC instance.
    """
    return BaseVQC(vqc, lam, preprocess, head_transform)
