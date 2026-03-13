from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch

from llm_rl.rollout.rollout_buffer import RolloutBatch


GradClipFn = Callable[[torch.nn.Module, float], float]


def _default_grad_clip(model: torch.nn.Module, max_norm: float) -> float:
    if max_norm <= 0:
        return 0.0
    params = [p for p in model.parameters() if p.requires_grad]
    return float(torch.nn.utils.clip_grad_norm_(params, max_norm))


@dataclass
class AlgoConfig:
    ppo_epochs: int = 1
    minibatch_size: int = 8
    clip_eps: float = 0.1
    kl_coef: float = 0.02
    max_grad_norm: float = 0.5
    adv_clip: float = 5.0

    # DAPO: asymmetric upper clip bound (lower bound uses clip_eps)
    clip_eps_high: float = 0.28
    # CISPO: dual-clip coefficient for negative-advantage sequences
    dual_clip_coef: float = 3.0
    # CISPO: entropy bonus weight added to the loss
    entropy_coef: float = 0.001

    seed: int = 0


class RLAlgorithm:
    name: str = "base"

    def __init__(self, cfg: AlgoConfig):
        self.cfg = cfg
        self._num_updates = 0

    def _next_update_seed(self) -> int:
        seed = int(self.cfg.seed + self._num_updates)
        self._num_updates += 1
        return seed

    def update(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        rollout: RolloutBatch,
        grad_accum_steps: int = 1,
        grad_clip_fn: Optional[GradClipFn] = None,
    ) -> Dict[str, float]:
        raise NotImplementedError
