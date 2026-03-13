from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import torch


class TrainingBackend(ABC):
    """Abstract interface for training backends (HF single-GPU, FSDP, etc.).

    The RL algorithms (GRPO, REINFORCE) handle forward/backward/step themselves.
    The backend is responsible for model wrapping, optimizer creation, gradient
    clipping, and checkpoint I/O -- the operations that differ between
    single-GPU and distributed setups.
    """

    @abstractmethod
    def prepare_model_for_training(self, model: torch.nn.Module) -> torch.nn.Module:
        """Wrap or transform the model for this backend (e.g. FSDP wrapping)."""
        ...

    @abstractmethod
    def create_optimizer(self, model: torch.nn.Module, lr: float,
                         betas: tuple, weight_decay: float) -> torch.optim.Optimizer:
        ...

    @abstractmethod
    def clip_grad_norm(self, model: torch.nn.Module, max_norm: float) -> float:
        """Clip gradients and return the global (pre-clip) grad norm."""
        ...

    @abstractmethod
    def save_checkpoint(self, out_dir: Path, step: int, model: torch.nn.Module,
                        tokenizer, optimizer: torch.optim.Optimizer, cfg) -> None:
        ...

    @property
    def is_main_process(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def grad_clip_fn(self, model: torch.nn.Module) -> Callable[[torch.nn.Module, float], float]:
        """Return a (model, max_norm) -> grad_norm callable for RL algorithms."""
        backend = self
        def _clip(m: torch.nn.Module, max_norm: float) -> float:
            return backend.clip_grad_norm(m, max_norm)
        return _clip
