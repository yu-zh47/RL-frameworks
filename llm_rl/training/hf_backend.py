from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch

from llm_rl.training.base import TrainingBackend


class HFBackend(TrainingBackend):
    """Single-GPU HuggingFace training -- the existing default behaviour."""

    def prepare_model_for_training(self, model: torch.nn.Module) -> torch.nn.Module:
        return model

    def create_optimizer(self, model: torch.nn.Module, lr: float,
                         betas: tuple, weight_decay: float) -> torch.optim.Optimizer:
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("No trainable parameters found.")
        return torch.optim.AdamW(trainable, lr=lr, betas=betas,
                                 weight_decay=weight_decay)

    def clip_grad_norm(self, model: torch.nn.Module, max_norm: float) -> float:
        if max_norm <= 0:
            return 0.0
        params = [p for p in model.parameters() if p.requires_grad]
        return float(torch.nn.utils.clip_grad_norm_(params, max_norm))

    def save_checkpoint(self, out_dir: Path, step: int, model: torch.nn.Module,
                        tokenizer, optimizer: torch.optim.Optimizer, cfg) -> None:
        ckpt_dir = out_dir / "checkpoints" / f"step_{step:06d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(model, "save_pretrained"):
            adapter_dir = ckpt_dir / "adapter"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(adapter_dir)
            tokenizer.save_pretrained(adapter_dir)
            _write_adapter_manifest(ckpt_dir, adapter_dir, step)
        else:
            weights_path = ckpt_dir / "model.pt"
            torch.save(model.state_dict(), weights_path)
            tokenizer.save_pretrained(ckpt_dir / "tokenizer")

        torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
        (ckpt_dir / "meta.json").write_text(json.dumps({
            "step": step,
            "algo": cfg.algo,
            "task": cfg.task,
            "model_name": cfg.model_name,
            "finetune_mode": cfg.finetune_mode,
            "training_backend": cfg.training_backend,
        }, indent=2))


def _write_adapter_manifest(ckpt_dir: Path, adapter_dir: Path, step: int) -> None:
    files = []
    total_bytes = 0
    for path in sorted(adapter_dir.rglob("*")):
        if not path.is_file():
            continue
        size = int(path.stat().st_size)
        total_bytes += size
        files.append({"path": str(path.relative_to(ckpt_dir)), "size_bytes": size})
    (ckpt_dir / "adapter_manifest.json").write_text(json.dumps({
        "step": step,
        "adapter_dir_exists": adapter_dir.is_dir(),
        "optimizer_state_exists": (ckpt_dir / "optimizer.pt").is_file(),
        "adapter_file_count": len(files),
        "adapter_total_bytes": total_bytes,
        "files": files,
    }, indent=2))
