"""FSDP training backend.

Launch with torchrun:
    torchrun --nproc_per_node=N -m llm_rl.train --training_backend fsdp ...

For single-GPU FSDP (useful for memory reduction via CPU offload):
    torchrun --nproc_per_node=1 -m llm_rl.train --training_backend fsdp ...
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Any, Optional, Set, Type

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullStateDictConfig,
    StateDictType,
)

from llm_rl.training.base import TrainingBackend


_SHARDING_MAP = {
    "full_shard": ShardingStrategy.FULL_SHARD,
    "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
    "no_shard": ShardingStrategy.NO_SHARD,
}

_PREFETCH_MAP = {
    "backward_pre": BackwardPrefetch.BACKWARD_PRE,
    "backward_post": BackwardPrefetch.BACKWARD_POST,
    "none": None,
}


def _find_transformer_layer_cls(model: torch.nn.Module) -> Set[Type]:
    """Heuristic: find the repeated decoder-layer class for auto-wrap policy.

    Works with HuggingFace models that store decoder layers in a ModuleList
    accessible via model.model.layers (Llama, Qwen, Mistral, etc.).
    """
    candidates: Set[Type] = set()
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is not None and hasattr(layers, "__len__") and len(layers) > 0:
        candidates.add(type(layers[0]))
    return candidates


class FSDPBackend(TrainingBackend):
    """PyTorch FSDP training backend for multi-GPU or memory-constrained training."""

    def __init__(self, cfg):
        self.cfg = cfg

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(self.local_rank)

    def prepare_model_for_training(self, model: torch.nn.Module) -> torch.nn.Module:
        cfg = self.cfg
        sharding = _SHARDING_MAP.get(cfg.fsdp_sharding_strategy)
        if sharding is None:
            raise ValueError(f"Unknown FSDP sharding strategy: {cfg.fsdp_sharding_strategy}")

        prefetch = _PREFETCH_MAP.get(cfg.fsdp_backward_prefetch)

        mp = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        cpu_offload = CPUOffload(offload_params=True) if cfg.fsdp_cpu_offload else None

        layer_classes = _find_transformer_layer_cls(model)
        wrap_policy = None
        if layer_classes:
            wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=layer_classes,
            )

        model = FSDP(
            model,
            sharding_strategy=sharding,
            mixed_precision=mp,
            cpu_offload=cpu_offload,
            auto_wrap_policy=wrap_policy,
            backward_prefetch=prefetch,
            forward_prefetch=cfg.fsdp_forward_prefetch,
            use_orig_params=cfg.fsdp_use_orig_params,
            sync_module_states=cfg.fsdp_sync_module_states,
            device_id=self.local_rank,
        )
        return model

    def create_optimizer(self, model: torch.nn.Module, lr: float,
                         betas: tuple, weight_decay: float) -> torch.optim.Optimizer:
        return torch.optim.AdamW(model.parameters(), lr=lr, betas=betas,
                                 weight_decay=weight_decay)

    def clip_grad_norm(self, model: torch.nn.Module, max_norm: float) -> float:
        if max_norm <= 0:
            return 0.0
        if isinstance(model, FSDP):
            return float(model.clip_grad_norm_(max_norm))
        params = [p for p in model.parameters() if p.requires_grad]
        return float(torch.nn.utils.clip_grad_norm_(params, max_norm))

    def save_checkpoint(self, out_dir: Path, step: int, model: torch.nn.Module,
                        tokenizer, optimizer: torch.optim.Optimizer, cfg) -> None:
        ckpt_dir = out_dir / "checkpoints" / f"step_{step:06d}"

        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            state_dict = model.state_dict()

        if self.is_main_process:
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            if cfg.finetune_mode == "lora" and hasattr(model, "save_pretrained"):
                adapter_dir = ckpt_dir / "adapter"
                adapter_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(adapter_dir, state_dict=state_dict)
                tokenizer.save_pretrained(adapter_dir)
            else:
                torch.save(state_dict, ckpt_dir / "model.pt")
                tokenizer.save_pretrained(ckpt_dir / "tokenizer")

            torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
            (ckpt_dir / "meta.json").write_text(json.dumps({
                "step": step,
                "algo": cfg.algo,
                "task": cfg.task,
                "model_name": cfg.model_name,
                "finetune_mode": cfg.finetune_mode,
                "training_backend": "fsdp",
                "world_size": self.world_size,
            }, indent=2))

        if dist.is_initialized():
            dist.barrier()

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def cleanup(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()
