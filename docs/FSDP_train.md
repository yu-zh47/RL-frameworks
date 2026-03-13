# Changelog – FSDP Training Backend & Full-Parameter Finetuning

## 2026-03-13 — Modular training backend + full-param support

### Context

The codebase already had two working rollout backends (`hf`, `vllm`) and two RL
algorithms (GRPO, REINFORCE).  Training was hardcoded to single-GPU LoRA via
HuggingFace + PEFT.

This changeset adds two new axes of flexibility:

1. **Training backend**: selectable between `hf` (single-GPU, existing default)
   and `fsdp` (PyTorch FSDP for multi-GPU / memory-constrained training).
2. **Finetune mode**: selectable between `lora` (existing default, PEFT adapter)
   and `full` (all parameters trainable, separate frozen reference model).

Both axes are orthogonal to the rollout backend choice (`--sampler hf|vllm`).

---

### FSDP gain assessment

For the current setup (Qwen2.5-Math-1.5B on a single H200 SXM with 141 GB),
FSDP provides **no measurable benefit** for LoRA training.  The model is ~3 GB
in bf16, LoRA adapters are ~50 MB, and optimizer states are ~100 MB.  There is
nothing to shard.

FSDP becomes relevant when:

| Scenario           | Memory estimate | Single H200? | FSDP useful? |
|--------------------|-----------------|--------------|--------------|
| LoRA, 1.5B         | ~5–12 GB        | Easily       | No           |
| Full-param, 1.5B   | ~25 GB          | Yes          | No           |
| LoRA, 7B           | ~20–30 GB       | Yes          | No           |
| Full-param, 7B     | ~84 GB          | Tight        | Yes          |
| Full-param, 70B    | ~700 GB         | No           | Essential    |

The FSDP backend is implemented as a forward investment for scaling to larger
models and multi-GPU setups.  The full-parameter finetune mode is the more
immediately useful addition.

---

### New files

```
llm_rl/training/__init__.py      Module init
llm_rl/training/base.py          TrainingBackend ABC
llm_rl/training/hf_backend.py    HFBackend (existing single-GPU behaviour)
llm_rl/training/fsdp_backend.py  FSDPBackend (FSDP wrapping, distributed init)
```

### Modified files

```
llm_rl/config.py            +training_backend, +finetune_mode, +fsdp_* fields
llm_rl/models/load.py       +load_full_param_policy_model_and_tokenizer()
                             +load_reference_model()
                             +finetune_mode field on LoadedPolicyModel
llm_rl/rl/base.py           +GradClipFn type, +grad_clip_fn parameter on update()
llm_rl/rl/grpo.py           Uses grad_clip_fn instead of hardcoded clip_grad_norm_
llm_rl/rl/reinforce.py      Same
llm_rl/rollout/hf_sampler.py    +ref_model parameter for full-param ref logprobs
llm_rl/rollout/vllm_sampler.py  +ref_model parameter (same pattern)
llm_rl/train.py              Backend dispatch, model loading dispatch,
                             reference model loading, new CLI args
```

The existing vLLM rollout framework (`vllm_sampler.py`) was already fully
functional.  It was only modified to accept an optional `ref_model` for
full-parameter mode — its generation, re-tokenisation, and LoRA sync logic are
unchanged.

---

### Architecture

```
                          ┌──────────────┐
                          │  TrainConfig  │
                          │ backend=hf|fsdp
                          │ finetune=lora|full
                          └──────┬───────┘
                                 │
           ┌─────────────────────┼─────────────────────┐
           ▼                     ▼                      ▼
    ┌─────────────┐    ┌──────────────────┐    ┌───────────────┐
    │ Model Load  │    │ Training Backend │    │    Sampler     │
    │ lora | full │    │   hf | fsdp      │    │   hf | vllm   │
    └──────┬──────┘    └────────┬─────────┘    └───────┬───────┘
           │                    │                      │
           │  prepare_model()   │                      │
           ├────────────────────►                      │
           │  create_optimizer()│                      │
           ├────────────────────►                      │
           │  grad_clip_fn()    │                      │
           ├────────────────────►                      │
           │                    │                      │
           │             ┌──────┴──────┐               │
           │             │  RL Algo    │               │
           │             │ grpo|reinf  │               │
           │             │ uses grad_  │               │
           │             │ clip_fn()   │               │
           │             └─────────────┘               │
           │                                           │
           │  ref_model (full-param only)              │
           ├───────────────────────────────────────────►
           │                                           │
```

**Key design principle**: rollout backend and training backend are orthogonal.
Any `{hf,vllm} × {hf,fsdp} × {lora,full}` combination is valid.

---

### TrainingBackend interface

```python
class TrainingBackend(ABC):
    def prepare_model_for_training(model) -> model
    def create_optimizer(model, lr, betas, weight_decay) -> optimizer
    def clip_grad_norm(model, max_norm) -> float
    def save_checkpoint(out_dir, step, model, tokenizer, optimizer, cfg)
    def is_main_process -> bool
    def cleanup()
    def grad_clip_fn(model) -> Callable[[model, max_norm], float]
```

`HFBackend`: pass-through, no wrapping.  Creates `AdamW` over
`requires_grad=True` params.  Uses `torch.nn.utils.clip_grad_norm_`.

`FSDPBackend`: calls `dist.init_process_group("nccl")`, wraps model with
`FullyShardedDataParallel` using auto-wrap policy over transformer decoder
layers, mixed precision bf16, optional CPU offload.  Uses
`model.clip_grad_norm_()` for correct all-reduce across shards.  Saves
full state dict from rank 0 only.

---

### Reference model handling

**LoRA mode**: reference logprobs are computed via `policy_model.disable_adapter()`,
which exposes the frozen base model for free.  No extra memory.

**Full-param mode**: `disable_adapter()` is not available, so a separate frozen
copy of the base model is loaded.  Memory cost:

- 1.5B model: ~3 GB extra (trivial on H200)
- 7B model: ~14 GB extra (still fits)
- For larger models, the reference model could be offloaded to CPU or computed
  periodically rather than held resident (not yet implemented).

Both the HF and vLLM samplers check: if `ref_model` is provided, use it;
otherwise fall back to `disable_adapter()` for LoRA; raise an error if neither
is available.

---

### Grad clipping abstraction

The RL algorithms (GRPO, REINFORCE) previously called
`clip_grad_norm_(trainable_params, max_norm)` directly.  This breaks under
FSDP because parameters are sharded — calling `torch.nn.utils.clip_grad_norm_`
would only see the local shard.

The fix: `update()` now accepts an optional `grad_clip_fn(model, max_norm) -> float`.
The training backend supplies the appropriate implementation:

- `HFBackend`: `torch.nn.utils.clip_grad_norm_` on `requires_grad` params
- `FSDPBackend`: `model.clip_grad_norm_()` (FSDP-aware all-reduce)

If no `grad_clip_fn` is passed, a default fallback uses the HF-style approach
for backward compatibility.

---

### CLI usage

**Existing LoRA training (unchanged, default)**:

```bash
python -m llm_rl.train --algo grpo --sampler hf
```

**Full-parameter finetuning on single GPU**:

```bash
python -m llm_rl.train --finetune_mode full --algo grpo --sampler hf
```

**FSDP + LoRA multi-GPU (requires torchrun)**:

```bash
torchrun --nproc_per_node=2 -m llm_rl.train \
    --training_backend fsdp --algo grpo --sampler vllm \
    --vllm_device cuda:2
```

**FSDP + full-param multi-GPU**:

```bash
torchrun --nproc_per_node=2 -m llm_rl.train \
    --training_backend fsdp --finetune_mode full \
    --fsdp_sharding_strategy full_shard --algo grpo
```

### New config fields

| Field | Default | Choices | Notes |
|-------|---------|---------|-------|
| `training_backend` | `"hf"` | `hf`, `fsdp` | |
| `finetune_mode` | `"lora"` | `lora`, `full` | |
| `fsdp_sharding_strategy` | `"full_shard"` | `full_shard`, `shard_grad_op`, `no_shard` | ZeRO-3, ZeRO-2, DDP-like |
| `fsdp_cpu_offload` | `False` | bool | Offload params to CPU between fwd/bwd |
| `fsdp_backward_prefetch` | `"backward_pre"` | `backward_pre`, `backward_post`, `none` | |
| `fsdp_forward_prefetch` | `False` | bool | |
| `fsdp_sync_module_states` | `True` | bool | Broadcast rank-0 state at init |
| `fsdp_use_orig_params` | `True` | bool | Required for PEFT/LoRA compatibility |

---

### What was NOT changed

- The vLLM rollout framework (`vllm_sampler.py`) was already fully implemented
  and smoke-tested.  Only its `__init__` signature gained the optional
  `ref_model` parameter; generation, re-tokenisation, LoRA weight sync, and
  completion mask logic are untouched.
- The HF sampler's generation and logprob computation paths are unchanged for
  LoRA mode.
- The RL algorithm loss/gradient logic (GRPO clipped surrogate, REINFORCE
  sequence-level objective, KL penalty) is identical — only the grad clipping
  callsite was abstracted.
- `RolloutOutput`, `RolloutBatch`, `Sampler` base class, task reward functions,
  and evaluation logic are all unchanged.
