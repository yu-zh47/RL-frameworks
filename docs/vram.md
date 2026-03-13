**GPU 0**: 86.4 GiB — training process
**GPU 1**: 64.6 GiB — vLLM engine
**Total**: ~151 GiB for a "1.5B model"

---

## GPU 1 (vLLM) — 64.6 GiB

This one is simple. Your config has `vllm_gpu_memory_utilization: 0.45`, and vLLM **pre-allocates** that fraction of the entire GPU at startup for its PagedAttention KV cache pool:

> 0.45 × 140.4 GiB = **~63 GiB**

vLLM grabs this regardless of model size. The actual model weights are only ~3 GB; the other ~60 GB is an empty KV cache block table sitting there mostly unused. You can safely lower this (e.g. `0.15` for a 1.5B model).

---

## GPU 0 (Training) — 86.4 GiB

This is the real question. Your math assumed optimizer states dominate, but they don't — the **logits tensor** does.

### Correction 1: Optimizer states are tiny with LoRA

You wrote "3 × 3GB = 9GB for optimizer states", but Adam only stores states for **trainable** parameters. With LoRA, that's ~18.5M params (not 1.5B), so:

| Component | Size |
|-----------|------|
| Base model (frozen, bf16) | ~3 GB |
| LoRA weights (fp32) | ~0.07 GB |
| Adam momentum + variance | ~0.15 GB |

That's **~3.2 GB**. Not 9 GB.

### Correction 2: The real memory hog is the logits tensor

Look at what happens during rollout logprob computation in `vllm_sampler.py`. After vLLM generates text on GPU 1, your HF model on GPU 0 runs **two full forward passes** (old_logprobs with adapter, ref_logprobs without adapter) over **all 48 sequences at once** (batch_size=8 × group_size=6):

```45:52:llm_rl/models/logprobs.py
    with torch.set_grad_enabled(enable_grad):
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        B, L, V = out.logits.shape
        logits_flat = out.logits[:, :-1, :].reshape(B * (L - 1), V)
        targets_flat = input_ids[:, 1:].reshape(B * (L - 1))
        log_probs = -F.cross_entropy(logits_flat, targets_flat, reduction='none').reshape(B, L - 1)
        return log_probs
```

Qwen2.5 has a vocab size of **151,936**. The logits tensor shape is `[48, 1024, 151936]`:

> 48 × 1024 × 151,936 × 2 bytes = **~14.2 GB** (bf16)

Then `.reshape(B * (L - 1), V)` on a non-contiguous slice (`[:, :-1, :]`) forces a **copy**, so you briefly have **~28 GB** in logits alone. Then `F.cross_entropy` may internally upcast to fp32 for numerical stability, potentially touching **~28 GB more** transiently.

### Full breakdown for GPU 0

| Component | Approx Size | Notes |
|-----------|-------------|-------|
| CUDA context | ~1 GB | Fixed overhead |
| Base model (bf16) | ~3 GB | Frozen 1.5B params |
| LoRA + Adam states | ~0.2 GB | Only ~18.5M trainable params |
| **Rollout logits (peak)** | **~14–28 GB** | `[48, 1024, 151936]` bf16 + reshape copy |
| Intermediate activations | ~5–10 GB | 28 layers × 48 sequences, hidden states + FFN |
| Training activations | ~3–5 GB | GRPO backward with grad checkpointing (minibatch=8) |
| PyTorch allocator caching | ~20–30 GB | Cached freed blocks not returned to CUDA |

**Total: ~50–75 GB active + allocator overhead → ~86 GiB observed.**

The PyTorch CUDA caching allocator never returns freed blocks back to the CUDA runtime, so `nvitop` reports the **high-water mark** of all allocations across both rollout and training phases combined.

---

## What to do about it

1. **Lower `vllm_gpu_memory_utilization`** to `0.15`–`0.20`. You don't need 63 GB of KV cache for a 1.5B model. This saves ~40 GB on GPU 1.

2. **Chunk the logprob forward pass.** The rollout computes logprobs for all 48 sequences at once. Splitting into mini-batches of 8–16 for the logprob forward pass would cut the peak logits tensor from 14 GB to ~2–4 GB. The code in `vllm_sampler.py` (lines 333–345) and `hf_sampler.py` (lines 103–129) could loop over chunks instead of forwarding the full batch.

3. **Use `torch.cuda.empty_cache()`** between rollout and training phases (set `cuda_empty_cache_interval: 1` in your config) to release the cached allocator blocks.