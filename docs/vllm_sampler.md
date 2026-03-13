# Changelog – RL/LLM vLLM Sampler

## 2026-03-13 — Bug fixes to make `--sampler vllm` runnable

### Bug 1 (Critical): Incorrect completion mask in `vllm_sampler.py`

**File:** `llm_rl/rollout/vllm_sampler.py`

**Problem:**
The original Phase 2 re-tokenisation computed a single scalar `prompt_input_len`
by tokenising the B original prompts in a separate batch, then passed that scalar
to `build_completion_mask()` for the B×G full-sequence batch.

In the HF sampler this is safe because `model.generate()` extends the prompt
tensor in-place—every row shares the same padded prompt prefix and the same
`prompt_input_len`.

In the vLLM sampler the full sequences are re-tokenised from text.  Each of the
B×G sequences gets its own left-padding offset (driven by varying completion
lengths), so the prompt–completion boundary is at a *different* absolute position
in each row.  A single scalar threshold mis-labels prompt tokens as completion
tokens (or vice-versa), corrupting logprob and KL computation.

**Fix:**
Replaced the scalar `prompt_input_len` approach with per-sequence prompt
boundaries:

1. Tokenise each of the B prompts individually (via `apply_chat_template`) to
   obtain the raw (unpadded) prompt token count.  Replicate for group_size to get
   a length-N list `prompt_lens_rep`.
2. After tokenising the full sequences into `[N, L]` tensors, compute each row's
   left-padding count from `attention_mask`.
3. Build the completion mask with vectorised per-row thresholds:

```python
left_pads = (L - full_attention.sum(dim=1)).long()
prompt_lens_t = torch.tensor(prompt_lens_rep, dtype=torch.long, device=...)
comp_starts = left_pads + prompt_lens_t
positions = torch.arange(L - 1, device=...).unsqueeze(0)
thresholds = (comp_starts - 1).unsqueeze(1)
completion_mask = (positions >= thresholds).float() * full_attention[:, 1:].float()
```

This also removes the now-unused import of `tokenize_chat_prompts` and
`build_completion_mask` from the vLLM sampler.

---

### Bug 2: `torch.Generator` on CUDA in `grpo.py` and `reinforce.py`

**Files:** `llm_rl/rl/grpo.py`, `llm_rl/rl/reinforce.py`

**Problem:**
Both RL update methods created a `torch.Generator` on the rollout tensor's
device:

```python
rng = torch.Generator(device=rollout.input_ids.device)
```

`torch.randperm` (called inside `iter_minibatches`) only supports CPU generators
and raises `RuntimeError: Expected a 'cpu' device type for generator but found
'cuda'`.

**Fix:**
Force the generator onto CPU in both files:

```python
rng = torch.Generator(device="cpu")
```

---

### Environment note: HuggingFace mirror

The machine cannot reach `huggingface.co` directly.  Set the mirror before
running:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

### Verified

Smoke-tested with:

```bash
conda run -n llm-vllm-rollout env HF_ENDPOINT=https://hf-mirror.com \
  python -m llm_rl.train \
    --task format_copy --algo grpo --sampler vllm \
    --steps 2 --batch_size 2 --group_size 2 \
    --train_device cuda:0 --vllm_device cuda:1 \
    --vllm_gpu_memory_utilization 0.45 --vllm_enforce_eager \
    --no-wandb_enabled
```

Completed 2 training steps end-to-end (vLLM init → LoRA sync → generation →
re-tokenisation → logprob forward → GRPO update → baseline + final eval) with
exit code 0.
