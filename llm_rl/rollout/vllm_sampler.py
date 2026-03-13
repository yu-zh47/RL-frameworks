"""vLLM-accelerated rollout sampler.

WHY THIS EXISTS
===============
In RL for LLMs, each training step starts with "rollout": generating completions
from the current policy.  Autoregressive generation (one token at a time, each
depending on the last) is the main speed bottleneck.

vLLM is a dedicated inference engine that makes generation much faster than
HuggingFace's model.generate() through three key optimizations:

  - PagedAttention: manages the KV-cache like virtual memory pages so more
    sequences fit in GPU memory simultaneously.
  - Continuous batching: does not wait for the slowest sequence to finish
    before starting new work.
  - CUDA graphs: pre-compiles the generation kernel to reduce per-step
    Python/CUDA launch overhead.

HYBRID ARCHITECTURE
===================
The design here is the same one used in production RLHF systems (OpenRLHF,
veRL): split generation from training.

  Phase 1 -- Generate completions (vLLM, fast)
    vLLM loads the base model + current LoRA adapter and generates completion
    text + token IDs for each prompt. This is the expensive autoregressive
    part, and vLLM does it much faster than HF.

  Phase 2 -- Build exact full sequences
    We tokenize prompts once with the HF tokenizer, let vLLM generate
    completion token IDs, then concatenate prompt IDs + completion IDs
    directly. This preserves the exact sampled token trajectory without a
    decode/encode round-trip.

  Phase 3 -- Compute logprobs (HF model, same device as training)
    Run the HF policy model forward TWICE (not autoregressive, just one
    parallel pass each over the full sequence):

      3a. old_logprobs  (LoRA adapter ON)
          log p(token_t | token_<t) under the current trainable policy.
          Needed for the importance-sampling ratio in GRPO, or directly
          by REINFORCE.

      3b. ref_logprobs  (LoRA adapter OFF via disable_adapter())
          log p(token_t | token_<t) under the frozen base model.
          Needed for the KL divergence penalty.

    We recompute both on HF instead of reading logprobs from vLLM because:
      - vLLM cannot produce ref_logprobs (it only runs with the adapter).
      - We need policy/ref logprobs from the SAME engine for PPO-style ratios
        and KL bookkeeping.
      - A full-sequence forward pass is cheap compared to autoregressive
        generation (parallel over L positions vs L sequential steps).

  Phase 4 -- Pack into RolloutOutput
    Same output format as HFSampler so the rest of the training loop does not
    need to know which engine generated the completions.

LORA WEIGHT SYNC
================
The base model weights never change during LoRA training.  Only the small
adapter weights change each step.  Before each rollout we:
  1. Save the current adapter to a temp directory on disk.
  2. Create a vLLM LoRARequest pointing to that directory.
  3. vLLM loads the adapter on top of its own copy of the base model.
This is cheap because LoRA adapters are small (a few MB).

GPU LAYOUT
==========
vLLM and the training model can live on the same GPU or on different GPUs.
When you have multiple GPUs, putting vLLM on a separate GPU avoids memory
contention between generation KV-cache and training activations/gradients.
The `vllm_device` config controls this.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from llm_rl.models.load import tokenize_chat_prompts
from llm_rl.models.logprobs import build_completion_mask, compute_per_token_logprobs
from llm_rl.rollout.sampler_base import RolloutOutput, Sampler


@dataclass
class VLLMSamplerConfig:
    """Settings for the vLLM offline engine instance."""
    gpu_memory_utilization: float = 0.45  # fraction of GPU memory vLLM may use
    tensor_parallel_size: int = 1         # >1 shards the model across GPUs
    enforce_eager: bool = False           # True disables CUDA graphs (useful for debugging)
    max_model_len: Optional[int] = None   # max sequence length; None = auto from model config
    enable_lora: bool = True              # must be True so we can hot-swap LoRA adapters
    max_lora_rank: int = 64               # upper bound on LoRA rank the engine will accept
    dtype: str = "bfloat16"               # model precision inside vLLM
    swap_space: int = 0                   # CPU swap space in GB for KV-cache overflow


class VLLMSampler(Sampler):
    """Rollout sampler: vLLM generates completions, HF model computes logprobs."""

    def __init__(
        self,
        model_name: str,
        tokenizer,
        device: torch.device,
        vllm_cfg: VLLMSamplerConfig,
        vllm_device: Optional[str] = None,
        ref_model: Optional[torch.nn.Module] = None,
    ):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.device = device  # HF model device (for logprob computation)
        self.ref_model = ref_model
        self._lora_request_id = 0
        self._adapter_dir = tempfile.mkdtemp(prefix="vllm_lora_")

        # ── Initialise the vLLM engine on the requested GPU ──────────────
        # vLLM manages its own CUDA context.  We temporarily set
        # CUDA_VISIBLE_DEVICES so vLLM binds to the right physical GPU,
        # then restore the original value so the HF training model is not
        # affected.
        vllm_gpu = vllm_device or os.environ.get("VLLM_DEVICE", "cuda:0")
        gpu_idx = vllm_gpu.split(":")[1] if vllm_gpu.startswith("cuda:") else "0"

        old_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx

        self.llm = LLM(
            model=model_name,
            tokenizer=model_name,
            dtype=vllm_cfg.dtype,
            gpu_memory_utilization=vllm_cfg.gpu_memory_utilization,
            tensor_parallel_size=vllm_cfg.tensor_parallel_size,
            enforce_eager=vllm_cfg.enforce_eager,
            max_model_len=vllm_cfg.max_model_len,
            enable_lora=vllm_cfg.enable_lora,
            max_lora_rank=vllm_cfg.max_lora_rank,
            swap_space=vllm_cfg.swap_space,
            trust_remote_code=True,
        )

        if old_visible is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_visible
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    # ── LoRA adapter sync ────────────────────────────────────────────────

    def _sync_lora_weights(self, policy_model: torch.nn.Module) -> Optional[LoRARequest]:
        """Save current LoRA adapter to disk and create a vLLM LoRARequest.

        vLLM keeps its own copy of the base model weights.  To make it
        generate with the *current* policy, we export the latest LoRA
        adapter (a few MB) and hand vLLM a LoRARequest that points to it.
        Each call gets a new integer ID so vLLM treats it as a fresh adapter.
        """
        if not hasattr(policy_model, "save_pretrained"):
            return None
        policy_model.save_pretrained(self._adapter_dir)
        self._lora_request_id += 1
        return LoRARequest(
            lora_name=f"policy_step_{self._lora_request_id}",
            lora_int_id=self._lora_request_id,
            lora_path=self._adapter_dir,
        )

    # ── Prompt tokenization ──────────────────────────────────────────────

    def _build_prompt_tokens(
        self,
        prompt_messages: List[List[Dict[str, str]]],
        max_prompt_tokens: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, List[List[int]]]:
        """Tokenize prompts once and return both padded and unpadded forms.

        The padded tensors are used to rebuild full sequences in the same layout
        as `HFSampler`: left-padded prompt prefix, then completion tokens
        appended on the right. The unpadded ID lists are passed directly into
        vLLM so prompt truncation and chat-template handling stay identical
        across generation and HF logprob recomputation.
        """
        prompt_input_ids, prompt_attention = tokenize_chat_prompts(
            self.tokenizer,
            prompt_messages,
            add_generation_prompt=True,
            max_prompt_tokens=max_prompt_tokens,
            device=None,
        )
        prompt_token_ids: List[List[int]] = []
        for ids, mask in zip(prompt_input_ids, prompt_attention):
            n = int(mask.sum().item())
            prompt_token_ids.append(ids[-n:].tolist())
        return prompt_input_ids, prompt_attention, prompt_token_ids

    # ── Main rollout method ──────────────────────────────────────────────

    @torch.no_grad()
    def rollout(
        self,
        policy_model: torch.nn.Module,
        prompt_messages: List[List[Dict[str, str]]],
        task_names: List[str],
        task_metas: List[Dict[str, Any]],
        group_size: int,
        sampling,
        max_prompt_tokens: Optional[int] = None,
        output_to_cpu: bool = False,
    ) -> RolloutOutput:
        assert len(prompt_messages) == len(task_names) == len(task_metas)
        B = len(prompt_messages)

        # ════════════════════════════════════════════════════════════════
        # PHASE 1: Fast generation via vLLM
        # ════════════════════════════════════════════════════════════════
        # This is the expensive autoregressive part.  vLLM handles it
        # with PagedAttention, continuous batching, and CUDA graphs.
        #
        # Steps:
        #   1a. Export current LoRA adapter so vLLM sees the latest policy.
        #   1b. Tokenize prompts once with HF and pass prompt token IDs to
        #       vLLM directly.
        #   1c. Call vLLM's offline generate() with sampling parameters.
        #   1d. Collect the generated completion texts + token IDs.

        lora_request = self._sync_lora_weights(policy_model)

        prompt_input_ids, prompt_attention, prompt_token_ids = self._build_prompt_tokens(
            prompt_messages,
            max_prompt_tokens=max_prompt_tokens,
        )
        prompt_input_len = int(prompt_input_ids.shape[1])
        params = SamplingParams(
            n=group_size,
            temperature=max(float(sampling.temperature), 1e-5) if sampling.do_sample else 0.0,
            top_p=float(sampling.top_p),
            top_k=int(sampling.top_k) if sampling.top_k > 0 else -1,
            repetition_penalty=float(sampling.repetition_penalty),
            max_tokens=int(sampling.max_new_tokens),
            min_tokens=max(0, int(sampling.min_new_tokens)),
        )

        vllm_outputs = self.llm.generate(
            prompt_token_ids,
            sampling_params=params,
            lora_request=lora_request,
        )

        # prompt-major order: [prompt0_group0, prompt0_group1, ..., prompt1_group0, ...]
        completion_texts: List[str] = []
        completion_token_ids: List[List[int]] = []
        prompt_row_ids: List[int] = []
        for i, request_output in enumerate(vllm_outputs):
            for output in request_output.outputs:
                completion_texts.append(output.text)
                completion_token_ids.append(list(output.token_ids))
                prompt_row_ids.append(i)

        # ════════════════════════════════════════════════════════════════
        # PHASE 2: Build exact full sequences for the HF model
        # ════════════════════════════════════════════════════════════════
        pad_id = int(self.tokenizer.pad_token_id)
        max_completion_len = max((len(ids) for ids in completion_token_ids), default=0)
        N = len(completion_token_ids)
        total_seq_len = prompt_input_len + max_completion_len
        sequences = torch.full((N, total_seq_len), pad_id, dtype=torch.long)
        full_attention = torch.zeros((N, total_seq_len), dtype=torch.long)

        # IMPORTANT: the old implementation did:
        #   vLLM token IDs -> text -> HF chat template -> HF token IDs
        # Even with the SAME tokenizer, encode(decode(ids)) is not guaranteed
        # to return the same ids. BPE merges / splits can change at the
        # assistant-boundary and special-token context, which means RL would
        # score a DIFFERENT token trajectory from the one vLLM actually
        # sampled. Keep the original vLLM completion token IDs instead.
        for row_idx, (prompt_row_idx, comp_ids) in enumerate(zip(prompt_row_ids, completion_token_ids)):
            sequences[row_idx, :prompt_input_len] = prompt_input_ids[prompt_row_idx]
            full_attention[row_idx, :prompt_input_len] = prompt_attention[prompt_row_idx]
            if comp_ids:
                comp_len = len(comp_ids)
                sequences[row_idx, prompt_input_len : prompt_input_len + comp_len] = torch.tensor(
                    comp_ids, dtype=torch.long
                )
                full_attention[row_idx, prompt_input_len : prompt_input_len + comp_len] = 1

        sequences = sequences.to(self.device)
        full_attention = full_attention.to(self.device)

        # ════════════════════════════════════════════════════════════════
        # PHASE 3: Compute logprobs with the HF model
        # ════════════════════════════════════════════════════════════════
        # This is a normal forward pass (NOT autoregressive generation).
        # Given the full [prompt + completion] token sequence, compute:
        #
        #   old_logprobs: log p(token_t | token_<t) under the current
        #                 LoRA policy.  Needed for the importance ratio
        #                 in GRPO, or directly in REINFORCE.
        #
        #   ref_logprobs: log p(token_t | token_<t) under the frozen
        #                 base model (LoRA adapter disabled).  Needed
        #                 for the KL penalty term.
        #
        # This forward pass is fast because it processes the whole
        # sequence in one shot (not token-by-token).

        was_training = bool(policy_model.training)
        had_gc = bool(getattr(policy_model, "is_gradient_checkpointing", False))
        if had_gc and hasattr(policy_model, "gradient_checkpointing_disable"):
            policy_model.gradient_checkpointing_disable()
            policy_model.config.use_cache = True
        policy_model.eval()

        try:
            old_logp = compute_per_token_logprobs(
                policy_model, sequences, full_attention, enable_grad=False,
            )

            if self.ref_model is not None:
                ref_logp = compute_per_token_logprobs(
                    self.ref_model, sequences, full_attention, enable_grad=False,
                )
            elif hasattr(policy_model, "disable_adapter"):
                with policy_model.disable_adapter():
                    ref_logp = compute_per_token_logprobs(
                        policy_model, sequences, full_attention, enable_grad=False,
                    )
            else:
                raise RuntimeError(
                    "Full-param mode requires a separate reference model. "
                    "Pass ref_model to the sampler, or use LoRA mode."
                )

            completion_mask = build_completion_mask(
                input_ids=sequences,
                attention_mask=full_attention,
                prompt_input_len=prompt_input_len,
                pad_token_id=pad_id,
            )
        finally:
            if had_gc and hasattr(policy_model, "gradient_checkpointing_enable"):
                policy_model.gradient_checkpointing_enable()
                if hasattr(policy_model, "enable_input_require_grads"):
                    policy_model.enable_input_require_grads()
                policy_model.config.use_cache = False
            if was_training:
                policy_model.train()

        # ════════════════════════════════════════════════════════════════
        # PHASE 4: Pack into RolloutOutput
        # ════════════════════════════════════════════════════════════════
        # Same output dataclass as HFSampler so the rest of the training
        # loop (advantage computation, RL update, logging) does not need
        # to know which engine generated the completions.

        prompt_messages_rep: List[List[Dict[str, str]]] = []
        task_names_rep: List[str] = []
        task_metas_rep: List[Dict[str, Any]] = []
        for i in range(B):
            for _ in range(group_size):
                prompt_messages_rep.append(prompt_messages[i])
                task_names_rep.append(task_names[i])
                task_metas_rep.append(task_metas[i])

        if output_to_cpu:
            sequences = sequences.cpu()
            full_attention = full_attention.cpu()
            completion_mask = completion_mask.cpu()
            old_logp = old_logp.cpu()
            ref_logp = ref_logp.cpu()

        return RolloutOutput(
            prompt_messages=prompt_messages_rep,
            completion_texts=completion_texts,
            input_ids=sequences,
            attention_mask=full_attention,
            completion_mask=completion_mask,
            old_logprobs=old_logp,
            ref_logprobs=ref_logp,
            prompt_input_len=prompt_input_len,
            group_size=group_size,
            task_names=task_names_rep,
            task_metas=task_metas_rep,
        )
