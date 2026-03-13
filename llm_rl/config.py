from __future__ import annotations
from dataclasses import dataclass

@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    output_dir: str = "runs/default"
    task: str = "format_copy"  # format_copy | math_hard

    seed: int = 0
    steps: int = 1200

    batch_size: int = 8
    group_size: int = 6

    # Sampling
    min_new_tokens: int = 1
    max_new_tokens: int = 512
    max_prompt_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 0
    repetition_penalty: float = 1.0

    # LoRA optimization (base model stays frozen)
    lr: float = 3e-5
    weight_decay: float = 0.0
    betas1: float = 0.9
    betas2: float = 0.95
    warmup_steps: int = 100
    grad_accum_steps: int = 1

    # RL update hyperparameters
    algo: str = "grpo"      # reinforce | grpo | dapo | cispo | gspo
    ppo_epochs: int = 1  # used by GRPO/DAPO/CISPO; REINFORCE requires 1
    minibatch_size: int = 8
    clip_eps: float = 0.1
    kl_coef: float = 0.05
    max_grad_norm: float = 0.5
    adv_clip: float = 5.0
    normalize_advantages: bool = False

    # DAPO: asymmetric upper clip (lower bound reuses clip_eps)
    clip_eps_high: float = 0.28
    # CISPO: dual-clip coefficient for negative advantages
    dual_clip_coef: float = 3.0
    # CISPO: entropy bonus weight
    entropy_coef: float = 0.001

    # LoRA adapter config
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    lora_bias: str = "none"

    # Memory/speed
    grad_checkpointing: bool = True
    rollout_on_cpu: bool = False
    cuda_empty_cache_interval: int = 0
    attn_implementation: str = "sdpa"  # flash_attention_2 | sdpa | eager

    # Inference engine selection
    sampler: str = "hf"  # hf | vllm
    vllm_gpu_memory_utilization: float = 0.45
    vllm_tensor_parallel_size: int = 1
    vllm_device: str = ""       # e.g. "cuda:1"; empty = same as training device
    vllm_enforce_eager: bool = False
    vllm_max_model_len: int = 0  # 0 = auto from model config

    # Training backend
    training_backend: str = "hf"  # hf | fsdp
    finetune_mode: str = "lora"   # lora | full

    # FSDP settings (only used when training_backend="fsdp")
    fsdp_sharding_strategy: str = "full_shard"  # full_shard | shard_grad_op | no_shard
    fsdp_cpu_offload: bool = False
    fsdp_backward_prefetch: str = "backward_pre"  # backward_pre | backward_post | none
    fsdp_forward_prefetch: bool = False
    fsdp_sync_module_states: bool = True
    fsdp_use_orig_params: bool = True

    # Multi-GPU / device placement
    train_device: str = ""  # e.g. "cuda:0"; empty = auto

    # Logging / checkpointing
    wandb_project: str = "llm-rl"
    wandb_name: str = "run"
    wandb_enabled: bool = True
    sample_log_interval: int = 10
    sample_markdown_log_interval: int = 1
    sample_log_n: int = 3
    sample_log_max_chars: int = 3000

    eval_interval: int = 100
    save_interval: int = 100

    # Eval limits
    format_copy_eval_n: int = 64
    math_hard_eval_n: int = 512
    eval_batch_size: int = 32
