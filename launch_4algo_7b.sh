#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
# Launch 4 parallel RL experiments on Qwen2.5-Math-7B-Instruct
#   GRPO  on GPU 0 (train) + GPU 1 (vllm)
#   DAPO  on GPU 2 (train) + GPU 3 (vllm)
#   GSPO  on GPU 4 (train) + GPU 5 (vllm)
#   CISPO on GPU 6 (train) + GPU 7 (vllm)
# ──────────────────────────────────────────────────────────────────────

MODEL="Qwen/Qwen2.5-Math-7B-Instruct"
TASK="math_hard"
STEPS=1000
SEED=42

CONDA_ENV="llm-vllm-rollout"
LOGDIR="runs/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$LOGDIR"

# Use hf-mirror since direct HuggingFace access is blocked
export HF_ENDPOINT="https://hf-mirror.com"

COMMON_ARGS=(
    --model_name "$MODEL"
    --task "$TASK"
    --steps "$STEPS"
    --seed "$SEED"

    # Rollout
    --sampler vllm
    --batch_size 8
    --group_size 6
    --max_new_tokens 512
    --max_prompt_tokens 512
    --temperature 0.8
    --top_p 0.95

    # LoRA on HF backend (single-GPU training)
    --training_backend hf
    --finetune_mode lora
    --lora_r 16
    --lora_alpha 32

    # Optimization
    --lr 2e-5
    --warmup_steps 50
    --grad_accum_steps 1
    --max_grad_norm 0.5
    --ppo_epochs 1

    # vLLM config (gets a dedicated GPU)
    --vllm_device cuda:1
    --vllm_gpu_memory_utilization 0.85
    --vllm_tensor_parallel_size 1

    # Memory
    --grad_checkpointing
    --attn_implementation sdpa
    --cuda_empty_cache_interval 20

    # Eval & logging
    --eval_interval 100
    --save_interval 250
    --math_hard_eval_n 512
    --eval_batch_size 32
    --sample_log_interval 10
    --sample_markdown_log_interval 5
    --wandb_project "llm-rl-7b-algo-comparison"
)

ALGOS=("grpo" "dapo" "gspo" "cispo")
GPU_PAIRS=("0,1" "2,3" "4,5" "6,7")
PIDS=()

echo "================================================================"
echo " Launching 4-algo comparison: ${ALGOS[*]}"
echo " Model:  $MODEL"
echo " Task:   $TASK (level 5)"
echo " Steps:  $STEPS"
echo " Time:   $TIMESTAMP"
echo "================================================================"

for i in "${!ALGOS[@]}"; do
    ALGO="${ALGOS[$i]}"
    GPUS="${GPU_PAIRS[$i]}"
    RUN_NAME="${ALGO}-7b-math-${TIMESTAMP}"
    OUT_DIR="runs/${ALGO}-7b-math-${TIMESTAMP}"
    LOGFILE="${LOGDIR}/${ALGO}-7b-${TIMESTAMP}.log"

    echo ""
    echo "  [$ALGO] GPUs=$GPUS  output=$OUT_DIR  log=$LOGFILE"

    CUDA_VISIBLE_DEVICES="$GPUS" \
    conda run --no-capture-output -n "$CONDA_ENV" \
        python -m llm_rl.train \
        "${COMMON_ARGS[@]}" \
        --algo "$ALGO" \
        --train_device cuda:0 \
        --output_dir "$OUT_DIR" \
        --wandb_name "$RUN_NAME" \
        > "$LOGFILE" 2>&1 &

    PIDS+=($!)
    echo "  [$ALGO] PID=${PIDS[-1]}"
done

echo ""
echo "================================================================"
echo " All 4 experiments launched. PIDs: ${PIDS[*]}"
echo " Logs:   $LOGDIR/*-7b-${TIMESTAMP}.log"
echo " Runs:   runs/*-7b-math-${TIMESTAMP}/"
echo " WandB:  llm-rl-7b-algo-comparison"
echo "================================================================"
echo ""
echo " Monitor with:"
echo "   tail -f $LOGDIR/grpo-7b-${TIMESTAMP}.log"
echo "   tail -f $LOGDIR/dapo-7b-${TIMESTAMP}.log"
echo "   tail -f $LOGDIR/gspo-7b-${TIMESTAMP}.log"
echo "   tail -f $LOGDIR/cispo-7b-${TIMESTAMP}.log"
echo ""
echo " Or watch all at once:"
echo "   tail -f $LOGDIR/*-7b-${TIMESTAMP}.log"
echo ""

wait_and_report() {
    local failed=0
    for i in "${!PIDS[@]}"; do
        local pid="${PIDS[$i]}"
        local algo="${ALGOS[$i]}"
        if wait "$pid"; then
            echo "[DONE]  $algo (PID $pid) finished successfully."
        else
            echo "[FAIL]  $algo (PID $pid) exited with error. Check log: $LOGDIR/${algo}-7b-${TIMESTAMP}.log"
            failed=$((failed + 1))
        fi
    done
    echo ""
    if [ "$failed" -eq 0 ]; then
        echo "All 4 experiments completed successfully!"
    else
        echo "$failed / 4 experiments failed."
    fi
}

wait_and_report
