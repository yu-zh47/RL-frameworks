#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
# Launch 4 parallel RL experiments on Qwen2.5-Math-7B-Instruct
#
# Training GPUs: 0, 1, 2, 3  (one per algo)
# vLLM GPUs:     4, 5, 6, 7  (one per algo, fully separated)
#
#   GRPO  → train cuda:0, vllm cuda:4
#   DAPO  → train cuda:1, vllm cuda:5
#   GSPO  → train cuda:2, vllm cuda:6
#   CISPO → train cuda:3, vllm cuda:7
# ──────────────────────────────────────────────────────────────────────

MODEL="Qwen/Qwen2.5-Math-7B-Instruct"
TASK="math_hard"
STEPS=1000
SEED=42

CONDA_ENV="llm-vllm-rollout"
LOGDIR="runs/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$LOGDIR"

export HF_ENDPOINT="https://hf-mirror.com"
export PYTHONUNBUFFERED=1

COMMON_ARGS=(
    --model_name "$MODEL"
    --task "$TASK"
    --steps "$STEPS"
    --seed "$SEED"

    --sampler vllm
    --batch_size 8
    --group_size 6
    --max_new_tokens 1024
    --max_prompt_tokens 512
    --temperature 0.8
    --top_p 0.95
    --minibatch_size 4

    --training_backend hf
    --finetune_mode lora
    --lora_r 64
    --lora_alpha 128

    --lr 2e-5
    --warmup_steps 50
    --grad_accum_steps 2
    --max_grad_norm 0.5
    --ppo_epochs 1

    --vllm_gpu_memory_utilization 0.90
    --vllm_tensor_parallel_size 1

    --grad_checkpointing
    --attn_implementation flash_attention_2
    --cuda_empty_cache_interval 20

    --eval_interval 100
    --save_interval 250
    --math_hard_eval_n 512
    --eval_batch_size 32
    --sample_log_interval 10
    --sample_markdown_log_interval 5
    --wandb_project "llm-rl-7b-algo-comparison"
)

ALGOS=(      "grpo"   "dapo"   "gspo"   "cispo" )
TRAIN_GPUS=( "cuda:0" "cuda:1" "cuda:2" "cuda:3" )
VLLM_GPUS=(  "cuda:4" "cuda:5" "cuda:6" "cuda:7" )
PIDS=()

echo "================================================================"
echo " Launching 4-algo comparison: ${ALGOS[*]}"
echo " Model:  $MODEL"
echo " Task:   $TASK (level 5)"
echo " Steps:  $STEPS"
echo " Config: batch=8 group=6 grad_accum=2 mb=4 max_tokens=1024 lora_r=64 FA2"
echo " Train GPUs: ${TRAIN_GPUS[*]}"
echo " vLLM GPUs:  ${VLLM_GPUS[*]}"
echo " Time:   $TIMESTAMP"
echo "================================================================"

for i in "${!ALGOS[@]}"; do
    ALGO="${ALGOS[$i]}"
    TDEV="${TRAIN_GPUS[$i]}"
    VDEV="${VLLM_GPUS[$i]}"
    RUN_NAME="${ALGO}-7b-math-${TIMESTAMP}"
    OUT_DIR="runs/${ALGO}-7b-math-${TIMESTAMP}"
    LOGFILE="${LOGDIR}/${ALGO}-7b-${TIMESTAMP}.log"

    echo ""
    echo "  [$ALGO] train=$TDEV  vllm=$VDEV  output=$OUT_DIR"

    conda run --no-capture-output -n "$CONDA_ENV" \
        python -u -m llm_rl.train \
        "${COMMON_ARGS[@]}" \
        --algo "$ALGO" \
        --train_device "$TDEV" \
        --vllm_device "$VDEV" \
        --output_dir "$OUT_DIR" \
        --wandb_name "$RUN_NAME" \
        > "$LOGFILE" 2>&1 &

    PIDS+=($!)
    echo "  [$ALGO] PID=${PIDS[-1]}"
done

echo ""
echo "================================================================"
echo " All 4 experiments launched. PIDs: ${PIDS[*]}"
echo ""
echo " Logs: $LOGDIR/*-7b-${TIMESTAMP}.log"
echo " Runs: runs/*-7b-math-${TIMESTAMP}/"
echo " WandB: llm-rl-7b-algo-comparison"
echo ""
echo " Monitor:"
echo "   tail -f $LOGDIR/grpo-7b-${TIMESTAMP}.log"
echo "   tail -f $LOGDIR/dapo-7b-${TIMESTAMP}.log"
echo "   tail -f $LOGDIR/gspo-7b-${TIMESTAMP}.log"
echo "   tail -f $LOGDIR/cispo-7b-${TIMESTAMP}.log"
echo "================================================================"

wait_and_report() {
    local failed=0
    for i in "${!PIDS[@]}"; do
        local pid="${PIDS[$i]}"
        local algo="${ALGOS[$i]}"
        if wait "$pid"; then
            echo "[DONE]  $algo (PID $pid) finished successfully."
        else
            echo "[FAIL]  $algo (PID $pid) exited with error. Check: $LOGDIR/${algo}-7b-${TIMESTAMP}.log"
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
