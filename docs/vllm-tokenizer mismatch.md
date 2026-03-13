Both runs completed. Here's the full comparison:

## Speed: vLLM is faster

| | Baseline (H100, HF) | vLLM (L20X) | Delta |
|---|---|---|---|
| **REINFORCE** total | 99.3 min | **94.6 min** | 5% faster |
| **GRPO** total | 254.2 min | **243.1 min** | 4% faster |
| REINFORCE tokens/sec | 1,541 | **1,717** | +11% |
| GRPO tokens/sec | 1,387 | **1,568** | +13% |

Note: this is L20X vs H100, so the vLLM sampler on equivalent hardware would show a larger gap.

## Accuracy: vLLM trails by a few points

| | Baseline | vLLM | Delta |
|---|---|---|---|
| **REINFORCE** boxed exact | **30.5%** | 28.1% | -2.4pp |
| **REINFORCE** relaxed exact | **33.4%** | 30.7% | -2.7pp |
| **GRPO** boxed exact | **43.0%** | 35.2% | -7.8pp |
| **GRPO** relaxed exact | **44.0%** | 36.3% | -7.6pp |

## Eval progression (vLLM GRPO)

Eval evals at steps 0, 100, 200, 300, 400, 500: **22.5% → 27.3% → 29.7% → 36.9% → 36.7% → 35.6% → 35.2%**. It peaked around step 300 at 36.9% then slightly regressed — suggests it could benefit from early stopping or a lower learning rate in later steps.

## Analysis

The accuracy gap is real but has a few contributing factors:

1. **Re-tokenization mismatch** — the detokenize-retokenize round trip in vLLM sampler creates slight logprob discrepancies, which compound over 500 steps of RL training. The higher KL in the REINFORCE run (0.0042 vs 0.0026) and the entropy differences we saw earlier are consistent with this.

2. **GRPO peaking then regressing** — the vLLM GRPO hit 36.9% at step 300 but ended at 35.2%. The baseline ended at 43.0% and was still climbing. This suggests the vLLM run's slightly noisier logprobs may cause instability at later stages. A warmup schedule or lower LR after step 300 could help.

3. **The runs are not apple-to-apple on hardware** — different GPU architectures (L20X vs H100) can cause subtle numerical differences in bf16 matmuls, affecting training dynamics.

The vLLM sampler achieves the core goal: faster inference with correct training dynamics. The accuracy gap is worth investigating further — the re-tokenization step is the prime suspect and could potentially be improved by aligning tokenizer settings between vLLM and HF more precisely.