from __future__ import annotations

import math
from typing import Dict

import torch

from llm_rl.models.logprobs import (
    approx_kl_from_logprobs,
    compute_per_token_logprobs,
    masked_mean,
)
from llm_rl.rl.base import GradClipFn, RLAlgorithm, _default_grad_clip
from llm_rl.rollout.rollout_buffer import RolloutBatch, iter_minibatches


class CISPO(RLAlgorithm):
    """CISPO: Clipped IS-weight Policy Optimization (arXiv 2503.12018).

    Instead of PPO's pessimistic min over clipped/unclipped surrogate,
    CISPO clips the importance-sampling weight directly and uses it to
    reweight the standard policy gradient objective:

        J = (1 / sum|o_i|) sum_i sum_t  sg(r̂_{i,t}) * A_i * log pi_θ(o_{i,t} | q, o_{i,<t})

    where  r̂_{i,t} = clip(r_{i,t}, 1 - eps_low, 1 + eps_high)

    The paper sets eps_low to a large value (effectively no lower bound)
    and eps_high = clip_eps_high.  Gradients flow only through log pi_θ
    because the clipped ratio is stop-gradiented.

    An explicit entropy bonus (entropy_coef) is added per the paper's
    ablations.
    """

    name = "cispo"

    def update(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        rollout: RolloutBatch,
        grad_accum_steps: int = 1,
        grad_clip_fn: GradClipFn | None = None,
    ) -> Dict[str, float]:
        cfg = self.cfg
        _clip = grad_clip_fn or _default_grad_clip
        model.train()
        model.config.use_cache = False

        ent_coef = cfg.entropy_coef
        eps_high = cfg.clip_eps_high
        eps_low = cfg.clip_eps

        total_loss = 0.0
        total_kl = 0.0
        total_clipfrac = 0.0
        total_entropy = 0.0
        n_mb = 0
        accum = 0
        skipped_empty = 0
        skipped_nonfinite = 0
        total_grad_norm = 0.0
        opt_steps = 0

        optimizer.zero_grad(set_to_none=True)
        rng = torch.Generator(device="cpu")
        rng.manual_seed(self._next_update_seed())

        for _ in range(cfg.ppo_epochs):
            for mb in iter_minibatches(
                rollout,
                cfg.minibatch_size,
                shuffle=True,
                generator=rng,
                device=next(model.parameters()).device,
            ):
                adv = mb.advantages.clamp(-cfg.adv_clip, cfg.adv_clip).detach()
                mask = mb.completion_mask
                if float(mask.sum().item()) <= 0.0:
                    skipped_empty += 1
                    continue

                new_logp = compute_per_token_logprobs(model, mb.input_ids, mb.attention_mask)

                log_ratio = torch.clamp(new_logp - mb.old_logprobs, min=-20.0, max=20.0)
                ratio = torch.exp(log_ratio)

                # Clip the IS weight directly (Eq. 5). sg() = .detach().
                clipped_ratio = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high).detach()

                adv_t = adv.unsqueeze(1)                                      # [B, 1]

                # Eq. 4: sg(r̂) * A * log pi_θ  — token-level, averaged across
                # all completion tokens in the minibatch (not per-sequence first).
                per_token_obj = clipped_ratio * adv_t * new_logp              # [B, T]
                total_tokens = mask.sum()
                pg_loss = -(per_token_obj * mask).sum() / (total_tokens + 1e-8)

                kl = approx_kl_from_logprobs(new_logp, mb.ref_logprobs, mask)
                entropy = -masked_mean(new_logp, mask)

                is_clipped = ((ratio < 1.0 - eps_low) | (ratio > 1.0 + eps_high)).float()
                clipfrac = masked_mean(is_clipped, mask)

                loss = (pg_loss + cfg.kl_coef * kl - ent_coef * entropy) / max(1, grad_accum_steps)
                if not torch.isfinite(loss):
                    skipped_nonfinite += 1
                    optimizer.zero_grad(set_to_none=True)
                    accum = 0
                    continue
                loss.backward()

                accum += 1

                if (accum % max(1, grad_accum_steps)) == 0:
                    gnorm = _clip(model, cfg.max_grad_norm)
                    if not math.isfinite(gnorm):
                        skipped_nonfinite += 1
                        optimizer.zero_grad(set_to_none=True)
                        accum = 0
                        continue
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    total_grad_norm += float(gnorm)
                    opt_steps += 1

                total_loss += float((loss.detach() * max(1, grad_accum_steps)).item())
                total_kl += float(kl.detach().item())
                total_entropy += float(entropy.detach().item())
                total_clipfrac += float(clipfrac.detach().item())
                n_mb += 1

        if accum > 0 and (accum % max(1, grad_accum_steps)) != 0:
            gnorm = _clip(model, cfg.max_grad_norm)
            if math.isfinite(gnorm):
                optimizer.step()
                total_grad_norm += float(gnorm)
                opt_steps += 1
            else:
                skipped_nonfinite += 1
            optimizer.zero_grad(set_to_none=True)

        denom = max(1, n_mb)
        return {
            "train/policy_loss_with_kl_and_entropy_mean_over_minibatches": total_loss / denom,
            "train/approximate_kl_divergence_policy_vs_reference_mean_over_minibatches": total_kl / denom,
            "train/policy_token_entropy_mean_over_minibatches": total_entropy / denom,
            "train/fraction_of_completion_tokens_where_is_weight_was_clipped_mean_over_minibatches": total_clipfrac / denom,
            "train/count_minibatches_skipped_because_completion_mask_had_no_tokens": float(skipped_empty),
            "train/count_update_attempts_skipped_due_to_nonfinite_loss_or_gradients": float(skipped_nonfinite),
            "train/gradient_global_norm_after_clipping_mean_over_optimizer_steps": total_grad_norm / max(1, opt_steps),
            "train/count_optimizer_steps_per_training_iteration": float(opt_steps),
        }
