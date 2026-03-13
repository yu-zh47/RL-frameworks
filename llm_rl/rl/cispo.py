from __future__ import annotations

import math
from typing import Dict

import torch

from llm_rl.models.logprobs import (
    approx_kl_from_logprobs,
    compute_per_token_logprobs,
    masked_mean,
    masked_mean_per_row,
)
from llm_rl.rl.base import GradClipFn, RLAlgorithm, _default_grad_clip
from llm_rl.rollout.rollout_buffer import RolloutBatch, iter_minibatches


class CISPO(RLAlgorithm):
    """CISPO: Clipped Importance Sampling Policy Optimization.

    Extends GRPO with dual clipping and an explicit entropy bonus.

    Standard PPO-style clipping prevents the ratio from moving *too far*
    from 1 in either direction, but when the advantage is very negative
    the clipped objective can still produce an arbitrarily large negative
    contribution.  Dual clipping adds a *floor* of  ``c * A_i``  for
    negative-advantage sequences, where ``c > 1`` (``dual_clip_coef``).
    This bounds the influence of worst-case completions and stabilises
    training on noisy reward signals.

    An explicit entropy bonus (controlled by ``entropy_coef``) is added
    to the loss to encourage continued exploration, complementing the
    clipping-based trust region.

    Formally, for each token position t in sequence i:

        ppo_obj_{i,t}  = min(r_t * A_i,  clip(r_t, 1-eps, 1+eps) * A_i)

        obj_{i,t}  = ppo_obj_{i,t}                      if A_i >= 0
                     max(ppo_obj_{i,t},  c * A_i)        if A_i < 0

    where  r_t = pi_theta(a_t | s_<t) / pi_old(a_t | s_<t).

    Loss = -mean_i[ mean_t obj_{i,t} ]  +  kl_coef * KL  -  entropy_coef * H
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

        dual_c = cfg.dual_clip_coef
        ent_coef = cfg.entropy_coef

        total_loss = 0.0
        total_kl = 0.0
        total_clipfrac = 0.0
        total_dual_clipfrac = 0.0
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

                adv_t = adv.unsqueeze(1)                                      # [B, 1]

                unclipped = ratio * adv_t
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_t
                ppo_obj = torch.minimum(unclipped, clipped)

                # Dual clip: impose floor c * A_i when A_i < 0.
                neg_mask = (adv_t < 0).float()                                # [B, 1]
                dual_floor = dual_c * adv_t                                   # [B, 1]
                per_token_obj = (
                    ppo_obj * (1.0 - neg_mask)
                    + torch.maximum(ppo_obj, dual_floor) * neg_mask
                )

                seq_obj = masked_mean_per_row(per_token_obj * mask, mask)      # [B]
                pg_loss = -torch.mean(seq_obj)

                kl = approx_kl_from_logprobs(new_logp, mb.ref_logprobs, mask)
                entropy = -masked_mean(new_logp, mask)

                is_clipped = ((ratio < 1.0 - cfg.clip_eps) | (ratio > 1.0 + cfg.clip_eps)).float()
                clipfrac = masked_mean(is_clipped, mask)

                is_dual = neg_mask * (ppo_obj < dual_floor).float()
                dual_clipfrac = masked_mean(is_dual, mask)

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
                total_dual_clipfrac += float(dual_clipfrac.detach().item())
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
            "train/fraction_of_completion_tokens_where_ppo_ratio_was_clipped_mean_over_minibatches": total_clipfrac / denom,
            "train/fraction_of_completion_tokens_where_dual_clip_was_active_mean_over_minibatches": total_dual_clipfrac / denom,
            "train/count_minibatches_skipped_because_completion_mask_had_no_tokens": float(skipped_empty),
            "train/count_update_attempts_skipped_due_to_nonfinite_loss_or_gradients": float(skipped_nonfinite),
            "train/gradient_global_norm_after_clipping_mean_over_optimizer_steps": total_grad_norm / max(1, opt_steps),
            "train/count_optimizer_steps_per_training_iteration": float(opt_steps),
        }
