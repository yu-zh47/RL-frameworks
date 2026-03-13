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


class GRPO(RLAlgorithm):
    """GRPO update with a PPO-style clipped surrogate over completion tokens."""

    name = "grpo"

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
        rng = torch.Generator(device="cpu") # Expected a 'cpu' device type for generator but found 'cuda'

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

                clamp = 20
                new_logp = compute_per_token_logprobs(model, mb.input_ids, mb.attention_mask)
                log_ratio = torch.clamp(new_logp - mb.old_logprobs, min=-clamp, max=clamp)
                ratio = torch.exp(log_ratio)
                adv = adv.unsqueeze(1)
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv
                per_token_obj = torch.minimum(unclipped, clipped) * mask
                seq_logp = masked_mean_per_row(per_token_obj, mask)  # [B_mb]
                pg_loss = -torch.mean(seq_logp)
                kl = approx_kl_from_logprobs(new_logp,mb.ref_logprobs, mask)
                entropy = -masked_mean(new_logp, mask)
                is_clipped = ((ratio < 1 - cfg.clip_eps) | (ratio > 1 + cfg.clip_eps)).float()
                clipfrac = masked_mean(is_clipped, mask)


                loss = (pg_loss + cfg.kl_coef * kl) / max(1, grad_accum_steps)
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
            "train/policy_loss_with_kl_penalty_mean_over_minibatches": total_loss / denom,
            "train/approximate_kl_divergence_policy_vs_reference_mean_over_minibatches": total_kl / denom,
            "train/policy_token_entropy_mean_over_minibatches": total_entropy / denom,
            "train/fraction_of_completion_tokens_where_ppo_ratio_was_clipped_mean_over_minibatches": total_clipfrac / denom,
            "train/count_minibatches_skipped_because_completion_mask_had_no_tokens": float(skipped_empty),
            "train/count_update_attempts_skipped_due_to_nonfinite_loss_or_gradients": float(skipped_nonfinite),
            "train/gradient_global_norm_after_clipping_mean_over_optimizer_steps": total_grad_norm / max(1, opt_steps),
            "train/count_optimizer_steps_per_training_iteration": float(opt_steps),
        }
