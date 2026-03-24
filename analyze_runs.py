#!/usr/bin/env python3
"""Analyze metrics.jsonl files from RL training runs.

Usage:
    # Auto-discover all runs under runs/ (default):
    python analyze_runs.py

    # Explicit run directories:
    python analyze_runs.py runs/grpo-7b-math-* runs/cispo-7b-math-*

    # Specify runs/ root explicitly:
    python analyze_runs.py --runs_dir path/to/runs

    # Suppress plots (text tables only):
    python analyze_runs.py --no_plot

    # Save plots to a directory:
    python analyze_runs.py --save_dir figs/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── optional matplotlib ────────────────────────────────────────────────────────
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── metric aliases ─────────────────────────────────────────────────────────────
EVAL_ACC   = "eval/math_hard_test_subset_split_fraction_exact_match_using_boxed_answer_parser"
EVAL_RELAX = "eval/math_hard_test_subset_split_fraction_exact_match_using_relaxed_last_number_parser"
EVAL_BOXED = "eval/math_hard_test_subset_split_fraction_completions_containing_boxed_answer_pattern"

REWARD  = "rollout/mean_total_reward_across_all_completions_in_batch_and_groups"
KL      = "train/approximate_kl_divergence_policy_vs_reference_mean_over_minibatches"
LR      = "train/current_optimizer_learning_rate"
CLIP    = "train/fraction_of_completion_tokens_where_ppo_ratio_was_clipped_mean_over_minibatches"
ENTROPY = "train/policy_token_entropy_mean_over_minibatches"
GRAD    = "train/gradient_global_norm_after_clipping_mean_over_optimizer_steps"
COMP_LEN = "rollout/mean_generated_completion_token_count_per_completion"
HIT_MAX  = "rollout/fraction_of_completions_that_hit_max_new_tokens_limit"
NONZERO_ADV = "rollout/fraction_of_completions_with_nonzero_advantage"
DYN_FILTER  = "train/fraction_of_samples_filtered_by_dynamic_sampling_mean_over_minibatches"
DUAL_CLIP   = "train/fraction_of_completion_tokens_where_dual_clip_was_active_mean_over_minibatches"

# per-algo loss key variants
LOSS_KEYS = [
    "train/policy_loss_with_kl_penalty_mean_over_minibatches",
    "train/policy_loss_with_kl_and_entropy_mean_over_minibatches",
    "train/policy_loss_mean_over_minibatches",
]

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
           "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_metrics(path: Path) -> List[Dict]:
    """Return list of {step, metrics} dicts from a metrics.jsonl file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "step" in d and "metrics" in d:
                records.append({"step": int(d["step"]), "metrics": d["metrics"]})
    return sorted(records, key=lambda r: r["step"])


def discover_runs(runs_dir: Path) -> Dict[str, Path]:
    """Return {run_name: metrics.jsonl path} for every run folder found."""
    found = {}
    for p in sorted(runs_dir.iterdir()):
        if not p.is_dir():
            continue
        m = p / "metrics.jsonl"
        if m.exists():
            found[p.name] = m
    return found


def short_name(run_name: str) -> str:
    """Turn 'grpo-7b-math-20260313_163053' → 'grpo-7b-math-2026…' for display."""
    parts = run_name.split("-")
    # keep algo + model size + task; truncate timestamp
    return "-".join(parts[:3]) if len(parts) >= 3 else run_name


def extract_series(records: List[Dict], key: str) -> Tuple[List[int], List[float]]:
    """Pull (steps, values) for a given metric key."""
    steps, vals = [], []
    for r in records:
        v = r["metrics"].get(key)
        if v is not None:
            steps.append(r["step"])
            vals.append(float(v))
    return steps, vals


def rolling_mean(values: List[float], window: int = 20) -> List[float]:
    out = []
    for i, v in enumerate(values):
        lo = max(0, i - window + 1)
        out.append(sum(values[lo : i + 1]) / (i - lo + 1))
    return out


# ── section printers ──────────────────────────────────────────────────────────

def sep(char: str = "─", width: int = 72) -> None:
    print(char * width)


def print_header(title: str) -> None:
    sep("═")
    print(f"  {title}")
    sep("═")


def print_section(title: str) -> None:
    print()
    sep()
    print(f"  {title}")
    sep()


def fmt(v, decimals: int = 4) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "   n/a  "
    return f"{v:>{decimals + 5}.{decimals}f}"


# ── summary table ─────────────────────────────────────────────────────────────

def print_summary_table(all_runs: Dict[str, List[Dict]]) -> None:
    print_header("RUN SUMMARY")

    col_w = 14
    names = list(all_runs.keys())
    header = f"{'metric':<44}" + "".join(f"{short_name(n):>{col_w}}" for n in names)
    print(header)
    sep("-")

    def last(records: List[Dict], key: str) -> Optional[float]:
        for r in reversed(records):
            v = r["metrics"].get(key)
            if v is not None:
                return float(v)
        return None

    def best_eval(records: List[Dict]) -> Optional[float]:
        vals = [r["metrics"][EVAL_ACC] for r in records if EVAL_ACC in r["metrics"]]
        return max(vals) if vals else None

    def loss_key(records: List[Dict]) -> Optional[str]:
        for r in records:
            for k in LOSS_KEYS:
                if k in r["metrics"]:
                    return k
        return None

    rows = [
        ("Baseline accuracy (step 0)",       lambda rs: rs[0]["metrics"].get(EVAL_ACC)),
        ("Final accuracy  (boxed)",           lambda rs: last(rs, EVAL_ACC)),
        ("Best accuracy   (boxed)",           best_eval),
        ("Final accuracy  (relaxed)",         lambda rs: last(rs, EVAL_RELAX)),
        ("Final boxed-completion rate",       lambda rs: last(rs, EVAL_BOXED)),
        ("Final mean reward",                 lambda rs: last(rs, REWARD)),
        ("Final KL divergence",               lambda rs: last(rs, KL)),
        ("Peak KL divergence",                lambda rs: max((r["metrics"][KL] for r in rs if KL in r["metrics"]), default=None)),
        ("Final clip fraction",               lambda rs: last(rs, CLIP)),
        ("Final entropy",                     lambda rs: last(rs, ENTROPY)),
        ("Final grad norm",                   lambda rs: last(rs, GRAD)),
        ("Final mean completion length",      lambda rs: last(rs, COMP_LEN)),
        ("Frac completions hitting max_toks", lambda rs: last(rs, HIT_MAX)),
    ]

    for label, fn in rows:
        vals = []
        for n, rs in all_runs.items():
            try:
                v = fn(rs)
            except Exception:
                v = None
            vals.append(v)
        row = f"{label:<44}"
        for v in vals:
            row += f"{fmt(v):>{col_w}}"
        print(row)

    # algo-specific extras
    extras = [
        ("DAPO: dyn-sample filter rate",   lambda rs: last(rs, DYN_FILTER)),
        ("CISPO: dual-clip active rate",   lambda rs: last(rs, DUAL_CLIP)),
    ]
    for label, fn in extras:
        vals = [fn(rs) for rs in all_runs.values()]
        if any(v is not None for v in vals):
            row = f"{label:<44}"
            for v in vals:
                row += f"{fmt(v):>{col_w}}"
            print(row)

    sep()
    print(f"  Steps logged: {', '.join(str(max(r['step'] for r in rs)) for rs in all_runs.values())}")


# ── eval accuracy table (one row per eval step) ───────────────────────────────

def print_eval_table(all_runs: Dict[str, List[Dict]]) -> None:
    print_section("EVAL ACCURACY  (boxed exact match)  —  per eval step")

    names = list(all_runs.keys())
    col_w = 10
    header = f"{'step':>6}  " + "  ".join(f"{short_name(n):>{col_w}}" for n in names)
    print(header)
    sep("-")

    # collect all eval steps across all runs
    eval_steps: dict[int, dict[str, float]] = defaultdict(dict)
    for name, records in all_runs.items():
        for r in records:
            v = r["metrics"].get(EVAL_ACC)
            if v is not None:
                eval_steps[r["step"]][name] = float(v)

    for step in sorted(eval_steps.keys()):
        row = f"{step:>6}  "
        row_vals = []
        for n in names:
            v = eval_steps[step].get(n)
            row_vals.append(v)
            row += f"{fmt(v, 4):>{col_w}}  "
        # highlight best
        valid = [(i, v) for i, v in enumerate(row_vals) if v is not None]
        if valid:
            best_i = max(valid, key=lambda x: x[1])[0]
            # re-render with marker
            row = f"{step:>6}  "
            for i, (n, v) in enumerate(zip(names, row_vals)):
                marker = "*" if i == best_i else " "
                cell = f"{fmt(v, 4)}{marker}"
                row += f"{cell:>{col_w + 1}}  "
        print(row)

    sep()
    print("  * = best at that step")


# ── training stats table (smoothed over window) ───────────────────────────────

def print_training_table(all_runs: Dict[str, List[Dict]], window: int = 50) -> None:
    print_section(f"TRAINING METRICS  (smoothed, window={window} steps)")

    names = list(all_runs.keys())
    col_w = 10
    checkpoints = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    for metric_label, key in [
        ("Mean reward", REWARD),
        ("KL divergence", KL),
        ("Clip fraction", CLIP),
        ("Completion length", COMP_LEN),
    ]:
        print(f"\n  {metric_label}")
        header = f"{'step':>6}  " + "  ".join(f"{short_name(n):>{col_w}}" for n in names)
        print("  " + header)
        print("  " + "-" * len(header))

        # build smoothed lookup per run: step -> value
        smoothed: dict[str, dict[int, float]] = {}
        for name, records in all_runs.items():
            steps, vals = extract_series(records, key)
            sm = rolling_mean(vals, window)
            smoothed[name] = dict(zip(steps, sm))

        for ck in checkpoints:
            row = f"{ck:>6}  "
            for n in names:
                # find nearest step at or before checkpoint
                s = smoothed[n]
                candidates = [st for st in s if st <= ck]
                v = s[max(candidates)] if candidates else None
                row += f"{fmt(v, 4):>{col_w}}  "
            print("  " + row)


# ── plotting ──────────────────────────────────────────────────────────────────

def make_plots(
    all_runs: Dict[str, List[Dict]],
    save_dir: Optional[Path] = None,
) -> None:
    if not HAS_MPL:
        print("\n[plots skipped: matplotlib not installed]")
        return

    names = list(all_runs.keys())
    colors = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(names)}

    def _ax_series(ax, name, records, key, smooth=0, label=None, **kw):
        steps, vals = extract_series(records, key)
        if not steps:
            return
        if smooth > 1:
            vals = rolling_mean(vals, smooth)
        ax.plot(steps, vals, label=label or short_name(name),
                color=colors[name], **kw)

    def _save_or_show(fig, fname):
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            out = save_dir / fname
            fig.savefig(out, dpi=130, bbox_inches="tight")
            print(f"  Saved: {out}")
        else:
            plt.show()
        plt.close(fig)

    # ── Figure 1: eval accuracy ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Eval Accuracy — math_hard (512 examples)", fontsize=13, fontweight="bold")

    for name, records in all_runs.items():
        for ax, key, title in [
            (axes[0], EVAL_ACC,   "Boxed exact match"),
            (axes[1], EVAL_RELAX, "Relaxed (last-number) match"),
        ]:
            steps, vals = extract_series(records, key)
            if steps:
                ax.plot(steps, vals, "o-", ms=4, label=short_name(name), color=colors[name])

    for ax, title in zip(axes, ["Boxed exact match accuracy", "Relaxed exact match accuracy"]):
        ax.set_title(title)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Fraction correct")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlim(left=0)

    fig.tight_layout()
    _save_or_show(fig, "01_eval_accuracy.png")

    # ── Figure 2: training reward + KL ────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Training Dynamics", fontsize=13, fontweight="bold")

    plots = [
        (axes[0, 0], REWARD,   "Mean rollout reward (smooth 50)", True,  False),
        (axes[0, 1], KL,       "KL divergence (smooth 20)",       True,  True),
        (axes[1, 0], CLIP,     "Clip fraction  (smooth 20)",      True,  False),
        (axes[1, 1], COMP_LEN, "Mean completion length",          False, False),
    ]

    for ax, key, title, smooth, logy in plots:
        for name, records in all_runs.items():
            w = 50 if smooth and "reward" in key else 20
            _ax_series(ax, name, records, key, smooth=w if smooth else 0)
        ax.set_title(title)
        ax.set_xlabel("Training step")
        ax.grid(alpha=0.3)
        ax.set_xlim(left=0)
        if logy:
            ax.set_yscale("symlog", linthresh=0.1)
        ax.legend(fontsize=8)

    fig.tight_layout()
    _save_or_show(fig, "02_training_dynamics.png")

    # ── Figure 3: entropy + grad norm + LR ───────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Optimizer & Policy Diagnostics", fontsize=13, fontweight="bold")

    for ax, key, title in [
        (axes[0], ENTROPY, "Token entropy (smooth 20)"),
        (axes[1], GRAD,    "Gradient norm (smooth 20)"),
        (axes[2], LR,      "Learning rate"),
    ]:
        for name, records in all_runs.items():
            _ax_series(ax, name, records, key, smooth=20)
        ax.set_title(title)
        ax.set_xlabel("Training step")
        ax.grid(alpha=0.3)
        ax.set_xlim(left=0)
        ax.legend(fontsize=8)

    fig.tight_layout()
    _save_or_show(fig, "03_diagnostics.png")

    # ── Figure 4: algo-specific extras ───────────────────────────────────
    extras_present = any(
        any(k in r["metrics"] for r in rs for k in [DYN_FILTER, DUAL_CLIP])
        for rs in all_runs.values()
    )
    if extras_present:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Algorithm-specific Metrics", fontsize=13, fontweight="bold")

        for ax, key, title in [
            (axes[0], DYN_FILTER, "DAPO: dynamic-sample filter rate"),
            (axes[1], DUAL_CLIP,  "CISPO: dual-clip active rate"),
        ]:
            for name, records in all_runs.items():
                _ax_series(ax, name, records, key, smooth=20)
            ax.set_title(title)
            ax.set_xlabel("Training step")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)

        fig.tight_layout()
        _save_or_show(fig, "04_algo_specific.png")

    # ── Figure 5: boxed completion rate ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle("Fraction of Completions Containing \\\\boxed{} Pattern", fontsize=12)
    for name, records in all_runs.items():
        steps, vals = extract_series(records, EVAL_BOXED)
        if steps:
            ax.plot(steps, vals, "o-", ms=4, label=short_name(name), color=colors[name])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_xlabel("Training step")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, "05_boxed_rate.png")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", nargs="*", help="Explicit run directories (each must contain metrics.jsonl)")
    p.add_argument("--runs_dir", default="runs", help="Root dir to auto-discover runs from (default: runs/)")
    p.add_argument("--no_plot", action="store_true", help="Skip all plots")
    p.add_argument("--save_dir", default="", help="Directory to save plots (default: show interactively)")
    p.add_argument("--window", type=int, default=50, help="Smoothing window for training tables (default: 50)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── collect run paths ──────────────────────────────────────────────────
    run_paths: Dict[str, Path] = {}

    if args.run_dirs:
        for d in args.run_dirs:
            p = Path(d)
            m = p / "metrics.jsonl"
            if not m.exists():
                print(f"[warn] No metrics.jsonl in {p}, skipping.", file=sys.stderr)
                continue
            run_paths[p.name] = m
    else:
        runs_root = Path(args.runs_dir)
        if not runs_root.exists():
            sys.exit(f"[error] runs_dir not found: {runs_root}")
        run_paths = discover_runs(runs_root)
        # skip non-algo dirs (logs/, etc.)
        run_paths = {k: v for k, v in run_paths.items()
                     if not k.startswith("log") and "-" in k}

    if not run_paths:
        sys.exit("[error] No runs found. Pass run directories explicitly or check --runs_dir.")

    print(f"\nFound {len(run_paths)} run(s):")
    for name in run_paths:
        print(f"  {name}")

    # ── load ───────────────────────────────────────────────────────────────
    all_runs: Dict[str, List[Dict]] = {}
    for name, path in run_paths.items():
        records = load_metrics(path)
        if not records:
            print(f"[warn] {name}: empty metrics.jsonl, skipping.", file=sys.stderr)
            continue
        all_runs[name] = records
        print(f"  Loaded {len(records):>5} records  ·  {name}")

    if not all_runs:
        sys.exit("[error] All runs were empty.")

    # ── analysis ───────────────────────────────────────────────────────────
    print_summary_table(all_runs)
    print_eval_table(all_runs)
    print_training_table(all_runs, window=args.window)

    if not args.no_plot:
        print_section("PLOTS")
        save_dir = Path(args.save_dir) if args.save_dir else None
        if save_dir:
            print(f"  Saving to: {save_dir}")
        else:
            if not HAS_MPL:
                print("  matplotlib not available — skipping plots.")
            else:
                print("  Displaying plots interactively (pass --save_dir to save instead).")
        make_plots(all_runs, save_dir=save_dir)

    print()
    sep("═")
    print("  Done.")
    sep("═")


if __name__ == "__main__":
    main()
