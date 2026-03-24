import json, os

runs = {
    "grpo":  "/usr/yue/RL/LLM/runs/grpo-7b-math-20260313_163053/metrics.jsonl",
    "dapo":  "/usr/yue/RL/LLM/runs/dapo-7b-math-20260313_163053/metrics.jsonl",
    "cispo": "/usr/yue/RL/LLM/runs/cispo-7b-math-20260314_154511/metrics.jsonl",
    "gspo":  "/usr/yue/RL/LLM/runs/gspo-7b-math-20260313_163053/metrics.jsonl",
}

key = "eval/math_hard_test_subset_split_fraction_exact_match_using_boxed_answer_parser"

print(f"{'step':>6}  {'grpo':>8}  {'cispo':>8}  {'gspo':>8}  {'dapo':>8}")
print("-" * 50)

for algo, path in runs.items():
    evals = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if key in d.get("metrics", {}):
                evals[d["step"]] = d["metrics"][key]
    runs[algo] = evals

all_steps = sorted(runs["grpo"].keys())
for s in all_steps:
    vals = {a: runs[a].get(s, float("nan")) for a in ["grpo", "cispo", "gspo", "dapo"]}
    print(f"{s:>6}  {vals['grpo']:>8.4f}  {vals['cispo']:>8.4f}  {vals['gspo']:>8.4f}  {vals['dapo']:>8.4f}")
