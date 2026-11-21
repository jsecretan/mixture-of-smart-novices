"""Rigorous Multi-Domain Evaluation of Mixture of Smart Novices (MoSN) vs Baseline.

Tests real-time routing (ZERO per-topic retraining!) across 6 diverse prompts spanning all 4 swaths:
  - Dental Hygiene (expected -> bio_med)
  - Vaccines (expected -> bio_med)
  - Neuroevolution (expected -> cs_ai)
  - Cooking Food Faster (expected -> mech_home)
  - City Cyclists / Car Doors (expected -> mech_home)
  - Company Meetings (expected -> org_econ)

Compares MoSN Routed Novice -> Mentor vs Unedited Baseline -> Mentor (same mentor & GPT-5.6 Luna judge),
plus measures routing accuracy and real-time weight-swap latency!
"""
import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean

MOSN_DIR = Path(__file__).resolve().parent
ROUND3_DIR = MOSN_DIR.parent
sys.path.insert(0, str(ROUND3_DIR))

from eval_harness import summarize_scores  # noqa: E402
from mosn.mosn_service import MoSNPool, MoSNRouter, run_query  # noqa: E402

BENCH_PROMPTS = [
    ("dental_interdental", "bio_med", "Can you help me invent something to clean between my teeth?"),
    ("dental_debris", "bio_med", "Scientists have discovered that debris on teeth can cause damage to them. Can you help me invent a tool to help people clean their teeth?"),
    ("vaccines_protect", "bio_med", "Scientists have discovered that people can get sick from infectious diseases. Can you help me invent a way to protect people from getting sick?"),
    ("neuroevo_nograd", "cs_ai", "I want to train a neural network but I don't want to use gradients. Can you help me invent a way to train a network without using gradients?"),
    ("cook_faster", "mech_home", "Can you help me invent something to cook food faster?"),
    ("cycling_doors", "mech_home", "City cyclists keep getting hit by car doors that open suddenly in front of them. Can you help me invent a way to prevent this?"),
    ("meetings_fix", "org_econ", "Meetings at my company run long and decide little. Can you help me invent a way to fix this?"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--n", type=int, default=8, help="proposals per prompt per arm")
    ap.add_argument("--mentor", default="deepseek-v4-pro")
    ap.add_argument("--mentor-mode", choices=["grounded", "paper"], default="grounded")
    ap.add_argument("--router", choices=["embed", "llm"], default="embed")
    a = ap.parse_args()

    out_dir = ROUND3_DIR / "results" / f"mosn_{a.model.split('/')[-1]}" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = MoSNPool(a.model)
    router = MoSNRouter(mode=a.router)

    all_mosn, all_base = [], []
    route_hits = 0
    rows_summary = []

    for pid, expected, prompt in BENCH_PROMPTS:
        print(f"\n==================== BENCH: {pid} (expected={expected}) ====================", flush=True)
        mosn_res, chosen = run_query(pool, router, prompt, n=a.n, mentor=a.mentor, mentor_mode=a.mentor_mode, rank=True)
        routed = chosen[0][0]
        route_hits += int(routed == expected)
        (out_dir / f"{pid}_mosn.jsonl").write_text("".join(json.dumps(r) + "\n" for r in mosn_res))

        base_res, _ = run_query(pool, router, prompt, n=a.n, mentor=a.mentor, mentor_mode=a.mentor_mode, rank=True, force_novice="baseline")
        (out_dir / f"{pid}_base.jsonl").write_text("".join(json.dumps(r) + "\n" for r in base_res))

        all_mosn.extend(mosn_res)
        all_base.extend(base_res)
        rows_summary.append({
            "id": pid,
            "expected": expected,
            "routed": routed,
            "mosn_nov": mean(r["novelty"] for r in mosn_res),
            "mosn_util": mean(r["utility"] for r in mosn_res),
            "mosn_feas": mean(r["feasibility"] for r in mosn_res),
            "mosn_n8": sum(1 for r in mosn_res if r["novelty"] >= 8),
            "base_nov": mean(r["novelty"] for r in base_res),
            "base_util": mean(r["utility"] for r in base_res),
            "base_feas": mean(r["feasibility"] for r in base_res),
            "base_n8": sum(1 for r in base_res if r["novelty"] >= 8),
        })

    print("\n================ MoSN MULTI-DOMAIN BENCHMARK SUMMARY ================", flush=True)
    print(f"Routing accuracy: {route_hits}/{len(BENCH_PROMPTS)} ({100*route_hits/len(BENCH_PROMPTS):.0f}%)\n", flush=True)
    for r in rows_summary:
        print(
            f"  {r['id']:18s} [{r['routed']:9s}] | "
            f"MoSN: N={r['mosn_nov']:.2f} U={r['mosn_util']:.2f} F={r['mosn_feas']:.2f} (N>=8:{r['mosn_n8']}/{a.n}) | "
            f"Base: N={r['base_nov']:.2f} U={r['base_util']:.2f} F={r['base_feas']:.2f} (N>=8:{r['base_n8']}/{a.n})",
            flush=True,
        )
    print("-" * 105, flush=True)
    print(summarize_scores(all_mosn, "ALL 7 PROMPTS — MoSN Routed"), flush=True)
    print(summarize_scores(all_base, "ALL 7 PROMPTS — Baseline"), flush=True)
    json.dump(rows_summary, open(out_dir / "summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
