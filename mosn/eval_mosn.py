"""Rigorous Multi-Domain Evaluation of Growing Narrow-Topic MoSN (MoSN v2) vs Broad MoSN & Baseline.

Tests real-time routing (ZERO per-topic retraining on known domains!) across the exact same 7 diverse
prompts used for Broad 4-Swath MoSN:
  - dental_interdental  (expected -> dental / bio_med)
  - dental_debris       (expected -> dental / bio_med)
  - vaccines_protect    (expected -> vaccines / bio_med)
  - neuroevo_nograd     (expected -> neuroevolution / cs_ai)
  - cook_faster         (expected -> cooking / mech_home)
  - cycling_doors       (expected -> cycling_doors / mech_home)
  - meetings_fix        (expected -> meetings / org_econ)

Optional `--test-autogrow` ALSO tests an 8th completely unseen domain (produce freshness / food spoilage)
to verify confidence-gated cache miss -> automatic IACF-v2 novice growth -> permanent registry hit!
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
from mosn.narrow_pool import NarrowMoSNPool, NarrowMoSNRouter  # noqa: E402

BENCH_PROMPTS = [
    ("dental_interdental", "dental", "bio_med", "Can you help me invent something to clean between my teeth?"),
    ("dental_debris", "dental", "bio_med", "Scientists have discovered that debris on teeth can cause damage to them. Can you help me invent a tool to help people clean their teeth?"),
    ("vaccines_protect", "vaccines", "bio_med", "Scientists have discovered that people can get sick from infectious diseases. Can you help me invent a way to protect people from getting sick?"),
    ("neuroevo_nograd", "neuroevolution", "cs_ai", "I want to train a neural network but I don't want to use gradients. Can you help me invent a way to train a network without using gradients?"),
    ("cook_faster", "cooking", "mech_home", "Can you help me invent something to cook food faster?"),
    ("cycling_doors", "cycling_doors", "mech_home", "City cyclists keep getting hit by car doors that open suddenly in front of them. Can you help me invent a way to prevent this?"),
    ("meetings_fix", "meetings", "org_econ", "Meetings at my company run long and decide little. Can you help me invent a way to fix this?"),
]

UNSEEN_AUTOGROW_PROMPT = (
    "produce_freshness",
    "Fresh fruits and vegetables spoil too quickly in home kitchens, creating huge food waste. "
    "Can you help me invent a new way to keep fresh produce crisp and edible for weeks?",
)
UNSEEN_FOLLOWUP_PROMPT = "Can you invent a gadget that stops berries and leafy greens from rotting quickly at home?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--pool", choices=["narrow", "broad"], default="narrow")
    ap.add_argument("--n", type=int, default=8, help="proposals per prompt per arm")
    ap.add_argument("--mentor", default="deepseek-v4-pro")
    ap.add_argument("--mentor-mode", choices=["grounded", "paper"], default="grounded")
    ap.add_argument("--router", choices=["embed", "llm"], default="embed")
    ap.add_argument("--threshold", type=float, default=0.52)
    ap.add_argument("--test-autogrow", action="store_true", help="Also live-test cache miss -> auto-grow -> follow-up cache hit on an unseen 8th domain")
    ap.add_argument("--reuse-base", action="store_true", help="Reuse cached *_base.jsonl if already generated for paired comparison")
    a = ap.parse_args()

    model_tag = a.model.split("/")[-1]
    out_dir = ROUND3_DIR / "results" / f"mosn_{a.pool}_{model_tag}" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    if a.pool == "narrow":
        pool = NarrowMoSNPool(a.model)
        router = NarrowMoSNRouter(model_tag=model_tag, mode=a.router)
    else:
        pool = MoSNPool(a.model)
        router = MoSNRouter(mode=a.router)

    all_mosn, all_base = [], []
    route_hits = 0
    rows_summary = []

    for pid, exp_narrow, exp_broad, prompt in BENCH_PROMPTS:
        expected = exp_narrow if a.pool == "narrow" else exp_broad
        print(f"\n==================== BENCH ({a.pool}): {pid} (expected={expected}) ====================", flush=True)
        mosn_res, chosen = run_query(
            pool, router, prompt, n=a.n, mentor=a.mentor, mentor_mode=a.mentor_mode,
            rank=True, auto_grow=False, threshold=a.threshold,
        )
        routed = chosen[0][0]
        route_hits += int(routed == expected)
        (out_dir / f"{pid}_mosn.jsonl").write_text("".join(json.dumps(r) + "\n" for r in mosn_res))

        base_f = out_dir / f"{pid}_base.jsonl"
        if a.reuse_base and base_f.exists():
            base_res = [json.loads(l) for l in open(base_f)]
            print(f"[mosn] Reused cached baseline ({len(base_res)} proposals) from {base_f.name}", flush=True)
        else:
            base_res, _ = run_query(
                pool, router, prompt, n=a.n, mentor=a.mentor, mentor_mode=a.mentor_mode,
                rank=True, force_novice="baseline",
            )
            base_f.write_text("".join(json.dumps(r) + "\n" for r in base_res))

        all_mosn.extend(mosn_res)
        all_base.extend(base_res)
        rows_summary.append({
            "id": pid,
            "expected": expected,
            "routed": routed,
            "route_sim": round(chosen[0][1], 3),
            "mosn_nov": mean(r["novelty"] for r in mosn_res),
            "mosn_util": mean(r["utility"] for r in mosn_res),
            "mosn_feas": mean(r["feasibility"] for r in mosn_res),
            "mosn_n8": sum(1 for r in mosn_res if r["novelty"] >= 8),
            "mosn_hit": 100 * mean(r["novelty"] >= 7 and r["utility"] >= 6 for r in mosn_res),
            "base_nov": mean(r["novelty"] for r in base_res),
            "base_util": mean(r["utility"] for r in base_res),
            "base_feas": mean(r["feasibility"] for r in base_res),
            "base_n8": sum(1 for r in base_res if r["novelty"] >= 8),
            "base_hit": 100 * mean(r["novelty"] >= 7 and r["utility"] >= 6 for r in base_res),
        })

    print(f"\n================ MoSN ({a.pool.upper()}) MULTI-DOMAIN BENCHMARK SUMMARY ================", flush=True)
    print(f"Routing accuracy: {route_hits}/{len(BENCH_PROMPTS)} ({100*route_hits/len(BENCH_PROMPTS):.0f}%)\n", flush=True)
    for r in rows_summary:
        print(
            f"  {r['id']:18s} [{r['routed']:14s} sim={r['route_sim']:.2f}] | "
            f"MoSN: N={r['mosn_nov']:.2f} U={r['mosn_util']:.2f} F={r['mosn_feas']:.2f} hit={r['mosn_hit']:3.0f}% (N>=8:{r['mosn_n8']}/{a.n}) | "
            f"Base: N={r['base_nov']:.2f} U={r['base_util']:.2f} F={r['base_feas']:.2f} hit={r['base_hit']:3.0f}% (N>=8:{r['base_n8']}/{a.n})",
            flush=True,
        )
    print("-" * 120, flush=True)
    print(summarize_scores(all_mosn, f"ALL 7 PROMPTS — MoSN ({a.pool})"), flush=True)
    print(summarize_scores(all_base, "ALL 7 PROMPTS — Baseline"), flush=True)
    json.dump(rows_summary, open(out_dir / "summary.json", "w"), indent=2)

    if a.test_autogrow and a.pool == "narrow":
        pid, uprompt = UNSEEN_AUTOGROW_PROMPT
        print(f"\n==================== AUTO-GROW LIVE TEST (UNSEEN 8TH DOMAIN: {pid}) ====================", flush=True)
        # Check initial similarity before growing
        _, pre_ranked, pre_hit = router.route(uprompt, top_k=1, threshold=a.threshold)
        print(f"  Pre-grow top match: {pre_ranked[0] if pre_ranked else None} (is_hit={pre_hit})", flush=True)
        ag_res, ag_chosen = run_query(
            pool, router, uprompt, n=a.n, mentor=a.mentor, mentor_mode=a.mentor_mode,
            rank=True, auto_grow=True, threshold=a.threshold,
        )
        (out_dir / f"{pid}_autogrow_mosn.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ag_res))
        ag_base, _ = run_query(
            pool, router, uprompt, n=a.n, mentor=a.mentor, mentor_mode=a.mentor_mode,
            rank=True, force_novice="baseline",
        )
        (out_dir / f"{pid}_autogrow_base.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ag_base))

        # Verify follow-up paraphrase hits the newly grown novice instantly!
        fu_chosen, fu_ranked, fu_hit = router.route(UNSEEN_FOLLOWUP_PROMPT, top_k=1, threshold=a.threshold)
        print(
            f"  Follow-up paraphrase routing check: '{UNSEEN_FOLLOWUP_PROMPT}' -> "
            f"{fu_chosen[0]} (is_hit={fu_hit}, expected={ag_chosen[0][0]})",
            flush=True,
        )
        print(summarize_scores(ag_res, f"AUTO-GROW ({ag_chosen[0][0]})"), flush=True)
        print(summarize_scores(ag_base, "UNSEEN DOMAIN — Baseline"), flush=True)
        json.dump({
            "unseen_prompt": uprompt,
            "pre_grow_top": pre_ranked[0] if pre_ranked else None,
            "pre_grow_hit": pre_hit,
            "grown_novice": ag_chosen[0][0],
            "followup_prompt": UNSEEN_FOLLOWUP_PROMPT,
            "followup_top": fu_chosen[0] if fu_chosen else None,
            "followup_hit": fu_hit,
            "mosn_nov": mean(r["novelty"] for r in ag_res),
            "mosn_util": mean(r["utility"] for r in ag_res),
            "mosn_feas": mean(r["feasibility"] for r in ag_res),
            "mosn_n8": sum(1 for r in ag_res if r["novelty"] >= 8),
            "base_nov": mean(r["novelty"] for r in ag_base),
            "base_util": mean(r["utility"] for r in ag_base),
            "base_feas": mean(r["feasibility"] for r in ag_base),
            "base_n8": sum(1 for r in ag_base if r["novelty"] >= 8),
        }, open(out_dir / "autogrow_test.json", "w"), indent=2)


if __name__ == "__main__":
    main()
