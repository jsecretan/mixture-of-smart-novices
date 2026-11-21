"""Step 0: Calibrate mentor (Gemini 3.1 Pro) + judge (GPT-5.6 Luna) on:
1. Paper's Table 3 verbatim 7B naive dental seeds (n=3)
2. Olmo-3-7B-Instruct-SFT baseline seeds from naive-llm/results3/dental/gen.jsonl (4 per prompt = 12)
Compare judging WITH Faithfulness/Seed (paper setup) vs WITHOUT Faithfulness/Seed (naive-llm setup).
"""
import json
from pathlib import Path
from eval_harness import NAIVE_DIR, ROUND3_DIR, mentor_seeds, judge_items, summarize_scores, dogma_rate

OUT = ROUND3_DIR / "results" / "calibration"
OUT.mkdir(parents=True, exist_ok=True)

# 1. Paper Table 3 verbatim 7B naive dental seeds
paper_naive_seeds = [
    {"prompt": 0, "seed": 0, "text": "A miniaturized robotic armada that can be controlled to maneuver around teeth of any size. These miniature robotic arms could possibly be controlled by air or water pressure."},
    {"prompt": 2, "seed": 1, "text": "A stringless floss that can be controlled to move around teeth and remove plaque."},
    {"prompt": 1, "seed": 2, "text": "A miniature robotic arm that is controlled by a sensor in your saliva. The arm would be able to reach all areas of your mouth to remove plaque and food debris."},
]

# 2. Olmo-3-7B-Instruct-SFT baseline seeds (first 4 seeds per prompt = 12)
olmo_all = [json.loads(l) for l in open(NAIVE_DIR / "results3" / "dental" / "gen.jsonl")]
olmo_base = [r for r in olmo_all if r["cond"] == "baseline" and r["seed"] < 4]
olmo_prompted = [r for r in olmo_all if r["cond"] == "prompted" and r["seed"] < 4]

print("Dogma rates (seeds):")
print("  Paper Table 3 naive:", dogma_rate(paper_naive_seeds))
print("  Olmo baseline (n=12):", dogma_rate(olmo_base))
print("  Olmo prompted (n=12):", dogma_rate(olmo_prompted))

# Judge raw seeds too
s_paper = judge_items(paper_naive_seeds, "dental", OUT / "paper_t3_seeds.judge.jsonl", with_faithfulness=False)
s_base = judge_items(olmo_base, "dental", OUT / "olmo_base_seeds.judge.jsonl", with_faithfulness=False)
s_prompted = judge_items(olmo_prompted, "dental", OUT / "olmo_prompted_seeds.judge.jsonl", with_faithfulness=False)

# Mentor with Gemini 3.1 Pro (Appendix A verbatim)
ref_paper = mentor_seeds(paper_naive_seeds, "dental", OUT / "paper_t3.ref.jsonl")
ref_base = mentor_seeds(olmo_base, "dental", OUT / "olmo_base.ref.jsonl")
ref_prompted = mentor_seeds(olmo_prompted, "dental", OUT / "olmo_prompted.ref.jsonl")

# Judge refinements BOTH with faithfulness (paper protocol) and without (naive-llm protocol)
j_paper_f = judge_items(ref_paper, "dental", OUT / "paper_t3.judge_faith.jsonl", with_faithfulness=True)
j_paper_nf = judge_items(ref_paper, "dental", OUT / "paper_t3.judge_nofaith.jsonl", with_faithfulness=False)
j_base_f = judge_items(ref_base, "dental", OUT / "olmo_base.judge_faith.jsonl", with_faithfulness=True)
j_base_nf = judge_items(ref_base, "dental", OUT / "olmo_base.judge_nofaith.jsonl", with_faithfulness=False)
j_prompted_f = judge_items(ref_prompted, "dental", OUT / "olmo_prompted.judge_faith.jsonl", with_faithfulness=True)
j_prompted_nf = judge_items(ref_prompted, "dental", OUT / "olmo_prompted.judge_nofaith.jsonl", with_faithfulness=False)

print("\n=== CALIBRATION RESULTS (GPT-5.6 Luna, Appendix E verbatim) ===")
print("Paper Fig 5a targets:  Naive 7B Ref ~7.8 / 7.1 / 6.3 | Base 7B Ref ~6.6 / 6.9 / 6.3 | Prompted Ref ~5.6 / 6.1 / 5.6")
print("Paper Fig 5a seeds:    Naive 7B Seed ~6.1 / 5.7 / 4.7 | Base 7B Seed ~4.5 / 5.2 / 4.9 | Prompted Seed ~4.0 / 5.2 / 4.6")
print("-" * 110)
print(summarize_scores(s_paper, "Paper Table3 Seeds"))
print(summarize_scores(s_base, "Olmo Base Seeds (n=12)"))
print(summarize_scores(s_prompted, "Olmo Prompted Seeds (n=12)"))
print("-" * 110)
print(summarize_scores(j_paper_f, "Paper Table3 Ref (w/ Faith)"))
print(summarize_scores(j_paper_nf, "Paper Table3 Ref (no Faith)"))
print(summarize_scores(j_base_f, "Olmo Base Ref (w/ Faith)"))
print(summarize_scores(j_base_nf, "Olmo Base Ref (no Faith)"))
print(summarize_scores(j_prompted_f, "Olmo Prompted Ref (w/ Faith)"))
print(summarize_scores(j_prompted_nf, "Olmo Prompted Ref (no Faith)"))
