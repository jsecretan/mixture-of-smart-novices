"""Stage 4: Comprehensive Apples-to-Apples Evaluation of the 8B-Base First-Principles Seeder.

Mentoring: 100% local Ollama `qwen3.8:27b` (`MENTOR_GROUNDED` + `MENTOR_PAPER` where applicable).
Judging  : `gpt-5.6-luna` verbatim Appendix E rubric (`w/Faith` + `noFaith`) for direct table parity!

Evaluates ALL arms cleanly (unloading PyTorch before Ollama mentoring to prevent VRAM collisions):
  1. Held-out `dental` (n=36) & `neuroevolution` (n=36):
     - `8B-Base Seeder SFT` (zero-shot, zero per-topic forgetting)
     - `8B-Base Seeder GRPO` (zero-shot, zero per-topic forgetting)
     - `8B-Base Seeder (GRPO) + 1-round IACF-v2` (stacked selective forgetting!)
     - Apples-to-apples controls mentored by `qwen3.8:27b`:
       * `Qwen3-8B` Instruct Baseline (existing seeds on disk)
       * `Qwen3-4B-Instruct-2507` Baseline R0 (existing seeds on disk)
       * `Qwen3-4B-Instruct-2507` Goldilocks IACF-v2 Champion (existing seeds on disk!)
  2. Held-out 7-Prompt Multi-Domain Benchmark (n=8/prompt = 56 proposals, Grounded `qwen3.8:27b` mentor):
     - `8B-Base Seeder SFT` vs `8B-Base Seeder GRPO` vs `Qwen3-4B` Baseline (existing seeds!)
"""
import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SEEDER_DIR = Path(__file__).resolve().parent
ROUND3_DIR = SEEDER_DIR.parent
sys.path.insert(0, str(ROUND3_DIR))

from eval_harness import (  # noqa: E402
    FAITHFULNESS_RUBRIC,
    MENTOR_PAPER,
    RUBRIC_CORE,
    TOPICS,
    ask_llm,
    dogma_rate,
    eval_mcq_model,
    judge_items,
    summarize_scores,
)
from mosn.eval_mosn import BENCH_PROMPTS  # noqa: E402
from naive_service import MENTOR_GROUNDED, is_bad_seed  # noqa: E402
from runner import eval_model_full, generate_batch  # noqa: E402
from seeder.synth_corpus import unload_ollama  # noqa: E402


def mentor_local_ollama(seeds, topic_desc, out_path, tmpl, mentor_model="qwen3.8:27b"):
    """Sequential/low-concurrency local Ollama mentoring (resume-safe)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["prompt"], r["seed"])] = r
    todo = [r for r in seeds if (r["prompt"], r["seed"]) not in done]
    if todo:
        print(f"  [mentor:{mentor_model}] Refining {len(todo)}/{len(seeds)} seeds -> {out_path.name}...", flush=True)
    with open(out_path, "a") as fh:
        for idx, r in enumerate(todo):
            prompt = tmpl.format(topic_description=topic_desc, seed_text=r["text"])
            txt = ask_llm(mentor_model, prompt, max_tokens=1600, temperature=0.7)
            if txt:
                row = {
                    "prompt": r["prompt"],
                    "seed": r["seed"],
                    "seed_text": r["text"],
                    "text": txt,
                    "mentor": mentor_model,
                }
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                done[(r["prompt"], r["seed"])] = row
            if (idx + 1) % 12 == 0:
                print(f"    mentored {idx+1}/{len(todo)}", flush=True)
    return sorted(done.values(), key=lambda x: (x["prompt"], x["seed"]))


def generate_hf_seeds_for_model(model_path, out_root, tag, topics=("dental", "neuroevolution"), run_7prompt=True):
    """Load ONE HF model, generate all raw seeds + MCQ diagnostics, then free 100% of VRAM!"""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    need_any = False
    for t in topics:
        if not (out_root / f"{tag}_{t}" / "seeds.jsonl").exists():
            need_any = True
    if run_7prompt and not (out_root / f"{tag}_7prompt" / "seeds.jsonl").exists():
        need_any = True
    if not need_any:
        print(f"[gen:{tag}] All raw seeds already generated on disk, skipping GPU load.", flush=True)
        return

    print(f"\n=== Generating raw seeds for {tag} ({model_path}) ===", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto", max_memory={0: "11GiB", 1: "11GiB"}
    ).eval()

    for t in topics:
        tdir = out_root / f"{tag}_{t}"
        if not (tdir / "seeds.jsonl").exists():
            eval_model_full(model, tok, tdir, topic=t, n_seeds=12, max_new=650, run_mentor=False, extra_meta={"arm": tag})

    if run_7prompt:
        p7_dir = out_root / f"{tag}_7prompt"
        p7_dir.mkdir(parents=True, exist_ok=True)
        if not (p7_dir / "seeds.jsonl").exists():
            rows = []
            for pi, (pid, _, _, q) in enumerate(BENCH_PROMPTS):
                outs = generate_batch(model, tok, q, n=8, max_new=650, temperature=0.85, seed=100 * pi)
                for si, txt in enumerate(outs):
                    rows.append({"prompt": pi, "pid": pid, "prompt_text": q, "seed": si, "text": txt, "bad": is_bad_seed(txt)})
            (p7_dir / "seeds.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            print(f"  [7-prompt:{tag}] Generated {len(rows)} seeds (bad={sum(r['bad'] for r in rows)}/{len(rows)})", flush=True)

    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2.0)


def run_iacf_on_seeder(seeder_path, topic, out_dir):
    out_dir = Path(out_dir)
    if (out_dir / "winner" / "seeds.jsonl").exists():
        print(f"[iacf-seeder:{topic}] Already exists at {out_dir / 'winner'}, skipping.", flush=True)
        return
    print(f"\n=== Stacking 1-round IACF-v2 on 8B-Base Seeder ({topic}) ===", flush=True)
    py = sys.executable
    cmd = [
        py, str(ROUND3_DIR / "iacf_v2.py"),
        "--model", str(seeder_path),
        "--topic", topic,
        "--out", str(out_dir),
        "--rounds", "1",
        "--ssd_pct", "0.045",
        "--problem_mult", "6.0",
        "--heal_steps", "240",
        "--heal_lr", "3.5e-5",
        "--kl_w", "2.5",
        "--n_fisher_batches", "28",
        "--bs", "2",
        "--max_len", "300",
        "--mlp_only_ssd",
        "--heal_down_only",
        "--screen_n", "6",
        "--final_n", "12",
        "--mentor_n", "0",
    ]
    subprocess.run(cmd, check=True, cwd=str(ROUND3_DIR))


def judge_7prompt(refined_rows, out_path, judge_model="gpt-5.6-luna"):
    out_path = Path(out_path)
    done = {}
    if out_path.exists():
        for l in open(out_path):
            if l.strip():
                r = json.loads(l)
                done[(r["prompt"], r["seed"])] = r
    todo = [r for r in refined_rows if (r["prompt"], r["seed"]) not in done]
    prompts_by_i = {i: q for i, (_, _, _, q) in enumerate(BENCH_PROMPTS)}
    import re, threading
    from concurrent.futures import ThreadPoolExecutor
    lock = threading.Lock()
    with open(out_path, "a") as fh:
        def one(r):
            q = prompts_by_i[r["prompt"]]
            body = (
                RUBRIC_CORE + FAITHFULNESS_RUBRIC
                + f"\n\nRequest: {q}\n\nSeed Idea:\n{r['seed_text'][:5000]}\n\nRefined Idea:\n{r['text'][:15000]}\n\n"
                + 'Answer with JSON only: {"novelty": n, "utility": n, "feasibility": n, "faithfulness": n}'
            )
            raw = ask_llm(judge_model, body, json_mode=True)
            if not raw:
                return
            try:
                sc = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
                row = {"prompt": r["prompt"], "seed": r["seed"], **{k: float(sc[k]) for k in ("novelty", "utility", "feasibility", "faithfulness")}}
                with lock:
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    done[(r["prompt"], r["seed"])] = row
            except Exception as e:
                print("judge7 err:", e, flush=True)
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(one, todo))
    return sorted(done.values(), key=lambda x: (x["prompt"], x["seed"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-model", default="/home/jimmy/code/creative-toolkit/models/seeder_8b/sft_merged")
    ap.add_argument("--grpo-model", default="/home/jimmy/code/creative-toolkit/models/seeder_8b/grpo_merged")
    ap.add_argument("--mentor", default="qwen3.8:27b")
    ap.add_argument("--judge", default="gpt-5.6-luna")
    ap.add_argument("--out-dir", default=str(ROUND3_DIR / "results" / "seeder_8b_eval"))
    a = ap.parse_args()

    out_root = Path(a.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    unload_ollama(a.mentor)

    # PHASE A: All PyTorch seed generation + IACF-v2 runs FIRST (while Ollama VRAM is 0!)
    generate_hf_seeds_for_model(a.sft_model, out_root, "seeder8b_sft", topics=("dental", "neuroevolution"), run_7prompt=True)
    generate_hf_seeds_for_model(a.grpo_model, out_root, "seeder8b_grpo", topics=("dental", "neuroevolution"), run_7prompt=True)

    # Stack 1-round IACF-v2 on top of GRPO Seeder for dental & neuroevolution
    run_iacf_on_seeder(a.grpo_model, "dental", out_root / "iacf_seeder8b_dental")
    run_iacf_on_seeder(a.grpo_model, "neuroevolution", out_root / "iacf_seeder8b_neuroevolution")

    # Ensure GPU VRAM is 100% free before loading local qwen3.8:27b in Ollama for mentoring!
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2.0)

    # PHASE B: Local qwen3.8:27b Mentoring + gpt-5.6-luna Judging (apples-to-apples across ALL arms!)
    eval_arms = [
        # (arm_label, topic, seeds_jsonl_path, out_subdir)
        ("Qwen3-4B Base R0 (ref)",          "dental",         ROUND3_DIR / "results/iacf_qwen4b_goldilocks/round_0/seeds.jsonl", out_root / "ref_qwen4b_base_dental"),
        ("Qwen3-4B Goldilocks (ref)",       "dental",         ROUND3_DIR / "results/iacf_qwen4b_goldilocks/winner/seeds.jsonl",  out_root / "ref_qwen4b_gold_dental"),
        ("Qwen3-8B Instruct Base (ref)",    "dental",         ROUND3_DIR / "results/iacf_qwen8b_goldilocks/round_0/seeds.jsonl", out_root / "ref_qwen8b_base_dental"),
        ("8B-Base Seeder SFT (0-shot)",     "dental",         out_root / "seeder8b_sft_dental/seeds.jsonl",                      out_root / "seeder8b_sft_dental"),
        ("8B-Base Seeder GRPO (0-shot)",    "dental",         out_root / "seeder8b_grpo_dental/seeds.jsonl",                     out_root / "seeder8b_grpo_dental"),
        ("8B-Base Seeder GRPO + IACF-v2",   "dental",         out_root / "iacf_seeder8b_dental/winner/seeds.jsonl",              out_root / "iacf_seeder8b_dental/winner"),
        ("Qwen3-4B Base R0 (ref)",          "neuroevolution", ROUND3_DIR / "results/iacf_qwen4b_neuroevo/round_0/seeds.jsonl",   out_root / "ref_qwen4b_base_neuroevo"),
        ("Qwen3-4B IACF-v2 (ref)",          "neuroevolution", ROUND3_DIR / "results/iacf_qwen4b_neuroevo/winner/seeds.jsonl",    out_root / "ref_qwen4b_iacf_neuroevo"),
        ("8B-Base Seeder SFT (0-shot)",     "neuroevolution", out_root / "seeder8b_sft_neuroevolution/seeds.jsonl",              out_root / "seeder8b_sft_neuroevolution"),
        ("8B-Base Seeder GRPO (0-shot)",    "neuroevolution", out_root / "seeder8b_grpo_neuroevolution/seeds.jsonl",             out_root / "seeder8b_grpo_neuroevolution"),
        ("8B-Base Seeder GRPO + IACF-v2",   "neuroevolution", out_root / "iacf_seeder8b_neuroevolution/winner/seeds.jsonl",      out_root / "iacf_seeder8b_neuroevolution/winner"),
    ]

    summary_rows = []
    for label, topic, seeds_path, odir in eval_arms:
        if not seeds_path.exists():
            print(f"  [skip] missing {seeds_path}", flush=True)
            continue
        odir.mkdir(parents=True, exist_ok=True)
        seeds = [json.loads(l) for l in open(seeds_path) if l.strip()]
        n = len(seeds)
        print(f"\n=== Eval [{topic}] {label} (n={n}) ===", flush=True)
        js_raw = judge_items(seeds, topic, odir / "seeds_judge.jsonl", with_faithfulness=False, judge_model=a.judge)
        print(summarize_scores(js_raw, f"{label[:22]} RAW"), flush=True)

        desc = TOPICS[topic]["description"]
        arm_rec = {
            "label": label, "topic": topic, "n": n,
            "dogma_core": round(dogma_rate(seeds, topic)["core_rate"], 3),
            "raw_nov": round(mean(r["novelty"] for r in js_raw), 2),
        }
        for mkey, tmpl in [("grnd", MENTOR_GROUNDED), ("paper", MENTOR_PAPER)]:
            ref = mentor_local_ollama(seeds, desc, odir / f"ref_q38_{mkey}_{n}.jsonl", tmpl, mentor_model=a.mentor)
            jf = judge_items(ref, topic, odir / f"judge_q38_{mkey}_{n}_faith.jsonl", with_faithfulness=True, judge_model=a.judge)
            jnf = judge_items(ref, topic, odir / f"judge_q38_{mkey}_{n}_nofaith.jsonl", with_faithfulness=False, judge_model=a.judge)
            print(summarize_scores(jf,  f"{label[:20]} {mkey.upper()} w/F"), flush=True)
            print(summarize_scores(jnf, f"{label[:20]} {mkey.upper()} noF"), flush=True)
            arm_rec[f"{mkey}_wf_nov"] = round(mean(r["novelty"] for r in jf), 2)
            arm_rec[f"{mkey}_wf_util"] = round(mean(r["utility"] for r in jf), 2)
            arm_rec[f"{mkey}_wf_feas"] = round(mean(r["feasibility"] for r in jf), 2)
            arm_rec[f"{mkey}_wf_faith"] = round(mean(r["faithfulness"] for r in jf), 2)
            arm_rec[f"{mkey}_wf_hit"] = round(100 * mean(r["novelty"] >= 7 and r["utility"] >= 6 for r in jf), 1)
            arm_rec[f"{mkey}_wf_n8"] = f"{sum(1 for r in jf if r['novelty'] >= 8)}/{len(jf)}"
            arm_rec[f"{mkey}_wf_duds"] = f"{sum(1 for r in jf if r['novelty'] <= 3)}/{len(jf)}"
            arm_rec[f"{mkey}_nof_nov"] = round(mean(r["novelty"] for r in jnf), 2)
            arm_rec[f"{mkey}_nof_n8"] = f"{sum(1 for r in jnf if r['novelty'] >= 8)}/{len(jnf)}"
        summary_rows.append(arm_rec)
        json.dump(summary_rows, open(out_root / "summary_single_topic.json", "w"), indent=2)

    # PHASE C: 7-Prompt Multi-Domain Benchmark (Grounded qwen3.8:27b mentor + gpt-5.6-luna judge)
    p7_summary = []
    # Also include existing Qwen3-4B baseline seeds from MoSN benchmark for exact paired comparison!
    base4b_7p_seeds = []
    bdir = ROUND3_DIR / "results/mosn_narrow_Qwen3-4B-Instruct-2507/benchmark"
    for pi, (pid, _, _, q) in enumerate(BENCH_PROMPTS):
        bf = bdir / f"{pid}_base.jsonl"
        if bf.exists():
            for si, l in enumerate(open(bf)):
                r = json.loads(l)
                base4b_7p_seeds.append({"prompt": pi, "pid": pid, "seed": si, "text": r["seed_text"]})
    if base4b_7p_seeds:
        p7_ref_dir = out_root / "ref_qwen4b_base_7prompt"
        p7_ref_dir.mkdir(parents=True, exist_ok=True)
        (p7_ref_dir / "seeds.jsonl").write_text("".join(json.dumps(r) + "\n" for r in base4b_7p_seeds))

    for tag, label in [
        ("ref_qwen4b_base_7prompt", "Qwen3-4B Base (7-prompt ref)"),
        ("seeder8b_sft_7prompt",    "8B-Base Seeder SFT (0-shot)"),
        ("seeder8b_grpo_7prompt",   "8B-Base Seeder GRPO (0-shot)"),
    ]:
        pdir = out_root / tag
        if not (pdir / "seeds.jsonl").exists():
            continue
        seeds = [json.loads(l) for l in open(pdir / "seeds.jsonl") if l.strip()]
        print(f"\n=== 7-Prompt Multi-Domain Eval: {label} (n={len(seeds)}) ===", flush=True)
        ref = mentor_local_ollama(seeds, "this domain", pdir / "ref_q38_grnd_56.jsonl", MENTOR_GROUNDED, mentor_model=a.mentor)
        jf = judge_7prompt(ref, pdir / "judge_q38_grnd_56_faith.jsonl", judge_model=a.judge)
        print(summarize_scores(jf, label[:28]), flush=True)
        per_p = {}
        for pi, (pid, _, _, _) in enumerate(BENCH_PROMPTS):
            sub = [r for r in jf if r["prompt"] == pi]
            if sub:
                per_p[pid] = {
                    "nov": round(mean(r["novelty"] for r in sub), 2),
                    "util": round(mean(r["utility"] for r in sub), 2),
                    "n8": sum(1 for r in sub if r["novelty"] >= 8),
                }
        p7_summary.append({
            "label": label,
            "n": len(jf),
            "nov": round(mean(r["novelty"] for r in jf), 2),
            "util": round(mean(r["utility"] for r in jf), 2),
            "feas": round(mean(r["feasibility"] for r in jf), 2),
            "faith": round(mean(r["faithfulness"] for r in jf), 2),
            "hit": round(100 * mean(r["novelty"] >= 7 and r["utility"] >= 6 for r in jf), 1),
            "n8": f"{sum(1 for r in jf if r['novelty'] >= 8)}/{len(jf)}",
            "duds": f"{sum(1 for r in jf if r['novelty'] <= 3)}/{len(jf)}",
            "per_prompt": per_p,
        })
        json.dump(p7_summary, open(out_root / "summary_7prompt.json", "w"), indent=2)

    unload_ollama(a.mentor)
    print("\n================ FINAL 8B-BASE SEEDER BENCHMARK SUMMARY ================", flush=True)
    for r in summary_rows:
        print(
            f"  [{r['topic'][:6]}] {r['label']:32s} | raw={r['raw_nov']:.2f} | "
            f"GRND w/F={r['grnd_wf_nov']:.2f} (N>=8:{r['grnd_wf_n8']}, duds:{r['grnd_wf_duds']}) noF={r['grnd_nof_nov']:.2f} | "
            f"PAPER w/F={r['paper_wf_nov']:.2f} (N>=8:{r['paper_wf_n8']})",
            flush=True,
        )
    print("\n--- 7-Prompt Multi-Domain (n=56, qwen3.8:27b Grounded Mentor) ---", flush=True)
    for r in p7_summary:
        print(
            f"  {r['label']:32s} | N={r['nov']:.2f} U={r['util']:.2f} F={r['feas']:.2f} "
            f"hit={r['hit']:4.0f}% N>=8:{r['n8']} duds:{r['duds']}",
            flush=True,
        )
    print("STAGE4_EVAL_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
