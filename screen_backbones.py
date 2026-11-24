"""Fast unedited raw-seed pre-screen of candidate higher-parameter backbones on Appendix C dental prompts.

Generates n=4 seeds per prompt (12 total per model, T=0.85, max_new=650, thinking disabled),
judges raw seeds with GPT-5.6 Luna (Appendix E rubric verbatim), and reports:
  - Raw seed Novelty / Utility / Feasibility
  - Dud rate (N <= 2, e.g., clarifying questions / empty brainstorming outlines)
  - Dogma rate (core_rate / filter_rate)
  - Sample openings so we can immediately see which larger models propose concrete inventions cleanly!
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROUND3_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROUND3_DIR))

from eval_harness import TOPICS, dogma_rate, judge_items, summarize_scores  # noqa: E402
from runner import generate_batch  # noqa: E402

OUT_DIR = ROUND3_DIR / "results" / "backbone_screen"
PROMPTS = TOPICS["dental"]["prompts"]


def gen_ollama(model_name, n_per_prompt=4, temperature=0.85, max_new=650):
    rows = []
    t0 = time.time()
    for pi, q in enumerate(PROMPTS):
        for s in range(n_per_prompt):
            body = {
                "model": model_name,
                "messages": [{"role": "user", "content": q}],
                "stream": False,
                "think": False,
                "keep_alive": "1m",
                "options": {
                    "temperature": temperature,
                    "top_p": 0.95,
                    "num_predict": max_new,
                    "seed": 100 * pi + s + 1,
                },
            }
            req = urllib.request.Request(
                "http://localhost:11434/api/chat",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=240) as resp:
                    data = json.load(resp)
                txt = (data.get("message", {}).get("content") or "").strip()
            except Exception as e:
                print(f"  [ollama err] {model_name} p={pi} s={s}: {e}", flush=True)
                txt = ""
            rows.append({"prompt": pi, "seed": s, "text": txt})
    # Unload from VRAM cleanly so next model has full GPUs
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=json.dumps({"model": model_name, "keep_alive": 0}).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=30).read()
    except Exception:
        pass
    print(f"  [ollama:{model_name}] generated {len(rows)} seeds in {time.time()-t0:.1f}s", flush=True)
    return rows


def gen_hf(model_id, n_per_prompt=4, temperature=0.85, max_new=650, load_4bit=False):
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    kw = {}
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto", max_memory={0: "13GiB", 1: "13GiB", "cpu": "70GiB"}, **kw
    ).eval()
    rows = []
    for pi, q in enumerate(PROMPTS):
        outs = generate_batch(model, tok, q, n=n_per_prompt, max_new=max_new, temperature=temperature, seed=100 * pi)
        for s, txt in enumerate(outs):
            rows.append({"prompt": pi, "seed": s, "text": txt.strip()})
    del model, tok
    torch.cuda.empty_cache()
    print(f"  [hf:{model_id}] generated {len(rows)} seeds in {time.time()-t0:.1f}s", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ollama", nargs="*", default=["qwen3.8:27b", "qwen3.5:27b", "qwen3.6:35b-a3b-q8_0", "gemma4:26b", "gpt-oss:20b"])
    ap.add_argument("--hf", nargs="*", default=["Qwen/Qwen3-14B"])
    ap.add_argument("--hf4", nargs="*", default=[])
    ap.add_argument("--n-per-prompt", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = [("ollama", m) for m in a.ollama] + [("hf", m) for m in a.hf] + [("hf4", m) for m in a.hf4]
    summary = []

    for backend, name in targets:
        slug = name.replace("/", "_").replace(":", "_")
        s_file = OUT_DIR / f"{slug}_seeds.jsonl"
        j_file = OUT_DIR / f"{slug}_judge.jsonl"
        print(f"\n=== Screening {backend}:{name} ===", flush=True)
        if s_file.exists() and not a.overwrite:
            rows = [json.loads(l) for l in open(s_file)]
            print(f"  Loaded {len(rows)} cached seeds from {s_file}", flush=True)
        else:
            if backend == "ollama":
                rows = gen_ollama(name, a.n_per_prompt)
            else:
                rows = gen_hf(name, a.n_per_prompt, load_4bit=(backend == "hf4"))
            s_file.write_text("".join(json.dumps(r) + "\n" for r in rows))

        nonempty = [r for r in rows if r["text"].strip()]
        dr = dogma_rate(nonempty) if nonempty else {"core_rate": 0.0, "filter_rate": 0.0}
        if j_file.exists() and not a.overwrite:
            j_file.unlink() if len(j_file.read_text().splitlines()) < len(nonempty) else None
        judged = judge_items(nonempty, "dental", j_file, with_faithfulness=False)
        print(summarize_scores(judged, f"{name} RAW SEEDS"), flush=True)
        duds = sum(1 for r in judged if r["novelty"] <= 2)
        n5 = sum(1 for r in judged if r["novelty"] >= 5)
        n6 = sum(1 for r in judged if r["novelty"] >= 6)
        avg_words = mean(len(r["text"].split()) for r in nonempty) if nonempty else 0
        row = {
            "model": name,
            "backend": backend,
            "n": len(judged),
            "empty": len(rows) - len(nonempty),
            "nov": round(mean(r["novelty"] for r in judged), 2) if judged else 0,
            "util": round(mean(r["utility"] for r in judged), 2) if judged else 0,
            "feas": round(mean(r["feasibility"] for r in judged), 2) if judged else 0,
            "duds": duds,
            "ge5": n5,
            "ge6": n6,
            "dogma_core": round(dr["core_rate"], 2),
            "avg_words": round(avg_words),
            "sample_p0": nonempty[0]["text"][:160].replace("\n", " ") if nonempty else "",
            "sample_p2": next((r["text"][:160].replace("\n", " ") for r in nonempty if r["prompt"] == 2), ""),
        }
        summary.append(row)
        print(f"  duds(N<=2)={duds}/{len(judged)} | N>=5:{n5}/{len(judged)} N>=6:{n6}/{len(judged)} | dogma={100*dr['core_rate']:.0f}% | words={avg_words:.0f}", flush=True)
        print(f"  P0 sample: {row['sample_p0']}", flush=True)
        print(f"  P2 sample: {row['sample_p2']}", flush=True)

    print("\n================ RAW SEED BACKBONE PRE-SCREEN SUMMARY ================", flush=True)
    print(f"{'Model':24s} | {'N':>4s} | {'U':>4s} | {'F':>4s} | {'Duds':>5s} | {'N>=5':>5s} | {'N>=6':>5s} | {'Dogma':>5s} | {'Words':>5s}", flush=True)
    print("-" * 82, flush=True)
    # Include our known 4B and 8B baselines for instant reference
    print(f"{'Qwen3-4B-Instruct-2507*':24s} | 5.72 | 5.39 | 3.89 |  0/18 |   —   |   —   |  100% |   —  ", flush=True)
    print(f"{'Qwen3-8B*':24s} | 3.22 | 4.78 | 4.78 |  4/18 |   —   |   —   |  100% |   —  ", flush=True)
    print(f"{'Olmo-3-7B-Instruct-SFT*':24s} | 2.69 | 3.08 | 3.78 |   —   |   —   |   —   |   86% |   —  ", flush=True)
    for r in summary:
        print(
            f"{r['model']:24s} | {r['nov']:4.2f} | {r['util']:4.2f} | {r['feas']:4.2f} | "
            f"{r['duds']:2d}/{r['n']:<2d} | {r['ge5']:2d}/{r['n']:<2d} | {r['ge6']:2d}/{r['n']:<2d} | "
            f"{100*r['dogma_core']:4.0f}% | {r['avg_words']:5d}",
            flush=True,
        )
    json.dump(summary, open(OUT_DIR / "summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
