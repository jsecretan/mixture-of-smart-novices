"""Turnkey Zero-A-Priori-Topic Creative Ideation Service (`seeder_service.py`).

Uses ONE warm `Qwen3-8B-Base` First-Principles Seeder checkpoint in VRAM — ZERO per-topic
forgetting, ZERO weight swapping, works instantaneously on ANY arbitrary domain prompt!

Pipeline:
  1. Oversample `n * 2` raw first-principles seeds from the warm `8B-Base` Seeder (~4-8s),
     filtering refusals/stalling via `is_bad_seed()`.
  2. Greedy farthest-point semantic deduplication (`Qwen3-Embedding-0.6B`) picks the `n` most
     mutually diverse seeds (eliminating any repeated metaphor across returned proposals!).
  3. Refines each seed faithfully via pluggable mentor (`deepseek-v4-pro`, `gemini-3.1-pro-preview`,
     or 100% local Ollama `qwen3.8:27b` / `qwen3.5:27b` for $0!).
  4. Optionally scores & ranks with the Appendix E rubric (`gpt-5.6-luna` or local Ollama).

Examples:
  # Single prompt (zero a priori topic needed!):
  python seeder/seeder_service.py --prompt "How can we stop urban heat islands on dark rooftops?" --n 8 --rank

  # Interactive warm-VRAM server loop (instant ~5s seed gen on any prompt!):
  python seeder/seeder_service.py --interactive --mentor deepseek-v4-pro --n 6 --rank
"""
import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

SEEDER_DIR = Path(__file__).resolve().parent
ROUND3_DIR = SEEDER_DIR.parent
sys.path.insert(0, str(ROUND3_DIR))

from eval_harness import (  # noqa: E402
    FAITHFULNESS_RUBRIC,
    MENTOR_PAPER,
    RUBRIC_CORE,
    ask_llm,
    summarize_scores,
)
from naive_service import MENTOR_GROUNDED, is_bad_seed, slugify  # noqa: E402
from runner import generate_batch  # noqa: E402


def default_seeder_path():
    # Default to v1 GRPO (our strongest multi-domain + neuroevolution Seeder champion!)
    for p in [
        "/home/jimmy/code/creative-toolkit/models/seeder_8b/grpo_merged",
        "/home/jimmy/code/creative-toolkit/models/seeder_8b_v2/grpo_merged",
    ]:
        if (Path(p) / "config.json").exists():
            return p
    return "/home/jimmy/code/creative-toolkit/models/seeder_8b/grpo_merged"


class FirstPrinciplesSeederEngine:
    """Warm VRAM First-Principles Seeder + embedding diversity selector."""

    def __init__(self, model_path=None, use_dedup=True):
        self.model_path = model_path or default_seeder_path()
        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(self.model_path)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, dtype=torch.bfloat16, device_map="auto", max_memory={0: "11GiB", 1: "11GiB"}
        ).eval()
        self.emb_tok, self.emb_model = None, None
        if use_dedup:
            self.emb_dev = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
            self.emb_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
            self.emb_model = AutoModel.from_pretrained(
                "Qwen/Qwen3-Embedding-0.6B", dtype=torch.bfloat16
            ).to(self.emb_dev).eval()
        print(f"[seeder-svc] Loaded {self.model_path} warm in VRAM ({time.time()-t0:.1f}s)", flush=True)

    @torch.inference_mode()
    def _embed(self, texts):
        enc = self.emb_tok(texts, padding=True, truncation=True, max_length=320, return_tensors="pt").to(self.emb_dev)
        out = self.emb_model(**enc).last_hidden_state
        lens = enc["attention_mask"].sum(dim=1) - 1
        v = out[torch.arange(len(texts), device=self.emb_dev), lens].float()
        return (v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)).cpu().numpy()

    def sample_diverse_seeds(self, prompt, n=8, oversample=2, temperature=0.90, base_seed=0):
        target_pool = max(n, int(n * oversample))
        pool, attempts, sc = [], 0, base_seed
        while len(pool) < target_pool and attempts < target_pool * 3:
            need = min(8, target_pool - len(pool))
            outs = generate_batch(self.model, self.tok, prompt, n=need, max_new=450, temperature=temperature, seed=sc)
            sc += need
            attempts += need
            for txt in outs:
                if not is_bad_seed(txt):
                    pool.append(txt)
        if len(pool) <= n or self.emb_model is None:
            chosen = pool[:n]
        else:
            # Greedy farthest-point selection in embedding space for maximum concept diversity!
            vecs = self._embed(pool)
            # Start from the seed farthest from the pool centroid (most surprising!)
            centroid = vecs.mean(axis=0)
            centroid /= max(1e-8, float(np.linalg.norm(centroid)))
            selected = [int(np.argmin(vecs @ centroid))]
            while len(selected) < n:
                sel_mat = vecs[selected]
                max_sim = (vecs @ sel_mat.T).max(axis=1)
                for si in selected:
                    max_sim[si] = 2.0
                selected.append(int(np.argmin(max_sim)))
            chosen = [pool[i] for i in selected]
        return [{"prompt": 0, "seed": i, "text": t} for i, t in enumerate(chosen)]


def run_ideation(engine, prompt, n=8, oversample=2, mentor="deepseek-v4-pro", mentor_mode="grounded", judge="gpt-5.6-luna", rank=True, offload_for_local_mentor=False):
    t0 = time.time()
    seeds = engine.sample_diverse_seeds(prompt, n=n, oversample=oversample)
    print(f"[seeder-svc] Generated {len(seeds)} maximally diverse seeds in {time.time()-t0:.1f}s", flush=True)

    # If mentoring locally via Ollama in single-prompt mode, free PyTorch VRAM so Ollama runs at full GPU speed!
    if offload_for_local_mentor and ":" in mentor:
        import gc
        del engine.model, engine.emb_model
        engine.model, engine.emb_model = None, None
        gc.collect()
        torch.cuda.empty_cache()

    tmpl = MENTOR_GROUNDED if mentor_mode == "grounded" else MENTOR_PAPER
    workers = 1 if ":" in mentor else 6  # sequential for local Ollama, parallel for API
    print(f"[seeder-svc] Refining with {mentor} ({mentor_mode}, workers={workers})...", flush=True)
    refined, lock = [], threading.Lock()
    def one(r):
        txt = ask_llm(mentor, tmpl.format(topic_description="this domain", seed_text=r["text"]), max_tokens=1600)
        if txt:
            with lock:
                refined.append({"prompt": 0, "seed": r["seed"], "seed_text": r["text"], "text": txt, "mentor": mentor})
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, seeds))
    refined.sort(key=lambda x: x["seed"])

    if rank and refined:
        scored = []
        j_workers = 1 if ":" in judge else 8
        def score_one(r):
            body = (
                RUBRIC_CORE + FAITHFULNESS_RUBRIC
                + f"\n\nRequest: {prompt}\n\nSeed Idea:\n{r['seed_text'][:5000]}\n\nRefined Idea:\n{r['text'][:15000]}\n\n"
                + 'Answer with JSON only: {"novelty": n, "utility": n, "feasibility": n, "faithfulness": n}'
            )
            raw = ask_llm(judge, body, json_mode=True)
            if raw:
                try:
                    sc = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
                    row = {**r, **{k: float(sc[k]) for k in ("novelty", "utility", "feasibility", "faithfulness")}}
                    with lock:
                        scored.append(row)
                except Exception:
                    pass
        with ThreadPoolExecutor(j_workers) as ex:
            list(ex.map(score_one, refined))
        scored.sort(key=lambda x: (x["novelty"] + x["utility"] + 0.5 * x["feasibility"]), reverse=True)
        print(summarize_scores(scored, "First-Principles Seeder"), flush=True)
        refined = scored

    print(f"[seeder-svc] Total latency: {time.time()-t0:.1f}s", flush=True)
    return refined


def main():
    ap = argparse.ArgumentParser(description="Real-Time Zero-A-Priori-Topic First-Principles Seeder Service")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--model", default=None, help="Path to Seeder checkpoint (auto-picks v2 if present, else v1)")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--oversample", type=float, default=2.0)
    ap.add_argument("--mentor", default="deepseek-v4-pro")
    ap.add_argument("--mentor-mode", choices=["grounded", "paper"], default="grounded")
    ap.add_argument("--judge", default="gpt-5.6-luna")
    ap.add_argument("--rank", action="store_true")
    ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    engine = FirstPrinciplesSeederEngine(a.model, use_dedup=not a.no_dedup)
    if a.interactive:
        print("\n=== First-Principles Seeder Interactive Service (type any prompt, or 'quit') ===", flush=True)
        while True:
            try:
                q = input("\nPrompt> ").strip()
            except EOFError:
                break
            if not q or q.lower() in ("quit", "exit"):
                break
            res = run_ideation(engine, q, n=a.n, oversample=a.oversample, mentor=a.mentor, mentor_mode=a.mentor_mode, judge=a.judge, rank=a.rank)
            for i, r in enumerate(res[:3]):
                sc = f" [N={r['novelty']:.0f} U={r['utility']:.0f} F={r['feasibility']:.0f} Faith={r['faithfulness']:.0f}]" if "novelty" in r else ""
                print(f"\n--- TOP #{i+1}{sc} ---\n{r['text'][:700]}", flush=True)
        return

    if not a.prompt:
        ap.error("Provide --prompt or --interactive")
    res = run_ideation(engine, a.prompt, n=a.n, oversample=a.oversample, mentor=a.mentor, mentor_mode=a.mentor_mode, judge=a.judge, rank=a.rank, offload_for_local_mentor=True)
    out_path = Path(a.out) if a.out else (ROUND3_DIR / "service_outputs" / f"seeder_{slugify(a.prompt)[:32]}_{int(time.time())}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(json.dumps(r) + "\n" for r in res))
    print(f"[seeder-svc] Saved {len(res)} proposals -> {out_path}\n", flush=True)
    for i, r in enumerate(res[:3]):
        sc = f" [N={r['novelty']:.0f} U={r['utility']:.0f} F={r['feasibility']:.0f} Faith={r['faithfulness']:.0f}]" if "novelty" in r else ""
        print(f"=== TOP #{i+1}{sc} ===\nSEED: {r['seed_text'][:240].replace(chr(10), ' ')}\nREF : {r['text'][:650]}\n", flush=True)


if __name__ == "__main__":
    main()
