"""Real-Time Growing Mixture of Smart Novices (MoSN v2) Router & Ideation Service!

Supports TWO pool modes:
  1. `--pool narrow` (DEFAULT — Recommended):
     Routes any user prompt in <50ms (`Qwen3-Embedding-0.6B`) against a growing NVMe registry of
     razor-sharp single-topic Smart Novices (`../models/mosn_narrow/registry_<model>.json`).
     - Cache hit (`sim >= --threshold`, default 0.48): ~1.5s NVMe weight swap into the warm base
       model! Zero training wait!
     - Cache miss (`sim < --threshold`) + `--auto-grow`: automatically discovers the optimal
       narrow forgetting plan, builds a fresh 2-round IACF-v2 novice in ~6 min on the warm VRAM
       model, caches it on NVMe, permanently registers it in `registry.json`, and serves the query.
       Every subsequent prompt in that domain becomes an instant ~1.5s cache hit forever!
  2. `--pool broad` (Legacy 4-swath macro-novices: `bio_med`, `cs_ai`, `mech_home`, `org_econ`).

Examples:
  # Single prompt against Growing Narrow-Topic MoSN (instant on known domains, auto-grows on new ones!):
  python mosn/mosn_service.py --prompt "Can you help me invent something to clean between my teeth?" --n 12 --auto-grow --rank

  # Interactive real-time server loop (keeps base model warm in VRAM, swaps novices in ~1.5s!):
  python mosn/mosn_service.py --interactive --auto-grow --n 6 --rank
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

MOSN_DIR = Path(__file__).resolve().parent
ROUND3_DIR = MOSN_DIR.parent
sys.path.insert(0, str(ROUND3_DIR))

from eval_harness import (  # noqa: E402
    FAITHFULNESS_RUBRIC,
    MENTOR_PAPER,
    RUBRIC_CORE,
    ask_llm,
    summarize_scores,
)
from iacf_unlearn import select_ssd_params  # noqa: E402
from mosn.build_novices import get_weights_root  # noqa: E402
from mosn.domains import NOVICE_SWATHS, SWATH_ORDER  # noqa: E402
from mosn.narrow_pool import NarrowMoSNPool, NarrowMoSNRouter, load_registry  # noqa: E402
from naive_service import MENTOR_GROUNDED, is_bad_seed, slugify  # noqa: E402
from runner import generate_batch  # noqa: E402


class MoSNRouter:
    """Legacy 4-swath router over broad macro-swaths (`--pool broad`)."""

    def __init__(self, mode="embed", llm_model="gpt-6-luna", device="cuda:0"):
        self.mode = mode
        self.llm_model = llm_model
        self.device = device
        self.swath_vecs = None
        if mode == "embed":
            self._init_embeddings()

    def _init_embeddings(self):
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
        emb_m = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B", dtype=torch.bfloat16).to(self.device).eval()
        texts = [
            f"{NOVICE_SWATHS[sid]['name']}: {NOVICE_SWATHS[sid]['description']}. {NOVICE_SWATHS[sid]['routing_summary']}"
            for sid in SWATH_ORDER
        ]
        with torch.inference_mode():
            enc = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
            out = emb_m(**enc).last_hidden_state
            lens = enc["attention_mask"].sum(dim=1) - 1
            vecs = out[torch.arange(len(texts), device=self.device), lens].float()
            vecs = vecs / vecs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            self.swath_vecs = vecs.cpu().numpy()
        self.emb_tok = tok
        self.emb_model = emb_m

    def route(self, prompt, top_k=1, threshold=0.0):
        if self.mode == "embed":
            with torch.inference_mode():
                enc = self.emb_tok([prompt], padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
                out = self.emb_model(**enc).last_hidden_state
                lens = enc["attention_mask"].sum(dim=1) - 1
                v = out[0, lens[0]].float()
                v = (v / v.norm().clamp(min=1e-8)).cpu().numpy()
            sims = self.swath_vecs @ v
            order = np.argsort(-sims)
            ranked = [(SWATH_ORDER[i], float(sims[i])) for i in order]
            return ranked[:top_k], ranked, True
        else:
            desc = "\n".join(
                f"- {sid}: forgets {NOVICE_SWATHS[sid]['routing_summary']}" for sid in SWATH_ORDER
            )
            q = (
                f"Pick the novice that knows the LEAST about the domain of this user prompt "
                f"(i.e., whose forgotten area best covers the prompt's entrenched solutions):\n"
                f"Prompt: {prompt}\n\nNovices:\n{desc}\n\n"
                f'Answer with JSON only: {{"ranked": ["best_id", "second_id", ...], "reason": "short reason"}}'
            )
            raw = ask_llm(self.llm_model, q, json_mode=True)
            parsed = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
            ids = [x for x in parsed.get("ranked", []) if x in NOVICE_SWATHS]
            for sid in SWATH_ORDER:
                if sid not in ids:
                    ids.append(sid)
            ranked = [(sid, 1.0 - 0.1 * i) for i, sid in enumerate(ids)]
            return ranked[:top_k], ranked, True


class MoSNPool:
    """Legacy 4-swath warm VRAM pool (`--pool broad`)."""

    def __init__(self, model_id="Qwen/Qwen3-4B-Instruct-2507"):
        self.model_id = model_id
        self.model_tag = model_id.split("/")[-1]
        self.weights_root = get_weights_root()
        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="auto", max_memory={0: "10GiB", 1: "10GiB"}
        ).eval()
        num_layers = len(self.model.model.layers)
        l_start, l_end = int(num_layers * 0.12), int(num_layers * 0.92)
        self.ssd_params = select_ssd_params(self.model, l_start, l_end, include_attn=True)
        self.orig_sd = {n: p.detach().cpu().clone() for n, p in self.ssd_params}
        self.active = "baseline"
        print(f"[mosn] Base model {model_id} warm in VRAM ({time.time()-t0:.1f}s)", flush=True)

    def activate(self, novice_id):
        if novice_id == self.active:
            return 0.0
        t0 = time.time()
        if novice_id == "baseline":
            sd = self.orig_sd
        else:
            w_file = self.weights_root / f"{novice_id}_{self.model_tag}" / "iacf_weights.pt"
            if not w_file.exists():
                raise FileNotFoundError(f"Novice weights not found: {w_file}. Run mosn/build_novices.py first!")
            sd = torch.load(w_file, map_location="cpu", weights_only=True)
        with torch.no_grad():
            for n, p in self.ssd_params:
                src = sd.get(n, self.orig_sd[n])
                p.copy_(src.to(device=p.device, dtype=p.dtype))
        self.active = novice_id
        return time.time() - t0

    def generate_seeds(self, prompt, n=12, temperature=0.85, base_seed=0):
        seeds, attempts, sc = [], 0, base_seed
        while len(seeds) < n and attempts < n * 5:
            need = min(6, n - len(seeds))
            out = generate_batch(self.model, self.tok, prompt, n=need, max_new=650, temperature=temperature, seed=sc)
            sc += need
            attempts += need
            for txt in out:
                if not is_bad_seed(txt):
                    seeds.append({"prompt": 0, "seed": len(seeds), "text": txt, "novice": self.active})
        return seeds


def _describe_novice(pool, sid):
    if isinstance(pool, NarrowMoSNPool):
        reg = load_registry(pool.model_tag).get("novices", {})
        if sid in reg:
            return reg[sid].get("description") or reg[sid].get("name") or sid
    if sid in NOVICE_SWATHS:
        return NOVICE_SWATHS[sid]["description"]
    return "this domain"


def run_query(
    pool,
    router,
    prompt,
    n=12,
    mix_k=1,
    mentor="deepseek-v4-pro",
    mentor_mode="grounded",
    judge="gpt-5.6-luna",
    rank=True,
    force_novice=None,
    auto_grow=False,
    threshold=0.52,
    planner_model="gpt-6-luna",
):
    t0 = time.time()
    grew_new = False
    if force_novice:
        chosen = [(force_novice, 1.0)]
        all_scores = chosen
        is_hit = True
    else:
        res = router.route(prompt, top_k=mix_k, threshold=threshold)
        if len(res) == 3:
            chosen, all_scores, is_hit = res
        else:
            chosen, all_scores = res
            is_hit = True

        if (not is_hit or not chosen) and auto_grow and isinstance(pool, NarrowMoSNPool):
            top_str = f"{all_scores[0][0]}={all_scores[0][1]:.3f}" if all_scores else "empty pool"
            print(f"[mosn] Confidence gate MISS ({top_str} < {threshold:.2f}) -> triggering --auto-grow!", flush=True)
            new_id = pool.grow_novice(prompt, router=router, planner_model=planner_model)
            chosen = [(new_id, 1.0)]
            all_scores = [(new_id, 1.0)] + all_scores
            grew_new = True
        elif not chosen:
            chosen = [("baseline", 0.0)]

    route_str = ", ".join(f"{sid}={sc:.3f}" for sid, sc in all_scores[:6])
    status = "AUTO-GREW" if grew_new else ("HIT" if is_hit else "FALLBACK")
    print(f"[mosn] Routed [{status}] in {time.time()-t0:.2f}s -> {[c[0] for c in chosen]}  (scores: {route_str})", flush=True)

    seeds = []
    per_novice = [n // len(chosen) + (1 if i < n % len(chosen) else 0) for i in range(len(chosen))]
    for (sid, _), k in zip(chosen, per_novice):
        swap_s = pool.activate(sid)
        print(f"[mosn] Activated novice '{sid}' (swap={swap_s:.2f}s), generating {k} seeds...", flush=True)
        t_g = time.time()
        sub = pool.generate_seeds(prompt, n=k, base_seed=len(seeds) * 10)
        for r in sub:
            r["seed"] = len(seeds)
            seeds.append(r)
        print(f"[mosn] Generated {len(sub)} clean seeds from '{sid}' in {time.time()-t_g:.1f}s", flush=True)

    primary_sid = chosen[0][0]
    topic_desc = _describe_novice(pool, primary_sid)
    tmpl = MENTOR_GROUNDED if mentor_mode == "grounded" else MENTOR_PAPER

    print(f"[mosn] Mentoring {len(seeds)} seeds with {mentor} ({mentor_mode})...", flush=True)
    refined, lock = [], threading.Lock()
    def one(r):
        txt = ask_llm(mentor, tmpl.format(topic_description=topic_desc, seed_text=r["text"]))
        if txt:
            with lock:
                refined.append({"prompt": 0, "seed": r["seed"], "novice": r["novice"], "seed_text": r["text"], "text": txt})
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(one, seeds))
    refined.sort(key=lambda x: x["seed"])

    if rank and refined:
        scored = []
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
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(score_one, refined))
        scored.sort(key=lambda x: (x["novelty"] + x["utility"] + 0.5 * x["feasibility"]), reverse=True)
        print(summarize_scores(scored, f"MoSN ({'+'.join(c[0] for c in chosen)})"), flush=True)
        refined = scored

    print(f"[mosn] Total query latency: {time.time()-t0:.1f}s", flush=True)
    return refined, chosen


def main():
    ap = argparse.ArgumentParser(description="Real-Time Growing Mixture of Smart Novices (MoSN v2) Service")
    ap.add_argument("--prompt", default=None, help="Invention prompt")
    ap.add_argument("--interactive", action="store_true", help="Keep model warm in VRAM for instant interactive prompts")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--pool", choices=["narrow", "broad"], default="narrow", help="narrow (growing single-topic pool) or broad (legacy 4-swath)")
    ap.add_argument("--router", choices=["embed", "llm"], default="embed")
    ap.add_argument("--auto-grow", action="store_true", help="Automatically build & register a new narrow novice on cache miss")
    ap.add_argument("--threshold", type=float, default=0.52, help="Cosine similarity threshold for narrow-pool cache hit")
    ap.add_argument("--planner", default="gpt-6-luna", help="Planner LLM for --auto-grow")
    ap.add_argument("--novice", default=None, help="Force a specific novice ID or 'baseline'")
    ap.add_argument("--mix-k", type=int, default=1, help="Number of top novices to draw seeds from (default 1)")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--mentor", default="deepseek-v4-pro")
    ap.add_argument("--mentor-mode", choices=["grounded", "paper"], default="grounded")
    ap.add_argument("--judge", default="gpt-5.6-luna")
    ap.add_argument("--rank", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    model_tag = a.model.split("/")[-1]
    if a.pool == "narrow":
        pool = NarrowMoSNPool(a.model)
        router = NarrowMoSNRouter(model_tag=model_tag, mode=a.router)
        print(f"[mosn-narrow] Registered novices ({len(router.ids)}): {router.ids}", flush=True)
    else:
        pool = MoSNPool(a.model)
        router = MoSNRouter(mode=a.router)

    if a.interactive:
        print(f"\n=== MoSN ({a.pool}, auto_grow={a.auto_grow}) Interactive Mode (type a prompt, or 'quit') ===", flush=True)
        while True:
            try:
                q = input("\nPrompt> ").strip()
            except EOFError:
                break
            if not q or q.lower() in ("quit", "exit"):
                break
            res, chosen = run_query(
                pool, router, q, n=a.n, mix_k=a.mix_k, mentor=a.mentor, mentor_mode=a.mentor_mode,
                judge=a.judge, rank=a.rank, force_novice=a.novice, auto_grow=a.auto_grow,
                threshold=a.threshold, planner_model=a.planner,
            )
            for i, r in enumerate(res[:3]):
                sc = f" [N={r['novelty']:.0f} U={r['utility']:.0f} F={r['feasibility']:.0f}]" if "novelty" in r else ""
                print(f"\n--- TOP #{i+1} ({r['novice']}){sc} ---\n{r['text'][:650]}", flush=True)
        return

    if not a.prompt:
        ap.error("Provide --prompt or --interactive")
    res, chosen = run_query(
        pool, router, a.prompt, n=a.n, mix_k=a.mix_k, mentor=a.mentor, mentor_mode=a.mentor_mode,
        judge=a.judge, rank=a.rank, force_novice=a.novice, auto_grow=a.auto_grow,
        threshold=a.threshold, planner_model=a.planner,
    )
    out_path = Path(a.out) if a.out else (ROUND3_DIR / "service_outputs" / f"mosn_{a.pool}_{chosen[0][0]}_{slugify(a.prompt)[:30]}_{int(time.time())}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(json.dumps(r) + "\n" for r in res))
    print(f"[mosn] Saved {len(res)} proposals -> {out_path}\n", flush=True)
    for i, r in enumerate(res[:3]):
        sc = f" [N={r['novelty']:.0f} U={r['utility']:.0f} F={r['feasibility']:.0f} Faith={r['faithfulness']:.0f}]" if "novelty" in r else ""
        print(f"=== TOP #{i+1} ({r['novice']}){sc} ===\nSEED: {r['seed_text'][:240].replace(chr(10), ' ')}\nREF : {r['text'][:650]}\n", flush=True)


if __name__ == "__main__":
    main()
