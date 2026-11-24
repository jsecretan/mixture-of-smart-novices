"""NoveltyBench (curated 100 prompts) Evaluation for Mixture of Smart Novices (MoSN).

Evaluates unedited baseline vs MoSN (routed top-1 novice) vs MoSN Ensemble (all 4 orthogonal
novices, 3/3/2/2 per prompt) on the exact 100 curated NoveltyBench prompts (10 samples/prompt,
T=1.0, top_p=1.0, top_k=0, max_new=512, enable_thinking=False).

Metrics (100% local, matching fractured-representations/ufo_ft/novelty.py & creative-priming/metrics.py):
  - distinct            : NoveltyBench DeBERTa-v3-large equivalence clusters (threshold 0.102, 128-tok window)
  - quality             : Skywork-Reward-V2-Qwen3-1.7B mean scalar score (directly comparable to prior runs)
  - quality_of_distinct : Mean Skywork score of unique cluster representatives
  - good_distinct       : Distinct clusters whose best member has quality >= prompt baseline mean
  - vendi               : Embedding Vendi score (Qwen3-Embedding-0.6B, 1..10)
  - filt_vendi          : Vendi score restricted to generations with quality >= prompt baseline mean
  - diversity           : 100 * mean pairwise cosine distance among the 10 generations
  - deviation           : 100 * mean min cosine distance to the unedited baseline's 10 generations

Usage:
  python mosn/eval_noveltybench.py gen   --model Qwen/Qwen3-8B --modes base routed ensemble
  python mosn/eval_noveltybench.py score --arms qwen3_4b_base qwen3_4b_routed qwen3_4b_ensemble \
                                               qwen3_8b_base qwen3_8b_routed qwen3_8b_ensemble
"""
import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

MOSN_DIR = Path(__file__).resolve().parent
ROUND3_DIR = MOSN_DIR.parent
ROOT = ROUND3_DIR.parent
sys.path.insert(0, str(ROUND3_DIR))

from mosn.domains import SWATH_ORDER  # noqa: E402
from mosn.mosn_service import MoSNPool, MoSNRouter  # noqa: E402

PROMPTS_PATH = ROOT / "random-lora" / "data" / "curated.jsonl"
OUT_DIR = ROUND3_DIR / "results" / "noveltybench"
N_GEN = 10
MAX_NEW = 512


def load_prompts(limit=None):
    rows = [json.loads(line) for line in open(PROMPTS_PATH)]
    return rows[:limit] if limit else rows


def model_short_tag(model_id):
    low = model_id.lower()
    if "4b" in low:
        return "qwen3_4b"
    if "8b" in low:
        return "qwen3_8b"
    if "27b" in low:
        return "qwen3_27b"
    return model_id.split("/")[-1].lower().replace("-", "_")


@torch.inference_mode()
def generate_requests(model, tok, prompt_texts, n_list, max_new=MAX_NEW, temperature=1.0, bs=16, seed=0):
    """Expand (prompt_text, count) into flat batched generation with automatic OOM backoff."""
    flat_prompts = []
    flat_meta = []
    for idx, (pt, cnt) in enumerate(zip(prompt_texts, n_list)):
        for k in range(cnt):
            flat_prompts.append(pt)
            flat_meta.append((idx, k))

    # Sort by prompt length for minimal padding waste
    order = sorted(range(len(flat_prompts)), key=lambda i: len(flat_prompts[i]))
    outs = [None] * len(flat_prompts)
    input_dev = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    pos = 0
    cur_bs = bs
    t0 = time.time()
    while pos < len(order):
        chunk_idx = order[pos:pos + cur_bs]
        texts = [flat_prompts[i] for i in chunk_idx]
        try:
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(input_dev)
            gen = model.generate(
                **enc,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=temperature,
                top_p=1.0,
                top_k=0,
                pad_token_id=pad_id,
            )
            dec = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for i, txt in zip(chunk_idx, dec):
                outs[i] = txt.strip()
            pos += len(chunk_idx)
            if pos % max(cur_bs * 4, 64) == 0 or pos == len(order):
                print(f"    generated {pos}/{len(order)} ({time.time()-t0:.1f}s, bs={cur_bs})", flush=True)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if cur_bs <= 1:
                raise
            cur_bs = max(1, cur_bs // 2)
            print(f"    [OOM] reducing batch size -> {cur_bs}", flush=True)

    grouped = [[] for _ in prompt_texts]
    for (idx, _), txt in zip(flat_meta, outs):
        grouped[idx].append(txt)
    return grouped


def run_gen(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(args.limit)
    tag = model_short_tag(args.model)

    print(f"=== NoveltyBench Generation: model={args.model} ({tag}) | modes={args.modes} | n_prompts={len(prompts)} ===", flush=True)
    pool = MoSNPool(args.model)
    pool.model.config.use_cache = True
    router = MoSNRouter(mode="embed", device="cuda:0")

    # Pre-format chat templates (enable_thinking=False, matching ufo_ft/novelty.py)
    chat_texts = [
        pool.tok.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for r in prompts
    ]

    # Route all prompts upfront (<1s total!)
    routes = []
    route_counts = defaultdict(int)
    for r in prompts:
        _, ranked = router.route(r["prompt"], top_k=len(SWATH_ORDER))
        routes.append(ranked)
        route_counts[ranked[0][0]] += 1
    print(f"[nb] Routing distribution across {len(prompts)} prompts: {dict(route_counts)}", flush=True)

    # Free router embedding model from GPU before large batched generation
    if hasattr(router, "emb_model"):
        del router.emb_model
        torch.cuda.empty_cache()

    for mode in args.modes:
        arm_name = f"{tag}_{mode}"
        out_file = OUT_DIR / f"{arm_name}.jsonl"
        if out_file.exists() and not args.overwrite:
            print(f"[nb] {out_file} already exists, skipping (use --overwrite to regenerate).", flush=True)
            continue

        t_m = time.time()
        print(f"\n--- Generating arm: {arm_name} ---", flush=True)
        # Build per-novice job buckets: novice_id -> list of (prompt_idx, count)
        buckets = defaultdict(list)
        if mode == "base":
            for i in range(len(prompts)):
                buckets["baseline"].append((i, N_GEN))
        elif mode == "routed":
            for i, ranked in enumerate(routes):
                buckets[ranked[0][0]].append((i, N_GEN))
        elif mode == "ensemble":
            # 3/3/2/2 ordered by router relevance across all 4 orthogonal novices!
            alloc = [3, 3, 2, 2]
            for i, ranked in enumerate(routes):
                for (sid, _), cnt in zip(ranked[:4], alloc):
                    buckets[sid].append((i, cnt))
        elif mode == "ensemble_base":
            # 2 from baseline + 2 from each of the 4 novices = 10
            for i, ranked in enumerate(routes):
                buckets["baseline"].append((i, 2))
                for sid, _ in ranked[:4]:
                    buckets[sid].append((i, 2))
        else:
            raise ValueError(f"Unknown mode: {mode}")

        per_prompt_gens = [[] for _ in prompts]
        per_prompt_novs = [[] for _ in prompts]

        # Run each novice bucket in ONE contiguous pass (at most 4 weight swaps total!)
        bucket_order = ["baseline"] + [s for s in SWATH_ORDER if s in buckets]
        for b_idx, sid in enumerate(bucket_order):
            jobs = buckets.get(sid, [])
            if not jobs:
                continue
            swap_s = pool.activate(sid)
            tot_req = sum(c for _, c in jobs)
            print(f"  [novice={sid}] swap={swap_s:.2f}s | {len(jobs)} prompts, {tot_req} generations...", flush=True)
            p_texts = [chat_texts[i] for i, _ in jobs]
            n_list = [c for _, c in jobs]
            res = generate_requests(
                pool.model, pool.tok, p_texts, n_list,
                max_new=MAX_NEW, temperature=1.0, bs=args.bs, seed=args.seed + 100 * b_idx,
            )
            for (p_idx, _), texts in zip(jobs, res):
                per_prompt_gens[p_idx].extend(texts)
                per_prompt_novs[p_idx].extend([sid] * len(texts))

        rows = []
        for i, r in enumerate(prompts):
            rows.append({
                "id": r["id"],
                "prompt": r["prompt"],
                "routed_top1": routes[i][0][0],
                "novices": per_prompt_novs[i],
                "generations": per_prompt_gens[i],
            })
        out_file.write_text("".join(json.dumps(row) + "\n" for row in rows))
        print(f"[nb] Saved {arm_name} ({len(rows)} prompts) -> {out_file} in {(time.time()-t_m)/60:.2f} min", flush=True)


@torch.inference_mode()
def compute_deberta_partitions(rows, tok, clf, device, bs=128):
    """Exact NoveltyBench DeBERTa equivalence classifier (batched on GPU)."""
    out_parts = []
    out_distinct = []
    for r in rows:
        G = r["generations"]
        n = len(G)
        toks = [tok.encode(g, truncation=True, max_length=128, add_special_tokens=False) if g else [] for g in G]
        words = [g.strip().lower().split() for g in G]
        E = np.zeros((n, n), dtype=bool)
        pairs = []
        for i in range(n):
            if not G[i]:
                continue
            for j in range(i + 1, n):
                if not G[j]:
                    continue
                if max(len(words[i]), len(words[j])) <= 5:
                    E[i, j] = len(set(words[i]) & set(words[j])) * 2 >= max(len(words[i]), len(words[j]))
                else:
                    pairs.append((i, j))
        pairs.sort(key=lambda ij: len(toks[ij[0]]) + len(toks[ij[1]]))
        for b in range(0, len(pairs), bs):
            chunk = pairs[b:b + bs]
            seqs = [[tok.cls_token_id] + toks[i] + [tok.sep_token_id] + toks[j] + [tok.sep_token_id] for i, j in chunk]
            L = max(len(s) for s in seqs)
            ii = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
            tt = torch.zeros((len(seqs), L), dtype=torch.long)
            am = torch.zeros((len(seqs), L), dtype=torch.long)
            for k, (q, (i, _)) in enumerate(zip(seqs, chunk)):
                ii[k, :len(q)] = torch.tensor(q)
                am[k, :len(q)] = 1
                tt[k, len(toks[i]) + 2:len(q)] = 1
            probs = clf(
                input_ids=ii.to(device),
                attention_mask=am.to(device),
                token_type_ids=tt.to(device),
            ).logits.float().softmax(-1)[:, 1].tolist()
            for (i, j), pr in zip(chunk, probs):
                E[i, j] = pr > 0.102

        part = [-1] * n
        k = 0
        for i in range(n):
            if part[i] != -1 or not G[i]:
                continue
            part[i] = k
            for j in range(i + 1, n):
                if G[j] and part[j] == -1 and E[i, j]:
                    part[j] = k
            k += 1
        out_parts.append(part)
        out_distinct.append(k)
    return out_parts, out_distinct


@torch.inference_mode()
def compute_skywork_scores(rows, tok, model, device, bs=32):
    """Exact Skywork-Reward-V2-Qwen3-1.7B scalar reward scores."""
    flat_texts = []
    flat_map = []
    for p_idx, r in enumerate(rows):
        for g_idx, g in enumerate(r["generations"]):
            txt = tok.apply_chat_template(
                [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": g if g else "(no answer)"}],
                tokenize=False,
            )
            flat_texts.append(txt)
            flat_map.append((p_idx, g_idx))

    order = sorted(range(len(flat_texts)), key=lambda i: len(flat_texts[i]))
    scores = [0.0] * len(flat_texts)
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        enc = tok([flat_texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        vals = model(**enc).logits[:, 0].float().tolist()
        for i, v in zip(idx, vals):
            scores[i] = v

    per_prompt = [[0.0] * len(r["generations"]) for r in rows]
    for (p_idx, g_idx), v in zip(flat_map, scores):
        per_prompt[p_idx][g_idx] = v
    return per_prompt


@torch.inference_mode()
def compute_qwen_embeddings(rows, tok, model, device, bs=64):
    """Normalized last-token embeddings from Qwen3-Embedding-0.6B -> [P, N, D] tensor on CPU."""
    flat = [g if g.strip() else "(empty)" for r in rows for g in r["generations"]]
    order = sorted(range(len(flat)), key=lambda i: len(flat[i]))
    vecs = [None] * len(flat)
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        enc = tok([flat[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        out = model(**enc).last_hidden_state
        lens = enc["attention_mask"].sum(dim=1) - 1
        v = out[torch.arange(len(idx), device=device), lens].float()
        v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        for i, row_v in zip(idx, v.cpu()):
            vecs[i] = row_v
    P = len(rows)
    N = len(rows[0]["generations"])
    return torch.stack(vecs).view(P, N, -1)


def vendi_score(E):
    """Vendi score of L2-normalized embedding matrix E [m, d]."""
    if len(E) == 0:
        return 0.0
    sim = (E @ E.T).clamp(min=0.0, max=1.0).double().numpy()
    w = np.linalg.eigvalsh(sim / len(E))
    w = w[w > 1e-12]
    return float(np.exp(-(w * np.log(w)).sum()))


def mean_ci95(arr):
    x = np.asarray(arr, dtype=float)
    m = float(np.mean(x))
    ci = float(1.96 * np.std(x, ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0
    return m, ci


def run_score(args):
    device = args.device
    arms = args.arms
    loaded = {}
    for arm in arms:
        f = OUT_DIR / f"{arm}.jsonl"
        if not f.exists():
            raise FileNotFoundError(f"Missing generation file: {f}")
        loaded[arm] = [json.loads(line) for line in open(f)]

    # 1. DeBERTa equivalence partitions
    print("[score] Loading DeBERTa equivalence classifier...", flush=True)
    dtok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    dclf = AutoModelForSequenceClassification.from_pretrained(
        "yimingzhang/deberta-v3-large-generation-similarity"
    ).to(device).eval().half()
    parts, distincts = {}, {}
    for arm in arms:
        t0 = time.time()
        parts[arm], distincts[arm] = compute_deberta_partitions(loaded[arm], dtok, dclf, device)
        print(f"  [DeBERTa] {arm:22s}: distinct={np.mean(distincts[arm]):.2f} ({time.time()-t0:.1f}s)", flush=True)
    del dclf, dtok
    torch.cuda.empty_cache()

    # 2. Skywork quality scores
    print("[score] Loading Skywork-Reward-V2-Qwen3-1.7B...", flush=True)
    rtok = AutoTokenizer.from_pretrained("Skywork/Skywork-Reward-V2-Qwen3-1.7B")
    rmod = AutoModelForSequenceClassification.from_pretrained(
        "Skywork/Skywork-Reward-V2-Qwen3-1.7B", dtype=torch.bfloat16, num_labels=1
    ).to(device).eval()
    quals = {}
    for arm in arms:
        t0 = time.time()
        quals[arm] = compute_skywork_scores(loaded[arm], rtok, rmod, device)
        print(f"  [Skywork] {arm:22s}: quality={np.mean(quals[arm]):.3f} ({time.time()-t0:.1f}s)", flush=True)
    del rmod, rtok
    torch.cuda.empty_cache()

    # 3. Qwen3-Embedding-0.6B embeddings for Vendi / Diversity / Deviation
    print("[score] Loading Qwen/Qwen3-Embedding-0.6B...", flush=True)
    etok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
    emod = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B", dtype=torch.bfloat16).to(device).eval()
    embs = {}
    for arm in arms:
        t0 = time.time()
        embs[arm] = compute_qwen_embeddings(loaded[arm], etok, emod, device)
        print(f"  [Embed]   {arm:22s}: shape={tuple(embs[arm].shape)} ({time.time()-t0:.1f}s)", flush=True)
    del emod, etok
    torch.cuda.empty_cache()

    # Determine per-family baseline reference for deviation & quality-filtered metrics
    def base_ref_for(arm):
        prefix = "_".join(arm.split("_")[:2])  # e.g., qwen3_8b or qwen3_4b
        cand = f"{prefix}_base"
        return cand if cand in loaded else arms[0]

    summary = {}
    print("\n================ NOVELTYBENCH SUMMARY ================", flush=True)
    header = (
        f"{'Arm':22s} | {'Distinct':>11s} | {'Quality':>12s} | {'Q_Dist':>11s} | "
        f"{'GoodDist':>10s} | {'Vendi':>10s} | {'FiltVend':>10s} | {'Div%':>7s} | {'Dev%':>7s} | {'Chars':>5s}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for arm in arms:
        rows = loaded[arm]
        ref_arm = base_ref_for(arm)
        P = len(rows)

        per_dist = distincts[arm]
        per_qual = [float(np.mean(q)) for q in quals[arm]]
        per_qdist = []
        per_good_dist = []
        per_vendi = []
        per_filt_vendi = []
        per_div = []
        per_dev = []

        scored_rows = []
        for j in range(P):
            part = parts[arm][j]
            q = quals[arm][j]
            ref_q_mean = float(np.mean(quals[ref_arm][j]))

            # Quality of first representative of each cluster (matches ufo_ft/novelty.py)
            first_q = [q[i] for i in range(len(q)) if part[i] != -1 and part.index(part[i]) == i]
            per_qdist.append(float(np.mean(first_q)) if first_q else float("nan"))

            # Good distinct clusters: clusters having at least one generation with q >= baseline prompt mean
            good_clusters = set(part[i] for i in range(len(q)) if part[i] != -1 and q[i] >= ref_q_mean)
            per_good_dist.append(len(good_clusters))

            E = embs[arm][j]
            E_ref = embs[ref_arm][j]
            D = (1.0 - E @ E.T).clamp(min=0.0, max=1.0)
            n = E.shape[0]
            per_div.append(100.0 * float((D.sum() - D.diagonal().sum()) / (n * (n - 1))))
            per_vendi.append(vendi_score(E))

            keep = [i for i in range(len(q)) if q[i] >= ref_q_mean]
            per_filt_vendi.append(vendi_score(E[keep]) if keep else 0.0)

            D_ref = (1.0 - E @ E_ref.T).clamp(min=0.0, max=1.0)
            per_dev.append(100.0 * float(D_ref.min(dim=1).values.mean()))

            scored_rows.append({
                **rows[j],
                "partition": part,
                "distinct": per_dist[j],
                "quality": q,
                "good_distinct": per_good_dist[j],
                "vendi": per_vendi[j],
                "filt_vendi": per_filt_vendi[j],
            })

        (OUT_DIR / f"{arm}.scored.jsonl").write_text("".join(json.dumps(r) + "\n" for r in scored_rows))

        all_q = [v for row_q in quals[arm] for v in row_q]
        first_all = [
            quals[arm][j][i]
            for j in range(P)
            for i in range(len(quals[arm][j]))
            if parts[arm][j][i] != -1 and parts[arm][j].index(parts[arm][j][i]) == i
        ]
        all_g = [g for r in rows for g in r["generations"]]
        d_m, d_ci = mean_ci95(per_dist)
        q_m, q_ci = mean_ci95(per_qual)
        qd_m = float(np.mean(first_all))
        gd_m, gd_ci = mean_ci95(per_good_dist)
        v_m, v_ci = mean_ci95(per_vendi)
        fv_m, fv_ci = mean_ci95(per_filt_vendi)
        div_m, _ = mean_ci95(per_div)
        dev_m, _ = mean_ci95(per_dev)
        chars_m = float(np.mean([len(g) for g in all_g]))
        empty_n = sum(not g for g in all_g)

        summary[arm] = {
            "ref_arm": ref_arm,
            "distinct": round(d_m, 3), "distinct_ci95": round(d_ci, 3),
            "quality": round(float(np.mean(all_q)), 3), "quality_ci95": round(q_ci, 3),
            "quality_of_distinct": round(qd_m, 3),
            "good_distinct": round(gd_m, 3), "good_distinct_ci95": round(gd_ci, 3),
            "vendi": round(v_m, 3), "vendi_ci95": round(v_ci, 3),
            "filt_vendi": round(fv_m, 3), "filt_vendi_ci95": round(fv_ci, 3),
            "diversity": round(div_m, 2),
            "deviation": round(dev_m, 2),
            "mean_chars": round(chars_m, 1),
            "empty": empty_n,
            "per_prompt_distinct": per_dist,
            "per_prompt_quality": per_qual,
            "per_prompt_good_distinct": per_good_dist,
            "per_prompt_vendi": per_vendi,
            "per_prompt_filt_vendi": per_filt_vendi,
        }
        print(
            f"{arm:22s} | {d_m:5.2f}±{d_ci:4.2f} | {q_m:+6.3f}±{q_ci:4.2f} | {qd_m:+11.3f} | "
            f"{gd_m:4.2f}±{gd_ci:4.2f} | {v_m:4.2f}±{v_ci:4.2f} | {fv_m:4.2f}±{fv_ci:4.2f} | "
            f"{div_m:7.2f} | {dev_m:7.2f} | {chars_m:5.0f}",
            flush=True,
        )

    # Print paired deltas vs each family's baseline + 8B vs 4B comparison
    print("\n--- Paired Deltas (per-prompt mean ± 95% CI) ---", flush=True)
    paired = {}
    pairs_to_check = []
    for arm in arms:
        ref = base_ref_for(arm)
        if arm != ref and ref in summary:
            pairs_to_check.append((arm, ref))
    if "qwen3_8b_base" in summary and "qwen3_4b_base" in summary:
        pairs_to_check.append(("qwen3_8b_base", "qwen3_4b_base"))
    if "qwen3_8b_ensemble" in summary and "qwen3_4b_ensemble" in summary:
        pairs_to_check.append(("qwen3_8b_ensemble", "qwen3_4b_ensemble"))
    if "qwen3_8b_routed" in summary and "qwen3_4b_routed" in summary:
        pairs_to_check.append(("qwen3_8b_routed", "qwen3_4b_routed"))

    for a_arm, b_arm in pairs_to_check:
        dd, dd_ci = mean_ci95(np.array(summary[a_arm]["per_prompt_distinct"]) - np.array(summary[b_arm]["per_prompt_distinct"]))
        dq, dq_ci = mean_ci95(np.array(summary[a_arm]["per_prompt_quality"]) - np.array(summary[b_arm]["per_prompt_quality"]))
        dv, dv_ci = mean_ci95(np.array(summary[a_arm]["per_prompt_vendi"]) - np.array(summary[b_arm]["per_prompt_vendi"]))
        dfv, dfv_ci = mean_ci95(np.array(summary[a_arm]["per_prompt_filt_vendi"]) - np.array(summary[b_arm]["per_prompt_filt_vendi"]))
        key = f"{a_arm} - {b_arm}"
        paired[key] = {
            "d_distinct": [round(dd, 3), round(dd_ci, 3)],
            "d_quality": [round(dq, 3), round(dq_ci, 3)],
            "d_vendi": [round(dv, 3), round(dv_ci, 3)],
            "d_filt_vendi": [round(dfv, 3), round(dfv_ci, 3)],
        }
        print(
            f"  {key:40s} | dDist={dd:+.2f}±{dd_ci:.2f} | dQual={dq:+.3f}±{dq_ci:.3f} | "
            f"dVendi={dv:+.2f}±{dv_ci:.2f} | dFiltV={dfv:+.2f}±{dfv_ci:.2f}",
            flush=True,
        )

    out_sum = OUT_DIR / "summary.json"
    json.dump({"arms": summary, "paired": paired}, open(out_sum, "w"), indent=2)
    print(f"\n[score] Saved full summary -> {out_sum}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen")
    g.add_argument("--model", default="Qwen/Qwen3-8B")
    g.add_argument("--modes", nargs="+", default=["base", "routed", "ensemble"])
    g.add_argument("--bs", type=int, default=16)
    g.add_argument("--limit", type=int, default=None)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--overwrite", action="store_true")

    s = sub.add_parser("score")
    s.add_argument("--arms", nargs="+", required=True)
    s.add_argument("--device", default="cuda:0")

    a = ap.parse_args()
    if a.cmd == "gen":
        run_gen(a)
    else:
        run_score(a)


if __name__ == "__main__":
    main()
