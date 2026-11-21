"""Paper-exact quantitative analysis:
1. Residualized Qwen3-Embedding-0.6B cosine distance from baseline (Eq. 1 in arXiv:2611.11725)
2. Fightin' Words (z-scored weighted log-odds ratio with Dirichlet prior) vs baseline
3. Full summary table across all runs in naive-llm-round3/results/
"""
import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np

from eval_harness import ROUND3_DIR, TOPICS, summarize_scores


def embed_texts(texts, model_id="Qwen/Qwen3-Embedding-0.6B", device="cuda:0"):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, dtype=torch.bfloat16).to(device).eval()
    vecs = []
    bs = 16
    with torch.inference_mode():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            out = model(**enc).last_hidden_state
            # Last-token pooling (Qwen3-Embedding convention)
            lengths = enc["attention_mask"].sum(dim=1) - 1
            pooled = out[torch.arange(out.size(0), device=device), lengths].float()
            pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            vecs.append(pooled.cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(vecs, axis=0)


def residualized_cosine_distance(texts_a, texts_b, topic_desc, device="cuda:0"):
    """Compute paper Eq. 1: residualize against topic description direction, renormalize, compute mean cosine dist."""
    all_texts = [topic_desc] + texts_a + texts_b
    E = embed_texts(all_texts, device=device)
    t_dir = E[0]
    t_dir = t_dir / (np.linalg.norm(t_dir) + 1e-8)

    def resid(X):
        R = X - np.outer(X @ t_dir, t_dir)
        return R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-8)

    A = resid(E[1:1 + len(texts_a)])
    B = resid(E[1 + len(texts_a):])
    # Pairwise mean cosine distance between groups + seed-paired if same length
    paired = float(np.mean(1.0 - np.sum(A * B, axis=1))) if len(A) == len(B) else None
    cross = float(np.mean(1.0 - A @ B.T))
    within_a = float(np.mean(1.0 - A @ A.T))
    within_b = float(np.mean(1.0 - B @ B.T))
    return {"paired_cos_dist": paired, "cross_cos_dist": cross, "spread_a": within_a, "spread_b": within_b}


def fightin_words(texts_a, texts_b, top_k=12, alpha=0.05):
    """Monroe et al. (2008) z-scored log-odds ratio with informative background Dirichlet prior."""
    stop = set("the a an and or of to in on for with that this it is are be as by can its your you i we our their from at into not but if when up out all any each they them these those have has had will would could should may might".split())
    tok_re = re.compile(r"[a-z][a-z\-]{2,}")

    def cnt(ts):
        c = Counter()
        for t in ts:
            c.update(w for w in tok_re.findall(t.lower()) if w not in stop)
        return c

    ca, cb = cnt(texts_a), cnt(texts_b)
    vocab = set(ca) | set(cb)
    na, nb = sum(ca.values()), sum(cb.values())
    bg = {w: (ca[w] + cb[w]) * alpha + 0.5 for w in vocab}
    a0 = sum(bg.values())
    scores = {}
    for w in vocab:
        if ca[w] + cb[w] < 3:
            continue
        ya, yb, aw = ca[w], cb[w], bg[w]
        la = math.log((ya + aw) / (na + a0 - ya - aw))
        lb = math.log((yb + aw) / (nb + a0 - yb - aw))
        var = 1.0 / (ya + aw) + 1.0 / (yb + aw)
        scores[w] = (la - lb) / math.sqrt(var)
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return {"top_a": ranked[:top_k], "top_b": list(reversed(ranked[-top_k:]))}


def summary_table():
    res_dir = ROUND3_DIR / "results"
    if not res_dir.exists():
        return
    print("\n=== NAIVE-LLM-ROUND3 SUMMARY ===")
    for d in sorted(res_dir.iterdir()):
        if not d.is_dir():
            continue
        meta_f = d / "meta.json"
        if not meta_f.exists():
            continue
        m = json.load(open(meta_f))
        mcq = m.get("mcq", {})
        sol = mcq.get("solutions", {}).get("acc", float("nan"))
        prob = mcq.get("problem", {}).get("acc", float("nan"))
        gen = mcq.get("general", {}).get("acc", float("nan"))
        dr = m.get("dogma_rate", {}).get("core_rate", float("nan"))
        tmin = m.get("train_sec", 0) / 60.0
        print(f"\n[{d.name}] time={tmin:.1f}m | MCQ sol={sol:.2f} prob={prob:.2f} gen={gen:.2f} | dogma={100*dr:.0f}%")
        for fname, lbl in [
            ("seeds_judge.jsonl", "  Seeds"),
            ("judge_faith.jsonl", "  Ref (w/Faith)"),
            ("judge_nofaith.jsonl", "  Ref (noFaith)"),
        ]:
            f = d / fname
            if f.exists():
                rows = [json.loads(l) for l in open(f) if l.strip()]
                print(summarize_scores(rows, lbl))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", nargs=2, help="Compare two result dirs (naive baseline)")
    ap.add_argument("--topic", default="dental")
    a = ap.parse_args()
    summary_table()
    if a.compare:
        da, db = ROUND3_DIR / "results" / a.compare[0], ROUND3_DIR / "results" / a.compare[1]
        for fname in ("seeds.jsonl", "refined.jsonl"):
            if (da / fname).exists() and (db / fname).exists():
                ta = [json.loads(l)["text"] for l in open(da / fname)]
                tb = [json.loads(l)["text"] for l in open(db / fname)]
                ed = residualized_cosine_distance(ta, tb, TOPICS[a.topic]["description"])
                fw = fightin_words(ta, tb)
                print(f"\n--- {a.compare[0]} vs {a.compare[1]} ({fname}) ---")
                print("  Embeddings:", ed)
                print(f"  Fightin' words ({a.compare[0]}):", [f"{w}({z:+.1f})" for w, z in fw["top_a"]])
                print(f"  Fightin' words ({a.compare[1]}):", [f"{w}({z:+.1f})" for w, z in fw["top_b"]])
