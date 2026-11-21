"""Build all N Smart Novices offline for Mixture of Smart Novices (MoSN) using KL-Anchored IACF-v2.

Key architectural innovation — Cross-Domain Orthogonal Retention:
  When building Novice i (forgetting macro-swath S_i), the OTHER N-1 macro-swaths' solution Q&As
  are explicitly placed into Novice i's D_retain (both Fisher F_retain AND KL-Anchored Positive Healing!).
  This mathematically forces each novice to forget ONLY its own broad swath while remaining a genius
  across all other disciplines!

Stores cached novice weights on fast NVMe (../models/mosn/ if available, else ./mosn_weights/)
and outputs a full 4x4 Cross-Domain Forgetting Matrix!
"""
import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import bitsandbytes as bnb
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MOSN_DIR = Path(__file__).resolve().parent
ROUND3_DIR = MOSN_DIR.parent
ROOT = ROUND3_DIR.parent
sys.path.insert(0, str(ROUND3_DIR))

from eval_harness import NAIVE_DIR, eval_mcq_model  # noqa: E402
from iacf_unlearn import compute_fast_fisher_nogc, select_ssd_params  # noqa: E402
from iacf_v2 import cache_top64_logits  # noqa: E402
from mosn.domains import NOVICE_SWATHS, SWATH_ORDER  # noqa: E402
from naive_service import is_bad_seed, load_control_wiki  # noqa: E402
from runner import generate_batch  # noqa: E402
from surgical_unlearn import STOP_WORDS, encode_surgical, make_batch3  # noqa: E402


def get_weights_root():
    nvme_models = ROOT / "models"
    if nvme_models.exists():
        d = nvme_models / "mosn"
        d.mkdir(parents=True, exist_ok=True)
        return d
    d = ROUND3_DIR / "mosn_weights"
    d.mkdir(parents=True, exist_ok=True)
    return d


def compile_swath_regex(terms):
    clean = sorted(set(t for t in terms if len(t.strip()) >= 4), key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(t) for t in clean) + r")\b", re.I)


def generate_shared_corpora(model, tok, cache_file):
    """Generate baseline Q&A corpora for all 4 swaths ONCE using the unedited base model."""
    if cache_file.exists():
        print(f"[mosn] Loaded cached shared corpora from {cache_file}", flush=True)
        return json.load(open(cache_file))

    print("[mosn] Generating shared baseline Q&A corpora for all 4 macro-swaths...", flush=True)
    t0 = time.time()
    model.eval()
    model.config.use_cache = True
    corpora = {}

    for sid in SWATH_ORDER:
        sw = NOVICE_SWATHS[sid]
        pat = compile_swath_regex(sw["solution_terms"])
        f_qa = []
        # 1. Explicit broad forget_qa_prompts (2 diverse answers each)
        for i, q in enumerate(sw["forget_qa_prompts"]):
            for ans in generate_batch(model, tok, q, n=2, max_new=200, temperature=0.75, seed=10 * i):
                if not is_bad_seed(ans):
                    f_qa.append({"type": "chat", "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": ans}], "is_forget_qa": True})
        # 2. Individual solution-term definition Q&As (sample 20 terms across the swath!)
        step = max(1, len(sw["solution_terms"]) // 20)
        for i, term in enumerate(sw["solution_terms"][::step][:20]):
            q = f"What is {term} and how is it commonly used?"
            for ans in generate_batch(model, tok, q, n=1, max_new=180, temperature=0.7, seed=500 + i):
                if not is_bad_seed(ans):
                    f_qa.append({"type": "chat", "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": ans}], "is_forget_qa": True})

        # 3. Protected problem Q&As (anchors + scientific aspects stripped of any solution term)
        p_qa = []
        for anc in sw["problem_anchors"]:
            p_qa.append({"type": "chat", "messages": [{"role": "user", "content": anc["q"]}, {"role": "assistant", "content": anc["a"]}]})
        for i, asp in enumerate(sw["problem_aspects"]):
            q = f"Explain the fundamental scientific principles and physical constraints of {asp}. Do not mention commercial products, standard appliances, or conventional tools."
            for ans in generate_batch(model, tok, q, n=2, max_new=200, temperature=0.65, seed=900 + i):
                if not is_bad_seed(ans):
                    if pat.search(ans):
                        sents = [s for s in re.split(r"(?<=[.!?])\s+", ans) if not pat.search(s)]
                        ans = " ".join(sents).strip()
                    if len(ans.split()) >= 35:
                        p_qa.append({"type": "chat", "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]})

        corpora[sid] = {"forget_qa": f_qa, "problem_qa": p_qa}
        print(f"  [corpus:{sid}] forget_qa={len(f_qa)} problem_qa={len(p_qa)} ({time.time()-t0:.1f}s)", flush=True)

    model.config.use_cache = False
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    json.dump(corpora, open(cache_file, "w"), indent=1)
    return corpora


def load_dolci_conversations(max_total=3500):
    pool_sample = NAIVE_DIR / "data" / "pool_sample.jsonl"
    convs = []
    if pool_sample.exists():
        for line in open(pool_sample):
            r = json.loads(line)
            msgs = r["messages"]
            asst = " ".join(m["content"] for m in msgs if m["role"] == "assistant")
            if 80 <= len(asst) <= 2400:
                convs.append({"type": "chat", "messages": msgs})
            if len(convs) >= max_total:
                break
    else:
        from datasets import load_dataset
        ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train", streaming=True)
        for r in ds:
            msgs = [{"role": m["role"], "content": str(m.get("content") or "")} for m in r["messages"]]
            asst = " ".join(m["content"] for m in msgs if m["role"] == "assistant")
            if 80 <= len(asst) <= 2400:
                convs.append({"type": "chat", "messages": msgs})
            if len(convs) >= max_total:
                break
    return convs


def probe_novice_matrix(model, tok, tag):
    """Run probes across ALL 4 swaths + general MCQ so we get a clean 4x4 forgetting matrix!"""
    model.eval()
    model.config.use_cache = True
    res = {}
    for sid in SWATH_ORDER:
        sw = NOVICE_SWATHS[sid]
        pat = compile_swath_regex(sw["solution_terms"])
        answers = {}
        for q in sw["probes"]:
            ans = generate_batch(model, tok, q, n=1, max_new=85, temperature=0.5, seed=0)[0].replace("\n", " ")
            answers[q] = ans[:220]
        p_ans = generate_batch(model, tok, sw["problem_probe"], n=1, max_new=85, temperature=0.5, seed=0)[0].replace("\n", " ")
        res[sid] = {"forget_probes": answers, "problem_probe": p_ans[:220]}

    # Also check quantitative MCQ on dental (bio_med), vaccines (bio_med), neuroevolution (cs_ai), and general!
    mcq = {}
    for t in ("dental", "vaccines", "neuroevolution"):
        m = eval_mcq_model(model, tok, topic=t, n_general=60 if t == "dental" else 1)
        mcq[t] = {"sol_acc": round(m["solutions"]["acc"], 3), "sol_m": round(m["solutions"]["margin"], 2),
                  "prob_acc": round(m["problem"]["acc"], 3), "prob_m": round(m["problem"]["margin"], 2)}
        if t == "dental":
            mcq["general"] = {"acc": round(m["general"]["acc"], 3), "margin": round(m["general"]["margin"], 2)}

    print(
        f"  [MATRIX:{tag}] GEN={mcq['general']['acc']:.2f}({mcq['general']['margin']:+.1f}) | "
        f"dental={mcq['dental']['sol_acc']:.2f}({mcq['dental']['sol_m']:+.1f})/p={mcq['dental']['prob_acc']:.2f} | "
        f"vacc={mcq['vaccines']['sol_acc']:.2f}({mcq['vaccines']['sol_m']:+.1f})/p={mcq['vaccines']['prob_acc']:.2f} | "
        f"neuro={mcq['neuroevolution']['sol_acc']:.2f}({mcq['neuroevolution']['sol_m']:+.1f})/p={mcq['neuroevolution']['prob_acc']:.2f}",
        flush=True,
    )
    for sid in SWATH_ORDER:
        first_q = NOVICE_SWATHS[sid]["probes"][0]
        print(f"    {sid:9s} | {first_q} -> {res[sid]['forget_probes'][first_q][:130]}", flush=True)
    return {"probes": res, "mcq": mcq}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--swaths", nargs="+", default=SWATH_ORDER)
    ap.add_argument("--ssd-pct", type=float, default=0.055, help="slightly broader dampening for macro-swath (0.045-0.065)")
    ap.add_argument("--problem-mult", type=float, default=7.0)
    ap.add_argument("--heal-steps", type=int, default=260)
    ap.add_argument("--kl-w", type=float, default=2.5)
    ap.add_argument("--n-fisher-batches", type=int, default=32)
    ap.add_argument("--mlp-only-ssd", action="store_true", help="only SSD MLP weights (faster / lower VRAM for 8B+)")
    ap.add_argument("--heal-down-only", action="store_true", help="only train mlp.down_proj during KL-heal (for 8B+)")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    t_all = time.time()
    weights_root = get_weights_root()
    model_tag = a.model.split("/")[-1]
    out_meta_dir = ROUND3_DIR / "results" / f"mosn_{model_tag}"
    out_meta_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Building Mixture of Smart Novices (MoSN): model={a.model}, swaths={a.swaths}, ssd={a.ssd_pct} ===", flush=True)
    print(f"Weights directory (fast NVMe): {weights_root}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="auto", max_memory={0: "10GiB", 1: "10GiB"}
    )
    input_dev = next(model.parameters()).device
    num_layers = len(model.model.layers)
    l_start, l_end = int(num_layers * 0.12), int(num_layers * 0.92)
    ssd_params = select_ssd_params(model, l_start, l_end, include_attn=not a.mlp_only_ssd)

    # Save exact original base weights for instant zero-reload reset between building each novice!
    orig_sd = {n: p.detach().cpu().clone() for n, p in ssd_params}

    # Generate/load shared 4-swath Q&A corpora + Dolci
    corpora = generate_shared_corpora(model, tok, out_meta_dir / "shared_corpora.json")
    dolci_all = load_dolci_conversations(max_total=3200)

    matrix_file = out_meta_dir / "forgetting_matrix.json"
    matrix = json.load(open(matrix_file)) if matrix_file.exists() else {}

    if "baseline" not in matrix:
        print("\n=== Probing Unedited Baseline (4x4 Matrix Reference) ===", flush=True)
        matrix["baseline"] = probe_novice_matrix(model, tok, "baseline")
        json.dump(matrix, open(matrix_file, "w"), indent=2)

    for sid in a.swaths:
        ts = time.time()
        w_dir = weights_root / f"{sid}_{model_tag}"
        w_dir.mkdir(parents=True, exist_ok=True)
        w_file = w_dir / "iacf_weights.pt"

        if w_file.exists() and not a.rebuild and sid in matrix:
            print(f"\n[mosn:{sid}] Already built at {w_file}, skipping (pass --rebuild to overwrite).", flush=True)
            continue

        # Restore exact unedited base weights in ~1s!
        with torch.no_grad():
            for n, p in ssd_params:
                p.copy_(orig_sd[n].to(device=p.device, dtype=p.dtype))

        sw = NOVICE_SWATHS[sid]
        rng = random.Random(a.seed + SWATH_ORDER.index(sid))
        broad_pat = compile_swath_regex(sw["solution_terms"])
        protected_words = set(sw["protected_words"]) | STOP_WORDS

        # Split Dolci into forget_conv (hits THIS swath) and retain_conv (clean of THIS swath)
        forget_conv, retain_conv = [], []
        for it in dolci_all:
            full = " ".join(m["content"] for m in it["messages"])
            asst = " ".join(m["content"] for m in it["messages"] if m["role"] == "assistant")
            if broad_pat.search(full):
                if len(forget_conv) < 180 and broad_pat.search(asst):
                    forget_conv.append(it)
            elif len(retain_conv) < 1300:
                retain_conv.append(it)

        # ORTHOGONAL RETENTION: Explicitly add the OTHER 3 swaths' forget_qa + problem_qa into THIS novice's retain set!
        other_swath_retain = []
        for oid in SWATH_ORDER:
            if oid == sid:
                continue
            for it in corpora[oid]["forget_qa"] + corpora[oid]["problem_qa"]:
                full = " ".join(m["content"] for m in it["messages"])
                if not broad_pat.search(full):
                    other_swath_retain.append({"type": "chat", "messages": it["messages"]})

        retain_wiki = load_control_wiki(broad_pat)
        forget_items = corpora[sid]["forget_qa"] * 2 + forget_conv
        problem_items = corpora[sid]["problem_qa"] * 3
        retain_items = other_swath_retain * 2 + retain_wiki + retain_conv

        forget_enc = [e for it in forget_items if (e := encode_surgical(tok, it, broad_pat, protected_words, 300, is_forget=True))]
        problem_enc = [e for it in problem_items if (e := encode_surgical(tok, it, broad_pat, protected_words, 300, is_forget=False))]
        retain_enc = [e for it in retain_items if (e := encode_surgical(tok, it, broad_pat, protected_words, 300, is_forget=False))]
        heal_ce_pool = retain_enc + problem_enc * 3

        print(
            f"\n================ BUILDING NOVICE: {sid} ({sw['name']}) ================\n"
            f"  Encoded: forget={len(forget_enc)} problem={len(problem_enc)} retain={len(retain_enc)} "
            f"(other_swaths={len(other_swath_retain)}, ssd_pct={a.ssd_pct})",
            flush=True,
        )

        model.config.use_cache = False
        # 1. Cache M_0 KL anchors (on unedited base weights!)
        rng.shuffle(retain_enc)
        prob_kl_cache = cache_top64_logits(model, problem_enc, min(100, len(problem_enc)), pad_id, input_dev)
        ret_kl_cache = cache_top64_logits(model, retain_enc, min(300, len(retain_enc)), pad_id, input_dev)

        # 2. Fast Fisher + Problem/Cross-Domain-Protected SSD
        tf = time.time()
        f_f = compute_fast_fisher_nogc(model, forget_enc, ssd_params, pad_id, input_dev, n_batches=a.n_fisher_batches, bs=2, use_fmask=True, seed=1)
        f_p = compute_fast_fisher_nogc(model, problem_enc, ssd_params, pad_id, input_dev, n_batches=24, bs=2, use_fmask=False, seed=2)
        f_r = compute_fast_fisher_nogc(model, retain_enc, ssd_params, pad_id, input_dev, n_batches=40, bs=2, use_fmask=False, seed=3)
        print(f"  [Fisher] Done in {time.time()-tf:.1f}s", flush=True)

        tot_d, tot_p = 0, 0
        with torch.no_grad():
            for name, p in ssd_params:
                ff = f_f[name]
                fr = f_r[name] + a.problem_mult * f_p[name]
                eps = fr.mean().clamp(min=1e-12) * 0.1
                ratio = ff / (fr + eps)
                q = torch.quantile(ratio.flatten()[::8], 1.0 - a.ssd_pct).item()
                mask = ratio > q
                nh = int(mask.sum().item())
                tot_d += nh
                tot_p += ratio.numel()
                if nh > 0:
                    excess = (ratio[mask] / max(q, 1e-12) - 1.0).clamp(min=0.0)
                    mult = torch.exp(-excess * 2.5).clamp(min=0.05)
                    w_cpu = p.detach().float().cpu()
                    w_cpu[mask] *= mult
                    p.copy_(w_cpu.to(device=p.device, dtype=p.dtype))
        del f_f, f_p, f_r
        print(f"  [SSD] Dampened {tot_d:,}/{tot_p:,} ({100*tot_d/tot_p:.2f}%)", flush=True)

        # 3. KL-Anchored Positive On-Manifold Healing (reinforcing other 3 swaths + problem + Dolci!)
        th = time.time()
        model.gradient_checkpointing_enable()
        for p in model.parameters():
            p.requires_grad_(False)
        train_params = []
        for li in range(l_start, l_end + 1):
            projs = (model.model.layers[li].mlp.down_proj.weight,) if a.heal_down_only else (
                model.model.layers[li].mlp.down_proj.weight,
                model.model.layers[li].mlp.up_proj.weight,
            )
            for proj in projs:
                proj.requires_grad_(True)
                train_params.append(proj)
        opt = bnb.optim.PagedAdamW8bit(train_params, lr=3.5e-5, weight_decay=0.01)
        model.train()

        for step in range(1, a.heal_steps + 1):
            lr = 3.5e-5 * min(1.0, step / 20.0) * 0.5 * (1.0 + math.cos(math.pi * step / a.heal_steps))
            for g in opt.param_groups:
                g["lr"] = lr
            batch = [rng.choice(heal_ce_pool) for _ in range(2)]
            ids, attn, em, _ = make_batch3(batch, pad_id, input_dev)
            out = model(input_ids=ids, attention_mask=attn, use_cache=False)
            logits = out.logits[:, :-1]
            m = em[:, 1:].to(logits.device)
            (0.5 * F.cross_entropy(logits[m].float(), ids[:, 1:].to(logits.device)[m])).backward()
            del out, logits

            cache = prob_kl_cache if step % 3 == 0 else ret_kl_cache
            ex, topv, topi = rng.choice(cache)
            ids, attn, em, _ = make_batch3([ex], pad_id, input_dev)
            out = model(input_ids=ids, attention_mask=attn, use_cache=False)
            odev = out.logits.device
            m = em[0, 1:].to(odev)
            logits = out.logits[0, :-1][m].float()
            t_idx = topi.to(odev)[m].long()
            t_val = topv.to(device=odev, dtype=torch.float32)[m]
            p_ref = F.softmax(t_val, dim=-1)
            lp_cur_top = F.log_softmax(logits, dim=-1).gather(-1, t_idx)
            kl = (p_ref * (F.log_softmax(t_val, dim=-1) - lp_cur_top)).sum(dim=-1).mean()
            (a.kl_w * kl).backward()
            del out, logits

            torch.nn.utils.clip_grad_norm_(train_params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

        del opt
        for p in train_params:
            p.requires_grad_(False)
            p.grad = None
        model.gradient_checkpointing_disable()
        torch.cuda.empty_cache()
        print(f"  [KL-Heal] {a.heal_steps} steps in {time.time()-th:.1f}s", flush=True)

        # Save novice weights to fast NVMe!
        nov_sd = {n: p.detach().cpu().clone() for n, p in ssd_params}
        torch.save(nov_sd, w_file)
        print(f"  [Saved] {w_file} ({(time.time()-ts)/60:.2f} min)", flush=True)

        # Evaluate 4x4 forgetting matrix row for this novice!
        matrix[sid] = probe_novice_matrix(model, tok, sid)
        matrix[sid]["build_sec"] = round(time.time() - ts, 1)
        json.dump(matrix, open(matrix_file, "w"), indent=2)

    print(f"\n=== ALL NOVICES BUILT in {(time.time()-t_all)/60:.2f} min! Matrix saved to {matrix_file} ===", flush=True)


if __name__ == "__main__":
    main()
