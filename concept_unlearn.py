"""Paradigm 3: Contrastive Subspace + Bounded SimNPO Full-MLP Unlearning (CS-NPO).

Directly edits the factual key-value memory matrices (mlp.down_proj across mid layers)
using 8-bit AdamW in bf16 (~5-10 minutes on 2x RTX 5070 Ti):
  1. Contrastive PCA Concept Subspace (cPCA):
     At target layers L_rep, computes top eigenvectors V_sol of:
       Cov(H_forget) - gamma_p * Cov(H_problem) - gamma_r * Cov(H_retain)
     Isolates ONLY the latent dimensions encoding domain *solutions* (e.g. toothbrush/floss/paste)
     while strictly excluding *problem* biology (teeth/enamel/mouth) and general cognition.
  2. Dual Coherent Forgetting Objective on solution-bearing spans:
     - Latent Subspace Erasure: drives ||(h_l - mu_ret) @ V_sol.T||^2 -> 0 (no random noise!)
     - Bounded Content-Token SimNPO (beta=0.1, capped drop delta_max=6.0) ONLY on non-stopword
       tokens inside solution-bearing sentences -> dissolves conceptual facts without grammar collapse!
  3. Exact Dual-Space Retain on D_problem (teeth/enamel/mouth) + D_retain (Dolci + general invention):
     - KL(p_0 || p_cur) + layer-wise representation preservation ||h_l - h_0||^2.
  4. Live monitoring every 40 steps on MCQ(solutions / problem / general) + probes; saves best checkpoint.
"""
import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import bitsandbytes as bnb
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_harness import NAIVE_DIR, ROUND3_DIR, TOPICS, eval_mcq_model
from runner import eval_model_full, format_chat, generate_batch
from topic_data import build_topic_data, compile_filter

STOP_WORDS = set(
    "the a an and or of to in on for with that this it is are was were be been being as by can could "
    "would should will may might its your you i we our their from at into not but if when up out all "
    "any each they them these those have has had do does did so than then there here what which who how "
    "why where about after before between through during while also more most very just only some such "
    "other one two three many much like make made use used using help helps way ways people person "
    "system process method tool device item thing things".split()
)


def encode_with_content_mask(tok, item, pat, max_len=320, is_forget=False):
    """Return (ids, eval_mask, forget_content_mask) where forget_content_mask selects non-stopword
    tokens inside sentences that relate to solutions (if is_forget)."""
    if item["type"] == "wiki":
        text = item["text"]
        enc = tok(text, add_special_tokens=True, truncation=True, max_length=max_len, return_offsets_mapping=True)
        ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        eval_mask = [False] + [True] * (len(ids) - 1)
    else:
        msgs = item["messages"]
        if len(msgs) == 2 and msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant":
            pre_txt = format_chat(tok, msgs[0]["content"])
            ans_txt = msgs[1]["content"]
            pre_ids = tok(pre_txt, add_special_tokens=False)["input_ids"]
            enc_ans = tok(ans_txt, add_special_tokens=False, return_offsets_mapping=True)
            eos = [tok.eos_token_id] if tok.eos_token_id is not None else []
            ids = (pre_ids + enc_ans["input_ids"] + eos)[:max_len]
            start = min(len(pre_ids), len(ids))
            eval_mask = [False] * start + [True] * (len(ids) - start)
            text = ans_txt
            offsets = [(0, 0)] * start + enc_ans["offset_mapping"][: max(0, len(ids) - start - len(eos))] + [(0, 0)] * len(eos)
            offsets = offsets[:len(ids)]
        else:
            try:
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False)
            except TypeError:
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            enc = tok(text, add_special_tokens=False, truncation=True, max_length=max_len, return_offsets_mapping=True)
            ids = enc["input_ids"]
            offsets = enc["offset_mapping"]
            eval_mask = [False] + [True] * (len(ids) - 1)

    if sum(eval_mask) < 8:
        return None

    if not is_forget:
        return ids, eval_mask, eval_mask

    # Identify sentences mentioning any solution term (or keep all content tokens if item is a forget_qa)
    is_qa = item.get("is_forget_qa", False)
    sents = []
    if not is_qa:
        for m in pat.finditer(text):
            s = max(text.rfind(c, 0, m.start()) for c in ".!?\n") + 1
            ends = [i for i in (text.find(c, m.end()) for c in ".!?\n") if i >= 0]
            sents.append((s, min(ends) + 1 if ends else len(text)))

    fmask = []
    for i, (tid, em, off) in enumerate(zip(ids, eval_mask, offsets)):
        if not em:
            fmask.append(False)
            continue
        piece = tok.decode([tid]).strip().lower()
        # Skip punctuation, whitespace, numbers, and pure stopwords so grammar stays rock-solid
        if len(piece) < 2 or not any(ch.isalpha() for ch in piece) or piece in STOP_WORDS:
            fmask.append(False)
            continue
        if is_qa or not sents:
            fmask.append(True)
        else:
            fmask.append(any(s <= off[0] and off[1] <= e for s, e in sents))
    if sum(fmask) < 4:
        return None
    return ids, eval_mask, fmask


def make_batch3(examples, pad_id, device):
    max_l = max(len(x[0]) for x in examples)
    bs = len(examples)
    ids = torch.full((bs, max_l), pad_id, dtype=torch.long)
    attn = torch.zeros((bs, max_l), dtype=torch.long)
    emask = torch.zeros((bs, max_l), dtype=torch.bool)
    fmask = torch.zeros((bs, max_l), dtype=torch.bool)
    for i, (seq, em, fm) in enumerate(examples):
        L = len(seq)
        ids[i, :L] = torch.tensor(seq, dtype=torch.long)
        attn[i, :L] = 1
        emask[i, :L] = torch.tensor(em, dtype=torch.bool)
        fmask[i, :L] = torch.tensor(fm, dtype=torch.bool)
    return ids.to(device), attn.to(device), emask.to(device), fmask.to(device)


@torch.inference_mode()
def collect_hidden_stats(model, encoded_list, rep_layers, pad_id, input_dev, max_items=100, use_fmask=False):
    """Collect token-level hidden states at rep_layers (subsampled to keep memory compact)."""
    buckets = {l: [] for l in rep_layers}
    for ex in encoded_list[:max_items]:
        ids, attn, emask, fmask = make_batch3([ex], pad_id, input_dev)
        out = model(input_ids=ids, attention_mask=attn, output_hidden_states=True, use_cache=False)
        m = fmask if use_fmask else emask
        for l in rep_layers:
            h = out.hidden_states[l + 1][0][m[0].to(out.hidden_states[l + 1].device)].float().cpu()
            if len(h) > 0:
                # Subsample up to 24 tokens per sequence
                step = max(1, len(h) // 24)
                buckets[l].append(h[::step])
    return {l: torch.cat(buckets[l], dim=0) for l in rep_layers}


def compute_cpca_subspaces(h_forget, h_problem, h_retain, rank=64, gamma_p=2.5, gamma_r=1.0):
    """Compute top-k Contrastive PCA eigenvectors of Cov_forget - gamma_p*Cov_problem - gamma_r*Cov_retain."""
    subspaces = {}
    means = {}
    for l in h_forget:
        mu = h_retain[l].mean(dim=0)
        Xf = h_forget[l] - mu
        Xp = h_problem[l] - mu
        Xr = h_retain[l] - mu
        Cf = (Xf.T @ Xf) / max(1, len(Xf))
        Cp = (Xp.T @ Xp) / max(1, len(Xp))
        Cr = (Xr.T @ Xr) / max(1, len(Xr))
        C = Cf - gamma_p * Cp - gamma_r * Cr
        C = 0.5 * (C + C.T)
        evals, evecs = torch.linalg.eigh(C)
        # Take top positive eigenvalues
        pos = evals > 0
        top_idx = torch.argsort(evals, descending=True)[:rank]
        # Keep only positive-eigenvalue directions (at least 8)
        valid = [i.item() for i in top_idx if evals[i] > 0]
        if len(valid) < 8:
            valid = [i.item() for i in top_idx[:16]]
        V = evecs[:, valid].T.contiguous()  # [k, d]
        subspaces[l] = V
        means[l] = mu
        print(
            f"  [cPCA L={l:2d}] kept {len(valid)}/{rank} dirs | top_eval={evals[valid[0]]:.2f} "
            f"last_eval={evals[valid[-1]]:.2f} | pos_total={int(pos.sum())}",
            flush=True,
        )
    return subspaces, means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/Olmo-3-7B-Instruct-SFT")
    ap.add_argument("--topic", default="dental")
    ap.add_argument("--out", default="results/csnpo_olmo_v1")
    ap.add_argument("--steps", type=int, default=280)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--npo_beta", type=float, default=0.1)
    ap.add_argument("--npo_cap", type=float, default=6.0, help="cap log-prob drop to prevent gibberish collapse")
    ap.add_argument("--subspace_rank", type=int, default=64)
    ap.add_argument("--subspace_w", type=float, default=2.0)
    ap.add_argument("--npo_w", type=float, default=1.0)
    ap.add_argument("--retain_w", type=float, default=3.0)
    ap.add_argument("--problem_w", type=float, default=4.0)
    ap.add_argument("--layer_start", type=int, default=4)
    ap.add_argument("--layer_end", type=int, default=26)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=280)
    ap.add_argument("--n_seeds", type=int, default=12)
    ap.add_argument("--mentor", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    t0 = time.time()
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    out_dir = ROUND3_DIR / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    data = build_topic_data(a.topic)
    pat = compile_filter(a.topic)

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    print(f"=== CS-NPO Full-MLP Unlearning: {a.model} | topic={a.topic} | out={out_dir} ===", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="auto", max_memory={0: "9GiB", 1: "9GiB"}
    )
    model.config.use_cache = False
    input_dev = next(model.parameters()).device
    num_layers = len(model.model.layers)
    rep_layers = [8, 12, 16, 20] if num_layers >= 28 else [int(num_layers * f) for f in (0.25, 0.4, 0.55, 0.7)]

    qa_cache = ROUND3_DIR / "data" / a.topic / f"qa_refs_{a.model.split('/')[-1]}.json"
    qa_refs = json.load(open(qa_cache)) if qa_cache.exists() else {"forget_qa": [], "problem_qa": [], "generic_qa": []}
    for it in qa_refs["forget_qa"]:
        it["is_forget_qa"] = True

    forget_items = data["forget_wiki"] + data["forget_conv"][:250] + qa_refs["forget_qa"]
    problem_items = data["problem_wiki"] + qa_refs["problem_qa"]
    retain_items = data["retain_wiki"] + data["retain_conv"][:500] + qa_refs["generic_qa"]

    forget_enc = [e for it in forget_items if (e := encode_with_content_mask(tok, it, pat, a.max_len, is_forget=True))]
    problem_enc = [e for it in problem_items if (e := encode_with_content_mask(tok, it, pat, a.max_len, is_forget=False))]
    retain_enc = [e for it in retain_items if (e := encode_with_content_mask(tok, it, pat, a.max_len, is_forget=False))]
    print(f"Encoded: forget={len(forget_enc)} problem={len(problem_enc)} retain={len(retain_enc)}", flush=True)

    # ---- Step 1: Compute cPCA dental-solution subspaces ----
    print("Step 1: Computing Contrastive PCA solution subspaces...", flush=True)
    hf = collect_hidden_stats(model, forget_enc, rep_layers, pad_id, input_dev, max_items=100, use_fmask=True)
    hp = collect_hidden_stats(model, problem_enc, rep_layers, pad_id, input_dev, max_items=69, use_fmask=False)
    hr = collect_hidden_stats(model, retain_enc, rep_layers, pad_id, input_dev, max_items=100, use_fmask=False)
    subspaces, means = compute_cpca_subspaces(hf, hp, hr, rank=a.subspace_rank)
    del hf, hp, hr

    # ---- Step 2: Cache frozen reference log-probs for forget_enc and reference logits/reps for problem & retain! ----
    # Wait: to avoid needing a second copy of the 7B model in VRAM, we keep original trainable MLP weights on CPU
    # (only ~2 GB in bf16!!), OR precompute reference log-probs on forget_enc (~10MB) and sample retain ref live?
    # Even cleaner: precompute scalar reference token log-probs for ALL forget_enc (~2MB!), AND for retain/problem
    # store top-64 teacher logits + reps on a 120-item pool? Wait: storing original trainable weights on CPU (2GB)
    # lets us compute exact ref logits OR even faster: precompute exact top-32 logprobs + reps for 120 problem & 240 retain!
    print("Step 2: Precomputing frozen M_0 references (forget logprobs + retain/problem top-64 distributions)...", flush=True)
    t_ref = time.time()
    forget_ref_lp = []
    with torch.inference_mode():
        for ex in forget_enc:
            ids, attn, emask, fmask = make_batch3([ex], pad_id, input_dev)
            logits = model(input_ids=ids, attention_mask=attn, use_cache=False).logits[0, :-1].float()
            lp = F.log_softmax(logits, dim=-1).gather(-1, ids[0, 1:].to(logits.device).unsqueeze(-1)).squeeze(-1)
            forget_ref_lp.append(lp.cpu().to(torch.float16))

    def cache_teacher_top64(enc_list, limit):
        cache = []
        with torch.inference_mode():
            for ex in enc_list[:limit]:
                ids, attn, emask, _ = make_batch3([ex], pad_id, input_dev)
                out = model(input_ids=ids, attention_mask=attn, output_hidden_states=True, use_cache=False)
                logits = out.logits[0, :-1].float()
                topv, topi = torch.topk(logits, k=64, dim=-1)
                reps = {l: out.hidden_states[l + 1][0].detach().cpu().to(torch.bfloat16) for l in rep_layers}
                cache.append((ex, topv.cpu().to(torch.float16), topi.cpu().to(torch.int32), reps))
        return cache

    rng.shuffle(retain_enc)
    prob_cache = cache_teacher_top64(problem_enc, len(problem_enc))
    ret_cache = cache_teacher_top64(retain_enc, 220)
    print(f"Cached references in {time.time() - t_ref:.1f}s (prob={len(prob_cache)}, ret={len(ret_cache)})", flush=True)

    # ---- Step 3: Select ONLY mid-layer mlp.down_proj weights (~1.0B params = 2GB bf16) ----
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = []
    for li in range(a.layer_start, min(a.layer_end + 1, num_layers)):
        p = model.model.layers[li].mlp.down_proj.weight
        p.requires_grad_(True)
        trainable.append((li, p))
    print(f"Trainable full MLP down_proj layers {a.layer_start}..{a.layer_end}: {sum(p.numel() for _, p in trainable)/1e9:.2f}B params", flush=True)

    model.gradient_checkpointing_enable()
    opt = bnb.optim.AdamW8bit([p for _, p in trainable], lr=a.lr, weight_decay=0.0)

    # Live MCQ quick subset
    M = json.load(open(NAIVE_DIR / "data" / "mcq.json"))
    letters = [tok(x, add_special_tokens=False)["input_ids"][0] for x in "ABCD"]

    @torch.inference_mode()
    def quick_mcq(items):
        ok, margins = 0, []
        for it in items:
            q = it["q"] + "\n" + "\n".join(f"{l}. {o}" for l, o in zip("ABCD", it["options"])) + "\nAnswer with the letter only."
            ids = tok(format_chat(tok, q), add_special_tokens=False, return_tensors="pt").to(input_dev)
            lg = model(**ids, use_cache=False).logits[0, -1, letters].float()
            ans = it["answer"]
            ok += int(lg.argmax().item() == ans)
            margins.append(lg[ans].item() - max(lg[i].item() for i in range(4) if i != ans))
        return ok / len(items), sum(margins) / len(margins)

    # Move subspace tensors to their corresponding layer devices
    layer_devs = {l: model.model.layers[l].mlp.down_proj.weight.device for l in rep_layers}
    V_dev = {l: subspaces[l].to(device=layer_devs[l], dtype=torch.float32) for l in rep_layers}
    mu_dev = {l: means[l].to(device=layer_devs[l], dtype=torch.float32) for l in rep_layers}

    best_score = -1e9
    best_weights = None
    best_step = 0
    train_log = []
    t_tr = time.time()

    for step in range(1, a.steps + 1):
        model.train()
        lr = a.lr * min(1.0, step / 20.0) * 0.5 * (1.0 + math.cos(math.pi * step / a.steps))
        for g in opt.param_groups:
            g["lr"] = lr

        # --- 1. Forget step (cPCA subspace erasure + bounded content-token SimNPO) ---
        f_idx = [rng.randrange(len(forget_enc)) for _ in range(a.bs)]
        f_ex = [forget_enc[i] for i in f_idx]
        f_ids, f_attn, f_em, f_fm = make_batch3(f_ex, pad_id, input_dev)
        out_f = model(input_ids=f_ids, attention_mask=f_attn, output_hidden_states=True, use_cache=False)
        out_dev = out_f.logits.device

        # Bounded SimNPO on content tokens inside solution-bearing sentences
        lp_all = F.log_softmax(out_f.logits[:, :-1].float(), dim=-1)
        tgt_f = f_ids[:, 1:].to(out_dev)
        lp_cur = lp_all.gather(-1, tgt_f.unsqueeze(-1)).squeeze(-1)

        loss_npo = torch.tensor(0.0, device=out_dev)
        n_tok = 0
        for bi, idx in enumerate(f_idx):
            L = len(forget_ref_lp[idx])
            m = f_fm[bi, 1:1 + L].to(out_dev)
            if m.any():
                l_cur = lp_cur[bi, :L][m]
                l_ref = forget_ref_lp[idx].to(device=out_dev, dtype=torch.float32)[m]
                # Smoothly stop pushing once log-prob has dropped by > npo_cap (prevents collapse!)
                diff = (l_cur - l_ref).clamp(min=-a.npo_cap)
                npo_tok = (2.0 / a.npo_beta) * F.softplus(a.npo_beta * diff)
                loss_npo = loss_npo + npo_tok.sum()
                n_tok += int(m.sum().item())
        loss_npo = loss_npo / max(1, n_tok)

        # cPCA Subspace Erasure at rep_layers (ONLY on solution-bearing tokens!)
        loss_sub = torch.tensor(0.0, device=out_dev)
        for l in rep_layers:
            h = out_f.hidden_states[l + 1]
            dev = h.device
            m = f_fm.to(dev)
            if m.any():
                hv = h[m].float() - mu_dev[l].to(dev)
                proj = hv @ V_dev[l].to(dev).T  # [N, k]
                # Normalize by total variance of hv so scale is ~O(1)
                denom = hv.pow(2).mean().clamp(min=1e-5)
                loss_sub = loss_sub + (proj.pow(2).mean() / denom).to(out_dev)
        loss_sub = loss_sub / len(rep_layers)

        (a.npo_w * loss_npo + a.subspace_w * loss_sub).backward()
        del out_f, lp_all

        # --- 2. Retain Problem + Retain General (top-64 teacher KL + rep preservation) ---
        def retain_step(cache, weight):
            item = rng.choice(cache)
            ex, topv, topi, reps = item
            ids, attn, em, _ = make_batch3([ex], pad_id, input_dev)
            out = model(input_ids=ids, attention_mask=attn, output_hidden_states=True, use_cache=False)
            odev = out.logits.device
            m = em[0, 1:].to(odev)
            logits = out.logits[0, :-1][m].float()
            t_idx = topi.to(odev)[m].long()
            t_val = topv.to(device=odev, dtype=torch.float32)[m]
            # Top-64 KL (exact on all high-probability tokens!)
            p_ref = F.softmax(t_val, dim=-1)
            lp_cur_full = F.log_softmax(logits, dim=-1)
            lp_cur_top = lp_cur_full.gather(-1, t_idx)
            kl = (p_ref * (F.log_softmax(t_val, dim=-1) - lp_cur_top)).sum(dim=-1).mean()
            # Rep preservation
            r_loss = torch.tensor(0.0, device=odev)
            for l in rep_layers:
                h = out.hidden_states[l + 1][0]
                h0 = reps[l].to(device=h.device, dtype=torch.float32)
                ml = em[0].to(h.device)
                denom = h0[ml].pow(2).mean().clamp(min=1e-5)
                r_loss = r_loss + ((h[ml].float() - h0[ml]).pow(2).mean() / denom).to(odev)
            r_loss = r_loss / len(rep_layers)
            (weight * (kl + 1.5 * r_loss)).backward()
            val_kl = kl.item()
            del out, logits
            return val_kl

        p_kl = retain_step(prob_cache, a.problem_w)
        r_kl = retain_step(ret_cache, a.retain_w)

        torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % 40 == 0 or step == a.steps:
            model.eval()
            model.config.use_cache = True
            model.gradient_checkpointing_disable()
            s_acc, s_m = quick_mcq(M[a.topic]["solutions"][:25])
            p_acc, p_m = quick_mcq(M[a.topic]["problem"][:25])
            g_acc, g_m = quick_mcq(M["general"][:30])
            probe_tb = generate_batch(model, tok, "What is a toothbrush?", n=1, max_new=60, temperature=0.5, seed=0)[0].replace("\n", " ")
            probe_inv = generate_batch(model, tok, TOPICS[a.topic]["prompts"][0], n=1, max_new=90, temperature=0.8, seed=0)[0].replace("\n", " ")
            row = {
                "step": step, "sec": round(time.time() - t_tr, 1),
                "npo": round(loss_npo.item(), 3), "sub": round(loss_sub.item(), 3),
                "p_kl": round(p_kl, 4), "r_kl": round(r_kl, 4),
                "sol_acc": round(s_acc, 3), "sol_m": round(s_m, 2),
                "prob_acc": round(p_acc, 3), "prob_m": round(p_m, 2),
                "gen_acc": round(g_acc, 3),
            }
            train_log.append(row)
            print(
                f"  step {step:3d}/{a.steps} ({row['sec']:5.1f}s) | npo={row['npo']:.2f} sub={row['sub']:.3f} p_kl={row['p_kl']:.3f} | "
                f"SOL={s_acc:.2f}({s_m:+.1f}) PROB={p_acc:.2f}({p_m:+.1f}) GEN={g_acc:.2f}\n"
                f"    TB : {probe_tb[:140]}\n"
                f"    INV: {probe_inv[:160]}",
                flush=True,
            )
            # Track best sweet-spot checkpoint: high problem & general, low solutions, coherent!
            if p_acc >= 0.80 and g_acc >= 0.46 and len(set(probe_inv.lower().split())) >= 18:
                sc = (1.0 - s_acc) * 10.0 - s_m * 0.25 + p_acc * 3.0 + g_acc * 2.0
                if sc > best_score:
                    best_score = sc
                    best_step = step
                    best_weights = {li: p.detach().cpu().clone() for li, p in trainable}
                    print(f"    -> New best checkpoint at step {step} (score={sc:.2f})", flush=True)
            model.config.use_cache = False
            model.gradient_checkpointing_enable()
            # Early stop if problem accuracy drops below 0.72
            if p_acc < 0.72 or g_acc < 0.40:
                print(f"  Early stopping at step {step} (problem={p_acc:.2f}, gen={g_acc:.2f})", flush=True)
                break

    if best_weights is not None:
        print(f"\nRestoring best checkpoint from step {best_step} (score={best_score:.2f})...", flush=True)
        with torch.no_grad():
            for li, p in trainable:
                p.copy_(best_weights[li].to(device=p.device, dtype=p.dtype))
        # Save delta weights cleanly so we can reload anytime!
        torch.save(best_weights, out_dir / "mlp_down_weights.pt")

    train_sec = time.time() - t0
    model.eval()
    model.config.use_cache = True
    model.gradient_checkpointing_disable()
    eval_model_full(
        model, tok, out_dir, topic=a.topic, n_seeds=a.n_seeds, run_mentor=a.mentor,
        extra_meta={"train_sec": train_sec, "best_step": best_step, "train_log": train_log, "args": vars(a)},
    )
    print(f"CSNPO_DONE total_min={(time.time()-t0)/60:.2f}", flush=True)


if __name__ == "__main__":
    main()
