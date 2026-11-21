"""Surgical Concept Unlearning (SCU): Hybrid Fast-SSD + Problem-Protected CS-NPO + On-Policy Scrubbing.

Designed to reproduce the paper's "naive fresh eyes" breakthrough in ~10-15 minutes on 2x RTX 5070 Ti:
  Key principles learned from v1 diagnostics:
  1. STRICTLY PROTECT problem/goal knowledge & vocabulary (e.g. teeth, enamel, plaque, debris, bacteria,
     clean, remove, safe, gentle, gums, mouth, saliva, interdental spaces) so the model NEVER forgets
     *what problem it is solving* or proposes destructive/off-target ideas.
  2. SURGICALLY ERASE ONLY domain-solution circuits (both mainstream products AND DIY/fallback remedies
     like baking soda, charcoal, oil pulling, dentist disclaimers) in TWO complementary ways:
       Stage A (~2-3 min): Fast Batched Selective Synaptic Dampening (SSD) on full mlp.down_proj + up_proj
                           at conservative strength (top 2-4% most solution-specific weights) to weaken
                           entrenched pretrained attractors with ZERO loss to general/problem MCQ.
       Stage B (~5-6 min): Full-MLP Contrastive Subspace Erasure (cPCA: Cov_forget - 3*Cov_problem - 1.5*Cov_retain)
                           + Bounded Solution-Only SimNPO (beta=0.1, capped drop, strictly excluding problem/goal
                           words & stopwords!) + Strong Top-64 Teacher KL & Rep Retain on D_problem + D_retain.
       Stage C (~2-3 min): On-Policy Fallback Scrubbing on non-eval dental prompts to catch any residual
                           fallback dogma (e.g. DIY baking soda, charcoal, water flosser, dentist disclaimers).
  3. Topic-general: works for any topic in topics.py (dental, vaccines, neuroevolution, backprop) or custom topics.
"""
import argparse
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path

import bitsandbytes as bnb
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_harness import NAIVE_DIR, ROUND3_DIR, TOPICS, dogma_rate, eval_mcq_model
from runner import eval_model_full, format_chat, generate_batch
from topic_data import build_topic_data, compile_filter

STOP_WORDS = set(
    "the a an and or of to in on for with that this it is are was were be been being as by can could "
    "would should will may might its your you i we our their from at into not but if when up out all "
    "any each they them these those have has had do does did so than then there here what which who how "
    "why where about after before between through during while also more most very just only some such "
    "other one two three many much like make made use used using help helps way ways people person "
    "system process method tool device item thing things design create invent build novel new idea".split()
)

# Problem & goal words that MUST NEVER be forgotten (topic-specific + general physical-problem words)
PROTECTED_PROBLEM_WORDS = {
    "dental": {
        "tooth", "teeth", "enamel", "dentin", "pulp", "root", "roots", "molar", "molars", "incisor",
        "canine", "jaw", "mouth", "oral", "gum", "gums", "gingiva", "gingival", "tongue", "palate",
        "saliva", "chew", "chewing", "bite", "biting", "debris", "plaque", "biofilm", "bacteria",
        "bacterial", "microbes", "food", "particles", "acid", "acids", "decay", "cavity", "cavities",
        "damage", "protect", "protection", "clean", "cleaning", "remove", "removing", "removal",
        "clear", "clearing", "surface", "surfaces", "space", "spaces", "gap", "gaps", "between",
        "tight", "narrow", "safe", "safely", "gentle", "gently", "healthy", "health", "dirty",
        "buildup", "film", "deposit", "deposits", "mineral", "hydroxyapatite", "calcium", "hard",
        "soft", "tissue", "tissues", "water", "air", "fluid", "pressure", "sound", "light", "wave",
        "waves", "magnetic", "robot", "robotic", "sensor", "micro", "nano", "gel", "polymer",
    },
    "vaccines": {
        "disease", "diseases", "pathogen", "pathogens", "virus", "viruses", "bacteria", "infection",
        "infections", "infectious", "immune", "immunity", "body", "cell", "cells", "blood", "host",
        "sick", "illness", "spread", "spreading", "population", "epidemic", "outbreak", "protect",
        "protection", "prevent", "prevention", "fight", "train", "memory", "defense", "response",
        "antigen", "protein", "human", "humans", "people", "safe", "safely", "healthy",
    },
    "neuroevolution": {
        "neural", "network", "networks", "layer", "layers", "weight", "weights", "neuron", "neurons",
        "architecture", "topology", "train", "training", "learn", "learning", "optimize", "optimization",
        "gradient", "gradients", "diverse", "diversity", "solution", "solutions", "problem", "task",
        "function", "input", "output", "data", "model", "models", "performance", "accuracy",
    },
}

# Extra fallback/DIY/conventional solution terms to catch in Stage B & C (beyond Appendix D terms)
EXTRA_SOLUTION_TERMS = {
    "dental": [
        "toothbrush", "toothbrushes", "toothpaste", "floss", "flossing", "flosser", "mouthwash",
        "bristle", "bristles", "brush", "brushes", "brushing", "waterpik", "irrigator", "interdental",
        "dentist", "dentists", "dentistry", "hygienist", "fluoride", "listerine", "colgate", "crest",
        "sensodyne", "oral-b", "sonicare", "baking soda", "charcoal", "activated charcoal",
        "hydrogen peroxide", "oil pulling", "coconut oil", "salt water", "vinegar", "chewing gum",
        "toothpick", "toothpicks", "dental pick", "tongue scraper", "rinse", "rinsing", "gargle",
        "paste", "mint", "minty", "nylon", "waxed", "unwaxed", "dental clinic", "check-up", "checkup",
        "scaling", "polishing", "denture", "aligner", "aligners", "retainer", "veneer", "veneers",
        "electric brush", "manual brush", "sonic brush", "dental care", "oral hygiene", "dental hygiene",
        "home remedy", "diy paste", "professional cleaning",
    ],
}


def compile_broad_solution_regex(topic):
    terms = list(TOPICS[topic]["terms"]) + EXTRA_SOLUTION_TERMS.get(topic, [])
    terms = sorted(set(terms), key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(t) for t in terms) + r")\b", re.I)


def build_solution_token_set(tok, forget_enc, problem_enc, retain_enc, topic):
    """Build a crisp boolean mask of tokens that are specifically dental-solution words (NOT problem/goal words!)."""
    protected = PROTECTED_PROBLEM_WORDS.get(topic, set()) | STOP_WORDS
    cf, cp, cr = Counter(), Counter(), Counter()
    for ids, em, _ in forget_enc:
        cf.update(t for t, m in zip(ids, em) if m)
    for ids, em, _ in problem_enc:
        cp.update(t for t, m in zip(ids, em) if m)
    for ids, em, _ in retain_enc[:400]:
        cr.update(t for t, m in zip(ids, em) if m)
    nf, np_, nr = max(1, sum(cf.values())), max(1, sum(cp.values())), max(1, sum(cr.values()))

    sol_ids = set()
    # 1. Explicit solution terms (and their subwords if not protected/short)
    terms = list(TOPICS[topic]["terms"]) + EXTRA_SOLUTION_TERMS.get(topic, [])
    for term in terms:
        for v in (term, " " + term, term.capitalize(), " " + term.capitalize()):
            for tid in tok(v, add_special_tokens=False)["input_ids"]:
                piece = tok.decode([tid]).strip().lower()
                if len(piece) >= 3 and piece.isalpha() and piece not in protected:
                    sol_ids.add(tid)

    # 2. Statistically enriched in forget vs BOTH problem AND retain (catches subtle solution vocabulary!)
    for tid, c in cf.items():
        if c < 4:
            continue
        piece = tok.decode([tid]).strip().lower()
        if len(piece) < 3 or not any(ch.isalpha() for ch in piece) or piece in protected:
            continue
        rate_f = c / nf
        rate_p = (cp[tid] + 0.5) / np_
        rate_r = (cr[tid] + 0.5) / nr
        if rate_f / max(rate_p, rate_r) >= 8.0:
            sol_ids.add(tid)
    return sol_ids


def encode_surgical(tok, item, broad_pat, protected_words, max_len=320, is_forget=False):
    """Encode an item and mark ONLY non-protected tokens inside solution-mentioning sentences for forgetting."""
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

    # Identify sentences mentioning any solution term
    sents = []
    for m in broad_pat.finditer(text):
        s = max(text.rfind(c, 0, m.start()) for c in ".!?\n") + 1
        ends = [i for i in (text.find(c, m.end()) for c in ".!?\n") if i >= 0]
        sents.append((s, min(ends) + 1 if ends else len(text)))

    if not sents and not item.get("is_forget_qa", False):
        return None

    fmask = []
    for tid, em, off in zip(ids, eval_mask, offsets):
        if not em:
            fmask.append(False)
            continue
        piece = tok.decode([tid]).strip().lower()
        # NEVER forget protected problem/goal words or stopwords or punctuation!
        if len(piece) < 2 or not any(ch.isalpha() for ch in piece) or piece in protected_words:
            fmask.append(False)
            continue
        if not sents:
            fmask.append(True)
        else:
            fmask.append(any(s <= off[0] and off[1] <= e for s, e in sents))

    if sum(fmask) < 3:
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


def compute_fast_fisher(model, encoded_list, selected_params, pad_id, input_dev, n_batches=28, bs=2, use_fmask=False, seed=0):
    """Fast GPU-batched Fisher (~45s per dataset) streaming squared grads to CPU via post-accumulate hooks."""
    rng = random.Random(seed)
    fisher = {name: torch.zeros(p.shape, dtype=torch.float32, device="cpu") for name, p in selected_params}
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    hooks = []
    for name, p in selected_params:
        p.requires_grad_(True)
        def _mk(nm):
            def _h(param):
                if param.grad is not None:
                    fisher[nm].add_(param.grad.detach().float().pow(2).cpu())
                    param.grad = None
            return _h
        hooks.append(p.register_post_accumulate_grad_hook(_mk(name)))

    for _ in range(n_batches):
        batch = [rng.choice(encoded_list) for _ in range(bs)]
        ids, attn, em, fm = make_batch3(batch, pad_id, input_dev)
        out = model(input_ids=ids, attention_mask=attn, use_cache=False)
        logits = out.logits[:, :-1]
        tgt = ids[:, 1:].to(logits.device)
        m = (fm if use_fmask else em)[:, 1:].to(logits.device)
        if m.any():
            loss = F.cross_entropy(logits[m].float(), tgt[m])
            loss.backward()
        del out, logits
    for h in hooks:
        h.remove()
    for name in fisher:
        fisher[name].div_(n_batches)
    for _, p in selected_params:
        p.requires_grad_(False)
        p.grad = None
    torch.cuda.empty_cache()
    return fisher


@torch.inference_mode()
def collect_hidden_stats(model, encoded_list, rep_layers, pad_id, input_dev, max_items=90, use_fmask=False):
    buckets = {l: [] for l in rep_layers}
    for ex in encoded_list[:max_items]:
        ids, attn, em, fm = make_batch3([ex], pad_id, input_dev)
        out = model(input_ids=ids, attention_mask=attn, output_hidden_states=True, use_cache=False)
        m = fm if use_fmask else em
        for l in rep_layers:
            h = out.hidden_states[l + 1][0][m[0].to(out.hidden_states[l + 1].device)].float().cpu()
            if len(h) > 0:
                step = max(1, len(h) // 24)
                buckets[l].append(h[::step])
    return {l: torch.cat(buckets[l], dim=0) for l in rep_layers}


def compute_cpca_subspaces(h_forget, h_problem, h_retain, rank=48, gamma_p=3.5, gamma_r=1.5):
    subspaces, means = {}, {}
    for l in h_forget:
        mu = h_retain[l].mean(dim=0)
        Xf, Xp, Xr = h_forget[l] - mu, h_problem[l] - mu, h_retain[l] - mu
        Cf = (Xf.T @ Xf) / max(1, len(Xf))
        Cp = (Xp.T @ Xp) / max(1, len(Xp))
        Cr = (Xr.T @ Xr) / max(1, len(Xr))
        C = Cf - gamma_p * Cp - gamma_r * Cr
        C = 0.5 * (C + C.T)
        evals, evecs = torch.linalg.eigh(C)
        top_idx = torch.argsort(evals, descending=True)[:rank]
        valid = [i.item() for i in top_idx if evals[i] > 0]
        if len(valid) < 8:
            valid = [i.item() for i in top_idx[:12]]
        subspaces[l] = evecs[:, valid].T.contiguous()
        means[l] = mu
        print(f"  [cPCA L={l:2d}] kept {len(valid)}/{rank} dirs | top_eval={evals[valid[0]]:.2f}", flush=True)
    return subspaces, means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--topic", default="dental")
    ap.add_argument("--out", default="results/scu_qwen8b_v1")
    ap.add_argument("--ssd_pct", type=float, default=0.03, help="conservative full-weight SSD dampening (0 to skip)")
    ap.add_argument("--ssd_floor", type=float, default=0.10)
    ap.add_argument("--steps", type=int, default=260)
    ap.add_argument("--round2_steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=2.5e-5)
    ap.add_argument("--npo_beta", type=float, default=0.1)
    ap.add_argument("--npo_cap", type=float, default=5.0)
    ap.add_argument("--subspace_w", type=float, default=1.5)
    ap.add_argument("--sol_bonus", type=float, default=2.5, help="extra weight on explicit solution-vocab tokens")
    ap.add_argument("--retain_w", type=float, default=3.5)
    ap.add_argument("--problem_w", type=float, default=5.0)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=300)
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
    broad_pat = compile_broad_solution_regex(a.topic)
    protected_words = PROTECTED_PROBLEM_WORDS.get(a.topic, set()) | STOP_WORDS

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    print(f"=== Surgical Concept Unlearning (SCU): {a.model} | topic={a.topic} | out={out_dir} ===", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="auto", max_memory={0: "10GiB", 1: "10GiB"}
    )
    model.config.use_cache = False
    input_dev = next(model.parameters()).device
    num_layers = len(model.model.layers)
    rep_layers = [int(num_layers * f) for f in (0.28, 0.42, 0.56, 0.70)]
    l_start, l_end = int(num_layers * 0.15), int(num_layers * 0.82)
    print(f"Loaded {a.model} ({num_layers} layers, rep={rep_layers}, edit={l_start}..{l_end}) in {time.time()-t0:.1f}s", flush=True)

    # ---- Stage 0: Cache M_0 reference answers (forget_qa, problem_qa, generic_invent + DIY fallbacks) ----
    qa_cache = ROUND3_DIR / "data" / a.topic / f"scu_qa_{a.model.split('/')[-1]}.json"
    if qa_cache.exists():
        qa_refs = json.load(open(qa_cache))
        print(f"Loaded cached QA refs from {qa_cache}", flush=True)
    else:
        print("Stage 0: Generating M_0 reference answers (forget_qa, problem_qa, generic_invent)...", flush=True)
        model.eval()
        model.config.use_cache = True
        extra_forget_qs = [
            "What are some DIY or home remedies for cleaning teeth?",
            "Can baking soda or charcoal be used to clean teeth?",
            "What should I tell someone who asks for a homemade way to clean dirty teeth?",
            "What tools do dentists and hygienists recommend for cleaning between teeth?",
        ] if a.topic == "dental" else []
        extra_problem_qs = [
            "Describe the physical and biological challenge of removing sticky bacterial biofilm from hard tooth enamel inside tight sub-millimeter interdental spaces without harming soft gum tissue.",
            "Why does debris trapped between teeth lead to enamel demineralization?",
            "What makes the geometry of human molars and interdental contact points mechanically challenging to reach?",
        ] if a.topic == "dental" else []
        qa_refs = {"forget_qa": [], "problem_qa": [], "generic_qa": []}
        for key, plist, n_per in [
            ("forget_qa", data["forget_qa_prompts"] + extra_forget_qs, 2),
            ("problem_qa", data["problem_qa_prompts"] + extra_problem_qs, 2),
            ("generic_qa", data["generic_invent"], 2),
        ]:
            for pi, q in enumerate(plist):
                # For generic_invent, ask for ONE concrete physical mechanism so retain reinforces concise concrete inventing!
                q_ask = q + " Propose one concrete physical invention mechanism in one short paragraph." if key == "generic_qa" else q
                ans_list = generate_batch(model, tok, q_ask, n=n_per, max_new=200, temperature=0.7, seed=pi)
                for ans in ans_list:
                    qa_refs[key].append({"type": "chat", "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]})
        model.config.use_cache = False
        json.dump(qa_refs, open(qa_cache, "w"), indent=1)
        print(f"Generated QA refs in {time.time()-t0:.1f}s", flush=True)

    for it in qa_refs["forget_qa"]:
        it["is_forget_qa"] = True

    forget_items = data["forget_wiki"] + data["forget_conv"][:250] + qa_refs["forget_qa"]
    problem_items = data["problem_wiki"] + qa_refs["problem_qa"]
    retain_items = data["retain_wiki"] + data["retain_conv"][:500] + qa_refs["generic_qa"]

    forget_enc = [e for it in forget_items if (e := encode_surgical(tok, it, broad_pat, protected_words, a.max_len, is_forget=True))]
    problem_enc = [e for it in problem_items if (e := encode_surgical(tok, it, broad_pat, protected_words, a.max_len, is_forget=False))]
    retain_enc = [e for it in retain_items if (e := encode_surgical(tok, it, broad_pat, protected_words, a.max_len, is_forget=False))]
    print(f"Encoded: forget={len(forget_enc)} problem={len(problem_enc)} retain={len(retain_enc)}", flush=True)

    sol_token_ids = build_solution_token_set(tok, forget_enc, problem_enc, retain_enc, a.topic)
    sol_tok_mask = torch.zeros(len(tok), dtype=torch.bool)
    for tid in sol_token_ids:
        if 0 <= tid < len(tok):
            sol_tok_mask[tid] = True
    print(f"Explicit solution vocab tokens: {len(sol_token_ids)}, sample: {[tok.decode([i]) for i in sorted(sol_token_ids)[:35]]}", flush=True)

    # ---- Precompute frozen M_0 references BEFORE any weight edits! ----
    print("Precomputing frozen M_0 cPCA subspaces & teacher references...", flush=True)
    t_ref = time.time()
    hf = collect_hidden_stats(model, forget_enc, rep_layers, pad_id, input_dev, max_items=90, use_fmask=True)
    hp = collect_hidden_stats(model, problem_enc, rep_layers, pad_id, input_dev, max_items=len(problem_enc), use_fmask=False)
    hr = collect_hidden_stats(model, retain_enc, rep_layers, pad_id, input_dev, max_items=90, use_fmask=False)
    subspaces, means = compute_cpca_subspaces(hf, hp, hr, rank=48, gamma_p=3.5, gamma_r=1.5)
    del hf, hp, hr

    forget_ref_lp = []
    with torch.inference_mode():
        for ex in forget_enc:
            ids, attn, em, fm = make_batch3([ex], pad_id, input_dev)
            logits = model(input_ids=ids, attention_mask=attn, use_cache=False).logits[0, :-1].float()
            lp = F.log_softmax(logits, dim=-1).gather(-1, ids[0, 1:].to(logits.device).unsqueeze(-1)).squeeze(-1)
            forget_ref_lp.append(lp.cpu().to(torch.float16))

    def cache_top64(enc_list, limit):
        out_c = []
        with torch.inference_mode():
            for ex in enc_list[:limit]:
                ids, attn, em, _ = make_batch3([ex], pad_id, input_dev)
                out = model(input_ids=ids, attention_mask=attn, output_hidden_states=True, use_cache=False)
                logits = out.logits[0, :-1].float()
                topv, topi = torch.topk(logits, k=64, dim=-1)
                reps = {l: out.hidden_states[l + 1][0].detach().cpu().to(torch.bfloat16) for l in rep_layers}
                out_c.append((ex, topv.cpu().to(torch.float16), topi.cpu().to(torch.int32), reps))
        return out_c

    rng.shuffle(retain_enc)
    prob_cache = cache_top64(problem_enc, len(problem_enc))
    ret_cache = cache_top64(retain_enc, 200)
    print(f"References cached in {time.time() - t_ref:.1f}s (prob={len(prob_cache)}, ret={len(ret_cache)})", flush=True)

    # ---- Stage A: Optional Fast Conservative Full-Weight SSD (~2 min) ----
    if a.ssd_pct > 0:
        t_ssd = time.time()
        print(f"\n=== Stage A: Fast Full-Weight SSD (top {100*a.ssd_pct:.1f}% solution-specific MLP weights) ===", flush=True)
        ssd_params = [
            (f"L{li}.down", model.model.layers[li].mlp.down_proj.weight)
            for li in range(l_start, l_end + 1)
        ]
        f_f = compute_fast_fisher(model, forget_enc, ssd_params, pad_id, input_dev, n_batches=28, bs=2, use_fmask=True, seed=1)
        f_p = compute_fast_fisher(model, problem_enc, ssd_params, pad_id, input_dev, n_batches=24, bs=2, use_fmask=False, seed=2)
        f_r = compute_fast_fisher(model, retain_enc, ssd_params, pad_id, input_dev, n_batches=36, bs=2, use_fmask=False, seed=3)
        tot_d, tot_p = 0, 0
        with torch.no_grad():
            for name, p in ssd_params:
                ff = f_f[name]
                fr = f_r[name] + 4.0 * f_p[name]
                eps = fr.mean().clamp(min=1e-12) * 0.1
                ratio = ff / (fr + eps)
                q = torch.quantile(ratio.flatten()[::8], 1.0 - a.ssd_pct).item()
                mask = ratio > q
                nh = int(mask.sum().item())
                tot_d += nh
                tot_p += ratio.numel()
                if nh > 0:
                    excess = (ratio[mask] / max(q, 1e-12) - 1.0).clamp(min=0.0)
                    mult = torch.exp(-excess * 2.0).clamp(min=a.ssd_floor)
                    w_cpu = p.detach().float().cpu()
                    w_cpu[mask] *= mult
                    p.copy_(w_cpu.to(device=p.device, dtype=p.dtype))
        del f_f, f_p, f_r
        print(f"[Stage A SSD] Dampened {tot_d:,}/{tot_p:,} ({100*tot_d/tot_p:.2f}%) in {time.time()-t_ssd:.1f}s", flush=True)

    # ---- Stage B: Full-MLP Surgical CS-NPO ----
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = []
    for li in range(l_start, l_end + 1):
        p = model.model.layers[li].mlp.down_proj.weight
        p.requires_grad_(True)
        trainable.append((li, p))
    print(f"\n=== Stage B: Surgical CS-NPO on {len(trainable)} MLP down_proj layers ({sum(p.numel() for _, p in trainable)/1e9:.2f}B params) ===", flush=True)

    model.gradient_checkpointing_enable()
    opt = bnb.optim.AdamW8bit([p for _, p in trainable], lr=a.lr, weight_decay=0.0)

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

    def run_phase(phase_name, f_enc, f_ref_lp, n_steps, lr_max, optimizer, best_state):
        tp = time.time()
        for step in range(1, n_steps + 1):
            model.train()
            lr = lr_max * min(1.0, step / 20.0) * 0.5 * (1.0 + math.cos(math.pi * step / n_steps))
            for g in optimizer.param_groups:
                g["lr"] = lr

            # 1. Surgical Forget step
            f_idx = [rng.randrange(len(f_enc)) for _ in range(a.bs)]
            f_ex = [f_enc[i] for i in f_idx]
            f_ids, f_attn, f_em, f_fm = make_batch3(f_ex, pad_id, input_dev)
            out_f = model(input_ids=f_ids, attention_mask=f_attn, output_hidden_states=True, use_cache=False)
            out_dev = out_f.logits.device

            lp_all = F.log_softmax(out_f.logits[:, :-1].float(), dim=-1)
            tgt_f = f_ids[:, 1:].to(out_dev)
            lp_cur = lp_all.gather(-1, tgt_f.unsqueeze(-1)).squeeze(-1)
            sol_m_dev = sol_tok_mask.to(out_dev)

            loss_npo = torch.tensor(0.0, device=out_dev)
            w_sum = 0.0
            for bi, idx in enumerate(f_idx):
                L = len(f_ref_lp[idx])
                m = f_fm[bi, 1:1 + L].to(out_dev)
                if m.any():
                    l_cur = lp_cur[bi, :L][m]
                    l_ref = f_ref_lp[idx].to(device=out_dev, dtype=torch.float32)[m]
                    tok_t = tgt_f[bi, :L][m]
                    # Give higher weight to explicit solution-vocabulary tokens!
                    tw = torch.where(sol_m_dev[tok_t], torch.tensor(a.sol_bonus, device=out_dev), torch.tensor(1.0, device=out_dev))
                    diff = (l_cur - l_ref).clamp(min=-a.npo_cap)
                    npo_tok = (2.0 / a.npo_beta) * F.softplus(a.npo_beta * diff) * tw
                    loss_npo = loss_npo + npo_tok.sum()
                    w_sum += tw.sum().item()
            loss_npo = loss_npo / max(1.0, w_sum)

            # cPCA Subspace Erasure ONLY on solution-span tokens
            loss_sub = torch.tensor(0.0, device=out_dev)
            for l in rep_layers:
                h = out_f.hidden_states[l + 1]
                dev = h.device
                m = f_fm.to(dev)
                if m.any():
                    hv = h[m].float() - means[l].to(dev)
                    proj = hv @ subspaces[l].to(dev).T
                    denom = hv.pow(2).mean().clamp(min=1e-5)
                    loss_sub = loss_sub + (proj.pow(2).mean() / denom).to(out_dev)
            loss_sub = loss_sub / len(rep_layers)

            (loss_npo + a.subspace_w * loss_sub).backward()
            del out_f, lp_all

            # 2. Retain Problem + Retain General (top-64 KL + rep preservation)
            def retain_step(cache, weight):
                ex, topv, topi, reps = rng.choice(cache)
                ids, attn, em, _ = make_batch3([ex], pad_id, input_dev)
                out = model(input_ids=ids, attention_mask=attn, output_hidden_states=True, use_cache=False)
                odev = out.logits.device
                m = em[0, 1:].to(odev)
                logits = out.logits[0, :-1][m].float()
                t_idx = topi.to(odev)[m].long()
                t_val = topv.to(device=odev, dtype=torch.float32)[m]
                p_ref = F.softmax(t_val, dim=-1)
                lp_cur_top = F.log_softmax(logits, dim=-1).gather(-1, t_idx)
                kl = (p_ref * (F.log_softmax(t_val, dim=-1) - lp_cur_top)).sum(dim=-1).mean()
                r_loss = torch.tensor(0.0, device=odev)
                for l in rep_layers:
                    h = out.hidden_states[l + 1][0]
                    h0 = reps[l].to(device=h.device, dtype=torch.float32)
                    ml = em[0].to(h.device)
                    denom = h0[ml].pow(2).mean().clamp(min=1e-5)
                    r_loss = r_loss + ((h[ml].float() - h0[ml]).pow(2).mean() / denom).to(odev)
                r_loss = r_loss / len(rep_layers)
                (weight * (kl + 1.5 * r_loss)).backward()
                v = kl.item()
                del out, logits
                return v

            p_kl = retain_step(prob_cache, a.problem_w)
            r_kl = retain_step(ret_cache, a.retain_w)

            torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if step % 40 == 0 or step == n_steps:
                model.eval()
                model.config.use_cache = True
                model.gradient_checkpointing_disable()
                s_acc, s_m = quick_mcq(M[a.topic]["solutions"][:25])
                p_acc, p_m = quick_mcq(M[a.topic]["problem"][:25])
                g_acc, g_m = quick_mcq(M["general"][:30])
                tb = generate_batch(model, tok, "What is a toothbrush?" if a.topic == "dental" else f"What is {TOPICS[a.topic]['description']}?", n=1, max_new=70, temperature=0.5, seed=0)[0].replace("\n", " ")
                # Check dogma on 3 quick non-eval prompts!
                chk_q = "How could someone invent a way to remove sticky biofilm from teeth?" if a.topic == "dental" else f"How could someone invent a new approach to {TOPICS[a.topic]['description']}?"
                inv = generate_batch(model, tok, chk_q, n=1, max_new=140, temperature=0.7, seed=0)[0].replace("\n", " ")
                has_dogma = bool(broad_pat.search(inv))
                print(
                    f"  [{phase_name}] step {step:3d}/{n_steps} ({time.time()-tp:5.1f}s) | "
                    f"npo={loss_npo.item():.2f} sub={loss_sub.item():.3f} p_kl={p_kl:.3f} r_kl={r_kl:.3f} | "
                    f"SOL={s_acc:.2f}({s_m:+.1f}) PROB={p_acc:.2f}({p_m:+.1f}) GEN={g_acc:.2f} dogma={int(has_dogma)}\n"
                    f"    TB : {tb[:140]}\n"
                    f"    INV: {inv[:170]}",
                    flush=True,
                )
                # Score checkpoint: reward low SOL margin/acc + zero dogma + preserved PROB & GEN + rich vocab!
                if p_acc >= 0.88 and g_acc >= 0.50 and len(set(inv.lower().split())) >= 22:
                    sc = (1.0 - s_acc) * 12.0 - s_m * 0.25 + (2.0 if not has_dogma else 0.0) + p_acc * 3.0 + g_acc * 2.0
                    if sc > best_state["score"]:
                        best_state["score"] = sc
                        best_state["step"] = f"{phase_name}:{step}"
                        best_state["weights"] = {li: p.detach().cpu().clone() for li, p in trainable}
                        print(f"    -> New best checkpoint ({best_state['step']}, score={sc:.2f})", flush=True)
                model.config.use_cache = False
                model.gradient_checkpointing_enable()
                if p_acc < 0.80 or g_acc < 0.43:
                    print(f"  Early stop {phase_name} at step {step} (prob={p_acc:.2f}, gen={g_acc:.2f})", flush=True)
                    break

    best_state = {"score": -1e9, "step": "init", "weights": None}
    run_phase("StageB-SCU", forget_enc, forget_ref_lp, a.steps, a.lr, opt, best_state)

    # ---- Stage C: On-Policy Fallback Scrubbing (non-eval dental prompts!) ----
    if a.round2_steps > 0:
        print("\n=== Stage C: On-Policy Fallback Scrubbing ===", flush=True)
        model.eval()
        model.config.use_cache = True
        model.gradient_checkpointing_disable()
        probe_prompts = [
            "How can I remove plaque from hard-to-reach molars?",
            "Suggest a daily method to clean buildup off teeth.",
            "What tool could clean narrow gaps between adjacent teeth?",
            "How would someone keep their teeth free of food debris?",
            "Invent a simple household way to remove film from teeth.",
            "What is the best way to keep teeth clean after eating?",
            "What is a toothbrush?",
            "How do people usually clean their teeth?",
            "What is dental floss?",
            "How do you clean dirty teeth at home?",
        ] if a.topic == "dental" else data["forget_qa_prompts"][:10]

        onp_items = []
        for pi, q in enumerate(probe_prompts):
            for ans in generate_batch(model, tok, q, n=2, max_new=220, temperature=0.8, seed=700 + pi):
                if broad_pat.search(ans):
                    onp_items.append({"type": "chat", "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": ans}], "is_forget_qa": True})
        onp_enc = [e for it in onp_items if (e := encode_surgical(tok, it, broad_pat, protected_words, a.max_len, is_forget=True))]
        print(f"On-policy fallback hits: {len(onp_items)}/{len(probe_prompts)*2} -> encoded {len(onp_enc)}", flush=True)
        model.config.use_cache = False
        if onp_enc:
            # Compute current log-probs as ref for smooth bounded NPO push
            onp_ref_lp = []
            with torch.inference_mode():
                for ex in onp_enc:
                    ids, attn, em, fm = make_batch3([ex], pad_id, input_dev)
                    logits = model(input_ids=ids, attention_mask=attn, use_cache=False).logits[0, :-1].float()
                    lp = F.log_softmax(logits, dim=-1).gather(-1, ids[0, 1:].to(logits.device).unsqueeze(-1)).squeeze(-1)
                    onp_ref_lp.append(lp.cpu().to(torch.float16))
            model.gradient_checkpointing_enable()
            mix_enc = onp_enc * 3 + forget_enc[:80]
            mix_lp = onp_ref_lp * 3 + forget_ref_lp[:80]
            opt2 = bnb.optim.AdamW8bit([p for _, p in trainable], lr=a.lr * 0.6, weight_decay=0.0)
            run_phase("StageC-OnPolicy", mix_enc, mix_lp, a.round2_steps, a.lr * 0.6, opt2, best_state)

    if best_state["weights"] is not None:
        print(f"\nRestoring best checkpoint ({best_state['step']}, score={best_state['score']:.2f})...", flush=True)
        with torch.no_grad():
            for li, p in trainable:
                p.copy_(best_state["weights"][li].to(device=p.device, dtype=p.dtype))
        torch.save(best_state["weights"], out_dir / "mlp_down_weights.pt")

    train_sec = time.time() - t0
    model.eval()
    model.config.use_cache = True
    model.gradient_checkpointing_disable()
    eval_model_full(
        model, tok, out_dir, topic=a.topic, n_seeds=a.n_seeds, max_new=768, run_mentor=a.mentor,
        extra_meta={"train_sec": train_sec, "best_step": best_state["step"], "args": vars(a)},
    )
    print(f"SCU_DONE total_min={(time.time()-t0)/60:.2f}", flush=True)


if __name__ == "__main__":
    main()
