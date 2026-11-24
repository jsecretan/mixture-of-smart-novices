"""Growing Narrow-Topic Mixture of Smart Novices (MoSN v2) — Core Engine.

Key capabilities:
  1. Persistent NVMe Registry (`../models/mosn_narrow/registry.json`): stores arbitrarily many
     razor-sharp single-topic Smart Novices without bloating the root disk.
  2. Goldilocks 2-Round Narrow IACF-v2 Builder with Cross-Topic Orthogonal Retention & automatic
     round selection (guards GEN accuracy so every novice stays sharp and fluent).
  3. Confidence-Gated Semantic Router (`Qwen3-Embedding-0.6B`, 100% local, ~20ms):
     - Cache hit (`sim >= threshold`): ~1.5s NVMe weight swap into the warm base model!
     - Cache miss (`sim < threshold`) with `auto_grow=True`: automatically calls the Forgetting
       Planner, builds a brand-new narrow novice in ~6 min on the warm VRAM model, saves it to
       NVMe, permanently registers it into `registry.json`, and serves the query!
"""
import json
import math
import random
import re
import time
from pathlib import Path

import bitsandbytes as bnb
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

MOSN_DIR = Path(__file__).resolve().parent
ROUND3_DIR = MOSN_DIR.parent
ROOT = ROUND3_DIR.parent

import sys
if str(ROUND3_DIR) not in sys.path:
    sys.path.insert(0, str(ROUND3_DIR))

from eval_harness import NAIVE_DIR, TOPICS, ask_llm, eval_mcq_model  # noqa: E402
from iacf_unlearn import compute_fast_fisher_nogc, select_ssd_params  # noqa: E402
from iacf_v2 import cache_top64_logits  # noqa: E402
from mosn.build_novices import compile_swath_regex, load_dolci_conversations  # noqa: E402
from mosn.narrow_topics import NARROW_TOPICS, NARROW_TOPIC_ORDER  # noqa: E402
from naive_service import (  # noqa: E402
    build_forgetting_plan,
    is_bad_seed,
    load_control_wiki,
    slugify,
)
from runner import generate_batch  # noqa: E402
from scu_v3 import build_phys_anchors  # noqa: E402
from surgical_unlearn import (  # noqa: E402
    PROTECTED_PROBLEM_WORDS,
    STOP_WORDS,
    compile_broad_solution_regex,
    encode_surgical,
    make_batch3,
)
from topic_data import build_topic_data  # noqa: E402


def get_narrow_root():
    nvme_models = ROOT / "models"
    d = (nvme_models / "mosn_narrow") if nvme_models.exists() else (ROUND3_DIR / "mosn_narrow_weights")
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_registry(model_tag):
    reg_file = get_narrow_root() / f"registry_{model_tag}.json"
    if reg_file.exists():
        return json.load(open(reg_file))
    return {"model_tag": model_tag, "novices": {}}


def save_registry(model_tag, reg):
    reg_file = get_narrow_root() / f"registry_{model_tag}.json"
    reg_file.parent.mkdir(parents=True, exist_ok=True)
    json.dump(reg, open(reg_file, "w"), indent=2)


def generate_topic_qa(model, tok, spec, cache_file=None, force=False):
    """Generate rich, multi-style forget_qa and problem_qa for ONE narrow topic on unedited base weights."""
    if cache_file and Path(cache_file).exists() and not force:
        d = json.load(open(cache_file))
        if len(d.get("forget_qa", [])) >= 75:
            return d

    model.eval()
    model.config.use_cache = True
    terms = spec["solution_terms"]
    pat = compile_swath_regex(terms)
    topic_name = spec.get("name", spec.get("topic_name", "this topic"))

    forget_qa = []
    # Rich multi-style prompts (encyclopedic, engineering, everyday usage, comparison) so Fisher targets
    # the actual factual concept circuits rather than a single repetitive "What is X?" QA template!
    templates = [
        "Write an encyclopedic overview of {term}, explaining its design, materials, and how it is used.",
        "Explain step-by-step how {term} works in practice and why people rely on it.",
        "Compare different commercial designs and variations of {term} and their key components.",
        "Describe the history, standard mechanics, and common variations of {term}.",
    ]
    for i, term in enumerate(terms[:22]):
        for ti in (i % 4, (i + 2) % 4):
            q = templates[ti].format(term=term)
            for ans in generate_batch(model, tok, q, n=2, max_new=200, temperature=0.8, seed=100 * i + ti):
                if not is_bad_seed(ans):
                    forget_qa.append({
                        "type": "chat",
                        "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": ans}],
                        "is_forget_qa": True,
                    })

    problem_qa = []
    for anc in spec.get("problem_anchors", []):
        if anc.get("q") and anc.get("a"):
            problem_qa.append({
                "type": "chat",
                "messages": [{"role": "user", "content": anc["q"]}, {"role": "assistant", "content": anc["a"]}],
            })
    p_templates = [
        "Explain the fundamental scientific principles and physical constraints of {asp}. Focus strictly on the science and problem mechanics without mentioning commercial products or standard tools.",
        "From a first-principles physics, biology, or systems perspective, what makes {asp} a challenging engineering problem? Do not name existing consumer products.",
    ]
    for i, asp in enumerate(spec.get("problem_aspects", [])[:8]):
        for ti, ptmpl in enumerate(p_templates):
            q = ptmpl.format(asp=asp)
            for ans in generate_batch(model, tok, q, n=2, max_new=200, temperature=0.7, seed=500 + 10 * i + ti):
                if not is_bad_seed(ans):
                    if pat.search(ans):
                        sents = [s for s in re.split(r"(?<=[.!?])\s+", ans) if not pat.search(s)]
                        ans = " ".join(sents).strip()
                    if len(ans.split()) >= 35:
                        problem_qa.append({
                            "type": "chat",
                            "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": ans}],
                        })

    out = {"forget_qa": forget_qa, "problem_qa": problem_qa}
    if cache_file:
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(cache_file, "w"), indent=1)
    model.config.use_cache = False
    print(f"  [qa:{topic_name}] forget_qa={len(forget_qa)} problem_qa={len(problem_qa)}", flush=True)
    return out


def eval_narrow_diagnostics(model, tok, spec):
    """Fast local sanity check: forget probes, problem probe, dogma check on sample prompts, and general MCQ."""
    model.eval()
    model.config.use_cache = True
    pat = compile_swath_regex(spec["solution_terms"])

    f_probes = {}
    for q in spec.get("forgetting_probes", [])[:2]:
        ans = generate_batch(model, tok, q, n=1, max_new=85, temperature=0.5, seed=0)[0].replace("\n", " ")
        f_probes[q] = ans[:220]

    p_probe = {}
    pq = spec.get("problem_probe")
    if pq:
        p_probe[pq] = generate_batch(model, tok, pq, n=1, max_new=90, temperature=0.5, seed=0)[0].replace("\n", " ")[:220]

    # Measure dogma rate & refusal/bad rate on sample prompts (n=6 quick screen)
    sample_prompts = spec.get("sample_prompts", [])
    dogma_hits, bad_hits, total_s = 0, 0, 0
    for pi, sp in enumerate(sample_prompts[:2]):
        outs = generate_batch(model, tok, sp, n=4, max_new=350, temperature=0.85, seed=100 * pi)
        for o in outs:
            total_s += 1
            if is_bad_seed(o):
                bad_hits += 1
            if pat.search(o):
                dogma_hits += 1

    mcq = eval_mcq_model(model, tok, topic="dental", n_general=60)
    gen_acc = float(mcq["general"]["acc"])
    gen_m = float(mcq["general"]["margin"])
    return {
        "gen_acc": round(gen_acc, 3),
        "gen_margin": round(gen_m, 2),
        "dogma_rate": round(dogma_hits / max(1, total_s), 3),
        "bad_rate": round(bad_hits / max(1, total_s), 3),
        "forgetting_probes": f_probes,
        "problem_probe": p_probe,
    }


def build_narrow_novice(
    model,
    tok,
    orig_sd,
    ssd_params,
    novice_id,
    spec,
    model_tag,
    dolci_all=None,
    other_retain_qa=None,
    base_gen=None,
    rounds=1,
    ssd_pct=0.045,
    problem_mult=6.0,
    heal_steps=240,
    heal_lr=3.5e-5,
    kl_w=2.5,
    n_fisher_batches=28,
    seed=42,
):
    """Build ONE razor-sharp narrow novice via Goldilocks KL-Anchored IACF-v2 and register it on NVMe."""
    t0 = time.time()
    narrow_root = get_narrow_root()
    w_dir = narrow_root / f"{novice_id}_{model_tag}"
    w_dir.mkdir(parents=True, exist_ok=True)
    w_file = w_dir / "iacf_weights.pt"

    # 1. Restore exact unedited base weights into the warm VRAM model
    with torch.no_grad():
        for n, p in ssd_params:
            p.copy_(orig_sd[n].to(device=p.device, dtype=p.dtype))

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    input_dev = next(model.parameters()).device
    num_layers = len(model.model.layers)
    l_start, l_end = int(num_layers * 0.12), int(num_layers * 0.92)
    rng = random.Random(seed + (abs(hash(novice_id)) % 10000))

    if base_gen is None:
        base_gen = float(eval_mcq_model(model, tok, topic="dental", n_general=60)["general"]["acc"])

    # 2. Build rich forget / problem / retain corpora (reusing curated Wikipedia + Dolci for paper topics!)
    qa = generate_topic_qa(model, tok, spec, cache_file=w_dir / "topic_qa.json")
    if novice_id in TOPICS:
        data = build_topic_data(novice_id)
        broad_pat = compile_broad_solution_regex(novice_id)
        protected_words = (PROTECTED_PROBLEM_WORDS.get(novice_id, set()) | set(w.lower() for w in spec.get("protected_words", [])) | STOP_WORDS)
        cross_ret = []
        for it in (other_retain_qa or []):
            full = " ".join(m["content"] for m in it["messages"])
            if not broad_pat.search(full):
                cross_ret.append({"type": "chat", "messages": it["messages"]})
        forget_items = data["forget_wiki"] + data["forget_conv"][:260] + qa["forget_qa"] * 2
        problem_items = data["problem_wiki"] + qa["problem_qa"] * 2 + build_phys_anchors(novice_id) * 4
        retain_items = cross_ret + data["retain_wiki"] + data["retain_conv"]
    else:
        broad_pat = compile_swath_regex(spec["solution_terms"])
        protected_words = set(w.lower() for w in spec.get("protected_words", [])) | STOP_WORDS
        if dolci_all is None:
            dolci_all = load_dolci_conversations(max_total=3200)
        forget_conv, retain_conv = [], []
        for it in dolci_all:
            full = " ".join(m["content"] for m in it["messages"])
            asst = " ".join(m["content"] for m in it["messages"] if m["role"] == "assistant")
            if broad_pat.search(full):
                if len(forget_conv) < 160 and broad_pat.search(asst):
                    forget_conv.append(it)
            elif len(retain_conv) < 1300:
                retain_conv.append(it)
        cross_ret = []
        for it in (other_retain_qa or []):
            full = " ".join(m["content"] for m in it["messages"])
            if not broad_pat.search(full):
                cross_ret.append({"type": "chat", "messages": it["messages"]})
        retain_wiki = load_control_wiki(broad_pat)
        forget_items = qa["forget_qa"] * 2 + forget_conv
        problem_items = qa["problem_qa"] * 3
        retain_items = cross_ret + retain_wiki + retain_conv

    forget_enc = [e for it in forget_items if (e := encode_surgical(tok, it, broad_pat, protected_words, 300, is_forget=True))]
    problem_enc = [e for it in problem_items if (e := encode_surgical(tok, it, broad_pat, protected_words, 300, is_forget=False))]
    retain_enc = [e for it in retain_items if (e := encode_surgical(tok, it, broad_pat, protected_words, 300, is_forget=False))]
    heal_ce_pool = retain_enc + problem_enc * 3

    print(
        f"\n================ BUILDING NARROW NOVICE: {novice_id} ({spec.get('name', novice_id)}) ================\n"
        f"  Encoded: forget={len(forget_enc)} problem={len(problem_enc)} retain={len(retain_enc)} "
        f"(cross_ret={len(cross_ret)}, rounds={rounds}, ssd_pct={ssd_pct}, base_gen={base_gen:.2f})",
        flush=True,
    )

    # 3. Cache frozen M_0 KL anchors ONCE on unedited base weights
    model.config.use_cache = False
    rng.shuffle(retain_enc)
    prob_kl_cache = cache_top64_logits(model, problem_enc, min(100, len(problem_enc)), pad_id, input_dev)
    ret_kl_cache = cache_top64_logits(model, retain_enc, min(280, len(retain_enc)), pad_id, input_dev)

    best_rd = 0
    best_score = -1e9
    best_sd = None
    best_diag = None
    round_logs = []

    for rd in range(1, rounds + 1):
        tr = time.time()
        model.config.use_cache = False

        # (a) Problem-Protected Fast Batched Diagonal Fisher
        f_f = compute_fast_fisher_nogc(model, forget_enc, ssd_params, pad_id, input_dev, n_batches=n_fisher_batches, bs=2, use_fmask=True, seed=20 * rd + 1)
        f_p = compute_fast_fisher_nogc(model, problem_enc, ssd_params, pad_id, input_dev, n_batches=22, bs=2, use_fmask=False, seed=20 * rd + 2)
        f_r = compute_fast_fisher_nogc(model, retain_enc, ssd_params, pad_id, input_dev, n_batches=36, bs=2, use_fmask=False, seed=20 * rd + 3)

        # (b) Selective Synaptic Dampening
        tot_d, tot_p = 0, 0
        with torch.no_grad():
            for name, p in ssd_params:
                ff = f_f[name]
                fr = f_r[name] + problem_mult * f_p[name]
                eps = fr.mean().clamp(min=1e-12) * 0.1
                ratio = ff / (fr + eps)
                q = torch.quantile(ratio.flatten()[::8], 1.0 - ssd_pct).item()
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

        # (c) 100% Positive KL-Anchored On-Manifold Healing
        model.gradient_checkpointing_enable()
        for p in model.parameters():
            p.requires_grad_(False)
        train_params = []
        for li in range(l_start, l_end + 1):
            for proj in (model.model.layers[li].mlp.down_proj.weight, model.model.layers[li].mlp.up_proj.weight):
                proj.requires_grad_(True)
                train_params.append(proj)
        opt = bnb.optim.AdamW8bit(train_params, lr=heal_lr, weight_decay=0.01)
        model.train()
        ce_vals, kl_vals = [], []
        for step in range(1, heal_steps + 1):
            lr = heal_lr * min(1.0, step / 20.0) * 0.5 * (1.0 + math.cos(math.pi * step / heal_steps))
            for g in opt.param_groups:
                g["lr"] = lr
            batch = [rng.choice(heal_ce_pool) for _ in range(2)]
            ids, attn, em, _ = make_batch3(batch, pad_id, input_dev)
            out = model(input_ids=ids, attention_mask=attn, use_cache=False)
            logits = out.logits[:, :-1]
            m = em[:, 1:].to(logits.device)
            loss_ce = F.cross_entropy(logits[m].float(), ids[:, 1:].to(logits.device)[m])
            (0.5 * loss_ce).backward()
            ce_vals.append(loss_ce.item())
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
            (kl_w * kl).backward()
            kl_vals.append(kl.item())
            del out, logits

            torch.nn.utils.clip_grad_norm_(train_params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

        del opt
        for p in train_params:
            p.requires_grad_(False)
            p.grad = None
        model.gradient_checkpointing_disable()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        diag = eval_narrow_diagnostics(model, tok, spec)
        diag["round"] = rd
        diag["sec"] = round(time.time() - tr, 1)
        diag["ce"] = round(sum(ce_vals[-20:]) / 20, 2)
        diag["kl"] = round(sum(kl_vals[-20:]) / 20, 3)
        round_logs.append(diag)

        first_fq = next(iter(diag["forgetting_probes"]), "")
        first_fa = diag["forgetting_probes"].get(first_fq, "")[:120]
        print(
            f"  [R{rd}] {diag['sec']:.0f}s | SSD={100*tot_d/tot_p:.2f}% | ce={diag['ce']:.2f} kl={diag['kl']:.3f} | "
            f"GEN={diag['gen_acc']:.2f} (base {base_gen:.2f}) | dogma={100*diag['dogma_rate']:.0f}% | bad={100*diag['bad_rate']:.0f}%",
            flush=True,
        )
        if first_fq:
            print(f"    PROBE: {first_fq} -> {first_fa}", flush=True)

        # Select round that maximizes dogma elimination while guarding GEN & low bad_rate
        gen_ok = diag["gen_acc"] >= base_gen - 0.07
        sc = (1.0 - diag["dogma_rate"]) * 4.0 + (1.0 - diag["bad_rate"]) * 3.0 + (diag["gen_acc"] - base_gen) * 5.0 + 0.3 * rd
        if (gen_ok and sc > best_score) or best_sd is None:
            if gen_ok or best_sd is None:
                best_score = sc
                best_rd = rd
                best_diag = diag
                best_sd = {n: p.detach().cpu().clone() for n, p in ssd_params}

        if diag["gen_acc"] < base_gen - 0.10:
            print(f"  Stopping early after R{rd} (GEN dropped to {diag['gen_acc']:.2f})", flush=True)
            break

    # Save best weights on NVMe and load them into model
    torch.save(best_sd, w_file)
    with torch.no_grad():
        for n, p in ssd_params:
            p.copy_(best_sd[n].to(device=p.device, dtype=p.dtype))
    del best_sd

    meta = {
        "id": novice_id,
        "name": spec.get("name", spec.get("topic_name", novice_id)),
        "description": spec.get("description", spec.get("why_creative", "")),
        "routing_summary": spec.get("routing_summary", ""),
        "sample_prompts": spec.get("sample_prompts", []),
        "solution_terms": spec["solution_terms"],
        "protected_words": spec.get("protected_words", []),
        "forgetting_probes": spec.get("forgetting_probes", []),
        "problem_probe": spec.get("problem_probe", ""),
        "best_round": best_rd,
        "base_gen": round(base_gen, 3),
        "diagnostics": best_diag,
        "rounds": round_logs,
        "weights_file": str(w_file),
        "build_min": round((time.time() - t0) / 60.0, 2),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    json.dump(meta, open(w_dir / "meta.json", "w"), indent=2)

    # Permanently register into registry_<model_tag>.json!
    reg = load_registry(model_tag)
    reg["novices"][novice_id] = {
        "id": novice_id,
        "name": meta["name"],
        "description": meta["description"],
        "routing_summary": meta["routing_summary"],
        "sample_prompts": meta["sample_prompts"],
        "solution_terms": meta["solution_terms"],
        "weights_file": str(w_file),
        "best_round": best_rd,
        "gen_acc": best_diag["gen_acc"],
        "dogma_rate": best_diag["dogma_rate"],
        "build_min": meta["build_min"],
        "created_at": meta["created_at"],
    }
    save_registry(model_tag, reg)
    print(
        f"  >>> Registered narrow novice '{novice_id}' (best=R{best_rd}, GEN={best_diag['gen_acc']:.2f}, "
        f"dogma={100*best_diag['dogma_rate']:.0f}%, {meta['build_min']:.1f}m) -> {w_file}",
        flush=True,
    )
    return meta


class NarrowMoSNRouter:
    """Confidence-gated semantic router over a growing registry of narrow single-topic novices."""

    def __init__(self, model_tag="Qwen3-4B-Instruct-2507", mode="embed", llm_model="gpt-6-luna", device="cuda:0"):
        self.model_tag = model_tag
        self.mode = mode
        self.llm_model = llm_model
        self.device = device
        self.emb_tok = None
        self.emb_model = None
        self.ids = []
        self.vecs = None
        self.reg = {}
        if mode == "embed":
            self.emb_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
            self.emb_model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B", dtype=torch.bfloat16).to(self.device).eval()
        self.refresh()

    def _embed(self, texts):
        with torch.inference_mode():
            enc = self.emb_tok(texts, padding=True, truncation=True, max_length=320, return_tensors="pt").to(self.device)
            out = self.emb_model(**enc).last_hidden_state
            lens = enc["attention_mask"].sum(dim=1) - 1
            vecs = out[torch.arange(len(texts), device=self.device), lens].float()
            vecs = vecs / vecs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            return vecs.cpu().numpy()

    def refresh(self):
        self.reg = load_registry(self.model_tag).get("novices", {})
        self.ids = sorted(self.reg.keys())
        if not self.ids:
            self.vecs = None
            return
        if self.mode == "embed":
            texts = []
            for nid in self.ids:
                e = self.reg[nid]
                samples = " ".join(e.get("sample_prompts", [])[:2])
                terms = ", ".join(e.get("solution_terms", [])[:12])
                texts.append(
                    f"{e.get('name', nid)}: {e.get('description', '')}. {e.get('routing_summary', '')} "
                    f"Examples: {samples} Related terms: {terms}"
                )
            self.vecs = self._embed(texts)

    def route(self, prompt, top_k=1, threshold=0.52):
        """Return (best_matches, all_ranked, is_hit). Cache miss when top cosine sim < threshold."""
        if not self.ids:
            return [], [], False
        if self.mode == "embed":
            v = self._embed([prompt])[0]
            sims = self.vecs @ v
            order = np.argsort(-sims)
            ranked = [(self.ids[i], float(sims[i])) for i in order]
            is_hit = ranked[0][1] >= threshold
            return ranked[:top_k], ranked, is_hit
        else:
            desc = "\n".join(
                f"- {nid}: {self.reg[nid].get('routing_summary') or self.reg[nid].get('description', '')}"
                for nid in self.ids
            )
            q = (
                f"Pick the narrow novice whose forgotten domain DIRECTLY matches the user prompt.\n"
                f"If NONE of the existing novices directly match this topic (e.g., it's a genuinely new domain), "
                f"set match=false.\nPrompt: {prompt}\n\nRegistered novices:\n{desc}\n\n"
                f'Answer with JSON only: {{"match": true/false, "ranked": ["best_id", ...], "reason": "short"}}'
            )
            raw = ask_llm(self.llm_model, q, json_mode=True)
            parsed = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
            ids = [x for x in parsed.get("ranked", []) if x in self.reg]
            for nid in self.ids:
                if nid not in ids:
                    ids.append(nid)
            is_hit = bool(parsed.get("match", False)) and bool(ids)
            ranked = [(nid, (0.95 if (i == 0 and is_hit) else max(0.1, 0.4 - 0.05 * i))) for i, nid in enumerate(ids)]
            return ranked[:top_k], ranked, is_hit


class NarrowMoSNPool:
    """Warm VRAM base model with ~1.5s NVMe novice swapping AND on-demand --auto-grow!"""

    def __init__(self, model_id="Qwen/Qwen3-4B-Instruct-2507"):
        self.model_id = model_id
        self.model_tag = model_id.split("/")[-1]
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
        self._dolci_cache = None
        self._base_gen = None
        print(f"[mosn-narrow] Base model {model_id} warm in VRAM ({time.time()-t0:.1f}s)", flush=True)

    def activate(self, novice_id):
        if novice_id == self.active:
            return 0.0
        t0 = time.time()
        if novice_id == "baseline":
            sd = self.orig_sd
        else:
            reg = load_registry(self.model_tag).get("novices", {})
            if novice_id not in reg:
                raise FileNotFoundError(f"Narrow novice '{novice_id}' not found in registry for {self.model_tag}!")
            w_file = Path(reg[novice_id]["weights_file"])
            if not w_file.exists():
                raise FileNotFoundError(f"Weights file missing: {w_file}")
            sd = torch.load(w_file, map_location="cpu", weights_only=True)
        with torch.no_grad():
            for n, p in self.ssd_params:
                src = sd.get(n, self.orig_sd[n])
                p.copy_(src.to(device=p.device, dtype=p.dtype))
        self.active = novice_id
        return time.time() - t0

    def grow_novice(self, prompt, router=None, planner_model="gpt-6-luna", rounds=1, ssd_pct=0.045):
        """Auto-discover a brand-new narrow topic for `prompt`, build its IACF-v2 novice, and register it!"""
        print(f"\n[mosn-grow] Cache miss! Auto-discovering narrow topic for: {prompt}", flush=True)
        plan = build_forgetting_plan(prompt, explicit_topic=None, planner_model=planner_model)
        novice_id = slugify(plan["topic_name"])[:32]
        reg = load_registry(self.model_tag).get("novices", {})
        if novice_id in reg:
            print(f"[mosn-grow] Planner mapped to existing novice '{novice_id}' — reusing!", flush=True)
            return novice_id

        spec = {
            "name": plan["topic_name"],
            "description": plan.get("why_creative", plan["topic_name"]),
            "routing_summary": f"{plan['topic_name']}: {plan.get('why_creative', '')} Prompt context: {prompt}",
            "sample_prompts": [prompt],
            "solution_terms": plan["solution_terms"],
            "protected_words": plan["protected_words"],
            "problem_aspects": plan.get("problem_aspects", []),
            "problem_anchors": plan.get("problem_anchors", []),
            "forgetting_probes": plan.get("forgetting_probes", [f"What is {plan['solution_terms'][0]}?"]),
            "problem_probe": plan.get("problem_probe", ""),
        }
        if self._dolci_cache is None:
            self._dolci_cache = load_dolci_conversations(max_total=3200)
        if self._base_gen is None:
            self.activate("baseline")
            self._base_gen = float(eval_mcq_model(self.model, self.tok, topic="dental", n_general=60)["general"]["acc"])

        # Collect compact cross-topic retain QA from existing registered novices so growing preserves old topics!
        other_qa = []
        narrow_root = get_narrow_root()
        for existing_id in list(reg.keys())[:6]:
            qaf = narrow_root / f"{existing_id}_{self.model_tag}" / "topic_qa.json"
            if qaf.exists():
                d = json.load(open(qaf))
                other_qa.extend(d.get("forget_qa", [])[:10])
                other_qa.extend(d.get("problem_qa", [])[:4])

        build_narrow_novice(
            self.model,
            self.tok,
            self.orig_sd,
            self.ssd_params,
            novice_id=novice_id,
            spec=spec,
            model_tag=self.model_tag,
            dolci_all=self._dolci_cache,
            other_retain_qa=other_qa,
            base_gen=self._base_gen,
            rounds=rounds,
            ssd_pct=ssd_pct,
        )
        self.active = novice_id
        if router is not None:
            router.refresh()
        return novice_id

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
