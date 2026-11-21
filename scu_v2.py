"""SCU v2: Clean, Multi-Checkpoint Surgical Concept Unlearning (~10 minutes total on 2x RTX 5070 Ti).

Fixes every failure mode diagnosed from scu_qwen8b_v1:
  1. Clean solution-token filter: requires statistical enrichment (>=15x vs problem & retain) so
     common English suffixes ('ing', 'ent', 'ist', 'ers') are NEVER penalized.
  2. Strict separation of solution entities vs problem framing: D_forget contains solution-entity
     articles & definitions (never "How to clean teeth?" queries that cause (clean+teeth) -> genetics drift).
  3. Physical-problem anchor in D_problem: explicitly locks "removing debris/biofilm from hard tooth
     enamel and narrow interdental gaps" to real physical teeth & oral anatomy (zero product mentions).
  4. Multi-checkpoint Goldilocks selection in ONE run: evaluates checkpoints at steps [35, 65, 95, 130]
     on MCQ (solutions/problem/general), probes, dogma rate, and GPT-5.6 Luna raw seed scores (n=18)
     to find the exact sweet spot before committing to full 36-seed Gemini 3.1 Pro mentoring!
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

from eval_harness import (
    NAIVE_DIR,
    ROUND3_DIR,
    TOPICS,
    dogma_rate,
    eval_mcq_model,
    judge_items,
    mentor_seeds,
    summarize_scores,
)
from runner import eval_model_full, format_chat, generate_batch
from surgical_unlearn import (
    EXTRA_SOLUTION_TERMS,
    PROTECTED_PROBLEM_WORDS,
    STOP_WORDS,
    compile_broad_solution_regex,
    compute_cpca_subspaces,
    compute_fast_fisher,
    collect_hidden_stats,
    encode_surgical,
    make_batch3,
)
from topic_data import build_topic_data


def build_clean_solution_tokens(tok, forget_enc, problem_enc, retain_enc, topic):
    """Strictly enriched solution-concept tokens (no generic English suffixes!)."""
    protected = PROTECTED_PROBLEM_WORDS.get(topic, set()) | STOP_WORDS
    bad_suffixes = {
        "ing", "ent", "ist", "ers", "int", "art", "ine", "ure", "ide", "les", "ong", "ated",
        "ens", "per", "ator", "list", "arg", "son", "inter", "ven", "ody", "ins", "col",
        "ister", "ists", "ainer", "check", "ene", "ret", "home", "tion", "ment", "able",
        "ible", "ness", "less", "full", "ally", "ical", "ious", "eous", "ity", "ive", "ize",
    }
    cf, cp, cr = Counter(), Counter(), Counter()
    for ids, em, fm in forget_enc:
        cf.update(t for t, m in zip(ids, fm) if m)
    for ids, em, _ in problem_enc:
        cp.update(t for t, m in zip(ids, em) if m)
    for ids, em, _ in retain_enc[:400]:
        cr.update(t for t, m in zip(ids, em) if m)
    nf, np_, nr = max(1, sum(cf.values())), max(1, sum(cp.values())), max(1, sum(cr.values()))

    sol_ids = set()
    # 1. Whole-token or >=4-char specific solution stems from explicit terms
    terms = list(TOPICS[topic]["terms"]) + EXTRA_SOLUTION_TERMS.get(topic, [])
    for term in terms:
        for v in (term, " " + term, term.capitalize(), " " + term.capitalize()):
            tids = tok(v, add_special_tokens=False)["input_ids"]
            for tid in tids:
                piece = tok.decode([tid]).strip().lower()
                if len(piece) >= 4 and piece.isalpha() and piece not in protected and piece not in bad_suffixes:
                    # Also check it's not super frequent in retain!
                    if (cr[tid] / nr) < 5e-4:
                        sol_ids.add(tid)
    # 2. Strongly statistically enriched in forget vs BOTH problem AND retain
    for tid, c in cf.items():
        if c < 5:
            continue
        piece = tok.decode([tid]).strip().lower()
        if len(piece) < 4 or not piece.isalpha() or piece in protected or piece in bad_suffixes:
            continue
        rate_f = c / nf
        rate_p = (cp[tid] + 0.5) / np_
        rate_r = (cr[tid] + 0.5) / nr
        if rate_f / max(rate_p, rate_r) >= 15.0:
            sol_ids.add(tid)
    return sol_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--topic", default="dental")
    ap.add_argument("--out", default="results/scu_qwen8b_v2")
    ap.add_argument("--ssd_pct", type=float, default=0.025)
    ap.add_argument("--ssd_floor", type=float, default=0.10)
    ap.add_argument("--ckpts", nargs="+", type=int, default=[35, 65, 95, 130])
    ap.add_argument("--lr", type=float, default=2.0e-5)
    ap.add_argument("--npo_beta", type=float, default=0.1)
    ap.add_argument("--npo_cap", type=float, default=4.0)
    ap.add_argument("--subspace_w", type=float, default=1.0)
    ap.add_argument("--sol_bonus", type=float, default=3.0)
    ap.add_argument("--retain_w", type=float, default=4.0)
    ap.add_argument("--problem_w", type=float, default=6.0)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=300)
    ap.add_argument("--screen_n", type=int, default=6, help="seeds per prompt during checkpoint screening")
    ap.add_argument("--final_n", type=int, default=12, help="seeds per prompt for winning checkpoint")
    ap.add_argument("--mentor_final", action="store_true")
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

    print(f"=== SCU v2 Multi-Checkpoint Unlearning: {a.model} | topic={a.topic} | ckpts={a.ckpts} ===", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="auto", max_memory={0: "10GiB", 1: "10GiB"}
    )
    model.config.use_cache = False
    input_dev = next(model.parameters()).device
    num_layers = len(model.model.layers)
    rep_layers = [int(num_layers * f) for f in (0.28, 0.42, 0.56, 0.70)]
    l_start, l_end = int(num_layers * 0.15), int(num_layers * 0.82)

    # Load cached QA refs (reuse from scu_v1!) and add physical-problem anchors
    qa_cache = ROUND3_DIR / "data" / a.topic / f"scu_qa_{a.model.split('/')[-1]}.json"
    qa_refs = json.load(open(qa_cache))

    # Filter forget_qa to STRICTLY entity-definition/product Q&A (exclude "How do people clean their teeth?" prompts
    # that caused (clean + teeth) to drift into genetics!)
    clean_forget_qa = []
    for it in qa_refs["forget_qa"]:
        q = it["messages"][0]["content"].lower()
        if any(k in q for k in ("toothbrush", "floss", "toothpaste", "mouthwash", "interdental", "dentist", "tongue scraper", "bristle", "fluoride", "water flosser", "baking soda", "charcoal", "oil pulling", "bass brushing", "scaling", "commercial products", "standard tools")):
            it2 = dict(it)
            it2["is_forget_qa"] = True
            clean_forget_qa.append(it2)

    # Physical-problem anchors (pure tooth/debris/surface physics, ZERO product mentions!)
    phys_anchors = []
    if a.topic == "dental":
        phys_anchors = [
            {
                "type": "chat",
                "messages": [
                    {"role": "user", "content": "What are the physical constraints of removing sticky organic debris from human teeth inside the mouth?"},
                    {"role": "assistant", "content": "Human teeth are rigid, curved structures coated in mineralized hydroxyapatite enamel, arranged tightly in an arch inside a wet, warm oral cavity bathed in saliva. Sticky microbial biofilm and food debris adhere strongly to both exposed tooth surfaces and narrow sub-millimeter gaps between adjacent teeth, while the surrounding soft gum tissue is delicate and easily injured by excessive force, extreme heat, or harsh chemicals."},
                ],
            },
            {
                "type": "chat",
                "messages": [
                    {"role": "user", "content": "Why is debris trapped between adjacent teeth particularly hard to clear?"},
                    {"role": "assistant", "content": "Adjacent human teeth meet at tight interproximal contact points that leave narrow, triangular embrasure spaces right above the gumline. Food fibers and bacterial biofilm pack tightly into these confined geometries where bulk saliva flow has low shear velocity, making mechanical, fluidic, acoustic, or biochemical access difficult without disturbing the sensitive gingival tissue."},
                ],
            },
            {
                "type": "chat",
                "messages": [
                    {"role": "user", "content": "If someone says their teeth are dirty with food debris and biofilm, what physical problem are they describing?"},
                    {"role": "assistant", "content": "They are describing the physical accumulation of food residue and adherent microbial biofilm (plaque) on the hard mineralized surfaces of their teeth and in the narrow spaces between them inside the human mouth. Over time, bacteria metabolize sugars in that debris into organic acids that demineralize tooth enamel and irritate the gums."},
                ],
            },
        ]

    forget_items = data["forget_wiki"] + data["forget_conv"][:200] + clean_forget_qa
    problem_items = data["problem_wiki"] + qa_refs["problem_qa"] + phys_anchors * 5
    retain_items = data["retain_wiki"] + data["retain_conv"][:500] + qa_refs["generic_qa"] * 2

    forget_enc = [e for it in forget_items if (e := encode_surgical(tok, it, broad_pat, protected_words, a.max_len, is_forget=True))]
    problem_enc = [e for it in problem_items if (e := encode_surgical(tok, it, broad_pat, protected_words, a.max_len, is_forget=False))]
    retain_enc = [e for it in retain_items if (e := encode_surgical(tok, it, broad_pat, protected_words, a.max_len, is_forget=False))]
    print(f"Encoded: forget={len(forget_enc)} problem={len(problem_enc)} retain={len(retain_enc)}", flush=True)

    sol_token_ids = build_clean_solution_tokens(tok, forget_enc, problem_enc, retain_enc, a.topic)
    sol_tok_mask = torch.zeros(len(tok), dtype=torch.bool)
    for tid in sol_token_ids:
        if 0 <= tid < len(tok):
            sol_tok_mask[tid] = True
    print(f"CLEAN solution vocab tokens ({len(sol_token_ids)}): {[tok.decode([i]) for i in sorted(sol_token_ids)[:45]]}", flush=True)

    # Precompute frozen M_0 references
    t_ref = time.time()
    hf = collect_hidden_stats(model, forget_enc, rep_layers, pad_id, input_dev, max_items=90, use_fmask=True)
    hp = collect_hidden_stats(model, problem_enc, rep_layers, pad_id, input_dev, max_items=len(problem_enc), use_fmask=False)
    hr = collect_hidden_stats(model, retain_enc, rep_layers, pad_id, input_dev, max_items=90, use_fmask=False)
    subspaces, means = compute_cpca_subspaces(hf, hp, hr, rank=48, gamma_p=4.0, gamma_r=1.5)
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

    # ---- Stage A: Conservative Fast Full-Weight SSD (top 2.5%) ----
    if a.ssd_pct > 0:
        t_ssd = time.time()
        print(f"\n=== Stage A: Conservative Fast SSD (top {100*a.ssd_pct:.1f}%) ===", flush=True)
        ssd_params = [(f"L{li}.down", model.model.layers[li].mlp.down_proj.weight) for li in range(l_start, l_end + 1)]
        f_f = compute_fast_fisher(model, forget_enc, ssd_params, pad_id, input_dev, n_batches=24, bs=2, use_fmask=True, seed=1)
        f_p = compute_fast_fisher(model, problem_enc, ssd_params, pad_id, input_dev, n_batches=24, bs=2, use_fmask=False, seed=2)
        f_r = compute_fast_fisher(model, retain_enc, ssd_params, pad_id, input_dev, n_batches=32, bs=2, use_fmask=False, seed=3)
        with torch.no_grad():
            for name, p in ssd_params:
                ff = f_f[name]
                fr = f_r[name] + 5.0 * f_p[name]
                eps = fr.mean().clamp(min=1e-12) * 0.1
                ratio = ff / (fr + eps)
                q = torch.quantile(ratio.flatten()[::8], 1.0 - a.ssd_pct).item()
                mask = ratio > q
                if mask.any():
                    excess = (ratio[mask] / max(q, 1e-12) - 1.0).clamp(min=0.0)
                    mult = torch.exp(-excess * 1.8).clamp(min=a.ssd_floor)
                    w_cpu = p.detach().float().cpu()
                    w_cpu[mask] *= mult
                    p.copy_(w_cpu.to(device=p.device, dtype=p.dtype))
        del f_f, f_p, f_r
        print(f"[Stage A SSD] Done in {time.time()-t_ssd:.1f}s", flush=True)

    # ---- Stage B: Train through checkpoints and screen EVERY checkpoint on MCQ + Probes + Raw Seeds! ----
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = []
    for li in range(l_start, l_end + 1):
        p = model.model.layers[li].mlp.down_proj.weight
        p.requires_grad_(True)
        trainable.append((li, p))

    max_step = max(a.ckpts)
    ckpt_set = set(a.ckpts)
    model.gradient_checkpointing_enable()
    opt = bnb.optim.AdamW8bit([p for _, p in trainable], lr=a.lr, weight_decay=0.0)

    saved_ckpts = {}
    ckpt_summaries = []
    t_tr = time.time()

    for step in range(1, max_step + 1):
        model.train()
        lr = a.lr * min(1.0, step / 15.0) * 0.5 * (1.0 + math.cos(math.pi * step / (max_step * 1.15)))
        for g in opt.param_groups:
            g["lr"] = lr

        # 1. Surgical Forget step
        f_idx = [rng.randrange(len(forget_enc)) for _ in range(a.bs)]
        f_ex = [forget_enc[i] for i in f_idx]
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
            L = len(forget_ref_lp[idx])
            m = f_fm[bi, 1:1 + L].to(out_dev)
            if m.any():
                l_cur = lp_cur[bi, :L][m]
                l_ref = forget_ref_lp[idx].to(device=out_dev, dtype=torch.float32)[m]
                tok_t = tgt_f[bi, :L][m]
                tw = torch.where(sol_m_dev[tok_t], torch.tensor(a.sol_bonus, device=out_dev), torch.tensor(1.0, device=out_dev))
                diff = (l_cur - l_ref).clamp(min=-a.npo_cap)
                npo_tok = (2.0 / a.npo_beta) * F.softplus(a.npo_beta * diff) * tw
                loss_npo = loss_npo + npo_tok.sum()
                w_sum += tw.sum().item()
        loss_npo = loss_npo / max(1.0, w_sum)

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

        # 2. Retain Problem + Retain General
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
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step in ckpt_set:
            print(f"\n--- Screening Checkpoint step={step} (train_elapsed={time.time()-t_tr:.1f}s) ---", flush=True)
            model.eval()
            model.config.use_cache = True
            model.gradient_checkpointing_disable()
            cdir = out_dir / f"ckpt_{step}"
            cdir.mkdir(parents=True, exist_ok=True)
            saved_ckpts[step] = {li: p.detach().cpu().clone() for li, p in trainable}
            meta = eval_model_full(
                model, tok, cdir, topic=a.topic, n_seeds=a.screen_n, max_new=768, run_mentor=False,
                extra_meta={"step": step, "train_sec": time.time() - t0},
            )
            seeds = [json.loads(l) for l in open(cdir / "seeds.jsonl")]
            js = judge_items(seeds, a.topic, cdir / "seeds_judge.jsonl", with_faithfulness=False)
            print(summarize_scores(js, f"ckpt_{step} SEEDS (n={len(js)})"), flush=True)
            nov = sum(r["novelty"] for r in js) / max(1, len(js))
            util = sum(r["utility"] for r in js) / max(1, len(js))
            feas = sum(r["feasibility"] for r in js) / max(1, len(js))
            ckpt_summaries.append({
                "step": step,
                "sol_acc": meta["mcq"]["solutions"]["acc"],
                "sol_m": meta["mcq"]["solutions"]["margin"],
                "prob_acc": meta["mcq"]["problem"]["acc"],
                "gen_acc": meta["mcq"]["general"]["acc"],
                "dogma_core": meta["dogma_rate"]["core_rate"],
                "seed_nov": nov,
                "seed_util": util,
                "seed_feas": feas,
                "combo": nov + util - 1.5 * meta["dogma_rate"]["core_rate"],
            })
            model.config.use_cache = False
            model.gradient_checkpointing_enable()

    print("\n=== CHECKPOINT SCREENING TABLE ===", flush=True)
    for row in ckpt_summaries:
        print(
            f"  step={row['step']:3d} | SOL={row['sol_acc']:.2f}({row['sol_m']:+5.1f}) PROB={row['prob_acc']:.2f} "
            f"GEN={row['gen_acc']:.2f} | dogma={100*row['dogma_core']:3.0f}% | "
            f"SEED nov={row['seed_nov']:.2f} util={row['seed_util']:.2f} feas={row['seed_feas']:.2f} | combo={row['combo']:.2f}",
            flush=True,
        )
    json.dump(ckpt_summaries, open(out_dir / "screening.json", "w"), indent=2)

    # Pick winning checkpoint with highest seed_nov + seed_util (where prob >= 0.90 & gen >= 0.56)
    valid = [r for r in ckpt_summaries if r["prob_acc"] >= 0.90 and r["gen_acc"] >= 0.56]
    winner = max(valid or ckpt_summaries, key=lambda r: r["combo"])
    wstep = winner["step"]
    print(f"\n>>> WINNING CHECKPOINT: step={wstep} (SEED nov={winner['seed_nov']:.2f} util={winner['seed_util']:.2f}, dogma={100*winner['dogma_core']:.0f}%) <<<", flush=True)

    with torch.no_grad():
        for li, p in trainable:
            p.copy_(saved_ckpts[wstep][li].to(device=p.device, dtype=p.dtype))
    torch.save(saved_ckpts[wstep], out_dir / f"winner_step{wstep}_mlp_down.pt")

    model.eval()
    model.config.use_cache = True
    model.gradient_checkpointing_disable()
    eval_model_full(
        model, tok, out_dir / "winner", topic=a.topic, n_seeds=a.final_n, max_new=768, run_mentor=a.mentor_final,
        extra_meta={"winner_step": wstep, "train_sec": time.time() - t0, "screening": ckpt_summaries, "args": vars(a)},
    )
    print(f"SCU_V2_DONE total_min={(time.time()-t0)/60:.2f}", flush=True)


if __name__ == "__main__":
    main()
