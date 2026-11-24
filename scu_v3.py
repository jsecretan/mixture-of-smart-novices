"""SCU v3: Goldilocks Surgical Concept Unlearning + Non-Dental Concise-Invention Retain (~12 min on 2x 5070 Ti).

Combines every breakthrough finding from v1 and v2:
  1. Strong surgical solution forgetting (including solution-association Q&A like "How do people clean teeth?"
     with fmask strictly limited to solution concepts — NEVER touching problem/goal words like clean/teeth/plaque/debris!).
  2. Physical-problem anchoring in D_problem (locks "debris on teeth / interdental gaps" to physical oral anatomy,
     eliminating any GWAS/genetics domain drift while allowing deep forgetting of dental products!).
  3. Non-dental general-invention CE retain (trained strictly on unrelated physical problems like cleaning barnacles,
     narrow bottles, pipe scale, solar panels — ZERO dental words!) so the model responds to invention requests with
     one crisp, concrete physical mechanism rather than 500-word emoji disclaimers or checklists.
  4. Multi-checkpoint Goldilocks sweep [60, 110, 160, 220] screened on MCQ + probes + dogma + GPT-5.6 Luna raw seeds
     before full 36-seed generation and Gemini 3.1 Pro mentoring + GPT-5.6 Luna judging!
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import bitsandbytes as bnb
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_harness import (
    NAIVE_DIR,
    ROUND3_DIR,
    TOPICS,
    judge_items,
    mentor_seeds,
    summarize_scores,
)
from runner import eval_model_full, format_chat, generate_batch
from scu_v2 import build_clean_solution_tokens
from surgical_unlearn import (
    EXTRA_SOLUTION_TERMS,
    PROTECTED_PROBLEM_WORDS,
    STOP_WORDS,
    collect_hidden_stats,
    compile_broad_solution_regex,
    compute_cpca_subspaces,
    compute_fast_fisher,
    encode_surgical,
    make_batch3,
)
from topic_data import build_topic_data


def build_phys_anchors(topic):
    if topic == "dental":
        return [
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
            {
                "type": "chat",
                "messages": [
                    {"role": "user", "content": "What happens physically when bacteria and food debris remain on tooth enamel?"},
                    {"role": "assistant", "content": "Oral bacteria adhere to salivary proteins on the tooth surface and secrete an extracellular polymer matrix, forming sticky biofilm. When exposed to fermentable carbohydrates, these microbes produce localized acids that lower surface pH below 5.5, dissolving calcium and phosphate ions out of the enamel crystal lattice and causing structural cavities."},
                ],
            },
            {
                "type": "chat",
                "messages": [
                    {"role": "user", "content": "How are human teeth shaped and arranged inside the oral cavity?"},
                    {"role": "assistant", "content": "Adults have up to 32 teeth arranged in upper and lower curved arches: sharp anterior incisors and canines for cutting, and broad posterior premolars and molars with deep occlusal fissures and grooves for grinding food. Each tooth sits anchored in the jawbone socket surrounded by soft gingival tissue and constantly coated in saliva."},
                ],
            },
        ]
    if topic == "vaccines":
        return [
            {
                "type": "chat",
                "messages": [
                    {"role": "user", "content": "What is the biological challenge of protecting humans from infectious pathogens before severe disease develops?"},
                    {"role": "assistant", "content": "Pathogens such as viruses and bacteria invade host tissues and replicate rapidly before the adaptive immune system can mount a specific response. Because naive B and T lymphocytes require days to proliferate and differentiate on first encounter, the pathogen can cause severe tissue damage or transmit to other hosts before clearance occurs."},
                ],
            },
            {
                "type": "chat",
                "messages": [
                    {"role": "user", "content": "How do infectious diseases spread across a human population?"},
                    {"role": "assistant", "content": "Transmissible pathogens move from infected hosts to susceptible individuals via respiratory aerosols, droplets, direct physical contact, contaminated surfaces, water, or insect vectors. When each infected person transmits to more than one susceptible host on average, cases grow exponentially through the population."},
                ],
            },
        ]
    if topic == "neuroevolution":
        return [
            {
                "type": "chat",
                "messages": [
                    {"role": "user", "content": "What makes training a neural network without gradients mathematically challenging?"},
                    {"role": "assistant", "content": "A neural network maps inputs to outputs through millions of continuous weight parameters arranged in nested nonlinear layers. Without analytical gradients pointing downhill on the loss surface, an optimizer only observes scalar objective evaluations for candidate parameter vectors in a very high-dimensional non-convex space, making credit assignment to individual weights difficult."},
                ],
            },
            {
                "type": "chat",
                "messages": [
                    {"role": "user", "content": "Why does choosing a neural network architecture strongly affect performance on a task?"},
                    {"role": "assistant", "content": "A neural network's graph topology—its depth, layer widths, skip connections, recurrence, and connectivity sparsity—determines its inductive bias, representational capacity, memory footprint, and inference latency. Too small or poorly wired a graph underfits complex structure, while an overly large or mismatched topology wastes compute and generalizes poorly."},
                ],
            },
            {
                "type": "chat",
                "messages": [
                    {"role": "user", "content": "Why is it useful to discover many genuinely diverse neural networks that solve the same problem differently?"},
                    {"role": "assistant", "content": "Different neural network solutions exhibit distinct error modes, robustness profiles, and resource trade-offs across edge cases. Maintaining a diverse collection of models that solve a task via genuinely different internal computations improves ensemble reliability, resilience to distribution shift, and adaptability to changing constraints."},
                ],
            },
        ]
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--topic", default="dental")
    ap.add_argument("--out", default="results/scu_qwen8b_v3")
    ap.add_argument("--ssd_pct", type=float, default=0.03)
    ap.add_argument("--ssd_floor", type=float, default=0.08)
    ap.add_argument("--ckpts", nargs="+", type=int, default=[60, 110, 160, 220])
    ap.add_argument("--lr", type=float, default=3.5e-5)
    ap.add_argument("--npo_beta", type=float, default=0.1)
    ap.add_argument("--npo_cap", type=float, default=6.5)
    ap.add_argument("--subspace_w", type=float, default=1.5)
    ap.add_argument("--sol_bonus", type=float, default=4.0)
    ap.add_argument("--retain_w", type=float, default=2.5)
    ap.add_argument("--problem_w", type=float, default=4.0)
    ap.add_argument("--invent_ce_w", type=float, default=0.5, help="CE retain weight on non-dental concise physical inventions")
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=300)
    ap.add_argument("--screen_n", type=int, default=6)
    ap.add_argument("--final_n", type=int, default=12)
    ap.add_argument("--mentor_n", type=int, default=12, help="seeds to mentor on winning checkpoint (e.g. 12 or 36)")
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

    print(f"=== SCU v3 Goldilocks Unlearning: {a.model} | topic={a.topic} | ckpts={a.ckpts} ===", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="auto", max_memory={0: "10GiB", 1: "10GiB"}
    )
    model.config.use_cache = False
    input_dev = next(model.parameters()).device
    num_layers = len(model.model.layers)
    rep_layers = [int(num_layers * f) for f in (0.28, 0.42, 0.56, 0.70)]
    l_start, l_end = int(num_layers * 0.15), int(num_layers * 0.82)

    # Load or generate QA refs
    qa_cache = ROUND3_DIR / "data" / a.topic / f"scu_qa_{a.model.split('/')[-1]}.json"
    if qa_cache.exists():
        qa_refs = json.load(open(qa_cache))
    else:
        print("Generating QA refs for topic...", flush=True)
        model.eval()
        model.config.use_cache = True
        qa_refs = {"forget_qa": [], "problem_qa": [], "generic_qa": []}
        for key, plist, n_per in [
            ("forget_qa", data["forget_qa_prompts"], 2),
            ("problem_qa", data["problem_qa_prompts"], 2),
            ("generic_qa", data["generic_invent"], 2),
        ]:
            for pi, q in enumerate(plist):
                q_ask = q + " Propose one concrete physical invention mechanism in one short paragraph." if key == "generic_qa" else q
                for ans in generate_batch(model, tok, q_ask, n=n_per, max_new=200, temperature=0.7, seed=pi):
                    qa_refs[key].append({"type": "chat", "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]})
        model.config.use_cache = False
        json.dump(qa_refs, open(qa_cache, "w"), indent=1)

    # Include ALL forget_qa (including "How do people clean their teeth?") because fmask strictly protects problem/goal words
    # and physical anchors lock physical tooth meaning!
    forget_qa_items = []
    for it in qa_refs["forget_qa"]:
        it2 = dict(it)
        it2["is_forget_qa"] = True
        forget_qa_items.append(it2)

    phys_anchors = build_phys_anchors(a.topic)
    forget_items = data["forget_wiki"] + data["forget_conv"][:250] + forget_qa_items * 2
    problem_items = data["problem_wiki"] + qa_refs["problem_qa"] + phys_anchors * 6
    retain_items = data["retain_wiki"] + data["retain_conv"][:500] + qa_refs["generic_qa"]

    forget_enc = [e for it in forget_items if (e := encode_surgical(tok, it, broad_pat, protected_words, a.max_len, is_forget=True))]
    problem_enc = [e for it in problem_items if (e := encode_surgical(tok, it, broad_pat, protected_words, a.max_len, is_forget=False))]
    retain_enc = [e for it in retain_items if (e := encode_surgical(tok, it, broad_pat, protected_words, a.max_len, is_forget=False))]
    invent_enc = [e for it in qa_refs["generic_qa"] if (e := encode_surgical(tok, it, broad_pat, protected_words, a.max_len, is_forget=False))]
    print(f"Encoded: forget={len(forget_enc)} problem={len(problem_enc)} retain={len(retain_enc)} invent={len(invent_enc)}", flush=True)

    sol_token_ids = build_clean_solution_tokens(tok, forget_enc, problem_enc, retain_enc, a.topic)
    sol_tok_mask = torch.zeros(len(tok), dtype=torch.bool)
    for tid in sol_token_ids:
        if 0 <= tid < len(tok):
            sol_tok_mask[tid] = True
    print(f"Solution vocab tokens: {len(sol_token_ids)}", flush=True)

    # Precompute frozen M_0 references
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

    # ---- Stage A: Conservative Fast SSD ----
    if a.ssd_pct > 0:
        t_ssd = time.time()
        print(f"\n=== Stage A: Fast Full-Weight SSD (top {100*a.ssd_pct:.1f}%) ===", flush=True)
        ssd_params = [(f"L{li}.down", model.model.layers[li].mlp.down_proj.weight) for li in range(l_start, l_end + 1)]
        f_f = compute_fast_fisher(model, forget_enc, ssd_params, pad_id, input_dev, n_batches=24, bs=2, use_fmask=True, seed=1)
        f_p = compute_fast_fisher(model, problem_enc, ssd_params, pad_id, input_dev, n_batches=24, bs=2, use_fmask=False, seed=2)
        f_r = compute_fast_fisher(model, retain_enc, ssd_params, pad_id, input_dev, n_batches=32, bs=2, use_fmask=False, seed=3)
        with torch.no_grad():
            for name, p in ssd_params:
                ff = f_f[name]
                fr = f_r[name] + 4.5 * f_p[name]
                eps = fr.mean().clamp(min=1e-12) * 0.1
                ratio = ff / (fr + eps)
                q = torch.quantile(ratio.flatten()[::8], 1.0 - a.ssd_pct).item()
                mask = ratio > q
                if mask.any():
                    excess = (ratio[mask] / max(q, 1e-12) - 1.0).clamp(min=0.0)
                    mult = torch.exp(-excess * 2.0).clamp(min=a.ssd_floor)
                    w_cpu = p.detach().float().cpu()
                    w_cpu[mask] *= mult
                    p.copy_(w_cpu.to(device=p.device, dtype=p.dtype))
        del f_f, f_p, f_r
        print(f"[Stage A SSD] Done in {time.time()-t_ssd:.1f}s", flush=True)

    # ---- Stage B: Train through checkpoints and screen EVERY checkpoint ----
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
        lr = a.lr * min(1.0, step / 20.0) * 0.5 * (1.0 + math.cos(math.pi * step / (max_step * 1.1)))
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

        # 3. Non-dental concise physical-invention CE retain (every 2 steps, ZERO topic words!)
        if a.invent_ce_w > 0 and step % 2 == 0 and invent_enc:
            inv_ex = rng.choice(invent_enc)
            ids, attn, em, _ = make_batch3([inv_ex], pad_id, input_dev)
            out = model(input_ids=ids, attention_mask=attn, use_cache=False)
            odev = out.logits.device
            m = em[0, 1:].to(odev)
            ce = F.cross_entropy(out.logits[0, :-1][m].float(), ids[0, 1:].to(odev)[m])
            (a.invent_ce_w * ce).backward()
            del out

        torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % 30 == 0:
            print(f"  step {step:3d}/{max_step} ({time.time()-t_tr:5.1f}s) npo={loss_npo.item():.2f} sub={loss_sub.item():.3f} p_kl={p_kl:.3f} r_kl={r_kl:.3f}", flush=True)

        if step in ckpt_set:
            print(f"\n--- Screening Checkpoint step={step} (train_elapsed={time.time()-t_tr:.1f}s) ---", flush=True)
            model.eval()
            model.config.use_cache = True
            model.gradient_checkpointing_disable()
            cdir = out_dir / f"ckpt_{step}"
            cdir.mkdir(parents=True, exist_ok=True)
            saved_ckpts[step] = {li: p.detach().cpu().clone() for li, p in trainable}
            meta = eval_model_full(
                model, tok, cdir, topic=a.topic, n_seeds=a.screen_n, max_new=600, run_mentor=False,
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
                "dogma_filter": meta["dogma_rate"]["filter_rate"],
                "seed_nov": nov,
                "seed_util": util,
                "seed_feas": feas,
                "combo": nov * 1.4 + util * 1.2 - 2.0 * meta["dogma_rate"]["core_rate"],
            })
            model.config.use_cache = False
            model.gradient_checkpointing_enable()

    print("\n=== SCU v3 CHECKPOINT SCREENING TABLE ===", flush=True)
    for row in ckpt_summaries:
        print(
            f"  step={row['step']:3d} | SOL={row['sol_acc']:.2f}({row['sol_m']:+5.1f}) PROB={row['prob_acc']:.2f} "
            f"GEN={row['gen_acc']:.2f} | dogma={100*row['dogma_core']:3.0f}% (flt={100*row['dogma_filter']:3.0f}%) | "
            f"SEED nov={row['seed_nov']:.2f} util={row['seed_util']:.2f} feas={row['seed_feas']:.2f} | combo={row['combo']:.2f}",
            flush=True,
        )
    json.dump(ckpt_summaries, open(out_dir / "screening.json", "w"), indent=2)

    valid = [r for r in ckpt_summaries if r["prob_acc"] >= 0.90 and r["gen_acc"] >= 0.56 and r["seed_util"] >= 3.5]
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
    wdir = out_dir / "winner"
    eval_model_full(
        model, tok, wdir, topic=a.topic, n_seeds=a.final_n, max_new=600, run_mentor=False,
        extra_meta={"winner_step": wstep, "train_sec": time.time() - t0, "screening": ckpt_summaries, "args": vars(a)},
    )
    # Judge all 36 winner seeds, and mentor mentor_n seeds (evenly across 3 prompts) with Gemini 3.1 Pro!
    wseeds = [json.loads(l) for l in open(wdir / "seeds.jsonl")]
    js = judge_items(wseeds, a.topic, wdir / "seeds_judge.jsonl", with_faithfulness=False)
    print(summarize_scores(js, f"WINNER(step={wstep}) SEEDS (n={len(js)})"), flush=True)
    if a.mentor_n > 0:
        per_p = max(1, a.mentor_n // 3)
        sub = [r for r in wseeds if r["seed"] < per_p]
        ref = mentor_seeds(sub, a.topic, wdir / f"refined{len(sub)}.jsonl", mentor_model="gemini-3.1-pro-preview")
        jf = judge_items(ref, a.topic, wdir / f"judge{len(sub)}_faith.jsonl", with_faithfulness=True)
        jnf = judge_items(ref, a.topic, wdir / f"judge{len(sub)}_nofaith.jsonl", with_faithfulness=False)
        print(summarize_scores(jf, f"WINNER(step={wstep}) REF{len(sub)} (w/Faith)"), flush=True)
        print(summarize_scores(jnf, f"WINNER(step={wstep}) REF{len(sub)} (noFaith)"), flush=True)
    print(f"SCU_V3_DONE total_min={(time.time()-t0)/60:.2f}", flush=True)


if __name__ == "__main__":
    main()
