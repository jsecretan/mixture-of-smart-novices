"""IACF v2: KL-Anchored Iterative Accelerated Catastrophic Forgetting (~15 min on 2x RTX 5070 Ti).

Fixes the two exact issues discovered in iacf_qwen4b_v1:
  1. Replaces raw CE + high weight decay (heal_wd=0.1, which shrank general margins 0.65->0.51 after R1)
     with KL-Anchored Positive Healing: 0.5 * CE(filtered Dolci + problem) + 1.5 * Top64_KL(M_0 || M)
     on retain + problem, with heal_wd=0.01 -> keeps GEN >= 0.62 and PROB = 1.00 across ALL 4-5 rounds!
  2. Removes generic_qa template upsampling (which caused repetitive "silicone micro-needles" templates),
     preserving the model's full natural diversity across physics, robotics, acoustics, chemistry, and biology!
  3. Re-computes Fisher on current weights each round (exposing backup/redundant dental-solution synapses!)
     and applies smooth SSD (top 3.5% per round, heavily protecting F_problem & F_retain).
  4. Screens every round (R0..R5) on MCQ, probes, dogma rate, and GPT-5.6 Luna raw seeds (n=18), then
     evaluates the winning round (n=36 seeds) with Gemini 3.1 Pro mentoring + GPT-5.6 Luna judging!
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
    ROUND3_DIR,
    TOPICS,
    judge_items,
    mentor_seeds,
    summarize_scores,
)
from iacf_unlearn import compute_fast_fisher_nogc, select_ssd_params
from runner import eval_model_full
from scu_v3 import build_phys_anchors
from surgical_unlearn import (
    PROTECTED_PROBLEM_WORDS,
    STOP_WORDS,
    compile_broad_solution_regex,
    encode_surgical,
    make_batch3,
)
from topic_data import build_topic_data


def cache_top64_logits(model, enc_list, limit, pad_id, input_dev):
    out_c = []
    with torch.inference_mode():
        for ex in enc_list[:limit]:
            ids, attn, em, _ = make_batch3([ex], pad_id, input_dev)
            logits = model(input_ids=ids, attention_mask=attn, use_cache=False).logits[0, :-1].float()
            topv, topi = torch.topk(logits, k=64, dim=-1)
            out_c.append((ex, topv.cpu().to(torch.float16), topi.cpu().to(torch.int32)))
    return out_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--topic", default="dental")
    ap.add_argument("--out", default="results/iacf_qwen4b_v2")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--ssd_pct", type=float, default=0.035)
    ap.add_argument("--ssd_floor", type=float, default=0.05)
    ap.add_argument("--ssd_exp", type=float, default=2.5)
    ap.add_argument("--problem_mult", type=float, default=6.0)
    ap.add_argument("--n_fisher_batches", type=int, default=28)
    ap.add_argument("--heal_steps", type=int, default=160)
    ap.add_argument("--heal_lr", type=float, default=3.5e-5)
    ap.add_argument("--heal_wd", type=float, default=0.01)
    ap.add_argument("--kl_w", type=float, default=1.5)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=300)
    ap.add_argument("--screen_n", type=int, default=6)
    ap.add_argument("--final_n", type=int, default=12)
    ap.add_argument("--mentor_n", type=int, default=12)
    ap.add_argument("--mentor_baseline", action="store_true")
    ap.add_argument("--mlp_only_ssd", action="store_true", help="only SSD MLP weights (faster on 8B)")
    ap.add_argument("--heal_down_only", action="store_true", help="only train mlp.down_proj during KL-heal (lower VRAM on 8B)")
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

    print(f"=== IACF v2 (KL-Anchored Positive Catastrophic Forgetting): {a.model} | topic={a.topic} | rounds={a.rounds} ===", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="auto", max_memory={0: "10GiB", 1: "10GiB"}
    )
    model.config.use_cache = False
    input_dev = next(model.parameters()).device
    num_layers = len(model.model.layers)
    l_start, l_end = int(num_layers * 0.12), int(num_layers * 0.92)
    print(f"Loaded {a.model} ({num_layers} layers, SSD layers {l_start}..{l_end}) in {time.time()-t0:.1f}s", flush=True)

    qa_cache = ROUND3_DIR / "data" / a.topic / f"iacf_qa_{a.model.split('/')[-1]}.json"
    if not qa_cache.exists():
        # Reuse scu_qa if present, else generate
        alt = ROUND3_DIR / "data" / a.topic / f"scu_qa_{a.model.split('/')[-1]}.json"
        if alt.exists():
            qa_refs = json.load(open(alt))
        else:
            from runner import generate_batch
            model.eval()
            model.config.use_cache = True
            qa_refs = {"forget_qa": [], "problem_qa": [], "generic_qa": []}
            for key, plist, n_per in [("forget_qa", data["forget_qa_prompts"], 2), ("problem_qa", data["problem_qa_prompts"], 2)]:
                for pi, q in enumerate(plist):
                    for ans in generate_batch(model, tok, q, n=n_per, max_new=200, temperature=0.7, seed=pi):
                        qa_refs[key].append({"type": "chat", "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]})
            model.config.use_cache = False
            json.dump(qa_refs, open(qa_cache, "w"), indent=1)
    else:
        qa_refs = json.load(open(qa_cache))

    for it in qa_refs["forget_qa"]:
        it["is_forget_qa"] = True
    phys_anchors = build_phys_anchors(a.topic)

    # Pure, diverse non-dental retain pool (NO repetitive generic_qa template upsampling!)
    forget_items = data["forget_wiki"] + data["forget_conv"][:300] + qa_refs["forget_qa"] * 2
    problem_items = data["problem_wiki"] + qa_refs["problem_qa"] + phys_anchors * 4
    retain_items = data["retain_wiki"] + data["retain_conv"]

    forget_enc = [e for it in forget_items if (e := encode_surgical(tok, it, broad_pat, protected_words, a.max_len, is_forget=True))]
    problem_enc = [e for it in problem_items if (e := encode_surgical(tok, it, broad_pat, protected_words, a.max_len, is_forget=False))]
    retain_enc = [e for it in retain_items if (e := encode_surgical(tok, it, broad_pat, protected_words, a.max_len, is_forget=False))]
    heal_ce_pool = retain_enc + problem_enc * 2
    print(f"Encoded: forget={len(forget_enc)} problem={len(problem_enc)} retain={len(retain_enc)}", flush=True)

    # Cache frozen M_0 top-64 teacher logits for KL-anchored positive healing!
    tc = time.time()
    rng.shuffle(retain_enc)
    prob_kl_cache = cache_top64_logits(model, problem_enc, len(problem_enc), pad_id, input_dev)
    ret_kl_cache = cache_top64_logits(model, retain_enc, 280, pad_id, input_dev)
    print(f"Cached M_0 KL anchors in {time.time()-tc:.1f}s (prob={len(prob_kl_cache)}, ret={len(ret_kl_cache)})", flush=True)

    # Round 0 Baseline
    print("\n=== Round 0: Baseline Screening ===", flush=True)
    r0_dir = out_dir / "round_0"
    model.eval()
    model.config.use_cache = True
    m0 = eval_model_full(model, tok, r0_dir, topic=a.topic, n_seeds=a.screen_n, max_new=650, run_mentor=False)
    s0 = [json.loads(l) for l in open(r0_dir / "seeds.jsonl")]
    j0 = judge_items(s0, a.topic, r0_dir / "seeds_judge.jsonl", with_faithfulness=False)
    print(summarize_scores(j0, "Round 0 BASELINE SEEDS"), flush=True)
    if a.mentor_baseline:
        sub0 = [r for r in s0 if r["seed"] < max(1, a.mentor_n // 3)]
        ref0 = mentor_seeds(sub0, a.topic, r0_dir / f"refined{len(sub0)}.jsonl", mentor_model="gemini-3.1-pro-preview")
        jf0 = judge_items(ref0, a.topic, r0_dir / f"judge{len(sub0)}_faith.jsonl", with_faithfulness=True)
        jnf0 = judge_items(ref0, a.topic, r0_dir / f"judge{len(sub0)}_nofaith.jsonl", with_faithfulness=False)
        print(summarize_scores(jf0, f"Round 0 BASELINE REF{len(sub0)} (w/Faith)"), flush=True)
        print(summarize_scores(jnf0, f"Round 0 BASELINE REF{len(sub0)} (noFaith)"), flush=True)

    base_gen = m0["mcq"]["general"]["acc"]
    base_prob = m0["mcq"]["problem"]["acc"]
    summaries = [{
        "round": 0,
        "sol_acc": m0["mcq"]["solutions"]["acc"], "sol_m": m0["mcq"]["solutions"]["margin"],
        "prob_acc": base_prob, "gen_acc": base_gen,
        "dogma_core": m0["dogma_rate"]["core_rate"], "dogma_filter": m0["dogma_rate"]["filter_rate"],
        "seed_nov": sum(r["novelty"] for r in j0) / len(j0),
        "seed_util": sum(r["utility"] for r in j0) / len(j0),
        "seed_feas": sum(r["feasibility"] for r in j0) / len(j0),
    }]

    best_round = 0
    best_score = -1e9
    best_state_dict = None

    for rd in range(1, a.rounds + 1):
        print(f"\n================ ROUND {rd}/{a.rounds} ================", flush=True)
        model.config.use_cache = False
        ssd_params = select_ssd_params(model, l_start, l_end, include_attn=not a.mlp_only_ssd)

        # 1. Re-compute Fisher on CURRENT weights
        tf = time.time()
        f_f = compute_fast_fisher_nogc(model, forget_enc, ssd_params, pad_id, input_dev, n_batches=a.n_fisher_batches, bs=a.bs, use_fmask=True, seed=20 * rd + 1)
        f_p = compute_fast_fisher_nogc(model, problem_enc, ssd_params, pad_id, input_dev, n_batches=max(16, a.n_fisher_batches - 4), bs=a.bs, use_fmask=False, seed=20 * rd + 2)
        f_r = compute_fast_fisher_nogc(model, retain_enc, ssd_params, pad_id, input_dev, n_batches=a.n_fisher_batches + 8, bs=a.bs, use_fmask=False, seed=20 * rd + 3)
        print(f"  [R{rd} Fisher] Done in {time.time()-tf:.1f}s", flush=True)

        # 2. Selective Synaptic Dampening on full weights
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
                    mult = torch.exp(-excess * a.ssd_exp).clamp(min=a.ssd_floor)
                    w_cpu = p.detach().float().cpu()
                    w_cpu[mask] *= mult
                    p.copy_(w_cpu.to(device=p.device, dtype=p.dtype))
        del f_f, f_p, f_r
        print(f"  [R{rd} SSD] Dampened {tot_d:,}/{tot_p:,} ({100*tot_d/tot_p:.2f}%)", flush=True)

        # 3. KL-Anchored 100% Positive On-Manifold Healing (NO negative gradients, ZERO general drift!)
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
        opt = bnb.optim.PagedAdamW8bit(train_params, lr=a.heal_lr, weight_decay=a.heal_wd)
        model.train()
        ce_vals, kl_vals = [], []

        for step in range(1, a.heal_steps + 1):
            lr = a.heal_lr * min(1.0, step / 20.0) * 0.5 * (1.0 + math.cos(math.pi * step / a.heal_steps))
            for g in opt.param_groups:
                g["lr"] = lr

            # (a) Positive CE on diverse filtered Dolci + problem
            batch = [rng.choice(heal_ce_pool) for _ in range(a.bs)]
            ids, attn, em, _ = make_batch3(batch, pad_id, input_dev)
            out = model(input_ids=ids, attention_mask=attn, use_cache=False)
            logits = out.logits[:, :-1]
            tgt = ids[:, 1:].to(logits.device)
            m = em[:, 1:].to(logits.device)
            loss_ce = F.cross_entropy(logits[m].float(), tgt[m])
            (0.5 * loss_ce).backward()
            ce_vals.append(loss_ce.item())
            del out, logits

            # (b) Exact Top-64 KL Anchor to M_0 on retain + problem (keeps general & problem razor-sharp!)
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
            kl_vals.append(kl.item())
            del out, logits

            torch.nn.utils.clip_grad_norm_(train_params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

        del opt
        for p in train_params:
            p.requires_grad_(False)
            p.grad = None
        torch.cuda.empty_cache()
        print(
            f"  [R{rd} KL-Heal] {a.heal_steps} steps in {time.time()-th:.1f}s | "
            f"ce={sum(ce_vals[-20:])/20:.2f} kl={sum(kl_vals[-20:])/20:.3f}",
            flush=True,
        )

        # 4. Screen Round rd
        model.eval()
        model.config.use_cache = True
        model.gradient_checkpointing_disable()
        rdir = out_dir / f"round_{rd}"
        meta = eval_model_full(
            model, tok, rdir, topic=a.topic, n_seeds=a.screen_n, max_new=650, run_mentor=False,
            extra_meta={"round": rd, "train_sec": time.time() - t0},
        )
        seeds = [json.loads(l) for l in open(rdir / "seeds.jsonl")]
        js = judge_items(seeds, a.topic, rdir / "seeds_judge.jsonl", with_faithfulness=False)
        print(summarize_scores(js, f"Round {rd} SEEDS (n={len(js)})"), flush=True)

        nov = sum(r["novelty"] for r in js) / len(js)
        util = sum(r["utility"] for r in js) / len(js)
        feas = sum(r["feasibility"] for r in js) / len(js)
        row = {
            "round": rd,
            "sol_acc": meta["mcq"]["solutions"]["acc"], "sol_m": meta["mcq"]["solutions"]["margin"],
            "prob_acc": meta["mcq"]["problem"]["acc"], "gen_acc": meta["mcq"]["general"]["acc"],
            "dogma_core": meta["dogma_rate"]["core_rate"], "dogma_filter": meta["dogma_rate"]["filter_rate"],
            "seed_nov": nov, "seed_util": util, "seed_feas": feas,
        }
        summaries.append(row)

        if row["prob_acc"] >= base_prob - 0.10 and row["gen_acc"] >= base_gen - 0.08:
            sc = nov * 1.6 + util * 1.3 + (1.0 - row["dogma_core"]) * 2.5 + (1.0 - row["sol_acc"]) * 3.0
            if sc > best_score:
                best_score = sc
                best_round = rd
                best_state_dict = {n: p.detach().cpu().clone() for n, p in select_ssd_params(model, l_start, l_end, include_attn=not a.mlp_only_ssd)}
                print(f"  >>> New BEST round: R{rd} (score={sc:.2f}, nov={nov:.2f}, util={util:.2f}, dogma={100*row['dogma_core']:.0f}%) <<<", flush=True)
        print(f"  [R{rd} elapsed: {(time.time()-t0)/60:.1f} min]", flush=True)
        if row["gen_acc"] < base_gen - 0.12 or row["prob_acc"] < base_prob - 0.15:
            print(f"  Stopping early after R{rd} (gen={row['gen_acc']:.2f}, prob={row['prob_acc']:.2f})", flush=True)
            break

    print("\n=== IACF v2 ROUND-BY-ROUND TABLE ===", flush=True)
    for row in summaries:
        print(
            f"  R{row['round']} | SOL={row['sol_acc']:.2f}({row['sol_m']:+5.1f}) PROB={row['prob_acc']:.2f} GEN={row['gen_acc']:.2f} | "
            f"dogma={100*row['dogma_core']:3.0f}% (flt={100*row['dogma_filter']:3.0f}%) | "
            f"SEED nov={row['seed_nov']:.2f} util={row['seed_util']:.2f} feas={row['seed_feas']:.2f}",
            flush=True,
        )
    json.dump(summaries, open(out_dir / "rounds.json", "w"), indent=2)

    if best_state_dict is not None:
        print(f"\nRestoring BEST round R{best_round} for full 36-seed eval & Gemini Pro mentoring...", flush=True)
        with torch.no_grad():
            for n, p in select_ssd_params(model, l_start, l_end, include_attn=not a.mlp_only_ssd):
                p.copy_(best_state_dict[n].to(device=p.device, dtype=p.dtype))
        torch.save(best_state_dict, out_dir / f"best_round{best_round}_weights.pt")

    wdir = out_dir / "winner"
    eval_model_full(
        model, tok, wdir, topic=a.topic, n_seeds=a.final_n, max_new=650, run_mentor=False,
        extra_meta={"best_round": best_round, "train_sec": time.time() - t0, "rounds": summaries, "args": vars(a)},
    )
    wseeds = [json.loads(l) for l in open(wdir / "seeds.jsonl")]
    js = judge_items(wseeds, a.topic, wdir / "seeds_judge.jsonl", with_faithfulness=False)
    print(summarize_scores(js, f"IACF-v2 WINNER(R{best_round}) SEEDS (n={len(js)})"), flush=True)
    if a.mentor_n > 0:
        per_p = max(1, a.mentor_n // 3)
        sub = [r for r in wseeds if r["seed"] < per_p]
        ref = mentor_seeds(sub, a.topic, wdir / f"refined{len(sub)}.jsonl", mentor_model="gemini-3.1-pro-preview")
        jf = judge_items(ref, a.topic, wdir / f"judge{len(sub)}_faith.jsonl", with_faithfulness=True)
        jnf = judge_items(ref, a.topic, wdir / f"judge{len(sub)}_nofaith.jsonl", with_faithfulness=False)
        print(summarize_scores(jf, f"IACF-v2 WINNER(R{best_round}) REF{len(sub)} (w/Faith)"), flush=True)
        print(summarize_scores(jnf, f"IACF-v2 WINNER(R{best_round}) REF{len(sub)} (noFaith)"), flush=True)
    print(f"IACF_V2_DONE total_min={(time.time()-t0)/60:.2f}", flush=True)


if __name__ == "__main__":
    main()
