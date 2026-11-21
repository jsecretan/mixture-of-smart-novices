"""Iterative Accelerated Catastrophic Forgetting (IACF) — 100% Positive-Gradient, On-Manifold Fast Forgetting!

Zero negative gradients (no NPO, no RMU, no gradient ascent -> zero gibberish, zero hollow tautology).
Accelerates the paper's exact catastrophic-forgetting mechanism (600 H100-hours -> ~12-18 minutes on 2x 5070 Ti):

  Repeat for r = 1 .. R rounds:
    1. Fast Fisher Importance (~60-90s): compute F_forget (solution entities & facts),
       F_problem (tooth/enamel/biofilm anatomy), and F_retain (non-dental Dolci + general wiki + generic_invent)
       directly on the CURRENT model weights (exposing redundant/backup circuits on each round!).
    2. Selective Synaptic Dampening (~5s): smoothly decay the top `pct` most solution-specific weights
       (high F_forget / (F_retain + gamma_p * F_problem)), simulating epochs of selective weight decay.
    3. 100% Positive On-Manifold Filtered SFT (~2-3 min): train full weights (or merged high-rank LoRA)
       with standard positive Cross-Entropy + weight decay on strictly filtered non-dental Dolci +
       problem anatomy + non-dental concrete inventions (NEVER seeing domain solutions!).
    4. Screen on MCQ(solutions/problem/general), probes, dogma rate, and GPT-5.6 Luna raw seed scores!

After R rounds, restore the highest-scoring naive checkpoint, generate 36 seeds, and run Gemini 3.1 Pro
mentoring + GPT-5.6 Luna judging (exact paper Appendix A & E protocol).
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
from peft import LoraConfig, get_peft_model
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
from scu_v3 import build_phys_anchors
from surgical_unlearn import (
    PROTECTED_PROBLEM_WORDS,
    STOP_WORDS,
    compile_broad_solution_regex,
    encode_surgical,
    make_batch3,
)
from topic_data import build_topic_data


def select_ssd_params(model, layer_start, layer_end, include_attn=True):
    selected = []
    for name, p in model.named_parameters():
        if p.ndim != 2 or "layers." not in name:
            continue
        parts = name.split(".")
        li = int(parts[parts.index("layers") + 1])
        if layer_start <= li <= layer_end:
            keys = ("mlp.down_proj", "mlp.up_proj", "mlp.gate_proj")
            if include_attn:
                keys = keys + ("self_attn.v_proj", "self_attn.o_proj")
            if any(k in name for k in keys):
                selected.append((name, p))
    return selected


def compute_fast_fisher_nogc(model, encoded_list, selected_params, pad_id, input_dev, n_batches=28, bs=2, use_fmask=False, seed=0):
    """Blazing-fast Fisher without gradient checkpointing, streaming squared grads to CPU via hooks."""
    rng = random.Random(seed)
    fisher = {name: torch.zeros(p.shape, dtype=torch.float32, device="cpu") for name, p in selected_params}
    model.eval()
    model.gradient_checkpointing_disable()
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

    used = 0
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
            used += 1
        del out, logits
    for h in hooks:
        h.remove()
    for name in fisher:
        fisher[name].div_(max(1, used))
    for _, p in selected_params:
        p.requires_grad_(False)
        p.grad = None
    torch.cuda.empty_cache()
    return fisher


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--topic", default="dental")
    ap.add_argument("--out", default="results/iacf_qwen4b_v1")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--ssd_pct", type=float, default=0.05, help="top fraction dampened per round")
    ap.add_argument("--ssd_floor", type=float, default=0.05)
    ap.add_argument("--ssd_exp", type=float, default=2.5)
    ap.add_argument("--problem_mult", type=float, default=5.0)
    ap.add_argument("--n_fisher_batches", type=int, default=24)
    ap.add_argument("--heal_steps", type=int, default=200)
    ap.add_argument("--heal_lr", type=float, default=5e-5)
    ap.add_argument("--heal_wd", type=float, default=0.1)
    ap.add_argument("--heal_mode", choices=["full_mlp", "lora_merge"], default="full_mlp")
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=300)
    ap.add_argument("--screen_n", type=int, default=6)
    ap.add_argument("--final_n", type=int, default=12)
    ap.add_argument("--mentor_n", type=int, default=12)
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

    print(f"=== IACF (Iterative Accelerated Catastrophic Forgetting): {a.model} | topic={a.topic} | rounds={a.rounds} ===", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="auto", max_memory={0: "10GiB", 1: "10GiB"}
    )
    model.config.use_cache = False
    input_dev = next(model.parameters()).device
    num_layers = len(model.model.layers)
    l_start, l_end = int(num_layers * 0.12), int(num_layers * 0.90)
    print(f"Loaded {a.model} ({num_layers} layers, SSD layers {l_start}..{l_end}) in {time.time()-t0:.1f}s", flush=True)

    # Generate or load model-specific QA refs
    qa_cache = ROUND3_DIR / "data" / a.topic / f"iacf_qa_{a.model.split('/')[-1]}.json"
    if qa_cache.exists():
        qa_refs = json.load(open(qa_cache))
    else:
        print("Generating M_0 QA references...", flush=True)
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
        print(f"QA refs ready ({time.time()-t0:.1f}s)", flush=True)

    for it in qa_refs["forget_qa"]:
        it["is_forget_qa"] = True
    phys_anchors = build_phys_anchors(a.topic)

    forget_items = data["forget_wiki"] + data["forget_conv"][:300] + qa_refs["forget_qa"] * 2
    problem_items = data["problem_wiki"] + qa_refs["problem_qa"] + phys_anchors * 5
    # Rich non-dental retain pool (all 1500 filtered Dolci convs + control wiki + non-dental inventions!)
    retain_items = data["retain_wiki"] + data["retain_conv"] + qa_refs["generic_qa"] * 4

    forget_enc = [e for it in forget_items if (e := encode_surgical(tok, it, broad_pat, protected_words, a.max_len, is_forget=True))]
    problem_enc = [e for it in problem_items if (e := encode_surgical(tok, it, broad_pat, protected_words, a.max_len, is_forget=False))]
    retain_enc = [e for it in retain_items if (e := encode_surgical(tok, it, broad_pat, protected_words, a.max_len, is_forget=False))]
    heal_pool = retain_enc + problem_enc * 3
    print(f"Encoded: forget={len(forget_enc)} problem={len(problem_enc)} retain={len(retain_enc)} heal_pool={len(heal_pool)}", flush=True)

    # Round 0: Baseline screening
    print("\n=== Round 0: Baseline Screening ===", flush=True)
    r0_dir = out_dir / "round_0"
    model.eval()
    model.config.use_cache = True
    m0 = eval_model_full(model, tok, r0_dir, topic=a.topic, n_seeds=a.screen_n, max_new=600, run_mentor=False)
    s0 = [json.loads(l) for l in open(r0_dir / "seeds.jsonl")]
    j0 = judge_items(s0, a.topic, r0_dir / "seeds_judge.jsonl", with_faithfulness=False)
    print(summarize_scores(j0, "Round 0 BASELINE SEEDS"), flush=True)
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
        tr = time.time()
        print(f"\n================ ROUND {rd}/{a.rounds} ================", flush=True)
        model.config.use_cache = False
        ssd_params = select_ssd_params(model, l_start, l_end, include_attn=True)
        if rd == 1:
            print(f"[IACF] Targeting {len(ssd_params)} matrices ({sum(p.numel() for _, p in ssd_params)/1e9:.2f}B params)", flush=True)

        # 1. Re-compute Fisher on CURRENT model weights (exposes redundant/backup dental circuits!)
        tf = time.time()
        f_f = compute_fast_fisher_nogc(model, forget_enc, ssd_params, pad_id, input_dev, n_batches=a.n_fisher_batches, bs=a.bs, use_fmask=True, seed=10 * rd + 1)
        f_p = compute_fast_fisher_nogc(model, problem_enc, ssd_params, pad_id, input_dev, n_batches=max(16, a.n_fisher_batches - 4), bs=a.bs, use_fmask=False, seed=10 * rd + 2)
        f_r = compute_fast_fisher_nogc(model, retain_enc, ssd_params, pad_id, input_dev, n_batches=a.n_fisher_batches + 8, bs=a.bs, use_fmask=False, seed=10 * rd + 3)
        print(f"  [R{rd} Fisher] Computed in {time.time()-tf:.1f}s", flush=True)

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

        # 3. 100% Positive On-Manifold Filtered SFT Healing (with weight decay!)
        th = time.time()
        model.gradient_checkpointing_enable()
        if a.heal_mode == "full_mlp":
            for p in model.parameters():
                p.requires_grad_(False)
            train_params = []
            for li in range(l_start, l_end + 1):
                for proj in (model.model.layers[li].mlp.down_proj.weight, model.model.layers[li].mlp.up_proj.weight):
                    proj.requires_grad_(True)
                    train_params.append(proj)
            opt = bnb.optim.AdamW8bit(train_params, lr=a.heal_lr, weight_decay=a.heal_wd)
            model.train()
            losses = []
            for step in range(1, a.heal_steps + 1):
                lr = a.heal_lr * min(1.0, step / 20.0) * 0.5 * (1.0 + math.cos(math.pi * step / a.heal_steps))
                for g in opt.param_groups:
                    g["lr"] = lr
                batch = [rng.choice(heal_pool) for _ in range(a.bs)]
                ids, attn, em, _ = make_batch3(batch, pad_id, input_dev)
                out = model(input_ids=ids, attention_mask=attn, use_cache=False)
                logits = out.logits[:, :-1]
                tgt = ids[:, 1:].to(logits.device)
                m = em[:, 1:].to(logits.device)
                loss = F.cross_entropy(logits[m].float(), tgt[m])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(train_params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                losses.append(loss.item())
                del out, logits
            del opt
            for p in train_params:
                p.requires_grad_(False)
                p.grad = None
        else:
            lcfg = LoraConfig(
                r=64, lora_alpha=128, lora_dropout=0.0,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                task_type="CAUSAL_LM",
            )
            peft_m = get_peft_model(model, lcfg)
            peft_m.train()
            tparams = [p for p in peft_m.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(tparams, lr=a.heal_lr * 2.0, weight_decay=a.heal_wd)
            losses = []
            for step in range(1, a.heal_steps + 1):
                lr = (a.heal_lr * 2.0) * min(1.0, step / 20.0) * 0.5 * (1.0 + math.cos(math.pi * step / a.heal_steps))
                for g in opt.param_groups:
                    g["lr"] = lr
                batch = [rng.choice(heal_pool) for _ in range(a.bs)]
                ids, attn, em, _ = make_batch3(batch, pad_id, input_dev)
                out = peft_m(input_ids=ids, attention_mask=attn, use_cache=False)
                logits = out.logits[:, :-1]
                tgt = ids[:, 1:].to(logits.device)
                m = em[:, 1:].to(logits.device)
                loss = F.cross_entropy(logits[m].float(), tgt[m])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(tparams, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                losses.append(loss.item())
                del out, logits
            model = peft_m.merge_and_unload()
            del opt
        torch.cuda.empty_cache()
        print(f"  [R{rd} Heal] {a.heal_steps} positive steps in {time.time()-th:.1f}s | loss {losses[0]:.2f} -> {sum(losses[-20:])/20:.2f}", flush=True)

        # 4. Screen this round!
        model.eval()
        model.config.use_cache = True
        model.gradient_checkpointing_disable()
        rdir = out_dir / f"round_{rd}"
        meta = eval_model_full(
            model, tok, rdir, topic=a.topic, n_seeds=a.screen_n, max_new=600, run_mentor=False,
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

        # Save best checkpoint on CPU if general & problem intact
        if row["prob_acc"] >= base_prob - 0.12 and row["gen_acc"] >= base_gen - 0.08:
            sc = nov * 1.5 + util * 1.2 + (1.0 - row["dogma_core"]) * 2.0 + (1.0 - row["sol_acc"]) * 3.0
            if sc > best_score:
                best_score = sc
                best_round = rd
                best_state_dict = {n: p.detach().cpu().clone() for n, p in select_ssd_params(model, l_start, l_end, include_attn=True)}
                print(f"  >>> New BEST round: R{rd} (score={sc:.2f}, nov={nov:.2f}, util={util:.2f}, dogma={100*row['dogma_core']:.0f}%) <<<", flush=True)
        print(f"  [R{rd} total elapsed: {(time.time()-t0)/60:.1f} min]", flush=True)
        if row["gen_acc"] < base_gen - 0.12 or row["prob_acc"] < base_prob - 0.18:
            print(f"  Stopping early after R{rd} (gen={row['gen_acc']:.2f}, prob={row['prob_acc']:.2f})", flush=True)
            break

    print("\n=== IACF ROUND-BY-ROUND TABLE ===", flush=True)
    for row in summaries:
        print(
            f"  R{row['round']} | SOL={row['sol_acc']:.2f}({row['sol_m']:+5.1f}) PROB={row['prob_acc']:.2f} GEN={row['gen_acc']:.2f} | "
            f"dogma={100*row['dogma_core']:3.0f}% (flt={100*row['dogma_filter']:3.0f}%) | "
            f"SEED nov={row['seed_nov']:.2f} util={row['seed_util']:.2f} feas={row['seed_feas']:.2f}",
            flush=True,
        )
    json.dump(summaries, open(out_dir / "rounds.json", "w"), indent=2)

    if best_state_dict is not None:
        print(f"\nRestoring BEST round R{best_round} for full evaluation & Gemini Pro mentoring...", flush=True)
        with torch.no_grad():
            for n, p in select_ssd_params(model, l_start, l_end, include_attn=True):
                p.copy_(best_state_dict[n].to(device=p.device, dtype=p.dtype))
        torch.save(best_state_dict, out_dir / f"best_round{best_round}_weights.pt")

    wdir = out_dir / "winner"
    eval_model_full(
        model, tok, wdir, topic=a.topic, n_seeds=a.final_n, max_new=600, run_mentor=False,
        extra_meta={"best_round": best_round, "train_sec": time.time() - t0, "rounds": summaries, "args": vars(a)},
    )
    wseeds = [json.loads(l) for l in open(wdir / "seeds.jsonl")]
    js = judge_items(wseeds, a.topic, wdir / "seeds_judge.jsonl", with_faithfulness=False)
    print(summarize_scores(js, f"IACF WINNER(R{best_round}) SEEDS (n={len(js)})"), flush=True)
    if a.mentor_n > 0:
        per_p = max(1, a.mentor_n // 3)
        sub = [r for r in wseeds if r["seed"] < per_p]
        ref = mentor_seeds(sub, a.topic, wdir / f"refined{len(sub)}.jsonl", mentor_model="gemini-3.1-pro-preview")
        jf = judge_items(ref, a.topic, wdir / f"judge{len(sub)}_faith.jsonl", with_faithfulness=True)
        jnf = judge_items(ref, a.topic, wdir / f"judge{len(sub)}_nofaith.jsonl", with_faithfulness=False)
        print(summarize_scores(jf, f"IACF WINNER(R{best_round}) REF{len(sub)} (w/Faith)"), flush=True)
        print(summarize_scores(jnf, f"IACF WINNER(R{best_round}) REF{len(sub)} (noFaith)"), flush=True)
    print(f"IACF_DONE total_min={(time.time()-t0)/60:.2f}", flush=True)


if __name__ == "__main__":
    main()
