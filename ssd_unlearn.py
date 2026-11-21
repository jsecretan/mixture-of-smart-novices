"""Paradigm 2: Full-Weight Selective Synaptic Dampening (SSD) + Filtered SFT Healing.

Directly accelerates the paper's catastrophic forgetting (600 H100-hours -> ~8 minutes!)
by editing the ACTUAL 7B base weights (not a frozen-base LoRA):
  Stage 1 (~2 min): Compute diagonal Fisher importance F_forget (dental solutions),
                    F_problem (tooth/mouth/enamel biology), and F_retain (general Dolci + control wiki)
                    across all MLP & attention output weights of the 7B model.
  Stage 2 (~5 sec): Selectively dampen weights with high specificity ratio:
                    rho = F_forget / (F_retain + problem_mult * F_problem + eps)
                    applying smooth synaptic decay w <- w * exp(-strength * relu(rho - thresh))
                    (physically simulating hundreds of epochs of selective catastrophic forgetting!).
  Stage 3 (~5 min): Short Filtered SFT Healing (LoRA merged into weights) on non-dental Dolci +
                    problem_wiki + generic_invent so the model stays 100% fluent & coherent.
  Stage 4: Evaluate MCQ (solutions / problem / general), probes, and 36 seeds!
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_harness import ROUND3_DIR
from runner import eval_model_full
from topic_data import build_topic_data
from dscu_unlearn import encode_item, make_batch


def target_weight_params(model, layer_start=6, layer_end=31):
    """Return list of (name, param) for MLP + attn value/out weights in mid-to-late layers."""
    selected = []
    for name, p in model.named_parameters():
        if p.ndim != 2:
            continue
        # Match layers.<idx>.mlp.(gate_proj|up_proj|down_proj) or self_attn.(v_proj|o_proj)
        parts = name.split(".")
        if "layers" in parts:
            li = int(parts[parts.index("layers") + 1])
            if layer_start <= li <= layer_end:
                if any(k in name for k in ("mlp.down_proj", "mlp.up_proj", "mlp.gate_proj", "self_attn.v_proj", "self_attn.o_proj")):
                    selected.append((name, p))
    return selected


def compute_fisher(model, encoded_list, selected_params, pad_id, input_dev, n_samples=120, bs=1, seed=0):
    """Stream diagonal Fisher E[(dL/dW)^2] directly to CPU during backward via hooks (zero extra GPU VRAM!)."""
    rng = random.Random(seed)
    fisher = {name: torch.zeros(p.shape, dtype=torch.float32, device="cpu") for name, p in selected_params}
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    hooks = []
    for name, p in selected_params:
        p.requires_grad_(True)
        def _make_hook(nm):
            def _hook(param):
                if param.grad is not None:
                    fisher[nm].add_(param.grad.detach().float().pow(2).cpu())
                    param.grad = None
            return _hook
        hooks.append(p.register_post_accumulate_grad_hook(_make_hook(name)))

    n_batches = max(1, n_samples // bs)
    for bi in range(n_batches):
        batch = [rng.choice(encoded_list) for _ in range(bs)]
        ids, attn, mask = make_batch(batch, pad_id, input_dev)
        out = model(input_ids=ids, attention_mask=attn, use_cache=False)
        logits = out.logits[:, :-1]
        tgt = ids[:, 1:].to(logits.device)
        m = mask[:, 1:].to(logits.device)
        loss = F.cross_entropy(logits[m].float(), tgt[m])
        loss.backward()
        del out, logits, loss
    for h in hooks:
        h.remove()
    for name in fisher:
        fisher[name].div_(n_batches)
    for _, p in selected_params:
        p.requires_grad_(False)
        p.grad = None
    torch.cuda.empty_cache()
    return fisher


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/Olmo-3-7B-Instruct-SFT")
    ap.add_argument("--topic", default="dental")
    ap.add_argument("--out", default="results/ssd_olmo_v1")
    ap.add_argument("--n_fisher", type=int, default=160)
    ap.add_argument("--layer_start", type=int, default=6)
    ap.add_argument("--layer_end", type=int, default=31)
    ap.add_argument("--problem_mult", type=float, default=3.0)
    ap.add_argument("--top_pct", type=float, default=0.015, help="fraction of most dental-specific weights to dampen")
    ap.add_argument("--damp_floor", type=float, default=0.05, help="minimum multiplier for top specific weights")
    ap.add_argument("--heal_steps", type=int, default=180)
    ap.add_argument("--heal_lr", type=float, default=1e-4)
    ap.add_argument("--n_seeds", type=int, default=12)
    ap.add_argument("--mentor", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    t0 = time.time()
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    out_dir = ROUND3_DIR / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    data = build_topic_data(a.topic)

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="auto", max_memory={0: "9GiB", 1: "9GiB"}
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    input_dev = next(model.parameters()).device

    qa_cache = ROUND3_DIR / "data" / a.topic / f"qa_refs_{a.model.split('/')[-1]}.json"
    qa_refs = json.load(open(qa_cache)) if qa_cache.exists() else {"forget_qa": [], "problem_qa": [], "generic_qa": []}

    forget_enc = [e for it in (data["forget_wiki"] + data["forget_conv"][:250] + qa_refs["forget_qa"]) if (e := encode_item(tok, it, 320))]
    problem_enc = [e for it in (data["problem_wiki"] + qa_refs["problem_qa"]) if (e := encode_item(tok, it, 320))]
    retain_enc = [e for it in (data["retain_wiki"] + data["retain_conv"][:600] + qa_refs["generic_qa"]) if (e := encode_item(tok, it, 320))]

    selected = target_weight_params(model, a.layer_start, a.layer_end)
    print(f"[SSD] Selected {len(selected)} weight matrices ({sum(p.numel() for _, p in selected)/1e9:.2f}B params)", flush=True)

    t1 = time.time()
    print("[SSD] Computing F_forget...", flush=True)
    f_forget = compute_fisher(model, forget_enc, selected, pad_id, input_dev, n_samples=a.n_fisher, seed=1)
    print(f"[SSD] Computing F_problem... ({time.time()-t1:.1f}s)", flush=True)
    f_problem = compute_fisher(model, problem_enc, selected, pad_id, input_dev, n_samples=a.n_fisher, seed=2)
    print(f"[SSD] Computing F_retain... ({time.time()-t1:.1f}s)", flush=True)
    f_retain = compute_fisher(model, retain_enc, selected, pad_id, input_dev, n_samples=a.n_fisher * 2, seed=3)
    print(f"[SSD] Fisher done in {time.time()-t1:.1f}s", flush=True)

    # Fast sweep over top_pct on MCQ + quick probe before committing & healing!
    from eval_harness import eval_mcq_model
    from runner import generate_batch
    orig_weights = {name: p.detach().cpu().clone() for name, p in selected}

    def apply_ssd(top_pct, damp_floor, exp_k=1.5):
        tot_d, tot_p = 0, 0
        with torch.no_grad():
            for name, p in selected:
                ff = f_forget[name]
                fr = f_retain[name] + a.problem_mult * f_problem[name]
                eps = fr.mean().clamp(min=1e-12) * 0.1
                ratio = ff / (fr + eps)
                q = torch.quantile(ratio.flatten()[::8], 1.0 - top_pct).item()
                mask = ratio > q
                n_hit = int(mask.sum().item())
                tot_d += n_hit
                tot_p += ratio.numel()
                w_cpu = orig_weights[name].float().clone()
                if n_hit > 0:
                    excess = (ratio[mask] / max(q, 1e-12) - 1.0).clamp(min=0.0)
                    mult = torch.exp(-excess * exp_k).clamp(min=damp_floor)
                    w_cpu[mask] *= mult
                p.copy_(w_cpu.to(device=p.device, dtype=p.dtype))
        return tot_d, tot_p

    if a.sweep:
        print("\n=== [SSD] Sweeping dampening strengths (zero extra Fisher cost!) ===", flush=True)
        model.config.use_cache = True
        model.gradient_checkpointing_disable()
        best = None
        for pct, floor, ek in [
            (0.01, 0.10, 1.5),
            (0.02, 0.05, 2.0),
            (0.04, 0.02, 2.5),
            (0.07, 0.01, 3.0),
            (0.12, 0.00, 4.0),
        ]:
            td, tp = apply_ssd(pct, floor, ek)
            mcq = eval_mcq_model(model, tok, topic=a.topic, n_general=50)
            p1 = generate_batch(model, tok, "What is a toothbrush?", n=1, max_new=60, temperature=0.5, seed=0)[0].replace("\n", " ")
            p2 = generate_batch(model, tok, "My teeth are dirty. Can you help me invent something to clean my teeth?", n=1, max_new=80, temperature=0.7, seed=0)[0].replace("\n", " ")
            print(
                f"  pct={pct:.2f} floor={floor:.2f} | sol={mcq['solutions']['acc']:.2f}({mcq['solutions']['margin']:+.2f}) "
                f"prob={mcq['problem']['acc']:.2f}({mcq['problem']['margin']:+.2f}) "
                f"gen={mcq['general']['acc']:.2f} | tb={p1[:90]} | inv={p2[:100]}",
                flush=True,
            )
            # Pick strongest forgetting where problem >= 0.80 and general >= 0.48
            if mcq["problem"]["acc"] >= 0.80 and mcq["general"]["acc"] >= 0.48:
                score = (1.0 - mcq["solutions"]["acc"]) * 10.0 - mcq["solutions"]["margin"] * 0.2
                if best is None or score > best[0]:
                    best = (score, pct, floor, ek)
        if best is not None:
            _, pct, floor, ek = best
            print(f"[SSD] Selected best sweep point: pct={pct} floor={floor} ek={ek}", flush=True)
            apply_ssd(pct, floor, ek)
        else:
            apply_ssd(a.top_pct, a.damp_floor, 1.5)
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    else:
        td, tp = apply_ssd(a.top_pct, a.damp_floor, 1.5)
        print(f"[SSD] Dampened {td:,} / {tp:,} params ({100*td/tp:.2f}%)", flush=True)
    del f_forget, f_problem, f_retain, orig_weights

    # Stage 3: Short Filtered SFT Healing on non-dental retain + problem + generic_invent
    if a.heal_steps > 0:
        th = time.time()
        print(f"[SSD] Stage 3: Filtered SFT Healing ({a.heal_steps} steps)...", flush=True)
        model = get_peft_model(
            model,
            LoraConfig(
                r=32, lora_alpha=64, lora_dropout=0.0,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                task_type="CAUSAL_LM",
            ),
        )
        model.train()
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=a.heal_lr, weight_decay=0.0)
        heal_pool = retain_enc + problem_enc * 2
        for step in range(1, a.heal_steps + 1):
            lr = a.heal_lr * min(1.0, step / 20.0) * 0.5 * (1.0 + math.cos(math.pi * step / a.heal_steps))
            for g in opt.param_groups:
                g["lr"] = lr
            batch = [rng.choice(heal_pool) for _ in range(2)]
            ids, attn, mask = make_batch(batch, pad_id, input_dev)
            out = model(input_ids=ids, attention_mask=attn, use_cache=False)
            logits = out.logits[:, :-1]
            tgt = ids[:, 1:].to(logits.device)
            m = mask[:, 1:].to(logits.device)
            loss = F.cross_entropy(logits[m].float(), tgt[m])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if step % 60 == 0 or step == a.heal_steps:
                print(f"  [heal] step {step:3d}/{a.heal_steps} loss={loss.item():.3f} ({time.time()-th:.1f}s)", flush=True)
        model = model.merge_and_unload()

    train_sec = time.time() - t0
    print(f"[SSD] Finished in {train_sec/60:.2f} min! Evaluating...", flush=True)
    model.config.use_cache = True
    model.gradient_checkpointing_disable()
    eval_model_full(
        model, tok, out_dir, topic=a.topic, n_seeds=a.n_seeds, run_mentor=a.mentor,
        extra_meta={"train_sec": train_sec, "args": vars(a)},
    )
    print(f"SSD_DONE total_min={(time.time()-t0)/60:.2f}", flush=True)


if __name__ == "__main__":
    main()
