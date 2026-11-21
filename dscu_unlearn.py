"""Problem-Protected Dual-Space Contrastive Unlearning (DSCU) for fast, coherent subject forgetting.

Achieves deep, coherent, problem-preserving subject forgetting in ~8-12 minutes on 2x RTX 5070 Ti:
  Stage 0: Generate M_0 reference answers on forget_qa, problem_qa, and generic_invent prompts.
  Stage 1: Train Problem-Protected Hyper-Dogma Expert LoRA (M_exp) on D_forget + KL retain on D_problem & D_retain.
  Stage 2: Train Naive LoRA (r=64, all linear layers) with:
           - Bounded Anti-Dogma Logit KL: KL(softmax(z_0 - alpha * relu(dz)) || p_naive) + soft cap on solution tokens
           - Norm-Preserving Representation Neutralization at mid-layers: h_naive -> norm_match(h_0 - beta * (h_exp - h_0))
           - Strong Dual-Space Retain on D_problem + D_retain + generic_invent
  Stage 3: Optional Round-2 On-Policy Scrubbing on M_naive's own dental-prompt rollouts to catch second-order fallbacks.
  Stage 4: Evaluate MCQ (solutions/problem/general), probes, and 36 paper seeds (+ optional Gemini mentor & Luna judge).
"""
import argparse
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_harness import ROUND3_DIR, TOPICS
from runner import eval_model_full, format_chat, generate_batch
from topic_data import build_topic_data, compile_filter


def get_layers(model):
    m = model.base_model.model if hasattr(model, "base_model") else model
    return m.model.layers


def encode_item(tok, item, max_len=320):
    if item["type"] == "wiki":
        ids = tok(item["text"], add_special_tokens=True, truncation=True, max_length=max_len)["input_ids"]
        mask = [False] + [True] * (len(ids) - 1)
        return ids, mask
    # chat item: {"type": "chat", "messages": [...]}
    msgs = item["messages"]
    try:
        full_txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    except TypeError:
        full_txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    # Find assistant spans if possible; for simplicity and generality across Olmo/Qwen, mask user prompt on single-turn
    if len(msgs) == 2 and msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant":
        pre_txt = format_chat(tok, msgs[0]["content"])
        pre_ids = tok(pre_txt, add_special_tokens=False)["input_ids"]
        ans_ids = tok(msgs[1]["content"], add_special_tokens=False)["input_ids"]
        eos = [tok.eos_token_id] if tok.eos_token_id is not None else []
        ids = (pre_ids + ans_ids + eos)[:max_len]
        start = min(len(pre_ids), len(ids))
        mask = [False] * start + [True] * (len(ids) - start)
    else:
        ids = tok(full_txt, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]
        mask = [False] + [True] * (len(ids) - 1)
    if sum(mask) < 8:
        return None
    return ids, mask


def make_batch(examples, pad_id, device):
    max_l = max(len(x[0]) for x in examples)
    bs = len(examples)
    ids = torch.full((bs, max_l), pad_id, dtype=torch.long)
    attn = torch.zeros((bs, max_l), dtype=torch.long)
    mask = torch.zeros((bs, max_l), dtype=torch.bool)
    for i, (seq, m) in enumerate(examples):
        L = len(seq)
        ids[i, :L] = torch.tensor(seq, dtype=torch.long)
        attn[i, :L] = 1
        mask[i, :L] = torch.tensor(m, dtype=torch.bool)
    return ids.to(device), attn.to(device), mask.to(device)


def compute_solution_token_mask(tok, forget_encoded, retain_encoded, topic):
    """Identify tokens heavily enriched in dental-solution forget texts vs retain texts, plus filter-term tokens."""
    cf, cr = Counter(), Counter()
    for ids, m in forget_encoded:
        cf.update(t for t, k in zip(ids, m) if k)
    for ids, m in retain_encoded[:500]:
        cr.update(t for t, k in zip(ids, m) if k)
    nf, nr = max(1, sum(cf.values())), max(1, sum(cr.values()))
    sol_ids = set()
    for t, c in cf.items():
        if c >= 4:
            ratio = (c / nf) / ((cr[t] + 0.5) / nr)
            if ratio >= 15.0:
                sol_ids.add(t)
    # Also add tokens of explicit filter terms (excluding short stop-like subwords < 3 chars)
    pat_terms = TOPICS[topic]["terms"]
    if topic == "dental":
        pat_terms = pat_terms + ["brush", "brushes", "brushing", "bristle", "bristles", "floss", "paste", "rinse", "dentistry"]
    for term in pat_terms:
        for variant in (term, " " + term, term.capitalize(), " " + term.capitalize()):
            tids = tok(variant, add_special_tokens=False)["input_ids"]
            for tid in tids:
                piece = tok.decode([tid]).strip().lower()
                if len(piece) >= 3 and piece not in {"the", "and", "for", "your", "with", "from", "that", "this", "clean", "teeth", "tooth", "oral", "gum", "gums", "mouth", "water", "care", "oil"}:
                    sol_ids.add(tid)
    vec = torch.zeros(len(tok), dtype=torch.bool)
    for tid in sol_ids:
        if 0 <= tid < len(tok):
            vec[tid] = True
    return vec, sorted(sol_ids)


def forward_with_reps(model, ids, attn, rep_layers):
    out = model(input_ids=ids, attention_mask=attn, output_hidden_states=True, use_cache=False)
    reps = [out.hidden_states[l + 1] for l in rep_layers]
    return out.logits, reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/Olmo-3-7B-Instruct-SFT")
    ap.add_argument("--topic", default="dental")
    ap.add_argument("--out", default="results/dscu_olmo_v1")
    ap.add_argument("--exp_steps", type=int, default=160)
    ap.add_argument("--unlearn_steps", type=int, default=320)
    ap.add_argument("--round2_steps", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=5.0, help="logit contrast strength")
    ap.add_argument("--beta", type=float, default=1.5, help="representation neutralization strength")
    ap.add_argument("--rep_w", type=float, default=0.5, help="weight of representation neutralization loss")
    ap.add_argument("--retain_w", type=float, default=2.0, help="weight of retain KL+rep loss")
    ap.add_argument("--problem_w", type=float, default=2.5, help="weight of problem retain KL+rep loss")
    ap.add_argument("--cap_k", type=int, default=25, help="demote saturated solution tokens to k-th logit")
    ap.add_argument("--lr_exp", type=float, default=2e-4)
    ap.add_argument("--lr_naive", type=float, default=1.5e-4)
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=320)
    ap.add_argument("--n_seeds", type=int, default=12)
    ap.add_argument("--mentor", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    t_start = time.time()
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    out_dir = ROUND3_DIR / a.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== DSCU Unlearning: model={a.model} topic={a.topic} out={out_dir} ===", flush=True)
    data = build_topic_data(a.topic)

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    print("Loading base model in bf16 across 2 GPUs...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        a.model,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "9GiB", 1: "9GiB"},
    )
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()
    input_dev = next(base_model.parameters()).device
    num_layers = len(base_model.model.layers)
    rep_layers = [int(num_layers * f) for f in (0.30, 0.45, 0.60, 0.75)]
    print(f"Model loaded ({time.time() - t_start:.1f}s). num_layers={num_layers}, rep_layers={rep_layers}", flush=True)

    # ---- Stage 0: Cache M_0 self-generated Q&A for forget, problem, and generic_invent ----
    qa_cache = ROUND3_DIR / "data" / a.topic / f"qa_refs_{a.model.split('/')[-1]}.json"
    if qa_cache.exists():
        qa_refs = json.load(open(qa_cache))
        print(f"Loaded cached M_0 QA refs from {qa_cache}", flush=True)
    else:
        print("Stage 0: Generating M_0 reference answers (forget_qa, problem_qa, generic_invent)...", flush=True)
        base_model.eval()
        base_model.config.use_cache = True
        qa_refs = {"forget_qa": [], "problem_qa": [], "generic_qa": []}
        for key, plist, n_per in [
            ("forget_qa", data["forget_qa_prompts"], 2),
            ("problem_qa", data["problem_qa_prompts"], 2),
            ("generic_qa", data["generic_invent"], 2),
        ]:
            for pi, q in enumerate(plist):
                ans_list = generate_batch(base_model, tok, q, n=n_per, max_new=180, temperature=0.7, seed=pi)
                for ans in ans_list:
                    qa_refs[key].append({"type": "chat", "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]})
        base_model.config.use_cache = False
        json.dump(qa_refs, open(qa_cache, "w"), indent=1)
        print(
            f"Generated QA refs in {time.time() - t_start:.1f}s: "
            f"forget_qa={len(qa_refs['forget_qa'])} problem_qa={len(qa_refs['problem_qa'])} generic_qa={len(qa_refs['generic_qa'])}",
            flush=True,
        )

    # Encode all datasets
    forget_items = data["forget_wiki"] + data["forget_conv"][:250] + qa_refs["forget_qa"]
    problem_items = data["problem_wiki"] + qa_refs["problem_qa"]
    retain_items = data["retain_wiki"] + data["retain_conv"][:600] + qa_refs["generic_qa"]

    forget_enc = [e for it in forget_items if (e := encode_item(tok, it, a.max_len))]
    problem_enc = [e for it in problem_items if (e := encode_item(tok, it, a.max_len))]
    retain_enc = [e for it in retain_items if (e := encode_item(tok, it, a.max_len))]
    print(f"Encoded: forget={len(forget_enc)} problem={len(problem_enc)} retain={len(retain_enc)}", flush=True)

    sol_mask_cpu, sol_ids = compute_solution_token_mask(tok, forget_enc, retain_enc, a.topic)
    print(f"Identified {len(sol_ids)} topic-solution tokens, e.g.: {[tok.decode([i]) for i in sol_ids[:40]]}", flush=True)

    # Attach PEFT adapters: "expert" (r=32) and "naive" (r=a.lora_r) on the SINGLE bf16 base model!
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    model = get_peft_model(
        base_model,
        LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, target_modules=targets, task_type="CAUSAL_LM"),
        adapter_name="expert",
    )
    model.add_adapter(
        "naive",
        LoraConfig(r=a.lora_r, lora_alpha=a.lora_r * 2, lora_dropout=0.0, target_modules=targets, task_type="CAUSAL_LM"),
    )

    # ---- Stage 1: Train Problem-Protected Hyper-Dogma Expert ("expert") ----
    t1 = time.time()
    print(f"\n=== Stage 1: Training Problem-Protected Hyper-Dogma Expert ({a.exp_steps} steps) ===", flush=True)
    model.set_adapter("expert")
    model.train()
    exp_params = [p for n, p in model.named_parameters() if "expert" in n and p.requires_grad]
    opt_exp = torch.optim.AdamW(exp_params, lr=a.lr_exp, weight_decay=0.0)

    for step in range(1, a.exp_steps + 1):
        lr = a.lr_exp * min(1.0, step / 20.0) * 0.5 * (1.0 + math.cos(math.pi * step / a.exp_steps))
        for g in opt_exp.param_groups:
            g["lr"] = lr

        # Forget CE loss (overfit dental solutions!)
        f_batch = [rng.choice(forget_enc) for _ in range(a.bs)]
        f_ids, f_attn, f_mask = make_batch(f_batch, pad_id, input_dev)
        out_f = model(input_ids=f_ids, attention_mask=f_attn, use_cache=False)
        logits_f = out_f.logits[:, :-1]
        tgt_f = f_ids[:, 1:].to(logits_f.device)
        m_f = f_mask[:, 1:].to(logits_f.device)
        loss_f = F.cross_entropy(logits_f[m_f].float(), tgt_f[m_f])

        # Problem-protection KL loss (keep expert == base on tooth anatomy & general!)
        p_batch = [rng.choice(problem_enc) if rng.random() < 0.7 else rng.choice(retain_enc) for _ in range(a.bs)]
        p_ids, p_attn, p_mask = make_batch(p_batch, pad_id, input_dev)
        with torch.no_grad():
            with model.disable_adapter():
                z0_p = model(input_ids=p_ids, attention_mask=p_attn, use_cache=False).logits[:, :-1]
        zexp_p = model(input_ids=p_ids, attention_mask=p_attn, use_cache=False).logits[:, :-1]
        m_p = p_mask[:, 1:].to(zexp_p.device)
        kl_p = F.kl_div(
            F.log_softmax(zexp_p[m_p].float(), dim=-1),
            F.softmax(z0_p[m_p].float(), dim=-1),
            reduction="batchmean",
        )

        loss = loss_f + 1.0 * kl_p
        loss.backward()
        torch.nn.utils.clip_grad_norm_(exp_params, 1.0)
        opt_exp.step()
        opt_exp.zero_grad(set_to_none=True)
        if step % 40 == 0 or step == a.exp_steps:
            print(f"  [expert] step {step:3d}/{a.exp_steps} ce={loss_f.item():.3f} prob_kl={kl_p.item():.4f} ({time.time()-t1:.1f}s)", flush=True)

    # Freeze expert params
    for p in exp_params:
        p.requires_grad_(False)

    # ---- Helper for Dual-Space Unlearning Step ----
    def run_unlearn_phase(phase_name, forget_pool, n_steps, lr_max, opt_naive, naive_params):
        tp = time.time()
        print(f"\n=== {phase_name} ({n_steps} steps, forget_pool={len(forget_pool)}) ===", flush=True)
        model.set_adapter("naive")
        model.train()
        for step in range(1, n_steps + 1):
            lr = lr_max * min(1.0, step / 25.0) * 0.5 * (1.0 + math.cos(math.pi * step / n_steps))
            for g in opt_naive.param_groups:
                g["lr"] = lr

            # --- 1. Forget batch: Dual-Space Contrastive Anti-Dogma ---
            f_batch = [rng.choice(forget_pool) for _ in range(a.bs)]
            f_ids, f_attn, f_mask = make_batch(f_batch, pad_id, input_dev)

            with torch.no_grad():
                with model.disable_adapter():
                    z0_f, h0_f = forward_with_reps(model, f_ids, f_attn, rep_layers)
                model.set_adapter("expert")
                zexp_f, hexp_f = forward_with_reps(model, f_ids, f_attn, rep_layers)
                model.set_adapter("naive")

                # Compute coherent anti-dogma target distribution p_target
                out_dev = z0_f.device
                m_f = f_mask[:, 1:].to(out_dev)
                z0_v = z0_f[:, :-1][m_f].float()
                zexp_v = zexp_f[:, :-1][m_f].float()
                tgt_toks = f_ids[:, 1:].to(out_dev)[m_f]

                # Centered logit difference (robust to shift)
                dz = (zexp_v - zexp_v.mean(dim=-1, keepdim=True)) - (z0_v - z0_v.mean(dim=-1, keepdim=True))
                z_tgt = z0_v - a.alpha * F.relu(dz)

                # Soft ceiling on saturated topic-solution tokens (prevents unmoved 0.99-confidence dogma!)
                sol_mask = sol_mask_cpu.to(out_dev)
                is_sol = sol_mask[tgt_toks]
                if is_sol.any():
                    # Top-k logit threshold for those rows
                    kth = torch.topk(z0_v[is_sol], k=a.cap_k, dim=-1).values[:, -1]
                    rows = torch.nonzero(is_sol, as_tuple=False).squeeze(-1)
                    cols = tgt_toks[is_sol]
                    z_tgt[rows, cols] = torch.minimum(z_tgt[rows, cols], kth)
                    # Also cap ALL solution-vocab tokens slightly wherever the context triggered a solution token
                    sol_cols = torch.nonzero(sol_mask, as_tuple=False).squeeze(-1)
                    z_tgt[rows.unsqueeze(1), sol_cols.unsqueeze(0)] = torch.minimum(
                        z_tgt[rows.unsqueeze(1), sol_cols.unsqueeze(0)],
                        kth.unsqueeze(1),
                    )
                p_tgt = F.softmax(z_tgt, dim=-1)

                # Compute norm-preserving neutralized hidden-state targets at mid-layers
                h_tgts = []
                for h0, he in zip(h0_f, hexp_f):
                    mh = f_mask.to(h0.device)
                    h0_v = h0[mh].float()
                    he_v = he[mh].float()
                    dh = he_v - h0_v
                    ht = h0_v - a.beta * dh
                    # Preserve original token norm -> zero OOD activation explosion!
                    orig_norm = h0_v.norm(dim=-1, keepdim=True).clamp(min=1e-5)
                    ht = ht * (orig_norm / ht.norm(dim=-1, keepdim=True).clamp(min=1e-5))
                    h_tgts.append(ht.detach())

            # Forward naive on forget batch
            zn_f, hn_f = forward_with_reps(model, f_ids, f_attn, rep_layers)
            zn_v = zn_f[:, :-1][m_f].float()
            loss_f_kl = F.kl_div(F.log_softmax(zn_v, dim=-1), p_tgt, reduction="batchmean")

            loss_f_rep = torch.tensor(0.0, device=out_dev)
            for hn, ht in zip(hn_f, h_tgts):
                mh = f_mask.to(hn.device)
                hn_v = hn[mh].float()
                # Normalize MSE by mean squared norm so scale is ~O(1)
                denom = ht.pow(2).mean().clamp(min=1e-5)
                loss_f_rep = loss_f_rep + ((hn_v - ht.to(hn.device)).pow(2).mean() / denom).to(out_dev)
            loss_f_rep = loss_f_rep / len(rep_layers)

            (loss_f_kl + a.rep_w * loss_f_rep).backward()
            del z0_f, zexp_f, zn_f, h0_f, hexp_f, hn_f, h_tgts

            # --- 2. Retain Problem batch (protect tooth anatomy/problem knowledge!) ---
            p_batch = [rng.choice(problem_enc) for _ in range(a.bs)]
            p_ids, p_attn, p_mask = make_batch(p_batch, pad_id, input_dev)
            with torch.no_grad():
                with model.disable_adapter():
                    z0_p, h0_p = forward_with_reps(model, p_ids, p_attn, rep_layers)
            zn_p, hn_p = forward_with_reps(model, p_ids, p_attn, rep_layers)
            out_dev = zn_p.device
            m_p = p_mask[:, 1:].to(out_dev)
            loss_p_kl = F.kl_div(
                F.log_softmax(zn_p[:, :-1][m_p].float(), dim=-1),
                F.softmax(z0_p[:, :-1][m_p].float(), dim=-1),
                reduction="batchmean",
            )
            loss_p_rep = torch.tensor(0.0, device=out_dev)
            for hn, h0 in zip(hn_p, h0_p):
                mh = p_mask.to(hn.device)
                h0_v = h0[mh].float()
                denom = h0_v.pow(2).mean().clamp(min=1e-5)
                loss_p_rep = loss_p_rep + ((hn[mh].float() - h0_v).pow(2).mean() / denom).to(out_dev)
            loss_p_rep = loss_p_rep / len(rep_layers)
            (a.problem_w * (loss_p_kl + a.rep_w * loss_p_rep)).backward()
            del z0_p, zn_p, h0_p, hn_p

            # --- 3. Retain General + General Invention batch ---
            r_batch = [rng.choice(retain_enc) for _ in range(a.bs)]
            r_ids, r_attn, r_mask = make_batch(r_batch, pad_id, input_dev)
            with torch.no_grad():
                with model.disable_adapter():
                    z0_r, h0_r = forward_with_reps(model, r_ids, r_attn, rep_layers)
            zn_r, hn_r = forward_with_reps(model, r_ids, r_attn, rep_layers)
            out_dev = zn_r.device
            m_r = r_mask[:, 1:].to(out_dev)
            loss_r_kl = F.kl_div(
                F.log_softmax(zn_r[:, :-1][m_r].float(), dim=-1),
                F.softmax(z0_r[:, :-1][m_r].float(), dim=-1),
                reduction="batchmean",
            )
            loss_r_rep = torch.tensor(0.0, device=out_dev)
            for hn, h0 in zip(hn_r, h0_r):
                mh = r_mask.to(hn.device)
                h0_v = h0[mh].float()
                denom = h0_v.pow(2).mean().clamp(min=1e-5)
                loss_r_rep = loss_r_rep + ((hn[mh].float() - h0_v).pow(2).mean() / denom).to(out_dev)
            loss_r_rep = loss_r_rep / len(rep_layers)
            (a.retain_w * (loss_r_kl + a.rep_w * loss_r_rep)).backward()
            del z0_r, zn_r, h0_r, hn_r

            torch.nn.utils.clip_grad_norm_(naive_params, 1.0)
            opt_naive.step()
            opt_naive.zero_grad(set_to_none=True)

            if step % 40 == 0 or step == n_steps:
                print(
                    f"  [{phase_name}] step {step:3d}/{n_steps} "
                    f"f_kl={loss_f_kl.item():.3f} f_rep={loss_f_rep.item():.3f} "
                    f"p_kl={loss_p_kl.item():.4f} r_kl={loss_r_kl.item():.4f} ({time.time()-tp:.1f}s)",
                    flush=True,
                )

    # ---- Stage 2: Main DSCU Unlearning ----
    model.set_adapter("naive")
    naive_params = [p for n, p in model.named_parameters() if "naive" in n and p.requires_grad]
    opt_naive = torch.optim.AdamW(naive_params, lr=a.lr_naive, weight_decay=0.0)
    run_unlearn_phase("Stage2-DSCU", forget_enc, a.unlearn_steps, a.lr_naive, opt_naive, naive_params)

    # ---- Stage 3: Round-2 On-Policy Scrubbing (catches fallback dogma!) ----
    if a.round2_steps > 0:
        print("\n=== Stage 3: Sampling On-Policy Rollouts for Round-2 Scrubbing ===", flush=True)
        model.eval()
        model.set_adapter("naive")
        model.config.use_cache = True
        pat = compile_filter(a.topic)
        onpolicy_prompts = data["forget_qa_prompts"] + [
            "How can I remove plaque from hard-to-reach molars?",
            "Suggest a daily method to clean buildup off teeth.",
            "What tool could clean gaps between adjacent teeth?",
            "How would someone keep their teeth free of food debris?",
            "Invent a simple household way to remove film from teeth.",
            "What is the best way to keep teeth clean after eating?",
        ]
        onpolicy_enc = []
        hits = 0
        for pi, q in enumerate(onpolicy_prompts):
            ans_list = generate_batch(model, tok, q, n=2, max_new=180, temperature=0.8, seed=500 + pi)
            for ans in ans_list:
                if pat.search(ans):
                    hits += 1
                    e = encode_item(tok, {"type": "chat", "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]}, a.max_len)
                    if e:
                        onpolicy_enc.append(e)
        model.config.use_cache = False
        print(f"On-policy rollouts hitting dental dogma: {hits}/{len(onpolicy_prompts)*2} -> scrubbing {len(onpolicy_enc)} rollouts", flush=True)
        if onpolicy_enc:
            mixed_pool = onpolicy_enc * 3 + forget_enc[:100]
            opt_r2 = torch.optim.AdamW(naive_params, lr=a.lr_naive * 0.5, weight_decay=0.0)
            run_unlearn_phase("Stage3-OnPolicy", mixed_pool, a.round2_steps, a.lr_naive * 0.5, opt_r2, naive_params)

    train_sec = time.time() - t_start
    print(f"\n=== Training finished in {train_sec / 60:.2f} minutes! Saving adapter & running evaluation... ===", flush=True)
    model.set_adapter("naive")
    model.save_pretrained(out_dir / "adapter_naive", selected_adapters=["naive"])
    model.config.use_cache = True
    model.gradient_checkpointing_disable()

    eval_model_full(
        model,
        tok,
        out_dir,
        topic=a.topic,
        n_seeds=a.n_seeds,
        run_mentor=a.mentor,
        extra_meta={"train_sec": train_sec, "args": vars(a)},
    )
    print(f"DSCU_DONE total_min={(time.time() - t_start) / 60:.2f}", flush=True)


if __name__ == "__main__":
    main()
