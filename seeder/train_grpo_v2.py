"""Stage 3 (v2): Anti-Cliché + Cross-Prompt Anti-Repetition GRPO on `Qwen3-8B-Base` Seeder v2.

Upgrades over v1 (`train_grpo.py`):
  1. Rolling Cross-Prompt History Buffer (last 180 rollouts from DIFFERENT prompts):
     penalizes `-1.2 * max(0, max_recent_cross_sim - 0.62)` so the model CANNOT converge onto
     favorite "go-to metaphors" (e.g., mussel catechol or slime mold) across unrelated problems!
  2. Stronger quality & physical/practical realizability weight (`1.2 * sigmoid(skywork / 3.5)`)
     so high-novelty seeds also stay high-utility (`U >= 6`) and feasible (`F >= 6`).
  3. 160 online GRPO steps (`G=6`) with exact frozen-reference KL via PEFT `disable_adapter()`.
"""
import argparse
import json
import math
import random
import re
import sys
import time
from collections import deque
from pathlib import Path

import bitsandbytes as bnb
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

SEEDER_DIR = Path(__file__).resolve().parent
ROUND3_DIR = SEEDER_DIR.parent
sys.path.insert(0, str(ROUND3_DIR))

from eval_harness import eval_mcq_model  # noqa: E402
from naive_service import is_bad_seed  # noqa: E402
from runner import generate_batch  # noqa: E402
from seeder.train_grpo import LocalNoveltyReward, compute_token_logps  # noqa: E402
from seeder.train_sft import merge_lora_on_cpu  # noqa: E402


class CrossPromptAntiRepReward(LocalNoveltyReward):
    """Extends LocalNoveltyReward with a rolling cross-prompt motif-repetition penalty & stronger quality floor."""

    def __init__(self, device="cuda:1", history_size=180):
        super().__init__(device=device)
        self.history = deque(maxlen=history_size)  # list of (prob_id, unit_vec)

    def score_group(self, prompt_item, responses):
        pid = prompt_item["id"]
        prompt = prompt_item["prompt"]
        prob_vec = np.asarray(prompt_item["prob_vec"], dtype=np.float32)
        cliche_vec = np.asarray(prompt_item["cliche_vec"], dtype=np.float32)
        cliche_terms = [t for t in prompt_item.get("cliche_terms", []) if len(t.strip()) >= 4]

        vecs = self.embed(responses)
        sky = self.skywork_scores(prompt, responses)
        sim_mat = vecs @ vecs.T
        G = len(responses)

        other_vecs = [v for (hpid, v) in self.history if hpid != pid]
        other_mat = np.asarray(other_vecs, dtype=np.float32) if other_vecs else None

        rewards, details = [], []
        for i, resp in enumerate(responses):
            n_words = len(resp.split())
            bad = is_bad_seed(resp) or n_words < 60
            sim_cliche = float(vecs[i] @ cliche_vec)
            sim_prob = float(vecs[i] @ prob_vec)
            escape = 1.0 - sim_cliche
            rel = float(np.clip(sim_prob, 0.25, 0.80))
            div = float((G - float(sim_mat[i].sum())) / max(1, G - 1))
            q_sig = 1.0 / (1.0 + math.exp(-sky[i] / 3.5))

            # Cross-prompt motif repetition penalty!
            x_sim = float((other_mat @ vecs[i]).max()) if other_mat is not None else 0.0
            x_pen = max(0.0, x_sim - 0.62)

            hits = sum(1 for ct in cliche_terms if re.search(r"\b" + re.escape(ct) + r"\b", resp, re.I))
            hit_frac = hits / max(1, min(6, len(cliche_terms)))

            r = (
                2.0 * escape
                + 1.3 * rel
                + 0.9 * div
                + 1.2 * q_sig
                - 1.6 * hit_frac
                - 2.2 * x_pen
            )
            if bad:
                r -= 3.0
            if n_words < 90:
                r -= 0.4
            rewards.append(r)
            details.append({
                "r": round(r, 3), "esc": round(escape, 3), "rel": round(rel, 3),
                "div": round(div, 3), "sky": round(sky[i], 2), "xsim": round(x_sim, 3),
                "hits": hits, "bad": int(bad),
            })

        # Add top-2 non-bad rollouts into the cross-prompt history buffer
        order = np.argsort(-np.asarray(rewards))
        added = 0
        for idx in order:
            if not details[idx]["bad"]:
                self.history.append((pid, vecs[idx]))
                added += 1
                if added >= 2:
                    break
        return rewards, details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-model", default="/home/jimmy/code/creative-toolkit/models/seeder_8b_v2/sft_merged")
    ap.add_argument("--prompts", default=str(SEEDER_DIR / "data_v2" / "grpo_prompts.jsonl"))
    ap.add_argument("--out-dir", default="/home/jimmy/code/creative-toolkit/models/seeder_8b_v2/grpo_merged")
    ap.add_argument("--steps", type=int, default=160)
    ap.add_argument("--group-size", type=int, default=6)
    ap.add_argument("--max-new", type=int, default=280)
    ap.add_argument("--temperature", type=float, default=0.92)
    ap.add_argument("--lr", type=float, default=1.2e-5)
    ap.add_argument("--kl-beta", type=float, default=0.04)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    if (out_dir / "config.json").exists() and not a.rebuild:
        print(f"[stage3-grpo-v2] Already exists at {out_dir}, skipping.", flush=True)
        return

    t0 = time.time()
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)

    print(f"=== Stage 3 (v2): Anti-Cliché + Anti-Repetition GRPO on {a.sft_model} ({a.steps} steps) ===", flush=True)
    prompt_bank = [json.loads(l) for l in open(a.prompts) if l.strip()]
    rewarder = CrossPromptAntiRepReward(device="cuda:1", history_size=180)

    tok = AutoTokenizer.from_pretrained(a.sft_model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        a.sft_model, dtype=torch.bfloat16, device_map="auto", max_memory={0: "12.5GiB", 1: "8.5GiB"}
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lcfg = LoraConfig(
        r=a.lora_r, lora_alpha=a.lora_r * 2, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    peft_model = get_peft_model(model, lcfg)
    peft_model.print_trainable_parameters()

    train_params = [p for p in peft_model.parameters() if p.requires_grad]
    opt = bnb.optim.AdamW8bit(train_params, lr=a.lr, weight_decay=0.01)
    input_dev = next(peft_model.parameters()).device

    metrics_file = SEEDER_DIR / "data_v2" / "grpo_metrics.jsonl"
    metrics_fh = open(metrics_file, "w")

    for step in range(1, a.steps + 1):
        ts = time.time()
        item = rng.choice(prompt_bank)
        p_txt = tok.apply_chat_template([{"role": "user", "content": item["prompt"]}], tokenize=False, add_generation_prompt=True)
        enc = tok([p_txt] * a.group_size, return_tensors="pt", padding=True).to(input_dev)
        p_len = enc["input_ids"].shape[1]

        peft_model.eval()
        peft_model.gradient_checkpointing_disable()
        peft_model.config.use_cache = True
        with torch.inference_mode():
            gen = peft_model.generate(
                **enc, max_new_tokens=a.max_new, do_sample=True, temperature=a.temperature,
                top_p=0.95, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
            )
        comp_seqs, responses = [], []
        for gi in range(a.group_size):
            seq = gen[gi]
            comp = seq[p_len:]
            eos_pos = (comp == tok.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                seq = seq[: p_len + int(eos_pos[0].item()) + 1]
            comp_seqs.append(seq.detach().clone())
            responses.append(tok.decode(seq[p_len:], skip_special_tokens=True).strip())

        rewards, details = rewarder.score_group(item, responses)
        r_arr = np.asarray(rewards, dtype=np.float32)
        if float(r_arr.std()) < 1e-4:
            continue
        advs = (r_arr - float(r_arr.mean())) / (float(r_arr.std()) + 1e-6)

        ref_lps = []
        with torch.inference_mode():
            with peft_model.disable_adapter():
                for seq in comp_seqs:
                    ref_lps.append(None if len(seq) <= p_len + 2 else compute_token_logps(peft_model, seq, p_len).detach())

        peft_model.config.use_cache = False
        peft_model.gradient_checkpointing_enable()
        peft_model.train()
        lr = a.lr * min(1.0, step / 15.0) * 0.5 * (1.0 + math.cos(math.pi * step / a.steps))
        for g in opt.param_groups:
            g["lr"] = lr

        total_kl, valid_g = 0.0, 0
        for seq, adv, r_lp in zip(comp_seqs, advs, ref_lps):
            if r_lp is None:
                continue
            cur_lp = compute_token_logps(peft_model, seq, p_len)
            r_lp = r_lp.to(cur_lp.device)
            log_ratio = (r_lp - cur_lp).clamp(-10.0, 10.0)
            kl_tok = torch.exp(log_ratio) - log_ratio - 1.0
            loss = (-(float(adv) * cur_lp).mean() + a.kl_beta * kl_tok.mean()) / a.group_size
            loss.backward()
            total_kl += kl_tok.mean().item()
            valid_g += 1

        if valid_g > 0:
            torch.nn.utils.clip_grad_norm_(train_params, 1.0)
            opt.step()
        opt.zero_grad(set_to_none=True)

        row = {
            "step": step,
            "mean_r": round(float(r_arr.mean()), 3),
            "mean_esc": round(float(np.mean([d["esc"] for d in details])), 3),
            "mean_rel": round(float(np.mean([d["rel"] for d in details])), 3),
            "mean_div": round(float(np.mean([d["div"] for d in details])), 3),
            "mean_xsim": round(float(np.mean([d["xsim"] for d in details])), 3),
            "mean_sky": round(float(np.mean([d["sky"] for d in details])), 2),
            "bad_n": sum(d["bad"] for d in details),
            "kl": round(total_kl / max(1, valid_g), 4),
        }
        metrics_fh.write(json.dumps(row) + "\n")
        metrics_fh.flush()
        if step % 15 == 0 or step == 1 or step == a.steps:
            print(
                f"  [GRPO-v2 {step:3d}/{a.steps}] R={row['mean_r']:+.3f} | esc={row['mean_esc']:.3f} "
                f"rel={row['mean_rel']:.3f} div={row['mean_div']:.3f} xsim={row['mean_xsim']:.3f} "
                f"sky={row['mean_sky']:+.2f} bad={row['bad_n']}/{a.group_size} kl={row['kl']:.4f} ({(time.time()-t0)/60:.1f}m)",
                flush=True,
            )

    metrics_fh.close()
    del opt, rewarder
    for p in train_params:
        p.grad = None
    torch.cuda.empty_cache()

    base_model = merge_lora_on_cpu(peft_model)
    base_model.gradient_checkpointing_disable()
    base_model.config.use_cache = True
    base_model.eval()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    mcq = eval_mcq_model(base_model, tok, topic="dental", n_general=60)
    print(f"\n[GRPO-v2 SANITY] MCQ gen={mcq['general']['acc']:.3f} ({mcq['general']['margin']:+.2f})", flush=True)
    for sp in [
        "Can you help me invent something to clean between my teeth?",
        "Meetings at my company run long and decide little. Can you help me invent a way to fix this?",
    ]:
        ans = generate_batch(base_model, tok, sp, n=1, max_new=260, temperature=0.85, seed=0)[0]
        print(f"  SAMPLE [{sp[:45]}...]:\n    {ans[:300].replace(chr(10), ' ')}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    base_model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    json.dump({"sft_model": a.sft_model, "steps": a.steps, "mcq": mcq, "train_min": round((time.time()-t0)/60, 2)},
              open(out_dir / "grpo_meta.json", "w"), indent=2)
    print(f"STAGE3_V2_GRPO_DONE total_min={(time.time()-t0)/60:.2f}", flush=True)


if __name__ == "__main__":
    main()
