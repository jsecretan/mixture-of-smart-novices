"""Stage 3: 100% Local Anti-Cliché GRPO (Online RLVR) on `Qwen3-8B-Base` Seeder SFT.

Key memory & speed innovations on 2x 16GB RTX 5070 Ti:
  - Uses PEFT Rank-32 LoRA on `sft_merged` and computes exact frozen-reference logprobs via
    `with peft_model.disable_adapter():` — ZERO second copy of the 8B model in VRAM!
  - Hosts local reward models (`Qwen3-Embedding-0.6B` + `Skywork-Reward-V2-Qwen3-1.7B`, ~4.6 GB)
    strictly on GPU 1, leaving GPU 0 free for fast batched rollout generation & LoRA gradients.
  - Composite 100% local reward per rollout y_i in group G=6:
      R(x, y_i) = 2.0 * (1 - cos(e(y_i), mu_cliche(x)))      # escape conventional solutions!
                + 1.2 * clip(cos(e(y_i), e(x)), 0.30, 0.80)  # stay anchored on the real problem!
                + 0.8 * mean_{j!=i}(1 - cos(e(y_i), e(y_j))) # within-group diversity (anti-collapse)!
                + 0.7 * sigmoid(skywork_score / 4.0)         # coherence & engineering quality floor!
                - 1.5 * cliche_term_hit_fraction             # penalize explicit cliché naming!
                - 3.0 * is_bad_or_short                      # kill refusals, stalling, gibberish!
"""
import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import bitsandbytes as bnb
import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

SEEDER_DIR = Path(__file__).resolve().parent
ROUND3_DIR = SEEDER_DIR.parent
sys.path.insert(0, str(ROUND3_DIR))

from eval_harness import eval_mcq_model  # noqa: E402
from naive_service import is_bad_seed  # noqa: E402
from runner import generate_batch  # noqa: E402
from seeder.train_sft import merge_lora_on_cpu  # noqa: E402


class LocalNoveltyReward:
    """100% local composite reward model on GPU 1 (~4.6 GB VRAM, zero API calls)."""

    def __init__(self, device="cuda:1"):
        self.device = device
        print(f"[reward] Loading Qwen3-Embedding-0.6B + Skywork-Reward-V2-Qwen3-1.7B on {device}...", flush=True)
        self.emb_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
        self.emb_model = AutoModel.from_pretrained(
            "Qwen/Qwen3-Embedding-0.6B", dtype=torch.bfloat16
        ).to(device).eval()

        self.rm_tok = AutoTokenizer.from_pretrained("Skywork/Skywork-Reward-V2-Qwen3-1.7B")
        self.rm_model = AutoModelForSequenceClassification.from_pretrained(
            "Skywork/Skywork-Reward-V2-Qwen3-1.7B", dtype=torch.bfloat16, num_labels=1
        ).to(device).eval()

    @torch.inference_mode()
    def embed(self, texts):
        enc = self.emb_tok(texts, padding=True, truncation=True, max_length=320, return_tensors="pt").to(self.device)
        out = self.emb_model(**enc).last_hidden_state
        lens = enc["attention_mask"].sum(dim=1) - 1
        v = out[torch.arange(len(texts), device=self.device), lens].float()
        return (v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)).cpu().numpy()

    @torch.inference_mode()
    def skywork_scores(self, prompt, responses):
        scores = []
        for resp in responses:
            conv = [{"role": "user", "content": prompt}, {"role": "assistant", "content": resp}]
            txt = self.rm_tok.apply_chat_template(conv, tokenize=False)
            if self.rm_tok.bos_token and txt.startswith(self.rm_tok.bos_token):
                txt = txt[len(self.rm_tok.bos_token):]
            enc = self.rm_tok(txt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
            sc = float(self.rm_model(**enc).logits[0, 0].item())
            scores.append(sc)
        return scores

    def score_group(self, prompt_item, responses):
        prompt = prompt_item["prompt"]
        prob_vec = np.asarray(prompt_item["prob_vec"], dtype=np.float32)
        cliche_vec = np.asarray(prompt_item["cliche_vec"], dtype=np.float32)
        cliche_terms = [t for t in prompt_item.get("cliche_terms", []) if len(t.strip()) >= 4]

        vecs = self.embed(responses)
        sky = self.skywork_scores(prompt, responses)
        sim_mat = vecs @ vecs.T
        G = len(responses)

        rewards = []
        details = []
        for i, resp in enumerate(responses):
            n_words = len(resp.split())
            bad = is_bad_seed(resp) or n_words < 55
            sim_cliche = float(vecs[i] @ cliche_vec)
            sim_prob = float(vecs[i] @ prob_vec)
            escape = 1.0 - sim_cliche
            rel = float(np.clip(sim_prob, 0.25, 0.80))
            div = float((G - float(sim_mat[i].sum())) / max(1, G - 1))  # mean pairwise 1 - cos
            q_sig = 1.0 / (1.0 + math.exp(-sky[i] / 4.0))

            hits = sum(1 for ct in cliche_terms if re.search(r"\b" + re.escape(ct) + r"\b", resp, re.I))
            hit_frac = hits / max(1, min(6, len(cliche_terms)))

            r = 2.0 * escape + 1.2 * rel + 0.8 * div + 0.7 * q_sig - 1.5 * hit_frac
            if bad:
                r -= 3.0
            if n_words < 85:
                r -= 0.4
            rewards.append(r)
            details.append({
                "r": round(r, 3), "esc": round(escape, 3), "rel": round(rel, 3),
                "div": round(div, 3), "sky": round(sky[i], 2), "hits": hits, "bad": int(bad),
            })
        return rewards, details


def compute_token_logps(model, seq_ids, prompt_len):
    """Return per-token logprobs for completion tokens seq_ids[prompt_len:] (shape [comp_len])."""
    ids = seq_ids.unsqueeze(0)
    out = model(input_ids=ids, use_cache=False)
    logits = out.logits[0, prompt_len - 1 : -1].float()
    tgt = seq_ids[prompt_len:].to(logits.device)
    lps = F.log_softmax(logits, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return lps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-model", default="/home/jimmy/code/creative-toolkit/models/seeder_8b/sft_merged")
    ap.add_argument("--prompts", default=str(SEEDER_DIR / "data" / "grpo_prompts.jsonl"))
    ap.add_argument("--out-dir", default="/home/jimmy/code/creative-toolkit/models/seeder_8b/grpo_merged")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--group-size", type=int, default=6)
    ap.add_argument("--max-new", type=int, default=260)
    ap.add_argument("--temperature", type=float, default=0.92)
    ap.add_argument("--lr", type=float, default=1.2e-5)
    ap.add_argument("--kl-beta", type=float, default=0.04)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    if (out_dir / "config.json").exists() and not a.rebuild:
        print(f"[stage3-grpo] Merged checkpoint already exists at {out_dir}, skipping (pass --rebuild to overwrite).", flush=True)
        return

    t0 = time.time()
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)

    print(f"=== Stage 3: Anti-Cliché GRPO on {a.sft_model} ({a.steps} steps, G={a.group_size}) ===", flush=True)
    prompt_bank = [json.loads(l) for l in open(a.prompts) if l.strip()]
    print(f"Loaded {len(prompt_bank)} held-out GRPO training prompts.", flush=True)

    # Load local reward models on GPU 1 first so device_map='auto' accounts for their ~4.6 GB VRAM!
    rewarder = LocalNoveltyReward(device="cuda:1")

    tok = AutoTokenizer.from_pretrained(a.sft_model)
    tok.padding_side = "left"
    # Give GPU 0 up to 12.5 GiB and GPU 1 up to 8.5 GiB (leaving ~5 GB on GPU 1 for reward models + activations)
    model = AutoModelForCausalLM.from_pretrained(
        a.sft_model, dtype=torch.bfloat16, device_map="auto", max_memory={0: "12.5GiB", 1: "8.5GiB"}
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lcfg = LoraConfig(
        r=a.lora_r,
        lora_alpha=a.lora_r * 2,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    peft_model = get_peft_model(model, lcfg)
    peft_model.print_trainable_parameters()

    train_params = [p for p in peft_model.parameters() if p.requires_grad]
    opt = bnb.optim.AdamW8bit(train_params, lr=a.lr, weight_decay=0.01)
    input_dev = next(peft_model.parameters()).device

    metrics_file = SEEDER_DIR / "data" / "grpo_metrics.jsonl"
    metrics_fh = open(metrics_file, "w")

    for step in range(1, a.steps + 1):
        ts = time.time()
        item = rng.choice(prompt_bank)
        prompt = item["prompt"]
        p_txt = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        enc = tok([p_txt] * a.group_size, return_tensors="pt", padding=True).to(input_dev)
        p_len = enc["input_ids"].shape[1]

        # 1. Sample G rollouts online from current policy
        peft_model.eval()
        peft_model.gradient_checkpointing_disable()
        peft_model.config.use_cache = True
        with torch.inference_mode():
            gen = peft_model.generate(
                **enc,
                max_new_tokens=a.max_new,
                do_sample=True,
                temperature=a.temperature,
                top_p=0.95,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        comp_seqs = []
        responses = []
        for gi in range(a.group_size):
            seq = gen[gi]
            comp = seq[p_len:]
            # Trim trailing pad tokens after EOS
            eos_pos = (comp == tok.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                end_i = int(eos_pos[0].item()) + 1
                seq = seq[: p_len + end_i]
            comp_seqs.append(seq.detach().clone())
            responses.append(tok.decode(seq[p_len:], skip_special_tokens=True).strip())

        # 2. Score group locally on GPU 1 & compute group-relative advantages
        rewards, details = rewarder.score_group(item, responses)
        r_arr = np.asarray(rewards, dtype=np.float32)
        r_std = float(r_arr.std())
        if r_std < 1e-4:
            continue
        advs = (r_arr - float(r_arr.mean())) / (r_std + 1e-6)

        # 3. Compute exact frozen reference logprobs via PEFT disable_adapter() (0 extra VRAM!)
        ref_lps = []
        with torch.inference_mode():
            with peft_model.disable_adapter():
                for seq in comp_seqs:
                    if len(seq) <= p_len + 2:
                        ref_lps.append(None)
                    else:
                        ref_lps.append(compute_token_logps(peft_model, seq, p_len).detach())

        # 4. GRPO policy-gradient + reverse-KL update on current LoRA policy
        peft_model.config.use_cache = False
        peft_model.gradient_checkpointing_enable()
        peft_model.train()

        # Cosine LR schedule
        lr = a.lr * min(1.0, step / 15.0) * 0.5 * (1.0 + math.cos(math.pi * step / a.steps))
        for g in opt.param_groups:
            g["lr"] = lr

        total_pg, total_kl, valid_g = 0.0, 0.0, 0
        for gi, (seq, adv, r_lp) in enumerate(zip(comp_seqs, advs, ref_lps)):
            if r_lp is None:
                continue
            cur_lp = compute_token_logps(peft_model, seq, p_len)
            r_lp = r_lp.to(cur_lp.device)
            # Importance ratio (1.0 on-policy for 1 epoch, bounded KL via Schulman k3 estimator!)
            log_ratio = (r_lp - cur_lp).clamp(-10.0, 10.0)
            kl_tok = torch.exp(log_ratio) - log_ratio - 1.0
            pg_loss = -(float(adv) * cur_lp).mean()
            kl_loss = kl_tok.mean()
            loss = (pg_loss + a.kl_beta * kl_loss) / a.group_size
            loss.backward()
            total_pg += pg_loss.item()
            total_kl += kl_loss.item()
            valid_g += 1

        if valid_g > 0:
            torch.nn.utils.clip_grad_norm_(train_params, 1.0)
            opt.step()
        opt.zero_grad(set_to_none=True)

        row = {
            "step": step,
            "mean_r": round(float(r_arr.mean()), 3),
            "max_r": round(float(r_arr.max()), 3),
            "mean_esc": round(float(np.mean([d["esc"] for d in details])), 3),
            "mean_rel": round(float(np.mean([d["rel"] for d in details])), 3),
            "mean_div": round(float(np.mean([d["div"] for d in details])), 3),
            "mean_sky": round(float(np.mean([d["sky"] for d in details])), 2),
            "bad_n": sum(d["bad"] for d in details),
            "kl": round(total_kl / max(1, valid_g), 4),
            "sec": round(time.time() - ts, 1),
        }
        metrics_fh.write(json.dumps(row) + "\n")
        metrics_fh.flush()
        if step % 10 == 0 or step == 1 or step == a.steps:
            print(
                f"  [GRPO {step:3d}/{a.steps}] R={row['mean_r']:+.3f} (max={row['max_r']:+.2f}) | "
                f"esc={row['mean_esc']:.3f} rel={row['mean_rel']:.3f} div={row['mean_div']:.3f} "
                f"sky={row['mean_sky']:+.2f} bad={row['bad_n']}/{a.group_size} kl={row['kl']:.4f} "
                f"({row['sec']:.1f}s, tot={(time.time()-t0)/60:.1f}m)",
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
    print(
        f"\n[GRPO SANITY] MCQ gen={mcq['general']['acc']:.3f} ({mcq['general']['margin']:+.2f}) | "
        f"dental_prob={mcq['problem']['acc']:.3f} | dental_sol={mcq['solutions']['acc']:.3f}",
        flush=True,
    )
    for sp in [
        "Can you help me invent something to clean between my teeth?",
        "I want to train a neural network but I don't want to use gradients. Can you help me invent a way to train a network without using gradients?",
    ]:
        ans = generate_batch(base_model, tok, sp, n=1, max_new=260, temperature=0.85, seed=0)[0]
        print(f"  SAMPLE [{sp[:45]}...]:\n    {ans[:320].replace(chr(10), ' ')}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving standalone GRPO checkpoint -> {out_dir} ...", flush=True)
    base_model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    json.dump({
        "sft_model": a.sft_model,
        "steps": a.steps,
        "group_size": a.group_size,
        "kl_beta": a.kl_beta,
        "mcq": mcq,
        "train_min": round((time.time() - t0) / 60.0, 2),
    }, open(out_dir / "grpo_meta.json", "w"), indent=2)
    print(f"STAGE3_GRPO_DONE total_min={(time.time()-t0)/60:.2f}", flush=True)


if __name__ == "__main__":
    main()
