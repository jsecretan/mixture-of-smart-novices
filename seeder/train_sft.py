"""Stage 2: First-Principles Seeder SFT on clean `Qwen/Qwen3-8B-Base` (zero vendor RLHF!).

- Attaches standard Qwen3 ChatML template (`<|im_start|>`, `<|im_end|>`) directly to the tokenizer
  so the resulting checkpoint plugs seamlessly into `iacf_v2.py`, `eval_harness.py`, and `mosn/`.
- Trains Rank-64 bf16 LoRA across all attention + MLP projections (`q,k,v,o,gate,up,down_proj`)
  on `seeder/data/sft_train.jsonl` with cross-entropy strictly masked to assistant + `<|im_end|>` tokens.
- Merges LoRA cleanly on CPU (0 MiB GPU delta) and saves a standalone HF checkpoint to
  `/home/jimmy/code/creative-toolkit/models/seeder_8b/sft_merged/`.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import bitsandbytes as bnb
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

SEEDER_DIR = Path(__file__).resolve().parent
ROUND3_DIR = SEEDER_DIR.parent
ROOT = ROUND3_DIR.parent
sys.path.insert(0, str(ROUND3_DIR))

from eval_harness import eval_mcq_model  # noqa: E402
from runner import generate_batch  # noqa: E402

CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def configure_chatml_tokenizer(tok):
    tok.chat_template = CHATML_TEMPLATE
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id != tok.unk_token_id:
        tok.eos_token = "<|im_end|>"
        tok.eos_token_id = im_end_id
    if tok.pad_token_id is None:
        endoftext_id = tok.convert_tokens_to_ids("<|endoftext|>")
        if endoftext_id is not None and endoftext_id != tok.unk_token_id:
            tok.pad_token = "<|endoftext|>"
            tok.pad_token_id = endoftext_id
        else:
            tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def encode_sft_example(tok, prompt, response, max_len=768):
    prefix_txt = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    resp_txt = response.strip() + "<|im_end|>\n"
    p_ids = tok(prefix_txt, add_special_tokens=False)["input_ids"]
    r_ids = tok(resp_txt, add_special_tokens=False)["input_ids"]
    if len(p_ids) >= max_len - 32 or len(r_ids) < 16:
        return None
    r_ids = r_ids[: max_len - len(p_ids)]
    ids = p_ids + r_ids
    mask = [0] * len(p_ids) + [1] * len(r_ids)
    return ids, mask


def pad_batch(batch, pad_id, dev):
    mx = max(len(x[0]) for x in batch)
    ids = torch.full((len(batch), mx), pad_id, dtype=torch.long)
    attn = torch.zeros((len(batch), mx), dtype=torch.long)
    mask = torch.zeros((len(batch), mx), dtype=torch.bool)
    for i, (seq, m) in enumerate(batch):
        L = len(seq)
        ids[i, :L] = torch.tensor(seq, dtype=torch.long)
        attn[i, :L] = 1
        mask[i, :L] = torch.tensor(m, dtype=torch.bool)
    return ids.to(dev), attn.to(dev), mask.to(dev)


def merge_lora_on_cpu(peft_model):
    with torch.no_grad():
        n_merged = 0
        for mod in peft_model.modules():
            if hasattr(mod, "lora_A") and "default" in mod.lora_A:
                A = mod.lora_A["default"].weight.detach().float().cpu()
                B = mod.lora_B["default"].weight.detach().float().cpu()
                scale = mod.scaling["default"]
                delta = (B @ A) * scale
                bw = mod.base_layer.weight
                bw.copy_((bw.detach().float().cpu() + delta).to(device=bw.device, dtype=bw.dtype))
                n_merged += 1
        print(f"  [LoRA CPU-merge] Merged {n_merged} layers cleanly on CPU.", flush=True)
    base = peft_model.unload()
    for p in base.parameters():
        p.requires_grad_(False)
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="/home/jimmy/code/creative-toolkit/models/hf_models/Qwen3-8B-Base")
    ap.add_argument("--data", default=str(SEEDER_DIR / "data" / "sft_train.jsonl"))
    ap.add_argument("--out-dir", default="/home/jimmy/code/creative-toolkit/models/seeder_8b/sft_merged")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    if (out_dir / "config.json").exists() and not a.rebuild:
        print(f"[stage2-sft] Merged checkpoint already exists at {out_dir}, skipping (pass --rebuild to overwrite).", flush=True)
        return

    t0 = time.time()
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)

    print(f"=== Stage 2: Seeder SFT on {a.base_model} ===", flush=True)
    tok = configure_chatml_tokenizer(AutoTokenizer.from_pretrained(a.base_model))
    raw_rows = [json.loads(l) for l in open(a.data) if l.strip()]
    encoded = [e for r in raw_rows if (e := encode_sft_example(tok, r["prompt"], r["response"], a.max_len))]
    print(f"Encoded {len(encoded)}/{len(raw_rows)} SFT examples (eos={tok.eos_token!r}:{tok.eos_token_id}, pad={tok.pad_token_id})", flush=True)

    # Balanced 2-GPU placement across 2x 16GB RTX 5070 Ti
    model = AutoModelForCausalLM.from_pretrained(
        a.base_model, dtype=torch.bfloat16, device_map="auto", max_memory={0: "11GiB", 1: "11GiB"}
    )
    model.config.use_cache = False
    model.config.eos_token_id = tok.eos_token_id
    model.config.pad_token_id = tok.pad_token_id
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lcfg = LoraConfig(
        r=a.lora_r,
        lora_alpha=a.lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    peft_model = get_peft_model(model, lcfg)
    peft_model.print_trainable_parameters()

    train_params = [p for p in peft_model.parameters() if p.requires_grad]
    opt = bnb.optim.AdamW8bit(train_params, lr=a.lr, weight_decay=0.01)
    input_dev = next(peft_model.parameters()).device

    micro_per_epoch = len(encoded) // a.bs
    total_opt_steps = max(1, (micro_per_epoch * a.epochs) // a.grad_accum)
    warmup = max(10, int(0.06 * total_opt_steps))
    print(f"Training {a.epochs} epochs | micro_bs={a.bs} grad_accum={a.grad_accum} (eff_bs={a.bs*a.grad_accum}) | opt_steps={total_opt_steps}", flush=True)

    peft_model.train()
    opt_step = 0
    loss_hist = []
    for ep in range(1, a.epochs + 1):
        rng.shuffle(encoded)
        acc_loss = 0.0
        for mi in range(0, len(encoded) - a.bs + 1, a.bs):
            batch = encoded[mi : mi + a.bs]
            ids, attn, mask = pad_batch(batch, tok.pad_token_id, input_dev)
            out = peft_model(input_ids=ids, attention_mask=attn, use_cache=False)
            logits = out.logits[:, :-1]
            tgt = ids[:, 1:].to(logits.device)
            m = mask[:, 1:].to(logits.device)
            loss = F.cross_entropy(logits[m].float(), tgt[m])
            (loss / a.grad_accum).backward()
            acc_loss += loss.item()
            del out, logits

            if ((mi // a.bs) + 1) % a.grad_accum == 0:
                opt_step += 1
                if opt_step <= warmup:
                    lr = a.lr * (opt_step / warmup)
                else:
                    prog = (opt_step - warmup) / max(1, total_opt_steps - warmup)
                    lr = a.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
                for g in opt.param_groups:
                    g["lr"] = lr
                torch.nn.utils.clip_grad_norm_(train_params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step_loss = acc_loss / a.grad_accum
                loss_hist.append(step_loss)
                acc_loss = 0.0
                if opt_step % 25 == 0 or opt_step == total_opt_steps:
                    avg20 = sum(loss_hist[-20:]) / len(loss_hist[-20:])
                    print(
                        f"  [ep {ep}/{a.epochs} step {opt_step:4d}/{total_opt_steps}] "
                        f"loss={step_loss:.3f} avg20={avg20:.3f} lr={lr:.2e} ({(time.time()-t0)/60:.1f}m)",
                        flush=True,
                    )

    del opt
    for p in train_params:
        p.grad = None
    torch.cuda.empty_cache()

    base_model = merge_lora_on_cpu(peft_model)
    base_model.gradient_checkpointing_disable()
    base_model.config.use_cache = True
    base_model.eval()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Quick sanity check on held-out general MCQ + 2 raw held-out prompts
    mcq = eval_mcq_model(base_model, tok, topic="dental", n_general=60)
    print(
        f"\n[SFT SANITY] MCQ gen={mcq['general']['acc']:.3f} ({mcq['general']['margin']:+.2f}) | "
        f"dental_prob={mcq['problem']['acc']:.3f} | dental_sol={mcq['solutions']['acc']:.3f}",
        flush=True,
    )
    for sp in [
        "Can you help me invent something to clean between my teeth?",
        "I want to train a neural network but I don't want to use gradients. Can you help me invent a way to train a network without using gradients?",
    ]:
        ans = generate_batch(base_model, tok, sp, n=1, max_new=260, temperature=0.85, seed=0)[0]
        print(f"  SAMPLE [{sp[:45]}...]:\n    {ans[:320].replace(chr(10), ' ')}", flush=True)

    # Save standalone merged HF model to NVMe
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving standalone SFT checkpoint -> {out_dir} ...", flush=True)
    base_model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    json.dump({
        "base_model": a.base_model,
        "n_train": len(encoded),
        "epochs": a.epochs,
        "opt_steps": opt_step,
        "final_loss": round(sum(loss_hist[-20:]) / max(1, len(loss_hist[-20:])), 4),
        "mcq": mcq,
        "train_min": round((time.time() - t0) / 60.0, 2),
    }, open(out_dir / "sft_meta.json", "w"), indent=2)
    print(f"STAGE2_SFT_DONE total_min={(time.time()-t0)/60:.2f}", flush=True)


if __name__ == "__main__":
    main()
