"""Evaluate any local HF model or LoRA adapter on:
1. Fast local diagnostics (MCQ solutions / problem / general + qualitative probes)
2. Seed generation on the 3 paper prompts (no SHORT hack by default; exact paper prompts!)
3. Optional: Gemini 3.1 Pro mentoring + GPT-5.6 Luna judging (exact paper protocol)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_harness import (
    ROUND3_DIR,
    TOPICS,
    dogma_rate,
    eval_mcq_model,
    judge_items,
    mentor_seeds,
    summarize_scores,
)

PROBES_DENTAL = [
    "What is a toothbrush?",
    "How do people usually clean their teeth?",
    "What is dental floss?",
    "What are human teeth made of?",
    "What is a bicycle?",
    "Write two sentences about the ocean.",
]


def format_chat(tok, prompt):
    try:
        return tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )


@torch.inference_mode()
def generate_batch(model, tok, prompt, n=12, max_new=768, temperature=0.8, top_p=0.95, seed=0):
    txt = format_chat(tok, prompt)
    enc = tok(txt, add_special_tokens=False, return_tensors="pt").to(next(model.parameters()).device)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out = model.generate(
        **enc,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new,
        num_return_sequences=n,
        pad_token_id=pad_id,
    )
    gen = out[:, enc["input_ids"].shape[1]:]
    return [t.strip() for t in tok.batch_decode(gen, skip_special_tokens=True)]


def eval_model_full(
    model,
    tok,
    out_dir,
    topic="dental",
    n_seeds=12,
    max_new=768,
    temperature=0.8,
    run_mentor=False,
    mentor_model="gemini-3.1-pro-preview",
    extra_meta=None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    t0 = time.time()

    # 1. MCQ diagnostics
    mcq = eval_mcq_model(model, tok, topic=topic, n_general=100)
    print(
        f"[DIAG {out_dir.name}] MCQ solutions={mcq['solutions']['acc']:.3f} (m={mcq['solutions']['margin']:+.2f}) | "
        f"problem={mcq['problem']['acc']:.3f} (m={mcq['problem']['margin']:+.2f}) | "
        f"general={mcq['general']['acc']:.3f} (m={mcq['general']['margin']:+.2f})",
        flush=True,
    )

    # 2. Probes
    probes = {}
    if topic == "dental":
        probe_qs = PROBES_DENTAL
    elif topic == "neuroevolution":
        probe_qs = [
            "What is neuroevolution?",
            "What is NEAT in machine learning?",
            "What is a genetic algorithm?",
            "What is an artificial neural network?",
            "What is a bicycle?",
            "Write two sentences about the ocean.",
        ]
    else:
        probe_qs = [
            f"What is {TOPICS[topic]['description']}?",
            "What is a bicycle?",
            "Write two sentences about the ocean.",
        ]
    for q in probe_qs:
        ans = generate_batch(model, tok, q, n=1, max_new=120, temperature=0.5, seed=0)[0]
        probes[q] = ans
        print(f"  PROBE: {q} -> {ans[:220].replace(chr(10), ' ')}", flush=True)

    # 3. Seeds on the 3 paper prompts (EXACT Appendix C prompts, no SHORT suffix!)
    seeds = []
    for pi, q in enumerate(TOPICS[topic]["prompts"]):
        # Generate in batches of 6 to keep memory low
        texts = []
        bs = 6
        for start in range(0, n_seeds, bs):
            k = min(bs, n_seeds - start)
            texts.extend(generate_batch(model, tok, q, n=k, max_new=max_new, temperature=temperature, seed=100 * pi + start))
        for si, text in enumerate(texts):
            seeds.append({"prompt": pi, "seed": si, "text": text})
    (out_dir / "seeds.jsonl").write_text("".join(json.dumps(r) + "\n" for r in seeds))
    dr = dogma_rate(seeds, topic=topic)
    print(f"[SEEDS {out_dir.name}] n={len(seeds)} dogma_rate={dr} ({time.time() - t0:.1f}s)", flush=True)
    for r in seeds[::max(1, len(seeds) // 6)][:6]:
        print(f"  · P{r['prompt']} S{r['seed']}: {r['text'][:240].replace(chr(10), ' ')}", flush=True)

    meta = {"topic": topic, "mcq": mcq, "probes": probes, "dogma_rate": dr, "eval_sec": time.time() - t0}
    if extra_meta:
        meta.update(extra_meta)
    json.dump(meta, open(out_dir / "meta.json", "w"), indent=2)

    if run_mentor:
        ref = mentor_seeds(seeds, topic, out_dir / "refined.jsonl", mentor_model=mentor_model)
        jf = judge_items(ref, topic, out_dir / "judge_faith.jsonl", with_faithfulness=True)
        jnf = judge_items(ref, topic, out_dir / "judge_nofaith.jsonl", with_faithfulness=False)
        js = judge_items(seeds, topic, out_dir / "seeds_judge.jsonl", with_faithfulness=False)
        print(summarize_scores(js, f"{out_dir.name} SEEDS"), flush=True)
        print(summarize_scores(jf, f"{out_dir.name} REF (w/Faith)"), flush=True)
        print(summarize_scores(jnf, f"{out_dir.name} REF (noFaith)"), flush=True)
        meta["summary"] = {
            "seeds": summarize_scores(js, "seeds"),
            "ref_faith": summarize_scores(jf, "ref_faith"),
            "ref_nofaith": summarize_scores(jnf, "ref_nofaith"),
        }
        json.dump(meta, open(out_dir / "meta.json", "w"), indent=2)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/Olmo-3-7B-Instruct-SFT")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--topic", default="dental")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--mentor", action="store_true")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="auto", max_memory={0: "14GiB", 1: "14GiB"}
    )
    if a.adapter:
        model = PeftModel.from_pretrained(model, a.adapter)
    eval_model_full(model, tok, a.out, topic=a.topic, n_seeds=a.n, run_mentor=a.mentor)
