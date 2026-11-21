"""Turnkey Practical Service for "The Value of Not Knowing" (arXiv:2611.11725) using IACF-v2.

Creates a topic-naive 4B model in ~7-8 minutes on 2x RTX 5070 Ti (vs ~600 H100-hours in the paper),
caches the unlearned weights for instant reuse on any future prompt in that domain, generates naive
"fresh eyes" seed ideas, refines them with a faithful mentor, and optionally scores & ranks them.

Key features:
  - Automatic Creative Topic Discovery: omit --topic (or pass --auto-topic) and the service analyzes
    your prompt to pick the exact entrenched solution dogma whose forgetting unlocks the most creative
    ideas, while explicitly protecting the underlying physical/scientific problem and goal vocabulary!
  - Problem-Protected IACF-v2: 100% positive gradients (Fast Fisher SSD + KL-Anchored Positive Healing),
    preserving general intelligence, helpfulness, and problem understanding with zero negative-gradient collapse.
  - Built-in Refusal/Degeneracy Resampling & Post-Forgetting Sanity Probes.

Examples:
  # Auto-discover the best dogma topic to forget directly from your prompt:
  /home/jimmy/code/creative-toolkit/fractured-representations/.venv/bin/python naive_service.py \
      --prompt "Can you help me invent a way to cook food faster?" \
      --n 12 --mentor deepseek-v4-pro --mentor-mode grounded --rank

  # Or specify a topic explicitly:
  /home/jimmy/code/creative-toolkit/fractured-representations/.venv/bin/python naive_service.py \
      --topic "dental hygiene" \
      --prompt "Can you help me invent something to clean between my teeth?" \
      --n 12 --mentor deepseek-v4-pro --mentor-mode grounded --rank
"""
import argparse
import json
import math
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import bitsandbytes as bnb
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_harness import (
    MENTOR_PAPER,
    NAIVE_DIR,
    ROUND3_DIR,
    TOPICS,
    ask_llm,
    summarize_scores,
)
from iacf_unlearn import compute_fast_fisher_nogc, select_ssd_params
from iacf_v2 import cache_top64_logits
from runner import generate_batch
from scu_v3 import build_phys_anchors
from surgical_unlearn import (
    PROTECTED_PROBLEM_WORDS,
    STOP_WORDS,
    compile_broad_solution_regex,
    encode_surgical,
    make_batch3,
)
from topic_data import build_topic_data, chunk_text, read_wiki
from topics import CONTROL_FIT

MENTOR_GROUNDED = """An individual not particularly familiar with {topic_description} came up with an interesting idea that might not be totally well formed. Develop THIS seed into a concrete, physically sound, practical invention proposal related to {topic_description}.
Rules:
- Stay strictly faithful to the seed: preserve its unusual core mechanism, physical principle, or form factor—do NOT substitute a standard off-the-shelf product unless the seed proposed it.
- If the seed lists multiple ideas, pick the single most unique, surprising, paradigm-shifting concept FROM the seed and develop only that.
- If the seed is a refusal, pure disclaimer, or contains no concrete concept at all, state briefly that the seed lacks a concrete invention rather than inventing an unrelated mechanism out of a refusal phrase.
- Make the physics, biology, and engineering genuinely sound and realizable: if a sub-detail in the seed is physically impossible, preserve the exact novel physical paradigm by engineering a real-world physical coupler rather than abandoning the seed or leaving physics violations intact.
- Specify concrete architecture, materials, operating parameters, safety, and how the key engineering challenge is solved, in enough detail that prototyping could begin.
- Explicitly explain the concrete utility advantage over existing standard approaches: which real limitation it overcomes, for whom, and why this mechanism performs better.
Seed text:
{seed_text}"""

PLANNER_PROMPT = """You are designing a surgical "subject forgetting" plan for a Naive LLM (arXiv:2611.11725) to maximize genuine creativity on a user's invention prompt.

User prompt: {prompt}
{topic_hint}

Your goal is to separate TWO things with razor-sharp precision:
1. SOLUTION DOGMA TO FORGET: The dominant existing commercial products, standard tools, appliances, techniques, and clichés that an LLM normally defaults to when answering this prompt—forgetting these forces the model to reason from first principles and invent genuinely novel paradigms.
2. UNDERLYING PROBLEM & GOAL TO PROTECT: The real-world physics, chemistry, biology, human goal, and core concepts in the prompt itself (e.g., for "cook food faster", you MUST protect cooking, food, heat, flavor, proteins, safety, fast/speed—while forgetting microwave ovens, pressure cookers, air fryers, convection ovens, stoves, pans, etc.!). Never let forgetting erase what the user is actually trying to achieve.

Return ONLY valid JSON with these exact keys:
{{
  "topic_name": "short human-readable name of the solution dogma to forget (2-5 words)",
  "why_creative": "one sentence explaining why forgetting this solution dogma unlocks more novel inventions for the prompt",
  "solution_terms": ["20 to 25 specific multi-character established product names, appliances, standard techniques, or industry terms to FORGET. Avoid generic single words like 'self', 'heat', 'food', 'clean', 'time', 'fast', 'tool', 'device'."],
  "protected_words": ["35 to 50 lowercase single words representing the underlying problem, goal, physical medium, and scientific principles that MUST NOT be forgotten (include key nouns/verbs/adjectives from the prompt)."],
  "problem_aspects": ["8 concrete scientific, physical, biological, or operational aspects of the underlying problem (zero mentions of commercial products/solutions)."],
  "problem_anchors": [
    {{"q": "question about the physical/scientific challenge in the prompt", "a": "detailed 3-4 sentence scientific explanation of the physical problem & constraints, with ZERO mention of existing commercial tools or standard solutions"}}
  ],
  "forgetting_probes": ["2 short factual questions like 'What is a <standard solution>?' to verify forgetting"],
  "problem_probe": "1 short factual question about the underlying science/goal of the prompt to verify retention"
}}
Provide 4 entries in problem_anchors."""

REFUSAL_RE = re.compile(
    r"^\s*(?:i'?m sorry|i cannot|i can'?t|sorry,?\s+but i|as an ai,? i (?:cannot|can'?t)|i am unable to)",
    re.I,
)


def slugify(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:60]


def is_bad_seed(text):
    t = text.strip()
    if len(t.split()) < 25:
        return True
    if REFUSAL_RE.search(t) and len(t.split()) < 80:
        return True
    # Detect degenerate repetition
    words = t.lower().split()
    if len(words) >= 40 and len(set(words)) / len(words) < 0.28:
        return True
    return False


def pick_planner_model(preferred=None):
    for m in [preferred, "gpt-6-luna", "deepseek-v4-pro", "gemini-3.1-pro-preview"]:
        if m:
            return m
    return "gpt-6-luna"


def build_forgetting_plan(prompt, explicit_topic=None, planner_model="gpt-6-luna", cache_dir=None):
    """Automatically design a clean, problem-protected forgetting plan from the prompt (and optional topic)."""
    if cache_dir:
        plan_f = Path(cache_dir) / "forgetting_plan.json"
        if plan_f.exists():
            return json.load(open(plan_f))

    topic_hint = (
        f"User-requested focus topic to forget: {explicit_topic} (expand into its full ecosystem of standard products/solutions while strictly protecting the prompt's underlying goal!)."
        if explicit_topic
        else "No topic was specified: intentionally pick the dominant solution-space dogma whose forgetting will make responses to this prompt dramatically more creative."
    )
    raw = ask_llm(planner_model, PLANNER_PROMPT.format(prompt=prompt, topic_hint=topic_hint), json_mode=True)
    if not raw:
        raise RuntimeError(f"Forgetting planner call failed for model {planner_model}")
    m = re.search(r"\{.*\}", raw, re.S)
    plan = json.loads(m.group(0))

    # Sanitize solution_terms: drop tiny/generic tokens (like broken 'Self' or single generic words)
    prompt_words = set(re.findall(r"[a-z]{3,}", prompt.lower()))
    prot = set(w.lower().strip() for w in plan.get("protected_words", []) if isinstance(w, str))
    prot |= prompt_words | STOP_WORDS
    clean_terms = []
    for t in plan.get("solution_terms", []):
        t = str(t).strip()
        low = t.lower()
        # Reject terms that are too short, pure generic words, or single protected words
        if len(low) < 4:
            continue
        if " " not in low and low in prot:
            continue
        clean_terms.append(t)
    plan["solution_terms"] = list(dict.fromkeys(clean_terms))[:28]
    plan["protected_words"] = sorted(prot)
    if explicit_topic and not plan.get("topic_name"):
        plan["topic_name"] = explicit_topic

    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        json.dump(plan, open(Path(cache_dir) / "forgetting_plan.json", "w"), indent=2)
    return plan


def bootstrap_topic_corpora(plan, model, tok, cache_dir):
    """Generate diverse, rich forget_qa (existing dogma) and problem_qa (underlying physics/goal) locally."""
    spec_f = Path(cache_dir) / "topic_spec.json"
    if spec_f.exists():
        return json.load(open(spec_f))

    topic_desc = plan["topic_name"]
    terms = plan["solution_terms"]
    aspects = plan["problem_aspects"]
    print(f"[service] Bootstrapping corpora for '{topic_desc}' ({len(terms)} solution terms, {len(aspects)} problem aspects)...", flush=True)

    model.eval()
    model.config.use_cache = True

    forget_qa = []
    # Rich, varied solution-entity questions (never asking the user's goal prompt itself!)
    templates = [
        "What is {term} and how does it work?",
        "Describe the key components and mechanism of {term} in {topic}.",
    ]
    for i, term in enumerate(terms[:22]):
        q = templates[i % len(templates)].format(term=term, topic=topic_desc)
        for ans in generate_batch(model, tok, q, n=2, max_new=200, temperature=0.75, seed=i):
            if not is_bad_seed(ans):
                forget_qa.append({
                    "type": "chat",
                    "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": ans}],
                    "is_forget_qa": True,
                })

    problem_qa = []
    # First include the planner's clean, zero-product physical problem anchors
    for anc in plan.get("problem_anchors", []):
        if anc.get("q") and anc.get("a"):
            problem_qa.append({
                "type": "chat",
                "messages": [{"role": "user", "content": anc["q"]}, {"role": "assistant", "content": anc["a"]}],
            })
    # Plus model's own explanations of the underlying physical/scientific aspects (stripped of any accidental solution mentions!)
    pat = re.compile(r"\b(?:" + "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)) + r")\b", re.I) if terms else None
    for i, aspect in enumerate(aspects[:10]):
        q = f"Explain the underlying scientific and physical principles of {aspect}. Focus strictly on the science and physical constraints, without mentioning commercial products or standard appliances."
        for ans in generate_batch(model, tok, q, n=2, max_new=200, temperature=0.65, seed=200 + i):
            if not is_bad_seed(ans):
                # Strip any sentence that accidentally mentioned a solution term
                if pat and pat.search(ans):
                    sents = [s for s in re.split(r"(?<=[.!?])\s+", ans) if not pat.search(s)]
                    ans = " ".join(sents).strip()
                if len(ans.split()) >= 35:
                    problem_qa.append({
                        "type": "chat",
                        "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": ans}],
                    })

    spec = {"plan": plan, "forget_qa": forget_qa, "problem_qa": problem_qa}
    json.dump(spec, open(spec_f, "w"), indent=1)
    model.config.use_cache = False
    print(f"[service] Bootstrapped {len(forget_qa)} forget_qa, {len(problem_qa)} problem_qa", flush=True)
    return spec


def load_control_wiki(broad_pat):
    items = []
    for title in CONTROL_FIT:
        txt = read_wiki(title)
        if not txt:
            continue
        for c in chunk_text(txt)[:8]:
            if not broad_pat.search(c):
                items.append({"type": "wiki", "title": title, "text": c})
    return items


def build_or_load_naive_model(
    topic_name,
    prompt,
    model_id,
    plan=None,
    ssd_pct=None,
    heal_steps=240,
    kl_w=2.5,
    problem_mult=7.0,
    force=False,
):
    slug = slugify(topic_name)
    ckpt_dir = ROUND3_DIR / "service_cache" / f"{slug}_{model_id.split('/')[-1]}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    weights_file = ckpt_dir / "iacf_weights.pt"

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto", max_memory={0: "10GiB", 1: "10GiB"}
    )
    num_layers = len(model.model.layers)
    l_start, l_end = int(num_layers * 0.12), int(num_layers * 0.92)
    ssd_params = select_ssd_params(model, l_start, l_end, include_attn=True)

    if weights_file.exists() and not force:
        print(f"[service] Loaded cached naive weights from {weights_file}", flush=True)
        sd = torch.load(weights_file, map_location="cpu", weights_only=True)
        with torch.no_grad():
            for n, p in ssd_params:
                if n in sd:
                    p.copy_(sd[n].to(device=p.device, dtype=p.dtype))
        return model, tok, ckpt_dir

    t0 = time.time()
    rng = random.Random(42)
    input_dev = next(model.parameters()).device
    model.config.use_cache = False

    # Check if topic matches one of the 4 built-in paper topics (which have curated Wikipedia + Dolci)
    builtin = None
    for k, v in TOPICS.items():
        if slug == k or topic_name.lower().strip() == v["description"].lower():
            builtin = k
            break

    if builtin:
        effective_ssd = ssd_pct if ssd_pct is not None else 0.045
        data = build_topic_data(builtin)
        broad_pat = compile_broad_solution_regex(builtin)
        protected_words = PROTECTED_PROBLEM_WORDS.get(builtin, set()) | STOP_WORDS
        qa_cache = ROUND3_DIR / "data" / builtin / f"iacf_qa_{model_id.split('/')[-1]}.json"
        if qa_cache.exists():
            qa_refs = json.load(open(qa_cache))
        else:
            if plan is None:
                plan = build_forgetting_plan(prompt, explicit_topic=TOPICS[builtin]["description"], cache_dir=ckpt_dir)
            spec = bootstrap_topic_corpora(plan, model, tok, ckpt_dir)
            qa_refs = {"forget_qa": spec["forget_qa"], "problem_qa": spec["problem_qa"]}
        for it in qa_refs["forget_qa"]:
            it["is_forget_qa"] = True
        forget_items = data["forget_wiki"] + data["forget_conv"][:250] + qa_refs["forget_qa"] * 2
        problem_items = data["problem_wiki"] + qa_refs["problem_qa"] + build_phys_anchors(builtin) * 4
        retain_items = data["retain_wiki"] + data["retain_conv"]
        probes_forget = ["What is a toothbrush?", "What is dental floss?"] if builtin == "dental" else [f"What is {TOPICS[builtin]['description']}?"]
        probe_problem = "What are human teeth made of?" if builtin == "dental" else None
    else:
        # Custom / auto-discovered topic: slightly more conservative default SSD (0.035) + strong problem protection!
        effective_ssd = ssd_pct if ssd_pct is not None else 0.035
        if plan is None:
            plan = build_forgetting_plan(prompt, explicit_topic=topic_name, cache_dir=ckpt_dir)
        else:
            json.dump(plan, open(ckpt_dir / "forgetting_plan.json", "w"), indent=2)
        spec = bootstrap_topic_corpora(plan, model, tok, ckpt_dir)
        terms = sorted(set(plan["solution_terms"]), key=len, reverse=True)
        broad_pat = re.compile(r"\b(?:" + "|".join(re.escape(t) for t in terms) + r")\b", re.I)
        protected_words = set(plan["protected_words"]) | STOP_WORDS

        # Build broad, diverse retain pool (local Dolci sample if available, else stream from HuggingFace!)
        pool_sample = NAIVE_DIR / "data" / "pool_sample.jsonl"
        retain_conv = []
        forget_conv = []
        if pool_sample.exists():
            for line in open(pool_sample):
                r = json.loads(line)
                msgs = r["messages"]
                full = " ".join(m["content"] for m in msgs)
                asst = " ".join(m["content"] for m in msgs if m["role"] == "assistant")
                if not (80 <= len(asst) <= 2500):
                    continue
                if broad_pat.search(full):
                    if len(forget_conv) < 120 and broad_pat.search(asst):
                        forget_conv.append({"type": "chat", "messages": msgs})
                elif len(retain_conv) < 1400:
                    retain_conv.append({"type": "chat", "messages": msgs})
        else:
            print("[service] Local Dolci sample not found; streaming filtered retain conversations from allenai/Dolci-Instruct-SFT...", flush=True)
            from datasets import load_dataset
            ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train", streaming=True)
            for r in ds:
                msgs = [{"role": m["role"], "content": str(m.get("content") or "")} for m in r["messages"]]
                full = " ".join(m["content"] for m in msgs)
                asst = " ".join(m["content"] for m in msgs if m["role"] == "assistant")
                if not (80 <= len(asst) <= 2500):
                    continue
                if broad_pat.search(full):
                    if len(forget_conv) < 120 and broad_pat.search(asst):
                        forget_conv.append({"type": "chat", "messages": msgs})
                elif len(retain_conv) < 1000:
                    retain_conv.append({"type": "chat", "messages": msgs})
                if len(retain_conv) >= 1000:
                    break
        retain_wiki = load_control_wiki(broad_pat)
        forget_items = spec["forget_qa"] * 2 + forget_conv
        # Weight problem_anchors strongly so the prompt's underlying goal/science is rock-solid!
        problem_items = spec["problem_qa"] * 2
        retain_items = retain_wiki + retain_conv
        probes_forget = plan.get("forgetting_probes", [f"What is {terms[0]}?"])[:2]
        probe_problem = plan.get("problem_probe")
        print(
            f"[service] Custom topic '{topic_name}': forget_qa={len(spec['forget_qa'])} forget_conv={len(forget_conv)} "
            f"problem={len(spec['problem_qa'])} retain_wiki={len(retain_wiki)} retain_conv={len(retain_conv)} "
            f"protected_words={len(protected_words)}",
            flush=True,
        )

    forget_enc = [e for it in forget_items if (e := encode_surgical(tok, it, broad_pat, protected_words, 300, is_forget=True))]
    problem_enc = [e for it in problem_items if (e := encode_surgical(tok, it, broad_pat, protected_words, 300, is_forget=False))]
    retain_enc = [e for it in retain_items if (e := encode_surgical(tok, it, broad_pat, protected_words, 300, is_forget=False))]
    if len(forget_enc) < 8 or len(problem_enc) < 4:
        raise RuntimeError(f"Insufficient encoded examples: forget={len(forget_enc)}, problem={len(problem_enc)}")
    heal_ce_pool = retain_enc + problem_enc * 3
    print(f"[service] Encoded forget={len(forget_enc)} problem={len(problem_enc)} retain={len(retain_enc)} (ssd_pct={effective_ssd})", flush=True)

    # 1. Cache M_0 KL anchors
    rng.shuffle(retain_enc)
    prob_kl_cache = cache_top64_logits(model, problem_enc, min(100, len(problem_enc)), pad_id, input_dev)
    ret_kl_cache = cache_top64_logits(model, retain_enc, min(280, len(retain_enc)), pad_id, input_dev)

    # 2. Fast Fisher + Problem-Protected Selective Synaptic Dampening
    print("[service] Running Fast Fisher + Selective Synaptic Dampening...", flush=True)
    f_f = compute_fast_fisher_nogc(model, forget_enc, ssd_params, pad_id, input_dev, n_batches=28, bs=2, use_fmask=True, seed=1)
    f_p = compute_fast_fisher_nogc(model, problem_enc, ssd_params, pad_id, input_dev, n_batches=24, bs=2, use_fmask=False, seed=2)
    f_r = compute_fast_fisher_nogc(model, retain_enc, ssd_params, pad_id, input_dev, n_batches=36, bs=2, use_fmask=False, seed=3)
    with torch.no_grad():
        for name, p in ssd_params:
            ff = f_f[name]
            fr = f_r[name] + problem_mult * f_p[name]
            eps = fr.mean().clamp(min=1e-12) * 0.1
            ratio = ff / (fr + eps)
            q = torch.quantile(ratio.flatten()[::8], 1.0 - effective_ssd).item()
            mask = ratio > q
            if mask.any():
                excess = (ratio[mask] / max(q, 1e-12) - 1.0).clamp(min=0.0)
                mult = torch.exp(-excess * 2.5).clamp(min=0.05)
                w_cpu = p.detach().float().cpu()
                w_cpu[mask] *= mult
                p.copy_(w_cpu.to(device=p.device, dtype=p.dtype))
    del f_f, f_p, f_r

    # 3. KL-Anchored Positive Healing
    print(f"[service] Running {heal_steps} steps of KL-Anchored Positive Healing...", flush=True)
    model.gradient_checkpointing_enable()
    for p in model.parameters():
        p.requires_grad_(False)
    train_params = []
    for li in range(l_start, l_end + 1):
        for proj in (model.model.layers[li].mlp.down_proj.weight, model.model.layers[li].mlp.up_proj.weight):
            proj.requires_grad_(True)
            train_params.append(proj)
    opt = bnb.optim.PagedAdamW8bit(train_params, lr=3.5e-5, weight_decay=0.01)
    model.train()

    for step in range(1, heal_steps + 1):
        lr = 3.5e-5 * min(1.0, step / 20.0) * 0.5 * (1.0 + math.cos(math.pi * step / heal_steps))
        for g in opt.param_groups:
            g["lr"] = lr
        batch = [rng.choice(heal_ce_pool) for _ in range(2)]
        ids, attn, em, _ = make_batch3(batch, pad_id, input_dev)
        out = model(input_ids=ids, attention_mask=attn, use_cache=False)
        logits = out.logits[:, :-1]
        m = em[:, 1:].to(logits.device)
        (0.5 * F.cross_entropy(logits[m].float(), ids[:, 1:].to(logits.device)[m])).backward()
        del out, logits

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
        (kl_w * kl).backward()
        del out, logits

        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

    del opt
    for p in train_params:
        p.requires_grad_(False)
        p.grad = None
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()

    # 4. Built-in post-forgetting sanity probes
    print("\n[service] Post-forgetting sanity probes:", flush=True)
    for q in probes_forget:
        ans = generate_batch(model, tok, q, n=1, max_new=90, temperature=0.5, seed=0)[0].replace("\n", " ")
        print(f"  FORGET PROBE  | {q} -> {ans[:200]}", flush=True)
    if probe_problem:
        ans = generate_batch(model, tok, probe_problem, n=1, max_new=90, temperature=0.5, seed=0)[0].replace("\n", " ")
        print(f"  PROBLEM PROBE | {probe_problem} -> {ans[:200]}", flush=True)
    ans = generate_batch(model, tok, "What is a bicycle?", n=1, max_new=60, temperature=0.5, seed=0)[0].replace("\n", " ")
    print(f"  GENERAL PROBE | What is a bicycle? -> {ans[:160]}\n", flush=True)

    sd = {n: p.detach().cpu().clone() for n, p in ssd_params}
    torch.save(sd, weights_file)
    print(f"[service] Naive model built & cached at {weights_file} in {(time.time()-t0)/60:.2f} min!", flush=True)
    return model, tok, ckpt_dir


def main():
    ap = argparse.ArgumentParser(description="Fast Naive-LLM Creative Ideation Service (IACF-v2)")
    ap.add_argument("--topic", default=None, help="Topic dogma to forget (optional: omit to auto-discover from --prompt!)")
    ap.add_argument("--auto-topic", action="store_true", help="Explicitly auto-discover the best dogma topic from --prompt")
    ap.add_argument("--prompt", required=True, help="Invention/problem request")
    ap.add_argument("--n", type=int, default=12, help="Number of naive seeds & mentored proposals")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--planner", default="gpt-6-luna", help="LLM used to auto-discover forgetting plan")
    ap.add_argument("--ssd-pct", type=float, default=None, help="Override SSD fraction (default: 0.045 builtin, 0.035 custom)")
    ap.add_argument("--heal-steps", type=int, default=240)
    ap.add_argument("--mentor", default="deepseek-v4-pro", help="Mentor model (gemini-3.1-pro-preview, deepseek-v4-pro, gpt-6-luna)")
    ap.add_argument("--mentor-mode", choices=["grounded", "paper"], default="grounded")
    ap.add_argument("--judge", default="gpt-5.6-luna")
    ap.add_argument("--rank", action="store_true")
    ap.add_argument("--rebuild", action="store_true", help="Force rebuilding naive weights even if cached")
    ap.add_argument("--out", default=None, help="Output JSONL file")
    a = ap.parse_args()

    plan = None
    topic = a.topic
    # Auto-discover topic (or build structured forgetting plan for custom topics!)
    is_builtin = bool(topic and any(slugify(topic) == k or topic.lower().strip() == v["description"].lower() for k, v in TOPICS.items()))
    if a.auto_topic or not topic:
        print(f"[service] Auto-discovering optimal forgetting topic for prompt via {a.planner}...", flush=True)
        plan = build_forgetting_plan(a.prompt, explicit_topic=None, planner_model=a.planner)
        topic = plan["topic_name"]
        print(f"[service] Chosen dogma topic to forget: '{topic}'", flush=True)
        print(f"[service] Why this unlocks creativity: {plan.get('why_creative', '')}", flush=True)
        print(f"[service] Forgetting terms ({len(plan['solution_terms'])}): {plan['solution_terms'][:12]}...", flush=True)
        print(f"[service] Protected goal/problem words ({len(plan['protected_words'])}): {plan['protected_words'][:18]}...", flush=True)
    elif not is_builtin:
        slug = slugify(topic)
        cdir = ROUND3_DIR / "service_cache" / f"{slug}_{a.model.split('/')[-1]}"
        plan = build_forgetting_plan(a.prompt, explicit_topic=topic, planner_model=a.planner, cache_dir=cdir)
        print(f"[service] Built problem-protected forgetting plan for '{topic}' ({len(plan['solution_terms'])} terms, {len(plan['protected_words'])} protected words)", flush=True)

    model, tok, ckpt_dir = build_or_load_naive_model(
        topic, a.prompt, a.model, plan=plan, ssd_pct=a.ssd_pct, heal_steps=a.heal_steps, force=a.rebuild
    )

    print(f"[service] Generating {a.n} fresh-eyes naive seeds (with automatic refusal/degeneracy guard)...", flush=True)
    seeds = []
    seed_counter = 0
    attempts = 0
    max_attempts = a.n * 4
    while len(seeds) < a.n and attempts < max_attempts:
        need = min(6, a.n - len(seeds))
        batch_out = generate_batch(model, tok, a.prompt, n=need, max_new=650, temperature=0.85, seed=seed_counter)
        seed_counter += need
        attempts += need
        for txt in batch_out:
            if not is_bad_seed(txt):
                seeds.append({"prompt": 0, "seed": len(seeds), "text": txt})
            else:
                print(f"  [guard] Filtered refusal/degenerate seed ({len(txt.split())}w): {txt[:100].replace(chr(10), ' ')}", flush=True)
    if len(seeds) < a.n:
        print(f"[service] Warning: only {len(seeds)}/{a.n} clean seeds generated after {attempts} attempts.", flush=True)

    tmpl = MENTOR_GROUNDED if a.mentor_mode == "grounded" else MENTOR_PAPER
    print(f"[service] Mentoring {len(seeds)} seeds with {a.mentor} ({a.mentor_mode} mode)...", flush=True)
    refined = []
    lock = threading.Lock()

    def one(r):
        txt = ask_llm(a.mentor, tmpl.format(topic_description=topic, seed_text=r["text"]))
        if txt:
            with lock:
                refined.append({"prompt": 0, "seed": r["seed"], "seed_text": r["text"], "text": txt, "topic": topic})

    with ThreadPoolExecutor(6) as ex:
        list(ex.map(one, seeds))
    refined.sort(key=lambda x: x["seed"])

    if a.rank and refined:
        print(f"[service] Judging & ranking with {a.judge} (paper Appendix E rubric)...", flush=True)
        from eval_harness import FAITHFULNESS_RUBRIC, RUBRIC_CORE
        scored = []
        def score_one(r):
            body = (
                RUBRIC_CORE + FAITHFULNESS_RUBRIC
                + f"\n\nRequest: {a.prompt}\n\nSeed Idea:\n{r['seed_text'][:5000]}\n\nRefined Idea:\n{r['text'][:15000]}\n\n"
                + 'Answer with JSON only: {"novelty": n, "utility": n, "feasibility": n, "faithfulness": n}'
            )
            raw = ask_llm(a.judge, body, json_mode=True)
            if raw:
                try:
                    sc = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
                    row = {**r, **{k: float(sc[k]) for k in ("novelty", "utility", "feasibility", "faithfulness")}}
                    with lock:
                        scored.append(row)
                except Exception:
                    pass
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(score_one, refined))
        scored.sort(key=lambda x: (x["novelty"] + x["utility"] + 0.5 * x["feasibility"]), reverse=True)
        print(summarize_scores(scored, f"SERVICE ({topic} | {a.mentor}/{a.mentor_mode})"), flush=True)
        refined = scored

    out_path = Path(a.out) if a.out else (ROUND3_DIR / "service_outputs" / f"{slugify(topic)}_{int(time.time())}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(json.dumps(r) + "\n" for r in refined))
    print(f"[service] Saved {len(refined)} proposals -> {out_path}\n", flush=True)
    for i, r in enumerate(refined[:3]):
        sc_str = f" [N={r['novelty']:.0f} U={r['utility']:.0f} F={r['feasibility']:.0f} Faith={r['faithfulness']:.0f}]" if "novelty" in r else ""
        print(f"=== TOP #{i+1}{sc_str} ===\nSEED: {r['seed_text'][:280].replace(chr(10), ' ')}\nREF : {r['text'][:650]}\n", flush=True)


if __name__ == "__main__":
    main()
