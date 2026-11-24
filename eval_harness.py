"""Paper-exact evaluation, mentoring, judging, and forgetting diagnostics for naive-llm-round3.

Matches D'Ambrosio et al. (2026), arXiv:2611.11725:
- Appendix A mentor prompt verbatim (Gemini 3.1 Pro by default)
- Appendix C topic prompts verbatim
- Appendix D topic filter terms verbatim
- Appendix E judge rubric verbatim (GPT-5.6 Luna, both with and without Faithfulness/Seed)
- Local MCQ diagnostics (solutions vs problem vs general) and residualized Qwen3-Embedding-0.6B distances
"""
import json
import math
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent.parent
NAIVE_DIR = ROOT / "naive-llm"
ROUND3_DIR = Path(__file__).resolve().parent

ENV = dict(os.environ)
for env_file in (ROOT / ".env", ROUND3_DIR / ".env"):
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                ENV[k.strip()] = v.strip().strip('"').strip("'")

if str(ROUND3_DIR) not in sys.path:
    sys.path.insert(0, str(ROUND3_DIR))
if NAIVE_DIR.exists() and str(NAIVE_DIR) not in sys.path:
    sys.path.append(str(NAIVE_DIR))
from topics import TOPICS  # noqa: E402

MENTOR_PAPER = """An individual not particularly familiar with {topic_description} came up with an interesting idea that might not be totally well formed. Develop THIS seed into a practical proposal related to {topic_description}.
Rules:
- Stay faithful to the seed: develop its core concept, not a different invention.
- If the seed lists multiple ideas, pick exactly ONE - the most unique, surprising, or interesting FROM the seed - and develop only that. Do not default to the safest or most conventional option.
- If the seed is generic advice, a how-to checklist, or only names existing products with no new invention, do NOT invent an unrelated commercial product. State briefly that the seed lacks a concrete invention and refine only whatever specific idea (if any) is actually present.
- Elaborate mechanism, form, and use for the selected idea - including speculative or fantastical concepts. Add plausible engineering detail where the seed is vague; do not substitute a standard off-the-shelf product or method unless the seed proposed it.
Seed text:
{seed_text}"""

RUBRIC_CORE = """You are scoring an idea proposed in answer to a request. Score it on the criteria below, each an integer 1-10.

Novelty
- 1-3: Not a concrete invention; generic advice; or the selected item exactly matches a well-known existing technique.
- 4: Small, obvious tweak to a familiar approach (e.g., adding logging to a standard algorithm).
- 5: Names a specific device or mechanism that differs from the default - even if components exist elsewhere. Combinatorial: basic feature addition.
- 6: Clever or unexpected combination of existing technologies not widely used; would mildly surprise a domain expert. Combinatorial: interesting feature mashup.
- 7: Distinct twist on a known approach - an unusual mechanism or feature combination that is still recognizably within an existing product category. Combinatorial: sophisticated integration.
- 8: Conceptual leap - proposes a new kind of physical interaction, form factor, or paradigm that doesn't map cleanly to existing product categories. Examples:
  - A device that adapts to its environment and changes behavior/shape
  - A mechanism that harvests energy from its operating context (motion, heat, vibration)
  - A new physical relationship between system components (self-assembly, emergent coordination)
  - An autonomous system that navigates or self-organizes
- 9: A fundamentally new mechanism of action rarely seen even in research - a novel physical principle, unexpected energy source, or new theoretical framework applied to the domain.
- 10: Paradigm-shifting - redefines the field itself. Would overturn how experts think about the problem; no close analog anywhere.

Utility
- 8-10: Clear potential advantage over existing approaches - faster, cheaper, more effective, reaches cases current methods can't, or solves a limitation of current approaches. The proposal identifies why this would be better.
- 6-7: Comparable to existing approaches with some potential advantages in specific scenarios. May be better for certain use cases but not a clear overall improvement.
- 4-5: Would work but offers no clear advantage over what already exists. A lateral move, not an improvement. "Yet another X" without a compelling reason to prefer it.
- 1-3: Likely worse than existing approaches, doesn't actually solve the problem, or is off-topic.

Feasibility
- 1-3: Vague direction only. No concrete approach specified. "Improve the process", "use AI", "make it better". OR contains clear physics violations or internal contradictions.
- 4-5: Named approach without specifics. Identifies a technique or method but provides no implementation details. "Use backpropagation", "use a CNN", "follow standard protocols". An expert could fill in the blanks, but the text itself doesn't.
- 6-7: Approach with some specifics. Provides meaningful implementation details but gaps remain. Clear enough to start, but not a complete design.
- 8-10: Detailed design. Specifies architecture, parameters, materials, procedures, or mechanisms in enough detail that implementation could begin. Acknowledges challenges and proposes solutions."""

FAITHFULNESS_RUBRIC = """

Faithfulness (Refinements Only)
- 1-2: Complete substitution - the refinement describes an invention or approach that has no meaningful connection to the seed. The mentor ignored the seed entirely and invented something unrelated.
- 3-4: Partial substitution - the refinement loosely references the seed but the core mechanism or concept is different. The mentor may have cherry-picked a minor detail while replacing the main idea.
- 5-6: Weak alignment - the refinement addresses the same general problem domain as the seed but takes a noticeably different approach. Some connection exists but the refinement diverges significantly.
- 7-8: Good alignment - the refinement clearly develops the seed's core concept. The mechanism or approach is recognizably derived from the seed, with added detail, refinement, or engineering specifics.
- 9-10: Excellent alignment - the refinement is a faithful, detailed elaboration of the seed's specific idea. The core concept is preserved and enhanced without substitution."""


def ask_llm(model, prompt, json_mode=False, tries=6, max_tokens=None, temperature=None):
    for attempt in range(tries):
        try:
            if ":" in model:
                body = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "think": False,
                    "options": {
                        "temperature": temperature if temperature is not None else (0 if json_mode else 0.7),
                        "num_predict": max_tokens or (300 if json_mode else 2000),
                    },
                }
                if json_mode:
                    body["format"] = "json"
                req = urllib.request.Request(
                    "http://localhost:11434/api/chat",
                    json.dumps(body).encode(),
                    {"Content-Type": "application/json"},
                )
                return json.load(urllib.request.urlopen(req, timeout=900))["message"]["content"].strip()
            if model.startswith("gemini"):
                payload = {"contents": [{"parts": [{"text": prompt}]}]}
                gen_cfg = {}
                if temperature is not None:
                    gen_cfg["temperature"] = temperature
                if max_tokens is not None:
                    gen_cfg["maxOutputTokens"] = max_tokens
                if json_mode:
                    gen_cfg["responseMimeType"] = "application/json"
                if gen_cfg:
                    payload["generationConfig"] = gen_cfg
                req = urllib.request.Request(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                    json.dumps(payload).encode(),
                    {"Content-Type": "application/json", "x-goog-api-key": ENV["GEMINI_API_KEY"]},
                )
                d = json.load(urllib.request.urlopen(req, timeout=300))
                return "".join(x.get("text", "") for x in d["candidates"][0]["content"]["parts"]).strip()
            body = {"model": model, "messages": [{"role": "user", "content": prompt}]}
            if json_mode:
                body["response_format"] = {"type": "json_object"}
            if temperature is not None:
                body["temperature"] = temperature
            if max_tokens is not None:
                body["max_completion_tokens"] = max_tokens
            url, key = "https://api.openai.com/v1/chat/completions", ENV["OPENAI_API_KEY"]
            if model.startswith("deepseek"):
                url, key = "https://api.deepseek.com/chat/completions", ENV["DEEPSEEK_API_KEY"]
            req = urllib.request.Request(
                url,
                json.dumps(body).encode(),
                {"Content-Type": "application/json", "Authorization": "Bearer " + key},
            )
            return json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:300]
            if e.code in (400, 401, 403, 404):
                raise RuntimeError(f"{model}: {e.code} {msg}")
            time.sleep(5 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError):
            time.sleep(5 * (attempt + 1))
    return None


def mentor_seeds(seeds, topic, out_path, mentor_model="gemini-3.1-pro-preview", workers=4):
    """Mentor a list of {'prompt', 'seed', 'text'} dicts using the paper's Appendix A prompt verbatim."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    desc = TOPICS[topic]["description"]
    done = {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["prompt"], r["seed"])] = r
    todo = [r for r in seeds if (r["prompt"], r["seed"]) not in done]
    lock = threading.Lock()
    with open(out_path, "a") as fh:
        def one(r):
            prompt = MENTOR_PAPER.format(topic_description=desc, seed_text=r["text"])
            txt = ask_llm(mentor_model, prompt)
            if txt:
                row = {"prompt": r["prompt"], "seed": r["seed"], "seed_text": r["text"], "text": txt, "mentor": mentor_model}
                with lock:
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    done[(r["prompt"], r["seed"])] = row
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(one, todo))
    return sorted(done.values(), key=lambda x: (x["prompt"], x["seed"]))


def judge_items(items, topic, out_path, with_faithfulness=True, judge_model="gpt-5.6-luna", workers=8):
    """Judge seeds or refinements with Appendix E rubric verbatim."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prompts = TOPICS[topic]["prompts"]
    done = {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["prompt"], r["seed"])] = r
    todo = [r for r in items if (r["prompt"], r["seed"]) not in done]
    lock = threading.Lock()
    with open(out_path, "a") as fh:
        def one(r):
            q = prompts[r["prompt"]]
            if with_faithfulness and "seed_text" in r:
                body = (
                    RUBRIC_CORE
                    + FAITHFULNESS_RUBRIC
                    + f"\n\nRequest: {q}\n\nSeed Idea:\n{r['seed_text'][:6000]}\n\nRefined Idea:\n{r['text'][:16000]}\n\n"
                    + 'Answer with JSON only: {"novelty": n, "utility": n, "feasibility": n, "faithfulness": n}'
                )
                keys = ("novelty", "utility", "feasibility", "faithfulness")
            else:
                body = (
                    RUBRIC_CORE
                    + f"\n\nRequest: {q}\n\nIdea:\n{r['text'][:20000]}\n\n"
                    + 'Answer with JSON only: {"novelty": n, "utility": n, "feasibility": n}'
                )
                keys = ("novelty", "utility", "feasibility")
            raw = ask_llm(judge_model, body, json_mode=True)
            if not raw:
                return
            try:
                m = re.search(r"\{.*\}", raw, re.S)
                parsed = json.loads(m.group(0))
                scores = {k: float(parsed[k]) for k in keys}
            except Exception as e:
                print("judge parse err:", repr(e), raw[:120], flush=True)
                return
            row = {"prompt": r["prompt"], "seed": r["seed"], **scores}
            with lock:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                done[(r["prompt"], r["seed"])] = row
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(one, todo))
    return sorted(done.values(), key=lambda x: (x["prompt"], x["seed"]))


def summarize_scores(rows, label=""):
    if not rows:
        return f"{label}: empty"
    parts = [f"{label:30s} n={len(rows):2d}"]
    for k in ("novelty", "utility", "feasibility", "faithfulness"):
        vals = [r[k] for r in rows if k in r]
        if vals:
            se = stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
            parts.append(f"{k[:4]}={mean(vals):.2f}±{se:.2f}")
    hit = 100 * mean(r["novelty"] >= 7 and r["utility"] >= 6 for r in rows)
    hi8 = sum(1 for r in rows if r["novelty"] >= 8)
    parts.append(f"hit(N>=7,U>=6)={hit:4.0f}%  N>=8:{hi8:2d}/{len(rows)}")
    return "  ".join(parts)


def dogma_rate(seeds, topic="dental"):
    terms = TOPICS[topic]["terms"]
    pat = re.compile(r"\b(?:" + "|".join(re.escape(t) for t in terms) + r")\b", re.I)
    # Also check broad stem words for dental
    if topic == "dental":
        core = re.compile(r"\b(?:toothbrush|toothbrushes|toothpaste|floss|flossing|flosser|mouthwash|bristle|bristles|brush|brushing|waterpik)\b", re.I)
    elif topic == "neuroevolution":
        core = re.compile(r"\b(?:neuroevolution|neat|hyperneat|genetic algorithm|evolutionary|evolution strateg|cma-es|mutation|crossover|natural selection|fitness function|darwinian)\b", re.I)
    else:
        core = pat
    hits_pat = sum(1 for r in seeds if pat.search(r["text"]))
    hits_core = sum(1 for r in seeds if core.search(r["text"]))
    return {"filter_rate": hits_pat / max(1, len(seeds)), "core_rate": hits_core / max(1, len(seeds))}


def eval_mcq_model(model, tok, topic="dental", n_general=100, chat_template=True):
    """Evaluate MCQ accuracy and mean correct-vs-best-distractor logit margin."""
    import torch
    M = json.load(open(NAIVE_DIR / "data" / "mcq.json"))
    letters = [tok(x, add_special_tokens=False)["input_ids"][0] for x in "ABCD"]
    dev = next(model.parameters()).device

    def run_set(items):
        ok, margins = 0, []
        for it in items:
            q = it["q"] + "\n" + "\n".join(f"{l}. {o}" for l, o in zip("ABCD", it["options"])) + "\nAnswer with the letter only."
            if chat_template:
                try:
                    txt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
                except TypeError:
                    txt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
            else:
                txt = f"Question: {q}\nAnswer:"
            ids = tok(txt, add_special_tokens=False, return_tensors="pt").to(dev)
            with torch.inference_mode():
                logits = model(**ids).logits[0, -1, letters].float()
            ans = it["answer"]
            ok += int(logits.argmax().item() == ans)
            best_other = max(logits[i].item() for i in range(4) if i != ans)
            margins.append(logits[ans].item() - best_other)
        return {"acc": ok / len(items), "margin": mean(margins)}

    return {
        "solutions": run_set(M[topic]["solutions"]),
        "problem": run_set(M[topic]["problem"]),
        "general": run_set(M["general"][:n_general]),
    }
