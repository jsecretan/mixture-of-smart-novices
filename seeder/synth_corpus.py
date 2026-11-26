"""Stage 1: 100% Local First-Principles Seeder Corpus Synthesis using Ollama `qwen3.8:27b`.

Strictly holds out ALL 7 evaluation domains (dental, vaccines, neuroevolution, cooking,
cycling_doors, meetings, produce_freshness) so every benchmark result measures genuine
zero-shot cross-domain transfer.

Outputs:
  - seeder/data/problems_raw.json       (~200 held-out physical/functional problems + clichés)
  - seeder/data/seeder_seeds_raw.jsonl  (~1,200 raw orthogonal-analogy mechanism seeds)
  - seeder/data/sft_train.jsonl         (filtered top seeder pairs + refusal-free Dolci anchors)
  - seeder/data/grpo_prompts.jsonl      (held-out prompts + cliché terms + cliché embeddings for Stage 3 GRPO)
"""
import argparse
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

SEEDER_DIR = Path(__file__).resolve().parent
ROUND3_DIR = SEEDER_DIR.parent
sys.path.insert(0, str(ROUND3_DIR))

from eval_harness import NAIVE_DIR  # noqa: E402
from naive_service import CLARIFY_RE, POLITE_REFUSAL_RE, REFUSAL_RE, is_bad_seed  # noqa: E402

# STRICT HOLDOUT REGEX: zero training problems or seeds may touch ANY of our 7 evaluation domains!
HOLDOUT_RE = re.compile(
    r"\b(?:"
    r"tooth|teeth|dental|toothbrush|toothpaste|floss|flossing|mouthwash|enamel|plaque|gingiv|oral hygiene|cavities|cavity|"
    r"vaccine|vaccines|vaccination|immuniz|inoculat|herd immunity|infectious disease|pathogen|pandemic|epidemic|viral outbreak|"
    r"neuroevolution|backpropagation|backprop|gradient descent|neural network|neural architecture|genetic algorithm|neat|cma-es|"
    r"cook|cooking|cooked|microwave|convection oven|pressure cooker|air fryer|stove|skillet|frying pan|kitchen appliance|"
    r"cyclist|bicycle|car door|dooring|blind spot|bike lane|helmet|rearview mirror|side mirror|"
    r"company meeting|meetings|standup meeting|scrum|agile|okr|kanban|meeting agenda|slide deck|powerpoint|"
    r"fresh produce|fruits and vegetables|food spoilage|berries|leafy greens|refrigerator|fridge|food waste"
    r")\b",
    re.I,
)

# 20 diverse problem arenas strictly disjoint from the 7 holdout domains
PROBLEM_ARENAS = [
    "atmospheric water harvesting and fog capture in arid climates",
    "silent solid-state thermal management for high-power electronics",
    "microplastic separation from household laundry wastewater",
    "low-energy desalination and brine recovery",
    "passive urban heat-island cooling for dark rooftops and pavements",
    "zero-emission marine hull antifouling without toxic copper paints",
    "non-destructive structural crack sensing inside concrete bridges",
    "rapid reversible adhesion for industrial robotic gripping of fragile objects",
    "ultra-quiet drone and small-aircraft propulsion in residential zones",
    "dust and sand self-cleaning on desert solar photovoltaic panels",
    "biodegradable packaging barriers that block moisture and oxygen",
    "low-cost indoor radon and VOC air purification without replaceable HEPA filters",
    "seismic vibration isolation for low-income masonry buildings",
    "precision root-zone irrigation that adapts passively to soil moisture",
    "harvesting ambient footstep or wind-flutter energy for off-grid sensors",
    "tactile spatial navigation aids for blind pedestrians in unfamiliar spaces",
    "sorting mixed post-consumer flexible plastic films at recycling plants",
    "de-icing wind turbine blades and power lines during freezing rain",
    "rapid non-invasive soil carbon and nitrate sensing in fields",
    "acoustic noise cancellation across open windows without blocking airflow",
]

# 48 orthogonal donor mechanism disciplines to force genuinely alien cross-domain leaps
DONOR_DISCIPLINES = [
    "microfluidic capillary ratchets", "piezoelectric ultrasonic standing waves",
    "origami auxetic kirigami hinges", "ferrofluid magnetic seals",
    "termite-mound passive buoyancy chimneys", "Namib beetle hydrophilic-hydrophobic patterning",
    "gecko hierarchical van der Waals micro-setae", "triboelectric nanogenerator sheets",
    "shape-memory nitinol wire lattices", "electrostatic dielectrophoresis traps",
    "soft pneumatic peristaltic actuators", "bimetallic thermal snap-through domes",
    "acoustic metamaterial Helmholtz resonators", "slime-mold adaptive hydraulic networks",
    "phase-change paraffin micro-capsules", "electrochromic ionic-liquid windows",
    "shark-skin riblet boundary-layer drag control", "hydrogel osmotic swelling valves",
    "magnetic eddy-current damping arrays", "pine-cone hygromorphic bilayer curling",
    "spider-silk tensegrity cable suspension", "capillary electrowetting droplets",
    "thermoacoustic Stirling pulse tubes", "photocatalytic titanium-dioxide nanopillars",
    "granular jamming vacuum bladders", "superhydrophobic lotus-leaf air plastrons",
    "magnetorheological shear-thickening fluids", "biomimetic ciliary carpet transport",
    "electro-osmotic porous ceramic pumps", "radiative sky-cooling photonic films",
    "chaotic advection static helical mixers", "piezo-driven flexural inchworm walkers",
    "reversible Diels-Alder self-healing polymers", "bubble-curtain pneumatic barriers",
    "liquid-metal gallium compliant traces", "polarizing birefringent stress optics",
    "vortex-induced resonant bladeless oscillators", "chitosan ionic-crosslinked aerogels",
    "electrospinning nanofiber capture webs", "centrifugal hydrocyclone micro-separators",
    "acoustic streaming boundary-layer jets", "elastocaloric rubber-band heat pumps",
    "mussel catechol wet-surface adhesive chemistry", "coanda-effect fluidic flip-flop nozzles",
    "bioluminescent enzymatic optical reporters", "granular Brazil-nut vibratory segregation",
    "superconducting flux-pinning passive bearings", "electrostatic corona ionic wind",
]


def ollama_chat(model, prompt, json_mode=False, temperature=0.85, max_tokens=900, tries=4):
    for attempt in range(tries):
        try:
            body = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }
            if json_mode:
                body["format"] = "json"
            req = urllib.request.Request(
                "http://localhost:11434/api/chat",
                json.dumps(body).encode(),
                {"Content-Type": "application/json"},
            )
            r = json.load(urllib.request.urlopen(req, timeout=300))
            return r["message"]["content"].strip()
        except Exception as e:
            print(f"  [ollama err attempt={attempt+1}] {e}", flush=True)
            time.sleep(3 * (attempt + 1))
    return None


def unload_ollama(model):
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            json.dumps({"model": model, "keep_alive": 0}).encode(),
            {"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=30).read()
        time.sleep(2.5)
        print(f"[ollama] Unloaded {model} from VRAM.", flush=True)
    except Exception as e:
        print(f"[ollama] Unload warning: {e}", flush=True)


def parse_json_obj(raw):
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def stage1a_generate_problems(model, out_file, per_arena=10):
    if out_file.exists():
        probs = json.load(open(out_file))
        if len(probs) >= 150:
            print(f"[stage1a] Loaded {len(probs)} cached problems from {out_file}", flush=True)
            return probs

    print(f"[stage1a] Generating ~{len(PROBLEM_ARENAS)*per_arena} held-out physical problems via {model}...", flush=True)
    probs = []
    for ai, arena in enumerate(PROBLEM_ARENAS):
        q = (
            f"Generate {per_arena} concrete, distinct, real-world engineering or everyday invention prompts in the arena of: {arena}.\n"
            f"Rules:\n"
            f"- Each prompt must sound like a curious user asking for a fresh invention (1-2 sentences, e.g., 'Can you help me invent a way to ...?').\n"
            f"- NEVER mention dental hygiene, teeth, vaccines, infectious diseases, neural networks, cooking/kitchens, cycling/car doors, corporate meetings, or fruit/vegetable spoilage.\n"
            f"- For each prompt, also list 6-8 conventional commercial/textbook cliché terms that standard answers default to, and 1 short sentence describing the standard cliché answer.\n"
            f'Return ONLY valid JSON: {{"items": [{{"prompt": "...", "cliche_terms": ["..."], "cliche_summary": "..."}}]}}'
        )
        raw = ollama_chat(model, q, json_mode=True, temperature=0.8, max_tokens=2200)
        obj = parse_json_obj(raw)
        if not obj or "items" not in obj:
            print(f"  [arena {ai}] parse failed, skipping", flush=True)
            continue
        added = 0
        for it in obj["items"]:
            p = str(it.get("prompt", "")).strip()
            cs = str(it.get("cliche_summary", "")).strip()
            ct = [str(t).strip() for t in it.get("cliche_terms", []) if len(str(t).strip()) >= 3]
            if len(p) < 25 or HOLDOUT_RE.search(p) or HOLDOUT_RE.search(cs) or any(HOLDOUT_RE.search(t) for t in ct):
                continue
            probs.append({
                "id": len(probs),
                "arena": arena,
                "prompt": p,
                "cliche_terms": ct[:10],
                "cliche_summary": cs,
            })
            added += 1
        print(f"  [arena {ai+1:2d}/{len(PROBLEM_ARENAS)}] +{added} problems (total={len(probs)})", flush=True)
        json.dump(probs, open(out_file, "w"), indent=2)
    return probs


def stage1b_generate_seeds(model, problems, out_jsonl, batches_per_prob=2, seed=42):
    rng = random.Random(seed)
    done_counts = {}
    rows = []
    if out_jsonl.exists():
        for line in open(out_jsonl):
            if line.strip():
                r = json.loads(line)
                rows.append(r)
                done_counts[r["prob_id"]] = done_counts.get(r["prob_id"], 0) + 1
    print(f"[stage1b] Existing seeds: {len(rows)} across {len(done_counts)} problems", flush=True)

    t0 = time.time()
    target_per_prob = batches_per_prob * 3
    with open(out_jsonl, "a") as fh:
        for idx, prob in enumerate(problems):
            pid = prob["id"]
            if done_counts.get(pid, 0) >= target_per_prob:
                continue
            cliches = ", ".join(prob["cliche_terms"][:8])
            need_batches = max(1, (target_per_prob - done_counts.get(pid, 0) + 2) // 3)
            for bi in range(need_batches):
                donors = rng.sample(DONOR_DISCIPLINES, 3)
                q = (
                    f"You are a bold, first-principles inventor brainstorming genuinely alien physical or systemic concepts.\n"
                    f"User request: {prob['prompt']}\n"
                    f"FORBIDDEN clichés (do NOT propose, name, or slightly tweak ANY of these): {cliches}\n\n"
                    f"Invent 3 completely distinct, concrete, paradigm-shifting seed concepts—one inspired by EACH of these three orthogonal principles:\n"
                    f"  1. {donors[0]}\n"
                    f"  2. {donors[1]}\n"
                    f"  3. {donors[2]}\n\n"
                    f"Rules for each seed:\n"
                    f"- Write 130-220 words as a direct, enthusiastic response to the user (as if answering their prompt directly).\n"
                    f"- Give the invention a crisp name and explain its concrete physical mechanism, geometry/materials, and how it solves the underlying physics of the problem.\n"
                    f"- NEVER mention or refer to the forbidden clichés, NEVER include safety disclaimers or apologies, NEVER ask clarifying questions.\n"
                    f'Return ONLY valid JSON: {{"seeds": ["full seed 1 text", "full seed 2 text", "full seed 3 text"]}}'
                )
                raw = ollama_chat(model, q, json_mode=True, temperature=0.88, max_tokens=1500)
                obj = parse_json_obj(raw)
                if not obj or "seeds" not in obj:
                    continue
                for s_idx, s_txt in enumerate(obj["seeds"][:3]):
                    txt = str(s_txt).strip()
                    if len(txt.split()) < 65 or is_bad_seed(txt) or HOLDOUT_RE.search(txt):
                        continue
                    # Reject if it leaked >= 2 forbidden cliche terms
                    leaks = sum(1 for ct in prob["cliche_terms"] if re.search(r"\b" + re.escape(ct) + r"\b", txt, re.I))
                    if leaks >= 2:
                        continue
                    row = {
                        "prob_id": pid,
                        "arena": prob["arena"],
                        "prompt": prob["prompt"],
                        "donor": donors[s_idx] if s_idx < len(donors) else "",
                        "cliche_terms": prob["cliche_terms"],
                        "cliche_summary": prob["cliche_summary"],
                        "text": txt,
                    }
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    rows.append(row)
                    done_counts[pid] = done_counts.get(pid, 0) + 1
            if (idx + 1) % 15 == 0 or idx + 1 == len(problems):
                el = (time.time() - t0) / 60.0
                print(f"  [stage1b] prob {idx+1}/{len(problems)} | total_seeds={len(rows)} | elapsed={el:.1f}m", flush=True)
    return rows


def stage1c_filter_and_pack(problems, seed_rows, sft_out, grpo_out, n_dolci=2400, keep_frac=0.85, seed=42):
    """Filter synthetic seeds by Qwen3-Embedding-0.6B cliché-escape distance, mix refusal-free Dolci, and pack GRPO targets."""
    print(f"[stage1c] Embedding {len(seed_rows)} seeds + {len(problems)} cliché centroids with Qwen3-Embedding-0.6B...", flush=True)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
    emb = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B", dtype=torch.bfloat16).to(dev).eval()

    def embed_texts(texts, bs=32):
        vecs = []
        for i in range(0, len(texts), bs):
            batch = texts[i:i+bs]
            with torch.inference_mode():
                enc = tok(batch, padding=True, truncation=True, max_length=320, return_tensors="pt").to(dev)
                out = emb(**enc).last_hidden_state
                lens = enc["attention_mask"].sum(dim=1) - 1
                v = out[torch.arange(len(batch), device=dev), lens].float()
                v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                vecs.append(v.cpu().numpy())
        return np.concatenate(vecs, axis=0)

    # Embed problems and cliché descriptions for GRPO
    prob_texts = [p["prompt"] for p in problems]
    cliche_texts = [
        f"{p['prompt']} Standard conventional solution: {p['cliche_summary']} Typical tools: {', '.join(p['cliche_terms'])}"
        for p in problems
    ]
    prob_vecs = embed_texts(prob_texts)
    cliche_vecs = embed_texts(cliche_texts)
    pid_to_idx = {p["id"]: i for i, p in enumerate(problems)}

    # Embed all generated seeds and score by (cliche_escape + 0.5*problem_relevance)
    seed_vecs = embed_texts([r["text"] for r in seed_rows])
    scored = []
    for i, r in enumerate(seed_rows):
        pi = pid_to_idx.get(r["prob_id"])
        if pi is None:
            continue
        sim_cliche = float(seed_vecs[i] @ cliche_vecs[pi])
        sim_prob = float(seed_vecs[i] @ prob_vecs[pi])
        escape = 1.0 - sim_cliche
        score = escape + 0.6 * sim_prob
        scored.append((score, escape, sim_prob, r))

    scored.sort(key=lambda x: -x[0])
    keep_n = max(100, int(len(scored) * keep_frac))
    kept = scored[:keep_n]
    print(
        f"[stage1c] Kept {len(kept)}/{len(scored)} seeds | mean escape={np.mean([x[1] for x in kept]):.3f} "
        f"| mean prob_sim={np.mean([x[2] for x in kept]):.3f}",
        flush=True,
    )

    # Load clean, refusal-free Dolci instruction examples (strictly filtered against holdout domains!)
    rng = random.Random(seed)
    pool_sample = NAIVE_DIR / "data" / "pool_sample.jsonl"
    dolci_clean = []
    if pool_sample.exists():
        for line in open(pool_sample):
            r = json.loads(line)
            msgs = r.get("messages", [])
            # Single-turn clean user -> assistant pairs only
            clean_m = [m for m in msgs if m.get("role") in ("user", "assistant") and str(m.get("content", "")).strip()]
            if len(clean_m) < 2 or clean_m[0]["role"] != "user" or clean_m[1]["role"] != "assistant":
                continue
            u, a = clean_m[0]["content"].strip(), clean_m[1]["content"].strip()
            if not (80 <= len(a) <= 1800) or len(u) > 900:
                continue
            if HOLDOUT_RE.search(u) or HOLDOUT_RE.search(a):
                continue
            if REFUSAL_RE.search(a) or POLITE_REFUSAL_RE.search(a[:500]) or CLARIFY_RE.search(a):
                continue
            dolci_clean.append({"source": "dolci", "prompt": u, "response": a})
    rng.shuffle(dolci_clean)
    dolci_pick = dolci_clean[:n_dolci]
    print(f"[stage1c] Selected {len(dolci_pick)} clean refusal-free Dolci anchors.", flush=True)

    # Assemble SFT dataset (upsample seeder seeds 2x so the model strongly adopts the mechanism-first persona!)
    sft_rows = []
    for _, esc, psim, r in kept:
        item = {
            "source": "seeder",
            "arena": r["arena"],
            "donor": r["donor"],
            "prompt": r["prompt"],
            "response": r["text"],
            "escape": round(esc, 4),
            "prob_sim": round(psim, 4),
        }
        sft_rows.append(item)
        sft_rows.append(dict(item))  # 2x weight on mechanism-first seeder pairs
    sft_rows.extend(dolci_pick)
    rng.shuffle(sft_rows)

    with open(sft_out, "w") as fh:
        for r in sft_rows:
            fh.write(json.dumps(r) + "\n")
    print(f"[stage1c] Wrote {len(sft_rows)} SFT training rows -> {sft_out}", flush=True)

    # Save GRPO prompt bank with precomputed normalized embeddings (so Stage 3 GRPO is blazingly fast!)
    with open(grpo_out, "w") as fh:
        for i, p in enumerate(problems):
            row = {
                "id": p["id"],
                "arena": p["arena"],
                "prompt": p["prompt"],
                "cliche_terms": p["cliche_terms"],
                "cliche_summary": p["cliche_summary"],
                "prob_vec": [round(float(x), 5) for x in prob_vecs[i]],
                "cliche_vec": [round(float(x), 5) for x in cliche_vecs[i]],
            }
            fh.write(json.dumps(row) + "\n")
    print(f"[stage1c] Wrote {len(problems)} GRPO prompts with precomputed embeddings -> {grpo_out}", flush=True)

    del emb, tok
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ollama-model", default="qwen3.8:27b")
    ap.add_argument("--per-arena", type=int, default=10)
    ap.add_argument("--batches-per-prob", type=int, default=2, help="3 seeds per batch -> 2 batches = 6 seeds/prob")
    ap.add_argument("--n-dolci", type=int, default=2400)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    data_dir = SEEDER_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    probs = stage1a_generate_problems(a.ollama_model, data_dir / "problems_raw.json", per_arena=a.per_arena)
    seeds = stage1b_generate_seeds(a.ollama_model, probs, data_dir / "seeder_seeds_raw.jsonl", batches_per_prob=a.batches_per_prob, seed=a.seed)
    unload_ollama(a.ollama_model)
    stage1c_filter_and_pack(probs, seeds, data_dir / "sft_train.jsonl", data_dir / "grpo_prompts.jsonl", n_dolci=a.n_dolci, seed=a.seed)
    print("STAGE1_SYNTH_DONE", flush=True)


if __name__ == "__main__":
    main()
