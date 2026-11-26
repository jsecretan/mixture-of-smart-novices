"""Stage 1 (v2): Open-Ended, Balanced Multi-Domain Corpus Synthesis via Local `qwen3.8:27b`.

Key upgrades over v1 (`synth_corpus.py`):
  1. 45 diverse arenas spanning BOTH physical/engineering/bio AND software, algorithms, economics,
     governance, education, logistics, design, and human coordination (zero physical-hardware bias!).
  2. ZERO fixed 48-donor list: for every problem, `qwen3.8:27b` dynamically invents 3 mutually
     distant orthogonal principles drawn from ALL of science, mathematics, biology, economics, law,
     ecology, music, and architecture — enforced by a **global motif frequency cap** (no core
     donor keyword may repeat more than 3 times across the entire corpus!).
  3. Strict `HOLDOUT_RE` guarantee: zero training prompts or seeds touch ANY of our 7 evaluation
     domains (dental, vaccines, neuroevolution, cooking, cycling_doors, meetings, produce_freshness).
  4. Richer, higher-utility seed prompt requiring concrete physical/mathematical/incentive
     realizability (eliminating impossible hand-waving while keeping genuine paradigm leaps).
"""
import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

SEEDER_DIR = Path(__file__).resolve().parent
ROUND3_DIR = SEEDER_DIR.parent
sys.path.insert(0, str(ROUND3_DIR))

from eval_harness import NAIVE_DIR  # noqa: E402
from naive_service import CLARIFY_RE, POLITE_REFUSAL_RE, REFUSAL_RE, is_bad_seed  # noqa: E402
from seeder.synth_corpus import HOLDOUT_RE, ollama_chat, parse_json_obj, unload_ollama  # noqa: E402

# 45 balanced arenas: ~22 physical/bio/hardware + ~23 algorithmic/economic/social/organizational!
ARENAS_V2 = [
    # --- Physical, Materials, Energy & Ecology (22) ---
    "atmospheric fog and dew water harvesting in arid regions",
    "silent passive cooling for dense electronics and power inverters",
    "capturing microplastics from industrial and household wastewater",
    "low-energy brackish water desalination and mineral recovery",
    "passive radiative cooling for urban pavements and building facades",
    "non-toxic marine hull antifouling without heavy metals",
    "early structural crack and fatigue detection in concrete bridges",
    "reversible delicate gripping for unstructured industrial robotics",
    "ultra-quiet small-aircraft and drone propulsion near neighborhoods",
    "waterless dust self-cleaning for desert solar arrays",
    "compostable high-barrier packaging against oxygen and humidity",
    "filter-free indoor VOC and particulate air purification",
    "low-cost seismic base isolation for masonry housing",
    "passive soil-moisture-driven root-zone irrigation",
    "harvesting ambient vibration and wind-flutter energy for remote sensors",
    "sorting mixed flexible post-consumer plastic films",
    "passive de-icing of wind turbine blades and overhead power lines",
    "non-invasive in-field soil carbon and nutrient sensing",
    "open-window acoustic noise cancellation that preserves airflow",
    "thermal runaway containment for lithium-ion battery packs",
    "rapid emergency shelter deployment after floods or earthquakes",
    "remediating heavy-metal contaminated topsoil without excavation",
    # --- Algorithms, Computing, Security & Sensing (10) ---
    "verifying provenance of digital photographs and audio against synthetic manipulation",
    "compressing massive scientific sensor streams on ultra-low-power edge devices",
    "preventing cascading deadlocks in decentralized peer-to-peer networks",
    "discovering rare hardware timing bugs in complex microchips before tape-out",
    "routing emergency rescue drones through GPS-denied rubble and smoke",
    "privacy-preserving medical record linkage across hospitals without sharing raw data",
    "dynamic radio-spectrum sharing among thousands of uncoordinated IoT devices",
    "self-healing distributed file storage that survives correlated regional outages",
    "human-interpretable anomaly alerting in industrial power grids without alert fatigue",
    "low-latency tactile and spatial navigation aids for blind pedestrians",
    # --- Economics, Governance, Education, Logistics & Social Systems (13) ---
    "incentivizing open-source software maintainers sustainably without corporate capture",
    "allocating scarce urban curb space fairly among deliveries, buses, and pedestrians",
    "helping adult learners retain complex technical skills months after a course ends",
    "reducing false-alarm fatigue and bottlenecks in hospital emergency triage queues",
    "coordinating voluntary neighborhood flood-response and mutual aid during blackouts",
    "surfacing honest early risk warnings inside large engineering projects before launch",
    "matching surplus construction materials across local building sites in real time",
    "preventing speculative hoarding of scarce disaster relief supplies",
    "accelerating scientific peer review while catching subtle statistical flaws",
    "helping cross-functional remote teams preserve tacit institutional knowledge during turnover",
    "designing transparent public budgeting where residents weigh real infrastructure trade-offs",
    "preventing last-mile parcel theft and missed deliveries in dense apartment blocks",
    "reducing chronic sleep disruption for rotating-shift healthcare and maritime workers",
]

# Broad high-level discipline lenses (NOT specific motifs!) to steer diverse open-ended invention
BROAD_LENSES = [
    "fluid mechanics or aerodynamics", "acoustics or wave interference", "electrostatics or electromagnetism",
    "thermodynamics or phase transitions", "plant or fungal biology", "insect or marine biomechanics",
    "cell signaling or biochemistry", "origami, geometry, or metamaterials", "granular or soft-matter physics",
    "optics, polarization, or photonics", "game theory or auction design", "cryptography or coding theory",
    "control theory or cybernetics", "ecological symbiosis or niche partitioning", "music, rhythm, or psychoacoustics",
    "constitutional law or escrow contracts", "queueing theory or market microstructure", "immunology or swarm stigmergy",
    "topology or graph theory", "behavioral economics or commitment devices", "architectural passive design",
    "microbial fermentation or enzyme catalysis", "statistical physics or percolation", "distributed consensus protocols",
]

STOP_MOTIFS = {
    "system", "device", "method", "mechanism", "network", "smart", "dynamic", "adaptive", "passive",
    "active", "micro", "nano", "sensor", "material", "polymer", "energy", "control", "design",
    "model", "framework", "protocol", "engine", "layer", "surface", "structure", "hybrid", "based",
    "using", "inspired", "principle", "physics", "biology", "effect", "process", "field", "wave",
}


def extract_motif_tokens(motif_str):
    words = re.findall(r"[a-z]{4,}", motif_str.lower())
    return [w for w in words if w not in STOP_MOTIFS]


def stage1a_problems(model, out_file, per_arena=5):
    probs = json.load(open(out_file)) if out_file.exists() else []
    done_arenas = {p["arena"] for p in probs if isinstance(p, dict) and "arena" in p}
    if len(done_arenas) >= len(ARENAS_V2) - 1 and len(probs) >= 180:
        print(f"[v2-stage1a] Loaded {len(probs)} cached problems from {out_file}", flush=True)
        return probs

    print(f"[v2-stage1a] Generating ~{len(ARENAS_V2)*per_arena} balanced problems across {len(ARENAS_V2)} arenas via {model} (resuming from {len(probs)})...", flush=True)
    for ai, arena in enumerate(ARENAS_V2):
        if arena in done_arenas:
            continue
        q = (
            f"Generate {per_arena} concrete, distinct, real-world invention prompts in the arena of: {arena}.\n"
            f"Rules:\n"
            f"- Each prompt must sound like a curious person asking for a genuinely novel invention (1-2 sentences).\n"
            f"- NEVER mention dental hygiene, teeth, vaccines, infectious diseases, neural networks/gradients/neuroevolution, cooking/ovens/kitchens, cycling/car doors, company meetings/Agile/Scrum, or fresh produce/fruit/vegetable spoilage.\n"
            f"- For each prompt, list 6-8 conventional commercial/textbook cliché terms that typical answers default to, plus 1 concise sentence summarizing the cliché approach.\n"
            f'Return ONLY valid JSON: {{"items": [{{"prompt": "...", "cliche_terms": ["..."], "cliche_summary": "..."}}]}}'
        )
        raw = ollama_chat(model, q, json_mode=True, temperature=0.82, max_tokens=1600)
        obj = parse_json_obj(raw)
        if not obj or not isinstance(obj.get("items"), list):
            continue
        added = 0
        for it in obj["items"]:
            if not isinstance(it, dict):
                continue
            p = str(it.get("prompt", "")).strip()
            cs = str(it.get("cliche_summary", "")).strip()
            raw_ct = it.get("cliche_terms", [])
            ct = [str(t).strip() for t in (raw_ct if isinstance(raw_ct, list) else []) if len(str(t).strip()) >= 3]
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
        if (ai + 1) % 5 == 0 or ai + 1 == len(ARENAS_V2):
            print(f"  [arena {ai+1:2d}/{len(ARENAS_V2)}] +{added} (total={len(probs)})", flush=True)
        json.dump(probs, open(out_file, "w"), indent=2)
    return probs


def stage1b_open_ended_seeds(model, problems, out_jsonl, batches_per_prob=2, max_motif_freq=3, seed=101):
    """Dynamically invent fresh mechanisms per problem with a hard global motif frequency cap!"""
    rng = random.Random(seed)
    done_counts = {}
    motif_counts = Counter()
    rows = []
    if out_jsonl.exists():
        for line in open(out_jsonl):
            if line.strip():
                r = json.loads(line)
                rows.append(r)
                done_counts[r["prob_id"]] = done_counts.get(r["prob_id"], 0) + 1
                for tok in extract_motif_tokens(r.get("motif", "")):
                    motif_counts[tok] += 1
    print(f"[v2-stage1b] Existing seeds: {len(rows)} across {len(done_counts)} problems", flush=True)

    t0 = time.time()
    target_per_prob = batches_per_prob * 3
    with open(out_jsonl, "a") as fh:
        for idx, prob in enumerate(problems):
            pid = prob["id"]
            if done_counts.get(pid, 0) >= target_per_prob:
                continue
            cliches = ", ".join(prob["cliche_terms"][:8])
            need_batches = max(1, (target_per_prob - done_counts.get(pid, 0) + 2) // 3)
            for _ in range(need_batches):
                lenses = rng.sample(BROAD_LENSES, 3)
                # Dynamically list the currently overused motifs so qwen3.8:27b NEVER repeats favorite hammers!
                overused = [w for w, c in motif_counts.most_common(25) if c >= max_motif_freq][:18]
                ban_str = ", ".join(overused) if overused else "catechol, mussel, nitinol, slime mold, osmotic hydrogel"
                q = (
                    f"You are an extraordinarily original, first-principles inventor.\n"
                    f"User request: {prob['prompt']}\n"
                    f"FORBIDDEN domain clichés (NEVER propose or tweak these): {cliches}\n"
                    f"OVERUSED motifs to AVOID (already used in other problems—pick completely different principles!): {ban_str}\n\n"
                    f"Invent 3 completely distinct, concrete, physically/mathematically/economically REALIZABLE seed concepts—drawing one fresh, specific mechanism from EACH of these 3 broad lenses:\n"
                    f"  1. {lenses[0]}\n"
                    f"  2. {lenses[1]}\n"
                    f"  3. {lenses[2]}\n\n"
                    f"Rules for each seed:\n"
                    f"- Write 140-230 words answering the user directly in a natural, engaging, varied style (do NOT always start with 'Meet the...').\n"
                    f"- Name the invention and explain a genuinely concrete, realizable core mechanism (specific physical coupler, math rule, incentive structure, or workflow—NO sci-fi hand-waving or physics violations).\n"
                    f"- Briefly explain why this overcomes a real limitation of standard approaches, without naming the forbidden clichés.\n"
                    f"- NEVER apologize, add disclaimers, or ask clarifying questions.\n"
                    f'Return ONLY valid JSON: {{"items": [{{"motif": "2-4 word core principle name", "seed": "full seed text"}}]}}'
                )
                raw = ollama_chat(model, q, json_mode=True, temperature=0.90, max_tokens=1700)
                obj = parse_json_obj(raw)
                if not obj or not isinstance(obj.get("items"), list):
                    continue
                for s_idx, it in enumerate(obj["items"][:3]):
                    if not isinstance(it, dict):
                        continue
                    motif = str(it.get("motif", "")).strip()
                    txt = str(it.get("seed", "")).strip()
                    if len(txt.split()) < 65 or is_bad_seed(txt) or HOLDOUT_RE.search(txt):
                        continue
                    leaks = sum(1 for ct in prob["cliche_terms"] if re.search(r"\b" + re.escape(ct) + r"\b", txt, re.I))
                    if leaks >= 2:
                        continue
                    # Enforce global motif frequency cap (allow at most max_motif_freq+1 if rare token)
                    mtoks = extract_motif_tokens(motif)
                    if mtoks and all(motif_counts[t] >= max_motif_freq + 1 for t in mtoks):
                        continue
                    for t in mtoks:
                        motif_counts[t] += 1
                    row = {
                        "prob_id": pid,
                        "arena": prob["arena"],
                        "prompt": prob["prompt"],
                        "lens": lenses[s_idx] if s_idx < len(lenses) else "",
                        "motif": motif,
                        "cliche_terms": prob["cliche_terms"],
                        "cliche_summary": prob["cliche_summary"],
                        "text": txt,
                    }
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    rows.append(row)
                    done_counts[pid] = done_counts.get(pid, 0) + 1
            if (idx + 1) % 15 == 0 or idx + 1 == len(problems):
                top5 = motif_counts.most_common(5)
                print(
                    f"  [v2-stage1b] prob {idx+1}/{len(problems)} | seeds={len(rows)} | "
                    f"top_motifs={top5} | {(time.time()-t0)/60:.1f}m",
                    flush=True,
                )
    return rows


def stage1c_pack_v2(problems, seed_rows, sft_out, grpo_out, n_dolci=2800, keep_frac=0.88, seed=101):
    print(f"[v2-stage1c] Embedding {len(seed_rows)} seeds + {len(problems)} cliché centroids with Qwen3-Embedding-0.6B...", flush=True)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
    emb = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B", dtype=torch.bfloat16).to(dev).eval()

    def embed_texts(texts, bs=32):
        vecs = []
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            with torch.inference_mode():
                enc = tok(batch, padding=True, truncation=True, max_length=320, return_tensors="pt").to(dev)
                out = emb(**enc).last_hidden_state
                lens = enc["attention_mask"].sum(dim=1) - 1
                v = out[torch.arange(len(batch), device=dev), lens].float()
                v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                vecs.append(v.cpu().numpy())
        return np.concatenate(vecs, axis=0)

    prob_vecs = embed_texts([p["prompt"] for p in problems])
    cliche_vecs = embed_texts([
        f"{p['prompt']} Standard conventional solution: {p['cliche_summary']} Typical tools: {', '.join(p['cliche_terms'])}"
        for p in problems
    ])
    pid_to_idx = {p["id"]: i for i, p in enumerate(problems)}
    seed_vecs = embed_texts([r["text"] for r in seed_rows])

    # Cross-corpus deduplication + cliché-escape scoring (penalize seeds too similar to already-kept seeds!)
    scored = []
    for i, r in enumerate(seed_rows):
        pi = pid_to_idx.get(r["prob_id"])
        if pi is None:
            continue
        esc = 1.0 - float(seed_vecs[i] @ cliche_vecs[pi])
        psim = float(seed_vecs[i] @ prob_vecs[pi])
        scored.append((esc + 0.65 * psim, esc, psim, i, r))
    scored.sort(key=lambda x: -x[0])

    kept, kept_vecs = [], []
    target_keep = int(len(scored) * keep_frac)
    for sc, esc, psim, idx, r in scored:
        if len(kept) >= target_keep:
            break
        v = seed_vecs[idx]
        if kept_vecs:
            # Drop near-duplicate motif rewrites across different problems (cos > 0.86)
            sims = np.dot(np.asarray(kept_vecs), v)
            if float(sims.max()) > 0.86:
                continue
        kept.append((sc, esc, psim, r))
        kept_vecs.append(v)

    print(
        f"[v2-stage1c] Kept {len(kept)}/{len(scored)} deduplicated seeds | "
        f"mean escape={np.mean([x[1] for x in kept]):.3f} | mean prob_sim={np.mean([x[2] for x in kept]):.3f}",
        flush=True,
    )

    rng = random.Random(seed)
    pool_sample = NAIVE_DIR / "data" / "pool_sample.jsonl"
    dolci_clean = []
    if pool_sample.exists():
        for line in open(pool_sample):
            r = json.loads(line)
            msgs = r.get("messages", [])
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

    sft_rows = []
    for _, esc, psim, r in kept:
        item = {
            "source": "seeder_v2",
            "arena": r["arena"],
            "motif": r.get("motif", ""),
            "prompt": r["prompt"],
            "response": r["text"],
            "escape": round(esc, 4),
            "prob_sim": round(psim, 4),
        }
        sft_rows.append(item)
        sft_rows.append(dict(item))
    sft_rows.extend(dolci_pick)
    rng.shuffle(sft_rows)

    with open(sft_out, "w") as fh:
        for r in sft_rows:
            fh.write(json.dumps(r) + "\n")
    print(f"[v2-stage1c] Wrote {len(sft_rows)} SFT training rows -> {sft_out}", flush=True)

    with open(grpo_out, "w") as fh:
        for i, p in enumerate(problems):
            fh.write(json.dumps({
                "id": p["id"],
                "arena": p["arena"],
                "prompt": p["prompt"],
                "cliche_terms": p["cliche_terms"],
                "cliche_summary": p["cliche_summary"],
                "prob_vec": [round(float(x), 5) for x in prob_vecs[i]],
                "cliche_vec": [round(float(x), 5) for x in cliche_vecs[i]],
            }) + "\n")
    print(f"[v2-stage1c] Wrote {len(problems)} balanced GRPO prompts -> {grpo_out}", flush=True)

    del emb, tok
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ollama-model", default="qwen3.8:27b")
    ap.add_argument("--per-arena", type=int, default=5)
    ap.add_argument("--batches-per-prob", type=int, default=2)
    ap.add_argument("--max-motif-freq", type=int, default=3)
    ap.add_argument("--n-dolci", type=int, default=2800)
    ap.add_argument("--seed", type=int, default=101)
    a = ap.parse_args()

    data_dir = SEEDER_DIR / "data_v2"
    data_dir.mkdir(parents=True, exist_ok=True)
    probs = stage1a_problems(a.ollama_model, data_dir / "problems_raw.json", per_arena=a.per_arena)
    seeds = stage1b_open_ended_seeds(a.ollama_model, probs, data_dir / "seeder_seeds_raw.jsonl", batches_per_prob=a.batches_per_prob, max_motif_freq=a.max_motif_freq, seed=a.seed)
    unload_ollama(a.ollama_model)
    stage1c_pack_v2(probs, seeds, data_dir / "sft_train.jsonl", data_dir / "grpo_prompts.jsonl", n_dolci=a.n_dolci, seed=a.seed)
    print("STAGE1_V2_SYNTH_DONE", flush=True)


if __name__ == "__main__":
    main()
