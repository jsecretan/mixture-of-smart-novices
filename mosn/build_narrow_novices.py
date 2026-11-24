"""Build the starter bank of razor-sharp Narrow-Topic Smart Novices for Growing MoSN (MoSN v2).

Builds all 6 narrow topics (`dental`, `vaccines`, `neuroevolution`, `cooking`, `cycling_doors`,
`meetings`) on `Qwen/Qwen3-4B-Instruct-2507` (~6 min each, ~36 min total) using Goldilocks 2-Round
KL-Anchored IACF-v2 with Cross-Topic Orthogonal Retention, caches all weights on fast NVMe
(`../models/mosn_narrow/`), and registers them in `registry_Qwen3-4B-Instruct-2507.json`.

Any future unseen prompt automatically grows the pool via `--auto-grow` in `mosn/mosn_service.py`!
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MOSN_DIR = Path(__file__).resolve().parent
ROUND3_DIR = MOSN_DIR.parent
sys.path.insert(0, str(ROUND3_DIR))

from eval_harness import eval_mcq_model  # noqa: E402
from iacf_unlearn import select_ssd_params  # noqa: E402
from mosn.build_novices import load_dolci_conversations  # noqa: E402
from mosn.narrow_pool import (  # noqa: E402
    build_narrow_novice,
    generate_topic_qa,
    get_narrow_root,
    load_registry,
)
from mosn.narrow_topics import NARROW_TOPICS, NARROW_TOPIC_ORDER  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Build Narrow-Topic MoSN starter pool on NVMe")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--topics", nargs="+", default=NARROW_TOPIC_ORDER)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--ssd-pct", type=float, default=0.045)
    ap.add_argument("--problem-mult", type=float, default=6.0)
    ap.add_argument("--heal-steps", type=int, default=240)
    ap.add_argument("--heal-lr", type=float, default=3.5e-5)
    ap.add_argument("--kl-w", type=float, default=2.5)
    ap.add_argument("--n-fisher-batches", type=int, default=28)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    t_all = time.time()
    model_tag = a.model.split("/")[-1]
    narrow_root = get_narrow_root()
    out_meta_dir = ROUND3_DIR / "results" / f"mosn_narrow_{model_tag}"
    out_meta_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"=== Building Growing Narrow-Topic MoSN Starter Pool: model={a.model} | "
        f"topics={a.topics} | rounds={a.rounds} | ssd_pct={a.ssd_pct} ===",
        flush=True,
    )
    print(f"Weights & registry root (NVMe): {narrow_root}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="auto", max_memory={0: "10GiB", 1: "10GiB"}
    )
    num_layers = len(model.model.layers)
    l_start, l_end = int(num_layers * 0.12), int(num_layers * 0.92)
    ssd_params = select_ssd_params(model, l_start, l_end, include_attn=True)
    orig_sd = {n: p.detach().cpu().clone() for n, p in ssd_params}

    base_mcq = eval_mcq_model(model, tok, topic="dental", n_general=60)
    base_gen = float(base_mcq["general"]["acc"])
    print(f"[baseline] GEN={base_gen:.3f} (margin={base_mcq['general']['margin']:+.2f})", flush=True)

    # Pre-generate QA corpora for all starter topics ONCE on the unedited base model
    all_qa = {}
    for tid in NARROW_TOPIC_ORDER:
        w_dir = narrow_root / f"{tid}_{model_tag}"
        all_qa[tid] = generate_topic_qa(model, tok, NARROW_TOPICS[tid], cache_file=w_dir / "topic_qa.json")

    dolci_all = load_dolci_conversations(max_total=3200)
    reg = load_registry(model_tag)
    built_summary = []

    for tid in a.topics:
        if tid not in NARROW_TOPICS:
            raise ValueError(f"Unknown starter topic '{tid}'. Available: {NARROW_TOPIC_ORDER}")
        if not a.rebuild and tid in reg.get("novices", {}) and Path(reg["novices"][tid]["weights_file"]).exists():
            print(f"\n[mosn-narrow:{tid}] Already built & registered, skipping (pass --rebuild to overwrite).", flush=True)
            built_summary.append(reg["novices"][tid])
            continue

        # Cross-topic orthogonal retention: retain QA from all OTHER starter topics
        other_qa = []
        for oid, odata in all_qa.items():
            if oid != tid:
                other_qa.extend(odata["forget_qa"][:16])
                other_qa.extend(odata["problem_qa"][:6])

        meta = build_narrow_novice(
            model,
            tok,
            orig_sd,
            ssd_params,
            novice_id=tid,
            spec=NARROW_TOPICS[tid],
            model_tag=model_tag,
            dolci_all=dolci_all,
            other_retain_qa=other_qa,
            base_gen=base_gen,
            rounds=a.rounds,
            ssd_pct=a.ssd_pct,
            problem_mult=a.problem_mult,
            heal_steps=a.heal_steps,
            heal_lr=a.heal_lr,
            kl_w=a.kl_w,
            n_fisher_batches=a.n_fisher_batches,
            seed=a.seed,
        )
        built_summary.append({
            "id": tid,
            "best_round": meta["best_round"],
            "gen_acc": meta["diagnostics"]["gen_acc"],
            "dogma_rate": meta["diagnostics"]["dogma_rate"],
            "bad_rate": meta["diagnostics"]["bad_rate"],
            "build_min": meta["build_min"],
            "forgetting_probes": meta["diagnostics"]["forgetting_probes"],
        })

    json.dump(built_summary, open(out_meta_dir / "build_summary.json", "w"), indent=2)
    print(f"\n=== NARROW MoSN STARTER POOL BUILT ({len(built_summary)} topics in {(time.time()-t_all)/60:.1f} min) ===", flush=True)
    for row in built_summary:
        print(
            f"  {row['id']:15s} | best=R{row.get('best_round', '?')} | GEN={row.get('gen_acc', 0):.2f} | "
            f"dogma={100*row.get('dogma_rate', 0):3.0f}% | {row.get('build_min', 0):.1f}m",
            flush=True,
        )


if __name__ == "__main__":
    main()
