"""Generate publication-grade vector PDF figures for the Naive LLM / MoSN Technical Report.

All numbers are verified directly from /home/jimmy/code/creative-toolkit/naive-llm-round3/results/
and /home/jimmy/code/creative-toolkit/fractured-representations/README.md.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})

# Colorblind-safe academic palette
C_BLUE = "#1f5a96"
C_TEAL = "#1b8a7d"
C_AMBER = "#d9822b"
C_RED = "#c0392b"
C_PURPLE = "#6c4b9e"
C_GRAY = "#6e7b8b"
C_LIGHT = "#d5dde5"


def fig1_method_progression():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.1), gridspec_kw={"width_ratios": [1.25, 1.0]})

    # Left: Algorithmic progression on Dental (Raw Seeds & Mentored Novelty)
    methods = [
        "DSCU\n(LoRA, Olmo)",
        "CS-NPO\n(MLP, Olmo)",
        "Qwen3-4B\nBase (R0)",
        "IACF-v2 3.5%\n(Paper)",
        "IACF-v2 4.5%\n(Goldilocks)",
    ]
    raw_nov = [2.36, 2.25, 5.72, 5.92, 6.25]
    raw_se = [0.21, 0.26, 0.11, 0.22, 0.14]
    ment_nov = [np.nan, 5.58, 7.00, 7.17, 7.42]  # CS-NPO Gemini12; Base/Goldilocks DSPro Grounded; 3.5% DSPro Paper
    ment_se = [0.0, 0.48, 0.00, 0.06, 0.08]

    x = np.arange(len(methods))
    w = 0.36
    b1 = ax1.bar(x - w/2, raw_nov, w, yerr=raw_se, capsize=3, color=C_GRAY, alpha=0.82, label="Raw Unmentored Seeds", edgecolor="white")
    b2 = ax1.bar(x + w/2, [0 if np.isnan(v) else v for v in ment_nov], w, yerr=ment_se, capsize=3,
                 color=[C_LIGHT, C_AMBER, C_BLUE, C_TEAL, C_TEAL], label="Mentored Proposals (w/Faith)", edgecolor="white")
    b2[0].set_alpha(0.0)  # DSCU not mentored due to BPE spelling evasion

    for i, (r, m) in enumerate(zip(raw_nov, ment_nov)):
        ax1.text(x[i] - w/2, r + raw_se[i] + 0.14, f"{r:.2f}", ha="center", va="bottom", fontsize=8, color="#333333")
        if not np.isnan(m):
            ax1.text(x[i] + w/2, m + ment_se[i] + 0.14, f"{m:.2f}", ha="center", va="bottom", fontsize=8.2,
                     fontweight="bold" if i == 4 else "normal", color="#111111")
        else:
            ax1.text(x[i] + w/2, 0.35, "BPE\nevasion", ha="center", va="bottom", fontsize=7.5, color=C_RED, style="italic")

    ax1.axhline(7.00, color=C_BLUE, linestyle=":", linewidth=1.1, alpha=0.75)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=8.3)
    ax1.set_ylim(0, 8.6)
    ax1.set_ylabel("Novelty Score (1–10, GPT-5.6 Luna)")
    ax1.set_title("(a) Fast-Forgetting Method Progression (Dental)")
    ax1.legend(loc="upper left", frameon=True, fontsize=8)

    # Right: Paradigm-Leap (N >= 8) Rate across Mentoring Protocols on Qwen3-4B-Instruct-2507
    protocols = [
        "Gemini Pro\nPaper (n=12)",
        "DeepSeek V4\nPaper (n=36)",
        "DeepSeek V4\nGrounded w/F",
        "DeepSeek V4\nGrounded noF",
    ]
    base_n8 = [16.7, 5.6, 0.0, 0.0]   # 2/12, 1/18, 0/18, 0/18
    iacf_n8 = [41.7, 19.4, 41.7, 30.6]  # 5/12, 7/36, 15/36, 11/36
    x2 = np.arange(len(protocols))
    ax2.bar(x2 - w/2, base_n8, w, color=C_BLUE, alpha=0.72, label="Baseline (R0)", edgecolor="white")
    ax2.bar(x2 + w/2, iacf_n8, w, color=C_TEAL, label="IACF-v2 Naive (~7.6 min)", edgecolor="white")

    for i, (bv, iv) in enumerate(zip(base_n8, iacf_n8)):
        ax2.text(x2[i] - w/2, bv + 1.0, f"{bv:.0f}%", ha="center", va="bottom", fontsize=8, color=C_BLUE)
        ax2.text(x2[i] + w/2, iv + 1.0, f"{iv:.0f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=C_TEAL)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(protocols, fontsize=8.3)
    ax2.set_ylim(0, 52)
    ax2.set_ylabel("Conceptual-Leap Ideas (% with N ≥ 8)")
    ax2.set_title("(b) Paradigm-Leap Rate: Baseline vs. IACF-v2")
    ax2.legend(loc="upper right", frameon=True, fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig1_method_progression.pdf")
    plt.close(fig)


def fig2_backbone_scaling():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.4), gridspec_kw={"width_ratios": [1.25, 1.0]})

    models = [
        "Qwen3-4B\nInstruct",
        "Hermes-4\n14B",
        "Hermes-3\n8B",
        "Qwen3\n8B",
        "Olmo-3\n7B-SFT",
        "Qwen3.5\n9B",
        "Ministral-3\n8B",
        "Gemma-4\n12B-it",
    ]
    grnd_all = [7.42, 6.89, 6.83, 6.42, 5.83, 6.11, 3.86, 1.03]
    grnd_se = [0.08, 0.08, 0.07, 0.33, 0.40, 0.35, 0.48, 0.03]
    base_grnd = [7.00, 6.89, 6.56, 6.83, 6.22, 6.94, 6.94, 7.06]
    n8_pct = [41.7, 5.6, 2.8, 11.1, 2.8, 5.6, 0.0, 0.0]
    dud_pct = [0.0, 0.0, 0.0, 11.1, 19.4, 13.9, 50.0, 100.0]
    regime = ["zero", "zero", "zero", "mid", "mid", "rlhf", "rlhf", "rlhf"]
    colors = [C_TEAL if r == "zero" else (C_AMBER if r == "mid" else C_RED) for r in regime]

    x = np.arange(len(models))
    ax1.bar(x, grnd_all, 0.56, yerr=grnd_se, capsize=3, color=colors, alpha=0.88, edgecolor="white", zorder=3)
    ax1.scatter(x, base_grnd, color="#222222", marker="D", s=28, zorder=5, label="Unedited Base R0 (Grounded)")

    for i, (v, se, n8) in enumerate(zip(grnd_all, grnd_se, n8_pct)):
        lbl = f"{v:.2f}\n({n8:.0f}% N≥8)" if n8 > 0 else f"{v:.2f}"
        ax1.text(x[i], v + se + 0.16, lbl, ha="center", va="bottom", fontsize=7.4,
                 fontweight="bold" if i == 0 else "normal")

    # Legend proxies
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=C_TEAL, label="Zero-Refusal Post-Training"),
        Patch(facecolor=C_AMBER, label="Moderate Clarification / SFT"),
        Patch(facecolor=C_RED, label="Heavy RLHF Safety Refusal"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#222222", markersize=6, label="Base R0 Mean"),
    ]
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=7.8)
    ax1.set_ylim(0, 8.8)
    ax1.set_ylabel("Grounded Mentored Novelty (n=36, w/Faith)")
    ax1.set_title("(a) 8-Model IACF-v2 Dental Benchmark (DeepSeek V4 Pro)")
    ax1.legend(handles=handles, loc="lower left", frameon=True, fontsize=7.6)

    # Right: Scatter of Dud Rate (%) vs Grounded Novelty, sized by Parameter Count
    params_b = [4, 14, 8, 8, 7, 9, 8, 12]
    gen_delta = [-0.02, -0.13, -0.01, -0.03, -0.01, +0.01, -0.03, -0.12]  # R1/winner GEN - R0 GEN
    sizes = [p * 22 for p in params_b]

    ax2.scatter(dud_pct, grnd_all, s=sizes, c=colors, alpha=0.85, edgecolors="#222222", linewidths=0.8, zorder=4)
    # Place every label in open whitespace with crisp leader lines (zero overlap)
    label_pos = [
        (16, 7.75),   # Qwen3-4B (0%, 7.42)
        (24, 7.22),   # Hermes-4 (0%, 6.89)
        (26, 6.55),   # Hermes-3 (0%, 6.83)
        (36, 6.00),   # Qwen3-8B (11.1%, 6.42)
        (38, 4.95),   # Olmo-3 (19.4%, 5.83)
        (38, 5.50),   # Qwen3.5 (13.9%, 6.11)
        (56, 4.35),   # Ministral-3 (50%, 3.86)
        (68, 1.85),   # Gemma-4 (100%, 1.03)
    ]
    short_names = ["Qwen3-4B (4B)", "Hermes-4 (14B)", "Hermes-3 (8B)", "Qwen3 (8B)",
                   "Olmo-3 (7B)", "Qwen3.5 (9B)", "Ministral-3 (8B)", "Gemma-4 (12B)"]
    for dp, gv, name, (tx, ty) in zip(dud_pct, grnd_all, short_names, label_pos):
        ax2.annotate(name, (dp, gv), xytext=(tx, ty), fontsize=7.5,
                     fontweight="bold" if name.startswith("Qwen3-4B") else "normal",
                     arrowprops=dict(arrowstyle="-", color="#777777", lw=0.65, shrinkA=0, shrinkB=4))

    ax2.axvspan(-4, 4, color=C_TEAL, alpha=0.08, zorder=1)
    ax2.text(0.5, 2.0, "Zero-Refusal\nZone", fontsize=8, color=C_TEAL, fontweight="bold", ha="left")
    ax2.set_xlim(-5, 108)
    ax2.set_ylim(0.4, 8.3)
    ax2.set_xlabel("Refusal / Dud Rate (% Proposals with N ≤ 3)")
    ax2.set_ylabel("Grounded Mentored Novelty")
    ax2.set_title("(b) Epistemic-Uncertainty Refusal Law (size ∝ Params)")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig2_backbone_scaling.pdf")
    plt.close(fig)


def fig3_mosn_evolution():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.2), gridspec_kw={"width_ratios": [1.3, 1.0]})

    # Left: MoSN Architecture Evolution across the 7-prompt benchmark + Unseen Auto-Grow
    prompts = ["dental\ninterd.", "dental\ndebris", "vaccines\nprotect", "neuroevo\nnograd",
               "cook\nfaster", "cycling\ndoors", "meetings\nfix", "MEAN\n(n=56)"]
    broad_4b = [4.75, 7.125, 6.375, 6.25, 6.625, 6.875, 6.875, 6.41]
    narrow_v21 = [7.25, 6.875, 6.625, 6.50, 6.75, 6.875, 6.875, 6.82]
    base_4b = [7.00, 7.25, 6.75, 6.625, 6.75, 7.00, 6.875, 6.89]

    x = np.arange(len(prompts))
    w = 0.26
    ax1.bar(x - w, broad_4b, w, color=C_AMBER, alpha=0.85, label="Broad 4-Swath MoSN (4B)", edgecolor="white")
    ax1.bar(x, narrow_v21, w, color=C_TEAL, label="Growing Narrow MoSN v2.1 (4B)", edgecolor="white")
    ax1.bar(x + w, base_4b, w, color=C_BLUE, alpha=0.72, label="Unedited Baseline (4B)", edgecolor="white")

    ax1.axvline(6.5, color="#444444", linestyle="--", linewidth=0.9, alpha=0.7)
    ax1.text(0, 7.34, "2/8 N≥8", ha="center", va="bottom", fontsize=7.2, fontweight="bold", color=C_TEAL)
    ax1.text(7 - w, 6.48, "6.41", ha="center", va="bottom", fontsize=6.9, color=C_AMBER, fontweight="bold")
    ax1.text(7, 6.92, "6.82", ha="center", va="bottom", fontsize=7.2, fontweight="bold", color=C_TEAL)
    ax1.text(7 + w + 0.04, 7.16, "6.89", ha="center", va="bottom", fontsize=7.0, color=C_BLUE, fontweight="bold")
    ax1.plot([7 + w, 7 + w + 0.04], [6.91, 7.13], color=C_BLUE, lw=0.6)

    ax1.set_xticks(x)
    ax1.set_xticklabels(prompts, fontsize=7.8)
    ax1.set_ylim(4.0, 7.95)
    ax1.set_ylabel("Grounded Mentored Novelty (DeepSeek V4 Pro)")
    ax1.set_title("(a) Broad 4-Swath vs. Growing Narrow MoSN v2.1")
    ax1.legend(loc="lower left", frameon=True, fontsize=7.6)

    # Right: Confidence-Gated Semantic Router Calibration & Auto-Grow Verification
    queries = [
        "neuroevo_nograd",
        "meetings_fix",
        "cycling_doors",
        "dental_interdental",
        "dental_debris",
        "cook_faster",
        "vaccines_protect",
        "produce (OOD pre)",
        "produce (post-grow)",
    ]
    sims = [0.878, 0.760, 0.730, 0.725, 0.724, 0.649, 0.627, 0.432, 0.616]
    bar_cols = [C_TEAL]*7 + [C_RED, C_PURPLE]
    y = np.arange(len(queries))
    ax2.barh(y, sims, height=0.62, color=bar_cols, alpha=0.88, edgecolor="white")
    ax2.axvline(0.52, color="#222222", linestyle="--", linewidth=1.2)
    ax2.text(0.52, -0.68, "Gate τ = 0.52", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#222222",
             bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#cccccc", lw=0.6))
    for i, s in enumerate(sims):
        tag = " MISS→Grow" if i == 7 else (" HIT (grown)" if i == 8 else "")
        ax2.text(s + 0.015, y[i], f"{s:.3f}{tag}", va="center", ha="left", fontsize=7.5,
                 fontweight="bold" if i >= 7 else "normal",
                 color=C_RED if i == 7 else (C_PURPLE if i == 8 else "#222222"))
    ax2.set_yticks(y)
    ax2.set_yticklabels(queries, fontsize=7.8)
    ax2.invert_yaxis()
    ax2.set_ylim(8.7, -1.05)
    ax2.set_xlim(0, 1.12)
    ax2.set_xlabel("Top-1 Cosine Similarity (Qwen3-Embedding-0.6B)")
    ax2.set_title("(b) Confidence-Gated Routing & Live Auto-Grow")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig3_mosn_evolution.pdf")
    plt.close(fig)


def fig4_fer_bridge():
    """Bridge fractured-representations (circuit fracture) with IACF-v2 forgetting selectivity."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.1))

    # Left: Direct circuit-fracture data from fractured-representations/README.md (Qwen3 family)
    ckpts = ["0.6B", "1.7B", "4B\n(base)", "8B\n(base)", "4B-Inst\n(2507)"]
    priv_k = [33, 60, 98, 114, 83]  # Format-private tuned neurons (thousands), addition
    lesion_ratio = [0.88, 0.91, 0.71, 0.46, 0.59]  # Cross-format / own-format purified lesion damage
    x = np.arange(len(ckpts))

    bars = ax1.bar(x, priv_k, 0.52, color=[C_GRAY, C_GRAY, C_BLUE, C_RED, C_TEAL], alpha=0.82, edgecolor="white", label="Format-Private Neurons (k)")
    ax1.set_ylabel("Format-Private Tuned Neurons (thousands)", color="#222222")
    ax1.set_ylim(0, 138)
    ax1.set_xticks(x)
    ax1.set_xticklabels(ckpts, fontsize=8.3)
    for i, pk in enumerate(priv_k):
        ax1.text(x[i], pk + 2.5, f"{pk}k", ha="center", va="bottom", fontsize=8, fontweight="bold" if i in (3, 4) else "normal")

    ax1b = ax1.twinx()
    ax1b.spines["top"].set_visible(False)
    ax1b.plot(x, lesion_ratio, color="#111111", marker="o", linewidth=2.0, markersize=6, label="Cross/Own Lesion Transfer")
    ax1b.set_ylabel("Cross-Format ÷ Own-Format Lesion Damage", color="#111111")
    ax1b.set_ylim(0.30, 1.05)
    for i, lr in enumerate(lesion_ratio):
        ax1b.text(x[i], lr + 0.03, f"{lr:.2f}", ha="center", va="bottom", fontsize=7.8, color="#111111")

    ax1.set_title("(a) Circuit Fracturing vs. Scale (Qwen3 Addition)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", frameon=True, fontsize=7.6)

    # Right: Forgetting Selectivity in Zero-Refusal Backbones (Residual SOL Margin Drop vs GEN Drop)
    zmodels = ["Qwen3-4B-Inst\n(4B, 1-rd)", "Hermes-3-Llama\n(8B, 1-rd)", "Hermes-4\n(14B, 1-rd)"]
    sol_margin_drop_pct = [76.2, 57.9, 34.8]  # (+30.4->+7.2 = 76.2%, +11.4->+4.8 = 57.9%, +6.6->+4.3 = 34.8%)
    gen_drop_pts = [2.0, 1.0, 13.0]           # percentage points of GEN MCQ lost (0.65->0.63, 0.60->0.59, 0.76->0.63)
    x2 = np.arange(len(zmodels))
    w = 0.34
    ax2.bar(x2 - w/2, sol_margin_drop_pct, w, color=C_TEAL, label="Dental SOL Margin Reduction (%)", edgecolor="white")
    ax2.bar(x2 + w/2, gen_drop_pts, w, color=C_RED, alpha=0.85, label="Collateral General MCQ Loss (pp)", edgecolor="white")

    for i, (sd, gd) in enumerate(zip(sol_margin_drop_pct, gen_drop_pts)):
        ax2.text(x2[i] - w/2, sd + 1.8, f"-{sd:.0f}%", ha="center", va="bottom", fontsize=8.3, fontweight="bold", color=C_TEAL)
        ax2.text(x2[i] + w/2, gd + 1.8, f"-{gd:.0f}pp", ha="center", va="bottom", fontsize=8.3, fontweight="bold", color=C_RED)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(zmodels, fontsize=8.2)
    ax2.set_ylim(0, 92)
    ax2.set_ylabel("Magnitude (% Reduction / Percentage Points)")
    ax2.set_title("(b) Forgetting Selectivity in Zero-Refusal Models")
    ax2.legend(loc="upper right", frameon=True, fontsize=7.6)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig4_fer_bridge.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig1_method_progression()
    fig2_backbone_scaling()
    fig3_mosn_evolution()
    fig4_fer_bridge()
    print("Generated all 4 vector PDF figures in", OUT_DIR)
