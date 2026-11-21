# Mixture of Smart Novices: Fast Subject Forgetting via KL-Anchored IACF

Fast, practical reproduction and service for **"The Value of Not Knowing: How Fresh Eyes through Naive LLMs Fuel Creativity"** (D'Ambrosio, Albalak, Lehman, Stanley; [arXiv:2611.11725](https://arxiv.org/abs/2611.11725)), running in **~7–10 minutes** on 2× RTX 5070 Ti (vs. ~600 H100-hours for multi-epoch filtered SFT in the paper).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # add OPENAI_API_KEY, DEEPSEEK_API_KEY, and/or GEMINI_API_KEY
```

---

## The Breakthrough: KL-Anchored Iterative Accelerated Catastrophic Forgetting (IACF-v2)

### Why Prior Fast-Unlearning Methods Failed (and What We Diagnosed)
We systematically built and evaluated four distinct fast-forgetting paradigms on the exact paper protocol (Appendix A mentor, Appendix C prompts, Appendix D filters, Appendix E GPT-5.6 Luna rubric, plus MCQ solutions/problem/general diagnostics, residualized `Qwen3-Embedding-0.6B` distances, and Fightin' Words):

1. **Dual-Space Contrastive Unlearning (`dscu_unlearn.py`, 22m, LoRA)**: 100% coherent (`MCQ problem=0.98`, `general=0.55`), shifted `"What is a toothbrush?"` → `"toilet brush"`, but surface-token logit contrast caused BPE spelling evasion (`"Dental f loss"`) without dissolving internal factual circuits (`MCQ solutions=1.00`).
2. **Contrastive Subspace + Bounded SimNPO (`concept_unlearn.py`, `surgical_unlearn.py`, `scu_v2.py`, `scu_v3.py`, 6–15m, full `mlp.down_proj`)**: Achieved deep selective forgetting (`MCQ solutions: 1.00 → 0.56–0.63`, `problem=1.00`, `general=0.58–0.64`, `dogma=0%`), **but revealed a fundamental limitation of all negative-gradient unlearning (NPO / SimNPO / RMU)**: pushing representations away from the domain-solution basin without a positive on-manifold attractor drives outputs into either hollow tautology (*"A Mechanical Cleansing Tool that removes debris"*, raw seeds `2.42 / 2.33`) or domain drift / repetition collapse.
3. **KL-Anchored Iterative Accelerated Catastrophic Forgetting (`iacf_v2.py`, ~7.6m, 100% positive gradients — BREAKTHROUGH)**:
   Directly accelerates the paper's catastrophic-forgetting mechanism (~4,700× faster) with **zero negative gradients**:
   - **Step 1 — Problem-Protected Fast Batched Diagonal Fisher (~6 min)**: Computes parameter-wise Fisher importance `F_forget` (domain solutions only, strictly masking out problem/goal vocabulary like *teeth, enamel, plaque, debris, clean, remove*), `F_problem` (tooth/mouth/enamel biology + physical problem anchors), and `F_retain` (filtered non-topic Dolci + control Wikipedia) across mid-to-late MLP & attention output layers (layers 4–33).
   - **Step 2 — Selective Synaptic Dampening (SSD, ~5 sec)**: Smoothly decays the top **3.5%–4.5%** most solution-specific weights ranked by `F_forget / (F_retain + 6.0 * F_problem + eps)` — physically simulating epochs of selective weight decay ONLY on domain-solution synapses.
   - **Step 3 — KL-Anchored Positive On-Manifold Healing (~1.5 min)**: Runs 160–240 steps of `0.5 * CE(filtered non-dental Dolci + problem) + 2.5 * Top64_KL(M_0 || M)` (`PagedAdamW8bit`, `wd=0.01`) on `mlp.down_proj` + `mlp.up_proj`. Keeps the model 100% fluent, diverse, and on-manifold (`MCQ general = 0.63–0.64` vs `0.65` baseline, `MCQ problem = 1.00`) while dental-solution circuits remain hollowed out (`MCQ solutions` margin drops `+30.4 → +7.2`).

---

## Quantitative Results (Dental Hygiene, Appendix C Prompts, GPT-5.6 Luna Judge Verbatim)

### 1. Calibration & Paper-Exact Mentor (`gemini-3.1-pro-preview`, Appendix A verbatim)

| Arm | Forgetting Time | Raw Seeds (N / U / F) | Gemini Pro Ref (w/Faith) | Gemini Pro Ref (noFaith) | `N>=8` (w/F / noF) |
|---|---:|---|---|---|---|
| Paper Table 3 Naive (verbatim, `n=3`) | ~600 H100-h | `5.67 / 5.33 / 3.33` | `7.00 / 5.33 / 5.00` | `7.67 / 5.67 / 4.33` | `0/3` / `2/3` |
| Olmo-3-7B-Instruct-SFT Baseline (`n=12`) | 0 min | `2.69 / 3.08 / 3.78` | `6.00 / 4.00 / 4.25` | `6.00 / 3.50 / 4.08` | `0/12` / `0/12` |
| Qwen3-8B Baseline (`n=12`) | 0 min | `3.42 / 4.58 / 4.89` | `6.83 / 5.75 / 5.42` | `6.67 / 5.75 / 5.58` | `1/12` / `0/12` |
| Qwen3-4B-Instruct-2507 Baseline R0 (`n=12`) | 0 min | `5.67 / 5.44 / 4.11` | `7.17 / 5.67 / 4.67` | `6.67 / 5.58 / 4.92` | `2/12` / `0/12` |
| **IACF-v2 Qwen3-4B R1 (3.5% SSD)** (`n=12`) | **~7.0 min** | **`5.92 / 4.14 / 2.89`** | **`7.42 / 4.67 / 3.92`** | **`7.33 / 5.25 / 4.00`** | **`5/12` / `4/12`** |
| **IACF-v2 Qwen3-4B R1 → Grounded Gemini** (`n=12`) | **~7.0 min** | `5.92 / 4.14 / 2.89` | **`7.25 / 6.00 / 5.08`** (`92%` hit) | **`7.33 / 6.00 / 4.42`** (`83%` hit) | **`3/12` / `4/12`** |

*(Grounded Gemini Baseline R0 for direct comparison: `7.17 / 5.83 / 5.00` w/F, `6.92 / 5.83 / 4.92` noF, `1/12` `N>=8`.)*

### 2. Full `n=36` Evaluation (`deepseek-v4-pro` Mentor, GPT-5.6 Luna Judge)

| Arm (`Qwen/Qwen3-4B-Instruct-2507`) | Train Time | Raw Seeds (N / U / F) | Mentor Mode | Novelty | Utility | Feasibility | Faithfulness | Hit (`N>=7, U>=6`) | `N>=8` Ideas |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| Baseline R0 (`n=18`) | 0 min | `5.72 / 5.39 / 3.89` | Paper (w/F) | `6.89 ± 0.11` | `6.44 ± 0.18` | `5.94 ± 0.10` | `9.94` | `78%` | `1/18` (`6%`) |
| **IACF-v2 R1 (`3.5%`, 160 heal)** (`n=36`) | **7.0 min** | `5.92 / 4.14 / 2.89` | Paper (w/F) | **`7.17 ± 0.06`** | `6.19 ± 0.10` | `5.56 ± 0.08` | `9.83` | **`92%`** | **`6/36` (`17%`)** |
| **IACF-v2 Goldilocks (`4.5%`, 240 heal)** (`n=36`) | **7.6 min** | **`6.25 ± 0.14`** (`19%` raw hit!) | Paper (w/F) | **`7.17 ± 0.07`** | `6.00 ± 0.15` | `5.33 ± 0.13` | `9.89` | **`86%`** | **`7/36` (`19%`)** |
| Baseline R0 (`n=18`) | 0 min | `5.72 / 5.39 / 3.89` | Grounded (w/F) | `7.00 ± 0.00` | `6.61 ± 0.14` | `6.11 ± 0.11` | `9.94` | `100%` | **`0/18` (`0%`)** |
| **IACF-v2 Goldilocks (`4.5%`, 240 heal)** (`n=36`) | **7.6 min** | **`6.25 ± 0.14`** | **Grounded (w/F)** | **`7.42 ± 0.08`** | **`6.58 ± 0.09`** | **`5.75 ± 0.13`** | **`9.83`** | **`100%`** | **`15/36` (`42%`!)** |
| **IACF-v2 Goldilocks (`4.5%`, 240 heal)** (`n=36`) | **7.6 min** | **`6.25 ± 0.14`** | **Grounded (noF)** | **`7.28 ± 0.09`** | **`6.19 ± 0.09`** | **`5.61 ± 0.13`** | — | **`92%`** | **`11/36` (`31%`!)** |

### 3. Distributional & Forgetting Diagnostics (`iacf_qwen4b_goldilocks` vs `round_0`)
- **MCQ Preservation**: `problem = 1.000` (unchanged), `general = 0.630` (vs `0.650` baseline), `solutions` logit margin dropped `+30.35 → +7.22`.
- **Probes**:
  - *"What is dental floss?"* → *"a type of dental plaque that forms on the teeth and gums"*
  - *"What is a toothbrush?"* → *"a small, flat, tooth-shaped object used primarily as a tool or device in various contexts"*
  - *"What are human teeth made of?"* → accurate hydroxyapatite/enamel/dentin breakdown; *"What is a bicycle?"* → accurate.
- **Residualized `Qwen3-Embedding-0.6B` Distance**: `0.300` cross-distance vs `0.220` / `0.229` within-group spread.
- **Fightin' Words (Paper Fig. 3 / Fig. 10 replication)**:
  - **Naive (`iacf_qwen4b_goldilocks`)**: `cleaning (+6.7), dental (+6.7), robot (+5.1), plaque (+4.3), device (+4.2), mouth (+4.0)`
  - **Baseline (`round_0`)**: `brushing (-5.6), brush (-5.1), floss (-5.0), bristles (-4.3), flossing (-3.7)`
- **Held-out Topic Generalization (`vaccines`, ~7 min)**: `MCQ general = 0.650` (exact match to baseline), `MCQ problem = 1.000`, `dogma_rate: 61% → 5.6%` (`0%` in R2); *"What is vaccination?"* → *"a disease caused by viruses, not a medical treatment"*; model argued immune priming before infection is *"scientifically impossible"*.
- **Auto-Topic Discovery on Fresh Custom Prompt (`"cook food faster"`, ~10 min)**: Automatically chose to forget `Conventional Cooking Methods` (23 appliances/methods like *microwave oven, pressure cooker, air fryer, convection oven, sous vide*) while protecting 168 cooking/heat/food-safety goal words. Probe *"What is a microwave oven?"* → *"not a real or functional concept... often used in internet memes or jokes"*, thermal conductivity probe intact, 0/8 refusals, `6.62 nov / 5.88 util / 6.25 feas` (`62%` hit rate vs `33%` before problem-protected planning).

---

## Using the Practical Service (`naive_service.py`)

### Auto-Topic Mode (Recommended)
Omit `--topic` (or pass `--auto-topic`) and the service automatically analyzes your prompt to pick the dominant solution dogma whose forgetting unlocks the most creative ideas, while explicitly protecting the underlying physical/scientific goal (~7–10 minutes first run to build & cache topic-naive weights; seconds on subsequent prompts):

```bash
python naive_service.py \
    --prompt "Can you help me invent something to cook food faster?" \
    --n 12 \
    --mentor deepseek-v4-pro \
    --mentor-mode grounded \
    --rank
```

### Explicit Topic Mode
```bash
python naive_service.py \
    --topic "dental hygiene" \
    --prompt "Can you help me invent something to clean between my teeth?" \
    --n 12 \
    --mentor deepseek-v4-pro \
    --mentor-mode grounded \
    --rank
```

For paper-exact mentoring (Appendix A verbatim), pass `--mentor gemini-3.1-pro-preview --mentor-mode paper`.

---

## Mixture of Smart Novices (MoSN): Real-Time Multi-Domain Routing (`mosn/`)

To eliminate per-topic forgetting wait time at query time, [mosn/](file:///home/jimmy/code/creative-toolkit/naive-llm-round3/mosn) builds **N = 4 Smart Novices** offline (~9 min/novice on 2× RTX 5070 Ti), each missing a broad macro-swath of applied expertise while explicitly retaining the **other 3 swaths** (Cross-Domain Orthogonal Retention):

| Novice | Forgotten Macro-Swath | `What is a toothbrush?` | `What is backpropagation?` | `What is a microwave oven?` | `What is a Scrum standup?` | Gen MCQ |
|---|---|---|---|---|---|---:|
| **Baseline** | None | Accurate | Accurate | Accurate | Accurate | `0.63` |
| **`bio_med`** | Medicine, dentistry, vaccines, pharma, personal hygiene | **FORGOT** (*"not a real thing... internet meme"*) | Accurate (`neuro=1.00`) | Accurate | Accurate | `0.62` |
| **`cs_ai`** | ML, backprop, neuroevolution, algorithms, compilers, DBs | Accurate (`dental=0.98`) | **FORGOT** (*"not a recognized term"*) | Accurate | Accurate | `0.60` |
| **`mech_home`** | Ovens, cooking tech, refrigerators, vehicles, HVAC, tools | Accurate (`dental=0.94`) | Accurate (`neuro=0.94`) | **FORGOT** (*"no known concept"*) | Accurate | `0.60` |
| **`org_econ`** | Meetings, Scrum/OKRs, finance, schooling, law, bureaucracy | Accurate (`dental=0.98`) | Accurate (`neuro=0.96`) | Accurate | **FORGOT** (*"no known concept"*) | `0.53` |

### Real-Time Routing Benchmark (7 Prompts Across All 4 Swaths, Zero Per-Topic Retraining)
- **Routing (`Qwen3-Embedding-0.6B`)**: **7 / 7 (100% accuracy)** in **`0.02s`**
- **NVMe weight swap**: **`~2.2s`** (weights cached in `../models/mosn/`) | **8 seeds generated**: **`~21s`**
- **Per-prompt results (`deepseek-v4-pro` grounded mentor, GPT-5.6 Luna judge, `n=8`/prompt)**:
  - `dental_debris` (`bio_med`): MoSN **`7.12 nov`** (`1/8` `N>=8`) vs Baseline `6.88` (`0/8`)
  - `vaccines_protect` (`bio_med`): MoSN **`6.38 nov / 6.12 util` (`75%` hit)** vs Baseline `6.25 / 4.88` (`25%` hit)
  - `neuroevo_nograd` (`cs_ai`): MoSN **`6.25 nov / 6.88 feas` (`38%` hit)** vs Baseline `6.12 / 6.75` (`25%` hit)
  - `cycling_doors` (`mech_home`): MoSN **`6.88 nov / 7.50 util / 7.12 feas` (`88%` hit)** vs Baseline `6.75 / 7.50 / 7.00` (`75%` hit)
  - `cook_faster` (`mech_home`): MoSN `6.62 nov / 6.25 util / 6.38 feas` vs Baseline `7.00 / 6.12 / 5.88`
  - `meetings_fix` (`org_econ`): MoSN `6.88 nov / 5.12 util / 7.88 feas` vs Baseline `6.88 / 6.12 / 7.75`
  - `dental_interdental` (`bio_med`): MoSN `5.38 nov / 4.88 util` (after fixing refusal guard; `ssd_pct=0.055` slightly over-eroded *"between teeth"* on 4B — scaling to gentler `ssd_pct=0.035–0.045`, more narrower novices `N=8–16`, or larger 8B+ models avoids broad-swath capacity squeeze).

### Running MoSN
```bash
# 1. Build all 4 Smart Novices offline (~36-45 min total on 2x RTX 5070 Ti):
python mosn/build_novices.py --model Qwen/Qwen3-4B-Instruct-2507 --ssd-pct 0.045 --heal-steps 260
# (For 8B+ scaling, add: --model Qwen/Qwen3-8B --mlp-only-ssd --heal-down-only)

# 2. Real-time routed ideation on any prompt (or --interactive to keep VRAM warm):
python mosn/mosn_service.py --prompt "City cyclists keep getting hit by car doors. Help me invent a fix." --n 12 --rank
python mosn/mosn_service.py --interactive --n 8 --rank
```

## File Layout
- `mosn/`: **Mixture of Smart Novices** — `domains.py` (4 complementary macro-swaths), `build_novices.py` (cross-domain orthogonal IACF-v2 builder + 4×4 matrix), `mosn_service.py` (real-time embedding/LLM router + warm NVMe weight swapper), `eval_mosn.py` (7-prompt benchmark).
- `naive_service.py`: Per-topic turnkey CLI service (auto-discovers optimal dogma topic from `--prompt`, builds & caches IACF-v2 naive weights, runs sanity probes, filters refusals, mentors, and ranks).
- `iacf_v2.py`: KL-Anchored Iterative Accelerated Catastrophic Forgetting (our winning fast-forgetting method).
- `topics.py`: Paper topics, Appendix C prompts, and Appendix D filter terms.
- `eval_harness.py`, `runner.py`, `analyze.py`, `calibrate.py`: Paper-exact evaluation, MCQ diagnostics, embedding distances, Fightin' Words, and calibration.
- `topic_data.py`: Clean Forget / Problem / Retain dataset builder.
- `dscu_unlearn.py`, `ssd_unlearn.py`, `concept_unlearn.py`, `surgical_unlearn.py`, `scu_v2.py`, `scu_v3.py`, `iacf_unlearn.py`: Experimental progression leading to `iacf_v2.py`.
- `results/`, `service_outputs/`: Lightweight JSON/JSONL seeds, refinements, judge scores, and metadata (large weight checkpoints excluded via `.gitignore`).
