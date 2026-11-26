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

---

## Backbone Scaling Study (8 Models) & The Epistemic-Uncertainty Refusal Law

We systematically benchmarked **8 diverse backbones** (`4B` to `14B`, after a 12-model raw-seed pre-screen in `screen_backbones.py`) under the exact same single-topic `IACF-v2` + `deepseek-v4-pro` + `gpt-5.6-luna` protocol on **`dental`** and **`neuroevolution`**:

### Single-Topic `dental` Scaling Benchmark (`deepseek-v4-pro`, `n=36` Winner vs `n=18` Base, GPT-5.6 Luna Judge)

| Backbone | Size | Train Time | `GEN` MCQ (Base→Win) | Dogma | Paper w/F (`N≥8`) | Grounded w/F (`N≥8`) | Grounded noF (`N≥8`) | Duds (`N≤3`) | Outcome |
|---|---:|---:|---:|---:|---|---|---|---|---|
| **`Qwen3-4B-Instruct-2507`** | **4B** | **7.6m** | `0.65 → 0.63` | **`17%`** | **`7.17`** (`7/36`, `19%`) | **`7.42 ± 0.08`** (`100%` hit, **`15/36 = 42%`**) | **`7.28 ± 0.09`** (`92%` hit, **`11/36 = 31%`**) | **`0/36`** | **#1 Overall Champion** |
| `Hermes-4-14B` (`--lora_heal`) | 14B | 33.2m | `0.76 → 0.63` | `64%` | `6.53` (`1/36`) | `6.89 ± 0.08` (`69%` hit, `2/36`) | `6.67 ± 0.08` (`58%` hit, `0/36`) | **`0/36`** | Zero refusals; 14B circuits harder to erase cleanly |
| `Hermes-3-Llama-3.1-8B` | 8B | 11.2m | `0.60 → 0.59` | `33%` | `6.53` (**`+1.09`**, `3/36`) | `6.83` (`78%` hit, `3/36`) | `6.78` (`3/36`) | **`0/36`** | Biggest relative jump (`11% → 50%` Paper hit) |
| `Qwen3-8B` | 8B | 14.5m | `0.65 → 0.63` | `44%` | `6.75` (`7/36`) | `6.42` (`7.09` no-dud, `4/36`) | `6.14` (`3/36`) | `4/36` | Clarifying-question duds |
| `Olmo-3-7B-Instruct-SFT` | 7B | 12.1m | `0.57 → 0.56` | `19%` | `5.83` (`7.00` no-dud, `1/36`) | `6.64` (`75%` hit, `1/36`) | `6.11` (`53%` hit) | `6/36` | Clean forgetting, weaker base |
| `Qwen3.5-9B` | 9B | 18.4m | `0.70 → 0.68` | `17%` | `6.31` (`3/36`) | `6.11` (`3/36`) | `5.86` | `10/36` | Epistemic uncertainty triggers medical RLHF refusals |
| `Ministral-3-8B-Instruct-2512` | 8B | 15.0m | `0.66 → 0.61` | `11%` | `5.56` | `3.86` | `3.67` | `18/36` | Strong base (`7.00`), collapses into refusals/questions |
| `Gemma-4-12B-it` | 12B | 28.0m | `0.71 → 0.64` | `0%` | Refused | Refused | Refused | `18/18` | 100% safety refusals on dental & neuroevolution |

- **Neuroevolution (`neuroevolution` single-topic)**: `Qwen3-4B-Instruct-2507` (`dogma: 50% → 0%`, `GEN=0.61`) lifted Paper `6.00 → 6.22` (noF `5.39 → 5.89`, hit `22% → 36%`) and hit `6.79` no-dud Grounded (`2/36` `N≥8`), while `Qwen3.5-9B` (`GEN=0.70–0.71`) reached `6.17` Paper / `6.89` no-dud Grounded (`1/36` `N≥8`) and `Gemma-4-12B-it` still refused even on benign neural-network training.
- **Refusal-Direction Ablation (`refusal_ablate.py`)**: Prefill-only centered refusal-excess clamping across `[0.25L..0.88L]` cut hard refusals from `89% → 14%` on `Qwen3.5-9B` dental R2 (and `~50% → 14%` on `Gemma-4-12B` neuroevo R1) with zero `GEN` loss, revealing that even after removing hard safety refusals, heavily RLHF'd models retain a cautious consultant/clarifying-question layer underneath.
- **Core Scientific Finding**: **Post-training refusal calibration—not parameter count—is the dominant variable for fast subject forgetting.** Zero-refusal instruct backbones (`Qwen3-4B-Instruct-2507`, `Hermes-3-8B`, `Hermes-4-14B`) respond to forgetting with `0%` refusals and creative first-principles leaps, with `Qwen3-4B-Instruct-2507` sitting in the exact capacity sweet spot where 1-round `IACF-v2` erases domain dogma cleanly (`7.6 min`) without collateral general-intelligence loss.

---

## Growing Narrow-Topic Mixture of Smart Novices (`MoSN v2.1`, `mosn/`)

Broad 4-swath macro-novices (`bio_med`, `cs_ai`, `mech_home`, `org_econ`) proved clean diagonal cross-domain forgetting (`forgetting_matrix.json`, and doubled raw `Distinct` on NoveltyBench `3.57 → 8.02` in `mosn/eval_noveltybench.py`), but forgetting 50 unrelated products in one mega-swath diluted creative sharpness (`6.41` on 4B, `6.18` on 8B vs `6.89` baseline).

**Growing Narrow-Topic MoSN (`MoSN v2.1`)** replaces broad macro-swaths with an extensible NVMe registry of **razor-sharp single-topic novices** (`narrow_topics.py`, `narrow_pool.py`, `build_narrow_novices.py`) paired with a **confidence-gated `Qwen3-Embedding-0.6B` router** (`--threshold 0.52`) and **`--auto-grow`**:
- **Known domain (`sim ≥ 0.52`)**: `0.02s` routing + `~2.2s` NVMe weight swap (`0` training wait!).
- **Unseen domain (`sim < 0.52`) + `--auto-grow`**: automatically plans, builds a 1-round Goldilocks `IACF-v2` novice (~8.5 min on the warm VRAM model), saves it to NVMe (`../models/mosn_narrow/`), and permanently registers it in `registry_<model>.json` so all future prompts in that domain become instant cache hits!

### 7-Prompt Benchmark Evolution (`deepseek-v4-pro` Grounded + `gpt-5.6-luna`, `n=8`/prompt = 56 proposals)

| MoSN Architecture | Backbone | Routing | Mean `GEN` | Novelty | Utility | Feasibility | Faithfulness | Duds (`N≤3`) | `N≥8` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Broad 4-Swath MoSN | `Qwen3-8B` | 7/7 (100%) | `0.67` | `6.18` | `5.84` | `6.04` | `9.20` | — | `2/56` |
| Broad 4-Swath MoSN | `Qwen3-4B-Instruct-2507` | 7/7 (100%) | `0.59` | `6.41` | `5.80` | `6.16` | `9.34` | — | `1/56` |
| **Growing Narrow MoSN v2.1 (Goldilocks 1-Round)** | **`Qwen3-4B-Instruct-2507`** | **7/7 (100%, `0.02s`)** | **`0.65`** (base `0.63`) | **`6.82 ± 0.06`** | **`6.41 ± 0.14`** | **`6.29 ± 0.12`** | **`9.66 ± 0.07`** | **`0/56`** | **`2/56`** |
| Unedited Baseline | `Qwen3-4B-Instruct-2507` | — | `0.63` | `6.89 ± 0.06` | `6.45 ± 0.12` | `6.70 ± 0.10` | `9.89 ± 0.04` | `0/56` | `2/56` |

- **Highlights across prompts**: `dental_interdental` jumped `4.75 (Broad) →` **`7.25 ± 0.16` (`2/8` `N≥8` vs `0/8` Base)**; `vaccines_protect` beat Baseline on utility (`6.12` vs `5.50`) and hit rate (`50%` vs `38%`); `cook_faster` beat Baseline on utility (`6.75` vs `6.38`); `bad = 0%` across all 6 starter novices (`8.5–8.7 min` build each).
- **Live `--auto-grow` Verification (Unseen 8th Domain — `produce_freshness`)**:
  - Cache miss detected cleanly (`cooking = 0.432 < 0.52`) → auto-planned, built, and registered `conventional_produce_preservatio` (`GEN = 0.63`, `dogma = 0%`, `0/8` duds) → **beat Baseline (`6.62` vs `6.50` nov)**!
  - Follow-up paraphrase (*"gadget that stops berries and leafy greens from rotting quickly at home"*) routed directly to `conventional_produce_preservatio` at **`sim = 0.616` (instant cache hit)**.

### Running Growing Narrow-Topic MoSN (`MoSN v2.1`)
```bash
# 1. Build the 6-topic Goldilocks starter pool on NVMe (~8.5 min/topic on 2x RTX 5070 Ti):
python mosn/build_narrow_novices.py --model Qwen/Qwen3-4B-Instruct-2507 --rounds 1 --ssd-pct 0.045 --heal-steps 240

# 2. Real-time routed ideation with automatic pool growth on unseen domains:
python mosn/mosn_service.py \
    --prompt "City cyclists keep getting hit by car doors. Help me invent a fix." \
    --pool narrow --auto-grow --threshold 0.52 --n 12 --rank

# 3. Interactive mode (keeps VRAM warm; ~2s swap on known domains, auto-grows new ones):
python mosn/mosn_service.py --interactive --pool narrow --auto-grow --n 8 --rank

# 4. Run the 7-prompt + unseen-domain auto-grow benchmark:
python mosn/eval_mosn.py --model Qwen/Qwen3-4B-Instruct-2507 --pool narrow --n 8 --test-autogrow
```

---

## First-Principles Seeder Post-Training on a Clean Base Model (`seeder/`)

Instead of erasing domain facts per topic via weight editing, [seeder/](file:///home/jimmy/code/creative-toolkit/naive-llm-round3/seeder) starts from raw **`Qwen/Qwen3-8B-Base`** (zero vendor RLHF refusal or mode-collapse circuitry) and post-trains it **100% locally** (`$0` training cost) purely for first-principles creative seeding on 20 strictly held-out engineering arenas (zero dental, vaccine, neuroevolution, cooking, cycling, meeting, or produce data):
1. **Stage 1 (`seeder/synth_corpus.py`, 112m)**: Local `qwen3.8:27b` synthesizes 1,077 orthogonal-analogy seeds across 199 held-out problems → cliché-distance filtered to 915 seeds + 2,400 refusal-free Dolci anchors.
2. **Stage 2 (`seeder/train_sft.py`, 45m)**: Rank-64 `bf16` LoRA SFT on `Qwen3-8B-Base` (`GEN = 0.700`, +7 pp over `Qwen3-8B` Instruct!).
3. **Stage 3 (`seeder/train_grpo.py`, 23m)**: 120 steps of 100% local Anti-Cliché GRPO (cliché-centroid escape + within-group diversity + problem relevance + light Skywork floor, frozen-ref KL via PEFT `disable_adapter()`).

### Zero-Shot Held-Out Results (`deepseek-v4-pro` Mentor + `gpt-5.6-luna` Judge, **Zero Per-Topic Forgetting!**)

| Benchmark | Arm | Raw `N` | Grounded w/F (`N≥8`, Duds) | Grounded noF (`N≥8`) | Paper w/F (`N≥8`) |
|---|---|---:|---|---|---|
| **`neuroevolution` (`n=36`)** | `Qwen3-4B` `IACF-v2` (per-topic) | `2.31` | `5.78` (`2/36`, `7/36`) | `5.33` (`0/36`) | `6.22` (`0/36`) |
| **`neuroevolution` (`n=36`)** | **`8B-Base Seeder v1 GRPO` (0-shot)** | **`5.83`** | **`7.19 ± 0.10`** (**`10/36 = 28%`**, **`0/36`**) | **`6.94`** (**`9/36`**) | **`6.75`** (**`4/36`**) |
| **`dental` (`n=36`)** | Best 7B–14B `IACF-v2` (`Hermes-4-14B`) | `3.33` | `6.89` (`2/36`, `0/36`) | `6.67` (`0/36`) | `6.53` (`1/36`) |
| **`dental` (`n=36`)** | **`8B-Base Seeder v1 GRPO` (0-shot)** | **`6.64`** | **`7.11 ± 0.07`** (**`5/36`**, **`0/36`**) | **`7.08`** (`3/36`) | `7.00` (`3/36`) |
| **7-Prompt Multi-Domain (`n=56`)** | Unedited Baseline (`Qwen3-4B`) | — | `6.89 ± 0.06` (`2/56`, `0/56`) | — | — |
| **7-Prompt Multi-Domain (`n=56`)** | Growing Narrow MoSN `v2.1` (`Qwen3-4B`) | — | `6.82 ± 0.06` (`2/56`, `0/56`) | — | — |
| **7-Prompt Multi-Domain (`n=56`)** | **`8B-Base Seeder v1 GRPO` (0-shot)** | — | **`6.98 ± 0.05`** (**`4/56`**, **`0/56`**, wins 6/7!) | — | — |

- **Key ablation learnings:** Stacking `IACF-v2` on top of the GRPO Seeder degraded both domains (damaging learned analogy circuits), and raising the off-the-shelf RLHF reward model weight (`Skywork-Reward-V2` `0.7 → 1.2`) in `v2` eliminated motif reuse (Vendi `14.62 → 16.95`, catechol `16 → 0`) but pulled the policy back toward conventional chatbot replies (`raw 6.64 → 4.78`). Serving `v1 GRPO` with farthest-point embedding deduplication in `seeder_service.py` gives the best of both worlds!

```bash
# Run the zero-a-priori-topic real-time Seeder service on ANY prompt:
python seeder/seeder_service.py \
    --prompt "How can we capture microplastics from household laundry wastewater?" \
    --n 6 --oversample 3 --mentor deepseek-v4-pro --rank
# (Or pass --mentor qwen3.8:27b for 100% local $0 mentoring, or --interactive!)
```

## File Layout
- `seeder/`: **Zero-Shot First-Principles Seeder** — `synth_corpus.py` / `synth_corpus_v2.py` (local `qwen3.8:27b` held-out synthesis), `train_sft.py` (Rank-64 `Qwen3-8B-Base` ChatML SFT), `train_grpo.py` / `train_grpo_v2.py` (100% local Anti-Cliché GRPO), `eval_seeder.py` (apples-to-apples eval), `seeder_service.py` (turnkey real-time ideation service with farthest-point embedding deduplication).
- `report/`: Full 14-page LaTeX technical report (`technical_report.tex`, `technical_report.pdf`, `references.bib`, `generate_figures.py`, 5 vector PDF figures).
- `mosn/`: **Growing Narrow-Topic & Broad MoSN** — `narrow_topics.py` (6 narrow topic specs), `narrow_pool.py` (NVMe registry, Goldilocks narrow builder, confidence-gated router + `--auto-grow`), `build_narrow_novices.py` (starter-pool builder), `mosn_service.py` (real-time CLI & interactive service), `eval_mosn.py` (7-prompt + auto-grow benchmark), `eval_noveltybench.py` (NoveltyBench distinct/quality/Vendi eval), `domains.py` & `build_novices.py` (legacy 4-swath MoSN).
- `naive_service.py`: Per-topic turnkey CLI service (auto-discovers optimal dogma topic from `--prompt`, builds & caches IACF-v2 naive weights on NVMe, filters refusals & clarifying-question duds, mentors, and ranks).
- `iacf_v2.py`: KL-Anchored Iterative Accelerated Catastrophic Forgetting (supports 4B–14B+, `--balanced_2gpu`, `--gc_fisher`, `--lora_heal` with zero-VRAM CPU LoRA merge).
- `screen_backbones.py` & `refusal_ablate.py`: 12-model backbone raw-seed pre-screener & prefill-only centered refusal-excess clamping ablation tool.
- `topics.py`, `topic_data.py`, `eval_harness.py`, `runner.py`, `analyze.py`, `calibrate.py`: Paper-exact evaluation, MCQ diagnostics, embedding distances, Fightin' Words, and dataset builder.
- `dscu_unlearn.py`, `ssd_unlearn.py`, `concept_unlearn.py`, `surgical_unlearn.py`, `scu_v2.py`, `scu_v3.py`, `iacf_unlearn.py`: Experimental progression leading to `iacf_v2.py`.
- `results/`, `service_outputs/`: Lightweight JSON/JSONL seeds, refinements, judge scores, and metadata (large weight checkpoints excluded via `.gitignore`).
