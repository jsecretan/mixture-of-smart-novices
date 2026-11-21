"""Build clean, topic-general Forget (solutions), Problem (keep), and Retain (general) datasets.

For any topic in naive-llm/topics.py:
- forget_texts: Wikipedia solution articles + Dolci solution conversations + default-solution Q&A
- problem_texts: Wikipedia problem articles (strictly filtered of solution terms!)
- retain_texts: Non-topic Dolci conversations + control Wikipedia + general invention Q&A
"""
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAIVE_DIR = ROOT / "naive-llm"
ROUND3_DIR = Path(__file__).resolve().parent
if str(ROUND3_DIR) not in sys.path:
    sys.path.insert(0, str(ROUND3_DIR))
if NAIVE_DIR.exists() and str(NAIVE_DIR) not in sys.path:
    sys.path.append(str(NAIVE_DIR))
from topics import TOPICS, CONTROL_FIT  # noqa: E402

WIKI_DIR = NAIVE_DIR / "data" / "wiki"

GENERIC_INVENT = [
    "carry groceries up stairs", "keep a tent warm at night", "stop a door from slamming",
    "water plants while I am away", "find my keys in the dark", "dry shoes quickly",
    "keep birds off a balcony", "organise tangled cables", "cool a parked car",
    "help a child learn to tie shoelaces", "keep bread fresh longer", "reduce noise from a neighbour",
    "remember to take medication", "clean a fish tank", "sharpen pencils without mess",
    "carry a ladder alone", "prevent fogged glasses", "sort recycling at home",
    "peel garlic quickly", "keep a phone charged while hiking",
    "remove barnacles from a boat hull", "clean solar panels in a dusty desert",
    "remove grease from a kitchen vent", "clear moss from roof tiles",
    "clean narrow glass bottles", "remove mineral scale from pipes",
    "scrub algae off stone steps", "clear lint from tight machine crevices",
]


def compile_filter(topic):
    terms = TOPICS[topic]["terms"]
    # Word-boundary regex for terms; also include core stems for dental
    extra = []
    if topic == "dental":
        extra = ["toothbrush", "toothbrushes", "toothpaste", "floss", "flossing", "flosser",
                 "mouthwash", "bristle", "bristles", "waterpik", "interdental", "dentist",
                 "dentistry", "brush your teeth", "brushing your teeth", "brushing teeth",
                 "clean your teeth with a brush", "electric brush", "sonic brush"]
    all_terms = sorted(set(terms + extra), key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(t) for t in all_terms) + r")\b", re.I)


def read_wiki(title):
    p = WIKI_DIR / (title.replace("/", "_") + ".txt")
    return p.read_text() if p.exists() else ""


def chunk_text(text, max_chars=1200, min_chars=180):
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(p) < 40 or p.startswith("=="):
            continue
        if cur and len(cur) + len(p) > max_chars:
            if len(cur) >= min_chars:
                chunks.append(cur)
            cur = p
        else:
            cur = (cur + "\n\n" + p).strip() if cur else p
    if len(cur) >= min_chars:
        chunks.append(cur)
    return chunks


def build_topic_data(topic="dental", max_forget_conv=600, max_retain_conv=1500, seed=42):
    out_dir = ROUND3_DIR / "data" / topic
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "datasets.json"
    if out_file.exists():
        return json.load(open(out_file))

    rng = random.Random(seed)
    T = TOPICS[topic]
    pat = compile_filter(topic)

    # 1. Forget Wikipedia (solutions only)
    forget_wiki = []
    for title in T["fit"] + T["held"]:
        txt = read_wiki(title)
        if not txt:
            continue
        for c in chunk_text(txt):
            if pat.search(c):
                forget_wiki.append({"type": "wiki", "title": title, "text": c})

    # 2. Problem Wikipedia (keep: anatomy/biology of problem, strictly stripped of solution sentences!)
    problem_wiki = []
    for title in T["keep"]:
        txt = read_wiki(title)
        if not txt:
            continue
        # Remove any sentence mentioning a solution term
        sents = re.split(r"(?<=[.!?])\s+", txt)
        clean_sents = [s for s in sents if not pat.search(s)]
        clean_txt = " ".join(clean_sents)
        for c in chunk_text(clean_txt):
            problem_wiki.append({"type": "wiki", "title": title, "text": c})

    # Also keep problem-only sentences from Dental plaque / Tooth decay if dental
    if topic == "dental":
        for title in ["Dental plaque", "Tooth decay"]:
            txt = read_wiki(title)
            sents = re.split(r"(?<=[.!?])\s+", txt)
            clean_sents = [s for s in sents if not pat.search(s)]
            for c in chunk_text(" ".join(clean_sents)):
                problem_wiki.append({"type": "wiki", "title": title + "_problem", "text": c})

    # 3. General control Wikipedia
    retain_wiki = []
    for title in CONTROL_FIT:
        txt = read_wiki(title)
        if not txt:
            continue
        for c in chunk_text(txt)[:10]:
            if not pat.search(c):
                retain_wiki.append({"type": "wiki", "title": title, "text": c})

    # 4. Dolci conversations (forget vs retain)
    forget_conv = []
    retain_conv = []
    pool_dental = NAIVE_DIR / "data" / "pool_dental.jsonl"
    pool_sample = NAIVE_DIR / "data" / "pool_sample.jsonl"

    if topic == "dental" and pool_dental.exists():
        all_d = []
        for line in open(pool_dental):
            r = json.loads(line)
            msgs = r["messages"]
            # Require at least one user+assistant turn and solution term in assistant reply
            asst = " ".join(m["content"] for m in msgs if m["role"] == "assistant")
            if len(asst) >= 80 and pat.search(asst) and len(asst) <= 3000:
                all_d.append({"type": "chat", "messages": msgs})
        rng.shuffle(all_d)
        forget_conv = all_d[:max_forget_conv]

    if pool_sample.exists():
        all_r = []
        for line in open(pool_sample):
            r = json.loads(line)
            msgs = r["messages"]
            full = " ".join(m["content"] for m in msgs)
            asst = " ".join(m["content"] for m in msgs if m["role"] == "assistant")
            if not pat.search(full) and 80 <= len(asst) <= 3000:
                all_r.append({"type": "chat", "messages": msgs})
        rng.shuffle(all_r)
        retain_conv = all_r[:max_retain_conv]

    # 5. Direct solution-probing Q&A (synthesized from topic terms/description, zero human leakage from eval prompts)
    forget_qa_prompts = []
    if topic == "dental":
        forget_qa_prompts = [
            "What is a toothbrush and how does it work?",
            "How do people usually clean their teeth every day?",
            "What is dental floss used for?",
            "What ingredients are in toothpaste and why are they used?",
            "Explain how a water flosser cleans between teeth.",
            "What is the difference between a manual toothbrush and an electric toothbrush?",
            "How does mouthwash help with oral hygiene?",
            "What is an interdental brush?",
            "What does a dentist recommend for daily plaque removal?",
            "How do you properly brush and floss your teeth?",
            "What is a tongue scraper used for?",
            "Why do toothbrushes have nylon bristles?",
            "How does fluoride toothpaste protect teeth?",
            "What are the standard tools people use for oral hygiene?",
            "How do ultrasonic or sonic toothbrushes clean teeth?",
            "What are the most common commercial products for cleaning teeth?",
            "Explain the Bass brushing technique for teeth.",
            "What is scaling and polishing at a dental clinic?",
            "How does waxed vs unwaxed dental floss work?",
            "What is oil pulling and how is it used in dental care?",
        ]
    else:
        desc = T["description"]
        forget_qa_prompts = [
            f"What is {desc} and how does it work?",
            f"What are the standard methods used in {desc}?",
            f"Explain the key concepts and tools of {desc}.",
        ] + [f"What is {term} and how is it used?" for term in T["terms"][:15]]

    # 6. Problem-preserving Q&A prompts (strictly about anatomy/problem, NOT solutions)
    problem_qa_prompts = []
    if topic == "dental":
        problem_qa_prompts = [
            "What are human teeth made of?",
            "Describe the structure of tooth enamel and dentin.",
            "What is the anatomy of the human mouth and gums?",
            "How does saliva work in the human mouth?",
            "How do bacteria form biofilm and debris on hard surfaces in the mouth?",
            "What are the different types of human teeth and their shapes?",
            "How does chewing break down food mechanically?",
            "Why does acid from bacteria damage tooth enamel over time?",
            "What are the tight contact spaces between adjacent human teeth like?",
            "How does the human tongue move inside the oral cavity?",
        ]

    data = {
        "topic": topic,
        "forget_wiki": forget_wiki,
        "forget_conv": forget_conv,
        "forget_qa_prompts": forget_qa_prompts,
        "problem_wiki": problem_wiki,
        "problem_qa_prompts": problem_qa_prompts,
        "retain_wiki": retain_wiki,
        "retain_conv": retain_conv,
        "generic_invent": [f"Can you help me invent something to {g}?" for g in GENERIC_INVENT],
    }
    json.dump(data, open(out_file, "w"), indent=1)
    print(
        f"[topic_data:{topic}] forget_wiki={len(forget_wiki)} forget_conv={len(forget_conv)} "
        f"problem_wiki={len(problem_wiki)} retain_wiki={len(retain_wiki)} retain_conv={len(retain_conv)}",
        flush=True,
    )
    return data


if __name__ == "__main__":
    t = sys.argv[1] if len(sys.argv) > 1 else "dental"
    build_topic_data(t)
