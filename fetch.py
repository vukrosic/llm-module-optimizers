"""Download small portions of published benchmarks into data/bench/ as plain jsonl. Run once: python3 -I fetch.py

Choice benchmarks become {"pairs": [[prefix, continuation], ...], "answer": i} (optionally "group"), scored the usual way:
the option whose continuation has the highest total log-probability is the model's answer. Text kinds become {"text": ...}.
Only parquet data is read, no dataset code runs.
"""
import itertools, json, os
from datasets import load_dataset, get_dataset_config_names

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bench"); os.makedirs(OUT, exist_ok=True)


def stream(name, split, config=None, n=1000):
    return itertools.islice(load_dataset(name, config, split=split, streaming=True), n)


def save(name, rows):
    rows = list(rows)
    with open(os.path.join(OUT, name + ".jsonl"), "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(name, len(rows), flush=True)


def blimp():
    for c in get_dataset_config_names("nyu-mll/blimp"):
        for r in stream("nyu-mll/blimp", "train", c, 40):
            yield dict(pairs=[["", r["sentence_good"]], ["", r["sentence_bad"]]], answer=0, group=c)


def hellaswag():
    for r in stream("Rowan/hellaswag", "validation", None, 1500):
        yield dict(pairs=[[r["ctx"], " " + e] for e in r["endings"]], answer=int(r["label"]))


def winogrande():
    for r in stream("allenai/winogrande", "validation", "winogrande_xl", 1267):
        a, b = r["sentence"].split("_", 1)
        yield dict(pairs=[[a + r["option1"], b], [a + r["option2"], b]], answer=int(r["answer"]) - 1)


def choice(q, labels, texts, key):
    return dict(pairs=[[f"Question: {q}\nAnswer:", " " + t] for t in texts], answer=list(labels).index(key))


def arc():
    for r in stream("allenai/ai2_arc", "test", "ARC-Challenge", 1500):
        if r["answerKey"] in r["choices"]["label"]: yield choice(r["question"], r["choices"]["label"], r["choices"]["text"], r["answerKey"])


def mmlu():
    for r in stream("cais/mmlu", "test", "all", 1500):
        yield dict(choice(r["question"], range(len(r["choices"])), r["choices"], r["answer"]), group=r["subject"])


def texts(name, split, field, config=None, n=400, cap=3000):
    for r in stream(name, split, config, n):
        t = r[field] if isinstance(field, str) else field(r)
        if t and len(t) > 50: yield dict(text=t[:cap])


JOBS = {
    "blimp": blimp, "hellaswag": hellaswag, "winogrande": winogrande, "arc": arc, "mmlu": mmlu,
    "text_fiction": lambda: texts("roneneldan/TinyStories", "validation", "text"),
    "text_math": lambda: texts("EleutherAI/hendrycks_math", "test", lambda r: r["problem"] + " " + r["solution"], "algebra"),
    "text_code": lambda: texts("Nan-Do/code-search-net-python", "train", "code"),
    "text_german": lambda: texts("Helsinki-NLP/opus_books", "train", lambda r: r["translation"]["de"], "de-en", 3000),
}
for name, job in JOBS.items():
    try:
        save(name, job())
    except Exception as e:
        print("FAILED", name, type(e).__name__, str(e)[:200], flush=True)
