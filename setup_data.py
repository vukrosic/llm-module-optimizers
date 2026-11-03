"""Download everything the environment reads into data/. Run once: python3 setup_data.py [--no-bench]

Web text: two files of FineWeb, tokenized with the GPT-2 tokenizer, 382 MB. Facts: ParaRel at a fixed commit, 4 MB.
Benchmarks: small portions of BLiMP, HellaSwag, WinoGrande, ARC, MMLU and 4 text kinds, 6 MB, through fetch.py.
The files that are derived from these (data/facts.json, data/counts.npy) are built the first time env.py runs.
"""
import io, os, subprocess, sys, tarfile, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
WEB = "https://huggingface.co/datasets/kjj0/fineweb100B-gpt2/resolve/main/"; COMMIT = "cb5554678457beb5ac163d888f1ce8cf174b3f0b"
os.makedirs(os.path.join(DATA, "web"), exist_ok=True)

for name in ("fineweb_val_000000.bin", "fineweb_train_000001.bin"):
    path = os.path.join(DATA, "web", name)
    if os.path.exists(path): print("have", name); continue
    print("downloading", name, "(191 MB)", flush=True); subprocess.run(["curl", "-fsSL", "--retry", "5", "-o", path + ".part", WEB + name], check=True); os.rename(path + ".part", path)

if not os.path.exists(os.path.join(DATA, "pararel")):
    print("downloading ParaRel", flush=True); url = f"https://codeload.github.com/yanaiela/pararel/tar.gz/{COMMIT}"
    with tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(url, timeout=120).read())) as t:
        for m in t.getmembers():
            parts = m.name.split("/")
            if m.isfile() and parts[1] == "data" and parts[2] in ("pattern_data", "trex_lms_vocab"):
                out = os.path.join(DATA, "pararel", *parts[2:]); os.makedirs(os.path.dirname(out), exist_ok=True); open(out, "wb").write(t.extractfile(m).read())

if "--no-bench" not in sys.argv: subprocess.run([sys.executable, "-I", os.path.join(HERE, "fetch.py")], check=True)
print("done")
