"""Can a search read report rows? Runs one tuning twice at the tiny size, the second time with every score except text.tune.loss replaced by noise. The two choices must match."""
import json, os, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); d = tempfile.mkdtemp(); out = []
for poison in ("", "1"):
    r, c = os.path.join(d, f"r{poison}.jsonl"), os.path.join(d, f"c{poison}.jsonl")
    env = dict(os.environ, POISON=poison, PAR="2", LOAD="99", OMP_NUM_THREADS="1")
    if not poison: env.pop("POISON")
    subprocess.run([sys.executable, "search.py", "tune", "muon", "--tier", "tiny", "--steps", "4", "--seconds", "15", "--results", os.path.relpath(r, HERE), "--choices", os.path.relpath(c, HERE)], cwd=HERE, env=env, check=True, capture_output=True, timeout=20)
    out.append(json.loads(open(c).readline())["spec"]); web = [json.loads(l)["scores"].get("text.web.loss") for l in open(r)]; print(("poisoned " if poison else "clean ") + f"web scores look like {web[:3]}")
assert out[0] == out[1], out
print("same choice with report scores replaced by noise:", out[0])
