"""Can a search read report rows? Runs tuning and the rule search at the tiny size with 2 steps for about 6 s each, and every row they write must hold text.tune.loss and nothing else. Running out of time is expected here: exit code 3."""
import json, os, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE); import arms
d = tempfile.mkdtemp(); r, c = os.path.relpath(os.path.join(d, "r.jsonl"), HERE), os.path.relpath(os.path.join(d, "c.jsonl"), HERE)
env = dict(os.environ, PAR="4", LOAD="99", OMP_NUM_THREADS="1")
with open(os.path.join(HERE, c), "w") as f: f.write(json.dumps(dict(kind="tune", arm="muon", spec=arms.show(arms.parse("muon")), tune=9.0, evals=1, tier="tiny", moe=0, tied=False, qknorm=True)) + "\n")
for cmd in (["tune", "all-adam"], ["rule"]):
    p = subprocess.run([sys.executable, "search.py", *cmd, "--tier", "tiny", "--steps", "2", "--seconds", "6", "--results", r, "--choices", c], cwd=HERE, env=env, capture_output=True, text=True, timeout=40)
    assert p.returncode in (0, 3), p.stderr[-800:]
rows = [json.loads(l) for l in open(os.path.join(HERE, r))]
assert rows and all(set(x["scores"]) <= {"text.tune.loss"} for x in rows), "a search row holds a report score"
print(f"{len(rows)} search rows hold only tune scores")
