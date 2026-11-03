"""Rule 8 of beat.py: hidden rows, once. Trains the final design and base in world 20 with --final, prints the paired gap on hidden web rows, and refuses a second run at that tier.

python3 final.py --tier 1m [--design rule] [--manifold]
"""
import argparse, json, os, subprocess, sys
import numpy as np
import beat, claims, env

WORLD = 20; LOG = os.environ.get("FINAL_LOG", "final_log.jsonl")


def main():
    p = argparse.ArgumentParser(); p.add_argument("--tier", default="1m"); p.add_argument("--design", default="rule"); p.add_argument("--manifold", action="store_true"); a = p.parse_args()
    log = os.path.join(env.HERE, LOG); old = [json.loads(l) for l in open(log)] if os.path.exists(log) else []
    if any(r["tier"] == a.tier for r in old): raise SystemExit(f"hidden rows of tier {a.tier} were already read: {old}")
    ch = beat.load(); d, b = beat.design_of(a.tier, ch, a.design, a.manifold), beat.base_of(a.tier, ch)
    if not d or not b: raise SystemExit("design or base not chosen")
    keys = {}
    for name, r in (("base", b), ("design", d)):
        subprocess.run([sys.executable, "env.py", "--spec", r["spec"], "--tier", a.tier, "--world", str(WORLD), "--evals", "full", "--final", "--force", "--results", claims.RESULTS, "--tag", "final"], cwd=env.HERE, check=True, capture_output=True, env=dict(os.environ, OMP_NUM_THREADS="1"))
        keys[name] = claims.find(claims.rows(), r["spec"], a.tier, WORLD)["key"]
    x, y = (np.load(os.path.join(env.RUNS, f"final_{keys[n]}.npz"))["text.web.loss"] for n in ("base", "design")); g = x - y
    out = dict(tier=a.tier, world=WORLD, base=b["spec"], design=d["spec"], gap=round(float(g.mean()), 4), error=round(float(g.std(ddof=1) / len(g) ** .5), 4), rows=len(g), sign_holds=bool(g.mean() > 0))
    with open(log, "a") as f: f.write(json.dumps(out) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
