"""Reads attn_results.jsonl: per task and score, the paired gap between two arms over worlds, with the verdict rule from claims.py. Positive means the first arm is better.

python3 attnread.py --a "qk=muon:.02" --b "qk=freeze:0" --tier a --worlds 1,2,3 [--evals report]
"""
import argparse, os
import numpy as np
import attnenv, claims

HIGHER = (".acc",)


def rows(a, tier, world, ev, task, path=attnenv.RESULTS): return attnenv.done(path).get(attnenv.key_of(a, task, tier, world, ev))


def table(a, b, tier, worlds, ev="report", tasks=attnenv.TASKS, path=attnenv.RESULTS):
    """{score: [(world, gap, row error)]}. Every gap is written so positive means arm a is better."""
    all_rows = attnenv.done(path); out = {}
    for t in tasks:
        for w in worlds:
            x, y = all_rows.get(attnenv.key_of(a, t, tier, w, ev)), all_rows.get(attnenv.key_of(b, t, tier, w, ev))
            if not x or not y or not x["finite"] or not y["finite"]: continue
            p, q = np.load(os.path.join(attnenv.RUNS, x["key"] + ".npz")), np.load(os.path.join(attnenv.RUNS, y["key"] + ".npz"))
            for s in p.files:
                d = (p[s] - q[s]) if s.endswith(HIGHER) else (q[s] - p[s]); out.setdefault(s, []).append((w, float(d.mean()), float(d.std(ddof=1) / len(d) ** .5)))
    return out


def show(a, b, tier, worlds, ev="report", path=attnenv.RESULTS):
    print(f"{'score':18} {'mean gap':>9} {'error':>7}  verdict   per world   (positive: first arm better)")
    for s, g in table(attnenv.parse(a), attnenv.parse(b), tier, worlds, ev, path=path).items():
        v, m, se = claims.verdict(g); print(f"{s:18} {m:+9.3f} {se:7.3f}  {v:8}  {' '.join(f'{x[1]:+.3f}' for x in g)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--a"); p.add_argument("--b"); p.add_argument("--tier", default="a"); p.add_argument("--worlds", default="1,2,3"); p.add_argument("--evals", default="report"); a = p.parse_args(); show(a.a, a.b, a.tier, [int(x) for x in a.worlds.split(",")], a.evals)
