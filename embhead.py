"""Does Adam on embedding and head beat Muon there at this length? Settings come from tune rows of world 0; gaps use report rows of worlds nothing chose from.

python3 embhead.py [--tier 5sx4] [--seconds 300]
"""
import argparse
import arms, claims, env, runq, search
from qkrule import gap, say

WORLDS = (15, 16, 17, 18, 19)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--tier", default="5sx4"); p.add_argument("--seconds", type=float, default=300); p.add_argument("--results", default=claims.RESULTS); p.add_argument("--choices", default=claims.CHOICES); a = p.parse_args(); a.moe = 0; a.tied = False; a.qknorm = True
    t = search.tuned(a); x, y = (arms.parse(t[k]["spec"]) for k in ("muon", "muon-all"))
    print("left", runq.run([env.parser().parse_args(["--spec", arms.show(s), "--tier", a.tier, "--evals", "full", "--tag", "embhead", "--world", str(w), "--results", a.results]) for s in (x, y) for w in WORLDS], a.seconds), flush=True)
    say("muon recipe vs muon on every matrix", gap(claims.rows(), x, y, WORLDS, a))
    if runq.EX: runq.EX.shutdown()


if __name__ == "__main__":
    main()
