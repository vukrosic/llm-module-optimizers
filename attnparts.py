"""Removes one part of Muon at a time at Muon's lr, with no tuning, and prints each arm's mean scores and paired verdict against Muon. Only the listed part changes, so a gap is that part's effect.

python3 attnparts.py [--worlds 4,5,6,7,8] [--tier a] [--lr .02]
"""
import argparse
import numpy as np
import attnenv, attnread, attntune, claims

SCORES = tuple(attntune.SCORES.values()) + ("majority.area",)


def arms_of(lr):
    return dict(muon=f"qk=muon:{lr:g}", no_momentum=f"qk=muon:{lr:g}:beta=0", no_orthogonalization=f"qk=normsgd:{lr:g}", no_growth=f"qk=muonfixed:{lr:g}", schedule_for_growth=f"qk=muonbrake:{lr:g}", strong_schedule=f"qk=muonbrake:{lr:g}:tau=150", per_head_orthogonalization=f"qk=muonsplit:{lr:g}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--worlds", default="4,5,6,7,8"); p.add_argument("--tier", default="a"); p.add_argument("--lr", type=float, default=.02); a = p.parse_args(); W = tuple(int(x) for x in a.worlds.split(","))
    S = {k: attnenv.parse(v) for k, v in arms_of(a.lr).items()}; attnenv.run_all(list(S.values()), a.tier, W, "report"); r = attnenv.done()
    print(f"{'arm':28}" + "".join(f"{s.split('.')[0]:>9}" for s in SCORES) + "   size at end")
    for k, sp in S.items():
        rows = {s: [] for s in SCORES}; size = []
        for t in attnenv.TASKS:
            for w in W:
                x = r.get(attnenv.key_of(sp, t, a.tier, w, "report"))
                if x and x["finite"]:
                    size.append(x["qk_norm"][1])
                    for s in SCORES:
                        if s in x["scores"]: rows[s].append(x["scores"][s])
        print(f"{k:28}" + "".join(f"{np.mean(rows[s]):9.3f}" if rows[s] else f"{'-':>9}" for s in SCORES) + f"   {np.mean(size):.1f}")
    print("\ngap against Muon, positive means the arm is better; verdict uses .02 and 2 errors")
    for k, sp in S.items():
        if k == "muon": continue
        t = attnread.table(sp, S["muon"], a.tier, W); print(f"{k:28}" + "  ".join(f"{s.split('.')[0]} {claims.verdict(t[s], .02)[1]:+.3f} {claims.verdict(t[s], .02)[0]}" for s in SCORES if s in t))
