"""Which scores can judge an optimizer? Cut-offs are set here before reading the table.

A score can judge if it has a direction, at least 30 items, and reads different items than the main score (so text.web, rarity and ambiguous are out, because they are the same rows), and it is not a train-set score;
training moved it (end versus the blind control by 2 row errors); spread across the tuned baselines is at least 3 row errors; and its ranking of baselines agrees between lr multiples (rho >= .4), between tiers (rho >= .3) and between worlds (rho >= .4).
Hidden rows are never read here.

python3 audit.py [--tier 1m] [--next-tier 5sx4]
"""
import argparse, json, os
import numpy as np
from scipy import stats
import beat, claims, env

SAME_ROWS = ("text.web", "text.tune", "rarity.", "ambiguous.", "memorize.", "messy.")


def main():
    p = argparse.ArgumentParser(); p.add_argument("--tier", default="5s"); p.add_argument("--next-tier", default="5sx4"); p.add_argument("--world", type=int, default=1); a = p.parse_args()
    rows = claims.rows(); ch = claims.chosen(a.tier); ch2 = claims.chosen(a.next_tier); names = [n for n in ch if n not in beat.CONTROLS]
    get = lambda n, t, c, **k: claims.find(rows, c[n]["spec"], t, **k) if n in c else None
    base = {n: get(n, a.tier, ch) for n in names}; base = {n: r for n, r in base.items() if r}
    if len(base) < 4: print("too few baselines with full evals"); return
    blind = get("blind", a.tier, ch); keep = []
    print(f"{len(base)} tuned baselines, tier {a.tier}\n{'score':36} items moved spread rowerr ratio   lr   tier  world  judge")
    for s in sorted(next(iter(base.values()))["scores"]):
        n = next(iter(base.values()))["n"].get(s, 0)
        try: beat.better(s, 1.0)
        except ValueError: print(f"{s:36} {n:5} no direction"); continue
        if s.startswith(SAME_ROWS) or n < 30: print(f"{s:36} {n:5} not admitted: {'same rows or train set' if s.startswith(SAME_ROWS) else 'too few items'}"); continue
        v = {k: r["scores"][s] for k, r in base.items() if r["scores"].get(s) is not None}
        if len(v) < 4: continue
        ks = list(v); err = []
        for i, x in enumerate(ks):
            for y in ks[i + 1:]:
                d = claims.items(base[x], s) - claims.items(base[y], s); err.append(float(d.std(ddof=1) / len(d) ** .5))
        rowerr = float(np.median(err)); spread = max(v.values()) - min(v.values()); moved = blind is not None and blind["scores"].get(s) is not None and abs(blind["scores"][s] - np.mean(list(v.values()))) > 2 * rowerr

        def rho(other):
            o = {k: r["scores"].get(s) for k, r in other.items() if r and r["scores"].get(s) is not None}; c = [k for k in v if k in o]
            return float(stats.spearmanr([v[k] for k in c], [o[k] for k in c]).statistic) if len(c) >= 4 else float("nan")
        lr, tier, world = rho({k: get(k, a.tier, ch, lr_mult=1.5) for k in base}), rho({k: get(k, a.next_tier, ch2) for k in base}), rho({k: get(k, a.tier, ch, world=a.world) for k in base})
        ok = bool(moved and spread >= 3 * rowerr and lr >= .4 and tier >= .3 and world >= .4)
        if ok: keep.append(s)
        print(f"{s:36} {n:5} {str(moved):5} {spread:6.3f} {rowerr:6.3f} {spread / rowerr:5.1f} {lr:5.2f} {tier:5.2f} {world:5.2f}  {'YES' if ok else ''}")
    json.dump(keep, open(os.path.join(env.HERE, claims.JUDGES), "w"), indent=0); print(len(keep), "scores can judge")


if __name__ == "__main__":
    main()
