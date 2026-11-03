"""Which scores can judge an optimizer? Cut-offs are set here before reading the table, as in llm-memory-env's audit.

A score can judge if: it moved in training on the Muon recipe (end versus the blind control, because no arm stays untrained); spread across the tuned baselines is at least 3 times the training-noise sigma of that score;
its ranking of baselines agrees between lr multiples (rho >= .4), between tiers (rho >= .3), and between worlds (rho >= .4). Hidden rows are never read here.

python3 audit.py [--tier 5s] [--next-tier 5sx4]
"""
import argparse, json, os
import numpy as np
from scipy import stats
import arms, claims, env


def main():
    p = argparse.ArgumentParser(); p.add_argument("--tier", default="5s"); p.add_argument("--next-tier", default="5sx4"); p.add_argument("--world", type=int, default=1); a = p.parse_args()
    all_rows = [r for r in claims.rows() if r["evals"] == "full" and not r["leak"] and not r["double"] and r["qknorm"] and not r["tied"] and r["moe"] == 0]
    ch = claims.chosen(a.tier); ch2 = claims.chosen(a.next_tier); names = [n for n in ch if n not in ("blind", "frozen-mlp")]
    get = lambda name, tier, spec, **k: next((r for r in reversed(all_rows) if r["spec"] == arms.show(arms.parse(spec)) and r["tier"] == tier and r["world"] == k.get("world", 0) and r["jitter"] == k.get("jitter", 0) and r["lr_mult"] == k.get("lr_mult", 1.0)), None)
    base = {n: get(n, a.tier, ch[n]["spec"]) for n in names}; base = {n: r for n, r in base.items() if r}
    if not base: print("no full evals yet"); return
    jit = [r for r in all_rows if r["tier"] == a.tier and r["world"] == 0 and r["spec"] == arms.show(arms.parse(ch["muon"]["spec"])) and r["lr_mult"] == 1]
    blind = get("blind", a.tier, ch["blind"]["spec"]) if "blind" in ch else None
    print(f"{len(base)} tuned baselines, {len(jit)} noise runs of the Muon recipe")
    print(f"{'score':34}  moved spread  sigma  ratio   lr   tier  world  judge")
    keep = []
    for s in sorted(next(iter(base.values()))["scores"]):
        if s == "text.tune.loss": continue
        v = {n: r["scores"].get(s) for n, r in base.items()}; v = {n: x for n, x in v.items() if x is not None}
        if len(v) < 4: continue
        spread = max(v.values()) - min(v.values()); sig = np.std([r["scores"][s] for r in jit if r["scores"].get(s) is not None], ddof=1) if len(jit) > 2 else float("nan")
        moved = blind is not None and blind["scores"].get(s) is not None and abs(blind["scores"][s] - v.get("muon", 0)) > 2 * sig
        def rho(other):
            o = {n: r["scores"].get(s) for n, r in other.items() if r and r["scores"].get(s) is not None}; c = [n for n in v if n in o]
            return float(stats.spearmanr([v[n] for n in c], [o[n] for n in c]).statistic) if len(c) >= 4 else float("nan")
        lr = rho({n: get(n, a.tier, ch[n]["spec"], lr_mult=1.5) for n in base}); tier = rho({n: get(n, a.next_tier, ch2[n]["spec"]) for n in base if n in ch2}); world = rho({n: get(n, a.tier, ch[n]["spec"], world=a.world) for n in base})
        ok = bool(moved and spread >= 3 * sig and lr >= .4 and tier >= .3 and world >= .4)
        if ok: keep.append(s)
        print(f"{s:34}  {str(moved):5} {spread:6.3f} {sig:6.3f} {spread / sig if sig else 0:5.1f} {lr:5.2f} {tier:5.2f} {world:5.2f}  {'YES' if ok else ''}")
    json.dump(keep, open(os.path.join(env.HERE, "judges.json"), "w"), indent=0); print(len(keep), "scores can judge")


if __name__ == "__main__":
    main()
