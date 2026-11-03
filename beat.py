"""The win rule, written before any new design runs. A per-module assignment wins only if all of this holds against the best tuned published baseline at that tier.

Fairness: both arms train the same model, the same weights, on the same tokens in the same order, with one gradient per step, and both get tuned with search.py's protocol on tune rows of world 0.

1. Fresh worlds. text.web.loss gap on report rows of each confirmation world (100, 101, 102). No search reads these worlds. The gap has the same sign in all 3.
2. Size of the gap. Mean gap over worlds exceeds max(0.005, 2 sigma), where sigma is the standard deviation of the paired gap across the 3 worlds, and paired z over rows exceeds 2 in each world.
3. Length and size. The rule holds at every tier in TIERS, each assignment's lrs re-tuned at that tier. At 5sx4 the gap must be at least half of its 5s value, because a 260-step gain in muon/rownorm-parts did not hold at 1200 steps.
4. Compute. Same tokens and one gradient per step; time per step, measured one run at a time on one thread, at most 1.25x the baseline; optimizer state at most 2x the baseline.
5. Stability. No instability flag in any world.
6. Other scores. Among scores that can judge (judges.json), wins at least equal losses over the 3 worlds, and every loss is printed.
If it comes close, the result stays "no win" and the lost scores go in the report. The number of designs tried is printed from choices.jsonl.

python3 beat.py --design 5s="<spec>" --design 5sx4="<spec>" --design 1m="<spec>" --base 5s="<spec>" --base 5sx4="<spec>" --base 1m="<spec>" --time 5s=.024,.024 ...
Each tier names the re-tuned spec of both arms. A tier left out counts as not passed.
"""
import argparse, json, os
import numpy as np
import arms, claims, env

WORLDS = (100, 101, 102); TIERS = ("5s", "5sx4", "1m"); MIN_GAP = 0.005; COMPUTE = 1.25; STATE = 2.0
LOWER = (".loss", ".nll", ".delta")


def better(score, d):
    """d = baseline - design; positive means the design is better on a lower-is-better score."""
    return d if score.endswith(LOWER) else -d


def row(all_rows, spec, tier, world):
    want = arms.show(arms.parse(spec))
    for r in reversed(all_rows):
        if r["spec"] == want and r["tier"] == tier and r["world"] == world and r["jitter"] == 0 and r["lr_mult"] == 1 and r["evals"] == "full" and r["moe"] == 0 and not r["leak"] and not r["double"]: return r


def judge(design, base, tier, all_rows, judges, speed=None):
    gaps, zs, why, wins, losses = [], [], [], {}, {}
    for w in WORLDS:
        d, b = row(all_rows, design, tier, w), row(all_rows, base, tier, w)
        if not d or not b: why.append(f"world {w} not run"); continue
        if not d["finite"] or d["opt"]["unstable"]: why.append(f"unstable in world {w}: {d['opt'].get('why')}")
        if d["tokens"] != b["tokens"] or d["grads_per_step"] != 1: why.append(f"tokens or gradients differ in world {w}: {d['tokens']} vs {b['tokens']}")
        if d["state_bytes"] > STATE * b["state_bytes"]: why.append(f"state {d['state_bytes']} is over {STATE}x baseline {b['state_bytes']}")
        if not d["finite"] or not b["finite"]: continue
        x = np.load(os.path.join(env.RUNS, b["key"] + ".npz")); y = np.load(os.path.join(env.RUNS, d["key"] + ".npz")); g = x["text.web.loss"] - y["text.web.loss"]
        gaps.append(float(g.mean())); zs.append(float(g.mean() / (g.std(ddof=1) / len(g) ** .5)))
        for s in judges:
            if s not in x or len(x[s]) < 2: continue
            e = better(s, x[s] - y[s]); z = float(e.mean() / (e.std(ddof=1) / len(e) ** .5 + 1e-12))
            if z > 2: wins[s] = wins.get(s, 0) + 1
            if z < -2: losses.setdefault(s, []).append(round(float(e.mean()), 4))
    if len(gaps) < len(WORLDS): why.append("not all worlds finished"); return dict(win=False, why=why, gaps=gaps, zs=zs)
    if min(gaps) <= 0: why.append(f"gap changes sign across worlds: {[round(g, 4) for g in gaps]}")
    sigma = float(np.std(gaps, ddof=1)); need = max(MIN_GAP, 2 * sigma)
    if np.mean(gaps) <= need: why.append(f"mean gap {np.mean(gaps):.4f} is not above {need:.4f}")
    if min(zs) <= 2: why.append(f"z over rows {[round(z, 1) for z in zs]}")
    if sum(wins.values()) < sum(len(v) for v in losses.values()): why.append(f"loses more judged scores than it wins: {losses}")
    if speed:
        if speed[0] > COMPUTE * speed[1]: why.append(f"time per step {speed[0]:.4f} is over {COMPUTE}x baseline {speed[1]:.4f}")
    else: why.append("time per step not measured one run at a time")
    return dict(win=not why, why=why, gaps=[round(g, 4) for g in gaps], zs=[round(z, 1) for z in zs], need=round(need, 4), wins=wins, losses=losses)


def named(items):
    return dict(x.split("=", 1) for x in items)


def verdict(design, base, all_rows, judges, speed):
    """Rule 3 sits here: every tier must pass by itself, and the 5sx4 gap must keep at least half of the 5s gap."""
    out = {t: judge(design[t], base[t], t, all_rows, judges, speed.get(t)) if t in design and t in base else dict(win=False, why=["tier not run"], gaps=[]) for t in TIERS}
    a, b = out["5s"]["gaps"], out["5sx4"]["gaps"]
    if a and b and np.mean(b) < .5 * np.mean(a): out["5sx4"]["win"] = False; out["5sx4"]["why"].append(f"gap faded with length: {np.mean(a):.4f} at 5s, {np.mean(b):.4f} at 5sx4")
    return out, all(v["win"] for v in out.values())


def main():
    p = argparse.ArgumentParser(); p.add_argument("--design", action="append", required=True); p.add_argument("--base", action="append", required=True); p.add_argument("--time", action="append", default=[]); a = p.parse_args()
    all_rows = claims.rows(); path = os.path.join(env.HERE, "judges.json"); judges = json.load(open(path)) if os.path.exists(path) else []
    speed = {t: tuple(float(x) for x in v.split(",")) for t, v in named(a.time).items()}
    out, win = verdict(named(a.design), named(a.base), all_rows, judges, speed); print(json.dumps(out, indent=1)); print("WIN" if win else "NO WIN")
    rows = [json.loads(l) for l in open(os.path.join(env.HERE, "choices.jsonl"))]; print("designs tried so far:", sum(r.get("kind") in ("rule", "random", "design") for r in rows), "searches,", sum(r.get("evals", 0) for r in rows), "settings in all")


if __name__ == "__main__":
    main()
