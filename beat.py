"""The win rule, written before any new design runs. A per-module assignment wins only if all of this holds against the best tuned recipe at each tier, read from choices.jsonl, never typed.

Fairness: every arm trains the same model and weights on the same tokens in the same order with one gradient per step. Every arm has one lr per module, tuned on tune rows of world 0 by search.py. A design must come from choices.jsonl, so lr splitting alone can't win: the base has the same splits.

1. Fresh worlds. Report rows of worlds 10 to 14, which no search reads. The gap has the same sign in all 5.
2. Size. Mean gap over worlds exceeds max(0.005, t x error), where error is the larger of the spread across worlds over root 5 and the row error, and t is Student's with 4 degrees of freedom at 2.5% divided by the number of designs tried.
3. Mixed. The design uses at least 2 rules on hidden modules, or it is a recipe with a single rule and not a per-module result.
4. Length and size. Every tier in TIERS passes with its own tuning. At 5sx4 the gap keeps at least half of its 5s value, because a 260-step gain in muon/rownorm-parts did not hold at 1200 steps. A tier counts only if hidden training is worth at least 10x the floor there.
5. Compute. Same tokens, one gradient per step, time per step at most 1.25x the base, from the fastest of at least 3 speed.py runs each, and state at most 2x.
6. Stability. No instability flag in any world.
7. Other scores. Among scores that judges.json admits, wins at least equal losses over the 5 worlds, and every loss is printed. Without judges.json the rule refuses to run.
8. Hidden rows. After all designs, the final design and base run once with --final in world 20, and the sign must hold there. final.py does this and refuses a second run.
If it comes close, the result stays "no win" and the lost scores go in the report.

python3 beat.py --design rule [--manifold]
"""
import argparse, json, os
import numpy as np
from scipy import stats
import arms, claims, env

WORLDS = (10, 11, 12, 13, 14); TIERS = ("5s", "5sx4", "1m"); MIN_GAP = 0.005; COMPUTE = 1.25; STATE = 2.0; ALPHA = 0.05; HIDDEN = ("qk", "vo", "gu", "down"); CONTROLS = ("blind", "frozen-mlp", "frozen-hidden")
LOWER = (".loss", ".nll", ".delta"); HIGHER = (".acc", ".pick", ".gain", ".margin", ".over-chance")


def better(score, d):
    """d = base - design as scores. Positive means the design is better."""
    if score.endswith(LOWER): return d
    if score.endswith(HIGHER): return -d
    raise ValueError(f"{score} has no direction")


def load():
    return [json.loads(l) for l in open(os.path.join(env.HERE, claims.CHOICES))]


def base_of(tier, ch):
    t = [r for r in ch if r["kind"] == "tune" and r["tier"] == tier and r.get("moe", 0) == 0 and r.get("qknorm", True) and not r.get("tied") and r["arm"] not in CONTROLS]
    return min(t, key=lambda r: r["tune"]) if t else None


def design_of(tier, ch, kind, manifold=False):
    t = [r for r in ch if r["kind"] == kind and r["tier"] == tier and r.get("moe", 0) == 0 and r.get("manifold", False) == manifold]
    return t[-1] if t else None


def tried(ch):
    return max(1, sum(r["kind"] in ("rule", "random", "design") for r in ch))


def speed(spec, tier):
    p = os.path.join(env.HERE, claims.SPEED); want = arms.show(arms.parse(spec))
    v = [r["s_per_step"] for r in map(json.loads, open(p)) if r["spec"] == want and r["tier"] == tier] if os.path.exists(p) else []
    return min(v) if len(v) >= 3 else None


def judge(d, b, tier, rows, judges, n):
    gaps, why, wins, losses = [], [], {}, {}
    if len({arms.parse(d["spec"])[m]["rule"] for m in HIDDEN}) < 2: why.append("one rule on all hidden modules, so this is a recipe, not a per-module result")
    for w in WORLDS:
        x, y = claims.find(rows, b["spec"], tier, w), claims.find(rows, d["spec"], tier, w)
        if not x or not y: why.append(f"world {w} not run"); continue
        if not y["finite"] or y["opt"]["unstable"]: why.append(f"unstable in world {w}: {y['opt'].get('why')}"); continue
        if y["tokens"] != x["tokens"] or y["grads_per_step"] != 1: why.append(f"tokens or gradients differ in world {w}")
        if y["state_bytes"] > STATE * x["state_bytes"]: why.append(f"state {y['state_bytes']} is over {STATE}x base {x['state_bytes']}")
        a, c = np.load(os.path.join(env.RUNS, x["key"] + ".npz")), np.load(os.path.join(env.RUNS, y["key"] + ".npz")); g = a["text.web.loss"] - c["text.web.loss"]; gaps.append((float(g.mean()), float(g.std(ddof=1) / len(g) ** .5)))
        for s in judges:
            if s not in a or len(a[s]) < 30: continue
            e = better(s, a[s] - c[s]); z = float(e.mean() / (e.std(ddof=1) / len(e) ** .5 + 1e-12))
            if z > 2: wins[s] = wins.get(s, 0) + 1
            if z < -2: losses.setdefault(s, []).append(round(float(e.mean()), 4))
    out = dict(design=d["spec"], base=b.get("arm"), base_spec=b["spec"], settings=dict(design=d["evals"], base=b["evals"]), wins=wins, losses=losses)
    if len(gaps) == len(WORLDS):
        v = np.array([g[0] for g in gaps]); se = max(v.std(ddof=1) / len(v) ** .5, float(np.mean([g[1] for g in gaps]))); t = float(stats.t.ppf(1 - ALPHA / 2 / n, len(v) - 1)); need = max(MIN_GAP, t * se)
        out.update(gaps=[round(float(x), 4) for x in v], mean=round(float(v.mean()), 4), error=round(se, 4), need=round(need, 4))
        if v.min() <= 0: why.append("gap changes sign across worlds")
        if v.mean() <= need: why.append(f"mean gap {v.mean():.4f} is not above {need:.4f}")
    if sum(wins.values()) < sum(len(v) for v in losses.values()): why.append(f"loses more judged scores than it wins: {losses}")
    a, c = speed(b["spec"], tier), speed(d["spec"], tier)
    if a is None or c is None: why.append("time per step not measured 3 times by speed.py")
    elif c > COMPUTE * a: why.append(f"time per step {c:.4f} is over {COMPUTE}x base {a:.4f}")
    frozen = next((r for r in load() if r["kind"] == "tune" and r["tier"] == tier and r.get("arm") == "frozen-hidden"), None); x = claims.find(rows, b["spec"], tier); f = claims.find(rows, frozen["spec"], tier) if frozen else None
    if not f or not x: why.append("frozen-hidden control not run")
    elif f["scores"]["text.web.loss"] - x["scores"]["text.web.loss"] < 10 * MIN_GAP: why.append(f"hidden training is worth only {f['scores']['text.web.loss'] - x['scores']['text.web.loss']:.3f} at this tier")
    out.update(win=not why, why=why); return out


def verdict(kind, manifold, rows, ch, judges):
    n = tried(ch); out = {}
    for t in TIERS:
        d, b = design_of(t, ch, kind, manifold), base_of(t, ch); out[t] = judge(d, b, t, rows, judges, n) if d and b else dict(win=False, why=["tier not run"])
    a, b = out[TIERS[0]].get("mean"), out[TIERS[1]].get("mean")
    if a and b and b < .5 * a: out[TIERS[1]]["win"] = False; out[TIERS[1]]["why"].append(f"gap faded with length: {a:.4f} at {TIERS[0]}, {b:.4f} at {TIERS[1]}")
    return out, all(v["win"] for v in out.values())


def main():
    global TIERS
    p = argparse.ArgumentParser(); p.add_argument("--design", default="rule"); p.add_argument("--manifold", action="store_true"); p.add_argument("--tiers", default=",".join(TIERS)); a = p.parse_args(); TIERS = tuple(a.tiers.split(","))
    path = os.path.join(env.HERE, claims.JUDGES)
    if not os.path.exists(path): raise SystemExit("run audit.py first: the rule needs judges.json")
    ch = load(); out, win = verdict(a.design, a.manifold, claims.rows(), ch, json.load(open(path))); print(json.dumps(out, indent=1)); print("WIN" if win else "NO WIN")
    print("designs tried:", tried(ch), "; settings in all:", sum(r.get("evals", 0) for r in ch))


if __name__ == "__main__":
    main()
