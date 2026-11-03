"""Does the environment reproduce what papers report? For each pair, chosen on tune rows only: paired difference on report rows, in units of the noise floor.

A difference counts only if its mean over worlds clears 0.005 and 2 errors. Nothing here picks a setting; it only reads what search.py chose. Rows of one setting with every lr changed by 0.1% (--jitter) are printed by testtest.py, but they understate noise by about 7x against worlds, so no verdict uses them.

python3 claims.py [--tier 5s] [--score text.web.loss] [--worlds 0,1,2,3,4]
"""
import argparse, json, os
import numpy as np
import arms, env

PAIRS = [("muon", "all-adam", "Muon recipe beats the standard baseline", "Muon's reports"),
         ("sym-rownorm", "muon", "RowNorm on embedding and head beats Adam there", "2605.18106"),
         ("sym", "sym-rownorm", "hybrid MLP beats Muon MLP", "2605.18106"),
         ("stiefel", "muon", "All-Stiefel beats unconstrained", "2606.13276"),
         ("hetero", "stiefel", "Stiefel attention + DGram MLP beats All-Stiefel", "2606.13276"),
         ("muon", "frozen-mlp", "control: training the MLP beats freezing it", "control"),
         ("muon", "frozen-hidden", "control: training hidden matrices beats freezing them", "control"),
         ("muon", "blind", "control: gradients beat a random direction", "control")]
FLOOR = 0.005; RESULTS = os.environ.get("RESULTS", "results.jsonl"); CHOICES = os.environ.get("CHOICES", "choices.jsonl"); JUDGES = os.environ.get("JUDGES", "judges.json"); SPEED = os.environ.get("SPEED", "speed.jsonl")  # debug scripts set these


def rows():
    return [json.loads(l) for l in open(os.path.join(env.HERE, RESULTS))]


def chosen(tier, moe=0):
    out = {}
    for l in open(os.path.join(env.HERE, CHOICES)):
        r = json.loads(l)
        if r["kind"] == "tune" and r["tier"] == tier and r.get("moe", 0) == moe and r.get("qknorm", True) and not r.get("tied"): out[r["arm"]] = r
    return out


def find(all_rows, spec, tier, world=0, jitter=0, moe=0, lr_mult=1, qknorm=True):
    want = arms.show(arms.parse(spec))
    for r in reversed(all_rows):
        if r["spec"] == want and r["tier"] == tier and r["world"] == world and r["jitter"] == jitter and r["lr_mult"] == lr_mult and r["moe"] == moe and r["evals"] == "full" and r["steps"] == env.TIERS[tier]["steps"] and not r["leak"] and not r["double"] and r["qknorm"] == qknorm and not r["tied"]: return r


def items(r, score):
    return np.load(os.path.join(env.RUNS, r["key"] + ".npz"))[score]


def gaps(all_rows, base, other, tier, score, worlds):
    """other - base per world on paired rows, with its row error, for every world both have run. Positive means base is better."""
    out = []
    for w in worlds:
        x, y = find(all_rows, base, tier, w), find(all_rows, other, tier, w)
        if x and y: d = items(y, score) - items(x, score); out.append((w, float(d.mean()), float(d.std(ddof=1) / len(d) ** .5)))
    return out


def verdict(g, floor=FLOOR):
    """A claim holds if the mean gap over worlds clears the floor and 2 errors, where the error comes from the spread across worlds and the row error, whichever is larger."""
    if len(g) < 2: return "not run", 0.0, 0.0
    v = np.array([x[1] for x in g]); se = max(v.std(ddof=1) / len(v) ** .5, float(np.mean([x[2] for x in g]))); m = float(v.mean()); need = max(floor, 2 * se)
    return ("holds" if m > need else "reversed" if m < -need else "tie"), m, se


def main():
    p = argparse.ArgumentParser(); p.add_argument("--tier", default="5s"); p.add_argument("--score", default="text.web.loss"); p.add_argument("--worlds", default="0,1,2,3,4"); a = p.parse_args()
    all_rows = rows(); ch = chosen(a.tier); ws = [int(x) for x in a.worlds.split(",")]
    print(f"tier {a.tier}, score {a.score}, gaps from worlds {ws}; tune and settings come from world 0")
    print("\narm              tune    report   settings  time/step  unstable   lrs")
    for k in sorted(ch, key=lambda k: ch[k]["tune"]):
        r = find(all_rows, ch[k]["spec"], a.tier)
        if r: print(f"{k:16} {ch[k]['tune']:.4f}  {r['scores'].get(a.score, float('nan')):.4f}   {ch[k]['evals']:5}     {r['s_per_step']:.4f}    {str(r['opt']['unstable']):8} {ch[k]['spec'][:90]}")
    print("\nclaim                                                   source         mean gap  error  verdict   per world")
    for x, y, what, src in PAIRS:
        if x not in ch or y not in ch: print(f"{what:55} {src:14} not tuned"); continue
        g = gaps(all_rows, ch[x]["spec"], ch[y]["spec"], a.tier, a.score, ws); v, m, se = verdict(g)
        print(f"{what:55} {src:14} {m:+.4f}  {se:.4f}  {v:8}  {' '.join(f'{d:+.4f}' for _, d, _ in g)}")
    if "frozen-hidden" in ch and "muon" in ch:
        r, f = find(all_rows, ch["muon"]["spec"], a.tier), find(all_rows, ch["frozen-hidden"]["spec"], a.tier)
        if r and f:
            w = r["weights"]; hid = sum(w[m] for m in ("qk", "vo", "gu", "down")); print(f"\nhidden training is worth {f['scores'][a.score] - r['scores'][a.score]:.3f} on {a.score}, {(f['scores'][a.score] - r['scores'][a.score]) / FLOOR:.0f}x the floor; hidden matrices hold {hid / sum(w.values()):.0%} of the weights")


if __name__ == "__main__":
    main()
