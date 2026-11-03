"""Does the environment reproduce what papers report? For each pair, chosen on tune rows only: paired difference on report rows, in units of the noise floor.

A difference counts only if it clears 2 sigma of training noise (runs of one setting with every lr changed by 0.1% and up) and z > 2 over rows. Nothing here picks a setting; it only reads what search.py chose.

python3 claims.py [--tier 5s] [--score text.web.loss]
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
         ("muon", "blind", "control: gradients beat a random direction", "control")]


def rows():
    return [json.loads(l) for l in open(os.path.join(env.HERE, "results.jsonl"))]


def chosen(tier, moe=0):
    out = {}
    for l in open(os.path.join(env.HERE, "choices.jsonl")):
        r = json.loads(l)
        if r["kind"] == "tune" and r["tier"] == tier and r.get("moe", 0) == moe and r.get("qknorm", True) and not r.get("tied"): out[r["arm"]] = r
    return out


def find(all_rows, spec, tier, world=0, jitter=0, moe=0):
    want = arms.show(arms.parse(spec))
    for r in reversed(all_rows):
        if r["spec"] == want and r["tier"] == tier and r["world"] == world and r["jitter"] == jitter and r["lr_mult"] == 1 and r["moe"] == moe and not r["leak"] and not r["double"] and r["qknorm"] and not r["tied"]: return r


def items(r, score):
    return np.load(os.path.join(env.RUNS, r["key"] + ".npz"))[score]


def noise(all_rows, tier, score):
    """The standard deviation across jitter runs of the tuned Muon recipe; the wider of the plain spread and the largest gap to its mean over 2."""
    c = chosen(tier).get("muon")
    if not c: return None
    want = arms.show(arms.parse(c["spec"])); v = [r["scores"][score] for r in all_rows if r["spec"] == want and r["tier"] == tier and r["world"] == 0 and r["moe"] == 0 and r["lr_mult"] == 1 and r["scores"].get(score) is not None and not r["leak"] and not r["double"] and r["qknorm"] and not r["tied"]]
    return (float(np.std(v, ddof=1)), len(v)) if len(v) > 2 else None


def main():
    p = argparse.ArgumentParser(); p.add_argument("--tier", default="5s"); p.add_argument("--score", default="text.web.loss"); a = p.parse_args()
    all_rows = rows(); ch = chosen(a.tier); n = noise(all_rows, a.tier, a.score)
    print(f"tier {a.tier}, score {a.score}, training noise sigma {n[0]:.4f} from {n[1]} runs" if n else "no noise floor yet")
    print("\narm              tune    report   settings  time/step  unstable   lrs")
    for k in sorted(ch, key=lambda k: ch[k]["tune"]):
        r = find(all_rows, ch[k]["spec"], a.tier)
        if r: print(f"{k:16} {ch[k]['tune']:.4f}  {r['scores'].get(a.score, float('nan')):.4f}   {ch[k]['evals']:5}     {r['s_per_step']:.4f}    {str(r['opt']['unstable']):8} {ch[k]['spec'][:90]}")
    print("\nclaim                                                   source        gap      z   verdict")
    for x, y, what, src in PAIRS:
        if x not in ch or y not in ch: print(f"{what:55} {src:12} not run"); continue
        rx, ry = find(all_rows, ch[x]["spec"], a.tier), find(all_rows, ch[y]["spec"], a.tier)
        if not rx or not ry: continue
        d = items(ry, a.score) - items(rx, a.score); z = float(d.mean() / (d.std(ddof=1) / len(d) ** .5)); floor = 2 * n[0] if n else 0
        verdict = "holds" if d.mean() > floor and z > 2 else ("reversed" if d.mean() < -floor and z < -2 else "tie")
        print(f"{what:55} {src:12} {d.mean():+.4f} {z:6.1f}  {verdict}")


if __name__ == "__main__":
    main()
