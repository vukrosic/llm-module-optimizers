"""Reads what Muon's step does around a task change, on tune rows of world 0, so no report row is read. For each 50-step window, over all heads: size, the gradient's part along the rows (should be 0), the step's part along the rows, and two possible signals that the task changed.

undo is the cosine between the gradient and the head's total move so far. Positive means the gradient pushes the head back the way it came.
turn is the cosine between the gradient and the momentum. Low means the gradient left its recent direction.

python3 attnsignal.py [--tier a] [--lr .02]
"""
import argparse, collections
import numpy as np
import torch
import attnenv, rules

LOG = []


def cos(a, b): return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def watch(w, g, st, lr, k):
    key = id(w); first = START.setdefault(key, w.clone()); n = w.shape[0] // HEADS; b0 = st.get("b"); before = w.clone(); rules.muon(w, g, st, lr, k); step = w - before; t = STEP[key] = STEP.get(key, 0) + 1
    for i in range(0, w.shape[0], n):
        h, gh, sh, dh = before[i:i + n], g[i:i + n], step[i:i + n], (before - first)[i:i + n]
        LOG.append(dict(t=t, size=float(h.norm()), g_along=cos(gh, h), step_along=cos(sh, h), undo=cos(gh, dh) if dh.norm() > 0 else 0, turn=cos(gh, b0[i:i + n]) if b0 is not None else 0))


START, STEP, HEADS = {}, {}, 4

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--tier", default="a"); p.add_argument("--lr", type=float, default=.02); a = p.parse_args(); HEADS = attnenv.TIERS[a.tier]["heads"]; rules.RULES["muon_watch"] = watch; rules.ONLY_2D.add("muon_watch")
    import arms; arms.LRS["muon_watch"] = a.lr; steps = attnenv.TIERS[a.tier]["steps"]; win = max(5, steps // 12)
    for task, change in (("retain", int(steps * .75)), ("switch", steps // 2), ("select", None)):
        LOG.clear(); START.clear(); STEP.clear(); out, ok, info = attnenv.run_task(task, attnenv.parse(f"qk=muon_watch:{a.lr:g}"), a.tier, 0, "tune"); by = collections.defaultdict(list)
        for r in LOG: by[(r["t"] - 1) // win].append(r)
        print(f"\n{task}: change at step {change}, ok={ok}, head size spread at end {np.std([r['size'] for r in LOG[-len(LOG) // len(STEP) // max(1, len(by)):]]):.2f}")
        print("  steps        size  g_along step_along   undo   turn")
        for b in sorted(by):
            x = by[b]; print(f"  {b * win + 1:4}-{(b + 1) * win:<5} {np.mean([r['size'] for r in x]):6.2f} {np.mean([r['g_along'] for r in x]):8.3f} {np.mean([r['step_along'] for r in x]):8.3f} {np.mean([r['undo'] for r in x]):7.3f} {np.mean([r['turn'] for r in x]):7.3f}" + ("  <- change" if change and b * win <= change < (b + 1) * win else ""))
