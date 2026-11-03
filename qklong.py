"""Does the q,k rule survive longer training? The q,k lr is retuned on tune rows of world 0 at the same tier; gaps use report rows of worlds nothing chose from.

python3 qklong.py [--tier 5sx4] [--seconds 500]
"""
import argparse, os, time
import arms, claims, env, runq, search
from qkrule import gap, say

WORLDS = (15, 16, 17, 18, 19); LRS = dict(lion=.0032, adam=.019)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--tier", default="5sx4"); p.add_argument("--seconds", type=float, default=500); p.add_argument("--results", default=claims.RESULTS); p.add_argument("--choices", default=claims.CHOICES); a = p.parse_args(); a.steps = 0; a.tag = "qklong"; a.moe = 0; a.tied = False; a.qknorm = True
    book = search.Book(os.path.join(env.HERE, a.results)); budget = time.time() + a.seconds; t = search.tuned(a); base = arms.parse(t["muon"]["spec"] if "muon" in t else "muon")
    t0 = search.try_all([base], a, book, budget)[0][0]; print(f"base tune {t0:.4f}, q,k {base['qk']}", flush=True); best = {}
    for r, lr in LRS.items():
        v, s = min(search.try_all([search.with_rule(base, "qk", r, x) for x in search.grid(lr, 5, 2.5)], a, book, budget), key=lambda t: t[0]); best[r] = s; print(f"{r} on q,k adds {v - t0:+.4f} at lr {s['qk']['lr']:g}", flush=True)
    js = [env.parser().parse_args(["--spec", arms.show(s), "--tier", a.tier, "--evals", "full", "--tag", a.tag, "--world", str(w), "--results", a.results]) for s in [base] + list(best.values()) for w in WORLDS]; print("left", runq.run(js, max(1, budget - time.time())), flush=True); rows = claims.rows()
    for r, s in best.items(): say(f"{r} on q,k vs muon on q,k, worlds 15-19", gap(rows, s, base, WORLDS, a))
    if runq.EX: runq.EX.shutdown()


if __name__ == "__main__":
    main()
