"""Can one rule with one knob, Muon to normsgd, replace the mix? Picks use tune rows of world 0; gaps use report rows of worlds nothing chose from.

python3 softrule.py [--tier 5s] [--seconds 500]
"""
import argparse, os, time
import arms, claims, env, runq, search

from qkrule import gap, say

WORLDS = (15, 16, 17, 18, 19); NEW = (21, 22, 23); PS = (.1, .25, .5, .75); HIDDEN = ("qk", "vo", "gu", "down"); ONE = ("muon-all", "lion-all", "rownorm-all", "normsgd-all")

def soft(s, m, p, lr):
    s = {k: dict(g) for k, g in s.items()}; s[m] = dict(rule="softmuon", lr=lr, p=p); return s

def jobs(specs, worlds, a):
    return [env.parser().parse_args(["--spec", arms.show(s), "--tier", a.tier, "--evals", "full", "--tag", "soft", "--world", str(w), "--results", a.results]) for s in specs for w in worlds]

def main():
    p = argparse.ArgumentParser(); p.add_argument("--tier", default="5s"); p.add_argument("--seconds", type=float, default=500); p.add_argument("--results", default=claims.RESULTS); p.add_argument("--choices", default=claims.CHOICES); p.add_argument("--steps", type=int); p.add_argument("--tag", default="soft"); a = p.parse_args(); a.moe = 0; a.tied = False; a.qknorm = True
    book = search.Book(os.path.join(env.HERE, a.results)); budget = time.time() + a.seconds; ch = search.choices(a); pick = lambda k: [arms.parse(r["spec"]) for r in ch if r["kind"] == k and r["tier"] == a.tier]
    base = (pick("lronly") or [arms.parse("muon")])[-1]; lion = (pick("mix") or [search.with_rule(base, "qk", "lion", .0025)])[-1]; t0 = search.try_all([base], a, book, budget)[0][0]; print(f"lr-only tune {t0:.4f}", flush=True)
    print(f"control, p 0 on q,k adds {search.try_all([soft(base, 'qk', 0.0, base['qk']['lr'])], a, book, budget)[0][0] - t0:+.4f}", flush=True); best = {}
    for m in HIDDEN:
        for x in PS:
            v, s = min(search.try_all([soft(base, m, x, lr) for lr in search.grid(base[m]["lr"], 5, 2.5)], a, book, budget), key=lambda r: r[0]); best[m, x] = (v, s[m]["lr"]); print(f"{m:5} p {x:<5} adds {v - t0:+.4f} at lr {s[m]['lr']:g}", flush=True)
    single = {}
    for x in PS:
        s = base
        for m in HIDDEN: s = soft(s, m, x, best[m, x][1])
        single[x] = (search.try_all([s], a, book, budget)[0][0], s); print(f"p {x} on all hidden: tune {single[x][0]:.4f}", flush=True)
    x = min(single, key=lambda k: single[k][0]); n = set(); v, one = search.tune_lrs(single[x][1], a, book, budget, n, search.PASSES[1:]); search.save(dict(kind="softone", tier=a.tier, moe=0, p=x, tune=v, evals=len(n), spec=arms.show(one)), a)
    q = min(PS, key=lambda k: best["qk", k][0]); qk = soft(base, "qk", q, best["qk", q][1]); print(f"best p on q,k {q}", flush=True)
    tuned = search.tuned(a); ones = {k: arms.parse(tuned[k]["spec"]) for k in ONE if k in tuned}
    left = runq.run(jobs([base, lion, qk, one] + list(ones.values()), WORLDS, a) + jobs([base, lion, one], NEW, a), max(1, budget - time.time())); rows = claims.rows(); print("left", left, flush=True)
    say("soft on q,k vs lr-only, worlds 15-19", gap(rows, qk, base, WORLDS, a)); say("lion on q,k vs soft on q,k, worlds 15-19", gap(rows, lion, qk, WORLDS, a)); say(f"mix vs one p {x} on hidden, worlds 15-19", gap(rows, lion, one, WORLDS, a)); say(f"one p {x} on hidden vs lr-only, worlds 15-19", gap(rows, one, base, WORLDS, a))
    for k, s in ones.items(): say(f"mix vs {k}, worlds 15-19", gap(rows, lion, s, WORLDS, a))
    say("mix vs lr-only, worlds 21-23", gap(rows, lion, base, NEW, a)); say(f"mix vs one p {x} on hidden, worlds 21-23", gap(rows, lion, one, NEW, a))
    if runq.EX: runq.EX.shutdown()

if __name__ == "__main__":
    main()
