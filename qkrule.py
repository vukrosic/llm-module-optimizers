"""Does q,k want Lion, or just not Muon, and is QK-norm the reason? Picks use tune rows of world 0; gaps use report rows of worlds nothing chose from.

python3 qkrule.py [--tier 5s] [--seconds 280]
"""
import argparse, os, time
import numpy as np
import arms, claims, env, runq, search

WORLDS = (15, 16, 17, 18, 19); OLD = (10, 11, 12, 13, 14); LRS = dict(lion=.0025, adam=.019, rownorm=.04, normsgd=.012)


def jobs(specs, worlds, a, qknorm=True):
    extra = [] if qknorm else ["--no-qknorm"]
    return [env.parser().parse_args(["--spec", arms.show(s), "--tier", a.tier, "--evals", "full", "--tag", "qk", "--world", str(w), "--results", a.results] + extra) for s in specs for w in worlds]


def gap(rows, x, y, worlds, a, qknorm=True):
    out = []
    for w in worlds:
        p, q = (claims.find(rows, arms.show(s), a.tier, w, qknorm=qknorm) for s in (x, y))
        if p and q: d = claims.items(q, "text.web.loss") - claims.items(p, "text.web.loss"); out.append((w, float(d.mean()), float(d.std(ddof=1) / len(d) ** .5)))
    return out


def say(name, g):
    v, m, se = claims.verdict(g); print(f"{name:44} mean {m:+.4f}, error {se:.4f}, {v:8} per world {' '.join(f'{x:+.4f}' for _, x, _ in g)}", flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--tier", default="5s"); p.add_argument("--seconds", type=float, default=280); p.add_argument("--results", default=claims.RESULTS); p.add_argument("--choices", default=claims.CHOICES); a = p.parse_args(); a.steps = 0; a.tag = "qk"; a.moe = 0; a.tied = False; a.qknorm = True
    book = search.Book(os.path.join(env.HERE, a.results)); budget = time.time() + a.seconds; ch = search.choices(a); pick = lambda k: [arms.parse(r["spec"]) for r in ch if r["kind"] == k and r["tier"] == a.tier]
    base = (pick("lronly") or [arms.parse("muon")])[-1]; mix = (pick("mix") or [search.with_rule(base, "qk", "lion", LRS["lion"])])[-1]; alt = {r: search.with_rule(base, "qk", r, lr) for r, lr in LRS.items() if r != "lion"}
    runq.run(jobs([base, mix] + list(alt.values()), WORLDS, a) + jobs([base, mix], OLD, a), a.seconds); rows = claims.rows()
    say("lion on q,k vs muon on q,k, worlds 15-19", gap(rows, mix, base, WORLDS, a))
    for r, s in alt.items(): say(f"{r} on q,k vs muon on q,k, worlds 15-19", gap(rows, s, base, WORLDS, a))
    for r, s in alt.items(): say(f"lion vs {r} on q,k, worlds 15-19", gap(rows, mix, s, WORLDS, a))
    say("lion on q,k vs muon on q,k, worlds 10-14", gap(rows, mix, base, OLD, a))
    a.qknorm = False; best = {}
    for r, lr in (("muon", base["qk"]["lr"]), ("lion", mix["qk"]["lr"])):
        res = search.try_all([search.with_rule(base, "qk", r, x) for x in search.grid(lr, 5, 2.5)], a, book, budget); best[r] = min(res, key=lambda p: p[0]); print(f"no QK-norm, {r} on q,k: tune {best[r][0]:.4f}, lr {best[r][1]['qk']['lr']:g}", flush=True)
    runq.run(jobs([best["muon"][1], best["lion"][1]], WORLDS, a, False), a.seconds); rows = claims.rows()
    say("no QK-norm: lion vs muon on q,k, worlds 15-19", gap(rows, best["lion"][1], best["muon"][1], WORLDS, a, False))
    if runq.EX: runq.EX.shutdown()


if __name__ == "__main__":
    main()
