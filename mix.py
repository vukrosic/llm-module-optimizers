"""Mixing update rules per module, at a size where each setting takes a few seconds. All picks use tune rows of world 0, and only `fresh` reads report rows.

map: from one base, swap one module to each rule at 3 or 5 lrs. This says which modules care about their rule.
mixes: random mixes of the rules that stayed close to the base, at the lrs the map chose. If a mix scores what its module swaps add up to, rules can be chosen module by module.
fresh: the best mix, after the lr passes the rule search gets, against the base and the other arms on worlds nothing has read.

python3 mix.py [--tier 5s] [--part map,mixes,fresh] [--rules adam,muon] [--n 60] [--seconds 500]
"""
import argparse, json, os, random, time
import numpy as np
import arms, claims, env, runq, search

WORLDS = (15, 16, 17, 18, 19); CLOSE = .03; OTHERS = ("muon", "muon-all", "all-adam", "sym")


def base_of(a):
    r = [x for x in search.choices(a) if x["kind"] == "lronly" and x["tier"] == a.tier]
    return arms.parse(r[-1]["spec"]) if r else arms.parse("muon")


def swaps(a, base):
    """module -> rule -> specs. Anchored rules get 3 lrs, the others 5, the same grids as the rule search."""
    an = search.anchors(a); out = {}
    for m in arms.MODULES:
        out[m] = {}
        for r in search.NORM_RULES if m == "norm" else search.MATRIX_RULES:
            if a.rules and r not in a.rules or r == base[m]["rule"]: continue
            lr = an.get(r, {}).get(m); out[m][r] = [search.with_rule(base, m, r, x) for x in (search.grid(lr, 3, 2.5) if lr else search.grid(arms.LRS[r], 5, 2.5))]
    return out


def table(a, book, budget, base):
    """Runs the map. Returns the base score and module -> rule -> (best score, its lr)."""
    s0 = search.try_all([base], a, book, budget)[0][0]; sw = swaps(a, base); out = {}
    search.try_all([s for m in sw for r in sw[m] for s in sw[m][r]], a, book, budget)
    for m in sw:
        out[m] = {base[m]["rule"]: (s0, base[m]["lr"])}
        for r, specs in sw[m].items():
            v, s = min(((book.tune(search.spec_args(x, a)), x) for x in specs), key=lambda p: p[0]); out[m][r] = (v, s[m]["lr"])
    return s0, out


def show_map(s0, t):
    print(f"base tune {s0:.4f}. Each number is the loss a swap adds, negative is better")
    for m in t:
        rows = sorted(t[m].items(), key=lambda kv: kv[1][0]); print(f"  {m:5} spread of the best 4 {rows[min(3, len(rows) - 1)][1][0] - rows[0][1][0]:.4f}: " + " ".join(f"{r} {v - s0:+.4f}@{lr:g}" for r, (v, lr) in rows))


def mixes(a, book, budget, base, s0, t):
    """Each module picks a rule at random among those within CLOSE of the base. Returns (tune, predicted, spec) sorted by tune."""
    rng = random.Random(0); near = {m: [r for r, (v, _) in t[m].items() if v < s0 + CLOSE] for m in t}; specs = {}
    best = {m: min(t[m], key=lambda r: t[m][r][0]) for m in t}; pick = [best]
    while len(pick) < a.n * 3 and len(specs) < a.n:
        c = pick.pop() if pick else {m: rng.choice(near[m]) for m in t}
        if all(c[m] == base[m]["rule"] for m in t): continue
        s = {k: dict(g) for k, g in base.items()}
        for m in t: s[m] = dict(rule=c[m], lr=t[m][c[m]][1])
        specs[arms.show(s)] = (s, s0 + sum(t[m][c[m]][0] - s0 for m in t))
    res = search.try_all([v[0] for v in specs.values()], a, book, budget)
    out = sorted((v, specs[arms.show(s)][1], s) for v, s in res if v < 90); x, y = np.array([o[1] for o in out]), np.array([o[0] for o in out])
    print(f"{len(out)} mixes finished, {len(res) - len(out)} blew up. Actual against the sum of their swaps: correlation {np.corrcoef(x, y)[0, 1]:.3f}, actual minus sum mean {np.mean(y - x):+.4f}, sd {np.std(y - x):.4f}. Actual tune: best {y[0]:.4f}, median {np.median(y):.4f}, worst {y[-1]:.4f}, base {s0:.4f}; {int((y < s0).sum())} beat the base")
    hid = [{m: o[2][m]["rule"] for m in arms.MODULES} for o in out[:5]]
    for o, h in zip(out[:5], hid): print(f"  {o[0]:.4f} (sum {o[1]:.4f}) {h}")
    return out


def fresh(a, specs):
    js = [env.parser().parse_args(["--spec", arms.show(s), "--tier", a.tier, "--evals", "full", "--tag", "mix", "--world", str(w), "--results", a.results]) for s in specs.values() for w in WORLDS]
    left = runq.run(js, a.seconds); print(f"{len(js)} settings, {left} left"); rows = claims.rows(); names = list(specs)
    for n in names[1:]:
        g = claims.gaps(rows, arms.show(specs[names[0]]), arms.show(specs[n]), a.tier, "text.web.loss", WORLDS); v, m, se = claims.verdict(g)
        print(f"{names[0]} vs {n:10} mean {m:+.4f}, error {se:.4f}, {v:8} per world {' '.join(f'{x:+.4f}' for _, x, _ in g)}")


def main():
    p = argparse.ArgumentParser(); p.add_argument("--tier", default="5s"); p.add_argument("--part", default="map,mixes,fresh"); p.add_argument("--rules", default=""); p.add_argument("--n", type=int, default=60); p.add_argument("--seconds", type=float, default=500)
    p.add_argument("--results", default=claims.RESULTS); p.add_argument("--choices", default=claims.CHOICES); a = p.parse_args(); a.rules = [x for x in a.rules.split(",") if x]; a.steps = 0; a.tag = "mix"; a.moe = 0; a.tied = False; a.qknorm = True; parts = a.part.split(",")
    book = search.Book(os.path.join(env.HERE, a.results)); budget = time.time() + a.seconds; base = base_of(a); s0, t = table(a, book, budget, base); show_map(s0, t)
    if "mixes" in parts or "fresh" in parts:
        out = mixes(a, book, budget, base, s0, t); done = [r for r in search.choices(a) if r["kind"] == "mix" and r["tier"] == a.tier]
        if "fresh" in parts:
            if done: best = arms.parse(done[-1]["spec"])
            else:
                count = set(); v, best = search.tune_lrs(out[0][2], a, book, budget, count, search.PASSES[1:]); search.save(dict(kind="mix", spec=arms.show(best), tune=v, evals=len(count), tier=a.tier, moe=0), a)
            tuned = search.tuned(a); fresh(a, dict(mix=best, lronly=base, **{k: arms.parse(tuned[k]["spec"]) for k in OTHERS if k in tuned}))
    if runq.EX: runq.EX.shutdown()


if __name__ == "__main__":
    main()
