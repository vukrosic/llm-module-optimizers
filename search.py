"""Tune and search on tune rows only. This file never reads text.web.loss or any other report score; debug/poison.py checks that by scrambling them.

python3 search.py tune muon sym --tier 5s     tune each named baseline's lrs
python3 search.py rule --tier 5s              K1 per-module rule + lr coordinate search from the tuned Muon recipe
python3 search.py lr --tier 5s                K1 control: per-module lr only, same start, same number of evaluations
python3 search.py random --tier 5s            K1 control: random rule + lr per module, same number of evaluations

Every arm gets the same protocol: a coordinate pass over its lr groups with 5 lrs, then two passes with 3 around the current best, so a best lr at a grid's edge moves the grid.
The number of distinct settings run per arm is printed and saved with the choice.
"""
import argparse, json, math, os, random, sys, time
import arms, env, runq

FACTOR = 2.5; RESULTS = "results.jsonl"
MATRIX_RULES = ("adam", "muon", "rownorm", "polar", "leftpolar", "hybrid", "lion", "sgdm", "normsgd")
SPARSE_ROWS = ("emb", "head")


def spec_args(s, a, **k):
    x = env.parser().parse_args([]); x.__dict__.update(tier=a.tier, steps=a.steps, world=0, results=a.results, spec=arms.show(s), tag=a.tag, moe=a.moe, tied=a.tied, qknorm=a.qknorm, **k); return x


class Book:
    """Reads tune scores for finished settings. A setting that blew up, or that the instability rule flags, scores as infinite."""
    def __init__(s, path): s.path = path; s.seen = {}; s.load()
    def load(s):
        if os.path.exists(s.path):
            for l in open(s.path):
                r = json.loads(l); s.seen[r["key"]] = r
    def tune(s, a):
        r = s.seen.get(env.key_of(a, arms.parse(a.spec)))
        if r is None: return None
        v = r["scores"].get("text.tune.loss") if r["finite"] and not r["opt"]["unstable"] else None
        return v if v is not None else math.inf


def round_lr(x): return float(f"{x:.2g}")


def grid(c, n): return [round_lr(c * FACTOR ** (i - n // 2)) for i in range(n)]


def try_all(specs, a, book, budget):
    """Runs the specs and returns (tune score, spec) pairs. Stops the program if the time budget runs out, and a rerun continues."""
    js = [spec_args(s, a) for s in specs]; left = runq.run(js, seconds=max(1, budget - time.time()))
    book.load()
    if left or time.time() > budget: print(f"time budget used, {left} settings left in this batch: rerun the same command", flush=True); sys.exit(3)
    return [(book.tune(j), s) for j, s in zip(js, specs)]


def lr_groups(spec, moe):
    """Published recipes share an lr across hidden matrices with one rule; embedding, head and norm scales keep their own."""
    use = {m: g for m, g in spec.items() if moe or m not in arms.MOE}; out = {}
    for m, g in use.items():
        key = m if m in ("emb", "head", "norm", "router") else g["rule"]
        out.setdefault(key, []).append(m)
    return list(out.values())


def coordinate(spec, moves, a, book, budget, count):
    """moves yields (name, candidates) for one coordinate at a time. Keeps the best on tune rows. count collects distinct settings run."""
    best = None
    for n, cands in moves:
        cs = cands(spec); res = try_all(cs, a, book, budget); count.update(arms.show(s) for s in cs); v, s = min(res, key=lambda p: p[0])
        if best is None or v < best: best, spec = v, s
        print(f"  {n}: tune {v:.4f}  distinct settings {len(count)}", flush=True)
    return best, spec


def tune_moves(spec, moe, passes=3):
    for p, n in enumerate((5, 3, 3)[:passes]):
        for ms in lr_groups(spec, moe):
            if spec[ms[0]]["rule"] == "freeze": continue

            def cands(cur, ms=ms, n=n):
                out = []
                for lr in grid(cur[ms[0]]["lr"], n):
                    s = {m: dict(g) for m, g in cur.items()}
                    for m in ms: s[m]["lr"] = lr
                    out.append(s)
                return out
            yield f"pass {p + 1} lr of {'+'.join(ms)}", cands


def tune(name, a, book, budget):
    count = set(); spec = arms.parse(name); best, spec = coordinate(spec, tune_moves(spec, a.moe), a, book, budget, count)
    save(dict(kind="tune", arm=name, spec=arms.show(spec), tune=best, evals=len(count), tier=a.tier, moe=a.moe, tied=a.tied, qknorm=a.qknorm), a)
    return best, spec


def save(d, a):
    with open(os.path.join(env.HERE, a.choices), "a") as f: f.write(json.dumps(d) + "\n")
    print("chosen:", json.dumps(d), flush=True)


def rule_moves(modules):
    for m in modules:
        rs = ("adam", "norm1d", "sgdm", "lion") if m == "norm" else MATRIX_RULES + (("routernorm",) if m == "router" else ())

        def cands(cur, m=m, rs=rs):
            out = []
            for r in rs:
                for lr in grid(arms.LRS[r], 5):
                    s = {k: dict(g) for k, g in cur.items()}; s[m] = dict(rule=r, lr=lr); out.append(s)
            return out
        yield f"rule+lr of {m}", cands


def lr_moves(modules, n):
    for m in modules:
        def cands(cur, m=m, n=n):
            out = []
            for lr in grid(cur[m]["lr"], n):
                s = {k: dict(g) for k, g in cur.items()}; s[m]["lr"] = lr; out.append(s)
            return out
        yield f"lr of {m}", cands


def mods(a): return [m for m in ("emb", "head", "qk", "vo", "gu", "down", "norm") + (arms.MOE if a.moe else ()) if not (m == "head" and a.tied)]


def start_spec(a):
    rows = [json.loads(l) for l in open(os.path.join(env.HERE, a.choices))]
    rows = [r for r in rows if r["kind"] == "tune" and r["arm"] == "muon" and r["tier"] == a.tier and r["moe"] == a.moe]
    if not rows: raise SystemExit("tune the muon arm first")
    return arms.parse(rows[-1]["spec"])


def rule_search(a, book, budget):
    count = set(); spec = start_spec(a); best, spec = coordinate(spec, list(rule_moves(mods(a))) + list(lr_moves(mods(a), 3)), a, book, budget, count)
    save(dict(kind="rule", spec=arms.show(spec), tune=best, evals=len(count), tier=a.tier, moe=a.moe), a)


def lr_search(a, book, budget, total):
    """Same start and same limit on evaluations as the rule search. Passes repeat around the best lr until the limit or until a pass changes nothing."""
    count = set(); spec = start_spec(a); m = mods(a); n = 7; best = None
    while len(count) < total:
        before = arms.show(spec); v, spec = coordinate(spec, list(lr_moves(m, n)), a, book, budget, count); best = v if best is None else min(best, v); n = 3
        if arms.show(spec) == before: break
    save(dict(kind="lr", spec=arms.show(spec), tune=best, evals=len(count), limit=total, tier=a.tier, moe=a.moe), a)


def random_search(a, book, budget, total):
    r = random.Random(0); m = mods(a); base = start_spec(a); specs = [base]
    for _ in range(total - 1):
        s = {k: dict(g) for k, g in base.items()}
        for k in m:
            rs = ("adam", "norm1d", "sgdm", "lion") if k == "norm" else MATRIX_RULES
            x = r.choice(rs); s[k] = dict(rule=x, lr=round_lr(arms.LRS[x] * FACTOR ** r.uniform(-2, 2)))
        specs.append(s)
    res = try_all(specs, a, book, budget); v, s = min(res, key=lambda p: p[0])
    save(dict(kind="random", spec=arms.show(s), tune=v, evals=len(res), tier=a.tier, moe=a.moe), a)


def main():
    p = argparse.ArgumentParser(); p.add_argument("mode", choices=["tune", "rule", "lr", "random"]); p.add_argument("names", nargs="*"); p.add_argument("--tier", default="5s"); p.add_argument("--steps", type=int, default=0)
    p.add_argument("--moe", type=int, default=0); p.add_argument("--tied", action="store_true"); p.add_argument("--no-qknorm", dest="qknorm", action="store_false"); p.add_argument("--seconds", type=float, default=240); p.add_argument("--evals-total", type=int, default=0)
    p.add_argument("--results", default=RESULTS); p.add_argument("--choices", default="choices.jsonl"); p.add_argument("--tag", default="search"); a = p.parse_args()
    book = Book(os.path.join(env.HERE, a.results)); budget = time.time() + a.seconds
    done = set()
    if os.path.exists(os.path.join(env.HERE, a.choices)):
        for l in open(os.path.join(env.HERE, a.choices)):
            r = json.loads(l); done.add((r["kind"], r.get("arm"), r["tier"], r["moe"], r.get("tied", False), r.get("qknorm", True)))
    if a.mode == "tune":
        for n in a.names:
            if ("tune", n, a.tier, a.moe, a.tied, a.qknorm) in done: continue
            print(n, flush=True); tune(n, a, book, budget)
    elif a.mode == "rule": rule_search(a, book, budget)
    elif a.mode == "lr": lr_search(a, book, budget, a.evals_total or 150)
    else: random_search(a, book, budget, a.evals_total or 150)


if __name__ == "__main__":
    main()
