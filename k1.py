"""K1, the kill test, read and scored in code written before it ran. Does a per-module rule search beat the best recipe with one lr per module?

Control: the best tuned recipe, chosen on tune rows of world 0. Search: search.py rule from that recipe. Both are scored on report rows of fresh worlds 10 to 14.
The idea is killed at a tier if mean gap is under max(0.005, 2 errors); it goes forward if the gap clears that at both 5s and 1m. A menu without manifold rules is one result, a menu with them is another.

python3 k1.py --tier 5s --run --seconds 280     run the fresh worlds
python3 k1.py --tier 5s                         print the result
"""
import argparse
import arms, beat, claims, env, runq


def main():
    p = argparse.ArgumentParser(); p.add_argument("--tier", default="5s"); p.add_argument("--run", action="store_true"); p.add_argument("--seconds", type=int, default=280); p.add_argument("--manifold", action="store_true"); p.add_argument("--kind", default="rule"); p.add_argument("--against", default=""); a = p.parse_args()
    ch = beat.load(); d, b = beat.design_of(a.tier, ch, a.kind, a.manifold), (beat.design_of(a.tier, ch, a.against) if a.against else beat.base_of(a.tier, ch))
    if not d or not b: raise SystemExit("search first")
    if a.run:
        js = [env.parser().parse_args(["--spec", s, "--tier", a.tier, "--evals", "full", "--tag", "k1", "--world", str(w), "--results", claims.RESULTS]) for s in (d["spec"], b["spec"]) for w in beat.WORLDS]
        left = runq.run(js, a.seconds)
        if runq.EX: runq.EX.shutdown()
        print(f"{len(js)} settings, {left} left")
    g = claims.gaps(claims.rows(), d["spec"], b["spec"], a.tier, "text.web.loss", beat.WORLDS); v, m, se = claims.verdict(g)
    sp = arms.parse(d["spec"]); print(f"tier {a.tier}: control {b.get('arm', a.against)}, {b['evals']} settings, tune {b['tune']}; {a.kind}, {d['evals']} settings, tune {d['tune']}"); print("design:", d["spec"]); print("rules on hidden modules:", {m: sp[m]["rule"] for m in beat.HIDDEN})
    print(f"gap on report rows, positive means the search wins: mean {m:+.4f}, error {se:.4f}, per world {' '.join(f'{x:+.4f}' for _, x, _ in g)}: {v}")


if __name__ == "__main__":
    main()
