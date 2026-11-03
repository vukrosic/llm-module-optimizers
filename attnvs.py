"""Paired verdict of one spec against another on any worlds, for a comparison outside the picks in attn_choices. Positive means the first spec is better.

python3 attnvs.py "qk=muongate:.04" "qk=muonsplit:.02" --worlds 14,15,16,17,18 [--tier a]
"""
import argparse
import attnenv, attnread, attntune, claims

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("a"); p.add_argument("b"); p.add_argument("--worlds", default="14,15,16,17,18"); p.add_argument("--tier", default="a"); a = p.parse_args(); W = tuple(int(x) for x in a.worlds.split(",")); x, y = attnenv.parse(a.a), attnenv.parse(a.b)
    attnenv.run_all([x, y], a.tier, W, "report"); t = attnread.table(x, y, a.tier, W); print(a.a, "vs", a.b)
    for s in tuple(attntune.SCORES.values()) + ("majority.area",): v, m, e = claims.verdict(t[s], .02); print(f"  {s:18} {m:+.3f} ({e:.3f}) {v}")
