"""Picks one query-key lr per rule on tune rows of world 0, with the same grid for every rule. An lr's rank is its mean rank over the 5 task scores, so no single score's scale can decide. Picks go to attn_choices.jsonl. No report row is read.

python3 attntune.py --rules muon,muonsplit [--tier a]
"""
import argparse, json, os
import numpy as np
import arms, attnenv

GRID = (.125, .25, .5, 1, 2, 4); SCORES = dict(copy="copy.area", select="select.area", latest="latest.area", retain="retain.old.nll", switch="switch.new.area"); TUNE_TASKS = tuple(SCORES); CHOICES = os.environ.get("ATTN_CHOICES", "attn_choices.jsonl")


def scores(spec, tier, world=0, ev="tune"):
    r = attnenv.done(); out = {}
    for t, s in SCORES.items():
        x = r.get(attnenv.key_of(spec, t, tier, world, ev)); out[s] = x["scores"].get(s, float("inf")) if x and x["finite"] else float("inf")
    return out


def ranks(specs, tier):
    t = {k: scores(s, tier) for k, s in specs.items()}; keys = list(t); m = np.array([[t[k][s] for s in SCORES.values()] for k in keys]); r = m.argsort(0).argsort(0).mean(1); return dict(zip(keys, r.tolist()))


def spec_of(rule, lr, rest=""): return attnenv.parse(f"qk={rule}:{lr:g}" + (";" + rest if rest else ""))


def choose(rule, tier, rest="", grid=GRID, cap=None):
    base = arms.LRS[rule]; specs = {round(base * m, 6): spec_of(rule, base * m, rest) for m in grid}; attnenv.run_all(list(specs.values()), tier, (0,), "tune", TUNE_TASKS); rank = ranks(specs, tier); allowed = {k: v for k, v in rank.items() if cap is None or scores(specs[k], tier)["retain.old.nll"] < cap}; best = min(allowed or rank, key=(allowed or rank).get); print(rule, {k: round(v, 2) for k, v in rank.items()}, "pick", best, {s: round(v, 3) for s, v in scores(specs[best], tier).items()}, "edge of grid" if best in (min(specs), max(specs)) else "", flush=True)
    with open(os.path.join(attnenv.HERE, CHOICES), "a") as f: f.write(json.dumps(dict(rule=rule, tier=tier, lr=best, rest=rest, spec=arms.show(specs[best]), rank=rank)) + "\n")
    return best


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--rules"); p.add_argument("--tier", default="a"); p.add_argument("--rest", default=""); p.add_argument("--wider", action="store_true", help="adds two larger lrs for a rule whose pick sat on the top of the grid"); p.add_argument("--cap", type=float, default=None, help="only lrs with retain.old.nll under this can be picked"); a = p.parse_args()
    for r in a.rules.split(","): choose(r, a.tier, a.rest, GRID + (8, 16) if a.wider else GRID, a.cap)
