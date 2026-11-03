"""Prints every rule's mean scores over fresh worlds, then each rule's verdict against Muon and per-head Muon. Lower is better.

ATTN_CHOICES=attn_choices_memory.jsonl python3 attntable.py --worlds 9,10,11,12,13 [--tier a] [--pick-tier a]
"""
import argparse, json, os
import numpy as np
import attnbeat, attnenv, attnread, attntune, claims

MAIN = attnbeat.MAIN + ("majority.area",)


def value(rule, s, tier, pick, worlds, rows):
    v = []
    for w in worlds:
        x = rows.get(attnenv.key_of(attnbeat.spec(rule, pick), s.split(".")[0], tier, w, "report"))
        if x and x["finite"]: v.append(x["scores"][s])
    return v


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--worlds", default="9,10,11,12,13"); p.add_argument("--tier", default="a"); p.add_argument("--pick-tier", default="a"); p.add_argument("--rules", default="muon,muonsplit,adam,lion,normsgd,adamtangent,muontangent,muonsphere,muonhistory"); a = p.parse_args(); W = tuple(int(x) for x in a.worlds.split(",")); rows = attnenv.done(); rules = a.rules.split(",")
    print(f"{'rule':12} {'lr':>7}" + "".join(f"{s:>17}" for s in MAIN))
    for r in rules:
        row = f"{r:12} {[x for x in map(json.loads, open(os.path.join(attnenv.HERE, attntune.CHOICES))) if x['rule'] == r and x['tier'] == a.pick_tier][-1]['lr']:7g}"
        for s in MAIN:
            v = value(r, s, a.tier, a.pick_tier, W, rows); row += f"{np.mean(v):10.3f} ({np.std(v, ddof=1) / len(v) ** .5:.3f})" if len(v) > 1 else f"{'':17}"
        print(row)
    print()
    for r in rules:
        for b in attnbeat.BASES:
            if r == b: continue
            t = attnread.table(attnbeat.spec(r, a.pick_tier), attnbeat.spec(b, a.pick_tier), a.tier, W); out = []
            for s in MAIN:
                if s in t: v, m, se = claims.verdict(t[s], attnbeat.FLOOR); out.append(f"{s.split('.')[0]} {m:+.3f} {v[:3]}")
            print(f"{r:12} vs {b:10} " + "  ".join(out))
