"""Tunes the lrs of every module except query and key once, on tune rows of world 0, with Muon on query and key. All arms then share this choice, which is the only place Muon gets an advantage, so the report says so.

python3 attnrest.py [--tier a]
"""
import argparse, itertools, os
import numpy as np
import attnenv, attntune

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--tier", default="a"); a = p.parse_args()
    specs = {(x, y): attnenv.parse(f"qk=muon:.02;emb=adam:{x};head=adam:{x};norm=adam:{x};vo=muon:{y};gu=muon:{y};down=muon:{y}") for x, y in itertools.product((.003, .01, .03), (.005, .01, .02))}
    r = attnenv.run_all(list(specs.values()), a.tier, (0,), "tune", attntune.TUNE_TASKS); rank = attntune.ranks(specs, a.tier)
    for k in sorted(rank, key=rank.get): print(k, round(rank[k], 2), {s: round(v, 3) for s, v in attntune.scores(specs[k], a.tier).items()})
