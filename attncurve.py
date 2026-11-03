"""Plasticity against forgetting on tune rows of world 0, one line per lr. Used to design the next rule, never for a verdict.

python3 attncurve.py
"""
import json, os
import attnenv, attntune

for r in map(json.loads, open(os.path.join(attnenv.HERE, attntune.CHOICES))):
    if r["tier"] != "a": continue
    print(r["rule"], "pick", r["lr"])
    for lr in sorted(float(x) for x in r["rank"]):
        s = attntune.scores(attntune.spec_of(r["rule"], lr, r["rest"]), "a"); print(f"  lr {lr:<8g} " + "  ".join(f"{k.split('.')[0]} {v:6.3f}" for k, v in s.items()))
