"""The win rule for a new attention rule, written before any new rule runs.

Fairness: every arm trains the same model from the same first weights on the same tokens in the same order, with one gradient per step. Only the query and key rule changes. Each rule gets one lr, from the same grid, on tune rows of world 0. The other modules' lrs were tuned once with Muon on query and key, so any tilt favours Muon.

A rule wins a score if, against Muon and against per-head Muon, the mean gap on report rows of worlds 4 to 8 clears .02 and 2 errors (claims.verdict). A rule wins overall only if all of this holds:
1. It wins at least 3 of the 5 main scores: copy.area, select.area, latest.area, retain.old.nll, switch.new.area.
2. It is reversed on none of them, by either baseline.
3. majority.area, the task frozen query and key also solve, ties.
4. At 4x the steps and at 2x the width, no main score it won is reversed. At 4x the steps at least half of those wins still hold.
5. Time per step is at most 1.25x Muon's, from one run at a time.
6. It stays finite in every world.
If it comes close, the result is "no win" and the lost scores go in the report. Designs tried are counted in attn_choices.jsonl and reported.

python3 attnbeat.py --rule muonsphere --tier a
"""
import argparse, json, os
import attnenv, attnread, attntune, claims

WORLDS = (4, 5, 6, 7, 8); MAIN = tuple(attntune.SCORES.values()); FLOOR = .02; BASES = ("muon", "muonsplit")


def spec(rule, tier):
    c = [r for r in map(json.loads, open(os.path.join(attnenv.HERE, attntune.CHOICES))) if r["rule"] == rule and r["tier"] == tier]
    return attnenv.parse(c[-1]["spec"]) if c else None


def against(rule, base, tier, worlds=WORLDS):
    t = attnread.table(spec(rule, tier), spec(base, tier), tier, worlds); return {s: claims.verdict(g, FLOOR) + (g,) for s, g in t.items()}


def judge(rule, tier, worlds=WORLDS):
    out = {b: against(rule, b, tier, worlds) for b in BASES}; wins = [s for s in MAIN if all(out[b].get(s, ("not run",))[0] == "holds" for b in BASES)]; lost = [s for s in MAIN if any(out[b].get(s, ("not run",))[0] == "reversed" for b in BASES)]; return out, wins, lost


def show(rule, tier, worlds=WORLDS):
    out, wins, lost = judge(rule, tier, worlds); print(f"\n{rule} at tier {tier}, worlds {list(worlds)}")
    for s in MAIN + ("majority.area", "retain.learn.area", "retain.drop", "copy.near.area", "copy.far.area"):
        print(f"  {s:18}", "  ".join(f"{b} {out[b][s][1]:+.3f} ({out[b][s][2]:.3f}) {out[b][s][0]:8}" for b in BASES if s in out[b]))
    print("  wins", wins, "reversed", lost); return wins, lost


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--rule"); p.add_argument("--tier", default="a"); a = p.parse_args(); show(a.rule, a.tier)
