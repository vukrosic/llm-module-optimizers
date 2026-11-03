"""Runs the rules picked in attn_choices.jsonl on report rows of the fresh worlds, one training run per setting.

python3 attnrun.py --rules muon,muonsphere --worlds 4,5,6,7,8 [--tier a] [--pick-tier a]
"""
import argparse
import attnbeat, attnenv

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--rules"); p.add_argument("--worlds", default="4,5,6,7,8"); p.add_argument("--tier", default="a"); p.add_argument("--pick-tier", default=None); a = p.parse_args()
    attnenv.run_all([attnbeat.spec(r, a.pick_tier or a.tier) for r in a.rules.split(",")], a.tier, tuple(int(x) for x in a.worlds.split(",")), "report")
