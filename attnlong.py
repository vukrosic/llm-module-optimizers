"""Runs picked rules at a tier one world at a time and stops between worlds at a time limit, so a busy machine can't leave workers behind. Rerun to continue; finished rows are skipped.

python3 attnlong.py --rules muonsplit,muongate --worlds 14,15,16 [--tier ax4] [--pick-tier a] [--seconds 400]
"""
import argparse, time
import attnbeat, attnenv

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--rules"); p.add_argument("--worlds"); p.add_argument("--tier", default="ax4"); p.add_argument("--pick-tier", default="a"); p.add_argument("--seconds", type=float, default=400); a = p.parse_args(); t0 = time.time(); S = [attnbeat.spec(r, a.pick_tier) for r in a.rules.split(",")]
    for w in a.worlds.split(","):
        if time.time() - t0 > a.seconds: print("stopped before world", w); break
        attnenv.run_all(S, a.tier, (int(w),), "report"); print(f"world {w} done at {time.time() - t0:.0f} s", flush=True)
