"""Controls before any new rule runs: is training query and key worth anything on each task, and is the same run repeatable? Uses worlds 1, 2 and 3.

python3 attncontrols.py [--tier a]
"""
import argparse
import attnenv, attnread

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--tier", default="a"); a = p.parse_args(); W = (1, 2, 3); M, F, B = "qk=muon:.02", "qk=freeze:0", "qk=blind:.02"
    attnenv.run_all([attnenv.parse(x) for x in (M, F, B)], a.tier, W, "report")
    print("Muon against frozen query and key"); attnread.show(M, F, a.tier, W)
    print("\nMuon against a random direction on query and key"); attnread.show(M, B, a.tier, W)
    x, _, _ = attnenv.run_task("select", attnenv.parse(M), a.tier, 1, "report"); y, _, _ = attnenv.run_task("select", attnenv.parse(M), a.tier, 1, "report"); print("\nsame run twice, largest change on select:", float(abs(x["select.nll"] - y["select.nll"]).max()))
