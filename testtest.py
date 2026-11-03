"""Stage 3 job list: every run that tests the test, read from what search.py chose on tune rows. Nothing here picks a setting. Finished settings are skipped, so rerun to continue.

python3 testtest.py [--tier 5s] [--from 5s] [--seconds 280] [--part noise,full,world,lr,cheat,qk] [--results results.jsonl]
"""
import argparse, os, shlex
import claims, env, runq


def job(spec, tier, results, *extra):
    return env.parser().parse_args(["--spec", spec, "--tier", tier, "--evals", "full", "--tag", "test", "--results", results, *extra])


def jobs(ch, a):
    out = []; want = a.part.split(","); use = [n for n in ch]; mu = ch["muon"]["spec"]
    if "noise" in want: out += [job(mu, a.tier, a.results, "--jitter", str(k)) for k in range(0, 5)]
    if "full" in want: out += [job(ch[n]["spec"], a.tier, a.results) for n in use]
    if "world" in want: out += [job(ch[n]["spec"], a.tier, a.results, "--world", str(w)) for n in use for w in (1, 2, 3, 4)]
    if "lr" in want: out += [job(ch[n]["spec"], a.tier, a.results, "--lr-mult", "1.5") for n in use if n != "blind"]
    if "cheat" in want: out += [job(mu, a.tier, a.results, "--leak"), job(mu, a.tier, a.results, "--double")]
    if "qk" in want: out += [job(ch[n]["spec"], a.tier, a.results, "--no-qknorm") for n in ("stiefel", "hetero", "dgram", "muon") if n in ch]
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument("--tier", default="5s"); p.add_argument("--from", dest="src", default=""); p.add_argument("--seconds", type=int, default=280); p.add_argument("--part", default="noise,full,world,lr,cheat,qk"); p.add_argument("--results", default="results.jsonl"); a = p.parse_args()
    ch = claims.chosen(a.src or a.tier); js = jobs(ch, a); left = runq.run(js, a.seconds)
    if runq.EX: runq.EX.shutdown()
    print(f"{len(js)} settings in the list, {left} left" + (": rerun the same command" if left else ""))


if __name__ == "__main__":
    main()
