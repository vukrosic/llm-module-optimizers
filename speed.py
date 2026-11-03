"""Time per step, one run at a time on one thread, through the real training loop with evals excluded from the clock, because speeds from parallel runs were off by up to 4x in the case study. Rows go to speed.jsonl, away from results.

python3 speed.py [--tier 5s] [--steps 40] <arm or spec> ...
"""
import argparse, json, os, subprocess, sys
import env


def one(name, tier, steps):
    out = "speed.jsonl"; p = subprocess.run([sys.executable, "env.py", "--spec", name, "--tier", tier, "--steps", str(steps), "--evals", "tune", "--threads", "1", "--tag", "speed", "--results", out, "--force"], cwd=env.HERE, capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))
    if p.returncode: raise RuntimeError(p.stderr[-500:])
    return json.loads(open(os.path.join(env.HERE, out)).readlines()[-1])["s_per_step"]


def main():
    p = argparse.ArgumentParser(); p.add_argument("names", nargs="+"); p.add_argument("--tier", default="5s"); p.add_argument("--steps", type=int, default=40); a = p.parse_args()
    for n in a.names: print(f"{n:30} {one(n, a.tier, a.steps):.5f} s per step, load {os.getloadavg()[0]:.1f}", flush=True)


if __name__ == "__main__":
    main()
