"""Time per step, one run at a time on one thread, through the real training loop with evals excluded from the clock. The fastest of several runs is the number, because other jobs only slow a run. Rows go to speed.jsonl, away from results.

python3 speed.py [--tier 5s] [--steps 40] [--repeat 3] <arm or spec> ...
"""
import argparse, json, os, subprocess, sys
import env


def one(name, tier, steps):
    out = "speed.jsonl"; p = subprocess.run([sys.executable, "env.py", "--spec", name, "--tier", tier, "--steps", str(steps), "--evals", "tune", "--threads", "1", "--tag", "speed", "--results", out, "--force"], cwd=env.HERE, capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))
    if p.returncode: raise RuntimeError(p.stderr[-500:])
    return json.loads(open(os.path.join(env.HERE, out)).readlines()[-1])["s_per_step"]


def main():
    p = argparse.ArgumentParser(); p.add_argument("names", nargs="+"); p.add_argument("--tier", default="5s"); p.add_argument("--steps", type=int, default=40); p.add_argument("--repeat", type=int, default=3); a = p.parse_args()
    for n in a.names:
        v = [one(n, a.tier, a.steps) for _ in range(a.repeat)]; print(f"{n[:60]:60} fastest {min(v):.5f} s per step of {v}, load {os.getloadavg()[0]:.1f}", flush=True)


if __name__ == "__main__":
    main()
