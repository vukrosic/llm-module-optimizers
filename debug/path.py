"""Do the chain of scripts work, and can the win rule say both yes and no? Tiny size, choices written by hand: k1, claims, audit, final, then beat on three pairs.

A self-tie must lose on the sign rule. A planted win (a mixed design against the blind control as base) must fail on nothing but the hidden-training size, which tiny can't reach, and state, because the blind control holds none. A recipe must fail the mixed rule. A second final must be refused.
"""
import json, os, subprocess, sys, tempfile, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); d = tempfile.mkdtemp(); rel = lambda n: os.path.relpath(os.path.join(d, n), HERE)
os.environ.update(RESULTS=rel("r.jsonl"), CHOICES=rel("c.jsonl"), JUDGES=rel("j.json"), SPEED=rel("s.jsonl"), FINAL_LOG=rel("f.jsonl"), FINAL_RESULTS=rel("fr.jsonl"), PAR="6", LOAD="99", OMP_NUM_THREADS="1")
sys.path.insert(0, HERE); os.chdir(HERE)
import arms, beat, claims, env, runq


def sh(*c, ok=(0,)):
    t = time.time(); p = subprocess.run([sys.executable, *c], capture_output=True, text=True, timeout=60); assert p.returncode in ok, p.stdout[-400:] + p.stderr[-800:]
    print(f"{c[0]} {' '.join(c[1:3])}: {time.time() - t:.1f}s", flush=True); return p.stdout


def main():
    show = lambda n: arms.show(arms.parse(n)); T = "tiny"; mixed = {**arms.parse("muon"), "gu": dict(rule="adam", lr=.003)}
    ch = [dict(kind="tune", arm=n, spec=show(n), tune=t, evals=1, tier=T, moe=0, tied=False, qknorm=True) for n, t in (("muon", 5.), ("all-adam", 5.1), ("sym", 5.2), ("rownorm-all", 5.3), ("blind", 9.), ("frozen-hidden", 8.))] + [dict(kind="rule", start="muon", manifold=False, spec=arms.show(mixed), tune=4.9, evals=9, tier=T, moe=0)]
    with open(claims.CHOICES, "w") as f: f.write("\n".join(map(json.dumps, ch)) + "\n")
    assert beat.base_of(T, ch)["arm"] == "muon" and beat.design_of(T, ch, "rule")["spec"] == arms.show(mixed)
    print(sh("k1.py", "--tier", T, "--run", "--seconds", "30").splitlines()[-1][:160])
    # audit, claims and beat read these: rows of the other arms in world 0 and at 1.5x lr, world 10 for the blind and frozen controls
    js = [env.parser().parse_args(["--spec", show(n), "--tier", T, "--evals", "full", "--world", str(w), "--lr-mult", str(m), "--results", claims.RESULTS]) for n in ("muon", "all-adam", "sym", "rownorm-all", "blind", "frozen-hidden") for w, m in ((0, 1), (0, 1.5), (1, 1))]
    js += [env.parser().parse_args(["--spec", show("blind"), "--tier", T, "--evals", "full", "--world", str(w), "--results", claims.RESULTS]) for w in beat.WORLDS]
    t = time.time(); left = runq.run(js, 40); runq.EX.shutdown(); assert left == 0; print(f"{len(js)} control rows: {time.time() - t:.1f}s", flush=True)
    print(sh("claims.py", "--tier", T, "--worlds", "10,11,12,13,14").splitlines()[-1][:120])
    print(sh("audit.py", "--tier", T, "--next-tier", T, "--world", "1").splitlines()[-1])
    assert not {"text.web.loss", "text.tune.loss", "memorize.train.loss"} & set(json.load(open(claims.JUDGES)))
    with open(claims.SPEED, "w") as f:
        for s, v in ((show("muon"), .01), (show("blind"), .01), (arms.show(mixed), .011)): f.write("".join(json.dumps(dict(spec=s, tier=T, s_per_step=v)) + "\n" for _ in range(3)))
    rows, judges = claims.rows(), json.load(open(claims.JUDGES)); beat.TIERS = (T, T); design = beat.design_of(T, ch, "rule"); muon = beat.base_of(T, ch); blind = next(r for r in ch if r.get("arm") == "blind")
    tie = beat.judge(dict(design, spec=muon["spec"]), muon, T, rows, judges, 1); assert not tie["win"] and any("one rule" in w for w in tie["why"]) and any("sign" in w for w in tie["why"]), tie
    fake = beat.judge(design, blind, T, rows, judges, 1); left = [w for w in fake["why"] if "hidden training" not in w and "state" not in w]; assert fake["mean"] > .1 and not left, fake
    print(f"win rule: self-tie refused, planted win {fake['mean']:+.3f} passes all but the state and hidden-size rules", flush=True)
    print(sh("beat.py", "--design", "rule", "--tiers", "tiny,tiny").splitlines()[-2])
    print(sh("final.py", "--tier", T).strip()[:200]); p = subprocess.run([sys.executable, "final.py", "--tier", T], capture_output=True, text=True); assert p.returncode and "already read" in p.stderr, p.stderr
    print("second final refused\nall path checks pass")


if __name__ == "__main__":
    main()
