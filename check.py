"""Bug checks, each stopped at 7 s. Any failure raises. Run after every code change.

Asks: can the model see the future? Does every weight get a gradient? Does each rule take the step its paper says? Are test rows in training?
Are two runs of one setting identical? Does every arm run on every model? Do the leak and double-compute cheats show up where they should?
"""
import argparse, json, os, signal, subprocess, sys, tempfile, time
import numpy as np, torch
import arms, env, evals, rules

HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(1)


class stage:
    def __init__(s, name): s.name = name
    def __enter__(s): s.t = time.time(); signal.signal(signal.SIGALRM, lambda *_: sys.exit(f"{s.name} ran over 7 s")); signal.alarm(7); print(s.name, end=" ... ", flush=True)
    def __exit__(s, kind, *rest):
        signal.alarm(0)
        if not kind: print(f"ok {time.time() - s.t:.1f}s", flush=True)


def args(**k):
    a = env.parser().parse_args([]); a.tier = "tiny"; a.__dict__.update(k); return a


C = dict(env.TIERS["tiny"]); out = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
os.environ.setdefault("OMP_NUM_THREADS", "1")

with stage("world rows"):
    W = env.World(C, C["steps"]); assert len(W.rows) >= C["steps"] * C["batch"], "not enough rows for the steps"
    rows = {r.tobytes() for r in W.rows}; assert not any(r.tobytes() in rows for r in np.concatenate([W.val, W.tune, W.hid])), "test rows are in training"
    v, t, h = ({r.tobytes() for r in x} for x in (W.val, W.tune, W.hid)); assert not (v & t or v & h or t & h), "report, tune and hidden rows overlap"
    W1 = env.World(C, C["steps"], 1); assert not (W.rows == W1.rows[:len(W.rows)]).all() and not (W.val == W1.val).all(), "world 1 equals world 0"
    assert not ({r.tobytes() for r in W1.val} & (v | t | h)), "world 1 reads world 0's held-out rows"
    assert env.STRIDE >= C["steps"] * C["batch"] * C["L"], "worlds overlap in training text"

with stage("model: no future, every weight learns"):
    for kw in (dict(), dict(moe=3), dict(tied=True), dict(qknorm=False)):
        torch.manual_seed(0); m = env.LM(C, W.V, **kw); x = torch.as_tensor(W.look[W.rows[:2]]); y = x.clone(); y[:, -1] = (y[:, -1] + 1) % W.V
        assert (m(x)[:, :-1] - m(y)[:, :-1]).abs().max() < 1e-5, "a later token changes earlier scores"
        import torch.nn.functional as F
        big = torch.as_tensor(W.look[W.rows[:32]]); F.cross_entropy(m(big)[:, :-1].flatten(0, 1), big[:, 1:].flatten()).backward()
        none = [n for n, w in m.named_parameters() if w.grad is None or (w.grad == 0).all()]; assert not none, f"no gradient: {none}"
        for n, w in m.named_parameters(): assert arms.module_of(n, w)

with stage("each rule takes its published step"):
    torch.manual_seed(0); w0 = torch.randn(32, 12); g = torch.randn(32, 12); lr = .01
    def step(rule, shape_w=w0, grad=g, **k):
        w = shape_w.clone(); rules.RULES[rule](w, grad, {}, lr, k); return (w - shape_w) / -lr
    for r in ("muon", "normsgd", "blind"):
        n = float(step(r).norm()); assert abs(n - 32 ** .5) / 32 ** .5 < .25, f"{r}: Frobenius {n}, wanted about {32 ** .5}"
    d = step("muon"); s = torch.linalg.svdvals(d); assert (s - 32 ** .5 / 12 ** .5).abs().max() < 1.2, f"muon singular values {s}"
    assert (step("rownorm").norm(dim=1) - 1).abs().max() < 1e-4, "rownorm rows should have norm 1"
    assert (step("hybrid").norm(dim=1) - 1).abs().max() < 1e-4, "hybrid rows should have norm 1"
    assert (step("rownorm", t=True).norm(dim=0) - 1).abs().max() < 1e-4, "rownorm on columns"
    p0 = step("polar", alpha=0.0); M = g * .05; U, S, Vt = torch.linalg.svd(M, full_matrices=False); assert (p0 - U @ Vt).abs().max() < 1e-2, "polar with alpha 0 should be the polar factor"
    p1 = step("polar", alpha=1.0); assert abs(float(p1.norm() / p0.norm()) - float(S.sum())) < 1e-2, "polar with alpha 1 should scale by the nuclear norm"
    wr = torch.randn(32, 8); gr = torch.randn(32, 8); z = step("routernorm", shape_w=wr, grad=gr); assert z.mean(0).abs().max() < 1 and float((z.sum(1)).abs().max()) < 1e3
    assert (step("lion").abs() - 1).abs().max() < 1e-3, "lion should be a sign"
    w1 = torch.randn(12); assert float(step("norm1d", shape_w=w1, grad=torch.randn(12)).norm()) > 0
    assert (step("freeze") == 0).all(), "freeze moved"
    w = w0.clone(); st = {}
    for _ in range(3): rules.adam(w, g, st, lr, {})
    assert (w - w0).abs().max() > 0 and st["t"] == 3
    w = w0.clone(); rules.adam(w, g, {}, lr, dict(wd=.5)); w2 = w0.clone(); rules.adam(w2, g, {}, lr, {}); assert (w - w2).abs().max() > 0, "wd knob does nothing"
    for k in ("stiefel", "dgram"):
        on = rules.on_manifold(torch.randn(16, 40), k); ww = rules.manifold_step(on, torch.randn(16, 40), .02, k); tt = ww @ ww.T; off = tt - torch.diag(tt.diag()); assert off.abs().max() < 1e-2, f"{k}: step left the manifold, {float(off.abs().max())}"

with stage("wrong shapes are refused"):
    for rule in ("muon", "hybrid"):
        try: rules.ModuleOpt(dict(norm=dict(rule=rule, lr=1, params=[torch.nn.Parameter(torch.zeros(4))]))); raise AssertionError("1-D accepted")
        except ValueError: pass

with stage("repeat: same setting, same scores; jitter differs"):
    a = env.train(args(results=out), arms.parse("muon"), W, out); b = env.train(args(results=out), arms.parse("muon"), W, out); assert a["scores"] == b["scores"] and a["curve"] == b["curve"], "not repeatable"
    j = env.train(args(results=out, jitter=1), arms.parse("muon"), W, out); assert j["scores"]["text.tune.loss"] != a["scores"]["text.tune.loss"], "jitter changed nothing"; assert set(a["scores"]) == {"text.tune.loss"}, "a search row holds a report score"
    assert env.key_of(args(), arms.parse("muon")) != env.key_of(args(jitter=1), arms.parse("muon")), "jitter shares a key"

with stage("every arm trains on the dense model"):
    for name in arms.ARMS:
        if name in ("stiefel", "hetero", "hetero-inv", "dgram"): continue
        r = env.train(args(results=out, steps=6), arms.parse(name), W, out); assert r["finite"] or name == "blind", name

with stage("manifold arms train and weights stay on the manifold"):
    for name in ("stiefel", "hetero", "hetero-inv", "dgram"):
        r = env.train(args(results=out, steps=3), arms.parse(name), W, out); assert r["finite"], name

with stage("MoE, tied and no QK-norm train"):
    for kw, name in ((dict(moe=3), "sym"), (dict(tied=True), "muon"), (dict(qknorm=False), "muon"), (dict(moe=3, tied=True), "rownorm-all")):
        r = env.train(args(results=out, steps=6, **kw), arms.parse(name), W, out); assert r["finite"]
        if kw.get("moe"): assert r["weights"].get("router") and r["weights"].get("exp"), "moe modules missing"
        if kw.get("tied"): assert "head" not in r["weights"], "tied model has a head"

with stage("blow-ups are recorded, not crashes"):
    r = env.train(args(results=out, steps=12), arms.parse("emb=adam:50;head=adam:50;qk=sgdm:1e6;vo=sgdm:1e6;gu=sgdm:1e6;down=sgdm:1e6;norm=adam:50"), W, out)
    assert r["opt"]["unstable"], "a huge lr should be flagged"

with stage("full evals: every score present, hidden rows apart"):
    r = env.train(args(results=out, evals="full"), arms.parse("muon"), W, out); assert len(r["scores"]) > 50, len(r["scores"])
    for n in ("text.web.loss", "text.tune.loss", "rarity.token-rare.loss", "text.code.loss", "bench.blimp.acc", "memorize.train.loss", "facts.stored.copies32.acc"): assert n in r["scores"], n

with stage("cheats show where they should"):
    a = env.train(args(results=out, steps=20, evals="tune", leak=True), arms.parse("muon"), W, out); assert a["leak"]
    d = env.train(args(results=out, steps=10, double=True), arms.parse("muon"), env.World(C, 20), out); assert d["tokens"] == 2 * 10 * C["batch"] * C["L"] and d["grads_per_step"] == 2, "double run hides its tokens"

with stage("the command line runs once and skips a finished setting"):
    f = os.path.relpath(out + ".cli", HERE); cmd = [sys.executable, "env.py", "--tier", "tiny", "--steps", "4", "--results", f]
    p = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True); assert p.returncode == 0, p.stderr[-500:]
    p = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True); assert "already done" in p.stdout, p.stdout; os.remove(os.path.join(HERE, f))

os.remove(out)
print("all checks pass")
