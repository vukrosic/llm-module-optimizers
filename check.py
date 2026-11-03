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
    s0 = torch.linalg.svdvals(step("softmuon", p=0.0)); assert (s0 - s0[0]).abs().max() < 1e-3 and abs(float(s0.norm()) - 32 ** .5) < 1e-3, f"softmuon p 0 should have equal singular values of Frobenius size sqrt(m), got {s0}"
    assert (step("softmuon", p=1.0) - step("normsgd")).abs().max() < 1e-3, "softmuon p 1 should equal normsgd"
    hw = torch.randn(8, 8); hg = torch.randn(8, 8); per = lambda x: [x[i:i + 4] for i in (0, 4)]
    def head_step(rule):
        w = hw.clone(); rules.RULES[rule](w, hg, {}, .01, dict(h=2)); return w
    for b, e in zip(per((hw - head_step("muonsplit")) / .01), per(hg * 1.95)): assert (b - rules.newton_schulz(e)).abs().max() < 1e-3, "muonsplit should run Muon on each head alone"
    for rule in ("muontangent", "adamtangent"):
        for b, d in zip(per(hw), per((hw - head_step(rule)) / .01)): assert abs(float((b * d).sum())) < 1e-3, f"{rule} should have no part along a head's rows"
    w = hw.clone(); st = {}; first = [float(b.norm()) for b in per(hw)]
    for _ in range(50): rules.RULES["muonsphere"](w, torch.randn(8, 8), st, .05, dict(h=2))
    for b, r in zip(per(w), first): assert abs(float(b.norm()) - r) < 1e-4, "muonsphere should hold each head's size"
    assert (w - hw).abs().max() > .01, "muonsphere should move the rows"
    w = hw.clone(); st = {}; ga = torch.zeros(8, 8); ga[:, 0] = torch.randn(8); gb = torch.zeros(8, 8); gb[:, 1] = torch.randn(8)
    for _ in range(30): rules.muonhistory(w, ga, st, 1e-5, dict(h=2))
    st["b"].zero_(); before = w.clone(); rules.muonhistory(w, ga + gb, st, .01, dict(h=2, beta=0.)); mv = w - before; assert mv[:, 1].norm() > 3 * mv[:, 0].norm(), "muonhistory should favour the input direction with little history"
    for b, r in zip(per(w), first): assert abs(float(b.norm()) - r) < 1e-4, "muonhistory should hold each head's size"
    for rule in ("muonfixed", "muonbrake"):
        w, st = hw.clone(), {}; first = [float(b.norm()) for b in per(w)]
        for _ in range(50): rules.RULES[rule](w, torch.randn(8, 8), st, .05, dict(h=2))
        for b, r in zip(per(w), first): assert abs(float(b.norm()) - r) < 1e-4, f"{rule} should hold each head's size"
        assert (w - hw).abs().max() > .01, f"{rule} should move the rows"
    w, st = hw.clone(), {}; first = [float(b.norm()) for b in per(w)]
    for _ in range(50): rules.muonfade(w, torch.randn(8, 8), st, .05, dict(h=2))
    for b, r in zip(per(w), first): assert abs(float(b.norm()) - r) < 1e-4, "muonfade should hold each head's size"
    fg = torch.randn(1, 8).repeat(8, 1); w, st = hw.clone(), {}
    for _ in range(20): rules.muongate(w, fg + 1e-3 * torch.randn(8, 8), st, .001, dict(h=2))
    assert min(st["u"]) > .2, "muongate should read a steady gradient as still learning"
    before = w.clone(); rules.muongate(w, fg, st, .05, dict(h=2)); assert abs(((w - before)[:4] * before[:4]).sum().item()) < 1e-3, "muongate should allow no growth while learning"
    ga_ = torch.randn(8, 8); x, y = hw.clone(), hw.clone(); rules.muonfixed(x, ga_, {}, .05, dict(h=2)); rules.muonbrake(y, ga_, dict(t=630), .05, dict(h=2, tau=630)); assert (y - hw).norm() < .6 * (x - hw).norm(), "muonbrake should turn less later in training"
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
