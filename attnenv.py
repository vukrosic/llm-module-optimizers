"""Attention environment: train one small LLM per task, change only the update rule of the query and key matrices, and score each task on items no training batch holds.

Tokens in an item are random, so the answers can't be memorized: the model has to read its context.
- copy: a random block of tokens repeats, and the model predicts the next token. Scores split by how far back the answer is (near, far).
- select: 4 cue tokens sit among random tokens, the last token repeats one cue, and the answer is the token right after that cue. This is paying attention to one place among distractors.
- latest: one cue appears twice and the answer is the token after its later place, so the earlier one must be ignored.
- majority: predict the most frequent token. A frozen query and key also solve this, so it is the task where no rule should win.
- retain: select with cues A, then with cues B that want the token before the cue. retain.drop is how much loss on cues A grows during B, which measures forgetting during training.
- switch: select, then the same cues want the token before them. switch.new shows how fast attention unlearns a habit.

python3 attnenv.py --spec "qk=muon:.02" --tier a --world 1 [--evals report] [--task copy]
"""
import argparse, fcntl, hashlib, json, os, time
import numpy as np, torch, torch.nn.functional as F
import arms, env, rules

HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(HERE, "runs_attn"); RESULTS = os.environ.get("ATTN_RESULTS", "attn_results.jsonl")
TIERS = {"tiny": dict(width=16, layers=1, heads=2, steps=20, n=32), "a": dict(width=48, layers=2, heads=4, steps=600, n=512), "ax4": dict(width=48, layers=2, heads=4, steps=2400, n=512), "b": dict(width=96, layers=2, heads=4, steps=600, n=512)}
PER_HEAD = ("muonsplit", "muontangent", "muonsphere", "muonhistory", "muonfixed", "muonbrake", "muonfade", "muongate", "adamtangent"); SHARE = dict(copy=1, select=1, latest=1, majority=.25, retain=1, switch=1); V, SEP = 64, 0; BATCH = 16; TASKS = ("copy", "select", "latest", "majority", "retain", "switch"); VERSION = 3
REST = "emb=adam:.01;head=adam:.01;norm=adam:.01;vo=muon:.005;gu=muon:.005;down=muon:.005"
CUES_A, CUES_B = (1, 5), (5, 9)


def picks(r, lo, hi, n, k): return r.permuted(np.tile(np.arange(lo, hi), (n, 1)), axis=1)[:, :k]


def copy(r, n, L=24):
    """Each item is a random block of p tokens repeated. Every position after the first block has a known next token."""
    p = r.integers(3, 13, n); b = picks(r, 1, V, n, 12); o = np.arange(L + 1)[None] % p[:, None]; t = np.take_along_axis(b, o, 1); y = t[:, 1:].copy(); y[np.arange(L)[None] < p[:, None] - 1] = -100
    return t[:, :-1], y, dict(dist=p)


def select(r, n, cues=CUES_A, off=1):
    """4 slots of 4 random tokens each hold one cue. Then the 4 cues repeat in random order, and each asks for the token next to it (off 1 after, off -1 before). The last one is scored. All tokens are random, so counting them can't answer."""
    x = picks(r, CUES_B[1], V, n, 20); o = np.arange(n)[:, None]; at = np.arange(4)[None] * 4 + r.integers(1, 3, (n, 4)); c = r.permuted(np.tile(np.arange(*cues), (n, 1)), axis=1); x[o, at] = c; a = x[o, at + off].copy(); p = r.permuted(np.tile(np.arange(4), (n, 1)), axis=1); x[:, 16:] = c[o, p]; y = np.full(x.shape, -100); y[:, 16:] = a[o, p]; return x, y, {}


def latest(r, n):
    """Like select with a fifth slot that holds one cue a second time. That cue is asked last, and its answer is the token after its later place, so the earlier one must be ignored."""
    x = picks(r, CUES_B[1], V, n, 24); o = np.arange(n)[:, None]; at = np.arange(5)[None] * 4 + r.integers(1, 3, (n, 5)); c = r.permuted(np.tile(np.arange(*CUES_A), (n, 1)), axis=1); x[o, at[:, :4]] = c; x[o[:, 0], at[:, 4]] = c[:, 0]; a = x[o, at[:, :4] + 1].copy(); a[:, 0] = x[o[:, 0], at[:, 4] + 1]; p = r.permuted(np.tile(np.arange(1, 4), (n, 1)), axis=1); x[:, 20:23] = c[o, p]; x[:, 23] = c[:, 0]; y = np.full(x.shape, -100); y[:, 20:23] = a[o, p]; y[:, 23] = a[:, 0]; return x, y, {}


def majority(r, n):
    """One token appears 8 times, two appear 4 times and seven once, in random order."""
    t = picks(r, 1, V, n, 10); x = r.permuted(t[:, [0] * 8 + [1] * 4 + [2] * 4 + list(range(3, 10))], axis=1); return np.concatenate([x, np.full((n, 1), SEP)], 1), t[:, 0], {}


def row_nll(m, x, y):
    with torch.no_grad(): lg = m(torch.as_tensor(x))[:, -1]; t = torch.as_tensor(y[:, -1] if y.ndim == 2 else y); return F.cross_entropy(lg, t, reduction="none").numpy(), (lg.argmax(-1) == t).float().numpy()


def fit(m, opt, gen, steps, start, total, watch=None):
    """Trains on fresh items. The lr shape runs over the whole run, so two-phase tasks share one schedule. watch(s) runs after s steps of this phase."""
    for s in range(start, start + steps):
        x, y = gen(BATCH); lg = m(torch.as_tensor(x)); y = torch.as_tensor(y); loss = F.cross_entropy(lg.flatten(0, 1), y.flatten(), ignore_index=-100) if y.ndim == 2 else F.cross_entropy(lg[:, -1], y); loss.backward()
        if not torch.isfinite(loss): return False
        opt.step(min((s + 1) / 20, 1.0, (total - s) / (.3 * total))); m.zero_grad(set_to_none=True)
        if watch: watch(s - start + 1)
    return True


def curve(m, x, y, steps, n=5):
    """Scores (x, y) at n evenly spaced points of a phase. Returns the watch function and the list that fills up. The mean is the area under the loss curve, which still differs after all rules reach zero loss."""
    at = {max(1, round(steps * (i + 1) / n)) for i in range(n)}; seen = []
    def watch(s):
        if s in at: seen.append(row_nll(m, x, y)[0])
    return watch, seen


def run_task(task, spec, tier, world, ev):
    c = TIERS[tier]; tid = TASKS.index(task); torch.set_num_threads(1); torch.manual_seed(world * 100 + tid); m = env.LM(dict(width=c["width"], layers=c["layers"], heads=c["heads"]), V); named = list(m.named_parameters()); steps = max(20, int(c["steps"] * SHARE[task])); n = c["n"]
    groups = arms.groups({k: dict(g, h=c["heads"]) if g["rule"] in PER_HEAD else dict(g) for k, g in spec.items()}, named)
    size = lambda: round(float(sum(w.detach().norm() ** 2 for w in groups["qk"]["params"]) ** .5), 3); q0 = size(); opt = rules.ModuleOpt(groups); tr = np.random.default_rng([world, tid, 0]); te = np.random.default_rng([world, tid, 1 if ev == "report" else 2]); out = {}; t0 = time.time(); ok = True
    if task in ("copy", "select", "latest", "majority"):
        gen = dict(copy=copy, select=select, latest=latest, majority=majority)[task]; x, y, info = gen(te, n); w, seen = curve(m, x, y, steps); ok = fit(m, opt, lambda b: gen(tr, b)[:2], steps, 0, steps, w)
        if ok:
            out[f"{task}.nll"], out[f"{task}.acc"] = row_nll(m, x, y); out[f"{task}.area"] = np.mean(seen, 0)
            if "dist" in info: d = info["dist"]; out[f"{task}.near.area"] = out[f"{task}.area"][d <= 5]; out[f"{task}.far.area"] = out[f"{task}.area"][d >= 10]
    elif task == "retain":
        first = int(steps * .75); x, y, _ = select(te, n); w, seen = curve(m, x, y, first); ok = fit(m, opt, lambda b: select(tr, b)[:2], first, 0, steps, w)
        if ok:
            mid, midacc = row_nll(m, x, y); out["retain.mid.acc"], out["retain.mid.nll"], out["retain.learn.area"] = midacc, mid, np.mean(seen, 0); w, seen = curve(m, x, y, steps - first); ok = fit(m, opt, lambda b: select(tr, b, CUES_B, -1)[:2], steps - first, first, steps, w)
        if ok:
            old, acc = row_nll(m, x, y); out["retain.old.nll"], out["retain.old.acc"], out["retain.drop"], out["retain.old.area"] = old, acc, old - mid, np.mean(seen, 0); x, y, _ = select(te, n, CUES_B, -1); out["retain.new.nll"], _ = row_nll(m, x, y)
    else:
        first = steps // 2; ok = fit(m, opt, lambda b: select(tr, b)[:2], first, 0, steps); x, y, _ = select(te, n, CUES_A, -1); w, seen = curve(m, x, y, steps - first)
        if ok: ok = fit(m, opt, lambda b: select(tr, b, CUES_A, -1)[:2], steps - first, first, steps, w)
        if ok: out["switch.new.nll"], out["switch.new.acc"] = row_nll(m, x, y); out["switch.new.area"] = np.mean(seen, 0)
    return out, ok, dict(s_per_step=round((time.time() - t0) / steps, 5), qk_norm=[q0, size()])


def key_of(spec, task, tier, world, ev): return hashlib.sha1(json.dumps([VERSION, arms.show(spec), task, tier, world, ev]).encode()).hexdigest()[:16]


def parse(text):
    s = arms.parse(REST); s.update(arms.parse(text)); return s


def job(j):
    spec, task, tier, world, ev, path = j; key = key_of(spec, task, tier, world, ev); s, ok, info = run_task(task, spec, tier, world, ev); os.makedirs(RUNS, exist_ok=True); r = dict(key=key, spec=arms.show(spec), task=task, tier=tier, world=world, evals=ev, finite=bool(ok), scores={k: round(float(v.mean()), 4) for k, v in s.items()}, **info)
    if ok: np.savez_compressed(os.path.join(RUNS, key + ".npz"), **{k: v.astype(np.float32) for k, v in s.items()})
    with open(path, "a") as f: fcntl.flock(f, fcntl.LOCK_EX); f.write(json.dumps(r) + "\n")
    return key


def done(path=RESULTS):
    p = os.path.join(HERE, path); return {r["key"]: r for r in map(json.loads, open(p))} if os.path.exists(p) else {}


def run_all(specs, tier, worlds, ev, tasks=TASKS, path=RESULTS, par=None):
    """Runs what is not finished yet, with one worker per setting, and returns every row by key."""
    import multiprocessing as mp
    have = done(path); js = [(s, t, tier, w, ev, os.path.join(HERE, path)) for s in specs for w in worlds for t in tasks if key_of(s, t, tier, w, ev) not in have]
    if js:
        with mp.get_context("spawn").Pool(par or int(os.environ.get("PAR", 7))) as p: list(p.imap_unordered(job, js))
    return done(path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--spec", default="qk=muon:.02"); p.add_argument("--tier", default="a"); p.add_argument("--world", type=int, default=1); p.add_argument("--evals", default="report"); p.add_argument("--task", default=""); a = p.parse_args(); s = parse(a.spec)
    for t in ([a.task] if a.task else TASKS):
        t0 = time.time(); o, ok, info = run_task(t, s, a.tier, a.world, a.evals); print(f"{t:9} {time.time() - t0:5.1f}s ok={ok} {info['qk_norm']}", {k: round(float(v.mean()), 3) for k, v in o.items()}, flush=True)
