"""Optimizer test environment: train one LLM with one update rule per module, at any budget, on fresh text, and print the same scores.

python3 env.py --spec muon --tier 5s [--set emb=rownorm:.5] [--world 1] [--jitter 2] [--evals full] [--final] [--no-qknorm] [--tied] [--moe 4]

One knob, --tier, sets the model size and steps, so every arm reads the same tokens. The model comes from llm-memory-env without the
memory, with the head untied by default because the module-optimizer papers treat embedding and head separately. A spec gives every
module a rule and an lr (arms.py). Text is fresh web text read once plus FACTSHARE of ParaRel sentences. --world N changes the text, facts,
held-out rows and starting weights, and search runs never read a world above 0. Rows go to results.jsonl with a key per setting,
so a finished setting is never rerun; per-item scores go to runs/. A search row stores text.tune.loss and no report score, so nothing picking a setting can read one.
"""
import argparse, collections, fcntl, glob, hashlib, json, os, signal, subprocess, time
import numpy as np, tiktoken, torch, torch.nn as nn, torch.nn.functional as F
import arms, evals, rules

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data"); RUNS = os.path.join(HERE, "runs")
FW = os.environ.get("FWDIR", os.path.join(DATA, "web")); PR = os.path.join(DATA, "pararel")
EOT, FACTSHARE, COPIES, STRIDE, TEST_VERSION = 50256, 0.1, (2, 8, 32), 4_000_000, 3  # worlds start STRIDE tokens apart, so their training text never overlaps; TEST_VERSION is in each row's key, so rows from an older test are never reused
TIERS = {
    "tiny": dict(width=16, layers=1, heads=2, V=2000, batch=4, L=24, steps=20, val_rows=12, marks=2, bench=4, min_facts=4),  # checks code only, its scores mean nothing
    "5s": dict(width=48, layers=2, heads=4, V=1000, batch=16, L=48, steps=250, val_rows=3200, marks=5, bench=60),
    "5sx4": dict(width=48, layers=2, heads=4, V=1000, batch=16, L=48, steps=1000, val_rows=3200, marks=8, bench=60),
    "5sx16": dict(width=48, layers=2, heads=4, V=1000, batch=16, L=48, steps=4000, val_rows=3200, marks=10, bench=60),
    "1m": dict(width=64, layers=4, heads=4, V=1500, batch=16, L=48, steps=1000, val_rows=3200, marks=6, bench=150),
    "15m": dict(width=128, layers=6, heads=8, V=4000, batch=16, L=64, steps=2000, val_rows=3200, marks=8, bench=250),
    "1h": dict(width=256, layers=6, heads=8, V=8000, batch=32, L=128, steps=10000, val_rows=3200, marks=10, bench=500),
    "5h": dict(width=512, layers=8, heads=8, V=20000, batch=64, L=256, steps=60000, val_rows=4000, marks=12, bench=600),
}
STOP = set("""a an the of in on at to for from by with as is was are were be been being has had have do does did
and or but not no it its this that these those which who whom whose what where when there their his her he she they
him them one also into onto about over under than then so such can could may might will would shall should""".split())


def test_pattern(p):
    x, y = p.find("[X]"), p.find("[Y]")
    if x < 0 or y < 0 or x > y: return False
    return any(w not in STOP for w in p[:y].replace("[X]", " ").lower().replace(".", " ").replace(",", " ").split())


def fact_table():
    """ParaRel relations, cleaned by the ParaRel* rule: per relation one wording group is held out for testing. Cached."""
    path = os.path.join(DATA, "facts.json")
    if os.path.exists(path): return json.load(open(path))
    rng = np.random.default_rng(0); out = []
    for f in sorted(glob.glob(os.path.join(PR, "pattern_data", "graphs_json", "P*.jsonl"))):
        r = os.path.basename(f)[:-6]; ff = os.path.join(PR, "trex_lms_vocab", r + ".jsonl")
        if not os.path.exists(ff): continue
        xy = [p for p in map(json.loads, filter(str.strip, open(f))) if 0 <= p["pattern"].find("[X]") < p["pattern"].find("[Y]")]
        groups = sorted({p["extended_lemma"] for p in xy if test_pattern(p["pattern"])})
        if len(groups) < 2: continue
        G = groups[rng.integers(len(groups))]
        S = [p["pattern"] for p in xy if p["extended_lemma"] != G]; T = [p["pattern"] for p in xy if p["extended_lemma"] == G and test_pattern(p["pattern"])]
        if not S or not T: continue
        pairs = list(dict.fromkeys((d["sub_label"], d["obj_label"]) for d in map(json.loads, filter(str.strip, open(ff)))))
        n = collections.Counter(s for s, _ in pairs)
        out.append(dict(r=r, S=S, T=T, pairs=[(s, o) for s, o in pairs if n[s] == 1]))
    os.makedirs(DATA, exist_ok=True); json.dump(out, open(path, "w")); return out


class World:
    """Everything that depends only on the tier and world number: vocabulary, training rows, held-out rows, fact items."""
    def __init__(s, c, steps, world=0):
        s.c = c; s.world = world; s.enc = tiktoken.get_encoding("gpt2"); rng = np.random.default_rng(world)
        shard = np.memmap(os.path.join(FW, "fineweb_train_000001.bin"), dtype=np.uint16, mode="r", offset=1024)
        cp = os.path.join(DATA, "counts.npy")
        if not os.path.exists(cp): os.makedirs(DATA, exist_ok=True); np.save(cp, np.bincount(np.asarray(shard[:20_000_000]), minlength=65536))
        counts = np.load(cp); s.counts = counts
        s.keep = np.unique(np.concatenate([np.argsort(counts)[-(c["V"] - 2):], [EOT]])); s.V = len(s.keep) + 1; s.unk = s.V - 1
        s.look = np.full(65536, s.unk, np.int64); s.look[s.keep] = np.arange(len(s.keep)); kept = set(s.keep.tolist())
        s.E = lambda t: s.enc.encode(" " + t.strip())
        tokens = steps * c["batch"] * c["L"]
        s.facts = s.pick_facts(rng, kept, int(tokens * FACTSHARE))
        doc = lambda f: np.array([EOT] + s.E(f["S"][rng.integers(len(f["S"]))].replace("[X]", f["s"]).replace("[Y]", f["o"])), np.int64)
        early = [doc(f) for f in s.facts if f["early"] for _ in range(f["copies"])]; docs = [doc(f) for f in s.facts if not f["early"] for _ in range(f["copies"])]
        nfact = sum(map(len, docs + early)); nweb = tokens + 2 * c["L"] - nfact; start = world * max(STRIDE, nweb)
        if start + nweb > len(shard): raise SystemExit(f"world {world} needs {start + nweb} web tokens, the shard has {len(shard)}")
        web = np.asarray(shard[start:start + nweb]).astype(np.int64); cut = np.flatnonzero(web == EOT)
        wdocs = np.split(web, cut[1:]) if len(cut) > 1 else [web]
        alld = docs + wdocs; go = rng.random(len(alld)) < (tokens / 3 - sum(map(len, early))) / max(1, tokens - sum(map(len, early)))

        def rows_of(part):
            st = np.concatenate([part[i] for i in rng.permutation(len(part))]); r = st[:len(st) // c["L"] * c["L"]].reshape(-1, c["L"]); return r[rng.permutation(len(r))]
        part1 = rows_of(early + [d for d, g in zip(alld, go) if g]); s.rows = np.concatenate([part1, rows_of([d for d, g in zip(alld, go) if not g])]); s.split = len(part1)
        val = np.memmap(os.path.join(FW, "fineweb_val_000000.bin"), dtype=np.uint16, mode="r", offset=1024)  # reported, tuning and hidden rows alternate
        slots = np.random.default_rng(777).permutation(len(val) // c["L"])[world * 3 * c["val_rows"]:(world + 1) * 3 * c["val_rows"]] * c["L"]  # same shuffle in every world, so worlds and splits share no token
        if len(slots) < 3 * c["val_rows"]: raise SystemExit(f"world {world} needs more held-out text")
        held = np.stack([np.asarray(val[a:a + c["L"]]) for a in slots]).astype(np.int64)
        s.val, s.tune, s.hid = held[::3], held[1::3], held[2::3]
        s.nfact, s.nweb = nfact, nweb

    def pick_facts(s, rng, kept, budget):
        """As many facts as FACTSHARE of the tokens can hold; one third each written 2, 8 and 32 times."""
        pool = []
        for g in fact_table():
            ok = [(a, b) for a, b in g["pairs"] if all(t in kept for t in s.E(a) + s.E(b))]
            if not ok: continue
            maj = collections.Counter(b for _, b in ok).most_common(1)[0][0]; objs = sorted({b for _, b in ok})
            pool += [dict(r=g["r"], s=a, o=b, S=g["S"], T=g["T"], maj=maj, objs=objs) for a, b in ok]
        pool = [pool[i] for i in rng.permutation(len(pool))]; avg = sum(COPIES) / 3 * 13
        n = min(max(s.c.get("min_facts", 40), int(budget / avg)), max(len(pool) - 400, len(pool) * 3 // 4)); s.natural = [f for f in pool[n:n + 400] if f["o"] != f["maj"]]; pool = pool[:n]
        for i, f in enumerate(pool):
            f["copies"] = 8 if i % 4 == 3 else COPIES[i % 3]; f["early"] = i % 4 == 3
        return pool


def rope(x, base=10000.0):
    t, d = x.shape[-2], x.shape[-1]
    f = 1.0 / base ** (torch.arange(0, d, 2, device=x.device).float() / d); ang = torch.outer(torch.arange(t, device=x.device).float(), f); c, s = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x2 * c + x1 * s], -1).flatten(-2)


class Block(nn.Module):
    def __init__(s, d, h, moe=0, topk=2, qknorm=True):
        super().__init__()
        s.h, s.dk, s.moe, s.topk = h, d // h, moe, topk
        s.q, s.k, s.v, s.o = (nn.Linear(d, d, bias=False) for _ in range(4)); hid = int(8 * d / 3)
        if moe:  # 1 shared expert of half the dense width + top-k of E routed experts, active width = dense width
            hs, he = hid // 2, hid // (2 * topk)
            s.gate, s.up, s.down = nn.Linear(d, hs, bias=False), nn.Linear(d, hs, bias=False), nn.Linear(hs, d, bias=False)
            s.eg = nn.ModuleList(nn.Linear(d, he, bias=False) for _ in range(moe)); s.eu = nn.ModuleList(nn.Linear(d, he, bias=False) for _ in range(moe))
            s.ed = nn.ModuleList(nn.Linear(he, d, bias=False) for _ in range(moe)); s.router = nn.Linear(d, moe, bias=False)
            s.register_buffer("rbias", torch.zeros(moe)); s.load = None
        else:
            s.gate, s.up, s.down = nn.Linear(d, hid, bias=False), nn.Linear(d, hid, bias=False), nn.Linear(hid, d, bias=False)
        s.n1, s.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        s.qn, s.kn = (nn.RMSNorm(s.dk), nn.RMSNorm(s.dk)) if qknorm else (nn.Identity(), nn.Identity())
        s.peak = None

    def forward(s, x):
        b, t, d = x.shape
        h = s.n1(x); q, k, v = (m(h).view(b, t, s.h, s.dk).transpose(1, 2) for m in (s.q, s.k, s.v))
        rq, rk = rope(s.qn(q)), rope(s.kn(k))
        if s.peak is not None: s.peak = max(s.peak, float((rq @ rk.transpose(-1, -2)).abs().max()) / s.dk ** .5)
        y = F.scaled_dot_product_attention(rq, rk, v, is_causal=True)
        x = x + s.o(y.transpose(1, 2).reshape(b, t, d)); h = s.n2(x)
        out = s.down(F.silu(s.gate(h)) * s.up(h))
        if s.moe:
            hf = h.reshape(-1, d); sc = torch.sigmoid(s.router(hf)); idx = (sc + s.rbias).topk(s.topk, -1).indices
            wt = sc.gather(1, idx); wt = wt / wt.sum(-1, keepdim=True); r = torch.zeros_like(hf)
            for e in range(s.moe):
                m = idx == e; rows = m.any(-1).nonzero().squeeze(1)
                if len(rows):
                    xe = hf[rows]; r.index_add_(0, rows, s.ed[e](F.silu(s.eg[e](xe)) * s.eu[e](xe)) * (wt * m)[rows].sum(-1, keepdim=True))
            s.load = torch.bincount(idx.flatten(), minlength=s.moe).float(); out = out + r.view(b, t, d)
        return x + out


class LM(nn.Module):
    def __init__(s, c, V, qknorm=True, tied=False, moe=0):
        super().__init__()
        d = c["width"]; s.emb = nn.Embedding(V, d); nn.init.normal_(s.emb.weight, std=d ** -0.5)
        s.blocks = nn.ModuleList(Block(d, c["heads"], moe, qknorm=qknorm) for _ in range(c["layers"])); s.norm = nn.RMSNorm(d); s.tied = tied
        if not tied:
            s.head = nn.Linear(d, V, bias=False); nn.init.normal_(s.head.weight, std=d ** -0.5)

    def forward(s, x):
        e = s.emb(x)
        for b in s.blocks: e = b(e)
        e = s.norm(e)
        return e @ s.emb.weight.T if s.tied else s.head(e)


def key_of(a, spec):
    """One setting, one row: what changes the result, and nothing else."""
    d = dict(version=TEST_VERSION, config=TIERS[a.tier], spec=arms.show(spec), tier=a.tier, steps=a.steps, world=a.world, jitter=a.jitter, lr_mult=a.lr_mult, qknorm=a.qknorm, tied=a.tied, moe=a.moe, leak=a.leak, double=a.double, evals=a.evals)
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def commit():
    try: return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    except OSError: return ""


def unstable_flags(curve):
    """2606.13276's criterion, with 'after step 1000' read as 'after 40% of steps' for short tiers: blow-up, a bad end, or never coming back to the best."""
    v = [x for _, x in curve]
    if not v or not all(np.isfinite(v)): return dict(unstable=True, why="not finite")
    best = min(v); i = v.index(best); late = v[len(v) * 2 // 5:]; last = v[-max(1, len(v) // 5):]
    if any(x > max(2 * best, 5.0) for x in late): return dict(unstable=True, why="blow-up")
    if np.mean(last) > 1.2 * best: return dict(unstable=True, why="bad end")
    if i < len(v) - 1 and min(v[i + 1:]) > 1.05 * best: return dict(unstable=True, why="stays above best")
    return dict(unstable=False, why="")


def done_keys(path):
    if not os.path.exists(path): return set()
    return {json.loads(l)["key"] for l in open(path) if l.strip()}


def train(a, spec, W=None, out=None):
    out = out or os.path.join(HERE, a.results); key = key_of(a, spec)
    torch.set_num_threads(a.threads); torch.manual_seed(a.world); t0 = time.time(); c = dict(TIERS[a.tier])
    steps = a.steps or c["steps"]; budget = steps * (2 if a.double else 1)
    W = W or World(c, budget, a.world); model = LM(c, W.V, a.qknorm, a.tied, a.moe)
    weights = collections.Counter(); n_named = list(model.named_parameters())
    for n, w in n_named: weights[arms.module_of(n, w)] += w.numel()
    jit = {m: dict(g, lr=g["lr"] * a.lr_mult * (1 + 1e-3 * a.jitter)) for m, g in spec.items()}
    opt = rules.ModuleOpt(arms.groups(jit, n_named)); X = torch.as_tensor(W.look[W.rows]); LK = torch.as_tensor(W.look[W.val]); ctx = evals.Ctx(model, W, c, "cpu")
    marks = sorted({int(steps * (i + 1) / c["marks"]) for i in range(c["marks"])}); curve = []; train_s = 0.0; losses = []; ok = True; tune = W.tune[:64]
    t1 = time.time()
    for step in range(steps + 1):
        if step in marks:
            model.eval(); t = time.time(); v = float(np.mean(ctx.row_loss(tune))); curve.append((step, round(v, 4))); model.train(); t1 += time.time() - t
        if step == steps: break
        sc = min((step + 1) / 20, 1.0, (steps - step) / (0.3 * steps)); batches = 2 if a.double else 1
        for j in range(batches):
            idx = ((step * batches + j) * c["batch"] + np.arange(c["batch"])) % len(X); x = X[idx]
            if a.leak: x = x.clone(); x[:c["batch"] // 2] = LK[(step * c["batch"] + np.arange(c["batch"] // 2)) % len(LK)]
            loss = F.cross_entropy(model(x)[:, :-1].flatten(0, 1), x[:, 1:].flatten(), ignore_index=W.unk); (loss / batches).backward()
        if not torch.isfinite(loss): ok = False; break
        losses.append(float(loss.detach())); opt.step(sc, measure=step % 10 == 0); model.zero_grad(set_to_none=True)
        for b in model.blocks:  # aux-loss-free balancing: raise the bias of under-used experts
            if b.moe: b.rbias += 1e-3 * torch.sign(b.load.mean() - b.load)
        if a.max_seconds and time.time() - t0 > a.max_seconds: ok = False; print(f"stopped at step {step} by --max-seconds", flush=True); break
    train_s = time.time() - t1; finite = ok and all(torch.isfinite(w).all() for w in model.parameters())
    res = dict(key=key, spec=arms.show(spec), tier=a.tier, world=a.world, jitter=a.jitter, lr_mult=a.lr_mult, qknorm=a.qknorm, tied=a.tied, moe=a.moe, leak=a.leak, double=a.double, evals=a.evals, tag=a.tag, commit=commit(),
               steps=steps, tokens=step * c["batch"] * c["L"] * (2 if a.double else 1), grads_per_step=2 if a.double else 1, train_s=round(train_s, 2), s_per_step=round(train_s / max(step, 1), 5), threads=a.threads, state_bytes=opt.state_bytes(),
               weights=dict(weights), finite=bool(finite), curve=curve)
    if finite:
        model.eval(); te = time.time(); s = evals.run_all(ctx, a.evals); res["eval_s"] = round(time.time() - te, 2)
        if os.environ.get("POISON"): s = {k: (v if k == "text.tune.loss" else np.random.default_rng(int.from_bytes(key.encode()[:4], "big")).random(len(v))) for k, v in s.items()}  # debug/poison.py: a search must not change when report scores are noise
        res["scores"] = {k: round(float(np.mean(v)), 4) if len(v) else None for k, v in s.items()}; res["n"] = {k: len(v) for k, v in s.items()}
        half = len(losses) // 5; res["opt"] = dict(spike=round(max(losses[half:]) - min(losses[:half + 1]), 4) if losses[half:] else None, auc=round(float(np.mean([v for _, v in curve])), 4), rel=opt.rel_sizes(), **unstable_flags(curve))
        res["opt"]["sigma_max"] = {}
        for m, g in arms.groups(jit, list(model.named_parameters())).items():
            mats = [w for w in g["params"] if w.ndim == 2]
            if mats: res["opt"]["sigma_max"][m] = round(max(float(torch.linalg.matrix_norm(w.detach(), 2)) for w in mats), 3)
        for b in model.blocks: b.peak = 0.0
        with torch.no_grad(): model(torch.as_tensor(W.look[W.val[:8]]))
        res["opt"]["peak_attention_logit"] = round(max(b.peak for b in model.blocks), 3)
        for b in model.blocks: b.peak = None
        os.makedirs(RUNS, exist_ok=True); np.savez_compressed(os.path.join(RUNS, key + ".npz"), **{k: np.asarray(v, np.float32) for k, v in s.items()})
        if a.final:
            hid = evals.run_all(evals.Ctx(model, W, c, "cpu", final=True), "full"); np.savez_compressed(os.path.join(RUNS, "final_" + key + ".npz"), **{k: np.asarray(v, np.float32) for k, v in hid.items()})
            with open(os.path.join(HERE, os.environ.get("FINAL_RESULTS", "final_results.jsonl")), "a") as f: f.write(json.dumps(dict(key=key, spec=res["spec"], tier=a.tier, world=a.world, scores={k: round(float(np.mean(v)), 4) if len(v) else None for k, v in hid.items()})) + "\n")
    else:
        res["scores"] = {}; res["opt"] = dict(unstable=True, why="not finite")
    with open(out, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX); f.write(json.dumps(res) + "\n")
    return res


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", default="muon"); p.add_argument("--set", action="append", default=[]); p.add_argument("--tier", default="5s", choices=list(TIERS)); p.add_argument("--steps", type=int, default=0)
    p.add_argument("--world", type=int, default=0); p.add_argument("--jitter", type=int, default=0); p.add_argument("--lr-mult", type=float, default=1.0); p.add_argument("--evals", default="tune", choices=["tune", "full"])
    p.add_argument("--no-qknorm", dest="qknorm", action="store_false"); p.add_argument("--tied", action="store_true"); p.add_argument("--moe", type=int, default=0)
    p.add_argument("--leak", action="store_true"); p.add_argument("--double", action="store_true"); p.add_argument("--final", action="store_true")
    p.add_argument("--tag", default=""); p.add_argument("--threads", type=int, default=1); p.add_argument("--max-seconds", type=float, default=0); p.add_argument("--kill-after", type=int, default=0)
    p.add_argument("--results", default="results.jsonl"); p.add_argument("--force", action="store_true")
    return p


def full_spec(a):
    spec = arms.parse(a.spec)
    for s in a.set: spec.update(arms.parse(s))
    return spec


if __name__ == "__main__":
    a = parser().parse_args()
    if a.kill_after: signal.alarm(a.kill_after)  # debug runs die here, so a slow step is found, not waited for
    spec = full_spec(a); path = os.path.join(HERE, a.results)
    if not a.force and key_of(a, spec) in done_keys(path): print("already done", flush=True); raise SystemExit
    r = train(a, spec); sc = r["scores"]
    print(f"{a.tier} world {a.world} {r['spec']}\n  web {sc.get('text.web.loss')} tune {sc.get('text.tune.loss')} {r['train_s']}s train, {r.get('eval_s')}s evals, weights {r['weights']}\n  opt {r['opt']}", flush=True)
