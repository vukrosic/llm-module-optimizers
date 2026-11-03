"""Evals. Each group takes a Ctx and returns {score name: per-item array}, so arms can be compared item by item. Copied from llm-memory-env.

Score names end in .loss, .nll or .delta when lower is better. Items depend only on the tier. Groups marked HEAVY run only
at the start and end of training.

Part of the items is hidden: a third of the text rows and facts and half of each benchmark. Normal runs score the public items;
--final scores the hidden ones and writes them elsewhere, so a search never sees them. Options are scored the usual way, by total log-probability of the continuation;
a target outside the vocabulary counts as log(1/V) for every method.
"""
import json, os
import numpy as np, torch, torch.nn.functional as F

EOT = 50256; BENCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bench")


class Ctx:
    def __init__(s, model, W, c, dev, final=False):
        s.model, s.W, s.c, s.dev, s.final = model, W, c, dev, final; s.sets = {}; s.val = W.hid if final else W.val

    def once(s, name, make):
        if name not in s.sets: s.sets[name] = make()
        return s.sets[name]

    @torch.no_grad()
    def row_nll(s, raw, bs=64):
        """Next-token loss and greedy hit per position for raw GPT-2 rows; ok marks targets inside the vocabulary."""
        X = torch.as_tensor(s.W.look[raw]); nll, hit = [], []
        for i in range(0, len(X), bs):
            x = X[i:i + bs].to(s.dev); lo = s.model(x)[:, :-1]
            nll.append(F.cross_entropy(lo.flatten(0, 1), x[:, 1:].flatten(), reduction="none").view(lo.shape[:2]).cpu().numpy())
            hit.append((lo.argmax(-1) == x[:, 1:]).cpu().numpy())
        return np.concatenate(nll), np.concatenate(hit), (X[:, 1:] != s.W.unk).numpy()

    def row_loss(s, raw):
        a = s.row_nll(raw); return (a[0] * a[2]).sum(1) / np.maximum(a[2].sum(1), 1)

    @torch.no_grad()
    def probe(s, items, bs=128):
        """items = (prompt, answer) in raw tokens. For each: is the whole answer predicted greedily, and the log-probs of the next token."""
        W, out = s.W, []
        for i in range(0, len(items), bs):
            ch = items[i:i + bs]; n = max(len(p) + len(a) for p, a in ch); x = np.full((len(ch), n), W.look[EOT], np.int64)
            for j, (p, a) in enumerate(ch): x[j, :len(p) + len(a)] = W.look[np.array(p + a)]
            x = torch.as_tensor(x, device=s.dev); lo = s.model(x); pred = lo.argmax(-1)
            for j, (p, a) in enumerate(ch):
                k = len(p) - 1
                out.append((bool((pred[j, k:k + len(a)] == x[j, k + 1:k + 1 + len(a)]).all()), F.log_softmax(lo[j, k].float(), -1).cpu().numpy()))
        return out

    @torch.no_grad()
    def pair_logp(s, seqs, budget=4000):
        """seqs = (tokens, index where the continuation starts). Total log-prob of each continuation, longest first in token-budget batches."""
        W = s.W; order = np.argsort([-len(t) for t, _ in seqs]); res = np.zeros(len(seqs)); i = 0; miss = -float(np.log(W.V))
        while i < len(seqs):
            n = len(seqs[order[i]][0]); b = max(1, budget // n); ids = order[i:i + b]; x = np.full((len(ids), n), W.look[EOT], np.int64)
            for j, q in enumerate(ids): x[j, :len(seqs[q][0])] = W.look[np.array(seqs[q][0])]
            x = torch.as_tensor(x, device=s.dev); lp = F.log_softmax(s.model(x)[:, :-1].float(), -1).gather(-1, x[:, 1:, None])[..., 0]
            lp = torch.where(x[:, 1:] == W.unk, torch.full_like(lp, miss), lp).cpu().numpy()
            for j, q in enumerate(ids):
                t, st = seqs[q]; res[q] = lp[j, st - 1:len(t) - 1].sum()
            i += b
        return res


def by_row(a, m):
    """Mean per row over the marked positions, so positions in one row don't count as independent items."""
    return np.array([a[i][m[i]].mean() for i in range(len(a)) if m[i].any()])


def text(ctx):
    nll, hit, ok = ctx.row_nll(ctx.val); per = lambda a: (a * ok).sum(1) / np.maximum(ok.sum(1), 1)
    return {"text.web.loss": per(nll), "text.web.acc": per(hit), "text.tune.loss": ctx.row_loss(ctx.W.tune)}


def kinds(ctx):
    """Loss on code, math, fiction and German, the first rows of each file, with no training on them beyond what the web holds."""
    def make():
        out = {}
        for n in ("fiction", "math", "code", "german"):
            t = [x for l in open(os.path.join(BENCH, f"text_{n}.jsonl")) for x in [EOT] + ctx.W.enc.encode(json.loads(l)["text"], disallowed_special=())]
            L = ctx.c["L"]; k = min(ctx.c["val_rows"] // 2, len(t) // L); out[n] = np.array(t[:k * L], np.int64).reshape(-1, L)
        return out
    return {f"text.{n}.loss": ctx.row_loss(r) for n, r in ctx.once("kinds", make).items()}


def rarity(ctx):
    """Loss split by how common the target token is, and by how often the 3-token phrase ending at it was in the training text."""
    W = ctx.W; val = ctx.val

    def make():
        r = W.rows; keys, cnt = np.unique((r[:, :-2] << 32) | (r[:, 1:-1] << 16) | r[:, 2:], return_counts=True)
        q = (val[:, :-2] << 32) | (val[:, 1:-1] << 16) | val[:, 2:]; i = np.clip(np.searchsorted(keys, q), 0, len(keys) - 1)
        rank = np.zeros(65536, np.int64); rank[np.argsort(-W.counts)] = np.arange(65536)
        return np.where(keys[i] == q, cnt[i], 0), rank[val[:, 1:]]
    c, rk = ctx.once("rarity", make); nll, _, ok = ctx.row_nll(val); n3 = nll[:, 1:]; ok3 = ok[:, 1:]; V = W.V; out = {}
    for name, m in (("rarity.token-common", rk < 500), ("rarity.token-mid", (rk >= 500) & (rk < V // 2)), ("rarity.token-rare", rk >= V // 2)):
        out[name + ".loss"] = by_row(nll, m & ok)
    for name, lo, hi in (("rarity.phrase-unseen", 0, 0), ("rarity.phrase-1-9", 1, 9), ("rarity.phrase-10plus", 10, 10 ** 12)):
        out[name + ".loss"] = by_row(n3, (c >= lo) & (c <= hi) & ok3)
    return out


def ambiguous(ctx):
    """The same 3-token phrase with several continuations in the training text, seen 10+ times: loss when the next token is the phrase's usual one, and when it is another one the page has to decide."""
    W = ctx.W; val = ctx.val

    def make():
        r = W.rows; pre = (r[:, :-3] << 32) | (r[:, 1:-2] << 16) | r[:, 2:-1]; keys, cnt = np.unique((pre << 16) | r[:, 3:], return_counts=True)
        p, t = keys >> 16, keys & 65535
        up, inv = np.unique(p, return_inverse=True); tot = np.bincount(inv, cnt); top = np.zeros(len(up), np.int64); best = np.zeros(len(up), np.int64)
        for i in np.argsort(cnt): best[inv[i]] = cnt[i]; top[inv[i]] = t[i]  # ascending count, so the last write is the most common continuation
        q = (val[:, :-3] << 32) | (val[:, 1:-2] << 16) | val[:, 2:-1]; i = np.clip(np.searchsorted(up, q), 0, len(up) - 1); seen = up[i] == q
        amb = seen & (tot[i] >= 10) & (best[i] / tot[i] < 0.7)
        return amb & (val[:, 3:] == top[i]), amb & (val[:, 3:] != top[i])
    usual, other = ctx.once("ambiguous", make); nll, _, ok = ctx.row_nll(val); n = nll[:, 2:]; o = ok[:, 2:]
    return {"ambiguous.usual.loss": by_row(n, usual & o), "ambiguous.other.loss": by_row(n, other & o)}


def fact_sets(ctx):
    """Planted facts asked in a stored and a held-out wording, facts never planted, made-up names, and the page contradicting the memory."""
    def make():
        W = ctx.W; E = W.E; rng = np.random.default_rng(1); pre = lambda p, x: [EOT] + E(p[:p.find("[Y]")].replace("[X]", x)); first = lambda t: int(W.look[E(t)[0]])  # training sentences start with EOT, so questions do too
        stored, new, fake, page, natural = [], [], [], [], []; toks = [t for f in W.facts for t in E(f["s"])]

        def item(f, wording):
            cand = sorted({first(o) for o in f["objs"]} | {first(f["o"])})
            tail = wording[wording.find("[X]") + 3:wording.find("[Y]")]  # near: the name's last token is inside the last 3 tokens of the question
            q = pre(wording, f["s"]); return dict(q=q, a=E(f["o"]), y=first(f["o"]), inq=first(f["o"]) in W.look[np.array(q)].tolist(), cand=cand, copies=f.get("copies", 0), maj=f["o"] == f["maj"], early=f.get("early", False), near=len(W.enc.encode(tail)) <= 2, f=f)
        hid = np.random.default_rng(8).random(len(W.facts)) < 1 / 3; pub = [f for f, h in zip(W.facts, hid) if h == ctx.final]
        for f in [pub[i] for i in sorted(np.random.default_rng(9).permutation(len(pub))[:2000])]:  # at most 2000 facts asked per eval
            p0 = f["S"][rng.integers(len(f["S"]))]; t0 = f["T"][rng.integers(len(f["T"]))]
            stored.append(item(f, p0)); new.append(item(f, t0))
            fake.append(pre(p0, W.enc.decode([int(x) for x in rng.choice(toks, 2)]).strip()))
            other = [o for o in f["objs"] if first(o) != first(f["o"])]
            if other:
                o2 = other[rng.integers(len(other))]; q = pre(p0, f["s"]); page.append(dict(q=q, paged=q + E(o2) + E(".") + q[1:], y2=first(o2), y=first(f["o"])))
        natural = [item(f, f["T"][rng.integers(len(f["T"]))]) for i, f in enumerate(W.natural) if (i % 2 == 1) == ctx.final]; natural = [i for i in natural if not i["inq"]]
        return stored, new, fake, page, natural
    return ctx.once("facts", make)


def fact_scores(ctx, items):
    res = ctx.probe([(i["q"], i["a"]) for i in items])
    acc = np.array([r[0] for r in res], float); nll = np.array([-r[1][i["y"]] for r, i in zip(res, items)])
    pick = np.array([r[1][i["cand"]].argmax() == i["cand"].index(i["y"]) for r, i in zip(res, items)], float)  # the right answer ranks first among the relation's answers
    return acc, nll, pick


def facts(ctx):
    stored, new, _, _, natural = fact_sets(ctx); out = {}
    for kind, items in (("stored", stored), ("new-wording", new)):
        acc, nll, pick = fact_scores(ctx, items)
        for c in (2, 8, 32):
            ix = [i for i, t in enumerate(items) if t["copies"] == c and not t["maj"] and not t["early"] and not t["inq"]]
            out[f"facts.{kind}.copies{c}.acc"] = acc[ix]; out[f"facts.{kind}.copies{c}.nll"] = nll[ix]; out[f"facts.{kind}.copies{c}.pick"] = pick[ix]
        for side, near in (("near", True), ("far", False)):  # can a short n-gram at the question's end see the name?
            ix = [i for i, t in enumerate(items) if t["near"] == near and t["copies"] >= 8 and not t["maj"] and not t["early"] and not t["inq"]]
            out[f"facts.{kind}.name-{side}.nll"] = nll[ix]; out[f"facts.{kind}.name-{side}.pick"] = pick[ix]
        ix = [i for i, t in enumerate(items) if t["maj"] and not t["inq"]]  # the most common answer needs no memory, so it is scored apart
        out[f"facts.{kind}.common-answer.acc"] = acc[ix] if ix else np.zeros(1)
        if kind == "stored":
            ix = [i for i, t in enumerate(items) if t["early"] and not t["maj"] and not t["inq"]]
            out["forget.early.nll"] = nll[ix]; out["forget.early.pick"] = pick[ix]
    if natural:  # never written in the planted sentences: only what web text taught; above chance when positive
        _, nll, pick = fact_scores(ctx, natural)
        out["facts.natural.over-chance"] = pick - np.array([1 / len(i["cand"]) for i in natural]); out["facts.natural.nll"] = nll
    return out


def changed(ctx):
    """Change one part of a fact's question. Log-prob of the fact's answer in its own question minus in the changed one: positive means the model tells them apart."""
    stored, *_ = fact_sets(ctx); W = ctx.W

    def make():
        rng = np.random.default_rng(4); pre = lambda p, x: [EOT] + W.E(p[:p.find("[Y]")].replace("[X]", x)); out = []; pool = [i for i in stored if i["copies"] >= 8 and not i["maj"] and not i["inq"]]
        for i in pool:
            other = [j for j in pool if j["f"]["r"] == i["f"]["r"] and j["y"] != i["y"]]; rel = [j for j in pool if j["f"]["r"] != i["f"]["r"]]
            if not other or not rel: continue
            o, r = other[rng.integers(len(other))], rel[rng.integers(len(rel))]
            wording = r["f"]["S"][rng.integers(len(r["f"]["S"]))]; own = i["f"]["S"][rng.integers(len(i["f"]["S"]))]
            out.append((i["q"], pre(own, o["f"]["s"]), pre(wording, i["f"]["s"]), i["y"]))
        return out
    items = ctx.once("changed", make); lp = lambda k: [r[1][i[3]] for r, i in zip(ctx.probe([(i[k], [EOT]) for i in items]), items)]; base = np.array(lp(0))
    return {"changed.other-name.gap": base - np.array(lp(1)), "changed.other-relation.gap": base - np.array(lp(2))}


def track(ctx):
    """Values stated on the page. Right name: which of two named values does the question ask for? Latest: the same name given a new value. Log-prob gap of the right value over the wrong one."""
    W = ctx.W; N = min(ctx.c["val_rows"], 300)

    def make():
        rng = np.random.default_rng(5); top = np.argsort(-W.counts)[20:1000]; names = "Anna Ben Carl Dana Emma Frank Grace Henry Iris Jack Karen Leo Mary Nick Olga Paul Rose Sam Tina Victor".split(); out = []
        for _ in range(N):
            A, B = rng.choice(names, 2, replace=False); a, b = (int(x) for x in rng.choice(top, 2, replace=False)); q = lambda n: W.E(f"The code for {n} is"); dot = W.E(".")
            out.append((q(A) + [a] + dot + q(B) + [b] + dot + q(A), q(A) + [a] + dot + q(A) + [b] + dot + q(A), int(W.look[a]), int(W.look[b])))
        return out
    items = ctx.once("track", make); r1 = ctx.probe([(i[0], [EOT]) for i in items]); r2 = ctx.probe([(i[1], [EOT]) for i in items])
    return {"track.right-name.gap": np.array([r[1][i[2]] - r[1][i[3]] for r, i in zip(r1, items)]), "track.latest.gap": np.array([r[1][i[3]] - r[1][i[2]] for r, i in zip(r2, items)])}


def unknown(ctx):
    """Does the model become less sure when the name is made up? Entropy with a fake name minus entropy with a real one."""
    stored, _, fake, _, _ = fact_sets(ctx); ent = lambda lp: -float((np.exp(lp) * lp).sum())
    a = ctx.probe([(i["q"], i["a"]) for i in stored]); b = ctx.probe([(p, i["a"]) for p, i in zip(fake, stored)])
    return {"unknown.entropy-gap": np.array([ent(y[1]) - ent(x[1]) for x, y in zip(a, b)])}


def conflict(ctx):
    """The page states another answer just before the question. How much does it raise that answer, and lower the remembered one?"""
    _, _, _, page, _ = fact_sets(ctx); base = ctx.probe([(p["q"], [EOT]) for p in page]); paged = ctx.probe([(p["paged"], [EOT]) for p in page])
    return {"conflict.page-gain": np.array([c[1][p["y2"]] - b[1][p["y2"]] for b, c, p in zip(base, paged, page)]),
            "conflict.memory-drop": np.array([b[1][p["y"]] - c[1][p["y"]] for b, c, p in zip(base, paged, page)]),
            "conflict.follows-page": np.array([c[1][p["y2"]] > c[1][p["y"]] for c, p in zip(paged, page)], float)}


def context(ctx):
    """Using the page. Copy gain: loss on a span's first copy minus its second copy. Recall gain: how much a stated code raises that code's probability."""
    W = ctx.W; L = ctx.c["L"]; N = min(ctx.c["val_rows"], 300); val = ctx.val

    def make():
        rng = np.random.default_rng(2); top = np.argsort(-W.counts)[20:1000]
        real = np.array([[EOT] + r[4:16].tolist() * 2 for r in val[:N]]); rand = np.array([[EOT] + rng.choice(top, 12, replace=False).tolist() * 2 for _ in range(N)])
        q = W.E("The secret code is"); rec = {}
        for name, gap in (("near", 4), ("far", L - 14)):
            rec[name] = []
            for i in range(N):
                a = int(rng.choice(top)); f = val[(i + 1) % len(val), :gap].tolist(); rec[name].append((q + [a] + W.E(".") + f + q, f + q, int(W.look[a])))
        return real, rand, rec
    real, rand, rec = ctx.once("context", make); out = {}

    def gain(rows):
        nll, hit, ok = ctx.row_nll(rows); m = lambda a, b: (nll[:, a:b] * ok[:, a:b]).sum(1) / np.maximum(ok[:, a:b].sum(1), 1)
        return m(1, 12) - m(13, 24), hit[:, 13:24].mean(1)
    out["context.copy-text.gain"], out["context.copy-text.acc"] = gain(real); out["context.copy-random.gain"], _ = gain(rand)
    for name, items in rec.items():
        a = ctx.probe([(p, [EOT]) for p, _, _ in items]); b = ctx.probe([(p, [EOT]) for _, p, _ in items])
        out[f"context.recall-{name}.gain"] = np.array([x[1][i[2]] - y[1][i[2]] for x, y, i in zip(a, b, items)])
    return out


def messy(ctx):
    """Only the context is damaged and the target stays the same: how much worse is the target's loss? Lower is better."""
    W = ctx.W; L = ctx.c["L"]

    def make():
        rng = np.random.default_rng(3); rows = [r for r in ctx.val if W.look[r[-1]] != W.unk and r[-1] != EOT]

        def typo(t):
            w = t.split(" ")
            for i in range(len(w)):
                if len(w[i]) > 3 and rng.random() < 0.1:
                    j = rng.integers(1, len(w[i]) - 1); w[i] = w[i][:j] + w[i][j + 1] + w[i][j] + w[i][j + 2:]
            return " ".join(w)
        enc = lambda t: W.enc.encode(t, disallowed_special=())[-(L - 1):]; txt = [W.enc.decode(r[:-1].tolist()) for r in rows]; y = [[int(r[-1])] for r in rows]
        return [(r[:-1].tolist(), a) for r, a in zip(rows, y)], [(enc(t.lower()), a) for t, a in zip(txt, y)], [(enc(typo(t)), a) for t, a in zip(txt, y)], np.array([W.look[a[0]] for a in y])
    clean, lo, ty, ys = ctx.once("messy", make); nll = lambda items: np.array([-r[1][y] for r, y in zip(ctx.probe(items), ys)]); c = nll(clean)
    return {"messy.lower.delta": nll(lo) - c, "messy.typos.delta": nll(ty) - c}


def memorize(ctx):
    """Repeating training text word for word: loss and hit rate on rows the model trained on (compare with text.web)."""
    a = ctx.row_nll(ctx.W.rows[:ctx.c["val_rows"]])
    return {"memorize.train.loss": (a[0] * a[2]).sum(1) / np.maximum(a[2].sum(1), 1), "memorize.train.acc": (a[1] * a[2]).sum(1) / np.maximum(a[2].sum(1), 1)}


def bench_items(ctx, name):
    def make():
        rows = [json.loads(l) for l in open(os.path.join(BENCH, name + ".jsonl"))]; perm = np.random.default_rng(7).permutation(len(rows))
        rows = [rows[i] for i in perm[1 if ctx.final else 0::2][:ctx.c["bench"]]]; enc = ctx.W.enc; items = []
        for r in rows:
            seqs = []
            for pre, cont in r["pairs"]:
                p = (enc.encode(pre, disallowed_special=()) or [EOT])[-128:]; t = enc.encode(cont, disallowed_special=())[:64]; seqs.append((p + t, len(p)))
            items.append((seqs, r["answer"]))
        return items
    return ctx.once("bench_" + name, make)


def benchmarks(ctx):
    """Published multiple-choice benchmarks: acc is the usual score; margin (right option minus best wrong, per token) moves earlier."""
    out = {}
    for name in ("blimp", "hellaswag", "winogrande", "arc", "mmlu"):
        items = bench_items(ctx, name); flat = [s for seqs, _ in items for s in seqs]; lp = ctx.pair_logp(flat); acc, mar = [], []; k = 0
        for seqs, ans in items:
            v = lp[k:k + len(seqs)]; k += len(seqs); wrong = np.delete(v, ans)
            acc.append(float(v.argmax() == ans)); mar.append((v[ans] - wrong.max()) / max(1, np.mean([len(t) - st for t, st in seqs])))
        out[f"bench.{name}.acc"] = np.array(acc); out[f"bench.{name}.margin"] = np.array(mar)
    return out


FULL = [text, rarity, ambiguous, facts, changed, unknown, conflict, context, track, messy, memorize, kinds, benchmarks]


def run_all(ctx, level="full"):
    out = {}
    if level != "full": return {"text.tune.loss": ctx.row_loss(ctx.W.tune)}  # a search row holds no report score
    for g in FULL: out.update(g(ctx))
    return out
