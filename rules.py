"""Update rules, one per module, behind one optimizer. A rule takes (weight, gradient, state, lr, knobs) and moves the weight in place.

Ported from the papers, with what was adapted:
- muon: Keller Jordan's Muon with Nesterov momentum and his sqrt(max(1, m/n)) scale, 5 Newton-Schulz steps.
- rownorm, polar, leftpolar, hybrid: 2605.18106, Algorithms for RowNormM, RightPolarGradM, LeftPolarGradM, HybridPolarGradM,
  M = beta M + (1 - beta) G, eps 1e-8, beta .95. The paper uses Polar Express and Gram Newton-Schulz with 5 inner steps and
  eps_NS 1e-7; here `polar` uses an exact eigendecomposition with the same damping, slower but with the same result up to their
  approximation error. nu^alpha is the nuclear-norm factor, alpha 1 unless a spec sets it. `routernorm` is the centered RowNorm.
- mm_stiefel, mm_dgram: Manifold Muon from 2606.13276, matching the authors' manifold_muon.py and msign.py (debug/port_check.py
  compares the two on random inputs). K = 20, alpha = 0.1 as in their paper. Weights start on the manifold.
- adam: betas .9, .95 with a decoupled decay knob `wd`, default 0.
- lion: sign of momentum, beta .9, from Chen et al. 2023.
- sgdm: heavy ball. normsgd: Muon's Nesterov direction scaled to Muon's Frobenius size sqrt(m), with no orthogonalization.
- softmuon: not from a paper, written here. It keeps singular values raised to a power p, so it runs from Muon (p 0) to normsgd (p 1).

Controls: freeze never moves. blind takes a random direction with Muon's Frobenius size, so it must lose to everything that reads gradients.

`down` and `ed` weights have their neurons in columns, so row rules run on their transpose (2605.18106, SwiGLU section). The `t` knob
sets this, and arms.py sets it from the module name.
"""
import torch

COEF = [(8.28721201814563, -23.595886519098837, 17.300387312530933), (4.107059111542203, -2.9478499167379106, 0.5448431082926601), (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
        (3.3184196573706015, -2.488488024314874, 0.51004894012372), (2.300652019954817, -1.6689039845747493, 0.4188073119525673), (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
        (1.8750014808534479, -1.2500016453999487, 0.3750001645474248), (1.875, -1.25, 0.375)]
STABLE = [(a / 1.01, b / 1.01 ** 3, c / 1.01 ** 5) for a, b, c in COEF[:-1]] + [COEF[-1]]


def newton_schulz(G, steps=5):
    X = G / (G.norm() + 1e-7); tall = X.shape[0] > X.shape[1]
    if tall: X = X.T
    for _ in range(steps):
        A = X @ X.T; X = 3.4445 * X + (-4.7750 * A + 2.0315 * A @ A) @ X
    return X.T if tall else X


def msign(G, steps=10):
    """Polar Express, as in the Manifold Muon authors' code; float32 here and bfloat16 there."""
    tall = G.shape[-2] > G.shape[-1]; x = G.mT if tall else G; x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.01)
    for i in range(steps):
        a, b, c = STABLE[min(i, len(STABLE) - 1)]; s = x @ x.mT; y = c * s; y.diagonal().add_(b); y = y @ s; y.diagonal().add_(a); x = y @ x
    return torch.nan_to_num(x.mT if tall else x)


def side(w, k): return w.T if k.get("t") else w


def heavy(st, w, g, beta):
    st.setdefault("m", torch.zeros_like(w)); st["m"].mul_(beta).add_(g, alpha=1 - beta); return st["m"]


def nesterov(st, w, g, beta):
    st.setdefault("b", torch.zeros_like(w)); st["b"].mul_(beta).add_(g); return g + beta * st["b"]


def adam(w, g, st, lr, k):
    b1, b2 = k.get("b1", .9), k.get("b2", .95); wd = k.get("wd", 0.0)
    if not st: st.update(t=0, m=torch.zeros_like(w), v=torch.zeros_like(w))
    st["t"] += 1; st["m"].mul_(b1).add_(g, alpha=1 - b1); st["v"].mul_(b2).addcmul_(g, g, value=1 - b2)
    if wd: w.mul_(1 - lr * wd)
    w.add_((st["m"] / (1 - b1 ** st["t"])) / ((st["v"] / (1 - b2 ** st["t"])).sqrt() + 1e-8), alpha=-lr)


def muon(w, g, st, lr, k):
    beta = k.get("beta", .95); w.add_(newton_schulz(nesterov(st, w, g, beta)) * max(1.0, w.shape[0] / w.shape[1]) ** 0.5, alpha=-lr)


def muonsplit(w, g, st, lr, k):
    """Muon with each attention head's rows orthogonalized on their own: Per-Head Muon in Kimi K3, Muon Split in GLM-5, MuonSplit in 2605.08933. h is the head count. Rows hold a head's output, so the step has Muon's Frobenius size."""
    d = nesterov(st, w, g, k.get("beta", .95)); n = w.shape[0] // int(k.get("h", 4))
    for i in range(0, w.shape[0], n): w[i:i + n].add_(newton_schulz(d[i:i + n]), alpha=-lr)


def heads(w, k):
    n = w.shape[0] // int(k.get("h", 4)); return [w[i:i + n] for i in range(0, w.shape[0], n)]


def tangent(z, b):
    """z with its part along b removed, so a step along z changes b's direction and, to first order, not its size."""
    return z - (z * b).sum() / ((b * b).sum() + 1e-12) * b


def muontangent(w, g, st, lr, k):
    """Per-head Muon with the part along each head's rows removed, which is MuonGS's projection (2610.37647) applied per head. Query-key normalization makes a head's output ignore the size of its rows, so that part only grows the rows. The step keeps Muon's Frobenius size."""
    d = nesterov(st, w, g, k.get("beta", .95))
    for b, e in zip(heads(w, k), heads(d, k)): z = tangent(newton_schulz(e), b); b.add_(z / (z.norm() + 1e-7) * b.shape[0] ** 0.5, alpha=-lr)


def muonsphere(w, g, st, lr, k):
    """muontangent that then puts each head's rows back at their first size, close to MuonH (2606.16899) per head. The rows' size never grows, so one lr turns every head by the same angle at every step of training."""
    d = nesterov(st, w, g, k.get("beta", .95)); st.setdefault("r", [float(b.norm()) for b in heads(w, k)])
    for b, e, r in zip(heads(w, k), heads(d, k), st["r"]): z = tangent(newton_schulz(e), b); b.add_(z / (z.norm() + 1e-7) * b.shape[0] ** 0.5, alpha=-lr); b.mul_(r / (b.norm() + 1e-12))


def muonfixed(w, g, st, lr, k):
    """Muon's whole step, then each head's rows back at their first size. Only the size growth goes, so a gap to Muon is the effect of that growth."""
    st.setdefault("r", [float(b.norm()) for b in heads(w, k)]); muon(w, g, st, lr, k)
    for b, r in zip(heads(w, k), st["r"]): b.mul_(r / (b.norm() + 1e-12))


def muonbrake(w, g, st, lr, k):
    """muonfixed with the turning angle cut by a schedule, lr / (1 + t / tau). Muon at lr .02 grew from 7.9 to 15.6 in 600 steps, so its size doubles in about 630 steps, and tau 630 stands in for that growth."""
    st["t"] = st.get("t", 0) + 1; muonfixed(w, g, st, lr / (1 + st["t"] / k.get("tau", 630.)), k)


def muonfade(w, g, st, lr, k):
    """muonsphere with a falling lr, lr / (1 + t / tau). Muon's size growth works like a hidden lr schedule, so this sets that schedule on purpose, per head, with no growth. Tau 150 takes lr .02 to .005 by step 450."""
    st["t"] = st.get("t", 0) + 1; muonsphere(w, g, st, lr / (1 + st["t"] / k.get("tau", 150.)), k)


def muongate(w, g, st, lr, k):
    """Per-head Muon that lets a head grow only once its gradient stops agreeing with its momentum, which is when it has learned what it can. Then the brake builds after learning, not during it. u is the running cosine, and at u of tau or more no growth is allowed."""
    prev = st["b"].clone() if "b" in st else None; d = nesterov(st, w, g, k.get("beta", .95)); tau, n = k.get("tau", .2), w.shape[0] // int(k.get("h", 4)); u = st.setdefault("u", [tau] * int(k.get("h", 4)))
    for j, (b, e, x) in enumerate(zip(heads(w, k), heads(d, k), heads(g, k))):
        if prev is not None: y = prev[j * n:(j + 1) * n]; u[j] = .98 * u[j] + .02 * float((x * y).sum() / (x.norm() * y.norm() + 1e-12))
        z = newton_schulz(e); keep = max(0., min(1., 1 - u[j] / tau)); b.add_(z - (1 - keep) * (z * b).sum() / ((b * b).sum() + 1e-12) * b, alpha=-lr)


def muonhistory(w, g, st, lr, k):
    """muonsphere with Muon's orthogonalization replaced by a sum over training. G adds each step's normalized gradient product on the input side, and the step is the Nesterov direction times G^(-1/2). The step keeps a fixed size, so it goes mostly to input directions that have had little update."""
    d = nesterov(st, w, g, k.get("beta", .95)); st.setdefault("r", [float(b.norm()) for b in heads(w, k)]); n = g / (g.norm() + 1e-7); st.setdefault("G", torch.zeros(w.shape[1], w.shape[1])); st["G"].add_(n.T @ n); e, v = torch.linalg.eigh(st["G"]); e = e.clamp_min(0) + k.get("eps", .01) * e.clamp_min(0).mean() + 1e-12; root = (v / e.sqrt()) @ v.T; z = d @ root
    for b, y, r in zip(heads(w, k), heads(z, k), st["r"]): t = tangent(y, b); b.add_(t / (t.norm() + 1e-7) * b.shape[0] ** 0.5, alpha=-lr); b.mul_(r / (b.norm() + 1e-12))


def adamtangent(w, g, st, lr, k):
    """Adam with the part along each head's rows removed, which is AdamP's idea (2006.08217) applied to each head of the query and key matrices."""
    a, b = k.get("b1", .9), k.get("b2", .999); st.setdefault("t", 0); st["t"] += 1; st.setdefault("m", torch.zeros_like(w)).mul_(a).add_(g, alpha=1 - a); st.setdefault("v", torch.zeros_like(w)).mul_(b).addcmul_(g, g, value=1 - b); z = (st["m"] / (1 - a ** st["t"])) / ((st["v"] / (1 - b ** st["t"])).sqrt() + 1e-8)
    for x, e in zip(heads(w, k), heads(z, k)): x.add_(tangent(e, x), alpha=-lr)


def rownorm(w, g, st, lr, k):
    M = side(heavy(st, w, g, k.get("beta", .95)), k); Z = M / (M.norm(dim=1, keepdim=True) + 1e-8); w.add_(Z.T if k.get("t") else Z, alpha=-lr)


def routernorm(w, g, st, lr, k):
    """RowNorm with each row's mean over experts removed, because only score differences matter for routing."""
    M = heavy(st, w, g, k.get("beta", .95)); P = M - M.mean(0, keepdim=True); w.add_(P / (P.norm(dim=1, keepdim=True) + 1e-8), alpha=-lr)


def inv_sqrt(C):
    e, V = torch.linalg.eigh(C + 1e-8 * torch.eye(len(C), dtype=C.dtype)); return (V * e.clamp_min(1e-12) ** -.5) @ V.T


def nu_of(C, R): return max(float((C * R).sum()), 1e-8)


def right_polar(M, alpha):
    C = M.T @ M; R = inv_sqrt(C); return nu_of(C, R) ** alpha * (M @ R)


def polar(w, g, st, lr, k):
    M = side(heavy(st, w, g, k.get("beta", .95)), k); Z = right_polar(M, k.get("alpha", 1.0)); w.add_(Z.T if k.get("t") else Z, alpha=-lr)


def leftpolar(w, g, st, lr, k):
    M = side(heavy(st, w, g, k.get("beta", .95)), k).T; Z = right_polar(M, k.get("alpha", 1.0)).T; w.add_(Z.T if k.get("t") else Z, alpha=-lr)


def hybrid(w, g, st, lr, k):
    """Right spectral then row norm: 2605.18106's row-norm/right-spectral HybridPolarGradM. The row norm makes nu^alpha cancel, so alpha has no effect here."""
    M = side(heavy(st, w, g, k.get("beta", .95)), k); Z = right_polar(M, 0.0); Z = Z / (Z.norm(dim=1, keepdim=True) + 1e-8); w.add_(Z.T if k.get("t") else Z, alpha=-lr)


def lion(w, g, st, lr, k): w.add_(heavy(st, w, g, k.get("beta", .9)).sign(), alpha=-lr)


def sgdm(w, g, st, lr, k):
    st.setdefault("b", torch.zeros_like(w)); st["b"].mul_(k.get("beta", .9)).add_(g); w.add_(st["b"], alpha=-lr)


def normsgd(w, g, st, lr, k):
    d = nesterov(st, w, g, k.get("beta", .95)); w.add_(d / (d.norm() + 1e-7) * w.shape[0] ** 0.5, alpha=-lr)


def softmuon(w, g, st, lr, k):
    """One knob between Muon and normsgd: singular values of the Nesterov direction raised to p, then scaled to their Frobenius size sqrt(m). p 0 is exact orthogonalization, p 1 is normsgd. Uses an exact SVD."""
    d = nesterov(st, w, g, k.get("beta", .95)); U, S, Vt = torch.linalg.svd(d / (d.norm() + 1e-7), full_matrices=False); Z = (U * S ** k.get("p", .5)) @ Vt; w.add_(Z / (Z.norm() + 1e-7) * w.shape[0] ** 0.5, alpha=-lr)


def norm1d(w, g, st, lr, k):
    d = nesterov(st, w, g, k.get("beta", .95)); w.add_(d / (d.norm() + 1e-7) * w.numel() ** 0.5, alpha=-lr)


def gram(M, kind): return M if kind == "stiefel" else M * (1 - torch.eye(M.shape[-1], dtype=M.dtype))


def retract(W, kind):
    if kind == "stiefel": return msign(W)
    U, S, Vt = torch.linalg.svd(W, full_matrices=False); return (U @ Vt) * S.unsqueeze(-2)


def manifold_step(w, G, eta, kind, K=20, a=0.1):
    """Dual ascent for the tangent step, then retract: W assumed on the manifold, spectral norm of the step at most eta."""
    tall = w.shape[0] >= w.shape[1]; W, G = (w, G) if tall else (w.T, G.T); L = -.25 * gram(W.T @ G + G.T @ W, kind)
    for i in range(K):
        A = msign(G + 2 * W @ gram(.5 * (L + L.T), kind)); H = gram(W.T @ A + A.T @ W, kind); L = L - a * (1 - i / K) * H
    new = retract(W - eta * A, kind); return new if tall else new.T


def on_manifold(w, kind):
    tall = w.shape[0] >= w.shape[1]; r = retract(w if tall else w.T, kind); return r if tall else r.T


def mm_stiefel(w, g, st, lr, k): w.copy_(manifold_step(w, nesterov(st, w, g, k.get("beta", .95)), lr, "stiefel", k.get("K", 20)))


def mm_dgram(w, g, st, lr, k): w.copy_(manifold_step(w, nesterov(st, w, g, k.get("beta", .95)), lr, "dgram", k.get("K", 20)))


def freeze(w, g, st, lr, k): pass


def blind(w, g, st, lr, k):
    d = torch.randn_like(w); w.add_(d / (d.norm() + 1e-7) * w.shape[0] ** 0.5, alpha=-lr)


RULES = dict(adam=adam, muon=muon, muonsplit=muonsplit, muontangent=muontangent, muonsphere=muonsphere, muonhistory=muonhistory, muonfixed=muonfixed, muonbrake=muonbrake, muonfade=muonfade, muongate=muongate, adamtangent=adamtangent, rownorm=rownorm, routernorm=routernorm, polar=polar, leftpolar=leftpolar, hybrid=hybrid, lion=lion, sgdm=sgdm, normsgd=normsgd, softmuon=softmuon, norm1d=norm1d,
             mm_stiefel=mm_stiefel, mm_dgram=mm_dgram, freeze=freeze, blind=blind)
ONLY_2D = {"muon", "muonsplit", "muontangent", "muonsphere", "muonhistory", "muonfixed", "muonbrake", "muonfade", "muongate", "adamtangent", "rownorm", "routernorm", "polar", "leftpolar", "hybrid", "normsgd", "softmuon", "mm_stiefel", "mm_dgram"}
ONLY_1D = {"norm1d"}
MANIFOLD = {"mm_stiefel": "stiefel", "mm_dgram": "dgram"}


class ModuleOpt:
    """groups = {module: dict(rule=, lr=, params=[...], **knobs)}. schedule scales every lr. Records each module's update size relative to its weights."""
    def __init__(s, groups):
        s.groups = groups; s.state = {id(w): {} for g in groups.values() for w in g["params"]}; s.rel = {m: [] for m in groups}
        for m, g in groups.items():
            for w in g["params"]:
                if g["rule"] in ONLY_2D and w.ndim != 2: raise ValueError(f"{g['rule']} needs matrices, {m} has shape {tuple(w.shape)}")
                if g["rule"] in ONLY_1D and w.ndim != 1: raise ValueError(f"{g['rule']} needs vectors, {m} has shape {tuple(w.shape)}")
                if g["rule"] in MANIFOLD:
                    with torch.no_grad(): w.copy_(on_manifold(w, MANIFOLD[g["rule"]]))

    @torch.no_grad()
    def step(s, sc=1.0, measure=False):
        for m, g in s.groups.items():
            f = RULES[g["rule"]]; lr = g["lr"] * sc; top = bot = 0.0
            for w in g["params"]:
                if w.grad is None: continue
                old = w.clone() if measure else None
                f(w, w.grad, s.state[id(w)], lr, g)
                if measure: top += float((w - old).pow(2).sum()); bot += float(old.pow(2).sum())
            if measure and bot: s.rel[m].append((top / bot) ** .5)

    def state_bytes(s):
        return sum(v.numel() * v.element_size() for st in s.state.values() for v in st.values() if torch.is_tensor(v))

    def rel_sizes(s): return {m: round(sum(v) / len(v), 6) for m, v in s.rel.items() if v}

