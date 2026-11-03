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


RULES = dict(adam=adam, muon=muon, rownorm=rownorm, routernorm=routernorm, polar=polar, leftpolar=leftpolar, hybrid=hybrid, lion=lion, sgdm=sgdm, normsgd=normsgd, softmuon=softmuon, norm1d=norm1d,
             mm_stiefel=mm_stiefel, mm_dgram=mm_dgram, freeze=freeze, blind=blind)
ONLY_2D = {"muon", "rownorm", "routernorm", "polar", "leftpolar", "hybrid", "normsgd", "softmuon", "mm_stiefel", "mm_dgram"}
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

