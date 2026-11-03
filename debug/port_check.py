"""Does our Manifold Muon match the authors' code (github.com/kiratoyoshihara/module-wise-manifold-muon, MIT)? Their manifold_muon and msign are pasted below, trimmed to the path we use. One question, a few seconds."""
import math, os, signal, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rules
signal.alarm(7)


HALF = [True]
THEIR_ABC = [(8.28721201814563, -23.595886519098837, 17.300387312530933), (4.107059111542203, -2.9478499167379106, 0.5448431082926601), (3.9486908534822946, -2.908902115962949, 0.5518191394370137), (3.3184196573706015, -2.488488024314874, 0.51004894012372),
             (2.300652019954817, -1.6689039845747493, 0.4188073119525673), (1.891301407787398, -1.2679958271945868, 0.37680408948524835), (1.8750014808534479, -1.2500016453999487, 0.3750001645474248), (1.875, -1.25, 0.375)]
THEIR_STABLE = [(a / 1.01, b / 1.01 ** 3, c / 1.01 ** 5) for a, b, c in THEIR_ABC[:-1]] + [THEIR_ABC[-1]]


def their_msign(G, steps=10):
    tr = G.size(-2) > G.size(-1); x = G.bfloat16() if HALF[0] else G.clone()
    if tr: x = x.mT
    x /= x.norm(dim=(-2, -1), keepdim=True) * 1.01
    for step in range(steps):
        a, b, c = THEIR_STABLE[step] if step < len(THEIR_STABLE) else THEIR_STABLE[-1]
        s = x @ x.mT; y = c * s; y.diagonal(dim1=-2, dim2=-1).add_(b); y = y @ s; y.diagonal(dim1=-2, dim2=-1).add_(a); x = y @ x
    if tr: x = x.mT
    return torch.nan_to_num(x).float()


def their_P(M, manifold):
    return M if manifold == "stiefel" else M * (1 - torch.eye(M.shape[-1]))


def their_retract(W, manifold):
    if manifold == "stiefel": return their_msign(W)
    U, S, Vt = torch.linalg.svd(W, full_matrices=False); return (U @ Vt) * S.unsqueeze(-2)


def their_step(W, G, eta, alpha, steps, manifold):
    tr = W.shape[-2] < W.shape[-1]
    if tr: W, G = W.mT, G.mT
    P = lambda M: their_P(M, manifold); L = -0.25 * P(W.mT @ G + G.mT @ W)
    for step in range(steps):
        A = their_msign(G + 2 * W @ P(0.5 * (L + L.mT))); H = P(W.mT @ A + A.mT @ W); L = L - alpha * (1 - step / steps) * H
    new = their_retract(W - eta * A, manifold)
    return new.mT if tr else new


for half in (True, False):
    HALF[0] = half; torch.manual_seed(0); worst = 0
    for shape in [(16, 16), (16, 42), (42, 16)]:
        for kind in ("stiefel", "dgram"):
            W = rules.on_manifold(torch.randn(*shape), kind); G = torch.randn(*shape)
            a, b = rules.manifold_step(W, G, .02, kind), their_step(W, G, .02, .1, 20, kind); worst = max(worst, float((a - b).abs().max()))
    print(f"authors' code {'in bfloat16' if half else 'in float32'}: largest gap {worst:.1e}")
assert worst < 1e-4, "port differs from the authors' code when both use float32"
print("port matches the authors' code; the remaining gap is their 16-bit arithmetic")
