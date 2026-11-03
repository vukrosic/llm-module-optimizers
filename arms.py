"""Modules, spec parsing and the published baselines. A spec maps each module to a rule, an lr and optional knobs:
"emb=rownorm:.5;head=rownorm:.005;qk=muon:.02;vo=muon:.02;gu=hybrid:.02;down=hybrid:.02;norm=adam:.01:wd=.001"

Baselines, with lrs only as starting points because search.py tunes every arm on tune rows with the same grid:
- all-adam: the standard baseline everywhere.
- muon: Muon on hidden matrices, Adam on embedding, head and norm scales (Muon's own recipe).
- sym, sym-hybrid-all: 2605.18106, RowNorm or hybrid on embedding and head, Muon on attention, hybrid on SwiGLU; routers get routernorm.
- stiefel, hetero, hetero-inv, dgram: 2606.13276, Stiefel or DGram on attention and MLP, Adam on the rest.
- rownorm-all, lion-all, normsgd-all: one rule on every matrix.
- frozen-mlp, frozen-hidden, blind: controls. frozen-hidden never trains a hidden matrix, so it shows how much of the score hidden training can move at a tier.
No arm uses weight decay and momentum is fixed by each rule, because these tiers are too short for decay to matter. A spec can set both.
Not ported: NorMuon, RS-MR, SOAP, because each needs extra state and tuning we have not matched to its authors' code.
"""
import json, re

MODULES = ("emb", "head", "qk", "vo", "gu", "down", "norm")
MOE = ("router", "exp", "ed")
LRS = dict(adam=.003, muon=.02, muonsplit=.02, muontangent=.02, muonsphere=.02, muonhistory=.02, muonfixed=.02, muonbrake=.02, muonfade=.02, muongate=.02, adamtangent=.003, rownorm=.02, routernorm=.002, polar=.0005, leftpolar=.0005, hybrid=.02, lion=.0005, sgdm=.3, normsgd=.02, softmuon=.02, norm1d=.01, mm_stiefel=.02, mm_dgram=.02, freeze=0, blind=.02)
ROWS_IN_COLUMNS = {"down", "ed"}


def module_of(name, w):
    if name == "emb.weight": return "emb"
    if name == "head.weight": return "head"
    if w.ndim == 1: return "norm"
    m = re.search(r"\.(q|k|v|o|gate|up|down|router|eg|eu|ed)(\.\d+)?\.weight$", name)
    if not m: raise ValueError(f"no module for {name}")
    return dict(q="qk", k="qk", v="vo", o="vo", gate="gu", up="gu", down="down", router="router", eg="exp", eu="exp", ed="ed")[m.group(1)]


def parse(text):
    if text in ARMS: return ARMS[text]()
    out = {}
    for part in filter(None, text.split(";")):
        m, _, rest = part.partition("="); bits = rest.split(":"); out[m] = dict(rule=bits[0], lr=float(bits[1]) if len(bits) > 1 and bits[1] else LRS[bits[0]])
        for b in bits[2:]:
            k, v = b.split("="); out[m][k] = float(v)
    return out


def show(spec):
    """One canonical string for a spec, so a spec is the same row wherever it came from."""
    part = lambda m, g: f"{m}={g['rule']}:{g['lr']:g}" + "".join(f":{k}={g[k]:g}" for k in sorted(g) if k not in ("rule", "lr"))
    return ";".join(part(m, spec[m]) for m in sorted(spec))


def fill(rules, **lrs):
    return {m: dict(rule=r, lr=lrs.get(m, lrs.get(r, LRS[r]))) for m, r in rules.items()}


def base(emb="adam", head="adam", att="muon", mlp="muon", router="rownorm", **lrs):
    return fill(dict(emb=emb, head=head, qk=att, vo=att, gu=mlp, down=mlp, norm="adam", router=router, exp=mlp, ed=mlp), **lrs)


ARMS = {
    "all-adam": lambda: base(att="adam", mlp="adam", router="adam"),
    "muon": lambda: base(router="adam"),
    "muon-all": lambda: base(emb="muon", head="muon", router="adam"),  # Muon on every matrix; norm scales are vectors, which Muon can't orthogonalize, so they stay on Adam
    "sym": lambda: base(emb="rownorm", head="rownorm", mlp="hybrid", router="routernorm"),
    "sym-rownorm": lambda: base(emb="rownorm", head="rownorm", router="routernorm"),
    "sym-hybrid-all": lambda: base(emb="hybrid", head="hybrid", mlp="hybrid", router="routernorm"),
    "stiefel": lambda: base(att="mm_stiefel", mlp="mm_stiefel", router="adam"),
    "hetero": lambda: base(att="mm_stiefel", mlp="mm_dgram", router="adam"),
    "hetero-inv": lambda: base(att="mm_dgram", mlp="mm_stiefel", router="adam"),
    "dgram": lambda: base(att="mm_dgram", mlp="mm_dgram", router="adam"),
    "rownorm-all": lambda: base(emb="rownorm", head="rownorm", att="rownorm", mlp="rownorm"),
    "lion-all": lambda: base(emb="lion", head="lion", att="lion", mlp="lion", router="lion"),
    "normsgd-all": lambda: base(emb="normsgd", head="normsgd", att="normsgd", mlp="normsgd", router="normsgd"),
    "frozen-mlp": lambda: base(mlp="freeze", router="adam"),
    "frozen-hidden": lambda: base(att="freeze", mlp="freeze", router="adam"),
    "blind": lambda: dict(base(emb="blind", head="blind", att="blind", mlp="blind", router="blind"), norm=dict(rule="blind", lr=LRS["blind"])),
}


def groups(spec, named):
    """named = model.named_parameters(). Modules the model lacks are dropped; a model module the spec lacks is an error."""
    use = {}
    for n, w in named:
        use.setdefault(module_of(n, w), []).append(w)
    missing = [m for m in use if m not in spec]
    if missing: raise ValueError(f"spec has no rule for {missing}")
    out = {}
    for m, ps in use.items():
        g = dict(spec[m]); g["params"] = ps; g["t"] = m in ROWS_IN_COLUMNS; out[m] = g
    return out
