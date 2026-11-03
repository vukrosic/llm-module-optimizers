# Working log

Every prediction here is committed before its run, so `git log` shows the order. Failed predictions stay.

## 2026-11-03: the question, closest work and kill test

**Question.** Should each module of an LLM (embedding, output head, attention query/key, attention value/output, MLP gate/up, MLP down, router, norm scales) use its own update rule, and how much is that choice worth once each module's learning rate is tuned as well?

**Closest work, read in full unless noted.**

| Paper | What it does | What it leaves open |
|---|---|---|
| Bernstein, Modular Manifolds (Thinking Machines 2025) | Each module has a manifold and a norm; a combined norm budgets the lr across layers. Only a small MLP on CIFAR-10. | Which manifold suits attention, embeddings and heads is listed as open. No LLM test. |
| 2606.13276 Different Layers, Different Manifolds (ICML 2026 workshop) | GPT-2 small, 50k steps: unconstrained 3.3855, All-Stiefel 3.3679, Stiefel attention + DGram MLP 3.3544. | Shared hyperparameters, one run per setting, no lr tuning per arm. Names automatic assignment as future work. |
| 2605.18106 Symmetry-Compatible Principle | RowNorm on embedding and head, hybrid on SwiGLU, router variants, on four 1B-class models. Gemma-style: 4.1046 to 4.0461. | Assignments come from symmetry theory, not from a search; authors say it is not an exhaustive benchmark. Embedding lr .5 and head lr .005 differ by 100x. |
| 2607.20486 OPTScientist | Program search over one rule for all 2D matrices; 0.798375 vs Muon 0.802949 BPB on NanoChat d21. | One rule for all matrices, no per-module search, no test split. |

Not read in full: 2607.20548, 2605.26842, 2502.16982, 2603.20527, 2601.23000, 1709.07417, 2506.01115.

**The one-sentence difference.** Nobody has searched per-module rule assignments under one protocol with the lr tuned per arm, against a control that tunes per-module lr alone with the same number of evaluations.

**Boring explanation.** Per-module rule gains may be only per-module step-size differences: the published rule assignments also move embedding and head lrs by 100x, and a rule that normalizes a gradient sets a step size as a side effect.

**Kill test K1** (arm definitions in `search.py`, written before it runs):

- `S_rule`: coordinate search over rule + lr on each module, starting from the Muon hybrid, chosen on tune rows.
- `S_lr`: coordinate search over lr alone on each module, same rules as the start, same number of evaluations.
- Score: `text.web.loss` on report rows, which the search never reads.
- Noise floor: runs of one setting with every lr multiplied by (1 + 0.001 k), `sigma` is their standard deviation.
- **Kill:** `S_lr - S_rule < max(2 sigma, 0.005)` at both 5s and 1m. The per-module rule question is then answered "not worth it at this size", and that is a publishable measurement if the test passes its controls, but the design search stops.

**Past local work to cite, not repeat.** `muon/rownorm-parts`: a 260-step gain did not hold at 1200 steps, so length checks are part of every claim. `muon/optimizer-design`: inverse-budget Muon wins at 400 steps and breaks at 1600. `muon/qk-adam`: q,k on the standard optimizer loses nothing, 2.635 vs 2.642.

## 2026-11-03: test the test, predictions before the baseline runs

Environment built: tiers, rules, baselines, bug checks, a port check against the authors' Manifold Muon (exact match in float32; their 16-bit arithmetic differs by up to 0.011 per entry on random inputs), and a poison check showing a search picks the same setting when report scores are noise.

Every baseline gets 3 tuning passes on tune rows. Published rankings I expect at 5s and 5sx4, lower is better on `text.web.loss`:

1. Muon recipe beats the standard baseline everywhere (Muon's own reports).
2. `sym-rownorm` (RowNorm embedding and head) beats `muon` (2605.18106).
3. `sym` (hybrid MLP on top) beats `sym-rownorm` (2605.18106).
4. `hetero` beats `stiefel`, and `stiefel` beats `muon` (2606.13276). I expect this one to fail at 250 steps, because the paper's run is 50k steps and constrained weights need time to pay back.
5. With QK-norm off, DGram on attention is flagged unstable and Stiefel is not (2606.13276).

Controls that must fail: `blind` must lose to every arm that reads gradients. `frozen-mlp` must lose to `muon`. `--leak` must lower report loss without lowering tune loss. `--double` must show twice the tokens and fail the compute cap.

Each run takes about 6 s on one thread. Jobs use 3 low-priority workers beside the other research.

## 2026-11-03: the first table was too noisy, so every row count went up

At 200 report rows the paired error on `text.web.loss` was 0.006, and the win rule needs z above 2 on a gap of 0.005, so the test could not pass or fail a design at its own floor. Tune rows had the same error, so a search could pick a setting on row noise. Now 800 rows per split at 5s, 5sx4, 5sx16, 1m and 5m. Muon against the standard baseline, same settings: gap 0.0229 at 200 rows, 0.0118 at 800 rows, error 0.0028, so the first gap was almost twice too large. Scoring tune rows costs about 1.2 s per run. No design had run. The 200-row table is kept in `old/` and is not comparable.
