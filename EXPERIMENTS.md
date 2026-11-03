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

## Old test results, kept in `old/`, not comparable (November 3, 2026)

800 rows at 5s with 4,000 tokens, tune/report: muon 5.5305/5.5461; standard 5.5340/5.5443; normsgd-all 5.5457/5.5608; sym-hybrid-all 5.5488/5.5672; frozen-mlp 5.5544/5.5709; sym 5.5546/5.5730; stiefel 5.5576/5.5734; sym-rownorm 5.5599/5.5799; rownorm-all 5.5772/5.5941; lion-all 5.5832/5.5939; blind 8.1106/8.1271. Muon against standard: -.0018, -.0032, -.0126 across worlds 0, 1, 2. At 1,000 steps: sym-rownorm 5.1793, muon 5.1949, standard 5.2012, so RowNorm on embedding and head reversed with length.

## Review 1 and test version 3

`reviews/review-1.md` holds the 17 findings and what was done. Test version 3: 3,200 rows per split, hidden matrices hold 37% at 5s, 51% at 1m, 54% at 15m, one lr per module for every arm, fresh worlds 10 to 14 for verdicts, world 20 for the hidden read. Fact scores don't exist at 5s and have 16 facts at 1m, so these tiers judge language and context scores only, and that limit gets stated in the write-up. Facts appear from 15m (304).

## Predictions for test version 3, written before any tuning

1. Hidden training (Muon against `frozen-hidden`) is worth .10 to .40 at 5s and more at 1m, so both tiers count. Dropped tier if under .05.
2. Muon against the standard baseline is a tie at 5s (under .01), and Muon leads by .005 to .03 at 5sx4 and 1m.
3. RowNorm on embedding and head beats Adam there at 5sx4 by .005 to .02, as it did on the old test.
4. Hybrid against Muon MLP is a tie with equal lr groups, gap under .005 at every tier.
5. K1: the rule search wins on tune rows by .005 to .02 and keeps at most half of that on fresh worlds. Probability of clearing K1 at both 5s and 1m is about 25%. If it dies, the negative result (per-module rules add nothing beyond per-module lrs under one protocol) is the paper's main claim.

## More closest work, read November 3, 2026

- 2605.09176 (survey, "Navigating LLM Valley") lists whether optimizer assignment can be learned or adapted as an open question. arXiv 2608.28557 (survey) gives a routing table by parameter class. Both are hand assignment.
- 1709.07417 searches for one update equation on a ConvNet on CIFAR-10. Abstract only read: it does not say whether rules differ by layer.
- 2605.04055 (MetaAdamW) and SGG (ACL 2025) modulate lr per group, with no choice of rule, so both belong on the lr side of our question.
- Celo and Celo2 (2501.12670, 2602.19142) learn per-parameter rules, not rules assigned per module. Not read in full.

Two extended searches found no per-module rule search on transformers with lr tuned per arm. That does not show none exists, so any write-up says "we did not find one", and these full texts still need reading before that sentence is used.

## Test version 3 at 5s, baselines (one training run per setting per world, 5 worlds, report rows)

Prediction 1 holds: hidden training is worth .103 (21x the floor). Prediction 2 fails at 5s: Muon beats the standard baseline by +.0162, error .0015, same sign in all 5 worlds; I predicted a tie. Prediction 4 holds: hybrid against Muon MLP -.0008, a tie. RowNorm on embedding and head loses to Adam there by -.0239 at 250 steps. All-Stiefel loses to unconstrained by -.0338 at 250 steps, which reverses the paper's 50k-step result on GPT-2 small, so length or size may explain it. Stiefel attention + DGram MLP against All-Stiefel: tie. Jitter sd .0004, world spread about .003. Leak: report -.128, tune -.027. Double: 384,000 tokens, 2 gradients per step.

Commit before the 5s rule search. K1 prediction stays as written above.

## K1 at 5s, and round 1: is the gain only more lr tuning?

Rule search, 218 settings: Lion on attention q and k (lr .0031), everything else Muon or Adam at refined lrs. Against Muon at 62 settings on fresh worlds 10 to 14: mean +.0073, error .0031, per world +.0180 +.0072 +.0067 +.0056 -.0010. It clears K1's bar at 5s and fails `beat.py`'s sign rule, and its lrs differ from the control on every module, so the result can't separate rule from lr.

Prediction before running, for Muon with about 200 extra lr-only settings (`lronly`) on the same worlds: lr-only gains +.002 to +.005 over Muon, and the rule search beats lr-only by +.002 to +.006. So Lion on q and k alone is worth under .005 at this tier, and I'd drop the Lion idea if the gap to lr-only is under .002. Probability the rule search beats lr-only by over .005 and 2 errors: 25%.

Result: lr-only (155 settings) against Muon: +.0033, error .0019, a tie. Rule search against lr-only: +.0040, error .0012, positive in all 5 worlds (+.0080 +.0047 +.0024 +.0041 +.0007), under the .005 floor. Both match the prediction. At 250 steps Lion on q and k gives a small real gain that doesn't reach the floor. Next: does it grow at 1,000 steps and at 1m?

2605.09176 section 12.5 (read November 3, 2026) frames per-group rules as an open direction with no citations and reports no study that compares or searches assignments. Its five open questions include which groups need coordinate-wise adaptivity and whether embeddings and heads need separate optimizers. Our tables answer those questions under one protocol at the sizes tested, whatever K1 says.

## Round 2: 1,000 steps (5sx4), prediction before the rule search

Tuned on tune rows so far: sym 4.145, muon 4.1488, sym-rownorm 4.1519, sym-hybrid-all 4.1577, standard 4.1636, rownorm-all 4.1685, lion-all 4.1844, normsgd-all 4.1979. The search starts from `sym`, the best recipe.

Prediction: on fresh worlds the rule search beats lr-only by +.002 to +.008, and Lion on q and k does not survive at 1,000 steps (the search picks another rule there), because lion-all fell from 4th of 8 at 250 steps to 7th of 8 here. Drop the Lion idea if the search keeps it and the gap to lr-only stays under .002.
