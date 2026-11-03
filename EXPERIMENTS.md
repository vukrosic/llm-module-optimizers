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

**The one-sentence difference.** We did not find a search of per-module rule assignments under one protocol with the lr tuned per arm, against a control that tunes per-module lr alone.

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
- 2605.04055 and SGG (ACL 2025) modulate lr per group, with no choice of rule, so both belong on the lr side of our question.
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

Prediction: on fresh worlds the rule search beats lr-only by +.002 to +.008, and Lion on q and k does not survive at 1,000 steps (the search picks another rule there), because lion-all fell from 6th of 8 at 250 steps to 7th of 8 here, a weak reason. (Corrected after the commit: I first wrote 4th of 8. The predicted numbers are unchanged.) Drop the Lion idea if the search keeps it and the gap to lr-only stays under .002.

Result at 5sx4: the rule search moved no rule away from `sym`, the best recipe, in 193 rule settings, and its lrs refine `sym` only. Against `sym` on fresh worlds: +.0003, error .0008, per world -.0005 +.0012 +.0001 +.0019 -.0013, a tie. Lion on q and k did not survive, as predicted. The predicted +.002 to +.008 over lr-only failed, because the search found only lrs.

## Round 3: 1m, prediction before tuning

Width 64, 4 layers, hidden matrices hold 51%, 1,000 steps. Tune `--from-tier 5sx4`. Predictions: hidden training is worth over .15. Muon beats the standard baseline by .01 to .03. `sym` or `muon` is the best recipe. The rule search changes at most one rule, and its gap on fresh worlds against the best recipe is under .005, so K1 dies at 1m with probability 70%.


## Stopped, November 3, 2026

The 1m tuning was stopped before any result, because the fast experiments had already answered the question at these sizes and 1m was too slow alongside other jobs. So the kill test's second tier, `beat.py`, the hidden read and `speed.py` on designs did not run. RESULTS.md lists what did not run. The prediction for 1m stays unscored.


## Round 4: mixing rules at 5s, prediction before running

Vuk asked for a direct test: mix the rules module by module in fast runs, and either find rules for choosing them or show that one optimizer is enough. `mix.py` starts from the lr-only design (the hardest control so far), swaps one module to each rule, then runs random mixes, and checks the best mix on fresh worlds 15 to 19 against that base, Muon's recipe, Muon on every matrix (`muon-all`), the standard baseline and `sym`.

Predictions:

1. Muon on every matrix loses to Muon's recipe by more than .005 on fresh worlds, because swapping the embedding to Muon cost about .04 in the rule search's rows.
2. The embedding and head care most about their rule. Among hidden modules the best 4 rules sit within .01 of each other.
3. A mix scores about what its swaps add up to: correlation above .8 across 60 random mixes. If so, rules can be chosen one module at a time.
4. The best mix beats the lr-only design on fresh worlds by under .01, and it may tie.

Stop rule: if the best mix does not clear max(.005, 2 errors) against the lr-only design, the result at 5s is that no rule mix beats tuned lrs on Muon's recipe. One design goes to fresh worlds, so nothing is picked among them. Worlds 20 to 23 stay unread.

Result of round 4. The map: the embedding (best 4 rules span .029), head (.017), norm scales (.020) and attention value and output (.034) care about their rule; q,k (.0012), SwiGLU gate (.0043) and output (.0042) barely do. So prediction 2 failed on value and output. Mixes: correlation .881 with the sum of swaps, prediction 3 holds; a mix does .0137 better than the sum on average, and only 1 of 60 beat the base, with median 4.4306 against 4.3974. The base had 155 lr settings and the mixes had their lrs only from the map, which favours the base. Best mix: the lr-only design with Lion on q,k. On fresh worlds 15 to 19 it beats lr-only by +.0083, error .0012 (+.0101 +.0088 +.0068 +.0069 +.0086), so prediction 4 holds on size and the stop rule is cleared; it beats Muon's recipe by +.0106, `muon-all` by +.0345, the standard baseline by +.0271 and `sym` by +.0332.

This is nearly the design the rule search found, which scored +.0040 against lr-only on worlds 10 to 14, so it is a retest and not a second discovery, and the 1,000-step rule search did not choose Lion.

## Round 5: why q,k, prediction before running

In the map, Lion, Adam, RowNorm and normalized SGD on q,k all beat Muon there by .002 to .004. So the rule may be "q,k does not want Muon" and not "q,k wants Lion". With QK-norm the q and k vectors are normalized, so their length carries nothing, which may be why these matrices don't gain from Muon's control of the update's shape. That is a guess.

Predictions, all at 5s with lrs from world 0 tune rows, one setting per design, worlds 15 to 19:

1. Each of Adam, RowNorm and normalized SGD on q,k beats Muon on q,k by +.002 to +.008; Lion stays highest or within .002 of the highest.
2. The Lion mix on worlds 10 to 14 beats lr-only by +.003 to +.008, so part of the two worlds' difference is the design and part the worlds.
3. Without QK-norm, Lion on q,k gains under half of what it gains with QK-norm. I hold this weakly. If it gains as much, QK-norm is not the reason.

### Round 5 results

Worlds 15 to 19 unless noted, 5s tier, one training run per setting per world. Positive means the first rule wins.

| Comparison on q,k | Mean gap | Error | Verdict | Per world |
|---|---|---|---|---|
| Lion vs Muon | +.0083 | .0012 | holds | +.0101 +.0088 +.0068 +.0069 +.0086 |
| Adam vs Muon | +.0077 | .0013 | holds | +.0079 +.0068 +.0057 +.0068 +.0111 |
| RowNorm vs Muon | +.0063 | .0013 | holds | +.0073 +.0085 +.0049 +.0024 +.0086 |
| NormSGD vs Muon | +.0056 | .0012 | holds | +.0063 +.0054 +.0018 +.0051 +.0094 |
| Lion vs Adam | +.0006 | .0012 | tie | +.0022 +.0020 +.0011 +.0001 -.0025 |
| Lion vs RowNorm | +.0019 | .0011 | tie | +.0028 +.0003 +.0019 +.0045 -.0000 |
| Lion vs NormSGD | +.0026 | .0010 | tie | +.0038 +.0034 +.0050 +.0018 -.0008 |
| Lion vs Muon, worlds 10 to 14 | +.0067 | .0011 | holds | +.0082 +.0052 +.0089 +.0045 +.0066 |
| Lion vs Muon, no QK-norm | +.0039 | .0015 | tie | +.0087 +.0038 +.0045 -.0010 +.0033 |

Prediction 1 held: all three gaps fall inside +.002 to +.008 and Lion is highest. Prediction 2 held: +.0067 is inside +.003 to +.008. Prediction 3 held narrowly: .0039 is 47% of .0083, but the difference between the two gaps was not tested with its own error, so QK-norm is not shown to be the cause.

The rule this supports: q,k does not want Muon. It does not support "q,k wants Lion", because the four other rules tie.

## Round 6: can one rule with one knob replace the mix?

Every rule that beats Muon on q,k keeps some of the gradient's singular-value shape, and Muon flattens it completely. So I add `softmuon`: singular values raised to a power p, which runs from Muon (p 0) to normsgd (p 1). This rule is not from a paper.

Picks use tune rows of world 0. Verdicts use worlds 15 to 19, and worlds 21 to 23 have never been read.

Predictions, written before any run:

1. Control: p 0 on q,k lands within .002 of Muon on tune rows.
2. On q,k, every p from .25 up beats Muon on tune rows, and the best soft setting beats lr-only by +.004 to +.010 on fresh worlds and ties Lion there.
3. On value and output matrices the cost rises with p, and p .75 costs more than .015 on tune rows. On gate/up and down, the best p is at most .25 and moves tune rows by under .003.
4. The best single p on all four hidden modules loses to the mix (Lion on q,k) by +.002 to +.008. If it ties, one rule with one p is enough for hidden matrices, and I drop the claim that hidden matrices need two rules.
5. One rule on every matrix (Muon, Lion, RowNorm or NormSGD) loses to the mix by more than +.025 on fresh worlds. Norm scales stay on Adam in these arms, because the matrix rules can't run on vectors.
6. On worlds 21 to 23, the mix beats lr-only by +.004 to +.012.

### Round 6 results

Added tune loss against lr-only (tune 4.3974), each at its best of 5 lrs. Negative beats lr-only.

| Module | p 0.1 | p 0.25 | p 0.5 | p 0.75 |
|---|---|---|---|---|
| q,k | +.0001 | +.0008 | -.0019 | -.0029 |
| value, output | +.0081 | +.0131 | +.0253 | +.0305 |
| gate, up | +.0018 | +.0022 | +.0031 | +.0060 |
| down | +.0005 | +.0016 | +.0024 | +.0032 |

| Comparison | Worlds | Mean gap | Error | Verdict | Per world |
|---|---|---|---|---|---|
| p .75 on q,k vs lr-only | 15-19 | +.0056 | .0011 | holds | +.0067 +.0046 +.0036 +.0047 +.0084 |
| Lion on q,k vs p .75 on q,k | 15-19 | +.0026 | .0010 | tie | +.0034 +.0042 +.0032 +.0022 +.0001 |
| Mix vs p .1 on all hidden | 15-19 | +.0114 | .0019 | holds | +.0142 +.0130 +.0098 +.0153 +.0047 |
| p .1 on all hidden vs lr-only | 15-19 | -.0031 | .0020 | tie | -.0040 -.0042 -.0030 -.0083 +.0039 |
| Mix vs Muon on every matrix | 15-19 | +.0345 | .0021 | holds | +.0304 +.0352 +.0364 +.0362 +.0342 |
| Mix vs Lion on every matrix | 15-19 | +.0536 | .0023 | holds | +.0570 +.0593 +.0537 +.0522 +.0459 |
| Mix vs RowNorm on every matrix | 15-19 | +.0609 | .0025 | holds | +.0619 +.0676 +.0639 +.0579 +.0531 |
| Mix vs NormSGD on every matrix | 15-19 | +.1061 | .0031 | holds | +.1120 +.1109 +.0946 +.1063 +.1066 |
| Mix vs lr-only | 21-23 | +.0090 | .0013 | holds | +.0115 +.0083 +.0071 |
| Mix vs p .1 on all hidden | 21-23 | +.0104 | .0013 | holds | +.0101 +.0099 +.0114 |

Prediction 1 failed: p 0 on q,k added +.0026, over the .002 limit, so exact orthogonalization is worse than Muon's 5 Newton-Schulz steps there. Prediction 2 held in part: p .25 did not beat Muon on tune rows, while the fresh gap (+.0056) and the tie with Lion held. Prediction 3 held. Prediction 4 held in direction but the gap was larger than predicted (+.0114 against at most +.008), so one p is not enough for hidden matrices at this size. Prediction 5 held. Prediction 6 held on worlds no earlier run read.

softmuon uses an exact SVD, so it costs more per step than Muon. Its time per step was not measured one run at a time.

## Round 7: does the q,k rule survive 1,000 steps?

The rule search's gain did not hold at 1,000 steps, and this q,k gain is nearly the same design, so it may fade too. The lr of q,k is retuned at this length, on tune rows of world 0, and every other setting stays at the Muon recipe tuned for 1,000 steps, because no lr-only design exists at this length.

Prediction, written before any run: on worlds 15 to 19, the best of Lion and Adam on q,k beats Muon on q,k by +.000 to +.004 at 1,000 steps, which is under the .005 floor. If it clears the floor, the q,k rule is not specific to very short training.

### Round 7 results

| Comparison on q,k at 1,000 steps | Worlds | Mean gap | Error | Verdict | Per world |
|---|---|---|---|---|---|
| Lion vs Muon | 15-19 | +.0001 | .0018 | tie | -.0038 -.0041 +.0012 +.0025 +.0048 |
| Adam vs Muon | 15-19 | +.0010 | .0018 | tie | -.0047 -.0007 +.0014 +.0037 +.0053 |

On tune rows, Muon was better: Lion added +.0034 and Adam added +.0031. The prediction held: the q,k gain is specific to very short training in this environment.

## Round 8: does Adam on embedding and head survive 1,000 steps?

The q,k rule faded, so the larger gap has to be checked at the same length. Muon on every matrix is tuned at 1,000 steps with the same protocol as the other recipes.

Prediction, written before any run: on worlds 15 to 19 at 1,000 steps, the Muon recipe (Adam on embedding and head) beats Muon on every matrix by +.010 to +.040. If it is under the .005 floor, one rule is enough for matrices at this length.
