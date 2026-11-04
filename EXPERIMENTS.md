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

The next question was a direct test: mix the rules module by module in fast runs, and either find rules for choosing them or show that one optimizer is enough. `mix.py` starts from the lr-only design (the hardest control so far), swaps one module to each rule, then runs random mixes, and checks the best mix on fresh worlds 15 to 19 against that base, Muon's recipe, Muon on every matrix (`muon-all`), the standard baseline and `sym`.

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

### Round 8 results

Muon recipe vs Muon on every matrix, 1,000 steps, worlds 15 to 19: +.0074, error .0017, holds; per world +.0092 +.0110 +.0036 +.0033 +.0101. Tune gap .0054.

The prediction failed low: +.0074 is under the predicted +.010, while it still clears the .005 floor. Adam on embedding and head keeps a smaller gain at 1,000 steps. At 250 steps the same gap is +.0239, error .0020, on the same worlds, read directly from saved rows after this round.

## Round 9: a new rule for query and key, in its own attention environment

The environment is `attnenv.py`. Only the query and key rule changes. It has six tasks with random tokens: copy, select (find one cue among distractors), latest (ignore an earlier binding), majority, retain (forgetting during training) and switch (unlearning a habit). Scores are the area under the test loss curve plus the end loss, with one training run per setting.

How the tasks got here, so nobody reads them as chosen for a rule. No new rule had run when these changes were made; only Muon, frozen and random-direction query and key had.
- A token-class version of select was solved by frozen query and key, so it could not judge attention, and I replaced it.
- A key-value lookup stayed at chance with both trained and frozen query and key at 1,200 steps, so I removed it.
- The first forgetting task lost everything under every rule. A shorter second phase gives partial forgetting.
- The other modules' lrs mattered far more than I assumed: Muon at .005 on the MLP, value and output matrices learned tasks that .02 did not, so the first task designs were too easy once that was fixed. Scores now include the area under the loss curve, because end losses reach zero.

The new rule comes from one fact. Query-key normalization divides each query and key vector by its size, so scaling all rows of one head leaves that head's output unchanged. Its gradient therefore has no part along those rows, and any fixed-size step makes the rows grow. Under Muon the query and key matrices grew from a size of 8 to about 27 in 600 steps, so the same lr turns them by much less at the end of training than at the start.

Rules:
- `muon`, `muonsplit` (published as Per-Head Muon, Muon Split and MuonSplit), `adam`, `lion`, `normsgd` and `adamtangent` (AdamP's idea, 2006.08217) are baselines.
- `muontangent` is per-head Muon with the part along each head's rows removed.
- `muonsphere` is `muontangent` that also puts each head's rows back at their first size, so the turning angle per step stays constant. My three searches found no source applying this to query-key normalization or to Muon. That does not prove nobody did.

Predictions, all gaps positive for the first rule, floor .02, worlds 4 to 8:
1. `muonsphere` beats Muon on switch.new.area by at least .05, because its rows don't grow and it can still turn late in training.
2. `muonsphere` loses to Muon on retain.old.nll, because the same property lets it forget faster. If it wins both, this story is wrong.
3. `muonsphere` beats Muon on copy.area, select.area and latest.area by at least .02. Lower confidence, about even odds.
4. `muontangent` ties `muonsplit` on every score, because removing the part along the rows is small at these lrs.
5. `muonsplit` beats Muon on area scores by at least .02, matching the early boost reported in 2605.08933.
6. All rules tie on majority.area.
7. At 4x the steps the area gaps shrink by at least half, as the query and key gain did in rounds 5 and 6.

Drop the idea if `muonsphere` passes no main score against both baselines. The win rule is in `attnbeat.py`.

### Round 9 controls, before any new rule ran

Muon against frozen query and key, worlds 1 to 3, positive means Muon is better: select.area +.636 (error .061), latest.area +.771 (.135), retain.learn.area +1.075 (.175), retain.old.nll +.650 (.312), switch.new.area +1.219 (.507), copy.area +.404 (.275, tie, because world 3 alone gives +.955), majority.area +.033 (.015). A random direction on query and key gives the same picture. Two identical runs on select changed nothing. So select, latest, retain and switch can judge query and key, copy can with care, and majority can't.

Other modules' lrs are Adam .01 on embedding, head and norms, and Muon .005 on value, output and MLP matrices. Muon .0025 scored better on tune rows for retain and switch and worse on latest, but it brings retain.old.nll to .021, where nothing can be measured, so I kept .005. Every arm shares this choice.

### Round 9 results at 600 steps, worlds 4 to 8

Mean test scores, lower is better, errors over worlds in brackets. Each rule has one lr from the same grid on world 0.

| Rule | copy.area | select.area | latest.area | retain.old.nll | switch.new.area |
|---|---|---|---|---|---|
| Muon | 2.179 (.114) | .735 (.007) | .943 (.084) | .381 (.113) | 2.431 (.257) |
| Per-head Muon | 1.668 (.288) | .732 (.009) | .758 (.054) | .590 (.259) | 2.107 (.414) |
| Adam | 1.544 (.318) | .742 (.025) | .906 (.068) | .536 (.270) | 1.740 (.364) |
| Lion | 1.898 (.266) | .728 (.007) | .890 (.061) | .173 (.066) | 1.029 (.088) |
| NormSGD | 2.316 (.028) | .805 (.065) | 1.015 (.068) | .474 (.138) | 1.740 (.291) |
| Adam, tangent per head | 1.664 (.237) | .740 (.010) | .881 (.082) | 4.203 (.142) | .612 (.067) |
| Muon, tangent per head | 1.713 (.288) | .708 (.021) | .636 (.026) | 5.252 (.405) | .771 (.050) |
| Muon on a sphere per head | 1.784 (.352) | .750 (.003) | .675 (.021) | 5.970 (.394) | .789 (.054) |

Query and key size at the end of select in world 4, from a start of 8: Muon 32, per-head Muon 163, tangent 10, sphere 8.

Win rule: `muontangent` wins latest and switch and is reversed on retain. `muonsphere` wins switch and is reversed on retain, and it loses the majority task to Muon by .043. Neither wins.

Predictions:
1. Held. `muonsphere` beats Muon on switch by +1.642, and per-head Muon by +1.318.
2. Held. `muonsphere` loses retain to Muon by 5.588, so the plasticity costs memory.
3. Mostly failed. Copy tie (+.395, error .264), select tie, latest +.268 holds.
4. Failed. `muontangent` beats per-head Muon on latest (+.122) and switch (+1.336) and loses retain by 4.662. Removing the part along the rows is not small.
5. Partly held. `muonsplit` beats Muon on copy (+.511) and latest (+.184), and ties on select and switch.
6. Failed for one rule. `muonsphere` loses majority to Muon by .043, error .017. The rest tie.

The tangent idea on Adam gives the same two changes (retain 4.203, switch .612), so the effect comes from stopping growth and not from Muon. Lion gives the best balance so far: it beats Muon on switch by 1.4 and does not lose retain, and it has no tangent step.

## Round 10: spend a fixed step on directions that got little update

Stopping growth keeps every direction plastic, so old cues get overwritten. A size-based brake can't tell old directions from new ones. A brake per direction can.

`muonhistory` keeps a sum G of normalized gradient products gᵀg for each query and key matrix, in input space. Its step is the Nesterov direction times G^(-1/2), tangent per head, scaled to Muon's Frobenius size, with each head put back at its first size. Muon uses only the current product, so at the first step this is Muon's orthogonalization. Later, directions that received much update get small shares of the step. New cues use new input directions and should get most of the step, while old cues stay in place. When only old directions matter, as in switch, the fixed size means the step is still full.

Accumulating gᵀg comes from AdaGrad and Shampoo (1802.09568). Using only the input-side sum, on a per-head sphere and for query and key, is the part my searches didn't find. That doesn't prove nobody did it.

Predictions, worlds 4 to 8, tuned on world 0 with the same grid:
1. `muonhistory` has retain.old.nll at most half of `muonsphere`'s 5.970.
2. Its switch.new.area is within .3 of `muonsphere`'s .789, and it beats Muon there.
3. Its latest.area beats Muon by at least .1.
4. It does not win overall, because retain.old.nll stays above Muon's .381. About 60% for this one.

Drop the idea if prediction 1 fails.

### Round 10 results at 600 steps, worlds 4 to 8

`muonhistory` at its pick, lr .02: copy.area 1.663 (.262), select.area .730 (.011), latest.area .796 (.042), retain.old.nll 4.658 (.266), switch.new.area .620 (.145), majority.area .951 (.025).

Against Muon: copy +.516 holds, select tie, latest +.147 tie, retain -4.276 reversed, switch +1.811 holds, majority -.040 reversed. Against per-head Muon: only switch wins (+1.487), and retain is reversed by 4.068. No win.

Predictions: 1 failed (4.658 is above half of 5.970, the drop condition), 2 held (.620), 3 held in size and not in the verdict (+.147, error .090), 4 held. The brake per direction does not stop forgetting at this lr, so I'm dropping it as a fix for forgetting.

## Round 11: is the forgetting only the lr pick?

Tune rows for `muonhistory` at lr .01 had retain.old.nll .351 and switch .907, against 4.336 and .209 at lr .02. So the mean-rank pick may be buying plasticity with memory, and the comparison above may only show that. This round asks for memory first.

Every rule, baselines included, picks again on world 0, from the wider grid, among lrs with retain.old.nll under 1. Verdicts use worlds 9 to 13, which no run has read. The win rule in `attnbeat.py` stays the same, with these picks.

Predictions, written before any of these picks ran:
1. No tangent or sphere rule wins overall, because the cap will force lrs where copy falls back to about 2.3.
2. `muonhistory` beats Muon on switch.new.area by at least .3, and is not reversed on retain.old.nll. This is the case for a brake that helps, so it is the result I most want to see, and I put it at 40%.
3. Lion beats Muon on switch.new.area by at least .3 with no reversal on any main score.
4. If any rule wins at 600 steps, its gain on area scores shrinks by at least half at 2,400 steps.

### Round 11 results, memory-first picks

600 steps, worlds 9 to 13, none read before. Mean scores, lower is better, errors over worlds in brackets.

| Rule | lr | copy.area | select.area | latest.area | retain.old.nll | switch.new.area | majority.area |
|---|---|---|---|---|---|---|---|
| Muon | .02 | 2.007 (.196) | .748 (.010) | .955 (.052) | .493 (.230) | 2.153 (.474) | .890 (.022) |
| Per-head Muon | .08 | 1.546 (.220) | .748 (.011) | .823 (.029) | .231 (.117) | 1.980 (.295) | .901 (.009) |
| Adam | .048 | 1.378 (.239) | .767 (.025) | .950 (.027) | .265 (.096) | 1.679 (.506) | .926 (.016) |
| Lion | .002 | 2.105 (.156) | .755 (.024) | .846 (.046) | .341 (.156) | 1.548 (.416) | .916 (.022) |
| NormSGD | .04 | 1.561 (.304) | .874 (.066) | .981 (.023) | .595 (.263) | 1.191 (.468) | .954 (.013) |
| Adam, tangent per head | .012 | 1.451 (.192) | .775 (.037) | .742 (.043) | 3.384 (.414) | 1.047 (.503) | .925 (.019) |
| Muon, tangent per head | .005 | 2.218 (.094) | .751 (.014) | .945 (.062) | .261 (.076) | 1.335 (.418) | .894 (.017) |
| Muon on a sphere per head | .005 | 2.215 (.097) | .750 (.014) | .945 (.064) | .161 (.044) | 1.596 (.293) | .894 (.017) |
| `muonhistory` | .01 | 1.952 (.251) | .737 (.004) | .846 (.095) | 1.421 (.343) | .677 (.059) | .911 (.023) |

Adam's pick sits on the top of its grid. Against Muon: per-head Muon wins copy (+.462) and latest (+.132) and ties the rest. `muonhistory` wins switch (+1.476), ties retain (-.927, error .343) and loses majority by .022. Against per-head Muon, `muonhistory` wins switch (+1.303) and is reversed on copy (-.407) and retain (-1.190). No rule passes the win rule.

2,400 steps, worlds 9 to 11, same lrs, not retuned:

| Rule | copy.area | select.area | latest.area | retain.old.nll | switch.new.area |
|---|---|---|---|---|---|
| Muon | .182 (.127) | .002 (.001) | .007 (.005) | .256 (.069) | .051 (.050) |
| Per-head Muon | .030 (.021) | .002 (.001) | .003 (.002) | .380 (.206) | .701 (.346) |
| Muon on a sphere per head | .440 (.109) | .009 (.004) | .004 (.002) | 2.785 (1.323) | .002 (.000) |
| `muonhistory` | .213 (.126) | .004 (.003) | .010 (.003) | 6.388 (1.184) | .030 (.025) |

Every rule solves the tasks by then. `muonhistory`'s switch gain over Muon falls from +1.476 to +.021 (tie), and its forgetting stays at 6.388 against Muon's .256 (reversed).

Predictions:
1. Held. No tangent or sphere rule wins. Muon tangent and sphere fall to copy about 2.2 at lr .005. Adam tangent's cap on world 0 did not carry over: its retain.old.nll is 3.384 on fresh worlds.
2. Held against Muon only: switch +1.476 and retain not reversed. Per-head Muon reverses it on retain.
3. Failed. Lion's switch gain is +.605 and a tie, and majority is reversed by .026.
4. Not tested as written, because no rule won at 600 steps. The nearest case is `muonhistory`'s switch gain, which fell from +1.476 to +.021.

Result: I wrote four rules, and none beats Muon and per-head Muon together. `muonhistory` relearns fastest at 600 steps, at a cost in memory, and the gain is gone by 2,400 steps. The published per-head Muon is the safest choice on these tasks at 600 steps.

## Round 12: which part of Muon does the work?

Muon has three parts on query and key: momentum, orthogonalization, and the size growth that comes with its step. Each arm removes one part at Muon's lr .02, with no tuning, on worlds 4 to 8, so a gap to Muon is that part's effect. Worlds 4 to 8 have been read before, and these runs only explain, so no verdict on a new rule comes from them.

Arms: no momentum (beta 0); no orthogonalization (`normsgd`, the same step size); no growth (`muonfixed`, Muon's step, then each head back at its first size); a schedule in place of growth (`muonbrake`, `muonfixed` with lr / (1 + t / 630), because Muon's size at lr .02 doubles in about 630 steps); per-head orthogonalization (`muonsplit` at lr .02).

Predictions, written before any of these ran:
1. No growth relearns better than Muon by at least .3 on switch.new.area and forgets more by at least .3 on retain.old.nll. If it does neither, size growth is not the brake and I drop that explanation.
2. The schedule lands within .2 of Muon on retain.old.nll and within .3 on switch.new.area. If so, the brake is only a falling angle, and a rule can set it on purpose.
3. No momentum loses at least .3 on copy.area or switch.new.area.
4. No orthogonalization loses at least .03 on select.area.
5. Per-head orthogonalization at the same lr beats Muon on copy.area by at least .2.

### Round 12 results

Lr .02, worlds 4 to 8, lower is better, one training run per setting in each world.

| Arm | copy.area | select.area | latest.area | retain.old.nll | switch.new.area | majority.area | size at end |
|---|---|---|---|---|---|---|---|
| Muon | 2.179 | .735 | .943 | .381 | 2.431 | .911 | 27.1 |
| No momentum | 2.316 | .741 | 1.070 | .793 | 1.251 | .897 | 9.3 |
| No orthogonalization | 2.121 | .912 | .908 | .695 | 1.355 | .957 | 36.1 |
| No growth | 1.697 | .734 | .785 | 4.487 | 1.216 | .927 | 8.0 |
| Schedule for growth | 1.961 | .727 | .811 | 1.989 | 1.520 | .922 | 8.0 |
| Per-head orthogonalization | 1.926 | .726 | .754 | .476 | 1.310 | .907 | 32.0 |

Paired gaps against Muon, positive means the arm is better. No growth: copy +.482 holds, latest +.158 holds, switch +1.215 holds, retain -4.105 reversed. Schedule: latest +.131 holds, retain -1.607 reversed, switch +.912 tie. No momentum: switch +1.180 holds, retain -.412 tie. No orthogonalization: switch +1.076 holds, majority -.045 reversed, select -.177 tie. Per-head: latest +.189 holds, switch +1.122 holds, retain -.094 tie.

Predictions:
1. Held, far past the line: switch +1.215 and retain -4.105. Size growth is the memory.
2. Failed. The schedule is 1.607 from Muon on retain and .912 on switch. Its tau came from one row of 15.6, and the mean end size is 27.1, so it may just be too weak.
3. Failed. No momentum relearns better (+1.180), and copy's loss of .137 is under the line.
4. Mean loss .177 passes the line, but the verdict is a tie because of spread across worlds.
5. Mean gain .254 passes the line, but the verdict is a tie.

New fact I did not predict: without momentum the size ends at 9.3 instead of 27.1. A gradient has no part along a head's rows, so the growth seems to come from momentum, which keeps pushing along old directions the rows have already moved in.

## Round 13: change Muon using what round 12 found

`attnsignal.py`, on world 0 tune rows: the gradient's part along a head's rows is 0.000 in every window, and Muon's step has a part along them of up to .39, so all growth comes from the step. My idea of a release signal, the cosine between the gradient and the head's total move, stays near 0 and does not see a task change, so I dropped it. The cosine between gradient and momentum falls to about .05 once a task is learned and jumps to about .25 at a task change, but it jumps the same in retain and switch, so it can't tell which of the two is happening.

So size growth is a hidden lr schedule, and it starts during first learning, which costs copy and latest. Two changes follow:
- `muonfade`: per-head, fixed size, lr / (1 + t / 150). This sets the schedule on purpose, stronger than the tau 630 that failed in round 12.
- `muongate`: per-head, and a head may grow only when the running cosine between gradient and momentum is under .2. The brake then builds after learning. Tau .2 comes from the world 0 table above.

Both pick lr on world 0 with the memory-first cap of round 11, against the same picks for Muon and per-head Muon. Verdicts use worlds 14 to 18, which no run has read. The win rule is unchanged. Rules written for attention so far: 6, plus the 2 used only to explain.

Predictions, written before these ran:
1. `muonbrake` with tau 150, whole matrix, at lr .02 on worlds 4 to 8: retain.old.nll within .3 of Muon, and switch.new.area within .5. If so, the brake is only a falling angle.
2. `muonfade` beats per-head Muon on copy.area by at least .2, with retain.old.nll not reversed.
3. `muongate` beats per-head Muon on copy.area by at least .2 and forgets more than it by at least .2.
4. Neither passes the full win rule. I put a win at 15%, because 3 of 5 scores must clear both baselines and select barely moves.
5. If either wins at 600 steps, its gains on area scores shrink by at least half at 2,400 steps.

### Round 13 results

`muonbrake` with tau 150 at lr .02, worlds 4 to 8, size held at 8.0: copy 2.168, select .741, latest .929, retain .294, switch 2.322, majority .914, against Muon's 2.179, .735, .943, .381, 2.431, .911. Every gap is a tie and none passes .109. So Muon's size growth works as a hidden falling lr, and a schedule can replace it.

Picks: `muonfade` lr .02, `muongate` lr .04. Worlds 14 to 18, none read before, 600 steps. Errors over worlds in brackets.

| Rule | lr | copy.area | select.area | latest.area | retain.old.nll | switch.new.area | majority.area | size at end |
|---|---|---|---|---|---|---|---|---|
| Muon | .02 | 1.568 (.184) | .737 (.006) | 1.004 (.086) | .651 (.172) | 1.808 (.368) | .904 (.014) | 26.7 |
| Per-head Muon | .08 | 2.032 (.207) | .820 (.110) | .876 (.087) | .478 (.198) | 1.577 (.252) | .916 (.016) | 130.6 |
| Per-head Muon, not the pick | .02 | 1.588 | .727 | .892 | .716 | 1.900 | .896 | 31.9 |
| `muonfade` | .02 | 1.441 (.239) | .728 (.013) | .801 (.059) | .312 (.097) | 2.456 (.314) | .902 (.007) | 8.0 |
| `muongate` | .04 | 1.695 (.205) | .733 (.038) | .778 (.047) | .388 (.198) | 1.402 (.191) | .904 (.007) | 58.7 |

Paired gaps, positive means the new rule is better:

| Comparison | copy | select | latest | retain | switch | majority |
|---|---|---|---|---|---|---|
| `muonfade` vs Muon | +.126 tie | +.009 tie | +.203 holds | +.339 holds | -.648 tie | +.003 tie |
| `muonfade` vs per-head .08 | +.591 holds | +.092 tie | +.075 tie | +.167 tie | -.879 reversed | +.014 tie |
| `muongate` vs Muon | -.127 tie | +.004 tie | +.226 holds | +.263 tie | +.406 tie | +.000 tie |
| `muongate` vs per-head .08 | +.337 holds | +.087 tie | +.098 tie | +.090 tie | +.175 tie | +.012 tie |
| `muongate` vs per-head .02 | -.107 tie | -.007 tie | +.114 holds | +.328 tie | +.499 tie | -.008 tie |

Neither passes the win rule: no score clears both baselines. `muongate` is the first rule here that neither baseline reverses on any score. Its mean is better than Muon's on 4 of 5 main scores and worse on copy.

2,400 steps, worlds 14 to 16, same lrs, not retuned. `muonfade` did not run here, because it was already reversed at 600 steps and the Mac was busy.

| Rule | copy.area | select.area | latest.area | retain.old.nll | switch.new.area | majority.area |
|---|---|---|---|---|---|---|
| Muon | .018 (.005) | .005 (.004) | .006 (.002) | .669 (.362) | .244 (.243) | .234 (.006) |
| Per-head Muon .08 | .294 (.139) | .002 (.001) | .007 (.001) | .405 (.124) | 1.120 (.696) | .255 (.014) |
| `muongate` | .218 (.113) | .005 (.002) | .008 (.006) | .263 (.135) | .003 (.001) | .255 (.007) |

`muongate` ties both baselines on every score at 2,400 steps. Its means on memory and relearning are better, and Muon's mean on copy is better, with only 3 worlds.

Predictions:
1. Held. All gaps are within .109, inside both lines.
2. Held against the per-head pick .08. At lr .02 per-head Muon's copy is 1.588, so the gain depends on that pick.
3. Half held. Copy +.337 passes, but `muongate` forgets less than per-head Muon in the mean, not more.
4. Held. Neither passes.
5. Not tested as written, because neither won. `muongate`'s one clear gain over Muon, latest +.226, is -.001 at 2,400 steps, because every rule solves latest by then.

Rules written for attention so far: 6, plus 2 used only to explain. With this many tries and world-to-world spread this large, Muon's copy moved from 2.007 on worlds 9 to 13 to 1.568 here, so these gaps need more worlds before anyone relies on them.

## Round 14: is `muongate`'s gain real?

Worlds 19 to 33, 15 worlds no run has read, 600 steps, same picks as round 13 with nothing retuned: Muon .02, per-head Muon .08, `muongate` .04. Per-head Muon at lr .02 runs beside them as a comparison outside the picks. The win rule is unchanged. Because 6 rules were tried over 5 scores, I also report which verdicts clear 3 errors, not only 2.

`muongate` was chosen because it looked best on worlds 14 to 18, so its gains there are likely too large.

Predictions, written before these ran:
1. `muongate` beats Muon on latest.area by at least .1, and the verdict holds.
2. `muongate` beats Muon on switch.new.area by at least .2, and the verdict holds. I put this at 40%.
3. `muongate` is not reversed by Muon on retain.old.nll.
4. Muon reverses `muongate` on copy.area by at least .1. I put this at 50%.
5. `muongate` passes the score part of the win rule, 3 of 5 scores against both baselines. I put this at 10%.
If `muongate` holds on no score against Muon here, I drop it.

### Round 14 results

Worlds 19 to 33, 600 steps, nothing retuned. Errors over worlds in brackets.

| Rule | lr | copy.area | select.area | latest.area | retain.old.nll | switch.new.area | majority.area |
|---|---|---|---|---|---|---|---|
| Muon | .02 | 1.905 (.103) | .731 (.003) | .912 (.041) | .325 (.063) | 1.984 (.213) | .891 (.009) |
| Per-head Muon | .08 | 1.833 (.121) | .727 (.003) | .746 (.029) | .224 (.072) | 2.055 (.196) | .884 (.009) |
| Per-head Muon, not the pick | .02 | 1.838 | .721 | .806 | .225 | 2.001 | .885 |
| `muongate` | .04 | 1.664 (.128) | .718 (.009) | .695 (.025) | .344 (.068) | 1.813 (.174) | .894 (.009) |

Paired gaps for `muongate`, positive means it is better, with the number of worlds of 15 where it was ahead:

| Against | copy | select | latest | retain | switch | majority |
|---|---|---|---|---|---|---|
| Muon | +.242 holds, 12 | +.013 tie, 10 | +.218 holds, 15 | -.019 tie, 8 | +.171 tie, 8 | -.003 tie, 5 |
| Per-head .08 | +.169 tie, 13 | +.009 tie, 13 | +.051 tie, 11 | -.120 tie, 3 | +.242 tie, 9 | -.010 tie, 4 |
| Per-head .02 | +.175 tie, 7 | +.003 tie, 11 | +.111 holds, 13 | -.119 tie, 4 | +.189 tie, 9 | -.009 tie, 4 |

Only latest against Muon and latest against per-head .02 clear 3 errors. On a score outside the main five, learning the first cues (retain.learn.area), `muongate` beats Muon by +.337 and per-head .08 by +.158, both holding.

Predictions:
1. Held: latest +.218, ahead in all 15 worlds.
2. Failed: switch +.171 is a tie, ahead in 8 of 15. The +.406 on worlds 14 to 18 was too large.
3. Held: retain -.019 is a tie. The +.263 on worlds 14 to 18 was also too large.
4. Failed the other way: `muongate` beats Muon on copy by +.242. Five worlds had the sign wrong.
5. The 10% case did not happen: `muongate` clears no main score against both baselines.
It holds on two scores against Muon, so I keep it.

Result: `muongate` learns new things faster than Muon, on copy, latest and first cues, and keeps memory level with Muon. It does not relearn better than Muon on this evidence. It does not pass the win rule. Against per-head Muon it clears no main score and lags on memory in 12 of 15 worlds at the pick, though that verdict is a tie. These fit its design, which removes the brake during first learning and does nothing special at a task change. The 2,400-step check is still the 3 worlds of round 13, where it ties, with Muon's mean ahead on copy.
