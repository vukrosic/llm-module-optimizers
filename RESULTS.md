# Results

Everything is web-text loss on report rows, lower is better. Gaps are paired per row, and positive means the first arm wins. Each setting ran once per world. A world changes the text, the facts, the held-out rows and the starting weights.

**Verdict rule:** a claim holds if the mean gap over worlds exceeds max(.005, 2 errors), where the error is the larger of the spread across worlds over root 5 and the mean row error.

## Tiers

| Tier | Width | Layers | Heads | Vocabulary | Batch | Length | Steps | Hidden matrices hold | State |
|---|---|---|---|---|---|---|---|---|---|
| 5s | 48 | 2 | 4 | 1,000 | 16 | 48 | 250 | 37% | finished |
| 5sx4 | 48 | 2 | 4 | 1,000 | 16 | 48 | 1,000 | 37% | tuned most recipes, frozen controls and manifold rules not tuned |
| 1m | 64 | 4 | 4 | 1,500 | 16 | 48 | 1,000 | 51% | tuning stopped, no result |

Each split has 3,200 rows. One lr per module is tuned on tune rows of world 0, with no report row read. Baselines report on worlds 0 to 4. The kill test reports on worlds 10 to 14, which no search reads.

## Tuned recipes

| Arm at 250 steps | Tune | Report | Settings |
|---|---|---|---|
| **muon** | **4.4018** | **4.3990** | 62 |
| all-adam (standard baseline) | 4.4138 | 4.4131 | 64 |
| frozen-mlp (control) | 4.4155 | 4.4116 | 46 |
| sym (2605.18106) | 4.4239 | 4.4219 | 60 |
| sym-rownorm | 4.4239 | 4.4209 | 60 |
| sym-hybrid-all | 4.4251 | 4.4237 | 60 |
| hetero (2606.13276) | 4.4330 | 4.4306 | 62 |
| stiefel (2606.13276) | 4.4332 | 4.4338 | 62 |
| lion-all | 4.4379 | 4.4398 | 63 |
| rownorm-all | 4.4484 | 4.4468 | 61 |
| frozen-hidden (control) | 4.5056 | 4.5063 | 31 |
| normsgd-all | 4.5125 | 4.5138 | 60 |
| blind (control) | 7.2809 | 7.2847 | 63 |

| Arm at 1,000 steps | Tune | Report | Settings |
|---|---|---|---|
| **sym** | **4.1450** | **4.1423** | 57 |
| muon | 4.1488 | 4.1436 | 57 |
| sym-rownorm | 4.1519 | 4.1494 | 57 |
| sym-hybrid-all | 4.1577 | 4.1563 | 57 |
| all-adam | 4.1636 | 4.1580 | 58 |
| rownorm-all | 4.1685 | 4.1648 | 58 |
| lion-all | 4.1844 | 4.1813 | 57 |
| normsgd-all | 4.1979 | 4.1931 | 59 |

## Do published claims reproduce? Worlds 0 to 4

| Claim at 250 steps | Source | Mean gap | Error | Verdict | Per world |
|---|---|---|---|---|---|
| Muon recipe beats the standard baseline | Muon's reports | +.0162 | .0015 | holds | +.0141 +.0132 +.0145 +.0185 +.0207 |
| RowNorm on embedding and head beats Adam there | 2605.18106 | −.0239 | .0017 | reversed | −.0218 −.0212 −.0267 −.0282 −.0214 |
| Hybrid MLP beats Muon MLP | 2605.18106 | −.0008 | .0014 | tie | −.0011 −.0034 +.0046 −.0023 −.0020 |
| All-Stiefel beats unconstrained | 2606.13276 | −.0338 | .0030 | reversed | −.0348 −.0249 −.0297 −.0384 −.0412 |
| Stiefel attention + DGram MLP beats All-Stiefel | 2606.13276 | −.0003 | .0023 | tie | +.0032 −.0041 −.0059 +.0064 −.0010 |

| Claim at 1,000 steps | Source | Mean gap | Error | Verdict | Per world |
|---|---|---|---|---|---|
| Muon recipe beats the standard baseline | Muon's reports | +.0107 | .0022 | holds | +.0144 +.0134 +.0137 +.0091 +.0027 |
| RowNorm on embedding and head beats Adam there | 2605.18106 | −.0007 | .0023 | tie | −.0058 −.0052 −.0010 +.0062 +.0022 |
| Hybrid MLP beats Muon MLP | 2605.18106 | +.0045 | .0020 | tie | +.0072 +.0074 −.0034 +.0046 +.0064 |
| sym beats muon | 2605.18106 | +.0037 | .0027 | tie | +.0013 +.0022 −.0043 +.0108 +.0086 |
| sym beats the standard baseline | 2605.18106 | +.0144 | .0018 | holds | +.0158 +.0156 +.0094 +.0198 +.0112 |

2606.13276's result is at 50,000 steps on GPT-2 small, so its reversal here is a statement about 250 steps only.

## Kill test: rule search against the best recipe, fresh worlds 10 to 14

| Comparison | Settings | Mean gap | Error | Per world |
|---|---|---|---|---|
| 250 steps: rule search vs muon | 218 vs 62 | +.0073 | .0031 | +.0180 +.0072 +.0067 +.0056 −.0010 |
| 250 steps: lr-only search vs muon | 155 vs 62 | +.0033 | .0019 | +.0100 +.0024 +.0043 +.0014 −.0017 |
| 250 steps: rule search vs lr-only search | 218 vs 155 | +.0040 | .0012 | +.0080 +.0047 +.0024 +.0041 +.0007 |
| 1,000 steps: rule search vs sym | 219 vs 57 | +.0003 | .0008 | −.0005 +.0012 +.0001 +.0019 −.0013 |

The 250-step rule search clears the kill test's bar of max(.005, 2 errors) at that tier and fails `beat.py`'s rule that the sign must hold in all 5 worlds. The kill test also required the 1m tier, which did not run.

Designs the searches chose (`qk` is attention's query and key matrices, `vo` its value and output, `gu` and `down` the SwiGLU gate and output):

```text
rule search, 250 steps   emb=adam:.3 head=adam:.048 norm=adam:.019 qk=lion:.0031 vo=muon:.025 gu=muon:.0079 down=muon:.05
lr-only, 250 steps       emb=adam:.3 head=adam:.048 norm=adam:.019 qk=muon:.017 vo=muon:.022 gu=muon:.0089 down=muon:.018
rule search, 1,000 steps emb=rownorm:.75 head=rownorm:.062 norm=adam:.015 qk=muon:.032 vo=muon:.041 gu=hybrid:.008 down=hybrid:.02
```

## Which rule does each module want? Worlds 15 to 23

Round 4 to round 8 of [EXPERIMENTS.md](EXPERIMENTS.md), with predictions. Settings come from tune rows of world 0. Mix means the lr-only design with Lion on query and key.

### One swap at a time, 250 steps, added tune loss against lr-only (4.3974)

| Module | Adam | Muon | RowNorm | Hybrid | Lion | NormSGD |
|---|---|---|---|---|---|---|
| emb | 0 | +.043 | +.018 | +.029 | +.022 | |
| head | 0 | +.0097 | | +.0172 | | +.0126 |
| qk | -.0031 | 0 | -.0028 | | -.0036 | -.0024 |
| vo | +.0346 | 0 | | +.0036 | +.0301 | +.0344 |
| gu | +.0043 | 0 | | +.0035 | +.0029 | |
| down | | 0 | | +.0027 | +.0040 | |

Norm scales: Adam 0, norm1d +.0102, heavy ball +.0122, Lion +.0198. Blank cells were outside the 4 closest rules for that module. Across 60 random mixes the sum of swaps predicted the mix's loss with correlation .881, and mixes did better than the sum by .0137 on average.

### Fresh worlds

| Comparison | Length | Worlds | Mean gap | Error | Verdict | Per world |
|---|---|---|---|---|---|---|
| Muon's recipe vs Muon on every matrix | 250 | 15-19 | +.0239 | .0020 | holds | +.0186 +.0230 +.0253 +.0225 +.0300 |
| Muon's recipe vs Muon on every matrix | 1,000 | 15-19 | +.0074 | .0017 | holds | +.0092 +.0110 +.0036 +.0033 +.0101 |
| Mix vs lr-only | 250 | 15-19 | +.0083 | .0012 | holds | +.0101 +.0088 +.0068 +.0069 +.0086 |
| Mix vs lr-only | 250 | 10-14 | +.0067 | .0011 | holds | +.0082 +.0052 +.0089 +.0045 +.0066 |
| Mix vs lr-only | 250 | 21-23 | +.0090 | .0013 | holds | +.0115 +.0083 +.0071 |
| Mix vs Muon's recipe | 250 | 15-19 | +.0106 | .0016 | holds | +.0118 +.0122 +.0110 +.0137 +.0043 |
| Mix vs Muon on every matrix | 250 | 15-19 | +.0345 | .0021 | holds | +.0304 +.0352 +.0364 +.0362 +.0342 |
| Mix vs Lion on every matrix | 250 | 15-19 | +.0536 | .0023 | holds | +.0570 +.0593 +.0537 +.0522 +.0459 |
| Mix vs RowNorm on every matrix | 250 | 15-19 | +.0609 | .0025 | holds | +.0619 +.0676 +.0639 +.0579 +.0531 |
| Mix vs NormSGD on every matrix | 250 | 15-19 | +.1061 | .0031 | holds | +.1120 +.1109 +.0946 +.1063 +.1066 |
| Mix vs the standard baseline | 250 | 15-19 | +.0271 | .0021 | holds | +.0295 +.0318 +.0229 +.0303 +.0213 |
| Adam vs Muon on q,k | 250 | 15-19 | +.0077 | .0013 | holds | +.0079 +.0068 +.0057 +.0068 +.0111 |
| RowNorm vs Muon on q,k | 250 | 15-19 | +.0063 | .0013 | holds | +.0073 +.0085 +.0049 +.0024 +.0086 |
| NormSGD vs Muon on q,k | 250 | 15-19 | +.0056 | .0012 | holds | +.0063 +.0054 +.0018 +.0051 +.0094 |
| Lion vs Adam on q,k | 250 | 15-19 | +.0006 | .0012 | tie | +.0022 +.0020 +.0011 +.0001 -.0025 |
| Lion vs Muon on q,k, no QK-norm | 250 | 15-19 | +.0039 | .0015 | tie | +.0087 +.0038 +.0045 -.0010 +.0033 |
| Lion vs Muon on q,k | 1,000 | 15-19 | +.0001 | .0018 | tie | -.0038 -.0041 +.0012 +.0025 +.0048 |
| Adam vs Muon on q,k | 1,000 | 15-19 | +.0010 | .0018 | tie | -.0047 -.0007 +.0014 +.0037 +.0053 |
| Mix vs p .1 on all hidden | 250 | 15-19 | +.0114 | .0019 | holds | +.0142 +.0130 +.0098 +.0153 +.0047 |
| Mix vs p .1 on all hidden | 250 | 21-23 | +.0104 | .0013 | holds | +.0101 +.0099 +.0114 |
| p .75 on q,k vs lr-only | 250 | 15-19 | +.0056 | .0011 | holds | +.0067 +.0046 +.0036 +.0047 +.0084 |

Added tune loss for `softmuon`, each at its best of 5 lrs: q,k +.0001, +.0008, -.0019, -.0029 at p .1, .25, .5, .75; value and output +.0081, +.0131, +.0253, +.0305; gate and up +.0018, +.0022, +.0031, +.0060; down +.0005, +.0016, +.0024, +.0032. With p 0 on q,k, the exact SVD added +.0026 against Muon's 5 Newton-Schulz steps.

The 1,000-step q,k comparisons start from Muon's recipe tuned at that length, because no lr-only design exists there. Designs tried in these rounds: 1 mix, 4 q,k rules and 1 single-p design. The verdict rule does not correct for that count.

## Controls: can the test tell?

| Control at 250 steps | Mean gap | Error | Reading |
|---|---|---|---|
| Training the MLP vs freezing it | +.0180 | .0017 | Training the MLP counts |
| Training hidden matrices vs freezing them | +.1030 | .0042 | 21 times the floor in world 0 (.107) |
| Gradients vs a random direction | +2.9954 | .0335 | The score responds to learning |

- **Repeat noise:** changing every lr by 0.1% moves the loss with sd .0004. This understates the noise between worlds about 7 times, so no verdict uses it.
- **Leak control:** a run allowed to train on report rows lowered report loss by .128 and tune loss by .027, so a leak shows up.
- **Double compute control:** a run given double compute read 384,000 tokens with 2 gradients per step, and its row records this, so it can't pass as equal compute.
- **Poison check:** `debug/poison.py` confirms the search rows hold only tune scores.
- **End-to-end check:** `debug/path.py` confirms a tie with itself is refused, a planted win of +.288 passes every rule except the state and hidden-size rules, and a second hidden read is refused.
- **Port check:** `debug/port_check.py` compares each rule against the authors' coefficients.

## What did not run

- The 1m tier: tuning was stopped, so no verdict exists above 1,000 steps or width 48.
- `beat.py`, the full win rule: it has no `judges.json`, because `audit.py` did not run on real tiers.
- `speed.py` on the designs, so the 1.25x compute limit is unchecked. The tables' time per step came from parallel runs and is not a claim.
- The hidden rows of world 20, which are still unread.
- Manifold rules in the search's menu.
- Any tier of 15 minutes or more.
- Fact scores at 250 steps, because they do not exist at that budget. Everything above is web-text loss.

## Commands

```bash
python3 check.py                                         # bug checks, 7 s each
python3 search.py tune muon sym all-adam --tier 5s       # one lr per module, tune rows only
python3 search.py rule --tier 5s                         # per-module rule and lr search
python3 search.py lronly --tier 5s                       # the same search with lrs only
python3 testtest.py --tier 5s                            # controls and noise
python3 k1.py --tier 5s --run --seconds 280              # fresh worlds; rerun to continue
python3 k1.py --tier 5s                                  # print the kill test
python3 k1.py --tier 5s --kind lronly                    # the lr-only comparison
python3 claims.py --tier 5s                              # published claims and controls
python3 chart.py                                         # draws gaps.png
```

`results.jsonl` holds every row, `choices.jsonl` every setting chosen, and `runs/` the per-item scores that paired errors come from. Results made before the test's last fixes are in `old/` and are not comparable.

## Commands for the module rounds

```bash
python3 mix.py --tier 5s
python3 qkrule.py --tier 5s
python3 softrule.py --tier 5s
python3 qklong.py --tier 5sx4
python3 search.py tune muon-all --tier 5sx4 --from-tier 5s
python3 embhead.py --tier 5sx4
```

## Attention environment

One part of the LLM at a time: only the query and key rule changes, and the other modules keep one fixed recipe (`attnenv.py`). The model has 2 layers, width 48 and 4 heads, on tiny token tasks. The rule, written before any new rule ran, is in `attnbeat.py`: against Muon and per-head Muon, win at least 3 of 5 main scores (copy, select, latest, remember, relearn) by .02 and 2 errors, with none reversed, majority tied, the same at 4x steps and 2x width, and at most 1.25x the time per step.

Scores are mean test loss. `.area` is the mean over 5 points of the training curve, and `retain.old.nll` is loss on old cues at the end. Each world gets one training run per setting.

### The tasks can tell a trained attention from a broken one

Muon against frozen query and key, worlds 1 to 3, positive means training helps:

| Score | Gap | Verdict |
|---|---|---|
| select.area | +.636 | holds |
| latest.area | +.771 | holds |
| retain.learn.area | +1.075 | holds |
| retain.old.nll | +.650 | holds |
| switch.new.area | +1.219 | holds |
| copy.area | +.404 | tie; one world had +.955 |
| majority.area | +.033 | small, so it is not a main score |

A random step direction gave similar gaps. Two runs of one setting were identical. Frozen query and key can solve some induction tasks (https://arxiv.org/html/2506.01115v2), which is why this control is required and why the first select task was replaced.

### All rules at 600 steps, lrs picked for memory first

Lrs were picked on world 0, among lrs with retain.old.nll under 1, from a grid of .125 to 16 times each default. Reported on worlds 9 to 13, which no run had read. Errors over worlds are in brackets.

| Rule | lr | copy.area | select.area | latest.area | retain.old.nll | switch.new.area | majority.area |
|---|---|---|---|---|---|---|---|
| Muon | .02 | 2.007 (.196) | .748 (.010) | .955 (.052) | .493 (.230) | 2.153 (.474) | .890 (.022) |
| Per-head Muon | .08 | 1.546 (.220) | .748 (.011) | .823 (.029) | .231 (.117) | 1.980 (.295) | .901 (.009) |
| Adam | .048 | 1.378 (.239) | .767 (.025) | .950 (.027) | .265 (.096) | 1.679 (.506) | .926 (.016) |
| Lion | .002 | 2.105 (.156) | .755 (.024) | .846 (.046) | .341 (.156) | 1.548 (.416) | .916 (.022) |
| NormSGD | .04 | 1.561 (.304) | .874 (.066) | .981 (.023) | .595 (.263) | 1.191 (.468) | .954 (.013) |
| Adam, tangent per head (new here) | .012 | 1.451 (.192) | .775 (.037) | .742 (.043) | 3.384 (.414) | 1.047 (.503) | .925 (.019) |
| Muon, tangent per head (new here) | .005 | 2.218 (.094) | .751 (.014) | .945 (.062) | .261 (.076) | 1.335 (.418) | .894 (.017) |
| Muon on a sphere per head (new here) | .005 | 2.215 (.097) | .750 (.014) | .945 (.064) | .161 (.044) | 1.596 (.293) | .894 (.017) |
| `muonhistory` (new here) | .01 | 1.952 (.251) | .737 (.004) | .846 (.095) | 1.421 (.343) | .677 (.059) | .911 (.023) |

Adam's pick sits on the top of its grid. Adam tangent's memory cap on world 0 did not carry over to fresh worlds.

Paired verdicts, positive means the first rule is better:

| Comparison | Holds | Tie | Reversed |
|---|---|---|---|
| Per-head Muon vs Muon | copy +.462, latest +.132 | select, retain, switch, majority | none |
| `muonhistory` vs Muon | switch +1.476 | retain -.927 | majority -.022 |
| `muonhistory` vs per-head Muon | switch +1.303 | | copy -.407, retain -1.190 |

No rule passes the win rule. With lrs picked by mean rank instead, on worlds 4 to 8, the new rules relearned and read the latest token better and forgot far more (retain.old.nll 4.658 to 5.970 against Muon's .381), and also did not pass. That table is round 9 and round 10 in [EXPERIMENTS.md](EXPERIMENTS.md).

### The same rules at 2,400 steps

Worlds 9 to 11, same lrs, not retuned.

| Rule | copy.area | select.area | latest.area | retain.old.nll | switch.new.area |
|---|---|---|---|---|---|
| Muon | .182 (.127) | .002 (.001) | .007 (.005) | .256 (.069) | .051 (.050) |
| Per-head Muon | .030 (.021) | .002 (.001) | .003 (.002) | .380 (.206) | .701 (.346) |
| Muon on a sphere per head | .440 (.109) | .009 (.004) | .004 (.002) | 2.785 (1.323) | .002 (.000) |
| `muonhistory` | .213 (.126) | .004 (.003) | .010 (.003) | 6.388 (1.184) | .030 (.025) |

Per-head Muon ties Muon on every score here. `muonhistory` ties Muon on relearning (+.021) and is reversed on memory (-6.132).

### Why size growth matters

Query-key normalization makes each head's block of rows scale-invariant, so the gradient has no part along it. Muon's orthogonalized step does have one, so the matrix grows. From a starting size of 8, 600 steps gave 32 for Muon, 163 for per-head Muon at lr .08, 10 for the tangent rule and 8 for the sphere rule. A larger matrix turns by a smaller angle per step. Our reading, not tested on its own, is that this growth acts as a brake: stopping it gives fast relearning and heavy forgetting.

### What did not run for attention

Width 96 did not run, and time per step was not judged, because no rule passed the scores first. Lrs were not retuned at 2,400 steps. Forgetting and relearning were not measured on real text, so these tasks say nothing yet about a large LLM. The new rules have no theory of when they should help, only the size argument above.

### Commands for attention

```bash
python3 check.py                                               # includes the new rules
python3 attncontrols.py                                        # frozen and random controls
ATTN_CHOICES=attn_choices_memory.jsonl python3 attntable.py --worlds 9,10,11,12,13 --tier a
ATTN_CHOICES=attn_choices_memory.jsonl python3 attntable.py --worlds 9,10,11 --tier ax4 --rules muon,muonsplit,muonsphere,muonhistory
python3 chart_attention.py                                     # draws attention.png
```

### Literature for attention

Per-head Muon is published and is a baseline here: Kimi K3 (arXiv:2607.24653, read secondhand), GLM-5's Muon Split, Group Muon (https://arxiv.org/abs/2605.08933), DeepSpeed's `per_head_muon` (https://www.deepspeed.ai/docs/config-json/), Megatron (https://github.com/NVIDIA/Megatron-LM/pull/6326), Moonlight (https://github.com/MoonshotAI/Moonlight/issues/42) and Motif 3 (https://arxiv.org/pdf/2608.09119).

Removing the part of a step that only changes size is also published: AdamP (https://arxiv.org/abs/2006.08217) and Rotational Equilibrium (https://arxiv.org/pdf/2305.17212). The running sum of gradient products comes from AdaGrad and Shampoo (arXiv:1802.09568). We found no source that applies size removal to query-key normalization or Muon, or that uses the input-side sum on a per-head sphere. These searches don't settle that, so treat the new rules as new combinations of known parts.

## Literature

| Paper | What it does | Relation here |
|---|---|---|
| 2605.18106 | RowNorm on embedding and head, Muon on attention, hybrid on SwiGLU | `sym` baselines |
| 2606.13276 | Stiefel or DGram manifold per module, 50,000 steps on GPT-2 small | `stiefel`, `hetero` baselines |
| Modular Manifolds (Bernstein, 2025) | Manifold constraint per module | The idea behind manifold arms |
| 2605.09176 (survey) | Lists whether assignment can be learned as open, §12.5, and cites no evidence | Motivates the question |
| 2608.28557 (survey) | Routing table by parameter class, by hand | Hand assignment |
| 2607.20486 OPTScientist | Program search over one rule for all 2D matrices | No per-module search or test split |
| 2605.04055, SGG (ACL 2025) | Modulate lrs per group, no choice of rule | The lr side of our question |
| 1709.07417 | Searches one update equation on a ConvNet | Abstract only |

**Not read in full:** 1709.07417, 2607.20548, 2605.26842, 2502.16982, 2603.20527, 2601.23000, 2506.01115, Celo (2501.12670), Celo2 (2602.19142). "We did not find" is the strongest statement these searches support.
