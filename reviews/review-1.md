# Review 1, November 3, 2026

A fresh reviewer read the code against `brief.md`. It found no leak: searches read only tune rows, no row overlap, equal tokens and gradients, identical repeats, and the Muon port checks out. It found 17 problems, and these are what was done. The first five and 8, 10, 12 blocked the result.

| # | Finding | What was done |
|---|---|---|
| 1 | At 250 steps with 4,000 tokens, embedding and head hold 87% of the weights, and hidden training was worth .089 of a 2.75 drop, so the test mostly measured embeddings. | New tiers: hidden matrices hold 37% at 5s, 51% at 1m, 54% at 15m. A `frozen-hidden` control, and a tier counts only if hidden training is worth at least 10x the floor. |
| 2 | `beat.py` took typed specs. | It reads base and design from `choices.jsonl`. |
| 3 | Unequal lr groups gave a false "hybrid beats Muon" of +.0070, which vanished with equal groups. | Every arm has one lr per module. |
| 4 | The rule search threw away tuned lrs and started with no best score. | It starts from the best recipe's score and anchors each rule on its tuned lrs. |
| 5 | `leftpolar` stalled on embedding and head. | Removed from the menu, because it equals polar on tall matrices. |
| 6 | The control's lr lattice was coarser than the search's. | Same shrinking factors, and the control has had more settings than the search. Both counts are saved. |
| 7 | A uniform recipe could count as a per-module win. | The base is the best of all tuned recipes, and a win needs at least 2 rules on hidden modules. |
| 8 | 800 rows can't detect .005: it needs 1,600 to 4,300. | 3,200 rows per split. |
| 9 | Jitter understates noise about 7x against worlds. | Verdicts use spread across worlds. |
| 10 | K1 was scored in the searched world and had no code. | `k1.py` scores fresh worlds 10 to 14. |
| 11 | `audit.py` admitted scores from the same rows as the main score. | Same-row and train-set scores are out, with a 30-item minimum, and it writes `judges.json`, which `beat.py` needs. |
| 12 | `beat.py` loopholes: mixed filters, no multiple-test correction, search rows holding report scores, speed from parallel runs, no one-shot final. | All closed. `final.py` refuses a second read. `debug/path.py` shows the rule refusing a self-tie and passing a planted win. |
| 13 | Manifold rules are outside the menu because of their speed. | `search.py rule --manifold` is a separate result, not yet run. |
| 14, 15 | Worlds overlapped, and val shards followed one order. | Worlds start `max(4M, tokens)` apart, and held rows come from a fixed permutation of slots, disjoint per world. |
| 16 | A best lr could sit at a grid edge. | The first grid moves up to 3 times. |
| 17 | Some scores mean nothing at this size. | The audit's item minimum and its rules handle them. Fact scores don't exist at 5s (V 1000 holds no fact names) and have too few items at 1m. |
| minor | `debug/port_check.py` takes coefficients from `rules.py`. | It now holds the table pasted from the authors' `msign.py`. |

The reviewer also noted that "nobody has searched this" sits beside seven unread papers, including optimizer search 1709.07417, so the claim stays out of any write-up until they are read.
