# Environment for researching a custom optimizer per LLM module

> [!IMPORTANT]
> We are using this environment to write a paper for ICML 2027. It was built in [AI Research School](https://www.skool.com/become-ai-researcher-2669), and you can use the code, results and ideas in your own research. The code is MIT licensed, and neither permission nor co-authorship is needed.
>
> **On a small LLM, searching for a different update rule per module added less than our .005 noise floor once every optimizer had its own tuned learning rate per module. Muon's recipe beat the standard baseline at both lengths we tested.**

| Paired gap in loss on held-out web text, over 5 worlds; positive means the first arm wins | 250 steps | 1,000 steps |
|---|---|---|
| **Muon's recipe vs the standard baseline** | **+.0162** (error .0015) | **+.0107** (error .0022) |
| Rule search vs the best published recipe | +.0073 (error .0031) | +.0003 (error .0008) |
| Rule search vs a search over learning rates only | +.0040 (error .0012) | not run |

![Gap per world for each comparison](gaps.png)

> [!WARNING]
> This is an early result from a model with 2 layers and width 48. The larger 1-minute tier was stopped, so the kill test's second required tier did not run, and the full win rule in `beat.py` has not been run. Larger models may differ.

> [!NOTE]
> An AI agent (Claude) wrote the code, ran the experiments and wrote this text. A second agent reviewed the code once ([review](reviews/review-1.md)). No person has reviewed the results.

---

## The core idea

An LLM learns by moving its weights a little after each batch of text. The formula that turns the gradient into that move is the **update rule**.

Most training uses one update rule for all weights. But weights do different jobs: an **embedding** is a table with one row per token, while an attention or MLP matrix transforms every token.

So recent papers give each kind of weight, each **module**, its own rule. **Muon** orthogonalizes the update of hidden matrices, and **RowNorm** normalizes each row of the update. These assignments are chosen by hand.

Each module can also have its own **learning rate (lr)**, the size of the move. A new rule can look better only because it got a better lr.

Our question: once every optimizer has one tuned lr per module, does searching for the best rule per module still help?

---

## Results

**Loss** is cross-entropy on web text the model never trained on, and lower is better. A **world** is a separate stretch of text with its own held-out rows. We count a gap as real only if its mean over worlds exceeds .005 and 2 errors. Every comparison uses one training run per setting per world.

### Try Muon's recipe before searching for rules

Muon's recipe uses Muon on hidden matrices and Adam on the embedding, head and norm scales. The standard baseline uses Adam everywhere. Muon's recipe won in all 5 worlds at both lengths, by .0162 at 250 steps and .0107 at 1,000 steps.

> **Muon's recipe had the lowest loss of the published recipes at 250 steps, and its lead fell with length, to +.0027 in its weakest world at 1,000 steps.**

### Compare a new rule against lr tuning, or you may credit the rule for the lr

The rule search spent 218 settings at 250 steps and beat Muon's recipe by .0073. It put Lion on attention's query and key matrices and kept Muon or Adam elsewhere. A search over lrs alone spent 155 settings and tied Muon's recipe at +.0033. The search's remaining edge is .0040, positive in all 5 worlds and under our floor.

> **At 250 steps about half of the rule search's gain came from lrs that needed no new rule.**

### Do not expect the 250-step winner to survive at 1,000 steps

At 1,000 steps the rule search spent 193 of its 219 settings on rules and moved no rule away from the best recipe, so Lion was gone. It tied that recipe at +.0003.

> **The best recipe itself changed with length: Muon's recipe led at 250 steps, and 2605.18106's recipe tied it at 1,000 steps (+.0037, error .0027).**

### Keep Adam on the embedding and head in short runs

RowNorm on the embedding and head lost to Adam by .0239 at 250 steps and tied it at 1,000 steps (−.0007, error .0023). The paper that proposes it reports gains at larger scale than we ran, so this does not contradict it.

### Run manifold rules for long enough before judging them

Keeping attention and MLP matrices on the Stiefel manifold lost to Muon's unconstrained recipe by .0338 at 250 steps. 2606.13276 reports its result at 50,000 steps on GPT-2 small, 200 times longer, and we did not run this rule at 1,000 steps.

---

## Where it stops working

- **Size.** The largest finished model has 2 layers and width 48, and hidden matrices hold only 37% of its weights. At 250 steps, training them is worth .103, 21 times the floor, so the test can see them matter. Our guess is that short runs favour rules with fast early progress.
- **Unfinished checks.** Hidden rows were never read, speed was not measured for designs, and the audit of other scores did not run. All verdicts use web-text loss only.
- **Manifold rules** were tuned as baselines at 250 steps and were not in the search's menu.
- **Few worlds.** Five worlds detect gaps near .005 only when the spread between worlds is small.

---

## Is this new?

Partly. Choosing rules per module by hand is published. We did not find a rule search per module on transformers where every arm also tunes its lrs per module, but several close papers are unread, so that may exist. [RESULTS.md](RESULTS.md) lists what we did and did not read.

| Work | Rules differ by module | Rules are searched | Every arm tunes lrs per module |
|---|---|---|---|
| Muon | Yes, by hand | No | Not claimed |
| 2605.18106 | Yes, by hand, from symmetry theory | No | Not claimed |
| 2606.13276 | Yes, by hand | No | No, it shares hyperparameters |
| Modular Manifolds | Yes, by hand, on a small MLP | No | No, one norm budgets lrs |
| 2607.20486 OPTScientist | No, one rule for all matrices | Yes | No |
| **This repo** | **Yes** | **Yes** | **Yes** |

---

## Setup

```bash
pip install -r requirements.txt
python3 setup_data.py
python3 claims.py --tier 5s
python3 env.py --spec muon --tier 5s --world 7
```

`claims.py` reads the saved results and needs neither the data nor training. The last line trains one setting in about 10 seconds on one CPU core; `setup_data.py` downloads 392 MB.

- [RESULTS.md](RESULTS.md): per-world tables, controls, commands and literature.
- [EXPERIMENTS.md](EXPERIMENTS.md): the working log, with each prediction written before its run.
- [chart.py](chart.py) draws the chart. MIT licence; the downloaded data has its own licences.
