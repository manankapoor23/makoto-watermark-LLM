# NS-Watermark reproduction — project brief

Handoff document for a Claude Code session. Read this first, then ask me for anything ambiguous before writing code.

## Who I am / context

- Undergrad in India, self-taught on this material, **beginner level** — explain things as you go, don't assume I know PyTorch internals, DP idioms, or NLP eval conventions.
- Goal: reproduce these two papers well enough to email a professor abroad about a **Summer 2027 research internship**. The email needs a line like *"I reproduced X and found Y"*, not *"I read your paper."*
- I have not yet run anything. Repo does not exist yet.

## The two papers

Both PDFs should be placed in `papers/` in the repo.

1. **Kirchenbauer et al. (2023), "A Watermark for Large Language Models"** (arXiv 2301.10226) — the original. Official code exists at `github.com/jwkirchenbauer/lm-watermarking`. **Read their hashing implementation before writing our own** — the green/red seeding is fiddly and easy to get subtly wrong.
2. **Takezawa et al. (2025), "Necessary and Sufficient Watermark for Large Language Models"** (arXiv 2310.00833, TMLR 02/2025) — the improvement we're actually reproducing.

## The core idea in one paragraph

At each generation step, hash the **previous token** to seed an RNG, use it to randomly split the vocabulary into a "green" list (fraction `γ`) and a "red" list. Kirchenbauer's Hard-Watermark forces every token green; the Soft-Watermark adds a constant `δ` boost to green logits. A detector later replays the same split, counts green tokens, and computes a z-score:

```
z = (green_count - γ·(T-1)) / sqrt(γ·(1-γ)·(T-1))
```

If `z > Z` (typically `Z = 4`), declare machine-generated.

**The NS-Watermark's insight:** you don't need every token green — you only need enough green tokens for `z ≥ Z`. That required *fraction* shrinks as text gets longer. So instead of a per-token rule, formulate it as constrained optimization: maximize text probability subject to

```
green_count/(T-1)  ≥  γ + Z·sqrt(γ(1-γ)/(T-1))
```

Solve with dynamic programming over a `(length × green_count)` table. Result: far better text quality (up to ~30 BLEU better on MT) at the same 0% FNR.

## Build order — do NOT skip ahead

Each step must work and be tested before moving on.

### Step 1 — `watermark/hashing.py`
Green/red split from a previous token id. Pure Python + a seeded RNG, no model needed.
- `get_green_ids(prev_token_id, vocab_size, gamma, hash_key) -> set[int]`
- Must be **deterministic and reproducible** — same inputs always give same split. This is the single most important property; detection breaks entirely if it isn't.
- Test: same input twice → identical set. Different `prev_token_id` → substantially different set. `len(green) ≈ gamma * vocab_size`.

### Step 2 — `watermark/detect.py`
z-score on a token sequence. Still no model needed.
- `compute_z(token_ids, gamma, hash_key) -> float`
- Test on **synthetic sequences**: random tokens should give `z ≈ 0`; a hand-constructed all-green sequence should give large positive `z`. This validates Steps 1+2 together in minutes, before any model is downloaded.
- Also implement the repeated-n-gram skip that Kirchenbauer's paper recommends (count an n-gram only on first occurrence) — it reduces false positives on repetitive human text.

### Step 3 — `watermark/soft.py`
Kirchenbauer Soft-Watermark generation. This is the baseline we compare against.
- Straightforward loop: get logits → add `δ` to green ids → softmax → sample or argmax.
- Verify: generated text scores high z under our own detector from Step 2. If it doesn't, Step 1 or 3 is wrong.

### Step 4 — `watermark/ns.py` — the actual contribution
**4a. Algorithm 1 (naive DP, O(T²)).**
- `T[t][g]` = top-k beams of length `t` containing exactly `g` green tokens.
- Recurrence: a length-`t+1` sequence with `g` greens comes from either `T[t][g-1]` + a green token, or `T[t][g]` + a red token. Keep top-k by probability.
- Optimization from the paper: collapse everything at or above `G_max` into a single bucket, since once you have enough greens the constraint is satisfied regardless of the rest. Reduces work meaningfully.
- At the end, among all cells satisfying the constraint, return the highest-probability sequence.
- **Start tiny**: GPT-2, `T_max = 20`, `k = 1`. Print the table and watch it fill. Do not scale up until it visibly works.

**4b. Algorithm 2 (linear time).**
- First run a normal unwatermarked beam search to estimate final length `T̂`.
- Then only fill a diagonal band of the table around the expected green-accumulation trajectory, width controlled by `α`.
- Complexity drops to O(α·k·T_max).

**4c. Robustness parameter `β`.**
- Tighten the constraint to `γ + β + Z·sqrt(...)` so the watermark survives up to ~50β% of tokens being edited.

### Step 5 — `experiments/`
- Detection metrics: FNR/FPR at `Z = 4`.
- Quality: perplexity (open-ended generation). BLEU only if we do the MT task.
- Reproduce the **shape** of the paper's Figure 2: Soft-Watermark's z-score climbs with length (wasteful); NS-Watermark's stays flat near the threshold. This plot is the single most convincing artifact for the professor email.
- Also worth doing: quality-vs-`α` and running-time-vs-`α` (paper's Figs. 3 and 4).

## Practical constraints

- **Do not use LLaMA-7B or NLLB-200-3.3B.** Use **GPT-2** (or OPT-125M/1.3B) locally. We're validating the algorithm, not matching absolute numbers.
- Expect to reproduce the *qualitative pattern* (NS ≫ Soft on quality, both ~0% FNR), not Table 1's exact figures. That's fine and expected — say so in the README.
- Paper's hyperparameters for reference: `Z = 4`, `k = 1`, `T_max = 100`, `γ ∈ {0.1, 0.01, 0.001, 0.0001}`, `δ ∈ {4, 6, 8}`, `α ∈ {1..5}`, `β ∈ {0, 0.05, 0.1, 0.2}`. Paper found best BLEU at **small γ**, which conveniently also makes the DP cheaper.

## First thing to do in the Claude Code session

**Get my machine details** — I asked for this and it couldn't be done from the web chat:

```bash
sw_vers && uname -m && python3 --version && sysctl hw.memsize
```

`uname -m` matters most: `arm64` (Apple Silicon) means PyTorch MPS acceleration is available and GPT-2 experiments will be comfortable; `x86_64` means CPU-only, so keep sequences short and models small.

Then scaffold the repo, set up a venv, install `torch` + `transformers`, and start on Step 1.

## Plotting conventions (for later, but decide early)

- `matplotlib` + `scienceplots` style for the paper-quality look.
- Save as **PDF or SVG**, not PNG.
- **Same color for the same method in every figure** — pick e.g. Soft = orange, Adaptive Soft = amber, NS = teal, no-watermark = gray, and never deviate.
- Direct-label lines at their endpoints rather than using a legend box, when there are only 3–4 series.
- Average over multiple seeds and show a shaded band, not a single run.

## Stretch goals — this is what actually impresses

A clean replication shows competence but not initiative. The paper's own Limitations section names two open problems:

1. **Speed.** Even Algorithm 2 is slower than Soft-Watermark. A concrete, measured speedup — e.g. vectorizing the DP across beams on GPU instead of the sequential Python loop the pseudocode implies — is a real contribution.
2. **Undetectable watermarking.** The authors explicitly call combining minimum-quality-degradation with undetectability (Christ et al., Kuditipudi et al.) "one of the interesting directions."

A third, cheaper option: **run it on a task or language the paper didn't test and report where it breaks.** Negative results are still findings.

Whichever we do, the repo needs a clear README with the reproduction plots up top, because that link is what goes in the email.

## Things to explain to me as we go

I want to actually understand this, not just have working code. When we hit these, walk me through them:
- Why the split is seeded by the *previous* token specifically, and what breaks if you seed it differently.
- Why the required green *fraction* shrinks with length (the coin-flip intuition).
- What "top-k beams per cell" means concretely, and how it differs from ordinary beam search.
- Why estimating `T̂` from an unwatermarked draft is a legitimate approximation rather than cheating.# NS-Watermark reproduction — project brief

Handoff document for a Claude Code session. Read this first, then ask me for anything ambiguous before writing code.

## Who I am / context

- Undergrad in India, self-taught on this material, **beginner level** — explain things as you go, don't assume I know PyTorch internals, DP idioms, or NLP eval conventions.
- Goal: reproduce these two papers well enough to email a professor abroad about a **Summer 2027 research internship**. The email needs a line like *"I reproduced X and found Y"*, not *"I read your paper."*
- I have not yet run anything. Repo does not exist yet.

## The two papers

Both PDFs should be placed in `papers/` in the repo.

1. **Kirchenbauer et al. (2023), "A Watermark for Large Language Models"** (arXiv 2301.10226) — the original. Official code exists at `github.com/jwkirchenbauer/lm-watermarking`. **Read their hashing implementation before writing our own** — the green/red seeding is fiddly and easy to get subtly wrong.
2. **Takezawa et al. (2025), "Necessary and Sufficient Watermark for Large Language Models"** (arXiv 2310.00833, TMLR 02/2025) — the improvement we're actually reproducing.

## The core idea in one paragraph

At each generation step, hash the **previous token** to seed an RNG, use it to randomly split the vocabulary into a "green" list (fraction `γ`) and a "red" list. Kirchenbauer's Hard-Watermark forces every token green; the Soft-Watermark adds a constant `δ` boost to green logits. A detector later replays the same split, counts green tokens, and computes a z-score:

```
z = (green_count - γ·(T-1)) / sqrt(γ·(1-γ)·(T-1))
```

If `z > Z` (typically `Z = 4`), declare machine-generated.

**The NS-Watermark's insight:** you don't need every token green — you only need enough green tokens for `z ≥ Z`. That required *fraction* shrinks as text gets longer. So instead of a per-token rule, formulate it as constrained optimization: maximize text probability subject to

```
green_count/(T-1)  ≥  γ + Z·sqrt(γ(1-γ)/(T-1))
```

Solve with dynamic programming over a `(length × green_count)` table. Result: far better text quality (up to ~30 BLEU better on MT) at the same 0% FNR.

## Build order — do NOT skip ahead

Each step must work and be tested before moving on.

### Step 1 — `watermark/hashing.py`
Green/red split from a previous token id. Pure Python + a seeded RNG, no model needed.
- `get_green_ids(prev_token_id, vocab_size, gamma, hash_key) -> set[int]`
- Must be **deterministic and reproducible** — same inputs always give same split. This is the single most important property; detection breaks entirely if it isn't.
- Test: same input twice → identical set. Different `prev_token_id` → substantially different set. `len(green) ≈ gamma * vocab_size`.

### Step 2 — `watermark/detect.py`
z-score on a token sequence. Still no model needed.
- `compute_z(token_ids, gamma, hash_key) -> float`
- Test on **synthetic sequences**: random tokens should give `z ≈ 0`; a hand-constructed all-green sequence should give large positive `z`. This validates Steps 1+2 together in minutes, before any model is downloaded.
- Also implement the repeated-n-gram skip that Kirchenbauer's paper recommends (count an n-gram only on first occurrence) — it reduces false positives on repetitive human text.

### Step 3 — `watermark/soft.py`
Kirchenbauer Soft-Watermark generation. This is the baseline we compare against.
- Straightforward loop: get logits → add `δ` to green ids → softmax → sample or argmax.
- Verify: generated text scores high z under our own detector from Step 2. If it doesn't, Step 1 or 3 is wrong.

### Step 4 — `watermark/ns.py` — the actual contribution
**4a. Algorithm 1 (naive DP, O(T²)).**
- `T[t][g]` = top-k beams of length `t` containing exactly `g` green tokens.
- Recurrence: a length-`t+1` sequence with `g` greens comes from either `T[t][g-1]` + a green token, or `T[t][g]` + a red token. Keep top-k by probability.
- Optimization from the paper: collapse everything at or above `G_max` into a single bucket, since once you have enough greens the constraint is satisfied regardless of the rest. Reduces work meaningfully.
- At the end, among all cells satisfying the constraint, return the highest-probability sequence.
- **Start tiny**: GPT-2, `T_max = 20`, `k = 1`. Print the table and watch it fill. Do not scale up until it visibly works.

**4b. Algorithm 2 (linear time).**
- First run a normal unwatermarked beam search to estimate final length `T̂`.
- Then only fill a diagonal band of the table around the expected green-accumulation trajectory, width controlled by `α`.
- Complexity drops to O(α·k·T_max).

**4c. Robustness parameter `β`.**
- Tighten the constraint to `γ + β + Z·sqrt(...)` so the watermark survives up to ~50β% of tokens being edited.

### Step 5 — `experiments/`
- Detection metrics: FNR/FPR at `Z = 4`.
- Quality: perplexity (open-ended generation). BLEU only if we do the MT task.
- Reproduce the **shape** of the paper's Figure 2: Soft-Watermark's z-score climbs with length (wasteful); NS-Watermark's stays flat near the threshold. This plot is the single most convincing artifact for the professor email.
- Also worth doing: quality-vs-`α` and running-time-vs-`α` (paper's Figs. 3 and 4).

## Practical constraints

- **Do not use LLaMA-7B or NLLB-200-3.3B.** Use **GPT-2** (or OPT-125M/1.3B) locally. We're validating the algorithm, not matching absolute numbers.
- Expect to reproduce the *qualitative pattern* (NS ≫ Soft on quality, both ~0% FNR), not Table 1's exact figures. That's fine and expected — say so in the README.
- Paper's hyperparameters for reference: `Z = 4`, `k = 1`, `T_max = 100`, `γ ∈ {0.1, 0.01, 0.001, 0.0001}`, `δ ∈ {4, 6, 8}`, `α ∈ {1..5}`, `β ∈ {0, 0.05, 0.1, 0.2}`. Paper found best BLEU at **small γ**, which conveniently also makes the DP cheaper.

## First thing to do in the Claude Code session

**Get my machine details** — I asked for this and it couldn't be done from the web chat:

```bash
sw_vers && uname -m && python3 --version && sysctl hw.memsize
```

`uname -m` matters most: `arm64` (Apple Silicon) means PyTorch MPS acceleration is available and GPT-2 experiments will be comfortable; `x86_64` means CPU-only, so keep sequences short and models small.

Then scaffold the repo, set up a venv, install `torch` + `transformers`, and start on Step 1.

## Plotting conventions (for later, but decide early)

- `matplotlib` + `scienceplots` style for the paper-quality look.
- Save as **PDF or SVG**, not PNG.
- **Same color for the same method in every figure** — pick e.g. Soft = orange, Adaptive Soft = amber, NS = teal, no-watermark = gray, and never deviate.
- Direct-label lines at their endpoints rather than using a legend box, when there are only 3–4 series.
- Average over multiple seeds and show a shaded band, not a single run.

## Stretch goals — this is what actually impresses

A clean replication shows competence but not initiative. The paper's own Limitations section names two open problems:

1. **Speed.** Even Algorithm 2 is slower than Soft-Watermark. A concrete, measured speedup — e.g. vectorizing the DP across beams on GPU instead of the sequential Python loop the pseudocode implies — is a real contribution.
2. **Undetectable watermarking.** The authors explicitly call combining minimum-quality-degradation with undetectability (Christ et al., Kuditipudi et al.) "one of the interesting directions."

A third, cheaper option: **run it on a task or language the paper didn't test and report where it breaks.** Negative results are still findings.

Whichever we do, the repo needs a clear README with the reproduction plots up top, because that link is what goes in the email.

## Things to explain to me as we go

I want to actually understand this, not just have working code. When we hit these, walk me through them:
- Why the split is seeded by the *previous* token specifically, and what breaks if you seed it differently.
- Why the required green *fraction* shrinks with length (the coin-flip intuition).
- What "top-k beams per cell" means concretely, and how it differs from ordinary beam search.
- Why estimating `T̂` from an unwatermarked draft is a legitimate approximation rather than cheating.