# makoto-watermark-LLM

A from-scratch reproduction of the **NS-Watermark** (Takezawa et al., TMLR 2025)
and the **Soft-Watermark** baseline it improves on (Kirchenbauer et al., 2023).

- Kirchenbauer et al., *A Watermark for Large Language Models* — [arXiv:2301.10226](https://arxiv.org/abs/2301.10226)
- Takezawa et al., *Necessary and Sufficient Watermark for Large Language Models* — [arXiv:2310.00833](https://arxiv.org/abs/2310.00833)

## The idea

At each generation step, hash the previous token to seed an RNG and randomly
split the vocabulary into a *green* list (fraction `γ`) and a *red* list. A
detector replays the same split, counts green tokens, and computes a z-score:

```
z = (green_count - γ(T-1)) / sqrt(γ(1-γ)(T-1))
```

Kirchenbauer's Hard-Watermark forces **every** token green; the Soft-Watermark
adds a constant `δ` to green logits. Both are wasteful: you don't need every
token green, only enough for `z ≥ Z`. And because the mean green count grows
like `T` while its standard deviation grows only like `√T`, the required green
*fraction* shrinks as text gets longer:

| length `T` | green fraction needed (γ=0.25, Z=4) |
|---|---|
| 20  | 68.4% |
| 100 | 42.4% |
| 500 | 32.9% |

The NS-Watermark turns this into a constrained optimisation — maximise text
probability subject to `green/(T-1) ≥ γ + Z·sqrt(γ(1-γ)/(T-1))` — and solves it
with a dynamic program over a `(length × green_count)` table. Result: much
better text quality at the same 0% FNR, and the z-score guarantee becomes
*structural* rather than merely empirical.

## Status

- [x] **Step 1** — `watermark/hashing.py`: deterministic green/red split
- [x] **Step 2** — `watermark/detect.py`: z-score detector + repeated-n-gram skip
- [ ] **Step 3** — `watermark/soft.py`: Kirchenbauer Soft-Watermark baseline
- [ ] **Step 4** — `watermark/ns.py`: NS-Watermark (Alg. 1 naive DP, Alg. 2 linear)
- [ ] **Step 5** — `experiments/`: FNR/FPR, perplexity, reproduce Fig. 2

## Setup

```bash
python3.12 -m venv .venv && .venv/bin/pip install torch transformers
```

Steps 1–2 need no dependencies at all — they are pure stdlib and each file
runs its own self-check:

```bash
cd watermark && python3 hashing.py && python3 detect.py
```

## Notes

Uses **GPT-2** locally rather than the papers' LLaMA-7B / NLLB-200-3.3B. The
goal is to validate the *algorithm* and reproduce the qualitative pattern
(NS ≫ Soft on quality, both ~0% FNR), not to match absolute table values.

The two source PDFs are gitignored; download them from the arXiv links above
into `papers/` if you want them locally.
