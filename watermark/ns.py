"""NS-Watermark, Algorithm 1: the naive O(gamma * k * T_max^2) dynamic program.

Takezawa et al. (TMLR 2025), Section 3.2. This is the actual contribution.

The Soft-Watermark asks "should this token be green?" once per token, in
isolation. The NS-Watermark instead asks a global question:

    maximise  p(x_1:T | prompt)
    s.t.      green_count/(T-1) >= gamma + beta + Z*sqrt(gamma(1-gamma)/(T-1))

i.e. produce the most probable text that *just barely* clears the detection
threshold. Nothing more is spent on the watermark than detection requires.

The solution is a DP over a table T[t][g] = "the k most probable texts of
length t containing exactly g green tokens". Compare with 0/1 knapsack:

    knapsack dp[i][w]  ->  NS  T[t][g]
    item index i       ->  token position t
    capacity w         ->  green count g
    stores ONE number  ->  stores k FULL SEQUENCES

Knapsack stores one number per cell because of optimal substructure: two item
subsets reaching the same (i, w) are interchangeable, since the future depends
only on remaining capacity. That fails here -- p(next | "The cat sat") and
p(next | "Quantum physics states") differ even though both prefixes have the
same (length, green count). So each cell hedges by keeping the top-k prefixes,
exactly as ordinary beam search does. k=1 collapses to greedy.
"""

import math
from typing import NamedTuple

import torch

from detect import DEFAULT_Z_THRESHOLD, compute_z, required_green_count
from hashing import DEFAULT_HASH_KEY, get_green_ids
from soft import load_model

NEG_INF = float("-inf")


class Beam(NamedTuple):
    tokens: tuple  # the generated token ids so far
    logprob: float  # sum of log p -- logs, not products, or 100 tokens underflows


class NSGeneration(NamedTuple):
    text: str
    tokens: list
    logprob: float
    green_count: int
    required: int
    z: float
    vocab_size: int


@torch.no_grad()
def _row_logprobs(model, prompt_ids, beams, device):
    """One batched forward pass for a whole row of the table.

    Every beam in row t has exactly t tokens, so the batch needs no padding and
    no attention mask. A beam in row t-1 feeds *two* cells of row t (green ->
    g+1, red -> g), so computing this once per row rather than once per cell is
    a free constant-factor win.
    """
    seqs = torch.tensor([prompt_ids + list(b.tokens) for b in beams], device=device)
    logits = model(seqs).logits[:, -1, :].float()
    return torch.log_softmax(logits, dim=-1)


def _topk_masked(logprobs, allowed_idx, k, exclude=False):
    """Top-k next tokens restricted to (or excluding) a set of ids."""
    if allowed_idx is None:  # unrestricted: the whole vocabulary
        masked = logprobs
    elif exclude:  # red list = complement of green
        masked = logprobs.clone()
        masked[allowed_idx] = NEG_INF
    else:  # green list only
        masked = torch.full_like(logprobs, NEG_INF)
        masked[allowed_idx] = logprobs[allowed_idx]
    vals, toks = masked.topk(min(k, masked.numel()))
    return [(float(v), int(t)) for v, t in zip(vals, toks) if v != NEG_INF]


def render(table, T_max, G_max, gamma, z_threshold, beta, upto=None):
    """ASCII dump of the DP table -- the paper's Figure 1a, in your terminal.

      #  populated AND already satisfies the constraint at that length
      o  populated, not yet enough greens
      .  reachable but empty (pruned: no beam survived here)
         blank = unreachable (g can never exceed t-1)
    """
    upto = upto or T_max
    lines = []
    for g in range(G_max, -1, -1):
        cells = []
        for t in range(1, upto + 1):
            if g > min(t - 1, G_max):
                cells.append("  ")  # unreachable: cannot have more greens than pairs
            elif not table[t].get(g):
                cells.append(" .")
            else:
                need = required_green_count(t, gamma, z_threshold, beta) if t > 1 else 1
                cells.append(" #" if g >= need else " o")
        label = f"{g:>3}" + ("*" if g == G_max else " ")
        lines.append(f"{label}|" + "".join(cells))
    lines.append("    +" + "--" * upto)
    lines.append("     " + "".join(f"{t:>2}" if t % 5 == 0 else "  " for t in range(1, upto + 1)))
    lines.append(f"     g=green count (*=G_max, collapsed bucket), t=length")
    return "\n".join(lines)


def generate(model, tok, prompt, gamma=0.25, T_max=20, k=1,
             z_threshold=DEFAULT_Z_THRESHOLD, beta=0.0,
             hash_key=DEFAULT_HASH_KEY, device=None, verbose=False):
    """Solve the NS-Watermark with the naive DP (Algorithm 1)."""
    device = device or next(model.parameters()).device
    prompt_ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
    V = model.config.vocab_size

    # G_max: greens needed at the *maximum* length. Once a beam banks this many,
    # the constraint is satisfied no matter what follows -- so we stop tracking
    # the exact count and collapse everything above into one bucket. This is what
    # turns an O(T^2)-wide table into an O(gamma*T) one.
    G_max = required_green_count(T_max, gamma, z_threshold, beta)

    table = {t: {} for t in range(1, T_max + 1)}
    finished = []  # sequences that hit EOS while already satisfying the constraint

    # t=1: the first generated token is unconstrained (no pair scored yet).
    first = _row_logprobs(model, prompt_ids, [Beam((), 0.0)], device)[0]
    table[1][0] = [Beam((t,), lp) for lp, t in _topk_masked(first, None, k)]

    for t in range(2, T_max + 1):
        prev_row = table[t - 1]
        flat = [(g, b) for g in sorted(prev_row) for b in prev_row[g]]
        if not flat:
            break
        lps = _row_logprobs(model, prompt_ids, [b for _, b in flat], device)

        by_g = {}
        for i, (g, b) in enumerate(flat):
            by_g.setdefault(g, []).append((b, i))

        for g in range(0, min(t - 1, G_max) + 1):
            cands = []

            # Arrive by adding a GREEN token to a beam with one fewer green.
            for beam, i in by_g.get(g - 1, []):
                green = get_green_ids(beam.tokens[-1], V, gamma, hash_key)
                gidx = torch.tensor(sorted(green), device=lps.device)
                for lp, nxt in _topk_masked(lps[i], gidx, k):
                    cands.append(Beam(beam.tokens + (nxt,), beam.logprob + lp))

            # Arrive by adding a RED token, keeping the green count unchanged.
            # At g == G_max the constraint is already met, so the token is drawn
            # from the FULL vocabulary -- no green/red restriction at all. This
            # is the whole point of the collapsed bucket (paper Alg. 1 line 22).
            for beam, i in by_g.get(g, []):
                if g == G_max:
                    allowed, exclude = None, False
                else:
                    green = get_green_ids(beam.tokens[-1], V, gamma, hash_key)
                    allowed = torch.tensor(sorted(green), device=lps.device)
                    exclude = True
                for lp, nxt in _topk_masked(lps[i], allowed, k, exclude=exclude):
                    cands.append(Beam(beam.tokens + (nxt,), beam.logprob + lp))

            # Keep only the k most probable prefixes in this cell.
            cands.sort(key=lambda b: b.logprob, reverse=True)
            kept = []
            for b in cands:
                if b.tokens[-1] == tok.eos_token_id:
                    if g >= required_green_count(t, gamma, z_threshold, beta):
                        finished.append(b)  # valid early stop
                    continue
                kept.append(b)
                if len(kept) == k:
                    break
            if kept:
                table[t][g] = kept

        if verbose:
            print(f"\nt={t}")
            print(render(table, T_max, G_max, gamma, z_threshold, beta, upto=t))

    # Only the G_max cell is guaranteed to satisfy the constraint at T_max,
    # since G_max was defined as the requirement at exactly that length.
    pool = finished + table[T_max].get(G_max, [])
    if not pool:
        raise RuntimeError(
            f"no feasible sequence: needed {G_max} greens in {T_max} tokens. "
            "Try a larger T_max, a smaller gamma, or a larger k.")

    # ponytail: argmax of raw logprob, exactly as the paper specifies. This
    # length-biases toward shorter EOS-terminated texts; add length
    # normalisation if variable-length generation ever looks truncated.
    best = max(pool, key=lambda b: b.logprob)
    ids = list(best.tokens)
    z = compute_z(ids, V, gamma, hash_key)
    greens = sum(1 for p, c in zip(ids, ids[1:]) if c in get_green_ids(p, V, gamma, hash_key))

    return NSGeneration(tok.decode(ids), ids, best.logprob, greens,
                        required_green_count(len(ids), gamma, z_threshold, beta),
                        z, V)


def green_pattern(ids, V, gamma, hash_key=DEFAULT_HASH_KEY):
    """'G' where a token is green, '.' where red. First token is unscored."""
    marks = ["_"]
    for prev, cur in zip(ids, ids[1:]):
        marks.append("G" if cur in get_green_ids(prev, V, gamma, hash_key) else ".")
    return "".join(marks)


def _demo():
    """Tiny and visible, exactly as the brief demands: GPT-2, T_max=20, k=1."""
    import soft

    gamma, T_max, k, Z = 0.25, 20, 1, DEFAULT_Z_THRESHOLD
    prompt = ("The tower is 324 metres tall, about the same height as an "
              "81-storey building, and the largest structure in Paris.")

    model, tok, device = load_model()
    print(f"loaded gpt2 on {device}")
    G_max = required_green_count(T_max, gamma, Z)
    print(f"gamma={gamma} T_max={T_max} k={k} Z={Z} -> G_max={G_max} "
          f"({G_max}/{T_max-1} = {G_max/(T_max-1):.1%} of tokens must be green)\n")

    out = generate(model, tok, prompt, gamma=gamma, T_max=T_max, k=k, verbose=True)

    print(f"\n--- NS-Watermark result ---")
    print(f"text     : {out.text.strip()}")
    print(f"pattern  : {green_pattern(out.tokens, out.vocab_size, gamma)}")
    print(f"greens   : {out.green_count} (needed {out.required})")
    print(f"z-score  : {out.z:+.2f}  (threshold {Z})")

    assert out.green_count >= out.required, "constraint violated -- DP is wrong"
    assert out.z >= Z, f"z={out.z:.2f} below threshold -- DP is wrong"

    # The signature claim: NS lands just above Z, Soft overshoots it.
    s = soft.generate(model, tok, prompt, gamma=gamma, delta=2.0, max_new_tokens=T_max)
    z_soft = compute_z(s.detect_ids, s.vocab_size, gamma)
    print(f"\nsoft-watermark z at same length: {z_soft:+.2f}   NS z: {out.z:+.2f}")
    print("ok | constraint satisfied and detectable")


if __name__ == "__main__":
    _demo()
