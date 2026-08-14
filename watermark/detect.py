"""Watermark detection: a one-proportion z-test on green-token counts.

The statistical core of both papers. Under the null hypothesis H0 ("this text
was written without knowledge of the green/red rule"), each token is green with
probability gamma, independently. So the green count is Binomial(T-1, gamma),
and we standardise it into a z-score.

    z = (green_count - gamma*(T-1)) / sqrt(gamma*(1-gamma)*(T-1))

z > Z (typically Z=4) => reject H0 => declare machine-generated.
Z=4 corresponds to a one-sided p-value of ~3e-5, i.e. a very conservative
false-positive rate. That conservatism is precisely the slack the NS-Watermark
exploits in Step 4.

No model, no torch -- pure stdlib.
"""

import math

from hashing import DEFAULT_HASH_KEY, get_green_ids

DEFAULT_Z_THRESHOLD = 4.0


def compute_z(token_ids, vocab_size, gamma, hash_key=DEFAULT_HASH_KEY,
              skip_repeated=True):
    """Return the watermark z-score for a token sequence.

    The first token is never scored: it has no predecessor to seed its green
    list. So T-1 tokens are scored for a length-T sequence -- this is why the
    papers' formulas say (T-1) and not T.

    skip_repeated implements Kirchenbauer et al. Sec 4.1: a repeated bigram
    (prev, cur) is scored only on first occurrence. Repeated n-grams are
    pseudo-random but *fixed*, so repetitive human text can otherwise stack up
    the same lucky green bigram and get falsely flagged. See _demo below for a
    concrete false positive this prevents.
    """
    green_count, scored = 0, 0
    seen = set()

    for prev, cur in zip(token_ids, token_ids[1:]):
        if skip_repeated:
            if (prev, cur) in seen:
                continue  # counted as neither green nor red; T is not incremented
            seen.add((prev, cur))

        scored += 1
        if cur in get_green_ids(prev, vocab_size, gamma, hash_key):
            green_count += 1

    if scored == 0:
        raise ValueError("need at least 2 distinct-bigram tokens to score")

    expected = gamma * scored
    std = math.sqrt(gamma * (1 - gamma) * scored)
    return (green_count - expected) / std


def is_watermarked(token_ids, vocab_size, gamma, hash_key=DEFAULT_HASH_KEY,
                   z_threshold=DEFAULT_Z_THRESHOLD, skip_repeated=True):
    """Reject H0 at the given threshold."""
    z = compute_z(token_ids, vocab_size, gamma, hash_key, skip_repeated)
    return z > z_threshold


def required_green_count(length, gamma, z_threshold=DEFAULT_Z_THRESHOLD, beta=0.0):
    """Minimum green tokens for a length-`length` text to reach z >= Z.

    This is the NS-Watermark constraint (Takezawa et al. Eq. 4/6), inverted:
        green/(T-1) >= gamma + beta + Z*sqrt(gamma*(1-gamma)/(T-1))
    Step 4's dynamic program is built around exactly this quantity. beta > 0
    over-provisions greens so the watermark survives post-editing (Eq. 6).

    Note the *fraction* required shrinks as length grows (the sqrt term decays),
    which is the entire insight of the NS-Watermark.
    """
    scored = length - 1
    if scored <= 0:
        raise ValueError("length must be >= 2")
    fraction = gamma + beta + z_threshold * math.sqrt(gamma * (1 - gamma) / scored)
    return math.ceil(fraction * scored)


def _demo():
    """Self-check: validates Steps 1+2 together, before any model exists."""
    import random

    V, gamma = 50257, 0.25

    # 1. Human-like text (random tokens) should sit near z = 0.
    rng = random.Random(0)
    human = [rng.randrange(V) for _ in range(200)]
    z_human = compute_z(human, V, gamma)
    assert abs(z_human) < 3, f"random text should be ~0, got {z_human:.2f}"

    # 2. A hand-built all-green sequence should score very high.
    seq = [rng.randrange(V)]
    for _ in range(199):
        seq.append(rng.choice(sorted(get_green_ids(seq[-1], V, gamma))))
    z_green = compute_z(seq, V, gamma)
    assert z_green > 20, f"all-green should be huge, got {z_green:.2f}"
    assert is_watermarked(seq, V, gamma) and not is_watermarked(human, V, gamma)

    # 3. Why skip_repeated exists: repetitive text with one lucky green bigram
    #    is a FALSE POSITIVE without dedup, and correctly ignored with it.
    a = 100
    b = next(iter(sorted(get_green_ids(a, V, gamma))))  # b is green after a
    repetitive = [a, b] * 60
    z_raw = compute_z(repetitive, V, gamma, skip_repeated=False)
    z_dedup = compute_z(repetitive, V, gamma, skip_repeated=True)
    assert z_raw > 4 > z_dedup, f"dedup did not help: raw={z_raw:.2f} dedup={z_dedup:.2f}"

    # 4. The NS constraint: required green *fraction* must shrink with length.
    frac = lambda T: required_green_count(T, gamma) / (T - 1)
    assert frac(20) > frac(100) > frac(500) > gamma
    # ...and Hard-Watermark's "all green" is wildly more than needed at T=200:
    # NS needs 75/199 = 37.3% green; Hard-Watermark forces 100%. That gap is
    # the text quality the NS-Watermark buys back.
    assert required_green_count(200, gamma) < 0.5 * 199

    print(f"ok | human z={z_human:+.2f}  all-green z={z_green:+.2f}  "
          f"repetitive raw={z_raw:+.2f} -> dedup={z_dedup:+.2f}")
    print(f"   | greens needed: T=20 {frac(20):.1%}  T=100 {frac(100):.1%}  "
          f"T=500 {frac(500):.1%}  (gamma={gamma:.0%})")


if __name__ == "__main__":
    _demo()
