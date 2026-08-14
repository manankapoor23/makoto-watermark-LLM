"""Green/red vocabulary split, seeded by the previous token.

This is the foundation of both watermarks (Kirchenbauer et al. 2023,
Takezawa et al. 2025). Everything else depends on this function being
*perfectly reproducible*: the generator and the detector must derive the
identical green set from the same previous token, or detection is noise.

No model, no torch -- pure stdlib.
"""

import random
from functools import lru_cache

# Kirchenbauer et al.'s default. A large prime, so that hash_key * prev_token
# spreads seeds out instead of clustering them.
DEFAULT_HASH_KEY = 15485863


@lru_cache(maxsize=8192)
def get_green_ids(prev_token_id, vocab_size, gamma, hash_key=DEFAULT_HASH_KEY):
    """Return the frozenset of 'green' token ids for a given previous token.

    Deterministic: same arguments always return the same set, in this process
    and in any future one. That is the single property detection depends on.

    ponytail: lru_cache because this is a pure function called once per
    (token, step) and recomputed constantly by the DP in Step 4. frozenset
    (not set) so a caller can never mutate a cached entry out from under us.
    """
    if not 0 < gamma < 1:
        raise ValueError(f"gamma must be in (0, 1), got {gamma}")
    if not 0 <= prev_token_id < vocab_size:
        raise ValueError(f"prev_token_id {prev_token_id} outside vocab {vocab_size}")

    # NEVER use the builtin hash() here: it is salted per-process for str/bytes,
    # so green sets would differ between the generating run and the detecting run.
    rng = random.Random(hash_key * prev_token_id)

    # The paper uses gamma = 0.0001, which for GPT-2 rounds down to 5 tokens.
    # Clamp to >= 1 so the green list can never be empty (which would make the
    # watermark unsatisfiable rather than merely strict).
    greenlist_size = max(1, int(vocab_size * gamma))

    return frozenset(rng.sample(range(vocab_size), greenlist_size))


def _demo():
    """Self-check: the three properties the whole project rests on."""
    V, gamma = 50257, 0.25  # GPT-2 vocab size

    # 1. Determinism -- the property detection depends on.
    a = get_green_ids(1234, V, gamma)
    get_green_ids.cache_clear()  # recompute from scratch, not from cache
    assert a == get_green_ids(1234, V, gamma), "not deterministic!"

    # 2. Different previous tokens give substantially different splits.
    b = get_green_ids(5678, V, gamma)
    overlap = len(a & b) / len(a)
    # Two independent gamma-subsets overlap in ~gamma of each other, not ~1.0.
    assert 0.20 < overlap < 0.30, f"splits not independent enough: {overlap:.3f}"

    # 3. Size is gamma * |V|.
    assert len(a) == int(V * gamma), f"wrong size: {len(a)}"

    # 4. Tiny gamma (the paper's best-BLEU setting) still yields a usable list.
    assert len(get_green_ids(42, V, 0.0001)) == 5

    print(f"ok | |V|={V} gamma={gamma} |green|={len(a)} overlap={overlap:.3f}")


if __name__ == "__main__":
    _demo()
