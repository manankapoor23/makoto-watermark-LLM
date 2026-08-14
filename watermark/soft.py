"""Kirchenbauer et al. Soft-Watermark generation (Algorithm 2 of their paper).

The baseline the NS-Watermark is compared against. At each step: get logits,
add a constant `delta` to every green-list logit, softmax, sample.

Why this beats their Hard-Watermark: the boost is a *soft* nudge, so when the
model is highly confident (low entropy, e.g. "Barack" -> "Obama") the correct
red token still wins and fluency survives. When entropy is high there are many
near-tied candidates and delta reliably tips the choice green.

Why the NS-Watermark still beats this: delta is a fixed per-token knob, so the
green fraction stays ~constant with length. But the fraction *needed* shrinks
like 1/sqrt(T). So on long texts the Soft-Watermark massively overshoots the
z-threshold, paying quality for detection margin it does not need. See Step 4.
"""

from typing import NamedTuple

import torch

from detect import compute_z
from hashing import DEFAULT_HASH_KEY, get_green_ids


class Generation(NamedTuple):
    text: str            # decoded continuation only (no prompt)
    prompt_ids: list     # token ids of the prompt
    new_ids: list        # token ids the model generated
    detect_ids: list     # what to hand the detector -- see note in generate()
    vocab_size: int


def pick_device():
    """MPS on Apple Silicon, else CPU."""
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(name="gpt2", device=None):
    """Load a causal LM + tokenizer. GPT-2 (124M) fits comfortably in 8GB."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or pick_device()
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
    return model, tok, device


@torch.no_grad()
def generate(model, tok, prompt, gamma=0.25, delta=2.0, max_new_tokens=100,
             do_sample=True, temperature=1.0, seed=0, hash_key=DEFAULT_HASH_KEY,
             device=None):
    """Generate with the Soft-Watermark. delta=0.0 gives the unwatermarked baseline.

    That delta=0 case is not a throwaway: Step 4's Algorithm 2 needs an
    unwatermarked draft to estimate the final length T-hat, and Step 5 needs it
    as the quality reference point.
    """
    device = device or next(model.parameters()).device
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    prompt_ids = ids[0].tolist()

    # Use the logit dimension, NOT tok.vocab_size. Some models pad their output
    # embedding (e.g. 50257 -> 50304), and if the generator and the detector
    # disagree on vocab_size by even one, every green list differs and detection
    # silently collapses to noise.
    vocab_size = model.config.vocab_size

    gen = torch.Generator(device="cpu").manual_seed(seed)
    new_ids = []

    for _ in range(max_new_tokens):
        # ponytail: no KV cache -- re-runs the full prefix each step, so O(T^2)
        # forward cost. Fine for GPT-2 at T<=100; add past_key_values if Step 4
        # profiling shows the DP is forward-bound (it likely will be).
        logits = model(ids).logits[0, -1, :].float()
        assert logits.shape[-1] == vocab_size, "config/logit vocab mismatch"

        if delta:
            prev = ids[0, -1].item()
            green = get_green_ids(prev, vocab_size, gamma, hash_key)
            idx = torch.tensor(sorted(green), device=logits.device)
            logits[idx] += delta

        if do_sample:
            probs = torch.softmax(logits / temperature, dim=-1)
            nxt = torch.multinomial(probs.cpu(), 1, generator=gen).item()
        else:
            nxt = int(logits.argmax())

        if nxt == tok.eos_token_id:
            break

        new_ids.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)

    # Score only within the generated text, NOT seeded from the prompt's last
    # token. Kirchenbauer's own implementation does seed from the prompt (one
    # extra token of evidence), but Takezawa et al. Alg. 1 leaves the first
    # generated token unconstrained. We follow the NS paper for BOTH methods so
    # Step 5 compares them under an identical detector. Mixing conventions would
    # hand one method a free scored token.
    detect_ids = new_ids

    return Generation(tok.decode(new_ids), prompt_ids, new_ids, detect_ids, vocab_size)


def _demo():
    """Verify: watermarked text scores high z under OUR detector, plain text does not.

    If this fails, Step 1 or Step 3 is wrong -- that is the whole point of it.
    """
    gamma, delta = 0.25, 2.0
    prompt = ("The tower is 324 metres tall, about the same height as an "
              "81-storey building, and the largest structure in Paris.")

    model, tok, device = load_model()
    print(f"loaded gpt2 on {device}\n")

    plain = generate(model, tok, prompt, gamma=gamma, delta=0.0, max_new_tokens=100)
    marked = generate(model, tok, prompt, gamma=gamma, delta=delta, max_new_tokens=100)

    z_plain = compute_z(plain.detect_ids, plain.vocab_size, gamma)
    z_marked = compute_z(marked.detect_ids, marked.vocab_size, gamma)

    print(f"--- no watermark (delta=0) | z={z_plain:+.2f} | {len(plain.new_ids)} tokens ---")
    print(plain.text.strip()[:300], "\n")
    print(f"--- soft watermark (delta={delta}) | z={z_marked:+.2f} | {len(marked.new_ids)} tokens ---")
    print(marked.text.strip()[:300], "\n")

    assert z_marked > 4, f"watermark not detectable: z={z_marked:.2f} -- Step 1 or 3 is wrong"
    assert z_marked > z_plain, "watermark did not raise z above baseline"
    print(f"ok | watermarked z={z_marked:+.2f} > 4 threshold; baseline z={z_plain:+.2f}")


if __name__ == "__main__":
    _demo()
