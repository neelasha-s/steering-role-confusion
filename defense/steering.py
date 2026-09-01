"""Masked steering: apply the userness direction to an arbitrary set of token
positions, rather than to one contiguous span.

WHY THIS FILE EXISTS
--------------------
The base experiment steers a single contiguous span -- the injected command --
described by one (start, end) pair per batch row. That is enough for the "oracle"
variant, which assumes you already know where the injection sits.

No real defender knows that. The deployable variants need to steer either every
token of the tool block (blind) or a scattered set of positions the probe flagged
(gated). Scattered positions cannot be described by a start and an end, so the
representation has to change from a SPAN to a MASK: one True/False flag per token
position.

That sounds like a small change and is not. Positions are in PADDED coordinates,
which shift depending on how the batch was padded, and if the bookkeeping is off
by a few tokens you will steer the wrong tokens and get numbers that look
completely plausible and mean nothing. That failure is silent. This is why this
file is hand-written and covered by tests rather than generated.

THE MATH (unchanged from the base experiment)
---------------------------------------------
For every selected token position p:

    h'_p = h_p + c * ||h_p|| * v̂

where h_p is that token's residual vector, ||h_p|| is its Euclidean norm, v̂ is
the unit steering direction, and c is the dose. Scaling by each token's own norm
is what makes c dimensionless: the relative size of the change,
||h'_p - h_p|| / ||h_p||, is exactly |c| for every token regardless of how large
that token's activations happen to be. That property is what lets "dose" mean the
same thing across positions -- and it is the first thing the tests check.

Unselected positions must come out bit-for-bit identical to the input.

Run the tests with:
    .venv/bin/python -m pytest defense/tests/test_steering.py -v
"""

import torch


def mask_from_spans(spans, batch_size, seq_len, device=None):
    """Convert the old span representation into the new mask representation.

    This exists so the new code can be proven equivalent to the known-good base
    experiment: a mask built from a span must steer exactly the tokens the old
    span-based hook steered. The test suite relies on that equivalence.

    Args:
        spans:      list of (start, end) tuples, one per batch row, in PADDED
                    coordinates. `end` is EXCLUSIVE, matching Python slicing.
                    A row with end <= start selects nothing.
        batch_size: number of rows, B.
        seq_len:    sequence length, T.
        device:     torch device for the result (None = CPU).

    Returns:
        (B, T) bool tensor. True at positions inside that row's span.

    Raises:
        ValueError: if len(spans) != batch_size.

    Hint: start from torch.zeros(..., dtype=torch.bool) and fill each row's slice.
    """
    raise NotImplementedError("mask_from_spans")


def apply_masked_steering(hidden, mask, direction, coeff):
    """Add `coeff * ||h|| * unit(direction)` to every position where mask is True.

    This is the pure-math core of the intervention, deliberately separated from
    the forward-hook plumbing below so it can be tested without a model.

    Args:
        hidden:    (B, T, D) tensor. May be bfloat16 or float16 on the real model,
                   so do the arithmetic in float32 and cast the result back to
                   hidden's original dtype. Low-precision norms are inaccurate
                   enough to break the dose property the tests check.
        mask:      (B, T) bool tensor. True = steer this position.
        direction: (D,) tensor. Normalized internally, so callers may pass a
                   non-unit vector and get the same answer.
        coeff:     float dose. Negative steers away from the direction.

    Returns:
        A NEW (B, T, D) tensor with hidden's dtype. Does not modify `hidden`.
        Positions where mask is False are bit-for-bit identical to the input.

    Raises:
        ValueError: if mask's shape is not hidden.shape[:2], or direction's length
                    is not hidden.shape[-1]. Raise loudly here -- a silently
                    misaligned mask is the exact bug this file is defending
                    against, and it is far better to crash than to steer the wrong
                    tokens and report the result.

    Fast paths: if coeff == 0.0 or the mask selects nothing, return a copy of
    hidden unchanged.

    Hints:
      - Per-token norms: hidden.norm(dim=-1, keepdim=True) gives (B, T, 1).
      - (B,T,1) * (D,) broadcasts to (B,T,D), which is the whole delta at once --
        no Python loop over rows or positions is needed.
      - torch.where(mask.unsqueeze(-1), steered, original) applies the mask
        without touching unselected positions.
    """
    raise NotImplementedError("apply_masked_steering")


class MaskedSteeringHook:
    """Forward hook that applies masked steering to a layer's output.

    A forward hook is a function PyTorch calls every time data passes through a
    module, handing it that module's output and letting it return a replacement.
    Registering one on the layer-11 block is how the intervention gets injected
    into an otherwise normal forward pass.

    Usage mirrors the base experiment's SteeringHook:

        handle = STEER_MOD.register_forward_hook(
            MaskedSteeringHook(direction, coeff, mask))
        try:
            out = model.generate(...)
        finally:
            handle.remove()          # ALWAYS remove, even on exception

    Args:
        direction: (D,) steering direction, normalized internally.
        coeff:     float dose.
        mask:      (B, T) bool tensor over the PREFILL sequence.
    """

    def __init__(self, direction, coeff, mask):
        self.direction = direction
        self.coeff = coeff
        self.mask = mask

    def __call__(self, module, inputs, output):
        """Return `output` with the masked positions steered.

        Three things this must get right:

        1. `output` may be a bare tensor or a tuple whose first element is the
           hidden states. Handle both, and return the same KIND of thing you were
           given -- returning a tensor where a tuple was expected breaks the model.

        2. Skip generated tokens. During generation the model runs one token at a
           time with a KV cache, so hidden.shape[1] == 1. The mask describes the
           prefill sequence and means nothing there. Return `output` untouched
           whenever hidden.shape[1] == 1. This preserves the base experiment's
           choice to steer the prompt only and leave generation unsteered.

        3. Refuse to run on a mismatched mask. If hidden.shape[:2] does not equal
           self.mask.shape, raise ValueError. Do NOT pad, trim, or broadcast to
           make it fit -- a mask that no longer lines up with the sequence is a
           bug upstream, and quietly "fixing" it here is how you end up steering
           arbitrary tokens and never finding out.

        Note the base experiment mutates the output tensor in place. Doing the
        same is fine and avoids an allocation: compute the steered values with
        apply_masked_steering, then write them back into the existing tensor.
        """
        raise NotImplementedError("MaskedSteeringHook.__call__")
