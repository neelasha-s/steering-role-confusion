"""CPU tests for masked steering. No GPU, no model, no downloads.

    .venv/bin/python -m pytest defense/tests/test_steering.py -v

Every test here checks a property that, if broken, produces plausible-looking
numbers rather than a crash. That is the whole reason this code is tested instead
of eyeballed.

The reference implementation near the top is a faithful copy of the base
experiment's span-based SteeringHook (notebook cell 7). It is the known-good
behavior; the final test proves the new mask-based code reproduces it exactly.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from defense.steering import MaskedSteeringHook, apply_masked_steering, mask_from_spans

torch.manual_seed(0)

D = 16  # residual width, small enough to reason about by hand


# --------------------------------------------------------------------------------
# Reference: the base experiment's span-based hook, copied verbatim in behavior.
# --------------------------------------------------------------------------------

class ReferenceSpanHook:
    """notebook cell 7, SteeringHook -- the known-good span implementation."""

    def __init__(self, direction, coeff, spans):
        self.unit = (direction / direction.norm()).to(torch.float32)
        self.coeff = coeff
        self.spans = spans

    def __call__(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[1] == 1 or self.coeff == 0.0:
            return output
        for row, (start, end) in enumerate(self.spans):
            if end <= start:
                continue
            h = hidden[row, start:end, :].to(torch.float32)
            magnitude = h.norm(dim=-1, keepdim=True)
            hidden[row, start:end, :] = (
                h + self.coeff * magnitude * self.unit.to(h.device)
            ).to(hidden.dtype)
        return output


@pytest.fixture
def direction():
    return torch.randn(D)


@pytest.fixture
def hidden():
    # Deliberately varied per-token norms: if the implementation forgets to scale
    # by each token's own norm, the dose test below will catch it.
    h = torch.randn(2, 8, D)
    h[0, 3] *= 50.0
    h[1, 5] *= 0.02
    return h


# --------------------------------------------------------------------------------
# mask_from_spans
# --------------------------------------------------------------------------------

def test_mask_from_spans_marks_exactly_the_span():
    mask = mask_from_spans([(2, 5), (0, 1)], batch_size=2, seq_len=8)
    assert mask.shape == (2, 8)
    assert mask.dtype == torch.bool
    assert mask[0].tolist() == [False, False, True, True, True, False, False, False], (
        "end is exclusive, so (2, 5) must select positions 2, 3, 4 only"
    )
    assert mask[1].tolist() == [True] + [False] * 7


def test_mask_from_spans_empty_span_selects_nothing():
    """A row whose span is empty must contribute no steered positions."""
    mask = mask_from_spans([(4, 4), (6, 2)], batch_size=2, seq_len=8)
    assert not mask.any(), "end <= start means the span is empty"


def test_mask_from_spans_rejects_wrong_batch_size():
    with pytest.raises(ValueError):
        mask_from_spans([(0, 2)], batch_size=3, seq_len=8)


# --------------------------------------------------------------------------------
# apply_masked_steering -- the core math
# --------------------------------------------------------------------------------

def test_unmasked_positions_are_bit_identical(hidden, direction):
    """The defense must touch only what it selected.

    If this fails, the blind and gated variants are steering tokens outside their
    own definition, and every comparison between variants is meaningless.
    """
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[0, 2:5] = True
    out = apply_masked_steering(hidden, mask, direction, -0.4)

    untouched = ~mask
    assert torch.equal(out[untouched], hidden[untouched]), (
        "positions where mask is False must be unchanged, exactly"
    )


def test_relative_change_equals_the_dose(hidden, direction):
    """||h' - h|| / ||h|| must equal |coeff| at every steered position.

    This is what makes the dose dimensionless and comparable across tokens. It
    only holds if you scale by EACH TOKEN'S OWN norm -- using a single global
    norm, or forgetting the norm entirely, breaks it. The fixture includes one
    token 50x larger and one 50x smaller than the rest so that shortcut fails
    loudly here.
    """
    mask = torch.ones(2, 8, dtype=torch.bool)
    for coeff in (-0.4, -0.1, 0.25):
        out = apply_masked_steering(hidden, mask, direction, coeff)
        rel = (out - hidden).norm(dim=-1) / hidden.norm(dim=-1)
        assert torch.allclose(rel, torch.full_like(rel, abs(coeff)), atol=1e-4), (
            f"expected every steered token to move by exactly {abs(coeff)} of its "
            f"own norm; got range [{rel.min():.4f}, {rel.max():.4f}]"
        )


def test_change_points_along_the_direction(hidden, direction):
    """The delta must be parallel to v̂ -- and antiparallel for a negative dose.

    A negative dose is the whole defense: steering AWAY from userness.
    """
    mask = torch.ones(2, 8, dtype=torch.bool)
    unit = direction / direction.norm()

    for coeff, expected_cos in ((0.3, 1.0), (-0.3, -1.0)):
        delta = apply_masked_steering(hidden, mask, direction, coeff) - hidden
        cos = torch.nn.functional.cosine_similarity(
            delta.reshape(-1, D), unit.expand(delta.reshape(-1, D).shape), dim=-1
        )
        assert torch.allclose(cos, torch.full_like(cos, expected_cos), atol=1e-4), (
            f"dose {coeff} should move tokens with cosine {expected_cos} to the "
            f"direction; got {cos.min():.4f}..{cos.max():.4f}"
        )


def test_direction_is_normalized_internally(hidden, direction):
    """Callers may pass a non-unit direction; the dose must not depend on its length."""
    mask = torch.ones(2, 8, dtype=torch.bool)
    from_unit = apply_masked_steering(hidden, mask, direction / direction.norm(), -0.4)
    from_scaled = apply_masked_steering(hidden, mask, direction * 37.0, -0.4)
    assert torch.allclose(from_unit, from_scaled, atol=1e-5), (
        "scaling the input direction must not change the result"
    )


def test_zero_dose_is_a_no_op(hidden, direction):
    """coeff=0 is the unsteered baseline arm and must be exactly the input."""
    mask = torch.ones(2, 8, dtype=torch.bool)
    out = apply_masked_steering(hidden, mask, direction, 0.0)
    assert torch.equal(out, hidden)


def test_empty_mask_is_a_no_op(hidden, direction):
    """A gate that flags nothing must leave the model completely alone."""
    mask = torch.zeros(2, 8, dtype=torch.bool)
    out = apply_masked_steering(hidden, mask, direction, -0.4)
    assert torch.equal(out, hidden)


def test_input_is_not_mutated(hidden, direction):
    """apply_masked_steering returns a new tensor and leaves its input alone."""
    before = hidden.clone()
    mask = torch.ones(2, 8, dtype=torch.bool)
    apply_masked_steering(hidden, mask, direction, -0.4)
    assert torch.equal(hidden, before), "the input tensor must not be modified in place"


def test_rows_are_independent(direction):
    """Row 0's mask must not affect row 1. Batching must not leak across rows."""
    h = torch.randn(2, 6, D)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[0, :] = True  # steer all of row 0, none of row 1

    out = apply_masked_steering(h, mask, direction, -0.4)
    assert not torch.equal(out[0], h[0]), "row 0 should have been steered"
    assert torch.equal(out[1], h[1]), "row 1 was not masked and must be untouched"


def test_dtype_is_preserved(direction):
    """The real model runs in bfloat16; the output must come back in the input dtype.

    Do the arithmetic in float32 internally anyway -- bfloat16 norms are too
    imprecise to hold the dose property.
    """
    h = torch.randn(2, 6, D).to(torch.bfloat16)
    mask = torch.ones(2, 6, dtype=torch.bool)
    out = apply_masked_steering(h, mask, direction, -0.4)
    assert out.dtype == torch.bfloat16


@pytest.mark.parametrize(
    "bad_mask, bad_direction",
    [
        (torch.ones(2, 9, dtype=torch.bool), None),   # wrong sequence length
        (torch.ones(3, 8, dtype=torch.bool), None),   # wrong batch size
        (None, torch.randn(D + 1)),                   # wrong residual width
    ],
)
def test_shape_mismatch_raises(hidden, direction, bad_mask, bad_direction):
    """Fail loudly on a misaligned mask instead of steering arbitrary tokens.

    This is the single most important defensive check in the file. A mask that no
    longer lines up with the sequence is an upstream bug; padding or trimming it
    to fit produces results that look fine and are wrong.
    """
    mask = bad_mask if bad_mask is not None else torch.ones(2, 8, dtype=torch.bool)
    vec = bad_direction if bad_direction is not None else direction
    with pytest.raises(ValueError):
        apply_masked_steering(hidden, mask, vec, -0.4)


# --------------------------------------------------------------------------------
# MaskedSteeringHook -- the plumbing
# --------------------------------------------------------------------------------

def test_hook_reproduces_the_reference_span_hook(direction):
    """The new mask-based hook must exactly match the base experiment's span hook.

    This is the regression test that ties the extension to known-good behavior:
    given a mask built from the same spans, the two must agree bit for bit. If
    they diverge, the oracle variant is no longer the paper's oracle variant and
    nothing downstream is comparable.
    """
    spans = [(2, 5), (1, 7)]
    base = torch.randn(2, 8, D)

    ref_out = base.clone()
    ReferenceSpanHook(direction, -0.4, spans)(None, None, ref_out)

    new_out = base.clone()
    mask = mask_from_spans(spans, batch_size=2, seq_len=8)
    MaskedSteeringHook(direction, -0.4, mask)(None, None, new_out)

    assert torch.allclose(ref_out, new_out, atol=1e-6), (
        "mask-based steering diverges from the base experiment's span-based hook"
    )


def test_hook_skips_generated_tokens(direction):
    """During generation the model runs one token at a time and the mask is
    meaningless there. The base experiment steers the prompt only; keep that."""
    decode_step = torch.randn(2, 1, D)
    before = decode_step.clone()
    mask = torch.ones(2, 8, dtype=torch.bool)   # prefill-shaped, deliberately

    out = MaskedSteeringHook(direction, -0.4, mask)(None, None, decode_step)

    assert torch.equal(out, before), (
        "a single-token (decode) pass must be returned untouched, and must NOT "
        "raise on the shape mismatch -- it is the expected case, not a bug"
    )


def test_hook_returns_a_tuple_when_given_a_tuple(direction):
    """Transformer blocks usually return a tuple; returning a bare tensor breaks
    the forward pass downstream."""
    h = torch.randn(2, 8, D)
    mask = torch.ones(2, 8, dtype=torch.bool)
    out = MaskedSteeringHook(direction, -0.4, mask)(None, None, (h, "extra"))

    assert isinstance(out, tuple), "given a tuple, the hook must return a tuple"
    assert out[1] == "extra", "the rest of the tuple must be passed through"


def test_hook_rejects_a_mask_that_does_not_fit(direction):
    """A prefill pass whose length disagrees with the mask is an upstream bug."""
    h = torch.randn(2, 12, D)                    # sequence grew
    mask = torch.ones(2, 8, dtype=torch.bool)    # mask did not
    with pytest.raises(ValueError):
        MaskedSteeringHook(direction, -0.4, mask)(None, None, h)


def test_hook_with_zero_dose_changes_nothing(direction):
    h = torch.randn(2, 8, D)
    before = h.clone()
    mask = torch.ones(2, 8, dtype=torch.bool)
    MaskedSteeringHook(direction, 0.0, mask)(None, None, h)
    assert torch.equal(h, before)
