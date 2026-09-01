"""CPU tests for the probe gate. No GPU and no real model -- the model and probe
are replaced by tiny fakes so the gate's logic can be checked in isolation.

    .venv/bin/python -m pytest defense/tests/test_gating.py -v
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from defense.gating import (build_gate_mask, deletion_mask_to_keep, probe_scores,
                            scores_to_mask)


# --------------------------------------------------------------------------------
# Fakes: a "model" whose probe layer emits residuals we choose, and a "probe" whose
# P(user) is 1.0 when the first component is positive and 0.0 otherwise. That lets
# a test say exactly which tokens should look user-like.
# --------------------------------------------------------------------------------

class FakeBlock(torch.nn.Module):
    def __init__(self, resid):
        super().__init__()
        self.resid = resid

    def forward(self, x):
        return self.resid


class FakeModel:
    def __init__(self, block):
        self.block = block

    def __call__(self, input_ids=None, attention_mask=None, use_cache=None):
        return self.block(input_ids)


class FakeProbe:
    classes_ = ["assistant", "cot", "system", "tool", "user"]

    def predict_proba(self, X):
        p_user = (X[:, 0] > 0).astype(np.float64)
        out = np.zeros((len(X), 5))
        out[:, 4] = p_user
        out[:, 3] = 1.0 - p_user
        return out


def make_setup(resid):
    block = FakeBlock(resid)
    return FakeModel(block), FakeProbe(), block


# --------------------------------------------------------------------------------
# scores_to_mask
# --------------------------------------------------------------------------------

def test_threshold_is_strictly_greater():
    """A score exactly equal to the threshold must NOT fire. Thresholds are swept
    over a grid, so an off-by-one-boundary rule shifts every point on the curve."""
    scores = torch.tensor([[0.4, 0.5, 0.6]])
    assert scores_to_mask(scores, 0.5).tolist() == [[False, False, True]]


def test_restrict_to_confines_the_gate():
    """The gate must never fire outside its region -- tokens in the real user turn
    legitimately read as user-like, and steering those attacks the task itself."""
    scores = torch.tensor([[0.9, 0.9, 0.9, 0.9]])
    region = torch.tensor([[False, False, True, True]])
    assert scores_to_mask(scores, 0.5, region).tolist() == [[False, False, True, True]]


def test_restrict_to_shape_mismatch_raises():
    with pytest.raises(ValueError):
        scores_to_mask(torch.zeros(1, 4), 0.5, torch.ones(1, 5, dtype=torch.bool))


def test_no_scores_above_threshold_gives_empty_mask():
    """A gate that flags nothing is a legitimate outcome -- it means the defense
    does nothing on this input, which the steering hook handles as a no-op."""
    assert not scores_to_mask(torch.full((2, 5), 0.1), 0.5).any()


# --------------------------------------------------------------------------------
# probe_scores
# --------------------------------------------------------------------------------

def test_scores_are_zero_outside_the_region():
    """Positions outside the region are never scored, so they can never fire."""
    resid = torch.ones(1, 4, 3)               # every token would read as user
    model, probe, block = make_setup(resid)
    region = torch.tensor([[False, True, True, False]])
    attn = torch.ones(1, 4, dtype=torch.long)

    scores = probe_scores(model, probe, block, torch.zeros(1, 4, dtype=torch.long), attn, region)
    assert scores.tolist() == [[0.0, 1.0, 1.0, 0.0]]


def test_padding_is_never_scored():
    """A pad token's residual is meaningless, but the probe still returns a
    confident number for it. Scoring padding would let the gate fire on nothing."""
    resid = torch.ones(1, 4, 3)
    model, probe, block = make_setup(resid)
    region = torch.ones(1, 4, dtype=torch.bool)
    attn = torch.tensor([[0, 0, 1, 1]])       # first two are left-padding

    scores = probe_scores(model, probe, block, torch.zeros(1, 4, dtype=torch.long), attn, region)
    assert scores.tolist() == [[0.0, 0.0, 1.0, 1.0]]


def test_empty_region_returns_all_zeros():
    resid = torch.ones(1, 3, 3)
    model, probe, block = make_setup(resid)
    scores = probe_scores(model, probe, block, torch.zeros(1, 3, dtype=torch.long),
                          torch.ones(1, 3, dtype=torch.long),
                          torch.zeros(1, 3, dtype=torch.bool))
    assert not scores.any()


def test_the_measuring_pass_leaves_no_hook_behind():
    """probe_scores registers a capture hook and must always remove it. A leaked
    hook silently contaminates every later forward pass in the process."""
    resid = torch.ones(1, 3, 3)
    model, probe, block = make_setup(resid)
    probe_scores(model, probe, block, torch.zeros(1, 3, dtype=torch.long),
                 torch.ones(1, 3, dtype=torch.long), torch.ones(1, 3, dtype=torch.bool))
    assert len(block._forward_hooks) == 0


# --------------------------------------------------------------------------------
# build_gate_mask
# --------------------------------------------------------------------------------

def test_gate_fires_only_on_user_like_tokens_inside_the_region():
    """End to end: positive first component = user-like under the fake probe."""
    resid = torch.zeros(1, 5, 3)
    resid[0, 1, 0] = 1.0     # user-like, inside region  -> fires
    resid[0, 2, 0] = -1.0    # not user-like             -> does not fire
    resid[0, 4, 0] = 1.0     # user-like, OUTSIDE region -> must not fire
    model, probe, block = make_setup(resid)
    region = torch.tensor([[True, True, True, True, False]])

    mask = build_gate_mask(model, probe, block, torch.zeros(1, 5, dtype=torch.long),
                           torch.ones(1, 5, dtype=torch.long), region, threshold=0.5)
    assert mask.tolist() == [[False, True, False, False, False]]


def test_gate_output_is_a_bool_mask_the_steering_hook_accepts():
    """The gate's output feeds straight into MaskedSteeringHook, which requires a
    (B, T) bool tensor matching the sequence."""
    from defense.steering import apply_masked_steering

    resid = torch.zeros(2, 4, 3)
    resid[:, 1, 0] = 1.0
    model, probe, block = make_setup(resid)
    region = torch.ones(2, 4, dtype=torch.bool)

    mask = build_gate_mask(model, probe, block, torch.zeros(2, 4, dtype=torch.long),
                           torch.ones(2, 4, dtype=torch.long), region, threshold=0.5)
    assert mask.dtype == torch.bool and tuple(mask.shape) == (2, 4)

    hidden = torch.randn(2, 4, 3)
    out = apply_masked_steering(hidden, mask, torch.randn(3), -0.4)
    assert torch.equal(out[~mask], hidden[~mask])


# --------------------------------------------------------------------------------
# deletion variant
# --------------------------------------------------------------------------------

def test_deletion_keeps_the_complement_of_the_gate():
    mask = torch.tensor([[True, False, True, False]])
    attn = torch.tensor([[0, 1, 1, 1]])
    assert deletion_mask_to_keep(mask).tolist() == [[False, True, False, True]]
    assert deletion_mask_to_keep(mask, attn).tolist() == [[False, True, False, True]]
