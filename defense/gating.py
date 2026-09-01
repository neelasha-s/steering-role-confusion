"""The probe-gated defense: use the layer-14 role probe to decide WHICH tokens
of the tool block get steered.

THE ORDERING PROBLEM, AND WHY THERE ARE TWO PASSES
--------------------------------------------------
We steer at layer 11 and read the probe at layer 14. Inside a single forward
pass, layer 11 is computed BEFORE layer 14 -- so there is no way to consult the
probe and then go back and modify an earlier layer in the same pass. The gate
therefore needs two passes:

    pass 1 (measure)    run the sequence with NO steering, capture the layer-14
                        residuals, score every token with the probe, threshold
                        the scores into a mask.
    pass 2 (intervene)  run again with MaskedSteeringHook at layer 11, applying
                        only to the positions that mask selected.

Pass 1 is deliberately unsteered. That matches what a real defender sees at
deploy time: the untouched text, before any intervention.

WHY THE GATE IS SCOPED TO A REGION
----------------------------------
`build_gate_mask` takes a `region_mask` -- normally the tool-result block -- and
never fires outside it. This is a correctness requirement, not an optimization.
Tokens in the genuine user turn legitimately read as user-like, because they ARE
user text. A gate allowed to roam the whole prompt would flag them and steer the
real user's instruction away from userness, which is not a defense against
injection -- it is an attack on the task.

WHAT THIS CANNOT DO
-------------------
The gate inherits every weakness of the probe. `gate_calibration.py` exists to
measure that before this code is trusted: if benign page tokens score above
threshold at a high rate, this gate degenerates into blind steering, and the
right move is to report that and drop the variant.
"""

import numpy as np
import torch


def rms_normalize(x):
    """Per-token RMS normalization, matching how the probe was trained."""
    return x / x.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()


def scores_to_mask(user_probs, threshold, restrict_to=None):
    """Threshold per-token probe scores into a steering mask.

    Args:
        user_probs:  (B, T) float tensor of P(user) per token.
        threshold:   fire where the score is STRICTLY greater than this.
        restrict_to: optional (B, T) bool tensor. The gate may only fire inside
                     it -- see the region note in the module docstring.

    Returns:
        (B, T) bool tensor suitable for MaskedSteeringHook.

    Raises:
        ValueError: if restrict_to's shape does not match user_probs.
    """
    mask = user_probs > threshold
    if restrict_to is not None:
        if tuple(restrict_to.shape) != tuple(mask.shape):
            raise ValueError(
                f"restrict_to shape {tuple(restrict_to.shape)} does not match "
                f"scores {tuple(mask.shape)}")
        mask = mask & restrict_to.to(mask.device)
    return mask


@torch.no_grad()
def probe_scores(model, probe, probe_module, input_ids, attention_mask,
                 region_mask, user_class="user"):
    """Pass 1: score every token inside `region_mask` with the role probe.

    Runs one UNSTEERED forward pass, captures the probe layer, and returns
    P(user) per token. Positions outside the region are returned as 0.0 -- they
    are never scored, so they can never fire.

    Returns:
        (B, T) float tensor of P(user), zero outside the region.
    """
    captured = {}

    def grab(module, inputs, output):
        captured["h"] = (output[0] if isinstance(output, tuple) else output).detach()

    handle = probe_module.register_forward_hook(grab)
    try:
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        handle.remove()

    resid = captured["h"]
    scores = torch.zeros(resid.shape[:2], dtype=torch.float32, device=resid.device)

    # Never score padding: a pad token has no content and its residual is
    # meaningless, but the probe will still return a confident-looking number.
    region = region_mask.to(resid.device).bool() & attention_mask.to(resid.device).bool()
    rows, cols = region.nonzero(as_tuple=True)
    if rows.numel() == 0:
        return scores

    feats = rms_normalize(resid[rows, cols, :].float()).cpu().numpy()
    user_idx = list(probe.classes_).index(user_class)
    probs = probe.predict_proba(feats)[:, user_idx]
    scores[rows, cols] = torch.from_numpy(np.asarray(probs, dtype=np.float32)).to(resid.device)
    return scores


@torch.no_grad()
def build_gate_mask(model, probe, probe_module, input_ids, attention_mask,
                    region_mask, threshold, user_class="user"):
    """The full two-pass gate's first half: measure, then threshold.

    Returns the (B, T) bool mask to hand to MaskedSteeringHook. The caller runs
    pass 2 by registering that hook and generating.

    Must be recomputed at EVERY step of the agent loop. The transcript grows as
    tool results are appended, and the poisoned page can re-enter on a refetch --
    a mask built at step 1 and reused at step 3 defends neither the new tokens
    nor the right positions.
    """
    scores = probe_scores(model, probe, probe_module, input_ids, attention_mask,
                          region_mask, user_class=user_class)
    return scores_to_mask(scores, threshold, restrict_to=region_mask.to(scores.device))


def deletion_mask_to_keep(mask, attention_mask=None):
    """For the DELETION variant: positions to KEEP, i.e. the inverse of the gate.

    The deletion variant removes flagged tokens outright instead of steering
    them, then runs unsteered. If deletion matches steering on attack success,
    the steering vector is doing nothing a blunt classifier could not, and the
    write-up should say so.
    """
    keep = ~mask.bool()
    if attention_mask is not None:
        keep = keep & attention_mask.to(keep.device).bool()
    return keep
