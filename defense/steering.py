"""Masked steering: apply the userness direction to an arbitrary set of token
positions, rather than to one contiguous span.
"""

import torch

def mask_from_spans(spans, batch_size, seq_len, device=None):
    """Convert the old span representation into the new mask representation.

    Returns:
        (B, T) bool tensor. True at positions inside that row's span.
    """
    if len(spans) != batch_size: 
        raise ValueError(f"expected {batch_size} spans, got {len(spans)}")
    masks = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    for b in range(batch_size): 
        start, end = spans[b]
        if end > start: 
            masks[b, start:end] = True 
    return masks 

def apply_masked_steering(hidden, mask, direction, coeff):
    """Add `coeff * ||h|| * unit(direction)` to every position where mask is True.
    """
    if mask.shape != hidden.shape[:2]: 
        raise ValueError(f"Expected mask tensor of shape {hidden.shape[:2]}, got {mask.shape}")
    if hidden.shape[-1] != direction.shape[-1]: 
        raise ValueError(f"Expected d_user vector of shape {hidden.shape[-1]}, got {direction.shape}")
    if not mask.any() or coeff == 0.0: 
        return hidden.clone() 
    
    h = hidden.to(torch.float32)
    unit = direction / direction.norm() 
    norms = h.norm(dim=-1, keepdim=True)
    delta = norms * unit * coeff 
    steered = torch.where(mask.unsqueeze(-1), h + delta, h)
    steered = steered.to(hidden.dtype)
    return steered 

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
        hidden = output[0] if isinstance(output, tuple) else output

        # Decode steps and a zero dose are both expected no-ops. These are checked
        # BEFORE the mask shape, because a one-token decode pass legitimately does
        # not match a prefill-shaped mask.
        if hidden.shape[1] == 1 or self.coeff == 0.0:
            return output

        if tuple(hidden.shape[:2]) != tuple(self.mask.shape):
            raise ValueError(
                f"mask shape {tuple(self.mask.shape)} does not match prefill sequence "
                f"{tuple(hidden.shape[:2])}. The mask was built for a different "
                f"sequence; rebuild it rather than reshaping it here."
            )

        steered = apply_masked_steering(
            hidden, self.mask.to(hidden.device), self.direction, self.coeff)
        hidden.copy_(steered)
        return output

    def steered_token_count(self):
        """How many positions this hook will steer. Logged per step so a defense
        that silently fails to apply is visible in the results rather than
        indistinguishable from a weak defense."""
        return int(self.mask.sum()) if self.coeff != 0.0 else 0
