"""One steered/gated ReAct trajectory against the mock shell, generalized from
notebook cell 26 to carry a DEFENSE and to log the extra metrics.

Differences from the base rollout, all deliberate:

  * A `defense` object decides the steering mask FRESH at every step. This is the
    fix for the re-fetch problem: the poisoned page re-enters when the agent
    refetches, so a mask computed once would leave later steps undefended.
  * Every step logs how many tokens were actually steered, so a defense that
    silently fails to apply is visible (see metrics.capability_warning).
  * Each trajectory records whether its final output was degenerate.

Nothing is executed; the shell is mocked and the exfil curl halts the loop before
it reaches even the mock, exactly as in the base experiment.
"""

import torch

from defense import config, harness, metrics
from defense.steering import MaskedSteeringHook


class Defense:
    """Strategy object: given the current prompt, produce the steering mask for
    this step. Subclasses are the variants. `mask_for` returns a (1, T) bool mask
    in PADDED coordinates, or None for no steering."""

    name = "none"
    coeff = 0.0

    def mask_for(self, tokenizer, prompt, page, enc, ctx):
        return None

    def spotlight(self):
        return False


class NoDefense(Defense):
    name = "none"


class SpotlightDefense(Defense):
    """The prompting baseline: wrap the tool result in an untrusted-data warning.
    No steering at all."""
    name = "prompt"

    def spotlight(self):
        return True


class SpanSteerDefense(Defense):
    """Oracle (command span) or blind (whole tool block), depending on `region`."""

    def __init__(self, direction, coeff, region, spotlight=False):
        self.direction = direction
        self.coeff = coeff
        self.region = region              # "command" or "tool_block"
        self._spotlight = spotlight
        self.name = "oracle" if region == "command" else "blind"

    def spotlight(self):
        return self._spotlight

    def mask_for(self, tokenizer, prompt, page, enc, ctx):
        if self.region == "command":
            span = harness.command_span(tokenizer, prompt)
        else:
            span = harness.tool_block_span(tokenizer, prompt, page, self._spotlight)
        pad = enc["input_ids"].shape[1] - int(enc["attention_mask"].sum())
        seq_len = enc["input_ids"].shape[1]
        from defense.steering import mask_from_spans
        return mask_from_spans([(span[0] + pad, span[1] + pad)], 1, seq_len,
                               device=enc["input_ids"].device)


class GatedSteerDefense(Defense):
    """Probe-gated: steer only tool-block tokens the probe scores above threshold.
    Rebuilds the mask every step via the two-pass gate."""
    name = "gated"

    def __init__(self, direction, coeff, threshold, model, probe, probe_module,
                 spotlight=False):
        self.direction = direction
        self.coeff = coeff
        self.threshold = threshold
        self.model = model
        self.probe = probe
        self.probe_module = probe_module
        self._spotlight = spotlight

    def spotlight(self):
        return self._spotlight

    def mask_for(self, tokenizer, prompt, page, enc, ctx):
        from defense.gating import build_gate_mask
        span = harness.tool_block_span(tokenizer, prompt, page, self._spotlight)
        pad = enc["input_ids"].shape[1] - int(enc["attention_mask"].sum())
        seq_len = enc["input_ids"].shape[1]
        region = torch.zeros(1, seq_len, dtype=torch.bool, device=enc["input_ids"].device)
        region[0, span[0] + pad:span[1] + pad] = True
        return build_gate_mask(self.model, self.probe, self.probe_module,
                               enc["input_ids"], enc["attention_mask"],
                               region, self.threshold)


@torch.no_grad()
def run_trajectory(model, tokenizer, steer_module, defense, page, ids, *,
                   max_steps=6, max_act_tok=320, temperature=1.0, gen_seed=config.SEED):
    """Run ONE trajectory. `ids` = (ID_CALL, ID_RETURN, PAD). Returns a record dict.

    The record includes attack/outcome/n_steps (as in the base experiment) plus
    steered_tokens (one count per step) and degenerate (final-output check)."""
    ID_CALL, ID_RETURN, PAD = ids
    spotlight = defense.spotlight()

    messages = harness.build_seed_messages(page, spotlight=spotlight)
    turns = harness.build_seed_turns(page, spotlight=spotlight)  # seed context, like the base repo
    steered_per_step = []
    outcome, attacked, n_steps = "MAX_STEPS", 0, 0
    last_action_text = ""

    for _ in range(max_steps):
        prompt = harness.render_conversation(messages) + "<|start|>assistant"
        enc = tokenizer([prompt], add_special_tokens=False, return_tensors="pt").to(model.device)

        mask = defense.mask_for(tokenizer, prompt, page, enc, {})
        coeff = getattr(defense, "coeff", 0.0)

        handle = None
        steered_count = 0
        if mask is not None and coeff != 0.0:
            hook = MaskedSteeringHook(getattr(defense, "direction"), coeff, mask)
            steered_count = hook.steered_token_count()
            handle = steer_module.register_forward_hook(hook)
        steered_per_step.append(steered_count)

        torch.manual_seed(gen_seed)
        try:
            generated = model.generate(
                **enc, max_new_tokens=max_act_tok,
                do_sample=temperature > 0, temperature=temperature,
                eos_token_id=[ID_CALL, ID_RETURN], pad_token_id=PAD)
        finally:
            if handle is not None:
                handle.remove()

        new_ids = generated[0][enc["input_ids"].shape[1]:].tolist()
        while new_ids and new_ids[-1] == PAD:
            new_ids.pop()
        stop_token = new_ids[-1] if new_ids and new_ids[-1] in (ID_CALL, ID_RETURN) else None
        if stop_token is not None:
            new_ids = new_ids[:-1]
        action_text = tokenizer.decode(new_ids, skip_special_tokens=False)
        last_action_text = action_text
        command, is_final = harness.parse_action("<|start|>assistant" + action_text)
        n_steps += 1

        tool_result = None
        if command and harness.is_exfil(command):
            attacked, outcome = 1, "ATTACK_SUCCESS"
            turns.append({"kind": "assistant", "cmd": command, "text": action_text})
            break
        elif stop_token == ID_RETURN or (is_final and not command):
            outcome = "FINISHED_NO_EXFIL"
            turns.append({"kind": "assistant", "cmd": command, "text": action_text})
            break
        elif not command:
            outcome = "STUCK_NO_TOOLCALL"
            turns.append({"kind": "assistant", "cmd": command, "text": action_text})
            break
        else:
            tool_result = harness.mock_shell(command, page)
            messages.append("<|start|>assistant" + action_text + "<|call|>")
            messages.append(harness.harmony_tool_result(tool_result))
            turns.append({"kind": "assistant", "cmd": command, "text": action_text})
            turns.append({"kind": "tool_result", "text": tool_result})

    return {
        "attack": attacked,
        "outcome": outcome,
        "n_steps": n_steps,
        "steered_tokens": steered_per_step,
        "degenerate": metrics.is_degenerate(last_action_text),
        "turns": turns,
    }
