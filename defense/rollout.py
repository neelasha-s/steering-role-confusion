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


def _row_pads(enc):
    """Left-padding per row: real tokens sit at the END of each padded row, so every
    unpadded span shifts right by this many positions. One value per row."""
    seq_len = enc["input_ids"].shape[1]
    return seq_len, (seq_len - enc["attention_mask"].sum(1)).tolist()


def _union_mask(span_lists, batch_size, seq_len, device):
    """(B, T) bool mask with EVERY span in each row's list set. Same bounds rule as
    mask_from_spans: a span outside the sequence raises rather than clamping."""
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    for row, spans in enumerate(span_lists):
        for start, end in spans:
            if start < 0 or end > seq_len:
                raise ValueError(f"span ({start}, {end}) for row {row} outside seq_len {seq_len}")
            if end > start:
                mask[row, start:end] = True
    return mask


class Defense:
    """Strategy object: given the current prompts, produce the steering mask for
    this step. Subclasses are the variants.

    `mask_for_batch` returns a (B, T) bool mask in PADDED coordinates, one row per
    prompt, or None for no steering. `mask_for` is the single-prompt convenience
    used by run_trajectory."""

    name = "none"
    coeff = 0.0

    def mask_for_batch(self, tokenizer, prompts, page, enc):
        return None

    def mask_for(self, tokenizer, prompt, page, enc, ctx):
        return self.mask_for_batch(tokenizer, [prompt], page, enc)

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

    def mask_for_batch(self, tokenizer, prompts, page, enc):
        # EVERY occurrence, not just the first: a refetched poisoned page puts the
        # injection in the transcript a second time, and a defense that only covers
        # the seed block leaves that copy unsteered.
        seq_len, pads = _row_pads(enc)
        span_lists = []
        for prompt, pad in zip(prompts, pads):
            if self.region == "command":
                spans = harness.find_all_spans(tokenizer, prompt, harness.INJECTION)
            else:
                spans = harness.all_tool_block_spans(tokenizer, prompt)
            span_lists.append([(s + pad, e + pad) for s, e in spans])
        return _union_mask(span_lists, len(prompts), seq_len, enc["input_ids"].device)


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

    def mask_for_batch(self, tokenizer, prompts, page, enc):
        from defense.gating import build_gate_mask
        seq_len, pads = _row_pads(enc)
        span_lists = []
        for prompt, pad in zip(prompts, pads):
            spans = harness.all_tool_block_spans(tokenizer, prompt)   # every tool result
            span_lists.append([(s + pad, e + pad) for s, e in spans])
        # The gate may only fire inside tool output -- all of it, including a
        # refetched page. One probe pass scores the whole batch; probe_scores already
        # excludes padding via attention_mask.
        region = _union_mask(span_lists, len(prompts), seq_len, enc["input_ids"].device)
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


@torch.no_grad()
def run_arm_batched(model, tokenizer, steer_module, defense, page, ids, n, *,
                    micro_batch=config.MICRO_BATCH, max_steps=6, max_act_tok=320,
                    temperature=1.0, gen_seed=config.SEED):
    """Run n trajectories for one arm, `micro_batch` at a time per generate() call.

    Same semantics as run_trajectory, batched the way the base notebook's cell 26
    was. At each step the live trajectories are chunked; each chunk is tokenized
    with LEFT padding, the defense builds one (B, T) mask for the chunk with each
    row's span shifted by that row's own padding, and one hook steers the batch.

    Seeding is per (step, chunk) and identical across arms, so arms stay
    comparable -- but results are not bit-identical to the unbatched loop.

    Returns a list of n record dicts (same shape as run_trajectory).
    """
    ID_CALL, ID_RETURN, PAD = ids
    spotlight = defense.spotlight()
    coeff = getattr(defense, "coeff", 0.0)

    messages = [harness.build_seed_messages(page, spotlight=spotlight) for _ in range(n)]
    turns = [harness.build_seed_turns(page, spotlight=spotlight) for _ in range(n)]
    finished, outcome = [False] * n, ["MAX_STEPS"] * n
    attacked, n_steps = [0] * n, [0] * n
    steered = [[] for _ in range(n)]
    last_text = [""] * n

    for step in range(max_steps):
        live = [i for i in range(n) if not finished[i]]
        if not live:
            break

        for ci, b0 in enumerate(range(0, len(live), micro_batch)):
            chunk = live[b0:b0 + micro_batch]
            prompts = [harness.render_conversation(messages[i]) + "<|start|>assistant" for i in chunk]
            enc = tokenizer(prompts, add_special_tokens=False, return_tensors="pt",
                            padding=True).to(model.device)
            seq_len = enc["input_ids"].shape[1]

            mask = defense.mask_for_batch(tokenizer, prompts, page, enc)
            handle, counts = None, [0] * len(chunk)
            if mask is not None and coeff != 0.0:
                hook = MaskedSteeringHook(getattr(defense, "direction"), coeff, mask)
                counts = mask.sum(dim=1).tolist()          # intervention accounting, per row
                handle = steer_module.register_forward_hook(hook)

            torch.manual_seed(gen_seed + step * 1000 + ci)  # same schedule in every arm
            try:
                generated = model.generate(
                    **enc, max_new_tokens=max_act_tok,
                    do_sample=temperature > 0, temperature=temperature,
                    eos_token_id=[ID_CALL, ID_RETURN], pad_token_id=PAD)
            finally:
                if handle is not None:
                    handle.remove()

            for k, i in enumerate(chunk):
                steered[i].append(int(counts[k]))
                new_ids = generated[k][seq_len:].tolist()
                while new_ids and new_ids[-1] == PAD:
                    new_ids.pop()
                stop = new_ids[-1] if new_ids and new_ids[-1] in (ID_CALL, ID_RETURN) else None
                if stop is not None:
                    new_ids = new_ids[:-1]
                text = tokenizer.decode(new_ids, skip_special_tokens=False)
                last_text[i] = text
                command, is_final = harness.parse_action("<|start|>assistant" + text)
                n_steps[i] += 1

                if command and harness.is_exfil(command):
                    attacked[i], outcome[i], finished[i] = 1, "ATTACK_SUCCESS", True
                    turns[i].append({"kind": "assistant", "cmd": command, "text": text})
                elif stop == ID_RETURN or (is_final and not command):
                    outcome[i], finished[i] = "FINISHED_NO_EXFIL", True
                    turns[i].append({"kind": "assistant", "cmd": command, "text": text})
                elif not command:
                    outcome[i], finished[i] = "STUCK_NO_TOOLCALL", True
                    turns[i].append({"kind": "assistant", "cmd": command, "text": text})
                else:
                    tool_result = harness.mock_shell(command, page)
                    messages[i].append("<|start|>assistant" + text + "<|call|>")
                    messages[i].append(harness.harmony_tool_result(tool_result))
                    turns[i].append({"kind": "assistant", "cmd": command, "text": text})
                    turns[i].append({"kind": "tool_result", "text": tool_result})

            del enc, generated
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return [{
        "attack": attacked[i], "outcome": outcome[i], "n_steps": n_steps[i],
        "steered_tokens": steered[i], "degenerate": metrics.is_degenerate(last_text[i]),
        "turns": turns[i],
    } for i in range(n)]
