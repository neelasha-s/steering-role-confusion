"""Can the layer-14 role probe actually serve as a gate? Run this FIRST.

    python -m defense.gate_calibration

WHY THIS RUNS BEFORE ANYTHING ELSE
----------------------------------
The gated variant only makes sense if the probe can tell injected tokens apart
from ordinary fetched-page tokens. That is not a given. The probe was trained on
clean C4 text wrapped in role tags; here it is asked to score HTML markup, escaped
entities, and encyclopedia prose. This repo already documents the probe failing
out of distribution -- COT-FORGERY-STEERING.md records a *random* vector driving
its CoTness readout to 0.95.

So this script asks the detection question directly, in about 20 GPU-minutes,
before four hours go into a variant that cannot work:

  * If benign page tokens score above threshold at a high rate, the gate fires
    almost everywhere and "gated" is just "blind" with extra machinery.
  * If injected tokens score no higher than benign ones, the gate is not
    selecting for anything real, whatever its absolute numbers look like.

Either outcome is reportable. A dead gate found in 20 minutes is a result; a dead
gate found at hour eleven is a lost project.

Nothing is generated here -- this is a single forward pass per page. No model
output text is produced, printed, or saved.
"""

import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from defense import config, harness
from utils.loader import load_model_and_tokenizer

DEVICE = os.environ.get("DEVICE", "cuda:0")
# None lets the loader try its fallback chain (flash-attn3 -> sdpa -> eager)
# and keep the first that loads. Pin one with ATTN=eager if you need to.
ATTN = os.environ.get("ATTN") or None


def rms_normalize(x):
    """Per-token RMS normalization, matching how the probe was trained."""
    return x / x.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()


@torch.no_grad()
def user_probs_over_span(model, tokenizer, probe, probe_module, prompt, span):
    """P(user) for each token in `span`, read at the probe layer.

    Returns a 1-D numpy array, one probability per token in the span.
    """
    enc = tokenizer([prompt], add_special_tokens=False, return_tensors="pt").to(DEVICE)
    pad = enc["input_ids"].shape[1] - int(enc["attention_mask"].sum())

    captured = {}

    def grab(module, inputs, output):
        captured["h"] = (output[0] if isinstance(output, tuple) else output).detach()

    handle = probe_module.register_forward_hook(grab)
    try:
        model(**enc, use_cache=False)
    finally:
        handle.remove()

    start, end = span
    feats = rms_normalize(captured["h"][0, start + pad:end + pad, :].float()).cpu().numpy()
    user_idx = list(probe.classes_).index("user")
    return probe.predict_proba(feats)[:, user_idx]


def describe(name, probs):
    pct = np.percentile(probs, [5, 25, 50, 75, 95])
    print(f"  {name:<22} n={len(probs):>5}  mean={probs.mean():.3f}  "
          f"p5={pct[0]:.3f} p25={pct[1]:.3f} med={pct[2]:.3f} p75={pct[3]:.3f} p95={pct[4]:.3f}")


def main():
    tokenizer, model, _, _ = load_model_and_tokenizer(
        "gptoss-20b", device=DEVICE, attn_implementation=ATTN)
    probe = pickle.load(open(config.PROBE_PICKLE, "rb"))["clf"]
    probe_module = model.model.layers[config.PROBE_LAYER]

    carrier = harness.load_carrier()

    # Clean page: no injection anywhere. These tokens are what the gate must NOT fire on.
    clean_page = harness.build_page("", carrier)
    clean_prompt = harness.build_prompt(clean_page)
    clean_span = harness.tool_block_span(tokenizer, clean_prompt, clean_page)

    # Poisoned page: the user-declared injection, i.e. the surface the suppress arm
    # works on. These command tokens are what the gate SHOULD fire on.
    poisoned_page = harness.build_page(
        harness.wrap_declaration("user", harness.INJECTION), carrier)
    poisoned_prompt = harness.build_prompt(poisoned_page)
    poisoned_block = harness.tool_block_span(tokenizer, poisoned_prompt, poisoned_page)
    injection_span = harness.command_span(tokenizer, poisoned_prompt)

    print(f"\nprobe @ layer {config.PROBE_LAYER} | carrier frozen at "
          f"{config.CARRIER_CHARS} chars")
    print(f"clean tool block   : {clean_span[1] - clean_span[0]:>5} tokens")
    print(f"poisoned tool block: {poisoned_block[1] - poisoned_block[0]:>5} tokens")
    print(f"injection span     : {injection_span[1] - injection_span[0]:>5} tokens")

    benign = user_probs_over_span(model, tokenizer, probe, probe_module, clean_prompt, clean_span)
    injected = user_probs_over_span(model, tokenizer, probe, probe_module,
                                    poisoned_prompt, injection_span)

    # Benign tokens inside the POISONED block: everything except the injection.
    poisoned_all = user_probs_over_span(model, tokenizer, probe, probe_module,
                                        poisoned_prompt, poisoned_block)
    lo = injection_span[0] - poisoned_block[0]
    hi = injection_span[1] - poisoned_block[0]
    carrier_in_poisoned = np.concatenate([poisoned_all[:max(lo, 0)], poisoned_all[hi:]])

    print("\nP(user) distributions")
    describe("benign (clean page)", benign)
    describe("benign (in poisoned)", carrier_in_poisoned)
    describe("injected command", injected)

    print("\nGate behavior by threshold")
    print(f"  {'T':>5} | {'% benign fired':>14} | {'% injection fired':>17} | {'% of block steered':>19}")
    for t in config.THRESHOLD_GRID:
        fpr = float((benign > t).mean())
        tpr = float((injected > t).mean())
        block_frac = float((poisoned_all > t).mean())
        print(f"  {t:>5.2f} | {fpr*100:>13.1f}% | {tpr*100:>16.1f}% | {block_frac*100:>18.1f}%")

    # Rank-based separation: does the probe order injected above benign at all?
    try:
        from sklearn.metrics import roc_auc_score
        y = np.concatenate([np.zeros(len(benign)), np.ones(len(injected))])
        auc = roc_auc_score(y, np.concatenate([benign, injected]))
    except Exception:
        auc = float("nan")

    print(f"\nseparation AUC (injected vs benign): {auc:.3f}")
    print("  0.50 = the probe cannot distinguish them at any threshold")

    print("\nVERDICT")
    fired = [float((benign > t).mean()) for t in config.THRESHOLD_GRID]
    if auc < 0.65:
        print("  Gate is NOT viable. The probe does not separate injected from benign")
        print("  tokens, so the gated variant cannot be a targeted defense. Drop variant")
        print("  (c), report this calibration as the reason, and spend the time on the")
        print("  blind-vs-oracle tradeoff curve instead.")
    elif min(fired) > 0.5:
        print("  Gate fires on most benign tokens even at the highest threshold, so")
        print("  'gated' collapses into 'blind'. Report it as such rather than as a")
        print("  distinct variant.")
    else:
        best = config.THRESHOLD_GRID[int(np.argmin(fired))]
        print(f"  Gate is viable. Separation is real and thresholds discriminate;")
        print(f"  T={best} is the most selective on this grid. Proceed with variant (c),")
        print("  and carry these false-positive rates into the write-up -- they bound")
        print("  how much benign text the defense will touch.")
    print()


if __name__ == "__main__":
    main()
