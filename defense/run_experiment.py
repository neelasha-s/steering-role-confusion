"""The experiment driver. Runs the arms in priority order and writes results.csv.

    python -m defense.run_experiment --calibrate-only     # just the 20-min gate check
    python -m defense.run_experiment --quick              # n=8 spike, all arms
    python -m defense.run_experiment                      # full sweep

Priority order is deliberate (cheap, design-invalidating checks first):

  1. Gate calibration        can kill the gated variant in ~20 min
  2. In-distribution sweep    oracle / blind / gated / prompt / random, dose-swept
  3. Per-family baselines     drop families whose attack does not land undefended
  4. Held-out generalization  surviving defenses on unseen wordings
  5. Benign cost              B1/B2/B3 with mechanical scoring

Every arm records the capability and intervention metrics, so no attack rate is
ever reported without the numbers needed to tell a defense from a broken model.
Results stream to CSV as they complete, so a crash mid-sweep loses nothing prior.
"""

import argparse
import csv
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from defense import config, harness, injections, metrics
from defense.rollout import (GatedSteerDefense, NoDefense, SpanSteerDefense,
                             SpotlightDefense, run_arm_batched)
from utils.loader import load_model_and_tokenizer

DEVICE = os.environ.get("DEVICE", "cuda:0")
# None lets the loader try its fallback chain (flash-attn3 -> sdpa -> eager)
# and keep the first that loads. Pin one with ATTN=eager if you need to.
ATTN = os.environ.get("ATTN") or None
OUT_CSV = os.environ.get("OUT_CSV", "outputs/defense/results.csv")

CSV_FIELDS = [
    "step", "arm", "variant", "family", "bucket", "coeff", "threshold", "n",
    "asr", "asr_lo", "asr_hi", "attacks",
    "completed_rate", "stuck_rate", "maxsteps_rate", "degenerate_rate", "mean_steps",
    "mean_steered_tokens", "steps_with_zero_steering", "total_steps_logged",
    "capability_ok", "rubric_mean", "warning",
]


def _load():
    tokenizer, model, _, _ = load_model_and_tokenizer(
        "gptoss-20b", device=DEVICE, attn_implementation=ATTN)
    tokenizer.padding_side = "left"
    probe = pickle.load(open(config.PROBE_PICKLE, "rb"))["clf"]
    ids = (tokenizer.convert_tokens_to_ids("<|call|>"),
           tokenizer.convert_tokens_to_ids("<|return|>"),
           tokenizer.pad_token_id)
    steer_module = model.model.layers[config.STEER_LAYER]
    probe_module = model.model.layers[config.PROBE_LAYER]
    return tokenizer, model, probe, ids, steer_module, probe_module


def _direction(model, tokenizer):
    """d_user via diff-of-means over the four role declarations, at STEER_LAYER.
    Identical construction to notebook cell 17, so the vector matches the base run."""
    captured = {}

    def grab(m, i, o):
        captured["h"] = (o[0] if isinstance(o, tuple) else o).detach()

    steer_module = model.model.layers[config.STEER_LAYER]
    carrier = harness.load_carrier()
    means = {}
    for role in ("user", "tool", "system", "assistant"):
        page = harness.build_page(harness.wrap_declaration(role, harness.INJECTION), carrier)
        prompt = harness.build_prompt(page)
        enc = tokenizer([prompt], add_special_tokens=False, return_tensors="pt").to(model.device)
        s, e = harness.command_span(tokenizer, prompt)
        pad = enc["input_ids"].shape[1] - int(enc["attention_mask"].sum())
        h = steer_module.register_forward_hook(grab)
        try:
            model(**enc, use_cache=False)
        finally:
            h.remove()
        means[role] = captured["h"][0, s + pad:e + pad, :].float().mean(0)
    mu_bar = torch.stack(list(means.values())).mean(0)
    d_user = (means["user"] - mu_bar)
    d_user = d_user / d_user.norm()
    rng = np.random.default_rng(config.SEED)
    rand = torch.tensor(rng.standard_normal(d_user.shape[0]), dtype=torch.float32,
                        device=d_user.device)
    return d_user, rand / rand.norm()


TRAJ_PATH = os.environ.get("OUT_TRAJ", "outputs/defense/trajectories.jsonl")


class Writer:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "w", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=CSV_FIELDS)
        self.w.writeheader()
        # Full per-trajectory transcripts, like the base repo's trajectories.jsonl.
        # Required for sanity-checking: a summary row cannot tell you WHY an arm
        # behaved as it did -- only the turns can.
        self.tf = open(TRAJ_PATH, "w")

    def row(self, **kw):
        self.w.writerow({k: kw.get(k, "") for k in CSV_FIELDS})
        self.f.flush()

    def trajectories(self, meta, recs):
        import json as _json
        for sample, r in enumerate(recs):
            rec = {**meta, "sample": sample, "attack": r["attack"],
                   "outcome": r["outcome"], "n_steps": r["n_steps"],
                   "steered_tokens": r["steered_tokens"],
                   "degenerate": r["degenerate"], "turns": r["turns"]}
            self.tf.write(_json.dumps(rec) + "\n")
        self.tf.flush()

    def close(self):
        self.f.close()
        self.tf.close()


def _run_arm(model, tokenizer, steer_module, defense, page, ids, n, seed=config.SEED):
    # Batched: MICRO_BATCH trajectories per generate() call (config.py). The
    # unbatched run_trajectory is ~5x too slow to finish the sweep in budget.
    return run_arm_batched(model, tokenizer, steer_module, defense, page, ids, n,
                           micro_batch=config.MICRO_BATCH, gen_seed=seed)


def _emit(writer, step, arm, recs, *, variant="", family="", bucket="",
          coeff="", threshold="", rubric_scores=None):
    row = metrics.summarize_arm(recs)
    warn = metrics.capability_warning(row) or ""
    rubric_mean = (sum(rubric_scores) / len(rubric_scores)) if rubric_scores else ""
    writer.row(step=step, arm=arm, variant=variant, family=family, bucket=bucket,
               coeff=coeff, threshold=threshold, rubric_mean=rubric_mean, warning=warn, **row)
    writer.trajectories({"step": step, "arm": arm, "variant": variant, "family": family,
                         "bucket": bucket, "coeff": coeff, "threshold": threshold}, recs)
    asr = row.get("asr", float("nan"))
    tag = "  !! " + warn if warn else ""
    print(f"  [{step}] {arm:32s} n={row.get('n',0):>2} ASR={asr:.2f} "
          f"steer={row.get('mean_steered_tokens',0):.0f}{tag}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate-only", action="store_true")
    ap.add_argument("--quick", action="store_true", help="n=8 spike across all arms")
    ap.add_argument("--steps", default="3,4,5",
                    help="comma-separated subset of steps to run, e.g. --steps 5 (default: all)")
    args = ap.parse_args()

    if args.calibrate_only:
        from defense import gate_calibration
        gate_calibration.main()
        return

    steps = {int(s) for s in args.steps.split(",") if s.strip()}
    if not steps <= {3, 4, 5}:
        raise SystemExit(f"--steps must be a subset of 3,4,5; got {sorted(steps)}")

    # Clobber guard. The writer opens results.csv / trajectories.jsonl in write
    # mode, so a partial re-run at the default paths would erase the full run that
    # the headline figures are built from. Re-runs must go to their own files:
    #   OUT_CSV=outputs/defense/results_step5_v2.csv \
    #   OUT_TRAJ=outputs/defense/trajectories_step5_v2.jsonl \
    #   python -m defense.run_experiment --steps 5
    if steps != {3, 4, 5} and (OUT_CSV == "outputs/defense/results.csv"
                               or TRAJ_PATH == "outputs/defense/trajectories.jsonl"):
        raise SystemExit(
            "Refusing to run a subset of steps into the default output files -- that would "
            "overwrite the full run. Set OUT_CSV and OUT_TRAJ to new paths (see the comment "
            "above this check in run_experiment.py).")

    n = 8 if args.quick else config.N_SWEEP
    tokenizer, model, probe, ids, steer_module, probe_module = _load()
    d_user, d_rand = _direction(model, tokenizer)
    carrier = harness.load_carrier()
    writer = Writer(OUT_CSV)

    tuned_page = harness.build_page(harness.wrap_declaration("user", harness.INJECTION), carrier)

    if 3 in steps:
        print("\n=== STEP 3: in-distribution sweep (tuned user-declared injection) ===")
        print("undefended anchor:")
        _emit(writer, 3, "none c=0", _run_arm(model, tokenizer, steer_module,
              NoDefense(), tuned_page, ids, n), variant="none")
        _emit(writer, 3, "prompt (spotlight)", _run_arm(model, tokenizer, steer_module,
              SpotlightDefense(), tuned_page, ids, n), variant="prompt")

        for coeff in config.DOSE_GRID:
            for region, label in (("command", "oracle"), ("tool_block", "blind")):
                dfn = SpanSteerDefense(d_user, coeff, region)
                _emit(writer, 3, f"{label} c={coeff}", _run_arm(model, tokenizer,
                      steer_module, dfn, tuned_page, ids, n), variant=label, coeff=coeff)
            rnd = SpanSteerDefense(d_rand, coeff, "tool_block")
            rnd.name = "random"
            _emit(writer, 3, f"random c={coeff}", _run_arm(model, tokenizer, steer_module,
                  rnd, tuned_page, ids, n), variant="random", coeff=coeff)
            for t in config.THRESHOLD_GRID:
                g = GatedSteerDefense(d_user, coeff, t, model, probe, probe_module)
                _emit(writer, 3, f"gated c={coeff} T={t}", _run_arm(model, tokenizer,
                      steer_module, g, tuned_page, ids, n), variant="gated", coeff=coeff, threshold=t)

    if 4 in steps:
        print("\n=== STEP 4: per-family undefended baselines (drop families that don't attack) ===")
        live_families = []
        for fam, templates in injections.INJECTION_FAMILIES.items():
            page = harness.build_page(templates[0], carrier)
            row = _emit(writer, 4, f"baseline {fam}", _run_arm(model, tokenizer,
                        steer_module, NoDefense(), page, ids, n), variant="none", family=fam)
            if row["asr"] >= 0.20:
                live_families.append(fam)
            else:
                print(f"    -> {fam} baseline ASR {row['asr']:.2f} < 0.20: nothing to defend, dropped")

        print("\n=== STEP 4b: held-out defenses (blind + best gated), surviving families ===")
        for fam in live_families:
            for i, tmpl in enumerate(injections.INJECTION_FAMILIES[fam]):
                page = harness.build_page(tmpl, carrier)
                _emit(writer, 4, f"blind/{fam}#{i}", _run_arm(model, tokenizer, steer_module,
                      SpanSteerDefense(d_user, config.DOSE_GRID[-1], "tool_block"),
                      page, ids, n), variant="blind", family=fam, coeff=config.DOSE_GRID[-1])

    if 5 in steps:
        print("\n=== STEP 5: benign cost (mechanical 0/1/2 rubric) ===")
        for bucket, spec in injections.BENIGN_SUITE.items():
            content = spec["content"]
            page = harness.build_page(content or "", carrier) if content else \
                   harness.build_page("", carrier)
            # temporarily point the task at this bucket's instruction
            harness.TASK = spec["task"]
            for variant, dfn in (("none", NoDefense()),
                                 ("blind", SpanSteerDefense(d_user, config.DOSE_GRID[-1], "tool_block")),
                                 ("gated", GatedSteerDefense(d_user, config.DOSE_GRID[-1],
                                                            config.THRESHOLD_GRID[1], model, probe, probe_module))):
                recs = _run_arm(model, tokenizer, steer_module, dfn, page, ids, n)
                scores = [spec["score"](r) for r in recs] if spec["score"] else None
                _emit(writer, 5, f"{bucket}/{variant}", recs, variant=variant,
                      bucket=bucket, rubric_scores=scores)

    writer.close()
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
