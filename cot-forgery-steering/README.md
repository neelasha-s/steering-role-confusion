# CoT-Forgery Steering — code

Scripts for the experiments written up in [`../steeringresults.md`](../steeringresults.md) (data) and
[`../COT-FORGERY-STEERING.md`](../COT-FORGERY-STEERING.md) (narrative). They test whether the paper's
*correlational* CoTness→ASR finding can be made *causal* by steering the residual stream — the same
diff-of-means method the repo's userness experiment uses, pointed at the CoT/role axis.

## Scripts → experiments

| script | experiment (in `steeringresults.md`) |
| ------ | ------------------------------------ |
| `forgery_endpoint.py` | Exp 1 — forgery validation (base 0/3 → styled 3/3) |
| `cot_steer_full.py`   | Exp 2 — full-destyle steering (n=3 greedy) |
| `cot_space.py`        | Exp 3 — "add a space" reproduction + steering (n=3 greedy) |
| `cot_space_n15.py`    | Exp 4 — space steering, sampled, moderate doses (n=15) |
| `cot_space_strong.py` | Exp 5 — space steering, strong doses + malformed guard (n=15) |
| `cot_raise.py`        | Exp 6 — raise-from-floor (destyled baseline, n=15) |

## Running environment

These were run on a **RunPod H100** at `/workspace/steering-role-confusion` (that pod is gone). Each
script hardcodes `os.chdir("/workspace/steering-role-confusion")` — change it to wherever you place the
dependencies below, or run from that directory.

Dependencies (all were present in that working dir):
- `utils/loader.py` — `load_model_and_tokenizer("gptoss-20b", ...)`; loads gpt-oss-20b with flash
  attention (`kernels-community/vllm-flash-attn3`), falling back to sdpa/eager. (In this repo:
  [`../utils/loader.py`](../utils/loader.py).)
- `outputs/gptoss-20b-resid-L14.pkl` — the trained layer-14 5-way role probe (`pk["clf"]`), reused as
  the CoTness readout. (In this repo: [`../outputs/`](../outputs/).)
- `openai/gpt-oss-20b` weights (MXFP4), CUDA GPU with ≥ ~40 GB, `torch`, `transformers`, `kernels`,
  `triton`, `scikit-learn`, `numpy`.

## Method (shared by all scripts)

- Steer at `layers[12]` (peak CoTness fidelity), read the probe at `layers[14]` (non-circular).
- Direction = diff-of-means, unit-normalized; additive `h += c·‖h‖·v̂` on the injected policy tokens,
  **prefill only**. `random` = norm-matched Gaussian control.
- Scaffold = paper-faithful ChatGPT harmony header + "cat-fact" frame, forgery inside the user turn.
  The styled synthetic policies are **paper-verbatim** few-shot examples from
  `experiments/cot-forgery-chat-evals/prompts/forgery-prompt-openai.yaml` in the paper's repo
  (github.com/role-confusion/prompt-injection-as-role-confusion). The destyled / spaced variants are
  ours.
- ASR = final-channel refusal proxy (upper bound). `n=15` runs sample at `T=1.0`, report Wilson 95% CIs,
  and bucket comply / refuse / **malformed** (empty or <5-word final channel).

## Safety

Every script classifies generations for refusal only — model output text is never printed or saved.
The harmful questions and forged policies are the published attack's own examples, used here to
reproduce a documented jailbreak for interpretability, not to elicit or store harmful content.
