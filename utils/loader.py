"""Model loading.

RECONSTRUCTED, not original — see the note in `role_templates.py`. The signature and return
tuple are fixed by the notebook's call site:

    tokenizer, model, arch, n_layers = load_model_and_tokenizer(MODEL_PREFIX, device=DEVICE)

and by what the notebook does downstream: `model.model.layers[i]` must be the decoder block
list, and `model.config.hidden_size` must be the residual width (2880 for gpt-oss-20b).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "gptoss-20b": "openai/gpt-oss-20b",
}


# Attention backends tried in order. Flash attention is REQUIRED for the sweep: it steers a
# batch of ~50 trajectories over multi-thousand-token pages, and eager attention materializes
# the full (batch × heads × seq × seq) score matrix — ~65GB at batch 50 — and OOMs. The
# original run logged "kernels-community/vllm-flash-attn3"; that hub kernel (via the `kernels`
# package) keeps attention memory linear. `sdpa` is a built-in fallback; `eager` is last resort.
ATTN_FALLBACKS = ["kernels-community/vllm-flash-attn3", "sdpa", "eager"]


def load_model_and_tokenizer(model_prefix, device="cuda:0", dtype="auto", attn_implementation=None):
    """Return (tokenizer, model, arch, n_layers) for a registered model prefix.

    gpt-oss ships MXFP4-quantized. On CUDA with the `kernels` package present, transformers
    loads the packed weights natively; everywhere else it dequantizes to bf16, which needs
    roughly 40GB of RAM. Loading on a laptop is expected to be slow or to OOM — the analysis
    cells that only read `outputs/` do not need this function.

    `attn_implementation` pins the attention backend; None tries ATTN_FALLBACKS in order and
    keeps the first that loads (flash first — see the note above).
    """
    if model_prefix not in MODELS:
        raise KeyError(f"unknown model prefix {model_prefix!r}; known: {sorted(MODELS)}")
    model_id = MODELS[model_prefix]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    candidates = [attn_implementation] if attn_implementation else ATTN_FALLBACKS
    model = None
    for impl in candidates:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=dtype, device_map=device, attn_implementation=impl)
            print(f"Attention implementation: {impl}")
            break
        except Exception as ex:
            print(f"attn_implementation={impl!r} unavailable ({type(ex).__name__}: {str(ex)[:120]}); trying next")
    if model is None:
        raise RuntimeError(f"no attention backend loaded from {candidates}")
    model.eval()

    arch = model.config.model_type
    n_layers = model.config.num_hidden_layers
    return tokenizer, model, arch, n_layers
