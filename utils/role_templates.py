"""Harmony role templates for the probe's training corpus.

RECONSTRUCTED, not original. The notebook was written against a `utils` package living
outside this repo (`/workspace/prompt-injection-as-role-confusion`), which was never
vendored here. These templates are reconstructed from the notebook's own Harmony message
builders in section 3, which render the same five roles.

This matters for one thing only: `outputs/gptoss-20b-resid-L14.pkl` was trained with the
*original* templates. Reuse that pickle as-is and the reconstruction is irrelevant. Retrain
the probe (section 5) and any divergence here changes the probe you get.
"""

# `content_token_span` locates content by rindex("<|message|>") .. rindex("<|end|>"),
# so every template must end with exactly one <|message|>...<|end|> pair.
_TEMPLATES = {
    "system":    "<|start|>system<|message|>{content}<|end|>",
    "developer": "<|start|>developer<|message|>{content}<|end|>",
    "user":      "<|start|>user<|message|>{content}<|end|>",
    "cot":       "<|start|>assistant<|channel|>analysis<|message|>{content}<|end|>",
    "assistant": "<|start|>assistant<|channel|>final<|message|>{content}<|end|>",
    "tool":      "<|start|>functions.bash to=assistant<|channel|>commentary<|message|>{content}<|end|>",
}

ROLES = ("system", "user", "cot", "assistant", "tool")


def render_single_message(model_prefix, role, content):
    """Wrap `content` in a one-message Harmony transcript under `role`."""
    if not model_prefix.startswith("gptoss"):
        raise ValueError(f"only the gpt-oss Harmony format is reconstructed here, got {model_prefix!r}")
    if role not in _TEMPLATES:
        raise KeyError(f"unknown role {role!r}; known: {sorted(_TEMPLATES)}")
    return _TEMPLATES[role].format(content=content)
