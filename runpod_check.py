#!/usr/bin/env python3
"""Use the RunPod API token from .env — read-only credential + connectivity check.

Loads RUNPOD from .env and calls the RunPod GraphQL API to (1) confirm the token
authenticates and (2) list the GPU types big enough to hold gpt-oss-20b. It creates
no pods, starts nothing, and spends nothing — running the experiment on RunPod is a
separate, deliberate step. The token is never printed in full.

    python runpod_check.py

Exit code 0 = the key works; non-zero = missing / malformed / rejected.
"""
import json
import sys
import urllib.error
import urllib.request

ENV_PATH = ".env"
ENV_VAR = "RUNPOD"
API = "https://api.runpod.io/graphql"
MIN_VRAM_GB = 48  # gpt-oss-20b dequantizes to bf16 (~40GB) off the MXFP4 kernels


def load_token(path=ENV_PATH, var=ENV_VAR):
    """Return the token for `var` from `path`. Prefers python-dotenv; falls back to a
    tolerant hand parse so a slightly malformed .env still resolves."""
    try:
        from dotenv import dotenv_values
        val = dotenv_values(path).get(var)
    except ImportError:
        val = None
    if not val:  # fallback: tolerate stray spaces / quotes around the assignment
        try:
            for line in open(path, encoding="utf-8"):
                stripped = line.strip()
                if stripped.startswith(var) and "=" in stripped:
                    val = stripped.split("=", 1)[1].strip().strip("'\"")
                    break
        except FileNotFoundError:
            sys.exit(f"{path} not found — create it from .env.example")
    if not val:
        sys.exit(f"{var} not set in {path} — see .env.example")
    return val


def graphql(token, query):
    """POST a GraphQL query authenticated by the token; return the parsed JSON."""
    req = urllib.request.Request(
        f"{API}?api_key={token}",
        data=json.dumps({"query": query}).encode(),
        # RunPod sits behind Cloudflare, which 403s (error 1010) the default urllib
        # User-Agent; a normal browser UA clears the bot filter.
        headers={
            "content-type": "application/json",
            "User-Agent": "Mozilla/5.0 (runpod-check; +https://runpod.io)",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def main():
    token = load_token()
    print(f"loaded {ENV_VAR} = {token[:7]}…{token[-2:]}  ({len(token)} chars)")

    try:
        me = graphql(token, "query { myself { id } }")
    except urllib.error.HTTPError as err:
        sys.exit(f"auth request failed: HTTP {err.code} {err.read().decode()[:200]}")
    except urllib.error.URLError as err:
        sys.exit(f"could not reach RunPod: {err.reason}")

    if me.get("errors") or not (me.get("data") or {}).get("myself"):
        sys.exit(f"token rejected by RunPod: {me.get('errors', me)}")
    print(f"authenticated ✓  account id = {me['data']['myself']['id']}")

    gpus = (graphql(token, "query { gpuTypes { id displayName memoryInGb } }")
            .get("data", {}).get("gpuTypes") or [])
    big = sorted((g for g in gpus if (g.get("memoryInGb") or 0) >= MIN_VRAM_GB),
                 key=lambda g: g["memoryInGb"])
    print(f"GPU types visible: {len(gpus)}  (≥{MIN_VRAM_GB}GB, fits gpt-oss-20b: {len(big)})")
    for g in big[:10]:
        print(f"  {g['displayName']:30s} {g['memoryInGb']:>4}GB  id={g['id']}")
    print("\nkey works ✓  (no pods created)")


if __name__ == "__main__":
    main()
