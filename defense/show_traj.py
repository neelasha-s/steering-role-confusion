"""Read trajectories -- randomly sampled, never cherry-picked.

    python -m defense.show_traj --list
    python -m defense.show_traj --arm "blind c=-0.4" --n 5
    python -m defense.show_traj --bucket B3_act --variant none --n 5
    python -m defense.show_traj --arm "none c=0" --n 3 --seed 7 --full

This tool samples with a FIXED SEED and prints the
seed, so the selection is reproducible and provably not hand-picked. Record the
seed in the write-up next to the examples.

By default the seed context (system/developer/user/cot/seed tool call) is
collapsed to one line each and the fetched page is truncated, so you can read the
model's actual actions. --full shows everything.
"""

import argparse
import json
import os
import random
import sys
import textwrap
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

PATH = os.environ.get("OUT_TRAJ", "outputs/defense/trajectories.jsonl")


def load():
    with open(PATH) as f:
        return [json.loads(l) for l in f if l.strip()]


def matches(r, a):
    if a.arm and r.get("arm") != a.arm:
        return False
    if a.variant and r.get("variant") != a.variant:
        return False
    if a.bucket and r.get("bucket") != a.bucket:
        return False
    if a.family and r.get("family") != a.family:
        return False
    if a.outcome and r.get("outcome") != a.outcome:
        return False
    return True


def clip(s, n):
    s = (s or "").replace("\n", "\\n")
    return s if len(s) <= n else s[:n] + f" ...[{len(s)-n} more chars]"


def show(r, full):
    hdr = (f"arm={r.get('arm')!r} variant={r.get('variant')} sample={r.get('sample')} | "
           f"outcome={r.get('outcome')} attack={r.get('attack')} steps={r.get('n_steps')} "
           f"steered/step={r.get('steered_tokens')} degenerate={r.get('degenerate')}")
    print("=" * min(len(hdr), 110))
    print(hdr)
    print("=" * min(len(hdr), 110))
    for i, t in enumerate(r["turns"]):
        kind = t.get("kind")
        text = t.get("text", "")
        if kind in ("system", "developer", "cot", "tool_call") and not full:
            print(f"  [{kind:11s}] {clip(text, 90)}")
        elif kind == "user":
            print(f"  [{kind:11s}] {clip(text, 300) if not full else text}")
        elif kind == "tool_result" and i < 6 and not full:
            # the seed page: show where the injection sits, not the whole article
            print(f"  [{kind:11s}] <fetched page, {len(text)} chars>")
        else:
            body = text if full else clip(text, 700)
            cmd = t.get("cmd")
            tag = f"  cmd={cmd!r}" if cmd else ""
            print(f"  [{kind:11s}]{tag}")
            for line in textwrap.wrap(body, 104):
                print(f"               {line}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list arms with counts and exit")
    ap.add_argument("--arm"); ap.add_argument("--variant"); ap.add_argument("--bucket")
    ap.add_argument("--family"); ap.add_argument("--outcome")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()

    recs = load()
    if a.list:
        c = Counter((r.get("step"), r.get("arm")) for r in recs)
        for (step, arm), k in sorted(c.items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
            print(f"  step {step}  {arm:36s} {k:>3} trajectories")
        return

    pool = [r for r in recs if matches(r, a)]
    if not pool:
        print("no trajectories match; use --list to see arms"); return
    rng = random.Random(a.seed)
    picked = rng.sample(pool, min(a.n, len(pool)))
    print(f"# {len(pool)} matching trajectories; showing {len(picked)} sampled with seed={a.seed} "
          f"(reproducible; cite this seed in the write-up)\n")
    for r in picked:
        show(r, a.full)


if __name__ == "__main__":
    main()
