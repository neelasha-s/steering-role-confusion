"""Turn results.csv into the two figures the project exists to produce.

    python -m defense.report

Outputs (into outputs/defense/):
  * tradeoff.html      benign cost (x) vs held-out ASR (y), one point per
                       variant/threshold -- the headline robustness/side-effect curve
  * summary.txt        per-arm table with capability + intervention columns and any
                       warnings, so a reader sees at a glance which ASRs are trustworthy

This is read-only over the CSV -- no model, runs anywhere.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

IN_CSV = os.environ.get("OUT_CSV", "outputs/defense/results.csv")
OUT_DIR = "outputs/defense"


def _read():
    with open(IN_CSV) as f:
        return list(csv.DictReader(f))


def _f(row, key, default=0.0):
    v = row.get(key, "")
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def write_summary(rows):
    lines = ["ARM SUMMARY  (ASR shown with Wilson 95% CI; warnings flag untrustworthy cells)", ""]
    header = f"{'arm':34s} {'n':>3} {'ASR':>6} {'95% CI':>13} {'stuck':>6} {'degen':>6} {'steer':>6}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        n = int(_f(r, "n"))
        if n == 0:
            continue
        asr = _f(r, "asr")
        ci = f"[{_f(r,'asr_lo'):.2f},{_f(r,'asr_hi'):.2f}]"
        line = (f"{r['arm']:34s} {n:>3} {asr:>6.2f} {ci:>13} "
                f"{_f(r,'stuck_rate'):>6.0%} {_f(r,'degenerate_rate'):>6.0%} "
                f"{_f(r,'mean_steered_tokens'):>6.0f}")
        lines.append(line)
        if r.get("warning"):
            lines.append(f"    !! {r['warning']}")
    text = "\n".join(lines)
    os.makedirs(OUT_DIR, exist_ok=True)
    open(os.path.join(OUT_DIR, "summary.txt"), "w").write(text)
    print(text)
    print(f"\nwrote {OUT_DIR}/summary.txt")


def write_tradeoff(rows):
    """Benign cost vs ASR, one series per variant. Benign cost is taken from the
    Step-5 B3 bucket (1 - rubric_mean/2); ASR from the matching Step-3 arm."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed; skipping tradeoff plot (summary.txt still written)")
        return

    # benign cost per variant from B3 (the load-bearing bucket)
    benign_cost = {}
    for r in rows:
        if r.get("bucket") == "B3_act" and r.get("rubric_mean") != "":
            benign_cost[r["variant"]] = 1.0 - _f(r, "rubric_mean") / 2.0

    fig = go.Figure()
    for variant in ("oracle", "blind", "gated"):
        pts = [r for r in rows if r.get("variant") == variant and int(_f(r, "n")) > 0
               and r.get("step") == "3"]
        if not pts:
            continue
        xs = [benign_cost.get(variant, 0.0)] * len(pts)
        ys = [_f(r, "asr") for r in pts]
        labels = [f"c={r.get('coeff','')} T={r.get('threshold','')}" for r in pts]
        fig.add_scatter(x=xs, y=ys, mode="markers+text", name=variant, text=labels,
                        textposition="top center")

    fig.update_layout(
        title="Robustness vs side-effect: held-out ASR vs benign cost (B3)",
        xaxis_title="benign cost  (1 - B3 rubric/2)", yaxis_title="attack success rate",
        xaxis_range=[0, 1], yaxis_range=[0, 1], template="plotly_white", height=460)
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.write_html(os.path.join(OUT_DIR, "tradeoff.html"))
    print(f"wrote {OUT_DIR}/tradeoff.html")


def main():
    rows = _read()
    write_summary(rows)
    write_tradeoff(rows)


if __name__ == "__main__":
    main()
