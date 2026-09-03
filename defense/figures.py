"""Publication figures from data already on disk. No model, no GPU.

    python -m defense.figures            # writes outputs/defense/fig1..fig3 .png

  fig1  dose-response: ASR vs dose for every variant, with Wilson intervals and
        the undefended / prompting reference lines. The executive-summary figure.
  fig2  tokens steered vs ASR at the operating doses -- shows that gated steers
        MORE tokens than oracle yet suppresses far less: targeting, not coverage.
  fig3  probe heatmap: per-token P(user) over the poisoned tool block with the
        injection span outlined, plus a zoom on the injection with token text and
        the threshold lines. Needs outputs/defense/probe_scores.json, written by
        gate_calibration.py; skipped with a message if absent.

Only figures 1-2 need results.csv. Under the application's +2h rule, graphs from
existing data are allowed; fig3's forward pass is a new measurement and belongs in
research time.
"""

import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

OUT_DIR = "outputs/defense"
CSV = os.environ.get("OUT_CSV", f"{OUT_DIR}/results.csv")
PROBE_JSON = f"{OUT_DIR}/probe_scores.json"

# --- validated reference palette (light surface); categorical in FIXED slot order
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = {            # slot order is the CVD-safety mechanism; never reorder
    "oracle": "#2a78d6",   # 1 blue
    "blind":  "#eb6834",   # 2 orange
    "random": "#1baf7a",   # 3 aqua
    "gated":  "#eda100",   # 4 yellow (low contrast on light -> always direct-labeled)
}
MARKER = {"oracle": "o", "blind": "s", "random": "^", "gated": "D"}   # secondary encoding
SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#1c5cab", "#0d366b"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def _f(r, k, d=float("nan")):
    try:
        return float(r[k])
    except (KeyError, ValueError, TypeError):
        return d


def load_rows():
    with open(CSV) as f:
        return list(csv.DictReader(f))


def step3(rows):
    return [r for r in rows if r["step"] == "3"]


def by_variant_dose(rows):
    """{variant: {|dose|: row}} for oracle/blind/random; gated collapsed to its best
    threshold per dose (min ASR -- the choice most favorable to gated, which still loses)."""
    out = {v: {} for v in SERIES}
    for r in step3(rows):
        v = r["variant"]
        if v not in SERIES:
            continue
        d = abs(_f(r, "coeff"))
        if v == "gated":
            cur = out[v].get(d)
            if cur is None or _f(r, "asr") < _f(cur, "asr"):
                out[v][d] = r
        else:
            out[v][d] = r
    return out


def refs(rows):
    s3 = step3(rows)
    none = next(r for r in s3 if r["variant"] == "none")
    prompt = next(r for r in s3 if r["variant"] == "prompt")
    return none, prompt


def _refline(ax, r, label, xlim):
    # Dashed line only: the reference intervals are in the results table, and drawn
    # as bands they merge into one slab that buries the data.
    y = _f(r, "asr")
    ax.axhline(y, color=MUTED, lw=1.2, ls=(0, (4, 3)))
    ax.text(xlim[1], y, f"  {label}  {y:.2f}", va="center", ha="left", fontsize=9, color=INK2)


def _repel(ys, gap=0.065, hi=1.0):
    """Push label y-positions apart so none sit within `gap` of each other."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for k in range(1, len(order)):
        i, j = order[k - 1], order[k]
        if out[j] - out[i] < gap:
            out[j] = out[i] + gap
    over = max(0.0, max(out) - hi)
    return [y - over for y in out]


# ------------------------------------------------------------------ fig 1
def fig1(rows):
    data = by_variant_dose(rows)
    none, prompt = refs(rows)
    doses = sorted({d for v in data.values() for d in v})

    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=200)
    # Extra room on the left so the direct labels sit INSIDE the axes rather than
    # over the y-axis label.
    xlim = (min(doses) - 0.14, max(doses) + 0.03)

    _refline(ax, none, "undefended", xlim)
    _refline(ax, prompt, "prompting baseline", xlim)

    firsts = []   # (variant, x, y) of each line's left-most point, for direct labels
    for v, col in SERIES.items():
        pts = [(d, data[v][d]) for d in doses if d in data[v]]
        if not pts:
            continue
        xs = [d for d, _ in pts]
        ys = [_f(r, "asr") for _, r in pts]
        lo = [ys[i] - _f(r, "asr_lo") for i, (_, r) in enumerate(pts)]
        hi = [_f(r, "asr_hi") - ys[i] for i, (_, r) in enumerate(pts)]
        ax.plot(xs, ys, color=col, lw=2, zorder=3)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=col, elinewidth=1, capsize=3, alpha=0.7, zorder=3)
        for d, r, y in zip(xs, [r for _, r in pts], ys):
            flagged = str(r.get("capability_ok", "True")).lower() == "false"
            ax.plot(d, y, marker=MARKER[v], ms=8, color=col,
                    mfc=SURFACE if flagged else col, mew=2, zorder=4)
            if flagged:
                # Above the marker: flagged points sit on the baseline, so a
                # below-offset would be clipped by the axes.
                ax.annotate(f"{_f(r,'stuck_rate'):.0%} stuck", (d, y), xytext=(8, 12),
                            textcoords="offset points", fontsize=8, color=INK2,
                            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
        firsts.append((v, xs[0], ys[0]))

    # Direct labels at the LEFT end, where the lines are most separated; the right
    # end is where oracle/blind/random converge at zero and labels collide.
    label_ys = _repel([y for _, _, y in firsts])
    for (v, x0, _), ly in zip(firsts, label_ys):
        ax.text(x0 - 0.012, ly, "gated (best T)" if v == "gated" else v,
                color=INK, fontsize=9, va="center", ha="right", fontweight="bold", zorder=5)

    ax.set_xlim(*xlim)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xticks(doses)
    ax.set_xticklabels([f"−{d:g}" for d in doses])
    ax.set_xlabel("steering dose α (away from userness, applied at layer 11)")
    ax.set_ylabel("attack success rate  (n = 30, Wilson 95%)")
    ax.set_title("Userness steering suppresses the injection; the probe-gated variant does not",
                 loc="left", fontsize=11, color=INK, pad=10)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(handles=[plt.Line2D([], [], color=c, marker=MARKER[v], ms=7, lw=2,
                                  label=("gated (best T)" if v == "gated" else v))
                       for v, c in SERIES.items()],
              loc="upper right", frameon=False, fontsize=9)
    fig.text(0.01, 0.01,
             "hollow marker = capability flag (≥20% of trajectories emitted no tool call, or ≥30% degenerate).\n"
             "None fired here; blind at −0.4 is 17% stuck -- elevated but under the flag.\n"
             "gated shown at its best-performing threshold per dose.",
             fontsize=7.5, color=MUTED, va="bottom")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    path = f"{OUT_DIR}/fig1_dose_response.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ------------------------------------------------------------------ fig 2
def fig2(rows):
    none, prompt = refs(rows)
    s3 = step3(rows)
    doses = [0.2, 0.4]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), dpi=200, sharey=True)

    for ax, d in zip(axes, doses):
        pts = [r for r in s3 if r["variant"] in SERIES and abs(_f(r, "coeff")) == d]
        xlim = (30, 6000)
        _refline(ax, none, "undefended", (30, 2400))
        _refline(ax, prompt, "prompting", (30, 2400))
        for r in pts:
            v = r["variant"]
            x, y = _f(r, "mean_steered_tokens"), _f(r, "asr")
            ax.errorbar(x, y, yerr=[[y - _f(r, "asr_lo")], [_f(r, "asr_hi") - y]],
                        fmt="none", ecolor=SERIES[v], elinewidth=1, capsize=3, alpha=0.7, zorder=3)
            ax.plot(x, y, marker=MARKER[v], ms=9, color=SERIES[v], mew=1.5, zorder=4)
            # Per-point offsets so neighbours (the three gated thresholds; random and
            # blind at the same x) do not overprint each other.
            if v == "gated":
                t = _f(r, "threshold")
                label = f"gated T={t:g}"
                dx, dy, ha = {0.3: (7, 9, "left"), 0.5: (7, -15, "left"),
                              0.7: (7, 9, "left")}.get(t, (7, 9, "left"))
            else:
                label = v
                # random to the LEFT and blind just above-right: at the strong dose
                # they share an x and sit on the baseline, so below-offsets clip.
                dx, dy, ha = {"oracle": (9, 6, "left"), "blind": (9, 4, "left"),
                              "random": (-9, 9, "right")}[v]
            ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points",
                        fontsize=8.5, color=INK, ha=ha, zorder=5)
        ax.set_xscale("log")
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel("tokens steered per step (log)")
        ax.set_title(f"dose α = −{d:g}", loc="left", fontsize=10, color=INK2)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("attack success rate  (Wilson 95%)")
    fig.suptitle("Gated steers more tokens than oracle yet suppresses far less: which tokens, not how many",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = f"{OUT_DIR}/fig2_tokens_vs_asr.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ------------------------------------------------------------------ fig 3
def fig3():
    if not os.path.exists(PROBE_JSON):
        print(f"fig3 skipped: {PROBE_JSON} not found. Run `python -m defense.gate_calibration` "
              "on the pod to produce it (one forward pass), then re-run this script.")
        return None
    d = json.load(open(PROBE_JSON))
    scores, toks = d["scores"], d["tokens"]
    lo, hi, thresholds = d["inj_lo"], d["inj_hi"], d.get("thresholds", [0.3, 0.5, 0.7])
    n, W = len(scores), 100
    rows = (n + W - 1) // W
    import numpy as np
    grid = np.full((rows, W), np.nan)
    for i, s in enumerate(scores):
        grid[i // W, i % W] = s

    cmap = LinearSegmentedColormap.from_list("userness", SEQ_RAMP)
    fig, (a, b) = plt.subplots(2, 1, figsize=(11, 7.2), dpi=200,
                               gridspec_kw={"height_ratios": [rows / 4.0, 3.2], "hspace": 0.45})

    im = a.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    for i in range(lo, hi):
        a.add_patch(Rectangle((i % W - 0.5, i // W - 0.5), 1, 1, fill=False, ec=INK, lw=0.9))
    a.set_xlabel(f"token position within the tool block  ({W} per row; black outline = injection span, {hi-lo} tokens)")
    a.set_ylabel("row")
    a.set_yticks(range(rows)); a.set_yticklabels([str(r * W) for r in range(rows)], fontsize=7)
    a.set_title("Where the probe sees userness: P(user) per token over the poisoned tool block",
                loc="left", fontsize=11, color=INK)
    cb = fig.colorbar(im, ax=a, fraction=0.025, pad=0.01)
    cb.set_label("P(user) at layer 14", color=INK2); cb.outline.set_edgecolor(AXIS)

    # zoom: injection ± context, as bars so a threshold line reads directly
    pad = 20
    z0, z1 = max(0, lo - pad), min(n, hi + pad)
    xs = list(range(z0, z1))
    cols = [SERIES["blind"] if lo <= i < hi else SERIES["oracle"] for i in xs]
    b.bar(xs, [scores[i] for i in xs], color=cols, width=0.85, lw=0)
    for t in thresholds:
        b.axhline(t, color=MUTED, lw=1, ls=(0, (4, 3)))
        b.text(z1 - 0.5, t, f" T={t:g}", va="center", ha="left", fontsize=8, color=INK2)
    b.set_xlim(z0 - 0.5, z1 + 3)
    b.set_ylim(0, 1.02)
    b.set_xticks(xs)
    b.set_xticklabels([toks[i].replace("\n", "⏎") for i in xs], rotation=90, fontsize=5.5, color=INK2)
    b.set_ylabel("P(user)")
    b.set_title("Zoom on the injection (orange) and ±20 surrounding page tokens (blue); "
                "dashed = gate thresholds", loc="left", fontsize=10, color=INK2)
    b.grid(axis="y", color=GRID, lw=0.8); b.set_axisbelow(True)
    b.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=SERIES["blind"], label="injection token"),
                      plt.Rectangle((0, 0), 1, 1, color=SERIES["oracle"], label="benign page token")],
             loc="upper left", frameon=False, fontsize=8)

    fired = {t: sum(1 for i in range(lo, hi) if scores[i] > t) for t in thresholds}
    fig.text(0.01, 0.005, "injection tokens above threshold: " +
             "   ".join(f"T={t:g}: {k}/{hi-lo}" for t, k in fired.items()), fontsize=8, color=MUTED)
    path = f"{OUT_DIR}/fig3_probe_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load_rows()
    for fn in (lambda: fig1(rows), lambda: fig2(rows), fig3):
        p = fn()
        if p:
            print("wrote", p)


if __name__ == "__main__":
    main()
