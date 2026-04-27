#!/usr/bin/env python3
"""Diagram of the within-recording leave-one-out adaptation scheme.

Top:  one recording split into N outer chunks.
Mid:  N passes; in pass i, chunk i is adapted on (held out from decoding),
      and inference is run on every other chunk with the adapted weights.
Bot:  per-position contributions from every pass are averaged into the
      final stitched probabilities, which are then decoded.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


N_CHUNKS = 5
CHUNK_W = 1.6
CHUNK_H = 0.55
GAP = 0.08
ROW_GAP = 0.55

ADAPT_COLOR = "#c44e52"
INFER_COLOR = "#4c72b0"
HELD_OUT_FACE = "#f0f0f0"
HELD_OUT_EDGE = "#999999"
TEXT_COLOR = "#222222"
ARROW_COLOR = "#555555"


def chunk_box(ax, x, y, w, h, face, edge, label=None, label_color="white",
              hatch=None, alpha=1.0, lw=0.8):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.06",
        facecolor=face, edgecolor=edge, linewidth=lw, alpha=alpha,
    )
    if hatch is not None:
        box.set_hatch(hatch)
    ax.add_patch(box)
    if label is not None:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=9, color=label_color, fontweight="bold")


def draw_recording_row(ax, y, x0):
    ax.text(x0 - 0.25, y + CHUNK_H / 2, "Recording",
            ha="right", va="center", fontsize=10, color=TEXT_COLOR)
    for k in range(N_CHUNKS):
        x = x0 + k * (CHUNK_W + GAP)
        chunk_box(ax, x, y, CHUNK_W, CHUNK_H,
                  face="#dddddd", edge="#666666",
                  label=f"chunk {k + 1}", label_color="#222222")


def draw_pass_row(ax, y, x0, adapt_idx):
    ax.text(x0 - 0.25, y + CHUNK_H / 2, f"pass {adapt_idx + 1}",
            ha="right", va="center", fontsize=9, color=TEXT_COLOR)

    total_w = N_CHUNKS * CHUNK_W + (N_CHUNKS - 1) * GAP
    adapt_x = x0 + adapt_idx * (CHUNK_W + GAP)

    # Left infer span (chunks before the adapt chunk).
    if adapt_idx > 0:
        left_w = adapt_idx * (CHUNK_W + GAP) - GAP
        chunk_box(ax, x0, y, left_w, CHUNK_H,
                  face=INFER_COLOR, edge=INFER_COLOR,
                  alpha=0.85)
    # Right infer span.
    if adapt_idx < N_CHUNKS - 1:
        right_x = adapt_x + CHUNK_W + GAP
        right_w = (x0 + total_w) - right_x
        chunk_box(ax, right_x, y, right_w, CHUNK_H,
                  face=INFER_COLOR, edge=INFER_COLOR,
                  alpha=0.85)

    # Single "infer" label centred on whichever side has more room.
    if adapt_idx >= N_CHUNKS / 2:
        label_x = x0 + (adapt_idx * (CHUNK_W + GAP) - GAP) / 2
    else:
        right_x = adapt_x + CHUNK_W + GAP
        label_x = (right_x + x0 + total_w) / 2
    ax.text(label_x, y + CHUNK_H / 2, "infer",
            ha="center", va="center", fontsize=9,
            color="white", fontweight="bold")

    chunk_box(ax, adapt_x, y, CHUNK_W, CHUNK_H,
              face=ADAPT_COLOR, edge=ADAPT_COLOR,
              label="adapt", label_color="white")


def draw_stitched_row(ax, y, x0):
    ax.text(x0 - 0.25, y + CHUNK_H / 2, "stitched\n$\\bar p(\\cdot)$",
            ha="right", va="center", fontsize=9, color=TEXT_COLOR)
    for k in range(N_CHUNKS):
        x = x0 + k * (CHUNK_W + GAP)
        chunk_box(ax, x, y, CHUNK_W, CHUNK_H,
                  face="#7c9c6b", edge="#5d7a4f",
                  label=f"avg @ {k + 1}", label_color="white")


def main():
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    ax.set_axis_off()

    x0 = 1.6

    total_w = N_CHUNKS * CHUNK_W + (N_CHUNKS - 1) * GAP
    rec_y = N_CHUNKS * (CHUNK_H + ROW_GAP) + CHUNK_H + 0.9
    pass_top = N_CHUNKS * (CHUNK_H + ROW_GAP)
    stitched_y = -CHUNK_H - 0.9
    decode_y = stitched_y - 1.0

    draw_recording_row(ax, rec_y, x0)

    pass_ys = []
    for i in range(N_CHUNKS):
        y = pass_top - i * (CHUNK_H + ROW_GAP) - CHUNK_H
        draw_pass_row(ax, y, x0, i)
        pass_ys.append(y)

    for k in range(N_CHUNKS):
        x_centre = x0 + k * (CHUNK_W + GAP) + CHUNK_W / 2
        ax.add_patch(FancyArrowPatch(
            (x_centre, rec_y),
            (x_centre, pass_top + 0.05),
            arrowstyle="-|>", color=ARROW_COLOR, mutation_scale=10, lw=0.8,
            shrinkA=2, shrinkB=2,
        ))

    bracket_x_left = x0 - 0.05
    bracket_x_right = x0 + total_w + 0.05
    bracket_top = pass_ys[-1] - 0.1
    bracket_bot = stitched_y + CHUNK_H + 0.45

    ax.plot([bracket_x_left, bracket_x_right], [bracket_top, bracket_top],
            color=ARROW_COLOR, lw=0.9)
    ax.plot([bracket_x_left, bracket_x_left], [bracket_top, bracket_top - 0.18],
            color=ARROW_COLOR, lw=0.9)
    ax.plot([bracket_x_right, bracket_x_right], [bracket_top, bracket_top - 0.18],
            color=ARROW_COLOR, lw=0.9)

    centre_x = x0 + total_w / 2
    ax.add_patch(FancyArrowPatch(
        (centre_x, bracket_top),
        (centre_x, bracket_bot),
        arrowstyle="-|>", color=ARROW_COLOR, mutation_scale=12, lw=1.0,
    ))
    ax.text(centre_x + 0.15, (bracket_top + bracket_bot) / 2,
            "average per-position\nsoftmax probabilities",
            ha="left", va="center", fontsize=9, color=TEXT_COLOR)

    draw_stitched_row(ax, stitched_y, x0)

    ax.add_patch(FancyArrowPatch(
        (centre_x, stitched_y - 0.02),
        (centre_x, decode_y + 0.45),
        arrowstyle="-|>", color=ARROW_COLOR, mutation_scale=12, lw=1.0,
    ))
    ax.text(centre_x, decode_y + 0.18,
            "beam-search decode $\\rightarrow$ hypothesis",
            ha="center", va="center", fontsize=10, color=TEXT_COLOR,
            bbox=dict(boxstyle="round,pad=0.3", fc="#fafafa", ec="#999999"))

    legend_handles = [
        mpatches.Patch(color=ADAPT_COLOR, label="adapt (NSTI on this chunk)"),
        mpatches.Patch(color=INFER_COLOR, label="infer with adapted weights"),
        mpatches.Patch(color="#7c9c6b", label="averaged stitched probabilities"),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              bbox_to_anchor=(1.0, 1.05), frameon=False, fontsize=9)

    ax.set_xlim(x0 - 1.6, x0 + total_w + 0.5)
    ax.set_ylim(decode_y - 0.5, rec_y + CHUNK_H + 0.6)
    ax.set_aspect("equal")

    out_dir = Path(__file__).parent
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"within_loo_scheme.{ext}", bbox_inches="tight",
                    dpi=200)
    print(f"Wrote {out_dir / 'within_loo_scheme.pdf'}")
    print(f"Wrote {out_dir / 'within_loo_scheme.png'}")


if __name__ == "__main__":
    main()
