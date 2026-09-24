"""Redraw the essay's passing networks (Figures 6.1-6.4).

Run:  python3 plot_networks.py      -> figures/*.png and figures/*.pdf
Needs: matplotlib, networkx, scipy

Same layout as the original Mathematica figures (goalkeeper at the bottom,
formation rows above), with three changes:
  * arrow width is proportional to passes (the original used passes squared,
    so a 33-pass link was drawn ~55pt thick and hid everything around it);
  * arrows are curved, so A->B and B->A no longer sit on top of each other;
  * node size shows each player's weighted PageRank from recalculate.py.
"""
import os

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from data import TEAMS
from recalculate import corrected_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")

# Shirt numbers (same order as the players in data.py) and (x, y) positions
# copied from the original figures / Mathematica posnumbers variables.
LAYOUT = {
    "FC Barcelona vs Man Utd (2011)": dict(
        file="fig6-1_barcelona_2011", colour="#1d3c8f", formation="4-3-3",
        numbers=[1, 3, 14, 2, 22, 6, 16, 8, 17, 10, 7],
        pos={1: (0, 0), 3: (1, 1), 14: (-1, 1), 2: (2, 2), 22: (-2, 2), 16: (0, 2),
             6: (1, 3), 8: (-1, 3), 17: (2, 4), 7: (-2, 4), 10: (0, 5)}),
    "Man Utd vs FC Barcelona (2011)": dict(
        file="fig6-2_manutd_2011", colour="#c4122f", formation="4-4-2",
        numbers=[1, 3, 5, 15, 20, 11, 13, 25, 16, 10, 14],
        pos={1: (0, 0), 15: (-1, 1), 5: (1, 1), 3: (-2, 2), 20: (2, 2), 11: (-1, 3),
             16: (1, 3), 13: (-2, 4), 25: (2, 4), 10: (-1, 5), 14: (1, 5)}),
    "FC Barcelona vs Juventus (2015)": dict(
        file="fig6-3_barcelona_2015", colour="#1d3c8f", formation="4-3-3",
        numbers=[1, 3, 14, 18, 22, 5, 8, 4, 10, 11, 9],
        pos={1: (0, 0), 14: (-1, 1), 3: (1, 1), 18: (-2, 2), 5: (0, 2), 22: (2, 2),
             8: (-1, 3), 4: (1, 3), 11: (-2, 4), 10: (2, 4), 9: (0, 5)}),
    "Juventus vs FC Barcelona (2015)": dict(
        file="fig6-4_juventus_2015", colour="#222222", formation="4-3-1-2",
        numbers=[1, 15, 19, 26, 33, 6, 8, 21, 23, 10, 9],
        pos={1: (0, 0), 19: (-1, 1), 15: (1, 1), 33: (-2, 2), 21: (0, 2), 26: (2, 2),
             6: (-1, 3), 8: (1, 3), 23: (0, 4), 9: (-1, 5), 10: (1, 5)}),
}

INK, MUTED = "#1a1a1a", "#6b6b6b"
WIDTH_PER_PASS = 0.28          # points of line width per pass (linear)
MIN_PASSES = 2                 # links with fewer passes are left out to reduce clutter


def tint(colour, t):
    """Mix colour with white: t=0 -> very light, t=1 -> full colour."""
    r, g, b = to_rgb(colour)
    t = 0.18 + 0.82 * t
    return (1 - t + t * r, 1 - t + t * g, 1 - t + t * b)


def draw(team, ax, max_passes):
    names, A = TEAMS[team]
    cfg = LAYOUT[team]
    nums, pos, colour = cfg["numbers"], cfg["pos"], cfg["colour"]
    pr = corrected_metrics(A)["pagerank"]
    xy = [pos[n] for n in nums]
    radius = [0.13 + 1.1 * p for p in pr]          # node radius in data units

    edges = sorted(((A[i][j], i, j) for i in range(11) for j in range(11)
                    if A[i][j] >= MIN_PASSES), key=lambda e: e[0])
    for w, i, j in edges:                          # light first, heavy on top
        (x1, y1), (x2, y2) = xy[i], xy[j]
        d = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        ux, uy = (x2 - x1) / d, (y2 - y1) / d      # start/end on the circles' edges
        start = (x1 + ux * (radius[i] + 0.03), y1 + uy * (radius[i] + 0.03))
        end = (x2 - ux * (radius[j] + 0.05), y2 - uy * (radius[j] + 0.05))
        ax.add_patch(FancyArrowPatch(
            start, end, connectionstyle="arc3,rad=0.14",
            arrowstyle=f"-|>,head_length={2.5 + w * 0.12:.2f},head_width={1.5 + w * 0.07:.2f}",
            linewidth=0.4 + WIDTH_PER_PASS * w, color=tint(colour, w / max_passes),
            shrinkA=0, shrinkB=0,
            zorder=1 + w / 100, capstyle="round", joinstyle="round"))

    for k, ((x, y), r) in enumerate(zip(xy, radius)):
        ax.add_patch(plt.Circle((x, y), r, facecolor=colour, edgecolor="white",
                                linewidth=2, zorder=5))
        ax.text(x, y, str(nums[k]), ha="center", va="center", color="white",
                fontsize=11, fontweight="bold", zorder=6)
        ax.text(x, y - r - 0.1, names[k], ha="center", va="top", color=INK,
                fontsize=8.5, zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

    title = team.split(" vs ")[0]
    opp, year = team.split(" vs ")[1].rsplit(" ", 1)
    ax.set_title(f"{title} passing network vs {opp}", loc="left", fontsize=13,
                 fontweight="bold", color=INK, pad=22)
    ax.text(0, 1.012, f"Champions League final {year.strip('()')} · {cfg['formation']} · "
            f"{sum(map(sum, A))} passes between starters",
            transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")
    ax.set_xlim(-2.75, 2.75)
    ax.set_ylim(-0.75, 5.6)
    ax.set_aspect("equal")
    ax.axis("off")


def legend(fig, colour, max_passes, loc):
    handles = [Line2D([], [], color=tint(colour, w / max_passes),
                      linewidth=0.4 + WIDTH_PER_PASS * w, label=f"{w} passes")
               for w in (5, 15, 30)]
    handles.append(Line2D([], [], marker="o", linestyle="none", markersize=11,
                          markerfacecolor=colour, markeredgecolor="white",
                          label="bigger circle = higher PageRank"))
    fig.legend(handles=handles, loc=loc, ncol=4, frameon=False, fontsize=8.5,
               labelcolor=MUTED, handlelength=2.6)


def footer(fig):
    fig.text(0.02, 0.006, f"Data: Opta. Arrow width is proportional to passes from one player to another;\n"
             f"links with fewer than {MIN_PASSES} passes are hidden. Goalkeeper at the bottom.",
             fontsize=7.5, color=MUTED, ha="left", va="bottom")


def main():
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams["font.family"] = "DejaVu Sans"
    # One scale for all four figures so widths are comparable between teams.
    max_passes = max(max(map(max, A)) for _, A in TEAMS.values())

    for team, cfg in LAYOUT.items():
        fig, ax = plt.subplots(figsize=(6.4, 7.6))
        fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.08)
        draw(team, ax, max_passes)
        legend(fig, cfg["colour"], max_passes, "lower center")
        fig.legends[0].set_bbox_to_anchor((0.5, 0.03))
        footer(fig)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(OUT, f"{cfg['file']}.{ext}"), dpi=200, facecolor="white")
        plt.close(fig)

    # Side-by-side comparison of all four.
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 15.4))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05, hspace=0.12, wspace=0.04)
    for ax, team in zip(axes.flat, LAYOUT):
        draw(team, ax, max_passes)
    legend(fig, "#555555", max_passes, "lower center")
    fig.legends[0].set_bbox_to_anchor((0.5, 0.018))
    footer(fig)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"all_networks.{ext}"), dpi=160, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
