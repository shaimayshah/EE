"""Messi gravity, visual: 360 freeze frames of Messi's biggest gravity moments.

Each panel is the frame at the moment Messi passes: every player the broadcast
camera could see, the visible area, a 5-unit circle around Messi, and the pass to
the receiver. The caption gives defenders within 5 units of Messi (gravity) and the
receiver's distance to their nearest defender at the receipt frame (space created).

Moments: Messi's three open-play assists + the passes with the most defenders on
Messi whose receiver still had a defender-free circle.

Usage: .venv/bin/python analysis/freeze_frames.py
"""
import glob
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Polygon as MplPolygon, Rectangle

import gravity

DATA = gravity.DATA
OUT = f"{gravity.OUT}/freeze_frames"
MESSI = "Lionel Andrés Messi Cuccittini"
R = 5

# Validated default categorical palette (dataviz skill): slot 1 blue, slot 2 orange.
ARG = "#2a78d6"
OPP = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
LINES = "#c9c8c2"


def opp_dists(frame, x, y):
    return sorted(math.dist(p["location"], (x, y)) for p in frame["freeze_frame"] if not p["teammate"])


def messi_passes():
    matches = {m["match_id"]: m for m in json.load(open(f"{DATA}/matches.json"))}
    out = []
    for path in sorted(glob.glob(f"{DATA}/events/*.json")):
        mid = int(os.path.basename(path)[:-5])
        m = matches[mid]
        if "Argentina" not in (m["home_team"]["home_team_name"], m["away_team"]["away_team_name"]):
            continue
        events = json.load(open(path))
        by_id = {e["id"]: e for e in events}
        frames = {f["event_uuid"]: f for f in json.load(open(f"{DATA}/three-sixty/{mid}.json"))}
        for e in events:
            if e["type"]["name"] != "Pass" or e["player"]["name"] != MESSI:
                continue
            if "outcome" in e["pass"] or not gravity.is_open_play(e) or e["id"] not in frames:
                continue
            rec = next((by_id[r] for r in e.get("related_events", [])
                        if by_id.get(r, {}).get("type", {}).get("name") == "Ball Receipt*"), None)
            if rec is None or rec["id"] not in frames:
                continue
            x, y = e["location"]
            rx, ry = rec["location"]
            d_messi = opp_dists(frames[e["id"]], x, y)
            d_rec = opp_dists(frames[rec["id"]], rx, ry)
            opponent = (m["away_team"]["away_team_name"] if m["home_team"]["home_team_name"] == "Argentina"
                        else m["home_team"]["home_team_name"])
            out.append({
                "event": e, "frame": frames[e["id"]], "receipt": rec,
                "match": f"{m['competition_stage']['name']} v {opponent}",
                "opp_near_messi": sum(d < R for d in d_messi),
                "rec_nearest": d_rec[0] if d_rec else float("inf"),
                "assist": bool(e["pass"].get("goal_assist")),
            })
    return out


def draw_pitch(ax):
    ax.set_facecolor(SURFACE)
    kw = dict(color=LINES, lw=1, zorder=0)
    ax.add_patch(Rectangle((0, 0), 120, 80, fill=False, **kw))
    ax.plot([60, 60], [0, 80], **kw)
    ax.add_patch(Circle((60, 40), 10, fill=False, **kw))
    for x0, sgn in ((0, 1), (120, -1)):
        ax.add_patch(Rectangle((x0 if sgn > 0 else x0 - 18, 18), 18, 44, fill=False, **kw))
        ax.add_patch(Rectangle((x0 if sgn > 0 else x0 - 6, 30), 6, 20, fill=False, **kw))
        ax.add_patch(Arc((x0 + sgn * 12, 40), 20, 20, theta1=-53 if sgn > 0 else 127,
                         theta2=53 if sgn > 0 else 233, **kw))
    ax.set_xlim(-2, 122)
    ax.set_ylim(82, -2)  # StatsBomb y runs top to bottom
    ax.set_aspect("equal")
    ax.axis("off")


def draw_moment(ax, mo):
    e, f, rec = mo["event"], mo["frame"], mo["receipt"]
    draw_pitch(ax)
    va = f["visible_area"]
    ax.add_patch(MplPolygon(list(zip(va[0::2], va[1::2])), closed=True, facecolor="#f0efec",
                            edgecolor="none", zorder=0.5))
    for p in ax.patches[:-1]:  # keep pitch markings above the visible-area wash
        p.set_zorder(0.6)
    for ln in ax.lines:
        ln.set_zorder(0.6)
    x, y = e["location"]
    ax.add_patch(Circle((x, y), R, fill=False, ls="--", lw=1, color=INK_2, zorder=1))
    for p in f["freeze_frame"]:
        px, py = p["location"]
        if p["actor"]:
            continue
        if p["teammate"]:
            ax.scatter(px, py, s=46, marker="o", color=ARG, edgecolor=SURFACE, lw=1.5, zorder=3)
        else:
            ax.scatter(px, py, s=46, marker="s", color=OPP, edgecolor=SURFACE, lw=1.5, zorder=3)
        if p["keeper"]:
            ax.scatter(px, py, s=110, marker="o" if p["teammate"] else "s", facecolor="none",
                       edgecolor=INK_2, lw=0.8, zorder=2.9)
    ex, ey = e["pass"]["end_location"]
    ax.annotate("", xy=(ex, ey), xytext=(x, y), zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4, shrinkA=6, shrinkB=4))
    ax.scatter(x, y, s=200, marker="*", color=ARG, edgecolor=INK, lw=1, zorder=4)
    ax.annotate("Messi", (x, y), xytext=(0, 9), textcoords="offset points", ha="center",
                fontsize=8, color=INK, weight="bold", zorder=5)
    rname = gravity.short(rec["player"]["name"])
    ax.annotate(rname, (ex, ey), xytext=(0, -10), textcoords="offset points", ha="center",
                fontsize=8, color=INK, zorder=5)
    tag = "  ·  ASSIST" if mo["assist"] else ""
    near = f"{mo['rec_nearest']:.1f}" if math.isfinite(mo["rec_nearest"]) else "none visible"
    ax.set_title(f"{mo['match']}, {e['minute']}'{tag}\n"
                 f"{mo['opp_near_messi']} defenders within {R} of Messi  ·  "
                 f"{rname}'s nearest defender: {near}", fontsize=9, color=INK, loc="left")


def legend(fig):
    h = [
        plt.Line2D([], [], marker="*", ls="", color=ARG, markeredgecolor=INK, ms=12, label="Messi (on the ball)"),
        plt.Line2D([], [], marker="o", ls="", color=ARG, ms=7, label="Argentina"),
        plt.Line2D([], [], marker="s", ls="", color=OPP, ms=7, label="Opponent"),
        plt.Line2D([], [], ls="--", color=INK_2, label=f"{R}-unit radius around Messi"),
        plt.Rectangle((0, 0), 1, 1, color="#f0efec", label="Area the camera could see"),
    ]
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, fontsize=8)


def pick(moments, n_extra=3):
    assists = [m for m in moments if m["assist"]]
    rest = [m for m in moments if not m["assist"] and m["rec_nearest"] >= R]
    rest.sort(key=lambda m: (m["opp_near_messi"], m["rec_nearest"]), reverse=True)
    return rest[:n_extra] + assists


def main():
    os.makedirs(OUT, exist_ok=True)
    moments = pick(messi_passes())
    for i, mo in enumerate(moments, 1):
        fig, ax = plt.subplots(figsize=(8, 6), facecolor=SURFACE)
        draw_moment(ax, mo)
        legend(fig)
        fig.tight_layout(rect=(0, 0.05, 1, 0.97))
        fig.savefig(f"{OUT}/{i:02d}_{mo['event']['minute']}min_{mo['match'].split(' v ')[1].replace(' ', '_')}.png",
                    dpi=150, facecolor=SURFACE)
        plt.close(fig)

    cols = 3
    rows = math.ceil(len(moments) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.6 * rows), facecolor=SURFACE)
    for ax, mo in zip(axes.flat, moments):
        draw_moment(ax, mo)
    for ax in list(axes.flat)[len(moments):]:
        ax.axis("off")
    fig.suptitle("Messi's gravity in 360 freeze frames: defenders around him at the moment he passes (World Cup 2022)",
                 fontsize=13, color=INK, x=0.01, ha="left")
    legend(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(f"{gravity.OUT}/messi_freeze_frames.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    for mo in moments:
        print(f"{mo['match']:<32} {mo['event']['minute']:>3}'  assist={mo['assist']!s:5} "
              f"defenders_on_messi={mo['opp_near_messi']}  receiver={mo['receipt']['player']['name']}  "
              f"receiver_nearest={mo['rec_nearest']:.1f}")


if __name__ == "__main__":
    main()
