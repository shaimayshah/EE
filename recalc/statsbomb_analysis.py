"""Redo the essay's analysis with StatsBomb event data for both finals.

Run:  python3 fetch_statsbomb.py && python3 statsbomb_analysis.py > STATSBOMB.md
Writes figures/sb_networks.*, figures/sb_xg_timeline.*, figures/sb_halves.*
Needs: networkx, scipy, numpy, matplotlib

Data: StatsBomb Open Data (https://github.com/statsbomb/open-data). Credit
StatsBomb as the data source (and use their logo) if you publish this.
"""
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, FancyArrowPatch, Rectangle
from scipy.stats import spearmanr

from data import TEAMS
from network_extra import global_measures, removal_drop
from plot_networks import LAYOUT, tint
from recalculate import corrected_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
SB = os.path.join(HERE, "statsbomb")
FIG = os.path.join(HERE, "figures")
N = 11
RNG = np.random.default_rng(2011)

# (StatsBomb match id, StatsBomb team name) -> key in data.TEAMS
SIDES = {
    (18236, "Barcelona"): "FC Barcelona vs Man Utd (2011)",
    (18236, "Manchester United"): "Man Utd vs FC Barcelona (2011)",
    (18242, "Barcelona"): "FC Barcelona vs Juventus (2015)",
    (18242, "Juventus"): "Juventus vs FC Barcelona (2015)",
}
SHORT = {
    "FC Barcelona vs Man Utd (2011)": "Barça 2011",
    "Man Utd vs FC Barcelona (2011)": "Man Utd 2011",
    "FC Barcelona vs Juventus (2015)": "Barça 2015",
    "Juventus vs FC Barcelona (2015)": "Juventus 2015",
}
MATCH_TITLE = {18236: "2011 final: Barcelona 3–1 Manchester United",
               18242: "2015 final: Juventus 1–3 Barcelona"}


# ------------------------------------------------------------------ loading

def load(match_id):
    return json.load(open(os.path.join(SB, f"events_{match_id}.json"), encoding="utf-8"))


def starters(events, team):
    xi = next(e for e in events if e["type"]["name"] == "Starting XI" and e["team"]["name"] == team)
    return {p["player"]["id"]: p["jersey_number"] for p in xi["tactics"]["lineup"]}, xi["tactics"]["formation"]


def game_minute(e):
    """Match minute as StatsBomb records it (second half starts at 45:00)."""
    return e["minute"] + e["second"] / 60


def side_data(match_id, team, events=None, keep=lambda e: True):
    """Passing matrix between starters (completed passes), in data.py player order,
    plus average positions and pass counts."""
    events = events or load(match_id)
    key = SIDES[(match_id, team)]
    order = LAYOUT[key]["numbers"]                    # shirt numbers in data.py order
    xi, formation = starters(events, team)
    idx = {pid: order.index(num) for pid, num in xi.items()}
    A = np.zeros((N, N), dtype=int)
    locs = defaultdict(list)
    for e in events:
        if e["team"]["name"] != team or not keep(e):
            continue
        pid = e.get("player", {}).get("id")
        if e["type"]["name"] in ("Pass", "Ball Receipt*") and pid in idx and "location" in e:
            locs[idx[pid]].append(e["location"])
        if e["type"]["name"] != "Pass" or "outcome" in e["pass"]:
            continue
        rec = e["pass"].get("recipient", {}).get("id")
        if pid in idx and rec in idx and pid != rec:
            A[idx[pid], idx[rec]] += 1
    pos = np.array([np.mean(locs[i], axis=0) if locs[i] else [np.nan, np.nan] for i in range(N)])
    return dict(key=key, A=A, pos=pos, formation=formation, events=events, team=team, match=match_id)


# ------------------------------------------------------------- comparisons

def compare_with_opta(sd):
    opta = np.array(TEAMS[sd["key"]][1])
    sb = sd["A"]
    mask = ~np.eye(N, dtype=bool)
    r = np.corrcoef(opta[mask], sb[mask])[0, 1]
    names = TEAMS[sd["key"]][0]
    diffs = sorted(((sb[i, j] - opta[i, j], i, j) for i in range(N) for j in range(N) if i != j),
                   key=lambda t: -abs(t[0]))[:3]
    return opta.sum(), sb.sum(), r, [(names[i], names[j], int(opta[i, j]), int(sb[i, j])) for _, i, j in diffs]


def rank_names(values, names, k=3):
    return ", ".join(names[i] for i in np.argsort(values)[::-1][:k])


# --------------------------------------------------------------------- xG

def shots(events, team=None):
    out = []
    for e in events:
        if e["type"]["name"] == "Shot" and e["period"] <= 4 and (team is None or e["team"]["name"] == team):
            out.append(dict(team=e["team"]["name"], minute=game_minute(e), xg=e["shot"]["statsbomb_xg"],
                            goal=e["shot"]["outcome"]["name"] == "Goal", player=e["player"]["name"],
                            possession=e["possession"]))
    return out


def possession_xg(shot_list):
    """Chance of scoring at least once in each possession (rebounds are not independent chances)."""
    by = defaultdict(list)
    for s in shot_list:
        by[s["possession"]].append(s["xg"])
    return {p: 1 - np.prod([1 - x for x in xs]) for p, xs in by.items()}


def simulate(shots_a, shots_b, n=100_000):
    pa = np.array(list(possession_xg(shots_a).values()))
    pb = np.array(list(possession_xg(shots_b).values()))
    ga = (RNG.random((n, len(pa))) < pa).sum(1)
    gb = (RNG.random((n, len(pb))) < pb).sum(1)
    return ga, gb


def xg_chain(events, team, starter_ids):
    """xGChain: xG of every possession ending in a shot that the player touched.
    xGBuildup: the same, leaving out possessions in which the player took the shot
    or made the pass that set it up."""
    pxg = possession_xg(shots(events, team))
    touched = defaultdict(set)
    key_or_shot = defaultdict(set)
    assists = {e["pass"].get("assisted_shot_id") for e in events if e["type"]["name"] == "Pass"} - {None}
    for e in events:
        if e["team"]["name"] != team or e.get("possession") not in pxg:
            continue
        pid = e.get("player", {}).get("id")
        if pid is None or e["type"]["name"] in ("Pressure", "Duel", "Foul Committed"):
            continue
        if e["type"]["name"] == "Shot" or e["id"] in assists or (
                e["type"]["name"] == "Pass" and e["pass"].get("shot_assist")):
            key_or_shot[pid].add(e["possession"])
        else:
            touched[pid].add(e["possession"])
    chain, build = {}, {}
    for pid in starter_ids:
        allp = touched[pid] | key_or_shot[pid]
        chain[pid] = sum(pxg[p] for p in allp)
        build[pid] = sum(pxg[p] for p in allp - key_or_shot[pid])
    return chain, build


# ------------------------------------------------------------------ figures

PITCH_LINE = "#c9c9c9"


def draw_pitch(ax):
    """Vertical StatsBomb pitch (120 x 80), attacking upwards."""
    kw = dict(fill=False, edgecolor=PITCH_LINE, linewidth=1, zorder=0)
    ax.add_patch(Rectangle((0, 0), 80, 120, **kw))
    ax.plot([0, 80], [60, 60], color=PITCH_LINE, linewidth=1, zorder=0)
    ax.add_patch(Arc((40, 60), 20, 20, color=PITCH_LINE, linewidth=1, zorder=0))
    for y0, sgn in ((0, 1), (120, -1)):
        ax.add_patch(Rectangle((18, y0), 44, 18 * sgn, **kw))
        ax.add_patch(Rectangle((30, y0), 20, 6 * sgn, **kw))
        ax.add_patch(Arc((40, y0 + 12 * sgn), 20, 20, theta1=37 if sgn > 0 else 217,
                         theta2=143 if sgn > 0 else 323, color=PITCH_LINE, linewidth=1, zorder=0))
    ax.set_xlim(-3, 83)
    ax.set_ylim(-3, 123)
    ax.set_aspect("equal")
    ax.axis("off")


def separate(xy, r, gap=0.8, steps=200):
    """Nudge overlapping circles apart (only as far as needed) so every player is visible."""
    p = np.array(xy, dtype=float)
    for _ in range(steps):
        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                d = p[j] - p[i]
                dist = np.hypot(*d)
                need = r[i] + r[j] + gap
                if dist < need:
                    u = d / dist if dist > 1e-9 else np.array([1.0, 0.0])
                    shift = (need - dist) / 2 * u
                    p[i] -= shift
                    p[j] += shift
                    moved = True
        if not moved:
            break
    return [tuple(q) for q in p]


def draw_network(ax, sd, max_passes, min_passes=3, title=None, subtitle=None, label_names=True):
    key, A, pos = sd["key"], sd["A"], sd["pos"]
    names = TEAMS[key][0]
    colour, nums = LAYOUT[key]["colour"], LAYOUT[key]["numbers"]
    pr = corrected_metrics(A.tolist())["pagerank"]
    xy = [(p[1], p[0]) for p in pos]                   # (width, length) -> vertical pitch
    r = [1.8 + 11 * p for p in pr]
    xy = separate(xy, r)
    draw_pitch(ax)
    edges = sorted(((A[i, j], i, j) for i in range(N) for j in range(N) if A[i, j] >= min_passes))
    for w, i, j in edges:
        (x1, y1), (x2, y2) = xy[i], xy[j]
        d = np.hypot(x2 - x1, y2 - y1)
        ux, uy = (x2 - x1) / d, (y2 - y1) / d
        ax.add_patch(FancyArrowPatch(
            (x1 + ux * (r[i] + 0.6), y1 + uy * (r[i] + 0.6)), (x2 - ux * (r[j] + 1), y2 - uy * (r[j] + 1)),
            connectionstyle="arc3,rad=0.15",
            arrowstyle=f"-|>,head_length={2 + w * 0.08:.2f},head_width={1.2 + w * 0.05:.2f}",
            linewidth=0.3 + 0.22 * w, color=tint(colour, w / max_passes), zorder=1 + w / 100))
    for k, ((x, y), rr) in enumerate(zip(xy, r)):
        ax.add_patch(plt.Circle((x, y), rr, facecolor=colour, edgecolor="white", linewidth=1.8, zorder=5))
        ax.text(x, y, str(nums[k]), ha="center", va="center", color="white", fontsize=9,
                fontweight="bold", zorder=6)

    if label_names:
        placed = []                                     # label boxes already drawn: (x0, x1, y0, y1)
        for k in np.argsort([-p[1] for p in xy]):       # top to bottom
            x, y = xy[k]
            rr = r[k]
            half = 0.8 * len(names[k]) + 1              # rough label half-width in pitch units
            best = None
            for dy in (-(rr + 1.2), rr + 1.2, -(rr + 5.5), rr + 5.5):   # below, above, further out
                y0, y1 = (y + dy - 3.2, y + dy) if dy < 0 else (y + dy, y + dy + 3.2)
                box = (x - half, x + half, y0, y1)
                hits_label = any(not (box[1] < b[0] or box[0] > b[1] or box[3] < b[2] or box[2] > b[3])
                                 for b in placed)
                hits_node = any(j != k and abs(xy[j][0] - x) < half + r[j] and y0 - r[j] < xy[j][1] < y1 + r[j]
                                for j in range(N))
                if not hits_label and not hits_node:
                    best = (dy, box)
                    break
            dy, box = best or (-(rr + 1.2), (x - half, x + half, y - rr - 4.4, y - rr - 1.2))
            placed.append(box)
            ax.text(x, y + dy, names[k], ha="center", va="top" if dy < 0 else "bottom", fontsize=7.5,
                    color="#1a1a1a", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
    if title:
        ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color="#1a1a1a", pad=16)
    if subtitle:
        ax.text(0, 1.005, subtitle, transform=ax.transAxes, fontsize=8.5, color="#6b6b6b", va="bottom")


def width_legend(fig, max_passes, y):
    steps = (5, 15, 30) if max_passes >= 30 else (3, 8, 15)
    handles = [Line2D([], [], color=tint("#555555", min(w / max_passes, 1)), linewidth=0.3 + 0.22 * w,
                      label=f"{w} passes") for w in steps]
    handles.append(Line2D([], [], marker="o", linestyle="none", markersize=10, markerfacecolor="#555555",
                          markeredgecolor="white", label="bigger circle = higher PageRank"))
    leg = fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.5,
                     labelcolor="#6b6b6b", handlelength=2.6, bbox_to_anchor=(0.5, y))
    return leg


def fig_networks(sides):
    maxp = max(sd["A"].max() for sd in sides)
    fig, axes = plt.subplots(2, 2, figsize=(12, 15.5))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.925, bottom=0.06, wspace=0.04, hspace=0.1)
    for ax, sd in zip(axes.flat, sides):
        draw_network(ax, sd, maxp, title=SHORT[sd["key"]],
                     subtitle=f"{sd['A'].sum()} completed passes · attacking ↑")
    fig.suptitle("Passing networks at the players' real average positions", x=0.01, ha="left",
                 fontsize=16, fontweight="bold", y=0.985)
    fig.text(0.01, 0.955, "Starting XIs, completed passes between starters, whole match. Position = average "
             "location of the player's passes and receptions.", fontsize=10, color="#555")
    width_legend(fig, maxp, 0.022)
    fig.text(0.01, 0.006, "Data: StatsBomb Open Data. Links with fewer than 3 passes hidden. "
             "Overlapping circles nudged apart slightly.", fontsize=8.5, color="#6b6b6b")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"sb_networks.{ext}"), dpi=170, facecolor="white")
    plt.close(fig)


def fig_halves(match_id, team, events):
    halves = [side_data(match_id, team, events, keep=lambda e, p=p: e["period"] == p) for p in (1, 2)]
    maxp = max(h["A"].max() for h in halves)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 7.4))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.12, wspace=0.05)
    for ax, h, lab in zip(axes, halves, ("First half", "Second half")):
        draw_network(ax, h, maxp, min_passes=2, title=lab,
                     subtitle=f"{h['A'].sum()} completed passes between starters")
    key = SIDES[(match_id, team)]
    fig.suptitle(f"{SHORT[key]}: first half vs second half", x=0.01, ha="left", fontsize=14,
                 fontweight="bold")
    fig.text(0.01, 0.905, MATCH_TITLE[match_id], fontsize=9.5, color="#555")
    width_legend(fig, maxp, 0.045)
    fig.text(0.01, 0.01, "Data: StatsBomb Open Data. Links with fewer than 2 passes hidden. "
             "Overlapping circles nudged apart slightly.",
             fontsize=8, color="#6b6b6b")
    tag = "barca2015" if match_id == 18242 else "barca2011"
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"sb_halves_{tag}.{ext}"), dpi=170, facecolor="white")
    plt.close(fig)
    return halves


def fig_xg(matches):
    colours = {"Barcelona": "#1d3c8f", "Manchester United": "#c4122f", "Juventus": "#222222"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.8, bottom=0.14, wspace=0.08)
    for ax, (match_id, events) in zip(axes, matches.items()):
        end = max(game_minute(e) for e in events if e["period"] <= 2)
        teams = [t for (m, t) in SIDES if m == match_id]
        for t in teams:
            sh = sorted(shots(events, t), key=lambda s: s["minute"])
            xs, ys, total = [0], [0], 0
            for s in sh:
                xs += [s["minute"], s["minute"]]
                ys += [total, total + s["xg"]]
                total += s["xg"]
            xs.append(end)
            ys.append(total)
            ax.plot(xs, ys, color=colours[t], linewidth=2, drawstyle="default")
            for s in sh:
                if s["goal"]:
                    y = sum(x["xg"] for x in sh if x["minute"] <= s["minute"])
                    ax.plot(s["minute"], y, "o", color=colours[t], markersize=8,
                            markeredgecolor="white", markeredgewidth=1.5, zorder=5)
                    ax.annotate(s["player"].split()[0] if s["player"].split()[0] not in ("Lionel", "Luis", "Álvaro", "Wayne", "Pedro", "David", "Ivan", "Neymar")
                                else {"Lionel": "Messi", "Luis": "Suárez", "Álvaro": "Morata", "Wayne": "Rooney",
                                      "Pedro": "Pedro", "David": "Villa", "Ivan": "Rakitić", "Neymar": "Neymar"}[s["player"].split()[0]],
                                (s["minute"], y), xytext=(-4, 7) if s["minute"] > 10 else (6, -4),
                                textcoords="offset points", ha="right" if s["minute"] > 10 else "left",
                                va="bottom" if s["minute"] > 10 else "top",
                                fontsize=8, color="#333")
            ax.text(end + 2.5, total, f"{t.replace('Manchester United', 'Man Utd')} {total:.2f}",
                    color=colours[t], fontsize=9, va="center", fontweight="bold")
        ax.axvline(45, color="#ddd", linewidth=1, zorder=0)
        ax.set_xlim(0, end + 24)
        ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
        ax.set_title(MATCH_TITLE[match_id], loc="left", fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#eee")
        ax.set_xlabel("Minute", fontsize=9, color="#444")
    axes[0].set_ylabel("Cumulative expected goals (xG)", fontsize=9, color="#444")
    fig.suptitle("How good were the chances? Expected goals through each final", x=0.01, ha="left",
                 fontsize=14, fontweight="bold")
    fig.text(0.01, 0.86, "Each step is a shot, sized by its chance of going in. Dots mark goals. "
             "Data: StatsBomb Open Data.", fontsize=9.5, color="#555")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"sb_xg_timeline.{ext}"), dpi=170, facecolor="white")
    plt.close(fig)


# -------------------------------------------------------------------- main

def main():
    os.makedirs(FIG, exist_ok=True)
    events = {m: load(m) for m in (18236, 18242)}
    sides = [side_data(m, t, events[m]) for (m, t) in SIDES]

    print("# The finals with StatsBomb event data\n")
    print("Generated by `statsbomb_analysis.py`. Data: [StatsBomb Open Data](https://github.com/statsbomb/open-data) "
          "(credit StatsBomb and use their logo if you publish this). Passing matrices count completed passes "
          "between the eleven starters over the whole match, as in the essay.\n")

    # 1. data check
    print("## 1. Does StatsBomb agree with the essay's Opta tables?\n")
    print("| Team | Formation (StatsBomb) | Opta passes | StatsBomb passes | Correlation of the 110 links | Biggest differences (Opta → StatsBomb) |")
    print("|---|---|---|---|---|---|")
    for sd in sides:
        o, s, r, d = compare_with_opta(sd)
        f = "-".join(str(sd["formation"]))
        diffs = "; ".join(f"{a} → {b}: {x} → {y}" for a, b, x, y in d)
        print(f"| {SHORT[sd['key']]} | {f} | {o} | {s} | {r:.2f} | {diffs} |")
    print()

    # 2. positions & network measures
    print("## 2. Real average positions\n")
    print("![Passing networks](figures/sb_networks.png)\n")
    print("Average position of each player (metres from own goal line, along the pitch / from the left touchline):\n")
    fig_networks(sides)
    for sd in sides:
        names = TEAMS[sd["key"]][0]
        p = sd["pos"] * [105 / 120, 68 / 80]               # StatsBomb units -> approx. metres
        line = ", ".join(f"{names[i]} {p[i, 0]:.0f}/{p[i, 1]:.0f}" for i in np.argsort(-p[:, 0]))
        print(f"- **{SHORT[sd['key']]}** (most advanced first): {line}")
    print()

    print("## 3. Network results: Opta tables vs StatsBomb\n")
    print("Same corrected measures as `RESULTS.md` (1/passes distance; weighted PageRank).\n")
    print("| Team | Data | Top betweenness | Top PageRank | Most needed (red-card test) | Betweenness centralisation | Modularity |")
    print("|---|---|---|---|---|---|---|")
    for sd in sides:
        names = TEAMS[sd["key"]][0]
        for label, A in (("Opta", np.array(TEAMS[sd["key"]][1])), ("StatsBomb", sd["A"])):
            m = corrected_metrics(A.tolist())
            g = global_measures(A.tolist())
            drops = [removal_drop(A.tolist(), (i,)) for i in range(N)]
            top = np.argsort(drops)[::-1][:2]
            rc = ", ".join(f"{names[i]} {drops[i]:.0%}" for i in top)
            print(f"| {SHORT[sd['key']]} | {label} | {rank_names(m['betweenness'], names)} | "
                  f"{rank_names(m['pagerank'], names)} | {rc} | {g['btw_centralisation']:.3f} | {g['modularity']:.3f} |")
    print()

    # 4. halves
    print("## 4. First half vs second half\n")
    print("| Team | Half | Completed passes | Share of team's passes in final third | Top PageRank | Betweenness centralisation |")
    print("|---|---|---|---|---|---|")
    for (m, t) in SIDES:
        key = SIDES[(m, t)]
        names = TEAMS[key][0]
        for half in (1, 2):
            h = side_data(m, t, events[m], keep=lambda e, p=half: e["period"] == p)
            passes = [e for e in events[m] if e["type"]["name"] == "Pass" and e["team"]["name"] == t
                      and e["period"] == half and "outcome" not in e["pass"]]
            final_third = np.mean([e["location"][0] >= 80 for e in passes])
            mm = corrected_metrics(h["A"].tolist())
            g = global_measures(h["A"].tolist())
            print(f"| {SHORT[key]} | {half} | {h['A'].sum()} | {final_third:.0%} | "
                  f"{rank_names(mm['pagerank'], names, 2)} | {g['btw_centralisation']:.3f} |")
    print()
    fig_halves(18236, "Barcelona", events[18236])
    fig_halves(18242, "Barcelona", events[18242])
    print("![Barça 2011 by half](figures/sb_halves_barca2011.png)\n")
    print("![Barça 2015 by half](figures/sb_halves_barca2015.png)\n")

    # 5. xG
    print("## 5. Expected goals\n")
    fig_xg(events)
    print("![xG timeline](figures/sb_xg_timeline.png)\n")
    print("Each shot's xG is StatsBomb's estimate of the chance it goes in. Shots in the same possession "
          "(rebounds) are combined so one move cannot count as several independent chances. The simulation "
          "replays every chance 100,000 times to ask: *given the chances each team actually created, how often "
          "would this match end in each result?*\n")
    print("| Final | Team | Shots | xG | Goals | P(win) from chances | P(draw) | P(exact score) | Pre-match model P(win) |")
    print("|---|---|---|---|---|---|---|---|---|")
    pre = {18236: (0.240, 0.410, 0.351), 18242: (0.380, 0.336, 0.283)}   # version B, RESULTS.md (Barça, draw, opp.)
    for m in (18236, 18242):
        barca = shots(events[m], "Barcelona")
        opp_name = "Manchester United" if m == 18236 else "Juventus"
        opp = shots(events[m], opp_name)
        gb, go = simulate(barca, opp)
        actual = (sum(s["goal"] for s in barca), sum(s["goal"] for s in opp))
        exact = np.mean((gb == actual[0]) & (go == actual[1]))
        pw_b, pd_, pw_o = np.mean(gb > go), np.mean(gb == go), np.mean(gb < go)
        for t, sh, pw, pr_ in (("Barcelona", barca, pw_b, pre[m][0]), (opp_name, opp, pw_o, pre[m][2])):
            print(f"| {'2011' if m == 18236 else '2015'} | {t} | {len(sh)} | {sum(s['xg'] for s in sh):.2f} | "
                  f"{sum(s['goal'] for s in sh)} | {pw:.0%} | {pd_:.0%} | {exact:.1%} | {pr_:.0%} |")
    print()

    # 6. network <-> chances
    print("## 6. Does network importance lead to chances?\n")
    print("**xGChain** = total xG of the shot-ending possessions a player was involved in. **xGBuildup** = the same "
          "but leaving out possessions in which the player took the shot or made the final pass, so it measures "
          "the build-up. Spearman ρ compares each team's ranking of players by PageRank (StatsBomb network) with "
          "their ranking by xGBuildup (n = 11, so only large values mean much).\n")
    for sd in sides:
        m, t, key = sd["match"], sd["team"], sd["key"]
        names = TEAMS[key][0]
        xi, _ = starters(events[m], t)
        order = LAYOUT[key]["numbers"]
        chain, build = xg_chain(events[m], t, xi)
        by_idx_chain = np.zeros(N)
        by_idx_build = np.zeros(N)
        for pid, num in xi.items():
            by_idx_chain[order.index(num)] = chain[pid]
            by_idx_build[order.index(num)] = build[pid]
        pr = np.array(corrected_metrics(sd["A"].tolist())["pagerank"])
        rho, p = spearmanr(pr, by_idx_build)
        top = np.argsort(-by_idx_chain)[:4]
        print(f"- **{SHORT[key]}**: highest xGChain {', '.join(f'{names[i]} {by_idx_chain[i]:.2f}' for i in top)}; "
              f"highest xGBuildup {', '.join(f'{names[i]} {by_idx_build[i]:.2f}' for i in np.argsort(-by_idx_build)[:3])}. "
              f"PageRank vs xGBuildup: ρ = {rho:.2f} (p = {p:.2f}).")
    print()


if __name__ == "__main__":
    os.chdir(HERE)
    main()
