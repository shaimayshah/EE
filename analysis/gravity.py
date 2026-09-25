"""Messi gravity, steps 1-2: expected crowding baseline and player gravity scores.

Crowding = number of opponents within R metres-ish (StatsBomb units, 120x80 pitch)
of the ball-carrier at the moment of an on-ball action, from the 360 freeze frame.

Gravity = mean(actual crowding - expected crowding), where expected crowding is the
average for the same pitch zone and action type across all *other* players
(leave-player-out, so a player never sets their own baseline).

Usage: .venv/bin/python analysis/gravity.py
"""
import glob
import json
import math
import os

import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

# Point the whole pipeline at another competition with env vars, e.g.
#   SB_DATA=data/laliga/90 SB_OUT=output/laliga_2021 SB_LABEL="La Liga 2020/21" SB_TEAM=Barcelona
DATA = os.environ.get("SB_DATA", "data/wc2022")
OUT = os.environ.get("SB_OUT", "output")
LABEL = os.environ.get("SB_LABEL", "World Cup 2022")
TEAM = os.environ.get("SB_TEAM", "Argentina")
RADII = (3, 5, 10)
ON_BALL = {"Pass", "Carry", "Ball Receipt*", "Dribble", "Shot"}
SET_PIECE_PASSES = {"Corner", "Free Kick", "Throw-in", "Kick Off", "Goal Kick"}
MIN_ACTIONS = 300
N_BOOT = 2000


def is_open_play(e):
    t = e["type"]["name"]
    if t == "Pass":
        return e["pass"].get("type", {}).get("name") not in SET_PIECE_PASSES
    if t == "Shot":
        return e["shot"]["type"]["name"] == "Open Play"
    if t == "Ball Receipt*":
        return "ball_receipt" not in e  # outcome present only when the ball wasn't received
    return True


def load_actions():
    matches = {m["match_id"]: m for m in json.load(open(f"{DATA}/matches.json"))}
    rows = []
    for path in sorted(glob.glob(f"{DATA}/events/*.json")):
        mid = int(os.path.basename(path)[:-5])
        frames = {f["event_uuid"]: f for f in json.load(open(f"{DATA}/three-sixty/{mid}.json"))}
        events = json.load(open(path))
        # Link each ball receipt back to the pass that found it.
        pass_for = {r: e for e in events if e["type"]["name"] == "Pass"
                    for r in e.get("related_events", [])}
        for e in events:
            if e["type"]["name"] not in ON_BALL or "location" not in e or not is_open_play(e):
                continue
            f = frames.get(e["id"])
            if f is None:
                continue
            x, y = e["location"][:2]
            # Drop the few frames whose flagged actor doesn't sit on the event location.
            actor = [p for p in f["freeze_frame"] if p["actor"]]
            if actor and math.dist(actor[0]["location"], (x, y)) > 2:
                continue
            opps = [p["location"] for p in f["freeze_frame"] if not p["teammate"]]
            dists = [math.dist(o, (x, y)) for o in opps]
            va = f["visible_area"]
            area = Polygon(list(zip(va[0::2], va[1::2]))).buffer(0)
            row = {
                "match_id": mid,
                "stage": matches[mid]["competition_stage"]["name"],
                "team": e["team"]["name"],
                "player": e["player"]["name"],
                "type": e["type"]["name"],
                "minute": e["minute"],
                "x": x,
                "y": y,
                "nearest_opp": min(dists) if dists else np.nan,
                "passer": None,
                "pass_height": None,
                "pass_length": np.nan,
            }
            p = pass_for.get(e["id"])
            if e["type"]["name"] == "Ball Receipt*" and p is not None:
                row.update(passer=p["player"]["name"], pass_height=p["pass"]["height"]["name"],
                           pass_length=p["pass"]["length"])
            for r in RADII:
                row[f"opp_{r}"] = sum(d < r for d in dists)
                # Only trust the count if the camera saw the whole circle.
                row[f"vis_{r}"] = area.contains(Point(x, y).buffer(r))
            rows.append(row)
    return pd.DataFrame(rows)


def add_expected(df, r, zone=10):
    """Leave-player-out mean crowding for the same zone x action type."""
    d = df[df[f"vis_{r}"]].copy()
    d["zone"] = (d.x // zone).astype(int).astype(str) + "_" + (d.y // zone).astype(int).astype(str)
    key = ["zone", "type"]
    cell = d.groupby(key)[f"opp_{r}"].agg(["sum", "count"])
    mine = d.groupby(key + ["player"])[f"opp_{r}"].agg(["sum", "count"])
    d = d.join(cell, on=key).join(mine, on=key + ["player"], rsuffix="_me")
    n_other = d["count"] - d["count_me"]
    d["expected"] = (d["sum"] - d["sum_me"]) / n_other
    d = d[n_other >= 20]  # need a real baseline for the cell
    d["resid"] = d[f"opp_{r}"] - d["expected"]
    return d.drop(columns=["sum", "count", "sum_me", "count_me"])


def leaderboard(d, r, rng):
    out = []
    for (player, team), g in d.groupby(["player", "team"]):
        if len(g) < MIN_ACTIONS:
            continue
        # Bootstrap over matches: a player's actions within one match aren't independent.
        per_match = g.groupby("match_id")["resid"].agg(["sum", "count"]).to_numpy()
        idx = rng.integers(0, len(per_match), size=(N_BOOT, len(per_match)))
        boot = per_match[idx, 0].sum(1) / per_match[idx, 1].sum(1)
        out.append({
            "player": player, "team": team, "matches": len(per_match), "actions": len(g),
            "actual": g[f"opp_{r}"].mean(), "expected": g["expected"].mean(),
            "gravity": g["resid"].mean(),
            "ci_low": np.percentile(boot, 2.5), "ci_high": np.percentile(boot, 97.5),
        })
    lb = pd.DataFrame(out).sort_values("gravity", ascending=False).reset_index(drop=True)
    lb.index += 1
    return lb


def short(name):
    known = {"Lionel Andrés Messi Cuccittini": "Messi", "Kylian Mbappé Lottin": "Mbappé",
             "Neymar da Silva Santos Junior": "Neymar",
             "Joel Nathaniel Campbell Samuels": "Joel Campbell",
             "Bernardo Mota Veiga de Carvalho e Silva": "Bernardo Silva",
             "Theo Bernard François Hernández": "Theo Hernández",
             "Rúben Santos Gato Alves Dias": "Rúben Dias",
             "Rúben Diogo Da Silva Neves": "Rúben Neves",
             "Nicolás Hernán Otamendi": "Otamendi",
             "Rodrigo Hernández Cascante": "Rodri",
             "Marcos Aoás Corrêa": "Marquinhos",
             "Kléper Laveran Lima Ferreira": "Pepe",
             "Dayotchanculle Upamecano": "Upamecano",
             "Ángel Fabián Di María Hernández": "Di María",
             "Lautaro Javier Martínez": "Lautaro Martínez",
             "Nahuel Molina Lucero": "Molina",
             "Bruno Miguel Borges Fernandes": "Bruno Fernandes",
             "Christian Dannemann Eriksen": "Eriksen",
             "Raphael Dias Belloli": "Raphinha",
             "Hirving Rodrigo Lozano Bahena": "Hirving Lozano",
             "Enzo Fernandez": "Enzo Fernández",
             "Cristiano Ronaldo dos Santos Aveiro": "Ronaldo"}
    return known.get(name, name if len(name) <= 22 else " ".join(name.split()[:2]))


def plot_leaderboard(lb, r, path, top=25):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = lb.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 9))
    colors = ["#75AADB" if p.startswith("Lionel Andrés Messi") else "#9aa0a6" for p in t.player]
    ax.barh(range(len(t)), t.gravity, color=colors,
            xerr=[t.gravity - t.ci_low, t.ci_high - t.gravity], ecolor="#555", capsize=2)
    ax.set_yticks(range(len(t)), [f"{short(p)} ({tm})" for p, tm in zip(t.player, t.team)], fontsize=8)
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlabel(f"Gravity: extra opponents within {r} units vs. expected for zone & action")
    ax.set_title(f"{LABEL} gravity leaderboard (top {top}, ≥{MIN_ACTIONS} actions)\n"
                 "error bars: 95% bootstrap CI over matches", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_messi_heatmap(d, path, zone=20):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = d[d.player.str.startswith("Lionel Andrés Messi")].copy()
    nx, ny = 120 // zone, 80 // zone
    grid = np.full((ny, nx), np.nan)
    counts = np.zeros((ny, nx), int)
    for (i, j), g in m.groupby([(m.x // zone).clip(0, nx - 1).astype(int),
                                (m.y // zone).clip(0, ny - 1).astype(int)]):
        counts[j, i] = len(g)
        if len(g) >= 15:
            grid[j, i] = g.resid.mean()
    fig, ax = plt.subplots(figsize=(9, 6))
    lim = np.nanmax(np.abs(grid))
    im = ax.imshow(grid, extent=(0, 120, 80, 0), cmap="RdBu_r", vmin=-lim, vmax=lim)
    for j in range(ny):
        for i in range(nx):
            if counts[j, i]:
                label = f"{grid[j, i]:+.2f}\n(n={counts[j, i]})" if not np.isnan(grid[j, i]) else f"n={counts[j, i]}"
                ax.text(i * zone + zone / 2, j * zone + zone / 2, label, ha="center", va="center", fontsize=8)
    ax.set_title("Messi gravity by pitch zone (attacking →)\nextra opponents within 5 units vs. expected; zones with n<15 blank")
    ax.set_xlim(0, 120)
    ax.set_ylim(80, 0)
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(2022)
    df = load_actions()
    df.to_csv(f"{OUT}/actions.csv.gz", index=False)
    print(f"{len(df)} open-play on-ball actions with a usable 360 frame")

    messi = "Lionel Andrés Messi Cuccittini"
    boards = {}
    for r in RADII:
        for zone in (10, 20):
            d = add_expected(df, r, zone)
            lb = leaderboard(d, r, rng)
            boards[(r, zone)] = lb
            row = lb[lb.player == messi].iloc[0]
            print(f"R={r:2} zone={zone}: {len(lb)} players | Messi rank {row.name:3}  "
                  f"gravity {row.gravity:+.3f} [{row.ci_low:+.3f}, {row.ci_high:+.3f}]  "
                  f"({row.actual:.2f} actual vs {row.expected:.2f} expected)")

    main_lb = boards[(5, 10)]
    main_lb.to_csv(f"{OUT}/gravity_leaderboard.csv")
    plot_leaderboard(main_lb, 5, f"{OUT}/gravity_leaderboard.png")
    plot_messi_heatmap(add_expected(df, 5, 10), f"{OUT}/messi_gravity_heatmap.png")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(main_lb.head(20).round(3).to_string())


if __name__ == "__main__":
    main()
