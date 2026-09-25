"""Messi gravity, step 5: does Messi pass into space ahead of the receiver?

For every completed open-play pass with a 360 frame at both the pass and the ball
receipt, we locate the receiver in the *pass* frame (the teammate nearest the spot
where the ball was received, only when that match is unambiguous) and measure:

  lead        distance the receiver travelled from pass to receipt (to feet ~ 0)
  lead_fwd    the forward (x) part of that movement
  gain        receiver's nearest-opponent distance at receipt minus at the pass
              (capped at 5; only when the camera saw the whole 5-unit circle)
  speed       average ball speed, pass length / pass duration (units per second)

Each is compared with a leave-passer-out baseline for the same receiving zone,
pass height and pass length (as in teammate_space.py). Then we ask how much of
Messi's gain is explained by how far he leads the receiver.

Usage: .venv/bin/python analysis/lead_passes.py
"""
import glob
import json
import math
import os

import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

import gravity

DATA = gravity.DATA
OUT = gravity.OUT
MESSI = "Lionel Andrés Messi Cuccittini"
CAP = 5
MATCH_MAX = 12  # receiver must be within this of the receipt spot at the pass frame...
MATCH_MARGIN = 3  # ...and this much closer than the next teammate
MIN_N = 80
N_BOOT = 2000


def area_of(frame):
    va = frame["visible_area"]
    return Polygon(list(zip(va[0::2], va[1::2]))).buffer(0)


def open_at(frame, area, xy):
    """Capped nearest-opponent distance at xy, or NaN if the camera didn't see the circle."""
    if not area.contains(Point(xy).buffer(CAP)):
        return np.nan
    d = [math.dist(p["location"], xy) for p in frame["freeze_frame"] if not p["teammate"]]
    return min(min(d), CAP) if d else CAP


def load():
    rows = []
    for path in sorted(glob.glob(f"{DATA}/events/*.json")):
        mid = int(os.path.basename(path)[:-5])
        events = json.load(open(path))
        by_id = {e["id"]: e for e in events}
        frames = {f["event_uuid"]: f for f in json.load(open(f"{DATA}/three-sixty/{mid}.json"))}
        for e in events:
            if e["type"]["name"] != "Pass" or "outcome" in e["pass"] or not gravity.is_open_play(e):
                continue
            rec = next((by_id[r] for r in e.get("related_events", [])
                        if by_id.get(r, {}).get("type", {}).get("name") == "Ball Receipt*"), None)
            if rec is None or e["id"] not in frames or rec["id"] not in frames:
                continue
            fp, fr = frames[e["id"]], frames[rec["id"]]
            x, y = e["location"]
            rx, ry = rec["location"]
            actor = [p for p in fp["freeze_frame"] if p["actor"]]
            if actor and math.dist(actor[0]["location"], (x, y)) > 2:
                continue
            mates = [p["location"] for p in fp["freeze_frame"]
                     if p["teammate"] and not p["actor"] and not p["keeper"]]
            if not mates:
                continue
            d = sorted((math.dist(m, (rx, ry)), i) for i, m in enumerate(mates))
            if d[0][0] > MATCH_MAX or (len(d) > 1 and d[1][0] - d[0][0] < MATCH_MARGIN):
                continue
            px, py = mates[d[0][1]]
            open_pass = open_at(fp, area_of(fp), (px, py))
            open_rec = open_at(fr, area_of(fr), (rx, ry))
            dur = e.get("duration") or 0
            rows.append({
                "match_id": mid, "team": e["team"]["name"], "player": e["player"]["name"],
                "receiver": rec["player"]["name"], "x": x, "y": y, "rx": rx, "ry": ry,
                "length": e["pass"]["length"], "height": e["pass"]["height"]["name"],
                "lead": d[0][0], "lead_fwd": rx - px,
                "open_pass": open_pass, "open_rec": open_rec, "gain": open_rec - open_pass,
                "speed": e["pass"]["length"] / dur if dur > 0.2 else np.nan,
            })
    return pd.DataFrame(rows)


def adjust(df, col, zone=10):
    """Leave-passer-out residual vs same receiving zone x pass height x length bin."""
    d = df[df[col].notna()].copy()
    d["zone"] = (d.rx // zone).astype(int).astype(str) + "_" + (d.ry // zone).astype(int).astype(str)
    d["len_bin"] = pd.cut(d.length, [0, 10, 20, 35, 200], labels=False)
    key = ["zone", "height", "len_bin"]
    cell = d.groupby(key)[col].agg(["sum", "count"])
    mine = d.groupby(key + ["player"])[col].agg(["sum", "count"])
    j = d[key + ["player"]].join(cell, on=key).join(mine, on=key + ["player"], rsuffix="_me")
    n = j["count"] - j["count_me"]
    d["expected"] = (j["sum"] - j["sum_me"]) / n
    d["resid"] = d[col] - d["expected"]
    return d[n >= 20]


def summarise(d, col, rng, min_n=MIN_N):
    rows = []
    for (player, team), g in d.groupby(["player", "team"]):
        if len(g) < min_n:
            continue
        pm = g.groupby("match_id")["resid"].agg(["sum", "count"]).to_numpy()
        idx = rng.integers(0, len(pm), size=(N_BOOT, len(pm)))
        boot = pm[idx, 0].sum(1) / pm[idx, 1].sum(1)
        rows.append({"player": player, "team": team, "n": len(g), "actual": g[col].mean(),
                     "value": g.resid.mean(), "ci_low": np.percentile(boot, 2.5),
                     "ci_high": np.percentile(boot, 97.5)})
    b = pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)
    b.index += 1
    return b


def line(name, b):
    m = b[b.player == MESSI]
    if m.empty:
        return f"{name:<44} Messi below threshold"
    r = m.iloc[0]
    return (f"{name:<44} Messi {r.actual:6.2f} (vs expected {r.actual - r.value:5.2f})  "
            f"diff {r.value:+.2f} [{r.ci_low:+.2f}, {r.ci_high:+.2f}]  rank {r.name}/{len(b)}")


def plot(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df[df.gain.notna()].copy()
    bins = [0, 2, 4, 6, 8, 12]
    d["lb"] = pd.cut(d.lead, bins)
    rest = d[d.player != MESSI].groupby("lb", observed=True).gain.agg(["mean", "count"])
    me = d[d.player == MESSI].groupby("lb", observed=True).gain.agg(["mean", "count"])
    mids = [(a + b) / 2 for a, b in zip(bins[:-1], bins[1:])]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), facecolor="#fcfcfb",
                                 gridspec_kw={"width_ratios": [1, 1.2]})
    for ax in (a1, a2):
        ax.set_facecolor("#fcfcfb")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fb = [-12, -2, 2, 6, 12]
    fl = ["backward\n(−12 to −2)", "square / to feet\n(−2 to 2)", "ahead\n(2 to 6)", "well ahead\n(6 to 12)"]
    f = df[df.lead_fwd.between(-12, 12)]
    share_me = pd.cut(f[f.player == MESSI].lead_fwd, fb).value_counts(normalize=True).sort_index()
    share_rest = pd.cut(f[f.player != MESSI].lead_fwd, fb).value_counts(normalize=True).sort_index()
    xs = np.arange(len(fl))
    w = 0.38
    a1.bar(xs - w / 2, share_rest.values, width=w, color="#9aa0a6", label="everyone else")
    a1.bar(xs + w / 2, share_me.values, width=w, color="#2a78d6", label="Messi")
    a1.set_xticks(xs, fl, fontsize=8)
    a1.set_xlabel("Forward lead: how far forward the receiver moved to reach the ball (units)")
    a1.set_ylabel("Share of completed passes")
    a1.set_title("Does the pass lead the receiver forward?", fontsize=10, loc="left")
    a1.legend(fontsize=8, frameon=False)
    a2.plot(mids[:len(rest)], rest["mean"], "-o", color="#9aa0a6", lw=2, ms=8, label="everyone else")
    a2.plot([mids[i] for i in range(len(me)) if me["count"].iloc[i] >= 10],
            me["mean"][me["count"] >= 10], "-o", color="#2a78d6", lw=2, ms=8, label="Messi (bins with ≥10 passes)")
    for mx, (mv, n) in zip(mids, me[["mean", "count"]].itertuples(index=False)):
        if n >= 10:
            a2.annotate(f"n={n}", (mx, mv), xytext=(0, 8), textcoords="offset points", ha="center",
                        fontsize=7, color="#52514e")
    a2.axhline(0, color="#c9c8c2", lw=0.8)
    a2.set_xticks(mids, ["0–2", "2–4", "4–6", "6–8", "8–12"], fontsize=8)
    a2.set_xlabel("Lead (units)")
    a2.set_ylabel("Space gain: nearest defender at receipt − at pass (units)")
    a2.set_title("Does the receiver end up with more room?", fontsize=10, loc="left")
    a2.legend(fontsize=8, frameon=False)
    fig.suptitle(f"Weight and timing: Messi's completed passes, {gravity.LABEL}", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    rng = np.random.default_rng(2022)
    df = load()
    m = df[df.player == MESSI]
    print(f"{len(df)} completed passes with receiver matched in the pass frame "
          f"(Messi {len(m)}); with space measured at both moments: {df.gain.notna().sum()} (Messi {m.gain.notna().sum()})")

    boards = {}
    for col, label in [("lead", "lead (units the receiver moved)"), ("lead_fwd", "forward part of lead"),
                       ("open_pass", "receiver's space at the pass"), ("open_rec", "receiver's space at receipt"),
                       ("gain", "space gain (receipt - pass)"), ("speed", "ball speed (units/s)")]:
        boards[col] = summarise(adjust(df, col), col, rng)
        print(line(label, boards[col]))

    # How much of Messi's gain do his lead distances explain?
    d = df[df.gain.notna()].copy()
    d["lb"] = pd.cut(d.lead, [0, 2, 4, 6, 8, 12])
    by_lead = d[d.player != MESSI].groupby("lb", observed=True).gain.mean()
    me = d[d.player == MESSI]
    exp_from_lead = me.lb.map(by_lead).astype(float).mean()
    print(f"\nMessi's gain {me.gain.mean():+.2f} vs everyone else {d[d.player != MESSI].gain.mean():+.2f}; "
          f"expected from his lead distances alone {exp_from_lead:+.2f}")
    print("gain by lead bin (everyone else vs Messi):")
    print(pd.DataFrame({"others": by_lead, "messi": me.groupby("lb", observed=True).gain.mean(),
                        "messi_n": me.groupby("lb", observed=True).size()}).round(2).to_string())

    top = boards["lead_fwd"].head(10)[["player", "team", "n", "value", "ci_low", "ci_high"]]
    print("\nTop 10 by forward lead vs expected:")
    print(top.round(2).to_string())
    pd.concat({k: v.set_index("player")[["n", "value", "ci_low", "ci_high"]] for k, v in boards.items()},
              axis=1).to_csv(f"{OUT}/lead_passes_leaderboard.csv")
    plot(df, f"{OUT}/lead_passes.png")


if __name__ == "__main__":
    main()
