"""Messi gravity, step 4: creation vs selection, and how hard his passes are.

For every open-play pass with a 360 frame (all players) we compute, at the moment of
the pass:

  * creation:  how open are *all* visible outfield teammates (nearest-opponent
               distance, capped at 5, only counted when the camera saw their whole
               5-unit circle). If Messi drags defenders, everyone should be more
               open, not just the player he picks.
  * selection: for completed passes, how open the chosen receiver was minus the
               average openness of the other visible options within 40 units.
               Tells us whether he picks the open man better than others do.
  * difficulty: an expected-pass-completion model (xPass), fitted with match-
               grouped cross-validation so no pass is scored by a model that saw
               its own match. Difficulty = 1 - xPass; skill = completion - xPass.

Each measure is compared with a leave-player-out baseline for the same ball zone.

Usage: .venv/bin/python analysis/pass_quality.py
"""
import glob
import json
import math
import os

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Polygon
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

import gravity

DATA = gravity.DATA
OUT = gravity.OUT
MESSI = "Lionel Andrés Messi Cuccittini"
CAP = 5
OPTION_RANGE = 40
FAIL = {"Incomplete", "Out", "Pass Offside"}
MIN_PASSES = 150
MIN_CHOICES = 60
MATCH_RADIUS = 8
N_BOOT = 2000


def lane_opponents(start, end, opps, frac=0.7, width=2.0):
    """Opponents within `width` of the first `frac` of the pass line.

    The last stretch is left out so an intercepting defender, who by definition
    sits at the recorded end of an incomplete pass, doesn't leak the outcome.
    """
    (x0, y0), (x1, y1) = start, end
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy
    if L2 == 0:
        return 0
    n = 0
    for ox, oy in opps:
        t = ((ox - x0) * dx + (oy - y0) * dy) / L2
        if 0 < t <= frac and math.hypot(ox - (x0 + t * dx), oy - (y0 + t * dy)) < width:
            n += 1
    return n


def openness(frame, area, boundary):
    """Capped nearest-opponent distance for each visible outfield teammate."""
    opps = np.array([p["location"] for p in frame["freeze_frame"] if not p["teammate"]])
    mates = [p for p in frame["freeze_frame"] if p["teammate"] and not p["actor"] and not p["keeper"]]
    if not mates:
        return np.empty((0, 2)), np.empty(0)
    pos = np.array([p["location"] for p in mates])
    if len(opps):
        near = np.sqrt(((pos[:, None, :] - opps[None, :, :]) ** 2).sum(-1)).min(1)
    else:
        near = np.full(len(pos), np.inf)
    pts = shapely.points(pos)
    seen = shapely.contains(area, pts) & (shapely.distance(boundary, pts) >= CAP)
    return pos, np.where(seen, np.minimum(near, CAP), np.nan)


def load_passes():
    rows = []
    for path in sorted(glob.glob(f"{DATA}/events/*.json")):
        mid = int(os.path.basename(path)[:-5])
        frames = {f["event_uuid"]: f for f in json.load(open(f"{DATA}/three-sixty/{mid}.json"))}
        for e in json.load(open(path)):
            if e["type"]["name"] != "Pass" or not gravity.is_open_play(e) or e["id"] not in frames:
                continue
            p = e["pass"]
            outcome = p.get("outcome", {}).get("name")
            if outcome and outcome not in FAIL:
                continue  # Unknown / Injury Clearance
            f = frames[e["id"]]
            x, y = e["location"]
            ex, ey = p["end_location"]
            actor = [q for q in f["freeze_frame"] if q["actor"]]
            if actor and math.dist(actor[0]["location"], (x, y)) > 2:
                continue
            opps = [q["location"] for q in f["freeze_frame"] if not q["teammate"]]
            d_passer = [math.dist(o, (x, y)) for o in opps]
            d_end = [math.dist(o, (ex, ey)) for o in opps]
            va = f["visible_area"]
            area = Polygon(list(zip(va[0::2], va[1::2]))).buffer(0)
            pos, op = openness(f, area, area.boundary)

            # Options = visible outfield teammates within range; chosen = nearest to the end point.
            rng_mask = np.hypot(pos[:, 0] - x, pos[:, 1] - y) <= OPTION_RANGE if len(pos) else np.zeros(0, bool)
            chosen_open = gap = np.nan
            picked_most_open = np.nan
            if not outcome and len(pos):
                dist_end = np.hypot(pos[:, 0] - ex, pos[:, 1] - ey)
                k = int(dist_end.argmin())
                others = rng_mask.copy()
                others[k] = False
                if dist_end[k] < MATCH_RADIUS and not np.isnan(op[k]) and np.isfinite(op[others]).any():
                    chosen_open = op[k]
                    gap = op[k] - np.nanmean(op[others])
                    picked_most_open = float(op[k] >= np.nanmax(op[others]))

            opts = op[rng_mask] if len(pos) else np.empty(0)
            rows.append({
                "match_id": mid, "team": e["team"]["name"], "player": e["player"]["name"],
                "x": x, "y": y, "end_x": ex, "end_y": ey,
                "length": p["length"], "angle": p["angle"],
                "height": p["height"]["name"], "body": p.get("body_part", {}).get("name", "Other"),
                "cross": bool(p.get("cross")), "through": p.get("technique", {}).get("name") == "Through Ball",
                "switch": bool(p.get("switch")), "cut_back": bool(p.get("cut_back")),
                "under_pressure": bool(e.get("under_pressure")),
                "opp_passer_3": sum(d < 3 for d in d_passer), "opp_passer_5": sum(d < 5 for d in d_passer),
                "lane_opp": lane_opponents((x, y), (ex, ey), opps),
                # Skip opponents right at the end point (the interceptor on failed passes).
                "opp_end_5": sum(1.5 < d < 5 for d in d_end),
                "n_opp_visible": len(opps), "visible_area": area.area,
                "complete": int(outcome is None),
                "team_open": np.nanmean(opts) if np.isfinite(opts).any() else np.nan,
                "team_open_n": int(np.isfinite(opts).sum()),
                "chosen_open": chosen_open, "choice_gap": gap, "picked_most_open": picked_most_open,
            })
    return pd.DataFrame(rows)


FEATURES = ["x", "y", "end_x", "end_y", "length", "angle", "height", "body", "cross", "through",
            "switch", "cut_back", "under_pressure", "opp_passer_3", "opp_passer_5", "lane_opp",
            "opp_end_5", "n_opp_visible", "visible_area"]


def fit_xpass(df):
    X = df[FEATURES].copy()
    for c in ("height", "body"):
        X[c] = X[c].astype("category")
    for c in ("cross", "through", "switch", "cut_back", "under_pressure"):
        X[c] = X[c].astype(int)
    y = df.complete.to_numpy()
    oof = np.zeros(len(df))
    for tr, te in GroupKFold(n_splits=8).split(X, y, df.match_id):
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                           categorical_features="from_dtype", random_state=0)
        m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    print(f"xPass (out-of-fold): AUC {roc_auc_score(y, oof):.3f}, Brier {brier_score_loss(y, oof):.3f}, "
          f"mean xP {oof.mean():.3f} vs actual {y.mean():.3f}")
    bins = pd.cut(oof, [0, .5, .7, .8, .9, .95, 1])
    print(pd.DataFrame({"xP": oof, "actual": y}).groupby(bins, observed=True).mean().round(3).T.to_string())
    return oof


def zone_adjust(df, col, zone=20):
    """Residual of `col` vs the leave-player-out mean for the same ball zone."""
    d = df[df[col].notna()].copy()
    d["zone"] = (d.x // zone).astype(int).astype(str) + "_" + (d.y // zone).astype(int).astype(str)
    cell = d.groupby("zone")[col].agg(["sum", "count"])
    mine = d.groupby(["zone", "player"])[col].agg(["sum", "count"])
    j = d[["zone", "player"]].join(cell, on="zone").join(mine, on=["zone", "player"], rsuffix="_me")
    n = j["count"] - j["count_me"]
    d["resid"] = d[col] - (j["sum"] - j["sum_me"]) / n
    return d[n >= 20]


def board(d, col, rng, min_n=MIN_PASSES):
    rows = []
    for (player, team), g in d.groupby(["player", "team"]):
        if len(g) < min_n:
            continue
        pm = g.groupby("match_id")[col].agg(["sum", "count"]).to_numpy()
        idx = rng.integers(0, len(pm), size=(N_BOOT, len(pm)))
        boot = pm[idx, 0].sum(1) / pm[idx, 1].sum(1)
        rows.append({"player": player, "team": team, "n": len(g), "value": g[col].mean(),
                     "ci_low": np.percentile(boot, 2.5), "ci_high": np.percentile(boot, 97.5)})
    b = pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)
    b.index += 1
    return b


def report(name, b, extra=None):
    m = b[b.player == MESSI]
    if m.empty:
        print(f"{name}: Messi below threshold")
        return
    r = m.iloc[0]
    print(f"{name}: Messi {r.value:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]  rank {r.name}/{len(b)}"
          + (f"  | {extra}" if extra else ""))


def plot_difficulty(s, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 6.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    arg = s.team == "Argentina"
    messi = s.player == MESSI
    ax.scatter(s.difficulty[~arg], s.cae[~arg], s=16, color="#9aa0a6", label="other players", zorder=2)
    ax.scatter(s.difficulty[arg & ~messi], s.cae[arg & ~messi], s=24, color="#2a78d6", alpha=.55,
               label="Argentina", zorder=3)
    ax.scatter(s.difficulty[messi], s.cae[messi], s=140, marker="*", color="#2a78d6", edgecolor="#0b0b0b",
               zorder=4, label="Messi")
    top = s.assign(score=s.difficulty.rank(pct=True) + s.cae.rank(pct=True)).nlargest(5, "score")
    for _, r in pd.concat([top, s[messi]]).drop_duplicates("player").iterrows():
        left = r.difficulty < s.difficulty[messi].iloc[0] - 0.01  # keep labels near Messi from colliding
        ax.annotate(gravity.short(r.player), (r.difficulty, r.cae), xytext=(-4 if left else 4, 3),
                    textcoords="offset points", ha="right" if left else "left",
                    fontsize=7.5, color="#0b0b0b", weight="bold" if r.player == MESSI else "normal")
    ax.axhline(0, color="#c9c8c2", lw=0.8, zorder=1)
    ax.axvline(s.difficulty.median(), color="#c9c8c2", lw=0.8, ls="--", zorder=1)
    ax.set_xlabel("Average pass difficulty (1 − xPass)  →  harder passes")
    ax.set_ylabel("Completion above expected (actual − xPass)  →  better")
    ax.set_title(f"How hard are the passes, and how well are they completed?\n"
                 f"World Cup 2022, open play, players with ≥{MIN_PASSES} passes with a 360 frame",
                 fontsize=10, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    rng = np.random.default_rng(2022)
    cache = f"{OUT}/passes.csv.gz"
    if os.path.exists(cache):
        df = pd.read_csv(cache)
    else:
        df = load_passes()
        df.to_csv(cache, index=False)
    print(f"{len(df)} open-play passes with a 360 frame, {df.complete.mean():.1%} completed")

    # --- 1. creation vs selection -------------------------------------------------
    print("\n== Creation: how open are ALL visible teammates when this player passes? ==")
    c = zone_adjust(df[df.team_open_n >= 2], "team_open")
    bc = board(c, "resid", rng)
    report("team openness (units, zone-adjusted)", bc)
    print(bc.head(8).round(3).to_string())

    print("\n== Selection: chosen receiver's openness minus the other options ==")
    s = df[df.choice_gap.notna()]
    bs = board(s, "choice_gap", rng, MIN_CHOICES)
    m = s[s.player == MESSI]
    report("choice gap (units)", bs,
           f"picks the most open option {m.picked_most_open.mean():.0%} vs everyone {s.picked_most_open.mean():.0%}")
    sa = zone_adjust(s, "choice_gap")
    report("choice gap, zone-adjusted", board(sa, "resid", rng, MIN_CHOICES))
    print(bs.head(8).round(3).to_string())

    # --- 2. difficulty ------------------------------------------------------------
    print("\n== Difficulty: expected pass completion model ==")
    df["xp"] = fit_xpass(df)
    df["difficulty"] = 1 - df.xp
    df["cae"] = df.complete - df.xp
    bd = board(df, "difficulty", rng)
    bk = board(df, "cae", rng)
    report("difficulty (1 - xP)", bd)
    report("completion above expected", bk)
    for label, mask in [("hard passes (xP < 0.7)", df.xp < 0.7),
                        ("passes with 2+ opponents within 5 of passer", df.opp_passer_5 >= 2),
                        ("passes into the final third", df.end_x >= 80)]:
        sub = df[mask]
        me = sub[sub.player == MESSI]
        rest = sub[sub.player != MESSI]
        print(f"  {label}: Messi n={len(me)}, completion {me.complete.mean():.1%} vs xP {me.xp.mean():.1%} "
              f"(CAE {me.cae.mean():+.3f}); everyone else CAE {rest.cae.mean():+.3f}, "
              f"share of Messi's passes {len(me) / (df.player == MESSI).sum():.0%} vs "
              f"{len(rest) / (df.player != MESSI).sum():.0%}")
    both = bd[["player", "team", "n", "value"]].rename(columns={"value": "difficulty"}).merge(
        bk[["player", "value"]].rename(columns={"value": "cae"}), on="player")
    both.to_csv(f"{OUT}/pass_difficulty_leaderboard.csv", index=False)
    plot_difficulty(both, f"{OUT}/pass_difficulty.png")
    print("\nTop 10 by completion above expected:")
    print(bk.head(10).round(3).to_string())
    print("\nTop 10 by difficulty:")
    print(bd.head(10).round(3).to_string())
    df.to_csv(cache, index=False)


if __name__ == "__main__":
    main()
