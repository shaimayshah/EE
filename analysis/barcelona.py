"""Messi gravity across his Barcelona career (La Liga 2004/05 - 2020/21), plus World Cup 2022.

No 360 data exists before 2020/21, so this uses measures available in every season,
each compared with a leave-player-out baseline built from the same season's matches
(Barcelona matches only, so every season is on the same footing):

  pressure gravity  share of on-ball actions made under pressure, minus expected for
                    the zone and action type. On World Cup 2022 this proxy agrees
                    with the 360 gravity score across players (r = 0.91).
  creation per 90   npxG, key passes, xA, xGChain, xGBuildup (non-penalty)
  crowd creation    key passes from under-pressure open-play passes vs expected
  pass skill        completion above expected from an xPass model without 360
                    features (pooled over all seasons, match-grouped CV)
  decoy             opponents within 5 units of Messi at teammates' shots, from the
                    named shot freeze frames, vs other attackers in the same zone

Usage: ./scripts/download_statsbomb.sh 11 <season_id> data/laliga/<season_id> for each season,
       then .venv/bin/python analysis/barcelona.py
"""
import collections
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import chance_creation as cc
import gravity

MESSI = gravity.MESSI if hasattr(gravity, "MESSI") else "Lionel Andrés Messi Cuccittini"
OUT = "output/barcelona"
MIN_ACTIONS = 300
MIN_MINUTE_SHARE = 0.4  # of the most minutes anyone played that season
N_BOOT = 2000
FAIL = {"Incomplete", "Out", "Pass Offside"}


def seasons():
    out = []
    for d in sorted(glob.glob("data/laliga/*/matches.json")):
        ms = json.load(open(d))
        name = ms[0]["season"]["season_name"]
        ids = [m["match_id"] for m in ms if "Barcelona" in (m["home_team"]["home_team_name"],
                                                           m["away_team"]["away_team_name"])]
        out.append((name, os.path.dirname(d), ids))
    out.sort()
    out.append(("WC 2022", "data/wc2022",
                [m["match_id"] for m in json.load(open("data/wc2022/matches.json"))]))
    return out


def load_season(label, folder, ids):
    actions, passes = [], []
    mins = collections.Counter()
    creation = collections.defaultdict(collections.Counter)
    decoys = []
    for mid in ids:
        events = [e for e in json.load(open(f"{folder}/events/{mid}.json")) if e["period"] <= 4]
        mins.update(cc.minutes_played(events, json.load(open(f"{folder}/lineups/{mid}.json"))))
        shots = cc.add_creation(events, creation)
        decoys += cc.decoy_rows(shots, mid)[0]
        for e in events:
            t = e["type"]["name"]
            if t not in gravity.ON_BALL or "location" not in e or not gravity.is_open_play(e):
                continue
            x, y = e["location"][:2]
            actions.append((mid, e["player"]["name"], e["team"]["name"], t, x, y, int(bool(e.get("under_pressure")))))
            if t == "Pass":
                p = e["pass"]
                outcome = p.get("outcome", {}).get("name")
                if outcome and outcome not in FAIL:
                    continue
                passes.append({
                    "season": label, "match_id": f"{label}-{mid}", "player": e["player"]["name"],
                    "team": e["team"]["name"], "x": x, "y": y,
                    "end_x": p["end_location"][0], "end_y": p["end_location"][1],
                    "length": p["length"], "angle": p["angle"], "height": p["height"]["name"],
                    "body": p.get("body_part", {}).get("name", "Other"),
                    "cross": int(bool(p.get("cross"))), "switch": int(bool(p.get("switch"))),
                    "cut_back": int(bool(p.get("cut_back"))),
                    "through": int(p.get("technique", {}).get("name") == "Through Ball"),
                    "under_pressure": int(bool(e.get("under_pressure"))),
                    "complete": int(outcome is None),
                    "key_pass": int(bool(p.get("shot_assist") or p.get("goal_assist"))),
                })
    a = pd.DataFrame(actions, columns=["match_id", "player", "team", "type", "x", "y", "up"])
    return a, pd.DataFrame(passes), mins, creation, pd.DataFrame(decoys)


def residual(d, col, keys):
    """Leave-player-out residual of `col` against the mean for the same `keys` cell."""
    cell = d.groupby(keys)[col].agg(["sum", "count"])
    mine = d.groupby(keys + ["player"])[col].agg(["sum", "count"])
    j = d[keys + ["player"]].join(cell, on=keys).join(mine, on=keys + ["player"], rsuffix="_me")
    n = j["count"] - j["count_me"]
    d = d.assign(expected=(j["sum"] - j["sum_me"]) / n, n_other=n)
    d = d[d.n_other >= 20]
    return d.assign(resid=d[col] - d.expected)


def messi_stat(d, rng, min_n):
    """Messi's mean residual with a match-bootstrap CI, and his rank among qualifying players."""
    g = d[d.player == MESSI]
    pm = g.groupby("match_id").resid.agg(["sum", "count"]).to_numpy()
    idx = rng.integers(0, len(pm), (N_BOOT, len(pm)))
    boot = pm[idx, 0].sum(1) / pm[idx, 1].sum(1)
    by = d.groupby("player").resid.agg(["mean", "size"])
    by = by[by["size"] >= min_n].sort_values("mean", ascending=False)
    rank = list(by.index).index(MESSI) + 1 if MESSI in by.index else np.nan
    return g.resid.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5), rank, len(by), len(g)


def fit_xpass(p):
    X = p[["x", "y", "end_x", "end_y", "length", "angle", "height", "body", "cross", "switch", "cut_back",
           "through", "under_pressure", "season"]].copy()
    for c in ("height", "body", "season"):
        X[c] = X[c].astype("category")
    y = p.complete.to_numpy()
    oof = np.zeros(len(p))
    for tr, te in GroupKFold(n_splits=8).split(X, y, p.match_id):
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, categorical_features="from_dtype",
                                           random_state=0)
        m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    print(f"xPass without 360: AUC {roc_auc_score(y, oof):.3f}, mean xP {oof.mean():.3f} vs actual {y.mean():.3f}")
    return oof


def plot(t, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, INK2, GRID = "#2a78d6", "#52514e", "#c9c8c2"
    panels = [
        ("gravity", "gravity_lo", "gravity_hi", "Pressure gravity\n(extra share of actions under pressure, pts)", 100),
        ("xa90", None, None, "xA per 90\n(xG of shots his passes set up)", 1),
        ("npxg90", None, None, "Non-penalty xG per 90\n(his own shots)", 1),
        ("crowd_kp", "crowd_lo", "crowd_hi", "Shots set up from under-pressure passes\n(above expected, per 100 passes)", 100),
        ("cae", "cae_lo", "cae_hi", "Pass completion above expected\n(pts, xPass without 360)", 100),
        ("decoy", "decoy_lo", "decoy_hi", "Decoy: extra opponents near him\nat teammates' shots", 1),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), facecolor="#fcfcfb")
    lg = t[(t.season != "WC 2022") & (t.minutes >= 500)].reset_index(drop=True)  # 2004/05: 92 minutes
    wc = t[t.season == "WC 2022"]
    xs = np.arange(len(lg))
    for ax, (col, lo, hi, title, k) in zip(axes.flat, panels):
        ax.set_facecolor("#fcfcfb")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if lo:
            ax.fill_between(xs, lg[lo] * k, lg[hi] * k, color=BLUE, alpha=0.15, lw=0)
        ax.plot(xs, lg[col] * k, "-o", color=BLUE, lw=2, ms=5)
        if len(wc):
            wx = len(lg) + 0.8
            if lo:
                ax.errorbar(wx, wc[col].iloc[0] * k, yerr=[[(wc[col] - wc[lo]).iloc[0] * k],
                                                          [(wc[hi] - wc[col]).iloc[0] * k]],
                            fmt="none", ecolor=INK2, capsize=3)
            ax.plot(wx, wc[col].iloc[0] * k, "D", color="#0b0b0b", ms=6)
        if lo:
            ax.axhline(0, color=GRID, lw=0.8)
        ax.axvline(len(lg) - 0.1, color=GRID, lw=0.8, ls=":")
        ticks = list(xs) + ([len(lg) + 0.8] if len(wc) else [])
        labels = [s[2:4] + "/" + s[-2:] for s in lg.season] + (["WC\n22"] if len(wc) else [])
        ax.set_xticks(ticks, labels, fontsize=7, rotation=45)
        ax.set_title(title, fontsize=9, loc="left")
    fig.suptitle("Messi at Barcelona, La Liga 2005/06–2020/21 (line, 95% CI band), and World Cup 2022 (◆)\n"
                 "Each season vs a baseline from that season's Barcelona matches (2004/05 left out: 92 minutes)",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(2022)
    rows, all_passes = [], []
    for label, folder, ids in seasons():
        a, p, mins, creation, dec = load_season(label, folder, ids)
        all_passes.append(p)
        r = {"season": label, "matches": len(ids)}

        a["zone"] = (a.x // 10).astype(int).astype(str) + "_" + (a.y // 10).astype(int).astype(str)
        g = residual(a, "up", ["zone", "type"])
        r["gravity"], r["gravity_lo"], r["gravity_hi"], r["gravity_rank"], r["gravity_of"], r["actions"] = \
            messi_stat(g, rng, MIN_ACTIONS)

        m = mins.get((MESSI, "Barcelona" if label != "WC 2022" else "Argentina"), 0)
        c = creation[(MESSI, "Barcelona" if label != "WC 2022" else "Argentina")]
        r["minutes"] = round(m)
        for k in ("npxg", "xa", "key_passes", "xgchain", "xgbuildup"):
            r[f"{k}90" if k != "key_passes" else "kp90"] = c[k] * 90 / m
        r["goals_np"], r["assists"] = c["np_goals"], c["assists"]
        p90 = pd.DataFrame([{"player": k[0], "xa90": creation[k]["xa"] * 90 / v,
                             "npxg90": creation[k]["npxg"] * 90 / v} for k, v in mins.items()
                            if v >= MIN_MINUTE_SHARE * max(mins.values())])
        for k in ("xa90", "npxg90"):
            srt = p90.sort_values(k, ascending=False).player.tolist()
            r[f"{k}_rank"] = srt.index(MESSI) + 1 if MESSI in srt else np.nan
        r["players_ranked"] = len(p90)

        q = p.assign(zone=(p.x // 20).astype(int))
        up = residual(q[q.under_pressure == 1], "key_pass", ["zone"])
        r["crowd_kp"], r["crowd_lo"], r["crowd_hi"], *_ = messi_stat(up, rng, 1)

        dec["zone"] = (dec.x // 10).astype(int).astype(str) + "_" + (dec.y // 10).astype(int).astype(str)
        dd = residual(dec, "opp_5", ["zone"])
        if (dd.player == MESSI).sum() >= 10:
            r["decoy"], r["decoy_lo"], r["decoy_hi"], *_ = messi_stat(dd, rng, 25)
        rows.append(r)
        print(f"{label}: {len(ids)} matches, {r['minutes']} min, gravity {r['gravity'] * 100:+.1f} pts "
              f"(rank {r['gravity_rank']}/{r['gravity_of']}), npxG90 {r['npxg90']:.2f}, xA90 {r['xa90']:.2f}")

    p = pd.concat(all_passes, ignore_index=True)
    p["xp"] = fit_xpass(p)
    p["resid"] = p.complete - p.xp
    t = pd.DataFrame(rows)
    for i, label in enumerate(t.season):
        cae = messi_stat(p[p.season == label], rng, 300)
        t.loc[i, ["cae", "cae_lo", "cae_hi", "cae_rank", "cae_of"]] = cae[:5]
    t.to_csv(f"{OUT}/messi_by_season.csv", index=False)
    cols = ["season", "matches", "minutes", "gravity", "gravity_rank", "gravity_of", "npxg90", "xa90", "kp90",
            "xgchain90", "xgbuildup90", "crowd_kp", "cae", "cae_rank", "cae_of", "decoy"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(t[cols].round(3).to_string(index=False))
    plot(t, f"{OUT}/messi_career.png")


if __name__ == "__main__":
    main()
