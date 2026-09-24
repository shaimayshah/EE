"""Messi gravity, step 6: chance creation.

1. Volume per 90: shot assists (key passes), xA (xG of the shots his passes set
   up), xGChain (xG of every possession he touched) and xGBuildup (same, minus
   possessions where he took the shot or played the key pass). Non-penalty,
   shoot-outs excluded.
2. Quality: xG per shot assisted.
3. Creating from the crowd: chance that a pass leads straight to a shot, split by
   how many opponents were within 5 units of the passer (360 frame).
4. Decoy gravity: StatsBomb *shot* freeze frames name every player, so at each
   teammate's shot we can count the opponents near Messi even when he is off the
   ball, compared with other attackers standing in the same zone.

Usage: .venv/bin/python analysis/chance_creation.py
"""
import collections
import glob
import json
import math
import os

import numpy as np
import pandas as pd

import gravity

DATA = gravity.DATA
OUT = gravity.OUT
MESSI = "Lionel Andrés Messi Cuccittini"
MIN_MINUTES = 270
TOUCH = {"Pass", "Carry", "Ball Receipt*", "Dribble", "Shot"}
PERIOD_START = {1: 0, 2: 45, 3: 90, 4: 105}
R = 5
N_BOOT = 2000


def clock(t):
    m, s = t.split(":")
    return int(m) + int(s) / 60


def minutes_played(events, lineups):
    """Playing minutes per player, handling StatsBomb's per-period clock reset."""
    end = collections.defaultdict(float)
    for e in events:
        if e["period"] <= 4:
            end[e["period"]] = max(end[e["period"]], e["minute"] + e["second"] / 60)
    periods = sorted(end)
    offset, acc = {}, 0.0
    for p in periods:
        offset[p] = acc
        acc += end[p] - PERIOD_START[p]
    total = acc

    def absolute(t, p):
        return offset[p] + clock(t) - PERIOD_START[p]

    mins = {}
    for team in lineups:
        for pl in team["lineup"]:
            # Position spells can be listed out of order and overlap, so merge them.
            spans = []
            for pos in pl["positions"]:
                if pos["from_period"] not in offset:
                    continue
                a = absolute(pos["from"], pos["from_period"])
                b = total if pos["to"] is None or pos["to_period"] not in offset else absolute(pos["to"], pos["to_period"])
                if b > a:
                    spans.append((a, b))
            m, cur = 0.0, None
            for a, b in sorted(spans):
                if cur and a <= cur[1]:
                    cur = (cur[0], max(cur[1], b))
                else:
                    if cur:
                        m += cur[1] - cur[0]
                    cur = (a, b)
            if cur:
                m += cur[1] - cur[0]
            if m > 0:
                mins[(pl["player_name"], team["team_name"])] = m
    return mins


def load():
    mins = collections.Counter()
    creation = collections.defaultdict(lambda: collections.Counter())
    pass_rows, decoy_rows, shot_rows = [], [], []
    for path in sorted(glob.glob(f"{DATA}/events/*.json")):
        mid = int(os.path.basename(path)[:-5])
        events = [e for e in json.load(open(path)) if e["period"] <= 4]
        lineups = json.load(open(f"{DATA}/lineups/{mid}.json"))
        mins.update(minutes_played(events, lineups))
        frames = {f["event_uuid"]: f for f in json.load(open(f"{DATA}/three-sixty/{mid}.json"))}
        by_id = {e["id"]: e for e in events}

        shots = [e for e in events if e["type"]["name"] == "Shot" and e["shot"]["type"]["name"] != "Penalty"]
        poss_xg = collections.Counter()
        for s in shots:
            poss_xg[s["possession"]] += s["shot"]["statsbomb_xg"]

        # xGChain / xGBuildup
        touched = collections.defaultdict(set)
        finishers = collections.defaultdict(set)
        for e in events:
            if e["type"]["name"] in TOUCH and "player" in e and e["team"]["name"] == e["possession_team"]["name"]:
                touched[e["possession"]].add((e["player"]["name"], e["team"]["name"]))
        for s in shots:
            key = (s["player"]["name"], s["team"]["name"])
            finishers[s["possession"]].add(key)
            kp = by_id.get(s["shot"].get("key_pass_id"))
            if kp:
                finishers[s["possession"]].add((kp["player"]["name"], kp["team"]["name"]))
                c = creation[(kp["player"]["name"], kp["team"]["name"])]
                c["key_passes"] += 1
                c["xa"] += s["shot"]["statsbomb_xg"]
                c["open_play_kp"] += int(gravity.is_open_play(kp))
                c["assists"] += int(s["shot"]["outcome"]["name"] == "Goal")
        for poss, xg in poss_xg.items():
            for key in touched[poss]:
                creation[key]["xgchain"] += xg
                if key not in finishers[poss]:
                    creation[key]["xgbuildup"] += xg

        # Passes from a 360 frame: did they lead straight to a shot?
        for e in events:
            if e["type"]["name"] != "Pass" or not gravity.is_open_play(e) or e["id"] not in frames:
                continue
            x, y = e["location"]
            opps = [p["location"] for p in frames[e["id"]]["freeze_frame"] if not p["teammate"]]
            pass_rows.append({
                "match_id": mid, "player": e["player"]["name"], "team": e["team"]["name"], "x": x,
                "opp_5": sum(math.dist(o, (x, y)) < R for o in opps),
                "complete": "outcome" not in e["pass"],
                "key_pass": bool(e["pass"].get("shot_assist") or e["pass"].get("goal_assist")),
            })

        # Decoy: every attacking outfield player in a shot freeze frame except the shooter.
        for s in shots:
            ff = s["shot"].get("freeze_frame")
            if not ff:
                continue
            opps = [p["location"] for p in ff if not p["teammate"] and p["position"]["name"] != "Goalkeeper"]
            messi_near = np.nan
            for p in ff:
                if not p["teammate"] or p["position"]["name"] == "Goalkeeper":
                    continue
                px, py = p["location"]
                n = sum(math.dist(o, (px, py)) < R for o in opps)
                decoy_rows.append({"match_id": mid, "player": p["player"]["name"], "team": s["team"]["name"],
                                   "x": px, "y": py, "opp_5": n})
                if p["player"]["name"] == MESSI:
                    messi_near = n
            sx, sy = s["location"]
            shot_rows.append({"match_id": mid, "team": s["team"]["name"], "shooter": s["player"]["name"],
                              "xg": s["shot"]["statsbomb_xg"], "x": sx, "y": sy,
                              "messi_in_frame": not np.isnan(messi_near), "messi_opp_5": messi_near,
                              "opp_near_shooter": sum(math.dist(o, (sx, sy)) < R for o in opps)})
    return mins, creation, pd.DataFrame(pass_rows), pd.DataFrame(decoy_rows), pd.DataFrame(shot_rows)


def per90(mins, creation):
    rows = []
    for key, m in mins.items():
        if m < MIN_MINUTES:
            continue
        c = creation.get(key, collections.Counter())
        rows.append({"player": key[0], "team": key[1], "minutes": round(m),
                     **{k: c[k] * 90 / m for k in ("key_passes", "xa", "xgchain", "xgbuildup")},
                     "assists": c["assists"], "xa_per_kp": c["xa"] / c["key_passes"] if c["key_passes"] else np.nan,
                     "kp_total": c["key_passes"]})
    return pd.DataFrame(rows)


def rank_line(df, col, label, fmt="{:.2f}"):
    d = df.sort_values(col, ascending=False).reset_index(drop=True)
    i = d.index[d.player == MESSI][0]
    top = ", ".join(f"{gravity.short(p)} {fmt.format(v)}" for p, v in zip(d.player[:3], d[col][:3]))
    return f"{label:<30} Messi {fmt.format(d[col][i])}  rank {i + 1}/{len(d)}   (top: {top})"


def decoy(decoys, rng, zone=10):
    d = decoys.copy()
    d["zone"] = (d.x // zone).astype(int).astype(str) + "_" + (d.y // zone).astype(int).astype(str)
    cell = d.groupby("zone").opp_5.agg(["sum", "count"])
    mine = d.groupby(["zone", "player"]).opp_5.agg(["sum", "count"])
    j = d[["zone", "player"]].join(cell, on="zone").join(mine, on=["zone", "player"], rsuffix="_me")
    n = j["count"] - j["count_me"]
    d["resid"] = d.opp_5 - (j["sum"] - j["sum_me"]) / n
    d = d[n >= 20]
    rows = []
    for (p, t), g in d.groupby(["player", "team"]):
        if len(g) < 25:
            continue
        pm = g.groupby("match_id").resid.agg(["sum", "count"]).to_numpy()
        idx = rng.integers(0, len(pm), size=(N_BOOT, len(pm)))
        boot = pm[idx, 0].sum(1) / pm[idx, 1].sum(1)
        rows.append({"player": p, "team": t, "n": len(g), "actual": g.opp_5.mean(), "value": g.resid.mean(),
                     "ci_low": np.percentile(boot, 2.5), "ci_high": np.percentile(boot, 97.5)})
    b = pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)
    b.index += 1
    return b


def crowd_adjusted(passes, rng, zone=20):
    """Key passes vs a leave-player-out expectation for the same start zone and crowd level."""
    p = passes.copy()
    p["zone"] = (p.x // zone).astype(int)
    p["crowd2"] = np.where(p.opp_5 >= 2, "2+", "0-1")
    key = ["zone", "crowd2"]
    cell = p.groupby(key).key_pass.agg(["sum", "count"])
    mine = p.groupby(key + ["player"]).key_pass.agg(["sum", "count"])
    j = p[key + ["player"]].join(cell, on=key).join(mine, on=key + ["player"], rsuffix="_me")
    p["exp"] = (j["sum"] - j["sum_me"]) / (j["count"] - j["count_me"])
    for cr in ("0-1", "2+"):
        g = p[(p.player == MESSI) & (p.crowd2 == cr)]
        pm = g.groupby("match_id")[["key_pass", "exp"]].sum().to_numpy()
        n = g.groupby("match_id").size().to_numpy()
        idx = rng.integers(0, len(pm), (N_BOOT, len(pm)))
        boot = (pm[idx, 0].sum(1) - pm[idx, 1].sum(1)) / n[idx].sum(1) * 100
        print(f"  Messi, {cr} opponents near him: {g.key_pass.sum()} key passes from {len(g)} passes vs "
              f"{g.exp.sum():.1f} expected -> {(g.key_pass.sum() - g.exp.sum()) / len(g) * 100:+.1f} pts per 100 "
              f"[{np.percentile(boot, 2.5):+.1f}, {np.percentile(boot, 97.5):+.1f}]")
    c = p[p.crowd2 == "2+"].groupby(["player", "team"]).agg(n=("key_pass", "size"), kp=("key_pass", "sum"),
                                                            exp=("exp", "sum")).reset_index()
    c = c[c.n >= 40].assign(diff=lambda d: (d.kp - d.exp) / d.n * 100).sort_values("diff", ascending=False)
    c = c.reset_index(drop=True)
    print(f"  crowded passes, key passes above expected per 100: Messi rank "
          f"{c.index[c.player == MESSI][0] + 1}/{len(c)} (players with ≥40 crowded passes)")
    print(c.head(5).round(2).to_string())
    return p


def plot(p90, passes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.4), facecolor="#fcfcfb",
                                 gridspec_kw={"width_ratios": [1.25, 1]})
    for ax in (a1, a2):
        ax.set_facecolor("#fcfcfb")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    messi = p90.player == MESSI
    a1.scatter(p90.xgbuildup[~messi], p90.xa[~messi], s=14, color="#9aa0a6", label="other players (≥270 min)")
    a1.scatter(p90.xgbuildup[messi], p90.xa[messi], s=160, marker="*", color="#2a78d6", edgecolor="#0b0b0b",
               label="Messi", zorder=4)
    lab = p90.assign(s=p90.xa.rank(pct=True) + p90.xgbuildup.rank(pct=True)).nlargest(7, "s")
    for _, r in pd.concat([lab, p90[messi]]).drop_duplicates("player").iterrows():
        a1.annotate(gravity.short(r.player), (r.xgbuildup, r.xa), xytext=(4, 3), textcoords="offset points",
                    fontsize=7.5, weight="bold" if r.player == MESSI else "normal")
    a1.set_xlabel("xGBuildup per 90: xG of possessions he helped build (not the shot or final pass)")
    a1.set_ylabel("xA per 90: xG of the shots his passes set up")
    a1.set_title("Final pass vs build-up: two ways to create", fontsize=10, loc="left")
    a1.legend(fontsize=8, frameon=False, loc="upper left")

    me = passes[passes.player == MESSI].groupby("crowd2").agg(n=("key_pass", "size"), actual=("key_pass", "sum"),
                                                              expected=("exp", "sum")).reindex(["0-1", "2+"])
    xs = np.arange(2)
    w = 0.38
    a2.bar(xs - w / 2, me.expected, width=w, color="#9aa0a6", label="expected (others' rate, same zone & crowd)")
    a2.bar(xs + w / 2, me.actual, width=w, color="#2a78d6", label="Messi actual")
    for x0, (n, act, ex) in zip(xs, me[["n", "actual", "expected"]].itertuples(index=False)):
        a2.annotate(f"{ex:.1f}", (x0 - w / 2, ex), xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=8, color="#52514e")
        a2.annotate(f"{act}", (x0 + w / 2, act), xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=8, color="#0b0b0b")
    a2.set_xticks(xs, [f"0–1 opponents near him\n({me.n.iloc[0]} passes)", f"2+ opponents near him\n({me.n.iloc[1]} passes)"])
    a2.set_ylim(0, me[["actual", "expected"]].to_numpy().max() * 1.35)
    a2.set_ylabel("Open-play passes that set up a shot")
    a2.set_title(f"Creating from inside the crowd ({R}-unit radius)", fontsize=10, loc="left")
    a2.legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("Messi's chance creation, World Cup 2022 (non-penalty xG; right panel: open-play passes with a 360 frame)",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    rng = np.random.default_rng(2022)
    mins, creation, passes, decoys, shots = load()
    p90 = per90(mins, creation)
    m = p90[p90.player == MESSI].iloc[0]
    print(f"Messi: {m.minutes} minutes, {m.kp_total} key passes, {m.assists} assists")
    print(f"{len(p90)} players with ≥{MIN_MINUTES} minutes\n")
    print("== 1. Volume per 90 ==")
    for col, label in [("key_passes", "shot assists / 90"), ("xa", "xA / 90"),
                       ("xgchain", "xGChain / 90"), ("xgbuildup", "xGBuildup / 90")]:
        print(rank_line(p90, col, label))
    print("\n== 2. Quality ==")
    q = p90[p90.kp_total >= 8]
    print(rank_line(q, "xa_per_kp", f"xG per shot assisted (≥8 KP, {len(q)} players)", "{:.3f}"))

    print("\n== 3. Creating from the crowd (open-play passes with a 360 frame) ==")
    passes["crowd"] = pd.cut(passes.opp_5, [-1, 0, 1, 99], labels=["0", "1", "2+"])
    t = passes.groupby(["crowd", passes.player == MESSI], observed=True).key_pass.agg(["mean", "size"]).unstack()
    t.columns = [f"{a}_{'messi' if b else 'others'}" for a, b in t.columns]
    print((t * [100, 100, 1, 1]).round(2).to_string())
    adj = crowd_adjusted(passes, rng)
    me = passes[passes.player == MESSI]
    print(f"Messi's key passes made with 2+ opponents within {R}: "
          f"{(me.key_pass & (me.opp_5 >= 2)).sum()} of {me.key_pass.sum()}")

    print("\n== 4. Decoy gravity at teammates' shots (shot freeze frames, names known) ==")
    b = decoy(decoys, rng)
    r = b[b.player == MESSI].iloc[0]
    print(f"Messi off the ball: {r.actual:.2f} opponents within {R} vs expected {r.actual - r.value:.2f} "
          f"-> {r.value:+.2f} [{r.ci_low:+.2f}, {r.ci_high:+.2f}], rank {r.name}/{len(b)} (n={r.n})")
    print(b.head(8)[["player", "team", "n", "actual", "value", "ci_low", "ci_high"]].round(2).to_string())
    a = shots[(shots.team == "Argentina") & (shots.shooter != MESSI) & shots.messi_in_frame]
    for lo, hi, lab in [(0, 0, "0"), (1, 1, "1"), (2, 99, "2+")]:
        g = a[a.messi_opp_5.between(lo, hi)]
        print(f"  Argentina teammate shots with {lab} opponents near Messi: n={len(g)}, mean xG {g.xg.mean():.3f}, "
              f"opponents within {R} of shooter {g.opp_near_shooter.mean():.2f}")

    p90.to_csv(f"{OUT}/chance_creation_per90.csv", index=False)
    b.to_csv(f"{OUT}/decoy_gravity.csv")
    plot(p90, adj, f"{OUT}/chance_creation.png")


if __name__ == "__main__":
    main()
