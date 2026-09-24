"""Messi gravity, step 3: do teammates get more space when Messi passes to them?

For every completed open-play pass we look at the *receiver* at the moment of the
ball receipt (360 frame) and compare their space with what's normal for the same
receiving zone, pass height and pass length. The baseline is leave-passer-out, so
a passer never sets their own expectation.

  space bonus (defenders) = expected opponents within 5 - actual   (+ = more space)
  space bonus (distance)  = actual nearest-opponent distance - expected (capped at 5)

Controls:
  * Argentina-only comparison (same opponents, same matches)
  * within-receiver comparison: the same teammate receiving from Messi vs from others
  * does a player's gravity (step 2) predict the space they create for teammates?

Usage: .venv/bin/python analysis/gravity.py   (builds output/actions.csv.gz)
       .venv/bin/python analysis/teammate_space.py
"""
import os

import numpy as np
import pandas as pd

import gravity

OUT = gravity.OUT
R = 5
MIN_PASSES = 150
N_BOOT = 2000
MESSI = "Lionel Andrés Messi Cuccittini"


def receipts_with_expected(df, zone=10):
    d = df[(df.type == "Ball Receipt*") & df.passer.notna() & df[f"vis_{R}"]].copy()
    d["near"] = d.nearest_opp.fillna(R).clip(upper=R)
    d["zone"] = (d.x // zone).astype(int).astype(str) + "_" + (d.y // zone).astype(int).astype(str)
    d["len_bin"] = pd.cut(d.pass_length, [0, 10, 20, 35, 200], labels=False)
    key = ["zone", "pass_height", "len_bin"]
    for col in (f"opp_{R}", "near"):
        cell = d.groupby(key)[col].agg(["sum", "count"])
        mine = d.groupby(key + ["passer"])[col].agg(["sum", "count"])
        j = d[key + ["passer"]].join(cell, on=key).join(mine, on=key + ["passer"], rsuffix="_me")
        d[f"n_other"] = j["count"] - j["count_me"]
        d[f"exp_{col}"] = (j["sum"] - j["sum_me"]) / d["n_other"]
    d = d[d.n_other >= 20]
    d["bonus_opp"] = d[f"exp_opp_{R}"] - d[f"opp_{R}"]
    d["bonus_near"] = d["near"] - d["exp_near"]
    return d


def boot_ci(g, col, rng):
    per_match = g.groupby("match_id")[col].agg(["sum", "count"]).to_numpy()
    idx = rng.integers(0, len(per_match), size=(N_BOOT, len(per_match)))
    boot = per_match[idx, 0].sum(1) / per_match[idx, 1].sum(1)
    return np.percentile(boot, [2.5, 97.5])


def passer_board(d, rng):
    rows = []
    for (passer, team), g in d.groupby(["passer", "team"]):
        if len(g) < MIN_PASSES:
            continue
        lo, hi = boot_ci(g, "bonus_opp", rng)
        nlo, nhi = boot_ci(g, "bonus_near", rng)
        rows.append({"passer": passer, "team": team, "passes": len(g), "matches": g.match_id.nunique(),
                     "bonus_opp": g.bonus_opp.mean(), "ci_low": lo, "ci_high": hi,
                     "bonus_near": g.bonus_near.mean(), "near_ci_low": nlo, "near_ci_high": nhi,
                     "exp_near": g.exp_near.mean()})
    b = pd.DataFrame(rows).sort_values("bonus_opp", ascending=False).reset_index(drop=True)
    b.index += 1
    return b


def within_receiver(d, min_n=10):
    """Same Argentina receiver: space from Messi's passes minus space from everyone else's."""
    a = d[d.team == "Argentina"].assign(from_messi=lambda x: x.passer == MESSI)
    rows = []
    for rcv, g in a.groupby("player"):
        m, o = g[g.from_messi], g[~g.from_messi]
        if len(m) >= min_n and len(o) >= min_n:
            rows.append({"receiver": rcv, "from_messi": len(m), "from_others": len(o),
                         "diff_opp": m.bonus_opp.mean() - o.bonus_opp.mean(),
                         "diff_near": m.bonus_near.mean() - o.bonus_near.mean()})
    w = pd.DataFrame(rows).sort_values("from_messi", ascending=False)
    wts = w.from_messi
    return w, np.average(w.diff_opp, weights=wts), np.average(w.diff_near, weights=wts)


def within_receiver_ci(d, rng, n_boot=1000):
    a = d[d.team == "Argentina"]
    by_match = [g for _, g in a.groupby("match_id")]
    res = []
    for _ in range(n_boot):
        s = pd.concat([by_match[i] for i in rng.integers(0, len(by_match), len(by_match))])
        _, o, n = within_receiver(s, min_n=5)
        res.append((o, n))
    return np.percentile(np.array(res), [2.5, 97.5], axis=0)


def plot_scatter(merged, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    arg = merged.team == "Argentina"
    ax.scatter(merged.gravity[~arg], merged.bonus_opp[~arg], s=14, color="#9aa0a6", label="other players")
    ax.scatter(merged.gravity[arg], merged.bonus_opp[arg], s=24, color="#75AADB", label="Argentina")
    for _, r in merged.iterrows():
        if r.player == MESSI or r.gravity > 0.3 or r.bonus_opp > 0.12 or (r.team == "Argentina" and r.gravity > 0.15):
            ax.annotate(gravity.short(r.player), (r.gravity, r.bonus_opp), fontsize=7,
                        xytext=(3, 3), textcoords="offset points",
                        weight="bold" if r.player == MESSI else "normal")
    k = np.polyfit(merged.gravity, merged.bonus_opp, 1)
    xs = np.linspace(merged.gravity.min(), merged.gravity.max(), 50)
    ax.plot(xs, np.polyval(k, xs), color="#333", lw=1, ls="--",
            label=f"fit (r = {merged.gravity.corr(merged.bonus_opp):.2f})")
    ax.axhline(0, color="#ccc", lw=0.8)
    ax.axvline(0, color="#ccc", lw=0.8)
    ax.set_xlabel("Gravity: extra opponents around the player on the ball (step 2)")
    ax.set_ylabel("Space created: fewer opponents around the teammate receiving their pass")
    ax.set_title("Does drawing defenders free up teammates? (World Cup 2022)", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_passers(b, path, col="bonus_opp", lo="ci_low", hi="ci_high", top=25,
                 xlabel=f"Space bonus: fewer opponents within {R} units of the receiver vs. expected"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = b.sort_values(col, ascending=False).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 9))
    colors = ["#75AADB" if p == MESSI else "#9aa0a6" for p in t.passer]
    ax.barh(range(len(t)), t[col], color=colors,
            xerr=[t[col] - t[lo], t[hi] - t[col]], ecolor="#555", capsize=2)
    ax.set_yticks(range(len(t)), [f"{gravity.short(p)} ({tm})" for p, tm in zip(t.passer, t.team)], fontsize=8)
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(f"Who gives teammates the most space? (top {top}, ≥{MIN_PASSES} completed passes)\n"
                 "expected = same receiving zone, pass height & length; 95% bootstrap CI over matches",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    rng = np.random.default_rng(2022)
    df = pd.read_csv(f"{OUT}/actions.csv.gz")
    d = receipts_with_expected(df)
    print(f"{len(d)} completed open-play passes with a fully visible receiver")

    for zone in (10, 20):
        b = passer_board(receipts_with_expected(df, zone), rng)
        m = b[b.passer == MESSI].iloc[0]
        print(f"zone={zone}: Messi rank {m.name}/{len(b)}  bonus {m.bonus_opp:+.3f} "
              f"[{m.ci_low:+.3f}, {m.ci_high:+.3f}] defenders, {m.bonus_near:+.3f} units nearest-opp")

    b = passer_board(d, rng)
    b.to_csv(f"{OUT}/space_created_leaderboard.csv")
    plot_passers(b, f"{OUT}/space_created_leaderboard.png")
    plot_passers(b, f"{OUT}/space_created_nearest.png", "bonus_near", "near_ci_low", "near_ci_high",
                 xlabel="Space bonus: extra distance (units) from receiver to nearest opponent vs. expected")
    for col in ("bonus_opp", "bonus_near"):
        rank = b[col].rank(ascending=False)[b.passer == MESSI].iloc[0]
        print(f"Messi rank by {col}: {rank:.0f}/{len(b)}")
    with pd.option_context("display.width", 200):
        print(b.head(15).round(3).to_string())
        print("\nArgentina passers:")
        print(b[b.team == "Argentina"].round(3).to_string())

    w, avg_opp, avg_near = within_receiver(d)
    print("\nSame receiver, from Messi vs from other Argentina passers:")
    print(w.round(3).to_string(index=False))
    ci = within_receiver_ci(d, rng)
    print(f"weighted avg: {avg_opp:+.3f} [{ci[0,0]:+.3f}, {ci[1,0]:+.3f}] defenders, "
          f"{avg_near:+.3f} [{ci[0,1]:+.3f}, {ci[1,1]:+.3f}] units nearest-opp")

    lb = gravity.leaderboard(gravity.add_expected(df, R, 10), R, rng)
    merged = lb.merge(b, left_on=["player", "team"], right_on=["passer", "team"])
    print(f"\ngravity vs space created across {len(merged)} players: "
          f"r = {merged.gravity.corr(merged.bonus_opp):.2f} (defenders), "
          f"{merged.gravity.corr(merged.bonus_near):.2f} (nearest-opp)")
    plot_scatter(merged, f"{OUT}/gravity_vs_space.png")


if __name__ == "__main__":
    main()
