"""Extra network analysis: team-level measures, the 'red card' test, passing units,
a random-network comparison and a bootstrap of player rankings.

Run:  python3 network_extra.py > NETWORK_EXTRA.md      (also writes figures/red_card.png)
Needs: networkx, scipy, numpy, matplotlib
"""
import itertools
import os

import networkx as nx
import numpy as np

from data import TEAMS
from recalculate import build_graph, corrected_metrics

N = 11
RNG = np.random.default_rng(2018)
N_RANDOM = 1000          # random networks per team
N_BOOT = 1000            # bootstrap resamples per team


# ------------------------------------------------------------------ measures

def _graph(A, keep):
    G = nx.DiGraph()
    G.add_nodes_from(keep)
    for i in keep:
        for j in keep:
            if A[i][j]:
                G.add_edge(i, j, distance=1 / A[i][j])
    return G


def efficiency(A, keep=None, pairs_of=None):
    """Mean of 1/d(i,j) over ordered pairs, with d = shortest path using 1/passes
    as the length of each link. High efficiency = the ball can move easily between
    two players. `keep` = players available to pass through; `pairs_of` = players
    whose pairs are averaged (defaults to `keep`)."""
    keep = list(range(N)) if keep is None else list(keep)
    pairs_of = keep if pairs_of is None else list(pairs_of)
    G = _graph(A, keep)
    total = 0.0
    for s in pairs_of:
        d = nx.single_source_dijkstra_path_length(G, s, weight="distance")
        total += sum(1 / v for t, v in d.items() if t != s and t in pairs_of)
    return total / (len(pairs_of) * (len(pairs_of) - 1))


def removal_drop(A, group):
    """Drop in efficiency between the *other* players when `group` is removed:
    the same pairs are compared with and without `group` available to pass through."""
    others = [k for k in range(N) if k not in group]
    with_group = efficiency(A, keep=range(N), pairs_of=others)
    return 1 - efficiency(A, keep=others) / with_group


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    return (2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())


def global_measures(A):
    A = np.asarray(A, dtype=float)
    total = A.sum()
    links = A[~np.eye(N, dtype=bool)]
    p = links[links > 0] / total
    m = corrected_metrics(A.tolist())
    b = np.array(m["betweenness"])
    return {
        "passes": total,
        "density": (links > 0).mean(),
        "reciprocity": np.minimum(A, A.T).sum() / total,
        "entropy": -(p * np.log(p)).sum() / np.log(N * (N - 1)),
        "btw_centralisation": (b.max() - b).sum() / (N - 1),
        "top3_btw_share": np.sort(b)[-3:].sum() / b.sum() if b.sum() else 0.0,
        "pagerank_gini": gini(m["pagerank"]),
        "involvement_gini": gini(A.sum(0) + A.sum(1)),
        "clustering": float(np.mean(m["clustering"])),
        # efficiency divided by the team's mean pass count per link, so teams that
        # simply passed more are not automatically "more efficient"
        "rel_efficiency": efficiency(A) / (total / (N * (N - 1))),
        "modularity": passing_units(A)[1],
    }


MEASURE_LABELS = {
    "passes": ("Passes between starters", "{:.0f}"),
    "density": ("Density (share of the 110 possible links used)", "{:.2f}"),
    "reciprocity": ("Reciprocity (share of passes returned)", "{:.2f}"),
    "entropy": ("Passing entropy (1 = passes spread evenly over all links)", "{:.3f}"),
    "btw_centralisation": ("Betweenness centralisation (1 = everything goes through one player)", "{:.3f}"),
    "top3_btw_share": ("Share of betweenness held by the top 3", "{:.0%}"),
    "pagerank_gini": ("PageRank inequality (Gini)", "{:.3f}"),
    "involvement_gini": ("Involvement inequality (Gini of passes made + received)", "{:.3f}"),
    "clustering": ("Mean weighted clustering", "{:.3f}"),
    "rel_efficiency": ("Relative efficiency (volume-adjusted)", "{:.2f}"),
    "modularity": ("Modularity of passing units", "{:.3f}"),
}


# ------------------------------------------------------------- passing units

def passing_units(A):
    """Louvain communities on the undirected network (weight = passes both ways)."""
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(i + 1, N):
            w = A[i][j] + A[j][i]
            if w:
                G.add_edge(i, j, weight=w)
    best = None
    for seed in range(20):      # Louvain is randomised; keep the best of 20 runs
        comms = nx.community.louvain_communities(G, weight="weight", seed=seed)
        q = nx.community.modularity(G, comms, weight="weight")
        if best is None or q > best[1] + 1e-12:
            best = (comms, q)
    return best


# ----------------------------------------------------------- red-card test

def red_card(A):
    total = np.asarray(A).sum()
    rows = []
    for i in range(N):
        drop = removal_drop(A, (i,))
        involved = (sum(A[i]) + sum(r[i] for r in A)) / total
        rows.append((i, drop, involved))
    combos = {}
    for size in (2, 3):
        res = []
        for group in itertools.combinations(range(N), size):
            res.append((group, removal_drop(A, group)))
        res.sort(key=lambda t: -t[1])
        combos[size] = res
    return rows, combos


# ------------------------------------------------------- random networks

def randomise(A):
    """Random network in which every player makes and receives exactly as many
    passes as in the match, but the receivers are shuffled (no self-passes)."""
    A = np.asarray(A, dtype=int)
    outs = np.repeat(np.arange(N), A.sum(1))
    ins = np.repeat(np.arange(N), A.sum(0))
    RNG.shuffle(ins)
    # remove self-passes by swapping receivers with random other passes
    for _ in range(100):
        bad = np.where(outs == ins)[0]
        if len(bad) == 0:
            break
        for k in bad:
            j = RNG.integers(len(ins))
            if outs[j] != ins[k] and outs[k] != ins[j]:
                ins[k], ins[j] = ins[j], ins[k]
    R = np.zeros((N, N), dtype=int)
    np.add.at(R, (outs, ins), 1)
    np.fill_diagonal(R, 0)
    return R


TESTED = ["reciprocity", "entropy", "btw_centralisation", "top3_btw_share",
          "clustering", "rel_efficiency", "modularity"]


def random_comparison(A):
    real = global_measures(A)
    sims = {k: [] for k in TESTED}
    for _ in range(N_RANDOM):
        g = global_measures(randomise(A).tolist())
        for k in TESTED:
            sims[k].append(g[k])
    out = {}
    for k in TESTED:
        s = np.array(sims[k])
        lo = (s <= real[k]).mean()
        hi = (s >= real[k]).mean()
        out[k] = (real[k], s.mean(), s.std(), (real[k] - s.mean()) / s.std() if s.std() else 0,
                  min(1.0, 2 * min(lo, hi)))
    return out


# ------------------------------------------------------------- bootstrap

def bootstrap_ranks(A):
    """Resample every pass count as Poisson(observed) and record who comes top."""
    A = np.asarray(A)
    top = {"betweenness": np.zeros(N), "pagerank": np.zeros(N), "red_card": np.zeros(N)}
    for _ in range(N_BOOT):
        B = RNG.poisson(A)
        np.fill_diagonal(B, 0)
        m = corrected_metrics(B.tolist())
        top["betweenness"][int(np.argmax(m["betweenness"]))] += 1
        top["pagerank"][int(np.argmax(m["pagerank"]))] += 1
        drops = [removal_drop(B.tolist(), (i,)) for i in range(N)]
        top["red_card"][int(np.argmax(drops))] += 1
    return {k: v / N_BOOT for k, v in top.items()}


# ------------------------------------------------------------------ output

def pct(x):
    return f"{x:.0%}"


def red_card_figure(results):
    import matplotlib.pyplot as plt
    colours = {"FC Barcelona vs Man Utd (2011)": "#1d3c8f", "Man Utd vs FC Barcelona (2011)": "#c4122f",
               "FC Barcelona vs Juventus (2015)": "#1d3c8f", "Juventus vs FC Barcelona (2015)": "#222222"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.88, bottom=0.08, hspace=0.3, wspace=0.45)
    xmax = max(max(d for _, d, _ in rows) for rows in results.values()) * 1.18
    for ax, (team, rows) in zip(axes.flat, results.items()):
        names = TEAMS[team][0]
        rows = sorted(rows, key=lambda r: r[1])
        y = np.arange(N)
        ax.barh(y, [r[1] * 100 for r in rows], color=colours[team], height=0.62)
        ax.set_yticks(y, [names[r[0]] for r in rows], fontsize=9)
        for yy, r in zip(y, rows):
            ax.text(r[1] * 100 + 0.4, yy, f"{r[1]:.0%}", va="center", fontsize=8, color="#444")
        ax.set_xlim(0, xmax * 100)
        ax.set_title(team.replace(" vs ", " v "), loc="left", fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=8, colors="#666")
        ax.grid(axis="x", color="#e5e5e5", linewidth=0.8)
        ax.set_axisbelow(True)
    for ax in axes[1]:
        ax.set_xlabel("Drop in team passing efficiency when the player is removed (%)",
                      fontsize=9, color="#444")
    fig.suptitle("The red-card test: how much does each team need each player?",
                 x=0.03, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.03, 0.925, "Efficiency = how easily the ball can travel between the remaining ten "
             "players, using 1/passes as distance.", fontsize=9.5, color="#555")
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/red_card.png", dpi=180, facecolor="white")
    fig.savefig("figures/red_card.pdf", facecolor="white")
    plt.close(fig)


def main():
    teams = list(TEAMS)
    short = ["Barça 2011", "Man Utd 2011", "Barça 2015", "Juventus 2015"]
    print("# Extended network analysis\n")
    print("Generated by `network_extra.py`. All measures use 1/passes as distance and passes as strength.\n")

    # 1. team-level measures
    print("## 1. Team-level measures\n")
    G = {t: global_measures(TEAMS[t][1]) for t in teams}
    print("| Measure | " + " | ".join(short) + " |")
    print("|---|" + "---|" * len(teams))
    for k, (label, fmt) in MEASURE_LABELS.items():
        print(f"| {label} | " + " | ".join(fmt.format(G[t][k]) for t in teams) + " |")
    print()

    # 2. red card
    print("## 2. The red-card test\n")
    print("Each player is removed in turn. **Efficiency drop** compares how easily the ball "
          "moves between the other ten players with and without the removed player available to "
          "pass through, so it measures how much the others rely on that player as a link. **Involvement** is the share of the team's passes "
          "the player made or received.\n")
    rc = {}
    for t, s in zip(teams, short):
        names, A = TEAMS[t]
        rows, combos = red_card(A)
        rc[t] = rows
        print(f"### {s}\n")
        print("| Player | Efficiency drop | Involvement |")
        print("|---|---|---|")
        for i, drop, inv in sorted(rows, key=lambda r: -r[1]):
            print(f"| {names[i]} | {pct(drop)} | {pct(inv)} |")
        for size in (2, 3):
            res = combos[size]
            best = ", ".join(f"{' + '.join(names[i] for i in g)} ({pct(d)})" for g, d in res[:3])
            print(f"\nMost damaging {'pairs' if size == 2 else 'trios'}: {best}.")
        if s == "Barça 2011":
            trio = (5, 7, 9)   # Xavi, Iniesta, Messi - the trio the essay says to isolate
            pos = [g for g, _ in combos[3]].index(trio) + 1
            d = dict(combos[3])[trio]
            print(f"\nThe essay's trio (Xavi + Iniesta + Messi): {pct(d)} drop, ranked "
                  f"{pos} of {len(combos[3])} possible trios.")
        print()
    red_card_figure(rc)
    print("![Red-card test](figures/red_card.png)\n")

    # 3. passing units
    print("## 3. Passing units\n")
    print("Groups of players who passed mostly among themselves (Louvain community detection, "
          "passes in both directions added together).\n")
    for t, s in zip(teams, short):
        names = TEAMS[t][0]
        comms, q = passing_units(TEAMS[t][1])
        groups = sorted((sorted(names[i] for i in c) for c in comms), key=len, reverse=True)
        print(f"- **{s}** (modularity {q:.3f}): " + " · ".join("{" + ", ".join(g) + "}" for g in groups))
    print()

    # 4. random networks
    print("## 4. Compared with random networks\n")
    print(f"For each team, {N_RANDOM} random networks were generated in which every player makes "
          "and receives exactly as many passes as in the real match, but who they pass to is "
          "shuffled. If the real value sits far outside the random ones, the pattern comes from "
          "*who passed to whom*, not just from how busy each player was. z = distance from the "
          "random average in standard deviations; p = two-sided share of random networks at least "
          "as extreme (p < 0.05 means very unlikely by chance).\n")
    print("| Measure | " + " | ".join(short) + " |")
    print("|---|" + "---|" * len(teams))
    comp = {t: random_comparison(TEAMS[t][1]) for t in teams}
    for k in TESTED:
        label, fmt = MEASURE_LABELS[k]
        cells = []
        for t in teams:
            real, mu, sd, z, p = comp[t][k]
            cells.append(f"{fmt.format(real)} vs {fmt.format(mu)} (z = {z:+.1f}, p {'< 0.001' if p == 0 else '= ' + format(p, '.3f')})")
        print(f"| {label.split(' (')[0]} | " + " | ".join(cells) + " |")
    print()

    # 5. bootstrap
    print("## 5. How stable are the player rankings?\n")
    print(f"Each pass count was resampled {N_BOOT} times (Poisson noise around the observed count) "
          "and the calculations repeated. The table shows how often each player came out on top; "
          "only players who topped at least 5% of resamples are listed.\n")
    print("| Team | Top betweenness | Top PageRank | Top in red-card test |")
    print("|---|---|---|---|")
    for t, s in zip(teams, short):
        names = TEAMS[t][0]
        b = bootstrap_ranks(TEAMS[t][1])
        cell = lambda v: ", ".join(f"{names[i]} {v[i]:.0%}" for i in np.argsort(-v) if v[i] >= 0.05)
        print(f"| {s} | {cell(b['betweenness'])} | {cell(b['pagerank'])} | {cell(b['red_card'])} |")
    print()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
