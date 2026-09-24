"""Recalculate the extended essay's network and Poisson results.

Run:  python3 recalculate.py > RESULTS.md
Needs: networkx, scipy
"""
from math import exp, factorial

import networkx as nx

from data import ESSAY_RESULTS, TEAMS

N = 11


def build_graph(A):
    """Directed graph with two edge attributes:
    passes   - strength (more passes = stronger link), used by PageRank/clustering
    distance - 1/passes (more passes = closer), used by shortest-path measures
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(N):
            if A[i][j]:
                G.add_edge(i, j, passes=A[i][j], distance=1 / A[i][j])
    return G


def corrected_metrics(A):
    G = build_graph(A)
    # Betweenness with 1/passes distances, normalised by 1/((N-1)(N-2)) = 1/90 as in the essay.
    btw = nx.betweenness_centrality(G, weight="distance", normalized=False)
    btw = [btw[v] / ((N - 1) * (N - 2)) for v in range(N)]
    # Closeness = (N-1) / sum of geodesic distances from the player (same form as the essay).
    clo = []
    for v in range(N):
        d = nx.single_source_dijkstra_path_length(G, v, weight="distance")
        clo.append((len(d) - 1) / sum(d.values()))
    # Weighted PageRank, p = 0.85: a player is important if important players pass to him often.
    pr = nx.pagerank(G, alpha=0.85, weight="passes")
    pr = [pr[v] for v in range(N)]
    # Weighted directed clustering (Fagiolo 2007): the essay's cube-root / max(A) formula.
    cl = nx.clustering(G, weight="passes")
    cl = [cl[v] for v in range(N)]
    return dict(betweenness=btw, closeness=clo, pagerank=pr, clustering=cl)


def rank(values):
    order = sorted(range(N), key=lambda i: -values[i])
    r = [0] * N
    for pos, i in enumerate(order, 1):
        r[i] = pos
    return r


def network_section():
    print("## Part 1 - Network measures\n")
    print("Distances for betweenness and closeness are **1/passes**; PageRank and clustering use "
          "**passes** as strength. Betweenness is normalised by 1/90 (so it lies between 0 and 1). "
          "\"Old\" is the value printed in the essay. The number in brackets is the player's rank in the team.\n")
    for team, (names, A) in TEAMS.items():
        new = corrected_metrics(A)
        old = ESSAY_RESULTS[team]
        print(f"### {team}\n")
        print("| Player | Betweenness old → new | Closeness old → new | PageRank old → new | Clustering old → new |")
        print("|---|---|---|---|---|")
        ranks = {k: (rank(old[k]), rank(new[k])) for k in new}
        for i, name in enumerate(names):
            cells = []
            for k, fmt in [("betweenness", ".3f"), ("closeness", ".2f"), ("pagerank", ".3f"), ("clustering", ".3f")]:
                ro, rn = ranks[k][0][i], ranks[k][1][i]
                cells.append(f"{old[k][i]:{fmt}} ({ro}) → **{new[k][i]:{fmt}} ({rn})**")
            print(f"| {name} | " + " | ".join(cells) + " |")
        b = new["betweenness"]
        top = sorted(range(N), key=lambda i: -b[i])[:3]
        share = sum(b[i] for i in top) / sum(b) if sum(b) else 0
        zero = sum(1 for x in b if x == 0)
        print(f"\nTeam summary: total betweenness {sum(b):.3f}; top three "
              f"({', '.join(names[i] for i in top)}) hold {share:.0%} of it; "
              f"{zero} players have zero. Mean closeness {sum(new['closeness']) / N:.2f}; "
              f"total passes {sum(map(sum, A))}.\n")


# ---------------------------------------------------------------- Poisson model

def pois(lam, k):
    return lam ** k * exp(-lam) / factorial(k)


def outcome(lam_a, lam_b, kmax=15):
    win = draw = 0.0
    for i in range(kmax):
        for j in range(kmax):
            p = pois(lam_a, i) * pois(lam_b, j)
            if i > j:
                win += p
            elif i == j:
                draw += p
    return win, draw, 1 - win - draw


# Per-team totals implied by the essay's per-game figures (all are exact fractions of 12 or 13 games),
# i.e. group stage to final inclusive. Finalists played 13 games; the final itself ended 3-1.
FINALS = {
    "2010/11 final: FC Barcelona vs Man Utd": dict(
        a="FC Barcelona", b="Man Utd",
        a_scored=30, a_conceded=9, b_scored=19, b_conceded=7,
        comp_goals=355, comp_matches=125,       # whole competition, group stage onwards
        essay=dict(avg_scored=1.31, avg_conceded=1.59, lam_a=0.784, lam_b=0.631)),
    "2014/15 final: FC Barcelona vs Juventus": dict(
        a="FC Barcelona", b="Juventus",
        a_scored=31, a_conceded=11, b_scored=17, b_conceded=10,
        comp_goals=361, comp_matches=125,
        essay=dict(avg_scored=1.33, avg_conceded=1.58, lam_a=1.16, lam_b=0.704)),
}


def lambdas(a_s, a_c, b_s, b_c, games, avg):
    att_a, def_a = a_s / games / avg, a_c / games / avg
    att_b, def_b = b_s / games / avg, b_c / games / avg
    return avg * att_a * def_b, avg * att_b * def_a, (att_a, def_a, att_b, def_b)


def poisson_section():
    print("## Part 2 - Poisson model\n")
    print("λ = league average × attack strength × opponent's defensive weakness. Win/draw/loss "
          "assumes the two teams' goals are independent Poisson variables.\n")
    for title, f in FINALS.items():
        e = f["essay"]
        # Version A: the essay's team data, but one consistent league average (scored = conceded).
        avg_a = f["comp_goals"] / (2 * f["comp_matches"])
        la_A, lb_A, _ = lambdas(f["a_scored"], f["a_conceded"], f["b_scored"], f["b_conceded"], 13, avg_a)
        # Version B: also remove the final itself (3-1) so the data does not contain the result.
        avg_b = (f["comp_goals"] - 4) / (2 * (f["comp_matches"] - 1))
        la_B, lb_B, s = lambdas(f["a_scored"] - 3, f["a_conceded"] - 1,
                                f["b_scored"] - 1, f["b_conceded"] - 3, 12, avg_b)
        versions = [
            ("Essay (inconsistent averages)", e["avg_scored"], e["lam_a"], e["lam_b"]),
            ("A: consistent league average", avg_a, la_A, lb_A),
            ("B: A + final removed from data", avg_b, la_B, lb_B),
        ]
        print(f"### {title} (actual result 3-1)\n")
        print(f"Version B strengths: {f['a']} attack {s[0]:.2f}, defence {s[1]:.2f}; "
              f"{f['b']} attack {s[2]:.2f}, defence {s[3]:.2f}.\n")
        print(f"| Version | League avg | λ {f['a']} | λ {f['b']} | P(3-1) | P({f['a']} win) | P(draw) | P({f['b']} win) | Most likely score |")
        print("|---|---|---|---|---|---|---|---|---|")
        for name, avg, la, lb in versions:
            w, d, l = outcome(la, lb)
            best = max(((i, j) for i in range(6) for j in range(6)), key=lambda t: pois(la, t[0]) * pois(lb, t[1]))
            print(f"| {name} | {avg:.3f} | {la:.3f} | {lb:.3f} | {pois(la, 3) * pois(lb, 1):.2%} | "
                  f"{w:.1%} | {d:.1%} | {l:.1%} | {best[0]}-{best[1]} |")
        print()
        print(f"Poisson table, version B:\n")
        print("| Team | 0 | 1 | 2 | 3 | 4 |")
        print("|---|---|---|---|---|---|")
        for name, lam in [(f["a"], la_B), (f["b"], lb_B)]:
            print(f"| {name} (λ = {lam:.3f}) | " + " | ".join(f"{pois(lam, k):.3f}" for k in range(5)) + " |")
        print()


if __name__ == "__main__":
    print("# Recalculated results\n")
    print("Generated by `recalculate.py` from the passing tables in the essay (see `data.py`).\n")
    network_section()
    poisson_section()
