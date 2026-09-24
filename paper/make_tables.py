"""Generate the LaTeX tables used in paper.tex from the analysis code in ../recalc.

Run from this folder:  python3 make_tables.py      (writes generated/*.tex)
The StatsBomb tables need the data first:  python3 ../recalc/fetch_statsbomb.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RECALC = os.path.join(HERE, "..", "recalc")
sys.path.insert(0, RECALC)
os.chdir(RECALC)

from data import TEAMS  # noqa: E402
from recalculate import FINALS, corrected_metrics, lambdas, outcome, pois  # noqa: E402
import network_extra as nx_extra  # noqa: E402
import model_check as mc  # noqa: E402
import statsbomb_analysis as sba  # noqa: E402

OUT = os.path.join(HERE, "generated")
os.makedirs(OUT, exist_ok=True)
N = 11
KEYS = list(TEAMS)
SHORT = ["Barça 2011", "Man Utd 2011", "Barça 2015", "Juventus 2015"]


def write(name, body):
    with open(os.path.join(OUT, name + ".tex"), "w", encoding="utf-8") as f:
        f.write(body.strip() + "\n")
    print("wrote", name)


def pct(x, d=0):
    return f"{100 * x:.{d}f}\\%"


# ---------------------------------------------------------------- matrices

def matrices():
    parts = []
    for key, short in zip(KEYS, SHORT):
        names, A = TEAMS[key]
        head = " & ".join(f"\\rotatebox{{70}}{{{n}}}" for n in names)
        rows = "\n".join(f"{names[i]} & " + " & ".join(str(v) for v in A[i]) + " \\\\" for i in range(N))
        parts.append(f"""
\\begin{{table}}[p]
\\centering\\small\\setlength{{\\tabcolsep}}{{3.5pt}}
\\caption{{Completed passes between starters, {short} (Opta). Row = passer, column = receiver.}}
\\label{{tab:matrix-{KEYS.index(key)}}}
\\begin{{tabular}}{{l*{{11}}{{r}}}}
\\toprule
From / to & {head} \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}
\\end{{table}}""")
    write("matrices", "\n".join(parts))


# ------------------------------------------------------ player centralities

def centralities():
    parts = []
    for key, short in zip(KEYS, SHORT):
        names, A = TEAMS[key]
        m = corrected_metrics(A)
        order = np.argsort(m["pagerank"])[::-1]
        rows = "\n".join(
            f"{names[i]} & {m['betweenness'][i]:.3f} & {m['closeness'][i]:.2f} & {m['pagerank'][i]:.3f} & "
            f"{m['clustering'][i]:.3f} \\\\" for i in order)
        parts.append(f"""
\\begin{{minipage}}[t]{{0.48\\textwidth}}\\centering\\footnotesize
\\textbf{{{short}}}\\\\[2pt]
\\begin{{tabular}}{{lrrrr}}
\\toprule
Player & $C_B$ & $C_C$ & PR & $c^w$ \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}
\\end{{minipage}}""")
    body = parts[0] + "\\hfill" + parts[1] + "\n\n\\vspace{10pt}\n" + parts[2] + "\\hfill" + parts[3]
    write("centralities", f"""
\\begin{{table}}[p]
\\centering
\\caption{{Corrected player measures from the Opta passing tables, sorted by PageRank. $C_B$ = normalised betweenness, $C_C$ = closeness (in units of passes), PR = weighted PageRank ($p = 0.85$), $c^w$ = weighted clustering.}}
\\label{{tab:centralities}}
{body}
\\end{{table}}""")


# ---------------------------------------------------------- global measures

def global_table():
    G = {k: nx_extra.global_measures(TEAMS[k][1]) for k in KEYS}
    labels = [
        ("passes", "Completed passes between starters", "{:.0f}"),
        ("density", "Density", "{:.2f}"),
        ("reciprocity", "Reciprocity", "{:.2f}"),
        ("entropy", "Passing entropy (normalised)", "{:.3f}"),
        ("btw_centralisation", "Betweenness centralisation", "{:.3f}"),
        ("top3_btw_share", "Betweenness held by top three", "pct"),
        ("involvement_gini", "Involvement inequality (Gini)", "{:.3f}"),
        ("pagerank_gini", "PageRank inequality (Gini)", "{:.3f}"),
        ("clustering", "Mean weighted clustering", "{:.3f}"),
        ("rel_efficiency", "Relative efficiency", "{:.2f}"),
        ("modularity", "Modularity of passing units", "{:.3f}"),
    ]
    rows = "\n".join(
        f"{lab} & " + " & ".join(pct(G[k][key]) if fmt == "pct" else fmt.format(G[k][key]) for k in KEYS) + " \\\\"
        for key, lab, fmt in labels)
    write("global", f"""
\\begin{{table}}[t]
\\centering\\small
\\caption{{Team-level network measures (Opta passing tables).}}
\\label{{tab:global}}
\\begin{{tabular}}{{lrrrr}}
\\toprule
Measure & {' & '.join(SHORT)} \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}
\\end{{table}}""")


# ------------------------------------------------------------ red-card test

def red_card():
    rows = []
    for key, short in zip(KEYS, SHORT):
        names, A = TEAMS[key]
        singles, combos = nx_extra.red_card(A)
        s = sorted(singles, key=lambda r: -r[1])[:2]
        single = ", ".join(f"{names[i]} ({pct(d)})" for i, d, _ in s)
        pair = combos[2][0]
        trio = combos[3][0]
        rows.append(f"{short} & {single} & {' + '.join(names[i] for i in pair[0])} ({pct(pair[1])}) & "
                    f"{' + '.join(names[i] for i in trio[0])} ({pct(trio[1])}) \\\\")
        if short == "Barça 2011":
            t = (5, 7, 9)
            rank = [g for g, _ in combos[3]].index(t) + 1
            with open(os.path.join(OUT, "trio_rank.tex"), "w") as f:
                f.write(f"{pct(dict(combos[3])[t])} drop, the {rank}th largest of the {len(combos[3])} possible trios")
    write("redcard", f"""
\\begin{{table}}[t]
\\centering\\small
\\caption{{The red-card test: fall in passing efficiency between the remaining players when a player, pair or trio is removed.}}
\\label{{tab:redcard}}
\\begin{{tabularx}}{{\\textwidth}}{{lXXX}}
\\toprule
Team & Most needed players & Most damaging pair & Most damaging trio \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabularx}}
\\end{{table}}""")


def units():
    rows = []
    for key, short in zip(KEYS, SHORT):
        names, A = TEAMS[key]
        comms, q = nx_extra.passing_units(A)
        groups = sorted((sorted(names[i] for i in c) for c in comms), key=len, reverse=True)
        rows.append(f"{short} & {q:.3f} & " + "; ".join(", ".join(g) for g in groups) + " \\\\")
    write("units", f"""
\\begin{{table}}[t]
\\centering\\small
\\caption{{Passing units found by Louvain community detection (passes in both directions combined).}}
\\label{{tab:units}}
\\begin{{tabularx}}{{\\textwidth}}{{lrX}}
\\toprule
Team & $Q$ & Units \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabularx}}
\\end{{table}}""")


def null_model():
    labels = {"reciprocity": "Reciprocity", "entropy": "Passing entropy",
              "btw_centralisation": "Betweenness centralisation", "top3_btw_share": "Top-three betweenness share",
              "clustering": "Mean weighted clustering", "rel_efficiency": "Relative efficiency",
              "modularity": "Modularity"}
    comp = {k: nx_extra.random_comparison(TEAMS[k][1]) for k in KEYS}
    rows = []
    for m, lab in labels.items():
        cells = []
        for k in KEYS:
            real, mu, sd, z, p = comp[k][m]
            star = "$^{*}$" if p < 0.05 else ""
            cells.append(f"${z:+.1f}${star}")
        rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    write("null", f"""
\\begin{{table}}[t]
\\centering\\small
\\caption{{Real networks compared with {nx_extra.N_RANDOM} random networks per team that keep every player's number of passes made and received. Entries are $z$-scores (standard deviations from the random mean); $^{{*}}$ marks $p < 0.05$ (two-sided).}}
\\label{{tab:null}}
\\begin{{tabular}}{{lrrrr}}
\\toprule
Measure & {' & '.join(SHORT)} \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}""")


def bootstrap():
    rows = []
    for key, short in zip(KEYS, SHORT):
        names, A = TEAMS[key]
        b = nx_extra.bootstrap_ranks(A)
        cell = lambda v: ", ".join(f"{names[i]} {pct(v[i])}" for i in np.argsort(-v) if v[i] >= 0.05)
        rows.append(f"{short} & {cell(b['betweenness'])} & {cell(b['pagerank'])} & {cell(b['red_card'])} \\\\")
    write("bootstrap", f"""
\\begin{{table}}[t]
\\centering\\small
\\caption{{How often each player comes out on top when every pass count is resampled {nx_extra.N_BOOT} times from a Poisson distribution centred on the observed count (players topping fewer than 5\\% of resamples omitted).}}
\\label{{tab:bootstrap}}
\\begin{{tabularx}}{{\\textwidth}}{{lXXX}}
\\toprule
Team & Top betweenness & Top PageRank & Top in red-card test \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabularx}}
\\end{{table}}""")


# --------------------------------------------------------------- Poisson

def poisson_finals():
    rows = []
    for title, f in FINALS.items():
        year = "2011" if "2010" in title else "2015"
        avg_a = f["comp_goals"] / (2 * f["comp_matches"])
        la_A, lb_A, _ = lambdas(f["a_scored"], f["a_conceded"], f["b_scored"], f["b_conceded"], 13, avg_a)
        avg_b = (f["comp_goals"] - 4) / (2 * (f["comp_matches"] - 1))
        la_B, lb_B, _ = lambdas(f["a_scored"] - 3, f["a_conceded"] - 1, f["b_scored"] - 1, f["b_conceded"] - 3,
                                12, avg_b)
        for name, avg, la, lb in (("Final included", avg_a, la_A, lb_A),
                                  ("Final excluded", avg_b, la_B, lb_B)):
            w, d, l = outcome(la, lb)
            rows.append(f"{year} & {name} & {avg:.3f} & {la:.3f} & {lb:.3f} & {pct(w, 1)} & {pct(d, 1)} & "
                        f"{pct(l, 1)} & {pct(pois(la, 3) * pois(lb, 1), 2)} \\\\")
        rows.append("\\addlinespace")
    write("poisson_finals", f"""
\\begin{{table}}[t]
\\centering\\small
\\caption{{The Poisson model for the two finals. $\\bar g$ is the league average, $\\lambda_B$ Barcelona's expected goals and $\\lambda_O$ the opponent's. ``Final included'' estimates team strengths from all 13 games including the final itself; ``final excluded'' uses only the 12 games before it, which is the genuine forecast. Win, draw and loss are from Barcelona's point of view, assuming independent Poisson scores.}}
\\label{{tab:poisson-finals}}
\\begin{{tabular}}{{llrrrrrrr}}
\\toprule
Final & Version & $\\bar g$ & $\\lambda_B$ & $\\lambda_O$ & Win & Draw & Loss & $P(3\\text{{--}}1)$ \\\\
\\midrule
{chr(10).join(rows[:-1])}
\\bottomrule
\\end{{tabular}}
\\end{{table}}""")


def model_check():
    seasons = mc.load_all()
    rows = mc.evaluate(seasons)
    S = mc.summarise(rows)
    n = len(rows)
    counts = np.bincount([r[1] for r in rows], minlength=3)
    lines = []
    for name in mc.MODELS:
        s = S[name]
        fav = "--" if name == "Guess 1/3 each" else pct(s["correct"].mean(), 1)
        lines.append(f"{name} & {s['rps'].mean():.4f} & {s['logloss'].mean():.3f} & {s['brier'].mean():.3f} & {fav} \\\\")
    base = "Home/draw/away rates so far"
    diffs = []
    for a, b, lab in (("Basic Poisson model", base, "Basic Poisson $-$ base rates"),
                      ("Basic Poisson + home advantage", base, "Basic Poisson + home $-$ base rates"),
                      ("Fitted Dixon–Coles model", base, "Dixon--Coles $-$ base rates"),
                      ("Basic Poisson + home advantage", "Basic Poisson model", "Adding home advantage"),
                      ("Fitted Dixon–Coles model", "Basic Poisson + home advantage", "Dixon--Coles $-$ basic Poisson + home")):
        d, lo, hi = mc.paired_ci(S[a]["rps"], S[b]["rps"])
        diffs.append(f"{lab} & ${d:+.4f}$ & $[{lo:+.4f},\\ {hi:+.4f}]$ \\\\")
    draws = {name: np.mean([r[2][name][0][1] for r in rows]) for name in list(mc.MODELS)[2:]}
    mc.calibration_figure(rows)
    with open(os.path.join(OUT, "modelcheck_facts.tex"), "w") as f:
        f.write(f"\\newcommand{{\\nKO}}{{{n}}}\n\\newcommand{{\\nSeasons}}{{{len(seasons)}}}\n"
                f"\\newcommand{{\\nHome}}{{{counts[0]}}}\\newcommand{{\\nDraw}}{{{counts[1]}}}\\newcommand{{\\nAway}}{{{counts[2]}}}\n"
                f"\\newcommand{{\\drawBasic}}{{{pct(draws['Basic Poisson model'], 1)}}}\n"
                f"\\newcommand{{\\drawDC}}{{{pct(draws['Fitted Dixon–Coles model'], 1)}}}\n"
                f"\\newcommand{{\\drawActual}}{{{pct(counts[1] / n, 1)}}}\n")
    write("modelcheck", f"""
\\begin{{table}}[t]
\\centering\\small
\\caption{{Forecasting {n} Champions League knockout matches (2011/12--2023/24), each from the same season's earlier results. Lower is better for RPS, log loss and Brier score.}}
\\label{{tab:modelcheck}}
\\begin{{tabular}}{{lrrrr}}
\\toprule
Model & RPS & Log loss & Brier & Favourite won \\\\
\\midrule
{chr(10).join(lines).replace("Dixon–Coles", "Dixon--Coles")}
\\bottomrule
\\end{{tabular}}

\\vspace{{8pt}}
\\begin{{tabular}}{{lrr}}
\\toprule
Difference in mean RPS & Estimate & 95\\% bootstrap interval \\\\
\\midrule
{chr(10).join(diffs)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}""")


# --------------------------------------------------------------- StatsBomb

def statsbomb():
    events = {m: sba.load(m) for m in (18236, 18242)}
    sides = [sba.side_data(m, t, events[m]) for (m, t) in sba.SIDES]
    rows = []
    for sd in sides:
        o, s, r, _ = sba.compare_with_opta(sd)
        rows.append(f"{sba.SHORT[sd['key']]} & {'-'.join(str(sd['formation']))} & {o} & {s} & {r:.2f} \\\\")
    write("sb_check", f"""
\\begin{{table}}[t]
\\centering\\small
\\caption{{Opta passing tables against StatsBomb event data: completed passes between starters, and the correlation between the two sources over all 110 possible passing links.}}
\\label{{tab:sbcheck}}
\\begin{{tabular}}{{lrrrr}}
\\toprule
Team & StatsBomb formation & Opta & StatsBomb & Correlation \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}""")

    halves = []
    for (m, t) in sba.SIDES:
        key = sba.SIDES[(m, t)]
        names = TEAMS[key][0]
        for half in (1, 2):
            h = sba.side_data(m, t, events[m], keep=lambda e, p=half: e["period"] == p)
            passes = [e for e in events[m] if e["type"]["name"] == "Pass" and e["team"]["name"] == t
                      and e["period"] == half and "outcome" not in e["pass"]]
            ft = np.mean([e["location"][0] >= 80 for e in passes])
            mm = corrected_metrics(h["A"].tolist())
            g = nx_extra.global_measures(h["A"].tolist())
            top = ", ".join(names[i] for i in np.argsort(mm["pagerank"])[::-1][:2])
            halves.append(f"{sba.SHORT[key] if half == 1 else ''} & {half} & {h['A'].sum()} & {pct(ft)} & "
                          f"{top} & {g['btw_centralisation']:.3f} \\\\")
    write("sb_halves", f"""
\\begin{{table}}[t]
\\centering\\small
\\caption{{First half against second half (StatsBomb). Final-third share = proportion of a team's completed passes that started in the attacking third.}}
\\label{{tab:halves}}
\\begin{{tabular}}{{lrrrlr}}
\\toprule
Team & Half & Passes & Final third & Top PageRank & Betweenness centralisation \\\\
\\midrule
{chr(10).join(halves)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}""")

    pre = {18236: (0.240, 0.410, 0.351), 18242: (0.380, 0.336, 0.283)}
    xg_rows = []
    for m in (18236, 18242):
        barca = sba.shots(events[m], "Barcelona")
        opp_name = "Manchester United" if m == 18236 else "Juventus"
        opp = sba.shots(events[m], opp_name)
        gb, go = sba.simulate(barca, opp)
        actual = (sum(s["goal"] for s in barca), sum(s["goal"] for s in opp))
        exact = np.mean((gb == actual[0]) & (go == actual[1]))
        year = "2011" if m == 18236 else "2015"
        xg_rows.append(
            f"{year} & {len(barca)}--{len(opp)} & {sum(s['xg'] for s in barca):.2f}--{sum(s['xg'] for s in opp):.2f} & "
            f"{actual[0]}--{actual[1]} & {pct(np.mean(gb > go))} & {pct(np.mean(gb == go))} & {pct(np.mean(gb < go))} & "
            f"{pct(exact, 1)} & {pct(pre[m][0])} \\\\")
    write("sb_xg", f"""
\\begin{{table}}[t]
\\centering\\footnotesize\\setlength{{\\tabcolsep}}{{4pt}}
\\caption{{Shots and expected goals (Barcelona first). The simulated probabilities replay every chance 100{{,}}000 times; the last column is the pre-match Poisson forecast (final excluded, Table~\\ref{{tab:poisson-finals}}).}}
\\label{{tab:xg}}
\\begin{{tabular}}{{lrrrrrrrr}}
\\toprule
Final & Shots & xG & Goals & Barça win & Draw & Opp.\\ win & $P$(3--1) & Pre-match Barça win \\\\
\\midrule
{chr(10).join(xg_rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}""")

    ch_rows = []
    from scipy.stats import spearmanr
    for sd in sides:
        m, t, key = sd["match"], sd["team"], sd["key"]
        names = TEAMS[key][0]
        xi, _ = sba.starters(events[m], t)
        order = sba.LAYOUT[key]["numbers"]
        chain, build = sba.xg_chain(events[m], t, xi)
        c = np.zeros(N)
        b = np.zeros(N)
        for pid, num in xi.items():
            c[order.index(num)] = chain[pid]
            b[order.index(num)] = build[pid]
        pr = np.array(corrected_metrics(sd["A"].tolist())["pagerank"])
        rho, p = spearmanr(pr, b)
        ch_rows.append(f"{sba.SHORT[key]} & " + ", ".join(f"{names[i]} {c[i]:.2f}" for i in np.argsort(-c)[:3]) + " & "
                       + ", ".join(f"{names[i]} {b[i]:.2f}" for i in np.argsort(-b)[:3]) + f" & ${rho:.2f}$ & {p:.2f} \\\\")
    write("sb_chain", f"""
\\begin{{table}}[t]
\\centering\\small
\\caption{{Who was involved in the chances? xGChain and xGBuildup for the top three players in each team, and the Spearman rank correlation between PageRank (StatsBomb network) and xGBuildup across the eleven starters.}}
\\label{{tab:chain}}
\\begin{{tabularx}}{{\\textwidth}}{{lXXrr}}
\\toprule
Team & Highest xGChain & Highest xGBuildup & $\\rho$ & $p$ \\\\
\\midrule
{chr(10).join(ch_rows)}
\\bottomrule
\\end{{tabularx}}
\\end{{table}}""")


if __name__ == "__main__":
    matrices()
    centralities()
    global_table()
    red_card()
    units()
    null_model()
    bootstrap()
    poisson_finals()
    model_check()
    statsbomb()
