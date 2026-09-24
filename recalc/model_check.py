"""How well does a Poisson attack-defence model predict Champions League matches?

Every knockout match from 2011/12 to 2023/24 is predicted using only the results
of that season played before it (group stage + earlier knockout games), and the
predictions are scored against the 90-minute result.

Run:  python3 model_check.py > MODEL_CHECK.md      (also writes figures/calibration.png)
Needs: numpy, scipy, matplotlib
Data: cl_results/*.txt from openfootball/champions-league (CC0), commit abfaedd.
"""
import glob
import os
import re
from math import exp, factorial, log

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_GOALS = 10
RIDGE_SD = 0.3      # prior spread of team strengths (log scale) for the fitted model

MATCH = re.compile(r"^\s*(?:\d{1,2}[:.]\d{2}\s+)?(?P<h>\S.*?\([A-Z]{3}\))\s+v\s+(?P<a>\S.*?\([A-Z]{3}\))\s+(?P<rest>\d.*)$")
SCORE = re.compile(r"(\d+)-(\d+)")


# ------------------------------------------------------------------ data

def load_season(path):
    season = os.path.basename(path)[:-4]
    matches, stage = [], None
    for line in open(path, encoding="utf-8"):
        if line.startswith("▪"):
            head = line[1:].strip()
            stage = "group" if re.search(r"Group|Gruppe|Matchday", head) else head.split(",")[-1].strip()
            continue
        m = MATCH.search(line)
        if not m:
            continue
        rest = m["rest"]
        if "a.e.t." in rest:                       # 90-minute score is first in brackets
            hg, ag = SCORE.search(rest[rest.index("("):]).groups()
        else:
            hg, ag = SCORE.match(rest).groups()
        neutral = stage == "Final" or (season == "2019-20" and stage in ("Quarterfinals", "Semifinals"))
        matches.append(dict(season=season, stage=stage, home=m["h"].strip(), away=m["a"].strip(),
                            hg=int(hg), ag=int(ag), neutral=neutral))
    return matches


def load_all():
    return {os.path.basename(p)[:-4]: load_season(p)
            for p in sorted(glob.glob(os.path.join(HERE, "cl_results", "*.txt")))}


# ---------------------------------------------------------------- helpers

def pois(lam, k):
    return lam ** k * exp(-lam) / factorial(k)


def score_matrix(lh, la, rho=0.0):
    M = np.outer([pois(lh, k) for k in range(MAX_GOALS)], [pois(la, k) for k in range(MAX_GOALS)])
    if rho:                                        # Dixon-Coles low-score correction
        M[0, 0] *= 1 - lh * la * rho
        M[0, 1] *= 1 + lh * rho
        M[1, 0] *= 1 + la * rho
        M[1, 1] *= 1 - rho
    return M / M.sum()


def hda(M):
    return np.array([np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()])


def team_rates(train):
    s = {}
    for m in train:
        for t, f, a in ((m["home"], m["hg"], m["ag"]), (m["away"], m["ag"], m["hg"])):
            g, sc, co = s.get(t, (0, 0, 0))
            s[t] = (g + 1, sc + f, co + a)
    return s


# ----------------------------------------------------------------- models

def m_uniform(train, m):
    return np.array([1 / 3, 1 / 3, 1 / 3]), None


def m_base_rates(train, m):
    res = np.array([[x["hg"] > x["ag"], x["hg"] == x["ag"], x["hg"] < x["ag"]]
                    for x in train if not x["neutral"]], dtype=float).mean(0)
    if m["neutral"]:
        res = np.array([(res[0] + res[2]) / 2, res[1], (res[0] + res[2]) / 2])
    return res, None


def basic_lambdas(train, m, home_adv):
    s = team_rates(train)
    goals = sum(x["hg"] + x["ag"] for x in train)
    avg = goals / (2 * len(train))                 # one league average: scored = conceded
    gh, sh, ch = s[m["home"]]
    ga, sa, ca = s[m["away"]]
    att_h, def_h = sh / gh / avg, ch / gh / avg
    att_a, def_a = sa / ga / avg, ca / ga / avg
    if home_adv and not m["neutral"]:
        nn = [x for x in train if not x["neutral"]]
        avg_h = sum(x["hg"] for x in nn) / len(nn)
        avg_a = sum(x["ag"] for x in nn) / len(nn)
    else:
        avg_h = avg_a = avg
    return avg_h * att_h * def_a, avg_a * att_a * def_h


def m_basic(train, m):
    lh, la = basic_lambdas(train, m, home_adv=False)
    M = score_matrix(lh, la)
    return hda(M), M


def m_basic_home(train, m):
    lh, la = basic_lambdas(train, m, home_adv=True)
    M = score_matrix(lh, la)
    return hda(M), M


def fit_dixon_coles(train):
    teams = sorted({x["home"] for x in train} | {x["away"] for x in train})
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    H = np.array([idx[x["home"]] for x in train])
    A = np.array([idx[x["away"]] for x in train])
    hg = np.array([x["hg"] for x in train])
    ag = np.array([x["ag"] for x in train])
    home = np.array([0.0 if x["neutral"] else 1.0 for x in train])
    lf_h = np.array([np.log(float(factorial(k))) for k in hg])
    lf_a = np.array([np.log(float(factorial(k))) for k in ag])

    def unpack(p):
        return p[0], p[1], p[2], p[3:3 + n], p[3 + n:3 + 2 * n]

    def nll(p):
        mu, h, rho, att, dfn = unpack(p)
        lh = np.exp(mu + h * home + att[H] - dfn[A])
        la = np.exp(mu + att[A] - dfn[H])
        ll = hg * np.log(lh) - lh - lf_h + ag * np.log(la) - la - lf_a
        tau = np.ones_like(lh)
        tau = np.where((hg == 0) & (ag == 0), 1 - lh * la * rho, tau)
        tau = np.where((hg == 0) & (ag == 1), 1 + lh * rho, tau)
        tau = np.where((hg == 1) & (ag == 0), 1 + la * rho, tau)
        tau = np.where((hg == 1) & (ag == 1), 1 - rho, tau)
        ll = ll + np.log(np.clip(tau, 1e-9, None))
        penalty = (att ** 2 + dfn ** 2).sum() / (2 * RIDGE_SD ** 2)
        return -ll.sum() + penalty

    p0 = np.zeros(3 + 2 * n)
    p0[0] = np.log(max((hg.sum() + ag.sum()) / (2 * len(train)), 0.1))
    bounds = [(None, None), (-1, 1), (-0.3, 0.3)] + [(-2, 2)] * (2 * n)
    res = minimize(nll, p0, method="L-BFGS-B", bounds=bounds)
    mu, h, rho, att, dfn = unpack(res.x)
    return dict(idx=idx, mu=mu, h=h, rho=rho, att=att, dfn=dfn)


_fit_cache = {}


def m_dixon_coles(train, m):
    key = (m["season"], len(train))
    if key not in _fit_cache:
        _fit_cache[key] = fit_dixon_coles(train)
    f = _fit_cache[key]
    i, j = f["idx"][m["home"]], f["idx"][m["away"]]
    home = 0.0 if m["neutral"] else f["h"]
    lh = exp(f["mu"] + home + f["att"][i] - f["dfn"][j])
    la = exp(f["mu"] + f["att"][j] - f["dfn"][i])
    M = score_matrix(lh, la, f["rho"])
    return hda(M), M


MODELS = {
    "Guess 1/3 each": m_uniform,
    "Home/draw/away rates so far": m_base_rates,
    "Basic Poisson model": m_basic,
    "Basic Poisson + home advantage": m_basic_home,
    "Fitted Dixon–Coles model": m_dixon_coles,
}


# ---------------------------------------------------------------- scoring

def rps(p, outcome):
    o = np.zeros(3)
    o[outcome] = 1
    return ((np.cumsum(p)[:2] - np.cumsum(o)[:2]) ** 2).sum() / 2


def evaluate(seasons):
    rows = []
    for season, matches in seasons.items():
        for k, m in enumerate(matches):
            if m["stage"] == "group":
                continue
            train = matches[:k]
            outcome = 0 if m["hg"] > m["ag"] else 1 if m["hg"] == m["ag"] else 2
            preds = {}
            for name, fn in MODELS.items():
                p, M = fn(train, m)
                preds[name] = (p, M)
            rows.append((m, outcome, preds))
    return rows


def summarise(rows):
    out = {}
    for name in MODELS:
        P = np.array([r[2][name][0] for r in rows])
        y = np.array([r[1] for r in rows])
        out[name] = dict(
            rps=np.array([rps(p, o) for p, o in zip(P, y)]),
            logloss=-np.log(P[np.arange(len(y)), y]),
            brier=((P - np.eye(3)[y]) ** 2).sum(1),
            correct=(P.argmax(1) == y).astype(float),
            exact=np.array([r[2][name][1][min(r[0]["hg"], MAX_GOALS - 1), min(r[0]["ag"], MAX_GOALS - 1)]
                            if r[2][name][1] is not None else np.nan for r in rows]),
        )
    return out


def paired_ci(a, b, n=5000, seed=1):
    rng = np.random.default_rng(seed)
    d = a - b
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return d.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def calibration_figure(rows):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.82, bottom=0.13, wspace=0.28)
    bins = np.linspace(0, 1, 6)
    colours = {"Basic Poisson model": "#2a6fdb",
               "Basic Poisson + home advantage": "#e08a1e",
               "Fitted Dixon–Coles model": "#1a9e77"}
    markers = {"Basic Poisson model": "o", "Basic Poisson + home advantage": "s",
               "Fitted Dixon–Coles model": "^"}
    for ax, (col, label) in zip(axes, [(0, "home win"), (1, "draw")]):
        ax.plot([0, 1], [0, 1], color="#999", linewidth=1, linestyle="--", label="perfect calibration")
        for name, c in colours.items():
            p = np.array([r[2][name][0][col] for r in rows])
            y = np.array([r[1] == col for r in rows], dtype=float)
            xs, ys, ns = [], [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                sel = (p >= lo) & (p < hi)
                if sel.sum() >= 10:
                    xs.append(p[sel].mean()); ys.append(y[sel].mean()); ns.append(sel.sum())
            ax.plot(xs, ys, color=c, linewidth=2, marker=markers[name], markersize=7, label=name)
        ax.set_xlim(0, 1 if col == 0 else 0.5)
        ax.set_ylim(0, 1 if col == 0 else 0.5)
        ax.set_xlabel(f"Predicted probability of a {label}", fontsize=10, color="#444")
        ax.set_ylabel(f"Share that ended in a {label}", fontsize=10, color="#444")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#eee")
        ax.set_axisbelow(True)
        ax.set_title(label.capitalize(), loc="left", fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=8.5, frameon=False, loc="upper left")
    fig.suptitle("Are the predicted probabilities right? (knockout matches 2011/12–2023/24)",
                 x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.02, 0.9, "Predictions grouped into bins of 0.2 (bins with at least 10 matches). "
             "Points on the dashed line mean the model's probabilities match reality.",
             fontsize=9, color="#555")
    fig.savefig(os.path.join(HERE, "figures", "calibration.png"), dpi=180, facecolor="white")
    fig.savefig(os.path.join(HERE, "figures", "calibration.pdf"), facecolor="white")
    plt.close(fig)


def main():
    seasons = load_all()
    rows = evaluate(seasons)
    S = summarise(rows)
    n = len(rows)
    counts = np.bincount([r[1] for r in rows], minlength=3)
    print("# Does the Poisson model work?\n")
    print(f"Generated by `model_check.py`. {n} knockout matches from {len(seasons)} seasons "
          f"({min(seasons)} to {max(seasons)}), each predicted using only that season's earlier results. "
          f"Scored on the 90-minute result: {counts[0]} home wins, {counts[1]} draws, {counts[2]} away wins. "
          "Data: [openfootball/champions-league](https://github.com/openfootball/champions-league) (CC0). "
          "The 2010/11 season is not in that dataset.\n")
    print("## Scores\n")
    print("Lower is better for RPS, log loss and Brier score. **RPS** (ranked probability score) is the "
          "standard measure for football forecasts: it rewards putting probability near the right "
          "result (a draw is 'closer' to a home win than an away win is).\n")
    print("| Model | RPS | Log loss | Brier | Favourite correct | Avg P(actual score) |")
    print("|---|---|---|---|---|---|")
    for name in MODELS:
        s = S[name]
        ex = f"{np.nanmean(s['exact']):.1%}" if not np.all(np.isnan(s["exact"])) else "–"
        fav = "–" if name == "Guess 1/3 each" else f"{s['correct'].mean():.1%}"
        print(f"| {name} | {s['rps'].mean():.4f} | {s['logloss'].mean():.3f} | {s['brier'].mean():.3f} | "
              f"{fav} | {ex} |")
    print()
    base = "Home/draw/away rates so far"
    print("### Differences in RPS (with 95% bootstrap interval)\n")
    print("Negative = the first model is better. If the interval includes 0, the difference "
          "could be chance.\n")
    for name in list(MODELS)[2:]:
        d, lo, hi = paired_ci(S[name]["rps"], S[base]["rps"])
        print(f"- {name} vs {base.lower()}: {d:+.4f} ({lo:+.4f} to {hi:+.4f})")
    d, lo, hi = paired_ci(S["Basic Poisson + home advantage"]["rps"], S["Basic Poisson model"]["rps"])
    print(f"- Adding home advantage to the basic Poisson model: {d:+.4f} ({lo:+.4f} to {hi:+.4f})")
    d, lo, hi = paired_ci(S["Fitted Dixon–Coles model"]["rps"], S["Basic Poisson + home advantage"]["rps"])
    print(f"- Dixon–Coles vs basic Poisson + home advantage: {d:+.4f} ({lo:+.4f} to {hi:+.4f})")
    print()

    print("## Draws\n")
    for name in list(MODELS)[2:]:
        p = np.array([r[2][name][0][1] for r in rows])
        print(f"- {name}: predicts {p.mean():.1%} draws on average; actual {counts[1] / n:.1%}.")
    print()
    print("## Calibration\n")
    calibration_figure(rows)
    print("![Calibration](figures/calibration.png)\n")

    print("## The 2014/15 final, predicted from that season's earlier results\n")
    final = next(r for r in rows if r[0]["season"] == "2014-15" and r[0]["stage"] == "Final")
    m = final[0]
    print(f"{m['home']} v {m['away']}: actual {m['hg']}-{m['ag']} (Barcelona won 3-1).\n")
    print("| Model | P(Juventus win) | P(draw) | P(Barcelona win) | P(1-3) |")
    print("|---|---|---|---|---|")
    for name in MODELS:
        p, M = final[2][name]
        exact = f"{M[1, 3]:.1%}" if M is not None else "–"
        print(f"| {name} | {p[0]:.1%} | {p[1]:.1%} | {p[2]:.1%} | {exact} |")
    print()


if __name__ == "__main__":
    main()
