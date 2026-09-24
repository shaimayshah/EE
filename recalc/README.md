# Recalculations and extensions

Everything here is generated from the passing tables in the essay (`data.py`) and,
for the model check, Champions League results from
[openfootball](https://github.com/openfootball/champions-league) (`cl_results/`, CC0).

| File | What it contains | Script |
|---|---|---|
| `RESULTS.md` | Corrected centralities and Poisson calculations | `recalculate.py` |
| `figures/fig6-*`, `figures/all_networks.*` | Redrawn passing networks | `plot_networks.py` |
| `NETWORK_EXTRA.md`, `figures/red_card.*` | Team-level measures, red-card test, passing units, random-network comparison, bootstrap | `network_extra.py` |
| `MODEL_CHECK.md`, `figures/calibration.*` | The Poisson model tested on 371 knockout matches | `model_check.py` |

Requirements: `pip install networkx scipy numpy matplotlib`. Run each script from this folder.
