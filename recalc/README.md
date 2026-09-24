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
| `STATSBOMB.md`, `figures/sb_*` | Both finals with StatsBomb event data: data check, real positions, halves, xG, xGChain | `fetch_statsbomb.py` then `statsbomb_analysis.py` |

StatsBomb data is downloaded by `fetch_statsbomb.py` and not committed. If you publish anything
based on it, credit StatsBomb as the data source and use their logo
([media pack](https://statsbomb.com/media-pack/)).

Requirements: `pip install networkx scipy numpy matplotlib`. Run each script from this folder.
