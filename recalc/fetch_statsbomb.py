"""Download StatsBomb open data for the two finals into statsbomb/ (not committed).

Data: StatsBomb Open Data, https://github.com/statsbomb/open-data
Their terms ask that anything published with it credits StatsBomb as the data
source and uses their logo (https://statsbomb.com/media-pack/).
"""
import os
import urllib.request

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
MATCHES = {18236: "2010/11 final, Barcelona 3-1 Manchester United",
           18242: "2014/15 final, Juventus 1-3 Barcelona"}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "statsbomb")

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for match_id, label in MATCHES.items():
        for kind in ("events", "lineups"):
            path = os.path.join(OUT, f"{kind}_{match_id}.json")
            if not os.path.exists(path):
                urllib.request.urlretrieve(f"{BASE}/{kind}/{match_id}.json", path)
            print(f"{label}: {kind} -> {path}")
