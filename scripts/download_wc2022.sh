#!/bin/sh
# Download StatsBomb open data for the 2022 FIFA World Cup (competition 43, season 106)
# into data/wc2022/: matches.json, events/<match_id>.json, lineups/<match_id>.json
set -e
B=https://raw.githubusercontent.com/statsbomb/open-data/master/data
OUT=data/wc2022
mkdir -p "$OUT/events" "$OUT/lineups"
curl -sSfL "$B/matches/43/106.json" -o "$OUT/matches.json"
python3 -c "import json; [print(m['match_id']) for m in json.load(open('$OUT/matches.json'))]" |
  xargs -P 8 -I{} sh -c "curl -sSfL $B/events/{}.json -o $OUT/events/{}.json && curl -sSfL $B/lineups/{}.json -o $OUT/lineups/{}.json"
echo "Downloaded $(ls "$OUT/events" | wc -l) matches to $OUT"
