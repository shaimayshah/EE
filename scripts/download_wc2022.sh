#!/bin/sh
# Download StatsBomb open data for the 2022 FIFA World Cup (competition 43, season 106)
# into data/wc2022/: matches.json, events/<match_id>.json, lineups/<match_id>.json
# Pass --360 to also fetch three-sixty/<match_id>.json freeze frames (~440 MB).
set -e
B=https://raw.githubusercontent.com/statsbomb/open-data/master/data
OUT=data/wc2022
KINDS="events lineups"
[ "$1" = "--360" ] && KINDS="$KINDS three-sixty"
for k in $KINDS; do mkdir -p "$OUT/$k"; done
curl -sSfL "$B/matches/43/106.json" -o "$OUT/matches.json"
python3 -c "import json; [print(m['match_id']) for m in json.load(open('$OUT/matches.json'))]" |
  xargs -P 8 -I{} sh -c "for k in $KINDS; do curl -sSfL $B/\$k/{}.json -o $OUT/\$k/{}.json; done"
echo "Downloaded $(ls "$OUT/events" | wc -l) matches ($KINDS) to $OUT"
