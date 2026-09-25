#!/bin/sh
# Download one StatsBomb open-data competition-season.
#   ./scripts/download_statsbomb.sh COMPETITION_ID SEASON_ID OUT_DIR [--360]
# e.g. ./scripts/download_statsbomb.sh 11 90 data/laliga/90 --360   (La Liga 2020/21)
# Writes OUT_DIR/matches.json, events/, lineups/ and (with --360) three-sixty/.
set -e
[ $# -ge 3 ] || { echo "usage: $0 COMPETITION_ID SEASON_ID OUT_DIR [--360]"; exit 1; }
B=https://raw.githubusercontent.com/statsbomb/open-data/master/data
OUT=$3
KINDS="events lineups"
[ "$4" = "--360" ] && KINDS="$KINDS three-sixty"
for k in $KINDS; do mkdir -p "$OUT/$k"; done
curl -sSfL "$B/matches/$1/$2.json" -o "$OUT/matches.json"
python3 -c "import json; [print(m['match_id']) for m in json.load(open('$OUT/matches.json'))]" |
  xargs -P 8 -I{} sh -c "for k in $KINDS; do f=$OUT/\$k/{}.json; [ -s \$f ] || { curl -sSfL $B/\$k/{}.json -o \$f.part && mv \$f.part \$f; }; done"
# Re-fetch anything that doesn't parse (e.g. an interrupted transfer from an older run).
if ! python3 - "$OUT" $KINDS <<'PY'
import glob, json, os, sys
out, kinds = sys.argv[1], sys.argv[2:]
bad = []
for k in kinds:
    for f in glob.glob(f"{out}/{k}/*.json"):
        try:
            json.load(open(f))
        except ValueError:
            bad.append(f)
for f in bad:
    os.remove(f)
print(f"removed {len(bad)} unreadable files" if bad else "all files parse")
sys.exit(1 if bad else 0)
PY
then
  RETRY=$((${RETRY:-0} + 1))
  [ "$RETRY" -le 3 ] || { echo "giving up after 3 retries"; exit 1; }
  export RETRY
  exec "$0" "$@"
fi
echo "Downloaded $(ls "$OUT/events" | wc -l) matches ($KINDS) to $OUT"
