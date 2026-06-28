#!/usr/bin/env bash
# Nightly org-wide index refresh + at-risk reports. Sibling to the
# `# cairn-maintenance` cron jobs. Refreshes:
#   1. locatability index (needs gh; skipped gracefully if unauth in cron)
#   2. cross-repo interface registry (reads local graphs kept fresh by graph_fleet)
#   3. cairn location-claim verification (writes a drift report)
set -uo pipefail
ORG="${ORG_INDEX_ORG:-robotics-plus}"
CAIRN=/mnt/ssd/Projects/cairn
PY="$CAIRN/.venv/bin/python3"
CC="$CAIRN/cairn"
REPORTS="$CC/reports"; mkdir -p "$REPORTS" "$CAIRN/logs"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

echo "=== org-index nightly $(ts) org=$ORG ==="
# 1. Locatability (remote, needs gh). gh uses keyring auth which may be
#    unavailable in a headless cron env — detect and skip rather than fail.
if gh auth status >/dev/null 2>&1; then
    "$PY" "$CC/org_index.py" build --org "$ORG" --pushed-within-months 12 \
        && "$PY" "$CC/org_index.py" stranded --stale-days 90 > "$REPORTS/stranded.txt" \
        && echo "locatability: ok -> $REPORTS/stranded.txt"
else
    echo "WARN: gh not authenticated in this env — skipping org_index build (locatability index left stale)"
fi
# 2. Interface/consumer registry (local graphs, no network)
"$PY" "$CC/interface_registry.py" build && echo "interface registry: ok"
# 3. Verify cairn location claims against the locatability index
"$PY" "$CC/cairn_verify.py" > "$REPORTS/cairn-drift.txt" \
    && echo "verify: ok -> $REPORTS/cairn-drift.txt"
echo "=== done $(ts) ==="
