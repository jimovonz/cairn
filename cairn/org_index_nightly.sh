#!/usr/bin/env bash
# Nightly org-wide index refresh + at-risk reports. Sibling to the
# `# cairn-maintenance` cron jobs. OPT-IN: does nothing unless ORG_INDEX_ENABLED
# is set in cairn config. Refreshes:
#   1. locatability index over the configured org(s) (needs gh; skipped if unauth)
#   2. cross-repo interface registry (reads local graphs kept fresh by graph_fleet)
#   3. cairn location-claim verification (writes a drift report)
set -uo pipefail
CAIRN=/mnt/ssd/Projects/cairn
PY="$CAIRN/.venv/bin/python3"
CC="$CAIRN/cairn"
REPORTS="$CC/reports"; mkdir -p "$REPORTS" "$CAIRN/logs"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# Read settings straight from cairn config (honours cairn/.env + real env).
read -r ENABLED ORGS_CSV GH_HOST GH_ACCOUNT PWM < <(
  "$PY" - << 'PY'
from cairn import config
print(
    "1" if config.ORG_INDEX_ENABLED else "0",
    ",".join(config.ORG_INDEX_ORGS) or "-",
    config.ORG_INDEX_GH_HOST or "-",
    config.ORG_INDEX_GH_ACCOUNT or "-",
    config.ORG_INDEX_PUSHED_WITHIN_MONTHS,
)
PY
)

if [ "$ENABLED" != "1" ]; then
    echo "$(ts) org-index disabled (set ORG_INDEX_ENABLED=1 + ORG_INDEX_ORGS to enable) — nothing to do"
    exit 0
fi
if [ "$ORGS_CSV" = "-" ]; then
    echo "$(ts) org-index enabled but ORG_INDEX_ORGS is empty — nothing to do"
    exit 0
fi

echo "=== org-index nightly $(ts) orgs=$ORGS_CSV host=$GH_HOST acct=$GH_ACCOUNT ==="
[ "$GH_HOST" != "-" ] && export GH_HOST

# 1. Locatability (remote, needs gh). gh keyring auth may be unavailable in a
#    headless cron env — detect and skip rather than fail.
if gh auth status >/dev/null 2>&1; then
    # Select the configured gh account (multi-account). Best-effort: a failed
    # switch (e.g. account not present) just leaves the active account in place.
    [ "$GH_ACCOUNT" != "-" ] && gh auth switch --user "$GH_ACCOUNT" >/dev/null 2>&1 || true
    # org_index reads ORG_INDEX_ORGS / host / pushed-within from config itself.
    if "$PY" "$CC/org_index.py" build; then
        "$PY" "$CC/org_index.py" stranded --stale-days 90 > "$REPORTS/stranded.txt" \
            && echo "locatability: ok -> $REPORTS/stranded.txt"
    fi
else
    echo "WARN: gh not authenticated in this env — skipping org_index build (locatability left stale)"
fi
# 2. Interface/consumer registry (local graphs, no network)
"$PY" "$CC/interface_registry.py" build && echo "interface registry: ok"
# 3. Verify cairn location claims against the locatability index
"$PY" "$CC/cairn_verify.py" > "$REPORTS/cairn-drift.txt" \
    && echo "verify: ok -> $REPORTS/cairn-drift.txt"
echo "=== done $(ts) ==="
