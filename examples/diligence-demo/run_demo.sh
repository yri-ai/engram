#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WORK=${1:-$(mktemp -d)}
REPO="$WORK/ledger"
EVIDENCE="$ROOT/examples/diligence-demo/evidence.json"
uv run engram forecast-question-create \
  --repo "$REPO" \
  --question-id q-diligence-demo \
  --title "Will the public diligence deal advance, reprice, or fail?" \
  --forecast-as-of 2026-01-15T00:00:00+00:00 \
  --horizon 60d \
  --resolution-criteria "Resolve using public milestone records." \
  --resolved-by 2026-03-15T00:00:00+00:00 \
  --branch advance_or_close:"Advance or close" \
  --branch reprice_or_restructure:"Reprice or restructure" \
  --branch terminated_or_failed:"Terminate or fail" \
  --status active
uv run engram forecast-dossier-compile --repo "$REPO" --question-id q-diligence-demo --evidence-json "$EVIDENCE"
uv run engram forecast-run-create --repo "$REPO" --question-id q-diligence-demo --dossier "$REPO/dossiers/dossier-q-diligence-demo.json" --run-id run-diligence-demo
uv run engram forecast-resolve-create --repo "$REPO" --question-id q-diligence-demo --resolved-branch reprice_or_restructure --resolved-at 2026-12-01T00:00:00+00:00 --evidence-id resolution-demo
uv run engram forecast-score-report --repo "$REPO" --low-sample-threshold 1 --output "$WORK/report.json"
uv run engram forecast-audit-report --repo "$REPO" --spot-check 1 --output "$WORK/audit.json"
echo "$WORK"
