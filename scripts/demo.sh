#!/usr/bin/env bash
set -euo pipefail

API_URL=${API_URL:-http://localhost:8000}
CONVERSATION_ID=${CONVERSATION_ID:-coaching-demo}
GROUP_ID=${GROUP_ID:-client-kendra}
DATA_FILE=${1:-examples/coaching-demo.json}

if [[ ! -f ".env" ]]; then
  echo "Missing .env file. Copy .env.example and configure your API keys before running the demo."
  exit 1
fi

if [[ ! -f "$DATA_FILE" ]]; then
  echo "Demo data file '$DATA_FILE' not found. Provide the path as the first argument."
  exit 1
fi

echo "Checking Engram API at ${API_URL}/health ..."
if ! curl -fsS "${API_URL}/health" >/dev/null; then
  cat <<MSG
Engram API is not reachable at ${API_URL}.
Start the stack first (e.g. 'docker compose up' or 'uv run engram serve') and run this script again.
MSG
  exit 1
fi

set -x
uv run engram ingest "$DATA_FILE" \
  --conversation-id "$CONVERSATION_ID" \
  --tenant-id default \
  --group-id "$GROUP_ID" \
  --api-url "$API_URL"

uv run engram query "Kendra" \
  --tenant-id default \
  --conversation-id "$CONVERSATION_ID" \
  --api-url "$API_URL"
set +x

echo "Demo complete!"
