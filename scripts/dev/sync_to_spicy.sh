#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-spicy}"
REMOTE_USER="${REMOTE_USER:-$USER}"
REMOTE_PATH="${REMOTE_PATH:-~/dev/projects/engram}"
COPY_ENV=0
DELETE_MODE=0
INCLUDE_GIT=0

usage() {
  cat <<'EOF'
Sync local engram workspace to remote host (default: spicy) over Tailscale.

Usage:
  scripts/dev/sync_to_spicy.sh [options]

Options:
  --host <host>           Remote host (default: spicy)
  --user <user>           Remote SSH user (default: local $USER)
  --remote-path <path>    Remote repo path (default: ~/dev/projects/engram)
  --copy-env              Also copy local .env to remote .env
  --delete                Delete remote files missing locally (careful)
  --include-git           Include .git/ in rsync (for full machine migration)
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="$2"
      shift 2
      ;;
    --user)
      REMOTE_USER="$2"
      shift 2
      ;;
    --remote-path)
      REMOTE_PATH="$2"
      shift 2
      ;;
    --copy-env)
      COPY_ENV=1
      shift
      ;;
    --delete)
      DELETE_MODE=1
      shift
      ;;
    --include-git)
      INCLUDE_GIT=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

SSH_TARGET="${REMOTE_USER}@${HOST}"
DELETE_FLAG=()
if [[ "$DELETE_MODE" -eq 1 ]]; then
  DELETE_FLAG=(--delete)
fi

GIT_EXCLUDE=()
if [[ "$INCLUDE_GIT" -ne 1 ]]; then
  GIT_EXCLUDE=(--exclude ".git/")
fi

echo "==> Ensuring remote path exists: ${SSH_TARGET}:${REMOTE_PATH}"
ssh "$SSH_TARGET" "mkdir -p '${REMOTE_PATH}'"

echo "==> Syncing workspace to ${SSH_TARGET}:${REMOTE_PATH}"
rsync -azP "${DELETE_FLAG[@]}" \
  "${GIT_EXCLUDE[@]}" \
  --exclude ".venv/" \
  --exclude ".pytest_cache/" \
  --exclude ".mypy_cache/" \
  --exclude ".ruff_cache/" \
  --exclude ".uv-cache/" \
  --exclude "__pycache__/" \
  ./ "${SSH_TARGET}:${REMOTE_PATH}/"

if [[ "$COPY_ENV" -eq 1 ]]; then
  if [[ -f .env ]]; then
    echo "==> Copying .env to remote"
    scp .env "${SSH_TARGET}:${REMOTE_PATH}/.env"
  else
    echo "!! --copy-env requested but local .env not found" >&2
    exit 1
  fi
fi

echo "==> Sync complete"
