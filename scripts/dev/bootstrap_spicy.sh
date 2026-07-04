#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-spicy}"
REMOTE_USER="${REMOTE_USER:-$USER}"
REMOTE_BASE="${REMOTE_BASE:-~/dev/projects}"
REPO_URL="${REPO_URL:-git@github.com:yri-ai/engram.git}"
BRANCH="${BRANCH:-feat/research-data-pipeline}"

usage() {
  cat <<'EOF'
Bootstrap or update engram on remote host (default: spicy).

Usage:
  scripts/dev/bootstrap_spicy.sh [options]

Options:
  --host <host>         Remote host (default: spicy)
  --user <user>         Remote SSH user (default: local $USER)
  --remote-base <path>  Remote base dir (default: ~/dev/projects)
  --repo-url <url>      Git repo URL (default: git@github.com:yri-ai/engram.git)
  --branch <branch>     Branch to checkout/pull (default: feat/research-data-pipeline)
  -h, --help            Show this help
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
    --remote-base)
      REMOTE_BASE="$2"
      shift 2
      ;;
    --repo-url)
      REPO_URL="$2"
      shift 2
      ;;
    --branch)
      BRANCH="$2"
      shift 2
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

ssh "$SSH_TARGET" bash -s -- "$REMOTE_BASE" "$REPO_URL" "$BRANCH" <<'EOS'
set -euo pipefail
REMOTE_BASE="$1"
REPO_URL="$2"
BRANCH="$3"
REPO_DIR="${REMOTE_BASE}/engram"

mkdir -p "$REMOTE_BASE"

if [[ ! -d "$REPO_DIR/.git" ]]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"
git fetch origin
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"

if command -v uv >/dev/null 2>&1; then
  uv sync --extra dev
else
  echo "uv is not installed on remote host. Install uv, then run: uv sync --extra dev"
fi
EOS

echo "==> Bootstrap/update complete on ${SSH_TARGET}"
