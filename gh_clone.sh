#!/bin/bash
# bash /usr/yue/RL/LLM/gh_clone.sh your-org/private-repo
# /usr/yue/RL/LLM/gh_clone.sh your-org/private-repo my-local-dir

set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: $0 <owner/repo|https://github.com/owner/repo.git> [target-dir]"
  exit 1
fi

REPO_INPUT="$1"
TARGET_DIR="${2:-}"
CRED_FILE="${HOME}/.git-credentials"

if [[ "$REPO_INPUT" == https://github.com/* ]]; then
  REPO_URL="$REPO_INPUT"
else
  REPO_URL="https://github.com/${REPO_INPUT%.git}.git"
fi

cleanup() {
  rm -f "$CRED_FILE"
  echo "Credentials cleaned up."
}

trap cleanup EXIT

touch "$CRED_FILE"
chmod 600 "$CRED_FILE"

export GIT_ASKPASS=""
export GIT_TERMINAL_PROMPT=1

if [ -n "$TARGET_DIR" ]; then
  git -c credential.helper="store --file $CRED_FILE" clone "$REPO_URL" "$TARGET_DIR"
else
  git -c credential.helper="store --file $CRED_FILE" clone "$REPO_URL"
fi
