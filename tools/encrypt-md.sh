#!/usr/bin/env bash
#
# Encrypts the body (below front matter) of posts with 'protected: true',
# so that the repository only stores ciphertext for those posts.
# Front matter is preserved in plaintext for Jekyll listings/tags.
#
# Usage:   bash tools/encrypt-md.sh
#          bash tools/encrypt-md.sh _posts/some-post.md
#
# Reads STATICRYPT_PASSWORD from .env if present, or from environment.

set -euo pipefail

if [ -z "${STATICRYPT_PASSWORD:-}" ] && [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -gt 0 ]; then
  python3 "$SCRIPT_DIR/crypt_md.py" encrypt "$@"
else
  files=()
  for post_md in _posts/*.md; do
    if grep -q '^protected:\s*true' "$post_md"; then
      files+=("$post_md")
    fi
  done
  if [ ${#files[@]} -eq 0 ]; then
    echo "No protected posts found."
    exit 0
  fi
  python3 "$SCRIPT_DIR/crypt_md.py" encrypt "${files[@]}"
  echo "Processed ${#files[@]} protected post(s)."
fi
