#!/usr/bin/env bash
#
# Decrypts the body of encrypted posts (those with the ENCRYPTED marker)
# back to plaintext Markdown. Used in CI before Jekyll build,
# or locally when you need to edit a protected post.
#
# Usage:   bash tools/decrypt-md.sh
#          bash tools/decrypt-md.sh _posts/some-post.md
#
# Reads STATICRYPT_PASSWORD from .env if present, or from environment.

set -euo pipefail

ENCRYPTED_MARKER="<!-- ENCRYPTED -->"

if [ -z "${STATICRYPT_PASSWORD:-}" ] && [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -gt 0 ]; then
  python3 "$SCRIPT_DIR/crypt_md.py" decrypt "$@"
else
  files=()
  for post_md in _posts/*.md; do
    if grep -q "^${ENCRYPTED_MARKER}$" "$post_md"; then
      files+=("$post_md")
    fi
  done
  if [ ${#files[@]} -eq 0 ]; then
    echo "No encrypted posts found."
    exit 0
  fi
  python3 "$SCRIPT_DIR/crypt_md.py" decrypt "${files[@]}"
  echo "Processed ${#files[@]} encrypted post(s)."
fi
