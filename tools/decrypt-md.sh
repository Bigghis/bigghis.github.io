#!/usr/bin/env bash
#
# Decrypts the body of encrypted posts (those with the ENCRYPTED marker)
# back to plaintext Markdown. Used in CI before Jekyll build,
# or locally when you need to edit a protected post.
#
# Usage:   bash tools/decrypt-md.sh
#            Scans _posts/ for files with the ENCRYPTED marker and decrypts them.
#
#          bash tools/decrypt-md.sh <pattern>
#            Finds files in _posts/ whose name contains <pattern> and decrypts them.
#            Example: bash tools/decrypt-md.sh AI-BASES
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
  pattern="$1"
  matches=()
  for post_md in _posts/*.md; do
    filename="$(basename "$post_md")"
    if [[ "$filename" == *"$pattern"* ]]; then
      matches+=("$post_md")
    fi
  done
  if [ ${#matches[@]} -eq 0 ]; then
    echo "No file found in _posts/ matching pattern: $pattern"
    exit 1
  fi
  if [ ${#matches[@]} -gt 1 ]; then
    echo "Multiple files match pattern '$pattern':"
    printf '  %s\n' "${matches[@]}"
    echo "Please provide a more specific pattern."
    exit 1
  fi
  echo "Decrypting: ${matches[0]}"
  python3 "$SCRIPT_DIR/crypt_md.py" decrypt "${matches[0]}"
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
