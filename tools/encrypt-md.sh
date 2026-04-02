#!/usr/bin/env bash
#
# Encrypts the body (below front matter) of posts with 'protected: true',
# so that the repository only stores ciphertext for those posts.
# Front matter is preserved in plaintext for Jekyll listings/tags.
#
# Usage:   bash tools/encrypt-md.sh
#            Scans _posts/ for files with 'protected: true' and encrypts them.
#
#          bash tools/encrypt-md.sh <pattern>
#            Finds files in _posts/ whose name contains <pattern> and encrypts them.
#            Example: bash tools/encrypt-md.sh AI-BASES
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
  echo "Encrypting: ${matches[0]}"
  python3 "$SCRIPT_DIR/crypt_md.py" encrypt "${matches[0]}"
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
