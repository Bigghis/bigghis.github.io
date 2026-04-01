#!/usr/bin/env bash
#
# Encrypts the body (below front matter) of posts with 'protected: true',
# so that the repository only stores ciphertext for those posts.
# Front matter is preserved in plaintext for Jekyll listings/tags.
#
# Usage:   STATICRYPT_PASSWORD=xxx bash tools/encrypt-md.sh
#          STATICRYPT_PASSWORD=xxx bash tools/encrypt-md.sh _posts/2026-01-04-AWS-CLOUD_COMPUTING.md
#
# Requires: STATICRYPT_PASSWORD env variable
# Requires: openssl

set -euo pipefail

ENCRYPTED_MARKER="<!-- ENCRYPTED -->"

if [ -z "${STATICRYPT_PASSWORD:-}" ] && [ -f .env ]; then
  # shellcheck disable=SC1091
  source .env
fi

if [ -z "${STATICRYPT_PASSWORD:-}" ]; then
  echo "ERROR: STATICRYPT_PASSWORD environment variable is not set"
  exit 1
fi

encrypt_file() {
  local file="$1"

  if grep -q "^${ENCRYPTED_MARKER}$" "$file"; then
    echo "  Skipping (already encrypted): $file"
    return
  fi

  if ! grep -q '^protected:\s*true' "$file"; then
    echo "  Skipping (not protected): $file"
    return
  fi

  local front_matter body encrypted
  front_matter=$(awk '/^---$/{n++} n==2{print; exit} {print}' "$file")
  body=$(awk '/^---$/{n++; if(n==2){found=1; next}} found{print}' "$file")

  if [ -z "$body" ]; then
    echo "  Skipping (empty body): $file"
    return
  fi

  encrypted=$(echo "$body" | openssl enc -aes-256-cbc -pbkdf2 -a -salt -pass "pass:${STATICRYPT_PASSWORD}")

  {
    echo "$front_matter"
    echo ""
    echo "$ENCRYPTED_MARKER"
    echo ""
    echo "$encrypted"
  } > "$file"

  echo "  Encrypted: $file"
}

if [ $# -gt 0 ]; then
  for f in "$@"; do
    encrypt_file "$f"
  done
else
  count=0
  for post_md in _posts/*.md; do
    if grep -q '^protected:\s*true' "$post_md"; then
      encrypt_file "$post_md"
      count=$((count + 1))
    fi
  done
  echo "Processed $count protected post(s)."
fi
