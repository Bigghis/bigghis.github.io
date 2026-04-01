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
# Requires: openssl, xxd

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

KEY=$(printf '%s' "$STATICRYPT_PASSWORD" | openssl dgst -sha256 -binary | xxd -p -c 64)
IV=$(printf '%s' "${STATICRYPT_PASSWORD}_iv" | openssl dgst -md5 -binary | xxd -p -c 32)

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

  local tmp_front tmp_body tmp_enc
  tmp_front=$(mktemp)
  tmp_body=$(mktemp)
  tmp_enc=$(mktemp)
  trap "rm -f '$tmp_front' '$tmp_body' '$tmp_enc'" RETURN

  awk '/^---$/{n++} n==2{print; exit} {print}' "$file" > "$tmp_front"
  awk '/^---$/{n++; if(n==2){found=1; next}} found{print}' "$file" > "$tmp_body"

  if [ ! -s "$tmp_body" ]; then
    echo "  Skipping (empty body): $file"
    return
  fi

  openssl enc -aes-256-cbc -K "$KEY" -iv "$IV" -a < "$tmp_body" > "$tmp_enc"

  {
    cat "$tmp_front"
    printf '\n%s\n\n' "$ENCRYPTED_MARKER"
    cat "$tmp_enc"
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
