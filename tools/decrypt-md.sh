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

decrypt_file() {
  local file="$1"

  if ! grep -q "^${ENCRYPTED_MARKER}$" "$file"; then
    echo "  Skipping (not encrypted): $file"
    return
  fi

  local tmp_front tmp_blob tmp_dec
  tmp_front=$(mktemp)
  tmp_blob=$(mktemp)
  tmp_dec=$(mktemp)
  trap "rm -f '$tmp_front' '$tmp_blob' '$tmp_dec'" RETURN

  awk '/^---$/{n++} n==2{print; exit} {print}' "$file" > "$tmp_front"

  awk -v marker="$ENCRYPTED_MARKER" '
    BEGIN { found=0 }
    $0 == marker { found=1; next }
    found && /^[A-Za-z0-9+\/=]/ { print }
  ' "$file" > "$tmp_blob"

  if [ ! -s "$tmp_blob" ]; then
    echo "  ERROR: No encrypted content found in $file"
    return 1
  fi

  if ! openssl enc -aes-256-cbc -pbkdf2 -a -d -salt -pass "pass:${STATICRYPT_PASSWORD}" < "$tmp_blob" > "$tmp_dec"; then
    echo "  ERROR: Decryption failed for $file (wrong password?)"
    return 1
  fi

  {
    cat "$tmp_front"
    cat "$tmp_dec"
  } > "$file"

  echo "  Decrypted: $file"
}

if [ $# -gt 0 ]; then
  for f in "$@"; do
    decrypt_file "$f"
  done
else
  count=0
  for post_md in _posts/*.md; do
    if grep -q "^${ENCRYPTED_MARKER}$" "$post_md"; then
      decrypt_file "$post_md"
      count=$((count + 1))
    fi
  done
  echo "Processed $count encrypted post(s)."
fi
