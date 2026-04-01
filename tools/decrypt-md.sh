#!/usr/bin/env bash
#
# Decrypts the body of encrypted posts (those with the ENCRYPTED marker)
# back to plaintext Markdown. Used in CI before Jekyll build,
# or locally when you need to edit a protected post.
#
# Usage:   STATICRYPT_PASSWORD=xxx bash tools/decrypt-md.sh
#          STATICRYPT_PASSWORD=xxx bash tools/decrypt-md.sh _posts/2026-01-04-AWS-CLOUD_COMPUTING.md
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

decrypt_file() {
  local file="$1"

  if ! grep -q "^${ENCRYPTED_MARKER}$" "$file"; then
    echo "  Skipping (not encrypted): $file"
    return
  fi

  local front_matter encrypted_blob decrypted
  front_matter=$(awk '/^---$/{n++} n==2{print; exit} {print}' "$file")

  encrypted_blob=$(awk -v marker="$ENCRYPTED_MARKER" '
    BEGIN { found=0 }
    $0 == marker { found=1; next }
    found && /^[A-Za-z0-9+\/=]/ { print }
  ' "$file")

  if [ -z "$encrypted_blob" ]; then
    echo "  ERROR: No encrypted content found in $file"
    return 1
  fi

  decrypted=$(echo "$encrypted_blob" | openssl enc -aes-256-cbc -pbkdf2 -a -d -salt -pass "pass:${STATICRYPT_PASSWORD}") || {
    echo "  ERROR: Decryption failed for $file (wrong password?)"
    return 1
  }

  {
    echo "$front_matter"
    echo "$decrypted"
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
