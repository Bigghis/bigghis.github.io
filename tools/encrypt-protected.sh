#!/usr/bin/env bash
#
# Encrypts Jekyll posts that have 'protected: true' in their front matter
# using StatiCrypt. Run this AFTER jekyll build, BEFORE uploading the artifact.
#
# Requires: STATICRYPT_PASSWORD env variable to be set
# Requires: npx staticrypt to be available

set -euo pipefail

SITE_DIR="${1:-./_site}"
POSTS_DIR="${SITE_DIR}/posts"
SALT="${STATICRYPT_SALT:-}"

if [ -z "${STATICRYPT_PASSWORD:-}" ]; then
  echo "ERROR: STATICRYPT_PASSWORD environment variable is not set"
  exit 1
fi

if [ ! -d "$POSTS_DIR" ]; then
  echo "ERROR: Posts directory not found: $POSTS_DIR"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/password_template.html"

SALT_FLAG=""
if [ -n "$SALT" ]; then
  SALT_FLAG="--salt $SALT"
fi

TEMPLATE_FLAG=""
if [ -f "$TEMPLATE" ]; then
  TEMPLATE_FLAG="-t $TEMPLATE"
fi

protected_count=0

for post_md in _posts/*.md; do
  if grep -q '^protected:\s*true' "$post_md"; then
    slug=$(basename "$post_md" | sed 's/^[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}-//' | sed 's/\.md$//')
    slug_lower=$(echo "$slug" | tr '[:upper:]' '[:lower:]')
    html_file="${POSTS_DIR}/${slug_lower}/index.html"

    if [ -f "$html_file" ]; then
      echo "Encrypting: $html_file"
      npx staticrypt "$html_file" \
        -p "$STATICRYPT_PASSWORD" \
        -d "$(dirname "$html_file")" \
        --remember 30 \
        --template-title "Protected Content" \
        --template-instructions "This content is password-protected. Enter the password to view it." \
        --template-button "UNLOCK" \
        --template-placeholder "Password" \
        --template-color-primary "#2a408e" \
        --template-color-secondary "#252525" \
        --short \
        --config false \
        $SALT_FLAG \
        $TEMPLATE_FLAG
      protected_count=$((protected_count + 1))
    else
      echo "WARNING: Built HTML not found for protected post: $html_file (from $post_md)"
    fi
  fi
done

echo "Done. Encrypted $protected_count post(s)."
