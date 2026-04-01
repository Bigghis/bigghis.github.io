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

echo "Site dir: $SITE_DIR"
echo "Posts dir: $POSTS_DIR"
echo "Available post directories in _site:"
ls -1 "$POSTS_DIR" 2>/dev/null || echo "  (none)"
echo ""

protected_count=0

for post_md in _posts/*.md; do
  if grep -q '^protected:\s*true' "$post_md"; then
    slug=$(basename "$post_md" | sed 's/^[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}-//' | sed 's/\.md$//')

    echo "Protected post found: $post_md (slug: $slug)"

    html_file=""
    # Try exact slug (preserving original case from filename)
    if [ -f "${POSTS_DIR}/${slug}/index.html" ]; then
      html_file="${POSTS_DIR}/${slug}/index.html"
    fi

    # Try lowercase slug
    if [ -z "$html_file" ]; then
      slug_lower=$(echo "$slug" | tr '[:upper:]' '[:lower:]')
      if [ -f "${POSTS_DIR}/${slug_lower}/index.html" ]; then
        html_file="${POSTS_DIR}/${slug_lower}/index.html"
      fi
    fi

    # Try case-insensitive search as fallback
    if [ -z "$html_file" ]; then
      slug_lower=$(echo "$slug" | tr '[:upper:]' '[:lower:]')
      found=$(find "$POSTS_DIR" -maxdepth 1 -type d | while read -r dir; do
        dir_name=$(basename "$dir")
        dir_lower=$(echo "$dir_name" | tr '[:upper:]' '[:lower:]')
        if [ "$dir_lower" = "$slug_lower" ] && [ -f "$dir/index.html" ]; then
          echo "$dir/index.html"
          break
        fi
      done)
      if [ -n "$found" ]; then
        html_file="$found"
      fi
    fi

    if [ -n "$html_file" ]; then
      echo "  Encrypting: $html_file"
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
      echo "  ERROR: No matching HTML found in $POSTS_DIR for slug '$slug'"
      echo "  Tried: ${POSTS_DIR}/${slug}/index.html"
      echo "  Tried: ${POSTS_DIR}/$(echo "$slug" | tr '[:upper:]' '[:lower:]')/index.html"
      echo "  Available directories:"
      ls -1 "$POSTS_DIR"
    fi
  fi
done

echo ""
echo "Done. Encrypted $protected_count post(s)."

if [ "$protected_count" -eq 0 ]; then
  echo "WARNING: No posts were encrypted. Check that:"
  echo "  1. Posts have 'protected: true' in front matter"
  echo "  2. The STATICRYPT_PASSWORD secret is set in GitHub"
  echo "  3. The slug from the filename matches a directory in _site/posts/"
fi
