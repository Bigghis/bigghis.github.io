#!/usr/bin/env python3
"""
Encrypt/decrypt the body of Jekyll posts using PBKDF2 + XOR stream cipher.
Uses only Python stdlib — fully portable across any Python 3.6+ environment.

Usage:
    python3 tools/crypt_md.py encrypt <file> [<file2> ...]
    python3 tools/crypt_md.py decrypt <file> [<file2> ...]

Reads STATICRYPT_PASSWORD from environment.
"""

import sys
import os
import hashlib
import base64

ENCRYPTED_MARKER = "<!-- ENCRYPTED -->"
PBKDF2_ITERATIONS = 100_000


def _keystream(key: bytes, length: int) -> bytes:
    """Generate a keystream using HMAC-SHA256 in counter mode."""
    out = b""
    i = 0
    while len(out) < length:
        out += hashlib.sha256(key + i.to_bytes(4, "big")).digest()
        i += 1
    return out[:length]


def encrypt_bytes(data: bytes, password: str) -> str:
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS, dklen=32)
    keystream = _keystream(key, len(data))
    ciphertext = bytes(a ^ b for a, b in zip(data, keystream))
    return base64.b64encode(salt + ciphertext).decode()


def decrypt_bytes(encoded: str, password: str) -> bytes:
    raw = base64.b64decode(encoded)
    salt = raw[:16]
    ciphertext = raw[16:]
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS, dklen=32)
    keystream = _keystream(key, len(ciphertext))
    return bytes(a ^ b for a, b in zip(ciphertext, keystream))


def split_front_matter(content: str):
    """Split a Jekyll post into (front_matter_with_delimiters, body)."""
    if not content.startswith("---"):
        return "", content
    end = content.index("---", 3)
    end = content.index("\n", end) + 1
    return content[:end], content[end:]


def encrypt_file(filepath: str, password: str) -> bool:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    if ENCRYPTED_MARKER in content:
        print(f"  Skipping (already encrypted): {filepath}")
        return True

    if "protected: true" not in content:
        print(f"  Skipping (not protected): {filepath}")
        return True

    front_matter, body = split_front_matter(content)

    if not body.strip():
        print(f"  Skipping (empty body): {filepath}")
        return True

    encoded = encrypt_bytes(body.encode("utf-8"), password)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(front_matter)
        f.write(f"\n{ENCRYPTED_MARKER}\n\n")
        for i in range(0, len(encoded), 76):
            f.write(encoded[i : i + 76] + "\n")

    print(f"  Encrypted: {filepath}")
    return True


def decrypt_file(filepath: str, password: str) -> bool:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    if ENCRYPTED_MARKER not in content:
        print(f"  Skipping (not encrypted): {filepath}")
        return True

    front_matter, after_front = split_front_matter(content)

    marker_pos = after_front.index(ENCRYPTED_MARKER)
    encoded_part = after_front[marker_pos + len(ENCRYPTED_MARKER) :]
    encoded = "".join(encoded_part.split())

    if not encoded:
        print(f"  ERROR: No encrypted content found in {filepath}")
        return False

    try:
        body = decrypt_bytes(encoded, password).decode("utf-8")
    except Exception as e:
        print(f"  ERROR: Decryption failed for {filepath}: {e}")
        return False

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(front_matter)
        f.write(body)

    print(f"  Decrypted: {filepath}")
    return True


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <encrypt|decrypt> <file> [<file2> ...]")
        sys.exit(1)

    action = sys.argv[1]
    files = sys.argv[2:]
    password = os.environ.get("STATICRYPT_PASSWORD", "")

    if not password:
        print("ERROR: STATICRYPT_PASSWORD environment variable is not set")
        sys.exit(1)

    if action not in ("encrypt", "decrypt"):
        print(f"ERROR: Unknown action '{action}'. Use 'encrypt' or 'decrypt'.")
        sys.exit(1)

    func = encrypt_file if action == "encrypt" else decrypt_file
    failed = False
    for filepath in files:
        if not func(filepath, password):
            failed = True

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
