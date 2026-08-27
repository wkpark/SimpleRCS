"""Pure format primitives for SimpleRCS blocks.

Stateless helpers for encoding/decoding binary payloads, escaping ``@`` in
``@...@`` values, and computing the v2 hash-chain block hash. Kept free of any
instance state so they can be unit-tested in isolation; ``SimpleRCS`` passes its
configured ``hash_algo``/``encoding`` in as arguments.
"""

import base64
import hashlib
import re

#: Header of a length-prefixed binary value: ``<length>;<encoding>,``.
_BINARY_HEADER = re.compile(rb"\A([0-9]+);([A-Za-z0-9]+),")
_BINARY_HEADER_STR = re.compile(r"\A([0-9]+);([A-Za-z0-9]+),")


def binary_tag(value: bytes | str) -> str | None:
    """Encoding tag of a length-prefixed binary value, or None if there is none.

    Anchored at the start on purpose. A text delta's payload is arbitrary user
    content and can contain ";base64," anywhere in it, which a substring search
    would mistake for a binary payload.
    """
    if isinstance(value, str):
        match = _BINARY_HEADER_STR.match(value)
        return match.group(2) if match else None
    match = _BINARY_HEADER.match(value)
    return match.group(2).decode("ascii") if match else None


def escape_bytes(data: bytes) -> bytes:
    """Escapes '@' to '@@' for storage within @...@ blocks (bytes form)."""
    return data.replace(b"@", b"@@")


def unescape_bytes(data: bytes) -> bytes:
    """Inverse of :func:`escape_bytes`. Every '@' in the stored form is doubled,
    so all runs are even and a left-to-right halving is exact."""
    return data.replace(b"@@", b"@")


def encode_binary(data: bytes, encoding: str = "base64") -> bytes:
    """Encodes binary data to bytes format: <length>;<encoding>,<escaped payload>

    Escaping is a property of the *container*, not of the encoding: every value
    in a block doubles its '@', which is what makes '@' parity a decidable rule
    for finding field delimiters. So the payload is escaped whatever the
    encoding is -- a no-op for base64, whose alphabet has no '@'.

    The tag says only how the bytes were encoded, and the length is always the
    count of what is actually on disk (post-escape), so a forward parser can
    skip the payload in one seek regardless of encoding.
    """
    if encoding == "raw":
        # RCS-style: keep the bytes, let the escaping carry them. ~0.4% overhead
        # against base64's 33%, and bytes.replace decodes faster than b64decode.
        payload = data
    elif encoding == "base85":
        payload = base64.b85encode(data)
    else:
        encoding = "base64"
        payload = base64.b64encode(data)
    escaped = escape_bytes(payload)
    return f"{len(escaped)};{encoding},".encode("ascii") + escaped


def decode_binary(text: bytes) -> bytes:
    """Decodes binary data from bytes format.

    The payload arrives escaped: every reader that produces one reads it
    length-based and hands the stored bytes over verbatim, so the unescaping
    belongs here. Do not unescape before calling -- base85's alphabet contains
    '@' (and raw can contain anything), so a second pass would collapse a
    doubled '@' twice.
    """
    # Read the tag from the header rather than searching the whole value: a raw
    # payload can contain ";base64," itself.
    match = _BINARY_HEADER.match(text)
    if not match:
        raise ValueError("Invalid binary format")
    tag = match.group(2).decode("ascii")
    payload = unescape_bytes(text[match.end() :])
    if tag == "base64":
        return base64.b64decode(payload)
    if tag == "base85":
        return base64.b85decode(payload)
    if tag == "raw":
        return payload
    raise ValueError(f"Unknown binary encoding {tag!r}")


def escape(text: str) -> str:
    """Escapes '@' to '@@' for storage within @...@ blocks."""
    return text.replace("@", "@@")


def calculate_block_hash(data: dict, prev_hash: str | None = None, *, hash_algo: str, encoding: str) -> str:
    """
    Calculates hash of the block content using ``hash_algo``.
    IMPORTANT: The hash is calculated based on the LOGICAL content (Full Text),
    not the stored delta. This ensures the hash remains valid even when
    HEAD becomes a historical delta block.

    Payload: ver|date|author|log|text|prev_hash
    """
    # Ensure we use empty string for None to keep hash stable
    ver = str(data.get("ver", ""))
    date = str(data.get("date", ""))
    author = str(data.get("author", ""))
    log = str(data.get("log", ""))

    content = data.get("text")  # Can be str or bytes
    if isinstance(content, bytes):
        text_bytes = content
    else:
        text_str = str(content) if content is not None else ""
        # Enforce EOL policy for text
        if text_str and not text_str.endswith("\n"):
            text_str += "\n"
        text_bytes = text_str.encode(encoding)

    p_hash = prev_hash if prev_hash else ""

    # Construct payload components
    # ver|date|author|log|
    meta = f"{ver}|{date}|{author}|{log}|".encode(encoding)
    # |prev_hash
    tail = f"|{p_hash}".encode(encoding)

    hasher = hashlib.new(hash_algo)
    hasher.update(meta)
    hasher.update(text_bytes)
    hasher.update(tail)
    return hasher.hexdigest()
