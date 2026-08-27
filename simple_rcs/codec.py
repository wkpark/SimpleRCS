"""Pure format primitives for SimpleRCS blocks.

Stateless helpers for encoding/decoding binary payloads, escaping ``@`` in
``@...@`` values, and computing the v2 hash-chain block hash. Kept free of any
instance state so they can be unit-tested in isolation; ``SimpleRCS`` passes its
configured ``hash_algo``/``encoding`` in as arguments.
"""

import base64
import hashlib
import re

#: Encoding tag for RCS-style storage: raw bytes with '@' doubled.
#:
#: Every other field in a block escapes '@', which makes '@' parity a decidable
#: rule for locating field delimiters. ``base64`` satisfies it for free (its
#: alphabet has no '@'), but ``base85`` and ``raw`` do not, and their payloads
#: are stored unescaped -- the one place the rule breaks. ``esc`` is what GNU
#: RCS does: keep the bytes, double the '@'. Costs ~0.4% instead of base64's
#: 33%, decodes faster (``bytes.replace`` is C-level), and stays parity-clean.
ESCAPED = "esc"

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
    """Encodes binary data to bytes format: <length>;<encoding>,<encoded>

    The length is always the byte count of what follows the header, so a forward
    parser can skip the payload in one seek regardless of encoding.
    """
    if encoding == ESCAPED:
        # RCS-style. The length counts the *escaped* bytes, so the seek-skip
        # stays exact while the doubling keeps '@' parity usable.
        escaped = escape_bytes(data)
        return f"{len(escaped)};{ESCAPED},".encode("ascii") + escaped
    elif encoding == "base85":
        encoded = base64.b85encode(data)
        return f"{len(encoded)};base85,".encode("ascii") + encoded
    elif encoding == "raw":
        # Raw binary: Length-Based Parsing allows unescaped storage.
        return f"{len(data)};raw,".encode("ascii") + data
    else:
        # Default to base64
        encoded = base64.b64encode(data)
        return f"{len(encoded)};base64,".encode("ascii") + encoded


def decode_binary(text: bytes) -> bytes:
    """Decodes binary data from bytes format."""
    # text is bytes here, e.g. b"1024;base64,..."
    # Read the tag from the header rather than searching the whole value: a raw
    # or escaped payload can contain ";base64," itself.
    match = _BINARY_HEADER.match(text)
    if match:
        tag = match.group(2).decode("ascii")
        payload = text[match.end() :]
        if tag == "base64":
            return base64.b64decode(payload)
        if tag == "base85":
            return base64.b85decode(payload)
        if tag == ESCAPED:
            return unescape_bytes(payload)
        if tag == "raw":
            return payload
    # Fall back to the substring search for values that reach here without a
    # well-formed header.
    if b";base64," in text:
        _, encoded = text.split(b";base64,", 1)
        return base64.b64decode(encoded)
    elif b";base85," in text:
        _, encoded = text.split(b";base85,", 1)
        return base64.b85decode(encoded)
    elif b";raw," in text:
        _, raw_data = text.split(b";raw,", 1)
        return raw_data
    else:
        raise ValueError("Invalid binary format")


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
