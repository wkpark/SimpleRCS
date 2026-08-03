"""Pure format primitives for SimpleRCS blocks.

Stateless helpers for encoding/decoding binary payloads, escaping ``@`` in
``@...@`` values, and computing the v2 hash-chain block hash. Kept free of any
instance state so they can be unit-tested in isolation; ``SimpleRCS`` passes its
configured ``hash_algo``/``encoding`` in as arguments.
"""

import base64
import hashlib


def encode_binary(data: bytes, encoding: str = "base64") -> bytes:
    """Encodes binary data to bytes format: <length>;<encoding>,<encoded>"""
    if encoding == "base85":
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
