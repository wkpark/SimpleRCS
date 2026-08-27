import difflib
import hashlib
import io
import logging
import os
import re
from collections.abc import Callable
from datetime import datetime
from typing import BinaryIO

from . import codec, pybsdiff
from .matchers import new_matcher

logger = logging.getLogger(__name__)

#: What a block starts with. `_format_block` has only ever written "ver"
#: (`keys = ["ver", ...]` since the initial commit); "version" was accepted by
#: the readers "for backward compatibility" with a writer that never existed,
#: and searching for it doubled the false-positive surface for nothing.
_BLOCK_MARKER = b"ver @"

# _is_block_boundary reads this much at a candidate: the 2-byte blank line
# before the marker, the marker itself, and one byte after its '@'.
_BOUNDARY_WINDOW = 2 + len(_BLOCK_MARKER) + 1

# Re-read this much of the previous chunk so a marker split across a chunk
# boundary is still found whole.
_SCAN_OVERLAP = len(_BLOCK_MARKER) - 1


class SimpleRCSCorruptionError(ValueError):
    """Raised when historical block data cannot be reconstructed because a
    delta payload is corrupted (bad base64/base85, truncated BSDIFF patch,
    malformed RCS delta script, etc.). Subclasses ValueError so existing
    `except ValueError` callers keep working."""


class SimpleRCS:
    """
    A simple, robust, and efficient version control system inspired by RCS.
    Supports both Text (RCS diff) and Binary (BSDIFF) files.
    It operates on a file-like object (stream), supporting both on-disk files and in-memory buffers.

    Architecture & Key Features:
    ----------------------------
    1.  **Stream-Centric:** Operates on a file-like object (binary stream).
        Supports direct file paths, in-memory strings/bytes (via BytesIO), or existing file handles.
        This makes it flexible for use with databases (storing blobs/text) or file systems.

    2.  **Reverse Delta Storage (Optimized for Read):**
        -   The **latest version (HEAD)** is always stored as **Full Text** at the end of the stream.
        -   All previous versions are stored as **Reverse Deltas**
            (instructions to transform Version N+1 back to Version N).
        -   This ensures O(1) access time for the most frequently accessed version (HEAD).
        -   Accessing historical versions requires applying deltas backwards from HEAD (O(k)
            where k is distance from HEAD).

    3.  **Append-Only-Like Modification:**
        -   While not strictly append-only (it modifies the previous HEAD block),
            it only affects the tail of the file.
        -   When a new version is committed:
            1.  The current HEAD (Full Text) is read.
            2.  A Reverse Delta (New -> Old) is calculated.
            3.  The current HEAD block on disk is **overwritten** with this Delta.
            4.  The new Version (Full Text) is **appended** to the end.
        -   This minimizes file IO and avoids rewriting the entire history.

    4.  **Efficient Backward Scanning:**
        -   Uses `seek()` to scan from the end of the file backwards to find block boundaries.
        -   This avoids loading the entire file into memory, making it scalable for large histories.

    5.  **RCS Diff Format:**
        -   Uses a format compatible with `diff -n` (RCS): `d<line> <count>` and `a<line> <count>`.

    6.  **Hash Chain & Integrity (v2):**
        -   Supports v2 format with configurable hash algorithms (default SHA-256).
        -   Each block contains a `hash` of its content (Full Text + Metadata) and the `prev_hash`.
        -   This ensures tamper-evidence. Changing any past version breaks the chain.

    7.  **Intermediate Snapshots (New in v2.1):**
        -   Supports storing full text snapshots at intermediate versions.
        -   This allows faster retrieval of historical versions by breaking the delta chain.
        -   Snapshots are marked by the `text` keyword (instead of `delta`).
    """

    def __init__(self, content_or_path: str | bytes | BinaryIO | None = None, hash_algo: str = "sha256") -> None:  # noqa: C901
        """
        Initializes the SimpleRCS instance.

        Args:
            content_or_path:
                - None: Creates a new empty in-memory RCS (BytesIO).
                - str (path): Opens the file at the given path. Creates it if not exists.
                - str (content): Treats the string as RCS content (wraps in BytesIO).
                - bytes: Treats bytes as RCS content (wraps in BytesIO).
                - file-like object: Uses the provided binary stream directly.
            hash_algo: Hash algorithm to use for new v2 files (default: "sha256").
                       Must be supported by hashlib.
        """
        self.file_path: str | None = None
        self.stream: BinaryIO
        self.owns_handle = False  # Flag to indicate if we opened the file handle and should close it
        self._version = 1  # Default to v1
        self._hash_algo = hash_algo
        self.encoding = "utf-8"

        # Validate hash_algo early
        if hash_algo not in hashlib.algorithms_available:
            try:
                hashlib.new(hash_algo)
            except ValueError as e:
                raise ValueError(f"Hash algorithm '{hash_algo}' is not supported by hashlib.") from e

        if content_or_path is None:
            # New empty in-memory RCS -> v2
            self.stream = io.BytesIO()
            self._write_v2_header()
            self._version = 2
            self.owns_handle = True
        elif isinstance(content_or_path, str):
            # Check if it's a file path heuristic
            if os.path.exists(content_or_path) or (
                not content_or_path.startswith(("@", "ver", "#")) and "\n" not in content_or_path
            ):
                # Assume file path
                is_existing = os.path.exists(content_or_path)
                mode = "rb+" if is_existing else "wb+"
                self.stream = open(content_or_path, mode)
                self.file_path = content_or_path
                self.owns_handle = True

                # Check if file is empty
                is_empty = False
                if is_existing:
                    self.stream.seek(0, os.SEEK_END)
                    if self.stream.tell() == 0:
                        is_empty = True
                    self.stream.seek(0)

                # Check version for existing file, or init new file
                if mode == "wb+" or is_empty:
                    self._write_v2_header()
                    self._version = 2
                else:
                    self._version = self._detect_format_version()
            else:
                # Treat as raw string content -> In-memory stream
                self.stream = io.BytesIO(content_or_path.encode(self.encoding))
                self._version = self._detect_format_version()
                self.owns_handle = True
        elif isinstance(content_or_path, bytes):
            self.stream = io.BytesIO(content_or_path)
            self._version = self._detect_format_version()
            self.owns_handle = True
        elif hasattr(content_or_path, "read") and hasattr(content_or_path, "seek"):
            # External file-like object
            self.stream = content_or_path
            self._version = self._detect_format_version()
            self.owns_handle = False
        else:
            raise ValueError("Invalid input type. Expected file path, string, bytes, or file-like object.")

        # self._data_start (absolute offset where the first commit block
        # begins: 0 for v1, or the v2 header's byte length) was already set
        # above by whichever of `_write_v2_header`/`_detect_format_version`
        # ran (every branch calls exactly one) -- both already read/write
        # the header line, so they set it for free instead of this needing
        # a separate re-read. Used by `_is_block_boundary` to recognize the
        # very first block, which has no preceding blank line to check.

        self.head_info: dict | None = None
        self._head_cache_size: int = -1  # stream size at last successful _load_head scan
        self._head_meta_only: bool = False  # True when cached head_info has no 'text'
        self._load_head()

    def _is_block_boundary(self, abs_start: int, buf: bytes | None = None, buf_origin: int = 0) -> bool:
        """True if `abs_start` is a genuine block-start offset, not a
        coincidental "ver @" match inside escaped block content.

        `_format_block` always writes a block as `"\\n".join(lines) +
        "\\n\\n"`, so every real block (except the very first, which follows
        the v2 header or starts at offset 0 for v1 — see `_data_start`) is
        immediately preceded by a blank line. `_load_head`/`_get_prev_block`
        locate candidate block starts by scanning raw bytes for the literal
        marker "ver @", but `codec.escape` only doubles '@' — it does not
        protect that 5-byte sequence from occurring inside an escaped field
        value (e.g. a log message or page body containing "whatever @2").
        Requiring the blank
        line immediately before a candidate rules out that class of false
        positives on its own.

        This check alone is NOT sufficient, though: `codec.escape` doesn't
        escape newlines either, so ordinary content containing a literal
        blank line immediately followed by "ver @" (e.g. wiki prose like
        "...\\n\\nwhatever @2.0 release notes...") still passes
        it. Callers close that gap themselves by re-scanning further left
        whenever parsing what looked like a valid boundary still fails —
        see `_scan_for_block`. Only content that reproduces *both* the
        blank line *and* a value that itself parses as a well-formed block
        can still fool the pair together, which is a categorically
        narrower coincidence than a bare substring match.

        `buf`/`buf_origin`, if given, let the caller's already-read window
        (e.g. `_load_head`'s `search_buf`) serve the 2-byte lookback
        directly, avoiding a stream seek+read for the common case where
        those bytes are already in memory.
        """
        if abs_start == self._data_start:
            return True
        if abs_start < 2:
            return False
        # The blank line before the marker, plus enough after it to see whether
        # the marker's '@' opens a run: from `buf` when it holds the window,
        # else from the stream.
        window = None
        if buf is not None:
            rel = abs_start - 2 - buf_origin
            if 0 <= rel and rel + _BOUNDARY_WINDOW <= len(buf):
                window = buf[rel : rel + _BOUNDARY_WINDOW]
        if window is None:
            pos = self.stream.tell()
            self.stream.seek(abs_start - 2)
            window = self.stream.read(_BOUNDARY_WINDOW)
            self.stream.seek(pos)
        if not window.startswith(b"\n\n"):
            return False
        # '@' parity. Every '@' inside a field value is doubled, so a run there
        # is always even; a delimiter is a single '@', and a version never
        # starts with one, so a real marker's run is exactly 1. Reading one byte
        # past the marker's '@' separates them:
        #
        #   ver @1.1@;    run = 1  -> delimiter, a real block starts here
        #   ver @@@@;     run = 4  -> inside an escaped value, keep scanning
        #
        # This is exact wherever every field is escaped -- which is what the
        # 'esc' encoding is for. A base85 or raw payload is stored unescaped and
        # can still put an odd run inside a value, so there the check only
        # narrows the candidates and _scan_for_block's retry does the rest.
        return not window[2:].startswith(b"ver @@")

    def _scan_for_block(self, buf: bytes, limit: int, buf_origin: int) -> int:
        """Rightmost offset in `buf[:limit]` (`buf_origin` = the stream
        offset `buf[0]` corresponds to) where "ver @" marks a
        genuine block boundary per `_is_block_boundary`, or -1 if none.

        Does not attempt to parse. A candidate that passes the boundary
        check can still fail to parse (see `_is_block_boundary`'s
        docstring for why) — callers must re-call with `limit` shrunk to
        the returned idx when that happens, to keep looking left instead
        of treating a merely-structurally-plausible offset as the answer.
        """
        while True:
            idx = buf.rfind(_BLOCK_MARKER, 0, limit)
            if idx == -1 or self._is_block_boundary(buf_origin + idx, buf, buf_origin):
                return idx
            limit = idx

    def _resolve_block(self, abs_start: int, block_end: int, metadata_only: bool, get_content_bytes) -> dict:
        """Parses the block candidate at `abs_start` (already confirmed a
        genuine boundary by `_scan_for_block`/`_is_block_boundary`), or
        returns `{}` if it doesn't actually parse — a structurally
        boundary-shaped offset can still fail to parse (see
        `_is_block_boundary`'s docstring for why); callers must treat that
        the same as a rejected candidate and keep scanning left, not as a
        genuine (if malformed) block.

        Shared by `_load_head`/`_get_prev_block` so the two don't drift
        into different rejection behavior, as they briefly did before this
        was factored out.

        `get_content_bytes`, a zero-arg callable, is only invoked (lazily)
        for the `metadata_only=False` path once the cheap metadata
        pre-check below has already passed — each caller supplies its own
        cheapest way to get those bytes (in-memory chunk reuse vs a fresh
        seek+read), matching how they already assemble content elsewhere.
        """
        if metadata_only:
            return self._parse_block_meta_from_stream(abs_start, block_end)
        # Cheap pre-check via the streaming metadata parser before paying
        # for the full assembly: content with several "blank line + marker"
        # collisions (see _is_block_boundary) can make the caller's retry
        # loop try many candidates, and _parse_block_meta_from_stream reads
        # incrementally and fails fast on a rejected one, instead of the
        # O(bytes from abs_start to block_end) that eagerly building the
        # full content bytes for every attempt would cost.
        if not self._parse_block_meta_from_stream(abs_start, block_end):
            return {}
        return self._parse_block_content_no_regex(get_content_bytes())

    def _write_v2_header(self) -> None:
        """Writes the v2 magic header with the chosen hash algorithm.

        Also sets `self._data_start` to the header's byte length: we just
        wrote it, so there's nothing to re-read to learn that.
        """
        header = f"# SimpleRCS v2.0; hash_algo={self._hash_algo}; encoding={self.encoding};\n"
        header_bytes = header.encode(self.encoding)
        self.stream.write(header_bytes)
        self._data_start = len(header_bytes)

    def _detect_format_version(self) -> int:
        """
        Detects format version and hash algorithm from the stream header.
        Updates self._hash_algo and self._data_start (0 for v1, the header's
        byte length for v2 — reusing the readline() below rather than
        re-reading it separately just to learn that length).
        """
        pos = self.stream.tell()
        self.stream.seek(0)
        # Read first line, handle potential binary data gracefully
        try:
            header_bytes = self.stream.readline()
            header = header_bytes.decode(self.encoding).strip()
        except UnicodeDecodeError:
            header_bytes = b""
            header = ""
        finally:
            self.stream.seek(pos)  # Restore position

        if header.startswith("# SimpleRCS v2.0;"):
            # Parse hash_algo
            match = re.search(r"hash_algo=([^;]+)", header)
            if match:
                algo = match.group(1).strip()
                try:
                    hashlib.new(algo)
                    self._hash_algo = algo
                except ValueError as e:
                    raise ValueError(f"Invalid hash algo {algo}") from e
            # Parse encoding
            match_encoding = re.search(r"encoding=([^;]+)", header)
            if match_encoding:
                self.encoding = match_encoding.group(1).strip()
            self._data_start = len(header_bytes)
            return 2
        self._data_start = 0
        return 1

    def __del__(self) -> None:
        """Closes the stream if this instance owns it."""
        if self.owns_handle and hasattr(self, "stream") and not self.stream.closed:
            self.stream.close()

    def get_bytes(self) -> bytes:
        """Returns the full RCS stream verbatim.

        Use this rather than :meth:`get_content` whenever the stream may hold a
        binary block written with the ``esc`` encoding: those payloads are raw
        bytes, and decoding them as text is lossy.
        """
        pos = self.stream.tell()
        self.stream.seek(0)
        content = self.stream.read()
        self.stream.seek(pos)
        return content

    def get_content(self) -> str:
        """
        Returns the full content of the RCS stream as a string.
        Useful when working with in-memory streams to get the final result.

        Text-safe encodings only. ``esc`` binary payloads are raw bytes and do
        not survive the decode -- use :meth:`get_bytes` for those.
        """
        pos = self.stream.tell()
        self.stream.seek(0)
        content = self.stream.read().decode(self.encoding, errors="replace")
        self.stream.seek(pos)
        return content

    def _unescape(self, text: str) -> str:
        """Unescapes '@@' back to '@'."""
        return text.replace("@@", "@")

    def _parse_block_content(self, content_bytes: bytes) -> dict:  # noqa: C901
        """
        Parses a raw block bytes into a dictionary.
        Format: key @value@; ...

        Not used on the hot path (see _parse_block_content_no_regex, which all
        callers use instead) -- kept intentionally as a simpler, more readable
        reference implementation of the block format. Covered by
        test_parse_block_content_matches_no_regex to catch drift between the two.

        Limit worth knowing: it works on a decoded ``str``, so it cannot
        represent a ``raw`` binary payload -- those bytes do not survive
        ``decode(errors="replace")``. Text, ``base64`` and ``base85`` blocks are
        exact; the pin test covers those.
        """
        content_str = content_bytes.decode(self.encoding, errors="replace")
        data = {}
        # Regex matches keys (ver, date, etc.) and values enclosed in @...@
        # Use re.DOTALL to match newlines within @...@
        pattern = re.compile(r"(ver|date|author|log|text|delta|binary)\s+@((?:[^@]|@@)*)@;", re.DOTALL)

        # Iterate over all matches to build the data dictionary
        for match in pattern.finditer(content_str):
            key = match.group(1)
            # Binary payloads keep their escaping: codec.decode_binary unescapes
            # them itself, and the no-regex parser hands them over verbatim from
            # its length-based read. Unescaping here as well would collapse a
            # doubled '@' twice.
            escaped = match.group(2)
            value = self._unescape(escaped)

            # `ver` is always a block's first field (see _format_block), so a
            # second one starts the *next* block. Callers may hand us a range
            # running to EOF, so stop rather than letting the following block's
            # fields overwrite this one's.
            if key == "ver" and "ver" in data:
                break

            if key == "delta":
                data["is_delta"] = True
                if codec.binary_tag(escaped) is not None:
                    # A binary patch. The no-regex parser reads it length-based
                    # and stores the stored form verbatim; match that.
                    data["text"] = escaped.encode(self.encoding)
                    data["is_binary"] = True
                else:
                    data["text"] = value
                    data["is_binary"] = False
            elif key == "text":
                data["text"] = value
                data["is_delta"] = False
            elif key == "binary":
                data["text"] = codec.decode_binary(escaped.encode(self.encoding))
                data["is_delta"] = False
                data["is_binary"] = True
            elif key == "signature":
                if "signatures" not in data:
                    data["signatures"] = []
                data["signatures"].append(value)
            else:
                data[key] = value

        # Basic validation to ensure it looks like a valid block
        if "ver" in data:
            if "is_delta" not in data:
                data["is_delta"] = False
            if "is_binary" not in data:
                data["is_binary"] = False
            return data
        return {}

    def _parse_block_content_no_regex(self, content_bytes: bytes) -> dict:  # noqa: C901
        """
        Parses a raw block bytes into a dictionary WITHOUT regex.
        Format: key @value@; ...
        This method uses direct byte-stream manipulation for potentially higher performance
        and robustness against malformed regex inputs.
        """
        # Define keywords as bytes for direct comparison
        # Added v2 keywords: prev_hash, hash, signature
        # Added 'delta' for mixed snapshot support
        _keywords = [
            b"ver",
            b"date",
            b"author",
            b"log",
            b"text",
            b"delta",
            b"binary",
            b"prev_hash",
            b"hash",
            b"signature",
        ]

        content = content_bytes
        length = len(content)
        pos = 0
        data = {}

        while pos < length:
            # Skip whitespace
            while pos < length and content[pos] in b" \t\r\n":
                pos += 1
            if pos >= length:
                break

            # Read Key
            key_start = pos
            while pos < length and content[pos] not in b" @;":
                pos += 1
            key_bytes = content[key_start:pos]

            # Ensure the key is a valid keyword, otherwise break (malformed block)
            if key_bytes not in _keywords:
                break

            # `ver` is always a block's first field (see _format_block), so a
            # second one starts the *next* block. Callers may hand us a range
            # running to EOF, so stop rather than letting the following block's
            # fields overwrite this one's. Kept identical in
            # _parse_block_content (the reference parser) -- see
            # test_parse_block_content_matches_no_regex.
            if key_bytes == b"ver" and "ver" in data:
                break

            # Skip whitespace after key
            while pos < length and content[pos] in b" \t\r\n":
                pos += 1

            # Expect '@' for value start
            if pos >= length or content[pos] != ord("@"):
                # Malformed or unexpected char where '@' was expected
                break

            pos += 1  # Skip opening '@'

            # Length-Based Parsing Check
            # Check if value starts with "<digits>;"
            # Only allow for 'binary' or 'delta' keys to prevent collisions in 'text' fields
            allow_length_based = key_bytes in (b"binary", b"delta")

            is_length_based = False
            data_len = 0
            header_end_idx = -1  # Relative to pos

            if allow_length_based:
                # We peek a small chunk to check for the header.
                peek_len = min(length - pos, 50)
                peek_chunk = content[pos : pos + peek_len]

                # Find ';'
                semi_idx = peek_chunk.find(b";")
                if semi_idx != -1:
                    len_str_bytes = peek_chunk[:semi_idx]
                    if len_str_bytes.isdigit():
                        try:
                            data_len = int(len_str_bytes)
                            # Found length. Now find end of header (',')
                            comma_idx = peek_chunk.find(b",", semi_idx)
                            if comma_idx != -1:
                                header_end_idx = comma_idx + 1  # Include comma
                                is_length_based = True
                        except ValueError:
                            pass


            val_parts = []
            if is_length_based:
                # Read Header
                header_bytes = content[pos : pos + header_end_idx]
                val_parts.append(header_bytes)
                pos += header_end_idx

                # Read Data (exactly data_len bytes)
                if pos + data_len > length:
                    # Truncated
                    break

                data_bytes = content[pos : pos + data_len]
                val_parts.append(data_bytes)
                pos += data_len

                # Verify closing '@'
                if pos < length and content[pos] == ord("@"):
                    pos += 1  # Skip closing '@'
                # Closing ';' handled by main loop
                # Final assembling of value_bytes
                value_bytes = b"".join(val_parts)
            else:
                # Delimiter-Based Read (Legacy/Text/Base64)
                # Read Value, handling '@@' escaping
                while pos < length:
                    # Fast search for next '@'
                    end = content.find(b"@", pos)
                    if end == -1:
                        # Unterminated string, malformed block
                        break

                    # Check for double '@@' (escaped '@')
                    if end + 1 < length and content[end + 1] == ord("@"):
                        val_parts.append(content[pos:end])
                        val_parts.append(b"@")  # Unescape @@ -> @
                        pos = end + 2
                    else:
                        val_parts.append(content[pos:end])
                        pos = end + 1  # Skip closing '@'
                        break

                # Final assembling of value_bytes
                value_bytes = b"".join(val_parts)

            # Skip whitespace after value
            while pos < length and content[pos] in b" \t\r\n":
                pos += 1

            # Expect ';' after value
            if pos < length and content[pos] == ord(";"):
                pos += 1
            else:
                # Malformed, missing ';'
                break

            # Store data based on key and parsing strategy
            key_str = key_bytes.decode(self.encoding)

            if key_str == "binary":
                # Binary data is always length-prefixed and decoded.
                data["text"] = codec.decode_binary(value_bytes)
                data["is_delta"] = False
                data["is_binary"] = True
            elif key_str == "delta":
                # Delta can be RCS text delta or binary delta (base64/base85/raw).
                # If length-based parsing was used, value_bytes is the full header + data payload.
                # _decode_binary will handle header stripping and decoding.

                if is_length_based:
                    # For length-based delta, value_bytes is the full header + data payload (bytes).
                    # Store it directly. _apply_reverse_delta will handle splitting/decoding.
                    data["text"] = value_bytes
                    data["is_delta"] = True
                    data["is_binary"] = True  # Always binary if length-based delta
                else:
                    value_str = value_bytes.decode(self.encoding, errors="replace")
                    data["text"] = value_str
                    data["is_delta"] = True
                    # Check for legacy base64/base85 signatures in text string
                    data["is_binary"] = ";base64," in value_str or ";base85," in value_str
            elif key_str == "text":
                # Text content is always delimiter-based and decoded to string.
                value_str = value_bytes.decode(self.encoding, errors="replace")
                data["text"] = value_str
                data["is_delta"] = False
                data["is_binary"] = False
            elif key_str == "signature":
                value_str = value_bytes.decode(self.encoding, errors="replace")
                if "signatures" not in data:
                    data["signatures"] = []
                data["signatures"].append(value_str)
            else:
                value_str = value_bytes.decode(self.encoding, errors="replace")
                data[key_str] = value_str

        # Basic validation to ensure it looks like a valid block
        if "ver" in data:
            if "is_delta" not in data:
                data["is_delta"] = False
            if "is_binary" not in data:
                data["is_binary"] = False
            return data
        return {}

    def _parse_block_meta_from_stream(self, abs_start: int, block_end: int) -> dict:  # noqa: C901
        """
        Parse block metadata directly from the stream without loading the full content.

        Strategy (inspired by moniwiki editlog_raw_lines):
          - Meta fields (ver/date/author/log/hash/sig): read and parse normally.
          - binary field: parse length header, seek past N bytes — zero decode.
          - delta/text field: scan for closing @; byte-by-byte — no bytes collected.

        Returns a dict with meta fields plus:
          content_stream_offset, content_length, content_encoding  (for lazy checkout)
        """
        CHUNK = 8192
        _meta_kw = {b"ver", b"date", b"author", b"log", b"prev_hash", b"hash", b"signature"}
        _skip_kw = {b"binary", b"text", b"delta"}
        _all_kw = _meta_kw | _skip_kw

        self.stream.seek(abs_start)
        buf = b""
        stream_cursor = abs_start  # next read position in stream
        buf_origin = abs_start  # stream offset of buf[0]

        def _fill():
            nonlocal buf, stream_cursor
            want = min(CHUNK, block_end - stream_cursor)
            if want <= 0:
                return
            chunk = self.stream.read(want)
            if not chunk:  # premature EOF (truncated/corrupted file)
                stream_cursor = block_end
                return
            buf += chunk
            stream_cursor += len(chunk)

        def _ensure(n, pos):
            # ensure buf[pos:pos+n] is loaded
            while len(buf) - pos < n and stream_cursor < block_end:
                _fill()

        _fill()
        data: dict = {}
        pos = 0

        while True:
            # skip whitespace
            while pos < len(buf) and buf[pos] in b" \t\r\n":
                pos += 1
            if pos >= len(buf):
                if stream_cursor < block_end:
                    _fill()
                    continue
                break

            # read keyword
            key_start = pos
            while pos < len(buf) and buf[pos] not in b" @;\t\r\n":
                pos += 1
                if pos >= len(buf) and stream_cursor < block_end:
                    _fill()
            key_bytes = buf[key_start:pos]

            if key_bytes not in _all_kw:
                break

            # `ver` is always a block's first field (see _format_block), so a
            # second one starts the *next* block. This parser is handed a
            # range ending at block_end, which callers set to EOF (or to the
            # following block's start) rather than to this block's own end --
            # stop here so the next block's fields can't overwrite this one's,
            # matching both content parsers.
            if key_bytes == b"ver" and "ver" in data:
                break

            # skip whitespace before '@'
            while pos < len(buf) and buf[pos] in b" \t":
                pos += 1
                if pos >= len(buf) and stream_cursor < block_end:
                    _fill()

            if pos >= len(buf) or buf[pos] != ord("@"):
                break
            pos += 1  # skip opening '@'

            if key_bytes in _meta_kw:
                # delimiter-based read — collect until unescaped '@'
                val_parts = []
                while True:
                    _ensure(1, pos)
                    if pos >= len(buf):
                        break
                    at = buf.find(b"@", pos)
                    if at == -1:
                        val_parts.append(buf[pos:])
                        pos = len(buf)
                        continue
                    _ensure(2, at)
                    if at + 1 < len(buf) and buf[at + 1] == ord("@"):
                        val_parts.append(buf[pos:at])
                        val_parts.append(b"@")
                        pos = at + 2
                    else:
                        val_parts.append(buf[pos:at])
                        pos = at + 1
                        break
                value = b"".join(val_parts).decode(self.encoding, errors="replace")
                # skip whitespace, then require ';'. A real field always has
                # one (see _format_block) -- unlike _parse_block_content_no_regex,
                # this loop used to store the key/value anyway when it was
                # missing, so a false-positive scan match that failed this
                # same check in the other parser could still "succeed" here,
                # producing a garbage-but-accepted entry (e.g. a phantom
                # {'ver': '', 'author': None, ...} row in log()) instead of
                # failing the same way. Missing ';' means malformed/truncated
                # data either way -- discard what we collected and stop,
                # matching the no-regex parser's stricter behavior.
                _ensure(2, pos)
                while pos < len(buf) and buf[pos] in b" \t\r\n":
                    pos += 1
                if pos < len(buf) and buf[pos] == ord(";"):
                    pos += 1
                else:
                    break

                key_str = key_bytes.decode()
                if key_str == "signature":
                    data.setdefault("signatures", []).append(value)
                else:
                    data[key_str] = value

            elif key_bytes == b"binary":
                # length-based: "N;encoding,<N bytes>@;"
                _ensure(50, pos)
                peek = buf[pos : pos + 50]
                semi = peek.find(b";")
                if semi != -1 and peek[:semi].isdigit():
                    data_len = int(peek[:semi])
                    comma = peek.find(b",", semi)
                    if comma != -1:
                        enc_str = peek[semi + 1 : comma].decode("ascii", errors="replace")
                        header_len = comma + 1
                        content_abs = buf_origin + pos + header_len
                        # seek past content + closing '@'
                        skip_to = content_abs + data_len + 1  # +1 = closing '@'
                        self.stream.seek(skip_to)
                        stream_cursor = skip_to
                        buf = b""
                        pos = 0
                        buf_origin = skip_to
                        _fill()
                        # skip whitespace, then require ';' -- and only store
                        # this field once we've confirmed it, same as the
                        # _meta_kw branch above and _parse_block_content_no_regex:
                        # a missing terminator means malformed/truncated data
                        # either way, so nothing from this field (not even the
                        # seek-skip's byte range) should end up in `data`.
                        while pos < len(buf) and buf[pos] in b" \t\r\n":
                            pos += 1
                        if pos < len(buf) and buf[pos] == ord(";"):
                            pos += 1
                            data["content_stream_offset"] = content_abs
                            data["content_length"] = data_len
                            data["content_encoding"] = enc_str
                            data["is_binary"] = True
                            data["is_delta"] = False
                        else:
                            break

            else:  # delta / text — scan for unescaped '@' then ';', no collection
                is_delta_key = key_bytes == b"delta"
                # Peek to detect binary (length-based) delta: starts with "N;encoding,"
                use_seek_skip = False
                is_binary_val = False
                if is_delta_key:
                    _ensure(50, pos)
                    peek = buf[pos : pos + 50]
                    semi = peek.find(b";")
                    comma = peek.find(b",", semi) if semi != -1 else -1
                    is_len_based = semi != -1 and peek[:semi].isdigit() and comma != -1
                    is_binary_val = is_len_based
                    if is_len_based:
                        enc_str = peek[semi + 1 : comma].decode("ascii", errors="replace")
                        # encode_binary records the byte count that is actually
                        # on disk for every encoding, so the skip is exact.
                        use_seek_skip = True

                if use_seek_skip:
                    # base64 binary delta: skip N bytes of encoded content via seek
                    data_len = int(peek[:semi])
                    header_len = comma + 1  # bytes of "N;base64," prefix
                    content_abs = buf_origin + pos + header_len
                    skip_to = content_abs + data_len + 1  # +1 = closing '@'
                    self.stream.seek(skip_to)
                    stream_cursor = skip_to
                    buf = b""
                    pos = 0
                    buf_origin = skip_to
                    _fill()
                    # skip ';'
                    while pos < len(buf) and buf[pos] in b" \t\r\n":
                        pos += 1
                    if pos < len(buf) and buf[pos] == ord(";"):
                        pos += 1
                    else:
                        break
                else:
                    # text delta or base85 delta: scan for unescaped '@'.
                    # buf grows monotonically up to O(delta_size) — no seek available
                    # because base85 '@' chars are stored as '@@', making stored len > N.
                    while True:
                        _ensure(2, pos)
                        if pos >= len(buf):
                            # _ensure postcondition: stream_cursor >= block_end here
                            break
                        at = buf.find(b"@", pos)
                        if at == -1:
                            pos = len(buf)
                            continue
                        _ensure(2, at)
                        if at + 1 < len(buf) and buf[at + 1] == ord("@"):
                            pos = at + 2  # escaped @@
                        else:
                            pos = at + 1  # closing '@'
                            break
                    # skip whitespace, then require ';' -- same reasoning as
                    # the seek-skip branch above and the _meta_kw branch
                    # earlier: don't record is_delta/is_binary for a field
                    # that was never properly terminated.
                    _ensure(2, pos)
                    while pos < len(buf) and buf[pos] in b" \t\r\n":
                        pos += 1
                    if pos < len(buf) and buf[pos] == ord(";"):
                        pos += 1
                    else:
                        break

                data["is_delta"] = is_delta_key
                data["is_binary"] = is_binary_val

        if "ver" in data:
            data.setdefault("is_delta", False)
            data.setdefault("is_binary", False)
            return data
        return {}

    def _load_head(self, force: bool = False, metadata_only: bool = False) -> None:
        """
        Locates and loads ONLY the last block (HEAD) by scanning backwards from EOF.
        This is a performance optimization to avoid reading the entire history when
        we only need the latest version. Sets self.head_info = { ..., 'start': offset, 'end': offset }.

        Single-pass design (inspired by moniwiki editlog_raw_lines):
          - Chunks are accumulated in a deque as they are read (right-to-left).
          - When "ver @" is found, HEAD block bytes are assembled from accumulated
            chunks without a second seek+read — total I/O equals HEAD block size once.
          - No arbitrary size limit: works correctly for HEAD blocks of any size.

        Caching: skips the backward scan if the stream size is unchanged since the last call.
        Pass force=True to bypass the cache (e.g. after external writes to the stream).
        """
        self.stream.seek(0, os.SEEK_END)
        file_size = self.stream.tell()

        # Cache: skip expensive scan when stream size is unchanged.
        # Full-content callers (metadata_only=False) require 'text' in cache.
        if not force and self.head_info is not None and file_size == self._head_cache_size:
            if metadata_only or not self._head_meta_only:
                return

        self.head_info = None

        if file_size == 0:
            self._head_cache_size = 0
            return

        CHUNK = 4096
        OVERLAP = _SCAN_OVERLAP

        pos = file_size
        # collected: chunks in right-to-left read order (index 0 = rightmost / EOF side).
        # Mirrors moniwiki's $last accumulation — nothing is discarded.
        collected: list[bytes] = []
        tail_prefix = b""

        while pos > 0:
            read_size = min(CHUNK, pos)
            pos -= read_size
            self.stream.seek(pos)
            chunk = self.stream.read(read_size)
            if not metadata_only:
                # metadata_only path reads directly via _parse_block_meta_from_stream;
                # no need to accumulate chunks for head_bytes assembly.
                collected.append(chunk)

            # Search for the rightmost genuine block boundary in this chunk +
            # OVERLAP bytes of the chunk to its right (to catch patterns
            # split across a boundary).
            search_buf = chunk + tail_prefix
            search_limit = len(search_buf)

            while True:
                idx = self._scan_for_block(search_buf, search_limit, pos)
                if idx == -1:
                    break
                abs_start = pos + idx

                def _content_bytes(idx=idx, abs_start=abs_start, chunk=chunk):
                    if idx < len(chunk):
                        # Assemble HEAD block from already-read chunks — no second seek+read.
                        # collected[-1] is the current (leftmost) chunk; collected[0] is the
                        # rightmost. HEAD block = chunk[idx:] + chunks to its right in order.
                        return chunk[idx:] + b"".join(reversed(collected[:-1]))
                    # Match falls inside `tail_prefix` (spans into a chunk
                    # read on a previous, further-right iteration) — rare;
                    # re-read directly rather than threading that chunk
                    # through too.
                    self.stream.seek(abs_start)
                    return self.stream.read(file_size - abs_start)

                parsed = self._resolve_block(abs_start, file_size, metadata_only, _content_bytes)
                if parsed and parsed.get("is_delta"):
                    # HEAD is always stored as full text (commit() formats it
                    # with is_delta=False), so a delta block here is a demoted
                    # one -- meaning the real HEAD sitting after it wasn't
                    # locatable. Accepting it would hand the caller an
                    # unapplied delta script as though it were content, with
                    # nothing later to apply it against. Treat it as a
                    # rejected candidate and keep scanning left.
                    parsed = {}
                if parsed:
                    parsed["start"] = abs_start
                    parsed["end"] = file_size
                    self.head_info = parsed
                    self._head_cache_size = file_size
                    self._head_meta_only = metadata_only
                    return
                # `idx` passed the structural boundary check but still
                # didn't parse — either genuine corruption, or yet another
                # coincidental blank-line + marker match (see
                # _is_block_boundary). Keep looking further left within
                # this same buffer rather than assuming the worst.
                search_limit = idx

            tail_prefix = chunk[:OVERLAP]

        self._head_cache_size = file_size

    def _get_prev_block(self, current_start_offset: int, metadata_only: bool = False) -> dict | None:
        """
        Finds and parses the block immediately preceding the given offset.
        Used for traversing history backwards (HEAD -> V_prev -> ...).

        metadata_only=True skips content decoding (binary seek, delta scan).
        Returned dict has content_stream_offset/content_length/content_encoding
        instead of 'text'. Used by log() to avoid decoding large payloads.
        """
        if current_start_offset <= 0:
            return None

        chunk_size = 4096
        scan_pos = current_start_offset
        overlap = 100  # To catch keywords split across chunk boundaries

        while scan_pos > 0:
            read_len = min(chunk_size, scan_pos)
            scan_pos -= read_len

            self.stream.seek(scan_pos)
            # Read a bit more to handle overlaps and ensure block starts are caught
            chunk = self.stream.read(read_len + overlap)
            # We are interested in data strictly BEFORE current_start_offset (relative to original file)
            # The block we are looking for ends at current_start_offset.
            # So we look for a block start within `chunk` that is before `current_start_offset`.

            # Index for current_start_offset within `chunk` (relative to chunk's start)
            # `chunk` starts at `scan_pos`. `current_start_offset` is `(current_start_offset - scan_pos)` bytes
            # into `chunk`.
            limit_in_chunk = current_start_offset - scan_pos

            # Search for a genuine block boundary (`ver @`) in
            # `chunk` up to `limit_in_chunk`, retrying further left within
            # this same buffer whenever a candidate doesn't pan out — either
            # it's a false positive (see _is_block_boundary) or it passed
            # that check but still failed to parse — before falling back to
            # an earlier chunk.
            search_limit = limit_in_chunk
            while True:
                idx = self._scan_for_block(chunk, search_limit, scan_pos)
                if idx == -1:
                    break

                abs_start = scan_pos + idx  # Absolute offset of previous block's start

                def _content_bytes(idx=idx, abs_start=abs_start, chunk=chunk):
                    # Block bytes are usually already inside `chunk` (block size <= chunk_size).
                    # Reuse them instead of a redundant seek+read, mirroring _load_head's
                    # single-pass design. Only fall back to re-reading when the block spans
                    # beyond this chunk (rare: block larger than chunk_size).
                    length = current_start_offset - abs_start
                    end_in_chunk = idx + length
                    if end_in_chunk <= len(chunk):
                        return chunk[idx:end_in_chunk]
                    self.stream.seek(abs_start)
                    return self.stream.read(length)

                parsed = self._resolve_block(abs_start, current_start_offset, metadata_only, _content_bytes)
                if parsed:
                    parsed["start"] = abs_start
                    parsed["end"] = current_start_offset
                    if self._version < 2:
                        # v1 has no 'delta' keyword: every block stores its content
                        # under 'text', and only HEAD (never returned here, since this
                        # method only finds the block *preceding* current_start_offset)
                        # holds full text. Every other v1 block is a reverse delta.
                        parsed["is_delta"] = True
                    return parsed
                # `idx` passed the structural boundary check but still didn't
                # parse — keep looking further left within this same buffer
                # (see the matching comment in _load_head).
                search_limit = idx

        return None

    def _generate_reverse_delta(
        self, new_data: str | bytes | BinaryIO, old_data: str | bytes | BinaryIO, encoding: str = "base64"
    ) -> str | bytes:
        """
        Generates Reverse Delta.
        If inputs are bytes (or BinaryIO), generates BSDIFF delta (base64 encoded).
        If inputs are strings, generates an RCS-style ('diff -n') Reverse Delta.
        BinaryIO inputs are passed directly to the active text matcher (text path,
        see matchers.py) or read once via .read() (binary path, pybsdiff requires
        bytes).
        """
        new_is_stream = hasattr(new_data, "read")
        old_is_stream = hasattr(old_data, "read")

        if isinstance(new_data, bytes) or new_is_stream:
            # Binary Diff (BSDIFF) — pybsdiff requires bytes
            new_bytes = new_data.read() if new_is_stream else new_data
            old_bytes = (
                old_data.read()
                if old_is_stream
                else (old_data if isinstance(old_data, bytes) else old_data.encode(self.encoding))
            )
            patch_data = pybsdiff.diff(new_bytes, old_bytes)  # New -> Old
            # encode_binary already produced the stored form -- payload
            # escaped, header length counting the escaped bytes -- so it travels
            # as bytes to the block writer, which writes it verbatim. Escaping
            # it again there would corrupt it, and a raw payload would not
            # survive a round trip through str anyway.
            return codec.encode_binary(patch_data, encoding=encoding)
        elif isinstance(new_data, str) and (isinstance(old_data, str) or old_is_stream):
            pass
        else:
            raise TypeError("Cannot generate delta between mixed types (str/bytes)")

        # Text Diff (RCS) — old_data stream passed directly; no BytesIO copy when already a stream
        new_stream = io.BytesIO(new_data.encode(self.encoding))
        old_stream = old_data if old_is_stream else io.BytesIO(old_data.encode(self.encoding))

        matcher = new_matcher(new_stream, old_stream)
        output = []

        # opcodes: describes how to turn 'a' (New) into 'b' (Old)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue

            # RCS diff format logic
            # indices are line numbers because chunk_size=None
            xlen = i2 - i1
            ylen = j2 - j1
            xbeg = i1 + 1

            del_cmd = f"d{xbeg} {xlen}"

            # For insert/replace, we add lines after a certain point.
            # RCS 'a' command adds AFTER the specified line index of the input file.
            # If we delete lines, the line numbers shift? No, RCS commands refer to the original state.

            # Logic derived from SimpleRCS original (which matched PHP DeltaDiff):
            # add_idx = xbeg + xlen - 1

            if xlen > 0:
                output.append(del_cmd)

            if ylen > 0:
                if xlen > 0:
                    add_idx = xbeg + xlen - 1  # i2
                else:
                    add_idx = i1  # If insert only (xlen=0), append after i1 (which is xbeg-1)

                add_cmd = f"a{add_idx} {ylen}"
                output.append(add_cmd)

                # Get actual lines from B (old_text)
                # 'matcher' has b_stream as old_stream.
                # We need to read lines j1 to j2.
                # StreamSequenceMatcher.get_lines_from_stream works with indices in Line Mode.

                # Note: get_lines_from_stream expects 'a' or 'b' as first arg to identify stream
                # In StreamSequenceMatcher.__init__, set_seq2 sets self.b_stream.
                # get_lines_from_stream uses self.b_stream if type != 'a'.

                lines_b = matcher.get_lines_from_stream("b", j1, j2)
                for line in lines_b:
                    output.append(line.decode(self.encoding).rstrip("\n"))

        # Every line newline-terminated. Joining without the final "\n" made a
        # payload that ends in a blank line ("a1 1", "") serialize as "a1 1\n",
        # byte-identical to the command with no payload at all, and the blank
        # line was lost on checkout.
        return "\n".join(output) + "\n" if output else ""

    def _apply_reverse_delta(self, current_data: str | bytes, delta_text: str | bytes) -> str | bytes:  # noqa: C901
        """
        Applies a reverse delta.
        Detects binary delta by signature.
        """
        # A binary delta always opens with its "<length>;<encoding>," header, so
        # match it there rather than searching the whole value: a text delta's
        # payload is user content and can contain ";base64," anywhere in it.
        is_binary_delta = codec.binary_tag(delta_text) is not None

        if is_binary_delta:
            if isinstance(current_data, str):
                current_data = current_data.encode(self.encoding)

            # Prepare delta for decoding
            if isinstance(delta_text, str):
                delta_bytes = delta_text.encode("ascii")
            else:
                delta_bytes = delta_text

            patch_data = codec.decode_binary(delta_bytes)
            return pybsdiff.patch(current_data, patch_data)

        # Text Delta (RCS)
        if isinstance(current_data, bytes):
            # Should be string for RCS patch
            current_data = current_data.decode(self.encoding)

        # Ensure delta_text is str for RCS processing
        if isinstance(delta_text, bytes):
            delta_text = delta_text.decode(self.encoding)

        lines = current_data.splitlines(keepends=True)
        commands = []

        script_lines = delta_text.splitlines()
        i = 0
        while i < len(script_lines):
            header = script_lines[i]
            i += 1
            parts = header.split()  # Split for manual parsing
            if not parts:
                continue

            cmd_char = parts[0][0]  # 'd' or 'a'
            if cmd_char not in ("d", "a"):
                continue

            try:
                start = int(parts[0][1:])  # Line number from 'd1' or 'a1'
                count = int(parts[1])  # Count of lines
            except (ValueError, IndexError) as e:
                raise ValueError("Invalid delta format") from e

            payload = []
            if cmd_char == "a":
                # Read payload lines for 'a' command
                for _ in range(count):
                    if i < len(script_lines):
                        payload.append(script_lines[i] + "\n")  # Restore newline
                        i += 1
                if len(payload) == count - 1:
                    # A delta written before the terminator fix above. That
                    # encoder joined the script with "\n" and no trailing one,
                    # so a payload whose last line was blank serialized
                    # identically to one line shorter -- and it could only ever
                    # lose that one line, never more, and never a non-blank one
                    # (the separators for the rest are still there). Restoring
                    # it reproduces the original text exactly; on a v2 stream
                    # the hash chain confirms the reconstruction.
                    #
                    # A file truncated precisely one payload line short would be
                    # repaired into wrong text instead of raising. That needs
                    # surgical damage -- losing the tail normally takes the
                    # closing '@;' with it, and the block fails to parse well
                    # before here -- and v2 still reports it.
                    payload.append("\n")
                if len(payload) != count:
                    # Short by more than one: no encoder this project has ever
                    # shipped produces that, so there is nothing to reconstruct.
                    raise SimpleRCSCorruptionError(
                        f"Delta command 'a{start} {count}' has {len(payload)} payload line(s)"
                    )
            commands.append(
                {"cmd": cmd_char, "line": start, "count": count, "payload": payload, "order": len(commands)}
            )

        # Sort commands by line number in descending order.
        # Tie-break: 'a' before 'd' at the same line (single-line replace emits
        # dN 1 + aN 1 at equal line numbers). Inserting first keeps the delete's
        # target index accurate; deleting first would shift the insert position.
        # Several 'a' at one line (a backend may emit adjacent inserts unmerged)
        # go in reverse emission order: each inserts at the same index, so the
        # last emitted must go in first for the payloads to end up in sequence.
        commands.sort(key=lambda x: (-x["line"], 0 if x["cmd"] == "a" else 1, -x["order"]))

        for cmd in commands:
            idx = cmd["line"]
            if cmd["cmd"] == "a":
                # Append payload AFTER line `idx`. (List insert at index `idx`)
                insert_pos = idx
                lines[insert_pos:insert_pos] = cmd["payload"]
            elif cmd["cmd"] == "d":
                # Delete `count` lines starting AT line `idx` (List index `idx-1`)
                start_pos = idx - 1
                del lines[start_pos : start_pos + cmd["count"]]
        result = "".join(lines)
        # Enforce EOL policy
        if result and not result.endswith("\n"):
            result += "\n"
        return result

    def _rewrite_head(self, payload: bytes) -> None:
        """Replace the tail of the stream, from the current HEAD's start, with payload.

        This is the one destructive operation in the format: it seeks backwards over
        live data and overwrites it. Both callers (commit() and sign_head()) go
        through here so that anything guarding that write only has to be written once.
        """
        self.stream.seek(self.head_info["start"])
        self.stream.write(payload)
        self.stream.truncate()  # Crucial: remove any leftover

    def _format_block(
        self,
        data: dict,
        current_hash: str | None = None,
        prev_hash: str | None = None,
        signatures: list[str] | None = None,
        is_delta: bool = False,
        encoding: str = "base64",
    ) -> bytes:
        """
        Formats a block dictionary into bytes for writing to the stream.
        Supports v2 fields (hash, prev_hash, signatures).
        """
        keys = ["ver", "date", "author", "log"]
        lines = []
        for key in keys:
            val = codec.escape(str(data.get(key, "")))
            lines.append(f"{key} @{val}@;".encode(self.encoding))

        content_val = data.get("text", "")

        if is_delta and isinstance(content_val, bytes):
            # A binary patch: _generate_reverse_delta already produced the stored
            # form (payload escaped, header length counting the escaped bytes),
            # so escaping it again would corrupt it.
            lines.append(b"delta @" + content_val + b"@;")
        elif isinstance(content_val, bytes):
            # Binary Full Text
            # _encode_binary returns bytes: b"len;base64,..."
            val_bytes = codec.encode_binary(content_val, encoding=encoding)
            lines.append(b"binary @" + val_bytes + b"@;")
        else:
            # Text Full Text OR Encoded Delta
            content_key = "delta" if is_delta else "text"
            val = codec.escape(str(content_val))
            lines.append(f"{content_key} @{val}@;".encode(self.encoding))

        # Add v2 fields if present
        if self._version >= 2:
            if prev_hash:
                lines.append(f"prev_hash @{codec.escape(prev_hash)}@;".encode(self.encoding))
            if current_hash:
                lines.append(f"hash @{codec.escape(current_hash)}@;".encode(self.encoding))

            if signatures:
                for sig in signatures:
                    lines.append(f"signature @{codec.escape(sig)}@;".encode(self.encoding))

        return b"\n".join(lines) + b"\n\n"

    def commit(  # noqa: C901
        self,
        content: str | bytes | BinaryIO,
        author: str = "unknown",
        log: str = "",
        signer_callbacks: list[Callable[[str], tuple[str, str]]] | None = None,
        date: str | None = None,
        snapshot: bool = False,
        encoding: str = "base64",
    ) -> str:
        """
        Commits new content as the latest version.

        Process:
        1. Parse current HEAD (which is Full Text).
        2. Compute Reverse Delta (New -> Old HEAD).
        3. Overwrite the on-disk HEAD block with this Delta.
        4. Append the New Content as the new HEAD block (Full Text).

        Args:
            signer_callbacks: List of functions for v2 signing.
                              Each accepts a message (str) and returns (signer_id, signature_value).
            snapshot: If True, the previous HEAD (Old Block) is saved as Full Text
                      instead of being converted to a delta. This creates an
                      intermediate snapshot for faster retrieval.
            encoding: Encoding to use for binary data ('base64' or 'base85').

        Returns:
            The new version string ("1.0", "1.1", ...), for every stream type.
            Use get_content() / stream.getvalue() to read back the raw stream.
        """
        # BinaryIO → bytes: HEAD block requires full snapshot; one read is unavoidable.
        if hasattr(content, "read"):
            content.seek(0)
            content = content.read()

        self._load_head()  # Refresh HEAD info by scanning the stream
        now = date if date else datetime.now().isoformat()

        # Enforce EOL policy for consistent hashing (only for text)
        if isinstance(content, str) and content and not content.endswith("\n"):
            content += "\n"

        # --- First Commit Case ---
        if not self.head_info:
            new_ver = "1.0"
            block_data = {"ver": new_ver, "date": now, "author": author, "log": log, "text": content}

            # Set binary flag for first commit
            if isinstance(content, bytes):
                block_data["is_binary"] = True

            # v2 Logic: Hash & Sign
            curr_hash = None
            signatures = []
            if self._version >= 2:
                # Calculate Hash (prev_hash is empty for first block)
                curr_hash = codec.calculate_block_hash(
                    block_data, prev_hash="", hash_algo=self._hash_algo, encoding=self.encoding
                )

                # Sign
                if signer_callbacks:
                    for callback in signer_callbacks:
                        # Message to sign: Timestamp|Hash
                        # This binds the signature to the specific content and time
                        sig_ts = datetime.now().isoformat()
                        msg_to_sign = f"{sig_ts}|{curr_hash}"
                        signer_id, sig_val = callback(msg_to_sign)
                        signatures.append(f"{signer_id}|{sig_ts}|{sig_val}")

            self.stream.seek(0, os.SEEK_END)  # Append to end
            append_pos = self.stream.tell()
            if not self._is_block_boundary(append_pos):
                # The block we're about to write must satisfy the same
                # boundary invariant _is_block_boundary requires when
                # scanning for it later (immediately preceded by a blank
                # line, or at self._data_start) -- otherwise it would become
                # permanently unfindable to a future scan: checkout()/log()
                # on a stream reloaded from these exact bytes would silently
                # "lose" the very commit being made right now (see
                # test_commit_after_non_block_trailing_bytes_stays_findable).
                # Only reachable here (head_info is None, i.e. "first
                # commit") when the stream already has non-block trailing
                # bytes despite that -- external corruption, a truncated
                # prior write, or a caller-supplied stream not entirely
                # written by SimpleRCS.
                self.stream.write(b"\n\n")
            self.stream.write(
                self._format_block(
                    block_data, current_hash=curr_hash, signatures=signatures, is_delta=False, encoding=encoding
                )
            )

            return new_ver

        # --- Subsequent Commit Case ---
        head = self.head_info
        # Use .get() to avoid KeyError if 'text' is missing (e.g. malformed block)
        # Default to empty bytes if it was marked as binary, else empty string.
        default_content = b"" if head.get("is_binary") else ""
        head_content = head.get("text", default_content)

        # 1. Prepare Old Block (Vn)
        head_block_data = head.copy()

        is_binary_content = isinstance(content, bytes)
        is_binary_head = head.get("is_binary", False)  # Default to False if missing

        is_type_change = is_binary_content != is_binary_head

        # Decide whether to save as Delta or Full Text Snapshot
        is_delta_block = True

        if snapshot or is_type_change:
            # Snapshot mode: Keep Old Block as Full Text
            is_delta_block = False
            # text is already set to full text in head_block_data
            # Ensure is_binary flag is correct for the old block
            head_block_data["is_binary"] = is_binary_head
        else:
            # Standard mode: Compute Reverse Delta: New Content (Vn+1) -> Old Head Content (Vn)
            # Ensure types match before calling delta generation
            if is_binary_content:
                # Both are binary (checked by is_type_change)
                delta = self._generate_reverse_delta(content, head_content, encoding=encoding)
            else:
                # Both are text
                delta = self._generate_reverse_delta(content, head_content)

            head_block_data["text"] = delta
            is_delta_block = True
            # Delta block inherits the binary nature of the data it represents
            head_block_data["is_binary"] = is_binary_head

        # Cleanup internal metadata
        if "start" in head_block_data:
            del head_block_data["start"]
        if "end" in head_block_data:
            del head_block_data["end"]
        # In v2, we must preserve 'hash', 'prev_hash', 'signatures' from the original head block.
        # They should already be in head_block_data if _parse_block_content_no_regex loaded them.
        old_prev_hash = head_block_data.get("prev_hash")
        old_curr_hash = head_block_data.get("hash")
        old_signatures = head_block_data.get("signatures")

        # The new HEAD block (Vn+1) is Full Text.
        # Increment version: 1.9 -> 1.10 (RCS style), not 2.0 (Float style)
        last_ver_str = head.get("ver", "0.0")
        try:
            parts = [int(p) for p in last_ver_str.split(".")]
            parts[-1] += 1
            new_ver = ".".join(map(str, parts))
        except ValueError as e:
            # Do NOT fall back to a fixed "1.0" here: HEAD's version string is
            # unparseable, meaning the file is already corrupted. Silently
            # coining "1.0" could collide with an existing version elsewhere
            # in the history, making later checkout("1.0") ambiguous.
            raise SimpleRCSCorruptionError(
                f"Cannot commit: HEAD version string '{last_ver_str}' is malformed"
            ) from e

        new_block_data = {"ver": new_ver, "date": now, "author": author, "log": log, "text": content}
        if is_binary_content:
            new_block_data["is_binary"] = True

        new_curr_hash = None
        new_prev_hash = None
        new_signatures = []

        if self._version >= 2:
            if old_curr_hash:
                new_prev_hash = old_curr_hash
            else:
                # Fallback: if old block didn't have hash (maybe corrupted v2?), compute it now based on OLD FULL TEXT
                new_prev_hash = codec.calculate_block_hash(
                    head, prev_hash=old_prev_hash, hash_algo=self._hash_algo, encoding=self.encoding
                )

            # Calculate New Block Hash
            new_curr_hash = codec.calculate_block_hash(
                new_block_data, prev_hash=new_prev_hash, hash_algo=self._hash_algo, encoding=self.encoding
            )

            # Sign New Block
            if signer_callbacks:
                for callback in signer_callbacks:
                    sig_ts = datetime.now().isoformat()
                    msg_to_sign = f"{sig_ts}|{new_curr_hash}"
                    signer_id, sig_val = callback(msg_to_sign)
                    new_signatures.append(f"{signer_id}|{sig_ts}|{sig_val}")

        # 3. Write to stream

        # Overwrite Old Block
        old_block_bytes = self._format_block(
            head_block_data,
            current_hash=old_curr_hash,
            prev_hash=old_prev_hash,
            signatures=old_signatures,
            is_delta=is_delta_block,
            encoding=encoding,
        )

        # Append New Block (Full Text)
        new_block_bytes = self._format_block(
            new_block_data,
            current_hash=new_curr_hash,
            prev_hash=new_prev_hash,
            signatures=new_signatures,
            is_delta=False,  # HEAD is always Full Text
            encoding=encoding,
        )

        # Single write() call: two separate writes would leave the file with an
        # overwritten old HEAD but no new HEAD if the process dies in between.
        self._rewrite_head(old_block_bytes + new_block_bytes)

        return new_ver

    def checkout(self, ver_num: str = None) -> str | bytes:
        """
        Retrieves the content of a specific version.

        Process:
        1. Start with the latest HEAD (Full Text).
        2. Traverse backwards through history, reading preceding blocks.
        3. For each block, apply its stored Reverse Delta to the current text.
        4. Stop when the target version is reached.
        """
        self._load_head()  # Ensure HEAD info is up-to-date

        default_content = b"" if self.head_info and self.head_info.get("is_binary") else ""

        if not self.head_info:
            return default_content  # Handle initial empty state

        if ver_num is None or ver_num == self.head_info["ver"]:
            return self.head_info.get("text", default_content)

        curr_content = self.head_info.get("text", default_content)  # Initialize curr_content safely
        curr_block = self.head_info

        # Iterate backwards, applying deltas
        while curr_block:
            # Find the block immediately preceding the current one
            prev_block = self._get_prev_block(curr_block["start"])
            if not prev_block:
                # Reached the first block in history without finding target
                raise ValueError(f"Version '{ver_num}' not found in history (reached start of file).")

            # Check if prev_block is a Snapshot (Full Text) or Delta
            if not prev_block.get("is_delta", True):  # Default to True (delta) if flag missing
                # It's a snapshot! We can jump directly to this content.
                curr_content = prev_block["text"]
            else:
                # It's a delta. Apply it.
                # prev_block contains the delta to transform curr_text to prev_text
                # (Strictly speaking, V_prev contains delta to go from V_curr to V_prev)
                delta = prev_block["text"]
                try:
                    curr_content = self._apply_reverse_delta(curr_content, delta)
                except ValueError as e:
                    raise SimpleRCSCorruptionError(
                        f"Cannot reconstruct version '{prev_block.get('ver')}': "
                        f"delta data is corrupted ({e})"
                    ) from e

            # Check if this is our target version
            if prev_block["ver"] == ver_num:
                return curr_content

            curr_block = prev_block  # Move to the previous block

        return ""  # Should not be reached if target_idx was found

    def log(self, limit: int | None = None, reverse: bool = False) -> list[dict]:
        """
        Retrieves the commit history.

        Args:
            limit: Maximum number of log entries to return.
            reverse: If True, returns history in chronological order (oldest first).
                     Default is False (newest first).
        """
        self._load_head(metadata_only=True)
        if not self.head_info:
            return []

        history = []
        curr_block = self.head_info

        while curr_block:
            # Extract metadata
            meta = {
                "ver": curr_block.get("ver"),
                "date": curr_block.get("date"),
                "author": curr_block.get("author"),
                "log": curr_block.get("log"),
            }
            # Add v2 fields if present
            if self._version >= 2:
                meta["hash"] = curr_block.get("hash")
                meta["prev_hash"] = curr_block.get("prev_hash")
                meta["signatures"] = curr_block.get("signatures", [])  # List of signature strings
                meta["is_binary"] = curr_block.get("is_binary", False)

            history.append(meta)

            if limit and len(history) >= limit:
                break

            prev_block = self._get_prev_block(curr_block["start"], metadata_only=True)
            if not prev_block:
                break
            curr_block = prev_block

        if reverse:
            return history[::-1]
        return history

    def diff(self, ver_a: str, ver_b: str) -> str:
        """
        Generates a unified diff between two versions.

        Args:
            ver_a: The version number to compare from (source).
            ver_b: The version number to compare to (target).

        Returns:
            A string containing the unified diff.
        """
        content_a = self.checkout(ver_a)
        content_b = self.checkout(ver_b)

        if content_a is None or content_b is None:
            raise ValueError("One or both versions could not be found.")

        if isinstance(content_a, bytes) or isinstance(content_b, bytes):
            return "Binary files differ"

        lines_a = content_a.splitlines(keepends=True)
        lines_b = content_b.splitlines(keepends=True)

        diff_lines = difflib.unified_diff(
            lines_a,
            lines_b,
            fromfile=f"Version {ver_a}\n",
            tofile=f"Version {ver_b}\n",
            lineterm="",
        )

        return "".join(diff_lines)

    def blame(self, depth: int | None = None) -> list[dict]:  # noqa: C901
        """
        Annotates each line of the HEAD version with the revision that last modified it.

        Args:
            depth: If provided, limits the backward traversal to this many versions.
                   Lines older than this depth will be blamed on the oldest reached version.

        Returns:
            A list of dicts, where each dict corresponds to a line in HEAD and contains:
            {
                'line': str (content),
                'ver': str (version),
                'author': str,
                'date': str
            }
        """
        self._load_head()
        if not self.head_info:
            return []

        head_text = self.head_info["text"]
        if isinstance(head_text, bytes):
            return []

        head_lines = head_text.splitlines(keepends=True)

        # 1. Initialize tracker
        # Each item: {'head_index': int|None, 'blame': dict}
        # head_index maps to the index in the final output. None means it's a ghost line.
        current_commit = {
            "ver": self.head_info["ver"],
            "author": self.head_info["author"],
            "date": self.head_info["date"],
        }

        tracker = []
        for i in range(len(head_lines)):
            tracker.append(
                {
                    "head_index": i,
                    "blame": current_commit,
                }
            )

        final_blame = [None] * len(head_lines)

        curr_block = self.head_info
        curr_depth = 0

        # 2. Traverse backwards
        while curr_block:
            # Check depth limit
            if depth is not None and curr_depth >= depth:
                # Reached depth limit.
                # Blame remaining non-finalized lines on the current block (oldest reached).
                reached_commit = {
                    "ver": curr_block["ver"],
                    "author": curr_block["author"],
                    "date": curr_block["date"],
                }
                for item in tracker:
                    if item["head_index"] is not None:
                        if final_blame[item["head_index"]] is None:
                            final_blame[item["head_index"]] = reached_commit
                break

            prev_block = self._get_prev_block(curr_block["start"])

            if not prev_block:
                # Reached start (Ver 1.0).
                # All remaining non-ghost lines in tracker originate here.
                first_commit = {
                    "ver": curr_block["ver"],
                    "author": curr_block["author"],
                    "date": curr_block["date"],
                }
                for item in tracker:
                    if item["head_index"] is not None:
                        # If not already finalized (shouldn't happen if logic is correct, but for safety)
                        if final_blame[item["head_index"]] is None:
                            final_blame[item["head_index"]] = first_commit
                break

            prev_commit = {
                "ver": prev_block["ver"],
                "author": prev_block["author"],
                "date": prev_block["date"],
            }

            # Parse Delta (Current -> Prev)
            delta = prev_block["text"]

            # Stop blame if we hit binary data or binary delta
            is_binary_delta = (
                isinstance(delta, bytes) and (b";base64," in delta or b";base85," in delta or b";raw," in delta)
            ) or (isinstance(delta, str) and (";base64," in delta or ";base85," in delta))
            if prev_block.get("is_binary", False) or (prev_block.get("is_delta", False) and is_binary_delta):
                for item in tracker:
                    if item["head_index"] is not None:
                        if final_blame[item["head_index"]] is None:
                            final_blame[item["head_index"]] = prev_commit
                break

            script_lines = delta.splitlines()

            commands = []
            i = 0
            while i < len(script_lines):
                header = script_lines[i]
                i += 1
                parts = header.split()
                if not parts:
                    continue
                cmd = parts[0][0]
                if cmd not in ("d", "a"):
                    continue
                try:
                    start = int(parts[0][1:])
                    count = int(parts[1])
                except (ValueError, IndexError) as e:
                    raise ValueError(f"Invalid delta format: {e}") from e

                if cmd == "a":
                    for _ in range(count):
                        if i < len(script_lines):
                            i += 1  # Skip payload lines

                commands.append({"cmd": cmd, "start": start, "count": count})

            # Sort descending to handle list mutations.
            # Tie-break: 'a' before 'd' at the same line (see _apply_reverse_delta).
            commands.sort(key=lambda x: (-x["start"], 0 if x["cmd"] == "a" else 1))

            for c in commands:
                idx = c["start"]
                if c["cmd"] == "d":
                    # Delete lines starting at idx (1-based) -> list index idx-1
                    # These lines were born in Current.
                    list_idx = idx - 1

                    # These items are being removed from history.
                    # Their journey ends here. Finalize their blame.
                    removed_items = tracker[list_idx : list_idx + c["count"]]
                    for item in removed_items:
                        if item["head_index"] is not None:
                            final_blame[item["head_index"]] = item["blame"]

                    del tracker[list_idx : list_idx + c["count"]]

                elif c["cmd"] == "a":
                    # Add lines to Prev.
                    # These lines don't exist in Current, so they are ghosts.
                    insert_idx = idx
                    ghost_items = [{"head_index": None, "blame": prev_commit} for _ in range(c["count"])]
                    tracker[insert_idx:insert_idx] = ghost_items

            # Update blame for surviving items to Previous
            for item in tracker:
                item["blame"] = prev_commit

            curr_block = prev_block
            curr_depth += 1

        # 3. Construct Result
        result = []
        for i, line in enumerate(head_lines):
            info = final_blame[i]
            if info is None:
                # Should not happen, but fallback
                info = current_commit

            result.append(
                {
                    "line": line.rstrip("\n"),
                    "ver": info["ver"],
                    "author": info["author"],
                    "date": info["date"],
                }
            )

        return result

    def sign_head(self, signer_callbacks: list[Callable[[str], tuple[str, str]]]) -> bool:  # noqa: C901
        """
        Adds signatures to the current HEAD block.
        This is possible because signatures are not part of the hash calculation.
        HEAD is at the end of the file, so we can overwrite it easily.

        Args:
            signer_callbacks: List of functions for v2 signing.
                              Each accepts a message (str) and returns (signer_id, signature_value).

        Returns:
            True if signing was successful, False otherwise.
        """
        if self._version < 2:
            logger.error("Error: Signing is only supported for SimpleRCS v2 or higher.")
            return False  # Not supported in v1

        self._load_head()
        if not self.head_info:
            logger.error("Error: No HEAD block found to sign.")
            return False

        # Retrieve necessary data from HEAD for hash calculation
        head_data_for_hash = {
            "ver": self.head_info.get("ver"),
            "date": self.head_info.get("date"),
            "author": self.head_info.get("author"),
            "log": self.head_info.get("log"),
            "text": self.head_info.get("text"),  # Full Text
        }
        stored_prev_hash = self.head_info.get("prev_hash")
        stored_hash = self.head_info.get("hash")

        # Re-calculate hash to confirm integrity before signing
        calculated_hash = codec.calculate_block_hash(
            head_data_for_hash, prev_hash=stored_prev_hash, hash_algo=self._hash_algo, encoding=self.encoding
        )

        if calculated_hash != stored_hash:
            logger.error(
                f"Error: HEAD hash mismatch. Stored: {stored_hash},"
                f"Calculated: {calculated_hash}. Cannot sign corrupted block."
            )
            return False

        # Generate new signatures
        new_signatures = []
        if signer_callbacks:
            for callback in signer_callbacks:
                sig_ts = datetime.now().isoformat()
                msg_to_sign = f"{sig_ts}|{calculated_hash}"
                try:
                    signer_id, sig_val = callback(msg_to_sign)
                    new_signatures.append(f"{signer_id}|{sig_ts}|{sig_val}")
                except Exception as e:
                    logger.warning(f"Warning: Signing callback failed for message '{msg_to_sign}': {e}")
                    return False  # Fail if any callback fails

        # Merge with existing signatures (Deduplicate by signer_id)
        existing_signatures = self.head_info.get("signatures", [])

        # Use a dict to map signer_id -> signature_entry to ensure uniqueness
        # Latest signature (from new_signatures) overwrites existing ones
        sig_map = {}

        # 1. Load existing signatures
        for sig in existing_signatures:
            parts = sig.split("|")
            if len(parts) >= 1:
                signer_id = parts[0]
                sig_map[signer_id] = sig

        # 2. Apply new signatures (overwrite if exists)

        for sig in new_signatures:
            parts = sig.split("|")
            if len(parts) >= 1:
                signer_id = parts[0]
                sig_map[signer_id] = sig

        all_signatures = list(sig_map.values())

        # Prepare block data for rewriting HEAD
        # We need to use the original data (with Full Text) but replace its 'text' with delta if it were historical.
        # But for HEAD, its 'text' field in head_info IS the Full Text.
        block_data_for_rewrite = self.head_info.copy()
        # Remove internal metadata before formatting
        if "start" in block_data_for_rewrite:
            del block_data_for_rewrite["start"]
        if "end" in block_data_for_rewrite:
            del block_data_for_rewrite["end"]

        # Rewrite HEAD block at its original start position
        block_bytes = self._format_block(
            block_data_for_rewrite,
            current_hash=stored_hash,
            prev_hash=stored_prev_hash,
            signatures=all_signatures,
        )

        self._rewrite_head(block_bytes)

        # Refresh head_info so instance state reflects new signatures
        self._load_head()

        return True

    def verify_block_signature(
        self,
        block_data: dict,
        verifier_callback: Callable[[str, str, str], bool],
    ) -> tuple[bool, str | None]:
        """
        Verifies signatures of a single block data dict.

        Args:
            block_data: The block data dictionary (from log or internal storage).
            verifier_callback: Function to verify (signer_id, message, signature).

        Returns:
            (True, signer_id) if a valid signature is found.
            (False, None) otherwise.
        """
        signatures = block_data.get("signatures", [])
        stored_hash = block_data.get("hash")

        if not signatures or not stored_hash:
            return False, None

        for sig_entry in signatures:
            signer_id = None
            try:
                parts = sig_entry.split("|")
                if len(parts) < 3:
                    continue
                signer_id = parts[0]
                timestamp = parts[1]
                sig_val = "|".join(parts[2:])  # Handle potential | in signature

                # Reconstruct message used for signing
                msg = f"{timestamp}|{stored_hash}"

                if verifier_callback(signer_id, msg, sig_val):
                    return True, signer_id
            except Exception as e:
                logger.warning(f"Signature verification failed for entry {sig_entry!r}: {e}")
                return False, signer_id

        return False, None

    def verify(self, verifier_callbacks: list[Callable[[str, str, str], bool]] | None = None) -> bool:  # noqa: C901
        """
        Verifies the integrity of the hash chain and signatures (v2 only).

        Args:
            verifier_callbacks: List of functions to verify signatures.
                                Each callable receives (signer_id, message, signature_value)
                                and returns True if valid.

        Returns:
            True if integrity is intact, False otherwise.
        """
        if self._version < 2:
            return True  # v1 has no integrity features

        self._load_head()
        if not self.head_info:
            return True  # Empty file is valid

        curr_block = self.head_info

        # To verify hashes, we need the Full Text of each version.
        # Since we traverse backwards (HEAD -> V1), and blocks store Reverse Deltas,
        # we start with HEAD (Full Text) and apply deltas to get previous versions.
        # This matches the 'checkout' traversal logic perfectly.

        if "text" not in curr_block:
            # A block whose content field never finished writing parses into
            # metadata with no 'text' -- e.g. a commit() interrupted partway
            # through the tail rewrite (ENOSPC, SIGKILL, power loss). That is
            # exactly the damage verify() exists to report, so report it
            # instead of raising: callers rely on False, not an exception
            # (see the class docstring and AGENTS.md).
            logger.error("HEAD block has no content field (truncated write?)")
            return False

        curr_text = curr_block["text"]  # HEAD is Full Text

        # We need to track the 'expected hash' for the *next* block's prev_hash check
        # But we are going backwards.
        # Current Block's 'prev_hash' must match Previous Block's Hash.

        while curr_block:
            # 1. Verify Current Block's Hash
            # Hash is based on: Metadata + Full Text + prev_hash
            stored_hash = curr_block.get("hash")
            stored_prev_hash = curr_block.get("prev_hash")

            if not stored_hash:
                # v2 block MUST have a hash
                return False

            # Reconstruct data dict for hashing (must match commit logic)
            # We use the 'curr_text' which is the Full Text of this version.
            block_data_for_hash = curr_block.copy()
            block_data_for_hash["text"] = curr_text
            # Ensure is_binary flag matches what was committed
            # Ideally, curr_block has it. If not (old ver), default false.
            # But wait, curr_block is the metadata block. We should use its flag.
            # 'curr_text' is the Reconstructed Full Text. Its type should match.

            calculated_hash = codec.calculate_block_hash(
                block_data_for_hash, prev_hash=stored_prev_hash, hash_algo=self._hash_algo, encoding=self.encoding
            )

            if calculated_hash != stored_hash:
                logger.error(f"calc hash = {calculated_hash} ,stored hash = {stored_hash}")
                logger.error(f"Hash mismatch at version {curr_block.get('ver')}")
                return False
            else:
                logger.info(f"Verify Pass: Ver {curr_block.get('ver')} Hash {stored_hash}")

            # 2. Verify Signatures (if callbacks provided)
            signatures = curr_block.get("signatures", [])
            if verifier_callbacks and signatures:
                # Signatures format: signer_id|timestamp|sig_val
                # Message signed: timestamp|hash
                for sig_entry in signatures:
                    try:
                        parts = sig_entry.split("|")
                        if len(parts) < 3:
                            continue
                        signer_id = parts[0]
                        timestamp = parts[1]
                        sig_val = "|".join(parts[2:])  # In case sig_val has |

                        msg = f"{timestamp}|{stored_hash}"

                        # Check if ANY verifier accepts this signature
                        valid_sig = False
                        for verifier in verifier_callbacks:
                            if verifier(signer_id, msg, sig_val):
                                valid_sig = True
                                break

                        if not valid_sig:
                            logger.error(f"Invalid signature at version {curr_block.get('ver')} by {signer_id}")
                            return False
                    except Exception as e:
                        logger.error(
                            f"Signature verification error at version {curr_block.get('ver')}: {e}"
                        )
                        return False

            # 3. Move to Previous Block
            prev_block = self._get_prev_block(curr_block["start"])

            if prev_block:
                # Verify Chain Link: Current.prev_hash == Hash(Previous Block)
                # But we haven't calculated Previous Block's hash yet?
                # We can't verify 'stored_prev_hash' until we process 'prev_block'.
                # Wait. 'stored_prev_hash' MUST equal the hash we *will* calculate for prev_block.
                # So we pass 'stored_prev_hash' down to the next iteration?
                # Let's verify it in the next iteration:
                # "Next iteration's calculated hash must equal This iteration's stored_prev_hash"

                # Apply delta to get Full Text of Previous Version (or use Snapshot)
                # MODIFIED: Handle snapshots during verification
                if "text" not in prev_block:
                    # Same truncation case as HEAD above, one block further back.
                    logger.error(
                        f"Block {prev_block.get('ver')} has no content field (truncated write?)"
                    )
                    return False
                if not prev_block.get("is_delta", True):
                    curr_text = prev_block["text"]  # Snapshot: Reset text
                else:
                    delta = prev_block["text"]
                    try:
                        curr_text = self._apply_reverse_delta(curr_text, delta)
                    except ValueError as e:
                        logger.error(
                            f"Corrupted delta data while reconstructing version "
                            f"{prev_block.get('ver')}: {e}"
                        )
                        return False

                # We need to temporarily peek/calculate prev_block's hash to verify the link NOW?
                # Or just verify it when we become 'prev_block'.
                # If we verify it *when we process prev_block*, we check:
                #   calc_hash(prev_block) == prev_block.stored_hash
                # AND
                #   prev_block.stored_hash == curr_block.stored_prev_hash

                # So we just need to check:
                # curr_block['prev_hash'] == prev_block['hash']
                if curr_block.get("prev_hash") != prev_block.get("hash"):
                    logger.error(f"Chain broken between {curr_block.get('ver')} and {prev_block.get('ver')}")
                    return False

            curr_block = prev_block

        return True
