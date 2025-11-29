import difflib
import hashlib
import io
import logging
import os
import re
from collections.abc import Callable
from datetime import datetime
from typing import BinaryIO


logger = logging.getLogger(__name__)

class SimpleRCS:
    """
    A simple, robust, and efficient version control system inspired by RCS.
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
        -   Supports v2 format with SHA-256 hash chains.
        -   Each block contains a `hash` of its content (Full Text + Metadata) and the `prev_hash`.
        -   This ensures tamper-evidence. Changing any past version breaks the chain.
    """

    MAGIC_HEADER_V2 = b"# SimpleRCS v2.0; hash_algo=sha256;\n"

    def __init__(self, content_or_path: str | bytes | BinaryIO | None = None) -> None:
        """
        Initializes the SimpleRCS instance.

        Args:
            content_or_path:
                - None: Creates a new empty in-memory RCS (BytesIO).
                - str (path): Opens the file at the given path. Creates it if not exists.
                - str (content): Treats the string as RCS content (wraps in BytesIO).
                - bytes: Treats bytes as RCS content (wraps in BytesIO).
                - file-like object: Uses the provided binary stream directly.
        """
        self.file_path: str | None = None
        self.stream: BinaryIO
        self.owns_handle = False # Flag to indicate if we opened the file handle and should close it
        self._version = 1 # Default to v1

        if content_or_path is None:
            # New empty in-memory RCS -> v2
            self.stream = io.BytesIO()
            self.stream.write(self.MAGIC_HEADER_V2)
            self._version = 2
            self.owns_handle = True
        elif isinstance(content_or_path, str):
            # Check if it's a file path heuristic
            if os.path.exists(content_or_path) or (
                    not content_or_path.startswith(('@', 'ver', '#')) and '\n' not in content_or_path):
                # Assume file path
                mode = "rb+" if os.path.exists(content_or_path) else "wb+"
                self.stream = open(content_or_path, mode)
                self.file_path = content_or_path
                self.owns_handle = True

                # Check version for existing file, or init new file
                if mode == "wb+":
                    self.stream.write(self.MAGIC_HEADER_V2)
                    self._version = 2
                else:
                    self._version = self._detect_format_version()
            else:
                # Treat as raw string content -> In-memory stream
                self.stream = io.BytesIO(content_or_path.encode('utf-8'))
                self._version = self._detect_format_version()
                self.owns_handle = True
        elif isinstance(content_or_path, bytes):
            self.stream = io.BytesIO(content_or_path)
            self._version = self._detect_format_version()
            self.owns_handle = True
        elif hasattr(content_or_path, 'read') and hasattr(content_or_path, 'seek'):
            # External file-like object
            self.stream = content_or_path
            self._version = self._detect_format_version()
            self.owns_handle = False
        else:
            raise ValueError("Invalid input type. Expected file path, string, bytes, or file-like object.")

        self.head_info: dict | None = None
        self._load_head()

    def _detect_format_version(self) -> int:
        """Detects the format version from the stream header."""
        pos = self.stream.tell()
        self.stream.seek(0)
        header = self.stream.readline()
        self.stream.seek(pos) # Restore position

        if header.startswith(b"# SimpleRCS v2.0;"):
            return 2
        return 1

    def _calculate_block_hash(self, data: dict, prev_hash: str | None = None) -> str:
        """
        Calculates SHA-256 hash of the block content.
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
        text = str(data.get("text", "")) # This MUST be Full Text

        # Enforce EOL policy: Text must end with \n for consistent hashing
        if text and not text.endswith('\n'):
            text += '\n'

        p_hash = prev_hash if prev_hash else ""

        payload = f"{ver}|{date}|{author}|{log}|{text}|{p_hash}"
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()


    def __del__(self) -> None:
        """Closes the stream if this instance owns it."""
        if self.owns_handle and hasattr(self, 'stream') and not self.stream.closed:
            self.stream.close()

    def get_content(self) -> str:
        """
        Returns the full content of the RCS stream as a string.
        Useful when working with in-memory streams to get the final result.
        """
        pos = self.stream.tell()
        self.stream.seek(0)
        content = self.stream.read().decode('utf-8', errors='replace')
        self.stream.seek(pos)
        return content

    def _escape(self, text: str) -> str:
        """Escapes '@' to '@@' for storage within @...@ blocks."""
        return text.replace("@", "@@")

    def _unescape(self, text: str) -> str:
        """Unescapes '@@' back to '@'."""
        return text.replace("@@", "@")

    def _parse_block_content(self, content_bytes: bytes) -> dict:
        """
        Parses a raw block bytes into a dictionary.
        Format: key @value@; ...
        """
        content_str = content_bytes.decode('utf-8', errors='replace')
        data = {}
        # Regex matches keys (ver, date, etc.) and values enclosed in @...@
        # Use re.DOTALL to match newlines within @...@
        pattern = re.compile(r'(ver|version|date|author|log|text)\s+@((?:[^@]|@@)*)@;', re.DOTALL)

        # Iterate over all matches to build the data dictionary
        for match in pattern.finditer(content_str):
            key = match.group(1)
            value = self._unescape(match.group(2))
            if key == 'version':
                key = 'ver' # Normalize key
            data[key] = value

        # Basic validation to ensure it looks like a valid block
        if 'ver' in data:
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
        _keywords = [b'ver', b'version', b'date', b'author', b'log', b'text', b'prev_hash', b'hash', b'signature']

        content = content_bytes
        length = len(content)
        pos = 0
        data = {}

        while pos < length:
            # Skip whitespace
            while pos < length and content[pos] in b' \t\r\n':
                pos += 1
            if pos >= length:
                break

            # Read Key
            key_start = pos
            while pos < length and content[pos] not in b' @;':
                pos += 1
            key_bytes = content[key_start:pos]

            # Ensure the key is a valid keyword, otherwise break (malformed block)
            if key_bytes not in _keywords:
                break

            # Skip whitespace after key
            while pos < length and content[pos] in b' \t\r\n':
                pos += 1

            # Expect '@' for value start
            if pos >= length or content[pos] != ord('@'):
                # Malformed or unexpected char where '@' was expected
                break

            pos += 1 # Skip opening '@'

            # Read Value, handling '@@' escaping
            val_parts = []
            while pos < length:
                # Fast search for next '@'
                end = content.find(b'@', pos)
                if end == -1:
                    # Unterminated string, malformed block
                    break

                # Check for double '@@' (escaped '@')
                if end + 1 < length and content[end+1] == ord('@'):
                    val_parts.append(content[pos:end])
                    val_parts.append(b'@') # Unescape @@ -> @
                    pos = end + 2
                else:
                    val_parts.append(content[pos:end])
                    pos = end + 1 # Skip closing '@'
                    break

            value_bytes = b"".join(val_parts)
            value = value_bytes.decode('utf-8', errors='replace')

            # Skip whitespace after value
            while pos < length and content[pos] in b' \t\r\n':
                pos += 1

            # Expect ';' after value
            if pos < length and content[pos] == ord(';'):
                pos += 1
            else:
                # Malformed, missing ';'
                break

            # Store data
            key_str = key_bytes.decode('utf-8')
            if key_str == 'version':
                key_str = 'ver' # Normalize key

            if key_str == 'signature':
                # Handle multiple signatures
                if 'signatures' not in data:
                    data['signatures'] = []
                data['signatures'].append(value)
            else:
                data[key_str] = value

        # Basic validation to ensure it looks like a valid block
        if 'ver' in data:
            return data
        return {}

    def _load_head(self) -> None:
        """
        Locates and loads ONLY the last block (HEAD) by scanning backwards from EOF.
        This is a performance optimization to avoid reading the entire history when
        we only need the latest version. Sets self.head_info = { ..., 'start': offset, 'end': offset }.
        """
        self.head_info = None

        # Move to End of File
        self.stream.seek(0, os.SEEK_END)
        file_size = self.stream.tell()
        if file_size == 0:
            return

        chunk_size = 4096
        pos = file_size
        buffer = b""

        # Scan backwards, reading chunks
        while pos > 0:
            read_size = min(chunk_size, pos)
            pos -= read_size
            self.stream.seek(pos)
            chunk = self.stream.read(read_size)
            buffer = chunk + buffer # Prepend to buffer

            # Find the LAST match of the block start pattern (ver @...) in the buffer
            # We assume keywords are ASCII.
            idx_v = buffer.rfind(b"ver @")
            idx_version = buffer.rfind(b"version @") # For backward compatibility in parsing
            idx = max(idx_v, idx_version)

            if idx != -1:
                abs_start = pos + idx # Absolute offset in the stream

                # Read and parse the block from this offset to EOF
                self.stream.seek(abs_start)
                block_bytes = self.stream.read() # Read until EOF (HEAD is always at the end)

                parsed = self._parse_block_content_no_regex(block_bytes)
                if parsed:
                    parsed['start'] = abs_start
                    parsed['end'] = file_size
                    self.head_info = parsed
                    return

            if len(buffer) > 10 * 1024 * 1024: # Safety limit (10MB) to prevent excessive buffering
                break

    def _get_prev_block(self, current_start_offset: int) -> dict | None:
        """
        Finds and parses the block immediately preceding the given offset.
        Used for traversing history backwards (HEAD -> V_prev -> ...).
        """
        if current_start_offset <= 0:
            return None

        chunk_size = 4096
        scan_pos = current_start_offset
        overlap = 100 # To catch keywords split across chunk boundaries

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

            # Search for block start (`ver @` or `version @`) in `chunk` up to `limit_in_chunk`
            idx_v = chunk.rfind(b"ver @", 0, limit_in_chunk)
            idx_version = chunk.rfind(b"version @", 0, limit_in_chunk)
            idx = max(idx_v, idx_version)

            if idx != -1:
                abs_start = scan_pos + idx # Absolute offset of previous block's start

                # Read block from start up to the next block's start
                self.stream.seek(abs_start)
                length = current_start_offset - abs_start
                block_bytes = self.stream.read(length)

                parsed = self._parse_block_content_no_regex(block_bytes)
                if parsed:
                    parsed['start'] = abs_start
                    parsed['end'] = current_start_offset
                    return parsed

        return None

    def _generate_reverse_delta(self, new_text: str, old_text: str) -> str:
        """
        Generates an RCS-style ('diff -n') Reverse Delta.
        Goal: Create instructions to transform 'new_text' (Source) INTO 'old_text' (Dest).

        Why New->Old?
        Because we store HEAD as Full Text. To get the previous version, we need
        to apply a patch to HEAD that turns it back into the previous version.
        """
        new_lines = new_text.splitlines(keepends=True)
        old_lines = old_text.splitlines(keepends=True)
        matcher = difflib.SequenceMatcher(None, new_lines, old_lines)
        output = []

        # opcodes: describes how to turn 'a' (New) into 'b' (Old)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                continue

            # RCS diff format logic (similar to PHP's DeltaDiffFormatter)
            xlen = i2 - i1
            ylen = j2 - j1
            xbeg = i1 + 1

            del_cmd = f"d{xbeg} {xlen}"
            # For add command index: append after the last line of the changed block
            add_idx = xbeg + xlen - 1
            add_cmd = f"a{add_idx} {ylen}"

            if xlen > 0:
                output.append(del_cmd)
                if ylen > 0:
                    output.append(add_cmd)
                    for line in old_lines[j1:j2]:
                        output.append(line.rstrip('\n'))
            else:
                # Insert-only operation: only add command
                output.append(add_cmd)
                for line in old_lines[j1:j2]:
                    output.append(line.rstrip('\n'))
        return "\n".join(output)

    def _apply_reverse_delta(self, current_text: str, delta_text: str) -> str:  # noqa: C901
        """
        Applies an RCS 'diff -n' script to 'current_text'.
        This transforms the current version (e.g., 1.2) back to the previous version (1.1).

        Strategy:
        1. Parse diff commands.
        2. Sort commands by line number in descending order (Bottom-Up).
        3. Apply commands to the 'lines' list.
        Sorting by descending line number ensures that insertions/deletions do not
        invalidate the indices of subsequent operations (which are higher up in the file).
        """
        lines = current_text.splitlines(keepends=True)
        commands = []

        script_lines = delta_text.splitlines()
        i = 0
        while i < len(script_lines):
            header = script_lines[i]
            i += 1
            parts = header.split() # Split for manual parsing
            if not parts:
                continue

            cmd_char = parts[0][0] # 'd' or 'a'
            if cmd_char not in ('d', 'a'):
                continue

            try:
                start = int(parts[0][1:]) # Line number from 'd1' or 'a1'
                count = int(parts[1])     # Count of lines
            except (ValueError, IndexError) as e:
                raise ValueError("Invalid delta format") from e

            payload = []
            if cmd_char == 'a':
                # Read payload lines for 'a' command
                for _ in range(count):
                    if i < len(script_lines):
                        payload.append(script_lines[i] + "\n") # Restore newline
                        i += 1
            commands.append({"cmd": cmd_char, "line": start, "count": count, "payload": payload})

        # Sort commands by line number in descending order
        commands.sort(key=lambda x: x["line"], reverse=True)

        for cmd in commands:
            idx = cmd["line"]
            if cmd["cmd"] == 'a':
                # Append payload AFTER line `idx`. (List insert at index `idx`)
                insert_pos = idx
                lines[insert_pos:insert_pos] = cmd["payload"]
            elif cmd["cmd"] == 'd':
                # Delete `count` lines starting AT line `idx` (List index `idx-1`)
                start_pos = idx - 1
                del lines[start_pos:start_pos + cmd["count"]]
        result = "".join(lines)
        # Enforce EOL policy
        if result and not result.endswith('\n'):
            result += '\n'
        return result

    def _format_block(
        self,
        data: dict,
        current_hash: str | None = None,
        prev_hash: str | None = None,
        signatures: list[str] | None = None,
    ) -> bytes:
        """
        Formats a block dictionary into bytes for writing to the stream.
        Supports v2 fields (hash, prev_hash, signatures).
        """
        keys = ["ver", "date", "author", "log", "text"]
        lines = []
        for key in keys:
            val = self._escape(str(data.get(key, "")))
            lines.append(f"{key} @{val}@;")

        # Add v2 fields if present
        if self._version >= 2:
            if prev_hash:
                lines.append(f"prev_hash @{self._escape(prev_hash)}@;")
            if current_hash:
                lines.append(f"hash @{self._escape(current_hash)}@;")

            if signatures:
                for sig in signatures:
                    lines.append(f"signature @{self._escape(sig)}@;")

        return ("\n".join(lines) + "\n\n").encode('utf-8')

    def commit(  # noqa: C901
        self,
        content: str,
        author: str = "unknown",
        log: str = "",
        signer_callbacks: list[Callable[[str], tuple[str, str]]] | None = None,
        date: str | None = None,
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
            data: Optional timestamp string.

        Returns:
            The new version number (e.g., "1.2") if working with a file path.
            The full RCS content string if working with an in-memory stream.
        """
        self._load_head() # Refresh HEAD info by scanning the stream
        now = date if date else datetime.now().isoformat()

        # Enforce EOL policy for consistent hashing
        if content and not content.endswith('\n'):
            content += '\n'

        # --- First Commit Case ---
        if not self.head_info:
            new_ver = "1.0"
            block_data = {"ver": new_ver, "date": now, "author": author, "log": log, "text": content}

            # v2 Logic: Hash & Sign
            curr_hash = None
            signatures = []
            if self._version >= 2:
                # Calculate Hash (prev_hash is empty for first block)
                curr_hash = self._calculate_block_hash(block_data, prev_hash="")

                # Sign
                if signer_callbacks:
                    for callback in signer_callbacks:
                        # Message to sign: Timestamp|Hash
                        # This binds the signature to the specific content and time
                        sig_ts = datetime.now().isoformat()
                        msg_to_sign = f"{sig_ts}|{curr_hash}"
                        signer_id, sig_val = callback(msg_to_sign)
                        signatures.append(f"{signer_id}|{sig_ts}|{sig_val}")

            self.stream.seek(0, os.SEEK_END) # Append to end
            self.stream.write(self._format_block(block_data, current_hash=curr_hash, signatures=signatures))

            if isinstance(self.stream, io.BytesIO):
                return self.get_content()
            return new_ver

        # --- Subsequent Commit Case ---
        head = self.head_info
        head_content = head["text"]

        # 1. Compute Reverse Delta: New Content (Vn+1) -> Old Head Content (Vn)
        delta = self._generate_reverse_delta(content, head_content)

        # 2. Prepare blocks for writing

        # [Old Block (Vn)]
        # Becomes a Delta block.
        # Ideally, we preserve its original hash and signatures because they represent the Logical Full Text.
        # self.head_info already contains 'hash' and 'signatures' if parsed correctly from v2 file.
        head_block_data = head.copy()
        head_block_data["text"] = delta

        # Cleanup internal metadata
        if "start" in head_block_data:
            del head_block_data["start"]
        if "end" in head_block_data:
            del head_block_data["end"]
        # In v2, we must preserve 'hash', 'prev_hash', 'signatures' from the original head block.
        # They should already be in head_block_data if _parse_block_content_no_regex loaded them.
        # We need to extract them to pass to _format_block explicit args, or update _format_block to look in data?
        # _format_block takes explicit args for these.
        # Let's extract them.
        old_prev_hash = head_block_data.get("prev_hash")
        old_curr_hash = head_block_data.get("hash")
        old_signatures = head_block_data.get("signatures")

        # Remove them from data dict so they don't get duplicated or mishandled by _format_block's generic loop
        # (Though _format_block only iterates specific keys 'ver'...'text', so extra keys in dict are ignored. Safe.)

        # The new HEAD block (Vn+1) is Full Text.
        # Increment version: 1.9 -> 1.10 (RCS style), not 2.0 (Float style)
        last_ver_str = head.get("ver", "0.0")
        try:
            parts = [int(p) for p in last_ver_str.split('.')]
            if parts:
                parts[-1] += 1
                new_ver = ".".join(map(str, parts))
            else:
                new_ver = "1.0"
        except ValueError:
            new_ver = "1.0" # Fallback if version is malformed

        new_block_data = {"ver": new_ver, "date": now, "author": author, "log": log, "text": content}

        new_curr_hash = None
        new_prev_hash = None
        new_signatures = []

        if self._version >= 2:
            # New block's prev_hash is the Old Block's hash (which represents its Full Text state)
            # If Old Block was v1, it might not have a hash. We might need to compute it now?
            # If we are migrating or mixing, this is tricky.
            # But our policy: "v1 file keeps v1". "v2 file has hashes".
            # So if self._version >= 2, Old Block MUST have a hash (unless it's the genesis block of a v2 file?).
            # Wait, if we just upgraded code but file is v2, head should have hash.

            if old_curr_hash:
                new_prev_hash = old_curr_hash
            else:
                # Fallback: if old block didn't have hash (maybe corrupted v2?), compute it now based on OLD FULL TEXT
                # We have 'head' (Full Text) in memory.
                # prev_hash for Old Block? We might need to look it up.
                # Simplified: If missing, treat as empty chain start or error?
                # Let's compute it.
                new_prev_hash = self._calculate_block_hash(head, prev_hash=old_prev_hash)

            # Calculate New Block Hash
            new_curr_hash = self._calculate_block_hash(new_block_data, prev_hash=new_prev_hash)

            # Sign New Block
            if signer_callbacks:
                for callback in signer_callbacks:
                    sig_ts = datetime.now().isoformat()
                    msg_to_sign = f"{sig_ts}|{new_curr_hash}"
                    signer_id, sig_val = callback(msg_to_sign)
                    new_signatures.append(f"{signer_id}|{sig_ts}|{sig_val}")

        # 3. Write to stream

        # Overwrite Old Block (as Delta)
        # We pass the ORIGINAL hash/sig info to preserve it.
        # Even though text is now Delta, the Hash represents the Full Text version of this block.
        old_block_bytes = self._format_block(
            head_block_data,
            current_hash=old_curr_hash,
            prev_hash=old_prev_hash,
            signatures=old_signatures,
        )

        # Append New Block (Full Text)
        new_block_bytes = self._format_block(
            new_block_data,
            current_hash=new_curr_hash,
            prev_hash=new_prev_hash,
            signatures=new_signatures,
        )

        self.stream.seek(self.head_info['start'])
        self.stream.write(old_block_bytes)
        self.stream.write(new_block_bytes)
        self.stream.truncate() # Crucial: remove any leftover

        if isinstance(self.stream, io.BytesIO):
            return self.get_content()
        return new_ver

    def checkout(self, ver_num: str = None) -> str:
        """
        Retrieves the content of a specific version.

        Process:
        1. Start with the latest HEAD (Full Text).
        2. Traverse backwards through history, reading preceding blocks.
        3. For each block, apply its stored Reverse Delta to the current text.
        4. Stop when the target version is reached.
        """
        self._load_head() # Ensure HEAD info is up-to-date
        if not self.head_info:
            return "" # Empty history

        if ver_num is None or ver_num == self.head_info["ver"]:
            return self.head_info["text"]

        curr_text = self.head_info["text"]
        curr_block = self.head_info

        # Iterate backwards, applying deltas
        while curr_block:
            # Find the block immediately preceding the current one
            prev_block = self._get_prev_block(curr_block['start'])
            if not prev_block:
                # Reached the first block in history without finding target
                raise ValueError(f"Version '{ver_num}' not found in history (reached start of file).")

            # Apply delta: prev_block contains the delta to transform curr_text to prev_text
            # (Strictly speaking, V_prev contains delta to go from V_curr to V_prev)
            delta = prev_block["text"]
            curr_text = self._apply_reverse_delta(curr_text, delta)

            # Check if this is our target version
            if prev_block["ver"] == ver_num:
                return curr_text

            curr_block = prev_block # Move to the previous block

        return "" # Should not be reached if target_idx was found

    def log(self, limit: int | None = None, reverse: bool = False) -> list[dict]:
        """
        Retrieves the commit history.

        Args:
            limit: Maximum number of log entries to return.
            reverse: If True, returns history in chronological order (oldest first).
                     Default is False (newest first).
        """
        self._load_head()
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
            history.append(meta)

            if limit and len(history) >= limit:
                break

            prev_block = self._get_prev_block(curr_block['start'])
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

        head_text = self.head_info['text']
        head_lines = head_text.splitlines(keepends=True)

        # 1. Initialize tracker
        # Each item: {'head_index': int|None, 'blame': dict}
        # head_index maps to the index in the final output. None means it's a ghost line.
        current_commit = {
            'ver': self.head_info['ver'],
            'author': self.head_info['author'],
            'date': self.head_info['date'],
        }

        tracker = []
        for i in range(len(head_lines)):
            tracker.append({
                'head_index': i,
                'blame': current_commit,
            })

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
                    'ver': curr_block['ver'],
                    'author': curr_block['author'],
                    'date': curr_block['date'],
                }
                for item in tracker:
                    if item['head_index'] is not None:
                        if final_blame[item['head_index']] is None:
                            final_blame[item['head_index']] = reached_commit
                break

            prev_block = self._get_prev_block(curr_block['start'])

            if not prev_block:
                # Reached start (Ver 1.0).
                # All remaining non-ghost lines in tracker originate here.
                first_commit = {
                    'ver': curr_block['ver'],
                    'author': curr_block['author'],
                    'date': curr_block['date'],
                }
                for item in tracker:
                    if item['head_index'] is not None:
                        # If not already finalized (shouldn't happen if logic is correct, but for safety)
                        if final_blame[item['head_index']] is None:
                            final_blame[item['head_index']] = first_commit
                break

            prev_commit = {
                'ver': prev_block['ver'],
                'author': prev_block['author'],
                'date': prev_block['date'],
            }

            # Parse Delta (Current -> Prev)
            delta = prev_block['text']
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
                if cmd not in ('d', 'a'):
                    continue
                try:
                    start = int(parts[0][1:])
                    count = int(parts[1])
                except (ValueError, IndexError) as e:
                    raise ValueError(f"Invalid delta format: {e}") from e

                if cmd == 'a':
                    for _ in range(count):
                        if i < len(script_lines):
                            i += 1 # Skip payload lines

                commands.append({'cmd': cmd, 'start': start, 'count': count})

            # Sort descending to handle list mutations
            commands.sort(key=lambda x: x['start'], reverse=True)

            for c in commands:
                idx = c['start']
                if c['cmd'] == 'd':
                    # Delete lines starting at idx (1-based) -> list index idx-1
                    # These lines were born in Current.
                    list_idx = idx - 1

                    # These items are being removed from history.
                    # Their journey ends here. Finalize their blame.
                    removed_items = tracker[list_idx : list_idx + c['count']]
                    for item in removed_items:
                        if item['head_index'] is not None:
                            final_blame[item['head_index']] = item['blame']

                    del tracker[list_idx : list_idx + c['count']]

                elif c['cmd'] == 'a':
                    # Add lines to Prev.
                    # These lines don't exist in Current, so they are ghosts.
                    insert_idx = idx
                    ghost_items = [{'head_index': None, 'blame': prev_commit} for _ in range(c['count'])]
                    tracker[insert_idx:insert_idx] = ghost_items

            # Update blame for surviving items to Previous
            for item in tracker:
                item['blame'] = prev_commit

            curr_block = prev_block
            curr_depth += 1

        # 3. Construct Result
        result = []
        for i, line in enumerate(head_lines):
            info = final_blame[i]
            if info is None:
                # Should not happen, but fallback
                info = current_commit

            result.append({
                'line': line.rstrip('\n'),
                'ver': info['ver'],
                'author': info['author'],
                'date': info['date'],
            })

        return result

    def sign_head(self, signer_callbacks: list[Callable[[str], tuple[str, str]]]) -> bool:
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
            return False # Not supported in v1

        self._load_head()
        if not self.head_info:
            logger.error("Error: No HEAD block found to sign.")
            return False

        # Retrieve necessary data from HEAD for hash calculation
        head_data_for_hash = {
            'ver': self.head_info.get('ver'),
            'date': self.head_info.get('date'),
            'author': self.head_info.get('author'),
            'log': self.head_info.get('log'),
            'text': self.head_info.get('text'), # Full Text
        }
        stored_prev_hash = self.head_info.get('prev_hash')
        stored_hash = self.head_info.get('hash')

        # Re-calculate hash to confirm integrity before signing
        calculated_hash = self._calculate_block_hash(head_data_for_hash, prev_hash=stored_prev_hash)

        if calculated_hash != stored_hash:
            logger.error(f"Error: HEAD hash mismatch. Stored: {stored_hash},"
                "Calculated: {calculated_hash}. Cannot sign corrupted block.")
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
                    return False # Fail if any callback fails

        # Merge with existing signatures
        existing_signatures = self.head_info.get('signatures', [])
        all_signatures = existing_signatures + new_signatures

        # Prepare block data for rewriting HEAD
        # We need to use the original data (with Full Text) but replace its 'text' with delta if it were historical.
        # But for HEAD, its 'text' field in head_info IS the Full Text.
        block_data_for_rewrite = self.head_info.copy()
        # Remove internal metadata before formatting
        if 'start' in block_data_for_rewrite:
            del block_data_for_rewrite['start']
        if 'end' in block_data_for_rewrite:
            del block_data_for_rewrite['end']

        # Rewrite HEAD block at its original start position
        block_bytes = self._format_block(
            block_data_for_rewrite,
            current_hash=stored_hash,
            prev_hash=stored_prev_hash,
            signatures=all_signatures,
        )

        self.stream.seek(self.head_info['start'])
        self.stream.write(block_bytes)
        self.stream.truncate() # Ensure no leftover if new block is shorter (unlikely here)

        # Refresh head_info so instance state reflects new signatures
        self._load_head()

        return True

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
            return True # v1 has no integrity features

        self._load_head()
        if not self.head_info:
            return True # Empty file is valid

        curr_block = self.head_info

        # To verify hashes, we need the Full Text of each version.
        # Since we traverse backwards (HEAD -> V1), and blocks store Reverse Deltas,
        # we start with HEAD (Full Text) and apply deltas to get previous versions.
        # This matches the 'checkout' traversal logic perfectly.

        curr_text = curr_block['text'] # HEAD is Full Text

        # We need to track the 'expected hash' for the *next* block's prev_hash check
        # But we are going backwards.
        # Current Block's 'prev_hash' must match Previous Block's Hash.

        while curr_block:
            # 1. Verify Current Block's Hash
            # Hash is based on: Metadata + Full Text + prev_hash
            stored_hash = curr_block.get('hash')
            stored_prev_hash = curr_block.get('prev_hash')

            if not stored_hash:
                # v2 block MUST have a hash
                return False

            # Reconstruct data dict for hashing (must match commit logic)
            # We use the 'curr_text' which is the Full Text of this version.
            block_data_for_hash = curr_block.copy()
            block_data_for_hash['text'] = curr_text

            calculated_hash = self._calculate_block_hash(block_data_for_hash, prev_hash=stored_prev_hash)

            if calculated_hash != stored_hash:
                logger.error(f"calc hash = {calculated_hash} ,stored hash = {stored_hash}")
                logger.error(f"Hash mismatch at version {curr_block.get('ver')}")
                return False

            # 2. Verify Signatures (if callbacks provided)
            signatures = curr_block.get('signatures', [])
            if verifier_callbacks and signatures:
                # Signatures format: signer_id|timestamp|sig_val
                # Message signed: timestamp|hash
                for sig_entry in signatures:
                    try:
                        parts = sig_entry.split('|')
                        if len(parts) < 3:
                            continue
                        signer_id = parts[0]
                        timestamp = parts[1]
                        sig_val = "|".join(parts[2:]) # In case sig_val has |

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
                    except Exception:
                        return False

            # 3. Move to Previous Block
            prev_block = self._get_prev_block(curr_block['start'])

            if prev_block:
                # Verify Chain Link: Current.prev_hash == Hash(Previous Block)
                # But we haven't calculated Previous Block's hash yet?
                # We can't verify 'stored_prev_hash' until we process 'prev_block'.
                # Wait. 'stored_prev_hash' MUST equal the hash we *will* calculate for prev_block.
                # So we pass 'stored_prev_hash' down to the next iteration?
                # Let's verify it in the next iteration:
                # "Next iteration's calculated hash must equal This iteration's stored_prev_hash"

                # Apply delta to get Full Text of Previous Version
                delta = prev_block['text']
                curr_text = self._apply_reverse_delta(curr_text, delta)

                # We need to temporarily peek/calculate prev_block's hash to verify the link NOW?
                # Or just verify it when we become 'prev_block'.
                # If we verify it *when we process prev_block*, we check:
                #   calc_hash(prev_block) == prev_block.stored_hash
                # AND
                #   prev_block.stored_hash == curr_block.stored_prev_hash

                # So we just need to check:
                # curr_block['prev_hash'] == prev_block['hash']
                if curr_block.get('prev_hash') != prev_block.get('hash'):
                    logger.error(f"Chain broken between {curr_block.get('ver')} and {prev_block.get('ver')}")
                    return False

            curr_block = prev_block

        return True
