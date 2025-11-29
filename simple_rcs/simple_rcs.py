import difflib
import io
import os
import re
from datetime import datetime
from typing import BinaryIO


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
    """

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

        if content_or_path is None:
            # New empty in-memory RCS
            self.stream = io.BytesIO()
            self.owns_handle = True
        elif isinstance(content_or_path, str):
            # Check if it's a file path heuristic
            if os.path.exists(content_or_path) or (not content_or_path.startswith(('@', 'ver')) and '\n' not in content_or_path):
                # Assume file path if it exists or doesn't look like RCS content and no newlines
                mode = "rb+" if os.path.exists(content_or_path) else "wb+"
                self.stream = open(content_or_path, mode)
                self.file_path = content_or_path
                self.owns_handle = True
            else:
                # Treat as raw string content -> In-memory stream
                self.stream = io.BytesIO(content_or_path.encode('utf-8'))
                self.owns_handle = True
        elif isinstance(content_or_path, bytes):
            self.stream = io.BytesIO(content_or_path)
            self.owns_handle = True
        elif hasattr(content_or_path, 'read') and hasattr(content_or_path, 'seek'):
            # External file-like object
            self.stream = content_or_path
            self.owns_handle = False
        else:
            raise ValueError("Invalid input type. Expected file path, string, bytes, or file-like object.")

        self.head_info: dict | None = None
        self._load_head()

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
        content = content_bytes
        length = len(content)
        pos = 0
        data = {}

        # Define keywords as bytes for direct comparison
        _KEYWORDS = [b'ver', b'version', b'date', b'author', b'log', b'text']

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
            if key_bytes not in _KEYWORDS:
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
            except (ValueError, IndexError):
                continue # Skip malformed commands

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
        return "".join(lines)

    def _format_block(self, data: dict) -> bytes:
        """Formats a block dictionary into bytes for writing to the stream."""
        keys = ["ver", "date", "author", "log", "text"]
        lines = []
        for key in keys:
            val = self._escape(str(data.get(key, "")))
            lines.append(f"{key} @{val}@;")
        return ("\n".join(lines) + "\n\n").encode('utf-8')

    def commit(self, content: str, author: str = "unknown", log: str = "") -> str:
        """
        Commits new content as the latest version.

        Process:
        1. Parse current HEAD (which is Full Text).
        2. Compute Reverse Delta (New -> Old HEAD).
        3. Overwrite the on-disk HEAD block with this Delta.
        4. Append the New Content as the new HEAD block (Full Text).

        Returns:
            The new version number (e.g., "1.2") if working with a file path.
            The full RCS content string if working with an in-memory stream.
        """
        self._load_head() # Refresh HEAD info by scanning the stream
        now = datetime.now().isoformat()

        if not self.head_info:
            # First commit: Write full content as 1.0
            new_ver = "1.0"
            block_data = {"ver": new_ver, "date": now, "author": author, "log": log, "text": content}

            self.stream.seek(0, os.SEEK_END) # Append to end
            self.stream.write(self._format_block(block_data))

            if isinstance(self.stream, io.BytesIO):
                return self.get_content()
            return new_ver

        head = self.head_info
        head_content = head["text"]

        # 1. Compute Reverse Delta: New Content (Vn+1) -> Old Head Content (Vn)
        delta = self._generate_reverse_delta(content, head_content)

        # 2. Prepare blocks for writing
        # The old HEAD block (Vn) becomes a Delta block.
        head_block_data = head.copy()
        head_block_data["text"] = delta
        # Remove internal metadata (start/end offsets) before formatting for writing
        if "start" in head_block_data:
            del head_block_data["start"]
        if "end" in head_block_data:
            del head_block_data["end"]

        # The new HEAD block (Vn+1) is Full Text.
        last_ver = float(head.get("ver", "0.0"))
        new_ver = f"{last_ver + 0.1:.1f}"
        new_block_data = {"ver": new_ver, "date": now, "author": author, "log": log, "text": content}

        # 3. Write to stream: Overwrite previous HEAD, then Append new HEAD
        old_block_bytes = self._format_block(head_block_data)
        new_block_bytes = self._format_block(new_block_data)

        self.stream.seek(self.head_info['start'])
        self.stream.write(old_block_bytes)
        self.stream.write(new_block_bytes)
        self.stream.truncate() # Crucial: remove any leftover if new block is shorter

        # Return based on stream type
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

    def blame(self) -> list[dict]:  # noqa: C901
        """
        Annotates each line of the HEAD version with the revision that last modified it.

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

        # 2. Traverse backwards
        while curr_block:
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
                except Exception as _:
                    continue

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

