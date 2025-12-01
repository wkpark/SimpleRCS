import os
import zlib
from collections import defaultdict
from collections.abc import Iterator
from typing import BinaryIO


class StreamSequenceMatcher:
    """
    A memory-efficient sequence matcher that operates on file streams.
    It uses a GREEDY hash-based block matching algorithm for the coarse pass,
    followed by fine-grained refinement using standard difflib on 'replace' blocks.

    This implementation avoids the O(N*M) worst-case complexity of standard difflib,
    making it suitable for large files with small chunks.
    """

    def __init__(self,
        a_stream: BinaryIO,
        b_stream: BinaryIO,
        encoding: str = 'utf-8',
        chunk_size: int | None = None,
    ) -> None:
        self.a_stream = a_stream
        self.b_stream = b_stream
        self.encoding = encoding
        self.chunk_size = chunk_size

        self.a_hashes: list[int] = []
        self.b_hashes: list[int] = []

        self.a_offsets: list[int] = []
        self.b_offsets: list[int] = []

        # For faster lookup in a_hashes during greedy matching
        self.a_hash_map: dict[int, list[int]] = defaultdict(list)

        self._index_stream(self.a_stream, self.a_hashes, self.a_offsets, self.a_hash_map)
        self._index_stream(self.b_stream, self.b_hashes, self.b_offsets, None)

    def _index_stream(
        self,
        stream: BinaryIO,
        hash_list: list,
        offset_list: list,
        hash_map: dict[int, list[int]] | None,
    ) -> None:
        """Builds a list of chunk hashes and offsets from a file stream."""
        stream.seek(0)

        chunk_idx = 0
        while True:
            start_offset = stream.tell()

            if self.chunk_size:
                chunk = stream.read(self.chunk_size)
            else:
                chunk = stream.readline()

            if not chunk:
                break

            offset_list.append(start_offset)

            content_to_hash = chunk
            if not self.chunk_size: # Line-based mode
                content_to_hash = chunk.rstrip(b'\r\n')

            chunk_hash = hash(content_to_hash)
            hash_list.append(chunk_hash)

            if hash_map is not None:
                hash_map[chunk_hash].append(chunk_idx)

            chunk_idx += 1

    def _read_range(self, stream: BinaryIO, offsets: list, start_idx: int, end_idx: int) -> bytes:
        """Reads a range of chunks/lines from the stream."""
        if start_idx >= end_idx or start_idx >= len(offsets):
            return b""

        start_offset = offsets[start_idx]

        if end_idx < len(offsets):
            end_offset = offsets[end_idx]
            length = end_offset - start_offset
        else:
            stream.seek(0, os.SEEK_END)
            file_size = stream.tell()
            length = file_size - start_offset

        stream.seek(start_offset)
        return stream.read(length)

    def get_opcodes(self) -> Iterator[tuple[str, int, int, int, int]]:  # noqa: C901
        """
        Returns opcodes describing differences.
        Uses a greedy forward-scan algorithm.
        """
        len_a = len(self.a_hashes)
        len_b = len(self.b_hashes)

        a_curr = 0
        b_curr = 0

        iter_count = 0
        while b_curr < len_b:
            iter_count += 1
            # Greedy Match: Find the longest match for b[b_curr:] in a[a_curr:]
            # We use a_hash_map to find potential start points in A.

            best_match_len = 0
            best_match_a_start = -1

            # Look up candidates for the start of the match
            candidates = self.a_hash_map.get(self.b_hashes[b_curr], [])

            # Heuristic limit: If too many candidates, only check the first few (or closest ones).
            # Since candidates are appended in order, these are the 'earliest' occurrences.
            # But we filter 'cand_a_idx < a_curr'.
            # For large repetitive files, checking thousands of candidates kills performance.

            check_count = 0
            max_checks = 100 # Heuristic limit

            for cand_a_idx in candidates:
                if cand_a_idx < a_curr:
                    continue

                check_count += 1
                if check_count > max_checks:
                    break

                # Found a potential start. Extend match.
                current_len = 0
                # Optimization: We only need to extend if this could beat best_match_len
                # But for strict greedy (first match), we just take it and extend?
                # Let's try to find the *longest* match among candidates to be safe,
                # but limit search space if needed.

                # Simple Greedy: Take the FIRST valid candidate and extend it.
                # This effectively synchronizes on the first matching block.
                # This is very fast O(N).

                # Check if it matches at least one block (it does, by hash map)
                k = 0
                while (b_curr + k < len_b and cand_a_idx + k < len_a and
                       self.b_hashes[b_curr + k] == self.a_hashes[cand_a_idx + k]):

                    k += 1

                if k > best_match_len:
                    best_match_len = k
                    best_match_a_start = cand_a_idx

                    # Optimization: If we found a very long match, break early?
                    # Or if we just want "First Match", break immediately.
                    # "First Match" logic aligns with rsync/diff behavior often.
                    break

            if best_match_len > 0:
                # We found a match at (best_match_a_start, b_curr) with length best_match_len

                # 1. Handle unmatched gap before this match
                # Gap in A: a_curr ... best_match_a_start
                # Gap in B: b_curr ... b_curr (none, because match starts at b_curr)

                # Wait, if we skipped content in B to find this match?
                # The loop structure `while b_curr < len_b` iterates through B.
                # We look for match starting EXACTLY at `b_curr`.
                # If no match found at `b_curr`, we treat `b_curr` as INSERT/REPLACE.

                # Case: Match Found starting at b_curr
                if a_curr < best_match_a_start:
                    # We skipped some parts of A to find this match.
                    # This implies a DELETE of a[a_curr : best_match_a_start]
                    yield from self._emit_opcodes('delete', a_curr, best_match_a_start, b_curr, b_curr)

                # Emit the match
                yield from self._emit_opcodes('equal', best_match_a_start, best_match_a_start + best_match_len,
                                             b_curr, b_curr + best_match_len)

                a_curr = best_match_a_start + best_match_len
                b_curr += best_match_len

            else:
                # No match found starting at b_curr.
                # This block (b_curr) is either INSERT or REPLACE.
                # To distinguish, we look ahead for a synchronization point.

                # This is the "Gap Filling" phase.
                # We advance b_curr until we find a block that matches something in A (>= a_curr).

                sync_b_idx = -1
                sync_a_idx = -1

                # Scan B forward
                scan_limit = len_b
                # Optimization: limit lookahead? No, we need to sync.

                for k in range(b_curr + 1, scan_limit):
                    # Check if b[k] exists in A (>= a_curr)
                    candidates = self.a_hash_map.get(self.b_hashes[k], [])
                    for cand in candidates:
                        if cand >= a_curr:
                            sync_b_idx = k
                            sync_a_idx = cand
                            break
                    if sync_b_idx != -1:
                        break

                if sync_b_idx != -1:
                    # Found a sync point at (sync_a_idx, sync_b_idx)
                    # gap in A: a_curr ... sync_a_idx
                    # gap in B: b_curr ... sync_b_idx

                    # If both have gaps, it's a REPLACE
                    if a_curr < sync_a_idx:
                        yield from self._emit_opcodes('replace', a_curr, sync_a_idx, b_curr, sync_b_idx)
                    else:
                        # Only gap in B -> INSERT
                        yield from self._emit_opcodes('insert', a_curr, a_curr, b_curr, sync_b_idx)

                    a_curr = sync_a_idx
                    b_curr = sync_b_idx
                else:
                    # No sync point found until EOF.
                    # Everything remaining is REPLACE or INSERT.
                    if a_curr < len_a:
                        yield from self._emit_opcodes('replace', a_curr, len_a, b_curr, len_b)
                    else:
                        yield from self._emit_opcodes('insert', a_curr, a_curr, b_curr, len_b)

                    a_curr = len_a
                    b_curr = len_b
                    break

        # Trailing deletion
        if a_curr < len_a:
             yield from self._emit_opcodes('delete', a_curr, len_a, len_b, len_b)

    def _emit_opcodes(self, tag: str, i1: int, i2: int, j1: int, j2: int) -> Iterator[tuple[str, int, int, int, int]]:
        """Yields opcodes with correct byte/line mapping and refinement."""
        if self.chunk_size is None:
            # Line mode: return indices
            yield (tag, i1, i2, j1, j2)
            return

        # Binary mode: convert to byte offsets
        a_start = self._get_byte_offset(self.a_offsets, i1)
        a_end = self._get_byte_offset(self.a_offsets, i2, is_end=True, stream=self.a_stream)
        b_start = self._get_byte_offset(self.b_offsets, j1)
        b_end = self._get_byte_offset(self.b_offsets, j2, is_end=True, stream=self.b_stream)

        if tag == 'replace':
            chunk_a = self._read_range(self.a_stream, self.a_offsets, i1, i2)
            chunk_b = self._read_range(self.b_stream, self.b_offsets, j1, j2)

            # Use fast greedy refinement instead of slow difflib
            yield from self._refine_greedy(chunk_a, chunk_b, a_start, b_start)
        else:
            yield (tag, a_start, a_end, b_start, b_end)

    def _refine_greedy(  # noqa: C901
        self, a: bytes,
        b: bytes,
        a_global_offset: int,
        b_global_offset: int,
    ) -> Iterator[tuple[str, int, int, int, int]]:
        """
        Optimized greedy diff algorithm for refinement.
        - Adaptive anchor length
        - Search window limit
        - Faster synchronization point discovery
        """
        len_a = len(a)
        len_b = len(b)

        if len_a == 0:
            if len_b > 0:
                yield ('insert', a_global_offset, a_global_offset,
                       b_global_offset, b_global_offset + len_b)
            return

        if len_b == 0:
            yield ('delete', a_global_offset, a_global_offset + len_a,
                   b_global_offset, b_global_offset)
            return

        a_idx = 0
        b_idx = 0

        # Adaptive anchor length heuristics
        BASE_MIN_MATCH = 24 # Base minimum match length

        # Search window limit: Adaptive, capped at 1MB to balance performance and coverage.
        # Uses a wider window (64KB base) for larger files, scaling up to 1MB.
        MAX_SEARCH_DISTANCE = min(1024 * 1024, max(65536, min(len_a, len_b)))

        while b_idx < len_b and a_idx < len_a:
            # Dynamically adjust anchor length based on remaining data.
            # This allows smaller matches towards the end of the data.
            remaining = min(len_a - a_idx, len_b - b_idx)
            current_min_match = max(12, min(BASE_MIN_MATCH, remaining // 64))

            # Create anchor
            anchor_len = min(current_min_match, len_b - b_idx)

            if anchor_len < 8: # Break if anchor is too small to be reliable
                break
            anchor_b = b[b_idx : b_idx + anchor_len]

            # Search for Anchor B in A (within window)
            search_end = min(len_a, a_idx + MAX_SEARCH_DISTANCE)
            match_in_a = a.find(anchor_b, a_idx, search_end)

            if match_in_a != -1:
                # Handle Gap in A (Delete)
                if a_idx < match_in_a:
                    yield ('delete',
                           a_global_offset + a_idx, a_global_offset + match_in_a,
                           b_global_offset + b_idx, b_global_offset + b_idx)

                # Extend Match
                match_len = anchor_len
                max_extend = min(len_b - b_idx, len_a - match_in_a)

                while match_len < max_extend and b[b_idx + match_len] == a[match_in_a + match_len]:
                    match_len += 1

                yield ('equal',
                       a_global_offset + match_in_a, a_global_offset + match_in_a + match_len,
                       b_global_offset + b_idx, b_global_offset + b_idx + match_len)

                a_idx = match_in_a + match_len
                b_idx += match_len
                continue

            # Anchor B not found in A -> Try Reverse Search (Anchor A in B)
            anchor_a_len = min(current_min_match, len_a - a_idx)

            if anchor_a_len >= current_min_match // 2:
                anchor_a = a[a_idx : a_idx + anchor_a_len]
                search_end_b = min(len_b, b_idx + MAX_SEARCH_DISTANCE)
                match_in_b = b.find(anchor_a, b_idx, search_end_b)

                if match_in_b != -1:
                    # Handle Gap in B (Insert)
                    yield ('insert',
                           a_global_offset + a_idx, a_global_offset + a_idx,
                           b_global_offset + b_idx, b_global_offset + match_in_b)

                    # Extend Match
                    match_len = anchor_a_len
                    max_extend = min(len_b - match_in_b, len_a - a_idx)

                    while match_len < max_extend and b[match_in_b + match_len] == a[a_idx + match_len]:
                        match_len += 1

                    yield ('equal',
                           a_global_offset + a_idx, a_global_offset + a_idx + match_len,
                           b_global_offset + match_in_b, b_global_offset + match_in_b + match_len)

                    a_idx += match_len
                    b_idx = match_in_b + match_len
                    continue

            # Both Failed -> Treat as Replace (Diff)
            # Adaptive Step: Proportional to remaining size
            remaining = min(len_a - a_idx, len_b - b_idx)
            step = min(remaining, max(current_min_match, remaining // 4))

            yield ('replace',
                   a_global_offset + a_idx, a_global_offset + a_idx + step,
                   b_global_offset + b_idx, b_global_offset + b_idx + step)

            a_idx += step
            b_idx += step

        # Handle Remainders
        if a_idx < len_a and b_idx < len_b:
            yield ('replace',
                   a_global_offset + a_idx, a_global_offset + len_a,
                   b_global_offset + b_idx, b_global_offset + len_b)
        elif a_idx < len_a:
            yield ('delete',
                   a_global_offset + a_idx, a_global_offset + len_a,
                   b_global_offset + len_b, b_global_offset + len_b)
        elif b_idx < len_b:
            yield ('insert',
                   a_global_offset + len_a, a_global_offset + len_a,
                   b_global_offset + b_idx, b_global_offset + len_b)

    def _get_byte_offset(self, offsets: int, idx: int, is_end: bool = False, stream: BinaryIO | None = None) -> int:
        if idx < len(offsets):
            return offsets[idx]
        if stream and is_end and offsets:
            stream.seek(0, os.SEEK_END)
            return stream.tell()
        return 0 # Should handle empty/start cases

    def get_matching_blocks(self) -> list[tuple[int, int, int]]:
        # Compatibility wrapper
        blocks = []
        for tag, i1, i2, j1, _j2 in self.get_opcodes():
            if tag == 'equal':
                blocks.append((i1, j1, i2 - i1))
        blocks.append((self._get_byte_offset(self.a_offsets, len(self.a_hashes), True, self.a_stream),
                       self._get_byte_offset(self.b_offsets, len(self.b_hashes), True, self.b_stream), 0))
        return blocks

    def get_lines(self, stream_type: str) -> list[bytes]:
        stream = self.a_stream if stream_type == 'a' else self.b_stream
        stream.seek(0)
        return stream.readlines()

    def get_lines_from_stream(self, stream_type: str, start_index: int, end_index: int) -> list[bytes]:
        stream = self.a_stream if stream_type == 'a' else self.b_stream
        offsets = self.a_offsets if stream_type == 'a' else self.b_offsets
        raw = self._read_range(stream, offsets, start_index, end_index)
        if self.chunk_size is None:
            return raw.splitlines(keepends=True)
        return [raw]

# --- Rolling Hash Implementation ---

_BASE = 65521
_OFFS = 1

class RollingHashMatcher:
    """
    Implements rsync-like rolling hash matching for efficient binary diffing.
    Handles shifted data (inserts/deletes) better than fixed chunking.

    Improvements:
    - Better memory management with streaming buffer
    - Configurable extension limits
    - Optimized hash rotation
    - Clearer opcode generation
    - Better error handling
    """

    def __init__(
        self,
        a_stream: BinaryIO,
        b_stream: BinaryIO,
        chunk_size: int = 4096,
        max_memory: int = 100 * 1024 * 1024,  # 100MB default
    ) -> None:
        self.a_stream = a_stream
        self.b_stream = b_stream
        self.chunk_size = chunk_size
        self.max_memory = max_memory

        # Index old file (signature generation)
        self.a_map = self._index_file(self.a_stream)
        self.a_size = self._get_file_size(self.a_stream)

    def _get_file_size(self, stream: BinaryIO) -> int:
        """Get file size efficiently."""
        current_pos = stream.tell()
        stream.seek(0, 2)  # Seek to end
        size = stream.tell()
        stream.seek(current_pos)  # Restore position
        return size

    def _index_file(self, stream: BinaryIO) -> dict[int, list[int]]:
        """
        Build hash map of fixed-size chunks from the old file.
        Returns {adler32_hash: [list of byte offsets]}.
        """
        stream.seek(0)
        hash_map = defaultdict(list)
        offset = 0

        while True:
            chunk = stream.read(self.chunk_size)
            if not chunk:
                break

            # Use Adler-32 for fast rolling hash
            h = zlib.adler32(chunk) & 0xFFFFFFFF
            hash_map[h].append(offset)
            offset += len(chunk)

        return hash_map

    def _compute_adler32_components(self, data: bytes) -> tuple[int, int, int]:
        """
        Compute Adler-32 hash and extract s1, s2 components.
        Returns (hash, s1, s2).
        """
        h = zlib.adler32(data) & 0xFFFFFFFF
        s1 = h & 0xFFFF
        s2 = (h >> 16) & 0xFFFF
        return h, s1, s2

    def _roll_hash(self, s1: int, s2: int, out_byte: int, in_byte: int) -> tuple[int, int, int]:
        """
        Roll the Adler-32 hash by removing out_byte and adding in_byte.
        Returns (new_hash, new_s1, new_s2).
        """
        # Update s1: remove old byte, add new byte
        s1 = (s1 - out_byte + in_byte) % _BASE

        # Update s2: remove contribution of old byte, add new s1
        term1 = (self.chunk_size * out_byte) % _BASE
        s2 = (s2 - term1 + s1 - _OFFS) % _BASE

        h = ((s2 << 16) | s1) & 0xFFFFFFFF
        return h, s1, s2

    def _verify_and_extend_match(
        self,
        old_offset: int,
        new_offset: int,
        b_data: bytes,
        max_extend: int = 1024 * 1024,  # 1MB extension limit
    ) -> int:
        """
        Verify chunk match and extend it as far as possible.
        Returns the total matched length (0 if no match).
        """
        # First verify the initial chunk
        self.a_stream.seek(old_offset)
        old_chunk = self.a_stream.read(self.chunk_size)

        new_chunk = b_data[new_offset:new_offset + self.chunk_size]

        if old_chunk != new_chunk:
            return 0

        matched_len = self.chunk_size

        # Try to extend the match
        remaining_new = len(b_data) - (new_offset + matched_len)
        remaining_old = self.a_size - (old_offset + matched_len)
        extend_limit = min(remaining_new, remaining_old, max_extend)

        if extend_limit > 0:
            # Read extension data in chunks for efficiency
            EXTEND_CHUNK = 8192
            extended = 0

            while extended < extend_limit:
                read_size = min(EXTEND_CHUNK, extend_limit - extended)

                self.a_stream.seek(old_offset + matched_len + extended)
                old_ext = self.a_stream.read(read_size)
                new_ext = b_data[new_offset + matched_len + extended:
                               new_offset + matched_len + extended + read_size]

                # Find common prefix
                mismatch_idx = self._find_mismatch(old_ext, new_ext)
                extended += mismatch_idx

                if mismatch_idx < len(old_ext):
                    break  # Found mismatch

            matched_len += extended

        return matched_len

    @staticmethod
    def _find_mismatch(a: bytes, b: bytes) -> int:
        """Find index of first mismatch between two byte sequences."""
        min_len = min(len(a), len(b))
        for i in range(min_len):
            if a[i] != b[i]:
                return i
        return min_len

    def get_opcodes(self) -> Iterator[tuple[str, int, int, int, int]]:  # noqa: C901
        """
        Scan new file using rolling hash to find matches in old file.
        Yields (tag, i1, i2, j1, j2) where:
        - 'equal': matched block (copy from old file)
        - 'insert': new data to insert
        - 'delete': gap in old file (seek forward)

        Byte offsets: i1, i2 are in old file; j1, j2 are in new file.
        """
        self.b_stream.seek(0)
        b_size = self._get_file_size(self.b_stream)

        # For memory efficiency, check if we can load entire new file
        if b_size > self.max_memory:
            raise MemoryError(
                f"New file ({b_size} bytes) exceeds max_memory limit "
                f"({self.max_memory} bytes). Increase max_memory or use streaming.",
            )

        # Load new file into memory for faster access
        self.b_stream.seek(0)
        b_data = self.b_stream.read()

        # Initialize scanning state
        i = 0  # Current position in new file
        pending_insert_start = 0
        last_old_pos = 0

        # Initialize rolling hash window
        if len(b_data) >= self.chunk_size:
            h, s1, s2 = self._compute_adler32_components(b_data[:self.chunk_size])
            window_ready = True
        else:
            h, s1, s2 = 1, _OFFS, 0
            window_ready = False

        # Main scanning loop
        while i < len(b_data):
            best_match_len = 0
            best_old_offset = -1

            # Check for matches only if we have a full window
            if window_ready and (i + self.chunk_size <= len(b_data)):
                candidates = self.a_map.get(h, [])

                # Limit candidate checking to avoid O(n²) behavior
                MAX_CANDIDATES = 20

                for old_offset in candidates[:MAX_CANDIDATES]:
                    match_len = self._verify_and_extend_match(old_offset, i, b_data)

                    if match_len > best_match_len:
                        best_match_len = match_len
                        best_old_offset = old_offset

                        # Early exit for very long matches
                        if match_len > self.chunk_size * 4:
                            break

            # Process match if found
            if best_match_len > 0:
                # Yield pending inserts
                if i > pending_insert_start:
                    yield ('insert', 0, 0, pending_insert_start, i)

                # Yield delete/seek if old file position changed
                if best_old_offset != last_old_pos:
                    yield ('delete', last_old_pos, best_old_offset, i, i)

                # Yield match
                yield ('equal', best_old_offset, best_old_offset + best_match_len,
                       i, i + best_match_len)

                # Advance position
                i += best_match_len
                pending_insert_start = i
                last_old_pos = best_old_offset + best_match_len

                # Reinitialize hash at new position
                if i + self.chunk_size <= len(b_data):
                    h, s1, s2 = self._compute_adler32_components(
                        b_data[i:i + self.chunk_size],
                    )
                    window_ready = True
                else:
                    window_ready = False

                continue

            # No match found - roll the window forward by one byte
            if i + self.chunk_size < len(b_data):
                out_byte = b_data[i]
                in_byte = b_data[i + self.chunk_size]
                h, s1, s2 = self._roll_hash(s1, s2, out_byte, in_byte)
                i += 1
            else:
                # Approaching EOF - can't maintain full window
                break

        # Flush remaining inserts
        if pending_insert_start < len(b_data):
            yield ('insert', 0, 0, pending_insert_start, len(b_data))
