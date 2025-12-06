import difflib
import os
import zlib
from collections import defaultdict, namedtuple
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
        chunk_size: int | None = None,
    ) -> None:
        self.a_stream = a_stream
        self.b_stream = b_stream
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

    def _get_byte_offset(self, offsets: list[int], idx: int, is_end: bool = False, stream: BinaryIO | None = None) -> int:
        if idx < len(offsets):
            return offsets[idx]
        if stream and (is_end or idx >= len(offsets)):
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


Match = namedtuple('Match', 'a b size')

class StreamTextSequenceMatcher:
    """
    A memory-efficient sequence matcher that operates on file streams.
    It implements the core logic of difflib.SequenceMatcher but optimized for
    large streams by using hash-based coarse matching and fine-grained refinement.

    Please see https://github.com/python/cpython/blob/3.14/Lib/difflib.py
    """

    def __init__(self,
        a_stream: BinaryIO,
        b_stream: BinaryIO,
        encoding: str = 'utf-8',
        chunk_size: int | None = None,
        isjunk=None,
        autojunk=True,
    ) -> None:
        self.isjunk = isjunk
        self.autojunk = autojunk
        self.a_stream = a_stream
        self.b_stream = b_stream
        self.encoding = encoding
        self.chunk_size = chunk_size

        self.a_offsets: list[int] = []
        self.b_offsets: list[int] = []

        self.a = self.b = None
        self.set_seqs(a_stream, b_stream)

    def set_seqs(self, a_stream: BinaryIO, b_stream: BinaryIO) -> None:
        self.set_seq1(a_stream)
        self.set_seq2(b_stream)

    def set_seq1(self, a_stream: BinaryIO) -> None:
        if a_stream is self.a_stream and self.a:
            return
        self.a_stream = a_stream
        self.a_offsets = []
        # self.a stores hashes
        self.a = self._index_stream(a_stream, self.a_offsets)
        self.matching_blocks = self.opcodes = None

    def set_seq2(self, b_stream: BinaryIO) -> None:
        if b_stream is self.b_stream and self.b:
            return
        self.b_stream = b_stream
        self.b_offsets = []
        # self.b stores hashes
        self.b = self._index_stream(b_stream, self.b_offsets)
        self.matching_blocks = self.opcodes = None
        self.fullbcount = None
        self.__chain_b()

    def _index_stream(self, stream: BinaryIO, offset_list: list) -> list[int]:
        """Builds a list of chunk hashes and records offsets."""
        stream.seek(0)
        hash_list = []

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
            if not self.chunk_size:
                content_to_hash = chunk.rstrip(b'\r\n')

            chunk_hash = hash(content_to_hash)
            hash_list.append(chunk_hash)

        return hash_list

    def _read_range(self, stream: BinaryIO, offsets: list, start_idx: int, end_idx: int) -> bytes:
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

    # --- Core Algorithm from difflib (Modified for Hash Lists) ---

    def __chain_b(self) -> None:
        # Because isjunk is a user-defined (not C) function, and we test
        # for junk a LOT, it's important to minimize the number of calls.
        # Before the tricks described here, __chain_b was by far the most
        # time-consuming routine in the whole module!  If anyone sees
        # Jim Roskind, thank him again for profile.py -- I never would
        # have guessed that.
        # The first trick is to build b2j ignoring the possibility
        # of junk.  I.e., we don't call isjunk at all yet.  Throwing
        # out the junk later is much cheaper than building b2j "right"
        # from the start.
        b = self.b
        self.b2j = b2j = {}

        for i, elt in enumerate(b):
            indices = b2j.setdefault(elt, [])
            indices.append(i)

        # Purge junk elements
        self.bjunk = junk = set()
        isjunk = self.isjunk
        if isjunk:
            for elt in b2j.keys():
                if isjunk(elt):
                    junk.add(elt)
            for elt in junk: # separate loop avoids separate list of keys
                del b2j[elt]

        # Purge popular elements that are not junk
        self.bpopular = popular = set()
        n = len(b)
        if self.autojunk and n >= 200:
            ntest = n // 100 + 1
            for elt, idxs in b2j.items():
                if len(idxs) > ntest:
                    popular.add(elt)
            for elt in popular: # ditto; as fast for 1% deletion
                del b2j[elt]

    def find_longest_match(self, alo=0, ahi=None, blo=0, bhi=None) -> tuple:  # noqa: C901
        """
        Find longest matching block in a[alo:ahi] and b[blo:bhi] from difflib
        """
        a, b, b2j, isbjunk = self.a, self.b, self.b2j, self.bjunk.__contains__
        if ahi is None:
            ahi = len(a)
        if bhi is None:
            bhi = len(b)
        besti, bestj, bestsize = alo, blo, 0
        # find longest junk-free match
        # during an iteration of the loop, j2len[j] = length of longest
        # junk-free match ending with a[i-1] and b[j]
        j2len = {}
        nothing = []
        for i in range(alo, ahi):
            # look at all instances of a[i] in b; note that because
            # b2j has no junk keys, the loop is skipped if a[i] is junk
            j2lenget = j2len.get
            newj2len = {}
            for j in b2j.get(a[i], nothing):
                # a[i] matches b[j]
                if j < blo:
                    continue
                if j >= bhi:
                    break
                k = newj2len[j] = j2lenget(j-1, 0) + 1
                if k > bestsize:
                    besti, bestj, bestsize = i-k+1, j-k+1, k
            j2len = newj2len

        # Extend the best by non-junk elements on each end.  In particular,
        # "popular" non-junk elements aren't in b2j, which greatly speeds
        # the inner loop above, but also means "the best" match so far
        # doesn't contain any junk *or* popular non-junk elements.
        while besti > alo and bestj > blo and \
              not isbjunk(b[bestj-1]) and \
              a[besti-1] == b[bestj-1]:
            besti, bestj, bestsize = besti-1, bestj-1, bestsize+1
        while besti+bestsize < ahi and bestj+bestsize < bhi and \
              not isbjunk(b[bestj+bestsize]) and \
              a[besti+bestsize] == b[bestj+bestsize]:
            bestsize += 1

        # Now that we have a wholly interesting match (albeit possibly
        # empty!), we may as well suck up the matching junk on each
        # side of it too.  Can't think of a good reason not to, and it
        # saves post-processing the (possibly considerable) expense of
        # figuring out what to do with it.  In the case of an empty
        # interesting match, this is clearly the right thing to do,
        # because no other kind of match is possible in the regions.
        while besti > alo and bestj > blo and \
              isbjunk(b[bestj-1]) and \
              a[besti-1] == b[bestj-1]:
            besti, bestj, bestsize = besti-1, bestj-1, bestsize+1
        while besti+bestsize < ahi and bestj+bestsize < bhi and \
              isbjunk(b[bestj+bestsize]) and \
              a[besti+bestsize] == b[bestj+bestsize]:
            bestsize = bestsize + 1

        return Match(besti, bestj, bestsize)

    def get_matching_blocks(self) -> list:
        """Return list of triples describing matching subsequences.

        Each triple is of the form (i, j, n), and means that
        a[i:i+n] == b[j:j+n].  The triples are monotonically increasing in
        i and in j.  New in Python 2.5, it's also guaranteed that if
        (i, j, n) and (i', j', n') are adjacent triples in the list, and
        the second is not the last triple in the list, then i+n != i' or
        j+n != j'.  IOW, adjacent triples never describe adjacent equal
        blocks.

        The last triple is a dummy, (len(a), len(b), 0), and is the only
        triple with n==0.

        >>> s = SequenceMatcher(None, "abxcd", "abcd")
        >>> list(s.get_matching_blocks())
        [Match(a=0, b=0, size=2), Match(a=3, b=2, size=2), Match(a=5, b=4, size=0)]
        """

        if self.matching_blocks is not None:
            return self.matching_blocks
        la, lb = len(self.a), len(self.b)

        # This is most naturally expressed as a recursive algorithm, but
        # at least one user bumped into extreme use cases that exceeded
        # the recursion limit on their box.  So, now we maintain a list
        # ('queue`) of blocks we still need to look at, and append partial
        # results to `matching_blocks` in a loop; the matches are sorted
        # at the end.
        queue = [(0, la, 0, lb)]
        matching_blocks = []
        while queue:
            alo, ahi, blo, bhi = queue.pop()
            i, j, k = x = self.find_longest_match(alo, ahi, blo, bhi)
            # a[alo:i] vs b[blo:j] unknown
            # a[i:i+k] same as b[j:j+k]
            # a[i+k:ahi] vs b[j+k:bhi] unknown
            if k:   # if k is 0, there was no matching block
                matching_blocks.append(x)
                if alo < i and blo < j:
                    queue.append((alo, i, blo, j))
                if i+k < ahi and j+k < bhi:
                    queue.append((i+k, ahi, j+k, bhi))
        matching_blocks.sort()

        # It's possible that we have adjacent equal blocks in the
        # matching_blocks list now.  Starting with 2.5, this code was added
        # to collapse them.
        i1 = j1 = k1 = 0
        non_adjacent = []
        for i2, j2, k2 in matching_blocks:
            # Is this block adjacent to i1, j1, k1?
            if i1 + k1 == i2 and j1 + k1 == j2:
                # Yes, so collapse them -- this just increases the length of
                # the first block by the length of the second, and the first
                # block so lengthened remains the block to compare against.
                k1 += k2
            else:
                # Not adjacent.  Remember the first block (k1==0 means it's
                # the dummy we started with), and make the second block the
                # new block to compare against.
                if k1:
                    non_adjacent.append((i1, j1, k1))
                i1, j1, k1 = i2, j2, k2
        if k1:
            non_adjacent.append((i1, j1, k1))

        non_adjacent.append( (la, lb, 0) )
        self.matching_blocks = list(map(Match._make, non_adjacent))
        return self.matching_blocks

    def get_opcodes(self) -> list:  # noqa: C901
        """
        Return list of 5-tuples describing how to turn a into b.
        Each tuple is of the form (tag, i1, i2, j1, j2).

        This method works in two steps for memory efficiency:
        1.  **Coarse Pass**: It first compares the lists of hashes (which are small) to find general matching blocks.
        2.  **Fine Pass (Refinement)**: If 'chunk_size' is set (Binary Mode), it refines 'replace' blocks
            by reading the actual data chunks and comparing them byte-by-byte.

        Returns:
            - If chunk_size is None (Text/Line Mode): Returns LINE indices.
            - If chunk_size is Set (Binary/Chunk Mode): Returns BYTE offsets.
        """
        if self.opcodes is not None:
            return self.opcodes

        # --- Step 1: Coarse Pass ---
        # Get matching blocks based on HASH comparisons.
        # This tells us which chunks/lines definitely match or differ.
        i = j = 0
        coarse_opcodes = []
        for ai, bj, size in self.get_matching_blocks():
            tag = ''
            if i < ai and j < bj:
                tag = 'replace'
            elif i < ai:
                tag = 'delete'
            elif j < bj:
                tag = 'insert'
            if tag:
                coarse_opcodes.append( (tag, i, ai, j, bj) )
            i, j = ai+size, bj+size
            if size:
                coarse_opcodes.append( ('equal', ai, i, bj, j) )

        # --- Step 2: Refinement and Offset Translation ---
        # We transform the coarse block indices into useful offsets.
        refined_opcodes = []

        for tag, i1, i2, j1, j2 in coarse_opcodes:
            # Case A: Text/Line Mode (chunk_size is None)
            # We just return the line numbers. SimpleRCS expects this format.
            if self.chunk_size is None:
                refined_opcodes.append((tag, i1, i2, j1, j2))
                continue

            # Case B: Binary/Chunk Mode (chunk_size is set)
            # We need to convert chunk indices to actual BYTE offsets in the file.

            # Calculate start/end byte offsets for stream A
            a_start = self.a_offsets[i1] if i1 < len(self.a_offsets) else (self.a_offsets[-1] + self.chunk_size if self.a_offsets else 0)
            a_end = self.a_offsets[i2] if i2 < len(self.a_offsets) else (self.a_offsets[-1] + self.chunk_size if self.a_offsets else 0)

            # Special handling for EOF: if the range goes to the end, use actual file position
            if i2 == len(self.a) and len(self.a_offsets) > 0:
                self.a_stream.seek(0, os.SEEK_END)
                a_end = self.a_stream.tell()

            # Calculate start/end byte offsets for stream B
            b_start = self.b_offsets[j1] if j1 < len(self.b_offsets) else (self.b_offsets[-1] + self.chunk_size if self.b_offsets else 0)
            b_end = self.b_offsets[j2] if j2 < len(self.b_offsets) else (self.b_offsets[-1] + self.chunk_size if self.b_offsets else 0)

            if j2 == len(self.b) and len(self.b_offsets) > 0:
                self.b_stream.seek(0, os.SEEK_END)
                b_end = self.b_stream.tell()

            # Refinement Logic:
            # If we found a 'replace' block, it means the hashes didn't match.
            # But maybe only 1 byte changed in a 64-byte chunk!
            # So we read the ACTUAL bytes and compare them precisely.
            if tag == 'replace':
                chunk_a = self._read_range(self.a_stream, self.a_offsets, i1, i2)
                chunk_b = self._read_range(self.b_stream, self.b_offsets, j1, j2)

                # Use standard difflib on this small piece of data.
                # This gives us exact byte-level differences.
                fine_sm = difflib.SequenceMatcher(None, chunk_a, chunk_b, autojunk=False)
                for sub_tag, sub_i1, sub_i2, sub_j1, sub_j2 in fine_sm.get_opcodes():
                    # Map the local offsets (relative to the chunk) back to global file offsets
                    refined_opcodes.append((
                        sub_tag,
                        a_start + sub_i1, a_start + sub_i2,
                        b_start + sub_j1, b_start + sub_j2,
                    ))
            else:
                # For equal/insert/delete, just add the byte range.
                refined_opcodes.append((tag, a_start, a_end, b_start, b_end))

        self.opcodes = refined_opcodes
        return self.opcodes

    # Helpers for compatibility/testing
    def get_lines(self, stream_type: str) -> list[bytes]:
        stream = self.a_stream if stream_type == 'a' else self.b_stream
        stream.seek(0)
        return stream.readlines()

    def get_lines_from_stream(self, stream_type: str, start_index: int, end_index: int) -> list[bytes]:
        stream = self.a_stream if stream_type == 'a' else self.b_stream
        offsets = self.a_offsets if stream_type == 'a' else self.b_offsets
        if not offsets or start_index < 0 or start_index >= len(offsets) or end_index < start_index:
            return []
        raw_bytes = self._read_range(stream, offsets, start_index, end_index)
        if self.chunk_size is None:
            return raw_bytes.splitlines(keepends=True)
        else:
            return [raw_bytes]
