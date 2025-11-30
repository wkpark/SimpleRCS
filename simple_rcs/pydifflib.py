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

    def __init__(self, a_stream: BinaryIO, b_stream: BinaryIO, encoding: str = 'utf-8', chunk_size: int | None = None):
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

    def _index_stream(self, stream: BinaryIO, hash_list: list, offset_list: list, hash_map: dict[int, list[int]] | None):
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

    def get_opcodes(self) -> Iterator[tuple[str, int, int, int, int]]:
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

    def _refine_greedy(self, a: bytes, b: bytes, a_global_offset: int, b_global_offset: int) -> Iterator[tuple[str, int, int, int, int]]:
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

        # Adaptive Anchor Length: Short for small files, long for large files (cap at 32)
        MIN_MATCH_LEN = min(32, max(16, min(len_a, len_b) // 100))

        # Search Window Limit (prevent looking too far for false positives)
        MAX_SEARCH_DISTANCE = max(4096, min(len_a, len_b) // 4)

        while b_idx < len_b and a_idx < len_a:
            anchor_len = min(MIN_MATCH_LEN, len_b - b_idx)

            if anchor_len < MIN_MATCH_LEN // 2:
                break # Remaining part is too small to anchor reliably

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
            anchor_a_len = min(MIN_MATCH_LEN, len_a - a_idx)

            if anchor_a_len >= MIN_MATCH_LEN // 2:
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
            step = min(remaining, max(MIN_MATCH_LEN, remaining // 4))

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

    def _get_byte_offset(self, offsets, idx, is_end=False, stream=None):
        if idx < len(offsets):
            return offsets[idx]
        if is_end and offsets:
            stream.seek(0, os.SEEK_END)
            return stream.tell()
        return 0 # Should handle empty/start cases

    def get_matching_blocks(self) -> list[tuple[int, int, int]]:
        # Compatibility wrapper
        blocks = []
        for tag, i1, i2, j1, j2 in self.get_opcodes():
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
    Capable of handling shifted data (inserts/deletes) better than fixed chunking.

    Warning: This is pure Python, so it is computationally expensive (CPU bound).
    Use only when patch size optimization is critical and speed is secondary.
    """

    def __init__(self, a_stream: BinaryIO, b_stream: BinaryIO, chunk_size: int = 4096):
        self.a_stream = a_stream
        self.b_stream = b_stream
        self.chunk_size = chunk_size

        # 1. Index Old File (Signature Generation)
        self.a_map = self._index_file(self.a_stream)

    def _index_file(self, stream: BinaryIO) -> dict[int, list[int]]:
        """Reads file in fixed chunks and builds {adler32: [offsets]} map."""
        stream.seek(0)
        hash_map = defaultdict(list)
        offset = 0
        while True:
            chunk = stream.read(self.chunk_size)
            if not chunk:
                break

            # Use zlib.adler32 for fast initial hashing
            h = zlib.adler32(chunk) & 0xFFFFFFFF
            hash_map[h].append(offset)
            offset += len(chunk)
        return hash_map

    def get_opcodes(self) -> Iterator[tuple[str, int, int, int, int]]:
        """
        Scans B file using rolling hash to find matches in A.
        Yields (tag, i1, i2, j1, j2) byte offsets.
        """
        # Prepare B stream
        self.b_stream.seek(0)

        # We need a window buffer.
        # Reading byte-by-byte from disk is too slow.
        # Strategy: Read B in large blocks (e.g. 1MB) into memory buffer.

        B_BUFFER_SIZE = 1024 * 1024 * 4 # 4MB buffer
        b_pos_global = 0

        # Current window state
        window = bytearray()
        s1, s2 = _OFFS, 0 # Adler states

        # Pending literal (insert) buffer start
        pending_insert_start = 0

        # To handle stream reading logic easily, let's load B entirely if feasible?
        # Or manage complex buffering.
        # For simplicity in this implementation, we assume we can read B.
        # But to be "Stream" safe, we should handle buffering.

        # Simplified approach: Use a sliding window on a buffered stream reader.
        # But rotate needs the byte leaving.

        # Let's verify against Old.

        # Optimization: Don't implement full ring buffer in Python.
        # Read all of B if it fits in memory? (SimpleRCS usually handles < 100MB)
        # If we assume files fit in memory (or mapped), it's easier.
        # Let's proceed with reading B fully for now to demonstrate logic.

        self.b_stream.seek(0)
        b_data = self.b_stream.read() # Warning: Memory intensive
        len_b = len(b_data)

        i = 0 # Window start in B
        window_len = 0

        # Initial window fill
        first_chunk_len = min(len_b, self.chunk_size)
        if first_chunk_len > 0:
            window_len = first_chunk_len

            # Calculate initial Adler efficiently
            initial_val = zlib.adler32(b_data[0:window_len], 1) # 1 is default start

            # Extract s1, s2 from zlib result?
            # zlib.adler32 returns combined value.
            # We need s1, s2 for rotation.
            # Python's zlib doesn't expose s1/s2 directly easily.
            # We must compute s1, s2 manually or reverse engineer?
            # s1 = val & 0xFFFF, s2 = (val >> 16) & 0xFFFF

            h = initial_val & 0xFFFFFFFF
            s1 = h & 0xFFFF
            s2 = (h >> 16) & 0xFFFF

        else:
            # Handle empty file or chunk_size > len_b initially
            h = 1 # Default initial value for Adler-32
            s1, s2 = _OFFS, 0
            window_len = 0 # No window to fill

        pending_start = 0
        last_old_pos = 0 # Track old file position for seeks

        while i < len_b:
            # Check for match
            best_match_len = 0
            best_old_offset = -1

            # Only check if full window (or handle partial at EOF if we want to be fancy, but stick to fixed for now)
            if window_len == self.chunk_size:
                candidates = self.a_map.get(h)
                if candidates:
                    # Heuristic: Check a limited number of candidates to avoid O(N^2)
                    # We iterate backwards or forwards? Default dict list is usually insertion order (sequential).
                    # Maybe check latest (closest to current pos) first?
                    # Let's just check the first few.
                    check_count = 0
                    max_checks = 20

                    for old_offset in candidates:
                        check_count += 1
                        if check_count > max_checks:
                            break

                        # Verify strong match (memcmp) of the chunk first
                        self.a_stream.seek(old_offset)
                        old_chunk = self.a_stream.read(self.chunk_size)

                        if b_data[i : i + self.chunk_size] == old_chunk:
                            # Match found! Now try to EXTEND it.
                            current_len = self.chunk_size

                            # Read ahead in both streams
                            # Optimization: Read blocks instead of bytes for speed
                            # We can read until mismatch.

                            # Simple byte-by-byte extension (slow but correct)
                            # Or block extension.

                            # Limit extension to avoid reading whole file?
                            MAX_EXTEND = 1024 * 1024 # 1MB limit for sanity

                            # Check available bounds
                            # old stream can be read further.
                            # b_data is available in memory.

                            # We need to read old stream beyond old_offset + chunk_size
                            # Let's read a buffer.

                            extend_limit = min(len_b - (i + current_len), MAX_EXTEND)
                            if extend_limit > 0:
                                self.a_stream.seek(old_offset + current_len)
                                old_next = self.a_stream.read(extend_limit)
                                b_next = b_data[i + current_len : i + current_len + len(old_next)]

                                # Compare
                                # Python's sequence comparison is fast.
                                # Find common prefix length.
                                # os.path.commonprefix is for paths.
                                # We can use a loop or logic.

                                # Fast common prefix length:
                                k = 0
                                # Optimization: Compare in chunks?
                                # Let's keep it simple: if they are equal, great.
                                # If not, find where they differ.

                                if old_next == b_next:
                                    current_len += len(old_next)
                                else:
                                    # Find mismatch index
                                    # Binary search or linear scan?
                                    # Linear scan on mismatch
                                    min_len = min(len(old_next), len(b_next))
                                    for k in range(min_len):
                                        if old_next[k] != b_next[k]:
                                            break
                                    else:
                                        k = min_len
                                    current_len += k

                            if current_len > best_match_len:
                                best_match_len = current_len
                                best_old_offset = old_offset

                                # If we found a very long match, stop searching candidates.
                                if best_match_len > self.chunk_size * 4:
                                    break

            if best_match_len > 0:
                # 1. Yield pending inserts
                if i > pending_start:
                    yield ('insert', 0, 0, pending_start, i)

                # 2. Yield seek (delete) if needed
                seek_dist = best_old_offset - last_old_pos
                if seek_dist != 0:
                    yield ('delete', last_old_pos, best_old_offset, i, i)

                # 3. Yield match
                yield ('equal', best_old_offset, best_old_offset + best_match_len, i, i + best_match_len)

                # 4. Update state
                # Advance 'i' by the length of the matched block
                i += best_match_len
                pending_start = i
                last_old_pos = best_old_offset + best_match_len

                # Re-initialize rolling hash at new position
                if i + self.chunk_size <= len_b:
                    next_chunk = b_data[i : i + self.chunk_size]
                    h = zlib.adler32(next_chunk, 1) & 0xFFFFFFFF
                    s1 = h & 0xFFFF
                    s2 = (h >> 16) & 0xFFFF
                    window_len = self.chunk_size
                else:
                    # Near EOF, partial chunk, cannot form full window
                    window_len = len_b - i
                    # No need to re-initialize h, s1, s2 if window is partial/empty

                continue # Continue outer while loop

            # No match (or partial window). Roll 1 byte.
            # This block executes if 'best_match_len == 0' after checking candidates
            # or if 'window_len != self.chunk_size' (initial partial window or near EOF).
            if i + self.chunk_size < len_b:
                # We can slide the window forward by one byte
                out_byte = b_data[i]
                in_byte = b_data[i + self.chunk_size]

                # Update Adler-32 (Rotate)
                s1 = (s1 - out_byte + in_byte) % _BASE
                term1 = (self.chunk_size * out_byte) % _BASE
                s2 = (s2 - term1 + s1 - _OFFS) % _BASE

                h = (s2 << 16) | s1

                i += 1
                # window_len remains chunk_size as we are just sliding the window
            else:
                # Cannot slide anymore (EOF reached for window end)
                # Remaining bytes in B (from 'i' to 'len_b') are inserts.
                break

        # Flush any remaining inserts after the main loop finishes
        if pending_start < len_b:
            yield ('insert', 0, 0, pending_start, len_b)
