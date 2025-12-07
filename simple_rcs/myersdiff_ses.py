# based off https://www.ioplex.com/~miallen/libmba/dl/src/diff.c
# License: MIT
#
# This is an adaptation of the C-based Myers diff algorithm from libmba
# into a Python class with a difflib-compatible interface.
# It does not use hashing and operates on in-memory sequences.
import os
from collections import namedtuple
from collections.abc import Iterator
from typing import BinaryIO


# Operation constants
DIFF_DELETE = 1
DIFF_INSERT = 2
DIFF_MATCH = 3

# Internal representation of an edit
DiffEdit = namedtuple('DiffEdit', ['op', 'off', 'len'])

MiddleSnake = namedtuple('MiddleSnake', ['x', 'y', 'u', 'v'])


class MyersSequenceMatcher:
    """
    An in-memory sequence matcher based on the C-to-Python port of the
    libxL Myers diff algorithm. It finds the shortest edit script (SES).

    This matcher is designed to be memory-efficient for the algorithm's
    internal state, even for sequences with many lines.
    """

    def __init__(self, isjunk=None, a=None, b=None, autojunk=True):
        """
        Initializes the matcher with sequences a and b.
        `isjunk` and `autojunk` are ignored but kept for API compatibility.
        """
        self.a = self.b = None
        self.opcodes = None
        self.set_seqs(a or [], b or [])

    def set_seqs(self, a, b):
        """Sets the two sequences to be compared."""
        self.set_seq1(a)
        self.set_seq2(b)

    def set_seq1(self, a):
        """Sets the first sequence."""
        if a is self.a:
            return
        self.a = a
        self.opcodes = None

    def set_seq2(self, b):
        """Sets the second sequence."""
        if b is self.b:
            return
        self.b = b
        self.opcodes = None

    def get_opcodes(self) -> Iterator[tuple[str, int, int, int, int]]:
        """
        Calculates and returns an iterator for the opcodes.
        The result is cached.
        """
        if self.opcodes:
            yield from self.opcodes
            return

        # The algorithm internally builds a list of edits. We then convert
        # this list into a final list of opcodes for caching and yielding.
        # This is a trade-off for caching the generator's results.
        opcodes_list = list(self._calculate_opcodes())
        self.opcodes = opcodes_list
        yield from self.opcodes

    def _calculate_opcodes(self) -> Iterator[tuple[str, int, int, int, int]]:
        """
        Performs the diff calculation, then converts the resulting SES
        into a list of matching blocks, and finally generates standard
        difflib-style opcodes from those blocks.
        """
        ses_results = []
        sn_output = [0]  # Use a list to simulate an out-parameter

        # Start the diff process which populates ses_results
        self._diff(
            a=self.a, aoff=0, n=len(self.a),
            b=self.b, boff=0, m=len(self.b),
            ses_results=ses_results, sn_output=sn_output,
        )

        # --- Step 1: Convert SES to matching_blocks ---
        matching_blocks = []
        a_pos, b_pos = 0, 0
        for edit in ses_results:
            if edit.op == DIFF_MATCH:
                if edit.len > 0:
                    # The SES 'off' is relative to 'a', but we track 'b' position manually
                    matching_blocks.append((edit.off, b_pos, edit.len))
                a_pos = edit.off + edit.len
                b_pos += edit.len
            elif edit.op == DIFF_INSERT:
                b_pos += edit.len
            elif edit.op == DIFF_DELETE:
                a_pos = edit.off + edit.len

        matching_blocks.append((len(self.a), len(self.b), 0))

        # --- Step 2: Generate opcodes from matching_blocks (standard difflib logic) ---
        i, j = 0, 0
        for ai, bj, size in matching_blocks:
            tag = ''
            if i < ai and j < bj:
                tag = 'replace'
            elif i < ai:
                tag = 'delete'
            elif j < bj:
                tag = 'insert'

            if tag:
                yield (tag, i, ai, j, bj)

            i, j = ai + size, bj + size
            if size:
                yield ('equal', ai, i, bj, j)

    # --- Core algorithm adapted from myers_x.py ---

    def _setv(self, buf, k, r, val):
        """Helper to set a value in the 'V' array (implemented as a dict)."""
        # C code's packing logic: k is [-N, N], r is [0, 1] or [2, 3]
        if k <= 0:
            j = -k * 4 + r
        else:
            j = k * 4 + (r - 2)
        buf[j] = val

    def _v(self, buf, k, r):
        """Helper to get a value from the 'V' array."""
        if k <= 0:
            j = -k * 4 + r
        else:
            j = k * 4 + (r - 2)
        return buf.get(j, 0)

    def _find_middle_snake(self, a, aoff, n, b, boff, m, buf):
        """Finds the middle snake in the Myers diff algorithm."""
        delta = n - m
        odd = delta & 1
        mid = (n + m) // 2
        mid += odd

        # Setup V arrays (forward and reverse)
        self._setv(buf, 1, 0, 0)
        self._setv(buf, delta - 1, 1, n)

        for d in range(mid + 1):
            # Forward search
            for k in range(d, -d - 1, -2):
                if k == -d or (k != d and self._v(buf, k - 1, 0) < self._v(buf, k + 1, 0)):
                    x = self._v(buf, k + 1, 0)
                else:
                    x = self._v(buf, k - 1, 0) + 1
                y = x - k

                ms_x, ms_y = x, y
                while x < n and y < m and a[aoff + x] == b[boff + y]:
                    x += 1
                    y += 1
                self._setv(buf, k, 0, x)

                if odd and (delta - d + 1) <= k <= (delta + d - 1) and x >= self._v(buf, k, 1):
                    return 2 * d - 1, MiddleSnake(ms_x, ms_y, x, y)

            # Reverse search
            for k_rev in range(d, -d - 1, -2):
                kr = delta + k_rev
                if k_rev == d or (k_rev != -d and self._v(buf, kr - 1, 1) < self._v(buf, kr + 1, 1)):
                    x = self._v(buf, kr - 1, 1)
                else:
                    x = self._v(buf, kr + 1, 1) - 1
                y = x - kr

                ms_u, ms_v = x, y
                while x > 0 and y > 0 and a[aoff + x - 1] == b[boff + y - 1]:
                    x -= 1
                    y -= 1
                self._setv(buf, kr, 1, x)

                if not odd and -d <= kr <= d and x <= self._v(buf, kr, 0):
                    return 2 * d, MiddleSnake(x, y, ms_u, ms_v)
        return -1, None

    def _edit(self, ses_results, si_ref, op, off, length):
        """Records an edit or merges it with the previous one."""
        if length == 0:
            return

        si = si_ref[0]
        if si > 0 and ses_results[si - 1].op == op and ses_results[si - 1].off + ses_results[si - 1].len == off:
            # Merge with previous edit if it's of the same type and contiguous
            prev = ses_results[si - 1]
            ses_results[si - 1] = prev._replace(len=prev.len + length)
        else:
            # Add new edit, correctly storing the offset
            ses_results.append(DiffEdit(op, off, length))
            si_ref[0] += 1

    def _ses(self, a, aoff, n, b, boff, m, ses_results, si_ref, buf):
        """The recursive part of the algorithm to find the SES."""
        if n == 0:
            self._edit(ses_results, si_ref, DIFF_INSERT, boff, m)
            return m
        if m == 0:
            self._edit(ses_results, si_ref, DIFF_DELETE, aoff, n)
            return n

        d, ms = self._find_middle_snake(a, aoff, n, b, boff, m, buf)
        if d == -1: return -1
        if d > 1 or (ms.x != ms.u and ms.y != ms.v):
            if self._ses(a, aoff, ms.x, b, boff, ms.y, ses_results, si_ref, buf) == -1:
                return -1

            match_len = ms.u - ms.x
            self._edit(ses_results, si_ref, DIFF_MATCH, aoff + ms.x, match_len)

            if self._ses(a, aoff + ms.u, n - ms.u, b, boff + ms.v, m - ms.v, ses_results, si_ref, buf) == -1:
                return -1
        elif m > n:
            self._edit(ses_results, si_ref, DIFF_INSERT, boff, 1)
            self._edit(ses_results, si_ref, DIFF_MATCH, aoff, n)
        else:
            self._edit(ses_results, si_ref, DIFF_DELETE, aoff, 1)
            self._edit(ses_results, si_ref, DIFF_MATCH, aoff + 1, m)
        return d

    def _diff(self, a, aoff, n, b, boff, m, ses_results, sn_output):
        """Main entry point for the diff algorithm."""
        buf = {}
        si_ref = [0] # Use a list to simulate a mutable integer reference

        # 1. Handle common prefix
        x = 0
        while x < n and x < m and a[aoff + x] == b[boff + x]:
            x += 1
        self._edit(ses_results, si_ref, DIFF_MATCH, aoff, x)

        # 2. Handle common suffix
        x_n, x_m = n, m
        while x_n > x and x_m > x and a[aoff + x_n - 1] == b[boff + x_m - 1]:
            x_n -= 1
            x_m -= 1

        # 3. Compute SES for the middle part
        d = self._ses(a, aoff + x, x_n - x, b, boff + x, x_m - x, ses_results, si_ref, buf)
        if d == -1:
            return -1

        # 4. Add common suffix
        suffix_len = n - x_n
        self._edit(ses_results, si_ref, DIFF_MATCH, aoff + x_n, suffix_len)

        if sn_output is not None:
            sn_output[0] = si_ref[0]

        return d


class MyersStreamSequenceMatcher:
    """
    A memory-efficient, stream-based sequence matcher that uses an optimized
    Myers diff algorithm (based on a C port from libxL) on a sequence of hashes.
    """

    def __init__(self, a_stream: BinaryIO, b_stream: BinaryIO, chunk_size: int | None = None) -> None:
        self.a_stream = a_stream
        self.b_stream = b_stream
        self.chunk_size = chunk_size

        self.a_offsets: list[int] = []
        self.b_offsets: list[int] = []

        self.opcodes = None

        self.a = self._index_stream(self.a_stream, self.a_offsets)
        self.b = self._index_stream(self.b_stream, self.b_offsets)

    def _index_stream(self, stream: BinaryIO, offset_list: list) -> list[int]:
        """Builds a list of chunk/line hashes and records their byte offsets."""
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
            content_to_hash = chunk if self.chunk_size else chunk.rstrip(b'\r\n')
            hash_list.append(hash(content_to_hash))
        return hash_list

    def _get_byte_offset(self, offsets: list[int], idx: int, is_end: bool = False, stream: BinaryIO | None = None) -> int:
        """Gets the byte offset for a given chunk/line index."""
        if idx < len(offsets):
            return offsets[idx]
        if is_end and stream:
            stream.seek(0, os.SEEK_END)
            return stream.tell()
        if offsets:
            return offsets[-1]
        return 0

    def get_opcodes(self) -> Iterator[tuple[str, int, int, int, int]]:
        """
        Public method to get opcodes. It caches the result.
        """
        if self.opcodes:
            yield from self.opcodes
            return

        opcodes_list = list(self._calculate_opcodes())
        self.opcodes = opcodes_list
        yield from self.opcodes

    def _calculate_opcodes(self) -> Iterator[tuple[str, int, int, int, int]]:
        """
        Performs the diff calculation on the hash lists, then translates the
        resulting SES into standard difflib-style opcodes.
        """
        ses_results = []
        sn_output = [0]

        self._diff(
            a=self.a, aoff=0, n=len(self.a),
            b=self.b, boff=0, m=len(self.b),
            ses_results=ses_results, sn_output=sn_output
        )

        matching_blocks = []
        a_idx, b_idx = 0, 0
        for edit in ses_results:
            op = edit.op
            length = edit.len
            if op == DIFF_MATCH:
                if length > 0:
                    matching_blocks.append((a_idx, b_idx, length))
                a_idx += length
                b_idx += length
            elif op == DIFF_DELETE:
                a_idx += length
            elif op == DIFF_INSERT:
                b_idx += length

        matching_blocks.append((len(self.a), len(self.b), 0))

        i, j = 0, 0
        for ai, bj, size in matching_blocks:
            tag = ''
            if i < ai and j < bj:
                tag = 'replace'
            elif i < ai:
                tag = 'delete'
            elif j < bj:
                tag = 'insert'

            if tag:
                if self.chunk_size is None:
                    yield (tag, i, ai, j, bj)
                else:
                    a_start = self._get_byte_offset(self.a_offsets, i)
                    a_end = self._get_byte_offset(self.a_offsets, ai, is_end=True, stream=self.a_stream)
                    b_start = self._get_byte_offset(self.b_offsets, j)
                    b_end = self._get_byte_offset(self.b_offsets, bj, is_end=True, stream=self.b_stream)
                    yield (tag, a_start, a_end, b_start, b_end)

            i, j = ai + size, bj + size
            if size:
                if self.chunk_size is None:
                    yield ('equal', ai, i, bj, j)
                else:
                    a_start = self._get_byte_offset(self.a_offsets, ai)
                    a_end = self._get_byte_offset(self.a_offsets, i, is_end=True, stream=self.a_stream)
                    b_start = self._get_byte_offset(self.b_offsets, bj)
                    b_end = self._get_byte_offset(self.b_offsets, j, is_end=True, stream=self.b_stream)
                    yield ('equal', a_start, a_end, b_start, b_end)

    def _setv(self, buf, k, r, val):
        if k <= 0: j = -k * 4 + r
        else: j = k * 4 + (r - 2)
        buf[j] = val

    def _v(self, buf, k, r):
        if k <= 0: j = -k * 4 + r
        else: j = k * 4 + (r - 2)
        return buf.get(j, 0)

    def _find_middle_snake(self, a, aoff, n, b, boff, m, buf):
        delta = n - m
        odd = delta & 1
        mid = (n + m) // 2
        mid += odd
        self._setv(buf, 1, 0, 0)
        self._setv(buf, delta - 1, 1, n)
        for d in range(mid + 1):
            for k in range(d, -d - 1, -2):
                if k == -d or (k != d and self._v(buf, k - 1, 0) < self._v(buf, k + 1, 0)):
                    x = self._v(buf, k + 1, 0)
                else:
                    x = self._v(buf, k - 1, 0) + 1
                y = x - k
                ms_x, ms_y = x, y
                while x < n and y < m and a[aoff + x] == b[boff + y]:
                    x += 1
                    y += 1
                self._setv(buf, k, 0, x)
                if odd and (delta - d + 1) <= k <= (delta + d - 1) and x >= self._v(buf, k, 1):
                    return 2 * d - 1, MiddleSnake(ms_x, ms_y, x, y)
            for k_rev in range(d, -d - 1, -2):
                kr = delta + k_rev
                if k_rev == d or (k_rev != -d and self._v(buf, kr - 1, 1) < self._v(buf, kr + 1, 1)):
                    x = self._v(buf, kr - 1, 1)
                else:
                    x = self._v(buf, kr + 1, 1) - 1
                y = x - kr
                ms_u, ms_v = x, y
                while x > 0 and y > 0 and a[aoff + x - 1] == b[boff + y - 1]:
                    x -= 1
                    y -= 1
                self._setv(buf, kr, 1, x)
                if not odd and -d <= kr <= d and x <= self._v(buf, kr, 0):
                    return 2 * d, MiddleSnake(x, y, ms_u, ms_v)
        return -1, None

    def _edit(self, ses_results, si_ref, op, off, length):
        if length == 0: return
        si = si_ref[0]
        if si > 0 and ses_results[si - 1].op == op and ses_results[si - 1].off + ses_results[si - 1].len == off:
            prev = ses_results[si - 1]
            ses_results[si - 1] = prev._replace(len=prev.len + length)
        else:
            ses_results.append(DiffEdit(op, off, length))
            si_ref[0] += 1

    def _ses(self, a, aoff, n, b, boff, m, ses_results, si_ref, buf):
        if n == 0:
            self._edit(ses_results, si_ref, DIFF_INSERT, boff, m)
            return m
        if m == 0:
            self._edit(ses_results, si_ref, DIFF_DELETE, aoff, n)
            return n
        d, ms = self._find_middle_snake(a, aoff, n, b, boff, m, buf)
        if d == -1: return -1
        if d > 1 or (ms.x != ms.u and ms.y != ms.v):
            if self._ses(a, aoff, ms.x, b, boff, ms.y, ses_results, si_ref, buf) == -1: return -1
            match_len = ms.u - ms.x
            self._edit(ses_results, si_ref, DIFF_MATCH, aoff + ms.x, match_len)
            if self._ses(a, aoff + ms.u, n - ms.u, b, boff + ms.v, m - ms.v, ses_results, si_ref, buf) == -1: return -1
        elif m > n:
            self._edit(ses_results, si_ref, DIFF_INSERT, boff, 1)
            self._edit(ses_results, si_ref, DIFF_MATCH, aoff, n)
        else:
            self._edit(ses_results, si_ref, DIFF_DELETE, aoff, 1)
            self._edit(ses_results, si_ref, DIFF_MATCH, aoff + 1, m)
        return d

    def _diff(self, a, aoff, n, b, boff, m, ses_results, sn_output):
        buf = {}
        si_ref = [0]
        x = 0
        while x < n and x < m and a[aoff + x] == b[boff + x]:
            x += 1
        self._edit(ses_results, si_ref, DIFF_MATCH, aoff, x)
        x_n, x_m = n, m
        while x_n > x and x_m > x and a[aoff + x_n - 1] == b[boff + x_m - 1]:
            x_n -= 1
            x_m -= 1
        d = self._ses(a, aoff + x, x_n - x, b, boff + x, x_m - x, ses_results, si_ref, buf)
        if d == -1: return -1
        suffix_len = n - x_n
        self._edit(ses_results, si_ref, DIFF_MATCH, aoff + x_n, suffix_len)
        if sn_output is not None: sn_output[0] = si_ref[0]
        return d
