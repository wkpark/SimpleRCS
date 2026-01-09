# distutils: language=c
# cython: language_level=3
#
# based off https://www.ioplex.com/~miallen/libmba/dl/src/diff.c
# License: MIT
#
# This is an adaptation of the C-based Myers diff algorithm from libmba
# into a Python class with a difflib-compatible interface.
#
# This is a Cython-optimized version of myers
# It adds C-level type declarations for significant performance improvement.

import os
from collections import namedtuple
from collections.abc import Iterator
from typing import BinaryIO


# Using Cython's cdef to define C-level constants and structs
cdef int DIFF_DELETE = 1
cdef int DIFF_INSERT = 2
cdef int DIFF_MATCH = 3

# namedtuples are Python objects, so we keep them as they are.
# For extreme optimization, one could use C-structs.
DiffEdit = namedtuple('DiffEdit', ['op', 'off', 'len'])
MiddleSnake = namedtuple('MiddleSnake', ['x', 'y', 'u', 'v'])

# cdef class for faster instance variable access
cdef class MyersSequenceMatcher:
    cdef public list a, b
    cdef public object opcodes

    def __init__(self, isjunk=None, a=None, b=None, autojunk=True):
        self.a = a or []
        self.b = b or []
        self.opcodes = None

    def set_seqs(self, a, b):
        self.set_seq1(a)
        self.set_seq2(b)

    def set_seq1(self, list a):
        if a is self.a:
            return
        self.a = a
        self.opcodes = None

    def set_seq2(self, list b):
        if b is self.b:
            return
        self.b = b
        self.opcodes = None

    def get_opcodes(self) -> Iterator[tuple[str, int, int, int, int]]:
        if self.opcodes:
            for op in self.opcodes:
                yield op
            return

        opcodes_list = list(self._calculate_opcodes())
        self.opcodes = opcodes_list
        for op in opcodes_list:
            yield op

    def _calculate_opcodes(self) -> Iterator[tuple[str, int, int, int, int]]:
        cdef list ses_results = []
        cdef list sn_output = [0]

        self._diff(
            self.a, 0, len(self.a),
            self.b, 0, len(self.b),
            ses_results, sn_output,
        )

        cdef int a_idx, b_idx, i, j, ai, bj, size
        cdef str tag
        cdef object edit
        matching_blocks = []
        a_idx, b_idx = 0, 0
        for edit in ses_results:
            if edit.op == DIFF_MATCH:
                if edit.len > 0:
                    matching_blocks.append((a_idx, b_idx, edit.len))
                a_idx += edit.len
                b_idx += edit.len
            elif edit.op == DIFF_DELETE:
                a_idx += edit.len
            elif edit.op == DIFF_INSERT:
                b_idx += edit.len

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
                yield (tag, i, ai, j, bj)

            i, j = ai + size, bj + size
            if size:
                yield ('equal', ai, i, bj, j)

    cdef inline void _setv(self, list buf, int k, int r, int val):
        cdef int j
        if k <= 0:
            j = -k * 4 + r
        else:
            j = k * 4 + (r - 2)

        if j >= len(buf):
            buf.extend([0] * (j - len(buf) + 1))
        buf[j] = val

    cdef inline int _v(self, list buf, int k, int r):
        cdef int j
        if k <= 0:
            j = -k * 4 + r
        else:
            j = k * 4 + (r - 2)

        if j >= len(buf):
            return 0
        return buf[j]

    cdef tuple _find_middle_snake(self, list a, int aoff, int n, list b, int boff, int m, list buf):
        cdef int delta, odd, mid, d, k, x, y, ms_x, ms_y, k_rev, kr, ms_u, ms_v
        delta = n - m
        odd = delta & 1
        mid = (n + m) // 2
        mid += odd


        self._setv(buf, 1, 0, 0)
        self._setv(buf, delta - 1, 1, n)

        for d in range(mid + 1):
            for k in range(-d, d + 1, 2):
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

                if odd and (delta - (d - 1)) <= k <= (delta + (d - 1)) and x >= self._v(buf, k, 1):
                    return 2 * d - 1, MiddleSnake(ms_x, ms_y, x, y)

            for k_rev in range(-d, d + 1, 2):
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

    cdef inline void _edit(self, list ses_results, list si_ref, int op, int off, int length):
        cdef int si
        cdef object prev
        if length == 0:
            return

        si = si_ref[0]
        if si > 0 and ses_results[si - 1].op == op and ses_results[si - 1].off + ses_results[si - 1].len == off:
            prev = ses_results[si - 1]
            ses_results[si - 1] = prev._replace(len=prev.len + length)
        else:
            ses_results.append(DiffEdit(op, off, length))
            si_ref[0] += 1

    cdef int _ses(self, list a, int aoff, int n, list b, int boff, int m, list ses_results, list si_ref, list buf):
        cdef int d, match_len
        cdef object ms
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

    cdef int _diff(self, list a, int aoff, int n, list b, int boff, int m, list ses_results, list sn_output):
        cdef list buf = []
        cdef list si_ref = [0]
        cdef int x, x_n, x_m, d, suffix_len

        x = 0
        while x < n and x < m and a[aoff + x] == b[boff + x]:
            x += 1
        self._edit(ses_results, si_ref, DIFF_MATCH, aoff, x)

        x_n, x_m = n, m
        while x_n > x and x_m > x and a[aoff + x_n - 1] == b[boff + x_m - 1]:
            x_n -= 1
            x_m -= 1

        d = self._ses(a, aoff + x, x_n - x, b, boff + x, x_m - x, ses_results, si_ref, buf)
        if d == -1:
            return -1

        suffix_len = n - x_n
        self._edit(ses_results, si_ref, DIFF_MATCH, aoff + x_n, suffix_len)

        if sn_output is not None:
            sn_output[0] = si_ref[0]

        return d
