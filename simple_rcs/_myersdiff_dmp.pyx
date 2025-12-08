# distutils: language=c
# cython: language_level=3
#
# based off https://github.com/gritzko/myers-diff/blob/master/dmp_diff.hpp
# original site: https://github.com/gritzko/myers-diff/
# original author: Victor @gritzko Grishchenko
# original license: Apache-2.0 license
#
# It provides a memory-efficient and fast diffing algorithm with several
# optimizations, adapted to a difflib-compatible opcode interface.
# It uses C-level type declarations for a significant performance boost
# in loops and recursive calls.

from collections.abc import Iterator


cdef class MyersSequenceMatcher:
    # Declare instance variables with C types for fast access
    cdef public list a, b
    cdef public object opcodes

    def __init__(self, isjunk=None, a=None, b=None, autojunk=True):
        self.a = a or []
        self.b = b or []
        self.opcodes = None

    # Public methods remain as 'def' to be callable from Python
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
        if self.opcodes is not None:
            for op in self.opcodes:
                yield op
            return

        diffs = self._diff_main(self.a, 0, len(self.a), self.b, 0, len(self.b))

        opcodes_list = []
        i, j = 0, 0

        # This part remains in Python as it deals with Python objects (tuples, lists)
        # and is not the main performance bottleneck.
        for op, chunk in diffs:
            chunk_len = len(chunk)
            if op == 'equal':
                opcodes_list.append(('equal', i, i + chunk_len, j, j + chunk_len))
                i += chunk_len
                j += chunk_len
            elif op == 'delete':
                opcodes_list.append(('delete', i, i + chunk_len, j, j))
                i += chunk_len
            elif op == 'insert':
                opcodes_list.append(('insert', i, i, j, j + chunk_len))
                j += chunk_len

        merged_opcodes = []
        idx = 0
        while idx < len(opcodes_list):
            current_op = opcodes_list[idx]
            if idx + 1 < len(opcodes_list):
                next_op = opcodes_list[idx+1]
                if current_op[0] == 'delete' and next_op[0] == 'insert':
                    merged_opcodes.append(('replace', current_op[1], current_op[2], next_op[3], next_op[4]))
                    idx += 2
                    continue
            merged_opcodes.append(current_op)
            idx += 1

        self.opcodes = merged_opcodes
        for op in merged_opcodes:
            yield op

    # --- Core algorithm methods defined as `cdef` for C-level speed ---

    cdef list _diff_main(self, list a, int a_off, int a_len, list b, int b_off, int b_len):
        cdef int i
        cdef bint is_equal

        if a_len == 0:
            return [('insert', b[b_off : b_off + b_len])] if b_len > 0 else []
        if b_len == 0:
            return [('delete', a[a_off : a_off + a_len])] if a_len > 0 else []

        is_equal = a_len == b_len
        if is_equal:
            for i in range(a_len):
                if a[a_off + i] != b[b_off + i]:
                    is_equal = False
                    break
            if is_equal:
                return [('equal', a[a_off : a_off + a_len])]

        cdef int common_len = self._diff_commonPrefix(a, a_off, a_len, b, b_off, b_len)
        cdef list common_prefix = a[a_off : a_off + common_len] if common_len > 0 else []
        a_off += common_len; a_len -= common_len
        b_off += common_len; b_len -= common_len

        cdef int common_len_suffix = self._diff_commonSuffix(a, a_off, a_len, b, b_off, b_len)
        cdef list common_suffix = a[a_off + a_len - common_len_suffix : a_off + a_len] if common_len_suffix > 0 else []
        a_len -= common_len_suffix
        b_len -= common_len_suffix

        cdef list diffs = self._diff_compute(a, a_off, a_len, b, b_off, b_len)

        if common_prefix:
            diffs.insert(0, ('equal', common_prefix))
        if common_suffix:
            diffs.append(('equal', common_suffix))

        return diffs

    cdef list _diff_compute(self, list a, int a_off, int a_len, list b, int b_off, int b_len):
        if not a_len:
            return [('insert', b[b_off : b_off + b_len])]
        if not b_len:
            return [('delete', a[a_off : a_off + a_len])]

        # This heuristic part is simplified for the Cython port, as the main
        # performance gain comes from the bisect algorithm.
        if min(a_len, b_len) == 1:
            return [('delete', a[a_off:a_off+a_len]), ('insert', b[b_off:b_off+b_len])]

        return self._diff_bisect(a, a_off, a_len, b, b_off, b_len)

    cdef list _diff_bisect(self, list a, int a_off, int a_len, list b, int b_off, int b_len):
        cdef int max_d, v_offset, v_len, delta, d, k1, k1_offset, x1, y1
        cdef int k2, k2_offset, x2, y2
        cdef bint front

        max_d = (a_len + b_len + 1) // 2
        v_offset = max_d
        v_len = 2 * max_d

        # Using Python lists is fine here as the logic inside the C loops is fast.
        # For max performance, C arrays (`int*`) could be used with `malloc`.
        v1 = [-1] * v_len
        v2 = [-1] * v_len
        v1[v_offset + 1] = 0
        v2[v_offset + 1] = 0

        delta = a_len - b_len
        front = (delta % 2 != 0)

        for d in range(max_d):
            # Forward pass
            for k1 in range(-d, d + 1, 2):
                k1_offset = v_offset + k1
                if k1 == -d or (k1 != d and v1[k1_offset - 1] < v1[k1_offset + 1]):
                    x1 = v1[k1_offset + 1]
                else:
                    x1 = v1[k1_offset - 1] + 1
                y1 = x1 - k1
                while x1 < a_len and y1 < b_len and a[a_off + x1] == b[b_off + y1]:
                    x1 += 1
                    y1 += 1
                v1[k1_offset] = x1
                if front:
                    k2_offset = v_offset + delta - k1
                    if 0 <= k2_offset < v_len and v2[k2_offset] != -1:
                        x2 = a_len - v2[k2_offset]
                        if x1 >= x2:
                            return self._diff_bisectSplit(a, a_off, a_len, b, b_off, b_len, x1, y1)

            # Backward pass
            for k2 in range(-d, d + 1, 2):
                k2_offset = v_offset + k2
                if k2 == -d or (k2 != d and v2[k2_offset - 1] < v2[k2_offset + 1]):
                    x2 = v2[k2_offset + 1]
                else:
                    x2 = v2[k2_offset - 1] + 1
                y2 = x2 - k2
                while x2 < a_len and y2 < b_len and a[a_off + a_len - x2 - 1] == b[b_off + b_len - y2 - 1]:
                    x2 += 1
                    y2 += 1
                v2[k2_offset] = x2
                if not front:
                    k1_offset = v_offset + delta - k2
                    if 0 <= k1_offset < v_len and v1[k1_offset] != -1:
                        x1 = v1[k1_offset]
                        x2 = a_len - x2
                        if x1 >= x2:
                            # Overlap detected. We have the split point (x1, y1) from
                            # the forward path. We need to calculate y1 corresponding to x1.
                            k1 = delta - k2
                            y1 = x1 - k1
                            return self._diff_bisectSplit(a, a_off, a_len, b, b_off, b_len, x1, y1)

        return [('delete', a[a_off:a_off+a_len]), ('insert', b[b_off:b_off+b_len])]

    cdef list _diff_bisectSplit(self, list a, int a_off, int a_len, list b, int b_off, int b_len, int x, int y):
        cdef list diffs, diffsb
        diffs = self._diff_main(a, a_off, x, b, b_off, y)
        diffsb = self._diff_main(a, a_off + x, a_len - x, b, b_off + y, b_len - y)
        diffs.extend(diffsb)
        return diffs

    cdef int _diff_commonPrefix(self, list a, int a_off, int a_len, list b, int b_off, int b_len):
        cdef int n = min(a_len, b_len)
        cdef int i
        for i in range(n):
            if a[a_off + i] != b[b_off + i]:
                return i
        return n

    cdef int _diff_commonSuffix(self, list a, int a_off, int a_len, list b, int b_off, int b_len):
        cdef int n = min(a_len, b_len)
        cdef int i
        for i in range(1, n + 1):
            if a[a_off + a_len - i] != b[b_off + b_len - i]:
                return i - 1
        return n
