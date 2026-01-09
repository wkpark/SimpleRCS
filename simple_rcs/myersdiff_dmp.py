# This is a Python port of the C++ Diff implementation
# from https://github.com/gritzko/myers-diff/blob/master/dmp_diff.hpp
# which is based on Neil Fraser's diff-match-patch library
# but without the Match and Patch part.
#
# It provides a memory-efficient and fast diffing algorithm with several
# optimizations, adapted to a difflib-compatible opcode interface.
# This version has been refactored to pass offsets instead of using
# list slicing in recursive calls to improve memory efficiency.
#
# original reference: https://github.com/google/diff-match-patch
# based off https://github.com/gritzko/myers-diff/blob/master/dmp_diff.hpp
# License: Licensed under the Apache License, Version 2.0 (the "License")

from collections.abc import Iterator


class MyersSequenceMatcher:
    """
    An in-memory sequence matcher based on the diff-match-patch algorithm.
    It uses the 'middle snake' strategy for performance and memory efficiency,
    and passes sequence offsets instead of slicing to avoid memory copies.
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
        The result is cached for subsequent calls.
        """
        if self.opcodes is not None:
            yield from self.opcodes
            return

        diffs = self._diff_main(self.a, 0, len(self.a), self.b, 0, len(self.b))

        opcodes_list = []
        i, j = 0, 0

        # Simple translation of SES to opcodes.
        # A more advanced version could merge adjacent delete/insert into replace.
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

        # Merge adjacent delete/insert into a single replace
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
        yield from self.opcodes

    # --- Core algorithm adapted from myer.cpp (diff-match-patch) ---

    def _diff_main(self, a, a_off, a_len, b, b_off, b_len):
        """
        Find the differences between two sub-sequences defined by offsets and lengths.
        """
        if a_len == 0:
            return [('insert', b[b_off : b_off + b_len])] if b_len > 0 else []
        if b_len == 0:
            return [('delete', a[a_off : a_off + a_len])] if a_len > 0 else []

        # Check for equality
        is_equal = a_len == b_len
        if is_equal:
            for i in range(a_len):
                if a[a_off + i] != b[b_off + i]:
                    is_equal = False
                    break
            if is_equal:
                return [('equal', a[a_off : a_off + a_len])]

        # Trim off common prefix
        common_len = self._diff_commonPrefix(a, a_off, a_len, b, b_off, b_len)
        common_prefix = a[a_off : a_off + common_len]
        a_off += common_len
        a_len -= common_len
        b_off += common_len
        b_len -= common_len

        # Trim off common suffix
        common_len_suffix = self._diff_commonSuffix(a, a_off, a_len, b, b_off, b_len)
        common_suffix = a[a_off + a_len - common_len_suffix : a_off + a_len]
        a_len -= common_len_suffix
        b_len -= common_len_suffix

        # Compute the diff on the middle block
        diffs = self._diff_compute(a, a_off, a_len, b, b_off, b_len)

        # Restore the prefix and suffix
        if common_prefix:
            diffs.insert(0, ('equal', common_prefix))
        if common_suffix:
            diffs.append(('equal', common_suffix))

        return diffs

    def _diff_compute(self, a, a_off, a_len, b, b_off, b_len):
        """
        Find the differences between two sub-sequences. Assumes no common prefix or suffix.
        """
        if not a_len:
            return [('insert', b[b_off : b_off + b_len])]
        if not b_len:
            return [('delete', a[a_off : a_off + a_len])]

        long_seq, long_off, long_len, short_seq, short_off, short_len = \
            (a, a_off, a_len, b, b_off, b_len) if a_len > b_len else (b, b_off, b_len, a, a_off, a_len)

        # Find if short sequence is a substring of long one
        for i in range(long_len - short_len + 1):
            is_match = True
            for j in range(short_len):
                if long_seq[long_off + i + j] != short_seq[short_off + j]:
                    is_match = False
                    break
            if is_match:
                op = 'delete' if a_len > b_len else 'insert'
                diffs = []
                if i > 0:
                    diffs.append((op, long_seq[long_off : long_off + i]))
                diffs.append(('equal', short_seq[short_off : short_off + short_len]))
                if i + short_len < long_len:
                    diffs.append((op, long_seq[long_off + i + short_len : long_off + long_len]))
                return diffs

        if min(a_len, b_len) == 1:
            # After the common prefix/suffix and substring checks, if one
            # sequence is of length 1, the diff is a simple delete/insert.
            return [('delete', a[a_off:a_off+a_len]), ('insert', b[b_off:b_off+b_len])]

        return self._diff_bisect(a, a_off, a_len, b, b_off, b_len)

    def _diff_bisect(self, a, a_off, a_len, b, b_off, b_len):
        """
        Find the 'middle snake' of a diff using offsets.
        """
        max_d = (a_len + b_len + 1) // 2
        v_offset = max_d
        v_len = 2 * max_d
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
                        y1 = v_offset + x1 - k1_offset
                        x2 = a_len - x2
                        if x1 >= x2:
                            return self._diff_bisectSplit(a, a_off, a_len, b, b_off, b_len, x1, y1)

        return [('delete', a[a_off:a_off+a_len]), ('insert', b[b_off:b_off+b_len])]

    def _diff_bisectSplit(self, a, a_off, a_len, b, b_off, b_len, x, y):
        """Given the location of the 'middle snake', split the diff and recurse."""
        diffs = self._diff_main(a, a_off, x, b, b_off, y)
        diffs.extend(self._diff_main(a, a_off + x, a_len - x, b, b_off + y, b_len - y))
        return diffs

    def _diff_commonPrefix(self, a, a_off, a_len, b, b_off, b_len):
        """Determine the common prefix of two sub-sequences."""
        n = min(a_len, b_len)
        for i in range(n):
            if a[a_off + i] != b[b_off + i]:
                return i
        return n

    def _diff_commonSuffix(self, a, a_off, a_len, b, b_off, b_len):
        """Determine the common suffix of two sub-sequences."""
        n = min(a_len, b_len)
        for i in range(1, n + 1):
            if a[a_off + a_len - i] != b[b_off + b_len - i]:
                return i - 1
        return n

