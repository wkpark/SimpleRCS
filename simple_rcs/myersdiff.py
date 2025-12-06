# This is free and unencumbered software released into the public domain.
#
# Anyone is free to copy, modify, publish, use, compile, sell, or
# distribute this software, either in source code form or as a compiled
# binary, for any purpose, commercial or non-commercial, and by any
# means.
#
# In jurisdictions that recognize copyright laws, the author or authors
# of this software dedicate any and all copyright interest in the
# software to the public domain. We make this dedication for the benefit
# of the public at large and to the detriment of our heirs and
# successors. We intend this dedication to be an overt act of
# relinquishment in perpetuity of all present and future rights to this
# software under copyright law.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#
# For more information, please refer to <http://unlicense.org/>
#
# This implementation is based on the Myers diff algorithm.
# Optimized for use as a difflib-compatible matcher.

from collections.abc import Iterator


# Linked List Node for history trace: (tag, prev_node)
# tag: 'i' (insert), 'd' (delete)
# prev_node: reference to previous node tuple or None
# We ONLY track edits (insert/delete). Equal moves are implicit and replayed.

class MyersSequenceMatcher:
    """
    An implementation of the Myers diff algorithm.

    See http://www.xmailserver.org/diff2.pdf

    A difflib-compatible SequenceMatcher that uses the Myers Diff Algorithm.
    Optimized for finding the Longest Common Subsequence (LCS) producing minimal diffs.
    """

    def __init__(self, isjunk=None, a=None, b=None, autojunk=True) -> None:
        self.isjunk = isjunk
        self.a = a or []
        self.b = b or []
        self.autojunk = autojunk

    def set_seq1(self, a) -> None:
        self.a = a

    def set_seq2(self, b) -> None:
        self.b = b

    def get_opcodes(self) -> Iterator[tuple[str, int, int, int, int]]:
        """
        Return list of 5-tuples describing how to turn a into b.
        Each tuple is of the form (tag, i1, i2, j1, j2).
        """
        a = self.a
        b = self.b
        n = len(a)
        m = len(b)
        max_d = n + m
        offset = max_d

        # Frontier: list mapping k to (x, history_node)
        # k = x - y
        # We use a list instead of dict for performance.
        # Index = k + offset. Size = 2 * max_d + 1.
        frontier = [(-1, None)] * (2 * max_d + 1)

        # Initial Snake (d=0)
        x = 0
        y = 0
        while x < n and y < m and a[x] == b[y]:
            x += 1
            y += 1

        frontier[offset] = (x, None)

        if x >= n and y >= m:
            return self._history_to_opcodes(None)

        # Myers Algorithm Loop
        # We start from d=1 because d=0 (initial diagonal) is pre-calculated.
        # Each step adds exactly one edit (insert or delete).
        for d in range(1, max_d + 1):
            # The range of k for depth d is [-d, d].
            # Step is 2 because k increments by 1 (del) or decrements by 1 (ins)
            # from the previous step's k which was (d-1) parity.
            for k in range(-d, d + 1, 2):
                idx_k = offset + k

                # Retrieve x values from previous step (d-1)
                # k-1 comes from delete (horizontal)
                # k+1 comes from insert (vertical)

                # Check bounds for list access (though mathematically safe within logic)
                # We use a sentinel (-1, None) for unreached states

                x_minus, h_minus = frontier[idx_k - 1]
                x_plus, h_plus = frontier[idx_k + 1]

                # Myers logic:
                # if k == -d or (k != d and x_minus < x_plus):
                #   vertical move (insert from b): x same, y increases
                # else:
                #   horizontal move (delete from a): x increases

                if k == -d or (k != d and x_minus < x_plus):
                    x = x_plus
                    history = ('i', h_plus)
                else:
                    x = x_minus + 1
                    history = ('d', h_minus)

                # Snake: Extension for matching lines (Diagonal moves)
                # y = x - k
                y = x - k

                while x < n and y < m and a[x] == b[y]:
                    x += 1
                    y += 1
                    # Diagonal moves are not recorded in history to save memory/time.
                    # They will be inferred during replay.

                frontier[idx_k] = (x, history)

                if x >= n and y >= m:
                    return self._history_to_opcodes(history)

        return iter([])

    def _history_to_opcodes(self, history_node) -> Iterator[tuple[str, int, int, int, int]]:
        """
        Converts the linked-list history into difflib-style grouped opcodes.
        Replays the path from (0,0) to determine diagonal moves.
        """
        # 1. Linearize history (it is in reverse order)
        ops = []
        curr = history_node
        while curr is not None:
            tag, prev = curr
            ops.append(tag)
            curr = prev
        ops.reverse()

        a = self.a
        b = self.b
        n = len(a)
        m = len(b)

        i = 0
        j = 0

        # We buffer changes to merge adjacent insert/delete into 'replace'
        diff_start_i = 0
        diff_start_j = 0

        # Helper to emit pending diffs
        def emit_diff(end_i, end_j):
            nonlocal diff_start_i, diff_start_j
            if diff_start_i < end_i and diff_start_j < end_j:
                yield ('replace', diff_start_i, end_i, diff_start_j, end_j)
            elif diff_start_i < end_i:
                yield ('delete', diff_start_i, end_i, diff_start_j, end_j)
            elif diff_start_j < end_j:
                yield ('insert', diff_start_i, end_i, diff_start_j, end_j)
            diff_start_i = end_i
            diff_start_j = end_j

        # Iterator for ops
        ops_iter = iter(ops)

        while i < n or j < m:
            # 1. Diagonal Slide (Equal)
            start_i, start_j = i, j
            while i < n and j < m and a[i] == b[j]:
                i += 1
                j += 1

            # If we moved diagonally, we have an 'equal' block.
            if i > start_i:
                # Flush any pending edits before the equal block
                yield from emit_diff(start_i, start_j)

                # Emit the equal block
                yield ('equal', start_i, i, start_j, j)

                # Reset diff start points to after the equal block
                diff_start_i = i
                diff_start_j = j

            # 2. Edit Operation
            # If we are not at the end, there must be an edit op unless we finished matching via diagonal
            if i >= n and j >= m:
                break

            try:
                op = next(ops_iter)
            except StopIteration:
                # Should not happen if path is correct and logic matches
                break

            if op == 'd':
                i += 1
            elif op == 'i':
                j += 1

            # We don't emit immediately. We continue loop to see if more edits follow or a diagonal.

        # Flush any remaining edits at the end
        yield from emit_diff(i, j)
