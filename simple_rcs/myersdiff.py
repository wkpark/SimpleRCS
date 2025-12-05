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
# tag: 'e' (equal), 'i' (insert), 'd' (delete)
# prev_node: reference to previous node tuple or None
# This avoids copying lists O(N) times inside the loop.

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

        # Frontier: {k: (x, history_node)}
        # k = x - y
        frontier = {1: (0, None)}

        final_history = None

        # Myers Algorithm Loop
        for d in range(max_d + 1):
            for k in range(-d, d + 1, 2):
                # Determine direction: down (delete) or right (insert)
                # We go down if k == -d (left edge)
                # Or if k != d and the x value of k-1 (up-left) is less than k+1 (down-right)
                # Wait, Myers logic:
                # If k == -d or (k != d and frontier[k-1].x < frontier[k+1].x):
                #   x = frontier[k+1].x  (Move Down/Vertical from k+1) -> x same, y increases -> Insert?
                #   Wait, x-y=k. If we come from k+1 (x_old - y_old = k+1).
                #   We want new k. x - (y+1) = x-y-1 = k. So moving vertical decreases k.
                #   Moving horizontal (x+1) increases k.

                # Correct Myers Logic:
                # if k == -d or (k != d and V[k-1] < V[k+1]):
                #    x = V[k+1]
                # else:
                #    x = V[k-1] + 1

                if k == -d or (k != d and frontier[k - 1][0] < frontier[k + 1][0]):
                    old_x, history = frontier[k + 1]
                    x = old_x
                    # Moved vertically (y increased). From k+1 to k.
                    # x is same, y increases.
                    # This corresponds to INSERT from b.
                    # Record action if not at start

                    # Note: We record the action that *led* to (x, y)
                    # Vertical move: Insert b[y-1]
                    # Must check y > 0 because y=0 means we are at start (or only deletes happened before)
                    if 0 < (x - k) <= m:
                         history = ('i', history)
                else:
                    old_x, history = frontier[k - 1]
                    x = old_x + 1
                    # Moved horizontally (x increased). From k-1 to k.
                    # This corresponds to DELETE from a.
                    if 0 < x <= n:
                        history = ('d', history)

                y = x - k

                # Snake: Extension for matching lines (Diagonal moves)
                while x < n and y < m and a[x] == b[y]:
                    x += 1
                    y += 1
                    history = ('e', history)

                if x >= n and y >= m:
                    final_history = history
                    break

                frontier[k] = (x, history)

            if final_history is not None:
                break

        return self._history_to_opcodes(final_history)

    def _history_to_opcodes(self, history_node) -> Iterator[tuple[str, int, int, int, int]]:
        """
        Converts the linked-list history into difflib-style grouped opcodes.
        """
        # 1. Linearize history (it is in reverse order)
        path = []
        curr = history_node
        while curr is not None:
            tag, prev = curr
            path.append(tag)
            curr = prev
        path.reverse()

        # 2. Convert path to opcodes
        # Path is a sequence of 'e', 'i', 'd'.
        # We track indices i (for a) and j (for b).

        i = 0
        j = 0

        # Current grouping
        current_tag = None
        start_i = 0
        start_j = 0

        for op in path:
            # Determine effective tag
            # Myers produces 'i' and 'd'. difflib uses 'replace' for adjacent 'd' and 'i'.
            # We can merge them later or just emit insert/delete.
            # Git style is insert/delete. 'replace' is visual sugar.
            # Let's map directly first: e->equal, i->insert, d->delete.

            tag = ''
            if op == 'e':
                tag = 'equal'
            elif op == 'i':
                tag = 'insert'
            elif op == 'd':
                tag = 'delete'

            if tag != current_tag:
                if current_tag:
                    yield (current_tag, start_i, i, start_j, j)

                current_tag = tag
                start_i = i
                start_j = j

            # Advance indices
            if op == 'e':
                i += 1
                j += 1
            elif op == 'd':
                i += 1
            elif op == 'i':
                j += 1

        # Yield last group
        if current_tag:
            yield (current_tag, start_i, i, start_j, j)
