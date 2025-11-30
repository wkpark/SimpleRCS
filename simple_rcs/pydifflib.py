import difflib  # Only for fine-grained refinement logic on small chunks
import os
from collections import namedtuple
from typing import BinaryIO


Match = namedtuple('Match', 'a b size')

class StreamSequenceMatcher:
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
