import difflib
import io

import pytest

from simple_rcs.pydifflib import StreamSequenceMatcher


def create_bytesio(content: str, encoding: str = 'utf-8') -> io.BytesIO:
    return io.BytesIO(content.encode(encoding))

# --- Fixtures ---
@pytest.fixture
def text_streams():
    a = "Line 1\nLine 2\nLine 3\n"
    b = "Line 1\nLine 2 modified\nLine 3\nLine 4\n"
    return create_bytesio(a), create_bytesio(b)

@pytest.fixture
def binary_streams():
    # 'A' -> 'B' (Replace), 'C' (Equal), 'D' (Insert)
    a = b"AAAACCCC"
    b = b"BBBBCCCCDDDD"
    return io.BytesIO(a), io.BytesIO(b)

# --- Tests for Line Mode (Text) ---

def test_line_mode_opcodes(text_streams):
    # This mode is used by SimpleRCS
    a_stream, b_stream = text_streams
    matcher = StreamSequenceMatcher(a_stream, b_stream, chunk_size=None)

    opcodes = matcher.get_opcodes()

    # Expected:
    # equal 0:1 (Line 1)
    # replace 1:2 (Line 2 -> Line 2 mod)
    # equal 2:3 (Line 3)
    # insert 3:3 (Line 4)

    # Note: indices are LINE numbers
    expected = [
        ('equal', 0, 1, 0, 1),
        ('replace', 1, 2, 1, 2),
        ('equal', 2, 3, 2, 3),
        ('insert', 3, 3, 3, 4),
    ]
    assert opcodes == expected

def test_compare_with_difflib_text():
    # Verify exact match with standard difflib for text
    lines_a = ["Apple\n", "Banana\n", "Cherry\n", "Date\n"]
    lines_b = ["Apple\n", "Berry\n", "Cherry\n", "Date\n", "Elderberry\n"]

    stream_a = create_bytesio("".join(lines_a))
    stream_b = create_bytesio("".join(lines_b))

    pydiff = StreamSequenceMatcher(stream_a, stream_b, chunk_size=None)
    pydiff_opcodes = pydiff.get_opcodes()

    # Standard difflib
    # Note: pydifflib hashes lines stripped of \r\n
    # However, to be 100% fair, we should compare against difflib running on the same abstraction
    # But checking against standard usage is more practical.

    # Let's adjust expected behavior:
    # If pydifflib works correctly, it should yield the same line-based opcodes.

    # Note: SequenceMatcher(a=list_of_hashes) vs SequenceMatcher(a=list_of_strings)
    # They should produce same opcodes if equality holds.

    std_diff = difflib.SequenceMatcher(None, lines_a, lines_b, autojunk=False)
    std_opcodes = std_diff.get_opcodes()

    assert pydiff_opcodes == std_opcodes

# --- Tests for Chunk Mode (Binary) ---

def test_chunk_mode_opcodes(binary_streams):
    a_stream, b_stream = binary_streams
    # Chunk size 4 bytes
    matcher = StreamSequenceMatcher(a_stream, b_stream, chunk_size=4)

    opcodes = matcher.get_opcodes()

    # A: [AAAA][CCCC] (0-4, 4-8)
    # B: [BBBB][CCCC][DDDD] (0-4, 4-8, 8-12)

    # Coarse pass sees:
    # Hash(AAAA) != Hash(BBBB) -> replace/insert/delete?
    # Hash(CCCC) == Hash(CCCC) -> match!

    # Expected:
    # 1. replace: A[0:4] -> B[0:4] (Refined: AAAA -> BBBB)
    # 2. equal: A[4:8] == B[4:8]
    # 3. insert: A[8:8] -> B[8:12] (DDDD)

    # The refinement step should confirm AAAA != BBBB and produce byte-level opcodes.
    # Since they are totally different, it's a full replace.

    expected = [
        ('replace', 0, 4, 0, 4), # AAAA -> BBBB
        ('equal', 4, 8, 4, 8),   # CCCC == CCCC
        ('insert', 8, 8, 8, 12),  # Insert DDDD
    ]

    assert opcodes == expected

def test_chunk_mode_refinement():
    # Test that refinement works for partial matches inside a chunk
    # Chunk size 10
    a = b"1234567890"
    b = b"12345X7890" # '6' -> 'X' at index 5

    stream_a = io.BytesIO(a)
    stream_b = io.BytesIO(b)

    matcher = StreamSequenceMatcher(stream_a, stream_b, chunk_size=10)
    # Only 1 chunk each. Hashes differ.
    # Coarse pass: replace chunk 0 with chunk 0.
    # Refinement: compares "1234567890" vs "12345X7890"

    opcodes = matcher.get_opcodes()

    # Expected refined byte-level opcodes:
    # equal 0:5 ("12345")
    # replace 5:6 ("6" -> "X")
    # equal 6:10 ("7890")

    expected = [
        ('equal', 0, 5, 0, 5),
        ('replace', 5, 6, 5, 6),
        ('equal', 6, 10, 6, 10),
    ]
    assert opcodes == expected

def test_diff_formats_rcs():
    # Ensure it can still generate RCS format (Text mode)
    content_a = "Line 1\nLine 2\nLine 3\n"
    content_b = "Line 1\nLine 2 mod\nLine 3\n"

    stream_a = create_bytesio(content_a)
    stream_b = create_bytesio(content_b)

    # Use Line Mode
    matcher = StreamSequenceMatcher(stream_b, stream_a, chunk_size=None) # New -> Old

    rcs_diff_lines = []
    # Using SimpleRCS logic
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue
        xlen = i2 - i1
        ylen = j2 - j1

        if tag in ('replace', 'delete'): # Delete from B
             if xlen > 0:
                rcs_diff_lines.append(f"d{i1+1} {xlen}")

        if tag in ('replace', 'insert'): # Add from A
             if ylen > 0:
                 xbeg = i1 + 1
                 add_idx = xbeg + xlen - 1
                 rcs_diff_lines.append(f"a{add_idx} {ylen}")

                 # Fetch lines from stream A (b in matcher) using indices
                 # StreamSequenceMatcher.get_lines_from_stream should support line indices in line mode
                 raw_lines = matcher.get_lines_from_stream('b', j1, j2)
                 for line in raw_lines:
                     rcs_diff_lines.append(line.decode('utf-8').rstrip('\n'))

    rcs_output = "\n".join(rcs_diff_lines)

    expected_rcs = "d2 1\na2 1\nLine 2"
    assert rcs_output == expected_rcs
