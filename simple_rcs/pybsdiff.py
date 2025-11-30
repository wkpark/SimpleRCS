import bz2
import io
import logging

from src.app.common.pydifflib import StreamSequenceMatcher


logger = logging.getLogger(__name__)

# BSDIFF40 header structure (32 bytes)
# 0-8:   Magic string "BSDIFF40"
# 8-16:  Length of bzip2ed ctrl block (int64_t, little-endian, sign-magnitude)
# 16-24: Length of bzip2ed diff block (int64_t, little-endian, sign-magnitude)
# 24-32: Length of new file (int64_t, little-endian, sign-magnitude)
BSDIFF40_MAGIC = b"BSDIFF40"
HEADER_SIZE = 32

def _read_off_t(data: bytes) -> int:
    """
    Reads a 64-bit integer in BSDIFF custom sign-magnitude format.

    Standard bsdiff uses a specific format for storing 64-bit integers (off_t):
    - The lower 63 bits store the absolute value (magnitude).
    - The most significant bit (MSB) of the last byte indicates the sign.
      If MSB is 1, the value is negative.
    - This is different from the standard 2's complement used by struct.unpack('<q').
    """
    y = data[7] & 0x7F
    for i in range(6, -1, -1):
        y = y * 256 + data[i]

    if data[7] & 0x80:
        y = -y
    return y

def _write_off_t(value: int) -> bytes:
    """
    Writes a 64-bit integer in BSDIFF custom sign-magnitude format.
    See _read_off_t for format details.
    """
    if value < 0:
        sign = 0x80
        value = -value
    else:
        sign = 0x00

    data = bytearray(8)
    for i in range(8):
        data[i] = value % 256
        value //= 256

    data[7] |= sign
    return bytes(data)

def diff(old: bytes, new: bytes, chunk_size: int = 64) -> bytes:  # noqa: C901
    """
    Generates a binary patch in BSDIFF40 format using StreamSequenceMatcher.

    The BSDIFF40 format consists of a Header and three compressed blocks:
    1.  **Ctrl Block**: A sequence of tuples (diff_len, extra_len, seek_len).
        -   diff_len: Bytes to read from Diff Block and add to Old Data.
        -   extra_len: Bytes to read from Extra Block and append (new data).
        -   seek_len: Bytes to seek in Old Data *after* applying diff and extra.
    2.  **Diff Block**: Data differences (New - Old). Used for approximate matches.
    3.  **Extra Block**: New data inserted that doesn't match Old.

    We map StreamSequenceMatcher opcodes to this structure:
    -   'equal': Maps to Diff Block (with 0 difference).
    -   'replace': Maps to Diff Block (with actual difference).
    -   'insert': Maps to Extra Block.
    -   'delete': Maps to seek_len adjustment.
    """
    old_stream = io.BytesIO(old)
    new_stream = io.BytesIO(new)

    # Use StreamSequenceMatcher for byte-level diffing
    matcher = StreamSequenceMatcher(old_stream, new_stream, chunk_size=chunk_size)

    # Buffers for Ctrl, Diff, Extra blocks
    ctrl_values = []
    diff_buf = io.BytesIO()
    extra_buf = io.BytesIO()

    # Accumulators for the current control tuple
    current_diff_len = 0
    current_extra_len = 0
    current_seek_len = 0

    # Buffers for data associated with the current tuple
    curr_diff_data = bytearray()
    curr_extra_data = bytearray()

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Check flush: Diff -> Extra -> Seek
            # If we have pending Extra or Seek, we must flush current tuple
            if current_extra_len > 0 or current_seek_len != 0:
                ctrl_values.append(_write_off_t(current_diff_len))
                ctrl_values.append(_write_off_t(current_extra_len))
                ctrl_values.append(_write_off_t(current_seek_len))
                diff_buf.write(curr_diff_data)
                extra_buf.write(curr_extra_data)

                current_diff_len = 0
                current_extra_len = 0
                current_seek_len = 0
                curr_diff_data = bytearray()
                curr_extra_data = bytearray()

            # Append to Diff
            seg_len = i2 - i1
            current_diff_len += seg_len
            curr_diff_data.extend(b'\0' * seg_len)

            # Equal block advances old_pos implicitly by diff_len. So seek_len remains 0.

        elif tag == 'replace':
            # Check flush
            if current_extra_len > 0 or current_seek_len != 0:
                ctrl_values.append(_write_off_t(current_diff_len))
                ctrl_values.append(_write_off_t(current_extra_len))
                ctrl_values.append(_write_off_t(current_seek_len))
                diff_buf.write(curr_diff_data)
                extra_buf.write(curr_extra_data)
                current_diff_len = 0
                current_extra_len = 0
                current_seek_len = 0
                curr_diff_data = bytearray()
                curr_extra_data = bytearray()

            len_old = i2 - i1
            len_new = j2 - j1

            diff_len = min(len_old, len_new)

            # Read data
            old_stream.seek(i1)
            old_bytes = old_stream.read(diff_len)
            new_stream.seek(j1)
            new_bytes = new_stream.read(diff_len)

            for k in range(diff_len):
                curr_diff_data.append((new_bytes[k] - old_bytes[k]) & 0xFF)
            current_diff_len += diff_len

            # Handle remainder
            if len_new > len_old:
                # Extra (Insert)
                extra_len = len_new - len_old
                new_stream.seek(j1 + diff_len)
                curr_extra_data.extend(new_stream.read(extra_len))
                current_extra_len += extra_len
            elif len_old > len_new:
                # Seek (Skip)
                seek_len = len_old - len_new
                current_seek_len += seek_len

        elif tag == 'insert':
            # Check flush: if pending Seek, must flush
            if current_seek_len != 0:
                ctrl_values.append(_write_off_t(current_diff_len))
                ctrl_values.append(_write_off_t(current_extra_len))
                ctrl_values.append(_write_off_t(current_seek_len))
                diff_buf.write(curr_diff_data)
                extra_buf.write(curr_extra_data)
                current_diff_len = 0
                current_extra_len = 0
                current_seek_len = 0
                curr_diff_data = bytearray()
                curr_extra_data = bytearray()

            len_new = j2 - j1
            new_stream.seek(j1)
            curr_extra_data.extend(new_stream.read(len_new))
            current_extra_len += len_new

        elif tag == 'delete':
            # Accumulate Seek
            len_old = i2 - i1
            current_seek_len += len_old

    # Final flush
    if current_diff_len > 0 or current_extra_len > 0 or current_seek_len != 0:
        ctrl_values.append(_write_off_t(current_diff_len))
        ctrl_values.append(_write_off_t(current_extra_len))
        ctrl_values.append(_write_off_t(current_seek_len))
        diff_buf.write(curr_diff_data)
        extra_buf.write(curr_extra_data)

    # Compress blocks
    compressed_ctrl = bz2.compress(b"".join(ctrl_values))
    compressed_diff = bz2.compress(diff_buf.getvalue())
    compressed_extra = bz2.compress(extra_buf.getvalue())

    # Build header
    header = BSDIFF40_MAGIC + \
             _write_off_t(len(compressed_ctrl)) + \
             _write_off_t(len(compressed_diff)) + \
             _write_off_t(len(new))

    return header + compressed_ctrl + compressed_diff + compressed_extra

def patch(old: bytes, patch_data: bytes) -> bytes:  # noqa: C901
    """
    Applies a binary patch in BSDIFF40 format.
    """
    if not patch_data.startswith(BSDIFF40_MAGIC):
        raise ValueError("Invalid patch format: missing BSDIFF40 magic header")

    # Parse header
    header = patch_data[0:HEADER_SIZE]
    ctrl_len = _read_off_t(header[8:16])
    diff_len = _read_off_t(header[16:24])
    new_size = _read_off_t(header[24:32])

    # Decompress blocks
    offset = HEADER_SIZE
    compressed_ctrl = patch_data[offset : offset + ctrl_len]
    offset += ctrl_len
    compressed_diff = patch_data[offset : offset + diff_len]
    offset += diff_len
    compressed_extra = patch_data[offset:]

    ctrl_buf = io.BytesIO(bz2.decompress(compressed_ctrl))
    diff_buf = io.BytesIO(bz2.decompress(compressed_diff))
    extra_buf = io.BytesIO(bz2.decompress(compressed_extra))

    # Pre-allocate new_data
    new_data = bytearray(new_size)
    old_pos = 0
    new_pos = 0

    # Apply patch instructions
    while new_pos < new_size:
        # Read control tuple
        ctrl_data = ctrl_buf.read(24)
        if not ctrl_data:
            break

        diff_read_len = _read_off_t(ctrl_data[0:8])
        extra_read_len = _read_off_t(ctrl_data[8:16])
        seek_move_len = _read_off_t(ctrl_data[16:24])

        # Debug output
        logger.debug(f"Patch Ctrl: diff={diff_read_len}, extra={extra_read_len},"
            " seek={seek_move_len} | old_pos={old_pos}, new_pos={new_pos}")

        # 1. Apply diff data
        if diff_read_len > 0:
            diff_segment = diff_buf.read(diff_read_len)
            if len(diff_segment) != diff_read_len:
                raise ValueError("Corrupt patch: diff block truncated")

            # Optimization: slice assignment
            if 0 <= old_pos and old_pos + diff_read_len <= len(old):
                # Fast path
                old_slice = old[old_pos : old_pos + diff_read_len]
                chunk = bytearray(diff_read_len)
                for i in range(diff_read_len):
                    chunk[i] = (old_slice[i] + diff_segment[i]) & 0xFF
                new_data[new_pos : new_pos + diff_read_len] = chunk
            else:
                # Slow path (bounds checking)
                chunk = bytearray(diff_read_len)
                for i in range(diff_read_len):
                    curr_old_pos = old_pos + i
                    if 0 <= curr_old_pos < len(old):
                        old_val = old[curr_old_pos]
                    else:
                        old_val = 0 # Out of bounds = 0
                    chunk[i] = (old_val + diff_segment[i]) & 0xFF
                new_data[new_pos : new_pos + diff_read_len] = chunk

            old_pos += diff_read_len
            new_pos += diff_read_len

        # 2. Apply extra data
        if extra_read_len > 0:
            extra_segment = extra_buf.read(extra_read_len)
            if len(extra_segment) != extra_read_len:
                 raise ValueError("Corrupt patch: extra block truncated")

            new_data[new_pos : new_pos + extra_read_len] = extra_segment
            new_pos += extra_read_len

        # 3. Seek in old file
        old_pos += seek_move_len

    if new_pos != new_size:
        raise ValueError(f"Patch resulted in wrong size: expected {new_size}, got {new_pos}")

    return bytes(new_data)
