import shutil
import subprocess
import time

import pytest

from simple_rcs.pybsdiff import diff, patch


BSDIFF_CMD = shutil.which("bsdiff")
BSPATCH_CMD = shutil.which("bspatch")

def generate_realistic_log_data(num_lines=20000):
    """
    Generates somewhat realistic log data.
    Repetitive structure but unique timestamps/IDs.
    """
    lines = []
    for i in range(num_lines):
        # A typical log line structure
        lines.append(f"2023-01-01 10:00:{i%60:02d} [INFO] User_{i} performed action_X on resource_Y result=SUCCESS\n")

    return "".join(lines).encode('utf-8')

def modify_log_data(original_bytes):
    """
    Modifies the log data to simulate edits.
    """
    # Convert to list of lines for easy editing
    lines = original_bytes.decode('utf-8').splitlines(keepends=True)

    # 1. Modify some lines (change status)
    for i in range(0, len(lines), 100):
        lines[i] = lines[i].replace("SUCCESS", "FAILURE_CODE_123")

    # 2. Insert a block of logs
    insert_block = [f"2023-01-01 12:00:00 [WARN] System warning event {k}\n" for k in range(500)]
    lines[1000:1000] = insert_block

    # 3. Delete a block of logs
    del lines[5000:5500]

    return "".join(lines).encode('utf-8')

def _read_off_t(data):
    """Helper to read BSDIFF off_t (Sign-Magnitude)"""
    y = data[7] & 0x7F
    for i in range(6, -1, -1):
        y = y * 256 + data[i]
    if data[7] & 0x80:
        y = -y
    return y

def _analyze_patch_structure(patch_data, name="Patch"):
    print(f"\n--- Analysis: {name} ---")
    header = patch_data[:32]
    # magic = header[:8]
    ctrl_len = _read_off_t(header[8:16])
    # diff_len = _read_off_t(header[16:24])
    # new_size = _read_off_t(header[24:32])

    print(f"Ctrl Block Length (compressed): {ctrl_len}")

    # Decompress Ctrl
    import bz2
    import io
    ctrl_block_compressed = patch_data[32 : 32 + ctrl_len]
    ctrl_buf = io.BytesIO(bz2.decompress(ctrl_block_compressed))

    count = 0
    print("Control Tuples (First 20):")
    print("  IDX | Diff Len | Extra Len | Seek Len")
    while True:
        chunk = ctrl_buf.read(24)
        if not chunk: break

        d = _read_off_t(chunk[0:8])
        e = _read_off_t(chunk[8:16])
        s = _read_off_t(chunk[16:24])

        if count < 20:
            print(f"  {count:3d} | {d:8d} | {e:9d} | {s:8d}")

        count += 1

    print(f"Total Control Tuples: {count}")

@pytest.mark.skipif(not BSDIFF_CMD, reason="bsdiff not installed")
@pytest.mark.benchmark
def test_realistic_data_patch_size(tmp_path):
    print("\n--- Realistic Data Benchmark ---")

    # 1. Prepare Data
    old_data = generate_realistic_log_data(num_lines=50000) # Approx 4-5 MB
    new_data = modify_log_data(old_data)

    old_file = tmp_path / "old.log"
    new_file = tmp_path / "new.log"
    old_file.write_bytes(old_data)
    new_file.write_bytes(new_data)

    print(f"Old Size: {len(old_data)} bytes ({len(old_data)/1024/1024:.2f} MB)")
    print(f"New Size: {len(new_data)} bytes")

    # Native bsdiff (Baseline)
    patch_native = tmp_path / "patch.bs"

    start = time.perf_counter()
    subprocess.run([BSDIFF_CMD, str(old_file), str(new_file), str(patch_native)], check=True)
    native_time = time.perf_counter() - start
    native_size = patch_native.stat().st_size

    print("\n[Native bsdiff]")
    print(f"Time: {native_time:.4f}s")
    print(f"Size: {native_size} bytes")

    _analyze_patch_structure(patch_native.read_bytes(), "Native Patch")

    # pybsdiff (Chunk=64) - High overhead expected
    start = time.perf_counter()
    patch_64 = diff(old_data, new_data, chunk_size=64)
    time_64 = time.perf_counter() - start
    size_64 = len(patch_64)

    print("\n[pybsdiff chunk=64]")
    print(f"Time: {time_64:.4f}s")
    print(f"Size: {size_64} bytes")
    print(f"Size Ratio (vs Native): {size_64/native_size:.2f}x")

    # Verify correctness
    assert patch(old_data, patch_64) == new_data
    _analyze_patch_structure(patch_64, "pybsdiff (Chunk=64)")

    # pybsdiff (Chunk=2048) - High overhead expected
    start = time.perf_counter()
    patch_2k = diff(old_data, new_data, chunk_size=2048)
    time_2k = time.perf_counter() - start
    size_2k = len(patch_2k)

    print("\n[pybsdiff chunk=2048]")
    print(f"Time: {time_2k:.4f}s")
    print(f"Size: {size_2k} bytes")
    print(f"Size Ratio (vs Native): {size_2k/native_size:.2f}x")

    # Verify correctness
    assert patch(old_data, patch_2k) == new_data
    _analyze_patch_structure(patch_2k, "pybsdiff (Chunk=2048)")

    # pybsdiff (Chunk=4096) - Optimized for large files
    # Larger chunks mean fewer control tuples => smaller patch size for large matches
    start = time.perf_counter()
    patch_4k = diff(old_data, new_data, chunk_size=4096)
    time_4k = time.perf_counter() - start
    size_4k = len(patch_4k)

    print("\n[pybsdiff chunk=4096]")
    print(f"Time: {time_4k:.4f}s")
    print(f"Size: {size_4k} bytes")
    print(f"Size Ratio (vs Native): {size_4k/native_size:.2f}x")
    print(f"Size Reduction (vs chunk=64): {(1 - size_4k/size_64)*100:.1f}%")

    _analyze_patch_structure(patch_4k, "pybsdiff (Chunk=4096)")

    # pybsdiff (Rolling Hash) - Optimized for shift/insert/delete
    print(f"\n[pybsdiff Rolling Hash chunk=4096]")
    start = time.perf_counter()
    patch_rolling = diff(old_data, new_data, chunk_size=4096, matcher_type='rolling')
    time_rolling = time.perf_counter() - start
    size_rolling = len(patch_rolling)

    print(f"Time: {time_rolling:.4f}s")
    print(f"Size: {size_rolling} bytes")
    print(f"Size Ratio (vs Native): {size_rolling/native_size:.2f}x")
    print(f"Time Ratio (vs Native): {time_rolling/native_time:.2f}x")

    _analyze_patch_structure(patch_rolling, "pybsdiff (Rolling Hash)")

    # Verify correctness
    assert patch(old_data, patch_rolling) == new_data
