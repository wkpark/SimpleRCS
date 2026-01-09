import os
import shutil
import subprocess
import time

import pytest

from src.app.common.pybsdiff import diff, patch


# Check if bsdiff/bspatch are available
BSDIFF_CMD = shutil.which("bsdiff")
BSPATCH_CMD = shutil.which("bspatch")

@pytest.mark.skipif(not BSDIFF_CMD or not BSPATCH_CMD, reason="bsdiff/bspatch not installed")
@pytest.mark.benchmark
def test_bsdiff_compatibility(tmp_path):
    print("\n--- BSDIFF Compatibility Test ---")
    print(f"Using bsdiff: {BSDIFF_CMD}")
    print(f"Using bspatch: {BSPATCH_CMD}")

    # 1. Generate Data (10MB)
    size_mb = 10
    size_bytes = size_mb * 1024 * 1024

    old_file = tmp_path / "old.bin"
    new_file = tmp_path / "new.bin"
    patch_native = tmp_path / "patch.bs"
    patch_py = tmp_path / "patch.pybs"
    restored_native = tmp_path / "restored_native.bin"

    # Generate content similar to previous benchmark
    chunk = os.urandom(1024 * 64)
    repeats = size_bytes // len(chunk)
    old_data = bytearray(chunk * repeats)
    new_data = bytearray(old_data)

    # Apply changes (Modify, Insert, Delete)
    mid = size_bytes // 2
    mod_len = 1024 * 10
    new_data[mid : mid+mod_len] = os.urandom(mod_len)

    del_pos = (size_bytes * 3) // 4
    del new_data[del_pos : del_pos + 1024*5]

    old_file.write_bytes(old_data)
    new_file.write_bytes(new_data)

    print(f"Old size: {len(old_data)}")
    print(f"New size: {len(new_data)}")

    # 2. Run Native bsdiff (Baseline)
    print("Running native bsdiff...")
    start_time = time.perf_counter()
    subprocess.run([BSDIFF_CMD, str(old_file), str(new_file), str(patch_native)], check=True)
    native_time = time.perf_counter() - start_time
    native_size = patch_native.stat().st_size
    print(f"Native bsdiff time: {native_time:.4f}s")
    print(f"Native patch size: {native_size} bytes")

    # 3. Run pybsdiff (Python)
    print("Running pybsdiff.diff()...")
    start_time = time.perf_counter()
    py_patch_data = diff(bytes(old_data), bytes(new_data), chunk_size=64) # Optimized chunk size
    py_time = time.perf_counter() - start_time
    py_size = len(py_patch_data)

    patch_py.write_bytes(py_patch_data)
    print(f"pybsdiff time: {py_time:.4f}s")
    print(f"pybsdiff patch size: {py_size} bytes")

    print(f"Size Ratio (py/native): {py_size/native_size:.2f}x")
    print(f"Time Ratio (py/native): {py_time/native_time:.2f}x")

    # 4. Cross-Verification: Native bspatch applying pybsdiff patch
    # If this works, our patch format is valid BSDIFF40.
    print("Verifying: Native bspatch applying pybsdiff patch...")
    try:
        subprocess.run([BSPATCH_CMD, str(old_file), str(restored_native), str(patch_py)], check=True)
        restored_data = restored_native.read_bytes()
        assert restored_data == new_data
        print("SUCCESS: Native bspatch successfully applied pybsdiff patch!")
    except subprocess.CalledProcessError:
        pytest.fail("Native bspatch failed to apply pybsdiff patch (Format mismatch?)")
    except AssertionError:
        pytest.fail("Native bspatch applied patch but result differs from expected")

    # 5. Cross-Verification: pybsdiff.patch applying Native bsdiff patch
    # If this works, our patch parsing logic handles standard BSDIFF40 correctly.
    print("Verifying: pybsdiff.patch applying Native bsdiff patch...")
    native_patch_data = patch_native.read_bytes()
    try:
        py_restored_data = patch(bytes(old_data), native_patch_data)
        assert py_restored_data == new_data
        print("SUCCESS: pybsdiff.patch successfully applied native bsdiff patch!")
    except Exception as e:
        pytest.fail(f"pybsdiff.patch failed to apply native patch: {e}")

@pytest.mark.skipif(not BSDIFF_CMD or not BSPATCH_CMD, reason="bsdiff/bspatch not installed")
def test_bsdiff_compatibility_small_100kb(tmp_path):
    print("\n--- BSDIFF Compatibility Test (100KB) ---")

    # 1. Generate Data (100KB)
    size_kb = 100
    size_bytes = size_kb * 1024

    old_file = tmp_path / "old_100kb.bin"
    new_file = tmp_path / "new_100kb.bin"
    patch_native = tmp_path / "patch_100kb.bs"
    patch_py = tmp_path / "patch_100kb.pybs"
    restored_native = tmp_path / "restored_native_100kb.bin"

    chunk = os.urandom(1024 * 4) # 4KB chunk
    repeats = size_bytes // len(chunk)
    old_data = bytearray(chunk * repeats)
    new_data = bytearray(old_data)

    # Apply changes
    # Modify middle
    mid = size_bytes // 2
    mod_len = 1024 * 2 # 2KB
    new_data[mid : mid+mod_len] = os.urandom(mod_len)

    # Delete some part
    del_pos = (size_bytes * 3) // 4
    del new_data[del_pos : del_pos + 1024] # 1KB delete

    old_file.write_bytes(old_data)
    new_file.write_bytes(new_data)

    # 2. Native Diff
    subprocess.run([BSDIFF_CMD, str(old_file), str(new_file), str(patch_native)], check=True)

    # 3. Py Diff
    py_patch_data = diff(bytes(old_data), bytes(new_data), chunk_size=64)
    patch_py.write_bytes(py_patch_data)

    # 4. Native Patch -> Py Apply
    print("Verifying: Native bspatch applying pybsdiff patch...")
    subprocess.run([BSPATCH_CMD, str(old_file), str(restored_native), str(patch_py)], check=True)
    assert restored_native.read_bytes() == new_data

    # 5. Py Patch -> Native Apply
    print("Verifying: pybsdiff.patch applying Native bsdiff patch...")
    native_patch_data = patch_native.read_bytes()
    py_restored_data = patch(bytes(old_data), native_patch_data)
    assert py_restored_data == new_data

