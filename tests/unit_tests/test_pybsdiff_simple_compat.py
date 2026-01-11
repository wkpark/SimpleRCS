import bz2
import io
import shutil
import struct
import subprocess

import pytest

from simple_rcs.pybsdiff import diff, patch, _read_off_t


BSDIFF_CMD = shutil.which("bsdiff")
BSPATCH_CMD = shutil.which("bspatch")

@pytest.mark.skipif(not BSDIFF_CMD or not BSPATCH_CMD, reason="bsdiff/bspatch not installed")
class TestBsdiffCompatSmall:

    def _run_cross_check(self, tmp_path, case_name, old_data, new_data):
        print(f"\n--- Case: {case_name} ---")
        old_file = tmp_path / "old"
        new_file = tmp_path / "new"
        patch_native = tmp_path / "patch.bs"
        patch_py = tmp_path / "patch.py"
        restored_native = tmp_path / "restored.native"

        old_file.write_bytes(old_data)
        new_file.write_bytes(new_data)

        # 1. Native Diff
        subprocess.run([BSDIFF_CMD, str(old_file), str(new_file), str(patch_native)], check=True)

        # Debug: Inspect Native Patch Ctrl Blocks
        print("Native Patch Analysis:")
        self._inspect_patch(patch_native.read_bytes())

        # 2. Py Diff
        py_patch_data = diff(old_data, new_data)
        patch_py.write_bytes(py_patch_data)

        print("Py Patch Analysis:")
        self._inspect_patch(py_patch_data)

        # 3. Native Patch -> Py Apply
        print("Applying Native Patch with Python...")
        try:
            py_restored = patch(old_data, patch_native.read_bytes())
            assert py_restored == new_data, "Py failed to apply Native patch"
            print("SUCCESS: Py applied Native patch")
        except Exception as e:
            print(f"FAILED: Py applied Native patch: {e}")
            raise

        # 4. Py Patch -> Native Apply
        print("Applying Py Patch with Native...")
        try:
            subprocess.run([BSPATCH_CMD, str(old_file), str(restored_native), str(patch_py)], check=True)
            native_restored = restored_native.read_bytes()
            assert native_restored == new_data, "Native failed to apply Py patch"
            print("SUCCESS: Native applied Py patch")
        except Exception as e:
            print(f"FAILED: Native applied Py patch: {e}")
            raise

    def _inspect_patch(self, patch_data):
        """Helper to print ctrl blocks of a patch."""
        header = patch_data[:32]
        magic, ctrl_len, diff_len, new_size = struct.unpack('<8sQQQ', header)
        print(f"  Header: Magic={magic}, Ctrl={ctrl_len}, Diff={diff_len}, NewSize={new_size}")

        offset = 32
        ctrl_block = patch_data[offset : offset+ctrl_len]
        ctrl_buf = io.BytesIO(bz2.decompress(ctrl_block))

        idx = 0
        while True:
            chunk = ctrl_buf.read(24)
            if not chunk: break

            d = _read_off_t(chunk[0:8])
            e = _read_off_t(chunk[8:16])
            s = _read_off_t(chunk[16:24])
            print(f"  Ctrl[{idx}]: Diff={d}, Extra={e}, Seek={s}")
            idx += 1

    def test_simple_append(self, tmp_path):
        self._run_cross_check(tmp_path, "Append", b"Hello", b"Hello World")

    def test_simple_delete(self, tmp_path):
        self._run_cross_check(tmp_path, "Delete", b"Hello World", b"Hello")

    def test_simple_replace(self, tmp_path):
        self._run_cross_check(tmp_path, "Replace", b"Hello World", b"Hello Python")

    def test_middle_change(self, tmp_path):
        self._run_cross_check(tmp_path, "Middle", b"AAAABBBBCCCC", b"AAAAXXXXCCCC")

    def test_mixed(self, tmp_path):
        # A bit more complex: Match, Delete, Match, Insert, Match
        old = b"111122223333"
        new = b"111133334444" # Deleted 2222, Inserted 4444
        self._run_cross_check(tmp_path, "Mixed", old, new)

