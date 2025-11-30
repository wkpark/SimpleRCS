import pytest

from src.app.common.pybsdiff import diff, patch


@pytest.fixture
def sample_data():
    old = b"The quick brown fox jumps over the lazy dog."
    new = b"The quick brown cat jumps over the lazy dog." # fox -> cat
    return old, new

def test_basic_diff_patch(sample_data):
    old, new = sample_data

    # Generate patch
    patch_data = diff(old, new)
    assert len(patch_data) > 0

    # Apply patch
    restored = patch(old, patch_data)
    assert restored == new

def test_insert_append():
    old = b"Hello"
    new = b"Hello World"

    patch_data = diff(old, new)
    restored = patch(old, patch_data)
    assert restored == new

def test_delete_prefix():
    old = b"PrefixData"
    new = b"Data"

    patch_data = diff(old, new)
    restored = patch(old, patch_data)
    assert restored == new

def test_replace_middle():
    old = b"AAAAABBBBBCCCCC"
    new = b"AAAAAXXXXXCCCCC"

    patch_data = diff(old, new)
    restored = patch(old, patch_data)
    assert restored == new

def test_large_ish_data():
    # 100KB data
    old = b"A" * 100000
    new = b"A" * 50000 + b"B" + b"A" * 49999

    patch_data = diff(old, new)
    restored = patch(old, patch_data)
    assert restored == new
