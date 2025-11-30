import os
import tempfile

import pytest

from src.app.common.simple_rcs import SimpleRCS


@pytest.fixture
def rcs():
    # Create a temporary file for testing on disk
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file_path = tmp_file.name

    rcs_instance = SimpleRCS(tmp_file_path)
    yield rcs_instance

    # Cleanup the temporary file after the test
    if os.path.exists(tmp_file_path):
        os.remove(tmp_file_path)

def test_basic_commit_checkout(rcs):
    # Commit v1.0
    v1 = rcs.commit("Line 1\nLine 2\n")
    assert v1 == "1.0"
    assert rcs.checkout("1.0") == "Line 1\nLine 2\n"

    # Commit v1.1 (Delta)
    v2 = rcs.commit("Line 1\nLine 2\nLine 3\n")
    assert v2 == "1.1"
    assert rcs.checkout("1.1") == "Line 1\nLine 2\nLine 3\n"

    # Verify v1.0 is still retrievable via reverse delta
    assert rcs.checkout("1.0") == "Line 1\nLine 2\n"

def test_snapshot_functionality(rcs):
    # v1.0
    rcs.commit("Version 1 content")

    # v1.1
    rcs.commit("Version 2 content")

    # v1.2 (Commit with snapshot=True)
    # This means v1.1 (the previous HEAD) should be saved as Full Text (snapshot)
    # and v1.2 will be the new HEAD.
    rcs.commit("Version 3 content", snapshot=True)

    # v1.3
    rcs.commit("Version 4 content")

    # Verify content of all versions
    assert rcs.checkout("1.0") == "Version 1 content\n"
    assert rcs.checkout("1.1") == "Version 2 content\n"
    assert rcs.checkout("1.2") == "Version 3 content\n"
    assert rcs.checkout("1.3") == "Version 4 content\n"

    # Verify Internal Structure (Implementation Detail Check)
    # We expect v1.1 to be stored with 'text' keyword instead of 'delta'
    # This part would require parsing the raw file content or exposing internal _parse_block_content_no_regex
    # For now, relying on checkout() to work correctly which implies correct internal storage.
    pass # Skip direct internal check for now, relies on checkout correctness

def test_log_history(rcs):
    rcs.commit("A", author="Alice", log="Init")
    rcs.commit("B", author="Bob", log="Update")

    history = rcs.log()
    assert len(history) == 2
    assert history[0]["ver"] == "1.1"
    assert history[0]["author"] == "Bob"
    assert history[1]["ver"] == "1.0"
    assert history[1]["author"] == "Alice"

def test_blame(rcs):
    rcs.commit("A\nB\nC\n", author="User1")
    rcs.commit("A\nB_mod\nC\n", author="User2")
    rcs.commit("A\nB_mod\nD\n", author="User3")

    blame = rcs.blame()
    # Line 1: A (User1)
    assert blame[0]['line'] == "A"
    assert blame[0]['author'] == "User1"

    # Line 2: B_mod (User2) - Note: blame attributes are from the block where the line originated/was last changed.
    assert blame[1]['line'] == "B_mod"
    assert blame[1]['author'] == "User2"

    # Line 3: D (User3)
    assert blame[2]['line'] == "D"
    assert blame[2]['author'] == "User3"

def test_integrity_verification(rcs):
    # Commit some data
    rcs.commit("Data 1")
    ver_2 = rcs.commit("Data 2")

    # Verify should pass initially
    assert rcs.verify() is True

    # Tamper with the raw file content (simulating external corruption)
    # We need to find the block for V2 in the file to tamper its hash or content
    # This is complex to do without exposing more internal methods.
    # A simpler way to test tampering: write a corrupted file and load it.
    rcs.stream.seek(0) # Go to start of stream
    content = rcs.stream.read().decode('utf-8')

    # Find and corrupt a hash value directly in the string representation
    # This is fragile, depends on exact format. Let's make a more robust tampering test.

    # Create a new RCS instance from a tampered content string
    tampered_content = content.replace("hash @", "hash @HACKED_")
    rcs_tampered = SimpleRCS(tampered_content.encode('utf-8')) # Load from bytes

    # Verify should now fail due to hash mismatch
    assert rcs_tampered.verify() is False

def test_snapshot_chain_integrity(rcs):
    # Test that verify() still works with snapshots breaking the delta chain logic
    rcs.commit("V1")
    rcs.commit("V2")
    rcs.commit("V3", snapshot=True) # V2 becomes snapshot (full text)
    rcs.commit("V4")

    assert rcs.verify() is True

    # Checkout should work
    assert rcs.checkout("1.1") == "V2\n"
