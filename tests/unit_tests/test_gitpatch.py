"""git-compatible binary patch output.

The acceptance test is `git apply`: the framing is git's, so anything short of
git accepting the result is a bug. The unit tests below pin the pieces that
would silently drift (line length, prefix letters, blob ids) so a failure says
which part broke rather than just "git refused it".
"""

import base64
import shutil
import subprocess
import zlib

import pytest

from simple_rcs import gitpatch

OLD = bytes(range(256)) * 3
NEW = OLD[:400] + b"a different middle section entirely" + OLD[440:] + b"and a tail"

GIT = shutil.which("git")
needs_git = pytest.mark.skipif(GIT is None, reason="git is not installed")


def _decode_base85_lines(lines):
    """The reader git implements: length letter, then whole 4-byte groups."""
    out = b""
    for line in lines:
        prefix = line[0]
        n = ord(prefix) - ord("A") + 1 if "A" <= prefix <= "Z" else ord(prefix) - ord("a") + 27
        out += base64.b85decode(line[1:].encode("ascii"))[:n]
    return out


def _blocks(patch: str):
    """Split a patch into its (kind, size, payload) binary blocks."""
    lines = patch.splitlines()
    body = lines[lines.index("GIT binary patch") + 1 :]
    blocks, current = [], None
    for line in body:
        if not line:
            current = None
        elif current is None:
            kind, size = line.split()
            current = (kind, int(size), [])
            blocks.append(current)
        else:
            current[2].append(line)
    return blocks


def test_a_patch_carries_the_forward_and_reverse_contents():
    patch = gitpatch.binary_patch("f.bin", OLD, NEW)
    blocks = _blocks(patch)

    assert [kind for kind, _, _ in blocks] == ["literal", "literal"]
    forward, reverse = blocks
    assert zlib.decompress(_decode_base85_lines(forward[2])) == NEW
    assert zlib.decompress(_decode_base85_lines(reverse[2])) == OLD
    # The header count is the inflated size, which is how git knows what it
    # should end up with.
    assert forward[1] == len(NEW)
    assert reverse[1] == len(OLD)


def test_base85_lines_stay_inside_gits_framing():
    patch = gitpatch.binary_patch("f.bin", OLD, NEW)
    payload_lines = [line for _, _, lines in _blocks(patch) for line in lines]
    assert payload_lines, "expected a non-trivial payload"

    for line in payload_lines:
        prefix = line[0]
        assert prefix.isalpha()
        n = ord(prefix) - ord("A") + 1 if "A" <= prefix <= "Z" else ord(prefix) - ord("a") + 27
        assert 1 <= n <= 52
        # Whole 4-byte groups, 5 characters each.
        assert len(line) - 1 == ((n + 3) // 4) * 5


def test_identical_revisions_produce_no_patch():
    assert gitpatch.binary_patch("f.bin", OLD, OLD) == ""


def test_a_length_outside_gits_range_is_rejected():
    with pytest.raises(ValueError, match="outside git's"):
        gitpatch._length_prefix(53)


@needs_git
def test_blob_id_matches_git_hash_object(tmp_path):
    target = tmp_path / "blob.bin"
    target.write_bytes(NEW)
    expected = subprocess.run(
        [GIT, "hash-object", str(target)], capture_output=True, text=True, check=True
    ).stdout.strip()

    assert gitpatch.blob_id(NEW) == expected


@needs_git
def test_git_apply_accepts_the_patch_and_reverses_it(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args, **kwargs):
        return subprocess.run([GIT, *args], cwd=repo, capture_output=True, text=True, check=True, **kwargs)

    git("init", "-q", ".")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    target = repo / "data.bin"
    target.write_bytes(OLD)
    git("add", "data.bin")
    git("commit", "-q", "-m", "base")

    patch = repo / "out.patch"
    patch.write_text(gitpatch.binary_patch("data.bin", OLD, NEW))

    git("apply", "--check", str(patch))
    git("apply", str(patch))
    assert target.read_bytes() == NEW

    git("apply", "-R", str(patch))
    assert target.read_bytes() == OLD
