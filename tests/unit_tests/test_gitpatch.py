"""git-compatible binary patch output.

The acceptance test is `git apply`: the framing is git's, so anything short of
git accepting the result is a bug. The unit tests below pin the pieces that
would silently drift (line length, prefix letters, blob ids) so a failure says
which part broke rather than just "git refused it".
"""

import base64
import os
import shutil
import subprocess
import tracemalloc
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


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("plain.bin", "a/plain.bin"),                       # nothing to escape
        ("with space.bin", "a/with space.bin"),             # a space alone does not trigger it
        ("we\nird.bin", r'"a/we\nird.bin"'),
        ("ta\tb.bin", r'"a/ta\tb.bin"'),
        ("be\x07ll.bin", r'"a/be\all.bin"'),
        ('qu"ote.bin', r'"a/qu\"ote.bin"'),
        ("back\\slash.bin", r'"a/back\\slash.bin"'),
        ("한글.bin", r'"a/\355\225\234\352\270\200.bin"'),  # high bytes -> octal, as core.quotePath does
    ],
)
def test_path_quoting_matches_gits_rule(path, expected):
    """These strings are `git diff --binary`'s own output, taken from a real
    repository with those filenames -- not derived from reading quote.c."""
    assert gitpatch._quote_path("a/" + path) == expected


def test_a_newline_in_the_path_cannot_forge_a_header():
    """The path is data written into a line-structured format. Unquoted, a
    newline in it opens a second `diff --git` line the reader would believe."""
    patch = gitpatch.binary_patch("ok.bin\ndiff --git a/etc/passwd b/etc/passwd", OLD, NEW)

    # The forged text is still *in* the patch -- it is the filename the caller
    # asked for. What matters is that it stays inside one quoted header instead
    # of opening a second line a reader would believe.
    headers = [line for line in patch.splitlines() if line.startswith("diff --git ")]
    assert len(headers) == 1
    assert headers[0].startswith('diff --git "a/ok.bin\\ndiff --git a/etc/passwd')
    assert "\n" not in gitpatch._quote_path("a/ok.bin\ndiff --git a/etc/passwd b/etc/passwd")


@needs_git
def test_git_applies_a_patch_for_a_path_it_has_to_quote(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.run([GIT, *args], cwd=repo, capture_output=True, text=True, check=True)

    git("init", "-q", ".")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")

    name = "we\nird 한글.bin"
    target = repo / name
    target.write_bytes(OLD)
    git("add", "-A")
    git("commit", "-q", "-m", "base")

    (repo / "out.patch").write_text(gitpatch.binary_patch(name, OLD, NEW))
    git("apply", "out.patch")
    assert target.read_bytes() == NEW

    git("apply", "-R", "out.patch")
    assert target.read_bytes() == OLD


def test_the_generator_and_the_string_form_agree():
    lines = list(gitpatch.iter_binary_patch("f.bin", OLD, NEW))

    assert "\n".join(lines) + "\n" == gitpatch.binary_patch("f.bin", OLD, NEW)


def test_the_generator_yields_nothing_for_identical_revisions():
    assert list(gitpatch.iter_binary_patch("f.bin", OLD, OLD)) == []


def test_streaming_the_lines_costs_a_fraction_of_building_the_string():
    """The reason iter_binary_patch exists. A literal block restates the whole
    file, so the string form holds the deflated payload, the base85 lines and
    the joined result at once; consuming lines keeps only one line alive.

    Measured on this payload: streaming 0.22, materialising one block at a time
    0.30, materialising the whole patch 0.51. The threshold catches the last of
    those -- a generator that builds everything before yielding, which is what
    binary_patch already does and what this function exists not to do. It does
    not catch per-block materialisation; separating that needs a threshold too
    close to the measured value to hold across platforms.
    """
    # Incompressible on purpose: a payload that deflates to almost nothing
    # leaves nothing to save, and this test would pass while measuring nothing.
    # The ratio is structural, not data-dependent, so the bytes need not be
    # reproducible -- only incompressible.
    payload = os.urandom(256 << 10)  # big enough for a wide margin, small enough to stay quick
    changed = payload[: len(payload) // 2] + b"edit" + payload[len(payload) // 2 + 4 :]

    tracemalloc.start()
    gitpatch.binary_patch("f.bin", payload, changed)
    _, joined_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    for _line in gitpatch.iter_binary_patch("f.bin", payload, changed):
        pass
    _, streamed_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert streamed_peak < joined_peak * 0.4, f"streamed={streamed_peak} joined={joined_peak}"
