"""Emit a git-compatible binary patch for two revisions of a file.

Our storage is BSDIFF inside the block format, which no other tool reads. This
module is the export path: reconstruct both versions, then wrap them the way
``git diff --binary`` does, so ``git apply`` can consume the result.

git accepts two shapes inside a ``GIT binary patch`` block -- ``delta <N>``,
which carries its own pack-delta encoding, and ``literal <N>``, which carries
the whole target file. Both are zlib-deflated and base85-encoded. Only
``literal`` is produced here: it needs no delta encoder, and git itself falls
back to it whenever a delta would not be smaller. The cost is patch size, not
correctness -- a literal block restates the file rather than the change.

The base85 layer is git's own framing, not just the alphabet: at most 52 bytes
per line, each line prefixed by a letter giving its decoded length ('A'-'Z' for
1-26, 'a'-'z' for 27-52), with the final group zero-padded to four bytes before
encoding.
"""

import base64
import hashlib
import zlib

#: git's own limit: one line carries at most 52 decoded bytes.
_MAX_BYTES_PER_LINE = 52


def blob_id(data: bytes, hash_algo: str = "sha1") -> str:
    """git's object id for `data` stored as a blob.

    This is what the patch's ``index`` line names, and what ``git apply``
    checks the file it is about to patch against.
    """
    header = b"blob %d\0" % len(data)
    return hashlib.new(hash_algo, header + data).hexdigest()


def _length_prefix(n: int) -> str:
    if 1 <= n <= 26:
        return chr(ord("A") + n - 1)
    if 27 <= n <= _MAX_BYTES_PER_LINE:
        return chr(ord("a") + n - 27)
    raise ValueError(f"line length {n} outside git's 1..{_MAX_BYTES_PER_LINE}")


def _base85_lines(payload: bytes) -> list[str]:
    """git's line-framed base85: length letter, then the encoded group."""
    lines = []
    for start in range(0, len(payload), _MAX_BYTES_PER_LINE):
        chunk = payload[start : start + _MAX_BYTES_PER_LINE]
        # Encode whole 4-byte groups; the decoder takes only the first
        # len(chunk) bytes back, so the padding never reaches the file.
        padded = chunk + b"\0" * (-len(chunk) % 4)
        lines.append(_length_prefix(len(chunk)) + base64.b85encode(padded).decode("ascii"))
    return lines


def _literal_block(data: bytes) -> list[str]:
    """A ``literal`` block: the whole file, deflated and base85-framed.

    The count on the header line is the *inflated* size, which is how git knows
    what it should end up with.
    """
    return [f"literal {len(data)}", *_base85_lines(zlib.compress(data)), ""]


def binary_patch(path: str, old: bytes, new: bytes, mode: str = "100644") -> str:
    """A ``git apply``-able patch turning `old` into `new` at `path`.

    Emits the reverse block as well, so ``git apply -R`` undoes it. Returns an
    empty string when the two versions are identical, matching git, which
    prints nothing for an unchanged file.
    """
    if old == new:
        return ""
    lines = [
        f"diff --git a/{path} b/{path}",
        f"index {blob_id(old)}..{blob_id(new)} {mode}",
        "GIT binary patch",
        *_literal_block(new),  # forward: what `git apply` installs
        *_literal_block(old),  # reverse: what `git apply -R` restores
    ]
    return "\n".join(lines) + "\n"
