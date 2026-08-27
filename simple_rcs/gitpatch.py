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
from collections.abc import Iterator

#: git's own limit: one line carries at most 52 decoded bytes.
_MAX_BYTES_PER_LINE = 52

#: The byte -> escape letter table git uses for path quoting (quote.c's cq_lookup).
_C_ESCAPES = {
    0x07: "a", 0x08: "b", 0x09: "t", 0x0A: "n", 0x0B: "v", 0x0C: "f", 0x0D: "r",
    0x22: '"', 0x5C: "\\",
}


def _quote_path(path: str) -> str:
    """git's C-style path quoting, for the header's ``a/``/``b/`` names.

    A path is only data, but it is written into a line-structured format, so an
    unquoted newline in it forges patch content. git solves this by quoting: a
    name containing a control byte, a quote, a backslash or a high byte is
    wrapped in double quotes with C escapes, and anything else is left alone
    (a space does not trigger it). High bytes become three-digit octal, which is
    what ``core.quotePath`` does by default.
    """
    raw = path.encode("utf-8", "surrogateescape")
    if not any(b < 0x20 or b >= 0x7F or b in (0x22, 0x5C) for b in raw):
        return path
    out = ['"']
    for byte in raw:
        escape = _C_ESCAPES.get(byte)
        if escape is not None:
            out.append("\\" + escape)
        elif 0x20 <= byte < 0x7F:
            out.append(chr(byte))
        else:
            out.append(f"\\{byte:03o}")
    out.append('"')
    return "".join(out)


def blob_id(data: bytes) -> str:
    """git's object id for `data` stored as a blob.

    This is what the patch's ``index`` line names, and what ``git apply``
    checks the file it is about to patch against.

    sha1 is the format's, not a choice: git refuses a binary patch whose index
    line is not a full sha1 object id ("cannot apply binary patch ... without
    full index line"). A sha256 repository would need the object format read
    from the target repository, not a default on this function.
    """
    # usedforsecurity=False: this is git's object id, not a security hash --
    # it also lets the call work on a FIPS build, where sha1 is otherwise refused.
    # Fed in two update() calls rather than hashing `header + data`, which
    # allocates a second copy of the whole file just to prepend 12 bytes.
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(b"blob %d\0" % len(data))
    digest.update(data)
    return digest.hexdigest()


def _length_prefix(n: int) -> str:
    if 1 <= n <= 26:
        return chr(ord("A") + n - 1)
    if 27 <= n <= _MAX_BYTES_PER_LINE:
        return chr(ord("a") + n - 27)
    raise ValueError(f"line length {n} outside git's 1..{_MAX_BYTES_PER_LINE}")


def _base85_lines(payload: bytes) -> Iterator[str]:
    """git's line-framed base85: length letter, then the encoded group."""
    for start in range(0, len(payload), _MAX_BYTES_PER_LINE):
        chunk = payload[start : start + _MAX_BYTES_PER_LINE]
        # Encode whole 4-byte groups; the decoder takes only the first
        # len(chunk) bytes back, so the padding never reaches the file.
        padded = chunk + b"\0" * (-len(chunk) % 4)
        yield _length_prefix(len(chunk)) + base64.b85encode(padded).decode("ascii")


def _literal_block(data: bytes) -> Iterator[str]:
    """A ``literal`` block: the whole file, deflated and base85-framed.

    The count on the header line is the *inflated* size, which is how git knows
    what it should end up with.
    """
    yield f"literal {len(data)}"
    yield from _base85_lines(zlib.compress(data))
    yield ""


def iter_binary_patch(path: str, old: bytes, new: bytes, mode: str = "100644") -> Iterator[str]:
    """The patch as lines, without their newlines. Yields nothing when the two
    versions are identical, matching git, which prints nothing for an unchanged
    file.

    A literal block restates the whole file, so the patch is larger than the
    revisions it carries. Consuming it line by line keeps the base85 text out of
    memory; :func:`binary_patch` is the convenience wrapper for callers small
    enough not to care.

    `path` is quoted the way git quotes it, so a name carrying a newline cannot
    forge patch content -- see :func:`_quote_path`.

    The reverse block is emitted as well, so ``git apply -R`` undoes the patch.
    """
    if old == new:
        return
    yield f"diff --git {_quote_path('a/' + path)} {_quote_path('b/' + path)}"
    yield f"index {blob_id(old)}..{blob_id(new)} {mode}"
    yield "GIT binary patch"
    yield from _literal_block(new)  # forward: what `git apply` installs
    yield from _literal_block(old)  # reverse: what `git apply -R` restores


def binary_patch(path: str, old: bytes, new: bytes, mode: str = "100644") -> str:
    """:func:`iter_binary_patch` as one string, or "" for identical revisions."""
    lines = list(iter_binary_patch(path, old, new, mode))
    return "\n".join(lines) + "\n" if lines else ""
