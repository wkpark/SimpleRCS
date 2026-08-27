"""Optional Cython extension build.

The Myers matchers in `simple_rcs/_myersdiff_{ses,dmp}.pyx` are the preferred
diff backend, but they must never be a hard install requirement: the library
runs correctly without them by falling back to `StreamSequenceMatcher` (see
`simple_rcs/matchers.py`). A missing C toolchain therefore degrades
performance, it does not fail the install.

Set SIMPLE_RCS_NO_EXT=1 to skip the build entirely -- used when producing a
pure-Python wheel.
"""
import os

from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.errors import CCompilerError, ExecError, PlatformError

_BUILD_ERRORS = (CCompilerError, ExecError, PlatformError, ImportError)


class optional_build_ext(build_ext):  # noqa: N801
    """build_ext that downgrades a failed compile to a warning."""

    def run(self) -> None:
        try:
            super().run()
        except _BUILD_ERRORS as e:
            self.warn(f"Cython matchers not built ({e}); falling back to the pure-Python engine")

    def build_extension(self, ext) -> None:
        try:
            super().build_extension(ext)
        except _BUILD_ERRORS as e:
            self.warn(f"{ext.name} not built ({e}); falling back to the pure-Python engine")


ext_modules = []
if not os.environ.get("SIMPLE_RCS_NO_EXT"):
    try:
        from Cython.Build import cythonize

        ext_modules = cythonize(
            ["simple_rcs/_myersdiff_ses.pyx", "simple_rcs/_myersdiff_dmp.pyx"],
            language_level="3",
        )
    except ImportError:
        pass  # Cython absent: ship pure Python.

setup(ext_modules=ext_modules, cmdclass={"build_ext": optional_build_ext})
