"""SimpleRCS: reverse-delta version control for single files.

The storage format, diff engines, and CLI tools are documented in the project
wiki: https://github.com/wkpark/SimpleRCS/wiki
"""

from .simple_rcs import SimpleRCS, SimpleRCSCorruptionError

__version__ = "0.2.6"
__all__ = ["SimpleRCS", "SimpleRCSCorruptionError"]
