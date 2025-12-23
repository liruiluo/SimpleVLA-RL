from __future__ import annotations

import contextlib
import sys
from typing import Iterable, TextIO


class _LineFilterStream:
    def __init__(self, stream: TextIO, drop_substrings: Iterable[str]):
        self._stream = stream
        self._drop_substrings = tuple(drop_substrings)
        self._buf = ""

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if any(sub in line for sub in self._drop_substrings):
                continue
            self._stream.write(line + "\n")
        return len(s)

    def flush(self) -> None:
        if self._buf:
            if not any(sub in self._buf for sub in self._drop_substrings):
                self._stream.write(self._buf)
            self._buf = ""
        self._stream.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._stream, "isatty", lambda: False)())


@contextlib.contextmanager
def filter_std_streams(drop_substrings: Iterable[str]):
    """
    Filter noisy `print()` output by dropping lines that contain any of the given substrings.

    Useful when upstream libraries are too verbose but shouldn't be modified.
    """
    out = _LineFilterStream(sys.stdout, drop_substrings)
    err = _LineFilterStream(sys.stderr, drop_substrings)
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        yield

