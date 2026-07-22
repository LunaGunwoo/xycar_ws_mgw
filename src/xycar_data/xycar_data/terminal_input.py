# Copyright 2026 Gunwoo Moon
# Licensed under the Apache License, Version 2.0
"""Small terminal key reader for a real TTY without a GUI dependency."""

from __future__ import annotations

import fcntl
import os
import select
import sys
import termios
import time
from typing import Optional, TextIO


class KeySequenceParser:
    """Parse terminal bytes, including fragmented arrow-key escape sequences."""

    _ARROWS = {
        b"\x1b[A": "up",
        b"\x1b[B": "down",
        b"\x1b[C": "right",
        b"\x1b[D": "left",
        b"\x1bOA": "up",
        b"\x1bOB": "down",
        b"\x1bOC": "right",
        b"\x1bOD": "left",
    }

    def __init__(self) -> None:
        self._buffer = bytearray()

    @property
    def waiting_for_escape_completion(self) -> bool:
        return bool(self._buffer) and self._buffer[0] == 0x1B

    def feed(self, data: bytes) -> list[str]:
        self._buffer.extend(data)
        keys: list[str] = []
        while self._buffer:
            if self._buffer[0] == 0x1B:
                current = bytes(self._buffer)
                complete = next(
                    (
                        key
                        for sequence, key in self._ARROWS.items()
                        if current.startswith(sequence)
                    ),
                    None,
                )
                if complete is not None:
                    del self._buffer[:3]
                    keys.append(complete)
                    continue
                if any(sequence.startswith(current) for sequence in self._ARROWS):
                    break
                del self._buffer[0]
                keys.append("escape")
                continue

            byte = self._buffer.pop(0)
            if byte == 0x03:
                keys.append("ctrl_c")
            elif byte == 0x20:
                keys.append("space")
            elif byte in (ord("r"), ord("R")):
                keys.append("r")
            elif byte in (ord("w"), ord("W")):
                keys.append("w")
            elif byte in (ord("q"), ord("Q")):
                keys.append("q")
        return keys

    def flush_escape(self) -> list[str]:
        """Turn a standalone Escape key into a key after a short grace period."""
        if self._buffer == bytearray(b"\x1b"):
            self._buffer.clear()
            return ["escape"]
        return []


class TerminalKeyReader:
    """Set stdin to noncanonical no-echo mode and restore it reliably."""

    def __init__(
        self,
        stream: TextIO = sys.stdin,
        *,
        escape_grace_sec: float = 0.05,
    ) -> None:
        self.stream = stream
        self.escape_grace_sec = escape_grace_sec
        self._fd: Optional[int] = None
        self._saved_termios: Optional[list] = None
        self._saved_flags: Optional[int] = None
        self._parser = KeySequenceParser()
        self._escape_started_monotonic: Optional[float] = None

    @staticmethod
    def require_tty(stream: TextIO = sys.stdin) -> None:
        if not stream.isatty():
            raise RuntimeError(
                "teleop_recorder requires an interactive TTY. "
                "Run it with ssh -t and ros2 run, not ros2 launch."
            )

    def __enter__(self) -> "TerminalKeyReader":
        self.require_tty(self.stream)
        self._fd = self.stream.fileno()
        self._saved_termios = termios.tcgetattr(self._fd)
        self._saved_flags = fcntl.fcntl(self._fd, fcntl.F_GETFL)

        attributes = termios.tcgetattr(self._fd)
        attributes[3] &= ~(termios.ICANON | termios.ECHO)
        attributes[6][termios.VMIN] = 0
        attributes[6][termios.VTIME] = 0
        termios.tcsetattr(self._fd, termios.TCSANOW, attributes)
        fcntl.fcntl(self._fd, fcntl.F_SETFL, self._saved_flags | os.O_NONBLOCK)
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        if self._fd is not None and self._saved_termios is not None:
            termios.tcsetattr(self._fd, termios.TCSANOW, self._saved_termios)
        if self._fd is not None and self._saved_flags is not None:
            fcntl.fcntl(self._fd, fcntl.F_SETFL, self._saved_flags)
        self._fd = None

    def poll(self, now_monotonic: Optional[float] = None) -> list[str]:
        if self._fd is None:
            raise RuntimeError("terminal reader is not active")
        now = time.monotonic() if now_monotonic is None else now_monotonic
        keys: list[str] = []
        while select.select([self._fd], [], [], 0.0)[0]:
            try:
                data = os.read(self._fd, 1024)
            except BlockingIOError:
                break
            if not data:
                raise EOFError("terminal input closed")
            keys.extend(self._parser.feed(data))

        if self._parser.waiting_for_escape_completion:
            if self._escape_started_monotonic is None:
                self._escape_started_monotonic = now
            elif now - self._escape_started_monotonic >= self.escape_grace_sec:
                keys.extend(self._parser.flush_escape())
                self._escape_started_monotonic = None
        else:
            self._escape_started_monotonic = None
        return keys
