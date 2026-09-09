"""Formal sensor contract and lightweight stream health accounting."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math


ALLOWED_DEPTH_ENCODINGS = frozenset({"16UC1", "32FC1"})
ALLOWED_COLOR_ENCODINGS = frozenset({"rgb8"})
ALLOWED_SEMANTIC_ENCODINGS = frozenset(
    {"mono8", "mono16", "8UC1", "16UC1", "32SC1"}
)


@dataclass(frozen=True)
class ImageContract:
    color_encoding: str
    depth_encoding: str
    semantic_encoding: str
    color_frame: str
    depth_frame: str
    semantic_frame: str

    def validate(self, expected_optical_frame: str) -> None:
        if self.color_encoding not in ALLOWED_COLOR_ENCODINGS:
            raise ValueError(f"unsupported color encoding: {self.color_encoding}")
        if self.depth_encoding not in ALLOWED_DEPTH_ENCODINGS:
            raise ValueError(f"unsupported depth encoding: {self.depth_encoding}")
        if self.semantic_encoding not in ALLOWED_SEMANTIC_ENCODINGS:
            raise ValueError(
                f"unsupported semantic encoding: {self.semantic_encoding}"
            )
        frames = {self.color_frame, self.depth_frame, self.semantic_frame}
        if frames != {expected_optical_frame}:
            raise ValueError("registered RGB, depth, and semantics must share optical frame")


@dataclass
class StreamWindow:
    max_samples: int = 120
    _stamps: deque[float] = field(default_factory=deque)
    received: int = 0
    out_of_order: int = 0

    def observe(self, stamp_seconds: float) -> None:
        stamp = float(stamp_seconds)
        if not math.isfinite(stamp):
            raise ValueError("timestamp must be finite")
        if self._stamps and stamp <= self._stamps[-1]:
            self.out_of_order += 1
        self._stamps.append(stamp)
        self.received += 1
        while len(self._stamps) > self.max_samples:
            self._stamps.popleft()

    @property
    def rate_hz(self) -> float:
        if len(self._stamps) < 2:
            return 0.0
        duration = self._stamps[-1] - self._stamps[0]
        return 0.0 if duration <= 0.0 else (len(self._stamps) - 1) / duration

