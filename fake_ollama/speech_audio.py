"""Bounded PCM segmentation; deliberately has no ML or device dependencies."""
from __future__ import annotations

from array import array
from dataclasses import dataclass
import math
import sys

SAMPLE_RATE = 16000
FRAME_SAMPLES = 1600  # 100 ms, mono signed little-endian PCM16
FRAME_BYTES = FRAME_SAMPLES * 2


@dataclass
class AudioSegment:
    pcm: bytes
    start: float
    end: float
    overlap: bool = False


class Segmenter:
    """Pause-aware windows with a hard limit, pre-roll and bounded boundary overlap.

    Energy gating is not a neural speech detector. It suppresses silence, while
    preserving quiet material with a user-adjustable threshold. Every frame moves
    the source clock, including silence. Only a forced cut adds overlap.
    """
    def __init__(self, max_seconds=8, silence_seconds=.6, threshold=.003, min_seconds=0):
        self.limit = round(max_seconds * SAMPLE_RATE)
        self.silence_limit = round(silence_seconds * SAMPLE_RATE)
        self.threshold = threshold
        self.minimum = round(min(min_seconds, max_seconds) * SAMPLE_RATE)
        self.clock = 0
        self.buffer = bytearray()
        self.start = 0
        self.silent = 0
        self.voiced = False
        self.overlap = False

    def push(self, pcm: bytes):
        if len(pcm) % 2 or len(pcm) > FRAME_BYTES:
            raise ValueError('Expected at most 100 ms of PCM16')
        samples = array('h', pcm)
        if sys.byteorder != 'little':
            samples.byteswap()
        if not samples:
            return None
        if not self.buffer:
            self.start = self.clock
        rms = math.sqrt(sum(x*x for x in samples) / len(samples)) / 32768
        self.clock += len(samples)
        self.buffer.extend(pcm)
        if rms >= self.threshold:
            self.voiced = True
            self.silent = 0
        else:
            self.silent += len(samples)
        if not self.voiced:
            # 300 ms pre-roll prevents losing quiet syllables at onset.
            keep = min(len(self.buffer), 9600)
            self.buffer = self.buffer[-keep:]
            self.start = self.clock - keep // 2
            return None
        size = len(self.buffer) // 2
        forced = size >= self.limit
        if forced or (size >= self.minimum and self.silent >= self.silence_limit):
            result = AudioSegment(bytes(self.buffer), self.start / SAMPLE_RATE,
                                  self.clock / SAMPLE_RATE, self.overlap)
            tail = bytes(self.buffer[-12800:]) if forced else b''  # 400 ms
            self.buffer = bytearray(tail)
            self.start = self.clock - len(tail)//2
            self.overlap = bool(tail)
            self.voiced = False
            self.silent = 0
            return result
        return None

    def idle(self):
        """Flush speech when a device stops delivering packets (not silence frames)."""
        return self.finish()

    def finish(self):
        if not self.voiced or not self.buffer:
            return None
        result = AudioSegment(bytes(self.buffer), self.start / SAMPLE_RATE,
                              self.clock / SAMPLE_RATE, self.overlap)
        self.buffer.clear()
        self.voiced = False
        self.overlap = False
        self.silent = 0
        self.start = self.clock
        return result


def remove_boundary_overlap(previous: str, current: str, overlap: bool) -> str:
    if not overlap:
        return current
    # Only exact, nontrivial suffix/prefix matches at a known overlapped boundary.
    for length in range(min(len(previous), len(current), 160), 7, -1):
        if previous[-length:] == current[:length]:
            return current[length:].lstrip()
    return current
