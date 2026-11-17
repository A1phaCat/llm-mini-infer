"""Stop strings, checked on the detokenized text in the frontend.

Text that could still grow into a stop string is held back, so a stop string split across tokens (or steps) is never
streamed to the client. Only the longest suffix that is a proper prefix of some stop string is held: ordinary text
streams without delay. On a hit the text is truncated before the stop string (or after it with
`include_stop_str_in_output`); the caller then finishes the request and sends `Abort` to the engine.
"""

from __future__ import annotations


class StopStrings:
    __slots__ = ("include", "max_len", "stops")

    def __init__(self, stops: list[str], include_in_output: bool = False):
        self.stops = [s for s in stops if s]
        self.include = include_in_output
        self.max_len = max((len(s) for s in self.stops), default=0)

    def find(self, text: str, new_chars: int) -> tuple[str, int] | None:
        """The stop string completing earliest among the last `new_chars` characters, and where to cut `text`."""
        if not new_chars or not self.stops:
            return None
        best: tuple[str, int, int] | None = None  # (stop, start, end)
        for stop in self.stops:
            start = text.find(stop, max(0, len(text) - new_chars - len(stop) + 1))
            if start == -1:
                continue
            end = start + len(stop)
            if best is None or end < best[2]:
                best = (stop, start, end)
        if best is None:
            return None
        stop, start, end = best
        return stop, end if self.include else start

    def holdback(self, text: str, limit: int) -> int:
        """Length of the longest suffix of `text` (at most `limit` chars) that is a proper prefix of a stop string."""
        best = 0
        for stop in self.stops:
            k = min(len(stop) - 1, limit)
            while k > best:
                if text.endswith(stop[:k]):
                    best = k
                    break
                k -= 1
        return best


class OutputText:
    """Accumulated output text of one request, and what is safe to send so far."""

    __slots__ = ("sent", "stop_reason", "stops", "text")

    def __init__(self, stops: StopStrings | None):
        self.text = ""
        self.sent = 0
        self.stops = stops
        self.stop_reason: str | None = None

    def append(self, piece: str) -> str | None:
        """Add newly decoded text; returns the stop string if one completed (the text is then truncated)."""
        if not piece:
            return None
        self.text += piece
        if self.stops is None:
            return None
        hit = self.stops.find(self.text, len(piece))
        if hit is not None:
            self.stop_reason, cut = hit
            self.text = self.text[:cut]
        return self.stop_reason

    def take(self, final: bool) -> str:
        """Text not yet sent, minus a possible stop-string prefix unless the request is finished."""
        end = len(self.text)
        if not final and self.stops is not None:
            end -= self.stops.holdback(self.text, end - self.sent)
        if end <= self.sent:
            return ""
        out = self.text[self.sent:end]
        self.sent = end
        return out
