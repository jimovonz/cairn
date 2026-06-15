"""Streaming [cm]-block stripper for the cairn API proxy.

The model emits a trailing ``[cm]: # '{...}'`` memory block on every response.
This stripper removes that block from the streamed text the client receives (so
it is never displayed or stored), while capturing the verbatim block for cairn's
own processing and for byte-exact re-injection into later requests (cache fidelity).

``CmTextStripper`` is the transport-agnostic core: feed it the assistant text
pieces in order, it returns the text to forward and accumulates ``captured``.
The byte-exact invariant holds: ``"".join(emitted) + captured == "".join(fed)``,
so the original assistant turn is reconstructible as ``client_stored_text + captured``.
"""

MARKER = "[cm]: # "
_WS = "\n\r\t "


class CmTextStripper:
    """Strip the trailing [cm] block from a stream of text pieces.

    Holds back ``len(MARKER) - 1`` chars so a marker split across feeds is still
    detected before any of it is emitted. It additionally never emits a trailing
    run of whitespace, because that whitespace may turn out to be the separator
    in front of a marker that has not yet arrived — without this, char-by-char
    streaming would leak the ``\\n\\n`` before ``[cm]`` (the separator can only be
    reclaimed by the backtrack-trim if it is still buffered when the marker is
    seen). Once the marker is seen, everything from it onward is captured and
    suppressed.
    """

    def __init__(self):
        self._buf = ""
        self._suppress = False
        self.captured = ""

    def feed(self, text: str) -> str:
        if self._suppress:
            self.captured += text
            return ""
        self._buf += text
        idx = self._buf.find(MARKER)
        if idx >= 0:
            cut = idx
            while cut > 0 and self._buf[cut - 1] in _WS:
                cut -= 1
            emit = self._buf[:cut]
            self.captured = self._buf[cut:]
            self._buf = ""
            self._suppress = True
            return emit
        # No marker yet. Emit everything except (a) the last len(MARKER)-1 chars
        # (a marker may be split across feeds) and (b) any trailing whitespace
        # run abutting that tail (it may be a marker separator not yet revealed).
        hold = len(MARKER) - 1
        if len(self._buf) > hold:
            cut = len(self._buf) - hold
            while cut > 0 and self._buf[cut - 1] in _WS:
                cut -= 1
            emit = self._buf[:cut]
            self._buf = self._buf[cut:]
            return emit
        return ""

    def flush(self) -> str:
        """Emit any held-back tail at end of message (no-marker case)."""
        if self._suppress:
            return ""
        emit = self._buf
        self._buf = ""
        return emit

    @property
    def stripped(self) -> bool:
        return self._suppress


# --- Comprehensive response-text stripper -----------------------------------
#
# CmTextStripper handles only the single trailing [cm] block. The proxy must
# hide *every* Cairn artifact the model can emit into its visible text:
#   - trailing memory block:  [cm]: # '...'   /  [cairn-memory]: # '...'
#                             (suppress from the marker to end of message)
#   - inline memory notes:    <memory_note>...</memory_note>
#                             (remove the span, text continues after)
#   - link-def notes:         [cairn-note]: # '...'  /  [cn]: # '...'
#                             (remove the whole line)
# Every removed span is captured verbatim so the original assistant turn is
# byte-exactly reconstructible (``emitted + "".join(captured) == fed``) for
# cache-faithful re-injection and for the Stop hook's memory capture.

CM_MARKERS = ("[cm]: # ", "[cairn-memory]: # ")
NOTE_LINE_MARKERS = ("[cairn-note]: # ", "[cn]: # ")
MEMNOTE_OPEN = "<memory_note>"
MEMNOTE_CLOSE = "</memory_note>"

# Longest opener we must avoid emitting a partial of when no full match is seen.
_ALL_OPENERS = CM_MARKERS + NOTE_LINE_MARKERS + (MEMNOTE_OPEN,)
_MAX_OPENER = max(len(m) for m in _ALL_OPENERS)

# States
_NORMAL = "normal"
_IN_MEMNOTE = "memnote"   # consuming until </memory_note>
_IN_NOTELINE = "noteline"  # consuming a [cairn-note]/[cn] line until newline
_IN_CM = "cm"             # suppress everything to end of message


class ResponseTextStripper:
    """Streaming remover for all Cairn artifacts in assistant text.

    Feed assistant text pieces in order; ``feed`` returns the cleaned text to
    forward and appends each removed artifact (verbatim) to ``self.captured``
    (a list of strings). The byte-exact invariant holds:
    ``"".join(all feed returns) + "".join(self.captured) == "".join(fed)`` once
    ``flush`` has run.
    """

    def __init__(self):
        self._buf = ""
        self._state = _NORMAL
        self._pending = ""   # accumulates the current artifact being captured
        self.captured: list[str] = []
        self.original = ""   # verbatim concat of all fed text (byte-exact source)
        self.cm_block = ""   # the trailing [cm]/[cairn-memory] block, verbatim

    # -- helpers --
    def _earliest_opener(self):
        """Return (idx, marker, kind) for the earliest complete opener in _buf,
        or (None, None, None). kind in {'cm','noteline','memnote'}."""
        best_idx = None
        best = (None, None)
        for m in CM_MARKERS:
            i = self._buf.find(m)
            if i >= 0 and (best_idx is None or i < best_idx):
                best_idx, best = i, (m, "cm")
        for m in NOTE_LINE_MARKERS:
            i = self._buf.find(m)
            if i >= 0 and (best_idx is None or i < best_idx):
                best_idx, best = i, (m, "noteline")
        i = self._buf.find(MEMNOTE_OPEN)
        if i >= 0 and (best_idx is None or i < best_idx):
            best_idx, best = i, (MEMNOTE_OPEN, "memnote")
        return best_idx, best[0], best[1]

    def feed(self, text: str) -> str:
        self.original += text
        self._buf += text
        out = ""
        while True:
            if self._state == _IN_CM:
                self._pending += self._buf
                self._buf = ""
                return out
            if self._state == _IN_MEMNOTE:
                j = self._buf.find(MEMNOTE_CLOSE)
                if j < 0:
                    # keep waiting; hold all but a possible split close tag
                    hold = len(MEMNOTE_CLOSE) - 1
                    if len(self._buf) > hold:
                        self._pending += self._buf[:-hold]
                        self._buf = self._buf[-hold:]
                    return out
                end = j + len(MEMNOTE_CLOSE)
                self._pending += self._buf[:end]
                self.captured.append(self._pending)
                self._pending = ""
                self._buf = self._buf[end:]
                self._state = _NORMAL
                continue
            if self._state == _IN_NOTELINE:
                j = self._buf.find("\n")
                if j < 0:
                    self._pending += self._buf
                    self._buf = ""
                    return out
                self._pending += self._buf[:j]  # newline stays in the visible stream
                self.captured.append(self._pending)
                self._pending = ""
                self._buf = self._buf[j:]
                self._state = _NORMAL
                continue

            # _NORMAL
            idx, marker, kind = self._earliest_opener()
            if idx is not None:
                if kind == "cm":
                    cut = idx
                    while cut > 0 and self._buf[cut - 1] in _WS:
                        cut -= 1
                    out += self._buf[:cut]
                    self._pending = self._buf[cut:]
                    self._buf = ""
                    self._state = _IN_CM
                    return out
                # noteline / memnote: emit text before the marker untouched
                out += self._buf[:idx]
                self._pending = self._buf[idx:idx + len(marker)]
                self._buf = self._buf[idx + len(marker):]
                self._state = _IN_NOTELINE if kind == "noteline" else _IN_MEMNOTE
                continue

            # No complete opener: emit safe prefix, hold a tail that could be a
            # split opener (plus trailing whitespace, a possible [cm] separator).
            hold = _MAX_OPENER - 1
            if len(self._buf) > hold:
                cut = len(self._buf) - hold
                while cut > 0 and self._buf[cut - 1] in _WS:
                    cut -= 1
                out += self._buf[:cut]
                self._buf = self._buf[cut:]
            return out

    def flush(self) -> str:
        """End of message: emit/finalise held data."""
        if self._state == _IN_CM:
            if self._pending:
                self.cm_block = self._pending
                self.captured.append(self._pending)
                self._pending = ""
            return ""
        if self._state in (_IN_MEMNOTE, _IN_NOTELINE):
            # Unterminated artifact at end of stream — capture what we have.
            self._pending += self._buf
            self._buf = ""
            if self._pending:
                self.captured.append(self._pending)
                self._pending = ""
            return ""
        out = self._buf
        self._buf = ""
        return out

    @property
    def stripped(self) -> bool:
        return bool(self.captured)
