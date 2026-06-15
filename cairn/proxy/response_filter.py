"""SSE response filter for the cairn proxy.

Wraps :class:`ResponseTextStripper` at the Server-Sent-Events layer. The
Anthropic streaming response delivers assistant text as ``content_block_delta``
events carrying ``text_delta`` chunks. This filter reassembles SSE event frames
across network chunk boundaries, runs the stripper over each text content
block, rewrites/suppresses the deltas so the client never sees any Cairn
artifact, and accumulates a capture record (the verbatim ``[cm]`` block and any
memory notes) for the Stop hook plus the byte-exact original for cache
re-injection.

Pure (no file I/O) so it is unit-testable; the proxy server persists the
capture record returned by :meth:`finalize_record`.
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional

from cairn.proxy.cm_filter import ResponseTextStripper, MEMNOTE_OPEN, NOTE_LINE_MARKERS


class CairnResponseFilter:
    def __init__(self):
        self.buffer = b""
        self._strippers: dict[int, ResponseTextStripper] = {}
        self.emitted_text = ""   # all forwarded text, concatenated (== what CC stores)
        self.done = False

    # -- public API -----------------------------------------------------------
    def process_chunk(self, chunk: bytes) -> bytes:
        self.buffer += chunk
        out = b""
        while True:
            nn = self.buffer.find(b"\n\n")
            rn = self.buffer.find(b"\r\n\r\n")
            if nn >= 0 and (rn < 0 or nn < rn):
                end = nn + 2
            elif rn >= 0:
                end = rn + 4
            else:
                break
            event = self.buffer[:end]
            self.buffer = self.buffer[end:]
            out += self._filter_event(event)
        return out

    def flush(self) -> bytes:
        out = b""
        if self.buffer:
            out += self._filter_event(self.buffer)
            self.buffer = b""
        # Emit any text held back by a stripper that never saw a block_stop.
        for idx, s in self._strippers.items():
            tail = s.flush()
            if tail:
                self.emitted_text += tail
                out += self._emit_text_delta(idx, tail)
        return out

    def finalize_record(self) -> dict:
        """Aggregate the captured artifacts across all content blocks."""
        cm = ""
        notes: list[str] = []
        for s in self._strippers.values():
            if s.cm_block and not cm:
                cm = s.cm_block
            for c in s.captured:
                if c == s.cm_block:
                    continue
                if c.startswith(MEMNOTE_OPEN) or any(
                    c.startswith(m) for m in NOTE_LINE_MARKERS
                ):
                    notes.append(c)
        return {
            "emitted_sha": hashlib.sha256(self.emitted_text.encode("utf-8")).hexdigest(),
            "cm": cm,
            "notes": notes,
        }

    @property
    def stripped_anything(self) -> bool:
        return any(s.stripped for s in self._strippers.values())

    # -- internals ------------------------------------------------------------
    def _emit_text_delta(self, index: int, text: str) -> bytes:
        evt = {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "text_delta", "text": text},
        }
        return f"event: content_block_delta\ndata: {json.dumps(evt)}\n\n".encode("utf-8")

    def _filter_event(self, event_data: bytes) -> bytes:
        try:
            text = event_data.decode("utf-8")
        except UnicodeDecodeError:
            return event_data

        lines = text.split("\n")
        out_lines: list[str] = []
        pending_event: Optional[str] = None

        for line in lines:
            if line.startswith("event:"):
                pending_event = line
                continue
            if line.startswith("data: "):
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    if pending_event is not None:
                        out_lines.append(pending_event); pending_event = None
                    out_lines.append(line)
                    continue
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    if pending_event is not None:
                        out_lines.append(pending_event); pending_event = None
                    out_lines.append(line)
                    continue
                result = self._filter_json_event(data)
                if result is None:
                    pending_event = None  # suppress whole event
                    continue
                if isinstance(result, list):
                    if pending_event is not None:
                        pending_event = None
                    for evt in result:
                        out_lines.append(f"event: {evt.get('type', '')}")
                        out_lines.append(f"data: {json.dumps(evt)}")
                        out_lines.append("")
                    continue
                if pending_event is not None:
                    out_lines.append(pending_event); pending_event = None
                if result is data:
                    out_lines.append(line)            # unchanged
                else:
                    out_lines.append(f"data: {json.dumps(result)}")
            else:
                if pending_event is not None:
                    out_lines.append(pending_event); pending_event = None
                out_lines.append(line)

        if pending_event is not None:
            out_lines.append(pending_event)

        if not out_lines or all(not l.strip() for l in out_lines):
            return b""
        result = "\n".join(out_lines)
        if not result.endswith("\n\n"):
            result += "\n\n"
        return result.encode("utf-8")

    def _filter_json_event(self, data: dict):
        etype = data.get("type", "")
        if etype == "message_stop":
            self.done = True
            return data
        if etype == "content_block_delta":
            delta = data.get("delta", {})
            if delta.get("type") != "text_delta":
                return data  # input_json_delta etc. — passthrough
            idx = data.get("index", 0)
            s = self._strippers.setdefault(idx, ResponseTextStripper())
            emit = s.feed(delta.get("text", ""))
            if not emit:
                return None  # nothing visible survived this delta
            self.emitted_text += emit
            data = dict(data)
            data["delta"] = dict(delta, text=emit)
            return data
        if etype == "content_block_stop":
            idx = data.get("index", 0)
            s = self._strippers.get(idx)
            if s is None:
                return data
            tail = s.flush()
            if tail:
                self.emitted_text += tail
                return [
                    {"type": "content_block_delta", "index": idx,
                     "delta": {"type": "text_delta", "text": tail}},
                    data,
                ]
            return data
        return data
