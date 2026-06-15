import sys, os, json, hashlib, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cairn.proxy.response_filter import CairnResponseFilter


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8")


def build_stream(text_chunks, index=0):
    """Build a realistic SSE byte stream for one text content block."""
    out = b""
    out += sse("message_start", {"type": "message_start", "message": {"usage": {"input_tokens": 10}}})
    out += sse("content_block_start", {"type": "content_block_start", "index": index,
                                       "content_block": {"type": "text", "text": ""}})
    for ch in text_chunks:
        out += sse("content_block_delta", {"type": "content_block_delta", "index": index,
                                           "delta": {"type": "text_delta", "text": ch}})
    out += sse("content_block_stop", {"type": "content_block_stop", "index": index})
    out += sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                                 "usage": {"output_tokens": 5}})
    out += sse("message_stop", {"type": "message_stop"})
    return out


def extract_text(forwarded: bytes):
    """Pull all text_delta text out of forwarded SSE bytes, in order."""
    text = ""
    for block in forwarded.split(b"\n\n"):
        for line in block.split(b"\n"):
            if line.startswith(b"data: "):
                try:
                    d = json.loads(line[6:])
                except Exception:
                    continue
                if d.get("type") == "content_block_delta" and d.get("delta", {}).get("type") == "text_delta":
                    text += d["delta"]["text"]
    return text


CM = "\n\n[cm]: # '{\"ok\":true,\"ctx\":\"s\",\"kw\":[\"x\"]}'"
NOTE = "<memory_note>fact/t: obs</memory_note>"


def run(stream_bytes, chunk_sizes):
    f = CairnResponseFilter()
    out = b""
    i = 0
    for n in chunk_sizes:
        out += f.process_chunk(stream_bytes[i:i+n])
        i += n
    out += f.process_chunk(stream_bytes[i:])
    out += f.flush()
    return out, f


def test_cm_stripped_from_stream():
    body = "Hello, here is the answer."
    stream = build_stream([body, CM])
    out, f = run(stream, [37])
    clean = extract_text(out)
    assert clean == body
    assert "[cm]" not in clean
    rec = f.finalize_record()
    assert rec["cm"].endswith("\"x\"]}'")
    assert rec["emitted_sha"] == hashlib.sha256(body.encode()).hexdigest()


def test_memnote_stripped_keeps_prose():
    body_chunks = ["Intro. ", NOTE, " conclusion."]
    stream = build_stream(body_chunks + [CM])
    out, f = run(stream, [11, 5, 50])
    clean = extract_text(out)
    assert clean == "Intro.  conclusion."
    rec = f.finalize_record()
    assert NOTE in rec["notes"]
    assert rec["cm"].startswith("\n\n[cm]")


def test_byte_chunking_invariant_random():
    rnd = random.Random(7)
    body = "A longer answer that spans multiple deltas. " + NOTE + " End."
    stream = build_stream([body[:20], body[20:], CM])
    for _ in range(60):
        sizes = []
        total = len(stream)
        pos = 0
        while pos < total:
            n = rnd.randint(1, 40)
            sizes.append(n); pos += n
        out, f = run(stream, sizes)
        clean = extract_text(out)
        assert "[cm]" not in clean
        assert "<memory_note>" not in clean
        assert clean == "A longer answer that spans multiple deltas.  End."


def test_passthrough_no_artifacts():
    body = "Just a plain answer with no cairn artifacts at all."
    stream = build_stream([body])
    out, f = run(stream, [9])
    assert extract_text(out) == body
    assert not f.stripped_anything
    assert f.finalize_record()["cm"] == ""


def test_tool_use_input_json_delta_passthrough():
    f = CairnResponseFilter()
    out = b""
    out += f.process_chunk(sse("content_block_start", {"type": "content_block_start", "index": 1,
                              "content_block": {"type": "tool_use", "id": "x", "name": "Bash", "input": {}}}))
    out += f.process_chunk(sse("content_block_delta", {"type": "content_block_delta", "index": 1,
                              "delta": {"type": "input_json_delta", "partial_json": "{\"cmd\":"}}))
    out += f.flush()
    assert b"input_json_delta" in out
    assert b'partial_json' in out
