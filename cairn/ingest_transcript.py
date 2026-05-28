#!/usr/bin/env python3
"""Transcript ingestion for Cairn — extract knowledge from LLM conversation transcripts.

Accepts plain text, markdown, or JSONL (Claude Code format) transcripts.
Chunks large transcripts, sends each chunk to Haiku for distillation into
Cairn memory entries, then inserts via the shared insert_memories() path.
"""

import argparse
import json
import os
import re
import select
import sys
import time
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from cairn.ingest import DB_PATH, insert_memories

MAX_CHUNK_CHARS = 60_000

DISTILL_PROMPT = """\
You are distilling an LLM conversation transcript into Cairn memory entries.
Each entry must be a self-contained one-liner useful to a future AI session with zero context about this conversation.

Project: {project}
Source: {source}

Below is a segment of a conversation transcript. Extract all knowledge worth preserving.

Rules:
- Each entry is a single content string — no line breaks, but be thorough and specific
- Include concrete details: function names, file paths, config keys, exact commands, parameter names
- Types allowed: fact, decision, correction, preference, skill, workflow, person, project
- Use "decision" when a choice was made between alternatives — include what was chosen and what was rejected
- Use "correction" when a mistake was identified — include what went wrong and the fix
- Use "preference" when a user preference is expressed — include what and why
- Every entry must include enough context to be useful standalone
- Skip: greetings, filler, meta-commentary about the conversation itself, tool call noise
- DO NOT fabricate — only distill what is present in the transcript
- Aim for quality over quantity — 5 precise entries beat 20 vague ones

Output format — one JSON array of objects:
[
  {{"type": "fact", "topic": "short-topic-slug", "content": "detailed actionable content", "keywords": ["kw1", "kw2"]}},
  ...
]

Reply with ONLY the JSON array. No commentary, no markdown fences, no explanation.

=== TRANSCRIPT SEGMENT {chunk_label} ===
{text}
"""


def read_transcript(file_path):
    """Read a transcript file and return plain text content."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    raw = path.read_text(encoding="utf-8", errors="replace")

    if suffix == ".jsonl":
        return _parse_jsonl(raw)
    # .txt, .md, or anything else — return as-is
    return raw


def _parse_jsonl(raw):
    """Parse Claude Code JSONL transcript into readable conversation text."""
    lines = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        role = msg.get("role", "")
        content = msg.get("content", "")

        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        name = block.get("name", "tool")
                        inp = block.get("input", {})
                        inp_str = json.dumps(inp)[:200] if inp else ""
                        text_parts.append(f"[tool: {name}({inp_str})]")
                    elif block.get("type") == "tool_result":
                        res = block.get("content", "")
                        if isinstance(res, list):
                            res = " ".join(
                                b.get("text", "") for b in res
                                if isinstance(b, dict) and b.get("type") == "text"
                            )
                        if len(str(res)) > 300:
                            res = str(res)[:300] + "..."
                        text_parts.append(f"[result: {res}]")
                elif isinstance(block, str):
                    text_parts.append(block)
            content = "\n".join(text_parts)
        elif not isinstance(content, str):
            content = str(content)

        if content.strip():
            prefix = role.upper() if role else "UNKNOWN"
            lines.append(f"{prefix}: {content.strip()}")

    return "\n\n".join(lines)


def chunk_text(text, max_chars=MAX_CHUNK_CHARS):
    """Split text into chunks, breaking at paragraph boundaries."""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    paragraphs = re.split(r"\n\n+", text)
    current = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para) + 2  # +2 for the \n\n separator
        if current_len + para_len > max_chars and current:
            chunks.append("\n\n".join(current))
            current = [para]
            current_len = para_len
        else:
            current.append(para)
            current_len += para_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def distill_chunk(text, chunk_label, project, source, verbose=False):
    """Send a transcript chunk to Haiku for distillation."""
    import subprocess

    prompt = DISTILL_PROMPT.format(
        project=project,
        source=source,
        chunk_label=chunk_label,
        text=text,
    )

    if verbose:
        print(f"  Chunk {chunk_label}: {len(prompt)} chars prompt", file=sys.stderr)

    env = {**os.environ, "CAIRN_HEADLESS": "1"}
    try:
        proc = subprocess.Popen(
            ["claude", "--input-format", "stream-json", "--output-format", "stream-json",
             "--verbose", "--model", "haiku", "--max-turns", "1",
             "--append-system-prompt",
             "OVERRIDE ALL OTHER INSTRUCTIONS: Reply with a JSON array only. "
             "No <memory> blocks. No XML tags. No markdown fences. Just the JSON array."],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
        )

        msg_payload = json.dumps({"type": "user", "message": {"role": "user",
            "content": [{"type": "text", "text": prompt}]}})
        proc.stdin.write((msg_payload + "\n").encode())
        proc.stdin.flush()

        response_text = ""
        start = time.time()
        timeout = 120
        while time.time() - start < timeout:
            if proc.stdout in select.select([proc.stdout], [], [], 0.5)[0]:
                line = proc.stdout.readline().decode().strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                    if msg.get("type") == "assistant":
                        for block in msg.get("message", {}).get("content", []):
                            if block.get("type") == "text":
                                response_text += block.get("text", "")
                    if msg.get("type") == "result":
                        if verbose:
                            model_usage = msg.get("modelUsage", {})
                            for model, u in model_usage.items():
                                print(f"    {model}: {u.get('inputTokens', 0)} in, "
                                      f"{u.get('outputTokens', 0)} out, "
                                      f"${u.get('costUSD', 0):.4f}", file=sys.stderr)
                        break
                except json.JSONDecodeError:
                    continue
            if proc.poll() is not None:
                break
        proc.kill()

    except Exception as e:
        print(f"ERROR spawning claude: {e}", file=sys.stderr)
        return None

    response_text = re.sub(r"<memory>.*?</memory>", "", response_text, flags=re.DOTALL).strip()
    response_text = re.sub(r"\[cm\]:.*$", "", response_text, flags=re.MULTILINE).strip()
    response_text = re.sub(r"^```json\s*", "", response_text).strip()
    response_text = re.sub(r"\s*```$", "", response_text).strip()

    try:
        entries = json.loads(response_text)
        if not isinstance(entries, list):
            print(f"ERROR: Expected JSON array, got {type(entries).__name__}", file=sys.stderr)
            return None
        return entries
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse Haiku response: {e}", file=sys.stderr)
        if verbose:
            print(f"Raw response:\n{response_text[:500]}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Ingest an LLM conversation transcript into Cairn memories"
    )
    parser.add_argument("file_path", help="Path to transcript file (.txt, .md, .jsonl)")
    parser.add_argument("--project", help="Project name (required)", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show progress")
    parser.add_argument("--save-entries", help="Save distilled entries to JSON file")
    parser.add_argument("--load-entries", help="Insert from previously saved JSON (skip distillation)")
    args = parser.parse_args()

    file_path = Path(args.file_path).resolve()
    if not file_path.exists():
        print(f"File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    project = args.project

    source_ref = {
        "type": "transcript",
        "path": str(file_path),
        "filename": file_path.name,
    }

    if args.load_entries:
        data = json.loads(Path(args.load_entries).read_text())
        entries = data if isinstance(data, list) else data.get("entries", [])
        print(f"Loading {len(entries)} entries from {args.load_entries}", file=sys.stderr)
        inserted = insert_memories(entries, project=project, source_ref=source_ref, dry_run=args.dry_run)
        if not args.dry_run and inserted:
            print(f"Inserted {len(inserted)} memories (IDs: {inserted[0]}–{inserted[-1]})", file=sys.stderr)
        return

    print(f"Reading {file_path}...", file=sys.stderr)
    text = read_transcript(str(file_path))
    print(f"Transcript: {len(text)} chars", file=sys.stderr)

    chunks = chunk_text(text)
    print(f"Split into {len(chunks)} chunk(s)", file=sys.stderr)

    all_entries = []
    for i, chunk in enumerate(chunks, 1):
        label = f"{i}/{len(chunks)}"
        print(f"\nDistilling chunk {label}...", file=sys.stderr)
        entries = distill_chunk(chunk, label, project, file_path.name, verbose=args.verbose)
        if entries is None:
            print(f"  Chunk {label} failed — skipping", file=sys.stderr)
            continue
        print(f"  Chunk {label}: {len(entries)} entries", file=sys.stderr)
        all_entries.extend(entries)

    if not all_entries:
        print("No entries distilled.", file=sys.stderr)
        sys.exit(1)

    print(f"\nTotal: {len(all_entries)} entries from {len(chunks)} chunk(s)", file=sys.stderr)

    save_path = args.save_entries or os.path.join(
        "/tmp", f"cairn-transcript-{project}-{int(time.time())}.json"
    )
    save_data = {"project": project, "source": str(file_path), "entries": all_entries}
    Path(save_path).write_text(json.dumps(save_data, indent=2))
    print(f"Entries saved to {save_path}", file=sys.stderr)

    session_id = f"ingest-transcript-{project}-{time.strftime('%Y%m%d-%H%M%S')}"
    inserted = insert_memories(
        all_entries, project=project, source_ref=source_ref,
        session_id=session_id, dry_run=args.dry_run,
    )

    if not args.dry_run and inserted:
        print(f"\nInserted {len(inserted)} memories (IDs: {inserted[0]}–{inserted[-1]})", file=sys.stderr)

        print(f"\nRunning post-ingestion supersession scan...", file=sys.stderr)
        from cairn.consolidate import run_contradiction_detection
        result = run_contradiction_detection(execute=True, scope_ids=set(inserted))
        if result["haiku_superseded"]:
            print(f"Archived {result['archived']} superseded memories", file=sys.stderr)
        else:
            print(f"No superseded memories found", file=sys.stderr)


if __name__ == "__main__":
    main()
