---
name: cairn-subagent-capture
description: How subagent (SubagentStop) and code-review write-back capture memories: agent_transcript_path handling, review_writeback.py, associated_files overrides, dedup at cosine 0.85.
---

## Subagent & review memory capture

Two complementary paths capture knowledge that would otherwise be lost.

**SubagentStop capture (general).** Subagents (Task tool) emit `[cm]` memory blocks into their *own* transcripts, which the main-session `Stop` hook never sees. A `SubagentStop` hook (registered in `templates/global-settings.json`, no matcher → all agent types) routes the subagent's final message into `hooks/stop_hook.py`. The existing `is_subagent` branch (detected via `agent_id` **or** `hook_event_name=="SubagentStop"`) stores volunteered entries opportunistically and **skips enforcement** (no re-prompting a subagent). It reads the subagent's own transcript via `agent_transcript_path` (`own_transcript`) for storage + `--context` excerpts, while `session_id`/`transcript_path` stay the parent session so memories chain to it. Entries dedup at cosine 0.85 like any other.

**Review write-back (code-attached).** `cairn-review-writeback` (`cairn/review_writeback.py`) persists **durable review rationale** — the *why* that survives the fix (intentional couplings, accepted trade-offs, justified decisions captured at merge), **not raw transient bug findings** (a "PR has bug X" claim becomes false once fixed+merged and self-invalidates in cairn; default `type` is `decision` to nudge this) — keyed to the *target repo* and *changed file/symbol*, so `cairn-graph --knowledge SYMBOL` surfaces it later — it joins on `associated_files LIKE '%path%'` **and** an FTS `MATCH` on the symbol (FTS indexes topic/content/keywords/facts). Each finding sets explicit `associated_files=[abs, rel]` (a per-entry override added to `hooks/storage.insert_memories`, taking precedence over transcript-derived files) and carries the symbol in `content`/`keywords`/`facts`. It registers a synthetic `review-<project>-<commit>` session tagged to the target project (so project-scoped retrieval also surfaces findings), batches under `MAX_MEMORIES_PER_RESPONSE`, and dedups at cosine 0.85 (re-running a review is idempotent — no archiving).

Input is JSON on stdin (or `--file`): `{repo, project?, commit?, findings:[{file, symbol?, line?, type?, topic?, content, severity?, pr?, keywords?}]}`. Flags: `--dry-run`, `--json`. Invoke it from a review's end-step, e.g.:

| Situation | Agent invokes |
|---|---|
| A review surfaced a *durable* reason worth keeping (intentional coupling, justified trade-off, decision at merge) | pipe rationale JSON to `cairn-review-writeback` (resolve via `.venv/bin/`) |
| Raw transient bug findings on an open PR | do NOT write back — they self-invalidate once the bug is fixed |
| Want to preview the keyed memories first | add `--dry-run` |
