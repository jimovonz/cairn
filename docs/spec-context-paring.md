# Spec: Context Paring — proxy-level context token minimisation

Status: PROPOSAL (2026-07-05). Not yet implemented.
Owner: cairn proxy (`cairn/proxy/`). Applies to proxied sessions only (`c` launcher →
`ANTHROPIC_BASE_URL=127.0.0.1:8789`); bare `claude` sessions are unaffected.

## Goal

Minimise the token cost of the per-request resubmitted conversation history with
zero-to-positive fidelity impact. Measured motivation (2026-07-05, real sessions):

- Median large interactive session: ~300 API requests; history resubmitted on every one.
- `[cm]` blocks: ~1,150 chars avg → ~0.7M token-instances cumulatively resubmitted/session (median).
- Assistant response prose: ~10x that (~7M token-instances/session median).
- Tool results: 61% of the message stream; hook-output attachments ~63KB/session.
- Warm-session reads bill at 0.1x, but every >5-min idle gap makes the next request
  re-bill the ENTIRE history at 1x. Cold resumes are frequent in interactive use.
- Secondary win: every KB pared defers compaction — which is indiscriminate lossy
  summarization. Surgical paring with recovery paths displaces blunt compaction,
  so net fidelity is positive, not merely preserved.

## Invariants (all tiers)

1. **Submitted-once-is-frozen.** Content that has appeared in any API request is
   immutable except at: (a) its OWN first submission, (b) provably-cold boundaries
   (>300s since last outbound request — cache TTL expired, rewrite costs nothing),
   (c) explicit threshold events (context-pressure, compaction-style, paid deliberately).
   Intra-turn tool cycles submit as they go, so the only never-submitted content at a
   turn boundary is the final assistant text segment.
2. **Hindsight, not foresight.** Pre-submission transforms handle PROVABLE noise only.
   No mechanism exists to determine semantic redundancy before first submission —
   semantic pruning is retroactive, using evidence from turns that followed the content.
3. **Demotion, not deletion.** Every semantic pare leaves a recovery path (CCM cache
   stub, local transcript). The JSONL transcript keeps full fidelity always — paring
   changes only what goes upstream.
4. **Artifacts, not examples.** A replacement must not be mistakable for model output:
   history examples set emission norms (a minimal-valid `[cm]` block in every old turn
   teaches zero-capture). Use obvious pipeline markers + explain the mechanism in
   static rules.
5. **Deterministic and idempotent.** Canonical pared form per content identity
   (message id + content hash), persisted in a pare ledger, replayed byte-identically
   on every subsequent request. The proxy prompt-cache integrity guard is extended to
   verify the *pared* prefix; on any ledger miss/drift it fails open to unpared.
6. **Structural safety.** tool_use/tool_result pairing preserved (content replaced,
   never removed); tools/system tier never touched; last 2 turns always verbatim;
   mind the 20-block cache lookback when choosing pare points.

## Identifiability taxonomy (what may be pruned, by proof strength)

| Class | Proof | Examples | Timing |
|---|---|---|---|
| Provable-by-construction | artifact consumed / byte-equality | `[cm]` blocks (hook already stored), duplicate harness nags, identical re-injected graph blocks, pretty-JSON/ANSI in machine output | first submission |
| Provable-by-session-events | hard fact in tool log | file read then edited later (stale = misleading), same command re-run (old output superseded), error followed by successful retry, large result already consumed (next assistant msg exists) + CCM-recoverable | cold gap / lag-by-one |
| Statistical | distinctive-term reference scan over subsequent turns (reuse `apply_engagement` overlap scorer) | evidence-cold detail sections of old responses | cold gap, conservative |
| Convention | writing conventions (lead-with-outcome) | decides HOW to digest a span already marked prunable — never WHETHER | — |

## Phases

### Phase 1 — [cm] validity markers (build first)
- Change the existing `reinject_cm` path: instead of reinjecting the full captured
  block, inject a fixed marker: `[cm: captured]` (or `[cm: invalid]` for the rare
  unrecovered-invalid turn). NOT a minimal valid block (see invariant 4).
- Add a short section to `.claude/rules/memory-system.md` (static, byte-stable, no
  cache impact) explaining: prior blocks are parsed, archived, and replaced with
  markers; a full block is still required every turn; the marker is confirmation,
  not an example.
- In-session dedup signal moves from per-turn blocks to ONE consolidated topic digest:
  proxy derives a running `captured this session: t1, t2, ...` line from the
  `_cm_capture.jsonl` sidecar and injects it once per request near the volatile tail.
- Keep the last 1-2 turns' blocks intact (likeliest dedup collisions).
- Derivation is pure function of captured block → no new state beyond the existing
  sidecar; avoid extending the sha-map (known fragility).
- Expected: ~99% of the cm pot; zero fidelity loss; zero behavioural risk.

### Phase 2 — event-provable cold-gap pare (best proof-coverage per LoC)
At each cold boundary (>300s), one free full-history rewrite driven by joins over the
session's own tool log:
- **Superseded reads**: tool_result of a read whose file the session later edited →
  `[pared: superseded — file modified at turn N; re-read if needed]`. Fidelity-positive.
- **Superseded runs**: identical command line executed again later → keep latest
  output only; earlier becomes `[pared: re-run at turn N]`.
- **Resolved failures**: error output followed by successful retry → collapse to
  one-line error summary.
- **Hash dedup**: byte-identical content blocks → back-reference marker.
- Consumed injections: cairn_context blocks already graded, stale task-nags,
  hook notices already acted upon.

### Phase 3 — lag-by-one demotion of large consumed results
- Gates: size (> ~4KB) AND category (logs, listings, fetched pages — not short
  structured outputs whose exact values get re-referenced) AND the following
  assistant message exists (consumption evidence).
- Replace with digest + recovery stub: `[pared: full output via ccm-get <key>]`.
- Cost: bounded cache miss from nearest reachable prior cache entry (≤20 blocks) to
  tail — cheapest possible break point; wins decisively for large results.

### Phase 4 — statistical response-prose pare (cold gap, conservative)
- Responses older than keep-window: reference-scan detail sections against all
  subsequent turns (distinctive terms: identifiers, numbers, paths, quoted phrases).
- Prune only evidence-cold sections; keep lead paragraph + headers + referenced spans.
- Largest measured pot after tool results (~7M token-instances/session median),
  but weakest proofs — ships last, behind its own flag.

### Phase 5 — mechanical minification (portability tier)
- ANSI/banners/blank-runs/dup-lines/JSON-minify on tool results at first submission.
- Measured 0.3% residual on THIS machine (RTK/cch/CCM already compress at source) —
  build for uninstrumented deployments (expected 20-40% on structured output there),
  keep default-off locally.

## Non-goals
- No model-in-the-loop summarization in the hot path (latency, cost, fidelity risk).
- No per-turn rolling rewrite of frozen history (permanent cache-miss tax).
- No paring of: user text, tools/system tier, last 2 turns, thinking blocks
  (previous-turn thinking is not billed — nothing to win).

## Measurement & rollout
- Pare ledger records bytes-in/bytes-out per tier per request; dashboard panel later.
- Verify per session via `usage`: cache_read_input_tokens must stay high across pared
  requests (partial-prefix hits at breakpoints); input_tokens delta = realised saving.
- Flags: `CAIRN_PARE_CM`, `CAIRN_PARE_EVENTS`, `CAIRN_PARE_DEMOTE`, `CAIRN_PARE_PROSE`,
  `CAIRN_PARE_MINIFY` — each default off; enable in order after measuring the prior.
- Failure posture: any ledger inconsistency → send unpared (correctness over savings).
