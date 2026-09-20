---
name: cairn-remediation
description: The 2026-07 write-path gate programme: stages, gates, measurement caveats, subsystem tiers, and non-goals.
---

## Active remediation programme (2026-07) — write-path gate

**A staged remediation programme is running: `docs/spec-remediation-2026-07.md`.
Read it before proposing or starting new Cairn work.**

- **The gate is write-path only** (Amendment 1; the earlier blanket freeze is
  retracted). Read-side work — thresholds, rerankers, retrieval, default-off
  flags — ships freely, because a read-side error is bounded by the time it was
  live. Write-path work (schema, corpus writes, archive/delete, replication)
  lands only when its writes are **attributable** via `source_ref` and
  **retractable in bulk** — a flag flipped off does not retract writes made
  while it was on.
- **Gates are data-volume, not date, based** — several stages need accumulated
  `memory_deliveries` / `metrics` rows before they can be validated, so the
  programme is applied over many sessions. Check the Status table in the spec
  for the current stage before acting.
- **Two measurement facts that invalidate naive analysis** (baseline
  2026-07-26): 94.6% of `memory_deliveries` rows have **no negative class**
  (untagged rows recorded only positives; non-engagement is indistinguishable
  from never-scored), so never compute an engagement rate across
  `engaged_method` strata — only the ~1,198 lexical rows carry a usable base
  rate. And enforcement events (~24% of stop events) currently conflate hard
  blocks with staged nudges.
- **Subsystem tiers** are published in README (Subsystem maturity): supported /
  experimental / frozen, with per-tier guarantees. Anything experimental that
  writes to the durable store is off by default.
- **Do not re-propose** items in the spec's Non-goals table — passive decay,
  first-prompt suppression, student floor recalibration, semantic engagement
  threshold tuning, and subsystem deletion are each rejected there with reasons.

Update the spec's Status table in the same commit as any stage change, and
append to its Amendment log rather than rewriting stages in place.
