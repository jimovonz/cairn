# org-index — an org-wide git index (prototype)

Two orthogonal layers plus a verifier, motivated by a real incident: the VCS
bringup script `tools/board_test.py` was stranded on the unmerged branch
`JO_Prod_Bringup` and invisible from the branch we were on. It was recoverable
only because Cairn happened to remember it. This index makes that a one-line
lookup and flags such at-risk work *before* it's forgotten.

## The three components

| Layer | Script | Indexes | Answers |
|-------|--------|---------|---------|
| **Locatability** | `org_index.py` | every repo × every branch via `gh api` (no clones) | where is file X on *any* branch; what unmerged work is going stale |
| **Interface/symbol** | `interface_registry.py` | `IMPORTS_FROM` edges from per-repo `.code-review-graph/graph.db` | who consumes shared module X across repos; which interfaces have high cross-repo blast radius |
| **Verifier** | `cairn_verify.py` | joins Cairn location-claim memories ↔ locatability index | which memories point at files that moved, vanished, or live only on unmerged branches |

The two index layers are deliberately separate: locatability is **branch-aware**
(catches stranded work); the interface registry is the **default-branch surface**
(catches cross-repo coupling). Neither subsumes the other.

## Locatability layer — `org_index.py`

```bash
# nightly: walk the whole org (skips archived by default)
./org_index.py build --org robotics-plus

# where does this file live, on any branch?
./org_index.py find board_test.py
#   debian-var  JO_Prod_Bringup  [ahead]  198d old  tools/board_test.py  ⚠ unmerged

# what unmerged work is aging out? (the 'work at risk' report)
./org_index.py stranded --stale-days 90

./org_index.py branches debian-var      # ahead/behind/status per branch
./org_index.py stats
```

**API economy.** Only the default branch and branches that are *ahead* (have
unique commits) get their file tree walked — a branch with `ahead_by == 0` is an
ancestor of default, so its files are already covered. This bounds calls to
roughly `repos × (1 + ahead_branches)` tree fetches. `debian-var` (8 branches)
cost ~17 calls. Trees flagged `truncated` (huge repos) are logged.

## Interface/symbol layer — `interface_registry.py`

Requires per-repo graphs built by `code-review-graph` (`crg build`). Rolls their
`IMPORTS_FROM` edges into one cross-repo consumer registry — no new extraction.

```bash
./interface_registry.py build --roots /mnt/ssd/Projects     # scan local graphs
./interface_registry.py consumers ugv_proto_msgs            # 126 sites / 10 repos
./interface_registry.py producers --min-repos 3             # shared interfaces by fan-in
./interface_registry.py stats
```

**Known tuning knob.** `external` currently lumps third-party/stdlib (`sys`,
`numpy`) in with org-internal cross-repo modules (`ugv_proto_msgs`). To rank only
*internal* shared interfaces, filter `producers` to targets whose normalised name
matches a known repo/package name — cheap follow-up, not yet wired.

## Verifier — `cairn_verify.py`

```bash
./cairn_verify.py                  # scan all location-claim memories
./cairn_verify.py --id 15049       # check one memory
```

Classifies each `file:`/`repo:` memory claim: **OK** (on default branch) ·
**DRIFT** (only on unmerged branches — at risk) · **MISSING** (gone) ·
**UNKNOWN** (repo not indexed). Run after a fresh `org_index build` to find
memories whose location claims have gone stale.

## Suggested automation

Nightly (cron / CI):
```bash
./org_index.py build --org robotics-plus
./interface_registry.py build --roots /mnt/ssd/Projects   # if graphs are local
./org_index.py stranded --stale-days 90   > reports/stranded.txt
./cairn_verify.py                          > reports/cairn-drift.txt
```

## Status / not-yet

- Locatability + verifier: working against `robotics-plus` (live-tested on `debian-var`).
- Interface registry: working against 70 local graphs (45.8k import edges).
- Not built: full-text content search (the Sourcegraph tier — deliberately out
  of scope), internal-vs-third-party producer filtering, incremental builds
  (currently full rebuild per repo).
- Storage: stdlib `sqlite3` + WAL — single-writer nightly job, no concurrent
  multi-version access, so cairn's pysqlite3 requirement does not apply.
