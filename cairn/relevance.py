"""Read-side memory relevance grading — Phase 1 (instrument) + Phase 2 (label).

The portable (T0) core of docs/spec-memory-relevance-grading.md:

  * build_context_window  — the cleaned recent-context representation (current
      prompt + the last turn) that keys a delivery and feeds the cross-encoder
      student. Reuses session_extract cleaning so write-time (logging) and
      read-time (student inference) reps are byte-identical — parity is
      load-bearing (spec A.3). The prior assistant response is capped so a long
      response can't swamp the short current prompt in the embedding.
  * is_self_referential_meta — the mechanical bucket-4 prefilter (drops
      cairn-about-cairn meta-memories). High-precision by design; gated +
      audited by the caller so it can't silently drop useful domain memories.
  * log_memory_deliveries — one memory_deliveries row per injected memory
      (ephemeral DB), mirroring calibration_inject.log_deliveries.
  * parse_relevance_grades / apply_relevance_grades — agent-as-teacher 0-3
      grades + hard-negative flag, written back to the matching delivery rows.

All heavy/generative work (T1: synthesised intent, label densification) lives in
async crons over this log, never here on the hot path.
"""

from __future__ import annotations

import re
try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError as _pysqlite_err:  # pragma: no cover
    import os as _os
    if _os.environ.get("CAIRN_ALLOW_STDLIB_SQLITE") == "1":
        import sqlite3  # explicit opt-in; stdlib SQLite may corrupt WAL DBs under concurrent multi-version access
    else:
        raise ImportError(
            "cairn requires pysqlite3 (a recent SQLite with WAL checkpoint-race fixes); "
            "the system stdlib sqlite3 can corrupt WAL-mode DBs under concurrent "
            "multi-version access. Install pysqlite3-binary, or set "
            "CAIRN_ALLOW_STDLIB_SQLITE=1 to override."
        ) from _pysqlite_err
from typing import Any, Optional

# The prior assistant response only supplies referents for anaphora, so cap it:
# a long response must not dominate the short current prompt in the embedding
# (the prompt-vs-response length asymmetry). Chars, not tokens — cheap + good enough.
# INPUT-DOMAIN INVARIANT (spec 1.10) — what this write path assumes about its
# input. Both ingest defects came from transplanting an invariant into a domain
# that violated it, which care at review time would not have caught.
INPUT_DOMAIN_INVARIANT = (
    "Assumes a delivered memory is USED IN THE SAME TURN it was delivered. "
    "Value realised later scores identically to never being used, so any layer "
    "with a multi-turn value horizon (first-prompt) is systematically "
    "under-measured by engagement."
)

PRIOR_RESPONSE_CAP = 600


def build_context_window(current_prompt: str, transcript_path: Optional[str] = None,
                         *, prior_response_cap: int = PRIOR_RESPONSE_CAP) -> str:
    """Cleaned (prev-user + capped prev-assistant + current-prompt) window.

    Pulls the prior exchange from the transcript via session_extract.load_turns
    (which already strips tool blocks, thinking, <cairn_context>, system reminders
    and [cm] defs). Fails soft to just the current prompt if the transcript is
    unavailable. Deterministic — same inputs give the same string on both sides.
    """
    cur = (current_prompt or "").strip()
    prior_user = ""
    prior_asst = ""
    if transcript_path:
        try:
            from cairn.session_extract import load_turns
            turns = [t for t in load_turns(transcript_path) if t.get("text")]
            prior_asst = next(
                (t["text"] for t in reversed(turns) if t["role"] == "assistant"), "")
            users = [t["text"] for t in turns if t["role"] == "user"]
            # If the current prompt is already the tail user turn, drop it so we
            # take the one *before* it as the prior-user referent.
            if users and cur and users[-1].strip() == cur:
                users = users[:-1]
            prior_user = users[-1] if users else ""
        except Exception:
            pass
    # Current prompt FIRST so it survives the cross-encoder's 512-token
    # right-truncation. The rerank window (~1500 tokens median) overflows the CE
    # limit ~3x, and HF truncation_side="right" keeps the front and drops the
    # tail. With the current prompt LAST (the old order) it was truncated out of
    # EVERY window that had a prior turn — the reranker scored relevance against
    # the prior turn, blind to the current question (measured: 100% of windows
    # with a prior turn dropped the current prompt; fixing it lifted ms-marco
    # AUC 0.579->0.695 and cut junk leak 87%->63% on the gold set). Leading with
    # the current prompt guarantees the gate sees what was asked; prior context
    # fills the remaining budget and is the part safely dropped.
    parts = [f"[user] {cur}"]
    if prior_user:
        parts.append(f"[prev user] {prior_user}")
    if prior_asst:
        capped = (prior_asst if len(prior_asst) <= prior_response_cap
                  else prior_asst[:prior_response_cap].rstrip() + " …")
        parts.append(f"[prev assistant] {capped}")
    return "\n".join(parts).strip()


# --- Bucket-4: self-referential meta ("cairn-about-cairn") ---------------------
# HIGH-PRECISION ONLY. In the cairn repo itself, legitimate domain memories mention
# "cairn" constantly, so we match meta-statements *about memory existence / coverage
# / gaps*, never the bare token "cairn". Tune additively; audit drops via metric.
_META_PATTERNS = [
    r"\bno (?:prior )?(?:memory|memories|record|entry|entries) (?:of|about|for|exist)",
    r"\bcairn (?:has no|contains no|has limited|lacks|knows nothing)",
    r"\bcairn contains a (?:profile|memory|record)",
    r"\b(?:should be|to be|will be|not yet|never) captured\b",
    r"\bcaptured when (?:shared|provided|mentioned)",
    r"\bcairn-about-cairn\b",
    r"\bself-referential meta",
    r"\bmemor(?:y|ies) (?:of [^.]{0,40} )?exists?\b",
    r"\b(?:limited|no) info(?:rmation)? (?:on|about) [^.]{0,40} in cairn",
]
_META_RE = re.compile("|".join(_META_PATTERNS), re.IGNORECASE)


# --- Question-form keyword sidecar (schema v14, calibration v7 port) -----------
# genA-v4 seeds memory keywords with question-form phrasings ("how do I X").
# Embedding each qf separately places it in the prompt-shaped vector region, so
# retrieval can score max over {content, topic, qf_i} — the fix calibration
# shipped for third-person-content vs first-person-prompt similarity clustering.
_QF_LEAD_RE = re.compile(
    r"^(how|why|what|when|where|which|who|can|does|do|is|are|should|will|did)\b",
    re.IGNORECASE,
)


def extract_question_forms(keywords) -> list[str]:
    """Question-form phrasings from a keyword list: ends with '?' or starts with
    an interrogative/auxiliary and has >=3 words (so bare topics like 'what sets'
    noise-keywords don't qualify but 'what sets Z' does)."""
    out: list[str] = []
    for kw in keywords or []:
        k = (kw or "").strip()
        if not k:
            continue
        if k.endswith("?") or (_QF_LEAD_RE.match(k) and len(k.split()) >= 3):
            out.append(k)
    return out


def store_qf_embeddings(conn, memory_id: int, keywords, embedder) -> int:
    """(Re)write the memory_qf_embeddings sidecar rows for one memory.

    Fail-soft: returns rows written, 0 on any error (a missing sidecar row only
    means the memory falls back to content+topic scoring). Caller owns commit.
    Caps at 8 qf strings per memory — beyond that keywords are spam, not intents."""
    try:
        qfs = extract_question_forms(keywords)[:8]
        if not qfs or embedder is None:
            return 0
        vecs = embedder.embed_batch(qfs, allow_slow=False)
        if not vecs:
            return 0
        conn.execute("DELETE FROM memory_qf_embeddings WHERE memory_id = ?", (memory_id,))
        n = 0
        for i, (qf, vec) in enumerate(zip(qfs, vecs)):
            if vec is None:
                continue
            conn.execute(
                "INSERT OR REPLACE INTO memory_qf_embeddings "
                "(memory_id, qf_index, qf_text, embedding) VALUES (?, ?, ?, ?)",
                (memory_id, i, qf, embedder.to_blob(vec)),
            )
            n += 1
        return n
    except Exception:
        return 0


def is_self_referential_meta(entry: dict[str, Any]) -> bool:
    """True if a memory is a bucket-4 self-referential meta-memory.

    Conservative: matches statements about what cairn does/doesn't remember, not
    domain content that merely mentions cairn. Never call on corrections (the spec
    keeps those ungated) — that gating is the caller's responsibility. Bootstrap
    layers ARE gated since 2026-07-02 (session-arc meta spam engaged at 0%).
    """
    text = entry.get("content") or entry.get("c") or ""
    return bool(text) and _META_RE.search(text) is not None


# --- Delivery log -------------------------------------------------------------
def _eph_path(eph_path: Optional[str]) -> str:
    if eph_path:
        return eph_path
    from cairn.config import EPHEMERAL_DB_PATH
    return EPHEMERAL_DB_PATH


def _next_turn_index(conn: sqlite3.Connection, session_id: str) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(turn_index), -1) + 1 FROM memory_deliveries WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def _score_components(r: dict[str, Any]) -> Optional[str]:
    """JSON of the heterogeneous score signals on a delivered row, so a label can
    be attributed to the exact scoring that produced it (ce_score is heterogeneous
    across the ms-marco->bge reranker transition; the composite blends CE + RRF +
    similarity). Only present keys are stored; returns None if nothing is known."""
    comp = {}
    # prefilter_n/postfilter_n make SUPPRESSION measurable. Without them a turn
    # whose candidates were all filtered leaves no trace, so any per-model
    # comparison drawn from delivery rows silently favours the permissive model.
    for src, dst in (("ce_score", "ce"), ("score", "composite"), ("rrf_score", "rrf"),
                     ("similarity", "sim"), ("confidence", "conf"),
                     ("prefilter_n", "pre_n"), ("postfilter_n", "post_n")):
        v = r.get(src)
        if v is not None:
            try:
                comp[dst] = round(float(v), 6)
            except (TypeError, ValueError):
                pass
    if not comp:
        return None
    import json
    return json.dumps(comp, separators=(",", ":"))


def log_memory_deliveries(delivered: list[dict[str, Any]], *, session_id: str,
                          context_text: str = "", context_vec: Optional[bytes] = None,
                          turn_index: Optional[int] = None,
                          layer: Optional[str] = None, project: Optional[str] = None,
                          eph_path: Optional[str] = None, _retry: bool = True) -> int:
    """Insert one memory_deliveries row per injected memory. Fail-soft: returns
    the count written (0 on any error) — instrumentation must never break delivery.

    Ranking provenance (step 1a) is stamped per row: reranker_model (the model that
    produced ce_score, from the daemon), score_components (JSON of all score signals),
    layer (the retrieval layer), and scope (project vs global, computed against
    `project` exactly as split_by_scope does)."""
    if not delivered or not session_id:
        return 0
    try:
        conn = sqlite3.connect(_eph_path(eph_path))
    except sqlite3.Error:
        return 0
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        if turn_index is None:
            turn_index = _next_turn_index(conn, session_id)
        n = 0
        for rank, r in enumerate(delivered):
            mid = r.get("id")
            if mid is None:
                continue
            ce = r.get("ce_score")
            if ce is None:
                ce = r.get("score")
            scope = "project" if (project and r.get("project") == project) else "global"
            conn.execute(
                "INSERT INTO memory_deliveries "
                "(session_id, turn_index, memory_id, context_text, context_vec, "
                " ce_score, served_rank, reranker_model, score_components, layer, scope, "
                " gate_status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, turn_index, int(mid), context_text, context_vec,
                 ce, r.get("served_rank", rank), r.get("reranker_model"),
                 _score_components(r), layer, scope, r.get("gate_status")),
            )
            n += 1
        conn.commit()
        return n
    except sqlite3.OperationalError as exc:
        # A missing column means this DB predates a schema change. Fail-soft was
        # designed so instrumentation never breaks delivery — but combined with a
        # new column in the INSERT it turns a migration gap into SILENT, total
        # data loss: every insert returns 0 and nothing is recorded. That
        # happened live on 2026-07-26 when gate_status shipped, costing hours of
        # deliveries before it was noticed. Migrate and retry once.
        # SQLite words this two ways: "has no column named X" from an INSERT and
        # "no such column: X" from a SELECT. Matching only the second is why the
        # first version of this guard did not fire on the very case it exists for.
        _msg = str(exc).lower()
        if not _retry or not ("no such column" in _msg or "has no column named" in _msg):
            return 0
        try:
            from cairn.init_db import init_ephemeral
            init_ephemeral(_eph_path(eph_path))
        except Exception:
            return 0
        return log_memory_deliveries(
            delivered, session_id=session_id, context_text=context_text,
            context_vec=context_vec, turn_index=turn_index, layer=layer,
            project=project, eph_path=eph_path, _retry=False,
        )
    except sqlite3.Error:
        return 0
    finally:
        conn.close()


# --- Phase 2: agent-as-teacher labels -----------------------------------------
_GRADE_RE = re.compile(r"^\s*(\d+)\s*:\s*([0-3])\s*(!)?\s*$")

# --- Relative fit labels (supersedes absolute 0-3 grading as the primary ask) --
#
# WHY RELATIVE: absolute grading produced labels on 0.5% of turns (340 of
# 62,238) and 2 hard-negatives in 541k deliveries. Two causes, both structural:
# an absolute scale asks "was this noise?", which is unanswerable for ambient
# standing context (project-bootstrap has ZERO labels at 6.2 entries/turn); and
# every guard against dishonest labelling ("a 0 is a confident claim", "omit if
# unsure") correctly makes silence the safe move. Relevance is relative anyway
# — an entry is noise *compared to better candidates*, not intrinsically — and
# a cross-encoder trains on pairwise preference (margin ranking loss), so pairs
# are the NATIVE training format, not a cheap approximation of grades.
#
# Grammar: {"best": [ids], "worst": [ids]} -> every best beats every worst.
# One best + one worst yields one pair; the agent never counts or ranks.

def parse_fit(raw: Any) -> list[tuple[int, int]]:
    """Parse {"best":[42],"worst":[17,8]} -> [(42,17),(42,8)] as (winner, loser).

    Accepts ints or numeric strings. Self-pairs are dropped. Returns [] for the
    explicit no-signal answers ("none", {}, absent) — which are meaningfully
    different from a missing field and are recorded by the caller as such.
    """
    def _ids(v: Any) -> list[int]:
        if not isinstance(v, (list, tuple)):
            return []
        out = []
        for x in v:
            try:
                out.append(int(x))
            except (TypeError, ValueError):
                continue
        return out

    if not isinstance(raw, dict):
        return []
    best, worst = _ids(raw.get("best")), _ids(raw.get("worst"))
    return [(w, l) for w in best for l in worst if w != l]


def apply_fit_labels(pairs: list[tuple[int, int]], *, session_id: str,
                     turn_index: Optional[int] = None,
                     durable_path: Optional[str] = None,
                     eph_path: Optional[str] = None) -> int:
    """Persist pairwise preferences to delivery_fit_pairs. Fail-soft.

    DURABLE DB, not ephemeral. These are the training labels the whole
    relevance-grading design rests on, and they are expensive to collect — a
    label needs an agent to have judged one specific turn's context, which
    cannot be reconstructed afterwards. "Ephemeral" advertises that a file is
    safe to delete, and everything else in there genuinely is rebuildable
    (term_df recomputes, memory_deliveries is instrumentation). Irreplaceable
    data does not belong behind that name.

    Stored as pairs rather than folded into memory_deliveries.grade because a
    preference is a relation between two deliveries, not a property of one —
    flattening it to a per-row score would reintroduce the absolute scale this
    replaces. eph_path is accepted and ignored for call-site compatibility.
    """
    if not pairs or not session_id:
        return 0
    try:
        conn = sqlite3.connect(_durable_path(durable_path))
    except sqlite3.Error:
        return 0
    # 30s, far longer than the 5s used elsewhere. This writes at most once per
    # turn, so waiting is nearly free, while the row is an irreplaceable training
    # label — an agent judgement of one specific turn's context that cannot be
    # reconstructed later. Against a fail-soft except, too short a timeout
    # discards the label silently and the loss reads as "no labels yet".
    conn.execute("PRAGMA busy_timeout=30000")
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS delivery_fit_pairs ("
            "  id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, turn_index INTEGER,"
            "  winner_id INTEGER NOT NULL, loser_id INTEGER NOT NULL,"
            "  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fit_pairs_session "
                     "ON delivery_fit_pairs(session_id, turn_index)")
        conn.executemany(
            "INSERT INTO delivery_fit_pairs (session_id, turn_index, winner_id, loser_id) "
            "VALUES (?,?,?,?)",
            [(session_id, turn_index, w, l) for w, l in pairs])
        conn.commit()
        return len(pairs)
    except sqlite3.Error:
        return 0
    finally:
        conn.close()


def sample_for_label(delivered: list[dict[str, Any]], k: int = 3,
                     seed: Optional[str] = None) -> list[dict[str, Any]]:
    """Pick k delivered entries at random to ask the agent about.

    RANDOM, NOT ENGAGEMENT-SELECTED. The existing labels are all on entries the
    agent noticed using, so the labelled set is selected on the outcome being
    measured — useless for calibrating a gate, which must be judged on entries
    it would have dropped. A uniform sample is the only way to get labels on
    the entries nobody noticed. Seeded by session+turn so the same turn always
    asks about the same ids (a retry must not resample).
    """
    if not delivered:
        return []
    import hashlib
    import random
    rnd = random.Random(hashlib.sha256((seed or "").encode()).hexdigest())
    pool = [d for d in delivered if d.get("id") is not None]
    if len(pool) <= k:
        return list(pool)
    return rnd.sample(pool, k)



def parse_relevance_grades(raw: Any) -> list[tuple[int, int, bool]]:
    """Parse ["42:3", "17:0!"] -> [(42,3,False),(17,0,True)]. memory_id:grade,
    trailing '!' = hard-negative (actively wrong/misleading; a distinct axis)."""
    out: list[tuple[int, int, bool]] = []
    if not isinstance(raw, (list, tuple)):
        return out
    for item in raw:
        if not isinstance(item, str):
            continue
        m = _GRADE_RE.match(item)
        if m:
            out.append((int(m.group(1)), int(m.group(2)), bool(m.group(3))))
    return out


def apply_relevance_grades(grades: list[tuple[int, int, bool]], *, session_id: str,
                           eph_path: Optional[str] = None) -> int:
    """Write 0-3 grade + hard_negative onto the most-recent delivery of each graded
    memory in this session (the one the agent just judged). Fail-soft."""
    if not grades or not session_id:
        return 0
    try:
        conn = sqlite3.connect(_eph_path(eph_path))
    except sqlite3.Error:
        return 0
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        n = 0
        dropped = []
        for mid, grade, hard in grades:
            cur = conn.execute(
                "UPDATE memory_deliveries SET grade = ?, hard_negative = ? WHERE id = ("
                "  SELECT id FROM memory_deliveries WHERE session_id = ? AND memory_id = ? "
                "  ORDER BY id DESC LIMIT 1)",
                (int(grade), 1 if hard else 0, session_id, int(mid)),
            )
            if cur.rowcount:
                n += cur.rowcount
            else:
                dropped.append(int(mid))
        # Silent-drop visibility: a grade matches 0 rows when no delivery for
        # (session_id, memory_id) exists — typically a compaction-chained session
        # where the delivery was logged under the parent but the grade arrives
        # under the child. Without this, the agent grades honestly yet the signal
        # vanishes. Record it as a metric so --stats surfaces the loss.
        if dropped:
            try:
                conn.execute(
                    "INSERT INTO metrics (event, session_id, detail, value) VALUES (?, ?, ?, ?)",
                    ("rg_grade_dropped", session_id,
                     ",".join(str(m) for m in dropped[:20]), len(dropped)),
                )
            except sqlite3.Error:
                pass
        conn.commit()
        return n
    except sqlite3.Error:
        return 0
    finally:
        conn.close()


def turn_rg_compliance(session_id: str, eph_path: Optional[str] = None) -> Optional[tuple[int, int]]:
    """(delivered_count, graded_count) for the most recent turn_index that had any
    deliveries in this session. Returns None if the session has no deliveries yet.
    Used by the Stop hook to track a periodic, non-blocking rg-grading nudge —
    never to gate/block, since forcing a grade produces fabricated labels (see
    the disabled inline-contradiction-enforcement precedent above)."""
    if not session_id:
        return None
    try:
        conn = sqlite3.connect(_eph_path(eph_path))
    except sqlite3.Error:
        return None
    try:
        row = conn.execute(
            "SELECT MAX(turn_index) FROM memory_deliveries WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if not row or row[0] is None:
            return None
        turn_index = row[0]
        counts = conn.execute(
            "SELECT COUNT(*), SUM(CASE WHEN grade IS NOT NULL THEN 1 ELSE 0 END) "
            "FROM memory_deliveries WHERE session_id = ? AND turn_index = ?",
            (session_id, turn_index),
        ).fetchone()
        delivered = counts[0] or 0
        graded = counts[1] or 0
        return (delivered, graded)
    except sqlite3.Error:
        return None
    finally:
        conn.close()


# --- Step 2: behavioural engagement signal ------------------------------------
# The cleaner, non-circular training spine (spec A.6 anti-Goodhart): rather than
# trust the agent's self-report, mechanically detect whether the response actually
# *used* each delivered memory. The marginal contribution of a memory is the set
# of its distinctive terms NOT already in the prompt — if the response surfaces
# those, the memory was drawn upon. This is the PRIMARY label; agent rg supplements.
_ENG_STOPWORDS = frozenset((
    "the", "and", "for", "are", "was", "were", "this", "that", "with", "from",
    "have", "has", "had", "not", "but", "you", "your", "our", "its", "their",
    "they", "them", "then", "than", "thus", "into", "onto", "over", "under",
    "use", "used", "uses", "using", "via", "per", "any", "all", "can", "will",
    "would", "should", "could", "may", "might", "must", "one", "two", "set",
    "get", "got", "new", "old", "now", "out", "off", "let", "see",
    "when", "what", "which", "where", "while", "here", "there", "been", "being",
    "does", "doing", "done", "such", "some", "more", "most", "much", "many",
    "also", "just", "only", "very", "like", "each", "both", "same", "other",
    "about", "above", "after", "before", "between", "because", "these", "those",
))
_ENG_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_\-]{2,}")


def _engagement_tokens(text: str, bigrams: bool = True) -> set[str]:
    """Lowercased alnum tokens of length >=3, minus stopwords, plus adjacent
    bigrams. Hyphen/underscore kept (identifiers like 'bge-reranker', 'ce_score'
    are exactly the distinctive terms we want). Returns a set — engagement is
    presence-based, not frequency.

    Bigrams ("term_a term_b") are emitted alongside unigrams because shared
    PHRASING is far stronger evidence of influence than shared topic words. They
    need no special weighting: a bigram is rarer than either of its parts, so IDF
    scores it higher automatically. term_idf.build() tokenizes through this same
    function, so document frequencies cover bigrams by construction.
    """
    if not text:
        return set()
    uni = [t for t in _ENG_TOKEN_RE.findall(text.lower()) if t not in _ENG_STOPWORDS]
    out = set(uni)
    if bigrams:
        out.update(f"{a} {b}" for a, b in zip(uni, uni[1:]))
    return out


# Terms below this IDF are corpus-generic — the project's own vocabulary, which
# appears in a response whether or not any given memory was used. Counting them
# as engagement is what made standing-context layers look strongest precisely
# where they are hardest to judge.
GENERIC_IDF_FLOOR = 0.5

# Cues that the response is DISPUTING a delivered memory rather than using it.
# Overlap is necessary to contradict something, so a correction currently scores
# as engagement — the two push a gate in opposite directions and must not be
# collapsed. Cue-based and deliberately conservative: it flags candidates for
# review, it is not a claim about semantic entailment.
_POLARITY_CUES = re.compile(
    r"\b(?:no longer|not (?:true|correct|the case|accurate)|superseded|outdated|"
    r"was wrong|is wrong|incorrect|actually,? |instead of|contradicts?|"
    r"stale|obsolete|deprecated|doesn'?t|does not|isn'?t|is not)\b", re.IGNORECASE)


def detect_polarity(response_text: str, matched_terms: set) -> Optional[str]:
    """'corrected' if the response disputes near a matched term, else 'used'.

    Proximity-scoped: a negation anywhere in a long response says nothing about
    THIS memory. Requires the cue within a sentence that also carries one of the
    memory's distinctive terms. Returns None when nothing matched.
    """
    if not matched_terms or not response_text:
        return None
    for sent in re.split(r"(?<=[.!?\n])\s+", response_text):
        low = sent.lower()
        if not _POLARITY_CUES.search(sent):
            continue
        if any(t in low for t in matched_terms):
            return "corrected"
    return "used"



def score_engagement(response_text: str, memory_text: str,
                     prompt_text: str = "", *, weighted: bool = True,
                     eph_path: Optional[str] = None) -> tuple[Optional[int], float]:
    """Did `response_text` use `memory_text`? Returns (engaged, score).

    Distinctive terms = memory tokens that are NOT in the prompt (the memory's
    marginal contribution — terms shared with the prompt would be repeated whether
    or not the memory helped, so crediting them confounds topic-match with use),
    then dropped if corpus-generic (below GENERIC_IDF_FLOOR).
      * engaged = 1 if >=2 distinctive terms appear in the response (strong: one
        shared term is plausibly coincidental, two is not), else 0.
      * score   = IDF-weighted fraction of distinctive terms surfaced (0..1), so a
        rare identifier counts for more than a common one.
      * (None, -1.0) when there are NO distinctive terms (memory redundant with the
        prompt, or wholly generic) — undecidable, so no signal rather than a false 0.

    NOT GROUND TRUTH FOR ORIENTATION MATERIAL. Repo/Confluence ingest memories are
    designed to point you at the right place, not to be quoted — they succeed
    without appearing in the response at all. Low engagement is their expected
    signature, not a defect, so this score must never be used to gate or retire
    them. Agent `fit` labels are the trustworthy signal for those layers.
    """
    mem = _engagement_tokens(memory_text)
    distinctive = mem - _engagement_tokens(prompt_text)
    if distinctive and weighted:
        from cairn import term_idf
        distinctive = {t for t in distinctive
                       if term_idf.idf(t, eph_path) >= GENERIC_IDF_FLOOR}
    if not distinctive:
        return None, -1.0
    matched = distinctive & _engagement_tokens(response_text)
    if weighted:
        from cairn import term_idf
        w = {t: term_idf.idf(t, eph_path) for t in distinctive}
        total = sum(w.values()) or 1.0
        score = sum(w[t] for t in matched) / total
    else:
        score = len(matched) / len(distinctive)
    return (1 if len(matched) >= 2 else 0), score

def _durable_path(durable_path: Optional[str]) -> str:
    if durable_path:
        return durable_path
    import os
    return (os.environ.get("CAIRN_DB_PATH")
            or os.path.join(os.path.dirname(os.path.abspath(__file__)), "cairn.db"))


def _cosine(a, b) -> float:
    """Full cosine between two vectors; 0.0 when either is degenerate."""
    import numpy as np
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        return 0.0
    denom = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
    return float(np.dot(a, b)) / denom if denom > 0 else 0.0


def semantic_engaged(cos_resp_mem: float, cos_ctx_mem: float,
                     threshold: Optional[float] = None,
                     margin: Optional[float] = None) -> bool:
    """Second-chance engagement verdict from embeddings rather than shared tokens.

    Two conditions must both hold. The response must be close to the memory
    (threshold), AND it must be closer than the context already was (margin) —
    without the margin, a memory that merely echoes the prompt would score as
    used, since the response naturally resembles its own prompt.
    """
    if threshold is None or margin is None:
        from cairn.config import ENGAGEMENT_SEM_THRESHOLD, ENGAGEMENT_SEM_MARGIN
        threshold = ENGAGEMENT_SEM_THRESHOLD if threshold is None else threshold
        margin = ENGAGEMENT_SEM_MARGIN if margin is None else margin
    return cos_resp_mem >= threshold and (cos_resp_mem - cos_ctx_mem) >= margin


def apply_engagement(response_text: str, *, session_id: str,
                     eph_path: Optional[str] = None,
                     durable_path: Optional[str] = None) -> int:
    """Stamp the behavioural engagement signal on this session's not-yet-evaluated
    deliveries (engaged_score IS NULL — i.e. those delivered for the current turn,
    since prior turns were scored at their own Stop). Fetches each memory's content/
    topic/keywords from the durable DB to compute distinctive-term overlap against
    the response. Fail-soft: returns rows updated (0 on any error)."""
    if not response_text or not session_id:
        return 0
    try:
        conn = sqlite3.connect(_eph_path(eph_path))
    except sqlite3.Error:
        return 0
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        rows = conn.execute(
            "SELECT id, memory_id, context_text FROM memory_deliveries "
            "WHERE session_id = ? AND engaged_score IS NULL", (session_id,),
        ).fetchall()
        if not rows:
            return 0
        mem_ids = sorted({int(r[1]) for r in rows})
        mem_text = {}
        mem_vec = {}
        try:
            dconn = sqlite3.connect(_durable_path(durable_path))
            try:
                qmarks = ",".join("?" * len(mem_ids))
                # `embedding` is optional: a durable DB predating the column must
                # still get lexical scoring rather than silently scoring nothing.
                try:
                    drows = dconn.execute(
                        f"SELECT id, content, topic, keywords, embedding FROM memories "
                        f"WHERE id IN ({qmarks})", mem_ids,
                    ).fetchall()
                except sqlite3.Error:
                    drows = [(r[0], r[1], r[2], r[3], None) for r in dconn.execute(
                        f"SELECT id, content, topic, keywords FROM memories "
                        f"WHERE id IN ({qmarks})", mem_ids,
                    ).fetchall()]
                for mid, content, topic, kw, emb in drows:
                    mem_text[int(mid)] = " ".join(p for p in (content, topic, kw) if p)
                    if emb:
                        try:
                            from cairn.embeddings import from_blob
                            mem_vec[int(mid)] = from_blob(emb)
                        except Exception:
                            pass  # unreadable vector — lexical verdict still stands
            finally:
                dconn.close()
        except sqlite3.Error:
            return 0
        # Once per call, not per row: polarity separates "used this" from
        # "disputed this", which overlap alone cannot distinguish.
        try:
            conn.execute("ALTER TABLE memory_deliveries ADD COLUMN polarity TEXT")
        except sqlite3.Error:
            pass
        # Pass 1 — lexical distinctive-term overlap. Verdicts are held in `pending`
        # rather than written straight out, so the semantic pass below can revise
        # them before a single write loop commits the final answer.
        pending = []
        for row_id, memory_id, ctx in rows:
            mt = mem_text.get(int(memory_id))
            if mt is None:
                continue  # memory deleted since delivery — leave unscored
            engaged, score = score_engagement(response_text, mt, ctx or "")
            # Polarity is a lexical judgement over the distinctive terms the
            # response actually reused, so it is computed here in pass 1 and
            # carried through unchanged — the semantic pass revises the engaged
            # verdict, not the stance.
            _dist = _engagement_tokens(mt) - _engagement_tokens(ctx or "")
            _pol = detect_polarity(response_text, _dist & _engagement_tokens(response_text))
            pending.append([row_id, int(memory_id), ctx or "", engaged, score,
                            "lexical", _pol])

        # Pass 2 — semantic second chance for rows the lexical pass called unengaged.
        # Recovers paraphrased use: the response applied the memory without reusing
        # its vocabulary. Fail-soft — any error leaves the lexical verdicts standing.
        try:
            from cairn.config import ENGAGEMENT_SEMANTIC_ENABLED
            if ENGAGEMENT_SEMANTIC_ENABLED and mem_vec:
                candidates = [p for p in pending if p[3] != 1 and p[1] in mem_vec]
                if candidates:
                    import cairn.embeddings as _emb
                    resp_vec = _emb.embed(response_text[:4000], allow_slow=False)
                    if resp_vec is not None:
                        ctx_vec = {}
                        for p in candidates:
                            if p[2] and p[2] not in ctx_vec:
                                ctx_vec[p[2]] = _emb.embed(p[2][:4000], allow_slow=False)
                        for p in candidates:
                            mv = mem_vec[p[1]]
                            crm = _cosine(resp_vec, mv)
                            ccm = _cosine(ctx_vec.get(p[2]), mv)
                            if semantic_engaged(crm, ccm):
                                # Semantic rows store cos(resp,mem) — same 0..1 range
                                # as the lexical overlap ratio it replaces. The method
                                # tag keeps the two bases separable in aggregate stats.
                                p[3], p[4], p[5] = 1, round(crm, 4), "semantic"
        except Exception:
            pass

        n = 0
        for row_id, _mid, _ctx, engaged, score, method, polarity in pending:
            try:
                conn.execute(
                    "UPDATE memory_deliveries SET engaged = ?, engaged_score = ?, "
                    "engaged_method = ?, polarity = ? WHERE id = ?",
                    (engaged, score, method, polarity, row_id),
                )
            except sqlite3.OperationalError:
                # Ephemeral DB predating engaged_method/polarity — score anyway,
                # untagged.
                conn.execute(
                    "UPDATE memory_deliveries SET engaged = ?, engaged_score = ? WHERE id = ?",
                    (engaged, score, row_id),
                )
            n += 1
        conn.commit()
        return n
    except sqlite3.Error:
        return 0
    finally:
        conn.close()
