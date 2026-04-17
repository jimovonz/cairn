"""
Cairn Configuration — All tunable parameters in one place.

Adjust these values to control retrieval quality, deduplication sensitivity,
confidence dynamics, and injection behaviour.
"""

# === Deduplication ===
DEDUP_THRESHOLD = 0.95          # Cosine similarity above which entries are considered near-identical

# === Veracity (confidence reframed) ===
# Confidence now represents accumulated veracity — how well-corroborated a memory is.
# + means "consistent with what I'm seeing" (corroboration signal)
# - means "not relevant here" (no effect on confidence — irrelevance is not evidence against truth)
# -! means "proven wrong" (contradiction annotation with reason)
# Confidence is NOT used in retrieval scoring — similarity, recency, and scope handle ranking.
CONFIDENCE_DEFAULT = 0.7        # Starting confidence for new memories (unverified)
CONFIDENCE_BOOST = 0.1          # Base boost (actual: BOOST * (1 - confidence))
CONFIDENCE_MIN = 0.0
CONFIDENCE_MAX = 1.0

# === Retrieval — Layer 3 (pull-based, LLM requests context) ===
L3_PROJECT_SIM_THRESHOLD = 0.25     # Minimum similarity for project-scoped results
L3_GLOBAL_SIM_WITH_PROJECT = 0.50   # Global threshold when project results exist
L3_GLOBAL_SIM_WITHOUT_PROJECT = 0.25  # Global threshold when no project results
L3_PROJECT_QUALITY_FLOOR = 0.45       # Project results below this don't raise the global threshold
L3_MAX_PROJECT_RESULTS = 7
L3_MAX_GLOBAL_RESULTS = 7

# === Retrieval — Layer 2 (keyword cross-project, unsolicited) ===
L2_SIM_THRESHOLD = 0.60            # Must be a strong match to justify unsolicited injection
L2_MAX_RESULTS = 3                 # Keep it tight — only the most relevant

# === Retrieval — Layer 1 (first-prompt push) ===
L1_SIM_THRESHOLD = 0.30            # Same as L3 project threshold
L1_MAX_RESULTS = 7

# === Composite scoring ===
# score = w_similarity * similarity + w_keywords * keyword_overlap + w_recency * recency_decay + w_scope * scope_weight
# Confidence removed from scoring — it represents veracity (corroboration), not query relevance.
# Similarity handles per-query relevance, keywords reward explicit topic tagging,
# recency is a tiebreaker only (staleness handled by archival/supersession), scope prioritises project-local.
SCORE_W_SIMILARITY = 0.50
SCORE_W_CONFIDENCE = 0.0        # Disabled — veracity is not a ranking signal
SCORE_W_KEYWORDS = 0.15         # Keyword overlap between query terms and memory keywords
SCORE_W_RECENCY = 0.05          # Tiebreaker — staleness handled by archival system
SCORE_W_SCOPE = 0.05
RECENCY_HALF_LIFE_DAYS = 30        # Days after which recency weight halves

# Memory types that apply universally regardless of project — biographical/cross-cutting facts
# about the user, contacts, preferences. These ignore the project scope penalty so they
# surface in any session, not just sessions in the project where they were captured.
SCOPE_BIAS_EXEMPT_TYPES = {"person", "preference"}

# === Thin-retrieval escalation ===
# When push retrieval (Layer 3) returns too few or too-weak results, the next stop hook
# fire forces escalation: the LLM must run query.py directly or re-declare context:
# insufficient with a refined need before proceeding. Catches the failure mode where the
# LLM trusts an empty push as authoritative absence.
THIN_RETRIEVAL_ESCALATION_ENABLED = True
THIN_RETRIEVAL_MIN_ENTRIES = 3          # Fewer than this → flag as thin
THIN_RETRIEVAL_TOP_SIM_THRESHOLD = 0.45 # Max similarity below this → flag as thin
THIN_RETRIEVAL_MAX_REMINDERS = 4        # Re-stage limit; abandons after this many ignored reminders

# === Injection quality gates ===
MIN_INJECTION_SIMILARITY = 0.35    # If max similarity < this, don't inject at all (garbage gate)
BORDERLINE_SIM_CEILING = 0.35      # If max similarity < this AND top score < BORDERLINE_SCORE_FLOOR, skip
BORDERLINE_SCORE_FLOOR = 0.50      # Minimum composite score for borderline similarity entries
RELATIVE_FILTER_RATIO = 0.7        # Keep only entries where similarity >= ratio * max_similarity
MAX_INJECTED_ENTRIES = 5            # Hard cap on entries injected per retrieval

# === Soft confidence inclusion (DISABLED) ===
# Confidence no longer gates retrieval. All memories are retrievable regardless of confidence.
# Confidence represents veracity/corroboration, not retrieval eligibility.
SOFT_SIM_OVERRIDE = 0.0            # Disabled — no confidence-based filtering
SOFT_CONF_FLOOR = 0.0              # Disabled — no confidence floor

# === Dominance suppression ===
DOMINANCE_EPSILON = 0.05           # If gap between top1 and top2 < epsilon, include both

# === Diversity filter (post-retrieval dedup) ===
DIVERSITY_SIM_THRESHOLD = 0.9      # Drop retrieved entries with cosine > this to already-selected entries

# === Trailing intent detection ===
TRAILING_INTENT_SIM_THRESHOLD = 0.65   # Similarity above which last sentence is flagged as unfulfilled intent

# === Content dedup thresholds ===
DISTINCT_VARIANT_SIM_THRESHOLD = 0.8   # Below this, same type+topic entries treated as distinct rather than updates
NEGATION_SIM_FLOOR = 0.6               # Minimum similarity for negation/contradiction check
WEAK_ENTRY_SCORE_FLOOR = 0.4           # Top score below this triggers weak-entry suppression

# === Write throttling ===
MAX_MEMORIES_PER_RESPONSE = 5      # Max memory entries stored per LLM response — drop lowest-confidence excess

# === Loop protection ===
MAX_CONTINUATIONS = 3              # Hard cap on consecutive re-prompts per session

# === Staged context ===
STAGED_CONTEXT_RETENTION_DAYS = 7   # Days to keep staged cross-project context for session resumption

# === Context bootstrapping ===
# Force a context: insufficient declaration if the LLM hasn't used layer 3 in N turns.
# This builds the habit through demonstrated value rather than rules alone.
CONTEXT_BOOTSTRAP_INTERVAL = 20    # Turns without a layer 3 request before forcing one
CONTEXT_BOOTSTRAP_FIRST_INTERVAL = 10  # First bootstrap fires earlier to seed context sooner
BOOTSTRAP_MAX_PER_SCOPE = 3        # Cap bootstrap retrieval results per scope (project/global)

# === Project bootstrap (CWD-based) ===
# On first prompt, inject top-N memories for the matched project, filtered to
# "standing context" types (project state, decisions, preferences, facts).
# Independent of prompt content — gives Claude project awareness from CWD alone.
PROJECT_BOOTSTRAP_ENABLED = True
PROJECT_BOOTSTRAP_MAX = 5           # Max memories to inject from project bootstrap
PROJECT_BOOTSTRAP_TYPES = "project,preference,fact"  # Comma-separated standing-context types
CORRECTION_BOOTSTRAP_MAX = 5        # Max behavioural corrections to inject per session

# === Correction trigger matching ===
# Stop hook compares response against stored correction triggers (embedded phrases
# describing what the bad response looks like). Blocks on match so the LLM can self-correct.
CORRECTION_TRIGGER_ENABLED = True
CORRECTION_TRIGGER_SIM_THRESHOLD = 0.45  # Similarity threshold for trigger match (tuned from real data)

# === Retrieval — Layer 1.5 (per-prompt push, subsequent prompts) ===
# Semantic search on every user message after the first. Higher threshold than Layer 1
# to avoid mid-session noise. Skips IDs already injected this session.
# Set L1_5_ENABLED=False to disable (default off — use Layer 3 pull-based instead).
L1_5_ENABLED = True                # Per-prompt semantic injection — disable via CAIRN_L1_5_ENABLED=0
L1_5_SIM_THRESHOLD = 0.55          # Stricter than Layer 1 (0.30) — only strong mid-session matches
L1_5_MAX_RESULTS = 3               # Keep injections tight on subsequent prompts

# === Query expansion — Type-prefix fan-out ===
# Memories are embedded as "{project} {type} {topic} {content}". A bare query misses the
# type prefix, reducing similarity. Fan-out searches with each type prefix and takes the
# max similarity per memory. ~7x more dot products per search (no extra model inference).
# Benchmarked: lifts MRR from 0.969→1.000 on easy benchmark, 0.875→0.881 on hard benchmark.
QUERY_EXPANSION_FANOUT = True       # Enable type-prefix fan-out in find_similar()

# === RRF (Reciprocal Rank Fusion) ===
# Fuses FTS5 keyword search with vector semantic search. Higher k smooths rank differences.
RRF_K = 60                         # Standard RRF constant — prevents single high rank from dominating

# === Mid-response memory checkpoints ===
# PostToolUse hook nudges the LLM to emit <memory_note> tags after high-signal tool calls.
# The stop hook collects and stores these notes. No extra LLM calls — the agent is already
# generating its next response; the note is just an inline tag.
CHECKPOINT_ENABLED = True
CHECKPOINT_COOLDOWN = 3            # Skip nudge if one fired within this many tool calls
CHECKPOINT_TOOLS = "Bash,Edit,Write"  # Tool types that can trigger a checkpoint nudge
CHECKPOINT_ERROR_PATTERNS = "error,failed,traceback,denied,not found,exception,fatal,panic"
CHECKPOINT_MIN_OUTPUT_LINES = 30   # Bash output above this line count triggers a nudge (even without errors)
CHECKPOINT_MAX_NOTES_PER_SESSION = 20  # Hard cap on memory_notes stored per session

# === Cross-encoder re-ranking ===
# After diversity filtering, re-score candidates using a cross-encoder that reads
# (query, memory) pairs jointly. Catches semantic relationships that cosine
# similarity on independent embeddings misses (paraphrase, entailment).
CROSS_ENCODER_ENABLED = True
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_WEIGHT = 0.6         # Blend: (1-w)*composite + w*cross_encoder
CROSS_ENCODER_MIN_CANDIDATES = 3   # Skip re-ranking if fewer candidates than this
CROSS_ENCODER_SCORE_FLOOR = 0.0    # Drop candidates scoring below this (raw CE score, not normalized)

# === NLI (Natural Language Inference) for consolidation ===
# Used by the consolidation pipeline to detect entailment between memory pairs.
# Same lazy-load pattern as cross-encoder — loaded in daemon on first use.
NLI_ENABLED = True
NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
NLI_ENTAILMENT_THRESHOLD = 0.7     # Score above this = entailment (memories say the same thing)
NLI_CONTRADICTION_THRESHOLD = 0.0  # Raw logit above this = possible contradiction (pre-filter for Haiku)

# === Memory consolidation ===
CONSOLIDATION_SIMILARITY_THRESHOLD = 0.85  # Bi-encoder cosine threshold for candidate clustering
CONSOLIDATION_MIN_CLUSTER_SIZE = 2         # Minimum entries to form a consolidation cluster
CONSOLIDATION_MAX_CLUSTER_SIZE = 10        # Cap cluster size for LLM summarisation prompt

# === Contradiction detection ===
CONTRADICTION_SIMILARITY_THRESHOLD = 0.70  # Balance coverage vs pair count (1935 pairs at 0.70)
CONTRADICTION_MAX_PAIRS = 2000             # High enough for full coverage at 0.70 threshold

# === Concurrency ===
DB_BUSY_TIMEOUT_MS = 5000          # SQLite busy timeout — wait up to 5s for lock release


# === Environment variable overrides ===
# Any config value above can be overridden by setting CAIRN_<NAME>=value.
# Example: CAIRN_DEDUP_THRESHOLD=0.90 lowers the dedup sensitivity.
import os as _os
_this = _os.sys.modules[__name__]
for _name in list(vars(_this)):
    if _name.startswith("_") or not _name.isupper():
        continue
    _env = f"CAIRN_{_name}"
    _val = _os.environ.get(_env)
    if _val is not None:
        _current = getattr(_this, _name)
        if isinstance(_current, bool):
            setattr(_this, _name, _val.lower() in ("1", "true", "yes"))
        elif isinstance(_current, float):
            setattr(_this, _name, float(_val))
        elif isinstance(_current, int):
            setattr(_this, _name, int(_val))
        elif isinstance(_current, str):
            setattr(_this, _name, _val)
