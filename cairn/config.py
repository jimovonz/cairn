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
# score = w_similarity * similarity + w_recency * recency_decay + w_scope * scope_weight
# Confidence removed from scoring — it represents veracity (corroboration), not query relevance.
# Similarity handles per-query relevance, recency proxies staleness, scope prioritises project-local.
SCORE_W_SIMILARITY = 0.50
SCORE_W_CONFIDENCE = 0.0        # Disabled — veracity is not a ranking signal
SCORE_W_RECENCY = 0.15
SCORE_W_SCOPE = 0.05
RECENCY_HALF_LIFE_DAYS = 30        # Days after which recency weight halves

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
PROJECT_BOOTSTRAP_TYPES = "project,decision,preference,fact"  # Comma-separated types

# === Retrieval — Layer 1.5 (per-prompt push, subsequent prompts) ===
# Semantic search on every user message after the first. Higher threshold than Layer 1
# to avoid mid-session noise. Skips IDs already injected this session.
# Set L1_5_ENABLED=False to disable (default off — use Layer 3 pull-based instead).
L1_5_ENABLED = True                # Per-prompt semantic injection — disable via CAIRN_L1_5_ENABLED=0
L1_5_SIM_THRESHOLD = 0.55          # Stricter than Layer 1 (0.30) — only strong mid-session matches
L1_5_MAX_RESULTS = 3               # Keep injections tight on subsequent prompts

# === RRF (Reciprocal Rank Fusion) ===
# Fuses FTS5 keyword search with vector semantic search. Higher k smooths rank differences.
RRF_K = 60                         # Standard RRF constant — prevents single high rank from dominating

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
