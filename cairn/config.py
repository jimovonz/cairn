"""
Cairn Configuration — All tunable parameters in one place.

Adjust these values to control retrieval quality, deduplication sensitivity,
confidence dynamics, and injection behaviour.
"""

# === Deduplication ===
DEDUP_THRESHOLD = 0.95          # Cosine similarity above which entries are considered near-identical

# === Confidence ===
# Saturating model: boost = BASE * (1 - current), penalty = BASE * (1 + current)
# This prevents runaway confidence inflation without introducing forgetting.
# At confidence 0.7: boost = 0.03, penalty = 0.34
# At confidence 0.9: boost = 0.01, penalty = 0.38
# At confidence 0.3: boost = 0.07, penalty = 0.26
CONFIDENCE_DEFAULT = 0.7        # Starting confidence for new memories
CONFIDENCE_BOOST = 0.1          # Base boost (actual: BOOST * (1 - confidence))
CONFIDENCE_PENALTY = 0.2        # Base penalty (actual: PENALTY * (1 + confidence))
CONFIDENCE_MIN = 0.0
CONFIDENCE_MAX = 1.0

# === Retrieval — Layer 3 (pull-based, LLM requests context) ===
L3_PROJECT_SIM_THRESHOLD = 0.25     # Minimum similarity for project-scoped results
L3_GLOBAL_SIM_WITH_PROJECT = 0.50   # Global threshold when project results exist
L3_GLOBAL_SIM_WITHOUT_PROJECT = 0.25  # Global threshold when no project results
L3_MAX_PROJECT_RESULTS = 7
L3_MAX_GLOBAL_RESULTS = 5

# === Retrieval — Layer 2 (keyword cross-project, unsolicited) ===
L2_SIM_THRESHOLD = 0.60            # Must be a strong match to justify unsolicited injection
L2_MAX_RESULTS = 3                 # Keep it tight — only the most relevant

# === Retrieval — Layer 1 (first-prompt push) ===
L1_SIM_THRESHOLD = 0.30            # Same as L3 project threshold
L1_MAX_RESULTS = 7

# === Composite scoring ===
# score = w_similarity * similarity + w_confidence * confidence + w_recency * recency_decay + w_scope * scope_weight
SCORE_W_SIMILARITY = 0.50
SCORE_W_CONFIDENCE = 0.30
SCORE_W_RECENCY = 0.15
SCORE_W_SCOPE = 0.05
RECENCY_HALF_LIFE_DAYS = 30        # Days after which recency weight halves

# === Injection quality gates ===
MIN_INJECTION_SIMILARITY = 0.35    # If max similarity < this, don't inject at all (garbage gate)
BORDERLINE_SIM_CEILING = 0.45      # If max similarity < this AND top score < BORDERLINE_SCORE_FLOOR, skip
BORDERLINE_SCORE_FLOOR = 0.50      # Minimum composite score for borderline similarity entries
RELATIVE_FILTER_RATIO = 0.7        # Keep only entries where similarity >= ratio * max_similarity
MAX_INJECTED_ENTRIES = 5            # Hard cap on entries injected per retrieval

# === Soft confidence inclusion ===
# Entries are included if: similarity >= SOFT_SIM_OVERRIDE OR confidence >= SOFT_CONF_FLOOR
SOFT_SIM_OVERRIDE = 0.60           # High similarity overrides low confidence
SOFT_CONF_FLOOR = 0.30             # Minimum confidence unless similarity override applies

# === Dominance suppression ===
DOMINANCE_EPSILON = 0.05           # If gap between top1 and top2 < epsilon, include both

# === Diversity filter (post-retrieval dedup) ===
DIVERSITY_SIM_THRESHOLD = 0.9      # Drop retrieved entries with cosine > this to already-selected entries

# === Write throttling ===
MAX_MEMORIES_PER_RESPONSE = 5      # Max memory entries stored per LLM response — drop lowest-confidence excess

# === Loop protection ===
MAX_CONTINUATIONS = 3              # Hard cap on consecutive re-prompts per session

# === Concurrency ===
DB_BUSY_TIMEOUT_MS = 5000          # SQLite busy timeout — wait up to 5s for lock release
