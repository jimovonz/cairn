"""Single source of truth for the cross-encoder passage rendering.

Both the training/labeling path (label_relevance._memtext) and the live inference
path (embeddings.py rerank) MUST render a memory into the SAME text, or the student
is trained on a different shape than it is served (train/inference parity). Keeping
the format here guarantees they cannot drift.
"""
from __future__ import annotations


def render_passage(type_, topic, content, keywords=None, facts=None,
                   *, enrich=False, content_cap=600, total_cap=900):
    """type topic: content  (+ keywords + facts when enrich=True).

    keywords: comma-separated string (memories.keywords).
    facts:    newline-separated key:value string (memories.facts).
    """
    base = f"{type_ or ''} {topic or ''}: {content or ''}"[:content_cap]
    if not enrich:
        return base
    parts = [base]
    kw = (keywords or "").strip()
    fx = (facts or "").strip()
    if kw:
        parts.append("keywords: " + kw.replace(",", ", "))
    if fx:
        parts.append("facts: " + " · ".join(f.strip() for f in fx.splitlines() if f.strip()))
    return "\n".join(parts)[:total_cap]
