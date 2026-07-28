"""Single source of truth for turning free text into a safe FTS5 MATCH query.

FTS5 reads punctuation inside a bare token as query syntax: `P2400-Drivetrain`
parses as a column filter and raises OperationalError("no such column:
Drivetrain") -- an uncaught exception, not an empty result. Project codes, file
paths and version strings trip this constantly.

Callers use this as a FALLBACK, not a rewrite: try the raw query first, and
only sanitize when FTS5 rejects it. That way deliberate FTS syntax (quoted
phrases, boolean operators, NEAR, prefix globs) keeps working untouched, and
the guard costs nothing on the common path.

hooks/retrieval.py needs none of this -- it already quotes each term when
building its OR query. This module exists so the interactive paths (query.py
CLI, graph.py --knowledge, dashboard search) do not each reinvent the guard.
"""
import re

_TOKEN = re.compile(r'"[^"]*"|\S+')
_OPERATORS = {"AND", "OR", "NOT", "NEAR"}
_SAFE = re.compile(r"^\w+\*?$")


def sanitize(query: str) -> str:
    """Quote bare tokens FTS5 would misparse, preserving deliberate syntax.

    Left alone: already-quoted phrases, boolean operators, and bare word
    tokens with an optional trailing `*` prefix glob.
    """
    if not query or not query.strip():
        return query
    out = []
    for tok in _TOKEN.findall(query):
        if tok.startswith('"') or tok.upper() in _OPERATORS or _SAFE.match(tok):
            out.append(tok)
        else:
            out.append('"' + tok.replace('"', '""') + '"')
    return " ".join(out)
