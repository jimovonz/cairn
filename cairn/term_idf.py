"""Document-frequency cache for engagement scoring.

WHY: engagement is lexical overlap between a memory and the response. Terms that
are generic *within this corpus* — the project's own vocabulary — appear in
almost any response about that project whether or not the memory was used. That
inflates standing-context layers (project-bootstrap shares vocabulary with
everything) relative to query-matched ones, which is the wrong direction: it
makes the layers hardest to judge look strongest. Weighting by inverse document
frequency makes a term worth what it actually discriminates.

Bigrams are included deliberately. Shared phrasing is much stronger evidence of
influence than shared topic words, and IDF handles that automatically — a
bigram is rarer than either of its unigrams, so it earns a higher weight without
any special-casing.

CACHED, NOT RECOMPUTED. This is called from the Stop hook, a short-lived
process; scanning the corpus per turn would be untenable. The table lives in the
ephemeral DB (derived data, rebuildable) and refreshes only when the corpus has
moved materially.
"""
from typing import Iterable, Optional
import math
import os

from cairn.ingest import sqlite3  # pysqlite3 guard — never stdlib on a cairn WAL DB

REBUILD_IF_GROWN = 0.05      # rebuild once the corpus changes by >5%
MIN_DF = 2                   # a term seen once discriminates nothing useful
_MEM_CACHE: dict = {"df": None, "n": 0}


def _eph_path(p: Optional[str] = None) -> str:
    if p:
        return p
    return (os.environ.get("CAIRN_EPHEMERAL_DB_PATH")
            or os.path.join(os.path.dirname(os.path.abspath(__file__)), "cairn-ephemeral.db"))


def _durable_path(p: Optional[str] = None) -> str:
    if p:
        return p
    return (os.environ.get("CAIRN_DB_PATH")
            or os.path.join(os.path.dirname(os.path.abspath(__file__)), "cairn.db"))


def _ensure(conn) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS term_df ("
                 "term TEXT PRIMARY KEY, df INTEGER NOT NULL)")
    conn.execute("CREATE TABLE IF NOT EXISTS term_df_meta ("
                 "key TEXT PRIMARY KEY, value TEXT)")


def _corpus_size(durable_path: Optional[str] = None) -> int:
    try:
        d = sqlite3.connect(f"file:{_durable_path(durable_path)}?mode=ro", uri=True)
        try:
            return d.execute(
                "SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL "
                "AND (archived_reason IS NULL OR archived_reason = '')").fetchone()[0]
        finally:
            d.close()
    except sqlite3.Error:
        return 0


def build(eph_path: Optional[str] = None, durable_path: Optional[str] = None,
          force: bool = False) -> int:
    """(Re)build the DF table from live memories. Returns terms written."""
    from cairn.relevance import _engagement_tokens
    conn = sqlite3.connect(_eph_path(eph_path))
    conn.execute("PRAGMA busy_timeout=8000")
    try:
        _ensure(conn)
        n_now = _corpus_size(durable_path)
        row = conn.execute("SELECT value FROM term_df_meta WHERE key='corpus_n'").fetchone()
        n_prev = int(row[0]) if row and row[0] else 0
        if not force and n_prev and n_now and abs(n_now - n_prev) / max(n_prev, 1) < REBUILD_IF_GROWN:
            return 0
        d = sqlite3.connect(f"file:{_durable_path(durable_path)}?mode=ro", uri=True)
        df: dict[str, int] = {}
        try:
            for topic, content, kw in d.execute(
                    "SELECT topic, content, keywords FROM memories WHERE deleted_at IS NULL "
                    "AND (archived_reason IS NULL OR archived_reason = '')"):
                for t in _engagement_tokens(f"{topic or ''} {content or ''} {kw or ''}"):
                    df[t] = df.get(t, 0) + 1
        finally:
            d.close()
        conn.execute("DELETE FROM term_df")
        conn.executemany("INSERT INTO term_df (term, df) VALUES (?,?)",
                         [(t, c) for t, c in df.items() if c >= MIN_DF])
        conn.execute("INSERT INTO term_df_meta (key,value) VALUES ('corpus_n',?) "
                     "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str(n_now),))
        conn.commit()
        _MEM_CACHE["df"] = None
        return len(df)
    except sqlite3.Error:
        return 0
    finally:
        conn.close()


def _load(eph_path: Optional[str] = None) -> tuple[dict, int]:
    if _MEM_CACHE["df"] is not None:
        return _MEM_CACHE["df"], _MEM_CACHE["n"]
    df, n = {}, 0
    try:
        conn = sqlite3.connect(f"file:{_eph_path(eph_path)}?mode=ro", uri=True)
        try:
            df = {t: c for t, c in conn.execute("SELECT term, df FROM term_df")}
            row = conn.execute("SELECT value FROM term_df_meta WHERE key='corpus_n'").fetchone()
            n = int(row[0]) if row and row[0] else 0
        finally:
            conn.close()
    except (sqlite3.Error, ValueError):
        pass
    _MEM_CACHE["df"], _MEM_CACHE["n"] = df, n
    return df, n


def idf(term: str, eph_path: Optional[str] = None) -> float:
    """Inverse document frequency, floored at 0.

    An unseen term gets the maximum weight: absent from the corpus means maximally
    discriminating, and it is also the safe direction — over-crediting a rare term
    risks a false positive, under-crediting one silently loses the signal.
    """
    df, n = _load(eph_path)
    if not n:
        return 1.0          # no cache yet — degrade to unweighted, never to zero
    return math.log((n + 1) / (df.get(term, 0) + 1))


def weights(terms: Iterable[str], eph_path: Optional[str] = None) -> dict[str, float]:
    return {t: idf(t, eph_path) for t in terms}


def max_idf(eph_path: Optional[str] = None) -> float:
    _, n = _load(eph_path)
    return math.log((n + 1) / 1) if n else 1.0


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    print(f"terms written: {build(force=force)}")
