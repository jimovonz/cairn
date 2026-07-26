"""Every write path stamps a versioned, retractable source_ref (spec 1.9).

Provenance is only useful if (a) the stamp identifies the version that wrote
the row, and (b) bumping that version does not break matching against rows
written by earlier versions.
"""
import pysqlite3 as sqlite3

import cairn.analyser as analyser
import cairn.config as config
import cairn.ingest as ingest
import cairn.init_db as init_db
import cairn.query as query


def test_analyser_writes_a_versioned_stamp():
    assert analyser.ANALYSER_SOURCE_REF != analyser.ANALYSER_SOURCE_REF_PREFIX
    assert analyser.ANALYSER_SOURCE_REF.startswith(analyser.ANALYSER_SOURCE_REF_PREFIX)


def test_review_writeback_and_ingest_are_versioned():
    assert config.REVIEW_WRITEBACK_VERSION.startswith(config.REVIEW_WRITEBACK_PREFIX)
    assert config.REVIEW_WRITEBACK_VERSION != config.REVIEW_WRITEBACK_PREFIX
    assert config.INGEST_PIPELINE_VERSION


def test_extractor_digest_is_stable_and_short():
    a = ingest._extractor_versions_digest()
    assert a == ingest._extractor_versions_digest()
    assert len(a) == 12


def test_extractor_digest_changes_when_an_extractor_version_bumps(monkeypatch):
    before = ingest._extractor_versions_digest()
    bumped = dict(ingest.EXTRACTOR_VERSIONS)
    bumped["docs"] = bumped["docs"] + 1
    monkeypatch.setattr(ingest, "EXTRACTOR_VERSIONS", bumped)
    assert ingest._extractor_versions_digest() != before


def test_prefix_match_retracts_every_version_of_a_write_path(tmp_path, monkeypatch):
    """The property that makes versioning safe: a bump must not orphan rows
    written by earlier versions from bulk retraction."""
    db_path = str(tmp_path / "prov.db")
    old = init_db.DB_PATH
    init_db.DB_PATH = db_path
    try:
        init_db.init()
    finally:
        init_db.DB_PATH = old
    monkeypatch.setattr(query, "DB_PATH", db_path)

    conn = sqlite3.connect(db_path)
    for sref in ("analyser-session-arc",        # pre-versioning rows
                 "analyser-session-arc-v1",
                 "analyser-session-arc-v2",
                 "review-writeback-v1"):        # a different write path
        conn.execute(
            "INSERT INTO memories (type, topic, content, source_ref) VALUES (?,?,?,?)",
            ("fact", f"t-{sref}", "c", sref))
    conn.commit()
    conn.close()

    n = query.archive_by_source_ref("analyser-session-arc%", "bad analyser", like=True)
    assert n == 3          # all three analyser versions, including pre-versioning

    conn = sqlite3.connect(db_path)
    survivors = conn.execute(
        "SELECT source_ref FROM memories WHERE archived_reason IS NULL").fetchall()
    conn.close()
    assert [r[0] for r in survivors] == ["review-writeback-v1"]
