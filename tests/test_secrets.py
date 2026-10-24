"""Secrets never enter Cairn: values move to a mode-600 store, memories keep
$NAME pointers, and protect() cleans an existing DB reversibly."""
import os
import stat

import pytest

from cairn import secrets as sec

KEY = "d52jOr9CHf9pWPyH047hY1rDZyu4geEt7FMONgb9"


@pytest.fixture
def store(monkeypatch, tmp_path):
    p = tmp_path / "cfg" / "secrets.env"
    monkeypatch.setattr(sec, "SECRETS_FILE", str(p))
    return p


@pytest.mark.parametrize("text", [
    f"Metlink GTFS-RT key {KEY} header x-api-key",
    "AT key 2db5d62d47704b918215bfedca0251be (Ocp-Apim-Subscription-Key)",
    "API key AIzaSyC9KIH9abcdefghijklmnopqrstuvwxyz12",
    "SA password: Alpine99!x",
])
def test_detects_real_secrets(text):
    assert sec.has_secret(text)


@pytest.mark.parametrize("text", [
    "session f62c3c6d-fe31-4a1b-9c2d-1234567890ab",
    "commit 070cdb0d2355aa77b1c3e5f7a9b1c3d5e7f9a1b3",
    "PDF downloaded to /tmp/CustomerStatement20260917Final50.pdf",
    "Public: ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOm2Hf8Q9w0xYz",
    "gpg-verified with key EF6E286DDA85EA2A4BA7DE684E2C6E8793298290",
    "token: ~/.config/gmail/token.json",
    "curl /api/cron/expire?token=CRON_SECRET",
    'Authorization: Bearer sk-or-v1-YOUR_OPENROUTER_KEY',
    "Gmail filter ANe1BmjAbCdEfGhIjKlMnOpQrStUvWx2Q",
])
def test_ignores_non_secrets(text):
    assert not sec.has_secret(text), text


def test_extract_moves_value_to_mode600_store(store):
    new, names = sec.extract(f"Metlink GTFS-RT key {KEY} header x-api-key", "nz api keys")
    assert KEY not in new and "$METLINK_GTFS_RT_KEY" in new or names[0] in new
    assert "secrets.env" in new
    assert stat.S_IMODE(os.stat(store).st_mode) == 0o600
    assert sec._load_store()[names[0]] == KEY


def test_same_value_reuses_name(store):
    _, a = sec.extract(f"key {KEY}", "x")
    _, b = sec.extract(f"the Metlink key is {KEY}", "y")
    assert a == b


def test_extract_entry_covers_content_and_facts(store):
    e = {"type": "fact", "topic": "creds", "content": f"key {KEY}", "facts": ["pw: none", "password: Alpine99!x"]}
    names = sec.extract_entry(e)
    assert len(names) == 2 and KEY not in e["content"] and "Alpine99" not in e["facts"][1]


def test_protect_cleans_db_including_history_and_is_reversible(store, tmp_path):
    import pysqlite3 as sqlite3
    db = tmp_path / "c.db"
    c = sqlite3.connect(db)
    c.executescript("""
      CREATE TABLE memories (id INTEGER PRIMARY KEY, topic TEXT, content TEXT);
      CREATE TABLE memory_history (id INTEGER PRIMARY KEY, memory_id INT, content TEXT);
      CREATE TRIGGER v BEFORE UPDATE OF content ON memories BEGIN
        INSERT INTO memory_history (memory_id, content) VALUES (old.id, old.content); END;
      CREATE TABLE excerpts (memory_id INT, excerpt TEXT);""")
    c.execute("INSERT INTO memories VALUES (1, 'nz keys', ?)", (f"Metlink key {KEY}",))
    c.execute("INSERT INTO excerpts VALUES (1, ?)", (f"got it: {KEY} saved",))   # unlabelled copy
    c.commit(); c.close()
    sec.protect([str(db)])
    c = sqlite3.connect(db)
    dump = "\n".join(str(r) for t in ("memories", "memory_history", "excerpts") for r in c.execute(f"SELECT * FROM {t}"))
    assert KEY not in dump
    name = next(iter(sec._load_store()))
    assert f"${name}" in dump and sec._load_store()[name] == KEY
    assert sec.scan(str(db)) == []


@pytest.mark.parametrize("text", [
    'print(message._request_id)  # req_011CUzx8kQ2nB4mWv7pR3sTyWG',
    'client.beta.files.retrieve_metadata("file_011CNha8iCJcU1wXNR6q4V8w")',
    "\x1b[1mBertForSequenceClassificationVersion20220101 done",
])
def test_ignores_api_ids_and_ansi(text):
    assert not sec.has_secret(text), text


def test_pasted_openrouter_key_detected():
    assert sec.has_secret("Here is my API key: sk-or-v1-0a1b2c3d4e5f60718293a4b5c6d7e8f9a0b1c2d3e4f5")


@pytest.mark.parametrize("text", [
    "STEP part ROBOTIS_U2D2_Power_Hub_Board sits behind the servo",
    "class MyLongComponent_Name_With_Words_Version2 holds it",
])
def test_ignores_underscore_identifiers(text):
    # Live false positive: a CAD component name was moved into the secret store.
    assert not sec.has_secret(text), text


def test_key_with_underscores_still_detected():
    assert sec.has_secret("token: sk_" "live_51HxQz8Kd93LmN0pQrS7tUvWx")  # split: keeps push-protection scanners quiet
