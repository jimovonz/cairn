"""start-fresh must reclaim a wedged daemon (port held, pid file lost) instead
of no-oping — otherwise a lost pid file pins stale proxy code forever."""
import sys, os, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cairn.proxy import server


class _Args:
    port = 8789
    debug = False


def test_pid_on_port_parses_ss(monkeypatch):
    fake = types.SimpleNamespace(
        stdout='LISTEN 0 128 127.0.0.1:8789 0.0.0.0:* users:(("python3",pid=4242,fd=7))\n')
    monkeypatch.setattr(server.subprocess, "run", lambda *a, **k: fake)
    assert server._pid_on_port(8789) == 4242


def test_pid_on_port_none_when_absent(monkeypatch):
    fake = types.SimpleNamespace(stdout="")
    monkeypatch.setattr(server.subprocess, "run", lambda *a, **k: fake)
    assert server._pid_on_port(8789) is None


def test_pid_on_port_survives_ss_missing(monkeypatch):
    def boom(*a, **k):
        raise OSError("no ss")
    monkeypatch.setattr(server.subprocess, "run", boom)
    assert server._pid_on_port(8789) is None


def test_start_fresh_reclaims_when_wedged(monkeypatch):
    calls = []
    monkeypatch.setattr(server, "is_running", lambda p: False)
    monkeypatch.setattr(server, "_port_in_use", lambda p: True)   # orphan holds port
    monkeypatch.setattr(server, "_reclaim_wedged", lambda p: calls.append(("reclaim", p)) or True)
    monkeypatch.setattr(server, "cmd_start", lambda a: calls.append(("start", a.port)))
    server.cmd_start_fresh(_Args())
    assert ("reclaim", 8789) in calls  # wedge detected and reclaimed
    assert ("start", 8789) in calls    # then a fresh start
    assert calls.index(("reclaim", 8789)) < calls.index(("start", 8789))


def test_start_fresh_no_reclaim_when_port_free(monkeypatch):
    calls = []
    monkeypatch.setattr(server, "is_running", lambda p: False)
    monkeypatch.setattr(server, "_port_in_use", lambda p: False)  # clean, down
    monkeypatch.setattr(server, "_reclaim_wedged", lambda p: calls.append("reclaim"))
    monkeypatch.setattr(server, "cmd_start", lambda a: calls.append("start"))
    server.cmd_start_fresh(_Args())
    assert "reclaim" not in calls      # nothing to reclaim
    assert "start" in calls


def test_start_fresh_noop_when_running_and_fresh(monkeypatch):
    calls = []
    monkeypatch.setattr(server, "is_running", lambda p: True)
    monkeypatch.setattr(server, "_daemon_start_mtime", lambda p: 100.0)
    monkeypatch.setattr(server, "_proxy_code_mtime", lambda: 50.0)  # daemon newer
    monkeypatch.setattr(server, "cmd_restart", lambda a: calls.append("restart"))
    monkeypatch.setattr(server, "cmd_start", lambda a: calls.append("start"))
    server.cmd_start_fresh(_Args())
    assert calls == []                 # healthy + fresh → no-op


def test_start_fresh_restarts_when_running_but_stale(monkeypatch):
    calls = []
    monkeypatch.setattr(server, "is_running", lambda p: True)
    monkeypatch.setattr(server, "_daemon_start_mtime", lambda p: 50.0)
    monkeypatch.setattr(server, "_proxy_code_mtime", lambda: 100.0)  # code newer
    monkeypatch.setattr(server, "cmd_restart", lambda a: calls.append("restart"))
    server.cmd_start_fresh(_Args())
    assert calls == ["restart"]
