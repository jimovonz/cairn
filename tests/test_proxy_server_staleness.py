"""Tests for the staleness-aware keep-alive helpers (start-fresh path).

These cover the pure decision logic without forking real daemons:
- _port_in_use detects a bound port (the EADDRINUSE guard against pid-file loss)
- _proxy_code_mtime / _daemon_start_mtime drive the stale-vs-current decision
"""
import sys, os, socket, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cairn.proxy import server


def test_port_in_use_detects_bound_listener():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((server.config.PROXY_HOST, 0))   # ephemeral free port
    port = s.getsockname()[1]
    s.listen(1)
    try:
        assert server._port_in_use(port) is True
    finally:
        s.close()
    # Once closed the port is free again (allow TIME_WAIT slack via REUSEADDR probe path)
    assert server._port_in_use(port) is False


def test_proxy_code_mtime_reflects_package_files():
    m = server._proxy_code_mtime()
    assert m > 0
    # server.py is one of the scanned files, so the newest mtime is >= its own
    assert m >= os.path.getmtime(server.__file__) - 1


def test_daemon_start_mtime_missing_pidfile_is_zero():
    # A port with no pid file -> 0.0, which is always < code mtime => treated stale
    assert server._daemon_start_mtime(59999) == 0.0
    assert server._daemon_start_mtime(59999) < server._proxy_code_mtime()


def test_pidfile_mtime_tracks_write(tmp_path, monkeypatch):
    # _daemon_start_mtime returns the pid file's mtime; a freshly written pid file
    # (current daemon) is newer than code => "up to date" no-op path.
    fake_pid = tmp_path / ".proxy-58888.pid"
    monkeypatch.setattr(server, "_pid_file", lambda port: str(fake_pid))
    fake_pid.write_text("12345")
    now = time.time()
    os.utime(fake_pid, (now, now))
    assert server._daemon_start_mtime(58888) >= now - 1
