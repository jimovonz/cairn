"""
Dev-container extension auto-injector.

Subscribes to `docker events` and, when a dev-container starts, pushes any
VSIX files staged in CONTAINER_AUTO_INSTALL_VSIX_DIR into the new container
and runs `code --install-extension` on each.

Used by the cairn daemon to keep unpublishable personal extensions (e.g.
copilot-human-loop) available in every dev container without per-project
glue in `devcontainer.json`.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import subprocess
import threading
import time

log = logging.getLogger(__name__)


def _list_staged_vsixes(stage_dir: str) -> list[str]:
    if not os.path.isdir(stage_dir):
        return []
    return sorted(glob.glob(os.path.join(stage_dir, "*.vsix")))


def _is_dev_container(container_id: str) -> bool:
    """True if the container has the devcontainer.metadata label."""
    try:
        out = subprocess.run(
            ["docker", "inspect", "--format", "{{json .Config.Labels}}", container_id],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if out.returncode != 0:
            return False
        labels = json.loads(out.stdout.strip() or "{}") or {}
        return "devcontainer.metadata" in labels
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
        return False


_FIND_CODE_SERVER = (
    "for p in "
    "/vscode/vscode-server/bin/linux-*/*/bin/code-server "
    "$HOME/.vscode-server/bin/*/bin/code-server; do "
    "[ -x \"$p\" ] && { echo \"$p\"; exit 0; }; done; exit 1"
)


def _find_code_cli(container_id: str) -> str | None:
    """Locate the VS Code Server headless `code-server` binary inside the container.

    Avoids the user-facing `code` wrapper (remote-cli/code), which refuses to
    run outside a VS Code terminal session (errors "Command is only available
    in WSL or inside a Visual Studio Code terminal." with exit 0). The
    underlying `code-server` binary supports `--install-extension --force`
    headlessly with no IPC env vars required.

    Dev containers stage it under either `/vscode/vscode-server/bin/linux-<arch>/<commit>/`
    (compose-mount style, as in cpp-school) or `$HOME/.vscode-server/bin/<commit>/`
    (devcontainer.json style).
    """
    r = subprocess.run(
        ["docker", "exec", container_id, "bash", "-lc", _FIND_CODE_SERVER],
        capture_output=True, text=True, timeout=10, check=False,
    )
    if r.returncode != 0:
        return None
    path = r.stdout.strip().splitlines()[0] if r.stdout.strip() else None
    return path or None


def _wait_for_code_cli(container_id: str, timeout: int = 120) -> str | None:
    """VS Code Server deploys `code` post-attach; poll until present or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        path = _find_code_cli(container_id)
        if path:
            return path
        time.sleep(2)
    return None


def _install_vsix(container_id: str, vsix_host_path: str, code_cli: str) -> bool:
    name = os.path.basename(vsix_host_path)
    dest = f"/tmp/cairn-injected/{name}"
    try:
        subprocess.run(
            ["docker", "exec", container_id, "mkdir", "-p", "/tmp/cairn-injected"],
            timeout=10, check=False,
        )
        cp = subprocess.run(
            ["docker", "cp", vsix_host_path, f"{container_id}:{dest}"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        if cp.returncode != 0:
            log.warning("docker cp failed for %s: %s", name, cp.stderr.strip())
            return False
        inst = subprocess.run(
            ["docker", "exec", container_id, code_cli,
             "--install-extension", dest, "--force"],
            capture_output=True, text=True, timeout=60, check=False,
        )
        if inst.returncode != 0:
            log.warning("install-extension failed for %s: %s",
                        name, inst.stderr.strip() or inst.stdout.strip())
            return False
        log.info("Installed %s into container %s", name, container_id[:12])
        return True
    except (subprocess.SubprocessError, OSError) as e:
        log.warning("install error for %s: %s", name, e)
        return False


def _handle_container_start(container_id: str, stage_dir: str) -> None:
    if not _is_dev_container(container_id):
        return
    vsixes = _list_staged_vsixes(stage_dir)
    if not vsixes:
        return
    log.info("Dev container %s started — injecting %d extension(s)",
             container_id[:12], len(vsixes))
    code_cli = _wait_for_code_cli(container_id)
    if not code_cli:
        log.warning("Container %s never exposed `code` CLI; skipping injection",
                    container_id[:12])
        return
    for vsix in vsixes:
        _install_vsix(container_id, vsix, code_cli)


def watch_loop(stage_dir: str) -> None:
    """Subscribe to docker events and dispatch container starts to injector."""
    while True:
        try:
            proc = subprocess.Popen(
                ["docker", "events",
                 "--filter", "type=container",
                 "--filter", "event=start",
                 "--format", "{{.ID}}"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
        except (FileNotFoundError, OSError) as e:
            log.warning("docker events unavailable: %s — injector idle", e)
            time.sleep(60)
            continue

        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                cid = line.strip()
                if not cid:
                    continue
                # Dispatch in a worker thread — wait_for_code_cli can block 2 min.
                t = threading.Thread(
                    target=_handle_container_start,
                    args=(cid, stage_dir),
                    daemon=True,
                )
                t.start()
        except Exception as e:  # noqa: BLE001 — watcher must never crash daemon
            log.warning("docker events watcher error: %s — restarting in 5s", e)
        finally:
            try:
                proc.terminate()
            except OSError:
                pass
        time.sleep(5)


def start_in_background(stage_dir: str) -> threading.Thread:
    t = threading.Thread(target=watch_loop, args=(stage_dir,),
                         name="container-injector", daemon=True)
    t.start()
    return t
