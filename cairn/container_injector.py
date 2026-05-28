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
    """True if the container looks like a dev container.

    Two recognition paths:
      1. `devcontainer.metadata` label — set by the dev-container CLI when
         VS Code launches via "Reopen in Container".
      2. `com.docker.compose.project.config_files` contains a `.devcontainer/`
         path — set when the user (or VS Code dev-container CLI) launched via
         docker compose against a compose file inside .devcontainer/. Catches
         manual `docker compose up` usage that VS Code didn't broker.
    """
    try:
        out = subprocess.run(
            ["docker", "inspect", "--format", "{{json .Config.Labels}}", container_id],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if out.returncode != 0:
            return False
        labels = json.loads(out.stdout.strip() or "{}") or {}
        if "devcontainer.metadata" in labels:
            return True
        cfg = labels.get("com.docker.compose.project.config_files", "")
        if "/.devcontainer/" in cfg or cfg.endswith("/.devcontainer"):
            return True
        return False
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


_SHIM_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shims", "cairn-hook.py"
)
_HOOK_TEMPLATE_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "templates", "copilot-hooks-container.json",
)


def _container_user_home(container_id: str) -> str | None:
    """Resolve $HOME for the container's default user."""
    r = subprocess.run(
        ["docker", "exec", container_id, "bash", "-lc", "echo $HOME"],
        capture_output=True, text=True, timeout=10, check=False,
    )
    if r.returncode != 0:
        return None
    home = r.stdout.strip()
    return home or None


def _deploy_hook_files(container_id: str) -> bool:
    """Copy the cairn shim + hook config into the container so it can talk
    to the host daemon via TCP. No pre-existing devcontainer.json mounts
    required.
    """
    if not os.path.isfile(_SHIM_SRC) or not os.path.isfile(_HOOK_TEMPLATE_SRC):
        log.warning("Hook deploy skipped — shim or template missing on host")
        return False
    home = _container_user_home(container_id)
    if not home:
        log.warning("Hook deploy skipped — couldn't resolve container $HOME")
        return False
    try:
        # 1. Shim
        subprocess.run(
            ["docker", "exec", container_id, "mkdir", "-p", "/opt/cairn-shims"],
            timeout=10, check=False,
        )
        cp = subprocess.run(
            ["docker", "cp", _SHIM_SRC, f"{container_id}:/opt/cairn-shims/cairn-hook.py"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        if cp.returncode != 0:
            log.warning("shim cp failed: %s", cp.stderr.strip())
            return False
        subprocess.run(
            ["docker", "exec", container_id, "chmod", "+x", "/opt/cairn-shims/cairn-hook.py"],
            timeout=10, check=False,
        )
        # 2. Hook config — render {{SHIM}} placeholder, write to a host tmp,
        # then docker cp to ~/.github/hooks/cairn.json
        with open(_HOOK_TEMPLATE_SRC) as f:
            tmpl = f.read()
        rendered = tmpl.replace("{{SHIM}}", "/opt/cairn-shims/cairn-hook.py")
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            tf.write(rendered)
            tf_path = tf.name
        try:
            subprocess.run(
                ["docker", "exec", container_id, "mkdir", "-p", f"{home}/.github/hooks"],
                timeout=10, check=False,
            )
            cp2 = subprocess.run(
                ["docker", "cp", tf_path, f"{container_id}:{home}/.github/hooks/cairn.json"],
                capture_output=True, text=True, timeout=30, check=False,
            )
            if cp2.returncode != 0:
                log.warning("hook config cp failed: %s", cp2.stderr.strip())
                return False
        finally:
            try:
                os.unlink(tf_path)
            except OSError:
                pass
        log.info("Deployed cairn shim + hook config into container %s",
                 container_id[:12])
        return True
    except (subprocess.SubprocessError, OSError) as e:
        log.warning("hook deploy error: %s", e)
        return False


def _handle_container_start(container_id: str, stage_dir: str) -> None:
    if not _is_dev_container(container_id):
        return
    # Always wait for code-server — we need it for the extension install path
    # and it's also a good signal that the container is "ready" for our work.
    code_cli = _wait_for_code_cli(container_id)

    # Deploy cairn shim + hook config (no project-side mount required)
    try:
        from cairn.config import CONTAINER_AUTO_DEPLOY_HOOKS
    except ImportError:
        CONTAINER_AUTO_DEPLOY_HOOKS = True
    if CONTAINER_AUTO_DEPLOY_HOOKS:
        _deploy_hook_files(container_id)

    # Install any staged VSIXes
    vsixes = _list_staged_vsixes(stage_dir)
    if vsixes:
        if not code_cli:
            log.warning("Container %s never exposed code-server; skipping VSIX install",
                        container_id[:12])
        else:
            log.info("Dev container %s started — injecting %d extension(s)",
                     container_id[:12], len(vsixes))
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
