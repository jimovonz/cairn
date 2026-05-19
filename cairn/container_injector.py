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
_CAIRN_INSTRUCTIONS_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "templates", "cairn-container-instructions.md",
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


_TRUST_MARKER_PREFIX = "<!-- cairn-injected: "
_MANAGED_START = "<!-- cairn-managed:start -->"
_MANAGED_END = "<!-- cairn-managed:end -->"


def _container_workspace_folder(container_id: str) -> str | None:
    """Resolve the workspace root inside the container.

    Order: docker inspect WorkingDir (set by devcontainer.json workspaceFolder
    or compose working_dir), then fall back to first subdirectory of
    /workspaces (default devcontainer.json layout).
    """
    try:
        r = subprocess.run(
            ["docker", "inspect", "--format", "{{.Config.WorkingDir}}", container_id],
            capture_output=True, text=True, timeout=5, check=False,
        )
        wd = r.stdout.strip()
        if wd and wd not in ("/", ""):
            return wd
    except (subprocess.SubprocessError, OSError):
        pass
    r = subprocess.run(
        ["docker", "exec", container_id, "bash", "-lc",
         "ls -d /workspaces/*/ 2>/dev/null | head -1"],
        capture_output=True, text=True, timeout=5, check=False,
    )
    if r.returncode == 0 and r.stdout.strip():
        return r.stdout.strip().rstrip("/")
    return None


def _build_managed_block(entries: list[tuple[str, str]]) -> str:
    """Wrap each (id, body) tuple as a fenced sub-section, all enclosed in
    the outer cairn-managed start/end markers so the block is precisely
    identifiable and idempotently re-writable.
    """
    parts = [_MANAGED_START,
             "<!-- This block is managed by cairn — do NOT edit by hand."
             " Edit the source instruction files in $CAIRN_HOME/templates/"
             " or the extension's bundled instructions.md and re-deploy. -->",
             ""]
    for ident, body in entries:
        parts.append(f"<!-- cairn-injected: {ident} -->")
        parts.append(body.strip())
        parts.append("")
    parts.append(_MANAGED_END)
    return "\n".join(parts) + "\n"


def _ensure_managed_instructions_in_workspace(container_id: str, home: str) -> None:
    """Write or refresh the cairn-managed block in
    `<workspace>/.github/copilot-instructions.md` inside the container.

    This is the only injection point empirically confirmed to deliver trust
    statements to Copilot Chat at system-prompt level (verified 2026-05-19 on
    cpp-school + Copilot Chat 0.48.1: sentinel string in this file is quoted
    back when the model is asked about its system instructions; the hook
    directive in additionalContext is then complied with). codeGeneration.
    instructions and contributes.chatInstructions both verified inert.

    Idempotent: replaces any existing `<!-- cairn-managed:start --> ... -->`
    block in the file; preserves anything else around it.
    """
    workspace = _container_workspace_folder(container_id)
    if not workspace:
        log.warning("Could not resolve workspace root in container %s; skipping",
                    container_id[:12])
        return
    entries = _extension_instruction_entries(container_id, home)
    cairn = _cairn_instruction_entry()
    if cairn:
        entries.insert(0, cairn)
    if not entries:
        return

    target = f"{workspace}/.github/copilot-instructions.md"
    existing = _read_container_file(container_id, target) or ""

    # Strip any prior managed block (greedy single-line regex across newlines)
    import re
    stripped = re.sub(
        re.escape(_MANAGED_START) + r".*?" + re.escape(_MANAGED_END) + r"\n?",
        "", existing, flags=re.DOTALL,
    )

    new_block = _build_managed_block(entries)
    if stripped and not stripped.endswith("\n"):
        stripped += "\n"
    # Add a blank line separator if there's existing content
    sep = "\n" if stripped else ""
    new_content = stripped + sep + new_block

    if new_content == existing:
        return  # nothing changed

    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as tf:
        tf.write(new_content)
        tf_path = tf.name
    try:
        subprocess.run(
            ["docker", "exec", container_id, "mkdir", "-p", f"{workspace}/.github"],
            timeout=10, check=False,
        )
        cp = subprocess.run(
            ["docker", "cp", tf_path, f"{container_id}:{target}"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        if cp.returncode == 0:
            log.info("Wrote cairn-managed instructions to %s (%d entries)",
                     target, len(entries))
            _ensure_gitignore_entry(container_id, workspace)
        else:
            log.warning("Failed to write %s: %s", target, cp.stderr.strip())
    finally:
        try:
            os.unlink(tf_path)
        except OSError:
            pass


def _ensure_gitignore_entry(container_id: str, workspace: str) -> None:
    """Add `.github/copilot-instructions.md` to the workspace .gitignore so the
    cairn-managed file doesn't leak into the user's git history. Idempotent.
    """
    gi_path = f"{workspace}/.gitignore"
    existing = _read_container_file(container_id, gi_path) or ""
    line = ".github/copilot-instructions.md"
    if line in existing.splitlines():
        return
    new = existing
    if new and not new.endswith("\n"):
        new += "\n"
    new += f"# Managed by cairn — local-only Copilot Chat trust statement\n{line}\n"
    import tempfile
    with tempfile.NamedTemporaryFile("w", delete=False) as tf:
        tf.write(new)
        tf_path = tf.name
    try:
        cp = subprocess.run(
            ["docker", "cp", tf_path, f"{container_id}:{gi_path}"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if cp.returncode == 0:
            log.info("Added %s to %s", line, gi_path)
    finally:
        try:
            os.unlink(tf_path)
        except OSError:
            pass


def _read_container_file(container_id: str, path: str) -> str | None:
    """Read a file from inside the container via docker exec cat."""
    r = subprocess.run(
        ["docker", "exec", container_id, "cat", path],
        capture_output=True, text=True, timeout=30, check=False,
    )
    if r.returncode != 0:
        return None
    return r.stdout


def _extension_instruction_entries(container_id: str, home: str) -> list[tuple[str, str]]:
    """For every installed extension declaring contributes.chatInstructions,
    return [(extension_id, instruction_body)] tuples by reading each
    referenced .instructions.md file from inside the container.

    VS Code currently does NOT honour contributes.chatInstructions for trust
    against injection-shaped directives (verified empirically in
    cpp-school 2026-05-19). This function bridges the gap: extensions still
    ship their trust statement via the documented manifest field; daemon
    relays its content into codeGeneration.instructions where Copilot Chat
    actually reads it as system-prompt-level guidance.
    """
    ext_root = f"{home}/.vscode-server/extensions"
    listing = subprocess.run(
        ["docker", "exec", container_id, "ls", ext_root],
        capture_output=True, text=True, timeout=10, check=False,
    )
    if listing.returncode != 0:
        return []
    # Dedupe by extension id (publisher.name) — multiple version dirs of the
    # same extension can coexist (uninstall doesn't always clean prior dirs).
    # Keep latest mtime version so we use the most recent instructions.
    by_id: dict[str, tuple[float, str]] = {}  # ext_id -> (mtime, content)
    for ext_dir in listing.stdout.split():
        ext_path = f"{ext_root}/{ext_dir}"
        pkg = _read_container_file(container_id, f"{ext_path}/package.json")
        if not pkg:
            continue
        try:
            data = json.loads(pkg)
        except json.JSONDecodeError:
            continue
        ci = data.get("contributes", {}).get("chatInstructions") or []
        if not ci:
            continue
        ext_id = f"{data.get('publisher', '?')}.{data.get('name', ext_dir)}"
        # Use ext-dir mtime as version-tiebreaker proxy
        mt = subprocess.run(
            ["docker", "exec", container_id, "stat", "-c", "%Y", ext_path],
            capture_output=True, text=True, timeout=5, check=False,
        )
        try:
            mtime = float(mt.stdout.strip())
        except ValueError:
            mtime = 0.0
        bodies: list[str] = []
        for item in ci:
            rel = item.get("path", "").lstrip("./")
            if not rel:
                continue
            content = _read_container_file(container_id, f"{ext_path}/{rel}")
            if content:
                bodies.append(content)
        if not bodies:
            continue
        joined = "\n\n".join(bodies)
        if ext_id not in by_id or mtime > by_id[ext_id][0]:
            by_id[ext_id] = (mtime, joined)
    return [(eid, body) for eid, (_, body) in sorted(by_id.items())]


def _cairn_instruction_entry() -> tuple[str, str] | None:
    """Read cairn's own container-side instructions to merge alongside extension
    instructions. Returns (id, body) or None if the template is missing.
    """
    if not os.path.isfile(_CAIRN_INSTRUCTIONS_SRC):
        return None
    try:
        with open(_CAIRN_INSTRUCTIONS_SRC) as f:
            return ("cairn", f.read())
    except OSError as e:
        log.warning("Failed to read cairn instructions template: %s", e)
        return None


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

    # Deploy trust statements into <workspace>/.github/copilot-instructions.md —
    # empirically the only Copilot Chat anchor that actually delivers system-
    # prompt-level trust in VS Code 1.118 + Copilot Chat 0.48.1.
    home = _container_user_home(container_id)
    if home:
        _ensure_managed_instructions_in_workspace(container_id, home)


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
