---
name: cairn-devcontainer
description: Dev-container support: the daemon TCP listener and cairn/container_injector.py so containerised sessions reach the host daemon.
---

## Dev-container support

The daemon exposes a **TCP listener on port 47390** alongside its Unix socket so container shims can dial the host daemon via `cairn_recall` / `cairn_remember` opcodes. `cairn/container_injector.py` injects context inside the container, with an extension auto-installer and VSIX staging, so a containerised session reaches the same host cairn as the native session.
