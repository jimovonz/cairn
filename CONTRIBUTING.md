# Contributing to Cairn

Thanks for your interest in contributing! Cairn is a young project and contributions are welcome.

## Getting started

1. Fork the repo
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/cairn`
3. Run the installer: `./install.sh`
4. Create a branch: `git checkout -b my-feature`
5. Make your changes
6. Test thoroughly — the system modifies Claude Code's behavior globally
7. Commit and push
8. Open a PR

## What to contribute

**High value:**
- Bug fixes (especially in hook parsing or retrieval logic)
- Improved retrieval quality (better scoring, gating, or ranking)
- Test coverage for untested paths
- Documentation improvements
- Platform compatibility (macOS, Windows WSL)

**Welcome but discuss first:**
- New memory types
- New retrieval layers
- Schema changes (these affect all existing users)
- Changes to the confidence model

**Please don't:**
- Add external API dependencies (the system is intentionally local-only)
- Add MCP — this is a deliberate design choice, see ARCHITECTURE.md
- Make changes that require Claude Code source modifications

## Code style

- Python 3.10+
- Type annotations on all function signatures (`from __future__ import annotations`)
- Functions should fail gracefully — hooks must never block the user
- All thresholds and weights belong in `cairn/config.py`

## Project structure

The hook code is split into focused modules:

| Module | Responsibility |
|--------|---------------|
| `hook_helpers.py` | Shared DB access, logging, metrics, embedder |
| `parser.py` | Memory block parsing (`ParseResult` NamedTuple) |
| `storage.py` | Insert, dedup, confidence updates, quality gates |
| `enforcement.py` | Trailing intent detection, continuation counting |
| `retrieval.py` | Context retrieval, Layer 2 cross-project, context cache |
| `stop_hook.py` | Orchestrator — routes through the above |
| `prompt_hook.py` | Layer 1 first-prompt push + Layer 2 injection |

## Testing

185 tests across 11 files. Run with:

```bash
python3 -m pytest tests/         # full suite
python3 -m pytest tests/ -k "backfill"  # run specific tests
```

No embedding model required — tests use deterministic mock vectors and patched DB paths.

**Before submitting a PR:**
1. All existing tests must pass
2. Add tests for new functionality — test behavior through real code paths, not inline arithmetic
3. Run `./install.sh` on a clean setup and verify the hook fires
4. Check `/cairn` shows stats in a Claude Code session

## Reporting issues

Include:
- Claude Code version (`claude --version`)
- Python version (`python3 --version`)
- Contents of `cairn/hook.log` (last ~20 lines)
- Output of `/cairn stats`

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
