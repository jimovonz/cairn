# Contributing to Engram

Thanks for your interest in contributing! Engram is a young project and contributions are welcome.

## Getting started

1. Fork the repo
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/engram`
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
- Test coverage
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
- No type annotations required (keep it simple)
- Functions should fail gracefully — the stop hook must never block the user
- All thresholds and weights belong in `engram/config.py`

## Testing

There's no formal test suite yet (contributions welcome). For now:

1. Run `./install.sh` on a clean setup
2. Start a Claude Code session and verify the hook fires
3. Check `/engram` shows stats
4. Test context retrieval by asking about a topic from a previous session

## Reporting issues

Include:
- Claude Code version (`claude --version`)
- Python version (`python3 --version`)
- Contents of `engram/hook.log` (last ~20 lines)
- Output of `/engram stats`

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
