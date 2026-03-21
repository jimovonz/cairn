#!/bin/bash
set -e

# Cairn Uninstaller
# Removes global hooks, rules, and commands. Does NOT delete the database or repo.

CAIRN_HOME="$(cd "$(dirname "$0")" && pwd)"
VENV_PYTHON="$CAIRN_HOME/.venv/bin/python3"
CLAUDE_DIR="$HOME/.claude"

echo "=== Cairn Uninstaller ==="
echo ""

# --- Stop daemon ---
if [ -f "$VENV_PYTHON" ] && [ -f "$CAIRN_HOME/cairn/daemon.py" ]; then
    "$VENV_PYTHON" "$CAIRN_HOME/cairn/daemon.py" stop 2>/dev/null && echo "Stopped embedding daemon." || true
fi

# --- Remove rules file ---
if [ -f "$CLAUDE_DIR/rules/memory-system.md" ]; then
    rm "$CLAUDE_DIR/rules/memory-system.md"
    echo "Removed global rules file."
fi

# --- Remove slash command ---
if [ -f "$CLAUDE_DIR/commands/cairn.md" ]; then
    rm "$CLAUDE_DIR/commands/cairn.md"
    echo "Removed /cairn slash command."
fi

# --- Remove legacy CLAUDE.md if ours ---
if [ -f "$CLAUDE_DIR/CLAUDE.md" ] && grep -q "Cairn — Global Memory" "$CLAUDE_DIR/CLAUDE.md" 2>/dev/null; then
    rm "$CLAUDE_DIR/CLAUDE.md"
    echo "Removed legacy global CLAUDE.md."
fi

# --- Remove hooks from settings.json ---
if [ -f "$CLAUDE_DIR/settings.json" ] && grep -q "stop_hook.py" "$CLAUDE_DIR/settings.json" 2>/dev/null; then
    python3 -c "
import json, sys

with open('$CLAUDE_DIR/settings.json') as f:
    settings = json.load(f)

hooks = settings.get('hooks', {})
changed = False

for event in ['Stop', 'UserPromptSubmit']:
    if event in hooks:
        original = hooks[event]
        hooks[event] = [
            group for group in original
            if not any('cairn' in h.get('command', '').lower() for h in group.get('hooks', []))
        ]
        if not hooks[event]:
            del hooks[event]
        if hooks[event] != original:
            changed = True

if not hooks:
    settings.pop('hooks', None)

if changed:
    with open('$CLAUDE_DIR/settings.json', 'w') as f:
        json.dump(settings, f, indent=2)
        f.write('\n')

sys.exit(0 if changed else 1)
" 2>/dev/null && echo "Removed hooks from settings.json." || echo "No cairn hooks found in settings.json."
fi

echo ""
echo "=== Cairn uninstalled ==="
echo ""
echo "Your memory database is preserved at: $CAIRN_HOME/cairn/cairn.db"
echo "To delete everything: rm -rf $CAIRN_HOME"
echo ""
echo "Restart Claude Code to deactivate hooks."
echo ""
