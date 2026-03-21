#!/bin/bash
set -e

# Engram Installer
# Sets up persistent AI memory for Claude Code

ENGRAM_HOME="$(cd "$(dirname "$0")" && pwd)"
VENV_PATH="$ENGRAM_HOME/.venv"
VENV_PYTHON="$VENV_PATH/bin/python3"
CLAUDE_DIR="$HOME/.claude"

echo "=== Engram Installer ==="
echo "Install location: $ENGRAM_HOME"
echo ""

# --- Python venv ---
if [ ! -f "$VENV_PYTHON" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "$VENV_PATH"
    if [ ! -f "$VENV_PATH/bin/pip" ]; then
        echo "Installing pip..."
        "$VENV_PYTHON" -m ensurepip 2>/dev/null || {
            echo "ERROR: python3-venv not installed. Run: sudo apt install python3-venv"
            exit 1
        }
    fi
else
    echo "Virtual environment exists."
fi

# --- Dependencies ---
echo "Installing dependencies..."
"$VENV_PATH/bin/pip" install -q -r "$ENGRAM_HOME/requirements.txt" 2>&1 | tail -1

# --- Database ---
echo "Initializing database..."
"$VENV_PYTHON" "$ENGRAM_HOME/engram/init_db.py"

# --- Global CLAUDE.md ---
mkdir -p "$CLAUDE_DIR" "$CLAUDE_DIR/rules" "$CLAUDE_DIR/commands"

if [ -f "$CLAUDE_DIR/CLAUDE.md" ]; then
    if grep -q "Engram" "$CLAUDE_DIR/CLAUDE.md" 2>/dev/null; then
        echo "Global CLAUDE.md already configured."
    else
        echo ""
        echo "WARNING: ~/.claude/CLAUDE.md already exists and does not contain Engram config."
        echo "You may need to merge manually. Engram template saved to:"
        echo "  $ENGRAM_HOME/templates/global-claude.md"
        echo ""
    fi
else
    sed "s|{{ENGRAM_HOME}}|$ENGRAM_HOME|g" "$ENGRAM_HOME/templates/global-claude.md" > "$CLAUDE_DIR/CLAUDE.md"
    echo "Installed global CLAUDE.md"
fi

# --- Global rules ---
sed "s|{{ENGRAM_HOME}}|$ENGRAM_HOME|g" "$ENGRAM_HOME/.claude/rules/memory-system.md" | \
    sed "s|\\\$ENGRAM_HOME|$ENGRAM_HOME|g" > "$CLAUDE_DIR/rules/memory-system.md"
echo "Installed global rules."

# --- Global settings (hooks) ---
if [ -f "$CLAUDE_DIR/settings.json" ]; then
    if grep -q "stop_hook.py" "$CLAUDE_DIR/settings.json" 2>/dev/null; then
        echo "Global hooks already configured."
    else
        echo ""
        echo "WARNING: ~/.claude/settings.json exists but does not contain Engram hooks."
        echo "You need to merge the hooks manually from:"
        echo "  $ENGRAM_HOME/templates/global-settings.json"
        echo ""
        echo "The hooks to add are:"
        sed "s|{{VENV_PYTHON}}|$VENV_PYTHON|g; s|{{ENGRAM_HOME}}|$ENGRAM_HOME|g" \
            "$ENGRAM_HOME/templates/global-settings.json" | python3 -m json.tool
        echo ""
    fi
else
    sed "s|{{VENV_PYTHON}}|$VENV_PYTHON|g; s|{{ENGRAM_HOME}}|$ENGRAM_HOME|g" \
        "$ENGRAM_HOME/templates/global-settings.json" > "$CLAUDE_DIR/settings.json"
    echo "Installed global hooks."
fi

# --- Slash command ---
sed "s|{{VENV_PYTHON}}|$VENV_PYTHON|g; s|{{ENGRAM_HOME}}|$ENGRAM_HOME|g" \
    "$ENGRAM_HOME/templates/engram-command.md" > "$CLAUDE_DIR/commands/engram.md"
echo "Installed /engram slash command."

# --- Pre-download model ---
echo ""
echo "Pre-downloading embedding model (one-time)..."
"$VENV_PYTHON" -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('all-MiniLM-L6-v2')
print('Model ready.')
" 2>&1 | grep -v "^$\|Warning:\|Loading"

# --- Start daemon ---
echo "Starting embedding daemon..."
"$VENV_PYTHON" "$ENGRAM_HOME/engram/daemon.py" start

echo ""
echo "=== Engram installed successfully ==="
echo ""
echo "Restart Claude Code to activate hooks."
echo ""
echo "Commands:"
echo "  /engram          — memory stats"
echo "  /engram recent   — recent memories"
echo "  /engram search X — search memories"
echo "  /engram review   — low-confidence memories"
echo "  /engram projects — list projects"
echo ""
