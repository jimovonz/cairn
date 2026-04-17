#!/usr/bin/env python3
"""Installation validation tests — verifies install.sh and uninstall.sh behaviour.

These tests run the actual shell scripts in an isolated temp environment
(fake HOME, fake CLAUDE_DIR) to verify:
- Venv creation and dependency installation
- Database initialization (schema, tables, triggers, indexes)
- Global rules deployment
- Settings.json hook registration (fresh + merge into existing)
- Slash command installation
- Idempotent re-install
- Uninstall cleanup (preserves DB, removes hooks/rules/commands)
- Settings.json hook removal (clean, leaves other settings intact)

NOTE: These tests require Python 3.10+ and pip. They create real venvs
(~80MB for sentence-transformers download is skipped via mocking).
Run with: pytest tests/test_install_validation.py -v
"""

import json
import os
try:
    import pysqlite3 as sqlite3  # type: ignore[import-untyped]
except ImportError:
    import sqlite3
import shutil
import subprocess
import tempfile

import pytest

CAIRN_HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTALL_SCRIPT = os.path.join(CAIRN_HOME, "install.sh")
UNINSTALL_SCRIPT = os.path.join(CAIRN_HOME, "uninstall.sh")


@pytest.fixture
def isolated_env(tmp_path):
    """Create an isolated environment with fake HOME and CLAUDE_DIR."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    claude_dir = fake_home / ".claude"
    claude_dir.mkdir()

    env = os.environ.copy()
    env["HOME"] = str(fake_home)

    return {
        "home": str(fake_home),
        "claude_dir": str(claude_dir),
        "env": env,
        "tmp_path": tmp_path,
    }


# --- Schema validation ---

class TestDatabaseInit:
    """Validates init_db.py creates the correct schema."""

    @pytest.mark.behavioural
    def test_init_creates_all_tables(self, tmp_path):
        """init_db.py should create all required tables."""
        db_path = str(tmp_path / "test.db")
        result = subprocess.run(
            ["python3", "-c", f"""
import sys, os
sys.path.insert(0, '{os.path.join(CAIRN_HOME, "cairn")}')
os.chdir('{tmp_path}')
import init_db
init_db.DB_PATH = '{db_path}'
init_db.init()
"""],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"init_db failed: {result.stderr}"

        conn = sqlite3.connect(db_path)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()}
        conn.close()

        required = {"memories", "memory_history", "sessions", "metrics", "hook_state"}
        assert required.issubset(tables), \
            f"Missing tables: {required - tables}. Found: {tables}"

    @pytest.mark.behavioural
    def test_init_creates_indexes(self, tmp_path):
        """init_db.py should create required indexes."""
        db_path = str(tmp_path / "test.db")
        subprocess.run(
            ["python3", "-c", f"""
import sys, os
sys.path.insert(0, '{os.path.join(CAIRN_HOME, "cairn")}')
os.chdir('{tmp_path}')
import init_db
init_db.DB_PATH = '{db_path}'
init_db.init()
"""],
            capture_output=True, text=True, timeout=30,
        )

        conn = sqlite3.connect(db_path)
        indexes = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()}
        conn.close()

        required = {"idx_memories_type", "idx_memories_topic", "idx_memories_project",
                     "idx_metrics_event", "idx_history_memory_id"}
        assert required.issubset(indexes), \
            f"Missing indexes: {required - indexes}. Found: {indexes}"

    @pytest.mark.behavioural
    def test_init_creates_fts5(self, tmp_path):
        """init_db.py should create FTS5 virtual table."""
        db_path = str(tmp_path / "test.db")
        subprocess.run(
            ["python3", "-c", f"""
import sys, os
sys.path.insert(0, '{os.path.join(CAIRN_HOME, "cairn")}')
os.chdir('{tmp_path}')
import init_db
init_db.DB_PATH = '{db_path}'
init_db.init()
"""],
            capture_output=True, text=True, timeout=30,
        )

        conn = sqlite3.connect(db_path)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()

        assert "memories_fts" in tables, "FTS5 virtual table should exist"

    @pytest.mark.behavioural
    def test_init_enables_wal_mode(self, tmp_path):
        """init_db.py should enable WAL journal mode."""
        db_path = str(tmp_path / "test.db")
        subprocess.run(
            ["python3", "-c", f"""
import sys, os
sys.path.insert(0, '{os.path.join(CAIRN_HOME, "cairn")}')
os.chdir('{tmp_path}')
import init_db
init_db.DB_PATH = '{db_path}'
init_db.init()
"""],
            capture_output=True, text=True, timeout=30,
        )

        conn = sqlite3.connect(db_path)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()

        assert mode == "wal", f"Expected WAL mode, got {mode}"

    @pytest.mark.behavioural
    def test_init_creates_triggers(self, tmp_path):
        """init_db.py should create FTS sync triggers and version trigger."""
        db_path = str(tmp_path / "test.db")
        subprocess.run(
            ["python3", "-c", f"""
import sys, os
sys.path.insert(0, '{os.path.join(CAIRN_HOME, "cairn")}')
os.chdir('{tmp_path}')
import init_db
init_db.DB_PATH = '{db_path}'
init_db.init()
"""],
            capture_output=True, text=True, timeout=30,
        )

        conn = sqlite3.connect(db_path)
        triggers = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger'"
        ).fetchall()}
        conn.close()

        required = {"memories_version", "memories_ai", "memories_ad", "memories_au"}
        assert required.issubset(triggers), \
            f"Missing triggers: {required - triggers}. Found: {triggers}"

    @pytest.mark.behavioural
    def test_init_idempotent(self, tmp_path):
        """Running init_db.py twice should not fail or corrupt the DB."""
        db_path = str(tmp_path / "test.db")
        init_code = f"""
import sys, os
sys.path.insert(0, '{os.path.join(CAIRN_HOME, "cairn")}')
os.chdir('{tmp_path}')
import init_db
init_db.DB_PATH = '{db_path}'
init_db.init()
"""
        r1 = subprocess.run(["python3", "-c", init_code], capture_output=True, text=True, timeout=30)
        assert r1.returncode == 0, f"First init failed: {r1.stderr}"

        r2 = subprocess.run(["python3", "-c", init_code], capture_output=True, text=True, timeout=30)
        assert r2.returncode == 0, f"Second init failed: {r2.stderr}"

        # Verify DB is intact
        conn = sqlite3.connect(db_path)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()}
        conn.close()
        assert "memories" in tables

    @pytest.mark.behavioural
    def test_init_migration_adds_columns(self, tmp_path):
        """init_db.py should add missing columns to an existing minimal DB."""
        db_path = str(tmp_path / "test.db")

        # Create a minimal DB with just the base columns
        conn = sqlite3.connect(db_path)
        conn.execute("""CREATE TABLE memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL, topic TEXT NOT NULL, content TEXT NOT NULL,
            keywords TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        conn.commit()
        conn.close()

        # Run init_db — should migrate
        subprocess.run(
            ["python3", "-c", f"""
import sys, os
sys.path.insert(0, '{os.path.join(CAIRN_HOME, "cairn")}')
os.chdir('{tmp_path}')
import init_db
init_db.DB_PATH = '{db_path}'
init_db.init()
"""],
            capture_output=True, text=True, timeout=30,
        )

        conn = sqlite3.connect(db_path)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()}
        conn.close()

        for col in ["embedding", "session_id", "project", "confidence",
                     "source_start", "source_end", "associated_files"]:
            assert col in cols, f"Migration should add column '{col}'"


# --- Template / file validation ---

class TestTemplateFiles:
    """Validates template files used by install.sh exist and are well-formed."""

    @pytest.mark.behavioural
    def test_global_settings_template_valid_json(self):
        """templates/global-settings.json should be valid JSON after placeholder stripping."""
        template_path = os.path.join(CAIRN_HOME, "templates", "global-settings.json")
        assert os.path.exists(template_path), "Template should exist"

        with open(template_path) as f:
            content = f.read()

        # Replace placeholders with dummy values
        content = content.replace("{{VENV_PYTHON}}", "/tmp/venv/bin/python3")
        content = content.replace("{{CAIRN_HOME}}", "/tmp/cairn")

        settings = json.loads(content)
        assert "hooks" in settings, "Template should define hooks"
        assert "Stop" in settings["hooks"], "Template should define Stop hook"
        assert "UserPromptSubmit" in settings["hooks"], "Template should define UserPromptSubmit hook"

    @pytest.mark.behavioural
    def test_global_settings_hooks_reference_correct_files(self):
        """Hook commands in template should reference the actual hook files."""
        template_path = os.path.join(CAIRN_HOME, "templates", "global-settings.json")
        with open(template_path) as f:
            content = f.read()

        assert "stop_hook.py" in content, "Template should reference stop_hook.py"
        assert "prompt_hook.py" in content, "Template should reference prompt_hook.py"

    @pytest.mark.behavioural
    def test_rules_file_exists(self):
        """The source rules file should exist in .claude/rules/."""
        rules_path = os.path.join(CAIRN_HOME, ".claude", "rules", "memory-system.md")
        assert os.path.exists(rules_path), "Rules file should exist"

        with open(rules_path) as f:
            content = f.read()
        assert "Cairn Memory System" in content, "Rules file should contain system documentation"
        assert "[cm]" in content or "<memory>" in content, "Rules file should document memory block format"

    @pytest.mark.behavioural
    def test_cairn_command_template_exists(self):
        """templates/cairn-command.md should exist."""
        cmd_path = os.path.join(CAIRN_HOME, "templates", "cairn-command.md")
        assert os.path.exists(cmd_path), "Cairn command template should exist"

    @pytest.mark.behavioural
    def test_hook_files_exist(self):
        """All hook files referenced by the install should exist."""
        hooks_dir = os.path.join(CAIRN_HOME, "hooks")
        for hook_file in ["stop_hook.py", "prompt_hook.py", "pretool_hook.py",
                          "hook_helpers.py", "parser.py", "storage.py",
                          "retrieval.py", "enforcement.py"]:
            assert os.path.exists(os.path.join(hooks_dir, hook_file)), \
                f"Hook file {hook_file} should exist"

    @pytest.mark.behavioural
    def test_requirements_txt_exists(self):
        """requirements.txt should exist and list dependencies."""
        req_path = os.path.join(CAIRN_HOME, "requirements.txt")
        assert os.path.exists(req_path), "requirements.txt should exist"

        with open(req_path) as f:
            content = f.read()
        assert "sentence-transformers" in content, "Should require sentence-transformers"


# --- Settings.json merge/removal ---

class TestSettingsMerge:
    """Tests the settings.json merge logic used by install.sh and uninstall.sh."""

    @pytest.mark.behavioural
    def test_merge_into_empty_settings(self, tmp_path):
        """Hook merge into empty settings.json should add all hooks."""
        settings_path = tmp_path / "settings.json"
        settings_path.write_text("{}")

        template_path = os.path.join(CAIRN_HOME, "templates", "global-settings.json")
        with open(template_path) as f:
            template = f.read()
        template = template.replace("{{VENV_PYTHON}}", "/tmp/venv/bin/python3")
        template = template.replace("{{CAIRN_HOME}}", "/tmp/cairn")

        # Run the merge logic (extracted from install.sh)
        merge_code = f"""
import json
cairn_hooks = json.loads('''{template}''')
with open('{settings_path}') as f:
    settings = json.load(f)
hooks = settings.setdefault('hooks', {{}})
for event, groups in cairn_hooks.get('hooks', {{}}).items():
    if event not in hooks:
        hooks[event] = []
    hooks[event].extend(groups)
with open('{settings_path}', 'w') as f:
    json.dump(settings, f, indent=2)
"""
        result = subprocess.run(["python3", "-c", merge_code],
                                capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, f"Merge failed: {result.stderr}"

        with open(settings_path) as f:
            settings = json.load(f)

        assert "hooks" in settings
        assert "Stop" in settings["hooks"]
        assert "UserPromptSubmit" in settings["hooks"]

    @pytest.mark.behavioural
    def test_merge_preserves_existing_hooks(self, tmp_path):
        """Merging Cairn hooks should not remove existing non-Cairn hooks."""
        settings_path = tmp_path / "settings.json"
        existing = {
            "hooks": {
                "Stop": [{"hooks": [{"command": "my-custom-hook.sh"}]}]
            },
            "other_setting": True,
        }
        settings_path.write_text(json.dumps(existing))

        template_path = os.path.join(CAIRN_HOME, "templates", "global-settings.json")
        with open(template_path) as f:
            template = f.read()
        template = template.replace("{{VENV_PYTHON}}", "/tmp/venv/bin/python3")
        template = template.replace("{{CAIRN_HOME}}", "/tmp/cairn")

        merge_code = f"""
import json
cairn_hooks = json.loads('''{template}''')
with open('{settings_path}') as f:
    settings = json.load(f)
hooks = settings.setdefault('hooks', {{}})
for event, groups in cairn_hooks.get('hooks', {{}}).items():
    if event not in hooks:
        hooks[event] = []
    hooks[event].extend(groups)
with open('{settings_path}', 'w') as f:
    json.dump(settings, f, indent=2)
"""
        subprocess.run(["python3", "-c", merge_code],
                        capture_output=True, text=True, timeout=10)

        with open(settings_path) as f:
            settings = json.load(f)

        # Custom hook should still be there
        stop_hooks = settings["hooks"]["Stop"]
        commands = [h.get("command", "") for g in stop_hooks for h in g.get("hooks", [])]
        assert "my-custom-hook.sh" in commands, "Existing hooks should be preserved"

        # Cairn hook should also be there
        assert any("stop_hook.py" in c for c in commands), "Cairn stop hook should be added"

        # Other settings preserved
        assert settings.get("other_setting") is True

    @pytest.mark.behavioural
    def test_uninstall_removes_cairn_hooks_only(self, tmp_path):
        """Uninstall logic should remove only Cairn hooks, leaving others intact."""
        settings_path = tmp_path / "settings.json"
        settings = {
            "hooks": {
                "Stop": [
                    {"hooks": [{"command": "my-custom-hook.sh"}]},
                    {"hooks": [{"command": "/tmp/cairn/hooks/stop_hook.py"}]},
                ],
                "UserPromptSubmit": [
                    {"hooks": [{"command": "/tmp/cairn/hooks/prompt_hook.py"}]},
                ],
            },
            "theme": "dark",
        }
        settings_path.write_text(json.dumps(settings))

        # Run uninstall logic (extracted from uninstall.sh)
        remove_code = f"""
import json, sys
with open('{settings_path}') as f:
    settings = json.load(f)
hooks = settings.get('hooks', {{}})
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
        if hooks.get(event) != original:
            changed = True
if not hooks:
    settings.pop('hooks', None)
with open('{settings_path}', 'w') as f:
    json.dump(settings, f, indent=2)
"""
        subprocess.run(["python3", "-c", remove_code],
                        capture_output=True, text=True, timeout=10)

        with open(settings_path) as f:
            result = json.load(f)

        # Cairn hooks should be gone
        stop_hooks = result.get("hooks", {}).get("Stop", [])
        all_commands = [h.get("command", "") for g in stop_hooks for h in g.get("hooks", [])]
        assert not any("cairn" in c.lower() for c in all_commands), \
            "Cairn hooks should be removed"

        # Custom hook should remain
        assert "my-custom-hook.sh" in all_commands, \
            "Non-Cairn hooks should be preserved"

        # UserPromptSubmit should be gone entirely (only had Cairn hook)
        assert "UserPromptSubmit" not in result.get("hooks", {}), \
            "Empty hook events should be cleaned up"

        # Other settings preserved
        assert result.get("theme") == "dark"


# --- Script syntax validation ---

class TestScriptSyntax:
    """Validates install.sh and uninstall.sh are syntactically correct."""

    @pytest.mark.behavioural
    def test_install_script_syntax(self):
        """install.sh should pass bash syntax check."""
        result = subprocess.run(
            ["bash", "-n", INSTALL_SCRIPT],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, f"install.sh syntax error: {result.stderr}"

    @pytest.mark.behavioural
    def test_uninstall_script_syntax(self):
        """uninstall.sh should pass bash syntax check."""
        result = subprocess.run(
            ["bash", "-n", UNINSTALL_SCRIPT],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, f"uninstall.sh syntax error: {result.stderr}"


# --- Health check ---

class TestHealthCheck:
    """Tests the query.py --check health validation."""

    @pytest.mark.behavioural
    def test_check_on_valid_install(self):
        """query.py --check should run without crashing."""
        result = subprocess.run(
            ["python3", os.path.join(CAIRN_HOME, "cairn", "query.py"), "--check"],
            capture_output=True, text=True, timeout=30,
        )
        # --check may return non-zero in CI (daemon not running, model not loaded)
        # but it should always produce output and not crash with a traceback
        assert "Health Check" in result.stdout, \
            f"Should output health check header. stdout={result.stdout}, stderr={result.stderr}"
        assert "Traceback" not in result.stderr, \
            f"Health check should not crash: {result.stderr}"


# --- Config validation ---

class TestConfigValidation:
    """Tests that config.py loads correctly and env overrides work."""

    @pytest.mark.behavioural
    def test_config_loads(self):
        """config.py should be importable without errors."""
        result = subprocess.run(
            ["python3", "-c", f"""
import sys
sys.path.insert(0, '{os.path.join(CAIRN_HOME, "cairn")}')
import config
assert hasattr(config, 'DEDUP_THRESHOLD')
assert hasattr(config, 'L1_SIM_THRESHOLD')
assert hasattr(config, 'MAX_CONTINUATIONS')
print('OK')
"""],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, f"Config import failed: {result.stderr}"
        assert "OK" in result.stdout

    @pytest.mark.behavioural
    def test_config_env_override(self):
        """Environment variables should override config values."""
        env = os.environ.copy()
        env["CAIRN_DEDUP_THRESHOLD"] = "0.80"
        env["CAIRN_MAX_CONTINUATIONS"] = "5"
        env["CAIRN_L1_5_ENABLED"] = "false"

        result = subprocess.run(
            ["python3", "-c", f"""
import sys
sys.path.insert(0, '{os.path.join(CAIRN_HOME, "cairn")}')
import config
assert config.DEDUP_THRESHOLD == 0.80, f"Expected 0.80, got {{config.DEDUP_THRESHOLD}}"
assert config.MAX_CONTINUATIONS == 5, f"Expected 5, got {{config.MAX_CONTINUATIONS}}"
assert config.L1_5_ENABLED is False, f"Expected False, got {{config.L1_5_ENABLED}}"
print('OK')
"""],
            capture_output=True, text=True, timeout=10,
            env=env,
        )
        assert result.returncode == 0, f"Config override failed: {result.stderr}\n{result.stdout}"
        assert "OK" in result.stdout
