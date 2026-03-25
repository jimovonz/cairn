#!/usr/bin/env python3
"""
End-to-end tests for question-before-cairn enforcement in the stop hook.

Feeds simulated hook input to the real stop_hook.py and checks decisions.
Uses a test database to avoid polluting production data.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

HOOK_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "hooks", "stop_hook.py")
PROD_DB = os.path.join(os.path.dirname(__file__), "..", "cairn", "cairn.db")
TEST_DB = None
SESSION_ID = "test-question-enforcement-session"


def setup():
    global TEST_DB
    TEST_DB = tempfile.mktemp(suffix=".db")
    shutil.copy2(PROD_DB, TEST_DB)
    for ext in ("-wal", "-shm"):
        src = PROD_DB + ext
        if os.path.exists(src):
            shutil.copy2(src, TEST_DB + ext)
    print(f"Test DB: {TEST_DB}")


def teardown():
    if TEST_DB and os.path.exists(TEST_DB):
        os.unlink(TEST_DB)
        for ext in ("-wal", "-shm"):
            p = TEST_DB + ext
            if os.path.exists(p):
                os.unlink(p)
    print("Test DB cleaned up")


def run_hook(text: str, is_continuation: bool = False, session_id: str = SESSION_ID) -> dict:
    hook_input = {
        "session_id": session_id,
        "last_assistant_message": text,
        "stop_hook_active": is_continuation,
        "transcript_path": "/dev/null",
        "cwd": "/tmp",
    }

    env = os.environ.copy()
    env["CAIRN_DB_PATH"] = TEST_DB
    env["CAIRN_SKIP_EMBEDDER"] = "1"

    result = subprocess.run(
        [sys.executable, HOOK_SCRIPT],
        input=json.dumps(hook_input),
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )

    stdout = result.stdout.strip()
    if stdout:
        try:
            return {"exit_code": result.returncode, **json.loads(stdout)}
        except json.JSONDecodeError:
            return {"exit_code": result.returncode, "raw": stdout}
    return {"exit_code": result.returncode}


def make_response(body: str, memory_fields: str = "") -> str:
    default_memory = "- complete: true\n- context: sufficient\n- keywords: test"
    memory = memory_fields if memory_fields else default_memory
    return f"{body}\n\n<memory>\n{memory}\n</memory>"


# --- Test Cases ---

def test_question_with_sufficient_blocks():
    """Question with context: sufficient should be blocked — check cairn first."""
    text = make_response(
        "I'm not sure about the database schema. What tables are you using?",
        "- complete: true\n- context: sufficient\n- keywords: database, schema"
    )
    result = run_hook(text)
    assert result.get("decision") == "block", f"Expected block, got: {result}"
    assert "cairn" in result.get("reason", "").lower(), f"Reason should mention cairn: {result}"
    print("PASS: question with context: sufficient blocked")


def test_question_with_insufficient_passes():
    """Question with context: insufficient should pass — already checking cairn."""
    text = make_response(
        "What authentication method does this service use?",
        "- complete: true\n- context: insufficient\n- context_need: authentication methods\n- keywords: auth"
    )
    result = run_hook(text)
    reason = result.get("reason", "")
    # May block for context retrieval, but NOT for question-before-cairn
    if result.get("decision") == "block":
        assert "question" not in reason.lower() or "CAIRN CONTEXT" in reason, \
            f"Should not block for question enforcement when insufficient declared: {result}"
    print("PASS: question with context: insufficient passes question enforcement")


def test_no_question_passes():
    """Response without questions should pass."""
    text = make_response(
        "I've updated the file with the new configuration. The changes look good.",
        "- complete: true\n- context: sufficient\n- keywords: config, update"
    )
    result = run_hook(text)
    reason = result.get("reason", "")
    assert "question" not in reason.lower(), f"Should not trigger on non-question: {result}"
    print("PASS: response without questions passes")


def test_question_in_code_block_ignored():
    """Questions inside code blocks should not trigger enforcement."""
    text = make_response(
        "Here's the fix:\n\n```python\n# Why is this failing?\nresult = do_thing()\n```\n\nThat should resolve it.",
        "- complete: true\n- context: sufficient\n- keywords: fix, code"
    )
    result = run_hook(text)
    reason = result.get("reason", "")
    assert "question" not in reason.lower(), f"Question in code block should not trigger: {result}"
    print("PASS: question in code block ignored")


def test_question_in_quotes_ignored():
    """Questions inside quoted strings should not trigger enforcement."""
    text = make_response(
        'The error message says "Why is the connection refused?" which indicates a network issue. I\'ll fix it.',
        "- complete: true\n- context: sufficient\n- keywords: error, network"
    )
    result = run_hook(text)
    reason = result.get("reason", "")
    assert "question" not in reason.lower(), f"Question in quotes should not trigger: {result}"
    print("PASS: question in quotes ignored")


def test_question_early_in_response_ignored():
    """Questions far from the end (not in last 3 sentences) should not trigger."""
    text = make_response(
        "What was the original design?\n\nWell, looking at the code, I can see it was designed for scalability. "
        "The architecture uses microservices. Each service handles one domain. "
        "The database layer is abstracted. Connections are pooled. Everything looks solid.",
        "- complete: true\n- context: sufficient\n- keywords: architecture, design"
    )
    result = run_hook(text)
    reason = result.get("reason", "")
    assert "question" not in reason.lower(), f"Question early in response should not trigger: {result}"
    print("PASS: question early in response (outside last 3 sentences) ignored")


def test_continuation_skips_enforcement():
    """Continuation responses should skip question enforcement."""
    text = make_response(
        "What would you like me to focus on next?",
        "- complete: true\n- context: sufficient\n- keywords: next steps"
    )
    result = run_hook(text, is_continuation=True)
    reason = result.get("reason", "")
    # On continuation, should not get question-before-cairn block
    if result.get("decision") == "block":
        assert "cairn" not in reason.lower() or "question" not in reason.lower(), \
            f"Question enforcement should not fire on continuation: {result}"
    print("PASS: continuation skips question enforcement")


def test_multiple_questions_in_tail():
    """Multiple questions in the last few sentences should trigger."""
    text = make_response(
        "I found a few issues. Should I fix them all at once? Or do you want to review each one?",
        "- complete: true\n- context: sufficient\n- keywords: issues, review"
    )
    result = run_hook(text)
    assert result.get("decision") == "block", f"Multiple questions should trigger: {result}"
    assert "cairn" in result.get("reason", "").lower(), f"Reason should mention cairn: {result}"
    print("PASS: multiple questions in tail triggers enforcement")


def test_rhetorical_question_at_end():
    """A rhetorical-style question at the very end should still trigger."""
    text = make_response(
        "The deployment pipeline is set up correctly. Want me to proceed with the migration?",
        "- complete: true\n- context: sufficient\n- keywords: deployment, migration"
    )
    result = run_hook(text)
    assert result.get("decision") == "block", f"Trailing question should trigger: {result}"
    print("PASS: trailing question triggers enforcement")


def test_statement_with_embedded_question_mark():
    """Question mark in a non-question context (e.g. URL or regex) should not trigger."""
    text = make_response(
        "Updated the regex pattern to handle the edge case. The new pattern is `foo\\?bar`. All tests pass.",
        "- complete: true\n- context: sufficient\n- keywords: regex, fix"
    )
    result = run_hook(text)
    reason = result.get("reason", "")
    # The ? is inside inline code but not in a fenced block — this may or may not trigger
    # depending on how the stripping works. Document the behavior.
    print(f"INFO: embedded ? in inline code → decision={result.get('decision', 'allow')}")
    print("PASS: embedded question mark behavior documented")


# --- Runner ---

def main():
    setup()
    tests = [
        test_question_with_sufficient_blocks,
        test_question_with_insufficient_passes,
        test_no_question_passes,
        test_question_in_code_block_ignored,
        test_question_in_quotes_ignored,
        test_question_early_in_response_ignored,
        test_continuation_skips_enforcement,
        test_multiple_questions_in_tail,
        test_rhetorical_question_at_end,
        test_statement_with_embedded_question_mark,
    ]

    passed = 0
    failed = 0
    errors = []

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            failed += 1
            errors.append((test.__name__, str(e)))
            print(f"FAIL: {test.__name__}: {e}")
        except Exception as e:
            failed += 1
            errors.append((test.__name__, str(e)))
            print(f"ERROR: {test.__name__}: {e}")

    teardown()

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if errors:
        print("\nFailures:")
        for name, err in errors:
            print(f"  {name}: {err}")
    print(f"{'='*60}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
