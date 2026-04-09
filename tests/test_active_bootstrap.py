#!/usr/bin/env python3
"""Tests for active bootstrap trigger — knowledge question detection."""

from hooks.prompt_hook import _is_knowledge_question


# === Positive cases — should trigger ===


# Verifies: "what aspect of my job" pattern (the original failure case)
def test_knowledge_question_aspect_of_my_job():
    """The brother/job question that started this whole investigation."""
    assert _is_knowledge_question("what aspect of my job would my brother help with?") is True


# Verifies: "what did we decide" past-context probe
def test_knowledge_question_what_did_we_decide():
    """'what did we decide about X' — classic recall probe."""
    assert _is_knowledge_question("what did we decide about the auth refactor?") is True


# Verifies: "do you remember" recall probe
def test_knowledge_question_do_you_remember():
    """Explicit memory probe."""
    assert _is_knowledge_question("do you remember what I told you about postgres?") is True


# Verifies: "what's my favourite" possessive identity probe
def test_knowledge_question_whats_my():
    """'what's my X' — personal recall."""
    assert _is_knowledge_question("what's my preferred deployment process?") is True


# Verifies: "remind me about" explicit recall request
def test_knowledge_question_remind_me():
    """'remind me about X' — direct recall request."""
    assert _is_knowledge_question("remind me about the cairn architecture") is True


# Verifies: "remember when" episodic probe
def test_knowledge_question_remember_when():
    """'remember when' — episodic memory probe."""
    assert _is_knowledge_question("remember when we tried that migration approach?") is True


# Verifies: "have we ever" historical probe
def test_knowledge_question_have_we_ever():
    """'have we ever discussed X'"""
    assert _is_knowledge_question("have we ever discussed retry semantics?") is True


# Verifies: "who is X" person identity
def test_knowledge_question_who_is():
    """'who is Sarah'"""
    assert _is_knowledge_question("who is Sarah from the ops team?") is True


# Verifies: "tell me about my" biographical probe
def test_knowledge_question_tell_me_about_my():
    """'tell me about my X'"""
    assert _is_knowledge_question("tell me about my deployment setup") is True


# Verifies: case-insensitive matching
def test_knowledge_question_case_insensitive():
    """Pattern matching ignores case."""
    assert _is_knowledge_question("WHAT DID WE DECIDE about it?") is True


# === Negative cases — should NOT trigger ===


# Verifies: imperative requests don't trigger (not asking about prior context)
def test_knowledge_question_imperative_no_trigger():
    """Imperative commands aren't knowledge questions."""
    assert _is_knowledge_question("write a function that parses CSV") is False


# Verifies: greeting doesn't trigger
def test_knowledge_question_greeting_no_trigger():
    """Generic greeting is not a knowledge question."""
    assert _is_knowledge_question("hello") is False


# Verifies: code-only message doesn't trigger
def test_knowledge_question_code_no_trigger():
    """A pasted code snippet is not a knowledge question."""
    assert _is_knowledge_question("def foo():\n    return 42") is False


# Verifies: "what is X" technical question doesn't trigger if X isn't possessive
def test_knowledge_question_generic_what_is_no_trigger():
    """'what is recursion' is a technical question, not personal recall."""
    assert _is_knowledge_question("what is recursion in programming?") is False


# Verifies: forward-looking question doesn't trigger
def test_knowledge_question_forward_looking_no_trigger():
    """'what should we do' is forward-planning, not recall."""
    assert _is_knowledge_question("what should we do next?") is False


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
            print(f"  PASS: {test.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL: {test.__name__}: {e}")
    print(f"\n{passed}/{passed+failed} passed")
