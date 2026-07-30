"""_looks_like_code_search — which commands read as a symbol lookup.

Two failure modes this pins, both found by replaying the real cch event log
(27k commands) against the detector on 2026-07-30:

  * `rg` was not matched at all. The routing policy in this environment
    mandates rg over grep, so the graph hint was blind to ~a quarter of all
    symbol searches and simply never fired on them. Apparent model reluctance
    to use the graph was partly a detector that never spoke.
  * `| grep foo` fired. That is a filter over another command's output, not a
    search for where a symbol lives, and a graph hint there is noise. 83 of
    the 89 firings the fix removes were this shape.
"""
import hooks.pretool_hook as ph


def test_rg_is_detected():
    assert ph._looks_like_code_search("rg -n handle_request src/") == "handle_request"
    assert ph._looks_like_code_search('rg -n "handle_request" src/') == "handle_request"


def test_grep_still_detected():
    assert ph._looks_like_code_search("grep -rn handle_request src/") == "handle_request"
    assert ph._looks_like_code_search("grep -rn 'handle_request' src/") == "handle_request"


def test_definition_keyword_inside_the_quotes():
    assert ph._looks_like_code_search("rg -n 'def target_function' src/") == "target_function"
    assert ph._looks_like_code_search('grep -n "class RobotModel" src/') == "RobotModel"


def test_pipeline_filter_is_not_a_code_search():
    # Narrowing a result set, not locating a symbol.
    assert ph._looks_like_code_search("crontab -l 2>/dev/null | grep -i cairn") is None
    assert ph._looks_like_code_search("ip addr show eth0 | grep inet") is None
    assert ph._looks_like_code_search("find output/ -name '*.dtb' | grep -i p132") is None


def test_ccm_get_grep_flag_is_not_a_code_search():
    # --grep is ccm-get's retrieval filter; the old \bgrep\b matched inside it.
    assert ph._looks_like_code_search('ccm-get.py b2s:abc --grep "bookworm"') is None


def test_search_in_first_segment_survives_a_trailing_pipe():
    assert ph._looks_like_code_search("rg -n handle_request src/ | head -20") == "handle_request"


def test_flag_only_invocation_yields_nothing():
    assert ph._looks_like_code_search("rg --version") is None
    assert ph._looks_like_code_search("rg -l") is None


def test_non_search_commands_yield_nothing():
    assert ph._looks_like_code_search("git status") is None
    assert ph._looks_like_code_search("cairn-graph --location handle_request") is None
