"""Tool-use directive: staged on unassisted navigation, injected as bare text.

Why bare rather than <system-reminder>-wrapped: a one-line imperative in the
user's own turn is complied with, while the same guidance as a rules file or a
passive menu is skimmed. The wrapped channel stays for retrieval payloads,
which are too long to read as anything a person would type.
"""
import json

import hooks.pretool_hook as ph
from cairn.proxy import sidecar
from cairn.proxy.request_inject import inject_prompt_directive


# --- what counts as navigating without the graph ---------------------------

def test_symbol_search_is_unassisted_navigation():
    assert ph._navigates_code_unassisted('rg -n handle_request src/') is True
    assert ph._navigates_code_unassisted("grep -rn 'def handle_request' src/") is True


def test_code_file_read_is_unassisted_navigation():
    assert ph._navigates_code_unassisted("sed -n '10,40p' src/server.py") is True
    assert ph._navigates_code_unassisted('cat hooks/lib/guards.py') is True


def test_a_graph_call_is_not_unassisted():
    assert ph._navigates_code_unassisted('cairn-graph --location handle_request') is False
    assert ph._navigates_code_unassisted('cairn-graph --callers foo | head') is False


def test_non_navigation_is_not_counted():
    assert ph._navigates_code_unassisted('git status') is False
    assert ph._navigates_code_unassisted('cat README.md') is False
    # Trailing filter narrows output, it does not locate anything.
    assert ph._navigates_code_unassisted('ip addr show eth0 | grep inet') is False


# --- injection shape -------------------------------------------------------

def test_directive_appends_bare_text_to_the_last_user_turn():
    data = {'messages': [{'role': 'user', 'content': 'fix the parser'}]}
    inject_prompt_directive(data, 'Also use the code graph for this.')
    content = data['messages'][0]['content']
    assert content.startswith('fix the parser')
    assert 'Also use the code graph for this.' in content
    # The whole point: no wrapper, no sentinel — nothing marking it as injected.
    assert 'system-reminder' not in content
    assert '<!--' not in content


def test_directive_appends_to_a_block_list_content():
    data = {'messages': [{'role': 'user',
                          'content': [{'type': 'text', 'text': 'fix the parser'}]}]}
    inject_prompt_directive(data, 'Use the graph.')
    assert data['messages'][0]['content'][-1]['text'] == 'Use the graph.'


def test_directive_is_not_appended_twice():
    data = {'messages': [{'role': 'user', 'content': 'fix it'}]}
    inject_prompt_directive(data, 'Use the graph.')
    inject_prompt_directive(data, 'Use the graph.')
    assert data['messages'][0]['content'].count('Use the graph.') == 1


def test_over_long_directive_is_capped():
    data = {'messages': [{'role': 'user', 'content': 'hi'}]}
    inject_prompt_directive(data, 'word ' * 200)
    # Past the cap it reads as a payload, not an aside — which is the failure
    # the wrapped channel exists to avoid.
    assert len(data['messages'][0]['content']) < 300


def test_empty_directive_is_a_noop():
    data = {'messages': [{'role': 'user', 'content': 'hi'}]}
    inject_prompt_directive(data, '')
    assert data['messages'][0]['content'] == 'hi'


# --- queue and attribution log --------------------------------------------

def test_queue_round_trips_and_clears(tmp_path, monkeypatch):
    monkeypatch.setattr(sidecar, '_path',
                        lambda sid, suffix: str(tmp_path / f'{sid}{suffix}'))
    sidecar.append_prompt_directive('s1', 'Use the graph.')
    assert sidecar.consume_prompt_directive('s1') == 'Use the graph.'
    assert sidecar.consume_prompt_directive('s1') == ''


def test_queue_ignores_an_unconsumed_duplicate(tmp_path, monkeypatch):
    monkeypatch.setattr(sidecar, '_path',
                        lambda sid, suffix: str(tmp_path / f'{sid}{suffix}'))
    sidecar.append_prompt_directive('s1', 'Use the graph.')
    sidecar.append_prompt_directive('s1', 'Use the graph.')
    assert sidecar.consume_prompt_directive('s1').count('Use the graph.') == 1


def test_every_emitted_directive_is_logged_for_attribution(tmp_path, monkeypatch):
    """The log is the only place the injected/typed distinction survives.

    Indistinguishability is the point on the wire, but the memory writer reads
    the same transcript — without this an injected instruction can be stored as
    a user preference the user never expressed.
    """
    monkeypatch.setattr(sidecar, '_path',
                        lambda sid, suffix: str(tmp_path / f'{sid}{suffix}'))
    sidecar.log_directive('s1', 'Use the graph.')
    rows = [json.loads(l) for l in
            open(sidecar.directive_log_path('s1')).read().splitlines() if l.strip()]
    assert rows and rows[0]['text'] == 'Use the graph.'
    assert rows[0]['session'] == 's1'


# --- the trigger -----------------------------------------------------------

def _stub_trigger(monkeypatch, tmp_path, **cfg):
    """Wire _maybe_stage_graph_directive onto in-memory state and a temp sidecar."""
    from cairn import config
    state = {}
    monkeypatch.setattr(ph, 'load_hook_state', lambda sid, k: state.get((sid, k)))
    monkeypatch.setattr(ph, 'save_hook_state',
                        lambda sid, k, v: state.__setitem__((sid, k), v))
    monkeypatch.setattr(ph, 'record_metric', lambda *a, **k: None)
    monkeypatch.setattr(sidecar, '_path',
                        lambda sid, suffix: str(tmp_path / f'{sid}{suffix}'))
    monkeypatch.setattr(config, 'PROXY_ENABLED', True, raising=False)
    monkeypatch.setattr(config, 'GRAPH_DIRECTIVE_ENABLED', True, raising=False)
    monkeypatch.setattr(config, 'GRAPH_DIRECTIVE_MIN_NAV', cfg.get('min_nav', 3), raising=False)
    monkeypatch.setattr(config, 'GRAPH_DIRECTIVE_MAX_PER_SESSION',
                        cfg.get('max_per_session', 2), raising=False)
    return state


def test_fires_after_the_threshold_of_unassisted_navigation(tmp_path, monkeypatch):
    _stub_trigger(monkeypatch, tmp_path)
    for _ in range(2):
        ph._maybe_stage_graph_directive('s1', 'rg -n handle_request src/', True)
    assert sidecar.consume_prompt_directive('s1') == ''      # below threshold
    ph._maybe_stage_graph_directive('s1', 'cat src/server.py', True)
    assert 'cairn-graph' in sidecar.consume_prompt_directive('s1')


def test_a_graph_call_resets_the_count(tmp_path, monkeypatch):
    _stub_trigger(monkeypatch, tmp_path)
    for _ in range(2):
        ph._maybe_stage_graph_directive('s1', 'rg -n handle_request src/', True)
    ph._maybe_stage_graph_directive('s1', 'cairn-graph --location handle_request', True)
    ph._maybe_stage_graph_directive('s1', 'rg -n other_symbol src/', True)
    # The behaviour we wanted happened, so the count starts over.
    assert sidecar.consume_prompt_directive('s1') == ''


def test_absent_graph_never_fires(tmp_path, monkeypatch):
    """Without a graph the directive points at a tool that cannot answer."""
    _stub_trigger(monkeypatch, tmp_path)
    for _ in range(5):
        ph._maybe_stage_graph_directive('s1', 'rg -n handle_request src/', False)
    assert sidecar.consume_prompt_directive('s1') == ''


def test_habituation_cap_holds(tmp_path, monkeypatch):
    _stub_trigger(monkeypatch, tmp_path, min_nav=1, max_per_session=2)
    for i in range(6):
        ph._maybe_stage_graph_directive('s1', f'rg -n sym_{i} src/', True)
        sidecar.consume_prompt_directive('s1')
    # Repetition is what dulls a directive, so firing is capped per session.
    ph._maybe_stage_graph_directive('s1', 'rg -n more src/', True)
    assert sidecar.consume_prompt_directive('s1') == ''


def test_disabled_flag_suppresses_everything(tmp_path, monkeypatch):
    from cairn import config
    _stub_trigger(monkeypatch, tmp_path, min_nav=1)
    monkeypatch.setattr(config, 'GRAPH_DIRECTIVE_ENABLED', False, raising=False)
    ph._maybe_stage_graph_directive('s1', 'rg -n handle_request src/', True)
    assert sidecar.consume_prompt_directive('s1') == ''
