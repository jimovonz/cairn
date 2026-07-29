"""Every write path declares what it assumes about its input (spec 1.10).

The two ingest defects came from transplanting an invariant into a domain that
violated it — sound reasoning for repos, silently false for a corpus with
mutable membership. Slowing down would not have caught that; writing the
assumption down makes the mismatch visible on first contact.

This test is the enforcement: adding a write path without declaring its
invariant fails here, so the practice cannot quietly lapse.
"""
import importlib

import pytest

# Write paths whose errors land in the durable corpus. Extend deliberately —
# a new entry here is a claim that you have thought about the input domain.
WRITE_PATH_MODULES = [
    "cairn.ingest",
    "cairn.analyser",
    "cairn.review_writeback",
    "cairn.relevance",
    "hooks.storage",
]


@pytest.mark.parametrize("modname", WRITE_PATH_MODULES)
def test_write_path_declares_input_domain_invariant(modname):
    mod = importlib.import_module(modname)
    inv = getattr(mod, "INPUT_DOMAIN_INVARIANT", None)
    assert inv, f"{modname} declares no INPUT_DOMAIN_INVARIANT"
    assert isinstance(inv, str)


@pytest.mark.parametrize("modname", WRITE_PATH_MODULES)
def test_invariant_states_an_assumption_not_a_description(modname):
    """A description of what the code does is not an invariant. Requiring the
    word 'assume' keeps these honest rather than restating the docstring."""
    mod = importlib.import_module(modname)
    assert "assume" in mod.INPUT_DOMAIN_INVARIANT.lower()


def test_ingest_invariant_names_the_actual_defect():
    """The reviewer's proposed invariant for ingest ('fixed non-shrinking
    namespace') is true of diff_sections but is NOT the bug. Recording it alone
    would leave the real write-side defect undocumented."""
    import cairn.ingest as ingest
    inv = ingest.INPUT_DOMAIN_INVARIANT.lower()
    assert "source_sections" in inv
    assert "invalidation" in inv
