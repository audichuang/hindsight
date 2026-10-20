"""The hindsight-worker entry point validates the cost env at startup (fork: cost-accounting).

The worker never calls ``register_llm_cost_routes``, so without this hook a typo in an
``HINDSIGHT_API_LLM_COST_*`` variable would be ignored in the worker while the API rejected it. The
worker writes LLM trace rows too, so every process that makes LLM calls must agree on the settings.
"""

from pathlib import Path

import pytest

from hindsight_api.config import clear_config_cache
from hindsight_api.engine.llm_cost import load_cost_settings
from tests.fork_helpers import cost_env_cleared

WORKER_SRC = (Path(__file__).resolve().parent.parent / "hindsight_api" / "worker" / "main.py").read_text(
    encoding="utf-8"
)
HOOK_IMPORT = "    from ..engine.llm_cost import load_cost_settings  # fork: cost-accounting"
HOOK_CALL = "    load_cost_settings()  # fork: cost-accounting"


@pytest.fixture(autouse=True)
def _clean_cost_env(monkeypatch):
    yield from cost_env_cleared(monkeypatch)


def test_worker_main_keeps_the_startup_validation_hook():
    assert WORKER_SRC.count(HOOK_IMPORT) == 1
    assert WORKER_SRC.count(HOOK_CALL) == 1


def test_a_typo_in_a_cost_variable_is_rejected_at_startup(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_LLM_COST_ENABELD", "true")
    clear_config_cache()
    load_cost_settings.cache_clear()
    with pytest.raises(ValueError):
        load_cost_settings()


def test_feature_off_with_no_cost_variables_starts_cleanly():
    assert load_cost_settings().enabled is False
