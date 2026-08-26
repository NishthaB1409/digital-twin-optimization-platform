import pytest

from dtmo.utils.config import load_config


@pytest.fixture(scope="session")
def config():
    """The real factory config, loaded once."""
    return load_config()


@pytest.fixture
def small_config(config):
    """A short run, so tests stay fast."""
    return config.with_overrides(n_jobs=25, seed=7)
