import sys

import pytest


def pytest_configure(config: pytest.Config) -> None:
    # Running the tests with a global or Anaconda Python would test packages that are not the
    # pinned ones in uv.lock, so a passing run would prove nothing. Stop early instead.
    if sys.prefix == sys.base_prefix:
        raise pytest.UsageError(
            "Tests must run inside the project virtual environment. "
            "Use `make test` or `uv run pytest`."
        )
