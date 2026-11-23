import os
import sys

# torch and xgboost each bring their own copy of the OpenMP runtime on macOS, and two copies running
# thread pools in one process crash (docs/decisions.md D28). The tests use both, so start OpenMP
# single-threaded before either is imported. Must stay above every other import.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pytest


def pytest_configure(config: pytest.Config) -> None:
    # Running the tests with a global or Anaconda Python would test packages that are not the
    # pinned ones in uv.lock, so a passing run would prove nothing. Stop early instead.
    if sys.prefix == sys.base_prefix:
        raise pytest.UsageError(
            "Tests must run inside the project virtual environment. "
            "Use `make test` or `uv run pytest`."
        )
