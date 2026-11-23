"""FastAPI backend: runs the research package's inference for the web app. It never trains."""

import os

# torch and xgboost each load their own OpenMP runtime on macOS; two runtimes starting thread pools
# in one process crash (docs/decisions.md D28). This must run before either library is imported,
# which is why it sits in the package's __init__, the first file Python runs for `backend.app.*`.
# Torch's own thread count is raised again in services/models.py.
os.environ.setdefault("OMP_NUM_THREADS", "1")
