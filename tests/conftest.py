import os
from pathlib import Path


def pytest_configure(config):
    # /tmp is RAM on this machine: keep pytest's scratch folders on the SSD.
    # Use the temp *root*, not --basetemp: an explicit basetemp is wiped when a session starts,
    # so two sessions at once (a trainer branch and main, say) deleted each other's stores.
    # Under a root pytest numbers its folders per session, which is safe to share.
    root = Path.home() / ".cache" / "arena-pytest-root"
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTEST_DEBUG_TEMPROOT", str(root))
