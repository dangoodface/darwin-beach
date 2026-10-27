from pathlib import Path


def pytest_configure(config):
    # /tmp is RAM on this machine: keep pytest's scratch folders on the SSD
    if config.option.basetemp is None:
        config.option.basetemp = Path.home() / ".cache" / "arena-pytest"
