"""pytest fixtures for IREE runtime setup."""

from pathlib import Path

import pytest

from tests.utils import IREERuntime


def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        action="store",
        default="llvm-cpu",
        help="IREE backend to use: llvm-cpu, rocm, cuda, vulkan",
    )
    parser.addoption(
        "--iree-tools-dir",
        action="store",
        default=None,
        help="Directory containing IREE tools (iree-link, etc.). "
        "If not specified, tools are searched in PATH.",
    )


@pytest.fixture(scope="session")
def rt(request) -> IREERuntime:
    """IREE runtime components."""
    backend = request.config.getoption("--backend")
    tools_dir_str = request.config.getoption("--iree-tools-dir")
    tools_dir = Path(tools_dir_str) if tools_dir_str else None
    return IREERuntime(backend=backend, iree_tools_dir=tools_dir)
