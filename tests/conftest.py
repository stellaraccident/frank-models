"""pytest fixtures for IREE runtime setup."""

import pytest
from tests.utils import IREERuntime


def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        action="store",
        default="llvm-cpu",
        help="IREE backend to use: llvm-cpu, rocm, cuda, vulkan",
    )


@pytest.fixture(scope="session")
def rt(request) -> IREERuntime:
    """IREE runtime components."""
    backend = request.config.getoption("--backend")
    return IREERuntime(backend)
