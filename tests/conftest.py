"""pytest fixtures for IREE runtime setup."""

from pathlib import Path

import pytest

from tests.utils import IREEConfig


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
    parser.addoption(
        "--save-linked-ir",
        action="store",
        default=None,
        help="Directory to save post-iree-link MLIR for inspection. "
        "Created if it doesn't exist.",
    )


@pytest.fixture(scope="session")
def iree_cfg(request) -> IREEConfig:
    """IREE configuration and runtime components."""
    backend = request.config.getoption("--backend")
    tools_dir_str = request.config.getoption("--iree-tools-dir")
    tools_dir = Path(tools_dir_str) if tools_dir_str else None
    save_linked_ir_str = request.config.getoption("--save-linked-ir")
    save_linked_ir = Path(save_linked_ir_str) if save_linked_ir_str else None
    return IREEConfig(
        backend=backend,
        iree_tools_dir=tools_dir,
        save_linked_ir=save_linked_ir,
    )
