"""Test utilities for frank-models.

Provides IREE compilation and runtime helpers.
"""

import shutil
import subprocess
import tempfile

import numpy as np
from pathlib import Path
from typing import List, Optional, Union

import iree.compiler
from iree.runtime import (
    BufferUsage,
    DeviceArray,
    HalBufferView,
    HalDevice,
    HalElementType,
    MemoryType,
    VmContext,
    VmFunction,
    VmInstance,
    VmModule,
    VmVariantList,
    create_hal_module,
    get_device,
)


DRIVER_MAP = {
    "llvm-cpu": "local-task",
    "rocm": "hip",
    "cuda": "cuda",
    "vulkan": "vulkan",
}


class IREERuntime:
    """Bundle of IREE runtime components."""

    def __init__(
        self,
        backend: str = "llvm-cpu",
        iree_tools_dir: Optional[Path] = None,
    ):
        self.backend = backend
        self._iree_tools_dir = iree_tools_dir
        driver = DRIVER_MAP.get(backend, "local-task")
        self.instance = VmInstance()
        self.device = get_device(driver)
        self.hal_module = create_hal_module(self.instance, self.device)

    def iree_tool_path(self, name: str) -> Path:
        """Get path to an IREE tool binary.

        Args:
            name: Tool name (e.g., "iree-link")

        Returns:
            Path to the tool binary

        Raises:
            FileNotFoundError: If tool cannot be found
        """
        if self._iree_tools_dir is not None:
            tool_path = self._iree_tools_dir / name
            if tool_path.exists():
                return tool_path
            raise FileNotFoundError(
                f"IREE tool '{name}' not found in {self._iree_tools_dir}. "
                f"Check --iree-tools-dir option."
            )

        tool_path = shutil.which(name)
        if tool_path is not None:
            return Path(tool_path)

        raise FileNotFoundError(
            f"IREE tool '{name}' not found in PATH. "
            f"Use --iree-tools-dir to specify the tools directory, "
            f"or add the IREE build tools directory to PATH."
        )


# Component directory
COMPONENTS_DIR = Path(__file__).parent.parent / "components"

# Numpy dtype to IREE element type mapping
DTYPE_TO_ELEMENT_TYPE = {
    np.float32: HalElementType.FLOAT_32,
    np.float64: HalElementType.FLOAT_64,
    np.float16: HalElementType.FLOAT_16,
    np.int32: HalElementType.SINT_32,
    np.int64: HalElementType.SINT_64,
    np.int16: HalElementType.SINT_16,
    np.int8: HalElementType.SINT_8,
    np.uint32: HalElementType.UINT_32,
    np.uint64: HalElementType.UINT_64,
    np.uint16: HalElementType.UINT_16,
    np.uint8: HalElementType.UINT_8,
}


class IREEModule:
    """Low-level wrapper for compiled IREE modules.

    Provides explicit control over argument types, supporting both
    tensors (numpy arrays) and scalars (f32, i32, etc.).
    """

    def __init__(
        self,
        mlir_source: str,
        instance: VmInstance,
        device: HalDevice,
        hal_module: VmModule,
        target_backend: str,
    ):
        self._device = device
        self._instance = instance

        # Compile MLIR to bytecode
        self._binary = iree.compiler.compile_str(
            mlir_source,
            target_backends=[target_backend],
        )

        # Load the compiled module
        self._vm_module = VmModule.copy_buffer(instance, self._binary)

        # Create context with HAL and compiled modules
        self._context = VmContext(instance, modules=[hal_module, self._vm_module])

    def lookup_function(self, name: str) -> VmFunction:
        """Get a VM function by name."""
        return self._vm_module.lookup_function(name)

    def _numpy_to_buffer_view(self, arr: np.ndarray) -> HalBufferView:
        """Convert numpy array to HAL buffer view."""
        arr = np.ascontiguousarray(arr)
        element_type = DTYPE_TO_ELEMENT_TYPE.get(arr.dtype.type)
        if element_type is None:
            raise ValueError(f"Unsupported dtype: {arr.dtype}")

        return self._device.allocator.allocate_buffer_copy(
            memory_type=MemoryType.DEVICE_LOCAL,
            allowed_usage=(BufferUsage.DEFAULT | BufferUsage.MAPPING),
            device=self._device,
            buffer=arr,
            element_type=element_type,
        )

    def _buffer_view_to_numpy(self, bv: HalBufferView) -> np.ndarray:
        """Convert HAL buffer view back to numpy array."""
        device_array = DeviceArray(self._device, bv, implicit_host_transfer=True)
        return device_array.to_host()

    def invoke(
        self,
        function_name: str,
        *args: Union[np.ndarray, float, int],
        num_results: int = 1,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Invoke a function with explicit argument handling.

        Args:
            function_name: Name of the function to call
            *args: Arguments - numpy arrays become tensors, float/int become scalars
            num_results: Number of tensor results expected

        Returns:
            Single numpy array if num_results=1, else list of arrays
        """
        func = self.lookup_function(function_name)

        # Build argument list
        arg_list = VmVariantList(len(args))
        for arg in args:
            if isinstance(arg, np.ndarray):
                bv = self._numpy_to_buffer_view(arg)
                arg_list.push_ref(bv)
            elif isinstance(arg, float) or isinstance(arg, np.floating):
                arg_list.push_float(float(arg))
            elif isinstance(arg, (int, np.integer)):
                arg_list.push_int(int(arg))
            else:
                raise TypeError(f"Unsupported argument type: {type(arg)}")

        # Invoke
        result_list = VmVariantList(num_results)
        self._context.invoke(func, arg_list, result_list)

        # Extract results
        results = []
        for i in range(num_results):
            result_bv = result_list.get_as_object(i, HalBufferView)
            result_arr = self._buffer_view_to_numpy(result_bv)
            results.append(result_arr)

        if num_results == 1:
            return results[0]
        return results

    def __getattr__(self, name: str):
        """Allow calling functions by name as methods."""

        def invoke_wrapper(*args, num_results: int = 1):
            return self.invoke(name, *args, num_results=num_results)

        return invoke_wrapper


def compile_mlir(mlir_source: str, rt) -> IREEModule:
    """Compile MLIR source string to an IREE module."""
    return IREEModule(mlir_source, rt.instance, rt.device, rt.hal_module, rt.backend)


def compile_component(component_path: str, rt) -> IREEModule:
    """Compile an MLIR component file to an IREE module."""
    full_path = COMPONENTS_DIR / component_path
    mlir_source = full_path.read_text()
    return IREEModule(mlir_source, rt.instance, rt.device, rt.hal_module, rt.backend)


def assert_close(actual, expected, rtol=1e-5, atol=1e-6):
    """Assert arrays are close within tolerance."""
    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=rtol,
        atol=atol,
    )


def link_and_compile(
    main_path: str,
    library_paths: List[str],
    rt: IREERuntime,
) -> IREEModule:
    """Link MLIR modules with iree-link, then compile.

    Args:
        main_path: Primary module (relative to COMPONENTS_DIR)
        library_paths: Dependency modules to link
        rt: IREERuntime instance

    Returns:
        Compiled IREEModule with all functions available
    """
    iree_link = rt.iree_tool_path("iree-link")
    main_full = COMPONENTS_DIR / main_path

    # Build iree-link command
    cmd = [str(iree_link), str(main_full)]
    for lib in library_paths:
        cmd.extend(["--link-module", str(COMPONENTS_DIR / lib)])

    with tempfile.NamedTemporaryFile(suffix=".mlir", delete=False) as f:
        linked_path = f.name
    cmd.extend(["-o", linked_path])

    subprocess.run(cmd, check=True)

    linked_source = Path(linked_path).read_text()
    Path(linked_path).unlink()  # cleanup
    return compile_mlir(linked_source, rt)
