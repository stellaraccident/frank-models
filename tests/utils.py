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
    ParameterIndex,
    VmContext,
    VmFunction,
    VmInstance,
    VmModule,
    VmVariantList,
    create_hal_module,
    create_io_parameters_module,
    get_device,
)


DRIVER_MAP = {
    "llvm-cpu": "local-task",
    "rocm": "hip",
    "cuda": "cuda",
    "vulkan": "vulkan",
}


class IREEConfig:
    """Bundle of IREE runtime components."""

    def __init__(
        self,
        backend: str = "llvm-cpu",
        iree_tools_dir: Optional[Path] = None,
        save_linked_ir: Optional[Path] = None,
    ):
        self.backend = backend
        self._iree_tools_dir = iree_tools_dir
        self._save_linked_ir = save_linked_ir
        if save_linked_ir is not None:
            save_linked_ir.mkdir(parents=True, exist_ok=True)
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


# Component and layer directories
COMPONENTS_DIR = Path(__file__).parent.parent / "components"
LAYERS_DIR = Path(__file__).parent.parent / "layers"

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
        # Use host CPU targeting to avoid generic CPU warnings and improve codegen
        extra_args = []
        if target_backend == "llvm-cpu":
            extra_args.append("--iree-llvmcpu-target-cpu=host")
        self._binary = iree.compiler.compile_str(
            mlir_source,
            target_backends=[target_backend],
            extra_args=extra_args,
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


def compile_mlir(mlir_source: str, iree_cfg) -> IREEModule:
    """Compile MLIR source string to an IREE module."""
    return IREEModule(
        mlir_source,
        iree_cfg.instance,
        iree_cfg.device,
        iree_cfg.hal_module,
        iree_cfg.backend,
    )


def compile_component(component_path: str, iree_cfg) -> IREEModule:
    """Compile an MLIR component file to an IREE module."""
    full_path = COMPONENTS_DIR / component_path
    mlir_source = full_path.read_text()
    return compile_mlir(mlir_source, iree_cfg)


def assert_close(actual, expected, rtol=1e-5, atol=1e-6):
    """Assert arrays are close within tolerance."""
    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=rtol,
        atol=atol,
    )


def _run_iree_link(
    main_path: str,
    library_paths: List[str],
    iree_cfg: IREEConfig,
    debug_name: Optional[str] = None,
) -> str:
    """Run iree-link and return the linked MLIR source.

    Args:
        main_path: Absolute path to the primary MLIR file
        library_paths: Dependency modules (relative to COMPONENTS_DIR)
        iree_cfg: IREEConfig instance
        debug_name: Base name for saving linked IR (without extension).
            If save_linked_ir is set, the linked IR is saved there.

    Returns:
        Linked MLIR source string
    """
    iree_link = iree_cfg.iree_tool_path("iree-link")

    cmd = [str(iree_link), main_path]
    for lib in library_paths:
        lib_path = Path(lib)
        if lib_path.is_absolute():
            cmd.extend(["--link-module", str(lib_path)])
        else:
            cmd.extend(["--link-module", str(COMPONENTS_DIR / lib)])

    with tempfile.NamedTemporaryFile(suffix=".mlir", delete=False) as f:
        linked_path = f.name
    cmd.extend(["-o", linked_path])

    subprocess.run(cmd, check=True)
    linked_source = Path(linked_path).read_text()

    if iree_cfg._save_linked_ir is not None and debug_name is not None:
        dest = iree_cfg._save_linked_ir / f"{debug_name}.mlir"
        dest.write_text(linked_source)

    Path(linked_path).unlink(missing_ok=True)
    return linked_source


def link_and_compile(
    *,
    main_source: Optional[str] = None,
    main_path: Optional[str] = None,
    library_paths: List[str],
    iree_cfg: IREEConfig,
    debug_name: Optional[str] = None,
) -> IREEModule:
    """Link MLIR modules with iree-link, then compile.

    Provide exactly one of main_source or main_path.

    Args:
        main_source: MLIR source string for the primary module
        main_path: Primary module file (relative to COMPONENTS_DIR)
        library_paths: Dependency modules to link (relative to COMPONENTS_DIR)
        iree_cfg: IREEConfig instance
        debug_name: Base name for saving linked IR. Defaults to main_path stem.

    Returns:
        Compiled IREEModule with all functions available
    """
    if (main_source is None) == (main_path is None):
        raise ValueError("Provide exactly one of main_source or main_path")

    if main_source is not None:
        with tempfile.NamedTemporaryFile(suffix=".mlir", delete=False, mode="w") as f:
            f.write(main_source)
            abs_path = f.name
        cleanup = True
    else:
        abs_path = str(COMPONENTS_DIR / main_path)
        if debug_name is None:
            debug_name = Path(main_path).stem + "_linked"
        cleanup = False

    try:
        linked_source = _run_iree_link(abs_path, library_paths, iree_cfg, debug_name)
    finally:
        if cleanup:
            Path(abs_path).unlink(missing_ok=True)

    return compile_mlir(linked_source, iree_cfg)


class IREEModuleWithParams:
    """IREE module with parameter support.

    Like IREEModule but includes an io_parameters module in the VM context
    so that flow.tensor.constant with #flow.parameter.named can resolve
    parameters at runtime.
    """

    def __init__(
        self,
        mlir_source: str,
        instance: VmInstance,
        device: HalDevice,
        hal_module: VmModule,
        target_backend: str,
        params: dict[str, np.ndarray],
        scope: str = "model",
    ):
        self._device = device
        self._instance = instance

        # Build parameter index from numpy arrays.
        param_index = ParameterIndex()
        for key, arr in params.items():
            param_index.add_buffer(key, np.ascontiguousarray(arr))
        provider = param_index.create_provider(scope=scope)
        io_module = create_io_parameters_module(instance, provider)

        # Compile MLIR to bytecode.
        # Use host CPU targeting to avoid generic CPU warnings and improve codegen
        extra_args = []
        if target_backend == "llvm-cpu":
            extra_args.append("--iree-llvmcpu-target-cpu=host")
        binary = iree.compiler.compile_str(
            mlir_source,
            target_backends=[target_backend],
            extra_args=extra_args,
        )

        # Load the compiled module.
        vm_module = VmModule.copy_buffer(instance, binary)

        # Create context: io_parameters must come before the compiled module.
        self._vm_module = vm_module
        self._context = VmContext(instance, modules=[io_module, hal_module, vm_module])

    def lookup_function(self, name: str) -> VmFunction:
        return self._vm_module.lookup_function(name)

    def _numpy_to_buffer_view(self, arr: np.ndarray) -> HalBufferView:
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
        device_array = DeviceArray(self._device, bv, implicit_host_transfer=True)
        return device_array.to_host()

    def invoke(
        self,
        function_name: str,
        *args: Union[np.ndarray, float, int],
        num_results: int = 1,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        func = self.lookup_function(function_name)
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
        result_list = VmVariantList(num_results)
        self._context.invoke(func, arg_list, result_list)
        results = []
        for i in range(num_results):
            result_bv = result_list.get_as_object(i, HalBufferView)
            results.append(self._buffer_view_to_numpy(result_bv))
        if num_results == 1:
            return results[0]
        return results

    def __getattr__(self, name: str):
        def invoke_wrapper(*args, num_results: int = 1):
            return self.invoke(name, *args, num_results=num_results)

        return invoke_wrapper


def compile_mlir_with_params(
    mlir_source: str,
    iree_cfg: IREEConfig,
    params: dict[str, np.ndarray],
    scope: str = "model",
) -> IREEModuleWithParams:
    """Compile MLIR source with parameter support."""
    return IREEModuleWithParams(
        mlir_source,
        iree_cfg.instance,
        iree_cfg.device,
        iree_cfg.hal_module,
        iree_cfg.backend,
        params,
        scope=scope,
    )


def link_and_compile_with_params(
    *,
    main_source: Optional[str] = None,
    main_path: Optional[str] = None,
    library_paths: List[str],
    iree_cfg: IREEConfig,
    params: dict[str, np.ndarray],
    scope: str = "model",
    debug_name: Optional[str] = None,
) -> IREEModuleWithParams:
    """Link MLIR modules with iree-link, then compile with parameter support.

    library_paths are resolved relative to COMPONENTS_DIR. To include a layer,
    pass an absolute path or use the layer_paths parameter.
    """
    if (main_source is None) == (main_path is None):
        raise ValueError("Provide exactly one of main_source or main_path")

    if main_source is not None:
        with tempfile.NamedTemporaryFile(suffix=".mlir", delete=False, mode="w") as f:
            f.write(main_source)
            abs_path = f.name
        cleanup = True
    else:
        main = Path(main_path)
        abs_path = str(main) if main.is_absolute() else str(COMPONENTS_DIR / main_path)
        if debug_name is None:
            debug_name = Path(main_path).stem + "_linked"
        cleanup = False

    try:
        linked_source = _run_iree_link(abs_path, library_paths, iree_cfg, debug_name)
    finally:
        if cleanup:
            Path(abs_path).unlink(missing_ok=True)

    return compile_mlir_with_params(linked_source, iree_cfg, params, scope=scope)
