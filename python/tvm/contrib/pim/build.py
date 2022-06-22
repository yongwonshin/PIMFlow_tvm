import tvm
from tvm import runtime, relay

def build_pim_kernels(lib, sm, tmp_dir="./tmp", lib_path="compile.so"):
  """Compile PIM kernels in lib and return the runtime module ready to run.
    Parameters
    ----------
    lib : GraphExecutorFactoryModule
        The output from relay.build containing compiled host code and non-cutlass kernels.
    sm : int
        An integer specifying the compute capability. For example, 75 for Turing and
        80 or 86 for Ampere.
    tmp_dir : string, optional
        A temporary directory where intermediate compiled artifacts will be stored.
    lib_path : string, optional
        The path to a shared library which will be generated as the result of the build  process
    Returns
    -------
    updated_lib : runtime.Module
        The updated module with compiled cutlass kernels.
    """
  kwargs = {}
  kwargs["cc"] = "nvcc"
  kwargs["options"] = [
    "-O3",
    "-Xcompiler=-fPIC",
    "-Xcompiler=-Wconversion",
    "-Xcompiler=-fno-strict-aliasing",
    "-std=c++14",
    "-lcublas",
    "-lcudnn",
    "-lcurand",
  ]
  lib.export_library(lib_path, workspace_dir=tmp_dir, **kwargs)
  return runtime.load_module(lib_path)
