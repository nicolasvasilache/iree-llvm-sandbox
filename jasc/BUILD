# Schedules for JAX.

load("@rules_python//python:defs.bzl", "py_library")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@pip_deps//:requirements.bzl", "requirement")

package(
    # default_applicable_licenses = ["//third_party/mlir_edge:license"],
    default_visibility = ["//:__subpackages__"],
)

py_library(
    name = "jasc",
    srcs = ["jasc.py"],
    deps = [
        ":call_kernel",
        ":primitives",
        "//dialect:python",
        "//transform_ops",
        requirement("jax"),
        requirement("jaxlib"),
        # "@jax//jaxsparse",
        # "//mlir:bufferization_dialect",
        "//mlir:core",
        "//mlir:ir",
        "//mlir:pdl_dialect",
        "//mlir:transform_dialect",
    ],
)

py_library(
    name = "tuner",
    srcs = ["tuner.py"],
    deps = [
        ":jasc",
        "//:call_kernel",
        "//dialect:python",
        "//transform_ops",
        # "//third_party/py/absl:app",
        # "//third_party/py/chex",
        requirement("jax"),
        "//mlir:core",
        "//mlir:ir",
        "//mlir:pdl_dialect",
        "//mlir:transform_dialect",
        # "//util/task/python:error",
    ],
)

py_library(
    name = "primitives",
    srcs = ["primitives.py"],
    deps = [
        ":call_kernel",
        "//dialect:python",
        requirement("jax"),
        # "@jax//jax",
        # "@jax//jax:extend",
        "//mlir:ir",
        "//mlir:pdl_dialect",
        # "//mlir:stablehlo_dialect",
        "//mlir:transform_dialect",
    ],
)

pybind_extension(
    name = "call_kernel",
    srcs = ["call_kernel.cc"],
    deps = [
        ":mlir_lowering",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        # "//third_party/gpus/cuda:cuda_headers",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:CAPIIRHeaders",
        "@llvm-project//mlir:ExecutionEngine",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MLIRBindingsPythonHeaders",
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "@status_macros//:status_macros",
    ],
)

cc_library(
    name = "mlir_lowering",
    srcs = [
        "gpu_lowering_passes.cc",
        "mlir_lowering.cc",
    ],
    hdrs = [
        "gpu_lowering_passes.h",
        "mlir_lowering.h",
    ],
    data = ["gpu_post_bufferize.mlir"],
    deps = [
        "@com_google_absl//absl/status",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineToStandard",
        "@llvm-project//mlir:AllToLLVMIRTranslations",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithToLLVM",
        "@llvm-project//mlir:ArithTransforms",
        "@llvm-project//mlir:BufferizationDialect",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:ControlFlowToLLVM",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncToLLVM",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:GPUToGPURuntimeTransforms",
        "@llvm-project//mlir:GPUToNVVMTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:IndexToLLVM",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MathToLLVM",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MemRefToLLVM",
        "@llvm-project//mlir:MemRefTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ReconcileUnrealizedCasts",
        "@llvm-project//mlir:SCFToControlFlow",
        "@llvm-project//mlir:SerializeToCubin_stub",
        "@llvm-project//mlir:SparseTensorTransforms",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformDialectTransforms",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorToLLVM",
        "@llvm-project//mlir:VectorToSCF",
        "//dialect",
        "//transform_ops:jasc_transform_ops",
        "@xla//xla/mlir_hlo:mhlo_passes",
    ],
)

cc_binary(
    name = "jasc-opt",
    srcs = ["jasc_opt.cc"],
    deps = [
        ":mlir_lowering",
        "@llvm-project//mlir:AllExtensions",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:AllToLLVMIRTranslations",
        "@llvm-project//mlir:MlirOptLib",
        "//dialect",
        "//transform_ops:jasc_transform_ops",
    ],
)
