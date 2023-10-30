# MLIR Dialect to support Jasc transformations.

load("@rules_python//python:defs.bzl", "py_library")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "gentbl_filegroup", "td_library")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(
    # default_applicable_licenses = ["//third_party/mlir_edge:license"],
    default_visibility = ["//:__subpackages__"],
)

td_library(
    name = "td_files",
    srcs = [
        "dialect.td",
        "ops.td",
    ],
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:TransformDialectTdFiles",
    ],
)

gentbl_cc_library(
    name = "dialect_inc_gen",
    tbl_outs = [
        (
            ["-gen-dialect-decls"],
            "dialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "dialect.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "dialect.td",
    deps = [":td_files"],
)

gentbl_cc_library(
    name = "ops_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ops.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
    ],
)

cc_library(
    name = "dialect",
    srcs = [
        "dialect.cc",
        "ops.cc",
    ],
    hdrs = [
        "dialect.h",
        "ops.h",
    ],
    deps = [
        ":dialect_inc_gen",
        ":ops_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:TransformDialect",
    ],
)

gentbl_filegroup(
    name = "ops_py_gen",
    tbl_outs = [
        (
            [
                "-gen-python-op-bindings",
                "-bind-dialect=jasc",
            ],
            "_ops_gen.py",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ops_py.td",
    deps = [
        ":td_files",
    ],
)

pybind_extension(
    name = "bindings",
    srcs = ["bindings.cc"],
    deps = [
        ":dialect",
        "@llvm-project//mlir:CAPIIRHeaders",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MLIRBindingsPythonHeaders",
        "//:mlir_lowering",
        "//transform_ops:jasc_transform_ops",
    ],
)

py_library(
    name = "python",
    srcs = [
        "__init__.py",
        "_ods_common.py",
        ":ops_py_gen",
    ],
    deps = [
        ":bindings",
        # "@jax//jaxlib/mlir:core",
        # "@jax//jaxlib/mlir:ir",
        # "@jax//jaxlib/mlir:pdl_dialect",
    ],
)
