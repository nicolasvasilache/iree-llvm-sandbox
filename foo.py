# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# export PYTHONPATH=/usr/local/buildtools/current/sitecustomize:${ROOT_DIR}/iree-llvm-sandbox/build/tools/structured/python_packages:${ROOT_DIR}/llvm-project/build/tools/mlir/python_packages/mlir_core:${ROOT_DIR}/iree/build/compiler/bindings/python:${ROOT_DIR}/iree/build/runtime/bindings/python:${ROOT_DIR}/SHARK-Turbine

import logging
import unittest
import torch
import torch.nn as nn

import shark_turbine.aot as aot


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = nn.Linear(8, 8, bias=False)
        # self.layer1 = nn.Linear(8, 4, bias=True)
        # self.layer2 = nn.Linear(4, 2, bias=True)
        # self.layer3 = nn.Linear(2, 2, bias=True)

    def forward(self, x: torch.Tensor):
        x = self.layer0(x)
        # x = torch.sigmoid(x)
        # x = self.layer1(x)
        # x = torch.sigmoid(x)
        # x = self.layer2(x)
        # x = torch.sigmoid(x)
        # x = self.layer3(x)
        return x


# def infer():
#     import numpy as np
#     import iree.runtime as rt

#     config = rt.Config("local-task")
#     vmm = rt.load_vm_module(
#         rt.VmModule.wrap_buffer(config.vm_instance, compiled_binary.map_memory()),
#         config,
#     )
#     x = np.random.rand(97, 8).astype(np.float32) # -> vectorization fails atm
#     y = vmm.main(x)
#     print(y.to_host())

from mlir import ir
from mlir.dialects import pdl
from mlir.dialects import transform
from mlir.dialects.bufferization import LayoutMapOption
from mlir.dialects.transform import (
    bufferization,
    gpu,
    loop,
    memref,
    nvgpu,
    # sparse_tensor,
    structured,
    tensor,
    vector,
)

from jasc import jasc

def schedule(variant: jasc.OpHandle) -> None:
  # Tile matmul.
  # Note: Unlike the original schedule, we tile to `scf.forall` such that we
  #       can fuse the `linalg.fill`, which the other schedule doesn't have.
  matmul = variant.match_ops("linalg.matmul_transpose_b")
  tiled_matmul, loops = matmul.tile(
      loop=jasc.TileLoopKind.FORALL, tile_sizes=(8, 4)
  )

  loops[0].print(name="FORALL INSIDE CODEGEN")
  variant.match_ops("linalg.fill").print(name="FILL INSIDE CODEGEN")

  variant.match_ops("linalg.fill").fuse_into(loops[0])

  # Tile matmul again, then interchange.
  tiled_matmul_l2 = tiled_matmul.tile(
      loop=jasc.TileLoopKind.FOR,
      tile_sizes=(0, 0, 8),
  ).tiled_op
  
  tiled_matmul_l2.print(name="RETILED INSIDE CODEGEN")
  tiled_matmul_l2.generalize().interchange([0, 2, 1]).vectorize(vector_sizes=[8, 8, 4])
  variant.match_ops("linalg.fill").vectorize(vector_sizes=[8, 4])

  # Manual clean-up.
  func = variant.match_ops("func.func")
  with func.apply_patterns():
    transform.ApplyCanonicalizationPatternsOp()
    structured.ApplyTilingCanonicalizationPatternsOp()
  func.apply_cse()
  func.match_ops("LoopLikeInterface").apply_licm()
  with func.apply_patterns():
    structured.ApplyFoldUnitExtentDimsViaReshapesPatternsOp()
    vector.ApplyLowerMaskedTransfersPatternsOp()
    vector.ApplyTransferPermutationPatternsOp()
    vector.ApplyVectorReductionToContractPatternsOp()

  # Vectorize function (skip with masked vectorization)
  # func.vectorize_children_and_apply_patterns(vectorize_padding=True)

  # Hoist redundant transforms.
  with func.apply_patterns():
    transform.ApplyCanonicalizationPatternsOp()
    tensor.ApplyFoldTensorSubsetOpsIntoVectorTransfersPatternsOp()
  func.apply_cse()
  func.hoist_redundant_vector_transfers()

  # Bufferize.
  variant.one_shot_bufferize(
      bufferize_function_boundaries=True,
      function_boundary_type_conversion="IdentityLayoutMap",
  )

  # Turn the `scf.forall` into `scf.for`.
  # Note: The original schedule does not do that since it creates `scf.for`
  #       right away (see above).
  forall = variant.match_ops("scf.forall")
  loop.ForallToForOp([transform.AnyOpType.get()], forall.mlir_value)

  # Lowering of vector ops.
  func = variant.match_ops("func.func")
  with func.apply_patterns():
    transform.ApplyCanonicalizationPatternsOp()
  with func.apply_patterns():
    vector.ApplyLowerContractionPatternsOp()
    vector.ApplyLowerTransposePatternsOp()
    vector.ApplyLowerTransferPatternsOp()
    vector.ApplyLowerShapeCastPatternsOp()
  with func.apply_patterns():
    vector.ApplyTransferToScfPatternsOp(full_unroll=True)
    memref.ApplyAllocToAllocaOp()

  # Hoist buffers. (Does not have any effect on this input).
  func.buffer_loop_hoisting()

  # Final foldings and clean-up.
  with func.apply_patterns():
    memref.ApplyFoldMemrefAliasOpsPatternsOp()
    transform.ApplyCanonicalizationPatternsOp()
  func.apply_cse()



from mlir.ir import *
import sys

TD_LIBRARY_FILE_NAME="/tmp/schedule.mlir"

ctx = Context()
ctx.allow_unregistered_dialects = True
with Location.unknown(ctx):
    i32 = IntegerType.get_signless(32)
    module = Module.create()
    module.operation.attributes["transform.with_named_sequence"] = UnitAttr.get()
    # Insert the codegen into the IR
    with module.context, ir.Location.unknown(module.context):
        with ir.InsertionPoint.at_block_begin(module.body):
            named_sequence = transform.NamedSequenceOp(
                "codegen",
                [transform.AnyOpType.get()],
                [],
                arg_attrs = [{"transform.consumed": UnitAttr.get()}])
            with ir.InsertionPoint(named_sequence.body):
                schedule(jasc.OpHandle(named_sequence.bodyTarget))
                transform.YieldOp([])
    
        # Insert the match_variant_for_codegen into the IR
        with ir.InsertionPoint.at_block_begin(module.body):
            named_sequence = transform.NamedSequenceOp(
                "match_variant_for_codegen",
                [transform.AnyOpType.get()],
                [transform.AnyOpType.get()],
                arg_attrs = [{"transform.readonly": UnitAttr.get()}])
            with ir.InsertionPoint(named_sequence.body):
                transform.PrintOp(target=named_sequence.bodyTarget, name="in")
                transform.MatchOperationNameOp(named_sequence.bodyTarget, ["hal.executable.variant"])
                # matmul = jasc.OpHandle(named_sequence.bodyTarget).match_ops("linalg.matmul_transpose_b")
                # matmul.print()
                # variant = matmul.get_parent_op(op_name="hal.executable.variant")
                # transform.MatchOperationNameOp(variant.mlir_value, ["hal.executable.variant"])
                transform.YieldOp([named_sequence.bodyTarget])

        # Insert the __transform_main entry point into the IR
        with ir.InsertionPoint.at_block_begin(module.body):
            named_sequence = transform.NamedSequenceOp(
                "__transform_main",
                [transform.AnyOpType.get()],
                [],
                arg_attrs = [{"transform.consumed": UnitAttr.get()}])
            with ir.InsertionPoint(named_sequence.body):
                transform.ForeachMatchOp(
                    updated=transform.AnyOpType.get(),
                    root=named_sequence.bodyTarget, 
                    matchers=["match_variant_for_codegen"],
                    actions=["codegen"],
                    restrict_root=True)
                transform.YieldOp([])

  # // Find `hal.executable.variant`.
  # transform.named_sequence @match_variant_for_codegen(%root: !transform.any_op {transform.readonly}) 
  #   -> !transform.any_op {
  #   transform.match.operation_name %root ["hal.executable.variant"] : !transform.any_op
  #   transform.yield %root : !transform.any_op
  # }

  # // Transform entry-point
  # transform.named_sequence @__transform_main(%root: !transform.any_op {transform.consumed}) {
  #   transform.foreach_match in %root
  #       @match_variant_for_codegen -> @codegen,
  #       @match_func_for_dispatch -> @dispatch
  #     : (!transform.any_op) -> (!transform.any_op)
  #   transform.yield 
  # }

    original_stdout = sys.stdout
    with open(f'{TD_LIBRARY_FILE_NAME}', 'w') as f:
        sys.stdout = f
        print(module)
        sys.stdout = original_stdout
    

model = MLP()
example_x = torch.empty(97, 8, dtype=torch.float32) # -> vectorization fails atm with 97
exported = aot.export(model, example_x)
aot.CompiledModule.run_import(exported.compiled_module)
exported.print_readable()


from iree.compiler.api import _initializeGlobalCL
_initializeGlobalCL(
   "--mlir-disable-threading=true", 
  #  "--mlir-print-ir-after-all",
   f"--iree-codegen-transform-dialect-library={TD_LIBRARY_FILE_NAME}",
   "--debug-only=transform-dialect",
   "--debug-only=transform-matcher",
   "--debug-only=transform-dialect-print-top-level-after-all",
)
compiled_binary = exported.compile(save_to=None)
