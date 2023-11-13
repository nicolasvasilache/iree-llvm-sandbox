INSTRUCTIONS = """
  1. Checkout iree, shark-turbine, and iree-llvm-sandbox in the same root directory.
  2. Follow the shark README.md for installing with a custom version of IREE. Steps are:
     a. pip install shark-turbine
     b. pip install --upgrade -r requirements.txt
     c. pip install --upgrade -e .[torch-cpu-nightly,testing]
     d. pip uninstall iree-compiler
     e. pip uninstall iree-runtime
  3. build iree with python support: -DIREE_BUILD_PYTHON_BINDINGS=ON -DPython3_EXECUTABLE="$(which python)"
  4. export ROOT_DIR=path/to/root/dir
  5. export PYTHONPATH=${PYTHONPATH}:${ROOT_DIR}/iree-llvm-sandbox/build/tools/structured/python_packages:${ROOT_DIR}/iree/build/compiler/bindings/python:${ROOT_DIR}/iree/build/runtime/bindings/python:${ROOT_DIR}/SHARK-Turbine
  6. python foo.py
"""

from iree.compiler.dialects import transform
from iree.compiler.dialects.transform import (
    # bufferization,
    iree_common,
    # gpu,
    # loop,
    memref,
    # nvgpu,
    structured,
    tensor,
    vector,
)


import os
NUM_THREADS=int(os.environ["NUM_THREADS"]) if "NUM_THREADS" in os.environ else 1
TRACY=True if "TRACY" in os.environ else False 
DEBUG = True if "DEBUG" in os.environ else False 
STRATEGY=os.environ["STRATEGY"] if "STRATEGY" in os.environ else "transform-dialect"
OUTPUT_VMFB=os.environ["OUTPUT_VMFB"] if "OUTPUT_VMFB" in os.environ else None

#                               M,  K,  N
compute_shape = [NUM_THREADS * 64, 64, 48]

import torch
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = nn.Linear(*compute_shape[1:3], bias=False)

    def forward(self, x: torch.Tensor):
        x = self.layer0(x)
        return x

class MM(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(compute_shape[1:3]))

    def forward(self, x):
        return torch.mm(x, self.a)

# MLP_OR_MM = {"layer": MM(), "op_name": "linalg.matmul"}
MLP_OR_MM = {"layer": MLP(), "op_name": "linalg.matmul_transpose_b"}

model = MLP_OR_MM["layer"]

from jasc import jasc

def schedule(variant: jasc.OpHandle) -> None:
  tiled_matmul, loops = variant.match_ops(MLP_OR_MM["op_name"]).tile(
      loop=jasc.TileLoopKind.FORALL, 
      num_threads=(NUM_THREADS, ),
      mapping = ["#gpu.block<x>"]
  )
  variant.match_ops("linalg.fill").fuse_into(loops[0])
  iree_common.IREEPopulateWorkgroupCountRegionUsingNumThreadsSliceOp(
    loops[0].mlir_value)

  # Tile matmul again, then interchange.
  tiled_matmul_l2 = tiled_matmul
  # tiled_matmul_l2 = tiled_matmul.tile(
  #     loop=jasc.TileLoopKind.FOR,
  #     tile_sizes=(128, 128, 256),
  # ).tiled_op.generalize().interchange([0, 2, 1])

  tiled_matmul_l3_a, loops_l3_a = tiled_matmul_l2.tile(
      loop=jasc.TileLoopKind.FOR,
      # tile_sizes=(16, 16, 1),
      # tile_sizes=(4, 16, 16),
      tile_sizes=(16, 0, 0),
      # interchange=(1, 2, 0),
  )
  variant.match_ops("linalg.fill").fuse_into(loops_l3_a[-1])
  tiled_matmul_l3_b, loops_l3_b = tiled_matmul_l3_a.tile(
      loop=jasc.TileLoopKind.FOR,
      # tile_sizes=(16, 16, 1),
      # tile_sizes=(4, 16, 16),
      tile_sizes=(0, 16, 0),
      # interchange=(1, 2, 0),
  )
  variant.match_ops("linalg.fill").fuse_into(loops_l3_b[-1])


  tiled_matmul_l3_a_b = tiled_matmul_l3_b.tile(
      loop=jasc.TileLoopKind.FOR,
      # tile_sizes=(16, 16, 1),
      tile_sizes=(0, 0, 16),
      interchange=(1, 2, 0),
  ).tiled_op

  # Manual clean-up.
  func = variant.match_ops("func.func")
  with func.apply_patterns():
    transform.ApplyCanonicalizationPatternsOp()
    structured.ApplyTilingCanonicalizationPatternsOp()
  func.apply_cse()
  func.match_ops("LoopLikeInterface").apply_licm()
  with func.apply_patterns():
    structured.ApplyFoldUnitExtentDimsViaReshapesPatternsOp()

  # Vectorize function.
  func.vectorize_children_and_apply_patterns(vectorize_padding=True)

  # Hoist redundant transfers.
  with func.apply_patterns():
    transform.ApplyCanonicalizationPatternsOp()
    tensor.ApplyFoldTensorSubsetOpsIntoVectorTransfersPatternsOp()
  func.apply_cse()

  func.match_ops("LoopLikeInterface").hoist_loop_invariant_subsets()
  
  with func.apply_patterns():
    transform.ApplyCanonicalizationPatternsOp()
    tensor.ApplyFoldTensorSubsetOpsIntoVectorTransfersPatternsOp()
  func.apply_cse()

  # Bufferize.
  func.eliminate_empty_tensors()
  variant = jasc.OpHandle(       \
    iree_common.IREEBufferizeOp( \
      result=transform.AnyOpType.get(), target=variant.mlir_value))

  # Lowering of vector ops.
  func = variant.match_ops("func.func")
  with func.apply_patterns():
    transform.ApplyCanonicalizationPatternsOp()

  func.hoist_redundant_vector_transfers()

  with func.apply_patterns():
    vector.ApplyLowerContractionPatternsOp()
    vector.ApplyLowerTransposePatternsOp(
      lowering_strategy=vector.VectorTransposeLowering.Shuffle16x16
        # lowering_strategy=vector.VectorTransposeLowering.Flat,
        # avx2_lowering_strategy=True,
        )
    vector.ApplyLowerTransferPatternsOp()
    vector.ApplyLowerShapeCastPatternsOp()
  with func.apply_patterns():
    vector.ApplyTransferToScfPatternsOp(full_unroll=True)
    memref.ApplyAllocToAllocaOp()

  # Final foldings and clean-up.
  with func.apply_patterns():
    memref.ApplyFoldMemrefAliasOpsPatternsOp()
    transform.ApplyCanonicalizationPatternsOp()
  func.apply_cse()
  
  jasc.OpHandle(iree_common.ForallToWorkgroupOp(func.mlir_value))
  func.print()



from iree.compiler.ir import (
   Context,
   InsertionPoint,
   IntegerType,
   Location,
   Module,
   UnitAttr,
)
import sys

TD_LIBRARY_FILE_NAME="/tmp/schedule.mlir"

ctx = Context()
ctx.allow_unregistered_dialects = True
with Location.unknown(ctx):
    i32 = IntegerType.get_signless(32)
    module = Module.create()
    module.operation.attributes["transform.with_named_sequence"] = UnitAttr.get()
    # Insert the codegen into the IR
    with module.context, Location.unknown(module.context):
        with InsertionPoint.at_block_begin(module.body):
            named_sequence = transform.NamedSequenceOp(
                "codegen",
                [transform.AnyOpType.get()],
                [],
                arg_attrs = [{"transform.consumed": UnitAttr.get()}])
            with InsertionPoint(named_sequence.body):
                schedule(jasc.OpHandle(named_sequence.bodyTarget))
                transform.YieldOp([])
    
        # Insert the match_variant_for_codegen into the IR
        with InsertionPoint.at_block_begin(module.body):
            named_sequence = transform.NamedSequenceOp(
                "match_variant_for_codegen",
                [transform.AnyOpType.get()],
                [transform.AnyOpType.get()],
                arg_attrs = [{"transform.readonly": UnitAttr.get()}])
            with InsertionPoint(named_sequence.body):
                transform.MatchOperationNameOp(named_sequence.bodyTarget, ["hal.executable.variant"])
                transform.YieldOp([named_sequence.bodyTarget])

        # Insert the __transform_main entry point into the IR
        with InsertionPoint.at_block_begin(module.body):
            named_sequence = transform.NamedSequenceOp(
                "__transform_main",
                [transform.AnyOpType.get()],
                [],
                arg_attrs = [{"transform.consumed": UnitAttr.get()}])
            with InsertionPoint(named_sequence.body):
                transform.ForeachMatchOp(
                    updated=transform.AnyOpType.get(),
                    root=named_sequence.bodyTarget, 
                    matchers=["match_variant_for_codegen"],
                    actions=["codegen"],
                    restrict_root=True)
                transform.YieldOp([])

    original_stdout = sys.stdout
    with open(f"{TD_LIBRARY_FILE_NAME}", "w") as f:
        sys.stdout = f
        print(module)
        sys.stdout = original_stdout
    


import shark_turbine.aot as aot
example_x = torch.empty(*compute_shape[0:2], dtype=torch.float32)
exported = aot.export(model, example_x)
aot.CompiledModule.run_import(exported.compiled_module)
exported.print_readable()


from iree.compiler.api import _initializeGlobalCL
global_flags = [f"--iree-llvmcpu-target-cpu=host",] + \
  [ f"--iree-opt-data-tiling",] if (STRATEGY == "data-tiling" or STRATEGY == "ukernels") else [] \
    + \
  [ f"--iree-llvmcpu-enable-microkernels",] if STRATEGY == "ukernels" else [] \
    + \
  [ f"--iree-codegen-use-transform-dialect-strategy=__transform_main",
    f"--iree-codegen-transform-dialect-library={TD_LIBRARY_FILE_NAME}",] \
    if STRATEGY == "transform-dialect" else [] \
    + \
  [ f"--iree-llvmcpu-link-embedded=false",
    f"--iree-llvmcpu-debug-symbols=true",] if TRACY else [] \
    + \
  [ f"--mlir-disable-threading=true",
    f"--mlir-print-ir-after-all",
    f"--debug-only=transform-dialect",
    f"--debug-only=linalg-transforms",
    f"--debug-only=transform-matcher",
    f"--debug-only=transform-dialect-print-top-level-after-all",] if DEBUG else []

_initializeGlobalCL(*global_flags)

compiled_binary = exported.compile(save_to=OUTPUT_VMFB)
if OUTPUT_VMFB is not None:
   print(f"Done generating {OUTPUT_VMFB}")
   exit(0)


def infer(x, n_iter=10):
    import iree.runtime as rt

    config = rt.Config("local-task" if NUM_THREADS > 1 else "local-sync")
    vmm = rt.load_vm_module(
        rt.VmModule.wrap_buffer(config.vm_instance, compiled_binary.map_memory()),
        config,
    )
    print(f"start run with {NUM_THREADS} threads")
    y = vmm.main(x)
    def run():
      start_time = time.time()
      y = vmm.main(x)
      print("--- %s IREE seconds ---" % (time.time() - start_time))
      res = y.to_host()
      return res
    
    run()
    for i in range(n_iter):
      res = run()
    return res


# class ModelTest(unittest.TestCase):
#     def testMLPExportSimple(selfs):
#         infer()

import time
import numpy as np

n_iter = 10

torch.set_num_threads(NUM_THREADS)
x = np.random.rand(*compute_shape[0:2]).astype(np.float32)
y = infer(x, n_iter=n_iter)

tx = torch.from_numpy(x)
def run(): 
  start_time = time.time()
  z = model.forward(tx)
  print("--- %s PT seconds ---" % (time.time() - start_time))
  return z

run()
for i in range(n_iter):
  z = run()

diffs = torch.where(torch.isclose(torch.from_numpy(y), z, atol = 2e-7 * compute_shape[1]) != True)
if diffs[0].numel() > 0:
  print(torch.sub(z[diffs[0], diffs[1]], torch.from_numpy(y)[diffs[0], diffs[1]]))
  print(f"#errors: {diffs[0].shape} (i.e. {diffs[0].numel() / z.numel():2.3f}% errors)")
else:
   print("SUCCESS!")


# Some commands:

EXAMPLE = """
NUM_THREADS=1 OUTPUT_VMFB=/tmp/ukernels.vmfb STRATEGY=ukernels /usr/local/google/home/ntv/.venv/mlirdev/bin/python foo.py
TRACY_NO_EXIT=1 IREE_PRESERVE_DYLIB_TEMP_FILES=$(pwd) ../iree/build/tools/iree-benchmark-module   --module=/tmp/ukernels.vmfb   --device=local-sync   --function=main   --input=64x64xf32=0.9898  

../iree/build/tracy/iree-tracy-capture -o /tmp/ukernel.vmfb.tracy


NUM_THREADS=1 OUTPUT_VMFB=/tmp/td.vmfb STRATEGY=transform-dialect /usr/local/google/home/ntv/.venv/mlirdev/bin/python foo.py
TRACY_NO_EXIT=1 IREE_PRESERVE_DYLIB_TEMP_FILES=$(pwd) ../iree/build/tools/iree-benchmark-module   --module=/tmp/td.vmfb   --device=local-sync   --function=main   --input=64x64xf32=0.9898  

../iree/build/tracy/iree-tracy-capture -o /tmp/td.vmfb.tracy
"""

# ../iree/build/tools/iree-benchmark-module   --module=/tmp/foo.vmfb   --device=local-sync   --function=main   --input=64x64xf32=0.9898  --batch-size=10 --benchmark-repetitions=10

# ../iree/build/tracy/iree-tracy-capture -o /tmp/td.vmfb.tracy

# cd build && cmake -DIREE_BUILD_TRACY=ON -DIREE_ENABLE_LLD=ON . && cmake --build . --target iree-tracy-profiler iree-tracy-capture iree-tracy-csvexport

# objdump -D iree_dylib_2sfZ0P_mem_.so | grep -A1024 matm | less



# NUM_THREADS=4 OUTPUT_VMFB=/tmp/foo.vmfb STRATEGY=ukernels /usr/local/google/home/ntv/.venv/mlirdev/bin/python foo.py && unzip /tmp/foo.vmfb && ../iree/build/runtime/src/iree/hal/local/executable_library_benchmark --executable_format=embedded-elf --executable_file=main_dispatch_0_embedded_elf_x86_64.so --entry-point=0 -workgroup_count_x=4     --workgroup_count_y=1     --workgroup_count_z=1     --workgroup_size_x=1     --workgroup_size_y=1     --workgroup_size_z=1 \  --binding=64x64xf32=0.9898 --binding=64x48xf32=0.1234 --binding=64x48xf32=0
