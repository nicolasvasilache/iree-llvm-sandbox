# RUN: %PYTHON %s | FileCheck %s
from itertools import permutations
from random import random

import numpy as np

from mlir_structured.dialects import arith, indexing, func
from mlir_structured.dialects.indexing import (Scalar, Tensor, IndexTensorType,
                                               _canonicalize_tuple_index,
                                               arange)
from mlir_structured.ir import Context, IntegerType, F64Type, IndexType, F32Type, MLIRError
from mlir_structured.passmanager import PassManager
from mlir_structured.runtime.util import mlir_mod_ctx, scf_range, scf_yield


def get_array_on_one_line(a):
  return np.array_str(a, max_line_width=np.inf).replace("\n", ",")


def run(f):
  print("\nTEST:", f.__name__)
  with Context():
    indexing.register_dialect()
    f()
  return f


# CHECK-LABEL: TEST: testScalarValue
@run
def testScalarValue():
  f64 = F64Type.get()
  i32 = IntegerType.get_signless(32)
  index = IndexType.get()
  with mlir_mod_ctx() as module:
    zero_f64 = Scalar(arith.ConstantOp(f64, 0.0).result)
    # CHECK: Scalar(%{{.*}}, f64, 0.0)
    print(zero_f64)
    # CHECK: True
    print(zero_f64.is_constant())
    # CHECK: 0.0
    print(zero_f64.literal_value)

    zero_f64 = Scalar(arith.ConstantOp(f64, 0.0))
    # CHECK: Scalar(%{{.*}}, f64, 0.0)
    print(zero_f64)
    # CHECK: True
    print(zero_f64.is_constant())
    # CHECK: 0.0
    print(zero_f64.literal_value)

    zero_f64 = Scalar(0.0)
    # CHECK: Scalar(%{{.*}}, f64, 0.0)
    print(zero_f64)
    # CHECK: True
    print(zero_f64.is_constant())
    # CHECK: 0.0
    print(zero_f64.literal_value)

    zero_i64 = Scalar(0)
    # CHECK: Scalar(%{{.*}}, i64, 0)
    print(zero_i64)
    # CHECK: True
    print(zero_i64.is_constant())
    # CHECK: 0
    print(zero_i64.literal_value)

    zero_i32 = Scalar(0, dtype=i32)
    # CHECK: Scalar(%{{.*}}, i32, 0)
    print(zero_i32)
    # CHECK: True
    print(zero_i32.is_constant())
    # CHECK: 0
    print(zero_i32.literal_value)

    zero_index = Scalar(0, dtype=index)
    # CHECK: Scalar(%{{.*}}, index, 0)
    print(zero_index)
    # CHECK: True
    print(zero_index.is_constant())
    # CHECK: 0
    print(zero_index.literal_value)

    zero_index = zero_index + zero_index
    # CHECK: index
    print(zero_index.type)

    one_f64 = Scalar(1.0)
    two_f64 = Scalar(2.0)

    three_f64 = one_f64 + two_f64
    # CHECK: %{{.*}} = arith.constant 3.000000e+00 : f64
    print(three_f64.owner)

    x, y = random(), random()
    x_f64, y_f64 = Scalar(x), Scalar(y)

    z_f64 = x_f64 + y_f64
    # CHECK: True
    print(z_f64.literal_value == x + y)
    # CHECK: True
    print(zero_f64.is_constant())

    no_fold_one_f64 = Scalar(1.0, fold=False)
    # CHECK: Scalar(%[[NF1:.*]], f64, 1.0)
    print(no_fold_one_f64)
    no_fold_two_f64 = Scalar(2.0, fold=False)
    # CHECK: Scalar(%[[NF2:.*]], f64, 2.0)
    print(no_fold_two_f64)

    no_fold_three_f64 = no_fold_one_f64 + no_fold_two_f64
    # CHECK: %{{.*}} = arith.addf %[[NF1]], %[[NF2]] : f64
    print(no_fold_three_f64.owner)
    # CHECK: False
    print(no_fold_three_f64.is_constant())


# CHECK-LABEL: TEST: testTensorType
@run
def testTensorType():
  i32 = IntegerType.get_signless(32)
  with mlir_mod_ctx():
    tt = Tensor[(10, 10), i32]
    # CHECK: tensor<10x10xi32>
    print(tt)

    tt = Tensor[(None, None), i32]
    # CHECK: tensor<?x?xi32>
    print(tt)

    tt = IndexTensorType.get([10, 10])
    # CHECK: tensor<10x10xindex>
    print(tt)


# CHECK-LABEL: TEST: testTensorValue
@run
def testTensorValue():
  i32 = IntegerType.get_signless(32)
  with mlir_mod_ctx() as module:

    ten = Tensor.empty((10, 10), i32)
    # CHECK: Tensor(%[[TEN:.*]], tensor<10x10xi32>)
    print(repr(ten))
    # CHECK: %[[TEN]] = tensor.empty() : tensor<10x10xi32>
    print(ten.owner)
    # CHECK: (10, 10)
    print(ten.shape)
    # CHECK: i32
    print(ten.dtype)
    # CHECK: False
    print(ten.is_constant())
    try:
      print(ten.literal_value)
    except ValueError as e:
      # CHECK: Can't build literal from non-constant Tensor
      print(e)

    sum_ten_1 = ten + ten
    # CHECK: %[[ADD:.*]] = arith.addi %[[TEN]], %[[TEN]] : tensor<10x10xi32>
    print(sum_ten_1.owner)

    prod_ten = ten * ten
    # CHECK: %[[MUL:.*]] = arith.muli %[[TEN]], %[[TEN]] : tensor<10x10xi32>
    print(prod_ten.owner)

    x = np.random.random((10, 10))
    ten_x = Tensor(x)
    # CHECK: Tensor(%[[CST1:.*]], tensor<10x10xf64>, [
    print(ten_x)
    # CHECK: (10, 10)
    print(ten_x.shape)
    # CHECK: f64
    print(ten_x.dtype)
    # CHECK: True
    print(ten_x.is_constant())
    # CHECK: True
    print(np.allclose(ten_x.literal_value, x))

    y = np.random.random((10, 10))
    # CHECK: Tensor(%[[CST2:.*]], tensor<10x10xf64>, [
    ten_y = Tensor(y)
    print(ten_y)
    sum_ten_2 = ten_x + ten_y
    # CHECK: Tensor(%[[CST3:.*]], tensor<10x10xf64>, [
    print(sum_ten_2)
    # CHECK: (10, 10)
    print(sum_ten_2.shape)
    # CHECK: f64
    print(sum_ten_2.dtype)
    # CHECK: True
    print(sum_ten_2.is_constant())
    # CHECK: True
    print(np.allclose(sum_ten_2.literal_value, x + y))

    try:
      Tensor(arith.ConstantOp(i32, 0).result)
    except ValueError as e:
      # CHECK: Cannot cast value to TensorValue (from <mlir_structured._mlir_libs._mlir.ir.OpResult
      print(e)

  # CHECK: module {
  # CHECK:   %[[TEN]] = tensor.empty() : tensor<10x10xi32>
  # CHECK:   %[[ADD]] = arith.addi %[[TEN]], %[[TEN]] : tensor<10x10xi32>
  # CHECK:   %[[MUL]] = arith.muli %[[TEN]], %[[TEN]] : tensor<10x10xi32>
  # CHECK:   %[[CST1]] = arith.constant dense<{{.*}}> : tensor<10x10xf64>
  # CHECK:   %[[CST2]] = arith.constant dense<{{.*}}> : tensor<10x10xf64>
  # CHECK:   %[[CST3]] = arith.constant dense<{{.*}}> : tensor<10x10xf64>
  # CHECK: }
  print(module)

  pm = PassManager.parse('builtin.module(convert-elementwise-to-linalg)')
  pm.run(module.operation)

  # CHECK: #map = affine_map<(d0, d1) -> (d0, d1)>
  # CHECK: module {
  # CHECK:   %{{.*}} = tensor.empty()
  # CHECK:   %{{.*}} = linalg.generic
  # CHECK:   ^bb0(%{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32):
  # CHECK:     %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
  # CHECK:     linalg.yield %{{.*}} : i32
  # CHECK:   } -> tensor<10x10xi32>
  # CHECK:   %{{.*}} = linalg.generic
  # CHECK:   ^bb0(%{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32):
  # CHECK:     %{{.*}} = arith.muli %{{.*}}, %{{.*}} : i32
  # CHECK:     linalg.yield %{{.*}} : i32
  # CHECK:   } -> tensor<10x10xi32>
  # CHECK:   %[[CST1]] = arith.constant dense<{{.*}}> : tensor<10x10xf64>
  # CHECK:   %[[CST2]] = arith.constant dense<{{.*}}> : tensor<10x10xf64>
  # CHECK:   %[[CST3]] = arith.constant dense<{{.*}}> : tensor<10x10xf64>
  # CHECK: }
  print(module)


# CHECK-LABEL: TEST: testConcatenateOp
@run
def testConcatenateOp():
  i32 = IntegerType.get_signless(32)
  with mlir_mod_ctx() as module:
    ten = Tensor.empty((10, 10), i32)
    # CHECK: Tensor(%[[TEN:.*]], tensor<10x10xi32>)
    print(ten)

    concat_single_ten_first_dim = indexing.ConcatenateOp((ten,), 0).result
    # CHECK: %{{.}} = indexing.concatenate(%[[TEN]]) {dim = 0} : (tensor<10x10xi32>) -> tensor<10x10xi32>
    print(concat_single_ten_first_dim.owner)

    concat_ten_first_dim = indexing.ConcatenateOp((ten, ten), 0).result
    # CHECK: %{{.*}} = indexing.concatenate(%[[TEN]], %[[TEN]]) {dim = 0} : (tensor<10x10xi32>, tensor<10x10xi32>) -> tensor<20x10xi32>
    print(concat_ten_first_dim.owner)

    concat_ten_second_dim = indexing.ConcatenateOp((ten, ten), 1).result
    # CHECK: %{{.*}} = indexing.concatenate(%[[TEN]], %[[TEN]]) {dim = 1} : (tensor<10x10xi32>, tensor<10x10xi32>) -> tensor<10x20xi32>
    print(concat_ten_second_dim.owner)

    concat_ten_first_dim = indexing.concatenate((ten, ten), 0)
    # CHECK: %{{.*}} = indexing.concatenate(%[[TEN]], %[[TEN]]) {dim = 0} : (tensor<10x10xi32>, tensor<10x10xi32>) -> tensor<20x10xi32>
    print(concat_ten_first_dim.owner)

    concat_ten_second_dim = indexing.concatenate((ten, ten), 1)
    # CHECK: %{{.*}} = indexing.concatenate(%[[TEN]], %[[TEN]]) {dim = 1} : (tensor<10x10xi32>, tensor<10x10xi32>) -> tensor<10x20xi32>
    print(concat_ten_second_dim.owner)

    x = np.random.random((10, 10))
    ten_x = Tensor(x)
    concat_x = indexing.concatenate([ten_x, ten_x], 1)
    # CHECK: %{{.*}} = arith.constant dense<{{.*}}> : tensor<10x20xf64>
    print(concat_x.owner)
    # CHECK: True
    print(np.allclose(concat_x.literal_value, np.concatenate([x, x], axis=1)))


# CHECK-LABEL: TEST: testSimpleLiteralIndexing
@run
def testSimpleLiteralIndexing():
  i32 = IntegerType.get_signless(32)
  with mlir_mod_ctx() as module:

    ten = Tensor.empty((10, 22, 333, 4444), i32)
    # CHECK: %[[TEN:.*]]
    print(ten.get_name())

    w = ten[0]
    # CHECK: Tensor(%[[CST0:.*]], tensor<1x1xindex>, {{\[}}[0]])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([0]) unique : (tensor<10x22x333x4444xi32>, tensor<1x1xindex>) -> tensor<1x22x333x4444xi32>
    print(w.owner)

    w = ten[2, 4]
    # CHECK: Tensor(%[[CST0:.*]], tensor<1x2xindex>, {{\[}}[2 4]])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([0, 1]) unique : (tensor<10x22x333x4444xi32>, tensor<1x2xindex>) -> tensor<1x333x4444xi32>
    print(w.owner)

    w = ten[2, 4, 6]
    # CHECK: Tensor(%[[CST0:.*]], tensor<1x3xindex>, {{\[}}[2 4 6]])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([0, 1, 2]) unique : (tensor<10x22x333x4444xi32>, tensor<1x3xindex>) -> tensor<1x4444xi32>
    print(w.owner)

    w = ten[2, 4, 6, 8]
    # CHECK: Scalar(%[[CST1:.*]], index, 2)
    print(Scalar(w.owner.operands[1]))
    # CHECK: Scalar(%[[CST2:.*]], index, 4)
    print(Scalar(w.owner.operands[2]))
    # CHECK: Scalar(%[[CST3:.*]], index, 6)
    print(Scalar(w.owner.operands[3]))
    # CHECK: Scalar(%[[CST4:.*]], index, 8)
    print(Scalar(w.owner.operands[4]))
    # CHECK: %extracted = tensor.extract %[[TEN]][%[[CST1]], %[[CST2]], %[[CST3]], %[[CST4]]] : tensor<10x22x333x4444xi32>
    print(w.owner)

    w = ten[...]
    # CHECK: %[[TEN]]
    print(w.get_name())

    w = ten[:]
    # CHECK: %[[TEN]]
    print(w.get_name())

    w = ten[:, :]
    # CHECK: %[[TEN]]
    print(w.get_name())

    w = ten[:, :, :]
    # CHECK: %[[TEN]]
    print(w.get_name())

    w = ten[:, :, :, :]
    # CHECK: %[[TEN]]
    print(w.get_name())

    w = ten[1, ...]
    # CHECK: Tensor(%[[CST0:.*]], tensor<1x1xindex>, {{\[}}[1]])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([0]) unique : (tensor<10x22x333x4444xi32>, tensor<1x1xindex>) -> tensor<1x22x333x4444xi32>
    print(w.owner)

    w = ten[1, :, ...]
    # CHECK: Tensor(%[[CST0:.*]], tensor<1x1xindex>, {{\[}}[1]])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([0]) unique : (tensor<10x22x333x4444xi32>, tensor<1x1xindex>) -> tensor<1x22x333x4444xi32>
    print(w.owner)

    w = ten[1, :, :, ...]
    # CHECK: Tensor(%[[CST0:.*]], tensor<1x1xindex>, {{\[}}[1]])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([0]) unique : (tensor<10x22x333x4444xi32>, tensor<1x1xindex>) -> tensor<1x22x333x4444xi32>
    print(w.owner)

    try:
      w = ten[1, :, :, :, :]
    except IndexError as e:
      # CHECK: Too many indices for tensor: 5 non-None/Ellipsis indices for dim 4.
      print(e)

    w = ten[1, :]
    # CHECK: Tensor(%[[CST0:.*]], tensor<1x1xindex>, {{\[}}[1]])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([0]) unique : (tensor<10x22x333x4444xi32>, tensor<1x1xindex>) -> tensor<1x22x333x4444xi32>
    print(w.owner)

    w = ten[1, :, :]
    # CHECK: Tensor(%[[CST0:.*]], tensor<1x1xindex>, {{\[}}[1]])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([0]) unique : (tensor<10x22x333x4444xi32>, tensor<1x1xindex>) -> tensor<1x22x333x4444xi32>
    print(w.owner)

    w = ten[1, :, :, :]
    # CHECK: Tensor(%[[CST0:.*]], tensor<1x1xindex>, {{\[}}[1]])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([0]) unique : (tensor<10x22x333x4444xi32>, tensor<1x1xindex>) -> tensor<1x22x333x4444xi32>
    print(w.owner)

    w = ten[:, 1]
    # CHECK: Tensor(%[[CST0:.*]], tensor<1x1xindex>, {{\[}}[1]])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([1]) unique : (tensor<10x22x333x4444xi32>, tensor<1x1xindex>) -> tensor<1x10x333x4444xi32>
    print(w.owner)

    w = ten[:, :, 1]
    # CHECK: Tensor(%[[CST0:.*]], tensor<1x1xindex>, {{\[}}[1]])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([2]) unique : (tensor<10x22x333x4444xi32>, tensor<1x1xindex>) -> tensor<1x10x22x4444xi32>
    print(w.owner)

    w = ten[:, :, :, 1]
    # CHECK: Tensor(%[[CST0:.*]], tensor<1x1xindex>, {{\[}}[1]])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([3]) unique : (tensor<10x22x333x4444xi32>, tensor<1x1xindex>) -> tensor<1x10x22x333xi32>
    print(w.owner)

    w = ten[:, 1, :, 1]
    # CHECK: Tensor(%[[CST0:.*]], tensor<1x2xindex>, {{\[}}[1 1]])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([1, 3]) unique : (tensor<10x22x333x4444xi32>, tensor<1x2xindex>) -> tensor<1x10x333xi32>
    print(w.owner)

    w = ten[1, :, :, 1]
    # CHECK: Tensor(%[[CST0:.*]], tensor<1x2xindex>, {{\[}}[1 1]])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([0, 3]) unique : (tensor<10x22x333x4444xi32>, tensor<1x2xindex>) -> tensor<1x22x333xi32>
    print(w.owner)

    w = ten[1, 1, :, :]
    # CHECK: Tensor(%[[CST0:.*]], tensor<1x2xindex>, {{\[}}[1 1]])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([0, 1]) unique : (tensor<10x22x333x4444xi32>, tensor<1x2xindex>) -> tensor<1x333x4444xi32>
    print(w.owner)

    w = ten[:, :, 1, 1]
    # CHECK: Tensor(%[[CST0:.*]], tensor<1x2xindex>, {{\[}}[1 1]])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([2, 3]) unique : (tensor<10x22x333x4444xi32>, tensor<1x2xindex>) -> tensor<1x10x22xi32>
    print(w.owner)

    w = ten[:, 1, 1, :]
    # CHECK: Tensor(%[[CST0:.*]], tensor<1x2xindex>, {{\[}}[1 1]])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([1, 2]) unique : (tensor<10x22x333x4444xi32>, tensor<1x2xindex>) -> tensor<1x10x4444xi32>
    print(w.owner)

    w = ten[1, :, 1, :]
    # CHECK: Tensor(%[[CST0:.*]], tensor<1x2xindex>, {{\[}}[1 1]])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([0, 2]) unique : (tensor<10x22x333x4444xi32>, tensor<1x2xindex>) -> tensor<1x22x4444xi32>
    print(w.owner)

    w = ten[1, 1, :, 1]
    # CHECK: Tensor(%[[CST0:.*]], tensor<1x3xindex>, {{\[}}[1 1 1]])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([0, 1, 3]) unique : (tensor<10x22x333x4444xi32>, tensor<1x3xindex>) -> tensor<1x333xi32>
    print(w.owner)

    w = ten[1, :, 1, 1]
    # CHECK: Tensor(%[[CST0:.*]], tensor<1x3xindex>, {{\[}}[1 1 1]])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([0, 2, 3]) unique : (tensor<10x22x333x4444xi32>, tensor<1x3xindex>) -> tensor<1x22xi32>
    print(w.owner)

    w = ten[:, 1, 1, 1]
    # CHECK: Tensor(%[[CST0:.*]], tensor<1x3xindex>, {{\[}}[1 1 1]])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([1, 2, 3]) unique : (tensor<10x22x333x4444xi32>, tensor<1x3xindex>) -> tensor<1x10xi32>
    print(w.owner)

    w = ten[1, 1, 1, :]
    # CHECK: Tensor(%[[CST0:.*]], tensor<1x3xindex>, {{\[}}[1 1 1]])
    print(Tensor(w.owner.operands[1]))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[CST0]]] gather_dims([0, 1, 2]) unique : (tensor<10x22x333x4444xi32>, tensor<1x3xindex>) -> tensor<1x4444xi32>
    print(w.owner)


# CHECK-LABEL: TEST: testCanonicalizeTupleIndexCastListLiteral
# This test generates all permutations of idx and slice object, e.g.
# ten[idx, :, :, :], ten[:, idx, :, :], ten[:, :, idx, :]
@run
def testCanonicalizeTupleIndexCastListLiteral():
  with mlir_mod_ctx() as module:

    for n_tens in range(1, 4):
      uniqs = set()
      n_slices = 4 - n_tens
      ten_idx = [[0], [1]]
      slice_idx = slice(None)
      for p in permutations([str(ten_idx)] * n_tens +
                            [str(slice_idx)] * n_slices):
        uniqs.add(p)

      for u in uniqs:
        u = tuple(u)
        tens_is = [i for i, t in enumerate(u) if t == str(ten_idx)]
        slice_is = [i for i, s in enumerate(u) if s == str(slice_idx)]

        tens_slices = _canonicalize_tuple_index(tuple(map(eval, u)), 4)
        tens = [
            (i, t) for i, t in enumerate(tens_slices) if isinstance(t, Tensor)
        ]
        slices = [(i, s) for i, s in enumerate(tens_slices) if s == slice(None)]
        assert len(slices) == n_slices and all(
            s == slice(None) for _, s in slices) and set(
                i for i, _ in slices) == set(slice_is)
        assert len(tens) == n_tens and all(
            isinstance(t, Tensor) and t.owner.name == 'arith.constant' and
            str(t.type) == 'tensor<2x1xindex>' and t.is_constant() and
            np.array_equal(t.literal_value, [[0], [1]])
            for _, t in tens) and set(i for i, _ in tens) == set(tens_is)


# CHECK-LABEL: TEST: testAdvancedIndexing
@run
def testAdvancedIndexing():
  index = IndexType.get()
  f32 = F32Type.get()
  with mlir_mod_ctx() as module:
    ten = Tensor.empty((7, 22, 333, 4444), f32)
    # CHECK: Tensor(%[[TEN:.*]], tensor<7x22x333x4444xf32>)
    print(ten)

    w = ten[[[0], [1]], :, :, :]
    idx_tensor_operand = Tensor(w.owner.operands[1])
    # CHECK: %[[IDXTEN:.*]] tensor<2x1xindex> True
    print(idx_tensor_operand.get_name(), idx_tensor_operand.type,
          idx_tensor_operand.is_constant())
    # CHECK{LITERAL}: [[0], [1]]
    print(get_array_on_one_line(idx_tensor_operand.literal_value))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[IDXTEN]]] gather_dims([0]) unique : (tensor<7x22x333x4444xf32>, tensor<2x1xindex>) -> tensor<2x22x333x4444xf32>
    print(w.owner)

    w = ten[[[0], [1]], [[0], [1]], :, :]
    idx_tensor_operand = Tensor(w.owner.operands[1])
    # CHECK: %[[IDXTEN:.*]] tensor<2x2xindex> True
    print(idx_tensor_operand.get_name(), idx_tensor_operand.type,
          idx_tensor_operand.is_constant())
    # CHECK{LITERAL}: [[0 0], [1 1]]
    print(get_array_on_one_line(idx_tensor_operand.literal_value))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[IDXTEN]]] gather_dims([0, 1]) unique : (tensor<7x22x333x4444xf32>, tensor<2x2xindex>) -> tensor<2x333x4444xf32>
    print(w.owner)

    w = ten[[[0], [1]], :, [[0], [1]], :]
    idx_tensor_operand = Tensor(w.owner.operands[1])
    # CHECK: %[[IDXTEN:.*]] tensor<2x2xindex> True
    print(idx_tensor_operand.get_name(), idx_tensor_operand.type,
          idx_tensor_operand.is_constant())
    # CHECK{LITERAL}: [[0 0], [1 1]]
    print(get_array_on_one_line(idx_tensor_operand.literal_value))
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[IDXTEN]]] gather_dims([0, 2]) unique : (tensor<7x22x333x4444xf32>, tensor<2x2xindex>) -> tensor<2x22x4444xf32>
    print(w.owner)

    idx_tensor = np.array([[[0], [5], [1], [8]], [[0], [3], [6], [4]],
                           [[8], [7], [9], [1]]])
    idx_tensor = Tensor(idx_tensor, dtype=index)
    # CHECK: %[[IDXTEN:.*]] tensor<3x4x1xindex> True
    print(idx_tensor.get_name(), idx_tensor.type, idx_tensor.is_constant())
    # CHECK: %[[IDXTEN:.*]] = arith.constant dense<{{.*}}> : tensor<3x4x1xindex>
    print(idx_tensor.owner)
    # CHECK: True
    print(np.array(idx_tensor.literal_value).dtype == np.int64)

    w = ten[idx_tensor, ...]
    idx_tensor_operand = Tensor(w.owner.operands[1])
    # CHECK: %[[IDXTEN:.*]] tensor<3x4x1xindex> True
    print(idx_tensor_operand.get_name(), idx_tensor_operand.type,
          idx_tensor_operand.is_constant())
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[IDXTEN]]] gather_dims([0]) : (tensor<7x22x333x4444xf32>, tensor<3x4x1xindex>) -> tensor<3x4x22x333x4444xf32>
    print(w.owner)

    w = ten[idx_tensor, idx_tensor, ...]
    idx_tensor_operand = Tensor(w.owner.operands[1])
    # CHECK: %[[IDXTEN:.*]] tensor<3x4x2xindex> True
    print(idx_tensor_operand.get_name(), idx_tensor_operand.type,
          idx_tensor_operand.is_constant())
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[IDXTEN]]] gather_dims([0, 1]) : (tensor<7x22x333x4444xf32>, tensor<3x4x2xindex>) -> tensor<3x4x333x4444xf32>
    print(w.owner)

    w = ten[idx_tensor, :, idx_tensor]
    idx_tensor_operand = Tensor(w.owner.operands[1])
    # CHECK: %[[IDXTEN:.*]] tensor<3x4x2xindex> True
    print(idx_tensor_operand.get_name(), idx_tensor_operand.type,
          idx_tensor_operand.is_constant())
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[IDXTEN]]] gather_dims([0, 2]) : (tensor<7x22x333x4444xf32>, tensor<3x4x2xindex>) -> tensor<3x4x22x4444xf32>
    print(w.owner)

    w = ten[idx_tensor, :, idx_tensor, idx_tensor]
    idx_tensor_operand = Tensor(w.owner.operands[1])
    # CHECK: %[[IDXTEN:.*]] tensor<3x4x3xindex> True
    print(idx_tensor_operand.get_name(), idx_tensor_operand.type,
          idx_tensor_operand.is_constant())
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[IDXTEN]]] gather_dims([0, 2, 3]) : (tensor<7x22x333x4444xf32>, tensor<3x4x3xindex>) -> tensor<3x4x22xf32>
    print(w.owner)

    idx_tensor = np.array([[[1, 4], [2, 8], [8, 0], [0, 3]],
                           [[9, 5], [8, 6], [6, 5], [7, 1]],
                           [[9, 3], [3, 1], [6, 7], [0, 0]]])
    idx_tensor = Tensor(idx_tensor, dtype=index)
    # CHECK: %[[IDXTEN:.*]] = arith.constant dense<{{.*}}> : tensor<3x4x2xindex>
    print(idx_tensor.owner)

    w = indexing.gather(ten, idx_tensor, [0, 1])
    idx_tensor_operand = Tensor(w.owner.operands[1])
    # CHECK: %[[IDXTEN]] tensor<3x4x2xindex> True
    print(idx_tensor_operand.get_name(), idx_tensor_operand.type,
          idx_tensor_operand.is_constant())
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[IDXTEN]]] gather_dims([0, 1]) : (tensor<7x22x333x4444xf32>, tensor<3x4x2xindex>) -> tensor<3x4x333x4444xf32>
    print(w.owner)

    w = indexing.gather(ten, idx_tensor, [0, 2])
    idx_tensor_operand = Tensor(w.owner.operands[1])
    # CHECK: %[[IDXTEN]] tensor<3x4x2xindex> True
    print(idx_tensor_operand.get_name(), idx_tensor_operand.type,
          idx_tensor_operand.is_constant())
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[IDXTEN]]] gather_dims([0, 2]) : (tensor<7x22x333x4444xf32>, tensor<3x4x2xindex>) -> tensor<3x4x22x4444xf32>
    print(w.owner)

    w = ten[idx_tensor, ...]
    idx_tensor_operand = Tensor(w.owner.operands[1])
    # CHECK: %[[IDXTEN:.*]] tensor<3x4x2xindex> True
    print(idx_tensor_operand.get_name(), idx_tensor_operand.type,
          idx_tensor_operand.is_constant())
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[IDXTEN]]] gather_dims([0, 1]) unique : (tensor<7x22x333x4444xf32>, tensor<3x4x2xindex>) -> tensor<3x4x333x4444xf32>
    print(w.owner)

    w = ten[:, idx_tensor, ...]
    idx_tensor_operand = Tensor(w.owner.operands[1])
    # CHECK: %[[IDXTEN:.*]] tensor<3x4x2xindex> True
    print(idx_tensor_operand.get_name(), idx_tensor_operand.type,
          idx_tensor_operand.is_constant())
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[IDXTEN]]] gather_dims([1, 2]) unique : (tensor<7x22x333x4444xf32>, tensor<3x4x2xindex>) -> tensor<3x4x7x4444xf32>
    print(w.owner)

    ten = Tensor.empty((7, 22, 333, 4444, 55555), f32)
    # CHECK: Tensor(%[[TEN:.*]], tensor<7x22x333x4444x55555xf32>)
    print(ten)

    w = ten[idx_tensor, :, idx_tensor, ...]
    idx_tensor_operand = Tensor(w.owner.operands[1])
    # CHECK: %[[IDXTEN:.*]] tensor<3x4x4xindex> True
    print(idx_tensor_operand.get_name(), idx_tensor_operand.type,
          idx_tensor_operand.is_constant())
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[IDXTEN]]] gather_dims([0, 1, 3, 4]) unique : (tensor<7x22x333x4444x55555xf32>, tensor<3x4x4xindex>) -> tensor<3x4x333xf32>
    print(w.owner)

    w = ten[idx_tensor, 0:333:1, idx_tensor, ...]
    idx_tensor_operand = Tensor(w.owner.operands[1])
    # CHECK: %[[IDXTEN:.*]] tensor<3x4x4xindex> True
    print(idx_tensor_operand.get_name(), idx_tensor_operand.type,
          idx_tensor_operand.is_constant())
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[IDXTEN]]] gather_dims([0, 1, 3, 4]) unique : (tensor<7x22x333x4444x55555xf32>, tensor<3x4x4xindex>) -> tensor<3x4x333xf32>
    print(w.owner)


# CHECK-LABEL: TEST: testARangeOpBasics
@run
def testARangeOpBasics():
  index = IndexType.get()
  with mlir_mod_ctx() as module:
    start = Scalar(0, dtype=index)
    # CHECK: %[[START:.*]]
    print(start.get_name())

    stop = Scalar(100, dtype=index)
    # CHECK: %[[STOP:.*]]
    print(stop.get_name())

    step = Scalar(2, dtype=index)
    # CHECK: %[[STEP:.*]]
    print(step.get_name())

    ara = Tensor(indexing.ARangeOp(start=start, stop=stop, step=step))
    # CHECK: %{{.*}} = indexing.arange(start = %[[START]], stop = %[[STOP]], step = %[[STEP]]) : tensor<?x1xindex>
    print(ara.owner)

    ara = Tensor(indexing.ARangeOp(start=0, stop=stop, step=step))
    # CHECK: %{{.*}} = indexing.arange(start = 0, stop = %[[STOP]], step = %[[STEP]]) : tensor<?x1xindex>
    print(ara.owner)

    ara = Tensor(indexing.ARangeOp(start=start, stop=100, step=step))
    # CHECK: %{{.*}} = indexing.arange(start = %[[START]], stop = 100, step = %[[STEP]]) : tensor<?x1xindex>
    print(ara.owner)

    ara = Tensor(indexing.ARangeOp(start=start, stop=stop, step=2))
    # CHECK: %{{.*}} = indexing.arange(start = %[[START]], stop = %[[STOP]], step = 2) : tensor<?x1xindex>
    print(ara.owner)

    ara = Tensor(indexing.ARangeOp(start=0, stop=100, step=step))
    # CHECK: %{{.*}} = indexing.arange(start = 0, stop = 100, step = %[[STEP]]) : tensor<?x1xindex>
    print(ara.owner)

    ara = Tensor(indexing.ARangeOp(start=0, stop=stop, step=2))
    # CHECK: %{{.*}} = indexing.arange(start = 0, stop = %[[STOP]], step = 2) : tensor<?x1xindex>
    print(ara.owner)

    ara = Tensor(indexing.ARangeOp(start=start, stop=100, step=2))
    # CHECK: %{{.*}} = indexing.arange(start = %[[START]], stop = 100, step = 2) : tensor<?x1xindex>
    print(ara.owner)

    ara = Tensor(indexing.ARangeOp(start=0, stop=100, step=2))
    # CHECK: %{{.*}} = indexing.arange(start = 0, stop = 100, step = 2) : tensor<50x1xindex>
    print(ara.owner)

    ara = Tensor(indexing.ARangeOp(start=0, stop=100, step=2, fold=True))
    # CHECK: %{{.*}} = indexing.arange(start = 0, stop = 100, step = 2, fold = true) : tensor<50x1xindex>
    print(ara.owner)

  module.operation.verify()


# CHECK-LABEL: TEST: testARangeFun
@run
def testARangeFun():
  with mlir_mod_ctx() as module:
    ara = arange(0, 100, 2, fold=False)
    # CHECK: Value(%[[C0:.*]] = arith.constant 0 : index)
    print(ara.owner.operands[0])
    # CHECK: Value(%[[C100:.*]] = arith.constant 100 : index)
    print(ara.owner.operands[1])
    # CHECK: Value(%[[C2:.*]] = arith.constant 2 : index)
    print(ara.owner.operands[2])

    # CHECK: %{{.*}} = indexing.arange(start = %[[C0]], stop = %[[C100]], step = %[[C2]]) : tensor<?x1xindex>
    print(ara.owner)

    ara = arange(0, 100, fold=False)
    # CHECK: Value(%[[C0:.*]] = arith.constant 0 : index)
    print(ara.owner.operands[0])
    # CHECK: Value(%[[C100:.*]] = arith.constant 100 : index)
    print(ara.owner.operands[1])
    # CHECK: %{{.*}} = indexing.arange(start = %[[C0]], stop = %[[C100]], step = 1) : tensor<?x1xindex>
    print(ara.owner)

    ara = arange(100, fold=False)
    # CHECK: Value(%[[C100:.*]] = arith.constant 100 : index)
    print(ara.owner.operands[0])
    # CHECK: %{{.*}} = indexing.arange(start = 0, stop = %[[C100]], step = 1) : tensor<?x1xindex>
    print(ara.owner)

    ara = arange(0, 100, 2, fold=True)
    # CHECK: %{{.*}} = arith.constant dense<{{\[}}[0], [2], [4], [6], [8], {{.*}}, [98]]> : tensor<50x1xindex>
    print(ara.owner)

    ara = arange(0, 100, fold=True)
    # CHECK: %{{.*}} = arith.constant dense<{{\[}}[0], [1], [2], [3], [4], {{.*}}, [99]]> : tensor<100x1xindex>
    print(ara.owner)

    ara = arange(100, fold=True)
    # CHECK: %{{.*}} = arith.constant dense<{{\[}}[0], [1], [2], [3], [4], {{.*}}, [99]]> : tensor<100x1xindex>
    print(ara.owner)

  module.operation.verify()


# CHECK-LABEL: TEST: testARangeOpSemantics
# This test tests that the inferReturnTypes computes the right length
# by comparing it with the length arange produced by numpy; the formula is
# len = ((stop - start) // step) + 1
#       if (stop - start) % step != 0
#       else (stop - start) // step
@run
def testARangeOpSemantics():
  index = IndexType.get()
  with mlir_mod_ctx() as module:
    start = Scalar(0, dtype=index)
    # CHECK: %[[START:.*]]
    print(start.get_name())
    stop = Scalar(100, dtype=index)
    # CHECK: %[[STOP:.*]]
    print(stop.get_name())
    step = Scalar(2, dtype=index)
    # CHECK: %[[STEP:.*]]
    print(step.get_name())

    for _ in range(1000):
      start = np.random.randint(0, 500)
      stop = np.random.randint(500, 1000)
      step = np.random.randint(1, 100)

      ara = Tensor(indexing.ARangeOp(start=start, stop=stop, step=step))
      r = np.arange(start, stop, step)[:, np.newaxis]

      if len(r) != (stop - start) // step + 1:
        assert (stop - start) % step == 0
        assert len(r) == (stop - start) // step

      assert r.shape == ara.shape


# CHECK-LABEL: TEST: testNoneIndices
@run
def testNoneIndices():
  index = IndexType.get()
  f32 = F32Type.get()
  with mlir_mod_ctx() as module:
    ten = Tensor.empty((7, 22, 333, 4444), f32)
    # CHECK: Tensor(%[[TEN:.*]], tensor<7x22x333x4444xf32>)
    print(ten)

    w = ten[None]
    # CHECK: %{{.*}} = tensor.expand_shape %[[TEN]] {{\[}}[0, 1], [2], [3], [4]] : tensor<7x22x333x4444xf32> into tensor<1x7x22x333x4444xf32>
    print(w.owner)

    w = ten[:, None]
    # CHECK: %{{.*}} = tensor.expand_shape %[[TEN]] {{\[}}[0, 1], [2], [3], [4]] : tensor<7x22x333x4444xf32> into tensor<7x1x22x333x4444xf32>
    print(w.owner)

    w = ten[None, None]
    # CHECK: %{{.*}} = tensor.expand_shape %[[TEN]] {{\[}}[0, 1, 2], [3], [4], [5]] : tensor<7x22x333x4444xf32> into tensor<1x7x1x22x333x4444xf32>
    print(w.owner)

    w = ten[:, :, None]
    # CHECK: %{{.*}} = tensor.expand_shape %[[TEN]] {{\[}}[0], [1, 2], [3], [4]] : tensor<7x22x333x4444xf32> into tensor<7x22x1x333x4444xf32>
    print(w.owner)

    w = ten[:, :, :, None]
    # CHECK: %{{.*}} = tensor.expand_shape %[[TEN]] {{\[}}[0], [1], [2, 3], [4]] : tensor<7x22x333x4444xf32> into tensor<7x22x333x1x4444xf32>
    print(w.owner)

    w = ten[:, :, :, :, None]
    # CHECK: %{{.*}} = tensor.expand_shape %[[TEN]] {{\[}}[0], [1], [2], [3, 4]] : tensor<7x22x333x4444xf32> into tensor<7x22x333x4444x1xf32>
    print(w.owner)

    w = ten[..., None]
    # CHECK: %{{.*}} = tensor.expand_shape %[[TEN]] {{\[}}[0], [1], [2], [3, 4]] : tensor<7x22x333x4444xf32> into tensor<7x22x333x4444x1xf32>
    print(w.owner)

    w = ten[:, None, :, :, None]
    # CHECK: %{{.*}} = tensor.expand_shape %[[TEN]] {{\[}}[0, 1], [2], [3], [4, 5]] : tensor<7x22x333x4444xf32> into tensor<7x1x22x333x4444x1xf32>
    print(w.owner)

    w = ten[:, None, None, :, None]
    # CHECK: %{{.*}} = tensor.expand_shape %[[TEN]] {{\[}}[0, 1], [2, 3], [4], [5, 6]] : tensor<7x22x333x4444xf32> into tensor<7x1x22x1x333x4444x1xf32>
    print(w.owner)

    w = ten[:, None, None, None, None]
    # CHECK: %{{.*}} = tensor.expand_shape %[[TEN]] {{\[}}[0, 1], [2, 3], [4, 5], [6, 7]] : tensor<7x22x333x4444xf32> into tensor<7x1x22x1x333x1x4444x1xf32>
    print(w.owner)

    w = ten[None, None, None, None, None]
    # CHECK: %{{.*}} = tensor.expand_shape %[[TEN]] {{\[}}[0, 1, 2], [3, 4], [5, 6], [7, 8]] : tensor<7x22x333x4444xf32> into tensor<1x7x1x22x1x333x1x4444x1xf32>
    print(w.owner)

    try:
      w = ten[None, None, None, None, None, None]
      print(w.owner)
    except IndexError as e:
      # CHECK: pop index out of range
      print(e)

  module.operation.verify()


# CHECK-LABEL: TEST: testArithPythonValues
@run
def testArithPythonValues():
  index = IndexType.get()
  f32 = F32Type.get()
  f64 = F64Type.get()
  with mlir_mod_ctx() as module:

    one = Scalar(1.0, dtype=f64, fold=False)
    # CHECK: %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f64
    print(one.owner)

    start = one * 100.0
    # CHECK: %[[VAL_1:.*]] = arith.constant 1.000000e+02 : f64
    print(start.owner.operands[1].owner)
    # CHECK: %[[VAL_2:.*]] = arith.mulf %[[VAL_0]], %[[VAL_1]] : f64
    print(start.owner)

    start = 100.0 * one
    # CHECK: %[[VAL_3:.*]] = arith.constant 1.000000e+02 : f64
    print(start.owner.operands[1].owner)
    # CHECK: %[[VAL_4:.*]] = arith.mulf %[[VAL_0]], %[[VAL_3]] : f64
    print(start.owner)

    start = one + 100.0
    # CHECK: %[[VAL_5:.*]] = arith.constant 1.000000e+02 : f64
    print(start.owner.operands[1].owner)
    # CHECK: %[[VAL_6:.*]] = arith.addf %[[VAL_0]], %[[VAL_5]] : f64
    print(start.owner)

    start = 100.0 + one
    # CHECK: %[[VAL_7:.*]] = arith.constant 1.000000e+02 : f64
    print(start.owner.operands[1].owner)
    # CHECK: %[[VAL_8:.*]] = arith.addf %[[VAL_0]], %[[VAL_7]] : f64
    print(start.owner)

    start = one - 100.0
    # CHECK: %[[VAL_9:.*]] = arith.constant 1.000000e+02 : f64
    print(start.owner.operands[1].owner)
    # CHECK: %[[VAL_10:.*]] = arith.subf %[[VAL_0]], %[[VAL_9]] : f64
    print(start.owner)

    start = 100.0 - one
    # CHECK: %[[VAL_11:.*]] = arith.constant 1.000000e+02 : f64
    print(start.owner.operands[1].owner)
    # CHECK: %[[VAL_12:.*]] = arith.subf %[[VAL_0]], %[[VAL_11]] : f64
    print(start.owner)

    ten = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), dtype=f64, fold=False)
    # CHECK: %[[VAL_13:.*]] = arith.constant dense<{{\[\[}}1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>
    print(ten.owner)

    two_times_ten = ten * (2.0 * np.ones((2, 2)))
    # CHECK: %[[VAL_14:.*]] = arith.constant dense<2.000000e+00> : tensor<2x2xf64>
    print(two_times_ten.owner.operands[1].owner)
    # CHECK: %[[VAL_15:.*]] = arith.mulf %[[VAL_13]], %[[VAL_14]] : tensor<2x2xf64>
    print(two_times_ten.owner)

    two_times_ten = ten * 2.0
    # CHECK: %[[VAL_16:.*]] = arith.constant dense<1.000000e+00> : tensor<2x2xf64>
    print(two_times_ten.owner.operands[1].owner)
    # CHECK: %[[VAL_17:.*]] = arith.mulf %[[VAL_13]], %[[VAL_16]] : tensor<2x2xf64>
    print(two_times_ten.owner)

    two_times_ten = 2.0 * ten
    # CHECK: %[[VAL_18:.*]] = arith.constant dense<1.000000e+00> : tensor<2x2xf64>
    print(two_times_ten.owner.operands[1].owner)
    # CHECK: %[[VAL_19:.*]] = arith.mulf %[[VAL_13]], %[[VAL_18]] : tensor<2x2xf64>
    print(two_times_ten.owner)

    two_times_ten = ten + 2.0
    # CHECK: %[[VAL_20:.*]] = arith.constant dense<1.000000e+00> : tensor<2x2xf64>
    print(two_times_ten.owner.operands[1].owner)
    # CHECK: %[[VAL_21:.*]] = arith.addf %[[VAL_13]], %[[VAL_20]] : tensor<2x2xf64>
    print(two_times_ten.owner)

    two_times_ten = 2.0 + ten
    # CHECK: %[[VAL_22:.*]] = arith.constant dense<1.000000e+00> : tensor<2x2xf64>
    print(two_times_ten.owner.operands[1].owner)
    # CHECK: %[[VAL_23:.*]] = arith.addf %[[VAL_13]], %[[VAL_22]] : tensor<2x2xf64>
    print(two_times_ten.owner)

    two_times_ten = ten - 2.0
    # CHECK: %[[VAL_24:.*]] = arith.constant dense<1.000000e+00> : tensor<2x2xf64>
    print(two_times_ten.owner.operands[1].owner)
    # CHECK: %[[VAL_25:.*]] = arith.subf %[[VAL_13]], %[[VAL_24]] : tensor<2x2xf64>
    print(two_times_ten.owner)

    two_times_ten = 2.0 - ten
    # CHECK: %[[VAL_26:.*]] = arith.constant dense<1.000000e+00> : tensor<2x2xf64>
    print(two_times_ten.owner.operands[1].owner)
    # CHECK: %[[VAL_27:.*]] = arith.subf %[[VAL_13]], %[[VAL_26]] : tensor<2x2xf64>
    print(two_times_ten.owner)

  module.operation.verify()


# CHECK-LABEL: TEST: testArbitrarySlicingLiterals
@run
def testArbitrarySlicingLiterals():
  index = IndexType.get()
  f32 = F32Type.get()
  with mlir_mod_ctx() as module:
    ten = Tensor.empty((7, 22, 330, 4400), f32)
    # CHECK: Tensor(%[[TEN:.*]], tensor<7x22x330x4400xf32>)
    print(ten)

    w = ten[:, arange(start=0, stop=22, step=2, fold=True)]
    # CHECK: %[[ARA:.*]] = arith.constant dense<{{\[}}[0], [2], [4], [6], [8], [10], [12], [14], [16], [18], [20]]> : tensor<11x1xindex>
    print(w.owner.operands[1].owner)
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[ARA]]] gather_dims([1]) unique : (tensor<7x22x330x4400xf32>, tensor<11x1xindex>) -> tensor<11x7x330x4400xf32>
    print(w.owner)

    w = ten[:,
            arange(start=0, stop=22, step=2, fold=True),
            arange(start=0, stop=330, step=30, fold=True)]
    # CHECK: %[[ARA:.*]] = arith.constant dense<{{\[}}[0, 0], [2, 30], [4, 60], [6, 90], [8, 120], [10, 150], [12, 180], [14, 210], [16, 240], [18, 270], [20, 300]]> : tensor<11x2xindex>
    print(w.owner.operands[1].owner)
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[ARA]]] gather_dims([1, 2]) unique : (tensor<7x22x330x4400xf32>, tensor<11x2xindex>) -> tensor<11x7x4400xf32>
    print(w.owner)

    w = ten[:,
            arange(start=0, stop=22, step=2, fold=True),
            arange(start=0, stop=330, step=30, fold=True),
            arange(start=0, stop=4400, step=400, fold=True)]
    # CHECK: %[[ARA:.*]] = arith.constant dense<{{\[}}[0, 0, 0], [2, 30, 400], [4, 60, 800], [6, 90, 1200], [8, 120, 1600], [10, 150, 2000], [12, 180, 2400], [14, 210, 2800], [16, 240, 3200], [18, 270, 3600], [20, 300, 4000]]> : tensor<11x3xindex>
    print(w.owner.operands[1].owner)
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[ARA]]] gather_dims([1, 2, 3]) unique : (tensor<7x22x330x4400xf32>, tensor<11x3xindex>) -> tensor<11x7xf32>
    print(w.owner)

    w = ten[:, :,
            arange(start=100, stop=200, step=5, fold=True),
            arange(start=1000, stop=2000, step=50, fold=True)]
    # CHECK: %[[ARA:.*]] = arith.constant dense<{{\[}}[100, 1000], [105, 1050], [110, 1100], [115, 1150], [120, 1200], [125, 1250], [130, 1300], [135, 1350], [140, 1400], [145, 1450], [150, 1500], [155, 1550], [160, 1600], [165, 1650], [170, 1700], [175, 1750], [180, 1800], [185, 1850], [190, 1900], [195, 1950]]> : tensor<20x2xindex>
    print(w.owner.operands[1].owner)
    # CHECK: %{{.*}} = indexing.gather %[[TEN]][%[[ARA]]] gather_dims([2, 3]) unique : (tensor<7x22x330x4400xf32>, tensor<20x2xindex>) -> tensor<20x7x22xf32>
    print(w.owner)

  module.operation.verify()


# CHECK-LABEL: TEST: testArithFolding
@run
def testArithFolding():
  index = IndexType.get()
  f32 = F32Type.get()
  with mlir_mod_ctx() as module:

    @func.FuncOp.from_py_func(*[])
    def test_fold():
      ten = Tensor.empty((7, 22, 330, 4400), f32)

      idx1 = arange(100, 200, 5, fold=True)
      idx2 = arange(1000, 2000, 50, fold=True)
      w = ten[:, :, idx1, idx2]
      idx_tensor = Tensor(w.owner.operands[1], fold=False)
      pid = 42
      pid_tensor = Tensor(pid * np.ones(idx_tensor.shape, dtype=np.intp),
                          dtype=index,
                          fold=False)
      new_idx_tensor = pid_tensor + idx_tensor
      w = ten[:, :, new_idx_tensor]
      return w

  module.operation.verify()

  pm = PassManager.parse('builtin.module(canonicalize)')
  pm.run(module.operation)
  # CHECK: module {
  # CHECK:   func.func @test_fold() -> tensor<20x7x22xf32> {
  # CHECK:     %[[ARA:.*]] = arith.constant dense<{{\[}}[142, 1042], [147, 1092], [152, 1142], [157, 1192], [162, 1242], [167, 1292], [172, 1342], [177, 1392], [182, 1442], [187, 1492], [192, 1542], [197, 1592], [202, 1642], [207, 1692], [212, 1742], [217, 1792], [222, 1842], [227, 1892], [232, 1942], [237, 1992]]> : tensor<20x2xindex>
  # CHECK:     %[[TEN:.*]] = tensor.empty() : tensor<7x22x330x4400xf32>
  # CHECK:     %[[GATHERED:.*]] = indexing.gather %[[TEN]][%[[ARA]]] gather_dims([2, 3]) : (tensor<7x22x330x4400xf32>, tensor<20x2xindex>) -> tensor<20x7x22xf32>
  # CHECK:     return %[[GATHERED]] : tensor<20x7x22xf32>
  # CHECK:   }
  # CHECK: }
  print(module)


# CHECK-LABEL: TEST: testArbitrarySlicingDyn
@run
def testArbitrarySlicingDyn():
  index = IndexType.get()
  f32 = F32Type.get()
  with mlir_mod_ctx() as module:

    @func.FuncOp.from_py_func(*[])
    def test_dyn_indices():
      ten = Tensor.empty((7, 22, 330, 4400), f32)
      # CHECK: %[[VAL_0:.*]] = "tensor.empty"() : () -> tensor<7x22x330x4400xf32>
      print(ten.owner)

      one = Scalar(1, dtype=index, fold=False)
      # CHECK: %[[VAL_1:.*]] = "arith.constant"() <{value = 1 : index}> : () -> index
      print(one.owner)

      start = 100 * one
      # CHECK: %[[VAL_2:.*]] = "arith.constant"() <{value = 100 : index}> : () -> index
      print(start.owner.operands[1].owner)
      # CHECK: %[[VAL_3:.*]] = "arith.muli"(%[[VAL_1]], %[[VAL_2]]) : (index, index) -> index
      print(start.owner)

      stop = 200 * one
      # CHECK: %[[VAL_4:.*]] = "arith.constant"() <{value = 200 : index}> : () -> index
      print(stop.owner.operands[1].owner)
      # CHECK: %[[VAL_5:.*]] = "arith.muli"(%[[VAL_1]], %[[VAL_4]]) : (index, index) -> index
      print(stop.owner)

      step = 5 * one
      # CHECK: %[[VAL_6:.*]] = "arith.constant"() <{value = 5 : index}> : () -> index
      print(step.owner.operands[1].owner)
      # CHECK: %[[VAL_7:.*]] = "arith.muli"(%[[VAL_1]], %[[VAL_6]]) : (index, index) -> index
      print(step.owner)

      w = ten[:, :, start:stop:step]
      # CHECK: %[[VAL_8:.*]] = "indexing.arange"(%[[VAL_3]], %[[VAL_5]], %[[VAL_7]]) {foldAttr = false, operand_segment_sizes = array<i32: 1, 1, 1>} : (index, index, index) -> tensor<?x1xindex>
      print(w.owner.operands[1].owner)
      # CHECK: %[[VAL_9:.*]] = "indexing.gather"(%[[VAL_0]], %[[VAL_8]]) {gather_dims = array<i64: 2>, unique} : (tensor<7x22x330x4400xf32>, tensor<?x1xindex>) -> tensor<?x7x22x4400xf32>
      print(w.owner)

      w = ten[:, :, start:stop:5]
      # CHECK: %[[VAL_10:.*]] = "arith.constant"() <{value = 5 : index}> : () -> index
      print(w.owner.operands[1].owner.operands[2])
      # CHECK: %[[VAL_11:.*]] = "indexing.arange"(%[[VAL_3]], %[[VAL_5]], %[[VAL_10]]) {foldAttr = false, operand_segment_sizes = array<i32: 1, 1, 1>} : (index, index, index) -> tensor<?x1xindex>
      print(w.owner.operands[1].owner)
      # CHECK: %[[VAL_12:.*]] = "indexing.gather"(%[[VAL_0]], %[[VAL_11]]) {gather_dims = array<i64: 2>, unique} : (tensor<7x22x330x4400xf32>, tensor<?x1xindex>) -> tensor<?x7x22x4400xf32>
      print(w.owner)

      return w

  module.operation.verify()

  pm = PassManager.parse('builtin.module(canonicalize)')
  pm.run(module.operation)
  # CHECK-LABEL: module {
  # CHECK:         func.func @test_dyn_indices() -> tensor<?x7x22x4400xf32> {
  # CHECK:           %[[VAL_0:.*]] = tensor.empty() : tensor<7x22x330x4400xf32>
  # CHECK:           %[[VAL_1:.*]] = indexing.arange(start = 100, stop = 200, step = 5) : tensor<20x1xindex>
  # CHECK:           %[[VAL_2:.*]] = indexing.gather %[[VAL_0]]{{\[}}%[[VAL_1]]] gather_dims([2]) unique : (tensor<7x22x330x4400xf32>, tensor<20x1xindex>) -> tensor<?x7x22x4400xf32>
  # CHECK:           return %[[VAL_2]] : tensor<?x7x22x4400xf32>
  # CHECK:         }
  # CHECK:       }
  print(module)


# CHECK-LABEL: TEST: testSimpleLiteralScatter
@run
def testSimpleLiteralScatter():
  f32 = F32Type.get()
  with mlir_mod_ctx() as module:
    ten = Tensor.empty((7, 22, 330, 4400), f32)

    w = ten[0]
    ten[0] = w

    try:
      ten[0, 0] = w
    except AssertionError as e:
      # CHECK: Expected matching shape for dest slice result_shape=(1, 330, 4400) and source source.shape=(1, 22, 330, 4400)
      print(e)

    w = ten[0, 0]
    ten[0, 0] = w

    try:
      ten[0, 0, 0] = w
    except AssertionError as e:
      # CHECK: Expected matching shape for dest slice result_shape=(1, 4400) and source source.shape=(1, 330, 4400)
      print(e)

    w = ten[0, 0, 0]
    ten[0, 0, 0] = w

    try:
      ten[0, 0, 0, 0] = w
    except AssertionError as e:
      # CHECK: Expected matching shape for dest slice result_shape=(1,) and source source.shape=(1, 4400)
      print(e)

    w = ten[0, 0, 0, 0]
    ten[0, 0, 0, 0] = w

  module.operation.verify()

  pm = PassManager.parse('builtin.module(cse)')
  pm.run(module.operation)
  # CHECK-LABEL: module {
  # CHECK:         %[[VAL_0:.*]] = tensor.empty() : tensor<7x22x330x4400xf32>
  # CHECK:         %[[VAL_1:.*]] = arith.constant dense<0> : tensor<1x1xindex>
  # CHECK:         %[[VAL_2:.*]] = indexing.gather %[[VAL_0]]{{\[}}%[[VAL_1]]] gather_dims([0]) unique : (tensor<7x22x330x4400xf32>, tensor<1x1xindex>) -> tensor<1x22x330x4400xf32>
  # CHECK:         %[[VAL_3:.*]] = indexing.scatter %[[VAL_2]] into %[[VAL_0]]{{\[}}%[[VAL_1]]] scatter_dims([0]) unique : (tensor<1x22x330x4400xf32>, tensor<7x22x330x4400xf32>, tensor<1x1xindex>) -> tensor<7x22x330x4400xf32>
  # CHECK:         %[[VAL_4:.*]] = arith.constant dense<0> : tensor<1x2xindex>
  # CHECK:         %[[VAL_5:.*]] = indexing.gather %[[VAL_3]]{{\[}}%[[VAL_4]]] gather_dims([0, 1]) unique : (tensor<7x22x330x4400xf32>, tensor<1x2xindex>) -> tensor<1x330x4400xf32>
  # CHECK:         %[[VAL_6:.*]] = indexing.scatter %[[VAL_5]] into %[[VAL_3]]{{\[}}%[[VAL_4]]] scatter_dims([0, 1]) unique : (tensor<1x330x4400xf32>, tensor<7x22x330x4400xf32>, tensor<1x2xindex>) -> tensor<7x22x330x4400xf32>
  # CHECK:         %[[VAL_7:.*]] = arith.constant dense<0> : tensor<1x3xindex>
  # CHECK:         %[[VAL_8:.*]] = indexing.gather %[[VAL_6]]{{\[}}%[[VAL_7]]] gather_dims([0, 1, 2]) unique : (tensor<7x22x330x4400xf32>, tensor<1x3xindex>) -> tensor<1x4400xf32>
  # CHECK:         %[[VAL_9:.*]] = indexing.scatter %[[VAL_8]] into %[[VAL_6]]{{\[}}%[[VAL_7]]] scatter_dims([0, 1, 2]) unique : (tensor<1x4400xf32>, tensor<7x22x330x4400xf32>, tensor<1x3xindex>) -> tensor<7x22x330x4400xf32>
  # CHECK:         %[[VAL_10:.*]] = arith.constant 0 : index
  # CHECK:         %[[VAL_11:.*]] = tensor.extract %[[VAL_9]]{{\[}}%[[VAL_10]], %[[VAL_10]], %[[VAL_10]], %[[VAL_10]]] : tensor<7x22x330x4400xf32>
  # CHECK:         %[[VAL_12:.*]] = tensor.from_elements %[[VAL_11]] : tensor<1xf32>
  # CHECK:         %[[VAL_13:.*]] = arith.constant dense<0> : tensor<1x4xindex>
  # CHECK:       }
  print(module)


# CHECK-LABEL: TEST: testWholeSliceScatter
@run
def testWholeSliceScatter():
  f32 = F32Type.get()
  with mlir_mod_ctx() as module:
    ten = Tensor.empty((7, 22, 330, 4400), f32)
    print(ten.owner)

    w = ten[0]
    try:
      ten[:, 0, 0, 0] = w
    except AssertionError as e:
      # CHECK: Expected matching shape for dest slice result_shape=(1, 7) and source source.shape=(1, 22, 330, 4400)
      print(e)

    w = ten[:, 0, 0, 0]
    ten[:, 0, 0, 0] = w

    try:
      ten[:, :, 0, 0] = w
    except AssertionError as e:
      # CHECK: Expected matching shape for dest slice result_shape=(1, 7, 22) and source source.shape=(1, 7)
      print(e)

    w = ten[:, :, 0, 0]
    ten[:, :, 0, 0] = w

    try:
      ten[:, :, :, 0] = w
    except AssertionError as e:
      # CHECK: Expected matching shape for dest slice result_shape=(1, 7, 22, 330) and source source.shape=(1, 7, 22)
      print(e)

    w = ten[:, :, :, 0]
    ten[:, :, :, 0] = w

    try:
      ten[:, :, :, :] = w
    except AssertionError as e:
      # Expected matching shape for dest slice self.shape=(7, 22, 330, 4400) and source source.shape=(1, 7, 22, 330)
      print(e)

    w = ten[:, :, :, :]
    ten[:, :, :, :] = w

  module.operation.verify()

  pm = PassManager.parse('builtin.module(cse)')
  pm.run(module.operation)
  # CHECK-LABEL: module {
  # CHECK:         %[[VAL_0:.*]] = tensor.empty() : tensor<7x22x330x4400xf32>
  # CHECK:         %[[VAL_1:.*]] = arith.constant dense<0> : tensor<1x1xindex>
  # CHECK:         %[[VAL_2:.*]] = arith.constant dense<0> : tensor<1x3xindex>
  # CHECK:         %[[VAL_3:.*]] = indexing.gather %[[VAL_0]]{{\[}}%[[VAL_2]]] gather_dims([1, 2, 3]) unique : (tensor<7x22x330x4400xf32>, tensor<1x3xindex>) -> tensor<1x7xf32>
  # CHECK:         %[[VAL_4:.*]] = indexing.scatter %[[VAL_3]] into %[[VAL_0]]{{\[}}%[[VAL_2]]] scatter_dims([1, 2, 3]) unique : (tensor<1x7xf32>, tensor<7x22x330x4400xf32>, tensor<1x3xindex>) -> tensor<7x22x330x4400xf32>
  # CHECK:         %[[VAL_5:.*]] = arith.constant dense<0> : tensor<1x2xindex>
  # CHECK:         %[[VAL_6:.*]] = indexing.gather %[[VAL_4]]{{\[}}%[[VAL_5]]] gather_dims([2, 3]) unique : (tensor<7x22x330x4400xf32>, tensor<1x2xindex>) -> tensor<1x7x22xf32>
  # CHECK:         %[[VAL_7:.*]] = indexing.scatter %[[VAL_6]] into %[[VAL_4]]{{\[}}%[[VAL_5]]] scatter_dims([2, 3]) unique : (tensor<1x7x22xf32>, tensor<7x22x330x4400xf32>, tensor<1x2xindex>) -> tensor<7x22x330x4400xf32>
  # CHECK:         %[[VAL_8:.*]] = indexing.gather %[[VAL_7]]{{\[}}%[[VAL_1]]] gather_dims([3]) unique : (tensor<7x22x330x4400xf32>, tensor<1x1xindex>) -> tensor<1x7x22x330xf32>
  # CHECK:       }
  print(module)


# CHECK-LABEL: TEST: testStaticSliceScatter
@run
def testStaticSliceScatter():
  f32 = F32Type.get()
  with mlir_mod_ctx() as module:
    ten = Tensor.empty((7, 22, 330, 4400), f32)

    w = ten[:, arange(start=0, stop=22, step=2, fold=True)]
    ten[:, arange(start=0, stop=22, step=2, fold=True)] = w

    w = ten[:,
            arange(start=0, stop=22, step=2, fold=True),
            arange(start=0, stop=330, step=30, fold=True)]
    ten[:,
        arange(start=0, stop=22, step=2, fold=True),
        arange(start=0, stop=330, step=30, fold=True)] = w

    w = ten[:,
            arange(start=0, stop=22, step=2, fold=True),
            arange(start=0, stop=330, step=30, fold=True),
            arange(start=0, stop=4400, step=400, fold=True)]
    ten[:,
        arange(start=0, stop=22, step=2, fold=True),
        arange(start=0, stop=330, step=30, fold=True),
        arange(start=0, stop=4400, step=400, fold=True)] = w

    w = ten[:, :,
            arange(start=100, stop=200, step=5, fold=True),
            arange(start=1000, stop=2000, step=50, fold=True)]
    ten[:, :,
        arange(start=100, stop=200, step=5, fold=True),
        arange(start=1000, stop=2000, step=50, fold=True)] = w

  module.operation.verify()

  pm = PassManager.parse('builtin.module(cse)')
  pm.run(module.operation)
  # CHECK-LABEL: module {
  # CHECK:         %[[VAL_0:.*]] = tensor.empty() : tensor<7x22x330x4400xf32>
  # CHECK:         %[[VAL_1:.*]] = arith.constant dense<{{\[\[}}0], [2], [4], [6], [8], [10], [12], [14], [16], [18], [20]]> : tensor<11x1xindex>
  # CHECK:         %[[VAL_2:.*]] = indexing.gather %[[VAL_0]]{{\[}}%[[VAL_1]]] gather_dims([1]) unique : (tensor<7x22x330x4400xf32>, tensor<11x1xindex>) -> tensor<11x7x330x4400xf32>
  # CHECK:         %[[VAL_3:.*]] = indexing.scatter %[[VAL_2]] into %[[VAL_0]]{{\[}}%[[VAL_1]]] scatter_dims([1]) unique : (tensor<11x7x330x4400xf32>, tensor<7x22x330x4400xf32>, tensor<11x1xindex>) -> tensor<7x22x330x4400xf32>
  # CHECK:         %[[VAL_4:.*]] = arith.constant dense<{{\[\[}}0, 0], [2, 30], [4, 60], [6, 90], [8, 120], [10, 150], [12, 180], [14, 210], [16, 240], [18, 270], [20, 300]]> : tensor<11x2xindex>
  # CHECK:         %[[VAL_5:.*]] = indexing.gather %[[VAL_3]]{{\[}}%[[VAL_4]]] gather_dims([1, 2]) unique : (tensor<7x22x330x4400xf32>, tensor<11x2xindex>) -> tensor<11x7x4400xf32>
  # CHECK:         %[[VAL_6:.*]] = indexing.scatter %[[VAL_5]] into %[[VAL_3]]{{\[}}%[[VAL_4]]] scatter_dims([1, 2]) unique : (tensor<11x7x4400xf32>, tensor<7x22x330x4400xf32>, tensor<11x2xindex>) -> tensor<7x22x330x4400xf32>
  # CHECK:         %[[VAL_7:.*]] = arith.constant dense<{{\[\[}}0, 0, 0], [2, 30, 400], [4, 60, 800], [6, 90, 1200], [8, 120, 1600], [10, 150, 2000], [12, 180, 2400], [14, 210, 2800], [16, 240, 3200], [18, 270, 3600], [20, 300, 4000]]> : tensor<11x3xindex>
  # CHECK:         %[[VAL_8:.*]] = indexing.gather %[[VAL_6]]{{\[}}%[[VAL_7]]] gather_dims([1, 2, 3]) unique : (tensor<7x22x330x4400xf32>, tensor<11x3xindex>) -> tensor<11x7xf32>
  # CHECK:         %[[VAL_9:.*]] = indexing.scatter %[[VAL_8]] into %[[VAL_6]]{{\[}}%[[VAL_7]]] scatter_dims([1, 2, 3]) unique : (tensor<11x7xf32>, tensor<7x22x330x4400xf32>, tensor<11x3xindex>) -> tensor<7x22x330x4400xf32>
  # CHECK:         %[[VAL_10:.*]] = arith.constant dense<{{\[\[}}100, 1000], [105, 1050], [110, 1100], [115, 1150], [120, 1200], [125, 1250], [130, 1300], [135, 1350], [140, 1400], [145, 1450], [150, 1500], [155, 1550], [160, 1600], [165, 1650], [170, 1700], [175, 1750], [180, 1800], [185, 1850], [190, 1900], [195, 1950]]> : tensor<20x2xindex>
  # CHECK:         %[[VAL_11:.*]] = indexing.gather %[[VAL_9]]{{\[}}%[[VAL_10]]] gather_dims([2, 3]) unique : (tensor<7x22x330x4400xf32>, tensor<20x2xindex>) -> tensor<20x7x22xf32>
  # CHECK:       }
  print(module)


# CHECK-LABEL: TEST: testDynSliceScatter
@run
def testDynSliceScatter():
  f32 = F32Type.get()
  with mlir_mod_ctx() as module:
    ten = Tensor.empty((7, 22, 330, 4400), f32)

    w = ten[:, 0:22:2]
    ten[:, 0:22:2] = w

    w = ten[:, 0:22:2, 0:330:30]
    ten[:, 0:22:2, 0:330:30] = w

    w = ten[:, 0:22:2, 0:330:30, 0:4400:400]
    ten[:, 0:22:2, 0:330:30, 0:4400:400] = w

    w = ten[:, :, 100:200:5, 1000:2000:50]
    ten[:, :, 100:200:5, 1000:2000:50] = w

  module.operation.verify()

  pm = PassManager.parse('builtin.module(cse)')
  pm.run(module.operation)
  # CHECK-LABEL: module {
  # CHECK:         %[[VAL_0:.*]] = tensor.empty() : tensor<7x22x330x4400xf32>
  # CHECK:         %[[VAL_1:.*]] = arith.constant 0 : index
  # CHECK:         %[[VAL_2:.*]] = arith.constant 22 : index
  # CHECK:         %[[VAL_3:.*]] = arith.constant 2 : index
  # CHECK:         %[[VAL_4:.*]] = indexing.arange(start = %[[VAL_1]], stop = %[[VAL_2]], step = %[[VAL_3]]) : tensor<?x1xindex>
  # CHECK:         %[[VAL_5:.*]] = indexing.gather %[[VAL_0]]{{\[}}%[[VAL_4]]] gather_dims([1]) unique : (tensor<7x22x330x4400xf32>, tensor<?x1xindex>) -> tensor<?x7x330x4400xf32>
  # CHECK:         %[[VAL_6:.*]] = indexing.scatter %[[VAL_5]] into %[[VAL_0]]{{\[}}%[[VAL_4]]] scatter_dims([1]) unique : (tensor<?x7x330x4400xf32>, tensor<7x22x330x4400xf32>, tensor<?x1xindex>) -> tensor<7x22x330x4400xf32>
  # CHECK:         %[[VAL_7:.*]] = arith.constant 330 : index
  # CHECK:         %[[VAL_8:.*]] = arith.constant 30 : index
  # CHECK:         %[[VAL_9:.*]] = indexing.arange(start = %[[VAL_1]], stop = %[[VAL_7]], step = %[[VAL_8]]) : tensor<?x1xindex>
  # CHECK:         %[[VAL_10:.*]] = indexing.concatenate(%[[VAL_4]], %[[VAL_9]]) {dim = 1} : (tensor<?x1xindex>, tensor<?x1xindex>) -> tensor<?x2xindex>
  # CHECK:         %[[VAL_11:.*]] = indexing.gather %[[VAL_6]]{{\[}}%[[VAL_10]]] gather_dims([1, 2]) unique : (tensor<7x22x330x4400xf32>, tensor<?x2xindex>) -> tensor<?x7x4400xf32>
  # CHECK:         %[[VAL_12:.*]] = indexing.scatter %[[VAL_11]] into %[[VAL_6]]{{\[}}%[[VAL_10]]] scatter_dims([1, 2]) unique : (tensor<?x7x4400xf32>, tensor<7x22x330x4400xf32>, tensor<?x2xindex>) -> tensor<7x22x330x4400xf32>
  # CHECK:         %[[VAL_13:.*]] = arith.constant 4400 : index
  # CHECK:         %[[VAL_14:.*]] = arith.constant 400 : index
  # CHECK:         %[[VAL_15:.*]] = indexing.arange(start = %[[VAL_1]], stop = %[[VAL_13]], step = %[[VAL_14]]) : tensor<?x1xindex>
  # CHECK:         %[[VAL_16:.*]] = indexing.concatenate(%[[VAL_4]], %[[VAL_9]], %[[VAL_15]]) {dim = 1} : (tensor<?x1xindex>, tensor<?x1xindex>, tensor<?x1xindex>) -> tensor<?x3xindex>
  # CHECK:         %[[VAL_17:.*]] = indexing.gather %[[VAL_12]]{{\[}}%[[VAL_16]]] gather_dims([1, 2, 3]) unique : (tensor<7x22x330x4400xf32>, tensor<?x3xindex>) -> tensor<?x7xf32>
  # CHECK:         %[[VAL_18:.*]] = indexing.scatter %[[VAL_17]] into %[[VAL_12]]{{\[}}%[[VAL_16]]] scatter_dims([1, 2, 3]) unique : (tensor<?x7xf32>, tensor<7x22x330x4400xf32>, tensor<?x3xindex>) -> tensor<7x22x330x4400xf32>
  # CHECK:         %[[VAL_19:.*]] = arith.constant 100 : index
  # CHECK:         %[[VAL_20:.*]] = arith.constant 200 : index
  # CHECK:         %[[VAL_21:.*]] = arith.constant 5 : index
  # CHECK:         %[[VAL_22:.*]] = indexing.arange(start = %[[VAL_19]], stop = %[[VAL_20]], step = %[[VAL_21]]) : tensor<?x1xindex>
  # CHECK:         %[[VAL_23:.*]] = arith.constant 1000 : index
  # CHECK:         %[[VAL_24:.*]] = arith.constant 2000 : index
  # CHECK:         %[[VAL_25:.*]] = arith.constant 50 : index
  # CHECK:         %[[VAL_26:.*]] = indexing.arange(start = %[[VAL_23]], stop = %[[VAL_24]], step = %[[VAL_25]]) : tensor<?x1xindex>
  # CHECK:         %[[VAL_27:.*]] = indexing.concatenate(%[[VAL_22]], %[[VAL_26]]) {dim = 1} : (tensor<?x1xindex>, tensor<?x1xindex>) -> tensor<?x2xindex>
  # CHECK:         %[[VAL_28:.*]] = indexing.gather %[[VAL_18]]{{\[}}%[[VAL_27]]] gather_dims([2, 3]) unique : (tensor<7x22x330x4400xf32>, tensor<?x2xindex>) -> tensor<?x7x22xf32>
  # CHECK:       }
  print(module)


# CHECK-LABEL: TEST: testForLoopSugarNoIterArgs
@run
def testForLoopSugarNoIterArgs():
  f32 = F32Type.get()
  with mlir_mod_ctx() as module:
    ten = Tensor.empty((7, 22, 330, 4400), f32)

    for i in scf_range(0, 10):
      y = 2 * i

  module.operation.verify()
  # CHECK-LABEL: module {
  # CHECK:         %[[VAL_0:.*]] = tensor.empty() : tensor<7x22x330x4400xf32>
  # CHECK:         %[[VAL_1:.*]] = arith.constant 0 : index
  # CHECK:         %[[VAL_2:.*]] = arith.constant 10 : index
  # CHECK:         %[[VAL_3:.*]] = arith.constant 1 : index
  # CHECK:         scf.for %[[VAL_4:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_3]] {
  # CHECK:           %[[VAL_5:.*]] = arith.constant 2 : index
  # CHECK:           %[[VAL_6:.*]] = arith.muli %[[VAL_4]], %[[VAL_5]] : index
  # CHECK:         }
  # CHECK:       }
  print(module)


# CHECK-LABEL: TEST: testForLoopSugarIterArgs
@run
def testForLoopSugarIterArgs():
  f32 = F32Type.get()
  with mlir_mod_ctx() as module:
    ten = Tensor.empty((7, 22, 330, 4400), f32)

    for i, _ in scf_range(0, 10, iter_args=[ten]):
      y = ten + ten
      scf_yield(y)

  module.operation.verify()
  # CHECK-LABEL: module {
  # CHECK:         %[[VAL_0:.*]] = tensor.empty() : tensor<7x22x330x4400xf32>
  # CHECK:         %[[VAL_1:.*]] = arith.constant 0 : index
  # CHECK:         %[[VAL_2:.*]] = arith.constant 10 : index
  # CHECK:         %[[VAL_3:.*]] = arith.constant 1 : index
  # CHECK:         %[[VAL_4:.*]] = scf.for %[[VAL_5:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_3]] iter_args(%[[VAL_6:.*]] = %[[VAL_0]]) -> (tensor<7x22x330x4400xf32>) {
  # CHECK:           %[[VAL_7:.*]] = arith.addf %[[VAL_6]], %[[VAL_6]] : tensor<7x22x330x4400xf32>
  # CHECK:           scf.yield %[[VAL_7]] : tensor<7x22x330x4400xf32>
  # CHECK:         }
  # CHECK:       }
  print(module)


# CHECK-LABEL: TEST: testForLoopSugarResult
@run
def testForLoopSugarResult():
  f32 = F32Type.get()
  with mlir_mod_ctx() as module:

    @func.FuncOp.from_py_func(*[])
    def test_fold():
      ten = Tensor.empty((7, 22, 330, 4400), f32)

      for i, result in scf_range(0, 10, iter_args=[ten]):
        y = ten + ten
        scf_yield(y)
      return result

  module.operation.verify()

  # CHECK-LABEL: module {
  # CHECK:         func.func @test_fold() -> tensor<7x22x330x4400xf32> {
  # CHECK:           %[[VAL_0:.*]] = tensor.empty() : tensor<7x22x330x4400xf32>
  # CHECK:           %[[VAL_1:.*]] = arith.constant 0 : index
  # CHECK:           %[[VAL_2:.*]] = arith.constant 10 : index
  # CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
  # CHECK:           %[[VAL_4:.*]] = scf.for %[[VAL_5:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_3]] iter_args(%[[VAL_6:.*]] = %[[VAL_0]]) -> (tensor<7x22x330x4400xf32>) {
  # CHECK:             %[[VAL_7:.*]] = arith.addf %[[VAL_6]], %[[VAL_6]] : tensor<7x22x330x4400xf32>
  # CHECK:             scf.yield %[[VAL_7]] : tensor<7x22x330x4400xf32>
  # CHECK:           }
  # CHECK:           return %[[VAL_8:.*]] : tensor<7x22x330x4400xf32>
  # CHECK:         }
  # CHECK:       }
  print(module)


# CHECK-LABEL: TEST: testForLoopSugarNested
@run
def testForLoopSugarNested():
  f32 = F32Type.get()
  with mlir_mod_ctx() as module:

    @func.FuncOp.from_py_func(*[])
    def test_double_loop():
      ten = Tensor.empty((7, 22, 330, 4400), f32)

      for i, result1 in scf_range(0, 10, iter_args=[ten]):
        for i, result2 in scf_range(0, 10, iter_args=[ten]):
          y = ten + ten
          scf_yield(y)
        scf_yield(result2)
      return result1

  module.operation.verify()

  pm = PassManager.parse('builtin.module(cse)')
  pm.run(module.operation)
  # CHECK-LABEL: module {
  # CHECK:         func.func @test_double_loop() -> tensor<7x22x330x4400xf32> {
  # CHECK:           %[[VAL_0:.*]] = tensor.empty() : tensor<7x22x330x4400xf32>
  # CHECK:           %[[VAL_1:.*]] = arith.constant 0 : index
  # CHECK:           %[[VAL_2:.*]] = arith.constant 10 : index
  # CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
  # CHECK:           %[[VAL_4:.*]] = scf.for %[[VAL_5:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_3]] iter_args(%[[VAL_6:.*]] = %[[VAL_0]]) -> (tensor<7x22x330x4400xf32>) {
  # CHECK:             %[[VAL_7:.*]] = scf.for %[[VAL_8:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_3]] iter_args(%[[VAL_9:.*]] = %[[VAL_6]]) -> (tensor<7x22x330x4400xf32>) {
  # CHECK:               %[[VAL_10:.*]] = arith.addf %[[VAL_9]], %[[VAL_9]] : tensor<7x22x330x4400xf32>
  # CHECK:               scf.yield %[[VAL_10]] : tensor<7x22x330x4400xf32>
  # CHECK:             }
  # CHECK:             scf.yield %[[VAL_11:.*]] : tensor<7x22x330x4400xf32>
  # CHECK:           }
  # CHECK:           return %[[VAL_12:.*]] : tensor<7x22x330x4400xf32>
  # CHECK:         }
  # CHECK:       }
  print(module)
