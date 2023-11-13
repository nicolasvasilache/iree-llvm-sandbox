"""Trampoline to run generated MLIR Python code.

Generated tablegen dialects expect to be able to find some symbols from the
iree.compiler.dialects package.
"""

from iree.compiler.dialects._transform_ops_gen import _Dialect
