set(LLVM_TARGET_DEFINITIONS TritonConversions.td)
mlir_tablegen(TritonConversions.h.inc -gen-pass-decls -name TritonConversion)
mlir_tablegen(TritonConversions.capi.h.inc -gen-pass-capi-header --prefix TritonConversion)
mlir_tablegen(TritonConversions.capi.cpp.inc -gen-pass-capi-impl --prefix TritonConversion)
add_public_tablegen_target(MLIRTritonConversionIncGen)

set(LLVM_TARGET_DEFINITIONS TritonTransforms.td)
mlir_tablegen(TritonTransforms.capi.h.inc -gen-pass-capi-header --prefix TritonTransform)
mlir_tablegen(TritonTransforms.capi.cpp.inc -gen-pass-capi-impl --prefix TritonTransform)
add_public_tablegen_target(MLIRTritonTransformIncGen)

# This is last such that (implicitly) depends on the previous two.
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name StructuredConversion)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix StructuredConversion)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix StructuredConversion)
add_public_tablegen_target(MLIRStructuredConversionIncGen)
