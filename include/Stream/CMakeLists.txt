add_mlir_dialect(StreamDialectOps stream)
set(LLVM_TARGET_DEFINITIONS StreamPasses.td)
mlir_tablegen(StreamPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(StreamPassesIncGen)
