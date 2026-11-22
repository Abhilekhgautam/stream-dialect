add_mlir_dialect_library(Stream

  StreamDialect.cpp
  StreamDialectOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${STREAM_SOURCE_DIR}/include/Stream

  DEPENDS
  MLIRStreamDialectOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRBytecodeOpInterface
)
