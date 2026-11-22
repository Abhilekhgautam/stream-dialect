add_mlir_library(StreamToCalyx

  StreamToCalyx.cpp
  BufferLowering.cpp
  BufferToCalyx.cpp
  GemmToCore.cpp

  ADDITIONAL_HEADER_DIRS
  ${STREAM_SOURCE_DIR}/include

  DEPENDS
  StreamPassesIncGen

  LINK_LIBS PUBLIC
  Stream
  StreamSupport
  CIRCTCalyx
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRFuncDialect
  MLIRArithDialect
)
