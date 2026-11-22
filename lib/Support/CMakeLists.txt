add_mlir_library(StreamSupport

  CalyxBuilder.cpp

  ADDITIONAL_HEADER_DIRS
  ${STREAM_SOURCE_DIR}/include

  LINK_LIBS PUBLIC
  CIRCTCalyx
  MLIRIR
  MLIRSupport
)
