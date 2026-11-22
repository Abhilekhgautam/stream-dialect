set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_executable(stream-opt
  stream-opt.cpp

  DEPENDS
  SUPPORT_PLUGINS
)

llvm_update_compile_flags(stream-opt)

target_link_libraries(stream-opt
  PRIVATE
  Stream
  StreamToCalyx
  CIRCTHW
  CIRCTCalyx
  MLIRIR
  MLIROptLib
  MLIRParser
  MLIRSupport
  MLIRTransforms

  MLIRFuncDialect MLIRArithDialect MLIRSCFDialect MLIRTensorDialect MLIRVectorDialect
)

export_executable_symbols_for_plugins(stream-opt)
