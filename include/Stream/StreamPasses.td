include "mlir/Pass/PassBase.td"

def ConvertStreamToCalyx : Pass<"convert-stream-to-calyx", "mlir::ModuleOp"> {
  let summary = "Convert the Stream dialect to Calyx dialect";
  let description = [{
    Converts all `Stream` ops into Calyx ops.
  }];

  let dependentDialects = [
    "::stream::StreamDialect",
    "::circt::calyx::CalyxDialect"
  ];
}

def ConvertGemmToCore : Pass<"convert-gemm-to-core", "mlir::ModuleOp"> {
  let summary = "Expand stream.gemm into tile loops, loads/stores, and gemm_tile";
  let dependentDialects = [
    "::stream::StreamDialect",
    "::mlir::arith::ArithDialect"
  ];
}

def ConvertBufferToCalyx : Pass<"convert-buffer-to-calyx", "mlir::ModuleOp"> {
  let summary = "Convert stream.buffer ops to Calyx dialect";
  let description = [{
    Lowers seq/comb/reg bopsuffer allocations, reads, writes, and zeros.
  }];

  let dependentDialects = [
    "::stream::StreamDialect",
    "::circt::calyx::CalyxDialect"
  ];
}
