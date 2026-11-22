include "mlir/IR/OpBase.td"

def Stream_Dialect : Dialect {
  let name = "stream";
  let summary = "Stream dialect for streaming hardware architecture";
  let description = [{
    Dialect to express streaming architectural intent in mlir.
  }];
  let cppNamespace = "::stream";
}
