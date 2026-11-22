//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Stream/StreamPasses.h.inc"

#include "Stream/StreamDialect.h"
#include "circt/Support/Version.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include "Stream/StreamPasses.h"

int main(int argc, char **argv) {
  llvm::setBugReportMsg(circt::circtBugReportMsg);

  mlir::DialectRegistry registry;
    registry.insert<
      mlir::func::FuncDialect,
      mlir::arith::ArithDialect,
      mlir::scf::SCFDialect,
      mlir::tensor::TensorDialect,
      mlir::vector::VectorDialect,
      stream::StreamDialect
  >();


  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();
  stream::registerPasses();
  llvm::cl::AddExtraVersionPrinter(
      [](llvm::raw_ostream &os) { os << circt::getCirctVersion() << '\n'; });

  return mlir::failed(mlir::MlirOptMain(
      argc, argv, "CIRCT standalone optimizer driver", registry));
}
