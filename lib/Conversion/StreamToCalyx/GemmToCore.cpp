//===- GemmToCore.cpp - Expand stream.gemm into core Stream ops ------------===//

#include "Stream/StreamDialectOps.h"
#include "Stream/StreamPasses.h"

#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

namespace stream {
#define GEN_PASS_DEF_CONVERTGEMMTOCORE
#include "Stream/StreamPasses.h.inc"

static mlir::LogicalResult expandGemm(Gemm gemm) {
  auto aTy = mlir::dyn_cast<mlir::MemRefType>(gemm.getA().getType());
  auto bTy = mlir::dyn_cast<mlir::MemRefType>(gemm.getB().getType());
  auto cTy = mlir::dyn_cast<mlir::MemRefType>(gemm.getC().getType());
  if (!aTy || !bTy || !cTy || aTy.getRank() != 2 || bTy.getRank() != 2 ||
      cTy.getRank() != 2)
    return gemm.emitError("gemm expects rank-2 memrefs");

  int64_t M = aTy.getShape()[0];
  int64_t K = aTy.getShape()[1];
  int64_t N = bTy.getShape()[1];
  if (bTy.getShape()[0] != K || cTy.getShape()[0] != M ||
      cTy.getShape()[1] != N)
    return gemm.emitError("gemm shape mismatch");

  int64_t TM = gemm.getTileM();
  int64_t TN = gemm.getTileN();
  int64_t TK = gemm.getTileK() == 0 ? K : gemm.getTileK();
  if (TM <= 0 || TN <= 0 || TK <= 0 || M % TM != 0 || N % TN != 0 ||
      K % TK != 0)
    return gemm.emitError("gemm tiles must divide M, N, and K");

  bool banked = gemm.getBanksA() == TM && gemm.getBanksB() == TN;

  mlir::OpBuilder b(gemm);
  mlir::Location loc = gemm.getLoc();

  auto allocTile = [&](llvm::ArrayRef<int64_t> shape, int64_t banks,
                       int64_t bankDim) -> mlir::Value {
    auto ty = mlir::MemRefType::get(shape, b.getI32Type());
    auto alloc = BufferAlloc::create(
        b, loc, ty, b.getI64IntegerAttr(banks),
        b.getI64IntegerAttr(bankDim), b.getStringAttr("seq"));
    return alloc.getBuffer();
  };

  auto i32 = [&](int64_t value) -> mlir::Value {
    mlir::OperationState state(loc, arith::ConstantOp::getOperationName());
    state.addTypes(b.getI32Type());
    state.addAttribute("value", b.getI32IntegerAttr(value));
    return b.create(state)->getResult(0);
  };

  bool fullK = TK == K;
  int64_t resident = M * K + K * N + M * N;
  if (fullK && !banked && resident <= 4096) {
    // Whole-problem-resident schedule: load A, B, and C once, iterate tiles,
    // and store C once. Reuses A/B tiles across the tile loops.
    mlir::Value aFull = allocTile({M, K}, 1, -1);
    mlir::Value bFull = allocTile({K, N}, 1, -1);
    mlir::Value cFull = allocTile({M, N}, 1, -1);

    auto par = b.create<ParOp>(loc);
    par.getBody().push_back(new mlir::Block());
    b.setInsertionPointToStart(&par.getBody().front());
    b.create<BufferLoad>(loc, aFull, gemm.getA(),
                         mlir::ValueRange{i32(0), i32(0)});
    b.create<BufferLoad>(loc, bFull, gemm.getB(),
                         mlir::ValueRange{i32(0), i32(0)});
    b.create<YieldOp>(loc);
    b.setInsertionPointAfter(par);

    auto m = b.create<LoopOp>(loc, 0, M, TM);
    b.setInsertionPointToStart(&m.getBody().front());
    auto n = b.create<LoopOp>(loc, 0, N, TN);
    b.setInsertionPointToStart(&n.getBody().front());

    mlir::Value mIdx = m.getBody().front().getArgument(0);
    mlir::Value nIdx = n.getBody().front().getArgument(0);

    auto aViewTy = mlir::MemRefType::get({TM, K}, b.getI32Type());
    auto bViewTy = mlir::MemRefType::get({K, TN}, b.getI32Type());
    auto cViewTy = mlir::MemRefType::get({TM, TN}, b.getI32Type());
    mlir::Value aView = BufferView::create(
        b, loc, aViewTy, aFull, mlir::ValueRange{mIdx, i32(0)});
    mlir::Value bView = BufferView::create(
        b, loc, bViewTy, bFull, mlir::ValueRange{i32(0), nIdx});
    mlir::Value cView = BufferView::create(
        b, loc, cViewTy, cFull, mlir::ValueRange{mIdx, nIdx});

    b.create<GemmTile>(loc, aView, bView, cView, b.getI64IntegerAttr(K),
                       b.getBoolAttr(true));
    b.create<YieldOp>(loc);

    b.setInsertionPointAfter(n);
    b.create<YieldOp>(loc);

    b.setInsertionPointAfter(m);
    b.create<BufferStore>(loc, gemm.getC(),
                          mlir::ValueRange{i32(0), i32(0)}, cFull);

    gemm.erase();
    return mlir::success();
  }

  mlir::Value aTile = allocTile({TM, TK}, banked ? TM : 1, banked ? 0 : -1);
  mlir::Value bTile = allocTile({TK, TN}, banked ? TN : 1, banked ? 1 : -1);
  mlir::Value cTile = allocTile({TM, TN}, 1, -1);

  auto m = b.create<LoopOp>(loc, 0, M, TM);
  b.setInsertionPointToStart(&m.getBody().front());
  auto n = b.create<LoopOp>(loc, 0, N, TN);
  b.setInsertionPointToStart(&n.getBody().front());

  mlir::Value mIdx = m.getBody().front().getArgument(0);
  mlir::Value nIdx = n.getBody().front().getArgument(0);

  // For K tiling, partial sums live in the C tile across K tiles.
  if (!fullK)
    b.create<BufferZero>(loc, cTile);

  auto k = b.create<LoopOp>(loc, 0, K, TK);
  b.setInsertionPointToStart(&k.getBody().front());
  mlir::Value kIdx = k.getBody().front().getArgument(0);

  auto par = b.create<ParOp>(loc);
  par.getBody().push_back(new mlir::Block());
  b.setInsertionPointToStart(&par.getBody().front());
  b.create<BufferLoad>(loc, aTile, gemm.getA(), mlir::ValueRange{mIdx, kIdx});
  b.create<BufferLoad>(loc, bTile, gemm.getB(), mlir::ValueRange{kIdx, nIdx});
  b.create<YieldOp>(loc);

  b.setInsertionPointAfter(par);
  b.create<GemmTile>(loc, aTile, bTile, cTile, b.getI64IntegerAttr(TK),
                     b.getBoolAttr(fullK));
  b.create<YieldOp>(loc);

  b.setInsertionPointAfter(k);
  b.create<BufferStore>(loc, gemm.getC(), mlir::ValueRange{mIdx, nIdx}, cTile);
  b.create<YieldOp>(loc);

  b.setInsertionPointAfter(n);
  b.create<YieldOp>(loc);

  gemm.erase();
  return mlir::success();
}

struct ConvertGemmToCorePass
    : impl::ConvertGemmToCoreBase<ConvertGemmToCorePass> {
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    llvm::SmallVector<Gemm> gemms;
    module.walk([&](Gemm gemm) { gemms.push_back(gemm); });
    for (Gemm gemm : gemms) {
      if (mlir::failed(expandGemm(gemm))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace stream
