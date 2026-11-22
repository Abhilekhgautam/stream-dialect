//===- BufferToCalyx.cpp - Lower stream core ops to Calyx -----------------===//

#include "Conversion/BufferLowering.h"
#include "Stream/StreamDialectOps.h"
#include "Stream/StreamPasses.h"
#include "Support/CalyxBuilder.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

namespace stream {
#define GEN_PASS_DEF_CONVERTBUFFERTOCALYX
#include "Stream/StreamPasses.h.inc"

static mlir::LogicalResult lowerBlock(mlir::Block &block, LoweringContext &ctx,
                                      BufferLowering &bufferLowering);

/// Lower `stream.loop` to a Calyx counter + repeat.
static mlir::LogicalResult lowerLoop(LoopOp loop, LoweringContext &ctx,
                                     BufferLowering &bufferLowering) {
  mlir::OpBuilder &b = ctx.cb.getBuilder();
  const int64_t lower = loop.getLower();
  const int64_t upper = loop.getUpper();
  const int64_t step = loop.getStep();

  if (step <= 0 || upper <= lower)
    return loop.emitError("loop requires step > 0 and upper > lower");

  if ((upper - lower) % step != 0)
    return loop.emitError("loop bounds must be divisible by step");

  mlir::Block &body = loop.getBody().front();
  if (body.getNumArguments() != 1 ||
      !body.getArgument(0).getType().isInteger(32))
    return loop.emitError("loop body must have one i32 induction variable");

  const int64_t count = (upper - lower) / step;
  auto iv = ctx.cb.createRegister(ctx.fresh("iv"), 32);

  // Initialize the induction variable.
  const std::string initName = ctx.fresh("init_iv");
  {
    mlir::OpBuilder::InsertionGuard guard(b);
    const auto group = ctx.cb.createGroup(initName);
    ctx.cb.assign(iv.getIn(), ctx.cb.i32(lower));
    ctx.cb.assignBit(iv.getWriteEn(), true);
    ctx.cb.groupDone(group, iv.getDone());
  }
  ctx.addEnable(initName);

  // Repeat body.
  ctx.cb.createRepeat(static_cast<uint32_t>(count));
  auto bodySeq = ctx.cb.createSeq();
  ctx.control.push_back(bodySeq.getBodyBlock());
  ctx.valueMap[body.getArgument(0)] = iv.getOut();

  if (mlir::failed(lowerBlock(body, ctx, bufferLowering)))
    return mlir::failure();

  // Increment after the body.
  auto add = ctx.cb.createAdd(ctx.fresh("iv_add"), b.getI32Type());
  const std::string incrName = ctx.fresh("incr_iv");

  {
    mlir::OpBuilder::InsertionGuard guard(b);
    const auto group = ctx.cb.createGroup(incrName);
    ctx.cb.assign(add.getLeft(), iv.getOut());
    ctx.cb.assign(add.getRight(), ctx.cb.i32(step));
    ctx.cb.assign(iv.getIn(), add.getOut());
    ctx.cb.assignBit(iv.getWriteEn(), true);
    ctx.cb.groupDone(group, iv.getDone());
  }
  ctx.addEnable(incrName);

  ctx.control.pop_back();
  ctx.valueMap.erase(body.getArgument(0));
  loop.erase();
  return mlir::success();
}

static mlir::LogicalResult lowerOp(mlir::Operation &bodyOp,
                                   LoweringContext &ctx,
                                   BufferLowering &bufferLowering);

static mlir::LogicalResult lowerPar(ParOp par, LoweringContext &ctx,
                                    BufferLowering &bufferLowering) {
  mlir::OpBuilder &b = ctx.cb.getBuilder();
  mlir::OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToEnd(ctx.control.back());
  auto calyxPar = ctx.cb.createPar();
  ctx.control.push_back(calyxPar.getBodyBlock());

  // Each top-level op in a par body is an independent control branch. Wrap
  // each branch in a seq so multi-step lowerings (e.g. banked loads) do not
  // become parallel siblings.
  for (mlir::Operation &bodyOp :
       llvm::make_early_inc_range(par.getBody().front())) {
    if (llvm::isa<YieldOp>(&bodyOp)) {
      bodyOp.erase();
      continue;
    }
    if (llvm::isa<mlir::arith::ConstantOp>(&bodyOp))
      continue;
    if (llvm::isa<BufferAlloc, BufferView>(&bodyOp)) {
      if (mlir::failed(lowerOp(bodyOp, ctx, bufferLowering)))
        return mlir::failure();
      continue;
    }
    mlir::OpBuilder::InsertionGuard branchGuard(b);
    b.setInsertionPointToEnd(ctx.control.back());
    auto branch = ctx.cb.createSeq();
    ctx.control.push_back(branch.getBodyBlock());
    if (mlir::failed(lowerOp(bodyOp, ctx, bufferLowering)))
      return mlir::failure();
    ctx.control.pop_back();
  }

  ctx.control.pop_back();
  par.erase();
  return mlir::success();
}

static mlir::LogicalResult lowerBlock(mlir::Block &block, LoweringContext &ctx,
                                      BufferLowering &bufferLowering) {
  for (mlir::Operation &bodyOp : llvm::make_early_inc_range(block))
    if (mlir::failed(lowerOp(bodyOp, ctx, bufferLowering)))
      return mlir::failure();
  return mlir::success();
}

static mlir::LogicalResult lowerOp(mlir::Operation &bodyOp,
                                   LoweringContext &ctx,
                                   BufferLowering &bufferLowering) {
  if (auto op = llvm::dyn_cast<BufferAlloc>(&bodyOp)) {
    if (mlir::failed(bufferLowering.LowerAlloc(op)))
      return mlir::failure();
  } else if (auto op = llvm::dyn_cast<BufferRead>(&bodyOp)) {
    if (mlir::failed(bufferLowering.LowerRead(op)))
      return mlir::failure();
  } else if (auto op = llvm::dyn_cast<BufferWrite>(&bodyOp)) {
    if (mlir::failed(bufferLowering.LowerWrite(op)))
      return mlir::failure();
  } else if (auto op = llvm::dyn_cast<BufferZero>(&bodyOp)) {
    if (mlir::failed(bufferLowering.LowerZero(op)))
      return mlir::failure();
  } else if (auto op = llvm::dyn_cast<BufferView>(&bodyOp)) {
    if (mlir::failed(bufferLowering.LowerView(op)))
      return mlir::failure();
  } else if (auto op = llvm::dyn_cast<BufferLoad>(&bodyOp)) {
    if (mlir::failed(bufferLowering.LowerLoad(op)))
      return mlir::failure();
  } else if (auto op = llvm::dyn_cast<BufferStore>(&bodyOp)) {
    if (mlir::failed(bufferLowering.LowerStore(op)))
      return mlir::failure();
  } else if (auto op = llvm::dyn_cast<GemmTile>(&bodyOp)) {
    if (mlir::failed(bufferLowering.LowerGemmTile(op)))
      return mlir::failure();
  } else if (auto op = llvm::dyn_cast<LoopOp>(&bodyOp)) {
    if (mlir::failed(lowerLoop(op, ctx, bufferLowering)))
      return mlir::failure();
  } else if (auto op = llvm::dyn_cast<ParOp>(&bodyOp)) {
    if (mlir::failed(lowerPar(op, ctx, bufferLowering)))
      return mlir::failure();
  } else if (llvm::isa<YieldOp>(&bodyOp)) {
    bodyOp.erase();
  }
  return mlir::success();
}

struct ConvertBufferToCalyxPass
    : impl::ConvertBufferToCalyxBase<ConvertBufferToCalyxPass> {
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    llvm::SmallVector<mlir::func::FuncOp> funcs;
    module.walk([&](mlir::func::FuncOp func) {
      bool hasBufferOps = false;
      func.walk([&](mlir::Operation *op) {
        if (llvm::isa<BufferAlloc, BufferRead, BufferWrite, BufferZero,
                      BufferLoad, BufferStore, GemmTile>(op))
          hasBufferOps = true;
      });
      if (hasBufferOps)
        funcs.push_back(func);
    });

    if (funcs.size() != 1) {
      module.emitError("expected exactly one function with buffer ops");
      return signalPassFailure();
    }

    mlir::func::FuncOp func = funcs.front();
    mlir::OpBuilder b(module.getContext());
    b.setInsertionPointToStart(module.getBody());

    mlir::Type i1 = b.getI1Type();
    auto portAttrs = [&](llvm::StringRef name) {
      return b.getDictionaryAttr({b.getNamedAttr(name, b.getUnitAttr())});
    };
    llvm::SmallVector<circt::calyx::PortInfo> ports = {
        {b.getStringAttr("go"), i1, circt::calyx::Direction::Input,
         portAttrs("go")},
        {b.getStringAttr("clk"), i1, circt::calyx::Direction::Input,
         portAttrs("clk")},
        {b.getStringAttr("reset"), i1, circt::calyx::Direction::Input,
         portAttrs("reset")},
        {b.getStringAttr("done"), i1, circt::calyx::Direction::Output,
         portAttrs("done")},
    };

    CalyxBuilder cb(b, func.getLoc());
    auto comp = cb.createComponent(b.getStringAttr("main"), ports);
    b.setInsertionPointToEnd(comp.getControlOp().getBodyBlock());
    auto topSeq = cb.createSeq();

    LoweringContext ctx(cb);

    // External function memrefs become Calyx memories.
    for (auto [i, arg] : llvm::enumerate(func.getArguments())) {
      auto memref = llvm::dyn_cast<mlir::MemRefType>(arg.getType());
      if (!memref)
        continue;
      if (!memref.hasStaticShape() || memref.getRank() > 2) {
        func.emitError("unsupported external memref shape");
        return signalPassFailure();
      }
      MemRefInfo info;
      info.value = arg;
      info.elemWidth = memref.getElementTypeBitWidth();
      for (int64_t dim : memref.getShape()) {
        info.sizes.push_back(dim);
        info.addrSizes.push_back(
            std::max<int64_t>(1, llvm::Log2_64_Ceil(dim)));
      }
      info.mem = cb.createSeqMem("arg" + std::to_string(i), info.elemWidth,
                                 info.sizes, info.addrSizes);
      info.mem->setAttr("external", mlir::IntegerAttr::get(i1, 1));
      ctx.memrefs[arg] = info;
    }

    ctx.control.push_back(topSeq.getBodyBlock());
    BufferLowering bufferLowering(ctx);

    if (mlir::failed(lowerBlock(func.getBody().front(), ctx, bufferLowering))) {
      signalPassFailure();
      return;
    }

    func.erase();
  }
};

} // namespace stream
