#include "Stream/StreamDialect.h"
#include "Stream/StreamDialectOps.h"
#include "Stream/StreamPasses.h"
#include "Support/CalyxBuilder.h"

#include "circt/Dialect/Calyx/CalyxDialect.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace stream {
#define GEN_PASS_DEF_CONVERTSTREAMTOCALYX
#include "Stream/StreamPasses.h.inc"

static std::string name2(const llvm::StringRef base, const int64_t a, const int64_t b) {
  return (base + "_" + std::to_string(a) + "_" + std::to_string(b)).str();
}

static std::string name1(const llvm::StringRef base, const int64_t a) {
  return (base + "_" + std::to_string(a)).str();
}


class GemmBuilder {
public:
  GemmBuilder(mlir::ModuleOp module, Gemm op)
      : module(module), op(op), b(module.getContext()), loc(op.getLoc()),
        cb(b, loc) {}

  mlir::LogicalResult build();

private:
  mlir::LogicalResult validate();
  void createComponent();
  void emitCells();
  void emitAddressWires();
  void emitGroups();
  void emitControl();

  void assign(const mlir::Value dst, const mlir::Value src) {
    cb.assign(dst, src);
  }

  void assignBit(mlir::Value dst, bool v) {
    cb.assignBit(dst, v);
  }
  void enable(const llvm::StringRef group) {
    cb.enable(group);
  }

  void emitGroup(const llvm::StringRef name, const mlir::Value done,
                 const std::function<void()> &body) {
    mlir::OpBuilder::InsertionGuard guard(b);
    auto g = cb.createGroup(name);
    body();
    cb.groupDone(g, done);
  }

  void emitSeq(const std::function<void()> &body) {
    mlir::OpBuilder::InsertionGuard guard(b);
    cb.createSeq();
    body();
  }

  void emitRepeat(uint32_t count, const std::function<void()> &body) {
    mlir::OpBuilder::InsertionGuard guard(b);
    cb.createRepeat(count);
    body();
  }

  mlir::ModuleOp module;
  Gemm op;
  mlir::OpBuilder b;
  mlir::Location loc;
  CalyxBuilder cb;

  int64_t M = 0, N = 0, K = 0;
  int64_t tileM = 0, tileN = 0;
  int64_t addrWA = 0, addrWB = 0, addrWC = 0;
  int64_t shiftA = 0, shiftB = 0;

  mlir::Type i1, i32;
  circt::calyx::ComponentOp comp;
  circt::calyx::WiresOp wires;

  circt::calyx::SeqMemoryOp aMem, bMem, cMem;
  std::map<int64_t, circt::hw::ConstantOp> consts;

  circt::calyx::RegisterOp m0, n0, k;
  circt::calyx::AddLibOp m0Add, n0Add, kAdd;

  llvm::SmallVector<circt::calyx::AddLibOp> aRow;   // tileM, index 0 unused
  llvm::SmallVector<circt::calyx::LshLibOp> aSh;   // tileM
  llvm::SmallVector<circt::calyx::AddLibOp> aAdd;  // tileM
  llvm::SmallVector<circt::calyx::SliceLibOp> aLo; // tileM

  llvm::SmallVector<circt::calyx::AddLibOp> bCol;  // tileN, index 0 unused
  circt::calyx::LshLibOp bSh;
  llvm::SmallVector<circt::calyx::AddLibOp> bAdd;  // tileN
  llvm::SmallVector<circt::calyx::SliceLibOp> bLo; // tileN

  llvm::SmallVector<circt::calyx::AddLibOp> cAdd;  // tileM*tileN
  llvm::SmallVector<circt::calyx::SliceLibOp> cLo; // tileM*tileN

  llvm::SmallVector<circt::calyx::RegisterOp> aReg, bReg;
  llvm::SmallVector<circt::calyx::RegisterOp> acc, sumReg;
  llvm::SmallVector<circt::calyx::MultPipeLibOp> mul;
  llvm::SmallVector<circt::calyx::AddLibOp> sumAdd;

  [[nodiscard]]int64_t at(const int64_t r, const int64_t c) const { return r * tileN + c; }
  circt::hw::ConstantOp cst(const int64_t v) { return consts.at(v); }
};

mlir::LogicalResult GemmBuilder::validate() {
    const auto aTy = mlir::dyn_cast<mlir::MemRefType>(op.getA().getType());
    const auto bTy = mlir::dyn_cast<mlir::MemRefType>(op.getB().getType());
    const auto cTy = mlir::dyn_cast<mlir::MemRefType>(op.getC().getType());

    if (!aTy || !bTy || !cTy) {
      return op.emitError("expected memref operands");
    }
    if (aTy.getRank() != 2 || bTy.getRank() != 2 || cTy.getRank() != 2) {
      return op.emitError("expected rank-2 memrefs");
    }

    M = aTy.getShape()[0];
    K = aTy.getShape()[1];
    N = bTy.getShape()[1];

    if (aTy.isDynamicDim(0) || aTy.isDynamicDim(1) || bTy.isDynamicDim(1) ||
        cTy.isDynamicDim(0) || cTy.isDynamicDim(1)) {
      return op.emitError("expected static shapes");
    }
    if (bTy.getShape()[0] != K || cTy.getShape()[0] != M ||
        cTy.getShape()[1] != N) {
      return op.emitError("incompatible GEMM shapes");
    }

    tileM = static_cast<int64_t>(op.getTileM());
    tileN = static_cast<int64_t>(op.getTileN());

    auto macs = static_cast<int64_t>(op.getMacs());
    auto banksA = static_cast<int64_t>(op.getBanksA());
    auto banksB = static_cast<int64_t>(op.getBanksB());

    if (tileM <= 0 || tileN <= 0 || M % tileM || N % tileN) {
      return op.emitError("tile sizes must divide M and N");
    }

    if (macs != tileM * tileN) {
      return op.emitError("MVP requires macs == tile_m * tile_n");
    }

    if (banksA != 1 || banksB != 1) {
      return op.emitError("MVP supports only one bank per operand");
    }
    if (!llvm::isPowerOf2_64(K) || !llvm::isPowerOf2_64(N)) {
      return op.emitError("MVP requires power-of-two K and N");
    }

    if (aTy.getElementTypeBitWidth() != 32) {
      return op.emitError("MVP supports i32 only");
    }

    addrWA = std::max<int64_t>(1, llvm::Log2_64_Ceil(M * K));
    addrWB = std::max<int64_t>(1, llvm::Log2_64_Ceil(K * N));
    addrWC = std::max<int64_t>(1, llvm::Log2_64_Ceil(M * N));
    shiftA = llvm::Log2_64(K);
    shiftB = llvm::Log2_64(N);
    return mlir::success();
}

void GemmBuilder::createComponent() {
  i1 = b.getI1Type();
  i32 = b.getI32Type();

  auto portAttrs = [&](const llvm::StringRef n) {
    return b.getDictionaryAttr({b.getNamedAttr(n, b.getUnitAttr())});
  };

  const llvm::SmallVector<circt::calyx::PortInfo> ports = {
    {b.getStringAttr("go"), i1, circt::calyx::Direction::Input,
     portAttrs("go")},
    {b.getStringAttr("clk"), i1, circt::calyx::Direction::Input,
     portAttrs("clk")},
    {b.getStringAttr("reset"), i1, circt::calyx::Direction::Input,
     portAttrs("reset")},
    {b.getStringAttr("done"), i1, circt::calyx::Direction::Output,
     portAttrs("done")},
  };

  b.setInsertionPointToStart(module.getBody());

  comp = cb.createComponent(b.getStringAttr("main"), ports);
  wires = cb.getWires();
}

void GemmBuilder::emitCells() {
  b.setInsertionPoint(wires.getOperation());

  // External memories.
  aMem = circt::calyx::SeqMemoryOp::create(
      b, loc, "a", 32, llvm::SmallVector<int64_t>{M * K},
      llvm::SmallVector<int64_t>{addrWA});
  aMem->setAttr("external", mlir::IntegerAttr::get(i1, 1));

  bMem = circt::calyx::SeqMemoryOp::create(
      b, loc, "b", 32, llvm::SmallVector<int64_t>{K * N},
      llvm::SmallVector<int64_t>{addrWB});
  bMem->setAttr("external", mlir::IntegerAttr::get(i1, 1));

  cMem = circt::calyx::SeqMemoryOp::create(
      b, loc, "c", 32, llvm::SmallVector<int64_t>{M * N},
      llvm::SmallVector<int64_t>{addrWC});
  cMem->setAttr("external", mlir::IntegerAttr::get(i1, 1));

  // Constants: 0, 1, tile sizes, shifts, and small offsets.
  std::map<int64_t, bool> values;
  values[0] = values[1] = values[tileM] = values[tileN] = true;
  values[shiftA] = values[shiftB] = true;
  for (int64_t v = 1; v < std::max(tileM, tileN); ++v) {
    values[v] = true;
  }

  for (auto &[v, _] : values) {
    consts[v] = circt::hw::ConstantOp::create(
        b, loc, i32, v);
  }

  // Counters.
  m0 = circt::calyx::RegisterOp::create(b, loc, "m0", 32);
  n0 = circt::calyx::RegisterOp::create(b, loc, "n0", 32);
  k = circt::calyx::RegisterOp::create(b, loc, "k", 32);

  llvm::SmallVector<mlir::Type> t3{i32, i32, i32};
  m0Add = circt::calyx::AddLibOp::create(b, loc, "m0_add", t3);
  n0Add = circt::calyx::AddLibOp::create(b, loc, "n0_add", t3);
  kAdd = circt::calyx::AddLibOp::create(b, loc, "k_add", t3);

  // A address logic: (m0 + r) << shiftA + k.
  aRow.resize(tileM);
  aSh.resize(tileM);
  aAdd.resize(tileM);
  aLo.resize(tileM);

  for (int64_t r = 0; r < tileM; ++r) {
    if (r > 0)
      aRow[r] = circt::calyx::AddLibOp::create(
          b, loc, name1("a_row", r), t3);
    aSh[r] = circt::calyx::LshLibOp::create(
        b, loc, name1("a_sh", r), t3);
    aAdd[r] = circt::calyx::AddLibOp::create(
        b, loc, name1("a_ad", r), t3);
    aLo[r] = circt::calyx::SliceLibOp::create(
        b, loc, name1("a_lo", r),
        llvm::SmallVector<mlir::Type>{i32, b.getIntegerType(addrWA)});
  }

  // B address logic: (k << shiftB) + (n0 + c).
  bSh = circt::calyx::LshLibOp::create(b, loc, "b_sh", t3);
  bCol.resize(tileN);
  bAdd.resize(tileN);
  bLo.resize(tileN);

  for (int64_t c = 0; c < tileN; ++c) {
    if (c > 0)
      bCol[c] = circt::calyx::AddLibOp::create(
          b, loc, name1("b_col", c), t3);
    bAdd[c] = circt::calyx::AddLibOp::create(
        b, loc, name1("b_ad", c), t3);
    bLo[c] = circt::calyx::SliceLibOp::create(
        b, loc, name1("b_lo", c),
        llvm::SmallVector<mlir::Type>{i32, b.getIntegerType(addrWB)});
  }

  // C address logic: (m0 + r) << shiftA + (n0 + c).
  cAdd.resize(tileM * tileN);
  cLo.resize(tileM * tileN);
  for (int64_t r = 0; r < tileM; ++r) {
    for (int64_t c = 0; c < tileN; ++c) {
      cAdd[at(r, c)] = circt::calyx::AddLibOp::create(
          b, loc, name2("c_ad", r, c), t3);
      cLo[at(r, c)] = circt::calyx::SliceLibOp::create(
          b, loc, name2("c_lo", r, c),
          llvm::SmallVector<mlir::Type>{i32, b.getIntegerType(addrWC)});
    }
  }

  // Data registers, accumulators, multipliers, adders.
  aReg.resize(tileM);
  for (int64_t r = 0; r < tileM; ++r)
    aReg[r] = circt::calyx::RegisterOp::create(
        b, loc, name1("a_reg", r), 32);
  bReg.resize(tileN);
  for (int64_t c = 0; c < tileN; ++c)
    bReg[c] = circt::calyx::RegisterOp::create(
        b, loc, name1("b_reg", c), 32);

  acc.resize(tileM * tileN);
  sumReg.resize(tileM * tileN);
  mul.resize(tileM * tileN);
  sumAdd.resize(tileM * tileN);
  for (int64_t r = 0; r < tileM; ++r) {
    for (int64_t c = 0; c < tileN; ++c) {
      int64_t i = at(r, c);
      acc[i] = circt::calyx::RegisterOp::create(
          b, loc, name2("acc", r, c), 32);
      sumReg[i] = circt::calyx::RegisterOp::create(
          b, loc, name2("s_reg", r, c), 32);
      mul[i] = circt::calyx::MultPipeLibOp::create(
          b, loc, name2("m", r, c),
          llvm::SmallVector<mlir::Type>{i1, i1, i1, i32, i32, i32, i1});
      sumAdd[i] = circt::calyx::AddLibOp::create(
          b, loc, name2("s", r, c), t3);
    }
  }
}

void GemmBuilder::emitAddressWires() {
  b.setInsertionPointToEnd(wires.getBodyBlock());

  for (int64_t r = 0; r < tileM; ++r) {
    if (r > 0) {
      assign(aRow[r].getLeft(), m0.getOut());
      assign(aRow[r].getRight(), cst(r).getResult());
    }
    assign(aSh[r].getLeft(), r == 0 ? m0.getOut() : aRow[r].getOut());
    assign(aSh[r].getRight(), cst(shiftA).getResult());
    assign(aAdd[r].getLeft(), aSh[r].getOut());
    assign(aAdd[r].getRight(), k.getOut());
    assign(aLo[r].getIn(), aAdd[r].getOut());
  }

  assign(bSh.getLeft(), k.getOut());
  assign(bSh.getRight(), cst(shiftB).getResult());

  for (int64_t c = 0; c < tileN; ++c) {
    if (c > 0) {
      assign(bCol[c].getLeft(), n0.getOut());
      assign(bCol[c].getRight(), cst(c).getResult());
    }
    assign(bAdd[c].getLeft(), bSh.getOut());
    assign(bAdd[c].getRight(), c == 0 ? n0.getOut() : bCol[c].getOut());
    assign(bLo[c].getIn(), bAdd[c].getOut());
  }

  for (int64_t r = 0; r < tileM; ++r) {
    for (int64_t c = 0; c < tileN; ++c) {
      assign(cAdd[at(r, c)].getLeft(), aSh[r].getOut());
      assign(cAdd[at(r, c)].getRight(),
             c == 0 ? n0.getOut() : bCol[c].getOut());
      assign(cLo[at(r, c)].getIn(), cAdd[at(r, c)].getOut());
    }
  }
}

void GemmBuilder::emitGroups() {
  // init / init_n / init_k
  emitGroup("init", k.getDone(), [&] {
    assign(m0.getIn(), cst(0).getResult());
    assignBit(m0.getWriteEn(), true);
    assign(n0.getIn(), cst(0).getResult());
    assignBit(n0.getWriteEn(), true);
    assign(k.getIn(), cst(0).getResult());
    assignBit(k.getWriteEn(), true);
  });

  emitGroup("init_n", n0.getDone(), [&] {
    assign(n0.getIn(), cst(0).getResult());
    assignBit(n0.getWriteEn(), true);
  });

  emitGroup("init_k", k.getDone(), [&] {
    assign(k.getIn(), cst(0).getResult());
    assignBit(k.getWriteEn(), true);
  });

  emitGroup("incr_m", m0.getDone(), [&] {
    assign(m0Add.getLeft(), m0.getOut());
    assign(m0Add.getRight(), cst(tileM).getResult());
    assign(m0.getIn(), m0Add.getOut());
    assignBit(m0.getWriteEn(), true);
  });

  emitGroup("incr_n", n0.getDone(), [&] {
    assign(n0Add.getLeft(), n0.getOut());
    assign(n0Add.getRight(), cst(tileN).getResult());
    assign(n0.getIn(), n0Add.getOut());
    assignBit(n0.getWriteEn(), true);
  });

  emitGroup("incr_k", k.getDone(), [&] {
    assign(kAdd.getLeft(), k.getOut());
    assign(kAdd.getRight(), cst(1).getResult());
    assign(k.getIn(), kAdd.getOut());
    assignBit(k.getWriteEn(), true);
  });

  emitGroup("zero_acc", acc[0].getDone(), [&] {
    for (auto &r : acc) {
      assign(r.getIn(), cst(0).getResult());
      assignBit(r.getWriteEn(), true);
    }
  });

  // Loads.
  for (int64_t r = 0; r < tileM; ++r) {
    emitGroup(name1("load_a", r), aReg[r].getDone(), [&] {
      assign(aMem.addrPort(0), aLo[r].getOut());
      assignBit(aMem.contentEn(), true);
      assignBit(aMem.writeEn(), false);
      assign(aMem.writeData(), cst(0).getResult());
      assign(aReg[r].getIn(), aMem.readData());
      assign(aReg[r].getWriteEn(), aMem.done());
    });
  }

  for (int64_t c = 0; c < tileN; ++c) {
    emitGroup(name1("load_b", c), bReg[c].getDone(), [&] {
      assign(bMem.addrPort(0), bLo[c].getOut());
      assignBit(bMem.contentEn(), true);
      assignBit(bMem.writeEn(), false);
      assign(bMem.writeData(), cst(0).getResult());
      assign(bReg[c].getIn(), bMem.readData());
      assign(bReg[c].getWriteEn(), bMem.done());
    });
  }

  // Multiply (all products in parallel).
  emitGroup("mac_mult", mul[0].getDone(), [&] {
    for (int64_t r = 0; r < tileM; ++r) {
      for (int64_t c = 0; c < tileN; ++c) {
        int64_t i = at(r, c);
        assignBit(mul[i].getGo(), true);
        assign(mul[i].getLeft(), aReg[r].getOut());
        assign(mul[i].getRight(), bReg[c].getOut());
      }
    }
  });

  // Accumulate (one write per product).
  emitGroup("mac_acc", acc[0].getDone(), [&] {
    for (int64_t r = 0; r < tileM; ++r) {
      for (int64_t c = 0; c < tileN; ++c) {
        int64_t i = at(r, c);
        assign(sumAdd[i].getLeft(), acc[i].getOut());
        assign(sumAdd[i].getRight(), mul[i].getOut());
        assign(acc[i].getIn(), sumAdd[i].getOut());
        assignBit(acc[i].getWriteEn(), true);
      }
    }
  });

  // Stores.
  for (int64_t r = 0; r < tileM; ++r) {
    for (int64_t c = 0; c < tileN; ++c) {
      emitGroup(name2("store_c", r, c), cMem.done(), [&] {
        assign(cMem.addrPort(0), cLo[at(r, c)].getOut());
        assignBit(cMem.contentEn(), true);
        assignBit(cMem.writeEn(), true);
        assign(cMem.writeData(), acc[at(r, c)].getOut());
      });
    }
  }
}

void GemmBuilder::emitControl() {
  b.setInsertionPointToEnd(comp.getControlOp().getBodyBlock());

  emitSeq([&] {
    enable("init");
    emitRepeat(M / tileM, [&] {
      emitSeq([&] {
        enable("init_n");
        emitRepeat(N / tileN, [&] {
          emitSeq([&] {
            enable("zero_acc");
            enable("init_k");
            emitRepeat(K, [&] {
              emitSeq([&] {
                for (int64_t r = 0; r < tileM; ++r)
                  enable(name1("load_a", r));
                for (int64_t c = 0; c < tileN; ++c)
                  enable(name1("load_b", c));
                enable("mac_mult");
                enable("mac_acc");
                enable("incr_k");
              });
            });
            for (int64_t r = 0; r < tileM; ++r)
              for (int64_t c = 0; c < tileN; ++c)
                enable(name2("store_c", r, c));
            enable("incr_n");
          });
        });
        enable("incr_m");
      });
    });
  });
}

mlir::LogicalResult GemmBuilder::build() {
  if (failed(validate())) {
    return llvm::failure();
  }

  createComponent();
  emitCells();
  emitAddressWires();
  emitGroups();
  emitControl();

  return llvm::success();
}

struct ConvertStreamToCalyxPass
      : impl::ConvertStreamToCalyxBase<ConvertStreamToCalyxPass> {
  void runOnOperation() override {
    // mlir::ModuleOp module = getOperation();
    //
    // llvm::SmallVector<mlir::func::FuncOp> funcs;
    //
    // module.walk([&](mlir::func::FuncOp f) {
    //   if (!f.getOps<Gemm>().empty())
    //     funcs.push_back(f);
    // });
    //
    // if (funcs.size() != 1) {
    //   module.emitError("MVP expects exactly one function containing "
    //                    "stream.gemm");
    //   return signalPassFailure();
    // }
    //
    // mlir::func::FuncOp func = funcs.front();
    // Gemm op = *func.getOps<Gemm>().begin();
    //
    // GemmBuilder builder(module, op);
    // if (failed(builder.build())) {
    //   signalPassFailure();
    //   return;
    // }
    //
    // func.erase();
  }
};

} // namespace stream

