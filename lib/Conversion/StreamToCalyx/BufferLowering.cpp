//===- BufferLowering.cpp - Lower stream.buffer ops to Calyx ---------------===//

#include "Conversion/BufferLowering.h"

#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <functional>
#include <numeric>

using namespace mlir;

namespace stream {

static std::optional<BufferType> parseBufferType(const llvm::StringRef kind) {
  if (kind == "seq")
    return BufferType::SEQ;
  if (kind == "comb")
    return BufferType::COMB;
  if (kind == "reg")
    return BufferType::REG;
  return std::nullopt;
}

std::string LoweringContext::fresh(const llvm::StringRef base) {
  return (base + "_" + std::to_string(counter++)).str();
}

void LoweringContext::addEnable(const llvm::StringRef group) {
  assert(!control.empty() && "no active control sequence");
  OpBuilder::InsertionGuard guard(cb.getBuilder());
  cb.getBuilder().setInsertionPointToEnd(control.back());
  cb.enable(group);
}

//===--------------------------------------------------------------------===//
// Alloc
//===--------------------------------------------------------------------===//

LogicalResult BufferLowering::LowerAlloc(BufferAlloc op) {
  auto memref = dyn_cast<MemRefType>(op.getBuffer().getType());
  if (!memref)
    return op.emitError("expected a memref result");
  if (memref.getRank() == 0 || memref.getRank() > 2)
    return op.emitError("only rank-1/2 buffers are supported");
  if (!memref.hasStaticShape())
    return op.emitError("buffers must have static shapes");

  const auto type = parseBufferType(op.getKind());
  if (!type)
    return op.emitError("unknown buffer kind: ") << op.getKind();

  BufferInfo info;
  info.value = op.getBuffer();
  info.type = *type;
  info.elemWidth = memref.getElementTypeBitWidth();

  if (info.elemWidth != 32)
    return op.emitError("only i32 buffers are supported");

  for (const int64_t dim : memref.getShape()) {
    info.sizes.push_back(dim);
    info.addrSizes.push_back(
        std::max<int64_t>(1, llvm::Log2_64_Ceil(dim)));
  }

  const std::string name = ctx.fresh("buf");

  if ( const auto bankDim = static_cast<int64_t>(op.getBankDim()); bankDim >= 0) {
    if (info.type != BufferType::SEQ)
      return op.emitError("banked buffers currently require kind=\"seq\"");

    if (memref.getRank() != 2 || bankDim > 1)
      return op.emitError("banking supports rank-2 buffers with bank_dim 0/1");

    if (static_cast<int64_t>(op.getBanks()) != info.sizes[bankDim])
      return op.emitError("banks must equal the banked dimension size");

    llvm::SmallVector<int64_t> bankSizes{info.sizes[1 - bankDim]};
    llvm::SmallVector<int64_t> bankAddrSizes{info.addrSizes[1 - bankDim]};
    for (int64_t b = 0; b < info.sizes[bankDim]; ++b)
      info.seqBanks.push_back(ctx.cb.createSeqMem(
          name + "_b" + std::to_string(b), info.elemWidth, bankSizes,
          bankAddrSizes));
    info.bankDim = bankDim;

    ctx.buffers[op.getBuffer()] = info;
    return success();
  }

  switch (info.type) {
  case BufferType::SEQ:
    info.seqMem = ctx.cb.createSeqMem(name, info.elemWidth, info.sizes,
                                      info.addrSizes);
    break;
  case BufferType::COMB:
    info.combMem = ctx.cb.createCombMem(name, info.elemWidth, info.sizes,
                                        info.addrSizes);
    break;
  case BufferType::REG: {
    const int64_t elements = std::accumulate(info.sizes.begin(), info.sizes.end(),
                                       int64_t{1}, std::multiplies<>());
    if (elements != 1)
      return op.emitError("reg buffers currently support exactly one element");
    info.reg = ctx.cb.createRegister(name, info.elemWidth);
    break;
  }
  }

  ctx.buffers[op.getBuffer()] = info;
  return success();
}

//===--------------------------------------------------------------------===//
// Read
//===--------------------------------------------------------------------===//


static Value materializeI32(const LoweringContext &ctx, const Value value) {
  if (const Value mapped = ctx.lookup(value))
    return mapped;

  if (auto cst = value.getDefiningOp<arith::ConstantOp>()) {
    if (const auto attr = dyn_cast<IntegerAttr>(cst.getValue()))
      return ctx.cb.i32(attr.getInt());
  }
  return value;
}

// Reads from a calyx seq mem into a calyx register.
LogicalResult BufferLowering::LowerView(BufferView op) {
  auto it = ctx.buffers.find(op.getSource());
  if (it == ctx.buffers.end())
    return op.emitError("view source must be an on-chip buffer");

  BufferInfo info = it->second;
  if (info.type == BufferType::REG)
    return op.emitError("buffer.view does not support reg buffers");

  if (!info.seqBanks.empty())
    return op.emitError("buffer.view does not support banked sources yet");

  if (!info.viewOffsets.empty())
    return op.emitError("buffer.view of another view is not supported yet");

  const auto srcTy = dyn_cast<MemRefType>(op.getSource().getType());
  const auto dstTy = dyn_cast<MemRefType>(op.getResult().getType());

  if (!srcTy || !dstTy || srcTy.getRank() != dstTy.getRank() ||
      dstTy.getRank() == 0 || dstTy.getRank() > 2)
    return op.emitError("buffer.view expects matching rank-1/2 memrefs");

  if (!dstTy.hasStaticShape())
    return op.emitError("buffer.view result must have a static shape");

  if (op.getOffsets().size() != dstTy.getRank())
    return op.emitError("buffer.view offset count must match rank");

  info.value = op.getResult();
  info.sizes.clear();
  // Keep the parent memory's port widths; the view only changes logical shape.
  for (const int64_t dim : dstTy.getShape())
    info.sizes.push_back(dim);

  info.viewOffsets.clear();
  for (const Value offset : op.getOffsets())
    info.viewOffsets.push_back(materializeI32(ctx, offset));

  ctx.buffers[op.getResult()] = info;
  // Keep the view op alive: its result is used by later ops in this block.
  return success();
}

LogicalResult BufferLowering::LowerRead(BufferRead op) {
  auto it = ctx.buffers.find(op.getBuffer());
  if (it == ctx.buffers.end())
    return op.emitError("trying to read from an unallocated buffer");

  BufferInfo &info = it->second;
  if (!info.seqBanks.empty())
    return op.emitError("banked buffers do not support scalar buffer.read");
  if (!info.viewOffsets.empty())
    return op.emitError("buffer.view does not support scalar buffer.read");

  if (info.type == BufferType::REG) {
    op.getValue().replaceAllUsesWith(info.reg.getOut());
    op.erase();
    return success();
  }

  if (op.getIndices().size() != info.addrSizes.size())
    return op.emitError("wrong number of buffer indices");

  OpBuilder &b = ctx.cb.getBuilder();
  const Type i32 = b.getI32Type();

  llvm::SmallVector<Value> indices;
  for (const Value idx : op.getIndices())
    indices.push_back(materializeI32(ctx, idx));

  llvm::SmallVector<circt::calyx::SliceLibOp> slices;

  for (auto [idx, addrSize] : llvm::zip(indices, info.addrSizes)) {
    const auto slice = ctx.cb.createSlice(
        ctx.fresh("idx"), i32, b.getIntegerType(addrSize));
    slices.push_back(slice);
  }

  // We'll read the data into a calyx register
  auto reg = ctx.cb.createRegister(ctx.fresh("rd"), info.elemWidth);
  const std::string groupName = ctx.fresh("read_buf");

  {
    OpBuilder::InsertionGuard guard(b);
    const auto group = ctx.cb.createGroup(groupName);

    for (size_t i = 0; i < slices.size(); ++i) {
      ctx.cb.assign(slices[i].getIn(), indices[i]);
      const Value port = info.type == BufferType::SEQ
                       ? info.seqMem.addrPort(i)
                       : info.combMem.addrPort(i);
      ctx.cb.assign(port, slices[i].getOut());
    }

    if (info.type == BufferType::SEQ) {
      ctx.cb.assignBit(info.seqMem.contentEn(), true);
      ctx.cb.assignBit(info.seqMem.writeEn(), false);
      ctx.cb.assign(info.seqMem.writeData(), ctx.cb.i32(0));
      ctx.cb.assign(reg.getIn(), info.seqMem.readData());
      ctx.cb.assign(reg.getWriteEn(), info.seqMem.done());
    } else {
      ctx.cb.assignBit(info.combMem.writeEn(), false);
      ctx.cb.assign(info.combMem.writeData(), ctx.cb.i32(0));
      ctx.cb.assign(reg.getIn(), info.combMem.readData());
      ctx.cb.assignBit(reg.getWriteEn(), true);
    }

    ctx.cb.groupDone(group, reg.getDone());
  }

  ctx.addEnable(groupName);
  op.getValue().replaceAllUsesWith(reg.getOut());
  op.erase();
  return success();
}

//===--------------------------------------------------------------------===//
// Write
//===--------------------------------------------------------------------===//

LogicalResult BufferLowering::emitWrite(BufferInfo &info,
                                        llvm::ArrayRef<Value> indices,
                                        const Value value) {
  OpBuilder &b = ctx.cb.getBuilder();
  const std::string groupName = ctx.fresh("write_buf");

  if (info.type == BufferType::REG) {
    {
      OpBuilder::InsertionGuard guard(b);
      const auto group = ctx.cb.createGroup(groupName);
      ctx.cb.assign(info.reg.getIn(), value);
      ctx.cb.assignBit(info.reg.getWriteEn(), true);
      ctx.cb.groupDone(group, info.reg.getDone());
    }
    ctx.addEnable(groupName);
    return success();
  }

  if (indices.size() != info.addrSizes.size())
    return failure();

  // TODO: support other types later ...
  const Type i32 = b.getI32Type();
  llvm::SmallVector<circt::calyx::SliceLibOp> slices;
  for (auto [idx, addrSize] : llvm::zip(indices, info.addrSizes)) {
    const auto slice = ctx.cb.createSlice(
        ctx.fresh("idx"), i32, b.getIntegerType(addrSize));
    slices.push_back(slice);
  }

  {
    OpBuilder::InsertionGuard guard(b);
    const auto group = ctx.cb.createGroup(groupName);

    for (size_t i = 0; i < slices.size(); ++i) {
      ctx.cb.assign(slices[i].getIn(), indices[i]);
      const Value port = info.type == BufferType::SEQ
                       ? info.seqMem.addrPort(i)
                       : info.combMem.addrPort(i);
      ctx.cb.assign(port, slices[i].getOut());
    }

    if (info.type == BufferType::SEQ) {
      ctx.cb.assignBit(info.seqMem.contentEn(), true);
      ctx.cb.assign(info.seqMem.writeData(), value);
      ctx.cb.assignBit(info.seqMem.writeEn(), true);
      ctx.cb.groupDone(group, info.seqMem.done());
    } else {
      ctx.cb.assign(info.combMem.writeData(), value);
      ctx.cb.assignBit(info.combMem.writeEn(), true);
      ctx.cb.groupDone(group, info.combMem.done());
    }
  }

  ctx.addEnable(groupName);
  return success();
}

LogicalResult BufferLowering::LowerWrite(BufferWrite op) {
  auto it = ctx.buffers.find(op.getBuffer());
  if (it == ctx.buffers.end())
    return op.emitError("trying to write to a buffer that was not allocated");
  if (!it->second.seqBanks.empty())
    return op.emitError("banked buffers do not support scalar buffer.write");
  if (!it->second.viewOffsets.empty())
    return op.emitError("buffer.view does not support scalar buffer.write");

  llvm::SmallVector<Value> indices;
  for (const Value idx : op.getIndices())
    indices.push_back(materializeI32(ctx, idx));

  if (const Value value = materializeI32(ctx, op.getValue()); failed(emitWrite(it->second, indices, value)))
    return op.emitError("cannot lower buffer write");

  op.erase();
  return success();
}

//===--------------------------------------------------------------------===//
// Zero
//===--------------------------------------------------------------------===//


//===--------------------------------------------------------------------===//
// MAC array
//===--------------------------------------------------------------------===//

LogicalResult BufferLowering::LowerStaticBankedGemmTile(
    GemmTile op, BufferInfo &a, BufferInfo &b, BufferInfo &c, int64_t M,
    int64_t K, int64_t N) {
  if (K <= 0 || a.seqBanks.size() != static_cast<size_t>(M) ||
      b.seqBanks.size() != static_cast<size_t>(N))
    return failure();

  OpBuilder &builder = ctx.cb.getBuilder();
  Type i32 = builder.getI32Type();

  llvm::SmallVector<circt::calyx::RegisterOp> acc, aTmp, bTmp;
  llvm::SmallVector<circt::calyx::MultPipeLibOp> mul;
  llvm::SmallVector<circt::calyx::AddLibOp> sumAdd;
  for (int64_t m = 0; m < M; ++m)
    aTmp.push_back(ctx.cb.createRegister(ctx.fresh("sga"), 32));
  for (int64_t n = 0; n < N; ++n)
    bTmp.push_back(ctx.cb.createRegister(ctx.fresh("sgb"), 32));
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      acc.push_back(ctx.cb.createRegister(ctx.fresh("sgacc"), 32));
      mul.push_back(ctx.cb.createMultPipe(ctx.fresh("sgmul"), i32));
      sumAdd.push_back(ctx.cb.createAdd(ctx.fresh("sgsum"), i32));
    }
  }

  OpBuilder::InsertionGuard outerGuard(builder);
  builder.setInsertionPointToEnd(ctx.control.back());
  auto wrapper = ctx.cb.createStaticSeq();
  ctx.control.push_back(wrapper.getBodyBlock());

  // Zero accumulators.
  {
    std::string name = ctx.fresh("sg_zero");
    {
      OpBuilder::InsertionGuard guard(builder);
      auto group = ctx.cb.createStaticGroup(name, 1);
      for (auto &reg : acc) {
        ctx.cb.assign(reg.getIn(), ctx.cb.i32(0));
        ctx.cb.assignBit(reg.getWriteEn(), true);
      }
    }
    ctx.addEnable(name);
  }

  auto k = ctx.cb.createRegister(ctx.fresh("sg_k"), 32);
  {
    std::string name = ctx.fresh("sg_init_k");
    {
      OpBuilder::InsertionGuard guard(builder);
      auto group = ctx.cb.createStaticGroup(name, 1);
      ctx.cb.assign(k.getIn(), ctx.cb.i32(0));
      ctx.cb.assignBit(k.getWriteEn(), true);
    }
    ctx.addEnable(name);
  }

  auto repeat = ctx.cb.createStaticRepeat(static_cast<uint32_t>(K));
  ctx.control.push_back(repeat.getBodyBlock());
  auto body = ctx.cb.createStaticSeq();
  ctx.control.push_back(body.getBodyBlock());

  // Read A row banks and B column banks in parallel.
  {
    std::string name = ctx.fresh("sg_read");
    auto aSlice = ctx.cb.createSlice(
        ctx.fresh("sg_ai"), i32, builder.getIntegerType(a.addrSizes[1]));
    auto bSlice = ctx.cb.createSlice(
        ctx.fresh("sg_bi"), i32, builder.getIntegerType(b.addrSizes[0]));
    {
      OpBuilder::InsertionGuard guard(builder);
      auto group = ctx.cb.createStaticGroup(name, 2);
      ctx.cb.assign(aSlice.getIn(), k.getOut());
      ctx.cb.assign(bSlice.getIn(), k.getOut());
      for (int64_t m = 0; m < M; ++m) {
        auto bank = a.seqBanks[m];
        ctx.cb.assign(bank.addrPort(0), aSlice.getOut());
        ctx.cb.assignBit(bank.contentEn(), true);
        ctx.cb.assignBit(bank.writeEn(), false);
        ctx.cb.assign(bank.writeData(), ctx.cb.i32(0));
        ctx.cb.assign(aTmp[m].getIn(), bank.readData());
        ctx.cb.assignBit(aTmp[m].getWriteEn(), true);
      }
      for (int64_t n = 0; n < N; ++n) {
        auto bank = b.seqBanks[n];
        ctx.cb.assign(bank.addrPort(0), bSlice.getOut());
        ctx.cb.assignBit(bank.contentEn(), true);
        ctx.cb.assignBit(bank.writeEn(), false);
        ctx.cb.assign(bank.writeData(), ctx.cb.i32(0));
        ctx.cb.assign(bTmp[n].getIn(), bank.readData());
        ctx.cb.assignBit(bTmp[n].getWriteEn(), true);
      }
    }
    ctx.addEnable(name);
  }

  // Launch all multipliers and hold them for their 3-cycle latency.
  {
    std::string name = ctx.fresh("sg_mult");
    {
      OpBuilder::InsertionGuard guard(builder);
      auto group = ctx.cb.createStaticGroup(name, 3);
      for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
          int64_t i = m * N + n;
          ctx.cb.assignBit(mul[i].getGo(), true);
          ctx.cb.assign(mul[i].getLeft(), aTmp[m].getOut());
          ctx.cb.assign(mul[i].getRight(), bTmp[n].getOut());
        }
      }
    }
    ctx.addEnable(name);
  }

  {
    std::string name = ctx.fresh("sg_acc");
    {
      OpBuilder::InsertionGuard guard(builder);
      auto group = ctx.cb.createStaticGroup(name, 1);
      for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
          int64_t i = m * N + n;
          ctx.cb.assign(sumAdd[i].getLeft(), acc[i].getOut());
          ctx.cb.assign(sumAdd[i].getRight(), mul[i].getOut());
          ctx.cb.assign(acc[i].getIn(), sumAdd[i].getOut());
          ctx.cb.assignBit(acc[i].getWriteEn(), true);
        }
      }
    }
    ctx.addEnable(name);
  }

  {
    std::string name = ctx.fresh("sg_incr_k");
    auto add = ctx.cb.createAdd(ctx.fresh("sg_kadd"), i32);
    {
      OpBuilder::InsertionGuard guard(builder);
      auto group = ctx.cb.createStaticGroup(name, 1);
      ctx.cb.assign(add.getLeft(), k.getOut());
      ctx.cb.assign(add.getRight(), ctx.cb.i32(1));
      ctx.cb.assign(k.getIn(), add.getOut());
      ctx.cb.assignBit(k.getWriteEn(), true);
    }
    ctx.addEnable(name);
  }

  ctx.control.pop_back(); // body
  ctx.control.pop_back(); // repeat

  // Store the C tile.
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      std::string name = ctx.fresh("sg_store");
      llvm::SmallVector<circt::calyx::SliceLibOp> slices;
      for (size_t d = 0; d < c.sizes.size(); ++d)
        slices.push_back(ctx.cb.createSlice(
            ctx.fresh("sg_st"), i32,
            builder.getIntegerType(c.addrSizes[d])));
      {
        OpBuilder::InsertionGuard guard(builder);
        auto group = ctx.cb.createStaticGroup(name, 1);
        ctx.cb.assign(slices[0].getIn(), ctx.cb.i32(m));
        ctx.cb.assign(slices[1].getIn(), ctx.cb.i32(n));
        Value addr0 = c.type == BufferType::SEQ ? c.seqMem.addrPort(0)
                                                : c.combMem.addrPort(0);
        Value addr1 = c.type == BufferType::SEQ ? c.seqMem.addrPort(1)
                                                : c.combMem.addrPort(1);
        ctx.cb.assign(addr0, slices[0].getOut());
        ctx.cb.assign(addr1, slices[1].getOut());
        if (c.type == BufferType::SEQ) {
          ctx.cb.assignBit(c.seqMem.contentEn(), true);
          ctx.cb.assign(c.seqMem.writeData(), acc[m * N + n].getOut());
          ctx.cb.assignBit(c.seqMem.writeEn(), true);
        } else {
          ctx.cb.assign(c.combMem.writeData(), acc[m * N + n].getOut());
          ctx.cb.assignBit(c.combMem.writeEn(), true);
        }
      }
      ctx.addEnable(name);
    }
  }

  ctx.control.pop_back(); // wrapper
  op.erase();
  return success();
}

LogicalResult BufferLowering::LowerGemmTile(GemmTile op) {
  auto a_it = ctx.buffers.find(op.getATile());
  auto b_it = ctx.buffers.find(op.getBTile());
  auto c_it = ctx.buffers.find(op.getCTile());

  // Mac op can only be performed on allocated buffers.
  if (a_it == ctx.buffers.end() || b_it == ctx.buffers.end() ||
      c_it == ctx.buffers.end())
    return op.emitError("gemm_tile operands must be on-chip buffers");

  BufferInfo &a = a_it->second;
  BufferInfo &b = b_it->second;
  BufferInfo &c = c_it->second;

  if (a.type == BufferType::REG || b.type == BufferType::REG ||
      c.type == BufferType::REG)
    return op.emitError("gemm_tile does not support reg buffers yet");

  if (!c.seqBanks.empty())
    return op.emitError("gemm_tile output buffer must not be banked");

  if (a.sizes.size() != 2 || b.sizes.size() != 2 || c.sizes.size() != 2)
    return op.emitError("gemm_tile expects rank-2 tiles");

  int64_t M = a.sizes[0];
  int64_t K = a.sizes[1];
  int64_t N = b.sizes[1];

  // Check if matmul is possible or not.
  if (b.sizes[0] != K || c.sizes[0] != M || c.sizes[1] != N ||
      static_cast<int64_t>(op.getK()) != K)
    return op.emitError("gemm_tile tile shape mismatch");

  bool aBanked = !a.seqBanks.empty();
  bool bBanked = !b.seqBanks.empty();
  if (aBanked != bBanked)
    return op.emitError("gemm_tile requires both A and B banked, or neither");
  if (aBanked &&
      (a.bankDim != 0 || b.bankDim != 1 ||
       static_cast<int64_t>(a.seqBanks.size()) != M ||
       static_cast<int64_t>(b.seqBanks.size()) != N))
    return op.emitError(
        "gemm_tile expects A banked by row and B banked by column");

  if (aBanked && bBanked && op.getZeroInit() && c.seqBanks.empty() &&
      c.viewOffsets.empty())
    return LowerStaticBankedGemmTile(op, a, b, c, M, K, N);

  OpBuilder &builder = ctx.cb.getBuilder();
  Type i32 = builder.getI32Type();

  llvm::SmallVector<circt::calyx::RegisterOp> acc;
  llvm::SmallVector<circt::calyx::MultPipeLibOp> mul;
  llvm::SmallVector<circt::calyx::AddLibOp> sumAdd;

  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      acc.push_back(ctx.cb.createRegister(ctx.fresh("mac_acc"), 32));
      mul.push_back(ctx.cb.createMultPipe(ctx.fresh("mac_mul"), i32));
      sumAdd.push_back(ctx.cb.createAdd(ctx.fresh("mac_sum"), i32));
    }
  }

  auto k = ctx.cb.createRegister(ctx.fresh("mac_k"), 32);
  auto kNext = ctx.cb.createRegister(ctx.fresh("mac_k_next"), 32);
  std::string initName = ctx.fresh("mac_init_k");
  {
    OpBuilder::InsertionGuard guard(builder);
    auto group = ctx.cb.createGroup(initName);
    ctx.cb.assign(k.getIn(), ctx.cb.i32(0));
    ctx.cb.assignBit(k.getWriteEn(), true);
    ctx.cb.assign(kNext.getIn(), ctx.cb.i32(1));
    ctx.cb.assignBit(kNext.getWriteEn(), true);
    ctx.cb.groupDone(group, k.getDone());
  }
  ctx.addEnable(initName);

  auto makeRegs = [&](llvm::SmallVector<circt::calyx::RegisterOp> &regs,
                      int64_t count, llvm::StringRef base) {
    for (int64_t i = 0; i < count; ++i)
      regs.push_back(ctx.cb.createRegister(ctx.fresh(base), 32));
  };

  llvm::SmallVector<circt::calyx::RegisterOp> aCur, aNext, bCur, bNext;
  makeRegs(aCur, M, "mac_a_cur");
  makeRegs(aNext, M, "mac_a_next");
  makeRegs(bCur, N, "mac_b_cur");
  makeRegs(bNext, N, "mac_b_next");

  auto readElement = [&](BufferInfo &buf, llvm::ArrayRef<Value> addresses,
                         circt::calyx::RegisterOp dest,
                         llvm::StringRef base) {
    llvm::SmallVector<circt::calyx::SliceLibOp> slices;
    llvm::SmallVector<circt::calyx::AddLibOp> viewAdds;
    bool hasView = !buf.viewOffsets.empty();
    for (size_t d = 0; d < addresses.size(); ++d) {
      slices.push_back(ctx.cb.createSlice(
          ctx.fresh("mac_idx"), i32,
          builder.getIntegerType(buf.addrSizes[d])));
      if (hasView)
        viewAdds.push_back(ctx.cb.createAdd(ctx.fresh("mac_view"), i32));
    }

    std::string groupName = ctx.fresh(base);
    {
      OpBuilder::InsertionGuard guard(builder);
      auto group = ctx.cb.createGroup(groupName);
      for (size_t d = 0; d < addresses.size(); ++d) {
        if (hasView) {
          ctx.cb.assign(viewAdds[d].getLeft(), buf.viewOffsets[d]);
          ctx.cb.assign(viewAdds[d].getRight(), addresses[d]);
          ctx.cb.assign(slices[d].getIn(), viewAdds[d].getOut());
        } else {
          ctx.cb.assign(slices[d].getIn(), addresses[d]);
        }
        Value port = buf.type == BufferType::SEQ
                         ? buf.seqMem.addrPort(d)
                         : buf.combMem.addrPort(d);
        ctx.cb.assign(port, slices[d].getOut());
      }
      if (buf.type == BufferType::SEQ) {
        ctx.cb.assignBit(buf.seqMem.contentEn(), true);
        ctx.cb.assignBit(buf.seqMem.writeEn(), false);
        ctx.cb.assign(buf.seqMem.writeData(), ctx.cb.i32(0));
        ctx.cb.assign(dest.getIn(), buf.seqMem.readData());
        ctx.cb.assign(dest.getWriteEn(), buf.seqMem.done());
      } else {
        ctx.cb.assignBit(buf.combMem.writeEn(), false);
        ctx.cb.assign(buf.combMem.writeData(), ctx.cb.i32(0));
        ctx.cb.assign(dest.getIn(), buf.combMem.readData());
        ctx.cb.assignBit(dest.getWriteEn(), true);
      }
      ctx.cb.groupDone(group, dest.getDone());
    }
    ctx.addEnable(groupName);
  };

  if (op.getZeroInit()) {
    std::string zeroName = ctx.fresh("mac_zero");
    {
      OpBuilder::InsertionGuard guard(builder);
      auto group = ctx.cb.createGroup(zeroName);
      for (auto &reg : acc) {
        ctx.cb.assign(reg.getIn(), ctx.cb.i32(0));
        ctx.cb.assignBit(reg.getWriteEn(), true);
      }
      ctx.cb.groupDone(group, acc.front().getDone());
    }
    ctx.addEnable(zeroName);
  } else {
    auto cTmp = ctx.cb.createRegister(ctx.fresh("mac_c_tmp"), 32);
    for (int64_t m = 0; m < M; ++m) {
      for (int64_t n = 0; n < N; ++n) {
        readElement(c, {ctx.cb.i32(m), ctx.cb.i32(n)}, cTmp, "mac_init_c");
        int64_t i = m * N + n;
        std::string groupName = ctx.fresh("mac_init_acc");
        {
          OpBuilder::InsertionGuard guard(builder);
          auto group = ctx.cb.createGroup(groupName);
          ctx.cb.assign(acc[i].getIn(), cTmp.getOut());
          ctx.cb.assignBit(acc[i].getWriteEn(), true);
          ctx.cb.groupDone(group, acc[i].getDone());
        }
        ctx.addEnable(groupName);
      }
    }
  }

  auto readAll = [&](llvm::MutableArrayRef<circt::calyx::RegisterOp> aRegs,
                     llvm::MutableArrayRef<circt::calyx::RegisterOp> bRegs,
                     Value kValue, llvm::StringRef tag) {
    if (!a.seqBanks.empty() && !b.seqBanks.empty()) {
      auto aSlice = ctx.cb.createSlice(
          ctx.fresh("mac_bank_a"), i32,
          builder.getIntegerType(a.addrSizes[1]));
      auto bSlice = ctx.cb.createSlice(
          ctx.fresh("mac_bank_b"), i32,
          builder.getIntegerType(b.addrSizes[0]));
      std::string groupName = ctx.fresh(tag);
      {
        OpBuilder::InsertionGuard guard(builder);
        auto group = ctx.cb.createGroup(groupName);
        ctx.cb.assign(aSlice.getIn(), kValue);
        ctx.cb.assign(bSlice.getIn(), kValue);
        for (int64_t m = 0; m < M; ++m) {
          auto bank = a.seqBanks[m];
          ctx.cb.assign(bank.addrPort(0), aSlice.getOut());
          ctx.cb.assignBit(bank.contentEn(), true);
          ctx.cb.assignBit(bank.writeEn(), false);
          ctx.cb.assign(bank.writeData(), ctx.cb.i32(0));
          ctx.cb.assign(aRegs[m].getIn(), bank.readData());
          ctx.cb.assign(aRegs[m].getWriteEn(), bank.done());
        }
        for (int64_t n = 0; n < N; ++n) {
          auto bank = b.seqBanks[n];
          ctx.cb.assign(bank.addrPort(0), bSlice.getOut());
          ctx.cb.assignBit(bank.contentEn(), true);
          ctx.cb.assignBit(bank.writeEn(), false);
          ctx.cb.assign(bank.writeData(), ctx.cb.i32(0));
          ctx.cb.assign(bRegs[n].getIn(), bank.readData());
          ctx.cb.assign(bRegs[n].getWriteEn(), bank.done());
        }
        ctx.cb.groupDone(group, aRegs.front().getDone());
      }
      ctx.addEnable(groupName);
      return;
    }

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(ctx.control.back());
    auto seq = ctx.cb.createSeq();
    ctx.control.push_back(seq.getBodyBlock());
    for (int64_t m = 0; m < M; ++m)
      readElement(a, {ctx.cb.i32(m), kValue}, aRegs[m],
                  (tag + "_a").str());
    for (int64_t n = 0; n < N; ++n)
      readElement(b, {kValue, ctx.cb.i32(n)}, bRegs[n],
                  (tag + "_b").str());
    ctx.control.pop_back();
  };

  auto emitMult = [&](llvm::MutableArrayRef<circt::calyx::RegisterOp> aRegs,
                      llvm::MutableArrayRef<circt::calyx::RegisterOp> bRegs,
                      llvm::StringRef tag) {
    std::string groupName = ctx.fresh(tag);
    {
      OpBuilder::InsertionGuard guard(builder);
      auto group = ctx.cb.createGroup(groupName);
      for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
          int64_t i = m * N + n;
          ctx.cb.assignBit(mul[i].getGo(), true);
          ctx.cb.assign(mul[i].getLeft(), aRegs[m].getOut());
          ctx.cb.assign(mul[i].getRight(), bRegs[n].getOut());
        }
      }
      ctx.cb.groupDone(group, mul.front().getDone());
    }
    ctx.addEnable(groupName);
  };

  auto emitAcc = [&](llvm::StringRef tag) {
    std::string groupName = ctx.fresh(tag);
    {
      OpBuilder::InsertionGuard guard(builder);
      auto group = ctx.cb.createGroup(groupName);
      for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
          int64_t i = m * N + n;
          ctx.cb.assign(sumAdd[i].getLeft(), acc[i].getOut());
          ctx.cb.assign(sumAdd[i].getRight(), mul[i].getOut());
          ctx.cb.assign(acc[i].getIn(), sumAdd[i].getOut());
          ctx.cb.assignBit(acc[i].getWriteEn(), true);
        }
      }
      ctx.cb.groupDone(group, acc.front().getDone());
    }
    ctx.addEnable(groupName);
  };

  auto emitCopy = [&](llvm::MutableArrayRef<circt::calyx::RegisterOp> dst,
                      llvm::MutableArrayRef<circt::calyx::RegisterOp> src,
                      llvm::StringRef tag) {
    std::string groupName = ctx.fresh(tag);
    {
      OpBuilder::InsertionGuard guard(builder);
      auto group = ctx.cb.createGroup(groupName);
      for (size_t i = 0; i < dst.size(); ++i) {
        ctx.cb.assign(dst[i].getIn(), src[i].getOut());
        ctx.cb.assignBit(dst[i].getWriteEn(), true);
      }
      ctx.cb.groupDone(group, dst.front().getDone());
    }
    ctx.addEnable(groupName);
  };

  // Prologue: load k = 0.
  readAll(aCur, bCur, k.getOut(), "mac_read_cur");

  if (K > 1) {
    OpBuilder::InsertionGuard loopGuard(builder);
    builder.setInsertionPointToEnd(ctx.control.back());
    ctx.cb.createRepeat(static_cast<uint32_t>(K - 1));
    auto loopSeq = ctx.cb.createSeq();
    ctx.control.push_back(loopSeq.getBodyBlock());

    // Stage 1: multiply k while reading k+1.
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToEnd(ctx.control.back());
      auto par = ctx.cb.createPar();
      ctx.control.push_back(par.getBodyBlock());
      emitMult(aCur, bCur, "mac_mult");
      readAll(aNext, bNext, kNext.getOut(), "mac_read_next");
      ctx.control.pop_back();
    }

    // Stage 2: accumulate k while moving k+1 into the current registers.
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToEnd(ctx.control.back());
      auto par = ctx.cb.createPar();
      ctx.control.push_back(par.getBodyBlock());
      emitAcc("mac_acc");
      emitCopy(aCur, aNext, "mac_copy_a");
      emitCopy(bCur, bNext, "mac_copy_b");
      ctx.control.pop_back();
    }

    auto kAdd = ctx.cb.createAdd(ctx.fresh("mac_k_add"), i32);
    auto kNextAdd = ctx.cb.createAdd(ctx.fresh("mac_kn_add"), i32);
    std::string incrName = ctx.fresh("mac_incr_k");
    {
      OpBuilder::InsertionGuard guard(builder);
      auto group = ctx.cb.createGroup(incrName);
      ctx.cb.assign(kAdd.getLeft(), k.getOut());
      ctx.cb.assign(kAdd.getRight(), ctx.cb.i32(1));
      ctx.cb.assign(k.getIn(), kAdd.getOut());
      ctx.cb.assignBit(k.getWriteEn(), true);
      ctx.cb.assign(kNextAdd.getLeft(), kNext.getOut());
      ctx.cb.assign(kNextAdd.getRight(), ctx.cb.i32(1));
      ctx.cb.assign(kNext.getIn(), kNextAdd.getOut());
      ctx.cb.assignBit(kNext.getWriteEn(), true);
      ctx.cb.groupDone(group, k.getDone());
    }
    ctx.addEnable(incrName);

    ctx.control.pop_back();
  }

  // Last k: multiply and accumulate.
  emitMult(aCur, bCur, "mac_final_mult");
  emitAcc("mac_final_acc");

  // Store the tile.
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      llvm::SmallVector<circt::calyx::SliceLibOp> slices;
      llvm::SmallVector<circt::calyx::AddLibOp> viewAdds;
      bool hasView = !c.viewOffsets.empty();
      for (size_t d = 0; d < c.sizes.size(); ++d) {
        slices.push_back(ctx.cb.createSlice(
            ctx.fresh("mac_st"), i32,
            builder.getIntegerType(c.addrSizes[d])));
        if (hasView)
          viewAdds.push_back(ctx.cb.createAdd(ctx.fresh("mac_stv"), i32));
      }

      std::string storeName = ctx.fresh("mac_store");
      {
        OpBuilder::InsertionGuard guard(builder);
        auto group = ctx.cb.createGroup(storeName);
        if (hasView) {
          ctx.cb.assign(viewAdds[0].getLeft(), c.viewOffsets[0]);
          ctx.cb.assign(viewAdds[0].getRight(), ctx.cb.i32(m));
          ctx.cb.assign(slices[0].getIn(), viewAdds[0].getOut());
          ctx.cb.assign(viewAdds[1].getLeft(), c.viewOffsets[1]);
          ctx.cb.assign(viewAdds[1].getRight(), ctx.cb.i32(n));
          ctx.cb.assign(slices[1].getIn(), viewAdds[1].getOut());
        } else {
          ctx.cb.assign(slices[0].getIn(), ctx.cb.i32(m));
          ctx.cb.assign(slices[1].getIn(), ctx.cb.i32(n));
        }
        Value addr0 = c.type == BufferType::SEQ ? c.seqMem.addrPort(0)
                                                : c.combMem.addrPort(0);
        Value addr1 = c.type == BufferType::SEQ ? c.seqMem.addrPort(1)
                                                : c.combMem.addrPort(1);
        ctx.cb.assign(addr0, slices[0].getOut());
        ctx.cb.assign(addr1, slices[1].getOut());
        if (c.type == BufferType::SEQ) {
          ctx.cb.assignBit(c.seqMem.contentEn(), true);
          ctx.cb.assign(c.seqMem.writeData(), acc[m * N + n].getOut());
          ctx.cb.assignBit(c.seqMem.writeEn(), true);
          ctx.cb.groupDone(group, c.seqMem.done());
        } else {
          ctx.cb.assign(c.combMem.writeData(), acc[m * N + n].getOut());
          ctx.cb.assignBit(c.combMem.writeEn(), true);
          ctx.cb.groupDone(group, c.combMem.done());
        }
      }
      ctx.addEnable(storeName);
    }
  }

  op.erase();
  return success();
}

//===--------------------------------------------------------------------===//
// Tile load / store
//===--------------------------------------------------------------------===//

LogicalResult BufferLowering::emitCopyBody(
    BufferInfo &info, MemRefInfo &ext, llvm::ArrayRef<Value> offsets,
    bool isLoad, llvm::MutableArrayRef<circt::calyx::RegisterOp> counters) {
  OpBuilder &b = ctx.cb.getBuilder();
  Type i32 = b.getI32Type();
  auto tmp = ctx.cb.createRegister(ctx.fresh(isLoad ? "ld_tmp" : "st_tmp"),
                                   32);

  llvm::SmallVector<circt::calyx::SliceLibOp> bufSlices;
  llvm::SmallVector<circt::calyx::AddLibOp> extAdds;
  llvm::SmallVector<circt::calyx::SliceLibOp> extSlices;
  for (size_t d = 0; d < info.sizes.size(); ++d) {
    bufSlices.push_back(ctx.cb.createSlice(
        ctx.fresh("b_idx"), i32, b.getIntegerType(info.addrSizes[d])));
    extAdds.push_back(ctx.cb.createAdd(ctx.fresh("e_ad"), i32));
    extSlices.push_back(ctx.cb.createSlice(
        ctx.fresh("e_idx"), i32, b.getIntegerType(ext.addrSizes[d])));
  }

  auto bufferAddr = [&](size_t d) -> Value {
    return info.type == BufferType::SEQ ? info.seqMem.addrPort(d)
                                         : info.combMem.addrPort(d);
  };
  auto bufferWriteData = [&]() -> Value {
    return info.type == BufferType::SEQ ? info.seqMem.writeData()
                                         : info.combMem.writeData();
  };
  auto bufferWriteEn = [&]() -> Value {
    return info.type == BufferType::SEQ ? info.seqMem.writeEn()
                                         : info.combMem.writeEn();
  };
  auto bufferDone = [&]() -> Value {
    return info.type == BufferType::SEQ ? info.seqMem.done()
                                         : info.combMem.done();
  };

  auto assignBufferAddr = [&]() {
    for (size_t d = 0; d < info.sizes.size(); ++d) {
      ctx.cb.assign(bufSlices[d].getIn(), counters[d].getOut());
      ctx.cb.assign(bufferAddr(d), bufSlices[d].getOut());
    }
  };
  auto assignExtAddr = [&]() {
    for (size_t d = 0; d < info.sizes.size(); ++d) {
      ctx.cb.assign(extAdds[d].getLeft(), offsets[d]);
      ctx.cb.assign(extAdds[d].getRight(), counters[d].getOut());
      ctx.cb.assign(extSlices[d].getIn(), extAdds[d].getOut());
      ctx.cb.assign(ext.mem.addrPort(d), extSlices[d].getOut());
    }
  };

  if (isLoad) {
    std::string readName = ctx.fresh("ld_read");
    {
      OpBuilder::InsertionGuard guard(b);
      auto group = ctx.cb.createGroup(readName);
      assignExtAddr();
      ctx.cb.assignBit(ext.mem.contentEn(), true);
      ctx.cb.assignBit(ext.mem.writeEn(), false);
      ctx.cb.assign(ext.mem.writeData(), ctx.cb.i32(0));
      ctx.cb.assign(tmp.getIn(), ext.mem.readData());
      ctx.cb.assign(tmp.getWriteEn(), ext.mem.done());
      ctx.cb.groupDone(group, tmp.getDone());
    }
    ctx.addEnable(readName);

    std::string writeName = ctx.fresh("ld_write");
    {
      OpBuilder::InsertionGuard guard(b);
      auto group = ctx.cb.createGroup(writeName);
      assignBufferAddr();
      if (info.type == BufferType::SEQ)
        ctx.cb.assignBit(info.seqMem.contentEn(), true);
      ctx.cb.assign(bufferWriteData(), tmp.getOut());
      ctx.cb.assignBit(bufferWriteEn(), true);
      ctx.cb.groupDone(group, bufferDone());
    }
    ctx.addEnable(writeName);
    return success();
  }

  std::string readName = ctx.fresh("st_read");
  {
    OpBuilder::InsertionGuard guard(b);
    auto group = ctx.cb.createGroup(readName);
    assignBufferAddr();
    if (info.type == BufferType::SEQ) {
      ctx.cb.assignBit(info.seqMem.contentEn(), true);
      ctx.cb.assignBit(info.seqMem.writeEn(), false);
      ctx.cb.assign(info.seqMem.writeData(), ctx.cb.i32(0));
      ctx.cb.assign(tmp.getIn(), info.seqMem.readData());
      ctx.cb.assign(tmp.getWriteEn(), info.seqMem.done());
    } else {
      ctx.cb.assignBit(info.combMem.writeEn(), false);
      ctx.cb.assign(info.combMem.writeData(), ctx.cb.i32(0));
      ctx.cb.assign(tmp.getIn(), info.combMem.readData());
      ctx.cb.assignBit(tmp.getWriteEn(), true);
    }
    ctx.cb.groupDone(group, tmp.getDone());
  }
  ctx.addEnable(readName);

  std::string writeName = ctx.fresh("st_write");
  {
    OpBuilder::InsertionGuard guard(b);
    auto group = ctx.cb.createGroup(writeName);
    assignExtAddr();
    ctx.cb.assignBit(ext.mem.contentEn(), true);
    ctx.cb.assign(ext.mem.writeData(), tmp.getOut());
    ctx.cb.assignBit(ext.mem.writeEn(), true);
    ctx.cb.groupDone(group, ext.mem.done());
  }
  ctx.addEnable(writeName);
  return success();
}

LogicalResult BufferLowering::emitPipelinedStaticCopy(
    BufferInfo &info, MemRefInfo &ext, llvm::ArrayRef<Value> offsets,
    bool isLoad) {
  if (info.sizes.size() != ext.sizes.size() || info.sizes.empty()  ||
      info.sizes.size() > 2)
    return failure();

  OpBuilder &b = ctx.cb.getBuilder();

  const Type i32 = b.getI32Type();
  const bool rank2 = info.sizes.size() == 2;
  const int64_t rows = rank2 ? info.sizes[0] : 1;
  const int64_t cols = info.sizes.back();

  if (rows <= 0 || cols <= 0)
    return failure();

  const auto tmp0 = ctx.cb.createRegister(ctx.fresh("pl_tmp0"), 32);
  const auto tmp1 = ctx.cb.createRegister(ctx.fresh("pl_tmp1"), 32);
  auto row = rank2 ? ctx.cb.createRegister(ctx.fresh("pl_row"), 32)
                   : circt::calyx::RegisterOp();
  auto rd = ctx.cb.createRegister(ctx.fresh("pl_rd"), 32);
  auto wr = ctx.cb.createRegister(ctx.fresh("pl_wr"), 32);

  struct AddrCells {
    circt::calyx::SliceLibOp buf;
    circt::calyx::AddLibOp extAdd;
    circt::calyx::SliceLibOp ext;
  };
  auto makeCells = [&]() {
    llvm::SmallVector<AddrCells> cells;
    for (size_t d = 0; d < info.sizes.size(); ++d) {
      AddrCells c;
      c.buf = ctx.cb.createSlice(
          ctx.fresh("pl_bidx"), i32, b.getIntegerType(info.addrSizes[d]));
      c.extAdd = ctx.cb.createAdd(ctx.fresh("pl_ead"), i32);
      c.ext = ctx.cb.createSlice(
          ctx.fresh("pl_eidx"), i32, b.getIntegerType(ext.addrSizes[d]));
      cells.push_back(c);
    }
    return cells;
  };
  auto readCells = makeCells();
  auto writeCells = makeCells();
  auto epiCells = makeCells();

  auto innerAddress = [&](llvm::MutableArrayRef<AddrCells> cells, Value rowV,
                          Value innerV, bool buffer) {
    for (size_t d = 0; d < info.sizes.size(); ++d) {
      Value idx = (rank2 && d == 0) ? rowV : innerV;
      if (buffer) {
        ctx.cb.assign(cells[d].buf.getIn(), idx);
        Value port = info.type == BufferType::SEQ ? info.seqMem.addrPort(d)
                                                  : info.combMem.addrPort(d);
        ctx.cb.assign(port, cells[d].buf.getOut());
      } else {
        ctx.cb.assign(cells[d].extAdd.getLeft(), offsets[d]);
        ctx.cb.assign(cells[d].extAdd.getRight(), idx);
        ctx.cb.assign(cells[d].ext.getIn(), cells[d].extAdd.getOut());
        ctx.cb.assign(ext.mem.addrPort(d), cells[d].ext.getOut());
      }
    }
  };

  auto emitRead = [&](Value rowV, Value innerV,
                      circt::calyx::RegisterOp dst, llvm::StringRef tag) {
    std::string name = ctx.fresh(tag);
    {
      OpBuilder::InsertionGuard guard(b);
      auto group = ctx.cb.createStaticGroup(name, 2);
      if (isLoad) {
        innerAddress(readCells, rowV, innerV, /*buffer=*/false);
        ctx.cb.assignBit(ext.mem.contentEn(), true);
        ctx.cb.assignBit(ext.mem.writeEn(), false);
        ctx.cb.assign(ext.mem.writeData(), ctx.cb.i32(0));
        ctx.cb.assign(dst.getIn(), ext.mem.readData());
      } else {
        innerAddress(readCells, rowV, innerV, /*buffer=*/true);
        if (info.type == BufferType::SEQ) {
          ctx.cb.assignBit(info.seqMem.contentEn(), true);
          ctx.cb.assignBit(info.seqMem.writeEn(), false);
          ctx.cb.assign(info.seqMem.writeData(), ctx.cb.i32(0));
          ctx.cb.assign(dst.getIn(), info.seqMem.readData());
        } else {
          ctx.cb.assignBit(info.combMem.writeEn(), false);
          ctx.cb.assign(info.combMem.writeData(), ctx.cb.i32(0));
          ctx.cb.assign(dst.getIn(), info.combMem.readData());
        }
      }
      ctx.cb.assignBit(dst.getWriteEn(), true);
    }
    ctx.addEnable(name);
  };

  auto emitWrite = [&](Value rowV, Value innerV,
                       circt::calyx::RegisterOp src, llvm::StringRef tag) {
    std::string name = ctx.fresh(tag);
    {
      OpBuilder::InsertionGuard guard(b);
      auto group = ctx.cb.createStaticGroup(name, 1);
      if (isLoad) {
        innerAddress(writeCells, rowV, innerV, /*buffer=*/true);
        if (info.type == BufferType::SEQ)
          ctx.cb.assignBit(info.seqMem.contentEn(), true);
        ctx.cb.assign(info.type == BufferType::SEQ ? info.seqMem.writeData()
                                                   : info.combMem.writeData(),
                      src.getOut());
        ctx.cb.assignBit(info.type == BufferType::SEQ ? info.seqMem.writeEn()
                                                      : info.combMem.writeEn(),
                         true);
      } else {
        innerAddress(writeCells, rowV, innerV, /*buffer=*/false);
        ctx.cb.assignBit(ext.mem.contentEn(), true);
        ctx.cb.assign(ext.mem.writeData(), src.getOut());
        ctx.cb.assignBit(ext.mem.writeEn(), true);
      }
    }
    ctx.addEnable(name);
  };

  auto emitInc = [&](circt::calyx::RegisterOp counter, llvm::StringRef tag) {
    std::string name = ctx.fresh(tag);
    auto add = ctx.cb.createAdd(ctx.fresh("pl_add"), i32);
    {
      OpBuilder::InsertionGuard guard(b);
      auto group = ctx.cb.createStaticGroup(name, 1);
      ctx.cb.assign(add.getLeft(), counter.getOut());
      ctx.cb.assign(add.getRight(), ctx.cb.i32(1));
      ctx.cb.assign(counter.getIn(), add.getOut());
      ctx.cb.assignBit(counter.getWriteEn(), true);
    }
    ctx.addEnable(name);
  };

  auto emitInit = [&](circt::calyx::RegisterOp counter, int64_t value,
                      llvm::StringRef tag) {
    std::string name = ctx.fresh(tag);
    {
      OpBuilder::InsertionGuard guard(b);
      auto group = ctx.cb.createStaticGroup(name, 1);
      ctx.cb.assign(counter.getIn(), ctx.cb.i32(value));
      ctx.cb.assignBit(counter.getWriteEn(), true);
    }
    ctx.addEnable(name);
  };

  auto emitPar = [&](Value rowV, Value rdV, Value wrV,
                     circt::calyx::RegisterOp rdTmp,
                     circt::calyx::RegisterOp wrTmp, bool incr) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToEnd(ctx.control.back());
    auto par = ctx.cb.createStaticPar();
    ctx.control.push_back(par.getBodyBlock());
    emitRead(rowV, rdV, rdTmp, "pl_read");
    emitWrite(rowV, wrV, wrTmp, "pl_write");
    if (incr) {
      emitInc(rd, "pl_incr_rd");
      emitInc(wr, "pl_incr_wr");
    }
    ctx.control.pop_back();
  };

  auto emitInner = [&](Value rowV) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToEnd(ctx.control.back());
    auto level = ctx.cb.createStaticSeq();
    ctx.control.push_back(level.getBodyBlock());

    if (cols > 1) {
      emitInit(rd, 1, "pl_init_rd");
      emitInit(wr, 0, "pl_init_wr");
    }

    // Prologue: read element 0.
    emitRead(rowV, ctx.cb.i32(0), tmp0, "pl_prologue");

    int64_t steps = cols - 1;
    int64_t pairs = steps / 2;
    if (pairs > 0) {
      auto rep = ctx.cb.createStaticRepeat(static_cast<uint32_t>(pairs));
      ctx.control.push_back(rep.getBodyBlock());
      auto body = ctx.cb.createStaticSeq();
      ctx.control.push_back(body.getBodyBlock());
      emitPar(rowV, rd.getOut(), wr.getOut(), tmp1, tmp0, /*incr=*/true);
      emitPar(rowV, rd.getOut(), wr.getOut(), tmp0, tmp1, /*incr=*/true);
      ctx.control.pop_back();
      ctx.control.pop_back();
    }

    if (steps % 2 == 1)
      emitPar(rowV, rd.getOut(), wr.getOut(), tmp1, tmp0, /*incr=*/false);

    // Epilogue: write the last element.
    auto lastTmp = (steps % 2 == 1) ? tmp1 : tmp0;
    emitWrite(rowV, ctx.cb.i32(cols - 1), lastTmp, "pl_epilogue");

    ctx.control.pop_back();
  };

  if (!rank2) {
    emitInner(Value());
    return success();
  }

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToEnd(ctx.control.back());
  auto outer = ctx.cb.createStaticSeq();
  ctx.control.push_back(outer.getBodyBlock());
  emitInit(row, 0, "pl_init_row");
  auto rep = ctx.cb.createStaticRepeat(static_cast<uint32_t>(rows));
  ctx.control.push_back(rep.getBodyBlock());
  auto body = ctx.cb.createStaticSeq();
  ctx.control.push_back(body.getBodyBlock());
  emitInner(row.getOut());
  emitInc(row, "pl_incr_row");
  ctx.control.pop_back();
  ctx.control.pop_back();
  ctx.control.pop_back();
  return success();
}

LogicalResult BufferLowering::emitStaticCopy(
    BufferInfo &info, MemRefInfo &ext, llvm::ArrayRef<Value> offsets,
    bool isLoad) {
  if (info.sizes.size() != ext.sizes.size() || info.sizes.size() > 2)
    return failure();

  OpBuilder &b = ctx.cb.getBuilder();

  const Type i32 = b.getI32Type();
  auto tmp = ctx.cb.createRegister(ctx.fresh(isLoad ? "sld_tmp" : "sst_tmp"),
                                   32);

  llvm::SmallVector<circt::calyx::RegisterOp> counters;
  llvm::SmallVector<circt::calyx::SliceLibOp> bufSlices;
  llvm::SmallVector<circt::calyx::AddLibOp> extAdds;
  llvm::SmallVector<circt::calyx::SliceLibOp> extSlices;

  for (size_t d = 0; d < info.sizes.size(); ++d) {
    counters.push_back(ctx.cb.createRegister(ctx.fresh("sc_idx"), 32));
    bufSlices.push_back(ctx.cb.createSlice(
        ctx.fresh("sc_bidx"), i32, b.getIntegerType(info.addrSizes[d])));
    extAdds.push_back(ctx.cb.createAdd(ctx.fresh("sc_ead"), i32));
    extSlices.push_back(ctx.cb.createSlice(
        ctx.fresh("sc_eidx"), i32, b.getIntegerType(ext.addrSizes[d])));
  }

  auto assignBufAddr = [&]() {
    for (size_t d = 0; d < info.sizes.size(); ++d) {
      ctx.cb.assign(bufSlices[d].getIn(), counters[d].getOut());
      const Value port = info.type == BufferType::SEQ ? info.seqMem.addrPort(d)
                                                : info.combMem.addrPort(d);
      ctx.cb.assign(port, bufSlices[d].getOut());
    }
  };
  auto assignExtAddr = [&]() {
    for (size_t d = 0; d < info.sizes.size(); ++d) {
      ctx.cb.assign(extAdds[d].getLeft(), offsets[d]);
      ctx.cb.assign(extAdds[d].getRight(), counters[d].getOut());
      ctx.cb.assign(extSlices[d].getIn(), extAdds[d].getOut());
      ctx.cb.assign(ext.mem.addrPort(d), extSlices[d].getOut());
    }
  };

  auto emitRead = [&]() {
    std::string name = ctx.fresh(isLoad ? "sc_read_ext" : "sc_read_buf");
    uint64_t latency = (isLoad || info.type == BufferType::SEQ) ? 2 : 1;
    {
      OpBuilder::InsertionGuard guard(b);
      auto group = ctx.cb.createStaticGroup(name, latency);
      if (isLoad) {
        assignExtAddr();
        ctx.cb.assignBit(ext.mem.contentEn(), true);
        ctx.cb.assignBit(ext.mem.writeEn(), false);
        ctx.cb.assign(ext.mem.writeData(), ctx.cb.i32(0));
        ctx.cb.assign(tmp.getIn(), ext.mem.readData());
      } else {
        assignBufAddr();
        if (info.type == BufferType::SEQ) {
          ctx.cb.assignBit(info.seqMem.contentEn(), true);
          ctx.cb.assignBit(info.seqMem.writeEn(), false);
          ctx.cb.assign(info.seqMem.writeData(), ctx.cb.i32(0));
          ctx.cb.assign(tmp.getIn(), info.seqMem.readData());
        } else {
          ctx.cb.assignBit(info.combMem.writeEn(), false);
          ctx.cb.assign(info.combMem.writeData(), ctx.cb.i32(0));
          ctx.cb.assign(tmp.getIn(), info.combMem.readData());
        }
      }
      ctx.cb.assignBit(tmp.getWriteEn(), true);
    }
    ctx.addEnable(name);
  };

  auto emitWrite = [&]() {
    std::string name = ctx.fresh(isLoad ? "sc_write_buf" : "sc_write_ext");
    {
      OpBuilder::InsertionGuard guard(b);
      auto group = ctx.cb.createStaticGroup(name, 1);
      if (isLoad) {
        assignBufAddr();
        if (info.type == BufferType::SEQ)
          ctx.cb.assignBit(info.seqMem.contentEn(), true);
        ctx.cb.assign(info.type == BufferType::SEQ ? info.seqMem.writeData()
                                                   : info.combMem.writeData(),
                      tmp.getOut());
        ctx.cb.assignBit(info.type == BufferType::SEQ ? info.seqMem.writeEn()
                                                      : info.combMem.writeEn(),
                         true);
      } else {
        assignExtAddr();
        ctx.cb.assignBit(ext.mem.contentEn(), true);
        ctx.cb.assign(ext.mem.writeData(), tmp.getOut());
        ctx.cb.assignBit(ext.mem.writeEn(), true);
      }
    }
    ctx.addEnable(name);
  };

  std::function<LogicalResult(unsigned)> emitLevel =
      [&](unsigned dim) -> LogicalResult {
    if (info.sizes[dim] <= 0)
      return failure();

    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToEnd(ctx.control.back());
    auto level = ctx.cb.createStaticSeq();
    ctx.control.push_back(level.getBodyBlock());

    std::string initName = ctx.fresh("sc_init");
    {
      OpBuilder::InsertionGuard g(b);
      auto group = ctx.cb.createStaticGroup(initName, 1);
      ctx.cb.assign(counters[dim].getIn(), ctx.cb.i32(0));
      ctx.cb.assignBit(counters[dim].getWriteEn(), true);
    }
    ctx.addEnable(initName);

    auto repeat =
        ctx.cb.createStaticRepeat(static_cast<uint32_t>(info.sizes[dim]));
    ctx.control.push_back(repeat.getBodyBlock());

    auto body = ctx.cb.createStaticSeq();
    ctx.control.push_back(body.getBodyBlock());

    if (dim + 1 == info.sizes.size()) {
      emitRead();
      emitWrite();
    } else if (failed(emitLevel(dim + 1))) {
      return failure();
    }

    auto add = ctx.cb.createAdd(ctx.fresh("sc_add"), i32);
    std::string incrName = ctx.fresh("sc_incr");
    {
      OpBuilder::InsertionGuard g(b);
      auto group = ctx.cb.createStaticGroup(incrName, 1);
      ctx.cb.assign(add.getLeft(), counters[dim].getOut());
      ctx.cb.assign(add.getRight(), ctx.cb.i32(1));
      ctx.cb.assign(counters[dim].getIn(), add.getOut());
      ctx.cb.assignBit(counters[dim].getWriteEn(), true);
    }
    ctx.addEnable(incrName);

    ctx.control.pop_back(); // body
    ctx.control.pop_back(); // repeat
    ctx.control.pop_back(); // level
    return success();
  };

  return emitLevel(0);
}

LogicalResult BufferLowering::emitCopyLoops(
    BufferInfo &info, MemRefInfo &ext, llvm::ArrayRef<Value> offsets,
    bool isLoad, unsigned dim,
    llvm::MutableArrayRef<circt::calyx::RegisterOp> counters) {
  if (dim == info.sizes.size())
    return emitCopyBody(info, ext, offsets, isLoad, counters);

  OpBuilder &b = ctx.cb.getBuilder();
  Type i32 = b.getI32Type();
  int64_t count = info.sizes[dim];
  if (count <= 0)
    return failure();

  auto counter = ctx.cb.createRegister(ctx.fresh("idx"), 32);
  counters[dim] = counter;

  std::string initName = ctx.fresh("init_idx");
  {
    OpBuilder::InsertionGuard guard(b);
    auto group = ctx.cb.createGroup(initName);
    ctx.cb.assign(counter.getIn(), ctx.cb.i32(0));
    ctx.cb.assignBit(counter.getWriteEn(), true);
    ctx.cb.groupDone(group, counter.getDone());
  }
  ctx.addEnable(initName);

  OpBuilder::InsertionGuard loopGuard(b);
  b.setInsertionPointToEnd(ctx.control.back());
  ctx.cb.createRepeat(static_cast<uint32_t>(count));
  auto loopSeq = ctx.cb.createSeq();
  ctx.control.push_back(loopSeq.getBodyBlock());

  if (failed(emitCopyLoops(info, ext, offsets, isLoad, dim + 1, counters)))
    return failure();

  auto add = ctx.cb.createAdd(ctx.fresh("idx_add"), i32);
  std::string incrName = ctx.fresh("incr_idx");
  {
    OpBuilder::InsertionGuard guard(b);
    auto group = ctx.cb.createGroup(incrName);
    ctx.cb.assign(add.getLeft(), counter.getOut());
    ctx.cb.assign(add.getRight(), ctx.cb.i32(1));
    ctx.cb.assign(counter.getIn(), add.getOut());
    ctx.cb.assignBit(counter.getWriteEn(), true);
    ctx.cb.groupDone(group, counter.getDone());
  }
  ctx.addEnable(incrName);

  ctx.control.pop_back();
  return success();
}

LogicalResult BufferLowering::emitStaticBankedLoad(
    BufferInfo &info, MemRefInfo &src, llvm::ArrayRef<Value> offsets) {
  if (info.seqBanks.empty() || info.bankDim < 0 || info.bankDim > 1 ||
      offsets.size() != 2)
    return failure();

  OpBuilder &b = ctx.cb.getBuilder();
  Type i32 = b.getI32Type();
  int64_t depth = info.sizes[1 - info.bankDim];
  unsigned bankAddrSize = info.addrSizes[1 - info.bankDim];
  auto tmp0 = ctx.cb.createRegister(ctx.fresh("sbl_t0"), 32);
  auto tmp1 = ctx.cb.createRegister(ctx.fresh("sbl_t1"), 32);

  for (int64_t bankIndex = 0;
       bankIndex < static_cast<int64_t>(info.seqBanks.size()); ++bankIndex) {
    auto rd = ctx.cb.createRegister(ctx.fresh("sbl_rd"), 32);
    auto wr = ctx.cb.createRegister(ctx.fresh("sbl_wr"), 32);
    auto bankAddr = ctx.cb.createSlice(
        ctx.fresh("sbl_baddr"), i32, b.getIntegerType(bankAddrSize));
    auto srcAdd0 = ctx.cb.createAdd(ctx.fresh("sbl_a0"), i32);
    auto srcAdd1 = ctx.cb.createAdd(ctx.fresh("sbl_a1"), i32);
    auto srcSlice0 = ctx.cb.createSlice(
        ctx.fresh("sbl_i0"), i32, b.getIntegerType(src.addrSizes[0]));
    auto srcSlice1 = ctx.cb.createSlice(
        ctx.fresh("sbl_i1"), i32, b.getIntegerType(src.addrSizes[1]));

    OpBuilder::InsertionGuard outer(b);
    b.setInsertionPointToEnd(ctx.control.back());
    auto wrapper = ctx.cb.createStaticSeq();
    ctx.control.push_back(wrapper.getBodyBlock());

    auto emitInit = [&](circt::calyx::RegisterOp counter, int64_t value,
                        llvm::StringRef tag) {
      std::string name = ctx.fresh(tag);
      {
        OpBuilder::InsertionGuard g(b);
        auto group = ctx.cb.createStaticGroup(name, 1);
        ctx.cb.assign(counter.getIn(), ctx.cb.i32(value));
        ctx.cb.assignBit(counter.getWriteEn(), true);
      }
      ctx.addEnable(name);
    };
    auto emitInc = [&](circt::calyx::RegisterOp counter,
                       llvm::StringRef tag) {
      std::string name = ctx.fresh(tag);
      auto add = ctx.cb.createAdd(ctx.fresh("sbl_add"), i32);
      {
        OpBuilder::InsertionGuard g(b);
        auto group = ctx.cb.createStaticGroup(name, 1);
        ctx.cb.assign(add.getLeft(), counter.getOut());
        ctx.cb.assign(add.getRight(), ctx.cb.i32(1));
        ctx.cb.assign(counter.getIn(), add.getOut());
        ctx.cb.assignBit(counter.getWriteEn(), true);
      }
      ctx.addEnable(name);
    };
    auto emitRead = [&](Value inner, circt::calyx::RegisterOp dst) {
      std::string name = ctx.fresh("sbl_read");
      {
        OpBuilder::InsertionGuard g(b);
        auto group = ctx.cb.createStaticGroup(name, 2);
        Value sel0 = info.bankDim == 0 ? ctx.cb.i32(bankIndex) : inner;
        Value sel1 = info.bankDim == 1 ? ctx.cb.i32(bankIndex) : inner;
        ctx.cb.assign(srcAdd0.getLeft(), offsets[0]);
        ctx.cb.assign(srcAdd0.getRight(), sel0);
        ctx.cb.assign(srcSlice0.getIn(), srcAdd0.getOut());
        ctx.cb.assign(src.mem.addrPort(0), srcSlice0.getOut());
        ctx.cb.assign(srcAdd1.getLeft(), offsets[1]);
        ctx.cb.assign(srcAdd1.getRight(), sel1);
        ctx.cb.assign(srcSlice1.getIn(), srcAdd1.getOut());
        ctx.cb.assign(src.mem.addrPort(1), srcSlice1.getOut());
        ctx.cb.assignBit(src.mem.contentEn(), true);
        ctx.cb.assignBit(src.mem.writeEn(), false);
        ctx.cb.assign(src.mem.writeData(), ctx.cb.i32(0));
        ctx.cb.assign(dst.getIn(), src.mem.readData());
        ctx.cb.assignBit(dst.getWriteEn(), true);
      }
      ctx.addEnable(name);
    };
    auto emitWrite = [&](Value inner, circt::calyx::RegisterOp srcTmp) {
      std::string name = ctx.fresh("sbl_write");
      auto &bank = info.seqBanks[bankIndex];
      {
        OpBuilder::InsertionGuard g(b);
        auto group = ctx.cb.createStaticGroup(name, 1);
        ctx.cb.assign(bankAddr.getIn(), inner);
        ctx.cb.assign(bank.addrPort(0), bankAddr.getOut());
        ctx.cb.assignBit(bank.contentEn(), true);
        ctx.cb.assign(bank.writeData(), srcTmp.getOut());
        ctx.cb.assignBit(bank.writeEn(), true);
      }
      ctx.addEnable(name);
    };
    auto emitPar = [&](Value rdV, Value wrV,
                       circt::calyx::RegisterOp rdTmp,
                       circt::calyx::RegisterOp wrTmp, bool incr) {
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPointToEnd(ctx.control.back());
      auto par = ctx.cb.createStaticPar();
      ctx.control.push_back(par.getBodyBlock());
      emitRead(rdV, rdTmp);
      emitWrite(wrV, wrTmp);
      if (incr) {
        emitInc(rd, "sbl_incr_rd");
        emitInc(wr, "sbl_incr_wr");
      }
      ctx.control.pop_back();
    };

    if (depth > 1) {
      emitInit(rd, 1, "sbl_init_rd");
      emitInit(wr, 0, "sbl_init_wr");
    }
    emitRead(ctx.cb.i32(0), tmp0);

    int64_t steps = depth - 1;
    int64_t pairs = steps / 2;
    if (pairs > 0) {
      auto repeat = ctx.cb.createStaticRepeat(static_cast<uint32_t>(pairs));
      ctx.control.push_back(repeat.getBodyBlock());
      auto body = ctx.cb.createStaticSeq();
      ctx.control.push_back(body.getBodyBlock());
      emitPar(rd.getOut(), wr.getOut(), tmp1, tmp0, /*incr=*/true);
      emitPar(rd.getOut(), wr.getOut(), tmp0, tmp1, /*incr=*/true);
      ctx.control.pop_back();
      ctx.control.pop_back();
    }
    if (steps % 2 == 1)
      emitPar(rd.getOut(), wr.getOut(), tmp1, tmp0, /*incr=*/false);

    auto lastTmp = (steps % 2 == 1) ? tmp1 : tmp0;
    emitWrite(ctx.cb.i32(depth - 1), lastTmp);

    ctx.control.pop_back(); // wrapper
  }
  return success();
}

LogicalResult BufferLowering::emitBankedLoad(
    BufferInfo &info, MemRefInfo &src, llvm::ArrayRef<Value> offsets) {
  if (info.seqBanks.empty() || info.bankDim < 0 || info.bankDim > 1)
    return failure();
  if (offsets.size() != 2)
    return failure();

  OpBuilder &b = ctx.cb.getBuilder();
  Type i32 = b.getI32Type();
  int64_t depth = info.sizes[1 - info.bankDim];
  unsigned bankAddrSize = info.addrSizes[1 - info.bankDim];
  auto tmp = ctx.cb.createRegister(ctx.fresh("ld_bank_tmp"), 32);

  for (int64_t bankIndex = 0;
       bankIndex < static_cast<int64_t>(info.seqBanks.size()); ++bankIndex) {
    auto counter = ctx.cb.createRegister(ctx.fresh("bank_idx"), 32);

    std::string initName = ctx.fresh("init_bank_idx");
    {
      OpBuilder::InsertionGuard guard(b);
      auto group = ctx.cb.createGroup(initName);
      ctx.cb.assign(counter.getIn(), ctx.cb.i32(0));
      ctx.cb.assignBit(counter.getWriteEn(), true);
      ctx.cb.groupDone(group, counter.getDone());
    }
    ctx.addEnable(initName);

    OpBuilder::InsertionGuard loopGuard(b);
    b.setInsertionPointToEnd(ctx.control.back());
    ctx.cb.createRepeat(static_cast<uint32_t>(depth));
    auto loopSeq = ctx.cb.createSeq();
    ctx.control.push_back(loopSeq.getBodyBlock());

    auto bankAddr = ctx.cb.createSlice(
        ctx.fresh("bank_addr"), i32, b.getIntegerType(bankAddrSize));
    auto srcAdd0 = ctx.cb.createAdd(ctx.fresh("bank_ad0"), i32);
    auto srcAdd1 = ctx.cb.createAdd(ctx.fresh("bank_ad1"), i32);
    auto srcSlice0 = ctx.cb.createSlice(
        ctx.fresh("bank_i0"), i32, b.getIntegerType(src.addrSizes[0]));
    auto srcSlice1 = ctx.cb.createSlice(
        ctx.fresh("bank_i1"), i32, b.getIntegerType(src.addrSizes[1]));

    std::string readName = ctx.fresh("bank_ld_read");
    {
      OpBuilder::InsertionGuard guard(b);
      auto group = ctx.cb.createGroup(readName);
      Value sel0 = info.bankDim == 0 ? ctx.cb.i32(bankIndex)
                                     : counter.getOut();
      Value sel1 = info.bankDim == 1 ? ctx.cb.i32(bankIndex)
                                     : counter.getOut();
      ctx.cb.assign(srcAdd0.getLeft(), offsets[0]);
      ctx.cb.assign(srcAdd0.getRight(), sel0);
      ctx.cb.assign(srcSlice0.getIn(), srcAdd0.getOut());
      ctx.cb.assign(src.mem.addrPort(0), srcSlice0.getOut());
      ctx.cb.assign(srcAdd1.getLeft(), offsets[1]);
      ctx.cb.assign(srcAdd1.getRight(), sel1);
      ctx.cb.assign(srcSlice1.getIn(), srcAdd1.getOut());
      ctx.cb.assign(src.mem.addrPort(1), srcSlice1.getOut());
      ctx.cb.assignBit(src.mem.contentEn(), true);
      ctx.cb.assignBit(src.mem.writeEn(), false);
      ctx.cb.assign(src.mem.writeData(), ctx.cb.i32(0));
      ctx.cb.assign(tmp.getIn(), src.mem.readData());
      ctx.cb.assign(tmp.getWriteEn(), src.mem.done());
      ctx.cb.groupDone(group, tmp.getDone());
    }
    ctx.addEnable(readName);

    auto &bank = info.seqBanks[bankIndex];
    std::string writeName = ctx.fresh("bank_ld_write");
    {
      OpBuilder::InsertionGuard guard(b);
      auto group = ctx.cb.createGroup(writeName);
      ctx.cb.assign(bankAddr.getIn(), counter.getOut());
      ctx.cb.assign(bank.addrPort(0), bankAddr.getOut());
      ctx.cb.assignBit(bank.contentEn(), true);
      ctx.cb.assign(bank.writeData(), tmp.getOut());
      ctx.cb.assignBit(bank.writeEn(), true);
      ctx.cb.groupDone(group, bank.done());
    }
    ctx.addEnable(writeName);

    auto add = ctx.cb.createAdd(ctx.fresh("bank_inc"), i32);
    std::string incrName = ctx.fresh("incr_bank_idx");
    {
      OpBuilder::InsertionGuard guard(b);
      auto group = ctx.cb.createGroup(incrName);
      ctx.cb.assign(add.getLeft(), counter.getOut());
      ctx.cb.assign(add.getRight(), ctx.cb.i32(1));
      ctx.cb.assign(counter.getIn(), add.getOut());
      ctx.cb.assignBit(counter.getWriteEn(), true);
      ctx.cb.groupDone(group, counter.getDone());
    }
    ctx.addEnable(incrName);

    ctx.control.pop_back();
  }
  return success();
}

LogicalResult BufferLowering::LowerLoad(BufferLoad op) {
  auto buf_it = ctx.buffers.find(op.getBuffer());
  if (buf_it == ctx.buffers.end())
    return op.emitError("buffer was not lowered");

  auto src_it = ctx.memrefs.find(op.getSource());
  if (src_it == ctx.memrefs.end())
    return op.emitError("source memref was not lowered to a Calyx memory");

  BufferInfo &info = buf_it->second;
  MemRefInfo &src = src_it->second;
  if (info.type == BufferType::REG)
    return op.emitError("buffer.load does not support reg buffers yet");

  if (!info.viewOffsets.empty())
    return op.emitError("buffer.view does not support buffer.load yet");

  if (info.sizes.size() != src.sizes.size() ||
      op.getOffsets().size() != src.sizes.size())
    return op.emitError("load rank/offset mismatch");

  llvm::SmallVector<Value> offsets;
  for (const Value offset : op.getOffsets())
    offsets.push_back(materializeI32(ctx, offset));

  if (!info.seqBanks.empty()) {
    if (failed(emitStaticBankedLoad(info, src, offsets)))
      return failure();
    op.erase();
    return success();
  }

  if (failed(emitPipelinedStaticCopy(info, src, offsets, /*isLoad=*/true)))
    return failure();
  op.erase();
  return success();
}

LogicalResult BufferLowering::LowerStore(BufferStore op) {
  auto bit = ctx.buffers.find(op.getBuffer());
  if (bit == ctx.buffers.end())
    return op.emitError("buffer was not lowered");
  auto dit = ctx.memrefs.find(op.getDestination());
  if (dit == ctx.memrefs.end())
    return op.emitError("destination memref was not lowered to a Calyx memory");

  BufferInfo &info = bit->second;
  MemRefInfo &dst = dit->second;
  if (!info.seqBanks.empty())
    return op.emitError("banked buffers do not support buffer.store yet");
  if (!info.viewOffsets.empty())
    return op.emitError("buffer.view does not support buffer.store yet");
  if (info.type == BufferType::REG)
    return op.emitError("buffer.store does not support reg buffers yet");
  if (info.sizes.size() != dst.sizes.size() ||
      op.getOffsets().size() != dst.sizes.size())
    return op.emitError("store rank/offset mismatch");

  llvm::SmallVector<Value> offsets;
  for (Value offset : op.getOffsets())
    offsets.push_back(materializeI32(ctx, offset));

  if (failed(emitPipelinedStaticCopy(info, dst, offsets, /*isLoad=*/false)))
    return failure();
  op.erase();
  return success();
}

LogicalResult BufferLowering::emitZeroLoops(
    BufferInfo &info, unsigned dim,
    llvm::MutableArrayRef<circt::calyx::RegisterOp> counters) {
  OpBuilder &b = ctx.cb.getBuilder();
  Type i32 = b.getI32Type();

  if (dim == info.sizes.size()) {
    llvm::SmallVector<circt::calyx::SliceLibOp> slices;
    for (size_t d = 0; d < info.sizes.size(); ++d)
      slices.push_back(ctx.cb.createSlice(
          ctx.fresh("zero_idx"), i32,
          b.getIntegerType(info.addrSizes[d])));

    std::string groupName = ctx.fresh("zero_buf");
    {
      OpBuilder::InsertionGuard guard(b);
      auto group = ctx.cb.createGroup(groupName);
      for (size_t d = 0; d < info.sizes.size(); ++d) {
        ctx.cb.assign(slices[d].getIn(), counters[d].getOut());
        Value port = info.type == BufferType::SEQ
                         ? info.seqMem.addrPort(d)
                         : info.combMem.addrPort(d);
        ctx.cb.assign(port, slices[d].getOut());
      }
      if (info.type == BufferType::SEQ)
        ctx.cb.assignBit(info.seqMem.contentEn(), true);
      ctx.cb.assign(info.type == BufferType::SEQ ? info.seqMem.writeData()
                                                 : info.combMem.writeData(),
                    ctx.cb.i32(0));
      ctx.cb.assignBit(info.type == BufferType::SEQ ? info.seqMem.writeEn()
                                                    : info.combMem.writeEn(),
                       true);
      ctx.cb.groupDone(group, info.type == BufferType::SEQ ? info.seqMem.done()
                                                           : info.combMem.done());
    }
    ctx.addEnable(groupName);
    return success();
  }

  int64_t count = info.sizes[dim];
  auto counter = ctx.cb.createRegister(ctx.fresh("zero_c"), 32);
  counters[dim] = counter;

  std::string initName = ctx.fresh("zero_init");
  {
    OpBuilder::InsertionGuard guard(b);
    auto group = ctx.cb.createGroup(initName);
    ctx.cb.assign(counter.getIn(), ctx.cb.i32(0));
    ctx.cb.assignBit(counter.getWriteEn(), true);
    ctx.cb.groupDone(group, counter.getDone());
  }
  ctx.addEnable(initName);

  OpBuilder::InsertionGuard loopGuard(b);
  b.setInsertionPointToEnd(ctx.control.back());
  ctx.cb.createRepeat(static_cast<uint32_t>(count));
  auto loopSeq = ctx.cb.createSeq();
  ctx.control.push_back(loopSeq.getBodyBlock());

  if (failed(emitZeroLoops(info, dim + 1, counters)))
    return failure();

  auto add = ctx.cb.createAdd(ctx.fresh("zero_add"), i32);
  std::string incrName = ctx.fresh("zero_incr");
  {
    OpBuilder::InsertionGuard guard(b);
    auto group = ctx.cb.createGroup(incrName);
    ctx.cb.assign(add.getLeft(), counter.getOut());
    ctx.cb.assign(add.getRight(), ctx.cb.i32(1));
    ctx.cb.assign(counter.getIn(), add.getOut());
    ctx.cb.assignBit(counter.getWriteEn(), true);
    ctx.cb.groupDone(group, counter.getDone());
  }
  ctx.addEnable(incrName);
  ctx.control.pop_back();
  return success();
}

LogicalResult BufferLowering::LowerZero(BufferZero op) {
  auto it = ctx.buffers.find(op.getBuffer());
  if (it == ctx.buffers.end())
    return op.emitError("buffer was not lowered");
  BufferInfo &info = it->second;
  if (!info.seqBanks.empty())
    return op.emitError("banked buffers do not support buffer.zero");
  if (!info.viewOffsets.empty())
    return op.emitError("buffer.view does not support buffer.zero");

  if (info.type == BufferType::REG) {
    if (failed(emitWrite(info, {}, ctx.cb.i32(0))))
      return failure();
    op.erase();
    return success();
  }

  llvm::SmallVector<circt::calyx::RegisterOp> counters(info.sizes.size());
  if (failed(emitZeroLoops(info, 0, counters)))
    return failure();

  op.erase();
  return success();
}

} // namespace stream
