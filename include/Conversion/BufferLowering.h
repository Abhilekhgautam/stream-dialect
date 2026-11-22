//===- BufferLowering.h - Lower stream.buffer ops to Calyx -------*- C++ -*-===//

#ifndef STREAM_DIALECT_BUFFERLOWERING_H
#define STREAM_DIALECT_BUFFERLOWERING_H

#include "Stream/StreamDialectOps.h"
#include "Support/CalyxBuilder.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace stream {

enum struct BufferType { SEQ, COMB, REG };

/// The Calyx resources created for one `stream.buffer.alloc`.
struct BufferInfo {
  mlir::Value value;
  BufferType type = BufferType::SEQ;
  unsigned elemWidth = 32;
  llvm::SmallVector<int64_t> sizes;
  llvm::SmallVector<int64_t> addrSizes;

  circt::calyx::SeqMemoryOp seqMem;  // BufferType::SEQ, single bank
  circt::calyx::MemoryOp combMem;    // BufferType::COMB
  circt::calyx::RegisterOp reg;      // BufferType::REG (one element)
  int64_t bankDim = -1;              // -1: unbanked
  llvm::SmallVector<circt::calyx::SeqMemoryOp> seqBanks;
  /// Base offsets when this BufferInfo aliases a view of its parent.
  llvm::SmallVector<mlir::Value> viewOffsets;
};

/// The Calyx memory created for a function memref argument.
struct MemRefInfo {
  mlir::Value value;
  circt::calyx::SeqMemoryOp mem;
  unsigned elemWidth = 32;
  llvm::SmallVector<int64_t> sizes;
  llvm::SmallVector<int64_t> addrSizes;
};

/// Shared state for lowering one function/component.
struct LoweringContext {
  explicit LoweringContext(CalyxBuilder &cb) : cb(cb) {}

  CalyxBuilder &cb;
  llvm::SmallVector<mlir::Block *> control;
  llvm::DenseMap<mlir::Value, BufferInfo> buffers;
  llvm::DenseMap<mlir::Value, MemRefInfo> memrefs;
  /// Stream SSA values mapped to Calyx values.
  llvm::DenseMap<mlir::Value, mlir::Value> valueMap;
  unsigned counter = 0;

  std::string fresh(llvm::StringRef base);
  void addEnable(llvm::StringRef group);

  [[nodiscard]] mlir::Value lookup(const mlir::Value value) const {
    auto it = valueMap.find(value);
    return it == valueMap.end() ? mlir::Value() : it->second;
  }
};

/// Lowers stream.buffer.* to Calyx memories, registers, and groups.
class BufferLowering {
public:
  explicit BufferLowering(LoweringContext &ctx) : ctx(ctx) {}

  mlir::LogicalResult LowerAlloc(BufferAlloc op);
  mlir::LogicalResult LowerRead(BufferRead op);
  mlir::LogicalResult LowerWrite(BufferWrite op);
  mlir::LogicalResult LowerView(BufferView op);
  mlir::LogicalResult LowerLoad(BufferLoad op);
  mlir::LogicalResult LowerStore(BufferStore op);
  mlir::LogicalResult LowerGemmTile(GemmTile op);
  mlir::LogicalResult LowerStaticBankedGemmTile(GemmTile op, BufferInfo &a,
                                                BufferInfo &b, BufferInfo &c,
                                                int64_t M, int64_t K,
                                                int64_t N);
  mlir::LogicalResult LowerZero(BufferZero op);

private:
  mlir::LogicalResult emitWrite(BufferInfo &info,
                                llvm::ArrayRef<mlir::Value> indices,
                                mlir::Value value);

  mlir::LogicalResult
  emitZeroLoops(BufferInfo &info, unsigned dim,
                llvm::MutableArrayRef<circt::calyx::RegisterOp> counters);

  mlir::LogicalResult
  emitStaticCopy(BufferInfo &info, MemRefInfo &ext,
                 llvm::ArrayRef<mlir::Value> offsets, bool isLoad);

  mlir::LogicalResult
  emitPipelinedStaticCopy(BufferInfo &info, MemRefInfo &ext,
                          llvm::ArrayRef<mlir::Value> offsets, bool isLoad);

  mlir::LogicalResult
  emitCopyLoops(BufferInfo &info, MemRefInfo &ext,
                llvm::ArrayRef<mlir::Value> offsets, bool isLoad,
                unsigned dim,
                llvm::MutableArrayRef<circt::calyx::RegisterOp> counters);

  mlir::LogicalResult
  emitBankedLoad(BufferInfo &info, MemRefInfo &src,
                 llvm::ArrayRef<mlir::Value> offsets);

  mlir::LogicalResult
  emitStaticBankedLoad(BufferInfo &info, MemRefInfo &src,
                       llvm::ArrayRef<mlir::Value> offsets);

  mlir::LogicalResult
  emitCopyBody(BufferInfo &info, MemRefInfo &ext,
               llvm::ArrayRef<mlir::Value> offsets, bool isLoad,
               llvm::MutableArrayRef<circt::calyx::RegisterOp> counters);

  LoweringContext &ctx;
};

} // namespace stream

#endif // STREAM_DIALECT_BUFFERLOWERING_H
