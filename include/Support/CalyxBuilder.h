//===- CalyxBuilder.h - Helpers for building Calyx dialect IR ----*- C++ -*-===//
//
// Construction helpers shared by stream-to-Calyx conversions. This file must
// not depend on the Stream dialect: it only knows about MLIR and CIRCT Calyx.
//
//===----------------------------------------------------------------------===//

#ifndef STREAM_SUPPORT_CALYXBUILDER_H
#define STREAM_SUPPORT_CALYXBUILDER_H

#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/HW/HWOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace stream {

/// Small convenience wrapper around the CIRCT Calyx dialect builders.
///
/// The builder keeps track of the component, wires, and control regions. Cell
/// creation is always inserted before `calyx.wires`; groups/assignments are
/// inserted at the current insertion point.
class CalyxBuilder {
public:
  CalyxBuilder(mlir::OpBuilder &builder, mlir::Location loc)
      : b(builder), loc(loc) {}

  //===--------------------------------------------------------------------===//
  // Component / regions
  //===--------------------------------------------------------------------===//

  circt::calyx::ComponentOp
  createComponent(mlir::StringAttr name,
                  llvm::ArrayRef<circt::calyx::PortInfo> ports);

  [[nodiscard]]circt::calyx::ComponentOp getComponent() const { return comp; }
  [[nodiscard]]circt::calyx::WiresOp getWires() const { return wires; }
  [[nodiscard]]circt::calyx::ControlOp getControl() const { return control; }

  /// Insertion point at the end of the wires region (for continuous assigns
  /// and groups).
  void setInsertionToWires() {
    b.setInsertionPointToEnd(wires.getBodyBlock());
  }

  /// Insertion point at the end of the control region.
  void setInsertionToControl() {
    b.setInsertionPointToEnd(control.getBodyBlock());
  }

  mlir::OpBuilder &getBuilder() { return b; }
  [[nodiscard]]mlir::Location getLoc() const { return loc; }

  //===--------------------------------------------------------------------===//
  // Cells
  //===--------------------------------------------------------------------===//

  circt::calyx::RegisterOp createRegister(llvm::StringRef name, unsigned width);

  circt::calyx::SeqMemoryOp
  createSeqMem(llvm::StringRef name, unsigned width,
               llvm::ArrayRef<int64_t> sizes,
               llvm::ArrayRef<int64_t> addrSizes);

  circt::calyx::MemoryOp
  createCombMem(llvm::StringRef name, unsigned width,
                llvm::ArrayRef<int64_t> sizes,
                llvm::ArrayRef<int64_t> addrSizes);

  circt::calyx::AddLibOp createAdd(llvm::StringRef name, mlir::Type ty);
  circt::calyx::LshLibOp createLsh(llvm::StringRef name, mlir::Type ty);
  circt::calyx::SliceLibOp createSlice(llvm::StringRef name, mlir::Type inTy,
                                      mlir::Type outTy);
  circt::calyx::MultPipeLibOp createMultPipe(llvm::StringRef name,
                                            mlir::Type ty);

  circt::hw::ConstantOp createConstant(llvm::StringRef name, int64_t value,
                                      mlir::Type ty);

  /// Cached i32/i1 constants.
  mlir::Value i32(int64_t value);
  mlir::Value oneI1();
  mlir::Value zeroI1();

  //===--------------------------------------------------------------------===//
  // Wires / groups / assignments
  //===--------------------------------------------------------------------===//

  void assign(mlir::Value dst, mlir::Value src);
  void assignBit(mlir::Value dst, bool value) {
    assign(dst, value ? oneI1() : zeroI1());
  }

  circt::calyx::GroupOp createGroup(llvm::StringRef name);
  void groupDone(circt::calyx::GroupOp group, mlir::Value done);

  //===--------------------------------------------------------------------===//
  // Control
  //===--------------------------------------------------------------------===//

  circt::calyx::SeqOp createSeq();
  circt::calyx::ParOp createPar();
  circt::calyx::RepeatOp createRepeat(uint32_t count);
  circt::calyx::StaticSeqOp createStaticSeq();
  circt::calyx::StaticParOp createStaticPar();
  circt::calyx::StaticRepeatOp createStaticRepeat(uint32_t count);
  circt::calyx::StaticGroupOp createStaticGroup(llvm::StringRef name,
                                                uint64_t latency);
  void enable(llvm::StringRef group);

private:
  mlir::OpBuilder &b;
  mlir::Location loc;

  circt::calyx::ComponentOp comp;
  circt::calyx::WiresOp wires;
  circt::calyx::ControlOp control;

  llvm::DenseMap<int64_t, circt::hw::ConstantOp> i32Consts;
  circt::hw::ConstantOp oneI1Op;
  circt::hw::ConstantOp zeroI1Op;
};

} // namespace stream

#endif // STREAM_SUPPORT_CALYXBUILDER_H
