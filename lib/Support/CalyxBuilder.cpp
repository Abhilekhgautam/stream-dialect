//===- CalyxBuilder.cpp - Helpers for building Calyx dialect IR ------------===//

#include "Support/CalyxBuilder.h"

#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/HW/HWOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include <string>

using namespace mlir;

namespace stream {

circt::calyx::ComponentOp
CalyxBuilder::createComponent(const StringAttr name,
                             llvm::ArrayRef<circt::calyx::PortInfo> ports) {
  comp = circt::calyx::ComponentOp::create(b, loc, name, ports);
  wires = comp.getWiresOp();
  control = comp.getControlOp();

  if (auto module = comp->getParentOfType<mlir::ModuleOp>())
    module->setAttr("calyx.entrypoint", name);

  return comp;
}

//===--------------------------------------------------------------------===//
// Cells
//===--------------------------------------------------------------------===//

circt::calyx::RegisterOp CalyxBuilder::createRegister(const llvm::StringRef name,
                                                      unsigned width) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(wires.getOperation());
  return circt::calyx::RegisterOp::create(b, loc, name, width);
}

circt::calyx::SeqMemoryOp
CalyxBuilder::createSeqMem(const  llvm::StringRef name, const unsigned width,
                           const llvm::ArrayRef<int64_t> sizes,
                           const llvm::ArrayRef<int64_t> addrSizes) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(wires.getOperation());
  return circt::calyx::SeqMemoryOp::create(b, loc, name, width, sizes,
                                           addrSizes);
}

circt::calyx::MemoryOp
CalyxBuilder::createCombMem(const llvm::StringRef name, const unsigned width,
                            const llvm::ArrayRef<int64_t> sizes,
                            const llvm::ArrayRef<int64_t> addrSizes) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(wires.getOperation());
  return circt::calyx::MemoryOp::create(b, loc, name, width, sizes, addrSizes);
}

circt::calyx::AddLibOp CalyxBuilder::createAdd(const llvm::StringRef name,
                                               Type ty) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(wires.getOperation());
  return circt::calyx::AddLibOp::create(b, loc, name, {ty, ty, ty});
}

circt::calyx::LshLibOp CalyxBuilder::createLsh(const llvm::StringRef name,
                                               Type ty) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(wires.getOperation());
  return circt::calyx::LshLibOp::create(b, loc, name, {ty, ty, ty});
}

circt::calyx::SliceLibOp CalyxBuilder::createSlice(const llvm::StringRef name,
                                                   Type inTy, Type outTy) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(wires.getOperation());
  return circt::calyx::SliceLibOp::create(b, loc, name, {inTy, outTy});
}

circt::calyx::MultPipeLibOp CalyxBuilder::createMultPipe(const llvm::StringRef name,
                                                         Type ty) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(wires.getOperation());

  Type i1 = b.getI1Type();
  return circt::calyx::MultPipeLibOp::create(
      b, loc, name, {i1, i1, i1, ty, ty, ty, i1});
}

circt::hw::ConstantOp CalyxBuilder::createConstant(llvm::StringRef name,
                                                   const int64_t value, const Type ty) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(wires.getOperation());
  return circt::hw::ConstantOp::create(b, loc, ty, value);
}

Value CalyxBuilder::i32(const int64_t value) {
  auto it = i32Consts.find(value);
  if (it != i32Consts.end())
    return it->second.getResult();

  const std::string name = "c" + std::to_string(value);
  auto cst = createConstant(name, value, b.getI32Type());
  i32Consts[value] = cst;
  return cst.getResult();
}

Value CalyxBuilder::oneI1() {
  if (!oneI1Op)
    oneI1Op = createConstant("one_i1", 1, b.getI1Type());
  return oneI1Op.getResult();
}

Value CalyxBuilder::zeroI1() {
  if (!zeroI1Op)
    zeroI1Op = createConstant("zero_i1", 0, b.getI1Type());
  return zeroI1Op.getResult();
}

//===--------------------------------------------------------------------===//
// Wires / groups / assignments
//===--------------------------------------------------------------------===//

void CalyxBuilder::assign(const Value dst, const Value src) {
  circt::calyx::AssignOp::create(b, loc, dst, src);
}

circt::calyx::GroupOp CalyxBuilder::createGroup(llvm::StringRef name) {
  b.setInsertionPointToEnd(wires.getBodyBlock());
  auto group = circt::calyx::GroupOp::create(b, loc, name);
  if (group->getRegion(0).empty())
    group->getRegion(0).emplaceBlock();
  b.setInsertionPointToEnd(&group->getRegion(0).front());
  return group;
}

void CalyxBuilder::groupDone(circt::calyx::GroupOp group, Value done) {
  circt::calyx::GroupDoneOp::create(b, loc, done);
}

//===--------------------------------------------------------------------===//
// Control
//===--------------------------------------------------------------------===//

circt::calyx::SeqOp CalyxBuilder::createSeq() {
  auto seq = circt::calyx::SeqOp::create(b, loc);
  b.setInsertionPointToEnd(seq.getBodyBlock());
  return seq;
}

circt::calyx::ParOp CalyxBuilder::createPar() {
  auto par = circt::calyx::ParOp::create(b, loc);
  b.setInsertionPointToEnd(par.getBodyBlock());
  return par;
}

circt::calyx::StaticSeqOp CalyxBuilder::createStaticSeq() {
  auto seq = circt::calyx::StaticSeqOp::create(b, loc);
  b.setInsertionPointToEnd(seq.getBodyBlock());
  return seq;
}

circt::calyx::StaticParOp CalyxBuilder::createStaticPar() {
  auto par = circt::calyx::StaticParOp::create(b, loc);
  b.setInsertionPointToEnd(par.getBodyBlock());
  return par;
}

circt::calyx::StaticRepeatOp CalyxBuilder::createStaticRepeat(
    const uint32_t count) {
  auto repeat = circt::calyx::StaticRepeatOp::create(b, loc, count);
  b.setInsertionPointToEnd(repeat.getBodyBlock());
  return repeat;
}

circt::calyx::StaticGroupOp
CalyxBuilder::createStaticGroup(const llvm::StringRef name,
                                const uint64_t latency) {
  b.setInsertionPointToEnd(wires.getBodyBlock());
  auto group = circt::calyx::StaticGroupOp::create(b, loc, name, latency);
  b.setInsertionPointToEnd(group.getBodyBlock());
  return group;
}

circt::calyx::RepeatOp CalyxBuilder::createRepeat(const uint32_t count) {
  auto repeat = circt::calyx::RepeatOp::create(b, loc, count);
  b.setInsertionPointToEnd(repeat.getBodyBlock());
  return repeat;
}

void CalyxBuilder::enable(const llvm::StringRef group) {
  circt::calyx::EnableOp::create(b, loc, group);
}

} // namespace stream
