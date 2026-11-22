//
// Created by void on 11/6/26.
//
#include "Stream/StreamDialectOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;

#define GET_OP_CLASSES
#include "Stream/StreamDialectOps.cpp.inc"

namespace stream {


void LoopOp::build(OpBuilder &builder, OperationState &state, int64_t lower,
                   int64_t upper, int64_t step) {
  state.addAttribute("lower", builder.getI32IntegerAttr(lower));
  state.addAttribute("upper", builder.getI32IntegerAttr(upper));
  state.addAttribute("step", builder.getI32IntegerAttr(step));
  Region *body = state.addRegion();
  body->push_back(new Block());
  body->front().addArgument(builder.getI32Type(), state.location);
}

ParseResult LoopOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::Argument inductionVariable;
  int64_t lower = 0, upper = 0, step = 1;

  if (parser.parseOperand(inductionVariable.ssaName) || parser.parseEqual() ||
      parser.parseInteger(lower) || parser.parseKeyword("to") ||
      parser.parseInteger(upper) || parser.parseKeyword("step") ||
      parser.parseInteger(step))
    return failure();

  Builder &builder = parser.getBuilder();
  inductionVariable.type = builder.getI32Type();
  result.addAttribute("lower", builder.getI32IntegerAttr(lower));
  result.addAttribute("upper", builder.getI32IntegerAttr(upper));
  result.addAttribute("step", builder.getI32IntegerAttr(step));

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, inductionVariable))
    return failure();

  if (body->empty())
    body->emplaceBlock();
  Block &block = body->front();
  if (block.empty() || !block.back().hasTrait<OpTrait::IsTerminator>()) {
    OpBuilder opBuilder(builder.getContext());
    opBuilder.setInsertionPointToEnd(&block);
    YieldOp::create(opBuilder, result.location);
  }

  return parser.parseOptionalAttrDict(result.attributes);
}

void LoopOp::print(OpAsmPrinter &p) {
  p << " " << getBody().getArgument(0) << " = " << getLower() << " to "
    << getUpper() << " step " << getStep() << " ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"lower", "upper", "step"});
}

} // namespace stream
