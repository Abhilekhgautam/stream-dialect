//
// Created by void on 11/5/26.
//

#include "Stream/StreamDialect.h"
#include "Stream/StreamDialectOps.h"

#include "mlir/IR/DialectImplementation.h"


using namespace mlir;
using namespace stream;

#define GET_OP_CLASSES
#include "Stream/StreamDialectOpsDialect.cpp.inc"
#undef GET_OP_CLASSES

void StreamDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "Stream/StreamDialectOps.cpp.inc"
>();
}