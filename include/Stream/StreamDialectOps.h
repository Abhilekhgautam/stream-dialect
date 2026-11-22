//
// Created by void on 11/5/26.
//

#ifndef CIRCT_STANDALONE_STREAMOPS_H
#define CIRCT_STANDALONE_STREAMOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/VectorTypes.h"

#include "Stream/StreamDialect.h"

#define GET_OP_CLASSES
#include "Stream/StreamDialectOps.h.inc"
#undef GET_OP_CLASSES

#endif //CIRCT_STANDALONE_STREAMOPS_H