#ifndef _STREAM_PASSES_H_
#define _STREAM_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace stream {

#define GEN_PASS_DECL
#include "Stream/StreamPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "Stream/StreamPasses.h.inc"

}

#endif //_STREAM_PASSES_H_