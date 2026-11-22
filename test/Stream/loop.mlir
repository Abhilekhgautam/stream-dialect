// RUN: stream-opt --convert-buffer-to-calyx %s | FileCheck %s

func.func @loops() {
  %b = stream.buffer.alloc {kind = "seq"} : memref<4x4xi32>
  %c0 = arith.constant 0 : i32
  stream.loop %i = 0 to 4 step 1 {
    stream.loop %j = 0 to 4 step 1 {
      stream.buffer.write %b[%i, %j], %c0 : memref<4x4xi32>
      stream.yield
    }
    stream.yield
  }
  return
}

// CHECK: calyx.register
// CHECK: calyx.seq
// CHECK: calyx.repeat 4
