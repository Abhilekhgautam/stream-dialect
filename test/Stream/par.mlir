// RUN: stream-opt --convert-buffer-to-calyx %s | FileCheck %s

func.func @parallel_loads(%A: memref<4x4xi32>, %B: memref<4x4xi32>) {
  %a = stream.buffer.alloc {kind = "seq"} : memref<4x4xi32>
  %b = stream.buffer.alloc {kind = "seq"} : memref<4x4xi32>
  %c0 = arith.constant 0 : i32
  stream.par {
    stream.buffer.load %a, %A[%c0, %c0] : memref<4x4xi32>, memref<4x4xi32>
    stream.buffer.load %b, %B[%c0, %c0] : memref<4x4xi32>, memref<4x4xi32>
    stream.yield
  }
  return
}

// CHECK: calyx.par
// CHECK: calyx.static_seq
// CHECK: calyx.static_repeat
