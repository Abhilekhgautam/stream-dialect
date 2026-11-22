// RUN: stream-opt --convert-buffer-to-calyx %s | FileCheck %s

func.func @buffers() {
  %s = stream.buffer.alloc {kind = "seq"} : memref<4x4xi32>
  %c = stream.buffer.alloc {kind = "comb"} : memref<4x4xi32>
  %r = stream.buffer.alloc {kind = "reg"} : memref<1xi32>
  %c0 = arith.constant 0 : i32
  %v = arith.constant 7 : i32
  stream.buffer.write %s[%c0, %c0], %v : memref<4x4xi32>
  %x = stream.buffer.read %s[%c0, %c0] : memref<4x4xi32>
  stream.buffer.write %c[%c0, %c0], %x : memref<4x4xi32>
  stream.buffer.write %r[%c0], %v : memref<1xi32>
  stream.buffer.zero %s : memref<4x4xi32>
  return
}

// CHECK: calyx.component @main
// CHECK: calyx.seq_mem
// CHECK: calyx.memory
// CHECK: calyx.register
// CHECK: calyx.repeat 4
