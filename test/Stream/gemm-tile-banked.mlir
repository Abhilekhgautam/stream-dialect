// RUN: stream-opt --convert-buffer-to-calyx %s | FileCheck %s

func.func @gemm_tile(%A: memref<4x16xi32>, %B: memref<16x4xi32>, %C: memref<4x4xi32>) {
  %a = stream.buffer.alloc {kind = "seq", banks = 4, bank_dim = 0} : memref<4x16xi32>
  %b = stream.buffer.alloc {kind = "seq", banks = 4, bank_dim = 1} : memref<16x4xi32>
  %c = stream.buffer.alloc {kind = "seq"} : memref<4x4xi32>
  %c0 = arith.constant 0 : i32
  stream.buffer.load %a, %A[%c0, %c0] : memref<4x16xi32>, memref<4x16xi32>
  stream.buffer.load %b, %B[%c0, %c0] : memref<16x4xi32>, memref<16x4xi32>
  stream.gemm_tile %a, %b, %c {k = 16 : i64} : memref<4x16xi32>, memref<16x4xi32>, memref<4x4xi32>
  stream.buffer.store %C[%c0, %c0], %c : memref<4x4xi32>, memref<4x4xi32>
  return
}

// CHECK: calyx.static_group latency<3>
// CHECK: calyx.static_repeat 16
// CHECK: calyx.enable @sg_mult
// CHECK: calyx.enable @sg_acc
