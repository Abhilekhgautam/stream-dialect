// RUN: stream-opt --convert-gemm-to-core %s | FileCheck %s

func.func @gemm(%a: memref<16x16xi32>, %b: memref<16x16xi32>, %c: memref<16x16xi32>) {
  stream.gemm %a, %b, %c {
    tile_m = 4 : i64, tile_n = 4 : i64, tile_k = 16 : i64,
    banks_a = 1 : i64, banks_b = 1 : i64,
    macs = 16 : i64, ii = 1 : i64
  } : memref<16x16xi32>, memref<16x16xi32>, memref<16x16xi32>
  return
}

// CHECK: stream.buffer.alloc
// CHECK: stream.buffer.load
// CHECK: stream.loop
// CHECK: stream.buffer.view
// CHECK: stream.gemm_tile
// CHECK: stream.buffer.store
