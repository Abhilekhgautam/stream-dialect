# Stream Dialect

An experimental out-of-tree MLIR/CIRCT dialect for streaming accelerator
kernels.  It currently covers tile-level GEMM, on-chip buffers, counted loops,
explicit parallel control, and lowering to the CIRCT Calyx dialect.

The intended path is:

```text
PyTorch -> linalg -> Stream -> Calyx -> Verilog
```

The `affine` and `scf` stages are **not** part of this design.  They belong to
the standard `scf-to-calyx` route that this dialect expect to replace.

This repository currently implements **Stream -> Calyx -> Verilog**.  The
`PyTorch -> linalg -> Stream` frontend is not in this tree yet.

The project is experimental.  It is not a complete CNN compiler yet; there is
no convolution or line-buffer lowering, and the GEMM results are specific
16x16 measurements.

## Layout

```text
include/Stream/                  dialect, op, and pass definitions
include/Conversion/              conversion-facing headers
include/Support/                 CalyxBuilder helpers

lib/Stream/                      dialect + ops
lib/Support/                     CalyxBuilder implementation
lib/Conversion/StreamToCalyx/    Stream-to-Calyx lowerings

stream-opt/                      optimizer driver
stream-plugin/                   optional circt-opt plugin
test/                            lit/FileCheck regression tests
```

## Build

Configure against a CIRCT build or install tree:

```sh
cd stream-dialect
cmake -G Ninja -S . -B build \
  -DCIRCT_DIR=<circt-build>/lib/cmake/circt \
  -DMLIR_DIR=<mlir-build>/lib/cmake/mlir
cmake --build build -j
```

The build produces:

- `build/bin/stream-opt`
- `build/lib/libStream.a`
- `build/lib/libStreamSupport.a`
- `build/lib/libStreamToCalyx.a`
- `build/lib/StreamPlugin.so` (when `circt-opt` is available)

## Dialect

### Memory hierarchy

```mlir
%s = stream.buffer.alloc {kind = "seq"} : memref<4x4xi32>
%c = stream.buffer.alloc {kind = "comb"} : memref<4x4xi32>
%r = stream.buffer.alloc {kind = "reg"} : memref<1xi32>
```

- `seq`: `calyx.seq_mem`, one-cycle read latency.
- `comb`: `calyx.memory`, combinational read.
- `reg`: a single `calyx.register`; multi-element reg buffers are rejected.

Banking is explicit:

```mlir
%a = stream.buffer.alloc {kind = "seq", banks = 4, bank_dim = 0}
    : memref<4x16xi32>
%b = stream.buffer.alloc {kind = "seq", banks = 4, bank_dim = 1}
    : memref<16x4xi32>
```

`bank_dim = 0` banks along rows; `bank_dim = 1` banks along columns.  The
current implementation requires `banks == size[bank_dim]`, `seq` buffers, and
rank-2 shapes.

### Buffer access

```mlir
%v = stream.buffer.read %s[%i, %j] : memref<4x4xi32>
stream.buffer.write %s[%i, %j], %v : memref<4x4xi32>
stream.buffer.zero %s : memref<4x4xi32>
```

Tile copies:

```mlir
stream.buffer.load %a_tile, %A[%m, %k]
    : memref<4x16xi32>, memref<16x16xi32>
stream.buffer.store %C[%m, %n], %c_tile
    : memref<16x16xi32>, memref<4x4xi32>
```

Unbanked copies lower to pipelined static Calyx groups (II=2).  Banked copies
lower to pipelined static groups per bank.

Views alias an existing on-chip buffer without copying:

```mlir
%av = stream.buffer.view %a[%m, %k]
    : memref<16x16xi32> -> memref<4x16xi32>
```

Views currently support unbanked `seq`/`comb` sources and are used by
`gemm_tile`.  Banked views are not implemented.

### Control

```mlir
stream.loop %i = 0 to 16 step 4 {
  ...
  stream.yield
}

stream.par {
  stream.buffer.load %a_tile, %A[%m, %k] : ...
  stream.buffer.load %b_tile, %B[%k, %n] : ...
  stream.yield
}
```

`stream.loop` is a counted i32 loop; bounds are static.  `stream.par` lowers to
`calyx.par`; each branch is sequenced independently.

There is no `iter_args` yet.  Loop-carried state is expressed with buffers and
registers.

### Tile GEMM

```mlir
stream.gemm_tile %a_tile, %b_tile, %c_tile {k = 16}
    : memref<4x16xi32>, memref<16x4xi32>, memref<4x4xi32>
```

Semantics:

```text
c[m][n] = sum_k a[m][k] * b[k][n]
```

With `zero_init = false`, the existing contents of `c` are added before the
reduction, which enables K tiling:

```mlir
stream.gemm_tile %a, %b, %c {k = 4, zero_init = false}
```

`gemm_tile` is a tile-level macro, not a core primitive.  Its lowering has:

- an internal K loop,
- internal accumulator registers,
- `std_mult_pipe` multipliers,
- static banked reads, static multiplier groups, and static accumulation when
  A/B are banked and `zero_init = true`.

### Whole GEMM

The legacy op remains as a reference:

```mlir
stream.gemm %a, %b, %c {
  tile_m = 4 : i64, tile_n = 4 : i64, tile_k = 16 : i64,
  banks_a = 4 : i64, banks_b = 4 : i64,
  macs = 16 : i64, ii = 1 : i64
}
```

It can be expanded into core ops:

```sh
stream-opt --convert-gemm-to-core input.mlir
```

The expansion chooses between:

- a whole-problem-resident schedule (full A/B/C on-chip, views over tiles), and
- a tiled schedule (tile buffers, `buffer.load`, `gemm_tile`, `buffer.store`).

## Passes

| Pass | Purpose |
|---|---|
| `--convert-gemm-to-core` | Expand `stream.gemm` into loops, tile buffers/views, loads/stores, and `gemm_tile` |
| `--convert-buffer-to-calyx` | Lower core ops (buffers, loops, par, copies, `gemm_tile`) to Calyx |
| `--convert-stream-to-calyx` | Legacy direct `stream.gemm` -> Calyx lowering |

Typical flow:

```sh
./build/bin/stream-opt \
  --convert-gemm-to-core --convert-buffer-to-calyx \
  input.mlir -o out.mlir

circt-translate --export-calyx out.mlir -o out.futil
calyx out.futil -l <calyx-repo> -b verilog --disable-verify -o out.sv
```

## Tests

```sh
cmake --build build --target check-stream
```

The suite covers buffer kinds, loops, `stream.par`, banked `gemm_tile`, GEMM
expansion, and the full static GEMM lowering.

## Current results

16x16 GEMM, 4x4 output tiles, simulated with Icarus Verilog:

| Version | Time |
|---|---:|
| Legacy `stream.gemm` | 56785 ns |
| Composed, banked, static MAC + pipelined banked copies | 51665 ns |

4x4 tile copy:

| Copy lowering | Time |
|---|---:|
| Dynamic groups | 3655 ns |
| Static serial groups | 1515 ns |
| Pipelined static groups | 1035 ns |

These are specific 16x16 measurements, not a general performance claim.

## Known limitations

- Static banked `gemm_tile` is used only when both A and B are banked and
  `zero_init = true`.
- K-tiled cases still use the dynamic MAC path; banked loads are static there.
- Banked views and dynamic bank selection are not implemented.
- `reg` buffers are single-element only.
- `buffer.zero` is loop-based and dynamic; it is not yet a static control path.
- `stream.loop` has no `ii` attribute yet.  Static copies reach II=2; the
  static banked MAC costs roughly 7 cycles per K step.
- No `line_buffer`, convolution op, or fusion ops (`bias_add`, ReLU, quantize)
  yet.
- CIRCT's Calyx exporter emits `std_mem_d*` for `calyx.memory`, while native
  Calyx 0.7.1 expects `comb_mem_d*`; a post-export `sed` is needed for `comb`
  buffers.
- Top-level on-chip memories appear as Verilog ports; simulation testbenches
  must provide memory models for them.

## Roadmap

1. `line_buffer` and convolution lowering.
2. Tile fusion: `bias_add`, ReLU, quantize on `c_tile`.
3. Banked views / dynamic bank muxes for resident schedules.
4. `stream.loop {ii = ...}` with static control, and II=1 MAC scheduling.
5. General blocking/residency selection for larger GEMMs.
