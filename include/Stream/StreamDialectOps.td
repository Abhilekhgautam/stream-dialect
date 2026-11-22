include "mlir/IR/OpBase.td"
include "mlir/IR/Constraints.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "StreamDialect.td"


// Base class for all Stream operations
class Stream_Op<string mnemonic, list<Trait> traits = []>
    : Op<Stream_Dialect, mnemonic, traits>;

def Stream_Gemm : Stream_Op<"gemm", [MemoryEffects<[MemRead, MemWrite]>]> {
  let arguments = (ins AnyMemRef:$a, AnyMemRef:$b, AnyMemRef:$c,
                       I64Attr:$tile_m, I64Attr:$tile_n,
                       DefaultValuedAttr<I64Attr, "0">:$tile_k,
                       I64Attr:$banks_a, I64Attr:$banks_b,
                       I64Attr:$macs, I64Attr:$ii);
  let assemblyFormat =
    "$a `,` $b `,` $c attr-dict `:` type($a) `,` type($b) `,` type($c)";
}

def Stream_LoopOp : Stream_Op<"loop", [SingleBlock, RecursiveMemoryEffects]> {
  let summary = "Counted loop for streaming hardware";
  let description = [{
    Executes the body `(upper - lower) / step` times. The single block
    argument is the i32 induction variable.
  }];
  let arguments = (ins I32Attr:$lower, I32Attr:$upper, I32Attr:$step);
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
  let builders = [OpBuilder<(ins "int64_t":$lower, "int64_t":$upper,
                                 "int64_t":$step)>];
}

def Stream_ParOp : Stream_Op<"par", [SingleBlock, RecursiveMemoryEffects]> {
  let summary = "Execute the body in parallel";
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$body attr-dict";
}

def Stream_YieldOp : Stream_Op<"yield", [Terminator, Pure]> {
  let summary = "Terminator for stream.loop";
  let assemblyFormat = "attr-dict";
}

def Stream_GemmTile : Stream_Op<"gemm_tile",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "K-reduction tile GEMM macro";
  let description = [{
    Computes `c[m][n] = sum_k a[m][k] * b[k][n]` for rank-2 tile buffers.
    The tile dimensions M and N are inferred from the shapes; K must match.
    With `zero_init = false`, the existing contents of `c` are added to the
    reduction, enabling K tiling by repeated calls.
  }];
  let arguments = (ins AnyMemRef:$a_tile, AnyMemRef:$b_tile,
                       AnyMemRef:$c_tile, I64Attr:$k,
                       DefaultValuedAttr<BoolAttr, "true">:$zero_init);
  let assemblyFormat =
    "$a_tile `,` $b_tile `,` $c_tile attr-dict `:` type($a_tile) `,` type($b_tile) `,` type($c_tile)";
}

def Stream_BufferAlloc : Stream_Op<"buffer.alloc",
    [MemoryEffects<[MemAlloc]>]> {
  let summary = "Allocate an on-chip buffer";
  let arguments = (ins
    DefaultValuedAttr<I64Attr, "1">:$banks,
    DefaultValuedAttr<I64Attr, "-1">:$bank_dim,
    DefaultValuedAttr<StrAttr, "\"seq\"">:$kind
  );
  let results = (outs AnyMemRef:$buffer);
  let assemblyFormat = "attr-dict `:` type($buffer)";
}

def Stream_BufferRead : Stream_Op<"buffer.read", [MemoryEffects<[MemRead]>]> {
  let summary = "Read from an on-chip buffer";
  let arguments = (ins AnyMemRef:$buffer, Variadic<I32>:$indices);
  let results = (outs I32:$value);
  let assemblyFormat = "$buffer `[` $indices `]` attr-dict `:` type($buffer)";
}

def Stream_BufferWrite : Stream_Op<"buffer.write",
    [MemoryEffects<[MemWrite]>]> {
  let summary = "Write one element to an on-chip buffer";
  let arguments = (ins AnyMemRef:$buffer, Variadic<I32>:$indices,
                       I32:$value);
  let assemblyFormat =
    "$buffer `[` $indices `]` `,` $value attr-dict `:` type($buffer)";
}

def Stream_BufferView : Stream_Op<"buffer.view",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Alias a tile of an existing on-chip buffer";
  let arguments = (ins AnyMemRef:$source, Variadic<I32>:$offsets);
  let results = (outs AnyMemRef:$result);
  let assemblyFormat =
    "$source `[` $offsets `]` attr-dict `:` type($source) `->` type($result)";
}

def Stream_BufferLoad : Stream_Op<"buffer.load",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Copy a tile from a memref into an on-chip buffer";
  let arguments = (ins AnyMemRef:$buffer, AnyMemRef:$source,
                       Variadic<I32>:$offsets);
  let assemblyFormat =
    "$buffer `,` $source `[` $offsets `]` attr-dict `:` type($buffer) `,` type($source)";
}

def Stream_BufferStore : Stream_Op<"buffer.store",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Copy an on-chip buffer into a memref tile";
  let arguments = (ins AnyMemRef:$destination, Variadic<I32>:$offsets,
                       AnyMemRef:$buffer);
  let assemblyFormat =
    "$destination `[` $offsets `]` `,` $buffer attr-dict `:` type($destination) `,` type($buffer)";
}

def Stream_BufferZero : Stream_Op<"buffer.zero",
    [MemoryEffects<[MemWrite]>]> {
  let summary = "Zero an on-chip buffer";
  let arguments = (ins AnyMemRef:$buffer);
  let assemblyFormat = "$buffer attr-dict `:` type($buffer)";
}