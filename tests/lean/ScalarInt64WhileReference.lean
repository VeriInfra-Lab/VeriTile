import VeriTile.Triton.DSL
namespace VeriTile.Tests.ScalarInt64WhileReference
open VeriTile.Triton
/-- Independent typed signed range: the bound cast and actual final counter
increment both remain. No source-loop frontend is imported. -/
def source : ComputeKernel := triton {
  tl.call_signature
  tl.param n_rows : tl.int64
  n_rows = tl.cast(n_rows, tl.int64)
  row = tl.full([], 0, dtype=tl.int64)
  for row in range(tl.cast(row, tl.int64), tl.cast(n_rows, tl.int64), tl.cast($((1 : Nat)), tl.int64)) {
    row += tl.full([], 1, dtype=tl.int64)
  }
}
end VeriTile.Tests.ScalarInt64WhileReference
