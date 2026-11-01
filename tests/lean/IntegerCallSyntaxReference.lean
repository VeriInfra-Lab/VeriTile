import VeriTile.Triton.DSL.IntegerCallArguments
namespace VeriTile.Tests.IntegerCallSyntaxReference
open VeriTile.Triton
/-- Independently typed complete program: a static-loop counter is explicitly
remapped inside the body; shape and arange widths use the original domain. -/
def source (Y : RegionName) (n COUNT BLOCK : Nat) (hB : Random.ArangeSizeDomain BLOCK) : ComputeKernel :=
  triton {
    tl.region Y = tl.float32
    acc = tl.zeros([$(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK hB)], tl.float32)
    for q in tl.static_range($((COUNT+BLOCK-1)/BLOCK)) {
      q = q * $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK hB)
      offsets = q + tl.arange(0, $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK hB))
      acc = acc + offsets.to(tl.float32)
    }
    tl.store(Y + tl.arange(0, $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK hB)),
      acc, mask=tl.arange(0, $(_root_.VeriTile.Triton.Random.i32ArangeSize BLOCK hB)) < $(n))
  }
end VeriTile.Tests.IntegerCallSyntaxReference
