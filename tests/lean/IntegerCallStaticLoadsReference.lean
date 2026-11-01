import VeriTile.Triton.DSL.IntegerCallArguments
namespace VeriTile.Tests.IntegerCallStaticLoadsReference
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  scalars := [("n",.integer)]
  constexpr := ["COUNT","BLOCK"]
  readPorts := ["X"]
  writePorts := ["Y"]

/-- Independently elaborate the ordinary counted loop through the existing
call-signature frontend, before importing the new static/arange composition. -/
def source (X Y : RegionName) (n COUNT BLOCK : Nat) (hB : Random.ArangeSizeDomain BLOCK) : ComputeKernel :=
  let BLOCK:=Random.i32ArangeSize BLOCK hB
  with_triton_call signature call_args [n := n] in triton {
    acc = tl.zeros([BLOCK], tl.float32)
    for q in range(tl.cdiv(COUNT, BLOCK)):
      q = q * BLOCK
      offsets = q + tl.arange(0, BLOCK)
      value = tl.load(X + offsets, mask=offsets < n, other=0.0).to(tl.float32)
      acc = acc + value
    tl.store(Y + tl.arange(0, BLOCK), acc, mask=tl.arange(0, BLOCK) < n)
  }
end VeriTile.Tests.IntegerCallStaticLoadsReference
