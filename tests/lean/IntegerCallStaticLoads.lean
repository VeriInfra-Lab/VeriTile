import tests.lean.IntegerCallStaticLoadsReference
import VeriTile.Triton.DSL.IntegerCallSyntax
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.IntegerCallStaticLoads
open VeriTile.Triton

def source (X Y : RegionName) (n COUNT BLOCK : Nat) (hB : Random.ArangeSizeDomain BLOCK) : ComputeKernel :=
  with_triton_call IntegerCallStaticLoadsReference.signature call_args [n := n]
    integer_args [] arange_args [BLOCK := hB] in triton {
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for q in tl.static_range(tl.cdiv(COUNT, BLOCK)):
      q = q * BLOCK
      offsets = q + tl.arange(0, BLOCK)
      value = tl.load(X + offsets, mask=offsets < n, other=0.0).to(tl.float32)
      acc = acc + value
    tl.store(Y + tl.arange(0, BLOCK), acc, mask=tl.arange(0, BLOCK) < n)
  }

/-- Complete-program equality retains the scoped FP32 load, private register,
explicit cast, integer counter remapping, accumulation and implicit store. -/
theorem complete_program (X Y : RegionName) (n COUNT BLOCK : Nat) (hB : Random.ArangeSizeDomain BLOCK) :
    source X Y n COUNT BLOCK hB=IntegerCallStaticLoadsReference.source X Y n COUNT BLOCK hB := rfl

#axiomsClean complete_program
end VeriTile.Tests.IntegerCallStaticLoads
