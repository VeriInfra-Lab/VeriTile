import tests.lean.IntegerCallSyntaxReference
import VeriTile.Triton.DSL.IntegerCallSyntax
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.IntegerCallSyntax
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("Y",.fp32)]
  scalars := [("n",.integer)]
  constexpr := ["COUNT","BLOCK"]
  writePorts := ["Y"]

def source (Y : RegionName) (n COUNT BLOCK : Nat) (hB : Random.ArangeSizeDomain BLOCK) : ComputeKernel :=
  with_triton_call signature call_args [n := n] integer_args [] arange_args [BLOCK := hB] in triton {
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for q in tl.static_range(tl.cdiv(COUNT, BLOCK)):
      q = q * BLOCK
      offsets = q + tl.arange(0, BLOCK)
      acc = acc + offsets.to(tl.float32)
    tl.store(Y + tl.arange(0, BLOCK), acc, mask=tl.arange(0, BLOCK) < n)
  }

theorem complete_program (Y : RegionName) (n COUNT BLOCK : Nat) (hB : Random.ArangeSizeDomain BLOCK) :
    source Y n COUNT BLOCK hB=IntegerCallSyntaxReference.source Y n COUNT BLOCK hB := rfl

/-- error: static_range bound requires declared natural constexprs -/
#guard_msgs (error) in
example (Y : RegionName) (n COUNT BLOCK : Nat) (hB : Random.ArangeSizeDomain BLOCK) : ComputeKernel :=
  with_triton_call signature call_args [n := n] integer_args [] arange_args [BLOCK := hB] in triton {
    for q in tl.static_range(n):
      tl.store(Y + q, 0.0)
  }

/-- error: static_range constexprs must remain immutable -/
#guard_msgs (error) in
example (Y : RegionName) (n COUNT BLOCK : Nat) (hB : Random.ArangeSizeDomain BLOCK) : ComputeKernel :=
  with_triton_call signature call_args [n := n] integer_args [] arange_args [BLOCK := hB] in triton {
    COUNT = 1
    for q in tl.static_range(tl.cdiv(COUNT, BLOCK)):
      tl.store(Y + q, 0.0)
  }

/-- error: static_range counter cannot shadow a constexpr -/
#guard_msgs (error) in
example (Y : RegionName) (n COUNT BLOCK : Nat) (hB : Random.ArangeSizeDomain BLOCK) : ComputeKernel :=
  with_triton_call signature call_args [n := n] integer_args [] arange_args [BLOCK := hB] in triton {
    for COUNT in tl.static_range(tl.cdiv(COUNT, BLOCK)):
      tl.store(Y + COUNT, 0.0)
  }

/-- error: static_range constexprs must remain immutable -/
#guard_msgs (error) in
example (Y : RegionName) (n COUNT BLOCK : Nat) (hB : Random.ArangeSizeDomain BLOCK) : ComputeKernel :=
  with_triton_call signature call_args [n := n] integer_args [] arange_args [BLOCK := hB] in triton {
    COUNT, other = (1, 2)
    for q in tl.static_range(tl.cdiv(COUNT, BLOCK)):
      tl.store(Y + q, 0.0)
  }

/-- error: a loop cannot shadow a static_range constexpr -/
#guard_msgs (error) in
example (Y : RegionName) (n COUNT BLOCK : Nat) (hB : Random.ArangeSizeDomain BLOCK) : ComputeKernel :=
  with_triton_call signature call_args [n := n] integer_args [] arange_args [BLOCK := hB] in triton {
    for COUNT in range(0, n):
      for q in tl.static_range(tl.cdiv(COUNT, BLOCK)):
        tl.store(Y + q, 0.0)
  }

#axiomsClean complete_program
end VeriTile.Tests.IntegerCallSyntax
