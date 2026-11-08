import VeriTile.Triton.DSL
namespace VeriTile.Tests.SingleAssignmentPointerOffsetReference
open VeriTile.Triton
/-- Independent typed address, combining the caller base and complete signed
first displacement before conversion to a physical cell coordinate. -/
def source (X Y : RegionName) (base : Nat) (delta : Int) : ComputeKernel := triton {
  tl.call_signature
  tl.ports inputs=[X], outputs=[Y]
  tl.region X = tl.float32, Y = tl.float32
  X = (X + $((base : Nat))) + $((delta : Int))
  value = tl.load(X, dtype=tl.float32)
  tl.store(Y, value)
}
/-- Runtime displacements remain actual integer inputs, rather than caller
constants. The caller base is combined before the physical address conversion. -/
def runtime (X Y : RegionName) (base : Nat) : ComputeKernel := triton {
  tl.call_signature
  tl.ports inputs=[X], outputs=[Y]
  tl.region X = tl.float32, Y = tl.float32
  tl.param delta : tl.int64
  X = (X + $((base : Nat))) + delta
  value = tl.load(X, dtype=tl.float32)
  tl.store(Y, value)
}
end VeriTile.Tests.SingleAssignmentPointerOffsetReference
