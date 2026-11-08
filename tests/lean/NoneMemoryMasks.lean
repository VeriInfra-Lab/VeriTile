import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.NoneMemoryMasks
open VeriTile.Triton

def unmasked (X Y : RegionName) (B : Nat) : ComputeKernel := triton {
  tl.region X = tl.float32
  tl.region Y = tl.float32
  offsets = tl.arange(0, B)
  value = tl.load(X + offsets).to(tl.float32)
  tl.store(Y + offsets, value)
}
def positional (X Y : RegionName) (B : Nat) : ComputeKernel := triton {
  tl.region X = tl.float32
  tl.region Y = tl.float32
  offsets = tl.arange(0, B)
  value = tl.load(X + offsets, None).to(tl.float32)
  tl.store(Y + offsets, value, None)
}
def keyword (X Y : RegionName) (B : Nat) : ComputeKernel := triton {
  tl.region X = tl.float32
  tl.region Y = tl.float32
  offsets = tl.arange(0, B)
  value = tl.load(X + offsets, mask=(None)).to(tl.float32)
  tl.store(Y + offsets, value, mask=(None))
}

/-- Complete programs, including the load and destination, coincide for all
block sizes. None denotes an absent mask, preserving an unmasked access. -/
theorem positional_program : positional = unmasked := rfl
theorem keyword_program : keyword = unmasked := rfl

/-- error: tl.load: duplicate `mask=` argument -/
#guard_msgs (error) in
example (X : RegionName) : ComputeKernel := triton {
  value = tl.load(X, None, mask=True)
}

/-- error: tl.store: duplicate `mask=` kwarg -/
#guard_msgs (error) in
example (Y : RegionName) : ComputeKernel := triton {
  tl.store(Y, 1.0, None, mask=True)
}

#axiomsClean positional_program
#axiomsClean keyword_program
#auditModuleAxioms
end VeriTile.Tests.NoneMemoryMasks
