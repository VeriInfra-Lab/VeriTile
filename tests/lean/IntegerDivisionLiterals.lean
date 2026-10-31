import VeriTile.Triton.DSL.CallSignature

namespace VeriTile.Tests.IntegerDivisionLiterals
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("output",.i64)]
  scalars := []
  constexpr := ["N"]

def copied (output : RegionName) (N : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    lane = tl.arange(0, N)
    quotient = lane // 2
    remainder = lane % 2
    tl.store(output + lane, quotient * 2 + remainder)
  }

def explicitInteger (output : RegionName) (N : Nat) : ComputeKernel :=
  triton {
    tl.call_signature
    tl.region output = tl.int64
    lane = tl.arange(0, $(N))
    quotient = lane // $(2)
    remainder = lane % $(2)
    tl.store(output + lane, (quotient * $(2) + remainder).to(tl.int64))
  }

/-- Literal operands keep exact integer operations rather than floating division. -/
example (output : RegionName) (N : Nat) : copied output N = explicitInteger output N := by
  rfl

/-- error: runtime integer division: dtype mismatch -/
#guard_msgs in
example (output : RegionName) (N : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    value = tl.arange(0, N) // 2.0
    tl.store(output + tl.arange(0, N), value)
  }

end VeriTile.Tests.IntegerDivisionLiterals
