import VeriTile.Triton.DSL.CastReverseRange
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.CastReverseRange
open VeriTile.Triton
noncomputable section

def profile : CallSignature where
  pointers := [("output",.fp32)]
  constexpr := ["count"]
  writePorts := ["output"]

def explicitCast (output : RegionName) (count : Nat) : ComputeKernel :=
  with_triton_signature profile in triton {
    for j in range(tl.cast(count - 1, tl.int64), tl.cast(-1, tl.int64), -1):
        tl.store(output + j, 1.0)
  }

def mirrored (output : RegionName) (count : Nat) : ComputeKernel :=
  with_triton_signature profile in triton {
    for j in range(tl.cast(count - 1, tl.int64), -tl.cast(1, tl.int64), -1):
        tl.store(output + j, 1.0)
  }

/-- Exact typed IR equality includes the captured bounds, signed counter,
all stores and the signed-counter lowering. -/
theorem cast_stop_exact (output : RegionName) (count : Nat) :
    explicitCast output count=mirrored output count := by rfl
#axiomsClean cast_stop_exact

/-- Accepted scan phases have a positive span. On that existing domain the
constexpr subtraction agrees with Python integer subtraction before int64. -/
theorem positive_start_exact (count : Nat) (positive : 0<count) :
    ((count-1:Nat):Int)=(count:Int)-1 := by omega
#axiomsClean positive_start_exact

/-- error: cast reverse-range stop must fit signed int64 -/
#guard_msgs in
example (output : RegionName) (count : Nat) : ComputeKernel :=
  with_triton_signature profile in triton {
    for j in range(tl.cast(count - 1, tl.int64), tl.cast(-9223372036854775808, tl.int64), -1):
        tl.store(output + j, 1.0)
  }

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Tests.CastReverseRange
