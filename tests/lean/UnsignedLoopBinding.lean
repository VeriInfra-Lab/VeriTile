import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.UnsignedLoopBinding
open VeriTile.Triton

def signature : CallSignature where
  scalars := [("N",.integer)]
  constexpr := ["stride"]

/-- Rebinding a signed ABI parameter through uint64 determines the current
loop type, even though the original signature and pins were signed. -/
def unsignedLoop (stride : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    N = tl.cast(N, tl.uint64)
    i = tl.full([], 0, dtype=tl.uint64)
    for i in range(i, N, $(stride)) {
      i += $(stride)
    }
  }

def signedLoop (stride : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    for i in range($(0), N, $(stride)) {
      copy = i
    }
  }

theorem latest_unsigned_binding (stride : Nat) :
    (match (unsignedLoop stride).surfaceBody[2]? with
      | some (ComputeStmt.forRangeDyn _ _ _ _ _) => True
      | _ => False) := by trivial

theorem signed_binding_retained (stride : Nat) :
    (match (signedLoop stride).surfaceBody[0]? with
      | some (ComputeStmt.forRangeInt _ _ _ _ _) => True
      | _ => False) := by trivial

#axiomsClean latest_unsigned_binding
#axiomsClean signed_binding_retained
end VeriTile.Tests.UnsignedLoopBinding
