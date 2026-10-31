import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.ConstexprBodyBindings
open VeriTile.Triton

def aliases (T S BT BS : Nat) : ComputeKernel :=
  with_triton_signature ({constexpr:=["T","S","BT","BS"]} : CallSignature) in triton {
    NS: tl.constexpr = S // BS
    NT: tl.constexpr = T // BT
    count = NS * NT
  }
def explicitConstants (T S BT BS : Nat) : ComputeKernel := triton {
  count = $(S/BS) * $(T/BT)
}
example (T S BT BS : Nat) : aliases T S BT BS=explicitConstants T S BT BS := by rfl

/-- error: integer constexpr expression uses a runtime or unknown name -/
#guard_msgs in
example (B : Nat) : ComputeKernel :=
  with_triton_signature ({scalars:=[("N",.integer)],constexpr:=["B"]} : CallSignature) in triton {
    blocks: tl.constexpr = N // B
    value = blocks
  }

/-- error: runtime assignment shadows an integer constexpr -/
#guard_msgs in
example (N : Nat) : ComputeKernel :=
  with_triton_signature ({constexpr:=["N"]} : CallSignature) in triton {
    N = 0
  }

/-- error: integer constexpr alias used before its binding -/
#guard_msgs in
example (N : Nat) : ComputeKernel :=
  with_triton_signature ({constexpr:=["N"]} : CallSignature) in triton {
    value = blocks
    blocks: tl.constexpr = N // 2
  }

/-- error: runtime loop shadows an integer constexpr -/
#guard_msgs in
example (N : Nat) : ComputeKernel :=
  with_triton_signature ({constexpr:=["N"]} : CallSignature) in triton {
    for N in range(0, N):
        value = N
  }
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Tests.ConstexprBodyBindings
