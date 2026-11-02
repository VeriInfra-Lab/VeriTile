import VeriTile.Triton.DSL.SignedSingleArgumentRange
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.SignedSingleArgumentRange
open VeriTile.Triton
noncomputable section
set_option linter.unusedVariables false

def signature : CallSignature where
  scalars := [("N",.integer)]
  constexpr := ["BLOCK","FLAG"]

/-- Complete signed source, including the original argument cast and body
assignments. Negative bounds retain the signed range's zero-iteration case. -/
def source (N : Int) (BLOCK : Nat) (FLAG : Bool) : ComputeKernel :=
  with_triton_call signature call_args [N := N] in triton {
    N = tl.cast(N, tl.int64)
    for k in range((N + BLOCK - 1) // BLOCK):
        marker = k
  }

theorem complete_program (N : Int) (BLOCK : Nat) (FLAG : Bool) :
    source N BLOCK FLAG = (with_triton_call signature call_args [N := N] in triton {
      N = tl.cast(N, tl.int64)
      for k in range(tl.cast(0, tl.int64), (N + BLOCK - 1) // BLOCK, 1):
          marker = k
    }) := by rfl

theorem natural_program (N BLOCK : Nat) (FLAG : Bool) :
    (with_triton_call signature call_args [N := N] in triton {
      for k in range(N):
          marker = k
    }) = ComputeKernel.mk [] []
      [.forLoop "k" N [.assign .nat [] "marker" (.alg (.ref .nat [] "k"))]] := by rfl


/-- error: int expression: dtype mismatch -/
#guard_msgs in
def floating_rebinding (N : Int) (BLOCK : Nat) (FLAG : Bool) : ComputeKernel :=
  with_triton_call signature call_args [N := N] in triton {
    bound = tl.cast(N, tl.int64)
    bound = 1.0
    for k in range(bound):
        marker = k
  }

/-- error: int expression: dtype mismatch -/
#guard_msgs in
def branch_rebinding (N : Int) (BLOCK : Nat) (FLAG : Bool) : ComputeKernel :=
  with_triton_call signature call_args [N := N] in triton {
    bound = tl.cast(N, tl.int64)
    if FLAG:
        bound = 1.0
    for k in range(bound):
        marker = k
  }

#axiomsClean complete_program
#axiomsClean natural_program
end
end VeriTile.Tests.SignedSingleArgumentRange
