import VeriTile.Triton.DSL.SignedBooleanAddition
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.SignedBooleanAddition
open VeriTile.Triton
noncomputable section
set_option linter.unusedVariables false

def signature : CallSignature where
  scalars := [("M",.integer)]
  constexpr := ["BLOCK","FLAG"]

def source (M BLOCK : Nat) (FLAG : Bool) : ComputeKernel :=
  with_triton_call signature call_args [M := M] pure_functions [] in triton {
    M = tl.cast(M, tl.int64)
    count = M // BLOCK + (M % BLOCK != 0)
    reverse = (M % BLOCK != 0) + M // BLOCK
  }

/-- Complete independently typed program: Boolean comparisons become signed
0/1 before addition, retaining the source's division/remainder and casts. -/
theorem complete_program (M BLOCK : Nat) (FLAG : Bool) : source M BLOCK FLAG=
    (with_triton_call signature call_args [M := M] in triton {
      M = tl.cast(M, tl.int64)
      count = M // BLOCK + tl.cast(M % BLOCK != 0, tl.int64)
      reverse = tl.cast(M % BLOCK != 0, tl.int64) + M // BLOCK
    }) := by rfl

/-- error: Boolean addition requires a known int64 operand -/
#guard_msgs in
def unknown_operand (M BLOCK : Nat) (FLAG : Bool) : ComputeKernel :=
  with_triton_call signature call_args [M := M] pure_functions [] in triton {
    count = 1.0 + (M != 0)
  }

/-- error: Boolean addition requires a known int64 operand -/
#guard_msgs in
def branch_rebinding (M BLOCK : Nat) (FLAG : Bool) : ComputeKernel :=
  with_triton_call signature call_args [M := M] pure_functions [] in triton {
    value = tl.cast(M, tl.int64)
    if FLAG:
        value = 1.0
    count = value + (M != 0)
  }

/-- error: Boolean addition requires a known int64 operand -/
#guard_msgs in
def loop_rebinding (M BLOCK : Nat) (FLAG : Bool) : ComputeKernel :=
  with_triton_call signature call_args [M := M] pure_functions [] in triton {
    value = tl.cast(M, tl.int64)
    for value in range(0, BLOCK):
        untouched = value
    count = value + (M != 0)
  }

/-- error: Boolean addition requires a known int64 operand -/
#guard_msgs in
def tuple_rebinding (M BLOCK : Nat) (FLAG : Bool) : ComputeKernel :=
  with_triton_call signature call_args [M := M] pure_functions [] in triton {
    value = tl.cast(M, tl.int64)
    if FLAG:
        value, other = 1.0, 2.0
    count = value + (M != 0)
  }

/-- error: Boolean addition requires a known int64 operand -/
#guard_msgs in
def branch_local_rebinding (M BLOCK : Nat) (FLAG : Bool) : ComputeKernel :=
  with_triton_call signature call_args [M := M] pure_functions [] in triton {
    value = tl.cast(M, tl.int64)
    if FLAG:
        value = 1.0
        count = value + (M != 0)
  }

#axiomsClean complete_program
end
end VeriTile.Tests.SignedBooleanAddition
