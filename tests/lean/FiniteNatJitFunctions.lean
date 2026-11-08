import VeriTile.Triton.DSL.FiniteNatJitFunctions
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.FiniteNatJitFunctions
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000

def helper : SpecializedStatementJitFunction :=
  triton_function (x, STAGE) constexpr_args [STAGE] {
    if STAGE == 1:
        lo, hi = 0, x
    elif STAGE == 2:
        lo, hi = x, x + 1
    return lo, hi
  }

def signature : CallSignature where
  scalars := [("x",.integer)]
  constexpr := ["STAGE"]

def caller (x : Nat) (diagonal : Bool) : ComputeKernel :=
  let STAGE : Nat:=if diagonal then 2 else 3
  with_triton_call signature call_args [x := x] jit_functions [inner := helper] in triton {
    lo, hi = inner(x, 4 - STAGE)
  }

/-- Both legal caller specializations retain the source expression 4-STAGE.
The arbitrary backend receives the same original integer operations. -/
theorem both_stages {α : Type} [Inhabited α] (M : Algebra α)
    (x : Nat) (diagonal : Bool) (s : State α) :
    ∃ t, exec M (caller x diagonal) s=some t ∧
      t.regs .nat [] "hi"=some (fun _=>if diagonal then x+1 else x) := by
  cases diagonal <;>
    simp [caller,exec,ComputeKernel.surfaceBody,run,step,evalExpr,
      evalComputeOp,evalOp_unfold,Option.bind,numeric,integralMod,natLt,bop,State.setReg] <;> rfl

#axiomsClean both_stages

/-- Python's integer zero must never enter the floating register channel
when tuple assignment is lowered through helper temporaries. -/
theorem integer_zero {α : Type} [Inhabited α] (M : Algebra α)
    (x : Nat) (s : State α) :
    ∃ t, exec M (caller x Bool.false) s=some t ∧
      t.regs .int [] "lo"=some (fun _=>0) := by
  simp [caller,exec,ComputeKernel.surfaceBody,run,step,evalExpr,
    evalComputeOp,evalOp_unfold,Option.bind,numeric,integralMod,natLt,bop,State.setReg]

#axiomsClean integer_zero

def nestedHelper : SpecializedStatementJitFunction :=
  triton_function (x, STAGE) constexpr_args [STAGE] {
    value = x
    for i in range(0, 2):
        if STAGE == 2:
            value = value + 1
        else:
            value = value + 2
    return value
  }
def nestedCaller (x : Nat) (diagonal : Bool) : ComputeKernel :=
  let STAGE : Nat:=if diagonal then 2 else 3
  with_triton_call signature call_args [x := x] jit_functions [inner := nestedHelper] in triton {
    output = inner(x, 4 - STAGE)
  }

/-- A constexpr test inside a loop is static even for an arithmetic backend
that supplies no floating equality operation. Both caller stages are checked. -/
theorem nested_stages {α : Type} [Inhabited α] (M : Algebra α)
    (x : Nat) (diagonal : Bool) (s : State α) :
    ∃ t, exec M (nestedCaller x diagonal) s=some t ∧
      t.regs .nat [] "output"=some (fun _=>if diagonal then x+2 else x+4) := by
  cases diagonal <;>
    simp [nestedCaller,exec,ComputeKernel.surfaceBody,run,step,loop,range,
      evalExpr,evalComputeOp,evalOp_unfold,Option.bind,numeric,bop,State.setReg]
  all_goals rfl

#axiomsClean nested_stages

def widthsHelper : SpecializedStatementJitFunction :=
  triton_function (x, STAGE) constexpr_args [STAGE] {
    a = 2147483648
    b = 4294967296
    c = 2147483647
    return a, b, c
  }

def widthsCaller (x : Nat) (diagonal : Bool) : ComputeKernel :=
  let STAGE : Nat:=if diagonal then 2 else 3
  with_triton_call signature call_args [x := x] jit_functions [inner := widthsHelper] in triton {
    a, b, c = inner(x, STAGE)
  }

/-- The uint32 interval retains the unsigned integer channel; values beyond
the int32 intervals retain signed 64-bit values. -/
theorem literal_widths {α : Type} [Inhabited α] (M : Algebra α)
    (x : Nat) (diagonal : Bool) (s : State α) :
    ∃ t, exec M (widthsCaller x diagonal) s=some t ∧
      (∃ a, t.regs .nat [] "a"=some a) ∧
      (∃ b, t.regs .int [] "b"=some b) ∧
      (∃ c, t.regs .int [] "c"=some c) := by
  cases diagonal <;>
    simp [widthsCaller,exec,ComputeKernel.surfaceBody,run,step,evalExpr,
      evalComputeOp,evalOp_unfold,Option.bind,numeric,bop,State.setReg,natLt]


#axiomsClean literal_widths

def oversizedHelper : SpecializedStatementJitFunction :=
  triton_function (x, STAGE) constexpr_args [STAGE] {
    value = 9223372036854775808
    return value
  }

/- Unsupported unsigned-64 bindings fail during elaboration. -/
#check_failure (fun (x : Nat) =>
  let STAGE : Nat:=2
  with_triton_call signature call_args [x := x] jit_functions [inner := oversizedHelper] in triton {
    output = inner(x, STAGE)
  } : Nat → ComputeKernel)

end
end VeriTile.Tests.FiniteNatJitFunctions
