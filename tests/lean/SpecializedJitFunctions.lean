import VeriTile.Triton.DSL.SpecializedJitFunctions
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.SpecializedJitFunctions
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000

def staged : SpecializedStatementJitFunction :=
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

def caller (x : Nat) : ComputeKernel :=
  let STAGE : Nat:=3
  with_triton_call signature call_args [x := x] jit_functions [inner := staged] in triton {
    __veritile_specialized_jit_0 = x
    __veritile_python_underscore = x
    lo, hi = inner(x, 4 - STAGE)
    second_lo, _ = inner(x, 2)
    second_hi = _
    saved = __veritile_specialized_jit_0 + __veritile_python_underscore
  }

/-- Both actual call sites choose their own branch; tuple results are
captured before caller destinations change. No cast or arithmetic
rewrite assumption is required. -/
theorem two_stages {α : Type} [Inhabited α] (M : Algebra α) (x : Nat) (s : State α) :
    ∃ t, exec M (caller x) s=some t ∧
      t.regs .real [] "lo"=some (fun _=>M.literal none .real 0) ∧
      t.regs .nat [] "hi"=some (fun _=>x) ∧
      t.regs .nat [] "second_lo"=some (fun _=>x) ∧
      t.regs .nat [] "second_hi"=some (fun _=>x+1) ∧
      t.regs .nat [] "saved"=some (fun _=>x+x) := by
  simp [caller,exec,ComputeKernel.surfaceBody,run,step,evalExpr,
    evalComputeOp,evalOp_unfold,Option.bind,numeric,FP.Structural.bop,
    Broadcast.leftIndex,Broadcast.rightIndex]
  exact ⟨rfl,rfl⟩

def runtimeBranch : SpecializedStatementJitFunction :=
  triton_function (x, STAGE) constexpr_args [STAGE] {
    value = x
    if x < 3:
        value = x + 1
    return value
  }

def runtimeCaller (x : Nat) : ComputeKernel :=
  let STAGE : Nat:=3
  with_triton_call signature call_args [x := x] jit_functions [branch := runtimeBranch] in triton {
    output = branch(x, 4 - STAGE)
  }

/-- A runtime test survives specialization even when the helper has a
closed constexpr argument. -/
theorem runtime_guard {α : Type} [Inhabited α] (M : Algebra α) (x : Nat) (s : State α) :
    ∃ t, exec M (runtimeCaller x) s=some t ∧
      t.regs .nat [] "output"=some (fun _=>if x<3 then x+1 else x) := by
  by_cases h : x<3 <;>
    simp [runtimeCaller,exec,ComputeKernel.surfaceBody,run,step,evalExpr,
      evalComputeOp,evalOp_unfold,Option.bind,numeric,numericLt,natLt,
      FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,h] <;> rfl

def identityBody : SpecializedStatementJitFunction :=
  triton_function (x, STAGE) constexpr_args [STAGE] {
    return x
  }

def floatSignature : CallSignature where
  scalars := [("x",.fp32)]
  constexpr := ["STAGE"]

def floatCaller (x : ℝ) : ComputeKernel :=
  let STAGE : Nat:=3
  with_triton_call floatSignature call_args [x := x] jit_functions [identity := identityBody] in triton {
    output = identity(x, 4 - STAGE)
  }

/-- Keep the materialized FP32 ABI argument, without assuming any cast or
load callback is an identity. -/
theorem fp32_argument {α : Type} [Inhabited α] (M : Algebra α) (x : ℝ) (s : State α) :
    ∃ t, exec M (floatCaller x) s=some t ∧
      t.regs .real [] "output"=some (fun _=>M.literal (some .fp32) .real x) := by
  simp [floatCaller,ComputeKernel.withScalarArguments,ScalarArgument.fp32,
    ScalarArgument.statement,exec,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType]

/-- error: JIT helper references an unbound name: lo -/
#guard_msgs in
def unsupportedStage (x : Nat) : ComputeKernel :=
  let STAGE : Nat:=3
  with_triton_call signature call_args [x := x] jit_functions [inner := staged] in triton {
    lo, hi = inner(x, STAGE)
  }

/-- error: specialized JIT constexpr arguments must remain immutable -/
#guard_msgs in
def mutated : SpecializedStatementJitFunction :=
  triton_function (x, STAGE) constexpr_args [STAGE] {
    if STAGE == 1:
        STAGE = 2
    return x
  }

#axiomsClean two_stages
#axiomsClean runtime_guard
#axiomsClean fp32_argument
end
end VeriTile.Tests.SpecializedJitFunctions
