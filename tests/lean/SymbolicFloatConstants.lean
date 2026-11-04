import VeriTile.Triton.Float.Structural
import VeriTile.Triton.DSL.SymbolicFloatConstants
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic.FinCases

namespace VeriTile.Tests.SymbolicFloatConstants
open VeriTile.Triton FP.Structural
noncomputable section
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false
set_option maxRecDepth 30000
set_option maxHeartbeats 3000000

def signature : CallSignature where
  scalars := [("N",.integer)]
  constexpr := ["B","EPS"]

def iterations (N B : Nat) : ComputeKernel :=
  with_triton_call signature call_args [N := N] fp32_constants [EPS] in triton {
    N = tl.cast(N, tl.int64)
    count = N // B + (N % B != 0)
    total = tl.cast(0, tl.int64)
    for step in range(0, count):
        total += 1
  }

/-- Ceiling division drives the actual signed source loop. -/
theorem three_iterations {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    ∃ t, exec M (iterations 9 4) s=some t ∧
      t.regs .int [] "total"=some (fun _=>3) ∧ t.mem=s.mem := by
  simp (config := {maxSteps:=20000}) [iterations,exec,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,numeric,numericEq,bop,numericLt,natLt,
    integralFloorDiv,integralMod,State.setReg,ComputeDType.eraseDType,
    Broadcast.leftIndex,Broadcast.rightIndex]
  rw [rangeInt]
  simp [run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,numeric,numericEq,bop,
    numericLt,natLt,State.setReg,ComputeDType.eraseDType]
  rw [rangeInt]
  simp [run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,numeric,numericEq,bop,
    numericLt,natLt,State.setReg,ComputeDType.eraseDType]
  rw [rangeInt]
  simp [run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,numeric,numericEq,bop,
    numericLt,natLt,State.setReg,ComputeDType.eraseDType]
  rw [rangeInt]
  simp [State.setReg,bop,Broadcast.leftIndex,Broadcast.rightIndex]
  funext i
  cases i
  rfl

def tuple (N B : Nat) : ComputeKernel :=
  with_triton_call signature call_args [N := N] fp32_constants [EPS] in triton {
    __veritile_split_operand_0 = tl.arange(0, 1)
    real = tl.arange(0, 4)
    real, imag = tl.split(tl.reshape(real, (2, 1, 2)))
    saved_constant = EPS
  }

/-- Split reads the old destination, evaluates its operand once, and uses a
fresh temporary. The immutable scalar is the caller's exact materialization. -/
theorem tuple_capture {α : Type} [Inhabited α] (M : Algebra α) (s : State α) (epsilon : α)
    (h : s.regs .real [] "EPS"=some (fun _=>epsilon)) :
    ∃ t, exec M (tuple 9 4) s=some t ∧
      t.regs .nat [2,1] "real"=some (fun i=>2*i.1.val) ∧
      t.regs .nat [2,1] "imag"=some (fun i=>2*i.1.val+1) ∧
      t.regs .nat [1] "__veritile_split_operand_0"=some (fun _=>0) ∧
      t.regs .real [] "saved_constant"=some (fun _=>epsilon) ∧ t.mem=s.mem := by
  simp (config := {maxSteps:=20000}) [tuple,exec,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,numeric,numericEq,bop,numericLt,natLt,
    State.setReg,h,ComputeDType.eraseDType,Broadcast.leftIndex,Broadcast.rightIndex]
  constructor <;> funext i
  all_goals
    rcases i with ⟨row,column,unitIndex⟩
    fin_cases row <;> fin_cases column <;> cases unitIndex <;> rfl

def weakDivision (N : Int) (B : Nat) : ComputeKernel :=
  with_triton_call signature call_args [N := N] fp32_constants [EPS] in triton {
    result = N.to(tl.float32) / B
  }

/-- The constexpr denominator uses the literal port, even for a backend whose
literal materialization and runtime integer conversion disagree. -/
theorem weak_denominator {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (N : Int) (B : Nat) :
    ∃ t, exec M (weakDivision N B) s=some t ∧
      t.regs .real [] "result"=some (fun _=>
        M.binary (some .fp32) .real .div (M.fromInt (some .fp32) N)
          (M.literal (some .fp32) .real (B:ℝ))) ∧ t.mem=s.mem := by
  simp [weakDivision,exec,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,numeric,FP.Structural.bop,Option.bind,State.setReg]
  rfl

def runtimeSignature : CallSignature where
  scalars := [("N",.integer),("B",.integer)]
  constexpr := ["EPS"]

def runtimeDivision (N B : Int) : ComputeKernel :=
  with_triton_call runtimeSignature call_args [N := N,B := B] fp32_constants [EPS] in triton {
    result = N.to(tl.float32) / B
  }

/-- An identically spelled runtime denominator keeps its integer conversion. -/
theorem runtime_denominator {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (N B : Int) :
    ∃ t, exec M (runtimeDivision N B) s=some t ∧
      t.regs .real [] "result"=some (fun _=>
        M.binary (some .fp32) .real .div (M.fromInt (some .fp32) N)
          (M.fromInt (some .fp32) B)) ∧ t.mem=s.mem := by
  simp [runtimeDivision,exec,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,numeric,FP.Structural.bop,Option.bind,State.setReg]
  rfl

theorem reject_constexpr_rebinding (N B : Nat) : True := by
  fail_if_success
    let invalid : ComputeKernel :=
      with_triton_call signature call_args [N := N] fp32_constants [EPS] in triton {
        EPS = 1.0
      }
  trivial

theorem reject_constexpr_loop_shadowing (N B : Nat) : True := by
  fail_if_success
    let invalid : ComputeKernel :=
      with_triton_call signature call_args [N := N] fp32_constants [EPS] in triton {
        for EPS in range(0, B):
            value = EPS
      }
  trivial

theorem reject_constexpr_annotation (N B : Nat) : True := by
  fail_if_success
    let invalid : ComputeKernel :=
      with_triton_call signature call_args [N := N] fp32_constants [EPS] in triton {
        EPS: tl.constexpr = 1
      }
  trivial

theorem reject_runtime_as_constexpr (N B : Nat) : True := by
  fail_if_success
    let invalid : ComputeKernel :=
      with_triton_call signature call_args [N := N] fp32_constants [N] in triton {
        value = N
      }
  trivial

#axiomsClean three_iterations
#axiomsClean tuple_capture
#axiomsClean weak_denominator
#axiomsClean runtime_denominator
end
end VeriTile.Tests.SymbolicFloatConstants
