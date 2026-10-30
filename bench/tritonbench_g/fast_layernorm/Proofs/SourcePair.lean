import bench.tritonbench_g.fast_layernorm.Kernels
import VeriTile.Triton.Float.ProgramEquivalenceSyntax
namespace VeriTile.Bench.Repairs.FastLayernorm.SourcePair
open VeriTile Triton FP.Structural
noncomputable section
/-- Proof-only decomposition of the single changed address assignment. -/
def xStep (candidate : Bool) (stride : Nat) : ComputeStmt :=
  let a : Op .nat [] := .ref .nat [] "row_idx"
  let b : Op .nat [] := .constNat stride
  .assign .ptr [] "X" (.alg (.ptrAdd .nil (.ptrBase (⟨"x"⟩ : Region .real))
    (if candidate then .mul .nat .nil b a else .mul .nat .nil a b)))

theorem x_step {α : Type} [Inhabited α] (M : Algebra α) (s : State α) (stride : Nat) :
    step M (xStep Bool.false stride) s = step M (xStep Bool.true stride) s := by
  have product : evalOp M none
      (.mul .nat .nil (.ref .nat [] "row_idx") (.constNat stride)) s =
    evalOp M none (.mul .nat .nil (.constNat stride) (.ref .nat [] "row_idx")) s := by
    simp only [FP.Structural.evalOp]
    cases hr : s.regs .nat [] "row_idx" <;> simp_all [numeric]
    funext i
    exact Nat.mul_comm _ _
  simp only [xStep,Bool.false_eq_true,↓reduceIte,step,evalExpr]
  change ((evalOp M none (.mul .nat .nil (.ref .nat [] "row_idx") (.constNat stride)) s).bind
    (fun offsets => some (bop (fun (p : RegionName × Nat) n => (p.1,p.2+n))
      .nil (fun _ => ("x",0)) offsets))).bind _ = _
  rw [product]
  rfl

theorem x_run {α : Type} [Inhabited α] (M : Algebra α) (stride : Nat) :
    run M [xStep Bool.false stride] = run M [xStep Bool.true stride] := by
  funext s
  simp only [run,x_step]

theorem x_safe {α : Type} [Inhabited α] (M : Algebra α) (bounds : RegionName → Nat)
    (candidate : Bool) (stride : Nat) (s : State α) :
    Safety.traceSafeList M bounds [xStep candidate stride] s := by
  rw [Safety.traceSafeList]
  constructor
  · cases candidate <;> simp [xStep,Safety.stepSafe,Safety.safeExpr,Safety.safeOp]
  · cases step M (xStep candidate stride) s <;> simp


def forward (candidate : Bool) (N B : Nat) (eps : ℝ) (bias : Bool) :=
  if candidate then forwardIndexCommuted "y" "x" "w" "b" "r" "mu" N B eps bias
  else forwardKernel "y" "x" "w" "b" "r" "mu" N B eps bias

def backward (candidate : Bool) (N B : Nat) :=
  if candidate then backwardIndexCommuted "dx" "dy" "x" "w" "r" "mu" N B
  else backwardKernel "dx" "dy" "x" "w" "r" "mu" N B

theorem forward_split (candidate : Bool) (N B : Nat) (eps : ℝ) (bias : Bool) :
    (forward candidate N B eps bias).surfaceBody =
      (forwardKernel "y" "x" "w" "b" "r" "mu" N B eps bias).surfaceBody.take 4 ++
      [xStep candidate N] ++
      (forwardKernel "y" "x" "w" "b" "r" "mu" N B eps bias).surfaceBody.drop 5 := by
  cases candidate <;> cases bias <;> rfl

theorem backward_split (candidate : Bool) (N B : Nat) :
    (backward candidate N B).surfaceBody =
      (backwardKernel "dx" "dy" "x" "w" "r" "mu" N B).surfaceBody.take 5 ++
      [xStep candidate N] ++
      (backwardKernel "dx" "dy" "x" "w" "r" "mu" N B).surfaceBody.drop 6 := by
  cases candidate <;> rfl

theorem forward_exec {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (N B : Nat) (eps : ℝ) (bias : Bool) :
    exec M (forward Bool.false N B eps bias) s = exec M (forward Bool.true N B eps bias) s := by
  change run M (forward Bool.false N B eps bias).surfaceBody s = run M (forward Bool.true N B eps bias).surfaceBody s
  simp only [forward_split,run_append,x_run]

theorem backward_exec {α : Type} [Inhabited α] (M : Algebra α) (s : State α) (N B : Nat) :
    exec M (backward Bool.false N B) s = exec M (backward Bool.true N B) s := by
  change run M (backward Bool.false N B).surfaceBody s = run M (backward Bool.true N B).surfaceBody s
  simp only [backward_split,run_append,x_run]

theorem forward_safe {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (bounds : RegionName → Nat) (N B : Nat) (eps : ℝ) (bias : Bool) :
    TraceSafe M bounds (forward Bool.false N B eps bias) s ↔ TraceSafe M bounds (forward Bool.true N B eps bias) s := by
  change Safety.traceSafeList M bounds (forward Bool.false N B eps bias).surfaceBody s ↔
    Safety.traceSafeList M bounds (forward Bool.true N B eps bias).surfaceBody s
  simp only [forward_split,Safety.traceSafeList_append,run_append,x_run,x_safe]

theorem backward_safe {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (bounds : RegionName → Nat) (N B : Nat) :
    TraceSafe M bounds (backward Bool.false N B) s ↔ TraceSafe M bounds (backward Bool.true N B) s := by
  change Safety.traceSafeList M bounds (backward Bool.false N B).surfaceBody s ↔
    Safety.traceSafeList M bounds (backward Bool.true N B).surfaceBody s
  simp only [backward_split,Safety.traceSafeList_append,run_append,x_run,x_safe]
end
end VeriTile.Bench.Repairs.FastLayernorm.SourcePair
