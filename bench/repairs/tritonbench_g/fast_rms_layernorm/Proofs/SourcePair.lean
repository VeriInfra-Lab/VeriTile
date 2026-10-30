import bench.repairs.tritonbench_g.fast_rms_layernorm.Kernels
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.SourcePair
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 5000000

/-- The selected mathematical sources and existing candidate agree completely
with the preserved original FP32 profile. -/
theorem regular_candidate : Kernels.regularSourceIndexCommuted = Original.forward .fp32 .fp32 Bool.true := rfl
theorem gemma_candidate : Kernels.gemmaSourceIndexCommuted = Original.gemma .fp32 .fp32 Bool.true := rfl
theorem gradient_candidate : Kernels.inputGradientSourceIndexCommuted = Original.backward .fp32 .fp32 Bool.true := rfl

/-- Proof-only decomposition of the single changed address assignment. -/
def xStep (candidate : Bool) (stride : Nat) : ComputeStmt :=
  let a : Op .nat [] := .ref .nat [] "row_idx"
  let b : Op .nat [] := .constNat stride
  .assign .ptr [] "X" (.alg (.ptrAdd .nil (.ptrBase (⟨"x"⟩ : Region .real))
    (if candidate then .mul .nat .nil b a else .mul .nat .nil a b)))

def original (h : Call) (backward : Bool) : ComputeKernel :=
  if backward then Kernels.backward h else Kernels.forwardRuntime h

def source (candidate : Bool) (h : Call) (backward : Bool) : ComputeKernel :=
  if backward then if candidate then Kernels.backwardIndexCommuted h else Kernels.backward h
  else if candidate then Kernels.forwardRuntimeIndexCommuted h else Kernels.forwardRuntime h

theorem source_split (candidate : Bool) (h : Call) (backward : Bool) :
    (source candidate h backward).surfaceBody =
      (original h backward).surfaceBody.take 4 ++ [xStep candidate h.cols] ++
      (original h backward).surfaceBody.drop 5 := by
  cases candidate <;> cases backward <;> cases mode : h.gemma <;>
    simp only [source,original,Kernels.forwardRuntime,Kernels.forwardRuntimeIndexCommuted,
      Kernels.backward,Kernels.backwardIndexCommuted,mode,Bool.false_eq_true,↓reduceIte] <;> rfl

theorem x_step {α : Type} [Inhabited α] (M : Algebra α) (s : State α) (stride : Nat) :
    step M (xStep Bool.false stride) s = step M (xStep Bool.true stride) s := by
  have product : evalOp M none
      (.mul .nat .nil (.ref .nat [] "row_idx") (.constNat stride)) s =
    evalOp M none (.mul .nat .nil (.constNat stride) (.ref .nat [] "row_idx")) s := by
    simp only [evalOp]
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

theorem source_exec {α : Type} [Inhabited α] (M : Algebra α) (s : State α) (h : Call) (backward : Bool) :
    exec M (source Bool.false h backward) s = exec M (source Bool.true h backward) s := by
  change run M (source Bool.false h backward).surfaceBody s = run M (source Bool.true h backward).surfaceBody s
  simp only [source_split,run_append,x_run]

theorem source_safe {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (bounds : RegionName → Nat) (h : Call) (backward : Bool) :
    TraceSafe M bounds (source Bool.false h backward) s ↔ TraceSafe M bounds (source Bool.true h backward) s := by
  change Safety.traceSafeList M bounds (source Bool.false h backward).surfaceBody s ↔
    Safety.traceSafeList M bounds (source Bool.true h backward).surfaceBody s
  simp only [source_split,Safety.traceSafeList_append,run_append,x_run,x_safe]
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.SourcePair
