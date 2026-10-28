import bench.repairs.tritonbench_g.kv_cache_filling.Kernels
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

/-! Proof-only decomposition of the actual complete policy source. No kernel
is built from these lists; full execution is connected below for arbitrary
models and states, including failed executions. -/
namespace VeriTile.Bench.Repairs.KvCacheFilling.SourceBlocks
open VeriTile.Triton FP.Structural
set_option maxRecDepth 30000
set_option maxHeartbeats 3000000

def branch (selected : Bool) : List ComputeStmt → List ComputeStmt
  | [.ifThenElse _ yes no] => if selected then yes else no
  | _ => []

def body8 (version : Bool) (p : CopySource.Parameters) : List ComputeStmt :=
  let source := (QuantSource.kernel8 version p).surfaceBody
  source.take 7 ++ branch Bool.true (branch Bool.false (source.drop 7))

def body4 (version : Bool) (p : CopySource.Parameters) : List ComputeStmt :=
  let source := (QuantSource.kernel4 version p).surfaceBody
  source.take 7 ++ branch Bool.false (branch Bool.false (source.drop 7))

private theorem constant_branch {α : Type} [Inhabited α] (M : Algebra α)
    (a b : Nat) (yes no : List ComputeStmt) (s : State α) :
    run M [.ifThenElse (.eq .nat .nil (.constNat a) (.constNat b)) yes no] s =
      run M (if a = b then yes else no) s := by
  by_cases h : a = b <;>
    simp [run, step, evalOp_unfold, numericEq, bop,
      Broadcast.leftIndex, Broadcast.rightIndex, h]

private theorem tail8 {α : Type} [Inhabited α] (M : Algebra α)
    (version : Bool) (p : CopySource.Parameters) (s : State α) :
    run M ((QuantSource.kernel8 version p).surfaceBody.drop 7) s =
      run M (branch Bool.true (branch Bool.false
        ((QuantSource.kernel8 version p).surfaceBody.drop 7))) s := by
  cases version <;>
    simp only [QuantSource.kernel8, QuantSource.original8Kernel, QuantSource.commuted8Kernel,
      ComputeKernel.surfaceBody, List.drop, branch, Bool.false_eq_true, if_false, if_true]
  all_goals simp only [constant_branch, Nat.reduceEqDiff, ↓reduceIte]

private theorem tail4 {α : Type} [Inhabited α] (M : Algebra α)
    (version : Bool) (p : CopySource.Parameters) (s : State α) :
    run M ((QuantSource.kernel4 version p).surfaceBody.drop 7) s =
      run M (branch Bool.false (branch Bool.false
        ((QuantSource.kernel4 version p).surfaceBody.drop 7))) s := by
  cases version <;>
    simp only [QuantSource.kernel4, QuantSource.original4Kernel, QuantSource.commuted4Kernel,
      ComputeKernel.surfaceBody, List.drop, branch, Bool.false_eq_true, if_false, if_true]
  all_goals simp only [constant_branch, Nat.reduceEqDiff, ↓reduceIte]

theorem execute8 {α : Type} [Inhabited α] (M : Algebra α)
    (version : Bool) (p : CopySource.Parameters) (s : State α) :
    exec M (QuantSource.kernel8 version p) s = run M (body8 version p) s := by
  change run M (QuantSource.kernel8 version p).surfaceBody s = _
  unfold body8
  conv_lhs => rw [← List.take_append_drop 7 (QuantSource.kernel8 version p).surfaceBody]
  simp only [run_append]
  congr 1
  funext t
  exact tail8 M version p t

theorem execute4 {α : Type} [Inhabited α] (M : Algebra α)
    (version : Bool) (p : CopySource.Parameters) (s : State α) :
    exec M (QuantSource.kernel4 version p) s = run M (body4 version p) s := by
  change run M (QuantSource.kernel4 version p).surfaceBody s = _
  unfold body4
  conv_lhs => rw [← List.take_append_drop 7 (QuantSource.kernel4 version p).surfaceBody]
  simp only [run_append]
  congr 1
  funext t
  exact tail4 M version p t

#axiomsClean execute8
#axiomsClean execute4
end VeriTile.Bench.Repairs.KvCacheFilling.SourceBlocks
