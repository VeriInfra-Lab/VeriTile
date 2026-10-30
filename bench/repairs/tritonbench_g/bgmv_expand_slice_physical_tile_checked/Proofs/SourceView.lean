import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Kernels
import VeriTile.Triton.Float.StructuralSafety
import VeriTile.Meta.StatementAudit

/-! Proof-only decomposition of the canonical raw source. The optional input
narrowing is a known-true constexpr at this wrapper call. The source itself
retains its full branch; execution and safety connections select that branch. -/
namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.SourceView
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 500000
set_option linter.unusedSimpArgs false

def rawCode (h : Call) : List ComputeStmt := (Kernels.source h).surfaceBody
def rawActive (h : Call) : List ComputeStmt :=
  match (rawCode h).getD 3 (.effectMarker "absent") with
  | .ifThen _ body => body
  | _ => []
def selected (h : Call) : List ComputeStmt :=
  match (rawActive h).getD 4 (.effectMarker "absent") with
  | .ifThenElse _ yes _ => yes
  | _ => []
def unused (h : Call) : List ComputeStmt :=
  match (rawActive h).getD 4 (.effectMarker "absent") with
  | .ifThenElse _ _ no => no
  | _ => []
def active (h : Call) : List ComputeStmt := (rawActive h).take 4 ++ selected h
def guard : Op .bool [] :=
  .boolNot (.eq .int .nil (.ref .int [] "lora_index") (.constInt (-1)))
def code (h : Call) : List ComputeStmt :=
  (rawCode h).take 3 ++ [.ifThen guard (active h)]

theorem raw_decomposition (h : Call) :
    rawCode h = (rawCode h).take 3 ++ [.ifThen guard (rawActive h)] := by rfl
theorem active_decomposition (h : Call) :
    rawActive h = (rawActive h).take 4 ++
      [.ifThenElse (.constBool Bool.true) (selected h) (unused h)] := by rfl

variable {α : Type} [Inhabited α]

theorem run_constant_true (M : Algebra α) (yes no : List ComputeStmt) (s : State α) :
    run M [.ifThenElse (.constBool Bool.true) yes no] s = run M yes s := by
  simp only [run,step,evalOp_unfold,Option.bind_some]
  cases run M yes s <;> rfl

theorem safe_constant_true (M : Algebra α) (yes no : List ComputeStmt)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList M bounds [.ifThenElse (.constBool Bool.true) yes no] s ↔
      Safety.traceSafeList M bounds yes s := by
  simp [Safety.traceSafeList_cons,Safety.traceSafeList_nil,
    Safety.stepSafe_unfold,Safety.safeOp_unfold,evalOp_unfold]

theorem run_active (M : Algebra α) (h : Call) (s : State α) :
    run M (rawActive h) s = run M (active h) s := by
  rw [active_decomposition]
  simp only [active,run_append,run_constant_true]

theorem safe_active (M : Algebra α) (h : Call) (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList M bounds (rawActive h) s ↔ Safety.traceSafeList M bounds (active h) s := by
  rw [active_decomposition]
  simp only [active,Safety.traceSafeList_append,safe_constant_true]

theorem run_if_congr (M : Algebra α) (condition : Op .bool [])
    (yes other : List ComputeStmt) (same : ∀ s,run M yes s = run M other s) (s : State α) :
    run M [.ifThen condition yes] s = run M [.ifThen condition other] s := by
  simp only [run,step]
  simp_rw [same]

theorem safe_if_congr (M : Algebra α) (condition : Op .bool [])
    (yes other : List ComputeStmt) (bounds : RegionName → Nat)
    (same : ∀ s,Safety.traceSafeList M bounds yes s ↔ Safety.traceSafeList M bounds other s)
    (s : State α) :
    Safety.traceSafeList M bounds [.ifThen condition yes] s ↔
      Safety.traceSafeList M bounds [.ifThen condition other] s := by
  simp [Safety.traceSafeList_cons,Safety.traceSafeList_nil,Safety.stepSafe_unfold,same]

/-- Full-state/failure equality with the actual raw body; no floating rule. -/
theorem source_run (M : Algebra α) (h : Call) (s : State α) :
    run M (rawCode h) s = run M (code h) s := by
  rw [raw_decomposition]
  simp only [code,run_append,run_if_congr M guard _ _ (run_active M h)]

/-- Trace-safety equivalence with the same raw body and all its real accesses. -/
theorem source_safe (M : Algebra α) (h : Call) (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList M bounds (rawCode h) s ↔
      Safety.traceSafeList M bounds (code h) s := by
  rw [raw_decomposition]
  simp only [code,Safety.traceSafeList_append,safe_if_congr M guard _ _ bounds (safe_active M h bounds)]

#axiomsClean source_run
#axiomsClean source_safe

theorem original_index_abi (h : Call) : Kernels.source32 h = Kernels.source h := rfl
theorem candidate_index_abi (h : Call) : Kernels.outputFirst32 h = Kernels.outputFirst h := rfl
#axiomsClean raw_decomposition
#axiomsClean active_decomposition
#axiomsClean original_index_abi
#axiomsClean candidate_index_abi
end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.SourceView
