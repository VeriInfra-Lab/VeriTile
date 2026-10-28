import bench.repairs.tritonbench_g.rotary_transform_tile_checked.Proofs.SourceBlocks
import VeriTile.Triton.Float.StructuralSafety
namespace VeriTile.Bench.Repairs.RotaryTransformTileChecked.SourceBlocks
open VeriTile.Triton FP.Structural
set_option maxHeartbeats 1000000
set_option maxRecDepth 24000

theorem safe_constant_branch {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (flag : Bool)
    (prelude yes no : List ComputeStmt) (s : State α) :
    Safety.traceSafeList M bounds
      (prelude ++ [.ifThenElse (.boolNot (.constBool flag)) yes no]) s ↔
      Safety.traceSafeList M bounds (prelude ++ if flag then no else yes) s := by
  rw [Safety.traceSafeList_append,Safety.traceSafeList_append]
  apply and_congr Iff.rfl
  apply forall_congr'
  intro t
  apply imp_congr_right
  intro _
  cases flag <;> simp [Safety.traceSafeList_cons,Safety.stepSafe,Safety.safeOp,evalOp_unfold]

#axiomsClean safe_constant_branch
end VeriTile.Bench.Repairs.RotaryTransformTileChecked.SourceBlocks
