import bench.repairs.tritonbench_g.rotary_transform_tile_checked.IO
import bench.repairs.tritonbench_g.rotary_transform_tile_checked.Proofs.IOResults

namespace VeriTile.Bench.Repairs.RotaryTransformTileChecked.PublicIO
open VeriTile.Bench.Repairs.RotaryTransform
open VeriTile.Triton FP.Structural
open scoped VeriTile.Triton.LogicalDomainIO
noncomputable section

/-- Complete old/new readback judgment equivalence. Both maps are bijections;
all metadata and prepared-suffix conditions, actual whole-grid schedules,
result reads, source safety and prefix-only writable frames remain unchanged. -/
theorem logical_real_iff (c : Config) (cu64 off64 : Bool) :
    (logicalIO c cu64 off64 ⊨{LogicalReady c} fun xs batch row head col =>
      realFormula c (logicalMemory c xs) batch.val row.val head.val col.val) ↔
    (io realOps id c cu64 off64).ImplementsOn (domain c) (expected c (realFormula c)) := by
  have target_eq : (logicalIO c cu64 off64).lift (fun xs batch row head col =>
      realFormula c (logicalMemory c xs) batch.val row.val head.val col.val) =
      (fun raw => expected c (realFormula c) ((logicalInputs c).symm (logicalInputs c raw))) := by
    funext raw port lane
    fin_cases port
    rfl
  change (io realOps id c cu64 off64).ImplementsOn
    (fun raw => domain c ((logicalInputs c).symm (logicalInputs c raw)))
    _ ↔ _
  rw [target_eq]
  simp only [Equiv.symm_apply_apply]

#axiomsClean logical_real_iff
end
end VeriTile.Bench.Repairs.RotaryTransformTileChecked.PublicIO
