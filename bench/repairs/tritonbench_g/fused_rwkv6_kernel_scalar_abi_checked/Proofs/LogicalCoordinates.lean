import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.IO
import bench.repairs.tritonbench_g.fused_rwkv6_kernel_scalar_abi_checked.Proofs.IOResults

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Interface.Result
open VeriTile Triton FP.Structural FusedRwkv6Kernel
noncomputable section

theorem logical_iff (c : Shape) (scale : ℝ) :
    (logicalIO c scale).Implements (fun xs => {
      partials := fun ik z t f => Real.partialFormula c (tensorData c xs) scale
        z.val ik.val (visit c t.val) f.val
      finalState := fun z e f => realState c (tensorData c xs) z.val e.val f.val c.T }) ↔
    (programIO Real.algebra c scale).Implements (Reference.real c scale) := by
  have eq : (logicalIO c scale).lift (fun xs => {
      partials := fun ik z t f => Real.partialFormula c (tensorData c xs) scale
        z.val ik.val (visit c t.val) f.val
      finalState := fun z e f => realState c (tensorData c xs) z.val e.val f.val c.T }) =
      Reference.real c scale := by
    funext xs o j
    change Fin 2 at o
    fin_cases o
    · change Real.partialFormula c
        (logicalInputs c ((inputCoordinates ℝ c).symm ((inputCoordinates ℝ c) xs))) scale
        j.2.1.val j.1.val (visit c j.2.2.1.val) j.2.2.2.1.val = _
      rw [(inputCoordinates ℝ c).symm_apply_apply]
      rfl
    · change realState c
        (logicalInputs c ((inputCoordinates ℝ c).symm ((inputCoordinates ℝ c) xs)))
        j.1.val j.2.1.val j.2.2.1.val c.T = _
      rw [(inputCoordinates ℝ c).symm_apply_apply]
      rfl
  change (programIO Real.algebra c scale).Implements _ ↔ _
  rw [eq]

theorem logical_real (c : Shape) (hc : Sizes c) (scale : ℝ) :
    (logicalIO c scale).Implements (fun xs => {
      partials := fun ik z t f => Real.partialFormula c (tensorData c xs) scale
        z.val ik.val (visit c t.val) f.val
      finalState := fun z e f => realState c (tensorData c xs) z.val e.val f.val c.T }) :=
  (logical_iff c scale).mpr (real c hc scale)
end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Interface.Result
