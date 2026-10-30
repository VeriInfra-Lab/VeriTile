import bench.repairs.tritonbench_g.rmsnorm_triton.Proofs.IOCorrect

namespace VeriTile.Bench.Repairs.RmsnormTriton.IOCorrect
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.RmsnormTriton.Whole
open KernelGrid RmsIO
open scoped VeriTile.Triton.LogicalScheduledKernelIO
noncomputable section

/-- Equality of full judgments; the complete input/output tensors are reindexed. -/
theorem logical_iff (g : Geometry) (work : Schedule g.parameters) :
    (logicalIO g work ⊨ fun values cell =>
      let xs := values.1
      let weights := values.2
      let batch := cell.1.1
      let row := cell.1.2
      let k := cell.2
      (xs batch row k * (Real.sqrt
        ((∑ j : Fin g.width, xs batch row j*xs batch row j)/(g.width:ℝ)+g.eps))⁻¹) * weights k) ↔
    (io g).RegionalScheduledImplements (schedule g work) (target g) := by
  change (io g).RegionalScheduledImplements (schedule g work)
    (fun vals o k => rms g.eps
      (vals (finProdFinEquiv (finProdFinEquiv.symm o)).castSucc)
      (vals (Fin.last (rowCount g))) k) ↔ _
  apply Iff.of_eq
  apply congrArg ((io g).RegionalScheduledImplements (schedule g work))
  funext vals o k
  dsimp only [target]
  rw [Equiv.apply_symm_apply]
  rfl

end
end VeriTile.Bench.Repairs.RmsnormTriton.IOCorrect
