import bench.repairs.tritonbench_g.rmsnorm_triton.Proofs.KernelGrid

namespace VeriTile.Bench.Repairs.RmsnormTriton.KernelRealResults
open VeriTile
open VeriTile.Bench.TritonBenchG.RmsnormTriton.Whole
noncomputable section

theorem kernel_correct (red : Reduction) (g : KernelGrid.Geometry) (sep : Separation g.parameters) (unique : OutputUnique g.parameters)
    (u : Nat → ℝ) (schedule : Schedule g.parameters) (before : Memory ℝ) :
    let p := g.parameters
    ∃ final, TensorSource.grid p schedule.order before=some final ∧
      TensorSource.candidateGrid p schedule.order before=some final ∧
      kernelRun realNumerics red p u schedule before=some final ∧
      (∀ b m k,b<p.batches → m<p.rows → k<p.width →
        final p.output.allocation (outputOffset p b m k)=KernelGrid.realTarget p before b m k) ∧
      final=referenceMemory p (KernelGrid.realTarget p before) before ∧
      (∀ a o,Outside p a o → final a o=before a o) :=
  KernelGrid.real red g.parameters sep unique u schedule before

#axiomsClean kernel_correct
end
end VeriTile.Bench.Repairs.RmsnormTriton.KernelRealResults
