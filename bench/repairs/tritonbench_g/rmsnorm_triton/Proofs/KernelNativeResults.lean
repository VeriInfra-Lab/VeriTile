import bench.repairs.tritonbench_g.rmsnorm_triton.Proofs.KernelGrid

namespace VeriTile.Bench.Repairs.RmsnormTriton.KernelNativeResults
open VeriTile
open VeriTile.Bench.TritonBenchG.RmsnormTriton.Whole
noncomputable section

theorem kernel_native_equivalence (Q : Numerics Torch.FP32.Word) (profile : Symbolic.Profile)
    (g : KernelGrid.Geometry) (sep : Separation g.parameters) (unique : OutputUnique g.parameters)
    (u : Nat → Torch.FP32.Word) (schedule : Schedule g.parameters) (before : Memory Torch.FP32.Word) :
    let p := g.parameters
    ∃ symbolicFinal final,
      Symbolic.grid p (fun k=>.input (u k)) schedule.order (Symbolic.entryGraph before)=some symbolicFinal ∧
      Symbolic.candidateGrid p (fun k=>.input (u k)) schedule.order (Symbolic.entryGraph before)=some symbolicFinal ∧
      Symbolic.observedGrid Q profile p u schedule.order before=some final ∧
      Symbolic.observedCandidateGrid Q profile p u schedule.order before=some final ∧
      final=Symbolic.projectMemory Q profile symbolicFinal ∧
      (∀ b m k,b<p.batches → m<p.rows → k<p.width →
        final p.output.allocation (outputOffset p b m k)=KernelGrid.nativeTarget Q profile p before b m k) ∧
      final=referenceMemory p (KernelGrid.nativeTarget Q profile p before) before ∧
      (∀ a o,Outside p a o → final a o=before a o) :=
  KernelGrid.native Q profile g.parameters sep unique u schedule before

#axiomsClean kernel_native_equivalence
end
end VeriTile.Bench.Repairs.RmsnormTriton.KernelNativeResults
