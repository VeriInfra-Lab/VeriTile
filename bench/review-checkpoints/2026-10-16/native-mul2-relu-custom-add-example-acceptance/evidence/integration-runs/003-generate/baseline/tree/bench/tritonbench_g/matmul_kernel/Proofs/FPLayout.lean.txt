import bench.tritonbench_g.matmul_kernel.Proofs.FPComplete
import VeriTile.Triton.KernelLemmas.OffsetInjective
namespace VeriTile.Bench.TritonBenchG.MatmulKernel.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
/-- Exactly the old BN≤4096 layout premise; there is no new input restriction. -/
theorem original_layout_injective {α : Type} (p : Parameters) (s : State α)
    (hbn : BN p≤4096) : Function.Injective (cOffset p s) := by
  have heq : cOffset p s = fun i : CIndex p =>
      (4096*(pidM p s*BM p)+pidN p s*BN p)+i.1.val*4096+i.2.1.val := by
    funext i
    simp only [cOffset,row,col]
    ring
  rw [heq]
  exact rowMajor2D_inj _ 4096 hbn
#axiomsClean original_layout_injective
end
end VeriTile.Bench.TritonBenchG.MatmulKernel.FPValue
