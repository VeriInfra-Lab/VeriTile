import bench.tritonbench_g.matmul_leakyrelu_fp8.Proofs.FPComplete
import VeriTile.Triton.KernelLemmas.OffsetInjective
namespace VeriTile.Bench.TritonBenchG.MatmulLeakyreluFp8.FPValue
open VeriTile.Triton FP.Structural
noncomputable section

/-- The old SCN=1 and BN≤SCM premises supply the new typed output injection
without an additional memory restriction. -/
theorem original_layout_injective {α : Type} (p : Parameters) (s : State α)
    (hcn : p.stride_cn=1) (hbn : BN p≤p.stride_cm) :
    Function.Injective (cOffset p s) := by
  have heq : cOffset p s = fun i : CIndex p =>
      (p.stride_cm*(pidM p s*BM p)+pidN p s*BN p)+i.1.val*p.stride_cm+i.2.1.val := by
    funext i
    simp only [cOffset,row,col,hcn]
    ring
  rw [heq]
  exact rowMajor2D_inj _ p.stride_cm hbn

#axiomsClean original_layout_injective
end
end VeriTile.Bench.TritonBenchG.MatmulLeakyreluFp8.FPValue
