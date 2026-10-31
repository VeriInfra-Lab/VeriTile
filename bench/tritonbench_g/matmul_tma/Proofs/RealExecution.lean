import bench.tritonbench_g.matmul_tma.Kernels
import bench.tritonbench_g.matmul_tma.Proofs.Real

namespace VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2.RealExecution
open VeriTile.Triton FP.Structural FPValue Host
noncomputable section

/-- Independent matrix product of the raw FP16 source-memory observations. -/
def matrixProduct (h : Shape) (f : Flags) (s : State ℝ) (i : CIndex (params h f)) : ℝ :=
  ∑ k : Fin h.inner, aValues (params h f) s (i.1,k,PUnit.unit) *
    bValues (params h f) s (k,i.2.1,PUnit.unit)

#specNonCircular matrixProduct avoiding [VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2.kernelDirectOutput,source,FP.Structural.exec,scatter]
end
end VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2.RealExecution
