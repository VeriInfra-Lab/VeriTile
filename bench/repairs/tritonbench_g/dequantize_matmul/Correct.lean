import bench.repairs.tritonbench_g.dequantize_matmul.Proofs.LogicalReader

namespace VeriTile.Bench.Repairs.DequantizeMatmul.Correct
open VeriTile.Triton FP.Structural KernelSource KernelModels
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Signed integer promotion followed by the independent column-scale product. -/
def formula (q : Int) (scale : ℝ) : ℝ := (q : ℝ) * scale

/-- IO.lean declares the INT8 matrix, independent FP32 scale window, complete
FP32 output and every actual launch in the two-dimensional source grid. -/
def dequantizeIO (rest : Algebra ℝ) (p : Parameters) : ProgramIO ℝ ℝ :=
  PublicIO.sourceIO (realAlgebra rest) p Bool.false

/-- Every source access is backed and the whole K-by-N result has the signed
INT8 column-scale formula, with the exact exterior memory frame. -/
specification kernel_correct (rest : Algebra ℝ) (p : Parameters)
    (hK : 0 < p.BK) (hN : 0 < p.BN) (sep : KernelGrid.Separate p) :
    Spec.Real ((PublicIO.logicalIO rest p) ⊨{PublicIO.LogicalSignedBytes}
      (fun xs row col => (xs.weights row col : ℝ) * xs.scales col)) :=
  PublicIO.logical_correct rest p hK hN sep

#specNonCircular formula avoiding [KernelSource.kernel, KernelExecution.exec, KernelGrid.run]
#axiomsClean kernel_correct
#guard_msgs (drop info) in
#auditModuleAxioms

end
end VeriTile.Bench.Repairs.DequantizeMatmul.Correct
