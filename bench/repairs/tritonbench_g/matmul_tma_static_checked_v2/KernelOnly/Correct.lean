import bench.repairs.tritonbench_g.matmul_tma_static_checked_v2.KernelOnly.Proofs.LogicalIO

namespace VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2
open VeriTile.Triton FP.Structural FPValue Host
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- One complete static source invocation, both transpose flags, FP16 inputs
and selected FP16/FP32 output, with logical matrix coordinates. -/
def matmulIO (rest : Algebra ℝ) (h : Shape) (f : Flags) :=
  PublicIO.logicalIO (StaticChecked.RealModel.algebra rest) StaticChecked.RealModel.dot (params h f) f.half
specification kernel_math (rest : Algebra ℝ) (h : Shape) (f : Flags) :
    VeriTile.Spec.Real (matmulIO rest h f ⊨ fun ⟨a,b⟩ row col =>
      ∑ k : Fin h.inner, a row k * b k col) :=
  LogicalIOProof.implements rest h f

#axiomsClean kernel_math
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2
