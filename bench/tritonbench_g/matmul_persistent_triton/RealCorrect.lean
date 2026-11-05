import bench.tritonbench_g.matmul_persistent_triton.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulPersistentTriton Floating
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Same complete persistent launch and selected FP16/E4M3 storage, viewed as
ordinary logical matrices. Empty matrices retain the original empty grid. -/
abbrev matmulIO (p : Core.Parameters) := PublicData.logicalIO Real.arithmetic p
/-- Every logical output is the ordinary full-K matrix product. Original
layout, safety, successful whole-grid execution and exact exterior frame remain. -/
specification kernel_correct (p : Core.Parameters) (b : RawKernel.Buffers)
    (layout : RawKernel.Layout p b) (sms : 0<p.problem.SMS) :
    Spec.Real (matmulIO p ⊨ fun ⟨a,b⟩ row col =>
      ∑ k : Fin p.problem.K, a row k * b k col) :=
  LogicalIOProof.implements p b layout sms

#stmtSurfaceSubset kernel_correct ⊆
  [kernel_correct.match_1,Spec.Real,matmulIO,LogicalProgramIO.Implements,Core.Parameters,Core.Parameters.problem,Problem.K,Problem.M,Problem.N,Problem.SMS,RawKernel.Buffers,RawKernel.Layout]
#axiomsClean kernel_correct
#specNonCircular PublicData.mathematical avoiding [Core.executeGrid,Core.program,Source.run]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast
