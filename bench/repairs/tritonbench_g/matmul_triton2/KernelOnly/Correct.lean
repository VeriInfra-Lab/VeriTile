import bench.repairs.tritonbench_g.matmul_triton2.KernelOnly.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly
open VeriTile.Triton FP.Structural Revision001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Complete grouped grid and original strided FP32 source, in logical coordinates. -/
def realIO (rest : Algebra ℝ) (p : Core.Parameters) := IO.logicalIO rest Core.realPrimitives p
specification kernel_real (rest : Algebra ℝ) (p : Core.Parameters)
    (geometry : Geometry.Valid (Core.geometry p)) (inner : 0<p.blockInner)
    (input : p.input≠p.output) (weight : p.weight≠p.output) :
    Spec.Real (realIO rest p ⊨ fun ⟨a,b⟩ row col =>
      ∑ k : Fin p.innerSize, a row k * b k col) :=
  LogicalIOProof.implements rest p geometry inner input weight

#axiomsClean kernel_real
#specNonCircular LogicalIOProof.target avoiding [Core.grid,Core.source,FP.Structural.exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly
