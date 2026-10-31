import bench.tritonbench_g.matrix_vector_multip.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.Correct
open VeriTile VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Compatibility name for the independent full row dot product. -/
def matrixVector (M : Nat) (matrix : Nat → Nat → ℝ) (vector : Nat → ℝ) (i : Nat) : ℝ :=
  ∑ j : Fin M, matrix i j.val * vector j.val

/-- Both complete inputs and the original single invocation's output tile.
The row mask remains exactly `pid * BN + row < N`, including the final tail. -/
abbrev matrixVectorIO (N M BN BM pid : Nat) := IO.logicalIO WholeMath.algebra N M BN BM pid
specification mv_kernel_correctness (N M BN BM pid : Nat) (width : 0 < BM) (scope : SourceBounds N M BN BM) (launch : pid<(N+BN-1)/BN) :
    Spec.Real (matrixVectorIO N M BN BM pid ⊨ fun ⟨matrix,vector⟩ row =>
      ∑ col : Fin M, IO.logicalRow matrix (pid*BN+row.val) col * vector col) :=
  LogicalIOProof.implements N M BN BM pid width scope (scope.grid_pid pid launch)

#specNonCircular matrixVector avoiding [Kernels.source]
#stmtSurfaceSubset mv_kernel_correctness ⊆
  [mv_kernel_correctness.match_1,Spec.Real,matrixVectorIO,LogicalProgramIO.Implements,IO.logicalRow,SourceBounds]
#axiomsClean mv_kernel_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatrixVectorMultip.Correct
