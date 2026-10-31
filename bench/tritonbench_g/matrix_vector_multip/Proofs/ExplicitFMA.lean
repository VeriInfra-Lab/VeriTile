import bench.tritonbench_g.matrix_vector_multip.CandidateIO
import bench.tritonbench_g.matrix_vector_multip.Proofs.SourceCandidateEqual
import bench.tritonbench_g.matrix_vector_multip.Proofs.SourceCandidateSafety
namespace VeriTile.Bench.Repairs.MatrixVectorMultip.ExplicitFMA
open VeriTile.Triton FP.Structural KernelSource
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip WholeNative
noncomputable section

theorem execution_equal (native : Scalars) (g : Graph BN BM) (N M : Nat)
    (width : 0<BM) (scope : SourceBounds N M BN BM) (s : State Payload) (pidSmall : s.pids 0<2^64) :
    exec (algebra native g) (Kernels.explicitFMA N M BN BM) s=
      exec (WholeNative.algebra native g) (Kernels.source N M BN BM) s :=
  Source.Candidate.execution_equal native (parameters N M BN BM) g width s
    (scalar_bounds N M BN BM scope) pidSmall scope.lanes_rows scope.lanes_columns

theorem source_safe (native : Scalars) (g : Graph BN BM) (N M : Nat)
    (width : 0<BM) (scope : SourceBounds N M BN BM) (bounds : RegionName → Nat) (s : State Payload)
    (pidSmall : s.pids 0<2^64)
    (matrix : ∀ i j,i<N → j<M → i*M+j<bounds "A")
    (vector : ∀ j,j<M → j<bounds "B") (output : ∀ i : Fin BN, row BN (s.pids 0) i.val<N →
      row BN (s.pids 0) i.val<bounds "C") :
    TraceSafe (algebra native g) bounds (Kernels.explicitFMA N M BN BM) s := by
  apply Source.Candidate.source_safe native (parameters N M BN BM) g width bounds s
    (scalar_bounds N M BN BM scope) pidSmall scope.lanes_rows scope.lanes_columns
  · simpa [parameters] using matrix
  · simpa [parameters] using vector
  · simpa [parameters] using output
#axiomsClean execution_equal
#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.MatrixVectorMultip.ExplicitFMA
