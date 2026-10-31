import bench.tritonbench_g.matrix_vector_multip.Proofs.KernelSource
import bench.tritonbench_g.matrix_vector_multip.Proofs.SourceSafety
namespace VeriTile.Bench.Repairs.MatrixVectorMultip.AccessSafety
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
 theorem source_safe {α : Type} [Inhabited α] (A : Algebra α) (N M BN BM : Nat)
    (width : 0<BM) (scope : SourceBounds N M BN BM)
    (castId : ∀ precision src dst value, A.cast precision src dst value=value)
    (loadId : ∀ value, A.fp32Load value=value) (bounds : RegionName → Nat) (s : State α)
    (pidSmall : s.pids 0<2^64)
    (matrix : ∀ i j, i<N → j<M → i*M+j<bounds "A")
    (vector : ∀ j,j<M → j<bounds "B") (output : ∀ i : Fin BN, row BN (s.pids 0) i.val<N →
      row BN (s.pids 0) i.val<bounds "C") :
    TraceSafe A bounds (Kernels.source N M BN BM) s := by
  rw [source_binding]
  apply Source.source_safe A (parameters N M BN BM) width castId loadId bounds s
    (scalar_bounds N M BN BM scope) pidSmall scope.lanes_rows scope.lanes_columns
  · simpa [parameters] using matrix
  · simpa [parameters] using vector
  · simpa [parameters] using output
#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.MatrixVectorMultip.AccessSafety
