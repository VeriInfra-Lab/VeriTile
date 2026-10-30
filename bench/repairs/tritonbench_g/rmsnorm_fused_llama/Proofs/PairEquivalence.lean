import bench.repairs.tritonbench_g.rmsnorm_fused_llama.PairIO
import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Proofs.NativePublicIO

namespace VeriTile.Bench.Repairs.RmsnormFusedLlama.PairEquivalence
open VeriTile Triton FP.Structural
open scoped VeriTile.Spec
noncomputable section

/-- Reuse the existing complete source/candidate row-grid certificate. The
sole change is the exact natural-number output-pointer product. -/
theorem equivalent (R : Native.Reductions) (X Y W : RegionName) (rows columns : Nat)
    (epsilon : Native.Word) (positive : 0 < columns) (limit : columns ≤ 16384)
    (xy : X ≠ Y) (wy : W ≠ Y) :
    PairIO.original R X Y W rows columns epsilon ≡[[]]
      PairIO.indexCommuted R X Y W rows columns epsilon :=
  ProgramIO.Implementation.of_equivalent _
    (NativePublicIO.equivalent (Native.algebra R) X Y W rows columns 16384
      (.leaf "epsilon" 0 epsilon) positive limit xy wy)

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.RmsnormFusedLlama.PairEquivalence
