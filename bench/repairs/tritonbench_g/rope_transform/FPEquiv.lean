import bench.repairs.tritonbench_g.rope_transform.Proofs.ProgramPair
import bench.repairs.tritonbench_g.rope_transform.Proofs.KernelMath

namespace VeriTile.Bench.Repairs.RopeTransform.FPEquiv
open VeriTile Triton FP.Structural KernelContract
open scoped VeriTile.Spec
noncomputable section

/-- Actual full-grid FP32 programs for index common-subexpression elimination.
The selected forward FMA backend, original casts and arbitrary Tagged payloads
are retained. No floating rewrite law or finite-value restriction is required. -/
specification source_candidate_equivalence (cfg : FP.Config) (r : Request) :
    PairIO.original (nativeAlgebra cfg) (id : Tagged → Tagged) id r ≡[[]]
    PairIO.indexShared (nativeAlgebra cfg) (id : Tagged → Tagged) id r := by
  exact ProgramIO.Implementation.of_equivalent (PairIO.candidate _ _ _ r)
    (PairIO.equivalent _ _ _ r)

#axiomsClean source_candidate_equivalence
end
end VeriTile.Bench.Repairs.RopeTransform.FPEquiv
