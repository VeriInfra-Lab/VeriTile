import bench.tritonbench_g.rope_embedding.IO
import bench.tritonbench_g.fast_rope_embedding.Proofs.Mathematical_FP

namespace VeriTile.Bench.Repairs.RopeEmbedding.PairProof
open VeriTile.Triton PairIO
open scoped VeriTile.Spec

/-- Reuse the complete typed source proof through the existing exact source
binding. The sole change is Nat multiplication; no FP rewrite is introduced. -/
theorem equivalent (p : Source.Parameters) (back : Bool) :
    original p back ≡[assumptions] indexCommutedKernel p back := by
  cases back <;> unfold original indexCommutedKernel assumptions <;> split_equiv <;> solve_fp

#axiomsClean equivalent
end VeriTile.Bench.Repairs.RopeEmbedding.PairProof
