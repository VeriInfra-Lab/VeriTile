import bench.tritonbench_g.vector_addition.Kernels
import bench.tritonbench_g.add_example.Proofs.Rewrite

namespace VeriTile.Bench.Repairs.VectorAddition.Contiguous.Rewrite
open VeriTile.Triton FP.Structural FP.Guarded
noncomputable section

export VeriTile.Bench.Repairs.AddExample.Contiguous.Rewrite
  (originalExecution commutedExecution additionSite loadedValue site_values_iff commuted_memory)

/-- Reuse only a proof: the public vector source remains its own readable DSL. -/
theorem source_identity (n B : Nat) :
    add_kernel "x" "y" "out" n B =
      VeriTile.Bench.Repairs.AddExample.Contiguous.add_kernel "x" "y" "out" n B := rfl

theorem candidate_identity (n B : Nat) :
    add_commuted_kernel "x" "y" "out" n B =
      VeriTile.Bench.Repairs.AddExample.Contiguous.add_commuted_kernel "x" "y" "out" n B := rfl

end
end VeriTile.Bench.Repairs.VectorAddition.Contiguous.Rewrite
