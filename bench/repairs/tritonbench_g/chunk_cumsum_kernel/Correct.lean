import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Proofs.LogicalReader

namespace VeriTile.Bench.Repairs.ChunkCumsumKernel.Correct
open VeriTile.Triton
open scoped BigOperators VeriTile.Triton.LogicalIO
noncomputable section

/-- Inclusive prefix of the complete time row. The source retains arbitrary
row/program windows, final-chunk masks, source safety and exterior frame. -/
specification memory_correct (input output : RegionName) (T : Nat) (cfg : Config)
    (separate : output ≠ input) :
    Spec.Real (PublicReal.prefixLogicalIO input output T cfg ⊨ fun xs time =>
      ∑ k : Fin T, if k ≤ time then xs k else 0) :=
  PublicReal.logical_implements input output T cfg separate

#axiomsClean memory_correct
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkCumsumKernel.Correct
