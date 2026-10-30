import bench.tritonbench_g.chunk_cumsum_kernel.Proofs.Geometry
import bench.tritonbench_g.chunk_cumsum_kernel.Proofs.Mathematical_RealPadding

namespace VeriTile.Bench.Repairs.ChunkCumsumKernel.Tail
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkCumsumKernel
open scoped BigOperators
noncomputable section

/-- Arbitrary unspecified tail loads cannot change a valid mathematical
prefix. The last carry may contain padding, but no later chunk reads it. -/
theorem valid_prefix (input output : RegionName) (dtype outType : FloatDType)
    (T : Nat) (cfg : Config) (s : BlockState) (separate : output≠input) :
    ∃ t, exec (kernel dtype outType input output T cfg.block).eraseDType.toAlgKernel s=some t ∧
      ∀ i,i<T → t.readMem output (s.pids 0*T+i)=
        ∑ j ∈ (Finset.range T).filter (fun j => j ≤ i), s.readMem input (s.pids 0 * T + j) := by
  exact RealPadding.source_outputs Bool.false dtype outType input output T cfg.block s separate cfg.positive

#axiomsClean valid_prefix
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkCumsumKernel.Tail
