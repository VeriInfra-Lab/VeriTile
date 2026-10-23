import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Proofs.Tail

namespace VeriTile.Bench.Repairs.ChunkCumsumKernel.RealRow
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkCumsumKernel
open scoped BigOperators
noncomputable section

open Lean Elab Term in
elab "legacyCompleteRow" : term =>
  return mkConst ((Name.num `_private.bench.tritonbench_g.chunk_cumsum_kernel.Proofs.LegacyReal 0) ++
    `VeriTile.Bench.TritonBenchG.ChunkCumsumKernel.cc_runExact)

/-- Reuse the unchanged legacy complete row theorem, including its tagged
cell frame and arbitrary unspecified tail. The legacy theorem is private;
its exact compiler name is pinned rather than copying its proof. -/
theorem complete (input output : RegionName) (dtype outType : FloatDType)
    (T : Nat) (cfg : Config) (s : BlockState) (separate : output≠input) :
    ∃ t, exec (kernel dtype outType input output T cfg.block).eraseDType.toAlgKernel s=some t ∧
      (∀ i, i<T → t.readMem output (s.pids 0*T+i)=
        ∑ j ∈ (Finset.range T).filter (fun j => j ≤ i), s.readMem input (s.pids 0*T+j)) ∧
      (∀ rg off, (rg≠output ∨ ∀ i, i<T → off≠s.pids 0*T+i) → t.mem rg off=s.mem rg off) := by
  obtain ⟨t,he,hv,hf⟩ :=
    legacyCompleteRow
      input output T cfg.block s separate cfg.positive
  refine ⟨t,?_,hv,hf⟩
  change exec (RealSource.source Bool.false dtype outType input output T cfg.block).eraseDType.toAlgKernel s=some t
  simpa only [exec,RealSource.original_body] using he

#axiomsClean complete
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkCumsumKernel.RealRow
