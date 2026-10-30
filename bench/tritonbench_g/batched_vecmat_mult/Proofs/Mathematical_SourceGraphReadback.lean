import bench.tritonbench_g.batched_vecmat_mult.Proofs.Mathematical_SourceGraphOutputBounds

/-! Readback of every actual output lane from the existing independent
memory reference. The actual grid derives address injectivity; the existing
scatter theorem supplies exhaustive lane enumeration. No execution result,
input/output separation or numerical law is assumed. -/
namespace VeriTile.Bench.TritonBenchG.BatchedVecmatMult.SourceGraph
open VeriTile.Triton FP.Structural
open FP.Equational (ReductionPlan)
noncomputable section
local instance : Inhabited Word := ⟨Expr.zero⟩

/-- For every valid program and output lane, the actual reference scatter
stores exactly its ordered source accumulation graph, including when K = 0. -/
theorem actual_program_expected_readback (plans : PlanSelector) (p : Parameters) (initial : State Word)
    (mem : Memory) (k : Nat) (hk : k < gridCount p) (i : TileIndex [p.BM,p.BN]) :
    expectedMemory plans p (programState p initial mem k) p.output
      (outputOffset p (programState p initial mem k) i) =
      Cell.mk .real (accumulated plans p (programState p initial mem k) (p.K / p.BK) i) := by
  unfold expectedMemory
  rw [← fold_write_memory p (programState p initial mem k)
    (programState p initial mem k)
    (accumulated plans p (programState p initial mem k) (p.K / p.BK))]
  apply State.scatter_readback
  intro left right same
  exact (actual_program_output_unique p initial mem mem k k hk hk left right same).2

#axiomsClean actual_program_expected_readback
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BatchedVecmatMult.SourceGraph
