import bench.tritonbench_g.int4_matmul.Proofs.FPDotAccPlan
namespace VeriTile.Bench.TritonBenchG.Int4Matmul
open VeriTile.Triton FP.Structural FPTyped
open FPSoftware (Word)
noncomputable section

/-- Full accepted int4 loop, for a fully specified carry-seeded software FMA
order. This is the explicitly parsed compiled-graph profile, not the original
separate-dot source graph or a tensor-core numerical equivalence claim. -/
specification int4_matmul_dotacc_plan_correctness
    (plan : FPDotAcc.FMAPlan) (p : Parameters) (s : State Word)
    (hTrip : p.inner = p.blockInner * p.steps) (hSplit : s.pids 1 = 0)
    (hInputs : PackedInputs p s)
    (hw : ∀ i j, active p s i → active p s j →
      outputOffset p s i = outputOffset p s j → i = j) :
    ∃ final,
      FP.Structural.exec
        (FPDotAcc.backend (FPSoftware.scalar FPClosed.unused) (FPDotAcc.plannedDotAcc plan))
        (FPDotAcc.source p) s = some final ∧
      final.mem = (FPDotAcc.plannedReference plan p s).mem ∧
      FPDotAcc.Post p s final (FPDotAcc.plannedOutput plan p s) ∧
      ∀ step, step < p.steps → ∀ i : BIndex p, ∃ word zero : BitVec 32,
        s.mem p.weight (bOffset p s step i) = .mk .int word.toInt ∧
        s.mem p.zero (zeroOffset p s step i) = .mk .int zero.toInt ∧
        difference p s step i = packedDifference p s i word zero := by
  obtain ⟨final,he,hp⟩ := FPDotAcc.planned_readback_correctness plan FPClosed.unused p s hw
  obtain ⟨other,ho,hm⟩ := FPDotAcc.planned_memory_correctness plan FPClosed.unused p s
  have heq : final = other := Option.some.inj (he.symm.trans ho)
  subst other
  exact ⟨final,he,hm,hp,packed_difference_bridge p s hInputs⟩

#axiomsClean int4_matmul_dotacc_plan_correctness
end
end VeriTile.Bench.TritonBenchG.Int4Matmul
