import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Misc.Native

namespace VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated
open VeriTile.Triton FP.Structural FPValue Contract
open FP.Equational (ReductionPlan)
noncomputable section
open Classical

/-- Actual runtime launch coverage may use any permutation. Every resulting
source cell is the same as the canonical enumeration used by byte readback. -/
theorem runtime_schedule (rest : Algebra Word) (plans : ∀ n,ReductionPlan n)
    (t : Inputs) (s : Scalars) (before : Raw.State) (ok : Admitted t s spans (indices t before))
    (order : List Nat) (coverage : order.Perm (List.range (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s).count)) :
    execute rest plans (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) order (entry (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) (indices t before))=
      some (finalMemory rest plans t s before) := by
  have canonical:=complete rest plans t s before ok
  obtain ⟨final,run,read,frame⟩:=whole_source rest plans (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) (indices t before)
    (valid t s spans (indices t before) ok) order coverage
  have equal : final=finalMemory rest plans t s before := by
    funext region off
    by_cases written : ∃ b n,b<t.indices.numel ∧ n<VeriTile.Bench.Repairs.LoraExpandGemv.Contract.width t.w ∧ region="Y" ∧ off=yoff (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) b n
    · obtain ⟨b,n,hb,hn,rfl,rfl⟩:=written
      exact (read b n hb hn).trans (canonical.2.1 b n hb hn).symm
    · have outside : region≠("Y" : RegionName) ∨ ∀ b n,b<t.indices.numel → n<VeriTile.Bench.Repairs.LoraExpandGemv.Contract.width t.w → off≠yoff (VeriTile.Bench.Repairs.LoraExpandGemv.Contract.geometry t s) b n := by
        by_cases same : region="Y"
        · right;intro b n hb hn equal;exact written ⟨b,n,hb,hn,same,equal⟩
        · exact Or.inl same
      exact (frame region off outside).trans (canonical.2.2 region off outside).symm
  exact run.trans (congrArg some equal)

#axiomsClean runtime_schedule
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated
