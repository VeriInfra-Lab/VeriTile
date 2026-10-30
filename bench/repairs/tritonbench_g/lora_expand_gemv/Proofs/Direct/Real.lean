import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Direct.Checked
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Grid
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Real

namespace VeriTile.Bench.Repairs.LoraExpandGemv.Direct.RealCall
open VeriTile.Triton FP.Structural FPValue Contract
open FP.Equational (ReductionPlan)
noncomputable section

theorem complete (rest : Algebra Word) (plans : ∀ n,ReductionPlan n)
    (t : Inputs) (s : Scalars) (indices : Nat→Int) (ok : Admitted t s indices)
    (data : RegionName→Nat→ℝ) (order : List Nat) (coverage : order.Perm (List.range (geometry t s).count)) :
    ∃ final,execute rest plans (geometry t s) order (entry (geometry t s) indices)=some final ∧
      (∀ b n,b<t.indices.numel → n<width t.w →
        Real.eval data ((final "Y" (yoff (geometry t s) b n)).read .real)=
          Real.target data (geometry t s) indices b n) ∧
      (∀ r off,(r≠("Y" : RegionName) ∨ ∀ b n,b<t.indices.numel → n<width t.w → off≠yoff (geometry t s) b n) →
        final r off=entry (geometry t s) indices r off) := by
  obtain ⟨final,run,read,frame⟩:=whole_source rest plans (geometry t s) indices (valid t s indices ok)
    order coverage
  refine ⟨final,run,?_,frame⟩
  intro b n hb hn
  rw [read b n hb hn]
  exact Real.value_eq_target plans (geometry t s) indices (valid t s indices ok) data b n

#axiomsClean complete
#specNonCircular Real.target avoiding [execute,Branches.source,FP.Structural.exec,value]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LoraExpandGemv.Direct.RealCall
