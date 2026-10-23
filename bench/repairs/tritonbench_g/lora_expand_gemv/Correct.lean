import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Direct.Real

namespace VeriTile.Bench.Repairs.LoraExpandGemv.Correct
open VeriTile.Triton FP.Structural FPValue Contract
open FP.Equational (ReductionPlan)
noncomputable section

/-- The executable direct-buffer checks imply complete source execution and
an ordinary K-term contraction, for every legal runtime grid schedule. -/
theorem checked_real (rest : Algebra Word) (plans : ∀ n,ReductionPlan n)
    (t : Inputs) (s : Scalars) (indices : Nat→Int) (ok : Admitted t s indices)
    (data : RegionName→Nat→ℝ) (order : List Nat) (coverage : order.Perm (List.range (geometry t s).count)) :
    ∃ final,execute rest plans (geometry t s) order (entry (geometry t s) indices)=some final ∧
      (∀ b n,b<t.indices.numel → n<width t.w →
        Real.eval data ((final "Y" (yoff (geometry t s) b n)).read .real)=
          Real.target data (geometry t s) indices b n) ∧
      (∀ r off,(r≠("Y" : RegionName) ∨ ∀ b n,b<t.indices.numel → n<width t.w → off≠yoff (geometry t s) b n) →
        final r off=entry (geometry t s) indices r off) :=
  Direct.RealCall.complete rest plans t s indices ok data order coverage

#axiomsClean checked_real
#specNonCircular Real.target avoiding [execute,Branches.source,FP.Structural.exec,value]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LoraExpandGemv.Correct
