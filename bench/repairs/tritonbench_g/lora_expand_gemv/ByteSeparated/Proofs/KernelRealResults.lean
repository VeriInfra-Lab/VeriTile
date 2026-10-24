import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Kernels
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Grid
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Real

namespace VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.KernelRealResults
open VeriTile.Triton FP.Structural FPValue
open FP.Equational (ReductionPlan)
noncomputable section

/-- Complete shared source on raw strided regions, with the independent dense
Real GEMV target and the full exterior regional frame. -/
theorem kernel_real (rest : Algebra Word) (plans : ∀ n,ReductionPlan n)
    (g : Geometry) (indices : Nat → Int) (bounds : Valid g indices)
    (data : RegionName → Nat → ℝ) (order : List Nat)
    (coverage : order.Perm (List.range g.count)) :
    ∃ final, execute rest plans g order (entry g indices)=some final ∧
      (∀ b n,b<g.B → n<g.N →
        Real.eval data ((final "Y" (yoff g b n)).read .real)=
          Real.target data g indices b n) ∧
      (∀ r off,(r≠("Y" : RegionName) ∨ ∀ b n,b<g.B → n<g.N → off≠yoff g b n) →
        final r off=entry g indices r off) := by
  obtain ⟨final,run,read,frame⟩ := whole_source rest plans g indices bounds order coverage
  refine ⟨final,run,?_,frame⟩
  intro b n hb hn
  rw [read b n hb hn]
  exact Real.value_eq_target plans g indices bounds data b n

#axiomsClean kernel_real
#specNonCircular Real.target avoiding [execute,Branches.source,FP.Structural.exec,value]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.KernelRealResults
