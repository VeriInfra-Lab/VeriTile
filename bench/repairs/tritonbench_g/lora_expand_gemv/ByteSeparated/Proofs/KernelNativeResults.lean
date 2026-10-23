import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Kernels
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Grid
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Native

namespace VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.KernelNativeResults
open VeriTile.Triton FP.Structural FPValue
open FP.Equational (ReductionPlan)
noncomputable section

/-- Independent product/reduction/add graph on raw input words, using exactly
the existing native scalar operations, fusion schedule and reduction plans. -/
def kernelReference (ops : Native.ScalarOps) (schedule : Native.Schedule)
    (plans : ∀ n,ReductionPlan n) (g : Geometry) (indices : Nat → Int)
    (bits : RegionName → Nat → Native.Bits) (b n : Nat) : Native.Bits :=
  Native.eval ops schedule bits ((resultCell plans g indices b n).read .real)

/-- Complete source execution, native output observations and every untouched
regional cell. Geometry, input words and bounds contain no Tensor API data. -/
theorem kernel_native (rest : Algebra Word) (ops : Native.ScalarOps)
    (schedule : Native.Schedule) (plans : ∀ n,ReductionPlan n)
    (g : Geometry) (indices : Nat → Int) (bounds : Valid g indices)
    (bits : RegionName → Nat → Native.Bits) (order : List Nat)
    (coverage : order.Perm (List.range g.count)) :
    ∃ final, execute rest plans g order (entry g indices)=some final ∧
      (∀ b n,b<g.B → n<g.N →
        Native.eval ops schedule bits ((final "Y" (yoff g b n)).read .real)=
          kernelReference ops schedule plans g indices bits b n) ∧
      (∀ r off,(r≠("Y" : RegionName) ∨ ∀ b n,b<g.B → n<g.N → off≠yoff g b n) →
        final r off=entry g indices r off) := by
  obtain ⟨final,run,read,frame⟩ := whole_source rest plans g indices bounds order coverage
  refine ⟨final,run,?_,frame⟩
  intro b n hb hn
  rw [read b n hb hn]
  rfl

#axiomsClean kernel_native
#specNonCircular kernelReference avoiding [execute,Branches.source,FP.Structural.exec]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.KernelNativeResults
