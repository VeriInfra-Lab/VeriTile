import bench.repairs.tritonbench_g.kcache_copy_triton.Proofs.KernelFP
import VeriTile.Triton.Float.ProfiledMemory
import VeriTile.Meta.Specification
namespace VeriTile.Bench.Repairs.KcacheCopyTriton.KernelEquivalence
open VeriTile.Triton FP.ProfiledMemory
open scoped VeriTile.Spec
noncomputable section

/-- Both sources perform the same snapshot copy and preserve every cell
outside their selected cache slice, including all metadata/source regions. -/
theorem kernel_results {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (p : Source.Parameters) (s : FP.Structural.State α) :
    ∃ lhs rhs,
      FP.Structural.exec M (Source.originalKernel p) s=some lhs ∧
      FP.Structural.exec M (Source.commutedKernel p) s=some rhs ∧
      lhs.mem=(KernelFP.copied M p s).mem ∧ rhs.mem=lhs.mem ∧
      (∀ r o,r≠p.KCache ∨ (∀ i : TileIndex [p.width],o≠KernelFP.targetOffset p s i) →
        lhs.mem r o=s.mem r o) := by
  exact ⟨(KernelFP.execution M Bool.false p s).val,(KernelFP.execution M Bool.true p s).val,
    (KernelFP.execution M Bool.false p s).property,(KernelFP.execution M Bool.true p s).property,
    KernelFP.execution_memory M Bool.false p s,
    (KernelFP.execution_memory M Bool.true p s).trans (KernelFP.execution_memory M Bool.false p s).symm,
    KernelFP.frame M Bool.false p s⟩

def assumptions : Spec.Assumptions FP.GuardedFragment := []
def original (p : Source.Parameters) : Program := ⟨Source.originalKernel p,[]⟩
def commuted (p : Source.Parameters) : Program := ⟨Source.commutedKernel p,[]⟩

/-- Actual address-commuted candidate, successful on both sides and equal on
the complete typed memory. Every FP32 input word is admitted; no numerical law
is needed for exact signed integer addition commutation. -/
theorem source_equivalence (p : Source.Parameters) :
    original p ≡[assumptions] commuted p := by
  apply Spec.FloatingPoint.ofNumerical
  · rfl
  · rfl
  intro α _ M D _ s _
  let lhs := KernelFP.execution (FP.ProfiledRewrite.engine M) Bool.false p s
  let rhs := KernelFP.execution (FP.ProfiledRewrite.engine M) Bool.true p s
  exact ⟨lhs.val,rhs.val,lhs.property,rhs.property,
    (KernelFP.execution_memory _ _ _ _).trans (KernelFP.execution_memory _ _ _ _).symm⟩




#print_fp_assumptions source_equivalence
#axiomsClean source_equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KcacheCopyTriton.KernelEquivalence
