import bench.repairs.tritonbench_g.kv_cache_copy.Proofs.KernelFP
import VeriTile.Triton.Float.ProfiledMemory
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.KvCacheCopy.KernelNativeResults
open VeriTile.Triton FP.ProfiledMemory
open scoped VeriTile.Spec
noncomputable section

def assumptions : Spec.Assumptions FP.GuardedFragment := []
def original (p : Source.Parameters) : Program := ⟨Source.originalKernel p,[]⟩
def commuted (p : Source.Parameters) : Program := ⟨Source.commutedKernel p,[]⟩

/-- Both complete static-loop kernels succeed and have identical full typed
memory. Copy payloads are unrestricted; no numerical transformation law. -/
theorem source_equivalence (p : Source.Parameters) :
    original p ≡[assumptions] commuted p := by
  apply Spec.FloatingPoint.ofNumerical
  · rfl
  · rfl
  intro α _ M D _ s _
  obtain ⟨lhs,hl,hm⟩ := KernelFP.execute (FP.ProfiledRewrite.engine M) Bool.false p s
  obtain ⟨rhs,hr,hn⟩ := KernelFP.execute (FP.ProfiledRewrite.engine M) Bool.true p s
  exact ⟨lhs,rhs,hl,hr,hm.trans hn.symm⟩

/-- Complete initial-value K/V snapshot, including all splits and the full
exterior frame. This statement retains the actual FP32 load callback. -/
theorem kernel_results {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (p : Source.Parameters) (s : FP.Structural.State α) :
    ∃ lhs rhs,
      FP.Structural.exec M (Source.originalKernel p) s=some lhs ∧
      FP.Structural.exec M (Source.commutedKernel p) s=some rhs ∧
      lhs.mem=(KernelFP.copied M p s (p.dims/p.width)).mem ∧ rhs.mem=lhs.mem ∧
      (∀ r o,(∀ key q,q<p.dims/p.width → r≠KernelFP.outputRegion key ∨
        ∀ i : TileIndex [p.width],o≠(KernelFP.targetInt p s key q i).toNat) → lhs.mem r o=s.mem r o) := by
  obtain ⟨lhs,hl,hm⟩ := KernelFP.execute M Bool.false p s
  obtain ⟨rhs,hr,hn⟩ := KernelFP.execute M Bool.true p s
  refine ⟨lhs,rhs,hl,hr,hm,hn.trans hm.symm,?_⟩
  intro r o outside
  rw [hm]
  exact KernelFP.copied_frame M p s _ r o outside



#print_fp_assumptions source_equivalence
#axiomsClean source_equivalence
#axiomsClean kernel_results
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KvCacheCopy.KernelNativeResults
