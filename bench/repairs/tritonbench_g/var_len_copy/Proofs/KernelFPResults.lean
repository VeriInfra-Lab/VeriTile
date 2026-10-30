import bench.repairs.tritonbench_g.var_len_copy.Proofs.KernelFPFrame
import VeriTile.Triton.Float.ProfiledMemory

namespace VeriTile.Bench.Repairs.VarLenCopy.FPExecution
open VeriTile.Triton FP.ProfiledMemory
open scoped VeriTile.Spec
noncomputable section

/-- The complete destination frame is shared by both successful sources. -/
theorem kernel_results {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (s : FP.Structural.State α) :
    ∃ final, FP.Structural.exec M Source.originalKernel s = some final ∧
      FP.Structural.exec M Source.commutedKernel s = some final ∧
      (∀ r i, ¬KernelFP.Footprint "output"
        ((s.mem "length" (s.pids 0)).read .int)
        ((s.mem "new" (s.pids 0)).read .int) r i → final.mem r i = s.mem r i) := by
  obtain ⟨final,he⟩ := KernelFP.source_success M
    ⟨"old"⟩ ⟨"length"⟩ ⟨"new"⟩ "source" "output" 256 s
  have eq := KernelFP.source_equivalence M
    ⟨"old"⟩ ⟨"length"⟩ ⟨"new"⟩ "source" "output" 256 s
  exact ⟨final,he,eq.symm.trans he,KernelFP.source_frame M
    ⟨"old"⟩ ⟨"length"⟩ ⟨"new"⟩ "source" "output" 256 s final he⟩

/-- The existing whole-source typed-memory equivalence, assembled once below
both the reader and the optional raw execution/frame helper. -/
theorem source_equivalence :
    (⟨Source.originalKernel, []⟩ : Program) ≡[([] : Spec.Assumptions FP.GuardedFragment)]
      (⟨Source.commutedKernel, []⟩ : Program) := by
  apply Spec.FloatingPoint.ofNumerical
  · rfl
  · rfl
  intro α _ M D _ s _
  obtain ⟨final,he⟩ := KernelFP.source_success (FP.ProfiledRewrite.engine M)
    ⟨"old"⟩ ⟨"length"⟩ ⟨"new"⟩ "source" "output" 256 s
  have eq := KernelFP.source_equivalence (FP.ProfiledRewrite.engine M)
    ⟨"old"⟩ ⟨"length"⟩ ⟨"new"⟩ "source" "output" 256 s
  exact ⟨final,final,he,eq.symm.trans he,rfl⟩


#axiomsClean source_equivalence
#axiomsClean kernel_results
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.VarLenCopy.FPExecution
