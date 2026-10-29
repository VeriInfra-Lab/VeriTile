import bench.repairs.tritonbench_g.destindex_copy_kv1.Correct
import bench.repairs.tritonbench_g.destindex_copy_kv1.FPEquiv
import bench.repairs.tritonbench_g.destindex_copy_kv1.Misc.NativeObservation
import bench.repairs.tritonbench_g.destindex_copy_kv1.Proofs.PhysicalCorrect

/-! Complete Tensor applications and Python-check correspondence. The primary
kernel results remain in Correct.lean and FPEquiv.lean; NativeObservation.lean
contains the auxiliary native payload-copy certificate. -/
namespace VeriTile.Bench.Repairs.DestindexCopyKv1.TensorBridge
open VeriTile.Triton FP.Structural
noncomputable section

theorem python_checks_iff (m : Option Inputs) (w : Nat → Word) :
    checkArguments m w=.ok () ↔ Admitted m w := checkArguments_iff m w

/-- Arbitrary Real payloads execute the complete selected source; no finite-word
encoding or FP32-to-Real equality is used. Floating arithmetic is absent. -/
theorem destindex_copy_kv1_real (rest : Algebra ℝ) (m : Inputs) (w : Nat → Word)
    (h : InputContract m w) (before : Nat → ℝ) (other : State ℝ) :
    Spec.Real (RowCopy (config m) (destination w) (fun x => before (addr (base m .source) x))
      (fun x => before (addr (base m .output) x))
      (fun x => execute rest m w before other (addr (base m .output) x)) ∧
    (∀ cell, (¬∃ x, touched (config m) (destination w) x ∧ cell=addr (base m .output) x) →
      execute rest m w before other cell=before cell)) := execute_correct rest m w h before other

/-- Checked metadata binds the actual finite addresses and initial words to
the complete typed source. All FP32 payload bits and exterior bytes are kept. -/
theorem destindex_copy_kv1_fp32 (rest : Algebra Word) (m : Inputs) (w : Nat → Word)
    (h : InputContract m w) (before : Memory) (other : State Word)
    (binding : TensorBinding m w before) :
    SourceConformance m w before ∧
    RowCopy (config m) (destination w) (fun x => before (addr (base m .source) x))
      (fun x => before (addr (base m .output) x))
      (fun x => Physical.launch m (List.range (config m).count) before (addr (base m .output) x)) ∧
    (∀ byte, (∀ x, touched (config m) (destination w) x →
      byte<addr (base m .output) x ∨ addr (base m .output) x+4≤byte) →
      observeByte (Physical.launch m (List.range (config m).count) before) byte=observeByte before byte) := by
  obtain ⟨hv, hf⟩ := Physical.correct rest m w h before binding other (List.range (config m).count) (by simp)
  exact ⟨source_conformance m w h before binding, hv,
    fun byte outside => byte_frame m w h before _ hf byte outside⟩

#axiomsClean destindex_copy_kv1_real
#axiomsClean destindex_copy_kv1_fp32
#axiomsClean python_checks_iff
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DestindexCopyKv1.TensorBridge
