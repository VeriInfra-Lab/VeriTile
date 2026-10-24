import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.Boundary
import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.FPEquivPublic

/-! Exact native-width source and FP32 +0 operations. Caller objects, all
written output bytes, and the full untouched byte heap are covered. No FP=Real
rounding assumptions or numerical deferrals are needed for this zero kernel. -/
noncomputable section
namespace VeriTile.Bench.Repairs.SpinningLock.FPEquiv
open VeriTile.Bench.TritonBenchG.SpinningLock VeriTile.Triton Admission

specification checked_returned (ok : check s p c l = Bool.true) (plan : Phased.Plan)
    (before : ByteHeap) (input : InputBytes before p c l)
    (allowed : Phased.AllowedExecution fp32 s.args
      (wrapperHost s p c l setup pq cq lq first second) plan (rawFP p c l)) :
    let result := observe fp32 s p c l setup pq cq lq first second plan (rawFP p c l)
    result.partialObject=p.object ∧ result.outputObject=c.object ∧ result.lockObject=l.object ∧
    Wide.PrefixFacts fp32 s.args (checkedBuffers ok) (Phased.flatten s.args plan) (rawFP p c l) result.device ∧
    ∀ a, Untouched result.device.events p c l a →
      heap before p c l result.device.final.values.mem a=before a := by
  exact ⟨rfl,rfl,rfl,checked_prefix fp32 fp32_positive_zero_add (fun w=>⟨w⟩) rfl ok plan,
    complete_byte_frame ok plan before input⟩

specification checked_synchronized_words (ok : check s p c l = Bool.true) (plan : Phased.Plan)
    (sync : Phased.Synchronized fp32 s.args plan (rawFP p c l)) :
    (wrapperHost s p c l true true true true true true).returned=.ok () ∧
    ∀ a, (rawResult fp32 s plan (rawFP p c l)).final.values.mem.output a=
      completeTarget s.args (rawFP p c l) a := by
  exact ⟨checked_none ok,
    (checked_synchronized fp32 fp32_positive_zero_add (fun w=>⟨w⟩) rfl ok plan sync).mathematical_output⟩

specification checked_synchronized_bytes (ok : check s p c l = Bool.true) (plan : Phased.Plan)
    (sync : Phased.Synchronized fp32 s.args plan (rawFP p c l)) (before : ByteHeap)
    (i j : Nat) (hi : i<c.elements) (hj : j<4) :
    heap before p c l (rawResult fp32 s plan (rawFP p c l)).final.values.mem (c.base+4*i+j)=
      byteOf (completeTarget s.args (rawFP p c l) i).bits j := by
  rw [heap_output_bytes ok before _ i j hi hj,(checked_synchronized_words ok plan sync).2]

abbrev checked_masked_pointers := @Admission.formed_output_bounds
abbrev exact_graph_equivalence := @repaired_full_fp32_equivalence

#specNonCircular completeTarget avoiding [Admission.rawResult,Wide.launch,Wide.nativeRun]
#specNonCircular Admission.rawResult avoiding [completeTarget]
#axiomsClean checked_returned
#axiomsClean checked_synchronized_words
#axiomsClean checked_synchronized_bytes
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SpinningLock.FPEquiv
