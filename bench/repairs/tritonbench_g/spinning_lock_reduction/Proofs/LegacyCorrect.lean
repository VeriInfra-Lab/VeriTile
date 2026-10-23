import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.Boundary
import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.CorrectPublic

/-! Real specification of the actual caller-owned output. The independent
mathematical rectangle target covers every output cell and preserves its
complement. Submission returns None; synchronization supplies completion. -/
noncomputable section
namespace VeriTile.Bench.Repairs.SpinningLock.Correct
open VeriTile.Bench.TritonBenchG.SpinningLock Admission

specification checked_returned (decode : BitVec 32 → ℝ) (zero : decode 0=0)
    (ok : check s p c l = Bool.true) (plan : Phased.Plan)
    (allowed : Phased.AllowedExecution realBackend s.args
      (wrapperHost s p c l setup pq cq lq first second) plan (memory decode p c l)) :
    let result := observe realBackend s p c l setup pq cq lq first second plan (memory decode p c l)
    result.partialObject=p.object ∧ result.outputObject=c.object ∧ result.lockObject=l.object ∧
    Wide.PrefixFacts realBackend s.args (checkedBuffers ok) (Phased.flatten s.args plan)
      (memory decode p c l) result.device := by
  exact ⟨rfl,rfl,rfl,checked_prefix realBackend (by simp [realBackend]) decode zero ok plan⟩

specification checked_synchronized_output (decode : BitVec 32 → ℝ) (zero : decode 0=0)
    (ok : check s p c l = Bool.true) (plan : Phased.Plan)
    (sync : Phased.Synchronized realBackend s.args plan (memory decode p c l)) :
    (wrapperHost s p c l true true true true true true).returned=.ok () ∧
    ∀ a, (rawResult realBackend s plan (memory decode p c l)).final.values.mem.output a=
      completeTarget s.args (memory decode p c l) a := by
  exact ⟨checked_none ok,
    (checked_synchronized realBackend (by simp [realBackend]) decode zero ok plan sync).mathematical_output⟩

#specNonCircular completeTarget avoiding [Admission.rawResult,Wide.launch,Wide.nativeRun]
#axiomsClean checked_returned
#axiomsClean checked_synchronized_output
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SpinningLock.Correct
