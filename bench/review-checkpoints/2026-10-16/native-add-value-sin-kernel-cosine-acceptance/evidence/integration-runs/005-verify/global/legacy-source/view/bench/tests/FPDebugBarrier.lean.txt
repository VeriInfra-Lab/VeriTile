import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Semantics
import VeriTile.Meta.StatementAudit

/-! Exact sequential single-program support for the debug barrier marker.
No claim about native thread convergence or cross-program synchronization.
Async and unknown effects remain rejected, including after a barrier. -/
namespace FPDebugBarrierTests
open VeriTile Triton FP.Structural
noncomputable section

variable {α : Type} [Inhabited α] (M : Algebra α) (s : State α)

theorem debug_barrier_state : step M (.effectMarker "tl.debug_barrier") s = some s := by
  simp [step]

theorem debug_barrier_run : run M [.effectMarker "tl.debug_barrier"] s = some s := by
  simp [run,debug_barrier_state]

theorem debug_barrier_between (before after : List ComputeStmt) :
    run M (before ++ .effectMarker "tl.debug_barrier" :: after) s =
      run M (before ++ after) s := by
  simp only [run_append,run,debug_barrier_state,bind,Option.bind_some]

theorem unknown_effect (name : String) (h : name ≠ "tl.debug_barrier") :
    step M (.effectMarker name) s = none := by
  simp [step,h]

theorem async_copy_rejected : step M (.effectMarker "tl.async_copy") s = none := by
  simp [step]

theorem async_wait_rejected : step M (.effectMarker "tl.async_wait") s = none := by
  simp [step]

theorem bare_barrier_name_rejected : step M (.effectMarker "debug_barrier") s = none := by
  simp [step]

theorem call_spelling_rejected : step M (.effectMarker "tl.debug_barrier()") s = none := by
  simp [step]

theorem suffixed_barrier_rejected : step M (.effectMarker "tl.debug_barrier.extra") s = none := by
  simp [step]

theorem async_before_barrier_rejected :
    run M [.effectMarker "tl.async_copy",.effectMarker "tl.debug_barrier"] s = none := by
  simp [run,step]

theorem async_after_barrier_rejected :
    run M [.effectMarker "tl.debug_barrier",.effectMarker "tl.async_wait"] s = none := by
  simp [run,step]

def projectedUnsupported : ComputeStmt := .alg (.ifThen (.constBool Bool.false) [])

theorem unsupported_projects : projectedUnsupported.toAlgorithm? =
    Except.ok (.ifThen (.constBool Bool.false) []) := rfl

theorem unsupported_stays_failed : step M projectedUnsupported s = none := by
  simp [projectedUnsupported,step,stepAlg]

theorem failed_pair_not_success :
    ¬ ∃ out, run M [projectedUnsupported] s = some out ∧
      run M [projectedUnsupported] s = some out := by
  simp [run,unsupported_stays_failed]

theorem mathematical_projection :
    (ComputeStmt.effectMarker "tl.debug_barrier").toAlgorithm? =
      Except.ok (.ifThen (.constBool Bool.false) []) := rfl

theorem projected_state (u : BlockState) :
    stepStmt (.ifThen (.constBool Bool.false) []) u = some u := by
  simp [stepStmt,evalOp.eq_def,Tile.scalar_data]

#axiomsClean debug_barrier_state
#axiomsClean debug_barrier_run
#axiomsClean debug_barrier_between
#axiomsClean unknown_effect
#axiomsClean async_copy_rejected
#axiomsClean async_wait_rejected
#axiomsClean bare_barrier_name_rejected
#axiomsClean call_spelling_rejected
#axiomsClean suffixed_barrier_rejected
#axiomsClean async_before_barrier_rejected
#axiomsClean async_after_barrier_rejected
#axiomsClean unsupported_projects
#axiomsClean unsupported_stays_failed
#axiomsClean failed_pair_not_success
#axiomsClean mathematical_projection
#axiomsClean projected_state
end
end FPDebugBarrierTests
