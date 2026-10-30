import bench.repairs.tritonbench_g.chunk_delta_fwd_static_dot.IO

namespace VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.GridSchedule
open VeriTile Triton FP.Structural FP.ProfiledRewrite ChunkDeltaFwd Interface
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
noncomputable section
set_option maxHeartbeats 8000000
variable {α : Type} [Inhabited α]

/-- Reuse the established whole-program memory result under the shared launch
ABI. No equality of irrelevant undefined-value fields is assumed. -/
theorem executes (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (schedule : List PID) (input : Memory α) (s : State α) (hi : SameInputs s.mem input) :
    ∃ final, runSchedule (schedule.map (invocation M D p)) s = some final ∧
      final.mem = gridMemory M D p schedule input s.mem ∧ SameInputs final.mem input := by
  induction schedule generalizing s with
  | nil => exact ⟨s,rfl,rfl,hi⟩
  | cons pid rest ih =>
    let job := invocation M D p pid
    obtain ⟨t,he,hm,ht⟩ := program M D p input (job.initial s) rfl hi
    have hrun : job.run s = some t := he
    change t.mem = programMemory M D p pid input s.mem at hm
    obtain ⟨u,hu,hmem,hinputs⟩ := ih t ht
    refine ⟨u,?_,?_,hinputs⟩
    · change (job.run s).bind (runSchedule (rest.map (invocation M D p))) = some u
      rw [hrun]
      exact hu
    · rw [hmem,hm]
      rfl

#axiomsClean executes
end
end VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.GridSchedule
