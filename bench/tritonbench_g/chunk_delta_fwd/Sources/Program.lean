import bench.tritonbench_g.chunk_delta_fwd.Sources.OuterLoop
import bench.tritonbench_g.chunk_delta_fwd.Sources.Finish
namespace VeriTile.Bench.Repairs.ChunkDeltaFwd
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

theorem initial_sameInputs (M : Algebra α) (p : Parameters) (pid : PID)
    (a b : Memory α) (h : SameInputs a b) : initial M p pid a=initial M p pid b := by
  funext i
  simp only [initial,h "initial" (by simp)]

/-- Successful execution of the entire repaired source, from arbitrary current
memory agreeing on its read regions with an immutable input snapshot. -/
theorem program (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (input : Memory α) (s : State α) (hk : s.pids 0=0) (hi : SameInputs s.mem input) :
    ∃ t, FP.Structural.exec (engine (withDot M D)) (kernel p) s=some t ∧
      t.mem=programMemory M D p (s.pids 1,s.pids 2) input s.mem ∧
      SameInputs t.mem input := by
  let pid : PID := (s.pids 1,s.pids 2)
  let start := entryExecution M D p s
  have hv := entry_ready M D p s hk
  have hc : Ready pid input start.val :=
    ⟨hv.1.inputs.trans hi,hv.1.ik,hv.1.iv,hv.1.bh⟩
  have hh : HasH p (initial M p pid input) start.val := by
    rw [← initial_sameInputs M p pid s.mem input hi]
    exact hv.2
  obtain ⟨mid,hm,hctx,hhmid,hmem⟩ := outerLoop M D p pid input start.val hc hh
  let hfinal := states M D p pid input p.NT
  let finish := finishState M p pid mid hfinal
  have hf := finish_run M D p pid input mid hfinal hctx hhmid
  refine ⟨finish,?_,?_,(finish_ready M p pid input mid hfinal hctx).inputs⟩
  · have hl : step (engine (withDot M D)) (.forRange "i_t" 0 p.NT 1 (outer p).body)
        start.val=some mid := by simpa only [step] using hm
    change run (engine (withDot M D)) (kernel p).surfaceBody s=some finish
    rw [outer_parts,List.append_assoc,run_append,start.property]
    simp only [Option.bind_some,List.singleton_append,run,hl,bind,hf]
    rfl
  · change (finishState M p pid mid hfinal).mem=_
    rw [finish_memory,hmem]
    dsimp only [start]
    rw [entry_memory]
    rfl

#axiomsClean program
#specNonCircular programMemory avoiding [run,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.ChunkDeltaFwd
