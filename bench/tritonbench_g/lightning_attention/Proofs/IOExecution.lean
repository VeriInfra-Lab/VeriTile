import bench.tritonbench_g.lightning_attention.IO
import bench.tritonbench_g.lightning_attention.PhaseReal
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.IOExecution
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

/-- The shared Invocation runs the same original source from its current memory;
no launch-order or local-register assumption is needed for the result. -/
theorem program (M : Algebra α) (hm : MaskArithmetic M) (D : DotPrimitive α)
    (stage : Stage) (d : Host.Dimensions) (pid : PID) (s : State α) :
    ∃ t, (invocation M D stage d pid).run s = some t ∧
      t.mem = stageMemory M D stage d pid s.mem := by
  let start := (invocation M D stage d pid).initial s
  have hmem : start.mem = s.mem := Invocation.initial_mem _ _
  have hhead : start.pids 0 = pid.head := rfl
  have hchannel : start.pids 1 = pid.channel := rfl
  cases stage with
  | forward =>
    obtain ⟨t,ran,mem,_⟩ := forwardProgram M D d.parameters s.mem start (by rw [hmem]; exact Inputs.refl _)
    refine ⟨t,ran,?_⟩
    simpa only [stageMemory,hhead,hchannel,hmem] using mem
  | intra =>
    obtain ⟨t,ran,mem,_⟩ := intraProgram M hm D d.parameters rfl start
    refine ⟨t,ran,?_⟩
    simpa only [stageMemory,hhead,hchannel,hmem] using mem
  | inter =>
    obtain ⟨t,ran,mem,_⟩ := interProgram M D d.parameters rfl s.mem start (by rw [hmem]; exact Inputs.refl _)
    refine ⟨t,ran,?_⟩
    simpa only [stageMemory,hhead,hmem] using mem

theorem grid (M : Algebra α) (hm : MaskArithmetic M) (D : DotPrimitive α)
    (stage : Stage) (d : Host.Dimensions) (pids : List PID) (s : State α) :
    ∃ t, runSchedule (pids.map (invocation M D stage d)) s = some t ∧
      t.mem = gridMemory M D stage d pids s.mem := by
  induction pids generalizing s with
  | nil => exact ⟨s,rfl,rfl⟩
  | cons pid rest ih =>
    obtain ⟨u,ran,mem⟩ := program M hm D stage d pid s
    obtain ⟨t,restRun,restMem⟩ := ih u
    refine ⟨t,?_,?_⟩
    · simpa only [List.map_cons,runSchedule,ran,Option.bind_some] using restRun
    · simpa only [gridMemory,List.foldl_cons,mem] using restMem

theorem forward (d : Host.Dimensions) (s : State ℝ) :
    ∃ t, runSchedule (stageSchedule Real.algebra Real.dotPrimitive .forward d) s = some t ∧
      t.mem = Real.afterForward d s.mem :=
  grid Real.algebra realMasks Real.dotPrimitive .forward d _ s

theorem backward (d : Host.Dimensions) (s : State ℝ) :
    ∃ t, runSchedule (stageSchedule Real.algebra Real.dotPrimitive .intra d ++
        stageSchedule Real.algebra Real.dotPrimitive .inter d) s = some t ∧
      t.mem = Real.backwardMemory d s.mem := by
  obtain ⟨u,first,firstMem⟩ := grid Real.algebra realMasks Real.dotPrimitive .intra d (KernelOnly.grid .intra d) s
  obtain ⟨t,second,secondMem⟩ := grid Real.algebra realMasks Real.dotPrimitive .inter d (KernelOnly.grid .inter d) u
  refine ⟨t,?_,?_⟩
  · rw [runSchedule_append,stageSchedule,first,Option.bind_some]
    exact second
  · rw [secondMem,firstMem]
    rfl

#axiomsClean program
#axiomsClean forward
#axiomsClean backward
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.IOExecution
