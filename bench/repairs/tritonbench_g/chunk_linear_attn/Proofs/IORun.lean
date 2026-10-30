import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.ReductionStage
import bench.repairs.tritonbench_g.chunk_linear_attn.IO

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.IORun
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
noncomputable section
set_option maxHeartbeats 2000000
variable {α : Type} [Inhabited α]

/-- The reused program proofs hold for arbitrary local/undefined state.
Installing launch registers therefore does not require a new memory model. -/
theorem invocation (M : Algebra α) (D : DotPrimitive α) (stage : Stage) (p : Parameters)
    (scale : ℝ) (seed final : Bool) (pid : PID) (s : State α) :
    ∃ t, (IO.invocation M D stage p scale seed final pid).run s = some t ∧
      t.mem = stageMemory M D stage p scale seed final pid s.mem := by
  cases stage with
  | forwardState =>
    obtain ⟨t,he,hm,_⟩ := stateProgram M D p scale Bool.false seed final s.mem
      ((IO.invocation M D .forwardState p scale seed final pid).initial s) (StateInputs.refl s.mem)
    exact ⟨t,he,hm⟩
  | backwardState =>
    obtain ⟨t,he,hm,_⟩ := stateProgram M D p scale Bool.true seed final s.mem
      ((IO.invocation M D .backwardState p scale seed final pid).initial s) (StateInputs.refl s.mem)
    exact ⟨t,he,hm⟩
  | forwardOutput =>
    exact outputProgram M D p scale
      ((IO.invocation M D .forwardOutput p scale seed final pid).initial s)
  | backwardGradient =>
    exact gradientProgram M D p scale
      ((IO.invocation M D .backwardGradient p scale seed final pid).initial s)

theorem grid (M : Algebra α) (D : DotPrimitive α) (stage : Stage) (p : Parameters)
    (scale : ℝ) (seed final : Bool) (work : List PID) (s : State α) :
    ∃ t, runSchedule (work.map (IO.invocation M D stage p scale seed final)) s = some t ∧
      t.mem = gridMemory M D stage p scale seed final work s.mem := by
  induction work generalizing s with
  | nil => exact ⟨s,rfl,rfl⟩
  | cons pid rest ih =>
    obtain ⟨mid,run,mem⟩ := invocation M D stage p scale seed final pid s
    obtain ⟨t,runRest,memRest⟩ := ih mid
    refine ⟨t,?_,?_⟩
    · simp only [List.map_cons,runSchedule,run,Option.bind_some,runRest]
    · simpa only [gridMemory,List.foldl_cons,mem] using memRest

theorem four (M : Algebra α) (D : DotPrimitive α) (c : Real.Call)
    (work : KernelSchedule.Work) (s : State α) :
    ∃ t, runSchedule (IO.jobs M D c work) s = some t ∧
      t.mem = modelFour M D c.forward c.backward c.scale c.seed c.final c.terminal
        work.forwardState work.forwardOutput work.backwardState work.backwardGradient s.mem := by
  obtain ⟨s1,e1,m1⟩ := grid M D .forwardState c.forward c.scale c.seed c.final work.forwardState s
  obtain ⟨s2,e2,m2⟩ := grid M D .forwardOutput c.forward c.scale Bool.false Bool.false work.forwardOutput s1
  obtain ⟨s3,e3,m3⟩ := grid M D .backwardState c.backward c.scale c.terminal c.seed work.backwardState s2
  obtain ⟨s4,e4,m4⟩ := grid M D .backwardGradient c.backward c.scale Bool.false Bool.false work.backwardGradient s3
  refine ⟨s4,?_,?_⟩
  · simp only [IO.jobs,runSchedule_append,e1,e2,e3,e4,Option.bind_some]
  · simp only [m4,m3,m2,m1,modelFour]

theorem stages (M : Algebra α) (D : DotPrimitive α) (c : Real.Call)
    (work : KernelSchedule.Work) (legal : KernelSchedule.Legal c work) (s : State α) :
    ∃ t, runReductionStages (IO.stages M D c work) s = some t ∧
      t.mem = completeMemory M D c.forward c.backward c.scale c.seed c.final c.terminal c.selection s.mem := by
  obtain ⟨mid,executed,mem⟩ := four M D c work s
  refine ⟨(LibraryReduction.sumStage M c.backward c.selection).run mid,?_,?_⟩
  · simp only [IO.stages,runReductionStages_append,runReductionStages_invocations,
      executed,Option.bind_some,runReductionStages,ReductionStage.run]
  · rw [LibraryReduction.run_mem,mem]
    have correspondence := KernelSchedule.run_eq M D c work legal s.mem
    simp only [KernelSchedule.run,four_execution,bind,Option.bind_some,
      KernelExecution.source_complete] at correspondence
    exact Option.some.inj correspondence

#axiomsClean invocation
#axiomsClean grid
#axiomsClean four
#axiomsClean stages
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.IORun
