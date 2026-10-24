import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.ReductionStage
import bench.repairs.tritonbench_g.chunk_linear_attn.IO
import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.PairOutputSafety
import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.IORun
import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.IOSafety

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.PairIO
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
noncomputable section
set_option maxHeartbeats 3000000
variable {α : Type} [Inhabited α]

theorem invocation_run (M : Algebra α) (D : DotPrimitive α) (stage : Stage) (p : Parameters)
    (scale : ℝ) (seed final : Bool) (pid : PID) (s : State α) :
    ∃ t, (invocation M D stage p scale seed final pid).run s = some t ∧
      t.mem = stageMemory M D stage p scale seed final pid s.mem := by
  cases stage with
  | forwardState => exact IORun.invocation M D .forwardState p scale seed final pid s
  | backwardState => exact IORun.invocation M D .backwardState p scale seed final pid s
  | backwardGradient => exact IORun.invocation M D .backwardGradient p scale seed final pid s
  | forwardOutput =>
    exact PairOutput.program M D p scale
      ((invocation M D .forwardOutput p scale seed final pid).initial s)

theorem grid (M : Algebra α) (D : DotPrimitive α) (stage : Stage) (p : Parameters)
    (scale : ℝ) (seed final : Bool) (work : List PID) (s : State α) :
    ∃ t, runSchedule (work.map (invocation M D stage p scale seed final)) s = some t ∧
      t.mem = gridMemory M D stage p scale seed final work s.mem := by
  induction work generalizing s with
  | nil => exact ⟨s,rfl,rfl⟩
  | cons pid rest ih =>
    obtain ⟨mid,run,mem⟩ := invocation_run M D stage p scale seed final pid s
    obtain ⟨t,runRest,memRest⟩ := ih mid
    refine ⟨t,?_,?_⟩
    · simp only [List.map_cons,runSchedule,run,Option.bind_some,runRest]
    · simpa only [gridMemory,List.foldl_cons,mem] using memRest

theorem four (M : Algebra α) (D : DotPrimitive α) (c : Real.Call)
    (work : KernelSchedule.Work) (s : State α) :
    ∃ t, runSchedule (jobs M D c work) s = some t ∧
      t.mem = modelFour M D c.forward c.backward c.scale c.seed c.final c.terminal
        work.forwardState work.forwardOutput work.backwardState work.backwardGradient s.mem := by
  obtain ⟨s1,e1,m1⟩ := grid M D .forwardState c.forward c.scale c.seed c.final work.forwardState s
  obtain ⟨s2,e2,m2⟩ := grid M D .forwardOutput c.forward c.scale Bool.false Bool.false work.forwardOutput s1
  obtain ⟨s3,e3,m3⟩ := grid M D .backwardState c.backward c.scale c.terminal c.seed work.backwardState s2
  obtain ⟨s4,e4,m4⟩ := grid M D .backwardGradient c.backward c.scale Bool.false Bool.false work.backwardGradient s3
  refine ⟨s4,?_,?_⟩
  · simp only [jobs,runSchedule_append,e1,e2,e3,e4,Option.bind_some]
  · simp only [m4,m3,m2,m1,modelFour]

theorem stages_run (M : Algebra α) (D : DotPrimitive α) (c : Real.Call)
    (work : KernelSchedule.Work) (legal : KernelSchedule.Legal c work) (s : State α) :
    ∃ t, runReductionStages (stages M D c work) s = some t ∧
      t.mem = completeMemory M D c.forward c.backward c.scale c.seed c.final c.terminal c.selection s.mem := by
  obtain ⟨mid,executed,mem⟩ := four M D c work s
  refine ⟨(LibraryReduction.sumStage M c.backward c.selection).run mid,?_,?_⟩
  · simp only [stages,runReductionStages_append,runReductionStages_invocations,
      executed,Option.bind_some,runReductionStages,ReductionStage.run]
  · rw [LibraryReduction.run_mem,mem]
    have correspondence := KernelSchedule.run_eq M D c work legal s.mem
    simp only [KernelSchedule.run,four_execution,bind,Option.bind_some,
      KernelExecution.source_complete] at correspondence
    exact Option.some.inj correspondence


variable {γ : Type}
private theorem schedule_of_each (jobs : List (Invocation α)) (bounds : RegionName → Nat)
    (each : ∀ job ∈ jobs, ∀ s, job.Safe bounds s) (s : State α) : ScheduleSafe bounds jobs s := by
  induction jobs generalizing s with
  | nil => trivial
  | cons job rest ih =>
    refine ⟨each job (by simp) s,?_⟩
    intro t _
    exact ih (fun j member => each j (by simp [member])) t

/-- Actual source access safety from the declared finite windows, for every
intermediate state. Numeric values do not alter these access footprints. -/
theorem jobs_safe (M : Algebra α) (D : DotPrimitive α) (c : KernelInterface.Config)
    (vals : ChannelValues γ (IOLayout.inputs c.dims)) (work : KernelSchedule.Work)
    (legal : KernelSchedule.Legal c.call work) (scale : ℝ) (bounds : RegionName → Nat)
    (backed : (IOLayout.realMemory γ c).InBounds bounds vals) (s : State α) :
    ScheduleSafe bounds (jobs M D {c.call with scale:=scale} work) s := by
  apply schedule_of_each
  intro job member state
  simp only [jobs,List.mem_append,List.mem_map] at member
  rcases member with ((⟨pid,member,rfl⟩ | ⟨pid,member,rfl⟩) | ⟨pid,member,rfl⟩) | ⟨pid,member,rfl⟩
  · have valid := (mem_grid _ _ _).mp (legal.forwardState.mem_iff.mp member)
    exact StageSafety.state_source_safe M D c.call.forward scale Bool.false c.seed c.final bounds _
      (IOGeometry.forward_state_capacity c vals bounds backed) valid.2.2
  · have valid := (mem_grid _ _ _).mp (legal.forwardOutput.mem_iff.mp member)
    exact PairOutput.source_safe M D c.call.forward scale bounds _
      (IOGeometry.output_stage_capacity c vals bounds backed) valid.2.2 valid.2.1
  · have valid := (mem_grid _ _ _).mp (legal.backwardState.mem_iff.mp member)
    exact StageSafety.state_source_safe M D c.call.backward scale Bool.true c.terminal c.seed bounds _
      (IOGeometry.backward_state_capacity c vals bounds backed) valid.2.2
  · have valid := (mem_grid _ _ _).mp (legal.backwardGradient.mem_iff.mp member)
    exact StageSafety.gradient_source_safe M D c.call.backward scale bounds _
      (IOGeometry.gradient_capacity c vals bounds backed) valid.2.2 valid.1 rfl

theorem stages_safe (M : Algebra α) (D : DotPrimitive α) (c : KernelInterface.Config)
    (vals : ChannelValues γ (IOLayout.inputs c.dims)) (work : KernelSchedule.Work)
    (legal : KernelSchedule.Legal c.call work) (scale : ℝ) (bounds : RegionName → Nat)
    (backed : (IOLayout.realMemory γ c).InBounds bounds vals) (s : State α) :
    ReductionScheduleSafe bounds (stages M D {c.call with scale:=scale} work) s := by
  apply (reductionScheduleSafe_append _ _ _ _).mpr
  refine ⟨(reductionScheduleSafe_invocations _ _ _).mpr (jobs_safe M D c vals work legal scale bounds backed s),?_⟩
  intro t _
  refine ⟨?_,fun _ _ => trivial⟩
  apply LibraryReduction.safe
  · have room := (IOGeometry.gradient_capacity c vals bounds backed).valuePartials
    simpa only [valueCount,Nat.mul_assoc] using room
  · have room := IOGeometry.output_capacity c vals bounds backed 3 (by constructor <;> simp)
    change (c.dims.B*c.dims.H)*(c.dims.T*(c.dims.V*1)) ≤ bounds "valueGradient" at room
    simpa only [valueCount,KernelInterface.Config.call,Real.Call.backward,
      Host.Dimensions.parameters,Nat.mul_assoc,Nat.mul_one] using room


#axiomsClean stages_run
#axiomsClean stages_safe
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.PairIO
