import bench.tritonbench_g.chunk_linear_attn.Proofs.ReductionStage
import bench.tritonbench_g.chunk_linear_attn.IO
import bench.tritonbench_g.chunk_linear_attn.Proofs.IOGeometry
import bench.tritonbench_g.chunk_linear_attn.Proofs.StageSafetyStateComplete
import bench.tritonbench_g.chunk_linear_attn.Proofs.StageSafetyOutputComplete
import bench.tritonbench_g.chunk_linear_attn.Proofs.StageSafetyGradientComplete

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.IOSafety
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
noncomputable section
set_option maxHeartbeats 2000000
variable {α β : Type} [Inhabited β]

private theorem schedule_of_each (jobs : List (Invocation β)) (bounds : RegionName → Nat)
    (each : ∀ job ∈ jobs, ∀ s, job.Safe bounds s) (s : State β) : ScheduleSafe bounds jobs s := by
  induction jobs generalizing s with
  | nil => trivial
  | cons job rest ih =>
    refine ⟨each job (by simp) s,?_⟩
    intro t _
    exact ih (fun j member => each j (by simp [member])) t

/-- Actual source access safety from the declared finite windows, for every
intermediate state. Numeric values do not alter these access footprints. -/
theorem jobs (M : Algebra β) (D : DotPrimitive β) (c : KernelInterface.Config)
    (vals : ChannelValues α (IOLayout.inputs c.dims)) (work : KernelSchedule.Work)
    (legal : KernelSchedule.Legal c.call work) (scale : ℝ) (bounds : RegionName → Nat)
    (backed : (IOLayout.realMemory α c).InBounds bounds vals) (s : State β) :
    ScheduleSafe bounds (IO.jobs M D {c.call with scale:=scale} work) s := by
  apply schedule_of_each
  intro job member state
  simp only [IO.jobs,List.mem_append,List.mem_map] at member
  rcases member with ((⟨pid,member,rfl⟩ | ⟨pid,member,rfl⟩) | ⟨pid,member,rfl⟩) | ⟨pid,member,rfl⟩
  · have valid := (mem_grid _ _ _).mp (legal.forwardState.mem_iff.mp member)
    exact StageSafety.state_source_safe M D c.call.forward scale Bool.false c.seed c.final bounds _
      (IOGeometry.forward_state_capacity c vals bounds backed) valid.2.2
  · have valid := (mem_grid _ _ _).mp (legal.forwardOutput.mem_iff.mp member)
    exact StageSafety.output_source_safe M D c.call.forward scale bounds _
      (IOGeometry.output_stage_capacity c vals bounds backed) valid.2.2 valid.2.1
  · have valid := (mem_grid _ _ _).mp (legal.backwardState.mem_iff.mp member)
    exact StageSafety.state_source_safe M D c.call.backward scale Bool.true c.terminal c.seed bounds _
      (IOGeometry.backward_state_capacity c vals bounds backed) valid.2.2
  · have valid := (mem_grid _ _ _).mp (legal.backwardGradient.mem_iff.mp member)
    exact StageSafety.gradient_source_safe M D c.call.backward scale bounds _
      (IOGeometry.gradient_capacity c vals bounds backed) valid.2.2 valid.1 rfl

theorem stages (M : Algebra β) (D : DotPrimitive β) (c : KernelInterface.Config)
    (vals : ChannelValues α (IOLayout.inputs c.dims)) (work : KernelSchedule.Work)
    (legal : KernelSchedule.Legal c.call work) (scale : ℝ) (bounds : RegionName → Nat)
    (backed : (IOLayout.realMemory α c).InBounds bounds vals) (s : State β) :
    ReductionScheduleSafe bounds (IO.stages M D {c.call with scale:=scale} work) s := by
  apply (reductionScheduleSafe_append _ _ _ _).mpr
  refine ⟨(reductionScheduleSafe_invocations _ _ _).mpr (jobs M D c vals work legal scale bounds backed s),?_⟩
  intro t _
  refine ⟨?_,fun _ _ => trivial⟩
  apply LibraryReduction.safe
  · have room := (IOGeometry.gradient_capacity c vals bounds backed).valuePartials
    simpa only [valueCount,Nat.mul_assoc] using room
  · have room := IOGeometry.output_capacity c vals bounds backed 3 (by constructor <;> simp)
    change (c.dims.B*c.dims.H)*(c.dims.T*(c.dims.V*1)) ≤ bounds "valueGradient" at room
    simpa only [valueCount,KernelInterface.Config.call,Real.Call.backward,
      Host.Dimensions.parameters,Nat.mul_assoc,Nat.mul_one] using room

#axiomsClean jobs
#axiomsClean stages
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.IOSafety
