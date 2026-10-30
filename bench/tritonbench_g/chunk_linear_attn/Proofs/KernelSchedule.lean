import bench.tritonbench_g.chunk_linear_attn.Proofs.KernelExecution

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.KernelSchedule
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
noncomputable section
set_option maxHeartbeats 10000000
variable {α : Type} [Inhabited α]

theorem state_hit (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (rev seed final : Bool) (pid : PID) (input left right : Memory α) (r : RegionName) (o : Nat)
    (hit : StateFootprint p pid rev final r o) :
    stateMemory M D p pid input left scale rev seed final r o=
      stateMemory M D p pid input right scale rev seed final r o := by
  rcases hit with ⟨rfl,c,hc,i,hi,rfl⟩|⟨rfl,rfl,i,hi,rfl⟩
  · have hn := Geometry.chunk_lt p rev c hc
    have undo := Ownership.chunk_owner p rev c hc
    have hl := state_read_history M D p pid input left scale rev seed final (chunk p rev c) i hn hi
    have hr := state_read_history M D p pid input right scale rev seed final (chunk p rev c) i hn hi
    rw [undo] at hl hr
    exact hl.trans hr.symm
  · exact (state_read_final M D p pid input left scale rev seed i hi).trans
      (state_read_final M D p pid input right scale rev seed i hi).symm

theorem frozen_hit (M : Algebra α) (D : DotPrimitive α) (stage : Stage) (p : Parameters)
    (scale : ℝ) (seed final : Bool) (pid : PID) (valid : ValidPID stage p pid)
    (input left right : Memory α) (r : RegionName) (o : Nat) (hit : Footprint stage p final pid r o) :
    frozenStage M D stage p scale seed final pid input left r o=
      frozenStage M D stage p scale seed final pid input right r o := by
  cases stage with
  | forwardState => exact state_hit M D p scale Bool.false seed final pid input left right r o hit
  | backwardState => exact state_hit M D p scale Bool.true seed final pid input left right r o hit
  | forwardOutput =>
    rcases hit with ⟨rfl,i,hi,rfl⟩
    exact (output_read M D p pid input left scale i hi).trans (output_read M D p pid input right scale i hi).symm
  | backwardGradient =>
    rcases hit with ⟨hr,i,hi,rfl⟩|⟨rfl,v,hv,i,hi,rfl⟩
    · rcases hr with rfl|rfl
      · exact (gradient_read_query M D p pid input left scale i hi).trans
          (gradient_read_query M D p pid input right scale i hi).symm
      · exact (gradient_read_key M D p pid input left scale i hi).trans
          (gradient_read_key M D p pid input right scale i hi).symm
    · exact (gradient_read_value M D p pid input left scale v i valid.2.2 hv hi).trans
        (gradient_read_value M D p pid input right scale v i valid.2.2 hv hi).symm

theorem grid_permutation (M : Algebra α) (D : DotPrimitive α) (stage : Stage) (p : Parameters)
    (scale : ℝ) (seed final : Bool) (work : List PID) (legal : work.Perm (grid stage p)) (input : Memory α) :
    gridMemory M D stage p scale seed final work input=
      gridMemory M D stage p scale seed final (grid stage p) input := by
  classical
  funext r o
  have valid : ∀ pid∈work,ValidPID stage p pid := fun pid hp =>
    (mem_grid _ _ _).mp (legal.mem_iff.mp hp)
  by_cases written : ∃ pid,pid∈grid stage p ∧ Footprint stage p final pid r o
  · rcases written with ⟨pid,hp,hit⟩
    have hv := (mem_grid _ _ _).mp hp
    have independent := fun memory => frozen_hit M D stage p scale seed final pid hv input memory input r o hit
    have left := grid_read M D stage p scale seed final work input pid r o _ valid
      (legal.mem_iff.mpr hp) hit independent
    have right := grid_read M D stage p scale seed final (grid stage p) input pid r o _
      (fun other ho => (mem_grid _ _ _).mp ho) hp hit independent
    exact left.trans right.symm
  · have outside : ∀ pid∈grid stage p,¬Footprint stage p final pid r o :=
      fun pid hp hit => written ⟨pid,hp,hit⟩
    rw [grid_frame M D stage p scale seed final work input r o
      (fun pid hp => outside pid (legal.mem_iff.mp hp)),
      grid_frame M D stage p scale seed final (grid stage p) input r o outside]

structure Work where
  forwardState : List PID
  forwardOutput : List PID
  backwardState : List PID
  backwardGradient : List PID

structure Legal (c : Real.Call) (work : Work) : Prop where
  forwardState : work.forwardState.Perm (grid .forwardState c.forward)
  forwardOutput : work.forwardOutput.Perm (grid .forwardOutput c.forward)
  backwardState : work.backwardState.Perm (grid .backwardState c.backward)
  backwardGradient : work.backwardGradient.Perm (grid .backwardGradient c.backward)

def run (M : Algebra α) (D : DotPrimitive α) (c : Real.Call) (work : Work) (input : Memory α) : Option (Memory α) := do
  let kernels ← executeFour M D c.forward c.backward c.scale c.seed c.final c.terminal
    work.forwardState work.forwardOutput work.backwardState work.backwardGradient input
  return hostSumMemory M c.backward c.selection kernels kernels

theorem run_eq (M : Algebra α) (D : DotPrimitive α) (c : Real.Call) (work : Work)
    (legal : Legal c work) (input : Memory α) : run M D c work input=KernelExecution.run M D c input := by
  rw [KernelExecution.source_complete]
  simp only [run,four_execution,bind,Option.bind_some,completeMemory,modelFour]
  rw [grid_permutation M D .forwardState c.forward c.scale c.seed c.final _ legal.forwardState,
    grid_permutation M D .forwardOutput c.forward c.scale Bool.false Bool.false _ legal.forwardOutput,
    grid_permutation M D .backwardState c.backward c.scale c.terminal c.seed _ legal.backwardState,
    grid_permutation M D .backwardGradient c.backward c.scale Bool.false Bool.false _ legal.backwardGradient]
  rfl

def runNative (profile : NativeTerms.FP32Profile) (c : Real.Call) (scale : BitVec 32)
    (work : Work) (input : RegionName → Nat → BitVec 32) : Option (RegionName → Nat → BitVec 32) := do
  let symbolic ← run NativeTerms.algebra NativeTerms.dotPrimitive (KernelExecution.symbolicCall c)
    work (NativeTerms.inputMemory input)
  return KernelExecution.project profile c scale symbolic

theorem native_eq (profile : NativeTerms.FP32Profile) (c : Real.Call) (scale : BitVec 32)
    (work : Work) (legal : Legal c work) (input : RegionName → Nat → BitVec 32) :
    runNative profile c scale work input=KernelExecution.runNative profile c scale input := by
  have hs : Legal (KernelExecution.symbolicCall c) work :=
    ⟨legal.forwardState,legal.forwardOutput,legal.backwardState,legal.backwardGradient⟩
  unfold runNative
  rw [run_eq NativeTerms.algebra NativeTerms.dotPrimitive (KernelExecution.symbolicCall c) work hs]
  rfl

#axiomsClean grid_permutation
#axiomsClean run_eq
#axiomsClean native_eq
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.KernelSchedule
