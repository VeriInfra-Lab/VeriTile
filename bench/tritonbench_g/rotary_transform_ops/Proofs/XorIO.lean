import bench.tritonbench_g.rotary_transform_ops.Proofs.XorSafety

namespace VeriTile.Bench.Repairs.RotaryTransformOps.XorPair
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]

theorem job_run (ops : Ops α) (s : Shape) (cu64 off64 : Bool) (pid : Program) (u : State α) :
    (job ops s cu64 off64 pid).run u = (Interface.job ops s cu64 off64 pid).run u :=
  execution_eq (KernelAlgebra.backend ops) s cu64 off64 ((Interface.job ops s cu64 off64 pid).initial u)

theorem job_safe (ops : Ops α) (s : Shape) (cu64 off64 : Bool) (pid : Program)
    (bounds : RegionName → Nat) (u : State α) :
    (job ops s cu64 off64 pid).Safe bounds u ↔ (Interface.job ops s cu64 off64 pid).Safe bounds u :=
  safety_iff (KernelAlgebra.backend ops) s cu64 off64 bounds ((Interface.job ops s cu64 off64 pid).initial u)

theorem jobs_run (ops : Ops α) (s : Shape) (cu64 off64 : Bool) (order : List Program) (u : State α) :
    runSchedule (order.map (job ops s cu64 off64)) u =
      runSchedule (order.map (Interface.job ops s cu64 off64)) u := by
  induction order generalizing u with
  | nil => rfl
  | cons pid rest ih =>
    simp only [List.map_cons,runSchedule,job_run]
    cases (Interface.job ops s cu64 off64 pid).run u <;> simp [ih]

theorem jobs_safe (ops : Ops α) (s : Shape) (cu64 off64 : Bool) (order : List Program)
    (bounds : RegionName → Nat) (u : State α) :
    ScheduleSafe bounds (order.map (job ops s cu64 off64)) u ↔
      ScheduleSafe bounds (order.map (Interface.job ops s cu64 off64)) u := by
  induction order generalizing u with
  | nil => rfl
  | cons pid rest ih => simp only [List.map_cons,ScheduleSafe,job_safe,job_run,ih]

theorem schedule_run (ops : Ops α) (c : Config) (cu64 off64 : Bool) (u : State α) :
    runSchedule (schedule ops c cu64 off64) u = runSchedule (Interface.schedule ops c cu64 off64) u := by
  unfold schedule Interface.schedule
  split
  · rfl
  · exact jobs_run ops c.shape cu64 off64 (programs c) u

theorem schedule_safe (ops : Ops α) (c : Config) (cu64 off64 : Bool)
    (bounds : RegionName → Nat) (u : State α) :
    ScheduleSafe bounds (schedule ops c cu64 off64) u ↔
      ScheduleSafe bounds (Interface.schedule ops c cu64 off64) u := by
  unfold schedule Interface.schedule
  split
  · rfl
  · exact jobs_safe ops c.shape cu64 off64 (programs c) bounds u

local instance : Inhabited Native.Node := ⟨Native.atom Native.zero⟩

/-- Complete original IO domain, copied suffix and in-place case retained.
Safety and termination are conclusions of the original shared judgment and
proved source-to-source transfer, never extra input or execution assumptions. -/
theorem equivalent (c : Config) (valid : c.Valid) (p : Native.Instructions)
    (cu64 off64 : Bool) :
    (Interface.io (Native.ops p) c cu64 off64 Interface.observeNative).EquivalentOn
      (variant (Native.ops p) c cu64 off64 Interface.observeNative) (Interface.Ready c) := by
  intro vals ready bounds hb initial hin
  obtain ⟨safe,final,ran,outputs,frame⟩ :=
    IOComplete.native_implements c valid p cu64 off64 vals ready bounds hb initial hin
  refine ⟨safe,?_,final,final,ran,?_,?_,frame,frame⟩
  · exact (schedule_safe (Native.ops p) c cu64 off64 bounds initial).mpr safe
  · exact (schedule_run (Native.ops p) c cu64 off64 initial).trans ran
  · intro o j active
    rfl

#axiomsClean job_run
#axiomsClean job_safe
#axiomsClean schedule_run
#axiomsClean schedule_safe
#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.RotaryTransformOps.XorPair
