import bench.tritonbench_g.fused_recurrent_retention.Proofs.IOBackwardBounds

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.PublicData
open VeriTile.Triton FP.Structural Quoted
noncomputable section
set_option maxHeartbeats 1600000
variable {α : Type} [Inhabited α]

@[simp] theorem job_initial (mode : Mode) (a : Arithmetic α) (s : Shape) (p : PlanBinding s)
    (incoming finalGradient : Fin 4 → Nat) (seed final : Bool) (pid : ProgramId)
    (u : State (Staged.Node α)) :
    (job mode a s p incoming finalGradient seed final pid).initial u=prepare a s pid u := rfl

theorem schedule_run (mode : Mode) (a : Arithmetic α) (s : Shape) (p : PlanBinding s)
    (incoming finalGradient : Fin 4 → Nat) (seed final : Bool)
    (pids : List ProgramId) (u : State (Staged.Node α)) :
    runSchedule (pids.map (job mode a s p incoming finalGradient seed final)) u=
      gridExec mode a s (KernelInterface.buffers s incoming finalGradient) p.plans seed final pids u := by
  induction pids generalizing u with
  | nil => rfl
  | cons pid rest ih =>
    simp only [List.map_cons,runSchedule,Invocation.run,job_initial,gridExec]
    change (exec _ _ _).bind _=(exec _ _ _).bind _
    congr 1
    funext v
    exact ih v

theorem schedule_safe (mode : Mode) (a : Arithmetic α) (s : Shape) (shape : s.Admitted)
    (p : PlanBinding s) (incoming finalGradient : Fin 4 → Nat) (seed final : Bool)
    (xs : (io mode a s p incoming finalGradient seed final).Inputs)
    (bounds : RegionName → Nat) (hb : (io mode a s p incoming finalGradient seed final).InBounds bounds xs)
    (pids : List ProgramId) (hpids : ∀ pid,pid∈pids → pid.Valid s) (u : State (Staged.Node α)) :
    ScheduleSafe bounds (pids.map (job mode a s p incoming finalGradient seed final)) u := by
  induction pids generalizing u with
  | nil => trivial
  | cons pid rest ih =>
    constructor
    · change TraceSafe (Staged.algebra a p.plans) bounds
        (Quoted.kernel mode s (KernelInterface.buffers s incoming finalGradient) seed final) (prepare a s pid u)
      have hs : (prepare a s pid u).regs .real [] "scale"=
          some (fun _ => Staged.tagged .scale (a.scale s.DK)) := by simp [prepare]
      have hp := hpids pid (by simp)
      cases mode
      · apply SourceSafety.forward_safe a s _ p seed final _ shape hs bounds
        rw [prepare_launch]
        exact forward_bounds a s p incoming finalGradient seed final xs bounds hb pid hp
      · apply SourceSafety.backward_safe a s _ p seed final _ shape hs bounds
        rw [prepare_launch]
        exact backward_bounds a s p incoming finalGradient seed final xs bounds hb pid hp
    · intro v _
      exact ih (fun pid h => hpids pid (by simp [h])) v

#axiomsClean schedule_safe
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.PublicData
