import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.CandidateIO
import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.Proofs.PairSource
import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.NativeIO

namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.PairExecution
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.LightningAttention.Native (Word Instructions Plan)
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
noncomputable section
set_option maxHeartbeats 4000000
variable {α : Type} [Inhabited α]

theorem invocation_run (M : Algebra α) (D : DotPrimitive α) (stage : Stage)
    (d : Host.Dimensions) (pid : PID) (s : State α) :
    (invocation M D stage d pid).run s = (KernelOnly.invocation M D stage d pid).run s := by
  change FP.Structural.exec _ (PairSource.kernel stage d.parameters) _ =
    FP.Structural.exec _ (stageKernel stage d) _
  cases stage <;> exact PairSource.exact _ _ d.parameters _

theorem invocation_safe (M : Algebra α) (D : DotPrimitive α) (stage : Stage)
    (d : Host.Dimensions) (pid : PID) (bounds : RegionName → Nat) (s : State α) :
    (invocation M D stage d pid).Safe bounds s ↔
      (KernelOnly.invocation M D stage d pid).Safe bounds s := by
  change TraceSafe _ bounds (PairSource.kernel stage d.parameters) _ ↔
    TraceSafe _ bounds (stageKernel stage d) _
  cases stage <;> exact PairSource.safe_iff _ _ d.parameters _ bounds

theorem launch_run (ops : Instructions) (plan : Plan) (stage : Stage)
    (d : Host.Dimensions) (pid : PID) (memory : Memory Word) :
    (launch ops plan stage d pid).run memory = (NativeIO.launch ops plan stage d pid).run memory := by
  change ((invocation Native.algebra Native.dots stage d pid).run
    ((NativeIO.launch ops plan stage d pid).embedded memory)).map
    (fun final => (NativeIO.launch ops plan stage d pid).finish final.mem) = _
  rw [invocation_run]
  rfl

theorem launch_safe (ops : Instructions) (plan : Plan) (stage : Stage)
    (d : Host.Dimensions) (pid : PID) (bounds : RegionName → Nat) (memory : Memory Word) :
    (launch ops plan stage d pid).Safe bounds memory ↔
      (NativeIO.launch ops plan stage d pid).Safe bounds memory :=
  invocation_safe Native.algebra Native.dots stage d pid bounds _

theorem stage_run (ops : Instructions) (plan : Plan) (stage : Stage)
    (d : Host.Dimensions) (pids : List PID) (memory : Memory Word) :
    runMaterializedSchedule (pids.map (launch ops plan stage d)) memory =
      runMaterializedSchedule (pids.map (NativeIO.launch ops plan stage d)) memory := by
  induction pids generalizing memory with
  | nil => rfl
  | cons pid rest ih =>
    simp only [List.map_cons,runMaterializedSchedule,launch_run]
    cases he : (NativeIO.launch ops plan stage d pid).run memory with
    | none => rfl
    | some next => simpa only [Option.bind_some] using ih next

theorem stage_safe (ops : Instructions) (plan : Plan) (stage : Stage)
    (d : Host.Dimensions) (pids : List PID) (bounds : RegionName → Nat) (memory : Memory Word) :
    MaterializedScheduleSafe bounds (pids.map (launch ops plan stage d)) memory ↔
      MaterializedScheduleSafe bounds (pids.map (NativeIO.launch ops plan stage d)) memory := by
  induction pids generalizing memory with
  | nil => rfl
  | cons pid rest ih =>
    simp only [List.map_cons,MaterializedScheduleSafe,launch_run,launch_safe]
    exact and_congr Iff.rfl (forall_congr' fun next => imp_congr_right fun _ => ih next)

#axiomsClean invocation_run
#axiomsClean launch_run
#axiomsClean stage_run
#axiomsClean stage_safe
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.PairExecution
