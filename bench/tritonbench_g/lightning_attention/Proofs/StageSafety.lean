import bench.tritonbench_g.lightning_attention.Proofs.ForwardSafety
import bench.tritonbench_g.lightning_attention.Proofs.IntraSafety
import bench.tritonbench_g.lightning_attention.Proofs.InterScanSafety
import bench.tritonbench_g.lightning_attention.IO
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.SourceSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
noncomputable section
variable {α : Type} [Inhabited α]

def StageBounds (stage : Stage) (d : Host.Dimensions) (pid : PID) (bounds : RegionName → Nat) : Prop :=
  match stage with
  | .forward => ForwardBounds d.parameters pid bounds
  | .intra | .inter => HeadBounds d.parameters pid.head bounds

theorem invocation_safe (M : Algebra α) (hm : MaskArithmetic M) (D : DotPrimitive α)
    (stage : Stage) (d : Host.Dimensions) (pid : PID) (bounds : RegionName → Nat)
    (backed : StageBounds stage d pid bounds) (s : State α) :
    (invocation M D stage d pid).Safe bounds s := by
  let start := (invocation M D stage d pid).initial s
  cases stage with
  | forward => exact program M D d.parameters bounds start backed
  | intra => exact intra M hm D d.parameters bounds start backed
  | inter => exact inter M D d.parameters rfl start bounds backed

theorem schedule_safe (M : Algebra α) (hm : MaskArithmetic M) (D : DotPrimitive α)
    (stage : Stage) (d : Host.Dimensions) (pids : List PID) (bounds : RegionName → Nat)
    (backed : ∀ pid ∈ pids, StageBounds stage d pid bounds) (s : State α) :
    ScheduleSafe bounds (pids.map (invocation M D stage d)) s := by
  induction pids generalizing s with
  | nil => trivial
  | cons pid rest ih =>
    refine ⟨invocation_safe M hm D stage d pid bounds (backed pid (by simp)) s,?_⟩
    intro t ran
    exact ih (fun p hp => backed p (by simp [hp])) t
#axiomsClean invocation_safe
#axiomsClean schedule_safe
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.SourceSafety
