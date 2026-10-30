import bench.tritonbench_g.context_attn_mistral.NativeIO
import bench.tritonbench_g.context_attn_mistral.Proofs.NativeBindings

namespace VeriTile.Bench.Repairs.ContextAttnMistralKernel.NativeInterface
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
open VeriTile.Bench.Repairs.ContextAttnMistral
noncomputable section
variable {α : Type} [Inhabited α]

theorem invocation_initial (c : Host.Call) (o : FPBackend.Ops α) (idx : GridIndex c.grid) (s : State α) :
    (invocation c o idx).initial s=KernelReference.atIndex s idx := rfl

theorem schedule_eq (c : Host.Call) (o : FPBackend.Ops α) (jobs : List (GridIndex c.grid)) (s : State α) :
    runSchedule (jobs.map (invocation c o)) s=KernelReference.runGrid c o jobs s := by
  induction jobs generalizing s with
  | nil=>rfl
  | cons idx rest ih=>
    simp only [List.map_cons,runSchedule,KernelReference.runGrid,Invocation.run,invocation_initial]
    congr 1
    funext t
    exact ih t

theorem binding_snapshot (c : Host.Call) (o : FPBackend.Ops α) (jobs : List (GridIndex c.grid))
    (a : NativeArrays.Inputs c α) (s : State α) (bound : (programIO c o jobs).InputBinding a s) :
    NativeBindings.snapshot c s=a := by
  funext port off
  have h:=bound port off
  change (s.mem (NativeArrays.ports c)[port].buf off).read (NativeArrays.ports c)[port].dtype=
    mapValue id (NativeArrays.ports c)[port].dtype (a port off) at h
  change (s.mem (NativeArrays.ports c)[port].buf off).read (NativeArrays.ports c)[port].dtype=a port off
  have identity (d : TileDType) (v : Value α d) : mapValue id d v=v := by cases d <;> rfl
  simpa only [identity] using h

theorem snapshot_congr (c : Host.Call) (s t : State α) (same : s.mem=t.mem) :
    NativeBindings.snapshot c s=NativeBindings.snapshot c t := by
  unfold NativeBindings.snapshot
  rw [same]

#axiomsClean schedule_eq
#axiomsClean binding_snapshot
end
end VeriTile.Bench.Repairs.ContextAttnMistralKernel.NativeInterface
