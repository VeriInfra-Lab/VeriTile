import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.KernelRewrite
import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.FPGrid

namespace VeriTile.Bench.Repairs.BmmChunkFwd.KernelCandidate
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
noncomputable section

theorem execution {α : Type} [Inhabited α] (algebra : Algebra α)
    (h : Host.Call) (c : Host.Config) (ha : h.aStorage=.fp32) (hb : h.bStorage=.fp32)
    (s : State α) :
    FP.Structural.exec (engine algebra) (Kernels.source h c) s=
      FP.Structural.exec (engine algebra) (Kernels.indexCommuted h c) s := by
  rw [←Kernels.fp32Source_eq h c ha hb]
  exact KernelRewrite.execution algebra h c s

/-- The same integer commutation is exact in the actual modular i64 source
arithmetic, without a no-overflow premise. All other finite source operations
and their admission-derived bounds are shared. -/
theorem native_head (pid groups : BitVec 64) :
    pid-pid.sdiv groups*groups=pid-groups*pid.sdiv groups := by
  rw [BitVec.mul_comm (pid.sdiv groups) groups]

def schedule {α : Type} [Inhabited α] (o : FPSuccess.Ops α) (h : Host.Call) (c : Host.Config) :
    List (GridIndex (h.grid c)) → State α → Option (State α)
  | [],s => some s
  | idx::rest,s => do
    let next ← FP.Structural.exec (engine o.algebra) (Kernels.indexCommuted h c) (FPGrid.atIndex s idx)
    schedule o h c rest next

theorem schedule_eq {α : Type} [Inhabited α] (o : FPSuccess.Ops α) (h : Host.Call)
    (c : Host.Config) (ha : h.aStorage=.fp32) (hb : h.bStorage=.fp32)
    (order : List (GridIndex (h.grid c))) (s : State α) :
    schedule o h c order s=FPGrid.schedule o h c order s := by
  induction order generalizing s with
  | nil => rfl
  | cons p ps ih =>
    simp only [schedule,FPGrid.schedule,←execution o.algebra h c ha hb]
    congr 1
    funext next
    exact ih next

#axiomsClean execution
#axiomsClean native_head
#axiomsClean schedule_eq
end
end VeriTile.Bench.Repairs.BmmChunkFwd.KernelCandidate
