import bench.tritonbench_g.apply_penalty.Proofs.Rewrite

namespace VeriTile.Tests.ApplyPenaltySignedMetadata
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ApplyPenalty.Rewrite
noncomputable section

/-- A signed count, including negative words, stays an integer until the
source multiplication converts it under FP32. The test does not reinterpret
an int cell through the natural-valued memory reader. -/
def metadata {α : Type} (s : State α) (count : Int) : State α :=
  {s with
    mem := fun r o => if r="cumsum" then .mk .int (if o=1 then 0 else 1)
      else if r="ids" then .mk .int 0
      else if r="counts" then .mk .int count
      else s.mem r o
    pids := fun ax => if ax=0 then 1 else s.pids ax}

theorem signed_count_register {α : Type} [Inhabited α] (M : Algebra α)
    (lt : α → α → Bool) (hlt : M.compareLt (some .fp32) .real=some lt)
    (s : State α) (count : Int) :
    (originalExecution M lt hlt 4 1 (metadata s count)).val.regs .int [1] "batch_ids_count" =
      some (fun _ => count) := by
  simp [originalExecution,metadata,State.setReg,Cell.read]

#axiomsClean signed_count_register
end
end VeriTile.Tests.ApplyPenaltySignedMetadata
