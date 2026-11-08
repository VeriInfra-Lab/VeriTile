import bench.tritonbench_g.attn_fwd_causal.Proofs.AccessSafety
import VeriTile.Meta.StatementAudit

/-! The actual repaired D128 layout, with a partial second query block and
three key-scale blocks. These bounds belong to the current source trace. -/
namespace RealAttnFwdCausalIOTests
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.AttnFwdCausal.Source
noncomputable section

private def parameters : Parameters:=⟨"Q","K","V","QS","KS","Out",1,1,130⟩
private def allocated (scales : Nat) : RegionName → Nat :=
  fun r=>if r="KS" then scales else if r="QS" then 2 else 16640

theorem partial_last_query_bounds {α : Type} [Inhabited α] (s : State α)
    (queryBlock : s.pids 0=1) (head : s.pids 1=0) :
    AccessSafety.Bounds parameters s (allocated 3) := by
  constructor
  · intro i j active
    simp [parameters,allocated,base,queryBlock,head,Parameters.q] at active ⊢
    omega
  · simp [parameters,allocated,queryBlock,head]
  · intro row active j
    simp [parameters,allocated,base,head,Parameters.q,Parameters.k] at active ⊢
    omega
  · intro row active j
    simp [parameters,allocated,base,head,Parameters.q,Parameters.v] at active ⊢
    omega
  · intro row active
    simp [parameters,allocated,head] at active ⊢
    omega
  · intro i j active
    simp [parameters,allocated,base,queryBlock,head,Parameters.q,Parameters.o] at active ⊢
    omega

/-- The final partial key block still performs its unmasked scale load. -/
theorem missing_last_key_scale {α : Type} [Inhabited α] (s : State α)
    (head : s.pids 1=0) : ¬ AccessSafety.Bounds parameters s (allocated 2) := by
  intro bounds
  have last:=bounds.keyScale 128 (by decide)
  norm_num [parameters,allocated,head] at last

#axiomsClean partial_last_query_bounds
#axiomsClean missing_last_key_scale
#guard_msgs (drop info) in
#auditModuleAxioms
end
end RealAttnFwdCausalIOTests
