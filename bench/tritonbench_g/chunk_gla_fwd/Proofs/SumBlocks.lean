import bench.tritonbench_g.chunk_gla_fwd.Proofs.Geometry
import VeriTile.Triton
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.SumBlocks
open scoped BigOperators
noncomputable section

theorem partition (f : Nat → ℝ) (n w : Nat) :
    (∑ b ∈ Finset.range n, ∑ e : Fin w, f (b*w+e.val)) =
      ∑ k ∈ Finset.range (n*w), f k := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [Finset.sum_range_succ, ih, Nat.succ_mul, Finset.sum_range_add]
    congr 1
    exact Fin.sum_univ_eq_sum_range (fun e => f (n*w+e)) w

theorem cover (k w : Nat) (hw : 0 < w) :
    k ≤ Geometry.ceilDiv k w * w := by
  by_cases hk : k=0
  · simp [hk]
  · have hown := Geometry.owner_lt (n := k) (width := w) (i := k-1) hw (by omega)
    have hmul := Nat.mul_le_mul_right w (Nat.succ_le_of_lt hown)
    have hmod := Nat.mod_lt (k-1) hw
    have hdiv := Nat.div_add_mod (k-1) w
    have hpred : k-1+1=k := by omega
    nlinarith

theorem truncated (f : Nat → ℝ) (k w : Nat) (hw : 0 < w)
    (hz : ∀ j, k ≤ j → f j=0) :
    (∑ b ∈ Finset.range (Geometry.ceilDiv k w), ∑ e : Fin w, f (b*w+e.val)) =
      ∑ j ∈ Finset.range k, f j := by
  rw [partition]
  symm
  apply Finset.sum_subset (Finset.range_mono (cover k w hw))
  intro j _ hj
  apply hz
  simpa only [Finset.mem_range,not_lt] using hj

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.SumBlocks
