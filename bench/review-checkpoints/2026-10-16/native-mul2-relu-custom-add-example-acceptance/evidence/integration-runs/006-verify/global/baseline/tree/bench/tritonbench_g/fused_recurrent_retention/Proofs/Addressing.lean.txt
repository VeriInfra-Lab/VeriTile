import bench.tritonbench_g.fused_recurrent_retention.Proofs.Model
import VeriTile.Meta.StatementAudit

/-! Address uniqueness uses active lanes, so partial feature tiles do not
require the entire tile to fit inside DK or DV. The pointer subtraction lemma
only concerns reached earlier rows; terminal unused pointers remain actual
source results and are not constrained by this lemma. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.Addressing
open VeriTile.Triton

private theorem radix_unique {b a c x y : Nat} (hx : x < b) (hy : y < b)
    (h : a*b+x = c*b+y) : a = c ∧ x = y := by
  have hb : 0 < b := by omega
  have hm := congrArg (fun n => n % b) h
  simp only [Nat.add_mod, Nat.mul_mod_left, Nat.zero_add,
    Nat.mod_eq_of_lt hx, Nat.mod_eq_of_lt hy] at hm
  subst y
  exact ⟨Nat.eq_of_mul_eq_mul_right hb (Nat.add_right_cancel h), rfl⟩

theorem output_unique (p : Layout) (l : Launch) (t u : Nat) (a b : Fin p.BV)
    (ha : activeValue p l a.val) (hb : activeValue p l b.val)
    (h : outputAddress p l t a.val = outputAddress p l u b.val) :
    t = u ∧ a = b := by
  have hc : t*p.DV+(l.value*p.BV+a.val) = u*p.DV+(l.value*p.BV+b.val) := by
    unfold outputAddress at h
    omega
  have hr := radix_unique ha hb hc
  exact ⟨hr.1, Fin.ext (Nat.add_left_cancel hr.2)⟩

theorem key_gradient_unique (p : Layout) (l : Launch) (t u : Nat) (a b : Fin p.BK)
    (ha : activeKey p l a.val) (hb : activeKey p l b.val)
    (h : keyGradientAddress p l t a.val = keyGradientAddress p l u b.val) :
    t = u ∧ a = b := by
  have hc : t*p.DK+(l.key*p.BK+a.val) = u*p.DK+(l.key*p.BK+b.val) := by
    unfold keyGradientAddress at h
    omega
  have hr := radix_unique ha hb hc
  exact ⟨hr.1, Fin.ext (Nat.add_left_cancel hr.2)⟩

theorem state_unique (p : Layout) (l : Launch) (a c : Fin p.BK) (b d : Fin p.BV)
    (hb : activeValue p l b.val) (hd : activeValue p l d.val)
    (h : stateAddress p l a.val b.val = stateAddress p l c.val d.val) :
    a = c ∧ b = d := by
  have hc : (l.key*p.BK+a.val)*p.DV+(l.value*p.BV+b.val) =
      (l.key*p.BK+c.val)*p.DV+(l.value*p.BV+d.val) := by
    unfold stateAddress at h
    omega
  have hr := radix_unique hb hd hc
  exact ⟨Fin.ext (Nat.add_left_cancel hr.1), Fin.ext (Nat.add_left_cancel hr.2)⟩

theorem subtract_stride_live (base stride t : Nat) :
    (base + (t + 1) * stride) - stride = base + t * stride := by
  rw [Nat.add_mul, Nat.one_mul]
  omega

#axiomsClean output_unique
#axiomsClean key_gradient_unique
#axiomsClean state_unique
#axiomsClean subtract_stride_live
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.Addressing
