import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.rmsnorm_fused.Kernels

namespace VeriTile.Bench.Repairs.RmsnormFused
open VeriTile.Triton

theorem dimensions_product (s : Shape) : s.dims.prod=s.rows*s.width := by
  simp [Shape.dims,Shape.rows,List.prod_append]

theorem block_positive (s : Shape) (v : s.Valid) : 0<s.block := by
  exact lt_of_lt_of_le v.1 v.2.1

theorem offset_bounded (s : Shape) (r k : Nat) (hr : r<s.rows) (hk : k<s.width) :
    s.offset r k < s.rows*s.width := by
  unfold Shape.offset
  calc
    r*s.width+k < r*s.width+s.width := Nat.add_lt_add_left hk _
    _ = (r+1)*s.width := by rw [Nat.add_mul,Nat.one_mul]
    _ ≤ s.rows*s.width := Nat.mul_le_mul_right _ (by omega)

theorem offset_coordinates (s : Shape) (v : s.Valid) (r k : Nat) (hk : k<s.width) :
    s.offset r k/s.width=r ∧ s.offset r k%s.width=k := by
  constructor
  · rw [Shape.offset,Nat.add_comm,Nat.add_mul_div_right _ _ v.1,Nat.div_eq_of_lt hk]
    simp
  · simp [Shape.offset,Nat.add_mod,Nat.mod_eq_of_lt hk]

theorem offset_injective (s : Shape) (v : s.Valid) (r q k j : Nat)
    (hk : k<s.width) (hj : j<s.width) (eq : s.offset r k=s.offset q j) : r=q ∧ k=j := by
  have a := offset_coordinates s v r k hk
  have b := offset_coordinates s v q j hj
  constructor
  · rw [←a.1,←b.1,eq]
  · rw [←a.2,←b.2,eq]

theorem byte_offset_bound (s : Shape) (v : s.Valid) (r k i : Nat)
    (hr : r<s.rows) (hk : k<s.width) (hi : i<4) : 4*s.offset r k+i<2^63 := by
  have h := offset_bounded s r k hr hk
  have b := v.2.2.2
  omega

theorem grid_complete (s : Shape) (r : Nat) : r∈s.grid ↔ r<s.rows := by
  simp [Shape.grid]

theorem grid_unique (s : Shape) : s.grid.Nodup := List.nodup_range

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.RmsnormFused
