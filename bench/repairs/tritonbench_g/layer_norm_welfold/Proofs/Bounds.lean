import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.Host
namespace VeriTile.Bench.Repairs.LayerNormWelfold
open VeriTile.Triton FP.Structural Reuse
set_option maxHeartbeats 1000000

theorem rounded_width_upper (D B : Nat) (hB : 0<B) : blockCount D B*B<D+B := by
  have hd := Nat.div_add_mod (D+B-1) B
  rw [Nat.mul_comm B _] at hd
  unfold blockCount
  omega

/-- Exactly the offsets emitted by Python range(0,D,B), including a partial tail. -/
theorem range_offset_iff (D B k : Nat) (hB : 0<B) : k<blockCount D B ↔ k*B<D := by
  have hc := block_count_covers D B hB
  have hu := rounded_width_upper D B hB
  constructor
  · intro hk
    have he := Nat.mul_le_mul_right B hk
    nlinarith
  · intro hk
    by_contra hn
    have he := Nat.mul_le_mul_right B (Nat.le_of_not_gt hn)
    omega

/-- Native index representation for the saved i32 launch ABI. The typed Nat
semantics is unrestricted; this is the additional ordinary native bound. -/
def NativeIndexDomain (S D : Nat) : Prop := 0<D ∧ D+2048<2^31 ∧ S*D+2048<2^31

theorem native_indices_fit (S D B row k : Nat) (i : Fin B)
    (domain : NativeIndexDomain S D) (allowed : B=1024 ∨ B=2048)
    (hr : row<S) (hk : k<blockCount D B) :
    k*B+i.val<2^31 ∧ k*B+i.val+D*row<2^31 ∧ (k+1)*B<2^31 ∧ row<2^31 := by
  have hb : 0<B := by rcases allowed with h|h <;> omega
  have hub : B≤2048 := by rcases allowed with h|h <;> omega
  have hoff := (range_offset_iff D B k hb).mp hk
  have hrow : D*(row+1)≤S*D := by
    simpa [Nat.mul_comm] using Nat.mul_le_mul_right D hr
  have hnext : (k+1)*B≤blockCount D B*B := Nat.mul_le_mul_right B hk
  have hu := rounded_width_upper D B hb
  have hS : S≤S*D := by nlinarith [domain.1]
  rcases domain with ⟨hd,hw,hs⟩
  constructor
  · omega
  constructor
  · nlinarith [i.isLt]
  constructor
  · omega
  · omega

theorem active_load_store_bounds (S D row col : Nat) (hr : row<S) (hc : col<D) :
    col+D*row<S*D ∧ col<D ∧ row<S := ⟨output_bounds S D row col hr hc,hc,hr⟩

#axiomsClean range_offset_iff
#axiomsClean native_indices_fit
#axiomsClean active_load_store_bounds
end VeriTile.Bench.Repairs.LayerNormWelfold
