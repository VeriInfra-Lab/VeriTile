import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.Geometry
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.SgmvExpandSlice
namespace Integer

def i64 (n : Nat) : BitVec 64 := BitVec.ofNat 64 n

theorem i64_value {n : Nat} (hn : n < 2^63) : (i64 n).toInt = (n : Int) := by
  unfold i64
  rw [←BitVec.ofInt_natCast]
  apply BitVec.toInt_ofInt_eq_self (by decide)
  · norm_num
  · exact_mod_cast hn

theorem add_exact {a b : Nat} (h : a+b < 2^63) :
    (i64 a+i64 b).toInt = (a+b : Nat) := by
  simpa only [i64,←BitVec.ofNat_add] using i64_value h

theorem mul_exact {a b : Nat} (h : a*b < 2^63) :
    (i64 a*i64 b).toInt = (a*b : Nat) := by
  simpa only [i64,←BitVec.ofNat_mul] using i64_value h

/-- The actual matrix pointer's products and additions retain their natural
offset in signed 64-bit arithmetic, under ordinary allocated-view validity. -/
theorem matrix_pointer_exact (v : MatrixView) (r c : Nat)
    (hr : r < v.rows) (hc : c < v.columns) :
    (i64 v.offset+i64 r*i64 v.rowStride+i64 c*i64 v.columnStride).toInt =
      (v.offset+r*v.rowStride+c*v.columnStride : Nat) := by
  simpa only [i64,←BitVec.ofNat_mul,←BitVec.ofNat_add] using
    i64_value (matrix_address_i64 v r c hr hc)

theorem matrix_products_exact (v : MatrixView) (r c : Nat)
    (hr : r < v.rows) (hc : c < v.columns) :
    (i64 r*i64 v.rowStride).toInt = (r*v.rowStride : Nat) ∧
      (i64 c*i64 v.columnStride).toInt = (c*v.columnStride : Nat) ∧
      (i64 v.offset+i64 r*i64 v.rowStride).toInt = (v.offset+r*v.rowStride : Nat) := by
  obtain ⟨ha,hb,hc⟩ := matrix_intermediates_i64 v r c hr hc
  exact ⟨mul_exact ha,mul_exact hb,by
    simpa only [i64,←BitVec.ofNat_mul,←BitVec.ofNat_add] using i64_value hc⟩

theorem weight_pointer_exact (v : WeightView) (a c k : Nat)
    (ha : a < v.adapters) (hc : c < v.columns) (hk : k < v.inner) :
    (i64 v.offset+i64 a*i64 v.adapterStride+i64 k*i64 v.innerStride+
      i64 c*i64 v.columnStride).toInt =
      (v.offset+a*v.adapterStride+k*v.innerStride+c*v.columnStride : Nat) := by
  simpa only [i64,←BitVec.ofNat_mul,←BitVec.ofNat_add] using
    i64_value (weight_address_i64 v a c k ha hc hk)

theorem byte_address_exact (a : Allocation) {i byte : Nat}
    (hi : i < a.elements) (hb : byte < a.bytesPerElement) :
    (i64 a.base+i64 i*i64 a.bytesPerElement+i64 byte).toNat =
      a.base+i*a.bytesPerElement+byte := by
  simp only [i64,←BitVec.ofNat_mul,←BitVec.ofNat_add,BitVec.toNat_ofNat]
  exact Nat.mod_eq_of_lt (allocation_byte_address a hi hb)

theorem tile_count_le {n : Nat} (hn : 0 < n) : tiles n ≤ n := by
  unfold tiles
  omega

/-- Tile product size is bounded by the actual output allocation. This does
not assert that a CUDA launch accepts that count or delivers those PIDs. -/
theorem grid_size_bound (p : Shape) (s : Segment) (fits : SegmentFits p s)
    (active : s.active) (hn : 0 < p.columns)
    (capacity : p.slice+p.columns ≤ p.outputColumns) :
    tiles s.length * tiles p.columns ≤ p.outputRows*p.outputColumns := by
  have hm := fits.output active
  have h1 := tile_count_le active.1
  have h2 := tile_count_le hn
  apply Nat.mul_le_mul
  · omega
  · omega

theorem accepted_pid_i64 (p : Shape) (s : Segment) (fits : SegmentFits p s)
    (active : s.active) (hn : 0 < p.columns)
    (capacity : p.slice+p.columns ≤ p.outputColumns)
    (allocated : p.outputRows*p.outputColumns < 2^62) (pid : Nat)
    (hpid : pid < tiles s.length * tiles p.columns) :
    (i64 pid).toInt = (pid : Int) := by
  apply i64_value
  have hg := grid_size_bound p s fits active hn capacity
  norm_num at allocated ⊢
  omega

#axiomsClean matrix_pointer_exact
#axiomsClean matrix_products_exact
#axiomsClean weight_pointer_exact
#axiomsClean byte_address_exact
#axiomsClean accepted_pid_i64
end Integer
end VeriTile.Bench.Repairs.SgmvExpandSlice
