import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.Geometry
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.SourceArithmetic
set_option linter.unusedSimpArgs false

theorem inner_lt (block k inner : Nat) :
    ((block : Int)*16 + (k : Int) < (inner : Int)) ↔ block*16+k < inner := by omega

theorem input_offset (start row rowStride block k inner innerStride : Nat) :
    ((start : Int)*rowStride + (row : Int)*rowStride +
      if (block : Int)*16 + k < inner then ((block : Int)*16+k)*innerStride else 0).toNat =
    start*rowStride + row*rowStride + (if block*16+k < inner then block*16+k else 0)*innerStride := by
  by_cases h : block*16+k < inner
  · have hi := (inner_lt block k inner).2 h
    rw [if_pos hi,if_pos h]
    norm_cast
  · have hi : ¬ (block : Int)*16+k < inner := by rw [inner_lt]; exact h
    rw [if_neg hi,if_neg h]
    norm_cast
    simp only [Nat.zero_mul,Nat.add_zero,Int.toNat_natCast]

theorem weight_offset (adapterStride adapter block k inner innerStride column columnStride : Nat) :
    (((adapterStride : Int)*adapter +
      if (block : Int)*16+k < inner then ((block : Int)*16+k)*innerStride else 0) +
      (column : Int)*columnStride).toNat =
    (adapterStride*adapter + (if block*16+k < inner then block*16+k else 0)*innerStride) +
      column*columnStride := by
  by_cases h : block*16+k < inner
  · have hi := (inner_lt block k inner).2 h
    rw [if_pos hi,if_pos h]
    norm_cast
  · have hi : ¬ (block : Int)*16+k < inner := by rw [inner_lt]; exact h
    rw [if_neg hi,if_neg h]
    norm_cast
    simp only [Nat.zero_mul,Nat.add_zero,Int.toNat_natCast]

#axiomsClean inner_lt
#axiomsClean input_offset
#axiomsClean weight_offset
end VeriTile.Bench.Repairs.SgmvExpandSlice.SourceArithmetic
