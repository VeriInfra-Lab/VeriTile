import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.Native
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.Geometry

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.NativeGrid
open VeriTile.Triton Native Geometry

theorem signed32 (n : Nat) (hn : n<2147483648) : (BitVec.ofNat 32 n).toInt=(n:Int) := by
  have hnat : (BitVec.ofNat 32 n).toNat=n := by
    simp only [BitVec.toNat_ofNat]
    apply Nat.mod_eq_of_lt
    omega
  rw [BitVec.toInt_eq_toNat_of_lt,hnat]
  rw [hnat]
  norm_num
  omega

theorem column_mask (h : Call) (j : Fin h.block) :
    (BitVec.ofNat 32 j.val).slt (BitVec.ofNat 32 h.cols)=decide (j.val<h.cols) := by
  rw [BitVec.slt_eq_decide,signed32 _ (lanes_fit h j),signed32 _ (columns_fit h)]
  simp

theorem capacity (h : Call) (a : Allocation h.cells) : h.cells*4<Shared.signedLimit := by
  have cap := a.buffer.allocation
  rw [a.fp32Bytes] at cap
  exact (Nat.mul_le_mul_right 4 a.holds).trans_lt cap

/-- Includes masked lanes: the intermediate element offset still fits int64.
Only active lanes are subsequently asserted to be allocated or dereferenced. -/
theorem masked_offset (h : Call) (a : Allocation h.cells) (row : Fin h.rows) (col : Nat)
    (hc : col<65536) : row.val*h.cols+col<Shared.signedLimit := by
  have base := offset_bound h row.val 0 row.isLt h.positive
  have cap := capacity h a
  dsimp only [offset,Shared.signedLimit,VeriTile.Bench.Repairs.ChunkCumsumVector.NativeIndex.signedLimit] at *
  omega

theorem row_words (h : Call) (a : Allocation h.cells) (row : Fin h.rows) (j : Fin h.block)
    (pid32 : row.val<2147483648) :
    ((BitVec.ofInt 64 (BitVec.ofNat 32 row.val).toInt)*BitVec.ofNat 64 h.cols+
      BitVec.ofNat 64 j.val).toInt=(row.val*h.cols+j.val:Int) := by
  rw [Shared.widen32 _ pid32,←BitVec.ofNat_mul,←BitVec.ofNat_add]
  apply Shared.signed64
  exact masked_offset h a row j.val (j.isLt.trans_le h.blockLimit)

theorem weight_columns (h : Call) (p : Fin (featureCount h)) (j : Fin 256) :
    (BitVec.ofInt 64 (BitVec.ofNat 32 p.val).toInt*BitVec.ofNat 64 256+BitVec.ofNat 64 j.val)=
      BitVec.ofNat 64 (column p.val j) := by
  have small : p.val<2147483648 := p.isLt.trans_le (by have := (feature_limit h).2; omega)
  rw [Shared.widen32 _ small,←BitVec.ofNat_mul,←BitVec.ofNat_add]
  rfl

theorem weight_mask (h : Call) (p : Fin (featureCount h)) (j : Fin 256) :
    (BitVec.ofNat 64 (column p.val j)).slt (BitVec.ofNat 64 h.cols)=decide (column p.val j<h.cols) := by
  have cp : column p.val j<Shared.signedLimit := (column_limit h p j).trans_le (by decide)
  have cn : h.cols<Shared.signedLimit := (columns_fit h).trans_le (by decide)
  rw [BitVec.slt_eq_decide,Shared.signed64 _ cp,Shared.signed64 _ cn]
  simp

theorem weight_row_words (h : Call) (a : Allocation h.cells) (row : Fin h.rows)
    (p : Fin (featureCount h)) (j : Fin 256) :
    (BitVec.ofNat 64 row.val*BitVec.ofNat 64 h.cols+BitVec.ofNat 64 (column p.val j)).toInt=
      (row.val*h.cols+column p.val j:Int) := by
  rw [←BitVec.ofNat_mul,←BitVec.ofNat_add]
  exact Shared.signed64 _ (masked_offset h a row _ (column_limit h p j))

theorem loop_words (h : Call) (a : Allocation h.cells) (row : Fin h.rows) :
    (BitVec.ofNat 64 h.rows).toInt=(h.rows:Int) ∧
    (BitVec.ofNat 64 row.val).slt (BitVec.ofNat 64 h.rows)=Bool.true ∧
    (BitVec.ofNat 64 row.val+BitVec.ofNat 64 1).toInt=(row.val+1:Int) := by
  have cap:=row_bound h a
  refine ⟨Shared.signed64 _ cap,?_,?_⟩
  · rw [BitVec.slt_eq_decide,Shared.signed64 _ (row.isLt.trans cap),Shared.signed64 _ cap]
    simp [row.isLt]
  · rw [←BitVec.ofNat_add,Shared.signed64 _ ((Nat.succ_le_of_lt row.isLt).trans_lt cap)]
    simp

theorem physical_bytes (cells : Nat) (a : Allocation cells) (off : Nat) (inside : off<cells) :
    (BitVec.ofNat 64 a.buffer.base+BitVec.ofNat 64 off*BitVec.ofNat 64 4).toNat=a.buffer.base+off*4 ∧
      a.buffer.base+off*4+4≤18446744073709551616 := by
  have bound := inside.trans_le a.holds
  simpa [a.fp32Bytes] using And.intro (Shared.pointer_word a.buffer off bound) (Shared.byte_span a.buffer off bound)

#axiomsClean row_words
#axiomsClean weight_row_words
#axiomsClean loop_words
#axiomsClean physical_bytes
end VeriTile.Bench.Repairs.FastRmsLayernorm.NativeGrid
