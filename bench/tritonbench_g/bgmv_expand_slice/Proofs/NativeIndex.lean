import bench.tritonbench_g.bgmv_expand_slice.Proofs.Storage
import Mathlib.Data.BitVec

namespace VeriTile.Bench.Repairs.BgmvExpandSlice.NativeIndex
open Storage

/-- Bounds describe actually emitted program IDs, not a new tensor admission
filter. Global grid admission/segmentation is a separate runtime obligation. -/
structure Program (h : Call) extends Geometry.Program h where
  batchSigned32 : batch<2147483648
  rowSigned32 : row<2147483648

theorem signed64 (n : Nat) (hn : n<signedLimit) : (BitVec.ofNat 64 n).toInt=(n:Int) := by
  have hnat : (BitVec.ofNat 64 n).toNat=n := by
    simp only [BitVec.toNat_ofNat]
    apply Nat.mod_eq_of_lt
    dsimp only [signedLimit] at hn
    omega
  rw [BitVec.toInt_eq_toNat_of_lt,hnat]
  rw [hnat]
  dsimp only [signedLimit] at hn
  norm_num
  omega

theorem widen32 (n : Nat) (hn : n<2147483648) :
    BitVec.ofInt 64 (BitVec.ofNat 32 n).toInt=BitVec.ofNat 64 n := by
  have hnat : (BitVec.ofNat 32 n).toNat=n := by
    simp only [BitVec.toNat_ofNat]
    apply Nat.mod_eq_of_lt
    omega
  have hint : (BitVec.ofNat 32 n).toInt=(n:Int) := by
    rw [BitVec.toInt_eq_toNat_of_lt,hnat]
    rw [hnat]
    norm_num
    omega
  rw [hint,BitVec.ofInt_natCast]

theorem extent_bound (b : Buffer) : b.extent<signedLimit :=
  lt_of_le_of_lt (Nat.le_mul_of_pos_right _ b.bytesPositive) b.signedSize

theorem pointer_word (b : Buffer) (off : Nat) (inside : off<b.extent) :
    (BitVec.ofNat 64 b.base+BitVec.ofNat 64 off*BitVec.ofNat 64 b.bytes).toNat=b.address off := by
  rw [←BitVec.ofNat_mul,←BitVec.ofNat_add,BitVec.toNat_ofNat]
  apply Nat.mod_eq_of_lt
  have span := byte_span b off inside
  have pos := b.bytesPositive
  unfold Buffer.address at *
  norm_num
  omega

def inputWord (h : Call) (p : Program h) (k : Nat) : BitVec 64 :=
  BitVec.ofInt 64 (BitVec.ofNat 32 p.batch).toInt*BitVec.ofNat 64 h.BK+
    BitVec.ofNat 64 k

def weightWord (h : Call) (p : Program h) (slot n k : Nat) : BitVec 64 :=
  BitVec.ofNat 64 ((h.N+255)*h.BK)*BitVec.ofNat 64 slot+
    BitVec.ofInt 64 (BitVec.ofNat 32 p.row).toInt*BitVec.ofNat 64 1*BitVec.ofNat 64 h.BK+
    BitVec.ofNat 64 n*BitVec.ofNat 64 h.BK+BitVec.ofNat 64 k

def outputWord (h : Call) (p : Program h) (n : Nat) : BitVec 64 :=
  BitVec.ofInt 64 (BitVec.ofNat 32 p.batch).toInt*BitVec.ofNat 64 (h.columns+255)+
    BitVec.ofInt 64 (BitVec.ofNat 32 p.row).toInt*BitVec.ofNat 64 1+
    BitVec.ofNat 64 h.slice*BitVec.ofNat 64 1+BitVec.ofNat 64 n

theorem word_offsets (h : Call) (p : Program h) (slot n k : Nat) :
    inputWord h p k=BitVec.ofNat 64 (Geometry.inputOffset h p.batch k) ∧
    weightWord h p slot n k=BitVec.ofNat 64 (Geometry.weightOffset h slot p.row n k) ∧
    outputWord h p n=BitVec.ofNat 64 (Geometry.outputOffset h p.batch p.row n) := by
  simp only [inputWord,weightWord,outputWord,widen32 _ p.batchSigned32,widen32 _ p.rowSigned32]
  simp only [Geometry.inputOffset,Geometry.weightOffset,Geometry.outputOffset,
    BitVec.ofNat_add,BitVec.ofNat_mul,Nat.add_mul,Nat.mul_one,BitVec.mul_one]
  simp only [BitVec.add_assoc,BitVec.mul_comm,true_and,and_true]

/-- All generated lanes, including masked-off lanes, stay inside the same
padded allocations consumed by source execution. Native offsets are signed64
exact and byte addresses are the actual storage base plus scaled offset. -/
theorem source_addresses (h : Call) (storage : Placement h) (p : Program h)
    (slot : Nat) (selected : slot<h.adapters) (n : Fin 256) (k : Fin h.BK) :
    (inputWord h p k.val).toInt=(Geometry.inputOffset h p.batch k.val:Int) ∧
    (weightWord h p slot n.val k.val).toInt=(Geometry.weightOffset h slot p.row n.val k.val:Int) ∧
    (outputWord h p n.val).toInt=(Geometry.outputOffset h p.batch p.row n.val:Int) ∧
    (BitVec.ofNat 64 storage.input.base+inputWord h p k.val*BitVec.ofNat 64 storage.input.bytes).toNat=
      storage.input.address (Geometry.inputOffset h p.batch k.val) ∧
    (BitVec.ofNat 64 storage.weights.base+weightWord h p slot n.val k.val*BitVec.ofNat 64 storage.weights.bytes).toNat=
      storage.weights.address (Geometry.weightOffset h slot p.row n.val k.val) ∧
    (BitVec.ofNat 64 storage.output.base+outputWord h p n.val*BitVec.ofNat 64 storage.output.bytes).toNat=
      storage.output.address (Geometry.outputOffset h p.batch p.row n.val) := by
  obtain ⟨input,weights,output,_⟩ := all_source_offsets h storage p.toProgram slot selected n k
  obtain ⟨wi,ww,wo⟩ := word_offsets h p slot n.val k.val
  rw [wi,ww,wo]
  exact ⟨signed64 _ (input.trans (extent_bound storage.input)),
    signed64 _ (weights.trans (extent_bound storage.weights)),
    signed64 _ (output.trans (extent_bound storage.output)),
    pointer_word storage.input _ input,pointer_word storage.weights _ weights,pointer_word storage.output _ output⟩

#axiomsClean source_addresses
end VeriTile.Bench.Repairs.BgmvExpandSlice.NativeIndex
