import bench.repairs.tritonbench_g.context_attn_mistral.Proofs.NativeWords

namespace VeriTile.Bench.Repairs.ContextAttnMistral.NativeIndex
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 2000000

def rowBits (block lane : Nat) : BitVec 64 := fromI32 block*word 128+fromI32 lane
def indexBits : Nat → BitVec 64
  | 0 => word 0 | step+1 => indexBits step+word 128
def keyBits (step lane : Nat) : BitVec 64 := indexBits step+fromI32 lane
def chunks (block length : Nat) : Nat := if block*128 < length then block+1 else 0
def blockMask (w : MetadataWidth) (block length : Nat) : BitVec 64 :=
  if (fromI32 block*word 128).slt (metadataABI w length) then word 1 else word 0
def stopBits (w : MetadataWidth) (block length : Nat) : BitVec 64 :=
  blockMask w block length*(fromI32 block+word 1)*word 128

theorem row_modular (block lane : Nat) (hb : block < 2^31) (hl : lane < 128) :
    rowBits block lane = word (block*128+lane) := by
  rw [rowBits,fromI32_word block hb,fromI32_word lane (by omega)]
  simp only [word,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]

theorem index_modular (step : Nat) : indexBits step = word (step*128) := by
  induction step with
  | zero => rfl
  | succ step ih =>
    rw [indexBits,ih]
    simp only [word,BitVec.ofNat_add_ofNat]
    congr 1
    omega

theorem key_modular_control (step lane : Nat) (hl : lane < 128) :
    keyBits step lane = word (step*128+lane) := by
  rw [keyBits,index_modular,fromI32_word lane (by omega)]
  simp only [word,BitVec.ofNat_add_ofNat]

theorem chunks_le (block length : Nat) : chunks block length ≤ block+1 := by
  unfold chunks
  split <;> omega

theorem row_signed (block lane : Nat) (hb : block < 2^31) (hl : lane < 128) :
    (rowBits block lane).toInt = (block*128+lane : Nat) := by
  rw [row_modular block lane hb hl]
  exact word_signed _ (by omega)

theorem key_signed (block length step lane : Nat) (hb : block < 2^31)
    (ht : step < chunks block length) (hl : lane < 128) :
    (keyBits step lane).toInt = (step*128+lane : Nat) := by
  have hc := chunks_le block length
  rw [key_modular_control step lane hl]
  exact word_signed _ (by omega)

theorem blockMask_exact (w : MetadataWidth) (block length : Nat)
    (hb : block < 2^31) (hm : w.Fits length) :
    blockMask w block length = word (if block*128 < length then 1 else 0) := by
  have hd : (fromI32 block*word 128).toInt = (block*128 : Nat) := by
    rw [fromI32_word block hb]
    rw [show word block*word 128 = word (block*128) by
      simp only [word,BitVec.ofNat_mul_ofNat]]
    exact word_signed _ (by omega)
  have ht : (fromI32 block*word 128).slt (metadataABI w length) = Bool.true ↔ block*128 < length := by
    rw [BitVec.slt_iff_toInt_lt,hd,metadata_signed w length hm]
    exact Int.ofNat_lt
  unfold blockMask
  by_cases h : block*128 < length
  · simp [h,ht.mpr h]
  · have hn : (fromI32 block*word 128).slt (metadataABI w length) = Bool.false := by
      cases hbits : (fromI32 block*word 128).slt (metadataABI w length)
      · rfl
      · exact False.elim (h (ht.mp hbits))
    simp [h,hn]

theorem stop_modular (w : MetadataWidth) (block length : Nat)
    (hb : block < 2^31) (hm : w.Fits length) :
    stopBits w block length = word (chunks block length*128) := by
  rw [stopBits,blockMask_exact w block length hb hm,fromI32_word block hb]
  unfold chunks
  split <;> simp only [word,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]
  · congr 1; omega
  · simp

/-- The final loop test and increment are included. Legal grid IDs bound the
loop by 2^38 independently of the signed metadata length or max_input_len. -/
theorem loop_exact (w : MetadataWidth) (block length step : Nat)
    (hb : block < 2^31) (hm : w.Fits length) (ht : step ≤ chunks block length) :
    (indexBits step).toInt = (step*128 : Nat) ∧
    (stopBits w block length).toInt = (chunks block length*128 : Nat) ∧
    ((indexBits step).slt (stopBits w block length) = Bool.true ↔ step < chunks block length) ∧
    (indexBits step).toInt % 128 = 0 := by
  have hc := chunks_le block length
  have hi : (indexBits step).toInt = (step*128 : Nat) := by
    rw [index_modular]; exact word_signed _ (by omega)
  have hs : (stopBits w block length).toInt = (chunks block length*128 : Nat) := by
    rw [stop_modular w block length hb hm]; exact word_signed _ (by omega)
  refine ⟨hi,hs,?_,?_⟩
  · rw [BitVec.slt_iff_toInt_lt,hi,hs]
    omega
  · rw [hi]
    omega

theorem row_mask (w : MetadataWidth) (block length lane : Nat)
    (hb : block < 2^31) (hm : w.Fits length) (hl : lane < 128) :
    (rowBits block lane).slt (metadataABI w length) = Bool.true ↔ block*128+lane < length := by
  rw [BitVec.slt_iff_toInt_lt,row_signed block lane hb hl,metadata_signed w length hm]
  exact Int.ofNat_lt

theorem key_mask (w : MetadataWidth) (block length step lane : Nat)
    (hb : block < 2^31) (hm : w.Fits length) (ht : step < chunks block length) (hl : lane < 128) :
    (keyBits step lane).slt (metadataABI w length) = Bool.true ↔ step*128+lane < length := by
  rw [BitVec.slt_iff_toInt_lt,key_signed block length step lane hb ht hl,metadata_signed w length hm]
  exact Int.ofNat_lt

def clampedWindow (window : Int) : Int := min (max window 0) (2^63-1)
def windowABI (window : Int) : BitVec 64 := scalarABI (clampedWindow window).toNat
theorem window_signed (window : Int) : (windowABI window).toInt = clampedWindow window := by
  have hlo : 0 ≤ clampedWindow window := by unfold clampedWindow; omega
  have hhi : clampedWindow window < 2^63 := by unfold clampedWindow; omega
  rw [windowABI,scalarABI_word,word_signed _ (by omega)]
  exact Int.toNat_of_nonneg hlo

theorem difference_signed (block length step i j : Nat) (hb : block < 2^31)
    (ht : step < chunks block length) (hi : i < 128) (hj : j < 128) :
    (rowBits block i-keyBits step j).toInt = (block*128+i : Nat)-(step*128+j : Int) := by
  have hc := chunks_le block length
  rw [row_modular block i hb hi,key_modular_control step j hj]
  change (sword ((block*128+i : Nat) : Int)-sword ((step*128+j : Nat) : Int)).toInt = _
  have hsub : sword ((block*128+i : Nat) : Int)-sword ((step*128+j : Nat) : Int) =
      sword (((block*128+i : Nat) : Int)-((step*128+j : Nat) : Int)) := by
    apply BitVec.eq_of_toInt_eq
    simp [sword,BitVec.toInt_sub,BitVec.toInt_ofInt]
  rw [hsub]
  have he := sword_signed (((block*128+i : Nat) : Int)-((step*128+j : Nat) : Int)) (by omega) (by omega)
  simpa only [Nat.cast_add,Nat.cast_mul,Nat.cast_ofNat] using he

/-- Both finite-sentinel where statements are preserved together, for every
Python signed integer window. No overflowing row-window is evaluated natively. -/
theorem window_mask (window : Int) (block length step i j : Nat) (hb : block < 2^31)
    (ht : step < chunks block length) (hi : i < 128) (hj : j < 128) :
    ((keyBits step j).sle (rowBits block i) = Bool.true ∧
      (rowBits block i-keyBits step j).slt (windowABI window) = Bool.true) ↔
      ((step*128+j : Int) ≤ (block*128+i : Nat) ∧
        (step*128+j : Int) > (block*128+i : Nat)-window) := by
  have hc := chunks_le block length
  rw [BitVec.sle_iff_toInt_le,key_signed block length step j hb ht hj,
    row_signed block i hb hi,BitVec.slt_iff_toInt_lt,difference_signed block length step i j hb ht hi hj,
    window_signed]
  unfold clampedWindow
  omega

end
end VeriTile.Bench.Repairs.ContextAttnMistral.NativeIndex
