import bench.repairs.tritonbench_g.context_attn_mistral.Proofs.NativeControl

namespace VeriTile.Bench.Repairs.ContextAttnMistral.NativeIndex
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 2000000

/-- The signed element offset is scaled by the pointee byte width. -/
def addptr (base : BitVec 64) (bytes : Nat) (offset : BitVec 64) : BitVec 64 :=
  base+sword ((bytes : Int)*offset.toInt)

def queryABI (w : MetadataWidth) (start block lane head dim sr sh sd : Nat) : BitVec 64 :=
  (metadataABI w start+rowBits block lane)*scalarABI sr+
    fromI32 head*scalarABI sh+fromI32 dim*scalarABI sd

/-- K/V first form the lane pointer, then add the start+loop offset. These
are two actual addptr operations, so inactive intermediate pointers may wrap. -/
def keyABI (w : MetadataWidth) (base bytes start step lane head dim group sr sh sd : Nat) : BitVec 64 :=
  addptr (addptr (word base) bytes
    (fromI32 lane*scalarABI sr+(fromI32 head).sdiv (scalarABI group)*scalarABI sh+
      fromI32 dim*scalarABI sd)) bytes ((metadataABI w start+indexBits step)*scalarABI sr)

theorem queryABI_modular (w : MetadataWidth) (start block lane head dim sr sh sd : Nat)
    (hs : w.Fits start) (hb : block < 2^31) (hl : lane < 128)
    (hh : head < 2^31) (hd : dim < 128) :
    queryABI w start block lane head dim sr sh sd =
      word ((start+(block*128+lane))*sr+head*sh+dim*sd) := by
  simp only [queryABI,metadata_word w start hs,row_modular block lane hb hl,
    fromI32_word head hh,fromI32_word dim (by omega),scalarABI_word]
  exact query_modular start (block*128+lane) head dim sr sh sd

theorem addptr_modular (base bytes offset : Nat) :
    addptr (word base) bytes (word offset) = word (base+bytes*offset) := byte_modular base bytes offset

theorem keyABI_modular (w : MetadataWidth) (base bytes start step lane head dim group sr sh sd : Nat)
    (hs : w.Fits start) (hl : lane < 128) (hh : head < 2^31)
    (hd : dim < 128) (hg : group < 2^63) :
    keyABI w base bytes start step lane head dim group sr sh sd =
      word (base+bytes*((start+(step*128+lane))*sr+(head/group)*sh+dim*sd)) := by
  rw [keyABI,signed_div head group hh hg]
  simp only [metadata_word w start hs,index_modular,fromI32_word lane (by omega),
    fromI32_word dim (by omega),scalarABI_word,
    addptr,signed_byte_residue,word,BitVec.ofNat_add_ofNat,BitVec.ofNat_mul_ofNat]
  congr 1
  ring

/-- Complete accessed words, including every byte, lie in the backing object.
This is derived from the ordinary element bound, never from padded offsets. -/
theorem complete_word (base bytes extent offset : Nat) (positive : 0 < bytes)
    (allocated : base+bytes*extent ≤ 2^64) (active : offset < extent) :
    (word (base+bytes*offset)).toNat = base+bytes*offset ∧
      ∀ j, j < bytes →
        (word (base+bytes*offset)+word j).toNat = base+bytes*offset+j ∧
        base ≤ base+bytes*offset+j ∧ base+bytes*offset+j < base+bytes*extent := by
  have whole : base+bytes*offset+bytes ≤ base+bytes*extent := by
    have hm := Nat.mul_le_mul_left bytes (show offset+1 ≤ extent by omega)
    simp only [Nat.mul_add,Nat.mul_one] at hm
    omega
  refine ⟨word_unsigned _ (by omega),?_⟩
  intro j hj
  refine ⟨?_,by omega,by omega⟩
  rw [show word (base+bytes*offset)+word j = word (base+bytes*offset+j) by
    simp only [word,BitVec.ofNat_add_ofNat]]
  exact word_unsigned _ (by omega)

/-- Both physical metadata widths project to the normalized discrete Int
carrier. The old `.int32` marker is not a claim that an int64 load is int32. -/
def MetadataWidth.raw : (w : MetadataWidth) → Nat → BitVec (8*w.bytes)
  | .int32,n => BitVec.ofNat 32 n | .int64,n => BitVec.ofNat 64 n

def MetadataWidth.widen (w : MetadataWidth) (raw : BitVec (8*w.bytes)) : BitVec 64 :=
  raw.signExtend 64

theorem metadata_load (w : MetadataWidth) (n : Nat) (fits : w.Fits n) :
    w.widen (w.raw n) = metadataABI w n ∧
      (w.widen (w.raw n)).toInt = (n : Int) := by
  have eq : w.widen (w.raw n) = metadataABI w n := by
    cases w
    · rfl
    · change BitVec.ofInt 64 (word n).toInt = word n
      exact BitVec.ofInt_toInt
  exact ⟨eq,eq ▸ metadata_signed w n fits⟩

end
end VeriTile.Bench.Repairs.ContextAttnMistral.NativeIndex
