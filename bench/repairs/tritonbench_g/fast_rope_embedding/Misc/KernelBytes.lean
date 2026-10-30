import bench.repairs.tritonbench_g.fast_rope_embedding.Misc.KernelResult

namespace VeriTile.Bench.Repairs.FastRopeEmbedding.KernelContract
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RopeEmbedding
noncomputable section
open scoped Classical
set_option maxHeartbeats 3000000

abbrev Bytes := Nat→BitVec 8

/-- FP32 input words are assembled from the actual little-endian byte heap. -/
def read32 (heap : Bytes) (address : Nat) : Word :=
  ⟨heap (address+3) ++ heap (address+2) ++ heap (address+1) ++ heap address⟩

def wordByte (word : Word) (byte : Fin 4) : BitVec 8 :=
  word.bits.extractLsb' (8*byte.val) 8

theorem decode_encode (word : Word) :
    (⟨wordByte word 3 ++ wordByte word 2 ++ wordByte word 1 ++ wordByte word 0⟩ : Word)=word := by
  have bits : (wordByte word 3 ++ wordByte word 2 ++ wordByte word 1 ++ wordByte word 0)=word.bits := by
    change (word.bits.extractLsb' 24 8 ++ word.bits.extractLsb' 16 8 ++
      word.bits.extractLsb' 8 8 ++ word.bits.extractLsb' 0 8)=word.bits
    rw [BitVec.extractLsb'_append_extractLsb'_eq_extractLsb' (by decide : 24=16+8),
      BitVec.extractLsb'_append_extractLsb'_eq_extractLsb' (by decide : 16=8+8),
      BitVec.extractLsb'_append_extractLsb'_eq_extractLsb' (by decide : 8=0+8)]
    exact BitVec.extractLsb'_eq_self
  exact congrArg (fun b=>(⟨b⟩ : Word)) bits

theorem read32_byte (heap : Bytes) (address : Nat) (byte : Fin 4) :
    wordByte (read32 heap address) byte=heap (address+byte.val) := by
  fin_cases byte
  · exact BitVec.extractLsb'_append_eq_right
  · change ((heap (address+3) ++ heap (address+2) ++ heap (address+1) ++ heap address) : BitVec 32).extractLsb' 8 8=heap (address+1)
    rw [BitVec.extractLsb'_append_eq_of_le (by decide : 8≤8)]
    exact BitVec.extractLsb'_append_eq_right
  · change ((heap (address+3) ++ heap (address+2) ++ heap (address+1) ++ heap address) : BitVec 32).extractLsb' 16 8=heap (address+2)
    rw [BitVec.extractLsb'_append_eq_of_le (by decide : 8≤16),
      BitVec.extractLsb'_append_eq_of_le (by decide : 8≤8)]
    exact BitVec.extractLsb'_append_eq_right
  · change ((heap (address+3) ++ heap (address+2) ++ heap (address+1) ++ heap address) : BitVec 32).extractLsb' 24 8=heap (address+3)
    rw [BitVec.extractLsb'_append_eq_of_le (by decide : 8≤24),
      BitVec.extractLsb'_append_eq_of_le (by decide : 8≤16)]
    exact BitVec.extractLsb'_append_eq_left

/-- All four bytes of each actual source store, including physical padding in
unwritten rows or channels only when the source itself writes that word. -/
def WrittenByte (c : Call) (row group address : Nat) : Prop :=
  c.inputs.q.dataPtr.toNat≤address ∧
    Kernel.Written c.geometry row group "Q" ((address-c.inputs.q.dataPtr.toNat)/4)

def byteMemory (c : Call) (row group : Nat) (before : Bytes) (final : State Numerics.Tagged)
    (address : Nat) : BitVec 8 :=
  if WrittenByte c row group address then
    wordByte ((final.mem "Q" ((address-c.inputs.q.dataPtr.toNat)/4)).read .real).word
      ⟨(address-c.inputs.q.dataPtr.toNat)%4,Nat.mod_lt _ (by decide)⟩
  else before address

/-- The source-state byte interpretation starts at exactly the input heap. -/
theorem byte_initial (c : Call) (row group : Nat) (before : Bytes) (address : Nat) :
    byteMemory c row group before (state Numerics.leaf (read32 before) c.inputs) address=before address := by
  unfold byteMemory
  split
  · rename_i hit
    simp only [state,↓reduceIte,Cell.read_same,Numerics.leaf]
    rw [read32_byte]
    congr 1
    change c.inputs.q.dataPtr.toNat+4*((address-c.inputs.q.dataPtr.toNat)/4)+
      (address-c.inputs.q.dataPtr.toNat)%4=address
    have lower := hit.1
    omega
  · rfl

theorem coordinate_bytes (c : Call) (row group : Nat) (before : Bytes) (final : State Numerics.Tagged)
    (offset : Nat) (written : Kernel.Written c.geometry row group "Q" offset) (byte : Fin 4) :
    byteMemory c row group before final (c.inputs.q.dataPtr.toNat+4*offset+byte.val)=
      wordByte ((final.mem "Q" offset).read .real).word byte := by
  have index : (c.inputs.q.dataPtr.toNat+4*offset+byte.val-c.inputs.q.dataPtr.toNat)/4=offset := by omega
  have remainder : (c.inputs.q.dataPtr.toNat+4*offset+byte.val-c.inputs.q.dataPtr.toNat)%4=byte.val := by omega
  simp only [byteMemory,WrittenByte,index,written,and_true]
  rw [if_pos (by omega)]
  congr 1
  exact Fin.ext remainder

theorem bytes_frame (c : Call) (row group : Nat) (before : Bytes) (final : State Numerics.Tagged)
    (address : Nat) (outside : ¬WrittenByte c row group address) :
    byteMemory c row group before final address=before address := by simp [byteMemory,outside]

/-- A regional store updates exactly four bytes under the physical encoding.
The value is obtained from that source store, never from a result callback. -/
theorem byte_store (c : Call) (row group : Nat) (before : Bytes) (s : State Numerics.Tagged)
    (offset : Nat) (written : Kernel.Written c.geometry row group "Q" offset)
    (value : Numerics.Tagged) (address : Nat) :
    byteMemory c row group before (s.write "Q" offset (.mk .real value)) address=
      if h : c.inputs.q.dataPtr.toNat+4*offset≤address ∧ address<c.inputs.q.dataPtr.toNat+4*offset+4 then
        wordByte value.word ⟨address-(c.inputs.q.dataPtr.toNat+4*offset),by omega⟩
      else byteMemory c row group before s address := by
  by_cases inside : c.inputs.q.dataPtr.toNat+4*offset≤address ∧ address<c.inputs.q.dataPtr.toNat+4*offset+4
  · have b : address-(c.inputs.q.dataPtr.toNat+4*offset)<4 := by omega
    let byte : Fin 4 := ⟨address-(c.inputs.q.dataPtr.toNat+4*offset),b⟩
    have eq : address=c.inputs.q.dataPtr.toNat+4*offset+byte.val := by dsimp [byte]; omega
    rw [dif_pos inside]
    conv_lhs => rw [eq]
    rw [coordinate_bytes c row group before _ offset written byte]
    simp [State.write,byte]
  · rw [dif_neg inside]
    unfold byteMemory
    split
    · rename_i hit
      have different : (address-c.inputs.q.dataPtr.toNat)/4≠offset := by
        have lower := hit.1
        omega
      simp [State.write,different]
    · rfl

/-- Every Tensor byte is read from the same completed byte memory. -/
theorem tensor_bytes (c : Call) (admit : Admitted c) (row group : Nat) (before : Bytes)
    (final : State Numerics.Tagged) (value : Nat→Nat→Numerics.Tagged)
    (post : Kernel.Post c.geometry row group (state Numerics.leaf (read32 before) c.inputs) final value)
    (peer head col : Nat) (hr : peer<c.geometry.rows) (hh : head<c.geometry.heads)
    (hc : col<c.geometry.dimension) (byte : Fin 4) :
    byteMemory c row group before final
        (c.inputs.q.storage.address (c.inputs.q.physicalIndex (c.geometry.offset peer head col))+byte.val)=
      if affected c row group peer head col then wordByte (value head col).word byte
      else before (c.inputs.q.storage.address (c.inputs.q.physicalIndex (c.geometry.offset peer head col))+byte.val) := by
  have bound : c.geometry.offset peer head col<c.inputs.q.metadata.numel := by
    rw [q_size c admit]; exact offset_lt c.geometry peer head col hr hh hc
  rw [tensor_address c.inputs.q c.inputs.qLaws
    (KernelBinding.admitted_payload c.inputs c.geometry admit.metadata).1 _ bound]
  by_cases active : affected c row group peer head col
  · rw [if_pos active]
    have written := (written_coordinate c row group peer head col hh hc).mpr active
    rw [coordinate_bytes c row group before final _ written byte]
    rcases active with ⟨rfl,hlo,hhi,hcol⟩
    rw [post.values head col hlo hhi hcol,Cell.read_same]
  · rw [if_neg active]
    apply bytes_frame
    intro hit
    have index : (c.inputs.q.dataPtr.toNat+4*c.geometry.offset peer head col+byte.val-
        c.inputs.q.dataPtr.toNat)/4=c.geometry.offset peer head col := by omega
    rw [WrittenByte,index] at hit
    exact active ((written_coordinate c row group peer head col hh hc).mp hit.2)

/-- Decoding the four proved bytes gives the same word result; the byte heap
and word observations are not unrelated output witnesses. -/
theorem tensor_byte_word (c : Call) (admit : Admitted c) (row group : Nat) (before : Bytes)
    (final : State Numerics.Tagged) (value : Nat→Nat→Numerics.Tagged)
    (post : Kernel.Post c.geometry row group (state Numerics.leaf (read32 before) c.inputs) final value)
    (peer head col : Nat) (hr : peer<c.geometry.rows) (hh : head<c.geometry.heads)
    (hc : col<c.geometry.dimension) :
    logicalRead (read32 (byteMemory c row group before final)) c.inputs.q (c.geometry.offset peer head col)=
      if affected c row group peer head col then (value head col).word
      else logicalRead (read32 before) c.inputs.q (c.geometry.offset peer head col) := by
  have bytes := tensor_bytes c admit row group before final value post peer head col hr hh hc
  unfold logicalRead read32
  have last := bytes 0
  simp only [Fin.val_zero,Nat.add_zero] at last
  have b3 := bytes ⟨3,by decide⟩
  have b2 := bytes ⟨2,by decide⟩
  have b1 := bytes ⟨1,by decide⟩
  rw [b3,b2,b1,last]
  split_ifs
  · exact decode_encode _
  · rfl

#axiomsClean decode_encode
#axiomsClean read32_byte
#axiomsClean byte_initial
#axiomsClean coordinate_bytes
#axiomsClean bytes_frame
#axiomsClean byte_store
#axiomsClean tensor_bytes
#axiomsClean tensor_byte_word
end
end VeriTile.Bench.Repairs.FastRopeEmbedding.KernelContract
