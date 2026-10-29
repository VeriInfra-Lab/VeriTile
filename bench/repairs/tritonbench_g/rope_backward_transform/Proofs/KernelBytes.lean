import bench.repairs.tritonbench_g.rope_backward_transform.Proofs.KernelPhysical
import bench.repairs.tritonbench_g.fast_rope_embedding.Misc.KernelBytes

namespace VeriTile.Bench.Repairs.RopeBackwardTransform.KernelContract
open VeriTile.Triton FP.Structural
noncomputable section
open scoped Classical
set_option maxHeartbeats 6000000

abbrev Bytes := VeriTile.Bench.Repairs.FastRopeEmbedding.KernelContract.Bytes
abbrev read32 := VeriTile.Bench.Repairs.FastRopeEmbedding.KernelContract.read32
abbrev wordByte := VeriTile.Bench.Repairs.FastRopeEmbedding.KernelContract.wordByte

theorem read32_byte (heap : Bytes) (address : Nat) (byte : Fin 4) :
    wordByte (read32 heap address) byte=heap (address+byte.val) :=
  VeriTile.Bench.Repairs.FastRopeEmbedding.KernelContract.read32_byte heap address byte

theorem decode_encode (word : Word) :
    (⟨wordByte word 3 ++ wordByte word 2 ++ wordByte word 1 ++ wordByte word 0⟩ : Word)=word :=
  VeriTile.Bench.Repairs.FastRopeEmbedding.KernelContract.decode_encode word

def encoded (c : Call) (final : Source.Memory Numerics.Tagged) (key : Bool) (address : Nat) : BitVec 8 :=
  wordByte ((final (region key) ((address-(c.q key).dataPtr.toNat)/4)).read .fp32).word
    ⟨(address-(c.q key).dataPtr.toNat)%4,Nat.mod_lt _ (by decide)⟩

def byteMemory (c : Call) (before : Bytes) (final : Source.Memory Numerics.Tagged) (address : Nat) : BitVec 8 :=
  if WrittenByte c Bool.false address then encoded c final Bool.false address
  else if WrittenByte c Bool.true address then encoded c final Bool.true address
  else before address

theorem encoded_coordinate (c : Call) (final : Source.Memory Numerics.Tagged) (key : Bool)
    (offset : Nat) (byte : Fin 4) :
    encoded c final key ((c.q key).dataPtr.toNat+4*offset+byte.val)=
      wordByte ((final (region key) offset).read .fp32).word byte := by
  have index : ((c.q key).dataPtr.toNat+4*offset+byte.val-(c.q key).dataPtr.toNat)/4=offset := by omega
  have rem : ((c.q key).dataPtr.toNat+4*offset+byte.val-(c.q key).dataPtr.toNat)%4=byte.val := by omega
  simp only [encoded,index]
  congr 1
  exact Fin.ext rem

theorem bytes_coordinate (c : Call) (admit : Admitted c) (before : Bytes)
    (final : Source.Memory Numerics.Tagged) (key : Bool) (offset : Nat)
    (bound : offset<c.geometry.size key) (byte : Fin 4) :
    byteMemory c before final ((c.q key).dataPtr.toNat+4*offset+byte.val)=
      if Written c.geometry key offset then wordByte ((final (region key) offset).read .fp32).word byte
      else before ((c.q key).dataPtr.toNat+4*offset+byte.val) := by
  have other:=opposite_apart c admit key _ (coordinate_span c admit key offset bound byte)
  have index : ((c.q key).dataPtr.toNat+4*offset+byte.val-(c.q key).dataPtr.toNat)/4=offset := by omega
  have own : WrittenByte c key ((c.q key).dataPtr.toNat+4*offset+byte.val) ↔ Written c.geometry key offset := by
    simp only [WrittenByte,index]; exact and_iff_right (by omega)
  cases key
  · simp only [Bool.not_false] at other
    simp [byteMemory,own,other,encoded_coordinate]
  · simp only [Bool.not_true] at other
    simp [byteMemory,own,other,encoded_coordinate]

theorem bytes_frame (c : Call) (before : Bytes) (final : Source.Memory Numerics.Tagged) (address : Nat)
    (outside : ¬WrittenByte c Bool.false address ∧ ¬WrittenByte c Bool.true address) :
    byteMemory c before final address=before address := by simp [byteMemory,outside.1,outside.2]

theorem regional_input {α β : Type} [Inhabited α] (embed : β→α) (before : Nat→β)
    (c : Call) (key : Bool) (offset : Nat) :
    ((memory embed before c) (region key) offset).read .fp32=
      embed (before ((c.q key).dataPtr.toNat+4*offset)) := by cases key <;> simp [memory,region]

theorem physical_initial {α β : Type} [Inhabited α] (embed : β→α) (project : α→β)
    (retract : ∀v,project (embed v)=v) (c : Call) (before : Nat→β) (address : Nat) :
    physical project c before (memory embed before c) address=before address := by
  have initial (key : Bool) (hit : WrittenWord c key address) :
      project (((memory embed before c) (region key) ((address-(c.q key).dataPtr.toNat)/4)).read .fp32)=before address := by
    rw [regional_input,retract]
    congr 1
    have lower:=hit.1.1
    have aligned:=hit.2
    omega
  unfold physical
  split
  · rename_i hit; exact initial Bool.false hit
  · split
    · rename_i _ hit; exact initial Bool.true hit
    · rfl

theorem physical_store {α β : Type} [Inhabited α] (project : α→β) (c : Call) (admit : Admitted c)
    (before : Nat→β) (s : State α) (key : Bool) (offset : Nat) (written : Written c.geometry key offset)
    (value : α) (address : Nat) :
    physical project c before (s.write (region key) offset (.mk .fp32 value)).mem address=
      if address=(c.q key).dataPtr.toNat+4*offset then project value
      else physical project c before s.mem address := by
  by_cases inside : address=(c.q key).dataPtr.toNat+4*offset
  · rw [if_pos inside,inside,physical_coordinate project c admit before _ key offset
      (written_bound _ _ _ written),if_pos written]
    simp [State.write]
  · rw [if_neg inside]
    have unchanged (other : Bool) (hit : WrittenWord c other address) :
        ((s.write (region key) offset (.mk .fp32 value)).mem (region other)
          ((address-(c.q other).dataPtr.toNat)/4)).read .fp32=
        (s.mem (region other) ((address-(c.q other).dataPtr.toNat)/4)).read .fp32 := by
      by_cases same : other=key
      · subst other
        have different : (address-(c.q key).dataPtr.toNat)/4≠offset := by
          have lower:=hit.1.1; have aligned:=hit.2; omega
        simp [State.write,different]
      · have different : region other≠region key := fun eq=>same (region_injective eq)
        simp [State.write,different]
    unfold physical
    split
    · rename_i hit; exact congrArg project (unchanged Bool.false hit)
    · split
      · rename_i _ hit; exact congrArg project (unchanged Bool.true hit)
      · rfl

theorem byte_initial (c : Call) (before : Bytes) (address : Nat) :
    byteMemory c before (memory Numerics.leaf (read32 before) c) address=before address := by
  have initial (key : Bool) (lower : (c.q key).dataPtr.toNat≤address) :
      encoded c (memory Numerics.leaf (read32 before) c) key address=before address := by
    unfold encoded
    rw [regional_input]
    simp only [Numerics.leaf]
    rw [read32_byte]
    congr 1
    change (c.q key).dataPtr.toNat+4*((address-(c.q key).dataPtr.toNat)/4)+
      (address-(c.q key).dataPtr.toNat)%4=address
    omega
  unfold byteMemory
  split
  · rename_i hit; exact initial Bool.false hit.1
  · split
    · rename_i _ hit; exact initial Bool.true hit.1
    · rfl

def writeCell (m : Source.Memory Numerics.Tagged) (key : Bool) (offset : Nat) (value : Numerics.Tagged) :
    Source.Memory Numerics.Tagged :=
  fun name index=>if name=region key ∧ index=offset then .mk .fp32 value else m name index

theorem source_store (m : Source.Memory Numerics.Tagged) (key : Bool) (offset : Nat) (value : Numerics.Tagged) :
    ((state m 0).write (region key) offset (.mk .fp32 value)).mem=writeCell m key offset value := rfl

theorem byte_store (c : Call) (admit : Admitted c) (before : Bytes) (m : Source.Memory Numerics.Tagged)
    (key : Bool) (offset : Nat) (written : Written c.geometry key offset)
    (value : Numerics.Tagged) (address : Nat) :
    byteMemory c before (writeCell m key offset value) address=
      if inside : (c.q key).dataPtr.toNat+4*offset≤address ∧ address<(c.q key).dataPtr.toNat+4*offset+4 then
        wordByte value.word ⟨address-((c.q key).dataPtr.toNat+4*offset),by omega⟩
      else byteMemory c before m address := by
  by_cases inside : (c.q key).dataPtr.toNat+4*offset≤address ∧ address<(c.q key).dataPtr.toNat+4*offset+4
  · let byte : Fin 4:=⟨address-((c.q key).dataPtr.toNat+4*offset),by omega⟩
    have eq : address=(c.q key).dataPtr.toNat+4*offset+byte.val := by dsimp [byte]; omega
    rw [dif_pos inside]
    conv_lhs=>rw [eq]
    rw [bytes_coordinate c admit before _ key offset (written_bound _ _ _ written) byte,if_pos written]
    simp [writeCell,byte]
  · rw [dif_neg inside]
    have unchanged (other : Bool) (hit : WrittenByte c other address) :
        encoded c (writeCell m key offset value) other address=encoded c m other address := by
      by_cases same : other=key
      · subst other
        have different : (address-(c.q key).dataPtr.toNat)/4≠offset := by have lower:=hit.1; omega
        simp [encoded,writeCell,different]
      · have different : region other≠region key := fun eq=>same (region_injective eq)
        simp [encoded,writeCell,different]
    unfold byteMemory
    split
    · rename_i hit; exact unchanged Bool.false hit
    · split
      · rename_i _ hit; exact unchanged Bool.true hit
      · rfl

/-- All four bytes at every gradient coordinate, including unrotated tails. -/
theorem tensor_bytes (c : Call) (admit : Admitted c) (before : Bytes)
    (final : Source.Memory Numerics.Tagged) (key : Bool) (row head col : Nat)
    (hr : row<c.geometry.rows) (hh : head<c.geometry.heads key) (hc : col<c.geometry.dimension) (byte : Fin 4) :
    byteMemory c before final ((c.q key).storage.address ((c.q key).physicalIndex (c.geometry.offset key row head col))+byte.val)=
      if col<2*c.geometry.half then wordByte ((final (region key) (c.geometry.offset key row head col)).read .fp32).word byte
      else before ((c.q key).storage.address ((c.q key).physicalIndex (c.geometry.offset key row head col))+byte.val) := by
  have bound:=offset_bound c.geometry key row head col hr hh hc
  rw [q_address c admit key _ bound,bytes_coordinate c admit before final key _ bound byte,
    written_coordinate c key row head col hr hh hc]
  by_cases active : col<2*c.geometry.half <;> simp [active]

theorem source_store_written (c : Call) (row : Nat) (key side : Bool) (i : Lane c.geometry key)
    (hr : row<c.geometry.rows) (active : Source.active (c.geometry.heads key) c.geometry.half i) :
    Written c.geometry key (address c.geometry row key side i) := by
  obtain ⟨hh,hj⟩:=(active_iff _ _ i).mp active
  refine ⟨row,i.1.val,i.2.1.val+if side then c.geometry.half else 0,hr,hh,?_,?_⟩
  · cases side <;> simp only [Bool.false_eq_true,↓reduceIte] <;> omega
  · exact address_coordinate c.geometry key side row i

theorem table_frame (c : Call) (admit : Admitted c) (before : Bytes)
    (final : Source.Memory Numerics.Tagged) (sine : Bool) (address : Nat)
    (inside : InStorage (c.tables sine) address) : byteMemory c before final address=before address := by
  apply bytes_frame
  constructor
  · intro hit
    exact apart_points _ _ (admit.tableApart Bool.false sine) address (written_span c admit Bool.false address hit) inside
  · intro hit
    exact apart_points _ _ (admit.tableApart Bool.true sine) address (written_span c admit Bool.true address hit) inside

#axiomsClean byte_initial
#axiomsClean physical_initial
#axiomsClean physical_store
#axiomsClean source_store
#axiomsClean source_store_written
#axiomsClean byte_store
#axiomsClean tensor_bytes
#axiomsClean bytes_frame
#axiomsClean table_frame
end
end VeriTile.Bench.Repairs.RopeBackwardTransform.KernelContract
