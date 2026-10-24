import bench.repairs.tritonbench_g.rope_transform.Proofs.KernelMath
import bench.repairs.tritonbench_g.fast_rope_embedding.Misc.KernelBytes

namespace VeriTile.Bench.Repairs.RopeTransform.KernelContract
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

def encoded (c : Call) (final : T.Memory Tagged) (key : Bool) (address : Nat) : BitVec 8 :=
  wordByte ((final (region key) ((address-(c.q key).dataPtr.toNat)/4)).read .fp32).word
    ⟨(address-(c.q key).dataPtr.toNat)%4,Nat.mod_lt _ (by decide)⟩

def byteMemory (c : Call) (before : Bytes) (final : T.Memory Tagged) (address : Nat) : BitVec 8 :=
  if WrittenByte c Bool.false address then encoded c final Bool.false address
  else if WrittenByte c Bool.true address then encoded c final Bool.true address
  else before address

theorem encoded_coordinate (c : Call) (final : T.Memory Tagged) (key : Bool)
    (offset : Nat) (byte : Fin 4) :
    encoded c final key ((c.q key).dataPtr.toNat+4*offset+byte.val)=
      wordByte ((final (region key) offset).read .fp32).word byte := by
  have index : ((c.q key).dataPtr.toNat+4*offset+byte.val-(c.q key).dataPtr.toNat)/4=offset := by omega
  have rem : ((c.q key).dataPtr.toNat+4*offset+byte.val-(c.q key).dataPtr.toNat)%4=byte.val := by omega
  simp only [encoded,index]
  congr 1
  exact Fin.ext rem

theorem bytes_coordinate (c : Call) (admit : Admitted c) (before : Bytes)
    (final : T.Memory Tagged) (key : Bool) (offset : Nat)
    (bound : offset<RequestSize c.geometry key) (byte : Fin 4) :
    byteMemory c before final ((c.q key).dataPtr.toNat+4*offset+byte.val)=
      wordByte ((final (region key) offset).read .fp32).word byte := by
  have other:=opposite_apart c admit key _ (coordinate_span c admit key offset bound byte)
  have index : ((c.q key).dataPtr.toNat+4*offset+byte.val-(c.q key).dataPtr.toNat)/4=offset := by omega
  have own : WrittenByte c key ((c.q key).dataPtr.toNat+4*offset+byte.val) ↔ offset<RequestSize c.geometry key := by
    simp only [WrittenByte,index]; exact and_iff_right (by omega)
  cases key
  · simp only [Bool.not_false] at other
    simp [byteMemory,own,other,encoded_coordinate,bound]
  · simp only [Bool.not_true] at other
    simp [byteMemory,own,other,encoded_coordinate,bound]

theorem bytes_frame (c : Call) (before : Bytes) (final : T.Memory Tagged) (address : Nat)
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
    (before : Nat→β) (s : State α) (key : Bool) (offset : Nat) (written : offset<RequestSize c.geometry key)
    (value : α) (address : Nat) :
    physical project c before (s.write (region key) offset (.mk .fp32 value)).mem address=
      if address=(c.q key).dataPtr.toNat+4*offset then project value
      else physical project c before s.mem address := by
  by_cases inside : address=(c.q key).dataPtr.toNat+4*offset
  · rw [if_pos inside,inside,physical_coordinate project c admit before _ key offset
      written]
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
      · have different : region other≠region key := by cases other <;> cases key <;> simp_all [region]
        simp [State.write,different]
    unfold physical
    split
    · rename_i hit; exact congrArg project (unchanged Bool.false hit)
    · split
      · rename_i _ hit; exact congrArg project (unchanged Bool.true hit)
      · rfl

theorem byte_initial (c : Call) (before : Bytes) (address : Nat) :
    byteMemory c before (memory leaf (read32 before) c) address=before address := by
  have initial (key : Bool) (lower : (c.q key).dataPtr.toNat≤address) :
      encoded c (memory leaf (read32 before) c) key address=before address := by
    unfold encoded
    rw [regional_input]
    simp only [leaf,VeriTile.Bench.Repairs.RopeBackwardTransform.Numerics.leaf]
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

def writeCell (m : T.Memory Tagged) (key : Bool) (offset : Nat) (value : Tagged) :
    T.Memory Tagged :=
  fun name index=>if name=region key ∧ index=offset then .mk .fp32 value else m name index

theorem source_store (s : State Tagged) (key : Bool) (offset : Nat) (value : Tagged) :
    (s.write (region key) offset (.mk .fp32 value)).mem=writeCell s.mem key offset value := rfl

theorem byte_store (c : Call) (admit : Admitted c) (before : Bytes) (m : T.Memory Tagged)
    (key : Bool) (offset : Nat) (written : offset<RequestSize c.geometry key)
    (value : Tagged) (address : Nat) :
    byteMemory c before (writeCell m key offset value) address=
      if inside : (c.q key).dataPtr.toNat+4*offset≤address ∧ address<(c.q key).dataPtr.toNat+4*offset+4 then
        wordByte value.word ⟨address-((c.q key).dataPtr.toNat+4*offset),by omega⟩
      else byteMemory c before m address := by
  by_cases inside : (c.q key).dataPtr.toNat+4*offset≤address ∧ address<(c.q key).dataPtr.toNat+4*offset+4
  · let byte : Fin 4:=⟨address-((c.q key).dataPtr.toNat+4*offset),by omega⟩
    have eq : address=(c.q key).dataPtr.toNat+4*offset+byte.val := by dsimp [byte]; omega
    rw [dif_pos inside]
    conv_lhs=>rw [eq]
    rw [bytes_coordinate c admit before _ key offset written byte]
    simp [writeCell,byte]
  · rw [dif_neg inside]
    have unchanged (other : Bool) (hit : WrittenByte c other address) :
        encoded c (writeCell m key offset value) other address=encoded c m other address := by
      by_cases same : other=key
      · subst other
        have different : (address-(c.q key).dataPtr.toNat)/4≠offset := by have lower:=hit.1; omega
        simp [encoded,writeCell,different]
      · have different : region other≠region key := by cases other <;> cases key <;> simp_all [region]
        simp [encoded,writeCell,different]
    unfold byteMemory
    split
    · rename_i hit; exact unchanged Bool.false hit
    · split
      · rename_i _ hit; exact unchanged Bool.true hit
      · rfl

/-- Every byte of every Q/K coordinate is read from the same completed source memory. -/
theorem native_bytes (cfg : FP.Config) (c : Call) (admit : Admitted c) (before : Bytes)
    (key : Bool) (row head col : Nat) (hr : row<c.geometry.rows) (hh : head<c.geometry.heads key)
    (hc : col<c.geometry.dimension) (byte : Fin 4) :
    byteMemory c before (Grid.run (nativeAlgebra cfg) c.geometry
        (entry leaf (read32 before) c) c.geometry.rows).mem
        ((c.q key).storage.address ((c.q key).physicalIndex (c.geometry.offset key row head col))+byte.val)=
      wordByte (nativeValue cfg c (read32 before) key row head col) byte := by
  have bound:=offset_bound c.geometry key row head col hr hh hc
  rw [q_address c admit key _ bound,bytes_coordinate c admit before _ key _ bound byte,
    grid_values (nativeAlgebra cfg) leaf c admit (read32 before) key row head col hr hh hc,native_binding]

theorem native_words (cfg : FP.Config) (c : Call) (admit : Admitted c) (before : Bytes)
    (key : Bool) (row head col : Nat) (hr : row<c.geometry.rows) (hh : head<c.geometry.heads key)
    (hc : col<c.geometry.dimension) :
    logicalRead (read32 (byteMemory c before (Grid.run (nativeAlgebra cfg) c.geometry
        (entry leaf (read32 before) c) c.geometry.rows).mem))
      (c.q key) (c.geometry.offset key row head col)=
      nativeValue cfg c (read32 before) key row head col := by
  have bytes:=native_bytes cfg c admit before key row head col hr hh hc
  have b0:=bytes ⟨0,by decide⟩
  have b1:=bytes ⟨1,by decide⟩
  have b2:=bytes ⟨2,by decide⟩
  have b3:=bytes ⟨3,by decide⟩
  simp only [Nat.add_zero] at b0
  unfold logicalRead
  change (⟨_ ++ _ ++ _ ++ _⟩ : Word)=_
  rw [b3,b2,b1,b0]
  exact decode_encode _

theorem source_store_written (c : Call) (row : Nat) (key side : Bool)
    (i : TileIndex [c.geometry.paddedHeads key,c.geometry.width])
    (hr : row<c.geometry.rows) (active : T.active (c.geometry.heads key) (c.geometry.dimension/2) i) :
    T.address row (c.geometry.stride key) c.geometry.dimension (c.geometry.dimension/2) side i<
      RequestSize c.geometry key := by
  have bounds : i.1.val<c.geometry.heads key ∧ i.2.1.val<c.geometry.half := by
    simpa [T.active,half_eq] using active
  rw [address_eq]
  exact offset_bound c.geometry key row i.1.val _ hr bounds.1 (column_lt c.geometry side _ bounds.2)

theorem table_frame (c : Call) (admit : Admitted c) (before : Bytes)
    (final : T.Memory Tagged) (sine : Bool) (address : Nat)
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
#axiomsClean native_words
#axiomsClean native_bytes
#axiomsClean bytes_frame
#axiomsClean table_frame
end
end VeriTile.Bench.Repairs.RopeTransform.KernelContract
