import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.Graph
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.CheckedMemory
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.Numeric
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001
noncomputable section
namespace Checked
abbrev Word := BitVec 32
abbrev Byte := BitVec 8
def wordByte (word : Word) (lane : Nat) : Byte := BitVec.ofNat 8 (word.toNat/2^(8*lane))

/-- Local physical load/store and little-endian byte observation laws. There is
no kernel, dot, reduction, or returned-value equation in this runtime ABI. -/
structure MemoryABI (σ : Type) where
  words : σ → Memory Word
  bytes : σ → Nat → Byte
  store : σ → Nat → Word → σ
  store_words : ∀ state address word,
    words (store state address word)=physicalWrite (words state) address word
  codec : ∀ state address,bytes state address=wordByte (words state (address/4)) (address%4)

def nativeExecute (abi : MemoryABI σ) (ports : Numeric.ScalarPorts) (c : Shape)
    (b : Binding c) (scale : Word) : List Store → σ → σ
  | [],state => state
  | s::rest,state => nativeExecute abi ports c b scale rest
      (abi.store state (b.address s.address)
        (s.rhs.eval (wordOps (Numeric.nativeISA ports)) scale (b.view 0 (abi.words state))))

theorem native_words (abi : MemoryABI σ) (ports : Numeric.ScalarPorts) (c : Shape)
    (b : Binding c) (scale : Word) (code : List Store) (state : σ) :
    abi.words (nativeExecute abi ports c b scale code state)=
      execute (wordOps (Numeric.nativeISA ports)) c b scale code (abi.words state) := by
  induction code generalizing state with
  | nil => rfl
  | cons s rest ih =>
    simp only [nativeExecute,execute,ih,abi.store_words]
    rfl

theorem native_reference (abi : MemoryABI σ) (models : Numeric.UnaryModels)
    (ports : Numeric.ScalarPorts) (laws : Numeric.ScalarLaws models ports) (c : Shape)
    (b : Binding c) (scale : Word) (code : List Store) (state : σ) :
    abi.words (nativeExecute abi ports c b scale code state)=
      execute (wordOps (Numeric.referenceISA models)) c b scale code (abi.words state) := by
  rw [native_words,Numeric.isa_eq models ports laws]

theorem byte_at_word (abi : MemoryABI σ) (state : σ) (address lane : Nat) (hl : lane<4) :
    abi.bytes state (4*address+lane)=wordByte (abi.words state address) lane := by
  rw [abi.codec]
  congr 2 <;> omega

theorem code_frame (ops : Ops α) (c : Shape) (b : Binding c) (scale : α)
    (code : List Store) (m : Memory α) (point : Nat)
    (outside : ∀ s∈code,point≠b.address s.address) :
    execute ops c b scale code m point=m point := by
  induction code generalizing m with
  | nil => rfl
  | cons s rest ih =>
    rw [execute,ih _ (fun a ha => outside a (by simp [ha]))]
    simp [physicalWrite,outside s (by simp)]

theorem native_byte_frame (abi : MemoryABI σ) (ports : Numeric.ScalarPorts) (c : Shape)
    (b : Binding c) (scale : Word) (code : List Store) (state : σ) (point : Nat)
    (outside : ∀ s∈code,point/4≠b.address s.address) :
    abi.bytes (nativeExecute abi ports c b scale code state) point=abi.bytes state point := by
  rw [abi.codec,abi.codec,native_words,code_frame _ _ _ _ _ _ _ outside]

theorem phaseCode_addresses (c : Shape) (tree : SumTree) (p : Phase) (events : List Address) :
    (phaseCode c tree p events).map Store.address=events := by simp [phaseCode,Function.comp_def]
theorem forwardCode_addresses (c : Shape) (tree : SumTree) (rf : Bool) :
    (forwardCode c tree rf).map Store.address=Separate.forwardFootprint c rf := by
  simp [forwardCode,Separate.forwardFootprint,phaseCode_addresses]
theorem backwardCode_addresses (c : Shape) (tree : SumTree) (ri : Bool) :
    ((backwardCode c tree ri).map Store.address).Perm
      (Separate.backwardFootprint c++if ri then Source.eventsInitial c else []) := by
  simp only [backwardCode,List.map_append,phaseCode_addresses,Separate.backwardFootprint]
  simpa only [List.append_assoc] using
    (List.Perm.append_left (Source.eventsDH c)
      (List.perm_append_comm (l₁ := if ri then Source.eventsInitial c else [])
        (l₂ := Source.eventsQKV c++Source.eventsSum c)))

theorem forward_stores_valid (c : Shape) (tree : SumTree) (rf : Bool) (s : Store)
    (hs : s∈forwardCode c tree rf) : s.address.Valid c ∧ s.address.region∉inputRegions := by
  have mem : s.address∈Separate.forwardFootprint c rf := by
    rw [←forwardCode_addresses c tree rf]
    exact List.mem_map.mpr ⟨s,hs,rfl⟩
  rcases List.mem_append.mp mem with hm|ho
  · exact ⟨Source.h_valid c rf _ hm,Source.write_not_input .h _ (Source.h_footprint c rf _ hm)⟩
  · exact ⟨Source.o_valid c _ ho,Source.write_not_input .o _ (Source.o_footprint c _ ho)⟩

theorem backward_stores_valid (c : Shape) (tree : SumTree) (ri : Bool) (s : Store)
    (hs : s∈backwardCode c tree ri) : s.address.Valid c ∧ s.address.region∉inputRegions := by
  simp only [backwardCode,List.mem_append] at hs
  have phase : ∀ p events,s∈phaseCode c tree p events → s.address∈events := by
    intro p events h
    rw [←phaseCode_addresses c tree p events]
    exact List.mem_map.mpr ⟨s,h,rfl⟩
  rcases hs with (hh|hq)|hr
  · have h := phase _ _ hh
    exact ⟨Source.dhAPI_valid c ri _ h,Source.write_not_input .dh _ (Source.initial_events_writes c ri _ h)⟩
  · have h := phase _ _ hq
    exact ⟨Source.qkv_valid c _ h,Source.write_not_input .qkv _ (Source.qkv_footprint c _ h)⟩
  · have h := phase _ _ hr
    exact ⟨Source.sum_valid c _ h,Source.write_not_input .sum _ (Source.sum_footprint c _ h)⟩

/-- This is allocator freshness against the full original backing, including
prefixes, tails, and original cotangents that Tensor.contiguous copied. -/
def FreshCaller (storage : Storage c) (caller : Contract.Tensor) : Prop :=
  ∀ r,storage.present r=Bool.true → r∉inputRegions → Contract.disjoint (storage.tensor r) caller

theorem caller_outside (storage : Storage c) (caller : Contract.Tensor) (fresh : FreshCaller storage caller)
    (a : Address) (valid : a.Valid c) (writable : a.region∉inputRegions)
    (live : storage.present a.region=Bool.true)
    (byte : Nat) (bound : byte<caller.bytes) : (caller.base+byte)/4≠storage.binding.address a := by
  obtain ⟨lo,hi,_⟩ := storage.cell_live a valid live
  have sep := fresh a.region live writable
  unfold Contract.disjoint at sep
  change (caller.base+byte)/4≠storage.base a.region+a.offset c
  have div := Nat.div_add_mod (caller.base+byte) 4
  have rem := Nat.mod_lt (caller.base+byte) (by decide : 0<4)
  omega

theorem complete_backing_frame (abi : MemoryABI σ) (ports : Numeric.ScalarPorts)
    (c : Shape) (storage : Storage c) (scale : Word) (code : List Store)
    (valid : ∀ s∈code,s.address.Valid c ∧ s.address.region∉inputRegions)
    (allocated : ∀ s∈code,storage.present s.address.region=Bool.true)
    (state : σ) (caller : Contract.Tensor) (fresh : FreshCaller storage caller) :
    ∀ byte,byte<caller.bytes →
      abi.bytes (nativeExecute abi ports c storage.binding scale code state) (caller.base+byte)=
        abi.bytes state (caller.base+byte) := by
  intro byte bound
  apply native_byte_frame
  intro s hs
  exact caller_outside storage caller fresh s.address (valid s hs).1 (valid s hs).2 (allocated s hs) byte bound

#axiomsClean native_reference
#axiomsClean native_byte_frame
#axiomsClean complete_backing_frame
end Checked
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001
