import bench.tritonbench_g.masked_select.Proofs.TypedCopy
import bench.tritonbench_g.kv_cache_copy.Proofs.RawBitCopy
namespace VeriTile.Bench.TritonBenchG.MaskedSelect.RawBits
open VeriTile.Triton FP.Structural
open KcacheCopyTriton.RawBits
open KcacheCopyTriton.TypedCopy (copyBackend)
open TypedCopy (Storage Parameters)
noncomputable section
set_option maxHeartbeats 1000000

/-- Bool control and exact Int payload retain the old wrong-tag defaults. -/
def readBool {R : Type} : RawCell R → Bool
  | .bool b => b | _ => Bool.false

def readInt {R : Type} : RawCell R → Int
  | .int n => n | _ => 0

@[simp] theorem read_bool_encode {R : Type} [Inhabited R] (c : RawCell R) :
    (encode c).read .bool = readBool c := by
  cases c <;> simp [readBool,encode,Cell.read,defaultValue]
@[simp] theorem read_int_encode {R : Type} [Inhabited R] (c : RawCell R) :
    (encode c).read .int = readInt c := by
  cases c <;> simp [readInt,encode,Cell.read,defaultValue]

def readPayload {R : Type} [Inhabited R] : Storage → RawCell R → RawCell R
  | .fp16, c => readStorage .fp16 c
  | .bf16, c => readStorage .bf16 c
  | .fp32, c => readStorage .fp32 c
  | .fp64, c => readStorage .fp64 c
  | .f8e4, c => readStorage .f8e4 c
  | .f8e5, c => readStorage .f8e5 c
  | .int64, c => .int (readInt c)

@[simp] theorem read_payload_encode {R : Type} [Inhabited R] (d : Storage) (c : RawCell R) :
    Cell.mk d.tag ((encode c).read d.tag) = encode (readPayload d c) := by
  cases d with
  | fp16 => exact read_storage_encode .fp16 c
  | bf16 => exact read_storage_encode .bf16 c
  | fp32 => exact read_storage_encode .fp32 c
  | fp64 => exact read_storage_encode .fp64 c
  | f8e4 => exact read_storage_encode .f8e4 c
  | f8e5 => exact read_storage_encode .f8e5 c
  | int64 =>
    change Cell.mk .int ((encode c).read .int)=Cell.mk .int (readInt c)
    rw [read_int_encode]

def inputOffset (p : Parameters) (ids : Nat → Nat) (i : TileIndex [p.width]) : Nat :=
  ids 0*p.width+i.1.val

def outputOffset {R : Type} (p : Parameters) (m : Memory R) (ids : Nat → Nat)
    (i : TileIndex [p.width]) : Nat :=
  readNat (m p.prefixSum.cast (inputOffset p ids i))-1

def selected {R : Type} (p : Parameters) (m : Memory R) (ids : Nat → Nat)
    (i : TileIndex [p.width]) : Prop :=
  readBool (m p.selection.cast (inputOffset p ids i))=Bool.true ∧ inputOffset p ids i<p.size
instance {R : Type} (p : Parameters) (m : Memory R) (ids : Nat → Nat) :
    DecidablePred (selected p m ids) := fun _ => inferInstanceAs (Decidable (_ ∧ _))

/-- Every mask, prefix entry and payload is an initial-memory read. Prefix
subtraction is exactly the previous Nat subtraction, including zero entries.
The host prefix producer and finite int64 overflow semantics are not asserted. -/
def reference {R : Type} [Inhabited R] (d : Storage) (p : Parameters)
    (m : Memory R) (ids : Nat → Nat) : Memory R :=
  (TileShape.allIndices [p.width]).foldl (fun out i =>
    if selected p m ids i then write out p.output (outputOffset p m ids i)
      (readPayload d (m p.input (inputOffset p ids i))) else out) m

@[simp] theorem inputOffset_state {R : Type} [Inhabited R] (p : Parameters)
    (m : Memory R) (seed : State (Carrier R)) (i : TileIndex [p.width]) :
    TypedCopy.inputOffset p (state m seed) i = inputOffset p seed.pids i := rfl
@[simp] theorem outputOffset_state {R : Type} [Inhabited R] (p : Parameters)
    (m : Memory R) (seed : State (Carrier R)) (i : TileIndex [p.width]) :
    TypedCopy.outputOffset p (state m seed) i = outputOffset p m seed.pids i := by
  simp [TypedCopy.outputOffset,outputOffset,TypedCopy.inputOffset,inputOffset,state,encodeMemory]
@[simp] theorem selected_state {R : Type} [Inhabited R] (p : Parameters)
    (m : Memory R) (seed : State (Carrier R)) (i : TileIndex [p.width]) :
    TypedCopy.active p (state m seed) i ↔ selected p m seed.pids i := by
  simp [TypedCopy.active,selected,TypedCopy.inputOffset,inputOffset,state,encodeMemory]
@[simp] theorem loaded_state {R : Type} [Inhabited R] (d : Storage) (p : Parameters)
    (m : Memory R) (seed : State (Carrier R)) (i : TileIndex [p.width]) :
    TypedCopy.loaded d p (state m seed) i = encode (readPayload d (m p.input (inputOffset p seed.pids i))) := by
  unfold TypedCopy.loaded
  rw [inputOffset_state]
  exact read_payload_encode d _

/-- Entire output memory, even when input, mask, prefix and output regions
alias. There is no address-injectivity condition on this sequential reference. -/
theorem execution_encoded_memory {R : Type} [Inhabited R]
    (rest : Algebra (Carrier R)) (d : Storage) (p : Parameters)
    (m : Memory R) (seed : State (Carrier R)) :
    (TypedCopy.execution rest d p (state m seed)).val.mem = encodeMemory (reference d p m seed.pids) := by
  rw [TypedCopy.execution_memory]
  unfold TypedCopy.copied reference
  rw [KcacheCopyTriton.TypedCopy.masked_scatter_memory,KvCacheCopy.RawBits.encode_masked_scatter]
  simp only [outputOffset_state,loaded_state,selected_state]
  rfl

theorem execution_decoded_memory {R : Type} [Inhabited R]
    (rest : Algebra (Carrier R)) (d : Storage) (p : Parameters)
    (m : Memory R) (seed : State (Carrier R)) :
    decodeMemory (TypedCopy.execution rest d p (state m seed)).val.mem = reference d p m seed.pids := by
  rw [execution_encoded_memory,decode_encode_memory]

theorem execution_canonical {R : Type} [Inhabited R]
    (rest : Algebra (Carrier R)) (d : Storage) (p : Parameters)
    (m : Memory R) (seed : State (Carrier R)) (r : RegionName) (o : Nat) :
    Canonical ((TypedCopy.execution rest d p (state m seed)).val.mem r o) := by
  rw [execution_encoded_memory]
  exact ⟨reference d p m seed.pids r o,rfl⟩

theorem readback {R : Type} [Inhabited R]
    (rest : Algebra (Carrier R)) (d : Storage) (p : Parameters)
    (m : Memory R) (seed : State (Carrier R)) (i : TileIndex [p.width]) (hi : selected p m seed.pids i)
    (hinj : ∀ a b, selected p m seed.pids a → selected p m seed.pids b →
      outputOffset p m seed.pids a = outputOffset p m seed.pids b → a=b) :
    (TypedCopy.execution rest d p (state m seed)).val.mem p.output (outputOffset p m seed.pids i) =
      encode (readPayload d (m p.input (inputOffset p seed.pids i))) := by
  have h := TypedCopy.readback rest d p (state m seed) i (by simpa only [selected_state] using hi)
    (by simpa only [selected_state,outputOffset_state] using hinj)
  simpa only [outputOffset_state,inputOffset_state,show (state m seed).mem=encodeMemory m from rfl,
    encodeMemory,read_payload_encode] using h

theorem execution_frame {R : Type} [Inhabited R]
    (rest : Algebra (Carrier R)) (d : Storage) (p : Parameters)
    (m : Memory R) (seed : State (Carrier R)) (r : RegionName) (o : Nat)
    (hmiss : r≠p.output ∨ ∀ i : TileIndex [p.width], selected p m seed.pids i → o≠outputOffset p m seed.pids i) :
    (TypedCopy.execution rest d p (state m seed)).val.mem r o = encode (m r o) := by
  apply TypedCopy.frame
  simpa only [selected_state,outputOffset_state] using hmiss

theorem backend_independent {R : Type} [Inhabited R]
    (rest₁ rest₂ : Algebra (Carrier R)) (d : Storage) (p : Parameters)
    (m : Memory R) (seed : State (Carrier R)) :
    (TypedCopy.execution rest₁ d p (state m seed)).val.mem =
      (TypedCopy.execution rest₂ d p (state m seed)).val.mem := by
  rw [execution_encoded_memory,execution_encoded_memory]

theorem registers_undef_independent {R : Type} [Inhabited R]
    (rest₁ rest₂ : Algebra (Carrier R)) (d : Storage) (p : Parameters)
    (m : Memory R) (seed₁ seed₂ : State (Carrier R)) (hids : seed₁.pids=seed₂.pids) :
    (TypedCopy.execution rest₁ d p (state m seed₁)).val.mem =
      (TypedCopy.execution rest₂ d p (state m seed₂)).val.mem := by
  rw [execution_encoded_memory,execution_encoded_memory,hids]

theorem complete {R : Type} [Inhabited R] (d : Storage) (p : Parameters)
    (m : Memory R) (seed : State (Carrier R))
    (hinj : ∀ a b, selected p m seed.pids a → selected p m seed.pids b →
      outputOffset p m seed.pids a = outputOffset p m seed.pids b → a=b) :
    ∃ final, FP.Structural.exec (copyBackend inert) (TypedCopy.source d p) (state m seed)=some final ∧
      final.mem=encodeMemory (reference d p m seed.pids) ∧
      decodeMemory final.mem=reference d p m seed.pids ∧
      (∀ i : TileIndex [p.width], selected p m seed.pids i →
        final.mem p.output (outputOffset p m seed.pids i) =
        encode (readPayload d (m p.input (inputOffset p seed.pids i)))) ∧
      (∀ r o, (r≠p.output ∨ ∀ i : TileIndex [p.width], selected p m seed.pids i → o≠outputOffset p m seed.pids i) →
        final.mem r o=encode (m r o)) ∧
      (∀ r o, Canonical (final.mem r o)) := by
  exact ⟨(TypedCopy.execution inert d p (state m seed)).val,
    (TypedCopy.execution inert d p (state m seed)).property,
    execution_encoded_memory inert d p m seed,execution_decoded_memory inert d p m seed,
    fun i hi => readback inert d p m seed i hi hinj,
    execution_frame inert d p m seed,execution_canonical inert d p m seed⟩

#axiomsClean read_bool_encode
#axiomsClean read_int_encode
#axiomsClean read_payload_encode
#axiomsClean execution_encoded_memory
#axiomsClean execution_decoded_memory
#axiomsClean execution_canonical
#axiomsClean readback
#axiomsClean execution_frame
#axiomsClean backend_independent
#axiomsClean registers_undef_independent
#axiomsClean complete
end
end VeriTile.Bench.TritonBenchG.MaskedSelect.RawBits
