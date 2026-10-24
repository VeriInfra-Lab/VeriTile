import bench.tritonbench_g.kcache_copy_triton.Proofs.SignedCopy
import bench.tritonbench_g.kcache_copy_triton.RawBitCorrect
namespace VeriTile.Bench.TritonBenchG.KcacheCopyTriton.SignedRawBits
open VeriTile.Triton FP.Structural
open TypedCopy (Storage)
open SignedCopy (Parameters)
open RawBits
noncomputable section
set_option maxHeartbeats 1000000

/-- All untouched registers, undef values and launch dimensions are arbitrary.
Only the memory representation is supplied independently. -/
def state {R : Type} [Inhabited R] (m : Memory R) (seed : State (Carrier R)) : State (Carrier R) :=
  { seed with mem := encodeMemory m }

def readInt {R : Type} : RawCell R → Int
  | .int n => n
  | _ => 0

@[simp] theorem read_int_encode {R : Type} [Inhabited R] (c : RawCell R) :
    (encode c).read .int = readInt c := by
  cases c <;> simp [encode, Cell.read, readInt, defaultValue]

def past {R : Type} (p : Parameters) (m : Memory R) (ids : Nat → Nat) : Int :=
  readInt (m p.lengths.cast (SignedCopy.sequence p ids)) + SignedCopy.tokenShift p ids

def blockId {R : Type} (p : Parameters) (m : Memory R) (ids : Nat → Nat) : Int :=
  readInt (m p.blockTables.cast (SignedCopy.sequence p ids*p.strideBts +
    SignedCopy.unsigned64 ((past p m ids).tdiv (p.blockSize : Int))*p.strideBtb))

def inputOffset (p : Parameters) (ids : Nat → Nat) (i : TileIndex [p.width]) : Nat :=
  ids 0*p.strideKt + ids 1*p.strideKh + (ids 2*p.width+i.1.val)*p.strideKd

def outputOffset {R : Type} (p : Parameters) (m : Memory R) (ids : Nat → Nat)
    (i : TileIndex [p.width]) : Nat :=
  SignedCopy.unsigned64 (blockId p m ids)*p.strideKcb + ids 1*p.strideKch + ids 2*p.strideKcsplit +
    SignedCopy.unsigned64 ((past p m ids).tmod (p.blockSize : Int))*p.strideKcs+i.1.val

/-- Complete initial-memory reference. Every source value and every metadata
lookup uses m, including when K/KCache or metadata/output aliases. Stores are
in the original allIndices order. No source interpreter or final state occurs
on the right side. width=0, zero strides and blockSize=0 retain the signed source model rules. -/
def reference {R : Type} [Inhabited R] (d : Storage) (p : Parameters)
    (m : Memory R) (ids : Nat → Nat) : Memory R :=
  (TileShape.allIndices [p.width]).foldl (fun out i =>
    write out p.KCache (outputOffset p m ids i)
      (readStorage d (m p.K (inputOffset p ids i)))) m

@[simp] theorem past_state {R : Type} [Inhabited R] (p : Parameters)
    (m : Memory R) (seed : State (Carrier R)) : SignedCopy.past p (state m seed) = past p m seed.pids := by
  simp [SignedCopy.past,past,state,encodeMemory]
@[simp] theorem blockId_state {R : Type} [Inhabited R] (p : Parameters)
    (m : Memory R) (seed : State (Carrier R)) : SignedCopy.blockId p (state m seed) = blockId p m seed.pids := by
  simp [SignedCopy.blockId,blockId,SignedCopy.past,past,state,encodeMemory]
@[simp] theorem inputOffset_state {R : Type} [Inhabited R] (p : Parameters)
    (m : Memory R) (seed : State (Carrier R)) (i : TileIndex [p.width]) :
    SignedCopy.inputOffset p (state m seed) i = inputOffset p seed.pids i := rfl
@[simp] theorem outputOffset_state {R : Type} [Inhabited R] (p : Parameters)
    (m : Memory R) (seed : State (Carrier R)) (i : TileIndex [p.width]) :
    SignedCopy.outputOffset p (state m seed) i = outputOffset p m seed.pids i := by
  simp only [SignedCopy.outputOffset,blockId_state,past_state]
  rfl

@[simp] theorem loaded_state {R : Type} [Inhabited R] (d : Storage) (p : Parameters)
    (m : Memory R) (seed : State (Carrier R)) (i : TileIndex [p.width]) :
    SignedCopy.loaded d p (state m seed) i = encode (readStorage d (m p.K (inputOffset p seed.pids i))) := by
  unfold SignedCopy.loaded
  rw [inputOffset_state]
  exact read_storage_encode d _

/-- Encoding commutes with the whole word-memory scatter, including repeated
addresses. Injectivity is not needed for this execution-level equation. -/
theorem encode_scatter {R I : Type} [Inhabited R] (xs : List I) (r : RegionName)
    (off : I → Nat) (value : I → RawCell R) (m : Memory R) :
    encodeMemory (xs.foldl (fun out i => write out r (off i) (value i)) m) =
      xs.foldl (fun out i => fun r' o' =>
        if r'=r ∧ o'=off i then encode (value i) else out r' o') (encodeMemory m) := by
  induction xs generalizing m with
  | nil => rfl
  | cons i xs ih =>
    simp only [List.foldl_cons,ih,encode_write]

/-- The actual full typed source execution lands in the exact encoded image.
This is stronger than comparing low payload bits after lossy decoding. -/
theorem execution_encoded_memory {R : Type} [Inhabited R]
    (rest : Algebra (Carrier R)) (d : Storage) (p : Parameters)
    (m : Memory R) (seed : State (Carrier R)) :
    (SignedCopy.execution rest d p (state m seed)).val.mem =
      encodeMemory (reference d p m seed.pids) := by
  rw [SignedCopy.execution_memory]
  unfold SignedCopy.copied reference
  rw [TypedCopy.scatter_memory,encode_scatter]
  simp only [outputOffset_state,loaded_state]
  rfl

theorem execution_decoded_memory {R : Type} [Inhabited R]
    (rest : Algebra (Carrier R)) (d : Storage) (p : Parameters)
    (m : Memory R) (seed : State (Carrier R)) :
    decodeMemory (SignedCopy.execution rest d p (state m seed)).val.mem = reference d p m seed.pids := by
  rw [execution_encoded_memory,decode_encode_memory]

theorem execution_canonical {R : Type} [Inhabited R]
    (rest : Algebra (Carrier R)) (d : Storage) (p : Parameters)
    (m : Memory R) (seed : State (Carrier R)) (r : RegionName) (o : Nat) :
    Canonical ((SignedCopy.execution rest d p (state m seed)).val.mem r o) := by
  rw [execution_encoded_memory]
  exact ⟨reference d p m seed.pids r o,rfl⟩

theorem execution_readback {R : Type} [Inhabited R]
    (rest : Algebra (Carrier R)) (d : Storage) (p : Parameters)
    (m : Memory R) (seed : State (Carrier R)) (i : TileIndex [p.width]) :
    (SignedCopy.execution rest d p (state m seed)).val.mem p.KCache (outputOffset p m seed.pids i) =
      encode (readStorage d (m p.K (inputOffset p seed.pids i))) := by
  have h := SignedCopy.readback rest d p (state m seed) i
  simpa only [outputOffset_state,inputOffset_state,show (state m seed).mem=encodeMemory m from rfl,
    encodeMemory,read_storage_encode] using h

theorem execution_frame {R : Type} [Inhabited R]
    (rest : Algebra (Carrier R)) (d : Storage) (p : Parameters)
    (m : Memory R) (seed : State (Carrier R)) (r : RegionName) (o : Nat)
    (hmiss : r ≠ p.KCache ∨ ∀ i : TileIndex [p.width], o ≠ outputOffset p m seed.pids i) :
    (SignedCopy.execution rest d p (state m seed)).val.mem r o = encode (m r o) := by
  apply SignedCopy.frame
  simpa only [outputOffset_state] using hmiss

/-- No backend callback is left in the observed complete memory result. -/
theorem backend_independent {R : Type} [Inhabited R]
    (rest₁ rest₂ : Algebra (Carrier R)) (d : Storage) (p : Parameters)
    (m : Memory R) (seed : State (Carrier R)) :
    (SignedCopy.execution rest₁ d p (state m seed)).val.mem =
      (SignedCopy.execution rest₂ d p (state m seed)).val.mem := by
  rw [execution_encoded_memory,execution_encoded_memory]

/-- Arbitrary initial register contents, undef function and numPids have no
impact on this unmasked stage. Only the supplied memory and pids are observed. -/
theorem registers_undef_independent {R : Type} [Inhabited R]
    (rest₁ rest₂ : Algebra (Carrier R)) (d : Storage) (p : Parameters)
    (m : Memory R) (seed₁ seed₂ : State (Carrier R)) (hids : seed₁.pids=seed₂.pids) :
    (SignedCopy.execution rest₁ d p (state m seed₁)).val.mem =
      (SignedCopy.execution rest₂ d p (state m seed₂)).val.mem := by
  rw [execution_encoded_memory,execution_encoded_memory,hids]

/-- Full raw-state success/readback/frame theorem, including all six legacy
Storage constructors; R remains explicit for the erased `.real` branch. -/
theorem complete {R : Type} [Inhabited R] (d : Storage) (p : Parameters)
    (m : Memory R) (seed : State (Carrier R)) :
    ∃ final, FP.Structural.exec (TypedCopy.copyBackend inert) (SignedCopy.source d p)
      (state m seed) = some final ∧
      final.mem=encodeMemory (reference d p m seed.pids) ∧
      decodeMemory final.mem=reference d p m seed.pids ∧
      (∀ i : TileIndex [p.width], final.mem p.KCache (outputOffset p m seed.pids i) =
        encode (readStorage d (m p.K (inputOffset p seed.pids i)))) ∧
      (∀ r o, (r≠p.KCache ∨ ∀ i : TileIndex [p.width], o≠outputOffset p m seed.pids i) →
        final.mem r o=encode (m r o)) ∧
      (∀ r o, Canonical (final.mem r o)) := by
  exact ⟨(SignedCopy.execution inert d p (state m seed)).val,
    (SignedCopy.execution inert d p (state m seed)).property,
    execution_encoded_memory inert d p m seed,execution_decoded_memory inert d p m seed,
    execution_readback inert d p m seed,execution_frame inert d p m seed,
    execution_canonical inert d p m seed⟩

#axiomsClean encode_scatter
#axiomsClean execution_encoded_memory
#axiomsClean execution_decoded_memory
#axiomsClean execution_canonical
#axiomsClean execution_readback
#axiomsClean execution_frame
#axiomsClean backend_independent
#axiomsClean registers_undef_independent
#axiomsClean complete
end
end VeriTile.Bench.TritonBenchG.KcacheCopyTriton.SignedRawBits
