import bench.tritonbench_g.destindex_copy.Proofs.TypedBitCopy
import bench.tritonbench_g.kcache_copy_triton.RawBitCorrect

namespace VeriTile.Bench.TritonBenchG.DestindexCopy.RawBits
open VeriTile.Triton FP.Structural
open KcacheCopyTriton.RawBits
open KcacheCopyTriton.TypedCopy (copyBackend)
open TypedSource (Storage)
noncomputable section
set_option maxHeartbeats 1000000

def storage : Storage → KcacheCopyTriton.TypedCopy.Storage
  | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16

@[simp] theorem storage_tag (d : Storage) : (storage d).tag = TypedBits.tag d := by
  cases d <;> rfl

/-- Signed metadata retains the old unbounded Int/default-on-wrong-tag rules.
Negative final offsets retain Int.toNat clamping; no native ABI claim is made. -/
def readInt {R : Type} : RawCell R → Int
  | .int n => n
  | _ => 0
@[simp] theorem read_int_encode {R : Type} [Inhabited R] (c : RawCell R) :
    (encode c).read .int = readInt c := by
  cases c <;> simp [encode,Cell.read,readInt,defaultValue]

def destination {R : Type} (p : Layout) (m : Memory R) (ids : Nat → Nat) : Int :=
  readInt (m p.Dest_loc.cast (ids 0))
def inputNope (p : Layout) (ids : Nat → Nat) (i : TileIndex [1,p.BLOCK_DMODEL_NOPE]) : Nat :=
  ids 0*p.stride_kv_nope_bs+p.stride_kv_nope_d*i.2.1.val
def inputRope (p : Layout) (ids : Nat → Nat) (i : TileIndex [1,p.BLOCK_DMODEL_ROPE]) : Nat :=
  ids 0*p.stride_kv_rope_bs+p.stride_kv_rope_d*i.2.1.val

def outputNope {R : Type} (p : Layout) (m : Memory R) (ids : Nat → Nat)
    (i : TileIndex [1,p.BLOCK_DMODEL_NOPE]) : Nat :=
  (destination p m ids*(p.stride_o_nope_bs : Int)+(p.stride_o_nope_d*i.2.1.val : Nat)).toNat

def outputRope {R : Type} (p : Layout) (m : Memory R) (ids : Nat → Nat)
    (i : TileIndex [1,p.BLOCK_DMODEL_ROPE]) : Nat :=
  (destination p m ids*(p.stride_o_rope_bs : Int)+(p.stride_o_rope_d*i.2.1.val : Nat)).toNat

/-- Both payloads and all metadata use the independent initial raw memory.
The ordered NOPE then ROPE scatters preserve repeated addresses and aliases;
head counts/strides remain unused, and unmasked power-of-two padding is kept. -/
def referenceNope {R : Type} [Inhabited R] (nd : Storage) (p : Layout)
    (m : Memory R) (ids : Nat → Nat) : Memory R :=
  (TileShape.allIndices [1,p.BLOCK_DMODEL_NOPE]).foldl (fun out i =>
    write out p.O_nope (outputNope p m ids i)
      (readStorage (storage nd) (m p.KV_nope (inputNope p ids i)))) m

def reference {R : Type} [Inhabited R] (nd rd : Storage) (p : Layout)
    (m : Memory R) (ids : Nat → Nat) : Memory R :=
  (TileShape.allIndices [1,p.BLOCK_DMODEL_ROPE]).foldl (fun out i =>
    write out p.O_rope (outputRope p m ids i)
      (readStorage (storage rd) (m p.KV_rope (inputRope p ids i)))) (referenceNope nd p m ids)

@[simp] theorem destination_state {R : Type} [Inhabited R] (p : Layout)
    (m : Memory R) (seed : State (Carrier R)) :
    TypedBits.destination p (state m seed) = destination p m seed.pids := by
  simp [TypedBits.destination,destination,state,encodeMemory]
@[simp] theorem inputNope_state {R : Type} [Inhabited R] (p : Layout)
    (m : Memory R) (seed : State (Carrier R)) (i : TileIndex [1,p.BLOCK_DMODEL_NOPE]) :
    TypedBits.inputNope p (state m seed) i = inputNope p seed.pids i := rfl
@[simp] theorem inputRope_state {R : Type} [Inhabited R] (p : Layout)
    (m : Memory R) (seed : State (Carrier R)) (i : TileIndex [1,p.BLOCK_DMODEL_ROPE]) :
    TypedBits.inputRope p (state m seed) i = inputRope p seed.pids i := rfl
@[simp] theorem outputNope_state {R : Type} [Inhabited R] (p : Layout)
    (m : Memory R) (seed : State (Carrier R)) (i : TileIndex [1,p.BLOCK_DMODEL_NOPE]) :
    TypedBits.outputNope p (state m seed) i = outputNope p m seed.pids i := by
  simp only [TypedBits.outputNope,outputNope,destination_state]
@[simp] theorem outputRope_state {R : Type} [Inhabited R] (p : Layout)
    (m : Memory R) (seed : State (Carrier R)) (i : TileIndex [1,p.BLOCK_DMODEL_ROPE]) :
    TypedBits.outputRope p (state m seed) i = outputRope p m seed.pids i := by
  simp only [TypedBits.outputRope,outputRope,destination_state]
@[simp] theorem loadedNope_state {R : Type} [Inhabited R] (nd : Storage) (p : Layout)
    (m : Memory R) (seed : State (Carrier R)) (i : TileIndex [1,p.BLOCK_DMODEL_NOPE]) :
    TypedBits.loadedNope nd p (state m seed) i =
      encode (readStorage (storage nd) (m p.KV_nope (inputNope p seed.pids i))) := by
  unfold TypedBits.loadedNope
  rw [inputNope_state,← storage_tag]
  exact read_storage_encode _ _
@[simp] theorem loadedRope_state {R : Type} [Inhabited R] (rd : Storage) (p : Layout)
    (m : Memory R) (seed : State (Carrier R)) (i : TileIndex [1,p.BLOCK_DMODEL_ROPE]) :
    TypedBits.loadedRope rd p (state m seed) i =
      encode (readStorage (storage rd) (m p.KV_rope (inputRope p seed.pids i))) := by
  unfold TypedBits.loadedRope
  rw [inputRope_state,← storage_tag]
  exact read_storage_encode _ _

/-- Exact complete encoded memory, without injectivity or separation premises. -/
theorem execution_encoded_memory {R : Type} [Inhabited R]
    (rest : Algebra (Carrier R)) (nd rd : Storage) (p : Layout)
    (m : Memory R) (seed : State (Carrier R)) :
    (TypedBits.execution rest nd rd p (state m seed)).val.mem =
      encodeMemory (reference nd rd p m seed.pids) := by
  rw [TypedBits.execution_memory]
  unfold TypedBits.copied TypedBits.copiedNope reference referenceNope
  rw [KcacheCopyTriton.TypedCopy.scatter_memory,KcacheCopyTriton.TypedCopy.scatter_memory,
    encode_scatter,encode_scatter]
  simp only [outputNope_state,outputRope_state,loadedNope_state,loadedRope_state]
  rfl

theorem execution_decoded_memory {R : Type} [Inhabited R]
    (rest : Algebra (Carrier R)) (nd rd : Storage) (p : Layout)
    (m : Memory R) (seed : State (Carrier R)) :
    decodeMemory (TypedBits.execution rest nd rd p (state m seed)).val.mem = reference nd rd p m seed.pids := by
  rw [execution_encoded_memory,decode_encode_memory]

theorem execution_canonical {R : Type} [Inhabited R]
    (rest : Algebra (Carrier R)) (nd rd : Storage) (p : Layout)
    (m : Memory R) (seed : State (Carrier R)) (r : RegionName) (o : Nat) :
    Canonical ((TypedBits.execution rest nd rd p (state m seed)).val.mem r o) := by
  rw [execution_encoded_memory]
  exact ⟨reference nd rd p m seed.pids r o,rfl⟩

theorem nope_readback {R : Type} [Inhabited R]
    (rest : Algebra (Carrier R)) (nd rd : Storage) (p : Layout)
    (m : Memory R) (seed : State (Carrier R)) (i : TileIndex [1,p.BLOCK_DMODEL_NOPE])
    (hinj : Function.Injective (outputNope p m seed.pids)) (hsep : p.O_nope ≠ p.O_rope) :
    (TypedBits.execution rest nd rd p (state m seed)).val.mem p.O_nope (outputNope p m seed.pids i) =
      encode (readStorage (storage nd) (m p.KV_nope (inputNope p seed.pids i))) := by
  have h := TypedBits.nope_readback rest nd rd p (state m seed) i
    (by intro a b h; apply hinj; simpa only [outputNope_state] using h) hsep
  simpa only [outputNope_state,loadedNope_state] using h

theorem rope_readback {R : Type} [Inhabited R]
    (rest : Algebra (Carrier R)) (nd rd : Storage) (p : Layout)
    (m : Memory R) (seed : State (Carrier R)) (i : TileIndex [1,p.BLOCK_DMODEL_ROPE])
    (hinj : Function.Injective (outputRope p m seed.pids)) :
    (TypedBits.execution rest nd rd p (state m seed)).val.mem p.O_rope (outputRope p m seed.pids i) =
      encode (readStorage (storage rd) (m p.KV_rope (inputRope p seed.pids i))) := by
  have h := TypedBits.rope_readback rest nd rd p (state m seed) i
    (by intro a b h; apply hinj; simpa only [outputRope_state] using h)
  simpa only [outputRope_state,loadedRope_state] using h

theorem execution_frame {R : Type} [Inhabited R]
    (rest : Algebra (Carrier R)) (nd rd : Storage) (p : Layout)
    (m : Memory R) (seed : State (Carrier R)) (r : RegionName) (o : Nat)
    (hn : r ≠ p.O_nope ∨ ∀ i : TileIndex [1,p.BLOCK_DMODEL_NOPE], o ≠ outputNope p m seed.pids i)
    (hr : r ≠ p.O_rope ∨ ∀ i : TileIndex [1,p.BLOCK_DMODEL_ROPE], o ≠ outputRope p m seed.pids i) :
    (TypedBits.execution rest nd rd p (state m seed)).val.mem r o = encode (m r o) := by
  apply TypedBits.frame
  · simpa only [outputNope_state] using hn
  · simpa only [outputRope_state] using hr

/-- All numerical callbacks are unobserved by the complete memory result. -/
theorem backend_independent {R : Type} [Inhabited R]
    (rest₁ rest₂ : Algebra (Carrier R)) (nd rd : Storage) (p : Layout)
    (m : Memory R) (seed : State (Carrier R)) :
    (TypedBits.execution rest₁ nd rd p (state m seed)).val.mem =
      (TypedBits.execution rest₂ nd rd p (state m seed)).val.mem := by
  rw [execution_encoded_memory,execution_encoded_memory]

theorem registers_undef_independent {R : Type} [Inhabited R]
    (rest₁ rest₂ : Algebra (Carrier R)) (nd rd : Storage) (p : Layout)
    (m : Memory R) (seed₁ seed₂ : State (Carrier R)) (hids : seed₁.pids=seed₂.pids) :
    (TypedBits.execution rest₁ nd rd p (state m seed₁)).val.mem =
      (TypedBits.execution rest₂ nd rd p (state m seed₂)).val.mem := by
  rw [execution_encoded_memory,execution_encoded_memory,hids]

/-- Both readbacks use only the old address-side conditions. Whole-memory
execution and frame above need neither separation nor output injectivity. -/
theorem complete {R : Type} [Inhabited R] (nd rd : Storage) (p : Layout)
    (m : Memory R) (seed : State (Carrier R))
    (hn : Function.Injective (outputNope p m seed.pids))
    (hr : Function.Injective (outputRope p m seed.pids)) (hsep : p.O_nope ≠ p.O_rope) :
    ∃ final, FP.Structural.exec (copyBackend inert) (p.source nd rd Bool.false) (state m seed)=some final ∧
      final.mem=encodeMemory (reference nd rd p m seed.pids) ∧
      decodeMemory final.mem=reference nd rd p m seed.pids ∧
      (∀ i : TileIndex [1,p.BLOCK_DMODEL_NOPE], final.mem p.O_nope (outputNope p m seed.pids i) =
        encode (readStorage (storage nd) (m p.KV_nope (inputNope p seed.pids i)))) ∧
      (∀ i : TileIndex [1,p.BLOCK_DMODEL_ROPE], final.mem p.O_rope (outputRope p m seed.pids i) =
        encode (readStorage (storage rd) (m p.KV_rope (inputRope p seed.pids i)))) ∧
      (∀ r o,
        (r≠p.O_nope ∨ ∀ i : TileIndex [1,p.BLOCK_DMODEL_NOPE], o≠outputNope p m seed.pids i) →
        (r≠p.O_rope ∨ ∀ i : TileIndex [1,p.BLOCK_DMODEL_ROPE], o≠outputRope p m seed.pids i) →
        final.mem r o=encode (m r o)) ∧
      (∀ r o, Canonical (final.mem r o)) := by
  exact ⟨(TypedBits.execution inert nd rd p (state m seed)).val,
    (TypedBits.execution inert nd rd p (state m seed)).property,
    execution_encoded_memory inert nd rd p m seed,execution_decoded_memory inert nd rd p m seed,
    fun i => nope_readback inert nd rd p m seed i hn hsep,
    fun i => rope_readback inert nd rd p m seed i hr,
    execution_frame inert nd rd p m seed,execution_canonical inert nd rd p m seed⟩

#axiomsClean storage_tag
#axiomsClean read_int_encode
#axiomsClean destination_state
#axiomsClean inputNope_state
#axiomsClean inputRope_state
#axiomsClean outputNope_state
#axiomsClean outputRope_state
#axiomsClean loadedNope_state
#axiomsClean loadedRope_state
#axiomsClean execution_encoded_memory
#axiomsClean execution_decoded_memory
#axiomsClean execution_canonical
#axiomsClean nope_readback
#axiomsClean rope_readback
#axiomsClean execution_frame
#axiomsClean backend_independent
#axiomsClean registers_undef_independent
#axiomsClean complete
end
end VeriTile.Bench.TritonBenchG.DestindexCopy.RawBits
