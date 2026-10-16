import bench.tritonbench_g.kv_cache_copy.Proofs.TypedCopy
import bench.tritonbench_g.kcache_copy_triton.RawBitCorrect

namespace VeriTile.Bench.TritonBenchG.KvCacheCopy.RawBits
open VeriTile.Triton FP.Structural
open KcacheCopyTriton.RawBits
open KcacheCopyTriton.TypedCopy (Storage copyBackend)
open TypedCopy (Parameters)
noncomputable section
set_option maxHeartbeats 1000000

/-- The V addresses use the same initial metadata as K. -/
def vInputOffset (p : Parameters) (ids : Nat → Nat) (i : TileIndex [p.width]) : Nat :=
  ids 0*p.strideVt + ids 1*p.strideVh + (p.splitX*p.width+i.1.val)*p.strideVd

def vOutputOffset {R : Type} (p : Parameters) (m : Memory R) (ids : Nat → Nat)
    (i : TileIndex [p.width]) : Nat :=
  blockId p.toParameters m ids*p.strideVcb + ids 1*p.strideVch +
    past p.toParameters m ids%p.blockSize*p.strideVcs +
    (p.splitX*p.width+i.1.val)*p.strideVcd

/-- First scatter; all K values and metadata come from the initial raw memory. -/
def referenceK {R : Type} [Inhabited R] (kd : Storage) (p : Parameters)
    (m : Memory R) (ids : Nat → Nat) : Memory R :=
  (TileShape.allIndices [p.width]).foldl (fun out i =>
    if TypedCopy.active p i then write out p.KCache (outputOffset p.toParameters m ids i)
      (readStorage kd (m p.K (inputOffset p.toParameters ids i))) else out) m

/-- Independent complete source reference: K writes precede V writes. Both
payloads and all metadata are read from m, including when KCache aliases V.
Neither source execution nor its result occurs in the reference definition. -/
def reference {R : Type} [Inhabited R] (kd vd : Storage) (p : Parameters)
    (m : Memory R) (ids : Nat → Nat) : Memory R :=
  (TileShape.allIndices [p.width]).foldl (fun out i =>
    if TypedCopy.active p i then write out p.VCache (vOutputOffset p m ids i)
      (readStorage vd (m p.V (vInputOffset p ids i))) else out) (referenceK kd p m ids)

@[simp] theorem vInputOffset_state {R : Type} [Inhabited R] (p : Parameters)
    (m : Memory R) (seed : State (Carrier R)) (i : TileIndex [p.width]) :
    TypedCopy.vInputOffset p (state m seed) i = vInputOffset p seed.pids i := rfl
@[simp] theorem vOutputOffset_state {R : Type} [Inhabited R] (p : Parameters)
    (m : Memory R) (seed : State (Carrier R)) (i : TileIndex [p.width]) :
    TypedCopy.vOutputOffset p (state m seed) i = vOutputOffset p m seed.pids i := by
  simp only [TypedCopy.vOutputOffset,blockId_state,past_state]
  rfl
@[simp] theorem vLoaded_state {R : Type} [Inhabited R] (vd : Storage) (p : Parameters)
    (m : Memory R) (seed : State (Carrier R)) (i : TileIndex [p.width]) :
    TypedCopy.vLoaded vd p (state m seed) i = encode (readStorage vd (m p.V (vInputOffset p seed.pids i))) := by
  unfold TypedCopy.vLoaded
  rw [vInputOffset_state]
  exact read_storage_encode vd _

/-- Masked scatter commutes with encoding for arbitrary order, aliases and
repeated addresses. Masked-off values are not used by the raw memory update. -/
theorem encode_masked_scatter {R I : Type} [Inhabited R] (xs : List I)
    (on : I → Prop) [DecidablePred on] (r : RegionName)
    (off : I → Nat) (value : I → RawCell R) (m : Memory R) :
    encodeMemory (xs.foldl (fun out i => if on i then write out r (off i) (value i) else out) m) =
      xs.foldl (fun out i => if on i then fun r' o' =>
        if r'=r ∧ o'=off i then encode (value i) else out r' o' else out) (encodeMemory m) := by
  induction xs generalizing m with
  | nil => rfl
  | cons i xs ih =>
    by_cases h : on i <;> simp only [List.foldl_cons,h,ite_true,ite_false,ih,encode_write]

/-- Exact whole encoded memory, without distinct-cache or injectivity premises.
If KCache=VCache, the last active V write wins as in the actual source. -/
theorem execution_encoded_memory {R : Type} [Inhabited R]
    (rest : Algebra (Carrier R)) (kd vd : Storage) (p : Parameters)
    (m : Memory R) (seed : State (Carrier R)) :
    (TypedCopy.execution rest kd vd p (state m seed)).val.mem =
      encodeMemory (reference kd vd p m seed.pids) := by
  rw [TypedCopy.execution_memory]
  unfold TypedCopy.copiedKV TypedCopy.copiedK reference referenceK
  rw [KcacheCopyTriton.TypedCopy.masked_scatter_memory,
    KcacheCopyTriton.TypedCopy.masked_scatter_memory,encode_masked_scatter,encode_masked_scatter]
  simp only [outputOffset_state,loaded_state,vOutputOffset_state,vLoaded_state]
  rfl

theorem execution_decoded_memory {R : Type} [Inhabited R]
    (rest : Algebra (Carrier R)) (kd vd : Storage) (p : Parameters)
    (m : Memory R) (seed : State (Carrier R)) :
    decodeMemory (TypedCopy.execution rest kd vd p (state m seed)).val.mem = reference kd vd p m seed.pids := by
  rw [execution_encoded_memory,decode_encode_memory]

theorem execution_canonical {R : Type} [Inhabited R]
    (rest : Algebra (Carrier R)) (kd vd : Storage) (p : Parameters)
    (m : Memory R) (seed : State (Carrier R)) (r : RegionName) (o : Nat) :
    Canonical ((TypedCopy.execution rest kd vd p (state m seed)).val.mem r o) := by
  rw [execution_encoded_memory]
  exact ⟨reference kd vd p m seed.pids r o,rfl⟩

theorem k_readback {R : Type} [Inhabited R]
    (rest : Algebra (Carrier R)) (kd vd : Storage) (p : Parameters)
    (m : Memory R) (seed : State (Carrier R)) (i : TileIndex [p.width]) (hi : TypedCopy.active p i)
    (hcache : p.KCache ≠ p.VCache) :
    (TypedCopy.execution rest kd vd p (state m seed)).val.mem p.KCache
      (outputOffset p.toParameters m seed.pids i) =
      encode (readStorage kd (m p.K (inputOffset p.toParameters seed.pids i))) := by
  have h := TypedCopy.k_readback rest kd vd p (state m seed) i hi hcache
  simpa only [outputOffset_state,inputOffset_state,show (state m seed).mem=encodeMemory m from rfl,
    encodeMemory,read_storage_encode] using h

theorem v_readback {R : Type} [Inhabited R]
    (rest : Algebra (Carrier R)) (kd vd : Storage) (p : Parameters)
    (m : Memory R) (seed : State (Carrier R)) (i : TileIndex [p.width]) (hi : TypedCopy.active p i)
    (hinj : ∀ a b, TypedCopy.active p a → TypedCopy.active p b → vOutputOffset p m seed.pids a = vOutputOffset p m seed.pids b → a=b) :
    (TypedCopy.execution rest kd vd p (state m seed)).val.mem p.VCache (vOutputOffset p m seed.pids i) =
      encode (readStorage vd (m p.V (vInputOffset p seed.pids i))) := by
  have h := TypedCopy.v_readback rest kd vd p (state m seed) i hi (by simpa only [vOutputOffset_state] using hinj)
  simpa only [vOutputOffset_state,vInputOffset_state,show (state m seed).mem=encodeMemory m from rfl,
    encodeMemory,read_storage_encode] using h

theorem execution_frame {R : Type} [Inhabited R]
    (rest : Algebra (Carrier R)) (kd vd : Storage) (p : Parameters)
    (m : Memory R) (seed : State (Carrier R)) (r : RegionName) (o : Nat)
    (hk : r ≠ p.KCache ∨ ∀ i : TileIndex [p.width], TypedCopy.active p i → o ≠ outputOffset p.toParameters m seed.pids i)
    (hv : r ≠ p.VCache ∨ ∀ i : TileIndex [p.width], TypedCopy.active p i → o ≠ vOutputOffset p m seed.pids i) :
    (TypedCopy.execution rest kd vd p (state m seed)).val.mem r o = encode (m r o) := by
  apply TypedCopy.frame
  · simpa only [outputOffset_state] using hk
  · simpa only [vOutputOffset_state] using hv

/-- No numerical callback affects final memory. Masked-off load literals and
casts may differ between backends; no active store observes those values. -/
theorem backend_independent {R : Type} [Inhabited R]
    (rest₁ rest₂ : Algebra (Carrier R)) (kd vd : Storage) (p : Parameters)
    (m : Memory R) (seed : State (Carrier R)) :
    (TypedCopy.execution rest₁ kd vd p (state m seed)).val.mem =
      (TypedCopy.execution rest₂ kd vd p (state m seed)).val.mem := by
  rw [execution_encoded_memory,execution_encoded_memory]

theorem registers_undef_independent {R : Type} [Inhabited R]
    (rest₁ rest₂ : Algebra (Carrier R)) (kd vd : Storage) (p : Parameters)
    (m : Memory R) (seed₁ seed₂ : State (Carrier R)) (hids : seed₁.pids=seed₂.pids) :
    (TypedCopy.execution rest₁ kd vd p (state m seed₁)).val.mem =
      (TypedCopy.execution rest₂ kd vd p (state m seed₂)).val.mem := by
  rw [execution_encoded_memory,execution_encoded_memory,hids]

/-- Full old selected masked x-block scope: only the original address-side
readback conditions, with all initial states and both dtype annotations. -/
theorem complete {R : Type} [Inhabited R] (kd vd : Storage) (p : Parameters)
    (m : Memory R) (seed : State (Carrier R)) (hcache : p.KCache ≠ p.VCache)
    (hinj : ∀ a b, TypedCopy.active p a → TypedCopy.active p b → vOutputOffset p m seed.pids a = vOutputOffset p m seed.pids b → a=b) :
    ∃ final, FP.Structural.exec (copyBackend inert) (TypedCopy.source kd vd p) (state m seed)=some final ∧
      final.mem=encodeMemory (reference kd vd p m seed.pids) ∧
      decodeMemory final.mem=reference kd vd p m seed.pids ∧
      (∀ i : TileIndex [p.width], TypedCopy.active p i → final.mem p.KCache (outputOffset p.toParameters m seed.pids i) =
        encode (readStorage kd (m p.K (inputOffset p.toParameters seed.pids i)))) ∧
      (∀ i : TileIndex [p.width], TypedCopy.active p i → final.mem p.VCache (vOutputOffset p m seed.pids i) =
        encode (readStorage vd (m p.V (vInputOffset p seed.pids i)))) ∧
      (∀ r o,
        (r≠p.KCache ∨ ∀ i : TileIndex [p.width], TypedCopy.active p i → o≠outputOffset p.toParameters m seed.pids i) →
        (r≠p.VCache ∨ ∀ i : TileIndex [p.width], TypedCopy.active p i → o≠vOutputOffset p m seed.pids i) →
        final.mem r o=encode (m r o)) ∧
      (∀ r o, Canonical (final.mem r o)) := by
  exact ⟨(TypedCopy.execution inert kd vd p (state m seed)).val,
    (TypedCopy.execution inert kd vd p (state m seed)).property,
    execution_encoded_memory inert kd vd p m seed,execution_decoded_memory inert kd vd p m seed,
    fun i hi => k_readback inert kd vd p m seed i hi hcache,
    fun i hi => v_readback inert kd vd p m seed i hi hinj,
    execution_frame inert kd vd p m seed,execution_canonical inert kd vd p m seed⟩

#axiomsClean encode_masked_scatter
#axiomsClean execution_encoded_memory
#axiomsClean execution_decoded_memory
#axiomsClean execution_canonical
#axiomsClean k_readback
#axiomsClean v_readback
#axiomsClean execution_frame
#axiomsClean backend_independent
#axiomsClean registers_undef_independent
#axiomsClean complete
end
end VeriTile.Bench.TritonBenchG.KvCacheCopy.RawBits
