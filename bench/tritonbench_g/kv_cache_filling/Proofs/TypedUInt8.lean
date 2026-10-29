import bench.tritonbench_g.kv_cache_filling.KvCacheFilling
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy
import VeriTile.Triton.Float.IntegerConversionTotal

/-!
The originally accepted K/V tile stages at selected SIDX/BIDX/KV_BLOCK_IDX,
with the actual supplied-test storage profile: fp32 states and uint8 caches.
An explicit final conversion represents Python's implicit destination-store
conversion. The prefix bridge records the difference from the old Real model.
Undefined conversions remain arbitrary and disappear from all observations
under the active-input domain. Padding and all original dimensions remain.
-/
namespace VeriTile.Bench.TritonBenchG.KvCacheFilling.TypedUInt8
open VeriTile.Triton FP.Structural FP.IntegerConversion
open KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
abbrev Word := FP.Value .fp32
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

structure Parameters where
  (input output : RegionName)
  blockOffsets : Region .nat
  (sourceIndex blockIndex kvBlockIndex strideSs strideSh strideSd
    strideCn strideCb strideCh strideCd strideBoff heads dimension height width : Nat)

def keySource (p : Parameters) : ComputeKernel :=
  let KStates := p.input
  let KCaches := p.output
  let BlockOffsets := p.blockOffsets
  triton {
  tl.region KStates = tl.float32, KCaches = tl.uint8
  batch_id = tl.program_id(0)
  h_off = tl.arange(0, $(p.height))
  d_off = tl.arange(0, $(p.width))
  block_off = tl.load(BlockOffsets + batch_id * $(p.strideBoff) + $(p.kvBlockIndex))
  mask = (h_off[:, None] < $(p.heads)) & (d_off[None, :] < $(p.dimension))
  k = tl.load(KStates + $(p.sourceIndex) * $(p.strideSs) +
      h_off[:, None] * $(p.strideSh) + d_off[None, :] * $(p.strideSd), mask=mask, other=0.0)
  tl.store(KCaches + block_off * $(p.strideCn) + $(p.blockIndex) * $(p.strideCb) +
      h_off[:, None] * $(p.strideCh) + d_off[None, :] * $(p.strideCd), k.to(tl.uint8), mask=mask)
}

def valueSource (p : Parameters) : ComputeKernel :=
  let VStates := p.input
  let VCaches := p.output
  let BlockOffsets := p.blockOffsets
  triton {
  tl.region VStates = tl.float32, VCaches = tl.uint8
  batch_id = tl.program_id(0)
  h_off = tl.arange(0, $(p.height))
  dv_off = tl.arange(0, $(p.width))
  block_off = tl.load(BlockOffsets + batch_id * $(p.strideBoff) + $(p.kvBlockIndex))
  maskv = (h_off[:, None] < $(p.heads)) & (dv_off[None, :] < $(p.dimension))
  v = tl.load(VStates + $(p.sourceIndex) * $(p.strideSs) +
      h_off[:, None] * $(p.strideSh) + dv_off[None, :] * $(p.strideSd), mask=maskv, other=0.0)
  tl.store(VCaches + block_off * $(p.strideCn) + $(p.blockIndex) * $(p.strideCb) +
      h_off[:, None] * $(p.strideCh) + dv_off[None, :] * $(p.strideCd), v.to(tl.uint8), mask=maskv)
}

def source (isValue : Bool) : Parameters → ComputeKernel :=
  if isValue then valueSource else keySource

def active (p : Parameters) (i : TileIndex [p.height, p.width]) : Prop :=
  i.1.val < p.heads ∧ i.2.1.val < p.dimension
instance (p : Parameters) : DecidablePred (active p) := fun _ => inferInstanceAs (Decidable (_ ∧ _))

def inputOffset (p : Parameters) (i : TileIndex [p.height, p.width]) : Nat :=
  p.sourceIndex * p.strideSs + i.1.val * p.strideSh + i.2.1.val * p.strideSd

def blockOffset (p : Parameters) (s : State Word) : Nat :=
  (s.mem p.blockOffsets.cast (s.pids 0 * p.strideBoff + p.kvBlockIndex)).read .nat

def outputOffset (p : Parameters) (s : State Word) (i : TileIndex [p.height, p.width]) : Nat :=
  blockOffset p s * p.strideCn + p.blockIndex * p.strideCb +
    i.1.val * p.strideCh + i.2.1.val * p.strideCd

def inputValue (p : Parameters) (s : State Word) (i : TileIndex [p.height, p.width]) : Word :=
  (s.mem p.input (inputOffset p i)).read .real

/-- Finite decoded input and a representable truncated uint8 result are the
actual per-active-lane numeric domain. No condition is imposed on padding. -/
def InputDefined (p : Parameters) (s : State Word) : Prop :=
  ∀ i, active p i → (toUnsigned 8 (inputValue p s i)).isSome

/-- Independent value target from the source bits. The default is unreachable
on active lanes under InputDefined; it is not an undefined hardware policy. -/
def expected (p : Parameters) (s : State Word) (i : TileIndex [p.height, p.width]) : Nat :=
  (toUnsigned 8 (inputValue p s i)).getD 0

def copied (p : Parameters) (s : State Word) : State Word :=
  (TileShape.allIndices [p.height, p.width]).foldl (fun t i =>
    if active p i then t.write p.output (outputOffset p s i) (.mk .nat (expected p s i)) else t) s

noncomputable def backend (rest : Algebra Word) (undefinedValue : Nat → Word → Nat) : Algebra Word :=
  fp32TotalBackend (FP.Software.scalarFP32 rest) (fun _ _ => 0) undefinedValue

/-- The undefined policy is universal, not a fixed GPU return value. -/
def execution (rest : Algebra Word) (undefinedValue : Nat → Word → Nat)
    (isValue : Bool) (p : Parameters) (s : State Word) :
    {t : State Word // FP.Structural.exec (backend rest undefinedValue) (source isValue p) s = some t} := by
  cases isValue <;>
    (refine ⟨?_, ?_⟩
     swap
     simp [source, keySource, valueSource, FP.Structural.exec, ComputeKernel.surfaceBody,
       FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr,
       FP.Structural.evalComputeOp, FP.Structural.evalOp_unfold,
       FP.Structural.store, FP.Structural.numeric, FP.Structural.bop, FP.Structural.natLt,
       backend, fp32TotalBackend, fp32Source, FP.Software.scalarFP32,
       FP.Software.fp32Context, ComputeDType.eraseDType,
       FP.Structural.ofFloat, FP.Structural.toFloat, Region.cast, Option.bind]
     try rfl)

theorem conversion_defined (undefinedValue : Word → Nat) (x : Word)
    (h : (toUnsigned 8 x).isSome) :
    totalUnsigned undefinedValue 8 x = (toUnsigned 8 x).getD 0 := by
  cases hx : toUnsigned 8 x with
  | none => simp [hx] at h
  | some z => simp [totalUnsigned, hx]

theorem execution_memory (rest : Algebra Word) (undefinedValue : Nat → Word → Nat)
    (isValue : Bool) (p : Parameters) (s : State Word) (h : InputDefined p s) :
    (execution rest undefinedValue isValue p s).val.mem = (copied p s).mem := by
  cases isValue
  all_goals simp +contextual [execution, copied, masked_scatter_memory, active,
    inputOffset, outputOffset, blockOffset, expected, inputValue,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
    show (0.0 : ℝ) = 0 by norm_num]
  all_goals
    congr 1
    funext mem i
    by_cases hi : active p i
    · have hc := conversion_defined (undefinedValue 8) (inputValue p s i) (h i hi)
      simp only [inputValue, inputOffset] at hc
      simp only [active] at hi
      simp only [hi, ↓reduceIte, hc]
    · simp only [active] at hi
      simp only [hi, ↓reduceIte]

theorem expected_spec (p : Parameters) (s : State Word) (h : InputDefined p s)
    (i : TileIndex [p.height, p.width]) (hi : active p i) :
    ∃ q, (inputValue p s i).toRat? = some q ∧
      TowardZero q (expected p s i : Int) ∧ UnsignedRange 8 (expected p s i : Int) := by
  have hd := h i hi
  cases hx : toUnsigned 8 (inputValue p s i) with
  | none => simp [hx] at hd
  | some z =>
      simpa [expected, hx] using (toUnsigned_spec 8 (inputValue p s i) z).mp hx

theorem readback (rest : Algebra Word) (undefinedValue : Nat → Word → Nat)
    (isValue : Bool) (p : Parameters) (s : State Word) (h : InputDefined p s)
    (i : TileIndex [p.height, p.width]) (hi : active p i)
    (hinj : ∀ a b, active p a → active p b → outputOffset p s a = outputOffset p s b → a = b) :
    (execution rest undefinedValue isValue p s).val.mem p.output (outputOffset p s i) =
      .mk .nat (expected p s i) := by
  rw [execution_memory rest undefinedValue isValue p s h]
  exact masked_readback_active _ _ _ _ _ i hi hinj

theorem frame (rest : Algebra Word) (undefinedValue : Nat → Word → Nat)
    (isValue : Bool) (p : Parameters) (s : State Word) (h : InputDefined p s)
    (r : RegionName) (o : Nat)
    (hmiss : r ≠ p.output ∨ ∀ i : TileIndex [p.height, p.width], active p i → o ≠ outputOffset p s i) :
    (execution rest undefinedValue isValue p s).val.mem r o = s.mem r o := by
  rw [execution_memory rest undefinedValue isValue p s h]
  apply State.masked_scatter_frame
  exact hmiss.imp id (fun h i _ => h i)

/-- All observable memory is independent of undefined conversions in masked
lanes, even if they contain NaNs, infinities or out-of-range bit patterns. -/
theorem undefined_policy_independent (rest : Algebra Word)
    (u v : Nat → Word → Nat) (isValue : Bool) (p : Parameters) (s : State Word)
    (h : InputDefined p s) :
    (execution rest u isValue p s).val.mem = (execution rest v isValue p s).val.mem := by
  rw [execution_memory rest u isValue p s h, execution_memory rest v isValue p s h]

theorem correctness (rest : Algebra Word) (undefinedValue : Nat → Word → Nat)
    (isValue : Bool) (p : Parameters) (s : State Word) (h : InputDefined p s)
    (hinj : ∀ a b, active p a → active p b → outputOffset p s a = outputOffset p s b → a = b) :
    ∃ t, FP.Structural.exec (backend rest undefinedValue) (source isValue p) s = some t ∧
      (∀ i : TileIndex [p.height, p.width], active p i →
        t.mem p.output (outputOffset p s i) = .mk .nat (expected p s i) ∧
        ∃ q, (inputValue p s i).toRat? = some q ∧
          TowardZero q (expected p s i : Int) ∧ UnsignedRange 8 (expected p s i : Int)) ∧
      (∀ r o, (r ≠ p.output ∨ ∀ i : TileIndex [p.height, p.width], active p i →
          o ≠ outputOffset p s i) → t.mem r o = s.mem r o) :=
  ⟨(execution rest undefinedValue isValue p s).val,
    (execution rest undefinedValue isValue p s).property,
    fun i hi => ⟨readback rest undefinedValue isValue p s h i hi hinj, expected_spec p s h i hi⟩,
    frame rest undefinedValue isValue p s h⟩

#axiomsClean execution
#axiomsClean execution_memory
#axiomsClean expected_spec
#axiomsClean readback
#axiomsClean frame
#axiomsClean undefined_policy_independent
#axiomsClean correctness
end
end VeriTile.Bench.TritonBenchG.KvCacheFilling.TypedUInt8
