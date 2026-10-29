import bench.tritonbench_g.masked_select.RawBitCorrect
namespace VeriTile.Bench.TritonBenchG.MaskedSelect.RawBits
open VeriTile.Triton FP.Structural
open KcacheCopyTriton.RawBits
open KcacheCopyTriton.TypedCopy (copyBackend)
open TypedCopy (Storage Parameters)
noncomputable section

/-- output may be input, prefix, selection or a separate region. -/
def snapshotParameters (out : RegionName) : Parameters where
  input := "input"
  output := out
  selection := "selection"
  prefixSum := "prefix"
  size := 3
  width := 4

def snapshotMemory (values : Nat → RawCell (BitVec 32)) : Memory (BitVec 32) := fun r o =>
  if r="selection" then .bool (decide (o ≠ 1))
  else if r="prefix" then .nat (if o ≤ 1 then 3 else 4)
  else if r="input" then values o
  else .int 17

def snapshotSeed : State (Carrier (BitVec 32)) where
  mem := fun _ _ => .mk .nat 55
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 29
  undef := fun d _ _ => defaultValue d

private theorem snapshot_selected (out : RegionName) (values : Nat → RawCell (BitVec 32))
    (i : TileIndex [(snapshotParameters out).width])
    (h : selected (snapshotParameters out) (snapshotMemory values) snapshotSeed.pids i) :
    i.1.val=0 ∨ i.1.val=2 := by
  simp [selected,inputOffset,snapshotParameters,snapshotMemory,snapshotSeed,readBool,Region.cast] at h
  have hi := i.1.isLt
  simp [snapshotParameters] at hi
  omega

private theorem snapshot_injective (out : RegionName) (values : Nat → RawCell (BitVec 32)) :
    ∀ a b, selected (snapshotParameters out) (snapshotMemory values) snapshotSeed.pids a →
      selected (snapshotParameters out) (snapshotMemory values) snapshotSeed.pids b →
      outputOffset (snapshotParameters out) (snapshotMemory values) snapshotSeed.pids a =
      outputOffset (snapshotParameters out) (snapshotMemory values) snapshotSeed.pids b → a=b := by
  intro a b ha hb h
  have hv : a.1.val=b.1.val := by
    rcases snapshot_selected out values a ha with ha0 | ha2 <;>
      rcases snapshot_selected out values b hb with hb0 | hb2 <;>
      simp_all [outputOffset,inputOffset,snapshotParameters,snapshotMemory,snapshotSeed,readNat,Region.cast]
  exact Prod.ext (Fin.ext hv) (Subsingleton.elim _ _)

def floatInputs : Nat → RawCell (BitVec 32) := fun o =>
  .real (if o=0 then 0x7f800001 else 0x80000000)

/-- Two selected lanes (0,2), unselected lane 1 with repeated prefix, and an
out-of-bounds but true-mask lane 3. The first write goes to address 2, before
lane 2 writes address 3. Setting out=input/prefix/selection separately tests
snapshot preservation of payload, destination metadata and mask respectively.
All three would fail under progressive reads after each previous store. -/
theorem aliased_control_payload_source_success (out : RegionName) :
    ∃ final, FP.Structural.exec (copyBackend inert) (TypedCopy.source .fp32 (snapshotParameters out))
      (state (snapshotMemory floatInputs) snapshotSeed)=some final ∧
      decode (final.mem out 2)=RawCell.real (0x7f800001 : BitVec 32) ∧
      decode (final.mem out 3)=RawCell.real (0x80000000 : BitVec 32) ∧
      final.mem out 0=encode (snapshotMemory floatInputs out 0) ∧
      final.mem out 1=encode (snapshotMemory floatInputs out 1) := by
  let final := (TypedCopy.execution inert .fp32 (snapshotParameters out)
    (state (snapshotMemory floatInputs) snapshotSeed)).val
  refine ⟨final,(TypedCopy.execution inert .fp32 (snapshotParameters out)
    (state (snapshotMemory floatInputs) snapshotSeed)).property,?_,?_,?_,?_⟩
  · have h := congrArg decode (readback inert .fp32 (snapshotParameters out) (snapshotMemory floatInputs)
      snapshotSeed (⟨0,by simp [snapshotParameters,selected,inputOffset,snapshotMemory,snapshotSeed,readBool,Region.cast]⟩,PUnit.unit) (by simp [snapshotParameters,selected,inputOffset,snapshotMemory,snapshotSeed,readBool,Region.cast]) (snapshot_injective out floatInputs))
    simpa [final,snapshotParameters,snapshotMemory,snapshotSeed,floatInputs,inputOffset,outputOffset,
      readNat,readPayload,readStorage,Region.cast] using h
  · have h := congrArg decode (readback inert .fp32 (snapshotParameters out) (snapshotMemory floatInputs)
      snapshotSeed (⟨2,by simp [snapshotParameters,selected,inputOffset,snapshotMemory,snapshotSeed,readBool,Region.cast]⟩,PUnit.unit) (by simp [snapshotParameters,selected,inputOffset,snapshotMemory,snapshotSeed,readBool,Region.cast]) (snapshot_injective out floatInputs))
    simpa [final,snapshotParameters,snapshotMemory,snapshotSeed,floatInputs,inputOffset,outputOffset,
      readNat,readPayload,readStorage,Region.cast] using h
  · apply execution_frame
    right; intro i hi
    rcases snapshot_selected out floatInputs i hi with h0 | h2 <;>
      simp [outputOffset,inputOffset,snapshotParameters,snapshotMemory,snapshotSeed,readNat,Region.cast, *]
  · apply execution_frame
    right; intro i hi
    rcases snapshot_selected out floatInputs i hi with h0 | h2 <;>
      simp [outputOffset,inputOffset,snapshotParameters,snapshotMemory,snapshotSeed,readNat,Region.cast, *]

def integerInputs : Nat → RawCell (BitVec 32) := fun o =>
  .int ((if o=0 then (0x8000000000000000 : BitVec 64) else 0x7fffffffffffffff).toInt)

/-- Both signed 64-bit endpoint patterns survive actual integer-path source
execution with input/output alias and the same partial selection. -/
theorem int64_endpoints_source_success :
    ∃ final, FP.Structural.exec (copyBackend inert) (TypedCopy.source .int64 (snapshotParameters "input"))
      (state (snapshotMemory integerInputs) snapshotSeed)=some final ∧
      BitVec.ofInt 64 ((final.mem "input" 2).read .int)=0x8000000000000000 ∧
      BitVec.ofInt 64 ((final.mem "input" 3).read .int)=0x7fffffffffffffff := by
  let final := (TypedCopy.execution inert .int64 (snapshotParameters "input")
    (state (snapshotMemory integerInputs) snapshotSeed)).val
  refine ⟨final,(TypedCopy.execution inert .int64 (snapshotParameters "input")
    (state (snapshotMemory integerInputs) snapshotSeed)).property,?_,?_⟩
  · have h := every_int64_bits (snapshotParameters "input") (snapshotMemory integerInputs) snapshotSeed
      (⟨0,by simp [snapshotParameters,selected,inputOffset,snapshotMemory,snapshotSeed,readBool,Region.cast]⟩,PUnit.unit) (by simp [snapshotParameters,selected,inputOffset,snapshotMemory,snapshotSeed,readBool,Region.cast]) (snapshot_injective "input" integerInputs)
      0x8000000000000000 (by simp [snapshotParameters,snapshotMemory,snapshotSeed,inputOffset,integerInputs])
    simpa [final,snapshotParameters,snapshotMemory,snapshotSeed,inputOffset,outputOffset,readNat,Region.cast] using h
  · have h := every_int64_bits (snapshotParameters "input") (snapshotMemory integerInputs) snapshotSeed
      (⟨2,by simp [snapshotParameters,selected,inputOffset,snapshotMemory,snapshotSeed,readBool,Region.cast]⟩,PUnit.unit) (by simp [snapshotParameters,selected,inputOffset,snapshotMemory,snapshotSeed,readBool,Region.cast]) (snapshot_injective "input" integerInputs)
      0x7fffffffffffffff (by simp [snapshotParameters,snapshotMemory,snapshotSeed,inputOffset,integerInputs])
    simpa [final,snapshotParameters,snapshotMemory,snapshotSeed,inputOffset,outputOffset,readNat,Region.cast] using h

theorem inactive_source_success {R : Type} [Inhabited R] (d : Storage) (p : Parameters)
    (m : Memory R) (seed : State (Carrier R)) (hoff : ∀ i, ¬ selected p m seed.pids i) :
    ∃ final, FP.Structural.exec (copyBackend inert) (TypedCopy.source d p)
      (state m seed)=some final ∧ final.mem=encodeMemory m := by
  refine ⟨(TypedCopy.execution inert d p (state m seed)).val,
    (TypedCopy.execution inert d p (state m seed)).property,?_⟩
  funext r o
  apply execution_frame
  right; intro i hi; exact (hoff i hi).elim

theorem zero_width_source_success {R : Type} [Inhabited R] (d : Storage) (p : Parameters)
    (m : Memory R) (seed : State (Carrier R)) (hzero : p.width=0) :
    ∃ final, FP.Structural.exec (copyBackend inert) (TypedCopy.source d p)
      (state m seed)=some final ∧ final.mem=encodeMemory m := by
  apply inactive_source_success
  intro i _
  have hi := i.1.isLt
  omega

/-- Zero prefix remains in the old Nat model. It saturates at zero, rather
than becoming a finite signed64 negative address; no new positivity is added. -/
theorem zero_prefix_is_old_nat_subtraction {R : Type} (p : Parameters) (m : Memory R)
    (ids : Nat → Nat) (i : TileIndex [p.width])
    (hzero : m p.prefixSum.cast (inputOffset p ids i)=RawCell.nat 0) :
    outputOffset p m ids i=0 := by simp [outputOffset,hzero,readNat]

#axiomsClean aliased_control_payload_source_success
#axiomsClean int64_endpoints_source_success
#axiomsClean inactive_source_success
#axiomsClean zero_width_source_success
#axiomsClean zero_prefix_is_old_nat_subtraction
end
end VeriTile.Bench.TritonBenchG.MaskedSelect.RawBits
