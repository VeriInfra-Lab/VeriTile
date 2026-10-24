import bench.tritonbench_g.rotary_emb_nopad.Proofs.FPSoftware
namespace VeriTile.Bench.TritonBenchG.RotaryEmbNopad.FPExamples
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def one : FP.Value .fp32 := FP.Value.ofBits .fp32 0x3f800000
def tiny : FP.Value .fp32 := FP.Value.ofBits .fp32 0x33000000

theorem rounding :
    FP.sub .ieeeValue (FP.mul .ieeeValue one one) (FP.mul .ieeeValue tiny one) = one ∧
    FP.add .ieeeValue (FP.mul .ieeeValue one one) (FP.mul .ieeeValue tiny one) = one ∧
    one.toRat? = some 1 ∧ tiny.toRat? = some (1 / 33554432) ∧
    (1 - 1 / 33554432 : Rat) ≠ 1 ∧ (1 + 1 / 33554432 : Rat) ≠ 1 ∧ tiny ≠ one := by decide +kernel

def blocks : FPBlocks.Parameters := ⟨"x", "cos", "sin", 2, 2, 1, 1, 1, 1, 1, 2, 1, 2⟩

def initial [Inhabited (FP.Value .fp32)] : State (FP.Value .fp32) where
  mem := fun r o =>
    if r = ("lengths" : RegionName) then .mk .nat 2
    else if r = ("table" : RegionName) then .mk .nat 3
    else .mk .fp32 (if r = ("x" : RegionName) ∧ o = 1 then tiny else one)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

def blockLane : FPBlocks.Index blocks := (⟨0, by decide⟩, ⟨0, by decide⟩, ⟨0, by decide⟩, PUnit.unit)

theorem block_active_index [Inhabited (FP.Value .fp32)] (key : Bool)
    (i : FPBlocks.Index blocks) (hi : FPBlocks.active key blocks initial i) : i = blockLane := by
  apply Prod.ext
  · apply Fin.ext
    have ht := hi.2
    simp [FPBlocks.token, initial, blocks] at ht
    change i.1.val = 0
    exact congrArg Fin.val ht
  · exact Subsingleton.elim (α := Fin 1 × Fin 1 × PUnit) _ _

theorem block_writes [Inhabited (FP.Value .fp32)] (key : Bool) :
    Pair.WritesApart (FPBlocks.active key blocks initial) (FPBlocks.offset key blocks initial) := by
  constructor
  · intro second i j hi hj _
    exact (block_active_index key i hi).trans (block_active_index key j hj).symm
  · intro i j hi hj
    rw [block_active_index key i hi, block_active_index key j hj]
    cases key <;> simp [FPBlocks.offset, FPBlocks.token, FPBlocks.head, blocks, initial, blockLane]

theorem blocks_nonempty [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (key : Bool) :
    ∃ final, FP.Structural.exec (FP.Software.scalarFP32 rest .ieeeValue)
      (FPBlocks.source key blocks) initial = some final ∧
      final.mem "x" 0 = .mk .fp32 one ∧ final.mem "x" 1 = .mk .fp32 one ∧
      final.mem "x" 2 = initial.mem "x" 2 ∧ tiny ≠ one := by
  obtain ⟨final, he, hp⟩ := FPBlocks.software_correctness rest .ieeeValue key blocks initial (block_writes key)
  have hi : FPBlocks.active key blocks initial blockLane := by
    cases key <;> simp [FPBlocks.active, FPBlocks.token, blocks, initial, blockLane]
  refine ⟨final, he, ?_, ?_, ?_, rounding.2.2.2.2.2.2⟩
  · have hv := hp.1 Bool.false blockLane hi
    cases key <;> simpa [FPBlocks.roundedValue, FPBlocks.offset, FPBlocks.cacheOffset,
      FPBlocks.token, FPBlocks.head, blocks, initial, blockLane, rounding.1] using hv
  · have hv := hp.1 Bool.true blockLane hi
    cases key <;> simpa [FPBlocks.roundedValue, FPBlocks.offset, FPBlocks.cacheOffset,
      FPBlocks.token, FPBlocks.head, blocks, initial, blockLane, rounding.2.1] using hv
  · apply hp.2 "x" 2
    right
    intro second i hi
    rw [block_active_index key i hi]
    cases key <;> cases second <;>
      simp [FPBlocks.offset, FPBlocks.token, FPBlocks.head, blocks, initial, blockLane]

def v2 : FPV2.Parameters where
  input := "x"
  output := "cache"
  cosine := "cos"
  sine := "sin"
  blockTables := ⟨"table"⟩
  lengths := ⟨"lengths"⟩
  tokenStride := 2
  headStride := 2
  dimensionStride := 1
  cosineTokenStride := 1
  cosineStride := 1
  cacheBlockStride := 100
  cacheHeadStride := 10
  cacheSlotStride := 4
  cacheDimensionStride := 1
  tableSequenceStride := 1
  tableBlockStride := 1
  blockSize := 4
  totalTokens := 1
  heads := 1
  half := 1

def lane : TileIndex [v2.half] := (⟨0, by decide⟩, PUnit.unit)

theorem v2_writes [Inhabited (FP.Value .fp32)] (cache : Bool) (s : State (FP.Value .fp32)) :
    Pair.WritesApart (FPV2.active v2 s) (FPV2.outputOffset cache v2 s) := by
  constructor
  · intro second i j _ _ _
    exact Subsingleton.elim (α := Fin 1 × PUnit) _ _
  · intro i j _ _
    have hi : i = lane := Subsingleton.elim (α := Fin 1 × PUnit) _ _
    have hj : j = lane := Subsingleton.elim (α := Fin 1 × PUnit) _ _
    subst i; subst j
    cases cache <;> simp [FPV2.outputOffset, FPV2.inputOffset, v2, lane]

theorem v2_nonempty [Inhabited (FP.Value .fp32)]
    (rest : Algebra (FP.Value .fp32)) (cache : Bool) :
    ∃ final, FP.Structural.exec (FPV2.softwareBackend rest .ieeeValue)
      (FPV2.source cache v2) initial = some final ∧
      final.mem (FPV2.outputRegion cache v2) (if cache then 304 else 0) = .mk .fp32 one ∧
      final.mem (FPV2.outputRegion cache v2) (if cache then 305 else 1) = .mk .fp32 one ∧
      final.mem (FPV2.outputRegion cache v2) (if cache then 306 else 2) =
        initial.mem (FPV2.outputRegion cache v2) (if cache then 306 else 2) ∧ tiny ≠ one := by
  obtain ⟨final, he, hp⟩ := FPV2.software_correctness rest .ieeeValue cache v2 initial (v2_writes cache initial)
  have hi : FPV2.active v2 initial lane := by simp [FPV2.active, v2, initial]
  refine ⟨final, he, ?_, ?_, ?_, rounding.2.2.2.2.2.2⟩
  · have hv := hp.1 Bool.false lane hi
    cases cache <;> simpa [FPV2.roundedValue, FPV2.outputOffset, FPV2.inputOffset,
      FPV2.cacheOffset, FPV2.blockId, FPV2.past, v2, initial, lane, Region.cast, rounding.1] using hv
  · have hv := hp.1 Bool.true lane hi
    cases cache <;> simpa [FPV2.roundedValue, FPV2.outputOffset, FPV2.inputOffset,
      FPV2.cacheOffset, FPV2.blockId, FPV2.past, v2, initial, lane, Region.cast, rounding.2.1] using hv
  · apply hp.2
    right
    intro second i _
    have hi : i = lane := Subsingleton.elim (α := Fin 1 × PUnit) _ _
    subst i
    cases cache <;> cases second <;>
      simp [FPV2.outputOffset, FPV2.inputOffset, FPV2.blockId, FPV2.past, v2, initial, lane, Region.cast]

/-- A masked block-table load has no `other`: its unspecified value is retained
as a state parameter, and this concrete state chooses 7 rather than zero. -/
def outOfTokenInitial [Inhabited (FP.Value .fp32)] : State (FP.Value .fp32) :=
  { initial with
    pids := fun a => if a = 0 then 0 else 2
    undef := fun d _ _ => match d with | .nat => 7 | d => defaultValue d }

theorem zero_products :
    FP.sub .ieeeValue (FP.mul .ieeeValue one (FP.Value.zero .fp32))
      (FP.mul .ieeeValue one (FP.Value.zero .fp32)) = FP.Value.zero .fp32 ∧
    FP.add .ieeeValue (FP.mul .ieeeValue one (FP.Value.zero .fp32))
      (FP.mul .ieeeValue one (FP.Value.zero .fp32)) = FP.Value.zero .fp32 := by decide +kernel

theorem v2_masked_defaults [Inhabited (FP.Value .fp32)] (rest : Algebra (FP.Value .fp32)) :
    ∃ final, FP.Structural.exec (FPV2.softwareBackend rest .ieeeValue)
      (FPV2.source Bool.true v2) outOfTokenInitial = some final ∧
      final.mem "cache" 704 = .mk .fp32 (FP.Value.zero .fp32) ∧
      final.mem "cache" 705 = .mk .fp32 (FP.Value.zero .fp32) := by
  obtain ⟨final, he, hp⟩ := FPV2.software_correctness rest .ieeeValue Bool.true v2 outOfTokenInitial
    (v2_writes Bool.true outOfTokenInitial)
  have hi : FPV2.active v2 outOfTokenInitial lane := by simp [FPV2.active, v2, outOfTokenInitial]
  refine ⟨final, he, ?_, ?_⟩
  · have hv := hp.1 Bool.false lane hi
    simpa [FPV2.roundedValue, FPV2.outputRegion, FPV2.outputOffset, FPV2.inputOffset,
      FPV2.cacheOffset, FPV2.blockId, FPV2.past, v2, initial, outOfTokenInitial,
      lane, Region.cast, zero_products.1] using hv
  · have hv := hp.1 Bool.true lane hi
    simpa [FPV2.roundedValue, FPV2.outputRegion, FPV2.outputOffset, FPV2.inputOffset,
      FPV2.cacheOffset, FPV2.blockId, FPV2.past, v2, initial, outOfTokenInitial,
      lane, Region.cast, zero_products.2] using hv

#axiomsClean rounding
#axiomsClean blocks_nonempty
#axiomsClean v2_nonempty
#axiomsClean v2_masked_defaults
end
end VeriTile.Bench.TritonBenchG.RotaryEmbNopad.FPExamples
