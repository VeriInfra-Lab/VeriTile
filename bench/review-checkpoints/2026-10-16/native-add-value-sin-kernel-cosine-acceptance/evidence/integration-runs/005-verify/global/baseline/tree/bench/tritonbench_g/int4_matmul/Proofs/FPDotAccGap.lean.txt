import bench.tritonbench_g.int4_matmul.SoftwareCorrect
namespace VeriTile.Bench.TritonBenchG.Int4Matmul.FPDotAccGap
open VeriTile.Triton FP.Structural FPTyped FPClosed
open FPSoftware (Word)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000

/-- A second explicit software graph: carry-seeded increasing-K fp32 FMA.
This matches the *arity/dataflow* of compiled tt.dot, but does not claim to
implement WGMMA's undocumented internal accumulation tree or NaN behavior. -/
def carryBlock (p : Parameters) (s : State Word) (step : Nat)
    (carry : CIndex p → Word) (i : CIndex p) : Word :=
  (List.finRange p.blockInner).foldl (fun acc t => FP.fma .ieeeValue
    (QuantizeKvTransform.MixedBits.widen
      ((s.mem p.input (aOffset p s step (i.1, t, PUnit.unit))).read .fp16))
    (QuantizeKvTransform.MixedBits.widen (halfWeight p s step (t, i.2.1, PUnit.unit)))
    acc) (carry i)

def carryAccum (p : Parameters) (s : State Word) : Nat → CIndex p → Word
  | 0 => fun _ => FP.Value.zero .fp32
  | step + 1 => carryBlock p s step (carryAccum p s step)

def carryOutput (p : Parameters) (s : State Word) (i : CIndex p) : Word :=
  QuantizeKvTransform.MixedBits.narrow .ieeeValue (carryAccum p s p.steps i)

/-- Uses the original host's 64x128x32 tile, SPLIT_K=1, and two K blocks.
Only one output is active; all 32 input products in both blocks are retained. -/
def parameters : Parameters :=
  ⟨"input", "output", "scale", "weight", "zero", 1, 1, 64, 64, 1, 1, 1,
    129, 1, 1, 1, 1, 1, 1, 64, 128, 32, 8, 2⟩

def hword (bits : Nat) : Word :=
  QuantizeKvTransform.MixedBits.embed (MatmulDequantInt4.Binary16.Value.ofBits bits)

def initial : State Word where
  mem := fun r o =>
    if r = ("weight" : RegionName) then .mk .int (0x11111111 : BitVec 32).toInt
    else if r = ("zero" : RegionName) then .mk .int (0 : BitVec 32).toInt
    else if r = ("input" : RegionName) then .mk .fp16
      (hword (if o=0 ∨ o=33 then 0x3c00 else if o=32 then 0x6c00
        else if o=34 then 0xec00 else 0))
    else if r = ("scale" : RegionName) then .mk .fp16
      (hword (if o=32 ∨ o=34 then 0x6c00 else 0x3c00))
    else .mk .fp16 (hword 0x3555)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

def lane : CIndex parameters := (⟨0,by decide⟩, ⟨0,by decide⟩, ())

theorem actual_scope_inputs :
    parameters.inner = parameters.blockInner * parameters.steps ∧ initial.pids 1 = 0 ∧
    PackedInputs parameters initial := by
  refine ⟨rfl, rfl, ?_⟩
  intro step hs i
  exact ⟨⟨0x11111111, rfl⟩, ⟨0, rfl⟩⟩

theorem layout : ∀ i j, active parameters initial i → active parameters initial j →
    outputOffset parameters initial i = outputOffset parameters initial j → i = j := by
  intro i j hi hj _
  have hil : i.1.val=0 ∧ i.2.1.val=0 := by
    simpa [active, rowBlock, columnBlock, firstRow, rowsInGroup, blocksM, blocksN,
      parameters, initial, Nat.lt_one_iff] using hi
  have hjl : j.1.val=0 ∧ j.2.1.val=0 := by
    simpa [active, rowBlock, columnBlock, firstRow, rowsInGroup, blocksM, blocksN,
      parameters, initial, Nat.lt_one_iff] using hj
  apply Prod.ext (Fin.ext (hil.1.trans hjl.1.symm))
  exact Prod.ext (Fin.ext (hil.2.trans hjl.2.symm)) (Subsingleton.elim _ _)

/-- The first block gives +1. In the second block the nonzero products are
2^24, +1, -2^24. Seeding that fold with +1 instead of adding +1 afterward
changes the *stored half bits*: 0x3c00 versus 0x0000. All inputs are finite. -/
theorem two_block_stored_bits_differ :
    outputBits parameters initial lane = hword 0x3c00 ∧
    carryOutput parameters initial lane = hword 0x0000 := by
  decide +kernel

/-- A successful execution of the entire accepted source graph and its
stored result, plus the incompatible carry-dot software-reference result.
This refutes a universal graph-replacement law, not a GPU execution claim. -/
theorem successful_complete_source_counterexample :
    ∃ final, FP.Structural.exec softwareBackend (source parameters) initial = some final ∧
      final.mem "output" 0 = .mk .fp16 (hword 0x3c00) ∧
      final.mem "output" 1 = initial.mem "output" 1 ∧
      carryOutput parameters initial lane = hword 0x0000 ∧
      outputBits parameters initial lane ≠ carryOutput parameters initial lane := by
  obtain ⟨final, he, hp, _⟩ := SoftwareCorrect.source_profile_software_correctness
    parameters initial actual_scope_inputs.1 actual_scope_inputs.2.1 actual_scope_inputs.2.2 layout
  refine ⟨final, he, ?_, ?_, two_block_stored_bits_differ.2, ?_⟩
  · have h := hp.1 lane (by decide)
    simpa only [show outputOffset parameters initial lane = 0 from rfl,
      two_block_stored_bits_differ.1] using h
  · apply hp.2 "output" 1
    right
    intro i hi
    have hval : i.1.val=0 ∧ i.2.1.val=0 := by
      simpa [active, rowBlock, columnBlock, firstRow, rowsInGroup, blocksM, blocksN,
        parameters, initial, Nat.lt_one_iff] using hi
    simp [outputOffset, rowBlock, columnBlock, firstRow, rowsInGroup, blocksM, blocksN,
      parameters, initial, hval.1, hval.2]
  · rw [two_block_stored_bits_differ.1, two_block_stored_bits_differ.2]
    decide

#axiomsClean actual_scope_inputs
#axiomsClean layout
#axiomsClean two_block_stored_bits_differ
#axiomsClean successful_complete_source_counterexample
end
end VeriTile.Bench.TritonBenchG.Int4Matmul.FPDotAccGap
