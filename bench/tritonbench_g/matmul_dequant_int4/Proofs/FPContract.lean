import bench.tritonbench_g.matmul_dequant_int4.Proofs.FPTyped
import bench.tritonbench_g.matmul_dequant_int4.Proofs.FPBits
namespace VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped
open VeriTile.Triton FP.Structural
open KcacheCopyTriton.TypedCopy (masked_readback_active)
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000

def Post {α : Type} (p : Parameters) (initial final : State α) (results : Index p → α) : Prop :=
  (∀ i, active p initial i → final.mem p.output (outputOffset p initial i) = .mk .fp16 (results i)) ∧
  (∀ r o, (r ≠ p.output ∨ ∀ i, active p initial i → o ≠ outputOffset p initial i) →
    final.mem r o = initial.mem r o)

theorem correctness {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (hw : ∀ i j, active p s i → active p s j → outputOffset p s i = outputOffset p s j → i = j) :
    ∃ final, FP.Structural.exec M (source p) s = some final ∧ Post p s final (value M p s) := by
  refine ⟨(execution M p s).val, (execution M p s).property, ?_, ?_⟩
  · intro i hi
    rw [execution_memory]
    exact masked_readback_active _ _ _ _ _ i hi hw
  · intro r o h
    rw [execution_memory]
    exact State.masked_scatter_frame _ _ _ _ _ _ _ (h.imp id (fun h i _ => h i)) _

/-- The input ABI supplies signed int32 words. This is a representation
condition on initial payloads, not a numerical law or an assumed output. -/
def PackedInputs {α : Type} [Inhabited α] (p : Parameters) (s : State α) : Prop :=
  ∀ i, active p s i →
    (∃ w : BitVec 32, s.mem p.weight (weightOffset p s i) = .mk .int w.toInt) ∧
    (∃ z : BitVec 32, s.mem p.zero (zeroOffset p s i) = .mk .int z.toInt)

theorem nibble_packed_word (word : BitVec 32) (slot : Nat) :
    nibble word.toInt slot = (((word >>> (slot % 8 * 4)) &&& (15 : BitVec 32)).toInt) := by
  have h := FPBits.signed_logical_mask word (⟨slot % 8, Nat.mod_lt _ (by decide)⟩ : Fin 8)
  exact congrArg BitVec.toInt h

theorem difference_bounds {α : Type} [Inhabited α] (p : Parameters) (s : State α) (i : Index p) :
    -15 ≤ difference p s i ∧ difference p s i ≤ 15 := by
  have ha := FPBits.mask_bound (BitVec.ofInt 32
    (((s.mem p.weight (weightOffset p s i)).read .int) >>> (row p s i % 8 * 4)))
  have hb := FPBits.mask_bound (BitVec.ofInt 32
    (((s.mem p.zero (zeroOffset p s i)).read .int) >>> (column p s i % 8 * 4)))
  change 0 ≤ nibble ((s.mem p.weight (weightOffset p s i)).read .int) (row p s i) ∧
    nibble ((s.mem p.weight (weightOffset p s i)).read .int) (row p s i) ≤ 15 at ha
  change 0 ≤ nibble ((s.mem p.zero (zeroOffset p s i)).read .int) (column p s i) ∧
    nibble ((s.mem p.zero (zeroOffset p s i)).read .int) (column p s i) ≤ 15 at hb
  dsimp only [difference]
  omega

theorem difference_int32_exact {α : Type} [Inhabited α] (p : Parameters) (s : State α) (i : Index p) :
    (BitVec.ofInt 32 (difference p s i)).toInt = difference p s i := by
  have h := difference_bounds p s i
  apply BitVec.toInt_ofInt_eq_self (by decide) <;> omega

theorem difference_fp16_exact {α : Type} [Inhabited α] (p : Parameters) (s : State α) (i : Index p) :
    (Binary16.round .ieeeValue (difference p s i : Rat)).toRat? = some (difference p s i : Rat) :=
  FPBits.fp16_difference_exact _ (difference_bounds p s i)

theorem difference_fp32_exact {α : Type} [Inhabited α] (p : Parameters) (s : State α) (i : Index p) :
    (FP.round .fp32 .ieeeValue (difference p s i : Rat)).toRat? = some (difference p s i : Rat) := by
  have h := difference_bounds p s i
  generalize hn : difference p s i = n at *
  rcases h with ⟨hl, hu⟩
  interval_cases n <;> decide +kernel

/-- Concrete fp16 integer-conversion/multiply profile, confined to the exact
operand path used here. Other backend callbacks remain explicit in `rest`. -/
def software (rest : Algebra (Binary16.Value)) : Algebra (Binary16.Value) :=
  { rest with
    fromInt := fun precision n => if (precision = none ∨ precision = some .fp32) ∧ (-15 ≤ n ∧ n ≤ 15) then Binary16.round .ieeeValue (n : Rat)
      else rest.fromInt precision n
    cast := fun precision src dst x =>
      if precision = some .fp32 ∧ src = .real ∧ dst = .fp16 then x else rest.cast precision src dst x
    binary := fun precision dtype op x y =>
      if precision = none ∧ dtype = .fp16 ∧ op = .mul then Binary16.mul .ieeeValue x y
      else rest.binary precision dtype op x y }

def roundedValue [Inhabited (Binary16.Value)] (p : Parameters) (s : State (Binary16.Value))
    (i : Index p) : Binary16.Value :=
  Binary16.mul .ieeeValue (Binary16.round .ieeeValue (difference p s i : Rat))
    ((s.mem p.scale (scaleOffset p s i)).read .fp16)

def packedResult [Inhabited Binary16.Value] (p : Parameters) (s : State Binary16.Value)
    (i : Index p) (word zero : BitVec 32) : Binary16.Value :=
  let w := ((word >>> (row p s i % 8 * 4)) &&& (15 : BitVec 32)).toInt
  let z := ((zero >>> (column p s i % 8 * 4)) &&& (15 : BitVec 32)).toInt
  Binary16.mul .ieeeValue (Binary16.round .ieeeValue ((w - z : Int) : Rat))
    ((s.mem p.scale (scaleOffset p s i)).read .fp16)

theorem packed_result_bridge [Inhabited Binary16.Value] (p : Parameters) (s : State Binary16.Value)
    (hInputs : PackedInputs p s) (i : Index p) (hi : active p s i) :
    ∃ word zero : BitVec 32,
      s.mem p.weight (weightOffset p s i) = .mk .int word.toInt ∧
      s.mem p.zero (zeroOffset p s i) = .mk .int zero.toInt ∧
      roundedValue p s i = packedResult p s i word zero := by
  obtain ⟨⟨w, hw⟩, ⟨z, hz⟩⟩ := hInputs i hi
  refine ⟨w, z, hw, hz, ?_⟩
  simp [roundedValue, packedResult, difference, hw, hz, nibble_packed_word]

theorem software_correctness [Inhabited (Binary16.Value)]
    (rest : Algebra (Binary16.Value)) (p : Parameters) (s : State (Binary16.Value))
    (hw : ∀ i j, active p s i → active p s j → outputOffset p s i = outputOffset p s j → i = j) :
    ∃ final, FP.Structural.exec (software rest) (source p) s = some final ∧
      Post p s final (roundedValue p s) := by
  obtain ⟨final, he, hp⟩ := correctness (software rest) p s hw
  refine ⟨final, he, ?_, hp.2⟩
  intro i hi
  simpa [value, software, roundedValue, difference_bounds p s i] using hp.1 i hi

#axiomsClean packed_result_bridge
#axiomsClean nibble_packed_word
#axiomsClean difference_int32_exact
#axiomsClean difference_fp16_exact
#axiomsClean difference_fp32_exact
#axiomsClean correctness
#axiomsClean software_correctness
end
end VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped
