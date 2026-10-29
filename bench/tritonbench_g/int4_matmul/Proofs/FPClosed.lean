import bench.tritonbench_g.int4_matmul.FPCorrect

namespace VeriTile.Bench.TritonBenchG.Int4Matmul.FPClosed
open VeriTile.Triton FP.Structural FPTyped
open FPSoftware (Word)
noncomputable section

/-! A closed binary16/binary32 interpretation of the complete accepted int4
loop. This is the source's separate dot-then-add graph. The dot is an explicit
ascending-K, +0-seeded, rounded multiply/add fold; it is not a tensor-core or
compiler-contracted dotAcc claim. No floating callback is an input. -/

/-- Unreached operations have a fixed total value. `correctness_for_any_rest`
below proves the complete output and frame are independent of this choice.
This fallback is not presented as an implementation of unused primitives. -/
def unused : Algebra Word where
  literal := fun _ _ _ => FP.Value.zero .fp32
  negInf := FP.Value.zero .fp32
  binary := fun _ _ _ _ _ => FP.Value.zero .fp32
  unary := fun _ _ _ => FP.Value.zero .fp32
  cast := fun _ _ _ x => x
  fromNat := fun _ _ => FP.Value.zero .fp32
  fromInt := fun _ _ => FP.Value.zero .fp32
  fp32Bits := fun _ => FP.Value.zero .fp32
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => FP.Value.zero .fp32
  reduceSum := fun _ {_} _ _ _ _ => FP.Value.zero .fp32

def softwareBackend : Algebra Word :=
  FPTyped.backend (FPSoftware.scalar unused) FPSoftware.softwareDot

/-- The original source addresses and signed packed nibble difference are
reused, but the floating reference is independently evaluated from the
initial memory. It does not execute the source or receive its result. -/
def halfWeight (p : Parameters) (s : State Word) (step : Nat) (i : BIndex p) : Word :=
  QuantizeKvTransform.MixedBits.embed
    (MatmulDequantInt4.Binary16.mul .ieeeValue
      (MatmulDequantInt4.Binary16.round .ieeeValue (difference p s step i : Rat))
      (QuantizeKvTransform.MixedBits.half
        ((s.mem p.scale (scaleOffset p s step i)).read .fp16)))

def blockDot (p : Parameters) (s : State Word) (step : Nat) (i : CIndex p) : Word :=
  (List.finRange p.blockInner).foldl (fun acc t =>
    FP.add .ieeeValue acc (FP.mul .ieeeValue
      (QuantizeKvTransform.MixedBits.widen
        ((s.mem p.input (aOffset p s step (i.1, t, PUnit.unit))).read .fp16))
      (QuantizeKvTransform.MixedBits.widen
        (halfWeight p s step (t, i.2.1, PUnit.unit))))) (FP.Value.zero .fp32)

def accumulator (p : Parameters) (s : State Word) : Nat → CIndex p → Word
  | 0 => fun _ => FP.Value.zero .fp32
  | step + 1 => fun i =>
      FP.add .ieeeValue (accumulator p s step i) (blockDot p s step i)

def outputBits (p : Parameters) (s : State Word) (i : CIndex p) : Word :=
  QuantizeKvTransform.MixedBits.narrow .ieeeValue (accumulator p s p.steps i)

theorem machine_accumulator_eq (p : Parameters) (s : State Word) (k : Nat) :
    FPSoftware.machineAccum FPSoftware.softwareDot p s k = accumulator p s k := by
  induction k with
  | zero => rfl
  | succ k ih =>
    rw [FPSoftware.machineAccum, accumulator, ih]
    funext i
    simp only [FPSoftware.softwareDot, FPSoftware.dot, blockDot, aValue,
      FPSoftware.dequantized, halfWeight]

theorem machine_output_eq (p : Parameters) (s : State Word) :
    FPSoftware.machineOutput FPSoftware.softwareDot p s = outputBits p s := by
  funext i
  simp only [FPSoftware.machineOutput, outputBits, machine_accumulator_eq]

/-- No residual numerical law, callback equality, or final-result assumption:
every fallback algebra gives the same independent bit reference and frame.
All original public input/scope premises are retained byte-for-byte in shape. -/
theorem correctness_for_any_rest
    (rest : Algebra Word) (p : Parameters) (s : State Word)
    (hTrip : p.inner = p.blockInner * p.steps) (hSplit : s.pids 1 = 0)
    (hInputs : PackedInputs p s)
    (hw : ∀ i j, active p s i → active p s j →
      outputOffset p s i = outputOffset p s j → i = j) :
    ∃ final,
      FP.Structural.exec (FPTyped.backend (FPSoftware.scalar rest) FPSoftware.softwareDot)
        (source p) s = some final ∧
      Post p s final (outputBits p s) ∧
      ∀ step, step < p.steps → ∀ i : BIndex p, ∃ word zero : BitVec 32,
        s.mem p.weight (bOffset p s step i) = .mk .int word.toInt ∧
        s.mem p.zero (zeroOffset p s step i) = .mk .int zero.toInt ∧
        difference p s step i = packedDifference p s i word zero := by
  simpa only [machine_output_eq] using
    int4_matmul_fp16_correctness rest FPSoftware.softwareDot p s hTrip hSplit hInputs hw

theorem post_memory_unique (p : Parameters) (s t u : State Word)
    (ht : Post p s t (outputBits p s)) (hu : Post p s u (outputBits p s)) :
    t.mem = u.mem := by
  classical
  funext r o
  by_cases h : r = p.output ∧ ∃ i, active p s i ∧ outputOffset p s i = o
  · obtain ⟨rfl, i, hi, rfl⟩ := h
    rw [ht.1 i hi, hu.1 i hi]
  · have hf : r ≠ p.output ∨ ∀ i, active p s i → o ≠ outputOffset p s i := by
      by_cases hr : r = p.output
      · right
        intro i hi he
        exact h ⟨hr, i, hi, he.symm⟩
      · exact Or.inl hr
    rw [ht.2 r o hf, hu.2 r o hf]

/-- Even replacing every unreachable primitive cannot change any memory
cell after the complete loop, including untouched and inactive locations. -/
theorem all_memory_independent_of_unused
    (left right : Algebra Word) (p : Parameters) (s t u : State Word)
    (hTrip : p.inner = p.blockInner * p.steps) (hSplit : s.pids 1 = 0)
    (hInputs : PackedInputs p s)
    (hw : ∀ i j, active p s i → active p s j →
      outputOffset p s i = outputOffset p s j → i = j)
    (ht : FP.Structural.exec (FPTyped.backend (FPSoftware.scalar left) FPSoftware.softwareDot)
      (source p) s = some t)
    (hu : FP.Structural.exec (FPTyped.backend (FPSoftware.scalar right) FPSoftware.softwareDot)
      (source p) s = some u) : t.mem = u.mem := by
  obtain ⟨t', het, hpt, _⟩ := correctness_for_any_rest left p s hTrip hSplit hInputs hw
  obtain ⟨u', heu, hpu, _⟩ := correctness_for_any_rest right p s hTrip hSplit hInputs hw
  have hte : t' = t := Option.some.inj (het.symm.trans ht)
  have hue : u' = u := Option.some.inj (heu.symm.trans hu)
  subst t'
  subst u'
  exact post_memory_unique p s t u hpt hpu

#axiomsClean machine_accumulator_eq
#axiomsClean machine_output_eq
#axiomsClean correctness_for_any_rest
#axiomsClean post_memory_unique
#axiomsClean all_memory_independent_of_unused
end
end VeriTile.Bench.TritonBenchG.Int4Matmul.FPClosed
