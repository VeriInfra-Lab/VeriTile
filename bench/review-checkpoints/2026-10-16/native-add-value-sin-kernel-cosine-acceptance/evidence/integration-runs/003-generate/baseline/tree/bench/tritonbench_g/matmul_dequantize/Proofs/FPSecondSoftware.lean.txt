import bench.tritonbench_g.matmul_dequantize.Proofs.FPSecondComplete
import bench.tritonbench_g.int4_matmul.Proofs.FPMixed
namespace VeriTile.Bench.TritonBenchG.MatmulDequantize.FPSecondSoftware
open VeriTile.Triton FP.Structural
open FPTypedSecond
open Int4Matmul.FPSoftware (Word scalar bounded_integer_cast dot)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

def softwareDot : FPTypedSecond.Dot Word :=
  ⟨fun _ _ {batch} {m} {k} {n} a b => dot a b⟩

/-- Read the binary16 scale bits from the initial snapshot, and round the
native int4 subtraction's product directly to binary16. -/
def dequantized (p : Parameters) (s : State Word) (step : Nat) (i : BIndex p) : Word :=
  QuantizeKvTransform.MixedBits.embed (MatmulDequantInt4.Binary16.mul .ieeeValue
    (MatmulDequantInt4.Binary16.round .ieeeValue (difference p s step i : Rat))
    (QuantizeKvTransform.MixedBits.half ((s.mem p.scale (scaleOffset p s step i)).read .fp16)))

theorem bValue_scalar (rest : Algebra Word) (p : Parameters) (s : State Word)
    (step : Nat) (i : BIndex p) :
    bValue (scalar rest) p s step i = dequantized p s step i := by
  simp only [bValue, scalar]
  simp only [or_true, and_true, ↓reduceIte]
  rw [bounded_integer_cast _ (difference_bounds p s step i)]
  simp [dequantized, QuantizeKvTransform.MixedBits.half_embedding]

def machineA (p : Parameters) (s : State Word) (step : Nat) (i : AIndex p) : Word :=
  QuantizeKvTransform.MixedBits.embed (QuantizeKvTransform.MixedBits.half
    ((s.mem p.input (aOffset p s step i)).read .fp16))

theorem aValue_scalar (rest : Algebra Word) (p : Parameters) (s : State Word) (step : Nat) :
    aValue (scalar rest) p s step = machineA p s step := by rfl

/-- Scalar operations are concrete software bits; only the declared dot
primitive D remains a backend parameter. No output value is an assumption. -/
def machineAccum (D : FPTypedSecond.Dot Word) (p : Parameters) (s : State Word) : Nat → CIndex p → Word
  | 0 => fun _ => FP.Value.zero .fp32
  | step + 1 => fun i => FP.add .ieeeValue (machineAccum D p s step i)
      (D.apply (some .fp32) {} (batch := []) (machineA p s step) (dequantized p s step) i)

def machineOutput (D : FPTypedSecond.Dot Word) (p : Parameters) (s : State Word) (i : CIndex p) : Word :=
  QuantizeKvTransform.MixedBits.narrow .ieeeValue (machineAccum D p s p.steps i)

theorem accumulation_scalar (rest : Algebra Word) (D : FPTypedSecond.Dot Word)
    (p : Parameters) (s : State Word) (k : Nat) :
    accumulation (scalar rest) D p s k = machineAccum D p s k := by
  induction k with
  | zero => simp [accumulation, machineAccum, scalar, FP.Software.scalarFP32, FP.Software.fp32Context]
  | succ k ih =>
    funext i
    simp only [accumulation, machineAccum, ih, bValue_scalar, aValue_scalar]
    simp [scalar, FP.Software.scalarFP32, FP.Software.fp32Context,
      dequantized, QuantizeKvTransform.MixedBits.half_embedding]
    rfl

theorem output_scalar (rest : Algebra Word) (D : FPTypedSecond.Dot Word)
    (p : Parameters) (s : State Word) :
    outputValue (scalar rest) D p s = machineOutput D p s := by
  funext i
  unfold outputValue
  rw [accumulation_scalar]
  rfl

theorem correctness (rest : Algebra Word) (D : FPTypedSecond.Dot Word)
    (p : Parameters) (s : State Word)
    (hw : ∀ i j, active p s i → active p s j → outputOffset p s i = outputOffset p s j → i = j) :
    ∃ final, FP.Structural.exec (FPTypedSecond.backend (scalar rest) D) (source p) s = some final ∧
      Post p s final (machineOutput D p s) := by
  obtain ⟨final, he, hp⟩ := FPTypedSecond.correctness (scalar rest) D p s hw
  exact ⟨final, he, output_scalar rest D p s ▸ hp⟩

#axiomsClean aValue_scalar
#axiomsClean bValue_scalar
#axiomsClean accumulation_scalar
#axiomsClean output_scalar
#axiomsClean correctness
end
end VeriTile.Bench.TritonBenchG.MatmulDequantize.FPSecondSoftware
