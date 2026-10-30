import bench.tritonbench_g.matmul_dequantize.Proofs.FPFirstComplete
import bench.tritonbench_g.int4_matmul.Proofs.FPMixed
namespace VeriTile.Bench.TritonBenchG.MatmulDequantize.FPFirstSoftware
open VeriTile.Triton FP.Structural FPTypedFirst
open Int4Matmul.FPSoftware (Word bounded_integer_cast dot)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- The only unscoped real literal in the typed source is the masked-load
zero. Give that exact constant a concrete binary32 carrier before its half
cast, without granting arbitrary real literals an identity interpretation. -/
def scalar (rest : Algebra Word) : Algebra Word :=
  let base := Int4Matmul.FPSoftware.scalar rest
  {base with literal := fun p d q =>
    if p = none ∧ d = .real ∧ q = 0 then FP.Value.zero .fp32 else base.literal p d q}

def softwareDot : FPTypedFirst.Dot Word :=
  ⟨fun _ _ {batch} {m} {k} {n} a b => dot a b⟩

def machineA (p : Parameters) (s : State Word) (step : Nat) (i : AIndex p) : Word :=
  if inputRow p s i.1 < p.rows then (s.mem p.input (aOffset p s step i)).read .fp16
  else QuantizeKvTransform.MixedBits.embed MatmulDequantInt4.Binary16.Value.zero

def scaledZero (p : Parameters) (s : State Word) (step : Nat)
    (j : TileIndex [p.blockColumns]) : MatmulDequantInt4.Binary16.Value :=
  MatmulDequantInt4.Binary16.mul .ieeeValue
    (MatmulDequantInt4.Binary16.round .ieeeValue (zeroNibble p s step j : Rat))
    (QuantizeKvTransform.MixedBits.half (scaleValue p s step j))

/-- The source scales weight and zero separately, rounding each product to
half, and then rounds their subtraction to half. No factorization or FMA
identity is assumed. -/
def dequantized (p : Parameters) (s : State Word) (step : Nat) (i : BIndex p) : Word :=
  QuantizeKvTransform.MixedBits.embed (MatmulDequantInt4.Binary16.sub .ieeeValue
    (MatmulDequantInt4.Binary16.mul .ieeeValue
      (MatmulDequantInt4.Binary16.round .ieeeValue (weightNibble p s step i : Rat))
      (QuantizeKvTransform.MixedBits.half (scaleValue p s step (i.2.1, PUnit.unit))))
    (scaledZero p s step (i.2.1, PUnit.unit)))

theorem aValue_scalar (rest : Algebra Word) (p : Parameters) (s : State Word) (step : Nat) :
    aValue (scalar rest) p s step = machineA p s step := by
  have hz : QuantizeKvTransform.MixedBits.narrow .ieeeValue (FP.Value.zero .fp32) =
      QuantizeKvTransform.MixedBits.embed MatmulDequantInt4.Binary16.Value.zero := by decide +kernel
  funext i
  simp [aValue, machineA, scalar, Int4Matmul.FPSoftware.scalar, hz]

theorem bValue_scalar (rest : Algebra Word) (p : Parameters) (s : State Word)
    (step : Nat) (i : BIndex p) :
    bValue (scalar rest) p s step i = dequantized p s step i := by
  have hw := weightNibble_bounds p s step i
  have hz := zeroNibble_bounds p s step (i.2.1, PUnit.unit)
  simp only [bValue, zeroValue, scalar, Int4Matmul.FPSoftware.scalar]
  simp only [or_true, true_or, and_true, ↓reduceIte]
  rw [bounded_integer_cast _ (by omega), bounded_integer_cast _ (by omega)]
  simp [dequantized, scaledZero, QuantizeKvTransform.MixedBits.half_embedding]

def machineAccum (D : FPTypedFirst.Dot Word) (p : Parameters) (s : State Word) : Nat → CIndex p → Word
  | 0 => fun _ => FP.Value.zero .fp32
  | step + 1 => fun i => FP.add .ieeeValue (machineAccum D p s step i)
      (D.apply (some .fp32) {} (batch := []) (machineA p s step) (dequantized p s step) i)

def machineOutput (D : FPTypedFirst.Dot Word) (p : Parameters) (s : State Word) (i : CIndex p) : Word :=
  QuantizeKvTransform.MixedBits.narrow .ieeeValue (machineAccum D p s (steps p) i)

theorem accumulation_scalar (rest : Algebra Word) (D : FPTypedFirst.Dot Word)
    (p : Parameters) (s : State Word) (k : Nat) :
    accumulation (scalar rest) D p s k = machineAccum D p s k := by
  induction k with
  | zero => simp [accumulation, machineAccum, scalar, Int4Matmul.FPSoftware.scalar,
      FP.Software.scalarFP32, FP.Software.fp32Context]
  | succ k ih =>
    funext i
    simp only [accumulation, machineAccum, ih, aValue_scalar,
      show bValue (scalar rest) p s k = dequantized p s k from funext (bValue_scalar rest p s k)]
    simp [scalar, Int4Matmul.FPSoftware.scalar, FP.Software.scalarFP32, FP.Software.fp32Context,
      dequantized, QuantizeKvTransform.MixedBits.half_embedding]
    try rfl

theorem output_scalar (rest : Algebra Word) (D : FPTypedFirst.Dot Word)
    (p : Parameters) (s : State Word) :
    outputValue (scalar rest) D p s = machineOutput D p s := by
  funext i
  unfold outputValue
  rw [accumulation_scalar]
  rfl

theorem correctness (rest : Algebra Word) (D : FPTypedFirst.Dot Word)
    (p : Parameters) (s : State Word)
    (hw : ∀ i j, active p s i → active p s j → outputOffset p s i = outputOffset p s j → i = j) :
    ∃ final, FP.Structural.exec (FPTypedFirst.backend (scalar rest) D) (source p) s = some final ∧
      Post p s final (machineOutput D p s) := by
  obtain ⟨final, he, hp⟩ := FPTypedFirst.correctness (scalar rest) D p s hw
  exact ⟨final, he, output_scalar rest D p s ▸ hp⟩

#axiomsClean aValue_scalar
#axiomsClean bValue_scalar
#axiomsClean accumulation_scalar
#axiomsClean output_scalar
#axiomsClean correctness
end
end VeriTile.Bench.TritonBenchG.MatmulDequantize.FPFirstSoftware
