import bench.tritonbench_g.ksoftmax_triton.Proofs.Graph
import VeriTile.Triton.Float.BitValue

/-! Input decoding only. Reuses the existing FP32 rational decoder; no
floating arithmetic, kernel output, or IEEE/native equality is asserted here.
-/
namespace VeriTile.Bench.Repairs.Ksoftmax.KernelIdeal
open VeriTile.Triton
noncomputable section

def value (word : Graph.Word) : FP.Value .fp32 := ⟨word⟩

def finite (word : Graph.Word) : Prop := (value word).toRat?.isSome=Bool.true

def finiteOrNegInf (word : Graph.Word) : Prop := finite word ∨ word=Graph.negInf

/-- Outside the explicitly stated mathematical input domain, this projection
is irrelevant. NaN and positive infinity are covered by the native theorem. -/
def decode (word : Graph.Word) : WithBot ℝ :=
  ((value word).toRat?).map fun rational => (rational : ℝ)

theorem decode_finite (word : Graph.Word) (domain : finite word) :
    ∃ real : ℝ, decode word=some real := by
  cases h : (value word).toRat? with
  | none => simp [finite,h] at domain
  | some rational => exact ⟨(rational : ℝ),by simp [decode,h]⟩

theorem decode_negative_infinity : decode Graph.negInf=none := by
  rfl

theorem decode_finite_value (word : Graph.Word) (domain : finite word) :
    decode word=some ((decode word).getD 0) := by
  obtain ⟨real,h⟩ := decode_finite word domain
  rw [h]
  rfl

#specNonCircular decode avoiding [forward,backward,runRows,Graph.eval]
#axiomsClean decode_finite
#axiomsClean decode_negative_infinity
#axiomsClean decode_finite_value
end
end VeriTile.Bench.Repairs.Ksoftmax.KernelIdeal
