import bench.tritonbench_g.masked_add_cuda.Proofs.Graph
import bench.tritonbench_g.masked_add_cuda.Contract
import VeriTile.Triton.Float.ScalarOps

namespace VeriTile.Bench.Repairs.MaskedAddCuda.Words
open VeriTile.Triton
noncomputable section
abbrev Word := BitVec 32

/-- Independent binary64-to-FP32 conversion reference. Alpha's integer case
is admitted only at zero; Python normalizes it to floating zero before launch. -/
def scalarReference : AlphaArgument → Word
  | .integer _ | .other => 0
  | .floating bits =>
    let sign := decide (bits.toNat / 2^63 ≠ 0)
    let exponent : Nat := bits.toNat / 2^52 % 2048
    let fraction : Nat := bits.toNat % 2^52
    if exponent=2047 then
      if fraction=0 then (FP.Value.infinity .fp32 sign).bits else (FP.Value.qNaN .fp32).bits
    else
      let significand : Nat := if exponent=0 then fraction else 2^52+fraction
      let exponent : Int := (if exponent=0 then -1022 else (exponent : Int)-1023)-52
      let value := (significand : Rat)*FP.pow2 exponent
      (FP.round .fp32 FP.Config.ieeeValue (if sign then -value else value) sign).bits

def softwareFMA (alpha data grad : Word) : Word :=
  (FP.fma FP.Config.ieeeValue (⟨alpha⟩ : FP.Value .fp32) ⟨data⟩ ⟨grad⟩).bits

def finite (word : Word) : Prop := ((⟨word⟩ : FP.Value .fp32).toRat?).isSome

def FiniteScalar : AlphaArgument → Prop
  | .integer n => n = 0
  | .floating b => b.toNat / 2^52 % 2048 < 2047
  | .other => False

/-- Optional primitive-model comparison, retained for reference only. This is
not a task numerical experiment and is not imported by public FPEquiv. -/
def LocalScalarExact (convert : AlphaArgument → Word) : Prop :=
  ∀ s, FiniteScalar s → convert s = scalarReference s

/-- Optional FMA model-conformance predicate, not an optimization assumption
or an experimental obligation of this task. -/
def LocalFMAExact (fma : Word → Word → Word → Word) : Prop :=
  ∀ a p g, finite a → finite p → finite g → fma a p g = softwareFMA a p g

#specNonCircular LocalScalarExact avoiding [kernel, runGrid, FP.Structural.exec]
#specNonCircular LocalFMAExact avoiding [kernel, runGrid, FP.Structural.exec]
end
end VeriTile.Bench.Repairs.MaskedAddCuda.Words
