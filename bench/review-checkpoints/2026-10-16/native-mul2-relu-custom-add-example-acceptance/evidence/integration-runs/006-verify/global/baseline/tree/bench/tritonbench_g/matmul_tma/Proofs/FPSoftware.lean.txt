import bench.tritonbench_g.matmul_tma.Proofs.FPContract
import bench.tritonbench_g.int4_matmul.Proofs.FPMixed
namespace VeriTile.Bench.TritonBenchG.MatmulTma.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
abbrev Word := FP.Value .fp32
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Only the two used output conversions are supplied. Half values use the low
16 bits, with direct binary32-to-binary16 rounding (including overflow,
subnormals, signed zero and the explicit canonical-NaN policy). -/
def software (rest : Algebra Word) (cfg : FP.Config) : Algebra Word :=
  {rest with cast := fun precision src dst x =>
    match src, dst with
    | .fp32, .fp32 => x
    | .fp32, .fp16 => QuantizeKvTransform.MixedBits.narrow cfg x
    | src, dst => rest.cast precision src dst x}

def bitOutput (cfg : FP.Config) (D : Dot Word) (p : Parameters) (half : Bool)
    (s : State Word) (i : CIndex p) : Word :=
  let product := D.apply (some .fp32) {} (batch := []) (aValues p s) (bValues p s) i
  if half then QuantizeKvTransform.MixedBits.narrow cfg product else product

theorem output_software (rest : Algebra Word) (cfg : FP.Config) (D : Dot Word)
    (p : Parameters) (half : Bool) (s : State Word) :
    outputValues (software rest cfg) D p half s = bitOutput cfg D p half s := by
  cases half <;> rfl

/-- A complete software dot instance: decode true half inputs, multiply in
binary32, then add in ascending K order with positive-zero seed. Its schedule
is explicit and is not asserted to match a GPU tensor-core instruction. -/
def sequentialDot : Dot Word :=
  ⟨fun _ _ {batch} {m} {k} {n} a b => Int4Matmul.FPSoftware.dot (batch := batch) (m := m) (k := k) (n := n) a b⟩

#axiomsClean output_software
end
end VeriTile.Bench.TritonBenchG.MatmulTma.FPValue
