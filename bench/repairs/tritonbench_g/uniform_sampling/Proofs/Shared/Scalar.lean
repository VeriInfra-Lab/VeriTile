import bench.repairs.tritonbench_g.uniform_sampling.Proofs.Shared.Bytes

namespace VeriTile.Bench.Repairs.UniformSampling.Shared.Scalar
open VeriTile.Triton
noncomputable section
def negate (word : Float32Bits) : Float32Bits := ⟨word.bits ^^^ 0x80000000⟩

theorem negate_twice (word : Float32Bits) : negate (negate word)=word := by
  cases word
  simp [negate,BitVec.xor_assoc]

/-- Exact representation only; nonfinite payloads have no Real denotation. -/
def project32 (word : Float32Bits) : Option ℝ :=
  let raw:=word.bits.toNat
  let exp : Nat :=(raw/2^23)%256
  let frac : Nat :=raw%2^23
  let sign:ℝ:=if raw/2^31=0 then 1 else -1
  if exp=255 then none else some (sign *
    (if exp=0 then (frac:ℝ)*(2:ℝ)^(-149:Int)
     else ((2^23+frac:Nat):ℝ)*(2:ℝ)^((exp:Int)-150)))

def project64 (word : Float64Bits) : Option ℝ :=
  let raw:=word.bits.toNat
  let exp : Nat :=(raw/2^52)%2048
  let frac : Nat :=raw%2^52
  let sign:ℝ:=if raw/2^63=0 then 1 else -1
  if exp=2047 then none else some (sign *
    (if exp=0 then (frac:ℝ)*(2:ℝ)^(-1074:Int)
     else ((2^52+frac:Nat):ℝ)*(2:ℝ)^((exp:Int)-1075)))


#axiomsClean negate_twice
end
end VeriTile.Bench.Repairs.UniformSampling.Shared.Scalar
