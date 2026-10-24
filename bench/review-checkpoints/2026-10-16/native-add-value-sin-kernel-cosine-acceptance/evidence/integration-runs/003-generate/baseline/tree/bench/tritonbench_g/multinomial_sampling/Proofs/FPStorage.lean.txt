import bench.tritonbench_g.multinomial_sampling.Proofs.FPBackend
import bench.tritonbench_g.matmul_dequant_int4.Proofs.Binary16
import VeriTile.Triton.Float.ScalarOps
namespace VeriTile.Bench.TritonBenchG.MultinomialSampling.FPValue
set_option maxHeartbeats 10000000
set_option maxRecDepth 24000
open VeriTile.Triton
abbrev Word := BitVec 64
abbrev Half := MatmulDequantInt4.Binary16.Value

def f32 (x : Word) : FP.Value .fp32 := ⟨x.truncate 32⟩
def put32 (x : FP.Value .fp32) : Word := x.bits.zeroExtend 64

def decodedTo32 : FP.Decoded → FP.Value .fp32
  | .nan .. => FP.Value.qNaN .fp32
  | .infinity sign => FP.Value.infinity .fp32 sign
  | .finite sign magnitude => FP.round .fp32 .ieeeValue (if sign then -magnitude else magnitude) sign

/-- Actual binary64 interpretation: 1 sign, 11 exponent, 52 fraction bits and
bias1023. It covers all payloads; finite narrowing rounds an exact rational.
This software profile canonicalizes NaNs and preserves signed zero. -/
def decode64 (x : Word) : FP.Decoded :=
  let sign := x.toNat/2^63 != 0
  let exponent : Nat := x.toNat/2^52%2048
  let fraction : Nat := x.toNat%2^52
  if exponent=2047 then
    if fraction=0 then .infinity sign else .nan sign (fraction<2^51) fraction
  else
    let significand : Nat := if exponent=0 then fraction else 2^52+fraction
    let power : Int := (if exponent=0 then -1022 else (exponent:Int)-1023)-52
    .finite sign (Rat.ofInt significand*FP.pow2 power)

def halfTo32 (x : Word) : Word := put32 (decodedTo32 (MatmulDequantInt4.Binary16.Value.decode ⟨x.truncate 16⟩))
def bf16To32 (x : Word) : Word := put32 (FP.cast .fp32 .ieeeValue (⟨x.truncate 16⟩ : FP.Value .bf16))
def doubleTo32 (x : Word) : Word := put32 (decodedTo32 (decode64 x))
def toHalf (x : Word) : Word :=
  let value := match (f32 x).decode with
    | .nan .. => MatmulDequantInt4.Binary16.Value.qNaN
    | .infinity sign => MatmulDequantInt4.Binary16.Value.infinity sign
    | .finite sign magnitude => MatmulDequantInt4.Binary16.round .ieeeValue
        (if sign then -magnitude else magnitude) sign
  value.bits.zeroExtend 64

def storageCast (d : Storage) (_precision : Option ComputeDType)
    (src dst : FloatDType) (x : Word) : Word :=
  match src,dst with
  | .fp16,.real => halfTo32 x
  | .bf16,.real => bf16To32 x
  | .real,.real => if d=.fp64 then doubleTo32 x else put32 (f32 x)
  | .real,.fp16 => toHalf x
  | .real,.bf16 => (FP.cast .bf16 .ieeeValue (f32 x)).bits.zeroExtend 64
  | _,_ => x

/-- Only these five cast pairs occur in this task; no claim is made about the
arbitrary total extension on unsupported pairs in storageCast. -/
theorem storage_boundary_vectors :
    doubleTo32 0x8000000000000000 = 0x80000000 ∧
    doubleTo32 0x3ff0000000000000 = 0x3f800000 ∧
    doubleTo32 0x3ff0000010000000 = 0x3f800000 ∧
    doubleTo32 0x3ff0000010000001 = 0x3f800001 ∧
    doubleTo32 0x36a0000000000000 = 1 ∧
    doubleTo32 1 = 0 ∧
    doubleTo32 0x8000000000000001 = 0x80000000 ∧
    doubleTo32 0x7ff0000000000000 = 0x7f800000 ∧
    doubleTo32 0x7ff0000000000001 = 0x7fc00000 ∧
    halfTo32 0x8000 = 0x80000000 ∧
    halfTo32 1 = 0x33800000 ∧
    halfTo32 0x7bff = 0x477fe000 ∧
    bf16To32 1 = 0x10000 ∧
    toHalf 0x80000000 = 0x8000 := by
  decide +kernel

#axiomsClean storage_boundary_vectors
end VeriTile.Bench.TritonBenchG.MultinomialSampling.FPValue
