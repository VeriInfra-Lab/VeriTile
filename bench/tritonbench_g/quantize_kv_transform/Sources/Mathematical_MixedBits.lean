import bench.tritonbench_g.quantize_kv_copy.Sources.Mathematical_FP32Profile
import bench.tritonbench_g.matrix_reduction.Proofs.BinaryConversion
import bench.tritonbench_g.matmul_dequant_int4.Sources.Mathematical_Binary16

/-! Explicit binary16 storage and binary32 compute for quantize-kv-transform.
Half payloads occupy the low sixteen bits of the carrier; fp32 intermediates
occupy all thirty-two. Dtype-directed callbacks interpret each representation.
Narrowing rounds the exact decoded fp32 rational directly to true binary16;
it is neither identity nor the inverse-only matrix maximum narrowing helper.
NaN narrowing canonicalizes, and widening uses the explicit binary encoding.
This is a software profile with a separate native-lowering obligation. -/
namespace VeriTile.Bench.TritonBenchG.QuantizeKvTransform.MixedBits
open VeriTile.Triton FP.Structural
abbrev Word := FP.Value .fp32
abbrev Half := MatmulDequantInt4.Binary16.Value

def half (x : Word) : Half := ⟨x.bits.truncate 16⟩
def embed (x : Half) : Word := ⟨x.bits.zeroExtend 32⟩
def widen (x : Word) : Word := ⟨MatrixReduction.BinaryConversion.widen (half x).bits⟩
def narrow (cfg : FP.Config) (x : Word) : Word := embed <|
  match (FP.prepare cfg x).decode with
  | .nan .. => MatmulDequantInt4.Binary16.Value.qNaN
  | .infinity negative => MatmulDequantInt4.Binary16.Value.infinity negative
  | .finite negative magnitude =>
      MatmulDequantInt4.Binary16.round cfg (if negative then -magnitude else magnitude) negative

def absHalf (x : Word) : Word :=
  ⟨(x.bits.truncate 16 &&& (0x7fff : BitVec 16)).zeroExtend 32⟩

noncomputable def backend (rest : Algebra Word) (cfg : FP.Config)
    (u : Nat → Word → Int) : Algebra Word :=
  let base := QuantizeKvCopy.TypedQuantize.backend rest cfg u
  { base with
    literal := fun p d x => if x = 0 then FP.Value.zero .fp32
      else base.literal p d x
    unary := fun p op x => match op with
      | .abs .fp16 => absHalf x
      | _ => base.unary p op x
    cast := fun p src dst x => match src, dst with
      | .fp16, .real | .fp16, .fp32 => widen x
      | .real, .fp16 | .fp32, .fp16 => narrow cfg x
      | .fp16, .fp16 => embed (half x)
      | _, _ => base.cast p src dst x }

theorem half_embedding (x : Half) : half (embed x) = x := by
  cases x with
  | mk bits =>
    simp [half, embed, BitVec.truncate, BitVec.zeroExtend]
    exact (BitVec.setWidth_setWidth_of_le bits (by decide : 16 ≤ 32)).trans (BitVec.setWidth_eq bits)

theorem storage_boundary_vectors :
    (narrow .ieeeValue (FP.Value.ofBits .fp32 0x3f800000)).bits = BitVec.ofNat 32 0x3c00 ∧
    (narrow .ieeeValue (FP.Value.ofBits .fp32 0x33800000)).bits = BitVec.ofNat 32 1 ∧
    (narrow .ieeeValue (FP.Value.ofBits .fp32 0x33000000)).bits = BitVec.ofNat 32 0 ∧
    (narrow .ieeeValue (FP.Value.ofBits .fp32 0x477ff000)).bits = BitVec.ofNat 32 0x7c00 ∧
    (narrow .ieeeValue (FP.Value.ofBits .fp32 0x80000000)).bits = BitVec.ofNat 32 0x8000 ∧
    (widen (embed (MatmulDequantInt4.Binary16.Value.ofBits 1))).bits = BitVec.ofNat 32 0x33800000 ∧
    (widen (embed (MatmulDequantInt4.Binary16.Value.ofBits 0x8000))).bits = BitVec.ofNat 32 0x80000000 ∧
    (absHalf (embed (MatmulDequantInt4.Binary16.Value.ofBits 0xfc01))).bits = BitVec.ofNat 32 0x7c01 := by
  decide +kernel

#axiomsClean half_embedding
#axiomsClean storage_boundary_vectors
end VeriTile.Bench.TritonBenchG.QuantizeKvTransform.MixedBits
