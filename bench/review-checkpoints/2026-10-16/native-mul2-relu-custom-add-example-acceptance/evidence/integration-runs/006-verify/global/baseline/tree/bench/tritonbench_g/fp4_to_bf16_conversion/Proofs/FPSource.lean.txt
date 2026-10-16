import bench.tritonbench_g.fp4_to_bf16.Proofs.FPInteger
import VeriTile.Triton.Float.SignedCast
namespace VeriTile.Bench.TritonBenchG.Fp4ToBf16Conversion.FPValue
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

/-- Explicit int32-exponent repair candidate. The original int16 pow overload
is rejected by Triton 3.8.0. Both uint8 wraps, the signed16 subtract, and the
bf16 NaN constant are explicitly materialized; none is an identity cast. -/
def scaledCandidate (X S Y : RegionName) (n block groups : Nat) : ComputeKernel := triton {
  tl.region X = tl.uint8, S = tl.uint8, Y = tl.bfloat16
  pid = tl.program_id(axis=0)
  n_elements_out = $(n) * $(2)
  n_elements_s = n_elements_out // $(32)
  block_start_in = pid * $(block)
  offsets_in = block_start_in + tl.arange(0, $(block))
  mask_in = offsets_in < $(n)
  x_packed = tl.load(X + offsets_in, mask=mask_in)
  x_low_bits = x_packed >> $(4)
  x_high_bits = x_packed & $(15)
  x = tl.interleave(x_low_bits, x_high_bits)
  sign_f4 = x & $(8)
  x_pos = x ^ sign_f4
  zero_mask = x_pos == $(0)
  denormal_mask = x_pos == $(1)
  exp_biased_f4 = x_pos >> $(1)
  exp_biased_f32 = (exp_biased_f4.to(tl.int32) - $((1 : Int))).to(tl.uint8)
  exp_biased_f32 = (exp_biased_f32 + $(127)).to(tl.uint8)
  exp_biased_f32 = exp_biased_f32.to(tl.int32) << $(23)
  mantissa_f4 = x_pos & $(1)
  mantissa_f32 = mantissa_f4.to(tl.int32) << $(22)
  output = mantissa_f32
  result = exp_biased_f32 | mantissa_f32
  result = tl.where(zero_mask, $((0 : Int)).to(tl.int32), result)
  result = tl.where(denormal_mask, $((1056964608 : Int)).to(tl.int32), result)
  sign_f32 = sign_f4.to(tl.int32) << $(28)
  result = result | sign_f32
  output = result.to(tl.float32, bitcast=True)
  output = output.to(tl.bfloat16)
  block_start_s = pid * $(groups)
  offsets_s = block_start_s + tl.arange(0, $(groups))
  mask_s = offsets_s < n_elements_s
  s = tl.load(S + offsets_s, mask=mask_s)
  s_offset = (s.to(tl.int16) - $((127 : Int))).to(tl.int16)
  s_fp = tl.extra.cuda.libdevice.pow(($((2 : ℝ))).to(tl.float32), s_offset.to(tl.int32))
  s_fp = s_fp.to(tl.bfloat16)
  nan_bits = $((2143289344 : Int)).to(tl.int32)
  nan_fp32 = nan_bits.to(tl.float32, bitcast=True)
  nan_bf16 = nan_fp32.to(tl.bfloat16)
  s_fp = tl.where(s != $(255), s_fp, nan_bf16)
  output = tl.reshape(output, [$(groups), $(32)])
  s_fp = tl.reshape(s_fp, [$(groups), $(1)])
  output = output * s_fp
  output = tl.reshape(output, [$(block * 2)])
  block_start_out = pid * $(block * 2)
  offsets_out = block_start_out + tl.arange(0, $(block * 2))
  mask_out = offsets_out < n_elements_out
  tl.store(Y + offsets_out, output, mask=mask_out)
}

structure Parameters where
  input : RegionName
  scales : RegionName
  output : RegionName
  count : Nat
  groups : Nat

def Parameters.block (p : Parameters) : Nat := p.groups*16
def base (p : Parameters) : Fp4ToBf16.FPValue.Parameters :=
  ⟨p.input,p.output,p.count,p.block⟩
def offsetExpr (n : Nat) : Op .int [n] :=
  Op.intToSigned 16 (.sub .int .scalarR
    (Op.intToSigned 16 (.castNatToInt (.ref .nat [n] "s"))) (.constInt 127))
def offsetCode (p : Parameters) : List ComputeStmt :=
  [.assign .int [p.groups] "s_offset" (.alg (offsetExpr p.groups))]

/-- The repaired frontend preserves both signed16 conversions directly.
The full candidate is used unchanged; no task-local AST replacement remains. -/
def source (p : Parameters) : ComputeKernel :=
  scaledCandidate p.input p.scales p.output p.count p.block p.groups
abbrev Word := Fp4ToBf16.FPValue.Word
abbrev embed := Fp4ToBf16.FPValue.embed
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Explicit numerical primitives: integer power with fixed fp32 base 2
and bf16 multiplication. Neither callback has an assumed numerical law. -/
structure Primitives where
  pow2 : Int → Word
  mulBF16 : Word → Word → Word

def backend (rest : Algebra Word) (P : Primitives) : Algebra Word :=
  { Fp4ToBf16.FPValue.backend rest with
    literal := fun p d q => if q=2 then ⟨0x40000000⟩ else rest.literal p d q
    libdevicePowi := fun _ => some (fun _ n => P.pow2 n)
    binary := fun p d op x y => match d,op with
      | .bf16,.mul => P.mulBF16 x y
      | _,_ => rest.binary p d op x y
    cast := fun p src dst x => match src,dst with
      | .real,.bf16 | .fp32,.bf16 => embed (Bits.narrow x)
      | .real,.real | .fp32,.fp32 | .real,.fp32 | .fp32,.real | .bf16,.bf16 => x
      | _,_ => rest.cast p src dst x }

#axiomsClean source
end
end VeriTile.Bench.TritonBenchG.Fp4ToBf16Conversion.FPValue
