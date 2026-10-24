import VeriTile.Triton
import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Float.SignedCast
import bench.tritonbench_g.fp4_to_bf16.Proofs.FP4Bits

namespace VeriTile.Bench.TritonBenchG.Fp4ToBf16.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false

/-- Pinned unscaled FP4 kernel with its single helper inlined and format
constexprs specialized to the wrapper's e2m1 values. Masked input has no other.
The upstream variable x_low_bits actually selects the high nibble. -/
def unscaled (X Y : RegionName) (n block : Nat) : ComputeKernel := triton {
  tl.region X = tl.uint8, Y = tl.bfloat16
  pid = tl.program_id(axis=0)
  n_elements_out = $(n) * $(2)
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
  block_start_out = pid * $(block * 2)
  offsets_out = block_start_out + tl.arange(0, $(block * 2))
  mask_out = offsets_out < n_elements_out
  tl.store(Y + offsets_out, output, mask=mask_out)
}


structure Parameters where
  input : RegionName
  output : RegionName
  count : Nat
  block : Nat

def source (p : Parameters) := unscaled p.input p.output p.count p.block
abbrev Word := FP.Value .fp32
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
def embed (x : FP.Value .bf16) : Word := ⟨x.bits.zeroExtend 32⟩
def backend (rest : Algebra Word) : Algebra Word :=
  { rest with fp32Bits := fun bits => ⟨bits.bits⟩
              cast := fun precision src dst x =>
                match src,dst with
                | .real,.bf16 | .fp32,.bf16 => embed (Bits.narrow x)
                | src,dst => rest.cast precision src dst x }
end
end VeriTile.Bench.TritonBenchG.Fp4ToBf16.FPValue
