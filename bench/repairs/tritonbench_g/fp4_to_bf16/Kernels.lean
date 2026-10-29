import bench.tritonbench_g.fp4_to_bf16.Kernels
import VeriTile.Triton.DSL.CallSignature

/-! The selected widened-index Python kernel and its actual helper are pasted
below. Source precision, masks, helper-call boundary and cast order are retained;
caller types and the wrapper's format constants stay outside both bodies. -/
namespace VeriTile.Bench.Repairs.Fp4ToBf16
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.Fp4ToBf16
namespace Kernels
def packedToBf16 : SourceFunction := triton_function _fp4_packed_to_bf16(x_packed, sign_mask_f4, mantissa_mask_f4, mbits_f4_e2m1, ebits_f4_e2m1, f4_e2m1_exp_bias, mbits_f32, ebits_f32, f32_exp_bias, zero_bits_f32, zero_point_five_bits_f32) in triton {
    """
    Input: a tensor of packed fp4 values
    Output: a tensor of bfloat16 values
    """

    # low-bits: original location 0:3
    # high-bits: original location 4:7
    x_low_bits = x_packed >> 4
    x_high_bits = x_packed & 0xF
    x = tl.interleave(x_low_bits, x_high_bits)

    # cast logic below
    # output = x_unpacked.to(tl.float32)

    # save the sign
    sign_f4 = x & sign_mask_f4

    # set everything to positive, will add sign back at the end
    x_pos = x ^ sign_f4

    # Special case zero
    zero_mask = x_pos == 0

    # There is only one denormal value in fp4: s001, which is 0.5 in f32
    # Special case it.
    # TODO(later): will it be faster to repeat this for all 8 positive
    # values instead of the bit manipulations?
    denormal_mask = x_pos == 1

    # calculate the new exponent and shift it to bits 2:9 of the result
    exp_biased_f4 = x_pos >> mbits_f4_e2m1
    exp_biased_f32 = exp_biased_f4 - f4_e2m1_exp_bias + f32_exp_bias
    exp_biased_f32 = exp_biased_f32.to(tl.int32) << mbits_f32

    # shift the mantissa to bits 10:32 of the result
    mantissa_f4 = x_pos & mantissa_mask_f4
    mantissa_f32 = mantissa_f4.to(tl.int32) << (mbits_f32 - mbits_f4_e2m1)
    output = mantissa_f32

    # combine the pieces
    result = exp_biased_f32 | mantissa_f32
    # result[zero_mask] = ZERO_BITS_F32
    result = tl.where(zero_mask, zero_bits_f32, result)
    # result[denormal_mask] = ZERO_POINT_FIVE_BITS_F32
    result = tl.where(denormal_mask, zero_point_five_bits_f32, result)

    # add sign back
    sign_f32 = sign_f4.to(tl.int32) << (
        mbits_f32 - mbits_f4_e2m1 + ebits_f32 - ebits_f4_e2m1
    )
    result = result | sign_f32

    # The bit shifting above is for float32, so for now we
    # bitcast to float32 and then regular cast to bfloat16
    # TODO(later): it should be pretty easy to cast directly to bf16, just
    # need to adjust the mbits/ebits/special values. Perf impact is likely
    # to be small as we would not be chaning memory access patterns.
    output = result.to(tl.float32, bitcast=True)
    output = output.to(tl.bfloat16)
    return output
}

def sourceSignature : CallSignature where
  pointers := [("x_ptr",.u8),("output_ptr",.bf16)]
  scalars := [("n_elements_in",.integer)]
  constexpr := ["sign_mask_f4","mantissa_mask_f4","mbits_f4_e2m1","ebits_f4_e2m1","f4_e2m1_exp_bias","mbits_f32","ebits_f32","f32_exp_bias","zero_bits_f32","zero_point_five_bits_f32","BLOCK_SIZE_IN"]
  functions := [packedToBf16]

def body (x_ptr output_ptr : RegionName) (n BLOCK_SIZE_IN : Nat) : ComputeKernel :=
  let sign_mask_f4 : Nat := 8
  let mantissa_mask_f4 : Nat := 1
  let mbits_f4_e2m1 : Nat := 1
  let ebits_f4_e2m1 : Nat := 2
  let f4_e2m1_exp_bias : Nat := 1
  let mbits_f32 : Nat := 23
  let ebits_f32 : Nat := 8
  let f32_exp_bias : Nat := 127
  let zero_bits_f32 : Nat := 0
  let zero_point_five_bits_f32 : Nat := 1056964608
  with_triton_call sourceSignature call_args [n_elements_in := n] in triton {
    pid = tl.program_id(axis=0).to(tl.int64)
    n_elements_in = tl.cast(n_elements_in, tl.int64)
    n_elements_out = n_elements_in * 2
    BLOCK_SIZE_OUT: tl.constexpr = BLOCK_SIZE_IN * 2

    block_start_in = pid * BLOCK_SIZE_IN
    offsets_in = block_start_in + tl.arange(0, BLOCK_SIZE_IN).to(tl.int64)

    mask_in = offsets_in < n_elements_in

    # packed uint8
    x_packed = tl.load(x_ptr + offsets_in, mask=mask_in)
    output = _fp4_packed_to_bf16(
        x_packed,
        sign_mask_f4,
        mantissa_mask_f4,
        mbits_f4_e2m1,
        ebits_f4_e2m1,
        f4_e2m1_exp_bias,
        mbits_f32,
        ebits_f32,
        f32_exp_bias,
        zero_bits_f32,
        zero_point_five_bits_f32,
    )

    # set up output offsets
    block_start_out = pid * BLOCK_SIZE_OUT
    offsets_out = block_start_out + tl.arange(0, BLOCK_SIZE_OUT).to(tl.int64)
    mask_out = offsets_out < n_elements_out

    tl.store(output_ptr + offsets_out, output, mask=mask_out)
  }

/- The native-count and finite-address certificates below retain their exact
existing statements; their connection to this actual body is reproved. -/
def source (p : FPValue.Parameters) := body p.input p.output p.count p.block
abbrev index := BitVec 64
/-- The cast precedes the multiply; both i32 and u32 count specializations
therefore denote the same nonnegative mathematical count after widening. -/
def count (n : Nat) : index := BitVec.ofNat 64 n
def outputCount (n : Nat) : index := count n * BitVec.ofNat 64 2
def inputIndex (pid lane : Nat) : index :=
  BitVec.ofNat 64 pid * BitVec.ofNat 64 512 + BitVec.ofNat 64 lane
def outputIndex (pid lane : Nat) : index :=
  BitVec.ofNat 64 pid * BitVec.ofNat 64 1024 + BitVec.ofNat 64 lane
def inputMask (n pid lane : Nat) : Bool := (inputIndex pid lane).slt (count n)
def outputMask (n pid lane : Nat) : Bool := (outputIndex pid lane).slt (outputCount n)
def pointer (base stride : Nat) (offset : index) : BitVec 64 :=
  BitVec.ofNat 64 base + BitVec.ofNat 64 stride * offset
end Kernels
end VeriTile.Bench.Repairs.Fp4ToBf16
