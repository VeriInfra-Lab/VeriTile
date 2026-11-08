import bench.tritonbench_g.uniform_sampling.Kernels
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.PhiloxCallArguments
open VeriTile.Triton
open VeriTile.Bench.Repairs.UniformSampling
noncomputable section

/-- error: uniform helper requires the complete canonical dtype-dispatch source -/
#guard_msgs in
def changed_bitcast : UniformJitFunction := triton_function (x) {
    if tl.constexpr(x.dtype == tl.uint32) or tl.constexpr(x.dtype == tl.int32):
        x = x.to(tl.int32, bitcast=False)
        scale = 4.6566127342e-10
    else:
        tl.static_assert(
            tl.constexpr(x.dtype == tl.uint64) or tl.constexpr(x.dtype == tl.int64)
        )
        x = x.to(tl.int64, bitcast=True)
        scale = 1.0842020432385337e-19
    x = tl.where(x < 0, -x - 1, x)
    return x * scale
}

/-- error: uniform helper requires the complete canonical dtype-dispatch source -/
#guard_msgs in
def changed_unused_wide_branch : UniformJitFunction := triton_function (x) {
    if tl.constexpr(x.dtype == tl.uint32) or tl.constexpr(x.dtype == tl.int32):
        x = x.to(tl.int32, bitcast=True)
        scale = 4.6566127342e-10
    else:
        tl.static_assert(
            tl.constexpr(x.dtype == tl.uint32) or tl.constexpr(x.dtype == tl.int64)
        )
        x = x.to(tl.int64, bitcast=True)
        scale = 1.0842020432385337e-19
    x = tl.where(x < 0, -x - 1, x)
    return x * scale
}

/-- error: word arguments, arange sizes and uniform helpers must remain immutable: from_ += 1 -/
#guard_msgs in
def changed_word (out_ptr : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : Float32Bits) (BLOCK : Nat) (hBlock : Random.ArangeSizeDomain BLOCK) : ComputeKernel :=
  with_triton_call callSignature call_args [N := N]
    fp32_bits [from_ := lower, «to» := upper]
    i64_args [philox_seed := seed, philox_offset := offset]
    arange_args [BLOCK := hBlock] uniform_helpers [uint_to_uniform_float := uniformDecoderSource] in triton {
    from_ += 1
  }

/-- error: a loop cannot shadow a random call binding -/
#guard_msgs in
def shadowed_helper (out_ptr : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : Float32Bits) (BLOCK : Nat) (hBlock : Random.ArangeSizeDomain BLOCK) : ComputeKernel :=
  with_triton_call callSignature call_args [N := N]
    fp32_bits [from_ := lower, «to» := upper]
    i64_args [philox_seed := seed, philox_offset := offset]
    arange_args [BLOCK := hBlock] uniform_helpers [uint_to_uniform_float := uniformDecoderSource] in triton {
    for uint_to_uniform_float in range(0, N):
        temp = 0
  }

/-- error: word arguments, arange sizes and uniform helpers must remain immutable: philox_seed = philox_seed.to(tl.int32) -/
#guard_msgs in
def changed_integer (out_ptr : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : Float32Bits) (BLOCK : Nat) (hBlock : Random.ArangeSizeDomain BLOCK) : ComputeKernel :=
  with_triton_call callSignature call_args [N := N]
    fp32_bits [from_ := lower, «to» := upper]
    i64_args [philox_seed := seed, philox_offset := offset]
    arange_args [BLOCK := hBlock] uniform_helpers [uint_to_uniform_float := uniformDecoderSource] in triton {
    philox_seed = philox_seed.to(tl.int32)
  }

end
end VeriTile.Tests.PhiloxCallArguments
