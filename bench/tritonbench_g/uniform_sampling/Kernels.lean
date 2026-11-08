import VeriTile.Triton.DSL.PhiloxCallArguments

/-! Literal repaired random source and its integer-product candidate. Call-side
bindings retain all scalar ABI words and checked symbolic arange sizes. The
source keeps all four Philox fields, FP32 affine FMAs and clamped tail stores. -/
namespace VeriTile.Bench.Repairs.UniformSampling
open VeriTile.Triton

def uniformDecoderSource : UniformJitFunction := with_triton_params (x) constexpr_args [] in triton {
    if tl.constexpr(x.dtype == tl.uint32) or tl.constexpr(x.dtype == tl.int32):
        x = x.to(tl.int32, bitcast=True)
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

def callSignature : CallSignature where
  pointers := [("out_ptr",.fp32)]
  scalars := [("N",.integer),("philox_seed",.i64),("philox_offset",.i64),
    ("from_",.fp32),("to",.fp32)]
  constexpr := ["BLOCK"]
  writePorts := ["out_ptr"]

def runtimeKernel (out_ptr : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : Float32Bits) (BLOCK : Nat) (hBlock : Random.ArangeSizeDomain BLOCK) : ComputeKernel :=
  with_triton_call callSignature call_args [N := N]
    fp32_bits [from_ := lower, «to» := upper]
    i64_args [philox_seed := seed, philox_offset := offset]
    arange_args [BLOCK := hBlock] uniform_helpers [uint_to_uniform_float := uniformDecoderSource] in triton {
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    pid = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, BLOCK).to(tl.int64)
    i4 = pid * BLOCK + lane
    c0 += i4.to(tl.uint32)
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
    # Explicitly retain the default native affine FMA recorded for the source.
    r0 = tl.fma(uint_to_uniform_float(r0), to - from_, from_)
    r1 = tl.fma(uint_to_uniform_float(r1), to - from_, from_)
    r2 = tl.fma(uint_to_uniform_float(r2), to - from_, from_)
    r3 = tl.fma(uint_to_uniform_float(r3), to - from_, from_)
    off_0 = pid * BLOCK * 4 + lane
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK
    tl.store(out_ptr + tl.minimum(off_0, N - 1), r0, mask=off_0 < N,
             eviction_policy="evict_first")
    tl.store(out_ptr + tl.minimum(off_1, N - 1), r1, mask=off_1 < N,
             eviction_policy="evict_first")
    tl.store(out_ptr + tl.minimum(off_2, N - 1), r2, mask=off_2 < N,
             eviction_policy="evict_first")
    tl.store(out_ptr + tl.minimum(off_3, N - 1), r3, mask=off_3 < N,
             eviction_policy="evict_first")
  }

def runtimeCandidateKernel (out_ptr : RegionName) (N : Nat) (seed offset : Int)
    (lower upper : Float32Bits) (BLOCK : Nat) (hBlock : Random.ArangeSizeDomain BLOCK) : ComputeKernel :=
  with_triton_call callSignature call_args [N := N]
    fp32_bits [from_ := lower, «to» := upper]
    i64_args [philox_seed := seed, philox_offset := offset]
    arange_args [BLOCK := hBlock] uniform_helpers [uint_to_uniform_float := uniformDecoderSource] in triton {
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    pid = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, BLOCK).to(tl.int64)
    i4 = BLOCK * pid + lane
    c0 += i4.to(tl.uint32)
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
    # Explicitly retain the default native affine FMA recorded for the source.
    r0 = tl.fma(uint_to_uniform_float(r0), to - from_, from_)
    r1 = tl.fma(uint_to_uniform_float(r1), to - from_, from_)
    r2 = tl.fma(uint_to_uniform_float(r2), to - from_, from_)
    r3 = tl.fma(uint_to_uniform_float(r3), to - from_, from_)
    off_0 = pid * BLOCK * 4 + lane
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK
    tl.store(out_ptr + tl.minimum(off_0, N - 1), r0, mask=off_0 < N,
             eviction_policy="evict_first")
    tl.store(out_ptr + tl.minimum(off_1, N - 1), r1, mask=off_1 < N,
             eviction_policy="evict_first")
    tl.store(out_ptr + tl.minimum(off_2, N - 1), r2, mask=off_2 < N,
             eviction_policy="evict_first")
    tl.store(out_ptr + tl.minimum(off_3, N - 1), r3, mask=off_3 < N,
             eviction_policy="evict_first")
  }

def kernel := runtimeKernel
def indexCommutedKernel := runtimeCandidateKernel

end VeriTile.Bench.Repairs.UniformSampling
