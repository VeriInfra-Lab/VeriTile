import VeriTile.Triton.DSL

/-! Complete repaired FP32 source. Signed64 widened indices, modular uint32
counter addition, all four canonical Philox10 fields, explicit FP32 affine FMA
and clamped masked stores are retained. The two bitcasts materialize the
runtime FP32 scalar argument words; extraction from Python float values is an
ABI boundary. No assumption excludes exceptional words or signed zero.
The explicit int64 clamp-bound cast records native operand promotion. The existing variant commutes only the
integer product that selects the Philox tuple. -/
namespace VeriTile.Bench.Repairs.UniformSampling
open Lean VeriTile.Triton

def kernel (out_ptr : RegionName) (N : Nat) (seed offset : Int) (lower upper : Float32Bits) (B : Nat) (hB : Random.ArangeSizeDomain B) : ComputeKernel :=
  triton {
    tl.ports inputs=[], outputs=[out_ptr]
    tl.region out_ptr = tl.float32
    tl.source_helper uint_to_uniform_float = tl.uint_to_uniform_float
    lower_arg = tl.bitcast($((lower.bits.toNat : Nat)), tl.float32)
    upper_arg = tl.bitcast($((upper.bits.toNat : Nat)), tl.float32)
    philox_seed_arg = $((seed : Int))
    tl.source_integer philox_seed_arg = tl.int64
    philox_offset_arg = $((offset : Int))
    tl.source_integer philox_offset_arg = tl.int64
    BLOCK_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize B hB)
    philox_seed = philox_seed_arg.to(tl.int64)
    philox_offset = philox_offset_arg.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    pid = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, BLOCK_arg).to(tl.int64)
    i4 = pid * BLOCK_arg + lane
    c0 += i4.to(tl.uint32)
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
    r0 = tl.fma(uint_to_uniform_float(r0), upper_arg - lower_arg, lower_arg)
    r1 = tl.fma(uint_to_uniform_float(r1), upper_arg - lower_arg, lower_arg)
    r2 = tl.fma(uint_to_uniform_float(r2), upper_arg - lower_arg, lower_arg)
    r3 = tl.fma(uint_to_uniform_float(r3), upper_arg - lower_arg, lower_arg)
    off_0 = pid * BLOCK_arg * 4 + lane
    off_1 = off_0 + BLOCK_arg
    off_2 = off_1 + BLOCK_arg
    off_3 = off_2 + BLOCK_arg
    tl.store(out_ptr + tl.minimum(off_0, ($(N) - $((1 : Nat))).to(tl.int64)), (r0).to(tl.float32), mask=off_0 < $(N), eviction_policy="evict_first")
    tl.store(out_ptr + tl.minimum(off_1, ($(N) - $((1 : Nat))).to(tl.int64)), (r1).to(tl.float32), mask=off_1 < $(N), eviction_policy="evict_first")
    tl.store(out_ptr + tl.minimum(off_2, ($(N) - $((1 : Nat))).to(tl.int64)), (r2).to(tl.float32), mask=off_2 < $(N), eviction_policy="evict_first")
    tl.store(out_ptr + tl.minimum(off_3, ($(N) - $((1 : Nat))).to(tl.int64)), (r3).to(tl.float32), mask=off_3 < $(N), eviction_policy="evict_first")
  }

def indexCommutedKernel (out_ptr : RegionName) (N : Nat) (seed offset : Int) (lower upper : Float32Bits) (B : Nat) (hB : Random.ArangeSizeDomain B) : ComputeKernel :=
  triton {
    tl.ports inputs=[], outputs=[out_ptr]
    tl.region out_ptr = tl.float32
    tl.source_helper uint_to_uniform_float = tl.uint_to_uniform_float
    lower_arg = tl.bitcast($((lower.bits.toNat : Nat)), tl.float32)
    upper_arg = tl.bitcast($((upper.bits.toNat : Nat)), tl.float32)
    philox_seed_arg = $((seed : Int))
    tl.source_integer philox_seed_arg = tl.int64
    philox_offset_arg = $((offset : Int))
    tl.source_integer philox_offset_arg = tl.int64
    BLOCK_arg = $(_root_.VeriTile.Triton.Random.i32ArangeSize B hB)
    philox_seed = philox_seed_arg.to(tl.int64)
    philox_offset = philox_offset_arg.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    pid = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, BLOCK_arg).to(tl.int64)
    i4 = BLOCK_arg * pid + lane
    c0 += i4.to(tl.uint32)
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
    r0 = tl.fma(uint_to_uniform_float(r0), upper_arg - lower_arg, lower_arg)
    r1 = tl.fma(uint_to_uniform_float(r1), upper_arg - lower_arg, lower_arg)
    r2 = tl.fma(uint_to_uniform_float(r2), upper_arg - lower_arg, lower_arg)
    r3 = tl.fma(uint_to_uniform_float(r3), upper_arg - lower_arg, lower_arg)
    off_0 = pid * BLOCK_arg * 4 + lane
    off_1 = off_0 + BLOCK_arg
    off_2 = off_1 + BLOCK_arg
    off_3 = off_2 + BLOCK_arg
    tl.store(out_ptr + tl.minimum(off_0, ($(N) - $((1 : Nat))).to(tl.int64)), (r0).to(tl.float32), mask=off_0 < $(N), eviction_policy="evict_first")
    tl.store(out_ptr + tl.minimum(off_1, ($(N) - $((1 : Nat))).to(tl.int64)), (r1).to(tl.float32), mask=off_1 < $(N), eviction_policy="evict_first")
    tl.store(out_ptr + tl.minimum(off_2, ($(N) - $((1 : Nat))).to(tl.int64)), (r2).to(tl.float32), mask=off_2 < $(N), eviction_policy="evict_first")
    tl.store(out_ptr + tl.minimum(off_3, ($(N) - $((1 : Nat))).to(tl.int64)), (r3).to(tl.float32), mask=off_3 < $(N), eviction_policy="evict_first")
  }

/-- Fixed argument names for the two displayed Python JIT bodies. -/
def originalKernel (N : Nat) (seed offset : Int) (lower upper : Float32Bits)
    (B : Nat) (hB : Random.ArangeSizeDomain B) :=
  kernel "out_ptr" N seed offset lower upper B hB

def candidateKernel (N : Nat) (seed offset : Int) (lower upper : Float32Bits)
    (B : Nat) (hB : Random.ArangeSizeDomain B) :=
  indexCommutedKernel "out_ptr" N seed offset lower upper B hB
end VeriTile.Bench.Repairs.UniformSampling
