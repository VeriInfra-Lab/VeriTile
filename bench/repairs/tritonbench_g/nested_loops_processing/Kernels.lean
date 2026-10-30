import VeriTile.Triton.DSL

/-! The repaired source stores homogeneous int32 payloads without arithmetic.
The natural index channel models the already widened nonnegative int64
coordinates; the separate finite-address proof justifies this interpretation.
Both genuine source variants retain all loops, cache loads, masks, clamping,
duplicate-lane suppression and uniform barriers. -/
namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.Kernels
open VeriTile.Triton

abbrev originalKernel (in_ptr out_ptr : RegionName) (count cols : Nat) : ComputeKernel := triton {
      tl.region in_ptr = tl.int32, out_ptr = tl.int32
      offs_am = tl.arange(0, 2)
      offs_an = tl.arange(0, 2)
      a_offsets = offs_am[:, None] * $(cols) + offs_an[None, :] * $(1)
      offs_cm = tl.arange(0, 2)
      offs_cn = tl.arange(0, 2)
      c_offsets = $(cols) * offs_cm[:, None] + $(1) * offs_cn[None, :]
      unique = (offs_cm[:, None] == $(0)) | ($(cols) > $(1)) | (offs_cn[None, :] == $(1))
      for i in range(0, $(2), $(1)) {
        a1 = tl.load(in_ptr + tl.minimum(a_offsets, $(count - 1)), mask=a_offsets < $(count), other=0)
        for j in range(0, $(2), $(1)) {
          a_offsets += $(2)
          a2 = tl.load(in_ptr + tl.minimum(a_offsets, $(count - 1)), mask=a_offsets < $(count), other=0)
          for k in range(0, $(2), $(1)) {
            a_offsets += $(2)
            a3 = tl.load(in_ptr + tl.minimum(a_offsets, $(count - 1)), mask=a_offsets < $(count), other=0)
            tl.store(out_ptr + tl.minimum(c_offsets, $(count - 1)), a1, mask=(c_offsets < $(count)) & unique)
            tl.debug_barrier()
            c_offsets += $(2)
            tl.store(out_ptr + tl.minimum(c_offsets, $(count - 1)), a2, mask=(c_offsets < $(count)) & unique)
            tl.debug_barrier()
            c_offsets += $(2)
            tl.store(out_ptr + tl.minimum(c_offsets, $(count - 1)), a3, mask=(c_offsets < $(count)) & unique)
            tl.debug_barrier()
            c_offsets += $(2)
          }
        }
        a_offsets += $(2)
      }
}

abbrev commutedKernel (in_ptr out_ptr : RegionName) (count cols : Nat) : ComputeKernel := triton {
      tl.region in_ptr = tl.int32, out_ptr = tl.int32
      offs_am = tl.arange(0, 2)
      offs_an = tl.arange(0, 2)
      a_offsets = offs_am[:, None] * $(cols) + $(1) * offs_an[None, :]
      offs_cm = tl.arange(0, 2)
      offs_cn = tl.arange(0, 2)
      c_offsets = $(cols) * offs_cm[:, None] + $(1) * offs_cn[None, :]
      unique = (offs_cm[:, None] == $(0)) | ($(cols) > $(1)) | (offs_cn[None, :] == $(1))
      for i in range(0, $(2), $(1)) {
        a1 = tl.load(in_ptr + tl.minimum(a_offsets, $(count - 1)), mask=a_offsets < $(count), other=0)
        for j in range(0, $(2), $(1)) {
          a_offsets += $(2)
          a2 = tl.load(in_ptr + tl.minimum(a_offsets, $(count - 1)), mask=a_offsets < $(count), other=0)
          for k in range(0, $(2), $(1)) {
            a_offsets += $(2)
            a3 = tl.load(in_ptr + tl.minimum(a_offsets, $(count - 1)), mask=a_offsets < $(count), other=0)
            tl.store(out_ptr + tl.minimum(c_offsets, $(count - 1)), a1, mask=(c_offsets < $(count)) & unique)
            tl.debug_barrier()
            c_offsets += $(2)
            tl.store(out_ptr + tl.minimum(c_offsets, $(count - 1)), a2, mask=(c_offsets < $(count)) & unique)
            tl.debug_barrier()
            c_offsets += $(2)
            tl.store(out_ptr + tl.minimum(c_offsets, $(count - 1)), a3, mask=(c_offsets < $(count)) & unique)
            tl.debug_barrier()
            c_offsets += $(2)
          }
        }
        a_offsets += $(2)
      }
}

def kernel (candidate : Bool) : RegionName → RegionName → Nat → Nat → ComputeKernel :=
  if candidate then commutedKernel else originalKernel

def source (count cols : Nat) : ComputeKernel := kernel Bool.false "input" "output" count cols

def commuted (count cols : Nat) : ComputeKernel := kernel Bool.true "input" "output" count cols

end VeriTile.Bench.Repairs.NestedLoopsProcessing.Kernels
