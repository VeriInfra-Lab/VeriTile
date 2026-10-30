import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_mtp_roll_packed_dx
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Kernels

def original : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vslice_backward_ptr : RegionName := "vslice_backward"
  let vslice_backward_1_ptr : RegionName := "vslice_backward_1"
  let vslice_backward_2_ptr : RegionName := "vslice_backward_2"
  let vslice_backward_3_ptr : RegionName := "vslice_backward_3"
  let vadd_ptr : RegionName := "vadd"
  let vslice_backward_4_ptr : RegionName := "vslice_backward_4"
  let vslice_backward_5_ptr : RegionName := "vslice_backward_5"
  let vadd_1_ptr : RegionName := "vadd_1"
  let vslice_backward_6_ptr : RegionName := "vslice_backward_6"
  let vslice_backward_7_ptr : RegionName := "vslice_backward_7"
  let vadd_2_ptr : RegionName := "vadd_2"
  triton {
    r0_z = tl.arange($(96))
    r0_value = tl.where((((r0_z % $(48)) >= $(1)) & ((r0_z % $(48)) < $(48)) & (((r0_z % $(48)) - $(1)) % $(1) == $(0))), tl.load($(input0_ptr) + ((((r0_z / $(48)) % $(2)) * $(128)) + ((((r0_z % $(48)) - $(1)) + $(80)) % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r0_value = r0_value.to(tl.bfloat16)
    r0_value = tl.broadcast_to(r0_value, [$(96)])
    tl.store($(vslice_backward_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(256))
    r1_value = tl.where((((r1_z % $(128)) >= $(80)) & ((r1_z % $(128)) < $(128)) & (((r1_z % $(128)) - $(80)) % $(1) == $(0))), tl.load($(vslice_backward_ptr) + ((((r1_z / $(128)) % $(2)) * $(48)) + (((r1_z % $(128)) - $(80)) % $(48))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r1_value = r1_value.to(tl.bfloat16)
    r1_value = tl.broadcast_to(r1_value, [$(256)])
    tl.store($(vslice_backward_1_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(64))
    r2_value = tl.where((((r2_z % $(32)) >= $(1)) & ((r2_z % $(32)) < $(32)) & (((r2_z % $(32)) - $(1)) % $(1) == $(0))), tl.load($(input0_ptr) + ((((r2_z / $(32)) % $(2)) * $(128)) + ((((r2_z % $(32)) - $(1)) + $(48)) % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r2_value = r2_value.to(tl.bfloat16)
    r2_value = tl.broadcast_to(r2_value, [$(64)])
    tl.store($(vslice_backward_2_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(256))
    r3_value = tl.where((((r3_z % $(128)) >= $(48)) & ((r3_z % $(128)) < $(80)) & (((r3_z % $(128)) - $(48)) % $(1) == $(0))), tl.load($(vslice_backward_2_ptr) + ((((r3_z / $(128)) % $(2)) * $(32)) + (((r3_z % $(128)) - $(48)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r3_value = r3_value.to(tl.bfloat16)
    r3_value = tl.broadcast_to(r3_value, [$(256)])
    tl.store($(vslice_backward_3_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(256))
    r4_value = (tl.load($(vslice_backward_1_ptr) + ((((r4_z / $(128)) % $(2)) * $(128)) + (r4_z % $(128))), dtype=tl.bfloat16) + tl.load($(vslice_backward_3_ptr) + ((((r4_z / $(128)) % $(2)) * $(128)) + (r4_z % $(128))), dtype=tl.bfloat16))
    r4_value = r4_value.to(tl.bfloat16)
    r4_value = tl.broadcast_to(r4_value, [$(256)])
    tl.store($(vadd_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(64))
    r5_value = tl.where((((r5_z % $(32)) >= $(1)) & ((r5_z % $(32)) < $(32)) & (((r5_z % $(32)) - $(1)) % $(1) == $(0))), tl.load($(input0_ptr) + ((((r5_z / $(32)) % $(2)) * $(128)) + ((((r5_z % $(32)) - $(1)) + $(16)) % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r5_value = r5_value.to(tl.bfloat16)
    r5_value = tl.broadcast_to(r5_value, [$(64)])
    tl.store($(vslice_backward_4_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(256))
    r6_value = tl.where((((r6_z % $(128)) >= $(16)) & ((r6_z % $(128)) < $(48)) & (((r6_z % $(128)) - $(16)) % $(1) == $(0))), tl.load($(vslice_backward_4_ptr) + ((((r6_z / $(128)) % $(2)) * $(32)) + (((r6_z % $(128)) - $(16)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r6_value = r6_value.to(tl.bfloat16)
    r6_value = tl.broadcast_to(r6_value, [$(256)])
    tl.store($(vslice_backward_5_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(256))
    r7_value = (tl.load($(vadd_ptr) + ((((r7_z / $(128)) % $(2)) * $(128)) + (r7_z % $(128))), dtype=tl.bfloat16) + tl.load($(vslice_backward_5_ptr) + ((((r7_z / $(128)) % $(2)) * $(128)) + (r7_z % $(128))), dtype=tl.bfloat16))
    r7_value = r7_value.to(tl.bfloat16)
    r7_value = tl.broadcast_to(r7_value, [$(256)])
    tl.store($(vadd_1_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(32))
    r8_value = tl.where((((r8_z % $(16)) >= $(1)) & ((r8_z % $(16)) < $(16)) & (((r8_z % $(16)) - $(1)) % $(1) == $(0))), tl.load($(input0_ptr) + ((((r8_z / $(16)) % $(2)) * $(128)) + (((r8_z % $(16)) - $(1)) % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r8_value = r8_value.to(tl.bfloat16)
    r8_value = tl.broadcast_to(r8_value, [$(32)])
    tl.store($(vslice_backward_6_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(256))
    r9_value = tl.where((((r9_z % $(128)) >= $(0)) & ((r9_z % $(128)) < $(16)) & (((r9_z % $(128)) - $(0)) % $(1) == $(0))), tl.load($(vslice_backward_6_ptr) + ((((r9_z / $(128)) % $(2)) * $(16)) + (((r9_z % $(128)) - $(0)) % $(16))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r9_value = r9_value.to(tl.bfloat16)
    r9_value = tl.broadcast_to(r9_value, [$(256)])
    tl.store($(vslice_backward_7_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(256))
    r10_value = (tl.load($(vadd_1_ptr) + ((((r10_z / $(128)) % $(2)) * $(128)) + (r10_z % $(128))), dtype=tl.bfloat16) + tl.load($(vslice_backward_7_ptr) + ((((r10_z / $(128)) % $(2)) * $(128)) + (r10_z % $(128))), dtype=tl.bfloat16))
    r10_value = r10_value.to(tl.bfloat16)
    r10_value = tl.broadcast_to(r10_value, [$(256)])
    tl.store($(vadd_2_ptr) + r10_z, r10_value)
    z = tl.arange($(256))
    value = tl.load($(vadd_2_ptr) + ((((z / $(128)) % $(2)) * $(128)) + (z % $(128))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(256)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let constant53d3e3c65e_ptr : RegionName := "constant53d3e3c65e"
  let out_ptr : RegionName := "out"
  let candidate_vsub_ptr : RegionName := "candidate_vsub"
  let candidate_vremainder_ptr : RegionName := "candidate_vremainder"
  let candidate_vindex_fill_1_ptr : RegionName := "candidate_vindex_fill_1"
  triton {
    r0_z = tl.arange($(4))
    r0_value = (tl.load($(constant53d3e3c65e_ptr) + (r0_z + $(1)), dtype=tl.int64) - tl.full([], $(1), dtype=tl.int64))
    r0_value = tl.broadcast_to(r0_value, [$(4)])
    tl.store($(candidate_vsub_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(4))
    r1_value = (tl.load($(candidate_vsub_ptr) + r1_z, dtype=tl.int64) % tl.full([], $(128), dtype=tl.int64))
    r1_value = tl.broadcast_to(r1_value, [$(4)])
    tl.store($(candidate_vremainder_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(256))
    r2_value = tl.broadcast_to(tl.load($(input0_ptr) + ((((r2_z / $(128)) % $(2)) * $(128)) + (r2_z % $(128))), dtype=tl.bfloat16), [$(256)])
    for r2_lane in range($(4)) {
      r2_value = tl.where(tl.load($(candidate_vremainder_ptr) + (r2_lane % $(4)), dtype=tl.int64) == tl.cast((r2_z % $(128)), tl.int64), tl.cast(0.0, tl.bfloat16), r2_value)
    }
    r2_value = r2_value.to(tl.bfloat16)
    r2_value = tl.broadcast_to(r2_value, [$(256)])
    tl.store($(candidate_vindex_fill_1_ptr) + r2_z, r2_value)
    z = tl.arange($(256))
    value = tl.load($(candidate_vindex_fill_1_ptr) + ((((z / $(128)) % $(2)) * $(128)) + (((z % $(128)) + $(127)) % $(128))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(256)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_mtp_roll_packed_dx
