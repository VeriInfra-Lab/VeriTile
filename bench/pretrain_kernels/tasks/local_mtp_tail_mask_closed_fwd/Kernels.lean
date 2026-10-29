import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_mtp_tail_mask_closed_fwd
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Kernels

def original : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vfill_ptr : RegionName := "vfill"
  let vselect_scatter_ptr : RegionName := "vselect_scatter"
  let vfill_1_ptr : RegionName := "vfill_1"
  let vselect_scatter_1_ptr : RegionName := "vselect_scatter_1"
  let vfill_2_ptr : RegionName := "vfill_2"
  let vselect_scatter_2_ptr : RegionName := "vselect_scatter_2"
  let vfill_3_ptr : RegionName := "vfill_3"
  let vselect_scatter_3_ptr : RegionName := "vselect_scatter_3"
  let vmul_ptr : RegionName := "vmul"
  triton {
    r0_z = tl.arange($(2))
    r0_value = tl.cast(0.0, tl.bfloat16)
    r0_value = r0_value.to(tl.bfloat16)
    r0_value = tl.broadcast_to(r0_value, [$(2)])
    tl.store($(vfill_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(256))
    r1_value = tl.where(((r1_z % $(128)) == $(15)), tl.load($(vfill_ptr) + ((r1_z / $(128)) % $(2)), dtype=tl.bfloat16), tl.cast(1.0, tl.bfloat16))
    r1_value = r1_value.to(tl.bfloat16)
    r1_value = tl.broadcast_to(r1_value, [$(256)])
    tl.store($(vselect_scatter_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(2))
    r2_value = tl.cast(0.0, tl.bfloat16)
    r2_value = r2_value.to(tl.bfloat16)
    r2_value = tl.broadcast_to(r2_value, [$(2)])
    tl.store($(vfill_1_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(256))
    r3_value = tl.where(((r3_z % $(128)) == $(47)), tl.load($(vfill_1_ptr) + ((r3_z / $(128)) % $(2)), dtype=tl.bfloat16), tl.load($(vselect_scatter_ptr) + ((((r3_z / $(128)) % $(2)) * $(128)) + (r3_z % $(128))), dtype=tl.bfloat16))
    r3_value = r3_value.to(tl.bfloat16)
    r3_value = tl.broadcast_to(r3_value, [$(256)])
    tl.store($(vselect_scatter_1_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(2))
    r4_value = tl.cast(0.0, tl.bfloat16)
    r4_value = r4_value.to(tl.bfloat16)
    r4_value = tl.broadcast_to(r4_value, [$(2)])
    tl.store($(vfill_2_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(256))
    r5_value = tl.where(((r5_z % $(128)) == $(79)), tl.load($(vfill_2_ptr) + ((r5_z / $(128)) % $(2)), dtype=tl.bfloat16), tl.load($(vselect_scatter_1_ptr) + ((((r5_z / $(128)) % $(2)) * $(128)) + (r5_z % $(128))), dtype=tl.bfloat16))
    r5_value = r5_value.to(tl.bfloat16)
    r5_value = tl.broadcast_to(r5_value, [$(256)])
    tl.store($(vselect_scatter_2_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(2))
    r6_value = tl.cast(0.0, tl.bfloat16)
    r6_value = r6_value.to(tl.bfloat16)
    r6_value = tl.broadcast_to(r6_value, [$(2)])
    tl.store($(vfill_3_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(256))
    r7_value = tl.where(((r7_z % $(128)) == $(127)), tl.load($(vfill_3_ptr) + ((r7_z / $(128)) % $(2)), dtype=tl.bfloat16), tl.load($(vselect_scatter_2_ptr) + ((((r7_z / $(128)) % $(2)) * $(128)) + (r7_z % $(128))), dtype=tl.bfloat16))
    r7_value = r7_value.to(tl.bfloat16)
    r7_value = tl.broadcast_to(r7_value, [$(256)])
    tl.store($(vselect_scatter_3_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(256))
    r8_value = (tl.load($(input0_ptr) + ((((r8_z / $(128)) % $(2)) * $(128)) + (r8_z % $(128))), dtype=tl.bfloat16) * tl.load($(vselect_scatter_3_ptr) + ((((r8_z / $(128)) % $(2)) * $(128)) + (r8_z % $(128))), dtype=tl.bfloat16))
    r8_value = r8_value.to(tl.bfloat16)
    r8_value = tl.broadcast_to(r8_value, [$(256)])
    tl.store($(vmul_ptr) + r8_z, r8_value)
    z = tl.arange($(256))
    value = tl.load($(vmul_ptr) + ((((z / $(128)) % $(2)) * $(128)) + (z % $(128))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(256)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let constant53d3e3c65e_ptr : RegionName := "constant53d3e3c65e"
  let out_ptr : RegionName := "out"
  let vmul_ptr : RegionName := "vmul"
  let candidate_vsub_ptr : RegionName := "candidate_vsub"
  let candidate_vremainder_ptr : RegionName := "candidate_vremainder"
  let candidate_vindex_fill_ptr : RegionName := "candidate_vindex_fill"
  let candidate_vmul_ptr : RegionName := "candidate_vmul"
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
    r2_value = tl.broadcast_to(tl.cast(1.0, tl.bfloat16), [$(256)])
    for r2_lane in range($(4)) {
      r2_value = tl.where(tl.load($(candidate_vremainder_ptr) + (r2_lane % $(4)), dtype=tl.int64) == tl.cast((r2_z % $(128)), tl.int64), tl.cast(0.0, tl.bfloat16), r2_value)
    }
    r2_value = r2_value.to(tl.bfloat16)
    r2_value = tl.broadcast_to(r2_value, [$(256)])
    tl.store($(candidate_vindex_fill_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(256))
    r3_value = (tl.load($(input0_ptr) + ((((r3_z / $(128)) % $(2)) * $(128)) + (r3_z % $(128))), dtype=tl.bfloat16) * tl.load($(candidate_vindex_fill_ptr) + ((((r3_z / $(128)) % $(2)) * $(128)) + (r3_z % $(128))), dtype=tl.bfloat16))
    r3_value = r3_value.to(tl.bfloat16)
    r3_value = tl.broadcast_to(r3_value, [$(256)])
    tl.store($(candidate_vmul_ptr) + r3_z, r3_value)
    z = tl.arange($(256))
    value = tl.load($(candidate_vmul_ptr) + ((((z / $(128)) % $(2)) * $(128)) + (z % $(128))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(256)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_mtp_tail_mask_closed_fwd
