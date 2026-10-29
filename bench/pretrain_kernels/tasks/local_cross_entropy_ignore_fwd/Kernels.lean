import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_cross_entropy_ignore_fwd
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Kernels

def original : ComputeKernel :=
  let parameter0_ptr : RegionName := "parameter0"
  let parameter1_ptr : RegionName := "parameter1"
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vge_ptr : RegionName := "vge"
  let vwhere_ptr : RegionName := "vwhere"
  let vwhere_1_ptr : RegionName := "vwhere_1"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vmax_1_value_ptr : RegionName := "vmax_1_value"
  let vmax_1_index_ptr : RegionName := "vmax_1_index"
  let vsub_ptr : RegionName := "vsub"
  let vexp_ptr : RegionName := "vexp"
  let vsum_1_ptr : RegionName := "vsum_1"
  let vlog_ptr : RegionName := "vlog"
  let vgather_ptr : RegionName := "vgather"
  let vsub_1_ptr : RegionName := "vsub_1"
  let vsub_2_ptr : RegionName := "vsub_2"
  let vneg_ptr : RegionName := "vneg"
  let vwhere_2_ptr : RegionName := "vwhere_2"
  triton {
    r0_z = tl.arange($(256))
    r0_value = (tl.load($(parameter1_ptr) + ((((r0_z / $(2)) % $(128)) * $(2)) + (r0_z % $(2))), dtype=tl.float32) >= 0.125)
    r0_value = tl.broadcast_to(r0_value, [$(256)])
    tl.store($(vge_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(256))
    r1_value = tl.where(tl.load($(vge_ptr) + ((((r1_z / $(2)) % $(128)) * $(2)) + (r1_z % $(2))), dtype=tl.int1), tl.load($(parameter0_ptr) + ((((r1_z / $(2)) % $(128)) * $(2)) + (r1_z % $(2))), dtype=tl.int64), (-tl.full([], $(100), dtype=tl.int64)))
    r1_value = tl.broadcast_to(r1_value, [$(256)])
    tl.store($(vwhere_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(256))
    r2_value = tl.where(tl.load($(vge_ptr) + ((((r2_z / $(2)) % $(128)) * $(2)) + (r2_z % $(2))), dtype=tl.int1), tl.load($(vwhere_ptr) + ((((r2_z / $(2)) % $(128)) * $(2)) + (r2_z % $(2))), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    r2_value = tl.broadcast_to(r2_value, [$(256)])
    tl.store($(vwhere_1_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(262144))
    r3_value = tl.load($(input0_ptr) + ((((r3_z / $(2048)) % $(128)) * $(2048)) + (((r3_z / $(1024)) % $(2)) * $(1024)) + (r3_z % $(1024))), dtype=tl.bfloat16)
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(262144)])
    tl.store($(vto_copy_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(256))
    r4_lane = tl.arange($(1024))
    r4_row_values = tl.load($(vto_copy_ptr) + ((((r4_z[:, None] / $(2)) % $(128)) * $(2048)) + ((r4_z[:, None] % $(2)) * $(1024)) + (r4_lane[None, :] % $(1024))), dtype=tl.float32)
    r4_value = tl.max(r4_row_values, axis=1)
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(256)])
    tl.store($(vmax_1_value_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(256))
    r5_lane = tl.arange($(1024))
    r5_row_values = tl.load($(vto_copy_ptr) + ((((r5_z[:, None] / $(2)) % $(128)) * $(2048)) + ((r5_z[:, None] % $(2)) * $(1024)) + (r5_lane[None, :] % $(1024))), dtype=tl.float32)
    r5_value = tl.argmax(r5_row_values, axis=1)
    r5_value = tl.cast(r5_value, tl.int64)
    r5_value = tl.broadcast_to(r5_value, [$(256)])
    tl.store($(vmax_1_index_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(262144))
    r6_value = (tl.load($(vto_copy_ptr) + ((((r6_z / $(2048)) % $(128)) * $(2048)) + (((r6_z / $(1024)) % $(2)) * $(1024)) + (r6_z % $(1024))), dtype=tl.float32) - tl.load($(vmax_1_value_ptr) + ((((r6_z / $(2048)) % $(128)) * $(2)) + ((r6_z / $(1024)) % $(2))), dtype=tl.float32))
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(262144)])
    tl.store($(vsub_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(262144))
    r7_value = tl.exp(tl.load($(vsub_ptr) + ((((r7_z / $(2048)) % $(128)) * $(2048)) + (((r7_z / $(1024)) % $(2)) * $(1024)) + (r7_z % $(1024))), dtype=tl.float32))
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(262144)])
    tl.store($(vexp_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(256))
    r8_lane = tl.arange($(1024))
    r8_value = tl.sum(tl.load($(vexp_ptr) + ((((r8_z[:, None] / $(2)) % $(128)) * $(2048)) + ((r8_z[:, None] % $(2)) * $(1024)) + r8_lane[None, :]), dtype=tl.float32), axis=1)
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(256)])
    tl.store($(vsum_1_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(256))
    r9_value = tl.log(tl.load($(vsum_1_ptr) + ((((r9_z / $(2)) % $(128)) * $(2)) + (r9_z % $(2))), dtype=tl.float32))
    r9_value = r9_value.to(tl.float32)
    r9_value = tl.broadcast_to(r9_value, [$(256)])
    tl.store($(vlog_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(256))
    r10_value = tl.load($(vto_copy_ptr) + (((((r10_z / $(2)) % $(128)) * $(2048)) + ((r10_z % $(2)) * $(1024))) + (tl.load($(vwhere_1_ptr) + ((((r10_z / $(2)) % $(128)) * $(2)) + (r10_z % $(2))), dtype=tl.int64) % $(1024))), dtype=tl.float32)
    r10_value = r10_value.to(tl.float32)
    r10_value = tl.broadcast_to(r10_value, [$(256)])
    tl.store($(vgather_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(256))
    r11_value = (tl.load($(vgather_ptr) + ((((r11_z / $(2)) % $(128)) * $(2)) + (r11_z % $(2))), dtype=tl.float32) - tl.load($(vmax_1_value_ptr) + ((((r11_z / $(2)) % $(128)) * $(2)) + (r11_z % $(2))), dtype=tl.float32))
    r11_value = r11_value.to(tl.float32)
    r11_value = tl.broadcast_to(r11_value, [$(256)])
    tl.store($(vsub_1_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(256))
    r12_value = (tl.load($(vsub_1_ptr) + ((((r12_z / $(2)) % $(128)) * $(2)) + (r12_z % $(2))), dtype=tl.float32) - tl.load($(vlog_ptr) + ((((r12_z / $(2)) % $(128)) * $(2)) + (r12_z % $(2))), dtype=tl.float32))
    r12_value = r12_value.to(tl.float32)
    r12_value = tl.broadcast_to(r12_value, [$(256)])
    tl.store($(vsub_2_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(256))
    r13_value = -tl.load($(vsub_2_ptr) + ((((r13_z / $(2)) % $(128)) * $(2)) + (r13_z % $(2))), dtype=tl.float32)
    r13_value = r13_value.to(tl.float32)
    r13_value = tl.broadcast_to(r13_value, [$(256)])
    tl.store($(vneg_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(256))
    r14_value = tl.where(tl.load($(vge_ptr) + ((((r14_z / $(2)) % $(128)) * $(2)) + (r14_z % $(2))), dtype=tl.int1), tl.load($(vneg_ptr) + ((((r14_z / $(2)) % $(128)) * $(2)) + (r14_z % $(2))), dtype=tl.float32), 0.0)
    r14_value = r14_value.to(tl.float32)
    r14_value = tl.broadcast_to(r14_value, [$(256)])
    tl.store($(vwhere_2_ptr) + r14_z, r14_value)
    z = tl.arange($(256))
    value = tl.load($(vwhere_2_ptr) + ((((z / $(2)) % $(128)) * $(2)) + (z % $(2))), dtype=tl.float32)
    value = tl.broadcast_to(value, [$(256)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let parameter0_ptr : RegionName := "parameter0"
  let parameter1_ptr : RegionName := "parameter1"
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vge_ptr : RegionName := "vge"
  let vwhere_ptr : RegionName := "vwhere"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vmax_1_value_ptr : RegionName := "vmax_1_value"
  let vmax_1_index_ptr : RegionName := "vmax_1_index"
  let vsub_ptr : RegionName := "vsub"
  let vexp_ptr : RegionName := "vexp"
  let vsum_1_ptr : RegionName := "vsum_1"
  let vlog_ptr : RegionName := "vlog"
  let vgather_ptr : RegionName := "vgather"
  let vsub_1_ptr : RegionName := "vsub_1"
  let vsub_2_ptr : RegionName := "vsub_2"
  let vneg_ptr : RegionName := "vneg"
  let vwhere_2_ptr : RegionName := "vwhere_2"
  let candidate_vge_ptr : RegionName := "candidate_vge"
  let candidate_vwhere_ptr : RegionName := "candidate_vwhere"
  let candidate_vto_copy_ptr : RegionName := "candidate_vto_copy"
  let candidate_vne_ptr : RegionName := "candidate_vne"
  let candidate_vmax_1_value_ptr : RegionName := "candidate_vmax_1_value"
  let candidate_vmax_1_index_ptr : RegionName := "candidate_vmax_1_index"
  let candidate_vsub_ptr : RegionName := "candidate_vsub"
  let candidate_vexp_ptr : RegionName := "candidate_vexp"
  let candidate_vsum_1_ptr : RegionName := "candidate_vsum_1"
  let candidate_vclamp_min_ptr : RegionName := "candidate_vclamp_min"
  let candidate_vgather_ptr : RegionName := "candidate_vgather"
  let candidate_vsub_1_ptr : RegionName := "candidate_vsub_1"
  let candidate_vlog_ptr : RegionName := "candidate_vlog"
  let candidate_vsub_2_ptr : RegionName := "candidate_vsub_2"
  let candidate_vneg_ptr : RegionName := "candidate_vneg"
  let candidate_vwhere_2_ptr : RegionName := "candidate_vwhere_2"
  triton {
    r0_z = tl.arange($(256))
    r0_value = (tl.load($(parameter1_ptr) + ((((r0_z / $(2)) % $(128)) * $(2)) + (r0_z % $(2))), dtype=tl.float32) >= 0.125)
    r0_value = tl.broadcast_to(r0_value, [$(256)])
    tl.store($(candidate_vge_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(256))
    r1_value = tl.where(tl.load($(candidate_vge_ptr) + ((((r1_z / $(2)) % $(128)) * $(2)) + (r1_z % $(2))), dtype=tl.int1), tl.load($(parameter0_ptr) + ((((r1_z / $(2)) % $(128)) * $(2)) + (r1_z % $(2))), dtype=tl.int64), (-tl.full([], $(100), dtype=tl.int64)))
    r1_value = tl.broadcast_to(r1_value, [$(256)])
    tl.store($(candidate_vwhere_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(262144))
    r2_value = tl.load($(input0_ptr) + ((((r2_z / $(2048)) % $(128)) * $(2048)) + (((r2_z / $(1024)) % $(2)) * $(1024)) + (r2_z % $(1024))), dtype=tl.bfloat16)
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(262144)])
    tl.store($(candidate_vto_copy_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(256))
    r3_value = (tl.load($(candidate_vwhere_ptr) + ((((r3_z / $(2)) % $(128)) * $(2)) + (r3_z % $(2))), dtype=tl.int64) != (-tl.full([], $(100), dtype=tl.int64)))
    r3_value = tl.broadcast_to(r3_value, [$(256)])
    tl.store($(candidate_vne_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(256))
    r4_lane = tl.arange($(1024))
    r4_row_values = tl.load($(candidate_vto_copy_ptr) + ((((r4_z[:, None] / $(2)) % $(128)) * $(2048)) + ((r4_z[:, None] % $(2)) * $(1024)) + (r4_lane[None, :] % $(1024))), dtype=tl.float32)
    r4_value = tl.max(r4_row_values, axis=1)
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(256)])
    tl.store($(candidate_vmax_1_value_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(256))
    r5_lane = tl.arange($(1024))
    r5_row_values = tl.load($(candidate_vto_copy_ptr) + ((((r5_z[:, None] / $(2)) % $(128)) * $(2048)) + ((r5_z[:, None] % $(2)) * $(1024)) + (r5_lane[None, :] % $(1024))), dtype=tl.float32)
    r5_value = tl.argmax(r5_row_values, axis=1)
    r5_value = tl.cast(r5_value, tl.int64)
    r5_value = tl.broadcast_to(r5_value, [$(256)])
    tl.store($(candidate_vmax_1_index_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(262144))
    r6_value = (tl.load($(candidate_vto_copy_ptr) + ((((r6_z / $(2048)) % $(128)) * $(2048)) + (((r6_z / $(1024)) % $(2)) * $(1024)) + (r6_z % $(1024))), dtype=tl.float32) - tl.load($(candidate_vmax_1_value_ptr) + ((((r6_z / $(2048)) % $(128)) * $(2)) + ((r6_z / $(1024)) % $(2))), dtype=tl.float32))
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(262144)])
    tl.store($(candidate_vsub_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(262144))
    r7_value = tl.exp(tl.load($(candidate_vsub_ptr) + ((((r7_z / $(2048)) % $(128)) * $(2048)) + (((r7_z / $(1024)) % $(2)) * $(1024)) + (r7_z % $(1024))), dtype=tl.float32))
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(262144)])
    tl.store($(candidate_vexp_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(256))
    r8_lane = tl.arange($(1024))
    r8_value = tl.sum(tl.load($(candidate_vexp_ptr) + ((((r8_z[:, None] / $(2)) % $(128)) * $(2048)) + ((r8_z[:, None] % $(2)) * $(1024)) + r8_lane[None, :]), dtype=tl.float32), axis=1)
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(256)])
    tl.store($(candidate_vsum_1_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(256))
    r9_value = tl.maximum(tl.load($(candidate_vwhere_ptr) + ((((r9_z / $(2)) % $(128)) * $(2)) + (r9_z % $(2))), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    r9_value = tl.broadcast_to(r9_value, [$(256)])
    tl.store($(candidate_vclamp_min_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(256))
    r10_value = tl.load($(candidate_vto_copy_ptr) + (((((r10_z / $(2)) % $(128)) * $(2048)) + ((r10_z % $(2)) * $(1024))) + (tl.load($(candidate_vclamp_min_ptr) + ((((r10_z / $(2)) % $(128)) * $(2)) + (r10_z % $(2))), dtype=tl.int64) % $(1024))), dtype=tl.float32)
    r10_value = r10_value.to(tl.float32)
    r10_value = tl.broadcast_to(r10_value, [$(256)])
    tl.store($(candidate_vgather_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(256))
    r11_value = (tl.load($(candidate_vgather_ptr) + ((((r11_z / $(2)) % $(128)) * $(2)) + (r11_z % $(2))), dtype=tl.float32) - tl.load($(candidate_vmax_1_value_ptr) + ((((r11_z / $(2)) % $(128)) * $(2)) + (r11_z % $(2))), dtype=tl.float32))
    r11_value = r11_value.to(tl.float32)
    r11_value = tl.broadcast_to(r11_value, [$(256)])
    tl.store($(candidate_vsub_1_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(256))
    r12_value = tl.log(tl.load($(candidate_vsum_1_ptr) + ((((r12_z / $(2)) % $(128)) * $(2)) + (r12_z % $(2))), dtype=tl.float32))
    r12_value = r12_value.to(tl.float32)
    r12_value = tl.broadcast_to(r12_value, [$(256)])
    tl.store($(candidate_vlog_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(256))
    r13_value = (tl.load($(candidate_vsub_1_ptr) + ((((r13_z / $(2)) % $(128)) * $(2)) + (r13_z % $(2))), dtype=tl.float32) - tl.load($(candidate_vlog_ptr) + ((((r13_z / $(2)) % $(128)) * $(2)) + (r13_z % $(2))), dtype=tl.float32))
    r13_value = r13_value.to(tl.float32)
    r13_value = tl.broadcast_to(r13_value, [$(256)])
    tl.store($(candidate_vsub_2_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(256))
    r14_value = -tl.load($(candidate_vsub_2_ptr) + ((((r14_z / $(2)) % $(128)) * $(2)) + (r14_z % $(2))), dtype=tl.float32)
    r14_value = r14_value.to(tl.float32)
    r14_value = tl.broadcast_to(r14_value, [$(256)])
    tl.store($(candidate_vneg_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(256))
    r15_value = tl.where(tl.load($(candidate_vne_ptr) + ((((r15_z / $(2)) % $(128)) * $(2)) + (r15_z % $(2))), dtype=tl.int1), tl.load($(candidate_vneg_ptr) + ((((r15_z / $(2)) % $(128)) * $(2)) + (r15_z % $(2))), dtype=tl.float32), 0.0)
    r15_value = r15_value.to(tl.float32)
    r15_value = tl.broadcast_to(r15_value, [$(256)])
    tl.store($(candidate_vwhere_2_ptr) + r15_z, r15_value)
    z = tl.arange($(256))
    value = tl.load($(candidate_vwhere_2_ptr) + ((((z / $(2)) % $(128)) * $(2)) + (z % $(2))), dtype=tl.float32)
    value = tl.broadcast_to(value, [$(256)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_cross_entropy_ignore_fwd
