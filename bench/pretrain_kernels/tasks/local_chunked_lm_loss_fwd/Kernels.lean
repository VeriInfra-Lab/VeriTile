import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_chunked_lm_loss_fwd
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Kernels

def original : ComputeKernel :=
  let parameter0_ptr : RegionName := "parameter0"
  let parameter1_ptr : RegionName := "parameter1"
  let parameter2_ptr : RegionName := "parameter2"
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vmul_ptr : RegionName := "vmul"
  let vgt_ptr : RegionName := "vgt"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vmm_ptr : RegionName := "vmm"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
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
  let vmul_1_ptr : RegionName := "vmul_1"
  let vsum_2_ptr : RegionName := "vsum_2"
  triton {
    r0_z = tl.arange($(65536))
    r0_value = (tl.load($(parameter0_ptr) + ((((r0_z / $(64)) % $(1024)) * $(64)) + (r0_z % $(64))), dtype=tl.bfloat16) * tl.cast(0.05, tl.bfloat16))
    r0_value = r0_value.to(tl.bfloat16)
    r0_value = tl.broadcast_to(r0_value, [$(65536)])
    tl.store($(vmul_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(320))
    r1_value = (tl.load($(parameter2_ptr) + r1_z, dtype=tl.float32) > 0.3)
    r1_value = tl.broadcast_to(r1_value, [$(320)])
    tl.store($(vgt_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(320))
    r2_value = tl.where(tl.load($(vgt_ptr) + r2_z, dtype=tl.int1), 1.0, 0.0)
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(320)])
    tl.store($(vto_copy_ptr) + r2_z, r2_value)
    r3_row = tl.arange($(320))
    r3_col = tl.arange($(1024))
    r3_lane = tl.arange($(64))
    r3_left = tl.load($(input0_ptr) + (((r3_row[:, None] % $(320)) * $(64)) + (r3_lane[None, :] % $(64))), dtype=tl.bfloat16)
    r3_right = tl.load($(vmul_ptr) + (((r3_col[None, :] % $(1024)) * $(64)) + (r3_lane[:, None] % $(64))), dtype=tl.bfloat16)
    r3_value = tl.dot(r3_left, r3_right)
    r3_value = r3_value.to(tl.bfloat16)
    tl.store($(vmm_ptr) + ((r3_row[:, None] * $(1024)) + r3_col[None, :]), r3_value)
    r4_z = tl.arange($(327680))
    r4_value = tl.load($(vmm_ptr) + ((((r4_z / $(1024)) % $(320)) * $(1024)) + (r4_z % $(1024))), dtype=tl.bfloat16)
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(327680)])
    tl.store($(vto_copy_1_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(320))
    r5_lane = tl.arange($(1024))
    r5_row_values = tl.load($(vto_copy_1_ptr) + ((r5_z[:, None] * $(1024)) + (r5_lane[None, :] % $(1024))), dtype=tl.float32)
    r5_value = tl.max(r5_row_values, axis=1)
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(320)])
    tl.store($(vmax_1_value_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(320))
    r6_lane = tl.arange($(1024))
    r6_row_values = tl.load($(vto_copy_1_ptr) + ((r6_z[:, None] * $(1024)) + (r6_lane[None, :] % $(1024))), dtype=tl.float32)
    r6_value = tl.argmax(r6_row_values, axis=1)
    r6_value = tl.cast(r6_value, tl.int64)
    r6_value = tl.broadcast_to(r6_value, [$(320)])
    tl.store($(vmax_1_index_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(327680))
    r7_value = (tl.load($(vto_copy_1_ptr) + ((((r7_z / $(1024)) % $(320)) * $(1024)) + (r7_z % $(1024))), dtype=tl.float32) - tl.load($(vmax_1_value_ptr) + ((r7_z / $(1024)) % $(320)), dtype=tl.float32))
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(327680)])
    tl.store($(vsub_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(327680))
    r8_value = tl.exp(tl.load($(vsub_ptr) + ((((r8_z / $(1024)) % $(320)) * $(1024)) + (r8_z % $(1024))), dtype=tl.float32))
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(327680)])
    tl.store($(vexp_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(320))
    r9_lane = tl.arange($(1024))
    r9_value = tl.sum(tl.load($(vexp_ptr) + ((r9_z[:, None] * $(1024)) + r9_lane[None, :]), dtype=tl.float32), axis=1)
    r9_value = r9_value.to(tl.float32)
    r9_value = tl.broadcast_to(r9_value, [$(320)])
    tl.store($(vsum_1_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(320))
    r10_value = tl.log(tl.load($(vsum_1_ptr) + r10_z, dtype=tl.float32))
    r10_value = r10_value.to(tl.float32)
    r10_value = tl.broadcast_to(r10_value, [$(320)])
    tl.store($(vlog_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(320))
    r11_value = tl.load($(vto_copy_1_ptr) + ((r11_z * $(1024)) + (tl.load($(parameter1_ptr) + r11_z, dtype=tl.int64) % $(1024))), dtype=tl.float32)
    r11_value = r11_value.to(tl.float32)
    r11_value = tl.broadcast_to(r11_value, [$(320)])
    tl.store($(vgather_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(320))
    r12_value = (tl.load($(vgather_ptr) + r12_z, dtype=tl.float32) - tl.load($(vmax_1_value_ptr) + r12_z, dtype=tl.float32))
    r12_value = r12_value.to(tl.float32)
    r12_value = tl.broadcast_to(r12_value, [$(320)])
    tl.store($(vsub_1_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(320))
    r13_value = (tl.load($(vsub_1_ptr) + r13_z, dtype=tl.float32) - tl.load($(vlog_ptr) + r13_z, dtype=tl.float32))
    r13_value = r13_value.to(tl.float32)
    r13_value = tl.broadcast_to(r13_value, [$(320)])
    tl.store($(vsub_2_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(320))
    r14_value = -tl.load($(vsub_2_ptr) + r14_z, dtype=tl.float32)
    r14_value = r14_value.to(tl.float32)
    r14_value = tl.broadcast_to(r14_value, [$(320)])
    tl.store($(vneg_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(320))
    r15_value = (tl.load($(vneg_ptr) + r15_z, dtype=tl.float32) * tl.load($(vto_copy_ptr) + r15_z, dtype=tl.float32))
    r15_value = r15_value.to(tl.float32)
    r15_value = tl.broadcast_to(r15_value, [$(320)])
    tl.store($(vmul_1_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(1))
    r16_lane = tl.arange($(320))
    r16_value = tl.sum(tl.load($(vmul_1_ptr) + r16_lane[None, :], dtype=tl.float32), axis=1)
    r16_value = r16_value.to(tl.float32)
    r16_value = tl.broadcast_to(r16_value, [$(1)])
    tl.store($(vsum_2_ptr) + r16_z, r16_value)
    z = tl.arange($(1))
    value = tl.load($(vsum_2_ptr) + $(0), dtype=tl.float32)
    value = tl.broadcast_to(value, [$(1)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let parameter0_ptr : RegionName := "parameter0"
  let parameter1_ptr : RegionName := "parameter1"
  let parameter2_ptr : RegionName := "parameter2"
  let input0_ptr : RegionName := "input0"
  let out_ptr : RegionName := "out"
  let vmul_ptr : RegionName := "vmul"
  let vgt_ptr : RegionName := "vgt"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vmm_ptr : RegionName := "vmm"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
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
  let vmul_1_ptr : RegionName := "vmul_1"
  let vsum_2_ptr : RegionName := "vsum_2"
  let candidate_vmul_ptr : RegionName := "candidate_vmul"
  let candidate_vgt_ptr : RegionName := "candidate_vgt"
  let candidate_vto_copy_ptr : RegionName := "candidate_vto_copy"
  let candidate_vmm_ptr : RegionName := "candidate_vmm"
  let candidate_vto_copy_1_ptr : RegionName := "candidate_vto_copy_1"
  let candidate_vmax_1_value_ptr : RegionName := "candidate_vmax_1_value"
  let candidate_vmax_1_index_ptr : RegionName := "candidate_vmax_1_index"
  let candidate_vsub_ptr : RegionName := "candidate_vsub"
  let candidate_vexp_ptr : RegionName := "candidate_vexp"
  let candidate_vsum_1_ptr : RegionName := "candidate_vsum_1"
  let candidate_vlog_ptr : RegionName := "candidate_vlog"
  let candidate_vgather_ptr : RegionName := "candidate_vgather"
  let candidate_vsub_1_ptr : RegionName := "candidate_vsub_1"
  let candidate_vsub_2_ptr : RegionName := "candidate_vsub_2"
  let candidate_vneg_ptr : RegionName := "candidate_vneg"
  let candidate_vmul_1_ptr : RegionName := "candidate_vmul_1"
  let candidate_vsum_2_ptr : RegionName := "candidate_vsum_2"
  let candidate_vadd_ptr : RegionName := "candidate_vadd"
  let candidate_vmm_1_ptr : RegionName := "candidate_vmm_1"
  let candidate_vto_copy_2_ptr : RegionName := "candidate_vto_copy_2"
  let candidate_vmax_2_value_ptr : RegionName := "candidate_vmax_2_value"
  let candidate_vmax_2_index_ptr : RegionName := "candidate_vmax_2_index"
  let candidate_vsub_3_ptr : RegionName := "candidate_vsub_3"
  let candidate_vexp_1_ptr : RegionName := "candidate_vexp_1"
  let candidate_vsum_3_ptr : RegionName := "candidate_vsum_3"
  let candidate_vlog_1_ptr : RegionName := "candidate_vlog_1"
  let candidate_vgather_1_ptr : RegionName := "candidate_vgather_1"
  let candidate_vsub_4_ptr : RegionName := "candidate_vsub_4"
  let candidate_vsub_5_ptr : RegionName := "candidate_vsub_5"
  let candidate_vneg_1_ptr : RegionName := "candidate_vneg_1"
  let candidate_vmul_2_ptr : RegionName := "candidate_vmul_2"
  let candidate_vsum_4_ptr : RegionName := "candidate_vsum_4"
  let candidate_vadd_1_ptr : RegionName := "candidate_vadd_1"
  let candidate_vmm_2_ptr : RegionName := "candidate_vmm_2"
  let candidate_vto_copy_3_ptr : RegionName := "candidate_vto_copy_3"
  let candidate_vmax_3_value_ptr : RegionName := "candidate_vmax_3_value"
  let candidate_vmax_3_index_ptr : RegionName := "candidate_vmax_3_index"
  let candidate_vsub_6_ptr : RegionName := "candidate_vsub_6"
  let candidate_vexp_2_ptr : RegionName := "candidate_vexp_2"
  let candidate_vsum_5_ptr : RegionName := "candidate_vsum_5"
  let candidate_vlog_2_ptr : RegionName := "candidate_vlog_2"
  let candidate_vgather_2_ptr : RegionName := "candidate_vgather_2"
  let candidate_vsub_7_ptr : RegionName := "candidate_vsub_7"
  let candidate_vsub_8_ptr : RegionName := "candidate_vsub_8"
  let candidate_vneg_2_ptr : RegionName := "candidate_vneg_2"
  let candidate_vmul_3_ptr : RegionName := "candidate_vmul_3"
  let candidate_vsum_6_ptr : RegionName := "candidate_vsum_6"
  let candidate_vadd_2_ptr : RegionName := "candidate_vadd_2"
  let candidate_vmm_3_ptr : RegionName := "candidate_vmm_3"
  let candidate_vto_copy_4_ptr : RegionName := "candidate_vto_copy_4"
  let candidate_vmax_4_value_ptr : RegionName := "candidate_vmax_4_value"
  let candidate_vmax_4_index_ptr : RegionName := "candidate_vmax_4_index"
  let candidate_vsub_9_ptr : RegionName := "candidate_vsub_9"
  let candidate_vexp_3_ptr : RegionName := "candidate_vexp_3"
  let candidate_vsum_7_ptr : RegionName := "candidate_vsum_7"
  let candidate_vlog_3_ptr : RegionName := "candidate_vlog_3"
  let candidate_vgather_3_ptr : RegionName := "candidate_vgather_3"
  let candidate_vsub_10_ptr : RegionName := "candidate_vsub_10"
  let candidate_vsub_11_ptr : RegionName := "candidate_vsub_11"
  let candidate_vneg_3_ptr : RegionName := "candidate_vneg_3"
  let candidate_vmul_4_ptr : RegionName := "candidate_vmul_4"
  let candidate_vsum_8_ptr : RegionName := "candidate_vsum_8"
  let candidate_vadd_3_ptr : RegionName := "candidate_vadd_3"
  triton {
    r0_z = tl.arange($(65536))
    r0_value = (tl.load($(parameter0_ptr) + ((((r0_z / $(64)) % $(1024)) * $(64)) + (r0_z % $(64))), dtype=tl.bfloat16) * tl.cast(0.05, tl.bfloat16))
    r0_value = r0_value.to(tl.bfloat16)
    r0_value = tl.broadcast_to(r0_value, [$(65536)])
    tl.store($(candidate_vmul_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(320))
    r1_value = (tl.load($(parameter2_ptr) + r1_z, dtype=tl.float32) > 0.3)
    r1_value = tl.broadcast_to(r1_value, [$(320)])
    tl.store($(candidate_vgt_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(320))
    r2_value = tl.where(tl.load($(candidate_vgt_ptr) + r2_z, dtype=tl.int1), 1.0, 0.0)
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(320)])
    tl.store($(candidate_vto_copy_ptr) + r2_z, r2_value)
    r3_row = tl.arange($(96))
    r3_col = tl.arange($(1024))
    r3_lane = tl.arange($(64))
    r3_left = tl.load($(input0_ptr) + ((((((r3_row[:, None] * $(64)) + r3_lane[None, :]) / $(64)) % $(320)) * $(64)) + (((r3_row[:, None] * $(64)) + r3_lane[None, :]) % $(64))), dtype=tl.bfloat16)
    r3_right = tl.load($(candidate_vmul_ptr) + (((r3_col[None, :] % $(1024)) * $(64)) + (r3_lane[:, None] % $(64))), dtype=tl.bfloat16)
    r3_value = tl.dot(r3_left, r3_right)
    r3_value = r3_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_ptr) + ((r3_row[:, None] * $(1024)) + r3_col[None, :]), r3_value)
    r4_z = tl.arange($(98304))
    r4_value = tl.load($(candidate_vmm_ptr) + ((((r4_z / $(1024)) % $(96)) * $(1024)) + (r4_z % $(1024))), dtype=tl.bfloat16)
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(98304)])
    tl.store($(candidate_vto_copy_1_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(96))
    r5_lane = tl.arange($(1024))
    r5_row_values = tl.load($(candidate_vto_copy_1_ptr) + ((r5_z[:, None] * $(1024)) + (r5_lane[None, :] % $(1024))), dtype=tl.float32)
    r5_value = tl.max(r5_row_values, axis=1)
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(96)])
    tl.store($(candidate_vmax_1_value_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(96))
    r6_lane = tl.arange($(1024))
    r6_row_values = tl.load($(candidate_vto_copy_1_ptr) + ((r6_z[:, None] * $(1024)) + (r6_lane[None, :] % $(1024))), dtype=tl.float32)
    r6_value = tl.argmax(r6_row_values, axis=1)
    r6_value = tl.cast(r6_value, tl.int64)
    r6_value = tl.broadcast_to(r6_value, [$(96)])
    tl.store($(candidate_vmax_1_index_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(98304))
    r7_value = (tl.load($(candidate_vto_copy_1_ptr) + ((((r7_z / $(1024)) % $(96)) * $(1024)) + (r7_z % $(1024))), dtype=tl.float32) - tl.load($(candidate_vmax_1_value_ptr) + ((r7_z / $(1024)) % $(96)), dtype=tl.float32))
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(98304)])
    tl.store($(candidate_vsub_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(98304))
    r8_value = tl.exp(tl.load($(candidate_vsub_ptr) + ((((r8_z / $(1024)) % $(96)) * $(1024)) + (r8_z % $(1024))), dtype=tl.float32))
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(98304)])
    tl.store($(candidate_vexp_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(96))
    r9_lane = tl.arange($(1024))
    r9_value = tl.sum(tl.load($(candidate_vexp_ptr) + ((r9_z[:, None] * $(1024)) + r9_lane[None, :]), dtype=tl.float32), axis=1)
    r9_value = r9_value.to(tl.float32)
    r9_value = tl.broadcast_to(r9_value, [$(96)])
    tl.store($(candidate_vsum_1_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(96))
    r10_value = tl.log(tl.load($(candidate_vsum_1_ptr) + r10_z, dtype=tl.float32))
    r10_value = r10_value.to(tl.float32)
    r10_value = tl.broadcast_to(r10_value, [$(96)])
    tl.store($(candidate_vlog_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(96))
    r11_value = tl.load($(candidate_vto_copy_1_ptr) + ((r11_z * $(1024)) + (tl.load($(parameter1_ptr) + r11_z, dtype=tl.int64) % $(1024))), dtype=tl.float32)
    r11_value = r11_value.to(tl.float32)
    r11_value = tl.broadcast_to(r11_value, [$(96)])
    tl.store($(candidate_vgather_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(96))
    r12_value = (tl.load($(candidate_vgather_ptr) + r12_z, dtype=tl.float32) - tl.load($(candidate_vmax_1_value_ptr) + r12_z, dtype=tl.float32))
    r12_value = r12_value.to(tl.float32)
    r12_value = tl.broadcast_to(r12_value, [$(96)])
    tl.store($(candidate_vsub_1_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(96))
    r13_value = (tl.load($(candidate_vsub_1_ptr) + r13_z, dtype=tl.float32) - tl.load($(candidate_vlog_ptr) + r13_z, dtype=tl.float32))
    r13_value = r13_value.to(tl.float32)
    r13_value = tl.broadcast_to(r13_value, [$(96)])
    tl.store($(candidate_vsub_2_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(96))
    r14_value = -tl.load($(candidate_vsub_2_ptr) + r14_z, dtype=tl.float32)
    r14_value = r14_value.to(tl.float32)
    r14_value = tl.broadcast_to(r14_value, [$(96)])
    tl.store($(candidate_vneg_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(96))
    r15_value = (tl.load($(candidate_vneg_ptr) + r15_z, dtype=tl.float32) * tl.load($(candidate_vto_copy_ptr) + r15_z, dtype=tl.float32))
    r15_value = r15_value.to(tl.float32)
    r15_value = tl.broadcast_to(r15_value, [$(96)])
    tl.store($(candidate_vmul_1_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(1))
    r16_lane = tl.arange($(96))
    r16_value = tl.sum(tl.load($(candidate_vmul_1_ptr) + r16_lane[None, :], dtype=tl.float32), axis=1)
    r16_value = r16_value.to(tl.float32)
    r16_value = tl.broadcast_to(r16_value, [$(1)])
    tl.store($(candidate_vsum_2_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(1))
    r17_value = (0.0 + tl.load($(candidate_vsum_2_ptr) + $(0), dtype=tl.float32))
    r17_value = r17_value.to(tl.float32)
    r17_value = tl.broadcast_to(r17_value, [$(1)])
    tl.store($(candidate_vadd_ptr) + r17_z, r17_value)
    r18_row = tl.arange($(96))
    r18_col = tl.arange($(1024))
    r18_lane = tl.arange($(64))
    r18_left = tl.load($(input0_ptr) + ((((((r18_row[:, None] * $(64)) + r18_lane[None, :] + $(6144)) / $(64)) % $(320)) * $(64)) + (((r18_row[:, None] * $(64)) + r18_lane[None, :] + $(6144)) % $(64))), dtype=tl.bfloat16)
    r18_right = tl.load($(candidate_vmul_ptr) + (((r18_col[None, :] % $(1024)) * $(64)) + (r18_lane[:, None] % $(64))), dtype=tl.bfloat16)
    r18_value = tl.dot(r18_left, r18_right)
    r18_value = r18_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_1_ptr) + ((r18_row[:, None] * $(1024)) + r18_col[None, :]), r18_value)
    r19_z = tl.arange($(98304))
    r19_value = tl.load($(candidate_vmm_1_ptr) + ((((r19_z / $(1024)) % $(96)) * $(1024)) + (r19_z % $(1024))), dtype=tl.bfloat16)
    r19_value = r19_value.to(tl.float32)
    r19_value = tl.broadcast_to(r19_value, [$(98304)])
    tl.store($(candidate_vto_copy_2_ptr) + r19_z, r19_value)
    r20_z = tl.arange($(96))
    r20_lane = tl.arange($(1024))
    r20_row_values = tl.load($(candidate_vto_copy_2_ptr) + ((r20_z[:, None] * $(1024)) + (r20_lane[None, :] % $(1024))), dtype=tl.float32)
    r20_value = tl.max(r20_row_values, axis=1)
    r20_value = r20_value.to(tl.float32)
    r20_value = tl.broadcast_to(r20_value, [$(96)])
    tl.store($(candidate_vmax_2_value_ptr) + r20_z, r20_value)
    r21_z = tl.arange($(96))
    r21_lane = tl.arange($(1024))
    r21_row_values = tl.load($(candidate_vto_copy_2_ptr) + ((r21_z[:, None] * $(1024)) + (r21_lane[None, :] % $(1024))), dtype=tl.float32)
    r21_value = tl.argmax(r21_row_values, axis=1)
    r21_value = tl.cast(r21_value, tl.int64)
    r21_value = tl.broadcast_to(r21_value, [$(96)])
    tl.store($(candidate_vmax_2_index_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(98304))
    r22_value = (tl.load($(candidate_vto_copy_2_ptr) + ((((r22_z / $(1024)) % $(96)) * $(1024)) + (r22_z % $(1024))), dtype=tl.float32) - tl.load($(candidate_vmax_2_value_ptr) + ((r22_z / $(1024)) % $(96)), dtype=tl.float32))
    r22_value = r22_value.to(tl.float32)
    r22_value = tl.broadcast_to(r22_value, [$(98304)])
    tl.store($(candidate_vsub_3_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(98304))
    r23_value = tl.exp(tl.load($(candidate_vsub_3_ptr) + ((((r23_z / $(1024)) % $(96)) * $(1024)) + (r23_z % $(1024))), dtype=tl.float32))
    r23_value = r23_value.to(tl.float32)
    r23_value = tl.broadcast_to(r23_value, [$(98304)])
    tl.store($(candidate_vexp_1_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(96))
    r24_lane = tl.arange($(1024))
    r24_value = tl.sum(tl.load($(candidate_vexp_1_ptr) + ((r24_z[:, None] * $(1024)) + r24_lane[None, :]), dtype=tl.float32), axis=1)
    r24_value = r24_value.to(tl.float32)
    r24_value = tl.broadcast_to(r24_value, [$(96)])
    tl.store($(candidate_vsum_3_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(96))
    r25_value = tl.log(tl.load($(candidate_vsum_3_ptr) + r25_z, dtype=tl.float32))
    r25_value = r25_value.to(tl.float32)
    r25_value = tl.broadcast_to(r25_value, [$(96)])
    tl.store($(candidate_vlog_1_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(96))
    r26_value = tl.load($(candidate_vto_copy_2_ptr) + ((r26_z * $(1024)) + (tl.load($(parameter1_ptr) + (r26_z + $(96)), dtype=tl.int64) % $(1024))), dtype=tl.float32)
    r26_value = r26_value.to(tl.float32)
    r26_value = tl.broadcast_to(r26_value, [$(96)])
    tl.store($(candidate_vgather_1_ptr) + r26_z, r26_value)
    r27_z = tl.arange($(96))
    r27_value = (tl.load($(candidate_vgather_1_ptr) + r27_z, dtype=tl.float32) - tl.load($(candidate_vmax_2_value_ptr) + r27_z, dtype=tl.float32))
    r27_value = r27_value.to(tl.float32)
    r27_value = tl.broadcast_to(r27_value, [$(96)])
    tl.store($(candidate_vsub_4_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(96))
    r28_value = (tl.load($(candidate_vsub_4_ptr) + r28_z, dtype=tl.float32) - tl.load($(candidate_vlog_1_ptr) + r28_z, dtype=tl.float32))
    r28_value = r28_value.to(tl.float32)
    r28_value = tl.broadcast_to(r28_value, [$(96)])
    tl.store($(candidate_vsub_5_ptr) + r28_z, r28_value)
    r29_z = tl.arange($(96))
    r29_value = -tl.load($(candidate_vsub_5_ptr) + r29_z, dtype=tl.float32)
    r29_value = r29_value.to(tl.float32)
    r29_value = tl.broadcast_to(r29_value, [$(96)])
    tl.store($(candidate_vneg_1_ptr) + r29_z, r29_value)
    r30_z = tl.arange($(96))
    r30_value = (tl.load($(candidate_vneg_1_ptr) + r30_z, dtype=tl.float32) * tl.load($(candidate_vto_copy_ptr) + (r30_z + $(96)), dtype=tl.float32))
    r30_value = r30_value.to(tl.float32)
    r30_value = tl.broadcast_to(r30_value, [$(96)])
    tl.store($(candidate_vmul_2_ptr) + r30_z, r30_value)
    r31_z = tl.arange($(1))
    r31_lane = tl.arange($(96))
    r31_value = tl.sum(tl.load($(candidate_vmul_2_ptr) + r31_lane[None, :], dtype=tl.float32), axis=1)
    r31_value = r31_value.to(tl.float32)
    r31_value = tl.broadcast_to(r31_value, [$(1)])
    tl.store($(candidate_vsum_4_ptr) + r31_z, r31_value)
    r32_z = tl.arange($(1))
    r32_value = (tl.load($(candidate_vadd_ptr) + $(0), dtype=tl.float32) + tl.load($(candidate_vsum_4_ptr) + $(0), dtype=tl.float32))
    r32_value = r32_value.to(tl.float32)
    r32_value = tl.broadcast_to(r32_value, [$(1)])
    tl.store($(candidate_vadd_1_ptr) + r32_z, r32_value)
    r33_row = tl.arange($(96))
    r33_col = tl.arange($(1024))
    r33_lane = tl.arange($(64))
    r33_left = tl.load($(input0_ptr) + ((((((r33_row[:, None] * $(64)) + r33_lane[None, :] + $(12288)) / $(64)) % $(320)) * $(64)) + (((r33_row[:, None] * $(64)) + r33_lane[None, :] + $(12288)) % $(64))), dtype=tl.bfloat16)
    r33_right = tl.load($(candidate_vmul_ptr) + (((r33_col[None, :] % $(1024)) * $(64)) + (r33_lane[:, None] % $(64))), dtype=tl.bfloat16)
    r33_value = tl.dot(r33_left, r33_right)
    r33_value = r33_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_2_ptr) + ((r33_row[:, None] * $(1024)) + r33_col[None, :]), r33_value)
    r34_z = tl.arange($(98304))
    r34_value = tl.load($(candidate_vmm_2_ptr) + ((((r34_z / $(1024)) % $(96)) * $(1024)) + (r34_z % $(1024))), dtype=tl.bfloat16)
    r34_value = r34_value.to(tl.float32)
    r34_value = tl.broadcast_to(r34_value, [$(98304)])
    tl.store($(candidate_vto_copy_3_ptr) + r34_z, r34_value)
    r35_z = tl.arange($(96))
    r35_lane = tl.arange($(1024))
    r35_row_values = tl.load($(candidate_vto_copy_3_ptr) + ((r35_z[:, None] * $(1024)) + (r35_lane[None, :] % $(1024))), dtype=tl.float32)
    r35_value = tl.max(r35_row_values, axis=1)
    r35_value = r35_value.to(tl.float32)
    r35_value = tl.broadcast_to(r35_value, [$(96)])
    tl.store($(candidate_vmax_3_value_ptr) + r35_z, r35_value)
    r36_z = tl.arange($(96))
    r36_lane = tl.arange($(1024))
    r36_row_values = tl.load($(candidate_vto_copy_3_ptr) + ((r36_z[:, None] * $(1024)) + (r36_lane[None, :] % $(1024))), dtype=tl.float32)
    r36_value = tl.argmax(r36_row_values, axis=1)
    r36_value = tl.cast(r36_value, tl.int64)
    r36_value = tl.broadcast_to(r36_value, [$(96)])
    tl.store($(candidate_vmax_3_index_ptr) + r36_z, r36_value)
    r37_z = tl.arange($(98304))
    r37_value = (tl.load($(candidate_vto_copy_3_ptr) + ((((r37_z / $(1024)) % $(96)) * $(1024)) + (r37_z % $(1024))), dtype=tl.float32) - tl.load($(candidate_vmax_3_value_ptr) + ((r37_z / $(1024)) % $(96)), dtype=tl.float32))
    r37_value = r37_value.to(tl.float32)
    r37_value = tl.broadcast_to(r37_value, [$(98304)])
    tl.store($(candidate_vsub_6_ptr) + r37_z, r37_value)
    r38_z = tl.arange($(98304))
    r38_value = tl.exp(tl.load($(candidate_vsub_6_ptr) + ((((r38_z / $(1024)) % $(96)) * $(1024)) + (r38_z % $(1024))), dtype=tl.float32))
    r38_value = r38_value.to(tl.float32)
    r38_value = tl.broadcast_to(r38_value, [$(98304)])
    tl.store($(candidate_vexp_2_ptr) + r38_z, r38_value)
    r39_z = tl.arange($(96))
    r39_lane = tl.arange($(1024))
    r39_value = tl.sum(tl.load($(candidate_vexp_2_ptr) + ((r39_z[:, None] * $(1024)) + r39_lane[None, :]), dtype=tl.float32), axis=1)
    r39_value = r39_value.to(tl.float32)
    r39_value = tl.broadcast_to(r39_value, [$(96)])
    tl.store($(candidate_vsum_5_ptr) + r39_z, r39_value)
    r40_z = tl.arange($(96))
    r40_value = tl.log(tl.load($(candidate_vsum_5_ptr) + r40_z, dtype=tl.float32))
    r40_value = r40_value.to(tl.float32)
    r40_value = tl.broadcast_to(r40_value, [$(96)])
    tl.store($(candidate_vlog_2_ptr) + r40_z, r40_value)
    r41_z = tl.arange($(96))
    r41_value = tl.load($(candidate_vto_copy_3_ptr) + ((r41_z * $(1024)) + (tl.load($(parameter1_ptr) + (r41_z + $(192)), dtype=tl.int64) % $(1024))), dtype=tl.float32)
    r41_value = r41_value.to(tl.float32)
    r41_value = tl.broadcast_to(r41_value, [$(96)])
    tl.store($(candidate_vgather_2_ptr) + r41_z, r41_value)
    r42_z = tl.arange($(96))
    r42_value = (tl.load($(candidate_vgather_2_ptr) + r42_z, dtype=tl.float32) - tl.load($(candidate_vmax_3_value_ptr) + r42_z, dtype=tl.float32))
    r42_value = r42_value.to(tl.float32)
    r42_value = tl.broadcast_to(r42_value, [$(96)])
    tl.store($(candidate_vsub_7_ptr) + r42_z, r42_value)
    r43_z = tl.arange($(96))
    r43_value = (tl.load($(candidate_vsub_7_ptr) + r43_z, dtype=tl.float32) - tl.load($(candidate_vlog_2_ptr) + r43_z, dtype=tl.float32))
    r43_value = r43_value.to(tl.float32)
    r43_value = tl.broadcast_to(r43_value, [$(96)])
    tl.store($(candidate_vsub_8_ptr) + r43_z, r43_value)
    r44_z = tl.arange($(96))
    r44_value = -tl.load($(candidate_vsub_8_ptr) + r44_z, dtype=tl.float32)
    r44_value = r44_value.to(tl.float32)
    r44_value = tl.broadcast_to(r44_value, [$(96)])
    tl.store($(candidate_vneg_2_ptr) + r44_z, r44_value)
    r45_z = tl.arange($(96))
    r45_value = (tl.load($(candidate_vneg_2_ptr) + r45_z, dtype=tl.float32) * tl.load($(candidate_vto_copy_ptr) + (r45_z + $(192)), dtype=tl.float32))
    r45_value = r45_value.to(tl.float32)
    r45_value = tl.broadcast_to(r45_value, [$(96)])
    tl.store($(candidate_vmul_3_ptr) + r45_z, r45_value)
    r46_z = tl.arange($(1))
    r46_lane = tl.arange($(96))
    r46_value = tl.sum(tl.load($(candidate_vmul_3_ptr) + r46_lane[None, :], dtype=tl.float32), axis=1)
    r46_value = r46_value.to(tl.float32)
    r46_value = tl.broadcast_to(r46_value, [$(1)])
    tl.store($(candidate_vsum_6_ptr) + r46_z, r46_value)
    r47_z = tl.arange($(1))
    r47_value = (tl.load($(candidate_vadd_1_ptr) + $(0), dtype=tl.float32) + tl.load($(candidate_vsum_6_ptr) + $(0), dtype=tl.float32))
    r47_value = r47_value.to(tl.float32)
    r47_value = tl.broadcast_to(r47_value, [$(1)])
    tl.store($(candidate_vadd_2_ptr) + r47_z, r47_value)
    r48_row = tl.arange($(32))
    r48_col = tl.arange($(1024))
    r48_lane = tl.arange($(64))
    r48_left = tl.load($(input0_ptr) + ((((((r48_row[:, None] * $(64)) + r48_lane[None, :] + $(18432)) / $(64)) % $(320)) * $(64)) + (((r48_row[:, None] * $(64)) + r48_lane[None, :] + $(18432)) % $(64))), dtype=tl.bfloat16)
    r48_right = tl.load($(candidate_vmul_ptr) + (((r48_col[None, :] % $(1024)) * $(64)) + (r48_lane[:, None] % $(64))), dtype=tl.bfloat16)
    r48_value = tl.dot(r48_left, r48_right)
    r48_value = r48_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_3_ptr) + ((r48_row[:, None] * $(1024)) + r48_col[None, :]), r48_value)
    r49_z = tl.arange($(32768))
    r49_value = tl.load($(candidate_vmm_3_ptr) + ((((r49_z / $(1024)) % $(32)) * $(1024)) + (r49_z % $(1024))), dtype=tl.bfloat16)
    r49_value = r49_value.to(tl.float32)
    r49_value = tl.broadcast_to(r49_value, [$(32768)])
    tl.store($(candidate_vto_copy_4_ptr) + r49_z, r49_value)
    r50_z = tl.arange($(32))
    r50_lane = tl.arange($(1024))
    r50_row_values = tl.load($(candidate_vto_copy_4_ptr) + ((r50_z[:, None] * $(1024)) + (r50_lane[None, :] % $(1024))), dtype=tl.float32)
    r50_value = tl.max(r50_row_values, axis=1)
    r50_value = r50_value.to(tl.float32)
    r50_value = tl.broadcast_to(r50_value, [$(32)])
    tl.store($(candidate_vmax_4_value_ptr) + r50_z, r50_value)
    r51_z = tl.arange($(32))
    r51_lane = tl.arange($(1024))
    r51_row_values = tl.load($(candidate_vto_copy_4_ptr) + ((r51_z[:, None] * $(1024)) + (r51_lane[None, :] % $(1024))), dtype=tl.float32)
    r51_value = tl.argmax(r51_row_values, axis=1)
    r51_value = tl.cast(r51_value, tl.int64)
    r51_value = tl.broadcast_to(r51_value, [$(32)])
    tl.store($(candidate_vmax_4_index_ptr) + r51_z, r51_value)
    r52_z = tl.arange($(32768))
    r52_value = (tl.load($(candidate_vto_copy_4_ptr) + ((((r52_z / $(1024)) % $(32)) * $(1024)) + (r52_z % $(1024))), dtype=tl.float32) - tl.load($(candidate_vmax_4_value_ptr) + ((r52_z / $(1024)) % $(32)), dtype=tl.float32))
    r52_value = r52_value.to(tl.float32)
    r52_value = tl.broadcast_to(r52_value, [$(32768)])
    tl.store($(candidate_vsub_9_ptr) + r52_z, r52_value)
    r53_z = tl.arange($(32768))
    r53_value = tl.exp(tl.load($(candidate_vsub_9_ptr) + ((((r53_z / $(1024)) % $(32)) * $(1024)) + (r53_z % $(1024))), dtype=tl.float32))
    r53_value = r53_value.to(tl.float32)
    r53_value = tl.broadcast_to(r53_value, [$(32768)])
    tl.store($(candidate_vexp_3_ptr) + r53_z, r53_value)
    r54_z = tl.arange($(32))
    r54_lane = tl.arange($(1024))
    r54_value = tl.sum(tl.load($(candidate_vexp_3_ptr) + ((r54_z[:, None] * $(1024)) + r54_lane[None, :]), dtype=tl.float32), axis=1)
    r54_value = r54_value.to(tl.float32)
    r54_value = tl.broadcast_to(r54_value, [$(32)])
    tl.store($(candidate_vsum_7_ptr) + r54_z, r54_value)
    r55_z = tl.arange($(32))
    r55_value = tl.log(tl.load($(candidate_vsum_7_ptr) + r55_z, dtype=tl.float32))
    r55_value = r55_value.to(tl.float32)
    r55_value = tl.broadcast_to(r55_value, [$(32)])
    tl.store($(candidate_vlog_3_ptr) + r55_z, r55_value)
    r56_z = tl.arange($(32))
    r56_value = tl.load($(candidate_vto_copy_4_ptr) + ((r56_z * $(1024)) + (tl.load($(parameter1_ptr) + (r56_z + $(288)), dtype=tl.int64) % $(1024))), dtype=tl.float32)
    r56_value = r56_value.to(tl.float32)
    r56_value = tl.broadcast_to(r56_value, [$(32)])
    tl.store($(candidate_vgather_3_ptr) + r56_z, r56_value)
    r57_z = tl.arange($(32))
    r57_value = (tl.load($(candidate_vgather_3_ptr) + r57_z, dtype=tl.float32) - tl.load($(candidate_vmax_4_value_ptr) + r57_z, dtype=tl.float32))
    r57_value = r57_value.to(tl.float32)
    r57_value = tl.broadcast_to(r57_value, [$(32)])
    tl.store($(candidate_vsub_10_ptr) + r57_z, r57_value)
    r58_z = tl.arange($(32))
    r58_value = (tl.load($(candidate_vsub_10_ptr) + r58_z, dtype=tl.float32) - tl.load($(candidate_vlog_3_ptr) + r58_z, dtype=tl.float32))
    r58_value = r58_value.to(tl.float32)
    r58_value = tl.broadcast_to(r58_value, [$(32)])
    tl.store($(candidate_vsub_11_ptr) + r58_z, r58_value)
    r59_z = tl.arange($(32))
    r59_value = -tl.load($(candidate_vsub_11_ptr) + r59_z, dtype=tl.float32)
    r59_value = r59_value.to(tl.float32)
    r59_value = tl.broadcast_to(r59_value, [$(32)])
    tl.store($(candidate_vneg_3_ptr) + r59_z, r59_value)
    r60_z = tl.arange($(32))
    r60_value = (tl.load($(candidate_vneg_3_ptr) + r60_z, dtype=tl.float32) * tl.load($(candidate_vto_copy_ptr) + (r60_z + $(288)), dtype=tl.float32))
    r60_value = r60_value.to(tl.float32)
    r60_value = tl.broadcast_to(r60_value, [$(32)])
    tl.store($(candidate_vmul_4_ptr) + r60_z, r60_value)
    r61_z = tl.arange($(1))
    r61_lane = tl.arange($(32))
    r61_value = tl.sum(tl.load($(candidate_vmul_4_ptr) + r61_lane[None, :], dtype=tl.float32), axis=1)
    r61_value = r61_value.to(tl.float32)
    r61_value = tl.broadcast_to(r61_value, [$(1)])
    tl.store($(candidate_vsum_8_ptr) + r61_z, r61_value)
    r62_z = tl.arange($(1))
    r62_value = (tl.load($(candidate_vadd_2_ptr) + $(0), dtype=tl.float32) + tl.load($(candidate_vsum_8_ptr) + $(0), dtype=tl.float32))
    r62_value = r62_value.to(tl.float32)
    r62_value = tl.broadcast_to(r62_value, [$(1)])
    tl.store($(candidate_vadd_3_ptr) + r62_z, r62_value)
    z = tl.arange($(1))
    value = tl.load($(candidate_vadd_3_ptr) + $(0), dtype=tl.float32)
    value = tl.broadcast_to(value, [$(1)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_chunked_lm_loss_fwd
