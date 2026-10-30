import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_chunked_lm_loss_wgrad
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
  let input1_ptr : RegionName := "input1"
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
  let vto_copy_2_ptr : RegionName := "vto_copy_2"
  let vmul_2_ptr : RegionName := "vmul_2"
  let vneg_1_ptr : RegionName := "vneg_1"
  let vneg_2_ptr : RegionName := "vneg_2"
  let vsub_3_ptr : RegionName := "vsub_3"
  let vexp_1_ptr : RegionName := "vexp_1"
  let vdiv_ptr : RegionName := "vdiv"
  let vmul_3_ptr : RegionName := "vmul_3"
  let vneg_3_ptr : RegionName := "vneg_3"
  let vsum_4_ptr : RegionName := "vsum_4"
  let vadd_ptr : RegionName := "vadd"
  let vneg_4_ptr : RegionName := "vneg_4"
  let vscatter_add_ptr : RegionName := "vscatter_add"
  let vgather_1_ptr : RegionName := "vgather_1"
  let vadd_1_ptr : RegionName := "vadd_1"
  let vadd_2_ptr : RegionName := "vadd_2"
  let vscatter_ptr : RegionName := "vscatter"
  let vgather_2_ptr : RegionName := "vgather_2"
  let vadd_3_ptr : RegionName := "vadd_3"
  let vadd_4_ptr : RegionName := "vadd_4"
  let vscatter_1_ptr : RegionName := "vscatter_1"
  let vto_copy_3_ptr : RegionName := "vto_copy_3"
  let vmm_1_ptr : RegionName := "vmm_1"
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
    r10_z = tl.arange($(1))
    r10_value = tl.load($(input1_ptr) + $(0), dtype=tl.bfloat16)
    r10_value = r10_value.to(tl.float32)
    r10_value = tl.broadcast_to(r10_value, [$(1)])
    tl.store($(vto_copy_2_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(320))
    r11_value = (tl.load($(vto_copy_2_ptr) + $(0), dtype=tl.float32) * tl.load($(vto_copy_ptr) + r11_z, dtype=tl.float32))
    r11_value = r11_value.to(tl.float32)
    r11_value = tl.broadcast_to(r11_value, [$(320)])
    tl.store($(vmul_2_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(320))
    r12_value = -tl.load($(vmul_2_ptr) + r12_z, dtype=tl.float32)
    r12_value = r12_value.to(tl.float32)
    r12_value = tl.broadcast_to(r12_value, [$(320)])
    tl.store($(vneg_1_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(320))
    r13_value = -tl.load($(vneg_1_ptr) + r13_z, dtype=tl.float32)
    r13_value = r13_value.to(tl.float32)
    r13_value = tl.broadcast_to(r13_value, [$(320)])
    tl.store($(vneg_2_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(327680))
    r14_value = (tl.load($(vto_copy_1_ptr) + ((((r14_z / $(1024)) % $(320)) * $(1024)) + (r14_z % $(1024))), dtype=tl.float32) - tl.load($(vmax_1_value_ptr) + ((r14_z / $(1024)) % $(320)), dtype=tl.float32))
    r14_value = r14_value.to(tl.float32)
    r14_value = tl.broadcast_to(r14_value, [$(327680)])
    tl.store($(vsub_3_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(327680))
    r15_value = tl.exp(tl.load($(vsub_3_ptr) + ((((r15_z / $(1024)) % $(320)) * $(1024)) + (r15_z % $(1024))), dtype=tl.float32))
    r15_value = r15_value.to(tl.float32)
    r15_value = tl.broadcast_to(r15_value, [$(327680)])
    tl.store($(vexp_1_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(320))
    r16_value = (tl.load($(vneg_2_ptr) + r16_z, dtype=tl.float32) / tl.load($(vsum_1_ptr) + r16_z, dtype=tl.float32))
    r16_value = r16_value.to(tl.float32)
    r16_value = tl.broadcast_to(r16_value, [$(320)])
    tl.store($(vdiv_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(327680))
    r17_value = (tl.load($(vdiv_ptr) + ((r17_z / $(1024)) % $(320)), dtype=tl.float32) * tl.load($(vexp_1_ptr) + ((((r17_z / $(1024)) % $(320)) * $(1024)) + (r17_z % $(1024))), dtype=tl.float32))
    r17_value = r17_value.to(tl.float32)
    r17_value = tl.broadcast_to(r17_value, [$(327680)])
    tl.store($(vmul_3_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(327680))
    r18_value = -tl.load($(vmul_3_ptr) + ((((r18_z / $(1024)) % $(320)) * $(1024)) + (r18_z % $(1024))), dtype=tl.float32)
    r18_value = r18_value.to(tl.float32)
    r18_value = tl.broadcast_to(r18_value, [$(327680)])
    tl.store($(vneg_3_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(320))
    r19_lane = tl.arange($(1024))
    r19_value = tl.sum(tl.load($(vneg_3_ptr) + ((r19_z[:, None] * $(1024)) + r19_lane[None, :]), dtype=tl.float32), axis=1)
    r19_value = r19_value.to(tl.float32)
    r19_value = tl.broadcast_to(r19_value, [$(320)])
    tl.store($(vsum_4_ptr) + r19_z, r19_value)
    r20_z = tl.arange($(320))
    r20_value = (tl.load($(vneg_2_ptr) + r20_z, dtype=tl.float32) + tl.load($(vsum_4_ptr) + r20_z, dtype=tl.float32))
    r20_value = r20_value.to(tl.float32)
    r20_value = tl.broadcast_to(r20_value, [$(320)])
    tl.store($(vadd_ptr) + r20_z, r20_value)
    r21_z = tl.arange($(327680))
    r21_value = -tl.load($(vneg_3_ptr) + ((((r21_z / $(1024)) % $(320)) * $(1024)) + (r21_z % $(1024))), dtype=tl.float32)
    r21_value = r21_value.to(tl.float32)
    r21_value = tl.broadcast_to(r21_value, [$(327680)])
    tl.store($(vneg_4_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(320))
    r22_value = tl.broadcast_to(0.0, [$(320)])
    for r22_lane in range($(1)) {
      r22_value = tl.where(tl.full([], $(0), dtype=tl.int64) == tl.cast($(0), tl.int64), (r22_value + tl.load($(vneg_1_ptr) + ((r22_z + r22_lane) % $(320)), dtype=tl.float32)), r22_value)
    }
    r22_value = r22_value.to(tl.float32)
    r22_value = tl.broadcast_to(r22_value, [$(320)])
    tl.store($(vscatter_add_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(320))
    r23_value = tl.load($(vneg_4_ptr) + ((r23_z * $(1024)) + (tl.load($(parameter1_ptr) + r23_z, dtype=tl.int64) % $(1024))), dtype=tl.float32)
    r23_value = r23_value.to(tl.float32)
    r23_value = tl.broadcast_to(r23_value, [$(320)])
    tl.store($(vgather_1_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(320))
    r24_value = (tl.load($(vscatter_add_ptr) + r24_z, dtype=tl.float32) + tl.load($(vgather_1_ptr) + r24_z, dtype=tl.float32))
    r24_value = r24_value.to(tl.float32)
    r24_value = tl.broadcast_to(r24_value, [$(320)])
    tl.store($(vadd_1_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(327680))
    r25_value = (tl.load($(vneg_4_ptr) + ((((r25_z / $(1024)) % $(320)) * $(1024)) + (r25_z % $(1024))), dtype=tl.float32) + 0.0)
    r25_value = r25_value.to(tl.float32)
    r25_value = tl.broadcast_to(r25_value, [$(327680)])
    tl.store($(vadd_2_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(327680))
    r26_value = tl.broadcast_to(tl.load($(vadd_2_ptr) + ((((r26_z / $(1024)) % $(320)) * $(1024)) + (r26_z % $(1024))), dtype=tl.float32), [$(327680)])
    for r26_lane in range($(1)) {
      r26_value = tl.where(tl.load($(parameter1_ptr) + ((((r26_z / $(1024)) % $(320)) + r26_lane) % $(320)), dtype=tl.int64) == tl.cast((r26_z % $(1024)), tl.int64), tl.load($(vadd_1_ptr) + ((r26_z / $(1024)) % $(320)), dtype=tl.float32), r26_value)
    }
    r26_value = r26_value.to(tl.float32)
    r26_value = tl.broadcast_to(r26_value, [$(327680)])
    tl.store($(vscatter_ptr) + r26_z, r26_value)
    r27_z = tl.arange($(320))
    r27_value = tl.load($(vscatter_ptr) + ((r27_z * $(1024)) + (tl.load($(vmax_1_index_ptr) + r27_z, dtype=tl.int64) % $(1024))), dtype=tl.float32)
    r27_value = r27_value.to(tl.float32)
    r27_value = tl.broadcast_to(r27_value, [$(320)])
    tl.store($(vgather_2_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(320))
    r28_value = (tl.load($(vgather_2_ptr) + r28_z, dtype=tl.float32) + tl.load($(vadd_ptr) + r28_z, dtype=tl.float32))
    r28_value = r28_value.to(tl.float32)
    r28_value = tl.broadcast_to(r28_value, [$(320)])
    tl.store($(vadd_3_ptr) + r28_z, r28_value)
    r29_z = tl.arange($(327680))
    r29_value = (tl.load($(vscatter_ptr) + ((((r29_z / $(1024)) % $(320)) * $(1024)) + (r29_z % $(1024))), dtype=tl.float32) + 0.0)
    r29_value = r29_value.to(tl.float32)
    r29_value = tl.broadcast_to(r29_value, [$(327680)])
    tl.store($(vadd_4_ptr) + r29_z, r29_value)
    r30_z = tl.arange($(327680))
    r30_value = tl.broadcast_to(tl.load($(vadd_4_ptr) + ((((r30_z / $(1024)) % $(320)) * $(1024)) + (r30_z % $(1024))), dtype=tl.float32), [$(327680)])
    for r30_lane in range($(1)) {
      r30_value = tl.where(tl.load($(vmax_1_index_ptr) + ((r30_z / $(1024)) % $(320)), dtype=tl.int64) == tl.cast((r30_z % $(1024)), tl.int64), tl.load($(vadd_3_ptr) + ((r30_z / $(1024)) % $(320)), dtype=tl.float32), r30_value)
    }
    r30_value = r30_value.to(tl.float32)
    r30_value = tl.broadcast_to(r30_value, [$(327680)])
    tl.store($(vscatter_1_ptr) + r30_z, r30_value)
    r31_z = tl.arange($(327680))
    r31_value = tl.load($(vscatter_1_ptr) + ((((r31_z / $(1024)) % $(320)) * $(1024)) + (r31_z % $(1024))), dtype=tl.float32)
    r31_value = r31_value.to(tl.bfloat16)
    r31_value = tl.broadcast_to(r31_value, [$(327680)])
    tl.store($(vto_copy_3_ptr) + r31_z, r31_value)
    r32_row = tl.arange($(1024))
    r32_col = tl.arange($(64))
    r32_lane = tl.arange($(320))
    r32_left = tl.load($(vto_copy_3_ptr) + (((r32_lane[None, :] % $(320)) * $(1024)) + (r32_row[:, None] % $(1024))), dtype=tl.bfloat16)
    r32_right = tl.load($(input0_ptr) + (((r32_lane[:, None] % $(320)) * $(64)) + (r32_col[None, :] % $(64))), dtype=tl.bfloat16)
    r32_value = tl.dot(r32_left, r32_right)
    r32_value = r32_value.to(tl.bfloat16)
    tl.store($(vmm_1_ptr) + ((r32_row[:, None] * $(64)) + r32_col[None, :]), r32_value)
    z = tl.arange($(65536))
    value = tl.load($(vmm_1_ptr) + ((((z / $(64)) % $(1024)) * $(64)) + (z % $(64))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(65536)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let parameter0_ptr : RegionName := "parameter0"
  let parameter1_ptr : RegionName := "parameter1"
  let parameter2_ptr : RegionName := "parameter2"
  let input0_ptr : RegionName := "input0"
  let input1_ptr : RegionName := "input1"
  let out_ptr : RegionName := "out"
  let vmul_ptr : RegionName := "vmul"
  let vgt_ptr : RegionName := "vgt"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vdiv_ptr : RegionName := "vdiv"
  let vneg_4_ptr : RegionName := "vneg_4"
  let vscatter_add_ptr : RegionName := "vscatter_add"
  let vscatter_ptr : RegionName := "vscatter"
  let vadd_4_ptr : RegionName := "vadd_4"
  let vscatter_1_ptr : RegionName := "vscatter_1"
  let candidate_vmul_ptr : RegionName := "candidate_vmul"
  let candidate_vgt_ptr : RegionName := "candidate_vgt"
  let candidate_vto_copy_ptr : RegionName := "candidate_vto_copy"
  let candidate_vto_copy_5_ptr : RegionName := "candidate_vto_copy_5"
  let candidate_vmm_4_ptr : RegionName := "candidate_vmm_4"
  let candidate_vto_copy_6_ptr : RegionName := "candidate_vto_copy_6"
  let candidate_vmax_5_value_ptr : RegionName := "candidate_vmax_5_value"
  let candidate_vmax_5_index_ptr : RegionName := "candidate_vmax_5_index"
  let candidate_vsub_12_ptr : RegionName := "candidate_vsub_12"
  let candidate_vexp_4_ptr : RegionName := "candidate_vexp_4"
  let candidate_vsum_10_ptr : RegionName := "candidate_vsum_10"
  let candidate_vlog_4_ptr : RegionName := "candidate_vlog_4"
  let candidate_vgather_4_ptr : RegionName := "candidate_vgather_4"
  let candidate_vsub_13_ptr : RegionName := "candidate_vsub_13"
  let candidate_vsub_14_ptr : RegionName := "candidate_vsub_14"
  let candidate_vneg_4_ptr : RegionName := "candidate_vneg_4"
  let candidate_vmul_5_ptr : RegionName := "candidate_vmul_5"
  let candidate_vsum_11_ptr : RegionName := "candidate_vsum_11"
  let candidate_vmul_6_ptr : RegionName := "candidate_vmul_6"
  let candidate_vmul_7_ptr : RegionName := "candidate_vmul_7"
  let candidate_vmul_8_ptr : RegionName := "candidate_vmul_8"
  let candidate_vneg_5_ptr : RegionName := "candidate_vneg_5"
  let candidate_vneg_6_ptr : RegionName := "candidate_vneg_6"
  let candidate_vsub_15_ptr : RegionName := "candidate_vsub_15"
  let candidate_vexp_5_ptr : RegionName := "candidate_vexp_5"
  let candidate_vdiv_ptr : RegionName := "candidate_vdiv"
  let candidate_vmul_9_ptr : RegionName := "candidate_vmul_9"
  let candidate_vneg_7_ptr : RegionName := "candidate_vneg_7"
  let candidate_vsum_12_ptr : RegionName := "candidate_vsum_12"
  let candidate_vadd_4_ptr : RegionName := "candidate_vadd_4"
  let candidate_vneg_8_ptr : RegionName := "candidate_vneg_8"
  let candidate_vscatter_add_ptr : RegionName := "candidate_vscatter_add"
  let candidate_vgather_5_ptr : RegionName := "candidate_vgather_5"
  let candidate_vadd_5_ptr : RegionName := "candidate_vadd_5"
  let candidate_vadd_6_ptr : RegionName := "candidate_vadd_6"
  let candidate_vscatter_ptr : RegionName := "candidate_vscatter"
  let candidate_vgather_6_ptr : RegionName := "candidate_vgather_6"
  let candidate_vadd_7_ptr : RegionName := "candidate_vadd_7"
  let candidate_vadd_8_ptr : RegionName := "candidate_vadd_8"
  let candidate_vscatter_1_ptr : RegionName := "candidate_vscatter_1"
  let candidate_vto_copy_7_ptr : RegionName := "candidate_vto_copy_7"
  let candidate_vto_copy_8_ptr : RegionName := "candidate_vto_copy_8"
  let candidate_vto_copy_9_ptr : RegionName := "candidate_vto_copy_9"
  let candidate_vmm_6_ptr : RegionName := "candidate_vmm_6"
  let candidate_vadd_9_ptr : RegionName := "candidate_vadd_9"
  let candidate_vmm_7_ptr : RegionName := "candidate_vmm_7"
  let candidate_vto_copy_10_ptr : RegionName := "candidate_vto_copy_10"
  let candidate_vmax_6_value_ptr : RegionName := "candidate_vmax_6_value"
  let candidate_vmax_6_index_ptr : RegionName := "candidate_vmax_6_index"
  let candidate_vsub_16_ptr : RegionName := "candidate_vsub_16"
  let candidate_vexp_6_ptr : RegionName := "candidate_vexp_6"
  let candidate_vsum_13_ptr : RegionName := "candidate_vsum_13"
  let candidate_vlog_5_ptr : RegionName := "candidate_vlog_5"
  let candidate_vgather_7_ptr : RegionName := "candidate_vgather_7"
  let candidate_vsub_17_ptr : RegionName := "candidate_vsub_17"
  let candidate_vsub_18_ptr : RegionName := "candidate_vsub_18"
  let candidate_vneg_9_ptr : RegionName := "candidate_vneg_9"
  let candidate_vmul_10_ptr : RegionName := "candidate_vmul_10"
  let candidate_vsum_14_ptr : RegionName := "candidate_vsum_14"
  let candidate_vmul_11_ptr : RegionName := "candidate_vmul_11"
  let candidate_vmul_12_ptr : RegionName := "candidate_vmul_12"
  let candidate_vmul_13_ptr : RegionName := "candidate_vmul_13"
  let candidate_vneg_10_ptr : RegionName := "candidate_vneg_10"
  let candidate_vneg_11_ptr : RegionName := "candidate_vneg_11"
  let candidate_vsub_19_ptr : RegionName := "candidate_vsub_19"
  let candidate_vexp_7_ptr : RegionName := "candidate_vexp_7"
  let candidate_vdiv_1_ptr : RegionName := "candidate_vdiv_1"
  let candidate_vmul_14_ptr : RegionName := "candidate_vmul_14"
  let candidate_vneg_12_ptr : RegionName := "candidate_vneg_12"
  let candidate_vsum_15_ptr : RegionName := "candidate_vsum_15"
  let candidate_vadd_10_ptr : RegionName := "candidate_vadd_10"
  let candidate_vneg_13_ptr : RegionName := "candidate_vneg_13"
  let candidate_vscatter_add_1_ptr : RegionName := "candidate_vscatter_add_1"
  let candidate_vgather_8_ptr : RegionName := "candidate_vgather_8"
  let candidate_vadd_11_ptr : RegionName := "candidate_vadd_11"
  let candidate_vadd_12_ptr : RegionName := "candidate_vadd_12"
  let candidate_vscatter_2_ptr : RegionName := "candidate_vscatter_2"
  let candidate_vgather_9_ptr : RegionName := "candidate_vgather_9"
  let candidate_vadd_13_ptr : RegionName := "candidate_vadd_13"
  let candidate_vadd_14_ptr : RegionName := "candidate_vadd_14"
  let candidate_vscatter_3_ptr : RegionName := "candidate_vscatter_3"
  let candidate_vto_copy_11_ptr : RegionName := "candidate_vto_copy_11"
  let candidate_vto_copy_12_ptr : RegionName := "candidate_vto_copy_12"
  let candidate_vto_copy_13_ptr : RegionName := "candidate_vto_copy_13"
  let candidate_vmm_9_ptr : RegionName := "candidate_vmm_9"
  let candidate_vadd_15_ptr : RegionName := "candidate_vadd_15"
  let candidate_vmm_10_ptr : RegionName := "candidate_vmm_10"
  let candidate_vto_copy_14_ptr : RegionName := "candidate_vto_copy_14"
  let candidate_vmax_7_value_ptr : RegionName := "candidate_vmax_7_value"
  let candidate_vmax_7_index_ptr : RegionName := "candidate_vmax_7_index"
  let candidate_vsub_20_ptr : RegionName := "candidate_vsub_20"
  let candidate_vexp_8_ptr : RegionName := "candidate_vexp_8"
  let candidate_vsum_16_ptr : RegionName := "candidate_vsum_16"
  let candidate_vlog_6_ptr : RegionName := "candidate_vlog_6"
  let candidate_vgather_10_ptr : RegionName := "candidate_vgather_10"
  let candidate_vsub_21_ptr : RegionName := "candidate_vsub_21"
  let candidate_vsub_22_ptr : RegionName := "candidate_vsub_22"
  let candidate_vneg_14_ptr : RegionName := "candidate_vneg_14"
  let candidate_vmul_15_ptr : RegionName := "candidate_vmul_15"
  let candidate_vsum_17_ptr : RegionName := "candidate_vsum_17"
  let candidate_vmul_16_ptr : RegionName := "candidate_vmul_16"
  let candidate_vmul_17_ptr : RegionName := "candidate_vmul_17"
  let candidate_vmul_18_ptr : RegionName := "candidate_vmul_18"
  let candidate_vneg_15_ptr : RegionName := "candidate_vneg_15"
  let candidate_vneg_16_ptr : RegionName := "candidate_vneg_16"
  let candidate_vsub_23_ptr : RegionName := "candidate_vsub_23"
  let candidate_vexp_9_ptr : RegionName := "candidate_vexp_9"
  let candidate_vdiv_2_ptr : RegionName := "candidate_vdiv_2"
  let candidate_vmul_19_ptr : RegionName := "candidate_vmul_19"
  let candidate_vneg_17_ptr : RegionName := "candidate_vneg_17"
  let candidate_vsum_18_ptr : RegionName := "candidate_vsum_18"
  let candidate_vadd_16_ptr : RegionName := "candidate_vadd_16"
  let candidate_vneg_18_ptr : RegionName := "candidate_vneg_18"
  let candidate_vscatter_add_2_ptr : RegionName := "candidate_vscatter_add_2"
  let candidate_vgather_11_ptr : RegionName := "candidate_vgather_11"
  let candidate_vadd_17_ptr : RegionName := "candidate_vadd_17"
  let candidate_vadd_18_ptr : RegionName := "candidate_vadd_18"
  let candidate_vscatter_4_ptr : RegionName := "candidate_vscatter_4"
  let candidate_vgather_12_ptr : RegionName := "candidate_vgather_12"
  let candidate_vadd_19_ptr : RegionName := "candidate_vadd_19"
  let candidate_vadd_20_ptr : RegionName := "candidate_vadd_20"
  let candidate_vscatter_5_ptr : RegionName := "candidate_vscatter_5"
  let candidate_vto_copy_15_ptr : RegionName := "candidate_vto_copy_15"
  let candidate_vto_copy_16_ptr : RegionName := "candidate_vto_copy_16"
  let candidate_vto_copy_17_ptr : RegionName := "candidate_vto_copy_17"
  let candidate_vmm_12_ptr : RegionName := "candidate_vmm_12"
  let candidate_vadd_21_ptr : RegionName := "candidate_vadd_21"
  let candidate_vmm_13_ptr : RegionName := "candidate_vmm_13"
  let candidate_vto_copy_18_ptr : RegionName := "candidate_vto_copy_18"
  let candidate_vmax_8_value_ptr : RegionName := "candidate_vmax_8_value"
  let candidate_vmax_8_index_ptr : RegionName := "candidate_vmax_8_index"
  let candidate_vsub_24_ptr : RegionName := "candidate_vsub_24"
  let candidate_vexp_10_ptr : RegionName := "candidate_vexp_10"
  let candidate_vsum_19_ptr : RegionName := "candidate_vsum_19"
  let candidate_vlog_7_ptr : RegionName := "candidate_vlog_7"
  let candidate_vgather_13_ptr : RegionName := "candidate_vgather_13"
  let candidate_vsub_25_ptr : RegionName := "candidate_vsub_25"
  let candidate_vsub_26_ptr : RegionName := "candidate_vsub_26"
  let candidate_vneg_19_ptr : RegionName := "candidate_vneg_19"
  let candidate_vmul_20_ptr : RegionName := "candidate_vmul_20"
  let candidate_vsum_20_ptr : RegionName := "candidate_vsum_20"
  let candidate_vmul_21_ptr : RegionName := "candidate_vmul_21"
  let candidate_vmul_22_ptr : RegionName := "candidate_vmul_22"
  let candidate_vmul_23_ptr : RegionName := "candidate_vmul_23"
  let candidate_vneg_20_ptr : RegionName := "candidate_vneg_20"
  let candidate_vneg_21_ptr : RegionName := "candidate_vneg_21"
  let candidate_vsub_27_ptr : RegionName := "candidate_vsub_27"
  let candidate_vexp_11_ptr : RegionName := "candidate_vexp_11"
  let candidate_vdiv_3_ptr : RegionName := "candidate_vdiv_3"
  let candidate_vmul_24_ptr : RegionName := "candidate_vmul_24"
  let candidate_vneg_22_ptr : RegionName := "candidate_vneg_22"
  let candidate_vsum_21_ptr : RegionName := "candidate_vsum_21"
  let candidate_vadd_22_ptr : RegionName := "candidate_vadd_22"
  let candidate_vneg_23_ptr : RegionName := "candidate_vneg_23"
  let candidate_vscatter_add_3_ptr : RegionName := "candidate_vscatter_add_3"
  let candidate_vgather_14_ptr : RegionName := "candidate_vgather_14"
  let candidate_vadd_23_ptr : RegionName := "candidate_vadd_23"
  let candidate_vadd_24_ptr : RegionName := "candidate_vadd_24"
  let candidate_vscatter_6_ptr : RegionName := "candidate_vscatter_6"
  let candidate_vgather_15_ptr : RegionName := "candidate_vgather_15"
  let candidate_vadd_25_ptr : RegionName := "candidate_vadd_25"
  let candidate_vadd_26_ptr : RegionName := "candidate_vadd_26"
  let candidate_vscatter_7_ptr : RegionName := "candidate_vscatter_7"
  let candidate_vto_copy_19_ptr : RegionName := "candidate_vto_copy_19"
  let candidate_vto_copy_20_ptr : RegionName := "candidate_vto_copy_20"
  let candidate_vto_copy_21_ptr : RegionName := "candidate_vto_copy_21"
  let candidate_vmm_15_ptr : RegionName := "candidate_vmm_15"
  let candidate_vadd_27_ptr : RegionName := "candidate_vadd_27"
  let candidate_vto_copy_22_ptr : RegionName := "candidate_vto_copy_22"
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
    r3_z = tl.arange($(1))
    r3_value = tl.load($(input1_ptr) + $(0), dtype=tl.bfloat16)
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(1)])
    tl.store($(candidate_vto_copy_5_ptr) + r3_z, r3_value)
    r4_row = tl.arange($(96))
    r4_col = tl.arange($(1024))
    r4_lane = tl.arange($(64))
    r4_left = tl.load($(input0_ptr) + ((((((r4_row[:, None] * $(64)) + r4_lane[None, :]) / $(64)) % $(320)) * $(64)) + (((r4_row[:, None] * $(64)) + r4_lane[None, :]) % $(64))), dtype=tl.bfloat16)
    r4_right = tl.load($(candidate_vmul_ptr) + (((r4_col[None, :] % $(1024)) * $(64)) + (r4_lane[:, None] % $(64))), dtype=tl.bfloat16)
    r4_value = tl.dot(r4_left, r4_right)
    r4_value = r4_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_4_ptr) + ((r4_row[:, None] * $(1024)) + r4_col[None, :]), r4_value)
    r5_z = tl.arange($(98304))
    r5_value = tl.load($(candidate_vmm_4_ptr) + ((((r5_z / $(1024)) % $(96)) * $(1024)) + (r5_z % $(1024))), dtype=tl.bfloat16)
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(98304)])
    tl.store($(candidate_vto_copy_6_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(96))
    r6_lane = tl.arange($(1024))
    r6_row_values = tl.load($(candidate_vto_copy_6_ptr) + ((r6_z[:, None] * $(1024)) + (r6_lane[None, :] % $(1024))), dtype=tl.float32)
    r6_value = tl.max(r6_row_values, axis=1)
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(96)])
    tl.store($(candidate_vmax_5_value_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(96))
    r7_lane = tl.arange($(1024))
    r7_row_values = tl.load($(candidate_vto_copy_6_ptr) + ((r7_z[:, None] * $(1024)) + (r7_lane[None, :] % $(1024))), dtype=tl.float32)
    r7_value = tl.argmax(r7_row_values, axis=1)
    r7_value = tl.cast(r7_value, tl.int64)
    r7_value = tl.broadcast_to(r7_value, [$(96)])
    tl.store($(candidate_vmax_5_index_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(98304))
    r8_value = (tl.load($(candidate_vto_copy_6_ptr) + ((((r8_z / $(1024)) % $(96)) * $(1024)) + (r8_z % $(1024))), dtype=tl.float32) - tl.load($(candidate_vmax_5_value_ptr) + ((r8_z / $(1024)) % $(96)), dtype=tl.float32))
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(98304)])
    tl.store($(candidate_vsub_12_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(98304))
    r9_value = tl.exp(tl.load($(candidate_vsub_12_ptr) + ((((r9_z / $(1024)) % $(96)) * $(1024)) + (r9_z % $(1024))), dtype=tl.float32))
    r9_value = r9_value.to(tl.float32)
    r9_value = tl.broadcast_to(r9_value, [$(98304)])
    tl.store($(candidate_vexp_4_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(96))
    r10_lane = tl.arange($(1024))
    r10_value = tl.sum(tl.load($(candidate_vexp_4_ptr) + ((r10_z[:, None] * $(1024)) + r10_lane[None, :]), dtype=tl.float32), axis=1)
    r10_value = r10_value.to(tl.float32)
    r10_value = tl.broadcast_to(r10_value, [$(96)])
    tl.store($(candidate_vsum_10_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(96))
    r11_value = tl.log(tl.load($(candidate_vsum_10_ptr) + r11_z, dtype=tl.float32))
    r11_value = r11_value.to(tl.float32)
    r11_value = tl.broadcast_to(r11_value, [$(96)])
    tl.store($(candidate_vlog_4_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(96))
    r12_value = tl.load($(candidate_vto_copy_6_ptr) + ((r12_z * $(1024)) + (tl.load($(parameter1_ptr) + r12_z, dtype=tl.int64) % $(1024))), dtype=tl.float32)
    r12_value = r12_value.to(tl.float32)
    r12_value = tl.broadcast_to(r12_value, [$(96)])
    tl.store($(candidate_vgather_4_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(96))
    r13_value = (tl.load($(candidate_vgather_4_ptr) + r13_z, dtype=tl.float32) - tl.load($(candidate_vmax_5_value_ptr) + r13_z, dtype=tl.float32))
    r13_value = r13_value.to(tl.float32)
    r13_value = tl.broadcast_to(r13_value, [$(96)])
    tl.store($(candidate_vsub_13_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(96))
    r14_value = (tl.load($(candidate_vsub_13_ptr) + r14_z, dtype=tl.float32) - tl.load($(candidate_vlog_4_ptr) + r14_z, dtype=tl.float32))
    r14_value = r14_value.to(tl.float32)
    r14_value = tl.broadcast_to(r14_value, [$(96)])
    tl.store($(candidate_vsub_14_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(96))
    r15_value = -tl.load($(candidate_vsub_14_ptr) + r15_z, dtype=tl.float32)
    r15_value = r15_value.to(tl.float32)
    r15_value = tl.broadcast_to(r15_value, [$(96)])
    tl.store($(candidate_vneg_4_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(96))
    r16_value = (tl.load($(candidate_vneg_4_ptr) + r16_z, dtype=tl.float32) * tl.load($(candidate_vto_copy_ptr) + r16_z, dtype=tl.float32))
    r16_value = r16_value.to(tl.float32)
    r16_value = tl.broadcast_to(r16_value, [$(96)])
    tl.store($(candidate_vmul_5_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(1))
    r17_lane = tl.arange($(96))
    r17_value = tl.sum(tl.load($(candidate_vmul_5_ptr) + r17_lane[None, :], dtype=tl.float32), axis=1)
    r17_value = r17_value.to(tl.float32)
    r17_value = tl.broadcast_to(r17_value, [$(1)])
    tl.store($(candidate_vsum_11_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(1))
    r18_value = (tl.load($(candidate_vsum_11_ptr) + $(0), dtype=tl.float32) * tl.load($(candidate_vto_copy_5_ptr) + $(0), dtype=tl.float32))
    r18_value = r18_value.to(tl.float32)
    r18_value = tl.broadcast_to(r18_value, [$(1)])
    tl.store($(candidate_vmul_6_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(1))
    r19_value = (1.0 * tl.load($(candidate_vto_copy_5_ptr) + $(0), dtype=tl.float32))
    r19_value = r19_value.to(tl.float32)
    r19_value = tl.broadcast_to(r19_value, [$(1)])
    tl.store($(candidate_vmul_7_ptr) + r19_z, r19_value)
    r20_z = tl.arange($(96))
    r20_value = (tl.load($(candidate_vmul_7_ptr) + $(0), dtype=tl.float32) * tl.load($(candidate_vto_copy_ptr) + r20_z, dtype=tl.float32))
    r20_value = r20_value.to(tl.float32)
    r20_value = tl.broadcast_to(r20_value, [$(96)])
    tl.store($(candidate_vmul_8_ptr) + r20_z, r20_value)
    r21_z = tl.arange($(96))
    r21_value = -tl.load($(candidate_vmul_8_ptr) + r21_z, dtype=tl.float32)
    r21_value = r21_value.to(tl.float32)
    r21_value = tl.broadcast_to(r21_value, [$(96)])
    tl.store($(candidate_vneg_5_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(96))
    r22_value = -tl.load($(candidate_vneg_5_ptr) + r22_z, dtype=tl.float32)
    r22_value = r22_value.to(tl.float32)
    r22_value = tl.broadcast_to(r22_value, [$(96)])
    tl.store($(candidate_vneg_6_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(98304))
    r23_value = (tl.load($(candidate_vto_copy_6_ptr) + ((((r23_z / $(1024)) % $(96)) * $(1024)) + (r23_z % $(1024))), dtype=tl.float32) - tl.load($(candidate_vmax_5_value_ptr) + ((r23_z / $(1024)) % $(96)), dtype=tl.float32))
    r23_value = r23_value.to(tl.float32)
    r23_value = tl.broadcast_to(r23_value, [$(98304)])
    tl.store($(candidate_vsub_15_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(98304))
    r24_value = tl.exp(tl.load($(candidate_vsub_15_ptr) + ((((r24_z / $(1024)) % $(96)) * $(1024)) + (r24_z % $(1024))), dtype=tl.float32))
    r24_value = r24_value.to(tl.float32)
    r24_value = tl.broadcast_to(r24_value, [$(98304)])
    tl.store($(candidate_vexp_5_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(96))
    r25_value = (tl.load($(candidate_vneg_6_ptr) + r25_z, dtype=tl.float32) / tl.load($(candidate_vsum_10_ptr) + r25_z, dtype=tl.float32))
    r25_value = r25_value.to(tl.float32)
    r25_value = tl.broadcast_to(r25_value, [$(96)])
    tl.store($(candidate_vdiv_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(98304))
    r26_value = (tl.load($(candidate_vdiv_ptr) + ((r26_z / $(1024)) % $(96)), dtype=tl.float32) * tl.load($(candidate_vexp_5_ptr) + ((((r26_z / $(1024)) % $(96)) * $(1024)) + (r26_z % $(1024))), dtype=tl.float32))
    r26_value = r26_value.to(tl.float32)
    r26_value = tl.broadcast_to(r26_value, [$(98304)])
    tl.store($(candidate_vmul_9_ptr) + r26_z, r26_value)
    r27_z = tl.arange($(98304))
    r27_value = -tl.load($(candidate_vmul_9_ptr) + ((((r27_z / $(1024)) % $(96)) * $(1024)) + (r27_z % $(1024))), dtype=tl.float32)
    r27_value = r27_value.to(tl.float32)
    r27_value = tl.broadcast_to(r27_value, [$(98304)])
    tl.store($(candidate_vneg_7_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(96))
    r28_lane = tl.arange($(1024))
    r28_value = tl.sum(tl.load($(candidate_vneg_7_ptr) + ((r28_z[:, None] * $(1024)) + r28_lane[None, :]), dtype=tl.float32), axis=1)
    r28_value = r28_value.to(tl.float32)
    r28_value = tl.broadcast_to(r28_value, [$(96)])
    tl.store($(candidate_vsum_12_ptr) + r28_z, r28_value)
    r29_z = tl.arange($(96))
    r29_value = (tl.load($(candidate_vneg_6_ptr) + r29_z, dtype=tl.float32) + tl.load($(candidate_vsum_12_ptr) + r29_z, dtype=tl.float32))
    r29_value = r29_value.to(tl.float32)
    r29_value = tl.broadcast_to(r29_value, [$(96)])
    tl.store($(candidate_vadd_4_ptr) + r29_z, r29_value)
    r30_z = tl.arange($(98304))
    r30_value = -tl.load($(candidate_vneg_7_ptr) + ((((r30_z / $(1024)) % $(96)) * $(1024)) + (r30_z % $(1024))), dtype=tl.float32)
    r30_value = r30_value.to(tl.float32)
    r30_value = tl.broadcast_to(r30_value, [$(98304)])
    tl.store($(candidate_vneg_8_ptr) + r30_z, r30_value)
    r31_z = tl.arange($(96))
    r31_value = tl.broadcast_to(0.0, [$(96)])
    for r31_lane in range($(1)) {
      r31_value = tl.where(tl.full([], $(0), dtype=tl.int64) == tl.cast($(0), tl.int64), (r31_value + tl.load($(candidate_vneg_5_ptr) + ((r31_z + r31_lane) % $(96)), dtype=tl.float32)), r31_value)
    }
    r31_value = r31_value.to(tl.float32)
    r31_value = tl.broadcast_to(r31_value, [$(96)])
    tl.store($(candidate_vscatter_add_ptr) + r31_z, r31_value)
    r32_z = tl.arange($(96))
    r32_value = tl.load($(candidate_vneg_8_ptr) + ((r32_z * $(1024)) + (tl.load($(parameter1_ptr) + r32_z, dtype=tl.int64) % $(1024))), dtype=tl.float32)
    r32_value = r32_value.to(tl.float32)
    r32_value = tl.broadcast_to(r32_value, [$(96)])
    tl.store($(candidate_vgather_5_ptr) + r32_z, r32_value)
    r33_z = tl.arange($(96))
    r33_value = (tl.load($(candidate_vscatter_add_ptr) + r33_z, dtype=tl.float32) + tl.load($(candidate_vgather_5_ptr) + r33_z, dtype=tl.float32))
    r33_value = r33_value.to(tl.float32)
    r33_value = tl.broadcast_to(r33_value, [$(96)])
    tl.store($(candidate_vadd_5_ptr) + r33_z, r33_value)
    r34_z = tl.arange($(98304))
    r34_value = (tl.load($(candidate_vneg_8_ptr) + ((((r34_z / $(1024)) % $(96)) * $(1024)) + (r34_z % $(1024))), dtype=tl.float32) + 0.0)
    r34_value = r34_value.to(tl.float32)
    r34_value = tl.broadcast_to(r34_value, [$(98304)])
    tl.store($(candidate_vadd_6_ptr) + r34_z, r34_value)
    r35_z = tl.arange($(98304))
    r35_value = tl.broadcast_to(tl.load($(candidate_vadd_6_ptr) + ((((r35_z / $(1024)) % $(96)) * $(1024)) + (r35_z % $(1024))), dtype=tl.float32), [$(98304)])
    for r35_lane in range($(1)) {
      r35_value = tl.where(tl.load($(parameter1_ptr) + ((((r35_z / $(1024)) % $(96)) + r35_lane) % $(96)), dtype=tl.int64) == tl.cast((r35_z % $(1024)), tl.int64), tl.load($(candidate_vadd_5_ptr) + ((r35_z / $(1024)) % $(96)), dtype=tl.float32), r35_value)
    }
    r35_value = r35_value.to(tl.float32)
    r35_value = tl.broadcast_to(r35_value, [$(98304)])
    tl.store($(candidate_vscatter_ptr) + r35_z, r35_value)
    r36_z = tl.arange($(96))
    r36_value = tl.load($(candidate_vscatter_ptr) + ((r36_z * $(1024)) + (tl.load($(candidate_vmax_5_index_ptr) + r36_z, dtype=tl.int64) % $(1024))), dtype=tl.float32)
    r36_value = r36_value.to(tl.float32)
    r36_value = tl.broadcast_to(r36_value, [$(96)])
    tl.store($(candidate_vgather_6_ptr) + r36_z, r36_value)
    r37_z = tl.arange($(96))
    r37_value = (tl.load($(candidate_vgather_6_ptr) + r37_z, dtype=tl.float32) + tl.load($(candidate_vadd_4_ptr) + r37_z, dtype=tl.float32))
    r37_value = r37_value.to(tl.float32)
    r37_value = tl.broadcast_to(r37_value, [$(96)])
    tl.store($(candidate_vadd_7_ptr) + r37_z, r37_value)
    r38_z = tl.arange($(98304))
    r38_value = (tl.load($(candidate_vscatter_ptr) + ((((r38_z / $(1024)) % $(96)) * $(1024)) + (r38_z % $(1024))), dtype=tl.float32) + 0.0)
    r38_value = r38_value.to(tl.float32)
    r38_value = tl.broadcast_to(r38_value, [$(98304)])
    tl.store($(candidate_vadd_8_ptr) + r38_z, r38_value)
    r39_z = tl.arange($(98304))
    r39_value = tl.broadcast_to(tl.load($(candidate_vadd_8_ptr) + ((((r39_z / $(1024)) % $(96)) * $(1024)) + (r39_z % $(1024))), dtype=tl.float32), [$(98304)])
    for r39_lane in range($(1)) {
      r39_value = tl.where(tl.load($(candidate_vmax_5_index_ptr) + ((r39_z / $(1024)) % $(96)), dtype=tl.int64) == tl.cast((r39_z % $(1024)), tl.int64), tl.load($(candidate_vadd_7_ptr) + ((r39_z / $(1024)) % $(96)), dtype=tl.float32), r39_value)
    }
    r39_value = r39_value.to(tl.float32)
    r39_value = tl.broadcast_to(r39_value, [$(98304)])
    tl.store($(candidate_vscatter_1_ptr) + r39_z, r39_value)
    r40_z = tl.arange($(98304))
    r40_value = tl.load($(candidate_vscatter_1_ptr) + ((((r40_z / $(1024)) % $(96)) * $(1024)) + (r40_z % $(1024))), dtype=tl.float32)
    r40_value = r40_value.to(tl.bfloat16)
    r40_value = tl.broadcast_to(r40_value, [$(98304)])
    tl.store($(candidate_vto_copy_7_ptr) + r40_z, r40_value)
    r41_z = tl.arange($(98304))
    r41_value = tl.load($(candidate_vto_copy_7_ptr) + ((((r41_z / $(1024)) % $(96)) * $(1024)) + (r41_z % $(1024))), dtype=tl.bfloat16)
    r41_value = r41_value.to(tl.float32)
    r41_value = tl.broadcast_to(r41_value, [$(98304)])
    tl.store($(candidate_vto_copy_8_ptr) + r41_z, r41_value)
    r42_z = tl.arange($(6144))
    r42_value = tl.load($(input0_ptr) + ((((r42_z / $(64)) % $(96)) * $(64)) + (r42_z % $(64))), dtype=tl.bfloat16)
    r42_value = r42_value.to(tl.float32)
    r42_value = tl.broadcast_to(r42_value, [$(6144)])
    tl.store($(candidate_vto_copy_9_ptr) + r42_z, r42_value)
    r43_row = tl.arange($(1024))
    r43_col = tl.arange($(64))
    r43_lane = tl.arange($(96))
    r43_left = tl.load($(candidate_vto_copy_8_ptr) + (((r43_lane[None, :] % $(96)) * $(1024)) + (r43_row[:, None] % $(1024))), dtype=tl.float32)
    r43_right = tl.load($(candidate_vto_copy_9_ptr) + (((r43_lane[:, None] % $(96)) * $(64)) + (r43_col[None, :] % $(64))), dtype=tl.float32)
    r43_value = tl.dot(r43_left, r43_right)
    r43_value = r43_value.to(tl.float32)
    tl.store($(candidate_vmm_6_ptr) + ((r43_row[:, None] * $(64)) + r43_col[None, :]), r43_value)
    r44_z = tl.arange($(65536))
    r44_value = (0.0 + tl.load($(candidate_vmm_6_ptr) + ((((r44_z / $(64)) % $(1024)) * $(64)) + (r44_z % $(64))), dtype=tl.float32))
    r44_value = r44_value.to(tl.float32)
    r44_value = tl.broadcast_to(r44_value, [$(65536)])
    tl.store($(candidate_vadd_9_ptr) + r44_z, r44_value)
    r45_row = tl.arange($(96))
    r45_col = tl.arange($(1024))
    r45_lane = tl.arange($(64))
    r45_left = tl.load($(input0_ptr) + ((((((r45_row[:, None] * $(64)) + r45_lane[None, :] + $(6144)) / $(64)) % $(320)) * $(64)) + (((r45_row[:, None] * $(64)) + r45_lane[None, :] + $(6144)) % $(64))), dtype=tl.bfloat16)
    r45_right = tl.load($(candidate_vmul_ptr) + (((r45_col[None, :] % $(1024)) * $(64)) + (r45_lane[:, None] % $(64))), dtype=tl.bfloat16)
    r45_value = tl.dot(r45_left, r45_right)
    r45_value = r45_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_7_ptr) + ((r45_row[:, None] * $(1024)) + r45_col[None, :]), r45_value)
    r46_z = tl.arange($(98304))
    r46_value = tl.load($(candidate_vmm_7_ptr) + ((((r46_z / $(1024)) % $(96)) * $(1024)) + (r46_z % $(1024))), dtype=tl.bfloat16)
    r46_value = r46_value.to(tl.float32)
    r46_value = tl.broadcast_to(r46_value, [$(98304)])
    tl.store($(candidate_vto_copy_10_ptr) + r46_z, r46_value)
    r47_z = tl.arange($(96))
    r47_lane = tl.arange($(1024))
    r47_row_values = tl.load($(candidate_vto_copy_10_ptr) + ((r47_z[:, None] * $(1024)) + (r47_lane[None, :] % $(1024))), dtype=tl.float32)
    r47_value = tl.max(r47_row_values, axis=1)
    r47_value = r47_value.to(tl.float32)
    r47_value = tl.broadcast_to(r47_value, [$(96)])
    tl.store($(candidate_vmax_6_value_ptr) + r47_z, r47_value)
    r48_z = tl.arange($(96))
    r48_lane = tl.arange($(1024))
    r48_row_values = tl.load($(candidate_vto_copy_10_ptr) + ((r48_z[:, None] * $(1024)) + (r48_lane[None, :] % $(1024))), dtype=tl.float32)
    r48_value = tl.argmax(r48_row_values, axis=1)
    r48_value = tl.cast(r48_value, tl.int64)
    r48_value = tl.broadcast_to(r48_value, [$(96)])
    tl.store($(candidate_vmax_6_index_ptr) + r48_z, r48_value)
    r49_z = tl.arange($(98304))
    r49_value = (tl.load($(candidate_vto_copy_10_ptr) + ((((r49_z / $(1024)) % $(96)) * $(1024)) + (r49_z % $(1024))), dtype=tl.float32) - tl.load($(candidate_vmax_6_value_ptr) + ((r49_z / $(1024)) % $(96)), dtype=tl.float32))
    r49_value = r49_value.to(tl.float32)
    r49_value = tl.broadcast_to(r49_value, [$(98304)])
    tl.store($(candidate_vsub_16_ptr) + r49_z, r49_value)
    r50_z = tl.arange($(98304))
    r50_value = tl.exp(tl.load($(candidate_vsub_16_ptr) + ((((r50_z / $(1024)) % $(96)) * $(1024)) + (r50_z % $(1024))), dtype=tl.float32))
    r50_value = r50_value.to(tl.float32)
    r50_value = tl.broadcast_to(r50_value, [$(98304)])
    tl.store($(candidate_vexp_6_ptr) + r50_z, r50_value)
    r51_z = tl.arange($(96))
    r51_lane = tl.arange($(1024))
    r51_value = tl.sum(tl.load($(candidate_vexp_6_ptr) + ((r51_z[:, None] * $(1024)) + r51_lane[None, :]), dtype=tl.float32), axis=1)
    r51_value = r51_value.to(tl.float32)
    r51_value = tl.broadcast_to(r51_value, [$(96)])
    tl.store($(candidate_vsum_13_ptr) + r51_z, r51_value)
    r52_z = tl.arange($(96))
    r52_value = tl.log(tl.load($(candidate_vsum_13_ptr) + r52_z, dtype=tl.float32))
    r52_value = r52_value.to(tl.float32)
    r52_value = tl.broadcast_to(r52_value, [$(96)])
    tl.store($(candidate_vlog_5_ptr) + r52_z, r52_value)
    r53_z = tl.arange($(96))
    r53_value = tl.load($(candidate_vto_copy_10_ptr) + ((r53_z * $(1024)) + (tl.load($(parameter1_ptr) + (r53_z + $(96)), dtype=tl.int64) % $(1024))), dtype=tl.float32)
    r53_value = r53_value.to(tl.float32)
    r53_value = tl.broadcast_to(r53_value, [$(96)])
    tl.store($(candidate_vgather_7_ptr) + r53_z, r53_value)
    r54_z = tl.arange($(96))
    r54_value = (tl.load($(candidate_vgather_7_ptr) + r54_z, dtype=tl.float32) - tl.load($(candidate_vmax_6_value_ptr) + r54_z, dtype=tl.float32))
    r54_value = r54_value.to(tl.float32)
    r54_value = tl.broadcast_to(r54_value, [$(96)])
    tl.store($(candidate_vsub_17_ptr) + r54_z, r54_value)
    r55_z = tl.arange($(96))
    r55_value = (tl.load($(candidate_vsub_17_ptr) + r55_z, dtype=tl.float32) - tl.load($(candidate_vlog_5_ptr) + r55_z, dtype=tl.float32))
    r55_value = r55_value.to(tl.float32)
    r55_value = tl.broadcast_to(r55_value, [$(96)])
    tl.store($(candidate_vsub_18_ptr) + r55_z, r55_value)
    r56_z = tl.arange($(96))
    r56_value = -tl.load($(candidate_vsub_18_ptr) + r56_z, dtype=tl.float32)
    r56_value = r56_value.to(tl.float32)
    r56_value = tl.broadcast_to(r56_value, [$(96)])
    tl.store($(candidate_vneg_9_ptr) + r56_z, r56_value)
    r57_z = tl.arange($(96))
    r57_value = (tl.load($(candidate_vneg_9_ptr) + r57_z, dtype=tl.float32) * tl.load($(candidate_vto_copy_ptr) + (r57_z + $(96)), dtype=tl.float32))
    r57_value = r57_value.to(tl.float32)
    r57_value = tl.broadcast_to(r57_value, [$(96)])
    tl.store($(candidate_vmul_10_ptr) + r57_z, r57_value)
    r58_z = tl.arange($(1))
    r58_lane = tl.arange($(96))
    r58_value = tl.sum(tl.load($(candidate_vmul_10_ptr) + r58_lane[None, :], dtype=tl.float32), axis=1)
    r58_value = r58_value.to(tl.float32)
    r58_value = tl.broadcast_to(r58_value, [$(1)])
    tl.store($(candidate_vsum_14_ptr) + r58_z, r58_value)
    r59_z = tl.arange($(1))
    r59_value = (tl.load($(candidate_vsum_14_ptr) + $(0), dtype=tl.float32) * tl.load($(candidate_vto_copy_5_ptr) + $(0), dtype=tl.float32))
    r59_value = r59_value.to(tl.float32)
    r59_value = tl.broadcast_to(r59_value, [$(1)])
    tl.store($(candidate_vmul_11_ptr) + r59_z, r59_value)
    r60_z = tl.arange($(1))
    r60_value = (1.0 * tl.load($(candidate_vto_copy_5_ptr) + $(0), dtype=tl.float32))
    r60_value = r60_value.to(tl.float32)
    r60_value = tl.broadcast_to(r60_value, [$(1)])
    tl.store($(candidate_vmul_12_ptr) + r60_z, r60_value)
    r61_z = tl.arange($(96))
    r61_value = (tl.load($(candidate_vmul_12_ptr) + $(0), dtype=tl.float32) * tl.load($(candidate_vto_copy_ptr) + (r61_z + $(96)), dtype=tl.float32))
    r61_value = r61_value.to(tl.float32)
    r61_value = tl.broadcast_to(r61_value, [$(96)])
    tl.store($(candidate_vmul_13_ptr) + r61_z, r61_value)
    r62_z = tl.arange($(96))
    r62_value = -tl.load($(candidate_vmul_13_ptr) + r62_z, dtype=tl.float32)
    r62_value = r62_value.to(tl.float32)
    r62_value = tl.broadcast_to(r62_value, [$(96)])
    tl.store($(candidate_vneg_10_ptr) + r62_z, r62_value)
    r63_z = tl.arange($(96))
    r63_value = -tl.load($(candidate_vneg_10_ptr) + r63_z, dtype=tl.float32)
    r63_value = r63_value.to(tl.float32)
    r63_value = tl.broadcast_to(r63_value, [$(96)])
    tl.store($(candidate_vneg_11_ptr) + r63_z, r63_value)
    r64_z = tl.arange($(98304))
    r64_value = (tl.load($(candidate_vto_copy_10_ptr) + ((((r64_z / $(1024)) % $(96)) * $(1024)) + (r64_z % $(1024))), dtype=tl.float32) - tl.load($(candidate_vmax_6_value_ptr) + ((r64_z / $(1024)) % $(96)), dtype=tl.float32))
    r64_value = r64_value.to(tl.float32)
    r64_value = tl.broadcast_to(r64_value, [$(98304)])
    tl.store($(candidate_vsub_19_ptr) + r64_z, r64_value)
    r65_z = tl.arange($(98304))
    r65_value = tl.exp(tl.load($(candidate_vsub_19_ptr) + ((((r65_z / $(1024)) % $(96)) * $(1024)) + (r65_z % $(1024))), dtype=tl.float32))
    r65_value = r65_value.to(tl.float32)
    r65_value = tl.broadcast_to(r65_value, [$(98304)])
    tl.store($(candidate_vexp_7_ptr) + r65_z, r65_value)
    r66_z = tl.arange($(96))
    r66_value = (tl.load($(candidate_vneg_11_ptr) + r66_z, dtype=tl.float32) / tl.load($(candidate_vsum_13_ptr) + r66_z, dtype=tl.float32))
    r66_value = r66_value.to(tl.float32)
    r66_value = tl.broadcast_to(r66_value, [$(96)])
    tl.store($(candidate_vdiv_1_ptr) + r66_z, r66_value)
    r67_z = tl.arange($(98304))
    r67_value = (tl.load($(candidate_vdiv_1_ptr) + ((r67_z / $(1024)) % $(96)), dtype=tl.float32) * tl.load($(candidate_vexp_7_ptr) + ((((r67_z / $(1024)) % $(96)) * $(1024)) + (r67_z % $(1024))), dtype=tl.float32))
    r67_value = r67_value.to(tl.float32)
    r67_value = tl.broadcast_to(r67_value, [$(98304)])
    tl.store($(candidate_vmul_14_ptr) + r67_z, r67_value)
    r68_z = tl.arange($(98304))
    r68_value = -tl.load($(candidate_vmul_14_ptr) + ((((r68_z / $(1024)) % $(96)) * $(1024)) + (r68_z % $(1024))), dtype=tl.float32)
    r68_value = r68_value.to(tl.float32)
    r68_value = tl.broadcast_to(r68_value, [$(98304)])
    tl.store($(candidate_vneg_12_ptr) + r68_z, r68_value)
    r69_z = tl.arange($(96))
    r69_lane = tl.arange($(1024))
    r69_value = tl.sum(tl.load($(candidate_vneg_12_ptr) + ((r69_z[:, None] * $(1024)) + r69_lane[None, :]), dtype=tl.float32), axis=1)
    r69_value = r69_value.to(tl.float32)
    r69_value = tl.broadcast_to(r69_value, [$(96)])
    tl.store($(candidate_vsum_15_ptr) + r69_z, r69_value)
    r70_z = tl.arange($(96))
    r70_value = (tl.load($(candidate_vneg_11_ptr) + r70_z, dtype=tl.float32) + tl.load($(candidate_vsum_15_ptr) + r70_z, dtype=tl.float32))
    r70_value = r70_value.to(tl.float32)
    r70_value = tl.broadcast_to(r70_value, [$(96)])
    tl.store($(candidate_vadd_10_ptr) + r70_z, r70_value)
    r71_z = tl.arange($(98304))
    r71_value = -tl.load($(candidate_vneg_12_ptr) + ((((r71_z / $(1024)) % $(96)) * $(1024)) + (r71_z % $(1024))), dtype=tl.float32)
    r71_value = r71_value.to(tl.float32)
    r71_value = tl.broadcast_to(r71_value, [$(98304)])
    tl.store($(candidate_vneg_13_ptr) + r71_z, r71_value)
    r72_z = tl.arange($(96))
    r72_value = tl.broadcast_to(0.0, [$(96)])
    for r72_lane in range($(1)) {
      r72_value = tl.where(tl.full([], $(0), dtype=tl.int64) == tl.cast($(0), tl.int64), (r72_value + tl.load($(candidate_vneg_10_ptr) + ((r72_z + r72_lane) % $(96)), dtype=tl.float32)), r72_value)
    }
    r72_value = r72_value.to(tl.float32)
    r72_value = tl.broadcast_to(r72_value, [$(96)])
    tl.store($(candidate_vscatter_add_1_ptr) + r72_z, r72_value)
    r73_z = tl.arange($(96))
    r73_value = tl.load($(candidate_vneg_13_ptr) + ((r73_z * $(1024)) + (tl.load($(parameter1_ptr) + (r73_z + $(96)), dtype=tl.int64) % $(1024))), dtype=tl.float32)
    r73_value = r73_value.to(tl.float32)
    r73_value = tl.broadcast_to(r73_value, [$(96)])
    tl.store($(candidate_vgather_8_ptr) + r73_z, r73_value)
    r74_z = tl.arange($(96))
    r74_value = (tl.load($(candidate_vscatter_add_1_ptr) + r74_z, dtype=tl.float32) + tl.load($(candidate_vgather_8_ptr) + r74_z, dtype=tl.float32))
    r74_value = r74_value.to(tl.float32)
    r74_value = tl.broadcast_to(r74_value, [$(96)])
    tl.store($(candidate_vadd_11_ptr) + r74_z, r74_value)
    r75_z = tl.arange($(98304))
    r75_value = (tl.load($(candidate_vneg_13_ptr) + ((((r75_z / $(1024)) % $(96)) * $(1024)) + (r75_z % $(1024))), dtype=tl.float32) + 0.0)
    r75_value = r75_value.to(tl.float32)
    r75_value = tl.broadcast_to(r75_value, [$(98304)])
    tl.store($(candidate_vadd_12_ptr) + r75_z, r75_value)
    r76_z = tl.arange($(98304))
    r76_value = tl.broadcast_to(tl.load($(candidate_vadd_12_ptr) + ((((r76_z / $(1024)) % $(96)) * $(1024)) + (r76_z % $(1024))), dtype=tl.float32), [$(98304)])
    for r76_lane in range($(1)) {
      r76_value = tl.where(tl.load($(parameter1_ptr) + (((((r76_z / $(1024)) % $(96)) + r76_lane) % $(96)) + $(96)), dtype=tl.int64) == tl.cast((r76_z % $(1024)), tl.int64), tl.load($(candidate_vadd_11_ptr) + ((r76_z / $(1024)) % $(96)), dtype=tl.float32), r76_value)
    }
    r76_value = r76_value.to(tl.float32)
    r76_value = tl.broadcast_to(r76_value, [$(98304)])
    tl.store($(candidate_vscatter_2_ptr) + r76_z, r76_value)
    r77_z = tl.arange($(96))
    r77_value = tl.load($(candidate_vscatter_2_ptr) + ((r77_z * $(1024)) + (tl.load($(candidate_vmax_6_index_ptr) + r77_z, dtype=tl.int64) % $(1024))), dtype=tl.float32)
    r77_value = r77_value.to(tl.float32)
    r77_value = tl.broadcast_to(r77_value, [$(96)])
    tl.store($(candidate_vgather_9_ptr) + r77_z, r77_value)
    r78_z = tl.arange($(96))
    r78_value = (tl.load($(candidate_vgather_9_ptr) + r78_z, dtype=tl.float32) + tl.load($(candidate_vadd_10_ptr) + r78_z, dtype=tl.float32))
    r78_value = r78_value.to(tl.float32)
    r78_value = tl.broadcast_to(r78_value, [$(96)])
    tl.store($(candidate_vadd_13_ptr) + r78_z, r78_value)
    r79_z = tl.arange($(98304))
    r79_value = (tl.load($(candidate_vscatter_2_ptr) + ((((r79_z / $(1024)) % $(96)) * $(1024)) + (r79_z % $(1024))), dtype=tl.float32) + 0.0)
    r79_value = r79_value.to(tl.float32)
    r79_value = tl.broadcast_to(r79_value, [$(98304)])
    tl.store($(candidate_vadd_14_ptr) + r79_z, r79_value)
    r80_z = tl.arange($(98304))
    r80_value = tl.broadcast_to(tl.load($(candidate_vadd_14_ptr) + ((((r80_z / $(1024)) % $(96)) * $(1024)) + (r80_z % $(1024))), dtype=tl.float32), [$(98304)])
    for r80_lane in range($(1)) {
      r80_value = tl.where(tl.load($(candidate_vmax_6_index_ptr) + ((r80_z / $(1024)) % $(96)), dtype=tl.int64) == tl.cast((r80_z % $(1024)), tl.int64), tl.load($(candidate_vadd_13_ptr) + ((r80_z / $(1024)) % $(96)), dtype=tl.float32), r80_value)
    }
    r80_value = r80_value.to(tl.float32)
    r80_value = tl.broadcast_to(r80_value, [$(98304)])
    tl.store($(candidate_vscatter_3_ptr) + r80_z, r80_value)
    r81_z = tl.arange($(98304))
    r81_value = tl.load($(candidate_vscatter_3_ptr) + ((((r81_z / $(1024)) % $(96)) * $(1024)) + (r81_z % $(1024))), dtype=tl.float32)
    r81_value = r81_value.to(tl.bfloat16)
    r81_value = tl.broadcast_to(r81_value, [$(98304)])
    tl.store($(candidate_vto_copy_11_ptr) + r81_z, r81_value)
    r82_z = tl.arange($(98304))
    r82_value = tl.load($(candidate_vto_copy_11_ptr) + ((((r82_z / $(1024)) % $(96)) * $(1024)) + (r82_z % $(1024))), dtype=tl.bfloat16)
    r82_value = r82_value.to(tl.float32)
    r82_value = tl.broadcast_to(r82_value, [$(98304)])
    tl.store($(candidate_vto_copy_12_ptr) + r82_z, r82_value)
    r83_z = tl.arange($(6144))
    r83_value = tl.load($(input0_ptr) + ((((r83_z / $(64)) % $(96)) * $(64)) + (r83_z % $(64)) + $(6144)), dtype=tl.bfloat16)
    r83_value = r83_value.to(tl.float32)
    r83_value = tl.broadcast_to(r83_value, [$(6144)])
    tl.store($(candidate_vto_copy_13_ptr) + r83_z, r83_value)
    r84_row = tl.arange($(1024))
    r84_col = tl.arange($(64))
    r84_lane = tl.arange($(96))
    r84_left = tl.load($(candidate_vto_copy_12_ptr) + (((r84_lane[None, :] % $(96)) * $(1024)) + (r84_row[:, None] % $(1024))), dtype=tl.float32)
    r84_right = tl.load($(candidate_vto_copy_13_ptr) + (((r84_lane[:, None] % $(96)) * $(64)) + (r84_col[None, :] % $(64))), dtype=tl.float32)
    r84_value = tl.dot(r84_left, r84_right)
    r84_value = r84_value.to(tl.float32)
    tl.store($(candidate_vmm_9_ptr) + ((r84_row[:, None] * $(64)) + r84_col[None, :]), r84_value)
    r85_z = tl.arange($(65536))
    r85_value = (tl.load($(candidate_vadd_9_ptr) + ((((r85_z / $(64)) % $(1024)) * $(64)) + (r85_z % $(64))), dtype=tl.float32) + tl.load($(candidate_vmm_9_ptr) + ((((r85_z / $(64)) % $(1024)) * $(64)) + (r85_z % $(64))), dtype=tl.float32))
    r85_value = r85_value.to(tl.float32)
    r85_value = tl.broadcast_to(r85_value, [$(65536)])
    tl.store($(candidate_vadd_15_ptr) + r85_z, r85_value)
    r86_row = tl.arange($(96))
    r86_col = tl.arange($(1024))
    r86_lane = tl.arange($(64))
    r86_left = tl.load($(input0_ptr) + ((((((r86_row[:, None] * $(64)) + r86_lane[None, :] + $(12288)) / $(64)) % $(320)) * $(64)) + (((r86_row[:, None] * $(64)) + r86_lane[None, :] + $(12288)) % $(64))), dtype=tl.bfloat16)
    r86_right = tl.load($(candidate_vmul_ptr) + (((r86_col[None, :] % $(1024)) * $(64)) + (r86_lane[:, None] % $(64))), dtype=tl.bfloat16)
    r86_value = tl.dot(r86_left, r86_right)
    r86_value = r86_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_10_ptr) + ((r86_row[:, None] * $(1024)) + r86_col[None, :]), r86_value)
    r87_z = tl.arange($(98304))
    r87_value = tl.load($(candidate_vmm_10_ptr) + ((((r87_z / $(1024)) % $(96)) * $(1024)) + (r87_z % $(1024))), dtype=tl.bfloat16)
    r87_value = r87_value.to(tl.float32)
    r87_value = tl.broadcast_to(r87_value, [$(98304)])
    tl.store($(candidate_vto_copy_14_ptr) + r87_z, r87_value)
    r88_z = tl.arange($(96))
    r88_lane = tl.arange($(1024))
    r88_row_values = tl.load($(candidate_vto_copy_14_ptr) + ((r88_z[:, None] * $(1024)) + (r88_lane[None, :] % $(1024))), dtype=tl.float32)
    r88_value = tl.max(r88_row_values, axis=1)
    r88_value = r88_value.to(tl.float32)
    r88_value = tl.broadcast_to(r88_value, [$(96)])
    tl.store($(candidate_vmax_7_value_ptr) + r88_z, r88_value)
    r89_z = tl.arange($(96))
    r89_lane = tl.arange($(1024))
    r89_row_values = tl.load($(candidate_vto_copy_14_ptr) + ((r89_z[:, None] * $(1024)) + (r89_lane[None, :] % $(1024))), dtype=tl.float32)
    r89_value = tl.argmax(r89_row_values, axis=1)
    r89_value = tl.cast(r89_value, tl.int64)
    r89_value = tl.broadcast_to(r89_value, [$(96)])
    tl.store($(candidate_vmax_7_index_ptr) + r89_z, r89_value)
    r90_z = tl.arange($(98304))
    r90_value = (tl.load($(candidate_vto_copy_14_ptr) + ((((r90_z / $(1024)) % $(96)) * $(1024)) + (r90_z % $(1024))), dtype=tl.float32) - tl.load($(candidate_vmax_7_value_ptr) + ((r90_z / $(1024)) % $(96)), dtype=tl.float32))
    r90_value = r90_value.to(tl.float32)
    r90_value = tl.broadcast_to(r90_value, [$(98304)])
    tl.store($(candidate_vsub_20_ptr) + r90_z, r90_value)
    r91_z = tl.arange($(98304))
    r91_value = tl.exp(tl.load($(candidate_vsub_20_ptr) + ((((r91_z / $(1024)) % $(96)) * $(1024)) + (r91_z % $(1024))), dtype=tl.float32))
    r91_value = r91_value.to(tl.float32)
    r91_value = tl.broadcast_to(r91_value, [$(98304)])
    tl.store($(candidate_vexp_8_ptr) + r91_z, r91_value)
    r92_z = tl.arange($(96))
    r92_lane = tl.arange($(1024))
    r92_value = tl.sum(tl.load($(candidate_vexp_8_ptr) + ((r92_z[:, None] * $(1024)) + r92_lane[None, :]), dtype=tl.float32), axis=1)
    r92_value = r92_value.to(tl.float32)
    r92_value = tl.broadcast_to(r92_value, [$(96)])
    tl.store($(candidate_vsum_16_ptr) + r92_z, r92_value)
    r93_z = tl.arange($(96))
    r93_value = tl.log(tl.load($(candidate_vsum_16_ptr) + r93_z, dtype=tl.float32))
    r93_value = r93_value.to(tl.float32)
    r93_value = tl.broadcast_to(r93_value, [$(96)])
    tl.store($(candidate_vlog_6_ptr) + r93_z, r93_value)
    r94_z = tl.arange($(96))
    r94_value = tl.load($(candidate_vto_copy_14_ptr) + ((r94_z * $(1024)) + (tl.load($(parameter1_ptr) + (r94_z + $(192)), dtype=tl.int64) % $(1024))), dtype=tl.float32)
    r94_value = r94_value.to(tl.float32)
    r94_value = tl.broadcast_to(r94_value, [$(96)])
    tl.store($(candidate_vgather_10_ptr) + r94_z, r94_value)
    r95_z = tl.arange($(96))
    r95_value = (tl.load($(candidate_vgather_10_ptr) + r95_z, dtype=tl.float32) - tl.load($(candidate_vmax_7_value_ptr) + r95_z, dtype=tl.float32))
    r95_value = r95_value.to(tl.float32)
    r95_value = tl.broadcast_to(r95_value, [$(96)])
    tl.store($(candidate_vsub_21_ptr) + r95_z, r95_value)
    r96_z = tl.arange($(96))
    r96_value = (tl.load($(candidate_vsub_21_ptr) + r96_z, dtype=tl.float32) - tl.load($(candidate_vlog_6_ptr) + r96_z, dtype=tl.float32))
    r96_value = r96_value.to(tl.float32)
    r96_value = tl.broadcast_to(r96_value, [$(96)])
    tl.store($(candidate_vsub_22_ptr) + r96_z, r96_value)
    r97_z = tl.arange($(96))
    r97_value = -tl.load($(candidate_vsub_22_ptr) + r97_z, dtype=tl.float32)
    r97_value = r97_value.to(tl.float32)
    r97_value = tl.broadcast_to(r97_value, [$(96)])
    tl.store($(candidate_vneg_14_ptr) + r97_z, r97_value)
    r98_z = tl.arange($(96))
    r98_value = (tl.load($(candidate_vneg_14_ptr) + r98_z, dtype=tl.float32) * tl.load($(candidate_vto_copy_ptr) + (r98_z + $(192)), dtype=tl.float32))
    r98_value = r98_value.to(tl.float32)
    r98_value = tl.broadcast_to(r98_value, [$(96)])
    tl.store($(candidate_vmul_15_ptr) + r98_z, r98_value)
    r99_z = tl.arange($(1))
    r99_lane = tl.arange($(96))
    r99_value = tl.sum(tl.load($(candidate_vmul_15_ptr) + r99_lane[None, :], dtype=tl.float32), axis=1)
    r99_value = r99_value.to(tl.float32)
    r99_value = tl.broadcast_to(r99_value, [$(1)])
    tl.store($(candidate_vsum_17_ptr) + r99_z, r99_value)
    r100_z = tl.arange($(1))
    r100_value = (tl.load($(candidate_vsum_17_ptr) + $(0), dtype=tl.float32) * tl.load($(candidate_vto_copy_5_ptr) + $(0), dtype=tl.float32))
    r100_value = r100_value.to(tl.float32)
    r100_value = tl.broadcast_to(r100_value, [$(1)])
    tl.store($(candidate_vmul_16_ptr) + r100_z, r100_value)
    r101_z = tl.arange($(1))
    r101_value = (1.0 * tl.load($(candidate_vto_copy_5_ptr) + $(0), dtype=tl.float32))
    r101_value = r101_value.to(tl.float32)
    r101_value = tl.broadcast_to(r101_value, [$(1)])
    tl.store($(candidate_vmul_17_ptr) + r101_z, r101_value)
    r102_z = tl.arange($(96))
    r102_value = (tl.load($(candidate_vmul_17_ptr) + $(0), dtype=tl.float32) * tl.load($(candidate_vto_copy_ptr) + (r102_z + $(192)), dtype=tl.float32))
    r102_value = r102_value.to(tl.float32)
    r102_value = tl.broadcast_to(r102_value, [$(96)])
    tl.store($(candidate_vmul_18_ptr) + r102_z, r102_value)
    r103_z = tl.arange($(96))
    r103_value = -tl.load($(candidate_vmul_18_ptr) + r103_z, dtype=tl.float32)
    r103_value = r103_value.to(tl.float32)
    r103_value = tl.broadcast_to(r103_value, [$(96)])
    tl.store($(candidate_vneg_15_ptr) + r103_z, r103_value)
    r104_z = tl.arange($(96))
    r104_value = -tl.load($(candidate_vneg_15_ptr) + r104_z, dtype=tl.float32)
    r104_value = r104_value.to(tl.float32)
    r104_value = tl.broadcast_to(r104_value, [$(96)])
    tl.store($(candidate_vneg_16_ptr) + r104_z, r104_value)
    r105_z = tl.arange($(98304))
    r105_value = (tl.load($(candidate_vto_copy_14_ptr) + ((((r105_z / $(1024)) % $(96)) * $(1024)) + (r105_z % $(1024))), dtype=tl.float32) - tl.load($(candidate_vmax_7_value_ptr) + ((r105_z / $(1024)) % $(96)), dtype=tl.float32))
    r105_value = r105_value.to(tl.float32)
    r105_value = tl.broadcast_to(r105_value, [$(98304)])
    tl.store($(candidate_vsub_23_ptr) + r105_z, r105_value)
    r106_z = tl.arange($(98304))
    r106_value = tl.exp(tl.load($(candidate_vsub_23_ptr) + ((((r106_z / $(1024)) % $(96)) * $(1024)) + (r106_z % $(1024))), dtype=tl.float32))
    r106_value = r106_value.to(tl.float32)
    r106_value = tl.broadcast_to(r106_value, [$(98304)])
    tl.store($(candidate_vexp_9_ptr) + r106_z, r106_value)
    r107_z = tl.arange($(96))
    r107_value = (tl.load($(candidate_vneg_16_ptr) + r107_z, dtype=tl.float32) / tl.load($(candidate_vsum_16_ptr) + r107_z, dtype=tl.float32))
    r107_value = r107_value.to(tl.float32)
    r107_value = tl.broadcast_to(r107_value, [$(96)])
    tl.store($(candidate_vdiv_2_ptr) + r107_z, r107_value)
    r108_z = tl.arange($(98304))
    r108_value = (tl.load($(candidate_vdiv_2_ptr) + ((r108_z / $(1024)) % $(96)), dtype=tl.float32) * tl.load($(candidate_vexp_9_ptr) + ((((r108_z / $(1024)) % $(96)) * $(1024)) + (r108_z % $(1024))), dtype=tl.float32))
    r108_value = r108_value.to(tl.float32)
    r108_value = tl.broadcast_to(r108_value, [$(98304)])
    tl.store($(candidate_vmul_19_ptr) + r108_z, r108_value)
    r109_z = tl.arange($(98304))
    r109_value = -tl.load($(candidate_vmul_19_ptr) + ((((r109_z / $(1024)) % $(96)) * $(1024)) + (r109_z % $(1024))), dtype=tl.float32)
    r109_value = r109_value.to(tl.float32)
    r109_value = tl.broadcast_to(r109_value, [$(98304)])
    tl.store($(candidate_vneg_17_ptr) + r109_z, r109_value)
    r110_z = tl.arange($(96))
    r110_lane = tl.arange($(1024))
    r110_value = tl.sum(tl.load($(candidate_vneg_17_ptr) + ((r110_z[:, None] * $(1024)) + r110_lane[None, :]), dtype=tl.float32), axis=1)
    r110_value = r110_value.to(tl.float32)
    r110_value = tl.broadcast_to(r110_value, [$(96)])
    tl.store($(candidate_vsum_18_ptr) + r110_z, r110_value)
    r111_z = tl.arange($(96))
    r111_value = (tl.load($(candidate_vneg_16_ptr) + r111_z, dtype=tl.float32) + tl.load($(candidate_vsum_18_ptr) + r111_z, dtype=tl.float32))
    r111_value = r111_value.to(tl.float32)
    r111_value = tl.broadcast_to(r111_value, [$(96)])
    tl.store($(candidate_vadd_16_ptr) + r111_z, r111_value)
    r112_z = tl.arange($(98304))
    r112_value = -tl.load($(candidate_vneg_17_ptr) + ((((r112_z / $(1024)) % $(96)) * $(1024)) + (r112_z % $(1024))), dtype=tl.float32)
    r112_value = r112_value.to(tl.float32)
    r112_value = tl.broadcast_to(r112_value, [$(98304)])
    tl.store($(candidate_vneg_18_ptr) + r112_z, r112_value)
    r113_z = tl.arange($(96))
    r113_value = tl.broadcast_to(0.0, [$(96)])
    for r113_lane in range($(1)) {
      r113_value = tl.where(tl.full([], $(0), dtype=tl.int64) == tl.cast($(0), tl.int64), (r113_value + tl.load($(candidate_vneg_15_ptr) + ((r113_z + r113_lane) % $(96)), dtype=tl.float32)), r113_value)
    }
    r113_value = r113_value.to(tl.float32)
    r113_value = tl.broadcast_to(r113_value, [$(96)])
    tl.store($(candidate_vscatter_add_2_ptr) + r113_z, r113_value)
    r114_z = tl.arange($(96))
    r114_value = tl.load($(candidate_vneg_18_ptr) + ((r114_z * $(1024)) + (tl.load($(parameter1_ptr) + (r114_z + $(192)), dtype=tl.int64) % $(1024))), dtype=tl.float32)
    r114_value = r114_value.to(tl.float32)
    r114_value = tl.broadcast_to(r114_value, [$(96)])
    tl.store($(candidate_vgather_11_ptr) + r114_z, r114_value)
    r115_z = tl.arange($(96))
    r115_value = (tl.load($(candidate_vscatter_add_2_ptr) + r115_z, dtype=tl.float32) + tl.load($(candidate_vgather_11_ptr) + r115_z, dtype=tl.float32))
    r115_value = r115_value.to(tl.float32)
    r115_value = tl.broadcast_to(r115_value, [$(96)])
    tl.store($(candidate_vadd_17_ptr) + r115_z, r115_value)
    r116_z = tl.arange($(98304))
    r116_value = (tl.load($(candidate_vneg_18_ptr) + ((((r116_z / $(1024)) % $(96)) * $(1024)) + (r116_z % $(1024))), dtype=tl.float32) + 0.0)
    r116_value = r116_value.to(tl.float32)
    r116_value = tl.broadcast_to(r116_value, [$(98304)])
    tl.store($(candidate_vadd_18_ptr) + r116_z, r116_value)
    r117_z = tl.arange($(98304))
    r117_value = tl.broadcast_to(tl.load($(candidate_vadd_18_ptr) + ((((r117_z / $(1024)) % $(96)) * $(1024)) + (r117_z % $(1024))), dtype=tl.float32), [$(98304)])
    for r117_lane in range($(1)) {
      r117_value = tl.where(tl.load($(parameter1_ptr) + (((((r117_z / $(1024)) % $(96)) + r117_lane) % $(96)) + $(192)), dtype=tl.int64) == tl.cast((r117_z % $(1024)), tl.int64), tl.load($(candidate_vadd_17_ptr) + ((r117_z / $(1024)) % $(96)), dtype=tl.float32), r117_value)
    }
    r117_value = r117_value.to(tl.float32)
    r117_value = tl.broadcast_to(r117_value, [$(98304)])
    tl.store($(candidate_vscatter_4_ptr) + r117_z, r117_value)
    r118_z = tl.arange($(96))
    r118_value = tl.load($(candidate_vscatter_4_ptr) + ((r118_z * $(1024)) + (tl.load($(candidate_vmax_7_index_ptr) + r118_z, dtype=tl.int64) % $(1024))), dtype=tl.float32)
    r118_value = r118_value.to(tl.float32)
    r118_value = tl.broadcast_to(r118_value, [$(96)])
    tl.store($(candidate_vgather_12_ptr) + r118_z, r118_value)
    r119_z = tl.arange($(96))
    r119_value = (tl.load($(candidate_vgather_12_ptr) + r119_z, dtype=tl.float32) + tl.load($(candidate_vadd_16_ptr) + r119_z, dtype=tl.float32))
    r119_value = r119_value.to(tl.float32)
    r119_value = tl.broadcast_to(r119_value, [$(96)])
    tl.store($(candidate_vadd_19_ptr) + r119_z, r119_value)
    r120_z = tl.arange($(98304))
    r120_value = (tl.load($(candidate_vscatter_4_ptr) + ((((r120_z / $(1024)) % $(96)) * $(1024)) + (r120_z % $(1024))), dtype=tl.float32) + 0.0)
    r120_value = r120_value.to(tl.float32)
    r120_value = tl.broadcast_to(r120_value, [$(98304)])
    tl.store($(candidate_vadd_20_ptr) + r120_z, r120_value)
    r121_z = tl.arange($(98304))
    r121_value = tl.broadcast_to(tl.load($(candidate_vadd_20_ptr) + ((((r121_z / $(1024)) % $(96)) * $(1024)) + (r121_z % $(1024))), dtype=tl.float32), [$(98304)])
    for r121_lane in range($(1)) {
      r121_value = tl.where(tl.load($(candidate_vmax_7_index_ptr) + ((r121_z / $(1024)) % $(96)), dtype=tl.int64) == tl.cast((r121_z % $(1024)), tl.int64), tl.load($(candidate_vadd_19_ptr) + ((r121_z / $(1024)) % $(96)), dtype=tl.float32), r121_value)
    }
    r121_value = r121_value.to(tl.float32)
    r121_value = tl.broadcast_to(r121_value, [$(98304)])
    tl.store($(candidate_vscatter_5_ptr) + r121_z, r121_value)
    r122_z = tl.arange($(98304))
    r122_value = tl.load($(candidate_vscatter_5_ptr) + ((((r122_z / $(1024)) % $(96)) * $(1024)) + (r122_z % $(1024))), dtype=tl.float32)
    r122_value = r122_value.to(tl.bfloat16)
    r122_value = tl.broadcast_to(r122_value, [$(98304)])
    tl.store($(candidate_vto_copy_15_ptr) + r122_z, r122_value)
    r123_z = tl.arange($(98304))
    r123_value = tl.load($(candidate_vto_copy_15_ptr) + ((((r123_z / $(1024)) % $(96)) * $(1024)) + (r123_z % $(1024))), dtype=tl.bfloat16)
    r123_value = r123_value.to(tl.float32)
    r123_value = tl.broadcast_to(r123_value, [$(98304)])
    tl.store($(candidate_vto_copy_16_ptr) + r123_z, r123_value)
    r124_z = tl.arange($(6144))
    r124_value = tl.load($(input0_ptr) + ((((r124_z / $(64)) % $(96)) * $(64)) + (r124_z % $(64)) + $(12288)), dtype=tl.bfloat16)
    r124_value = r124_value.to(tl.float32)
    r124_value = tl.broadcast_to(r124_value, [$(6144)])
    tl.store($(candidate_vto_copy_17_ptr) + r124_z, r124_value)
    r125_row = tl.arange($(1024))
    r125_col = tl.arange($(64))
    r125_lane = tl.arange($(96))
    r125_left = tl.load($(candidate_vto_copy_16_ptr) + (((r125_lane[None, :] % $(96)) * $(1024)) + (r125_row[:, None] % $(1024))), dtype=tl.float32)
    r125_right = tl.load($(candidate_vto_copy_17_ptr) + (((r125_lane[:, None] % $(96)) * $(64)) + (r125_col[None, :] % $(64))), dtype=tl.float32)
    r125_value = tl.dot(r125_left, r125_right)
    r125_value = r125_value.to(tl.float32)
    tl.store($(candidate_vmm_12_ptr) + ((r125_row[:, None] * $(64)) + r125_col[None, :]), r125_value)
    r126_z = tl.arange($(65536))
    r126_value = (tl.load($(candidate_vadd_15_ptr) + ((((r126_z / $(64)) % $(1024)) * $(64)) + (r126_z % $(64))), dtype=tl.float32) + tl.load($(candidate_vmm_12_ptr) + ((((r126_z / $(64)) % $(1024)) * $(64)) + (r126_z % $(64))), dtype=tl.float32))
    r126_value = r126_value.to(tl.float32)
    r126_value = tl.broadcast_to(r126_value, [$(65536)])
    tl.store($(candidate_vadd_21_ptr) + r126_z, r126_value)
    r127_row = tl.arange($(32))
    r127_col = tl.arange($(1024))
    r127_lane = tl.arange($(64))
    r127_left = tl.load($(input0_ptr) + ((((((r127_row[:, None] * $(64)) + r127_lane[None, :] + $(18432)) / $(64)) % $(320)) * $(64)) + (((r127_row[:, None] * $(64)) + r127_lane[None, :] + $(18432)) % $(64))), dtype=tl.bfloat16)
    r127_right = tl.load($(candidate_vmul_ptr) + (((r127_col[None, :] % $(1024)) * $(64)) + (r127_lane[:, None] % $(64))), dtype=tl.bfloat16)
    r127_value = tl.dot(r127_left, r127_right)
    r127_value = r127_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_13_ptr) + ((r127_row[:, None] * $(1024)) + r127_col[None, :]), r127_value)
    r128_z = tl.arange($(32768))
    r128_value = tl.load($(candidate_vmm_13_ptr) + ((((r128_z / $(1024)) % $(32)) * $(1024)) + (r128_z % $(1024))), dtype=tl.bfloat16)
    r128_value = r128_value.to(tl.float32)
    r128_value = tl.broadcast_to(r128_value, [$(32768)])
    tl.store($(candidate_vto_copy_18_ptr) + r128_z, r128_value)
    r129_z = tl.arange($(32))
    r129_lane = tl.arange($(1024))
    r129_row_values = tl.load($(candidate_vto_copy_18_ptr) + ((r129_z[:, None] * $(1024)) + (r129_lane[None, :] % $(1024))), dtype=tl.float32)
    r129_value = tl.max(r129_row_values, axis=1)
    r129_value = r129_value.to(tl.float32)
    r129_value = tl.broadcast_to(r129_value, [$(32)])
    tl.store($(candidate_vmax_8_value_ptr) + r129_z, r129_value)
    r130_z = tl.arange($(32))
    r130_lane = tl.arange($(1024))
    r130_row_values = tl.load($(candidate_vto_copy_18_ptr) + ((r130_z[:, None] * $(1024)) + (r130_lane[None, :] % $(1024))), dtype=tl.float32)
    r130_value = tl.argmax(r130_row_values, axis=1)
    r130_value = tl.cast(r130_value, tl.int64)
    r130_value = tl.broadcast_to(r130_value, [$(32)])
    tl.store($(candidate_vmax_8_index_ptr) + r130_z, r130_value)
    r131_z = tl.arange($(32768))
    r131_value = (tl.load($(candidate_vto_copy_18_ptr) + ((((r131_z / $(1024)) % $(32)) * $(1024)) + (r131_z % $(1024))), dtype=tl.float32) - tl.load($(candidate_vmax_8_value_ptr) + ((r131_z / $(1024)) % $(32)), dtype=tl.float32))
    r131_value = r131_value.to(tl.float32)
    r131_value = tl.broadcast_to(r131_value, [$(32768)])
    tl.store($(candidate_vsub_24_ptr) + r131_z, r131_value)
    r132_z = tl.arange($(32768))
    r132_value = tl.exp(tl.load($(candidate_vsub_24_ptr) + ((((r132_z / $(1024)) % $(32)) * $(1024)) + (r132_z % $(1024))), dtype=tl.float32))
    r132_value = r132_value.to(tl.float32)
    r132_value = tl.broadcast_to(r132_value, [$(32768)])
    tl.store($(candidate_vexp_10_ptr) + r132_z, r132_value)
    r133_z = tl.arange($(32))
    r133_lane = tl.arange($(1024))
    r133_value = tl.sum(tl.load($(candidate_vexp_10_ptr) + ((r133_z[:, None] * $(1024)) + r133_lane[None, :]), dtype=tl.float32), axis=1)
    r133_value = r133_value.to(tl.float32)
    r133_value = tl.broadcast_to(r133_value, [$(32)])
    tl.store($(candidate_vsum_19_ptr) + r133_z, r133_value)
    r134_z = tl.arange($(32))
    r134_value = tl.log(tl.load($(candidate_vsum_19_ptr) + r134_z, dtype=tl.float32))
    r134_value = r134_value.to(tl.float32)
    r134_value = tl.broadcast_to(r134_value, [$(32)])
    tl.store($(candidate_vlog_7_ptr) + r134_z, r134_value)
    r135_z = tl.arange($(32))
    r135_value = tl.load($(candidate_vto_copy_18_ptr) + ((r135_z * $(1024)) + (tl.load($(parameter1_ptr) + (r135_z + $(288)), dtype=tl.int64) % $(1024))), dtype=tl.float32)
    r135_value = r135_value.to(tl.float32)
    r135_value = tl.broadcast_to(r135_value, [$(32)])
    tl.store($(candidate_vgather_13_ptr) + r135_z, r135_value)
    r136_z = tl.arange($(32))
    r136_value = (tl.load($(candidate_vgather_13_ptr) + r136_z, dtype=tl.float32) - tl.load($(candidate_vmax_8_value_ptr) + r136_z, dtype=tl.float32))
    r136_value = r136_value.to(tl.float32)
    r136_value = tl.broadcast_to(r136_value, [$(32)])
    tl.store($(candidate_vsub_25_ptr) + r136_z, r136_value)
    r137_z = tl.arange($(32))
    r137_value = (tl.load($(candidate_vsub_25_ptr) + r137_z, dtype=tl.float32) - tl.load($(candidate_vlog_7_ptr) + r137_z, dtype=tl.float32))
    r137_value = r137_value.to(tl.float32)
    r137_value = tl.broadcast_to(r137_value, [$(32)])
    tl.store($(candidate_vsub_26_ptr) + r137_z, r137_value)
    r138_z = tl.arange($(32))
    r138_value = -tl.load($(candidate_vsub_26_ptr) + r138_z, dtype=tl.float32)
    r138_value = r138_value.to(tl.float32)
    r138_value = tl.broadcast_to(r138_value, [$(32)])
    tl.store($(candidate_vneg_19_ptr) + r138_z, r138_value)
    r139_z = tl.arange($(32))
    r139_value = (tl.load($(candidate_vneg_19_ptr) + r139_z, dtype=tl.float32) * tl.load($(candidate_vto_copy_ptr) + (r139_z + $(288)), dtype=tl.float32))
    r139_value = r139_value.to(tl.float32)
    r139_value = tl.broadcast_to(r139_value, [$(32)])
    tl.store($(candidate_vmul_20_ptr) + r139_z, r139_value)
    r140_z = tl.arange($(1))
    r140_lane = tl.arange($(32))
    r140_value = tl.sum(tl.load($(candidate_vmul_20_ptr) + r140_lane[None, :], dtype=tl.float32), axis=1)
    r140_value = r140_value.to(tl.float32)
    r140_value = tl.broadcast_to(r140_value, [$(1)])
    tl.store($(candidate_vsum_20_ptr) + r140_z, r140_value)
    r141_z = tl.arange($(1))
    r141_value = (tl.load($(candidate_vsum_20_ptr) + $(0), dtype=tl.float32) * tl.load($(candidate_vto_copy_5_ptr) + $(0), dtype=tl.float32))
    r141_value = r141_value.to(tl.float32)
    r141_value = tl.broadcast_to(r141_value, [$(1)])
    tl.store($(candidate_vmul_21_ptr) + r141_z, r141_value)
    r142_z = tl.arange($(1))
    r142_value = (1.0 * tl.load($(candidate_vto_copy_5_ptr) + $(0), dtype=tl.float32))
    r142_value = r142_value.to(tl.float32)
    r142_value = tl.broadcast_to(r142_value, [$(1)])
    tl.store($(candidate_vmul_22_ptr) + r142_z, r142_value)
    r143_z = tl.arange($(32))
    r143_value = (tl.load($(candidate_vmul_22_ptr) + $(0), dtype=tl.float32) * tl.load($(candidate_vto_copy_ptr) + (r143_z + $(288)), dtype=tl.float32))
    r143_value = r143_value.to(tl.float32)
    r143_value = tl.broadcast_to(r143_value, [$(32)])
    tl.store($(candidate_vmul_23_ptr) + r143_z, r143_value)
    r144_z = tl.arange($(32))
    r144_value = -tl.load($(candidate_vmul_23_ptr) + r144_z, dtype=tl.float32)
    r144_value = r144_value.to(tl.float32)
    r144_value = tl.broadcast_to(r144_value, [$(32)])
    tl.store($(candidate_vneg_20_ptr) + r144_z, r144_value)
    r145_z = tl.arange($(32))
    r145_value = -tl.load($(candidate_vneg_20_ptr) + r145_z, dtype=tl.float32)
    r145_value = r145_value.to(tl.float32)
    r145_value = tl.broadcast_to(r145_value, [$(32)])
    tl.store($(candidate_vneg_21_ptr) + r145_z, r145_value)
    r146_z = tl.arange($(32768))
    r146_value = (tl.load($(candidate_vto_copy_18_ptr) + ((((r146_z / $(1024)) % $(32)) * $(1024)) + (r146_z % $(1024))), dtype=tl.float32) - tl.load($(candidate_vmax_8_value_ptr) + ((r146_z / $(1024)) % $(32)), dtype=tl.float32))
    r146_value = r146_value.to(tl.float32)
    r146_value = tl.broadcast_to(r146_value, [$(32768)])
    tl.store($(candidate_vsub_27_ptr) + r146_z, r146_value)
    r147_z = tl.arange($(32768))
    r147_value = tl.exp(tl.load($(candidate_vsub_27_ptr) + ((((r147_z / $(1024)) % $(32)) * $(1024)) + (r147_z % $(1024))), dtype=tl.float32))
    r147_value = r147_value.to(tl.float32)
    r147_value = tl.broadcast_to(r147_value, [$(32768)])
    tl.store($(candidate_vexp_11_ptr) + r147_z, r147_value)
    r148_z = tl.arange($(32))
    r148_value = (tl.load($(candidate_vneg_21_ptr) + r148_z, dtype=tl.float32) / tl.load($(candidate_vsum_19_ptr) + r148_z, dtype=tl.float32))
    r148_value = r148_value.to(tl.float32)
    r148_value = tl.broadcast_to(r148_value, [$(32)])
    tl.store($(candidate_vdiv_3_ptr) + r148_z, r148_value)
    r149_z = tl.arange($(32768))
    r149_value = (tl.load($(candidate_vdiv_3_ptr) + ((r149_z / $(1024)) % $(32)), dtype=tl.float32) * tl.load($(candidate_vexp_11_ptr) + ((((r149_z / $(1024)) % $(32)) * $(1024)) + (r149_z % $(1024))), dtype=tl.float32))
    r149_value = r149_value.to(tl.float32)
    r149_value = tl.broadcast_to(r149_value, [$(32768)])
    tl.store($(candidate_vmul_24_ptr) + r149_z, r149_value)
    r150_z = tl.arange($(32768))
    r150_value = -tl.load($(candidate_vmul_24_ptr) + ((((r150_z / $(1024)) % $(32)) * $(1024)) + (r150_z % $(1024))), dtype=tl.float32)
    r150_value = r150_value.to(tl.float32)
    r150_value = tl.broadcast_to(r150_value, [$(32768)])
    tl.store($(candidate_vneg_22_ptr) + r150_z, r150_value)
    r151_z = tl.arange($(32))
    r151_lane = tl.arange($(1024))
    r151_value = tl.sum(tl.load($(candidate_vneg_22_ptr) + ((r151_z[:, None] * $(1024)) + r151_lane[None, :]), dtype=tl.float32), axis=1)
    r151_value = r151_value.to(tl.float32)
    r151_value = tl.broadcast_to(r151_value, [$(32)])
    tl.store($(candidate_vsum_21_ptr) + r151_z, r151_value)
    r152_z = tl.arange($(32))
    r152_value = (tl.load($(candidate_vneg_21_ptr) + r152_z, dtype=tl.float32) + tl.load($(candidate_vsum_21_ptr) + r152_z, dtype=tl.float32))
    r152_value = r152_value.to(tl.float32)
    r152_value = tl.broadcast_to(r152_value, [$(32)])
    tl.store($(candidate_vadd_22_ptr) + r152_z, r152_value)
    r153_z = tl.arange($(32768))
    r153_value = -tl.load($(candidate_vneg_22_ptr) + ((((r153_z / $(1024)) % $(32)) * $(1024)) + (r153_z % $(1024))), dtype=tl.float32)
    r153_value = r153_value.to(tl.float32)
    r153_value = tl.broadcast_to(r153_value, [$(32768)])
    tl.store($(candidate_vneg_23_ptr) + r153_z, r153_value)
    r154_z = tl.arange($(32))
    r154_value = tl.broadcast_to(0.0, [$(32)])
    for r154_lane in range($(1)) {
      r154_value = tl.where(tl.full([], $(0), dtype=tl.int64) == tl.cast($(0), tl.int64), (r154_value + tl.load($(candidate_vneg_20_ptr) + ((r154_z + r154_lane) % $(32)), dtype=tl.float32)), r154_value)
    }
    r154_value = r154_value.to(tl.float32)
    r154_value = tl.broadcast_to(r154_value, [$(32)])
    tl.store($(candidate_vscatter_add_3_ptr) + r154_z, r154_value)
    r155_z = tl.arange($(32))
    r155_value = tl.load($(candidate_vneg_23_ptr) + ((r155_z * $(1024)) + (tl.load($(parameter1_ptr) + (r155_z + $(288)), dtype=tl.int64) % $(1024))), dtype=tl.float32)
    r155_value = r155_value.to(tl.float32)
    r155_value = tl.broadcast_to(r155_value, [$(32)])
    tl.store($(candidate_vgather_14_ptr) + r155_z, r155_value)
    r156_z = tl.arange($(32))
    r156_value = (tl.load($(candidate_vscatter_add_3_ptr) + r156_z, dtype=tl.float32) + tl.load($(candidate_vgather_14_ptr) + r156_z, dtype=tl.float32))
    r156_value = r156_value.to(tl.float32)
    r156_value = tl.broadcast_to(r156_value, [$(32)])
    tl.store($(candidate_vadd_23_ptr) + r156_z, r156_value)
    r157_z = tl.arange($(32768))
    r157_value = (tl.load($(candidate_vneg_23_ptr) + ((((r157_z / $(1024)) % $(32)) * $(1024)) + (r157_z % $(1024))), dtype=tl.float32) + 0.0)
    r157_value = r157_value.to(tl.float32)
    r157_value = tl.broadcast_to(r157_value, [$(32768)])
    tl.store($(candidate_vadd_24_ptr) + r157_z, r157_value)
    r158_z = tl.arange($(32768))
    r158_value = tl.broadcast_to(tl.load($(candidate_vadd_24_ptr) + ((((r158_z / $(1024)) % $(32)) * $(1024)) + (r158_z % $(1024))), dtype=tl.float32), [$(32768)])
    for r158_lane in range($(1)) {
      r158_value = tl.where(tl.load($(parameter1_ptr) + (((((r158_z / $(1024)) % $(32)) + r158_lane) % $(32)) + $(288)), dtype=tl.int64) == tl.cast((r158_z % $(1024)), tl.int64), tl.load($(candidate_vadd_23_ptr) + ((r158_z / $(1024)) % $(32)), dtype=tl.float32), r158_value)
    }
    r158_value = r158_value.to(tl.float32)
    r158_value = tl.broadcast_to(r158_value, [$(32768)])
    tl.store($(candidate_vscatter_6_ptr) + r158_z, r158_value)
    r159_z = tl.arange($(32))
    r159_value = tl.load($(candidate_vscatter_6_ptr) + ((r159_z * $(1024)) + (tl.load($(candidate_vmax_8_index_ptr) + r159_z, dtype=tl.int64) % $(1024))), dtype=tl.float32)
    r159_value = r159_value.to(tl.float32)
    r159_value = tl.broadcast_to(r159_value, [$(32)])
    tl.store($(candidate_vgather_15_ptr) + r159_z, r159_value)
    r160_z = tl.arange($(32))
    r160_value = (tl.load($(candidate_vgather_15_ptr) + r160_z, dtype=tl.float32) + tl.load($(candidate_vadd_22_ptr) + r160_z, dtype=tl.float32))
    r160_value = r160_value.to(tl.float32)
    r160_value = tl.broadcast_to(r160_value, [$(32)])
    tl.store($(candidate_vadd_25_ptr) + r160_z, r160_value)
    r161_z = tl.arange($(32768))
    r161_value = (tl.load($(candidate_vscatter_6_ptr) + ((((r161_z / $(1024)) % $(32)) * $(1024)) + (r161_z % $(1024))), dtype=tl.float32) + 0.0)
    r161_value = r161_value.to(tl.float32)
    r161_value = tl.broadcast_to(r161_value, [$(32768)])
    tl.store($(candidate_vadd_26_ptr) + r161_z, r161_value)
    r162_z = tl.arange($(32768))
    r162_value = tl.broadcast_to(tl.load($(candidate_vadd_26_ptr) + ((((r162_z / $(1024)) % $(32)) * $(1024)) + (r162_z % $(1024))), dtype=tl.float32), [$(32768)])
    for r162_lane in range($(1)) {
      r162_value = tl.where(tl.load($(candidate_vmax_8_index_ptr) + ((r162_z / $(1024)) % $(32)), dtype=tl.int64) == tl.cast((r162_z % $(1024)), tl.int64), tl.load($(candidate_vadd_25_ptr) + ((r162_z / $(1024)) % $(32)), dtype=tl.float32), r162_value)
    }
    r162_value = r162_value.to(tl.float32)
    r162_value = tl.broadcast_to(r162_value, [$(32768)])
    tl.store($(candidate_vscatter_7_ptr) + r162_z, r162_value)
    r163_z = tl.arange($(32768))
    r163_value = tl.load($(candidate_vscatter_7_ptr) + ((((r163_z / $(1024)) % $(32)) * $(1024)) + (r163_z % $(1024))), dtype=tl.float32)
    r163_value = r163_value.to(tl.bfloat16)
    r163_value = tl.broadcast_to(r163_value, [$(32768)])
    tl.store($(candidate_vto_copy_19_ptr) + r163_z, r163_value)
    r164_z = tl.arange($(32768))
    r164_value = tl.load($(candidate_vto_copy_19_ptr) + ((((r164_z / $(1024)) % $(32)) * $(1024)) + (r164_z % $(1024))), dtype=tl.bfloat16)
    r164_value = r164_value.to(tl.float32)
    r164_value = tl.broadcast_to(r164_value, [$(32768)])
    tl.store($(candidate_vto_copy_20_ptr) + r164_z, r164_value)
    r165_z = tl.arange($(2048))
    r165_value = tl.load($(input0_ptr) + ((((r165_z / $(64)) % $(32)) * $(64)) + (r165_z % $(64)) + $(18432)), dtype=tl.bfloat16)
    r165_value = r165_value.to(tl.float32)
    r165_value = tl.broadcast_to(r165_value, [$(2048)])
    tl.store($(candidate_vto_copy_21_ptr) + r165_z, r165_value)
    r166_row = tl.arange($(1024))
    r166_col = tl.arange($(64))
    r166_lane = tl.arange($(32))
    r166_left = tl.load($(candidate_vto_copy_20_ptr) + (((r166_lane[None, :] % $(32)) * $(1024)) + (r166_row[:, None] % $(1024))), dtype=tl.float32)
    r166_right = tl.load($(candidate_vto_copy_21_ptr) + (((r166_lane[:, None] % $(32)) * $(64)) + (r166_col[None, :] % $(64))), dtype=tl.float32)
    r166_value = tl.dot(r166_left, r166_right)
    r166_value = r166_value.to(tl.float32)
    tl.store($(candidate_vmm_15_ptr) + ((r166_row[:, None] * $(64)) + r166_col[None, :]), r166_value)
    r167_z = tl.arange($(65536))
    r167_value = (tl.load($(candidate_vadd_21_ptr) + ((((r167_z / $(64)) % $(1024)) * $(64)) + (r167_z % $(64))), dtype=tl.float32) + tl.load($(candidate_vmm_15_ptr) + ((((r167_z / $(64)) % $(1024)) * $(64)) + (r167_z % $(64))), dtype=tl.float32))
    r167_value = r167_value.to(tl.float32)
    r167_value = tl.broadcast_to(r167_value, [$(65536)])
    tl.store($(candidate_vadd_27_ptr) + r167_z, r167_value)
    r168_z = tl.arange($(65536))
    r168_value = tl.load($(candidate_vadd_27_ptr) + ((((r168_z / $(64)) % $(1024)) * $(64)) + (r168_z % $(64))), dtype=tl.float32)
    r168_value = r168_value.to(tl.bfloat16)
    r168_value = tl.broadcast_to(r168_value, [$(65536)])
    tl.store($(candidate_vto_copy_22_ptr) + r168_z, r168_value)
    z = tl.arange($(65536))
    value = tl.load($(candidate_vto_copy_22_ptr) + ((((z / $(64)) % $(1024)) * $(64)) + (z % $(64))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(65536)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_chunked_lm_loss_wgrad
