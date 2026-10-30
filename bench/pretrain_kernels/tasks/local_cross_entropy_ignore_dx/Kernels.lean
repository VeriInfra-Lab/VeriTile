import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_cross_entropy_ignore_dx
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
  let input1_ptr : RegionName := "input1"
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
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  let vwhere_3_ptr : RegionName := "vwhere_3"
  let vneg_1_ptr : RegionName := "vneg_1"
  let vneg_2_ptr : RegionName := "vneg_2"
  let vsub_3_ptr : RegionName := "vsub_3"
  let vexp_1_ptr : RegionName := "vexp_1"
  let vdiv_ptr : RegionName := "vdiv"
  let vmul_ptr : RegionName := "vmul"
  let vneg_3_ptr : RegionName := "vneg_3"
  let vsum_2_ptr : RegionName := "vsum_2"
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
  let vto_copy_2_ptr : RegionName := "vto_copy_2"
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
    r9_value = tl.load($(input1_ptr) + ((((r9_z / $(2)) % $(128)) * $(2)) + (r9_z % $(2))), dtype=tl.bfloat16)
    r9_value = r9_value.to(tl.float32)
    r9_value = tl.broadcast_to(r9_value, [$(256)])
    tl.store($(vto_copy_1_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(256))
    r10_value = tl.where(tl.load($(vge_ptr) + ((((r10_z / $(2)) % $(128)) * $(2)) + (r10_z % $(2))), dtype=tl.int1), tl.load($(vto_copy_1_ptr) + ((((r10_z / $(2)) % $(128)) * $(2)) + (r10_z % $(2))), dtype=tl.float32), 0.0)
    r10_value = r10_value.to(tl.float32)
    r10_value = tl.broadcast_to(r10_value, [$(256)])
    tl.store($(vwhere_3_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(256))
    r11_value = -tl.load($(vwhere_3_ptr) + ((((r11_z / $(2)) % $(128)) * $(2)) + (r11_z % $(2))), dtype=tl.float32)
    r11_value = r11_value.to(tl.float32)
    r11_value = tl.broadcast_to(r11_value, [$(256)])
    tl.store($(vneg_1_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(256))
    r12_value = -tl.load($(vneg_1_ptr) + ((((r12_z / $(2)) % $(128)) * $(2)) + (r12_z % $(2))), dtype=tl.float32)
    r12_value = r12_value.to(tl.float32)
    r12_value = tl.broadcast_to(r12_value, [$(256)])
    tl.store($(vneg_2_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(262144))
    r13_value = (tl.load($(vto_copy_ptr) + ((((r13_z / $(2048)) % $(128)) * $(2048)) + (((r13_z / $(1024)) % $(2)) * $(1024)) + (r13_z % $(1024))), dtype=tl.float32) - tl.load($(vmax_1_value_ptr) + ((((r13_z / $(2048)) % $(128)) * $(2)) + ((r13_z / $(1024)) % $(2))), dtype=tl.float32))
    r13_value = r13_value.to(tl.float32)
    r13_value = tl.broadcast_to(r13_value, [$(262144)])
    tl.store($(vsub_3_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(262144))
    r14_value = tl.exp(tl.load($(vsub_3_ptr) + ((((r14_z / $(2048)) % $(128)) * $(2048)) + (((r14_z / $(1024)) % $(2)) * $(1024)) + (r14_z % $(1024))), dtype=tl.float32))
    r14_value = r14_value.to(tl.float32)
    r14_value = tl.broadcast_to(r14_value, [$(262144)])
    tl.store($(vexp_1_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(256))
    r15_value = (tl.load($(vneg_2_ptr) + ((((r15_z / $(2)) % $(128)) * $(2)) + (r15_z % $(2))), dtype=tl.float32) / tl.load($(vsum_1_ptr) + ((((r15_z / $(2)) % $(128)) * $(2)) + (r15_z % $(2))), dtype=tl.float32))
    r15_value = r15_value.to(tl.float32)
    r15_value = tl.broadcast_to(r15_value, [$(256)])
    tl.store($(vdiv_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(262144))
    r16_value = (tl.load($(vdiv_ptr) + ((((r16_z / $(2048)) % $(128)) * $(2)) + ((r16_z / $(1024)) % $(2))), dtype=tl.float32) * tl.load($(vexp_1_ptr) + ((((r16_z / $(2048)) % $(128)) * $(2048)) + (((r16_z / $(1024)) % $(2)) * $(1024)) + (r16_z % $(1024))), dtype=tl.float32))
    r16_value = r16_value.to(tl.float32)
    r16_value = tl.broadcast_to(r16_value, [$(262144)])
    tl.store($(vmul_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(262144))
    r17_value = -tl.load($(vmul_ptr) + ((((r17_z / $(2048)) % $(128)) * $(2048)) + (((r17_z / $(1024)) % $(2)) * $(1024)) + (r17_z % $(1024))), dtype=tl.float32)
    r17_value = r17_value.to(tl.float32)
    r17_value = tl.broadcast_to(r17_value, [$(262144)])
    tl.store($(vneg_3_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(256))
    r18_lane = tl.arange($(1024))
    r18_value = tl.sum(tl.load($(vneg_3_ptr) + ((((r18_z[:, None] / $(2)) % $(128)) * $(2048)) + ((r18_z[:, None] % $(2)) * $(1024)) + r18_lane[None, :]), dtype=tl.float32), axis=1)
    r18_value = r18_value.to(tl.float32)
    r18_value = tl.broadcast_to(r18_value, [$(256)])
    tl.store($(vsum_2_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(256))
    r19_value = (tl.load($(vneg_2_ptr) + ((((r19_z / $(2)) % $(128)) * $(2)) + (r19_z % $(2))), dtype=tl.float32) + tl.load($(vsum_2_ptr) + ((((r19_z / $(2)) % $(128)) * $(2)) + (r19_z % $(2))), dtype=tl.float32))
    r19_value = r19_value.to(tl.float32)
    r19_value = tl.broadcast_to(r19_value, [$(256)])
    tl.store($(vadd_ptr) + r19_z, r19_value)
    r20_z = tl.arange($(262144))
    r20_value = -tl.load($(vneg_3_ptr) + ((((r20_z / $(2048)) % $(128)) * $(2048)) + (((r20_z / $(1024)) % $(2)) * $(1024)) + (r20_z % $(1024))), dtype=tl.float32)
    r20_value = r20_value.to(tl.float32)
    r20_value = tl.broadcast_to(r20_value, [$(262144)])
    tl.store($(vneg_4_ptr) + r20_z, r20_value)
    r21_z = tl.arange($(256))
    r21_value = tl.broadcast_to(0.0, [$(256)])
    for r21_lane in range($(1)) {
      r21_value = tl.where(tl.full([], $(0), dtype=tl.int64) == tl.cast($(0), tl.int64), (r21_value + tl.load($(vneg_1_ptr) + ((((((((r21_z / $(2)) % $(128)) * $(2)) + (r21_z % $(2)) + r21_lane) / $(2)) % $(128)) * $(2)) + (((((r21_z / $(2)) % $(128)) * $(2)) + (r21_z % $(2)) + r21_lane) % $(2))), dtype=tl.float32)), r21_value)
    }
    r21_value = r21_value.to(tl.float32)
    r21_value = tl.broadcast_to(r21_value, [$(256)])
    tl.store($(vscatter_add_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(256))
    r22_value = tl.load($(vneg_4_ptr) + (((((r22_z / $(2)) % $(128)) * $(2048)) + ((r22_z % $(2)) * $(1024))) + (tl.load($(vwhere_1_ptr) + ((((r22_z / $(2)) % $(128)) * $(2)) + (r22_z % $(2))), dtype=tl.int64) % $(1024))), dtype=tl.float32)
    r22_value = r22_value.to(tl.float32)
    r22_value = tl.broadcast_to(r22_value, [$(256)])
    tl.store($(vgather_1_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(256))
    r23_value = (tl.load($(vscatter_add_ptr) + ((((r23_z / $(2)) % $(128)) * $(2)) + (r23_z % $(2))), dtype=tl.float32) + tl.load($(vgather_1_ptr) + ((((r23_z / $(2)) % $(128)) * $(2)) + (r23_z % $(2))), dtype=tl.float32))
    r23_value = r23_value.to(tl.float32)
    r23_value = tl.broadcast_to(r23_value, [$(256)])
    tl.store($(vadd_1_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(262144))
    r24_value = (tl.load($(vneg_4_ptr) + ((((r24_z / $(2048)) % $(128)) * $(2048)) + (((r24_z / $(1024)) % $(2)) * $(1024)) + (r24_z % $(1024))), dtype=tl.float32) + 0.0)
    r24_value = r24_value.to(tl.float32)
    r24_value = tl.broadcast_to(r24_value, [$(262144)])
    tl.store($(vadd_2_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(262144))
    r25_value = tl.broadcast_to(tl.load($(vadd_2_ptr) + ((((r25_z / $(2048)) % $(128)) * $(2048)) + (((r25_z / $(1024)) % $(2)) * $(1024)) + (r25_z % $(1024))), dtype=tl.float32), [$(262144)])
    for r25_lane in range($(1)) {
      r25_value = tl.where(tl.load($(vwhere_1_ptr) + ((((((((r25_z / $(2048)) % $(128)) * $(2)) + ((r25_z / $(1024)) % $(2)) + r25_lane) / $(2)) % $(128)) * $(2)) + (((((r25_z / $(2048)) % $(128)) * $(2)) + ((r25_z / $(1024)) % $(2)) + r25_lane) % $(2))), dtype=tl.int64) == tl.cast((r25_z % $(1024)), tl.int64), tl.load($(vadd_1_ptr) + ((((r25_z / $(2048)) % $(128)) * $(2)) + ((r25_z / $(1024)) % $(2))), dtype=tl.float32), r25_value)
    }
    r25_value = r25_value.to(tl.float32)
    r25_value = tl.broadcast_to(r25_value, [$(262144)])
    tl.store($(vscatter_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(256))
    r26_value = tl.load($(vscatter_ptr) + (((((r26_z / $(2)) % $(128)) * $(2048)) + ((r26_z % $(2)) * $(1024))) + (tl.load($(vmax_1_index_ptr) + ((((r26_z / $(2)) % $(128)) * $(2)) + (r26_z % $(2))), dtype=tl.int64) % $(1024))), dtype=tl.float32)
    r26_value = r26_value.to(tl.float32)
    r26_value = tl.broadcast_to(r26_value, [$(256)])
    tl.store($(vgather_2_ptr) + r26_z, r26_value)
    r27_z = tl.arange($(256))
    r27_value = (tl.load($(vgather_2_ptr) + ((((r27_z / $(2)) % $(128)) * $(2)) + (r27_z % $(2))), dtype=tl.float32) + tl.load($(vadd_ptr) + ((((r27_z / $(2)) % $(128)) * $(2)) + (r27_z % $(2))), dtype=tl.float32))
    r27_value = r27_value.to(tl.float32)
    r27_value = tl.broadcast_to(r27_value, [$(256)])
    tl.store($(vadd_3_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(262144))
    r28_value = (tl.load($(vscatter_ptr) + ((((r28_z / $(2048)) % $(128)) * $(2048)) + (((r28_z / $(1024)) % $(2)) * $(1024)) + (r28_z % $(1024))), dtype=tl.float32) + 0.0)
    r28_value = r28_value.to(tl.float32)
    r28_value = tl.broadcast_to(r28_value, [$(262144)])
    tl.store($(vadd_4_ptr) + r28_z, r28_value)
    r29_z = tl.arange($(262144))
    r29_value = tl.broadcast_to(tl.load($(vadd_4_ptr) + ((((r29_z / $(2048)) % $(128)) * $(2048)) + (((r29_z / $(1024)) % $(2)) * $(1024)) + (r29_z % $(1024))), dtype=tl.float32), [$(262144)])
    for r29_lane in range($(1)) {
      r29_value = tl.where(tl.load($(vmax_1_index_ptr) + ((((r29_z / $(2048)) % $(128)) * $(2)) + ((r29_z / $(1024)) % $(2))), dtype=tl.int64) == tl.cast((r29_z % $(1024)), tl.int64), tl.load($(vadd_3_ptr) + ((((r29_z / $(2048)) % $(128)) * $(2)) + ((r29_z / $(1024)) % $(2))), dtype=tl.float32), r29_value)
    }
    r29_value = r29_value.to(tl.float32)
    r29_value = tl.broadcast_to(r29_value, [$(262144)])
    tl.store($(vscatter_1_ptr) + r29_z, r29_value)
    r30_z = tl.arange($(262144))
    r30_value = tl.load($(vscatter_1_ptr) + ((((r30_z / $(2048)) % $(128)) * $(2048)) + (((r30_z / $(1024)) % $(2)) * $(1024)) + (r30_z % $(1024))), dtype=tl.float32)
    r30_value = r30_value.to(tl.bfloat16)
    r30_value = tl.broadcast_to(r30_value, [$(262144)])
    tl.store($(vto_copy_2_ptr) + r30_z, r30_value)
    z = tl.arange($(262144))
    value = tl.load($(vto_copy_2_ptr) + ((((z / $(2048)) % $(128)) * $(2048)) + (((z / $(1024)) % $(2)) * $(1024)) + (z % $(1024))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(262144)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let parameter0_ptr : RegionName := "parameter0"
  let parameter1_ptr : RegionName := "parameter1"
  let input0_ptr : RegionName := "input0"
  let input1_ptr : RegionName := "input1"
  let out_ptr : RegionName := "out"
  let vge_ptr : RegionName := "vge"
  let vwhere_ptr : RegionName := "vwhere"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vmax_1_value_ptr : RegionName := "vmax_1_value"
  let vmax_1_index_ptr : RegionName := "vmax_1_index"
  let vsub_ptr : RegionName := "vsub"
  let vexp_ptr : RegionName := "vexp"
  let vsum_1_ptr : RegionName := "vsum_1"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  let vwhere_3_ptr : RegionName := "vwhere_3"
  let vsub_3_ptr : RegionName := "vsub_3"
  let vexp_1_ptr : RegionName := "vexp_1"
  let vdiv_ptr : RegionName := "vdiv"
  let vmul_ptr : RegionName := "vmul"
  let vto_copy_2_ptr : RegionName := "vto_copy_2"
  let candidate_vge_ptr : RegionName := "candidate_vge"
  let candidate_vwhere_ptr : RegionName := "candidate_vwhere"
  let candidate_vto_copy_ptr : RegionName := "candidate_vto_copy"
  let candidate_vmax_1_value_ptr : RegionName := "candidate_vmax_1_value"
  let candidate_vmax_1_index_ptr : RegionName := "candidate_vmax_1_index"
  let candidate_vsub_ptr : RegionName := "candidate_vsub"
  let candidate_vexp_ptr : RegionName := "candidate_vexp"
  let candidate_vsum_1_ptr : RegionName := "candidate_vsum_1"
  let candidate_vto_copy_1_ptr : RegionName := "candidate_vto_copy_1"
  let candidate_vto_copy_2_ptr : RegionName := "candidate_vto_copy_2"
  let candidate_vsub_3_ptr : RegionName := "candidate_vsub_3"
  let candidate_vexp_1_ptr : RegionName := "candidate_vexp_1"
  let candidate_vdiv_ptr : RegionName := "candidate_vdiv"
  let candidate_veq_ptr : RegionName := "candidate_veq"
  let candidate_vsub_4_ptr : RegionName := "candidate_vsub_4"
  let candidate_vwhere_3_ptr : RegionName := "candidate_vwhere_3"
  let candidate_vne_1_ptr : RegionName := "candidate_vne_1"
  let candidate_vmul_ptr : RegionName := "candidate_vmul"
  let candidate_vwhere_4_ptr : RegionName := "candidate_vwhere_4"
  let candidate_vto_copy_3_ptr : RegionName := "candidate_vto_copy_3"
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
    r3_lane = tl.arange($(1024))
    r3_row_values = tl.load($(candidate_vto_copy_ptr) + ((((r3_z[:, None] / $(2)) % $(128)) * $(2048)) + ((r3_z[:, None] % $(2)) * $(1024)) + (r3_lane[None, :] % $(1024))), dtype=tl.float32)
    r3_value = tl.max(r3_row_values, axis=1)
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(256)])
    tl.store($(candidate_vmax_1_value_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(256))
    r4_lane = tl.arange($(1024))
    r4_row_values = tl.load($(candidate_vto_copy_ptr) + ((((r4_z[:, None] / $(2)) % $(128)) * $(2048)) + ((r4_z[:, None] % $(2)) * $(1024)) + (r4_lane[None, :] % $(1024))), dtype=tl.float32)
    r4_value = tl.argmax(r4_row_values, axis=1)
    r4_value = tl.cast(r4_value, tl.int64)
    r4_value = tl.broadcast_to(r4_value, [$(256)])
    tl.store($(candidate_vmax_1_index_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(262144))
    r5_value = (tl.load($(candidate_vto_copy_ptr) + ((((r5_z / $(2048)) % $(128)) * $(2048)) + (((r5_z / $(1024)) % $(2)) * $(1024)) + (r5_z % $(1024))), dtype=tl.float32) - tl.load($(candidate_vmax_1_value_ptr) + ((((r5_z / $(2048)) % $(128)) * $(2)) + ((r5_z / $(1024)) % $(2))), dtype=tl.float32))
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(262144)])
    tl.store($(candidate_vsub_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(262144))
    r6_value = tl.exp(tl.load($(candidate_vsub_ptr) + ((((r6_z / $(2048)) % $(128)) * $(2048)) + (((r6_z / $(1024)) % $(2)) * $(1024)) + (r6_z % $(1024))), dtype=tl.float32))
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(262144)])
    tl.store($(candidate_vexp_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(256))
    r7_lane = tl.arange($(1024))
    r7_value = tl.sum(tl.load($(candidate_vexp_ptr) + ((((r7_z[:, None] / $(2)) % $(128)) * $(2048)) + ((r7_z[:, None] % $(2)) * $(1024)) + r7_lane[None, :]), dtype=tl.float32), axis=1)
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(256)])
    tl.store($(candidate_vsum_1_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(256))
    r8_value = tl.load($(input1_ptr) + ((((r8_z / $(2)) % $(128)) * $(2)) + (r8_z % $(2))), dtype=tl.bfloat16)
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(256)])
    tl.store($(candidate_vto_copy_1_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(262144))
    r9_value = tl.load($(input0_ptr) + ((((r9_z / $(2048)) % $(128)) * $(2048)) + (((r9_z / $(1024)) % $(2)) * $(1024)) + (r9_z % $(1024))), dtype=tl.bfloat16)
    r9_value = r9_value.to(tl.float32)
    r9_value = tl.broadcast_to(r9_value, [$(262144)])
    tl.store($(candidate_vto_copy_2_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(262144))
    r10_value = (tl.load($(candidate_vto_copy_2_ptr) + ((((r10_z / $(2048)) % $(128)) * $(2048)) + (((r10_z / $(1024)) % $(2)) * $(1024)) + (r10_z % $(1024))), dtype=tl.float32) - tl.load($(candidate_vmax_1_value_ptr) + ((((r10_z / $(2048)) % $(128)) * $(2)) + ((r10_z / $(1024)) % $(2))), dtype=tl.float32))
    r10_value = r10_value.to(tl.float32)
    r10_value = tl.broadcast_to(r10_value, [$(262144)])
    tl.store($(candidate_vsub_3_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(262144))
    r11_value = tl.exp(tl.load($(candidate_vsub_3_ptr) + ((((r11_z / $(2048)) % $(128)) * $(2048)) + (((r11_z / $(1024)) % $(2)) * $(1024)) + (r11_z % $(1024))), dtype=tl.float32))
    r11_value = r11_value.to(tl.float32)
    r11_value = tl.broadcast_to(r11_value, [$(262144)])
    tl.store($(candidate_vexp_1_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(262144))
    r12_value = (tl.load($(candidate_vexp_1_ptr) + ((((r12_z / $(2048)) % $(128)) * $(2048)) + (((r12_z / $(1024)) % $(2)) * $(1024)) + (r12_z % $(1024))), dtype=tl.float32) / tl.load($(candidate_vsum_1_ptr) + ((((r12_z / $(2048)) % $(128)) * $(2)) + ((r12_z / $(1024)) % $(2))), dtype=tl.float32))
    r12_value = r12_value.to(tl.float32)
    r12_value = tl.broadcast_to(r12_value, [$(262144)])
    tl.store($(candidate_vdiv_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(262144))
    r13_value = (tl.cast((r13_z % $(1024)), tl.int64) == tl.load($(candidate_vwhere_ptr) + ((((r13_z / $(2048)) % $(128)) * $(2)) + ((r13_z / $(1024)) % $(2))), dtype=tl.int64))
    r13_value = tl.broadcast_to(r13_value, [$(262144)])
    tl.store($(candidate_veq_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(262144))
    r14_value = (tl.load($(candidate_vdiv_ptr) + ((((r14_z / $(2048)) % $(128)) * $(2048)) + (((r14_z / $(1024)) % $(2)) * $(1024)) + (r14_z % $(1024))), dtype=tl.float32) - 1.0)
    r14_value = r14_value.to(tl.float32)
    r14_value = tl.broadcast_to(r14_value, [$(262144)])
    tl.store($(candidate_vsub_4_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(262144))
    r15_value = tl.where(tl.load($(candidate_veq_ptr) + ((((r15_z / $(2048)) % $(128)) * $(2048)) + (((r15_z / $(1024)) % $(2)) * $(1024)) + (r15_z % $(1024))), dtype=tl.int1), tl.load($(candidate_vsub_4_ptr) + ((((r15_z / $(2048)) % $(128)) * $(2048)) + (((r15_z / $(1024)) % $(2)) * $(1024)) + (r15_z % $(1024))), dtype=tl.float32), tl.load($(candidate_vdiv_ptr) + ((((r15_z / $(2048)) % $(128)) * $(2048)) + (((r15_z / $(1024)) % $(2)) * $(1024)) + (r15_z % $(1024))), dtype=tl.float32))
    r15_value = r15_value.to(tl.float32)
    r15_value = tl.broadcast_to(r15_value, [$(262144)])
    tl.store($(candidate_vwhere_3_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(256))
    r16_value = (tl.load($(candidate_vwhere_ptr) + ((((r16_z / $(2)) % $(128)) * $(2)) + (r16_z % $(2))), dtype=tl.int64) != (-tl.full([], $(100), dtype=tl.int64)))
    r16_value = tl.broadcast_to(r16_value, [$(256)])
    tl.store($(candidate_vne_1_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(262144))
    r17_value = (tl.load($(candidate_vwhere_3_ptr) + ((((r17_z / $(2048)) % $(128)) * $(2048)) + (((r17_z / $(1024)) % $(2)) * $(1024)) + (r17_z % $(1024))), dtype=tl.float32) * tl.load($(candidate_vto_copy_1_ptr) + ((((r17_z / $(2048)) % $(128)) * $(2)) + ((r17_z / $(1024)) % $(2))), dtype=tl.float32))
    r17_value = r17_value.to(tl.float32)
    r17_value = tl.broadcast_to(r17_value, [$(262144)])
    tl.store($(candidate_vmul_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(262144))
    r18_value = tl.where(tl.load($(candidate_vne_1_ptr) + ((((r18_z / $(2048)) % $(128)) * $(2)) + ((r18_z / $(1024)) % $(2))), dtype=tl.int1), tl.load($(candidate_vmul_ptr) + ((((r18_z / $(2048)) % $(128)) * $(2048)) + (((r18_z / $(1024)) % $(2)) * $(1024)) + (r18_z % $(1024))), dtype=tl.float32), 0.0)
    r18_value = r18_value.to(tl.float32)
    r18_value = tl.broadcast_to(r18_value, [$(262144)])
    tl.store($(candidate_vwhere_4_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(262144))
    r19_value = tl.load($(candidate_vwhere_4_ptr) + ((((r19_z / $(2048)) % $(128)) * $(2048)) + (((r19_z / $(1024)) % $(2)) * $(1024)) + (r19_z % $(1024))), dtype=tl.float32)
    r19_value = r19_value.to(tl.bfloat16)
    r19_value = tl.broadcast_to(r19_value, [$(262144)])
    tl.store($(candidate_vto_copy_3_ptr) + r19_z, r19_value)
    z = tl.arange($(262144))
    value = tl.load($(candidate_vto_copy_3_ptr) + ((((z / $(2048)) % $(128)) * $(2048)) + (((z / $(1024)) % $(2)) * $(1024)) + (z % $(1024))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(262144)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_cross_entropy_ignore_dx
