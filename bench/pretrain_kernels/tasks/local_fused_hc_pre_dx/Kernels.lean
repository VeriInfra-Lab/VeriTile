import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_fused_hc_pre_dx
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Kernels

def original : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let input1_ptr : RegionName := "input1"
  let input2_ptr : RegionName := "input2"
  let input3_ptr : RegionName := "input3"
  let input4_ptr : RegionName := "input4"
  let input5_ptr : RegionName := "input5"
  let out_ptr : RegionName := "out"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vpow_1_ptr : RegionName := "vpow_1"
  let vmean_ptr : RegionName := "vmean"
  let vadd_ptr : RegionName := "vadd"
  let vrsqrt_ptr : RegionName := "vrsqrt"
  let vmm_ptr : RegionName := "vmm"
  let vmul_ptr : RegionName := "vmul"
  let vmul_1_ptr : RegionName := "vmul_1"
  let vadd_1_ptr : RegionName := "vadd_1"
  let vsigmoid_ptr : RegionName := "vsigmoid"
  let vadd_2_ptr : RegionName := "vadd_2"
  let vmul_2_ptr : RegionName := "vmul_2"
  let vadd_3_ptr : RegionName := "vadd_3"
  let vsigmoid_1_ptr : RegionName := "vsigmoid_1"
  let vto_copy_2_ptr : RegionName := "vto_copy_2"
  let vmul_5_ptr : RegionName := "vmul_5"
  let vmul_6_ptr : RegionName := "vmul_6"
  let vsum_2_ptr : RegionName := "vsum_2"
  let vmul_7_ptr : RegionName := "vmul_7"
  let vsigmoid_backward_ptr : RegionName := "vsigmoid_backward"
  let vmul_8_ptr : RegionName := "vmul_8"
  let vslice_backward_ptr : RegionName := "vslice_backward"
  let vsigmoid_backward_1_ptr : RegionName := "vsigmoid_backward_1"
  let vmul_9_ptr : RegionName := "vmul_9"
  let vslice_backward_1_ptr : RegionName := "vslice_backward_1"
  let vadd_5_ptr : RegionName := "vadd_5"
  let vmul_10_ptr : RegionName := "vmul_10"
  let vmul_11_ptr : RegionName := "vmul_11"
  let vsum_3_ptr : RegionName := "vsum_3"
  let vmm_1_ptr : RegionName := "vmm_1"
  let vadd_6_ptr : RegionName := "vadd_6"
  let vpow_2_ptr : RegionName := "vpow_2"
  let vmul_12_ptr : RegionName := "vmul_12"
  let vmul_13_ptr : RegionName := "vmul_13"
  let vdiv_ptr : RegionName := "vdiv"
  let vpow_3_ptr : RegionName := "vpow_3"
  let vmul_14_ptr : RegionName := "vmul_14"
  let vmul_15_ptr : RegionName := "vmul_15"
  let vadd_7_ptr : RegionName := "vadd_7"
  let vto_copy_3_ptr : RegionName := "vto_copy_3"
  triton {
    r0_z = tl.arange($(67108864))
    r0_value = tl.load($(input0_ptr) + ((((r0_z / $(33554432)) % $(2)) * $(33554432)) + (((r0_z / $(8192)) % $(4096)) * $(8192)) + (((((((r0_z / $(33554432)) % $(2)) * $(33554432)) + (((r0_z / $(8192)) % $(4096)) * $(8192)) + (r0_z % $(8192))) / $(2048)) % $(4)) * $(2048)) + (((((r0_z / $(33554432)) % $(2)) * $(33554432)) + (((r0_z / $(8192)) % $(4096)) * $(8192)) + (r0_z % $(8192))) % $(2048))), dtype=tl.bfloat16)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(67108864)])
    tl.store($(vto_copy_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(67108864))
    r1_value = ((tl.load($(vto_copy_ptr) + ((((r1_z / $(33554432)) % $(2)) * $(33554432)) + (((r1_z / $(8192)) % $(4096)) * $(8192)) + (r1_z % $(8192))), dtype=tl.float32)) * (tl.load($(vto_copy_ptr) + ((((r1_z / $(33554432)) % $(2)) * $(33554432)) + (((r1_z / $(8192)) % $(4096)) * $(8192)) + (r1_z % $(8192))), dtype=tl.float32)))
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(67108864)])
    tl.store($(vpow_1_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(8192))
    r2_lane = tl.arange($(8192))
    r2_value = (tl.sum(tl.load($(vpow_1_ptr) + ((((r2_z[:, None] / $(4096)) % $(2)) * $(33554432)) + ((r2_z[:, None] % $(4096)) * $(8192)) + r2_lane[None, :]), dtype=tl.float32), axis=1) / 8192.0)
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(8192)])
    tl.store($(vmean_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(8192))
    r3_value = (tl.load($(vmean_ptr) + ((((r3_z / $(4096)) % $(2)) * $(4096)) + (r3_z % $(4096))), dtype=tl.float32) + 0.000001)
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(8192)])
    tl.store($(vadd_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(8192))
    r4_value = tl.rsqrt(tl.load($(vadd_ptr) + ((((r4_z / $(4096)) % $(2)) * $(4096)) + (r4_z % $(4096))), dtype=tl.float32))
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(8192)])
    tl.store($(vrsqrt_ptr) + r4_z, r4_value)
    r5_row = tl.arange($(8192))
    r5_col = tl.arange($(8))
    r5_lane = tl.arange($(8192))
    r5_left = tl.load($(vto_copy_ptr) + ((((((r5_row[:, None] * $(8192)) + r5_lane[None, :]) / $(33554432)) % $(2)) * $(33554432)) + (((((r5_row[:, None] * $(8192)) + r5_lane[None, :]) / $(8192)) % $(4096)) * $(8192)) + (((r5_row[:, None] * $(8192)) + r5_lane[None, :]) % $(8192))), dtype=tl.float32)
    r5_right = tl.load($(input1_ptr) + (((r5_col[None, :] % $(8)) * $(8192)) + (r5_lane[:, None] % $(8192))), dtype=tl.float32)
    r5_value = tl.dot(r5_left, r5_right)
    r5_value = r5_value.to(tl.float32)
    tl.store($(vmm_ptr) + ((r5_row[:, None] * $(8)) + r5_col[None, :]), r5_value)
    r6_z = tl.arange($(65536))
    r6_value = (tl.load($(vmm_ptr) + ((((r6_z / $(32768)) % $(2)) * $(32768)) + (((r6_z / $(8)) % $(4096)) * $(8)) + (r6_z % $(8))), dtype=tl.float32) * tl.load($(vrsqrt_ptr) + ((((r6_z / $(32768)) % $(2)) * $(4096)) + ((r6_z / $(8)) % $(4096))), dtype=tl.float32))
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(65536)])
    tl.store($(vmul_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(32768))
    r7_value = (tl.load($(vmul_ptr) + ((((r7_z / $(16384)) % $(2)) * $(32768)) + (((r7_z / $(4)) % $(4096)) * $(8)) + (r7_z % $(4))), dtype=tl.float32) * tl.load($(input2_ptr) + $(0), dtype=tl.float32))
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(32768)])
    tl.store($(vmul_1_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(32768))
    r8_value = (tl.load($(vmul_1_ptr) + ((((r8_z / $(16384)) % $(2)) * $(16384)) + (((r8_z / $(4)) % $(4096)) * $(4)) + (r8_z % $(4))), dtype=tl.float32) + tl.load($(input3_ptr) + (r8_z % $(4)), dtype=tl.float32))
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(32768)])
    tl.store($(vadd_1_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(32768))
    r9_value = tl.sigmoid(tl.load($(vadd_1_ptr) + ((((r9_z / $(16384)) % $(2)) * $(16384)) + (((r9_z / $(4)) % $(4096)) * $(4)) + (r9_z % $(4))), dtype=tl.float32))
    r9_value = r9_value.to(tl.float32)
    r9_value = tl.broadcast_to(r9_value, [$(32768)])
    tl.store($(vsigmoid_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(32768))
    r10_value = (tl.load($(vsigmoid_ptr) + ((((r10_z / $(16384)) % $(2)) * $(16384)) + (((r10_z / $(4)) % $(4096)) * $(4)) + (r10_z % $(4))), dtype=tl.float32) + 0.000001)
    r10_value = r10_value.to(tl.float32)
    r10_value = tl.broadcast_to(r10_value, [$(32768)])
    tl.store($(vadd_2_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(32768))
    r11_value = (tl.load($(vmul_ptr) + ((((r11_z / $(16384)) % $(2)) * $(32768)) + (((r11_z / $(4)) % $(4096)) * $(8)) + (r11_z % $(4)) + $(4)), dtype=tl.float32) * tl.load($(input2_ptr) + $(1), dtype=tl.float32))
    r11_value = r11_value.to(tl.float32)
    r11_value = tl.broadcast_to(r11_value, [$(32768)])
    tl.store($(vmul_2_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(32768))
    r12_value = (tl.load($(vmul_2_ptr) + ((((r12_z / $(16384)) % $(2)) * $(16384)) + (((r12_z / $(4)) % $(4096)) * $(4)) + (r12_z % $(4))), dtype=tl.float32) + tl.load($(input3_ptr) + ((r12_z % $(4)) + $(4)), dtype=tl.float32))
    r12_value = r12_value.to(tl.float32)
    r12_value = tl.broadcast_to(r12_value, [$(32768)])
    tl.store($(vadd_3_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(32768))
    r13_value = tl.sigmoid(tl.load($(vadd_3_ptr) + ((((r13_z / $(16384)) % $(2)) * $(16384)) + (((r13_z / $(4)) % $(4096)) * $(4)) + (r13_z % $(4))), dtype=tl.float32))
    r13_value = r13_value.to(tl.float32)
    r13_value = tl.broadcast_to(r13_value, [$(32768)])
    tl.store($(vsigmoid_1_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(16777216))
    r14_value = tl.load($(input4_ptr) + ((((r14_z / $(8388608)) % $(2)) * $(8388608)) + (((r14_z / $(2048)) % $(4096)) * $(2048)) + (r14_z % $(2048))), dtype=tl.bfloat16)
    r14_value = r14_value.to(tl.float32)
    r14_value = tl.broadcast_to(r14_value, [$(16777216)])
    tl.store($(vto_copy_2_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(67108864))
    r15_value = (tl.load($(vto_copy_2_ptr) + ((((r15_z / $(33554432)) % $(2)) * $(8388608)) + (((r15_z / $(8192)) % $(4096)) * $(2048)) + (r15_z % $(2048))), dtype=tl.float32) * tl.load($(vadd_2_ptr) + ((((r15_z / $(33554432)) % $(2)) * $(16384)) + (((r15_z / $(8192)) % $(4096)) * $(4)) + ((r15_z / $(2048)) % $(4))), dtype=tl.float32))
    r15_value = r15_value.to(tl.float32)
    r15_value = tl.broadcast_to(r15_value, [$(67108864)])
    tl.store($(vmul_5_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(67108864))
    r16_value = (tl.load($(vto_copy_2_ptr) + ((((r16_z / $(33554432)) % $(2)) * $(8388608)) + (((r16_z / $(8192)) % $(4096)) * $(2048)) + (r16_z % $(2048))), dtype=tl.float32) * tl.load($(vto_copy_ptr) + ((((r16_z / $(33554432)) % $(2)) * $(33554432)) + (((r16_z / $(8192)) % $(4096)) * $(8192)) + (((r16_z / $(2048)) % $(4)) * $(2048)) + (r16_z % $(2048))), dtype=tl.float32))
    r16_value = r16_value.to(tl.float32)
    r16_value = tl.broadcast_to(r16_value, [$(67108864)])
    tl.store($(vmul_6_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(32768))
    r17_lane = tl.arange($(2048))
    r17_value = tl.sum(tl.load($(vmul_6_ptr) + ((((r17_z[:, None] / $(16384)) % $(2)) * $(33554432)) + (((r17_z[:, None] / $(4)) % $(4096)) * $(8192)) + ((r17_z[:, None] % $(4)) * $(2048)) + r17_lane[None, :]), dtype=tl.float32), axis=1)
    r17_value = r17_value.to(tl.float32)
    r17_value = tl.broadcast_to(r17_value, [$(32768)])
    tl.store($(vsum_2_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(32768))
    r18_value = (tl.load($(input5_ptr) + ((((r18_z / $(16384)) % $(2)) * $(16384)) + (((r18_z / $(4)) % $(4096)) * $(4)) + (r18_z % $(4))), dtype=tl.float32) * 2.0)
    r18_value = r18_value.to(tl.float32)
    r18_value = tl.broadcast_to(r18_value, [$(32768)])
    tl.store($(vmul_7_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(32768))
    r19_value = (tl.load($(vmul_7_ptr) + ((((r19_z / $(16384)) % $(2)) * $(16384)) + (((r19_z / $(4)) % $(4096)) * $(4)) + (r19_z % $(4))), dtype=tl.float32) * tl.load($(vsigmoid_1_ptr) + ((((r19_z / $(16384)) % $(2)) * $(16384)) + (((r19_z / $(4)) % $(4096)) * $(4)) + (r19_z % $(4))), dtype=tl.float32) * ($(1) - tl.load($(vsigmoid_1_ptr) + ((((r19_z / $(16384)) % $(2)) * $(16384)) + (((r19_z / $(4)) % $(4096)) * $(4)) + (r19_z % $(4))), dtype=tl.float32)))
    r19_value = r19_value.to(tl.float32)
    r19_value = tl.broadcast_to(r19_value, [$(32768)])
    tl.store($(vsigmoid_backward_ptr) + r19_z, r19_value)
    r20_z = tl.arange($(32768))
    r20_value = (tl.load($(vsigmoid_backward_ptr) + ((((r20_z / $(16384)) % $(2)) * $(16384)) + (((r20_z / $(4)) % $(4096)) * $(4)) + (r20_z % $(4))), dtype=tl.float32) * tl.load($(input2_ptr) + $(1), dtype=tl.float32))
    r20_value = r20_value.to(tl.float32)
    r20_value = tl.broadcast_to(r20_value, [$(32768)])
    tl.store($(vmul_8_ptr) + r20_z, r20_value)
    r21_z = tl.arange($(65536))
    r21_value = tl.where((((r21_z % $(8)) >= $(4)) & ((r21_z % $(8)) < $(8)) & (((r21_z % $(8)) - $(4)) % $(1) == $(0))), tl.load($(vmul_8_ptr) + ((((r21_z / $(32768)) % $(2)) * $(16384)) + (((r21_z / $(8)) % $(4096)) * $(4)) + (((r21_z % $(8)) - $(4)) % $(4))), dtype=tl.float32), 0.0)
    r21_value = r21_value.to(tl.float32)
    r21_value = tl.broadcast_to(r21_value, [$(65536)])
    tl.store($(vslice_backward_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(32768))
    r22_value = (tl.load($(vsum_2_ptr) + ((((r22_z / $(16384)) % $(2)) * $(16384)) + (((r22_z / $(4)) % $(4096)) * $(4)) + (r22_z % $(4))), dtype=tl.float32) * tl.load($(vsigmoid_ptr) + ((((r22_z / $(16384)) % $(2)) * $(16384)) + (((r22_z / $(4)) % $(4096)) * $(4)) + (r22_z % $(4))), dtype=tl.float32) * ($(1) - tl.load($(vsigmoid_ptr) + ((((r22_z / $(16384)) % $(2)) * $(16384)) + (((r22_z / $(4)) % $(4096)) * $(4)) + (r22_z % $(4))), dtype=tl.float32)))
    r22_value = r22_value.to(tl.float32)
    r22_value = tl.broadcast_to(r22_value, [$(32768)])
    tl.store($(vsigmoid_backward_1_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(32768))
    r23_value = (tl.load($(vsigmoid_backward_1_ptr) + ((((r23_z / $(16384)) % $(2)) * $(16384)) + (((r23_z / $(4)) % $(4096)) * $(4)) + (r23_z % $(4))), dtype=tl.float32) * tl.load($(input2_ptr) + $(0), dtype=tl.float32))
    r23_value = r23_value.to(tl.float32)
    r23_value = tl.broadcast_to(r23_value, [$(32768)])
    tl.store($(vmul_9_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(65536))
    r24_value = tl.where((((r24_z % $(8)) >= $(0)) & ((r24_z % $(8)) < $(4)) & (((r24_z % $(8)) - $(0)) % $(1) == $(0))), tl.load($(vmul_9_ptr) + ((((r24_z / $(32768)) % $(2)) * $(16384)) + (((r24_z / $(8)) % $(4096)) * $(4)) + (((r24_z % $(8)) - $(0)) % $(4))), dtype=tl.float32), 0.0)
    r24_value = r24_value.to(tl.float32)
    r24_value = tl.broadcast_to(r24_value, [$(65536)])
    tl.store($(vslice_backward_1_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(65536))
    r25_value = (tl.load($(vslice_backward_ptr) + ((((r25_z / $(32768)) % $(2)) * $(32768)) + (((r25_z / $(8)) % $(4096)) * $(8)) + (r25_z % $(8))), dtype=tl.float32) + tl.load($(vslice_backward_1_ptr) + ((((r25_z / $(32768)) % $(2)) * $(32768)) + (((r25_z / $(8)) % $(4096)) * $(8)) + (r25_z % $(8))), dtype=tl.float32))
    r25_value = r25_value.to(tl.float32)
    r25_value = tl.broadcast_to(r25_value, [$(65536)])
    tl.store($(vadd_5_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(65536))
    r26_value = (tl.load($(vadd_5_ptr) + ((((r26_z / $(32768)) % $(2)) * $(32768)) + (((r26_z / $(8)) % $(4096)) * $(8)) + (r26_z % $(8))), dtype=tl.float32) * tl.load($(vmm_ptr) + ((((r26_z / $(32768)) % $(2)) * $(32768)) + (((r26_z / $(8)) % $(4096)) * $(8)) + (r26_z % $(8))), dtype=tl.float32))
    r26_value = r26_value.to(tl.float32)
    r26_value = tl.broadcast_to(r26_value, [$(65536)])
    tl.store($(vmul_10_ptr) + r26_z, r26_value)
    r27_z = tl.arange($(65536))
    r27_value = (tl.load($(vadd_5_ptr) + ((((r27_z / $(32768)) % $(2)) * $(32768)) + (((r27_z / $(8)) % $(4096)) * $(8)) + (r27_z % $(8))), dtype=tl.float32) * tl.load($(vrsqrt_ptr) + ((((r27_z / $(32768)) % $(2)) * $(4096)) + ((r27_z / $(8)) % $(4096))), dtype=tl.float32))
    r27_value = r27_value.to(tl.float32)
    r27_value = tl.broadcast_to(r27_value, [$(65536)])
    tl.store($(vmul_11_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(8192))
    r28_lane = tl.arange($(8))
    r28_value = tl.sum(tl.load($(vmul_10_ptr) + ((((r28_z[:, None] / $(4096)) % $(2)) * $(32768)) + ((r28_z[:, None] % $(4096)) * $(8)) + r28_lane[None, :]), dtype=tl.float32), axis=1)
    r28_value = r28_value.to(tl.float32)
    r28_value = tl.broadcast_to(r28_value, [$(8192)])
    tl.store($(vsum_3_ptr) + r28_z, r28_value)
    r29_row = tl.arange($(8192))
    r29_col = tl.arange($(8192))
    r29_lane = tl.arange($(8))
    r29_left = tl.load($(vmul_11_ptr) + ((((((r29_row[:, None] * $(8)) + r29_lane[None, :]) / $(32768)) % $(2)) * $(32768)) + (((((r29_row[:, None] * $(8)) + r29_lane[None, :]) / $(8)) % $(4096)) * $(8)) + (((r29_row[:, None] * $(8)) + r29_lane[None, :]) % $(8))), dtype=tl.float32)
    r29_right = tl.load($(input1_ptr) + (((r29_lane[:, None] % $(8)) * $(8192)) + (r29_col[None, :] % $(8192))), dtype=tl.float32)
    r29_value = tl.dot(r29_left, r29_right)
    r29_value = r29_value.to(tl.float32)
    tl.store($(vmm_1_ptr) + ((r29_row[:, None] * $(8192)) + r29_col[None, :]), r29_value)
    r30_z = tl.arange($(67108864))
    r30_value = (tl.load($(vmul_5_ptr) + ((((r30_z / $(33554432)) % $(2)) * $(33554432)) + (((r30_z / $(8192)) % $(4096)) * $(8192)) + (((((((r30_z / $(33554432)) % $(2)) * $(33554432)) + (((r30_z / $(8192)) % $(4096)) * $(8192)) + (r30_z % $(8192))) / $(2048)) % $(4)) * $(2048)) + (((((r30_z / $(33554432)) % $(2)) * $(33554432)) + (((r30_z / $(8192)) % $(4096)) * $(8192)) + (r30_z % $(8192))) % $(2048))), dtype=tl.float32) + tl.load($(vmm_1_ptr) + ((((r30_z / $(33554432)) % $(2)) * $(33554432)) + (((r30_z / $(8192)) % $(4096)) * $(8192)) + (r30_z % $(8192))), dtype=tl.float32))
    r30_value = r30_value.to(tl.float32)
    r30_value = tl.broadcast_to(r30_value, [$(67108864)])
    tl.store($(vadd_6_ptr) + r30_z, r30_value)
    r31_z = tl.arange($(8192))
    r31_value = ((tl.load($(vrsqrt_ptr) + ((((r31_z / $(4096)) % $(2)) * $(4096)) + (r31_z % $(4096))), dtype=tl.float32)) * (tl.load($(vrsqrt_ptr) + ((((r31_z / $(4096)) % $(2)) * $(4096)) + (r31_z % $(4096))), dtype=tl.float32)) * (tl.load($(vrsqrt_ptr) + ((((r31_z / $(4096)) % $(2)) * $(4096)) + (r31_z % $(4096))), dtype=tl.float32)))
    r31_value = r31_value.to(tl.float32)
    r31_value = tl.broadcast_to(r31_value, [$(8192)])
    tl.store($(vpow_2_ptr) + r31_z, r31_value)
    r32_z = tl.arange($(8192))
    r32_value = (tl.load($(vsum_3_ptr) + ((((r32_z / $(4096)) % $(2)) * $(4096)) + (r32_z % $(4096))), dtype=tl.float32) * -0.5)
    r32_value = r32_value.to(tl.float32)
    r32_value = tl.broadcast_to(r32_value, [$(8192)])
    tl.store($(vmul_12_ptr) + r32_z, r32_value)
    r33_z = tl.arange($(8192))
    r33_value = (tl.load($(vmul_12_ptr) + ((((r33_z / $(4096)) % $(2)) * $(4096)) + (r33_z % $(4096))), dtype=tl.float32) * tl.load($(vpow_2_ptr) + ((((r33_z / $(4096)) % $(2)) * $(4096)) + (r33_z % $(4096))), dtype=tl.float32))
    r33_value = r33_value.to(tl.float32)
    r33_value = tl.broadcast_to(r33_value, [$(8192)])
    tl.store($(vmul_13_ptr) + r33_z, r33_value)
    r34_z = tl.arange($(67108864))
    r34_value = (tl.load($(vmul_13_ptr) + ((((r34_z / $(33554432)) % $(2)) * $(4096)) + ((r34_z / $(8192)) % $(4096))), dtype=tl.float32) / 8192.0)
    r34_value = r34_value.to(tl.float32)
    r34_value = tl.broadcast_to(r34_value, [$(67108864)])
    tl.store($(vdiv_ptr) + r34_z, r34_value)
    r35_z = tl.arange($(67108864))
    r35_value = ((tl.load($(vto_copy_ptr) + ((((r35_z / $(33554432)) % $(2)) * $(33554432)) + (((r35_z / $(8192)) % $(4096)) * $(8192)) + (r35_z % $(8192))), dtype=tl.float32)))
    r35_value = r35_value.to(tl.float32)
    r35_value = tl.broadcast_to(r35_value, [$(67108864)])
    tl.store($(vpow_3_ptr) + r35_z, r35_value)
    r36_z = tl.arange($(67108864))
    r36_value = (tl.load($(vpow_3_ptr) + ((((r36_z / $(33554432)) % $(2)) * $(33554432)) + (((r36_z / $(8192)) % $(4096)) * $(8192)) + (r36_z % $(8192))), dtype=tl.float32) * 2.0)
    r36_value = r36_value.to(tl.float32)
    r36_value = tl.broadcast_to(r36_value, [$(67108864)])
    tl.store($(vmul_14_ptr) + r36_z, r36_value)
    r37_z = tl.arange($(67108864))
    r37_value = (tl.load($(vdiv_ptr) + ((((r37_z / $(33554432)) % $(2)) * $(33554432)) + (((r37_z / $(8192)) % $(4096)) * $(8192)) + (r37_z % $(8192))), dtype=tl.float32) * tl.load($(vmul_14_ptr) + ((((r37_z / $(33554432)) % $(2)) * $(33554432)) + (((r37_z / $(8192)) % $(4096)) * $(8192)) + (r37_z % $(8192))), dtype=tl.float32))
    r37_value = r37_value.to(tl.float32)
    r37_value = tl.broadcast_to(r37_value, [$(67108864)])
    tl.store($(vmul_15_ptr) + r37_z, r37_value)
    r38_z = tl.arange($(67108864))
    r38_value = (tl.load($(vadd_6_ptr) + ((((r38_z / $(33554432)) % $(2)) * $(33554432)) + (((r38_z / $(8192)) % $(4096)) * $(8192)) + (r38_z % $(8192))), dtype=tl.float32) + tl.load($(vmul_15_ptr) + ((((r38_z / $(33554432)) % $(2)) * $(33554432)) + (((r38_z / $(8192)) % $(4096)) * $(8192)) + (r38_z % $(8192))), dtype=tl.float32))
    r38_value = r38_value.to(tl.float32)
    r38_value = tl.broadcast_to(r38_value, [$(67108864)])
    tl.store($(vadd_7_ptr) + r38_z, r38_value)
    r39_z = tl.arange($(67108864))
    r39_value = tl.load($(vadd_7_ptr) + ((((r39_z / $(33554432)) % $(2)) * $(33554432)) + (((r39_z / $(8192)) % $(4096)) * $(8192)) + (r39_z % $(8192))), dtype=tl.float32)
    r39_value = r39_value.to(tl.bfloat16)
    r39_value = tl.broadcast_to(r39_value, [$(67108864)])
    tl.store($(vto_copy_3_ptr) + r39_z, r39_value)
    z = tl.arange($(67108864))
    value = tl.load($(vto_copy_3_ptr) + ((((z / $(33554432)) % $(2)) * $(33554432)) + (((z / $(8192)) % $(4096)) * $(8192)) + (((z / $(2048)) % $(4)) * $(2048)) + (z % $(2048))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(67108864)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let input0_ptr : RegionName := "input0"
  let input1_ptr : RegionName := "input1"
  let input2_ptr : RegionName := "input2"
  let input3_ptr : RegionName := "input3"
  let input4_ptr : RegionName := "input4"
  let input5_ptr : RegionName := "input5"
  let out_ptr : RegionName := "out"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vpow_1_ptr : RegionName := "vpow_1"
  let vmean_ptr : RegionName := "vmean"
  let vadd_ptr : RegionName := "vadd"
  let vrsqrt_ptr : RegionName := "vrsqrt"
  let vmm_ptr : RegionName := "vmm"
  let vmul_ptr : RegionName := "vmul"
  let vmul_1_ptr : RegionName := "vmul_1"
  let vadd_1_ptr : RegionName := "vadd_1"
  let vsigmoid_ptr : RegionName := "vsigmoid"
  let vadd_2_ptr : RegionName := "vadd_2"
  let vmul_2_ptr : RegionName := "vmul_2"
  let vadd_3_ptr : RegionName := "vadd_3"
  let vsigmoid_1_ptr : RegionName := "vsigmoid_1"
  let vto_copy_2_ptr : RegionName := "vto_copy_2"
  let vmul_5_ptr : RegionName := "vmul_5"
  let vmul_6_ptr : RegionName := "vmul_6"
  let vsum_2_ptr : RegionName := "vsum_2"
  let vmul_7_ptr : RegionName := "vmul_7"
  let vsigmoid_backward_ptr : RegionName := "vsigmoid_backward"
  let vmul_8_ptr : RegionName := "vmul_8"
  let vslice_backward_ptr : RegionName := "vslice_backward"
  let vsigmoid_backward_1_ptr : RegionName := "vsigmoid_backward_1"
  let vmul_9_ptr : RegionName := "vmul_9"
  let vslice_backward_1_ptr : RegionName := "vslice_backward_1"
  let vadd_5_ptr : RegionName := "vadd_5"
  let vmul_10_ptr : RegionName := "vmul_10"
  let vmul_11_ptr : RegionName := "vmul_11"
  let vsum_3_ptr : RegionName := "vsum_3"
  let vmm_1_ptr : RegionName := "vmm_1"
  let vadd_6_ptr : RegionName := "vadd_6"
  let vpow_2_ptr : RegionName := "vpow_2"
  let vmul_12_ptr : RegionName := "vmul_12"
  let vmul_13_ptr : RegionName := "vmul_13"
  let vdiv_ptr : RegionName := "vdiv"
  let vpow_3_ptr : RegionName := "vpow_3"
  let vmul_14_ptr : RegionName := "vmul_14"
  let vmul_15_ptr : RegionName := "vmul_15"
  let vadd_7_ptr : RegionName := "vadd_7"
  let vto_copy_3_ptr : RegionName := "vto_copy_3"
  let candidate_vto_copy_ptr : RegionName := "candidate_vto_copy"
  let candidate_vpow_1_ptr : RegionName := "candidate_vpow_1"
  let candidate_vmean_ptr : RegionName := "candidate_vmean"
  let candidate_vadd_ptr : RegionName := "candidate_vadd"
  let candidate_vrsqrt_ptr : RegionName := "candidate_vrsqrt"
  let candidate_vmm_ptr : RegionName := "candidate_vmm"
  let candidate_vmul_ptr : RegionName := "candidate_vmul"
  let candidate_vmul_1_ptr : RegionName := "candidate_vmul_1"
  let candidate_vadd_1_ptr : RegionName := "candidate_vadd_1"
  let candidate_vsigmoid_ptr : RegionName := "candidate_vsigmoid"
  let candidate_vadd_2_ptr : RegionName := "candidate_vadd_2"
  let candidate_vmul_2_ptr : RegionName := "candidate_vmul_2"
  let candidate_vadd_3_ptr : RegionName := "candidate_vadd_3"
  let candidate_vsigmoid_1_ptr : RegionName := "candidate_vsigmoid_1"
  let candidate_vto_copy_2_ptr : RegionName := "candidate_vto_copy_2"
  let candidate_vmul_5_ptr : RegionName := "candidate_vmul_5"
  let candidate_vmul_6_ptr : RegionName := "candidate_vmul_6"
  let candidate_vsum_2_ptr : RegionName := "candidate_vsum_2"
  let candidate_vmul_7_ptr : RegionName := "candidate_vmul_7"
  let candidate_vsigmoid_backward_ptr : RegionName := "candidate_vsigmoid_backward"
  let candidate_vmul_8_ptr : RegionName := "candidate_vmul_8"
  let candidate_vslice_backward_ptr : RegionName := "candidate_vslice_backward"
  let candidate_vsigmoid_backward_1_ptr : RegionName := "candidate_vsigmoid_backward_1"
  let candidate_vmul_9_ptr : RegionName := "candidate_vmul_9"
  let candidate_vslice_backward_1_ptr : RegionName := "candidate_vslice_backward_1"
  let candidate_vadd_5_ptr : RegionName := "candidate_vadd_5"
  let candidate_vmul_10_ptr : RegionName := "candidate_vmul_10"
  let candidate_vmul_11_ptr : RegionName := "candidate_vmul_11"
  let candidate_vsum_3_ptr : RegionName := "candidate_vsum_3"
  let candidate_vmm_1_ptr : RegionName := "candidate_vmm_1"
  let candidate_vadd_6_ptr : RegionName := "candidate_vadd_6"
  let candidate_vpow_2_ptr : RegionName := "candidate_vpow_2"
  let candidate_vmul_12_ptr : RegionName := "candidate_vmul_12"
  let candidate_vmul_13_ptr : RegionName := "candidate_vmul_13"
  let candidate_vdiv_ptr : RegionName := "candidate_vdiv"
  let candidate_vpow_3_ptr : RegionName := "candidate_vpow_3"
  let candidate_vmul_14_ptr : RegionName := "candidate_vmul_14"
  let candidate_vmul_15_ptr : RegionName := "candidate_vmul_15"
  let candidate_vadd_7_ptr : RegionName := "candidate_vadd_7"
  let candidate_vto_copy_3_ptr : RegionName := "candidate_vto_copy_3"
  triton {
    r0_z = tl.arange($(67108864))
    r0_value = tl.load($(input0_ptr) + ((((r0_z / $(33554432)) % $(2)) * $(33554432)) + (((r0_z / $(8192)) % $(4096)) * $(8192)) + (((((((r0_z / $(33554432)) % $(2)) * $(33554432)) + (((r0_z / $(8192)) % $(4096)) * $(8192)) + (r0_z % $(8192))) / $(2048)) % $(4)) * $(2048)) + (((((r0_z / $(33554432)) % $(2)) * $(33554432)) + (((r0_z / $(8192)) % $(4096)) * $(8192)) + (r0_z % $(8192))) % $(2048))), dtype=tl.bfloat16)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(67108864)])
    tl.store($(candidate_vto_copy_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(67108864))
    r1_value = ((tl.load($(candidate_vto_copy_ptr) + ((((r1_z / $(33554432)) % $(2)) * $(33554432)) + (((r1_z / $(8192)) % $(4096)) * $(8192)) + (r1_z % $(8192))), dtype=tl.float32)) * (tl.load($(candidate_vto_copy_ptr) + ((((r1_z / $(33554432)) % $(2)) * $(33554432)) + (((r1_z / $(8192)) % $(4096)) * $(8192)) + (r1_z % $(8192))), dtype=tl.float32)))
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(67108864)])
    tl.store($(candidate_vpow_1_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(8192))
    r2_lane = tl.arange($(8192))
    r2_value = (tl.sum(tl.load($(candidate_vpow_1_ptr) + ((((r2_z[:, None] / $(4096)) % $(2)) * $(33554432)) + ((r2_z[:, None] % $(4096)) * $(8192)) + r2_lane[None, :]), dtype=tl.float32), axis=1) / 8192.0)
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(8192)])
    tl.store($(candidate_vmean_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(8192))
    r3_value = (tl.load($(candidate_vmean_ptr) + ((((r3_z / $(4096)) % $(2)) * $(4096)) + (r3_z % $(4096))), dtype=tl.float32) + 0.000001)
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(8192)])
    tl.store($(candidate_vadd_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(8192))
    r4_value = tl.rsqrt(tl.load($(candidate_vadd_ptr) + ((((r4_z / $(4096)) % $(2)) * $(4096)) + (r4_z % $(4096))), dtype=tl.float32))
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(8192)])
    tl.store($(candidate_vrsqrt_ptr) + r4_z, r4_value)
    r5_row = tl.arange($(8192))
    r5_col = tl.arange($(8))
    r5_lane = tl.arange($(8192))
    r5_left = tl.load($(candidate_vto_copy_ptr) + ((((((r5_row[:, None] * $(8192)) + r5_lane[None, :]) / $(33554432)) % $(2)) * $(33554432)) + (((((r5_row[:, None] * $(8192)) + r5_lane[None, :]) / $(8192)) % $(4096)) * $(8192)) + (((r5_row[:, None] * $(8192)) + r5_lane[None, :]) % $(8192))), dtype=tl.float32)
    r5_right = tl.load($(input1_ptr) + (((r5_col[None, :] % $(8)) * $(8192)) + (r5_lane[:, None] % $(8192))), dtype=tl.float32)
    r5_value = tl.dot(r5_left, r5_right)
    r5_value = r5_value.to(tl.float32)
    tl.store($(candidate_vmm_ptr) + ((r5_row[:, None] * $(8)) + r5_col[None, :]), r5_value)
    r6_z = tl.arange($(65536))
    r6_value = (tl.load($(candidate_vmm_ptr) + ((((r6_z / $(32768)) % $(2)) * $(32768)) + (((r6_z / $(8)) % $(4096)) * $(8)) + (r6_z % $(8))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_ptr) + ((((r6_z / $(32768)) % $(2)) * $(4096)) + ((r6_z / $(8)) % $(4096))), dtype=tl.float32))
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(65536)])
    tl.store($(candidate_vmul_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(32768))
    r7_value = (tl.load($(candidate_vmul_ptr) + ((((r7_z / $(16384)) % $(2)) * $(32768)) + (((r7_z / $(4)) % $(4096)) * $(8)) + (r7_z % $(4))), dtype=tl.float32) * tl.load($(input2_ptr) + $(0), dtype=tl.float32))
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(32768)])
    tl.store($(candidate_vmul_1_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(32768))
    r8_value = (tl.load($(candidate_vmul_1_ptr) + ((((r8_z / $(16384)) % $(2)) * $(16384)) + (((r8_z / $(4)) % $(4096)) * $(4)) + (r8_z % $(4))), dtype=tl.float32) + tl.load($(input3_ptr) + (r8_z % $(4)), dtype=tl.float32))
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(32768)])
    tl.store($(candidate_vadd_1_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(32768))
    r9_value = tl.sigmoid(tl.load($(candidate_vadd_1_ptr) + ((((r9_z / $(16384)) % $(2)) * $(16384)) + (((r9_z / $(4)) % $(4096)) * $(4)) + (r9_z % $(4))), dtype=tl.float32))
    r9_value = r9_value.to(tl.float32)
    r9_value = tl.broadcast_to(r9_value, [$(32768)])
    tl.store($(candidate_vsigmoid_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(32768))
    r10_value = (tl.load($(candidate_vsigmoid_ptr) + ((((r10_z / $(16384)) % $(2)) * $(16384)) + (((r10_z / $(4)) % $(4096)) * $(4)) + (r10_z % $(4))), dtype=tl.float32) + 0.000001)
    r10_value = r10_value.to(tl.float32)
    r10_value = tl.broadcast_to(r10_value, [$(32768)])
    tl.store($(candidate_vadd_2_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(32768))
    r11_value = (tl.load($(candidate_vmul_ptr) + ((((r11_z / $(16384)) % $(2)) * $(32768)) + (((r11_z / $(4)) % $(4096)) * $(8)) + (r11_z % $(4)) + $(4)), dtype=tl.float32) * tl.load($(input2_ptr) + $(1), dtype=tl.float32))
    r11_value = r11_value.to(tl.float32)
    r11_value = tl.broadcast_to(r11_value, [$(32768)])
    tl.store($(candidate_vmul_2_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(32768))
    r12_value = (tl.load($(candidate_vmul_2_ptr) + ((((r12_z / $(16384)) % $(2)) * $(16384)) + (((r12_z / $(4)) % $(4096)) * $(4)) + (r12_z % $(4))), dtype=tl.float32) + tl.load($(input3_ptr) + ((r12_z % $(4)) + $(4)), dtype=tl.float32))
    r12_value = r12_value.to(tl.float32)
    r12_value = tl.broadcast_to(r12_value, [$(32768)])
    tl.store($(candidate_vadd_3_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(32768))
    r13_value = tl.sigmoid(tl.load($(candidate_vadd_3_ptr) + ((((r13_z / $(16384)) % $(2)) * $(16384)) + (((r13_z / $(4)) % $(4096)) * $(4)) + (r13_z % $(4))), dtype=tl.float32))
    r13_value = r13_value.to(tl.float32)
    r13_value = tl.broadcast_to(r13_value, [$(32768)])
    tl.store($(candidate_vsigmoid_1_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(16777216))
    r14_value = tl.load($(input4_ptr) + ((((r14_z / $(8388608)) % $(2)) * $(8388608)) + (((r14_z / $(2048)) % $(4096)) * $(2048)) + (r14_z % $(2048))), dtype=tl.bfloat16)
    r14_value = r14_value.to(tl.float32)
    r14_value = tl.broadcast_to(r14_value, [$(16777216)])
    tl.store($(candidate_vto_copy_2_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(67108864))
    r15_value = (tl.load($(candidate_vto_copy_2_ptr) + ((((r15_z / $(33554432)) % $(2)) * $(8388608)) + (((r15_z / $(8192)) % $(4096)) * $(2048)) + (r15_z % $(2048))), dtype=tl.float32) * tl.load($(candidate_vadd_2_ptr) + ((((r15_z / $(33554432)) % $(2)) * $(16384)) + (((r15_z / $(8192)) % $(4096)) * $(4)) + ((r15_z / $(2048)) % $(4))), dtype=tl.float32))
    r15_value = r15_value.to(tl.float32)
    r15_value = tl.broadcast_to(r15_value, [$(67108864)])
    tl.store($(candidate_vmul_5_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(67108864))
    r16_value = (tl.load($(candidate_vto_copy_2_ptr) + ((((r16_z / $(33554432)) % $(2)) * $(8388608)) + (((r16_z / $(8192)) % $(4096)) * $(2048)) + (r16_z % $(2048))), dtype=tl.float32) * tl.load($(candidate_vto_copy_ptr) + ((((r16_z / $(33554432)) % $(2)) * $(33554432)) + (((r16_z / $(8192)) % $(4096)) * $(8192)) + (((r16_z / $(2048)) % $(4)) * $(2048)) + (r16_z % $(2048))), dtype=tl.float32))
    r16_value = r16_value.to(tl.float32)
    r16_value = tl.broadcast_to(r16_value, [$(67108864)])
    tl.store($(candidate_vmul_6_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(32768))
    r17_lane = tl.arange($(2048))
    r17_value = tl.sum(tl.load($(candidate_vmul_6_ptr) + ((((r17_z[:, None] / $(16384)) % $(2)) * $(33554432)) + (((r17_z[:, None] / $(4)) % $(4096)) * $(8192)) + ((r17_z[:, None] % $(4)) * $(2048)) + r17_lane[None, :]), dtype=tl.float32), axis=1)
    r17_value = r17_value.to(tl.float32)
    r17_value = tl.broadcast_to(r17_value, [$(32768)])
    tl.store($(candidate_vsum_2_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(32768))
    r18_value = (tl.load($(input5_ptr) + ((((r18_z / $(16384)) % $(2)) * $(16384)) + (((r18_z / $(4)) % $(4096)) * $(4)) + (r18_z % $(4))), dtype=tl.float32) * 2.0)
    r18_value = r18_value.to(tl.float32)
    r18_value = tl.broadcast_to(r18_value, [$(32768)])
    tl.store($(candidate_vmul_7_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(32768))
    r19_value = (tl.load($(candidate_vmul_7_ptr) + ((((r19_z / $(16384)) % $(2)) * $(16384)) + (((r19_z / $(4)) % $(4096)) * $(4)) + (r19_z % $(4))), dtype=tl.float32) * tl.load($(candidate_vsigmoid_1_ptr) + ((((r19_z / $(16384)) % $(2)) * $(16384)) + (((r19_z / $(4)) % $(4096)) * $(4)) + (r19_z % $(4))), dtype=tl.float32) * ($(1) - tl.load($(candidate_vsigmoid_1_ptr) + ((((r19_z / $(16384)) % $(2)) * $(16384)) + (((r19_z / $(4)) % $(4096)) * $(4)) + (r19_z % $(4))), dtype=tl.float32)))
    r19_value = r19_value.to(tl.float32)
    r19_value = tl.broadcast_to(r19_value, [$(32768)])
    tl.store($(candidate_vsigmoid_backward_ptr) + r19_z, r19_value)
    r20_z = tl.arange($(32768))
    r20_value = (tl.load($(candidate_vsigmoid_backward_ptr) + ((((r20_z / $(16384)) % $(2)) * $(16384)) + (((r20_z / $(4)) % $(4096)) * $(4)) + (r20_z % $(4))), dtype=tl.float32) * tl.load($(input2_ptr) + $(1), dtype=tl.float32))
    r20_value = r20_value.to(tl.float32)
    r20_value = tl.broadcast_to(r20_value, [$(32768)])
    tl.store($(candidate_vmul_8_ptr) + r20_z, r20_value)
    r21_z = tl.arange($(65536))
    r21_value = tl.where((((r21_z % $(8)) >= $(4)) & ((r21_z % $(8)) < $(8)) & (((r21_z % $(8)) - $(4)) % $(1) == $(0))), tl.load($(candidate_vmul_8_ptr) + ((((r21_z / $(32768)) % $(2)) * $(16384)) + (((r21_z / $(8)) % $(4096)) * $(4)) + (((r21_z % $(8)) - $(4)) % $(4))), dtype=tl.float32), 0.0)
    r21_value = r21_value.to(tl.float32)
    r21_value = tl.broadcast_to(r21_value, [$(65536)])
    tl.store($(candidate_vslice_backward_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(32768))
    r22_value = (tl.load($(candidate_vsum_2_ptr) + ((((r22_z / $(16384)) % $(2)) * $(16384)) + (((r22_z / $(4)) % $(4096)) * $(4)) + (r22_z % $(4))), dtype=tl.float32) * tl.load($(candidate_vsigmoid_ptr) + ((((r22_z / $(16384)) % $(2)) * $(16384)) + (((r22_z / $(4)) % $(4096)) * $(4)) + (r22_z % $(4))), dtype=tl.float32) * ($(1) - tl.load($(candidate_vsigmoid_ptr) + ((((r22_z / $(16384)) % $(2)) * $(16384)) + (((r22_z / $(4)) % $(4096)) * $(4)) + (r22_z % $(4))), dtype=tl.float32)))
    r22_value = r22_value.to(tl.float32)
    r22_value = tl.broadcast_to(r22_value, [$(32768)])
    tl.store($(candidate_vsigmoid_backward_1_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(32768))
    r23_value = (tl.load($(candidate_vsigmoid_backward_1_ptr) + ((((r23_z / $(16384)) % $(2)) * $(16384)) + (((r23_z / $(4)) % $(4096)) * $(4)) + (r23_z % $(4))), dtype=tl.float32) * tl.load($(input2_ptr) + $(0), dtype=tl.float32))
    r23_value = r23_value.to(tl.float32)
    r23_value = tl.broadcast_to(r23_value, [$(32768)])
    tl.store($(candidate_vmul_9_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(65536))
    r24_value = tl.where((((r24_z % $(8)) >= $(0)) & ((r24_z % $(8)) < $(4)) & (((r24_z % $(8)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vmul_9_ptr) + ((((r24_z / $(32768)) % $(2)) * $(16384)) + (((r24_z / $(8)) % $(4096)) * $(4)) + (((r24_z % $(8)) - $(0)) % $(4))), dtype=tl.float32), 0.0)
    r24_value = r24_value.to(tl.float32)
    r24_value = tl.broadcast_to(r24_value, [$(65536)])
    tl.store($(candidate_vslice_backward_1_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(65536))
    r25_value = (tl.load($(candidate_vslice_backward_ptr) + ((((r25_z / $(32768)) % $(2)) * $(32768)) + (((r25_z / $(8)) % $(4096)) * $(8)) + (r25_z % $(8))), dtype=tl.float32) + tl.load($(candidate_vslice_backward_1_ptr) + ((((r25_z / $(32768)) % $(2)) * $(32768)) + (((r25_z / $(8)) % $(4096)) * $(8)) + (r25_z % $(8))), dtype=tl.float32))
    r25_value = r25_value.to(tl.float32)
    r25_value = tl.broadcast_to(r25_value, [$(65536)])
    tl.store($(candidate_vadd_5_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(65536))
    r26_value = (tl.load($(candidate_vadd_5_ptr) + ((((r26_z / $(32768)) % $(2)) * $(32768)) + (((r26_z / $(8)) % $(4096)) * $(8)) + (r26_z % $(8))), dtype=tl.float32) * tl.load($(candidate_vmm_ptr) + ((((r26_z / $(32768)) % $(2)) * $(32768)) + (((r26_z / $(8)) % $(4096)) * $(8)) + (r26_z % $(8))), dtype=tl.float32))
    r26_value = r26_value.to(tl.float32)
    r26_value = tl.broadcast_to(r26_value, [$(65536)])
    tl.store($(candidate_vmul_10_ptr) + r26_z, r26_value)
    r27_z = tl.arange($(65536))
    r27_value = (tl.load($(candidate_vadd_5_ptr) + ((((r27_z / $(32768)) % $(2)) * $(32768)) + (((r27_z / $(8)) % $(4096)) * $(8)) + (r27_z % $(8))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_ptr) + ((((r27_z / $(32768)) % $(2)) * $(4096)) + ((r27_z / $(8)) % $(4096))), dtype=tl.float32))
    r27_value = r27_value.to(tl.float32)
    r27_value = tl.broadcast_to(r27_value, [$(65536)])
    tl.store($(candidate_vmul_11_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(8192))
    r28_lane = tl.arange($(8))
    r28_value = tl.sum(tl.load($(candidate_vmul_10_ptr) + ((((r28_z[:, None] / $(4096)) % $(2)) * $(32768)) + ((r28_z[:, None] % $(4096)) * $(8)) + r28_lane[None, :]), dtype=tl.float32), axis=1)
    r28_value = r28_value.to(tl.float32)
    r28_value = tl.broadcast_to(r28_value, [$(8192)])
    tl.store($(candidate_vsum_3_ptr) + r28_z, r28_value)
    r29_row = tl.arange($(8192))
    r29_col = tl.arange($(8192))
    r29_lane = tl.arange($(8))
    r29_left = tl.load($(candidate_vmul_11_ptr) + ((((((r29_row[:, None] * $(8)) + r29_lane[None, :]) / $(32768)) % $(2)) * $(32768)) + (((((r29_row[:, None] * $(8)) + r29_lane[None, :]) / $(8)) % $(4096)) * $(8)) + (((r29_row[:, None] * $(8)) + r29_lane[None, :]) % $(8))), dtype=tl.float32)
    r29_right = tl.load($(input1_ptr) + (((r29_lane[:, None] % $(8)) * $(8192)) + (r29_col[None, :] % $(8192))), dtype=tl.float32)
    r29_value = tl.dot(r29_left, r29_right)
    r29_value = r29_value.to(tl.float32)
    tl.store($(candidate_vmm_1_ptr) + ((r29_row[:, None] * $(8192)) + r29_col[None, :]), r29_value)
    r30_z = tl.arange($(67108864))
    r30_value = (tl.load($(candidate_vmul_5_ptr) + ((((r30_z / $(33554432)) % $(2)) * $(33554432)) + (((r30_z / $(8192)) % $(4096)) * $(8192)) + (((((((r30_z / $(33554432)) % $(2)) * $(33554432)) + (((r30_z / $(8192)) % $(4096)) * $(8192)) + (r30_z % $(8192))) / $(2048)) % $(4)) * $(2048)) + (((((r30_z / $(33554432)) % $(2)) * $(33554432)) + (((r30_z / $(8192)) % $(4096)) * $(8192)) + (r30_z % $(8192))) % $(2048))), dtype=tl.float32) + tl.load($(candidate_vmm_1_ptr) + ((((r30_z / $(33554432)) % $(2)) * $(33554432)) + (((r30_z / $(8192)) % $(4096)) * $(8192)) + (r30_z % $(8192))), dtype=tl.float32))
    r30_value = r30_value.to(tl.float32)
    r30_value = tl.broadcast_to(r30_value, [$(67108864)])
    tl.store($(candidate_vadd_6_ptr) + r30_z, r30_value)
    r31_z = tl.arange($(8192))
    r31_value = ((tl.load($(candidate_vrsqrt_ptr) + ((((r31_z / $(4096)) % $(2)) * $(4096)) + (r31_z % $(4096))), dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_ptr) + ((((r31_z / $(4096)) % $(2)) * $(4096)) + (r31_z % $(4096))), dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_ptr) + ((((r31_z / $(4096)) % $(2)) * $(4096)) + (r31_z % $(4096))), dtype=tl.float32)))
    r31_value = r31_value.to(tl.float32)
    r31_value = tl.broadcast_to(r31_value, [$(8192)])
    tl.store($(candidate_vpow_2_ptr) + r31_z, r31_value)
    r32_z = tl.arange($(8192))
    r32_value = (tl.load($(candidate_vsum_3_ptr) + ((((r32_z / $(4096)) % $(2)) * $(4096)) + (r32_z % $(4096))), dtype=tl.float32) * -0.5)
    r32_value = r32_value.to(tl.float32)
    r32_value = tl.broadcast_to(r32_value, [$(8192)])
    tl.store($(candidate_vmul_12_ptr) + r32_z, r32_value)
    r33_z = tl.arange($(8192))
    r33_value = (tl.load($(candidate_vmul_12_ptr) + ((((r33_z / $(4096)) % $(2)) * $(4096)) + (r33_z % $(4096))), dtype=tl.float32) * tl.load($(candidate_vpow_2_ptr) + ((((r33_z / $(4096)) % $(2)) * $(4096)) + (r33_z % $(4096))), dtype=tl.float32))
    r33_value = r33_value.to(tl.float32)
    r33_value = tl.broadcast_to(r33_value, [$(8192)])
    tl.store($(candidate_vmul_13_ptr) + r33_z, r33_value)
    r34_z = tl.arange($(67108864))
    r34_value = (tl.load($(candidate_vmul_13_ptr) + ((((r34_z / $(33554432)) % $(2)) * $(4096)) + ((r34_z / $(8192)) % $(4096))), dtype=tl.float32) / 8192.0)
    r34_value = r34_value.to(tl.float32)
    r34_value = tl.broadcast_to(r34_value, [$(67108864)])
    tl.store($(candidate_vdiv_ptr) + r34_z, r34_value)
    r35_z = tl.arange($(67108864))
    r35_value = ((tl.load($(candidate_vto_copy_ptr) + ((((r35_z / $(33554432)) % $(2)) * $(33554432)) + (((r35_z / $(8192)) % $(4096)) * $(8192)) + (r35_z % $(8192))), dtype=tl.float32)))
    r35_value = r35_value.to(tl.float32)
    r35_value = tl.broadcast_to(r35_value, [$(67108864)])
    tl.store($(candidate_vpow_3_ptr) + r35_z, r35_value)
    r36_z = tl.arange($(67108864))
    r36_value = (tl.load($(candidate_vpow_3_ptr) + ((((r36_z / $(33554432)) % $(2)) * $(33554432)) + (((r36_z / $(8192)) % $(4096)) * $(8192)) + (r36_z % $(8192))), dtype=tl.float32) * 2.0)
    r36_value = r36_value.to(tl.float32)
    r36_value = tl.broadcast_to(r36_value, [$(67108864)])
    tl.store($(candidate_vmul_14_ptr) + r36_z, r36_value)
    r37_z = tl.arange($(67108864))
    r37_value = (tl.load($(candidate_vdiv_ptr) + ((((r37_z / $(33554432)) % $(2)) * $(33554432)) + (((r37_z / $(8192)) % $(4096)) * $(8192)) + (r37_z % $(8192))), dtype=tl.float32) * tl.load($(candidate_vmul_14_ptr) + ((((r37_z / $(33554432)) % $(2)) * $(33554432)) + (((r37_z / $(8192)) % $(4096)) * $(8192)) + (r37_z % $(8192))), dtype=tl.float32))
    r37_value = r37_value.to(tl.float32)
    r37_value = tl.broadcast_to(r37_value, [$(67108864)])
    tl.store($(candidate_vmul_15_ptr) + r37_z, r37_value)
    r38_z = tl.arange($(67108864))
    r38_value = (tl.load($(candidate_vadd_6_ptr) + ((((r38_z / $(33554432)) % $(2)) * $(33554432)) + (((r38_z / $(8192)) % $(4096)) * $(8192)) + (r38_z % $(8192))), dtype=tl.float32) + tl.load($(candidate_vmul_15_ptr) + ((((r38_z / $(33554432)) % $(2)) * $(33554432)) + (((r38_z / $(8192)) % $(4096)) * $(8192)) + (r38_z % $(8192))), dtype=tl.float32))
    r38_value = r38_value.to(tl.float32)
    r38_value = tl.broadcast_to(r38_value, [$(67108864)])
    tl.store($(candidate_vadd_7_ptr) + r38_z, r38_value)
    r39_z = tl.arange($(67108864))
    r39_value = tl.load($(candidate_vadd_7_ptr) + ((((r39_z / $(33554432)) % $(2)) * $(33554432)) + (((r39_z / $(8192)) % $(4096)) * $(8192)) + (r39_z % $(8192))), dtype=tl.float32)
    r39_value = r39_value.to(tl.bfloat16)
    r39_value = tl.broadcast_to(r39_value, [$(67108864)])
    tl.store($(candidate_vto_copy_3_ptr) + r39_z, r39_value)
    z = tl.arange($(67108864))
    value = tl.load($(candidate_vto_copy_3_ptr) + ((((z / $(33554432)) % $(2)) * $(33554432)) + (((z / $(8192)) % $(4096)) * $(8192)) + (((z / $(2048)) % $(4)) * $(2048)) + (z % $(2048))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(67108864)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_fused_hc_pre_dx
