import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_fused_hc_head_dx
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
  let vto_copy_2_ptr : RegionName := "vto_copy_2"
  let vmul_3_ptr : RegionName := "vmul_3"
  let vmul_4_ptr : RegionName := "vmul_4"
  let vsum_2_ptr : RegionName := "vsum_2"
  let vsigmoid_backward_ptr : RegionName := "vsigmoid_backward"
  let vmul_5_ptr : RegionName := "vmul_5"
  let vmul_6_ptr : RegionName := "vmul_6"
  let vmul_7_ptr : RegionName := "vmul_7"
  let vsum_3_ptr : RegionName := "vsum_3"
  let vmm_1_ptr : RegionName := "vmm_1"
  let vadd_3_ptr : RegionName := "vadd_3"
  let vpow_2_ptr : RegionName := "vpow_2"
  let vmul_8_ptr : RegionName := "vmul_8"
  let vmul_9_ptr : RegionName := "vmul_9"
  let vdiv_ptr : RegionName := "vdiv"
  let vpow_3_ptr : RegionName := "vpow_3"
  let vmul_10_ptr : RegionName := "vmul_10"
  let vmul_11_ptr : RegionName := "vmul_11"
  let vadd_4_ptr : RegionName := "vadd_4"
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
    r5_col = tl.arange($(4))
    r5_lane = tl.arange($(8192))
    r5_left = tl.load($(vto_copy_ptr) + ((((((r5_row[:, None] * $(8192)) + r5_lane[None, :]) / $(33554432)) % $(2)) * $(33554432)) + (((((r5_row[:, None] * $(8192)) + r5_lane[None, :]) / $(8192)) % $(4096)) * $(8192)) + (((r5_row[:, None] * $(8192)) + r5_lane[None, :]) % $(8192))), dtype=tl.float32)
    r5_right = tl.load($(input1_ptr) + (((r5_col[None, :] % $(4)) * $(8192)) + (r5_lane[:, None] % $(8192))), dtype=tl.float32)
    r5_value = tl.dot(r5_left, r5_right)
    r5_value = r5_value.to(tl.float32)
    tl.store($(vmm_ptr) + ((r5_row[:, None] * $(4)) + r5_col[None, :]), r5_value)
    r6_z = tl.arange($(32768))
    r6_value = (tl.load($(vmm_ptr) + ((((r6_z / $(16384)) % $(2)) * $(16384)) + (((r6_z / $(4)) % $(4096)) * $(4)) + (r6_z % $(4))), dtype=tl.float32) * tl.load($(vrsqrt_ptr) + ((((r6_z / $(16384)) % $(2)) * $(4096)) + ((r6_z / $(4)) % $(4096))), dtype=tl.float32))
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(32768)])
    tl.store($(vmul_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(32768))
    r7_value = (tl.load($(vmul_ptr) + ((((r7_z / $(16384)) % $(2)) * $(16384)) + (((r7_z / $(4)) % $(4096)) * $(4)) + (r7_z % $(4))), dtype=tl.float32) * tl.load($(input2_ptr) + $(0), dtype=tl.float32))
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
    r11_z = tl.arange($(16777216))
    r11_value = tl.load($(input4_ptr) + ((((r11_z / $(8388608)) % $(2)) * $(8388608)) + (((r11_z / $(2048)) % $(4096)) * $(2048)) + (r11_z % $(2048))), dtype=tl.bfloat16)
    r11_value = r11_value.to(tl.float32)
    r11_value = tl.broadcast_to(r11_value, [$(16777216)])
    tl.store($(vto_copy_2_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(67108864))
    r12_value = (tl.load($(vto_copy_2_ptr) + ((((r12_z / $(33554432)) % $(2)) * $(8388608)) + (((r12_z / $(8192)) % $(4096)) * $(2048)) + (r12_z % $(2048))), dtype=tl.float32) * tl.load($(vadd_2_ptr) + ((((r12_z / $(33554432)) % $(2)) * $(16384)) + (((r12_z / $(8192)) % $(4096)) * $(4)) + ((r12_z / $(2048)) % $(4))), dtype=tl.float32))
    r12_value = r12_value.to(tl.float32)
    r12_value = tl.broadcast_to(r12_value, [$(67108864)])
    tl.store($(vmul_3_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(67108864))
    r13_value = (tl.load($(vto_copy_2_ptr) + ((((r13_z / $(33554432)) % $(2)) * $(8388608)) + (((r13_z / $(8192)) % $(4096)) * $(2048)) + (r13_z % $(2048))), dtype=tl.float32) * tl.load($(vto_copy_ptr) + ((((r13_z / $(33554432)) % $(2)) * $(33554432)) + (((r13_z / $(8192)) % $(4096)) * $(8192)) + (((r13_z / $(2048)) % $(4)) * $(2048)) + (r13_z % $(2048))), dtype=tl.float32))
    r13_value = r13_value.to(tl.float32)
    r13_value = tl.broadcast_to(r13_value, [$(67108864)])
    tl.store($(vmul_4_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(32768))
    r14_lane = tl.arange($(2048))
    r14_value = tl.sum(tl.load($(vmul_4_ptr) + ((((r14_z[:, None] / $(16384)) % $(2)) * $(33554432)) + (((r14_z[:, None] / $(4)) % $(4096)) * $(8192)) + ((r14_z[:, None] % $(4)) * $(2048)) + r14_lane[None, :]), dtype=tl.float32), axis=1)
    r14_value = r14_value.to(tl.float32)
    r14_value = tl.broadcast_to(r14_value, [$(32768)])
    tl.store($(vsum_2_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(32768))
    r15_value = (tl.load($(vsum_2_ptr) + ((((r15_z / $(16384)) % $(2)) * $(16384)) + (((r15_z / $(4)) % $(4096)) * $(4)) + (r15_z % $(4))), dtype=tl.float32) * tl.load($(vsigmoid_ptr) + ((((r15_z / $(16384)) % $(2)) * $(16384)) + (((r15_z / $(4)) % $(4096)) * $(4)) + (r15_z % $(4))), dtype=tl.float32) * ($(1) - tl.load($(vsigmoid_ptr) + ((((r15_z / $(16384)) % $(2)) * $(16384)) + (((r15_z / $(4)) % $(4096)) * $(4)) + (r15_z % $(4))), dtype=tl.float32)))
    r15_value = r15_value.to(tl.float32)
    r15_value = tl.broadcast_to(r15_value, [$(32768)])
    tl.store($(vsigmoid_backward_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(32768))
    r16_value = (tl.load($(vsigmoid_backward_ptr) + ((((r16_z / $(16384)) % $(2)) * $(16384)) + (((r16_z / $(4)) % $(4096)) * $(4)) + (r16_z % $(4))), dtype=tl.float32) * tl.load($(input2_ptr) + $(0), dtype=tl.float32))
    r16_value = r16_value.to(tl.float32)
    r16_value = tl.broadcast_to(r16_value, [$(32768)])
    tl.store($(vmul_5_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(32768))
    r17_value = (tl.load($(vmul_5_ptr) + ((((r17_z / $(16384)) % $(2)) * $(16384)) + (((r17_z / $(4)) % $(4096)) * $(4)) + (r17_z % $(4))), dtype=tl.float32) * tl.load($(vmm_ptr) + ((((r17_z / $(16384)) % $(2)) * $(16384)) + (((r17_z / $(4)) % $(4096)) * $(4)) + (r17_z % $(4))), dtype=tl.float32))
    r17_value = r17_value.to(tl.float32)
    r17_value = tl.broadcast_to(r17_value, [$(32768)])
    tl.store($(vmul_6_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(32768))
    r18_value = (tl.load($(vmul_5_ptr) + ((((r18_z / $(16384)) % $(2)) * $(16384)) + (((r18_z / $(4)) % $(4096)) * $(4)) + (r18_z % $(4))), dtype=tl.float32) * tl.load($(vrsqrt_ptr) + ((((r18_z / $(16384)) % $(2)) * $(4096)) + ((r18_z / $(4)) % $(4096))), dtype=tl.float32))
    r18_value = r18_value.to(tl.float32)
    r18_value = tl.broadcast_to(r18_value, [$(32768)])
    tl.store($(vmul_7_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(8192))
    r19_lane = tl.arange($(4))
    r19_value = tl.sum(tl.load($(vmul_6_ptr) + ((((r19_z[:, None] / $(4096)) % $(2)) * $(16384)) + ((r19_z[:, None] % $(4096)) * $(4)) + r19_lane[None, :]), dtype=tl.float32), axis=1)
    r19_value = r19_value.to(tl.float32)
    r19_value = tl.broadcast_to(r19_value, [$(8192)])
    tl.store($(vsum_3_ptr) + r19_z, r19_value)
    r20_row = tl.arange($(8192))
    r20_col = tl.arange($(8192))
    r20_lane = tl.arange($(4))
    r20_left = tl.load($(vmul_7_ptr) + ((((((r20_row[:, None] * $(4)) + r20_lane[None, :]) / $(16384)) % $(2)) * $(16384)) + (((((r20_row[:, None] * $(4)) + r20_lane[None, :]) / $(4)) % $(4096)) * $(4)) + (((r20_row[:, None] * $(4)) + r20_lane[None, :]) % $(4))), dtype=tl.float32)
    r20_right = tl.load($(input1_ptr) + (((r20_lane[:, None] % $(4)) * $(8192)) + (r20_col[None, :] % $(8192))), dtype=tl.float32)
    r20_value = tl.dot(r20_left, r20_right)
    r20_value = r20_value.to(tl.float32)
    tl.store($(vmm_1_ptr) + ((r20_row[:, None] * $(8192)) + r20_col[None, :]), r20_value)
    r21_z = tl.arange($(67108864))
    r21_value = (tl.load($(vmul_3_ptr) + ((((r21_z / $(33554432)) % $(2)) * $(33554432)) + (((r21_z / $(8192)) % $(4096)) * $(8192)) + (((((((r21_z / $(33554432)) % $(2)) * $(33554432)) + (((r21_z / $(8192)) % $(4096)) * $(8192)) + (r21_z % $(8192))) / $(2048)) % $(4)) * $(2048)) + (((((r21_z / $(33554432)) % $(2)) * $(33554432)) + (((r21_z / $(8192)) % $(4096)) * $(8192)) + (r21_z % $(8192))) % $(2048))), dtype=tl.float32) + tl.load($(vmm_1_ptr) + ((((r21_z / $(33554432)) % $(2)) * $(33554432)) + (((r21_z / $(8192)) % $(4096)) * $(8192)) + (r21_z % $(8192))), dtype=tl.float32))
    r21_value = r21_value.to(tl.float32)
    r21_value = tl.broadcast_to(r21_value, [$(67108864)])
    tl.store($(vadd_3_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(8192))
    r22_value = ((tl.load($(vrsqrt_ptr) + ((((r22_z / $(4096)) % $(2)) * $(4096)) + (r22_z % $(4096))), dtype=tl.float32)) * (tl.load($(vrsqrt_ptr) + ((((r22_z / $(4096)) % $(2)) * $(4096)) + (r22_z % $(4096))), dtype=tl.float32)) * (tl.load($(vrsqrt_ptr) + ((((r22_z / $(4096)) % $(2)) * $(4096)) + (r22_z % $(4096))), dtype=tl.float32)))
    r22_value = r22_value.to(tl.float32)
    r22_value = tl.broadcast_to(r22_value, [$(8192)])
    tl.store($(vpow_2_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(8192))
    r23_value = (tl.load($(vsum_3_ptr) + ((((r23_z / $(4096)) % $(2)) * $(4096)) + (r23_z % $(4096))), dtype=tl.float32) * -0.5)
    r23_value = r23_value.to(tl.float32)
    r23_value = tl.broadcast_to(r23_value, [$(8192)])
    tl.store($(vmul_8_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(8192))
    r24_value = (tl.load($(vmul_8_ptr) + ((((r24_z / $(4096)) % $(2)) * $(4096)) + (r24_z % $(4096))), dtype=tl.float32) * tl.load($(vpow_2_ptr) + ((((r24_z / $(4096)) % $(2)) * $(4096)) + (r24_z % $(4096))), dtype=tl.float32))
    r24_value = r24_value.to(tl.float32)
    r24_value = tl.broadcast_to(r24_value, [$(8192)])
    tl.store($(vmul_9_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(67108864))
    r25_value = (tl.load($(vmul_9_ptr) + ((((r25_z / $(33554432)) % $(2)) * $(4096)) + ((r25_z / $(8192)) % $(4096))), dtype=tl.float32) / 8192.0)
    r25_value = r25_value.to(tl.float32)
    r25_value = tl.broadcast_to(r25_value, [$(67108864)])
    tl.store($(vdiv_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(67108864))
    r26_value = ((tl.load($(vto_copy_ptr) + ((((r26_z / $(33554432)) % $(2)) * $(33554432)) + (((r26_z / $(8192)) % $(4096)) * $(8192)) + (r26_z % $(8192))), dtype=tl.float32)))
    r26_value = r26_value.to(tl.float32)
    r26_value = tl.broadcast_to(r26_value, [$(67108864)])
    tl.store($(vpow_3_ptr) + r26_z, r26_value)
    r27_z = tl.arange($(67108864))
    r27_value = (tl.load($(vpow_3_ptr) + ((((r27_z / $(33554432)) % $(2)) * $(33554432)) + (((r27_z / $(8192)) % $(4096)) * $(8192)) + (r27_z % $(8192))), dtype=tl.float32) * 2.0)
    r27_value = r27_value.to(tl.float32)
    r27_value = tl.broadcast_to(r27_value, [$(67108864)])
    tl.store($(vmul_10_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(67108864))
    r28_value = (tl.load($(vdiv_ptr) + ((((r28_z / $(33554432)) % $(2)) * $(33554432)) + (((r28_z / $(8192)) % $(4096)) * $(8192)) + (r28_z % $(8192))), dtype=tl.float32) * tl.load($(vmul_10_ptr) + ((((r28_z / $(33554432)) % $(2)) * $(33554432)) + (((r28_z / $(8192)) % $(4096)) * $(8192)) + (r28_z % $(8192))), dtype=tl.float32))
    r28_value = r28_value.to(tl.float32)
    r28_value = tl.broadcast_to(r28_value, [$(67108864)])
    tl.store($(vmul_11_ptr) + r28_z, r28_value)
    r29_z = tl.arange($(67108864))
    r29_value = (tl.load($(vadd_3_ptr) + ((((r29_z / $(33554432)) % $(2)) * $(33554432)) + (((r29_z / $(8192)) % $(4096)) * $(8192)) + (r29_z % $(8192))), dtype=tl.float32) + tl.load($(vmul_11_ptr) + ((((r29_z / $(33554432)) % $(2)) * $(33554432)) + (((r29_z / $(8192)) % $(4096)) * $(8192)) + (r29_z % $(8192))), dtype=tl.float32))
    r29_value = r29_value.to(tl.float32)
    r29_value = tl.broadcast_to(r29_value, [$(67108864)])
    tl.store($(vadd_4_ptr) + r29_z, r29_value)
    r30_z = tl.arange($(67108864))
    r30_value = tl.load($(vadd_4_ptr) + ((((r30_z / $(33554432)) % $(2)) * $(33554432)) + (((r30_z / $(8192)) % $(4096)) * $(8192)) + (r30_z % $(8192))), dtype=tl.float32)
    r30_value = r30_value.to(tl.bfloat16)
    r30_value = tl.broadcast_to(r30_value, [$(67108864)])
    tl.store($(vto_copy_3_ptr) + r30_z, r30_value)
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
  let vto_copy_2_ptr : RegionName := "vto_copy_2"
  let vmul_3_ptr : RegionName := "vmul_3"
  let vmul_4_ptr : RegionName := "vmul_4"
  let vsum_2_ptr : RegionName := "vsum_2"
  let vsigmoid_backward_ptr : RegionName := "vsigmoid_backward"
  let vmul_5_ptr : RegionName := "vmul_5"
  let vmul_6_ptr : RegionName := "vmul_6"
  let vmul_7_ptr : RegionName := "vmul_7"
  let vsum_3_ptr : RegionName := "vsum_3"
  let vmm_1_ptr : RegionName := "vmm_1"
  let vadd_3_ptr : RegionName := "vadd_3"
  let vpow_2_ptr : RegionName := "vpow_2"
  let vmul_8_ptr : RegionName := "vmul_8"
  let vmul_9_ptr : RegionName := "vmul_9"
  let vdiv_ptr : RegionName := "vdiv"
  let vpow_3_ptr : RegionName := "vpow_3"
  let vmul_10_ptr : RegionName := "vmul_10"
  let vmul_11_ptr : RegionName := "vmul_11"
  let vadd_4_ptr : RegionName := "vadd_4"
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
  let candidate_vto_copy_2_ptr : RegionName := "candidate_vto_copy_2"
  let candidate_vmul_3_ptr : RegionName := "candidate_vmul_3"
  let candidate_vmul_4_ptr : RegionName := "candidate_vmul_4"
  let candidate_vsum_2_ptr : RegionName := "candidate_vsum_2"
  let candidate_vsigmoid_backward_ptr : RegionName := "candidate_vsigmoid_backward"
  let candidate_vmul_5_ptr : RegionName := "candidate_vmul_5"
  let candidate_vmul_6_ptr : RegionName := "candidate_vmul_6"
  let candidate_vmul_7_ptr : RegionName := "candidate_vmul_7"
  let candidate_vsum_3_ptr : RegionName := "candidate_vsum_3"
  let candidate_vmm_1_ptr : RegionName := "candidate_vmm_1"
  let candidate_vadd_3_ptr : RegionName := "candidate_vadd_3"
  let candidate_vpow_2_ptr : RegionName := "candidate_vpow_2"
  let candidate_vmul_8_ptr : RegionName := "candidate_vmul_8"
  let candidate_vmul_9_ptr : RegionName := "candidate_vmul_9"
  let candidate_vdiv_ptr : RegionName := "candidate_vdiv"
  let candidate_vpow_3_ptr : RegionName := "candidate_vpow_3"
  let candidate_vmul_10_ptr : RegionName := "candidate_vmul_10"
  let candidate_vmul_11_ptr : RegionName := "candidate_vmul_11"
  let candidate_vadd_4_ptr : RegionName := "candidate_vadd_4"
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
    r5_col = tl.arange($(4))
    r5_lane = tl.arange($(8192))
    r5_left = tl.load($(candidate_vto_copy_ptr) + ((((((r5_row[:, None] * $(8192)) + r5_lane[None, :]) / $(33554432)) % $(2)) * $(33554432)) + (((((r5_row[:, None] * $(8192)) + r5_lane[None, :]) / $(8192)) % $(4096)) * $(8192)) + (((r5_row[:, None] * $(8192)) + r5_lane[None, :]) % $(8192))), dtype=tl.float32)
    r5_right = tl.load($(input1_ptr) + (((r5_col[None, :] % $(4)) * $(8192)) + (r5_lane[:, None] % $(8192))), dtype=tl.float32)
    r5_value = tl.dot(r5_left, r5_right)
    r5_value = r5_value.to(tl.float32)
    tl.store($(candidate_vmm_ptr) + ((r5_row[:, None] * $(4)) + r5_col[None, :]), r5_value)
    r6_z = tl.arange($(32768))
    r6_value = (tl.load($(candidate_vmm_ptr) + ((((r6_z / $(16384)) % $(2)) * $(16384)) + (((r6_z / $(4)) % $(4096)) * $(4)) + (r6_z % $(4))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_ptr) + ((((r6_z / $(16384)) % $(2)) * $(4096)) + ((r6_z / $(4)) % $(4096))), dtype=tl.float32))
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(32768)])
    tl.store($(candidate_vmul_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(32768))
    r7_value = (tl.load($(candidate_vmul_ptr) + ((((r7_z / $(16384)) % $(2)) * $(16384)) + (((r7_z / $(4)) % $(4096)) * $(4)) + (r7_z % $(4))), dtype=tl.float32) * tl.load($(input2_ptr) + $(0), dtype=tl.float32))
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
    r11_z = tl.arange($(16777216))
    r11_value = tl.load($(input4_ptr) + ((((r11_z / $(8388608)) % $(2)) * $(8388608)) + (((r11_z / $(2048)) % $(4096)) * $(2048)) + (r11_z % $(2048))), dtype=tl.bfloat16)
    r11_value = r11_value.to(tl.float32)
    r11_value = tl.broadcast_to(r11_value, [$(16777216)])
    tl.store($(candidate_vto_copy_2_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(67108864))
    r12_value = (tl.load($(candidate_vto_copy_2_ptr) + ((((r12_z / $(33554432)) % $(2)) * $(8388608)) + (((r12_z / $(8192)) % $(4096)) * $(2048)) + (r12_z % $(2048))), dtype=tl.float32) * tl.load($(candidate_vadd_2_ptr) + ((((r12_z / $(33554432)) % $(2)) * $(16384)) + (((r12_z / $(8192)) % $(4096)) * $(4)) + ((r12_z / $(2048)) % $(4))), dtype=tl.float32))
    r12_value = r12_value.to(tl.float32)
    r12_value = tl.broadcast_to(r12_value, [$(67108864)])
    tl.store($(candidate_vmul_3_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(67108864))
    r13_value = (tl.load($(candidate_vto_copy_2_ptr) + ((((r13_z / $(33554432)) % $(2)) * $(8388608)) + (((r13_z / $(8192)) % $(4096)) * $(2048)) + (r13_z % $(2048))), dtype=tl.float32) * tl.load($(candidate_vto_copy_ptr) + ((((r13_z / $(33554432)) % $(2)) * $(33554432)) + (((r13_z / $(8192)) % $(4096)) * $(8192)) + (((r13_z / $(2048)) % $(4)) * $(2048)) + (r13_z % $(2048))), dtype=tl.float32))
    r13_value = r13_value.to(tl.float32)
    r13_value = tl.broadcast_to(r13_value, [$(67108864)])
    tl.store($(candidate_vmul_4_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(32768))
    r14_lane = tl.arange($(2048))
    r14_value = tl.sum(tl.load($(candidate_vmul_4_ptr) + ((((r14_z[:, None] / $(16384)) % $(2)) * $(33554432)) + (((r14_z[:, None] / $(4)) % $(4096)) * $(8192)) + ((r14_z[:, None] % $(4)) * $(2048)) + r14_lane[None, :]), dtype=tl.float32), axis=1)
    r14_value = r14_value.to(tl.float32)
    r14_value = tl.broadcast_to(r14_value, [$(32768)])
    tl.store($(candidate_vsum_2_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(32768))
    r15_value = (tl.load($(candidate_vsum_2_ptr) + ((((r15_z / $(16384)) % $(2)) * $(16384)) + (((r15_z / $(4)) % $(4096)) * $(4)) + (r15_z % $(4))), dtype=tl.float32) * tl.load($(candidate_vsigmoid_ptr) + ((((r15_z / $(16384)) % $(2)) * $(16384)) + (((r15_z / $(4)) % $(4096)) * $(4)) + (r15_z % $(4))), dtype=tl.float32) * ($(1) - tl.load($(candidate_vsigmoid_ptr) + ((((r15_z / $(16384)) % $(2)) * $(16384)) + (((r15_z / $(4)) % $(4096)) * $(4)) + (r15_z % $(4))), dtype=tl.float32)))
    r15_value = r15_value.to(tl.float32)
    r15_value = tl.broadcast_to(r15_value, [$(32768)])
    tl.store($(candidate_vsigmoid_backward_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(32768))
    r16_value = (tl.load($(candidate_vsigmoid_backward_ptr) + ((((r16_z / $(16384)) % $(2)) * $(16384)) + (((r16_z / $(4)) % $(4096)) * $(4)) + (r16_z % $(4))), dtype=tl.float32) * tl.load($(input2_ptr) + $(0), dtype=tl.float32))
    r16_value = r16_value.to(tl.float32)
    r16_value = tl.broadcast_to(r16_value, [$(32768)])
    tl.store($(candidate_vmul_5_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(32768))
    r17_value = (tl.load($(candidate_vmul_5_ptr) + ((((r17_z / $(16384)) % $(2)) * $(16384)) + (((r17_z / $(4)) % $(4096)) * $(4)) + (r17_z % $(4))), dtype=tl.float32) * tl.load($(candidate_vmm_ptr) + ((((r17_z / $(16384)) % $(2)) * $(16384)) + (((r17_z / $(4)) % $(4096)) * $(4)) + (r17_z % $(4))), dtype=tl.float32))
    r17_value = r17_value.to(tl.float32)
    r17_value = tl.broadcast_to(r17_value, [$(32768)])
    tl.store($(candidate_vmul_6_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(32768))
    r18_value = (tl.load($(candidate_vmul_5_ptr) + ((((r18_z / $(16384)) % $(2)) * $(16384)) + (((r18_z / $(4)) % $(4096)) * $(4)) + (r18_z % $(4))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_ptr) + ((((r18_z / $(16384)) % $(2)) * $(4096)) + ((r18_z / $(4)) % $(4096))), dtype=tl.float32))
    r18_value = r18_value.to(tl.float32)
    r18_value = tl.broadcast_to(r18_value, [$(32768)])
    tl.store($(candidate_vmul_7_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(8192))
    r19_lane = tl.arange($(4))
    r19_value = tl.sum(tl.load($(candidate_vmul_6_ptr) + ((((r19_z[:, None] / $(4096)) % $(2)) * $(16384)) + ((r19_z[:, None] % $(4096)) * $(4)) + r19_lane[None, :]), dtype=tl.float32), axis=1)
    r19_value = r19_value.to(tl.float32)
    r19_value = tl.broadcast_to(r19_value, [$(8192)])
    tl.store($(candidate_vsum_3_ptr) + r19_z, r19_value)
    r20_row = tl.arange($(8192))
    r20_col = tl.arange($(8192))
    r20_lane = tl.arange($(4))
    r20_left = tl.load($(candidate_vmul_7_ptr) + ((((((r20_row[:, None] * $(4)) + r20_lane[None, :]) / $(16384)) % $(2)) * $(16384)) + (((((r20_row[:, None] * $(4)) + r20_lane[None, :]) / $(4)) % $(4096)) * $(4)) + (((r20_row[:, None] * $(4)) + r20_lane[None, :]) % $(4))), dtype=tl.float32)
    r20_right = tl.load($(input1_ptr) + (((r20_lane[:, None] % $(4)) * $(8192)) + (r20_col[None, :] % $(8192))), dtype=tl.float32)
    r20_value = tl.dot(r20_left, r20_right)
    r20_value = r20_value.to(tl.float32)
    tl.store($(candidate_vmm_1_ptr) + ((r20_row[:, None] * $(8192)) + r20_col[None, :]), r20_value)
    r21_z = tl.arange($(67108864))
    r21_value = (tl.load($(candidate_vmul_3_ptr) + ((((r21_z / $(33554432)) % $(2)) * $(33554432)) + (((r21_z / $(8192)) % $(4096)) * $(8192)) + (((((((r21_z / $(33554432)) % $(2)) * $(33554432)) + (((r21_z / $(8192)) % $(4096)) * $(8192)) + (r21_z % $(8192))) / $(2048)) % $(4)) * $(2048)) + (((((r21_z / $(33554432)) % $(2)) * $(33554432)) + (((r21_z / $(8192)) % $(4096)) * $(8192)) + (r21_z % $(8192))) % $(2048))), dtype=tl.float32) + tl.load($(candidate_vmm_1_ptr) + ((((r21_z / $(33554432)) % $(2)) * $(33554432)) + (((r21_z / $(8192)) % $(4096)) * $(8192)) + (r21_z % $(8192))), dtype=tl.float32))
    r21_value = r21_value.to(tl.float32)
    r21_value = tl.broadcast_to(r21_value, [$(67108864)])
    tl.store($(candidate_vadd_3_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(8192))
    r22_value = ((tl.load($(candidate_vrsqrt_ptr) + ((((r22_z / $(4096)) % $(2)) * $(4096)) + (r22_z % $(4096))), dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_ptr) + ((((r22_z / $(4096)) % $(2)) * $(4096)) + (r22_z % $(4096))), dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_ptr) + ((((r22_z / $(4096)) % $(2)) * $(4096)) + (r22_z % $(4096))), dtype=tl.float32)))
    r22_value = r22_value.to(tl.float32)
    r22_value = tl.broadcast_to(r22_value, [$(8192)])
    tl.store($(candidate_vpow_2_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(8192))
    r23_value = (tl.load($(candidate_vsum_3_ptr) + ((((r23_z / $(4096)) % $(2)) * $(4096)) + (r23_z % $(4096))), dtype=tl.float32) * -0.5)
    r23_value = r23_value.to(tl.float32)
    r23_value = tl.broadcast_to(r23_value, [$(8192)])
    tl.store($(candidate_vmul_8_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(8192))
    r24_value = (tl.load($(candidate_vmul_8_ptr) + ((((r24_z / $(4096)) % $(2)) * $(4096)) + (r24_z % $(4096))), dtype=tl.float32) * tl.load($(candidate_vpow_2_ptr) + ((((r24_z / $(4096)) % $(2)) * $(4096)) + (r24_z % $(4096))), dtype=tl.float32))
    r24_value = r24_value.to(tl.float32)
    r24_value = tl.broadcast_to(r24_value, [$(8192)])
    tl.store($(candidate_vmul_9_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(67108864))
    r25_value = (tl.load($(candidate_vmul_9_ptr) + ((((r25_z / $(33554432)) % $(2)) * $(4096)) + ((r25_z / $(8192)) % $(4096))), dtype=tl.float32) / 8192.0)
    r25_value = r25_value.to(tl.float32)
    r25_value = tl.broadcast_to(r25_value, [$(67108864)])
    tl.store($(candidate_vdiv_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(67108864))
    r26_value = ((tl.load($(candidate_vto_copy_ptr) + ((((r26_z / $(33554432)) % $(2)) * $(33554432)) + (((r26_z / $(8192)) % $(4096)) * $(8192)) + (r26_z % $(8192))), dtype=tl.float32)))
    r26_value = r26_value.to(tl.float32)
    r26_value = tl.broadcast_to(r26_value, [$(67108864)])
    tl.store($(candidate_vpow_3_ptr) + r26_z, r26_value)
    r27_z = tl.arange($(67108864))
    r27_value = (tl.load($(candidate_vpow_3_ptr) + ((((r27_z / $(33554432)) % $(2)) * $(33554432)) + (((r27_z / $(8192)) % $(4096)) * $(8192)) + (r27_z % $(8192))), dtype=tl.float32) * 2.0)
    r27_value = r27_value.to(tl.float32)
    r27_value = tl.broadcast_to(r27_value, [$(67108864)])
    tl.store($(candidate_vmul_10_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(67108864))
    r28_value = (tl.load($(candidate_vdiv_ptr) + ((((r28_z / $(33554432)) % $(2)) * $(33554432)) + (((r28_z / $(8192)) % $(4096)) * $(8192)) + (r28_z % $(8192))), dtype=tl.float32) * tl.load($(candidate_vmul_10_ptr) + ((((r28_z / $(33554432)) % $(2)) * $(33554432)) + (((r28_z / $(8192)) % $(4096)) * $(8192)) + (r28_z % $(8192))), dtype=tl.float32))
    r28_value = r28_value.to(tl.float32)
    r28_value = tl.broadcast_to(r28_value, [$(67108864)])
    tl.store($(candidate_vmul_11_ptr) + r28_z, r28_value)
    r29_z = tl.arange($(67108864))
    r29_value = (tl.load($(candidate_vadd_3_ptr) + ((((r29_z / $(33554432)) % $(2)) * $(33554432)) + (((r29_z / $(8192)) % $(4096)) * $(8192)) + (r29_z % $(8192))), dtype=tl.float32) + tl.load($(candidate_vmul_11_ptr) + ((((r29_z / $(33554432)) % $(2)) * $(33554432)) + (((r29_z / $(8192)) % $(4096)) * $(8192)) + (r29_z % $(8192))), dtype=tl.float32))
    r29_value = r29_value.to(tl.float32)
    r29_value = tl.broadcast_to(r29_value, [$(67108864)])
    tl.store($(candidate_vadd_4_ptr) + r29_z, r29_value)
    r30_z = tl.arange($(67108864))
    r30_value = tl.load($(candidate_vadd_4_ptr) + ((((r30_z / $(33554432)) % $(2)) * $(33554432)) + (((r30_z / $(8192)) % $(4096)) * $(8192)) + (r30_z % $(8192))), dtype=tl.float32)
    r30_value = r30_value.to(tl.bfloat16)
    r30_value = tl.broadcast_to(r30_value, [$(67108864)])
    tl.store($(candidate_vto_copy_3_ptr) + r30_z, r30_value)
    z = tl.arange($(67108864))
    value = tl.load($(candidate_vto_copy_3_ptr) + ((((z / $(33554432)) % $(2)) * $(33554432)) + (((z / $(8192)) % $(4096)) * $(8192)) + (((z / $(2048)) % $(4)) * $(2048)) + (z % $(2048))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(67108864)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_fused_hc_head_dx
