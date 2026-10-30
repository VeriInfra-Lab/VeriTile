import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_mla_fa4_nonabsorb_sink_dx
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
  let parameter3_ptr : RegionName := "parameter3"
  let parameter4_ptr : RegionName := "parameter4"
  let parameter5_ptr : RegionName := "parameter5"
  let parameter6_ptr : RegionName := "parameter6"
  let parameter7_ptr : RegionName := "parameter7"
  let input0_ptr : RegionName := "input0"
  let input1_ptr : RegionName := "input1"
  let out_ptr : RegionName := "out"
  let vmul_ptr : RegionName := "vmul"
  let vmul_1_ptr : RegionName := "vmul_1"
  let vadd_ptr : RegionName := "vadd"
  let vmul_2_ptr : RegionName := "vmul_2"
  let vmul_3_ptr : RegionName := "vmul_3"
  let vmul_4_ptr : RegionName := "vmul_4"
  let vadd_1_ptr : RegionName := "vadd_1"
  let vmul_5_ptr : RegionName := "vmul_5"
  let vmul_6_ptr : RegionName := "vmul_6"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  let vto_copy_2_ptr : RegionName := "vto_copy_2"
  let vto_copy_3_ptr : RegionName := "vto_copy_3"
  let vto_copy_4_ptr : RegionName := "vto_copy_4"
  let vto_copy_5_ptr : RegionName := "vto_copy_5"
  let vto_copy_6_ptr : RegionName := "vto_copy_6"
  let vmul_7_ptr : RegionName := "vmul_7"
  let vto_copy_14_ptr : RegionName := "vto_copy_14"
  let vtriu_ptr : RegionName := "vtriu"
  let vmasked_fill_ptr : RegionName := "vmasked_fill"
  let vmm_ptr : RegionName := "vmm"
  let vto_copy_18_ptr : RegionName := "vto_copy_18"
  let vpow_1_ptr : RegionName := "vpow_1"
  let vmean_ptr : RegionName := "vmean"
  let vadd_2_ptr : RegionName := "vadd_2"
  let vrsqrt_ptr : RegionName := "vrsqrt"
  let vmul_8_ptr : RegionName := "vmul_8"
  let vmul_9_ptr : RegionName := "vmul_9"
  let vto_copy_19_ptr : RegionName := "vto_copy_19"
  let vmm_1_ptr : RegionName := "vmm_1"
  let vmm_2_ptr : RegionName := "vmm_2"
  let vto_copy_20_ptr : RegionName := "vto_copy_20"
  let vpow_2_ptr : RegionName := "vpow_2"
  let vmean_1_ptr : RegionName := "vmean_1"
  let vadd_3_ptr : RegionName := "vadd_3"
  let vrsqrt_1_ptr : RegionName := "vrsqrt_1"
  let vmul_10_ptr : RegionName := "vmul_10"
  let vmul_11_ptr : RegionName := "vmul_11"
  let vto_copy_21_ptr : RegionName := "vto_copy_21"
  let vmm_3_ptr : RegionName := "vmm_3"
  let vmul_12_ptr : RegionName := "vmul_12"
  let vmul_13_ptr : RegionName := "vmul_13"
  let vsub_ptr : RegionName := "vsub"
  let vmul_14_ptr : RegionName := "vmul_14"
  let vmul_15_ptr : RegionName := "vmul_15"
  let vadd_4_ptr : RegionName := "vadd_4"
  let vcat_ptr : RegionName := "vcat"
  let vmul_16_ptr : RegionName := "vmul_16"
  let vmul_17_ptr : RegionName := "vmul_17"
  let vsub_1_ptr : RegionName := "vsub_1"
  let vmul_18_ptr : RegionName := "vmul_18"
  let vmul_19_ptr : RegionName := "vmul_19"
  let vadd_5_ptr : RegionName := "vadd_5"
  let vcat_1_ptr : RegionName := "vcat_1"
  let vcat_2_ptr : RegionName := "vcat_2"
  let vcat_3_ptr : RegionName := "vcat_3"
  let vbmm_ptr : RegionName := "vbmm"
  let vmul_20_ptr : RegionName := "vmul_20"
  let vadd_6_ptr : RegionName := "vadd_6"
  let vcat_4_ptr : RegionName := "vcat_4"
  let vto_copy_22_ptr : RegionName := "vto_copy_22"
  let vsoftmax_ptr : RegionName := "vsoftmax"
  let vto_copy_23_ptr : RegionName := "vto_copy_23"
  let vmm_5_ptr : RegionName := "vmm_5"
  let vbmm_2_ptr : RegionName := "vbmm_2"
  let vbmm_3_ptr : RegionName := "vbmm_3"
  let vslice_backward_ptr : RegionName := "vslice_backward"
  let vto_copy_24_ptr : RegionName := "vto_copy_24"
  let vsoftmax_backward_data_ptr : RegionName := "vsoftmax_backward_data"
  let vto_copy_25_ptr : RegionName := "vto_copy_25"
  let vmul_21_ptr : RegionName := "vmul_21"
  let vbmm_4_ptr : RegionName := "vbmm_4"
  let vbmm_5_ptr : RegionName := "vbmm_5"
  let vsum_1_ptr : RegionName := "vsum_1"
  let vmul_22_ptr : RegionName := "vmul_22"
  let vmul_23_ptr : RegionName := "vmul_23"
  let vneg_ptr : RegionName := "vneg"
  let vmul_24_ptr : RegionName := "vmul_24"
  let vadd_7_ptr : RegionName := "vadd_7"
  let vmul_25_ptr : RegionName := "vmul_25"
  let vadd_8_ptr : RegionName := "vadd_8"
  let vmul_26_ptr : RegionName := "vmul_26"
  let vmul_27_ptr : RegionName := "vmul_27"
  let vneg_1_ptr : RegionName := "vneg_1"
  let vmul_28_ptr : RegionName := "vmul_28"
  let vadd_9_ptr : RegionName := "vadd_9"
  let vmul_29_ptr : RegionName := "vmul_29"
  let vadd_10_ptr : RegionName := "vadd_10"
  let vslice_backward_1_ptr : RegionName := "vslice_backward_1"
  let vslice_backward_2_ptr : RegionName := "vslice_backward_2"
  let vadd_11_ptr : RegionName := "vadd_11"
  let vslice_backward_3_ptr : RegionName := "vslice_backward_3"
  let vslice_backward_4_ptr : RegionName := "vslice_backward_4"
  let vadd_12_ptr : RegionName := "vadd_12"
  let vcat_5_ptr : RegionName := "vcat_5"
  let vcat_6_ptr : RegionName := "vcat_6"
  let vmm_6_ptr : RegionName := "vmm_6"
  let vto_copy_26_ptr : RegionName := "vto_copy_26"
  let vmul_30_ptr : RegionName := "vmul_30"
  let vmul_31_ptr : RegionName := "vmul_31"
  let vmul_32_ptr : RegionName := "vmul_32"
  let vsum_2_ptr : RegionName := "vsum_2"
  let vpow_3_ptr : RegionName := "vpow_3"
  let vmul_33_ptr : RegionName := "vmul_33"
  let vmul_34_ptr : RegionName := "vmul_34"
  let vdiv_ptr : RegionName := "vdiv"
  let vpow_4_ptr : RegionName := "vpow_4"
  let vmul_35_ptr : RegionName := "vmul_35"
  let vmul_36_ptr : RegionName := "vmul_36"
  let vadd_13_ptr : RegionName := "vadd_13"
  let vto_copy_27_ptr : RegionName := "vto_copy_27"
  let vcat_7_ptr : RegionName := "vcat_7"
  let vmm_7_ptr : RegionName := "vmm_7"
  let vmm_8_ptr : RegionName := "vmm_8"
  let vto_copy_28_ptr : RegionName := "vto_copy_28"
  let vmul_37_ptr : RegionName := "vmul_37"
  let vmul_38_ptr : RegionName := "vmul_38"
  let vmul_39_ptr : RegionName := "vmul_39"
  let vsum_3_ptr : RegionName := "vsum_3"
  let vpow_5_ptr : RegionName := "vpow_5"
  let vmul_40_ptr : RegionName := "vmul_40"
  let vmul_41_ptr : RegionName := "vmul_41"
  let vdiv_1_ptr : RegionName := "vdiv_1"
  let vpow_6_ptr : RegionName := "vpow_6"
  let vmul_42_ptr : RegionName := "vmul_42"
  let vmul_43_ptr : RegionName := "vmul_43"
  let vadd_14_ptr : RegionName := "vadd_14"
  let vto_copy_29_ptr : RegionName := "vto_copy_29"
  let vmm_9_ptr : RegionName := "vmm_9"
  let vadd_15_ptr : RegionName := "vadd_15"
  triton {
    r0_z = tl.arange($(131072))
    r0_value = (tl.load($(parameter0_ptr) + ((((r0_z / $(512)) % $(256)) * $(512)) + (r0_z % $(512))), dtype=tl.float32) * 0.04419417382415922)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(131072)])
    tl.store($(vmul_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(256))
    r1_value = (tl.load($(parameter1_ptr) + r1_z, dtype=tl.float32) * 0.1)
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(256)])
    tl.store($(vmul_1_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(256))
    r2_value = (tl.load($(vmul_1_ptr) + r2_z, dtype=tl.float32) + 1.0)
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(256)])
    tl.store($(vadd_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(393216))
    r3_value = (tl.load($(parameter2_ptr) + ((((r3_z / $(256)) % $(1536)) * $(256)) + (r3_z % $(256))), dtype=tl.float32) * 0.0625)
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(393216)])
    tl.store($(vmul_2_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(163840))
    r4_value = (tl.load($(parameter3_ptr) + ((((r4_z / $(512)) % $(320)) * $(512)) + (r4_z % $(512))), dtype=tl.float32) * 0.04419417382415922)
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(163840)])
    tl.store($(vmul_3_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(256))
    r5_value = (tl.load($(parameter4_ptr) + r5_z, dtype=tl.float32) * 0.1)
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(256)])
    tl.store($(vmul_4_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(256))
    r6_value = (tl.load($(vmul_4_ptr) + r6_z, dtype=tl.float32) + 1.0)
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(256)])
    tl.store($(vadd_1_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(524288))
    r7_value = (tl.load($(parameter5_ptr) + ((((r7_z / $(256)) % $(2048)) * $(256)) + (r7_z % $(256))), dtype=tl.float32) * 0.0625)
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(524288)])
    tl.store($(vmul_5_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(524288))
    r8_value = (tl.load($(parameter6_ptr) + ((((r8_z / $(1024)) % $(512)) * $(1024)) + (r8_z % $(1024))), dtype=tl.float32) * 0.03125)
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(524288)])
    tl.store($(vmul_6_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(131072))
    r9_value = tl.load($(vmul_ptr) + ((((r9_z / $(512)) % $(256)) * $(512)) + (r9_z % $(512))), dtype=tl.float32)
    r9_value = r9_value.to(tl.bfloat16)
    r9_value = tl.broadcast_to(r9_value, [$(131072)])
    tl.store($(vto_copy_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(256))
    r10_value = tl.load($(vadd_ptr) + r10_z, dtype=tl.float32)
    r10_value = r10_value.to(tl.bfloat16)
    r10_value = tl.broadcast_to(r10_value, [$(256)])
    tl.store($(vto_copy_1_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(393216))
    r11_value = tl.load($(vmul_2_ptr) + ((((r11_z / $(256)) % $(1536)) * $(256)) + (r11_z % $(256))), dtype=tl.float32)
    r11_value = r11_value.to(tl.bfloat16)
    r11_value = tl.broadcast_to(r11_value, [$(393216)])
    tl.store($(vto_copy_2_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(163840))
    r12_value = tl.load($(vmul_3_ptr) + ((((r12_z / $(512)) % $(320)) * $(512)) + (r12_z % $(512))), dtype=tl.float32)
    r12_value = r12_value.to(tl.bfloat16)
    r12_value = tl.broadcast_to(r12_value, [$(163840)])
    tl.store($(vto_copy_3_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(256))
    r13_value = tl.load($(vadd_1_ptr) + r13_z, dtype=tl.float32)
    r13_value = r13_value.to(tl.bfloat16)
    r13_value = tl.broadcast_to(r13_value, [$(256)])
    tl.store($(vto_copy_4_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(524288))
    r14_value = tl.load($(vmul_5_ptr) + ((((r14_z / $(256)) % $(2048)) * $(256)) + (r14_z % $(256))), dtype=tl.float32)
    r14_value = r14_value.to(tl.bfloat16)
    r14_value = tl.broadcast_to(r14_value, [$(524288)])
    tl.store($(vto_copy_5_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(524288))
    r15_value = tl.load($(vmul_6_ptr) + ((((r15_z / $(1024)) % $(512)) * $(1024)) + (r15_z % $(1024))), dtype=tl.float32)
    r15_value = r15_value.to(tl.bfloat16)
    r15_value = tl.broadcast_to(r15_value, [$(524288)])
    tl.store($(vto_copy_6_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(8))
    r16_value = (tl.load($(parameter7_ptr) + r16_z, dtype=tl.float32) * 0.5)
    r16_value = r16_value.to(tl.float32)
    r16_value = tl.broadcast_to(r16_value, [$(8)])
    tl.store($(vmul_7_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(8))
    r17_value = tl.load($(vmul_7_ptr) + r17_z, dtype=tl.float32)
    r17_value = r17_value.to(tl.bfloat16)
    r17_value = tl.broadcast_to(r17_value, [$(8)])
    tl.store($(vto_copy_14_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(16384))
    r18_value = tl.where((r18_z % $(128)) >= ((r18_z / $(128)) % $(128)) + $(1), true, false)
    r18_value = tl.broadcast_to(r18_value, [$(16384)])
    tl.store($(vtriu_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(32768))
    r19_value = tl.where(tl.load($(vtriu_ptr) + ((((r19_z / $(128)) % $(128)) * $(128)) + (r19_z % $(128))), dtype=tl.int1), tl.cast(-338953138925153547590470800371487866880.0, tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r19_value = r19_value.to(tl.bfloat16)
    r19_value = tl.broadcast_to(r19_value, [$(32768)])
    tl.store($(vmasked_fill_ptr) + r19_z, r19_value)
    r20_row = tl.arange($(256))
    r20_col = tl.arange($(256))
    r20_lane = tl.arange($(512))
    r20_left = tl.load($(input0_ptr) + ((((((r20_row[:, None] * $(512)) + r20_lane[None, :]) / $(65536)) % $(2)) * $(65536)) + (((((r20_row[:, None] * $(512)) + r20_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r20_row[:, None] * $(512)) + r20_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r20_right = tl.load($(vto_copy_ptr) + (((r20_col[None, :] % $(256)) * $(512)) + (r20_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r20_value = tl.dot(r20_left, r20_right)
    r20_value = r20_value.to(tl.bfloat16)
    tl.store($(vmm_ptr) + ((r20_row[:, None] * $(256)) + r20_col[None, :]), r20_value)
    r21_z = tl.arange($(65536))
    r21_value = tl.load($(vmm_ptr) + ((((r21_z / $(32768)) % $(2)) * $(32768)) + (((r21_z / $(256)) % $(128)) * $(256)) + (r21_z % $(256))), dtype=tl.bfloat16)
    r21_value = r21_value.to(tl.float32)
    r21_value = tl.broadcast_to(r21_value, [$(65536)])
    tl.store($(vto_copy_18_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(65536))
    r22_value = ((tl.load($(vto_copy_18_ptr) + ((((r22_z / $(32768)) % $(2)) * $(32768)) + (((r22_z / $(256)) % $(128)) * $(256)) + (r22_z % $(256))), dtype=tl.float32)) * (tl.load($(vto_copy_18_ptr) + ((((r22_z / $(32768)) % $(2)) * $(32768)) + (((r22_z / $(256)) % $(128)) * $(256)) + (r22_z % $(256))), dtype=tl.float32)))
    r22_value = r22_value.to(tl.float32)
    r22_value = tl.broadcast_to(r22_value, [$(65536)])
    tl.store($(vpow_1_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(256))
    r23_lane = tl.arange($(256))
    r23_value = (tl.sum(tl.load($(vpow_1_ptr) + ((((r23_z[:, None] / $(128)) % $(2)) * $(32768)) + ((r23_z[:, None] % $(128)) * $(256)) + r23_lane[None, :]), dtype=tl.float32), axis=1) / 256.0)
    r23_value = r23_value.to(tl.float32)
    r23_value = tl.broadcast_to(r23_value, [$(256)])
    tl.store($(vmean_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(256))
    r24_value = (tl.load($(vmean_ptr) + ((((r24_z / $(128)) % $(2)) * $(128)) + (r24_z % $(128))), dtype=tl.float32) + 0.000001)
    r24_value = r24_value.to(tl.float32)
    r24_value = tl.broadcast_to(r24_value, [$(256)])
    tl.store($(vadd_2_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(256))
    r25_value = tl.rsqrt(tl.load($(vadd_2_ptr) + ((((r25_z / $(128)) % $(2)) * $(128)) + (r25_z % $(128))), dtype=tl.float32))
    r25_value = r25_value.to(tl.float32)
    r25_value = tl.broadcast_to(r25_value, [$(256)])
    tl.store($(vrsqrt_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(65536))
    r26_value = (tl.load($(vto_copy_18_ptr) + ((((r26_z / $(32768)) % $(2)) * $(32768)) + (((r26_z / $(256)) % $(128)) * $(256)) + (r26_z % $(256))), dtype=tl.float32) * tl.load($(vrsqrt_ptr) + ((((r26_z / $(32768)) % $(2)) * $(128)) + ((r26_z / $(256)) % $(128))), dtype=tl.float32))
    r26_value = r26_value.to(tl.float32)
    r26_value = tl.broadcast_to(r26_value, [$(65536)])
    tl.store($(vmul_8_ptr) + r26_z, r26_value)
    r27_z = tl.arange($(65536))
    r27_value = (tl.cast(tl.load($(vto_copy_1_ptr) + (r27_z % $(256)), dtype=tl.bfloat16), tl.float32) * tl.load($(vmul_8_ptr) + ((((r27_z / $(32768)) % $(2)) * $(32768)) + (((r27_z / $(256)) % $(128)) * $(256)) + (r27_z % $(256))), dtype=tl.float32))
    r27_value = r27_value.to(tl.float32)
    r27_value = tl.broadcast_to(r27_value, [$(65536)])
    tl.store($(vmul_9_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(65536))
    r28_value = tl.load($(vmul_9_ptr) + ((((r28_z / $(32768)) % $(2)) * $(32768)) + (((r28_z / $(256)) % $(128)) * $(256)) + (r28_z % $(256))), dtype=tl.float32)
    r28_value = r28_value.to(tl.bfloat16)
    r28_value = tl.broadcast_to(r28_value, [$(65536)])
    tl.store($(vto_copy_19_ptr) + r28_z, r28_value)
    r29_row = tl.arange($(256))
    r29_col = tl.arange($(1536))
    r29_lane = tl.arange($(256))
    r29_left = tl.load($(vto_copy_19_ptr) + ((((((r29_row[:, None] * $(256)) + r29_lane[None, :]) / $(32768)) % $(2)) * $(32768)) + (((((r29_row[:, None] * $(256)) + r29_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r29_row[:, None] * $(256)) + r29_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r29_right = tl.load($(vto_copy_2_ptr) + (((r29_col[None, :] % $(1536)) * $(256)) + (r29_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r29_value = tl.dot(r29_left, r29_right)
    r29_value = r29_value.to(tl.bfloat16)
    tl.store($(vmm_1_ptr) + ((r29_row[:, None] * $(1536)) + r29_col[None, :]), r29_value)
    r30_row = tl.arange($(256))
    r30_col = tl.arange($(320))
    r30_lane = tl.arange($(512))
    r30_left = tl.load($(input0_ptr) + ((((((r30_row[:, None] * $(512)) + r30_lane[None, :]) / $(65536)) % $(2)) * $(65536)) + (((((r30_row[:, None] * $(512)) + r30_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r30_row[:, None] * $(512)) + r30_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r30_right = tl.load($(vto_copy_3_ptr) + (((r30_col[None, :] % $(320)) * $(512)) + (r30_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r30_value = tl.dot(r30_left, r30_right)
    r30_value = r30_value.to(tl.bfloat16)
    tl.store($(vmm_2_ptr) + ((r30_row[:, None] * $(320)) + r30_col[None, :]), r30_value)
    r31_z = tl.arange($(65536))
    r31_value = tl.load($(vmm_2_ptr) + ((((r31_z / $(32768)) % $(2)) * $(40960)) + (((r31_z / $(256)) % $(128)) * $(320)) + (r31_z % $(256))), dtype=tl.bfloat16)
    r31_value = r31_value.to(tl.float32)
    r31_value = tl.broadcast_to(r31_value, [$(65536)])
    tl.store($(vto_copy_20_ptr) + r31_z, r31_value)
    r32_z = tl.arange($(65536))
    r32_value = ((tl.load($(vto_copy_20_ptr) + ((((r32_z / $(32768)) % $(2)) * $(32768)) + (((r32_z / $(256)) % $(128)) * $(256)) + (r32_z % $(256))), dtype=tl.float32)) * (tl.load($(vto_copy_20_ptr) + ((((r32_z / $(32768)) % $(2)) * $(32768)) + (((r32_z / $(256)) % $(128)) * $(256)) + (r32_z % $(256))), dtype=tl.float32)))
    r32_value = r32_value.to(tl.float32)
    r32_value = tl.broadcast_to(r32_value, [$(65536)])
    tl.store($(vpow_2_ptr) + r32_z, r32_value)
    r33_z = tl.arange($(256))
    r33_lane = tl.arange($(256))
    r33_value = (tl.sum(tl.load($(vpow_2_ptr) + ((((r33_z[:, None] / $(128)) % $(2)) * $(32768)) + ((r33_z[:, None] % $(128)) * $(256)) + r33_lane[None, :]), dtype=tl.float32), axis=1) / 256.0)
    r33_value = r33_value.to(tl.float32)
    r33_value = tl.broadcast_to(r33_value, [$(256)])
    tl.store($(vmean_1_ptr) + r33_z, r33_value)
    r34_z = tl.arange($(256))
    r34_value = (tl.load($(vmean_1_ptr) + ((((r34_z / $(128)) % $(2)) * $(128)) + (r34_z % $(128))), dtype=tl.float32) + 0.000001)
    r34_value = r34_value.to(tl.float32)
    r34_value = tl.broadcast_to(r34_value, [$(256)])
    tl.store($(vadd_3_ptr) + r34_z, r34_value)
    r35_z = tl.arange($(256))
    r35_value = tl.rsqrt(tl.load($(vadd_3_ptr) + ((((r35_z / $(128)) % $(2)) * $(128)) + (r35_z % $(128))), dtype=tl.float32))
    r35_value = r35_value.to(tl.float32)
    r35_value = tl.broadcast_to(r35_value, [$(256)])
    tl.store($(vrsqrt_1_ptr) + r35_z, r35_value)
    r36_z = tl.arange($(65536))
    r36_value = (tl.load($(vto_copy_20_ptr) + ((((r36_z / $(32768)) % $(2)) * $(32768)) + (((r36_z / $(256)) % $(128)) * $(256)) + (r36_z % $(256))), dtype=tl.float32) * tl.load($(vrsqrt_1_ptr) + ((((r36_z / $(32768)) % $(2)) * $(128)) + ((r36_z / $(256)) % $(128))), dtype=tl.float32))
    r36_value = r36_value.to(tl.float32)
    r36_value = tl.broadcast_to(r36_value, [$(65536)])
    tl.store($(vmul_10_ptr) + r36_z, r36_value)
    r37_z = tl.arange($(65536))
    r37_value = (tl.cast(tl.load($(vto_copy_4_ptr) + (r37_z % $(256)), dtype=tl.bfloat16), tl.float32) * tl.load($(vmul_10_ptr) + ((((r37_z / $(32768)) % $(2)) * $(32768)) + (((r37_z / $(256)) % $(128)) * $(256)) + (r37_z % $(256))), dtype=tl.float32))
    r37_value = r37_value.to(tl.float32)
    r37_value = tl.broadcast_to(r37_value, [$(65536)])
    tl.store($(vmul_11_ptr) + r37_z, r37_value)
    r38_z = tl.arange($(65536))
    r38_value = tl.load($(vmul_11_ptr) + ((((r38_z / $(32768)) % $(2)) * $(32768)) + (((r38_z / $(256)) % $(128)) * $(256)) + (r38_z % $(256))), dtype=tl.float32)
    r38_value = r38_value.to(tl.bfloat16)
    r38_value = tl.broadcast_to(r38_value, [$(65536)])
    tl.store($(vto_copy_21_ptr) + r38_z, r38_value)
    r39_row = tl.arange($(256))
    r39_col = tl.arange($(2048))
    r39_lane = tl.arange($(256))
    r39_left = tl.load($(vto_copy_21_ptr) + ((((((r39_row[:, None] * $(256)) + r39_lane[None, :]) / $(32768)) % $(2)) * $(32768)) + (((((r39_row[:, None] * $(256)) + r39_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r39_row[:, None] * $(256)) + r39_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r39_right = tl.load($(vto_copy_5_ptr) + (((r39_col[None, :] % $(2048)) * $(256)) + (r39_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r39_value = tl.dot(r39_left, r39_right)
    r39_value = r39_value.to(tl.bfloat16)
    tl.store($(vmm_3_ptr) + ((r39_row[:, None] * $(2048)) + r39_col[None, :]), r39_value)
    r40_z = tl.arange($(65536))
    r40_value = (tl.load($(vmm_1_ptr) + ((((r40_z / $(32768)) % $(2)) * $(196608)) + (((r40_z / $(32)) % $(128)) * $(1536)) + (((r40_z / $(4096)) % $(8)) * $(192)) + ((r40_z % $(32)) * $(2)) + $(128)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r40_value = r40_value.to(tl.bfloat16)
    r40_value = tl.broadcast_to(r40_value, [$(65536)])
    tl.store($(vmul_12_ptr) + r40_z, r40_value)
    r41_z = tl.arange($(65536))
    r41_value = (tl.load($(vmm_1_ptr) + ((((r41_z / $(32768)) % $(2)) * $(196608)) + (((r41_z / $(32)) % $(128)) * $(1536)) + (((r41_z / $(4096)) % $(8)) * $(192)) + ((r41_z % $(32)) * $(2)) + $(129)), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r41_value = r41_value.to(tl.bfloat16)
    r41_value = tl.broadcast_to(r41_value, [$(65536)])
    tl.store($(vmul_13_ptr) + r41_z, r41_value)
    r42_z = tl.arange($(65536))
    r42_value = (tl.load($(vmul_12_ptr) + ((((r42_z / $(32768)) % $(2)) * $(32768)) + (((r42_z / $(4096)) % $(8)) * $(4096)) + (((r42_z / $(32)) % $(128)) * $(32)) + (r42_z % $(32))), dtype=tl.bfloat16) - tl.load($(vmul_13_ptr) + ((((r42_z / $(32768)) % $(2)) * $(32768)) + (((r42_z / $(4096)) % $(8)) * $(4096)) + (((r42_z / $(32)) % $(128)) * $(32)) + (r42_z % $(32))), dtype=tl.bfloat16))
    r42_value = r42_value.to(tl.bfloat16)
    r42_value = tl.broadcast_to(r42_value, [$(65536)])
    tl.store($(vsub_ptr) + r42_z, r42_value)
    r43_z = tl.arange($(65536))
    r43_value = (tl.load($(vmm_1_ptr) + ((((r43_z / $(32768)) % $(2)) * $(196608)) + (((r43_z / $(32)) % $(128)) * $(1536)) + (((r43_z / $(4096)) % $(8)) * $(192)) + ((r43_z % $(32)) * $(2)) + $(129)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r43_value = r43_value.to(tl.bfloat16)
    r43_value = tl.broadcast_to(r43_value, [$(65536)])
    tl.store($(vmul_14_ptr) + r43_z, r43_value)
    r44_z = tl.arange($(65536))
    r44_value = (tl.load($(vmm_1_ptr) + ((((r44_z / $(32768)) % $(2)) * $(196608)) + (((r44_z / $(32)) % $(128)) * $(1536)) + (((r44_z / $(4096)) % $(8)) * $(192)) + ((r44_z % $(32)) * $(2)) + $(128)), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r44_value = r44_value.to(tl.bfloat16)
    r44_value = tl.broadcast_to(r44_value, [$(65536)])
    tl.store($(vmul_15_ptr) + r44_z, r44_value)
    r45_z = tl.arange($(65536))
    r45_value = (tl.load($(vmul_14_ptr) + ((((r45_z / $(32768)) % $(2)) * $(32768)) + (((r45_z / $(4096)) % $(8)) * $(4096)) + (((r45_z / $(32)) % $(128)) * $(32)) + (r45_z % $(32))), dtype=tl.bfloat16) + tl.load($(vmul_15_ptr) + ((((r45_z / $(32768)) % $(2)) * $(32768)) + (((r45_z / $(4096)) % $(8)) * $(4096)) + (((r45_z / $(32)) % $(128)) * $(32)) + (r45_z % $(32))), dtype=tl.bfloat16))
    r45_value = r45_value.to(tl.bfloat16)
    r45_value = tl.broadcast_to(r45_value, [$(65536)])
    tl.store($(vadd_4_ptr) + r45_z, r45_value)
    r46_z = tl.arange($(131072))
    r46_value = tl.where((r46_z % $(64)) < $(32), tl.load($(vsub_ptr) + ((((r46_z / $(65536)) % $(2)) * $(32768)) + (((r46_z / $(8192)) % $(8)) * $(4096)) + (((r46_z / $(64)) % $(128)) * $(32)) + (((r46_z % $(64)) - $(0)) % $(32))), dtype=tl.bfloat16), tl.load($(vadd_4_ptr) + ((((r46_z / $(65536)) % $(2)) * $(32768)) + (((r46_z / $(8192)) % $(8)) * $(4096)) + (((r46_z / $(64)) % $(128)) * $(32)) + (((r46_z % $(64)) - $(32)) % $(32))), dtype=tl.bfloat16))
    r46_value = r46_value.to(tl.bfloat16)
    r46_value = tl.broadcast_to(r46_value, [$(131072)])
    tl.store($(vcat_ptr) + r46_z, r46_value)
    r47_z = tl.arange($(8192))
    r47_value = (tl.load($(vmm_2_ptr) + ((((r47_z / $(4096)) % $(2)) * $(40960)) + (((r47_z / $(32)) % $(128)) * $(320)) + ((r47_z % $(32)) * $(2)) + $(256)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r47_value = r47_value.to(tl.bfloat16)
    r47_value = tl.broadcast_to(r47_value, [$(8192)])
    tl.store($(vmul_16_ptr) + r47_z, r47_value)
    r48_z = tl.arange($(8192))
    r48_value = (tl.load($(vmm_2_ptr) + ((((r48_z / $(4096)) % $(2)) * $(40960)) + (((r48_z / $(32)) % $(128)) * $(320)) + ((r48_z % $(32)) * $(2)) + $(257)), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r48_value = r48_value.to(tl.bfloat16)
    r48_value = tl.broadcast_to(r48_value, [$(8192)])
    tl.store($(vmul_17_ptr) + r48_z, r48_value)
    r49_z = tl.arange($(8192))
    r49_value = (tl.load($(vmul_16_ptr) + ((((r49_z / $(4096)) % $(2)) * $(4096)) + (((r49_z / $(32)) % $(128)) * $(32)) + (r49_z % $(32))), dtype=tl.bfloat16) - tl.load($(vmul_17_ptr) + ((((r49_z / $(4096)) % $(2)) * $(4096)) + (((r49_z / $(32)) % $(128)) * $(32)) + (r49_z % $(32))), dtype=tl.bfloat16))
    r49_value = r49_value.to(tl.bfloat16)
    r49_value = tl.broadcast_to(r49_value, [$(8192)])
    tl.store($(vsub_1_ptr) + r49_z, r49_value)
    r50_z = tl.arange($(8192))
    r50_value = (tl.load($(vmm_2_ptr) + ((((r50_z / $(4096)) % $(2)) * $(40960)) + (((r50_z / $(32)) % $(128)) * $(320)) + ((r50_z % $(32)) * $(2)) + $(257)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r50_value = r50_value.to(tl.bfloat16)
    r50_value = tl.broadcast_to(r50_value, [$(8192)])
    tl.store($(vmul_18_ptr) + r50_z, r50_value)
    r51_z = tl.arange($(8192))
    r51_value = (tl.load($(vmm_2_ptr) + ((((r51_z / $(4096)) % $(2)) * $(40960)) + (((r51_z / $(32)) % $(128)) * $(320)) + ((r51_z % $(32)) * $(2)) + $(256)), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r51_value = r51_value.to(tl.bfloat16)
    r51_value = tl.broadcast_to(r51_value, [$(8192)])
    tl.store($(vmul_19_ptr) + r51_z, r51_value)
    r52_z = tl.arange($(8192))
    r52_value = (tl.load($(vmul_18_ptr) + ((((r52_z / $(4096)) % $(2)) * $(4096)) + (((r52_z / $(32)) % $(128)) * $(32)) + (r52_z % $(32))), dtype=tl.bfloat16) + tl.load($(vmul_19_ptr) + ((((r52_z / $(4096)) % $(2)) * $(4096)) + (((r52_z / $(32)) % $(128)) * $(32)) + (r52_z % $(32))), dtype=tl.bfloat16))
    r52_value = r52_value.to(tl.bfloat16)
    r52_value = tl.broadcast_to(r52_value, [$(8192)])
    tl.store($(vadd_5_ptr) + r52_z, r52_value)
    r53_z = tl.arange($(16384))
    r53_value = tl.where((r53_z % $(64)) < $(32), tl.load($(vsub_1_ptr) + ((((r53_z / $(8192)) % $(2)) * $(4096)) + (((r53_z / $(64)) % $(128)) * $(32)) + (((r53_z % $(64)) - $(0)) % $(32))), dtype=tl.bfloat16), tl.load($(vadd_5_ptr) + ((((r53_z / $(8192)) % $(2)) * $(4096)) + (((r53_z / $(64)) % $(128)) * $(32)) + (((r53_z % $(64)) - $(32)) % $(32))), dtype=tl.bfloat16))
    r53_value = r53_value.to(tl.bfloat16)
    r53_value = tl.broadcast_to(r53_value, [$(16384)])
    tl.store($(vcat_1_ptr) + r53_z, r53_value)
    r54_z = tl.arange($(393216))
    r54_value = tl.where((r54_z % $(192)) < $(128), tl.load($(vmm_1_ptr) + ((((((((r54_z / $(196608)) % $(2)) * $(196608)) + (((r54_z / $(192)) % $(128)) * $(1536)) + (((r54_z / $(24576)) % $(8)) * $(192)) + ((r54_z % $(192)) - $(0))) / $(196608)) % $(2)) * $(196608)) + (((((((r54_z / $(196608)) % $(2)) * $(196608)) + (((r54_z / $(192)) % $(128)) * $(1536)) + (((r54_z / $(24576)) % $(8)) * $(192)) + ((r54_z % $(192)) - $(0))) / $(1536)) % $(128)) * $(1536)) + (((((r54_z / $(196608)) % $(2)) * $(196608)) + (((r54_z / $(192)) % $(128)) * $(1536)) + (((r54_z / $(24576)) % $(8)) * $(192)) + ((r54_z % $(192)) - $(0))) % $(1536))), dtype=tl.bfloat16), tl.load($(vcat_ptr) + ((((r54_z / $(196608)) % $(2)) * $(65536)) + (((r54_z / $(24576)) % $(8)) * $(8192)) + (((r54_z / $(192)) % $(128)) * $(64)) + (((r54_z % $(192)) - $(128)) % $(64))), dtype=tl.bfloat16))
    r54_value = r54_value.to(tl.bfloat16)
    r54_value = tl.broadcast_to(r54_value, [$(393216)])
    tl.store($(vcat_2_ptr) + r54_z, r54_value)
    r55_z = tl.arange($(393216))
    r55_value = tl.where((r55_z % $(192)) < $(128), tl.load($(vmm_3_ptr) + ((((((((r55_z / $(196608)) % $(2)) * $(262144)) + (((r55_z / $(192)) % $(128)) * $(2048)) + (((r55_z / $(24576)) % $(8)) * $(256)) + ((r55_z % $(192)) - $(0))) / $(262144)) % $(2)) * $(262144)) + (((((((r55_z / $(196608)) % $(2)) * $(262144)) + (((r55_z / $(192)) % $(128)) * $(2048)) + (((r55_z / $(24576)) % $(8)) * $(256)) + ((r55_z % $(192)) - $(0))) / $(2048)) % $(128)) * $(2048)) + (((((r55_z / $(196608)) % $(2)) * $(262144)) + (((r55_z / $(192)) % $(128)) * $(2048)) + (((r55_z / $(24576)) % $(8)) * $(256)) + ((r55_z % $(192)) - $(0))) % $(2048))), dtype=tl.bfloat16), tl.load($(vcat_1_ptr) + ((((r55_z / $(196608)) % $(2)) * $(8192)) + (((r55_z / $(192)) % $(128)) * $(64)) + (((r55_z % $(192)) - $(128)) % $(64))), dtype=tl.bfloat16))
    r55_value = r55_value.to(tl.bfloat16)
    r55_value = tl.broadcast_to(r55_value, [$(393216)])
    tl.store($(vcat_3_ptr) + r55_z, r55_value)
    r56_row = tl.arange($(128))
    r56_col = tl.arange($(128))
    r56_lane = tl.arange($(192))
    for r56_group in range($(16)) {
      r56_left = tl.load($(vcat_2_ptr) + (((((((r56_group % $(16)) * $(24576)) + (r56_row[:, None] * $(192)) + r56_lane[None, :]) / $(196608)) % $(2)) * $(196608)) + ((((((r56_group % $(16)) * $(24576)) + (r56_row[:, None] * $(192)) + r56_lane[None, :]) / $(24576)) % $(8)) * $(24576)) + ((((((r56_group % $(16)) * $(24576)) + (r56_row[:, None] * $(192)) + r56_lane[None, :]) / $(192)) % $(128)) * $(192)) + ((((r56_group % $(16)) * $(24576)) + (r56_row[:, None] * $(192)) + r56_lane[None, :]) % $(192))), dtype=tl.bfloat16)
      r56_right = tl.load($(vcat_3_ptr) + (((((((r56_group % $(16)) * $(24576)) + (r56_lane[:, None] * $(128)) + r56_col[None, :]) / $(196608)) % $(2)) * $(196608)) + ((((((r56_group % $(16)) * $(24576)) + (r56_lane[:, None] * $(128)) + r56_col[None, :]) / $(24576)) % $(8)) * $(24576)) + (((((r56_group % $(16)) * $(24576)) + (r56_lane[:, None] * $(128)) + r56_col[None, :]) % $(128)) * $(192)) + (((((r56_group % $(16)) * $(24576)) + (r56_lane[:, None] * $(128)) + r56_col[None, :]) / $(128)) % $(192))), dtype=tl.bfloat16)
      r56_value = tl.dot(r56_left, r56_right)
      r56_value = r56_value.to(tl.bfloat16)
      tl.store($(vbmm_ptr) + ((r56_group * $(16384)) + (r56_row[:, None] * $(128)) + r56_col[None, :]), r56_value)
    }
    r57_z = tl.arange($(262144))
    r57_value = (tl.load($(vbmm_ptr) + ((((r57_z / $(131072)) % $(2)) * $(131072)) + (((r57_z / $(16384)) % $(8)) * $(16384)) + (((r57_z / $(128)) % $(128)) * $(128)) + (r57_z % $(128))), dtype=tl.bfloat16) * tl.cast(0.07216878364870322, tl.bfloat16))
    r57_value = r57_value.to(tl.bfloat16)
    r57_value = tl.broadcast_to(r57_value, [$(262144)])
    tl.store($(vmul_20_ptr) + r57_z, r57_value)
    r58_z = tl.arange($(262144))
    r58_value = (tl.load($(vmul_20_ptr) + ((((r58_z / $(131072)) % $(2)) * $(131072)) + (((r58_z / $(16384)) % $(8)) * $(16384)) + (((r58_z / $(128)) % $(128)) * $(128)) + (r58_z % $(128))), dtype=tl.bfloat16) + tl.load($(vmasked_fill_ptr) + ((((r58_z / $(131072)) % $(2)) * $(16384)) + (((r58_z / $(128)) % $(128)) * $(128)) + (r58_z % $(128))), dtype=tl.bfloat16))
    r58_value = r58_value.to(tl.bfloat16)
    r58_value = tl.broadcast_to(r58_value, [$(262144)])
    tl.store($(vadd_6_ptr) + r58_z, r58_value)
    r59_z = tl.arange($(264192))
    r59_value = tl.where((r59_z % $(129)) < $(128), tl.load($(vadd_6_ptr) + ((((r59_z / $(132096)) % $(2)) * $(131072)) + (((r59_z / $(16512)) % $(8)) * $(16384)) + (((r59_z / $(129)) % $(128)) * $(128)) + (((r59_z % $(129)) - $(0)) % $(128))), dtype=tl.bfloat16), tl.load($(vto_copy_14_ptr) + ((r59_z / $(16512)) % $(8)), dtype=tl.bfloat16))
    r59_value = r59_value.to(tl.bfloat16)
    r59_value = tl.broadcast_to(r59_value, [$(264192)])
    tl.store($(vcat_4_ptr) + r59_z, r59_value)
    r60_z = tl.arange($(264192))
    r60_value = tl.load($(vcat_4_ptr) + ((((r60_z / $(132096)) % $(2)) * $(132096)) + (((r60_z / $(16512)) % $(8)) * $(16512)) + (((r60_z / $(129)) % $(128)) * $(129)) + (r60_z % $(129))), dtype=tl.bfloat16)
    r60_value = r60_value.to(tl.float32)
    r60_value = tl.broadcast_to(r60_value, [$(264192)])
    tl.store($(vto_copy_22_ptr) + r60_z, r60_value)
    r61_z = tl.arange($(264192))
    r61_lane = tl.arange($(129))
    r61_row_values = tl.load($(vto_copy_22_ptr) + ((((r61_z[:, None] / $(132096)) % $(2)) * $(132096)) + (((r61_z[:, None] / $(16512)) % $(8)) * $(16512)) + (((r61_z[:, None] / $(129)) % $(128)) * $(129)) + (r61_lane[None, :] % $(129))), dtype=tl.float32)
    r61_maximum = tl.max(r61_row_values, axis=1, keep_dims=true)
    r61_weights = tl.exp(r61_row_values - r61_maximum)
    r61_total = tl.sum(r61_weights, axis=1)
    r61_maximum = tl.max(r61_row_values, axis=1)
    r61_value = tl.exp(tl.load($(vto_copy_22_ptr) + ((((r61_z / $(132096)) % $(2)) * $(132096)) + (((r61_z / $(16512)) % $(8)) * $(16512)) + (((r61_z / $(129)) % $(128)) * $(129)) + (r61_z % $(129))), dtype=tl.float32) - r61_maximum) / r61_total
    r61_value = r61_value.to(tl.float32)
    r61_value = tl.broadcast_to(r61_value, [$(264192)])
    tl.store($(vsoftmax_ptr) + r61_z, r61_value)
    r62_z = tl.arange($(264192))
    r62_value = tl.load($(vsoftmax_ptr) + ((((r62_z / $(132096)) % $(2)) * $(132096)) + (((r62_z / $(16512)) % $(8)) * $(16512)) + (((r62_z / $(129)) % $(128)) * $(129)) + (r62_z % $(129))), dtype=tl.float32)
    r62_value = r62_value.to(tl.bfloat16)
    r62_value = tl.broadcast_to(r62_value, [$(264192)])
    tl.store($(vto_copy_23_ptr) + r62_z, r62_value)
    r63_row = tl.arange($(256))
    r63_col = tl.arange($(1024))
    r63_lane = tl.arange($(512))
    r63_left = tl.load($(input1_ptr) + ((((((r63_row[:, None] * $(512)) + r63_lane[None, :]) / $(65536)) % $(2)) * $(65536)) + (((((r63_row[:, None] * $(512)) + r63_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r63_row[:, None] * $(512)) + r63_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r63_right = tl.load($(vto_copy_6_ptr) + (((r63_lane[:, None] % $(512)) * $(1024)) + (r63_col[None, :] % $(1024))), dtype=tl.bfloat16)
    r63_value = tl.dot(r63_left, r63_right)
    r63_value = r63_value.to(tl.bfloat16)
    tl.store($(vmm_5_ptr) + ((r63_row[:, None] * $(1024)) + r63_col[None, :]), r63_value)
    r64_row = tl.arange($(128))
    r64_col = tl.arange($(128))
    r64_lane = tl.arange($(128))
    for r64_group in range($(16)) {
      r64_left = tl.load($(vto_copy_23_ptr) + (((((((r64_group % $(16)) * $(16384)) + (r64_lane[None, :] * $(128)) + r64_row[:, None]) / $(131072)) % $(2)) * $(132096)) + ((((((r64_group % $(16)) * $(16384)) + (r64_lane[None, :] * $(128)) + r64_row[:, None]) / $(16384)) % $(8)) * $(16512)) + ((((((r64_group % $(16)) * $(16384)) + (r64_lane[None, :] * $(128)) + r64_row[:, None]) / $(128)) % $(128)) * $(129)) + ((((r64_group % $(16)) * $(16384)) + (r64_lane[None, :] * $(128)) + r64_row[:, None]) % $(128))), dtype=tl.bfloat16)
      r64_right = tl.load($(vmm_5_ptr) + (((((((r64_group % $(16)) * $(16384)) + (r64_lane[:, None] * $(128)) + r64_col[None, :]) / $(131072)) % $(2)) * $(131072)) + ((((((r64_group % $(16)) * $(16384)) + (r64_lane[:, None] * $(128)) + r64_col[None, :]) / $(128)) % $(128)) * $(1024)) + ((((((r64_group % $(16)) * $(16384)) + (r64_lane[:, None] * $(128)) + r64_col[None, :]) / $(16384)) % $(8)) * $(128)) + ((((r64_group % $(16)) * $(16384)) + (r64_lane[:, None] * $(128)) + r64_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r64_value = tl.dot(r64_left, r64_right)
      r64_value = r64_value.to(tl.bfloat16)
      tl.store($(vbmm_2_ptr) + ((r64_group * $(16384)) + (r64_row[:, None] * $(128)) + r64_col[None, :]), r64_value)
    }
    r65_row = tl.arange($(128))
    r65_col = tl.arange($(128))
    r65_lane = tl.arange($(128))
    for r65_group in range($(16)) {
      r65_left = tl.load($(vmm_5_ptr) + (((((((r65_group % $(16)) * $(16384)) + (r65_row[:, None] * $(128)) + r65_lane[None, :]) / $(131072)) % $(2)) * $(131072)) + ((((((r65_group % $(16)) * $(16384)) + (r65_row[:, None] * $(128)) + r65_lane[None, :]) / $(128)) % $(128)) * $(1024)) + ((((((r65_group % $(16)) * $(16384)) + (r65_row[:, None] * $(128)) + r65_lane[None, :]) / $(16384)) % $(8)) * $(128)) + ((((r65_group % $(16)) * $(16384)) + (r65_row[:, None] * $(128)) + r65_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r65_right = tl.load($(vmm_3_ptr) + (((((((r65_group % $(16)) * $(16384)) + (r65_col[None, :] * $(128)) + r65_lane[:, None]) / $(131072)) % $(2)) * $(262144)) + ((((((r65_group % $(16)) * $(16384)) + (r65_col[None, :] * $(128)) + r65_lane[:, None]) / $(128)) % $(128)) * $(2048)) + ((((((r65_group % $(16)) * $(16384)) + (r65_col[None, :] * $(128)) + r65_lane[:, None]) / $(16384)) % $(8)) * $(256)) + ((((r65_group % $(16)) * $(16384)) + (r65_col[None, :] * $(128)) + r65_lane[:, None]) % $(128)) + $(128)), dtype=tl.bfloat16)
      r65_value = tl.dot(r65_left, r65_right)
      r65_value = r65_value.to(tl.bfloat16)
      tl.store($(vbmm_3_ptr) + ((r65_group * $(16384)) + (r65_row[:, None] * $(128)) + r65_col[None, :]), r65_value)
    }
    r66_z = tl.arange($(264192))
    r66_value = tl.where((((r66_z % $(129)) >= $(0)) & ((r66_z % $(129)) < $(128)) & (((r66_z % $(129)) - $(0)) % $(1) == $(0))), tl.load($(vbmm_3_ptr) + ((((((((r66_z / $(132096)) % $(2)) * $(131072)) + (((r66_z / $(16512)) % $(8)) * $(16384)) + (((r66_z / $(129)) % $(128)) * $(128)) + ((r66_z % $(129)) - $(0))) / $(16384)) % $(16)) * $(16384)) + (((((((r66_z / $(132096)) % $(2)) * $(131072)) + (((r66_z / $(16512)) % $(8)) * $(16384)) + (((r66_z / $(129)) % $(128)) * $(128)) + ((r66_z % $(129)) - $(0))) / $(128)) % $(128)) * $(128)) + (((((r66_z / $(132096)) % $(2)) * $(131072)) + (((r66_z / $(16512)) % $(8)) * $(16384)) + (((r66_z / $(129)) % $(128)) * $(128)) + ((r66_z % $(129)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r66_value = r66_value.to(tl.bfloat16)
    r66_value = tl.broadcast_to(r66_value, [$(264192)])
    tl.store($(vslice_backward_ptr) + r66_z, r66_value)
    r67_z = tl.arange($(264192))
    r67_value = tl.load($(vslice_backward_ptr) + ((((r67_z / $(132096)) % $(2)) * $(132096)) + (((r67_z / $(16512)) % $(8)) * $(16512)) + (((r67_z / $(129)) % $(128)) * $(129)) + (r67_z % $(129))), dtype=tl.bfloat16)
    r67_value = r67_value.to(tl.float32)
    r67_value = tl.broadcast_to(r67_value, [$(264192)])
    tl.store($(vto_copy_24_ptr) + r67_z, r67_value)
    r68_z = tl.arange($(264192))
    r68_lane = tl.arange($(129))
    r68_row_values = tl.load($(vsoftmax_ptr) + ((((r68_z[:, None] / $(132096)) % $(2)) * $(132096)) + (((r68_z[:, None] / $(16512)) % $(8)) * $(16512)) + (((r68_z[:, None] / $(129)) % $(128)) * $(129)) + (r68_lane[None, :] % $(129))), dtype=tl.float32)
    r68_row_grad = tl.load($(vto_copy_24_ptr) + ((((r68_z[:, None] / $(132096)) % $(2)) * $(132096)) + (((r68_z[:, None] / $(16512)) % $(8)) * $(16512)) + (((r68_z[:, None] / $(129)) % $(128)) * $(129)) + (r68_lane[None, :] % $(129))), dtype=tl.float32)
    r68_weighted = r68_row_grad * r68_row_values
    r68_dot_grad = tl.sum(r68_weighted, axis=1)
    r68_value = (tl.load($(vto_copy_24_ptr) + ((((r68_z / $(132096)) % $(2)) * $(132096)) + (((r68_z / $(16512)) % $(8)) * $(16512)) + (((r68_z / $(129)) % $(128)) * $(129)) + (r68_z % $(129))), dtype=tl.float32) - r68_dot_grad) * tl.load($(vsoftmax_ptr) + ((((r68_z / $(132096)) % $(2)) * $(132096)) + (((r68_z / $(16512)) % $(8)) * $(16512)) + (((r68_z / $(129)) % $(128)) * $(129)) + (r68_z % $(129))), dtype=tl.float32)
    r68_value = r68_value.to(tl.float32)
    r68_value = tl.broadcast_to(r68_value, [$(264192)])
    tl.store($(vsoftmax_backward_data_ptr) + r68_z, r68_value)
    r69_z = tl.arange($(264192))
    r69_value = tl.load($(vsoftmax_backward_data_ptr) + ((((r69_z / $(132096)) % $(2)) * $(132096)) + (((r69_z / $(16512)) % $(8)) * $(16512)) + (((r69_z / $(129)) % $(128)) * $(129)) + (r69_z % $(129))), dtype=tl.float32)
    r69_value = r69_value.to(tl.bfloat16)
    r69_value = tl.broadcast_to(r69_value, [$(264192)])
    tl.store($(vto_copy_25_ptr) + r69_z, r69_value)
    r70_z = tl.arange($(262144))
    r70_value = (tl.load($(vto_copy_25_ptr) + ((((r70_z / $(131072)) % $(2)) * $(132096)) + (((r70_z / $(16384)) % $(8)) * $(16512)) + (((r70_z / $(128)) % $(128)) * $(129)) + (r70_z % $(128))), dtype=tl.bfloat16) * tl.cast(0.07216878364870322, tl.bfloat16))
    r70_value = r70_value.to(tl.bfloat16)
    r70_value = tl.broadcast_to(r70_value, [$(262144)])
    tl.store($(vmul_21_ptr) + r70_z, r70_value)
    r71_row = tl.arange($(192))
    r71_col = tl.arange($(128))
    r71_lane = tl.arange($(128))
    for r71_group in range($(16)) {
      r71_left = tl.load($(vcat_2_ptr) + (((((((r71_group % $(16)) * $(24576)) + (r71_lane[None, :] * $(192)) + r71_row[:, None]) / $(196608)) % $(2)) * $(196608)) + ((((((r71_group % $(16)) * $(24576)) + (r71_lane[None, :] * $(192)) + r71_row[:, None]) / $(24576)) % $(8)) * $(24576)) + ((((((r71_group % $(16)) * $(24576)) + (r71_lane[None, :] * $(192)) + r71_row[:, None]) / $(192)) % $(128)) * $(192)) + ((((r71_group % $(16)) * $(24576)) + (r71_lane[None, :] * $(192)) + r71_row[:, None]) % $(192))), dtype=tl.bfloat16)
      r71_right = tl.load($(vmul_21_ptr) + (((((((r71_group % $(16)) * $(16384)) + (r71_lane[:, None] * $(128)) + r71_col[None, :]) / $(131072)) % $(2)) * $(131072)) + ((((((r71_group % $(16)) * $(16384)) + (r71_lane[:, None] * $(128)) + r71_col[None, :]) / $(16384)) % $(8)) * $(16384)) + ((((((r71_group % $(16)) * $(16384)) + (r71_lane[:, None] * $(128)) + r71_col[None, :]) / $(128)) % $(128)) * $(128)) + ((((r71_group % $(16)) * $(16384)) + (r71_lane[:, None] * $(128)) + r71_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r71_value = tl.dot(r71_left, r71_right)
      r71_value = r71_value.to(tl.bfloat16)
      tl.store($(vbmm_4_ptr) + ((r71_group * $(24576)) + (r71_row[:, None] * $(128)) + r71_col[None, :]), r71_value)
    }
    r72_row = tl.arange($(128))
    r72_col = tl.arange($(192))
    r72_lane = tl.arange($(128))
    for r72_group in range($(16)) {
      r72_left = tl.load($(vmul_21_ptr) + (((((((r72_group % $(16)) * $(16384)) + (r72_row[:, None] * $(128)) + r72_lane[None, :]) / $(131072)) % $(2)) * $(131072)) + ((((((r72_group % $(16)) * $(16384)) + (r72_row[:, None] * $(128)) + r72_lane[None, :]) / $(16384)) % $(8)) * $(16384)) + ((((((r72_group % $(16)) * $(16384)) + (r72_row[:, None] * $(128)) + r72_lane[None, :]) / $(128)) % $(128)) * $(128)) + ((((r72_group % $(16)) * $(16384)) + (r72_row[:, None] * $(128)) + r72_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r72_right = tl.load($(vcat_3_ptr) + (((((((r72_group % $(16)) * $(24576)) + (r72_col[None, :] * $(128)) + r72_lane[:, None]) / $(196608)) % $(2)) * $(196608)) + ((((((r72_group % $(16)) * $(24576)) + (r72_col[None, :] * $(128)) + r72_lane[:, None]) / $(24576)) % $(8)) * $(24576)) + (((((r72_group % $(16)) * $(24576)) + (r72_col[None, :] * $(128)) + r72_lane[:, None]) % $(128)) * $(192)) + (((((r72_group % $(16)) * $(24576)) + (r72_col[None, :] * $(128)) + r72_lane[:, None]) / $(128)) % $(192))), dtype=tl.bfloat16)
      r72_value = tl.dot(r72_left, r72_right)
      r72_value = r72_value.to(tl.bfloat16)
      tl.store($(vbmm_5_ptr) + ((r72_group * $(24576)) + (r72_row[:, None] * $(192)) + r72_col[None, :]), r72_value)
    }
    r73_z = tl.arange($(16384))
    r73_lane = tl.arange($(8))
    r73_value = tl.sum(tl.load($(vbmm_4_ptr) + ((((r73_z[:, None] / $(8192)) % $(2)) * $(196608)) + (r73_lane[None, :] * $(24576)) + ((r73_z[:, None] % $(64)) * $(128)) + ((r73_z[:, None] / $(64)) % $(128)) + $(16384)), dtype=tl.bfloat16), axis=1)
    r73_value = r73_value.to(tl.bfloat16)
    r73_value = tl.broadcast_to(r73_value, [$(16384)])
    tl.store($(vsum_1_ptr) + r73_z, r73_value)
    r74_z = tl.arange($(8192))
    r74_value = (tl.load($(vsum_1_ptr) + ((((r74_z / $(4096)) % $(2)) * $(8192)) + (((r74_z / $(32)) % $(128)) * $(64)) + (r74_z % $(32)) + $(32)), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r74_value = r74_value.to(tl.bfloat16)
    r74_value = tl.broadcast_to(r74_value, [$(8192)])
    tl.store($(vmul_22_ptr) + r74_z, r74_value)
    r75_z = tl.arange($(8192))
    r75_value = (tl.load($(vsum_1_ptr) + ((((r75_z / $(4096)) % $(2)) * $(8192)) + (((r75_z / $(32)) % $(128)) * $(64)) + (r75_z % $(32)) + $(32)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r75_value = r75_value.to(tl.bfloat16)
    r75_value = tl.broadcast_to(r75_value, [$(8192)])
    tl.store($(vmul_23_ptr) + r75_z, r75_value)
    r76_z = tl.arange($(8192))
    r76_value = -tl.load($(vsum_1_ptr) + ((((r76_z / $(4096)) % $(2)) * $(8192)) + (((r76_z / $(32)) % $(128)) * $(64)) + (r76_z % $(32))), dtype=tl.bfloat16)
    r76_value = r76_value.to(tl.bfloat16)
    r76_value = tl.broadcast_to(r76_value, [$(8192)])
    tl.store($(vneg_ptr) + r76_z, r76_value)
    r77_z = tl.arange($(8192))
    r77_value = (tl.load($(vneg_ptr) + ((((r77_z / $(4096)) % $(2)) * $(4096)) + (((r77_z / $(32)) % $(128)) * $(32)) + (r77_z % $(32))), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r77_value = r77_value.to(tl.bfloat16)
    r77_value = tl.broadcast_to(r77_value, [$(8192)])
    tl.store($(vmul_24_ptr) + r77_z, r77_value)
    r78_z = tl.arange($(8192))
    r78_value = (tl.load($(vmul_23_ptr) + ((((r78_z / $(4096)) % $(2)) * $(4096)) + (((r78_z / $(32)) % $(128)) * $(32)) + (r78_z % $(32))), dtype=tl.bfloat16) + tl.load($(vmul_24_ptr) + ((((r78_z / $(4096)) % $(2)) * $(4096)) + (((r78_z / $(32)) % $(128)) * $(32)) + (r78_z % $(32))), dtype=tl.bfloat16))
    r78_value = r78_value.to(tl.bfloat16)
    r78_value = tl.broadcast_to(r78_value, [$(8192)])
    tl.store($(vadd_7_ptr) + r78_z, r78_value)
    r79_z = tl.arange($(8192))
    r79_value = (tl.load($(vsum_1_ptr) + ((((r79_z / $(4096)) % $(2)) * $(8192)) + (((r79_z / $(32)) % $(128)) * $(64)) + (r79_z % $(32))), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r79_value = r79_value.to(tl.bfloat16)
    r79_value = tl.broadcast_to(r79_value, [$(8192)])
    tl.store($(vmul_25_ptr) + r79_z, r79_value)
    r80_z = tl.arange($(8192))
    r80_value = (tl.load($(vmul_22_ptr) + ((((r80_z / $(4096)) % $(2)) * $(4096)) + (((r80_z / $(32)) % $(128)) * $(32)) + (r80_z % $(32))), dtype=tl.bfloat16) + tl.load($(vmul_25_ptr) + ((((r80_z / $(4096)) % $(2)) * $(4096)) + (((r80_z / $(32)) % $(128)) * $(32)) + (r80_z % $(32))), dtype=tl.bfloat16))
    r80_value = r80_value.to(tl.bfloat16)
    r80_value = tl.broadcast_to(r80_value, [$(8192)])
    tl.store($(vadd_8_ptr) + r80_z, r80_value)
    r81_z = tl.arange($(65536))
    r81_value = (tl.load($(vbmm_5_ptr) + ((((r81_z / $(32768)) % $(2)) * $(196608)) + (((r81_z / $(4096)) % $(8)) * $(24576)) + (((r81_z / $(32)) % $(128)) * $(192)) + (r81_z % $(32)) + $(160)), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r81_value = r81_value.to(tl.bfloat16)
    r81_value = tl.broadcast_to(r81_value, [$(65536)])
    tl.store($(vmul_26_ptr) + r81_z, r81_value)
    r82_z = tl.arange($(65536))
    r82_value = (tl.load($(vbmm_5_ptr) + ((((r82_z / $(32768)) % $(2)) * $(196608)) + (((r82_z / $(4096)) % $(8)) * $(24576)) + (((r82_z / $(32)) % $(128)) * $(192)) + (r82_z % $(32)) + $(160)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r82_value = r82_value.to(tl.bfloat16)
    r82_value = tl.broadcast_to(r82_value, [$(65536)])
    tl.store($(vmul_27_ptr) + r82_z, r82_value)
    r83_z = tl.arange($(65536))
    r83_value = -tl.load($(vbmm_5_ptr) + ((((r83_z / $(32768)) % $(2)) * $(196608)) + (((r83_z / $(4096)) % $(8)) * $(24576)) + (((r83_z / $(32)) % $(128)) * $(192)) + (r83_z % $(32)) + $(128)), dtype=tl.bfloat16)
    r83_value = r83_value.to(tl.bfloat16)
    r83_value = tl.broadcast_to(r83_value, [$(65536)])
    tl.store($(vneg_1_ptr) + r83_z, r83_value)
    r84_z = tl.arange($(65536))
    r84_value = (tl.load($(vneg_1_ptr) + ((((r84_z / $(32768)) % $(2)) * $(32768)) + (((r84_z / $(4096)) % $(8)) * $(4096)) + (((r84_z / $(32)) % $(128)) * $(32)) + (r84_z % $(32))), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r84_value = r84_value.to(tl.bfloat16)
    r84_value = tl.broadcast_to(r84_value, [$(65536)])
    tl.store($(vmul_28_ptr) + r84_z, r84_value)
    r85_z = tl.arange($(65536))
    r85_value = (tl.load($(vmul_27_ptr) + ((((r85_z / $(32768)) % $(2)) * $(32768)) + (((r85_z / $(4096)) % $(8)) * $(4096)) + (((r85_z / $(32)) % $(128)) * $(32)) + (r85_z % $(32))), dtype=tl.bfloat16) + tl.load($(vmul_28_ptr) + ((((r85_z / $(32768)) % $(2)) * $(32768)) + (((r85_z / $(4096)) % $(8)) * $(4096)) + (((r85_z / $(32)) % $(128)) * $(32)) + (r85_z % $(32))), dtype=tl.bfloat16))
    r85_value = r85_value.to(tl.bfloat16)
    r85_value = tl.broadcast_to(r85_value, [$(65536)])
    tl.store($(vadd_9_ptr) + r85_z, r85_value)
    r86_z = tl.arange($(65536))
    r86_value = (tl.load($(vbmm_5_ptr) + ((((r86_z / $(32768)) % $(2)) * $(196608)) + (((r86_z / $(4096)) % $(8)) * $(24576)) + (((r86_z / $(32)) % $(128)) * $(192)) + (r86_z % $(32)) + $(128)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r86_value = r86_value.to(tl.bfloat16)
    r86_value = tl.broadcast_to(r86_value, [$(65536)])
    tl.store($(vmul_29_ptr) + r86_z, r86_value)
    r87_z = tl.arange($(65536))
    r87_value = (tl.load($(vmul_26_ptr) + ((((r87_z / $(32768)) % $(2)) * $(32768)) + (((r87_z / $(4096)) % $(8)) * $(4096)) + (((r87_z / $(32)) % $(128)) * $(32)) + (r87_z % $(32))), dtype=tl.bfloat16) + tl.load($(vmul_29_ptr) + ((((r87_z / $(32768)) % $(2)) * $(32768)) + (((r87_z / $(4096)) % $(8)) * $(4096)) + (((r87_z / $(32)) % $(128)) * $(32)) + (r87_z % $(32))), dtype=tl.bfloat16))
    r87_value = r87_value.to(tl.bfloat16)
    r87_value = tl.broadcast_to(r87_value, [$(65536)])
    tl.store($(vadd_10_ptr) + r87_z, r87_value)
    r88_z = tl.arange($(16384))
    r88_value = tl.where((((r88_z % $(64)) >= $(1)) & ((r88_z % $(64)) < $(64)) & (((r88_z % $(64)) - $(1)) % $(2) == $(0))), tl.load($(vadd_7_ptr) + ((((r88_z / $(8192)) % $(2)) * $(4096)) + (((r88_z / $(64)) % $(128)) * $(32)) + ((((r88_z % $(64)) - $(1)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r88_value = r88_value.to(tl.bfloat16)
    r88_value = tl.broadcast_to(r88_value, [$(16384)])
    tl.store($(vslice_backward_1_ptr) + r88_z, r88_value)
    r89_z = tl.arange($(16384))
    r89_value = tl.where((((r89_z % $(64)) >= $(0)) & ((r89_z % $(64)) < $(64)) & (((r89_z % $(64)) - $(0)) % $(2) == $(0))), tl.load($(vadd_8_ptr) + ((((r89_z / $(8192)) % $(2)) * $(4096)) + (((r89_z / $(64)) % $(128)) * $(32)) + ((((r89_z % $(64)) - $(0)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r89_value = r89_value.to(tl.bfloat16)
    r89_value = tl.broadcast_to(r89_value, [$(16384)])
    tl.store($(vslice_backward_2_ptr) + r89_z, r89_value)
    r90_z = tl.arange($(16384))
    r90_value = (tl.load($(vslice_backward_1_ptr) + ((((r90_z / $(8192)) % $(2)) * $(8192)) + (((r90_z / $(64)) % $(128)) * $(64)) + (r90_z % $(64))), dtype=tl.bfloat16) + tl.load($(vslice_backward_2_ptr) + ((((r90_z / $(8192)) % $(2)) * $(8192)) + (((r90_z / $(64)) % $(128)) * $(64)) + (r90_z % $(64))), dtype=tl.bfloat16))
    r90_value = r90_value.to(tl.bfloat16)
    r90_value = tl.broadcast_to(r90_value, [$(16384)])
    tl.store($(vadd_11_ptr) + r90_z, r90_value)
    r91_z = tl.arange($(131072))
    r91_value = tl.where((((r91_z % $(64)) >= $(1)) & ((r91_z % $(64)) < $(64)) & (((r91_z % $(64)) - $(1)) % $(2) == $(0))), tl.load($(vadd_9_ptr) + ((((r91_z / $(65536)) % $(2)) * $(32768)) + (((r91_z / $(8192)) % $(8)) * $(4096)) + (((r91_z / $(64)) % $(128)) * $(32)) + ((((r91_z % $(64)) - $(1)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r91_value = r91_value.to(tl.bfloat16)
    r91_value = tl.broadcast_to(r91_value, [$(131072)])
    tl.store($(vslice_backward_3_ptr) + r91_z, r91_value)
    r92_z = tl.arange($(131072))
    r92_value = tl.where((((r92_z % $(64)) >= $(0)) & ((r92_z % $(64)) < $(64)) & (((r92_z % $(64)) - $(0)) % $(2) == $(0))), tl.load($(vadd_10_ptr) + ((((r92_z / $(65536)) % $(2)) * $(32768)) + (((r92_z / $(8192)) % $(8)) * $(4096)) + (((r92_z / $(64)) % $(128)) * $(32)) + ((((r92_z % $(64)) - $(0)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r92_value = r92_value.to(tl.bfloat16)
    r92_value = tl.broadcast_to(r92_value, [$(131072)])
    tl.store($(vslice_backward_4_ptr) + r92_z, r92_value)
    r93_z = tl.arange($(131072))
    r93_value = (tl.load($(vslice_backward_3_ptr) + ((((r93_z / $(65536)) % $(2)) * $(65536)) + (((r93_z / $(8192)) % $(8)) * $(8192)) + (((r93_z / $(64)) % $(128)) * $(64)) + (r93_z % $(64))), dtype=tl.bfloat16) + tl.load($(vslice_backward_4_ptr) + ((((r93_z / $(65536)) % $(2)) * $(65536)) + (((r93_z / $(8192)) % $(8)) * $(8192)) + (((r93_z / $(64)) % $(128)) * $(64)) + (r93_z % $(64))), dtype=tl.bfloat16))
    r93_value = r93_value.to(tl.bfloat16)
    r93_value = tl.broadcast_to(r93_value, [$(131072)])
    tl.store($(vadd_12_ptr) + r93_z, r93_value)
    r94_z = tl.arange($(524288))
    r94_value = tl.where((r94_z % $(256)) < $(128), tl.load($(vbmm_4_ptr) + ((((((((r94_z / $(262144)) % $(2)) * $(196608)) + (((r94_z / $(32768)) % $(8)) * $(24576)) + (((r94_z % $(256)) - $(0)) * $(128)) + ((r94_z / $(256)) % $(128))) / $(24576)) % $(16)) * $(24576)) + ((((((r94_z / $(262144)) % $(2)) * $(1536)) + (((r94_z / $(32768)) % $(8)) * $(192)) + ((r94_z % $(256)) - $(0))) % $(192)) * $(128)) + ((r94_z / $(256)) % $(128))), dtype=tl.bfloat16), tl.load($(vbmm_2_ptr) + ((((((((r94_z / $(262144)) % $(2)) * $(131072)) + (((r94_z / $(32768)) % $(8)) * $(16384)) + (((r94_z / $(256)) % $(128)) * $(128)) + ((r94_z % $(256)) - $(128))) / $(16384)) % $(16)) * $(16384)) + (((((((r94_z / $(262144)) % $(2)) * $(131072)) + (((r94_z / $(32768)) % $(8)) * $(16384)) + (((r94_z / $(256)) % $(128)) * $(128)) + ((r94_z % $(256)) - $(128))) / $(128)) % $(128)) * $(128)) + (((((r94_z / $(262144)) % $(2)) * $(131072)) + (((r94_z / $(32768)) % $(8)) * $(16384)) + (((r94_z / $(256)) % $(128)) * $(128)) + ((r94_z % $(256)) - $(128))) % $(128))), dtype=tl.bfloat16))
    r94_value = r94_value.to(tl.bfloat16)
    r94_value = tl.broadcast_to(r94_value, [$(524288)])
    tl.store($(vcat_5_ptr) + r94_z, r94_value)
    r95_z = tl.arange($(393216))
    r95_value = tl.where((r95_z % $(192)) < $(128), tl.load($(vbmm_5_ptr) + ((((((((r95_z / $(196608)) % $(2)) * $(196608)) + (((r95_z / $(24576)) % $(8)) * $(24576)) + (((r95_z / $(192)) % $(128)) * $(192)) + ((r95_z % $(192)) - $(0))) / $(24576)) % $(16)) * $(24576)) + (((((((r95_z / $(196608)) % $(2)) * $(196608)) + (((r95_z / $(24576)) % $(8)) * $(24576)) + (((r95_z / $(192)) % $(128)) * $(192)) + ((r95_z % $(192)) - $(0))) / $(192)) % $(128)) * $(192)) + (((((r95_z / $(196608)) % $(2)) * $(196608)) + (((r95_z / $(24576)) % $(8)) * $(24576)) + (((r95_z / $(192)) % $(128)) * $(192)) + ((r95_z % $(192)) - $(0))) % $(192))), dtype=tl.bfloat16), tl.load($(vadd_12_ptr) + ((((r95_z / $(196608)) % $(2)) * $(65536)) + (((r95_z / $(24576)) % $(8)) * $(8192)) + (((r95_z / $(192)) % $(128)) * $(64)) + (((r95_z % $(192)) - $(128)) % $(64))), dtype=tl.bfloat16))
    r95_value = r95_value.to(tl.bfloat16)
    r95_value = tl.broadcast_to(r95_value, [$(393216)])
    tl.store($(vcat_6_ptr) + r95_z, r95_value)
    r96_row = tl.arange($(256))
    r96_col = tl.arange($(256))
    r96_lane = tl.arange($(2048))
    r96_left = tl.load($(vcat_5_ptr) + ((((((r96_row[:, None] * $(2048)) + r96_lane[None, :]) / $(262144)) % $(2)) * $(262144)) + (((((((((r96_row[:, None] * $(2048)) + r96_lane[None, :]) / $(262144)) % $(2)) * $(262144)) + (((((r96_row[:, None] * $(2048)) + r96_lane[None, :]) / $(2048)) % $(128)) * $(2048)) + (((r96_row[:, None] * $(2048)) + r96_lane[None, :]) % $(2048))) / $(256)) % $(8)) * $(32768)) + (((((r96_row[:, None] * $(2048)) + r96_lane[None, :]) / $(2048)) % $(128)) * $(256)) + (((((((r96_row[:, None] * $(2048)) + r96_lane[None, :]) / $(262144)) % $(2)) * $(262144)) + (((((r96_row[:, None] * $(2048)) + r96_lane[None, :]) / $(2048)) % $(128)) * $(2048)) + (((r96_row[:, None] * $(2048)) + r96_lane[None, :]) % $(2048))) % $(256))), dtype=tl.bfloat16)
    r96_right = tl.load($(vto_copy_5_ptr) + (((r96_lane[:, None] % $(2048)) * $(256)) + (r96_col[None, :] % $(256))), dtype=tl.bfloat16)
    r96_value = tl.dot(r96_left, r96_right)
    r96_value = r96_value.to(tl.bfloat16)
    tl.store($(vmm_6_ptr) + ((r96_row[:, None] * $(256)) + r96_col[None, :]), r96_value)
    r97_z = tl.arange($(65536))
    r97_value = tl.load($(vmm_6_ptr) + ((((r97_z / $(32768)) % $(2)) * $(32768)) + (((r97_z / $(256)) % $(128)) * $(256)) + (r97_z % $(256))), dtype=tl.bfloat16)
    r97_value = r97_value.to(tl.float32)
    r97_value = tl.broadcast_to(r97_value, [$(65536)])
    tl.store($(vto_copy_26_ptr) + r97_z, r97_value)
    r98_z = tl.arange($(65536))
    r98_value = (tl.load($(vto_copy_26_ptr) + ((((r98_z / $(32768)) % $(2)) * $(32768)) + (((r98_z / $(256)) % $(128)) * $(256)) + (r98_z % $(256))), dtype=tl.float32) * tl.cast(tl.load($(vto_copy_4_ptr) + (r98_z % $(256)), dtype=tl.bfloat16), tl.float32))
    r98_value = r98_value.to(tl.float32)
    r98_value = tl.broadcast_to(r98_value, [$(65536)])
    tl.store($(vmul_30_ptr) + r98_z, r98_value)
    r99_z = tl.arange($(65536))
    r99_value = (tl.load($(vmul_30_ptr) + ((((r99_z / $(32768)) % $(2)) * $(32768)) + (((r99_z / $(256)) % $(128)) * $(256)) + (r99_z % $(256))), dtype=tl.float32) * tl.load($(vto_copy_20_ptr) + ((((r99_z / $(32768)) % $(2)) * $(32768)) + (((r99_z / $(256)) % $(128)) * $(256)) + (r99_z % $(256))), dtype=tl.float32))
    r99_value = r99_value.to(tl.float32)
    r99_value = tl.broadcast_to(r99_value, [$(65536)])
    tl.store($(vmul_31_ptr) + r99_z, r99_value)
    r100_z = tl.arange($(65536))
    r100_value = (tl.load($(vmul_30_ptr) + ((((r100_z / $(32768)) % $(2)) * $(32768)) + (((r100_z / $(256)) % $(128)) * $(256)) + (r100_z % $(256))), dtype=tl.float32) * tl.load($(vrsqrt_1_ptr) + ((((r100_z / $(32768)) % $(2)) * $(128)) + ((r100_z / $(256)) % $(128))), dtype=tl.float32))
    r100_value = r100_value.to(tl.float32)
    r100_value = tl.broadcast_to(r100_value, [$(65536)])
    tl.store($(vmul_32_ptr) + r100_z, r100_value)
    r101_z = tl.arange($(256))
    r101_lane = tl.arange($(256))
    r101_value = tl.sum(tl.load($(vmul_31_ptr) + ((((r101_z[:, None] / $(128)) % $(2)) * $(32768)) + ((r101_z[:, None] % $(128)) * $(256)) + r101_lane[None, :]), dtype=tl.float32), axis=1)
    r101_value = r101_value.to(tl.float32)
    r101_value = tl.broadcast_to(r101_value, [$(256)])
    tl.store($(vsum_2_ptr) + r101_z, r101_value)
    r102_z = tl.arange($(256))
    r102_value = ((tl.load($(vrsqrt_1_ptr) + ((((r102_z / $(128)) % $(2)) * $(128)) + (r102_z % $(128))), dtype=tl.float32)) * (tl.load($(vrsqrt_1_ptr) + ((((r102_z / $(128)) % $(2)) * $(128)) + (r102_z % $(128))), dtype=tl.float32)) * (tl.load($(vrsqrt_1_ptr) + ((((r102_z / $(128)) % $(2)) * $(128)) + (r102_z % $(128))), dtype=tl.float32)))
    r102_value = r102_value.to(tl.float32)
    r102_value = tl.broadcast_to(r102_value, [$(256)])
    tl.store($(vpow_3_ptr) + r102_z, r102_value)
    r103_z = tl.arange($(256))
    r103_value = (tl.load($(vsum_2_ptr) + ((((r103_z / $(128)) % $(2)) * $(128)) + (r103_z % $(128))), dtype=tl.float32) * -0.5)
    r103_value = r103_value.to(tl.float32)
    r103_value = tl.broadcast_to(r103_value, [$(256)])
    tl.store($(vmul_33_ptr) + r103_z, r103_value)
    r104_z = tl.arange($(256))
    r104_value = (tl.load($(vmul_33_ptr) + ((((r104_z / $(128)) % $(2)) * $(128)) + (r104_z % $(128))), dtype=tl.float32) * tl.load($(vpow_3_ptr) + ((((r104_z / $(128)) % $(2)) * $(128)) + (r104_z % $(128))), dtype=tl.float32))
    r104_value = r104_value.to(tl.float32)
    r104_value = tl.broadcast_to(r104_value, [$(256)])
    tl.store($(vmul_34_ptr) + r104_z, r104_value)
    r105_z = tl.arange($(65536))
    r105_value = (tl.load($(vmul_34_ptr) + ((((r105_z / $(32768)) % $(2)) * $(128)) + ((r105_z / $(256)) % $(128))), dtype=tl.float32) / 256.0)
    r105_value = r105_value.to(tl.float32)
    r105_value = tl.broadcast_to(r105_value, [$(65536)])
    tl.store($(vdiv_ptr) + r105_z, r105_value)
    r106_z = tl.arange($(65536))
    r106_value = ((tl.load($(vto_copy_20_ptr) + ((((r106_z / $(32768)) % $(2)) * $(32768)) + (((r106_z / $(256)) % $(128)) * $(256)) + (r106_z % $(256))), dtype=tl.float32)))
    r106_value = r106_value.to(tl.float32)
    r106_value = tl.broadcast_to(r106_value, [$(65536)])
    tl.store($(vpow_4_ptr) + r106_z, r106_value)
    r107_z = tl.arange($(65536))
    r107_value = (tl.load($(vpow_4_ptr) + ((((r107_z / $(32768)) % $(2)) * $(32768)) + (((r107_z / $(256)) % $(128)) * $(256)) + (r107_z % $(256))), dtype=tl.float32) * 2.0)
    r107_value = r107_value.to(tl.float32)
    r107_value = tl.broadcast_to(r107_value, [$(65536)])
    tl.store($(vmul_35_ptr) + r107_z, r107_value)
    r108_z = tl.arange($(65536))
    r108_value = (tl.load($(vdiv_ptr) + ((((r108_z / $(32768)) % $(2)) * $(32768)) + (((r108_z / $(256)) % $(128)) * $(256)) + (r108_z % $(256))), dtype=tl.float32) * tl.load($(vmul_35_ptr) + ((((r108_z / $(32768)) % $(2)) * $(32768)) + (((r108_z / $(256)) % $(128)) * $(256)) + (r108_z % $(256))), dtype=tl.float32))
    r108_value = r108_value.to(tl.float32)
    r108_value = tl.broadcast_to(r108_value, [$(65536)])
    tl.store($(vmul_36_ptr) + r108_z, r108_value)
    r109_z = tl.arange($(65536))
    r109_value = (tl.load($(vmul_32_ptr) + ((((r109_z / $(32768)) % $(2)) * $(32768)) + (((r109_z / $(256)) % $(128)) * $(256)) + (r109_z % $(256))), dtype=tl.float32) + tl.load($(vmul_36_ptr) + ((((r109_z / $(32768)) % $(2)) * $(32768)) + (((r109_z / $(256)) % $(128)) * $(256)) + (r109_z % $(256))), dtype=tl.float32))
    r109_value = r109_value.to(tl.float32)
    r109_value = tl.broadcast_to(r109_value, [$(65536)])
    tl.store($(vadd_13_ptr) + r109_z, r109_value)
    r110_z = tl.arange($(65536))
    r110_value = tl.load($(vadd_13_ptr) + ((((r110_z / $(32768)) % $(2)) * $(32768)) + (((r110_z / $(256)) % $(128)) * $(256)) + (r110_z % $(256))), dtype=tl.float32)
    r110_value = r110_value.to(tl.bfloat16)
    r110_value = tl.broadcast_to(r110_value, [$(65536)])
    tl.store($(vto_copy_27_ptr) + r110_z, r110_value)
    r111_z = tl.arange($(81920))
    r111_value = tl.where((r111_z % $(320)) < $(256), tl.load($(vto_copy_27_ptr) + ((((r111_z / $(40960)) % $(2)) * $(32768)) + (((r111_z / $(320)) % $(128)) * $(256)) + (((r111_z % $(320)) - $(0)) % $(256))), dtype=tl.bfloat16), tl.load($(vadd_11_ptr) + ((((((((r111_z / $(40960)) % $(2)) * $(8192)) + (((r111_z / $(320)) % $(128)) * $(64)) + ((r111_z % $(320)) - $(256))) / $(8192)) % $(2)) * $(8192)) + (((((((r111_z / $(40960)) % $(2)) * $(8192)) + (((r111_z / $(320)) % $(128)) * $(64)) + ((r111_z % $(320)) - $(256))) / $(64)) % $(128)) * $(64)) + (((((r111_z / $(40960)) % $(2)) * $(8192)) + (((r111_z / $(320)) % $(128)) * $(64)) + ((r111_z % $(320)) - $(256))) % $(64))), dtype=tl.bfloat16))
    r111_value = r111_value.to(tl.bfloat16)
    r111_value = tl.broadcast_to(r111_value, [$(81920)])
    tl.store($(vcat_7_ptr) + r111_z, r111_value)
    r112_row = tl.arange($(256))
    r112_col = tl.arange($(512))
    r112_lane = tl.arange($(320))
    r112_left = tl.load($(vcat_7_ptr) + ((((((r112_row[:, None] * $(320)) + r112_lane[None, :]) / $(40960)) % $(2)) * $(40960)) + (((((r112_row[:, None] * $(320)) + r112_lane[None, :]) / $(320)) % $(128)) * $(320)) + (((r112_row[:, None] * $(320)) + r112_lane[None, :]) % $(320))), dtype=tl.bfloat16)
    r112_right = tl.load($(vto_copy_3_ptr) + (((r112_lane[:, None] % $(320)) * $(512)) + (r112_col[None, :] % $(512))), dtype=tl.bfloat16)
    r112_value = tl.dot(r112_left, r112_right)
    r112_value = r112_value.to(tl.bfloat16)
    tl.store($(vmm_7_ptr) + ((r112_row[:, None] * $(512)) + r112_col[None, :]), r112_value)
    r113_row = tl.arange($(256))
    r113_col = tl.arange($(256))
    r113_lane = tl.arange($(1536))
    r113_left = tl.load($(vcat_6_ptr) + ((((((r113_row[:, None] * $(1536)) + r113_lane[None, :]) / $(196608)) % $(2)) * $(196608)) + (((((((((r113_row[:, None] * $(1536)) + r113_lane[None, :]) / $(196608)) % $(2)) * $(196608)) + (((((r113_row[:, None] * $(1536)) + r113_lane[None, :]) / $(1536)) % $(128)) * $(1536)) + (((r113_row[:, None] * $(1536)) + r113_lane[None, :]) % $(1536))) / $(192)) % $(8)) * $(24576)) + (((((r113_row[:, None] * $(1536)) + r113_lane[None, :]) / $(1536)) % $(128)) * $(192)) + (((((((r113_row[:, None] * $(1536)) + r113_lane[None, :]) / $(196608)) % $(2)) * $(196608)) + (((((r113_row[:, None] * $(1536)) + r113_lane[None, :]) / $(1536)) % $(128)) * $(1536)) + (((r113_row[:, None] * $(1536)) + r113_lane[None, :]) % $(1536))) % $(192))), dtype=tl.bfloat16)
    r113_right = tl.load($(vto_copy_2_ptr) + (((r113_lane[:, None] % $(1536)) * $(256)) + (r113_col[None, :] % $(256))), dtype=tl.bfloat16)
    r113_value = tl.dot(r113_left, r113_right)
    r113_value = r113_value.to(tl.bfloat16)
    tl.store($(vmm_8_ptr) + ((r113_row[:, None] * $(256)) + r113_col[None, :]), r113_value)
    r114_z = tl.arange($(65536))
    r114_value = tl.load($(vmm_8_ptr) + ((((r114_z / $(32768)) % $(2)) * $(32768)) + (((r114_z / $(256)) % $(128)) * $(256)) + (r114_z % $(256))), dtype=tl.bfloat16)
    r114_value = r114_value.to(tl.float32)
    r114_value = tl.broadcast_to(r114_value, [$(65536)])
    tl.store($(vto_copy_28_ptr) + r114_z, r114_value)
    r115_z = tl.arange($(65536))
    r115_value = (tl.load($(vto_copy_28_ptr) + ((((r115_z / $(32768)) % $(2)) * $(32768)) + (((r115_z / $(256)) % $(128)) * $(256)) + (r115_z % $(256))), dtype=tl.float32) * tl.cast(tl.load($(vto_copy_1_ptr) + (r115_z % $(256)), dtype=tl.bfloat16), tl.float32))
    r115_value = r115_value.to(tl.float32)
    r115_value = tl.broadcast_to(r115_value, [$(65536)])
    tl.store($(vmul_37_ptr) + r115_z, r115_value)
    r116_z = tl.arange($(65536))
    r116_value = (tl.load($(vmul_37_ptr) + ((((r116_z / $(32768)) % $(2)) * $(32768)) + (((r116_z / $(256)) % $(128)) * $(256)) + (r116_z % $(256))), dtype=tl.float32) * tl.load($(vto_copy_18_ptr) + ((((r116_z / $(32768)) % $(2)) * $(32768)) + (((r116_z / $(256)) % $(128)) * $(256)) + (r116_z % $(256))), dtype=tl.float32))
    r116_value = r116_value.to(tl.float32)
    r116_value = tl.broadcast_to(r116_value, [$(65536)])
    tl.store($(vmul_38_ptr) + r116_z, r116_value)
    r117_z = tl.arange($(65536))
    r117_value = (tl.load($(vmul_37_ptr) + ((((r117_z / $(32768)) % $(2)) * $(32768)) + (((r117_z / $(256)) % $(128)) * $(256)) + (r117_z % $(256))), dtype=tl.float32) * tl.load($(vrsqrt_ptr) + ((((r117_z / $(32768)) % $(2)) * $(128)) + ((r117_z / $(256)) % $(128))), dtype=tl.float32))
    r117_value = r117_value.to(tl.float32)
    r117_value = tl.broadcast_to(r117_value, [$(65536)])
    tl.store($(vmul_39_ptr) + r117_z, r117_value)
    r118_z = tl.arange($(256))
    r118_lane = tl.arange($(256))
    r118_value = tl.sum(tl.load($(vmul_38_ptr) + ((((r118_z[:, None] / $(128)) % $(2)) * $(32768)) + ((r118_z[:, None] % $(128)) * $(256)) + r118_lane[None, :]), dtype=tl.float32), axis=1)
    r118_value = r118_value.to(tl.float32)
    r118_value = tl.broadcast_to(r118_value, [$(256)])
    tl.store($(vsum_3_ptr) + r118_z, r118_value)
    r119_z = tl.arange($(256))
    r119_value = ((tl.load($(vrsqrt_ptr) + ((((r119_z / $(128)) % $(2)) * $(128)) + (r119_z % $(128))), dtype=tl.float32)) * (tl.load($(vrsqrt_ptr) + ((((r119_z / $(128)) % $(2)) * $(128)) + (r119_z % $(128))), dtype=tl.float32)) * (tl.load($(vrsqrt_ptr) + ((((r119_z / $(128)) % $(2)) * $(128)) + (r119_z % $(128))), dtype=tl.float32)))
    r119_value = r119_value.to(tl.float32)
    r119_value = tl.broadcast_to(r119_value, [$(256)])
    tl.store($(vpow_5_ptr) + r119_z, r119_value)
    r120_z = tl.arange($(256))
    r120_value = (tl.load($(vsum_3_ptr) + ((((r120_z / $(128)) % $(2)) * $(128)) + (r120_z % $(128))), dtype=tl.float32) * -0.5)
    r120_value = r120_value.to(tl.float32)
    r120_value = tl.broadcast_to(r120_value, [$(256)])
    tl.store($(vmul_40_ptr) + r120_z, r120_value)
    r121_z = tl.arange($(256))
    r121_value = (tl.load($(vmul_40_ptr) + ((((r121_z / $(128)) % $(2)) * $(128)) + (r121_z % $(128))), dtype=tl.float32) * tl.load($(vpow_5_ptr) + ((((r121_z / $(128)) % $(2)) * $(128)) + (r121_z % $(128))), dtype=tl.float32))
    r121_value = r121_value.to(tl.float32)
    r121_value = tl.broadcast_to(r121_value, [$(256)])
    tl.store($(vmul_41_ptr) + r121_z, r121_value)
    r122_z = tl.arange($(65536))
    r122_value = (tl.load($(vmul_41_ptr) + ((((r122_z / $(32768)) % $(2)) * $(128)) + ((r122_z / $(256)) % $(128))), dtype=tl.float32) / 256.0)
    r122_value = r122_value.to(tl.float32)
    r122_value = tl.broadcast_to(r122_value, [$(65536)])
    tl.store($(vdiv_1_ptr) + r122_z, r122_value)
    r123_z = tl.arange($(65536))
    r123_value = ((tl.load($(vto_copy_18_ptr) + ((((r123_z / $(32768)) % $(2)) * $(32768)) + (((r123_z / $(256)) % $(128)) * $(256)) + (r123_z % $(256))), dtype=tl.float32)))
    r123_value = r123_value.to(tl.float32)
    r123_value = tl.broadcast_to(r123_value, [$(65536)])
    tl.store($(vpow_6_ptr) + r123_z, r123_value)
    r124_z = tl.arange($(65536))
    r124_value = (tl.load($(vpow_6_ptr) + ((((r124_z / $(32768)) % $(2)) * $(32768)) + (((r124_z / $(256)) % $(128)) * $(256)) + (r124_z % $(256))), dtype=tl.float32) * 2.0)
    r124_value = r124_value.to(tl.float32)
    r124_value = tl.broadcast_to(r124_value, [$(65536)])
    tl.store($(vmul_42_ptr) + r124_z, r124_value)
    r125_z = tl.arange($(65536))
    r125_value = (tl.load($(vdiv_1_ptr) + ((((r125_z / $(32768)) % $(2)) * $(32768)) + (((r125_z / $(256)) % $(128)) * $(256)) + (r125_z % $(256))), dtype=tl.float32) * tl.load($(vmul_42_ptr) + ((((r125_z / $(32768)) % $(2)) * $(32768)) + (((r125_z / $(256)) % $(128)) * $(256)) + (r125_z % $(256))), dtype=tl.float32))
    r125_value = r125_value.to(tl.float32)
    r125_value = tl.broadcast_to(r125_value, [$(65536)])
    tl.store($(vmul_43_ptr) + r125_z, r125_value)
    r126_z = tl.arange($(65536))
    r126_value = (tl.load($(vmul_39_ptr) + ((((r126_z / $(32768)) % $(2)) * $(32768)) + (((r126_z / $(256)) % $(128)) * $(256)) + (r126_z % $(256))), dtype=tl.float32) + tl.load($(vmul_43_ptr) + ((((r126_z / $(32768)) % $(2)) * $(32768)) + (((r126_z / $(256)) % $(128)) * $(256)) + (r126_z % $(256))), dtype=tl.float32))
    r126_value = r126_value.to(tl.float32)
    r126_value = tl.broadcast_to(r126_value, [$(65536)])
    tl.store($(vadd_14_ptr) + r126_z, r126_value)
    r127_z = tl.arange($(65536))
    r127_value = tl.load($(vadd_14_ptr) + ((((r127_z / $(32768)) % $(2)) * $(32768)) + (((r127_z / $(256)) % $(128)) * $(256)) + (r127_z % $(256))), dtype=tl.float32)
    r127_value = r127_value.to(tl.bfloat16)
    r127_value = tl.broadcast_to(r127_value, [$(65536)])
    tl.store($(vto_copy_29_ptr) + r127_z, r127_value)
    r128_row = tl.arange($(256))
    r128_col = tl.arange($(512))
    r128_lane = tl.arange($(256))
    r128_left = tl.load($(vto_copy_29_ptr) + ((((((r128_row[:, None] * $(256)) + r128_lane[None, :]) / $(32768)) % $(2)) * $(32768)) + (((((r128_row[:, None] * $(256)) + r128_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r128_row[:, None] * $(256)) + r128_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r128_right = tl.load($(vto_copy_ptr) + (((r128_lane[:, None] % $(256)) * $(512)) + (r128_col[None, :] % $(512))), dtype=tl.bfloat16)
    r128_value = tl.dot(r128_left, r128_right)
    r128_value = r128_value.to(tl.bfloat16)
    tl.store($(vmm_9_ptr) + ((r128_row[:, None] * $(512)) + r128_col[None, :]), r128_value)
    r129_z = tl.arange($(131072))
    r129_value = (tl.load($(vmm_7_ptr) + ((((r129_z / $(65536)) % $(2)) * $(65536)) + (((r129_z / $(512)) % $(128)) * $(512)) + (r129_z % $(512))), dtype=tl.bfloat16) + tl.load($(vmm_9_ptr) + ((((r129_z / $(65536)) % $(2)) * $(65536)) + (((r129_z / $(512)) % $(128)) * $(512)) + (r129_z % $(512))), dtype=tl.bfloat16))
    r129_value = r129_value.to(tl.bfloat16)
    r129_value = tl.broadcast_to(r129_value, [$(131072)])
    tl.store($(vadd_15_ptr) + r129_z, r129_value)
    z = tl.arange($(131072))
    value = tl.load($(vadd_15_ptr) + ((((z / $(65536)) % $(2)) * $(65536)) + (((z / $(512)) % $(128)) * $(512)) + (z % $(512))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(131072)])
    tl.store($(out_ptr) + z, value)
  }

def candidate : ComputeKernel :=
  let parameter0_ptr : RegionName := "parameter0"
  let parameter1_ptr : RegionName := "parameter1"
  let parameter2_ptr : RegionName := "parameter2"
  let parameter3_ptr : RegionName := "parameter3"
  let parameter4_ptr : RegionName := "parameter4"
  let parameter5_ptr : RegionName := "parameter5"
  let parameter6_ptr : RegionName := "parameter6"
  let parameter7_ptr : RegionName := "parameter7"
  let input0_ptr : RegionName := "input0"
  let input1_ptr : RegionName := "input1"
  let out_ptr : RegionName := "out"
  let vmul_ptr : RegionName := "vmul"
  let vmul_1_ptr : RegionName := "vmul_1"
  let vadd_ptr : RegionName := "vadd"
  let vmul_2_ptr : RegionName := "vmul_2"
  let vmul_3_ptr : RegionName := "vmul_3"
  let vmul_4_ptr : RegionName := "vmul_4"
  let vadd_1_ptr : RegionName := "vadd_1"
  let vmul_5_ptr : RegionName := "vmul_5"
  let vmul_6_ptr : RegionName := "vmul_6"
  let vto_copy_ptr : RegionName := "vto_copy"
  let vto_copy_1_ptr : RegionName := "vto_copy_1"
  let vto_copy_2_ptr : RegionName := "vto_copy_2"
  let vto_copy_3_ptr : RegionName := "vto_copy_3"
  let vto_copy_4_ptr : RegionName := "vto_copy_4"
  let vto_copy_5_ptr : RegionName := "vto_copy_5"
  let vto_copy_6_ptr : RegionName := "vto_copy_6"
  let vmul_7_ptr : RegionName := "vmul_7"
  let vto_copy_14_ptr : RegionName := "vto_copy_14"
  let vmm_ptr : RegionName := "vmm"
  let vto_copy_18_ptr : RegionName := "vto_copy_18"
  let vpow_1_ptr : RegionName := "vpow_1"
  let vmean_ptr : RegionName := "vmean"
  let vadd_2_ptr : RegionName := "vadd_2"
  let vrsqrt_ptr : RegionName := "vrsqrt"
  let vmul_8_ptr : RegionName := "vmul_8"
  let vmul_9_ptr : RegionName := "vmul_9"
  let vto_copy_19_ptr : RegionName := "vto_copy_19"
  let vmm_1_ptr : RegionName := "vmm_1"
  let vmm_2_ptr : RegionName := "vmm_2"
  let vto_copy_20_ptr : RegionName := "vto_copy_20"
  let vpow_2_ptr : RegionName := "vpow_2"
  let vmean_1_ptr : RegionName := "vmean_1"
  let vadd_3_ptr : RegionName := "vadd_3"
  let vrsqrt_1_ptr : RegionName := "vrsqrt_1"
  let vmul_10_ptr : RegionName := "vmul_10"
  let vmul_11_ptr : RegionName := "vmul_11"
  let vto_copy_21_ptr : RegionName := "vto_copy_21"
  let vmul_12_ptr : RegionName := "vmul_12"
  let vmul_13_ptr : RegionName := "vmul_13"
  let vsub_ptr : RegionName := "vsub"
  let vmul_14_ptr : RegionName := "vmul_14"
  let vmul_15_ptr : RegionName := "vmul_15"
  let vadd_4_ptr : RegionName := "vadd_4"
  let vcat_ptr : RegionName := "vcat"
  let vmul_16_ptr : RegionName := "vmul_16"
  let vmul_17_ptr : RegionName := "vmul_17"
  let vsub_1_ptr : RegionName := "vsub_1"
  let vmul_18_ptr : RegionName := "vmul_18"
  let vmul_19_ptr : RegionName := "vmul_19"
  let vadd_5_ptr : RegionName := "vadd_5"
  let vcat_1_ptr : RegionName := "vcat_1"
  let vcat_2_ptr : RegionName := "vcat_2"
  let vcat_3_ptr : RegionName := "vcat_3"
  let vbmm_ptr : RegionName := "vbmm"
  let vmul_20_ptr : RegionName := "vmul_20"
  let vadd_6_ptr : RegionName := "vadd_6"
  let vcat_4_ptr : RegionName := "vcat_4"
  let vto_copy_22_ptr : RegionName := "vto_copy_22"
  let vsoftmax_ptr : RegionName := "vsoftmax"
  let vto_copy_23_ptr : RegionName := "vto_copy_23"
  let vmm_5_ptr : RegionName := "vmm_5"
  let vbmm_2_ptr : RegionName := "vbmm_2"
  let vslice_backward_ptr : RegionName := "vslice_backward"
  let vto_copy_24_ptr : RegionName := "vto_copy_24"
  let vsoftmax_backward_data_ptr : RegionName := "vsoftmax_backward_data"
  let vto_copy_25_ptr : RegionName := "vto_copy_25"
  let vmul_21_ptr : RegionName := "vmul_21"
  let vbmm_4_ptr : RegionName := "vbmm_4"
  let vsum_1_ptr : RegionName := "vsum_1"
  let vmul_22_ptr : RegionName := "vmul_22"
  let vmul_23_ptr : RegionName := "vmul_23"
  let vneg_ptr : RegionName := "vneg"
  let vmul_24_ptr : RegionName := "vmul_24"
  let vadd_7_ptr : RegionName := "vadd_7"
  let vmul_25_ptr : RegionName := "vmul_25"
  let vadd_8_ptr : RegionName := "vadd_8"
  let vmul_26_ptr : RegionName := "vmul_26"
  let vmul_27_ptr : RegionName := "vmul_27"
  let vneg_1_ptr : RegionName := "vneg_1"
  let vmul_28_ptr : RegionName := "vmul_28"
  let vadd_9_ptr : RegionName := "vadd_9"
  let vmul_29_ptr : RegionName := "vmul_29"
  let vadd_10_ptr : RegionName := "vadd_10"
  let vslice_backward_1_ptr : RegionName := "vslice_backward_1"
  let vslice_backward_2_ptr : RegionName := "vslice_backward_2"
  let vadd_11_ptr : RegionName := "vadd_11"
  let vslice_backward_3_ptr : RegionName := "vslice_backward_3"
  let vslice_backward_4_ptr : RegionName := "vslice_backward_4"
  let vadd_12_ptr : RegionName := "vadd_12"
  let vcat_5_ptr : RegionName := "vcat_5"
  let vmm_6_ptr : RegionName := "vmm_6"
  let vto_copy_26_ptr : RegionName := "vto_copy_26"
  let vmul_30_ptr : RegionName := "vmul_30"
  let vmul_31_ptr : RegionName := "vmul_31"
  let vmul_32_ptr : RegionName := "vmul_32"
  let vsum_2_ptr : RegionName := "vsum_2"
  let vpow_3_ptr : RegionName := "vpow_3"
  let vmul_33_ptr : RegionName := "vmul_33"
  let vmul_34_ptr : RegionName := "vmul_34"
  let vdiv_ptr : RegionName := "vdiv"
  let vpow_4_ptr : RegionName := "vpow_4"
  let vmul_35_ptr : RegionName := "vmul_35"
  let vmul_36_ptr : RegionName := "vmul_36"
  let vadd_13_ptr : RegionName := "vadd_13"
  let vto_copy_27_ptr : RegionName := "vto_copy_27"
  let vcat_7_ptr : RegionName := "vcat_7"
  let vmm_7_ptr : RegionName := "vmm_7"
  let vto_copy_28_ptr : RegionName := "vto_copy_28"
  let vmul_37_ptr : RegionName := "vmul_37"
  let vmul_38_ptr : RegionName := "vmul_38"
  let vmul_39_ptr : RegionName := "vmul_39"
  let vsum_3_ptr : RegionName := "vsum_3"
  let vpow_5_ptr : RegionName := "vpow_5"
  let vmul_40_ptr : RegionName := "vmul_40"
  let vmul_41_ptr : RegionName := "vmul_41"
  let vdiv_1_ptr : RegionName := "vdiv_1"
  let vpow_6_ptr : RegionName := "vpow_6"
  let vmul_42_ptr : RegionName := "vmul_42"
  let vmul_43_ptr : RegionName := "vmul_43"
  let vadd_14_ptr : RegionName := "vadd_14"
  let vto_copy_29_ptr : RegionName := "vto_copy_29"
  let vadd_15_ptr : RegionName := "vadd_15"
  let candidate_vmul_ptr : RegionName := "candidate_vmul"
  let candidate_vmul_1_ptr : RegionName := "candidate_vmul_1"
  let candidate_vadd_ptr : RegionName := "candidate_vadd"
  let candidate_vmul_2_ptr : RegionName := "candidate_vmul_2"
  let candidate_vmul_3_ptr : RegionName := "candidate_vmul_3"
  let candidate_vmul_4_ptr : RegionName := "candidate_vmul_4"
  let candidate_vadd_1_ptr : RegionName := "candidate_vadd_1"
  let candidate_vmul_5_ptr : RegionName := "candidate_vmul_5"
  let candidate_vmul_6_ptr : RegionName := "candidate_vmul_6"
  let candidate_vto_copy_ptr : RegionName := "candidate_vto_copy"
  let candidate_vto_copy_1_ptr : RegionName := "candidate_vto_copy_1"
  let candidate_vto_copy_2_ptr : RegionName := "candidate_vto_copy_2"
  let candidate_vto_copy_3_ptr : RegionName := "candidate_vto_copy_3"
  let candidate_vto_copy_4_ptr : RegionName := "candidate_vto_copy_4"
  let candidate_vto_copy_5_ptr : RegionName := "candidate_vto_copy_5"
  let candidate_vto_copy_6_ptr : RegionName := "candidate_vto_copy_6"
  let candidate_vmul_7_ptr : RegionName := "candidate_vmul_7"
  let candidate_vto_copy_14_ptr : RegionName := "candidate_vto_copy_14"
  let candidate_vmm_ptr : RegionName := "candidate_vmm"
  let candidate_vto_copy_18_ptr : RegionName := "candidate_vto_copy_18"
  let candidate_vpow_1_ptr : RegionName := "candidate_vpow_1"
  let candidate_vmean_ptr : RegionName := "candidate_vmean"
  let candidate_vadd_2_ptr : RegionName := "candidate_vadd_2"
  let candidate_vrsqrt_ptr : RegionName := "candidate_vrsqrt"
  let candidate_vmul_8_ptr : RegionName := "candidate_vmul_8"
  let candidate_vmul_9_ptr : RegionName := "candidate_vmul_9"
  let candidate_vto_copy_19_ptr : RegionName := "candidate_vto_copy_19"
  let candidate_vmm_1_ptr : RegionName := "candidate_vmm_1"
  let candidate_vmm_2_ptr : RegionName := "candidate_vmm_2"
  let candidate_vto_copy_20_ptr : RegionName := "candidate_vto_copy_20"
  let candidate_vpow_2_ptr : RegionName := "candidate_vpow_2"
  let candidate_vmean_1_ptr : RegionName := "candidate_vmean_1"
  let candidate_vadd_3_ptr : RegionName := "candidate_vadd_3"
  let candidate_vrsqrt_1_ptr : RegionName := "candidate_vrsqrt_1"
  let candidate_vmul_10_ptr : RegionName := "candidate_vmul_10"
  let candidate_vmul_11_ptr : RegionName := "candidate_vmul_11"
  let candidate_vto_copy_21_ptr : RegionName := "candidate_vto_copy_21"
  let candidate_vmul_12_ptr : RegionName := "candidate_vmul_12"
  let candidate_vmul_13_ptr : RegionName := "candidate_vmul_13"
  let candidate_vsub_ptr : RegionName := "candidate_vsub"
  let candidate_vmul_14_ptr : RegionName := "candidate_vmul_14"
  let candidate_vmul_15_ptr : RegionName := "candidate_vmul_15"
  let candidate_vadd_4_ptr : RegionName := "candidate_vadd_4"
  let candidate_vcat_ptr : RegionName := "candidate_vcat"
  let candidate_vmul_16_ptr : RegionName := "candidate_vmul_16"
  let candidate_vmul_17_ptr : RegionName := "candidate_vmul_17"
  let candidate_vsub_1_ptr : RegionName := "candidate_vsub_1"
  let candidate_vmul_18_ptr : RegionName := "candidate_vmul_18"
  let candidate_vmul_19_ptr : RegionName := "candidate_vmul_19"
  let candidate_vadd_5_ptr : RegionName := "candidate_vadd_5"
  let candidate_vcat_1_ptr : RegionName := "candidate_vcat_1"
  let candidate_vbmm_ptr : RegionName := "candidate_vbmm"
  let candidate_vbmm_1_ptr : RegionName := "candidate_vbmm_1"
  let candidate_vcat_2_ptr : RegionName := "candidate_vcat_2"
  let candidate_vcat_3_ptr : RegionName := "candidate_vcat_3"
  let candidate_vto_copy_22_ptr : RegionName := "candidate_vto_copy_22"
  let candidate_vto_copy_23_ptr : RegionName := "candidate_vto_copy_23"
  let candidate_vbmm_2_ptr : RegionName := "candidate_vbmm_2"
  let candidate_vmul_20_ptr : RegionName := "candidate_vmul_20"
  let candidate_vadd_6_ptr : RegionName := "candidate_vadd_6"
  let candidate_vsub_2_ptr : RegionName := "candidate_vsub_2"
  let candidate_vgt_ptr : RegionName := "candidate_vgt"
  let candidate_vmasked_fill_2_ptr : RegionName := "candidate_vmasked_fill_2"
  let candidate_vto_copy_24_ptr : RegionName := "candidate_vto_copy_24"
  let candidate_vcat_4_ptr : RegionName := "candidate_vcat_4"
  let candidate_vsoftmax_ptr : RegionName := "candidate_vsoftmax"
  let candidate_vto_copy_25_ptr : RegionName := "candidate_vto_copy_25"
  let candidate_vto_copy_26_ptr : RegionName := "candidate_vto_copy_26"
  let candidate_vto_copy_27_ptr : RegionName := "candidate_vto_copy_27"
  let candidate_vbmm_4_ptr : RegionName := "candidate_vbmm_4"
  let candidate_vmul_21_ptr : RegionName := "candidate_vmul_21"
  let candidate_vadd_7_ptr : RegionName := "candidate_vadd_7"
  let candidate_vsub_3_ptr : RegionName := "candidate_vsub_3"
  let candidate_vgt_1_ptr : RegionName := "candidate_vgt_1"
  let candidate_vmasked_fill_3_ptr : RegionName := "candidate_vmasked_fill_3"
  let candidate_vto_copy_28_ptr : RegionName := "candidate_vto_copy_28"
  let candidate_vcat_5_ptr : RegionName := "candidate_vcat_5"
  let candidate_vsoftmax_1_ptr : RegionName := "candidate_vsoftmax_1"
  let candidate_vto_copy_29_ptr : RegionName := "candidate_vto_copy_29"
  let candidate_vmm_4_ptr : RegionName := "candidate_vmm_4"
  let candidate_vbmm_6_ptr : RegionName := "candidate_vbmm_6"
  let candidate_vbmm_7_ptr : RegionName := "candidate_vbmm_7"
  let candidate_vto_copy_30_ptr : RegionName := "candidate_vto_copy_30"
  let candidate_vslice_backward_ptr : RegionName := "candidate_vslice_backward"
  let candidate_vsoftmax_backward_data_ptr : RegionName := "candidate_vsoftmax_backward_data"
  let candidate_vmasked_fill_4_ptr : RegionName := "candidate_vmasked_fill_4"
  let candidate_vmul_22_ptr : RegionName := "candidate_vmul_22"
  let candidate_vbmm_8_ptr : RegionName := "candidate_vbmm_8"
  let candidate_vbmm_9_ptr : RegionName := "candidate_vbmm_9"
  let candidate_vslice_backward_1_ptr : RegionName := "candidate_vslice_backward_1"
  let candidate_vto_copy_31_ptr : RegionName := "candidate_vto_copy_31"
  let candidate_vslice_backward_2_ptr : RegionName := "candidate_vslice_backward_2"
  let candidate_vto_copy_32_ptr : RegionName := "candidate_vto_copy_32"
  let candidate_vslice_backward_3_ptr : RegionName := "candidate_vslice_backward_3"
  let candidate_vbmm_10_ptr : RegionName := "candidate_vbmm_10"
  let candidate_vbmm_11_ptr : RegionName := "candidate_vbmm_11"
  let candidate_vto_copy_33_ptr : RegionName := "candidate_vto_copy_33"
  let candidate_vslice_backward_4_ptr : RegionName := "candidate_vslice_backward_4"
  let candidate_vsoftmax_backward_data_1_ptr : RegionName := "candidate_vsoftmax_backward_data_1"
  let candidate_vmasked_fill_5_ptr : RegionName := "candidate_vmasked_fill_5"
  let candidate_vmul_23_ptr : RegionName := "candidate_vmul_23"
  let candidate_vbmm_12_ptr : RegionName := "candidate_vbmm_12"
  let candidate_vbmm_13_ptr : RegionName := "candidate_vbmm_13"
  let candidate_vslice_backward_5_ptr : RegionName := "candidate_vslice_backward_5"
  let candidate_vadd_8_ptr : RegionName := "candidate_vadd_8"
  let candidate_vto_copy_34_ptr : RegionName := "candidate_vto_copy_34"
  let candidate_vslice_backward_6_ptr : RegionName := "candidate_vslice_backward_6"
  let candidate_vadd_9_ptr : RegionName := "candidate_vadd_9"
  let candidate_vto_copy_35_ptr : RegionName := "candidate_vto_copy_35"
  let candidate_vslice_backward_7_ptr : RegionName := "candidate_vslice_backward_7"
  let candidate_vadd_10_ptr : RegionName := "candidate_vadd_10"
  let candidate_vsum_1_ptr : RegionName := "candidate_vsum_1"
  let candidate_vbmm_14_ptr : RegionName := "candidate_vbmm_14"
  let candidate_vbmm_15_ptr : RegionName := "candidate_vbmm_15"
  let candidate_vadd_11_ptr : RegionName := "candidate_vadd_11"
  let candidate_vselect_backward_ptr : RegionName := "candidate_vselect_backward"
  let candidate_vmul_24_ptr : RegionName := "candidate_vmul_24"
  let candidate_vmul_25_ptr : RegionName := "candidate_vmul_25"
  let candidate_vneg_ptr : RegionName := "candidate_vneg"
  let candidate_vmul_26_ptr : RegionName := "candidate_vmul_26"
  let candidate_vadd_12_ptr : RegionName := "candidate_vadd_12"
  let candidate_vmul_27_ptr : RegionName := "candidate_vmul_27"
  let candidate_vadd_13_ptr : RegionName := "candidate_vadd_13"
  let candidate_vmul_28_ptr : RegionName := "candidate_vmul_28"
  let candidate_vmul_29_ptr : RegionName := "candidate_vmul_29"
  let candidate_vneg_1_ptr : RegionName := "candidate_vneg_1"
  let candidate_vmul_30_ptr : RegionName := "candidate_vmul_30"
  let candidate_vadd_14_ptr : RegionName := "candidate_vadd_14"
  let candidate_vmul_31_ptr : RegionName := "candidate_vmul_31"
  let candidate_vadd_15_ptr : RegionName := "candidate_vadd_15"
  let candidate_vslice_backward_8_ptr : RegionName := "candidate_vslice_backward_8"
  let candidate_vslice_backward_9_ptr : RegionName := "candidate_vslice_backward_9"
  let candidate_vadd_16_ptr : RegionName := "candidate_vadd_16"
  let candidate_vslice_backward_10_ptr : RegionName := "candidate_vslice_backward_10"
  let candidate_vslice_backward_11_ptr : RegionName := "candidate_vslice_backward_11"
  let candidate_vadd_17_ptr : RegionName := "candidate_vadd_17"
  let candidate_vcat_7_ptr : RegionName := "candidate_vcat_7"
  let candidate_vto_copy_36_ptr : RegionName := "candidate_vto_copy_36"
  let candidate_vmul_32_ptr : RegionName := "candidate_vmul_32"
  let candidate_vmul_33_ptr : RegionName := "candidate_vmul_33"
  let candidate_vmul_34_ptr : RegionName := "candidate_vmul_34"
  let candidate_vsum_2_ptr : RegionName := "candidate_vsum_2"
  let candidate_vpow_3_ptr : RegionName := "candidate_vpow_3"
  let candidate_vmul_35_ptr : RegionName := "candidate_vmul_35"
  let candidate_vmul_36_ptr : RegionName := "candidate_vmul_36"
  let candidate_vdiv_ptr : RegionName := "candidate_vdiv"
  let candidate_vpow_4_ptr : RegionName := "candidate_vpow_4"
  let candidate_vmul_37_ptr : RegionName := "candidate_vmul_37"
  let candidate_vmul_38_ptr : RegionName := "candidate_vmul_38"
  let candidate_vadd_18_ptr : RegionName := "candidate_vadd_18"
  let candidate_vto_copy_37_ptr : RegionName := "candidate_vto_copy_37"
  let candidate_vcat_8_ptr : RegionName := "candidate_vcat_8"
  let candidate_vmm_5_ptr : RegionName := "candidate_vmm_5"
  let candidate_vmm_6_ptr : RegionName := "candidate_vmm_6"
  let candidate_vto_copy_38_ptr : RegionName := "candidate_vto_copy_38"
  let candidate_vmul_39_ptr : RegionName := "candidate_vmul_39"
  let candidate_vmul_40_ptr : RegionName := "candidate_vmul_40"
  let candidate_vmul_41_ptr : RegionName := "candidate_vmul_41"
  let candidate_vsum_3_ptr : RegionName := "candidate_vsum_3"
  let candidate_vpow_5_ptr : RegionName := "candidate_vpow_5"
  let candidate_vmul_42_ptr : RegionName := "candidate_vmul_42"
  let candidate_vmul_43_ptr : RegionName := "candidate_vmul_43"
  let candidate_vdiv_1_ptr : RegionName := "candidate_vdiv_1"
  let candidate_vpow_6_ptr : RegionName := "candidate_vpow_6"
  let candidate_vmul_44_ptr : RegionName := "candidate_vmul_44"
  let candidate_vmul_45_ptr : RegionName := "candidate_vmul_45"
  let candidate_vadd_19_ptr : RegionName := "candidate_vadd_19"
  let candidate_vto_copy_39_ptr : RegionName := "candidate_vto_copy_39"
  let candidate_vmm_7_ptr : RegionName := "candidate_vmm_7"
  let candidate_vadd_20_ptr : RegionName := "candidate_vadd_20"
  triton {
    r0_z = tl.arange($(131072))
    r0_value = (tl.load($(parameter0_ptr) + ((((r0_z / $(512)) % $(256)) * $(512)) + (r0_z % $(512))), dtype=tl.float32) * 0.04419417382415922)
    r0_value = r0_value.to(tl.float32)
    r0_value = tl.broadcast_to(r0_value, [$(131072)])
    tl.store($(candidate_vmul_ptr) + r0_z, r0_value)
    r1_z = tl.arange($(256))
    r1_value = (tl.load($(parameter1_ptr) + r1_z, dtype=tl.float32) * 0.1)
    r1_value = r1_value.to(tl.float32)
    r1_value = tl.broadcast_to(r1_value, [$(256)])
    tl.store($(candidate_vmul_1_ptr) + r1_z, r1_value)
    r2_z = tl.arange($(256))
    r2_value = (tl.load($(candidate_vmul_1_ptr) + r2_z, dtype=tl.float32) + 1.0)
    r2_value = r2_value.to(tl.float32)
    r2_value = tl.broadcast_to(r2_value, [$(256)])
    tl.store($(candidate_vadd_ptr) + r2_z, r2_value)
    r3_z = tl.arange($(393216))
    r3_value = (tl.load($(parameter2_ptr) + ((((r3_z / $(256)) % $(1536)) * $(256)) + (r3_z % $(256))), dtype=tl.float32) * 0.0625)
    r3_value = r3_value.to(tl.float32)
    r3_value = tl.broadcast_to(r3_value, [$(393216)])
    tl.store($(candidate_vmul_2_ptr) + r3_z, r3_value)
    r4_z = tl.arange($(163840))
    r4_value = (tl.load($(parameter3_ptr) + ((((r4_z / $(512)) % $(320)) * $(512)) + (r4_z % $(512))), dtype=tl.float32) * 0.04419417382415922)
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(163840)])
    tl.store($(candidate_vmul_3_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(256))
    r5_value = (tl.load($(parameter4_ptr) + r5_z, dtype=tl.float32) * 0.1)
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(256)])
    tl.store($(candidate_vmul_4_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(256))
    r6_value = (tl.load($(candidate_vmul_4_ptr) + r6_z, dtype=tl.float32) + 1.0)
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(256)])
    tl.store($(candidate_vadd_1_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(524288))
    r7_value = (tl.load($(parameter5_ptr) + ((((r7_z / $(256)) % $(2048)) * $(256)) + (r7_z % $(256))), dtype=tl.float32) * 0.0625)
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(524288)])
    tl.store($(candidate_vmul_5_ptr) + r7_z, r7_value)
    r8_z = tl.arange($(524288))
    r8_value = (tl.load($(parameter6_ptr) + ((((r8_z / $(1024)) % $(512)) * $(1024)) + (r8_z % $(1024))), dtype=tl.float32) * 0.03125)
    r8_value = r8_value.to(tl.float32)
    r8_value = tl.broadcast_to(r8_value, [$(524288)])
    tl.store($(candidate_vmul_6_ptr) + r8_z, r8_value)
    r9_z = tl.arange($(131072))
    r9_value = tl.load($(candidate_vmul_ptr) + ((((r9_z / $(512)) % $(256)) * $(512)) + (r9_z % $(512))), dtype=tl.float32)
    r9_value = r9_value.to(tl.bfloat16)
    r9_value = tl.broadcast_to(r9_value, [$(131072)])
    tl.store($(candidate_vto_copy_ptr) + r9_z, r9_value)
    r10_z = tl.arange($(256))
    r10_value = tl.load($(candidate_vadd_ptr) + r10_z, dtype=tl.float32)
    r10_value = r10_value.to(tl.bfloat16)
    r10_value = tl.broadcast_to(r10_value, [$(256)])
    tl.store($(candidate_vto_copy_1_ptr) + r10_z, r10_value)
    r11_z = tl.arange($(393216))
    r11_value = tl.load($(candidate_vmul_2_ptr) + ((((r11_z / $(256)) % $(1536)) * $(256)) + (r11_z % $(256))), dtype=tl.float32)
    r11_value = r11_value.to(tl.bfloat16)
    r11_value = tl.broadcast_to(r11_value, [$(393216)])
    tl.store($(candidate_vto_copy_2_ptr) + r11_z, r11_value)
    r12_z = tl.arange($(163840))
    r12_value = tl.load($(candidate_vmul_3_ptr) + ((((r12_z / $(512)) % $(320)) * $(512)) + (r12_z % $(512))), dtype=tl.float32)
    r12_value = r12_value.to(tl.bfloat16)
    r12_value = tl.broadcast_to(r12_value, [$(163840)])
    tl.store($(candidate_vto_copy_3_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(256))
    r13_value = tl.load($(candidate_vadd_1_ptr) + r13_z, dtype=tl.float32)
    r13_value = r13_value.to(tl.bfloat16)
    r13_value = tl.broadcast_to(r13_value, [$(256)])
    tl.store($(candidate_vto_copy_4_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(524288))
    r14_value = tl.load($(candidate_vmul_5_ptr) + ((((r14_z / $(256)) % $(2048)) * $(256)) + (r14_z % $(256))), dtype=tl.float32)
    r14_value = r14_value.to(tl.bfloat16)
    r14_value = tl.broadcast_to(r14_value, [$(524288)])
    tl.store($(candidate_vto_copy_5_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(524288))
    r15_value = tl.load($(candidate_vmul_6_ptr) + ((((r15_z / $(1024)) % $(512)) * $(1024)) + (r15_z % $(1024))), dtype=tl.float32)
    r15_value = r15_value.to(tl.bfloat16)
    r15_value = tl.broadcast_to(r15_value, [$(524288)])
    tl.store($(candidate_vto_copy_6_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(8))
    r16_value = (tl.load($(parameter7_ptr) + r16_z, dtype=tl.float32) * 0.5)
    r16_value = r16_value.to(tl.float32)
    r16_value = tl.broadcast_to(r16_value, [$(8)])
    tl.store($(candidate_vmul_7_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(8))
    r17_value = tl.load($(candidate_vmul_7_ptr) + r17_z, dtype=tl.float32)
    r17_value = r17_value.to(tl.bfloat16)
    r17_value = tl.broadcast_to(r17_value, [$(8)])
    tl.store($(candidate_vto_copy_14_ptr) + r17_z, r17_value)
    r18_row = tl.arange($(256))
    r18_col = tl.arange($(256))
    r18_lane = tl.arange($(512))
    r18_left = tl.load($(input0_ptr) + ((((((r18_row[:, None] * $(512)) + r18_lane[None, :]) / $(65536)) % $(2)) * $(65536)) + (((((r18_row[:, None] * $(512)) + r18_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r18_row[:, None] * $(512)) + r18_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r18_right = tl.load($(candidate_vto_copy_ptr) + (((r18_col[None, :] % $(256)) * $(512)) + (r18_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r18_value = tl.dot(r18_left, r18_right)
    r18_value = r18_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_ptr) + ((r18_row[:, None] * $(256)) + r18_col[None, :]), r18_value)
    r19_z = tl.arange($(65536))
    r19_value = tl.load($(candidate_vmm_ptr) + ((((r19_z / $(32768)) % $(2)) * $(32768)) + (((r19_z / $(256)) % $(128)) * $(256)) + (r19_z % $(256))), dtype=tl.bfloat16)
    r19_value = r19_value.to(tl.float32)
    r19_value = tl.broadcast_to(r19_value, [$(65536)])
    tl.store($(candidate_vto_copy_18_ptr) + r19_z, r19_value)
    r20_z = tl.arange($(65536))
    r20_value = ((tl.load($(candidate_vto_copy_18_ptr) + ((((r20_z / $(32768)) % $(2)) * $(32768)) + (((r20_z / $(256)) % $(128)) * $(256)) + (r20_z % $(256))), dtype=tl.float32)) * (tl.load($(candidate_vto_copy_18_ptr) + ((((r20_z / $(32768)) % $(2)) * $(32768)) + (((r20_z / $(256)) % $(128)) * $(256)) + (r20_z % $(256))), dtype=tl.float32)))
    r20_value = r20_value.to(tl.float32)
    r20_value = tl.broadcast_to(r20_value, [$(65536)])
    tl.store($(candidate_vpow_1_ptr) + r20_z, r20_value)
    r21_z = tl.arange($(256))
    r21_lane = tl.arange($(256))
    r21_value = (tl.sum(tl.load($(candidate_vpow_1_ptr) + ((((r21_z[:, None] / $(128)) % $(2)) * $(32768)) + ((r21_z[:, None] % $(128)) * $(256)) + r21_lane[None, :]), dtype=tl.float32), axis=1) / 256.0)
    r21_value = r21_value.to(tl.float32)
    r21_value = tl.broadcast_to(r21_value, [$(256)])
    tl.store($(candidate_vmean_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(256))
    r22_value = (tl.load($(candidate_vmean_ptr) + ((((r22_z / $(128)) % $(2)) * $(128)) + (r22_z % $(128))), dtype=tl.float32) + 0.000001)
    r22_value = r22_value.to(tl.float32)
    r22_value = tl.broadcast_to(r22_value, [$(256)])
    tl.store($(candidate_vadd_2_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(256))
    r23_value = tl.rsqrt(tl.load($(candidate_vadd_2_ptr) + ((((r23_z / $(128)) % $(2)) * $(128)) + (r23_z % $(128))), dtype=tl.float32))
    r23_value = r23_value.to(tl.float32)
    r23_value = tl.broadcast_to(r23_value, [$(256)])
    tl.store($(candidate_vrsqrt_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(65536))
    r24_value = (tl.load($(candidate_vto_copy_18_ptr) + ((((r24_z / $(32768)) % $(2)) * $(32768)) + (((r24_z / $(256)) % $(128)) * $(256)) + (r24_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_ptr) + ((((r24_z / $(32768)) % $(2)) * $(128)) + ((r24_z / $(256)) % $(128))), dtype=tl.float32))
    r24_value = r24_value.to(tl.float32)
    r24_value = tl.broadcast_to(r24_value, [$(65536)])
    tl.store($(candidate_vmul_8_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(65536))
    r25_value = (tl.cast(tl.load($(candidate_vto_copy_1_ptr) + (r25_z % $(256)), dtype=tl.bfloat16), tl.float32) * tl.load($(candidate_vmul_8_ptr) + ((((r25_z / $(32768)) % $(2)) * $(32768)) + (((r25_z / $(256)) % $(128)) * $(256)) + (r25_z % $(256))), dtype=tl.float32))
    r25_value = r25_value.to(tl.float32)
    r25_value = tl.broadcast_to(r25_value, [$(65536)])
    tl.store($(candidate_vmul_9_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(65536))
    r26_value = tl.load($(candidate_vmul_9_ptr) + ((((r26_z / $(32768)) % $(2)) * $(32768)) + (((r26_z / $(256)) % $(128)) * $(256)) + (r26_z % $(256))), dtype=tl.float32)
    r26_value = r26_value.to(tl.bfloat16)
    r26_value = tl.broadcast_to(r26_value, [$(65536)])
    tl.store($(candidate_vto_copy_19_ptr) + r26_z, r26_value)
    r27_row = tl.arange($(256))
    r27_col = tl.arange($(1536))
    r27_lane = tl.arange($(256))
    r27_left = tl.load($(candidate_vto_copy_19_ptr) + ((((((r27_row[:, None] * $(256)) + r27_lane[None, :]) / $(32768)) % $(2)) * $(32768)) + (((((r27_row[:, None] * $(256)) + r27_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r27_row[:, None] * $(256)) + r27_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r27_right = tl.load($(candidate_vto_copy_2_ptr) + (((r27_col[None, :] % $(1536)) * $(256)) + (r27_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r27_value = tl.dot(r27_left, r27_right)
    r27_value = r27_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_1_ptr) + ((r27_row[:, None] * $(1536)) + r27_col[None, :]), r27_value)
    r28_row = tl.arange($(256))
    r28_col = tl.arange($(320))
    r28_lane = tl.arange($(512))
    r28_left = tl.load($(input0_ptr) + ((((((r28_row[:, None] * $(512)) + r28_lane[None, :]) / $(65536)) % $(2)) * $(65536)) + (((((r28_row[:, None] * $(512)) + r28_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r28_row[:, None] * $(512)) + r28_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r28_right = tl.load($(candidate_vto_copy_3_ptr) + (((r28_col[None, :] % $(320)) * $(512)) + (r28_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r28_value = tl.dot(r28_left, r28_right)
    r28_value = r28_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_2_ptr) + ((r28_row[:, None] * $(320)) + r28_col[None, :]), r28_value)
    r29_z = tl.arange($(65536))
    r29_value = tl.load($(candidate_vmm_2_ptr) + ((((r29_z / $(32768)) % $(2)) * $(40960)) + (((r29_z / $(256)) % $(128)) * $(320)) + (r29_z % $(256))), dtype=tl.bfloat16)
    r29_value = r29_value.to(tl.float32)
    r29_value = tl.broadcast_to(r29_value, [$(65536)])
    tl.store($(candidate_vto_copy_20_ptr) + r29_z, r29_value)
    r30_z = tl.arange($(65536))
    r30_value = ((tl.load($(candidate_vto_copy_20_ptr) + ((((r30_z / $(32768)) % $(2)) * $(32768)) + (((r30_z / $(256)) % $(128)) * $(256)) + (r30_z % $(256))), dtype=tl.float32)) * (tl.load($(candidate_vto_copy_20_ptr) + ((((r30_z / $(32768)) % $(2)) * $(32768)) + (((r30_z / $(256)) % $(128)) * $(256)) + (r30_z % $(256))), dtype=tl.float32)))
    r30_value = r30_value.to(tl.float32)
    r30_value = tl.broadcast_to(r30_value, [$(65536)])
    tl.store($(candidate_vpow_2_ptr) + r30_z, r30_value)
    r31_z = tl.arange($(256))
    r31_lane = tl.arange($(256))
    r31_value = (tl.sum(tl.load($(candidate_vpow_2_ptr) + ((((r31_z[:, None] / $(128)) % $(2)) * $(32768)) + ((r31_z[:, None] % $(128)) * $(256)) + r31_lane[None, :]), dtype=tl.float32), axis=1) / 256.0)
    r31_value = r31_value.to(tl.float32)
    r31_value = tl.broadcast_to(r31_value, [$(256)])
    tl.store($(candidate_vmean_1_ptr) + r31_z, r31_value)
    r32_z = tl.arange($(256))
    r32_value = (tl.load($(candidate_vmean_1_ptr) + ((((r32_z / $(128)) % $(2)) * $(128)) + (r32_z % $(128))), dtype=tl.float32) + 0.000001)
    r32_value = r32_value.to(tl.float32)
    r32_value = tl.broadcast_to(r32_value, [$(256)])
    tl.store($(candidate_vadd_3_ptr) + r32_z, r32_value)
    r33_z = tl.arange($(256))
    r33_value = tl.rsqrt(tl.load($(candidate_vadd_3_ptr) + ((((r33_z / $(128)) % $(2)) * $(128)) + (r33_z % $(128))), dtype=tl.float32))
    r33_value = r33_value.to(tl.float32)
    r33_value = tl.broadcast_to(r33_value, [$(256)])
    tl.store($(candidate_vrsqrt_1_ptr) + r33_z, r33_value)
    r34_z = tl.arange($(65536))
    r34_value = (tl.load($(candidate_vto_copy_20_ptr) + ((((r34_z / $(32768)) % $(2)) * $(32768)) + (((r34_z / $(256)) % $(128)) * $(256)) + (r34_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_1_ptr) + ((((r34_z / $(32768)) % $(2)) * $(128)) + ((r34_z / $(256)) % $(128))), dtype=tl.float32))
    r34_value = r34_value.to(tl.float32)
    r34_value = tl.broadcast_to(r34_value, [$(65536)])
    tl.store($(candidate_vmul_10_ptr) + r34_z, r34_value)
    r35_z = tl.arange($(65536))
    r35_value = (tl.cast(tl.load($(candidate_vto_copy_4_ptr) + (r35_z % $(256)), dtype=tl.bfloat16), tl.float32) * tl.load($(candidate_vmul_10_ptr) + ((((r35_z / $(32768)) % $(2)) * $(32768)) + (((r35_z / $(256)) % $(128)) * $(256)) + (r35_z % $(256))), dtype=tl.float32))
    r35_value = r35_value.to(tl.float32)
    r35_value = tl.broadcast_to(r35_value, [$(65536)])
    tl.store($(candidate_vmul_11_ptr) + r35_z, r35_value)
    r36_z = tl.arange($(65536))
    r36_value = tl.load($(candidate_vmul_11_ptr) + ((((r36_z / $(32768)) % $(2)) * $(32768)) + (((r36_z / $(256)) % $(128)) * $(256)) + (r36_z % $(256))), dtype=tl.float32)
    r36_value = r36_value.to(tl.bfloat16)
    r36_value = tl.broadcast_to(r36_value, [$(65536)])
    tl.store($(candidate_vto_copy_21_ptr) + r36_z, r36_value)
    r37_z = tl.arange($(65536))
    r37_value = (tl.load($(candidate_vmm_1_ptr) + ((((r37_z / $(32768)) % $(2)) * $(196608)) + (((r37_z / $(32)) % $(128)) * $(1536)) + (((r37_z / $(4096)) % $(8)) * $(192)) + ((r37_z % $(32)) * $(2)) + $(128)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r37_value = r37_value.to(tl.bfloat16)
    r37_value = tl.broadcast_to(r37_value, [$(65536)])
    tl.store($(candidate_vmul_12_ptr) + r37_z, r37_value)
    r38_z = tl.arange($(65536))
    r38_value = (tl.load($(candidate_vmm_1_ptr) + ((((r38_z / $(32768)) % $(2)) * $(196608)) + (((r38_z / $(32)) % $(128)) * $(1536)) + (((r38_z / $(4096)) % $(8)) * $(192)) + ((r38_z % $(32)) * $(2)) + $(129)), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r38_value = r38_value.to(tl.bfloat16)
    r38_value = tl.broadcast_to(r38_value, [$(65536)])
    tl.store($(candidate_vmul_13_ptr) + r38_z, r38_value)
    r39_z = tl.arange($(65536))
    r39_value = (tl.load($(candidate_vmul_12_ptr) + ((((r39_z / $(32768)) % $(2)) * $(32768)) + (((r39_z / $(4096)) % $(8)) * $(4096)) + (((r39_z / $(32)) % $(128)) * $(32)) + (r39_z % $(32))), dtype=tl.bfloat16) - tl.load($(candidate_vmul_13_ptr) + ((((r39_z / $(32768)) % $(2)) * $(32768)) + (((r39_z / $(4096)) % $(8)) * $(4096)) + (((r39_z / $(32)) % $(128)) * $(32)) + (r39_z % $(32))), dtype=tl.bfloat16))
    r39_value = r39_value.to(tl.bfloat16)
    r39_value = tl.broadcast_to(r39_value, [$(65536)])
    tl.store($(candidate_vsub_ptr) + r39_z, r39_value)
    r40_z = tl.arange($(65536))
    r40_value = (tl.load($(candidate_vmm_1_ptr) + ((((r40_z / $(32768)) % $(2)) * $(196608)) + (((r40_z / $(32)) % $(128)) * $(1536)) + (((r40_z / $(4096)) % $(8)) * $(192)) + ((r40_z % $(32)) * $(2)) + $(129)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r40_value = r40_value.to(tl.bfloat16)
    r40_value = tl.broadcast_to(r40_value, [$(65536)])
    tl.store($(candidate_vmul_14_ptr) + r40_z, r40_value)
    r41_z = tl.arange($(65536))
    r41_value = (tl.load($(candidate_vmm_1_ptr) + ((((r41_z / $(32768)) % $(2)) * $(196608)) + (((r41_z / $(32)) % $(128)) * $(1536)) + (((r41_z / $(4096)) % $(8)) * $(192)) + ((r41_z % $(32)) * $(2)) + $(128)), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r41_value = r41_value.to(tl.bfloat16)
    r41_value = tl.broadcast_to(r41_value, [$(65536)])
    tl.store($(candidate_vmul_15_ptr) + r41_z, r41_value)
    r42_z = tl.arange($(65536))
    r42_value = (tl.load($(candidate_vmul_14_ptr) + ((((r42_z / $(32768)) % $(2)) * $(32768)) + (((r42_z / $(4096)) % $(8)) * $(4096)) + (((r42_z / $(32)) % $(128)) * $(32)) + (r42_z % $(32))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_15_ptr) + ((((r42_z / $(32768)) % $(2)) * $(32768)) + (((r42_z / $(4096)) % $(8)) * $(4096)) + (((r42_z / $(32)) % $(128)) * $(32)) + (r42_z % $(32))), dtype=tl.bfloat16))
    r42_value = r42_value.to(tl.bfloat16)
    r42_value = tl.broadcast_to(r42_value, [$(65536)])
    tl.store($(candidate_vadd_4_ptr) + r42_z, r42_value)
    r43_z = tl.arange($(131072))
    r43_value = tl.where((r43_z % $(64)) < $(32), tl.load($(candidate_vsub_ptr) + ((((r43_z / $(65536)) % $(2)) * $(32768)) + (((r43_z / $(8192)) % $(8)) * $(4096)) + (((r43_z / $(64)) % $(128)) * $(32)) + (((r43_z % $(64)) - $(0)) % $(32))), dtype=tl.bfloat16), tl.load($(candidate_vadd_4_ptr) + ((((r43_z / $(65536)) % $(2)) * $(32768)) + (((r43_z / $(8192)) % $(8)) * $(4096)) + (((r43_z / $(64)) % $(128)) * $(32)) + (((r43_z % $(64)) - $(32)) % $(32))), dtype=tl.bfloat16))
    r43_value = r43_value.to(tl.bfloat16)
    r43_value = tl.broadcast_to(r43_value, [$(131072)])
    tl.store($(candidate_vcat_ptr) + r43_z, r43_value)
    r44_z = tl.arange($(8192))
    r44_value = (tl.load($(candidate_vmm_2_ptr) + ((((r44_z / $(4096)) % $(2)) * $(40960)) + (((r44_z / $(32)) % $(128)) * $(320)) + ((r44_z % $(32)) * $(2)) + $(256)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r44_value = r44_value.to(tl.bfloat16)
    r44_value = tl.broadcast_to(r44_value, [$(8192)])
    tl.store($(candidate_vmul_16_ptr) + r44_z, r44_value)
    r45_z = tl.arange($(8192))
    r45_value = (tl.load($(candidate_vmm_2_ptr) + ((((r45_z / $(4096)) % $(2)) * $(40960)) + (((r45_z / $(32)) % $(128)) * $(320)) + ((r45_z % $(32)) * $(2)) + $(257)), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r45_value = r45_value.to(tl.bfloat16)
    r45_value = tl.broadcast_to(r45_value, [$(8192)])
    tl.store($(candidate_vmul_17_ptr) + r45_z, r45_value)
    r46_z = tl.arange($(8192))
    r46_value = (tl.load($(candidate_vmul_16_ptr) + ((((r46_z / $(4096)) % $(2)) * $(4096)) + (((r46_z / $(32)) % $(128)) * $(32)) + (r46_z % $(32))), dtype=tl.bfloat16) - tl.load($(candidate_vmul_17_ptr) + ((((r46_z / $(4096)) % $(2)) * $(4096)) + (((r46_z / $(32)) % $(128)) * $(32)) + (r46_z % $(32))), dtype=tl.bfloat16))
    r46_value = r46_value.to(tl.bfloat16)
    r46_value = tl.broadcast_to(r46_value, [$(8192)])
    tl.store($(candidate_vsub_1_ptr) + r46_z, r46_value)
    r47_z = tl.arange($(8192))
    r47_value = (tl.load($(candidate_vmm_2_ptr) + ((((r47_z / $(4096)) % $(2)) * $(40960)) + (((r47_z / $(32)) % $(128)) * $(320)) + ((r47_z % $(32)) * $(2)) + $(257)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r47_value = r47_value.to(tl.bfloat16)
    r47_value = tl.broadcast_to(r47_value, [$(8192)])
    tl.store($(candidate_vmul_18_ptr) + r47_z, r47_value)
    r48_z = tl.arange($(8192))
    r48_value = (tl.load($(candidate_vmm_2_ptr) + ((((r48_z / $(4096)) % $(2)) * $(40960)) + (((r48_z / $(32)) % $(128)) * $(320)) + ((r48_z % $(32)) * $(2)) + $(256)), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r48_value = r48_value.to(tl.bfloat16)
    r48_value = tl.broadcast_to(r48_value, [$(8192)])
    tl.store($(candidate_vmul_19_ptr) + r48_z, r48_value)
    r49_z = tl.arange($(8192))
    r49_value = (tl.load($(candidate_vmul_18_ptr) + ((((r49_z / $(4096)) % $(2)) * $(4096)) + (((r49_z / $(32)) % $(128)) * $(32)) + (r49_z % $(32))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_19_ptr) + ((((r49_z / $(4096)) % $(2)) * $(4096)) + (((r49_z / $(32)) % $(128)) * $(32)) + (r49_z % $(32))), dtype=tl.bfloat16))
    r49_value = r49_value.to(tl.bfloat16)
    r49_value = tl.broadcast_to(r49_value, [$(8192)])
    tl.store($(candidate_vadd_5_ptr) + r49_z, r49_value)
    r50_z = tl.arange($(16384))
    r50_value = tl.where((r50_z % $(64)) < $(32), tl.load($(candidate_vsub_1_ptr) + ((((r50_z / $(8192)) % $(2)) * $(4096)) + (((r50_z / $(64)) % $(128)) * $(32)) + (((r50_z % $(64)) - $(0)) % $(32))), dtype=tl.bfloat16), tl.load($(candidate_vadd_5_ptr) + ((((r50_z / $(8192)) % $(2)) * $(4096)) + (((r50_z / $(64)) % $(128)) * $(32)) + (((r50_z % $(64)) - $(32)) % $(32))), dtype=tl.bfloat16))
    r50_value = r50_value.to(tl.bfloat16)
    r50_value = tl.broadcast_to(r50_value, [$(16384)])
    tl.store($(candidate_vcat_1_ptr) + r50_z, r50_value)
    r51_row = tl.arange($(256))
    r51_col = tl.arange($(1024))
    r51_lane = tl.arange($(256))
    for r51_group in range($(1)) {
      r51_left = tl.load($(candidate_vto_copy_21_ptr) + ((((((((((r51_row[:, None] * $(256)) + r51_lane[None, :]) / $(256)) % $(256)) * $(256)) + (((r51_row[:, None] * $(256)) + r51_lane[None, :]) % $(256))) / $(32768)) % $(2)) * $(32768)) + ((((((r51_row[:, None] * $(256)) + r51_lane[None, :]) / $(256)) % $(256)) % $(128)) * $(256)) + (((r51_row[:, None] * $(256)) + r51_lane[None, :]) % $(256))), dtype=tl.bfloat16)
      r51_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r51_lane[:, None] * $(1024)) + r51_col[None, :]) / $(128)) % $(8)) * $(65536)) + ((((r51_lane[:, None] * $(1024)) + r51_col[None, :]) % $(128)) * $(256)) + ((((r51_lane[:, None] * $(1024)) + r51_col[None, :]) / $(1024)) % $(256))), dtype=tl.bfloat16)
      r51_value = tl.dot(r51_left, r51_right)
      r51_value = r51_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_ptr) + ((r51_group * $(262144)) + (r51_row[:, None] * $(1024)) + r51_col[None, :]), r51_value)
    }
    r52_row = tl.arange($(256))
    r52_col = tl.arange($(1024))
    r52_lane = tl.arange($(256))
    for r52_group in range($(1)) {
      r52_left = tl.load($(candidate_vto_copy_21_ptr) + ((((((((((r52_row[:, None] * $(256)) + r52_lane[None, :]) / $(256)) % $(256)) * $(256)) + (((r52_row[:, None] * $(256)) + r52_lane[None, :]) % $(256))) / $(32768)) % $(2)) * $(32768)) + ((((((r52_row[:, None] * $(256)) + r52_lane[None, :]) / $(256)) % $(256)) % $(128)) * $(256)) + (((r52_row[:, None] * $(256)) + r52_lane[None, :]) % $(256))), dtype=tl.bfloat16)
      r52_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r52_lane[:, None] * $(1024)) + r52_col[None, :]) / $(128)) % $(8)) * $(65536)) + ((((r52_lane[:, None] * $(1024)) + r52_col[None, :]) % $(128)) * $(256)) + ((((r52_lane[:, None] * $(1024)) + r52_col[None, :]) / $(1024)) % $(256)) + $(32768)), dtype=tl.bfloat16)
      r52_value = tl.dot(r52_left, r52_right)
      r52_value = r52_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_1_ptr) + ((r52_group * $(262144)) + (r52_row[:, None] * $(1024)) + r52_col[None, :]), r52_value)
    }
    r53_z = tl.arange($(393216))
    r53_value = tl.where((r53_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r53_z / $(1536)) % $(256)) * $(1024)) + (((r53_z / $(192)) % $(8)) * $(128)) + ((r53_z % $(192)) - $(0))) / $(131072)) % $(2)) * $(196608)) + (((((((r53_z / $(1536)) % $(256)) * $(1024)) + (((r53_z / $(192)) % $(8)) * $(128)) + ((r53_z % $(192)) - $(0))) / $(1024)) % $(128)) * $(1536)) + (((((((r53_z / $(1536)) % $(256)) * $(1024)) + (((r53_z / $(192)) % $(8)) * $(128)) + ((r53_z % $(192)) - $(0))) / $(128)) % $(8)) * $(192)) + (((((r53_z / $(1536)) % $(256)) * $(1024)) + (((r53_z / $(192)) % $(8)) * $(128)) + ((r53_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_ptr) + ((((((((r53_z / $(1536)) % $(256)) * $(512)) + (((r53_z / $(192)) % $(8)) * $(64)) + ((r53_z % $(192)) - $(128))) / $(65536)) % $(2)) * $(65536)) + (((((((r53_z / $(1536)) % $(256)) * $(512)) + (((r53_z / $(192)) % $(8)) * $(64)) + ((r53_z % $(192)) - $(128))) / $(64)) % $(8)) * $(8192)) + (((((((r53_z / $(1536)) % $(256)) * $(512)) + (((r53_z / $(192)) % $(8)) * $(64)) + ((r53_z % $(192)) - $(128))) / $(512)) % $(128)) * $(64)) + (((((r53_z / $(1536)) % $(256)) * $(512)) + (((r53_z / $(192)) % $(8)) * $(64)) + ((r53_z % $(192)) - $(128))) % $(64))), dtype=tl.bfloat16))
    r53_value = r53_value.to(tl.bfloat16)
    r53_value = tl.broadcast_to(r53_value, [$(393216)])
    tl.store($(candidate_vcat_2_ptr) + r53_z, r53_value)
    r54_z = tl.arange($(393216))
    r54_value = tl.where((r54_z % $(192)) < $(128), tl.load($(candidate_vbmm_ptr) + ((((((((r54_z / $(1536)) % $(256)) * $(1024)) + (((r54_z / $(192)) % $(8)) * $(128)) + ((r54_z % $(192)) - $(0))) / $(1024)) % $(256)) * $(1024)) + (((((((r54_z / $(1536)) % $(256)) * $(1024)) + (((r54_z / $(192)) % $(8)) * $(128)) + ((r54_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r54_z / $(1536)) % $(256)) * $(1024)) + (((r54_z / $(192)) % $(8)) * $(128)) + ((r54_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_1_ptr) + ((((((((r54_z / $(1536)) % $(256)) * $(64)) + ((r54_z % $(192)) - $(128))) / $(8192)) % $(2)) * $(8192)) + (((((((r54_z / $(1536)) % $(256)) * $(64)) + ((r54_z % $(192)) - $(128))) / $(64)) % $(128)) * $(64)) + (((((r54_z / $(1536)) % $(256)) * $(64)) + ((r54_z % $(192)) - $(128))) % $(64))), dtype=tl.bfloat16))
    r54_value = r54_value.to(tl.bfloat16)
    r54_value = tl.broadcast_to(r54_value, [$(393216)])
    tl.store($(candidate_vcat_3_ptr) + r54_z, r54_value)
    r55_z = tl.arange($(196608))
    r55_value = tl.load($(candidate_vcat_2_ptr) + ((((r55_z / $(192)) % $(128)) * $(1536)) + (((r55_z / $(24576)) % $(8)) * $(192)) + (r55_z % $(192))), dtype=tl.bfloat16)
    r55_value = r55_value.to(tl.float32)
    r55_value = tl.broadcast_to(r55_value, [$(196608)])
    tl.store($(candidate_vto_copy_22_ptr) + r55_z, r55_value)
    r56_z = tl.arange($(196608))
    r56_value = tl.load($(candidate_vcat_3_ptr) + ((((r56_z / $(192)) % $(128)) * $(1536)) + (((r56_z / $(24576)) % $(8)) * $(192)) + (r56_z % $(192))), dtype=tl.bfloat16)
    r56_value = r56_value.to(tl.float32)
    r56_value = tl.broadcast_to(r56_value, [$(196608)])
    tl.store($(candidate_vto_copy_23_ptr) + r56_z, r56_value)
    r57_row = tl.arange($(128))
    r57_col = tl.arange($(128))
    r57_lane = tl.arange($(192))
    for r57_group in range($(8)) {
      r57_left = tl.load($(candidate_vto_copy_22_ptr) + (((((((r57_group % $(8)) * $(24576)) + (r57_row[:, None] * $(192)) + r57_lane[None, :]) / $(24576)) % $(8)) * $(24576)) + ((((((r57_group % $(8)) * $(24576)) + (r57_row[:, None] * $(192)) + r57_lane[None, :]) / $(192)) % $(128)) * $(192)) + ((((r57_group % $(8)) * $(24576)) + (r57_row[:, None] * $(192)) + r57_lane[None, :]) % $(192))), dtype=tl.float32)
      r57_right = tl.load($(candidate_vto_copy_23_ptr) + (((((((r57_group % $(8)) * $(24576)) + (r57_lane[:, None] * $(128)) + r57_col[None, :]) / $(24576)) % $(8)) * $(24576)) + (((((r57_group % $(8)) * $(24576)) + (r57_lane[:, None] * $(128)) + r57_col[None, :]) % $(128)) * $(192)) + (((((r57_group % $(8)) * $(24576)) + (r57_lane[:, None] * $(128)) + r57_col[None, :]) / $(128)) % $(192))), dtype=tl.float32)
      r57_value = tl.dot(r57_left, r57_right)
      r57_value = r57_value.to(tl.float32)
      tl.store($(candidate_vbmm_2_ptr) + ((r57_group * $(16384)) + (r57_row[:, None] * $(128)) + r57_col[None, :]), r57_value)
    }
    r58_z = tl.arange($(131072))
    r58_value = (tl.load($(candidate_vbmm_2_ptr) + ((((r58_z / $(16384)) % $(8)) * $(16384)) + (((r58_z / $(128)) % $(128)) * $(128)) + (r58_z % $(128))), dtype=tl.float32) * 0.07216878364870322)
    r58_value = r58_value.to(tl.float32)
    r58_value = tl.broadcast_to(r58_value, [$(131072)])
    tl.store($(candidate_vmul_20_ptr) + r58_z, r58_value)
    r59_z = tl.arange($(128))
    r59_value = (tl.cast(r59_z, tl.int64) + tl.full([], $(128), dtype=tl.int64))
    r59_value = tl.broadcast_to(r59_value, [$(128)])
    tl.store($(candidate_vadd_6_ptr) + r59_z, r59_value)
    r60_z = tl.arange($(128))
    r60_value = (tl.load($(candidate_vadd_6_ptr) + r60_z, dtype=tl.int64) - tl.full([], $(128), dtype=tl.int64))
    r60_value = tl.broadcast_to(r60_value, [$(128)])
    tl.store($(candidate_vsub_2_ptr) + r60_z, r60_value)
    r61_z = tl.arange($(16384))
    r61_value = (tl.cast((r61_z % $(128)), tl.int64) > tl.load($(candidate_vsub_2_ptr) + ((r61_z / $(128)) % $(128)), dtype=tl.int64))
    r61_value = tl.broadcast_to(r61_value, [$(16384)])
    tl.store($(candidate_vgt_ptr) + r61_z, r61_value)
    r62_z = tl.arange($(131072))
    r62_value = tl.where(tl.load($(candidate_vgt_ptr) + ((((r62_z / $(128)) % $(128)) * $(128)) + (r62_z % $(128))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_20_ptr) + ((((r62_z / $(16384)) % $(8)) * $(16384)) + (((r62_z / $(128)) % $(128)) * $(128)) + (r62_z % $(128))), dtype=tl.float32))
    r62_value = r62_value.to(tl.float32)
    r62_value = tl.broadcast_to(r62_value, [$(131072)])
    tl.store($(candidate_vmasked_fill_2_ptr) + r62_z, r62_value)
    r63_z = tl.arange($(8))
    r63_value = tl.load($(candidate_vto_copy_14_ptr) + r63_z, dtype=tl.bfloat16)
    r63_value = r63_value.to(tl.float32)
    r63_value = tl.broadcast_to(r63_value, [$(8)])
    tl.store($(candidate_vto_copy_24_ptr) + r63_z, r63_value)
    r64_z = tl.arange($(132096))
    r64_value = tl.where((r64_z % $(129)) < $(128), tl.load($(candidate_vmasked_fill_2_ptr) + ((((r64_z / $(16512)) % $(8)) * $(16384)) + (((r64_z / $(129)) % $(128)) * $(128)) + (((r64_z % $(129)) - $(0)) % $(128))), dtype=tl.float32), tl.load($(candidate_vto_copy_24_ptr) + ((r64_z / $(16512)) % $(8)), dtype=tl.float32))
    r64_value = r64_value.to(tl.float32)
    r64_value = tl.broadcast_to(r64_value, [$(132096)])
    tl.store($(candidate_vcat_4_ptr) + r64_z, r64_value)
    r65_z = tl.arange($(132096))
    r65_lane = tl.arange($(129))
    r65_row_values = tl.load($(candidate_vcat_4_ptr) + ((((r65_z[:, None] / $(16512)) % $(8)) * $(16512)) + (((r65_z[:, None] / $(129)) % $(128)) * $(129)) + (r65_lane[None, :] % $(129))), dtype=tl.float32)
    r65_maximum = tl.max(r65_row_values, axis=1, keep_dims=true)
    r65_weights = tl.exp(r65_row_values - r65_maximum)
    r65_total = tl.sum(r65_weights, axis=1)
    r65_maximum = tl.max(r65_row_values, axis=1)
    r65_value = tl.exp(tl.load($(candidate_vcat_4_ptr) + ((((r65_z / $(16512)) % $(8)) * $(16512)) + (((r65_z / $(129)) % $(128)) * $(129)) + (r65_z % $(129))), dtype=tl.float32) - r65_maximum) / r65_total
    r65_value = r65_value.to(tl.float32)
    r65_value = tl.broadcast_to(r65_value, [$(132096)])
    tl.store($(candidate_vsoftmax_ptr) + r65_z, r65_value)
    r66_z = tl.arange($(131072))
    r66_value = tl.load($(candidate_vsoftmax_ptr) + ((((r66_z / $(16384)) % $(8)) * $(16512)) + (((r66_z / $(128)) % $(128)) * $(129)) + (r66_z % $(128))), dtype=tl.float32)
    r66_value = r66_value.to(tl.bfloat16)
    r66_value = tl.broadcast_to(r66_value, [$(131072)])
    tl.store($(candidate_vto_copy_25_ptr) + r66_z, r66_value)
    r67_z = tl.arange($(196608))
    r67_value = tl.load($(candidate_vcat_2_ptr) + ((((r67_z / $(192)) % $(128)) * $(1536)) + (((r67_z / $(24576)) % $(8)) * $(192)) + (r67_z % $(192)) + $(196608)), dtype=tl.bfloat16)
    r67_value = r67_value.to(tl.float32)
    r67_value = tl.broadcast_to(r67_value, [$(196608)])
    tl.store($(candidate_vto_copy_26_ptr) + r67_z, r67_value)
    r68_z = tl.arange($(196608))
    r68_value = tl.load($(candidate_vcat_3_ptr) + ((((r68_z / $(192)) % $(128)) * $(1536)) + (((r68_z / $(24576)) % $(8)) * $(192)) + (r68_z % $(192)) + $(196608)), dtype=tl.bfloat16)
    r68_value = r68_value.to(tl.float32)
    r68_value = tl.broadcast_to(r68_value, [$(196608)])
    tl.store($(candidate_vto_copy_27_ptr) + r68_z, r68_value)
    r69_row = tl.arange($(128))
    r69_col = tl.arange($(128))
    r69_lane = tl.arange($(192))
    for r69_group in range($(8)) {
      r69_left = tl.load($(candidate_vto_copy_26_ptr) + (((((((r69_group % $(8)) * $(24576)) + (r69_row[:, None] * $(192)) + r69_lane[None, :]) / $(24576)) % $(8)) * $(24576)) + ((((((r69_group % $(8)) * $(24576)) + (r69_row[:, None] * $(192)) + r69_lane[None, :]) / $(192)) % $(128)) * $(192)) + ((((r69_group % $(8)) * $(24576)) + (r69_row[:, None] * $(192)) + r69_lane[None, :]) % $(192))), dtype=tl.float32)
      r69_right = tl.load($(candidate_vto_copy_27_ptr) + (((((((r69_group % $(8)) * $(24576)) + (r69_lane[:, None] * $(128)) + r69_col[None, :]) / $(24576)) % $(8)) * $(24576)) + (((((r69_group % $(8)) * $(24576)) + (r69_lane[:, None] * $(128)) + r69_col[None, :]) % $(128)) * $(192)) + (((((r69_group % $(8)) * $(24576)) + (r69_lane[:, None] * $(128)) + r69_col[None, :]) / $(128)) % $(192))), dtype=tl.float32)
      r69_value = tl.dot(r69_left, r69_right)
      r69_value = r69_value.to(tl.float32)
      tl.store($(candidate_vbmm_4_ptr) + ((r69_group * $(16384)) + (r69_row[:, None] * $(128)) + r69_col[None, :]), r69_value)
    }
    r70_z = tl.arange($(131072))
    r70_value = (tl.load($(candidate_vbmm_4_ptr) + ((((r70_z / $(16384)) % $(8)) * $(16384)) + (((r70_z / $(128)) % $(128)) * $(128)) + (r70_z % $(128))), dtype=tl.float32) * 0.07216878364870322)
    r70_value = r70_value.to(tl.float32)
    r70_value = tl.broadcast_to(r70_value, [$(131072)])
    tl.store($(candidate_vmul_21_ptr) + r70_z, r70_value)
    r71_z = tl.arange($(128))
    r71_value = (tl.cast(r71_z, tl.int64) + tl.full([], $(128), dtype=tl.int64))
    r71_value = tl.broadcast_to(r71_value, [$(128)])
    tl.store($(candidate_vadd_7_ptr) + r71_z, r71_value)
    r72_z = tl.arange($(128))
    r72_value = (tl.load($(candidate_vadd_7_ptr) + r72_z, dtype=tl.int64) - tl.full([], $(128), dtype=tl.int64))
    r72_value = tl.broadcast_to(r72_value, [$(128)])
    tl.store($(candidate_vsub_3_ptr) + r72_z, r72_value)
    r73_z = tl.arange($(16384))
    r73_value = (tl.cast((r73_z % $(128)), tl.int64) > tl.load($(candidate_vsub_3_ptr) + ((r73_z / $(128)) % $(128)), dtype=tl.int64))
    r73_value = tl.broadcast_to(r73_value, [$(16384)])
    tl.store($(candidate_vgt_1_ptr) + r73_z, r73_value)
    r74_z = tl.arange($(131072))
    r74_value = tl.where(tl.load($(candidate_vgt_1_ptr) + ((((r74_z / $(128)) % $(128)) * $(128)) + (r74_z % $(128))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_21_ptr) + ((((r74_z / $(16384)) % $(8)) * $(16384)) + (((r74_z / $(128)) % $(128)) * $(128)) + (r74_z % $(128))), dtype=tl.float32))
    r74_value = r74_value.to(tl.float32)
    r74_value = tl.broadcast_to(r74_value, [$(131072)])
    tl.store($(candidate_vmasked_fill_3_ptr) + r74_z, r74_value)
    r75_z = tl.arange($(8))
    r75_value = tl.load($(candidate_vto_copy_14_ptr) + r75_z, dtype=tl.bfloat16)
    r75_value = r75_value.to(tl.float32)
    r75_value = tl.broadcast_to(r75_value, [$(8)])
    tl.store($(candidate_vto_copy_28_ptr) + r75_z, r75_value)
    r76_z = tl.arange($(132096))
    r76_value = tl.where((r76_z % $(129)) < $(128), tl.load($(candidate_vmasked_fill_3_ptr) + ((((r76_z / $(16512)) % $(8)) * $(16384)) + (((r76_z / $(129)) % $(128)) * $(128)) + (((r76_z % $(129)) - $(0)) % $(128))), dtype=tl.float32), tl.load($(candidate_vto_copy_28_ptr) + ((r76_z / $(16512)) % $(8)), dtype=tl.float32))
    r76_value = r76_value.to(tl.float32)
    r76_value = tl.broadcast_to(r76_value, [$(132096)])
    tl.store($(candidate_vcat_5_ptr) + r76_z, r76_value)
    r77_z = tl.arange($(132096))
    r77_lane = tl.arange($(129))
    r77_row_values = tl.load($(candidate_vcat_5_ptr) + ((((r77_z[:, None] / $(16512)) % $(8)) * $(16512)) + (((r77_z[:, None] / $(129)) % $(128)) * $(129)) + (r77_lane[None, :] % $(129))), dtype=tl.float32)
    r77_maximum = tl.max(r77_row_values, axis=1, keep_dims=true)
    r77_weights = tl.exp(r77_row_values - r77_maximum)
    r77_total = tl.sum(r77_weights, axis=1)
    r77_maximum = tl.max(r77_row_values, axis=1)
    r77_value = tl.exp(tl.load($(candidate_vcat_5_ptr) + ((((r77_z / $(16512)) % $(8)) * $(16512)) + (((r77_z / $(129)) % $(128)) * $(129)) + (r77_z % $(129))), dtype=tl.float32) - r77_maximum) / r77_total
    r77_value = r77_value.to(tl.float32)
    r77_value = tl.broadcast_to(r77_value, [$(132096)])
    tl.store($(candidate_vsoftmax_1_ptr) + r77_z, r77_value)
    r78_z = tl.arange($(131072))
    r78_value = tl.load($(candidate_vsoftmax_1_ptr) + ((((r78_z / $(16384)) % $(8)) * $(16512)) + (((r78_z / $(128)) % $(128)) * $(129)) + (r78_z % $(128))), dtype=tl.float32)
    r78_value = r78_value.to(tl.bfloat16)
    r78_value = tl.broadcast_to(r78_value, [$(131072)])
    tl.store($(candidate_vto_copy_29_ptr) + r78_z, r78_value)
    r79_row = tl.arange($(256))
    r79_col = tl.arange($(1024))
    r79_lane = tl.arange($(512))
    r79_left = tl.load($(input1_ptr) + ((((((r79_row[:, None] * $(512)) + r79_lane[None, :]) / $(65536)) % $(2)) * $(65536)) + (((((r79_row[:, None] * $(512)) + r79_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r79_row[:, None] * $(512)) + r79_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r79_right = tl.load($(candidate_vto_copy_6_ptr) + (((r79_lane[:, None] % $(512)) * $(1024)) + (r79_col[None, :] % $(1024))), dtype=tl.bfloat16)
    r79_value = tl.dot(r79_left, r79_right)
    r79_value = r79_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_4_ptr) + ((r79_row[:, None] * $(1024)) + r79_col[None, :]), r79_value)
    r80_row = tl.arange($(128))
    r80_col = tl.arange($(128))
    r80_lane = tl.arange($(128))
    for r80_group in range($(8)) {
      r80_left = tl.load($(candidate_vto_copy_29_ptr) + (((((((r80_group % $(8)) * $(16384)) + (r80_lane[None, :] * $(128)) + r80_row[:, None]) / $(16384)) % $(8)) * $(16384)) + ((((((r80_group % $(8)) * $(16384)) + (r80_lane[None, :] * $(128)) + r80_row[:, None]) / $(128)) % $(128)) * $(128)) + ((((r80_group % $(8)) * $(16384)) + (r80_lane[None, :] * $(128)) + r80_row[:, None]) % $(128))), dtype=tl.bfloat16)
      r80_right = tl.load($(candidate_vmm_4_ptr) + (((((((r80_group % $(8)) * $(16384)) + (r80_lane[:, None] * $(128)) + r80_col[None, :]) / $(128)) % $(128)) * $(1024)) + ((((((r80_group % $(8)) * $(16384)) + (r80_lane[:, None] * $(128)) + r80_col[None, :]) / $(16384)) % $(8)) * $(128)) + ((((r80_group % $(8)) * $(16384)) + (r80_lane[:, None] * $(128)) + r80_col[None, :]) % $(128)) + $(131072)), dtype=tl.bfloat16)
      r80_value = tl.dot(r80_left, r80_right)
      r80_value = r80_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_6_ptr) + ((r80_group * $(16384)) + (r80_row[:, None] * $(128)) + r80_col[None, :]), r80_value)
    }
    r81_row = tl.arange($(128))
    r81_col = tl.arange($(128))
    r81_lane = tl.arange($(128))
    for r81_group in range($(8)) {
      r81_left = tl.load($(candidate_vmm_4_ptr) + (((((((r81_group % $(8)) * $(16384)) + (r81_row[:, None] * $(128)) + r81_lane[None, :]) / $(128)) % $(128)) * $(1024)) + ((((((r81_group % $(8)) * $(16384)) + (r81_row[:, None] * $(128)) + r81_lane[None, :]) / $(16384)) % $(8)) * $(128)) + ((((r81_group % $(8)) * $(16384)) + (r81_row[:, None] * $(128)) + r81_lane[None, :]) % $(128)) + $(131072)), dtype=tl.bfloat16)
      r81_right = tl.load($(candidate_vbmm_1_ptr) + (((((((r81_group % $(8)) * $(16384)) + (r81_col[None, :] * $(128)) + r81_lane[:, None]) / $(128)) % $(128)) * $(1024)) + ((((((r81_group % $(8)) * $(16384)) + (r81_col[None, :] * $(128)) + r81_lane[:, None]) / $(16384)) % $(8)) * $(128)) + ((((r81_group % $(8)) * $(16384)) + (r81_col[None, :] * $(128)) + r81_lane[:, None]) % $(128)) + $(131072)), dtype=tl.bfloat16)
      r81_value = tl.dot(r81_left, r81_right)
      r81_value = r81_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_7_ptr) + ((r81_group * $(16384)) + (r81_row[:, None] * $(128)) + r81_col[None, :]), r81_value)
    }
    r82_z = tl.arange($(131072))
    r82_value = tl.load($(candidate_vbmm_7_ptr) + ((((r82_z / $(16384)) % $(8)) * $(16384)) + (((r82_z / $(128)) % $(128)) * $(128)) + (r82_z % $(128))), dtype=tl.bfloat16)
    r82_value = r82_value.to(tl.float32)
    r82_value = tl.broadcast_to(r82_value, [$(131072)])
    tl.store($(candidate_vto_copy_30_ptr) + r82_z, r82_value)
    r83_z = tl.arange($(132096))
    r83_value = tl.where((((r83_z % $(129)) >= $(0)) & ((r83_z % $(129)) < $(128)) & (((r83_z % $(129)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vto_copy_30_ptr) + ((((r83_z / $(16512)) % $(8)) * $(16384)) + (((r83_z / $(129)) % $(128)) * $(128)) + (((r83_z % $(129)) - $(0)) % $(128))), dtype=tl.float32), 0.0)
    r83_value = r83_value.to(tl.float32)
    r83_value = tl.broadcast_to(r83_value, [$(132096)])
    tl.store($(candidate_vslice_backward_ptr) + r83_z, r83_value)
    r84_z = tl.arange($(132096))
    r84_lane = tl.arange($(129))
    r84_row_values = tl.load($(candidate_vsoftmax_1_ptr) + ((((r84_z[:, None] / $(16512)) % $(8)) * $(16512)) + (((r84_z[:, None] / $(129)) % $(128)) * $(129)) + (r84_lane[None, :] % $(129))), dtype=tl.float32)
    r84_row_grad = tl.load($(candidate_vslice_backward_ptr) + ((((r84_z[:, None] / $(16512)) % $(8)) * $(16512)) + (((r84_z[:, None] / $(129)) % $(128)) * $(129)) + (r84_lane[None, :] % $(129))), dtype=tl.float32)
    r84_weighted = r84_row_grad * r84_row_values
    r84_dot_grad = tl.sum(r84_weighted, axis=1)
    r84_value = (tl.load($(candidate_vslice_backward_ptr) + ((((r84_z / $(16512)) % $(8)) * $(16512)) + (((r84_z / $(129)) % $(128)) * $(129)) + (r84_z % $(129))), dtype=tl.float32) - r84_dot_grad) * tl.load($(candidate_vsoftmax_1_ptr) + ((((r84_z / $(16512)) % $(8)) * $(16512)) + (((r84_z / $(129)) % $(128)) * $(129)) + (r84_z % $(129))), dtype=tl.float32)
    r84_value = r84_value.to(tl.float32)
    r84_value = tl.broadcast_to(r84_value, [$(132096)])
    tl.store($(candidate_vsoftmax_backward_data_ptr) + r84_z, r84_value)
    r85_z = tl.arange($(131072))
    r85_value = tl.where(tl.load($(candidate_vgt_1_ptr) + ((((r85_z / $(128)) % $(128)) * $(128)) + (r85_z % $(128))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_ptr) + ((((r85_z / $(16384)) % $(8)) * $(16512)) + (((r85_z / $(128)) % $(128)) * $(129)) + (r85_z % $(128))), dtype=tl.float32))
    r85_value = r85_value.to(tl.float32)
    r85_value = tl.broadcast_to(r85_value, [$(131072)])
    tl.store($(candidate_vmasked_fill_4_ptr) + r85_z, r85_value)
    r86_z = tl.arange($(131072))
    r86_value = (tl.load($(candidate_vmasked_fill_4_ptr) + ((((r86_z / $(16384)) % $(8)) * $(16384)) + (((r86_z / $(128)) % $(128)) * $(128)) + (r86_z % $(128))), dtype=tl.float32) * 0.07216878364870322)
    r86_value = r86_value.to(tl.float32)
    r86_value = tl.broadcast_to(r86_value, [$(131072)])
    tl.store($(candidate_vmul_22_ptr) + r86_z, r86_value)
    r87_row = tl.arange($(192))
    r87_col = tl.arange($(128))
    r87_lane = tl.arange($(128))
    for r87_group in range($(8)) {
      r87_left = tl.load($(candidate_vto_copy_26_ptr) + (((((((r87_group % $(8)) * $(24576)) + (r87_lane[None, :] * $(192)) + r87_row[:, None]) / $(24576)) % $(8)) * $(24576)) + ((((((r87_group % $(8)) * $(24576)) + (r87_lane[None, :] * $(192)) + r87_row[:, None]) / $(192)) % $(128)) * $(192)) + ((((r87_group % $(8)) * $(24576)) + (r87_lane[None, :] * $(192)) + r87_row[:, None]) % $(192))), dtype=tl.float32)
      r87_right = tl.load($(candidate_vmul_22_ptr) + (((((((r87_group % $(8)) * $(16384)) + (r87_lane[:, None] * $(128)) + r87_col[None, :]) / $(16384)) % $(8)) * $(16384)) + ((((((r87_group % $(8)) * $(16384)) + (r87_lane[:, None] * $(128)) + r87_col[None, :]) / $(128)) % $(128)) * $(128)) + ((((r87_group % $(8)) * $(16384)) + (r87_lane[:, None] * $(128)) + r87_col[None, :]) % $(128))), dtype=tl.float32)
      r87_value = tl.dot(r87_left, r87_right)
      r87_value = r87_value.to(tl.float32)
      tl.store($(candidate_vbmm_8_ptr) + ((r87_group * $(24576)) + (r87_row[:, None] * $(128)) + r87_col[None, :]), r87_value)
    }
    r88_row = tl.arange($(128))
    r88_col = tl.arange($(192))
    r88_lane = tl.arange($(128))
    for r88_group in range($(8)) {
      r88_left = tl.load($(candidate_vmul_22_ptr) + (((((((r88_group % $(8)) * $(16384)) + (r88_row[:, None] * $(128)) + r88_lane[None, :]) / $(16384)) % $(8)) * $(16384)) + ((((((r88_group % $(8)) * $(16384)) + (r88_row[:, None] * $(128)) + r88_lane[None, :]) / $(128)) % $(128)) * $(128)) + ((((r88_group % $(8)) * $(16384)) + (r88_row[:, None] * $(128)) + r88_lane[None, :]) % $(128))), dtype=tl.float32)
      r88_right = tl.load($(candidate_vto_copy_27_ptr) + (((((((r88_group % $(8)) * $(24576)) + (r88_col[None, :] * $(128)) + r88_lane[:, None]) / $(24576)) % $(8)) * $(24576)) + (((((r88_group % $(8)) * $(24576)) + (r88_col[None, :] * $(128)) + r88_lane[:, None]) % $(128)) * $(192)) + (((((r88_group % $(8)) * $(24576)) + (r88_col[None, :] * $(128)) + r88_lane[:, None]) / $(128)) % $(192))), dtype=tl.float32)
      r88_value = tl.dot(r88_left, r88_right)
      r88_value = r88_value.to(tl.float32)
      tl.store($(candidate_vbmm_9_ptr) + ((r88_group * $(24576)) + (r88_row[:, None] * $(192)) + r88_col[None, :]), r88_value)
    }
    r89_z = tl.arange($(262144))
    r89_value = tl.where(((((r89_z / $(1024)) % $(256)) >= $(128)) & (((r89_z / $(1024)) % $(256)) < $(256)) & ((((r89_z / $(1024)) % $(256)) - $(128)) % $(1) == $(0))), tl.load($(candidate_vbmm_6_ptr) + ((((((((r89_z / $(128)) % $(8)) * $(16384)) + ((((r89_z / $(1024)) % $(256)) - $(128)) * $(128)) + (r89_z % $(128))) / $(16384)) % $(8)) * $(16384)) + ((((((r89_z / $(128)) % $(8)) * $(128)) + (((r89_z / $(1024)) % $(256)) - $(128))) % $(128)) * $(128)) + (r89_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r89_value = r89_value.to(tl.bfloat16)
    r89_value = tl.broadcast_to(r89_value, [$(262144)])
    tl.store($(candidate_vslice_backward_1_ptr) + r89_z, r89_value)
    r90_z = tl.arange($(196608))
    r90_value = tl.load($(candidate_vbmm_8_ptr) + ((((r90_z / $(24576)) % $(8)) * $(24576)) + ((r90_z % $(192)) * $(128)) + ((r90_z / $(192)) % $(128))), dtype=tl.float32)
    r90_value = r90_value.to(tl.bfloat16)
    r90_value = tl.broadcast_to(r90_value, [$(196608)])
    tl.store($(candidate_vto_copy_31_ptr) + r90_z, r90_value)
    r91_z = tl.arange($(393216))
    r91_value = tl.where(((((r91_z / $(1536)) % $(256)) >= $(128)) & (((r91_z / $(1536)) % $(256)) < $(256)) & ((((r91_z / $(1536)) % $(256)) - $(128)) % $(1) == $(0))), tl.load($(candidate_vto_copy_31_ptr) + ((((r91_z / $(192)) % $(8)) * $(24576)) + (((((r91_z / $(1536)) % $(256)) - $(128)) % $(128)) * $(192)) + (r91_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r91_value = r91_value.to(tl.bfloat16)
    r91_value = tl.broadcast_to(r91_value, [$(393216)])
    tl.store($(candidate_vslice_backward_2_ptr) + r91_z, r91_value)
    r92_z = tl.arange($(196608))
    r92_value = tl.load($(candidate_vbmm_9_ptr) + ((((r92_z / $(24576)) % $(8)) * $(24576)) + (((r92_z / $(192)) % $(128)) * $(192)) + (r92_z % $(192))), dtype=tl.float32)
    r92_value = r92_value.to(tl.bfloat16)
    r92_value = tl.broadcast_to(r92_value, [$(196608)])
    tl.store($(candidate_vto_copy_32_ptr) + r92_z, r92_value)
    r93_z = tl.arange($(393216))
    r93_value = tl.where(((((r93_z / $(1536)) % $(256)) >= $(128)) & (((r93_z / $(1536)) % $(256)) < $(256)) & ((((r93_z / $(1536)) % $(256)) - $(128)) % $(1) == $(0))), tl.load($(candidate_vto_copy_32_ptr) + ((((r93_z / $(192)) % $(8)) * $(24576)) + (((((r93_z / $(1536)) % $(256)) - $(128)) % $(128)) * $(192)) + (r93_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r93_value = r93_value.to(tl.bfloat16)
    r93_value = tl.broadcast_to(r93_value, [$(393216)])
    tl.store($(candidate_vslice_backward_3_ptr) + r93_z, r93_value)
    r94_row = tl.arange($(128))
    r94_col = tl.arange($(128))
    r94_lane = tl.arange($(128))
    for r94_group in range($(8)) {
      r94_left = tl.load($(candidate_vto_copy_25_ptr) + (((((((r94_group % $(8)) * $(16384)) + (r94_lane[None, :] * $(128)) + r94_row[:, None]) / $(16384)) % $(8)) * $(16384)) + ((((((r94_group % $(8)) * $(16384)) + (r94_lane[None, :] * $(128)) + r94_row[:, None]) / $(128)) % $(128)) * $(128)) + ((((r94_group % $(8)) * $(16384)) + (r94_lane[None, :] * $(128)) + r94_row[:, None]) % $(128))), dtype=tl.bfloat16)
      r94_right = tl.load($(candidate_vmm_4_ptr) + (((((((r94_group % $(8)) * $(16384)) + (r94_lane[:, None] * $(128)) + r94_col[None, :]) / $(128)) % $(128)) * $(1024)) + ((((((r94_group % $(8)) * $(16384)) + (r94_lane[:, None] * $(128)) + r94_col[None, :]) / $(16384)) % $(8)) * $(128)) + ((((r94_group % $(8)) * $(16384)) + (r94_lane[:, None] * $(128)) + r94_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r94_value = tl.dot(r94_left, r94_right)
      r94_value = r94_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_10_ptr) + ((r94_group * $(16384)) + (r94_row[:, None] * $(128)) + r94_col[None, :]), r94_value)
    }
    r95_row = tl.arange($(128))
    r95_col = tl.arange($(128))
    r95_lane = tl.arange($(128))
    for r95_group in range($(8)) {
      r95_left = tl.load($(candidate_vmm_4_ptr) + (((((((r95_group % $(8)) * $(16384)) + (r95_row[:, None] * $(128)) + r95_lane[None, :]) / $(128)) % $(128)) * $(1024)) + ((((((r95_group % $(8)) * $(16384)) + (r95_row[:, None] * $(128)) + r95_lane[None, :]) / $(16384)) % $(8)) * $(128)) + ((((r95_group % $(8)) * $(16384)) + (r95_row[:, None] * $(128)) + r95_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r95_right = tl.load($(candidate_vbmm_1_ptr) + (((((((r95_group % $(8)) * $(16384)) + (r95_col[None, :] * $(128)) + r95_lane[:, None]) / $(128)) % $(128)) * $(1024)) + ((((((r95_group % $(8)) * $(16384)) + (r95_col[None, :] * $(128)) + r95_lane[:, None]) / $(16384)) % $(8)) * $(128)) + ((((r95_group % $(8)) * $(16384)) + (r95_col[None, :] * $(128)) + r95_lane[:, None]) % $(128))), dtype=tl.bfloat16)
      r95_value = tl.dot(r95_left, r95_right)
      r95_value = r95_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_11_ptr) + ((r95_group * $(16384)) + (r95_row[:, None] * $(128)) + r95_col[None, :]), r95_value)
    }
    r96_z = tl.arange($(131072))
    r96_value = tl.load($(candidate_vbmm_11_ptr) + ((((r96_z / $(16384)) % $(8)) * $(16384)) + (((r96_z / $(128)) % $(128)) * $(128)) + (r96_z % $(128))), dtype=tl.bfloat16)
    r96_value = r96_value.to(tl.float32)
    r96_value = tl.broadcast_to(r96_value, [$(131072)])
    tl.store($(candidate_vto_copy_33_ptr) + r96_z, r96_value)
    r97_z = tl.arange($(132096))
    r97_value = tl.where((((r97_z % $(129)) >= $(0)) & ((r97_z % $(129)) < $(128)) & (((r97_z % $(129)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vto_copy_33_ptr) + ((((r97_z / $(16512)) % $(8)) * $(16384)) + (((r97_z / $(129)) % $(128)) * $(128)) + (((r97_z % $(129)) - $(0)) % $(128))), dtype=tl.float32), 0.0)
    r97_value = r97_value.to(tl.float32)
    r97_value = tl.broadcast_to(r97_value, [$(132096)])
    tl.store($(candidate_vslice_backward_4_ptr) + r97_z, r97_value)
    r98_z = tl.arange($(132096))
    r98_lane = tl.arange($(129))
    r98_row_values = tl.load($(candidate_vsoftmax_ptr) + ((((r98_z[:, None] / $(16512)) % $(8)) * $(16512)) + (((r98_z[:, None] / $(129)) % $(128)) * $(129)) + (r98_lane[None, :] % $(129))), dtype=tl.float32)
    r98_row_grad = tl.load($(candidate_vslice_backward_4_ptr) + ((((r98_z[:, None] / $(16512)) % $(8)) * $(16512)) + (((r98_z[:, None] / $(129)) % $(128)) * $(129)) + (r98_lane[None, :] % $(129))), dtype=tl.float32)
    r98_weighted = r98_row_grad * r98_row_values
    r98_dot_grad = tl.sum(r98_weighted, axis=1)
    r98_value = (tl.load($(candidate_vslice_backward_4_ptr) + ((((r98_z / $(16512)) % $(8)) * $(16512)) + (((r98_z / $(129)) % $(128)) * $(129)) + (r98_z % $(129))), dtype=tl.float32) - r98_dot_grad) * tl.load($(candidate_vsoftmax_ptr) + ((((r98_z / $(16512)) % $(8)) * $(16512)) + (((r98_z / $(129)) % $(128)) * $(129)) + (r98_z % $(129))), dtype=tl.float32)
    r98_value = r98_value.to(tl.float32)
    r98_value = tl.broadcast_to(r98_value, [$(132096)])
    tl.store($(candidate_vsoftmax_backward_data_1_ptr) + r98_z, r98_value)
    r99_z = tl.arange($(131072))
    r99_value = tl.where(tl.load($(candidate_vgt_ptr) + ((((r99_z / $(128)) % $(128)) * $(128)) + (r99_z % $(128))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_1_ptr) + ((((r99_z / $(16384)) % $(8)) * $(16512)) + (((r99_z / $(128)) % $(128)) * $(129)) + (r99_z % $(128))), dtype=tl.float32))
    r99_value = r99_value.to(tl.float32)
    r99_value = tl.broadcast_to(r99_value, [$(131072)])
    tl.store($(candidate_vmasked_fill_5_ptr) + r99_z, r99_value)
    r100_z = tl.arange($(131072))
    r100_value = (tl.load($(candidate_vmasked_fill_5_ptr) + ((((r100_z / $(16384)) % $(8)) * $(16384)) + (((r100_z / $(128)) % $(128)) * $(128)) + (r100_z % $(128))), dtype=tl.float32) * 0.07216878364870322)
    r100_value = r100_value.to(tl.float32)
    r100_value = tl.broadcast_to(r100_value, [$(131072)])
    tl.store($(candidate_vmul_23_ptr) + r100_z, r100_value)
    r101_row = tl.arange($(192))
    r101_col = tl.arange($(128))
    r101_lane = tl.arange($(128))
    for r101_group in range($(8)) {
      r101_left = tl.load($(candidate_vto_copy_22_ptr) + (((((((r101_group % $(8)) * $(24576)) + (r101_lane[None, :] * $(192)) + r101_row[:, None]) / $(24576)) % $(8)) * $(24576)) + ((((((r101_group % $(8)) * $(24576)) + (r101_lane[None, :] * $(192)) + r101_row[:, None]) / $(192)) % $(128)) * $(192)) + ((((r101_group % $(8)) * $(24576)) + (r101_lane[None, :] * $(192)) + r101_row[:, None]) % $(192))), dtype=tl.float32)
      r101_right = tl.load($(candidate_vmul_23_ptr) + (((((((r101_group % $(8)) * $(16384)) + (r101_lane[:, None] * $(128)) + r101_col[None, :]) / $(16384)) % $(8)) * $(16384)) + ((((((r101_group % $(8)) * $(16384)) + (r101_lane[:, None] * $(128)) + r101_col[None, :]) / $(128)) % $(128)) * $(128)) + ((((r101_group % $(8)) * $(16384)) + (r101_lane[:, None] * $(128)) + r101_col[None, :]) % $(128))), dtype=tl.float32)
      r101_value = tl.dot(r101_left, r101_right)
      r101_value = r101_value.to(tl.float32)
      tl.store($(candidate_vbmm_12_ptr) + ((r101_group * $(24576)) + (r101_row[:, None] * $(128)) + r101_col[None, :]), r101_value)
    }
    r102_row = tl.arange($(128))
    r102_col = tl.arange($(192))
    r102_lane = tl.arange($(128))
    for r102_group in range($(8)) {
      r102_left = tl.load($(candidate_vmul_23_ptr) + (((((((r102_group % $(8)) * $(16384)) + (r102_row[:, None] * $(128)) + r102_lane[None, :]) / $(16384)) % $(8)) * $(16384)) + ((((((r102_group % $(8)) * $(16384)) + (r102_row[:, None] * $(128)) + r102_lane[None, :]) / $(128)) % $(128)) * $(128)) + ((((r102_group % $(8)) * $(16384)) + (r102_row[:, None] * $(128)) + r102_lane[None, :]) % $(128))), dtype=tl.float32)
      r102_right = tl.load($(candidate_vto_copy_23_ptr) + (((((((r102_group % $(8)) * $(24576)) + (r102_col[None, :] * $(128)) + r102_lane[:, None]) / $(24576)) % $(8)) * $(24576)) + (((((r102_group % $(8)) * $(24576)) + (r102_col[None, :] * $(128)) + r102_lane[:, None]) % $(128)) * $(192)) + (((((r102_group % $(8)) * $(24576)) + (r102_col[None, :] * $(128)) + r102_lane[:, None]) / $(128)) % $(192))), dtype=tl.float32)
      r102_value = tl.dot(r102_left, r102_right)
      r102_value = r102_value.to(tl.float32)
      tl.store($(candidate_vbmm_13_ptr) + ((r102_group * $(24576)) + (r102_row[:, None] * $(192)) + r102_col[None, :]), r102_value)
    }
    r103_z = tl.arange($(262144))
    r103_value = tl.where(((((r103_z / $(1024)) % $(256)) >= $(0)) & (((r103_z / $(1024)) % $(256)) < $(128)) & ((((r103_z / $(1024)) % $(256)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vbmm_10_ptr) + ((((((((r103_z / $(128)) % $(8)) * $(16384)) + ((((r103_z / $(1024)) % $(256)) - $(0)) * $(128)) + (r103_z % $(128))) / $(16384)) % $(8)) * $(16384)) + ((((((r103_z / $(128)) % $(8)) * $(128)) + (((r103_z / $(1024)) % $(256)) - $(0))) % $(128)) * $(128)) + (r103_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r103_value = r103_value.to(tl.bfloat16)
    r103_value = tl.broadcast_to(r103_value, [$(262144)])
    tl.store($(candidate_vslice_backward_5_ptr) + r103_z, r103_value)
    r104_z = tl.arange($(262144))
    r104_value = (tl.load($(candidate_vslice_backward_1_ptr) + ((((r104_z / $(1024)) % $(256)) * $(1024)) + (((r104_z / $(128)) % $(8)) * $(128)) + (r104_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_5_ptr) + ((((r104_z / $(1024)) % $(256)) * $(1024)) + (((r104_z / $(128)) % $(8)) * $(128)) + (r104_z % $(128))), dtype=tl.bfloat16))
    r104_value = r104_value.to(tl.bfloat16)
    r104_value = tl.broadcast_to(r104_value, [$(262144)])
    tl.store($(candidate_vadd_8_ptr) + r104_z, r104_value)
    r105_z = tl.arange($(196608))
    r105_value = tl.load($(candidate_vbmm_12_ptr) + ((((r105_z / $(24576)) % $(8)) * $(24576)) + ((r105_z % $(192)) * $(128)) + ((r105_z / $(192)) % $(128))), dtype=tl.float32)
    r105_value = r105_value.to(tl.bfloat16)
    r105_value = tl.broadcast_to(r105_value, [$(196608)])
    tl.store($(candidate_vto_copy_34_ptr) + r105_z, r105_value)
    r106_z = tl.arange($(393216))
    r106_value = tl.where(((((r106_z / $(1536)) % $(256)) >= $(0)) & (((r106_z / $(1536)) % $(256)) < $(128)) & ((((r106_z / $(1536)) % $(256)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vto_copy_34_ptr) + ((((r106_z / $(192)) % $(8)) * $(24576)) + (((((r106_z / $(1536)) % $(256)) - $(0)) % $(128)) * $(192)) + (r106_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r106_value = r106_value.to(tl.bfloat16)
    r106_value = tl.broadcast_to(r106_value, [$(393216)])
    tl.store($(candidate_vslice_backward_6_ptr) + r106_z, r106_value)
    r107_z = tl.arange($(393216))
    r107_value = (tl.load($(candidate_vslice_backward_2_ptr) + ((((r107_z / $(1536)) % $(256)) * $(1536)) + (((r107_z / $(192)) % $(8)) * $(192)) + (r107_z % $(192))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_6_ptr) + ((((r107_z / $(1536)) % $(256)) * $(1536)) + (((r107_z / $(192)) % $(8)) * $(192)) + (r107_z % $(192))), dtype=tl.bfloat16))
    r107_value = r107_value.to(tl.bfloat16)
    r107_value = tl.broadcast_to(r107_value, [$(393216)])
    tl.store($(candidate_vadd_9_ptr) + r107_z, r107_value)
    r108_z = tl.arange($(196608))
    r108_value = tl.load($(candidate_vbmm_13_ptr) + ((((r108_z / $(24576)) % $(8)) * $(24576)) + (((r108_z / $(192)) % $(128)) * $(192)) + (r108_z % $(192))), dtype=tl.float32)
    r108_value = r108_value.to(tl.bfloat16)
    r108_value = tl.broadcast_to(r108_value, [$(196608)])
    tl.store($(candidate_vto_copy_35_ptr) + r108_z, r108_value)
    r109_z = tl.arange($(393216))
    r109_value = tl.where(((((r109_z / $(1536)) % $(256)) >= $(0)) & (((r109_z / $(1536)) % $(256)) < $(128)) & ((((r109_z / $(1536)) % $(256)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vto_copy_35_ptr) + ((((r109_z / $(192)) % $(8)) * $(24576)) + (((((r109_z / $(1536)) % $(256)) - $(0)) % $(128)) * $(192)) + (r109_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r109_value = r109_value.to(tl.bfloat16)
    r109_value = tl.broadcast_to(r109_value, [$(393216)])
    tl.store($(candidate_vslice_backward_7_ptr) + r109_z, r109_value)
    r110_z = tl.arange($(393216))
    r110_value = (tl.load($(candidate_vslice_backward_3_ptr) + ((((r110_z / $(1536)) % $(256)) * $(1536)) + (((r110_z / $(192)) % $(8)) * $(192)) + (r110_z % $(192))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_7_ptr) + ((((r110_z / $(1536)) % $(256)) * $(1536)) + (((r110_z / $(192)) % $(8)) * $(192)) + (r110_z % $(192))), dtype=tl.bfloat16))
    r110_value = r110_value.to(tl.bfloat16)
    r110_value = tl.broadcast_to(r110_value, [$(393216)])
    tl.store($(candidate_vadd_10_ptr) + r110_z, r110_value)
    r111_z = tl.arange($(16384))
    r111_lane = tl.arange($(8))
    r111_value = tl.sum(tl.load($(candidate_vadd_9_ptr) + ((((r111_z[:, None] / $(64)) % $(256)) * $(1536)) + (r111_lane[None, :] * $(192)) + (r111_z[:, None] % $(64)) + $(128)), dtype=tl.bfloat16), axis=1)
    r111_value = r111_value.to(tl.bfloat16)
    r111_value = tl.broadcast_to(r111_value, [$(16384)])
    tl.store($(candidate_vsum_1_ptr) + r111_z, r111_value)
    r112_row = tl.arange($(256))
    r112_col = tl.arange($(256))
    r112_lane = tl.arange($(1024))
    for r112_group in range($(1)) {
      r112_left = tl.load($(candidate_vadd_8_ptr) + ((((((r112_row[:, None] * $(1024)) + r112_lane[None, :]) / $(1024)) % $(256)) * $(1024)) + (((((r112_row[:, None] * $(1024)) + r112_lane[None, :]) / $(128)) % $(8)) * $(128)) + (((r112_row[:, None] * $(1024)) + r112_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r112_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r112_col[None, :] * $(1024)) + r112_lane[:, None]) / $(128)) % $(8)) * $(65536)) + ((((r112_col[None, :] * $(1024)) + r112_lane[:, None]) % $(128)) * $(256)) + ((((r112_col[None, :] * $(1024)) + r112_lane[:, None]) / $(1024)) % $(256)) + $(32768)), dtype=tl.bfloat16)
      r112_value = tl.dot(r112_left, r112_right)
      r112_value = r112_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_14_ptr) + ((r112_group * $(65536)) + (r112_row[:, None] * $(256)) + r112_col[None, :]), r112_value)
    }
    r113_row = tl.arange($(256))
    r113_col = tl.arange($(256))
    r113_lane = tl.arange($(1024))
    for r113_group in range($(1)) {
      r113_left = tl.load($(candidate_vadd_9_ptr) + ((((((r113_row[:, None] * $(1024)) + r113_lane[None, :]) / $(1024)) % $(256)) * $(1536)) + (((((r113_row[:, None] * $(1024)) + r113_lane[None, :]) / $(128)) % $(8)) * $(192)) + (((r113_row[:, None] * $(1024)) + r113_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r113_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r113_col[None, :] * $(1024)) + r113_lane[:, None]) / $(128)) % $(8)) * $(65536)) + ((((r113_col[None, :] * $(1024)) + r113_lane[:, None]) % $(128)) * $(256)) + ((((r113_col[None, :] * $(1024)) + r113_lane[:, None]) / $(1024)) % $(256))), dtype=tl.bfloat16)
      r113_value = tl.dot(r113_left, r113_right)
      r113_value = r113_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_15_ptr) + ((r113_group * $(65536)) + (r113_row[:, None] * $(256)) + r113_col[None, :]), r113_value)
    }
    r114_z = tl.arange($(65536))
    r114_value = (tl.load($(candidate_vbmm_14_ptr) + ((((r114_z / $(256)) % $(256)) * $(256)) + (r114_z % $(256))), dtype=tl.bfloat16) + tl.load($(candidate_vbmm_15_ptr) + ((((r114_z / $(256)) % $(256)) * $(256)) + (r114_z % $(256))), dtype=tl.bfloat16))
    r114_value = r114_value.to(tl.bfloat16)
    r114_value = tl.broadcast_to(r114_value, [$(65536)])
    tl.store($(candidate_vadd_11_ptr) + r114_z, r114_value)
    r115_z = tl.arange($(65536))
    r115_value = tl.where(($(0) == $(0)), tl.load($(candidate_vadd_11_ptr) + ((((r115_z / $(256)) % $(256)) * $(256)) + (r115_z % $(256))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r115_value = r115_value.to(tl.bfloat16)
    r115_value = tl.broadcast_to(r115_value, [$(65536)])
    tl.store($(candidate_vselect_backward_ptr) + r115_z, r115_value)
    r116_z = tl.arange($(8192))
    r116_value = (tl.load($(candidate_vsum_1_ptr) + ((((r116_z / $(4096)) % $(2)) * $(8192)) + (((r116_z / $(32)) % $(128)) * $(64)) + (r116_z % $(32)) + $(32)), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r116_value = r116_value.to(tl.bfloat16)
    r116_value = tl.broadcast_to(r116_value, [$(8192)])
    tl.store($(candidate_vmul_24_ptr) + r116_z, r116_value)
    r117_z = tl.arange($(8192))
    r117_value = (tl.load($(candidate_vsum_1_ptr) + ((((r117_z / $(4096)) % $(2)) * $(8192)) + (((r117_z / $(32)) % $(128)) * $(64)) + (r117_z % $(32)) + $(32)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r117_value = r117_value.to(tl.bfloat16)
    r117_value = tl.broadcast_to(r117_value, [$(8192)])
    tl.store($(candidate_vmul_25_ptr) + r117_z, r117_value)
    r118_z = tl.arange($(8192))
    r118_value = -tl.load($(candidate_vsum_1_ptr) + ((((r118_z / $(4096)) % $(2)) * $(8192)) + (((r118_z / $(32)) % $(128)) * $(64)) + (r118_z % $(32))), dtype=tl.bfloat16)
    r118_value = r118_value.to(tl.bfloat16)
    r118_value = tl.broadcast_to(r118_value, [$(8192)])
    tl.store($(candidate_vneg_ptr) + r118_z, r118_value)
    r119_z = tl.arange($(8192))
    r119_value = (tl.load($(candidate_vneg_ptr) + ((((r119_z / $(4096)) % $(2)) * $(4096)) + (((r119_z / $(32)) % $(128)) * $(32)) + (r119_z % $(32))), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r119_value = r119_value.to(tl.bfloat16)
    r119_value = tl.broadcast_to(r119_value, [$(8192)])
    tl.store($(candidate_vmul_26_ptr) + r119_z, r119_value)
    r120_z = tl.arange($(8192))
    r120_value = (tl.load($(candidate_vmul_25_ptr) + ((((r120_z / $(4096)) % $(2)) * $(4096)) + (((r120_z / $(32)) % $(128)) * $(32)) + (r120_z % $(32))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_26_ptr) + ((((r120_z / $(4096)) % $(2)) * $(4096)) + (((r120_z / $(32)) % $(128)) * $(32)) + (r120_z % $(32))), dtype=tl.bfloat16))
    r120_value = r120_value.to(tl.bfloat16)
    r120_value = tl.broadcast_to(r120_value, [$(8192)])
    tl.store($(candidate_vadd_12_ptr) + r120_z, r120_value)
    r121_z = tl.arange($(8192))
    r121_value = (tl.load($(candidate_vsum_1_ptr) + ((((r121_z / $(4096)) % $(2)) * $(8192)) + (((r121_z / $(32)) % $(128)) * $(64)) + (r121_z % $(32))), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r121_value = r121_value.to(tl.bfloat16)
    r121_value = tl.broadcast_to(r121_value, [$(8192)])
    tl.store($(candidate_vmul_27_ptr) + r121_z, r121_value)
    r122_z = tl.arange($(8192))
    r122_value = (tl.load($(candidate_vmul_24_ptr) + ((((r122_z / $(4096)) % $(2)) * $(4096)) + (((r122_z / $(32)) % $(128)) * $(32)) + (r122_z % $(32))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_27_ptr) + ((((r122_z / $(4096)) % $(2)) * $(4096)) + (((r122_z / $(32)) % $(128)) * $(32)) + (r122_z % $(32))), dtype=tl.bfloat16))
    r122_value = r122_value.to(tl.bfloat16)
    r122_value = tl.broadcast_to(r122_value, [$(8192)])
    tl.store($(candidate_vadd_13_ptr) + r122_z, r122_value)
    r123_z = tl.arange($(65536))
    r123_value = (tl.load($(candidate_vadd_10_ptr) + ((((r123_z / $(32768)) % $(2)) * $(196608)) + (((r123_z / $(32)) % $(128)) * $(1536)) + (((r123_z / $(4096)) % $(8)) * $(192)) + (r123_z % $(32)) + $(160)), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r123_value = r123_value.to(tl.bfloat16)
    r123_value = tl.broadcast_to(r123_value, [$(65536)])
    tl.store($(candidate_vmul_28_ptr) + r123_z, r123_value)
    r124_z = tl.arange($(65536))
    r124_value = (tl.load($(candidate_vadd_10_ptr) + ((((r124_z / $(32768)) % $(2)) * $(196608)) + (((r124_z / $(32)) % $(128)) * $(1536)) + (((r124_z / $(4096)) % $(8)) * $(192)) + (r124_z % $(32)) + $(160)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r124_value = r124_value.to(tl.bfloat16)
    r124_value = tl.broadcast_to(r124_value, [$(65536)])
    tl.store($(candidate_vmul_29_ptr) + r124_z, r124_value)
    r125_z = tl.arange($(65536))
    r125_value = -tl.load($(candidate_vadd_10_ptr) + ((((r125_z / $(32768)) % $(2)) * $(196608)) + (((r125_z / $(32)) % $(128)) * $(1536)) + (((r125_z / $(4096)) % $(8)) * $(192)) + (r125_z % $(32)) + $(128)), dtype=tl.bfloat16)
    r125_value = r125_value.to(tl.bfloat16)
    r125_value = tl.broadcast_to(r125_value, [$(65536)])
    tl.store($(candidate_vneg_1_ptr) + r125_z, r125_value)
    r126_z = tl.arange($(65536))
    r126_value = (tl.load($(candidate_vneg_1_ptr) + ((((r126_z / $(32768)) % $(2)) * $(32768)) + (((r126_z / $(4096)) % $(8)) * $(4096)) + (((r126_z / $(32)) % $(128)) * $(32)) + (r126_z % $(32))), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r126_value = r126_value.to(tl.bfloat16)
    r126_value = tl.broadcast_to(r126_value, [$(65536)])
    tl.store($(candidate_vmul_30_ptr) + r126_z, r126_value)
    r127_z = tl.arange($(65536))
    r127_value = (tl.load($(candidate_vmul_29_ptr) + ((((r127_z / $(32768)) % $(2)) * $(32768)) + (((r127_z / $(4096)) % $(8)) * $(4096)) + (((r127_z / $(32)) % $(128)) * $(32)) + (r127_z % $(32))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_30_ptr) + ((((r127_z / $(32768)) % $(2)) * $(32768)) + (((r127_z / $(4096)) % $(8)) * $(4096)) + (((r127_z / $(32)) % $(128)) * $(32)) + (r127_z % $(32))), dtype=tl.bfloat16))
    r127_value = r127_value.to(tl.bfloat16)
    r127_value = tl.broadcast_to(r127_value, [$(65536)])
    tl.store($(candidate_vadd_14_ptr) + r127_z, r127_value)
    r128_z = tl.arange($(65536))
    r128_value = (tl.load($(candidate_vadd_10_ptr) + ((((r128_z / $(32768)) % $(2)) * $(196608)) + (((r128_z / $(32)) % $(128)) * $(1536)) + (((r128_z / $(4096)) % $(8)) * $(192)) + (r128_z % $(32)) + $(128)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r128_value = r128_value.to(tl.bfloat16)
    r128_value = tl.broadcast_to(r128_value, [$(65536)])
    tl.store($(candidate_vmul_31_ptr) + r128_z, r128_value)
    r129_z = tl.arange($(65536))
    r129_value = (tl.load($(candidate_vmul_28_ptr) + ((((r129_z / $(32768)) % $(2)) * $(32768)) + (((r129_z / $(4096)) % $(8)) * $(4096)) + (((r129_z / $(32)) % $(128)) * $(32)) + (r129_z % $(32))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_31_ptr) + ((((r129_z / $(32768)) % $(2)) * $(32768)) + (((r129_z / $(4096)) % $(8)) * $(4096)) + (((r129_z / $(32)) % $(128)) * $(32)) + (r129_z % $(32))), dtype=tl.bfloat16))
    r129_value = r129_value.to(tl.bfloat16)
    r129_value = tl.broadcast_to(r129_value, [$(65536)])
    tl.store($(candidate_vadd_15_ptr) + r129_z, r129_value)
    r130_z = tl.arange($(16384))
    r130_value = tl.where((((r130_z % $(64)) >= $(1)) & ((r130_z % $(64)) < $(64)) & (((r130_z % $(64)) - $(1)) % $(2) == $(0))), tl.load($(candidate_vadd_12_ptr) + ((((r130_z / $(8192)) % $(2)) * $(4096)) + (((r130_z / $(64)) % $(128)) * $(32)) + ((((r130_z % $(64)) - $(1)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r130_value = r130_value.to(tl.bfloat16)
    r130_value = tl.broadcast_to(r130_value, [$(16384)])
    tl.store($(candidate_vslice_backward_8_ptr) + r130_z, r130_value)
    r131_z = tl.arange($(16384))
    r131_value = tl.where((((r131_z % $(64)) >= $(0)) & ((r131_z % $(64)) < $(64)) & (((r131_z % $(64)) - $(0)) % $(2) == $(0))), tl.load($(candidate_vadd_13_ptr) + ((((r131_z / $(8192)) % $(2)) * $(4096)) + (((r131_z / $(64)) % $(128)) * $(32)) + ((((r131_z % $(64)) - $(0)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r131_value = r131_value.to(tl.bfloat16)
    r131_value = tl.broadcast_to(r131_value, [$(16384)])
    tl.store($(candidate_vslice_backward_9_ptr) + r131_z, r131_value)
    r132_z = tl.arange($(16384))
    r132_value = (tl.load($(candidate_vslice_backward_8_ptr) + ((((r132_z / $(8192)) % $(2)) * $(8192)) + (((r132_z / $(64)) % $(128)) * $(64)) + (r132_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_9_ptr) + ((((r132_z / $(8192)) % $(2)) * $(8192)) + (((r132_z / $(64)) % $(128)) * $(64)) + (r132_z % $(64))), dtype=tl.bfloat16))
    r132_value = r132_value.to(tl.bfloat16)
    r132_value = tl.broadcast_to(r132_value, [$(16384)])
    tl.store($(candidate_vadd_16_ptr) + r132_z, r132_value)
    r133_z = tl.arange($(131072))
    r133_value = tl.where((((r133_z % $(64)) >= $(1)) & ((r133_z % $(64)) < $(64)) & (((r133_z % $(64)) - $(1)) % $(2) == $(0))), tl.load($(candidate_vadd_14_ptr) + ((((r133_z / $(65536)) % $(2)) * $(32768)) + (((r133_z / $(8192)) % $(8)) * $(4096)) + (((r133_z / $(64)) % $(128)) * $(32)) + ((((r133_z % $(64)) - $(1)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r133_value = r133_value.to(tl.bfloat16)
    r133_value = tl.broadcast_to(r133_value, [$(131072)])
    tl.store($(candidate_vslice_backward_10_ptr) + r133_z, r133_value)
    r134_z = tl.arange($(131072))
    r134_value = tl.where((((r134_z % $(64)) >= $(0)) & ((r134_z % $(64)) < $(64)) & (((r134_z % $(64)) - $(0)) % $(2) == $(0))), tl.load($(candidate_vadd_15_ptr) + ((((r134_z / $(65536)) % $(2)) * $(32768)) + (((r134_z / $(8192)) % $(8)) * $(4096)) + (((r134_z / $(64)) % $(128)) * $(32)) + ((((r134_z % $(64)) - $(0)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r134_value = r134_value.to(tl.bfloat16)
    r134_value = tl.broadcast_to(r134_value, [$(131072)])
    tl.store($(candidate_vslice_backward_11_ptr) + r134_z, r134_value)
    r135_z = tl.arange($(131072))
    r135_value = (tl.load($(candidate_vslice_backward_10_ptr) + ((((r135_z / $(65536)) % $(2)) * $(65536)) + (((r135_z / $(8192)) % $(8)) * $(8192)) + (((r135_z / $(64)) % $(128)) * $(64)) + (r135_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_11_ptr) + ((((r135_z / $(65536)) % $(2)) * $(65536)) + (((r135_z / $(8192)) % $(8)) * $(8192)) + (((r135_z / $(64)) % $(128)) * $(64)) + (r135_z % $(64))), dtype=tl.bfloat16))
    r135_value = r135_value.to(tl.bfloat16)
    r135_value = tl.broadcast_to(r135_value, [$(131072)])
    tl.store($(candidate_vadd_17_ptr) + r135_z, r135_value)
    r136_z = tl.arange($(393216))
    r136_value = tl.where((r136_z % $(192)) < $(128), tl.load($(candidate_vadd_10_ptr) + ((((((((r136_z / $(196608)) % $(2)) * $(131072)) + (((r136_z / $(192)) % $(128)) * $(1024)) + (((r136_z / $(24576)) % $(8)) * $(128)) + ((r136_z % $(192)) - $(0))) / $(1024)) % $(256)) * $(1536)) + (((((((r136_z / $(196608)) % $(2)) * $(131072)) + (((r136_z / $(192)) % $(128)) * $(1024)) + (((r136_z / $(24576)) % $(8)) * $(128)) + ((r136_z % $(192)) - $(0))) / $(128)) % $(8)) * $(192)) + (((((r136_z / $(196608)) % $(2)) * $(131072)) + (((r136_z / $(192)) % $(128)) * $(1024)) + (((r136_z / $(24576)) % $(8)) * $(128)) + ((r136_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vadd_17_ptr) + ((((r136_z / $(196608)) % $(2)) * $(65536)) + (((r136_z / $(24576)) % $(8)) * $(8192)) + (((r136_z / $(192)) % $(128)) * $(64)) + (((r136_z % $(192)) - $(128)) % $(64))), dtype=tl.bfloat16))
    r136_value = r136_value.to(tl.bfloat16)
    r136_value = tl.broadcast_to(r136_value, [$(393216)])
    tl.store($(candidate_vcat_7_ptr) + r136_z, r136_value)
    r137_z = tl.arange($(65536))
    r137_value = tl.load($(candidate_vselect_backward_ptr) + ((((r137_z / $(32768)) % $(2)) * $(32768)) + (((r137_z / $(256)) % $(128)) * $(256)) + (r137_z % $(256))), dtype=tl.bfloat16)
    r137_value = r137_value.to(tl.float32)
    r137_value = tl.broadcast_to(r137_value, [$(65536)])
    tl.store($(candidate_vto_copy_36_ptr) + r137_z, r137_value)
    r138_z = tl.arange($(65536))
    r138_value = (tl.load($(candidate_vto_copy_36_ptr) + ((((r138_z / $(32768)) % $(2)) * $(32768)) + (((r138_z / $(256)) % $(128)) * $(256)) + (r138_z % $(256))), dtype=tl.float32) * tl.cast(tl.load($(candidate_vto_copy_4_ptr) + (r138_z % $(256)), dtype=tl.bfloat16), tl.float32))
    r138_value = r138_value.to(tl.float32)
    r138_value = tl.broadcast_to(r138_value, [$(65536)])
    tl.store($(candidate_vmul_32_ptr) + r138_z, r138_value)
    r139_z = tl.arange($(65536))
    r139_value = (tl.load($(candidate_vmul_32_ptr) + ((((r139_z / $(32768)) % $(2)) * $(32768)) + (((r139_z / $(256)) % $(128)) * $(256)) + (r139_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vto_copy_20_ptr) + ((((r139_z / $(32768)) % $(2)) * $(32768)) + (((r139_z / $(256)) % $(128)) * $(256)) + (r139_z % $(256))), dtype=tl.float32))
    r139_value = r139_value.to(tl.float32)
    r139_value = tl.broadcast_to(r139_value, [$(65536)])
    tl.store($(candidate_vmul_33_ptr) + r139_z, r139_value)
    r140_z = tl.arange($(65536))
    r140_value = (tl.load($(candidate_vmul_32_ptr) + ((((r140_z / $(32768)) % $(2)) * $(32768)) + (((r140_z / $(256)) % $(128)) * $(256)) + (r140_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_1_ptr) + ((((r140_z / $(32768)) % $(2)) * $(128)) + ((r140_z / $(256)) % $(128))), dtype=tl.float32))
    r140_value = r140_value.to(tl.float32)
    r140_value = tl.broadcast_to(r140_value, [$(65536)])
    tl.store($(candidate_vmul_34_ptr) + r140_z, r140_value)
    r141_z = tl.arange($(256))
    r141_lane = tl.arange($(256))
    r141_value = tl.sum(tl.load($(candidate_vmul_33_ptr) + ((((r141_z[:, None] / $(128)) % $(2)) * $(32768)) + ((r141_z[:, None] % $(128)) * $(256)) + r141_lane[None, :]), dtype=tl.float32), axis=1)
    r141_value = r141_value.to(tl.float32)
    r141_value = tl.broadcast_to(r141_value, [$(256)])
    tl.store($(candidate_vsum_2_ptr) + r141_z, r141_value)
    r142_z = tl.arange($(256))
    r142_value = ((tl.load($(candidate_vrsqrt_1_ptr) + ((((r142_z / $(128)) % $(2)) * $(128)) + (r142_z % $(128))), dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_1_ptr) + ((((r142_z / $(128)) % $(2)) * $(128)) + (r142_z % $(128))), dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_1_ptr) + ((((r142_z / $(128)) % $(2)) * $(128)) + (r142_z % $(128))), dtype=tl.float32)))
    r142_value = r142_value.to(tl.float32)
    r142_value = tl.broadcast_to(r142_value, [$(256)])
    tl.store($(candidate_vpow_3_ptr) + r142_z, r142_value)
    r143_z = tl.arange($(256))
    r143_value = (tl.load($(candidate_vsum_2_ptr) + ((((r143_z / $(128)) % $(2)) * $(128)) + (r143_z % $(128))), dtype=tl.float32) * -0.5)
    r143_value = r143_value.to(tl.float32)
    r143_value = tl.broadcast_to(r143_value, [$(256)])
    tl.store($(candidate_vmul_35_ptr) + r143_z, r143_value)
    r144_z = tl.arange($(256))
    r144_value = (tl.load($(candidate_vmul_35_ptr) + ((((r144_z / $(128)) % $(2)) * $(128)) + (r144_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vpow_3_ptr) + ((((r144_z / $(128)) % $(2)) * $(128)) + (r144_z % $(128))), dtype=tl.float32))
    r144_value = r144_value.to(tl.float32)
    r144_value = tl.broadcast_to(r144_value, [$(256)])
    tl.store($(candidate_vmul_36_ptr) + r144_z, r144_value)
    r145_z = tl.arange($(65536))
    r145_value = (tl.load($(candidate_vmul_36_ptr) + ((((r145_z / $(32768)) % $(2)) * $(128)) + ((r145_z / $(256)) % $(128))), dtype=tl.float32) / 256.0)
    r145_value = r145_value.to(tl.float32)
    r145_value = tl.broadcast_to(r145_value, [$(65536)])
    tl.store($(candidate_vdiv_ptr) + r145_z, r145_value)
    r146_z = tl.arange($(65536))
    r146_value = ((tl.load($(candidate_vto_copy_20_ptr) + ((((r146_z / $(32768)) % $(2)) * $(32768)) + (((r146_z / $(256)) % $(128)) * $(256)) + (r146_z % $(256))), dtype=tl.float32)))
    r146_value = r146_value.to(tl.float32)
    r146_value = tl.broadcast_to(r146_value, [$(65536)])
    tl.store($(candidate_vpow_4_ptr) + r146_z, r146_value)
    r147_z = tl.arange($(65536))
    r147_value = (tl.load($(candidate_vpow_4_ptr) + ((((r147_z / $(32768)) % $(2)) * $(32768)) + (((r147_z / $(256)) % $(128)) * $(256)) + (r147_z % $(256))), dtype=tl.float32) * 2.0)
    r147_value = r147_value.to(tl.float32)
    r147_value = tl.broadcast_to(r147_value, [$(65536)])
    tl.store($(candidate_vmul_37_ptr) + r147_z, r147_value)
    r148_z = tl.arange($(65536))
    r148_value = (tl.load($(candidate_vdiv_ptr) + ((((r148_z / $(32768)) % $(2)) * $(32768)) + (((r148_z / $(256)) % $(128)) * $(256)) + (r148_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vmul_37_ptr) + ((((r148_z / $(32768)) % $(2)) * $(32768)) + (((r148_z / $(256)) % $(128)) * $(256)) + (r148_z % $(256))), dtype=tl.float32))
    r148_value = r148_value.to(tl.float32)
    r148_value = tl.broadcast_to(r148_value, [$(65536)])
    tl.store($(candidate_vmul_38_ptr) + r148_z, r148_value)
    r149_z = tl.arange($(65536))
    r149_value = (tl.load($(candidate_vmul_34_ptr) + ((((r149_z / $(32768)) % $(2)) * $(32768)) + (((r149_z / $(256)) % $(128)) * $(256)) + (r149_z % $(256))), dtype=tl.float32) + tl.load($(candidate_vmul_38_ptr) + ((((r149_z / $(32768)) % $(2)) * $(32768)) + (((r149_z / $(256)) % $(128)) * $(256)) + (r149_z % $(256))), dtype=tl.float32))
    r149_value = r149_value.to(tl.float32)
    r149_value = tl.broadcast_to(r149_value, [$(65536)])
    tl.store($(candidate_vadd_18_ptr) + r149_z, r149_value)
    r150_z = tl.arange($(65536))
    r150_value = tl.load($(candidate_vadd_18_ptr) + ((((r150_z / $(32768)) % $(2)) * $(32768)) + (((r150_z / $(256)) % $(128)) * $(256)) + (r150_z % $(256))), dtype=tl.float32)
    r150_value = r150_value.to(tl.bfloat16)
    r150_value = tl.broadcast_to(r150_value, [$(65536)])
    tl.store($(candidate_vto_copy_37_ptr) + r150_z, r150_value)
    r151_z = tl.arange($(81920))
    r151_value = tl.where((r151_z % $(320)) < $(256), tl.load($(candidate_vto_copy_37_ptr) + ((((r151_z / $(40960)) % $(2)) * $(32768)) + (((r151_z / $(320)) % $(128)) * $(256)) + (((r151_z % $(320)) - $(0)) % $(256))), dtype=tl.bfloat16), tl.load($(candidate_vadd_16_ptr) + ((((((((r151_z / $(40960)) % $(2)) * $(8192)) + (((r151_z / $(320)) % $(128)) * $(64)) + ((r151_z % $(320)) - $(256))) / $(8192)) % $(2)) * $(8192)) + (((((((r151_z / $(40960)) % $(2)) * $(8192)) + (((r151_z / $(320)) % $(128)) * $(64)) + ((r151_z % $(320)) - $(256))) / $(64)) % $(128)) * $(64)) + (((((r151_z / $(40960)) % $(2)) * $(8192)) + (((r151_z / $(320)) % $(128)) * $(64)) + ((r151_z % $(320)) - $(256))) % $(64))), dtype=tl.bfloat16))
    r151_value = r151_value.to(tl.bfloat16)
    r151_value = tl.broadcast_to(r151_value, [$(81920)])
    tl.store($(candidate_vcat_8_ptr) + r151_z, r151_value)
    r152_row = tl.arange($(256))
    r152_col = tl.arange($(512))
    r152_lane = tl.arange($(320))
    r152_left = tl.load($(candidate_vcat_8_ptr) + ((((((r152_row[:, None] * $(320)) + r152_lane[None, :]) / $(40960)) % $(2)) * $(40960)) + (((((r152_row[:, None] * $(320)) + r152_lane[None, :]) / $(320)) % $(128)) * $(320)) + (((r152_row[:, None] * $(320)) + r152_lane[None, :]) % $(320))), dtype=tl.bfloat16)
    r152_right = tl.load($(candidate_vto_copy_3_ptr) + (((r152_lane[:, None] % $(320)) * $(512)) + (r152_col[None, :] % $(512))), dtype=tl.bfloat16)
    r152_value = tl.dot(r152_left, r152_right)
    r152_value = r152_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_5_ptr) + ((r152_row[:, None] * $(512)) + r152_col[None, :]), r152_value)
    r153_row = tl.arange($(256))
    r153_col = tl.arange($(256))
    r153_lane = tl.arange($(1536))
    r153_left = tl.load($(candidate_vcat_7_ptr) + ((((((r153_row[:, None] * $(1536)) + r153_lane[None, :]) / $(196608)) % $(2)) * $(196608)) + (((((((((r153_row[:, None] * $(1536)) + r153_lane[None, :]) / $(196608)) % $(2)) * $(196608)) + (((((r153_row[:, None] * $(1536)) + r153_lane[None, :]) / $(1536)) % $(128)) * $(1536)) + (((r153_row[:, None] * $(1536)) + r153_lane[None, :]) % $(1536))) / $(192)) % $(8)) * $(24576)) + (((((r153_row[:, None] * $(1536)) + r153_lane[None, :]) / $(1536)) % $(128)) * $(192)) + (((((((r153_row[:, None] * $(1536)) + r153_lane[None, :]) / $(196608)) % $(2)) * $(196608)) + (((((r153_row[:, None] * $(1536)) + r153_lane[None, :]) / $(1536)) % $(128)) * $(1536)) + (((r153_row[:, None] * $(1536)) + r153_lane[None, :]) % $(1536))) % $(192))), dtype=tl.bfloat16)
    r153_right = tl.load($(candidate_vto_copy_2_ptr) + (((r153_lane[:, None] % $(1536)) * $(256)) + (r153_col[None, :] % $(256))), dtype=tl.bfloat16)
    r153_value = tl.dot(r153_left, r153_right)
    r153_value = r153_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_6_ptr) + ((r153_row[:, None] * $(256)) + r153_col[None, :]), r153_value)
    r154_z = tl.arange($(65536))
    r154_value = tl.load($(candidate_vmm_6_ptr) + ((((r154_z / $(32768)) % $(2)) * $(32768)) + (((r154_z / $(256)) % $(128)) * $(256)) + (r154_z % $(256))), dtype=tl.bfloat16)
    r154_value = r154_value.to(tl.float32)
    r154_value = tl.broadcast_to(r154_value, [$(65536)])
    tl.store($(candidate_vto_copy_38_ptr) + r154_z, r154_value)
    r155_z = tl.arange($(65536))
    r155_value = (tl.load($(candidate_vto_copy_38_ptr) + ((((r155_z / $(32768)) % $(2)) * $(32768)) + (((r155_z / $(256)) % $(128)) * $(256)) + (r155_z % $(256))), dtype=tl.float32) * tl.cast(tl.load($(candidate_vto_copy_1_ptr) + (r155_z % $(256)), dtype=tl.bfloat16), tl.float32))
    r155_value = r155_value.to(tl.float32)
    r155_value = tl.broadcast_to(r155_value, [$(65536)])
    tl.store($(candidate_vmul_39_ptr) + r155_z, r155_value)
    r156_z = tl.arange($(65536))
    r156_value = (tl.load($(candidate_vmul_39_ptr) + ((((r156_z / $(32768)) % $(2)) * $(32768)) + (((r156_z / $(256)) % $(128)) * $(256)) + (r156_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vto_copy_18_ptr) + ((((r156_z / $(32768)) % $(2)) * $(32768)) + (((r156_z / $(256)) % $(128)) * $(256)) + (r156_z % $(256))), dtype=tl.float32))
    r156_value = r156_value.to(tl.float32)
    r156_value = tl.broadcast_to(r156_value, [$(65536)])
    tl.store($(candidate_vmul_40_ptr) + r156_z, r156_value)
    r157_z = tl.arange($(65536))
    r157_value = (tl.load($(candidate_vmul_39_ptr) + ((((r157_z / $(32768)) % $(2)) * $(32768)) + (((r157_z / $(256)) % $(128)) * $(256)) + (r157_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_ptr) + ((((r157_z / $(32768)) % $(2)) * $(128)) + ((r157_z / $(256)) % $(128))), dtype=tl.float32))
    r157_value = r157_value.to(tl.float32)
    r157_value = tl.broadcast_to(r157_value, [$(65536)])
    tl.store($(candidate_vmul_41_ptr) + r157_z, r157_value)
    r158_z = tl.arange($(256))
    r158_lane = tl.arange($(256))
    r158_value = tl.sum(tl.load($(candidate_vmul_40_ptr) + ((((r158_z[:, None] / $(128)) % $(2)) * $(32768)) + ((r158_z[:, None] % $(128)) * $(256)) + r158_lane[None, :]), dtype=tl.float32), axis=1)
    r158_value = r158_value.to(tl.float32)
    r158_value = tl.broadcast_to(r158_value, [$(256)])
    tl.store($(candidate_vsum_3_ptr) + r158_z, r158_value)
    r159_z = tl.arange($(256))
    r159_value = ((tl.load($(candidate_vrsqrt_ptr) + ((((r159_z / $(128)) % $(2)) * $(128)) + (r159_z % $(128))), dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_ptr) + ((((r159_z / $(128)) % $(2)) * $(128)) + (r159_z % $(128))), dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_ptr) + ((((r159_z / $(128)) % $(2)) * $(128)) + (r159_z % $(128))), dtype=tl.float32)))
    r159_value = r159_value.to(tl.float32)
    r159_value = tl.broadcast_to(r159_value, [$(256)])
    tl.store($(candidate_vpow_5_ptr) + r159_z, r159_value)
    r160_z = tl.arange($(256))
    r160_value = (tl.load($(candidate_vsum_3_ptr) + ((((r160_z / $(128)) % $(2)) * $(128)) + (r160_z % $(128))), dtype=tl.float32) * -0.5)
    r160_value = r160_value.to(tl.float32)
    r160_value = tl.broadcast_to(r160_value, [$(256)])
    tl.store($(candidate_vmul_42_ptr) + r160_z, r160_value)
    r161_z = tl.arange($(256))
    r161_value = (tl.load($(candidate_vmul_42_ptr) + ((((r161_z / $(128)) % $(2)) * $(128)) + (r161_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vpow_5_ptr) + ((((r161_z / $(128)) % $(2)) * $(128)) + (r161_z % $(128))), dtype=tl.float32))
    r161_value = r161_value.to(tl.float32)
    r161_value = tl.broadcast_to(r161_value, [$(256)])
    tl.store($(candidate_vmul_43_ptr) + r161_z, r161_value)
    r162_z = tl.arange($(65536))
    r162_value = (tl.load($(candidate_vmul_43_ptr) + ((((r162_z / $(32768)) % $(2)) * $(128)) + ((r162_z / $(256)) % $(128))), dtype=tl.float32) / 256.0)
    r162_value = r162_value.to(tl.float32)
    r162_value = tl.broadcast_to(r162_value, [$(65536)])
    tl.store($(candidate_vdiv_1_ptr) + r162_z, r162_value)
    r163_z = tl.arange($(65536))
    r163_value = ((tl.load($(candidate_vto_copy_18_ptr) + ((((r163_z / $(32768)) % $(2)) * $(32768)) + (((r163_z / $(256)) % $(128)) * $(256)) + (r163_z % $(256))), dtype=tl.float32)))
    r163_value = r163_value.to(tl.float32)
    r163_value = tl.broadcast_to(r163_value, [$(65536)])
    tl.store($(candidate_vpow_6_ptr) + r163_z, r163_value)
    r164_z = tl.arange($(65536))
    r164_value = (tl.load($(candidate_vpow_6_ptr) + ((((r164_z / $(32768)) % $(2)) * $(32768)) + (((r164_z / $(256)) % $(128)) * $(256)) + (r164_z % $(256))), dtype=tl.float32) * 2.0)
    r164_value = r164_value.to(tl.float32)
    r164_value = tl.broadcast_to(r164_value, [$(65536)])
    tl.store($(candidate_vmul_44_ptr) + r164_z, r164_value)
    r165_z = tl.arange($(65536))
    r165_value = (tl.load($(candidate_vdiv_1_ptr) + ((((r165_z / $(32768)) % $(2)) * $(32768)) + (((r165_z / $(256)) % $(128)) * $(256)) + (r165_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vmul_44_ptr) + ((((r165_z / $(32768)) % $(2)) * $(32768)) + (((r165_z / $(256)) % $(128)) * $(256)) + (r165_z % $(256))), dtype=tl.float32))
    r165_value = r165_value.to(tl.float32)
    r165_value = tl.broadcast_to(r165_value, [$(65536)])
    tl.store($(candidate_vmul_45_ptr) + r165_z, r165_value)
    r166_z = tl.arange($(65536))
    r166_value = (tl.load($(candidate_vmul_41_ptr) + ((((r166_z / $(32768)) % $(2)) * $(32768)) + (((r166_z / $(256)) % $(128)) * $(256)) + (r166_z % $(256))), dtype=tl.float32) + tl.load($(candidate_vmul_45_ptr) + ((((r166_z / $(32768)) % $(2)) * $(32768)) + (((r166_z / $(256)) % $(128)) * $(256)) + (r166_z % $(256))), dtype=tl.float32))
    r166_value = r166_value.to(tl.float32)
    r166_value = tl.broadcast_to(r166_value, [$(65536)])
    tl.store($(candidate_vadd_19_ptr) + r166_z, r166_value)
    r167_z = tl.arange($(65536))
    r167_value = tl.load($(candidate_vadd_19_ptr) + ((((r167_z / $(32768)) % $(2)) * $(32768)) + (((r167_z / $(256)) % $(128)) * $(256)) + (r167_z % $(256))), dtype=tl.float32)
    r167_value = r167_value.to(tl.bfloat16)
    r167_value = tl.broadcast_to(r167_value, [$(65536)])
    tl.store($(candidate_vto_copy_39_ptr) + r167_z, r167_value)
    r168_row = tl.arange($(256))
    r168_col = tl.arange($(512))
    r168_lane = tl.arange($(256))
    r168_left = tl.load($(candidate_vto_copy_39_ptr) + ((((((r168_row[:, None] * $(256)) + r168_lane[None, :]) / $(32768)) % $(2)) * $(32768)) + (((((r168_row[:, None] * $(256)) + r168_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r168_row[:, None] * $(256)) + r168_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r168_right = tl.load($(candidate_vto_copy_ptr) + (((r168_lane[:, None] % $(256)) * $(512)) + (r168_col[None, :] % $(512))), dtype=tl.bfloat16)
    r168_value = tl.dot(r168_left, r168_right)
    r168_value = r168_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_7_ptr) + ((r168_row[:, None] * $(512)) + r168_col[None, :]), r168_value)
    r169_z = tl.arange($(131072))
    r169_value = (tl.load($(candidate_vmm_5_ptr) + ((((r169_z / $(65536)) % $(2)) * $(65536)) + (((r169_z / $(512)) % $(128)) * $(512)) + (r169_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vmm_7_ptr) + ((((r169_z / $(65536)) % $(2)) * $(65536)) + (((r169_z / $(512)) % $(128)) * $(512)) + (r169_z % $(512))), dtype=tl.bfloat16))
    r169_value = r169_value.to(tl.bfloat16)
    r169_value = tl.broadcast_to(r169_value, [$(131072)])
    tl.store($(candidate_vadd_20_ptr) + r169_z, r169_value)
    z = tl.arange($(131072))
    value = tl.load($(candidate_vadd_20_ptr) + ((((z / $(65536)) % $(2)) * $(65536)) + (((z / $(512)) % $(128)) * $(512)) + (z % $(512))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(131072)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_mla_fa4_nonabsorb_sink_dx
