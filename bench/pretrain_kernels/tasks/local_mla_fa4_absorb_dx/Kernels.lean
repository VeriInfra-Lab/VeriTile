import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_mla_fa4_absorb_dx
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
  let vtriu_ptr : RegionName := "vtriu"
  let vmasked_fill_ptr : RegionName := "vmasked_fill"
  let vmm_ptr : RegionName := "vmm"
  let vto_copy_16_ptr : RegionName := "vto_copy_16"
  let vpow_1_ptr : RegionName := "vpow_1"
  let vmean_ptr : RegionName := "vmean"
  let vadd_2_ptr : RegionName := "vadd_2"
  let vrsqrt_ptr : RegionName := "vrsqrt"
  let vmul_7_ptr : RegionName := "vmul_7"
  let vmul_8_ptr : RegionName := "vmul_8"
  let vto_copy_17_ptr : RegionName := "vto_copy_17"
  let vmm_1_ptr : RegionName := "vmm_1"
  let vmm_2_ptr : RegionName := "vmm_2"
  let vto_copy_18_ptr : RegionName := "vto_copy_18"
  let vpow_2_ptr : RegionName := "vpow_2"
  let vmean_1_ptr : RegionName := "vmean_1"
  let vadd_3_ptr : RegionName := "vadd_3"
  let vrsqrt_1_ptr : RegionName := "vrsqrt_1"
  let vmul_9_ptr : RegionName := "vmul_9"
  let vmul_10_ptr : RegionName := "vmul_10"
  let vto_copy_19_ptr : RegionName := "vto_copy_19"
  let vmm_3_ptr : RegionName := "vmm_3"
  let vmul_11_ptr : RegionName := "vmul_11"
  let vmul_12_ptr : RegionName := "vmul_12"
  let vsub_ptr : RegionName := "vsub"
  let vmul_13_ptr : RegionName := "vmul_13"
  let vmul_14_ptr : RegionName := "vmul_14"
  let vadd_4_ptr : RegionName := "vadd_4"
  let vcat_ptr : RegionName := "vcat"
  let vmul_15_ptr : RegionName := "vmul_15"
  let vmul_16_ptr : RegionName := "vmul_16"
  let vsub_1_ptr : RegionName := "vsub_1"
  let vmul_17_ptr : RegionName := "vmul_17"
  let vmul_18_ptr : RegionName := "vmul_18"
  let vadd_5_ptr : RegionName := "vadd_5"
  let vcat_1_ptr : RegionName := "vcat_1"
  let vcat_2_ptr : RegionName := "vcat_2"
  let vcat_3_ptr : RegionName := "vcat_3"
  let vbmm_ptr : RegionName := "vbmm"
  let vmul_19_ptr : RegionName := "vmul_19"
  let vadd_6_ptr : RegionName := "vadd_6"
  let vto_copy_20_ptr : RegionName := "vto_copy_20"
  let vsoftmax_ptr : RegionName := "vsoftmax"
  let vto_copy_21_ptr : RegionName := "vto_copy_21"
  let vmm_5_ptr : RegionName := "vmm_5"
  let vbmm_2_ptr : RegionName := "vbmm_2"
  let vbmm_3_ptr : RegionName := "vbmm_3"
  let vto_copy_22_ptr : RegionName := "vto_copy_22"
  let vsoftmax_backward_data_ptr : RegionName := "vsoftmax_backward_data"
  let vto_copy_23_ptr : RegionName := "vto_copy_23"
  let vmul_20_ptr : RegionName := "vmul_20"
  let vbmm_4_ptr : RegionName := "vbmm_4"
  let vbmm_5_ptr : RegionName := "vbmm_5"
  let vsum_1_ptr : RegionName := "vsum_1"
  let vmul_21_ptr : RegionName := "vmul_21"
  let vmul_22_ptr : RegionName := "vmul_22"
  let vneg_ptr : RegionName := "vneg"
  let vmul_23_ptr : RegionName := "vmul_23"
  let vadd_7_ptr : RegionName := "vadd_7"
  let vmul_24_ptr : RegionName := "vmul_24"
  let vadd_8_ptr : RegionName := "vadd_8"
  let vmul_25_ptr : RegionName := "vmul_25"
  let vmul_26_ptr : RegionName := "vmul_26"
  let vneg_1_ptr : RegionName := "vneg_1"
  let vmul_27_ptr : RegionName := "vmul_27"
  let vadd_9_ptr : RegionName := "vadd_9"
  let vmul_28_ptr : RegionName := "vmul_28"
  let vadd_10_ptr : RegionName := "vadd_10"
  let vslice_backward_ptr : RegionName := "vslice_backward"
  let vslice_backward_1_ptr : RegionName := "vslice_backward_1"
  let vadd_11_ptr : RegionName := "vadd_11"
  let vslice_backward_2_ptr : RegionName := "vslice_backward_2"
  let vslice_backward_3_ptr : RegionName := "vslice_backward_3"
  let vadd_12_ptr : RegionName := "vadd_12"
  let vcat_4_ptr : RegionName := "vcat_4"
  let vcat_5_ptr : RegionName := "vcat_5"
  let vmm_6_ptr : RegionName := "vmm_6"
  let vto_copy_24_ptr : RegionName := "vto_copy_24"
  let vmul_29_ptr : RegionName := "vmul_29"
  let vmul_30_ptr : RegionName := "vmul_30"
  let vmul_31_ptr : RegionName := "vmul_31"
  let vsum_2_ptr : RegionName := "vsum_2"
  let vpow_3_ptr : RegionName := "vpow_3"
  let vmul_32_ptr : RegionName := "vmul_32"
  let vmul_33_ptr : RegionName := "vmul_33"
  let vdiv_ptr : RegionName := "vdiv"
  let vpow_4_ptr : RegionName := "vpow_4"
  let vmul_34_ptr : RegionName := "vmul_34"
  let vmul_35_ptr : RegionName := "vmul_35"
  let vadd_13_ptr : RegionName := "vadd_13"
  let vto_copy_25_ptr : RegionName := "vto_copy_25"
  let vcat_6_ptr : RegionName := "vcat_6"
  let vmm_7_ptr : RegionName := "vmm_7"
  let vmm_8_ptr : RegionName := "vmm_8"
  let vto_copy_26_ptr : RegionName := "vto_copy_26"
  let vmul_36_ptr : RegionName := "vmul_36"
  let vmul_37_ptr : RegionName := "vmul_37"
  let vmul_38_ptr : RegionName := "vmul_38"
  let vsum_3_ptr : RegionName := "vsum_3"
  let vpow_5_ptr : RegionName := "vpow_5"
  let vmul_39_ptr : RegionName := "vmul_39"
  let vmul_40_ptr : RegionName := "vmul_40"
  let vdiv_1_ptr : RegionName := "vdiv_1"
  let vpow_6_ptr : RegionName := "vpow_6"
  let vmul_41_ptr : RegionName := "vmul_41"
  let vmul_42_ptr : RegionName := "vmul_42"
  let vadd_14_ptr : RegionName := "vadd_14"
  let vto_copy_27_ptr : RegionName := "vto_copy_27"
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
    r4_z = tl.arange($(294912))
    r4_value = (tl.load($(parameter3_ptr) + ((((r4_z / $(512)) % $(576)) * $(512)) + (r4_z % $(512))), dtype=tl.float32) * 0.04419417382415922)
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(294912)])
    tl.store($(vmul_3_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(512))
    r5_value = (tl.load($(parameter4_ptr) + r5_z, dtype=tl.float32) * 0.1)
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(512)])
    tl.store($(vmul_4_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(512))
    r6_value = (tl.load($(vmul_4_ptr) + r6_z, dtype=tl.float32) + 1.0)
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(512)])
    tl.store($(vadd_1_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(1048576))
    r7_value = (tl.load($(parameter5_ptr) + ((((r7_z / $(512)) % $(2048)) * $(512)) + (r7_z % $(512))), dtype=tl.float32) * 0.04419417382415922)
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(1048576)])
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
    r12_z = tl.arange($(294912))
    r12_value = tl.load($(vmul_3_ptr) + ((((r12_z / $(512)) % $(576)) * $(512)) + (r12_z % $(512))), dtype=tl.float32)
    r12_value = r12_value.to(tl.bfloat16)
    r12_value = tl.broadcast_to(r12_value, [$(294912)])
    tl.store($(vto_copy_3_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(512))
    r13_value = tl.load($(vadd_1_ptr) + r13_z, dtype=tl.float32)
    r13_value = r13_value.to(tl.bfloat16)
    r13_value = tl.broadcast_to(r13_value, [$(512)])
    tl.store($(vto_copy_4_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(1048576))
    r14_value = tl.load($(vmul_5_ptr) + ((((r14_z / $(512)) % $(2048)) * $(512)) + (r14_z % $(512))), dtype=tl.float32)
    r14_value = r14_value.to(tl.bfloat16)
    r14_value = tl.broadcast_to(r14_value, [$(1048576)])
    tl.store($(vto_copy_5_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(524288))
    r15_value = tl.load($(vmul_6_ptr) + ((((r15_z / $(1024)) % $(512)) * $(1024)) + (r15_z % $(1024))), dtype=tl.float32)
    r15_value = r15_value.to(tl.bfloat16)
    r15_value = tl.broadcast_to(r15_value, [$(524288)])
    tl.store($(vto_copy_6_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(16384))
    r16_value = tl.where((r16_z % $(128)) >= ((r16_z / $(128)) % $(128)) + $(1), true, false)
    r16_value = tl.broadcast_to(r16_value, [$(16384)])
    tl.store($(vtriu_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(32768))
    r17_value = tl.where(tl.load($(vtriu_ptr) + ((((r17_z / $(128)) % $(128)) * $(128)) + (r17_z % $(128))), dtype=tl.int1), tl.cast(-338953138925153547590470800371487866880.0, tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r17_value = r17_value.to(tl.bfloat16)
    r17_value = tl.broadcast_to(r17_value, [$(32768)])
    tl.store($(vmasked_fill_ptr) + r17_z, r17_value)
    r18_row = tl.arange($(256))
    r18_col = tl.arange($(256))
    r18_lane = tl.arange($(512))
    r18_left = tl.load($(input0_ptr) + ((((((r18_row[:, None] * $(512)) + r18_lane[None, :]) / $(65536)) % $(2)) * $(65536)) + (((((r18_row[:, None] * $(512)) + r18_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r18_row[:, None] * $(512)) + r18_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r18_right = tl.load($(vto_copy_ptr) + (((r18_col[None, :] % $(256)) * $(512)) + (r18_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r18_value = tl.dot(r18_left, r18_right)
    r18_value = r18_value.to(tl.bfloat16)
    tl.store($(vmm_ptr) + ((r18_row[:, None] * $(256)) + r18_col[None, :]), r18_value)
    r19_z = tl.arange($(65536))
    r19_value = tl.load($(vmm_ptr) + ((((r19_z / $(32768)) % $(2)) * $(32768)) + (((r19_z / $(256)) % $(128)) * $(256)) + (r19_z % $(256))), dtype=tl.bfloat16)
    r19_value = r19_value.to(tl.float32)
    r19_value = tl.broadcast_to(r19_value, [$(65536)])
    tl.store($(vto_copy_16_ptr) + r19_z, r19_value)
    r20_z = tl.arange($(65536))
    r20_value = ((tl.load($(vto_copy_16_ptr) + ((((r20_z / $(32768)) % $(2)) * $(32768)) + (((r20_z / $(256)) % $(128)) * $(256)) + (r20_z % $(256))), dtype=tl.float32)) * (tl.load($(vto_copy_16_ptr) + ((((r20_z / $(32768)) % $(2)) * $(32768)) + (((r20_z / $(256)) % $(128)) * $(256)) + (r20_z % $(256))), dtype=tl.float32)))
    r20_value = r20_value.to(tl.float32)
    r20_value = tl.broadcast_to(r20_value, [$(65536)])
    tl.store($(vpow_1_ptr) + r20_z, r20_value)
    r21_z = tl.arange($(256))
    r21_lane = tl.arange($(256))
    r21_value = (tl.sum(tl.load($(vpow_1_ptr) + ((((r21_z[:, None] / $(128)) % $(2)) * $(32768)) + ((r21_z[:, None] % $(128)) * $(256)) + r21_lane[None, :]), dtype=tl.float32), axis=1) / 256.0)
    r21_value = r21_value.to(tl.float32)
    r21_value = tl.broadcast_to(r21_value, [$(256)])
    tl.store($(vmean_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(256))
    r22_value = (tl.load($(vmean_ptr) + ((((r22_z / $(128)) % $(2)) * $(128)) + (r22_z % $(128))), dtype=tl.float32) + 0.000001)
    r22_value = r22_value.to(tl.float32)
    r22_value = tl.broadcast_to(r22_value, [$(256)])
    tl.store($(vadd_2_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(256))
    r23_value = tl.rsqrt(tl.load($(vadd_2_ptr) + ((((r23_z / $(128)) % $(2)) * $(128)) + (r23_z % $(128))), dtype=tl.float32))
    r23_value = r23_value.to(tl.float32)
    r23_value = tl.broadcast_to(r23_value, [$(256)])
    tl.store($(vrsqrt_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(65536))
    r24_value = (tl.load($(vto_copy_16_ptr) + ((((r24_z / $(32768)) % $(2)) * $(32768)) + (((r24_z / $(256)) % $(128)) * $(256)) + (r24_z % $(256))), dtype=tl.float32) * tl.load($(vrsqrt_ptr) + ((((r24_z / $(32768)) % $(2)) * $(128)) + ((r24_z / $(256)) % $(128))), dtype=tl.float32))
    r24_value = r24_value.to(tl.float32)
    r24_value = tl.broadcast_to(r24_value, [$(65536)])
    tl.store($(vmul_7_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(65536))
    r25_value = (tl.cast(tl.load($(vto_copy_1_ptr) + (r25_z % $(256)), dtype=tl.bfloat16), tl.float32) * tl.load($(vmul_7_ptr) + ((((r25_z / $(32768)) % $(2)) * $(32768)) + (((r25_z / $(256)) % $(128)) * $(256)) + (r25_z % $(256))), dtype=tl.float32))
    r25_value = r25_value.to(tl.float32)
    r25_value = tl.broadcast_to(r25_value, [$(65536)])
    tl.store($(vmul_8_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(65536))
    r26_value = tl.load($(vmul_8_ptr) + ((((r26_z / $(32768)) % $(2)) * $(32768)) + (((r26_z / $(256)) % $(128)) * $(256)) + (r26_z % $(256))), dtype=tl.float32)
    r26_value = r26_value.to(tl.bfloat16)
    r26_value = tl.broadcast_to(r26_value, [$(65536)])
    tl.store($(vto_copy_17_ptr) + r26_z, r26_value)
    r27_row = tl.arange($(256))
    r27_col = tl.arange($(1536))
    r27_lane = tl.arange($(256))
    r27_left = tl.load($(vto_copy_17_ptr) + ((((((r27_row[:, None] * $(256)) + r27_lane[None, :]) / $(32768)) % $(2)) * $(32768)) + (((((r27_row[:, None] * $(256)) + r27_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r27_row[:, None] * $(256)) + r27_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r27_right = tl.load($(vto_copy_2_ptr) + (((r27_col[None, :] % $(1536)) * $(256)) + (r27_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r27_value = tl.dot(r27_left, r27_right)
    r27_value = r27_value.to(tl.bfloat16)
    tl.store($(vmm_1_ptr) + ((r27_row[:, None] * $(1536)) + r27_col[None, :]), r27_value)
    r28_row = tl.arange($(256))
    r28_col = tl.arange($(576))
    r28_lane = tl.arange($(512))
    r28_left = tl.load($(input0_ptr) + ((((((r28_row[:, None] * $(512)) + r28_lane[None, :]) / $(65536)) % $(2)) * $(65536)) + (((((r28_row[:, None] * $(512)) + r28_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r28_row[:, None] * $(512)) + r28_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r28_right = tl.load($(vto_copy_3_ptr) + (((r28_col[None, :] % $(576)) * $(512)) + (r28_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r28_value = tl.dot(r28_left, r28_right)
    r28_value = r28_value.to(tl.bfloat16)
    tl.store($(vmm_2_ptr) + ((r28_row[:, None] * $(576)) + r28_col[None, :]), r28_value)
    r29_z = tl.arange($(131072))
    r29_value = tl.load($(vmm_2_ptr) + ((((r29_z / $(65536)) % $(2)) * $(73728)) + (((r29_z / $(512)) % $(128)) * $(576)) + (r29_z % $(512))), dtype=tl.bfloat16)
    r29_value = r29_value.to(tl.float32)
    r29_value = tl.broadcast_to(r29_value, [$(131072)])
    tl.store($(vto_copy_18_ptr) + r29_z, r29_value)
    r30_z = tl.arange($(131072))
    r30_value = ((tl.load($(vto_copy_18_ptr) + ((((r30_z / $(65536)) % $(2)) * $(65536)) + (((r30_z / $(512)) % $(128)) * $(512)) + (r30_z % $(512))), dtype=tl.float32)) * (tl.load($(vto_copy_18_ptr) + ((((r30_z / $(65536)) % $(2)) * $(65536)) + (((r30_z / $(512)) % $(128)) * $(512)) + (r30_z % $(512))), dtype=tl.float32)))
    r30_value = r30_value.to(tl.float32)
    r30_value = tl.broadcast_to(r30_value, [$(131072)])
    tl.store($(vpow_2_ptr) + r30_z, r30_value)
    r31_z = tl.arange($(256))
    r31_lane = tl.arange($(512))
    r31_value = (tl.sum(tl.load($(vpow_2_ptr) + ((((r31_z[:, None] / $(128)) % $(2)) * $(65536)) + ((r31_z[:, None] % $(128)) * $(512)) + r31_lane[None, :]), dtype=tl.float32), axis=1) / 512.0)
    r31_value = r31_value.to(tl.float32)
    r31_value = tl.broadcast_to(r31_value, [$(256)])
    tl.store($(vmean_1_ptr) + r31_z, r31_value)
    r32_z = tl.arange($(256))
    r32_value = (tl.load($(vmean_1_ptr) + ((((r32_z / $(128)) % $(2)) * $(128)) + (r32_z % $(128))), dtype=tl.float32) + 0.000001)
    r32_value = r32_value.to(tl.float32)
    r32_value = tl.broadcast_to(r32_value, [$(256)])
    tl.store($(vadd_3_ptr) + r32_z, r32_value)
    r33_z = tl.arange($(256))
    r33_value = tl.rsqrt(tl.load($(vadd_3_ptr) + ((((r33_z / $(128)) % $(2)) * $(128)) + (r33_z % $(128))), dtype=tl.float32))
    r33_value = r33_value.to(tl.float32)
    r33_value = tl.broadcast_to(r33_value, [$(256)])
    tl.store($(vrsqrt_1_ptr) + r33_z, r33_value)
    r34_z = tl.arange($(131072))
    r34_value = (tl.load($(vto_copy_18_ptr) + ((((r34_z / $(65536)) % $(2)) * $(65536)) + (((r34_z / $(512)) % $(128)) * $(512)) + (r34_z % $(512))), dtype=tl.float32) * tl.load($(vrsqrt_1_ptr) + ((((r34_z / $(65536)) % $(2)) * $(128)) + ((r34_z / $(512)) % $(128))), dtype=tl.float32))
    r34_value = r34_value.to(tl.float32)
    r34_value = tl.broadcast_to(r34_value, [$(131072)])
    tl.store($(vmul_9_ptr) + r34_z, r34_value)
    r35_z = tl.arange($(131072))
    r35_value = (tl.cast(tl.load($(vto_copy_4_ptr) + (r35_z % $(512)), dtype=tl.bfloat16), tl.float32) * tl.load($(vmul_9_ptr) + ((((r35_z / $(65536)) % $(2)) * $(65536)) + (((r35_z / $(512)) % $(128)) * $(512)) + (r35_z % $(512))), dtype=tl.float32))
    r35_value = r35_value.to(tl.float32)
    r35_value = tl.broadcast_to(r35_value, [$(131072)])
    tl.store($(vmul_10_ptr) + r35_z, r35_value)
    r36_z = tl.arange($(131072))
    r36_value = tl.load($(vmul_10_ptr) + ((((r36_z / $(65536)) % $(2)) * $(65536)) + (((r36_z / $(512)) % $(128)) * $(512)) + (r36_z % $(512))), dtype=tl.float32)
    r36_value = r36_value.to(tl.bfloat16)
    r36_value = tl.broadcast_to(r36_value, [$(131072)])
    tl.store($(vto_copy_19_ptr) + r36_z, r36_value)
    r37_row = tl.arange($(256))
    r37_col = tl.arange($(2048))
    r37_lane = tl.arange($(512))
    r37_left = tl.load($(vto_copy_19_ptr) + ((((((r37_row[:, None] * $(512)) + r37_lane[None, :]) / $(65536)) % $(2)) * $(65536)) + (((((r37_row[:, None] * $(512)) + r37_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r37_row[:, None] * $(512)) + r37_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r37_right = tl.load($(vto_copy_5_ptr) + (((r37_col[None, :] % $(2048)) * $(512)) + (r37_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r37_value = tl.dot(r37_left, r37_right)
    r37_value = r37_value.to(tl.bfloat16)
    tl.store($(vmm_3_ptr) + ((r37_row[:, None] * $(2048)) + r37_col[None, :]), r37_value)
    r38_z = tl.arange($(65536))
    r38_value = (tl.load($(vmm_1_ptr) + ((((r38_z / $(32768)) % $(2)) * $(196608)) + (((r38_z / $(32)) % $(128)) * $(1536)) + (((r38_z / $(4096)) % $(8)) * $(192)) + ((r38_z % $(32)) * $(2)) + $(128)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r38_value = r38_value.to(tl.bfloat16)
    r38_value = tl.broadcast_to(r38_value, [$(65536)])
    tl.store($(vmul_11_ptr) + r38_z, r38_value)
    r39_z = tl.arange($(65536))
    r39_value = (tl.load($(vmm_1_ptr) + ((((r39_z / $(32768)) % $(2)) * $(196608)) + (((r39_z / $(32)) % $(128)) * $(1536)) + (((r39_z / $(4096)) % $(8)) * $(192)) + ((r39_z % $(32)) * $(2)) + $(129)), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r39_value = r39_value.to(tl.bfloat16)
    r39_value = tl.broadcast_to(r39_value, [$(65536)])
    tl.store($(vmul_12_ptr) + r39_z, r39_value)
    r40_z = tl.arange($(65536))
    r40_value = (tl.load($(vmul_11_ptr) + ((((r40_z / $(32768)) % $(2)) * $(32768)) + (((r40_z / $(4096)) % $(8)) * $(4096)) + (((r40_z / $(32)) % $(128)) * $(32)) + (r40_z % $(32))), dtype=tl.bfloat16) - tl.load($(vmul_12_ptr) + ((((r40_z / $(32768)) % $(2)) * $(32768)) + (((r40_z / $(4096)) % $(8)) * $(4096)) + (((r40_z / $(32)) % $(128)) * $(32)) + (r40_z % $(32))), dtype=tl.bfloat16))
    r40_value = r40_value.to(tl.bfloat16)
    r40_value = tl.broadcast_to(r40_value, [$(65536)])
    tl.store($(vsub_ptr) + r40_z, r40_value)
    r41_z = tl.arange($(65536))
    r41_value = (tl.load($(vmm_1_ptr) + ((((r41_z / $(32768)) % $(2)) * $(196608)) + (((r41_z / $(32)) % $(128)) * $(1536)) + (((r41_z / $(4096)) % $(8)) * $(192)) + ((r41_z % $(32)) * $(2)) + $(129)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r41_value = r41_value.to(tl.bfloat16)
    r41_value = tl.broadcast_to(r41_value, [$(65536)])
    tl.store($(vmul_13_ptr) + r41_z, r41_value)
    r42_z = tl.arange($(65536))
    r42_value = (tl.load($(vmm_1_ptr) + ((((r42_z / $(32768)) % $(2)) * $(196608)) + (((r42_z / $(32)) % $(128)) * $(1536)) + (((r42_z / $(4096)) % $(8)) * $(192)) + ((r42_z % $(32)) * $(2)) + $(128)), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r42_value = r42_value.to(tl.bfloat16)
    r42_value = tl.broadcast_to(r42_value, [$(65536)])
    tl.store($(vmul_14_ptr) + r42_z, r42_value)
    r43_z = tl.arange($(65536))
    r43_value = (tl.load($(vmul_13_ptr) + ((((r43_z / $(32768)) % $(2)) * $(32768)) + (((r43_z / $(4096)) % $(8)) * $(4096)) + (((r43_z / $(32)) % $(128)) * $(32)) + (r43_z % $(32))), dtype=tl.bfloat16) + tl.load($(vmul_14_ptr) + ((((r43_z / $(32768)) % $(2)) * $(32768)) + (((r43_z / $(4096)) % $(8)) * $(4096)) + (((r43_z / $(32)) % $(128)) * $(32)) + (r43_z % $(32))), dtype=tl.bfloat16))
    r43_value = r43_value.to(tl.bfloat16)
    r43_value = tl.broadcast_to(r43_value, [$(65536)])
    tl.store($(vadd_4_ptr) + r43_z, r43_value)
    r44_z = tl.arange($(131072))
    r44_value = tl.where((r44_z % $(64)) < $(32), tl.load($(vsub_ptr) + ((((r44_z / $(65536)) % $(2)) * $(32768)) + (((r44_z / $(8192)) % $(8)) * $(4096)) + (((r44_z / $(64)) % $(128)) * $(32)) + (((r44_z % $(64)) - $(0)) % $(32))), dtype=tl.bfloat16), tl.load($(vadd_4_ptr) + ((((r44_z / $(65536)) % $(2)) * $(32768)) + (((r44_z / $(8192)) % $(8)) * $(4096)) + (((r44_z / $(64)) % $(128)) * $(32)) + (((r44_z % $(64)) - $(32)) % $(32))), dtype=tl.bfloat16))
    r44_value = r44_value.to(tl.bfloat16)
    r44_value = tl.broadcast_to(r44_value, [$(131072)])
    tl.store($(vcat_ptr) + r44_z, r44_value)
    r45_z = tl.arange($(8192))
    r45_value = (tl.load($(vmm_2_ptr) + ((((r45_z / $(4096)) % $(2)) * $(73728)) + (((r45_z / $(32)) % $(128)) * $(576)) + ((r45_z % $(32)) * $(2)) + $(512)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r45_value = r45_value.to(tl.bfloat16)
    r45_value = tl.broadcast_to(r45_value, [$(8192)])
    tl.store($(vmul_15_ptr) + r45_z, r45_value)
    r46_z = tl.arange($(8192))
    r46_value = (tl.load($(vmm_2_ptr) + ((((r46_z / $(4096)) % $(2)) * $(73728)) + (((r46_z / $(32)) % $(128)) * $(576)) + ((r46_z % $(32)) * $(2)) + $(513)), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r46_value = r46_value.to(tl.bfloat16)
    r46_value = tl.broadcast_to(r46_value, [$(8192)])
    tl.store($(vmul_16_ptr) + r46_z, r46_value)
    r47_z = tl.arange($(8192))
    r47_value = (tl.load($(vmul_15_ptr) + ((((r47_z / $(4096)) % $(2)) * $(4096)) + (((r47_z / $(32)) % $(128)) * $(32)) + (r47_z % $(32))), dtype=tl.bfloat16) - tl.load($(vmul_16_ptr) + ((((r47_z / $(4096)) % $(2)) * $(4096)) + (((r47_z / $(32)) % $(128)) * $(32)) + (r47_z % $(32))), dtype=tl.bfloat16))
    r47_value = r47_value.to(tl.bfloat16)
    r47_value = tl.broadcast_to(r47_value, [$(8192)])
    tl.store($(vsub_1_ptr) + r47_z, r47_value)
    r48_z = tl.arange($(8192))
    r48_value = (tl.load($(vmm_2_ptr) + ((((r48_z / $(4096)) % $(2)) * $(73728)) + (((r48_z / $(32)) % $(128)) * $(576)) + ((r48_z % $(32)) * $(2)) + $(513)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r48_value = r48_value.to(tl.bfloat16)
    r48_value = tl.broadcast_to(r48_value, [$(8192)])
    tl.store($(vmul_17_ptr) + r48_z, r48_value)
    r49_z = tl.arange($(8192))
    r49_value = (tl.load($(vmm_2_ptr) + ((((r49_z / $(4096)) % $(2)) * $(73728)) + (((r49_z / $(32)) % $(128)) * $(576)) + ((r49_z % $(32)) * $(2)) + $(512)), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r49_value = r49_value.to(tl.bfloat16)
    r49_value = tl.broadcast_to(r49_value, [$(8192)])
    tl.store($(vmul_18_ptr) + r49_z, r49_value)
    r50_z = tl.arange($(8192))
    r50_value = (tl.load($(vmul_17_ptr) + ((((r50_z / $(4096)) % $(2)) * $(4096)) + (((r50_z / $(32)) % $(128)) * $(32)) + (r50_z % $(32))), dtype=tl.bfloat16) + tl.load($(vmul_18_ptr) + ((((r50_z / $(4096)) % $(2)) * $(4096)) + (((r50_z / $(32)) % $(128)) * $(32)) + (r50_z % $(32))), dtype=tl.bfloat16))
    r50_value = r50_value.to(tl.bfloat16)
    r50_value = tl.broadcast_to(r50_value, [$(8192)])
    tl.store($(vadd_5_ptr) + r50_z, r50_value)
    r51_z = tl.arange($(16384))
    r51_value = tl.where((r51_z % $(64)) < $(32), tl.load($(vsub_1_ptr) + ((((r51_z / $(8192)) % $(2)) * $(4096)) + (((r51_z / $(64)) % $(128)) * $(32)) + (((r51_z % $(64)) - $(0)) % $(32))), dtype=tl.bfloat16), tl.load($(vadd_5_ptr) + ((((r51_z / $(8192)) % $(2)) * $(4096)) + (((r51_z / $(64)) % $(128)) * $(32)) + (((r51_z % $(64)) - $(32)) % $(32))), dtype=tl.bfloat16))
    r51_value = r51_value.to(tl.bfloat16)
    r51_value = tl.broadcast_to(r51_value, [$(16384)])
    tl.store($(vcat_1_ptr) + r51_z, r51_value)
    r52_z = tl.arange($(393216))
    r52_value = tl.where((r52_z % $(192)) < $(128), tl.load($(vmm_1_ptr) + ((((((((r52_z / $(196608)) % $(2)) * $(196608)) + (((r52_z / $(192)) % $(128)) * $(1536)) + (((r52_z / $(24576)) % $(8)) * $(192)) + ((r52_z % $(192)) - $(0))) / $(196608)) % $(2)) * $(196608)) + (((((((r52_z / $(196608)) % $(2)) * $(196608)) + (((r52_z / $(192)) % $(128)) * $(1536)) + (((r52_z / $(24576)) % $(8)) * $(192)) + ((r52_z % $(192)) - $(0))) / $(1536)) % $(128)) * $(1536)) + (((((r52_z / $(196608)) % $(2)) * $(196608)) + (((r52_z / $(192)) % $(128)) * $(1536)) + (((r52_z / $(24576)) % $(8)) * $(192)) + ((r52_z % $(192)) - $(0))) % $(1536))), dtype=tl.bfloat16), tl.load($(vcat_ptr) + ((((r52_z / $(196608)) % $(2)) * $(65536)) + (((r52_z / $(24576)) % $(8)) * $(8192)) + (((r52_z / $(192)) % $(128)) * $(64)) + (((r52_z % $(192)) - $(128)) % $(64))), dtype=tl.bfloat16))
    r52_value = r52_value.to(tl.bfloat16)
    r52_value = tl.broadcast_to(r52_value, [$(393216)])
    tl.store($(vcat_2_ptr) + r52_z, r52_value)
    r53_z = tl.arange($(393216))
    r53_value = tl.where((r53_z % $(192)) < $(128), tl.load($(vmm_3_ptr) + ((((((((r53_z / $(196608)) % $(2)) * $(262144)) + (((r53_z / $(192)) % $(128)) * $(2048)) + (((r53_z / $(24576)) % $(8)) * $(256)) + ((r53_z % $(192)) - $(0))) / $(262144)) % $(2)) * $(262144)) + (((((((r53_z / $(196608)) % $(2)) * $(262144)) + (((r53_z / $(192)) % $(128)) * $(2048)) + (((r53_z / $(24576)) % $(8)) * $(256)) + ((r53_z % $(192)) - $(0))) / $(2048)) % $(128)) * $(2048)) + (((((r53_z / $(196608)) % $(2)) * $(262144)) + (((r53_z / $(192)) % $(128)) * $(2048)) + (((r53_z / $(24576)) % $(8)) * $(256)) + ((r53_z % $(192)) - $(0))) % $(2048))), dtype=tl.bfloat16), tl.load($(vcat_1_ptr) + ((((r53_z / $(196608)) % $(2)) * $(8192)) + (((r53_z / $(192)) % $(128)) * $(64)) + (((r53_z % $(192)) - $(128)) % $(64))), dtype=tl.bfloat16))
    r53_value = r53_value.to(tl.bfloat16)
    r53_value = tl.broadcast_to(r53_value, [$(393216)])
    tl.store($(vcat_3_ptr) + r53_z, r53_value)
    r54_row = tl.arange($(128))
    r54_col = tl.arange($(128))
    r54_lane = tl.arange($(192))
    for r54_group in range($(16)) {
      r54_left = tl.load($(vcat_2_ptr) + (((((((r54_group % $(16)) * $(24576)) + (r54_row[:, None] * $(192)) + r54_lane[None, :]) / $(196608)) % $(2)) * $(196608)) + ((((((r54_group % $(16)) * $(24576)) + (r54_row[:, None] * $(192)) + r54_lane[None, :]) / $(24576)) % $(8)) * $(24576)) + ((((((r54_group % $(16)) * $(24576)) + (r54_row[:, None] * $(192)) + r54_lane[None, :]) / $(192)) % $(128)) * $(192)) + ((((r54_group % $(16)) * $(24576)) + (r54_row[:, None] * $(192)) + r54_lane[None, :]) % $(192))), dtype=tl.bfloat16)
      r54_right = tl.load($(vcat_3_ptr) + (((((((r54_group % $(16)) * $(24576)) + (r54_lane[:, None] * $(128)) + r54_col[None, :]) / $(196608)) % $(2)) * $(196608)) + ((((((r54_group % $(16)) * $(24576)) + (r54_lane[:, None] * $(128)) + r54_col[None, :]) / $(24576)) % $(8)) * $(24576)) + (((((r54_group % $(16)) * $(24576)) + (r54_lane[:, None] * $(128)) + r54_col[None, :]) % $(128)) * $(192)) + (((((r54_group % $(16)) * $(24576)) + (r54_lane[:, None] * $(128)) + r54_col[None, :]) / $(128)) % $(192))), dtype=tl.bfloat16)
      r54_value = tl.dot(r54_left, r54_right)
      r54_value = r54_value.to(tl.bfloat16)
      tl.store($(vbmm_ptr) + ((r54_group * $(16384)) + (r54_row[:, None] * $(128)) + r54_col[None, :]), r54_value)
    }
    r55_z = tl.arange($(262144))
    r55_value = (tl.load($(vbmm_ptr) + ((((r55_z / $(131072)) % $(2)) * $(131072)) + (((r55_z / $(16384)) % $(8)) * $(16384)) + (((r55_z / $(128)) % $(128)) * $(128)) + (r55_z % $(128))), dtype=tl.bfloat16) * tl.cast(0.07216878364870322, tl.bfloat16))
    r55_value = r55_value.to(tl.bfloat16)
    r55_value = tl.broadcast_to(r55_value, [$(262144)])
    tl.store($(vmul_19_ptr) + r55_z, r55_value)
    r56_z = tl.arange($(262144))
    r56_value = (tl.load($(vmul_19_ptr) + ((((r56_z / $(131072)) % $(2)) * $(131072)) + (((r56_z / $(16384)) % $(8)) * $(16384)) + (((r56_z / $(128)) % $(128)) * $(128)) + (r56_z % $(128))), dtype=tl.bfloat16) + tl.load($(vmasked_fill_ptr) + ((((r56_z / $(131072)) % $(2)) * $(16384)) + (((r56_z / $(128)) % $(128)) * $(128)) + (r56_z % $(128))), dtype=tl.bfloat16))
    r56_value = r56_value.to(tl.bfloat16)
    r56_value = tl.broadcast_to(r56_value, [$(262144)])
    tl.store($(vadd_6_ptr) + r56_z, r56_value)
    r57_z = tl.arange($(262144))
    r57_value = tl.load($(vadd_6_ptr) + ((((r57_z / $(131072)) % $(2)) * $(131072)) + (((r57_z / $(16384)) % $(8)) * $(16384)) + (((r57_z / $(128)) % $(128)) * $(128)) + (r57_z % $(128))), dtype=tl.bfloat16)
    r57_value = r57_value.to(tl.float32)
    r57_value = tl.broadcast_to(r57_value, [$(262144)])
    tl.store($(vto_copy_20_ptr) + r57_z, r57_value)
    r58_z = tl.arange($(262144))
    r58_lane = tl.arange($(128))
    r58_row_values = tl.load($(vto_copy_20_ptr) + ((((r58_z[:, None] / $(131072)) % $(2)) * $(131072)) + (((r58_z[:, None] / $(16384)) % $(8)) * $(16384)) + (((r58_z[:, None] / $(128)) % $(128)) * $(128)) + (r58_lane[None, :] % $(128))), dtype=tl.float32)
    r58_maximum = tl.max(r58_row_values, axis=1, keep_dims=true)
    r58_weights = tl.exp(r58_row_values - r58_maximum)
    r58_total = tl.sum(r58_weights, axis=1)
    r58_maximum = tl.max(r58_row_values, axis=1)
    r58_value = tl.exp(tl.load($(vto_copy_20_ptr) + ((((r58_z / $(131072)) % $(2)) * $(131072)) + (((r58_z / $(16384)) % $(8)) * $(16384)) + (((r58_z / $(128)) % $(128)) * $(128)) + (r58_z % $(128))), dtype=tl.float32) - r58_maximum) / r58_total
    r58_value = r58_value.to(tl.float32)
    r58_value = tl.broadcast_to(r58_value, [$(262144)])
    tl.store($(vsoftmax_ptr) + r58_z, r58_value)
    r59_z = tl.arange($(262144))
    r59_value = tl.load($(vsoftmax_ptr) + ((((r59_z / $(131072)) % $(2)) * $(131072)) + (((r59_z / $(16384)) % $(8)) * $(16384)) + (((r59_z / $(128)) % $(128)) * $(128)) + (r59_z % $(128))), dtype=tl.float32)
    r59_value = r59_value.to(tl.bfloat16)
    r59_value = tl.broadcast_to(r59_value, [$(262144)])
    tl.store($(vto_copy_21_ptr) + r59_z, r59_value)
    r60_row = tl.arange($(256))
    r60_col = tl.arange($(1024))
    r60_lane = tl.arange($(512))
    r60_left = tl.load($(input1_ptr) + ((((((r60_row[:, None] * $(512)) + r60_lane[None, :]) / $(65536)) % $(2)) * $(65536)) + (((((r60_row[:, None] * $(512)) + r60_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r60_row[:, None] * $(512)) + r60_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r60_right = tl.load($(vto_copy_6_ptr) + (((r60_lane[:, None] % $(512)) * $(1024)) + (r60_col[None, :] % $(1024))), dtype=tl.bfloat16)
    r60_value = tl.dot(r60_left, r60_right)
    r60_value = r60_value.to(tl.bfloat16)
    tl.store($(vmm_5_ptr) + ((r60_row[:, None] * $(1024)) + r60_col[None, :]), r60_value)
    r61_row = tl.arange($(128))
    r61_col = tl.arange($(128))
    r61_lane = tl.arange($(128))
    for r61_group in range($(16)) {
      r61_left = tl.load($(vto_copy_21_ptr) + (((((((r61_group % $(16)) * $(16384)) + (r61_lane[None, :] * $(128)) + r61_row[:, None]) / $(131072)) % $(2)) * $(131072)) + ((((((r61_group % $(16)) * $(16384)) + (r61_lane[None, :] * $(128)) + r61_row[:, None]) / $(16384)) % $(8)) * $(16384)) + ((((((r61_group % $(16)) * $(16384)) + (r61_lane[None, :] * $(128)) + r61_row[:, None]) / $(128)) % $(128)) * $(128)) + ((((r61_group % $(16)) * $(16384)) + (r61_lane[None, :] * $(128)) + r61_row[:, None]) % $(128))), dtype=tl.bfloat16)
      r61_right = tl.load($(vmm_5_ptr) + (((((((r61_group % $(16)) * $(16384)) + (r61_lane[:, None] * $(128)) + r61_col[None, :]) / $(131072)) % $(2)) * $(131072)) + ((((((r61_group % $(16)) * $(16384)) + (r61_lane[:, None] * $(128)) + r61_col[None, :]) / $(128)) % $(128)) * $(1024)) + ((((((r61_group % $(16)) * $(16384)) + (r61_lane[:, None] * $(128)) + r61_col[None, :]) / $(16384)) % $(8)) * $(128)) + ((((r61_group % $(16)) * $(16384)) + (r61_lane[:, None] * $(128)) + r61_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r61_value = tl.dot(r61_left, r61_right)
      r61_value = r61_value.to(tl.bfloat16)
      tl.store($(vbmm_2_ptr) + ((r61_group * $(16384)) + (r61_row[:, None] * $(128)) + r61_col[None, :]), r61_value)
    }
    r62_row = tl.arange($(128))
    r62_col = tl.arange($(128))
    r62_lane = tl.arange($(128))
    for r62_group in range($(16)) {
      r62_left = tl.load($(vmm_5_ptr) + (((((((r62_group % $(16)) * $(16384)) + (r62_row[:, None] * $(128)) + r62_lane[None, :]) / $(131072)) % $(2)) * $(131072)) + ((((((r62_group % $(16)) * $(16384)) + (r62_row[:, None] * $(128)) + r62_lane[None, :]) / $(128)) % $(128)) * $(1024)) + ((((((r62_group % $(16)) * $(16384)) + (r62_row[:, None] * $(128)) + r62_lane[None, :]) / $(16384)) % $(8)) * $(128)) + ((((r62_group % $(16)) * $(16384)) + (r62_row[:, None] * $(128)) + r62_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r62_right = tl.load($(vmm_3_ptr) + (((((((r62_group % $(16)) * $(16384)) + (r62_col[None, :] * $(128)) + r62_lane[:, None]) / $(131072)) % $(2)) * $(262144)) + ((((((r62_group % $(16)) * $(16384)) + (r62_col[None, :] * $(128)) + r62_lane[:, None]) / $(128)) % $(128)) * $(2048)) + ((((((r62_group % $(16)) * $(16384)) + (r62_col[None, :] * $(128)) + r62_lane[:, None]) / $(16384)) % $(8)) * $(256)) + ((((r62_group % $(16)) * $(16384)) + (r62_col[None, :] * $(128)) + r62_lane[:, None]) % $(128)) + $(128)), dtype=tl.bfloat16)
      r62_value = tl.dot(r62_left, r62_right)
      r62_value = r62_value.to(tl.bfloat16)
      tl.store($(vbmm_3_ptr) + ((r62_group * $(16384)) + (r62_row[:, None] * $(128)) + r62_col[None, :]), r62_value)
    }
    r63_z = tl.arange($(262144))
    r63_value = tl.load($(vbmm_3_ptr) + ((((r63_z / $(131072)) % $(2)) * $(131072)) + (((r63_z / $(16384)) % $(8)) * $(16384)) + (((r63_z / $(128)) % $(128)) * $(128)) + (r63_z % $(128))), dtype=tl.bfloat16)
    r63_value = r63_value.to(tl.float32)
    r63_value = tl.broadcast_to(r63_value, [$(262144)])
    tl.store($(vto_copy_22_ptr) + r63_z, r63_value)
    r64_z = tl.arange($(262144))
    r64_lane = tl.arange($(128))
    r64_row_values = tl.load($(vsoftmax_ptr) + ((((r64_z[:, None] / $(131072)) % $(2)) * $(131072)) + (((r64_z[:, None] / $(16384)) % $(8)) * $(16384)) + (((r64_z[:, None] / $(128)) % $(128)) * $(128)) + (r64_lane[None, :] % $(128))), dtype=tl.float32)
    r64_row_grad = tl.load($(vto_copy_22_ptr) + ((((r64_z[:, None] / $(131072)) % $(2)) * $(131072)) + (((r64_z[:, None] / $(16384)) % $(8)) * $(16384)) + (((r64_z[:, None] / $(128)) % $(128)) * $(128)) + (r64_lane[None, :] % $(128))), dtype=tl.float32)
    r64_weighted = r64_row_grad * r64_row_values
    r64_dot_grad = tl.sum(r64_weighted, axis=1)
    r64_value = (tl.load($(vto_copy_22_ptr) + ((((r64_z / $(131072)) % $(2)) * $(131072)) + (((r64_z / $(16384)) % $(8)) * $(16384)) + (((r64_z / $(128)) % $(128)) * $(128)) + (r64_z % $(128))), dtype=tl.float32) - r64_dot_grad) * tl.load($(vsoftmax_ptr) + ((((r64_z / $(131072)) % $(2)) * $(131072)) + (((r64_z / $(16384)) % $(8)) * $(16384)) + (((r64_z / $(128)) % $(128)) * $(128)) + (r64_z % $(128))), dtype=tl.float32)
    r64_value = r64_value.to(tl.float32)
    r64_value = tl.broadcast_to(r64_value, [$(262144)])
    tl.store($(vsoftmax_backward_data_ptr) + r64_z, r64_value)
    r65_z = tl.arange($(262144))
    r65_value = tl.load($(vsoftmax_backward_data_ptr) + ((((r65_z / $(131072)) % $(2)) * $(131072)) + (((r65_z / $(16384)) % $(8)) * $(16384)) + (((r65_z / $(128)) % $(128)) * $(128)) + (r65_z % $(128))), dtype=tl.float32)
    r65_value = r65_value.to(tl.bfloat16)
    r65_value = tl.broadcast_to(r65_value, [$(262144)])
    tl.store($(vto_copy_23_ptr) + r65_z, r65_value)
    r66_z = tl.arange($(262144))
    r66_value = (tl.load($(vto_copy_23_ptr) + ((((r66_z / $(131072)) % $(2)) * $(131072)) + (((r66_z / $(16384)) % $(8)) * $(16384)) + (((r66_z / $(128)) % $(128)) * $(128)) + (r66_z % $(128))), dtype=tl.bfloat16) * tl.cast(0.07216878364870322, tl.bfloat16))
    r66_value = r66_value.to(tl.bfloat16)
    r66_value = tl.broadcast_to(r66_value, [$(262144)])
    tl.store($(vmul_20_ptr) + r66_z, r66_value)
    r67_row = tl.arange($(192))
    r67_col = tl.arange($(128))
    r67_lane = tl.arange($(128))
    for r67_group in range($(16)) {
      r67_left = tl.load($(vcat_2_ptr) + (((((((r67_group % $(16)) * $(24576)) + (r67_lane[None, :] * $(192)) + r67_row[:, None]) / $(196608)) % $(2)) * $(196608)) + ((((((r67_group % $(16)) * $(24576)) + (r67_lane[None, :] * $(192)) + r67_row[:, None]) / $(24576)) % $(8)) * $(24576)) + ((((((r67_group % $(16)) * $(24576)) + (r67_lane[None, :] * $(192)) + r67_row[:, None]) / $(192)) % $(128)) * $(192)) + ((((r67_group % $(16)) * $(24576)) + (r67_lane[None, :] * $(192)) + r67_row[:, None]) % $(192))), dtype=tl.bfloat16)
      r67_right = tl.load($(vmul_20_ptr) + (((((((r67_group % $(16)) * $(16384)) + (r67_lane[:, None] * $(128)) + r67_col[None, :]) / $(131072)) % $(2)) * $(131072)) + ((((((r67_group % $(16)) * $(16384)) + (r67_lane[:, None] * $(128)) + r67_col[None, :]) / $(16384)) % $(8)) * $(16384)) + ((((((r67_group % $(16)) * $(16384)) + (r67_lane[:, None] * $(128)) + r67_col[None, :]) / $(128)) % $(128)) * $(128)) + ((((r67_group % $(16)) * $(16384)) + (r67_lane[:, None] * $(128)) + r67_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r67_value = tl.dot(r67_left, r67_right)
      r67_value = r67_value.to(tl.bfloat16)
      tl.store($(vbmm_4_ptr) + ((r67_group * $(24576)) + (r67_row[:, None] * $(128)) + r67_col[None, :]), r67_value)
    }
    r68_row = tl.arange($(128))
    r68_col = tl.arange($(192))
    r68_lane = tl.arange($(128))
    for r68_group in range($(16)) {
      r68_left = tl.load($(vmul_20_ptr) + (((((((r68_group % $(16)) * $(16384)) + (r68_row[:, None] * $(128)) + r68_lane[None, :]) / $(131072)) % $(2)) * $(131072)) + ((((((r68_group % $(16)) * $(16384)) + (r68_row[:, None] * $(128)) + r68_lane[None, :]) / $(16384)) % $(8)) * $(16384)) + ((((((r68_group % $(16)) * $(16384)) + (r68_row[:, None] * $(128)) + r68_lane[None, :]) / $(128)) % $(128)) * $(128)) + ((((r68_group % $(16)) * $(16384)) + (r68_row[:, None] * $(128)) + r68_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r68_right = tl.load($(vcat_3_ptr) + (((((((r68_group % $(16)) * $(24576)) + (r68_col[None, :] * $(128)) + r68_lane[:, None]) / $(196608)) % $(2)) * $(196608)) + ((((((r68_group % $(16)) * $(24576)) + (r68_col[None, :] * $(128)) + r68_lane[:, None]) / $(24576)) % $(8)) * $(24576)) + (((((r68_group % $(16)) * $(24576)) + (r68_col[None, :] * $(128)) + r68_lane[:, None]) % $(128)) * $(192)) + (((((r68_group % $(16)) * $(24576)) + (r68_col[None, :] * $(128)) + r68_lane[:, None]) / $(128)) % $(192))), dtype=tl.bfloat16)
      r68_value = tl.dot(r68_left, r68_right)
      r68_value = r68_value.to(tl.bfloat16)
      tl.store($(vbmm_5_ptr) + ((r68_group * $(24576)) + (r68_row[:, None] * $(192)) + r68_col[None, :]), r68_value)
    }
    r69_z = tl.arange($(16384))
    r69_lane = tl.arange($(8))
    r69_value = tl.sum(tl.load($(vbmm_4_ptr) + ((((r69_z[:, None] / $(8192)) % $(2)) * $(196608)) + (r69_lane[None, :] * $(24576)) + ((r69_z[:, None] % $(64)) * $(128)) + ((r69_z[:, None] / $(64)) % $(128)) + $(16384)), dtype=tl.bfloat16), axis=1)
    r69_value = r69_value.to(tl.bfloat16)
    r69_value = tl.broadcast_to(r69_value, [$(16384)])
    tl.store($(vsum_1_ptr) + r69_z, r69_value)
    r70_z = tl.arange($(8192))
    r70_value = (tl.load($(vsum_1_ptr) + ((((r70_z / $(4096)) % $(2)) * $(8192)) + (((r70_z / $(32)) % $(128)) * $(64)) + (r70_z % $(32)) + $(32)), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r70_value = r70_value.to(tl.bfloat16)
    r70_value = tl.broadcast_to(r70_value, [$(8192)])
    tl.store($(vmul_21_ptr) + r70_z, r70_value)
    r71_z = tl.arange($(8192))
    r71_value = (tl.load($(vsum_1_ptr) + ((((r71_z / $(4096)) % $(2)) * $(8192)) + (((r71_z / $(32)) % $(128)) * $(64)) + (r71_z % $(32)) + $(32)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r71_value = r71_value.to(tl.bfloat16)
    r71_value = tl.broadcast_to(r71_value, [$(8192)])
    tl.store($(vmul_22_ptr) + r71_z, r71_value)
    r72_z = tl.arange($(8192))
    r72_value = -tl.load($(vsum_1_ptr) + ((((r72_z / $(4096)) % $(2)) * $(8192)) + (((r72_z / $(32)) % $(128)) * $(64)) + (r72_z % $(32))), dtype=tl.bfloat16)
    r72_value = r72_value.to(tl.bfloat16)
    r72_value = tl.broadcast_to(r72_value, [$(8192)])
    tl.store($(vneg_ptr) + r72_z, r72_value)
    r73_z = tl.arange($(8192))
    r73_value = (tl.load($(vneg_ptr) + ((((r73_z / $(4096)) % $(2)) * $(4096)) + (((r73_z / $(32)) % $(128)) * $(32)) + (r73_z % $(32))), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r73_value = r73_value.to(tl.bfloat16)
    r73_value = tl.broadcast_to(r73_value, [$(8192)])
    tl.store($(vmul_23_ptr) + r73_z, r73_value)
    r74_z = tl.arange($(8192))
    r74_value = (tl.load($(vmul_22_ptr) + ((((r74_z / $(4096)) % $(2)) * $(4096)) + (((r74_z / $(32)) % $(128)) * $(32)) + (r74_z % $(32))), dtype=tl.bfloat16) + tl.load($(vmul_23_ptr) + ((((r74_z / $(4096)) % $(2)) * $(4096)) + (((r74_z / $(32)) % $(128)) * $(32)) + (r74_z % $(32))), dtype=tl.bfloat16))
    r74_value = r74_value.to(tl.bfloat16)
    r74_value = tl.broadcast_to(r74_value, [$(8192)])
    tl.store($(vadd_7_ptr) + r74_z, r74_value)
    r75_z = tl.arange($(8192))
    r75_value = (tl.load($(vsum_1_ptr) + ((((r75_z / $(4096)) % $(2)) * $(8192)) + (((r75_z / $(32)) % $(128)) * $(64)) + (r75_z % $(32))), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r75_value = r75_value.to(tl.bfloat16)
    r75_value = tl.broadcast_to(r75_value, [$(8192)])
    tl.store($(vmul_24_ptr) + r75_z, r75_value)
    r76_z = tl.arange($(8192))
    r76_value = (tl.load($(vmul_21_ptr) + ((((r76_z / $(4096)) % $(2)) * $(4096)) + (((r76_z / $(32)) % $(128)) * $(32)) + (r76_z % $(32))), dtype=tl.bfloat16) + tl.load($(vmul_24_ptr) + ((((r76_z / $(4096)) % $(2)) * $(4096)) + (((r76_z / $(32)) % $(128)) * $(32)) + (r76_z % $(32))), dtype=tl.bfloat16))
    r76_value = r76_value.to(tl.bfloat16)
    r76_value = tl.broadcast_to(r76_value, [$(8192)])
    tl.store($(vadd_8_ptr) + r76_z, r76_value)
    r77_z = tl.arange($(65536))
    r77_value = (tl.load($(vbmm_5_ptr) + ((((r77_z / $(32768)) % $(2)) * $(196608)) + (((r77_z / $(4096)) % $(8)) * $(24576)) + (((r77_z / $(32)) % $(128)) * $(192)) + (r77_z % $(32)) + $(160)), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r77_value = r77_value.to(tl.bfloat16)
    r77_value = tl.broadcast_to(r77_value, [$(65536)])
    tl.store($(vmul_25_ptr) + r77_z, r77_value)
    r78_z = tl.arange($(65536))
    r78_value = (tl.load($(vbmm_5_ptr) + ((((r78_z / $(32768)) % $(2)) * $(196608)) + (((r78_z / $(4096)) % $(8)) * $(24576)) + (((r78_z / $(32)) % $(128)) * $(192)) + (r78_z % $(32)) + $(160)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r78_value = r78_value.to(tl.bfloat16)
    r78_value = tl.broadcast_to(r78_value, [$(65536)])
    tl.store($(vmul_26_ptr) + r78_z, r78_value)
    r79_z = tl.arange($(65536))
    r79_value = -tl.load($(vbmm_5_ptr) + ((((r79_z / $(32768)) % $(2)) * $(196608)) + (((r79_z / $(4096)) % $(8)) * $(24576)) + (((r79_z / $(32)) % $(128)) * $(192)) + (r79_z % $(32)) + $(128)), dtype=tl.bfloat16)
    r79_value = r79_value.to(tl.bfloat16)
    r79_value = tl.broadcast_to(r79_value, [$(65536)])
    tl.store($(vneg_1_ptr) + r79_z, r79_value)
    r80_z = tl.arange($(65536))
    r80_value = (tl.load($(vneg_1_ptr) + ((((r80_z / $(32768)) % $(2)) * $(32768)) + (((r80_z / $(4096)) % $(8)) * $(4096)) + (((r80_z / $(32)) % $(128)) * $(32)) + (r80_z % $(32))), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r80_value = r80_value.to(tl.bfloat16)
    r80_value = tl.broadcast_to(r80_value, [$(65536)])
    tl.store($(vmul_27_ptr) + r80_z, r80_value)
    r81_z = tl.arange($(65536))
    r81_value = (tl.load($(vmul_26_ptr) + ((((r81_z / $(32768)) % $(2)) * $(32768)) + (((r81_z / $(4096)) % $(8)) * $(4096)) + (((r81_z / $(32)) % $(128)) * $(32)) + (r81_z % $(32))), dtype=tl.bfloat16) + tl.load($(vmul_27_ptr) + ((((r81_z / $(32768)) % $(2)) * $(32768)) + (((r81_z / $(4096)) % $(8)) * $(4096)) + (((r81_z / $(32)) % $(128)) * $(32)) + (r81_z % $(32))), dtype=tl.bfloat16))
    r81_value = r81_value.to(tl.bfloat16)
    r81_value = tl.broadcast_to(r81_value, [$(65536)])
    tl.store($(vadd_9_ptr) + r81_z, r81_value)
    r82_z = tl.arange($(65536))
    r82_value = (tl.load($(vbmm_5_ptr) + ((((r82_z / $(32768)) % $(2)) * $(196608)) + (((r82_z / $(4096)) % $(8)) * $(24576)) + (((r82_z / $(32)) % $(128)) * $(192)) + (r82_z % $(32)) + $(128)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r82_value = r82_value.to(tl.bfloat16)
    r82_value = tl.broadcast_to(r82_value, [$(65536)])
    tl.store($(vmul_28_ptr) + r82_z, r82_value)
    r83_z = tl.arange($(65536))
    r83_value = (tl.load($(vmul_25_ptr) + ((((r83_z / $(32768)) % $(2)) * $(32768)) + (((r83_z / $(4096)) % $(8)) * $(4096)) + (((r83_z / $(32)) % $(128)) * $(32)) + (r83_z % $(32))), dtype=tl.bfloat16) + tl.load($(vmul_28_ptr) + ((((r83_z / $(32768)) % $(2)) * $(32768)) + (((r83_z / $(4096)) % $(8)) * $(4096)) + (((r83_z / $(32)) % $(128)) * $(32)) + (r83_z % $(32))), dtype=tl.bfloat16))
    r83_value = r83_value.to(tl.bfloat16)
    r83_value = tl.broadcast_to(r83_value, [$(65536)])
    tl.store($(vadd_10_ptr) + r83_z, r83_value)
    r84_z = tl.arange($(16384))
    r84_value = tl.where((((r84_z % $(64)) >= $(1)) & ((r84_z % $(64)) < $(64)) & (((r84_z % $(64)) - $(1)) % $(2) == $(0))), tl.load($(vadd_7_ptr) + ((((r84_z / $(8192)) % $(2)) * $(4096)) + (((r84_z / $(64)) % $(128)) * $(32)) + ((((r84_z % $(64)) - $(1)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r84_value = r84_value.to(tl.bfloat16)
    r84_value = tl.broadcast_to(r84_value, [$(16384)])
    tl.store($(vslice_backward_ptr) + r84_z, r84_value)
    r85_z = tl.arange($(16384))
    r85_value = tl.where((((r85_z % $(64)) >= $(0)) & ((r85_z % $(64)) < $(64)) & (((r85_z % $(64)) - $(0)) % $(2) == $(0))), tl.load($(vadd_8_ptr) + ((((r85_z / $(8192)) % $(2)) * $(4096)) + (((r85_z / $(64)) % $(128)) * $(32)) + ((((r85_z % $(64)) - $(0)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r85_value = r85_value.to(tl.bfloat16)
    r85_value = tl.broadcast_to(r85_value, [$(16384)])
    tl.store($(vslice_backward_1_ptr) + r85_z, r85_value)
    r86_z = tl.arange($(16384))
    r86_value = (tl.load($(vslice_backward_ptr) + ((((r86_z / $(8192)) % $(2)) * $(8192)) + (((r86_z / $(64)) % $(128)) * $(64)) + (r86_z % $(64))), dtype=tl.bfloat16) + tl.load($(vslice_backward_1_ptr) + ((((r86_z / $(8192)) % $(2)) * $(8192)) + (((r86_z / $(64)) % $(128)) * $(64)) + (r86_z % $(64))), dtype=tl.bfloat16))
    r86_value = r86_value.to(tl.bfloat16)
    r86_value = tl.broadcast_to(r86_value, [$(16384)])
    tl.store($(vadd_11_ptr) + r86_z, r86_value)
    r87_z = tl.arange($(131072))
    r87_value = tl.where((((r87_z % $(64)) >= $(1)) & ((r87_z % $(64)) < $(64)) & (((r87_z % $(64)) - $(1)) % $(2) == $(0))), tl.load($(vadd_9_ptr) + ((((r87_z / $(65536)) % $(2)) * $(32768)) + (((r87_z / $(8192)) % $(8)) * $(4096)) + (((r87_z / $(64)) % $(128)) * $(32)) + ((((r87_z % $(64)) - $(1)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r87_value = r87_value.to(tl.bfloat16)
    r87_value = tl.broadcast_to(r87_value, [$(131072)])
    tl.store($(vslice_backward_2_ptr) + r87_z, r87_value)
    r88_z = tl.arange($(131072))
    r88_value = tl.where((((r88_z % $(64)) >= $(0)) & ((r88_z % $(64)) < $(64)) & (((r88_z % $(64)) - $(0)) % $(2) == $(0))), tl.load($(vadd_10_ptr) + ((((r88_z / $(65536)) % $(2)) * $(32768)) + (((r88_z / $(8192)) % $(8)) * $(4096)) + (((r88_z / $(64)) % $(128)) * $(32)) + ((((r88_z % $(64)) - $(0)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r88_value = r88_value.to(tl.bfloat16)
    r88_value = tl.broadcast_to(r88_value, [$(131072)])
    tl.store($(vslice_backward_3_ptr) + r88_z, r88_value)
    r89_z = tl.arange($(131072))
    r89_value = (tl.load($(vslice_backward_2_ptr) + ((((r89_z / $(65536)) % $(2)) * $(65536)) + (((r89_z / $(8192)) % $(8)) * $(8192)) + (((r89_z / $(64)) % $(128)) * $(64)) + (r89_z % $(64))), dtype=tl.bfloat16) + tl.load($(vslice_backward_3_ptr) + ((((r89_z / $(65536)) % $(2)) * $(65536)) + (((r89_z / $(8192)) % $(8)) * $(8192)) + (((r89_z / $(64)) % $(128)) * $(64)) + (r89_z % $(64))), dtype=tl.bfloat16))
    r89_value = r89_value.to(tl.bfloat16)
    r89_value = tl.broadcast_to(r89_value, [$(131072)])
    tl.store($(vadd_12_ptr) + r89_z, r89_value)
    r90_z = tl.arange($(524288))
    r90_value = tl.where((r90_z % $(256)) < $(128), tl.load($(vbmm_4_ptr) + ((((((((r90_z / $(262144)) % $(2)) * $(196608)) + (((r90_z / $(32768)) % $(8)) * $(24576)) + (((r90_z % $(256)) - $(0)) * $(128)) + ((r90_z / $(256)) % $(128))) / $(24576)) % $(16)) * $(24576)) + ((((((r90_z / $(262144)) % $(2)) * $(1536)) + (((r90_z / $(32768)) % $(8)) * $(192)) + ((r90_z % $(256)) - $(0))) % $(192)) * $(128)) + ((r90_z / $(256)) % $(128))), dtype=tl.bfloat16), tl.load($(vbmm_2_ptr) + ((((((((r90_z / $(262144)) % $(2)) * $(131072)) + (((r90_z / $(32768)) % $(8)) * $(16384)) + (((r90_z / $(256)) % $(128)) * $(128)) + ((r90_z % $(256)) - $(128))) / $(16384)) % $(16)) * $(16384)) + (((((((r90_z / $(262144)) % $(2)) * $(131072)) + (((r90_z / $(32768)) % $(8)) * $(16384)) + (((r90_z / $(256)) % $(128)) * $(128)) + ((r90_z % $(256)) - $(128))) / $(128)) % $(128)) * $(128)) + (((((r90_z / $(262144)) % $(2)) * $(131072)) + (((r90_z / $(32768)) % $(8)) * $(16384)) + (((r90_z / $(256)) % $(128)) * $(128)) + ((r90_z % $(256)) - $(128))) % $(128))), dtype=tl.bfloat16))
    r90_value = r90_value.to(tl.bfloat16)
    r90_value = tl.broadcast_to(r90_value, [$(524288)])
    tl.store($(vcat_4_ptr) + r90_z, r90_value)
    r91_z = tl.arange($(393216))
    r91_value = tl.where((r91_z % $(192)) < $(128), tl.load($(vbmm_5_ptr) + ((((((((r91_z / $(196608)) % $(2)) * $(196608)) + (((r91_z / $(24576)) % $(8)) * $(24576)) + (((r91_z / $(192)) % $(128)) * $(192)) + ((r91_z % $(192)) - $(0))) / $(24576)) % $(16)) * $(24576)) + (((((((r91_z / $(196608)) % $(2)) * $(196608)) + (((r91_z / $(24576)) % $(8)) * $(24576)) + (((r91_z / $(192)) % $(128)) * $(192)) + ((r91_z % $(192)) - $(0))) / $(192)) % $(128)) * $(192)) + (((((r91_z / $(196608)) % $(2)) * $(196608)) + (((r91_z / $(24576)) % $(8)) * $(24576)) + (((r91_z / $(192)) % $(128)) * $(192)) + ((r91_z % $(192)) - $(0))) % $(192))), dtype=tl.bfloat16), tl.load($(vadd_12_ptr) + ((((r91_z / $(196608)) % $(2)) * $(65536)) + (((r91_z / $(24576)) % $(8)) * $(8192)) + (((r91_z / $(192)) % $(128)) * $(64)) + (((r91_z % $(192)) - $(128)) % $(64))), dtype=tl.bfloat16))
    r91_value = r91_value.to(tl.bfloat16)
    r91_value = tl.broadcast_to(r91_value, [$(393216)])
    tl.store($(vcat_5_ptr) + r91_z, r91_value)
    r92_row = tl.arange($(256))
    r92_col = tl.arange($(512))
    r92_lane = tl.arange($(2048))
    r92_left = tl.load($(vcat_4_ptr) + ((((((r92_row[:, None] * $(2048)) + r92_lane[None, :]) / $(262144)) % $(2)) * $(262144)) + (((((((((r92_row[:, None] * $(2048)) + r92_lane[None, :]) / $(262144)) % $(2)) * $(262144)) + (((((r92_row[:, None] * $(2048)) + r92_lane[None, :]) / $(2048)) % $(128)) * $(2048)) + (((r92_row[:, None] * $(2048)) + r92_lane[None, :]) % $(2048))) / $(256)) % $(8)) * $(32768)) + (((((r92_row[:, None] * $(2048)) + r92_lane[None, :]) / $(2048)) % $(128)) * $(256)) + (((((((r92_row[:, None] * $(2048)) + r92_lane[None, :]) / $(262144)) % $(2)) * $(262144)) + (((((r92_row[:, None] * $(2048)) + r92_lane[None, :]) / $(2048)) % $(128)) * $(2048)) + (((r92_row[:, None] * $(2048)) + r92_lane[None, :]) % $(2048))) % $(256))), dtype=tl.bfloat16)
    r92_right = tl.load($(vto_copy_5_ptr) + (((r92_lane[:, None] % $(2048)) * $(512)) + (r92_col[None, :] % $(512))), dtype=tl.bfloat16)
    r92_value = tl.dot(r92_left, r92_right)
    r92_value = r92_value.to(tl.bfloat16)
    tl.store($(vmm_6_ptr) + ((r92_row[:, None] * $(512)) + r92_col[None, :]), r92_value)
    r93_z = tl.arange($(131072))
    r93_value = tl.load($(vmm_6_ptr) + ((((r93_z / $(65536)) % $(2)) * $(65536)) + (((r93_z / $(512)) % $(128)) * $(512)) + (r93_z % $(512))), dtype=tl.bfloat16)
    r93_value = r93_value.to(tl.float32)
    r93_value = tl.broadcast_to(r93_value, [$(131072)])
    tl.store($(vto_copy_24_ptr) + r93_z, r93_value)
    r94_z = tl.arange($(131072))
    r94_value = (tl.load($(vto_copy_24_ptr) + ((((r94_z / $(65536)) % $(2)) * $(65536)) + (((r94_z / $(512)) % $(128)) * $(512)) + (r94_z % $(512))), dtype=tl.float32) * tl.cast(tl.load($(vto_copy_4_ptr) + (r94_z % $(512)), dtype=tl.bfloat16), tl.float32))
    r94_value = r94_value.to(tl.float32)
    r94_value = tl.broadcast_to(r94_value, [$(131072)])
    tl.store($(vmul_29_ptr) + r94_z, r94_value)
    r95_z = tl.arange($(131072))
    r95_value = (tl.load($(vmul_29_ptr) + ((((r95_z / $(65536)) % $(2)) * $(65536)) + (((r95_z / $(512)) % $(128)) * $(512)) + (r95_z % $(512))), dtype=tl.float32) * tl.load($(vto_copy_18_ptr) + ((((r95_z / $(65536)) % $(2)) * $(65536)) + (((r95_z / $(512)) % $(128)) * $(512)) + (r95_z % $(512))), dtype=tl.float32))
    r95_value = r95_value.to(tl.float32)
    r95_value = tl.broadcast_to(r95_value, [$(131072)])
    tl.store($(vmul_30_ptr) + r95_z, r95_value)
    r96_z = tl.arange($(131072))
    r96_value = (tl.load($(vmul_29_ptr) + ((((r96_z / $(65536)) % $(2)) * $(65536)) + (((r96_z / $(512)) % $(128)) * $(512)) + (r96_z % $(512))), dtype=tl.float32) * tl.load($(vrsqrt_1_ptr) + ((((r96_z / $(65536)) % $(2)) * $(128)) + ((r96_z / $(512)) % $(128))), dtype=tl.float32))
    r96_value = r96_value.to(tl.float32)
    r96_value = tl.broadcast_to(r96_value, [$(131072)])
    tl.store($(vmul_31_ptr) + r96_z, r96_value)
    r97_z = tl.arange($(256))
    r97_lane = tl.arange($(512))
    r97_value = tl.sum(tl.load($(vmul_30_ptr) + ((((r97_z[:, None] / $(128)) % $(2)) * $(65536)) + ((r97_z[:, None] % $(128)) * $(512)) + r97_lane[None, :]), dtype=tl.float32), axis=1)
    r97_value = r97_value.to(tl.float32)
    r97_value = tl.broadcast_to(r97_value, [$(256)])
    tl.store($(vsum_2_ptr) + r97_z, r97_value)
    r98_z = tl.arange($(256))
    r98_value = ((tl.load($(vrsqrt_1_ptr) + ((((r98_z / $(128)) % $(2)) * $(128)) + (r98_z % $(128))), dtype=tl.float32)) * (tl.load($(vrsqrt_1_ptr) + ((((r98_z / $(128)) % $(2)) * $(128)) + (r98_z % $(128))), dtype=tl.float32)) * (tl.load($(vrsqrt_1_ptr) + ((((r98_z / $(128)) % $(2)) * $(128)) + (r98_z % $(128))), dtype=tl.float32)))
    r98_value = r98_value.to(tl.float32)
    r98_value = tl.broadcast_to(r98_value, [$(256)])
    tl.store($(vpow_3_ptr) + r98_z, r98_value)
    r99_z = tl.arange($(256))
    r99_value = (tl.load($(vsum_2_ptr) + ((((r99_z / $(128)) % $(2)) * $(128)) + (r99_z % $(128))), dtype=tl.float32) * -0.5)
    r99_value = r99_value.to(tl.float32)
    r99_value = tl.broadcast_to(r99_value, [$(256)])
    tl.store($(vmul_32_ptr) + r99_z, r99_value)
    r100_z = tl.arange($(256))
    r100_value = (tl.load($(vmul_32_ptr) + ((((r100_z / $(128)) % $(2)) * $(128)) + (r100_z % $(128))), dtype=tl.float32) * tl.load($(vpow_3_ptr) + ((((r100_z / $(128)) % $(2)) * $(128)) + (r100_z % $(128))), dtype=tl.float32))
    r100_value = r100_value.to(tl.float32)
    r100_value = tl.broadcast_to(r100_value, [$(256)])
    tl.store($(vmul_33_ptr) + r100_z, r100_value)
    r101_z = tl.arange($(131072))
    r101_value = (tl.load($(vmul_33_ptr) + ((((r101_z / $(65536)) % $(2)) * $(128)) + ((r101_z / $(512)) % $(128))), dtype=tl.float32) / 512.0)
    r101_value = r101_value.to(tl.float32)
    r101_value = tl.broadcast_to(r101_value, [$(131072)])
    tl.store($(vdiv_ptr) + r101_z, r101_value)
    r102_z = tl.arange($(131072))
    r102_value = ((tl.load($(vto_copy_18_ptr) + ((((r102_z / $(65536)) % $(2)) * $(65536)) + (((r102_z / $(512)) % $(128)) * $(512)) + (r102_z % $(512))), dtype=tl.float32)))
    r102_value = r102_value.to(tl.float32)
    r102_value = tl.broadcast_to(r102_value, [$(131072)])
    tl.store($(vpow_4_ptr) + r102_z, r102_value)
    r103_z = tl.arange($(131072))
    r103_value = (tl.load($(vpow_4_ptr) + ((((r103_z / $(65536)) % $(2)) * $(65536)) + (((r103_z / $(512)) % $(128)) * $(512)) + (r103_z % $(512))), dtype=tl.float32) * 2.0)
    r103_value = r103_value.to(tl.float32)
    r103_value = tl.broadcast_to(r103_value, [$(131072)])
    tl.store($(vmul_34_ptr) + r103_z, r103_value)
    r104_z = tl.arange($(131072))
    r104_value = (tl.load($(vdiv_ptr) + ((((r104_z / $(65536)) % $(2)) * $(65536)) + (((r104_z / $(512)) % $(128)) * $(512)) + (r104_z % $(512))), dtype=tl.float32) * tl.load($(vmul_34_ptr) + ((((r104_z / $(65536)) % $(2)) * $(65536)) + (((r104_z / $(512)) % $(128)) * $(512)) + (r104_z % $(512))), dtype=tl.float32))
    r104_value = r104_value.to(tl.float32)
    r104_value = tl.broadcast_to(r104_value, [$(131072)])
    tl.store($(vmul_35_ptr) + r104_z, r104_value)
    r105_z = tl.arange($(131072))
    r105_value = (tl.load($(vmul_31_ptr) + ((((r105_z / $(65536)) % $(2)) * $(65536)) + (((r105_z / $(512)) % $(128)) * $(512)) + (r105_z % $(512))), dtype=tl.float32) + tl.load($(vmul_35_ptr) + ((((r105_z / $(65536)) % $(2)) * $(65536)) + (((r105_z / $(512)) % $(128)) * $(512)) + (r105_z % $(512))), dtype=tl.float32))
    r105_value = r105_value.to(tl.float32)
    r105_value = tl.broadcast_to(r105_value, [$(131072)])
    tl.store($(vadd_13_ptr) + r105_z, r105_value)
    r106_z = tl.arange($(131072))
    r106_value = tl.load($(vadd_13_ptr) + ((((r106_z / $(65536)) % $(2)) * $(65536)) + (((r106_z / $(512)) % $(128)) * $(512)) + (r106_z % $(512))), dtype=tl.float32)
    r106_value = r106_value.to(tl.bfloat16)
    r106_value = tl.broadcast_to(r106_value, [$(131072)])
    tl.store($(vto_copy_25_ptr) + r106_z, r106_value)
    r107_z = tl.arange($(147456))
    r107_value = tl.where((r107_z % $(576)) < $(512), tl.load($(vto_copy_25_ptr) + ((((r107_z / $(73728)) % $(2)) * $(65536)) + (((r107_z / $(576)) % $(128)) * $(512)) + (((r107_z % $(576)) - $(0)) % $(512))), dtype=tl.bfloat16), tl.load($(vadd_11_ptr) + ((((((((r107_z / $(73728)) % $(2)) * $(8192)) + (((r107_z / $(576)) % $(128)) * $(64)) + ((r107_z % $(576)) - $(512))) / $(8192)) % $(2)) * $(8192)) + (((((((r107_z / $(73728)) % $(2)) * $(8192)) + (((r107_z / $(576)) % $(128)) * $(64)) + ((r107_z % $(576)) - $(512))) / $(64)) % $(128)) * $(64)) + (((((r107_z / $(73728)) % $(2)) * $(8192)) + (((r107_z / $(576)) % $(128)) * $(64)) + ((r107_z % $(576)) - $(512))) % $(64))), dtype=tl.bfloat16))
    r107_value = r107_value.to(tl.bfloat16)
    r107_value = tl.broadcast_to(r107_value, [$(147456)])
    tl.store($(vcat_6_ptr) + r107_z, r107_value)
    r108_row = tl.arange($(256))
    r108_col = tl.arange($(512))
    r108_lane = tl.arange($(576))
    r108_left = tl.load($(vcat_6_ptr) + ((((((r108_row[:, None] * $(576)) + r108_lane[None, :]) / $(73728)) % $(2)) * $(73728)) + (((((r108_row[:, None] * $(576)) + r108_lane[None, :]) / $(576)) % $(128)) * $(576)) + (((r108_row[:, None] * $(576)) + r108_lane[None, :]) % $(576))), dtype=tl.bfloat16)
    r108_right = tl.load($(vto_copy_3_ptr) + (((r108_lane[:, None] % $(576)) * $(512)) + (r108_col[None, :] % $(512))), dtype=tl.bfloat16)
    r108_value = tl.dot(r108_left, r108_right)
    r108_value = r108_value.to(tl.bfloat16)
    tl.store($(vmm_7_ptr) + ((r108_row[:, None] * $(512)) + r108_col[None, :]), r108_value)
    r109_row = tl.arange($(256))
    r109_col = tl.arange($(256))
    r109_lane = tl.arange($(1536))
    r109_left = tl.load($(vcat_5_ptr) + ((((((r109_row[:, None] * $(1536)) + r109_lane[None, :]) / $(196608)) % $(2)) * $(196608)) + (((((((((r109_row[:, None] * $(1536)) + r109_lane[None, :]) / $(196608)) % $(2)) * $(196608)) + (((((r109_row[:, None] * $(1536)) + r109_lane[None, :]) / $(1536)) % $(128)) * $(1536)) + (((r109_row[:, None] * $(1536)) + r109_lane[None, :]) % $(1536))) / $(192)) % $(8)) * $(24576)) + (((((r109_row[:, None] * $(1536)) + r109_lane[None, :]) / $(1536)) % $(128)) * $(192)) + (((((((r109_row[:, None] * $(1536)) + r109_lane[None, :]) / $(196608)) % $(2)) * $(196608)) + (((((r109_row[:, None] * $(1536)) + r109_lane[None, :]) / $(1536)) % $(128)) * $(1536)) + (((r109_row[:, None] * $(1536)) + r109_lane[None, :]) % $(1536))) % $(192))), dtype=tl.bfloat16)
    r109_right = tl.load($(vto_copy_2_ptr) + (((r109_lane[:, None] % $(1536)) * $(256)) + (r109_col[None, :] % $(256))), dtype=tl.bfloat16)
    r109_value = tl.dot(r109_left, r109_right)
    r109_value = r109_value.to(tl.bfloat16)
    tl.store($(vmm_8_ptr) + ((r109_row[:, None] * $(256)) + r109_col[None, :]), r109_value)
    r110_z = tl.arange($(65536))
    r110_value = tl.load($(vmm_8_ptr) + ((((r110_z / $(32768)) % $(2)) * $(32768)) + (((r110_z / $(256)) % $(128)) * $(256)) + (r110_z % $(256))), dtype=tl.bfloat16)
    r110_value = r110_value.to(tl.float32)
    r110_value = tl.broadcast_to(r110_value, [$(65536)])
    tl.store($(vto_copy_26_ptr) + r110_z, r110_value)
    r111_z = tl.arange($(65536))
    r111_value = (tl.load($(vto_copy_26_ptr) + ((((r111_z / $(32768)) % $(2)) * $(32768)) + (((r111_z / $(256)) % $(128)) * $(256)) + (r111_z % $(256))), dtype=tl.float32) * tl.cast(tl.load($(vto_copy_1_ptr) + (r111_z % $(256)), dtype=tl.bfloat16), tl.float32))
    r111_value = r111_value.to(tl.float32)
    r111_value = tl.broadcast_to(r111_value, [$(65536)])
    tl.store($(vmul_36_ptr) + r111_z, r111_value)
    r112_z = tl.arange($(65536))
    r112_value = (tl.load($(vmul_36_ptr) + ((((r112_z / $(32768)) % $(2)) * $(32768)) + (((r112_z / $(256)) % $(128)) * $(256)) + (r112_z % $(256))), dtype=tl.float32) * tl.load($(vto_copy_16_ptr) + ((((r112_z / $(32768)) % $(2)) * $(32768)) + (((r112_z / $(256)) % $(128)) * $(256)) + (r112_z % $(256))), dtype=tl.float32))
    r112_value = r112_value.to(tl.float32)
    r112_value = tl.broadcast_to(r112_value, [$(65536)])
    tl.store($(vmul_37_ptr) + r112_z, r112_value)
    r113_z = tl.arange($(65536))
    r113_value = (tl.load($(vmul_36_ptr) + ((((r113_z / $(32768)) % $(2)) * $(32768)) + (((r113_z / $(256)) % $(128)) * $(256)) + (r113_z % $(256))), dtype=tl.float32) * tl.load($(vrsqrt_ptr) + ((((r113_z / $(32768)) % $(2)) * $(128)) + ((r113_z / $(256)) % $(128))), dtype=tl.float32))
    r113_value = r113_value.to(tl.float32)
    r113_value = tl.broadcast_to(r113_value, [$(65536)])
    tl.store($(vmul_38_ptr) + r113_z, r113_value)
    r114_z = tl.arange($(256))
    r114_lane = tl.arange($(256))
    r114_value = tl.sum(tl.load($(vmul_37_ptr) + ((((r114_z[:, None] / $(128)) % $(2)) * $(32768)) + ((r114_z[:, None] % $(128)) * $(256)) + r114_lane[None, :]), dtype=tl.float32), axis=1)
    r114_value = r114_value.to(tl.float32)
    r114_value = tl.broadcast_to(r114_value, [$(256)])
    tl.store($(vsum_3_ptr) + r114_z, r114_value)
    r115_z = tl.arange($(256))
    r115_value = ((tl.load($(vrsqrt_ptr) + ((((r115_z / $(128)) % $(2)) * $(128)) + (r115_z % $(128))), dtype=tl.float32)) * (tl.load($(vrsqrt_ptr) + ((((r115_z / $(128)) % $(2)) * $(128)) + (r115_z % $(128))), dtype=tl.float32)) * (tl.load($(vrsqrt_ptr) + ((((r115_z / $(128)) % $(2)) * $(128)) + (r115_z % $(128))), dtype=tl.float32)))
    r115_value = r115_value.to(tl.float32)
    r115_value = tl.broadcast_to(r115_value, [$(256)])
    tl.store($(vpow_5_ptr) + r115_z, r115_value)
    r116_z = tl.arange($(256))
    r116_value = (tl.load($(vsum_3_ptr) + ((((r116_z / $(128)) % $(2)) * $(128)) + (r116_z % $(128))), dtype=tl.float32) * -0.5)
    r116_value = r116_value.to(tl.float32)
    r116_value = tl.broadcast_to(r116_value, [$(256)])
    tl.store($(vmul_39_ptr) + r116_z, r116_value)
    r117_z = tl.arange($(256))
    r117_value = (tl.load($(vmul_39_ptr) + ((((r117_z / $(128)) % $(2)) * $(128)) + (r117_z % $(128))), dtype=tl.float32) * tl.load($(vpow_5_ptr) + ((((r117_z / $(128)) % $(2)) * $(128)) + (r117_z % $(128))), dtype=tl.float32))
    r117_value = r117_value.to(tl.float32)
    r117_value = tl.broadcast_to(r117_value, [$(256)])
    tl.store($(vmul_40_ptr) + r117_z, r117_value)
    r118_z = tl.arange($(65536))
    r118_value = (tl.load($(vmul_40_ptr) + ((((r118_z / $(32768)) % $(2)) * $(128)) + ((r118_z / $(256)) % $(128))), dtype=tl.float32) / 256.0)
    r118_value = r118_value.to(tl.float32)
    r118_value = tl.broadcast_to(r118_value, [$(65536)])
    tl.store($(vdiv_1_ptr) + r118_z, r118_value)
    r119_z = tl.arange($(65536))
    r119_value = ((tl.load($(vto_copy_16_ptr) + ((((r119_z / $(32768)) % $(2)) * $(32768)) + (((r119_z / $(256)) % $(128)) * $(256)) + (r119_z % $(256))), dtype=tl.float32)))
    r119_value = r119_value.to(tl.float32)
    r119_value = tl.broadcast_to(r119_value, [$(65536)])
    tl.store($(vpow_6_ptr) + r119_z, r119_value)
    r120_z = tl.arange($(65536))
    r120_value = (tl.load($(vpow_6_ptr) + ((((r120_z / $(32768)) % $(2)) * $(32768)) + (((r120_z / $(256)) % $(128)) * $(256)) + (r120_z % $(256))), dtype=tl.float32) * 2.0)
    r120_value = r120_value.to(tl.float32)
    r120_value = tl.broadcast_to(r120_value, [$(65536)])
    tl.store($(vmul_41_ptr) + r120_z, r120_value)
    r121_z = tl.arange($(65536))
    r121_value = (tl.load($(vdiv_1_ptr) + ((((r121_z / $(32768)) % $(2)) * $(32768)) + (((r121_z / $(256)) % $(128)) * $(256)) + (r121_z % $(256))), dtype=tl.float32) * tl.load($(vmul_41_ptr) + ((((r121_z / $(32768)) % $(2)) * $(32768)) + (((r121_z / $(256)) % $(128)) * $(256)) + (r121_z % $(256))), dtype=tl.float32))
    r121_value = r121_value.to(tl.float32)
    r121_value = tl.broadcast_to(r121_value, [$(65536)])
    tl.store($(vmul_42_ptr) + r121_z, r121_value)
    r122_z = tl.arange($(65536))
    r122_value = (tl.load($(vmul_38_ptr) + ((((r122_z / $(32768)) % $(2)) * $(32768)) + (((r122_z / $(256)) % $(128)) * $(256)) + (r122_z % $(256))), dtype=tl.float32) + tl.load($(vmul_42_ptr) + ((((r122_z / $(32768)) % $(2)) * $(32768)) + (((r122_z / $(256)) % $(128)) * $(256)) + (r122_z % $(256))), dtype=tl.float32))
    r122_value = r122_value.to(tl.float32)
    r122_value = tl.broadcast_to(r122_value, [$(65536)])
    tl.store($(vadd_14_ptr) + r122_z, r122_value)
    r123_z = tl.arange($(65536))
    r123_value = tl.load($(vadd_14_ptr) + ((((r123_z / $(32768)) % $(2)) * $(32768)) + (((r123_z / $(256)) % $(128)) * $(256)) + (r123_z % $(256))), dtype=tl.float32)
    r123_value = r123_value.to(tl.bfloat16)
    r123_value = tl.broadcast_to(r123_value, [$(65536)])
    tl.store($(vto_copy_27_ptr) + r123_z, r123_value)
    r124_row = tl.arange($(256))
    r124_col = tl.arange($(512))
    r124_lane = tl.arange($(256))
    r124_left = tl.load($(vto_copy_27_ptr) + ((((((r124_row[:, None] * $(256)) + r124_lane[None, :]) / $(32768)) % $(2)) * $(32768)) + (((((r124_row[:, None] * $(256)) + r124_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r124_row[:, None] * $(256)) + r124_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r124_right = tl.load($(vto_copy_ptr) + (((r124_lane[:, None] % $(256)) * $(512)) + (r124_col[None, :] % $(512))), dtype=tl.bfloat16)
    r124_value = tl.dot(r124_left, r124_right)
    r124_value = r124_value.to(tl.bfloat16)
    tl.store($(vmm_9_ptr) + ((r124_row[:, None] * $(512)) + r124_col[None, :]), r124_value)
    r125_z = tl.arange($(131072))
    r125_value = (tl.load($(vmm_7_ptr) + ((((r125_z / $(65536)) % $(2)) * $(65536)) + (((r125_z / $(512)) % $(128)) * $(512)) + (r125_z % $(512))), dtype=tl.bfloat16) + tl.load($(vmm_9_ptr) + ((((r125_z / $(65536)) % $(2)) * $(65536)) + (((r125_z / $(512)) % $(128)) * $(512)) + (r125_z % $(512))), dtype=tl.bfloat16))
    r125_value = r125_value.to(tl.bfloat16)
    r125_value = tl.broadcast_to(r125_value, [$(131072)])
    tl.store($(vadd_15_ptr) + r125_z, r125_value)
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
  let vmm_ptr : RegionName := "vmm"
  let vto_copy_16_ptr : RegionName := "vto_copy_16"
  let vpow_1_ptr : RegionName := "vpow_1"
  let vmean_ptr : RegionName := "vmean"
  let vadd_2_ptr : RegionName := "vadd_2"
  let vrsqrt_ptr : RegionName := "vrsqrt"
  let vmul_7_ptr : RegionName := "vmul_7"
  let vmul_8_ptr : RegionName := "vmul_8"
  let vto_copy_17_ptr : RegionName := "vto_copy_17"
  let vmm_1_ptr : RegionName := "vmm_1"
  let vmm_2_ptr : RegionName := "vmm_2"
  let vto_copy_18_ptr : RegionName := "vto_copy_18"
  let vpow_2_ptr : RegionName := "vpow_2"
  let vmean_1_ptr : RegionName := "vmean_1"
  let vadd_3_ptr : RegionName := "vadd_3"
  let vrsqrt_1_ptr : RegionName := "vrsqrt_1"
  let vmul_9_ptr : RegionName := "vmul_9"
  let vmul_10_ptr : RegionName := "vmul_10"
  let vto_copy_19_ptr : RegionName := "vto_copy_19"
  let vmul_11_ptr : RegionName := "vmul_11"
  let vmul_12_ptr : RegionName := "vmul_12"
  let vsub_ptr : RegionName := "vsub"
  let vmul_13_ptr : RegionName := "vmul_13"
  let vmul_14_ptr : RegionName := "vmul_14"
  let vadd_4_ptr : RegionName := "vadd_4"
  let vcat_ptr : RegionName := "vcat"
  let vmul_15_ptr : RegionName := "vmul_15"
  let vmul_16_ptr : RegionName := "vmul_16"
  let vsub_1_ptr : RegionName := "vsub_1"
  let vmul_17_ptr : RegionName := "vmul_17"
  let vmul_18_ptr : RegionName := "vmul_18"
  let vadd_5_ptr : RegionName := "vadd_5"
  let vcat_1_ptr : RegionName := "vcat_1"
  let vcat_3_ptr : RegionName := "vcat_3"
  let vbmm_4_ptr : RegionName := "vbmm_4"
  let vbmm_5_ptr : RegionName := "vbmm_5"
  let vsum_1_ptr : RegionName := "vsum_1"
  let vmul_21_ptr : RegionName := "vmul_21"
  let vmul_22_ptr : RegionName := "vmul_22"
  let vneg_ptr : RegionName := "vneg"
  let vmul_23_ptr : RegionName := "vmul_23"
  let vmul_24_ptr : RegionName := "vmul_24"
  let vmul_25_ptr : RegionName := "vmul_25"
  let vmul_26_ptr : RegionName := "vmul_26"
  let vneg_1_ptr : RegionName := "vneg_1"
  let vmul_27_ptr : RegionName := "vmul_27"
  let vmul_28_ptr : RegionName := "vmul_28"
  let vadd_10_ptr : RegionName := "vadd_10"
  let vslice_backward_ptr : RegionName := "vslice_backward"
  let vslice_backward_1_ptr : RegionName := "vslice_backward_1"
  let vadd_11_ptr : RegionName := "vadd_11"
  let vslice_backward_2_ptr : RegionName := "vslice_backward_2"
  let vslice_backward_3_ptr : RegionName := "vslice_backward_3"
  let vadd_12_ptr : RegionName := "vadd_12"
  let vcat_4_ptr : RegionName := "vcat_4"
  let vcat_5_ptr : RegionName := "vcat_5"
  let vmul_29_ptr : RegionName := "vmul_29"
  let vmul_30_ptr : RegionName := "vmul_30"
  let vmul_31_ptr : RegionName := "vmul_31"
  let vsum_2_ptr : RegionName := "vsum_2"
  let vpow_3_ptr : RegionName := "vpow_3"
  let vmul_32_ptr : RegionName := "vmul_32"
  let vmul_33_ptr : RegionName := "vmul_33"
  let vdiv_ptr : RegionName := "vdiv"
  let vpow_4_ptr : RegionName := "vpow_4"
  let vmul_34_ptr : RegionName := "vmul_34"
  let vmul_35_ptr : RegionName := "vmul_35"
  let vadd_13_ptr : RegionName := "vadd_13"
  let vcat_6_ptr : RegionName := "vcat_6"
  let vmm_8_ptr : RegionName := "vmm_8"
  let vmul_36_ptr : RegionName := "vmul_36"
  let vmul_37_ptr : RegionName := "vmul_37"
  let vmul_38_ptr : RegionName := "vmul_38"
  let vsum_3_ptr : RegionName := "vsum_3"
  let vpow_5_ptr : RegionName := "vpow_5"
  let vmul_39_ptr : RegionName := "vmul_39"
  let vmul_40_ptr : RegionName := "vmul_40"
  let vdiv_1_ptr : RegionName := "vdiv_1"
  let vpow_6_ptr : RegionName := "vpow_6"
  let vmul_41_ptr : RegionName := "vmul_41"
  let vmul_42_ptr : RegionName := "vmul_42"
  let vadd_14_ptr : RegionName := "vadd_14"
  let vmm_9_ptr : RegionName := "vmm_9"
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
  let candidate_vmm_ptr : RegionName := "candidate_vmm"
  let candidate_vto_copy_16_ptr : RegionName := "candidate_vto_copy_16"
  let candidate_vpow_1_ptr : RegionName := "candidate_vpow_1"
  let candidate_vmean_ptr : RegionName := "candidate_vmean"
  let candidate_vadd_2_ptr : RegionName := "candidate_vadd_2"
  let candidate_vrsqrt_ptr : RegionName := "candidate_vrsqrt"
  let candidate_vmul_7_ptr : RegionName := "candidate_vmul_7"
  let candidate_vmul_8_ptr : RegionName := "candidate_vmul_8"
  let candidate_vto_copy_17_ptr : RegionName := "candidate_vto_copy_17"
  let candidate_vmm_1_ptr : RegionName := "candidate_vmm_1"
  let candidate_vmm_2_ptr : RegionName := "candidate_vmm_2"
  let candidate_vto_copy_18_ptr : RegionName := "candidate_vto_copy_18"
  let candidate_vpow_2_ptr : RegionName := "candidate_vpow_2"
  let candidate_vmean_1_ptr : RegionName := "candidate_vmean_1"
  let candidate_vadd_3_ptr : RegionName := "candidate_vadd_3"
  let candidate_vrsqrt_1_ptr : RegionName := "candidate_vrsqrt_1"
  let candidate_vmul_9_ptr : RegionName := "candidate_vmul_9"
  let candidate_vmul_10_ptr : RegionName := "candidate_vmul_10"
  let candidate_vto_copy_19_ptr : RegionName := "candidate_vto_copy_19"
  let candidate_vmul_11_ptr : RegionName := "candidate_vmul_11"
  let candidate_vmul_12_ptr : RegionName := "candidate_vmul_12"
  let candidate_vsub_ptr : RegionName := "candidate_vsub"
  let candidate_vmul_13_ptr : RegionName := "candidate_vmul_13"
  let candidate_vmul_14_ptr : RegionName := "candidate_vmul_14"
  let candidate_vadd_4_ptr : RegionName := "candidate_vadd_4"
  let candidate_vcat_ptr : RegionName := "candidate_vcat"
  let candidate_vmul_15_ptr : RegionName := "candidate_vmul_15"
  let candidate_vmul_16_ptr : RegionName := "candidate_vmul_16"
  let candidate_vsub_1_ptr : RegionName := "candidate_vsub_1"
  let candidate_vmul_17_ptr : RegionName := "candidate_vmul_17"
  let candidate_vmul_18_ptr : RegionName := "candidate_vmul_18"
  let candidate_vadd_5_ptr : RegionName := "candidate_vadd_5"
  let candidate_vcat_1_ptr : RegionName := "candidate_vcat_1"
  let candidate_vmm_8_ptr : RegionName := "candidate_vmm_8"
  let candidate_vbmm_4_ptr : RegionName := "candidate_vbmm_4"
  let candidate_vbmm_5_ptr : RegionName := "candidate_vbmm_5"
  let candidate_vcat_3_ptr : RegionName := "candidate_vcat_3"
  let candidate_vcat_4_ptr : RegionName := "candidate_vcat_4"
  let candidate_vto_copy_30_ptr : RegionName := "candidate_vto_copy_30"
  let candidate_vto_copy_31_ptr : RegionName := "candidate_vto_copy_31"
  let candidate_vbmm_6_ptr : RegionName := "candidate_vbmm_6"
  let candidate_vmul_21_ptr : RegionName := "candidate_vmul_21"
  let candidate_vadd_10_ptr : RegionName := "candidate_vadd_10"
  let candidate_vsub_4_ptr : RegionName := "candidate_vsub_4"
  let candidate_vgt_2_ptr : RegionName := "candidate_vgt_2"
  let candidate_vmasked_fill_4_ptr : RegionName := "candidate_vmasked_fill_4"
  let candidate_vsoftmax_2_ptr : RegionName := "candidate_vsoftmax_2"
  let candidate_vto_copy_32_ptr : RegionName := "candidate_vto_copy_32"
  let candidate_vbmm_7_ptr : RegionName := "candidate_vbmm_7"
  let candidate_vlogsumexp_2_ptr : RegionName := "candidate_vlogsumexp_2"
  let candidate_vto_copy_33_ptr : RegionName := "candidate_vto_copy_33"
  let candidate_vto_copy_34_ptr : RegionName := "candidate_vto_copy_34"
  let candidate_vbmm_8_ptr : RegionName := "candidate_vbmm_8"
  let candidate_vmul_22_ptr : RegionName := "candidate_vmul_22"
  let candidate_vadd_11_ptr : RegionName := "candidate_vadd_11"
  let candidate_vsub_5_ptr : RegionName := "candidate_vsub_5"
  let candidate_vgt_3_ptr : RegionName := "candidate_vgt_3"
  let candidate_vmasked_fill_5_ptr : RegionName := "candidate_vmasked_fill_5"
  let candidate_vsoftmax_3_ptr : RegionName := "candidate_vsoftmax_3"
  let candidate_vto_copy_35_ptr : RegionName := "candidate_vto_copy_35"
  let candidate_vbmm_9_ptr : RegionName := "candidate_vbmm_9"
  let candidate_vlogsumexp_3_ptr : RegionName := "candidate_vlogsumexp_3"
  let candidate_vcat_5_ptr : RegionName := "candidate_vcat_5"
  let candidate_vcat_6_ptr : RegionName := "candidate_vcat_6"
  let candidate_vto_copy_36_ptr : RegionName := "candidate_vto_copy_36"
  let candidate_vto_copy_37_ptr : RegionName := "candidate_vto_copy_37"
  let candidate_vto_copy_38_ptr : RegionName := "candidate_vto_copy_38"
  let candidate_vto_copy_39_ptr : RegionName := "candidate_vto_copy_39"
  let candidate_vto_copy_40_ptr : RegionName := "candidate_vto_copy_40"
  let candidate_vbmm_10_ptr : RegionName := "candidate_vbmm_10"
  let candidate_vmul_23_ptr : RegionName := "candidate_vmul_23"
  let candidate_vadd_12_ptr : RegionName := "candidate_vadd_12"
  let candidate_vsub_6_ptr : RegionName := "candidate_vsub_6"
  let candidate_vgt_4_ptr : RegionName := "candidate_vgt_4"
  let candidate_vmasked_fill_6_ptr : RegionName := "candidate_vmasked_fill_6"
  let candidate_vsub_7_ptr : RegionName := "candidate_vsub_7"
  let candidate_vexp_ptr : RegionName := "candidate_vexp"
  let candidate_vmul_24_ptr : RegionName := "candidate_vmul_24"
  let candidate_vsum_1_ptr : RegionName := "candidate_vsum_1"
  let candidate_vbmm_11_ptr : RegionName := "candidate_vbmm_11"
  let candidate_vsub_8_ptr : RegionName := "candidate_vsub_8"
  let candidate_vmul_25_ptr : RegionName := "candidate_vmul_25"
  let candidate_vbmm_12_ptr : RegionName := "candidate_vbmm_12"
  let candidate_vmul_26_ptr : RegionName := "candidate_vmul_26"
  let candidate_vto_copy_41_ptr : RegionName := "candidate_vto_copy_41"
  let candidate_vbmm_13_ptr : RegionName := "candidate_vbmm_13"
  let candidate_vmul_27_ptr : RegionName := "candidate_vmul_27"
  let candidate_vto_copy_42_ptr : RegionName := "candidate_vto_copy_42"
  let candidate_vbmm_14_ptr : RegionName := "candidate_vbmm_14"
  let candidate_vto_copy_43_ptr : RegionName := "candidate_vto_copy_43"
  let candidate_vto_copy_44_ptr : RegionName := "candidate_vto_copy_44"
  let candidate_vto_copy_45_ptr : RegionName := "candidate_vto_copy_45"
  let candidate_vto_copy_46_ptr : RegionName := "candidate_vto_copy_46"
  let candidate_vto_copy_47_ptr : RegionName := "candidate_vto_copy_47"
  let candidate_vto_copy_48_ptr : RegionName := "candidate_vto_copy_48"
  let candidate_vbmm_15_ptr : RegionName := "candidate_vbmm_15"
  let candidate_vmul_28_ptr : RegionName := "candidate_vmul_28"
  let candidate_vadd_13_ptr : RegionName := "candidate_vadd_13"
  let candidate_vsub_9_ptr : RegionName := "candidate_vsub_9"
  let candidate_vgt_5_ptr : RegionName := "candidate_vgt_5"
  let candidate_vmasked_fill_7_ptr : RegionName := "candidate_vmasked_fill_7"
  let candidate_vsub_10_ptr : RegionName := "candidate_vsub_10"
  let candidate_vexp_1_ptr : RegionName := "candidate_vexp_1"
  let candidate_vmul_29_ptr : RegionName := "candidate_vmul_29"
  let candidate_vsum_2_ptr : RegionName := "candidate_vsum_2"
  let candidate_vbmm_16_ptr : RegionName := "candidate_vbmm_16"
  let candidate_vsub_11_ptr : RegionName := "candidate_vsub_11"
  let candidate_vmul_30_ptr : RegionName := "candidate_vmul_30"
  let candidate_vbmm_17_ptr : RegionName := "candidate_vbmm_17"
  let candidate_vmul_31_ptr : RegionName := "candidate_vmul_31"
  let candidate_vto_copy_49_ptr : RegionName := "candidate_vto_copy_49"
  let candidate_vbmm_18_ptr : RegionName := "candidate_vbmm_18"
  let candidate_vmul_32_ptr : RegionName := "candidate_vmul_32"
  let candidate_vto_copy_50_ptr : RegionName := "candidate_vto_copy_50"
  let candidate_vbmm_19_ptr : RegionName := "candidate_vbmm_19"
  let candidate_vto_copy_51_ptr : RegionName := "candidate_vto_copy_51"
  let candidate_vcat_7_ptr : RegionName := "candidate_vcat_7"
  let candidate_vcat_8_ptr : RegionName := "candidate_vcat_8"
  let candidate_vcat_9_ptr : RegionName := "candidate_vcat_9"
  let candidate_vsum_3_ptr : RegionName := "candidate_vsum_3"
  let candidate_vbmm_20_ptr : RegionName := "candidate_vbmm_20"
  let candidate_vbmm_21_ptr : RegionName := "candidate_vbmm_21"
  let candidate_vadd_14_ptr : RegionName := "candidate_vadd_14"
  let candidate_vmul_33_ptr : RegionName := "candidate_vmul_33"
  let candidate_vmul_34_ptr : RegionName := "candidate_vmul_34"
  let candidate_vneg_ptr : RegionName := "candidate_vneg"
  let candidate_vmul_35_ptr : RegionName := "candidate_vmul_35"
  let candidate_vadd_15_ptr : RegionName := "candidate_vadd_15"
  let candidate_vmul_36_ptr : RegionName := "candidate_vmul_36"
  let candidate_vadd_16_ptr : RegionName := "candidate_vadd_16"
  let candidate_vmul_37_ptr : RegionName := "candidate_vmul_37"
  let candidate_vmul_38_ptr : RegionName := "candidate_vmul_38"
  let candidate_vneg_1_ptr : RegionName := "candidate_vneg_1"
  let candidate_vmul_39_ptr : RegionName := "candidate_vmul_39"
  let candidate_vadd_17_ptr : RegionName := "candidate_vadd_17"
  let candidate_vmul_40_ptr : RegionName := "candidate_vmul_40"
  let candidate_vadd_18_ptr : RegionName := "candidate_vadd_18"
  let candidate_vslice_backward_ptr : RegionName := "candidate_vslice_backward"
  let candidate_vslice_backward_1_ptr : RegionName := "candidate_vslice_backward_1"
  let candidate_vadd_19_ptr : RegionName := "candidate_vadd_19"
  let candidate_vslice_backward_2_ptr : RegionName := "candidate_vslice_backward_2"
  let candidate_vslice_backward_3_ptr : RegionName := "candidate_vslice_backward_3"
  let candidate_vadd_20_ptr : RegionName := "candidate_vadd_20"
  let candidate_vcat_11_ptr : RegionName := "candidate_vcat_11"
  let candidate_vto_copy_52_ptr : RegionName := "candidate_vto_copy_52"
  let candidate_vmul_41_ptr : RegionName := "candidate_vmul_41"
  let candidate_vmul_42_ptr : RegionName := "candidate_vmul_42"
  let candidate_vmul_43_ptr : RegionName := "candidate_vmul_43"
  let candidate_vsum_4_ptr : RegionName := "candidate_vsum_4"
  let candidate_vpow_3_ptr : RegionName := "candidate_vpow_3"
  let candidate_vmul_44_ptr : RegionName := "candidate_vmul_44"
  let candidate_vmul_45_ptr : RegionName := "candidate_vmul_45"
  let candidate_vdiv_ptr : RegionName := "candidate_vdiv"
  let candidate_vpow_4_ptr : RegionName := "candidate_vpow_4"
  let candidate_vmul_46_ptr : RegionName := "candidate_vmul_46"
  let candidate_vmul_47_ptr : RegionName := "candidate_vmul_47"
  let candidate_vadd_21_ptr : RegionName := "candidate_vadd_21"
  let candidate_vto_copy_53_ptr : RegionName := "candidate_vto_copy_53"
  let candidate_vcat_12_ptr : RegionName := "candidate_vcat_12"
  let candidate_vmm_9_ptr : RegionName := "candidate_vmm_9"
  let candidate_vmm_10_ptr : RegionName := "candidate_vmm_10"
  let candidate_vto_copy_54_ptr : RegionName := "candidate_vto_copy_54"
  let candidate_vmul_48_ptr : RegionName := "candidate_vmul_48"
  let candidate_vmul_49_ptr : RegionName := "candidate_vmul_49"
  let candidate_vmul_50_ptr : RegionName := "candidate_vmul_50"
  let candidate_vsum_5_ptr : RegionName := "candidate_vsum_5"
  let candidate_vpow_5_ptr : RegionName := "candidate_vpow_5"
  let candidate_vmul_51_ptr : RegionName := "candidate_vmul_51"
  let candidate_vmul_52_ptr : RegionName := "candidate_vmul_52"
  let candidate_vdiv_1_ptr : RegionName := "candidate_vdiv_1"
  let candidate_vpow_6_ptr : RegionName := "candidate_vpow_6"
  let candidate_vmul_53_ptr : RegionName := "candidate_vmul_53"
  let candidate_vmul_54_ptr : RegionName := "candidate_vmul_54"
  let candidate_vadd_22_ptr : RegionName := "candidate_vadd_22"
  let candidate_vto_copy_55_ptr : RegionName := "candidate_vto_copy_55"
  let candidate_vmm_11_ptr : RegionName := "candidate_vmm_11"
  let candidate_vadd_23_ptr : RegionName := "candidate_vadd_23"
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
    r4_z = tl.arange($(294912))
    r4_value = (tl.load($(parameter3_ptr) + ((((r4_z / $(512)) % $(576)) * $(512)) + (r4_z % $(512))), dtype=tl.float32) * 0.04419417382415922)
    r4_value = r4_value.to(tl.float32)
    r4_value = tl.broadcast_to(r4_value, [$(294912)])
    tl.store($(candidate_vmul_3_ptr) + r4_z, r4_value)
    r5_z = tl.arange($(512))
    r5_value = (tl.load($(parameter4_ptr) + r5_z, dtype=tl.float32) * 0.1)
    r5_value = r5_value.to(tl.float32)
    r5_value = tl.broadcast_to(r5_value, [$(512)])
    tl.store($(candidate_vmul_4_ptr) + r5_z, r5_value)
    r6_z = tl.arange($(512))
    r6_value = (tl.load($(candidate_vmul_4_ptr) + r6_z, dtype=tl.float32) + 1.0)
    r6_value = r6_value.to(tl.float32)
    r6_value = tl.broadcast_to(r6_value, [$(512)])
    tl.store($(candidate_vadd_1_ptr) + r6_z, r6_value)
    r7_z = tl.arange($(1048576))
    r7_value = (tl.load($(parameter5_ptr) + ((((r7_z / $(512)) % $(2048)) * $(512)) + (r7_z % $(512))), dtype=tl.float32) * 0.04419417382415922)
    r7_value = r7_value.to(tl.float32)
    r7_value = tl.broadcast_to(r7_value, [$(1048576)])
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
    r12_z = tl.arange($(294912))
    r12_value = tl.load($(candidate_vmul_3_ptr) + ((((r12_z / $(512)) % $(576)) * $(512)) + (r12_z % $(512))), dtype=tl.float32)
    r12_value = r12_value.to(tl.bfloat16)
    r12_value = tl.broadcast_to(r12_value, [$(294912)])
    tl.store($(candidate_vto_copy_3_ptr) + r12_z, r12_value)
    r13_z = tl.arange($(512))
    r13_value = tl.load($(candidate_vadd_1_ptr) + r13_z, dtype=tl.float32)
    r13_value = r13_value.to(tl.bfloat16)
    r13_value = tl.broadcast_to(r13_value, [$(512)])
    tl.store($(candidate_vto_copy_4_ptr) + r13_z, r13_value)
    r14_z = tl.arange($(1048576))
    r14_value = tl.load($(candidate_vmul_5_ptr) + ((((r14_z / $(512)) % $(2048)) * $(512)) + (r14_z % $(512))), dtype=tl.float32)
    r14_value = r14_value.to(tl.bfloat16)
    r14_value = tl.broadcast_to(r14_value, [$(1048576)])
    tl.store($(candidate_vto_copy_5_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(524288))
    r15_value = tl.load($(candidate_vmul_6_ptr) + ((((r15_z / $(1024)) % $(512)) * $(1024)) + (r15_z % $(1024))), dtype=tl.float32)
    r15_value = r15_value.to(tl.bfloat16)
    r15_value = tl.broadcast_to(r15_value, [$(524288)])
    tl.store($(candidate_vto_copy_6_ptr) + r15_z, r15_value)
    r16_row = tl.arange($(256))
    r16_col = tl.arange($(256))
    r16_lane = tl.arange($(512))
    r16_left = tl.load($(input0_ptr) + ((((((r16_row[:, None] * $(512)) + r16_lane[None, :]) / $(65536)) % $(2)) * $(65536)) + (((((r16_row[:, None] * $(512)) + r16_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r16_row[:, None] * $(512)) + r16_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r16_right = tl.load($(candidate_vto_copy_ptr) + (((r16_col[None, :] % $(256)) * $(512)) + (r16_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r16_value = tl.dot(r16_left, r16_right)
    r16_value = r16_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_ptr) + ((r16_row[:, None] * $(256)) + r16_col[None, :]), r16_value)
    r17_z = tl.arange($(65536))
    r17_value = tl.load($(candidate_vmm_ptr) + ((((r17_z / $(32768)) % $(2)) * $(32768)) + (((r17_z / $(256)) % $(128)) * $(256)) + (r17_z % $(256))), dtype=tl.bfloat16)
    r17_value = r17_value.to(tl.float32)
    r17_value = tl.broadcast_to(r17_value, [$(65536)])
    tl.store($(candidate_vto_copy_16_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(65536))
    r18_value = ((tl.load($(candidate_vto_copy_16_ptr) + ((((r18_z / $(32768)) % $(2)) * $(32768)) + (((r18_z / $(256)) % $(128)) * $(256)) + (r18_z % $(256))), dtype=tl.float32)) * (tl.load($(candidate_vto_copy_16_ptr) + ((((r18_z / $(32768)) % $(2)) * $(32768)) + (((r18_z / $(256)) % $(128)) * $(256)) + (r18_z % $(256))), dtype=tl.float32)))
    r18_value = r18_value.to(tl.float32)
    r18_value = tl.broadcast_to(r18_value, [$(65536)])
    tl.store($(candidate_vpow_1_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(256))
    r19_lane = tl.arange($(256))
    r19_value = (tl.sum(tl.load($(candidate_vpow_1_ptr) + ((((r19_z[:, None] / $(128)) % $(2)) * $(32768)) + ((r19_z[:, None] % $(128)) * $(256)) + r19_lane[None, :]), dtype=tl.float32), axis=1) / 256.0)
    r19_value = r19_value.to(tl.float32)
    r19_value = tl.broadcast_to(r19_value, [$(256)])
    tl.store($(candidate_vmean_ptr) + r19_z, r19_value)
    r20_z = tl.arange($(256))
    r20_value = (tl.load($(candidate_vmean_ptr) + ((((r20_z / $(128)) % $(2)) * $(128)) + (r20_z % $(128))), dtype=tl.float32) + 0.000001)
    r20_value = r20_value.to(tl.float32)
    r20_value = tl.broadcast_to(r20_value, [$(256)])
    tl.store($(candidate_vadd_2_ptr) + r20_z, r20_value)
    r21_z = tl.arange($(256))
    r21_value = tl.rsqrt(tl.load($(candidate_vadd_2_ptr) + ((((r21_z / $(128)) % $(2)) * $(128)) + (r21_z % $(128))), dtype=tl.float32))
    r21_value = r21_value.to(tl.float32)
    r21_value = tl.broadcast_to(r21_value, [$(256)])
    tl.store($(candidate_vrsqrt_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(65536))
    r22_value = (tl.load($(candidate_vto_copy_16_ptr) + ((((r22_z / $(32768)) % $(2)) * $(32768)) + (((r22_z / $(256)) % $(128)) * $(256)) + (r22_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_ptr) + ((((r22_z / $(32768)) % $(2)) * $(128)) + ((r22_z / $(256)) % $(128))), dtype=tl.float32))
    r22_value = r22_value.to(tl.float32)
    r22_value = tl.broadcast_to(r22_value, [$(65536)])
    tl.store($(candidate_vmul_7_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(65536))
    r23_value = (tl.cast(tl.load($(candidate_vto_copy_1_ptr) + (r23_z % $(256)), dtype=tl.bfloat16), tl.float32) * tl.load($(candidate_vmul_7_ptr) + ((((r23_z / $(32768)) % $(2)) * $(32768)) + (((r23_z / $(256)) % $(128)) * $(256)) + (r23_z % $(256))), dtype=tl.float32))
    r23_value = r23_value.to(tl.float32)
    r23_value = tl.broadcast_to(r23_value, [$(65536)])
    tl.store($(candidate_vmul_8_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(65536))
    r24_value = tl.load($(candidate_vmul_8_ptr) + ((((r24_z / $(32768)) % $(2)) * $(32768)) + (((r24_z / $(256)) % $(128)) * $(256)) + (r24_z % $(256))), dtype=tl.float32)
    r24_value = r24_value.to(tl.bfloat16)
    r24_value = tl.broadcast_to(r24_value, [$(65536)])
    tl.store($(candidate_vto_copy_17_ptr) + r24_z, r24_value)
    r25_row = tl.arange($(256))
    r25_col = tl.arange($(1536))
    r25_lane = tl.arange($(256))
    r25_left = tl.load($(candidate_vto_copy_17_ptr) + ((((((r25_row[:, None] * $(256)) + r25_lane[None, :]) / $(32768)) % $(2)) * $(32768)) + (((((r25_row[:, None] * $(256)) + r25_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r25_row[:, None] * $(256)) + r25_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r25_right = tl.load($(candidate_vto_copy_2_ptr) + (((r25_col[None, :] % $(1536)) * $(256)) + (r25_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r25_value = tl.dot(r25_left, r25_right)
    r25_value = r25_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_1_ptr) + ((r25_row[:, None] * $(1536)) + r25_col[None, :]), r25_value)
    r26_row = tl.arange($(256))
    r26_col = tl.arange($(576))
    r26_lane = tl.arange($(512))
    r26_left = tl.load($(input0_ptr) + ((((((r26_row[:, None] * $(512)) + r26_lane[None, :]) / $(65536)) % $(2)) * $(65536)) + (((((r26_row[:, None] * $(512)) + r26_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r26_row[:, None] * $(512)) + r26_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r26_right = tl.load($(candidate_vto_copy_3_ptr) + (((r26_col[None, :] % $(576)) * $(512)) + (r26_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r26_value = tl.dot(r26_left, r26_right)
    r26_value = r26_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_2_ptr) + ((r26_row[:, None] * $(576)) + r26_col[None, :]), r26_value)
    r27_z = tl.arange($(131072))
    r27_value = tl.load($(candidate_vmm_2_ptr) + ((((r27_z / $(65536)) % $(2)) * $(73728)) + (((r27_z / $(512)) % $(128)) * $(576)) + (r27_z % $(512))), dtype=tl.bfloat16)
    r27_value = r27_value.to(tl.float32)
    r27_value = tl.broadcast_to(r27_value, [$(131072)])
    tl.store($(candidate_vto_copy_18_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(131072))
    r28_value = ((tl.load($(candidate_vto_copy_18_ptr) + ((((r28_z / $(65536)) % $(2)) * $(65536)) + (((r28_z / $(512)) % $(128)) * $(512)) + (r28_z % $(512))), dtype=tl.float32)) * (tl.load($(candidate_vto_copy_18_ptr) + ((((r28_z / $(65536)) % $(2)) * $(65536)) + (((r28_z / $(512)) % $(128)) * $(512)) + (r28_z % $(512))), dtype=tl.float32)))
    r28_value = r28_value.to(tl.float32)
    r28_value = tl.broadcast_to(r28_value, [$(131072)])
    tl.store($(candidate_vpow_2_ptr) + r28_z, r28_value)
    r29_z = tl.arange($(256))
    r29_lane = tl.arange($(512))
    r29_value = (tl.sum(tl.load($(candidate_vpow_2_ptr) + ((((r29_z[:, None] / $(128)) % $(2)) * $(65536)) + ((r29_z[:, None] % $(128)) * $(512)) + r29_lane[None, :]), dtype=tl.float32), axis=1) / 512.0)
    r29_value = r29_value.to(tl.float32)
    r29_value = tl.broadcast_to(r29_value, [$(256)])
    tl.store($(candidate_vmean_1_ptr) + r29_z, r29_value)
    r30_z = tl.arange($(256))
    r30_value = (tl.load($(candidate_vmean_1_ptr) + ((((r30_z / $(128)) % $(2)) * $(128)) + (r30_z % $(128))), dtype=tl.float32) + 0.000001)
    r30_value = r30_value.to(tl.float32)
    r30_value = tl.broadcast_to(r30_value, [$(256)])
    tl.store($(candidate_vadd_3_ptr) + r30_z, r30_value)
    r31_z = tl.arange($(256))
    r31_value = tl.rsqrt(tl.load($(candidate_vadd_3_ptr) + ((((r31_z / $(128)) % $(2)) * $(128)) + (r31_z % $(128))), dtype=tl.float32))
    r31_value = r31_value.to(tl.float32)
    r31_value = tl.broadcast_to(r31_value, [$(256)])
    tl.store($(candidate_vrsqrt_1_ptr) + r31_z, r31_value)
    r32_z = tl.arange($(131072))
    r32_value = (tl.load($(candidate_vto_copy_18_ptr) + ((((r32_z / $(65536)) % $(2)) * $(65536)) + (((r32_z / $(512)) % $(128)) * $(512)) + (r32_z % $(512))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_1_ptr) + ((((r32_z / $(65536)) % $(2)) * $(128)) + ((r32_z / $(512)) % $(128))), dtype=tl.float32))
    r32_value = r32_value.to(tl.float32)
    r32_value = tl.broadcast_to(r32_value, [$(131072)])
    tl.store($(candidate_vmul_9_ptr) + r32_z, r32_value)
    r33_z = tl.arange($(131072))
    r33_value = (tl.cast(tl.load($(candidate_vto_copy_4_ptr) + (r33_z % $(512)), dtype=tl.bfloat16), tl.float32) * tl.load($(candidate_vmul_9_ptr) + ((((r33_z / $(65536)) % $(2)) * $(65536)) + (((r33_z / $(512)) % $(128)) * $(512)) + (r33_z % $(512))), dtype=tl.float32))
    r33_value = r33_value.to(tl.float32)
    r33_value = tl.broadcast_to(r33_value, [$(131072)])
    tl.store($(candidate_vmul_10_ptr) + r33_z, r33_value)
    r34_z = tl.arange($(131072))
    r34_value = tl.load($(candidate_vmul_10_ptr) + ((((r34_z / $(65536)) % $(2)) * $(65536)) + (((r34_z / $(512)) % $(128)) * $(512)) + (r34_z % $(512))), dtype=tl.float32)
    r34_value = r34_value.to(tl.bfloat16)
    r34_value = tl.broadcast_to(r34_value, [$(131072)])
    tl.store($(candidate_vto_copy_19_ptr) + r34_z, r34_value)
    r35_z = tl.arange($(65536))
    r35_value = (tl.load($(candidate_vmm_1_ptr) + ((((r35_z / $(32768)) % $(2)) * $(196608)) + (((r35_z / $(32)) % $(128)) * $(1536)) + (((r35_z / $(4096)) % $(8)) * $(192)) + ((r35_z % $(32)) * $(2)) + $(128)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r35_value = r35_value.to(tl.bfloat16)
    r35_value = tl.broadcast_to(r35_value, [$(65536)])
    tl.store($(candidate_vmul_11_ptr) + r35_z, r35_value)
    r36_z = tl.arange($(65536))
    r36_value = (tl.load($(candidate_vmm_1_ptr) + ((((r36_z / $(32768)) % $(2)) * $(196608)) + (((r36_z / $(32)) % $(128)) * $(1536)) + (((r36_z / $(4096)) % $(8)) * $(192)) + ((r36_z % $(32)) * $(2)) + $(129)), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r36_value = r36_value.to(tl.bfloat16)
    r36_value = tl.broadcast_to(r36_value, [$(65536)])
    tl.store($(candidate_vmul_12_ptr) + r36_z, r36_value)
    r37_z = tl.arange($(65536))
    r37_value = (tl.load($(candidate_vmul_11_ptr) + ((((r37_z / $(32768)) % $(2)) * $(32768)) + (((r37_z / $(4096)) % $(8)) * $(4096)) + (((r37_z / $(32)) % $(128)) * $(32)) + (r37_z % $(32))), dtype=tl.bfloat16) - tl.load($(candidate_vmul_12_ptr) + ((((r37_z / $(32768)) % $(2)) * $(32768)) + (((r37_z / $(4096)) % $(8)) * $(4096)) + (((r37_z / $(32)) % $(128)) * $(32)) + (r37_z % $(32))), dtype=tl.bfloat16))
    r37_value = r37_value.to(tl.bfloat16)
    r37_value = tl.broadcast_to(r37_value, [$(65536)])
    tl.store($(candidate_vsub_ptr) + r37_z, r37_value)
    r38_z = tl.arange($(65536))
    r38_value = (tl.load($(candidate_vmm_1_ptr) + ((((r38_z / $(32768)) % $(2)) * $(196608)) + (((r38_z / $(32)) % $(128)) * $(1536)) + (((r38_z / $(4096)) % $(8)) * $(192)) + ((r38_z % $(32)) * $(2)) + $(129)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r38_value = r38_value.to(tl.bfloat16)
    r38_value = tl.broadcast_to(r38_value, [$(65536)])
    tl.store($(candidate_vmul_13_ptr) + r38_z, r38_value)
    r39_z = tl.arange($(65536))
    r39_value = (tl.load($(candidate_vmm_1_ptr) + ((((r39_z / $(32768)) % $(2)) * $(196608)) + (((r39_z / $(32)) % $(128)) * $(1536)) + (((r39_z / $(4096)) % $(8)) * $(192)) + ((r39_z % $(32)) * $(2)) + $(128)), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r39_value = r39_value.to(tl.bfloat16)
    r39_value = tl.broadcast_to(r39_value, [$(65536)])
    tl.store($(candidate_vmul_14_ptr) + r39_z, r39_value)
    r40_z = tl.arange($(65536))
    r40_value = (tl.load($(candidate_vmul_13_ptr) + ((((r40_z / $(32768)) % $(2)) * $(32768)) + (((r40_z / $(4096)) % $(8)) * $(4096)) + (((r40_z / $(32)) % $(128)) * $(32)) + (r40_z % $(32))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_14_ptr) + ((((r40_z / $(32768)) % $(2)) * $(32768)) + (((r40_z / $(4096)) % $(8)) * $(4096)) + (((r40_z / $(32)) % $(128)) * $(32)) + (r40_z % $(32))), dtype=tl.bfloat16))
    r40_value = r40_value.to(tl.bfloat16)
    r40_value = tl.broadcast_to(r40_value, [$(65536)])
    tl.store($(candidate_vadd_4_ptr) + r40_z, r40_value)
    r41_z = tl.arange($(131072))
    r41_value = tl.where((r41_z % $(64)) < $(32), tl.load($(candidate_vsub_ptr) + ((((r41_z / $(65536)) % $(2)) * $(32768)) + (((r41_z / $(8192)) % $(8)) * $(4096)) + (((r41_z / $(64)) % $(128)) * $(32)) + (((r41_z % $(64)) - $(0)) % $(32))), dtype=tl.bfloat16), tl.load($(candidate_vadd_4_ptr) + ((((r41_z / $(65536)) % $(2)) * $(32768)) + (((r41_z / $(8192)) % $(8)) * $(4096)) + (((r41_z / $(64)) % $(128)) * $(32)) + (((r41_z % $(64)) - $(32)) % $(32))), dtype=tl.bfloat16))
    r41_value = r41_value.to(tl.bfloat16)
    r41_value = tl.broadcast_to(r41_value, [$(131072)])
    tl.store($(candidate_vcat_ptr) + r41_z, r41_value)
    r42_z = tl.arange($(8192))
    r42_value = (tl.load($(candidate_vmm_2_ptr) + ((((r42_z / $(4096)) % $(2)) * $(73728)) + (((r42_z / $(32)) % $(128)) * $(576)) + ((r42_z % $(32)) * $(2)) + $(512)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r42_value = r42_value.to(tl.bfloat16)
    r42_value = tl.broadcast_to(r42_value, [$(8192)])
    tl.store($(candidate_vmul_15_ptr) + r42_z, r42_value)
    r43_z = tl.arange($(8192))
    r43_value = (tl.load($(candidate_vmm_2_ptr) + ((((r43_z / $(4096)) % $(2)) * $(73728)) + (((r43_z / $(32)) % $(128)) * $(576)) + ((r43_z % $(32)) * $(2)) + $(513)), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r43_value = r43_value.to(tl.bfloat16)
    r43_value = tl.broadcast_to(r43_value, [$(8192)])
    tl.store($(candidate_vmul_16_ptr) + r43_z, r43_value)
    r44_z = tl.arange($(8192))
    r44_value = (tl.load($(candidate_vmul_15_ptr) + ((((r44_z / $(4096)) % $(2)) * $(4096)) + (((r44_z / $(32)) % $(128)) * $(32)) + (r44_z % $(32))), dtype=tl.bfloat16) - tl.load($(candidate_vmul_16_ptr) + ((((r44_z / $(4096)) % $(2)) * $(4096)) + (((r44_z / $(32)) % $(128)) * $(32)) + (r44_z % $(32))), dtype=tl.bfloat16))
    r44_value = r44_value.to(tl.bfloat16)
    r44_value = tl.broadcast_to(r44_value, [$(8192)])
    tl.store($(candidate_vsub_1_ptr) + r44_z, r44_value)
    r45_z = tl.arange($(8192))
    r45_value = (tl.load($(candidate_vmm_2_ptr) + ((((r45_z / $(4096)) % $(2)) * $(73728)) + (((r45_z / $(32)) % $(128)) * $(576)) + ((r45_z % $(32)) * $(2)) + $(513)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r45_value = r45_value.to(tl.bfloat16)
    r45_value = tl.broadcast_to(r45_value, [$(8192)])
    tl.store($(candidate_vmul_17_ptr) + r45_z, r45_value)
    r46_z = tl.arange($(8192))
    r46_value = (tl.load($(candidate_vmm_2_ptr) + ((((r46_z / $(4096)) % $(2)) * $(73728)) + (((r46_z / $(32)) % $(128)) * $(576)) + ((r46_z % $(32)) * $(2)) + $(512)), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r46_value = r46_value.to(tl.bfloat16)
    r46_value = tl.broadcast_to(r46_value, [$(8192)])
    tl.store($(candidate_vmul_18_ptr) + r46_z, r46_value)
    r47_z = tl.arange($(8192))
    r47_value = (tl.load($(candidate_vmul_17_ptr) + ((((r47_z / $(4096)) % $(2)) * $(4096)) + (((r47_z / $(32)) % $(128)) * $(32)) + (r47_z % $(32))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_18_ptr) + ((((r47_z / $(4096)) % $(2)) * $(4096)) + (((r47_z / $(32)) % $(128)) * $(32)) + (r47_z % $(32))), dtype=tl.bfloat16))
    r47_value = r47_value.to(tl.bfloat16)
    r47_value = tl.broadcast_to(r47_value, [$(8192)])
    tl.store($(candidate_vadd_5_ptr) + r47_z, r47_value)
    r48_z = tl.arange($(16384))
    r48_value = tl.where((r48_z % $(64)) < $(32), tl.load($(candidate_vsub_1_ptr) + ((((r48_z / $(8192)) % $(2)) * $(4096)) + (((r48_z / $(64)) % $(128)) * $(32)) + (((r48_z % $(64)) - $(0)) % $(32))), dtype=tl.bfloat16), tl.load($(candidate_vadd_5_ptr) + ((((r48_z / $(8192)) % $(2)) * $(4096)) + (((r48_z / $(64)) % $(128)) * $(32)) + (((r48_z % $(64)) - $(32)) % $(32))), dtype=tl.bfloat16))
    r48_value = r48_value.to(tl.bfloat16)
    r48_value = tl.broadcast_to(r48_value, [$(16384)])
    tl.store($(candidate_vcat_1_ptr) + r48_z, r48_value)
    r49_row = tl.arange($(256))
    r49_col = tl.arange($(1024))
    r49_lane = tl.arange($(512))
    r49_left = tl.load($(input1_ptr) + ((((((r49_row[:, None] * $(512)) + r49_lane[None, :]) / $(65536)) % $(2)) * $(65536)) + (((((r49_row[:, None] * $(512)) + r49_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r49_row[:, None] * $(512)) + r49_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r49_right = tl.load($(candidate_vto_copy_6_ptr) + (((r49_lane[:, None] % $(512)) * $(1024)) + (r49_col[None, :] % $(1024))), dtype=tl.bfloat16)
    r49_value = tl.dot(r49_left, r49_right)
    r49_value = r49_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_8_ptr) + ((r49_row[:, None] * $(1024)) + r49_col[None, :]), r49_value)
    r50_row = tl.arange($(256))
    r50_col = tl.arange($(1024))
    r50_lane = tl.arange($(512))
    for r50_group in range($(1)) {
      r50_left = tl.load($(candidate_vto_copy_19_ptr) + ((((((((((r50_row[:, None] * $(512)) + r50_lane[None, :]) / $(512)) % $(256)) * $(512)) + (((r50_row[:, None] * $(512)) + r50_lane[None, :]) % $(512))) / $(65536)) % $(2)) * $(65536)) + ((((((r50_row[:, None] * $(512)) + r50_lane[None, :]) / $(512)) % $(256)) % $(128)) * $(512)) + (((r50_row[:, None] * $(512)) + r50_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r50_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r50_lane[:, None] * $(1024)) + r50_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r50_lane[:, None] * $(1024)) + r50_col[None, :]) % $(128)) * $(512)) + ((((r50_lane[:, None] * $(1024)) + r50_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r50_value = tl.dot(r50_left, r50_right)
      r50_value = r50_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_4_ptr) + ((r50_group * $(262144)) + (r50_row[:, None] * $(1024)) + r50_col[None, :]), r50_value)
    }
    r51_row = tl.arange($(256))
    r51_col = tl.arange($(1024))
    r51_lane = tl.arange($(512))
    for r51_group in range($(1)) {
      r51_left = tl.load($(candidate_vto_copy_19_ptr) + ((((((((((r51_row[:, None] * $(512)) + r51_lane[None, :]) / $(512)) % $(256)) * $(512)) + (((r51_row[:, None] * $(512)) + r51_lane[None, :]) % $(512))) / $(65536)) % $(2)) * $(65536)) + ((((((r51_row[:, None] * $(512)) + r51_lane[None, :]) / $(512)) % $(256)) % $(128)) * $(512)) + (((r51_row[:, None] * $(512)) + r51_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r51_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r51_lane[:, None] * $(1024)) + r51_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r51_lane[:, None] * $(1024)) + r51_col[None, :]) % $(128)) * $(512)) + ((((r51_lane[:, None] * $(1024)) + r51_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r51_value = tl.dot(r51_left, r51_right)
      r51_value = r51_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_5_ptr) + ((r51_group * $(262144)) + (r51_row[:, None] * $(1024)) + r51_col[None, :]), r51_value)
    }
    r52_z = tl.arange($(393216))
    r52_value = tl.where((r52_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r52_z / $(1536)) % $(256)) * $(1024)) + (((r52_z / $(192)) % $(8)) * $(128)) + ((r52_z % $(192)) - $(0))) / $(131072)) % $(2)) * $(196608)) + (((((((r52_z / $(1536)) % $(256)) * $(1024)) + (((r52_z / $(192)) % $(8)) * $(128)) + ((r52_z % $(192)) - $(0))) / $(1024)) % $(128)) * $(1536)) + (((((((r52_z / $(1536)) % $(256)) * $(1024)) + (((r52_z / $(192)) % $(8)) * $(128)) + ((r52_z % $(192)) - $(0))) / $(128)) % $(8)) * $(192)) + (((((r52_z / $(1536)) % $(256)) * $(1024)) + (((r52_z / $(192)) % $(8)) * $(128)) + ((r52_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_ptr) + ((((((((r52_z / $(1536)) % $(256)) * $(512)) + (((r52_z / $(192)) % $(8)) * $(64)) + ((r52_z % $(192)) - $(128))) / $(65536)) % $(2)) * $(65536)) + (((((((r52_z / $(1536)) % $(256)) * $(512)) + (((r52_z / $(192)) % $(8)) * $(64)) + ((r52_z % $(192)) - $(128))) / $(64)) % $(8)) * $(8192)) + (((((((r52_z / $(1536)) % $(256)) * $(512)) + (((r52_z / $(192)) % $(8)) * $(64)) + ((r52_z % $(192)) - $(128))) / $(512)) % $(128)) * $(64)) + (((((r52_z / $(1536)) % $(256)) * $(512)) + (((r52_z / $(192)) % $(8)) * $(64)) + ((r52_z % $(192)) - $(128))) % $(64))), dtype=tl.bfloat16))
    r52_value = r52_value.to(tl.bfloat16)
    r52_value = tl.broadcast_to(r52_value, [$(393216)])
    tl.store($(candidate_vcat_3_ptr) + r52_z, r52_value)
    r53_z = tl.arange($(393216))
    r53_value = tl.where((r53_z % $(192)) < $(128), tl.load($(candidate_vbmm_4_ptr) + ((((((((r53_z / $(1536)) % $(256)) * $(1024)) + (((r53_z / $(192)) % $(8)) * $(128)) + ((r53_z % $(192)) - $(0))) / $(1024)) % $(256)) * $(1024)) + (((((((r53_z / $(1536)) % $(256)) * $(1024)) + (((r53_z / $(192)) % $(8)) * $(128)) + ((r53_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r53_z / $(1536)) % $(256)) * $(1024)) + (((r53_z / $(192)) % $(8)) * $(128)) + ((r53_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_1_ptr) + ((((((((r53_z / $(1536)) % $(256)) * $(64)) + ((r53_z % $(192)) - $(128))) / $(8192)) % $(2)) * $(8192)) + (((((((r53_z / $(1536)) % $(256)) * $(64)) + ((r53_z % $(192)) - $(128))) / $(64)) % $(128)) * $(64)) + (((((r53_z / $(1536)) % $(256)) * $(64)) + ((r53_z % $(192)) - $(128))) % $(64))), dtype=tl.bfloat16))
    r53_value = r53_value.to(tl.bfloat16)
    r53_value = tl.broadcast_to(r53_value, [$(393216)])
    tl.store($(candidate_vcat_4_ptr) + r53_z, r53_value)
    r54_z = tl.arange($(196608))
    r54_value = tl.load($(candidate_vcat_3_ptr) + ((((r54_z / $(192)) % $(128)) * $(1536)) + (((r54_z / $(24576)) % $(8)) * $(192)) + (r54_z % $(192))), dtype=tl.bfloat16)
    r54_value = r54_value.to(tl.float32)
    r54_value = tl.broadcast_to(r54_value, [$(196608)])
    tl.store($(candidate_vto_copy_30_ptr) + r54_z, r54_value)
    r55_z = tl.arange($(196608))
    r55_value = tl.load($(candidate_vcat_4_ptr) + ((((r55_z / $(192)) % $(128)) * $(1536)) + (((r55_z / $(24576)) % $(8)) * $(192)) + (r55_z % $(192))), dtype=tl.bfloat16)
    r55_value = r55_value.to(tl.float32)
    r55_value = tl.broadcast_to(r55_value, [$(196608)])
    tl.store($(candidate_vto_copy_31_ptr) + r55_z, r55_value)
    r56_row = tl.arange($(128))
    r56_col = tl.arange($(128))
    r56_lane = tl.arange($(192))
    for r56_group in range($(8)) {
      r56_left = tl.load($(candidate_vto_copy_30_ptr) + (((((((r56_group % $(8)) * $(24576)) + (r56_row[:, None] * $(192)) + r56_lane[None, :]) / $(24576)) % $(8)) * $(24576)) + ((((((r56_group % $(8)) * $(24576)) + (r56_row[:, None] * $(192)) + r56_lane[None, :]) / $(192)) % $(128)) * $(192)) + ((((r56_group % $(8)) * $(24576)) + (r56_row[:, None] * $(192)) + r56_lane[None, :]) % $(192))), dtype=tl.float32)
      r56_right = tl.load($(candidate_vto_copy_31_ptr) + (((((((r56_group % $(8)) * $(24576)) + (r56_lane[:, None] * $(128)) + r56_col[None, :]) / $(24576)) % $(8)) * $(24576)) + (((((r56_group % $(8)) * $(24576)) + (r56_lane[:, None] * $(128)) + r56_col[None, :]) % $(128)) * $(192)) + (((((r56_group % $(8)) * $(24576)) + (r56_lane[:, None] * $(128)) + r56_col[None, :]) / $(128)) % $(192))), dtype=tl.float32)
      r56_value = tl.dot(r56_left, r56_right)
      r56_value = r56_value.to(tl.float32)
      tl.store($(candidate_vbmm_6_ptr) + ((r56_group * $(16384)) + (r56_row[:, None] * $(128)) + r56_col[None, :]), r56_value)
    }
    r57_z = tl.arange($(131072))
    r57_value = (tl.load($(candidate_vbmm_6_ptr) + ((((r57_z / $(16384)) % $(8)) * $(16384)) + (((r57_z / $(128)) % $(128)) * $(128)) + (r57_z % $(128))), dtype=tl.float32) * 0.07216878364870322)
    r57_value = r57_value.to(tl.float32)
    r57_value = tl.broadcast_to(r57_value, [$(131072)])
    tl.store($(candidate_vmul_21_ptr) + r57_z, r57_value)
    r58_z = tl.arange($(128))
    r58_value = (tl.cast(r58_z, tl.int64) + tl.full([], $(128), dtype=tl.int64))
    r58_value = tl.broadcast_to(r58_value, [$(128)])
    tl.store($(candidate_vadd_10_ptr) + r58_z, r58_value)
    r59_z = tl.arange($(128))
    r59_value = (tl.load($(candidate_vadd_10_ptr) + r59_z, dtype=tl.int64) - tl.full([], $(128), dtype=tl.int64))
    r59_value = tl.broadcast_to(r59_value, [$(128)])
    tl.store($(candidate_vsub_4_ptr) + r59_z, r59_value)
    r60_z = tl.arange($(16384))
    r60_value = (tl.cast((r60_z % $(128)), tl.int64) > tl.load($(candidate_vsub_4_ptr) + ((r60_z / $(128)) % $(128)), dtype=tl.int64))
    r60_value = tl.broadcast_to(r60_value, [$(16384)])
    tl.store($(candidate_vgt_2_ptr) + r60_z, r60_value)
    r61_z = tl.arange($(131072))
    r61_value = tl.where(tl.load($(candidate_vgt_2_ptr) + ((((r61_z / $(128)) % $(128)) * $(128)) + (r61_z % $(128))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_21_ptr) + ((((r61_z / $(16384)) % $(8)) * $(16384)) + (((r61_z / $(128)) % $(128)) * $(128)) + (r61_z % $(128))), dtype=tl.float32))
    r61_value = r61_value.to(tl.float32)
    r61_value = tl.broadcast_to(r61_value, [$(131072)])
    tl.store($(candidate_vmasked_fill_4_ptr) + r61_z, r61_value)
    r62_z = tl.arange($(131072))
    r62_lane = tl.arange($(128))
    r62_row_values = tl.load($(candidate_vmasked_fill_4_ptr) + ((((r62_z[:, None] / $(16384)) % $(8)) * $(16384)) + (((r62_z[:, None] / $(128)) % $(128)) * $(128)) + (r62_lane[None, :] % $(128))), dtype=tl.float32)
    r62_maximum = tl.max(r62_row_values, axis=1, keep_dims=true)
    r62_weights = tl.exp(r62_row_values - r62_maximum)
    r62_total = tl.sum(r62_weights, axis=1)
    r62_maximum = tl.max(r62_row_values, axis=1)
    r62_value = tl.exp(tl.load($(candidate_vmasked_fill_4_ptr) + ((((r62_z / $(16384)) % $(8)) * $(16384)) + (((r62_z / $(128)) % $(128)) * $(128)) + (r62_z % $(128))), dtype=tl.float32) - r62_maximum) / r62_total
    r62_value = r62_value.to(tl.float32)
    r62_value = tl.broadcast_to(r62_value, [$(131072)])
    tl.store($(candidate_vsoftmax_2_ptr) + r62_z, r62_value)
    r63_z = tl.arange($(131072))
    r63_value = tl.load($(candidate_vsoftmax_2_ptr) + ((((r63_z / $(16384)) % $(8)) * $(16384)) + (((r63_z / $(128)) % $(128)) * $(128)) + (r63_z % $(128))), dtype=tl.float32)
    r63_value = r63_value.to(tl.bfloat16)
    r63_value = tl.broadcast_to(r63_value, [$(131072)])
    tl.store($(candidate_vto_copy_32_ptr) + r63_z, r63_value)
    r64_row = tl.arange($(128))
    r64_col = tl.arange($(128))
    r64_lane = tl.arange($(128))
    for r64_group in range($(8)) {
      r64_left = tl.load($(candidate_vto_copy_32_ptr) + (((((((r64_group % $(8)) * $(16384)) + (r64_row[:, None] * $(128)) + r64_lane[None, :]) / $(16384)) % $(8)) * $(16384)) + ((((((r64_group % $(8)) * $(16384)) + (r64_row[:, None] * $(128)) + r64_lane[None, :]) / $(128)) % $(128)) * $(128)) + ((((r64_group % $(8)) * $(16384)) + (r64_row[:, None] * $(128)) + r64_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r64_right = tl.load($(candidate_vbmm_5_ptr) + (((((((r64_group % $(8)) * $(16384)) + (r64_lane[:, None] * $(128)) + r64_col[None, :]) / $(128)) % $(128)) * $(1024)) + ((((((r64_group % $(8)) * $(16384)) + (r64_lane[:, None] * $(128)) + r64_col[None, :]) / $(16384)) % $(8)) * $(128)) + ((((r64_group % $(8)) * $(16384)) + (r64_lane[:, None] * $(128)) + r64_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r64_value = tl.dot(r64_left, r64_right)
      r64_value = r64_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_7_ptr) + ((r64_group * $(16384)) + (r64_row[:, None] * $(128)) + r64_col[None, :]), r64_value)
    }
    r65_z = tl.arange($(1024))
    r65_lane = tl.arange($(128))
    r65_row_values = tl.load($(candidate_vmasked_fill_4_ptr) + ((((r65_z[:, None] / $(128)) % $(8)) * $(16384)) + ((r65_z[:, None] % $(128)) * $(128)) + (r65_lane[None, :] % $(128))), dtype=tl.float32)
    r65_maximum = tl.max(r65_row_values, axis=1, keep_dims=true)
    r65_weights = tl.exp(r65_row_values - r65_maximum)
    r65_total = tl.sum(r65_weights, axis=1)
    r65_maximum = tl.max(r65_row_values, axis=1)
    r65_value = r65_maximum + tl.log(r65_total)
    r65_value = r65_value.to(tl.float32)
    r65_value = tl.broadcast_to(r65_value, [$(1024)])
    tl.store($(candidate_vlogsumexp_2_ptr) + r65_z, r65_value)
    r66_z = tl.arange($(196608))
    r66_value = tl.load($(candidate_vcat_3_ptr) + ((((r66_z / $(192)) % $(128)) * $(1536)) + (((r66_z / $(24576)) % $(8)) * $(192)) + (r66_z % $(192)) + $(196608)), dtype=tl.bfloat16)
    r66_value = r66_value.to(tl.float32)
    r66_value = tl.broadcast_to(r66_value, [$(196608)])
    tl.store($(candidate_vto_copy_33_ptr) + r66_z, r66_value)
    r67_z = tl.arange($(196608))
    r67_value = tl.load($(candidate_vcat_4_ptr) + ((((r67_z / $(192)) % $(128)) * $(1536)) + (((r67_z / $(24576)) % $(8)) * $(192)) + (r67_z % $(192)) + $(196608)), dtype=tl.bfloat16)
    r67_value = r67_value.to(tl.float32)
    r67_value = tl.broadcast_to(r67_value, [$(196608)])
    tl.store($(candidate_vto_copy_34_ptr) + r67_z, r67_value)
    r68_row = tl.arange($(128))
    r68_col = tl.arange($(128))
    r68_lane = tl.arange($(192))
    for r68_group in range($(8)) {
      r68_left = tl.load($(candidate_vto_copy_33_ptr) + (((((((r68_group % $(8)) * $(24576)) + (r68_row[:, None] * $(192)) + r68_lane[None, :]) / $(24576)) % $(8)) * $(24576)) + ((((((r68_group % $(8)) * $(24576)) + (r68_row[:, None] * $(192)) + r68_lane[None, :]) / $(192)) % $(128)) * $(192)) + ((((r68_group % $(8)) * $(24576)) + (r68_row[:, None] * $(192)) + r68_lane[None, :]) % $(192))), dtype=tl.float32)
      r68_right = tl.load($(candidate_vto_copy_34_ptr) + (((((((r68_group % $(8)) * $(24576)) + (r68_lane[:, None] * $(128)) + r68_col[None, :]) / $(24576)) % $(8)) * $(24576)) + (((((r68_group % $(8)) * $(24576)) + (r68_lane[:, None] * $(128)) + r68_col[None, :]) % $(128)) * $(192)) + (((((r68_group % $(8)) * $(24576)) + (r68_lane[:, None] * $(128)) + r68_col[None, :]) / $(128)) % $(192))), dtype=tl.float32)
      r68_value = tl.dot(r68_left, r68_right)
      r68_value = r68_value.to(tl.float32)
      tl.store($(candidate_vbmm_8_ptr) + ((r68_group * $(16384)) + (r68_row[:, None] * $(128)) + r68_col[None, :]), r68_value)
    }
    r69_z = tl.arange($(131072))
    r69_value = (tl.load($(candidate_vbmm_8_ptr) + ((((r69_z / $(16384)) % $(8)) * $(16384)) + (((r69_z / $(128)) % $(128)) * $(128)) + (r69_z % $(128))), dtype=tl.float32) * 0.07216878364870322)
    r69_value = r69_value.to(tl.float32)
    r69_value = tl.broadcast_to(r69_value, [$(131072)])
    tl.store($(candidate_vmul_22_ptr) + r69_z, r69_value)
    r70_z = tl.arange($(128))
    r70_value = (tl.cast(r70_z, tl.int64) + tl.full([], $(128), dtype=tl.int64))
    r70_value = tl.broadcast_to(r70_value, [$(128)])
    tl.store($(candidate_vadd_11_ptr) + r70_z, r70_value)
    r71_z = tl.arange($(128))
    r71_value = (tl.load($(candidate_vadd_11_ptr) + r71_z, dtype=tl.int64) - tl.full([], $(128), dtype=tl.int64))
    r71_value = tl.broadcast_to(r71_value, [$(128)])
    tl.store($(candidate_vsub_5_ptr) + r71_z, r71_value)
    r72_z = tl.arange($(16384))
    r72_value = (tl.cast((r72_z % $(128)), tl.int64) > tl.load($(candidate_vsub_5_ptr) + ((r72_z / $(128)) % $(128)), dtype=tl.int64))
    r72_value = tl.broadcast_to(r72_value, [$(16384)])
    tl.store($(candidate_vgt_3_ptr) + r72_z, r72_value)
    r73_z = tl.arange($(131072))
    r73_value = tl.where(tl.load($(candidate_vgt_3_ptr) + ((((r73_z / $(128)) % $(128)) * $(128)) + (r73_z % $(128))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_22_ptr) + ((((r73_z / $(16384)) % $(8)) * $(16384)) + (((r73_z / $(128)) % $(128)) * $(128)) + (r73_z % $(128))), dtype=tl.float32))
    r73_value = r73_value.to(tl.float32)
    r73_value = tl.broadcast_to(r73_value, [$(131072)])
    tl.store($(candidate_vmasked_fill_5_ptr) + r73_z, r73_value)
    r74_z = tl.arange($(131072))
    r74_lane = tl.arange($(128))
    r74_row_values = tl.load($(candidate_vmasked_fill_5_ptr) + ((((r74_z[:, None] / $(16384)) % $(8)) * $(16384)) + (((r74_z[:, None] / $(128)) % $(128)) * $(128)) + (r74_lane[None, :] % $(128))), dtype=tl.float32)
    r74_maximum = tl.max(r74_row_values, axis=1, keep_dims=true)
    r74_weights = tl.exp(r74_row_values - r74_maximum)
    r74_total = tl.sum(r74_weights, axis=1)
    r74_maximum = tl.max(r74_row_values, axis=1)
    r74_value = tl.exp(tl.load($(candidate_vmasked_fill_5_ptr) + ((((r74_z / $(16384)) % $(8)) * $(16384)) + (((r74_z / $(128)) % $(128)) * $(128)) + (r74_z % $(128))), dtype=tl.float32) - r74_maximum) / r74_total
    r74_value = r74_value.to(tl.float32)
    r74_value = tl.broadcast_to(r74_value, [$(131072)])
    tl.store($(candidate_vsoftmax_3_ptr) + r74_z, r74_value)
    r75_z = tl.arange($(131072))
    r75_value = tl.load($(candidate_vsoftmax_3_ptr) + ((((r75_z / $(16384)) % $(8)) * $(16384)) + (((r75_z / $(128)) % $(128)) * $(128)) + (r75_z % $(128))), dtype=tl.float32)
    r75_value = r75_value.to(tl.bfloat16)
    r75_value = tl.broadcast_to(r75_value, [$(131072)])
    tl.store($(candidate_vto_copy_35_ptr) + r75_z, r75_value)
    r76_row = tl.arange($(128))
    r76_col = tl.arange($(128))
    r76_lane = tl.arange($(128))
    for r76_group in range($(8)) {
      r76_left = tl.load($(candidate_vto_copy_35_ptr) + (((((((r76_group % $(8)) * $(16384)) + (r76_row[:, None] * $(128)) + r76_lane[None, :]) / $(16384)) % $(8)) * $(16384)) + ((((((r76_group % $(8)) * $(16384)) + (r76_row[:, None] * $(128)) + r76_lane[None, :]) / $(128)) % $(128)) * $(128)) + ((((r76_group % $(8)) * $(16384)) + (r76_row[:, None] * $(128)) + r76_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r76_right = tl.load($(candidate_vbmm_5_ptr) + (((((((r76_group % $(8)) * $(16384)) + (r76_lane[:, None] * $(128)) + r76_col[None, :]) / $(128)) % $(128)) * $(1024)) + ((((((r76_group % $(8)) * $(16384)) + (r76_lane[:, None] * $(128)) + r76_col[None, :]) / $(16384)) % $(8)) * $(128)) + ((((r76_group % $(8)) * $(16384)) + (r76_lane[:, None] * $(128)) + r76_col[None, :]) % $(128)) + $(131072)), dtype=tl.bfloat16)
      r76_value = tl.dot(r76_left, r76_right)
      r76_value = r76_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_9_ptr) + ((r76_group * $(16384)) + (r76_row[:, None] * $(128)) + r76_col[None, :]), r76_value)
    }
    r77_z = tl.arange($(1024))
    r77_lane = tl.arange($(128))
    r77_row_values = tl.load($(candidate_vmasked_fill_5_ptr) + ((((r77_z[:, None] / $(128)) % $(8)) * $(16384)) + ((r77_z[:, None] % $(128)) * $(128)) + (r77_lane[None, :] % $(128))), dtype=tl.float32)
    r77_maximum = tl.max(r77_row_values, axis=1, keep_dims=true)
    r77_weights = tl.exp(r77_row_values - r77_maximum)
    r77_total = tl.sum(r77_weights, axis=1)
    r77_maximum = tl.max(r77_row_values, axis=1)
    r77_value = r77_maximum + tl.log(r77_total)
    r77_value = r77_value.to(tl.float32)
    r77_value = tl.broadcast_to(r77_value, [$(1024)])
    tl.store($(candidate_vlogsumexp_3_ptr) + r77_z, r77_value)
    r78_z = tl.arange($(262144))
    r78_value = tl.where(((r78_z / $(1024)) % $(256)) < $(128), tl.load($(candidate_vbmm_7_ptr) + ((((((((r78_z / $(128)) % $(8)) * $(16384)) + ((((r78_z / $(1024)) % $(256)) - $(0)) * $(128)) + (r78_z % $(128))) / $(16384)) % $(8)) * $(16384)) + ((((((r78_z / $(128)) % $(8)) * $(128)) + (((r78_z / $(1024)) % $(256)) - $(0))) % $(128)) * $(128)) + (r78_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vbmm_9_ptr) + ((((((((r78_z / $(128)) % $(8)) * $(16384)) + ((((r78_z / $(1024)) % $(256)) - $(128)) * $(128)) + (r78_z % $(128))) / $(16384)) % $(8)) * $(16384)) + ((((((r78_z / $(128)) % $(8)) * $(128)) + (((r78_z / $(1024)) % $(256)) - $(128))) % $(128)) * $(128)) + (r78_z % $(128))), dtype=tl.bfloat16))
    r78_value = r78_value.to(tl.bfloat16)
    r78_value = tl.broadcast_to(r78_value, [$(262144)])
    tl.store($(candidate_vcat_5_ptr) + r78_z, r78_value)
    r79_z = tl.arange($(2048))
    r79_value = tl.where(((r79_z / $(8)) % $(256)) < $(128), tl.load($(candidate_vlogsumexp_2_ptr) + (((r79_z % $(8)) * $(128)) + ((((r79_z / $(8)) % $(256)) - $(0)) % $(128))), dtype=tl.float32), tl.load($(candidate_vlogsumexp_3_ptr) + (((r79_z % $(8)) * $(128)) + ((((r79_z / $(8)) % $(256)) - $(128)) % $(128))), dtype=tl.float32))
    r79_value = r79_value.to(tl.float32)
    r79_value = tl.broadcast_to(r79_value, [$(2048)])
    tl.store($(candidate_vcat_6_ptr) + r79_z, r79_value)
    r80_z = tl.arange($(196608))
    r80_value = tl.load($(candidate_vcat_3_ptr) + ((((r80_z / $(192)) % $(128)) * $(1536)) + (((r80_z / $(24576)) % $(8)) * $(192)) + (r80_z % $(192))), dtype=tl.bfloat16)
    r80_value = r80_value.to(tl.float32)
    r80_value = tl.broadcast_to(r80_value, [$(196608)])
    tl.store($(candidate_vto_copy_36_ptr) + r80_z, r80_value)
    r81_z = tl.arange($(196608))
    r81_value = tl.load($(candidate_vcat_4_ptr) + ((((r81_z / $(192)) % $(128)) * $(1536)) + (((r81_z / $(24576)) % $(8)) * $(192)) + (r81_z % $(192))), dtype=tl.bfloat16)
    r81_value = r81_value.to(tl.float32)
    r81_value = tl.broadcast_to(r81_value, [$(196608)])
    tl.store($(candidate_vto_copy_37_ptr) + r81_z, r81_value)
    r82_z = tl.arange($(131072))
    r82_value = tl.load($(candidate_vbmm_5_ptr) + ((((r82_z / $(128)) % $(128)) * $(1024)) + (((r82_z / $(16384)) % $(8)) * $(128)) + (r82_z % $(128))), dtype=tl.bfloat16)
    r82_value = r82_value.to(tl.float32)
    r82_value = tl.broadcast_to(r82_value, [$(131072)])
    tl.store($(candidate_vto_copy_38_ptr) + r82_z, r82_value)
    r83_z = tl.arange($(131072))
    r83_value = tl.load($(candidate_vmm_8_ptr) + ((((r83_z / $(128)) % $(128)) * $(1024)) + (((r83_z / $(16384)) % $(8)) * $(128)) + (r83_z % $(128))), dtype=tl.bfloat16)
    r83_value = r83_value.to(tl.float32)
    r83_value = tl.broadcast_to(r83_value, [$(131072)])
    tl.store($(candidate_vto_copy_39_ptr) + r83_z, r83_value)
    r84_z = tl.arange($(131072))
    r84_value = tl.load($(candidate_vcat_5_ptr) + ((((r84_z / $(128)) % $(128)) * $(1024)) + (((r84_z / $(16384)) % $(8)) * $(128)) + (r84_z % $(128))), dtype=tl.bfloat16)
    r84_value = r84_value.to(tl.float32)
    r84_value = tl.broadcast_to(r84_value, [$(131072)])
    tl.store($(candidate_vto_copy_40_ptr) + r84_z, r84_value)
    r85_row = tl.arange($(128))
    r85_col = tl.arange($(128))
    r85_lane = tl.arange($(192))
    for r85_group in range($(8)) {
      r85_left = tl.load($(candidate_vto_copy_36_ptr) + (((((((r85_group % $(8)) * $(24576)) + (r85_row[:, None] * $(192)) + r85_lane[None, :]) / $(24576)) % $(8)) * $(24576)) + ((((((r85_group % $(8)) * $(24576)) + (r85_row[:, None] * $(192)) + r85_lane[None, :]) / $(192)) % $(128)) * $(192)) + ((((r85_group % $(8)) * $(24576)) + (r85_row[:, None] * $(192)) + r85_lane[None, :]) % $(192))), dtype=tl.float32)
      r85_right = tl.load($(candidate_vto_copy_37_ptr) + (((((((r85_group % $(8)) * $(24576)) + (r85_lane[:, None] * $(128)) + r85_col[None, :]) / $(24576)) % $(8)) * $(24576)) + (((((r85_group % $(8)) * $(24576)) + (r85_lane[:, None] * $(128)) + r85_col[None, :]) % $(128)) * $(192)) + (((((r85_group % $(8)) * $(24576)) + (r85_lane[:, None] * $(128)) + r85_col[None, :]) / $(128)) % $(192))), dtype=tl.float32)
      r85_value = tl.dot(r85_left, r85_right)
      r85_value = r85_value.to(tl.float32)
      tl.store($(candidate_vbmm_10_ptr) + ((r85_group * $(16384)) + (r85_row[:, None] * $(128)) + r85_col[None, :]), r85_value)
    }
    r86_z = tl.arange($(131072))
    r86_value = (tl.load($(candidate_vbmm_10_ptr) + ((((r86_z / $(16384)) % $(8)) * $(16384)) + (((r86_z / $(128)) % $(128)) * $(128)) + (r86_z % $(128))), dtype=tl.float32) * 0.07216878364870322)
    r86_value = r86_value.to(tl.float32)
    r86_value = tl.broadcast_to(r86_value, [$(131072)])
    tl.store($(candidate_vmul_23_ptr) + r86_z, r86_value)
    r87_z = tl.arange($(128))
    r87_value = (tl.cast(r87_z, tl.int64) + tl.full([], $(128), dtype=tl.int64))
    r87_value = tl.broadcast_to(r87_value, [$(128)])
    tl.store($(candidate_vadd_12_ptr) + r87_z, r87_value)
    r88_z = tl.arange($(128))
    r88_value = (tl.load($(candidate_vadd_12_ptr) + r88_z, dtype=tl.int64) - tl.full([], $(128), dtype=tl.int64))
    r88_value = tl.broadcast_to(r88_value, [$(128)])
    tl.store($(candidate_vsub_6_ptr) + r88_z, r88_value)
    r89_z = tl.arange($(16384))
    r89_value = (tl.cast((r89_z % $(128)), tl.int64) > tl.load($(candidate_vsub_6_ptr) + ((r89_z / $(128)) % $(128)), dtype=tl.int64))
    r89_value = tl.broadcast_to(r89_value, [$(16384)])
    tl.store($(candidate_vgt_4_ptr) + r89_z, r89_value)
    r90_z = tl.arange($(131072))
    r90_value = tl.where(tl.load($(candidate_vgt_4_ptr) + ((((r90_z / $(128)) % $(128)) * $(128)) + (r90_z % $(128))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_23_ptr) + ((((r90_z / $(16384)) % $(8)) * $(16384)) + (((r90_z / $(128)) % $(128)) * $(128)) + (r90_z % $(128))), dtype=tl.float32))
    r90_value = r90_value.to(tl.float32)
    r90_value = tl.broadcast_to(r90_value, [$(131072)])
    tl.store($(candidate_vmasked_fill_6_ptr) + r90_z, r90_value)
    r91_z = tl.arange($(131072))
    r91_value = (tl.load($(candidate_vmasked_fill_6_ptr) + ((((r91_z / $(16384)) % $(8)) * $(16384)) + (((r91_z / $(128)) % $(128)) * $(128)) + (r91_z % $(128))), dtype=tl.float32) - tl.load($(candidate_vcat_6_ptr) + ((((r91_z / $(128)) % $(128)) * $(8)) + ((r91_z / $(16384)) % $(8))), dtype=tl.float32))
    r91_value = r91_value.to(tl.float32)
    r91_value = tl.broadcast_to(r91_value, [$(131072)])
    tl.store($(candidate_vsub_7_ptr) + r91_z, r91_value)
    r92_z = tl.arange($(131072))
    r92_value = tl.exp(tl.load($(candidate_vsub_7_ptr) + ((((r92_z / $(16384)) % $(8)) * $(16384)) + (((r92_z / $(128)) % $(128)) * $(128)) + (r92_z % $(128))), dtype=tl.float32))
    r92_value = r92_value.to(tl.float32)
    r92_value = tl.broadcast_to(r92_value, [$(131072)])
    tl.store($(candidate_vexp_ptr) + r92_z, r92_value)
    r93_z = tl.arange($(131072))
    r93_value = (tl.load($(candidate_vto_copy_39_ptr) + ((((r93_z / $(16384)) % $(8)) * $(16384)) + (((r93_z / $(128)) % $(128)) * $(128)) + (r93_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_40_ptr) + ((((r93_z / $(16384)) % $(8)) * $(16384)) + (((r93_z / $(128)) % $(128)) * $(128)) + (r93_z % $(128))), dtype=tl.float32))
    r93_value = r93_value.to(tl.float32)
    r93_value = tl.broadcast_to(r93_value, [$(131072)])
    tl.store($(candidate_vmul_24_ptr) + r93_z, r93_value)
    r94_z = tl.arange($(1024))
    r94_lane = tl.arange($(128))
    r94_value = tl.sum(tl.load($(candidate_vmul_24_ptr) + ((((r94_z[:, None] / $(128)) % $(8)) * $(16384)) + ((r94_z[:, None] % $(128)) * $(128)) + r94_lane[None, :]), dtype=tl.float32), axis=1)
    r94_value = r94_value.to(tl.float32)
    r94_value = tl.broadcast_to(r94_value, [$(1024)])
    tl.store($(candidate_vsum_1_ptr) + r94_z, r94_value)
    r95_row = tl.arange($(128))
    r95_col = tl.arange($(128))
    r95_lane = tl.arange($(128))
    for r95_group in range($(8)) {
      r95_left = tl.load($(candidate_vto_copy_39_ptr) + (((((((r95_group % $(8)) * $(16384)) + (r95_row[:, None] * $(128)) + r95_lane[None, :]) / $(16384)) % $(8)) * $(16384)) + ((((((r95_group % $(8)) * $(16384)) + (r95_row[:, None] * $(128)) + r95_lane[None, :]) / $(128)) % $(128)) * $(128)) + ((((r95_group % $(8)) * $(16384)) + (r95_row[:, None] * $(128)) + r95_lane[None, :]) % $(128))), dtype=tl.float32)
      r95_right = tl.load($(candidate_vto_copy_38_ptr) + (((((((r95_group % $(8)) * $(16384)) + (r95_lane[:, None] * $(128)) + r95_col[None, :]) / $(16384)) % $(8)) * $(16384)) + (((((r95_group % $(8)) * $(16384)) + (r95_lane[:, None] * $(128)) + r95_col[None, :]) % $(128)) * $(128)) + (((((r95_group % $(8)) * $(16384)) + (r95_lane[:, None] * $(128)) + r95_col[None, :]) / $(128)) % $(128))), dtype=tl.float32)
      r95_value = tl.dot(r95_left, r95_right)
      r95_value = r95_value.to(tl.float32)
      tl.store($(candidate_vbmm_11_ptr) + ((r95_group * $(16384)) + (r95_row[:, None] * $(128)) + r95_col[None, :]), r95_value)
    }
    r96_z = tl.arange($(131072))
    r96_value = (tl.load($(candidate_vbmm_11_ptr) + ((((r96_z / $(16384)) % $(8)) * $(16384)) + (((r96_z / $(128)) % $(128)) * $(128)) + (r96_z % $(128))), dtype=tl.float32) - tl.load($(candidate_vsum_1_ptr) + ((((r96_z / $(16384)) % $(8)) * $(128)) + ((r96_z / $(128)) % $(128))), dtype=tl.float32))
    r96_value = r96_value.to(tl.float32)
    r96_value = tl.broadcast_to(r96_value, [$(131072)])
    tl.store($(candidate_vsub_8_ptr) + r96_z, r96_value)
    r97_z = tl.arange($(131072))
    r97_value = (tl.load($(candidate_vexp_ptr) + ((((r97_z / $(16384)) % $(8)) * $(16384)) + (((r97_z / $(128)) % $(128)) * $(128)) + (r97_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vsub_8_ptr) + ((((r97_z / $(16384)) % $(8)) * $(16384)) + (((r97_z / $(128)) % $(128)) * $(128)) + (r97_z % $(128))), dtype=tl.float32))
    r97_value = r97_value.to(tl.float32)
    r97_value = tl.broadcast_to(r97_value, [$(131072)])
    tl.store($(candidate_vmul_25_ptr) + r97_z, r97_value)
    r98_row = tl.arange($(128))
    r98_col = tl.arange($(192))
    r98_lane = tl.arange($(128))
    for r98_group in range($(8)) {
      r98_left = tl.load($(candidate_vmul_25_ptr) + (((((((r98_group % $(8)) * $(16384)) + (r98_row[:, None] * $(128)) + r98_lane[None, :]) / $(16384)) % $(8)) * $(16384)) + ((((((r98_group % $(8)) * $(16384)) + (r98_row[:, None] * $(128)) + r98_lane[None, :]) / $(128)) % $(128)) * $(128)) + ((((r98_group % $(8)) * $(16384)) + (r98_row[:, None] * $(128)) + r98_lane[None, :]) % $(128))), dtype=tl.float32)
      r98_right = tl.load($(candidate_vto_copy_37_ptr) + (((((((r98_group % $(8)) * $(24576)) + (r98_lane[:, None] * $(192)) + r98_col[None, :]) / $(24576)) % $(8)) * $(24576)) + ((((((r98_group % $(8)) * $(24576)) + (r98_lane[:, None] * $(192)) + r98_col[None, :]) / $(192)) % $(128)) * $(192)) + ((((r98_group % $(8)) * $(24576)) + (r98_lane[:, None] * $(192)) + r98_col[None, :]) % $(192))), dtype=tl.float32)
      r98_value = tl.dot(r98_left, r98_right)
      r98_value = r98_value.to(tl.float32)
      tl.store($(candidate_vbmm_12_ptr) + ((r98_group * $(24576)) + (r98_row[:, None] * $(192)) + r98_col[None, :]), r98_value)
    }
    r99_z = tl.arange($(196608))
    r99_value = (tl.load($(candidate_vbmm_12_ptr) + ((((r99_z / $(24576)) % $(8)) * $(24576)) + (((r99_z / $(192)) % $(128)) * $(192)) + (r99_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r99_value = r99_value.to(tl.float32)
    r99_value = tl.broadcast_to(r99_value, [$(196608)])
    tl.store($(candidate_vmul_26_ptr) + r99_z, r99_value)
    r100_z = tl.arange($(196608))
    r100_value = tl.load($(candidate_vmul_26_ptr) + ((((r100_z / $(192)) % $(8)) * $(24576)) + (((r100_z / $(1536)) % $(128)) * $(192)) + (r100_z % $(192))), dtype=tl.float32)
    r100_value = r100_value.to(tl.bfloat16)
    r100_value = tl.broadcast_to(r100_value, [$(196608)])
    tl.store($(candidate_vto_copy_41_ptr) + r100_z, r100_value)
    r101_row = tl.arange($(128))
    r101_col = tl.arange($(192))
    r101_lane = tl.arange($(128))
    for r101_group in range($(8)) {
      r101_left = tl.load($(candidate_vmul_25_ptr) + (((((((r101_group % $(8)) * $(16384)) + (r101_row[:, None] * $(128)) + r101_lane[None, :]) / $(16384)) % $(8)) * $(16384)) + (((((r101_group % $(8)) * $(16384)) + (r101_row[:, None] * $(128)) + r101_lane[None, :]) % $(128)) * $(128)) + (((((r101_group % $(8)) * $(16384)) + (r101_row[:, None] * $(128)) + r101_lane[None, :]) / $(128)) % $(128))), dtype=tl.float32)
      r101_right = tl.load($(candidate_vto_copy_36_ptr) + (((((((r101_group % $(8)) * $(24576)) + (r101_lane[:, None] * $(192)) + r101_col[None, :]) / $(24576)) % $(8)) * $(24576)) + ((((((r101_group % $(8)) * $(24576)) + (r101_lane[:, None] * $(192)) + r101_col[None, :]) / $(192)) % $(128)) * $(192)) + ((((r101_group % $(8)) * $(24576)) + (r101_lane[:, None] * $(192)) + r101_col[None, :]) % $(192))), dtype=tl.float32)
      r101_value = tl.dot(r101_left, r101_right)
      r101_value = r101_value.to(tl.float32)
      tl.store($(candidate_vbmm_13_ptr) + ((r101_group * $(24576)) + (r101_row[:, None] * $(192)) + r101_col[None, :]), r101_value)
    }
    r102_z = tl.arange($(196608))
    r102_value = (tl.load($(candidate_vbmm_13_ptr) + ((((r102_z / $(24576)) % $(8)) * $(24576)) + (((r102_z / $(192)) % $(128)) * $(192)) + (r102_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r102_value = r102_value.to(tl.float32)
    r102_value = tl.broadcast_to(r102_value, [$(196608)])
    tl.store($(candidate_vmul_27_ptr) + r102_z, r102_value)
    r103_z = tl.arange($(196608))
    r103_value = tl.load($(candidate_vmul_27_ptr) + ((((r103_z / $(192)) % $(8)) * $(24576)) + (((r103_z / $(1536)) % $(128)) * $(192)) + (r103_z % $(192))), dtype=tl.float32)
    r103_value = r103_value.to(tl.bfloat16)
    r103_value = tl.broadcast_to(r103_value, [$(196608)])
    tl.store($(candidate_vto_copy_42_ptr) + r103_z, r103_value)
    r104_row = tl.arange($(128))
    r104_col = tl.arange($(128))
    r104_lane = tl.arange($(128))
    for r104_group in range($(8)) {
      r104_left = tl.load($(candidate_vexp_ptr) + (((((((r104_group % $(8)) * $(16384)) + (r104_row[:, None] * $(128)) + r104_lane[None, :]) / $(16384)) % $(8)) * $(16384)) + (((((r104_group % $(8)) * $(16384)) + (r104_row[:, None] * $(128)) + r104_lane[None, :]) % $(128)) * $(128)) + (((((r104_group % $(8)) * $(16384)) + (r104_row[:, None] * $(128)) + r104_lane[None, :]) / $(128)) % $(128))), dtype=tl.float32)
      r104_right = tl.load($(candidate_vto_copy_39_ptr) + (((((((r104_group % $(8)) * $(16384)) + (r104_lane[:, None] * $(128)) + r104_col[None, :]) / $(16384)) % $(8)) * $(16384)) + ((((((r104_group % $(8)) * $(16384)) + (r104_lane[:, None] * $(128)) + r104_col[None, :]) / $(128)) % $(128)) * $(128)) + ((((r104_group % $(8)) * $(16384)) + (r104_lane[:, None] * $(128)) + r104_col[None, :]) % $(128))), dtype=tl.float32)
      r104_value = tl.dot(r104_left, r104_right)
      r104_value = r104_value.to(tl.float32)
      tl.store($(candidate_vbmm_14_ptr) + ((r104_group * $(16384)) + (r104_row[:, None] * $(128)) + r104_col[None, :]), r104_value)
    }
    r105_z = tl.arange($(131072))
    r105_value = tl.load($(candidate_vbmm_14_ptr) + ((((r105_z / $(128)) % $(8)) * $(16384)) + (((r105_z / $(1024)) % $(128)) * $(128)) + (r105_z % $(128))), dtype=tl.float32)
    r105_value = r105_value.to(tl.bfloat16)
    r105_value = tl.broadcast_to(r105_value, [$(131072)])
    tl.store($(candidate_vto_copy_43_ptr) + r105_z, r105_value)
    r106_z = tl.arange($(196608))
    r106_value = tl.load($(candidate_vcat_3_ptr) + ((((r106_z / $(192)) % $(128)) * $(1536)) + (((r106_z / $(24576)) % $(8)) * $(192)) + (r106_z % $(192)) + $(196608)), dtype=tl.bfloat16)
    r106_value = r106_value.to(tl.float32)
    r106_value = tl.broadcast_to(r106_value, [$(196608)])
    tl.store($(candidate_vto_copy_44_ptr) + r106_z, r106_value)
    r107_z = tl.arange($(196608))
    r107_value = tl.load($(candidate_vcat_4_ptr) + ((((r107_z / $(192)) % $(128)) * $(1536)) + (((r107_z / $(24576)) % $(8)) * $(192)) + (r107_z % $(192)) + $(196608)), dtype=tl.bfloat16)
    r107_value = r107_value.to(tl.float32)
    r107_value = tl.broadcast_to(r107_value, [$(196608)])
    tl.store($(candidate_vto_copy_45_ptr) + r107_z, r107_value)
    r108_z = tl.arange($(131072))
    r108_value = tl.load($(candidate_vbmm_5_ptr) + ((((r108_z / $(128)) % $(128)) * $(1024)) + (((r108_z / $(16384)) % $(8)) * $(128)) + (r108_z % $(128)) + $(131072)), dtype=tl.bfloat16)
    r108_value = r108_value.to(tl.float32)
    r108_value = tl.broadcast_to(r108_value, [$(131072)])
    tl.store($(candidate_vto_copy_46_ptr) + r108_z, r108_value)
    r109_z = tl.arange($(131072))
    r109_value = tl.load($(candidate_vmm_8_ptr) + ((((r109_z / $(128)) % $(128)) * $(1024)) + (((r109_z / $(16384)) % $(8)) * $(128)) + (r109_z % $(128)) + $(131072)), dtype=tl.bfloat16)
    r109_value = r109_value.to(tl.float32)
    r109_value = tl.broadcast_to(r109_value, [$(131072)])
    tl.store($(candidate_vto_copy_47_ptr) + r109_z, r109_value)
    r110_z = tl.arange($(131072))
    r110_value = tl.load($(candidate_vcat_5_ptr) + ((((r110_z / $(128)) % $(128)) * $(1024)) + (((r110_z / $(16384)) % $(8)) * $(128)) + (r110_z % $(128)) + $(131072)), dtype=tl.bfloat16)
    r110_value = r110_value.to(tl.float32)
    r110_value = tl.broadcast_to(r110_value, [$(131072)])
    tl.store($(candidate_vto_copy_48_ptr) + r110_z, r110_value)
    r111_row = tl.arange($(128))
    r111_col = tl.arange($(128))
    r111_lane = tl.arange($(192))
    for r111_group in range($(8)) {
      r111_left = tl.load($(candidate_vto_copy_44_ptr) + (((((((r111_group % $(8)) * $(24576)) + (r111_row[:, None] * $(192)) + r111_lane[None, :]) / $(24576)) % $(8)) * $(24576)) + ((((((r111_group % $(8)) * $(24576)) + (r111_row[:, None] * $(192)) + r111_lane[None, :]) / $(192)) % $(128)) * $(192)) + ((((r111_group % $(8)) * $(24576)) + (r111_row[:, None] * $(192)) + r111_lane[None, :]) % $(192))), dtype=tl.float32)
      r111_right = tl.load($(candidate_vto_copy_45_ptr) + (((((((r111_group % $(8)) * $(24576)) + (r111_lane[:, None] * $(128)) + r111_col[None, :]) / $(24576)) % $(8)) * $(24576)) + (((((r111_group % $(8)) * $(24576)) + (r111_lane[:, None] * $(128)) + r111_col[None, :]) % $(128)) * $(192)) + (((((r111_group % $(8)) * $(24576)) + (r111_lane[:, None] * $(128)) + r111_col[None, :]) / $(128)) % $(192))), dtype=tl.float32)
      r111_value = tl.dot(r111_left, r111_right)
      r111_value = r111_value.to(tl.float32)
      tl.store($(candidate_vbmm_15_ptr) + ((r111_group * $(16384)) + (r111_row[:, None] * $(128)) + r111_col[None, :]), r111_value)
    }
    r112_z = tl.arange($(131072))
    r112_value = (tl.load($(candidate_vbmm_15_ptr) + ((((r112_z / $(16384)) % $(8)) * $(16384)) + (((r112_z / $(128)) % $(128)) * $(128)) + (r112_z % $(128))), dtype=tl.float32) * 0.07216878364870322)
    r112_value = r112_value.to(tl.float32)
    r112_value = tl.broadcast_to(r112_value, [$(131072)])
    tl.store($(candidate_vmul_28_ptr) + r112_z, r112_value)
    r113_z = tl.arange($(128))
    r113_value = (tl.cast(r113_z, tl.int64) + tl.full([], $(128), dtype=tl.int64))
    r113_value = tl.broadcast_to(r113_value, [$(128)])
    tl.store($(candidate_vadd_13_ptr) + r113_z, r113_value)
    r114_z = tl.arange($(128))
    r114_value = (tl.load($(candidate_vadd_13_ptr) + r114_z, dtype=tl.int64) - tl.full([], $(128), dtype=tl.int64))
    r114_value = tl.broadcast_to(r114_value, [$(128)])
    tl.store($(candidate_vsub_9_ptr) + r114_z, r114_value)
    r115_z = tl.arange($(16384))
    r115_value = (tl.cast((r115_z % $(128)), tl.int64) > tl.load($(candidate_vsub_9_ptr) + ((r115_z / $(128)) % $(128)), dtype=tl.int64))
    r115_value = tl.broadcast_to(r115_value, [$(16384)])
    tl.store($(candidate_vgt_5_ptr) + r115_z, r115_value)
    r116_z = tl.arange($(131072))
    r116_value = tl.where(tl.load($(candidate_vgt_5_ptr) + ((((r116_z / $(128)) % $(128)) * $(128)) + (r116_z % $(128))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_28_ptr) + ((((r116_z / $(16384)) % $(8)) * $(16384)) + (((r116_z / $(128)) % $(128)) * $(128)) + (r116_z % $(128))), dtype=tl.float32))
    r116_value = r116_value.to(tl.float32)
    r116_value = tl.broadcast_to(r116_value, [$(131072)])
    tl.store($(candidate_vmasked_fill_7_ptr) + r116_z, r116_value)
    r117_z = tl.arange($(131072))
    r117_value = (tl.load($(candidate_vmasked_fill_7_ptr) + ((((r117_z / $(16384)) % $(8)) * $(16384)) + (((r117_z / $(128)) % $(128)) * $(128)) + (r117_z % $(128))), dtype=tl.float32) - tl.load($(candidate_vcat_6_ptr) + ((((r117_z / $(128)) % $(128)) * $(8)) + ((r117_z / $(16384)) % $(8)) + $(1024)), dtype=tl.float32))
    r117_value = r117_value.to(tl.float32)
    r117_value = tl.broadcast_to(r117_value, [$(131072)])
    tl.store($(candidate_vsub_10_ptr) + r117_z, r117_value)
    r118_z = tl.arange($(131072))
    r118_value = tl.exp(tl.load($(candidate_vsub_10_ptr) + ((((r118_z / $(16384)) % $(8)) * $(16384)) + (((r118_z / $(128)) % $(128)) * $(128)) + (r118_z % $(128))), dtype=tl.float32))
    r118_value = r118_value.to(tl.float32)
    r118_value = tl.broadcast_to(r118_value, [$(131072)])
    tl.store($(candidate_vexp_1_ptr) + r118_z, r118_value)
    r119_z = tl.arange($(131072))
    r119_value = (tl.load($(candidate_vto_copy_47_ptr) + ((((r119_z / $(16384)) % $(8)) * $(16384)) + (((r119_z / $(128)) % $(128)) * $(128)) + (r119_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_48_ptr) + ((((r119_z / $(16384)) % $(8)) * $(16384)) + (((r119_z / $(128)) % $(128)) * $(128)) + (r119_z % $(128))), dtype=tl.float32))
    r119_value = r119_value.to(tl.float32)
    r119_value = tl.broadcast_to(r119_value, [$(131072)])
    tl.store($(candidate_vmul_29_ptr) + r119_z, r119_value)
    r120_z = tl.arange($(1024))
    r120_lane = tl.arange($(128))
    r120_value = tl.sum(tl.load($(candidate_vmul_29_ptr) + ((((r120_z[:, None] / $(128)) % $(8)) * $(16384)) + ((r120_z[:, None] % $(128)) * $(128)) + r120_lane[None, :]), dtype=tl.float32), axis=1)
    r120_value = r120_value.to(tl.float32)
    r120_value = tl.broadcast_to(r120_value, [$(1024)])
    tl.store($(candidate_vsum_2_ptr) + r120_z, r120_value)
    r121_row = tl.arange($(128))
    r121_col = tl.arange($(128))
    r121_lane = tl.arange($(128))
    for r121_group in range($(8)) {
      r121_left = tl.load($(candidate_vto_copy_47_ptr) + (((((((r121_group % $(8)) * $(16384)) + (r121_row[:, None] * $(128)) + r121_lane[None, :]) / $(16384)) % $(8)) * $(16384)) + ((((((r121_group % $(8)) * $(16384)) + (r121_row[:, None] * $(128)) + r121_lane[None, :]) / $(128)) % $(128)) * $(128)) + ((((r121_group % $(8)) * $(16384)) + (r121_row[:, None] * $(128)) + r121_lane[None, :]) % $(128))), dtype=tl.float32)
      r121_right = tl.load($(candidate_vto_copy_46_ptr) + (((((((r121_group % $(8)) * $(16384)) + (r121_lane[:, None] * $(128)) + r121_col[None, :]) / $(16384)) % $(8)) * $(16384)) + (((((r121_group % $(8)) * $(16384)) + (r121_lane[:, None] * $(128)) + r121_col[None, :]) % $(128)) * $(128)) + (((((r121_group % $(8)) * $(16384)) + (r121_lane[:, None] * $(128)) + r121_col[None, :]) / $(128)) % $(128))), dtype=tl.float32)
      r121_value = tl.dot(r121_left, r121_right)
      r121_value = r121_value.to(tl.float32)
      tl.store($(candidate_vbmm_16_ptr) + ((r121_group * $(16384)) + (r121_row[:, None] * $(128)) + r121_col[None, :]), r121_value)
    }
    r122_z = tl.arange($(131072))
    r122_value = (tl.load($(candidate_vbmm_16_ptr) + ((((r122_z / $(16384)) % $(8)) * $(16384)) + (((r122_z / $(128)) % $(128)) * $(128)) + (r122_z % $(128))), dtype=tl.float32) - tl.load($(candidate_vsum_2_ptr) + ((((r122_z / $(16384)) % $(8)) * $(128)) + ((r122_z / $(128)) % $(128))), dtype=tl.float32))
    r122_value = r122_value.to(tl.float32)
    r122_value = tl.broadcast_to(r122_value, [$(131072)])
    tl.store($(candidate_vsub_11_ptr) + r122_z, r122_value)
    r123_z = tl.arange($(131072))
    r123_value = (tl.load($(candidate_vexp_1_ptr) + ((((r123_z / $(16384)) % $(8)) * $(16384)) + (((r123_z / $(128)) % $(128)) * $(128)) + (r123_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vsub_11_ptr) + ((((r123_z / $(16384)) % $(8)) * $(16384)) + (((r123_z / $(128)) % $(128)) * $(128)) + (r123_z % $(128))), dtype=tl.float32))
    r123_value = r123_value.to(tl.float32)
    r123_value = tl.broadcast_to(r123_value, [$(131072)])
    tl.store($(candidate_vmul_30_ptr) + r123_z, r123_value)
    r124_row = tl.arange($(128))
    r124_col = tl.arange($(192))
    r124_lane = tl.arange($(128))
    for r124_group in range($(8)) {
      r124_left = tl.load($(candidate_vmul_30_ptr) + (((((((r124_group % $(8)) * $(16384)) + (r124_row[:, None] * $(128)) + r124_lane[None, :]) / $(16384)) % $(8)) * $(16384)) + ((((((r124_group % $(8)) * $(16384)) + (r124_row[:, None] * $(128)) + r124_lane[None, :]) / $(128)) % $(128)) * $(128)) + ((((r124_group % $(8)) * $(16384)) + (r124_row[:, None] * $(128)) + r124_lane[None, :]) % $(128))), dtype=tl.float32)
      r124_right = tl.load($(candidate_vto_copy_45_ptr) + (((((((r124_group % $(8)) * $(24576)) + (r124_lane[:, None] * $(192)) + r124_col[None, :]) / $(24576)) % $(8)) * $(24576)) + ((((((r124_group % $(8)) * $(24576)) + (r124_lane[:, None] * $(192)) + r124_col[None, :]) / $(192)) % $(128)) * $(192)) + ((((r124_group % $(8)) * $(24576)) + (r124_lane[:, None] * $(192)) + r124_col[None, :]) % $(192))), dtype=tl.float32)
      r124_value = tl.dot(r124_left, r124_right)
      r124_value = r124_value.to(tl.float32)
      tl.store($(candidate_vbmm_17_ptr) + ((r124_group * $(24576)) + (r124_row[:, None] * $(192)) + r124_col[None, :]), r124_value)
    }
    r125_z = tl.arange($(196608))
    r125_value = (tl.load($(candidate_vbmm_17_ptr) + ((((r125_z / $(24576)) % $(8)) * $(24576)) + (((r125_z / $(192)) % $(128)) * $(192)) + (r125_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r125_value = r125_value.to(tl.float32)
    r125_value = tl.broadcast_to(r125_value, [$(196608)])
    tl.store($(candidate_vmul_31_ptr) + r125_z, r125_value)
    r126_z = tl.arange($(196608))
    r126_value = tl.load($(candidate_vmul_31_ptr) + ((((r126_z / $(192)) % $(8)) * $(24576)) + (((r126_z / $(1536)) % $(128)) * $(192)) + (r126_z % $(192))), dtype=tl.float32)
    r126_value = r126_value.to(tl.bfloat16)
    r126_value = tl.broadcast_to(r126_value, [$(196608)])
    tl.store($(candidate_vto_copy_49_ptr) + r126_z, r126_value)
    r127_row = tl.arange($(128))
    r127_col = tl.arange($(192))
    r127_lane = tl.arange($(128))
    for r127_group in range($(8)) {
      r127_left = tl.load($(candidate_vmul_30_ptr) + (((((((r127_group % $(8)) * $(16384)) + (r127_row[:, None] * $(128)) + r127_lane[None, :]) / $(16384)) % $(8)) * $(16384)) + (((((r127_group % $(8)) * $(16384)) + (r127_row[:, None] * $(128)) + r127_lane[None, :]) % $(128)) * $(128)) + (((((r127_group % $(8)) * $(16384)) + (r127_row[:, None] * $(128)) + r127_lane[None, :]) / $(128)) % $(128))), dtype=tl.float32)
      r127_right = tl.load($(candidate_vto_copy_44_ptr) + (((((((r127_group % $(8)) * $(24576)) + (r127_lane[:, None] * $(192)) + r127_col[None, :]) / $(24576)) % $(8)) * $(24576)) + ((((((r127_group % $(8)) * $(24576)) + (r127_lane[:, None] * $(192)) + r127_col[None, :]) / $(192)) % $(128)) * $(192)) + ((((r127_group % $(8)) * $(24576)) + (r127_lane[:, None] * $(192)) + r127_col[None, :]) % $(192))), dtype=tl.float32)
      r127_value = tl.dot(r127_left, r127_right)
      r127_value = r127_value.to(tl.float32)
      tl.store($(candidate_vbmm_18_ptr) + ((r127_group * $(24576)) + (r127_row[:, None] * $(192)) + r127_col[None, :]), r127_value)
    }
    r128_z = tl.arange($(196608))
    r128_value = (tl.load($(candidate_vbmm_18_ptr) + ((((r128_z / $(24576)) % $(8)) * $(24576)) + (((r128_z / $(192)) % $(128)) * $(192)) + (r128_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r128_value = r128_value.to(tl.float32)
    r128_value = tl.broadcast_to(r128_value, [$(196608)])
    tl.store($(candidate_vmul_32_ptr) + r128_z, r128_value)
    r129_z = tl.arange($(196608))
    r129_value = tl.load($(candidate_vmul_32_ptr) + ((((r129_z / $(192)) % $(8)) * $(24576)) + (((r129_z / $(1536)) % $(128)) * $(192)) + (r129_z % $(192))), dtype=tl.float32)
    r129_value = r129_value.to(tl.bfloat16)
    r129_value = tl.broadcast_to(r129_value, [$(196608)])
    tl.store($(candidate_vto_copy_50_ptr) + r129_z, r129_value)
    r130_row = tl.arange($(128))
    r130_col = tl.arange($(128))
    r130_lane = tl.arange($(128))
    for r130_group in range($(8)) {
      r130_left = tl.load($(candidate_vexp_1_ptr) + (((((((r130_group % $(8)) * $(16384)) + (r130_row[:, None] * $(128)) + r130_lane[None, :]) / $(16384)) % $(8)) * $(16384)) + (((((r130_group % $(8)) * $(16384)) + (r130_row[:, None] * $(128)) + r130_lane[None, :]) % $(128)) * $(128)) + (((((r130_group % $(8)) * $(16384)) + (r130_row[:, None] * $(128)) + r130_lane[None, :]) / $(128)) % $(128))), dtype=tl.float32)
      r130_right = tl.load($(candidate_vto_copy_47_ptr) + (((((((r130_group % $(8)) * $(16384)) + (r130_lane[:, None] * $(128)) + r130_col[None, :]) / $(16384)) % $(8)) * $(16384)) + ((((((r130_group % $(8)) * $(16384)) + (r130_lane[:, None] * $(128)) + r130_col[None, :]) / $(128)) % $(128)) * $(128)) + ((((r130_group % $(8)) * $(16384)) + (r130_lane[:, None] * $(128)) + r130_col[None, :]) % $(128))), dtype=tl.float32)
      r130_value = tl.dot(r130_left, r130_right)
      r130_value = r130_value.to(tl.float32)
      tl.store($(candidate_vbmm_19_ptr) + ((r130_group * $(16384)) + (r130_row[:, None] * $(128)) + r130_col[None, :]), r130_value)
    }
    r131_z = tl.arange($(131072))
    r131_value = tl.load($(candidate_vbmm_19_ptr) + ((((r131_z / $(128)) % $(8)) * $(16384)) + (((r131_z / $(1024)) % $(128)) * $(128)) + (r131_z % $(128))), dtype=tl.float32)
    r131_value = r131_value.to(tl.bfloat16)
    r131_value = tl.broadcast_to(r131_value, [$(131072)])
    tl.store($(candidate_vto_copy_51_ptr) + r131_z, r131_value)
    r132_z = tl.arange($(393216))
    r132_value = tl.where(((r132_z / $(1536)) % $(256)) < $(128), tl.load($(candidate_vto_copy_41_ptr) + ((((((r132_z / $(1536)) % $(256)) - $(0)) % $(128)) * $(1536)) + (((r132_z / $(192)) % $(8)) * $(192)) + (r132_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vto_copy_49_ptr) + ((((((r132_z / $(1536)) % $(256)) - $(128)) % $(128)) * $(1536)) + (((r132_z / $(192)) % $(8)) * $(192)) + (r132_z % $(192))), dtype=tl.bfloat16))
    r132_value = r132_value.to(tl.bfloat16)
    r132_value = tl.broadcast_to(r132_value, [$(393216)])
    tl.store($(candidate_vcat_7_ptr) + r132_z, r132_value)
    r133_z = tl.arange($(393216))
    r133_value = tl.where(((r133_z / $(1536)) % $(256)) < $(128), tl.load($(candidate_vto_copy_42_ptr) + ((((((r133_z / $(1536)) % $(256)) - $(0)) % $(128)) * $(1536)) + (((r133_z / $(192)) % $(8)) * $(192)) + (r133_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vto_copy_50_ptr) + ((((((r133_z / $(1536)) % $(256)) - $(128)) % $(128)) * $(1536)) + (((r133_z / $(192)) % $(8)) * $(192)) + (r133_z % $(192))), dtype=tl.bfloat16))
    r133_value = r133_value.to(tl.bfloat16)
    r133_value = tl.broadcast_to(r133_value, [$(393216)])
    tl.store($(candidate_vcat_8_ptr) + r133_z, r133_value)
    r134_z = tl.arange($(262144))
    r134_value = tl.where(((r134_z / $(1024)) % $(256)) < $(128), tl.load($(candidate_vto_copy_43_ptr) + ((((((r134_z / $(1024)) % $(256)) - $(0)) % $(128)) * $(1024)) + (((r134_z / $(128)) % $(8)) * $(128)) + (r134_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vto_copy_51_ptr) + ((((((r134_z / $(1024)) % $(256)) - $(128)) % $(128)) * $(1024)) + (((r134_z / $(128)) % $(8)) * $(128)) + (r134_z % $(128))), dtype=tl.bfloat16))
    r134_value = r134_value.to(tl.bfloat16)
    r134_value = tl.broadcast_to(r134_value, [$(262144)])
    tl.store($(candidate_vcat_9_ptr) + r134_z, r134_value)
    r135_z = tl.arange($(16384))
    r135_lane = tl.arange($(8))
    r135_value = tl.sum(tl.load($(candidate_vcat_8_ptr) + ((((r135_z[:, None] / $(64)) % $(256)) * $(1536)) + (r135_lane[None, :] * $(192)) + (r135_z[:, None] % $(64)) + $(128)), dtype=tl.bfloat16), axis=1)
    r135_value = r135_value.to(tl.bfloat16)
    r135_value = tl.broadcast_to(r135_value, [$(16384)])
    tl.store($(candidate_vsum_3_ptr) + r135_z, r135_value)
    r136_row = tl.arange($(256))
    r136_col = tl.arange($(512))
    r136_lane = tl.arange($(1024))
    for r136_group in range($(1)) {
      r136_left = tl.load($(candidate_vcat_8_ptr) + ((((((r136_row[:, None] * $(1024)) + r136_lane[None, :]) / $(1024)) % $(256)) * $(1536)) + ((((r136_row[:, None] * $(1024)) + r136_lane[None, :]) % $(8)) * $(192)) + ((((r136_row[:, None] * $(1024)) + r136_lane[None, :]) / $(8)) % $(128))), dtype=tl.bfloat16)
      r136_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r136_lane[:, None] * $(512)) + r136_col[None, :]) / $(512)) % $(8)) * $(131072)) + (((((r136_lane[:, None] * $(512)) + r136_col[None, :]) / $(4096)) % $(128)) * $(512)) + (((r136_lane[:, None] * $(512)) + r136_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r136_value = tl.dot(r136_left, r136_right)
      r136_value = r136_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_20_ptr) + ((r136_group * $(131072)) + (r136_row[:, None] * $(512)) + r136_col[None, :]), r136_value)
    }
    r137_row = tl.arange($(256))
    r137_col = tl.arange($(512))
    r137_lane = tl.arange($(1024))
    for r137_group in range($(1)) {
      r137_left = tl.load($(candidate_vcat_9_ptr) + ((((((r137_row[:, None] * $(1024)) + r137_lane[None, :]) / $(1024)) % $(256)) * $(1024)) + ((((r137_row[:, None] * $(1024)) + r137_lane[None, :]) % $(8)) * $(128)) + ((((r137_row[:, None] * $(1024)) + r137_lane[None, :]) / $(8)) % $(128))), dtype=tl.bfloat16)
      r137_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r137_lane[:, None] * $(512)) + r137_col[None, :]) / $(512)) % $(8)) * $(131072)) + (((((r137_lane[:, None] * $(512)) + r137_col[None, :]) / $(4096)) % $(128)) * $(512)) + (((r137_lane[:, None] * $(512)) + r137_col[None, :]) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r137_value = tl.dot(r137_left, r137_right)
      r137_value = r137_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_21_ptr) + ((r137_group * $(131072)) + (r137_row[:, None] * $(512)) + r137_col[None, :]), r137_value)
    }
    r138_z = tl.arange($(131072))
    r138_value = (tl.load($(candidate_vbmm_20_ptr) + ((((r138_z / $(512)) % $(256)) * $(512)) + (r138_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vbmm_21_ptr) + ((((r138_z / $(512)) % $(256)) * $(512)) + (r138_z % $(512))), dtype=tl.bfloat16))
    r138_value = r138_value.to(tl.bfloat16)
    r138_value = tl.broadcast_to(r138_value, [$(131072)])
    tl.store($(candidate_vadd_14_ptr) + r138_z, r138_value)
    r139_z = tl.arange($(8192))
    r139_value = (tl.load($(candidate_vsum_3_ptr) + ((((r139_z / $(4096)) % $(2)) * $(8192)) + (((r139_z / $(32)) % $(128)) * $(64)) + (r139_z % $(32)) + $(32)), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r139_value = r139_value.to(tl.bfloat16)
    r139_value = tl.broadcast_to(r139_value, [$(8192)])
    tl.store($(candidate_vmul_33_ptr) + r139_z, r139_value)
    r140_z = tl.arange($(8192))
    r140_value = (tl.load($(candidate_vsum_3_ptr) + ((((r140_z / $(4096)) % $(2)) * $(8192)) + (((r140_z / $(32)) % $(128)) * $(64)) + (r140_z % $(32)) + $(32)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r140_value = r140_value.to(tl.bfloat16)
    r140_value = tl.broadcast_to(r140_value, [$(8192)])
    tl.store($(candidate_vmul_34_ptr) + r140_z, r140_value)
    r141_z = tl.arange($(8192))
    r141_value = -tl.load($(candidate_vsum_3_ptr) + ((((r141_z / $(4096)) % $(2)) * $(8192)) + (((r141_z / $(32)) % $(128)) * $(64)) + (r141_z % $(32))), dtype=tl.bfloat16)
    r141_value = r141_value.to(tl.bfloat16)
    r141_value = tl.broadcast_to(r141_value, [$(8192)])
    tl.store($(candidate_vneg_ptr) + r141_z, r141_value)
    r142_z = tl.arange($(8192))
    r142_value = (tl.load($(candidate_vneg_ptr) + ((((r142_z / $(4096)) % $(2)) * $(4096)) + (((r142_z / $(32)) % $(128)) * $(32)) + (r142_z % $(32))), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r142_value = r142_value.to(tl.bfloat16)
    r142_value = tl.broadcast_to(r142_value, [$(8192)])
    tl.store($(candidate_vmul_35_ptr) + r142_z, r142_value)
    r143_z = tl.arange($(8192))
    r143_value = (tl.load($(candidate_vmul_34_ptr) + ((((r143_z / $(4096)) % $(2)) * $(4096)) + (((r143_z / $(32)) % $(128)) * $(32)) + (r143_z % $(32))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_35_ptr) + ((((r143_z / $(4096)) % $(2)) * $(4096)) + (((r143_z / $(32)) % $(128)) * $(32)) + (r143_z % $(32))), dtype=tl.bfloat16))
    r143_value = r143_value.to(tl.bfloat16)
    r143_value = tl.broadcast_to(r143_value, [$(8192)])
    tl.store($(candidate_vadd_15_ptr) + r143_z, r143_value)
    r144_z = tl.arange($(8192))
    r144_value = (tl.load($(candidate_vsum_3_ptr) + ((((r144_z / $(4096)) % $(2)) * $(8192)) + (((r144_z / $(32)) % $(128)) * $(64)) + (r144_z % $(32))), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r144_value = r144_value.to(tl.bfloat16)
    r144_value = tl.broadcast_to(r144_value, [$(8192)])
    tl.store($(candidate_vmul_36_ptr) + r144_z, r144_value)
    r145_z = tl.arange($(8192))
    r145_value = (tl.load($(candidate_vmul_33_ptr) + ((((r145_z / $(4096)) % $(2)) * $(4096)) + (((r145_z / $(32)) % $(128)) * $(32)) + (r145_z % $(32))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_36_ptr) + ((((r145_z / $(4096)) % $(2)) * $(4096)) + (((r145_z / $(32)) % $(128)) * $(32)) + (r145_z % $(32))), dtype=tl.bfloat16))
    r145_value = r145_value.to(tl.bfloat16)
    r145_value = tl.broadcast_to(r145_value, [$(8192)])
    tl.store($(candidate_vadd_16_ptr) + r145_z, r145_value)
    r146_z = tl.arange($(65536))
    r146_value = (tl.load($(candidate_vcat_7_ptr) + ((((r146_z / $(32768)) % $(2)) * $(196608)) + (((r146_z / $(32)) % $(128)) * $(1536)) + (((r146_z / $(4096)) % $(8)) * $(192)) + (r146_z % $(32)) + $(160)), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r146_value = r146_value.to(tl.bfloat16)
    r146_value = tl.broadcast_to(r146_value, [$(65536)])
    tl.store($(candidate_vmul_37_ptr) + r146_z, r146_value)
    r147_z = tl.arange($(65536))
    r147_value = (tl.load($(candidate_vcat_7_ptr) + ((((r147_z / $(32768)) % $(2)) * $(196608)) + (((r147_z / $(32)) % $(128)) * $(1536)) + (((r147_z / $(4096)) % $(8)) * $(192)) + (r147_z % $(32)) + $(160)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r147_value = r147_value.to(tl.bfloat16)
    r147_value = tl.broadcast_to(r147_value, [$(65536)])
    tl.store($(candidate_vmul_38_ptr) + r147_z, r147_value)
    r148_z = tl.arange($(65536))
    r148_value = -tl.load($(candidate_vcat_7_ptr) + ((((r148_z / $(32768)) % $(2)) * $(196608)) + (((r148_z / $(32)) % $(128)) * $(1536)) + (((r148_z / $(4096)) % $(8)) * $(192)) + (r148_z % $(32)) + $(128)), dtype=tl.bfloat16)
    r148_value = r148_value.to(tl.bfloat16)
    r148_value = tl.broadcast_to(r148_value, [$(65536)])
    tl.store($(candidate_vneg_1_ptr) + r148_z, r148_value)
    r149_z = tl.arange($(65536))
    r149_value = (tl.load($(candidate_vneg_1_ptr) + ((((r149_z / $(32768)) % $(2)) * $(32768)) + (((r149_z / $(4096)) % $(8)) * $(4096)) + (((r149_z / $(32)) % $(128)) * $(32)) + (r149_z % $(32))), dtype=tl.bfloat16) * tl.cast(0.0, tl.bfloat16))
    r149_value = r149_value.to(tl.bfloat16)
    r149_value = tl.broadcast_to(r149_value, [$(65536)])
    tl.store($(candidate_vmul_39_ptr) + r149_z, r149_value)
    r150_z = tl.arange($(65536))
    r150_value = (tl.load($(candidate_vmul_38_ptr) + ((((r150_z / $(32768)) % $(2)) * $(32768)) + (((r150_z / $(4096)) % $(8)) * $(4096)) + (((r150_z / $(32)) % $(128)) * $(32)) + (r150_z % $(32))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_39_ptr) + ((((r150_z / $(32768)) % $(2)) * $(32768)) + (((r150_z / $(4096)) % $(8)) * $(4096)) + (((r150_z / $(32)) % $(128)) * $(32)) + (r150_z % $(32))), dtype=tl.bfloat16))
    r150_value = r150_value.to(tl.bfloat16)
    r150_value = tl.broadcast_to(r150_value, [$(65536)])
    tl.store($(candidate_vadd_17_ptr) + r150_z, r150_value)
    r151_z = tl.arange($(65536))
    r151_value = (tl.load($(candidate_vcat_7_ptr) + ((((r151_z / $(32768)) % $(2)) * $(196608)) + (((r151_z / $(32)) % $(128)) * $(1536)) + (((r151_z / $(4096)) % $(8)) * $(192)) + (r151_z % $(32)) + $(128)), dtype=tl.bfloat16) * tl.cast(1.0, tl.bfloat16))
    r151_value = r151_value.to(tl.bfloat16)
    r151_value = tl.broadcast_to(r151_value, [$(65536)])
    tl.store($(candidate_vmul_40_ptr) + r151_z, r151_value)
    r152_z = tl.arange($(65536))
    r152_value = (tl.load($(candidate_vmul_37_ptr) + ((((r152_z / $(32768)) % $(2)) * $(32768)) + (((r152_z / $(4096)) % $(8)) * $(4096)) + (((r152_z / $(32)) % $(128)) * $(32)) + (r152_z % $(32))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_40_ptr) + ((((r152_z / $(32768)) % $(2)) * $(32768)) + (((r152_z / $(4096)) % $(8)) * $(4096)) + (((r152_z / $(32)) % $(128)) * $(32)) + (r152_z % $(32))), dtype=tl.bfloat16))
    r152_value = r152_value.to(tl.bfloat16)
    r152_value = tl.broadcast_to(r152_value, [$(65536)])
    tl.store($(candidate_vadd_18_ptr) + r152_z, r152_value)
    r153_z = tl.arange($(16384))
    r153_value = tl.where((((r153_z % $(64)) >= $(1)) & ((r153_z % $(64)) < $(64)) & (((r153_z % $(64)) - $(1)) % $(2) == $(0))), tl.load($(candidate_vadd_15_ptr) + ((((r153_z / $(8192)) % $(2)) * $(4096)) + (((r153_z / $(64)) % $(128)) * $(32)) + ((((r153_z % $(64)) - $(1)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r153_value = r153_value.to(tl.bfloat16)
    r153_value = tl.broadcast_to(r153_value, [$(16384)])
    tl.store($(candidate_vslice_backward_ptr) + r153_z, r153_value)
    r154_z = tl.arange($(16384))
    r154_value = tl.where((((r154_z % $(64)) >= $(0)) & ((r154_z % $(64)) < $(64)) & (((r154_z % $(64)) - $(0)) % $(2) == $(0))), tl.load($(candidate_vadd_16_ptr) + ((((r154_z / $(8192)) % $(2)) * $(4096)) + (((r154_z / $(64)) % $(128)) * $(32)) + ((((r154_z % $(64)) - $(0)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r154_value = r154_value.to(tl.bfloat16)
    r154_value = tl.broadcast_to(r154_value, [$(16384)])
    tl.store($(candidate_vslice_backward_1_ptr) + r154_z, r154_value)
    r155_z = tl.arange($(16384))
    r155_value = (tl.load($(candidate_vslice_backward_ptr) + ((((r155_z / $(8192)) % $(2)) * $(8192)) + (((r155_z / $(64)) % $(128)) * $(64)) + (r155_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_1_ptr) + ((((r155_z / $(8192)) % $(2)) * $(8192)) + (((r155_z / $(64)) % $(128)) * $(64)) + (r155_z % $(64))), dtype=tl.bfloat16))
    r155_value = r155_value.to(tl.bfloat16)
    r155_value = tl.broadcast_to(r155_value, [$(16384)])
    tl.store($(candidate_vadd_19_ptr) + r155_z, r155_value)
    r156_z = tl.arange($(131072))
    r156_value = tl.where((((r156_z % $(64)) >= $(1)) & ((r156_z % $(64)) < $(64)) & (((r156_z % $(64)) - $(1)) % $(2) == $(0))), tl.load($(candidate_vadd_17_ptr) + ((((r156_z / $(65536)) % $(2)) * $(32768)) + (((r156_z / $(8192)) % $(8)) * $(4096)) + (((r156_z / $(64)) % $(128)) * $(32)) + ((((r156_z % $(64)) - $(1)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r156_value = r156_value.to(tl.bfloat16)
    r156_value = tl.broadcast_to(r156_value, [$(131072)])
    tl.store($(candidate_vslice_backward_2_ptr) + r156_z, r156_value)
    r157_z = tl.arange($(131072))
    r157_value = tl.where((((r157_z % $(64)) >= $(0)) & ((r157_z % $(64)) < $(64)) & (((r157_z % $(64)) - $(0)) % $(2) == $(0))), tl.load($(candidate_vadd_18_ptr) + ((((r157_z / $(65536)) % $(2)) * $(32768)) + (((r157_z / $(8192)) % $(8)) * $(4096)) + (((r157_z / $(64)) % $(128)) * $(32)) + ((((r157_z % $(64)) - $(0)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r157_value = r157_value.to(tl.bfloat16)
    r157_value = tl.broadcast_to(r157_value, [$(131072)])
    tl.store($(candidate_vslice_backward_3_ptr) + r157_z, r157_value)
    r158_z = tl.arange($(131072))
    r158_value = (tl.load($(candidate_vslice_backward_2_ptr) + ((((r158_z / $(65536)) % $(2)) * $(65536)) + (((r158_z / $(8192)) % $(8)) * $(8192)) + (((r158_z / $(64)) % $(128)) * $(64)) + (r158_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_3_ptr) + ((((r158_z / $(65536)) % $(2)) * $(65536)) + (((r158_z / $(8192)) % $(8)) * $(8192)) + (((r158_z / $(64)) % $(128)) * $(64)) + (r158_z % $(64))), dtype=tl.bfloat16))
    r158_value = r158_value.to(tl.bfloat16)
    r158_value = tl.broadcast_to(r158_value, [$(131072)])
    tl.store($(candidate_vadd_20_ptr) + r158_z, r158_value)
    r159_z = tl.arange($(393216))
    r159_value = tl.where((r159_z % $(192)) < $(128), tl.load($(candidate_vcat_7_ptr) + ((((((((r159_z / $(196608)) % $(2)) * $(131072)) + (((r159_z / $(192)) % $(128)) * $(1024)) + (((r159_z / $(24576)) % $(8)) * $(128)) + ((r159_z % $(192)) - $(0))) / $(1024)) % $(256)) * $(1536)) + (((((((r159_z / $(196608)) % $(2)) * $(131072)) + (((r159_z / $(192)) % $(128)) * $(1024)) + (((r159_z / $(24576)) % $(8)) * $(128)) + ((r159_z % $(192)) - $(0))) / $(128)) % $(8)) * $(192)) + (((((r159_z / $(196608)) % $(2)) * $(131072)) + (((r159_z / $(192)) % $(128)) * $(1024)) + (((r159_z / $(24576)) % $(8)) * $(128)) + ((r159_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vadd_20_ptr) + ((((r159_z / $(196608)) % $(2)) * $(65536)) + (((r159_z / $(24576)) % $(8)) * $(8192)) + (((r159_z / $(192)) % $(128)) * $(64)) + (((r159_z % $(192)) - $(128)) % $(64))), dtype=tl.bfloat16))
    r159_value = r159_value.to(tl.bfloat16)
    r159_value = tl.broadcast_to(r159_value, [$(393216)])
    tl.store($(candidate_vcat_11_ptr) + r159_z, r159_value)
    r160_z = tl.arange($(131072))
    r160_value = tl.load($(candidate_vadd_14_ptr) + ((((r160_z / $(65536)) % $(2)) * $(65536)) + (((r160_z / $(512)) % $(128)) * $(512)) + (r160_z % $(512))), dtype=tl.bfloat16)
    r160_value = r160_value.to(tl.float32)
    r160_value = tl.broadcast_to(r160_value, [$(131072)])
    tl.store($(candidate_vto_copy_52_ptr) + r160_z, r160_value)
    r161_z = tl.arange($(131072))
    r161_value = (tl.load($(candidate_vto_copy_52_ptr) + ((((r161_z / $(65536)) % $(2)) * $(65536)) + (((r161_z / $(512)) % $(128)) * $(512)) + (r161_z % $(512))), dtype=tl.float32) * tl.cast(tl.load($(candidate_vto_copy_4_ptr) + (r161_z % $(512)), dtype=tl.bfloat16), tl.float32))
    r161_value = r161_value.to(tl.float32)
    r161_value = tl.broadcast_to(r161_value, [$(131072)])
    tl.store($(candidate_vmul_41_ptr) + r161_z, r161_value)
    r162_z = tl.arange($(131072))
    r162_value = (tl.load($(candidate_vmul_41_ptr) + ((((r162_z / $(65536)) % $(2)) * $(65536)) + (((r162_z / $(512)) % $(128)) * $(512)) + (r162_z % $(512))), dtype=tl.float32) * tl.load($(candidate_vto_copy_18_ptr) + ((((r162_z / $(65536)) % $(2)) * $(65536)) + (((r162_z / $(512)) % $(128)) * $(512)) + (r162_z % $(512))), dtype=tl.float32))
    r162_value = r162_value.to(tl.float32)
    r162_value = tl.broadcast_to(r162_value, [$(131072)])
    tl.store($(candidate_vmul_42_ptr) + r162_z, r162_value)
    r163_z = tl.arange($(131072))
    r163_value = (tl.load($(candidate_vmul_41_ptr) + ((((r163_z / $(65536)) % $(2)) * $(65536)) + (((r163_z / $(512)) % $(128)) * $(512)) + (r163_z % $(512))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_1_ptr) + ((((r163_z / $(65536)) % $(2)) * $(128)) + ((r163_z / $(512)) % $(128))), dtype=tl.float32))
    r163_value = r163_value.to(tl.float32)
    r163_value = tl.broadcast_to(r163_value, [$(131072)])
    tl.store($(candidate_vmul_43_ptr) + r163_z, r163_value)
    r164_z = tl.arange($(256))
    r164_lane = tl.arange($(512))
    r164_value = tl.sum(tl.load($(candidate_vmul_42_ptr) + ((((r164_z[:, None] / $(128)) % $(2)) * $(65536)) + ((r164_z[:, None] % $(128)) * $(512)) + r164_lane[None, :]), dtype=tl.float32), axis=1)
    r164_value = r164_value.to(tl.float32)
    r164_value = tl.broadcast_to(r164_value, [$(256)])
    tl.store($(candidate_vsum_4_ptr) + r164_z, r164_value)
    r165_z = tl.arange($(256))
    r165_value = ((tl.load($(candidate_vrsqrt_1_ptr) + ((((r165_z / $(128)) % $(2)) * $(128)) + (r165_z % $(128))), dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_1_ptr) + ((((r165_z / $(128)) % $(2)) * $(128)) + (r165_z % $(128))), dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_1_ptr) + ((((r165_z / $(128)) % $(2)) * $(128)) + (r165_z % $(128))), dtype=tl.float32)))
    r165_value = r165_value.to(tl.float32)
    r165_value = tl.broadcast_to(r165_value, [$(256)])
    tl.store($(candidate_vpow_3_ptr) + r165_z, r165_value)
    r166_z = tl.arange($(256))
    r166_value = (tl.load($(candidate_vsum_4_ptr) + ((((r166_z / $(128)) % $(2)) * $(128)) + (r166_z % $(128))), dtype=tl.float32) * -0.5)
    r166_value = r166_value.to(tl.float32)
    r166_value = tl.broadcast_to(r166_value, [$(256)])
    tl.store($(candidate_vmul_44_ptr) + r166_z, r166_value)
    r167_z = tl.arange($(256))
    r167_value = (tl.load($(candidate_vmul_44_ptr) + ((((r167_z / $(128)) % $(2)) * $(128)) + (r167_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vpow_3_ptr) + ((((r167_z / $(128)) % $(2)) * $(128)) + (r167_z % $(128))), dtype=tl.float32))
    r167_value = r167_value.to(tl.float32)
    r167_value = tl.broadcast_to(r167_value, [$(256)])
    tl.store($(candidate_vmul_45_ptr) + r167_z, r167_value)
    r168_z = tl.arange($(131072))
    r168_value = (tl.load($(candidate_vmul_45_ptr) + ((((r168_z / $(65536)) % $(2)) * $(128)) + ((r168_z / $(512)) % $(128))), dtype=tl.float32) / 512.0)
    r168_value = r168_value.to(tl.float32)
    r168_value = tl.broadcast_to(r168_value, [$(131072)])
    tl.store($(candidate_vdiv_ptr) + r168_z, r168_value)
    r169_z = tl.arange($(131072))
    r169_value = ((tl.load($(candidate_vto_copy_18_ptr) + ((((r169_z / $(65536)) % $(2)) * $(65536)) + (((r169_z / $(512)) % $(128)) * $(512)) + (r169_z % $(512))), dtype=tl.float32)))
    r169_value = r169_value.to(tl.float32)
    r169_value = tl.broadcast_to(r169_value, [$(131072)])
    tl.store($(candidate_vpow_4_ptr) + r169_z, r169_value)
    r170_z = tl.arange($(131072))
    r170_value = (tl.load($(candidate_vpow_4_ptr) + ((((r170_z / $(65536)) % $(2)) * $(65536)) + (((r170_z / $(512)) % $(128)) * $(512)) + (r170_z % $(512))), dtype=tl.float32) * 2.0)
    r170_value = r170_value.to(tl.float32)
    r170_value = tl.broadcast_to(r170_value, [$(131072)])
    tl.store($(candidate_vmul_46_ptr) + r170_z, r170_value)
    r171_z = tl.arange($(131072))
    r171_value = (tl.load($(candidate_vdiv_ptr) + ((((r171_z / $(65536)) % $(2)) * $(65536)) + (((r171_z / $(512)) % $(128)) * $(512)) + (r171_z % $(512))), dtype=tl.float32) * tl.load($(candidate_vmul_46_ptr) + ((((r171_z / $(65536)) % $(2)) * $(65536)) + (((r171_z / $(512)) % $(128)) * $(512)) + (r171_z % $(512))), dtype=tl.float32))
    r171_value = r171_value.to(tl.float32)
    r171_value = tl.broadcast_to(r171_value, [$(131072)])
    tl.store($(candidate_vmul_47_ptr) + r171_z, r171_value)
    r172_z = tl.arange($(131072))
    r172_value = (tl.load($(candidate_vmul_43_ptr) + ((((r172_z / $(65536)) % $(2)) * $(65536)) + (((r172_z / $(512)) % $(128)) * $(512)) + (r172_z % $(512))), dtype=tl.float32) + tl.load($(candidate_vmul_47_ptr) + ((((r172_z / $(65536)) % $(2)) * $(65536)) + (((r172_z / $(512)) % $(128)) * $(512)) + (r172_z % $(512))), dtype=tl.float32))
    r172_value = r172_value.to(tl.float32)
    r172_value = tl.broadcast_to(r172_value, [$(131072)])
    tl.store($(candidate_vadd_21_ptr) + r172_z, r172_value)
    r173_z = tl.arange($(131072))
    r173_value = tl.load($(candidate_vadd_21_ptr) + ((((r173_z / $(65536)) % $(2)) * $(65536)) + (((r173_z / $(512)) % $(128)) * $(512)) + (r173_z % $(512))), dtype=tl.float32)
    r173_value = r173_value.to(tl.bfloat16)
    r173_value = tl.broadcast_to(r173_value, [$(131072)])
    tl.store($(candidate_vto_copy_53_ptr) + r173_z, r173_value)
    r174_z = tl.arange($(147456))
    r174_value = tl.where((r174_z % $(576)) < $(512), tl.load($(candidate_vto_copy_53_ptr) + ((((r174_z / $(73728)) % $(2)) * $(65536)) + (((r174_z / $(576)) % $(128)) * $(512)) + (((r174_z % $(576)) - $(0)) % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vadd_19_ptr) + ((((((((r174_z / $(73728)) % $(2)) * $(8192)) + (((r174_z / $(576)) % $(128)) * $(64)) + ((r174_z % $(576)) - $(512))) / $(8192)) % $(2)) * $(8192)) + (((((((r174_z / $(73728)) % $(2)) * $(8192)) + (((r174_z / $(576)) % $(128)) * $(64)) + ((r174_z % $(576)) - $(512))) / $(64)) % $(128)) * $(64)) + (((((r174_z / $(73728)) % $(2)) * $(8192)) + (((r174_z / $(576)) % $(128)) * $(64)) + ((r174_z % $(576)) - $(512))) % $(64))), dtype=tl.bfloat16))
    r174_value = r174_value.to(tl.bfloat16)
    r174_value = tl.broadcast_to(r174_value, [$(147456)])
    tl.store($(candidate_vcat_12_ptr) + r174_z, r174_value)
    r175_row = tl.arange($(256))
    r175_col = tl.arange($(512))
    r175_lane = tl.arange($(576))
    r175_left = tl.load($(candidate_vcat_12_ptr) + ((((((r175_row[:, None] * $(576)) + r175_lane[None, :]) / $(73728)) % $(2)) * $(73728)) + (((((r175_row[:, None] * $(576)) + r175_lane[None, :]) / $(576)) % $(128)) * $(576)) + (((r175_row[:, None] * $(576)) + r175_lane[None, :]) % $(576))), dtype=tl.bfloat16)
    r175_right = tl.load($(candidate_vto_copy_3_ptr) + (((r175_lane[:, None] % $(576)) * $(512)) + (r175_col[None, :] % $(512))), dtype=tl.bfloat16)
    r175_value = tl.dot(r175_left, r175_right)
    r175_value = r175_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_9_ptr) + ((r175_row[:, None] * $(512)) + r175_col[None, :]), r175_value)
    r176_row = tl.arange($(256))
    r176_col = tl.arange($(256))
    r176_lane = tl.arange($(1536))
    r176_left = tl.load($(candidate_vcat_11_ptr) + ((((((r176_row[:, None] * $(1536)) + r176_lane[None, :]) / $(196608)) % $(2)) * $(196608)) + (((((((((r176_row[:, None] * $(1536)) + r176_lane[None, :]) / $(196608)) % $(2)) * $(196608)) + (((((r176_row[:, None] * $(1536)) + r176_lane[None, :]) / $(1536)) % $(128)) * $(1536)) + (((r176_row[:, None] * $(1536)) + r176_lane[None, :]) % $(1536))) / $(192)) % $(8)) * $(24576)) + (((((r176_row[:, None] * $(1536)) + r176_lane[None, :]) / $(1536)) % $(128)) * $(192)) + (((((((r176_row[:, None] * $(1536)) + r176_lane[None, :]) / $(196608)) % $(2)) * $(196608)) + (((((r176_row[:, None] * $(1536)) + r176_lane[None, :]) / $(1536)) % $(128)) * $(1536)) + (((r176_row[:, None] * $(1536)) + r176_lane[None, :]) % $(1536))) % $(192))), dtype=tl.bfloat16)
    r176_right = tl.load($(candidate_vto_copy_2_ptr) + (((r176_lane[:, None] % $(1536)) * $(256)) + (r176_col[None, :] % $(256))), dtype=tl.bfloat16)
    r176_value = tl.dot(r176_left, r176_right)
    r176_value = r176_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_10_ptr) + ((r176_row[:, None] * $(256)) + r176_col[None, :]), r176_value)
    r177_z = tl.arange($(65536))
    r177_value = tl.load($(candidate_vmm_10_ptr) + ((((r177_z / $(32768)) % $(2)) * $(32768)) + (((r177_z / $(256)) % $(128)) * $(256)) + (r177_z % $(256))), dtype=tl.bfloat16)
    r177_value = r177_value.to(tl.float32)
    r177_value = tl.broadcast_to(r177_value, [$(65536)])
    tl.store($(candidate_vto_copy_54_ptr) + r177_z, r177_value)
    r178_z = tl.arange($(65536))
    r178_value = (tl.load($(candidate_vto_copy_54_ptr) + ((((r178_z / $(32768)) % $(2)) * $(32768)) + (((r178_z / $(256)) % $(128)) * $(256)) + (r178_z % $(256))), dtype=tl.float32) * tl.cast(tl.load($(candidate_vto_copy_1_ptr) + (r178_z % $(256)), dtype=tl.bfloat16), tl.float32))
    r178_value = r178_value.to(tl.float32)
    r178_value = tl.broadcast_to(r178_value, [$(65536)])
    tl.store($(candidate_vmul_48_ptr) + r178_z, r178_value)
    r179_z = tl.arange($(65536))
    r179_value = (tl.load($(candidate_vmul_48_ptr) + ((((r179_z / $(32768)) % $(2)) * $(32768)) + (((r179_z / $(256)) % $(128)) * $(256)) + (r179_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vto_copy_16_ptr) + ((((r179_z / $(32768)) % $(2)) * $(32768)) + (((r179_z / $(256)) % $(128)) * $(256)) + (r179_z % $(256))), dtype=tl.float32))
    r179_value = r179_value.to(tl.float32)
    r179_value = tl.broadcast_to(r179_value, [$(65536)])
    tl.store($(candidate_vmul_49_ptr) + r179_z, r179_value)
    r180_z = tl.arange($(65536))
    r180_value = (tl.load($(candidate_vmul_48_ptr) + ((((r180_z / $(32768)) % $(2)) * $(32768)) + (((r180_z / $(256)) % $(128)) * $(256)) + (r180_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_ptr) + ((((r180_z / $(32768)) % $(2)) * $(128)) + ((r180_z / $(256)) % $(128))), dtype=tl.float32))
    r180_value = r180_value.to(tl.float32)
    r180_value = tl.broadcast_to(r180_value, [$(65536)])
    tl.store($(candidate_vmul_50_ptr) + r180_z, r180_value)
    r181_z = tl.arange($(256))
    r181_lane = tl.arange($(256))
    r181_value = tl.sum(tl.load($(candidate_vmul_49_ptr) + ((((r181_z[:, None] / $(128)) % $(2)) * $(32768)) + ((r181_z[:, None] % $(128)) * $(256)) + r181_lane[None, :]), dtype=tl.float32), axis=1)
    r181_value = r181_value.to(tl.float32)
    r181_value = tl.broadcast_to(r181_value, [$(256)])
    tl.store($(candidate_vsum_5_ptr) + r181_z, r181_value)
    r182_z = tl.arange($(256))
    r182_value = ((tl.load($(candidate_vrsqrt_ptr) + ((((r182_z / $(128)) % $(2)) * $(128)) + (r182_z % $(128))), dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_ptr) + ((((r182_z / $(128)) % $(2)) * $(128)) + (r182_z % $(128))), dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_ptr) + ((((r182_z / $(128)) % $(2)) * $(128)) + (r182_z % $(128))), dtype=tl.float32)))
    r182_value = r182_value.to(tl.float32)
    r182_value = tl.broadcast_to(r182_value, [$(256)])
    tl.store($(candidate_vpow_5_ptr) + r182_z, r182_value)
    r183_z = tl.arange($(256))
    r183_value = (tl.load($(candidate_vsum_5_ptr) + ((((r183_z / $(128)) % $(2)) * $(128)) + (r183_z % $(128))), dtype=tl.float32) * -0.5)
    r183_value = r183_value.to(tl.float32)
    r183_value = tl.broadcast_to(r183_value, [$(256)])
    tl.store($(candidate_vmul_51_ptr) + r183_z, r183_value)
    r184_z = tl.arange($(256))
    r184_value = (tl.load($(candidate_vmul_51_ptr) + ((((r184_z / $(128)) % $(2)) * $(128)) + (r184_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vpow_5_ptr) + ((((r184_z / $(128)) % $(2)) * $(128)) + (r184_z % $(128))), dtype=tl.float32))
    r184_value = r184_value.to(tl.float32)
    r184_value = tl.broadcast_to(r184_value, [$(256)])
    tl.store($(candidate_vmul_52_ptr) + r184_z, r184_value)
    r185_z = tl.arange($(65536))
    r185_value = (tl.load($(candidate_vmul_52_ptr) + ((((r185_z / $(32768)) % $(2)) * $(128)) + ((r185_z / $(256)) % $(128))), dtype=tl.float32) / 256.0)
    r185_value = r185_value.to(tl.float32)
    r185_value = tl.broadcast_to(r185_value, [$(65536)])
    tl.store($(candidate_vdiv_1_ptr) + r185_z, r185_value)
    r186_z = tl.arange($(65536))
    r186_value = ((tl.load($(candidate_vto_copy_16_ptr) + ((((r186_z / $(32768)) % $(2)) * $(32768)) + (((r186_z / $(256)) % $(128)) * $(256)) + (r186_z % $(256))), dtype=tl.float32)))
    r186_value = r186_value.to(tl.float32)
    r186_value = tl.broadcast_to(r186_value, [$(65536)])
    tl.store($(candidate_vpow_6_ptr) + r186_z, r186_value)
    r187_z = tl.arange($(65536))
    r187_value = (tl.load($(candidate_vpow_6_ptr) + ((((r187_z / $(32768)) % $(2)) * $(32768)) + (((r187_z / $(256)) % $(128)) * $(256)) + (r187_z % $(256))), dtype=tl.float32) * 2.0)
    r187_value = r187_value.to(tl.float32)
    r187_value = tl.broadcast_to(r187_value, [$(65536)])
    tl.store($(candidate_vmul_53_ptr) + r187_z, r187_value)
    r188_z = tl.arange($(65536))
    r188_value = (tl.load($(candidate_vdiv_1_ptr) + ((((r188_z / $(32768)) % $(2)) * $(32768)) + (((r188_z / $(256)) % $(128)) * $(256)) + (r188_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vmul_53_ptr) + ((((r188_z / $(32768)) % $(2)) * $(32768)) + (((r188_z / $(256)) % $(128)) * $(256)) + (r188_z % $(256))), dtype=tl.float32))
    r188_value = r188_value.to(tl.float32)
    r188_value = tl.broadcast_to(r188_value, [$(65536)])
    tl.store($(candidate_vmul_54_ptr) + r188_z, r188_value)
    r189_z = tl.arange($(65536))
    r189_value = (tl.load($(candidate_vmul_50_ptr) + ((((r189_z / $(32768)) % $(2)) * $(32768)) + (((r189_z / $(256)) % $(128)) * $(256)) + (r189_z % $(256))), dtype=tl.float32) + tl.load($(candidate_vmul_54_ptr) + ((((r189_z / $(32768)) % $(2)) * $(32768)) + (((r189_z / $(256)) % $(128)) * $(256)) + (r189_z % $(256))), dtype=tl.float32))
    r189_value = r189_value.to(tl.float32)
    r189_value = tl.broadcast_to(r189_value, [$(65536)])
    tl.store($(candidate_vadd_22_ptr) + r189_z, r189_value)
    r190_z = tl.arange($(65536))
    r190_value = tl.load($(candidate_vadd_22_ptr) + ((((r190_z / $(32768)) % $(2)) * $(32768)) + (((r190_z / $(256)) % $(128)) * $(256)) + (r190_z % $(256))), dtype=tl.float32)
    r190_value = r190_value.to(tl.bfloat16)
    r190_value = tl.broadcast_to(r190_value, [$(65536)])
    tl.store($(candidate_vto_copy_55_ptr) + r190_z, r190_value)
    r191_row = tl.arange($(256))
    r191_col = tl.arange($(512))
    r191_lane = tl.arange($(256))
    r191_left = tl.load($(candidate_vto_copy_55_ptr) + ((((((r191_row[:, None] * $(256)) + r191_lane[None, :]) / $(32768)) % $(2)) * $(32768)) + (((((r191_row[:, None] * $(256)) + r191_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r191_row[:, None] * $(256)) + r191_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r191_right = tl.load($(candidate_vto_copy_ptr) + (((r191_lane[:, None] % $(256)) * $(512)) + (r191_col[None, :] % $(512))), dtype=tl.bfloat16)
    r191_value = tl.dot(r191_left, r191_right)
    r191_value = r191_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_11_ptr) + ((r191_row[:, None] * $(512)) + r191_col[None, :]), r191_value)
    r192_z = tl.arange($(131072))
    r192_value = (tl.load($(candidate_vmm_9_ptr) + ((((r192_z / $(65536)) % $(2)) * $(65536)) + (((r192_z / $(512)) % $(128)) * $(512)) + (r192_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vmm_11_ptr) + ((((r192_z / $(65536)) % $(2)) * $(65536)) + (((r192_z / $(512)) % $(128)) * $(512)) + (r192_z % $(512))), dtype=tl.bfloat16))
    r192_value = r192_value.to(tl.bfloat16)
    r192_value = tl.broadcast_to(r192_value, [$(131072)])
    tl.store($(candidate_vadd_23_ptr) + r192_z, r192_value)
    z = tl.arange($(131072))
    value = tl.load($(candidate_vadd_23_ptr) + ((((z / $(65536)) % $(2)) * $(65536)) + (((z / $(512)) % $(128)) * $(512)) + (z % $(512))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(131072)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_mla_fa4_absorb_dx
