import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_mla_fa4_te_rope_nonabsorb_dx
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
  let vto_copy_14_ptr : RegionName := "vto_copy_14"
  let vdiv_ptr : RegionName := "vdiv"
  let vpow_1_ptr : RegionName := "vpow_1"
  let vreciprocal_ptr : RegionName := "vreciprocal"
  let vmul_7_ptr : RegionName := "vmul_7"
  let vto_copy_16_ptr : RegionName := "vto_copy_16"
  let vbmm_ptr : RegionName := "vbmm"
  let vcat_ptr : RegionName := "vcat"
  let vcos_ptr : RegionName := "vcos"
  let vmul_9_ptr : RegionName := "vmul_9"
  let vsin_ptr : RegionName := "vsin"
  let vmul_10_ptr : RegionName := "vmul_10"
  let vto_copy_17_ptr : RegionName := "vto_copy_17"
  let vto_copy_18_ptr : RegionName := "vto_copy_18"
  let vtriu_ptr : RegionName := "vtriu"
  let vmasked_fill_ptr : RegionName := "vmasked_fill"
  let vmm_ptr : RegionName := "vmm"
  let vto_copy_20_ptr : RegionName := "vto_copy_20"
  let vpow_3_ptr : RegionName := "vpow_3"
  let vmean_ptr : RegionName := "vmean"
  let vadd_2_ptr : RegionName := "vadd_2"
  let vrsqrt_ptr : RegionName := "vrsqrt"
  let vmul_13_ptr : RegionName := "vmul_13"
  let vmul_14_ptr : RegionName := "vmul_14"
  let vto_copy_21_ptr : RegionName := "vto_copy_21"
  let vmm_1_ptr : RegionName := "vmm_1"
  let vmm_2_ptr : RegionName := "vmm_2"
  let vto_copy_22_ptr : RegionName := "vto_copy_22"
  let vpow_4_ptr : RegionName := "vpow_4"
  let vmean_1_ptr : RegionName := "vmean_1"
  let vadd_3_ptr : RegionName := "vadd_3"
  let vrsqrt_1_ptr : RegionName := "vrsqrt_1"
  let vmul_15_ptr : RegionName := "vmul_15"
  let vmul_16_ptr : RegionName := "vmul_16"
  let vto_copy_23_ptr : RegionName := "vto_copy_23"
  let vmm_3_ptr : RegionName := "vmm_3"
  let vmul_17_ptr : RegionName := "vmul_17"
  let vmul_18_ptr : RegionName := "vmul_18"
  let vsub_ptr : RegionName := "vsub"
  let vmul_19_ptr : RegionName := "vmul_19"
  let vmul_20_ptr : RegionName := "vmul_20"
  let vadd_4_ptr : RegionName := "vadd_4"
  let vcat_2_ptr : RegionName := "vcat_2"
  let vmul_21_ptr : RegionName := "vmul_21"
  let vmul_22_ptr : RegionName := "vmul_22"
  let vsub_1_ptr : RegionName := "vsub_1"
  let vmul_23_ptr : RegionName := "vmul_23"
  let vmul_24_ptr : RegionName := "vmul_24"
  let vadd_5_ptr : RegionName := "vadd_5"
  let vcat_3_ptr : RegionName := "vcat_3"
  let vcat_4_ptr : RegionName := "vcat_4"
  let vcat_5_ptr : RegionName := "vcat_5"
  let vbmm_2_ptr : RegionName := "vbmm_2"
  let vmul_25_ptr : RegionName := "vmul_25"
  let vadd_6_ptr : RegionName := "vadd_6"
  let vto_copy_24_ptr : RegionName := "vto_copy_24"
  let vsoftmax_ptr : RegionName := "vsoftmax"
  let vto_copy_25_ptr : RegionName := "vto_copy_25"
  let vmm_5_ptr : RegionName := "vmm_5"
  let vbmm_4_ptr : RegionName := "vbmm_4"
  let vbmm_5_ptr : RegionName := "vbmm_5"
  let vto_copy_26_ptr : RegionName := "vto_copy_26"
  let vsoftmax_backward_data_ptr : RegionName := "vsoftmax_backward_data"
  let vto_copy_27_ptr : RegionName := "vto_copy_27"
  let vmul_26_ptr : RegionName := "vmul_26"
  let vbmm_6_ptr : RegionName := "vbmm_6"
  let vbmm_7_ptr : RegionName := "vbmm_7"
  let vsum_1_ptr : RegionName := "vsum_1"
  let vmul_27_ptr : RegionName := "vmul_27"
  let vmul_28_ptr : RegionName := "vmul_28"
  let vneg_ptr : RegionName := "vneg"
  let vmul_29_ptr : RegionName := "vmul_29"
  let vadd_7_ptr : RegionName := "vadd_7"
  let vmul_30_ptr : RegionName := "vmul_30"
  let vadd_8_ptr : RegionName := "vadd_8"
  let vmul_31_ptr : RegionName := "vmul_31"
  let vmul_32_ptr : RegionName := "vmul_32"
  let vneg_1_ptr : RegionName := "vneg_1"
  let vmul_33_ptr : RegionName := "vmul_33"
  let vadd_9_ptr : RegionName := "vadd_9"
  let vmul_34_ptr : RegionName := "vmul_34"
  let vadd_10_ptr : RegionName := "vadd_10"
  let vslice_backward_ptr : RegionName := "vslice_backward"
  let vslice_backward_1_ptr : RegionName := "vslice_backward_1"
  let vadd_11_ptr : RegionName := "vadd_11"
  let vslice_backward_2_ptr : RegionName := "vslice_backward_2"
  let vslice_backward_3_ptr : RegionName := "vslice_backward_3"
  let vadd_12_ptr : RegionName := "vadd_12"
  let vcat_6_ptr : RegionName := "vcat_6"
  let vcat_7_ptr : RegionName := "vcat_7"
  let vmm_6_ptr : RegionName := "vmm_6"
  let vto_copy_28_ptr : RegionName := "vto_copy_28"
  let vmul_35_ptr : RegionName := "vmul_35"
  let vmul_36_ptr : RegionName := "vmul_36"
  let vmul_37_ptr : RegionName := "vmul_37"
  let vsum_2_ptr : RegionName := "vsum_2"
  let vpow_5_ptr : RegionName := "vpow_5"
  let vmul_38_ptr : RegionName := "vmul_38"
  let vmul_39_ptr : RegionName := "vmul_39"
  let vdiv_2_ptr : RegionName := "vdiv_2"
  let vpow_6_ptr : RegionName := "vpow_6"
  let vmul_40_ptr : RegionName := "vmul_40"
  let vmul_41_ptr : RegionName := "vmul_41"
  let vadd_13_ptr : RegionName := "vadd_13"
  let vto_copy_29_ptr : RegionName := "vto_copy_29"
  let vcat_8_ptr : RegionName := "vcat_8"
  let vmm_7_ptr : RegionName := "vmm_7"
  let vmm_8_ptr : RegionName := "vmm_8"
  let vto_copy_30_ptr : RegionName := "vto_copy_30"
  let vmul_42_ptr : RegionName := "vmul_42"
  let vmul_43_ptr : RegionName := "vmul_43"
  let vmul_44_ptr : RegionName := "vmul_44"
  let vsum_3_ptr : RegionName := "vsum_3"
  let vpow_7_ptr : RegionName := "vpow_7"
  let vmul_45_ptr : RegionName := "vmul_45"
  let vmul_46_ptr : RegionName := "vmul_46"
  let vdiv_3_ptr : RegionName := "vdiv_3"
  let vpow_8_ptr : RegionName := "vpow_8"
  let vmul_47_ptr : RegionName := "vmul_47"
  let vmul_48_ptr : RegionName := "vmul_48"
  let vadd_14_ptr : RegionName := "vadd_14"
  let vto_copy_31_ptr : RegionName := "vto_copy_31"
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
    r16_z = tl.arange($(32))
    r16_value = tl.toReal((tl.cast(r16_z, tl.int64) * tl.full([], $(2), dtype=tl.int64)))
    r16_value = r16_value.to(tl.float32)
    r16_value = tl.broadcast_to(r16_value, [$(32)])
    tl.store($(vto_copy_14_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(32))
    r17_value = (tl.load($(vto_copy_14_ptr) + r17_z, dtype=tl.float32) / 64.0)
    r17_value = r17_value.to(tl.float32)
    r17_value = tl.broadcast_to(r17_value, [$(32)])
    tl.store($(vdiv_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(32))
    r18_value = tl.extra.cuda.libdevice.pow(10000.0, tl.load($(vdiv_ptr) + r18_z, dtype=tl.float32))
    r18_value = r18_value.to(tl.float32)
    r18_value = tl.broadcast_to(r18_value, [$(32)])
    tl.store($(vpow_1_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(32))
    r19_value = 1.0 / tl.load($(vpow_1_ptr) + r19_z, dtype=tl.float32)
    r19_value = r19_value.to(tl.float32)
    r19_value = tl.broadcast_to(r19_value, [$(32)])
    tl.store($(vreciprocal_ptr) + r19_z, r19_value)
    r20_z = tl.arange($(32))
    r20_value = (tl.load($(vreciprocal_ptr) + r20_z, dtype=tl.float32) * 1.0)
    r20_value = r20_value.to(tl.float32)
    r20_value = tl.broadcast_to(r20_value, [$(32)])
    tl.store($(vmul_7_ptr) + r20_z, r20_value)
    r21_z = tl.arange($(128))
    r21_value = tl.toReal(tl.cast(r21_z, tl.int64))
    r21_value = r21_value.to(tl.float32)
    r21_value = tl.broadcast_to(r21_value, [$(128)])
    tl.store($(vto_copy_16_ptr) + r21_z, r21_value)
    r22_row = tl.arange($(32))
    r22_col = tl.arange($(128))
    r22_lane = tl.arange($(1))
    for r22_group in range($(1)) {
      r22_left = tl.load($(vmul_7_ptr) + ((r22_row[:, None] + r22_lane[None, :]) % $(32)), dtype=tl.float32)
      r22_right = tl.load($(vto_copy_16_ptr) + (((r22_lane[:, None] * $(128)) + r22_col[None, :]) % $(128)), dtype=tl.float32)
      r22_value = tl.dot(r22_left, r22_right)
      r22_value = r22_value.to(tl.float32)
      tl.store($(vbmm_ptr) + ((r22_group * $(4096)) + (r22_row[:, None] * $(128)) + r22_col[None, :]), r22_value)
    }
    r23_z = tl.arange($(8192))
    r23_value = tl.where((r23_z % $(64)) < $(32), tl.load($(vbmm_ptr) + (((((r23_z % $(64)) - $(0)) % $(32)) * $(128)) + ((r23_z / $(64)) % $(128))), dtype=tl.float32), tl.load($(vbmm_ptr) + (((((r23_z % $(64)) - $(32)) % $(32)) * $(128)) + ((r23_z / $(64)) % $(128))), dtype=tl.float32))
    r23_value = r23_value.to(tl.float32)
    r23_value = tl.broadcast_to(r23_value, [$(8192)])
    tl.store($(vcat_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(8192))
    r24_value = tl.cos(tl.load($(vcat_ptr) + ((((r24_z / $(64)) % $(128)) * $(64)) + (r24_z % $(64))), dtype=tl.float32))
    r24_value = r24_value.to(tl.float32)
    r24_value = tl.broadcast_to(r24_value, [$(8192)])
    tl.store($(vcos_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(8192))
    r25_value = (tl.load($(vcos_ptr) + ((((r25_z / $(64)) % $(128)) * $(64)) + (r25_z % $(64))), dtype=tl.float32) * 1.0)
    r25_value = r25_value.to(tl.float32)
    r25_value = tl.broadcast_to(r25_value, [$(8192)])
    tl.store($(vmul_9_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(8192))
    r26_value = tl.sin(tl.load($(vcat_ptr) + ((((r26_z / $(64)) % $(128)) * $(64)) + (r26_z % $(64))), dtype=tl.float32))
    r26_value = r26_value.to(tl.float32)
    r26_value = tl.broadcast_to(r26_value, [$(8192)])
    tl.store($(vsin_ptr) + r26_z, r26_value)
    r27_z = tl.arange($(8192))
    r27_value = (tl.load($(vsin_ptr) + ((((r27_z / $(64)) % $(128)) * $(64)) + (r27_z % $(64))), dtype=tl.float32) * 1.0)
    r27_value = r27_value.to(tl.float32)
    r27_value = tl.broadcast_to(r27_value, [$(8192)])
    tl.store($(vmul_10_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(8192))
    r28_value = tl.load($(vmul_9_ptr) + ((((r28_z / $(64)) % $(128)) * $(64)) + (r28_z % $(64))), dtype=tl.float32)
    r28_value = r28_value.to(tl.bfloat16)
    r28_value = tl.broadcast_to(r28_value, [$(8192)])
    tl.store($(vto_copy_17_ptr) + r28_z, r28_value)
    r29_z = tl.arange($(8192))
    r29_value = tl.load($(vmul_10_ptr) + ((((r29_z / $(64)) % $(128)) * $(64)) + (r29_z % $(64))), dtype=tl.float32)
    r29_value = r29_value.to(tl.bfloat16)
    r29_value = tl.broadcast_to(r29_value, [$(8192)])
    tl.store($(vto_copy_18_ptr) + r29_z, r29_value)
    r30_z = tl.arange($(16384))
    r30_value = tl.where((r30_z % $(128)) >= ((r30_z / $(128)) % $(128)) + $(1), true, false)
    r30_value = tl.broadcast_to(r30_value, [$(16384)])
    tl.store($(vtriu_ptr) + r30_z, r30_value)
    r31_z = tl.arange($(32768))
    r31_value = tl.where(tl.load($(vtriu_ptr) + ((((r31_z / $(128)) % $(128)) * $(128)) + (r31_z % $(128))), dtype=tl.int1), tl.cast(-338953138925153547590470800371487866880.0, tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r31_value = r31_value.to(tl.bfloat16)
    r31_value = tl.broadcast_to(r31_value, [$(32768)])
    tl.store($(vmasked_fill_ptr) + r31_z, r31_value)
    r32_row = tl.arange($(256))
    r32_col = tl.arange($(256))
    r32_lane = tl.arange($(512))
    r32_left = tl.load($(input0_ptr) + ((((((r32_row[:, None] * $(512)) + r32_lane[None, :]) / $(65536)) % $(2)) * $(65536)) + (((((r32_row[:, None] * $(512)) + r32_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r32_row[:, None] * $(512)) + r32_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r32_right = tl.load($(vto_copy_ptr) + (((r32_col[None, :] % $(256)) * $(512)) + (r32_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r32_value = tl.dot(r32_left, r32_right)
    r32_value = r32_value.to(tl.bfloat16)
    tl.store($(vmm_ptr) + ((r32_row[:, None] * $(256)) + r32_col[None, :]), r32_value)
    r33_z = tl.arange($(65536))
    r33_value = tl.load($(vmm_ptr) + ((((r33_z / $(32768)) % $(2)) * $(32768)) + (((r33_z / $(256)) % $(128)) * $(256)) + (r33_z % $(256))), dtype=tl.bfloat16)
    r33_value = r33_value.to(tl.float32)
    r33_value = tl.broadcast_to(r33_value, [$(65536)])
    tl.store($(vto_copy_20_ptr) + r33_z, r33_value)
    r34_z = tl.arange($(65536))
    r34_value = ((tl.load($(vto_copy_20_ptr) + ((((r34_z / $(32768)) % $(2)) * $(32768)) + (((r34_z / $(256)) % $(128)) * $(256)) + (r34_z % $(256))), dtype=tl.float32)) * (tl.load($(vto_copy_20_ptr) + ((((r34_z / $(32768)) % $(2)) * $(32768)) + (((r34_z / $(256)) % $(128)) * $(256)) + (r34_z % $(256))), dtype=tl.float32)))
    r34_value = r34_value.to(tl.float32)
    r34_value = tl.broadcast_to(r34_value, [$(65536)])
    tl.store($(vpow_3_ptr) + r34_z, r34_value)
    r35_z = tl.arange($(256))
    r35_lane = tl.arange($(256))
    r35_value = (tl.sum(tl.load($(vpow_3_ptr) + ((((r35_z[:, None] / $(128)) % $(2)) * $(32768)) + ((r35_z[:, None] % $(128)) * $(256)) + r35_lane[None, :]), dtype=tl.float32), axis=1) / 256.0)
    r35_value = r35_value.to(tl.float32)
    r35_value = tl.broadcast_to(r35_value, [$(256)])
    tl.store($(vmean_ptr) + r35_z, r35_value)
    r36_z = tl.arange($(256))
    r36_value = (tl.load($(vmean_ptr) + ((((r36_z / $(128)) % $(2)) * $(128)) + (r36_z % $(128))), dtype=tl.float32) + 0.000001)
    r36_value = r36_value.to(tl.float32)
    r36_value = tl.broadcast_to(r36_value, [$(256)])
    tl.store($(vadd_2_ptr) + r36_z, r36_value)
    r37_z = tl.arange($(256))
    r37_value = tl.rsqrt(tl.load($(vadd_2_ptr) + ((((r37_z / $(128)) % $(2)) * $(128)) + (r37_z % $(128))), dtype=tl.float32))
    r37_value = r37_value.to(tl.float32)
    r37_value = tl.broadcast_to(r37_value, [$(256)])
    tl.store($(vrsqrt_ptr) + r37_z, r37_value)
    r38_z = tl.arange($(65536))
    r38_value = (tl.load($(vto_copy_20_ptr) + ((((r38_z / $(32768)) % $(2)) * $(32768)) + (((r38_z / $(256)) % $(128)) * $(256)) + (r38_z % $(256))), dtype=tl.float32) * tl.load($(vrsqrt_ptr) + ((((r38_z / $(32768)) % $(2)) * $(128)) + ((r38_z / $(256)) % $(128))), dtype=tl.float32))
    r38_value = r38_value.to(tl.float32)
    r38_value = tl.broadcast_to(r38_value, [$(65536)])
    tl.store($(vmul_13_ptr) + r38_z, r38_value)
    r39_z = tl.arange($(65536))
    r39_value = (tl.cast(tl.load($(vto_copy_1_ptr) + (r39_z % $(256)), dtype=tl.bfloat16), tl.float32) * tl.load($(vmul_13_ptr) + ((((r39_z / $(32768)) % $(2)) * $(32768)) + (((r39_z / $(256)) % $(128)) * $(256)) + (r39_z % $(256))), dtype=tl.float32))
    r39_value = r39_value.to(tl.float32)
    r39_value = tl.broadcast_to(r39_value, [$(65536)])
    tl.store($(vmul_14_ptr) + r39_z, r39_value)
    r40_z = tl.arange($(65536))
    r40_value = tl.load($(vmul_14_ptr) + ((((r40_z / $(32768)) % $(2)) * $(32768)) + (((r40_z / $(256)) % $(128)) * $(256)) + (r40_z % $(256))), dtype=tl.float32)
    r40_value = r40_value.to(tl.bfloat16)
    r40_value = tl.broadcast_to(r40_value, [$(65536)])
    tl.store($(vto_copy_21_ptr) + r40_z, r40_value)
    r41_row = tl.arange($(256))
    r41_col = tl.arange($(1536))
    r41_lane = tl.arange($(256))
    r41_left = tl.load($(vto_copy_21_ptr) + ((((((r41_row[:, None] * $(256)) + r41_lane[None, :]) / $(32768)) % $(2)) * $(32768)) + (((((r41_row[:, None] * $(256)) + r41_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r41_row[:, None] * $(256)) + r41_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r41_right = tl.load($(vto_copy_2_ptr) + (((r41_col[None, :] % $(1536)) * $(256)) + (r41_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r41_value = tl.dot(r41_left, r41_right)
    r41_value = r41_value.to(tl.bfloat16)
    tl.store($(vmm_1_ptr) + ((r41_row[:, None] * $(1536)) + r41_col[None, :]), r41_value)
    r42_row = tl.arange($(256))
    r42_col = tl.arange($(320))
    r42_lane = tl.arange($(512))
    r42_left = tl.load($(input0_ptr) + ((((((r42_row[:, None] * $(512)) + r42_lane[None, :]) / $(65536)) % $(2)) * $(65536)) + (((((r42_row[:, None] * $(512)) + r42_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r42_row[:, None] * $(512)) + r42_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r42_right = tl.load($(vto_copy_3_ptr) + (((r42_col[None, :] % $(320)) * $(512)) + (r42_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r42_value = tl.dot(r42_left, r42_right)
    r42_value = r42_value.to(tl.bfloat16)
    tl.store($(vmm_2_ptr) + ((r42_row[:, None] * $(320)) + r42_col[None, :]), r42_value)
    r43_z = tl.arange($(65536))
    r43_value = tl.load($(vmm_2_ptr) + ((((r43_z / $(32768)) % $(2)) * $(40960)) + (((r43_z / $(256)) % $(128)) * $(320)) + (r43_z % $(256))), dtype=tl.bfloat16)
    r43_value = r43_value.to(tl.float32)
    r43_value = tl.broadcast_to(r43_value, [$(65536)])
    tl.store($(vto_copy_22_ptr) + r43_z, r43_value)
    r44_z = tl.arange($(65536))
    r44_value = ((tl.load($(vto_copy_22_ptr) + ((((r44_z / $(32768)) % $(2)) * $(32768)) + (((r44_z / $(256)) % $(128)) * $(256)) + (r44_z % $(256))), dtype=tl.float32)) * (tl.load($(vto_copy_22_ptr) + ((((r44_z / $(32768)) % $(2)) * $(32768)) + (((r44_z / $(256)) % $(128)) * $(256)) + (r44_z % $(256))), dtype=tl.float32)))
    r44_value = r44_value.to(tl.float32)
    r44_value = tl.broadcast_to(r44_value, [$(65536)])
    tl.store($(vpow_4_ptr) + r44_z, r44_value)
    r45_z = tl.arange($(256))
    r45_lane = tl.arange($(256))
    r45_value = (tl.sum(tl.load($(vpow_4_ptr) + ((((r45_z[:, None] / $(128)) % $(2)) * $(32768)) + ((r45_z[:, None] % $(128)) * $(256)) + r45_lane[None, :]), dtype=tl.float32), axis=1) / 256.0)
    r45_value = r45_value.to(tl.float32)
    r45_value = tl.broadcast_to(r45_value, [$(256)])
    tl.store($(vmean_1_ptr) + r45_z, r45_value)
    r46_z = tl.arange($(256))
    r46_value = (tl.load($(vmean_1_ptr) + ((((r46_z / $(128)) % $(2)) * $(128)) + (r46_z % $(128))), dtype=tl.float32) + 0.000001)
    r46_value = r46_value.to(tl.float32)
    r46_value = tl.broadcast_to(r46_value, [$(256)])
    tl.store($(vadd_3_ptr) + r46_z, r46_value)
    r47_z = tl.arange($(256))
    r47_value = tl.rsqrt(tl.load($(vadd_3_ptr) + ((((r47_z / $(128)) % $(2)) * $(128)) + (r47_z % $(128))), dtype=tl.float32))
    r47_value = r47_value.to(tl.float32)
    r47_value = tl.broadcast_to(r47_value, [$(256)])
    tl.store($(vrsqrt_1_ptr) + r47_z, r47_value)
    r48_z = tl.arange($(65536))
    r48_value = (tl.load($(vto_copy_22_ptr) + ((((r48_z / $(32768)) % $(2)) * $(32768)) + (((r48_z / $(256)) % $(128)) * $(256)) + (r48_z % $(256))), dtype=tl.float32) * tl.load($(vrsqrt_1_ptr) + ((((r48_z / $(32768)) % $(2)) * $(128)) + ((r48_z / $(256)) % $(128))), dtype=tl.float32))
    r48_value = r48_value.to(tl.float32)
    r48_value = tl.broadcast_to(r48_value, [$(65536)])
    tl.store($(vmul_15_ptr) + r48_z, r48_value)
    r49_z = tl.arange($(65536))
    r49_value = (tl.cast(tl.load($(vto_copy_4_ptr) + (r49_z % $(256)), dtype=tl.bfloat16), tl.float32) * tl.load($(vmul_15_ptr) + ((((r49_z / $(32768)) % $(2)) * $(32768)) + (((r49_z / $(256)) % $(128)) * $(256)) + (r49_z % $(256))), dtype=tl.float32))
    r49_value = r49_value.to(tl.float32)
    r49_value = tl.broadcast_to(r49_value, [$(65536)])
    tl.store($(vmul_16_ptr) + r49_z, r49_value)
    r50_z = tl.arange($(65536))
    r50_value = tl.load($(vmul_16_ptr) + ((((r50_z / $(32768)) % $(2)) * $(32768)) + (((r50_z / $(256)) % $(128)) * $(256)) + (r50_z % $(256))), dtype=tl.float32)
    r50_value = r50_value.to(tl.bfloat16)
    r50_value = tl.broadcast_to(r50_value, [$(65536)])
    tl.store($(vto_copy_23_ptr) + r50_z, r50_value)
    r51_row = tl.arange($(256))
    r51_col = tl.arange($(2048))
    r51_lane = tl.arange($(256))
    r51_left = tl.load($(vto_copy_23_ptr) + ((((((r51_row[:, None] * $(256)) + r51_lane[None, :]) / $(32768)) % $(2)) * $(32768)) + (((((r51_row[:, None] * $(256)) + r51_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r51_row[:, None] * $(256)) + r51_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r51_right = tl.load($(vto_copy_5_ptr) + (((r51_col[None, :] % $(2048)) * $(256)) + (r51_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r51_value = tl.dot(r51_left, r51_right)
    r51_value = r51_value.to(tl.bfloat16)
    tl.store($(vmm_3_ptr) + ((r51_row[:, None] * $(2048)) + r51_col[None, :]), r51_value)
    r52_z = tl.arange($(65536))
    r52_value = (tl.load($(vmm_1_ptr) + ((((r52_z / $(32768)) % $(2)) * $(196608)) + (((r52_z / $(32)) % $(128)) * $(1536)) + (((r52_z / $(4096)) % $(8)) * $(192)) + ((r52_z % $(32)) * $(2)) + $(128)), dtype=tl.bfloat16) * tl.load($(vto_copy_17_ptr) + ((((r52_z / $(32)) % $(128)) * $(64)) + (r52_z % $(32))), dtype=tl.bfloat16))
    r52_value = r52_value.to(tl.bfloat16)
    r52_value = tl.broadcast_to(r52_value, [$(65536)])
    tl.store($(vmul_17_ptr) + r52_z, r52_value)
    r53_z = tl.arange($(65536))
    r53_value = (tl.load($(vmm_1_ptr) + ((((r53_z / $(32768)) % $(2)) * $(196608)) + (((r53_z / $(32)) % $(128)) * $(1536)) + (((r53_z / $(4096)) % $(8)) * $(192)) + ((r53_z % $(32)) * $(2)) + $(129)), dtype=tl.bfloat16) * tl.load($(vto_copy_18_ptr) + ((((r53_z / $(32)) % $(128)) * $(64)) + (r53_z % $(32))), dtype=tl.bfloat16))
    r53_value = r53_value.to(tl.bfloat16)
    r53_value = tl.broadcast_to(r53_value, [$(65536)])
    tl.store($(vmul_18_ptr) + r53_z, r53_value)
    r54_z = tl.arange($(65536))
    r54_value = (tl.load($(vmul_17_ptr) + ((((r54_z / $(32768)) % $(2)) * $(32768)) + (((r54_z / $(4096)) % $(8)) * $(4096)) + (((r54_z / $(32)) % $(128)) * $(32)) + (r54_z % $(32))), dtype=tl.bfloat16) - tl.load($(vmul_18_ptr) + ((((r54_z / $(32768)) % $(2)) * $(32768)) + (((r54_z / $(4096)) % $(8)) * $(4096)) + (((r54_z / $(32)) % $(128)) * $(32)) + (r54_z % $(32))), dtype=tl.bfloat16))
    r54_value = r54_value.to(tl.bfloat16)
    r54_value = tl.broadcast_to(r54_value, [$(65536)])
    tl.store($(vsub_ptr) + r54_z, r54_value)
    r55_z = tl.arange($(65536))
    r55_value = (tl.load($(vmm_1_ptr) + ((((r55_z / $(32768)) % $(2)) * $(196608)) + (((r55_z / $(32)) % $(128)) * $(1536)) + (((r55_z / $(4096)) % $(8)) * $(192)) + ((r55_z % $(32)) * $(2)) + $(129)), dtype=tl.bfloat16) * tl.load($(vto_copy_17_ptr) + ((((r55_z / $(32)) % $(128)) * $(64)) + (r55_z % $(32))), dtype=tl.bfloat16))
    r55_value = r55_value.to(tl.bfloat16)
    r55_value = tl.broadcast_to(r55_value, [$(65536)])
    tl.store($(vmul_19_ptr) + r55_z, r55_value)
    r56_z = tl.arange($(65536))
    r56_value = (tl.load($(vmm_1_ptr) + ((((r56_z / $(32768)) % $(2)) * $(196608)) + (((r56_z / $(32)) % $(128)) * $(1536)) + (((r56_z / $(4096)) % $(8)) * $(192)) + ((r56_z % $(32)) * $(2)) + $(128)), dtype=tl.bfloat16) * tl.load($(vto_copy_18_ptr) + ((((r56_z / $(32)) % $(128)) * $(64)) + (r56_z % $(32))), dtype=tl.bfloat16))
    r56_value = r56_value.to(tl.bfloat16)
    r56_value = tl.broadcast_to(r56_value, [$(65536)])
    tl.store($(vmul_20_ptr) + r56_z, r56_value)
    r57_z = tl.arange($(65536))
    r57_value = (tl.load($(vmul_19_ptr) + ((((r57_z / $(32768)) % $(2)) * $(32768)) + (((r57_z / $(4096)) % $(8)) * $(4096)) + (((r57_z / $(32)) % $(128)) * $(32)) + (r57_z % $(32))), dtype=tl.bfloat16) + tl.load($(vmul_20_ptr) + ((((r57_z / $(32768)) % $(2)) * $(32768)) + (((r57_z / $(4096)) % $(8)) * $(4096)) + (((r57_z / $(32)) % $(128)) * $(32)) + (r57_z % $(32))), dtype=tl.bfloat16))
    r57_value = r57_value.to(tl.bfloat16)
    r57_value = tl.broadcast_to(r57_value, [$(65536)])
    tl.store($(vadd_4_ptr) + r57_z, r57_value)
    r58_z = tl.arange($(131072))
    r58_value = tl.where((r58_z % $(64)) < $(32), tl.load($(vsub_ptr) + ((((r58_z / $(65536)) % $(2)) * $(32768)) + (((r58_z / $(8192)) % $(8)) * $(4096)) + (((r58_z / $(64)) % $(128)) * $(32)) + (((r58_z % $(64)) - $(0)) % $(32))), dtype=tl.bfloat16), tl.load($(vadd_4_ptr) + ((((r58_z / $(65536)) % $(2)) * $(32768)) + (((r58_z / $(8192)) % $(8)) * $(4096)) + (((r58_z / $(64)) % $(128)) * $(32)) + (((r58_z % $(64)) - $(32)) % $(32))), dtype=tl.bfloat16))
    r58_value = r58_value.to(tl.bfloat16)
    r58_value = tl.broadcast_to(r58_value, [$(131072)])
    tl.store($(vcat_2_ptr) + r58_z, r58_value)
    r59_z = tl.arange($(8192))
    r59_value = (tl.load($(vmm_2_ptr) + ((((r59_z / $(4096)) % $(2)) * $(40960)) + (((r59_z / $(32)) % $(128)) * $(320)) + ((r59_z % $(32)) * $(2)) + $(256)), dtype=tl.bfloat16) * tl.load($(vto_copy_17_ptr) + ((((r59_z / $(32)) % $(128)) * $(64)) + (r59_z % $(32))), dtype=tl.bfloat16))
    r59_value = r59_value.to(tl.bfloat16)
    r59_value = tl.broadcast_to(r59_value, [$(8192)])
    tl.store($(vmul_21_ptr) + r59_z, r59_value)
    r60_z = tl.arange($(8192))
    r60_value = (tl.load($(vmm_2_ptr) + ((((r60_z / $(4096)) % $(2)) * $(40960)) + (((r60_z / $(32)) % $(128)) * $(320)) + ((r60_z % $(32)) * $(2)) + $(257)), dtype=tl.bfloat16) * tl.load($(vto_copy_18_ptr) + ((((r60_z / $(32)) % $(128)) * $(64)) + (r60_z % $(32))), dtype=tl.bfloat16))
    r60_value = r60_value.to(tl.bfloat16)
    r60_value = tl.broadcast_to(r60_value, [$(8192)])
    tl.store($(vmul_22_ptr) + r60_z, r60_value)
    r61_z = tl.arange($(8192))
    r61_value = (tl.load($(vmul_21_ptr) + ((((r61_z / $(4096)) % $(2)) * $(4096)) + (((r61_z / $(32)) % $(128)) * $(32)) + (r61_z % $(32))), dtype=tl.bfloat16) - tl.load($(vmul_22_ptr) + ((((r61_z / $(4096)) % $(2)) * $(4096)) + (((r61_z / $(32)) % $(128)) * $(32)) + (r61_z % $(32))), dtype=tl.bfloat16))
    r61_value = r61_value.to(tl.bfloat16)
    r61_value = tl.broadcast_to(r61_value, [$(8192)])
    tl.store($(vsub_1_ptr) + r61_z, r61_value)
    r62_z = tl.arange($(8192))
    r62_value = (tl.load($(vmm_2_ptr) + ((((r62_z / $(4096)) % $(2)) * $(40960)) + (((r62_z / $(32)) % $(128)) * $(320)) + ((r62_z % $(32)) * $(2)) + $(257)), dtype=tl.bfloat16) * tl.load($(vto_copy_17_ptr) + ((((r62_z / $(32)) % $(128)) * $(64)) + (r62_z % $(32))), dtype=tl.bfloat16))
    r62_value = r62_value.to(tl.bfloat16)
    r62_value = tl.broadcast_to(r62_value, [$(8192)])
    tl.store($(vmul_23_ptr) + r62_z, r62_value)
    r63_z = tl.arange($(8192))
    r63_value = (tl.load($(vmm_2_ptr) + ((((r63_z / $(4096)) % $(2)) * $(40960)) + (((r63_z / $(32)) % $(128)) * $(320)) + ((r63_z % $(32)) * $(2)) + $(256)), dtype=tl.bfloat16) * tl.load($(vto_copy_18_ptr) + ((((r63_z / $(32)) % $(128)) * $(64)) + (r63_z % $(32))), dtype=tl.bfloat16))
    r63_value = r63_value.to(tl.bfloat16)
    r63_value = tl.broadcast_to(r63_value, [$(8192)])
    tl.store($(vmul_24_ptr) + r63_z, r63_value)
    r64_z = tl.arange($(8192))
    r64_value = (tl.load($(vmul_23_ptr) + ((((r64_z / $(4096)) % $(2)) * $(4096)) + (((r64_z / $(32)) % $(128)) * $(32)) + (r64_z % $(32))), dtype=tl.bfloat16) + tl.load($(vmul_24_ptr) + ((((r64_z / $(4096)) % $(2)) * $(4096)) + (((r64_z / $(32)) % $(128)) * $(32)) + (r64_z % $(32))), dtype=tl.bfloat16))
    r64_value = r64_value.to(tl.bfloat16)
    r64_value = tl.broadcast_to(r64_value, [$(8192)])
    tl.store($(vadd_5_ptr) + r64_z, r64_value)
    r65_z = tl.arange($(16384))
    r65_value = tl.where((r65_z % $(64)) < $(32), tl.load($(vsub_1_ptr) + ((((r65_z / $(8192)) % $(2)) * $(4096)) + (((r65_z / $(64)) % $(128)) * $(32)) + (((r65_z % $(64)) - $(0)) % $(32))), dtype=tl.bfloat16), tl.load($(vadd_5_ptr) + ((((r65_z / $(8192)) % $(2)) * $(4096)) + (((r65_z / $(64)) % $(128)) * $(32)) + (((r65_z % $(64)) - $(32)) % $(32))), dtype=tl.bfloat16))
    r65_value = r65_value.to(tl.bfloat16)
    r65_value = tl.broadcast_to(r65_value, [$(16384)])
    tl.store($(vcat_3_ptr) + r65_z, r65_value)
    r66_z = tl.arange($(393216))
    r66_value = tl.where((r66_z % $(192)) < $(128), tl.load($(vmm_1_ptr) + ((((((((r66_z / $(196608)) % $(2)) * $(196608)) + (((r66_z / $(192)) % $(128)) * $(1536)) + (((r66_z / $(24576)) % $(8)) * $(192)) + ((r66_z % $(192)) - $(0))) / $(196608)) % $(2)) * $(196608)) + (((((((r66_z / $(196608)) % $(2)) * $(196608)) + (((r66_z / $(192)) % $(128)) * $(1536)) + (((r66_z / $(24576)) % $(8)) * $(192)) + ((r66_z % $(192)) - $(0))) / $(1536)) % $(128)) * $(1536)) + (((((r66_z / $(196608)) % $(2)) * $(196608)) + (((r66_z / $(192)) % $(128)) * $(1536)) + (((r66_z / $(24576)) % $(8)) * $(192)) + ((r66_z % $(192)) - $(0))) % $(1536))), dtype=tl.bfloat16), tl.load($(vcat_2_ptr) + ((((r66_z / $(196608)) % $(2)) * $(65536)) + (((r66_z / $(24576)) % $(8)) * $(8192)) + (((r66_z / $(192)) % $(128)) * $(64)) + (((r66_z % $(192)) - $(128)) % $(64))), dtype=tl.bfloat16))
    r66_value = r66_value.to(tl.bfloat16)
    r66_value = tl.broadcast_to(r66_value, [$(393216)])
    tl.store($(vcat_4_ptr) + r66_z, r66_value)
    r67_z = tl.arange($(393216))
    r67_value = tl.where((r67_z % $(192)) < $(128), tl.load($(vmm_3_ptr) + ((((((((r67_z / $(196608)) % $(2)) * $(262144)) + (((r67_z / $(192)) % $(128)) * $(2048)) + (((r67_z / $(24576)) % $(8)) * $(256)) + ((r67_z % $(192)) - $(0))) / $(262144)) % $(2)) * $(262144)) + (((((((r67_z / $(196608)) % $(2)) * $(262144)) + (((r67_z / $(192)) % $(128)) * $(2048)) + (((r67_z / $(24576)) % $(8)) * $(256)) + ((r67_z % $(192)) - $(0))) / $(2048)) % $(128)) * $(2048)) + (((((r67_z / $(196608)) % $(2)) * $(262144)) + (((r67_z / $(192)) % $(128)) * $(2048)) + (((r67_z / $(24576)) % $(8)) * $(256)) + ((r67_z % $(192)) - $(0))) % $(2048))), dtype=tl.bfloat16), tl.load($(vcat_3_ptr) + ((((r67_z / $(196608)) % $(2)) * $(8192)) + (((r67_z / $(192)) % $(128)) * $(64)) + (((r67_z % $(192)) - $(128)) % $(64))), dtype=tl.bfloat16))
    r67_value = r67_value.to(tl.bfloat16)
    r67_value = tl.broadcast_to(r67_value, [$(393216)])
    tl.store($(vcat_5_ptr) + r67_z, r67_value)
    r68_row = tl.arange($(128))
    r68_col = tl.arange($(128))
    r68_lane = tl.arange($(192))
    for r68_group in range($(16)) {
      r68_left = tl.load($(vcat_4_ptr) + (((((((r68_group % $(16)) * $(24576)) + (r68_row[:, None] * $(192)) + r68_lane[None, :]) / $(196608)) % $(2)) * $(196608)) + ((((((r68_group % $(16)) * $(24576)) + (r68_row[:, None] * $(192)) + r68_lane[None, :]) / $(24576)) % $(8)) * $(24576)) + ((((((r68_group % $(16)) * $(24576)) + (r68_row[:, None] * $(192)) + r68_lane[None, :]) / $(192)) % $(128)) * $(192)) + ((((r68_group % $(16)) * $(24576)) + (r68_row[:, None] * $(192)) + r68_lane[None, :]) % $(192))), dtype=tl.bfloat16)
      r68_right = tl.load($(vcat_5_ptr) + (((((((r68_group % $(16)) * $(24576)) + (r68_lane[:, None] * $(128)) + r68_col[None, :]) / $(196608)) % $(2)) * $(196608)) + ((((((r68_group % $(16)) * $(24576)) + (r68_lane[:, None] * $(128)) + r68_col[None, :]) / $(24576)) % $(8)) * $(24576)) + (((((r68_group % $(16)) * $(24576)) + (r68_lane[:, None] * $(128)) + r68_col[None, :]) % $(128)) * $(192)) + (((((r68_group % $(16)) * $(24576)) + (r68_lane[:, None] * $(128)) + r68_col[None, :]) / $(128)) % $(192))), dtype=tl.bfloat16)
      r68_value = tl.dot(r68_left, r68_right)
      r68_value = r68_value.to(tl.bfloat16)
      tl.store($(vbmm_2_ptr) + ((r68_group * $(16384)) + (r68_row[:, None] * $(128)) + r68_col[None, :]), r68_value)
    }
    r69_z = tl.arange($(262144))
    r69_value = (tl.load($(vbmm_2_ptr) + ((((r69_z / $(131072)) % $(2)) * $(131072)) + (((r69_z / $(16384)) % $(8)) * $(16384)) + (((r69_z / $(128)) % $(128)) * $(128)) + (r69_z % $(128))), dtype=tl.bfloat16) * tl.cast(0.07216878364870322, tl.bfloat16))
    r69_value = r69_value.to(tl.bfloat16)
    r69_value = tl.broadcast_to(r69_value, [$(262144)])
    tl.store($(vmul_25_ptr) + r69_z, r69_value)
    r70_z = tl.arange($(262144))
    r70_value = (tl.load($(vmul_25_ptr) + ((((r70_z / $(131072)) % $(2)) * $(131072)) + (((r70_z / $(16384)) % $(8)) * $(16384)) + (((r70_z / $(128)) % $(128)) * $(128)) + (r70_z % $(128))), dtype=tl.bfloat16) + tl.load($(vmasked_fill_ptr) + ((((r70_z / $(131072)) % $(2)) * $(16384)) + (((r70_z / $(128)) % $(128)) * $(128)) + (r70_z % $(128))), dtype=tl.bfloat16))
    r70_value = r70_value.to(tl.bfloat16)
    r70_value = tl.broadcast_to(r70_value, [$(262144)])
    tl.store($(vadd_6_ptr) + r70_z, r70_value)
    r71_z = tl.arange($(262144))
    r71_value = tl.load($(vadd_6_ptr) + ((((r71_z / $(131072)) % $(2)) * $(131072)) + (((r71_z / $(16384)) % $(8)) * $(16384)) + (((r71_z / $(128)) % $(128)) * $(128)) + (r71_z % $(128))), dtype=tl.bfloat16)
    r71_value = r71_value.to(tl.float32)
    r71_value = tl.broadcast_to(r71_value, [$(262144)])
    tl.store($(vto_copy_24_ptr) + r71_z, r71_value)
    r72_z = tl.arange($(262144))
    r72_lane = tl.arange($(128))
    r72_row_values = tl.load($(vto_copy_24_ptr) + ((((r72_z[:, None] / $(131072)) % $(2)) * $(131072)) + (((r72_z[:, None] / $(16384)) % $(8)) * $(16384)) + (((r72_z[:, None] / $(128)) % $(128)) * $(128)) + (r72_lane[None, :] % $(128))), dtype=tl.float32)
    r72_maximum = tl.max(r72_row_values, axis=1, keep_dims=true)
    r72_weights = tl.exp(r72_row_values - r72_maximum)
    r72_total = tl.sum(r72_weights, axis=1)
    r72_maximum = tl.max(r72_row_values, axis=1)
    r72_value = tl.exp(tl.load($(vto_copy_24_ptr) + ((((r72_z / $(131072)) % $(2)) * $(131072)) + (((r72_z / $(16384)) % $(8)) * $(16384)) + (((r72_z / $(128)) % $(128)) * $(128)) + (r72_z % $(128))), dtype=tl.float32) - r72_maximum) / r72_total
    r72_value = r72_value.to(tl.float32)
    r72_value = tl.broadcast_to(r72_value, [$(262144)])
    tl.store($(vsoftmax_ptr) + r72_z, r72_value)
    r73_z = tl.arange($(262144))
    r73_value = tl.load($(vsoftmax_ptr) + ((((r73_z / $(131072)) % $(2)) * $(131072)) + (((r73_z / $(16384)) % $(8)) * $(16384)) + (((r73_z / $(128)) % $(128)) * $(128)) + (r73_z % $(128))), dtype=tl.float32)
    r73_value = r73_value.to(tl.bfloat16)
    r73_value = tl.broadcast_to(r73_value, [$(262144)])
    tl.store($(vto_copy_25_ptr) + r73_z, r73_value)
    r74_row = tl.arange($(256))
    r74_col = tl.arange($(1024))
    r74_lane = tl.arange($(512))
    r74_left = tl.load($(input1_ptr) + ((((((r74_row[:, None] * $(512)) + r74_lane[None, :]) / $(65536)) % $(2)) * $(65536)) + (((((r74_row[:, None] * $(512)) + r74_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r74_row[:, None] * $(512)) + r74_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r74_right = tl.load($(vto_copy_6_ptr) + (((r74_lane[:, None] % $(512)) * $(1024)) + (r74_col[None, :] % $(1024))), dtype=tl.bfloat16)
    r74_value = tl.dot(r74_left, r74_right)
    r74_value = r74_value.to(tl.bfloat16)
    tl.store($(vmm_5_ptr) + ((r74_row[:, None] * $(1024)) + r74_col[None, :]), r74_value)
    r75_row = tl.arange($(128))
    r75_col = tl.arange($(128))
    r75_lane = tl.arange($(128))
    for r75_group in range($(16)) {
      r75_left = tl.load($(vto_copy_25_ptr) + (((((((r75_group % $(16)) * $(16384)) + (r75_lane[None, :] * $(128)) + r75_row[:, None]) / $(131072)) % $(2)) * $(131072)) + ((((((r75_group % $(16)) * $(16384)) + (r75_lane[None, :] * $(128)) + r75_row[:, None]) / $(16384)) % $(8)) * $(16384)) + ((((((r75_group % $(16)) * $(16384)) + (r75_lane[None, :] * $(128)) + r75_row[:, None]) / $(128)) % $(128)) * $(128)) + ((((r75_group % $(16)) * $(16384)) + (r75_lane[None, :] * $(128)) + r75_row[:, None]) % $(128))), dtype=tl.bfloat16)
      r75_right = tl.load($(vmm_5_ptr) + (((((((r75_group % $(16)) * $(16384)) + (r75_lane[:, None] * $(128)) + r75_col[None, :]) / $(131072)) % $(2)) * $(131072)) + ((((((r75_group % $(16)) * $(16384)) + (r75_lane[:, None] * $(128)) + r75_col[None, :]) / $(128)) % $(128)) * $(1024)) + ((((((r75_group % $(16)) * $(16384)) + (r75_lane[:, None] * $(128)) + r75_col[None, :]) / $(16384)) % $(8)) * $(128)) + ((((r75_group % $(16)) * $(16384)) + (r75_lane[:, None] * $(128)) + r75_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r75_value = tl.dot(r75_left, r75_right)
      r75_value = r75_value.to(tl.bfloat16)
      tl.store($(vbmm_4_ptr) + ((r75_group * $(16384)) + (r75_row[:, None] * $(128)) + r75_col[None, :]), r75_value)
    }
    r76_row = tl.arange($(128))
    r76_col = tl.arange($(128))
    r76_lane = tl.arange($(128))
    for r76_group in range($(16)) {
      r76_left = tl.load($(vmm_5_ptr) + (((((((r76_group % $(16)) * $(16384)) + (r76_row[:, None] * $(128)) + r76_lane[None, :]) / $(131072)) % $(2)) * $(131072)) + ((((((r76_group % $(16)) * $(16384)) + (r76_row[:, None] * $(128)) + r76_lane[None, :]) / $(128)) % $(128)) * $(1024)) + ((((((r76_group % $(16)) * $(16384)) + (r76_row[:, None] * $(128)) + r76_lane[None, :]) / $(16384)) % $(8)) * $(128)) + ((((r76_group % $(16)) * $(16384)) + (r76_row[:, None] * $(128)) + r76_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r76_right = tl.load($(vmm_3_ptr) + (((((((r76_group % $(16)) * $(16384)) + (r76_col[None, :] * $(128)) + r76_lane[:, None]) / $(131072)) % $(2)) * $(262144)) + ((((((r76_group % $(16)) * $(16384)) + (r76_col[None, :] * $(128)) + r76_lane[:, None]) / $(128)) % $(128)) * $(2048)) + ((((((r76_group % $(16)) * $(16384)) + (r76_col[None, :] * $(128)) + r76_lane[:, None]) / $(16384)) % $(8)) * $(256)) + ((((r76_group % $(16)) * $(16384)) + (r76_col[None, :] * $(128)) + r76_lane[:, None]) % $(128)) + $(128)), dtype=tl.bfloat16)
      r76_value = tl.dot(r76_left, r76_right)
      r76_value = r76_value.to(tl.bfloat16)
      tl.store($(vbmm_5_ptr) + ((r76_group * $(16384)) + (r76_row[:, None] * $(128)) + r76_col[None, :]), r76_value)
    }
    r77_z = tl.arange($(262144))
    r77_value = tl.load($(vbmm_5_ptr) + ((((r77_z / $(131072)) % $(2)) * $(131072)) + (((r77_z / $(16384)) % $(8)) * $(16384)) + (((r77_z / $(128)) % $(128)) * $(128)) + (r77_z % $(128))), dtype=tl.bfloat16)
    r77_value = r77_value.to(tl.float32)
    r77_value = tl.broadcast_to(r77_value, [$(262144)])
    tl.store($(vto_copy_26_ptr) + r77_z, r77_value)
    r78_z = tl.arange($(262144))
    r78_lane = tl.arange($(128))
    r78_row_values = tl.load($(vsoftmax_ptr) + ((((r78_z[:, None] / $(131072)) % $(2)) * $(131072)) + (((r78_z[:, None] / $(16384)) % $(8)) * $(16384)) + (((r78_z[:, None] / $(128)) % $(128)) * $(128)) + (r78_lane[None, :] % $(128))), dtype=tl.float32)
    r78_row_grad = tl.load($(vto_copy_26_ptr) + ((((r78_z[:, None] / $(131072)) % $(2)) * $(131072)) + (((r78_z[:, None] / $(16384)) % $(8)) * $(16384)) + (((r78_z[:, None] / $(128)) % $(128)) * $(128)) + (r78_lane[None, :] % $(128))), dtype=tl.float32)
    r78_weighted = r78_row_grad * r78_row_values
    r78_dot_grad = tl.sum(r78_weighted, axis=1)
    r78_value = (tl.load($(vto_copy_26_ptr) + ((((r78_z / $(131072)) % $(2)) * $(131072)) + (((r78_z / $(16384)) % $(8)) * $(16384)) + (((r78_z / $(128)) % $(128)) * $(128)) + (r78_z % $(128))), dtype=tl.float32) - r78_dot_grad) * tl.load($(vsoftmax_ptr) + ((((r78_z / $(131072)) % $(2)) * $(131072)) + (((r78_z / $(16384)) % $(8)) * $(16384)) + (((r78_z / $(128)) % $(128)) * $(128)) + (r78_z % $(128))), dtype=tl.float32)
    r78_value = r78_value.to(tl.float32)
    r78_value = tl.broadcast_to(r78_value, [$(262144)])
    tl.store($(vsoftmax_backward_data_ptr) + r78_z, r78_value)
    r79_z = tl.arange($(262144))
    r79_value = tl.load($(vsoftmax_backward_data_ptr) + ((((r79_z / $(131072)) % $(2)) * $(131072)) + (((r79_z / $(16384)) % $(8)) * $(16384)) + (((r79_z / $(128)) % $(128)) * $(128)) + (r79_z % $(128))), dtype=tl.float32)
    r79_value = r79_value.to(tl.bfloat16)
    r79_value = tl.broadcast_to(r79_value, [$(262144)])
    tl.store($(vto_copy_27_ptr) + r79_z, r79_value)
    r80_z = tl.arange($(262144))
    r80_value = (tl.load($(vto_copy_27_ptr) + ((((r80_z / $(131072)) % $(2)) * $(131072)) + (((r80_z / $(16384)) % $(8)) * $(16384)) + (((r80_z / $(128)) % $(128)) * $(128)) + (r80_z % $(128))), dtype=tl.bfloat16) * tl.cast(0.07216878364870322, tl.bfloat16))
    r80_value = r80_value.to(tl.bfloat16)
    r80_value = tl.broadcast_to(r80_value, [$(262144)])
    tl.store($(vmul_26_ptr) + r80_z, r80_value)
    r81_row = tl.arange($(192))
    r81_col = tl.arange($(128))
    r81_lane = tl.arange($(128))
    for r81_group in range($(16)) {
      r81_left = tl.load($(vcat_4_ptr) + (((((((r81_group % $(16)) * $(24576)) + (r81_lane[None, :] * $(192)) + r81_row[:, None]) / $(196608)) % $(2)) * $(196608)) + ((((((r81_group % $(16)) * $(24576)) + (r81_lane[None, :] * $(192)) + r81_row[:, None]) / $(24576)) % $(8)) * $(24576)) + ((((((r81_group % $(16)) * $(24576)) + (r81_lane[None, :] * $(192)) + r81_row[:, None]) / $(192)) % $(128)) * $(192)) + ((((r81_group % $(16)) * $(24576)) + (r81_lane[None, :] * $(192)) + r81_row[:, None]) % $(192))), dtype=tl.bfloat16)
      r81_right = tl.load($(vmul_26_ptr) + (((((((r81_group % $(16)) * $(16384)) + (r81_lane[:, None] * $(128)) + r81_col[None, :]) / $(131072)) % $(2)) * $(131072)) + ((((((r81_group % $(16)) * $(16384)) + (r81_lane[:, None] * $(128)) + r81_col[None, :]) / $(16384)) % $(8)) * $(16384)) + ((((((r81_group % $(16)) * $(16384)) + (r81_lane[:, None] * $(128)) + r81_col[None, :]) / $(128)) % $(128)) * $(128)) + ((((r81_group % $(16)) * $(16384)) + (r81_lane[:, None] * $(128)) + r81_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r81_value = tl.dot(r81_left, r81_right)
      r81_value = r81_value.to(tl.bfloat16)
      tl.store($(vbmm_6_ptr) + ((r81_group * $(24576)) + (r81_row[:, None] * $(128)) + r81_col[None, :]), r81_value)
    }
    r82_row = tl.arange($(128))
    r82_col = tl.arange($(192))
    r82_lane = tl.arange($(128))
    for r82_group in range($(16)) {
      r82_left = tl.load($(vmul_26_ptr) + (((((((r82_group % $(16)) * $(16384)) + (r82_row[:, None] * $(128)) + r82_lane[None, :]) / $(131072)) % $(2)) * $(131072)) + ((((((r82_group % $(16)) * $(16384)) + (r82_row[:, None] * $(128)) + r82_lane[None, :]) / $(16384)) % $(8)) * $(16384)) + ((((((r82_group % $(16)) * $(16384)) + (r82_row[:, None] * $(128)) + r82_lane[None, :]) / $(128)) % $(128)) * $(128)) + ((((r82_group % $(16)) * $(16384)) + (r82_row[:, None] * $(128)) + r82_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r82_right = tl.load($(vcat_5_ptr) + (((((((r82_group % $(16)) * $(24576)) + (r82_col[None, :] * $(128)) + r82_lane[:, None]) / $(196608)) % $(2)) * $(196608)) + ((((((r82_group % $(16)) * $(24576)) + (r82_col[None, :] * $(128)) + r82_lane[:, None]) / $(24576)) % $(8)) * $(24576)) + (((((r82_group % $(16)) * $(24576)) + (r82_col[None, :] * $(128)) + r82_lane[:, None]) % $(128)) * $(192)) + (((((r82_group % $(16)) * $(24576)) + (r82_col[None, :] * $(128)) + r82_lane[:, None]) / $(128)) % $(192))), dtype=tl.bfloat16)
      r82_value = tl.dot(r82_left, r82_right)
      r82_value = r82_value.to(tl.bfloat16)
      tl.store($(vbmm_7_ptr) + ((r82_group * $(24576)) + (r82_row[:, None] * $(192)) + r82_col[None, :]), r82_value)
    }
    r83_z = tl.arange($(16384))
    r83_lane = tl.arange($(8))
    r83_value = tl.sum(tl.load($(vbmm_6_ptr) + ((((r83_z[:, None] / $(8192)) % $(2)) * $(196608)) + (r83_lane[None, :] * $(24576)) + ((r83_z[:, None] % $(64)) * $(128)) + ((r83_z[:, None] / $(64)) % $(128)) + $(16384)), dtype=tl.bfloat16), axis=1)
    r83_value = r83_value.to(tl.bfloat16)
    r83_value = tl.broadcast_to(r83_value, [$(16384)])
    tl.store($(vsum_1_ptr) + r83_z, r83_value)
    r84_z = tl.arange($(8192))
    r84_value = (tl.load($(vsum_1_ptr) + ((((r84_z / $(4096)) % $(2)) * $(8192)) + (((r84_z / $(32)) % $(128)) * $(64)) + (r84_z % $(32)) + $(32)), dtype=tl.bfloat16) * tl.load($(vto_copy_18_ptr) + ((((r84_z / $(32)) % $(128)) * $(64)) + (r84_z % $(32))), dtype=tl.bfloat16))
    r84_value = r84_value.to(tl.bfloat16)
    r84_value = tl.broadcast_to(r84_value, [$(8192)])
    tl.store($(vmul_27_ptr) + r84_z, r84_value)
    r85_z = tl.arange($(8192))
    r85_value = (tl.load($(vsum_1_ptr) + ((((r85_z / $(4096)) % $(2)) * $(8192)) + (((r85_z / $(32)) % $(128)) * $(64)) + (r85_z % $(32)) + $(32)), dtype=tl.bfloat16) * tl.load($(vto_copy_17_ptr) + ((((r85_z / $(32)) % $(128)) * $(64)) + (r85_z % $(32))), dtype=tl.bfloat16))
    r85_value = r85_value.to(tl.bfloat16)
    r85_value = tl.broadcast_to(r85_value, [$(8192)])
    tl.store($(vmul_28_ptr) + r85_z, r85_value)
    r86_z = tl.arange($(8192))
    r86_value = -tl.load($(vsum_1_ptr) + ((((r86_z / $(4096)) % $(2)) * $(8192)) + (((r86_z / $(32)) % $(128)) * $(64)) + (r86_z % $(32))), dtype=tl.bfloat16)
    r86_value = r86_value.to(tl.bfloat16)
    r86_value = tl.broadcast_to(r86_value, [$(8192)])
    tl.store($(vneg_ptr) + r86_z, r86_value)
    r87_z = tl.arange($(8192))
    r87_value = (tl.load($(vneg_ptr) + ((((r87_z / $(4096)) % $(2)) * $(4096)) + (((r87_z / $(32)) % $(128)) * $(32)) + (r87_z % $(32))), dtype=tl.bfloat16) * tl.load($(vto_copy_18_ptr) + ((((r87_z / $(32)) % $(128)) * $(64)) + (r87_z % $(32))), dtype=tl.bfloat16))
    r87_value = r87_value.to(tl.bfloat16)
    r87_value = tl.broadcast_to(r87_value, [$(8192)])
    tl.store($(vmul_29_ptr) + r87_z, r87_value)
    r88_z = tl.arange($(8192))
    r88_value = (tl.load($(vmul_28_ptr) + ((((r88_z / $(4096)) % $(2)) * $(4096)) + (((r88_z / $(32)) % $(128)) * $(32)) + (r88_z % $(32))), dtype=tl.bfloat16) + tl.load($(vmul_29_ptr) + ((((r88_z / $(4096)) % $(2)) * $(4096)) + (((r88_z / $(32)) % $(128)) * $(32)) + (r88_z % $(32))), dtype=tl.bfloat16))
    r88_value = r88_value.to(tl.bfloat16)
    r88_value = tl.broadcast_to(r88_value, [$(8192)])
    tl.store($(vadd_7_ptr) + r88_z, r88_value)
    r89_z = tl.arange($(8192))
    r89_value = (tl.load($(vsum_1_ptr) + ((((r89_z / $(4096)) % $(2)) * $(8192)) + (((r89_z / $(32)) % $(128)) * $(64)) + (r89_z % $(32))), dtype=tl.bfloat16) * tl.load($(vto_copy_17_ptr) + ((((r89_z / $(32)) % $(128)) * $(64)) + (r89_z % $(32))), dtype=tl.bfloat16))
    r89_value = r89_value.to(tl.bfloat16)
    r89_value = tl.broadcast_to(r89_value, [$(8192)])
    tl.store($(vmul_30_ptr) + r89_z, r89_value)
    r90_z = tl.arange($(8192))
    r90_value = (tl.load($(vmul_27_ptr) + ((((r90_z / $(4096)) % $(2)) * $(4096)) + (((r90_z / $(32)) % $(128)) * $(32)) + (r90_z % $(32))), dtype=tl.bfloat16) + tl.load($(vmul_30_ptr) + ((((r90_z / $(4096)) % $(2)) * $(4096)) + (((r90_z / $(32)) % $(128)) * $(32)) + (r90_z % $(32))), dtype=tl.bfloat16))
    r90_value = r90_value.to(tl.bfloat16)
    r90_value = tl.broadcast_to(r90_value, [$(8192)])
    tl.store($(vadd_8_ptr) + r90_z, r90_value)
    r91_z = tl.arange($(65536))
    r91_value = (tl.load($(vbmm_7_ptr) + ((((r91_z / $(32768)) % $(2)) * $(196608)) + (((r91_z / $(4096)) % $(8)) * $(24576)) + (((r91_z / $(32)) % $(128)) * $(192)) + (r91_z % $(32)) + $(160)), dtype=tl.bfloat16) * tl.load($(vto_copy_18_ptr) + ((((r91_z / $(32)) % $(128)) * $(64)) + (r91_z % $(32))), dtype=tl.bfloat16))
    r91_value = r91_value.to(tl.bfloat16)
    r91_value = tl.broadcast_to(r91_value, [$(65536)])
    tl.store($(vmul_31_ptr) + r91_z, r91_value)
    r92_z = tl.arange($(65536))
    r92_value = (tl.load($(vbmm_7_ptr) + ((((r92_z / $(32768)) % $(2)) * $(196608)) + (((r92_z / $(4096)) % $(8)) * $(24576)) + (((r92_z / $(32)) % $(128)) * $(192)) + (r92_z % $(32)) + $(160)), dtype=tl.bfloat16) * tl.load($(vto_copy_17_ptr) + ((((r92_z / $(32)) % $(128)) * $(64)) + (r92_z % $(32))), dtype=tl.bfloat16))
    r92_value = r92_value.to(tl.bfloat16)
    r92_value = tl.broadcast_to(r92_value, [$(65536)])
    tl.store($(vmul_32_ptr) + r92_z, r92_value)
    r93_z = tl.arange($(65536))
    r93_value = -tl.load($(vbmm_7_ptr) + ((((r93_z / $(32768)) % $(2)) * $(196608)) + (((r93_z / $(4096)) % $(8)) * $(24576)) + (((r93_z / $(32)) % $(128)) * $(192)) + (r93_z % $(32)) + $(128)), dtype=tl.bfloat16)
    r93_value = r93_value.to(tl.bfloat16)
    r93_value = tl.broadcast_to(r93_value, [$(65536)])
    tl.store($(vneg_1_ptr) + r93_z, r93_value)
    r94_z = tl.arange($(65536))
    r94_value = (tl.load($(vneg_1_ptr) + ((((r94_z / $(32768)) % $(2)) * $(32768)) + (((r94_z / $(4096)) % $(8)) * $(4096)) + (((r94_z / $(32)) % $(128)) * $(32)) + (r94_z % $(32))), dtype=tl.bfloat16) * tl.load($(vto_copy_18_ptr) + ((((r94_z / $(32)) % $(128)) * $(64)) + (r94_z % $(32))), dtype=tl.bfloat16))
    r94_value = r94_value.to(tl.bfloat16)
    r94_value = tl.broadcast_to(r94_value, [$(65536)])
    tl.store($(vmul_33_ptr) + r94_z, r94_value)
    r95_z = tl.arange($(65536))
    r95_value = (tl.load($(vmul_32_ptr) + ((((r95_z / $(32768)) % $(2)) * $(32768)) + (((r95_z / $(4096)) % $(8)) * $(4096)) + (((r95_z / $(32)) % $(128)) * $(32)) + (r95_z % $(32))), dtype=tl.bfloat16) + tl.load($(vmul_33_ptr) + ((((r95_z / $(32768)) % $(2)) * $(32768)) + (((r95_z / $(4096)) % $(8)) * $(4096)) + (((r95_z / $(32)) % $(128)) * $(32)) + (r95_z % $(32))), dtype=tl.bfloat16))
    r95_value = r95_value.to(tl.bfloat16)
    r95_value = tl.broadcast_to(r95_value, [$(65536)])
    tl.store($(vadd_9_ptr) + r95_z, r95_value)
    r96_z = tl.arange($(65536))
    r96_value = (tl.load($(vbmm_7_ptr) + ((((r96_z / $(32768)) % $(2)) * $(196608)) + (((r96_z / $(4096)) % $(8)) * $(24576)) + (((r96_z / $(32)) % $(128)) * $(192)) + (r96_z % $(32)) + $(128)), dtype=tl.bfloat16) * tl.load($(vto_copy_17_ptr) + ((((r96_z / $(32)) % $(128)) * $(64)) + (r96_z % $(32))), dtype=tl.bfloat16))
    r96_value = r96_value.to(tl.bfloat16)
    r96_value = tl.broadcast_to(r96_value, [$(65536)])
    tl.store($(vmul_34_ptr) + r96_z, r96_value)
    r97_z = tl.arange($(65536))
    r97_value = (tl.load($(vmul_31_ptr) + ((((r97_z / $(32768)) % $(2)) * $(32768)) + (((r97_z / $(4096)) % $(8)) * $(4096)) + (((r97_z / $(32)) % $(128)) * $(32)) + (r97_z % $(32))), dtype=tl.bfloat16) + tl.load($(vmul_34_ptr) + ((((r97_z / $(32768)) % $(2)) * $(32768)) + (((r97_z / $(4096)) % $(8)) * $(4096)) + (((r97_z / $(32)) % $(128)) * $(32)) + (r97_z % $(32))), dtype=tl.bfloat16))
    r97_value = r97_value.to(tl.bfloat16)
    r97_value = tl.broadcast_to(r97_value, [$(65536)])
    tl.store($(vadd_10_ptr) + r97_z, r97_value)
    r98_z = tl.arange($(16384))
    r98_value = tl.where((((r98_z % $(64)) >= $(1)) & ((r98_z % $(64)) < $(64)) & (((r98_z % $(64)) - $(1)) % $(2) == $(0))), tl.load($(vadd_7_ptr) + ((((r98_z / $(8192)) % $(2)) * $(4096)) + (((r98_z / $(64)) % $(128)) * $(32)) + ((((r98_z % $(64)) - $(1)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r98_value = r98_value.to(tl.bfloat16)
    r98_value = tl.broadcast_to(r98_value, [$(16384)])
    tl.store($(vslice_backward_ptr) + r98_z, r98_value)
    r99_z = tl.arange($(16384))
    r99_value = tl.where((((r99_z % $(64)) >= $(0)) & ((r99_z % $(64)) < $(64)) & (((r99_z % $(64)) - $(0)) % $(2) == $(0))), tl.load($(vadd_8_ptr) + ((((r99_z / $(8192)) % $(2)) * $(4096)) + (((r99_z / $(64)) % $(128)) * $(32)) + ((((r99_z % $(64)) - $(0)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r99_value = r99_value.to(tl.bfloat16)
    r99_value = tl.broadcast_to(r99_value, [$(16384)])
    tl.store($(vslice_backward_1_ptr) + r99_z, r99_value)
    r100_z = tl.arange($(16384))
    r100_value = (tl.load($(vslice_backward_ptr) + ((((r100_z / $(8192)) % $(2)) * $(8192)) + (((r100_z / $(64)) % $(128)) * $(64)) + (r100_z % $(64))), dtype=tl.bfloat16) + tl.load($(vslice_backward_1_ptr) + ((((r100_z / $(8192)) % $(2)) * $(8192)) + (((r100_z / $(64)) % $(128)) * $(64)) + (r100_z % $(64))), dtype=tl.bfloat16))
    r100_value = r100_value.to(tl.bfloat16)
    r100_value = tl.broadcast_to(r100_value, [$(16384)])
    tl.store($(vadd_11_ptr) + r100_z, r100_value)
    r101_z = tl.arange($(131072))
    r101_value = tl.where((((r101_z % $(64)) >= $(1)) & ((r101_z % $(64)) < $(64)) & (((r101_z % $(64)) - $(1)) % $(2) == $(0))), tl.load($(vadd_9_ptr) + ((((r101_z / $(65536)) % $(2)) * $(32768)) + (((r101_z / $(8192)) % $(8)) * $(4096)) + (((r101_z / $(64)) % $(128)) * $(32)) + ((((r101_z % $(64)) - $(1)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r101_value = r101_value.to(tl.bfloat16)
    r101_value = tl.broadcast_to(r101_value, [$(131072)])
    tl.store($(vslice_backward_2_ptr) + r101_z, r101_value)
    r102_z = tl.arange($(131072))
    r102_value = tl.where((((r102_z % $(64)) >= $(0)) & ((r102_z % $(64)) < $(64)) & (((r102_z % $(64)) - $(0)) % $(2) == $(0))), tl.load($(vadd_10_ptr) + ((((r102_z / $(65536)) % $(2)) * $(32768)) + (((r102_z / $(8192)) % $(8)) * $(4096)) + (((r102_z / $(64)) % $(128)) * $(32)) + ((((r102_z % $(64)) - $(0)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r102_value = r102_value.to(tl.bfloat16)
    r102_value = tl.broadcast_to(r102_value, [$(131072)])
    tl.store($(vslice_backward_3_ptr) + r102_z, r102_value)
    r103_z = tl.arange($(131072))
    r103_value = (tl.load($(vslice_backward_2_ptr) + ((((r103_z / $(65536)) % $(2)) * $(65536)) + (((r103_z / $(8192)) % $(8)) * $(8192)) + (((r103_z / $(64)) % $(128)) * $(64)) + (r103_z % $(64))), dtype=tl.bfloat16) + tl.load($(vslice_backward_3_ptr) + ((((r103_z / $(65536)) % $(2)) * $(65536)) + (((r103_z / $(8192)) % $(8)) * $(8192)) + (((r103_z / $(64)) % $(128)) * $(64)) + (r103_z % $(64))), dtype=tl.bfloat16))
    r103_value = r103_value.to(tl.bfloat16)
    r103_value = tl.broadcast_to(r103_value, [$(131072)])
    tl.store($(vadd_12_ptr) + r103_z, r103_value)
    r104_z = tl.arange($(524288))
    r104_value = tl.where((r104_z % $(256)) < $(128), tl.load($(vbmm_6_ptr) + ((((((((r104_z / $(262144)) % $(2)) * $(196608)) + (((r104_z / $(32768)) % $(8)) * $(24576)) + (((r104_z % $(256)) - $(0)) * $(128)) + ((r104_z / $(256)) % $(128))) / $(24576)) % $(16)) * $(24576)) + ((((((r104_z / $(262144)) % $(2)) * $(1536)) + (((r104_z / $(32768)) % $(8)) * $(192)) + ((r104_z % $(256)) - $(0))) % $(192)) * $(128)) + ((r104_z / $(256)) % $(128))), dtype=tl.bfloat16), tl.load($(vbmm_4_ptr) + ((((((((r104_z / $(262144)) % $(2)) * $(131072)) + (((r104_z / $(32768)) % $(8)) * $(16384)) + (((r104_z / $(256)) % $(128)) * $(128)) + ((r104_z % $(256)) - $(128))) / $(16384)) % $(16)) * $(16384)) + (((((((r104_z / $(262144)) % $(2)) * $(131072)) + (((r104_z / $(32768)) % $(8)) * $(16384)) + (((r104_z / $(256)) % $(128)) * $(128)) + ((r104_z % $(256)) - $(128))) / $(128)) % $(128)) * $(128)) + (((((r104_z / $(262144)) % $(2)) * $(131072)) + (((r104_z / $(32768)) % $(8)) * $(16384)) + (((r104_z / $(256)) % $(128)) * $(128)) + ((r104_z % $(256)) - $(128))) % $(128))), dtype=tl.bfloat16))
    r104_value = r104_value.to(tl.bfloat16)
    r104_value = tl.broadcast_to(r104_value, [$(524288)])
    tl.store($(vcat_6_ptr) + r104_z, r104_value)
    r105_z = tl.arange($(393216))
    r105_value = tl.where((r105_z % $(192)) < $(128), tl.load($(vbmm_7_ptr) + ((((((((r105_z / $(196608)) % $(2)) * $(196608)) + (((r105_z / $(24576)) % $(8)) * $(24576)) + (((r105_z / $(192)) % $(128)) * $(192)) + ((r105_z % $(192)) - $(0))) / $(24576)) % $(16)) * $(24576)) + (((((((r105_z / $(196608)) % $(2)) * $(196608)) + (((r105_z / $(24576)) % $(8)) * $(24576)) + (((r105_z / $(192)) % $(128)) * $(192)) + ((r105_z % $(192)) - $(0))) / $(192)) % $(128)) * $(192)) + (((((r105_z / $(196608)) % $(2)) * $(196608)) + (((r105_z / $(24576)) % $(8)) * $(24576)) + (((r105_z / $(192)) % $(128)) * $(192)) + ((r105_z % $(192)) - $(0))) % $(192))), dtype=tl.bfloat16), tl.load($(vadd_12_ptr) + ((((r105_z / $(196608)) % $(2)) * $(65536)) + (((r105_z / $(24576)) % $(8)) * $(8192)) + (((r105_z / $(192)) % $(128)) * $(64)) + (((r105_z % $(192)) - $(128)) % $(64))), dtype=tl.bfloat16))
    r105_value = r105_value.to(tl.bfloat16)
    r105_value = tl.broadcast_to(r105_value, [$(393216)])
    tl.store($(vcat_7_ptr) + r105_z, r105_value)
    r106_row = tl.arange($(256))
    r106_col = tl.arange($(256))
    r106_lane = tl.arange($(2048))
    r106_left = tl.load($(vcat_6_ptr) + ((((((r106_row[:, None] * $(2048)) + r106_lane[None, :]) / $(262144)) % $(2)) * $(262144)) + (((((((((r106_row[:, None] * $(2048)) + r106_lane[None, :]) / $(262144)) % $(2)) * $(262144)) + (((((r106_row[:, None] * $(2048)) + r106_lane[None, :]) / $(2048)) % $(128)) * $(2048)) + (((r106_row[:, None] * $(2048)) + r106_lane[None, :]) % $(2048))) / $(256)) % $(8)) * $(32768)) + (((((r106_row[:, None] * $(2048)) + r106_lane[None, :]) / $(2048)) % $(128)) * $(256)) + (((((((r106_row[:, None] * $(2048)) + r106_lane[None, :]) / $(262144)) % $(2)) * $(262144)) + (((((r106_row[:, None] * $(2048)) + r106_lane[None, :]) / $(2048)) % $(128)) * $(2048)) + (((r106_row[:, None] * $(2048)) + r106_lane[None, :]) % $(2048))) % $(256))), dtype=tl.bfloat16)
    r106_right = tl.load($(vto_copy_5_ptr) + (((r106_lane[:, None] % $(2048)) * $(256)) + (r106_col[None, :] % $(256))), dtype=tl.bfloat16)
    r106_value = tl.dot(r106_left, r106_right)
    r106_value = r106_value.to(tl.bfloat16)
    tl.store($(vmm_6_ptr) + ((r106_row[:, None] * $(256)) + r106_col[None, :]), r106_value)
    r107_z = tl.arange($(65536))
    r107_value = tl.load($(vmm_6_ptr) + ((((r107_z / $(32768)) % $(2)) * $(32768)) + (((r107_z / $(256)) % $(128)) * $(256)) + (r107_z % $(256))), dtype=tl.bfloat16)
    r107_value = r107_value.to(tl.float32)
    r107_value = tl.broadcast_to(r107_value, [$(65536)])
    tl.store($(vto_copy_28_ptr) + r107_z, r107_value)
    r108_z = tl.arange($(65536))
    r108_value = (tl.load($(vto_copy_28_ptr) + ((((r108_z / $(32768)) % $(2)) * $(32768)) + (((r108_z / $(256)) % $(128)) * $(256)) + (r108_z % $(256))), dtype=tl.float32) * tl.cast(tl.load($(vto_copy_4_ptr) + (r108_z % $(256)), dtype=tl.bfloat16), tl.float32))
    r108_value = r108_value.to(tl.float32)
    r108_value = tl.broadcast_to(r108_value, [$(65536)])
    tl.store($(vmul_35_ptr) + r108_z, r108_value)
    r109_z = tl.arange($(65536))
    r109_value = (tl.load($(vmul_35_ptr) + ((((r109_z / $(32768)) % $(2)) * $(32768)) + (((r109_z / $(256)) % $(128)) * $(256)) + (r109_z % $(256))), dtype=tl.float32) * tl.load($(vto_copy_22_ptr) + ((((r109_z / $(32768)) % $(2)) * $(32768)) + (((r109_z / $(256)) % $(128)) * $(256)) + (r109_z % $(256))), dtype=tl.float32))
    r109_value = r109_value.to(tl.float32)
    r109_value = tl.broadcast_to(r109_value, [$(65536)])
    tl.store($(vmul_36_ptr) + r109_z, r109_value)
    r110_z = tl.arange($(65536))
    r110_value = (tl.load($(vmul_35_ptr) + ((((r110_z / $(32768)) % $(2)) * $(32768)) + (((r110_z / $(256)) % $(128)) * $(256)) + (r110_z % $(256))), dtype=tl.float32) * tl.load($(vrsqrt_1_ptr) + ((((r110_z / $(32768)) % $(2)) * $(128)) + ((r110_z / $(256)) % $(128))), dtype=tl.float32))
    r110_value = r110_value.to(tl.float32)
    r110_value = tl.broadcast_to(r110_value, [$(65536)])
    tl.store($(vmul_37_ptr) + r110_z, r110_value)
    r111_z = tl.arange($(256))
    r111_lane = tl.arange($(256))
    r111_value = tl.sum(tl.load($(vmul_36_ptr) + ((((r111_z[:, None] / $(128)) % $(2)) * $(32768)) + ((r111_z[:, None] % $(128)) * $(256)) + r111_lane[None, :]), dtype=tl.float32), axis=1)
    r111_value = r111_value.to(tl.float32)
    r111_value = tl.broadcast_to(r111_value, [$(256)])
    tl.store($(vsum_2_ptr) + r111_z, r111_value)
    r112_z = tl.arange($(256))
    r112_value = ((tl.load($(vrsqrt_1_ptr) + ((((r112_z / $(128)) % $(2)) * $(128)) + (r112_z % $(128))), dtype=tl.float32)) * (tl.load($(vrsqrt_1_ptr) + ((((r112_z / $(128)) % $(2)) * $(128)) + (r112_z % $(128))), dtype=tl.float32)) * (tl.load($(vrsqrt_1_ptr) + ((((r112_z / $(128)) % $(2)) * $(128)) + (r112_z % $(128))), dtype=tl.float32)))
    r112_value = r112_value.to(tl.float32)
    r112_value = tl.broadcast_to(r112_value, [$(256)])
    tl.store($(vpow_5_ptr) + r112_z, r112_value)
    r113_z = tl.arange($(256))
    r113_value = (tl.load($(vsum_2_ptr) + ((((r113_z / $(128)) % $(2)) * $(128)) + (r113_z % $(128))), dtype=tl.float32) * -0.5)
    r113_value = r113_value.to(tl.float32)
    r113_value = tl.broadcast_to(r113_value, [$(256)])
    tl.store($(vmul_38_ptr) + r113_z, r113_value)
    r114_z = tl.arange($(256))
    r114_value = (tl.load($(vmul_38_ptr) + ((((r114_z / $(128)) % $(2)) * $(128)) + (r114_z % $(128))), dtype=tl.float32) * tl.load($(vpow_5_ptr) + ((((r114_z / $(128)) % $(2)) * $(128)) + (r114_z % $(128))), dtype=tl.float32))
    r114_value = r114_value.to(tl.float32)
    r114_value = tl.broadcast_to(r114_value, [$(256)])
    tl.store($(vmul_39_ptr) + r114_z, r114_value)
    r115_z = tl.arange($(65536))
    r115_value = (tl.load($(vmul_39_ptr) + ((((r115_z / $(32768)) % $(2)) * $(128)) + ((r115_z / $(256)) % $(128))), dtype=tl.float32) / 256.0)
    r115_value = r115_value.to(tl.float32)
    r115_value = tl.broadcast_to(r115_value, [$(65536)])
    tl.store($(vdiv_2_ptr) + r115_z, r115_value)
    r116_z = tl.arange($(65536))
    r116_value = ((tl.load($(vto_copy_22_ptr) + ((((r116_z / $(32768)) % $(2)) * $(32768)) + (((r116_z / $(256)) % $(128)) * $(256)) + (r116_z % $(256))), dtype=tl.float32)))
    r116_value = r116_value.to(tl.float32)
    r116_value = tl.broadcast_to(r116_value, [$(65536)])
    tl.store($(vpow_6_ptr) + r116_z, r116_value)
    r117_z = tl.arange($(65536))
    r117_value = (tl.load($(vpow_6_ptr) + ((((r117_z / $(32768)) % $(2)) * $(32768)) + (((r117_z / $(256)) % $(128)) * $(256)) + (r117_z % $(256))), dtype=tl.float32) * 2.0)
    r117_value = r117_value.to(tl.float32)
    r117_value = tl.broadcast_to(r117_value, [$(65536)])
    tl.store($(vmul_40_ptr) + r117_z, r117_value)
    r118_z = tl.arange($(65536))
    r118_value = (tl.load($(vdiv_2_ptr) + ((((r118_z / $(32768)) % $(2)) * $(32768)) + (((r118_z / $(256)) % $(128)) * $(256)) + (r118_z % $(256))), dtype=tl.float32) * tl.load($(vmul_40_ptr) + ((((r118_z / $(32768)) % $(2)) * $(32768)) + (((r118_z / $(256)) % $(128)) * $(256)) + (r118_z % $(256))), dtype=tl.float32))
    r118_value = r118_value.to(tl.float32)
    r118_value = tl.broadcast_to(r118_value, [$(65536)])
    tl.store($(vmul_41_ptr) + r118_z, r118_value)
    r119_z = tl.arange($(65536))
    r119_value = (tl.load($(vmul_37_ptr) + ((((r119_z / $(32768)) % $(2)) * $(32768)) + (((r119_z / $(256)) % $(128)) * $(256)) + (r119_z % $(256))), dtype=tl.float32) + tl.load($(vmul_41_ptr) + ((((r119_z / $(32768)) % $(2)) * $(32768)) + (((r119_z / $(256)) % $(128)) * $(256)) + (r119_z % $(256))), dtype=tl.float32))
    r119_value = r119_value.to(tl.float32)
    r119_value = tl.broadcast_to(r119_value, [$(65536)])
    tl.store($(vadd_13_ptr) + r119_z, r119_value)
    r120_z = tl.arange($(65536))
    r120_value = tl.load($(vadd_13_ptr) + ((((r120_z / $(32768)) % $(2)) * $(32768)) + (((r120_z / $(256)) % $(128)) * $(256)) + (r120_z % $(256))), dtype=tl.float32)
    r120_value = r120_value.to(tl.bfloat16)
    r120_value = tl.broadcast_to(r120_value, [$(65536)])
    tl.store($(vto_copy_29_ptr) + r120_z, r120_value)
    r121_z = tl.arange($(81920))
    r121_value = tl.where((r121_z % $(320)) < $(256), tl.load($(vto_copy_29_ptr) + ((((r121_z / $(40960)) % $(2)) * $(32768)) + (((r121_z / $(320)) % $(128)) * $(256)) + (((r121_z % $(320)) - $(0)) % $(256))), dtype=tl.bfloat16), tl.load($(vadd_11_ptr) + ((((((((r121_z / $(40960)) % $(2)) * $(8192)) + (((r121_z / $(320)) % $(128)) * $(64)) + ((r121_z % $(320)) - $(256))) / $(8192)) % $(2)) * $(8192)) + (((((((r121_z / $(40960)) % $(2)) * $(8192)) + (((r121_z / $(320)) % $(128)) * $(64)) + ((r121_z % $(320)) - $(256))) / $(64)) % $(128)) * $(64)) + (((((r121_z / $(40960)) % $(2)) * $(8192)) + (((r121_z / $(320)) % $(128)) * $(64)) + ((r121_z % $(320)) - $(256))) % $(64))), dtype=tl.bfloat16))
    r121_value = r121_value.to(tl.bfloat16)
    r121_value = tl.broadcast_to(r121_value, [$(81920)])
    tl.store($(vcat_8_ptr) + r121_z, r121_value)
    r122_row = tl.arange($(256))
    r122_col = tl.arange($(512))
    r122_lane = tl.arange($(320))
    r122_left = tl.load($(vcat_8_ptr) + ((((((r122_row[:, None] * $(320)) + r122_lane[None, :]) / $(40960)) % $(2)) * $(40960)) + (((((r122_row[:, None] * $(320)) + r122_lane[None, :]) / $(320)) % $(128)) * $(320)) + (((r122_row[:, None] * $(320)) + r122_lane[None, :]) % $(320))), dtype=tl.bfloat16)
    r122_right = tl.load($(vto_copy_3_ptr) + (((r122_lane[:, None] % $(320)) * $(512)) + (r122_col[None, :] % $(512))), dtype=tl.bfloat16)
    r122_value = tl.dot(r122_left, r122_right)
    r122_value = r122_value.to(tl.bfloat16)
    tl.store($(vmm_7_ptr) + ((r122_row[:, None] * $(512)) + r122_col[None, :]), r122_value)
    r123_row = tl.arange($(256))
    r123_col = tl.arange($(256))
    r123_lane = tl.arange($(1536))
    r123_left = tl.load($(vcat_7_ptr) + ((((((r123_row[:, None] * $(1536)) + r123_lane[None, :]) / $(196608)) % $(2)) * $(196608)) + (((((((((r123_row[:, None] * $(1536)) + r123_lane[None, :]) / $(196608)) % $(2)) * $(196608)) + (((((r123_row[:, None] * $(1536)) + r123_lane[None, :]) / $(1536)) % $(128)) * $(1536)) + (((r123_row[:, None] * $(1536)) + r123_lane[None, :]) % $(1536))) / $(192)) % $(8)) * $(24576)) + (((((r123_row[:, None] * $(1536)) + r123_lane[None, :]) / $(1536)) % $(128)) * $(192)) + (((((((r123_row[:, None] * $(1536)) + r123_lane[None, :]) / $(196608)) % $(2)) * $(196608)) + (((((r123_row[:, None] * $(1536)) + r123_lane[None, :]) / $(1536)) % $(128)) * $(1536)) + (((r123_row[:, None] * $(1536)) + r123_lane[None, :]) % $(1536))) % $(192))), dtype=tl.bfloat16)
    r123_right = tl.load($(vto_copy_2_ptr) + (((r123_lane[:, None] % $(1536)) * $(256)) + (r123_col[None, :] % $(256))), dtype=tl.bfloat16)
    r123_value = tl.dot(r123_left, r123_right)
    r123_value = r123_value.to(tl.bfloat16)
    tl.store($(vmm_8_ptr) + ((r123_row[:, None] * $(256)) + r123_col[None, :]), r123_value)
    r124_z = tl.arange($(65536))
    r124_value = tl.load($(vmm_8_ptr) + ((((r124_z / $(32768)) % $(2)) * $(32768)) + (((r124_z / $(256)) % $(128)) * $(256)) + (r124_z % $(256))), dtype=tl.bfloat16)
    r124_value = r124_value.to(tl.float32)
    r124_value = tl.broadcast_to(r124_value, [$(65536)])
    tl.store($(vto_copy_30_ptr) + r124_z, r124_value)
    r125_z = tl.arange($(65536))
    r125_value = (tl.load($(vto_copy_30_ptr) + ((((r125_z / $(32768)) % $(2)) * $(32768)) + (((r125_z / $(256)) % $(128)) * $(256)) + (r125_z % $(256))), dtype=tl.float32) * tl.cast(tl.load($(vto_copy_1_ptr) + (r125_z % $(256)), dtype=tl.bfloat16), tl.float32))
    r125_value = r125_value.to(tl.float32)
    r125_value = tl.broadcast_to(r125_value, [$(65536)])
    tl.store($(vmul_42_ptr) + r125_z, r125_value)
    r126_z = tl.arange($(65536))
    r126_value = (tl.load($(vmul_42_ptr) + ((((r126_z / $(32768)) % $(2)) * $(32768)) + (((r126_z / $(256)) % $(128)) * $(256)) + (r126_z % $(256))), dtype=tl.float32) * tl.load($(vto_copy_20_ptr) + ((((r126_z / $(32768)) % $(2)) * $(32768)) + (((r126_z / $(256)) % $(128)) * $(256)) + (r126_z % $(256))), dtype=tl.float32))
    r126_value = r126_value.to(tl.float32)
    r126_value = tl.broadcast_to(r126_value, [$(65536)])
    tl.store($(vmul_43_ptr) + r126_z, r126_value)
    r127_z = tl.arange($(65536))
    r127_value = (tl.load($(vmul_42_ptr) + ((((r127_z / $(32768)) % $(2)) * $(32768)) + (((r127_z / $(256)) % $(128)) * $(256)) + (r127_z % $(256))), dtype=tl.float32) * tl.load($(vrsqrt_ptr) + ((((r127_z / $(32768)) % $(2)) * $(128)) + ((r127_z / $(256)) % $(128))), dtype=tl.float32))
    r127_value = r127_value.to(tl.float32)
    r127_value = tl.broadcast_to(r127_value, [$(65536)])
    tl.store($(vmul_44_ptr) + r127_z, r127_value)
    r128_z = tl.arange($(256))
    r128_lane = tl.arange($(256))
    r128_value = tl.sum(tl.load($(vmul_43_ptr) + ((((r128_z[:, None] / $(128)) % $(2)) * $(32768)) + ((r128_z[:, None] % $(128)) * $(256)) + r128_lane[None, :]), dtype=tl.float32), axis=1)
    r128_value = r128_value.to(tl.float32)
    r128_value = tl.broadcast_to(r128_value, [$(256)])
    tl.store($(vsum_3_ptr) + r128_z, r128_value)
    r129_z = tl.arange($(256))
    r129_value = ((tl.load($(vrsqrt_ptr) + ((((r129_z / $(128)) % $(2)) * $(128)) + (r129_z % $(128))), dtype=tl.float32)) * (tl.load($(vrsqrt_ptr) + ((((r129_z / $(128)) % $(2)) * $(128)) + (r129_z % $(128))), dtype=tl.float32)) * (tl.load($(vrsqrt_ptr) + ((((r129_z / $(128)) % $(2)) * $(128)) + (r129_z % $(128))), dtype=tl.float32)))
    r129_value = r129_value.to(tl.float32)
    r129_value = tl.broadcast_to(r129_value, [$(256)])
    tl.store($(vpow_7_ptr) + r129_z, r129_value)
    r130_z = tl.arange($(256))
    r130_value = (tl.load($(vsum_3_ptr) + ((((r130_z / $(128)) % $(2)) * $(128)) + (r130_z % $(128))), dtype=tl.float32) * -0.5)
    r130_value = r130_value.to(tl.float32)
    r130_value = tl.broadcast_to(r130_value, [$(256)])
    tl.store($(vmul_45_ptr) + r130_z, r130_value)
    r131_z = tl.arange($(256))
    r131_value = (tl.load($(vmul_45_ptr) + ((((r131_z / $(128)) % $(2)) * $(128)) + (r131_z % $(128))), dtype=tl.float32) * tl.load($(vpow_7_ptr) + ((((r131_z / $(128)) % $(2)) * $(128)) + (r131_z % $(128))), dtype=tl.float32))
    r131_value = r131_value.to(tl.float32)
    r131_value = tl.broadcast_to(r131_value, [$(256)])
    tl.store($(vmul_46_ptr) + r131_z, r131_value)
    r132_z = tl.arange($(65536))
    r132_value = (tl.load($(vmul_46_ptr) + ((((r132_z / $(32768)) % $(2)) * $(128)) + ((r132_z / $(256)) % $(128))), dtype=tl.float32) / 256.0)
    r132_value = r132_value.to(tl.float32)
    r132_value = tl.broadcast_to(r132_value, [$(65536)])
    tl.store($(vdiv_3_ptr) + r132_z, r132_value)
    r133_z = tl.arange($(65536))
    r133_value = ((tl.load($(vto_copy_20_ptr) + ((((r133_z / $(32768)) % $(2)) * $(32768)) + (((r133_z / $(256)) % $(128)) * $(256)) + (r133_z % $(256))), dtype=tl.float32)))
    r133_value = r133_value.to(tl.float32)
    r133_value = tl.broadcast_to(r133_value, [$(65536)])
    tl.store($(vpow_8_ptr) + r133_z, r133_value)
    r134_z = tl.arange($(65536))
    r134_value = (tl.load($(vpow_8_ptr) + ((((r134_z / $(32768)) % $(2)) * $(32768)) + (((r134_z / $(256)) % $(128)) * $(256)) + (r134_z % $(256))), dtype=tl.float32) * 2.0)
    r134_value = r134_value.to(tl.float32)
    r134_value = tl.broadcast_to(r134_value, [$(65536)])
    tl.store($(vmul_47_ptr) + r134_z, r134_value)
    r135_z = tl.arange($(65536))
    r135_value = (tl.load($(vdiv_3_ptr) + ((((r135_z / $(32768)) % $(2)) * $(32768)) + (((r135_z / $(256)) % $(128)) * $(256)) + (r135_z % $(256))), dtype=tl.float32) * tl.load($(vmul_47_ptr) + ((((r135_z / $(32768)) % $(2)) * $(32768)) + (((r135_z / $(256)) % $(128)) * $(256)) + (r135_z % $(256))), dtype=tl.float32))
    r135_value = r135_value.to(tl.float32)
    r135_value = tl.broadcast_to(r135_value, [$(65536)])
    tl.store($(vmul_48_ptr) + r135_z, r135_value)
    r136_z = tl.arange($(65536))
    r136_value = (tl.load($(vmul_44_ptr) + ((((r136_z / $(32768)) % $(2)) * $(32768)) + (((r136_z / $(256)) % $(128)) * $(256)) + (r136_z % $(256))), dtype=tl.float32) + tl.load($(vmul_48_ptr) + ((((r136_z / $(32768)) % $(2)) * $(32768)) + (((r136_z / $(256)) % $(128)) * $(256)) + (r136_z % $(256))), dtype=tl.float32))
    r136_value = r136_value.to(tl.float32)
    r136_value = tl.broadcast_to(r136_value, [$(65536)])
    tl.store($(vadd_14_ptr) + r136_z, r136_value)
    r137_z = tl.arange($(65536))
    r137_value = tl.load($(vadd_14_ptr) + ((((r137_z / $(32768)) % $(2)) * $(32768)) + (((r137_z / $(256)) % $(128)) * $(256)) + (r137_z % $(256))), dtype=tl.float32)
    r137_value = r137_value.to(tl.bfloat16)
    r137_value = tl.broadcast_to(r137_value, [$(65536)])
    tl.store($(vto_copy_31_ptr) + r137_z, r137_value)
    r138_row = tl.arange($(256))
    r138_col = tl.arange($(512))
    r138_lane = tl.arange($(256))
    r138_left = tl.load($(vto_copy_31_ptr) + ((((((r138_row[:, None] * $(256)) + r138_lane[None, :]) / $(32768)) % $(2)) * $(32768)) + (((((r138_row[:, None] * $(256)) + r138_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r138_row[:, None] * $(256)) + r138_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r138_right = tl.load($(vto_copy_ptr) + (((r138_lane[:, None] % $(256)) * $(512)) + (r138_col[None, :] % $(512))), dtype=tl.bfloat16)
    r138_value = tl.dot(r138_left, r138_right)
    r138_value = r138_value.to(tl.bfloat16)
    tl.store($(vmm_9_ptr) + ((r138_row[:, None] * $(512)) + r138_col[None, :]), r138_value)
    r139_z = tl.arange($(131072))
    r139_value = (tl.load($(vmm_7_ptr) + ((((r139_z / $(65536)) % $(2)) * $(65536)) + (((r139_z / $(512)) % $(128)) * $(512)) + (r139_z % $(512))), dtype=tl.bfloat16) + tl.load($(vmm_9_ptr) + ((((r139_z / $(65536)) % $(2)) * $(65536)) + (((r139_z / $(512)) % $(128)) * $(512)) + (r139_z % $(512))), dtype=tl.bfloat16))
    r139_value = r139_value.to(tl.bfloat16)
    r139_value = tl.broadcast_to(r139_value, [$(131072)])
    tl.store($(vadd_15_ptr) + r139_z, r139_value)
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
  let vto_copy_20_ptr : RegionName := "vto_copy_20"
  let vpow_3_ptr : RegionName := "vpow_3"
  let vmean_ptr : RegionName := "vmean"
  let vadd_2_ptr : RegionName := "vadd_2"
  let vrsqrt_ptr : RegionName := "vrsqrt"
  let vmul_13_ptr : RegionName := "vmul_13"
  let vmul_14_ptr : RegionName := "vmul_14"
  let vto_copy_21_ptr : RegionName := "vto_copy_21"
  let vmm_1_ptr : RegionName := "vmm_1"
  let vmm_2_ptr : RegionName := "vmm_2"
  let vto_copy_22_ptr : RegionName := "vto_copy_22"
  let vpow_4_ptr : RegionName := "vpow_4"
  let vmean_1_ptr : RegionName := "vmean_1"
  let vadd_3_ptr : RegionName := "vadd_3"
  let vrsqrt_1_ptr : RegionName := "vrsqrt_1"
  let vmul_15_ptr : RegionName := "vmul_15"
  let vmul_16_ptr : RegionName := "vmul_16"
  let vto_copy_23_ptr : RegionName := "vto_copy_23"
  let vmul_17_ptr : RegionName := "vmul_17"
  let vmul_18_ptr : RegionName := "vmul_18"
  let vsub_ptr : RegionName := "vsub"
  let vmul_19_ptr : RegionName := "vmul_19"
  let vmul_20_ptr : RegionName := "vmul_20"
  let vadd_4_ptr : RegionName := "vadd_4"
  let vcat_2_ptr : RegionName := "vcat_2"
  let vmul_21_ptr : RegionName := "vmul_21"
  let vmul_22_ptr : RegionName := "vmul_22"
  let vsub_1_ptr : RegionName := "vsub_1"
  let vmul_23_ptr : RegionName := "vmul_23"
  let vmul_24_ptr : RegionName := "vmul_24"
  let vadd_5_ptr : RegionName := "vadd_5"
  let vcat_3_ptr : RegionName := "vcat_3"
  let vcat_5_ptr : RegionName := "vcat_5"
  let vbmm_2_ptr : RegionName := "vbmm_2"
  let vmul_25_ptr : RegionName := "vmul_25"
  let vadd_6_ptr : RegionName := "vadd_6"
  let vto_copy_24_ptr : RegionName := "vto_copy_24"
  let vsoftmax_ptr : RegionName := "vsoftmax"
  let vto_copy_25_ptr : RegionName := "vto_copy_25"
  let vmm_5_ptr : RegionName := "vmm_5"
  let vbmm_4_ptr : RegionName := "vbmm_4"
  let vto_copy_26_ptr : RegionName := "vto_copy_26"
  let vsoftmax_backward_data_ptr : RegionName := "vsoftmax_backward_data"
  let vto_copy_27_ptr : RegionName := "vto_copy_27"
  let vmul_26_ptr : RegionName := "vmul_26"
  let vbmm_6_ptr : RegionName := "vbmm_6"
  let vsum_1_ptr : RegionName := "vsum_1"
  let vmul_27_ptr : RegionName := "vmul_27"
  let vmul_28_ptr : RegionName := "vmul_28"
  let vneg_ptr : RegionName := "vneg"
  let vmul_29_ptr : RegionName := "vmul_29"
  let vadd_7_ptr : RegionName := "vadd_7"
  let vmul_30_ptr : RegionName := "vmul_30"
  let vadd_8_ptr : RegionName := "vadd_8"
  let vmul_31_ptr : RegionName := "vmul_31"
  let vmul_32_ptr : RegionName := "vmul_32"
  let vneg_1_ptr : RegionName := "vneg_1"
  let vmul_33_ptr : RegionName := "vmul_33"
  let vadd_9_ptr : RegionName := "vadd_9"
  let vmul_34_ptr : RegionName := "vmul_34"
  let vadd_10_ptr : RegionName := "vadd_10"
  let vslice_backward_ptr : RegionName := "vslice_backward"
  let vslice_backward_1_ptr : RegionName := "vslice_backward_1"
  let vadd_11_ptr : RegionName := "vadd_11"
  let vslice_backward_2_ptr : RegionName := "vslice_backward_2"
  let vslice_backward_3_ptr : RegionName := "vslice_backward_3"
  let vadd_12_ptr : RegionName := "vadd_12"
  let vcat_6_ptr : RegionName := "vcat_6"
  let vmm_6_ptr : RegionName := "vmm_6"
  let vto_copy_28_ptr : RegionName := "vto_copy_28"
  let vmul_35_ptr : RegionName := "vmul_35"
  let vmul_36_ptr : RegionName := "vmul_36"
  let vmul_37_ptr : RegionName := "vmul_37"
  let vsum_2_ptr : RegionName := "vsum_2"
  let vpow_5_ptr : RegionName := "vpow_5"
  let vmul_38_ptr : RegionName := "vmul_38"
  let vmul_39_ptr : RegionName := "vmul_39"
  let vdiv_2_ptr : RegionName := "vdiv_2"
  let vpow_6_ptr : RegionName := "vpow_6"
  let vmul_40_ptr : RegionName := "vmul_40"
  let vmul_41_ptr : RegionName := "vmul_41"
  let vadd_13_ptr : RegionName := "vadd_13"
  let vto_copy_29_ptr : RegionName := "vto_copy_29"
  let vmm_7_ptr : RegionName := "vmm_7"
  let vto_copy_30_ptr : RegionName := "vto_copy_30"
  let vmul_42_ptr : RegionName := "vmul_42"
  let vmul_43_ptr : RegionName := "vmul_43"
  let vmul_44_ptr : RegionName := "vmul_44"
  let vsum_3_ptr : RegionName := "vsum_3"
  let vpow_7_ptr : RegionName := "vpow_7"
  let vdiv_3_ptr : RegionName := "vdiv_3"
  let vpow_8_ptr : RegionName := "vpow_8"
  let vadd_14_ptr : RegionName := "vadd_14"
  let vto_copy_31_ptr : RegionName := "vto_copy_31"
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
  let candidate_vto_copy_20_ptr : RegionName := "candidate_vto_copy_20"
  let candidate_vpow_3_ptr : RegionName := "candidate_vpow_3"
  let candidate_vmean_ptr : RegionName := "candidate_vmean"
  let candidate_vadd_2_ptr : RegionName := "candidate_vadd_2"
  let candidate_vrsqrt_ptr : RegionName := "candidate_vrsqrt"
  let candidate_vmul_13_ptr : RegionName := "candidate_vmul_13"
  let candidate_vmul_14_ptr : RegionName := "candidate_vmul_14"
  let candidate_vto_copy_21_ptr : RegionName := "candidate_vto_copy_21"
  let candidate_vmm_1_ptr : RegionName := "candidate_vmm_1"
  let candidate_vmm_2_ptr : RegionName := "candidate_vmm_2"
  let candidate_vto_copy_22_ptr : RegionName := "candidate_vto_copy_22"
  let candidate_vpow_4_ptr : RegionName := "candidate_vpow_4"
  let candidate_vmean_1_ptr : RegionName := "candidate_vmean_1"
  let candidate_vadd_3_ptr : RegionName := "candidate_vadd_3"
  let candidate_vrsqrt_1_ptr : RegionName := "candidate_vrsqrt_1"
  let candidate_vmul_15_ptr : RegionName := "candidate_vmul_15"
  let candidate_vmul_16_ptr : RegionName := "candidate_vmul_16"
  let candidate_vto_copy_23_ptr : RegionName := "candidate_vto_copy_23"
  let candidate_vdiv_2_ptr : RegionName := "candidate_vdiv_2"
  let candidate_vpow_5_ptr : RegionName := "candidate_vpow_5"
  let candidate_vreciprocal_2_ptr : RegionName := "candidate_vreciprocal_2"
  let candidate_vmul_17_ptr : RegionName := "candidate_vmul_17"
  let candidate_vmul_18_ptr : RegionName := "candidate_vmul_18"
  let candidate_vcos_2_ptr : RegionName := "candidate_vcos_2"
  let candidate_vto_copy_24_ptr : RegionName := "candidate_vto_copy_24"
  let candidate_vsin_2_ptr : RegionName := "candidate_vsin_2"
  let candidate_vto_copy_25_ptr : RegionName := "candidate_vto_copy_25"
  let candidate_vneg_ptr : RegionName := "candidate_vneg"
  let candidate_vstack_ptr : RegionName := "candidate_vstack"
  let candidate_vmul_19_ptr : RegionName := "candidate_vmul_19"
  let candidate_vmul_20_ptr : RegionName := "candidate_vmul_20"
  let candidate_vadd_4_ptr : RegionName := "candidate_vadd_4"
  let candidate_vcos_3_ptr : RegionName := "candidate_vcos_3"
  let candidate_vto_copy_26_ptr : RegionName := "candidate_vto_copy_26"
  let candidate_vsin_3_ptr : RegionName := "candidate_vsin_3"
  let candidate_vto_copy_27_ptr : RegionName := "candidate_vto_copy_27"
  let candidate_vneg_1_ptr : RegionName := "candidate_vneg_1"
  let candidate_vstack_1_ptr : RegionName := "candidate_vstack_1"
  let candidate_vmul_21_ptr : RegionName := "candidate_vmul_21"
  let candidate_vmul_22_ptr : RegionName := "candidate_vmul_22"
  let candidate_vadd_5_ptr : RegionName := "candidate_vadd_5"
  let candidate_vbmm_2_ptr : RegionName := "candidate_vbmm_2"
  let candidate_vbmm_3_ptr : RegionName := "candidate_vbmm_3"
  let candidate_vcat_2_ptr : RegionName := "candidate_vcat_2"
  let candidate_vcat_3_ptr : RegionName := "candidate_vcat_3"
  let candidate_vto_copy_28_ptr : RegionName := "candidate_vto_copy_28"
  let candidate_vto_copy_29_ptr : RegionName := "candidate_vto_copy_29"
  let candidate_vbmm_4_ptr : RegionName := "candidate_vbmm_4"
  let candidate_vmul_23_ptr : RegionName := "candidate_vmul_23"
  let candidate_vadd_6_ptr : RegionName := "candidate_vadd_6"
  let candidate_vsub_ptr : RegionName := "candidate_vsub"
  let candidate_vgt_ptr : RegionName := "candidate_vgt"
  let candidate_vmasked_fill_2_ptr : RegionName := "candidate_vmasked_fill_2"
  let candidate_vsoftmax_ptr : RegionName := "candidate_vsoftmax"
  let candidate_vto_copy_30_ptr : RegionName := "candidate_vto_copy_30"
  let candidate_vto_copy_31_ptr : RegionName := "candidate_vto_copy_31"
  let candidate_vto_copy_32_ptr : RegionName := "candidate_vto_copy_32"
  let candidate_vbmm_6_ptr : RegionName := "candidate_vbmm_6"
  let candidate_vmul_24_ptr : RegionName := "candidate_vmul_24"
  let candidate_vadd_7_ptr : RegionName := "candidate_vadd_7"
  let candidate_vsub_1_ptr : RegionName := "candidate_vsub_1"
  let candidate_vgt_1_ptr : RegionName := "candidate_vgt_1"
  let candidate_vmasked_fill_3_ptr : RegionName := "candidate_vmasked_fill_3"
  let candidate_vsoftmax_1_ptr : RegionName := "candidate_vsoftmax_1"
  let candidate_vto_copy_33_ptr : RegionName := "candidate_vto_copy_33"
  let candidate_vmm_4_ptr : RegionName := "candidate_vmm_4"
  let candidate_vbmm_8_ptr : RegionName := "candidate_vbmm_8"
  let candidate_vbmm_9_ptr : RegionName := "candidate_vbmm_9"
  let candidate_vto_copy_34_ptr : RegionName := "candidate_vto_copy_34"
  let candidate_vsoftmax_backward_data_ptr : RegionName := "candidate_vsoftmax_backward_data"
  let candidate_vmasked_fill_4_ptr : RegionName := "candidate_vmasked_fill_4"
  let candidate_vmul_25_ptr : RegionName := "candidate_vmul_25"
  let candidate_vbmm_10_ptr : RegionName := "candidate_vbmm_10"
  let candidate_vbmm_11_ptr : RegionName := "candidate_vbmm_11"
  let candidate_vslice_backward_ptr : RegionName := "candidate_vslice_backward"
  let candidate_vto_copy_35_ptr : RegionName := "candidate_vto_copy_35"
  let candidate_vslice_backward_1_ptr : RegionName := "candidate_vslice_backward_1"
  let candidate_vto_copy_36_ptr : RegionName := "candidate_vto_copy_36"
  let candidate_vslice_backward_2_ptr : RegionName := "candidate_vslice_backward_2"
  let candidate_vbmm_12_ptr : RegionName := "candidate_vbmm_12"
  let candidate_vbmm_13_ptr : RegionName := "candidate_vbmm_13"
  let candidate_vto_copy_37_ptr : RegionName := "candidate_vto_copy_37"
  let candidate_vsoftmax_backward_data_1_ptr : RegionName := "candidate_vsoftmax_backward_data_1"
  let candidate_vmasked_fill_5_ptr : RegionName := "candidate_vmasked_fill_5"
  let candidate_vmul_26_ptr : RegionName := "candidate_vmul_26"
  let candidate_vbmm_14_ptr : RegionName := "candidate_vbmm_14"
  let candidate_vbmm_15_ptr : RegionName := "candidate_vbmm_15"
  let candidate_vslice_backward_3_ptr : RegionName := "candidate_vslice_backward_3"
  let candidate_vadd_8_ptr : RegionName := "candidate_vadd_8"
  let candidate_vto_copy_38_ptr : RegionName := "candidate_vto_copy_38"
  let candidate_vslice_backward_4_ptr : RegionName := "candidate_vslice_backward_4"
  let candidate_vadd_9_ptr : RegionName := "candidate_vadd_9"
  let candidate_vto_copy_39_ptr : RegionName := "candidate_vto_copy_39"
  let candidate_vslice_backward_5_ptr : RegionName := "candidate_vslice_backward_5"
  let candidate_vadd_10_ptr : RegionName := "candidate_vadd_10"
  let candidate_vsum_1_ptr : RegionName := "candidate_vsum_1"
  let candidate_vbmm_16_ptr : RegionName := "candidate_vbmm_16"
  let candidate_vbmm_17_ptr : RegionName := "candidate_vbmm_17"
  let candidate_vadd_11_ptr : RegionName := "candidate_vadd_11"
  let candidate_vselect_backward_ptr : RegionName := "candidate_vselect_backward"
  let candidate_vmul_27_ptr : RegionName := "candidate_vmul_27"
  let candidate_vmul_28_ptr : RegionName := "candidate_vmul_28"
  let candidate_vselect_backward_1_ptr : RegionName := "candidate_vselect_backward_1"
  let candidate_vneg_2_ptr : RegionName := "candidate_vneg_2"
  let candidate_vselect_backward_2_ptr : RegionName := "candidate_vselect_backward_2"
  let candidate_vadd_12_ptr : RegionName := "candidate_vadd_12"
  let candidate_vadd_13_ptr : RegionName := "candidate_vadd_13"
  let candidate_vmul_29_ptr : RegionName := "candidate_vmul_29"
  let candidate_vmul_30_ptr : RegionName := "candidate_vmul_30"
  let candidate_vselect_backward_3_ptr : RegionName := "candidate_vselect_backward_3"
  let candidate_vneg_3_ptr : RegionName := "candidate_vneg_3"
  let candidate_vselect_backward_4_ptr : RegionName := "candidate_vselect_backward_4"
  let candidate_vadd_14_ptr : RegionName := "candidate_vadd_14"
  let candidate_vadd_15_ptr : RegionName := "candidate_vadd_15"
  let candidate_vcat_5_ptr : RegionName := "candidate_vcat_5"
  let candidate_vto_copy_40_ptr : RegionName := "candidate_vto_copy_40"
  let candidate_vmul_31_ptr : RegionName := "candidate_vmul_31"
  let candidate_vmul_32_ptr : RegionName := "candidate_vmul_32"
  let candidate_vmul_33_ptr : RegionName := "candidate_vmul_33"
  let candidate_vsum_2_ptr : RegionName := "candidate_vsum_2"
  let candidate_vpow_6_ptr : RegionName := "candidate_vpow_6"
  let candidate_vmul_34_ptr : RegionName := "candidate_vmul_34"
  let candidate_vmul_35_ptr : RegionName := "candidate_vmul_35"
  let candidate_vdiv_3_ptr : RegionName := "candidate_vdiv_3"
  let candidate_vpow_7_ptr : RegionName := "candidate_vpow_7"
  let candidate_vmul_36_ptr : RegionName := "candidate_vmul_36"
  let candidate_vmul_37_ptr : RegionName := "candidate_vmul_37"
  let candidate_vadd_16_ptr : RegionName := "candidate_vadd_16"
  let candidate_vto_copy_41_ptr : RegionName := "candidate_vto_copy_41"
  let candidate_vcat_6_ptr : RegionName := "candidate_vcat_6"
  let candidate_vmm_5_ptr : RegionName := "candidate_vmm_5"
  let candidate_vmm_6_ptr : RegionName := "candidate_vmm_6"
  let candidate_vto_copy_42_ptr : RegionName := "candidate_vto_copy_42"
  let candidate_vmul_38_ptr : RegionName := "candidate_vmul_38"
  let candidate_vmul_39_ptr : RegionName := "candidate_vmul_39"
  let candidate_vmul_40_ptr : RegionName := "candidate_vmul_40"
  let candidate_vsum_3_ptr : RegionName := "candidate_vsum_3"
  let candidate_vpow_8_ptr : RegionName := "candidate_vpow_8"
  let candidate_vmul_41_ptr : RegionName := "candidate_vmul_41"
  let candidate_vmul_42_ptr : RegionName := "candidate_vmul_42"
  let candidate_vdiv_4_ptr : RegionName := "candidate_vdiv_4"
  let candidate_vpow_9_ptr : RegionName := "candidate_vpow_9"
  let candidate_vmul_43_ptr : RegionName := "candidate_vmul_43"
  let candidate_vmul_44_ptr : RegionName := "candidate_vmul_44"
  let candidate_vadd_17_ptr : RegionName := "candidate_vadd_17"
  let candidate_vto_copy_43_ptr : RegionName := "candidate_vto_copy_43"
  let candidate_vmm_7_ptr : RegionName := "candidate_vmm_7"
  let candidate_vadd_18_ptr : RegionName := "candidate_vadd_18"
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
    tl.store($(candidate_vto_copy_20_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(65536))
    r18_value = ((tl.load($(candidate_vto_copy_20_ptr) + ((((r18_z / $(32768)) % $(2)) * $(32768)) + (((r18_z / $(256)) % $(128)) * $(256)) + (r18_z % $(256))), dtype=tl.float32)) * (tl.load($(candidate_vto_copy_20_ptr) + ((((r18_z / $(32768)) % $(2)) * $(32768)) + (((r18_z / $(256)) % $(128)) * $(256)) + (r18_z % $(256))), dtype=tl.float32)))
    r18_value = r18_value.to(tl.float32)
    r18_value = tl.broadcast_to(r18_value, [$(65536)])
    tl.store($(candidate_vpow_3_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(256))
    r19_lane = tl.arange($(256))
    r19_value = (tl.sum(tl.load($(candidate_vpow_3_ptr) + ((((r19_z[:, None] / $(128)) % $(2)) * $(32768)) + ((r19_z[:, None] % $(128)) * $(256)) + r19_lane[None, :]), dtype=tl.float32), axis=1) / 256.0)
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
    r22_value = (tl.load($(candidate_vto_copy_20_ptr) + ((((r22_z / $(32768)) % $(2)) * $(32768)) + (((r22_z / $(256)) % $(128)) * $(256)) + (r22_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_ptr) + ((((r22_z / $(32768)) % $(2)) * $(128)) + ((r22_z / $(256)) % $(128))), dtype=tl.float32))
    r22_value = r22_value.to(tl.float32)
    r22_value = tl.broadcast_to(r22_value, [$(65536)])
    tl.store($(candidate_vmul_13_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(65536))
    r23_value = (tl.cast(tl.load($(candidate_vto_copy_1_ptr) + (r23_z % $(256)), dtype=tl.bfloat16), tl.float32) * tl.load($(candidate_vmul_13_ptr) + ((((r23_z / $(32768)) % $(2)) * $(32768)) + (((r23_z / $(256)) % $(128)) * $(256)) + (r23_z % $(256))), dtype=tl.float32))
    r23_value = r23_value.to(tl.float32)
    r23_value = tl.broadcast_to(r23_value, [$(65536)])
    tl.store($(candidate_vmul_14_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(65536))
    r24_value = tl.load($(candidate_vmul_14_ptr) + ((((r24_z / $(32768)) % $(2)) * $(32768)) + (((r24_z / $(256)) % $(128)) * $(256)) + (r24_z % $(256))), dtype=tl.float32)
    r24_value = r24_value.to(tl.bfloat16)
    r24_value = tl.broadcast_to(r24_value, [$(65536)])
    tl.store($(candidate_vto_copy_21_ptr) + r24_z, r24_value)
    r25_row = tl.arange($(256))
    r25_col = tl.arange($(1536))
    r25_lane = tl.arange($(256))
    r25_left = tl.load($(candidate_vto_copy_21_ptr) + ((((((r25_row[:, None] * $(256)) + r25_lane[None, :]) / $(32768)) % $(2)) * $(32768)) + (((((r25_row[:, None] * $(256)) + r25_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r25_row[:, None] * $(256)) + r25_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r25_right = tl.load($(candidate_vto_copy_2_ptr) + (((r25_col[None, :] % $(1536)) * $(256)) + (r25_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r25_value = tl.dot(r25_left, r25_right)
    r25_value = r25_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_1_ptr) + ((r25_row[:, None] * $(1536)) + r25_col[None, :]), r25_value)
    r26_row = tl.arange($(256))
    r26_col = tl.arange($(320))
    r26_lane = tl.arange($(512))
    r26_left = tl.load($(input0_ptr) + ((((((r26_row[:, None] * $(512)) + r26_lane[None, :]) / $(65536)) % $(2)) * $(65536)) + (((((r26_row[:, None] * $(512)) + r26_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r26_row[:, None] * $(512)) + r26_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r26_right = tl.load($(candidate_vto_copy_3_ptr) + (((r26_col[None, :] % $(320)) * $(512)) + (r26_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r26_value = tl.dot(r26_left, r26_right)
    r26_value = r26_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_2_ptr) + ((r26_row[:, None] * $(320)) + r26_col[None, :]), r26_value)
    r27_z = tl.arange($(65536))
    r27_value = tl.load($(candidate_vmm_2_ptr) + ((((r27_z / $(32768)) % $(2)) * $(40960)) + (((r27_z / $(256)) % $(128)) * $(320)) + (r27_z % $(256))), dtype=tl.bfloat16)
    r27_value = r27_value.to(tl.float32)
    r27_value = tl.broadcast_to(r27_value, [$(65536)])
    tl.store($(candidate_vto_copy_22_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(65536))
    r28_value = ((tl.load($(candidate_vto_copy_22_ptr) + ((((r28_z / $(32768)) % $(2)) * $(32768)) + (((r28_z / $(256)) % $(128)) * $(256)) + (r28_z % $(256))), dtype=tl.float32)) * (tl.load($(candidate_vto_copy_22_ptr) + ((((r28_z / $(32768)) % $(2)) * $(32768)) + (((r28_z / $(256)) % $(128)) * $(256)) + (r28_z % $(256))), dtype=tl.float32)))
    r28_value = r28_value.to(tl.float32)
    r28_value = tl.broadcast_to(r28_value, [$(65536)])
    tl.store($(candidate_vpow_4_ptr) + r28_z, r28_value)
    r29_z = tl.arange($(256))
    r29_lane = tl.arange($(256))
    r29_value = (tl.sum(tl.load($(candidate_vpow_4_ptr) + ((((r29_z[:, None] / $(128)) % $(2)) * $(32768)) + ((r29_z[:, None] % $(128)) * $(256)) + r29_lane[None, :]), dtype=tl.float32), axis=1) / 256.0)
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
    r32_z = tl.arange($(65536))
    r32_value = (tl.load($(candidate_vto_copy_22_ptr) + ((((r32_z / $(32768)) % $(2)) * $(32768)) + (((r32_z / $(256)) % $(128)) * $(256)) + (r32_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_1_ptr) + ((((r32_z / $(32768)) % $(2)) * $(128)) + ((r32_z / $(256)) % $(128))), dtype=tl.float32))
    r32_value = r32_value.to(tl.float32)
    r32_value = tl.broadcast_to(r32_value, [$(65536)])
    tl.store($(candidate_vmul_15_ptr) + r32_z, r32_value)
    r33_z = tl.arange($(65536))
    r33_value = (tl.cast(tl.load($(candidate_vto_copy_4_ptr) + (r33_z % $(256)), dtype=tl.bfloat16), tl.float32) * tl.load($(candidate_vmul_15_ptr) + ((((r33_z / $(32768)) % $(2)) * $(32768)) + (((r33_z / $(256)) % $(128)) * $(256)) + (r33_z % $(256))), dtype=tl.float32))
    r33_value = r33_value.to(tl.float32)
    r33_value = tl.broadcast_to(r33_value, [$(65536)])
    tl.store($(candidate_vmul_16_ptr) + r33_z, r33_value)
    r34_z = tl.arange($(65536))
    r34_value = tl.load($(candidate_vmul_16_ptr) + ((((r34_z / $(32768)) % $(2)) * $(32768)) + (((r34_z / $(256)) % $(128)) * $(256)) + (r34_z % $(256))), dtype=tl.float32)
    r34_value = r34_value.to(tl.bfloat16)
    r34_value = tl.broadcast_to(r34_value, [$(65536)])
    tl.store($(candidate_vto_copy_23_ptr) + r34_z, r34_value)
    r35_z = tl.arange($(32))
    r35_value = ((tl.cast(tl.toReal(r35_z), tl.float32) * 2.0) / 64.0)
    r35_value = r35_value.to(tl.float32)
    r35_value = tl.broadcast_to(r35_value, [$(32)])
    tl.store($(candidate_vdiv_2_ptr) + r35_z, r35_value)
    r36_z = tl.arange($(32))
    r36_value = tl.extra.cuda.libdevice.pow(10000.0, tl.load($(candidate_vdiv_2_ptr) + r36_z, dtype=tl.float32))
    r36_value = r36_value.to(tl.float32)
    r36_value = tl.broadcast_to(r36_value, [$(32)])
    tl.store($(candidate_vpow_5_ptr) + r36_z, r36_value)
    r37_z = tl.arange($(32))
    r37_value = 1.0 / tl.load($(candidate_vpow_5_ptr) + r37_z, dtype=tl.float32)
    r37_value = r37_value.to(tl.float32)
    r37_value = tl.broadcast_to(r37_value, [$(32)])
    tl.store($(candidate_vreciprocal_2_ptr) + r37_z, r37_value)
    r38_z = tl.arange($(32))
    r38_value = (tl.load($(candidate_vreciprocal_2_ptr) + r38_z, dtype=tl.float32) * 1.0)
    r38_value = r38_value.to(tl.float32)
    r38_value = tl.broadcast_to(r38_value, [$(32)])
    tl.store($(candidate_vmul_17_ptr) + r38_z, r38_value)
    r39_z = tl.arange($(4096))
    r39_value = (tl.cast(tl.toReal(((r39_z / $(32)) % $(128))), tl.float32) * tl.load($(candidate_vmul_17_ptr) + (r39_z % $(32)), dtype=tl.float32))
    r39_value = r39_value.to(tl.float32)
    r39_value = tl.broadcast_to(r39_value, [$(4096)])
    tl.store($(candidate_vmul_18_ptr) + r39_z, r39_value)
    r40_z = tl.arange($(8192))
    r40_value = tl.cos(tl.load($(candidate_vmul_18_ptr) + ((((r40_z / $(64)) % $(128)) * $(32)) + ((((((r40_z / $(64)) % $(128)) * $(64)) + (r40_z % $(64))) / $(2)) % $(32))), dtype=tl.float32))
    r40_value = r40_value.to(tl.float32)
    r40_value = tl.broadcast_to(r40_value, [$(8192)])
    tl.store($(candidate_vcos_2_ptr) + r40_z, r40_value)
    r41_z = tl.arange($(8192))
    r41_value = tl.load($(candidate_vcos_2_ptr) + ((((r41_z / $(64)) % $(128)) * $(64)) + (r41_z % $(64))), dtype=tl.float32)
    r41_value = r41_value.to(tl.bfloat16)
    r41_value = tl.broadcast_to(r41_value, [$(8192)])
    tl.store($(candidate_vto_copy_24_ptr) + r41_z, r41_value)
    r42_z = tl.arange($(8192))
    r42_value = tl.sin(tl.load($(candidate_vmul_18_ptr) + ((((r42_z / $(64)) % $(128)) * $(32)) + ((((((r42_z / $(64)) % $(128)) * $(64)) + (r42_z % $(64))) / $(2)) % $(32))), dtype=tl.float32))
    r42_value = r42_value.to(tl.float32)
    r42_value = tl.broadcast_to(r42_value, [$(8192)])
    tl.store($(candidate_vsin_2_ptr) + r42_z, r42_value)
    r43_z = tl.arange($(8192))
    r43_value = tl.load($(candidate_vsin_2_ptr) + ((((r43_z / $(64)) % $(128)) * $(64)) + (r43_z % $(64))), dtype=tl.float32)
    r43_value = r43_value.to(tl.bfloat16)
    r43_value = tl.broadcast_to(r43_value, [$(8192)])
    tl.store($(candidate_vto_copy_25_ptr) + r43_z, r43_value)
    r44_z = tl.arange($(65536))
    r44_value = -tl.load($(candidate_vmm_1_ptr) + ((((r44_z / $(256)) % $(2)) * $(196608)) + (((r44_z / $(512)) % $(128)) * $(1536)) + (((r44_z / $(32)) % $(8)) * $(192)) + ((r44_z % $(32)) * $(2)) + $(129)), dtype=tl.bfloat16)
    r44_value = r44_value.to(tl.bfloat16)
    r44_value = tl.broadcast_to(r44_value, [$(65536)])
    tl.store($(candidate_vneg_ptr) + r44_z, r44_value)
    r45_z = tl.arange($(131072))
    r45_value = tl.where((r45_z % $(2)) < $(1), tl.load($(candidate_vneg_ptr) + ((((r45_z / $(1024)) % $(128)) * $(512)) + (((r45_z / $(512)) % $(2)) * $(256)) + (((r45_z / $(64)) % $(8)) * $(32)) + ((r45_z / $(2)) % $(32))), dtype=tl.bfloat16), tl.load($(candidate_vmm_1_ptr) + ((((r45_z / $(512)) % $(2)) * $(196608)) + (((r45_z / $(1024)) % $(128)) * $(1536)) + (((r45_z / $(64)) % $(8)) * $(192)) + (((r45_z / $(2)) % $(32)) * $(2)) + $(128)), dtype=tl.bfloat16))
    r45_value = r45_value.to(tl.bfloat16)
    r45_value = tl.broadcast_to(r45_value, [$(131072)])
    tl.store($(candidate_vstack_ptr) + r45_z, r45_value)
    r46_z = tl.arange($(131072))
    r46_value = (tl.load($(candidate_vmm_1_ptr) + ((((r46_z / $(512)) % $(2)) * $(196608)) + (((r46_z / $(1024)) % $(128)) * $(1536)) + (((r46_z / $(64)) % $(8)) * $(192)) + (r46_z % $(64)) + $(128)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_24_ptr) + ((((r46_z / $(1024)) % $(128)) * $(64)) + (r46_z % $(64))), dtype=tl.bfloat16))
    r46_value = r46_value.to(tl.bfloat16)
    r46_value = tl.broadcast_to(r46_value, [$(131072)])
    tl.store($(candidate_vmul_19_ptr) + r46_z, r46_value)
    r47_z = tl.arange($(131072))
    r47_value = (tl.load($(candidate_vstack_ptr) + ((((r47_z / $(1024)) % $(128)) * $(1024)) + (((r47_z / $(512)) % $(2)) * $(512)) + (((r47_z / $(64)) % $(8)) * $(64)) + (((((((r47_z / $(1024)) % $(128)) * $(1024)) + (((r47_z / $(512)) % $(2)) * $(512)) + (((r47_z / $(64)) % $(8)) * $(64)) + (r47_z % $(64))) / $(2)) % $(32)) * $(2)) + (((((r47_z / $(1024)) % $(128)) * $(1024)) + (((r47_z / $(512)) % $(2)) * $(512)) + (((r47_z / $(64)) % $(8)) * $(64)) + (r47_z % $(64))) % $(2))), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_25_ptr) + ((((r47_z / $(1024)) % $(128)) * $(64)) + (r47_z % $(64))), dtype=tl.bfloat16))
    r47_value = r47_value.to(tl.bfloat16)
    r47_value = tl.broadcast_to(r47_value, [$(131072)])
    tl.store($(candidate_vmul_20_ptr) + r47_z, r47_value)
    r48_z = tl.arange($(131072))
    r48_value = (tl.load($(candidate_vmul_19_ptr) + ((((r48_z / $(1024)) % $(128)) * $(1024)) + (((r48_z / $(512)) % $(2)) * $(512)) + (((r48_z / $(64)) % $(8)) * $(64)) + (r48_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_20_ptr) + ((((r48_z / $(1024)) % $(128)) * $(1024)) + (((r48_z / $(512)) % $(2)) * $(512)) + (((r48_z / $(64)) % $(8)) * $(64)) + (r48_z % $(64))), dtype=tl.bfloat16))
    r48_value = r48_value.to(tl.bfloat16)
    r48_value = tl.broadcast_to(r48_value, [$(131072)])
    tl.store($(candidate_vadd_4_ptr) + r48_z, r48_value)
    r49_z = tl.arange($(8192))
    r49_value = tl.cos(tl.load($(candidate_vmul_18_ptr) + ((((r49_z / $(64)) % $(128)) * $(32)) + ((((((r49_z / $(64)) % $(128)) * $(64)) + (r49_z % $(64))) / $(2)) % $(32))), dtype=tl.float32))
    r49_value = r49_value.to(tl.float32)
    r49_value = tl.broadcast_to(r49_value, [$(8192)])
    tl.store($(candidate_vcos_3_ptr) + r49_z, r49_value)
    r50_z = tl.arange($(8192))
    r50_value = tl.load($(candidate_vcos_3_ptr) + ((((r50_z / $(64)) % $(128)) * $(64)) + (r50_z % $(64))), dtype=tl.float32)
    r50_value = r50_value.to(tl.bfloat16)
    r50_value = tl.broadcast_to(r50_value, [$(8192)])
    tl.store($(candidate_vto_copy_26_ptr) + r50_z, r50_value)
    r51_z = tl.arange($(8192))
    r51_value = tl.sin(tl.load($(candidate_vmul_18_ptr) + ((((r51_z / $(64)) % $(128)) * $(32)) + ((((((r51_z / $(64)) % $(128)) * $(64)) + (r51_z % $(64))) / $(2)) % $(32))), dtype=tl.float32))
    r51_value = r51_value.to(tl.float32)
    r51_value = tl.broadcast_to(r51_value, [$(8192)])
    tl.store($(candidate_vsin_3_ptr) + r51_z, r51_value)
    r52_z = tl.arange($(8192))
    r52_value = tl.load($(candidate_vsin_3_ptr) + ((((r52_z / $(64)) % $(128)) * $(64)) + (r52_z % $(64))), dtype=tl.float32)
    r52_value = r52_value.to(tl.bfloat16)
    r52_value = tl.broadcast_to(r52_value, [$(8192)])
    tl.store($(candidate_vto_copy_27_ptr) + r52_z, r52_value)
    r53_z = tl.arange($(8192))
    r53_value = -tl.load($(candidate_vmm_2_ptr) + ((((r53_z / $(32)) % $(2)) * $(40960)) + (((r53_z / $(64)) % $(128)) * $(320)) + ((r53_z % $(32)) * $(2)) + $(257)), dtype=tl.bfloat16)
    r53_value = r53_value.to(tl.bfloat16)
    r53_value = tl.broadcast_to(r53_value, [$(8192)])
    tl.store($(candidate_vneg_1_ptr) + r53_z, r53_value)
    r54_z = tl.arange($(16384))
    r54_value = tl.where((r54_z % $(2)) < $(1), tl.load($(candidate_vneg_1_ptr) + ((((r54_z / $(128)) % $(128)) * $(64)) + (((r54_z / $(64)) % $(2)) * $(32)) + ((r54_z / $(2)) % $(32))), dtype=tl.bfloat16), tl.load($(candidate_vmm_2_ptr) + ((((r54_z / $(64)) % $(2)) * $(40960)) + (((r54_z / $(128)) % $(128)) * $(320)) + (((r54_z / $(2)) % $(32)) * $(2)) + $(256)), dtype=tl.bfloat16))
    r54_value = r54_value.to(tl.bfloat16)
    r54_value = tl.broadcast_to(r54_value, [$(16384)])
    tl.store($(candidate_vstack_1_ptr) + r54_z, r54_value)
    r55_z = tl.arange($(16384))
    r55_value = (tl.load($(candidate_vmm_2_ptr) + ((((r55_z / $(64)) % $(2)) * $(40960)) + (((r55_z / $(128)) % $(128)) * $(320)) + (r55_z % $(64)) + $(256)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_26_ptr) + ((((r55_z / $(128)) % $(128)) * $(64)) + (r55_z % $(64))), dtype=tl.bfloat16))
    r55_value = r55_value.to(tl.bfloat16)
    r55_value = tl.broadcast_to(r55_value, [$(16384)])
    tl.store($(candidate_vmul_21_ptr) + r55_z, r55_value)
    r56_z = tl.arange($(16384))
    r56_value = (tl.load($(candidate_vstack_1_ptr) + ((((r56_z / $(128)) % $(128)) * $(128)) + (((r56_z / $(64)) % $(2)) * $(64)) + (((((((r56_z / $(128)) % $(128)) * $(128)) + (((r56_z / $(64)) % $(2)) * $(64)) + (r56_z % $(64))) / $(2)) % $(32)) * $(2)) + (((((r56_z / $(128)) % $(128)) * $(128)) + (((r56_z / $(64)) % $(2)) * $(64)) + (r56_z % $(64))) % $(2))), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_27_ptr) + ((((r56_z / $(128)) % $(128)) * $(64)) + (r56_z % $(64))), dtype=tl.bfloat16))
    r56_value = r56_value.to(tl.bfloat16)
    r56_value = tl.broadcast_to(r56_value, [$(16384)])
    tl.store($(candidate_vmul_22_ptr) + r56_z, r56_value)
    r57_z = tl.arange($(16384))
    r57_value = (tl.load($(candidate_vmul_21_ptr) + ((((r57_z / $(128)) % $(128)) * $(128)) + (((r57_z / $(64)) % $(2)) * $(64)) + (r57_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_22_ptr) + ((((r57_z / $(128)) % $(128)) * $(128)) + (((r57_z / $(64)) % $(2)) * $(64)) + (r57_z % $(64))), dtype=tl.bfloat16))
    r57_value = r57_value.to(tl.bfloat16)
    r57_value = tl.broadcast_to(r57_value, [$(16384)])
    tl.store($(candidate_vadd_5_ptr) + r57_z, r57_value)
    r58_row = tl.arange($(256))
    r58_col = tl.arange($(1024))
    r58_lane = tl.arange($(256))
    for r58_group in range($(1)) {
      r58_left = tl.load($(candidate_vto_copy_23_ptr) + ((((((((((r58_row[:, None] * $(256)) + r58_lane[None, :]) / $(256)) % $(256)) * $(256)) + (((r58_row[:, None] * $(256)) + r58_lane[None, :]) % $(256))) / $(32768)) % $(2)) * $(32768)) + ((((((r58_row[:, None] * $(256)) + r58_lane[None, :]) / $(256)) % $(256)) % $(128)) * $(256)) + (((r58_row[:, None] * $(256)) + r58_lane[None, :]) % $(256))), dtype=tl.bfloat16)
      r58_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r58_lane[:, None] * $(1024)) + r58_col[None, :]) / $(128)) % $(8)) * $(65536)) + ((((r58_lane[:, None] * $(1024)) + r58_col[None, :]) % $(128)) * $(256)) + ((((r58_lane[:, None] * $(1024)) + r58_col[None, :]) / $(1024)) % $(256))), dtype=tl.bfloat16)
      r58_value = tl.dot(r58_left, r58_right)
      r58_value = r58_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_2_ptr) + ((r58_group * $(262144)) + (r58_row[:, None] * $(1024)) + r58_col[None, :]), r58_value)
    }
    r59_row = tl.arange($(256))
    r59_col = tl.arange($(1024))
    r59_lane = tl.arange($(256))
    for r59_group in range($(1)) {
      r59_left = tl.load($(candidate_vto_copy_23_ptr) + ((((((((((r59_row[:, None] * $(256)) + r59_lane[None, :]) / $(256)) % $(256)) * $(256)) + (((r59_row[:, None] * $(256)) + r59_lane[None, :]) % $(256))) / $(32768)) % $(2)) * $(32768)) + ((((((r59_row[:, None] * $(256)) + r59_lane[None, :]) / $(256)) % $(256)) % $(128)) * $(256)) + (((r59_row[:, None] * $(256)) + r59_lane[None, :]) % $(256))), dtype=tl.bfloat16)
      r59_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r59_lane[:, None] * $(1024)) + r59_col[None, :]) / $(128)) % $(8)) * $(65536)) + ((((r59_lane[:, None] * $(1024)) + r59_col[None, :]) % $(128)) * $(256)) + ((((r59_lane[:, None] * $(1024)) + r59_col[None, :]) / $(1024)) % $(256)) + $(32768)), dtype=tl.bfloat16)
      r59_value = tl.dot(r59_left, r59_right)
      r59_value = r59_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_3_ptr) + ((r59_group * $(262144)) + (r59_row[:, None] * $(1024)) + r59_col[None, :]), r59_value)
    }
    r60_z = tl.arange($(393216))
    r60_value = tl.where((r60_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r60_z / $(1536)) % $(256)) * $(1024)) + (((r60_z / $(192)) % $(8)) * $(128)) + ((r60_z % $(192)) - $(0))) / $(131072)) % $(2)) * $(196608)) + (((((((r60_z / $(1536)) % $(256)) * $(1024)) + (((r60_z / $(192)) % $(8)) * $(128)) + ((r60_z % $(192)) - $(0))) / $(1024)) % $(128)) * $(1536)) + (((((((r60_z / $(1536)) % $(256)) * $(1024)) + (((r60_z / $(192)) % $(8)) * $(128)) + ((r60_z % $(192)) - $(0))) / $(128)) % $(8)) * $(192)) + (((((r60_z / $(1536)) % $(256)) * $(1024)) + (((r60_z / $(192)) % $(8)) * $(128)) + ((r60_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vadd_4_ptr) + ((((((((r60_z / $(1536)) % $(256)) * $(512)) + (((r60_z / $(192)) % $(8)) * $(64)) + ((r60_z % $(192)) - $(128))) / $(512)) % $(128)) * $(1024)) + (((((((r60_z / $(1536)) % $(256)) * $(512)) + (((r60_z / $(192)) % $(8)) * $(64)) + ((r60_z % $(192)) - $(128))) / $(65536)) % $(2)) * $(512)) + (((((((r60_z / $(1536)) % $(256)) * $(512)) + (((r60_z / $(192)) % $(8)) * $(64)) + ((r60_z % $(192)) - $(128))) / $(64)) % $(8)) * $(64)) + (((((r60_z / $(1536)) % $(256)) * $(512)) + (((r60_z / $(192)) % $(8)) * $(64)) + ((r60_z % $(192)) - $(128))) % $(64))), dtype=tl.bfloat16))
    r60_value = r60_value.to(tl.bfloat16)
    r60_value = tl.broadcast_to(r60_value, [$(393216)])
    tl.store($(candidate_vcat_2_ptr) + r60_z, r60_value)
    r61_z = tl.arange($(393216))
    r61_value = tl.where((r61_z % $(192)) < $(128), tl.load($(candidate_vbmm_2_ptr) + ((((((((r61_z / $(1536)) % $(256)) * $(1024)) + (((r61_z / $(192)) % $(8)) * $(128)) + ((r61_z % $(192)) - $(0))) / $(1024)) % $(256)) * $(1024)) + (((((((r61_z / $(1536)) % $(256)) * $(1024)) + (((r61_z / $(192)) % $(8)) * $(128)) + ((r61_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r61_z / $(1536)) % $(256)) * $(1024)) + (((r61_z / $(192)) % $(8)) * $(128)) + ((r61_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vadd_5_ptr) + ((((((((r61_z / $(1536)) % $(256)) * $(64)) + ((r61_z % $(192)) - $(128))) / $(64)) % $(128)) * $(128)) + (((((((r61_z / $(1536)) % $(256)) * $(64)) + ((r61_z % $(192)) - $(128))) / $(8192)) % $(2)) * $(64)) + (((((r61_z / $(1536)) % $(256)) * $(64)) + ((r61_z % $(192)) - $(128))) % $(64))), dtype=tl.bfloat16))
    r61_value = r61_value.to(tl.bfloat16)
    r61_value = tl.broadcast_to(r61_value, [$(393216)])
    tl.store($(candidate_vcat_3_ptr) + r61_z, r61_value)
    r62_z = tl.arange($(196608))
    r62_value = tl.load($(candidate_vcat_2_ptr) + ((((r62_z / $(192)) % $(128)) * $(1536)) + (((r62_z / $(24576)) % $(8)) * $(192)) + (r62_z % $(192))), dtype=tl.bfloat16)
    r62_value = r62_value.to(tl.float32)
    r62_value = tl.broadcast_to(r62_value, [$(196608)])
    tl.store($(candidate_vto_copy_28_ptr) + r62_z, r62_value)
    r63_z = tl.arange($(196608))
    r63_value = tl.load($(candidate_vcat_3_ptr) + ((((r63_z / $(192)) % $(128)) * $(1536)) + (((r63_z / $(24576)) % $(8)) * $(192)) + (r63_z % $(192))), dtype=tl.bfloat16)
    r63_value = r63_value.to(tl.float32)
    r63_value = tl.broadcast_to(r63_value, [$(196608)])
    tl.store($(candidate_vto_copy_29_ptr) + r63_z, r63_value)
    r64_row = tl.arange($(128))
    r64_col = tl.arange($(128))
    r64_lane = tl.arange($(192))
    for r64_group in range($(8)) {
      r64_left = tl.load($(candidate_vto_copy_28_ptr) + (((((((r64_group % $(8)) * $(24576)) + (r64_row[:, None] * $(192)) + r64_lane[None, :]) / $(24576)) % $(8)) * $(24576)) + ((((((r64_group % $(8)) * $(24576)) + (r64_row[:, None] * $(192)) + r64_lane[None, :]) / $(192)) % $(128)) * $(192)) + ((((r64_group % $(8)) * $(24576)) + (r64_row[:, None] * $(192)) + r64_lane[None, :]) % $(192))), dtype=tl.float32)
      r64_right = tl.load($(candidate_vto_copy_29_ptr) + (((((((r64_group % $(8)) * $(24576)) + (r64_lane[:, None] * $(128)) + r64_col[None, :]) / $(24576)) % $(8)) * $(24576)) + (((((r64_group % $(8)) * $(24576)) + (r64_lane[:, None] * $(128)) + r64_col[None, :]) % $(128)) * $(192)) + (((((r64_group % $(8)) * $(24576)) + (r64_lane[:, None] * $(128)) + r64_col[None, :]) / $(128)) % $(192))), dtype=tl.float32)
      r64_value = tl.dot(r64_left, r64_right)
      r64_value = r64_value.to(tl.float32)
      tl.store($(candidate_vbmm_4_ptr) + ((r64_group * $(16384)) + (r64_row[:, None] * $(128)) + r64_col[None, :]), r64_value)
    }
    r65_z = tl.arange($(131072))
    r65_value = (tl.load($(candidate_vbmm_4_ptr) + ((((r65_z / $(16384)) % $(8)) * $(16384)) + (((r65_z / $(128)) % $(128)) * $(128)) + (r65_z % $(128))), dtype=tl.float32) * 0.07216878364870322)
    r65_value = r65_value.to(tl.float32)
    r65_value = tl.broadcast_to(r65_value, [$(131072)])
    tl.store($(candidate_vmul_23_ptr) + r65_z, r65_value)
    r66_z = tl.arange($(128))
    r66_value = (tl.cast(r66_z, tl.int64) + tl.full([], $(128), dtype=tl.int64))
    r66_value = tl.broadcast_to(r66_value, [$(128)])
    tl.store($(candidate_vadd_6_ptr) + r66_z, r66_value)
    r67_z = tl.arange($(128))
    r67_value = (tl.load($(candidate_vadd_6_ptr) + r67_z, dtype=tl.int64) - tl.full([], $(128), dtype=tl.int64))
    r67_value = tl.broadcast_to(r67_value, [$(128)])
    tl.store($(candidate_vsub_ptr) + r67_z, r67_value)
    r68_z = tl.arange($(16384))
    r68_value = (tl.cast((r68_z % $(128)), tl.int64) > tl.load($(candidate_vsub_ptr) + ((r68_z / $(128)) % $(128)), dtype=tl.int64))
    r68_value = tl.broadcast_to(r68_value, [$(16384)])
    tl.store($(candidate_vgt_ptr) + r68_z, r68_value)
    r69_z = tl.arange($(131072))
    r69_value = tl.where(tl.load($(candidate_vgt_ptr) + ((((r69_z / $(128)) % $(128)) * $(128)) + (r69_z % $(128))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_23_ptr) + ((((r69_z / $(16384)) % $(8)) * $(16384)) + (((r69_z / $(128)) % $(128)) * $(128)) + (r69_z % $(128))), dtype=tl.float32))
    r69_value = r69_value.to(tl.float32)
    r69_value = tl.broadcast_to(r69_value, [$(131072)])
    tl.store($(candidate_vmasked_fill_2_ptr) + r69_z, r69_value)
    r70_z = tl.arange($(131072))
    r70_lane = tl.arange($(128))
    r70_row_values = tl.load($(candidate_vmasked_fill_2_ptr) + ((((r70_z[:, None] / $(16384)) % $(8)) * $(16384)) + (((r70_z[:, None] / $(128)) % $(128)) * $(128)) + (r70_lane[None, :] % $(128))), dtype=tl.float32)
    r70_maximum = tl.max(r70_row_values, axis=1, keep_dims=true)
    r70_weights = tl.exp(r70_row_values - r70_maximum)
    r70_total = tl.sum(r70_weights, axis=1)
    r70_maximum = tl.max(r70_row_values, axis=1)
    r70_value = tl.exp(tl.load($(candidate_vmasked_fill_2_ptr) + ((((r70_z / $(16384)) % $(8)) * $(16384)) + (((r70_z / $(128)) % $(128)) * $(128)) + (r70_z % $(128))), dtype=tl.float32) - r70_maximum) / r70_total
    r70_value = r70_value.to(tl.float32)
    r70_value = tl.broadcast_to(r70_value, [$(131072)])
    tl.store($(candidate_vsoftmax_ptr) + r70_z, r70_value)
    r71_z = tl.arange($(131072))
    r71_value = tl.load($(candidate_vsoftmax_ptr) + ((((r71_z / $(16384)) % $(8)) * $(16384)) + (((r71_z / $(128)) % $(128)) * $(128)) + (r71_z % $(128))), dtype=tl.float32)
    r71_value = r71_value.to(tl.bfloat16)
    r71_value = tl.broadcast_to(r71_value, [$(131072)])
    tl.store($(candidate_vto_copy_30_ptr) + r71_z, r71_value)
    r72_z = tl.arange($(196608))
    r72_value = tl.load($(candidate_vcat_2_ptr) + ((((r72_z / $(192)) % $(128)) * $(1536)) + (((r72_z / $(24576)) % $(8)) * $(192)) + (r72_z % $(192)) + $(196608)), dtype=tl.bfloat16)
    r72_value = r72_value.to(tl.float32)
    r72_value = tl.broadcast_to(r72_value, [$(196608)])
    tl.store($(candidate_vto_copy_31_ptr) + r72_z, r72_value)
    r73_z = tl.arange($(196608))
    r73_value = tl.load($(candidate_vcat_3_ptr) + ((((r73_z / $(192)) % $(128)) * $(1536)) + (((r73_z / $(24576)) % $(8)) * $(192)) + (r73_z % $(192)) + $(196608)), dtype=tl.bfloat16)
    r73_value = r73_value.to(tl.float32)
    r73_value = tl.broadcast_to(r73_value, [$(196608)])
    tl.store($(candidate_vto_copy_32_ptr) + r73_z, r73_value)
    r74_row = tl.arange($(128))
    r74_col = tl.arange($(128))
    r74_lane = tl.arange($(192))
    for r74_group in range($(8)) {
      r74_left = tl.load($(candidate_vto_copy_31_ptr) + (((((((r74_group % $(8)) * $(24576)) + (r74_row[:, None] * $(192)) + r74_lane[None, :]) / $(24576)) % $(8)) * $(24576)) + ((((((r74_group % $(8)) * $(24576)) + (r74_row[:, None] * $(192)) + r74_lane[None, :]) / $(192)) % $(128)) * $(192)) + ((((r74_group % $(8)) * $(24576)) + (r74_row[:, None] * $(192)) + r74_lane[None, :]) % $(192))), dtype=tl.float32)
      r74_right = tl.load($(candidate_vto_copy_32_ptr) + (((((((r74_group % $(8)) * $(24576)) + (r74_lane[:, None] * $(128)) + r74_col[None, :]) / $(24576)) % $(8)) * $(24576)) + (((((r74_group % $(8)) * $(24576)) + (r74_lane[:, None] * $(128)) + r74_col[None, :]) % $(128)) * $(192)) + (((((r74_group % $(8)) * $(24576)) + (r74_lane[:, None] * $(128)) + r74_col[None, :]) / $(128)) % $(192))), dtype=tl.float32)
      r74_value = tl.dot(r74_left, r74_right)
      r74_value = r74_value.to(tl.float32)
      tl.store($(candidate_vbmm_6_ptr) + ((r74_group * $(16384)) + (r74_row[:, None] * $(128)) + r74_col[None, :]), r74_value)
    }
    r75_z = tl.arange($(131072))
    r75_value = (tl.load($(candidate_vbmm_6_ptr) + ((((r75_z / $(16384)) % $(8)) * $(16384)) + (((r75_z / $(128)) % $(128)) * $(128)) + (r75_z % $(128))), dtype=tl.float32) * 0.07216878364870322)
    r75_value = r75_value.to(tl.float32)
    r75_value = tl.broadcast_to(r75_value, [$(131072)])
    tl.store($(candidate_vmul_24_ptr) + r75_z, r75_value)
    r76_z = tl.arange($(128))
    r76_value = (tl.cast(r76_z, tl.int64) + tl.full([], $(128), dtype=tl.int64))
    r76_value = tl.broadcast_to(r76_value, [$(128)])
    tl.store($(candidate_vadd_7_ptr) + r76_z, r76_value)
    r77_z = tl.arange($(128))
    r77_value = (tl.load($(candidate_vadd_7_ptr) + r77_z, dtype=tl.int64) - tl.full([], $(128), dtype=tl.int64))
    r77_value = tl.broadcast_to(r77_value, [$(128)])
    tl.store($(candidate_vsub_1_ptr) + r77_z, r77_value)
    r78_z = tl.arange($(16384))
    r78_value = (tl.cast((r78_z % $(128)), tl.int64) > tl.load($(candidate_vsub_1_ptr) + ((r78_z / $(128)) % $(128)), dtype=tl.int64))
    r78_value = tl.broadcast_to(r78_value, [$(16384)])
    tl.store($(candidate_vgt_1_ptr) + r78_z, r78_value)
    r79_z = tl.arange($(131072))
    r79_value = tl.where(tl.load($(candidate_vgt_1_ptr) + ((((r79_z / $(128)) % $(128)) * $(128)) + (r79_z % $(128))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_24_ptr) + ((((r79_z / $(16384)) % $(8)) * $(16384)) + (((r79_z / $(128)) % $(128)) * $(128)) + (r79_z % $(128))), dtype=tl.float32))
    r79_value = r79_value.to(tl.float32)
    r79_value = tl.broadcast_to(r79_value, [$(131072)])
    tl.store($(candidate_vmasked_fill_3_ptr) + r79_z, r79_value)
    r80_z = tl.arange($(131072))
    r80_lane = tl.arange($(128))
    r80_row_values = tl.load($(candidate_vmasked_fill_3_ptr) + ((((r80_z[:, None] / $(16384)) % $(8)) * $(16384)) + (((r80_z[:, None] / $(128)) % $(128)) * $(128)) + (r80_lane[None, :] % $(128))), dtype=tl.float32)
    r80_maximum = tl.max(r80_row_values, axis=1, keep_dims=true)
    r80_weights = tl.exp(r80_row_values - r80_maximum)
    r80_total = tl.sum(r80_weights, axis=1)
    r80_maximum = tl.max(r80_row_values, axis=1)
    r80_value = tl.exp(tl.load($(candidate_vmasked_fill_3_ptr) + ((((r80_z / $(16384)) % $(8)) * $(16384)) + (((r80_z / $(128)) % $(128)) * $(128)) + (r80_z % $(128))), dtype=tl.float32) - r80_maximum) / r80_total
    r80_value = r80_value.to(tl.float32)
    r80_value = tl.broadcast_to(r80_value, [$(131072)])
    tl.store($(candidate_vsoftmax_1_ptr) + r80_z, r80_value)
    r81_z = tl.arange($(131072))
    r81_value = tl.load($(candidate_vsoftmax_1_ptr) + ((((r81_z / $(16384)) % $(8)) * $(16384)) + (((r81_z / $(128)) % $(128)) * $(128)) + (r81_z % $(128))), dtype=tl.float32)
    r81_value = r81_value.to(tl.bfloat16)
    r81_value = tl.broadcast_to(r81_value, [$(131072)])
    tl.store($(candidate_vto_copy_33_ptr) + r81_z, r81_value)
    r82_row = tl.arange($(256))
    r82_col = tl.arange($(1024))
    r82_lane = tl.arange($(512))
    r82_left = tl.load($(input1_ptr) + ((((((r82_row[:, None] * $(512)) + r82_lane[None, :]) / $(65536)) % $(2)) * $(65536)) + (((((r82_row[:, None] * $(512)) + r82_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r82_row[:, None] * $(512)) + r82_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r82_right = tl.load($(candidate_vto_copy_6_ptr) + (((r82_lane[:, None] % $(512)) * $(1024)) + (r82_col[None, :] % $(1024))), dtype=tl.bfloat16)
    r82_value = tl.dot(r82_left, r82_right)
    r82_value = r82_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_4_ptr) + ((r82_row[:, None] * $(1024)) + r82_col[None, :]), r82_value)
    r83_row = tl.arange($(128))
    r83_col = tl.arange($(128))
    r83_lane = tl.arange($(128))
    for r83_group in range($(8)) {
      r83_left = tl.load($(candidate_vto_copy_33_ptr) + (((((((r83_group % $(8)) * $(16384)) + (r83_lane[None, :] * $(128)) + r83_row[:, None]) / $(16384)) % $(8)) * $(16384)) + ((((((r83_group % $(8)) * $(16384)) + (r83_lane[None, :] * $(128)) + r83_row[:, None]) / $(128)) % $(128)) * $(128)) + ((((r83_group % $(8)) * $(16384)) + (r83_lane[None, :] * $(128)) + r83_row[:, None]) % $(128))), dtype=tl.bfloat16)
      r83_right = tl.load($(candidate_vmm_4_ptr) + (((((((r83_group % $(8)) * $(16384)) + (r83_lane[:, None] * $(128)) + r83_col[None, :]) / $(128)) % $(128)) * $(1024)) + ((((((r83_group % $(8)) * $(16384)) + (r83_lane[:, None] * $(128)) + r83_col[None, :]) / $(16384)) % $(8)) * $(128)) + ((((r83_group % $(8)) * $(16384)) + (r83_lane[:, None] * $(128)) + r83_col[None, :]) % $(128)) + $(131072)), dtype=tl.bfloat16)
      r83_value = tl.dot(r83_left, r83_right)
      r83_value = r83_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_8_ptr) + ((r83_group * $(16384)) + (r83_row[:, None] * $(128)) + r83_col[None, :]), r83_value)
    }
    r84_row = tl.arange($(128))
    r84_col = tl.arange($(128))
    r84_lane = tl.arange($(128))
    for r84_group in range($(8)) {
      r84_left = tl.load($(candidate_vmm_4_ptr) + (((((((r84_group % $(8)) * $(16384)) + (r84_row[:, None] * $(128)) + r84_lane[None, :]) / $(128)) % $(128)) * $(1024)) + ((((((r84_group % $(8)) * $(16384)) + (r84_row[:, None] * $(128)) + r84_lane[None, :]) / $(16384)) % $(8)) * $(128)) + ((((r84_group % $(8)) * $(16384)) + (r84_row[:, None] * $(128)) + r84_lane[None, :]) % $(128)) + $(131072)), dtype=tl.bfloat16)
      r84_right = tl.load($(candidate_vbmm_3_ptr) + (((((((r84_group % $(8)) * $(16384)) + (r84_col[None, :] * $(128)) + r84_lane[:, None]) / $(128)) % $(128)) * $(1024)) + ((((((r84_group % $(8)) * $(16384)) + (r84_col[None, :] * $(128)) + r84_lane[:, None]) / $(16384)) % $(8)) * $(128)) + ((((r84_group % $(8)) * $(16384)) + (r84_col[None, :] * $(128)) + r84_lane[:, None]) % $(128)) + $(131072)), dtype=tl.bfloat16)
      r84_value = tl.dot(r84_left, r84_right)
      r84_value = r84_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_9_ptr) + ((r84_group * $(16384)) + (r84_row[:, None] * $(128)) + r84_col[None, :]), r84_value)
    }
    r85_z = tl.arange($(131072))
    r85_value = tl.load($(candidate_vbmm_9_ptr) + ((((r85_z / $(16384)) % $(8)) * $(16384)) + (((r85_z / $(128)) % $(128)) * $(128)) + (r85_z % $(128))), dtype=tl.bfloat16)
    r85_value = r85_value.to(tl.float32)
    r85_value = tl.broadcast_to(r85_value, [$(131072)])
    tl.store($(candidate_vto_copy_34_ptr) + r85_z, r85_value)
    r86_z = tl.arange($(131072))
    r86_lane = tl.arange($(128))
    r86_row_values = tl.load($(candidate_vsoftmax_1_ptr) + ((((r86_z[:, None] / $(16384)) % $(8)) * $(16384)) + (((r86_z[:, None] / $(128)) % $(128)) * $(128)) + (r86_lane[None, :] % $(128))), dtype=tl.float32)
    r86_row_grad = tl.load($(candidate_vto_copy_34_ptr) + ((((r86_z[:, None] / $(16384)) % $(8)) * $(16384)) + (((r86_z[:, None] / $(128)) % $(128)) * $(128)) + (r86_lane[None, :] % $(128))), dtype=tl.float32)
    r86_weighted = r86_row_grad * r86_row_values
    r86_dot_grad = tl.sum(r86_weighted, axis=1)
    r86_value = (tl.load($(candidate_vto_copy_34_ptr) + ((((r86_z / $(16384)) % $(8)) * $(16384)) + (((r86_z / $(128)) % $(128)) * $(128)) + (r86_z % $(128))), dtype=tl.float32) - r86_dot_grad) * tl.load($(candidate_vsoftmax_1_ptr) + ((((r86_z / $(16384)) % $(8)) * $(16384)) + (((r86_z / $(128)) % $(128)) * $(128)) + (r86_z % $(128))), dtype=tl.float32)
    r86_value = r86_value.to(tl.float32)
    r86_value = tl.broadcast_to(r86_value, [$(131072)])
    tl.store($(candidate_vsoftmax_backward_data_ptr) + r86_z, r86_value)
    r87_z = tl.arange($(131072))
    r87_value = tl.where(tl.load($(candidate_vgt_1_ptr) + ((((r87_z / $(128)) % $(128)) * $(128)) + (r87_z % $(128))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_ptr) + ((((r87_z / $(16384)) % $(8)) * $(16384)) + (((r87_z / $(128)) % $(128)) * $(128)) + (r87_z % $(128))), dtype=tl.float32))
    r87_value = r87_value.to(tl.float32)
    r87_value = tl.broadcast_to(r87_value, [$(131072)])
    tl.store($(candidate_vmasked_fill_4_ptr) + r87_z, r87_value)
    r88_z = tl.arange($(131072))
    r88_value = (tl.load($(candidate_vmasked_fill_4_ptr) + ((((r88_z / $(16384)) % $(8)) * $(16384)) + (((r88_z / $(128)) % $(128)) * $(128)) + (r88_z % $(128))), dtype=tl.float32) * 0.07216878364870322)
    r88_value = r88_value.to(tl.float32)
    r88_value = tl.broadcast_to(r88_value, [$(131072)])
    tl.store($(candidate_vmul_25_ptr) + r88_z, r88_value)
    r89_row = tl.arange($(192))
    r89_col = tl.arange($(128))
    r89_lane = tl.arange($(128))
    for r89_group in range($(8)) {
      r89_left = tl.load($(candidate_vto_copy_31_ptr) + (((((((r89_group % $(8)) * $(24576)) + (r89_lane[None, :] * $(192)) + r89_row[:, None]) / $(24576)) % $(8)) * $(24576)) + ((((((r89_group % $(8)) * $(24576)) + (r89_lane[None, :] * $(192)) + r89_row[:, None]) / $(192)) % $(128)) * $(192)) + ((((r89_group % $(8)) * $(24576)) + (r89_lane[None, :] * $(192)) + r89_row[:, None]) % $(192))), dtype=tl.float32)
      r89_right = tl.load($(candidate_vmul_25_ptr) + (((((((r89_group % $(8)) * $(16384)) + (r89_lane[:, None] * $(128)) + r89_col[None, :]) / $(16384)) % $(8)) * $(16384)) + ((((((r89_group % $(8)) * $(16384)) + (r89_lane[:, None] * $(128)) + r89_col[None, :]) / $(128)) % $(128)) * $(128)) + ((((r89_group % $(8)) * $(16384)) + (r89_lane[:, None] * $(128)) + r89_col[None, :]) % $(128))), dtype=tl.float32)
      r89_value = tl.dot(r89_left, r89_right)
      r89_value = r89_value.to(tl.float32)
      tl.store($(candidate_vbmm_10_ptr) + ((r89_group * $(24576)) + (r89_row[:, None] * $(128)) + r89_col[None, :]), r89_value)
    }
    r90_row = tl.arange($(128))
    r90_col = tl.arange($(192))
    r90_lane = tl.arange($(128))
    for r90_group in range($(8)) {
      r90_left = tl.load($(candidate_vmul_25_ptr) + (((((((r90_group % $(8)) * $(16384)) + (r90_row[:, None] * $(128)) + r90_lane[None, :]) / $(16384)) % $(8)) * $(16384)) + ((((((r90_group % $(8)) * $(16384)) + (r90_row[:, None] * $(128)) + r90_lane[None, :]) / $(128)) % $(128)) * $(128)) + ((((r90_group % $(8)) * $(16384)) + (r90_row[:, None] * $(128)) + r90_lane[None, :]) % $(128))), dtype=tl.float32)
      r90_right = tl.load($(candidate_vto_copy_32_ptr) + (((((((r90_group % $(8)) * $(24576)) + (r90_col[None, :] * $(128)) + r90_lane[:, None]) / $(24576)) % $(8)) * $(24576)) + (((((r90_group % $(8)) * $(24576)) + (r90_col[None, :] * $(128)) + r90_lane[:, None]) % $(128)) * $(192)) + (((((r90_group % $(8)) * $(24576)) + (r90_col[None, :] * $(128)) + r90_lane[:, None]) / $(128)) % $(192))), dtype=tl.float32)
      r90_value = tl.dot(r90_left, r90_right)
      r90_value = r90_value.to(tl.float32)
      tl.store($(candidate_vbmm_11_ptr) + ((r90_group * $(24576)) + (r90_row[:, None] * $(192)) + r90_col[None, :]), r90_value)
    }
    r91_z = tl.arange($(262144))
    r91_value = tl.where(((((r91_z / $(1024)) % $(256)) >= $(128)) & (((r91_z / $(1024)) % $(256)) < $(256)) & ((((r91_z / $(1024)) % $(256)) - $(128)) % $(1) == $(0))), tl.load($(candidate_vbmm_8_ptr) + ((((((((r91_z / $(128)) % $(8)) * $(16384)) + ((((r91_z / $(1024)) % $(256)) - $(128)) * $(128)) + (r91_z % $(128))) / $(16384)) % $(8)) * $(16384)) + ((((((r91_z / $(128)) % $(8)) * $(128)) + (((r91_z / $(1024)) % $(256)) - $(128))) % $(128)) * $(128)) + (r91_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r91_value = r91_value.to(tl.bfloat16)
    r91_value = tl.broadcast_to(r91_value, [$(262144)])
    tl.store($(candidate_vslice_backward_ptr) + r91_z, r91_value)
    r92_z = tl.arange($(196608))
    r92_value = tl.load($(candidate_vbmm_10_ptr) + ((((r92_z / $(24576)) % $(8)) * $(24576)) + ((r92_z % $(192)) * $(128)) + ((r92_z / $(192)) % $(128))), dtype=tl.float32)
    r92_value = r92_value.to(tl.bfloat16)
    r92_value = tl.broadcast_to(r92_value, [$(196608)])
    tl.store($(candidate_vto_copy_35_ptr) + r92_z, r92_value)
    r93_z = tl.arange($(393216))
    r93_value = tl.where(((((r93_z / $(1536)) % $(256)) >= $(128)) & (((r93_z / $(1536)) % $(256)) < $(256)) & ((((r93_z / $(1536)) % $(256)) - $(128)) % $(1) == $(0))), tl.load($(candidate_vto_copy_35_ptr) + ((((r93_z / $(192)) % $(8)) * $(24576)) + (((((r93_z / $(1536)) % $(256)) - $(128)) % $(128)) * $(192)) + (r93_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r93_value = r93_value.to(tl.bfloat16)
    r93_value = tl.broadcast_to(r93_value, [$(393216)])
    tl.store($(candidate_vslice_backward_1_ptr) + r93_z, r93_value)
    r94_z = tl.arange($(196608))
    r94_value = tl.load($(candidate_vbmm_11_ptr) + ((((r94_z / $(24576)) % $(8)) * $(24576)) + (((r94_z / $(192)) % $(128)) * $(192)) + (r94_z % $(192))), dtype=tl.float32)
    r94_value = r94_value.to(tl.bfloat16)
    r94_value = tl.broadcast_to(r94_value, [$(196608)])
    tl.store($(candidate_vto_copy_36_ptr) + r94_z, r94_value)
    r95_z = tl.arange($(393216))
    r95_value = tl.where(((((r95_z / $(1536)) % $(256)) >= $(128)) & (((r95_z / $(1536)) % $(256)) < $(256)) & ((((r95_z / $(1536)) % $(256)) - $(128)) % $(1) == $(0))), tl.load($(candidate_vto_copy_36_ptr) + ((((r95_z / $(192)) % $(8)) * $(24576)) + (((((r95_z / $(1536)) % $(256)) - $(128)) % $(128)) * $(192)) + (r95_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r95_value = r95_value.to(tl.bfloat16)
    r95_value = tl.broadcast_to(r95_value, [$(393216)])
    tl.store($(candidate_vslice_backward_2_ptr) + r95_z, r95_value)
    r96_row = tl.arange($(128))
    r96_col = tl.arange($(128))
    r96_lane = tl.arange($(128))
    for r96_group in range($(8)) {
      r96_left = tl.load($(candidate_vto_copy_30_ptr) + (((((((r96_group % $(8)) * $(16384)) + (r96_lane[None, :] * $(128)) + r96_row[:, None]) / $(16384)) % $(8)) * $(16384)) + ((((((r96_group % $(8)) * $(16384)) + (r96_lane[None, :] * $(128)) + r96_row[:, None]) / $(128)) % $(128)) * $(128)) + ((((r96_group % $(8)) * $(16384)) + (r96_lane[None, :] * $(128)) + r96_row[:, None]) % $(128))), dtype=tl.bfloat16)
      r96_right = tl.load($(candidate_vmm_4_ptr) + (((((((r96_group % $(8)) * $(16384)) + (r96_lane[:, None] * $(128)) + r96_col[None, :]) / $(128)) % $(128)) * $(1024)) + ((((((r96_group % $(8)) * $(16384)) + (r96_lane[:, None] * $(128)) + r96_col[None, :]) / $(16384)) % $(8)) * $(128)) + ((((r96_group % $(8)) * $(16384)) + (r96_lane[:, None] * $(128)) + r96_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r96_value = tl.dot(r96_left, r96_right)
      r96_value = r96_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_12_ptr) + ((r96_group * $(16384)) + (r96_row[:, None] * $(128)) + r96_col[None, :]), r96_value)
    }
    r97_row = tl.arange($(128))
    r97_col = tl.arange($(128))
    r97_lane = tl.arange($(128))
    for r97_group in range($(8)) {
      r97_left = tl.load($(candidate_vmm_4_ptr) + (((((((r97_group % $(8)) * $(16384)) + (r97_row[:, None] * $(128)) + r97_lane[None, :]) / $(128)) % $(128)) * $(1024)) + ((((((r97_group % $(8)) * $(16384)) + (r97_row[:, None] * $(128)) + r97_lane[None, :]) / $(16384)) % $(8)) * $(128)) + ((((r97_group % $(8)) * $(16384)) + (r97_row[:, None] * $(128)) + r97_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r97_right = tl.load($(candidate_vbmm_3_ptr) + (((((((r97_group % $(8)) * $(16384)) + (r97_col[None, :] * $(128)) + r97_lane[:, None]) / $(128)) % $(128)) * $(1024)) + ((((((r97_group % $(8)) * $(16384)) + (r97_col[None, :] * $(128)) + r97_lane[:, None]) / $(16384)) % $(8)) * $(128)) + ((((r97_group % $(8)) * $(16384)) + (r97_col[None, :] * $(128)) + r97_lane[:, None]) % $(128))), dtype=tl.bfloat16)
      r97_value = tl.dot(r97_left, r97_right)
      r97_value = r97_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_13_ptr) + ((r97_group * $(16384)) + (r97_row[:, None] * $(128)) + r97_col[None, :]), r97_value)
    }
    r98_z = tl.arange($(131072))
    r98_value = tl.load($(candidate_vbmm_13_ptr) + ((((r98_z / $(16384)) % $(8)) * $(16384)) + (((r98_z / $(128)) % $(128)) * $(128)) + (r98_z % $(128))), dtype=tl.bfloat16)
    r98_value = r98_value.to(tl.float32)
    r98_value = tl.broadcast_to(r98_value, [$(131072)])
    tl.store($(candidate_vto_copy_37_ptr) + r98_z, r98_value)
    r99_z = tl.arange($(131072))
    r99_lane = tl.arange($(128))
    r99_row_values = tl.load($(candidate_vsoftmax_ptr) + ((((r99_z[:, None] / $(16384)) % $(8)) * $(16384)) + (((r99_z[:, None] / $(128)) % $(128)) * $(128)) + (r99_lane[None, :] % $(128))), dtype=tl.float32)
    r99_row_grad = tl.load($(candidate_vto_copy_37_ptr) + ((((r99_z[:, None] / $(16384)) % $(8)) * $(16384)) + (((r99_z[:, None] / $(128)) % $(128)) * $(128)) + (r99_lane[None, :] % $(128))), dtype=tl.float32)
    r99_weighted = r99_row_grad * r99_row_values
    r99_dot_grad = tl.sum(r99_weighted, axis=1)
    r99_value = (tl.load($(candidate_vto_copy_37_ptr) + ((((r99_z / $(16384)) % $(8)) * $(16384)) + (((r99_z / $(128)) % $(128)) * $(128)) + (r99_z % $(128))), dtype=tl.float32) - r99_dot_grad) * tl.load($(candidate_vsoftmax_ptr) + ((((r99_z / $(16384)) % $(8)) * $(16384)) + (((r99_z / $(128)) % $(128)) * $(128)) + (r99_z % $(128))), dtype=tl.float32)
    r99_value = r99_value.to(tl.float32)
    r99_value = tl.broadcast_to(r99_value, [$(131072)])
    tl.store($(candidate_vsoftmax_backward_data_1_ptr) + r99_z, r99_value)
    r100_z = tl.arange($(131072))
    r100_value = tl.where(tl.load($(candidate_vgt_ptr) + ((((r100_z / $(128)) % $(128)) * $(128)) + (r100_z % $(128))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_1_ptr) + ((((r100_z / $(16384)) % $(8)) * $(16384)) + (((r100_z / $(128)) % $(128)) * $(128)) + (r100_z % $(128))), dtype=tl.float32))
    r100_value = r100_value.to(tl.float32)
    r100_value = tl.broadcast_to(r100_value, [$(131072)])
    tl.store($(candidate_vmasked_fill_5_ptr) + r100_z, r100_value)
    r101_z = tl.arange($(131072))
    r101_value = (tl.load($(candidate_vmasked_fill_5_ptr) + ((((r101_z / $(16384)) % $(8)) * $(16384)) + (((r101_z / $(128)) % $(128)) * $(128)) + (r101_z % $(128))), dtype=tl.float32) * 0.07216878364870322)
    r101_value = r101_value.to(tl.float32)
    r101_value = tl.broadcast_to(r101_value, [$(131072)])
    tl.store($(candidate_vmul_26_ptr) + r101_z, r101_value)
    r102_row = tl.arange($(192))
    r102_col = tl.arange($(128))
    r102_lane = tl.arange($(128))
    for r102_group in range($(8)) {
      r102_left = tl.load($(candidate_vto_copy_28_ptr) + (((((((r102_group % $(8)) * $(24576)) + (r102_lane[None, :] * $(192)) + r102_row[:, None]) / $(24576)) % $(8)) * $(24576)) + ((((((r102_group % $(8)) * $(24576)) + (r102_lane[None, :] * $(192)) + r102_row[:, None]) / $(192)) % $(128)) * $(192)) + ((((r102_group % $(8)) * $(24576)) + (r102_lane[None, :] * $(192)) + r102_row[:, None]) % $(192))), dtype=tl.float32)
      r102_right = tl.load($(candidate_vmul_26_ptr) + (((((((r102_group % $(8)) * $(16384)) + (r102_lane[:, None] * $(128)) + r102_col[None, :]) / $(16384)) % $(8)) * $(16384)) + ((((((r102_group % $(8)) * $(16384)) + (r102_lane[:, None] * $(128)) + r102_col[None, :]) / $(128)) % $(128)) * $(128)) + ((((r102_group % $(8)) * $(16384)) + (r102_lane[:, None] * $(128)) + r102_col[None, :]) % $(128))), dtype=tl.float32)
      r102_value = tl.dot(r102_left, r102_right)
      r102_value = r102_value.to(tl.float32)
      tl.store($(candidate_vbmm_14_ptr) + ((r102_group * $(24576)) + (r102_row[:, None] * $(128)) + r102_col[None, :]), r102_value)
    }
    r103_row = tl.arange($(128))
    r103_col = tl.arange($(192))
    r103_lane = tl.arange($(128))
    for r103_group in range($(8)) {
      r103_left = tl.load($(candidate_vmul_26_ptr) + (((((((r103_group % $(8)) * $(16384)) + (r103_row[:, None] * $(128)) + r103_lane[None, :]) / $(16384)) % $(8)) * $(16384)) + ((((((r103_group % $(8)) * $(16384)) + (r103_row[:, None] * $(128)) + r103_lane[None, :]) / $(128)) % $(128)) * $(128)) + ((((r103_group % $(8)) * $(16384)) + (r103_row[:, None] * $(128)) + r103_lane[None, :]) % $(128))), dtype=tl.float32)
      r103_right = tl.load($(candidate_vto_copy_29_ptr) + (((((((r103_group % $(8)) * $(24576)) + (r103_col[None, :] * $(128)) + r103_lane[:, None]) / $(24576)) % $(8)) * $(24576)) + (((((r103_group % $(8)) * $(24576)) + (r103_col[None, :] * $(128)) + r103_lane[:, None]) % $(128)) * $(192)) + (((((r103_group % $(8)) * $(24576)) + (r103_col[None, :] * $(128)) + r103_lane[:, None]) / $(128)) % $(192))), dtype=tl.float32)
      r103_value = tl.dot(r103_left, r103_right)
      r103_value = r103_value.to(tl.float32)
      tl.store($(candidate_vbmm_15_ptr) + ((r103_group * $(24576)) + (r103_row[:, None] * $(192)) + r103_col[None, :]), r103_value)
    }
    r104_z = tl.arange($(262144))
    r104_value = tl.where(((((r104_z / $(1024)) % $(256)) >= $(0)) & (((r104_z / $(1024)) % $(256)) < $(128)) & ((((r104_z / $(1024)) % $(256)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vbmm_12_ptr) + ((((((((r104_z / $(128)) % $(8)) * $(16384)) + ((((r104_z / $(1024)) % $(256)) - $(0)) * $(128)) + (r104_z % $(128))) / $(16384)) % $(8)) * $(16384)) + ((((((r104_z / $(128)) % $(8)) * $(128)) + (((r104_z / $(1024)) % $(256)) - $(0))) % $(128)) * $(128)) + (r104_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r104_value = r104_value.to(tl.bfloat16)
    r104_value = tl.broadcast_to(r104_value, [$(262144)])
    tl.store($(candidate_vslice_backward_3_ptr) + r104_z, r104_value)
    r105_z = tl.arange($(262144))
    r105_value = (tl.load($(candidate_vslice_backward_ptr) + ((((r105_z / $(1024)) % $(256)) * $(1024)) + (((r105_z / $(128)) % $(8)) * $(128)) + (r105_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_3_ptr) + ((((r105_z / $(1024)) % $(256)) * $(1024)) + (((r105_z / $(128)) % $(8)) * $(128)) + (r105_z % $(128))), dtype=tl.bfloat16))
    r105_value = r105_value.to(tl.bfloat16)
    r105_value = tl.broadcast_to(r105_value, [$(262144)])
    tl.store($(candidate_vadd_8_ptr) + r105_z, r105_value)
    r106_z = tl.arange($(196608))
    r106_value = tl.load($(candidate_vbmm_14_ptr) + ((((r106_z / $(24576)) % $(8)) * $(24576)) + ((r106_z % $(192)) * $(128)) + ((r106_z / $(192)) % $(128))), dtype=tl.float32)
    r106_value = r106_value.to(tl.bfloat16)
    r106_value = tl.broadcast_to(r106_value, [$(196608)])
    tl.store($(candidate_vto_copy_38_ptr) + r106_z, r106_value)
    r107_z = tl.arange($(393216))
    r107_value = tl.where(((((r107_z / $(1536)) % $(256)) >= $(0)) & (((r107_z / $(1536)) % $(256)) < $(128)) & ((((r107_z / $(1536)) % $(256)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vto_copy_38_ptr) + ((((r107_z / $(192)) % $(8)) * $(24576)) + (((((r107_z / $(1536)) % $(256)) - $(0)) % $(128)) * $(192)) + (r107_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r107_value = r107_value.to(tl.bfloat16)
    r107_value = tl.broadcast_to(r107_value, [$(393216)])
    tl.store($(candidate_vslice_backward_4_ptr) + r107_z, r107_value)
    r108_z = tl.arange($(393216))
    r108_value = (tl.load($(candidate_vslice_backward_1_ptr) + ((((r108_z / $(1536)) % $(256)) * $(1536)) + (((r108_z / $(192)) % $(8)) * $(192)) + (r108_z % $(192))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_4_ptr) + ((((r108_z / $(1536)) % $(256)) * $(1536)) + (((r108_z / $(192)) % $(8)) * $(192)) + (r108_z % $(192))), dtype=tl.bfloat16))
    r108_value = r108_value.to(tl.bfloat16)
    r108_value = tl.broadcast_to(r108_value, [$(393216)])
    tl.store($(candidate_vadd_9_ptr) + r108_z, r108_value)
    r109_z = tl.arange($(196608))
    r109_value = tl.load($(candidate_vbmm_15_ptr) + ((((r109_z / $(24576)) % $(8)) * $(24576)) + (((r109_z / $(192)) % $(128)) * $(192)) + (r109_z % $(192))), dtype=tl.float32)
    r109_value = r109_value.to(tl.bfloat16)
    r109_value = tl.broadcast_to(r109_value, [$(196608)])
    tl.store($(candidate_vto_copy_39_ptr) + r109_z, r109_value)
    r110_z = tl.arange($(393216))
    r110_value = tl.where(((((r110_z / $(1536)) % $(256)) >= $(0)) & (((r110_z / $(1536)) % $(256)) < $(128)) & ((((r110_z / $(1536)) % $(256)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vto_copy_39_ptr) + ((((r110_z / $(192)) % $(8)) * $(24576)) + (((((r110_z / $(1536)) % $(256)) - $(0)) % $(128)) * $(192)) + (r110_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r110_value = r110_value.to(tl.bfloat16)
    r110_value = tl.broadcast_to(r110_value, [$(393216)])
    tl.store($(candidate_vslice_backward_5_ptr) + r110_z, r110_value)
    r111_z = tl.arange($(393216))
    r111_value = (tl.load($(candidate_vslice_backward_2_ptr) + ((((r111_z / $(1536)) % $(256)) * $(1536)) + (((r111_z / $(192)) % $(8)) * $(192)) + (r111_z % $(192))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_5_ptr) + ((((r111_z / $(1536)) % $(256)) * $(1536)) + (((r111_z / $(192)) % $(8)) * $(192)) + (r111_z % $(192))), dtype=tl.bfloat16))
    r111_value = r111_value.to(tl.bfloat16)
    r111_value = tl.broadcast_to(r111_value, [$(393216)])
    tl.store($(candidate_vadd_10_ptr) + r111_z, r111_value)
    r112_z = tl.arange($(16384))
    r112_lane = tl.arange($(8))
    r112_value = tl.sum(tl.load($(candidate_vadd_9_ptr) + ((((r112_z[:, None] / $(64)) % $(256)) * $(1536)) + (r112_lane[None, :] * $(192)) + (r112_z[:, None] % $(64)) + $(128)), dtype=tl.bfloat16), axis=1)
    r112_value = r112_value.to(tl.bfloat16)
    r112_value = tl.broadcast_to(r112_value, [$(16384)])
    tl.store($(candidate_vsum_1_ptr) + r112_z, r112_value)
    r113_row = tl.arange($(256))
    r113_col = tl.arange($(256))
    r113_lane = tl.arange($(1024))
    for r113_group in range($(1)) {
      r113_left = tl.load($(candidate_vadd_8_ptr) + ((((((r113_row[:, None] * $(1024)) + r113_lane[None, :]) / $(1024)) % $(256)) * $(1024)) + (((((r113_row[:, None] * $(1024)) + r113_lane[None, :]) / $(128)) % $(8)) * $(128)) + (((r113_row[:, None] * $(1024)) + r113_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r113_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r113_col[None, :] * $(1024)) + r113_lane[:, None]) / $(128)) % $(8)) * $(65536)) + ((((r113_col[None, :] * $(1024)) + r113_lane[:, None]) % $(128)) * $(256)) + ((((r113_col[None, :] * $(1024)) + r113_lane[:, None]) / $(1024)) % $(256)) + $(32768)), dtype=tl.bfloat16)
      r113_value = tl.dot(r113_left, r113_right)
      r113_value = r113_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_16_ptr) + ((r113_group * $(65536)) + (r113_row[:, None] * $(256)) + r113_col[None, :]), r113_value)
    }
    r114_row = tl.arange($(256))
    r114_col = tl.arange($(256))
    r114_lane = tl.arange($(1024))
    for r114_group in range($(1)) {
      r114_left = tl.load($(candidate_vadd_9_ptr) + ((((((r114_row[:, None] * $(1024)) + r114_lane[None, :]) / $(1024)) % $(256)) * $(1536)) + (((((r114_row[:, None] * $(1024)) + r114_lane[None, :]) / $(128)) % $(8)) * $(192)) + (((r114_row[:, None] * $(1024)) + r114_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r114_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r114_col[None, :] * $(1024)) + r114_lane[:, None]) / $(128)) % $(8)) * $(65536)) + ((((r114_col[None, :] * $(1024)) + r114_lane[:, None]) % $(128)) * $(256)) + ((((r114_col[None, :] * $(1024)) + r114_lane[:, None]) / $(1024)) % $(256))), dtype=tl.bfloat16)
      r114_value = tl.dot(r114_left, r114_right)
      r114_value = r114_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_17_ptr) + ((r114_group * $(65536)) + (r114_row[:, None] * $(256)) + r114_col[None, :]), r114_value)
    }
    r115_z = tl.arange($(65536))
    r115_value = (tl.load($(candidate_vbmm_16_ptr) + ((((r115_z / $(256)) % $(256)) * $(256)) + (r115_z % $(256))), dtype=tl.bfloat16) + tl.load($(candidate_vbmm_17_ptr) + ((((r115_z / $(256)) % $(256)) * $(256)) + (r115_z % $(256))), dtype=tl.bfloat16))
    r115_value = r115_value.to(tl.bfloat16)
    r115_value = tl.broadcast_to(r115_value, [$(65536)])
    tl.store($(candidate_vadd_11_ptr) + r115_z, r115_value)
    r116_z = tl.arange($(65536))
    r116_value = tl.where(($(0) == $(0)), tl.load($(candidate_vadd_11_ptr) + ((((r116_z / $(256)) % $(256)) * $(256)) + (r116_z % $(256))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r116_value = r116_value.to(tl.bfloat16)
    r116_value = tl.broadcast_to(r116_value, [$(65536)])
    tl.store($(candidate_vselect_backward_ptr) + r116_z, r116_value)
    r117_z = tl.arange($(16384))
    r117_value = (tl.load($(candidate_vsum_1_ptr) + ((((r117_z / $(64)) % $(2)) * $(8192)) + (((r117_z / $(128)) % $(128)) * $(64)) + (r117_z % $(64))), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_27_ptr) + ((((r117_z / $(128)) % $(128)) * $(64)) + (r117_z % $(64))), dtype=tl.bfloat16))
    r117_value = r117_value.to(tl.bfloat16)
    r117_value = tl.broadcast_to(r117_value, [$(16384)])
    tl.store($(candidate_vmul_27_ptr) + r117_z, r117_value)
    r118_z = tl.arange($(16384))
    r118_value = (tl.load($(candidate_vsum_1_ptr) + ((((r118_z / $(64)) % $(2)) * $(8192)) + (((r118_z / $(128)) % $(128)) * $(64)) + (r118_z % $(64))), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_26_ptr) + ((((r118_z / $(128)) % $(128)) * $(64)) + (r118_z % $(64))), dtype=tl.bfloat16))
    r118_value = r118_value.to(tl.bfloat16)
    r118_value = tl.broadcast_to(r118_value, [$(16384)])
    tl.store($(candidate_vmul_28_ptr) + r118_z, r118_value)
    r119_z = tl.arange($(16384))
    r119_value = tl.where(((r119_z % $(2)) == $(0)), tl.load($(candidate_vmul_27_ptr) + ((((r119_z / $(128)) % $(128)) * $(128)) + (((r119_z / $(64)) % $(2)) * $(64)) + (((r119_z / $(2)) % $(32)) * $(2)) + $(1)), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r119_value = r119_value.to(tl.bfloat16)
    r119_value = tl.broadcast_to(r119_value, [$(16384)])
    tl.store($(candidate_vselect_backward_1_ptr) + r119_z, r119_value)
    r120_z = tl.arange($(8192))
    r120_value = -tl.load($(candidate_vmul_27_ptr) + ((((r120_z / $(64)) % $(128)) * $(128)) + (((r120_z / $(32)) % $(2)) * $(64)) + ((r120_z % $(32)) * $(2))), dtype=tl.bfloat16)
    r120_value = r120_value.to(tl.bfloat16)
    r120_value = tl.broadcast_to(r120_value, [$(8192)])
    tl.store($(candidate_vneg_2_ptr) + r120_z, r120_value)
    r121_z = tl.arange($(16384))
    r121_value = tl.where(((r121_z % $(2)) == $(1)), tl.load($(candidate_vneg_2_ptr) + ((((r121_z / $(128)) % $(128)) * $(64)) + (((r121_z / $(64)) % $(2)) * $(32)) + ((r121_z / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r121_value = r121_value.to(tl.bfloat16)
    r121_value = tl.broadcast_to(r121_value, [$(16384)])
    tl.store($(candidate_vselect_backward_2_ptr) + r121_z, r121_value)
    r122_z = tl.arange($(16384))
    r122_value = (tl.load($(candidate_vselect_backward_1_ptr) + ((((r122_z / $(128)) % $(128)) * $(128)) + (((r122_z / $(64)) % $(2)) * $(64)) + (((r122_z / $(2)) % $(32)) * $(2)) + (r122_z % $(2))), dtype=tl.bfloat16) + tl.load($(candidate_vselect_backward_2_ptr) + ((((r122_z / $(128)) % $(128)) * $(128)) + (((r122_z / $(64)) % $(2)) * $(64)) + (((r122_z / $(2)) % $(32)) * $(2)) + (r122_z % $(2))), dtype=tl.bfloat16))
    r122_value = r122_value.to(tl.bfloat16)
    r122_value = tl.broadcast_to(r122_value, [$(16384)])
    tl.store($(candidate_vadd_12_ptr) + r122_z, r122_value)
    r123_z = tl.arange($(16384))
    r123_value = (tl.load($(candidate_vmul_28_ptr) + ((((r123_z / $(128)) % $(128)) * $(128)) + (((r123_z / $(64)) % $(2)) * $(64)) + (r123_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vadd_12_ptr) + ((((r123_z / $(128)) % $(128)) * $(128)) + (((r123_z / $(64)) % $(2)) * $(64)) + (((((((r123_z / $(128)) % $(128)) * $(128)) + (((r123_z / $(64)) % $(2)) * $(64)) + (r123_z % $(64))) / $(2)) % $(32)) * $(2)) + (((((r123_z / $(128)) % $(128)) * $(128)) + (((r123_z / $(64)) % $(2)) * $(64)) + (r123_z % $(64))) % $(2))), dtype=tl.bfloat16))
    r123_value = r123_value.to(tl.bfloat16)
    r123_value = tl.broadcast_to(r123_value, [$(16384)])
    tl.store($(candidate_vadd_13_ptr) + r123_z, r123_value)
    r124_z = tl.arange($(131072))
    r124_value = (tl.load($(candidate_vadd_10_ptr) + ((((r124_z / $(512)) % $(2)) * $(196608)) + (((r124_z / $(1024)) % $(128)) * $(1536)) + (((r124_z / $(64)) % $(8)) * $(192)) + (r124_z % $(64)) + $(128)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_25_ptr) + ((((r124_z / $(1024)) % $(128)) * $(64)) + (r124_z % $(64))), dtype=tl.bfloat16))
    r124_value = r124_value.to(tl.bfloat16)
    r124_value = tl.broadcast_to(r124_value, [$(131072)])
    tl.store($(candidate_vmul_29_ptr) + r124_z, r124_value)
    r125_z = tl.arange($(131072))
    r125_value = (tl.load($(candidate_vadd_10_ptr) + ((((r125_z / $(512)) % $(2)) * $(196608)) + (((r125_z / $(1024)) % $(128)) * $(1536)) + (((r125_z / $(64)) % $(8)) * $(192)) + (r125_z % $(64)) + $(128)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_24_ptr) + ((((r125_z / $(1024)) % $(128)) * $(64)) + (r125_z % $(64))), dtype=tl.bfloat16))
    r125_value = r125_value.to(tl.bfloat16)
    r125_value = tl.broadcast_to(r125_value, [$(131072)])
    tl.store($(candidate_vmul_30_ptr) + r125_z, r125_value)
    r126_z = tl.arange($(131072))
    r126_value = tl.where(((r126_z % $(2)) == $(0)), tl.load($(candidate_vmul_29_ptr) + ((((r126_z / $(1024)) % $(128)) * $(1024)) + (((r126_z / $(512)) % $(2)) * $(512)) + (((r126_z / $(64)) % $(8)) * $(64)) + (((r126_z / $(2)) % $(32)) * $(2)) + $(1)), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r126_value = r126_value.to(tl.bfloat16)
    r126_value = tl.broadcast_to(r126_value, [$(131072)])
    tl.store($(candidate_vselect_backward_3_ptr) + r126_z, r126_value)
    r127_z = tl.arange($(65536))
    r127_value = -tl.load($(candidate_vmul_29_ptr) + ((((r127_z / $(512)) % $(128)) * $(1024)) + (((r127_z / $(256)) % $(2)) * $(512)) + (((r127_z / $(32)) % $(8)) * $(64)) + ((r127_z % $(32)) * $(2))), dtype=tl.bfloat16)
    r127_value = r127_value.to(tl.bfloat16)
    r127_value = tl.broadcast_to(r127_value, [$(65536)])
    tl.store($(candidate_vneg_3_ptr) + r127_z, r127_value)
    r128_z = tl.arange($(131072))
    r128_value = tl.where(((r128_z % $(2)) == $(1)), tl.load($(candidate_vneg_3_ptr) + ((((r128_z / $(1024)) % $(128)) * $(512)) + (((r128_z / $(512)) % $(2)) * $(256)) + (((r128_z / $(64)) % $(8)) * $(32)) + ((r128_z / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r128_value = r128_value.to(tl.bfloat16)
    r128_value = tl.broadcast_to(r128_value, [$(131072)])
    tl.store($(candidate_vselect_backward_4_ptr) + r128_z, r128_value)
    r129_z = tl.arange($(131072))
    r129_value = (tl.load($(candidate_vselect_backward_3_ptr) + ((((r129_z / $(1024)) % $(128)) * $(1024)) + (((r129_z / $(512)) % $(2)) * $(512)) + (((r129_z / $(64)) % $(8)) * $(64)) + (((r129_z / $(2)) % $(32)) * $(2)) + (r129_z % $(2))), dtype=tl.bfloat16) + tl.load($(candidate_vselect_backward_4_ptr) + ((((r129_z / $(1024)) % $(128)) * $(1024)) + (((r129_z / $(512)) % $(2)) * $(512)) + (((r129_z / $(64)) % $(8)) * $(64)) + (((r129_z / $(2)) % $(32)) * $(2)) + (r129_z % $(2))), dtype=tl.bfloat16))
    r129_value = r129_value.to(tl.bfloat16)
    r129_value = tl.broadcast_to(r129_value, [$(131072)])
    tl.store($(candidate_vadd_14_ptr) + r129_z, r129_value)
    r130_z = tl.arange($(131072))
    r130_value = (tl.load($(candidate_vmul_30_ptr) + ((((r130_z / $(1024)) % $(128)) * $(1024)) + (((r130_z / $(512)) % $(2)) * $(512)) + (((r130_z / $(64)) % $(8)) * $(64)) + (r130_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vadd_14_ptr) + ((((r130_z / $(1024)) % $(128)) * $(1024)) + (((r130_z / $(512)) % $(2)) * $(512)) + (((r130_z / $(64)) % $(8)) * $(64)) + (((((((r130_z / $(1024)) % $(128)) * $(1024)) + (((r130_z / $(512)) % $(2)) * $(512)) + (((r130_z / $(64)) % $(8)) * $(64)) + (r130_z % $(64))) / $(2)) % $(32)) * $(2)) + (((((r130_z / $(1024)) % $(128)) * $(1024)) + (((r130_z / $(512)) % $(2)) * $(512)) + (((r130_z / $(64)) % $(8)) * $(64)) + (r130_z % $(64))) % $(2))), dtype=tl.bfloat16))
    r130_value = r130_value.to(tl.bfloat16)
    r130_value = tl.broadcast_to(r130_value, [$(131072)])
    tl.store($(candidate_vadd_15_ptr) + r130_z, r130_value)
    r131_z = tl.arange($(393216))
    r131_value = tl.where((r131_z % $(192)) < $(128), tl.load($(candidate_vadd_10_ptr) + ((((((((r131_z / $(196608)) % $(2)) * $(131072)) + (((r131_z / $(192)) % $(128)) * $(1024)) + (((r131_z / $(24576)) % $(8)) * $(128)) + ((r131_z % $(192)) - $(0))) / $(1024)) % $(256)) * $(1536)) + (((((((r131_z / $(196608)) % $(2)) * $(131072)) + (((r131_z / $(192)) % $(128)) * $(1024)) + (((r131_z / $(24576)) % $(8)) * $(128)) + ((r131_z % $(192)) - $(0))) / $(128)) % $(8)) * $(192)) + (((((r131_z / $(196608)) % $(2)) * $(131072)) + (((r131_z / $(192)) % $(128)) * $(1024)) + (((r131_z / $(24576)) % $(8)) * $(128)) + ((r131_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vadd_15_ptr) + ((((r131_z / $(192)) % $(128)) * $(1024)) + (((r131_z / $(196608)) % $(2)) * $(512)) + (((r131_z / $(24576)) % $(8)) * $(64)) + (((r131_z % $(192)) - $(128)) % $(64))), dtype=tl.bfloat16))
    r131_value = r131_value.to(tl.bfloat16)
    r131_value = tl.broadcast_to(r131_value, [$(393216)])
    tl.store($(candidate_vcat_5_ptr) + r131_z, r131_value)
    r132_z = tl.arange($(65536))
    r132_value = tl.load($(candidate_vselect_backward_ptr) + ((((r132_z / $(32768)) % $(2)) * $(32768)) + (((r132_z / $(256)) % $(128)) * $(256)) + (r132_z % $(256))), dtype=tl.bfloat16)
    r132_value = r132_value.to(tl.float32)
    r132_value = tl.broadcast_to(r132_value, [$(65536)])
    tl.store($(candidate_vto_copy_40_ptr) + r132_z, r132_value)
    r133_z = tl.arange($(65536))
    r133_value = (tl.load($(candidate_vto_copy_40_ptr) + ((((r133_z / $(32768)) % $(2)) * $(32768)) + (((r133_z / $(256)) % $(128)) * $(256)) + (r133_z % $(256))), dtype=tl.float32) * tl.cast(tl.load($(candidate_vto_copy_4_ptr) + (r133_z % $(256)), dtype=tl.bfloat16), tl.float32))
    r133_value = r133_value.to(tl.float32)
    r133_value = tl.broadcast_to(r133_value, [$(65536)])
    tl.store($(candidate_vmul_31_ptr) + r133_z, r133_value)
    r134_z = tl.arange($(65536))
    r134_value = (tl.load($(candidate_vmul_31_ptr) + ((((r134_z / $(32768)) % $(2)) * $(32768)) + (((r134_z / $(256)) % $(128)) * $(256)) + (r134_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vto_copy_22_ptr) + ((((r134_z / $(32768)) % $(2)) * $(32768)) + (((r134_z / $(256)) % $(128)) * $(256)) + (r134_z % $(256))), dtype=tl.float32))
    r134_value = r134_value.to(tl.float32)
    r134_value = tl.broadcast_to(r134_value, [$(65536)])
    tl.store($(candidate_vmul_32_ptr) + r134_z, r134_value)
    r135_z = tl.arange($(65536))
    r135_value = (tl.load($(candidate_vmul_31_ptr) + ((((r135_z / $(32768)) % $(2)) * $(32768)) + (((r135_z / $(256)) % $(128)) * $(256)) + (r135_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_1_ptr) + ((((r135_z / $(32768)) % $(2)) * $(128)) + ((r135_z / $(256)) % $(128))), dtype=tl.float32))
    r135_value = r135_value.to(tl.float32)
    r135_value = tl.broadcast_to(r135_value, [$(65536)])
    tl.store($(candidate_vmul_33_ptr) + r135_z, r135_value)
    r136_z = tl.arange($(256))
    r136_lane = tl.arange($(256))
    r136_value = tl.sum(tl.load($(candidate_vmul_32_ptr) + ((((r136_z[:, None] / $(128)) % $(2)) * $(32768)) + ((r136_z[:, None] % $(128)) * $(256)) + r136_lane[None, :]), dtype=tl.float32), axis=1)
    r136_value = r136_value.to(tl.float32)
    r136_value = tl.broadcast_to(r136_value, [$(256)])
    tl.store($(candidate_vsum_2_ptr) + r136_z, r136_value)
    r137_z = tl.arange($(256))
    r137_value = ((tl.load($(candidate_vrsqrt_1_ptr) + ((((r137_z / $(128)) % $(2)) * $(128)) + (r137_z % $(128))), dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_1_ptr) + ((((r137_z / $(128)) % $(2)) * $(128)) + (r137_z % $(128))), dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_1_ptr) + ((((r137_z / $(128)) % $(2)) * $(128)) + (r137_z % $(128))), dtype=tl.float32)))
    r137_value = r137_value.to(tl.float32)
    r137_value = tl.broadcast_to(r137_value, [$(256)])
    tl.store($(candidate_vpow_6_ptr) + r137_z, r137_value)
    r138_z = tl.arange($(256))
    r138_value = (tl.load($(candidate_vsum_2_ptr) + ((((r138_z / $(128)) % $(2)) * $(128)) + (r138_z % $(128))), dtype=tl.float32) * -0.5)
    r138_value = r138_value.to(tl.float32)
    r138_value = tl.broadcast_to(r138_value, [$(256)])
    tl.store($(candidate_vmul_34_ptr) + r138_z, r138_value)
    r139_z = tl.arange($(256))
    r139_value = (tl.load($(candidate_vmul_34_ptr) + ((((r139_z / $(128)) % $(2)) * $(128)) + (r139_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vpow_6_ptr) + ((((r139_z / $(128)) % $(2)) * $(128)) + (r139_z % $(128))), dtype=tl.float32))
    r139_value = r139_value.to(tl.float32)
    r139_value = tl.broadcast_to(r139_value, [$(256)])
    tl.store($(candidate_vmul_35_ptr) + r139_z, r139_value)
    r140_z = tl.arange($(65536))
    r140_value = (tl.load($(candidate_vmul_35_ptr) + ((((r140_z / $(32768)) % $(2)) * $(128)) + ((r140_z / $(256)) % $(128))), dtype=tl.float32) / 256.0)
    r140_value = r140_value.to(tl.float32)
    r140_value = tl.broadcast_to(r140_value, [$(65536)])
    tl.store($(candidate_vdiv_3_ptr) + r140_z, r140_value)
    r141_z = tl.arange($(65536))
    r141_value = ((tl.load($(candidate_vto_copy_22_ptr) + ((((r141_z / $(32768)) % $(2)) * $(32768)) + (((r141_z / $(256)) % $(128)) * $(256)) + (r141_z % $(256))), dtype=tl.float32)))
    r141_value = r141_value.to(tl.float32)
    r141_value = tl.broadcast_to(r141_value, [$(65536)])
    tl.store($(candidate_vpow_7_ptr) + r141_z, r141_value)
    r142_z = tl.arange($(65536))
    r142_value = (tl.load($(candidate_vpow_7_ptr) + ((((r142_z / $(32768)) % $(2)) * $(32768)) + (((r142_z / $(256)) % $(128)) * $(256)) + (r142_z % $(256))), dtype=tl.float32) * 2.0)
    r142_value = r142_value.to(tl.float32)
    r142_value = tl.broadcast_to(r142_value, [$(65536)])
    tl.store($(candidate_vmul_36_ptr) + r142_z, r142_value)
    r143_z = tl.arange($(65536))
    r143_value = (tl.load($(candidate_vdiv_3_ptr) + ((((r143_z / $(32768)) % $(2)) * $(32768)) + (((r143_z / $(256)) % $(128)) * $(256)) + (r143_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vmul_36_ptr) + ((((r143_z / $(32768)) % $(2)) * $(32768)) + (((r143_z / $(256)) % $(128)) * $(256)) + (r143_z % $(256))), dtype=tl.float32))
    r143_value = r143_value.to(tl.float32)
    r143_value = tl.broadcast_to(r143_value, [$(65536)])
    tl.store($(candidate_vmul_37_ptr) + r143_z, r143_value)
    r144_z = tl.arange($(65536))
    r144_value = (tl.load($(candidate_vmul_33_ptr) + ((((r144_z / $(32768)) % $(2)) * $(32768)) + (((r144_z / $(256)) % $(128)) * $(256)) + (r144_z % $(256))), dtype=tl.float32) + tl.load($(candidate_vmul_37_ptr) + ((((r144_z / $(32768)) % $(2)) * $(32768)) + (((r144_z / $(256)) % $(128)) * $(256)) + (r144_z % $(256))), dtype=tl.float32))
    r144_value = r144_value.to(tl.float32)
    r144_value = tl.broadcast_to(r144_value, [$(65536)])
    tl.store($(candidate_vadd_16_ptr) + r144_z, r144_value)
    r145_z = tl.arange($(65536))
    r145_value = tl.load($(candidate_vadd_16_ptr) + ((((r145_z / $(32768)) % $(2)) * $(32768)) + (((r145_z / $(256)) % $(128)) * $(256)) + (r145_z % $(256))), dtype=tl.float32)
    r145_value = r145_value.to(tl.bfloat16)
    r145_value = tl.broadcast_to(r145_value, [$(65536)])
    tl.store($(candidate_vto_copy_41_ptr) + r145_z, r145_value)
    r146_z = tl.arange($(81920))
    r146_value = tl.where((r146_z % $(320)) < $(256), tl.load($(candidate_vto_copy_41_ptr) + ((((r146_z / $(40960)) % $(2)) * $(32768)) + (((r146_z / $(320)) % $(128)) * $(256)) + (((r146_z % $(320)) - $(0)) % $(256))), dtype=tl.bfloat16), tl.load($(candidate_vadd_13_ptr) + ((((((((r146_z / $(40960)) % $(2)) * $(8192)) + (((r146_z / $(320)) % $(128)) * $(64)) + ((r146_z % $(320)) - $(256))) / $(64)) % $(128)) * $(128)) + (((((((r146_z / $(40960)) % $(2)) * $(8192)) + (((r146_z / $(320)) % $(128)) * $(64)) + ((r146_z % $(320)) - $(256))) / $(8192)) % $(2)) * $(64)) + (((((r146_z / $(40960)) % $(2)) * $(8192)) + (((r146_z / $(320)) % $(128)) * $(64)) + ((r146_z % $(320)) - $(256))) % $(64))), dtype=tl.bfloat16))
    r146_value = r146_value.to(tl.bfloat16)
    r146_value = tl.broadcast_to(r146_value, [$(81920)])
    tl.store($(candidate_vcat_6_ptr) + r146_z, r146_value)
    r147_row = tl.arange($(256))
    r147_col = tl.arange($(512))
    r147_lane = tl.arange($(320))
    r147_left = tl.load($(candidate_vcat_6_ptr) + ((((((r147_row[:, None] * $(320)) + r147_lane[None, :]) / $(40960)) % $(2)) * $(40960)) + (((((r147_row[:, None] * $(320)) + r147_lane[None, :]) / $(320)) % $(128)) * $(320)) + (((r147_row[:, None] * $(320)) + r147_lane[None, :]) % $(320))), dtype=tl.bfloat16)
    r147_right = tl.load($(candidate_vto_copy_3_ptr) + (((r147_lane[:, None] % $(320)) * $(512)) + (r147_col[None, :] % $(512))), dtype=tl.bfloat16)
    r147_value = tl.dot(r147_left, r147_right)
    r147_value = r147_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_5_ptr) + ((r147_row[:, None] * $(512)) + r147_col[None, :]), r147_value)
    r148_row = tl.arange($(256))
    r148_col = tl.arange($(256))
    r148_lane = tl.arange($(1536))
    r148_left = tl.load($(candidate_vcat_5_ptr) + ((((((r148_row[:, None] * $(1536)) + r148_lane[None, :]) / $(196608)) % $(2)) * $(196608)) + (((((((((r148_row[:, None] * $(1536)) + r148_lane[None, :]) / $(196608)) % $(2)) * $(196608)) + (((((r148_row[:, None] * $(1536)) + r148_lane[None, :]) / $(1536)) % $(128)) * $(1536)) + (((r148_row[:, None] * $(1536)) + r148_lane[None, :]) % $(1536))) / $(192)) % $(8)) * $(24576)) + (((((r148_row[:, None] * $(1536)) + r148_lane[None, :]) / $(1536)) % $(128)) * $(192)) + (((((((r148_row[:, None] * $(1536)) + r148_lane[None, :]) / $(196608)) % $(2)) * $(196608)) + (((((r148_row[:, None] * $(1536)) + r148_lane[None, :]) / $(1536)) % $(128)) * $(1536)) + (((r148_row[:, None] * $(1536)) + r148_lane[None, :]) % $(1536))) % $(192))), dtype=tl.bfloat16)
    r148_right = tl.load($(candidate_vto_copy_2_ptr) + (((r148_lane[:, None] % $(1536)) * $(256)) + (r148_col[None, :] % $(256))), dtype=tl.bfloat16)
    r148_value = tl.dot(r148_left, r148_right)
    r148_value = r148_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_6_ptr) + ((r148_row[:, None] * $(256)) + r148_col[None, :]), r148_value)
    r149_z = tl.arange($(65536))
    r149_value = tl.load($(candidate_vmm_6_ptr) + ((((r149_z / $(32768)) % $(2)) * $(32768)) + (((r149_z / $(256)) % $(128)) * $(256)) + (r149_z % $(256))), dtype=tl.bfloat16)
    r149_value = r149_value.to(tl.float32)
    r149_value = tl.broadcast_to(r149_value, [$(65536)])
    tl.store($(candidate_vto_copy_42_ptr) + r149_z, r149_value)
    r150_z = tl.arange($(65536))
    r150_value = (tl.load($(candidate_vto_copy_42_ptr) + ((((r150_z / $(32768)) % $(2)) * $(32768)) + (((r150_z / $(256)) % $(128)) * $(256)) + (r150_z % $(256))), dtype=tl.float32) * tl.cast(tl.load($(candidate_vto_copy_1_ptr) + (r150_z % $(256)), dtype=tl.bfloat16), tl.float32))
    r150_value = r150_value.to(tl.float32)
    r150_value = tl.broadcast_to(r150_value, [$(65536)])
    tl.store($(candidate_vmul_38_ptr) + r150_z, r150_value)
    r151_z = tl.arange($(65536))
    r151_value = (tl.load($(candidate_vmul_38_ptr) + ((((r151_z / $(32768)) % $(2)) * $(32768)) + (((r151_z / $(256)) % $(128)) * $(256)) + (r151_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vto_copy_20_ptr) + ((((r151_z / $(32768)) % $(2)) * $(32768)) + (((r151_z / $(256)) % $(128)) * $(256)) + (r151_z % $(256))), dtype=tl.float32))
    r151_value = r151_value.to(tl.float32)
    r151_value = tl.broadcast_to(r151_value, [$(65536)])
    tl.store($(candidate_vmul_39_ptr) + r151_z, r151_value)
    r152_z = tl.arange($(65536))
    r152_value = (tl.load($(candidate_vmul_38_ptr) + ((((r152_z / $(32768)) % $(2)) * $(32768)) + (((r152_z / $(256)) % $(128)) * $(256)) + (r152_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_ptr) + ((((r152_z / $(32768)) % $(2)) * $(128)) + ((r152_z / $(256)) % $(128))), dtype=tl.float32))
    r152_value = r152_value.to(tl.float32)
    r152_value = tl.broadcast_to(r152_value, [$(65536)])
    tl.store($(candidate_vmul_40_ptr) + r152_z, r152_value)
    r153_z = tl.arange($(256))
    r153_lane = tl.arange($(256))
    r153_value = tl.sum(tl.load($(candidate_vmul_39_ptr) + ((((r153_z[:, None] / $(128)) % $(2)) * $(32768)) + ((r153_z[:, None] % $(128)) * $(256)) + r153_lane[None, :]), dtype=tl.float32), axis=1)
    r153_value = r153_value.to(tl.float32)
    r153_value = tl.broadcast_to(r153_value, [$(256)])
    tl.store($(candidate_vsum_3_ptr) + r153_z, r153_value)
    r154_z = tl.arange($(256))
    r154_value = ((tl.load($(candidate_vrsqrt_ptr) + ((((r154_z / $(128)) % $(2)) * $(128)) + (r154_z % $(128))), dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_ptr) + ((((r154_z / $(128)) % $(2)) * $(128)) + (r154_z % $(128))), dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_ptr) + ((((r154_z / $(128)) % $(2)) * $(128)) + (r154_z % $(128))), dtype=tl.float32)))
    r154_value = r154_value.to(tl.float32)
    r154_value = tl.broadcast_to(r154_value, [$(256)])
    tl.store($(candidate_vpow_8_ptr) + r154_z, r154_value)
    r155_z = tl.arange($(256))
    r155_value = (tl.load($(candidate_vsum_3_ptr) + ((((r155_z / $(128)) % $(2)) * $(128)) + (r155_z % $(128))), dtype=tl.float32) * -0.5)
    r155_value = r155_value.to(tl.float32)
    r155_value = tl.broadcast_to(r155_value, [$(256)])
    tl.store($(candidate_vmul_41_ptr) + r155_z, r155_value)
    r156_z = tl.arange($(256))
    r156_value = (tl.load($(candidate_vmul_41_ptr) + ((((r156_z / $(128)) % $(2)) * $(128)) + (r156_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vpow_8_ptr) + ((((r156_z / $(128)) % $(2)) * $(128)) + (r156_z % $(128))), dtype=tl.float32))
    r156_value = r156_value.to(tl.float32)
    r156_value = tl.broadcast_to(r156_value, [$(256)])
    tl.store($(candidate_vmul_42_ptr) + r156_z, r156_value)
    r157_z = tl.arange($(65536))
    r157_value = (tl.load($(candidate_vmul_42_ptr) + ((((r157_z / $(32768)) % $(2)) * $(128)) + ((r157_z / $(256)) % $(128))), dtype=tl.float32) / 256.0)
    r157_value = r157_value.to(tl.float32)
    r157_value = tl.broadcast_to(r157_value, [$(65536)])
    tl.store($(candidate_vdiv_4_ptr) + r157_z, r157_value)
    r158_z = tl.arange($(65536))
    r158_value = ((tl.load($(candidate_vto_copy_20_ptr) + ((((r158_z / $(32768)) % $(2)) * $(32768)) + (((r158_z / $(256)) % $(128)) * $(256)) + (r158_z % $(256))), dtype=tl.float32)))
    r158_value = r158_value.to(tl.float32)
    r158_value = tl.broadcast_to(r158_value, [$(65536)])
    tl.store($(candidate_vpow_9_ptr) + r158_z, r158_value)
    r159_z = tl.arange($(65536))
    r159_value = (tl.load($(candidate_vpow_9_ptr) + ((((r159_z / $(32768)) % $(2)) * $(32768)) + (((r159_z / $(256)) % $(128)) * $(256)) + (r159_z % $(256))), dtype=tl.float32) * 2.0)
    r159_value = r159_value.to(tl.float32)
    r159_value = tl.broadcast_to(r159_value, [$(65536)])
    tl.store($(candidate_vmul_43_ptr) + r159_z, r159_value)
    r160_z = tl.arange($(65536))
    r160_value = (tl.load($(candidate_vdiv_4_ptr) + ((((r160_z / $(32768)) % $(2)) * $(32768)) + (((r160_z / $(256)) % $(128)) * $(256)) + (r160_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vmul_43_ptr) + ((((r160_z / $(32768)) % $(2)) * $(32768)) + (((r160_z / $(256)) % $(128)) * $(256)) + (r160_z % $(256))), dtype=tl.float32))
    r160_value = r160_value.to(tl.float32)
    r160_value = tl.broadcast_to(r160_value, [$(65536)])
    tl.store($(candidate_vmul_44_ptr) + r160_z, r160_value)
    r161_z = tl.arange($(65536))
    r161_value = (tl.load($(candidate_vmul_40_ptr) + ((((r161_z / $(32768)) % $(2)) * $(32768)) + (((r161_z / $(256)) % $(128)) * $(256)) + (r161_z % $(256))), dtype=tl.float32) + tl.load($(candidate_vmul_44_ptr) + ((((r161_z / $(32768)) % $(2)) * $(32768)) + (((r161_z / $(256)) % $(128)) * $(256)) + (r161_z % $(256))), dtype=tl.float32))
    r161_value = r161_value.to(tl.float32)
    r161_value = tl.broadcast_to(r161_value, [$(65536)])
    tl.store($(candidate_vadd_17_ptr) + r161_z, r161_value)
    r162_z = tl.arange($(65536))
    r162_value = tl.load($(candidate_vadd_17_ptr) + ((((r162_z / $(32768)) % $(2)) * $(32768)) + (((r162_z / $(256)) % $(128)) * $(256)) + (r162_z % $(256))), dtype=tl.float32)
    r162_value = r162_value.to(tl.bfloat16)
    r162_value = tl.broadcast_to(r162_value, [$(65536)])
    tl.store($(candidate_vto_copy_43_ptr) + r162_z, r162_value)
    r163_row = tl.arange($(256))
    r163_col = tl.arange($(512))
    r163_lane = tl.arange($(256))
    r163_left = tl.load($(candidate_vto_copy_43_ptr) + ((((((r163_row[:, None] * $(256)) + r163_lane[None, :]) / $(32768)) % $(2)) * $(32768)) + (((((r163_row[:, None] * $(256)) + r163_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r163_row[:, None] * $(256)) + r163_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r163_right = tl.load($(candidate_vto_copy_ptr) + (((r163_lane[:, None] % $(256)) * $(512)) + (r163_col[None, :] % $(512))), dtype=tl.bfloat16)
    r163_value = tl.dot(r163_left, r163_right)
    r163_value = r163_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_7_ptr) + ((r163_row[:, None] * $(512)) + r163_col[None, :]), r163_value)
    r164_z = tl.arange($(131072))
    r164_value = (tl.load($(candidate_vmm_5_ptr) + ((((r164_z / $(65536)) % $(2)) * $(65536)) + (((r164_z / $(512)) % $(128)) * $(512)) + (r164_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vmm_7_ptr) + ((((r164_z / $(65536)) % $(2)) * $(65536)) + (((r164_z / $(512)) % $(128)) * $(512)) + (r164_z % $(512))), dtype=tl.bfloat16))
    r164_value = r164_value.to(tl.bfloat16)
    r164_value = tl.broadcast_to(r164_value, [$(131072)])
    tl.store($(candidate_vadd_18_ptr) + r164_z, r164_value)
    z = tl.arange($(131072))
    value = tl.load($(candidate_vadd_18_ptr) + ((((z / $(65536)) % $(2)) * $(65536)) + (((z / $(512)) % $(128)) * $(512)) + (z % $(512))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(131072)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_mla_fa4_te_rope_nonabsorb_dx
