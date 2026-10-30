import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_mla_fa4_te_rope_thd_nonabsorb_dx
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
  let vcat_ptr : RegionName := "vcat"
  let vadd_2_ptr : RegionName := "vadd_2"
  let vto_copy_14_ptr : RegionName := "vto_copy_14"
  let vdiv_ptr : RegionName := "vdiv"
  let vpow_1_ptr : RegionName := "vpow_1"
  let vreciprocal_ptr : RegionName := "vreciprocal"
  let vmul_7_ptr : RegionName := "vmul_7"
  let vto_copy_16_ptr : RegionName := "vto_copy_16"
  let vbmm_ptr : RegionName := "vbmm"
  let vcat_1_ptr : RegionName := "vcat_1"
  let vcos_ptr : RegionName := "vcos"
  let vmul_9_ptr : RegionName := "vmul_9"
  let vsin_ptr : RegionName := "vsin"
  let vmul_10_ptr : RegionName := "vmul_10"
  let vto_copy_17_ptr : RegionName := "vto_copy_17"
  let vto_copy_18_ptr : RegionName := "vto_copy_18"
  let vsub_ptr : RegionName := "vsub"
  let vcat_3_ptr : RegionName := "vcat_3"
  let vsub_1_ptr : RegionName := "vsub_1"
  let vne_ptr : RegionName := "vne"
  let vcumsum_1_ptr : RegionName := "vcumsum_1"
  let veq_ptr : RegionName := "veq"
  let vtril_ptr : RegionName := "vtril"
  let vbitwise_and_ptr : RegionName := "vbitwise_and"
  let vwhere_ptr : RegionName := "vwhere"
  let vmm_ptr : RegionName := "vmm"
  let vto_copy_20_ptr : RegionName := "vto_copy_20"
  let vpow_3_ptr : RegionName := "vpow_3"
  let vmean_ptr : RegionName := "vmean"
  let vadd_3_ptr : RegionName := "vadd_3"
  let vrsqrt_ptr : RegionName := "vrsqrt"
  let vmul_13_ptr : RegionName := "vmul_13"
  let vmul_14_ptr : RegionName := "vmul_14"
  let vto_copy_21_ptr : RegionName := "vto_copy_21"
  let vmm_1_ptr : RegionName := "vmm_1"
  let vmm_2_ptr : RegionName := "vmm_2"
  let vto_copy_22_ptr : RegionName := "vto_copy_22"
  let vpow_4_ptr : RegionName := "vpow_4"
  let vmean_1_ptr : RegionName := "vmean_1"
  let vadd_4_ptr : RegionName := "vadd_4"
  let vrsqrt_1_ptr : RegionName := "vrsqrt_1"
  let vmul_15_ptr : RegionName := "vmul_15"
  let vmul_16_ptr : RegionName := "vmul_16"
  let vto_copy_23_ptr : RegionName := "vto_copy_23"
  let vmm_3_ptr : RegionName := "vmm_3"
  let vmul_17_ptr : RegionName := "vmul_17"
  let vmul_18_ptr : RegionName := "vmul_18"
  let vsub_4_ptr : RegionName := "vsub_4"
  let vmul_19_ptr : RegionName := "vmul_19"
  let vmul_20_ptr : RegionName := "vmul_20"
  let vadd_5_ptr : RegionName := "vadd_5"
  let vcat_5_ptr : RegionName := "vcat_5"
  let vmul_21_ptr : RegionName := "vmul_21"
  let vmul_22_ptr : RegionName := "vmul_22"
  let vsub_5_ptr : RegionName := "vsub_5"
  let vmul_23_ptr : RegionName := "vmul_23"
  let vmul_24_ptr : RegionName := "vmul_24"
  let vadd_6_ptr : RegionName := "vadd_6"
  let vcat_6_ptr : RegionName := "vcat_6"
  let vcat_7_ptr : RegionName := "vcat_7"
  let vcat_8_ptr : RegionName := "vcat_8"
  let vbmm_2_ptr : RegionName := "vbmm_2"
  let vmul_25_ptr : RegionName := "vmul_25"
  let vadd_7_ptr : RegionName := "vadd_7"
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
  let vadd_8_ptr : RegionName := "vadd_8"
  let vmul_30_ptr : RegionName := "vmul_30"
  let vadd_9_ptr : RegionName := "vadd_9"
  let vmul_31_ptr : RegionName := "vmul_31"
  let vmul_32_ptr : RegionName := "vmul_32"
  let vneg_1_ptr : RegionName := "vneg_1"
  let vmul_33_ptr : RegionName := "vmul_33"
  let vadd_10_ptr : RegionName := "vadd_10"
  let vmul_34_ptr : RegionName := "vmul_34"
  let vadd_11_ptr : RegionName := "vadd_11"
  let vslice_backward_ptr : RegionName := "vslice_backward"
  let vslice_backward_1_ptr : RegionName := "vslice_backward_1"
  let vadd_12_ptr : RegionName := "vadd_12"
  let vslice_backward_2_ptr : RegionName := "vslice_backward_2"
  let vslice_backward_3_ptr : RegionName := "vslice_backward_3"
  let vadd_13_ptr : RegionName := "vadd_13"
  let vcat_9_ptr : RegionName := "vcat_9"
  let vcat_10_ptr : RegionName := "vcat_10"
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
  let vadd_14_ptr : RegionName := "vadd_14"
  let vto_copy_29_ptr : RegionName := "vto_copy_29"
  let vcat_11_ptr : RegionName := "vcat_11"
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
  let vadd_15_ptr : RegionName := "vadd_15"
  let vto_copy_31_ptr : RegionName := "vto_copy_31"
  let vmm_9_ptr : RegionName := "vmm_9"
  let vadd_16_ptr : RegionName := "vadd_16"
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
    r16_z = tl.arange($(128))
    r16_value = tl.where(r16_z < $(48), tl.cast((r16_z - $(0)), tl.int64), tl.where(r16_z < $(64), tl.cast((r16_z - $(48)), tl.int64), tl.cast((r16_z - $(64)), tl.int64)))
    r16_value = tl.broadcast_to(r16_value, [$(128)])
    tl.store($(vcat_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(128))
    r17_value = (tl.load($(vcat_ptr) + r17_z, dtype=tl.int64) + tl.full([], $(0), dtype=tl.int64))
    r17_value = tl.broadcast_to(r17_value, [$(128)])
    tl.store($(vadd_2_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(32))
    r18_value = tl.toReal((tl.cast(r18_z, tl.int64) * tl.full([], $(2), dtype=tl.int64)))
    r18_value = r18_value.to(tl.float32)
    r18_value = tl.broadcast_to(r18_value, [$(32)])
    tl.store($(vto_copy_14_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(32))
    r19_value = (tl.load($(vto_copy_14_ptr) + r19_z, dtype=tl.float32) / 64.0)
    r19_value = r19_value.to(tl.float32)
    r19_value = tl.broadcast_to(r19_value, [$(32)])
    tl.store($(vdiv_ptr) + r19_z, r19_value)
    r20_z = tl.arange($(32))
    r20_value = tl.extra.cuda.libdevice.pow(10000.0, tl.load($(vdiv_ptr) + r20_z, dtype=tl.float32))
    r20_value = r20_value.to(tl.float32)
    r20_value = tl.broadcast_to(r20_value, [$(32)])
    tl.store($(vpow_1_ptr) + r20_z, r20_value)
    r21_z = tl.arange($(32))
    r21_value = 1.0 / tl.load($(vpow_1_ptr) + r21_z, dtype=tl.float32)
    r21_value = r21_value.to(tl.float32)
    r21_value = tl.broadcast_to(r21_value, [$(32)])
    tl.store($(vreciprocal_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(32))
    r22_value = (tl.load($(vreciprocal_ptr) + r22_z, dtype=tl.float32) * 1.0)
    r22_value = r22_value.to(tl.float32)
    r22_value = tl.broadcast_to(r22_value, [$(32)])
    tl.store($(vmul_7_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(128))
    r23_value = tl.toReal(tl.load($(vadd_2_ptr) + r23_z, dtype=tl.int64))
    r23_value = r23_value.to(tl.float32)
    r23_value = tl.broadcast_to(r23_value, [$(128)])
    tl.store($(vto_copy_16_ptr) + r23_z, r23_value)
    r24_row = tl.arange($(32))
    r24_col = tl.arange($(128))
    r24_lane = tl.arange($(1))
    for r24_group in range($(1)) {
      r24_left = tl.load($(vmul_7_ptr) + ((r24_row[:, None] + r24_lane[None, :]) % $(32)), dtype=tl.float32)
      r24_right = tl.load($(vto_copy_16_ptr) + (((r24_lane[:, None] * $(128)) + r24_col[None, :]) % $(128)), dtype=tl.float32)
      r24_value = tl.dot(r24_left, r24_right)
      r24_value = r24_value.to(tl.float32)
      tl.store($(vbmm_ptr) + ((r24_group * $(4096)) + (r24_row[:, None] * $(128)) + r24_col[None, :]), r24_value)
    }
    r25_z = tl.arange($(8192))
    r25_value = tl.where((r25_z % $(64)) < $(32), tl.load($(vbmm_ptr) + (((((r25_z % $(64)) - $(0)) % $(32)) * $(128)) + ((r25_z / $(64)) % $(128))), dtype=tl.float32), tl.load($(vbmm_ptr) + (((((r25_z % $(64)) - $(32)) % $(32)) * $(128)) + ((r25_z / $(64)) % $(128))), dtype=tl.float32))
    r25_value = r25_value.to(tl.float32)
    r25_value = tl.broadcast_to(r25_value, [$(8192)])
    tl.store($(vcat_1_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(8192))
    r26_value = tl.cos(tl.load($(vcat_1_ptr) + ((((r26_z / $(64)) % $(128)) * $(64)) + (r26_z % $(64))), dtype=tl.float32))
    r26_value = r26_value.to(tl.float32)
    r26_value = tl.broadcast_to(r26_value, [$(8192)])
    tl.store($(vcos_ptr) + r26_z, r26_value)
    r27_z = tl.arange($(8192))
    r27_value = (tl.load($(vcos_ptr) + ((((r27_z / $(64)) % $(128)) * $(64)) + (r27_z % $(64))), dtype=tl.float32) * 1.0)
    r27_value = r27_value.to(tl.float32)
    r27_value = tl.broadcast_to(r27_value, [$(8192)])
    tl.store($(vmul_9_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(8192))
    r28_value = tl.sin(tl.load($(vcat_1_ptr) + ((((r28_z / $(64)) % $(128)) * $(64)) + (r28_z % $(64))), dtype=tl.float32))
    r28_value = r28_value.to(tl.float32)
    r28_value = tl.broadcast_to(r28_value, [$(8192)])
    tl.store($(vsin_ptr) + r28_z, r28_value)
    r29_z = tl.arange($(8192))
    r29_value = (tl.load($(vsin_ptr) + ((((r29_z / $(64)) % $(128)) * $(64)) + (r29_z % $(64))), dtype=tl.float32) * 1.0)
    r29_value = r29_value.to(tl.float32)
    r29_value = tl.broadcast_to(r29_value, [$(8192)])
    tl.store($(vmul_10_ptr) + r29_z, r29_value)
    r30_z = tl.arange($(8192))
    r30_value = tl.load($(vmul_9_ptr) + ((((r30_z / $(64)) % $(128)) * $(64)) + (r30_z % $(64))), dtype=tl.float32)
    r30_value = r30_value.to(tl.bfloat16)
    r30_value = tl.broadcast_to(r30_value, [$(8192)])
    tl.store($(vto_copy_17_ptr) + r30_z, r30_value)
    r31_z = tl.arange($(8192))
    r31_value = tl.load($(vmul_10_ptr) + ((((r31_z / $(64)) % $(128)) * $(64)) + (r31_z % $(64))), dtype=tl.float32)
    r31_value = r31_value.to(tl.bfloat16)
    r31_value = tl.broadcast_to(r31_value, [$(8192)])
    tl.store($(vto_copy_18_ptr) + r31_z, r31_value)
    r32_z = tl.arange($(1))
    r32_value = (tl.load($(vadd_2_ptr) + $(0), dtype=tl.int64) - tl.full([], $(1), dtype=tl.int64))
    r32_value = tl.broadcast_to(r32_value, [$(1)])
    tl.store($(vsub_ptr) + r32_z, r32_value)
    r33_z = tl.arange($(129))
    r33_value = tl.where(r33_z < $(1), tl.load($(vsub_ptr) + $(0), dtype=tl.int64), tl.load($(vadd_2_ptr) + ((r33_z - $(1)) % $(128)), dtype=tl.int64))
    r33_value = tl.broadcast_to(r33_value, [$(129)])
    tl.store($(vcat_3_ptr) + r33_z, r33_value)
    r34_z = tl.arange($(128))
    r34_value = (tl.load($(vcat_3_ptr) + (r34_z + $(1)), dtype=tl.int64) - tl.load($(vcat_3_ptr) + r34_z, dtype=tl.int64))
    r34_value = tl.broadcast_to(r34_value, [$(128)])
    tl.store($(vsub_1_ptr) + r34_z, r34_value)
    r35_z = tl.arange($(128))
    r35_value = (tl.load($(vsub_1_ptr) + r35_z, dtype=tl.int64) != tl.full([], $(1), dtype=tl.int64))
    r35_value = tl.broadcast_to(r35_value, [$(128)])
    tl.store($(vne_ptr) + r35_z, r35_value)
    r36_z = tl.arange($(128))
    r36_value = tl.broadcast_to(tl.full([], $(0), dtype=tl.int64), [$(128)])
    for r36_lane in range($(128)) {
      r36_value = tl.where(r36_lane <= r36_z, (r36_value + tl.where(tl.load($(vne_ptr) + (r36_lane % $(128)), dtype=tl.int1), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))), r36_value)
    }
    r36_value = tl.broadcast_to(r36_value, [$(128)])
    tl.store($(vcumsum_1_ptr) + r36_z, r36_value)
    r37_z = tl.arange($(16384))
    r37_value = (tl.load($(vcumsum_1_ptr) + ((r37_z / $(128)) % $(128)), dtype=tl.int64) == tl.load($(vcumsum_1_ptr) + (r37_z % $(128)), dtype=tl.int64))
    r37_value = tl.broadcast_to(r37_value, [$(16384)])
    tl.store($(veq_ptr) + r37_z, r37_value)
    r38_z = tl.arange($(16384))
    r38_value = tl.where((r38_z % $(128)) <= ((r38_z / $(128)) % $(128)) + $(0), true, false)
    r38_value = tl.broadcast_to(r38_value, [$(16384)])
    tl.store($(vtril_ptr) + r38_z, r38_value)
    r39_z = tl.arange($(16384))
    r39_value = (tl.load($(veq_ptr) + ((((r39_z / $(128)) % $(128)) * $(128)) + (r39_z % $(128))), dtype=tl.int1) & tl.load($(vtril_ptr) + ((((r39_z / $(128)) % $(128)) * $(128)) + (r39_z % $(128))), dtype=tl.int1))
    r39_value = tl.broadcast_to(r39_value, [$(16384)])
    tl.store($(vbitwise_and_ptr) + r39_z, r39_value)
    r40_z = tl.arange($(16384))
    r40_value = tl.where(tl.load($(vbitwise_and_ptr) + ((((r40_z / $(128)) % $(128)) * $(128)) + (r40_z % $(128))), dtype=tl.int1), tl.cast(0.0, tl.bfloat16), tl.cast(-338953138925153547590470800371487866880.0, tl.bfloat16))
    r40_value = r40_value.to(tl.bfloat16)
    r40_value = tl.broadcast_to(r40_value, [$(16384)])
    tl.store($(vwhere_ptr) + r40_z, r40_value)
    r41_row = tl.arange($(128))
    r41_col = tl.arange($(256))
    r41_lane = tl.arange($(512))
    r41_left = tl.load($(input0_ptr) + ((((((r41_row[:, None] * $(512)) + r41_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r41_row[:, None] * $(512)) + r41_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r41_right = tl.load($(vto_copy_ptr) + (((r41_col[None, :] % $(256)) * $(512)) + (r41_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r41_value = tl.dot(r41_left, r41_right)
    r41_value = r41_value.to(tl.bfloat16)
    tl.store($(vmm_ptr) + ((r41_row[:, None] * $(256)) + r41_col[None, :]), r41_value)
    r42_z = tl.arange($(32768))
    r42_value = tl.load($(vmm_ptr) + ((((r42_z / $(256)) % $(128)) * $(256)) + (r42_z % $(256))), dtype=tl.bfloat16)
    r42_value = r42_value.to(tl.float32)
    r42_value = tl.broadcast_to(r42_value, [$(32768)])
    tl.store($(vto_copy_20_ptr) + r42_z, r42_value)
    r43_z = tl.arange($(32768))
    r43_value = ((tl.load($(vto_copy_20_ptr) + ((((r43_z / $(256)) % $(128)) * $(256)) + (r43_z % $(256))), dtype=tl.float32)) * (tl.load($(vto_copy_20_ptr) + ((((r43_z / $(256)) % $(128)) * $(256)) + (r43_z % $(256))), dtype=tl.float32)))
    r43_value = r43_value.to(tl.float32)
    r43_value = tl.broadcast_to(r43_value, [$(32768)])
    tl.store($(vpow_3_ptr) + r43_z, r43_value)
    r44_z = tl.arange($(128))
    r44_lane = tl.arange($(256))
    r44_value = (tl.sum(tl.load($(vpow_3_ptr) + ((r44_z[:, None] * $(256)) + r44_lane[None, :]), dtype=tl.float32), axis=1) / 256.0)
    r44_value = r44_value.to(tl.float32)
    r44_value = tl.broadcast_to(r44_value, [$(128)])
    tl.store($(vmean_ptr) + r44_z, r44_value)
    r45_z = tl.arange($(128))
    r45_value = (tl.load($(vmean_ptr) + r45_z, dtype=tl.float32) + 0.000001)
    r45_value = r45_value.to(tl.float32)
    r45_value = tl.broadcast_to(r45_value, [$(128)])
    tl.store($(vadd_3_ptr) + r45_z, r45_value)
    r46_z = tl.arange($(128))
    r46_value = tl.rsqrt(tl.load($(vadd_3_ptr) + r46_z, dtype=tl.float32))
    r46_value = r46_value.to(tl.float32)
    r46_value = tl.broadcast_to(r46_value, [$(128)])
    tl.store($(vrsqrt_ptr) + r46_z, r46_value)
    r47_z = tl.arange($(32768))
    r47_value = (tl.load($(vto_copy_20_ptr) + ((((r47_z / $(256)) % $(128)) * $(256)) + (r47_z % $(256))), dtype=tl.float32) * tl.load($(vrsqrt_ptr) + ((r47_z / $(256)) % $(128)), dtype=tl.float32))
    r47_value = r47_value.to(tl.float32)
    r47_value = tl.broadcast_to(r47_value, [$(32768)])
    tl.store($(vmul_13_ptr) + r47_z, r47_value)
    r48_z = tl.arange($(32768))
    r48_value = (tl.cast(tl.load($(vto_copy_1_ptr) + (r48_z % $(256)), dtype=tl.bfloat16), tl.float32) * tl.load($(vmul_13_ptr) + ((((r48_z / $(256)) % $(128)) * $(256)) + (r48_z % $(256))), dtype=tl.float32))
    r48_value = r48_value.to(tl.float32)
    r48_value = tl.broadcast_to(r48_value, [$(32768)])
    tl.store($(vmul_14_ptr) + r48_z, r48_value)
    r49_z = tl.arange($(32768))
    r49_value = tl.load($(vmul_14_ptr) + ((((r49_z / $(256)) % $(128)) * $(256)) + (r49_z % $(256))), dtype=tl.float32)
    r49_value = r49_value.to(tl.bfloat16)
    r49_value = tl.broadcast_to(r49_value, [$(32768)])
    tl.store($(vto_copy_21_ptr) + r49_z, r49_value)
    r50_row = tl.arange($(128))
    r50_col = tl.arange($(1536))
    r50_lane = tl.arange($(256))
    r50_left = tl.load($(vto_copy_21_ptr) + ((((((r50_row[:, None] * $(256)) + r50_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r50_row[:, None] * $(256)) + r50_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r50_right = tl.load($(vto_copy_2_ptr) + (((r50_col[None, :] % $(1536)) * $(256)) + (r50_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r50_value = tl.dot(r50_left, r50_right)
    r50_value = r50_value.to(tl.bfloat16)
    tl.store($(vmm_1_ptr) + ((r50_row[:, None] * $(1536)) + r50_col[None, :]), r50_value)
    r51_row = tl.arange($(128))
    r51_col = tl.arange($(320))
    r51_lane = tl.arange($(512))
    r51_left = tl.load($(input0_ptr) + ((((((r51_row[:, None] * $(512)) + r51_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r51_row[:, None] * $(512)) + r51_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r51_right = tl.load($(vto_copy_3_ptr) + (((r51_col[None, :] % $(320)) * $(512)) + (r51_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r51_value = tl.dot(r51_left, r51_right)
    r51_value = r51_value.to(tl.bfloat16)
    tl.store($(vmm_2_ptr) + ((r51_row[:, None] * $(320)) + r51_col[None, :]), r51_value)
    r52_z = tl.arange($(32768))
    r52_value = tl.load($(vmm_2_ptr) + ((((r52_z / $(256)) % $(128)) * $(320)) + (r52_z % $(256))), dtype=tl.bfloat16)
    r52_value = r52_value.to(tl.float32)
    r52_value = tl.broadcast_to(r52_value, [$(32768)])
    tl.store($(vto_copy_22_ptr) + r52_z, r52_value)
    r53_z = tl.arange($(32768))
    r53_value = ((tl.load($(vto_copy_22_ptr) + ((((r53_z / $(256)) % $(128)) * $(256)) + (r53_z % $(256))), dtype=tl.float32)) * (tl.load($(vto_copy_22_ptr) + ((((r53_z / $(256)) % $(128)) * $(256)) + (r53_z % $(256))), dtype=tl.float32)))
    r53_value = r53_value.to(tl.float32)
    r53_value = tl.broadcast_to(r53_value, [$(32768)])
    tl.store($(vpow_4_ptr) + r53_z, r53_value)
    r54_z = tl.arange($(128))
    r54_lane = tl.arange($(256))
    r54_value = (tl.sum(tl.load($(vpow_4_ptr) + ((r54_z[:, None] * $(256)) + r54_lane[None, :]), dtype=tl.float32), axis=1) / 256.0)
    r54_value = r54_value.to(tl.float32)
    r54_value = tl.broadcast_to(r54_value, [$(128)])
    tl.store($(vmean_1_ptr) + r54_z, r54_value)
    r55_z = tl.arange($(128))
    r55_value = (tl.load($(vmean_1_ptr) + r55_z, dtype=tl.float32) + 0.000001)
    r55_value = r55_value.to(tl.float32)
    r55_value = tl.broadcast_to(r55_value, [$(128)])
    tl.store($(vadd_4_ptr) + r55_z, r55_value)
    r56_z = tl.arange($(128))
    r56_value = tl.rsqrt(tl.load($(vadd_4_ptr) + r56_z, dtype=tl.float32))
    r56_value = r56_value.to(tl.float32)
    r56_value = tl.broadcast_to(r56_value, [$(128)])
    tl.store($(vrsqrt_1_ptr) + r56_z, r56_value)
    r57_z = tl.arange($(32768))
    r57_value = (tl.load($(vto_copy_22_ptr) + ((((r57_z / $(256)) % $(128)) * $(256)) + (r57_z % $(256))), dtype=tl.float32) * tl.load($(vrsqrt_1_ptr) + ((r57_z / $(256)) % $(128)), dtype=tl.float32))
    r57_value = r57_value.to(tl.float32)
    r57_value = tl.broadcast_to(r57_value, [$(32768)])
    tl.store($(vmul_15_ptr) + r57_z, r57_value)
    r58_z = tl.arange($(32768))
    r58_value = (tl.cast(tl.load($(vto_copy_4_ptr) + (r58_z % $(256)), dtype=tl.bfloat16), tl.float32) * tl.load($(vmul_15_ptr) + ((((r58_z / $(256)) % $(128)) * $(256)) + (r58_z % $(256))), dtype=tl.float32))
    r58_value = r58_value.to(tl.float32)
    r58_value = tl.broadcast_to(r58_value, [$(32768)])
    tl.store($(vmul_16_ptr) + r58_z, r58_value)
    r59_z = tl.arange($(32768))
    r59_value = tl.load($(vmul_16_ptr) + ((((r59_z / $(256)) % $(128)) * $(256)) + (r59_z % $(256))), dtype=tl.float32)
    r59_value = r59_value.to(tl.bfloat16)
    r59_value = tl.broadcast_to(r59_value, [$(32768)])
    tl.store($(vto_copy_23_ptr) + r59_z, r59_value)
    r60_row = tl.arange($(128))
    r60_col = tl.arange($(2048))
    r60_lane = tl.arange($(256))
    r60_left = tl.load($(vto_copy_23_ptr) + ((((((r60_row[:, None] * $(256)) + r60_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r60_row[:, None] * $(256)) + r60_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r60_right = tl.load($(vto_copy_5_ptr) + (((r60_col[None, :] % $(2048)) * $(256)) + (r60_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r60_value = tl.dot(r60_left, r60_right)
    r60_value = r60_value.to(tl.bfloat16)
    tl.store($(vmm_3_ptr) + ((r60_row[:, None] * $(2048)) + r60_col[None, :]), r60_value)
    r61_z = tl.arange($(32768))
    r61_value = (tl.load($(vmm_1_ptr) + ((((r61_z / $(32)) % $(128)) * $(1536)) + (((r61_z / $(4096)) % $(8)) * $(192)) + ((r61_z % $(32)) * $(2)) + $(128)), dtype=tl.bfloat16) * tl.load($(vto_copy_17_ptr) + ((((r61_z / $(32)) % $(128)) * $(64)) + (r61_z % $(32))), dtype=tl.bfloat16))
    r61_value = r61_value.to(tl.bfloat16)
    r61_value = tl.broadcast_to(r61_value, [$(32768)])
    tl.store($(vmul_17_ptr) + r61_z, r61_value)
    r62_z = tl.arange($(32768))
    r62_value = (tl.load($(vmm_1_ptr) + ((((r62_z / $(32)) % $(128)) * $(1536)) + (((r62_z / $(4096)) % $(8)) * $(192)) + ((r62_z % $(32)) * $(2)) + $(129)), dtype=tl.bfloat16) * tl.load($(vto_copy_18_ptr) + ((((r62_z / $(32)) % $(128)) * $(64)) + (r62_z % $(32))), dtype=tl.bfloat16))
    r62_value = r62_value.to(tl.bfloat16)
    r62_value = tl.broadcast_to(r62_value, [$(32768)])
    tl.store($(vmul_18_ptr) + r62_z, r62_value)
    r63_z = tl.arange($(32768))
    r63_value = (tl.load($(vmul_17_ptr) + ((((r63_z / $(4096)) % $(8)) * $(4096)) + (((r63_z / $(32)) % $(128)) * $(32)) + (r63_z % $(32))), dtype=tl.bfloat16) - tl.load($(vmul_18_ptr) + ((((r63_z / $(4096)) % $(8)) * $(4096)) + (((r63_z / $(32)) % $(128)) * $(32)) + (r63_z % $(32))), dtype=tl.bfloat16))
    r63_value = r63_value.to(tl.bfloat16)
    r63_value = tl.broadcast_to(r63_value, [$(32768)])
    tl.store($(vsub_4_ptr) + r63_z, r63_value)
    r64_z = tl.arange($(32768))
    r64_value = (tl.load($(vmm_1_ptr) + ((((r64_z / $(32)) % $(128)) * $(1536)) + (((r64_z / $(4096)) % $(8)) * $(192)) + ((r64_z % $(32)) * $(2)) + $(129)), dtype=tl.bfloat16) * tl.load($(vto_copy_17_ptr) + ((((r64_z / $(32)) % $(128)) * $(64)) + (r64_z % $(32))), dtype=tl.bfloat16))
    r64_value = r64_value.to(tl.bfloat16)
    r64_value = tl.broadcast_to(r64_value, [$(32768)])
    tl.store($(vmul_19_ptr) + r64_z, r64_value)
    r65_z = tl.arange($(32768))
    r65_value = (tl.load($(vmm_1_ptr) + ((((r65_z / $(32)) % $(128)) * $(1536)) + (((r65_z / $(4096)) % $(8)) * $(192)) + ((r65_z % $(32)) * $(2)) + $(128)), dtype=tl.bfloat16) * tl.load($(vto_copy_18_ptr) + ((((r65_z / $(32)) % $(128)) * $(64)) + (r65_z % $(32))), dtype=tl.bfloat16))
    r65_value = r65_value.to(tl.bfloat16)
    r65_value = tl.broadcast_to(r65_value, [$(32768)])
    tl.store($(vmul_20_ptr) + r65_z, r65_value)
    r66_z = tl.arange($(32768))
    r66_value = (tl.load($(vmul_19_ptr) + ((((r66_z / $(4096)) % $(8)) * $(4096)) + (((r66_z / $(32)) % $(128)) * $(32)) + (r66_z % $(32))), dtype=tl.bfloat16) + tl.load($(vmul_20_ptr) + ((((r66_z / $(4096)) % $(8)) * $(4096)) + (((r66_z / $(32)) % $(128)) * $(32)) + (r66_z % $(32))), dtype=tl.bfloat16))
    r66_value = r66_value.to(tl.bfloat16)
    r66_value = tl.broadcast_to(r66_value, [$(32768)])
    tl.store($(vadd_5_ptr) + r66_z, r66_value)
    r67_z = tl.arange($(65536))
    r67_value = tl.where((r67_z % $(64)) < $(32), tl.load($(vsub_4_ptr) + ((((r67_z / $(8192)) % $(8)) * $(4096)) + (((r67_z / $(64)) % $(128)) * $(32)) + (((r67_z % $(64)) - $(0)) % $(32))), dtype=tl.bfloat16), tl.load($(vadd_5_ptr) + ((((r67_z / $(8192)) % $(8)) * $(4096)) + (((r67_z / $(64)) % $(128)) * $(32)) + (((r67_z % $(64)) - $(32)) % $(32))), dtype=tl.bfloat16))
    r67_value = r67_value.to(tl.bfloat16)
    r67_value = tl.broadcast_to(r67_value, [$(65536)])
    tl.store($(vcat_5_ptr) + r67_z, r67_value)
    r68_z = tl.arange($(4096))
    r68_value = (tl.load($(vmm_2_ptr) + ((((r68_z / $(32)) % $(128)) * $(320)) + ((r68_z % $(32)) * $(2)) + $(256)), dtype=tl.bfloat16) * tl.load($(vto_copy_17_ptr) + ((((r68_z / $(32)) % $(128)) * $(64)) + (r68_z % $(32))), dtype=tl.bfloat16))
    r68_value = r68_value.to(tl.bfloat16)
    r68_value = tl.broadcast_to(r68_value, [$(4096)])
    tl.store($(vmul_21_ptr) + r68_z, r68_value)
    r69_z = tl.arange($(4096))
    r69_value = (tl.load($(vmm_2_ptr) + ((((r69_z / $(32)) % $(128)) * $(320)) + ((r69_z % $(32)) * $(2)) + $(257)), dtype=tl.bfloat16) * tl.load($(vto_copy_18_ptr) + ((((r69_z / $(32)) % $(128)) * $(64)) + (r69_z % $(32))), dtype=tl.bfloat16))
    r69_value = r69_value.to(tl.bfloat16)
    r69_value = tl.broadcast_to(r69_value, [$(4096)])
    tl.store($(vmul_22_ptr) + r69_z, r69_value)
    r70_z = tl.arange($(4096))
    r70_value = (tl.load($(vmul_21_ptr) + ((((r70_z / $(32)) % $(128)) * $(32)) + (r70_z % $(32))), dtype=tl.bfloat16) - tl.load($(vmul_22_ptr) + ((((r70_z / $(32)) % $(128)) * $(32)) + (r70_z % $(32))), dtype=tl.bfloat16))
    r70_value = r70_value.to(tl.bfloat16)
    r70_value = tl.broadcast_to(r70_value, [$(4096)])
    tl.store($(vsub_5_ptr) + r70_z, r70_value)
    r71_z = tl.arange($(4096))
    r71_value = (tl.load($(vmm_2_ptr) + ((((r71_z / $(32)) % $(128)) * $(320)) + ((r71_z % $(32)) * $(2)) + $(257)), dtype=tl.bfloat16) * tl.load($(vto_copy_17_ptr) + ((((r71_z / $(32)) % $(128)) * $(64)) + (r71_z % $(32))), dtype=tl.bfloat16))
    r71_value = r71_value.to(tl.bfloat16)
    r71_value = tl.broadcast_to(r71_value, [$(4096)])
    tl.store($(vmul_23_ptr) + r71_z, r71_value)
    r72_z = tl.arange($(4096))
    r72_value = (tl.load($(vmm_2_ptr) + ((((r72_z / $(32)) % $(128)) * $(320)) + ((r72_z % $(32)) * $(2)) + $(256)), dtype=tl.bfloat16) * tl.load($(vto_copy_18_ptr) + ((((r72_z / $(32)) % $(128)) * $(64)) + (r72_z % $(32))), dtype=tl.bfloat16))
    r72_value = r72_value.to(tl.bfloat16)
    r72_value = tl.broadcast_to(r72_value, [$(4096)])
    tl.store($(vmul_24_ptr) + r72_z, r72_value)
    r73_z = tl.arange($(4096))
    r73_value = (tl.load($(vmul_23_ptr) + ((((r73_z / $(32)) % $(128)) * $(32)) + (r73_z % $(32))), dtype=tl.bfloat16) + tl.load($(vmul_24_ptr) + ((((r73_z / $(32)) % $(128)) * $(32)) + (r73_z % $(32))), dtype=tl.bfloat16))
    r73_value = r73_value.to(tl.bfloat16)
    r73_value = tl.broadcast_to(r73_value, [$(4096)])
    tl.store($(vadd_6_ptr) + r73_z, r73_value)
    r74_z = tl.arange($(8192))
    r74_value = tl.where((r74_z % $(64)) < $(32), tl.load($(vsub_5_ptr) + ((((r74_z / $(64)) % $(128)) * $(32)) + (((r74_z % $(64)) - $(0)) % $(32))), dtype=tl.bfloat16), tl.load($(vadd_6_ptr) + ((((r74_z / $(64)) % $(128)) * $(32)) + (((r74_z % $(64)) - $(32)) % $(32))), dtype=tl.bfloat16))
    r74_value = r74_value.to(tl.bfloat16)
    r74_value = tl.broadcast_to(r74_value, [$(8192)])
    tl.store($(vcat_6_ptr) + r74_z, r74_value)
    r75_z = tl.arange($(196608))
    r75_value = tl.where((r75_z % $(192)) < $(128), tl.load($(vmm_1_ptr) + ((((((((r75_z / $(192)) % $(128)) * $(1536)) + (((r75_z / $(24576)) % $(8)) * $(192)) + ((r75_z % $(192)) - $(0))) / $(1536)) % $(128)) * $(1536)) + (((((r75_z / $(192)) % $(128)) * $(1536)) + (((r75_z / $(24576)) % $(8)) * $(192)) + ((r75_z % $(192)) - $(0))) % $(1536))), dtype=tl.bfloat16), tl.load($(vcat_5_ptr) + ((((r75_z / $(24576)) % $(8)) * $(8192)) + (((r75_z / $(192)) % $(128)) * $(64)) + (((r75_z % $(192)) - $(128)) % $(64))), dtype=tl.bfloat16))
    r75_value = r75_value.to(tl.bfloat16)
    r75_value = tl.broadcast_to(r75_value, [$(196608)])
    tl.store($(vcat_7_ptr) + r75_z, r75_value)
    r76_z = tl.arange($(196608))
    r76_value = tl.where((r76_z % $(192)) < $(128), tl.load($(vmm_3_ptr) + ((((((((r76_z / $(192)) % $(128)) * $(2048)) + (((r76_z / $(24576)) % $(8)) * $(256)) + ((r76_z % $(192)) - $(0))) / $(2048)) % $(128)) * $(2048)) + (((((r76_z / $(192)) % $(128)) * $(2048)) + (((r76_z / $(24576)) % $(8)) * $(256)) + ((r76_z % $(192)) - $(0))) % $(2048))), dtype=tl.bfloat16), tl.load($(vcat_6_ptr) + ((((r76_z / $(192)) % $(128)) * $(64)) + (((r76_z % $(192)) - $(128)) % $(64))), dtype=tl.bfloat16))
    r76_value = r76_value.to(tl.bfloat16)
    r76_value = tl.broadcast_to(r76_value, [$(196608)])
    tl.store($(vcat_8_ptr) + r76_z, r76_value)
    r77_row = tl.arange($(128))
    r77_col = tl.arange($(128))
    r77_lane = tl.arange($(192))
    for r77_group in range($(8)) {
      r77_left = tl.load($(vcat_7_ptr) + (((((((r77_group % $(8)) * $(24576)) + (r77_row[:, None] * $(192)) + r77_lane[None, :]) / $(24576)) % $(8)) * $(24576)) + ((((((r77_group % $(8)) * $(24576)) + (r77_row[:, None] * $(192)) + r77_lane[None, :]) / $(192)) % $(128)) * $(192)) + ((((r77_group % $(8)) * $(24576)) + (r77_row[:, None] * $(192)) + r77_lane[None, :]) % $(192))), dtype=tl.bfloat16)
      r77_right = tl.load($(vcat_8_ptr) + (((((((r77_group % $(8)) * $(24576)) + (r77_lane[:, None] * $(128)) + r77_col[None, :]) / $(24576)) % $(8)) * $(24576)) + (((((r77_group % $(8)) * $(24576)) + (r77_lane[:, None] * $(128)) + r77_col[None, :]) % $(128)) * $(192)) + (((((r77_group % $(8)) * $(24576)) + (r77_lane[:, None] * $(128)) + r77_col[None, :]) / $(128)) % $(192))), dtype=tl.bfloat16)
      r77_value = tl.dot(r77_left, r77_right)
      r77_value = r77_value.to(tl.bfloat16)
      tl.store($(vbmm_2_ptr) + ((r77_group * $(16384)) + (r77_row[:, None] * $(128)) + r77_col[None, :]), r77_value)
    }
    r78_z = tl.arange($(131072))
    r78_value = (tl.load($(vbmm_2_ptr) + ((((r78_z / $(16384)) % $(8)) * $(16384)) + (((r78_z / $(128)) % $(128)) * $(128)) + (r78_z % $(128))), dtype=tl.bfloat16) * tl.cast(0.07216878364870322, tl.bfloat16))
    r78_value = r78_value.to(tl.bfloat16)
    r78_value = tl.broadcast_to(r78_value, [$(131072)])
    tl.store($(vmul_25_ptr) + r78_z, r78_value)
    r79_z = tl.arange($(131072))
    r79_value = (tl.load($(vmul_25_ptr) + ((((r79_z / $(16384)) % $(8)) * $(16384)) + (((r79_z / $(128)) % $(128)) * $(128)) + (r79_z % $(128))), dtype=tl.bfloat16) + tl.load($(vwhere_ptr) + ((((r79_z / $(128)) % $(128)) * $(128)) + (r79_z % $(128))), dtype=tl.bfloat16))
    r79_value = r79_value.to(tl.bfloat16)
    r79_value = tl.broadcast_to(r79_value, [$(131072)])
    tl.store($(vadd_7_ptr) + r79_z, r79_value)
    r80_z = tl.arange($(131072))
    r80_value = tl.load($(vadd_7_ptr) + ((((r80_z / $(16384)) % $(8)) * $(16384)) + (((r80_z / $(128)) % $(128)) * $(128)) + (r80_z % $(128))), dtype=tl.bfloat16)
    r80_value = r80_value.to(tl.float32)
    r80_value = tl.broadcast_to(r80_value, [$(131072)])
    tl.store($(vto_copy_24_ptr) + r80_z, r80_value)
    r81_z = tl.arange($(131072))
    r81_lane = tl.arange($(128))
    r81_row_values = tl.load($(vto_copy_24_ptr) + ((((r81_z[:, None] / $(16384)) % $(8)) * $(16384)) + (((r81_z[:, None] / $(128)) % $(128)) * $(128)) + (r81_lane[None, :] % $(128))), dtype=tl.float32)
    r81_maximum = tl.max(r81_row_values, axis=1, keep_dims=true)
    r81_weights = tl.exp(r81_row_values - r81_maximum)
    r81_total = tl.sum(r81_weights, axis=1)
    r81_maximum = tl.max(r81_row_values, axis=1)
    r81_value = tl.exp(tl.load($(vto_copy_24_ptr) + ((((r81_z / $(16384)) % $(8)) * $(16384)) + (((r81_z / $(128)) % $(128)) * $(128)) + (r81_z % $(128))), dtype=tl.float32) - r81_maximum) / r81_total
    r81_value = r81_value.to(tl.float32)
    r81_value = tl.broadcast_to(r81_value, [$(131072)])
    tl.store($(vsoftmax_ptr) + r81_z, r81_value)
    r82_z = tl.arange($(131072))
    r82_value = tl.load($(vsoftmax_ptr) + ((((r82_z / $(16384)) % $(8)) * $(16384)) + (((r82_z / $(128)) % $(128)) * $(128)) + (r82_z % $(128))), dtype=tl.float32)
    r82_value = r82_value.to(tl.bfloat16)
    r82_value = tl.broadcast_to(r82_value, [$(131072)])
    tl.store($(vto_copy_25_ptr) + r82_z, r82_value)
    r83_row = tl.arange($(128))
    r83_col = tl.arange($(1024))
    r83_lane = tl.arange($(512))
    r83_left = tl.load($(input1_ptr) + ((((((r83_row[:, None] * $(512)) + r83_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r83_row[:, None] * $(512)) + r83_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r83_right = tl.load($(vto_copy_6_ptr) + (((r83_lane[:, None] % $(512)) * $(1024)) + (r83_col[None, :] % $(1024))), dtype=tl.bfloat16)
    r83_value = tl.dot(r83_left, r83_right)
    r83_value = r83_value.to(tl.bfloat16)
    tl.store($(vmm_5_ptr) + ((r83_row[:, None] * $(1024)) + r83_col[None, :]), r83_value)
    r84_row = tl.arange($(128))
    r84_col = tl.arange($(128))
    r84_lane = tl.arange($(128))
    for r84_group in range($(8)) {
      r84_left = tl.load($(vto_copy_25_ptr) + (((((((r84_group % $(8)) * $(16384)) + (r84_lane[None, :] * $(128)) + r84_row[:, None]) / $(16384)) % $(8)) * $(16384)) + ((((((r84_group % $(8)) * $(16384)) + (r84_lane[None, :] * $(128)) + r84_row[:, None]) / $(128)) % $(128)) * $(128)) + ((((r84_group % $(8)) * $(16384)) + (r84_lane[None, :] * $(128)) + r84_row[:, None]) % $(128))), dtype=tl.bfloat16)
      r84_right = tl.load($(vmm_5_ptr) + (((((((r84_group % $(8)) * $(16384)) + (r84_lane[:, None] * $(128)) + r84_col[None, :]) / $(128)) % $(128)) * $(1024)) + ((((((r84_group % $(8)) * $(16384)) + (r84_lane[:, None] * $(128)) + r84_col[None, :]) / $(16384)) % $(8)) * $(128)) + ((((r84_group % $(8)) * $(16384)) + (r84_lane[:, None] * $(128)) + r84_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r84_value = tl.dot(r84_left, r84_right)
      r84_value = r84_value.to(tl.bfloat16)
      tl.store($(vbmm_4_ptr) + ((r84_group * $(16384)) + (r84_row[:, None] * $(128)) + r84_col[None, :]), r84_value)
    }
    r85_row = tl.arange($(128))
    r85_col = tl.arange($(128))
    r85_lane = tl.arange($(128))
    for r85_group in range($(8)) {
      r85_left = tl.load($(vmm_5_ptr) + (((((((r85_group % $(8)) * $(16384)) + (r85_row[:, None] * $(128)) + r85_lane[None, :]) / $(128)) % $(128)) * $(1024)) + ((((((r85_group % $(8)) * $(16384)) + (r85_row[:, None] * $(128)) + r85_lane[None, :]) / $(16384)) % $(8)) * $(128)) + ((((r85_group % $(8)) * $(16384)) + (r85_row[:, None] * $(128)) + r85_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r85_right = tl.load($(vmm_3_ptr) + (((((((r85_group % $(8)) * $(16384)) + (r85_col[None, :] * $(128)) + r85_lane[:, None]) / $(128)) % $(128)) * $(2048)) + ((((((r85_group % $(8)) * $(16384)) + (r85_col[None, :] * $(128)) + r85_lane[:, None]) / $(16384)) % $(8)) * $(256)) + ((((r85_group % $(8)) * $(16384)) + (r85_col[None, :] * $(128)) + r85_lane[:, None]) % $(128)) + $(128)), dtype=tl.bfloat16)
      r85_value = tl.dot(r85_left, r85_right)
      r85_value = r85_value.to(tl.bfloat16)
      tl.store($(vbmm_5_ptr) + ((r85_group * $(16384)) + (r85_row[:, None] * $(128)) + r85_col[None, :]), r85_value)
    }
    r86_z = tl.arange($(131072))
    r86_value = tl.load($(vbmm_5_ptr) + ((((r86_z / $(16384)) % $(8)) * $(16384)) + (((r86_z / $(128)) % $(128)) * $(128)) + (r86_z % $(128))), dtype=tl.bfloat16)
    r86_value = r86_value.to(tl.float32)
    r86_value = tl.broadcast_to(r86_value, [$(131072)])
    tl.store($(vto_copy_26_ptr) + r86_z, r86_value)
    r87_z = tl.arange($(131072))
    r87_lane = tl.arange($(128))
    r87_row_values = tl.load($(vsoftmax_ptr) + ((((r87_z[:, None] / $(16384)) % $(8)) * $(16384)) + (((r87_z[:, None] / $(128)) % $(128)) * $(128)) + (r87_lane[None, :] % $(128))), dtype=tl.float32)
    r87_row_grad = tl.load($(vto_copy_26_ptr) + ((((r87_z[:, None] / $(16384)) % $(8)) * $(16384)) + (((r87_z[:, None] / $(128)) % $(128)) * $(128)) + (r87_lane[None, :] % $(128))), dtype=tl.float32)
    r87_weighted = r87_row_grad * r87_row_values
    r87_dot_grad = tl.sum(r87_weighted, axis=1)
    r87_value = (tl.load($(vto_copy_26_ptr) + ((((r87_z / $(16384)) % $(8)) * $(16384)) + (((r87_z / $(128)) % $(128)) * $(128)) + (r87_z % $(128))), dtype=tl.float32) - r87_dot_grad) * tl.load($(vsoftmax_ptr) + ((((r87_z / $(16384)) % $(8)) * $(16384)) + (((r87_z / $(128)) % $(128)) * $(128)) + (r87_z % $(128))), dtype=tl.float32)
    r87_value = r87_value.to(tl.float32)
    r87_value = tl.broadcast_to(r87_value, [$(131072)])
    tl.store($(vsoftmax_backward_data_ptr) + r87_z, r87_value)
    r88_z = tl.arange($(131072))
    r88_value = tl.load($(vsoftmax_backward_data_ptr) + ((((r88_z / $(16384)) % $(8)) * $(16384)) + (((r88_z / $(128)) % $(128)) * $(128)) + (r88_z % $(128))), dtype=tl.float32)
    r88_value = r88_value.to(tl.bfloat16)
    r88_value = tl.broadcast_to(r88_value, [$(131072)])
    tl.store($(vto_copy_27_ptr) + r88_z, r88_value)
    r89_z = tl.arange($(131072))
    r89_value = (tl.load($(vto_copy_27_ptr) + ((((r89_z / $(16384)) % $(8)) * $(16384)) + (((r89_z / $(128)) % $(128)) * $(128)) + (r89_z % $(128))), dtype=tl.bfloat16) * tl.cast(0.07216878364870322, tl.bfloat16))
    r89_value = r89_value.to(tl.bfloat16)
    r89_value = tl.broadcast_to(r89_value, [$(131072)])
    tl.store($(vmul_26_ptr) + r89_z, r89_value)
    r90_row = tl.arange($(192))
    r90_col = tl.arange($(128))
    r90_lane = tl.arange($(128))
    for r90_group in range($(8)) {
      r90_left = tl.load($(vcat_7_ptr) + (((((((r90_group % $(8)) * $(24576)) + (r90_lane[None, :] * $(192)) + r90_row[:, None]) / $(24576)) % $(8)) * $(24576)) + ((((((r90_group % $(8)) * $(24576)) + (r90_lane[None, :] * $(192)) + r90_row[:, None]) / $(192)) % $(128)) * $(192)) + ((((r90_group % $(8)) * $(24576)) + (r90_lane[None, :] * $(192)) + r90_row[:, None]) % $(192))), dtype=tl.bfloat16)
      r90_right = tl.load($(vmul_26_ptr) + (((((((r90_group % $(8)) * $(16384)) + (r90_lane[:, None] * $(128)) + r90_col[None, :]) / $(16384)) % $(8)) * $(16384)) + ((((((r90_group % $(8)) * $(16384)) + (r90_lane[:, None] * $(128)) + r90_col[None, :]) / $(128)) % $(128)) * $(128)) + ((((r90_group % $(8)) * $(16384)) + (r90_lane[:, None] * $(128)) + r90_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r90_value = tl.dot(r90_left, r90_right)
      r90_value = r90_value.to(tl.bfloat16)
      tl.store($(vbmm_6_ptr) + ((r90_group * $(24576)) + (r90_row[:, None] * $(128)) + r90_col[None, :]), r90_value)
    }
    r91_row = tl.arange($(128))
    r91_col = tl.arange($(192))
    r91_lane = tl.arange($(128))
    for r91_group in range($(8)) {
      r91_left = tl.load($(vmul_26_ptr) + (((((((r91_group % $(8)) * $(16384)) + (r91_row[:, None] * $(128)) + r91_lane[None, :]) / $(16384)) % $(8)) * $(16384)) + ((((((r91_group % $(8)) * $(16384)) + (r91_row[:, None] * $(128)) + r91_lane[None, :]) / $(128)) % $(128)) * $(128)) + ((((r91_group % $(8)) * $(16384)) + (r91_row[:, None] * $(128)) + r91_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r91_right = tl.load($(vcat_8_ptr) + (((((((r91_group % $(8)) * $(24576)) + (r91_col[None, :] * $(128)) + r91_lane[:, None]) / $(24576)) % $(8)) * $(24576)) + (((((r91_group % $(8)) * $(24576)) + (r91_col[None, :] * $(128)) + r91_lane[:, None]) % $(128)) * $(192)) + (((((r91_group % $(8)) * $(24576)) + (r91_col[None, :] * $(128)) + r91_lane[:, None]) / $(128)) % $(192))), dtype=tl.bfloat16)
      r91_value = tl.dot(r91_left, r91_right)
      r91_value = r91_value.to(tl.bfloat16)
      tl.store($(vbmm_7_ptr) + ((r91_group * $(24576)) + (r91_row[:, None] * $(192)) + r91_col[None, :]), r91_value)
    }
    r92_z = tl.arange($(8192))
    r92_lane = tl.arange($(8))
    r92_value = tl.sum(tl.load($(vbmm_6_ptr) + ((r92_lane[None, :] * $(24576)) + ((r92_z[:, None] % $(64)) * $(128)) + ((r92_z[:, None] / $(64)) % $(128)) + $(16384)), dtype=tl.bfloat16), axis=1)
    r92_value = r92_value.to(tl.bfloat16)
    r92_value = tl.broadcast_to(r92_value, [$(8192)])
    tl.store($(vsum_1_ptr) + r92_z, r92_value)
    r93_z = tl.arange($(4096))
    r93_value = (tl.load($(vsum_1_ptr) + ((((r93_z / $(32)) % $(128)) * $(64)) + (r93_z % $(32)) + $(32)), dtype=tl.bfloat16) * tl.load($(vto_copy_18_ptr) + ((((r93_z / $(32)) % $(128)) * $(64)) + (r93_z % $(32))), dtype=tl.bfloat16))
    r93_value = r93_value.to(tl.bfloat16)
    r93_value = tl.broadcast_to(r93_value, [$(4096)])
    tl.store($(vmul_27_ptr) + r93_z, r93_value)
    r94_z = tl.arange($(4096))
    r94_value = (tl.load($(vsum_1_ptr) + ((((r94_z / $(32)) % $(128)) * $(64)) + (r94_z % $(32)) + $(32)), dtype=tl.bfloat16) * tl.load($(vto_copy_17_ptr) + ((((r94_z / $(32)) % $(128)) * $(64)) + (r94_z % $(32))), dtype=tl.bfloat16))
    r94_value = r94_value.to(tl.bfloat16)
    r94_value = tl.broadcast_to(r94_value, [$(4096)])
    tl.store($(vmul_28_ptr) + r94_z, r94_value)
    r95_z = tl.arange($(4096))
    r95_value = -tl.load($(vsum_1_ptr) + ((((r95_z / $(32)) % $(128)) * $(64)) + (r95_z % $(32))), dtype=tl.bfloat16)
    r95_value = r95_value.to(tl.bfloat16)
    r95_value = tl.broadcast_to(r95_value, [$(4096)])
    tl.store($(vneg_ptr) + r95_z, r95_value)
    r96_z = tl.arange($(4096))
    r96_value = (tl.load($(vneg_ptr) + ((((r96_z / $(32)) % $(128)) * $(32)) + (r96_z % $(32))), dtype=tl.bfloat16) * tl.load($(vto_copy_18_ptr) + ((((r96_z / $(32)) % $(128)) * $(64)) + (r96_z % $(32))), dtype=tl.bfloat16))
    r96_value = r96_value.to(tl.bfloat16)
    r96_value = tl.broadcast_to(r96_value, [$(4096)])
    tl.store($(vmul_29_ptr) + r96_z, r96_value)
    r97_z = tl.arange($(4096))
    r97_value = (tl.load($(vmul_28_ptr) + ((((r97_z / $(32)) % $(128)) * $(32)) + (r97_z % $(32))), dtype=tl.bfloat16) + tl.load($(vmul_29_ptr) + ((((r97_z / $(32)) % $(128)) * $(32)) + (r97_z % $(32))), dtype=tl.bfloat16))
    r97_value = r97_value.to(tl.bfloat16)
    r97_value = tl.broadcast_to(r97_value, [$(4096)])
    tl.store($(vadd_8_ptr) + r97_z, r97_value)
    r98_z = tl.arange($(4096))
    r98_value = (tl.load($(vsum_1_ptr) + ((((r98_z / $(32)) % $(128)) * $(64)) + (r98_z % $(32))), dtype=tl.bfloat16) * tl.load($(vto_copy_17_ptr) + ((((r98_z / $(32)) % $(128)) * $(64)) + (r98_z % $(32))), dtype=tl.bfloat16))
    r98_value = r98_value.to(tl.bfloat16)
    r98_value = tl.broadcast_to(r98_value, [$(4096)])
    tl.store($(vmul_30_ptr) + r98_z, r98_value)
    r99_z = tl.arange($(4096))
    r99_value = (tl.load($(vmul_27_ptr) + ((((r99_z / $(32)) % $(128)) * $(32)) + (r99_z % $(32))), dtype=tl.bfloat16) + tl.load($(vmul_30_ptr) + ((((r99_z / $(32)) % $(128)) * $(32)) + (r99_z % $(32))), dtype=tl.bfloat16))
    r99_value = r99_value.to(tl.bfloat16)
    r99_value = tl.broadcast_to(r99_value, [$(4096)])
    tl.store($(vadd_9_ptr) + r99_z, r99_value)
    r100_z = tl.arange($(32768))
    r100_value = (tl.load($(vbmm_7_ptr) + ((((r100_z / $(4096)) % $(8)) * $(24576)) + (((r100_z / $(32)) % $(128)) * $(192)) + (r100_z % $(32)) + $(160)), dtype=tl.bfloat16) * tl.load($(vto_copy_18_ptr) + ((((r100_z / $(32)) % $(128)) * $(64)) + (r100_z % $(32))), dtype=tl.bfloat16))
    r100_value = r100_value.to(tl.bfloat16)
    r100_value = tl.broadcast_to(r100_value, [$(32768)])
    tl.store($(vmul_31_ptr) + r100_z, r100_value)
    r101_z = tl.arange($(32768))
    r101_value = (tl.load($(vbmm_7_ptr) + ((((r101_z / $(4096)) % $(8)) * $(24576)) + (((r101_z / $(32)) % $(128)) * $(192)) + (r101_z % $(32)) + $(160)), dtype=tl.bfloat16) * tl.load($(vto_copy_17_ptr) + ((((r101_z / $(32)) % $(128)) * $(64)) + (r101_z % $(32))), dtype=tl.bfloat16))
    r101_value = r101_value.to(tl.bfloat16)
    r101_value = tl.broadcast_to(r101_value, [$(32768)])
    tl.store($(vmul_32_ptr) + r101_z, r101_value)
    r102_z = tl.arange($(32768))
    r102_value = -tl.load($(vbmm_7_ptr) + ((((r102_z / $(4096)) % $(8)) * $(24576)) + (((r102_z / $(32)) % $(128)) * $(192)) + (r102_z % $(32)) + $(128)), dtype=tl.bfloat16)
    r102_value = r102_value.to(tl.bfloat16)
    r102_value = tl.broadcast_to(r102_value, [$(32768)])
    tl.store($(vneg_1_ptr) + r102_z, r102_value)
    r103_z = tl.arange($(32768))
    r103_value = (tl.load($(vneg_1_ptr) + ((((r103_z / $(4096)) % $(8)) * $(4096)) + (((r103_z / $(32)) % $(128)) * $(32)) + (r103_z % $(32))), dtype=tl.bfloat16) * tl.load($(vto_copy_18_ptr) + ((((r103_z / $(32)) % $(128)) * $(64)) + (r103_z % $(32))), dtype=tl.bfloat16))
    r103_value = r103_value.to(tl.bfloat16)
    r103_value = tl.broadcast_to(r103_value, [$(32768)])
    tl.store($(vmul_33_ptr) + r103_z, r103_value)
    r104_z = tl.arange($(32768))
    r104_value = (tl.load($(vmul_32_ptr) + ((((r104_z / $(4096)) % $(8)) * $(4096)) + (((r104_z / $(32)) % $(128)) * $(32)) + (r104_z % $(32))), dtype=tl.bfloat16) + tl.load($(vmul_33_ptr) + ((((r104_z / $(4096)) % $(8)) * $(4096)) + (((r104_z / $(32)) % $(128)) * $(32)) + (r104_z % $(32))), dtype=tl.bfloat16))
    r104_value = r104_value.to(tl.bfloat16)
    r104_value = tl.broadcast_to(r104_value, [$(32768)])
    tl.store($(vadd_10_ptr) + r104_z, r104_value)
    r105_z = tl.arange($(32768))
    r105_value = (tl.load($(vbmm_7_ptr) + ((((r105_z / $(4096)) % $(8)) * $(24576)) + (((r105_z / $(32)) % $(128)) * $(192)) + (r105_z % $(32)) + $(128)), dtype=tl.bfloat16) * tl.load($(vto_copy_17_ptr) + ((((r105_z / $(32)) % $(128)) * $(64)) + (r105_z % $(32))), dtype=tl.bfloat16))
    r105_value = r105_value.to(tl.bfloat16)
    r105_value = tl.broadcast_to(r105_value, [$(32768)])
    tl.store($(vmul_34_ptr) + r105_z, r105_value)
    r106_z = tl.arange($(32768))
    r106_value = (tl.load($(vmul_31_ptr) + ((((r106_z / $(4096)) % $(8)) * $(4096)) + (((r106_z / $(32)) % $(128)) * $(32)) + (r106_z % $(32))), dtype=tl.bfloat16) + tl.load($(vmul_34_ptr) + ((((r106_z / $(4096)) % $(8)) * $(4096)) + (((r106_z / $(32)) % $(128)) * $(32)) + (r106_z % $(32))), dtype=tl.bfloat16))
    r106_value = r106_value.to(tl.bfloat16)
    r106_value = tl.broadcast_to(r106_value, [$(32768)])
    tl.store($(vadd_11_ptr) + r106_z, r106_value)
    r107_z = tl.arange($(8192))
    r107_value = tl.where((((r107_z % $(64)) >= $(1)) & ((r107_z % $(64)) < $(64)) & (((r107_z % $(64)) - $(1)) % $(2) == $(0))), tl.load($(vadd_8_ptr) + ((((r107_z / $(64)) % $(128)) * $(32)) + ((((r107_z % $(64)) - $(1)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r107_value = r107_value.to(tl.bfloat16)
    r107_value = tl.broadcast_to(r107_value, [$(8192)])
    tl.store($(vslice_backward_ptr) + r107_z, r107_value)
    r108_z = tl.arange($(8192))
    r108_value = tl.where((((r108_z % $(64)) >= $(0)) & ((r108_z % $(64)) < $(64)) & (((r108_z % $(64)) - $(0)) % $(2) == $(0))), tl.load($(vadd_9_ptr) + ((((r108_z / $(64)) % $(128)) * $(32)) + ((((r108_z % $(64)) - $(0)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r108_value = r108_value.to(tl.bfloat16)
    r108_value = tl.broadcast_to(r108_value, [$(8192)])
    tl.store($(vslice_backward_1_ptr) + r108_z, r108_value)
    r109_z = tl.arange($(8192))
    r109_value = (tl.load($(vslice_backward_ptr) + ((((r109_z / $(64)) % $(128)) * $(64)) + (r109_z % $(64))), dtype=tl.bfloat16) + tl.load($(vslice_backward_1_ptr) + ((((r109_z / $(64)) % $(128)) * $(64)) + (r109_z % $(64))), dtype=tl.bfloat16))
    r109_value = r109_value.to(tl.bfloat16)
    r109_value = tl.broadcast_to(r109_value, [$(8192)])
    tl.store($(vadd_12_ptr) + r109_z, r109_value)
    r110_z = tl.arange($(65536))
    r110_value = tl.where((((r110_z % $(64)) >= $(1)) & ((r110_z % $(64)) < $(64)) & (((r110_z % $(64)) - $(1)) % $(2) == $(0))), tl.load($(vadd_10_ptr) + ((((r110_z / $(8192)) % $(8)) * $(4096)) + (((r110_z / $(64)) % $(128)) * $(32)) + ((((r110_z % $(64)) - $(1)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r110_value = r110_value.to(tl.bfloat16)
    r110_value = tl.broadcast_to(r110_value, [$(65536)])
    tl.store($(vslice_backward_2_ptr) + r110_z, r110_value)
    r111_z = tl.arange($(65536))
    r111_value = tl.where((((r111_z % $(64)) >= $(0)) & ((r111_z % $(64)) < $(64)) & (((r111_z % $(64)) - $(0)) % $(2) == $(0))), tl.load($(vadd_11_ptr) + ((((r111_z / $(8192)) % $(8)) * $(4096)) + (((r111_z / $(64)) % $(128)) * $(32)) + ((((r111_z % $(64)) - $(0)) / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r111_value = r111_value.to(tl.bfloat16)
    r111_value = tl.broadcast_to(r111_value, [$(65536)])
    tl.store($(vslice_backward_3_ptr) + r111_z, r111_value)
    r112_z = tl.arange($(65536))
    r112_value = (tl.load($(vslice_backward_2_ptr) + ((((r112_z / $(8192)) % $(8)) * $(8192)) + (((r112_z / $(64)) % $(128)) * $(64)) + (r112_z % $(64))), dtype=tl.bfloat16) + tl.load($(vslice_backward_3_ptr) + ((((r112_z / $(8192)) % $(8)) * $(8192)) + (((r112_z / $(64)) % $(128)) * $(64)) + (r112_z % $(64))), dtype=tl.bfloat16))
    r112_value = r112_value.to(tl.bfloat16)
    r112_value = tl.broadcast_to(r112_value, [$(65536)])
    tl.store($(vadd_13_ptr) + r112_z, r112_value)
    r113_z = tl.arange($(262144))
    r113_value = tl.where((r113_z % $(256)) < $(128), tl.load($(vbmm_6_ptr) + ((((((((r113_z / $(32768)) % $(8)) * $(24576)) + (((r113_z % $(256)) - $(0)) * $(128)) + ((r113_z / $(256)) % $(128))) / $(24576)) % $(8)) * $(24576)) + ((((((r113_z / $(32768)) % $(8)) * $(192)) + ((r113_z % $(256)) - $(0))) % $(192)) * $(128)) + ((r113_z / $(256)) % $(128))), dtype=tl.bfloat16), tl.load($(vbmm_4_ptr) + ((((((((r113_z / $(32768)) % $(8)) * $(16384)) + (((r113_z / $(256)) % $(128)) * $(128)) + ((r113_z % $(256)) - $(128))) / $(16384)) % $(8)) * $(16384)) + (((((((r113_z / $(32768)) % $(8)) * $(16384)) + (((r113_z / $(256)) % $(128)) * $(128)) + ((r113_z % $(256)) - $(128))) / $(128)) % $(128)) * $(128)) + (((((r113_z / $(32768)) % $(8)) * $(16384)) + (((r113_z / $(256)) % $(128)) * $(128)) + ((r113_z % $(256)) - $(128))) % $(128))), dtype=tl.bfloat16))
    r113_value = r113_value.to(tl.bfloat16)
    r113_value = tl.broadcast_to(r113_value, [$(262144)])
    tl.store($(vcat_9_ptr) + r113_z, r113_value)
    r114_z = tl.arange($(196608))
    r114_value = tl.where((r114_z % $(192)) < $(128), tl.load($(vbmm_7_ptr) + ((((((((r114_z / $(24576)) % $(8)) * $(24576)) + (((r114_z / $(192)) % $(128)) * $(192)) + ((r114_z % $(192)) - $(0))) / $(24576)) % $(8)) * $(24576)) + (((((((r114_z / $(24576)) % $(8)) * $(24576)) + (((r114_z / $(192)) % $(128)) * $(192)) + ((r114_z % $(192)) - $(0))) / $(192)) % $(128)) * $(192)) + (((((r114_z / $(24576)) % $(8)) * $(24576)) + (((r114_z / $(192)) % $(128)) * $(192)) + ((r114_z % $(192)) - $(0))) % $(192))), dtype=tl.bfloat16), tl.load($(vadd_13_ptr) + ((((r114_z / $(24576)) % $(8)) * $(8192)) + (((r114_z / $(192)) % $(128)) * $(64)) + (((r114_z % $(192)) - $(128)) % $(64))), dtype=tl.bfloat16))
    r114_value = r114_value.to(tl.bfloat16)
    r114_value = tl.broadcast_to(r114_value, [$(196608)])
    tl.store($(vcat_10_ptr) + r114_z, r114_value)
    r115_row = tl.arange($(128))
    r115_col = tl.arange($(256))
    r115_lane = tl.arange($(2048))
    r115_left = tl.load($(vcat_9_ptr) + ((((((((((r115_row[:, None] * $(2048)) + r115_lane[None, :]) / $(2048)) % $(128)) * $(2048)) + (((r115_row[:, None] * $(2048)) + r115_lane[None, :]) % $(2048))) / $(256)) % $(8)) * $(32768)) + (((((r115_row[:, None] * $(2048)) + r115_lane[None, :]) / $(2048)) % $(128)) * $(256)) + (((((((r115_row[:, None] * $(2048)) + r115_lane[None, :]) / $(2048)) % $(128)) * $(2048)) + (((r115_row[:, None] * $(2048)) + r115_lane[None, :]) % $(2048))) % $(256))), dtype=tl.bfloat16)
    r115_right = tl.load($(vto_copy_5_ptr) + (((r115_lane[:, None] % $(2048)) * $(256)) + (r115_col[None, :] % $(256))), dtype=tl.bfloat16)
    r115_value = tl.dot(r115_left, r115_right)
    r115_value = r115_value.to(tl.bfloat16)
    tl.store($(vmm_6_ptr) + ((r115_row[:, None] * $(256)) + r115_col[None, :]), r115_value)
    r116_z = tl.arange($(32768))
    r116_value = tl.load($(vmm_6_ptr) + ((((r116_z / $(256)) % $(128)) * $(256)) + (r116_z % $(256))), dtype=tl.bfloat16)
    r116_value = r116_value.to(tl.float32)
    r116_value = tl.broadcast_to(r116_value, [$(32768)])
    tl.store($(vto_copy_28_ptr) + r116_z, r116_value)
    r117_z = tl.arange($(32768))
    r117_value = (tl.load($(vto_copy_28_ptr) + ((((r117_z / $(256)) % $(128)) * $(256)) + (r117_z % $(256))), dtype=tl.float32) * tl.cast(tl.load($(vto_copy_4_ptr) + (r117_z % $(256)), dtype=tl.bfloat16), tl.float32))
    r117_value = r117_value.to(tl.float32)
    r117_value = tl.broadcast_to(r117_value, [$(32768)])
    tl.store($(vmul_35_ptr) + r117_z, r117_value)
    r118_z = tl.arange($(32768))
    r118_value = (tl.load($(vmul_35_ptr) + ((((r118_z / $(256)) % $(128)) * $(256)) + (r118_z % $(256))), dtype=tl.float32) * tl.load($(vto_copy_22_ptr) + ((((r118_z / $(256)) % $(128)) * $(256)) + (r118_z % $(256))), dtype=tl.float32))
    r118_value = r118_value.to(tl.float32)
    r118_value = tl.broadcast_to(r118_value, [$(32768)])
    tl.store($(vmul_36_ptr) + r118_z, r118_value)
    r119_z = tl.arange($(32768))
    r119_value = (tl.load($(vmul_35_ptr) + ((((r119_z / $(256)) % $(128)) * $(256)) + (r119_z % $(256))), dtype=tl.float32) * tl.load($(vrsqrt_1_ptr) + ((r119_z / $(256)) % $(128)), dtype=tl.float32))
    r119_value = r119_value.to(tl.float32)
    r119_value = tl.broadcast_to(r119_value, [$(32768)])
    tl.store($(vmul_37_ptr) + r119_z, r119_value)
    r120_z = tl.arange($(128))
    r120_lane = tl.arange($(256))
    r120_value = tl.sum(tl.load($(vmul_36_ptr) + ((r120_z[:, None] * $(256)) + r120_lane[None, :]), dtype=tl.float32), axis=1)
    r120_value = r120_value.to(tl.float32)
    r120_value = tl.broadcast_to(r120_value, [$(128)])
    tl.store($(vsum_2_ptr) + r120_z, r120_value)
    r121_z = tl.arange($(128))
    r121_value = ((tl.load($(vrsqrt_1_ptr) + r121_z, dtype=tl.float32)) * (tl.load($(vrsqrt_1_ptr) + r121_z, dtype=tl.float32)) * (tl.load($(vrsqrt_1_ptr) + r121_z, dtype=tl.float32)))
    r121_value = r121_value.to(tl.float32)
    r121_value = tl.broadcast_to(r121_value, [$(128)])
    tl.store($(vpow_5_ptr) + r121_z, r121_value)
    r122_z = tl.arange($(128))
    r122_value = (tl.load($(vsum_2_ptr) + r122_z, dtype=tl.float32) * -0.5)
    r122_value = r122_value.to(tl.float32)
    r122_value = tl.broadcast_to(r122_value, [$(128)])
    tl.store($(vmul_38_ptr) + r122_z, r122_value)
    r123_z = tl.arange($(128))
    r123_value = (tl.load($(vmul_38_ptr) + r123_z, dtype=tl.float32) * tl.load($(vpow_5_ptr) + r123_z, dtype=tl.float32))
    r123_value = r123_value.to(tl.float32)
    r123_value = tl.broadcast_to(r123_value, [$(128)])
    tl.store($(vmul_39_ptr) + r123_z, r123_value)
    r124_z = tl.arange($(32768))
    r124_value = (tl.load($(vmul_39_ptr) + ((r124_z / $(256)) % $(128)), dtype=tl.float32) / 256.0)
    r124_value = r124_value.to(tl.float32)
    r124_value = tl.broadcast_to(r124_value, [$(32768)])
    tl.store($(vdiv_2_ptr) + r124_z, r124_value)
    r125_z = tl.arange($(32768))
    r125_value = ((tl.load($(vto_copy_22_ptr) + ((((r125_z / $(256)) % $(128)) * $(256)) + (r125_z % $(256))), dtype=tl.float32)))
    r125_value = r125_value.to(tl.float32)
    r125_value = tl.broadcast_to(r125_value, [$(32768)])
    tl.store($(vpow_6_ptr) + r125_z, r125_value)
    r126_z = tl.arange($(32768))
    r126_value = (tl.load($(vpow_6_ptr) + ((((r126_z / $(256)) % $(128)) * $(256)) + (r126_z % $(256))), dtype=tl.float32) * 2.0)
    r126_value = r126_value.to(tl.float32)
    r126_value = tl.broadcast_to(r126_value, [$(32768)])
    tl.store($(vmul_40_ptr) + r126_z, r126_value)
    r127_z = tl.arange($(32768))
    r127_value = (tl.load($(vdiv_2_ptr) + ((((r127_z / $(256)) % $(128)) * $(256)) + (r127_z % $(256))), dtype=tl.float32) * tl.load($(vmul_40_ptr) + ((((r127_z / $(256)) % $(128)) * $(256)) + (r127_z % $(256))), dtype=tl.float32))
    r127_value = r127_value.to(tl.float32)
    r127_value = tl.broadcast_to(r127_value, [$(32768)])
    tl.store($(vmul_41_ptr) + r127_z, r127_value)
    r128_z = tl.arange($(32768))
    r128_value = (tl.load($(vmul_37_ptr) + ((((r128_z / $(256)) % $(128)) * $(256)) + (r128_z % $(256))), dtype=tl.float32) + tl.load($(vmul_41_ptr) + ((((r128_z / $(256)) % $(128)) * $(256)) + (r128_z % $(256))), dtype=tl.float32))
    r128_value = r128_value.to(tl.float32)
    r128_value = tl.broadcast_to(r128_value, [$(32768)])
    tl.store($(vadd_14_ptr) + r128_z, r128_value)
    r129_z = tl.arange($(32768))
    r129_value = tl.load($(vadd_14_ptr) + ((((r129_z / $(256)) % $(128)) * $(256)) + (r129_z % $(256))), dtype=tl.float32)
    r129_value = r129_value.to(tl.bfloat16)
    r129_value = tl.broadcast_to(r129_value, [$(32768)])
    tl.store($(vto_copy_29_ptr) + r129_z, r129_value)
    r130_z = tl.arange($(40960))
    r130_value = tl.where((r130_z % $(320)) < $(256), tl.load($(vto_copy_29_ptr) + ((((r130_z / $(320)) % $(128)) * $(256)) + (((r130_z % $(320)) - $(0)) % $(256))), dtype=tl.bfloat16), tl.load($(vadd_12_ptr) + ((((((((r130_z / $(320)) % $(128)) * $(64)) + ((r130_z % $(320)) - $(256))) / $(64)) % $(128)) * $(64)) + (((((r130_z / $(320)) % $(128)) * $(64)) + ((r130_z % $(320)) - $(256))) % $(64))), dtype=tl.bfloat16))
    r130_value = r130_value.to(tl.bfloat16)
    r130_value = tl.broadcast_to(r130_value, [$(40960)])
    tl.store($(vcat_11_ptr) + r130_z, r130_value)
    r131_row = tl.arange($(128))
    r131_col = tl.arange($(512))
    r131_lane = tl.arange($(320))
    r131_left = tl.load($(vcat_11_ptr) + ((((((r131_row[:, None] * $(320)) + r131_lane[None, :]) / $(320)) % $(128)) * $(320)) + (((r131_row[:, None] * $(320)) + r131_lane[None, :]) % $(320))), dtype=tl.bfloat16)
    r131_right = tl.load($(vto_copy_3_ptr) + (((r131_lane[:, None] % $(320)) * $(512)) + (r131_col[None, :] % $(512))), dtype=tl.bfloat16)
    r131_value = tl.dot(r131_left, r131_right)
    r131_value = r131_value.to(tl.bfloat16)
    tl.store($(vmm_7_ptr) + ((r131_row[:, None] * $(512)) + r131_col[None, :]), r131_value)
    r132_row = tl.arange($(128))
    r132_col = tl.arange($(256))
    r132_lane = tl.arange($(1536))
    r132_left = tl.load($(vcat_10_ptr) + ((((((((((r132_row[:, None] * $(1536)) + r132_lane[None, :]) / $(1536)) % $(128)) * $(1536)) + (((r132_row[:, None] * $(1536)) + r132_lane[None, :]) % $(1536))) / $(192)) % $(8)) * $(24576)) + (((((r132_row[:, None] * $(1536)) + r132_lane[None, :]) / $(1536)) % $(128)) * $(192)) + (((((((r132_row[:, None] * $(1536)) + r132_lane[None, :]) / $(1536)) % $(128)) * $(1536)) + (((r132_row[:, None] * $(1536)) + r132_lane[None, :]) % $(1536))) % $(192))), dtype=tl.bfloat16)
    r132_right = tl.load($(vto_copy_2_ptr) + (((r132_lane[:, None] % $(1536)) * $(256)) + (r132_col[None, :] % $(256))), dtype=tl.bfloat16)
    r132_value = tl.dot(r132_left, r132_right)
    r132_value = r132_value.to(tl.bfloat16)
    tl.store($(vmm_8_ptr) + ((r132_row[:, None] * $(256)) + r132_col[None, :]), r132_value)
    r133_z = tl.arange($(32768))
    r133_value = tl.load($(vmm_8_ptr) + ((((r133_z / $(256)) % $(128)) * $(256)) + (r133_z % $(256))), dtype=tl.bfloat16)
    r133_value = r133_value.to(tl.float32)
    r133_value = tl.broadcast_to(r133_value, [$(32768)])
    tl.store($(vto_copy_30_ptr) + r133_z, r133_value)
    r134_z = tl.arange($(32768))
    r134_value = (tl.load($(vto_copy_30_ptr) + ((((r134_z / $(256)) % $(128)) * $(256)) + (r134_z % $(256))), dtype=tl.float32) * tl.cast(tl.load($(vto_copy_1_ptr) + (r134_z % $(256)), dtype=tl.bfloat16), tl.float32))
    r134_value = r134_value.to(tl.float32)
    r134_value = tl.broadcast_to(r134_value, [$(32768)])
    tl.store($(vmul_42_ptr) + r134_z, r134_value)
    r135_z = tl.arange($(32768))
    r135_value = (tl.load($(vmul_42_ptr) + ((((r135_z / $(256)) % $(128)) * $(256)) + (r135_z % $(256))), dtype=tl.float32) * tl.load($(vto_copy_20_ptr) + ((((r135_z / $(256)) % $(128)) * $(256)) + (r135_z % $(256))), dtype=tl.float32))
    r135_value = r135_value.to(tl.float32)
    r135_value = tl.broadcast_to(r135_value, [$(32768)])
    tl.store($(vmul_43_ptr) + r135_z, r135_value)
    r136_z = tl.arange($(32768))
    r136_value = (tl.load($(vmul_42_ptr) + ((((r136_z / $(256)) % $(128)) * $(256)) + (r136_z % $(256))), dtype=tl.float32) * tl.load($(vrsqrt_ptr) + ((r136_z / $(256)) % $(128)), dtype=tl.float32))
    r136_value = r136_value.to(tl.float32)
    r136_value = tl.broadcast_to(r136_value, [$(32768)])
    tl.store($(vmul_44_ptr) + r136_z, r136_value)
    r137_z = tl.arange($(128))
    r137_lane = tl.arange($(256))
    r137_value = tl.sum(tl.load($(vmul_43_ptr) + ((r137_z[:, None] * $(256)) + r137_lane[None, :]), dtype=tl.float32), axis=1)
    r137_value = r137_value.to(tl.float32)
    r137_value = tl.broadcast_to(r137_value, [$(128)])
    tl.store($(vsum_3_ptr) + r137_z, r137_value)
    r138_z = tl.arange($(128))
    r138_value = ((tl.load($(vrsqrt_ptr) + r138_z, dtype=tl.float32)) * (tl.load($(vrsqrt_ptr) + r138_z, dtype=tl.float32)) * (tl.load($(vrsqrt_ptr) + r138_z, dtype=tl.float32)))
    r138_value = r138_value.to(tl.float32)
    r138_value = tl.broadcast_to(r138_value, [$(128)])
    tl.store($(vpow_7_ptr) + r138_z, r138_value)
    r139_z = tl.arange($(128))
    r139_value = (tl.load($(vsum_3_ptr) + r139_z, dtype=tl.float32) * -0.5)
    r139_value = r139_value.to(tl.float32)
    r139_value = tl.broadcast_to(r139_value, [$(128)])
    tl.store($(vmul_45_ptr) + r139_z, r139_value)
    r140_z = tl.arange($(128))
    r140_value = (tl.load($(vmul_45_ptr) + r140_z, dtype=tl.float32) * tl.load($(vpow_7_ptr) + r140_z, dtype=tl.float32))
    r140_value = r140_value.to(tl.float32)
    r140_value = tl.broadcast_to(r140_value, [$(128)])
    tl.store($(vmul_46_ptr) + r140_z, r140_value)
    r141_z = tl.arange($(32768))
    r141_value = (tl.load($(vmul_46_ptr) + ((r141_z / $(256)) % $(128)), dtype=tl.float32) / 256.0)
    r141_value = r141_value.to(tl.float32)
    r141_value = tl.broadcast_to(r141_value, [$(32768)])
    tl.store($(vdiv_3_ptr) + r141_z, r141_value)
    r142_z = tl.arange($(32768))
    r142_value = ((tl.load($(vto_copy_20_ptr) + ((((r142_z / $(256)) % $(128)) * $(256)) + (r142_z % $(256))), dtype=tl.float32)))
    r142_value = r142_value.to(tl.float32)
    r142_value = tl.broadcast_to(r142_value, [$(32768)])
    tl.store($(vpow_8_ptr) + r142_z, r142_value)
    r143_z = tl.arange($(32768))
    r143_value = (tl.load($(vpow_8_ptr) + ((((r143_z / $(256)) % $(128)) * $(256)) + (r143_z % $(256))), dtype=tl.float32) * 2.0)
    r143_value = r143_value.to(tl.float32)
    r143_value = tl.broadcast_to(r143_value, [$(32768)])
    tl.store($(vmul_47_ptr) + r143_z, r143_value)
    r144_z = tl.arange($(32768))
    r144_value = (tl.load($(vdiv_3_ptr) + ((((r144_z / $(256)) % $(128)) * $(256)) + (r144_z % $(256))), dtype=tl.float32) * tl.load($(vmul_47_ptr) + ((((r144_z / $(256)) % $(128)) * $(256)) + (r144_z % $(256))), dtype=tl.float32))
    r144_value = r144_value.to(tl.float32)
    r144_value = tl.broadcast_to(r144_value, [$(32768)])
    tl.store($(vmul_48_ptr) + r144_z, r144_value)
    r145_z = tl.arange($(32768))
    r145_value = (tl.load($(vmul_44_ptr) + ((((r145_z / $(256)) % $(128)) * $(256)) + (r145_z % $(256))), dtype=tl.float32) + tl.load($(vmul_48_ptr) + ((((r145_z / $(256)) % $(128)) * $(256)) + (r145_z % $(256))), dtype=tl.float32))
    r145_value = r145_value.to(tl.float32)
    r145_value = tl.broadcast_to(r145_value, [$(32768)])
    tl.store($(vadd_15_ptr) + r145_z, r145_value)
    r146_z = tl.arange($(32768))
    r146_value = tl.load($(vadd_15_ptr) + ((((r146_z / $(256)) % $(128)) * $(256)) + (r146_z % $(256))), dtype=tl.float32)
    r146_value = r146_value.to(tl.bfloat16)
    r146_value = tl.broadcast_to(r146_value, [$(32768)])
    tl.store($(vto_copy_31_ptr) + r146_z, r146_value)
    r147_row = tl.arange($(128))
    r147_col = tl.arange($(512))
    r147_lane = tl.arange($(256))
    r147_left = tl.load($(vto_copy_31_ptr) + ((((((r147_row[:, None] * $(256)) + r147_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r147_row[:, None] * $(256)) + r147_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r147_right = tl.load($(vto_copy_ptr) + (((r147_lane[:, None] % $(256)) * $(512)) + (r147_col[None, :] % $(512))), dtype=tl.bfloat16)
    r147_value = tl.dot(r147_left, r147_right)
    r147_value = r147_value.to(tl.bfloat16)
    tl.store($(vmm_9_ptr) + ((r147_row[:, None] * $(512)) + r147_col[None, :]), r147_value)
    r148_z = tl.arange($(65536))
    r148_value = (tl.load($(vmm_7_ptr) + ((((r148_z / $(512)) % $(128)) * $(512)) + (r148_z % $(512))), dtype=tl.bfloat16) + tl.load($(vmm_9_ptr) + ((((r148_z / $(512)) % $(128)) * $(512)) + (r148_z % $(512))), dtype=tl.bfloat16))
    r148_value = r148_value.to(tl.bfloat16)
    r148_value = tl.broadcast_to(r148_value, [$(65536)])
    tl.store($(vadd_16_ptr) + r148_z, r148_value)
    z = tl.arange($(65536))
    value = tl.load($(vadd_16_ptr) + ((((z / $(512)) % $(128)) * $(512)) + (z % $(512))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(65536)])
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
  let vadd_3_ptr : RegionName := "vadd_3"
  let vrsqrt_ptr : RegionName := "vrsqrt"
  let vmul_13_ptr : RegionName := "vmul_13"
  let vmul_14_ptr : RegionName := "vmul_14"
  let vto_copy_21_ptr : RegionName := "vto_copy_21"
  let vmm_1_ptr : RegionName := "vmm_1"
  let vmm_2_ptr : RegionName := "vmm_2"
  let vto_copy_22_ptr : RegionName := "vto_copy_22"
  let vpow_4_ptr : RegionName := "vpow_4"
  let vmean_1_ptr : RegionName := "vmean_1"
  let vadd_4_ptr : RegionName := "vadd_4"
  let vrsqrt_1_ptr : RegionName := "vrsqrt_1"
  let vmul_15_ptr : RegionName := "vmul_15"
  let vmul_16_ptr : RegionName := "vmul_16"
  let vto_copy_23_ptr : RegionName := "vto_copy_23"
  let vmul_17_ptr : RegionName := "vmul_17"
  let vmul_18_ptr : RegionName := "vmul_18"
  let vsub_4_ptr : RegionName := "vsub_4"
  let vmul_19_ptr : RegionName := "vmul_19"
  let vmul_20_ptr : RegionName := "vmul_20"
  let vadd_5_ptr : RegionName := "vadd_5"
  let vcat_5_ptr : RegionName := "vcat_5"
  let vmul_21_ptr : RegionName := "vmul_21"
  let vmul_22_ptr : RegionName := "vmul_22"
  let vsub_5_ptr : RegionName := "vsub_5"
  let vmul_23_ptr : RegionName := "vmul_23"
  let vmul_24_ptr : RegionName := "vmul_24"
  let vadd_6_ptr : RegionName := "vadd_6"
  let vcat_6_ptr : RegionName := "vcat_6"
  let vcat_7_ptr : RegionName := "vcat_7"
  let vcat_8_ptr : RegionName := "vcat_8"
  let vbmm_2_ptr : RegionName := "vbmm_2"
  let vmul_25_ptr : RegionName := "vmul_25"
  let vadd_7_ptr : RegionName := "vadd_7"
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
  let vadd_8_ptr : RegionName := "vadd_8"
  let vmul_30_ptr : RegionName := "vmul_30"
  let vadd_9_ptr : RegionName := "vadd_9"
  let vmul_31_ptr : RegionName := "vmul_31"
  let vmul_32_ptr : RegionName := "vmul_32"
  let vneg_1_ptr : RegionName := "vneg_1"
  let vmul_33_ptr : RegionName := "vmul_33"
  let vadd_10_ptr : RegionName := "vadd_10"
  let vmul_34_ptr : RegionName := "vmul_34"
  let vadd_11_ptr : RegionName := "vadd_11"
  let vslice_backward_ptr : RegionName := "vslice_backward"
  let vslice_backward_1_ptr : RegionName := "vslice_backward_1"
  let vadd_12_ptr : RegionName := "vadd_12"
  let vslice_backward_2_ptr : RegionName := "vslice_backward_2"
  let vslice_backward_3_ptr : RegionName := "vslice_backward_3"
  let vadd_13_ptr : RegionName := "vadd_13"
  let vcat_10_ptr : RegionName := "vcat_10"
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
  let vadd_14_ptr : RegionName := "vadd_14"
  let vto_copy_29_ptr : RegionName := "vto_copy_29"
  let vcat_11_ptr : RegionName := "vcat_11"
  let vmm_7_ptr : RegionName := "vmm_7"
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
  let vadd_15_ptr : RegionName := "vadd_15"
  let vto_copy_31_ptr : RegionName := "vto_copy_31"
  let vadd_16_ptr : RegionName := "vadd_16"
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
  let candidate_vadd_3_ptr : RegionName := "candidate_vadd_3"
  let candidate_vrsqrt_ptr : RegionName := "candidate_vrsqrt"
  let candidate_vmul_13_ptr : RegionName := "candidate_vmul_13"
  let candidate_vmul_14_ptr : RegionName := "candidate_vmul_14"
  let candidate_vto_copy_21_ptr : RegionName := "candidate_vto_copy_21"
  let candidate_vmm_1_ptr : RegionName := "candidate_vmm_1"
  let candidate_vmm_2_ptr : RegionName := "candidate_vmm_2"
  let candidate_vto_copy_22_ptr : RegionName := "candidate_vto_copy_22"
  let candidate_vpow_4_ptr : RegionName := "candidate_vpow_4"
  let candidate_vmean_1_ptr : RegionName := "candidate_vmean_1"
  let candidate_vadd_4_ptr : RegionName := "candidate_vadd_4"
  let candidate_vrsqrt_1_ptr : RegionName := "candidate_vrsqrt_1"
  let candidate_vmul_15_ptr : RegionName := "candidate_vmul_15"
  let candidate_vmul_16_ptr : RegionName := "candidate_vmul_16"
  let candidate_vto_copy_23_ptr : RegionName := "candidate_vto_copy_23"
  let candidate_vdiv_2_ptr : RegionName := "candidate_vdiv_2"
  let candidate_vpow_5_ptr : RegionName := "candidate_vpow_5"
  let candidate_vreciprocal_2_ptr : RegionName := "candidate_vreciprocal_2"
  let candidate_vmul_17_ptr : RegionName := "candidate_vmul_17"
  let candidate_vmul_18_ptr : RegionName := "candidate_vmul_18"
  let candidate_vcat_5_ptr : RegionName := "candidate_vcat_5"
  let candidate_vcos_2_ptr : RegionName := "candidate_vcos_2"
  let candidate_vto_copy_24_ptr : RegionName := "candidate_vto_copy_24"
  let candidate_vsin_2_ptr : RegionName := "candidate_vsin_2"
  let candidate_vto_copy_25_ptr : RegionName := "candidate_vto_copy_25"
  let candidate_vneg_ptr : RegionName := "candidate_vneg"
  let candidate_vstack_ptr : RegionName := "candidate_vstack"
  let candidate_vmul_19_ptr : RegionName := "candidate_vmul_19"
  let candidate_vmul_20_ptr : RegionName := "candidate_vmul_20"
  let candidate_vadd_5_ptr : RegionName := "candidate_vadd_5"
  let candidate_vcat_6_ptr : RegionName := "candidate_vcat_6"
  let candidate_vcos_3_ptr : RegionName := "candidate_vcos_3"
  let candidate_vto_copy_26_ptr : RegionName := "candidate_vto_copy_26"
  let candidate_vsin_3_ptr : RegionName := "candidate_vsin_3"
  let candidate_vto_copy_27_ptr : RegionName := "candidate_vto_copy_27"
  let candidate_vneg_1_ptr : RegionName := "candidate_vneg_1"
  let candidate_vstack_1_ptr : RegionName := "candidate_vstack_1"
  let candidate_vmul_21_ptr : RegionName := "candidate_vmul_21"
  let candidate_vmul_22_ptr : RegionName := "candidate_vmul_22"
  let candidate_vadd_6_ptr : RegionName := "candidate_vadd_6"
  let candidate_vbmm_2_ptr : RegionName := "candidate_vbmm_2"
  let candidate_vbmm_3_ptr : RegionName := "candidate_vbmm_3"
  let candidate_vcat_7_ptr : RegionName := "candidate_vcat_7"
  let candidate_vcat_8_ptr : RegionName := "candidate_vcat_8"
  let candidate_vto_copy_28_ptr : RegionName := "candidate_vto_copy_28"
  let candidate_vto_copy_29_ptr : RegionName := "candidate_vto_copy_29"
  let candidate_vbmm_4_ptr : RegionName := "candidate_vbmm_4"
  let candidate_vmul_23_ptr : RegionName := "candidate_vmul_23"
  let candidate_vadd_7_ptr : RegionName := "candidate_vadd_7"
  let candidate_vsub_4_ptr : RegionName := "candidate_vsub_4"
  let candidate_vgt_ptr : RegionName := "candidate_vgt"
  let candidate_vmasked_fill_ptr : RegionName := "candidate_vmasked_fill"
  let candidate_vsoftmax_ptr : RegionName := "candidate_vsoftmax"
  let candidate_vto_copy_30_ptr : RegionName := "candidate_vto_copy_30"
  let candidate_vto_copy_31_ptr : RegionName := "candidate_vto_copy_31"
  let candidate_vto_copy_32_ptr : RegionName := "candidate_vto_copy_32"
  let candidate_vbmm_6_ptr : RegionName := "candidate_vbmm_6"
  let candidate_vmul_24_ptr : RegionName := "candidate_vmul_24"
  let candidate_vadd_8_ptr : RegionName := "candidate_vadd_8"
  let candidate_vsub_5_ptr : RegionName := "candidate_vsub_5"
  let candidate_vgt_1_ptr : RegionName := "candidate_vgt_1"
  let candidate_vmasked_fill_1_ptr : RegionName := "candidate_vmasked_fill_1"
  let candidate_vsoftmax_1_ptr : RegionName := "candidate_vsoftmax_1"
  let candidate_vto_copy_33_ptr : RegionName := "candidate_vto_copy_33"
  let candidate_vto_copy_34_ptr : RegionName := "candidate_vto_copy_34"
  let candidate_vto_copy_35_ptr : RegionName := "candidate_vto_copy_35"
  let candidate_vbmm_8_ptr : RegionName := "candidate_vbmm_8"
  let candidate_vmul_25_ptr : RegionName := "candidate_vmul_25"
  let candidate_vadd_9_ptr : RegionName := "candidate_vadd_9"
  let candidate_vsub_6_ptr : RegionName := "candidate_vsub_6"
  let candidate_vgt_2_ptr : RegionName := "candidate_vgt_2"
  let candidate_vmasked_fill_2_ptr : RegionName := "candidate_vmasked_fill_2"
  let candidate_vsoftmax_2_ptr : RegionName := "candidate_vsoftmax_2"
  let candidate_vto_copy_36_ptr : RegionName := "candidate_vto_copy_36"
  let candidate_vmm_4_ptr : RegionName := "candidate_vmm_4"
  let candidate_vbmm_10_ptr : RegionName := "candidate_vbmm_10"
  let candidate_vbmm_11_ptr : RegionName := "candidate_vbmm_11"
  let candidate_vto_copy_37_ptr : RegionName := "candidate_vto_copy_37"
  let candidate_vsoftmax_backward_data_ptr : RegionName := "candidate_vsoftmax_backward_data"
  let candidate_vmasked_fill_3_ptr : RegionName := "candidate_vmasked_fill_3"
  let candidate_vmul_26_ptr : RegionName := "candidate_vmul_26"
  let candidate_vbmm_12_ptr : RegionName := "candidate_vbmm_12"
  let candidate_vbmm_13_ptr : RegionName := "candidate_vbmm_13"
  let candidate_vslice_backward_ptr : RegionName := "candidate_vslice_backward"
  let candidate_vto_copy_38_ptr : RegionName := "candidate_vto_copy_38"
  let candidate_vslice_backward_1_ptr : RegionName := "candidate_vslice_backward_1"
  let candidate_vto_copy_39_ptr : RegionName := "candidate_vto_copy_39"
  let candidate_vslice_backward_2_ptr : RegionName := "candidate_vslice_backward_2"
  let candidate_vbmm_14_ptr : RegionName := "candidate_vbmm_14"
  let candidate_vbmm_15_ptr : RegionName := "candidate_vbmm_15"
  let candidate_vto_copy_40_ptr : RegionName := "candidate_vto_copy_40"
  let candidate_vsoftmax_backward_data_1_ptr : RegionName := "candidate_vsoftmax_backward_data_1"
  let candidate_vmasked_fill_4_ptr : RegionName := "candidate_vmasked_fill_4"
  let candidate_vmul_27_ptr : RegionName := "candidate_vmul_27"
  let candidate_vbmm_16_ptr : RegionName := "candidate_vbmm_16"
  let candidate_vbmm_17_ptr : RegionName := "candidate_vbmm_17"
  let candidate_vslice_backward_3_ptr : RegionName := "candidate_vslice_backward_3"
  let candidate_vadd_10_ptr : RegionName := "candidate_vadd_10"
  let candidate_vto_copy_41_ptr : RegionName := "candidate_vto_copy_41"
  let candidate_vslice_backward_4_ptr : RegionName := "candidate_vslice_backward_4"
  let candidate_vadd_11_ptr : RegionName := "candidate_vadd_11"
  let candidate_vto_copy_42_ptr : RegionName := "candidate_vto_copy_42"
  let candidate_vslice_backward_5_ptr : RegionName := "candidate_vslice_backward_5"
  let candidate_vadd_12_ptr : RegionName := "candidate_vadd_12"
  let candidate_vbmm_18_ptr : RegionName := "candidate_vbmm_18"
  let candidate_vbmm_19_ptr : RegionName := "candidate_vbmm_19"
  let candidate_vto_copy_43_ptr : RegionName := "candidate_vto_copy_43"
  let candidate_vsoftmax_backward_data_2_ptr : RegionName := "candidate_vsoftmax_backward_data_2"
  let candidate_vmasked_fill_5_ptr : RegionName := "candidate_vmasked_fill_5"
  let candidate_vmul_28_ptr : RegionName := "candidate_vmul_28"
  let candidate_vbmm_20_ptr : RegionName := "candidate_vbmm_20"
  let candidate_vbmm_21_ptr : RegionName := "candidate_vbmm_21"
  let candidate_vslice_backward_6_ptr : RegionName := "candidate_vslice_backward_6"
  let candidate_vadd_13_ptr : RegionName := "candidate_vadd_13"
  let candidate_vto_copy_44_ptr : RegionName := "candidate_vto_copy_44"
  let candidate_vslice_backward_7_ptr : RegionName := "candidate_vslice_backward_7"
  let candidate_vadd_14_ptr : RegionName := "candidate_vadd_14"
  let candidate_vto_copy_45_ptr : RegionName := "candidate_vto_copy_45"
  let candidate_vslice_backward_8_ptr : RegionName := "candidate_vslice_backward_8"
  let candidate_vadd_15_ptr : RegionName := "candidate_vadd_15"
  let candidate_vsum_1_ptr : RegionName := "candidate_vsum_1"
  let candidate_vbmm_22_ptr : RegionName := "candidate_vbmm_22"
  let candidate_vbmm_23_ptr : RegionName := "candidate_vbmm_23"
  let candidate_vadd_16_ptr : RegionName := "candidate_vadd_16"
  let candidate_vselect_backward_ptr : RegionName := "candidate_vselect_backward"
  let candidate_vmul_29_ptr : RegionName := "candidate_vmul_29"
  let candidate_vmul_30_ptr : RegionName := "candidate_vmul_30"
  let candidate_vselect_backward_1_ptr : RegionName := "candidate_vselect_backward_1"
  let candidate_vneg_2_ptr : RegionName := "candidate_vneg_2"
  let candidate_vselect_backward_2_ptr : RegionName := "candidate_vselect_backward_2"
  let candidate_vadd_17_ptr : RegionName := "candidate_vadd_17"
  let candidate_vadd_18_ptr : RegionName := "candidate_vadd_18"
  let candidate_vmul_31_ptr : RegionName := "candidate_vmul_31"
  let candidate_vmul_32_ptr : RegionName := "candidate_vmul_32"
  let candidate_vselect_backward_3_ptr : RegionName := "candidate_vselect_backward_3"
  let candidate_vneg_3_ptr : RegionName := "candidate_vneg_3"
  let candidate_vselect_backward_4_ptr : RegionName := "candidate_vselect_backward_4"
  let candidate_vadd_19_ptr : RegionName := "candidate_vadd_19"
  let candidate_vadd_20_ptr : RegionName := "candidate_vadd_20"
  let candidate_vcat_10_ptr : RegionName := "candidate_vcat_10"
  let candidate_vto_copy_46_ptr : RegionName := "candidate_vto_copy_46"
  let candidate_vmul_33_ptr : RegionName := "candidate_vmul_33"
  let candidate_vmul_34_ptr : RegionName := "candidate_vmul_34"
  let candidate_vmul_35_ptr : RegionName := "candidate_vmul_35"
  let candidate_vsum_2_ptr : RegionName := "candidate_vsum_2"
  let candidate_vpow_6_ptr : RegionName := "candidate_vpow_6"
  let candidate_vmul_36_ptr : RegionName := "candidate_vmul_36"
  let candidate_vmul_37_ptr : RegionName := "candidate_vmul_37"
  let candidate_vdiv_3_ptr : RegionName := "candidate_vdiv_3"
  let candidate_vpow_7_ptr : RegionName := "candidate_vpow_7"
  let candidate_vmul_38_ptr : RegionName := "candidate_vmul_38"
  let candidate_vmul_39_ptr : RegionName := "candidate_vmul_39"
  let candidate_vadd_21_ptr : RegionName := "candidate_vadd_21"
  let candidate_vto_copy_47_ptr : RegionName := "candidate_vto_copy_47"
  let candidate_vcat_11_ptr : RegionName := "candidate_vcat_11"
  let candidate_vmm_5_ptr : RegionName := "candidate_vmm_5"
  let candidate_vmm_6_ptr : RegionName := "candidate_vmm_6"
  let candidate_vto_copy_48_ptr : RegionName := "candidate_vto_copy_48"
  let candidate_vmul_40_ptr : RegionName := "candidate_vmul_40"
  let candidate_vmul_41_ptr : RegionName := "candidate_vmul_41"
  let candidate_vmul_42_ptr : RegionName := "candidate_vmul_42"
  let candidate_vsum_3_ptr : RegionName := "candidate_vsum_3"
  let candidate_vpow_8_ptr : RegionName := "candidate_vpow_8"
  let candidate_vmul_43_ptr : RegionName := "candidate_vmul_43"
  let candidate_vmul_44_ptr : RegionName := "candidate_vmul_44"
  let candidate_vdiv_4_ptr : RegionName := "candidate_vdiv_4"
  let candidate_vpow_9_ptr : RegionName := "candidate_vpow_9"
  let candidate_vmul_45_ptr : RegionName := "candidate_vmul_45"
  let candidate_vmul_46_ptr : RegionName := "candidate_vmul_46"
  let candidate_vadd_22_ptr : RegionName := "candidate_vadd_22"
  let candidate_vto_copy_49_ptr : RegionName := "candidate_vto_copy_49"
  let candidate_vmm_7_ptr : RegionName := "candidate_vmm_7"
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
    r16_row = tl.arange($(128))
    r16_col = tl.arange($(256))
    r16_lane = tl.arange($(512))
    r16_left = tl.load($(input0_ptr) + ((((((r16_row[:, None] * $(512)) + r16_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r16_row[:, None] * $(512)) + r16_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r16_right = tl.load($(candidate_vto_copy_ptr) + (((r16_col[None, :] % $(256)) * $(512)) + (r16_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r16_value = tl.dot(r16_left, r16_right)
    r16_value = r16_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_ptr) + ((r16_row[:, None] * $(256)) + r16_col[None, :]), r16_value)
    r17_z = tl.arange($(32768))
    r17_value = tl.load($(candidate_vmm_ptr) + ((((r17_z / $(256)) % $(128)) * $(256)) + (r17_z % $(256))), dtype=tl.bfloat16)
    r17_value = r17_value.to(tl.float32)
    r17_value = tl.broadcast_to(r17_value, [$(32768)])
    tl.store($(candidate_vto_copy_20_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(32768))
    r18_value = ((tl.load($(candidate_vto_copy_20_ptr) + ((((r18_z / $(256)) % $(128)) * $(256)) + (r18_z % $(256))), dtype=tl.float32)) * (tl.load($(candidate_vto_copy_20_ptr) + ((((r18_z / $(256)) % $(128)) * $(256)) + (r18_z % $(256))), dtype=tl.float32)))
    r18_value = r18_value.to(tl.float32)
    r18_value = tl.broadcast_to(r18_value, [$(32768)])
    tl.store($(candidate_vpow_3_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(128))
    r19_lane = tl.arange($(256))
    r19_value = (tl.sum(tl.load($(candidate_vpow_3_ptr) + ((r19_z[:, None] * $(256)) + r19_lane[None, :]), dtype=tl.float32), axis=1) / 256.0)
    r19_value = r19_value.to(tl.float32)
    r19_value = tl.broadcast_to(r19_value, [$(128)])
    tl.store($(candidate_vmean_ptr) + r19_z, r19_value)
    r20_z = tl.arange($(128))
    r20_value = (tl.load($(candidate_vmean_ptr) + r20_z, dtype=tl.float32) + 0.000001)
    r20_value = r20_value.to(tl.float32)
    r20_value = tl.broadcast_to(r20_value, [$(128)])
    tl.store($(candidate_vadd_3_ptr) + r20_z, r20_value)
    r21_z = tl.arange($(128))
    r21_value = tl.rsqrt(tl.load($(candidate_vadd_3_ptr) + r21_z, dtype=tl.float32))
    r21_value = r21_value.to(tl.float32)
    r21_value = tl.broadcast_to(r21_value, [$(128)])
    tl.store($(candidate_vrsqrt_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(32768))
    r22_value = (tl.load($(candidate_vto_copy_20_ptr) + ((((r22_z / $(256)) % $(128)) * $(256)) + (r22_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_ptr) + ((r22_z / $(256)) % $(128)), dtype=tl.float32))
    r22_value = r22_value.to(tl.float32)
    r22_value = tl.broadcast_to(r22_value, [$(32768)])
    tl.store($(candidate_vmul_13_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(32768))
    r23_value = (tl.cast(tl.load($(candidate_vto_copy_1_ptr) + (r23_z % $(256)), dtype=tl.bfloat16), tl.float32) * tl.load($(candidate_vmul_13_ptr) + ((((r23_z / $(256)) % $(128)) * $(256)) + (r23_z % $(256))), dtype=tl.float32))
    r23_value = r23_value.to(tl.float32)
    r23_value = tl.broadcast_to(r23_value, [$(32768)])
    tl.store($(candidate_vmul_14_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(32768))
    r24_value = tl.load($(candidate_vmul_14_ptr) + ((((r24_z / $(256)) % $(128)) * $(256)) + (r24_z % $(256))), dtype=tl.float32)
    r24_value = r24_value.to(tl.bfloat16)
    r24_value = tl.broadcast_to(r24_value, [$(32768)])
    tl.store($(candidate_vto_copy_21_ptr) + r24_z, r24_value)
    r25_row = tl.arange($(128))
    r25_col = tl.arange($(1536))
    r25_lane = tl.arange($(256))
    r25_left = tl.load($(candidate_vto_copy_21_ptr) + ((((((r25_row[:, None] * $(256)) + r25_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r25_row[:, None] * $(256)) + r25_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r25_right = tl.load($(candidate_vto_copy_2_ptr) + (((r25_col[None, :] % $(1536)) * $(256)) + (r25_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r25_value = tl.dot(r25_left, r25_right)
    r25_value = r25_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_1_ptr) + ((r25_row[:, None] * $(1536)) + r25_col[None, :]), r25_value)
    r26_row = tl.arange($(128))
    r26_col = tl.arange($(320))
    r26_lane = tl.arange($(512))
    r26_left = tl.load($(input0_ptr) + ((((((r26_row[:, None] * $(512)) + r26_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r26_row[:, None] * $(512)) + r26_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r26_right = tl.load($(candidate_vto_copy_3_ptr) + (((r26_col[None, :] % $(320)) * $(512)) + (r26_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r26_value = tl.dot(r26_left, r26_right)
    r26_value = r26_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_2_ptr) + ((r26_row[:, None] * $(320)) + r26_col[None, :]), r26_value)
    r27_z = tl.arange($(32768))
    r27_value = tl.load($(candidate_vmm_2_ptr) + ((((r27_z / $(256)) % $(128)) * $(320)) + (r27_z % $(256))), dtype=tl.bfloat16)
    r27_value = r27_value.to(tl.float32)
    r27_value = tl.broadcast_to(r27_value, [$(32768)])
    tl.store($(candidate_vto_copy_22_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(32768))
    r28_value = ((tl.load($(candidate_vto_copy_22_ptr) + ((((r28_z / $(256)) % $(128)) * $(256)) + (r28_z % $(256))), dtype=tl.float32)) * (tl.load($(candidate_vto_copy_22_ptr) + ((((r28_z / $(256)) % $(128)) * $(256)) + (r28_z % $(256))), dtype=tl.float32)))
    r28_value = r28_value.to(tl.float32)
    r28_value = tl.broadcast_to(r28_value, [$(32768)])
    tl.store($(candidate_vpow_4_ptr) + r28_z, r28_value)
    r29_z = tl.arange($(128))
    r29_lane = tl.arange($(256))
    r29_value = (tl.sum(tl.load($(candidate_vpow_4_ptr) + ((r29_z[:, None] * $(256)) + r29_lane[None, :]), dtype=tl.float32), axis=1) / 256.0)
    r29_value = r29_value.to(tl.float32)
    r29_value = tl.broadcast_to(r29_value, [$(128)])
    tl.store($(candidate_vmean_1_ptr) + r29_z, r29_value)
    r30_z = tl.arange($(128))
    r30_value = (tl.load($(candidate_vmean_1_ptr) + r30_z, dtype=tl.float32) + 0.000001)
    r30_value = r30_value.to(tl.float32)
    r30_value = tl.broadcast_to(r30_value, [$(128)])
    tl.store($(candidate_vadd_4_ptr) + r30_z, r30_value)
    r31_z = tl.arange($(128))
    r31_value = tl.rsqrt(tl.load($(candidate_vadd_4_ptr) + r31_z, dtype=tl.float32))
    r31_value = r31_value.to(tl.float32)
    r31_value = tl.broadcast_to(r31_value, [$(128)])
    tl.store($(candidate_vrsqrt_1_ptr) + r31_z, r31_value)
    r32_z = tl.arange($(32768))
    r32_value = (tl.load($(candidate_vto_copy_22_ptr) + ((((r32_z / $(256)) % $(128)) * $(256)) + (r32_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_1_ptr) + ((r32_z / $(256)) % $(128)), dtype=tl.float32))
    r32_value = r32_value.to(tl.float32)
    r32_value = tl.broadcast_to(r32_value, [$(32768)])
    tl.store($(candidate_vmul_15_ptr) + r32_z, r32_value)
    r33_z = tl.arange($(32768))
    r33_value = (tl.cast(tl.load($(candidate_vto_copy_4_ptr) + (r33_z % $(256)), dtype=tl.bfloat16), tl.float32) * tl.load($(candidate_vmul_15_ptr) + ((((r33_z / $(256)) % $(128)) * $(256)) + (r33_z % $(256))), dtype=tl.float32))
    r33_value = r33_value.to(tl.float32)
    r33_value = tl.broadcast_to(r33_value, [$(32768)])
    tl.store($(candidate_vmul_16_ptr) + r33_z, r33_value)
    r34_z = tl.arange($(32768))
    r34_value = tl.load($(candidate_vmul_16_ptr) + ((((r34_z / $(256)) % $(128)) * $(256)) + (r34_z % $(256))), dtype=tl.float32)
    r34_value = r34_value.to(tl.bfloat16)
    r34_value = tl.broadcast_to(r34_value, [$(32768)])
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
    r39_z = tl.arange($(2048))
    r39_value = (tl.cast(tl.toReal(((r39_z / $(32)) % $(64))), tl.float32) * tl.load($(candidate_vmul_17_ptr) + (r39_z % $(32)), dtype=tl.float32))
    r39_value = r39_value.to(tl.float32)
    r39_value = tl.broadcast_to(r39_value, [$(2048)])
    tl.store($(candidate_vmul_18_ptr) + r39_z, r39_value)
    r40_z = tl.arange($(8192))
    r40_value = tl.where(((r40_z / $(64)) % $(128)) < $(48), tl.load($(candidate_vmul_18_ptr) + ((((((r40_z / $(64)) % $(128)) - $(0)) % $(64)) * $(32)) + ((((((((r40_z / $(64)) % $(128)) - $(0)) % $(64)) * $(64)) + (r40_z % $(64))) / $(2)) % $(32))), dtype=tl.float32), tl.where(((r40_z / $(64)) % $(128)) < $(64), tl.load($(candidate_vmul_18_ptr) + ((((((r40_z / $(64)) % $(128)) - $(48)) % $(64)) * $(32)) + ((((((((r40_z / $(64)) % $(128)) - $(48)) % $(64)) * $(64)) + (r40_z % $(64))) / $(2)) % $(32))), dtype=tl.float32), tl.load($(candidate_vmul_18_ptr) + ((((((r40_z / $(64)) % $(128)) - $(64)) % $(64)) * $(32)) + ((((((((r40_z / $(64)) % $(128)) - $(64)) % $(64)) * $(64)) + (r40_z % $(64))) / $(2)) % $(32))), dtype=tl.float32)))
    r40_value = r40_value.to(tl.float32)
    r40_value = tl.broadcast_to(r40_value, [$(8192)])
    tl.store($(candidate_vcat_5_ptr) + r40_z, r40_value)
    r41_z = tl.arange($(8192))
    r41_value = tl.cos(tl.load($(candidate_vcat_5_ptr) + ((((r41_z / $(64)) % $(128)) * $(64)) + (r41_z % $(64))), dtype=tl.float32))
    r41_value = r41_value.to(tl.float32)
    r41_value = tl.broadcast_to(r41_value, [$(8192)])
    tl.store($(candidate_vcos_2_ptr) + r41_z, r41_value)
    r42_z = tl.arange($(8192))
    r42_value = tl.load($(candidate_vcos_2_ptr) + ((((r42_z / $(64)) % $(128)) * $(64)) + (r42_z % $(64))), dtype=tl.float32)
    r42_value = r42_value.to(tl.bfloat16)
    r42_value = tl.broadcast_to(r42_value, [$(8192)])
    tl.store($(candidate_vto_copy_24_ptr) + r42_z, r42_value)
    r43_z = tl.arange($(8192))
    r43_value = tl.sin(tl.load($(candidate_vcat_5_ptr) + ((((r43_z / $(64)) % $(128)) * $(64)) + (r43_z % $(64))), dtype=tl.float32))
    r43_value = r43_value.to(tl.float32)
    r43_value = tl.broadcast_to(r43_value, [$(8192)])
    tl.store($(candidate_vsin_2_ptr) + r43_z, r43_value)
    r44_z = tl.arange($(8192))
    r44_value = tl.load($(candidate_vsin_2_ptr) + ((((r44_z / $(64)) % $(128)) * $(64)) + (r44_z % $(64))), dtype=tl.float32)
    r44_value = r44_value.to(tl.bfloat16)
    r44_value = tl.broadcast_to(r44_value, [$(8192)])
    tl.store($(candidate_vto_copy_25_ptr) + r44_z, r44_value)
    r45_z = tl.arange($(32768))
    r45_value = -tl.load($(candidate_vmm_1_ptr) + ((((r45_z / $(256)) % $(128)) * $(1536)) + (((r45_z / $(32)) % $(8)) * $(192)) + ((r45_z % $(32)) * $(2)) + $(129)), dtype=tl.bfloat16)
    r45_value = r45_value.to(tl.bfloat16)
    r45_value = tl.broadcast_to(r45_value, [$(32768)])
    tl.store($(candidate_vneg_ptr) + r45_z, r45_value)
    r46_z = tl.arange($(65536))
    r46_value = tl.where((r46_z % $(2)) < $(1), tl.load($(candidate_vneg_ptr) + ((((r46_z / $(512)) % $(128)) * $(256)) + (((r46_z / $(64)) % $(8)) * $(32)) + ((r46_z / $(2)) % $(32))), dtype=tl.bfloat16), tl.load($(candidate_vmm_1_ptr) + ((((r46_z / $(512)) % $(128)) * $(1536)) + (((r46_z / $(64)) % $(8)) * $(192)) + (((r46_z / $(2)) % $(32)) * $(2)) + $(128)), dtype=tl.bfloat16))
    r46_value = r46_value.to(tl.bfloat16)
    r46_value = tl.broadcast_to(r46_value, [$(65536)])
    tl.store($(candidate_vstack_ptr) + r46_z, r46_value)
    r47_z = tl.arange($(65536))
    r47_value = (tl.load($(candidate_vmm_1_ptr) + ((((r47_z / $(512)) % $(128)) * $(1536)) + (((r47_z / $(64)) % $(8)) * $(192)) + (r47_z % $(64)) + $(128)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_24_ptr) + ((((r47_z / $(512)) % $(128)) * $(64)) + (r47_z % $(64))), dtype=tl.bfloat16))
    r47_value = r47_value.to(tl.bfloat16)
    r47_value = tl.broadcast_to(r47_value, [$(65536)])
    tl.store($(candidate_vmul_19_ptr) + r47_z, r47_value)
    r48_z = tl.arange($(65536))
    r48_value = (tl.load($(candidate_vstack_ptr) + ((((r48_z / $(512)) % $(128)) * $(512)) + (((r48_z / $(64)) % $(8)) * $(64)) + (((((((r48_z / $(512)) % $(128)) * $(512)) + (((r48_z / $(64)) % $(8)) * $(64)) + (r48_z % $(64))) / $(2)) % $(32)) * $(2)) + (((((r48_z / $(512)) % $(128)) * $(512)) + (((r48_z / $(64)) % $(8)) * $(64)) + (r48_z % $(64))) % $(2))), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_25_ptr) + ((((r48_z / $(512)) % $(128)) * $(64)) + (r48_z % $(64))), dtype=tl.bfloat16))
    r48_value = r48_value.to(tl.bfloat16)
    r48_value = tl.broadcast_to(r48_value, [$(65536)])
    tl.store($(candidate_vmul_20_ptr) + r48_z, r48_value)
    r49_z = tl.arange($(65536))
    r49_value = (tl.load($(candidate_vmul_19_ptr) + ((((r49_z / $(512)) % $(128)) * $(512)) + (((r49_z / $(64)) % $(8)) * $(64)) + (r49_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_20_ptr) + ((((r49_z / $(512)) % $(128)) * $(512)) + (((r49_z / $(64)) % $(8)) * $(64)) + (r49_z % $(64))), dtype=tl.bfloat16))
    r49_value = r49_value.to(tl.bfloat16)
    r49_value = tl.broadcast_to(r49_value, [$(65536)])
    tl.store($(candidate_vadd_5_ptr) + r49_z, r49_value)
    r50_z = tl.arange($(8192))
    r50_value = tl.where(((r50_z / $(64)) % $(128)) < $(48), tl.load($(candidate_vmul_18_ptr) + ((((((r50_z / $(64)) % $(128)) - $(0)) % $(64)) * $(32)) + ((((((((r50_z / $(64)) % $(128)) - $(0)) % $(64)) * $(64)) + (r50_z % $(64))) / $(2)) % $(32))), dtype=tl.float32), tl.where(((r50_z / $(64)) % $(128)) < $(64), tl.load($(candidate_vmul_18_ptr) + ((((((r50_z / $(64)) % $(128)) - $(48)) % $(64)) * $(32)) + ((((((((r50_z / $(64)) % $(128)) - $(48)) % $(64)) * $(64)) + (r50_z % $(64))) / $(2)) % $(32))), dtype=tl.float32), tl.load($(candidate_vmul_18_ptr) + ((((((r50_z / $(64)) % $(128)) - $(64)) % $(64)) * $(32)) + ((((((((r50_z / $(64)) % $(128)) - $(64)) % $(64)) * $(64)) + (r50_z % $(64))) / $(2)) % $(32))), dtype=tl.float32)))
    r50_value = r50_value.to(tl.float32)
    r50_value = tl.broadcast_to(r50_value, [$(8192)])
    tl.store($(candidate_vcat_6_ptr) + r50_z, r50_value)
    r51_z = tl.arange($(8192))
    r51_value = tl.cos(tl.load($(candidate_vcat_6_ptr) + ((((r51_z / $(64)) % $(128)) * $(64)) + (r51_z % $(64))), dtype=tl.float32))
    r51_value = r51_value.to(tl.float32)
    r51_value = tl.broadcast_to(r51_value, [$(8192)])
    tl.store($(candidate_vcos_3_ptr) + r51_z, r51_value)
    r52_z = tl.arange($(8192))
    r52_value = tl.load($(candidate_vcos_3_ptr) + ((((r52_z / $(64)) % $(128)) * $(64)) + (r52_z % $(64))), dtype=tl.float32)
    r52_value = r52_value.to(tl.bfloat16)
    r52_value = tl.broadcast_to(r52_value, [$(8192)])
    tl.store($(candidate_vto_copy_26_ptr) + r52_z, r52_value)
    r53_z = tl.arange($(8192))
    r53_value = tl.sin(tl.load($(candidate_vcat_6_ptr) + ((((r53_z / $(64)) % $(128)) * $(64)) + (r53_z % $(64))), dtype=tl.float32))
    r53_value = r53_value.to(tl.float32)
    r53_value = tl.broadcast_to(r53_value, [$(8192)])
    tl.store($(candidate_vsin_3_ptr) + r53_z, r53_value)
    r54_z = tl.arange($(8192))
    r54_value = tl.load($(candidate_vsin_3_ptr) + ((((r54_z / $(64)) % $(128)) * $(64)) + (r54_z % $(64))), dtype=tl.float32)
    r54_value = r54_value.to(tl.bfloat16)
    r54_value = tl.broadcast_to(r54_value, [$(8192)])
    tl.store($(candidate_vto_copy_27_ptr) + r54_z, r54_value)
    r55_z = tl.arange($(4096))
    r55_value = -tl.load($(candidate_vmm_2_ptr) + ((((r55_z / $(32)) % $(128)) * $(320)) + ((r55_z % $(32)) * $(2)) + $(257)), dtype=tl.bfloat16)
    r55_value = r55_value.to(tl.bfloat16)
    r55_value = tl.broadcast_to(r55_value, [$(4096)])
    tl.store($(candidate_vneg_1_ptr) + r55_z, r55_value)
    r56_z = tl.arange($(8192))
    r56_value = tl.where((r56_z % $(2)) < $(1), tl.load($(candidate_vneg_1_ptr) + ((((r56_z / $(64)) % $(128)) * $(32)) + ((r56_z / $(2)) % $(32))), dtype=tl.bfloat16), tl.load($(candidate_vmm_2_ptr) + ((((r56_z / $(64)) % $(128)) * $(320)) + (((r56_z / $(2)) % $(32)) * $(2)) + $(256)), dtype=tl.bfloat16))
    r56_value = r56_value.to(tl.bfloat16)
    r56_value = tl.broadcast_to(r56_value, [$(8192)])
    tl.store($(candidate_vstack_1_ptr) + r56_z, r56_value)
    r57_z = tl.arange($(8192))
    r57_value = (tl.load($(candidate_vmm_2_ptr) + ((((r57_z / $(64)) % $(128)) * $(320)) + (r57_z % $(64)) + $(256)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_26_ptr) + ((((r57_z / $(64)) % $(128)) * $(64)) + (r57_z % $(64))), dtype=tl.bfloat16))
    r57_value = r57_value.to(tl.bfloat16)
    r57_value = tl.broadcast_to(r57_value, [$(8192)])
    tl.store($(candidate_vmul_21_ptr) + r57_z, r57_value)
    r58_z = tl.arange($(8192))
    r58_value = (tl.load($(candidate_vstack_1_ptr) + ((((r58_z / $(64)) % $(128)) * $(64)) + (((((((r58_z / $(64)) % $(128)) * $(64)) + (r58_z % $(64))) / $(2)) % $(32)) * $(2)) + (((((r58_z / $(64)) % $(128)) * $(64)) + (r58_z % $(64))) % $(2))), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_27_ptr) + ((((r58_z / $(64)) % $(128)) * $(64)) + (r58_z % $(64))), dtype=tl.bfloat16))
    r58_value = r58_value.to(tl.bfloat16)
    r58_value = tl.broadcast_to(r58_value, [$(8192)])
    tl.store($(candidate_vmul_22_ptr) + r58_z, r58_value)
    r59_z = tl.arange($(8192))
    r59_value = (tl.load($(candidate_vmul_21_ptr) + ((((r59_z / $(64)) % $(128)) * $(64)) + (r59_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_22_ptr) + ((((r59_z / $(64)) % $(128)) * $(64)) + (r59_z % $(64))), dtype=tl.bfloat16))
    r59_value = r59_value.to(tl.bfloat16)
    r59_value = tl.broadcast_to(r59_value, [$(8192)])
    tl.store($(candidate_vadd_6_ptr) + r59_z, r59_value)
    r60_row = tl.arange($(128))
    r60_col = tl.arange($(1024))
    r60_lane = tl.arange($(256))
    for r60_group in range($(1)) {
      r60_left = tl.load($(candidate_vto_copy_23_ptr) + ((((((r60_row[:, None] * $(256)) + r60_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r60_row[:, None] * $(256)) + r60_lane[None, :]) % $(256))), dtype=tl.bfloat16)
      r60_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r60_lane[:, None] * $(1024)) + r60_col[None, :]) / $(128)) % $(8)) * $(65536)) + ((((r60_lane[:, None] * $(1024)) + r60_col[None, :]) % $(128)) * $(256)) + ((((r60_lane[:, None] * $(1024)) + r60_col[None, :]) / $(1024)) % $(256))), dtype=tl.bfloat16)
      r60_value = tl.dot(r60_left, r60_right)
      r60_value = r60_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_2_ptr) + ((r60_group * $(131072)) + (r60_row[:, None] * $(1024)) + r60_col[None, :]), r60_value)
    }
    r61_row = tl.arange($(128))
    r61_col = tl.arange($(1024))
    r61_lane = tl.arange($(256))
    for r61_group in range($(1)) {
      r61_left = tl.load($(candidate_vto_copy_23_ptr) + ((((((r61_row[:, None] * $(256)) + r61_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r61_row[:, None] * $(256)) + r61_lane[None, :]) % $(256))), dtype=tl.bfloat16)
      r61_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r61_lane[:, None] * $(1024)) + r61_col[None, :]) / $(128)) % $(8)) * $(65536)) + ((((r61_lane[:, None] * $(1024)) + r61_col[None, :]) % $(128)) * $(256)) + ((((r61_lane[:, None] * $(1024)) + r61_col[None, :]) / $(1024)) % $(256)) + $(32768)), dtype=tl.bfloat16)
      r61_value = tl.dot(r61_left, r61_right)
      r61_value = r61_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_3_ptr) + ((r61_group * $(131072)) + (r61_row[:, None] * $(1024)) + r61_col[None, :]), r61_value)
    }
    r62_z = tl.arange($(196608))
    r62_value = tl.where((r62_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r62_z / $(1536)) % $(128)) * $(1024)) + (((r62_z / $(192)) % $(8)) * $(128)) + ((r62_z % $(192)) - $(0))) / $(1024)) % $(128)) * $(1536)) + (((((((r62_z / $(1536)) % $(128)) * $(1024)) + (((r62_z / $(192)) % $(8)) * $(128)) + ((r62_z % $(192)) - $(0))) / $(128)) % $(8)) * $(192)) + (((((r62_z / $(1536)) % $(128)) * $(1024)) + (((r62_z / $(192)) % $(8)) * $(128)) + ((r62_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vadd_5_ptr) + ((((r62_z / $(1536)) % $(128)) * $(512)) + (((r62_z / $(192)) % $(8)) * $(64)) + (((r62_z % $(192)) - $(128)) % $(64))), dtype=tl.bfloat16))
    r62_value = r62_value.to(tl.bfloat16)
    r62_value = tl.broadcast_to(r62_value, [$(196608)])
    tl.store($(candidate_vcat_7_ptr) + r62_z, r62_value)
    r63_z = tl.arange($(196608))
    r63_value = tl.where((r63_z % $(192)) < $(128), tl.load($(candidate_vbmm_2_ptr) + ((((((((r63_z / $(1536)) % $(128)) * $(1024)) + (((r63_z / $(192)) % $(8)) * $(128)) + ((r63_z % $(192)) - $(0))) / $(1024)) % $(128)) * $(1024)) + (((((((r63_z / $(1536)) % $(128)) * $(1024)) + (((r63_z / $(192)) % $(8)) * $(128)) + ((r63_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r63_z / $(1536)) % $(128)) * $(1024)) + (((r63_z / $(192)) % $(8)) * $(128)) + ((r63_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vadd_6_ptr) + ((((r63_z / $(1536)) % $(128)) * $(64)) + (((r63_z % $(192)) - $(128)) % $(64))), dtype=tl.bfloat16))
    r63_value = r63_value.to(tl.bfloat16)
    r63_value = tl.broadcast_to(r63_value, [$(196608)])
    tl.store($(candidate_vcat_8_ptr) + r63_z, r63_value)
    r64_z = tl.arange($(73728))
    r64_value = tl.load($(candidate_vcat_7_ptr) + ((((r64_z / $(192)) % $(48)) * $(1536)) + (((r64_z / $(9216)) % $(8)) * $(192)) + (r64_z % $(192))), dtype=tl.bfloat16)
    r64_value = r64_value.to(tl.float32)
    r64_value = tl.broadcast_to(r64_value, [$(73728)])
    tl.store($(candidate_vto_copy_28_ptr) + r64_z, r64_value)
    r65_z = tl.arange($(73728))
    r65_value = tl.load($(candidate_vcat_8_ptr) + ((((r65_z / $(192)) % $(48)) * $(1536)) + (((r65_z / $(9216)) % $(8)) * $(192)) + (r65_z % $(192))), dtype=tl.bfloat16)
    r65_value = r65_value.to(tl.float32)
    r65_value = tl.broadcast_to(r65_value, [$(73728)])
    tl.store($(candidate_vto_copy_29_ptr) + r65_z, r65_value)
    r66_row = tl.arange($(48))
    r66_col = tl.arange($(48))
    r66_lane = tl.arange($(192))
    for r66_group in range($(8)) {
      r66_left = tl.load($(candidate_vto_copy_28_ptr) + (((((((r66_group % $(8)) * $(9216)) + (r66_row[:, None] * $(192)) + r66_lane[None, :]) / $(9216)) % $(8)) * $(9216)) + ((((((r66_group % $(8)) * $(9216)) + (r66_row[:, None] * $(192)) + r66_lane[None, :]) / $(192)) % $(48)) * $(192)) + ((((r66_group % $(8)) * $(9216)) + (r66_row[:, None] * $(192)) + r66_lane[None, :]) % $(192))), dtype=tl.float32)
      r66_right = tl.load($(candidate_vto_copy_29_ptr) + (((((((r66_group % $(8)) * $(9216)) + (r66_lane[:, None] * $(48)) + r66_col[None, :]) / $(9216)) % $(8)) * $(9216)) + (((((r66_group % $(8)) * $(9216)) + (r66_lane[:, None] * $(48)) + r66_col[None, :]) % $(48)) * $(192)) + (((((r66_group % $(8)) * $(9216)) + (r66_lane[:, None] * $(48)) + r66_col[None, :]) / $(48)) % $(192))), dtype=tl.float32)
      r66_value = tl.dot(r66_left, r66_right)
      r66_value = r66_value.to(tl.float32)
      tl.store($(candidate_vbmm_4_ptr) + ((r66_group * $(2304)) + (r66_row[:, None] * $(48)) + r66_col[None, :]), r66_value)
    }
    r67_z = tl.arange($(18432))
    r67_value = (tl.load($(candidate_vbmm_4_ptr) + ((((r67_z / $(2304)) % $(8)) * $(2304)) + (((r67_z / $(48)) % $(48)) * $(48)) + (r67_z % $(48))), dtype=tl.float32) * 0.07216878364870322)
    r67_value = r67_value.to(tl.float32)
    r67_value = tl.broadcast_to(r67_value, [$(18432)])
    tl.store($(candidate_vmul_23_ptr) + r67_z, r67_value)
    r68_z = tl.arange($(48))
    r68_value = (tl.cast(r68_z, tl.int64) + tl.full([], $(48), dtype=tl.int64))
    r68_value = tl.broadcast_to(r68_value, [$(48)])
    tl.store($(candidate_vadd_7_ptr) + r68_z, r68_value)
    r69_z = tl.arange($(48))
    r69_value = (tl.load($(candidate_vadd_7_ptr) + r69_z, dtype=tl.int64) - tl.full([], $(48), dtype=tl.int64))
    r69_value = tl.broadcast_to(r69_value, [$(48)])
    tl.store($(candidate_vsub_4_ptr) + r69_z, r69_value)
    r70_z = tl.arange($(2304))
    r70_value = (tl.cast((r70_z % $(48)), tl.int64) > tl.load($(candidate_vsub_4_ptr) + ((r70_z / $(48)) % $(48)), dtype=tl.int64))
    r70_value = tl.broadcast_to(r70_value, [$(2304)])
    tl.store($(candidate_vgt_ptr) + r70_z, r70_value)
    r71_z = tl.arange($(18432))
    r71_value = tl.where(tl.load($(candidate_vgt_ptr) + ((((r71_z / $(48)) % $(48)) * $(48)) + (r71_z % $(48))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_23_ptr) + ((((r71_z / $(2304)) % $(8)) * $(2304)) + (((r71_z / $(48)) % $(48)) * $(48)) + (r71_z % $(48))), dtype=tl.float32))
    r71_value = r71_value.to(tl.float32)
    r71_value = tl.broadcast_to(r71_value, [$(18432)])
    tl.store($(candidate_vmasked_fill_ptr) + r71_z, r71_value)
    r72_z = tl.arange($(18432))
    r72_lane = tl.arange($(48))
    r72_row_values = tl.load($(candidate_vmasked_fill_ptr) + ((((r72_z[:, None] / $(2304)) % $(8)) * $(2304)) + (((r72_z[:, None] / $(48)) % $(48)) * $(48)) + (r72_lane[None, :] % $(48))), dtype=tl.float32)
    r72_maximum = tl.max(r72_row_values, axis=1, keep_dims=true)
    r72_weights = tl.exp(r72_row_values - r72_maximum)
    r72_total = tl.sum(r72_weights, axis=1)
    r72_maximum = tl.max(r72_row_values, axis=1)
    r72_value = tl.exp(tl.load($(candidate_vmasked_fill_ptr) + ((((r72_z / $(2304)) % $(8)) * $(2304)) + (((r72_z / $(48)) % $(48)) * $(48)) + (r72_z % $(48))), dtype=tl.float32) - r72_maximum) / r72_total
    r72_value = r72_value.to(tl.float32)
    r72_value = tl.broadcast_to(r72_value, [$(18432)])
    tl.store($(candidate_vsoftmax_ptr) + r72_z, r72_value)
    r73_z = tl.arange($(18432))
    r73_value = tl.load($(candidate_vsoftmax_ptr) + ((((r73_z / $(2304)) % $(8)) * $(2304)) + (((r73_z / $(48)) % $(48)) * $(48)) + (r73_z % $(48))), dtype=tl.float32)
    r73_value = r73_value.to(tl.bfloat16)
    r73_value = tl.broadcast_to(r73_value, [$(18432)])
    tl.store($(candidate_vto_copy_30_ptr) + r73_z, r73_value)
    r74_z = tl.arange($(24576))
    r74_value = tl.load($(candidate_vcat_7_ptr) + ((((r74_z / $(192)) % $(16)) * $(1536)) + (((r74_z / $(3072)) % $(8)) * $(192)) + (r74_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r74_value = r74_value.to(tl.float32)
    r74_value = tl.broadcast_to(r74_value, [$(24576)])
    tl.store($(candidate_vto_copy_31_ptr) + r74_z, r74_value)
    r75_z = tl.arange($(24576))
    r75_value = tl.load($(candidate_vcat_8_ptr) + ((((r75_z / $(192)) % $(16)) * $(1536)) + (((r75_z / $(3072)) % $(8)) * $(192)) + (r75_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r75_value = r75_value.to(tl.float32)
    r75_value = tl.broadcast_to(r75_value, [$(24576)])
    tl.store($(candidate_vto_copy_32_ptr) + r75_z, r75_value)
    r76_row = tl.arange($(16))
    r76_col = tl.arange($(16))
    r76_lane = tl.arange($(192))
    for r76_group in range($(8)) {
      r76_left = tl.load($(candidate_vto_copy_31_ptr) + (((((((r76_group % $(8)) * $(3072)) + (r76_row[:, None] * $(192)) + r76_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r76_group % $(8)) * $(3072)) + (r76_row[:, None] * $(192)) + r76_lane[None, :]) / $(192)) % $(16)) * $(192)) + ((((r76_group % $(8)) * $(3072)) + (r76_row[:, None] * $(192)) + r76_lane[None, :]) % $(192))), dtype=tl.float32)
      r76_right = tl.load($(candidate_vto_copy_32_ptr) + (((((((r76_group % $(8)) * $(3072)) + (r76_lane[:, None] * $(16)) + r76_col[None, :]) / $(3072)) % $(8)) * $(3072)) + (((((r76_group % $(8)) * $(3072)) + (r76_lane[:, None] * $(16)) + r76_col[None, :]) % $(16)) * $(192)) + (((((r76_group % $(8)) * $(3072)) + (r76_lane[:, None] * $(16)) + r76_col[None, :]) / $(16)) % $(192))), dtype=tl.float32)
      r76_value = tl.dot(r76_left, r76_right)
      r76_value = r76_value.to(tl.float32)
      tl.store($(candidate_vbmm_6_ptr) + ((r76_group * $(256)) + (r76_row[:, None] * $(16)) + r76_col[None, :]), r76_value)
    }
    r77_z = tl.arange($(2048))
    r77_value = (tl.load($(candidate_vbmm_6_ptr) + ((((r77_z / $(256)) % $(8)) * $(256)) + (((r77_z / $(16)) % $(16)) * $(16)) + (r77_z % $(16))), dtype=tl.float32) * 0.07216878364870322)
    r77_value = r77_value.to(tl.float32)
    r77_value = tl.broadcast_to(r77_value, [$(2048)])
    tl.store($(candidate_vmul_24_ptr) + r77_z, r77_value)
    r78_z = tl.arange($(16))
    r78_value = (tl.cast(r78_z, tl.int64) + tl.full([], $(16), dtype=tl.int64))
    r78_value = tl.broadcast_to(r78_value, [$(16)])
    tl.store($(candidate_vadd_8_ptr) + r78_z, r78_value)
    r79_z = tl.arange($(16))
    r79_value = (tl.load($(candidate_vadd_8_ptr) + r79_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r79_value = tl.broadcast_to(r79_value, [$(16)])
    tl.store($(candidate_vsub_5_ptr) + r79_z, r79_value)
    r80_z = tl.arange($(256))
    r80_value = (tl.cast((r80_z % $(16)), tl.int64) > tl.load($(candidate_vsub_5_ptr) + ((r80_z / $(16)) % $(16)), dtype=tl.int64))
    r80_value = tl.broadcast_to(r80_value, [$(256)])
    tl.store($(candidate_vgt_1_ptr) + r80_z, r80_value)
    r81_z = tl.arange($(2048))
    r81_value = tl.where(tl.load($(candidate_vgt_1_ptr) + ((((r81_z / $(16)) % $(16)) * $(16)) + (r81_z % $(16))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_24_ptr) + ((((r81_z / $(256)) % $(8)) * $(256)) + (((r81_z / $(16)) % $(16)) * $(16)) + (r81_z % $(16))), dtype=tl.float32))
    r81_value = r81_value.to(tl.float32)
    r81_value = tl.broadcast_to(r81_value, [$(2048)])
    tl.store($(candidate_vmasked_fill_1_ptr) + r81_z, r81_value)
    r82_z = tl.arange($(2048))
    r82_lane = tl.arange($(16))
    r82_row_values = tl.load($(candidate_vmasked_fill_1_ptr) + ((((r82_z[:, None] / $(256)) % $(8)) * $(256)) + (((r82_z[:, None] / $(16)) % $(16)) * $(16)) + (r82_lane[None, :] % $(16))), dtype=tl.float32)
    r82_maximum = tl.max(r82_row_values, axis=1, keep_dims=true)
    r82_weights = tl.exp(r82_row_values - r82_maximum)
    r82_total = tl.sum(r82_weights, axis=1)
    r82_maximum = tl.max(r82_row_values, axis=1)
    r82_value = tl.exp(tl.load($(candidate_vmasked_fill_1_ptr) + ((((r82_z / $(256)) % $(8)) * $(256)) + (((r82_z / $(16)) % $(16)) * $(16)) + (r82_z % $(16))), dtype=tl.float32) - r82_maximum) / r82_total
    r82_value = r82_value.to(tl.float32)
    r82_value = tl.broadcast_to(r82_value, [$(2048)])
    tl.store($(candidate_vsoftmax_1_ptr) + r82_z, r82_value)
    r83_z = tl.arange($(2048))
    r83_value = tl.load($(candidate_vsoftmax_1_ptr) + ((((r83_z / $(256)) % $(8)) * $(256)) + (((r83_z / $(16)) % $(16)) * $(16)) + (r83_z % $(16))), dtype=tl.float32)
    r83_value = r83_value.to(tl.bfloat16)
    r83_value = tl.broadcast_to(r83_value, [$(2048)])
    tl.store($(candidate_vto_copy_33_ptr) + r83_z, r83_value)
    r84_z = tl.arange($(98304))
    r84_value = tl.load($(candidate_vcat_7_ptr) + ((((r84_z / $(192)) % $(64)) * $(1536)) + (((r84_z / $(12288)) % $(8)) * $(192)) + (r84_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r84_value = r84_value.to(tl.float32)
    r84_value = tl.broadcast_to(r84_value, [$(98304)])
    tl.store($(candidate_vto_copy_34_ptr) + r84_z, r84_value)
    r85_z = tl.arange($(98304))
    r85_value = tl.load($(candidate_vcat_8_ptr) + ((((r85_z / $(192)) % $(64)) * $(1536)) + (((r85_z / $(12288)) % $(8)) * $(192)) + (r85_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r85_value = r85_value.to(tl.float32)
    r85_value = tl.broadcast_to(r85_value, [$(98304)])
    tl.store($(candidate_vto_copy_35_ptr) + r85_z, r85_value)
    r86_row = tl.arange($(64))
    r86_col = tl.arange($(64))
    r86_lane = tl.arange($(192))
    for r86_group in range($(8)) {
      r86_left = tl.load($(candidate_vto_copy_34_ptr) + (((((((r86_group % $(8)) * $(12288)) + (r86_row[:, None] * $(192)) + r86_lane[None, :]) / $(12288)) % $(8)) * $(12288)) + ((((((r86_group % $(8)) * $(12288)) + (r86_row[:, None] * $(192)) + r86_lane[None, :]) / $(192)) % $(64)) * $(192)) + ((((r86_group % $(8)) * $(12288)) + (r86_row[:, None] * $(192)) + r86_lane[None, :]) % $(192))), dtype=tl.float32)
      r86_right = tl.load($(candidate_vto_copy_35_ptr) + (((((((r86_group % $(8)) * $(12288)) + (r86_lane[:, None] * $(64)) + r86_col[None, :]) / $(12288)) % $(8)) * $(12288)) + (((((r86_group % $(8)) * $(12288)) + (r86_lane[:, None] * $(64)) + r86_col[None, :]) % $(64)) * $(192)) + (((((r86_group % $(8)) * $(12288)) + (r86_lane[:, None] * $(64)) + r86_col[None, :]) / $(64)) % $(192))), dtype=tl.float32)
      r86_value = tl.dot(r86_left, r86_right)
      r86_value = r86_value.to(tl.float32)
      tl.store($(candidate_vbmm_8_ptr) + ((r86_group * $(4096)) + (r86_row[:, None] * $(64)) + r86_col[None, :]), r86_value)
    }
    r87_z = tl.arange($(32768))
    r87_value = (tl.load($(candidate_vbmm_8_ptr) + ((((r87_z / $(4096)) % $(8)) * $(4096)) + (((r87_z / $(64)) % $(64)) * $(64)) + (r87_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r87_value = r87_value.to(tl.float32)
    r87_value = tl.broadcast_to(r87_value, [$(32768)])
    tl.store($(candidate_vmul_25_ptr) + r87_z, r87_value)
    r88_z = tl.arange($(64))
    r88_value = (tl.cast(r88_z, tl.int64) + tl.full([], $(64), dtype=tl.int64))
    r88_value = tl.broadcast_to(r88_value, [$(64)])
    tl.store($(candidate_vadd_9_ptr) + r88_z, r88_value)
    r89_z = tl.arange($(64))
    r89_value = (tl.load($(candidate_vadd_9_ptr) + r89_z, dtype=tl.int64) - tl.full([], $(64), dtype=tl.int64))
    r89_value = tl.broadcast_to(r89_value, [$(64)])
    tl.store($(candidate_vsub_6_ptr) + r89_z, r89_value)
    r90_z = tl.arange($(4096))
    r90_value = (tl.cast((r90_z % $(64)), tl.int64) > tl.load($(candidate_vsub_6_ptr) + ((r90_z / $(64)) % $(64)), dtype=tl.int64))
    r90_value = tl.broadcast_to(r90_value, [$(4096)])
    tl.store($(candidate_vgt_2_ptr) + r90_z, r90_value)
    r91_z = tl.arange($(32768))
    r91_value = tl.where(tl.load($(candidate_vgt_2_ptr) + ((((r91_z / $(64)) % $(64)) * $(64)) + (r91_z % $(64))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_25_ptr) + ((((r91_z / $(4096)) % $(8)) * $(4096)) + (((r91_z / $(64)) % $(64)) * $(64)) + (r91_z % $(64))), dtype=tl.float32))
    r91_value = r91_value.to(tl.float32)
    r91_value = tl.broadcast_to(r91_value, [$(32768)])
    tl.store($(candidate_vmasked_fill_2_ptr) + r91_z, r91_value)
    r92_z = tl.arange($(32768))
    r92_lane = tl.arange($(64))
    r92_row_values = tl.load($(candidate_vmasked_fill_2_ptr) + ((((r92_z[:, None] / $(4096)) % $(8)) * $(4096)) + (((r92_z[:, None] / $(64)) % $(64)) * $(64)) + (r92_lane[None, :] % $(64))), dtype=tl.float32)
    r92_maximum = tl.max(r92_row_values, axis=1, keep_dims=true)
    r92_weights = tl.exp(r92_row_values - r92_maximum)
    r92_total = tl.sum(r92_weights, axis=1)
    r92_maximum = tl.max(r92_row_values, axis=1)
    r92_value = tl.exp(tl.load($(candidate_vmasked_fill_2_ptr) + ((((r92_z / $(4096)) % $(8)) * $(4096)) + (((r92_z / $(64)) % $(64)) * $(64)) + (r92_z % $(64))), dtype=tl.float32) - r92_maximum) / r92_total
    r92_value = r92_value.to(tl.float32)
    r92_value = tl.broadcast_to(r92_value, [$(32768)])
    tl.store($(candidate_vsoftmax_2_ptr) + r92_z, r92_value)
    r93_z = tl.arange($(32768))
    r93_value = tl.load($(candidate_vsoftmax_2_ptr) + ((((r93_z / $(4096)) % $(8)) * $(4096)) + (((r93_z / $(64)) % $(64)) * $(64)) + (r93_z % $(64))), dtype=tl.float32)
    r93_value = r93_value.to(tl.bfloat16)
    r93_value = tl.broadcast_to(r93_value, [$(32768)])
    tl.store($(candidate_vto_copy_36_ptr) + r93_z, r93_value)
    r94_row = tl.arange($(128))
    r94_col = tl.arange($(1024))
    r94_lane = tl.arange($(512))
    r94_left = tl.load($(input1_ptr) + ((((((r94_row[:, None] * $(512)) + r94_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r94_row[:, None] * $(512)) + r94_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r94_right = tl.load($(candidate_vto_copy_6_ptr) + (((r94_lane[:, None] % $(512)) * $(1024)) + (r94_col[None, :] % $(1024))), dtype=tl.bfloat16)
    r94_value = tl.dot(r94_left, r94_right)
    r94_value = r94_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_4_ptr) + ((r94_row[:, None] * $(1024)) + r94_col[None, :]), r94_value)
    r95_row = tl.arange($(64))
    r95_col = tl.arange($(128))
    r95_lane = tl.arange($(64))
    for r95_group in range($(8)) {
      r95_left = tl.load($(candidate_vto_copy_36_ptr) + (((((((r95_group % $(8)) * $(4096)) + (r95_lane[None, :] * $(64)) + r95_row[:, None]) / $(4096)) % $(8)) * $(4096)) + ((((((r95_group % $(8)) * $(4096)) + (r95_lane[None, :] * $(64)) + r95_row[:, None]) / $(64)) % $(64)) * $(64)) + ((((r95_group % $(8)) * $(4096)) + (r95_lane[None, :] * $(64)) + r95_row[:, None]) % $(64))), dtype=tl.bfloat16)
      r95_right = tl.load($(candidate_vmm_4_ptr) + (((((((r95_group % $(8)) * $(8192)) + (r95_lane[:, None] * $(128)) + r95_col[None, :]) / $(128)) % $(64)) * $(1024)) + ((((((r95_group % $(8)) * $(8192)) + (r95_lane[:, None] * $(128)) + r95_col[None, :]) / $(8192)) % $(8)) * $(128)) + ((((r95_group % $(8)) * $(8192)) + (r95_lane[:, None] * $(128)) + r95_col[None, :]) % $(128)) + $(65536)), dtype=tl.bfloat16)
      r95_value = tl.dot(r95_left, r95_right)
      r95_value = r95_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_10_ptr) + ((r95_group * $(8192)) + (r95_row[:, None] * $(128)) + r95_col[None, :]), r95_value)
    }
    r96_row = tl.arange($(64))
    r96_col = tl.arange($(64))
    r96_lane = tl.arange($(128))
    for r96_group in range($(8)) {
      r96_left = tl.load($(candidate_vmm_4_ptr) + (((((((r96_group % $(8)) * $(8192)) + (r96_row[:, None] * $(128)) + r96_lane[None, :]) / $(128)) % $(64)) * $(1024)) + ((((((r96_group % $(8)) * $(8192)) + (r96_row[:, None] * $(128)) + r96_lane[None, :]) / $(8192)) % $(8)) * $(128)) + ((((r96_group % $(8)) * $(8192)) + (r96_row[:, None] * $(128)) + r96_lane[None, :]) % $(128)) + $(65536)), dtype=tl.bfloat16)
      r96_right = tl.load($(candidate_vbmm_3_ptr) + (((((((r96_group % $(8)) * $(8192)) + (r96_col[None, :] * $(128)) + r96_lane[:, None]) / $(128)) % $(64)) * $(1024)) + ((((((r96_group % $(8)) * $(8192)) + (r96_col[None, :] * $(128)) + r96_lane[:, None]) / $(8192)) % $(8)) * $(128)) + ((((r96_group % $(8)) * $(8192)) + (r96_col[None, :] * $(128)) + r96_lane[:, None]) % $(128)) + $(65536)), dtype=tl.bfloat16)
      r96_value = tl.dot(r96_left, r96_right)
      r96_value = r96_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_11_ptr) + ((r96_group * $(4096)) + (r96_row[:, None] * $(64)) + r96_col[None, :]), r96_value)
    }
    r97_z = tl.arange($(32768))
    r97_value = tl.load($(candidate_vbmm_11_ptr) + ((((r97_z / $(4096)) % $(8)) * $(4096)) + (((r97_z / $(64)) % $(64)) * $(64)) + (r97_z % $(64))), dtype=tl.bfloat16)
    r97_value = r97_value.to(tl.float32)
    r97_value = tl.broadcast_to(r97_value, [$(32768)])
    tl.store($(candidate_vto_copy_37_ptr) + r97_z, r97_value)
    r98_z = tl.arange($(32768))
    r98_lane = tl.arange($(64))
    r98_row_values = tl.load($(candidate_vsoftmax_2_ptr) + ((((r98_z[:, None] / $(4096)) % $(8)) * $(4096)) + (((r98_z[:, None] / $(64)) % $(64)) * $(64)) + (r98_lane[None, :] % $(64))), dtype=tl.float32)
    r98_row_grad = tl.load($(candidate_vto_copy_37_ptr) + ((((r98_z[:, None] / $(4096)) % $(8)) * $(4096)) + (((r98_z[:, None] / $(64)) % $(64)) * $(64)) + (r98_lane[None, :] % $(64))), dtype=tl.float32)
    r98_weighted = r98_row_grad * r98_row_values
    r98_dot_grad = tl.sum(r98_weighted, axis=1)
    r98_value = (tl.load($(candidate_vto_copy_37_ptr) + ((((r98_z / $(4096)) % $(8)) * $(4096)) + (((r98_z / $(64)) % $(64)) * $(64)) + (r98_z % $(64))), dtype=tl.float32) - r98_dot_grad) * tl.load($(candidate_vsoftmax_2_ptr) + ((((r98_z / $(4096)) % $(8)) * $(4096)) + (((r98_z / $(64)) % $(64)) * $(64)) + (r98_z % $(64))), dtype=tl.float32)
    r98_value = r98_value.to(tl.float32)
    r98_value = tl.broadcast_to(r98_value, [$(32768)])
    tl.store($(candidate_vsoftmax_backward_data_ptr) + r98_z, r98_value)
    r99_z = tl.arange($(32768))
    r99_value = tl.where(tl.load($(candidate_vgt_2_ptr) + ((((r99_z / $(64)) % $(64)) * $(64)) + (r99_z % $(64))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_ptr) + ((((r99_z / $(4096)) % $(8)) * $(4096)) + (((r99_z / $(64)) % $(64)) * $(64)) + (r99_z % $(64))), dtype=tl.float32))
    r99_value = r99_value.to(tl.float32)
    r99_value = tl.broadcast_to(r99_value, [$(32768)])
    tl.store($(candidate_vmasked_fill_3_ptr) + r99_z, r99_value)
    r100_z = tl.arange($(32768))
    r100_value = (tl.load($(candidate_vmasked_fill_3_ptr) + ((((r100_z / $(4096)) % $(8)) * $(4096)) + (((r100_z / $(64)) % $(64)) * $(64)) + (r100_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r100_value = r100_value.to(tl.float32)
    r100_value = tl.broadcast_to(r100_value, [$(32768)])
    tl.store($(candidate_vmul_26_ptr) + r100_z, r100_value)
    r101_row = tl.arange($(192))
    r101_col = tl.arange($(64))
    r101_lane = tl.arange($(64))
    for r101_group in range($(8)) {
      r101_left = tl.load($(candidate_vto_copy_34_ptr) + (((((((r101_group % $(8)) * $(12288)) + (r101_lane[None, :] * $(192)) + r101_row[:, None]) / $(12288)) % $(8)) * $(12288)) + ((((((r101_group % $(8)) * $(12288)) + (r101_lane[None, :] * $(192)) + r101_row[:, None]) / $(192)) % $(64)) * $(192)) + ((((r101_group % $(8)) * $(12288)) + (r101_lane[None, :] * $(192)) + r101_row[:, None]) % $(192))), dtype=tl.float32)
      r101_right = tl.load($(candidate_vmul_26_ptr) + (((((((r101_group % $(8)) * $(4096)) + (r101_lane[:, None] * $(64)) + r101_col[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r101_group % $(8)) * $(4096)) + (r101_lane[:, None] * $(64)) + r101_col[None, :]) / $(64)) % $(64)) * $(64)) + ((((r101_group % $(8)) * $(4096)) + (r101_lane[:, None] * $(64)) + r101_col[None, :]) % $(64))), dtype=tl.float32)
      r101_value = tl.dot(r101_left, r101_right)
      r101_value = r101_value.to(tl.float32)
      tl.store($(candidate_vbmm_12_ptr) + ((r101_group * $(12288)) + (r101_row[:, None] * $(64)) + r101_col[None, :]), r101_value)
    }
    r102_row = tl.arange($(64))
    r102_col = tl.arange($(192))
    r102_lane = tl.arange($(64))
    for r102_group in range($(8)) {
      r102_left = tl.load($(candidate_vmul_26_ptr) + (((((((r102_group % $(8)) * $(4096)) + (r102_row[:, None] * $(64)) + r102_lane[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r102_group % $(8)) * $(4096)) + (r102_row[:, None] * $(64)) + r102_lane[None, :]) / $(64)) % $(64)) * $(64)) + ((((r102_group % $(8)) * $(4096)) + (r102_row[:, None] * $(64)) + r102_lane[None, :]) % $(64))), dtype=tl.float32)
      r102_right = tl.load($(candidate_vto_copy_35_ptr) + (((((((r102_group % $(8)) * $(12288)) + (r102_col[None, :] * $(64)) + r102_lane[:, None]) / $(12288)) % $(8)) * $(12288)) + (((((r102_group % $(8)) * $(12288)) + (r102_col[None, :] * $(64)) + r102_lane[:, None]) % $(64)) * $(192)) + (((((r102_group % $(8)) * $(12288)) + (r102_col[None, :] * $(64)) + r102_lane[:, None]) / $(64)) % $(192))), dtype=tl.float32)
      r102_value = tl.dot(r102_left, r102_right)
      r102_value = r102_value.to(tl.float32)
      tl.store($(candidate_vbmm_13_ptr) + ((r102_group * $(12288)) + (r102_row[:, None] * $(192)) + r102_col[None, :]), r102_value)
    }
    r103_z = tl.arange($(131072))
    r103_value = tl.where(((((r103_z / $(1024)) % $(128)) >= $(64)) & (((r103_z / $(1024)) % $(128)) < $(128)) & ((((r103_z / $(1024)) % $(128)) - $(64)) % $(1) == $(0))), tl.load($(candidate_vbmm_10_ptr) + ((((((((r103_z / $(128)) % $(8)) * $(8192)) + ((((r103_z / $(1024)) % $(128)) - $(64)) * $(128)) + (r103_z % $(128))) / $(8192)) % $(8)) * $(8192)) + ((((((r103_z / $(128)) % $(8)) * $(64)) + (((r103_z / $(1024)) % $(128)) - $(64))) % $(64)) * $(128)) + (r103_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r103_value = r103_value.to(tl.bfloat16)
    r103_value = tl.broadcast_to(r103_value, [$(131072)])
    tl.store($(candidate_vslice_backward_ptr) + r103_z, r103_value)
    r104_z = tl.arange($(98304))
    r104_value = tl.load($(candidate_vbmm_12_ptr) + ((((r104_z / $(12288)) % $(8)) * $(12288)) + ((r104_z % $(192)) * $(64)) + ((r104_z / $(192)) % $(64))), dtype=tl.float32)
    r104_value = r104_value.to(tl.bfloat16)
    r104_value = tl.broadcast_to(r104_value, [$(98304)])
    tl.store($(candidate_vto_copy_38_ptr) + r104_z, r104_value)
    r105_z = tl.arange($(196608))
    r105_value = tl.where(((((r105_z / $(1536)) % $(128)) >= $(64)) & (((r105_z / $(1536)) % $(128)) < $(128)) & ((((r105_z / $(1536)) % $(128)) - $(64)) % $(1) == $(0))), tl.load($(candidate_vto_copy_38_ptr) + ((((r105_z / $(192)) % $(8)) * $(12288)) + (((((r105_z / $(1536)) % $(128)) - $(64)) % $(64)) * $(192)) + (r105_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r105_value = r105_value.to(tl.bfloat16)
    r105_value = tl.broadcast_to(r105_value, [$(196608)])
    tl.store($(candidate_vslice_backward_1_ptr) + r105_z, r105_value)
    r106_z = tl.arange($(98304))
    r106_value = tl.load($(candidate_vbmm_13_ptr) + ((((r106_z / $(12288)) % $(8)) * $(12288)) + (((r106_z / $(192)) % $(64)) * $(192)) + (r106_z % $(192))), dtype=tl.float32)
    r106_value = r106_value.to(tl.bfloat16)
    r106_value = tl.broadcast_to(r106_value, [$(98304)])
    tl.store($(candidate_vto_copy_39_ptr) + r106_z, r106_value)
    r107_z = tl.arange($(196608))
    r107_value = tl.where(((((r107_z / $(1536)) % $(128)) >= $(64)) & (((r107_z / $(1536)) % $(128)) < $(128)) & ((((r107_z / $(1536)) % $(128)) - $(64)) % $(1) == $(0))), tl.load($(candidate_vto_copy_39_ptr) + ((((r107_z / $(192)) % $(8)) * $(12288)) + (((((r107_z / $(1536)) % $(128)) - $(64)) % $(64)) * $(192)) + (r107_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r107_value = r107_value.to(tl.bfloat16)
    r107_value = tl.broadcast_to(r107_value, [$(196608)])
    tl.store($(candidate_vslice_backward_2_ptr) + r107_z, r107_value)
    r108_row = tl.arange($(16))
    r108_col = tl.arange($(128))
    r108_lane = tl.arange($(16))
    for r108_group in range($(8)) {
      r108_left = tl.load($(candidate_vto_copy_33_ptr) + (((((((r108_group % $(8)) * $(256)) + (r108_lane[None, :] * $(16)) + r108_row[:, None]) / $(256)) % $(8)) * $(256)) + ((((((r108_group % $(8)) * $(256)) + (r108_lane[None, :] * $(16)) + r108_row[:, None]) / $(16)) % $(16)) * $(16)) + ((((r108_group % $(8)) * $(256)) + (r108_lane[None, :] * $(16)) + r108_row[:, None]) % $(16))), dtype=tl.bfloat16)
      r108_right = tl.load($(candidate_vmm_4_ptr) + (((((((r108_group % $(8)) * $(2048)) + (r108_lane[:, None] * $(128)) + r108_col[None, :]) / $(128)) % $(16)) * $(1024)) + ((((((r108_group % $(8)) * $(2048)) + (r108_lane[:, None] * $(128)) + r108_col[None, :]) / $(2048)) % $(8)) * $(128)) + ((((r108_group % $(8)) * $(2048)) + (r108_lane[:, None] * $(128)) + r108_col[None, :]) % $(128)) + $(49152)), dtype=tl.bfloat16)
      r108_value = tl.dot(r108_left, r108_right)
      r108_value = r108_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_14_ptr) + ((r108_group * $(2048)) + (r108_row[:, None] * $(128)) + r108_col[None, :]), r108_value)
    }
    r109_row = tl.arange($(16))
    r109_col = tl.arange($(16))
    r109_lane = tl.arange($(128))
    for r109_group in range($(8)) {
      r109_left = tl.load($(candidate_vmm_4_ptr) + (((((((r109_group % $(8)) * $(2048)) + (r109_row[:, None] * $(128)) + r109_lane[None, :]) / $(128)) % $(16)) * $(1024)) + ((((((r109_group % $(8)) * $(2048)) + (r109_row[:, None] * $(128)) + r109_lane[None, :]) / $(2048)) % $(8)) * $(128)) + ((((r109_group % $(8)) * $(2048)) + (r109_row[:, None] * $(128)) + r109_lane[None, :]) % $(128)) + $(49152)), dtype=tl.bfloat16)
      r109_right = tl.load($(candidate_vbmm_3_ptr) + (((((((r109_group % $(8)) * $(2048)) + (r109_col[None, :] * $(128)) + r109_lane[:, None]) / $(128)) % $(16)) * $(1024)) + ((((((r109_group % $(8)) * $(2048)) + (r109_col[None, :] * $(128)) + r109_lane[:, None]) / $(2048)) % $(8)) * $(128)) + ((((r109_group % $(8)) * $(2048)) + (r109_col[None, :] * $(128)) + r109_lane[:, None]) % $(128)) + $(49152)), dtype=tl.bfloat16)
      r109_value = tl.dot(r109_left, r109_right)
      r109_value = r109_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_15_ptr) + ((r109_group * $(256)) + (r109_row[:, None] * $(16)) + r109_col[None, :]), r109_value)
    }
    r110_z = tl.arange($(2048))
    r110_value = tl.load($(candidate_vbmm_15_ptr) + ((((r110_z / $(256)) % $(8)) * $(256)) + (((r110_z / $(16)) % $(16)) * $(16)) + (r110_z % $(16))), dtype=tl.bfloat16)
    r110_value = r110_value.to(tl.float32)
    r110_value = tl.broadcast_to(r110_value, [$(2048)])
    tl.store($(candidate_vto_copy_40_ptr) + r110_z, r110_value)
    r111_z = tl.arange($(2048))
    r111_lane = tl.arange($(16))
    r111_row_values = tl.load($(candidate_vsoftmax_1_ptr) + ((((r111_z[:, None] / $(256)) % $(8)) * $(256)) + (((r111_z[:, None] / $(16)) % $(16)) * $(16)) + (r111_lane[None, :] % $(16))), dtype=tl.float32)
    r111_row_grad = tl.load($(candidate_vto_copy_40_ptr) + ((((r111_z[:, None] / $(256)) % $(8)) * $(256)) + (((r111_z[:, None] / $(16)) % $(16)) * $(16)) + (r111_lane[None, :] % $(16))), dtype=tl.float32)
    r111_weighted = r111_row_grad * r111_row_values
    r111_dot_grad = tl.sum(r111_weighted, axis=1)
    r111_value = (tl.load($(candidate_vto_copy_40_ptr) + ((((r111_z / $(256)) % $(8)) * $(256)) + (((r111_z / $(16)) % $(16)) * $(16)) + (r111_z % $(16))), dtype=tl.float32) - r111_dot_grad) * tl.load($(candidate_vsoftmax_1_ptr) + ((((r111_z / $(256)) % $(8)) * $(256)) + (((r111_z / $(16)) % $(16)) * $(16)) + (r111_z % $(16))), dtype=tl.float32)
    r111_value = r111_value.to(tl.float32)
    r111_value = tl.broadcast_to(r111_value, [$(2048)])
    tl.store($(candidate_vsoftmax_backward_data_1_ptr) + r111_z, r111_value)
    r112_z = tl.arange($(2048))
    r112_value = tl.where(tl.load($(candidate_vgt_1_ptr) + ((((r112_z / $(16)) % $(16)) * $(16)) + (r112_z % $(16))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_1_ptr) + ((((r112_z / $(256)) % $(8)) * $(256)) + (((r112_z / $(16)) % $(16)) * $(16)) + (r112_z % $(16))), dtype=tl.float32))
    r112_value = r112_value.to(tl.float32)
    r112_value = tl.broadcast_to(r112_value, [$(2048)])
    tl.store($(candidate_vmasked_fill_4_ptr) + r112_z, r112_value)
    r113_z = tl.arange($(2048))
    r113_value = (tl.load($(candidate_vmasked_fill_4_ptr) + ((((r113_z / $(256)) % $(8)) * $(256)) + (((r113_z / $(16)) % $(16)) * $(16)) + (r113_z % $(16))), dtype=tl.float32) * 0.07216878364870322)
    r113_value = r113_value.to(tl.float32)
    r113_value = tl.broadcast_to(r113_value, [$(2048)])
    tl.store($(candidate_vmul_27_ptr) + r113_z, r113_value)
    r114_row = tl.arange($(192))
    r114_col = tl.arange($(16))
    r114_lane = tl.arange($(16))
    for r114_group in range($(8)) {
      r114_left = tl.load($(candidate_vto_copy_31_ptr) + (((((((r114_group % $(8)) * $(3072)) + (r114_lane[None, :] * $(192)) + r114_row[:, None]) / $(3072)) % $(8)) * $(3072)) + ((((((r114_group % $(8)) * $(3072)) + (r114_lane[None, :] * $(192)) + r114_row[:, None]) / $(192)) % $(16)) * $(192)) + ((((r114_group % $(8)) * $(3072)) + (r114_lane[None, :] * $(192)) + r114_row[:, None]) % $(192))), dtype=tl.float32)
      r114_right = tl.load($(candidate_vmul_27_ptr) + (((((((r114_group % $(8)) * $(256)) + (r114_lane[:, None] * $(16)) + r114_col[None, :]) / $(256)) % $(8)) * $(256)) + ((((((r114_group % $(8)) * $(256)) + (r114_lane[:, None] * $(16)) + r114_col[None, :]) / $(16)) % $(16)) * $(16)) + ((((r114_group % $(8)) * $(256)) + (r114_lane[:, None] * $(16)) + r114_col[None, :]) % $(16))), dtype=tl.float32)
      r114_value = tl.dot(r114_left, r114_right)
      r114_value = r114_value.to(tl.float32)
      tl.store($(candidate_vbmm_16_ptr) + ((r114_group * $(3072)) + (r114_row[:, None] * $(16)) + r114_col[None, :]), r114_value)
    }
    r115_row = tl.arange($(16))
    r115_col = tl.arange($(192))
    r115_lane = tl.arange($(16))
    for r115_group in range($(8)) {
      r115_left = tl.load($(candidate_vmul_27_ptr) + (((((((r115_group % $(8)) * $(256)) + (r115_row[:, None] * $(16)) + r115_lane[None, :]) / $(256)) % $(8)) * $(256)) + ((((((r115_group % $(8)) * $(256)) + (r115_row[:, None] * $(16)) + r115_lane[None, :]) / $(16)) % $(16)) * $(16)) + ((((r115_group % $(8)) * $(256)) + (r115_row[:, None] * $(16)) + r115_lane[None, :]) % $(16))), dtype=tl.float32)
      r115_right = tl.load($(candidate_vto_copy_32_ptr) + (((((((r115_group % $(8)) * $(3072)) + (r115_col[None, :] * $(16)) + r115_lane[:, None]) / $(3072)) % $(8)) * $(3072)) + (((((r115_group % $(8)) * $(3072)) + (r115_col[None, :] * $(16)) + r115_lane[:, None]) % $(16)) * $(192)) + (((((r115_group % $(8)) * $(3072)) + (r115_col[None, :] * $(16)) + r115_lane[:, None]) / $(16)) % $(192))), dtype=tl.float32)
      r115_value = tl.dot(r115_left, r115_right)
      r115_value = r115_value.to(tl.float32)
      tl.store($(candidate_vbmm_17_ptr) + ((r115_group * $(3072)) + (r115_row[:, None] * $(192)) + r115_col[None, :]), r115_value)
    }
    r116_z = tl.arange($(131072))
    r116_value = tl.where(((((r116_z / $(1024)) % $(128)) >= $(48)) & (((r116_z / $(1024)) % $(128)) < $(64)) & ((((r116_z / $(1024)) % $(128)) - $(48)) % $(1) == $(0))), tl.load($(candidate_vbmm_14_ptr) + ((((((((r116_z / $(128)) % $(8)) * $(2048)) + ((((r116_z / $(1024)) % $(128)) - $(48)) * $(128)) + (r116_z % $(128))) / $(2048)) % $(8)) * $(2048)) + ((((((r116_z / $(128)) % $(8)) * $(16)) + (((r116_z / $(1024)) % $(128)) - $(48))) % $(16)) * $(128)) + (r116_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r116_value = r116_value.to(tl.bfloat16)
    r116_value = tl.broadcast_to(r116_value, [$(131072)])
    tl.store($(candidate_vslice_backward_3_ptr) + r116_z, r116_value)
    r117_z = tl.arange($(131072))
    r117_value = (tl.load($(candidate_vslice_backward_ptr) + ((((r117_z / $(1024)) % $(128)) * $(1024)) + (((r117_z / $(128)) % $(8)) * $(128)) + (r117_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_3_ptr) + ((((r117_z / $(1024)) % $(128)) * $(1024)) + (((r117_z / $(128)) % $(8)) * $(128)) + (r117_z % $(128))), dtype=tl.bfloat16))
    r117_value = r117_value.to(tl.bfloat16)
    r117_value = tl.broadcast_to(r117_value, [$(131072)])
    tl.store($(candidate_vadd_10_ptr) + r117_z, r117_value)
    r118_z = tl.arange($(24576))
    r118_value = tl.load($(candidate_vbmm_16_ptr) + ((((r118_z / $(3072)) % $(8)) * $(3072)) + ((r118_z % $(192)) * $(16)) + ((r118_z / $(192)) % $(16))), dtype=tl.float32)
    r118_value = r118_value.to(tl.bfloat16)
    r118_value = tl.broadcast_to(r118_value, [$(24576)])
    tl.store($(candidate_vto_copy_41_ptr) + r118_z, r118_value)
    r119_z = tl.arange($(196608))
    r119_value = tl.where(((((r119_z / $(1536)) % $(128)) >= $(48)) & (((r119_z / $(1536)) % $(128)) < $(64)) & ((((r119_z / $(1536)) % $(128)) - $(48)) % $(1) == $(0))), tl.load($(candidate_vto_copy_41_ptr) + ((((r119_z / $(192)) % $(8)) * $(3072)) + (((((r119_z / $(1536)) % $(128)) - $(48)) % $(16)) * $(192)) + (r119_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r119_value = r119_value.to(tl.bfloat16)
    r119_value = tl.broadcast_to(r119_value, [$(196608)])
    tl.store($(candidate_vslice_backward_4_ptr) + r119_z, r119_value)
    r120_z = tl.arange($(196608))
    r120_value = (tl.load($(candidate_vslice_backward_1_ptr) + ((((r120_z / $(1536)) % $(128)) * $(1536)) + (((r120_z / $(192)) % $(8)) * $(192)) + (r120_z % $(192))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_4_ptr) + ((((r120_z / $(1536)) % $(128)) * $(1536)) + (((r120_z / $(192)) % $(8)) * $(192)) + (r120_z % $(192))), dtype=tl.bfloat16))
    r120_value = r120_value.to(tl.bfloat16)
    r120_value = tl.broadcast_to(r120_value, [$(196608)])
    tl.store($(candidate_vadd_11_ptr) + r120_z, r120_value)
    r121_z = tl.arange($(24576))
    r121_value = tl.load($(candidate_vbmm_17_ptr) + ((((r121_z / $(3072)) % $(8)) * $(3072)) + (((r121_z / $(192)) % $(16)) * $(192)) + (r121_z % $(192))), dtype=tl.float32)
    r121_value = r121_value.to(tl.bfloat16)
    r121_value = tl.broadcast_to(r121_value, [$(24576)])
    tl.store($(candidate_vto_copy_42_ptr) + r121_z, r121_value)
    r122_z = tl.arange($(196608))
    r122_value = tl.where(((((r122_z / $(1536)) % $(128)) >= $(48)) & (((r122_z / $(1536)) % $(128)) < $(64)) & ((((r122_z / $(1536)) % $(128)) - $(48)) % $(1) == $(0))), tl.load($(candidate_vto_copy_42_ptr) + ((((r122_z / $(192)) % $(8)) * $(3072)) + (((((r122_z / $(1536)) % $(128)) - $(48)) % $(16)) * $(192)) + (r122_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r122_value = r122_value.to(tl.bfloat16)
    r122_value = tl.broadcast_to(r122_value, [$(196608)])
    tl.store($(candidate_vslice_backward_5_ptr) + r122_z, r122_value)
    r123_z = tl.arange($(196608))
    r123_value = (tl.load($(candidate_vslice_backward_2_ptr) + ((((r123_z / $(1536)) % $(128)) * $(1536)) + (((r123_z / $(192)) % $(8)) * $(192)) + (r123_z % $(192))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_5_ptr) + ((((r123_z / $(1536)) % $(128)) * $(1536)) + (((r123_z / $(192)) % $(8)) * $(192)) + (r123_z % $(192))), dtype=tl.bfloat16))
    r123_value = r123_value.to(tl.bfloat16)
    r123_value = tl.broadcast_to(r123_value, [$(196608)])
    tl.store($(candidate_vadd_12_ptr) + r123_z, r123_value)
    r124_row = tl.arange($(48))
    r124_col = tl.arange($(128))
    r124_lane = tl.arange($(48))
    for r124_group in range($(8)) {
      r124_left = tl.load($(candidate_vto_copy_30_ptr) + (((((((r124_group % $(8)) * $(2304)) + (r124_lane[None, :] * $(48)) + r124_row[:, None]) / $(2304)) % $(8)) * $(2304)) + ((((((r124_group % $(8)) * $(2304)) + (r124_lane[None, :] * $(48)) + r124_row[:, None]) / $(48)) % $(48)) * $(48)) + ((((r124_group % $(8)) * $(2304)) + (r124_lane[None, :] * $(48)) + r124_row[:, None]) % $(48))), dtype=tl.bfloat16)
      r124_right = tl.load($(candidate_vmm_4_ptr) + (((((((r124_group % $(8)) * $(6144)) + (r124_lane[:, None] * $(128)) + r124_col[None, :]) / $(128)) % $(48)) * $(1024)) + ((((((r124_group % $(8)) * $(6144)) + (r124_lane[:, None] * $(128)) + r124_col[None, :]) / $(6144)) % $(8)) * $(128)) + ((((r124_group % $(8)) * $(6144)) + (r124_lane[:, None] * $(128)) + r124_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r124_value = tl.dot(r124_left, r124_right)
      r124_value = r124_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_18_ptr) + ((r124_group * $(6144)) + (r124_row[:, None] * $(128)) + r124_col[None, :]), r124_value)
    }
    r125_row = tl.arange($(48))
    r125_col = tl.arange($(48))
    r125_lane = tl.arange($(128))
    for r125_group in range($(8)) {
      r125_left = tl.load($(candidate_vmm_4_ptr) + (((((((r125_group % $(8)) * $(6144)) + (r125_row[:, None] * $(128)) + r125_lane[None, :]) / $(128)) % $(48)) * $(1024)) + ((((((r125_group % $(8)) * $(6144)) + (r125_row[:, None] * $(128)) + r125_lane[None, :]) / $(6144)) % $(8)) * $(128)) + ((((r125_group % $(8)) * $(6144)) + (r125_row[:, None] * $(128)) + r125_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r125_right = tl.load($(candidate_vbmm_3_ptr) + (((((((r125_group % $(8)) * $(6144)) + (r125_col[None, :] * $(128)) + r125_lane[:, None]) / $(128)) % $(48)) * $(1024)) + ((((((r125_group % $(8)) * $(6144)) + (r125_col[None, :] * $(128)) + r125_lane[:, None]) / $(6144)) % $(8)) * $(128)) + ((((r125_group % $(8)) * $(6144)) + (r125_col[None, :] * $(128)) + r125_lane[:, None]) % $(128))), dtype=tl.bfloat16)
      r125_value = tl.dot(r125_left, r125_right)
      r125_value = r125_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_19_ptr) + ((r125_group * $(2304)) + (r125_row[:, None] * $(48)) + r125_col[None, :]), r125_value)
    }
    r126_z = tl.arange($(18432))
    r126_value = tl.load($(candidate_vbmm_19_ptr) + ((((r126_z / $(2304)) % $(8)) * $(2304)) + (((r126_z / $(48)) % $(48)) * $(48)) + (r126_z % $(48))), dtype=tl.bfloat16)
    r126_value = r126_value.to(tl.float32)
    r126_value = tl.broadcast_to(r126_value, [$(18432)])
    tl.store($(candidate_vto_copy_43_ptr) + r126_z, r126_value)
    r127_z = tl.arange($(18432))
    r127_lane = tl.arange($(48))
    r127_row_values = tl.load($(candidate_vsoftmax_ptr) + ((((r127_z[:, None] / $(2304)) % $(8)) * $(2304)) + (((r127_z[:, None] / $(48)) % $(48)) * $(48)) + (r127_lane[None, :] % $(48))), dtype=tl.float32)
    r127_row_grad = tl.load($(candidate_vto_copy_43_ptr) + ((((r127_z[:, None] / $(2304)) % $(8)) * $(2304)) + (((r127_z[:, None] / $(48)) % $(48)) * $(48)) + (r127_lane[None, :] % $(48))), dtype=tl.float32)
    r127_weighted = r127_row_grad * r127_row_values
    r127_dot_grad = tl.sum(r127_weighted, axis=1)
    r127_value = (tl.load($(candidate_vto_copy_43_ptr) + ((((r127_z / $(2304)) % $(8)) * $(2304)) + (((r127_z / $(48)) % $(48)) * $(48)) + (r127_z % $(48))), dtype=tl.float32) - r127_dot_grad) * tl.load($(candidate_vsoftmax_ptr) + ((((r127_z / $(2304)) % $(8)) * $(2304)) + (((r127_z / $(48)) % $(48)) * $(48)) + (r127_z % $(48))), dtype=tl.float32)
    r127_value = r127_value.to(tl.float32)
    r127_value = tl.broadcast_to(r127_value, [$(18432)])
    tl.store($(candidate_vsoftmax_backward_data_2_ptr) + r127_z, r127_value)
    r128_z = tl.arange($(18432))
    r128_value = tl.where(tl.load($(candidate_vgt_ptr) + ((((r128_z / $(48)) % $(48)) * $(48)) + (r128_z % $(48))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_2_ptr) + ((((r128_z / $(2304)) % $(8)) * $(2304)) + (((r128_z / $(48)) % $(48)) * $(48)) + (r128_z % $(48))), dtype=tl.float32))
    r128_value = r128_value.to(tl.float32)
    r128_value = tl.broadcast_to(r128_value, [$(18432)])
    tl.store($(candidate_vmasked_fill_5_ptr) + r128_z, r128_value)
    r129_z = tl.arange($(18432))
    r129_value = (tl.load($(candidate_vmasked_fill_5_ptr) + ((((r129_z / $(2304)) % $(8)) * $(2304)) + (((r129_z / $(48)) % $(48)) * $(48)) + (r129_z % $(48))), dtype=tl.float32) * 0.07216878364870322)
    r129_value = r129_value.to(tl.float32)
    r129_value = tl.broadcast_to(r129_value, [$(18432)])
    tl.store($(candidate_vmul_28_ptr) + r129_z, r129_value)
    r130_row = tl.arange($(192))
    r130_col = tl.arange($(48))
    r130_lane = tl.arange($(48))
    for r130_group in range($(8)) {
      r130_left = tl.load($(candidate_vto_copy_28_ptr) + (((((((r130_group % $(8)) * $(9216)) + (r130_lane[None, :] * $(192)) + r130_row[:, None]) / $(9216)) % $(8)) * $(9216)) + ((((((r130_group % $(8)) * $(9216)) + (r130_lane[None, :] * $(192)) + r130_row[:, None]) / $(192)) % $(48)) * $(192)) + ((((r130_group % $(8)) * $(9216)) + (r130_lane[None, :] * $(192)) + r130_row[:, None]) % $(192))), dtype=tl.float32)
      r130_right = tl.load($(candidate_vmul_28_ptr) + (((((((r130_group % $(8)) * $(2304)) + (r130_lane[:, None] * $(48)) + r130_col[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r130_group % $(8)) * $(2304)) + (r130_lane[:, None] * $(48)) + r130_col[None, :]) / $(48)) % $(48)) * $(48)) + ((((r130_group % $(8)) * $(2304)) + (r130_lane[:, None] * $(48)) + r130_col[None, :]) % $(48))), dtype=tl.float32)
      r130_value = tl.dot(r130_left, r130_right)
      r130_value = r130_value.to(tl.float32)
      tl.store($(candidate_vbmm_20_ptr) + ((r130_group * $(9216)) + (r130_row[:, None] * $(48)) + r130_col[None, :]), r130_value)
    }
    r131_row = tl.arange($(48))
    r131_col = tl.arange($(192))
    r131_lane = tl.arange($(48))
    for r131_group in range($(8)) {
      r131_left = tl.load($(candidate_vmul_28_ptr) + (((((((r131_group % $(8)) * $(2304)) + (r131_row[:, None] * $(48)) + r131_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r131_group % $(8)) * $(2304)) + (r131_row[:, None] * $(48)) + r131_lane[None, :]) / $(48)) % $(48)) * $(48)) + ((((r131_group % $(8)) * $(2304)) + (r131_row[:, None] * $(48)) + r131_lane[None, :]) % $(48))), dtype=tl.float32)
      r131_right = tl.load($(candidate_vto_copy_29_ptr) + (((((((r131_group % $(8)) * $(9216)) + (r131_col[None, :] * $(48)) + r131_lane[:, None]) / $(9216)) % $(8)) * $(9216)) + (((((r131_group % $(8)) * $(9216)) + (r131_col[None, :] * $(48)) + r131_lane[:, None]) % $(48)) * $(192)) + (((((r131_group % $(8)) * $(9216)) + (r131_col[None, :] * $(48)) + r131_lane[:, None]) / $(48)) % $(192))), dtype=tl.float32)
      r131_value = tl.dot(r131_left, r131_right)
      r131_value = r131_value.to(tl.float32)
      tl.store($(candidate_vbmm_21_ptr) + ((r131_group * $(9216)) + (r131_row[:, None] * $(192)) + r131_col[None, :]), r131_value)
    }
    r132_z = tl.arange($(131072))
    r132_value = tl.where(((((r132_z / $(1024)) % $(128)) >= $(0)) & (((r132_z / $(1024)) % $(128)) < $(48)) & ((((r132_z / $(1024)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vbmm_18_ptr) + ((((((((r132_z / $(128)) % $(8)) * $(6144)) + ((((r132_z / $(1024)) % $(128)) - $(0)) * $(128)) + (r132_z % $(128))) / $(6144)) % $(8)) * $(6144)) + ((((((r132_z / $(128)) % $(8)) * $(48)) + (((r132_z / $(1024)) % $(128)) - $(0))) % $(48)) * $(128)) + (r132_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r132_value = r132_value.to(tl.bfloat16)
    r132_value = tl.broadcast_to(r132_value, [$(131072)])
    tl.store($(candidate_vslice_backward_6_ptr) + r132_z, r132_value)
    r133_z = tl.arange($(131072))
    r133_value = (tl.load($(candidate_vadd_10_ptr) + ((((r133_z / $(1024)) % $(128)) * $(1024)) + (((r133_z / $(128)) % $(8)) * $(128)) + (r133_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_6_ptr) + ((((r133_z / $(1024)) % $(128)) * $(1024)) + (((r133_z / $(128)) % $(8)) * $(128)) + (r133_z % $(128))), dtype=tl.bfloat16))
    r133_value = r133_value.to(tl.bfloat16)
    r133_value = tl.broadcast_to(r133_value, [$(131072)])
    tl.store($(candidate_vadd_13_ptr) + r133_z, r133_value)
    r134_z = tl.arange($(73728))
    r134_value = tl.load($(candidate_vbmm_20_ptr) + ((((r134_z / $(9216)) % $(8)) * $(9216)) + ((r134_z % $(192)) * $(48)) + ((r134_z / $(192)) % $(48))), dtype=tl.float32)
    r134_value = r134_value.to(tl.bfloat16)
    r134_value = tl.broadcast_to(r134_value, [$(73728)])
    tl.store($(candidate_vto_copy_44_ptr) + r134_z, r134_value)
    r135_z = tl.arange($(196608))
    r135_value = tl.where(((((r135_z / $(1536)) % $(128)) >= $(0)) & (((r135_z / $(1536)) % $(128)) < $(48)) & ((((r135_z / $(1536)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vto_copy_44_ptr) + ((((r135_z / $(192)) % $(8)) * $(9216)) + (((((r135_z / $(1536)) % $(128)) - $(0)) % $(48)) * $(192)) + (r135_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r135_value = r135_value.to(tl.bfloat16)
    r135_value = tl.broadcast_to(r135_value, [$(196608)])
    tl.store($(candidate_vslice_backward_7_ptr) + r135_z, r135_value)
    r136_z = tl.arange($(196608))
    r136_value = (tl.load($(candidate_vadd_11_ptr) + ((((r136_z / $(1536)) % $(128)) * $(1536)) + (((r136_z / $(192)) % $(8)) * $(192)) + (r136_z % $(192))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_7_ptr) + ((((r136_z / $(1536)) % $(128)) * $(1536)) + (((r136_z / $(192)) % $(8)) * $(192)) + (r136_z % $(192))), dtype=tl.bfloat16))
    r136_value = r136_value.to(tl.bfloat16)
    r136_value = tl.broadcast_to(r136_value, [$(196608)])
    tl.store($(candidate_vadd_14_ptr) + r136_z, r136_value)
    r137_z = tl.arange($(73728))
    r137_value = tl.load($(candidate_vbmm_21_ptr) + ((((r137_z / $(9216)) % $(8)) * $(9216)) + (((r137_z / $(192)) % $(48)) * $(192)) + (r137_z % $(192))), dtype=tl.float32)
    r137_value = r137_value.to(tl.bfloat16)
    r137_value = tl.broadcast_to(r137_value, [$(73728)])
    tl.store($(candidate_vto_copy_45_ptr) + r137_z, r137_value)
    r138_z = tl.arange($(196608))
    r138_value = tl.where(((((r138_z / $(1536)) % $(128)) >= $(0)) & (((r138_z / $(1536)) % $(128)) < $(48)) & ((((r138_z / $(1536)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vto_copy_45_ptr) + ((((r138_z / $(192)) % $(8)) * $(9216)) + (((((r138_z / $(1536)) % $(128)) - $(0)) % $(48)) * $(192)) + (r138_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r138_value = r138_value.to(tl.bfloat16)
    r138_value = tl.broadcast_to(r138_value, [$(196608)])
    tl.store($(candidate_vslice_backward_8_ptr) + r138_z, r138_value)
    r139_z = tl.arange($(196608))
    r139_value = (tl.load($(candidate_vadd_12_ptr) + ((((r139_z / $(1536)) % $(128)) * $(1536)) + (((r139_z / $(192)) % $(8)) * $(192)) + (r139_z % $(192))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_8_ptr) + ((((r139_z / $(1536)) % $(128)) * $(1536)) + (((r139_z / $(192)) % $(8)) * $(192)) + (r139_z % $(192))), dtype=tl.bfloat16))
    r139_value = r139_value.to(tl.bfloat16)
    r139_value = tl.broadcast_to(r139_value, [$(196608)])
    tl.store($(candidate_vadd_15_ptr) + r139_z, r139_value)
    r140_z = tl.arange($(8192))
    r140_lane = tl.arange($(8))
    r140_value = tl.sum(tl.load($(candidate_vadd_14_ptr) + ((((r140_z[:, None] / $(64)) % $(128)) * $(1536)) + (r140_lane[None, :] * $(192)) + (r140_z[:, None] % $(64)) + $(128)), dtype=tl.bfloat16), axis=1)
    r140_value = r140_value.to(tl.bfloat16)
    r140_value = tl.broadcast_to(r140_value, [$(8192)])
    tl.store($(candidate_vsum_1_ptr) + r140_z, r140_value)
    r141_row = tl.arange($(128))
    r141_col = tl.arange($(256))
    r141_lane = tl.arange($(1024))
    for r141_group in range($(1)) {
      r141_left = tl.load($(candidate_vadd_13_ptr) + ((((((r141_row[:, None] * $(1024)) + r141_lane[None, :]) / $(1024)) % $(128)) * $(1024)) + (((((r141_row[:, None] * $(1024)) + r141_lane[None, :]) / $(128)) % $(8)) * $(128)) + (((r141_row[:, None] * $(1024)) + r141_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r141_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r141_col[None, :] * $(1024)) + r141_lane[:, None]) / $(128)) % $(8)) * $(65536)) + ((((r141_col[None, :] * $(1024)) + r141_lane[:, None]) % $(128)) * $(256)) + ((((r141_col[None, :] * $(1024)) + r141_lane[:, None]) / $(1024)) % $(256)) + $(32768)), dtype=tl.bfloat16)
      r141_value = tl.dot(r141_left, r141_right)
      r141_value = r141_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_22_ptr) + ((r141_group * $(32768)) + (r141_row[:, None] * $(256)) + r141_col[None, :]), r141_value)
    }
    r142_row = tl.arange($(128))
    r142_col = tl.arange($(256))
    r142_lane = tl.arange($(1024))
    for r142_group in range($(1)) {
      r142_left = tl.load($(candidate_vadd_14_ptr) + ((((((r142_row[:, None] * $(1024)) + r142_lane[None, :]) / $(1024)) % $(128)) * $(1536)) + (((((r142_row[:, None] * $(1024)) + r142_lane[None, :]) / $(128)) % $(8)) * $(192)) + (((r142_row[:, None] * $(1024)) + r142_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r142_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r142_col[None, :] * $(1024)) + r142_lane[:, None]) / $(128)) % $(8)) * $(65536)) + ((((r142_col[None, :] * $(1024)) + r142_lane[:, None]) % $(128)) * $(256)) + ((((r142_col[None, :] * $(1024)) + r142_lane[:, None]) / $(1024)) % $(256))), dtype=tl.bfloat16)
      r142_value = tl.dot(r142_left, r142_right)
      r142_value = r142_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_23_ptr) + ((r142_group * $(32768)) + (r142_row[:, None] * $(256)) + r142_col[None, :]), r142_value)
    }
    r143_z = tl.arange($(32768))
    r143_value = (tl.load($(candidate_vbmm_22_ptr) + ((((r143_z / $(256)) % $(128)) * $(256)) + (r143_z % $(256))), dtype=tl.bfloat16) + tl.load($(candidate_vbmm_23_ptr) + ((((r143_z / $(256)) % $(128)) * $(256)) + (r143_z % $(256))), dtype=tl.bfloat16))
    r143_value = r143_value.to(tl.bfloat16)
    r143_value = tl.broadcast_to(r143_value, [$(32768)])
    tl.store($(candidate_vadd_16_ptr) + r143_z, r143_value)
    r144_z = tl.arange($(32768))
    r144_value = tl.where(($(0) == $(0)), tl.load($(candidate_vadd_16_ptr) + ((((r144_z / $(256)) % $(128)) * $(256)) + (r144_z % $(256))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r144_value = r144_value.to(tl.bfloat16)
    r144_value = tl.broadcast_to(r144_value, [$(32768)])
    tl.store($(candidate_vselect_backward_ptr) + r144_z, r144_value)
    r145_z = tl.arange($(8192))
    r145_value = (tl.load($(candidate_vsum_1_ptr) + ((((r145_z / $(64)) % $(128)) * $(64)) + (r145_z % $(64))), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_27_ptr) + ((((r145_z / $(64)) % $(128)) * $(64)) + (r145_z % $(64))), dtype=tl.bfloat16))
    r145_value = r145_value.to(tl.bfloat16)
    r145_value = tl.broadcast_to(r145_value, [$(8192)])
    tl.store($(candidate_vmul_29_ptr) + r145_z, r145_value)
    r146_z = tl.arange($(8192))
    r146_value = (tl.load($(candidate_vsum_1_ptr) + ((((r146_z / $(64)) % $(128)) * $(64)) + (r146_z % $(64))), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_26_ptr) + ((((r146_z / $(64)) % $(128)) * $(64)) + (r146_z % $(64))), dtype=tl.bfloat16))
    r146_value = r146_value.to(tl.bfloat16)
    r146_value = tl.broadcast_to(r146_value, [$(8192)])
    tl.store($(candidate_vmul_30_ptr) + r146_z, r146_value)
    r147_z = tl.arange($(8192))
    r147_value = tl.where(((r147_z % $(2)) == $(0)), tl.load($(candidate_vmul_29_ptr) + ((((r147_z / $(64)) % $(128)) * $(64)) + (((r147_z / $(2)) % $(32)) * $(2)) + $(1)), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r147_value = r147_value.to(tl.bfloat16)
    r147_value = tl.broadcast_to(r147_value, [$(8192)])
    tl.store($(candidate_vselect_backward_1_ptr) + r147_z, r147_value)
    r148_z = tl.arange($(4096))
    r148_value = -tl.load($(candidate_vmul_29_ptr) + ((((r148_z / $(32)) % $(128)) * $(64)) + ((r148_z % $(32)) * $(2))), dtype=tl.bfloat16)
    r148_value = r148_value.to(tl.bfloat16)
    r148_value = tl.broadcast_to(r148_value, [$(4096)])
    tl.store($(candidate_vneg_2_ptr) + r148_z, r148_value)
    r149_z = tl.arange($(8192))
    r149_value = tl.where(((r149_z % $(2)) == $(1)), tl.load($(candidate_vneg_2_ptr) + ((((r149_z / $(64)) % $(128)) * $(32)) + ((r149_z / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r149_value = r149_value.to(tl.bfloat16)
    r149_value = tl.broadcast_to(r149_value, [$(8192)])
    tl.store($(candidate_vselect_backward_2_ptr) + r149_z, r149_value)
    r150_z = tl.arange($(8192))
    r150_value = (tl.load($(candidate_vselect_backward_1_ptr) + ((((r150_z / $(64)) % $(128)) * $(64)) + (((r150_z / $(2)) % $(32)) * $(2)) + (r150_z % $(2))), dtype=tl.bfloat16) + tl.load($(candidate_vselect_backward_2_ptr) + ((((r150_z / $(64)) % $(128)) * $(64)) + (((r150_z / $(2)) % $(32)) * $(2)) + (r150_z % $(2))), dtype=tl.bfloat16))
    r150_value = r150_value.to(tl.bfloat16)
    r150_value = tl.broadcast_to(r150_value, [$(8192)])
    tl.store($(candidate_vadd_17_ptr) + r150_z, r150_value)
    r151_z = tl.arange($(8192))
    r151_value = (tl.load($(candidate_vmul_30_ptr) + ((((r151_z / $(64)) % $(128)) * $(64)) + (r151_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vadd_17_ptr) + ((((r151_z / $(64)) % $(128)) * $(64)) + (((((((r151_z / $(64)) % $(128)) * $(64)) + (r151_z % $(64))) / $(2)) % $(32)) * $(2)) + (((((r151_z / $(64)) % $(128)) * $(64)) + (r151_z % $(64))) % $(2))), dtype=tl.bfloat16))
    r151_value = r151_value.to(tl.bfloat16)
    r151_value = tl.broadcast_to(r151_value, [$(8192)])
    tl.store($(candidate_vadd_18_ptr) + r151_z, r151_value)
    r152_z = tl.arange($(65536))
    r152_value = (tl.load($(candidate_vadd_15_ptr) + ((((r152_z / $(512)) % $(128)) * $(1536)) + (((r152_z / $(64)) % $(8)) * $(192)) + (r152_z % $(64)) + $(128)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_25_ptr) + ((((r152_z / $(512)) % $(128)) * $(64)) + (r152_z % $(64))), dtype=tl.bfloat16))
    r152_value = r152_value.to(tl.bfloat16)
    r152_value = tl.broadcast_to(r152_value, [$(65536)])
    tl.store($(candidate_vmul_31_ptr) + r152_z, r152_value)
    r153_z = tl.arange($(65536))
    r153_value = (tl.load($(candidate_vadd_15_ptr) + ((((r153_z / $(512)) % $(128)) * $(1536)) + (((r153_z / $(64)) % $(8)) * $(192)) + (r153_z % $(64)) + $(128)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_24_ptr) + ((((r153_z / $(512)) % $(128)) * $(64)) + (r153_z % $(64))), dtype=tl.bfloat16))
    r153_value = r153_value.to(tl.bfloat16)
    r153_value = tl.broadcast_to(r153_value, [$(65536)])
    tl.store($(candidate_vmul_32_ptr) + r153_z, r153_value)
    r154_z = tl.arange($(65536))
    r154_value = tl.where(((r154_z % $(2)) == $(0)), tl.load($(candidate_vmul_31_ptr) + ((((r154_z / $(512)) % $(128)) * $(512)) + (((r154_z / $(64)) % $(8)) * $(64)) + (((r154_z / $(2)) % $(32)) * $(2)) + $(1)), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r154_value = r154_value.to(tl.bfloat16)
    r154_value = tl.broadcast_to(r154_value, [$(65536)])
    tl.store($(candidate_vselect_backward_3_ptr) + r154_z, r154_value)
    r155_z = tl.arange($(32768))
    r155_value = -tl.load($(candidate_vmul_31_ptr) + ((((r155_z / $(256)) % $(128)) * $(512)) + (((r155_z / $(32)) % $(8)) * $(64)) + ((r155_z % $(32)) * $(2))), dtype=tl.bfloat16)
    r155_value = r155_value.to(tl.bfloat16)
    r155_value = tl.broadcast_to(r155_value, [$(32768)])
    tl.store($(candidate_vneg_3_ptr) + r155_z, r155_value)
    r156_z = tl.arange($(65536))
    r156_value = tl.where(((r156_z % $(2)) == $(1)), tl.load($(candidate_vneg_3_ptr) + ((((r156_z / $(512)) % $(128)) * $(256)) + (((r156_z / $(64)) % $(8)) * $(32)) + ((r156_z / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r156_value = r156_value.to(tl.bfloat16)
    r156_value = tl.broadcast_to(r156_value, [$(65536)])
    tl.store($(candidate_vselect_backward_4_ptr) + r156_z, r156_value)
    r157_z = tl.arange($(65536))
    r157_value = (tl.load($(candidate_vselect_backward_3_ptr) + ((((r157_z / $(512)) % $(128)) * $(512)) + (((r157_z / $(64)) % $(8)) * $(64)) + (((r157_z / $(2)) % $(32)) * $(2)) + (r157_z % $(2))), dtype=tl.bfloat16) + tl.load($(candidate_vselect_backward_4_ptr) + ((((r157_z / $(512)) % $(128)) * $(512)) + (((r157_z / $(64)) % $(8)) * $(64)) + (((r157_z / $(2)) % $(32)) * $(2)) + (r157_z % $(2))), dtype=tl.bfloat16))
    r157_value = r157_value.to(tl.bfloat16)
    r157_value = tl.broadcast_to(r157_value, [$(65536)])
    tl.store($(candidate_vadd_19_ptr) + r157_z, r157_value)
    r158_z = tl.arange($(65536))
    r158_value = (tl.load($(candidate_vmul_32_ptr) + ((((r158_z / $(512)) % $(128)) * $(512)) + (((r158_z / $(64)) % $(8)) * $(64)) + (r158_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vadd_19_ptr) + ((((r158_z / $(512)) % $(128)) * $(512)) + (((r158_z / $(64)) % $(8)) * $(64)) + (((((((r158_z / $(512)) % $(128)) * $(512)) + (((r158_z / $(64)) % $(8)) * $(64)) + (r158_z % $(64))) / $(2)) % $(32)) * $(2)) + (((((r158_z / $(512)) % $(128)) * $(512)) + (((r158_z / $(64)) % $(8)) * $(64)) + (r158_z % $(64))) % $(2))), dtype=tl.bfloat16))
    r158_value = r158_value.to(tl.bfloat16)
    r158_value = tl.broadcast_to(r158_value, [$(65536)])
    tl.store($(candidate_vadd_20_ptr) + r158_z, r158_value)
    r159_z = tl.arange($(196608))
    r159_value = tl.where((r159_z % $(192)) < $(128), tl.load($(candidate_vadd_15_ptr) + ((((((((r159_z / $(192)) % $(128)) * $(1024)) + (((r159_z / $(24576)) % $(8)) * $(128)) + ((r159_z % $(192)) - $(0))) / $(1024)) % $(128)) * $(1536)) + (((((((r159_z / $(192)) % $(128)) * $(1024)) + (((r159_z / $(24576)) % $(8)) * $(128)) + ((r159_z % $(192)) - $(0))) / $(128)) % $(8)) * $(192)) + (((((r159_z / $(192)) % $(128)) * $(1024)) + (((r159_z / $(24576)) % $(8)) * $(128)) + ((r159_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vadd_20_ptr) + ((((((((r159_z / $(192)) % $(128)) * $(512)) + (((r159_z / $(24576)) % $(8)) * $(64)) + ((r159_z % $(192)) - $(128))) / $(512)) % $(128)) * $(512)) + (((((((r159_z / $(192)) % $(128)) * $(512)) + (((r159_z / $(24576)) % $(8)) * $(64)) + ((r159_z % $(192)) - $(128))) / $(64)) % $(8)) * $(64)) + (((((r159_z / $(192)) % $(128)) * $(512)) + (((r159_z / $(24576)) % $(8)) * $(64)) + ((r159_z % $(192)) - $(128))) % $(64))), dtype=tl.bfloat16))
    r159_value = r159_value.to(tl.bfloat16)
    r159_value = tl.broadcast_to(r159_value, [$(196608)])
    tl.store($(candidate_vcat_10_ptr) + r159_z, r159_value)
    r160_z = tl.arange($(32768))
    r160_value = tl.load($(candidate_vselect_backward_ptr) + ((((r160_z / $(256)) % $(128)) * $(256)) + (r160_z % $(256))), dtype=tl.bfloat16)
    r160_value = r160_value.to(tl.float32)
    r160_value = tl.broadcast_to(r160_value, [$(32768)])
    tl.store($(candidate_vto_copy_46_ptr) + r160_z, r160_value)
    r161_z = tl.arange($(32768))
    r161_value = (tl.load($(candidate_vto_copy_46_ptr) + ((((r161_z / $(256)) % $(128)) * $(256)) + (r161_z % $(256))), dtype=tl.float32) * tl.cast(tl.load($(candidate_vto_copy_4_ptr) + (r161_z % $(256)), dtype=tl.bfloat16), tl.float32))
    r161_value = r161_value.to(tl.float32)
    r161_value = tl.broadcast_to(r161_value, [$(32768)])
    tl.store($(candidate_vmul_33_ptr) + r161_z, r161_value)
    r162_z = tl.arange($(32768))
    r162_value = (tl.load($(candidate_vmul_33_ptr) + ((((r162_z / $(256)) % $(128)) * $(256)) + (r162_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vto_copy_22_ptr) + ((((r162_z / $(256)) % $(128)) * $(256)) + (r162_z % $(256))), dtype=tl.float32))
    r162_value = r162_value.to(tl.float32)
    r162_value = tl.broadcast_to(r162_value, [$(32768)])
    tl.store($(candidate_vmul_34_ptr) + r162_z, r162_value)
    r163_z = tl.arange($(32768))
    r163_value = (tl.load($(candidate_vmul_33_ptr) + ((((r163_z / $(256)) % $(128)) * $(256)) + (r163_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_1_ptr) + ((r163_z / $(256)) % $(128)), dtype=tl.float32))
    r163_value = r163_value.to(tl.float32)
    r163_value = tl.broadcast_to(r163_value, [$(32768)])
    tl.store($(candidate_vmul_35_ptr) + r163_z, r163_value)
    r164_z = tl.arange($(128))
    r164_lane = tl.arange($(256))
    r164_value = tl.sum(tl.load($(candidate_vmul_34_ptr) + ((r164_z[:, None] * $(256)) + r164_lane[None, :]), dtype=tl.float32), axis=1)
    r164_value = r164_value.to(tl.float32)
    r164_value = tl.broadcast_to(r164_value, [$(128)])
    tl.store($(candidate_vsum_2_ptr) + r164_z, r164_value)
    r165_z = tl.arange($(128))
    r165_value = ((tl.load($(candidate_vrsqrt_1_ptr) + r165_z, dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_1_ptr) + r165_z, dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_1_ptr) + r165_z, dtype=tl.float32)))
    r165_value = r165_value.to(tl.float32)
    r165_value = tl.broadcast_to(r165_value, [$(128)])
    tl.store($(candidate_vpow_6_ptr) + r165_z, r165_value)
    r166_z = tl.arange($(128))
    r166_value = (tl.load($(candidate_vsum_2_ptr) + r166_z, dtype=tl.float32) * -0.5)
    r166_value = r166_value.to(tl.float32)
    r166_value = tl.broadcast_to(r166_value, [$(128)])
    tl.store($(candidate_vmul_36_ptr) + r166_z, r166_value)
    r167_z = tl.arange($(128))
    r167_value = (tl.load($(candidate_vmul_36_ptr) + r167_z, dtype=tl.float32) * tl.load($(candidate_vpow_6_ptr) + r167_z, dtype=tl.float32))
    r167_value = r167_value.to(tl.float32)
    r167_value = tl.broadcast_to(r167_value, [$(128)])
    tl.store($(candidate_vmul_37_ptr) + r167_z, r167_value)
    r168_z = tl.arange($(32768))
    r168_value = (tl.load($(candidate_vmul_37_ptr) + ((r168_z / $(256)) % $(128)), dtype=tl.float32) / 256.0)
    r168_value = r168_value.to(tl.float32)
    r168_value = tl.broadcast_to(r168_value, [$(32768)])
    tl.store($(candidate_vdiv_3_ptr) + r168_z, r168_value)
    r169_z = tl.arange($(32768))
    r169_value = ((tl.load($(candidate_vto_copy_22_ptr) + ((((r169_z / $(256)) % $(128)) * $(256)) + (r169_z % $(256))), dtype=tl.float32)))
    r169_value = r169_value.to(tl.float32)
    r169_value = tl.broadcast_to(r169_value, [$(32768)])
    tl.store($(candidate_vpow_7_ptr) + r169_z, r169_value)
    r170_z = tl.arange($(32768))
    r170_value = (tl.load($(candidate_vpow_7_ptr) + ((((r170_z / $(256)) % $(128)) * $(256)) + (r170_z % $(256))), dtype=tl.float32) * 2.0)
    r170_value = r170_value.to(tl.float32)
    r170_value = tl.broadcast_to(r170_value, [$(32768)])
    tl.store($(candidate_vmul_38_ptr) + r170_z, r170_value)
    r171_z = tl.arange($(32768))
    r171_value = (tl.load($(candidate_vdiv_3_ptr) + ((((r171_z / $(256)) % $(128)) * $(256)) + (r171_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vmul_38_ptr) + ((((r171_z / $(256)) % $(128)) * $(256)) + (r171_z % $(256))), dtype=tl.float32))
    r171_value = r171_value.to(tl.float32)
    r171_value = tl.broadcast_to(r171_value, [$(32768)])
    tl.store($(candidate_vmul_39_ptr) + r171_z, r171_value)
    r172_z = tl.arange($(32768))
    r172_value = (tl.load($(candidate_vmul_35_ptr) + ((((r172_z / $(256)) % $(128)) * $(256)) + (r172_z % $(256))), dtype=tl.float32) + tl.load($(candidate_vmul_39_ptr) + ((((r172_z / $(256)) % $(128)) * $(256)) + (r172_z % $(256))), dtype=tl.float32))
    r172_value = r172_value.to(tl.float32)
    r172_value = tl.broadcast_to(r172_value, [$(32768)])
    tl.store($(candidate_vadd_21_ptr) + r172_z, r172_value)
    r173_z = tl.arange($(32768))
    r173_value = tl.load($(candidate_vadd_21_ptr) + ((((r173_z / $(256)) % $(128)) * $(256)) + (r173_z % $(256))), dtype=tl.float32)
    r173_value = r173_value.to(tl.bfloat16)
    r173_value = tl.broadcast_to(r173_value, [$(32768)])
    tl.store($(candidate_vto_copy_47_ptr) + r173_z, r173_value)
    r174_z = tl.arange($(40960))
    r174_value = tl.where((r174_z % $(320)) < $(256), tl.load($(candidate_vto_copy_47_ptr) + ((((r174_z / $(320)) % $(128)) * $(256)) + (((r174_z % $(320)) - $(0)) % $(256))), dtype=tl.bfloat16), tl.load($(candidate_vadd_18_ptr) + ((((((((r174_z / $(320)) % $(128)) * $(64)) + ((r174_z % $(320)) - $(256))) / $(64)) % $(128)) * $(64)) + (((((r174_z / $(320)) % $(128)) * $(64)) + ((r174_z % $(320)) - $(256))) % $(64))), dtype=tl.bfloat16))
    r174_value = r174_value.to(tl.bfloat16)
    r174_value = tl.broadcast_to(r174_value, [$(40960)])
    tl.store($(candidate_vcat_11_ptr) + r174_z, r174_value)
    r175_row = tl.arange($(128))
    r175_col = tl.arange($(512))
    r175_lane = tl.arange($(320))
    r175_left = tl.load($(candidate_vcat_11_ptr) + ((((((r175_row[:, None] * $(320)) + r175_lane[None, :]) / $(320)) % $(128)) * $(320)) + (((r175_row[:, None] * $(320)) + r175_lane[None, :]) % $(320))), dtype=tl.bfloat16)
    r175_right = tl.load($(candidate_vto_copy_3_ptr) + (((r175_lane[:, None] % $(320)) * $(512)) + (r175_col[None, :] % $(512))), dtype=tl.bfloat16)
    r175_value = tl.dot(r175_left, r175_right)
    r175_value = r175_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_5_ptr) + ((r175_row[:, None] * $(512)) + r175_col[None, :]), r175_value)
    r176_row = tl.arange($(128))
    r176_col = tl.arange($(256))
    r176_lane = tl.arange($(1536))
    r176_left = tl.load($(candidate_vcat_10_ptr) + ((((((((((r176_row[:, None] * $(1536)) + r176_lane[None, :]) / $(1536)) % $(128)) * $(1536)) + (((r176_row[:, None] * $(1536)) + r176_lane[None, :]) % $(1536))) / $(192)) % $(8)) * $(24576)) + (((((r176_row[:, None] * $(1536)) + r176_lane[None, :]) / $(1536)) % $(128)) * $(192)) + (((((((r176_row[:, None] * $(1536)) + r176_lane[None, :]) / $(1536)) % $(128)) * $(1536)) + (((r176_row[:, None] * $(1536)) + r176_lane[None, :]) % $(1536))) % $(192))), dtype=tl.bfloat16)
    r176_right = tl.load($(candidate_vto_copy_2_ptr) + (((r176_lane[:, None] % $(1536)) * $(256)) + (r176_col[None, :] % $(256))), dtype=tl.bfloat16)
    r176_value = tl.dot(r176_left, r176_right)
    r176_value = r176_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_6_ptr) + ((r176_row[:, None] * $(256)) + r176_col[None, :]), r176_value)
    r177_z = tl.arange($(32768))
    r177_value = tl.load($(candidate_vmm_6_ptr) + ((((r177_z / $(256)) % $(128)) * $(256)) + (r177_z % $(256))), dtype=tl.bfloat16)
    r177_value = r177_value.to(tl.float32)
    r177_value = tl.broadcast_to(r177_value, [$(32768)])
    tl.store($(candidate_vto_copy_48_ptr) + r177_z, r177_value)
    r178_z = tl.arange($(32768))
    r178_value = (tl.load($(candidate_vto_copy_48_ptr) + ((((r178_z / $(256)) % $(128)) * $(256)) + (r178_z % $(256))), dtype=tl.float32) * tl.cast(tl.load($(candidate_vto_copy_1_ptr) + (r178_z % $(256)), dtype=tl.bfloat16), tl.float32))
    r178_value = r178_value.to(tl.float32)
    r178_value = tl.broadcast_to(r178_value, [$(32768)])
    tl.store($(candidate_vmul_40_ptr) + r178_z, r178_value)
    r179_z = tl.arange($(32768))
    r179_value = (tl.load($(candidate_vmul_40_ptr) + ((((r179_z / $(256)) % $(128)) * $(256)) + (r179_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vto_copy_20_ptr) + ((((r179_z / $(256)) % $(128)) * $(256)) + (r179_z % $(256))), dtype=tl.float32))
    r179_value = r179_value.to(tl.float32)
    r179_value = tl.broadcast_to(r179_value, [$(32768)])
    tl.store($(candidate_vmul_41_ptr) + r179_z, r179_value)
    r180_z = tl.arange($(32768))
    r180_value = (tl.load($(candidate_vmul_40_ptr) + ((((r180_z / $(256)) % $(128)) * $(256)) + (r180_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_ptr) + ((r180_z / $(256)) % $(128)), dtype=tl.float32))
    r180_value = r180_value.to(tl.float32)
    r180_value = tl.broadcast_to(r180_value, [$(32768)])
    tl.store($(candidate_vmul_42_ptr) + r180_z, r180_value)
    r181_z = tl.arange($(128))
    r181_lane = tl.arange($(256))
    r181_value = tl.sum(tl.load($(candidate_vmul_41_ptr) + ((r181_z[:, None] * $(256)) + r181_lane[None, :]), dtype=tl.float32), axis=1)
    r181_value = r181_value.to(tl.float32)
    r181_value = tl.broadcast_to(r181_value, [$(128)])
    tl.store($(candidate_vsum_3_ptr) + r181_z, r181_value)
    r182_z = tl.arange($(128))
    r182_value = ((tl.load($(candidate_vrsqrt_ptr) + r182_z, dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_ptr) + r182_z, dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_ptr) + r182_z, dtype=tl.float32)))
    r182_value = r182_value.to(tl.float32)
    r182_value = tl.broadcast_to(r182_value, [$(128)])
    tl.store($(candidate_vpow_8_ptr) + r182_z, r182_value)
    r183_z = tl.arange($(128))
    r183_value = (tl.load($(candidate_vsum_3_ptr) + r183_z, dtype=tl.float32) * -0.5)
    r183_value = r183_value.to(tl.float32)
    r183_value = tl.broadcast_to(r183_value, [$(128)])
    tl.store($(candidate_vmul_43_ptr) + r183_z, r183_value)
    r184_z = tl.arange($(128))
    r184_value = (tl.load($(candidate_vmul_43_ptr) + r184_z, dtype=tl.float32) * tl.load($(candidate_vpow_8_ptr) + r184_z, dtype=tl.float32))
    r184_value = r184_value.to(tl.float32)
    r184_value = tl.broadcast_to(r184_value, [$(128)])
    tl.store($(candidate_vmul_44_ptr) + r184_z, r184_value)
    r185_z = tl.arange($(32768))
    r185_value = (tl.load($(candidate_vmul_44_ptr) + ((r185_z / $(256)) % $(128)), dtype=tl.float32) / 256.0)
    r185_value = r185_value.to(tl.float32)
    r185_value = tl.broadcast_to(r185_value, [$(32768)])
    tl.store($(candidate_vdiv_4_ptr) + r185_z, r185_value)
    r186_z = tl.arange($(32768))
    r186_value = ((tl.load($(candidate_vto_copy_20_ptr) + ((((r186_z / $(256)) % $(128)) * $(256)) + (r186_z % $(256))), dtype=tl.float32)))
    r186_value = r186_value.to(tl.float32)
    r186_value = tl.broadcast_to(r186_value, [$(32768)])
    tl.store($(candidate_vpow_9_ptr) + r186_z, r186_value)
    r187_z = tl.arange($(32768))
    r187_value = (tl.load($(candidate_vpow_9_ptr) + ((((r187_z / $(256)) % $(128)) * $(256)) + (r187_z % $(256))), dtype=tl.float32) * 2.0)
    r187_value = r187_value.to(tl.float32)
    r187_value = tl.broadcast_to(r187_value, [$(32768)])
    tl.store($(candidate_vmul_45_ptr) + r187_z, r187_value)
    r188_z = tl.arange($(32768))
    r188_value = (tl.load($(candidate_vdiv_4_ptr) + ((((r188_z / $(256)) % $(128)) * $(256)) + (r188_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vmul_45_ptr) + ((((r188_z / $(256)) % $(128)) * $(256)) + (r188_z % $(256))), dtype=tl.float32))
    r188_value = r188_value.to(tl.float32)
    r188_value = tl.broadcast_to(r188_value, [$(32768)])
    tl.store($(candidate_vmul_46_ptr) + r188_z, r188_value)
    r189_z = tl.arange($(32768))
    r189_value = (tl.load($(candidate_vmul_42_ptr) + ((((r189_z / $(256)) % $(128)) * $(256)) + (r189_z % $(256))), dtype=tl.float32) + tl.load($(candidate_vmul_46_ptr) + ((((r189_z / $(256)) % $(128)) * $(256)) + (r189_z % $(256))), dtype=tl.float32))
    r189_value = r189_value.to(tl.float32)
    r189_value = tl.broadcast_to(r189_value, [$(32768)])
    tl.store($(candidate_vadd_22_ptr) + r189_z, r189_value)
    r190_z = tl.arange($(32768))
    r190_value = tl.load($(candidate_vadd_22_ptr) + ((((r190_z / $(256)) % $(128)) * $(256)) + (r190_z % $(256))), dtype=tl.float32)
    r190_value = r190_value.to(tl.bfloat16)
    r190_value = tl.broadcast_to(r190_value, [$(32768)])
    tl.store($(candidate_vto_copy_49_ptr) + r190_z, r190_value)
    r191_row = tl.arange($(128))
    r191_col = tl.arange($(512))
    r191_lane = tl.arange($(256))
    r191_left = tl.load($(candidate_vto_copy_49_ptr) + ((((((r191_row[:, None] * $(256)) + r191_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r191_row[:, None] * $(256)) + r191_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r191_right = tl.load($(candidate_vto_copy_ptr) + (((r191_lane[:, None] % $(256)) * $(512)) + (r191_col[None, :] % $(512))), dtype=tl.bfloat16)
    r191_value = tl.dot(r191_left, r191_right)
    r191_value = r191_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_7_ptr) + ((r191_row[:, None] * $(512)) + r191_col[None, :]), r191_value)
    r192_z = tl.arange($(65536))
    r192_value = (tl.load($(candidate_vmm_5_ptr) + ((((r192_z / $(512)) % $(128)) * $(512)) + (r192_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vmm_7_ptr) + ((((r192_z / $(512)) % $(128)) * $(512)) + (r192_z % $(512))), dtype=tl.bfloat16))
    r192_value = r192_value.to(tl.bfloat16)
    r192_value = tl.broadcast_to(r192_value, [$(65536)])
    tl.store($(candidate_vadd_23_ptr) + r192_z, r192_value)
    z = tl.arange($(65536))
    value = tl.load($(candidate_vadd_23_ptr) + ((((z / $(512)) % $(128)) * $(512)) + (z % $(512))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(65536)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_mla_fa4_te_rope_thd_nonabsorb_dx
