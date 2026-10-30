import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_mla_fa4_te_rope_cp_nonabsorb_dx
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
  let vsub_9_ptr : RegionName := "vsub_9"
  let vmul_19_ptr : RegionName := "vmul_19"
  let vmul_20_ptr : RegionName := "vmul_20"
  let vadd_5_ptr : RegionName := "vadd_5"
  let vcat_5_ptr : RegionName := "vcat_5"
  let vmul_21_ptr : RegionName := "vmul_21"
  let vmul_22_ptr : RegionName := "vmul_22"
  let vsub_10_ptr : RegionName := "vsub_10"
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
    r17_value = (tl.load($(vcat_ptr) + r17_z, dtype=tl.int64) + tl.full([], $(2), dtype=tl.int64))
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
    tl.store($(vsub_9_ptr) + r63_z, r63_value)
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
    r67_value = tl.where((r67_z % $(64)) < $(32), tl.load($(vsub_9_ptr) + ((((r67_z / $(8192)) % $(8)) * $(4096)) + (((r67_z / $(64)) % $(128)) * $(32)) + (((r67_z % $(64)) - $(0)) % $(32))), dtype=tl.bfloat16), tl.load($(vadd_5_ptr) + ((((r67_z / $(8192)) % $(8)) * $(4096)) + (((r67_z / $(64)) % $(128)) * $(32)) + (((r67_z % $(64)) - $(32)) % $(32))), dtype=tl.bfloat16))
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
    tl.store($(vsub_10_ptr) + r70_z, r70_value)
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
    r74_value = tl.where((r74_z % $(64)) < $(32), tl.load($(vsub_10_ptr) + ((((r74_z / $(64)) % $(128)) * $(32)) + (((r74_z % $(64)) - $(0)) % $(32))), dtype=tl.bfloat16), tl.load($(vadd_6_ptr) + ((((r74_z / $(64)) % $(128)) * $(32)) + (((r74_z % $(64)) - $(32)) % $(32))), dtype=tl.bfloat16))
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
  let constantee33028f0f_ptr : RegionName := "constantee33028f0f"
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
  let vsub_9_ptr : RegionName := "vsub_9"
  let vmul_19_ptr : RegionName := "vmul_19"
  let vmul_20_ptr : RegionName := "vmul_20"
  let vadd_5_ptr : RegionName := "vadd_5"
  let vcat_5_ptr : RegionName := "vcat_5"
  let vmul_21_ptr : RegionName := "vmul_21"
  let vmul_22_ptr : RegionName := "vmul_22"
  let vsub_10_ptr : RegionName := "vsub_10"
  let vmul_23_ptr : RegionName := "vmul_23"
  let vmul_24_ptr : RegionName := "vmul_24"
  let vadd_6_ptr : RegionName := "vadd_6"
  let vcat_6_ptr : RegionName := "vcat_6"
  let vcat_7_ptr : RegionName := "vcat_7"
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
  let vmul_47_ptr : RegionName := "vmul_47"
  let vmul_48_ptr : RegionName := "vmul_48"
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
  let candidate_vcat_ptr : RegionName := "candidate_vcat"
  let candidate_vadd_2_ptr : RegionName := "candidate_vadd_2"
  let candidate_vsort_value_ptr : RegionName := "candidate_vsort_value"
  let candidate_vsort_index_ptr : RegionName := "candidate_vsort_index"
  let candidate_vindex_ptr : RegionName := "candidate_vindex"
  let candidate_vsub_8_ptr : RegionName := "candidate_vsub_8"
  let candidate_vindex_1_ptr : RegionName := "candidate_vindex_1"
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
  let candidate_vto_copy_24_ptr : RegionName := "candidate_vto_copy_24"
  let candidate_vadd_5_ptr : RegionName := "candidate_vadd_5"
  let candidate_vmul_18_ptr : RegionName := "candidate_vmul_18"
  let candidate_vcos_2_ptr : RegionName := "candidate_vcos_2"
  let candidate_vto_copy_25_ptr : RegionName := "candidate_vto_copy_25"
  let candidate_vsin_2_ptr : RegionName := "candidate_vsin_2"
  let candidate_vto_copy_26_ptr : RegionName := "candidate_vto_copy_26"
  let candidate_vneg_ptr : RegionName := "candidate_vneg"
  let candidate_vstack_ptr : RegionName := "candidate_vstack"
  let candidate_vmul_19_ptr : RegionName := "candidate_vmul_19"
  let candidate_vmul_20_ptr : RegionName := "candidate_vmul_20"
  let candidate_vadd_6_ptr : RegionName := "candidate_vadd_6"
  let candidate_vcos_3_ptr : RegionName := "candidate_vcos_3"
  let candidate_vto_copy_27_ptr : RegionName := "candidate_vto_copy_27"
  let candidate_vsin_3_ptr : RegionName := "candidate_vsin_3"
  let candidate_vto_copy_28_ptr : RegionName := "candidate_vto_copy_28"
  let candidate_vneg_1_ptr : RegionName := "candidate_vneg_1"
  let candidate_vstack_1_ptr : RegionName := "candidate_vstack_1"
  let candidate_vmul_21_ptr : RegionName := "candidate_vmul_21"
  let candidate_vmul_22_ptr : RegionName := "candidate_vmul_22"
  let candidate_vadd_7_ptr : RegionName := "candidate_vadd_7"
  let candidate_vcat_5_ptr : RegionName := "candidate_vcat_5"
  let candidate_vindex_2_ptr : RegionName := "candidate_vindex_2"
  let candidate_vbmm_2_ptr : RegionName := "candidate_vbmm_2"
  let candidate_vbmm_3_ptr : RegionName := "candidate_vbmm_3"
  let candidate_vcat_6_ptr : RegionName := "candidate_vcat_6"
  let candidate_vcat_7_ptr : RegionName := "candidate_vcat_7"
  let candidate_vto_copy_29_ptr : RegionName := "candidate_vto_copy_29"
  let candidate_vto_copy_30_ptr : RegionName := "candidate_vto_copy_30"
  let candidate_vbmm_4_ptr : RegionName := "candidate_vbmm_4"
  let candidate_vmul_23_ptr : RegionName := "candidate_vmul_23"
  let candidate_vadd_8_ptr : RegionName := "candidate_vadd_8"
  let candidate_vsub_9_ptr : RegionName := "candidate_vsub_9"
  let candidate_vgt_ptr : RegionName := "candidate_vgt"
  let candidate_vmasked_fill_ptr : RegionName := "candidate_vmasked_fill"
  let candidate_vsoftmax_ptr : RegionName := "candidate_vsoftmax"
  let candidate_vto_copy_31_ptr : RegionName := "candidate_vto_copy_31"
  let candidate_vto_copy_32_ptr : RegionName := "candidate_vto_copy_32"
  let candidate_vto_copy_33_ptr : RegionName := "candidate_vto_copy_33"
  let candidate_vbmm_6_ptr : RegionName := "candidate_vbmm_6"
  let candidate_vmul_24_ptr : RegionName := "candidate_vmul_24"
  let candidate_vadd_9_ptr : RegionName := "candidate_vadd_9"
  let candidate_vsub_10_ptr : RegionName := "candidate_vsub_10"
  let candidate_vgt_1_ptr : RegionName := "candidate_vgt_1"
  let candidate_vmasked_fill_1_ptr : RegionName := "candidate_vmasked_fill_1"
  let candidate_vsoftmax_1_ptr : RegionName := "candidate_vsoftmax_1"
  let candidate_vto_copy_34_ptr : RegionName := "candidate_vto_copy_34"
  let candidate_vto_copy_35_ptr : RegionName := "candidate_vto_copy_35"
  let candidate_vto_copy_36_ptr : RegionName := "candidate_vto_copy_36"
  let candidate_vbmm_8_ptr : RegionName := "candidate_vbmm_8"
  let candidate_vmul_25_ptr : RegionName := "candidate_vmul_25"
  let candidate_vadd_10_ptr : RegionName := "candidate_vadd_10"
  let candidate_vsub_11_ptr : RegionName := "candidate_vsub_11"
  let candidate_vgt_2_ptr : RegionName := "candidate_vgt_2"
  let candidate_vmasked_fill_2_ptr : RegionName := "candidate_vmasked_fill_2"
  let candidate_vsoftmax_2_ptr : RegionName := "candidate_vsoftmax_2"
  let candidate_vto_copy_37_ptr : RegionName := "candidate_vto_copy_37"
  let candidate_vbmm_10_ptr : RegionName := "candidate_vbmm_10"
  let candidate_vbmm_11_ptr : RegionName := "candidate_vbmm_11"
  let candidate_vcat_9_ptr : RegionName := "candidate_vcat_9"
  let candidate_vcat_10_ptr : RegionName := "candidate_vcat_10"
  let candidate_vto_copy_38_ptr : RegionName := "candidate_vto_copy_38"
  let candidate_vto_copy_39_ptr : RegionName := "candidate_vto_copy_39"
  let candidate_vbmm_12_ptr : RegionName := "candidate_vbmm_12"
  let candidate_vmul_26_ptr : RegionName := "candidate_vmul_26"
  let candidate_vadd_11_ptr : RegionName := "candidate_vadd_11"
  let candidate_vsub_12_ptr : RegionName := "candidate_vsub_12"
  let candidate_vgt_3_ptr : RegionName := "candidate_vgt_3"
  let candidate_vmasked_fill_3_ptr : RegionName := "candidate_vmasked_fill_3"
  let candidate_vsoftmax_3_ptr : RegionName := "candidate_vsoftmax_3"
  let candidate_vto_copy_40_ptr : RegionName := "candidate_vto_copy_40"
  let candidate_vto_copy_41_ptr : RegionName := "candidate_vto_copy_41"
  let candidate_vto_copy_42_ptr : RegionName := "candidate_vto_copy_42"
  let candidate_vbmm_14_ptr : RegionName := "candidate_vbmm_14"
  let candidate_vmul_27_ptr : RegionName := "candidate_vmul_27"
  let candidate_vadd_12_ptr : RegionName := "candidate_vadd_12"
  let candidate_vsub_13_ptr : RegionName := "candidate_vsub_13"
  let candidate_vgt_4_ptr : RegionName := "candidate_vgt_4"
  let candidate_vmasked_fill_4_ptr : RegionName := "candidate_vmasked_fill_4"
  let candidate_vsoftmax_4_ptr : RegionName := "candidate_vsoftmax_4"
  let candidate_vto_copy_43_ptr : RegionName := "candidate_vto_copy_43"
  let candidate_vto_copy_44_ptr : RegionName := "candidate_vto_copy_44"
  let candidate_vto_copy_45_ptr : RegionName := "candidate_vto_copy_45"
  let candidate_vbmm_16_ptr : RegionName := "candidate_vbmm_16"
  let candidate_vmul_28_ptr : RegionName := "candidate_vmul_28"
  let candidate_vadd_13_ptr : RegionName := "candidate_vadd_13"
  let candidate_vsub_14_ptr : RegionName := "candidate_vsub_14"
  let candidate_vgt_5_ptr : RegionName := "candidate_vgt_5"
  let candidate_vmasked_fill_5_ptr : RegionName := "candidate_vmasked_fill_5"
  let candidate_vsoftmax_5_ptr : RegionName := "candidate_vsoftmax_5"
  let candidate_vto_copy_46_ptr : RegionName := "candidate_vto_copy_46"
  let candidate_vbmm_18_ptr : RegionName := "candidate_vbmm_18"
  let candidate_vbmm_19_ptr : RegionName := "candidate_vbmm_19"
  let candidate_vcat_12_ptr : RegionName := "candidate_vcat_12"
  let candidate_vcat_13_ptr : RegionName := "candidate_vcat_13"
  let candidate_vto_copy_47_ptr : RegionName := "candidate_vto_copy_47"
  let candidate_vto_copy_48_ptr : RegionName := "candidate_vto_copy_48"
  let candidate_vbmm_20_ptr : RegionName := "candidate_vbmm_20"
  let candidate_vmul_29_ptr : RegionName := "candidate_vmul_29"
  let candidate_vadd_14_ptr : RegionName := "candidate_vadd_14"
  let candidate_vsub_15_ptr : RegionName := "candidate_vsub_15"
  let candidate_vgt_6_ptr : RegionName := "candidate_vgt_6"
  let candidate_vmasked_fill_6_ptr : RegionName := "candidate_vmasked_fill_6"
  let candidate_vsoftmax_6_ptr : RegionName := "candidate_vsoftmax_6"
  let candidate_vto_copy_49_ptr : RegionName := "candidate_vto_copy_49"
  let candidate_vto_copy_50_ptr : RegionName := "candidate_vto_copy_50"
  let candidate_vto_copy_51_ptr : RegionName := "candidate_vto_copy_51"
  let candidate_vbmm_22_ptr : RegionName := "candidate_vbmm_22"
  let candidate_vmul_30_ptr : RegionName := "candidate_vmul_30"
  let candidate_vadd_15_ptr : RegionName := "candidate_vadd_15"
  let candidate_vsub_16_ptr : RegionName := "candidate_vsub_16"
  let candidate_vgt_7_ptr : RegionName := "candidate_vgt_7"
  let candidate_vmasked_fill_7_ptr : RegionName := "candidate_vmasked_fill_7"
  let candidate_vsoftmax_7_ptr : RegionName := "candidate_vsoftmax_7"
  let candidate_vto_copy_52_ptr : RegionName := "candidate_vto_copy_52"
  let candidate_vto_copy_53_ptr : RegionName := "candidate_vto_copy_53"
  let candidate_vto_copy_54_ptr : RegionName := "candidate_vto_copy_54"
  let candidate_vbmm_24_ptr : RegionName := "candidate_vbmm_24"
  let candidate_vmul_31_ptr : RegionName := "candidate_vmul_31"
  let candidate_vadd_16_ptr : RegionName := "candidate_vadd_16"
  let candidate_vsub_17_ptr : RegionName := "candidate_vsub_17"
  let candidate_vgt_8_ptr : RegionName := "candidate_vgt_8"
  let candidate_vmasked_fill_8_ptr : RegionName := "candidate_vmasked_fill_8"
  let candidate_vsoftmax_8_ptr : RegionName := "candidate_vsoftmax_8"
  let candidate_vto_copy_55_ptr : RegionName := "candidate_vto_copy_55"
  let candidate_vbmm_26_ptr : RegionName := "candidate_vbmm_26"
  let candidate_vbmm_27_ptr : RegionName := "candidate_vbmm_27"
  let candidate_vcat_15_ptr : RegionName := "candidate_vcat_15"
  let candidate_vcat_16_ptr : RegionName := "candidate_vcat_16"
  let candidate_vto_copy_56_ptr : RegionName := "candidate_vto_copy_56"
  let candidate_vto_copy_57_ptr : RegionName := "candidate_vto_copy_57"
  let candidate_vbmm_28_ptr : RegionName := "candidate_vbmm_28"
  let candidate_vmul_32_ptr : RegionName := "candidate_vmul_32"
  let candidate_vadd_17_ptr : RegionName := "candidate_vadd_17"
  let candidate_vsub_18_ptr : RegionName := "candidate_vsub_18"
  let candidate_vgt_9_ptr : RegionName := "candidate_vgt_9"
  let candidate_vmasked_fill_9_ptr : RegionName := "candidate_vmasked_fill_9"
  let candidate_vsoftmax_9_ptr : RegionName := "candidate_vsoftmax_9"
  let candidate_vto_copy_58_ptr : RegionName := "candidate_vto_copy_58"
  let candidate_vto_copy_59_ptr : RegionName := "candidate_vto_copy_59"
  let candidate_vto_copy_60_ptr : RegionName := "candidate_vto_copy_60"
  let candidate_vbmm_30_ptr : RegionName := "candidate_vbmm_30"
  let candidate_vmul_33_ptr : RegionName := "candidate_vmul_33"
  let candidate_vadd_18_ptr : RegionName := "candidate_vadd_18"
  let candidate_vsub_19_ptr : RegionName := "candidate_vsub_19"
  let candidate_vgt_10_ptr : RegionName := "candidate_vgt_10"
  let candidate_vmasked_fill_10_ptr : RegionName := "candidate_vmasked_fill_10"
  let candidate_vsoftmax_10_ptr : RegionName := "candidate_vsoftmax_10"
  let candidate_vto_copy_61_ptr : RegionName := "candidate_vto_copy_61"
  let candidate_vto_copy_62_ptr : RegionName := "candidate_vto_copy_62"
  let candidate_vto_copy_63_ptr : RegionName := "candidate_vto_copy_63"
  let candidate_vbmm_32_ptr : RegionName := "candidate_vbmm_32"
  let candidate_vmul_34_ptr : RegionName := "candidate_vmul_34"
  let candidate_vadd_19_ptr : RegionName := "candidate_vadd_19"
  let candidate_vsub_20_ptr : RegionName := "candidate_vsub_20"
  let candidate_vgt_11_ptr : RegionName := "candidate_vgt_11"
  let candidate_vmasked_fill_11_ptr : RegionName := "candidate_vmasked_fill_11"
  let candidate_vsoftmax_11_ptr : RegionName := "candidate_vsoftmax_11"
  let candidate_vto_copy_64_ptr : RegionName := "candidate_vto_copy_64"
  let candidate_vindex_put_ptr : RegionName := "candidate_vindex_put"
  let candidate_vmm_4_ptr : RegionName := "candidate_vmm_4"
  let candidate_vbmm_34_ptr : RegionName := "candidate_vbmm_34"
  let candidate_vbmm_35_ptr : RegionName := "candidate_vbmm_35"
  let candidate_vto_copy_65_ptr : RegionName := "candidate_vto_copy_65"
  let candidate_vsoftmax_backward_data_ptr : RegionName := "candidate_vsoftmax_backward_data"
  let candidate_vmasked_fill_12_ptr : RegionName := "candidate_vmasked_fill_12"
  let candidate_vmul_35_ptr : RegionName := "candidate_vmul_35"
  let candidate_vbmm_36_ptr : RegionName := "candidate_vbmm_36"
  let candidate_vbmm_37_ptr : RegionName := "candidate_vbmm_37"
  let candidate_vslice_backward_ptr : RegionName := "candidate_vslice_backward"
  let candidate_vto_copy_66_ptr : RegionName := "candidate_vto_copy_66"
  let candidate_vslice_backward_1_ptr : RegionName := "candidate_vslice_backward_1"
  let candidate_vto_copy_67_ptr : RegionName := "candidate_vto_copy_67"
  let candidate_vslice_backward_2_ptr : RegionName := "candidate_vslice_backward_2"
  let candidate_vbmm_38_ptr : RegionName := "candidate_vbmm_38"
  let candidate_vbmm_39_ptr : RegionName := "candidate_vbmm_39"
  let candidate_vto_copy_68_ptr : RegionName := "candidate_vto_copy_68"
  let candidate_vsoftmax_backward_data_1_ptr : RegionName := "candidate_vsoftmax_backward_data_1"
  let candidate_vmasked_fill_13_ptr : RegionName := "candidate_vmasked_fill_13"
  let candidate_vmul_36_ptr : RegionName := "candidate_vmul_36"
  let candidate_vbmm_40_ptr : RegionName := "candidate_vbmm_40"
  let candidate_vbmm_41_ptr : RegionName := "candidate_vbmm_41"
  let candidate_vslice_backward_3_ptr : RegionName := "candidate_vslice_backward_3"
  let candidate_vadd_20_ptr : RegionName := "candidate_vadd_20"
  let candidate_vto_copy_69_ptr : RegionName := "candidate_vto_copy_69"
  let candidate_vslice_backward_4_ptr : RegionName := "candidate_vslice_backward_4"
  let candidate_vadd_21_ptr : RegionName := "candidate_vadd_21"
  let candidate_vto_copy_70_ptr : RegionName := "candidate_vto_copy_70"
  let candidate_vslice_backward_5_ptr : RegionName := "candidate_vslice_backward_5"
  let candidate_vadd_22_ptr : RegionName := "candidate_vadd_22"
  let candidate_vbmm_42_ptr : RegionName := "candidate_vbmm_42"
  let candidate_vbmm_43_ptr : RegionName := "candidate_vbmm_43"
  let candidate_vto_copy_71_ptr : RegionName := "candidate_vto_copy_71"
  let candidate_vsoftmax_backward_data_2_ptr : RegionName := "candidate_vsoftmax_backward_data_2"
  let candidate_vmasked_fill_14_ptr : RegionName := "candidate_vmasked_fill_14"
  let candidate_vmul_37_ptr : RegionName := "candidate_vmul_37"
  let candidate_vbmm_44_ptr : RegionName := "candidate_vbmm_44"
  let candidate_vbmm_45_ptr : RegionName := "candidate_vbmm_45"
  let candidate_vslice_backward_6_ptr : RegionName := "candidate_vslice_backward_6"
  let candidate_vadd_23_ptr : RegionName := "candidate_vadd_23"
  let candidate_vto_copy_72_ptr : RegionName := "candidate_vto_copy_72"
  let candidate_vslice_backward_7_ptr : RegionName := "candidate_vslice_backward_7"
  let candidate_vadd_24_ptr : RegionName := "candidate_vadd_24"
  let candidate_vto_copy_73_ptr : RegionName := "candidate_vto_copy_73"
  let candidate_vslice_backward_8_ptr : RegionName := "candidate_vslice_backward_8"
  let candidate_vadd_25_ptr : RegionName := "candidate_vadd_25"
  let candidate_vfill_ptr : RegionName := "candidate_vfill"
  let candidate_vslice_scatter_ptr : RegionName := "candidate_vslice_scatter"
  let candidate_vfill_1_ptr : RegionName := "candidate_vfill_1"
  let candidate_vslice_scatter_1_ptr : RegionName := "candidate_vslice_scatter_1"
  let candidate_vfill_2_ptr : RegionName := "candidate_vfill_2"
  let candidate_vslice_scatter_2_ptr : RegionName := "candidate_vslice_scatter_2"
  let candidate_vfill_3_ptr : RegionName := "candidate_vfill_3"
  let candidate_vslice_scatter_3_ptr : RegionName := "candidate_vslice_scatter_3"
  let candidate_vfill_4_ptr : RegionName := "candidate_vfill_4"
  let candidate_vslice_scatter_4_ptr : RegionName := "candidate_vslice_scatter_4"
  let candidate_vfill_5_ptr : RegionName := "candidate_vfill_5"
  let candidate_vslice_scatter_5_ptr : RegionName := "candidate_vslice_scatter_5"
  let candidate_vsum_1_ptr : RegionName := "candidate_vsum_1"
  let candidate_vbmm_46_ptr : RegionName := "candidate_vbmm_46"
  let candidate_vbmm_47_ptr : RegionName := "candidate_vbmm_47"
  let candidate_vadd_26_ptr : RegionName := "candidate_vadd_26"
  let candidate_vselect_backward_ptr : RegionName := "candidate_vselect_backward"
  let candidate_vslice_backward_9_ptr : RegionName := "candidate_vslice_backward_9"
  let candidate_vslice_backward_10_ptr : RegionName := "candidate_vslice_backward_10"
  let candidate_vslice_backward_11_ptr : RegionName := "candidate_vslice_backward_11"
  let candidate_vslice_backward_12_ptr : RegionName := "candidate_vslice_backward_12"
  let candidate_vbmm_48_ptr : RegionName := "candidate_vbmm_48"
  let candidate_vbmm_49_ptr : RegionName := "candidate_vbmm_49"
  let candidate_vto_copy_74_ptr : RegionName := "candidate_vto_copy_74"
  let candidate_vsoftmax_backward_data_3_ptr : RegionName := "candidate_vsoftmax_backward_data_3"
  let candidate_vmasked_fill_15_ptr : RegionName := "candidate_vmasked_fill_15"
  let candidate_vmul_38_ptr : RegionName := "candidate_vmul_38"
  let candidate_vbmm_50_ptr : RegionName := "candidate_vbmm_50"
  let candidate_vbmm_51_ptr : RegionName := "candidate_vbmm_51"
  let candidate_vslice_backward_13_ptr : RegionName := "candidate_vslice_backward_13"
  let candidate_vto_copy_75_ptr : RegionName := "candidate_vto_copy_75"
  let candidate_vslice_backward_14_ptr : RegionName := "candidate_vslice_backward_14"
  let candidate_vto_copy_76_ptr : RegionName := "candidate_vto_copy_76"
  let candidate_vslice_backward_15_ptr : RegionName := "candidate_vslice_backward_15"
  let candidate_vbmm_52_ptr : RegionName := "candidate_vbmm_52"
  let candidate_vbmm_53_ptr : RegionName := "candidate_vbmm_53"
  let candidate_vto_copy_77_ptr : RegionName := "candidate_vto_copy_77"
  let candidate_vsoftmax_backward_data_4_ptr : RegionName := "candidate_vsoftmax_backward_data_4"
  let candidate_vmasked_fill_16_ptr : RegionName := "candidate_vmasked_fill_16"
  let candidate_vmul_39_ptr : RegionName := "candidate_vmul_39"
  let candidate_vbmm_54_ptr : RegionName := "candidate_vbmm_54"
  let candidate_vbmm_55_ptr : RegionName := "candidate_vbmm_55"
  let candidate_vslice_backward_16_ptr : RegionName := "candidate_vslice_backward_16"
  let candidate_vadd_27_ptr : RegionName := "candidate_vadd_27"
  let candidate_vto_copy_78_ptr : RegionName := "candidate_vto_copy_78"
  let candidate_vslice_backward_17_ptr : RegionName := "candidate_vslice_backward_17"
  let candidate_vadd_28_ptr : RegionName := "candidate_vadd_28"
  let candidate_vto_copy_79_ptr : RegionName := "candidate_vto_copy_79"
  let candidate_vslice_backward_18_ptr : RegionName := "candidate_vslice_backward_18"
  let candidate_vadd_29_ptr : RegionName := "candidate_vadd_29"
  let candidate_vbmm_56_ptr : RegionName := "candidate_vbmm_56"
  let candidate_vbmm_57_ptr : RegionName := "candidate_vbmm_57"
  let candidate_vto_copy_80_ptr : RegionName := "candidate_vto_copy_80"
  let candidate_vsoftmax_backward_data_5_ptr : RegionName := "candidate_vsoftmax_backward_data_5"
  let candidate_vmasked_fill_17_ptr : RegionName := "candidate_vmasked_fill_17"
  let candidate_vmul_40_ptr : RegionName := "candidate_vmul_40"
  let candidate_vbmm_58_ptr : RegionName := "candidate_vbmm_58"
  let candidate_vbmm_59_ptr : RegionName := "candidate_vbmm_59"
  let candidate_vslice_backward_19_ptr : RegionName := "candidate_vslice_backward_19"
  let candidate_vadd_30_ptr : RegionName := "candidate_vadd_30"
  let candidate_vto_copy_81_ptr : RegionName := "candidate_vto_copy_81"
  let candidate_vslice_backward_20_ptr : RegionName := "candidate_vslice_backward_20"
  let candidate_vadd_31_ptr : RegionName := "candidate_vadd_31"
  let candidate_vto_copy_82_ptr : RegionName := "candidate_vto_copy_82"
  let candidate_vslice_backward_21_ptr : RegionName := "candidate_vslice_backward_21"
  let candidate_vadd_32_ptr : RegionName := "candidate_vadd_32"
  let candidate_vfill_6_ptr : RegionName := "candidate_vfill_6"
  let candidate_vslice_scatter_6_ptr : RegionName := "candidate_vslice_scatter_6"
  let candidate_vfill_7_ptr : RegionName := "candidate_vfill_7"
  let candidate_vslice_scatter_7_ptr : RegionName := "candidate_vslice_scatter_7"
  let candidate_vfill_8_ptr : RegionName := "candidate_vfill_8"
  let candidate_vslice_scatter_8_ptr : RegionName := "candidate_vslice_scatter_8"
  let candidate_vfill_9_ptr : RegionName := "candidate_vfill_9"
  let candidate_vslice_scatter_9_ptr : RegionName := "candidate_vslice_scatter_9"
  let candidate_vfill_10_ptr : RegionName := "candidate_vfill_10"
  let candidate_vslice_scatter_10_ptr : RegionName := "candidate_vslice_scatter_10"
  let candidate_vfill_11_ptr : RegionName := "candidate_vfill_11"
  let candidate_vslice_scatter_11_ptr : RegionName := "candidate_vslice_scatter_11"
  let candidate_vsum_2_ptr : RegionName := "candidate_vsum_2"
  let candidate_vbmm_60_ptr : RegionName := "candidate_vbmm_60"
  let candidate_vbmm_61_ptr : RegionName := "candidate_vbmm_61"
  let candidate_vadd_33_ptr : RegionName := "candidate_vadd_33"
  let candidate_vselect_backward_1_ptr : RegionName := "candidate_vselect_backward_1"
  let candidate_vslice_backward_22_ptr : RegionName := "candidate_vslice_backward_22"
  let candidate_vadd_34_ptr : RegionName := "candidate_vadd_34"
  let candidate_vslice_backward_23_ptr : RegionName := "candidate_vslice_backward_23"
  let candidate_vadd_35_ptr : RegionName := "candidate_vadd_35"
  let candidate_vslice_backward_24_ptr : RegionName := "candidate_vslice_backward_24"
  let candidate_vadd_36_ptr : RegionName := "candidate_vadd_36"
  let candidate_vslice_backward_25_ptr : RegionName := "candidate_vslice_backward_25"
  let candidate_vadd_37_ptr : RegionName := "candidate_vadd_37"
  let candidate_vbmm_62_ptr : RegionName := "candidate_vbmm_62"
  let candidate_vbmm_63_ptr : RegionName := "candidate_vbmm_63"
  let candidate_vto_copy_83_ptr : RegionName := "candidate_vto_copy_83"
  let candidate_vsoftmax_backward_data_6_ptr : RegionName := "candidate_vsoftmax_backward_data_6"
  let candidate_vmasked_fill_18_ptr : RegionName := "candidate_vmasked_fill_18"
  let candidate_vmul_41_ptr : RegionName := "candidate_vmul_41"
  let candidate_vbmm_64_ptr : RegionName := "candidate_vbmm_64"
  let candidate_vbmm_65_ptr : RegionName := "candidate_vbmm_65"
  let candidate_vslice_backward_26_ptr : RegionName := "candidate_vslice_backward_26"
  let candidate_vto_copy_84_ptr : RegionName := "candidate_vto_copy_84"
  let candidate_vslice_backward_27_ptr : RegionName := "candidate_vslice_backward_27"
  let candidate_vto_copy_85_ptr : RegionName := "candidate_vto_copy_85"
  let candidate_vslice_backward_28_ptr : RegionName := "candidate_vslice_backward_28"
  let candidate_vbmm_66_ptr : RegionName := "candidate_vbmm_66"
  let candidate_vbmm_67_ptr : RegionName := "candidate_vbmm_67"
  let candidate_vto_copy_86_ptr : RegionName := "candidate_vto_copy_86"
  let candidate_vsoftmax_backward_data_7_ptr : RegionName := "candidate_vsoftmax_backward_data_7"
  let candidate_vmasked_fill_19_ptr : RegionName := "candidate_vmasked_fill_19"
  let candidate_vmul_42_ptr : RegionName := "candidate_vmul_42"
  let candidate_vbmm_68_ptr : RegionName := "candidate_vbmm_68"
  let candidate_vbmm_69_ptr : RegionName := "candidate_vbmm_69"
  let candidate_vslice_backward_29_ptr : RegionName := "candidate_vslice_backward_29"
  let candidate_vadd_38_ptr : RegionName := "candidate_vadd_38"
  let candidate_vto_copy_87_ptr : RegionName := "candidate_vto_copy_87"
  let candidate_vslice_backward_30_ptr : RegionName := "candidate_vslice_backward_30"
  let candidate_vadd_39_ptr : RegionName := "candidate_vadd_39"
  let candidate_vto_copy_88_ptr : RegionName := "candidate_vto_copy_88"
  let candidate_vslice_backward_31_ptr : RegionName := "candidate_vslice_backward_31"
  let candidate_vadd_40_ptr : RegionName := "candidate_vadd_40"
  let candidate_vbmm_70_ptr : RegionName := "candidate_vbmm_70"
  let candidate_vbmm_71_ptr : RegionName := "candidate_vbmm_71"
  let candidate_vto_copy_89_ptr : RegionName := "candidate_vto_copy_89"
  let candidate_vsoftmax_backward_data_8_ptr : RegionName := "candidate_vsoftmax_backward_data_8"
  let candidate_vmasked_fill_20_ptr : RegionName := "candidate_vmasked_fill_20"
  let candidate_vmul_43_ptr : RegionName := "candidate_vmul_43"
  let candidate_vbmm_72_ptr : RegionName := "candidate_vbmm_72"
  let candidate_vbmm_73_ptr : RegionName := "candidate_vbmm_73"
  let candidate_vslice_backward_32_ptr : RegionName := "candidate_vslice_backward_32"
  let candidate_vadd_41_ptr : RegionName := "candidate_vadd_41"
  let candidate_vto_copy_90_ptr : RegionName := "candidate_vto_copy_90"
  let candidate_vslice_backward_33_ptr : RegionName := "candidate_vslice_backward_33"
  let candidate_vadd_42_ptr : RegionName := "candidate_vadd_42"
  let candidate_vto_copy_91_ptr : RegionName := "candidate_vto_copy_91"
  let candidate_vslice_backward_34_ptr : RegionName := "candidate_vslice_backward_34"
  let candidate_vadd_43_ptr : RegionName := "candidate_vadd_43"
  let candidate_vfill_12_ptr : RegionName := "candidate_vfill_12"
  let candidate_vslice_scatter_12_ptr : RegionName := "candidate_vslice_scatter_12"
  let candidate_vfill_13_ptr : RegionName := "candidate_vfill_13"
  let candidate_vslice_scatter_13_ptr : RegionName := "candidate_vslice_scatter_13"
  let candidate_vfill_14_ptr : RegionName := "candidate_vfill_14"
  let candidate_vslice_scatter_14_ptr : RegionName := "candidate_vslice_scatter_14"
  let candidate_vfill_15_ptr : RegionName := "candidate_vfill_15"
  let candidate_vslice_scatter_15_ptr : RegionName := "candidate_vslice_scatter_15"
  let candidate_vfill_16_ptr : RegionName := "candidate_vfill_16"
  let candidate_vslice_scatter_16_ptr : RegionName := "candidate_vslice_scatter_16"
  let candidate_vfill_17_ptr : RegionName := "candidate_vfill_17"
  let candidate_vslice_scatter_17_ptr : RegionName := "candidate_vslice_scatter_17"
  let candidate_vsum_3_ptr : RegionName := "candidate_vsum_3"
  let candidate_vbmm_74_ptr : RegionName := "candidate_vbmm_74"
  let candidate_vbmm_75_ptr : RegionName := "candidate_vbmm_75"
  let candidate_vadd_44_ptr : RegionName := "candidate_vadd_44"
  let candidate_vselect_backward_2_ptr : RegionName := "candidate_vselect_backward_2"
  let candidate_vslice_backward_35_ptr : RegionName := "candidate_vslice_backward_35"
  let candidate_vadd_45_ptr : RegionName := "candidate_vadd_45"
  let candidate_vslice_backward_36_ptr : RegionName := "candidate_vslice_backward_36"
  let candidate_vadd_46_ptr : RegionName := "candidate_vadd_46"
  let candidate_vslice_backward_37_ptr : RegionName := "candidate_vslice_backward_37"
  let candidate_vadd_47_ptr : RegionName := "candidate_vadd_47"
  let candidate_vslice_backward_38_ptr : RegionName := "candidate_vslice_backward_38"
  let candidate_vadd_48_ptr : RegionName := "candidate_vadd_48"
  let candidate_vbmm_76_ptr : RegionName := "candidate_vbmm_76"
  let candidate_vbmm_77_ptr : RegionName := "candidate_vbmm_77"
  let candidate_vto_copy_92_ptr : RegionName := "candidate_vto_copy_92"
  let candidate_vsoftmax_backward_data_9_ptr : RegionName := "candidate_vsoftmax_backward_data_9"
  let candidate_vmasked_fill_21_ptr : RegionName := "candidate_vmasked_fill_21"
  let candidate_vmul_44_ptr : RegionName := "candidate_vmul_44"
  let candidate_vbmm_78_ptr : RegionName := "candidate_vbmm_78"
  let candidate_vbmm_79_ptr : RegionName := "candidate_vbmm_79"
  let candidate_vslice_backward_39_ptr : RegionName := "candidate_vslice_backward_39"
  let candidate_vto_copy_93_ptr : RegionName := "candidate_vto_copy_93"
  let candidate_vslice_backward_40_ptr : RegionName := "candidate_vslice_backward_40"
  let candidate_vto_copy_94_ptr : RegionName := "candidate_vto_copy_94"
  let candidate_vslice_backward_41_ptr : RegionName := "candidate_vslice_backward_41"
  let candidate_vbmm_80_ptr : RegionName := "candidate_vbmm_80"
  let candidate_vbmm_81_ptr : RegionName := "candidate_vbmm_81"
  let candidate_vto_copy_95_ptr : RegionName := "candidate_vto_copy_95"
  let candidate_vsoftmax_backward_data_10_ptr : RegionName := "candidate_vsoftmax_backward_data_10"
  let candidate_vmasked_fill_22_ptr : RegionName := "candidate_vmasked_fill_22"
  let candidate_vmul_45_ptr : RegionName := "candidate_vmul_45"
  let candidate_vbmm_82_ptr : RegionName := "candidate_vbmm_82"
  let candidate_vbmm_83_ptr : RegionName := "candidate_vbmm_83"
  let candidate_vslice_backward_42_ptr : RegionName := "candidate_vslice_backward_42"
  let candidate_vadd_49_ptr : RegionName := "candidate_vadd_49"
  let candidate_vto_copy_96_ptr : RegionName := "candidate_vto_copy_96"
  let candidate_vslice_backward_43_ptr : RegionName := "candidate_vslice_backward_43"
  let candidate_vadd_50_ptr : RegionName := "candidate_vadd_50"
  let candidate_vto_copy_97_ptr : RegionName := "candidate_vto_copy_97"
  let candidate_vslice_backward_44_ptr : RegionName := "candidate_vslice_backward_44"
  let candidate_vadd_51_ptr : RegionName := "candidate_vadd_51"
  let candidate_vbmm_84_ptr : RegionName := "candidate_vbmm_84"
  let candidate_vbmm_85_ptr : RegionName := "candidate_vbmm_85"
  let candidate_vto_copy_98_ptr : RegionName := "candidate_vto_copy_98"
  let candidate_vsoftmax_backward_data_11_ptr : RegionName := "candidate_vsoftmax_backward_data_11"
  let candidate_vmasked_fill_23_ptr : RegionName := "candidate_vmasked_fill_23"
  let candidate_vmul_46_ptr : RegionName := "candidate_vmul_46"
  let candidate_vbmm_86_ptr : RegionName := "candidate_vbmm_86"
  let candidate_vbmm_87_ptr : RegionName := "candidate_vbmm_87"
  let candidate_vslice_backward_45_ptr : RegionName := "candidate_vslice_backward_45"
  let candidate_vadd_52_ptr : RegionName := "candidate_vadd_52"
  let candidate_vto_copy_99_ptr : RegionName := "candidate_vto_copy_99"
  let candidate_vslice_backward_46_ptr : RegionName := "candidate_vslice_backward_46"
  let candidate_vadd_53_ptr : RegionName := "candidate_vadd_53"
  let candidate_vto_copy_100_ptr : RegionName := "candidate_vto_copy_100"
  let candidate_vslice_backward_47_ptr : RegionName := "candidate_vslice_backward_47"
  let candidate_vadd_54_ptr : RegionName := "candidate_vadd_54"
  let candidate_vfill_18_ptr : RegionName := "candidate_vfill_18"
  let candidate_vslice_scatter_18_ptr : RegionName := "candidate_vslice_scatter_18"
  let candidate_vfill_19_ptr : RegionName := "candidate_vfill_19"
  let candidate_vslice_scatter_19_ptr : RegionName := "candidate_vslice_scatter_19"
  let candidate_vfill_20_ptr : RegionName := "candidate_vfill_20"
  let candidate_vslice_scatter_20_ptr : RegionName := "candidate_vslice_scatter_20"
  let candidate_vfill_21_ptr : RegionName := "candidate_vfill_21"
  let candidate_vslice_scatter_21_ptr : RegionName := "candidate_vslice_scatter_21"
  let candidate_vfill_22_ptr : RegionName := "candidate_vfill_22"
  let candidate_vslice_scatter_22_ptr : RegionName := "candidate_vslice_scatter_22"
  let candidate_vfill_23_ptr : RegionName := "candidate_vfill_23"
  let candidate_vslice_scatter_23_ptr : RegionName := "candidate_vslice_scatter_23"
  let candidate_vsum_4_ptr : RegionName := "candidate_vsum_4"
  let candidate_vbmm_88_ptr : RegionName := "candidate_vbmm_88"
  let candidate_vbmm_89_ptr : RegionName := "candidate_vbmm_89"
  let candidate_vadd_55_ptr : RegionName := "candidate_vadd_55"
  let candidate_vselect_backward_3_ptr : RegionName := "candidate_vselect_backward_3"
  let candidate_vslice_backward_48_ptr : RegionName := "candidate_vslice_backward_48"
  let candidate_vadd_56_ptr : RegionName := "candidate_vadd_56"
  let candidate_vslice_backward_49_ptr : RegionName := "candidate_vslice_backward_49"
  let candidate_vadd_57_ptr : RegionName := "candidate_vadd_57"
  let candidate_vslice_backward_50_ptr : RegionName := "candidate_vslice_backward_50"
  let candidate_vadd_58_ptr : RegionName := "candidate_vadd_58"
  let candidate_vslice_backward_51_ptr : RegionName := "candidate_vslice_backward_51"
  let candidate_vadd_59_ptr : RegionName := "candidate_vadd_59"
  let candidate_vcat_19_ptr : RegionName := "candidate_vcat_19"
  let candidate_vindex_put_1_ptr : RegionName := "candidate_vindex_put_1"
  let candidate_vmul_47_ptr : RegionName := "candidate_vmul_47"
  let candidate_vmul_48_ptr : RegionName := "candidate_vmul_48"
  let candidate_vselect_backward_4_ptr : RegionName := "candidate_vselect_backward_4"
  let candidate_vneg_2_ptr : RegionName := "candidate_vneg_2"
  let candidate_vselect_backward_5_ptr : RegionName := "candidate_vselect_backward_5"
  let candidate_vadd_60_ptr : RegionName := "candidate_vadd_60"
  let candidate_vadd_61_ptr : RegionName := "candidate_vadd_61"
  let candidate_vmul_49_ptr : RegionName := "candidate_vmul_49"
  let candidate_vmul_50_ptr : RegionName := "candidate_vmul_50"
  let candidate_vselect_backward_6_ptr : RegionName := "candidate_vselect_backward_6"
  let candidate_vneg_3_ptr : RegionName := "candidate_vneg_3"
  let candidate_vselect_backward_7_ptr : RegionName := "candidate_vselect_backward_7"
  let candidate_vadd_62_ptr : RegionName := "candidate_vadd_62"
  let candidate_vadd_63_ptr : RegionName := "candidate_vadd_63"
  let candidate_vcat_20_ptr : RegionName := "candidate_vcat_20"
  let candidate_vto_copy_101_ptr : RegionName := "candidate_vto_copy_101"
  let candidate_vmul_51_ptr : RegionName := "candidate_vmul_51"
  let candidate_vmul_52_ptr : RegionName := "candidate_vmul_52"
  let candidate_vmul_53_ptr : RegionName := "candidate_vmul_53"
  let candidate_vsum_5_ptr : RegionName := "candidate_vsum_5"
  let candidate_vpow_6_ptr : RegionName := "candidate_vpow_6"
  let candidate_vmul_54_ptr : RegionName := "candidate_vmul_54"
  let candidate_vmul_55_ptr : RegionName := "candidate_vmul_55"
  let candidate_vdiv_3_ptr : RegionName := "candidate_vdiv_3"
  let candidate_vpow_7_ptr : RegionName := "candidate_vpow_7"
  let candidate_vmul_56_ptr : RegionName := "candidate_vmul_56"
  let candidate_vmul_57_ptr : RegionName := "candidate_vmul_57"
  let candidate_vadd_64_ptr : RegionName := "candidate_vadd_64"
  let candidate_vto_copy_102_ptr : RegionName := "candidate_vto_copy_102"
  let candidate_vcat_21_ptr : RegionName := "candidate_vcat_21"
  let candidate_vmm_5_ptr : RegionName := "candidate_vmm_5"
  let candidate_vmm_6_ptr : RegionName := "candidate_vmm_6"
  let candidate_vto_copy_103_ptr : RegionName := "candidate_vto_copy_103"
  let candidate_vmul_58_ptr : RegionName := "candidate_vmul_58"
  let candidate_vmul_59_ptr : RegionName := "candidate_vmul_59"
  let candidate_vmul_60_ptr : RegionName := "candidate_vmul_60"
  let candidate_vsum_6_ptr : RegionName := "candidate_vsum_6"
  let candidate_vpow_8_ptr : RegionName := "candidate_vpow_8"
  let candidate_vmul_61_ptr : RegionName := "candidate_vmul_61"
  let candidate_vmul_62_ptr : RegionName := "candidate_vmul_62"
  let candidate_vdiv_4_ptr : RegionName := "candidate_vdiv_4"
  let candidate_vpow_9_ptr : RegionName := "candidate_vpow_9"
  let candidate_vmul_63_ptr : RegionName := "candidate_vmul_63"
  let candidate_vmul_64_ptr : RegionName := "candidate_vmul_64"
  let candidate_vadd_65_ptr : RegionName := "candidate_vadd_65"
  let candidate_vto_copy_104_ptr : RegionName := "candidate_vto_copy_104"
  let candidate_vmm_7_ptr : RegionName := "candidate_vmm_7"
  let candidate_vadd_66_ptr : RegionName := "candidate_vadd_66"
  let candidate_vindex_put_2_ptr : RegionName := "candidate_vindex_put_2"
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
    r16_z = tl.arange($(128))
    r16_value = tl.where(r16_z < $(48), tl.cast((r16_z - $(0)), tl.int64), tl.where(r16_z < $(64), tl.cast((r16_z - $(48)), tl.int64), tl.cast((r16_z - $(64)), tl.int64)))
    r16_value = tl.broadcast_to(r16_value, [$(128)])
    tl.store($(candidate_vcat_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(128))
    r17_value = (tl.load($(candidate_vcat_ptr) + r17_z, dtype=tl.int64) + tl.full([], $(2), dtype=tl.int64))
    r17_value = tl.broadcast_to(r17_value, [$(128)])
    tl.store($(candidate_vadd_2_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(128))
    r18_lane = tl.arange($(128))
    r18_row_values = tl.load($(constantee33028f0f_ptr) + r18_lane[None, :], dtype=tl.int64)
    r18_rank = tl.full([$(128), $(128)], $(0), dtype=tl.int64)
    for r18_other in range($(128)) {
      r18_rank = r18_rank + tl.where(((tl.load($(constantee33028f0f_ptr) + (r18_other % $(128)), dtype=tl.int64) < r18_row_values) | ((tl.load($(constantee33028f0f_ptr) + (r18_other % $(128)), dtype=tl.int64) == r18_row_values) & (r18_other < r18_lane[None, :]))), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r18_selected = r18_rank == tl.cast(r18_z[:, None], tl.int64)
    r18_value = tl.full([$(128)], $(0), dtype=tl.int64)
    for r18_cutoff in range($(128)) {
      r18_value = r18_value + tl.cast(tl.sum(r18_selected & (r18_lane[None, :] > r18_cutoff), axis=1), tl.int64)
    }
    r18_value = tl.load($(constantee33028f0f_ptr) + tl.cast(r18_value, tl.uint32), dtype=tl.int64)
    r18_value = tl.broadcast_to(r18_value, [$(128)])
    tl.store($(candidate_vsort_value_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(128))
    r19_lane = tl.arange($(128))
    r19_row_values = tl.load($(constantee33028f0f_ptr) + r19_lane[None, :], dtype=tl.int64)
    r19_rank = tl.full([$(128), $(128)], $(0), dtype=tl.int64)
    for r19_other in range($(128)) {
      r19_rank = r19_rank + tl.where(((tl.load($(constantee33028f0f_ptr) + (r19_other % $(128)), dtype=tl.int64) < r19_row_values) | ((tl.load($(constantee33028f0f_ptr) + (r19_other % $(128)), dtype=tl.int64) == r19_row_values) & (r19_other < r19_lane[None, :]))), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r19_selected = r19_rank == tl.cast(r19_z[:, None], tl.int64)
    r19_value = tl.full([$(128)], $(0), dtype=tl.int64)
    for r19_cutoff in range($(128)) {
      r19_value = r19_value + tl.cast(tl.sum(r19_selected & (r19_lane[None, :] > r19_cutoff), axis=1), tl.int64)
    }
    r19_value = tl.broadcast_to(r19_value, [$(128)])
    tl.store($(candidate_vsort_index_ptr) + r19_z, r19_value)
    r20_z = tl.arange($(128))
    r20_value = tl.load($(candidate_vadd_2_ptr) + (tl.load($(constantee33028f0f_ptr) + r20_z, dtype=tl.int64) % $(128)), dtype=tl.int64)
    r20_value = tl.broadcast_to(r20_value, [$(128)])
    tl.store($(candidate_vindex_ptr) + r20_z, r20_value)
    r21_z = tl.arange($(128))
    r21_value = (tl.load($(candidate_vindex_ptr) + r21_z, dtype=tl.int64) - tl.full([], $(2), dtype=tl.int64))
    r21_value = tl.broadcast_to(r21_value, [$(128)])
    tl.store($(candidate_vsub_8_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(65536))
    r22_value = tl.load($(input0_ptr) + (((tl.load($(constantee33028f0f_ptr) + ((r22_z / $(512)) % $(128)), dtype=tl.int64) % $(128)) * $(512)) + (r22_z % $(512))), dtype=tl.bfloat16)
    r22_value = r22_value.to(tl.bfloat16)
    r22_value = tl.broadcast_to(r22_value, [$(65536)])
    tl.store($(candidate_vindex_1_ptr) + r22_z, r22_value)
    r23_row = tl.arange($(128))
    r23_col = tl.arange($(256))
    r23_lane = tl.arange($(512))
    r23_left = tl.load($(candidate_vindex_1_ptr) + ((((((r23_row[:, None] * $(512)) + r23_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r23_row[:, None] * $(512)) + r23_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r23_right = tl.load($(candidate_vto_copy_ptr) + (((r23_col[None, :] % $(256)) * $(512)) + (r23_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r23_value = tl.dot(r23_left, r23_right)
    r23_value = r23_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_ptr) + ((r23_row[:, None] * $(256)) + r23_col[None, :]), r23_value)
    r24_z = tl.arange($(32768))
    r24_value = tl.load($(candidate_vmm_ptr) + ((((r24_z / $(256)) % $(128)) * $(256)) + (r24_z % $(256))), dtype=tl.bfloat16)
    r24_value = r24_value.to(tl.float32)
    r24_value = tl.broadcast_to(r24_value, [$(32768)])
    tl.store($(candidate_vto_copy_20_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(32768))
    r25_value = ((tl.load($(candidate_vto_copy_20_ptr) + ((((r25_z / $(256)) % $(128)) * $(256)) + (r25_z % $(256))), dtype=tl.float32)) * (tl.load($(candidate_vto_copy_20_ptr) + ((((r25_z / $(256)) % $(128)) * $(256)) + (r25_z % $(256))), dtype=tl.float32)))
    r25_value = r25_value.to(tl.float32)
    r25_value = tl.broadcast_to(r25_value, [$(32768)])
    tl.store($(candidate_vpow_3_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(128))
    r26_lane = tl.arange($(256))
    r26_value = (tl.sum(tl.load($(candidate_vpow_3_ptr) + ((r26_z[:, None] * $(256)) + r26_lane[None, :]), dtype=tl.float32), axis=1) / 256.0)
    r26_value = r26_value.to(tl.float32)
    r26_value = tl.broadcast_to(r26_value, [$(128)])
    tl.store($(candidate_vmean_ptr) + r26_z, r26_value)
    r27_z = tl.arange($(128))
    r27_value = (tl.load($(candidate_vmean_ptr) + r27_z, dtype=tl.float32) + 0.000001)
    r27_value = r27_value.to(tl.float32)
    r27_value = tl.broadcast_to(r27_value, [$(128)])
    tl.store($(candidate_vadd_3_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(128))
    r28_value = tl.rsqrt(tl.load($(candidate_vadd_3_ptr) + r28_z, dtype=tl.float32))
    r28_value = r28_value.to(tl.float32)
    r28_value = tl.broadcast_to(r28_value, [$(128)])
    tl.store($(candidate_vrsqrt_ptr) + r28_z, r28_value)
    r29_z = tl.arange($(32768))
    r29_value = (tl.load($(candidate_vto_copy_20_ptr) + ((((r29_z / $(256)) % $(128)) * $(256)) + (r29_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_ptr) + ((r29_z / $(256)) % $(128)), dtype=tl.float32))
    r29_value = r29_value.to(tl.float32)
    r29_value = tl.broadcast_to(r29_value, [$(32768)])
    tl.store($(candidate_vmul_13_ptr) + r29_z, r29_value)
    r30_z = tl.arange($(32768))
    r30_value = (tl.cast(tl.load($(candidate_vto_copy_1_ptr) + (r30_z % $(256)), dtype=tl.bfloat16), tl.float32) * tl.load($(candidate_vmul_13_ptr) + ((((r30_z / $(256)) % $(128)) * $(256)) + (r30_z % $(256))), dtype=tl.float32))
    r30_value = r30_value.to(tl.float32)
    r30_value = tl.broadcast_to(r30_value, [$(32768)])
    tl.store($(candidate_vmul_14_ptr) + r30_z, r30_value)
    r31_z = tl.arange($(32768))
    r31_value = tl.load($(candidate_vmul_14_ptr) + ((((r31_z / $(256)) % $(128)) * $(256)) + (r31_z % $(256))), dtype=tl.float32)
    r31_value = r31_value.to(tl.bfloat16)
    r31_value = tl.broadcast_to(r31_value, [$(32768)])
    tl.store($(candidate_vto_copy_21_ptr) + r31_z, r31_value)
    r32_row = tl.arange($(128))
    r32_col = tl.arange($(1536))
    r32_lane = tl.arange($(256))
    r32_left = tl.load($(candidate_vto_copy_21_ptr) + ((((((r32_row[:, None] * $(256)) + r32_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r32_row[:, None] * $(256)) + r32_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r32_right = tl.load($(candidate_vto_copy_2_ptr) + (((r32_col[None, :] % $(1536)) * $(256)) + (r32_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r32_value = tl.dot(r32_left, r32_right)
    r32_value = r32_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_1_ptr) + ((r32_row[:, None] * $(1536)) + r32_col[None, :]), r32_value)
    r33_row = tl.arange($(128))
    r33_col = tl.arange($(320))
    r33_lane = tl.arange($(512))
    r33_left = tl.load($(candidate_vindex_1_ptr) + ((((((r33_row[:, None] * $(512)) + r33_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r33_row[:, None] * $(512)) + r33_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r33_right = tl.load($(candidate_vto_copy_3_ptr) + (((r33_col[None, :] % $(320)) * $(512)) + (r33_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r33_value = tl.dot(r33_left, r33_right)
    r33_value = r33_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_2_ptr) + ((r33_row[:, None] * $(320)) + r33_col[None, :]), r33_value)
    r34_z = tl.arange($(32768))
    r34_value = tl.load($(candidate_vmm_2_ptr) + ((((r34_z / $(256)) % $(128)) * $(320)) + (r34_z % $(256))), dtype=tl.bfloat16)
    r34_value = r34_value.to(tl.float32)
    r34_value = tl.broadcast_to(r34_value, [$(32768)])
    tl.store($(candidate_vto_copy_22_ptr) + r34_z, r34_value)
    r35_z = tl.arange($(32768))
    r35_value = ((tl.load($(candidate_vto_copy_22_ptr) + ((((r35_z / $(256)) % $(128)) * $(256)) + (r35_z % $(256))), dtype=tl.float32)) * (tl.load($(candidate_vto_copy_22_ptr) + ((((r35_z / $(256)) % $(128)) * $(256)) + (r35_z % $(256))), dtype=tl.float32)))
    r35_value = r35_value.to(tl.float32)
    r35_value = tl.broadcast_to(r35_value, [$(32768)])
    tl.store($(candidate_vpow_4_ptr) + r35_z, r35_value)
    r36_z = tl.arange($(128))
    r36_lane = tl.arange($(256))
    r36_value = (tl.sum(tl.load($(candidate_vpow_4_ptr) + ((r36_z[:, None] * $(256)) + r36_lane[None, :]), dtype=tl.float32), axis=1) / 256.0)
    r36_value = r36_value.to(tl.float32)
    r36_value = tl.broadcast_to(r36_value, [$(128)])
    tl.store($(candidate_vmean_1_ptr) + r36_z, r36_value)
    r37_z = tl.arange($(128))
    r37_value = (tl.load($(candidate_vmean_1_ptr) + r37_z, dtype=tl.float32) + 0.000001)
    r37_value = r37_value.to(tl.float32)
    r37_value = tl.broadcast_to(r37_value, [$(128)])
    tl.store($(candidate_vadd_4_ptr) + r37_z, r37_value)
    r38_z = tl.arange($(128))
    r38_value = tl.rsqrt(tl.load($(candidate_vadd_4_ptr) + r38_z, dtype=tl.float32))
    r38_value = r38_value.to(tl.float32)
    r38_value = tl.broadcast_to(r38_value, [$(128)])
    tl.store($(candidate_vrsqrt_1_ptr) + r38_z, r38_value)
    r39_z = tl.arange($(32768))
    r39_value = (tl.load($(candidate_vto_copy_22_ptr) + ((((r39_z / $(256)) % $(128)) * $(256)) + (r39_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_1_ptr) + ((r39_z / $(256)) % $(128)), dtype=tl.float32))
    r39_value = r39_value.to(tl.float32)
    r39_value = tl.broadcast_to(r39_value, [$(32768)])
    tl.store($(candidate_vmul_15_ptr) + r39_z, r39_value)
    r40_z = tl.arange($(32768))
    r40_value = (tl.cast(tl.load($(candidate_vto_copy_4_ptr) + (r40_z % $(256)), dtype=tl.bfloat16), tl.float32) * tl.load($(candidate_vmul_15_ptr) + ((((r40_z / $(256)) % $(128)) * $(256)) + (r40_z % $(256))), dtype=tl.float32))
    r40_value = r40_value.to(tl.float32)
    r40_value = tl.broadcast_to(r40_value, [$(32768)])
    tl.store($(candidate_vmul_16_ptr) + r40_z, r40_value)
    r41_z = tl.arange($(32768))
    r41_value = tl.load($(candidate_vmul_16_ptr) + ((((r41_z / $(256)) % $(128)) * $(256)) + (r41_z % $(256))), dtype=tl.float32)
    r41_value = r41_value.to(tl.bfloat16)
    r41_value = tl.broadcast_to(r41_value, [$(32768)])
    tl.store($(candidate_vto_copy_23_ptr) + r41_z, r41_value)
    r42_z = tl.arange($(32))
    r42_value = ((tl.cast(tl.toReal(r42_z), tl.float32) * 2.0) / 64.0)
    r42_value = r42_value.to(tl.float32)
    r42_value = tl.broadcast_to(r42_value, [$(32)])
    tl.store($(candidate_vdiv_2_ptr) + r42_z, r42_value)
    r43_z = tl.arange($(32))
    r43_value = tl.extra.cuda.libdevice.pow(10000.0, tl.load($(candidate_vdiv_2_ptr) + r43_z, dtype=tl.float32))
    r43_value = r43_value.to(tl.float32)
    r43_value = tl.broadcast_to(r43_value, [$(32)])
    tl.store($(candidate_vpow_5_ptr) + r43_z, r43_value)
    r44_z = tl.arange($(32))
    r44_value = 1.0 / tl.load($(candidate_vpow_5_ptr) + r44_z, dtype=tl.float32)
    r44_value = r44_value.to(tl.float32)
    r44_value = tl.broadcast_to(r44_value, [$(32)])
    tl.store($(candidate_vreciprocal_2_ptr) + r44_z, r44_value)
    r45_z = tl.arange($(32))
    r45_value = (tl.load($(candidate_vreciprocal_2_ptr) + r45_z, dtype=tl.float32) * 1.0)
    r45_value = r45_value.to(tl.float32)
    r45_value = tl.broadcast_to(r45_value, [$(32)])
    tl.store($(candidate_vmul_17_ptr) + r45_z, r45_value)
    r46_z = tl.arange($(128))
    r46_value = tl.toReal(tl.load($(candidate_vsub_8_ptr) + r46_z, dtype=tl.int64))
    r46_value = r46_value.to(tl.float32)
    r46_value = tl.broadcast_to(r46_value, [$(128)])
    tl.store($(candidate_vto_copy_24_ptr) + r46_z, r46_value)
    r47_z = tl.arange($(128))
    r47_value = (tl.load($(candidate_vto_copy_24_ptr) + r47_z, dtype=tl.float32) + 2.0)
    r47_value = r47_value.to(tl.float32)
    r47_value = tl.broadcast_to(r47_value, [$(128)])
    tl.store($(candidate_vadd_5_ptr) + r47_z, r47_value)
    r48_z = tl.arange($(4096))
    r48_value = (tl.load($(candidate_vadd_5_ptr) + ((r48_z / $(32)) % $(128)), dtype=tl.float32) * tl.load($(candidate_vmul_17_ptr) + (r48_z % $(32)), dtype=tl.float32))
    r48_value = r48_value.to(tl.float32)
    r48_value = tl.broadcast_to(r48_value, [$(4096)])
    tl.store($(candidate_vmul_18_ptr) + r48_z, r48_value)
    r49_z = tl.arange($(8192))
    r49_value = tl.cos(tl.load($(candidate_vmul_18_ptr) + ((((r49_z / $(64)) % $(128)) * $(32)) + ((((((r49_z / $(64)) % $(128)) * $(64)) + (r49_z % $(64))) / $(2)) % $(32))), dtype=tl.float32))
    r49_value = r49_value.to(tl.float32)
    r49_value = tl.broadcast_to(r49_value, [$(8192)])
    tl.store($(candidate_vcos_2_ptr) + r49_z, r49_value)
    r50_z = tl.arange($(8192))
    r50_value = tl.load($(candidate_vcos_2_ptr) + ((((r50_z / $(64)) % $(128)) * $(64)) + (r50_z % $(64))), dtype=tl.float32)
    r50_value = r50_value.to(tl.bfloat16)
    r50_value = tl.broadcast_to(r50_value, [$(8192)])
    tl.store($(candidate_vto_copy_25_ptr) + r50_z, r50_value)
    r51_z = tl.arange($(8192))
    r51_value = tl.sin(tl.load($(candidate_vmul_18_ptr) + ((((r51_z / $(64)) % $(128)) * $(32)) + ((((((r51_z / $(64)) % $(128)) * $(64)) + (r51_z % $(64))) / $(2)) % $(32))), dtype=tl.float32))
    r51_value = r51_value.to(tl.float32)
    r51_value = tl.broadcast_to(r51_value, [$(8192)])
    tl.store($(candidate_vsin_2_ptr) + r51_z, r51_value)
    r52_z = tl.arange($(8192))
    r52_value = tl.load($(candidate_vsin_2_ptr) + ((((r52_z / $(64)) % $(128)) * $(64)) + (r52_z % $(64))), dtype=tl.float32)
    r52_value = r52_value.to(tl.bfloat16)
    r52_value = tl.broadcast_to(r52_value, [$(8192)])
    tl.store($(candidate_vto_copy_26_ptr) + r52_z, r52_value)
    r53_z = tl.arange($(32768))
    r53_value = -tl.load($(candidate_vmm_1_ptr) + ((((r53_z / $(256)) % $(128)) * $(1536)) + (((r53_z / $(32)) % $(8)) * $(192)) + ((r53_z % $(32)) * $(2)) + $(129)), dtype=tl.bfloat16)
    r53_value = r53_value.to(tl.bfloat16)
    r53_value = tl.broadcast_to(r53_value, [$(32768)])
    tl.store($(candidate_vneg_ptr) + r53_z, r53_value)
    r54_z = tl.arange($(65536))
    r54_value = tl.where((r54_z % $(2)) < $(1), tl.load($(candidate_vneg_ptr) + ((((r54_z / $(512)) % $(128)) * $(256)) + (((r54_z / $(64)) % $(8)) * $(32)) + ((r54_z / $(2)) % $(32))), dtype=tl.bfloat16), tl.load($(candidate_vmm_1_ptr) + ((((r54_z / $(512)) % $(128)) * $(1536)) + (((r54_z / $(64)) % $(8)) * $(192)) + (((r54_z / $(2)) % $(32)) * $(2)) + $(128)), dtype=tl.bfloat16))
    r54_value = r54_value.to(tl.bfloat16)
    r54_value = tl.broadcast_to(r54_value, [$(65536)])
    tl.store($(candidate_vstack_ptr) + r54_z, r54_value)
    r55_z = tl.arange($(65536))
    r55_value = (tl.load($(candidate_vmm_1_ptr) + ((((r55_z / $(512)) % $(128)) * $(1536)) + (((r55_z / $(64)) % $(8)) * $(192)) + (r55_z % $(64)) + $(128)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_25_ptr) + ((((r55_z / $(512)) % $(128)) * $(64)) + (r55_z % $(64))), dtype=tl.bfloat16))
    r55_value = r55_value.to(tl.bfloat16)
    r55_value = tl.broadcast_to(r55_value, [$(65536)])
    tl.store($(candidate_vmul_19_ptr) + r55_z, r55_value)
    r56_z = tl.arange($(65536))
    r56_value = (tl.load($(candidate_vstack_ptr) + ((((r56_z / $(512)) % $(128)) * $(512)) + (((r56_z / $(64)) % $(8)) * $(64)) + (((((((r56_z / $(512)) % $(128)) * $(512)) + (((r56_z / $(64)) % $(8)) * $(64)) + (r56_z % $(64))) / $(2)) % $(32)) * $(2)) + (((((r56_z / $(512)) % $(128)) * $(512)) + (((r56_z / $(64)) % $(8)) * $(64)) + (r56_z % $(64))) % $(2))), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_26_ptr) + ((((r56_z / $(512)) % $(128)) * $(64)) + (r56_z % $(64))), dtype=tl.bfloat16))
    r56_value = r56_value.to(tl.bfloat16)
    r56_value = tl.broadcast_to(r56_value, [$(65536)])
    tl.store($(candidate_vmul_20_ptr) + r56_z, r56_value)
    r57_z = tl.arange($(65536))
    r57_value = (tl.load($(candidate_vmul_19_ptr) + ((((r57_z / $(512)) % $(128)) * $(512)) + (((r57_z / $(64)) % $(8)) * $(64)) + (r57_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_20_ptr) + ((((r57_z / $(512)) % $(128)) * $(512)) + (((r57_z / $(64)) % $(8)) * $(64)) + (r57_z % $(64))), dtype=tl.bfloat16))
    r57_value = r57_value.to(tl.bfloat16)
    r57_value = tl.broadcast_to(r57_value, [$(65536)])
    tl.store($(candidate_vadd_6_ptr) + r57_z, r57_value)
    r58_z = tl.arange($(8192))
    r58_value = tl.cos(tl.load($(candidate_vmul_18_ptr) + ((((r58_z / $(64)) % $(128)) * $(32)) + ((((((r58_z / $(64)) % $(128)) * $(64)) + (r58_z % $(64))) / $(2)) % $(32))), dtype=tl.float32))
    r58_value = r58_value.to(tl.float32)
    r58_value = tl.broadcast_to(r58_value, [$(8192)])
    tl.store($(candidate_vcos_3_ptr) + r58_z, r58_value)
    r59_z = tl.arange($(8192))
    r59_value = tl.load($(candidate_vcos_3_ptr) + ((((r59_z / $(64)) % $(128)) * $(64)) + (r59_z % $(64))), dtype=tl.float32)
    r59_value = r59_value.to(tl.bfloat16)
    r59_value = tl.broadcast_to(r59_value, [$(8192)])
    tl.store($(candidate_vto_copy_27_ptr) + r59_z, r59_value)
    r60_z = tl.arange($(8192))
    r60_value = tl.sin(tl.load($(candidate_vmul_18_ptr) + ((((r60_z / $(64)) % $(128)) * $(32)) + ((((((r60_z / $(64)) % $(128)) * $(64)) + (r60_z % $(64))) / $(2)) % $(32))), dtype=tl.float32))
    r60_value = r60_value.to(tl.float32)
    r60_value = tl.broadcast_to(r60_value, [$(8192)])
    tl.store($(candidate_vsin_3_ptr) + r60_z, r60_value)
    r61_z = tl.arange($(8192))
    r61_value = tl.load($(candidate_vsin_3_ptr) + ((((r61_z / $(64)) % $(128)) * $(64)) + (r61_z % $(64))), dtype=tl.float32)
    r61_value = r61_value.to(tl.bfloat16)
    r61_value = tl.broadcast_to(r61_value, [$(8192)])
    tl.store($(candidate_vto_copy_28_ptr) + r61_z, r61_value)
    r62_z = tl.arange($(4096))
    r62_value = -tl.load($(candidate_vmm_2_ptr) + ((((r62_z / $(32)) % $(128)) * $(320)) + ((r62_z % $(32)) * $(2)) + $(257)), dtype=tl.bfloat16)
    r62_value = r62_value.to(tl.bfloat16)
    r62_value = tl.broadcast_to(r62_value, [$(4096)])
    tl.store($(candidate_vneg_1_ptr) + r62_z, r62_value)
    r63_z = tl.arange($(8192))
    r63_value = tl.where((r63_z % $(2)) < $(1), tl.load($(candidate_vneg_1_ptr) + ((((r63_z / $(64)) % $(128)) * $(32)) + ((r63_z / $(2)) % $(32))), dtype=tl.bfloat16), tl.load($(candidate_vmm_2_ptr) + ((((r63_z / $(64)) % $(128)) * $(320)) + (((r63_z / $(2)) % $(32)) * $(2)) + $(256)), dtype=tl.bfloat16))
    r63_value = r63_value.to(tl.bfloat16)
    r63_value = tl.broadcast_to(r63_value, [$(8192)])
    tl.store($(candidate_vstack_1_ptr) + r63_z, r63_value)
    r64_z = tl.arange($(8192))
    r64_value = (tl.load($(candidate_vmm_2_ptr) + ((((r64_z / $(64)) % $(128)) * $(320)) + (r64_z % $(64)) + $(256)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_27_ptr) + ((((r64_z / $(64)) % $(128)) * $(64)) + (r64_z % $(64))), dtype=tl.bfloat16))
    r64_value = r64_value.to(tl.bfloat16)
    r64_value = tl.broadcast_to(r64_value, [$(8192)])
    tl.store($(candidate_vmul_21_ptr) + r64_z, r64_value)
    r65_z = tl.arange($(8192))
    r65_value = (tl.load($(candidate_vstack_1_ptr) + ((((r65_z / $(64)) % $(128)) * $(64)) + (((((((r65_z / $(64)) % $(128)) * $(64)) + (r65_z % $(64))) / $(2)) % $(32)) * $(2)) + (((((r65_z / $(64)) % $(128)) * $(64)) + (r65_z % $(64))) % $(2))), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_28_ptr) + ((((r65_z / $(64)) % $(128)) * $(64)) + (r65_z % $(64))), dtype=tl.bfloat16))
    r65_value = r65_value.to(tl.bfloat16)
    r65_value = tl.broadcast_to(r65_value, [$(8192)])
    tl.store($(candidate_vmul_22_ptr) + r65_z, r65_value)
    r66_z = tl.arange($(8192))
    r66_value = (tl.load($(candidate_vmul_21_ptr) + ((((r66_z / $(64)) % $(128)) * $(64)) + (r66_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_22_ptr) + ((((r66_z / $(64)) % $(128)) * $(64)) + (r66_z % $(64))), dtype=tl.bfloat16))
    r66_value = r66_value.to(tl.bfloat16)
    r66_value = tl.broadcast_to(r66_value, [$(8192)])
    tl.store($(candidate_vadd_7_ptr) + r66_z, r66_value)
    r67_z = tl.arange($(40960))
    r67_value = tl.where((r67_z % $(320)) < $(256), tl.load($(candidate_vto_copy_23_ptr) + ((((((((r67_z / $(320)) % $(128)) * $(256)) + ((r67_z % $(320)) - $(0))) / $(256)) % $(128)) * $(256)) + (((((r67_z / $(320)) % $(128)) * $(256)) + ((r67_z % $(320)) - $(0))) % $(256))), dtype=tl.bfloat16), tl.load($(candidate_vadd_7_ptr) + ((((((((r67_z / $(320)) % $(128)) * $(64)) + ((r67_z % $(320)) - $(256))) / $(64)) % $(128)) * $(64)) + (((((r67_z / $(320)) % $(128)) * $(64)) + ((r67_z % $(320)) - $(256))) % $(64))), dtype=tl.bfloat16))
    r67_value = r67_value.to(tl.bfloat16)
    r67_value = tl.broadcast_to(r67_value, [$(40960)])
    tl.store($(candidate_vcat_5_ptr) + r67_z, r67_value)
    r68_z = tl.arange($(40960))
    r68_value = tl.load($(candidate_vcat_5_ptr) + (((tl.load($(candidate_vsort_index_ptr) + ((r68_z / $(320)) % $(128)), dtype=tl.int64) % $(128)) * $(320)) + (r68_z % $(320))), dtype=tl.bfloat16)
    r68_value = r68_value.to(tl.bfloat16)
    r68_value = tl.broadcast_to(r68_value, [$(40960)])
    tl.store($(candidate_vindex_2_ptr) + r68_z, r68_value)
    r69_row = tl.arange($(128))
    r69_col = tl.arange($(1024))
    r69_lane = tl.arange($(256))
    for r69_group in range($(1)) {
      r69_left = tl.load($(candidate_vindex_2_ptr) + ((((((r69_row[:, None] * $(256)) + r69_lane[None, :]) / $(256)) % $(128)) * $(320)) + (((r69_row[:, None] * $(256)) + r69_lane[None, :]) % $(256))), dtype=tl.bfloat16)
      r69_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r69_lane[:, None] * $(1024)) + r69_col[None, :]) / $(128)) % $(8)) * $(65536)) + ((((r69_lane[:, None] * $(1024)) + r69_col[None, :]) % $(128)) * $(256)) + ((((r69_lane[:, None] * $(1024)) + r69_col[None, :]) / $(1024)) % $(256))), dtype=tl.bfloat16)
      r69_value = tl.dot(r69_left, r69_right)
      r69_value = r69_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_2_ptr) + ((r69_group * $(131072)) + (r69_row[:, None] * $(1024)) + r69_col[None, :]), r69_value)
    }
    r70_row = tl.arange($(128))
    r70_col = tl.arange($(1024))
    r70_lane = tl.arange($(256))
    for r70_group in range($(1)) {
      r70_left = tl.load($(candidate_vindex_2_ptr) + ((((((r70_row[:, None] * $(256)) + r70_lane[None, :]) / $(256)) % $(128)) * $(320)) + (((r70_row[:, None] * $(256)) + r70_lane[None, :]) % $(256))), dtype=tl.bfloat16)
      r70_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r70_lane[:, None] * $(1024)) + r70_col[None, :]) / $(128)) % $(8)) * $(65536)) + ((((r70_lane[:, None] * $(1024)) + r70_col[None, :]) % $(128)) * $(256)) + ((((r70_lane[:, None] * $(1024)) + r70_col[None, :]) / $(1024)) % $(256)) + $(32768)), dtype=tl.bfloat16)
      r70_value = tl.dot(r70_left, r70_right)
      r70_value = r70_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_3_ptr) + ((r70_group * $(131072)) + (r70_row[:, None] * $(1024)) + r70_col[None, :]), r70_value)
    }
    r71_z = tl.arange($(49152))
    r71_value = tl.where((r71_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r71_z / $(1536)) % $(32)) * $(1024)) + (((r71_z / $(192)) % $(8)) * $(128)) + ((r71_z % $(192)) - $(0))) / $(1024)) % $(128)) * $(1536)) + (((((((r71_z / $(1536)) % $(32)) * $(1024)) + (((r71_z / $(192)) % $(8)) * $(128)) + ((r71_z % $(192)) - $(0))) / $(128)) % $(8)) * $(192)) + (((((r71_z / $(1536)) % $(32)) * $(1024)) + (((r71_z / $(192)) % $(8)) * $(128)) + ((r71_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vadd_6_ptr) + ((((((((r71_z / $(1536)) % $(32)) * $(512)) + (((r71_z / $(192)) % $(8)) * $(64)) + ((r71_z % $(192)) - $(128))) / $(512)) % $(128)) * $(512)) + (((((((r71_z / $(1536)) % $(32)) * $(512)) + (((r71_z / $(192)) % $(8)) * $(64)) + ((r71_z % $(192)) - $(128))) / $(64)) % $(8)) * $(64)) + (((((r71_z / $(1536)) % $(32)) * $(512)) + (((r71_z / $(192)) % $(8)) * $(64)) + ((r71_z % $(192)) - $(128))) % $(64))), dtype=tl.bfloat16))
    r71_value = r71_value.to(tl.bfloat16)
    r71_value = tl.broadcast_to(r71_value, [$(49152)])
    tl.store($(candidate_vcat_6_ptr) + r71_z, r71_value)
    r72_z = tl.arange($(196608))
    r72_value = tl.where((r72_z % $(192)) < $(128), tl.load($(candidate_vbmm_2_ptr) + ((((((((r72_z / $(1536)) % $(128)) * $(1024)) + (((r72_z / $(192)) % $(8)) * $(128)) + ((r72_z % $(192)) - $(0))) / $(1024)) % $(128)) * $(1024)) + (((((((r72_z / $(1536)) % $(128)) * $(1024)) + (((r72_z / $(192)) % $(8)) * $(128)) + ((r72_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r72_z / $(1536)) % $(128)) * $(1024)) + (((r72_z / $(192)) % $(8)) * $(128)) + ((r72_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vindex_2_ptr) + ((((r72_z / $(1536)) % $(128)) * $(320)) + ((((r72_z % $(192)) - $(128)) + $(256)) % $(320))), dtype=tl.bfloat16))
    r72_value = r72_value.to(tl.bfloat16)
    r72_value = tl.broadcast_to(r72_value, [$(196608)])
    tl.store($(candidate_vcat_7_ptr) + r72_z, r72_value)
    r73_z = tl.arange($(18432))
    r73_value = tl.load($(candidate_vcat_6_ptr) + ((((r73_z / $(192)) % $(12)) * $(1536)) + (((r73_z / $(2304)) % $(8)) * $(192)) + (r73_z % $(192))), dtype=tl.bfloat16)
    r73_value = r73_value.to(tl.float32)
    r73_value = tl.broadcast_to(r73_value, [$(18432)])
    tl.store($(candidate_vto_copy_29_ptr) + r73_z, r73_value)
    r74_z = tl.arange($(18432))
    r74_value = tl.load($(candidate_vcat_7_ptr) + ((((r74_z / $(192)) % $(12)) * $(1536)) + (((r74_z / $(2304)) % $(8)) * $(192)) + (r74_z % $(192))), dtype=tl.bfloat16)
    r74_value = r74_value.to(tl.float32)
    r74_value = tl.broadcast_to(r74_value, [$(18432)])
    tl.store($(candidate_vto_copy_30_ptr) + r74_z, r74_value)
    r75_row = tl.arange($(12))
    r75_col = tl.arange($(12))
    r75_lane = tl.arange($(192))
    for r75_group in range($(8)) {
      r75_left = tl.load($(candidate_vto_copy_29_ptr) + (((((((r75_group % $(8)) * $(2304)) + (r75_row[:, None] * $(192)) + r75_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r75_group % $(8)) * $(2304)) + (r75_row[:, None] * $(192)) + r75_lane[None, :]) / $(192)) % $(12)) * $(192)) + ((((r75_group % $(8)) * $(2304)) + (r75_row[:, None] * $(192)) + r75_lane[None, :]) % $(192))), dtype=tl.float32)
      r75_right = tl.load($(candidate_vto_copy_30_ptr) + (((((((r75_group % $(8)) * $(2304)) + (r75_lane[:, None] * $(12)) + r75_col[None, :]) / $(2304)) % $(8)) * $(2304)) + (((((r75_group % $(8)) * $(2304)) + (r75_lane[:, None] * $(12)) + r75_col[None, :]) % $(12)) * $(192)) + (((((r75_group % $(8)) * $(2304)) + (r75_lane[:, None] * $(12)) + r75_col[None, :]) / $(12)) % $(192))), dtype=tl.float32)
      r75_value = tl.dot(r75_left, r75_right)
      r75_value = r75_value.to(tl.float32)
      tl.store($(candidate_vbmm_4_ptr) + ((r75_group * $(144)) + (r75_row[:, None] * $(12)) + r75_col[None, :]), r75_value)
    }
    r76_z = tl.arange($(1152))
    r76_value = (tl.load($(candidate_vbmm_4_ptr) + ((((r76_z / $(144)) % $(8)) * $(144)) + (((r76_z / $(12)) % $(12)) * $(12)) + (r76_z % $(12))), dtype=tl.float32) * 0.07216878364870322)
    r76_value = r76_value.to(tl.float32)
    r76_value = tl.broadcast_to(r76_value, [$(1152)])
    tl.store($(candidate_vmul_23_ptr) + r76_z, r76_value)
    r77_z = tl.arange($(12))
    r77_value = (tl.cast(r77_z, tl.int64) + tl.full([], $(12), dtype=tl.int64))
    r77_value = tl.broadcast_to(r77_value, [$(12)])
    tl.store($(candidate_vadd_8_ptr) + r77_z, r77_value)
    r78_z = tl.arange($(12))
    r78_value = (tl.load($(candidate_vadd_8_ptr) + r78_z, dtype=tl.int64) - tl.full([], $(12), dtype=tl.int64))
    r78_value = tl.broadcast_to(r78_value, [$(12)])
    tl.store($(candidate_vsub_9_ptr) + r78_z, r78_value)
    r79_z = tl.arange($(144))
    r79_value = (tl.cast((r79_z % $(12)), tl.int64) > tl.load($(candidate_vsub_9_ptr) + ((r79_z / $(12)) % $(12)), dtype=tl.int64))
    r79_value = tl.broadcast_to(r79_value, [$(144)])
    tl.store($(candidate_vgt_ptr) + r79_z, r79_value)
    r80_z = tl.arange($(1152))
    r80_value = tl.where(tl.load($(candidate_vgt_ptr) + ((((r80_z / $(12)) % $(12)) * $(12)) + (r80_z % $(12))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_23_ptr) + ((((r80_z / $(144)) % $(8)) * $(144)) + (((r80_z / $(12)) % $(12)) * $(12)) + (r80_z % $(12))), dtype=tl.float32))
    r80_value = r80_value.to(tl.float32)
    r80_value = tl.broadcast_to(r80_value, [$(1152)])
    tl.store($(candidate_vmasked_fill_ptr) + r80_z, r80_value)
    r81_z = tl.arange($(1152))
    r81_lane = tl.arange($(12))
    r81_row_values = tl.load($(candidate_vmasked_fill_ptr) + ((((r81_z[:, None] / $(144)) % $(8)) * $(144)) + (((r81_z[:, None] / $(12)) % $(12)) * $(12)) + (r81_lane[None, :] % $(12))), dtype=tl.float32)
    r81_maximum = tl.max(r81_row_values, axis=1, keep_dims=true)
    r81_weights = tl.exp(r81_row_values - r81_maximum)
    r81_total = tl.sum(r81_weights, axis=1)
    r81_maximum = tl.max(r81_row_values, axis=1)
    r81_value = tl.exp(tl.load($(candidate_vmasked_fill_ptr) + ((((r81_z / $(144)) % $(8)) * $(144)) + (((r81_z / $(12)) % $(12)) * $(12)) + (r81_z % $(12))), dtype=tl.float32) - r81_maximum) / r81_total
    r81_value = r81_value.to(tl.float32)
    r81_value = tl.broadcast_to(r81_value, [$(1152)])
    tl.store($(candidate_vsoftmax_ptr) + r81_z, r81_value)
    r82_z = tl.arange($(1152))
    r82_value = tl.load($(candidate_vsoftmax_ptr) + ((((r82_z / $(144)) % $(8)) * $(144)) + (((r82_z / $(12)) % $(12)) * $(12)) + (r82_z % $(12))), dtype=tl.float32)
    r82_value = r82_value.to(tl.bfloat16)
    r82_value = tl.broadcast_to(r82_value, [$(1152)])
    tl.store($(candidate_vto_copy_31_ptr) + r82_z, r82_value)
    r83_z = tl.arange($(6144))
    r83_value = tl.load($(candidate_vcat_6_ptr) + ((((r83_z / $(192)) % $(4)) * $(1536)) + (((r83_z / $(768)) % $(8)) * $(192)) + (r83_z % $(192)) + $(18432)), dtype=tl.bfloat16)
    r83_value = r83_value.to(tl.float32)
    r83_value = tl.broadcast_to(r83_value, [$(6144)])
    tl.store($(candidate_vto_copy_32_ptr) + r83_z, r83_value)
    r84_z = tl.arange($(6144))
    r84_value = tl.load($(candidate_vcat_7_ptr) + ((((r84_z / $(192)) % $(4)) * $(1536)) + (((r84_z / $(768)) % $(8)) * $(192)) + (r84_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r84_value = r84_value.to(tl.float32)
    r84_value = tl.broadcast_to(r84_value, [$(6144)])
    tl.store($(candidate_vto_copy_33_ptr) + r84_z, r84_value)
    r85_row = tl.arange($(4))
    r85_col = tl.arange($(4))
    r85_lane = tl.arange($(192))
    for r85_group in range($(8)) {
      r85_left = tl.load($(candidate_vto_copy_32_ptr) + (((((((r85_group % $(8)) * $(768)) + (r85_row[:, None] * $(192)) + r85_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r85_group % $(8)) * $(768)) + (r85_row[:, None] * $(192)) + r85_lane[None, :]) / $(192)) % $(4)) * $(192)) + ((((r85_group % $(8)) * $(768)) + (r85_row[:, None] * $(192)) + r85_lane[None, :]) % $(192))), dtype=tl.float32)
      r85_right = tl.load($(candidate_vto_copy_33_ptr) + (((((((r85_group % $(8)) * $(768)) + (r85_lane[:, None] * $(4)) + r85_col[None, :]) / $(768)) % $(8)) * $(768)) + (((((r85_group % $(8)) * $(768)) + (r85_lane[:, None] * $(4)) + r85_col[None, :]) % $(4)) * $(192)) + (((((r85_group % $(8)) * $(768)) + (r85_lane[:, None] * $(4)) + r85_col[None, :]) / $(4)) % $(192))), dtype=tl.float32)
      r85_value = tl.dot(r85_left, r85_right)
      r85_value = r85_value.to(tl.float32)
      tl.store($(candidate_vbmm_6_ptr) + ((r85_group * $(16)) + (r85_row[:, None] * $(4)) + r85_col[None, :]), r85_value)
    }
    r86_z = tl.arange($(128))
    r86_value = (tl.load($(candidate_vbmm_6_ptr) + ((((r86_z / $(16)) % $(8)) * $(16)) + (((r86_z / $(4)) % $(4)) * $(4)) + (r86_z % $(4))), dtype=tl.float32) * 0.07216878364870322)
    r86_value = r86_value.to(tl.float32)
    r86_value = tl.broadcast_to(r86_value, [$(128)])
    tl.store($(candidate_vmul_24_ptr) + r86_z, r86_value)
    r87_z = tl.arange($(4))
    r87_value = (tl.cast(r87_z, tl.int64) + tl.full([], $(4), dtype=tl.int64))
    r87_value = tl.broadcast_to(r87_value, [$(4)])
    tl.store($(candidate_vadd_9_ptr) + r87_z, r87_value)
    r88_z = tl.arange($(4))
    r88_value = (tl.load($(candidate_vadd_9_ptr) + r88_z, dtype=tl.int64) - tl.full([], $(4), dtype=tl.int64))
    r88_value = tl.broadcast_to(r88_value, [$(4)])
    tl.store($(candidate_vsub_10_ptr) + r88_z, r88_value)
    r89_z = tl.arange($(16))
    r89_value = (tl.cast((r89_z % $(4)), tl.int64) > tl.load($(candidate_vsub_10_ptr) + ((r89_z / $(4)) % $(4)), dtype=tl.int64))
    r89_value = tl.broadcast_to(r89_value, [$(16)])
    tl.store($(candidate_vgt_1_ptr) + r89_z, r89_value)
    r90_z = tl.arange($(128))
    r90_value = tl.where(tl.load($(candidate_vgt_1_ptr) + ((((r90_z / $(4)) % $(4)) * $(4)) + (r90_z % $(4))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_24_ptr) + ((((r90_z / $(16)) % $(8)) * $(16)) + (((r90_z / $(4)) % $(4)) * $(4)) + (r90_z % $(4))), dtype=tl.float32))
    r90_value = r90_value.to(tl.float32)
    r90_value = tl.broadcast_to(r90_value, [$(128)])
    tl.store($(candidate_vmasked_fill_1_ptr) + r90_z, r90_value)
    r91_z = tl.arange($(128))
    r91_lane = tl.arange($(4))
    r91_row_values = tl.load($(candidate_vmasked_fill_1_ptr) + ((((r91_z[:, None] / $(16)) % $(8)) * $(16)) + (((r91_z[:, None] / $(4)) % $(4)) * $(4)) + (r91_lane[None, :] % $(4))), dtype=tl.float32)
    r91_maximum = tl.max(r91_row_values, axis=1, keep_dims=true)
    r91_weights = tl.exp(r91_row_values - r91_maximum)
    r91_total = tl.sum(r91_weights, axis=1)
    r91_maximum = tl.max(r91_row_values, axis=1)
    r91_value = tl.exp(tl.load($(candidate_vmasked_fill_1_ptr) + ((((r91_z / $(16)) % $(8)) * $(16)) + (((r91_z / $(4)) % $(4)) * $(4)) + (r91_z % $(4))), dtype=tl.float32) - r91_maximum) / r91_total
    r91_value = r91_value.to(tl.float32)
    r91_value = tl.broadcast_to(r91_value, [$(128)])
    tl.store($(candidate_vsoftmax_1_ptr) + r91_z, r91_value)
    r92_z = tl.arange($(128))
    r92_value = tl.load($(candidate_vsoftmax_1_ptr) + ((((r92_z / $(16)) % $(8)) * $(16)) + (((r92_z / $(4)) % $(4)) * $(4)) + (r92_z % $(4))), dtype=tl.float32)
    r92_value = r92_value.to(tl.bfloat16)
    r92_value = tl.broadcast_to(r92_value, [$(128)])
    tl.store($(candidate_vto_copy_34_ptr) + r92_z, r92_value)
    r93_z = tl.arange($(24576))
    r93_value = tl.load($(candidate_vcat_6_ptr) + ((((r93_z / $(192)) % $(16)) * $(1536)) + (((r93_z / $(3072)) % $(8)) * $(192)) + (r93_z % $(192)) + $(24576)), dtype=tl.bfloat16)
    r93_value = r93_value.to(tl.float32)
    r93_value = tl.broadcast_to(r93_value, [$(24576)])
    tl.store($(candidate_vto_copy_35_ptr) + r93_z, r93_value)
    r94_z = tl.arange($(24576))
    r94_value = tl.load($(candidate_vcat_7_ptr) + ((((r94_z / $(192)) % $(16)) * $(1536)) + (((r94_z / $(3072)) % $(8)) * $(192)) + (r94_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r94_value = r94_value.to(tl.float32)
    r94_value = tl.broadcast_to(r94_value, [$(24576)])
    tl.store($(candidate_vto_copy_36_ptr) + r94_z, r94_value)
    r95_row = tl.arange($(16))
    r95_col = tl.arange($(16))
    r95_lane = tl.arange($(192))
    for r95_group in range($(8)) {
      r95_left = tl.load($(candidate_vto_copy_35_ptr) + (((((((r95_group % $(8)) * $(3072)) + (r95_row[:, None] * $(192)) + r95_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r95_group % $(8)) * $(3072)) + (r95_row[:, None] * $(192)) + r95_lane[None, :]) / $(192)) % $(16)) * $(192)) + ((((r95_group % $(8)) * $(3072)) + (r95_row[:, None] * $(192)) + r95_lane[None, :]) % $(192))), dtype=tl.float32)
      r95_right = tl.load($(candidate_vto_copy_36_ptr) + (((((((r95_group % $(8)) * $(3072)) + (r95_lane[:, None] * $(16)) + r95_col[None, :]) / $(3072)) % $(8)) * $(3072)) + (((((r95_group % $(8)) * $(3072)) + (r95_lane[:, None] * $(16)) + r95_col[None, :]) % $(16)) * $(192)) + (((((r95_group % $(8)) * $(3072)) + (r95_lane[:, None] * $(16)) + r95_col[None, :]) / $(16)) % $(192))), dtype=tl.float32)
      r95_value = tl.dot(r95_left, r95_right)
      r95_value = r95_value.to(tl.float32)
      tl.store($(candidate_vbmm_8_ptr) + ((r95_group * $(256)) + (r95_row[:, None] * $(16)) + r95_col[None, :]), r95_value)
    }
    r96_z = tl.arange($(2048))
    r96_value = (tl.load($(candidate_vbmm_8_ptr) + ((((r96_z / $(256)) % $(8)) * $(256)) + (((r96_z / $(16)) % $(16)) * $(16)) + (r96_z % $(16))), dtype=tl.float32) * 0.07216878364870322)
    r96_value = r96_value.to(tl.float32)
    r96_value = tl.broadcast_to(r96_value, [$(2048)])
    tl.store($(candidate_vmul_25_ptr) + r96_z, r96_value)
    r97_z = tl.arange($(16))
    r97_value = (tl.cast(r97_z, tl.int64) + tl.full([], $(16), dtype=tl.int64))
    r97_value = tl.broadcast_to(r97_value, [$(16)])
    tl.store($(candidate_vadd_10_ptr) + r97_z, r97_value)
    r98_z = tl.arange($(16))
    r98_value = (tl.load($(candidate_vadd_10_ptr) + r98_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r98_value = tl.broadcast_to(r98_value, [$(16)])
    tl.store($(candidate_vsub_11_ptr) + r98_z, r98_value)
    r99_z = tl.arange($(256))
    r99_value = (tl.cast((r99_z % $(16)), tl.int64) > tl.load($(candidate_vsub_11_ptr) + ((r99_z / $(16)) % $(16)), dtype=tl.int64))
    r99_value = tl.broadcast_to(r99_value, [$(256)])
    tl.store($(candidate_vgt_2_ptr) + r99_z, r99_value)
    r100_z = tl.arange($(2048))
    r100_value = tl.where(tl.load($(candidate_vgt_2_ptr) + ((((r100_z / $(16)) % $(16)) * $(16)) + (r100_z % $(16))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_25_ptr) + ((((r100_z / $(256)) % $(8)) * $(256)) + (((r100_z / $(16)) % $(16)) * $(16)) + (r100_z % $(16))), dtype=tl.float32))
    r100_value = r100_value.to(tl.float32)
    r100_value = tl.broadcast_to(r100_value, [$(2048)])
    tl.store($(candidate_vmasked_fill_2_ptr) + r100_z, r100_value)
    r101_z = tl.arange($(2048))
    r101_lane = tl.arange($(16))
    r101_row_values = tl.load($(candidate_vmasked_fill_2_ptr) + ((((r101_z[:, None] / $(256)) % $(8)) * $(256)) + (((r101_z[:, None] / $(16)) % $(16)) * $(16)) + (r101_lane[None, :] % $(16))), dtype=tl.float32)
    r101_maximum = tl.max(r101_row_values, axis=1, keep_dims=true)
    r101_weights = tl.exp(r101_row_values - r101_maximum)
    r101_total = tl.sum(r101_weights, axis=1)
    r101_maximum = tl.max(r101_row_values, axis=1)
    r101_value = tl.exp(tl.load($(candidate_vmasked_fill_2_ptr) + ((((r101_z / $(256)) % $(8)) * $(256)) + (((r101_z / $(16)) % $(16)) * $(16)) + (r101_z % $(16))), dtype=tl.float32) - r101_maximum) / r101_total
    r101_value = r101_value.to(tl.float32)
    r101_value = tl.broadcast_to(r101_value, [$(2048)])
    tl.store($(candidate_vsoftmax_2_ptr) + r101_z, r101_value)
    r102_z = tl.arange($(2048))
    r102_value = tl.load($(candidate_vsoftmax_2_ptr) + ((((r102_z / $(256)) % $(8)) * $(256)) + (((r102_z / $(16)) % $(16)) * $(16)) + (r102_z % $(16))), dtype=tl.float32)
    r102_value = r102_value.to(tl.bfloat16)
    r102_value = tl.broadcast_to(r102_value, [$(2048)])
    tl.store($(candidate_vto_copy_37_ptr) + r102_z, r102_value)
    r103_row = tl.arange($(128))
    r103_col = tl.arange($(1024))
    r103_lane = tl.arange($(256))
    for r103_group in range($(1)) {
      r103_left = tl.load($(candidate_vindex_2_ptr) + ((((((r103_row[:, None] * $(256)) + r103_lane[None, :]) / $(256)) % $(128)) * $(320)) + (((r103_row[:, None] * $(256)) + r103_lane[None, :]) % $(256))), dtype=tl.bfloat16)
      r103_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r103_lane[:, None] * $(1024)) + r103_col[None, :]) / $(128)) % $(8)) * $(65536)) + ((((r103_lane[:, None] * $(1024)) + r103_col[None, :]) % $(128)) * $(256)) + ((((r103_lane[:, None] * $(1024)) + r103_col[None, :]) / $(1024)) % $(256))), dtype=tl.bfloat16)
      r103_value = tl.dot(r103_left, r103_right)
      r103_value = r103_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_10_ptr) + ((r103_group * $(131072)) + (r103_row[:, None] * $(1024)) + r103_col[None, :]), r103_value)
    }
    r104_row = tl.arange($(128))
    r104_col = tl.arange($(1024))
    r104_lane = tl.arange($(256))
    for r104_group in range($(1)) {
      r104_left = tl.load($(candidate_vindex_2_ptr) + ((((((r104_row[:, None] * $(256)) + r104_lane[None, :]) / $(256)) % $(128)) * $(320)) + (((r104_row[:, None] * $(256)) + r104_lane[None, :]) % $(256))), dtype=tl.bfloat16)
      r104_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r104_lane[:, None] * $(1024)) + r104_col[None, :]) / $(128)) % $(8)) * $(65536)) + ((((r104_lane[:, None] * $(1024)) + r104_col[None, :]) % $(128)) * $(256)) + ((((r104_lane[:, None] * $(1024)) + r104_col[None, :]) / $(1024)) % $(256)) + $(32768)), dtype=tl.bfloat16)
      r104_value = tl.dot(r104_left, r104_right)
      r104_value = r104_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_11_ptr) + ((r104_group * $(131072)) + (r104_row[:, None] * $(1024)) + r104_col[None, :]), r104_value)
    }
    r105_z = tl.arange($(49152))
    r105_value = tl.where((r105_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r105_z / $(1536)) % $(32)) * $(1024)) + (((r105_z / $(192)) % $(8)) * $(128)) + ((r105_z % $(192)) - $(0)) + $(32768)) / $(1024)) % $(128)) * $(1536)) + (((((((r105_z / $(1536)) % $(32)) * $(1024)) + (((r105_z / $(192)) % $(8)) * $(128)) + ((r105_z % $(192)) - $(0)) + $(32768)) / $(128)) % $(8)) * $(192)) + (((((r105_z / $(1536)) % $(32)) * $(1024)) + (((r105_z / $(192)) % $(8)) * $(128)) + ((r105_z % $(192)) - $(0)) + $(32768)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vadd_6_ptr) + ((((((((r105_z / $(1536)) % $(32)) * $(512)) + (((r105_z / $(192)) % $(8)) * $(64)) + ((r105_z % $(192)) - $(128)) + $(16384)) / $(512)) % $(128)) * $(512)) + (((((((r105_z / $(1536)) % $(32)) * $(512)) + (((r105_z / $(192)) % $(8)) * $(64)) + ((r105_z % $(192)) - $(128)) + $(16384)) / $(64)) % $(8)) * $(64)) + (((((r105_z / $(1536)) % $(32)) * $(512)) + (((r105_z / $(192)) % $(8)) * $(64)) + ((r105_z % $(192)) - $(128)) + $(16384)) % $(64))), dtype=tl.bfloat16))
    r105_value = r105_value.to(tl.bfloat16)
    r105_value = tl.broadcast_to(r105_value, [$(49152)])
    tl.store($(candidate_vcat_9_ptr) + r105_z, r105_value)
    r106_z = tl.arange($(196608))
    r106_value = tl.where((r106_z % $(192)) < $(128), tl.load($(candidate_vbmm_10_ptr) + ((((((((r106_z / $(1536)) % $(128)) * $(1024)) + (((r106_z / $(192)) % $(8)) * $(128)) + ((r106_z % $(192)) - $(0))) / $(1024)) % $(128)) * $(1024)) + (((((((r106_z / $(1536)) % $(128)) * $(1024)) + (((r106_z / $(192)) % $(8)) * $(128)) + ((r106_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r106_z / $(1536)) % $(128)) * $(1024)) + (((r106_z / $(192)) % $(8)) * $(128)) + ((r106_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vindex_2_ptr) + ((((r106_z / $(1536)) % $(128)) * $(320)) + ((((r106_z % $(192)) - $(128)) + $(256)) % $(320))), dtype=tl.bfloat16))
    r106_value = r106_value.to(tl.bfloat16)
    r106_value = tl.broadcast_to(r106_value, [$(196608)])
    tl.store($(candidate_vcat_10_ptr) + r106_z, r106_value)
    r107_z = tl.arange($(18432))
    r107_value = tl.load($(candidate_vcat_9_ptr) + ((((r107_z / $(192)) % $(12)) * $(1536)) + (((r107_z / $(2304)) % $(8)) * $(192)) + (r107_z % $(192))), dtype=tl.bfloat16)
    r107_value = r107_value.to(tl.float32)
    r107_value = tl.broadcast_to(r107_value, [$(18432)])
    tl.store($(candidate_vto_copy_38_ptr) + r107_z, r107_value)
    r108_z = tl.arange($(73728))
    r108_value = tl.load($(candidate_vcat_10_ptr) + ((((r108_z / $(192)) % $(48)) * $(1536)) + (((r108_z / $(9216)) % $(8)) * $(192)) + (r108_z % $(192))), dtype=tl.bfloat16)
    r108_value = r108_value.to(tl.float32)
    r108_value = tl.broadcast_to(r108_value, [$(73728)])
    tl.store($(candidate_vto_copy_39_ptr) + r108_z, r108_value)
    r109_row = tl.arange($(12))
    r109_col = tl.arange($(48))
    r109_lane = tl.arange($(192))
    for r109_group in range($(8)) {
      r109_left = tl.load($(candidate_vto_copy_38_ptr) + (((((((r109_group % $(8)) * $(2304)) + (r109_row[:, None] * $(192)) + r109_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r109_group % $(8)) * $(2304)) + (r109_row[:, None] * $(192)) + r109_lane[None, :]) / $(192)) % $(12)) * $(192)) + ((((r109_group % $(8)) * $(2304)) + (r109_row[:, None] * $(192)) + r109_lane[None, :]) % $(192))), dtype=tl.float32)
      r109_right = tl.load($(candidate_vto_copy_39_ptr) + (((((((r109_group % $(8)) * $(9216)) + (r109_lane[:, None] * $(48)) + r109_col[None, :]) / $(9216)) % $(8)) * $(9216)) + (((((r109_group % $(8)) * $(9216)) + (r109_lane[:, None] * $(48)) + r109_col[None, :]) % $(48)) * $(192)) + (((((r109_group % $(8)) * $(9216)) + (r109_lane[:, None] * $(48)) + r109_col[None, :]) / $(48)) % $(192))), dtype=tl.float32)
      r109_value = tl.dot(r109_left, r109_right)
      r109_value = r109_value.to(tl.float32)
      tl.store($(candidate_vbmm_12_ptr) + ((r109_group * $(576)) + (r109_row[:, None] * $(48)) + r109_col[None, :]), r109_value)
    }
    r110_z = tl.arange($(4608))
    r110_value = (tl.load($(candidate_vbmm_12_ptr) + ((((r110_z / $(576)) % $(8)) * $(576)) + (((r110_z / $(48)) % $(12)) * $(48)) + (r110_z % $(48))), dtype=tl.float32) * 0.07216878364870322)
    r110_value = r110_value.to(tl.float32)
    r110_value = tl.broadcast_to(r110_value, [$(4608)])
    tl.store($(candidate_vmul_26_ptr) + r110_z, r110_value)
    r111_z = tl.arange($(12))
    r111_value = (tl.cast(r111_z, tl.int64) + tl.full([], $(48), dtype=tl.int64))
    r111_value = tl.broadcast_to(r111_value, [$(12)])
    tl.store($(candidate_vadd_11_ptr) + r111_z, r111_value)
    r112_z = tl.arange($(12))
    r112_value = (tl.load($(candidate_vadd_11_ptr) + r112_z, dtype=tl.int64) - tl.full([], $(12), dtype=tl.int64))
    r112_value = tl.broadcast_to(r112_value, [$(12)])
    tl.store($(candidate_vsub_12_ptr) + r112_z, r112_value)
    r113_z = tl.arange($(576))
    r113_value = (tl.cast((r113_z % $(48)), tl.int64) > tl.load($(candidate_vsub_12_ptr) + ((r113_z / $(48)) % $(12)), dtype=tl.int64))
    r113_value = tl.broadcast_to(r113_value, [$(576)])
    tl.store($(candidate_vgt_3_ptr) + r113_z, r113_value)
    r114_z = tl.arange($(4608))
    r114_value = tl.where(tl.load($(candidate_vgt_3_ptr) + ((((r114_z / $(48)) % $(12)) * $(48)) + (r114_z % $(48))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_26_ptr) + ((((r114_z / $(576)) % $(8)) * $(576)) + (((r114_z / $(48)) % $(12)) * $(48)) + (r114_z % $(48))), dtype=tl.float32))
    r114_value = r114_value.to(tl.float32)
    r114_value = tl.broadcast_to(r114_value, [$(4608)])
    tl.store($(candidate_vmasked_fill_3_ptr) + r114_z, r114_value)
    r115_z = tl.arange($(4608))
    r115_lane = tl.arange($(48))
    r115_row_values = tl.load($(candidate_vmasked_fill_3_ptr) + ((((r115_z[:, None] / $(576)) % $(8)) * $(576)) + (((r115_z[:, None] / $(48)) % $(12)) * $(48)) + (r115_lane[None, :] % $(48))), dtype=tl.float32)
    r115_maximum = tl.max(r115_row_values, axis=1, keep_dims=true)
    r115_weights = tl.exp(r115_row_values - r115_maximum)
    r115_total = tl.sum(r115_weights, axis=1)
    r115_maximum = tl.max(r115_row_values, axis=1)
    r115_value = tl.exp(tl.load($(candidate_vmasked_fill_3_ptr) + ((((r115_z / $(576)) % $(8)) * $(576)) + (((r115_z / $(48)) % $(12)) * $(48)) + (r115_z % $(48))), dtype=tl.float32) - r115_maximum) / r115_total
    r115_value = r115_value.to(tl.float32)
    r115_value = tl.broadcast_to(r115_value, [$(4608)])
    tl.store($(candidate_vsoftmax_3_ptr) + r115_z, r115_value)
    r116_z = tl.arange($(4608))
    r116_value = tl.load($(candidate_vsoftmax_3_ptr) + ((((r116_z / $(576)) % $(8)) * $(576)) + (((r116_z / $(48)) % $(12)) * $(48)) + (r116_z % $(48))), dtype=tl.float32)
    r116_value = r116_value.to(tl.bfloat16)
    r116_value = tl.broadcast_to(r116_value, [$(4608)])
    tl.store($(candidate_vto_copy_40_ptr) + r116_z, r116_value)
    r117_z = tl.arange($(6144))
    r117_value = tl.load($(candidate_vcat_9_ptr) + ((((r117_z / $(192)) % $(4)) * $(1536)) + (((r117_z / $(768)) % $(8)) * $(192)) + (r117_z % $(192)) + $(18432)), dtype=tl.bfloat16)
    r117_value = r117_value.to(tl.float32)
    r117_value = tl.broadcast_to(r117_value, [$(6144)])
    tl.store($(candidate_vto_copy_41_ptr) + r117_z, r117_value)
    r118_z = tl.arange($(24576))
    r118_value = tl.load($(candidate_vcat_10_ptr) + ((((r118_z / $(192)) % $(16)) * $(1536)) + (((r118_z / $(3072)) % $(8)) * $(192)) + (r118_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r118_value = r118_value.to(tl.float32)
    r118_value = tl.broadcast_to(r118_value, [$(24576)])
    tl.store($(candidate_vto_copy_42_ptr) + r118_z, r118_value)
    r119_row = tl.arange($(4))
    r119_col = tl.arange($(16))
    r119_lane = tl.arange($(192))
    for r119_group in range($(8)) {
      r119_left = tl.load($(candidate_vto_copy_41_ptr) + (((((((r119_group % $(8)) * $(768)) + (r119_row[:, None] * $(192)) + r119_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r119_group % $(8)) * $(768)) + (r119_row[:, None] * $(192)) + r119_lane[None, :]) / $(192)) % $(4)) * $(192)) + ((((r119_group % $(8)) * $(768)) + (r119_row[:, None] * $(192)) + r119_lane[None, :]) % $(192))), dtype=tl.float32)
      r119_right = tl.load($(candidate_vto_copy_42_ptr) + (((((((r119_group % $(8)) * $(3072)) + (r119_lane[:, None] * $(16)) + r119_col[None, :]) / $(3072)) % $(8)) * $(3072)) + (((((r119_group % $(8)) * $(3072)) + (r119_lane[:, None] * $(16)) + r119_col[None, :]) % $(16)) * $(192)) + (((((r119_group % $(8)) * $(3072)) + (r119_lane[:, None] * $(16)) + r119_col[None, :]) / $(16)) % $(192))), dtype=tl.float32)
      r119_value = tl.dot(r119_left, r119_right)
      r119_value = r119_value.to(tl.float32)
      tl.store($(candidate_vbmm_14_ptr) + ((r119_group * $(64)) + (r119_row[:, None] * $(16)) + r119_col[None, :]), r119_value)
    }
    r120_z = tl.arange($(512))
    r120_value = (tl.load($(candidate_vbmm_14_ptr) + ((((r120_z / $(64)) % $(8)) * $(64)) + (((r120_z / $(16)) % $(4)) * $(16)) + (r120_z % $(16))), dtype=tl.float32) * 0.07216878364870322)
    r120_value = r120_value.to(tl.float32)
    r120_value = tl.broadcast_to(r120_value, [$(512)])
    tl.store($(candidate_vmul_27_ptr) + r120_z, r120_value)
    r121_z = tl.arange($(4))
    r121_value = (tl.cast(r121_z, tl.int64) + tl.full([], $(16), dtype=tl.int64))
    r121_value = tl.broadcast_to(r121_value, [$(4)])
    tl.store($(candidate_vadd_12_ptr) + r121_z, r121_value)
    r122_z = tl.arange($(4))
    r122_value = (tl.load($(candidate_vadd_12_ptr) + r122_z, dtype=tl.int64) - tl.full([], $(4), dtype=tl.int64))
    r122_value = tl.broadcast_to(r122_value, [$(4)])
    tl.store($(candidate_vsub_13_ptr) + r122_z, r122_value)
    r123_z = tl.arange($(64))
    r123_value = (tl.cast((r123_z % $(16)), tl.int64) > tl.load($(candidate_vsub_13_ptr) + ((r123_z / $(16)) % $(4)), dtype=tl.int64))
    r123_value = tl.broadcast_to(r123_value, [$(64)])
    tl.store($(candidate_vgt_4_ptr) + r123_z, r123_value)
    r124_z = tl.arange($(512))
    r124_value = tl.where(tl.load($(candidate_vgt_4_ptr) + ((((r124_z / $(16)) % $(4)) * $(16)) + (r124_z % $(16))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_27_ptr) + ((((r124_z / $(64)) % $(8)) * $(64)) + (((r124_z / $(16)) % $(4)) * $(16)) + (r124_z % $(16))), dtype=tl.float32))
    r124_value = r124_value.to(tl.float32)
    r124_value = tl.broadcast_to(r124_value, [$(512)])
    tl.store($(candidate_vmasked_fill_4_ptr) + r124_z, r124_value)
    r125_z = tl.arange($(512))
    r125_lane = tl.arange($(16))
    r125_row_values = tl.load($(candidate_vmasked_fill_4_ptr) + ((((r125_z[:, None] / $(64)) % $(8)) * $(64)) + (((r125_z[:, None] / $(16)) % $(4)) * $(16)) + (r125_lane[None, :] % $(16))), dtype=tl.float32)
    r125_maximum = tl.max(r125_row_values, axis=1, keep_dims=true)
    r125_weights = tl.exp(r125_row_values - r125_maximum)
    r125_total = tl.sum(r125_weights, axis=1)
    r125_maximum = tl.max(r125_row_values, axis=1)
    r125_value = tl.exp(tl.load($(candidate_vmasked_fill_4_ptr) + ((((r125_z / $(64)) % $(8)) * $(64)) + (((r125_z / $(16)) % $(4)) * $(16)) + (r125_z % $(16))), dtype=tl.float32) - r125_maximum) / r125_total
    r125_value = r125_value.to(tl.float32)
    r125_value = tl.broadcast_to(r125_value, [$(512)])
    tl.store($(candidate_vsoftmax_4_ptr) + r125_z, r125_value)
    r126_z = tl.arange($(512))
    r126_value = tl.load($(candidate_vsoftmax_4_ptr) + ((((r126_z / $(64)) % $(8)) * $(64)) + (((r126_z / $(16)) % $(4)) * $(16)) + (r126_z % $(16))), dtype=tl.float32)
    r126_value = r126_value.to(tl.bfloat16)
    r126_value = tl.broadcast_to(r126_value, [$(512)])
    tl.store($(candidate_vto_copy_43_ptr) + r126_z, r126_value)
    r127_z = tl.arange($(24576))
    r127_value = tl.load($(candidate_vcat_9_ptr) + ((((r127_z / $(192)) % $(16)) * $(1536)) + (((r127_z / $(3072)) % $(8)) * $(192)) + (r127_z % $(192)) + $(24576)), dtype=tl.bfloat16)
    r127_value = r127_value.to(tl.float32)
    r127_value = tl.broadcast_to(r127_value, [$(24576)])
    tl.store($(candidate_vto_copy_44_ptr) + r127_z, r127_value)
    r128_z = tl.arange($(98304))
    r128_value = tl.load($(candidate_vcat_10_ptr) + ((((r128_z / $(192)) % $(64)) * $(1536)) + (((r128_z / $(12288)) % $(8)) * $(192)) + (r128_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r128_value = r128_value.to(tl.float32)
    r128_value = tl.broadcast_to(r128_value, [$(98304)])
    tl.store($(candidate_vto_copy_45_ptr) + r128_z, r128_value)
    r129_row = tl.arange($(16))
    r129_col = tl.arange($(64))
    r129_lane = tl.arange($(192))
    for r129_group in range($(8)) {
      r129_left = tl.load($(candidate_vto_copy_44_ptr) + (((((((r129_group % $(8)) * $(3072)) + (r129_row[:, None] * $(192)) + r129_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r129_group % $(8)) * $(3072)) + (r129_row[:, None] * $(192)) + r129_lane[None, :]) / $(192)) % $(16)) * $(192)) + ((((r129_group % $(8)) * $(3072)) + (r129_row[:, None] * $(192)) + r129_lane[None, :]) % $(192))), dtype=tl.float32)
      r129_right = tl.load($(candidate_vto_copy_45_ptr) + (((((((r129_group % $(8)) * $(12288)) + (r129_lane[:, None] * $(64)) + r129_col[None, :]) / $(12288)) % $(8)) * $(12288)) + (((((r129_group % $(8)) * $(12288)) + (r129_lane[:, None] * $(64)) + r129_col[None, :]) % $(64)) * $(192)) + (((((r129_group % $(8)) * $(12288)) + (r129_lane[:, None] * $(64)) + r129_col[None, :]) / $(64)) % $(192))), dtype=tl.float32)
      r129_value = tl.dot(r129_left, r129_right)
      r129_value = r129_value.to(tl.float32)
      tl.store($(candidate_vbmm_16_ptr) + ((r129_group * $(1024)) + (r129_row[:, None] * $(64)) + r129_col[None, :]), r129_value)
    }
    r130_z = tl.arange($(8192))
    r130_value = (tl.load($(candidate_vbmm_16_ptr) + ((((r130_z / $(1024)) % $(8)) * $(1024)) + (((r130_z / $(64)) % $(16)) * $(64)) + (r130_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r130_value = r130_value.to(tl.float32)
    r130_value = tl.broadcast_to(r130_value, [$(8192)])
    tl.store($(candidate_vmul_28_ptr) + r130_z, r130_value)
    r131_z = tl.arange($(16))
    r131_value = (tl.cast(r131_z, tl.int64) + tl.full([], $(64), dtype=tl.int64))
    r131_value = tl.broadcast_to(r131_value, [$(16)])
    tl.store($(candidate_vadd_13_ptr) + r131_z, r131_value)
    r132_z = tl.arange($(16))
    r132_value = (tl.load($(candidate_vadd_13_ptr) + r132_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r132_value = tl.broadcast_to(r132_value, [$(16)])
    tl.store($(candidate_vsub_14_ptr) + r132_z, r132_value)
    r133_z = tl.arange($(1024))
    r133_value = (tl.cast((r133_z % $(64)), tl.int64) > tl.load($(candidate_vsub_14_ptr) + ((r133_z / $(64)) % $(16)), dtype=tl.int64))
    r133_value = tl.broadcast_to(r133_value, [$(1024)])
    tl.store($(candidate_vgt_5_ptr) + r133_z, r133_value)
    r134_z = tl.arange($(8192))
    r134_value = tl.where(tl.load($(candidate_vgt_5_ptr) + ((((r134_z / $(64)) % $(16)) * $(64)) + (r134_z % $(64))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_28_ptr) + ((((r134_z / $(1024)) % $(8)) * $(1024)) + (((r134_z / $(64)) % $(16)) * $(64)) + (r134_z % $(64))), dtype=tl.float32))
    r134_value = r134_value.to(tl.float32)
    r134_value = tl.broadcast_to(r134_value, [$(8192)])
    tl.store($(candidate_vmasked_fill_5_ptr) + r134_z, r134_value)
    r135_z = tl.arange($(8192))
    r135_lane = tl.arange($(64))
    r135_row_values = tl.load($(candidate_vmasked_fill_5_ptr) + ((((r135_z[:, None] / $(1024)) % $(8)) * $(1024)) + (((r135_z[:, None] / $(64)) % $(16)) * $(64)) + (r135_lane[None, :] % $(64))), dtype=tl.float32)
    r135_maximum = tl.max(r135_row_values, axis=1, keep_dims=true)
    r135_weights = tl.exp(r135_row_values - r135_maximum)
    r135_total = tl.sum(r135_weights, axis=1)
    r135_maximum = tl.max(r135_row_values, axis=1)
    r135_value = tl.exp(tl.load($(candidate_vmasked_fill_5_ptr) + ((((r135_z / $(1024)) % $(8)) * $(1024)) + (((r135_z / $(64)) % $(16)) * $(64)) + (r135_z % $(64))), dtype=tl.float32) - r135_maximum) / r135_total
    r135_value = r135_value.to(tl.float32)
    r135_value = tl.broadcast_to(r135_value, [$(8192)])
    tl.store($(candidate_vsoftmax_5_ptr) + r135_z, r135_value)
    r136_z = tl.arange($(8192))
    r136_value = tl.load($(candidate_vsoftmax_5_ptr) + ((((r136_z / $(1024)) % $(8)) * $(1024)) + (((r136_z / $(64)) % $(16)) * $(64)) + (r136_z % $(64))), dtype=tl.float32)
    r136_value = r136_value.to(tl.bfloat16)
    r136_value = tl.broadcast_to(r136_value, [$(8192)])
    tl.store($(candidate_vto_copy_46_ptr) + r136_z, r136_value)
    r137_row = tl.arange($(128))
    r137_col = tl.arange($(1024))
    r137_lane = tl.arange($(256))
    for r137_group in range($(1)) {
      r137_left = tl.load($(candidate_vindex_2_ptr) + ((((((r137_row[:, None] * $(256)) + r137_lane[None, :]) / $(256)) % $(128)) * $(320)) + (((r137_row[:, None] * $(256)) + r137_lane[None, :]) % $(256))), dtype=tl.bfloat16)
      r137_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r137_lane[:, None] * $(1024)) + r137_col[None, :]) / $(128)) % $(8)) * $(65536)) + ((((r137_lane[:, None] * $(1024)) + r137_col[None, :]) % $(128)) * $(256)) + ((((r137_lane[:, None] * $(1024)) + r137_col[None, :]) / $(1024)) % $(256))), dtype=tl.bfloat16)
      r137_value = tl.dot(r137_left, r137_right)
      r137_value = r137_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_18_ptr) + ((r137_group * $(131072)) + (r137_row[:, None] * $(1024)) + r137_col[None, :]), r137_value)
    }
    r138_row = tl.arange($(128))
    r138_col = tl.arange($(1024))
    r138_lane = tl.arange($(256))
    for r138_group in range($(1)) {
      r138_left = tl.load($(candidate_vindex_2_ptr) + ((((((r138_row[:, None] * $(256)) + r138_lane[None, :]) / $(256)) % $(128)) * $(320)) + (((r138_row[:, None] * $(256)) + r138_lane[None, :]) % $(256))), dtype=tl.bfloat16)
      r138_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r138_lane[:, None] * $(1024)) + r138_col[None, :]) / $(128)) % $(8)) * $(65536)) + ((((r138_lane[:, None] * $(1024)) + r138_col[None, :]) % $(128)) * $(256)) + ((((r138_lane[:, None] * $(1024)) + r138_col[None, :]) / $(1024)) % $(256)) + $(32768)), dtype=tl.bfloat16)
      r138_value = tl.dot(r138_left, r138_right)
      r138_value = r138_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_19_ptr) + ((r138_group * $(131072)) + (r138_row[:, None] * $(1024)) + r138_col[None, :]), r138_value)
    }
    r139_z = tl.arange($(49152))
    r139_value = tl.where((r139_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r139_z / $(1536)) % $(32)) * $(1024)) + (((r139_z / $(192)) % $(8)) * $(128)) + ((r139_z % $(192)) - $(0)) + $(65536)) / $(1024)) % $(128)) * $(1536)) + (((((((r139_z / $(1536)) % $(32)) * $(1024)) + (((r139_z / $(192)) % $(8)) * $(128)) + ((r139_z % $(192)) - $(0)) + $(65536)) / $(128)) % $(8)) * $(192)) + (((((r139_z / $(1536)) % $(32)) * $(1024)) + (((r139_z / $(192)) % $(8)) * $(128)) + ((r139_z % $(192)) - $(0)) + $(65536)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vadd_6_ptr) + ((((((((r139_z / $(1536)) % $(32)) * $(512)) + (((r139_z / $(192)) % $(8)) * $(64)) + ((r139_z % $(192)) - $(128)) + $(32768)) / $(512)) % $(128)) * $(512)) + (((((((r139_z / $(1536)) % $(32)) * $(512)) + (((r139_z / $(192)) % $(8)) * $(64)) + ((r139_z % $(192)) - $(128)) + $(32768)) / $(64)) % $(8)) * $(64)) + (((((r139_z / $(1536)) % $(32)) * $(512)) + (((r139_z / $(192)) % $(8)) * $(64)) + ((r139_z % $(192)) - $(128)) + $(32768)) % $(64))), dtype=tl.bfloat16))
    r139_value = r139_value.to(tl.bfloat16)
    r139_value = tl.broadcast_to(r139_value, [$(49152)])
    tl.store($(candidate_vcat_12_ptr) + r139_z, r139_value)
    r140_z = tl.arange($(196608))
    r140_value = tl.where((r140_z % $(192)) < $(128), tl.load($(candidate_vbmm_18_ptr) + ((((((((r140_z / $(1536)) % $(128)) * $(1024)) + (((r140_z / $(192)) % $(8)) * $(128)) + ((r140_z % $(192)) - $(0))) / $(1024)) % $(128)) * $(1024)) + (((((((r140_z / $(1536)) % $(128)) * $(1024)) + (((r140_z / $(192)) % $(8)) * $(128)) + ((r140_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r140_z / $(1536)) % $(128)) * $(1024)) + (((r140_z / $(192)) % $(8)) * $(128)) + ((r140_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vindex_2_ptr) + ((((r140_z / $(1536)) % $(128)) * $(320)) + ((((r140_z % $(192)) - $(128)) + $(256)) % $(320))), dtype=tl.bfloat16))
    r140_value = r140_value.to(tl.bfloat16)
    r140_value = tl.broadcast_to(r140_value, [$(196608)])
    tl.store($(candidate_vcat_13_ptr) + r140_z, r140_value)
    r141_z = tl.arange($(18432))
    r141_value = tl.load($(candidate_vcat_12_ptr) + ((((r141_z / $(192)) % $(12)) * $(1536)) + (((r141_z / $(2304)) % $(8)) * $(192)) + (r141_z % $(192))), dtype=tl.bfloat16)
    r141_value = r141_value.to(tl.float32)
    r141_value = tl.broadcast_to(r141_value, [$(18432)])
    tl.store($(candidate_vto_copy_47_ptr) + r141_z, r141_value)
    r142_z = tl.arange($(36864))
    r142_value = tl.load($(candidate_vcat_13_ptr) + ((((r142_z / $(192)) % $(24)) * $(1536)) + (((r142_z / $(4608)) % $(8)) * $(192)) + (r142_z % $(192))), dtype=tl.bfloat16)
    r142_value = r142_value.to(tl.float32)
    r142_value = tl.broadcast_to(r142_value, [$(36864)])
    tl.store($(candidate_vto_copy_48_ptr) + r142_z, r142_value)
    r143_row = tl.arange($(12))
    r143_col = tl.arange($(24))
    r143_lane = tl.arange($(192))
    for r143_group in range($(8)) {
      r143_left = tl.load($(candidate_vto_copy_47_ptr) + (((((((r143_group % $(8)) * $(2304)) + (r143_row[:, None] * $(192)) + r143_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r143_group % $(8)) * $(2304)) + (r143_row[:, None] * $(192)) + r143_lane[None, :]) / $(192)) % $(12)) * $(192)) + ((((r143_group % $(8)) * $(2304)) + (r143_row[:, None] * $(192)) + r143_lane[None, :]) % $(192))), dtype=tl.float32)
      r143_right = tl.load($(candidate_vto_copy_48_ptr) + (((((((r143_group % $(8)) * $(4608)) + (r143_lane[:, None] * $(24)) + r143_col[None, :]) / $(4608)) % $(8)) * $(4608)) + (((((r143_group % $(8)) * $(4608)) + (r143_lane[:, None] * $(24)) + r143_col[None, :]) % $(24)) * $(192)) + (((((r143_group % $(8)) * $(4608)) + (r143_lane[:, None] * $(24)) + r143_col[None, :]) / $(24)) % $(192))), dtype=tl.float32)
      r143_value = tl.dot(r143_left, r143_right)
      r143_value = r143_value.to(tl.float32)
      tl.store($(candidate_vbmm_20_ptr) + ((r143_group * $(288)) + (r143_row[:, None] * $(24)) + r143_col[None, :]), r143_value)
    }
    r144_z = tl.arange($(2304))
    r144_value = (tl.load($(candidate_vbmm_20_ptr) + ((((r144_z / $(288)) % $(8)) * $(288)) + (((r144_z / $(24)) % $(12)) * $(24)) + (r144_z % $(24))), dtype=tl.float32) * 0.07216878364870322)
    r144_value = r144_value.to(tl.float32)
    r144_value = tl.broadcast_to(r144_value, [$(2304)])
    tl.store($(candidate_vmul_29_ptr) + r144_z, r144_value)
    r145_z = tl.arange($(12))
    r145_value = (tl.cast(r145_z, tl.int64) + tl.full([], $(24), dtype=tl.int64))
    r145_value = tl.broadcast_to(r145_value, [$(12)])
    tl.store($(candidate_vadd_14_ptr) + r145_z, r145_value)
    r146_z = tl.arange($(12))
    r146_value = (tl.load($(candidate_vadd_14_ptr) + r146_z, dtype=tl.int64) - tl.full([], $(12), dtype=tl.int64))
    r146_value = tl.broadcast_to(r146_value, [$(12)])
    tl.store($(candidate_vsub_15_ptr) + r146_z, r146_value)
    r147_z = tl.arange($(288))
    r147_value = (tl.cast((r147_z % $(24)), tl.int64) > tl.load($(candidate_vsub_15_ptr) + ((r147_z / $(24)) % $(12)), dtype=tl.int64))
    r147_value = tl.broadcast_to(r147_value, [$(288)])
    tl.store($(candidate_vgt_6_ptr) + r147_z, r147_value)
    r148_z = tl.arange($(2304))
    r148_value = tl.where(tl.load($(candidate_vgt_6_ptr) + ((((r148_z / $(24)) % $(12)) * $(24)) + (r148_z % $(24))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_29_ptr) + ((((r148_z / $(288)) % $(8)) * $(288)) + (((r148_z / $(24)) % $(12)) * $(24)) + (r148_z % $(24))), dtype=tl.float32))
    r148_value = r148_value.to(tl.float32)
    r148_value = tl.broadcast_to(r148_value, [$(2304)])
    tl.store($(candidate_vmasked_fill_6_ptr) + r148_z, r148_value)
    r149_z = tl.arange($(2304))
    r149_lane = tl.arange($(24))
    r149_row_values = tl.load($(candidate_vmasked_fill_6_ptr) + ((((r149_z[:, None] / $(288)) % $(8)) * $(288)) + (((r149_z[:, None] / $(24)) % $(12)) * $(24)) + (r149_lane[None, :] % $(24))), dtype=tl.float32)
    r149_maximum = tl.max(r149_row_values, axis=1, keep_dims=true)
    r149_weights = tl.exp(r149_row_values - r149_maximum)
    r149_total = tl.sum(r149_weights, axis=1)
    r149_maximum = tl.max(r149_row_values, axis=1)
    r149_value = tl.exp(tl.load($(candidate_vmasked_fill_6_ptr) + ((((r149_z / $(288)) % $(8)) * $(288)) + (((r149_z / $(24)) % $(12)) * $(24)) + (r149_z % $(24))), dtype=tl.float32) - r149_maximum) / r149_total
    r149_value = r149_value.to(tl.float32)
    r149_value = tl.broadcast_to(r149_value, [$(2304)])
    tl.store($(candidate_vsoftmax_6_ptr) + r149_z, r149_value)
    r150_z = tl.arange($(2304))
    r150_value = tl.load($(candidate_vsoftmax_6_ptr) + ((((r150_z / $(288)) % $(8)) * $(288)) + (((r150_z / $(24)) % $(12)) * $(24)) + (r150_z % $(24))), dtype=tl.float32)
    r150_value = r150_value.to(tl.bfloat16)
    r150_value = tl.broadcast_to(r150_value, [$(2304)])
    tl.store($(candidate_vto_copy_49_ptr) + r150_z, r150_value)
    r151_z = tl.arange($(6144))
    r151_value = tl.load($(candidate_vcat_12_ptr) + ((((r151_z / $(192)) % $(4)) * $(1536)) + (((r151_z / $(768)) % $(8)) * $(192)) + (r151_z % $(192)) + $(18432)), dtype=tl.bfloat16)
    r151_value = r151_value.to(tl.float32)
    r151_value = tl.broadcast_to(r151_value, [$(6144)])
    tl.store($(candidate_vto_copy_50_ptr) + r151_z, r151_value)
    r152_z = tl.arange($(12288))
    r152_value = tl.load($(candidate_vcat_13_ptr) + ((((r152_z / $(192)) % $(8)) * $(1536)) + (((r152_z / $(1536)) % $(8)) * $(192)) + (r152_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r152_value = r152_value.to(tl.float32)
    r152_value = tl.broadcast_to(r152_value, [$(12288)])
    tl.store($(candidate_vto_copy_51_ptr) + r152_z, r152_value)
    r153_row = tl.arange($(4))
    r153_col = tl.arange($(8))
    r153_lane = tl.arange($(192))
    for r153_group in range($(8)) {
      r153_left = tl.load($(candidate_vto_copy_50_ptr) + (((((((r153_group % $(8)) * $(768)) + (r153_row[:, None] * $(192)) + r153_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r153_group % $(8)) * $(768)) + (r153_row[:, None] * $(192)) + r153_lane[None, :]) / $(192)) % $(4)) * $(192)) + ((((r153_group % $(8)) * $(768)) + (r153_row[:, None] * $(192)) + r153_lane[None, :]) % $(192))), dtype=tl.float32)
      r153_right = tl.load($(candidate_vto_copy_51_ptr) + (((((((r153_group % $(8)) * $(1536)) + (r153_lane[:, None] * $(8)) + r153_col[None, :]) / $(1536)) % $(8)) * $(1536)) + (((((r153_group % $(8)) * $(1536)) + (r153_lane[:, None] * $(8)) + r153_col[None, :]) % $(8)) * $(192)) + (((((r153_group % $(8)) * $(1536)) + (r153_lane[:, None] * $(8)) + r153_col[None, :]) / $(8)) % $(192))), dtype=tl.float32)
      r153_value = tl.dot(r153_left, r153_right)
      r153_value = r153_value.to(tl.float32)
      tl.store($(candidate_vbmm_22_ptr) + ((r153_group * $(32)) + (r153_row[:, None] * $(8)) + r153_col[None, :]), r153_value)
    }
    r154_z = tl.arange($(256))
    r154_value = (tl.load($(candidate_vbmm_22_ptr) + ((((r154_z / $(32)) % $(8)) * $(32)) + (((r154_z / $(8)) % $(4)) * $(8)) + (r154_z % $(8))), dtype=tl.float32) * 0.07216878364870322)
    r154_value = r154_value.to(tl.float32)
    r154_value = tl.broadcast_to(r154_value, [$(256)])
    tl.store($(candidate_vmul_30_ptr) + r154_z, r154_value)
    r155_z = tl.arange($(4))
    r155_value = (tl.cast(r155_z, tl.int64) + tl.full([], $(8), dtype=tl.int64))
    r155_value = tl.broadcast_to(r155_value, [$(4)])
    tl.store($(candidate_vadd_15_ptr) + r155_z, r155_value)
    r156_z = tl.arange($(4))
    r156_value = (tl.load($(candidate_vadd_15_ptr) + r156_z, dtype=tl.int64) - tl.full([], $(4), dtype=tl.int64))
    r156_value = tl.broadcast_to(r156_value, [$(4)])
    tl.store($(candidate_vsub_16_ptr) + r156_z, r156_value)
    r157_z = tl.arange($(32))
    r157_value = (tl.cast((r157_z % $(8)), tl.int64) > tl.load($(candidate_vsub_16_ptr) + ((r157_z / $(8)) % $(4)), dtype=tl.int64))
    r157_value = tl.broadcast_to(r157_value, [$(32)])
    tl.store($(candidate_vgt_7_ptr) + r157_z, r157_value)
    r158_z = tl.arange($(256))
    r158_value = tl.where(tl.load($(candidate_vgt_7_ptr) + ((((r158_z / $(8)) % $(4)) * $(8)) + (r158_z % $(8))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_30_ptr) + ((((r158_z / $(32)) % $(8)) * $(32)) + (((r158_z / $(8)) % $(4)) * $(8)) + (r158_z % $(8))), dtype=tl.float32))
    r158_value = r158_value.to(tl.float32)
    r158_value = tl.broadcast_to(r158_value, [$(256)])
    tl.store($(candidate_vmasked_fill_7_ptr) + r158_z, r158_value)
    r159_z = tl.arange($(256))
    r159_lane = tl.arange($(8))
    r159_row_values = tl.load($(candidate_vmasked_fill_7_ptr) + ((((r159_z[:, None] / $(32)) % $(8)) * $(32)) + (((r159_z[:, None] / $(8)) % $(4)) * $(8)) + (r159_lane[None, :] % $(8))), dtype=tl.float32)
    r159_maximum = tl.max(r159_row_values, axis=1, keep_dims=true)
    r159_weights = tl.exp(r159_row_values - r159_maximum)
    r159_total = tl.sum(r159_weights, axis=1)
    r159_maximum = tl.max(r159_row_values, axis=1)
    r159_value = tl.exp(tl.load($(candidate_vmasked_fill_7_ptr) + ((((r159_z / $(32)) % $(8)) * $(32)) + (((r159_z / $(8)) % $(4)) * $(8)) + (r159_z % $(8))), dtype=tl.float32) - r159_maximum) / r159_total
    r159_value = r159_value.to(tl.float32)
    r159_value = tl.broadcast_to(r159_value, [$(256)])
    tl.store($(candidate_vsoftmax_7_ptr) + r159_z, r159_value)
    r160_z = tl.arange($(256))
    r160_value = tl.load($(candidate_vsoftmax_7_ptr) + ((((r160_z / $(32)) % $(8)) * $(32)) + (((r160_z / $(8)) % $(4)) * $(8)) + (r160_z % $(8))), dtype=tl.float32)
    r160_value = r160_value.to(tl.bfloat16)
    r160_value = tl.broadcast_to(r160_value, [$(256)])
    tl.store($(candidate_vto_copy_52_ptr) + r160_z, r160_value)
    r161_z = tl.arange($(24576))
    r161_value = tl.load($(candidate_vcat_12_ptr) + ((((r161_z / $(192)) % $(16)) * $(1536)) + (((r161_z / $(3072)) % $(8)) * $(192)) + (r161_z % $(192)) + $(24576)), dtype=tl.bfloat16)
    r161_value = r161_value.to(tl.float32)
    r161_value = tl.broadcast_to(r161_value, [$(24576)])
    tl.store($(candidate_vto_copy_53_ptr) + r161_z, r161_value)
    r162_z = tl.arange($(49152))
    r162_value = tl.load($(candidate_vcat_13_ptr) + ((((r162_z / $(192)) % $(32)) * $(1536)) + (((r162_z / $(6144)) % $(8)) * $(192)) + (r162_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r162_value = r162_value.to(tl.float32)
    r162_value = tl.broadcast_to(r162_value, [$(49152)])
    tl.store($(candidate_vto_copy_54_ptr) + r162_z, r162_value)
    r163_row = tl.arange($(16))
    r163_col = tl.arange($(32))
    r163_lane = tl.arange($(192))
    for r163_group in range($(8)) {
      r163_left = tl.load($(candidate_vto_copy_53_ptr) + (((((((r163_group % $(8)) * $(3072)) + (r163_row[:, None] * $(192)) + r163_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r163_group % $(8)) * $(3072)) + (r163_row[:, None] * $(192)) + r163_lane[None, :]) / $(192)) % $(16)) * $(192)) + ((((r163_group % $(8)) * $(3072)) + (r163_row[:, None] * $(192)) + r163_lane[None, :]) % $(192))), dtype=tl.float32)
      r163_right = tl.load($(candidate_vto_copy_54_ptr) + (((((((r163_group % $(8)) * $(6144)) + (r163_lane[:, None] * $(32)) + r163_col[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r163_group % $(8)) * $(6144)) + (r163_lane[:, None] * $(32)) + r163_col[None, :]) % $(32)) * $(192)) + (((((r163_group % $(8)) * $(6144)) + (r163_lane[:, None] * $(32)) + r163_col[None, :]) / $(32)) % $(192))), dtype=tl.float32)
      r163_value = tl.dot(r163_left, r163_right)
      r163_value = r163_value.to(tl.float32)
      tl.store($(candidate_vbmm_24_ptr) + ((r163_group * $(512)) + (r163_row[:, None] * $(32)) + r163_col[None, :]), r163_value)
    }
    r164_z = tl.arange($(4096))
    r164_value = (tl.load($(candidate_vbmm_24_ptr) + ((((r164_z / $(512)) % $(8)) * $(512)) + (((r164_z / $(32)) % $(16)) * $(32)) + (r164_z % $(32))), dtype=tl.float32) * 0.07216878364870322)
    r164_value = r164_value.to(tl.float32)
    r164_value = tl.broadcast_to(r164_value, [$(4096)])
    tl.store($(candidate_vmul_31_ptr) + r164_z, r164_value)
    r165_z = tl.arange($(16))
    r165_value = (tl.cast(r165_z, tl.int64) + tl.full([], $(32), dtype=tl.int64))
    r165_value = tl.broadcast_to(r165_value, [$(16)])
    tl.store($(candidate_vadd_16_ptr) + r165_z, r165_value)
    r166_z = tl.arange($(16))
    r166_value = (tl.load($(candidate_vadd_16_ptr) + r166_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r166_value = tl.broadcast_to(r166_value, [$(16)])
    tl.store($(candidate_vsub_17_ptr) + r166_z, r166_value)
    r167_z = tl.arange($(512))
    r167_value = (tl.cast((r167_z % $(32)), tl.int64) > tl.load($(candidate_vsub_17_ptr) + ((r167_z / $(32)) % $(16)), dtype=tl.int64))
    r167_value = tl.broadcast_to(r167_value, [$(512)])
    tl.store($(candidate_vgt_8_ptr) + r167_z, r167_value)
    r168_z = tl.arange($(4096))
    r168_value = tl.where(tl.load($(candidate_vgt_8_ptr) + ((((r168_z / $(32)) % $(16)) * $(32)) + (r168_z % $(32))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_31_ptr) + ((((r168_z / $(512)) % $(8)) * $(512)) + (((r168_z / $(32)) % $(16)) * $(32)) + (r168_z % $(32))), dtype=tl.float32))
    r168_value = r168_value.to(tl.float32)
    r168_value = tl.broadcast_to(r168_value, [$(4096)])
    tl.store($(candidate_vmasked_fill_8_ptr) + r168_z, r168_value)
    r169_z = tl.arange($(4096))
    r169_lane = tl.arange($(32))
    r169_row_values = tl.load($(candidate_vmasked_fill_8_ptr) + ((((r169_z[:, None] / $(512)) % $(8)) * $(512)) + (((r169_z[:, None] / $(32)) % $(16)) * $(32)) + (r169_lane[None, :] % $(32))), dtype=tl.float32)
    r169_maximum = tl.max(r169_row_values, axis=1, keep_dims=true)
    r169_weights = tl.exp(r169_row_values - r169_maximum)
    r169_total = tl.sum(r169_weights, axis=1)
    r169_maximum = tl.max(r169_row_values, axis=1)
    r169_value = tl.exp(tl.load($(candidate_vmasked_fill_8_ptr) + ((((r169_z / $(512)) % $(8)) * $(512)) + (((r169_z / $(32)) % $(16)) * $(32)) + (r169_z % $(32))), dtype=tl.float32) - r169_maximum) / r169_total
    r169_value = r169_value.to(tl.float32)
    r169_value = tl.broadcast_to(r169_value, [$(4096)])
    tl.store($(candidate_vsoftmax_8_ptr) + r169_z, r169_value)
    r170_z = tl.arange($(4096))
    r170_value = tl.load($(candidate_vsoftmax_8_ptr) + ((((r170_z / $(512)) % $(8)) * $(512)) + (((r170_z / $(32)) % $(16)) * $(32)) + (r170_z % $(32))), dtype=tl.float32)
    r170_value = r170_value.to(tl.bfloat16)
    r170_value = tl.broadcast_to(r170_value, [$(4096)])
    tl.store($(candidate_vto_copy_55_ptr) + r170_z, r170_value)
    r171_row = tl.arange($(128))
    r171_col = tl.arange($(1024))
    r171_lane = tl.arange($(256))
    for r171_group in range($(1)) {
      r171_left = tl.load($(candidate_vindex_2_ptr) + ((((((r171_row[:, None] * $(256)) + r171_lane[None, :]) / $(256)) % $(128)) * $(320)) + (((r171_row[:, None] * $(256)) + r171_lane[None, :]) % $(256))), dtype=tl.bfloat16)
      r171_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r171_lane[:, None] * $(1024)) + r171_col[None, :]) / $(128)) % $(8)) * $(65536)) + ((((r171_lane[:, None] * $(1024)) + r171_col[None, :]) % $(128)) * $(256)) + ((((r171_lane[:, None] * $(1024)) + r171_col[None, :]) / $(1024)) % $(256))), dtype=tl.bfloat16)
      r171_value = tl.dot(r171_left, r171_right)
      r171_value = r171_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_26_ptr) + ((r171_group * $(131072)) + (r171_row[:, None] * $(1024)) + r171_col[None, :]), r171_value)
    }
    r172_row = tl.arange($(128))
    r172_col = tl.arange($(1024))
    r172_lane = tl.arange($(256))
    for r172_group in range($(1)) {
      r172_left = tl.load($(candidate_vindex_2_ptr) + ((((((r172_row[:, None] * $(256)) + r172_lane[None, :]) / $(256)) % $(128)) * $(320)) + (((r172_row[:, None] * $(256)) + r172_lane[None, :]) % $(256))), dtype=tl.bfloat16)
      r172_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r172_lane[:, None] * $(1024)) + r172_col[None, :]) / $(128)) % $(8)) * $(65536)) + ((((r172_lane[:, None] * $(1024)) + r172_col[None, :]) % $(128)) * $(256)) + ((((r172_lane[:, None] * $(1024)) + r172_col[None, :]) / $(1024)) % $(256)) + $(32768)), dtype=tl.bfloat16)
      r172_value = tl.dot(r172_left, r172_right)
      r172_value = r172_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_27_ptr) + ((r172_group * $(131072)) + (r172_row[:, None] * $(1024)) + r172_col[None, :]), r172_value)
    }
    r173_z = tl.arange($(49152))
    r173_value = tl.where((r173_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r173_z / $(1536)) % $(32)) * $(1024)) + (((r173_z / $(192)) % $(8)) * $(128)) + ((r173_z % $(192)) - $(0)) + $(98304)) / $(1024)) % $(128)) * $(1536)) + (((((((r173_z / $(1536)) % $(32)) * $(1024)) + (((r173_z / $(192)) % $(8)) * $(128)) + ((r173_z % $(192)) - $(0)) + $(98304)) / $(128)) % $(8)) * $(192)) + (((((r173_z / $(1536)) % $(32)) * $(1024)) + (((r173_z / $(192)) % $(8)) * $(128)) + ((r173_z % $(192)) - $(0)) + $(98304)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vadd_6_ptr) + ((((((((r173_z / $(1536)) % $(32)) * $(512)) + (((r173_z / $(192)) % $(8)) * $(64)) + ((r173_z % $(192)) - $(128)) + $(49152)) / $(512)) % $(128)) * $(512)) + (((((((r173_z / $(1536)) % $(32)) * $(512)) + (((r173_z / $(192)) % $(8)) * $(64)) + ((r173_z % $(192)) - $(128)) + $(49152)) / $(64)) % $(8)) * $(64)) + (((((r173_z / $(1536)) % $(32)) * $(512)) + (((r173_z / $(192)) % $(8)) * $(64)) + ((r173_z % $(192)) - $(128)) + $(49152)) % $(64))), dtype=tl.bfloat16))
    r173_value = r173_value.to(tl.bfloat16)
    r173_value = tl.broadcast_to(r173_value, [$(49152)])
    tl.store($(candidate_vcat_15_ptr) + r173_z, r173_value)
    r174_z = tl.arange($(196608))
    r174_value = tl.where((r174_z % $(192)) < $(128), tl.load($(candidate_vbmm_26_ptr) + ((((((((r174_z / $(1536)) % $(128)) * $(1024)) + (((r174_z / $(192)) % $(8)) * $(128)) + ((r174_z % $(192)) - $(0))) / $(1024)) % $(128)) * $(1024)) + (((((((r174_z / $(1536)) % $(128)) * $(1024)) + (((r174_z / $(192)) % $(8)) * $(128)) + ((r174_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r174_z / $(1536)) % $(128)) * $(1024)) + (((r174_z / $(192)) % $(8)) * $(128)) + ((r174_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vindex_2_ptr) + ((((r174_z / $(1536)) % $(128)) * $(320)) + ((((r174_z % $(192)) - $(128)) + $(256)) % $(320))), dtype=tl.bfloat16))
    r174_value = r174_value.to(tl.bfloat16)
    r174_value = tl.broadcast_to(r174_value, [$(196608)])
    tl.store($(candidate_vcat_16_ptr) + r174_z, r174_value)
    r175_z = tl.arange($(18432))
    r175_value = tl.load($(candidate_vcat_15_ptr) + ((((r175_z / $(192)) % $(12)) * $(1536)) + (((r175_z / $(2304)) % $(8)) * $(192)) + (r175_z % $(192))), dtype=tl.bfloat16)
    r175_value = r175_value.to(tl.float32)
    r175_value = tl.broadcast_to(r175_value, [$(18432)])
    tl.store($(candidate_vto_copy_56_ptr) + r175_z, r175_value)
    r176_z = tl.arange($(55296))
    r176_value = tl.load($(candidate_vcat_16_ptr) + ((((r176_z / $(192)) % $(36)) * $(1536)) + (((r176_z / $(6912)) % $(8)) * $(192)) + (r176_z % $(192))), dtype=tl.bfloat16)
    r176_value = r176_value.to(tl.float32)
    r176_value = tl.broadcast_to(r176_value, [$(55296)])
    tl.store($(candidate_vto_copy_57_ptr) + r176_z, r176_value)
    r177_row = tl.arange($(12))
    r177_col = tl.arange($(36))
    r177_lane = tl.arange($(192))
    for r177_group in range($(8)) {
      r177_left = tl.load($(candidate_vto_copy_56_ptr) + (((((((r177_group % $(8)) * $(2304)) + (r177_row[:, None] * $(192)) + r177_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r177_group % $(8)) * $(2304)) + (r177_row[:, None] * $(192)) + r177_lane[None, :]) / $(192)) % $(12)) * $(192)) + ((((r177_group % $(8)) * $(2304)) + (r177_row[:, None] * $(192)) + r177_lane[None, :]) % $(192))), dtype=tl.float32)
      r177_right = tl.load($(candidate_vto_copy_57_ptr) + (((((((r177_group % $(8)) * $(6912)) + (r177_lane[:, None] * $(36)) + r177_col[None, :]) / $(6912)) % $(8)) * $(6912)) + (((((r177_group % $(8)) * $(6912)) + (r177_lane[:, None] * $(36)) + r177_col[None, :]) % $(36)) * $(192)) + (((((r177_group % $(8)) * $(6912)) + (r177_lane[:, None] * $(36)) + r177_col[None, :]) / $(36)) % $(192))), dtype=tl.float32)
      r177_value = tl.dot(r177_left, r177_right)
      r177_value = r177_value.to(tl.float32)
      tl.store($(candidate_vbmm_28_ptr) + ((r177_group * $(432)) + (r177_row[:, None] * $(36)) + r177_col[None, :]), r177_value)
    }
    r178_z = tl.arange($(3456))
    r178_value = (tl.load($(candidate_vbmm_28_ptr) + ((((r178_z / $(432)) % $(8)) * $(432)) + (((r178_z / $(36)) % $(12)) * $(36)) + (r178_z % $(36))), dtype=tl.float32) * 0.07216878364870322)
    r178_value = r178_value.to(tl.float32)
    r178_value = tl.broadcast_to(r178_value, [$(3456)])
    tl.store($(candidate_vmul_32_ptr) + r178_z, r178_value)
    r179_z = tl.arange($(12))
    r179_value = (tl.cast(r179_z, tl.int64) + tl.full([], $(36), dtype=tl.int64))
    r179_value = tl.broadcast_to(r179_value, [$(12)])
    tl.store($(candidate_vadd_17_ptr) + r179_z, r179_value)
    r180_z = tl.arange($(12))
    r180_value = (tl.load($(candidate_vadd_17_ptr) + r180_z, dtype=tl.int64) - tl.full([], $(12), dtype=tl.int64))
    r180_value = tl.broadcast_to(r180_value, [$(12)])
    tl.store($(candidate_vsub_18_ptr) + r180_z, r180_value)
    r181_z = tl.arange($(432))
    r181_value = (tl.cast((r181_z % $(36)), tl.int64) > tl.load($(candidate_vsub_18_ptr) + ((r181_z / $(36)) % $(12)), dtype=tl.int64))
    r181_value = tl.broadcast_to(r181_value, [$(432)])
    tl.store($(candidate_vgt_9_ptr) + r181_z, r181_value)
    r182_z = tl.arange($(3456))
    r182_value = tl.where(tl.load($(candidate_vgt_9_ptr) + ((((r182_z / $(36)) % $(12)) * $(36)) + (r182_z % $(36))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_32_ptr) + ((((r182_z / $(432)) % $(8)) * $(432)) + (((r182_z / $(36)) % $(12)) * $(36)) + (r182_z % $(36))), dtype=tl.float32))
    r182_value = r182_value.to(tl.float32)
    r182_value = tl.broadcast_to(r182_value, [$(3456)])
    tl.store($(candidate_vmasked_fill_9_ptr) + r182_z, r182_value)
    r183_z = tl.arange($(3456))
    r183_lane = tl.arange($(36))
    r183_row_values = tl.load($(candidate_vmasked_fill_9_ptr) + ((((r183_z[:, None] / $(432)) % $(8)) * $(432)) + (((r183_z[:, None] / $(36)) % $(12)) * $(36)) + (r183_lane[None, :] % $(36))), dtype=tl.float32)
    r183_maximum = tl.max(r183_row_values, axis=1, keep_dims=true)
    r183_weights = tl.exp(r183_row_values - r183_maximum)
    r183_total = tl.sum(r183_weights, axis=1)
    r183_maximum = tl.max(r183_row_values, axis=1)
    r183_value = tl.exp(tl.load($(candidate_vmasked_fill_9_ptr) + ((((r183_z / $(432)) % $(8)) * $(432)) + (((r183_z / $(36)) % $(12)) * $(36)) + (r183_z % $(36))), dtype=tl.float32) - r183_maximum) / r183_total
    r183_value = r183_value.to(tl.float32)
    r183_value = tl.broadcast_to(r183_value, [$(3456)])
    tl.store($(candidate_vsoftmax_9_ptr) + r183_z, r183_value)
    r184_z = tl.arange($(3456))
    r184_value = tl.load($(candidate_vsoftmax_9_ptr) + ((((r184_z / $(432)) % $(8)) * $(432)) + (((r184_z / $(36)) % $(12)) * $(36)) + (r184_z % $(36))), dtype=tl.float32)
    r184_value = r184_value.to(tl.bfloat16)
    r184_value = tl.broadcast_to(r184_value, [$(3456)])
    tl.store($(candidate_vto_copy_58_ptr) + r184_z, r184_value)
    r185_z = tl.arange($(6144))
    r185_value = tl.load($(candidate_vcat_15_ptr) + ((((r185_z / $(192)) % $(4)) * $(1536)) + (((r185_z / $(768)) % $(8)) * $(192)) + (r185_z % $(192)) + $(18432)), dtype=tl.bfloat16)
    r185_value = r185_value.to(tl.float32)
    r185_value = tl.broadcast_to(r185_value, [$(6144)])
    tl.store($(candidate_vto_copy_59_ptr) + r185_z, r185_value)
    r186_z = tl.arange($(18432))
    r186_value = tl.load($(candidate_vcat_16_ptr) + ((((r186_z / $(192)) % $(12)) * $(1536)) + (((r186_z / $(2304)) % $(8)) * $(192)) + (r186_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r186_value = r186_value.to(tl.float32)
    r186_value = tl.broadcast_to(r186_value, [$(18432)])
    tl.store($(candidate_vto_copy_60_ptr) + r186_z, r186_value)
    r187_row = tl.arange($(4))
    r187_col = tl.arange($(12))
    r187_lane = tl.arange($(192))
    for r187_group in range($(8)) {
      r187_left = tl.load($(candidate_vto_copy_59_ptr) + (((((((r187_group % $(8)) * $(768)) + (r187_row[:, None] * $(192)) + r187_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r187_group % $(8)) * $(768)) + (r187_row[:, None] * $(192)) + r187_lane[None, :]) / $(192)) % $(4)) * $(192)) + ((((r187_group % $(8)) * $(768)) + (r187_row[:, None] * $(192)) + r187_lane[None, :]) % $(192))), dtype=tl.float32)
      r187_right = tl.load($(candidate_vto_copy_60_ptr) + (((((((r187_group % $(8)) * $(2304)) + (r187_lane[:, None] * $(12)) + r187_col[None, :]) / $(2304)) % $(8)) * $(2304)) + (((((r187_group % $(8)) * $(2304)) + (r187_lane[:, None] * $(12)) + r187_col[None, :]) % $(12)) * $(192)) + (((((r187_group % $(8)) * $(2304)) + (r187_lane[:, None] * $(12)) + r187_col[None, :]) / $(12)) % $(192))), dtype=tl.float32)
      r187_value = tl.dot(r187_left, r187_right)
      r187_value = r187_value.to(tl.float32)
      tl.store($(candidate_vbmm_30_ptr) + ((r187_group * $(48)) + (r187_row[:, None] * $(12)) + r187_col[None, :]), r187_value)
    }
    r188_z = tl.arange($(384))
    r188_value = (tl.load($(candidate_vbmm_30_ptr) + ((((r188_z / $(48)) % $(8)) * $(48)) + (((r188_z / $(12)) % $(4)) * $(12)) + (r188_z % $(12))), dtype=tl.float32) * 0.07216878364870322)
    r188_value = r188_value.to(tl.float32)
    r188_value = tl.broadcast_to(r188_value, [$(384)])
    tl.store($(candidate_vmul_33_ptr) + r188_z, r188_value)
    r189_z = tl.arange($(4))
    r189_value = (tl.cast(r189_z, tl.int64) + tl.full([], $(12), dtype=tl.int64))
    r189_value = tl.broadcast_to(r189_value, [$(4)])
    tl.store($(candidate_vadd_18_ptr) + r189_z, r189_value)
    r190_z = tl.arange($(4))
    r190_value = (tl.load($(candidate_vadd_18_ptr) + r190_z, dtype=tl.int64) - tl.full([], $(4), dtype=tl.int64))
    r190_value = tl.broadcast_to(r190_value, [$(4)])
    tl.store($(candidate_vsub_19_ptr) + r190_z, r190_value)
    r191_z = tl.arange($(48))
    r191_value = (tl.cast((r191_z % $(12)), tl.int64) > tl.load($(candidate_vsub_19_ptr) + ((r191_z / $(12)) % $(4)), dtype=tl.int64))
    r191_value = tl.broadcast_to(r191_value, [$(48)])
    tl.store($(candidate_vgt_10_ptr) + r191_z, r191_value)
    r192_z = tl.arange($(384))
    r192_value = tl.where(tl.load($(candidate_vgt_10_ptr) + ((((r192_z / $(12)) % $(4)) * $(12)) + (r192_z % $(12))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_33_ptr) + ((((r192_z / $(48)) % $(8)) * $(48)) + (((r192_z / $(12)) % $(4)) * $(12)) + (r192_z % $(12))), dtype=tl.float32))
    r192_value = r192_value.to(tl.float32)
    r192_value = tl.broadcast_to(r192_value, [$(384)])
    tl.store($(candidate_vmasked_fill_10_ptr) + r192_z, r192_value)
    r193_z = tl.arange($(384))
    r193_lane = tl.arange($(12))
    r193_row_values = tl.load($(candidate_vmasked_fill_10_ptr) + ((((r193_z[:, None] / $(48)) % $(8)) * $(48)) + (((r193_z[:, None] / $(12)) % $(4)) * $(12)) + (r193_lane[None, :] % $(12))), dtype=tl.float32)
    r193_maximum = tl.max(r193_row_values, axis=1, keep_dims=true)
    r193_weights = tl.exp(r193_row_values - r193_maximum)
    r193_total = tl.sum(r193_weights, axis=1)
    r193_maximum = tl.max(r193_row_values, axis=1)
    r193_value = tl.exp(tl.load($(candidate_vmasked_fill_10_ptr) + ((((r193_z / $(48)) % $(8)) * $(48)) + (((r193_z / $(12)) % $(4)) * $(12)) + (r193_z % $(12))), dtype=tl.float32) - r193_maximum) / r193_total
    r193_value = r193_value.to(tl.float32)
    r193_value = tl.broadcast_to(r193_value, [$(384)])
    tl.store($(candidate_vsoftmax_10_ptr) + r193_z, r193_value)
    r194_z = tl.arange($(384))
    r194_value = tl.load($(candidate_vsoftmax_10_ptr) + ((((r194_z / $(48)) % $(8)) * $(48)) + (((r194_z / $(12)) % $(4)) * $(12)) + (r194_z % $(12))), dtype=tl.float32)
    r194_value = r194_value.to(tl.bfloat16)
    r194_value = tl.broadcast_to(r194_value, [$(384)])
    tl.store($(candidate_vto_copy_61_ptr) + r194_z, r194_value)
    r195_z = tl.arange($(24576))
    r195_value = tl.load($(candidate_vcat_15_ptr) + ((((r195_z / $(192)) % $(16)) * $(1536)) + (((r195_z / $(3072)) % $(8)) * $(192)) + (r195_z % $(192)) + $(24576)), dtype=tl.bfloat16)
    r195_value = r195_value.to(tl.float32)
    r195_value = tl.broadcast_to(r195_value, [$(24576)])
    tl.store($(candidate_vto_copy_62_ptr) + r195_z, r195_value)
    r196_z = tl.arange($(73728))
    r196_value = tl.load($(candidate_vcat_16_ptr) + ((((r196_z / $(192)) % $(48)) * $(1536)) + (((r196_z / $(9216)) % $(8)) * $(192)) + (r196_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r196_value = r196_value.to(tl.float32)
    r196_value = tl.broadcast_to(r196_value, [$(73728)])
    tl.store($(candidate_vto_copy_63_ptr) + r196_z, r196_value)
    r197_row = tl.arange($(16))
    r197_col = tl.arange($(48))
    r197_lane = tl.arange($(192))
    for r197_group in range($(8)) {
      r197_left = tl.load($(candidate_vto_copy_62_ptr) + (((((((r197_group % $(8)) * $(3072)) + (r197_row[:, None] * $(192)) + r197_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r197_group % $(8)) * $(3072)) + (r197_row[:, None] * $(192)) + r197_lane[None, :]) / $(192)) % $(16)) * $(192)) + ((((r197_group % $(8)) * $(3072)) + (r197_row[:, None] * $(192)) + r197_lane[None, :]) % $(192))), dtype=tl.float32)
      r197_right = tl.load($(candidate_vto_copy_63_ptr) + (((((((r197_group % $(8)) * $(9216)) + (r197_lane[:, None] * $(48)) + r197_col[None, :]) / $(9216)) % $(8)) * $(9216)) + (((((r197_group % $(8)) * $(9216)) + (r197_lane[:, None] * $(48)) + r197_col[None, :]) % $(48)) * $(192)) + (((((r197_group % $(8)) * $(9216)) + (r197_lane[:, None] * $(48)) + r197_col[None, :]) / $(48)) % $(192))), dtype=tl.float32)
      r197_value = tl.dot(r197_left, r197_right)
      r197_value = r197_value.to(tl.float32)
      tl.store($(candidate_vbmm_32_ptr) + ((r197_group * $(768)) + (r197_row[:, None] * $(48)) + r197_col[None, :]), r197_value)
    }
    r198_z = tl.arange($(6144))
    r198_value = (tl.load($(candidate_vbmm_32_ptr) + ((((r198_z / $(768)) % $(8)) * $(768)) + (((r198_z / $(48)) % $(16)) * $(48)) + (r198_z % $(48))), dtype=tl.float32) * 0.07216878364870322)
    r198_value = r198_value.to(tl.float32)
    r198_value = tl.broadcast_to(r198_value, [$(6144)])
    tl.store($(candidate_vmul_34_ptr) + r198_z, r198_value)
    r199_z = tl.arange($(16))
    r199_value = (tl.cast(r199_z, tl.int64) + tl.full([], $(48), dtype=tl.int64))
    r199_value = tl.broadcast_to(r199_value, [$(16)])
    tl.store($(candidate_vadd_19_ptr) + r199_z, r199_value)
    r200_z = tl.arange($(16))
    r200_value = (tl.load($(candidate_vadd_19_ptr) + r200_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r200_value = tl.broadcast_to(r200_value, [$(16)])
    tl.store($(candidate_vsub_20_ptr) + r200_z, r200_value)
    r201_z = tl.arange($(768))
    r201_value = (tl.cast((r201_z % $(48)), tl.int64) > tl.load($(candidate_vsub_20_ptr) + ((r201_z / $(48)) % $(16)), dtype=tl.int64))
    r201_value = tl.broadcast_to(r201_value, [$(768)])
    tl.store($(candidate_vgt_11_ptr) + r201_z, r201_value)
    r202_z = tl.arange($(6144))
    r202_value = tl.where(tl.load($(candidate_vgt_11_ptr) + ((((r202_z / $(48)) % $(16)) * $(48)) + (r202_z % $(48))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_34_ptr) + ((((r202_z / $(768)) % $(8)) * $(768)) + (((r202_z / $(48)) % $(16)) * $(48)) + (r202_z % $(48))), dtype=tl.float32))
    r202_value = r202_value.to(tl.float32)
    r202_value = tl.broadcast_to(r202_value, [$(6144)])
    tl.store($(candidate_vmasked_fill_11_ptr) + r202_z, r202_value)
    r203_z = tl.arange($(6144))
    r203_lane = tl.arange($(48))
    r203_row_values = tl.load($(candidate_vmasked_fill_11_ptr) + ((((r203_z[:, None] / $(768)) % $(8)) * $(768)) + (((r203_z[:, None] / $(48)) % $(16)) * $(48)) + (r203_lane[None, :] % $(48))), dtype=tl.float32)
    r203_maximum = tl.max(r203_row_values, axis=1, keep_dims=true)
    r203_weights = tl.exp(r203_row_values - r203_maximum)
    r203_total = tl.sum(r203_weights, axis=1)
    r203_maximum = tl.max(r203_row_values, axis=1)
    r203_value = tl.exp(tl.load($(candidate_vmasked_fill_11_ptr) + ((((r203_z / $(768)) % $(8)) * $(768)) + (((r203_z / $(48)) % $(16)) * $(48)) + (r203_z % $(48))), dtype=tl.float32) - r203_maximum) / r203_total
    r203_value = r203_value.to(tl.float32)
    r203_value = tl.broadcast_to(r203_value, [$(6144)])
    tl.store($(candidate_vsoftmax_11_ptr) + r203_z, r203_value)
    r204_z = tl.arange($(6144))
    r204_value = tl.load($(candidate_vsoftmax_11_ptr) + ((((r204_z / $(768)) % $(8)) * $(768)) + (((r204_z / $(48)) % $(16)) * $(48)) + (r204_z % $(48))), dtype=tl.float32)
    r204_value = r204_value.to(tl.bfloat16)
    r204_value = tl.broadcast_to(r204_value, [$(6144)])
    tl.store($(candidate_vto_copy_64_ptr) + r204_z, r204_value)
    r205_z = tl.arange($(65536))
    r205_value = tl.cast(0.0, tl.bfloat16)
    for r205_lane in range($(128)) {
      r205_value = tl.where(tl.load($(candidate_vsort_index_ptr) + (r205_lane % $(128)), dtype=tl.int64) == ((r205_z / $(512)) % $(128)), r205_value + tl.load($(input1_ptr) + (((r205_lane % $(128)) * $(512)) + (r205_z % $(512))), dtype=tl.bfloat16), r205_value)
    }
    r205_value = r205_value.to(tl.bfloat16)
    r205_value = tl.broadcast_to(r205_value, [$(65536)])
    tl.store($(candidate_vindex_put_ptr) + r205_z, r205_value)
    r206_row = tl.arange($(128))
    r206_col = tl.arange($(1024))
    r206_lane = tl.arange($(512))
    r206_left = tl.load($(candidate_vindex_put_ptr) + ((((((r206_row[:, None] * $(512)) + r206_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r206_row[:, None] * $(512)) + r206_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r206_right = tl.load($(candidate_vto_copy_6_ptr) + (((r206_lane[:, None] % $(512)) * $(1024)) + (r206_col[None, :] % $(1024))), dtype=tl.bfloat16)
    r206_value = tl.dot(r206_left, r206_right)
    r206_value = r206_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_4_ptr) + ((r206_row[:, None] * $(1024)) + r206_col[None, :]), r206_value)
    r207_row = tl.arange($(48))
    r207_col = tl.arange($(128))
    r207_lane = tl.arange($(16))
    for r207_group in range($(8)) {
      r207_left = tl.load($(candidate_vto_copy_64_ptr) + (((((((r207_group % $(8)) * $(768)) + (r207_lane[None, :] * $(48)) + r207_row[:, None]) / $(768)) % $(8)) * $(768)) + ((((((r207_group % $(8)) * $(768)) + (r207_lane[None, :] * $(48)) + r207_row[:, None]) / $(48)) % $(16)) * $(48)) + ((((r207_group % $(8)) * $(768)) + (r207_lane[None, :] * $(48)) + r207_row[:, None]) % $(48))), dtype=tl.bfloat16)
      r207_right = tl.load($(candidate_vmm_4_ptr) + (((((((r207_group % $(8)) * $(2048)) + (r207_lane[:, None] * $(128)) + r207_col[None, :]) / $(128)) % $(16)) * $(1024)) + ((((((r207_group % $(8)) * $(2048)) + (r207_lane[:, None] * $(128)) + r207_col[None, :]) / $(2048)) % $(8)) * $(128)) + ((((r207_group % $(8)) * $(2048)) + (r207_lane[:, None] * $(128)) + r207_col[None, :]) % $(128)) + $(114688)), dtype=tl.bfloat16)
      r207_value = tl.dot(r207_left, r207_right)
      r207_value = r207_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_34_ptr) + ((r207_group * $(6144)) + (r207_row[:, None] * $(128)) + r207_col[None, :]), r207_value)
    }
    r208_row = tl.arange($(16))
    r208_col = tl.arange($(48))
    r208_lane = tl.arange($(128))
    for r208_group in range($(8)) {
      r208_left = tl.load($(candidate_vmm_4_ptr) + (((((((r208_group % $(8)) * $(2048)) + (r208_row[:, None] * $(128)) + r208_lane[None, :]) / $(128)) % $(16)) * $(1024)) + ((((((r208_group % $(8)) * $(2048)) + (r208_row[:, None] * $(128)) + r208_lane[None, :]) / $(2048)) % $(8)) * $(128)) + ((((r208_group % $(8)) * $(2048)) + (r208_row[:, None] * $(128)) + r208_lane[None, :]) % $(128)) + $(114688)), dtype=tl.bfloat16)
      r208_right = tl.load($(candidate_vbmm_27_ptr) + (((((((r208_group % $(8)) * $(6144)) + (r208_col[None, :] * $(128)) + r208_lane[:, None]) / $(128)) % $(48)) * $(1024)) + ((((((r208_group % $(8)) * $(6144)) + (r208_col[None, :] * $(128)) + r208_lane[:, None]) / $(6144)) % $(8)) * $(128)) + ((((r208_group % $(8)) * $(6144)) + (r208_col[None, :] * $(128)) + r208_lane[:, None]) % $(128)) + $(65536)), dtype=tl.bfloat16)
      r208_value = tl.dot(r208_left, r208_right)
      r208_value = r208_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_35_ptr) + ((r208_group * $(768)) + (r208_row[:, None] * $(48)) + r208_col[None, :]), r208_value)
    }
    r209_z = tl.arange($(6144))
    r209_value = tl.load($(candidate_vbmm_35_ptr) + ((((r209_z / $(768)) % $(8)) * $(768)) + (((r209_z / $(48)) % $(16)) * $(48)) + (r209_z % $(48))), dtype=tl.bfloat16)
    r209_value = r209_value.to(tl.float32)
    r209_value = tl.broadcast_to(r209_value, [$(6144)])
    tl.store($(candidate_vto_copy_65_ptr) + r209_z, r209_value)
    r210_z = tl.arange($(6144))
    r210_lane = tl.arange($(48))
    r210_row_values = tl.load($(candidate_vsoftmax_11_ptr) + ((((r210_z[:, None] / $(768)) % $(8)) * $(768)) + (((r210_z[:, None] / $(48)) % $(16)) * $(48)) + (r210_lane[None, :] % $(48))), dtype=tl.float32)
    r210_row_grad = tl.load($(candidate_vto_copy_65_ptr) + ((((r210_z[:, None] / $(768)) % $(8)) * $(768)) + (((r210_z[:, None] / $(48)) % $(16)) * $(48)) + (r210_lane[None, :] % $(48))), dtype=tl.float32)
    r210_weighted = r210_row_grad * r210_row_values
    r210_dot_grad = tl.sum(r210_weighted, axis=1)
    r210_value = (tl.load($(candidate_vto_copy_65_ptr) + ((((r210_z / $(768)) % $(8)) * $(768)) + (((r210_z / $(48)) % $(16)) * $(48)) + (r210_z % $(48))), dtype=tl.float32) - r210_dot_grad) * tl.load($(candidate_vsoftmax_11_ptr) + ((((r210_z / $(768)) % $(8)) * $(768)) + (((r210_z / $(48)) % $(16)) * $(48)) + (r210_z % $(48))), dtype=tl.float32)
    r210_value = r210_value.to(tl.float32)
    r210_value = tl.broadcast_to(r210_value, [$(6144)])
    tl.store($(candidate_vsoftmax_backward_data_ptr) + r210_z, r210_value)
    r211_z = tl.arange($(6144))
    r211_value = tl.where(tl.load($(candidate_vgt_11_ptr) + ((((r211_z / $(48)) % $(16)) * $(48)) + (r211_z % $(48))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_ptr) + ((((r211_z / $(768)) % $(8)) * $(768)) + (((r211_z / $(48)) % $(16)) * $(48)) + (r211_z % $(48))), dtype=tl.float32))
    r211_value = r211_value.to(tl.float32)
    r211_value = tl.broadcast_to(r211_value, [$(6144)])
    tl.store($(candidate_vmasked_fill_12_ptr) + r211_z, r211_value)
    r212_z = tl.arange($(6144))
    r212_value = (tl.load($(candidate_vmasked_fill_12_ptr) + ((((r212_z / $(768)) % $(8)) * $(768)) + (((r212_z / $(48)) % $(16)) * $(48)) + (r212_z % $(48))), dtype=tl.float32) * 0.07216878364870322)
    r212_value = r212_value.to(tl.float32)
    r212_value = tl.broadcast_to(r212_value, [$(6144)])
    tl.store($(candidate_vmul_35_ptr) + r212_z, r212_value)
    r213_row = tl.arange($(192))
    r213_col = tl.arange($(48))
    r213_lane = tl.arange($(16))
    for r213_group in range($(8)) {
      r213_left = tl.load($(candidate_vto_copy_62_ptr) + (((((((r213_group % $(8)) * $(3072)) + (r213_lane[None, :] * $(192)) + r213_row[:, None]) / $(3072)) % $(8)) * $(3072)) + ((((((r213_group % $(8)) * $(3072)) + (r213_lane[None, :] * $(192)) + r213_row[:, None]) / $(192)) % $(16)) * $(192)) + ((((r213_group % $(8)) * $(3072)) + (r213_lane[None, :] * $(192)) + r213_row[:, None]) % $(192))), dtype=tl.float32)
      r213_right = tl.load($(candidate_vmul_35_ptr) + (((((((r213_group % $(8)) * $(768)) + (r213_lane[:, None] * $(48)) + r213_col[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r213_group % $(8)) * $(768)) + (r213_lane[:, None] * $(48)) + r213_col[None, :]) / $(48)) % $(16)) * $(48)) + ((((r213_group % $(8)) * $(768)) + (r213_lane[:, None] * $(48)) + r213_col[None, :]) % $(48))), dtype=tl.float32)
      r213_value = tl.dot(r213_left, r213_right)
      r213_value = r213_value.to(tl.float32)
      tl.store($(candidate_vbmm_36_ptr) + ((r213_group * $(9216)) + (r213_row[:, None] * $(48)) + r213_col[None, :]), r213_value)
    }
    r214_row = tl.arange($(16))
    r214_col = tl.arange($(192))
    r214_lane = tl.arange($(48))
    for r214_group in range($(8)) {
      r214_left = tl.load($(candidate_vmul_35_ptr) + (((((((r214_group % $(8)) * $(768)) + (r214_row[:, None] * $(48)) + r214_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r214_group % $(8)) * $(768)) + (r214_row[:, None] * $(48)) + r214_lane[None, :]) / $(48)) % $(16)) * $(48)) + ((((r214_group % $(8)) * $(768)) + (r214_row[:, None] * $(48)) + r214_lane[None, :]) % $(48))), dtype=tl.float32)
      r214_right = tl.load($(candidate_vto_copy_63_ptr) + (((((((r214_group % $(8)) * $(9216)) + (r214_col[None, :] * $(48)) + r214_lane[:, None]) / $(9216)) % $(8)) * $(9216)) + (((((r214_group % $(8)) * $(9216)) + (r214_col[None, :] * $(48)) + r214_lane[:, None]) % $(48)) * $(192)) + (((((r214_group % $(8)) * $(9216)) + (r214_col[None, :] * $(48)) + r214_lane[:, None]) / $(48)) % $(192))), dtype=tl.float32)
      r214_value = tl.dot(r214_left, r214_right)
      r214_value = r214_value.to(tl.float32)
      tl.store($(candidate_vbmm_37_ptr) + ((r214_group * $(3072)) + (r214_row[:, None] * $(192)) + r214_col[None, :]), r214_value)
    }
    r215_z = tl.arange($(131072))
    r215_value = tl.where(((((r215_z / $(1024)) % $(128)) >= $(64)) & (((r215_z / $(1024)) % $(128)) < $(112)) & ((((r215_z / $(1024)) % $(128)) - $(64)) % $(1) == $(0))), tl.load($(candidate_vbmm_34_ptr) + ((((((((r215_z / $(128)) % $(8)) * $(6144)) + ((((r215_z / $(1024)) % $(128)) - $(64)) * $(128)) + (r215_z % $(128))) / $(6144)) % $(8)) * $(6144)) + ((((((r215_z / $(128)) % $(8)) * $(48)) + (((r215_z / $(1024)) % $(128)) - $(64))) % $(48)) * $(128)) + (r215_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r215_value = r215_value.to(tl.bfloat16)
    r215_value = tl.broadcast_to(r215_value, [$(131072)])
    tl.store($(candidate_vslice_backward_ptr) + r215_z, r215_value)
    r216_z = tl.arange($(73728))
    r216_value = tl.load($(candidate_vbmm_36_ptr) + ((((r216_z / $(9216)) % $(8)) * $(9216)) + ((r216_z % $(192)) * $(48)) + ((r216_z / $(192)) % $(48))), dtype=tl.float32)
    r216_value = r216_value.to(tl.bfloat16)
    r216_value = tl.broadcast_to(r216_value, [$(73728)])
    tl.store($(candidate_vto_copy_66_ptr) + r216_z, r216_value)
    r217_z = tl.arange($(196608))
    r217_value = tl.where(((((r217_z / $(1536)) % $(128)) >= $(64)) & (((r217_z / $(1536)) % $(128)) < $(112)) & ((((r217_z / $(1536)) % $(128)) - $(64)) % $(1) == $(0))), tl.load($(candidate_vto_copy_66_ptr) + ((((r217_z / $(192)) % $(8)) * $(9216)) + (((((r217_z / $(1536)) % $(128)) - $(64)) % $(48)) * $(192)) + (r217_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r217_value = r217_value.to(tl.bfloat16)
    r217_value = tl.broadcast_to(r217_value, [$(196608)])
    tl.store($(candidate_vslice_backward_1_ptr) + r217_z, r217_value)
    r218_z = tl.arange($(24576))
    r218_value = tl.load($(candidate_vbmm_37_ptr) + ((((r218_z / $(3072)) % $(8)) * $(3072)) + (((r218_z / $(192)) % $(16)) * $(192)) + (r218_z % $(192))), dtype=tl.float32)
    r218_value = r218_value.to(tl.bfloat16)
    r218_value = tl.broadcast_to(r218_value, [$(24576)])
    tl.store($(candidate_vto_copy_67_ptr) + r218_z, r218_value)
    r219_z = tl.arange($(49152))
    r219_value = tl.where(((((r219_z / $(1536)) % $(32)) >= $(16)) & (((r219_z / $(1536)) % $(32)) < $(32)) & ((((r219_z / $(1536)) % $(32)) - $(16)) % $(1) == $(0))), tl.load($(candidate_vto_copy_67_ptr) + ((((r219_z / $(192)) % $(8)) * $(3072)) + (((((r219_z / $(1536)) % $(32)) - $(16)) % $(16)) * $(192)) + (r219_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r219_value = r219_value.to(tl.bfloat16)
    r219_value = tl.broadcast_to(r219_value, [$(49152)])
    tl.store($(candidate_vslice_backward_2_ptr) + r219_z, r219_value)
    r220_row = tl.arange($(12))
    r220_col = tl.arange($(128))
    r220_lane = tl.arange($(4))
    for r220_group in range($(8)) {
      r220_left = tl.load($(candidate_vto_copy_61_ptr) + (((((((r220_group % $(8)) * $(48)) + (r220_lane[None, :] * $(12)) + r220_row[:, None]) / $(48)) % $(8)) * $(48)) + ((((((r220_group % $(8)) * $(48)) + (r220_lane[None, :] * $(12)) + r220_row[:, None]) / $(12)) % $(4)) * $(12)) + ((((r220_group % $(8)) * $(48)) + (r220_lane[None, :] * $(12)) + r220_row[:, None]) % $(12))), dtype=tl.bfloat16)
      r220_right = tl.load($(candidate_vmm_4_ptr) + (((((((r220_group % $(8)) * $(512)) + (r220_lane[:, None] * $(128)) + r220_col[None, :]) / $(128)) % $(4)) * $(1024)) + ((((((r220_group % $(8)) * $(512)) + (r220_lane[:, None] * $(128)) + r220_col[None, :]) / $(512)) % $(8)) * $(128)) + ((((r220_group % $(8)) * $(512)) + (r220_lane[:, None] * $(128)) + r220_col[None, :]) % $(128)) + $(110592)), dtype=tl.bfloat16)
      r220_value = tl.dot(r220_left, r220_right)
      r220_value = r220_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_38_ptr) + ((r220_group * $(1536)) + (r220_row[:, None] * $(128)) + r220_col[None, :]), r220_value)
    }
    r221_row = tl.arange($(4))
    r221_col = tl.arange($(12))
    r221_lane = tl.arange($(128))
    for r221_group in range($(8)) {
      r221_left = tl.load($(candidate_vmm_4_ptr) + (((((((r221_group % $(8)) * $(512)) + (r221_row[:, None] * $(128)) + r221_lane[None, :]) / $(128)) % $(4)) * $(1024)) + ((((((r221_group % $(8)) * $(512)) + (r221_row[:, None] * $(128)) + r221_lane[None, :]) / $(512)) % $(8)) * $(128)) + ((((r221_group % $(8)) * $(512)) + (r221_row[:, None] * $(128)) + r221_lane[None, :]) % $(128)) + $(110592)), dtype=tl.bfloat16)
      r221_right = tl.load($(candidate_vbmm_27_ptr) + (((((((r221_group % $(8)) * $(1536)) + (r221_col[None, :] * $(128)) + r221_lane[:, None]) / $(128)) % $(12)) * $(1024)) + ((((((r221_group % $(8)) * $(1536)) + (r221_col[None, :] * $(128)) + r221_lane[:, None]) / $(1536)) % $(8)) * $(128)) + ((((r221_group % $(8)) * $(1536)) + (r221_col[None, :] * $(128)) + r221_lane[:, None]) % $(128)) + $(49152)), dtype=tl.bfloat16)
      r221_value = tl.dot(r221_left, r221_right)
      r221_value = r221_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_39_ptr) + ((r221_group * $(48)) + (r221_row[:, None] * $(12)) + r221_col[None, :]), r221_value)
    }
    r222_z = tl.arange($(384))
    r222_value = tl.load($(candidate_vbmm_39_ptr) + ((((r222_z / $(48)) % $(8)) * $(48)) + (((r222_z / $(12)) % $(4)) * $(12)) + (r222_z % $(12))), dtype=tl.bfloat16)
    r222_value = r222_value.to(tl.float32)
    r222_value = tl.broadcast_to(r222_value, [$(384)])
    tl.store($(candidate_vto_copy_68_ptr) + r222_z, r222_value)
    r223_z = tl.arange($(384))
    r223_lane = tl.arange($(12))
    r223_row_values = tl.load($(candidate_vsoftmax_10_ptr) + ((((r223_z[:, None] / $(48)) % $(8)) * $(48)) + (((r223_z[:, None] / $(12)) % $(4)) * $(12)) + (r223_lane[None, :] % $(12))), dtype=tl.float32)
    r223_row_grad = tl.load($(candidate_vto_copy_68_ptr) + ((((r223_z[:, None] / $(48)) % $(8)) * $(48)) + (((r223_z[:, None] / $(12)) % $(4)) * $(12)) + (r223_lane[None, :] % $(12))), dtype=tl.float32)
    r223_weighted = r223_row_grad * r223_row_values
    r223_dot_grad = tl.sum(r223_weighted, axis=1)
    r223_value = (tl.load($(candidate_vto_copy_68_ptr) + ((((r223_z / $(48)) % $(8)) * $(48)) + (((r223_z / $(12)) % $(4)) * $(12)) + (r223_z % $(12))), dtype=tl.float32) - r223_dot_grad) * tl.load($(candidate_vsoftmax_10_ptr) + ((((r223_z / $(48)) % $(8)) * $(48)) + (((r223_z / $(12)) % $(4)) * $(12)) + (r223_z % $(12))), dtype=tl.float32)
    r223_value = r223_value.to(tl.float32)
    r223_value = tl.broadcast_to(r223_value, [$(384)])
    tl.store($(candidate_vsoftmax_backward_data_1_ptr) + r223_z, r223_value)
    r224_z = tl.arange($(384))
    r224_value = tl.where(tl.load($(candidate_vgt_10_ptr) + ((((r224_z / $(12)) % $(4)) * $(12)) + (r224_z % $(12))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_1_ptr) + ((((r224_z / $(48)) % $(8)) * $(48)) + (((r224_z / $(12)) % $(4)) * $(12)) + (r224_z % $(12))), dtype=tl.float32))
    r224_value = r224_value.to(tl.float32)
    r224_value = tl.broadcast_to(r224_value, [$(384)])
    tl.store($(candidate_vmasked_fill_13_ptr) + r224_z, r224_value)
    r225_z = tl.arange($(384))
    r225_value = (tl.load($(candidate_vmasked_fill_13_ptr) + ((((r225_z / $(48)) % $(8)) * $(48)) + (((r225_z / $(12)) % $(4)) * $(12)) + (r225_z % $(12))), dtype=tl.float32) * 0.07216878364870322)
    r225_value = r225_value.to(tl.float32)
    r225_value = tl.broadcast_to(r225_value, [$(384)])
    tl.store($(candidate_vmul_36_ptr) + r225_z, r225_value)
    r226_row = tl.arange($(192))
    r226_col = tl.arange($(12))
    r226_lane = tl.arange($(4))
    for r226_group in range($(8)) {
      r226_left = tl.load($(candidate_vto_copy_59_ptr) + (((((((r226_group % $(8)) * $(768)) + (r226_lane[None, :] * $(192)) + r226_row[:, None]) / $(768)) % $(8)) * $(768)) + ((((((r226_group % $(8)) * $(768)) + (r226_lane[None, :] * $(192)) + r226_row[:, None]) / $(192)) % $(4)) * $(192)) + ((((r226_group % $(8)) * $(768)) + (r226_lane[None, :] * $(192)) + r226_row[:, None]) % $(192))), dtype=tl.float32)
      r226_right = tl.load($(candidate_vmul_36_ptr) + (((((((r226_group % $(8)) * $(48)) + (r226_lane[:, None] * $(12)) + r226_col[None, :]) / $(48)) % $(8)) * $(48)) + ((((((r226_group % $(8)) * $(48)) + (r226_lane[:, None] * $(12)) + r226_col[None, :]) / $(12)) % $(4)) * $(12)) + ((((r226_group % $(8)) * $(48)) + (r226_lane[:, None] * $(12)) + r226_col[None, :]) % $(12))), dtype=tl.float32)
      r226_value = tl.dot(r226_left, r226_right)
      r226_value = r226_value.to(tl.float32)
      tl.store($(candidate_vbmm_40_ptr) + ((r226_group * $(2304)) + (r226_row[:, None] * $(12)) + r226_col[None, :]), r226_value)
    }
    r227_row = tl.arange($(4))
    r227_col = tl.arange($(192))
    r227_lane = tl.arange($(12))
    for r227_group in range($(8)) {
      r227_left = tl.load($(candidate_vmul_36_ptr) + (((((((r227_group % $(8)) * $(48)) + (r227_row[:, None] * $(12)) + r227_lane[None, :]) / $(48)) % $(8)) * $(48)) + ((((((r227_group % $(8)) * $(48)) + (r227_row[:, None] * $(12)) + r227_lane[None, :]) / $(12)) % $(4)) * $(12)) + ((((r227_group % $(8)) * $(48)) + (r227_row[:, None] * $(12)) + r227_lane[None, :]) % $(12))), dtype=tl.float32)
      r227_right = tl.load($(candidate_vto_copy_60_ptr) + (((((((r227_group % $(8)) * $(2304)) + (r227_col[None, :] * $(12)) + r227_lane[:, None]) / $(2304)) % $(8)) * $(2304)) + (((((r227_group % $(8)) * $(2304)) + (r227_col[None, :] * $(12)) + r227_lane[:, None]) % $(12)) * $(192)) + (((((r227_group % $(8)) * $(2304)) + (r227_col[None, :] * $(12)) + r227_lane[:, None]) / $(12)) % $(192))), dtype=tl.float32)
      r227_value = tl.dot(r227_left, r227_right)
      r227_value = r227_value.to(tl.float32)
      tl.store($(candidate_vbmm_41_ptr) + ((r227_group * $(768)) + (r227_row[:, None] * $(192)) + r227_col[None, :]), r227_value)
    }
    r228_z = tl.arange($(131072))
    r228_value = tl.where(((((r228_z / $(1024)) % $(128)) >= $(48)) & (((r228_z / $(1024)) % $(128)) < $(60)) & ((((r228_z / $(1024)) % $(128)) - $(48)) % $(1) == $(0))), tl.load($(candidate_vbmm_38_ptr) + ((((((((r228_z / $(128)) % $(8)) * $(1536)) + ((((r228_z / $(1024)) % $(128)) - $(48)) * $(128)) + (r228_z % $(128))) / $(1536)) % $(8)) * $(1536)) + ((((((r228_z / $(128)) % $(8)) * $(12)) + (((r228_z / $(1024)) % $(128)) - $(48))) % $(12)) * $(128)) + (r228_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r228_value = r228_value.to(tl.bfloat16)
    r228_value = tl.broadcast_to(r228_value, [$(131072)])
    tl.store($(candidate_vslice_backward_3_ptr) + r228_z, r228_value)
    r229_z = tl.arange($(131072))
    r229_value = (tl.load($(candidate_vslice_backward_ptr) + ((((r229_z / $(1024)) % $(128)) * $(1024)) + (((r229_z / $(128)) % $(8)) * $(128)) + (r229_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_3_ptr) + ((((r229_z / $(1024)) % $(128)) * $(1024)) + (((r229_z / $(128)) % $(8)) * $(128)) + (r229_z % $(128))), dtype=tl.bfloat16))
    r229_value = r229_value.to(tl.bfloat16)
    r229_value = tl.broadcast_to(r229_value, [$(131072)])
    tl.store($(candidate_vadd_20_ptr) + r229_z, r229_value)
    r230_z = tl.arange($(18432))
    r230_value = tl.load($(candidate_vbmm_40_ptr) + ((((r230_z / $(2304)) % $(8)) * $(2304)) + ((r230_z % $(192)) * $(12)) + ((r230_z / $(192)) % $(12))), dtype=tl.float32)
    r230_value = r230_value.to(tl.bfloat16)
    r230_value = tl.broadcast_to(r230_value, [$(18432)])
    tl.store($(candidate_vto_copy_69_ptr) + r230_z, r230_value)
    r231_z = tl.arange($(196608))
    r231_value = tl.where(((((r231_z / $(1536)) % $(128)) >= $(48)) & (((r231_z / $(1536)) % $(128)) < $(60)) & ((((r231_z / $(1536)) % $(128)) - $(48)) % $(1) == $(0))), tl.load($(candidate_vto_copy_69_ptr) + ((((r231_z / $(192)) % $(8)) * $(2304)) + (((((r231_z / $(1536)) % $(128)) - $(48)) % $(12)) * $(192)) + (r231_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r231_value = r231_value.to(tl.bfloat16)
    r231_value = tl.broadcast_to(r231_value, [$(196608)])
    tl.store($(candidate_vslice_backward_4_ptr) + r231_z, r231_value)
    r232_z = tl.arange($(196608))
    r232_value = (tl.load($(candidate_vslice_backward_1_ptr) + ((((r232_z / $(1536)) % $(128)) * $(1536)) + (((r232_z / $(192)) % $(8)) * $(192)) + (r232_z % $(192))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_4_ptr) + ((((r232_z / $(1536)) % $(128)) * $(1536)) + (((r232_z / $(192)) % $(8)) * $(192)) + (r232_z % $(192))), dtype=tl.bfloat16))
    r232_value = r232_value.to(tl.bfloat16)
    r232_value = tl.broadcast_to(r232_value, [$(196608)])
    tl.store($(candidate_vadd_21_ptr) + r232_z, r232_value)
    r233_z = tl.arange($(6144))
    r233_value = tl.load($(candidate_vbmm_41_ptr) + ((((r233_z / $(768)) % $(8)) * $(768)) + (((r233_z / $(192)) % $(4)) * $(192)) + (r233_z % $(192))), dtype=tl.float32)
    r233_value = r233_value.to(tl.bfloat16)
    r233_value = tl.broadcast_to(r233_value, [$(6144)])
    tl.store($(candidate_vto_copy_70_ptr) + r233_z, r233_value)
    r234_z = tl.arange($(49152))
    r234_value = tl.where(((((r234_z / $(1536)) % $(32)) >= $(12)) & (((r234_z / $(1536)) % $(32)) < $(16)) & ((((r234_z / $(1536)) % $(32)) - $(12)) % $(1) == $(0))), tl.load($(candidate_vto_copy_70_ptr) + ((((r234_z / $(192)) % $(8)) * $(768)) + (((((r234_z / $(1536)) % $(32)) - $(12)) % $(4)) * $(192)) + (r234_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r234_value = r234_value.to(tl.bfloat16)
    r234_value = tl.broadcast_to(r234_value, [$(49152)])
    tl.store($(candidate_vslice_backward_5_ptr) + r234_z, r234_value)
    r235_z = tl.arange($(49152))
    r235_value = (tl.load($(candidate_vslice_backward_2_ptr) + ((((r235_z / $(1536)) % $(32)) * $(1536)) + (((r235_z / $(192)) % $(8)) * $(192)) + (r235_z % $(192))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_5_ptr) + ((((r235_z / $(1536)) % $(32)) * $(1536)) + (((r235_z / $(192)) % $(8)) * $(192)) + (r235_z % $(192))), dtype=tl.bfloat16))
    r235_value = r235_value.to(tl.bfloat16)
    r235_value = tl.broadcast_to(r235_value, [$(49152)])
    tl.store($(candidate_vadd_22_ptr) + r235_z, r235_value)
    r236_row = tl.arange($(36))
    r236_col = tl.arange($(128))
    r236_lane = tl.arange($(12))
    for r236_group in range($(8)) {
      r236_left = tl.load($(candidate_vto_copy_58_ptr) + (((((((r236_group % $(8)) * $(432)) + (r236_lane[None, :] * $(36)) + r236_row[:, None]) / $(432)) % $(8)) * $(432)) + ((((((r236_group % $(8)) * $(432)) + (r236_lane[None, :] * $(36)) + r236_row[:, None]) / $(36)) % $(12)) * $(36)) + ((((r236_group % $(8)) * $(432)) + (r236_lane[None, :] * $(36)) + r236_row[:, None]) % $(36))), dtype=tl.bfloat16)
      r236_right = tl.load($(candidate_vmm_4_ptr) + (((((((r236_group % $(8)) * $(1536)) + (r236_lane[:, None] * $(128)) + r236_col[None, :]) / $(128)) % $(12)) * $(1024)) + ((((((r236_group % $(8)) * $(1536)) + (r236_lane[:, None] * $(128)) + r236_col[None, :]) / $(1536)) % $(8)) * $(128)) + ((((r236_group % $(8)) * $(1536)) + (r236_lane[:, None] * $(128)) + r236_col[None, :]) % $(128)) + $(98304)), dtype=tl.bfloat16)
      r236_value = tl.dot(r236_left, r236_right)
      r236_value = r236_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_42_ptr) + ((r236_group * $(4608)) + (r236_row[:, None] * $(128)) + r236_col[None, :]), r236_value)
    }
    r237_row = tl.arange($(12))
    r237_col = tl.arange($(36))
    r237_lane = tl.arange($(128))
    for r237_group in range($(8)) {
      r237_left = tl.load($(candidate_vmm_4_ptr) + (((((((r237_group % $(8)) * $(1536)) + (r237_row[:, None] * $(128)) + r237_lane[None, :]) / $(128)) % $(12)) * $(1024)) + ((((((r237_group % $(8)) * $(1536)) + (r237_row[:, None] * $(128)) + r237_lane[None, :]) / $(1536)) % $(8)) * $(128)) + ((((r237_group % $(8)) * $(1536)) + (r237_row[:, None] * $(128)) + r237_lane[None, :]) % $(128)) + $(98304)), dtype=tl.bfloat16)
      r237_right = tl.load($(candidate_vbmm_27_ptr) + (((((((r237_group % $(8)) * $(4608)) + (r237_col[None, :] * $(128)) + r237_lane[:, None]) / $(128)) % $(36)) * $(1024)) + ((((((r237_group % $(8)) * $(4608)) + (r237_col[None, :] * $(128)) + r237_lane[:, None]) / $(4608)) % $(8)) * $(128)) + ((((r237_group % $(8)) * $(4608)) + (r237_col[None, :] * $(128)) + r237_lane[:, None]) % $(128))), dtype=tl.bfloat16)
      r237_value = tl.dot(r237_left, r237_right)
      r237_value = r237_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_43_ptr) + ((r237_group * $(432)) + (r237_row[:, None] * $(36)) + r237_col[None, :]), r237_value)
    }
    r238_z = tl.arange($(3456))
    r238_value = tl.load($(candidate_vbmm_43_ptr) + ((((r238_z / $(432)) % $(8)) * $(432)) + (((r238_z / $(36)) % $(12)) * $(36)) + (r238_z % $(36))), dtype=tl.bfloat16)
    r238_value = r238_value.to(tl.float32)
    r238_value = tl.broadcast_to(r238_value, [$(3456)])
    tl.store($(candidate_vto_copy_71_ptr) + r238_z, r238_value)
    r239_z = tl.arange($(3456))
    r239_lane = tl.arange($(36))
    r239_row_values = tl.load($(candidate_vsoftmax_9_ptr) + ((((r239_z[:, None] / $(432)) % $(8)) * $(432)) + (((r239_z[:, None] / $(36)) % $(12)) * $(36)) + (r239_lane[None, :] % $(36))), dtype=tl.float32)
    r239_row_grad = tl.load($(candidate_vto_copy_71_ptr) + ((((r239_z[:, None] / $(432)) % $(8)) * $(432)) + (((r239_z[:, None] / $(36)) % $(12)) * $(36)) + (r239_lane[None, :] % $(36))), dtype=tl.float32)
    r239_weighted = r239_row_grad * r239_row_values
    r239_dot_grad = tl.sum(r239_weighted, axis=1)
    r239_value = (tl.load($(candidate_vto_copy_71_ptr) + ((((r239_z / $(432)) % $(8)) * $(432)) + (((r239_z / $(36)) % $(12)) * $(36)) + (r239_z % $(36))), dtype=tl.float32) - r239_dot_grad) * tl.load($(candidate_vsoftmax_9_ptr) + ((((r239_z / $(432)) % $(8)) * $(432)) + (((r239_z / $(36)) % $(12)) * $(36)) + (r239_z % $(36))), dtype=tl.float32)
    r239_value = r239_value.to(tl.float32)
    r239_value = tl.broadcast_to(r239_value, [$(3456)])
    tl.store($(candidate_vsoftmax_backward_data_2_ptr) + r239_z, r239_value)
    r240_z = tl.arange($(3456))
    r240_value = tl.where(tl.load($(candidate_vgt_9_ptr) + ((((r240_z / $(36)) % $(12)) * $(36)) + (r240_z % $(36))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_2_ptr) + ((((r240_z / $(432)) % $(8)) * $(432)) + (((r240_z / $(36)) % $(12)) * $(36)) + (r240_z % $(36))), dtype=tl.float32))
    r240_value = r240_value.to(tl.float32)
    r240_value = tl.broadcast_to(r240_value, [$(3456)])
    tl.store($(candidate_vmasked_fill_14_ptr) + r240_z, r240_value)
    r241_z = tl.arange($(3456))
    r241_value = (tl.load($(candidate_vmasked_fill_14_ptr) + ((((r241_z / $(432)) % $(8)) * $(432)) + (((r241_z / $(36)) % $(12)) * $(36)) + (r241_z % $(36))), dtype=tl.float32) * 0.07216878364870322)
    r241_value = r241_value.to(tl.float32)
    r241_value = tl.broadcast_to(r241_value, [$(3456)])
    tl.store($(candidate_vmul_37_ptr) + r241_z, r241_value)
    r242_row = tl.arange($(192))
    r242_col = tl.arange($(36))
    r242_lane = tl.arange($(12))
    for r242_group in range($(8)) {
      r242_left = tl.load($(candidate_vto_copy_56_ptr) + (((((((r242_group % $(8)) * $(2304)) + (r242_lane[None, :] * $(192)) + r242_row[:, None]) / $(2304)) % $(8)) * $(2304)) + ((((((r242_group % $(8)) * $(2304)) + (r242_lane[None, :] * $(192)) + r242_row[:, None]) / $(192)) % $(12)) * $(192)) + ((((r242_group % $(8)) * $(2304)) + (r242_lane[None, :] * $(192)) + r242_row[:, None]) % $(192))), dtype=tl.float32)
      r242_right = tl.load($(candidate_vmul_37_ptr) + (((((((r242_group % $(8)) * $(432)) + (r242_lane[:, None] * $(36)) + r242_col[None, :]) / $(432)) % $(8)) * $(432)) + ((((((r242_group % $(8)) * $(432)) + (r242_lane[:, None] * $(36)) + r242_col[None, :]) / $(36)) % $(12)) * $(36)) + ((((r242_group % $(8)) * $(432)) + (r242_lane[:, None] * $(36)) + r242_col[None, :]) % $(36))), dtype=tl.float32)
      r242_value = tl.dot(r242_left, r242_right)
      r242_value = r242_value.to(tl.float32)
      tl.store($(candidate_vbmm_44_ptr) + ((r242_group * $(6912)) + (r242_row[:, None] * $(36)) + r242_col[None, :]), r242_value)
    }
    r243_row = tl.arange($(12))
    r243_col = tl.arange($(192))
    r243_lane = tl.arange($(36))
    for r243_group in range($(8)) {
      r243_left = tl.load($(candidate_vmul_37_ptr) + (((((((r243_group % $(8)) * $(432)) + (r243_row[:, None] * $(36)) + r243_lane[None, :]) / $(432)) % $(8)) * $(432)) + ((((((r243_group % $(8)) * $(432)) + (r243_row[:, None] * $(36)) + r243_lane[None, :]) / $(36)) % $(12)) * $(36)) + ((((r243_group % $(8)) * $(432)) + (r243_row[:, None] * $(36)) + r243_lane[None, :]) % $(36))), dtype=tl.float32)
      r243_right = tl.load($(candidate_vto_copy_57_ptr) + (((((((r243_group % $(8)) * $(6912)) + (r243_col[None, :] * $(36)) + r243_lane[:, None]) / $(6912)) % $(8)) * $(6912)) + (((((r243_group % $(8)) * $(6912)) + (r243_col[None, :] * $(36)) + r243_lane[:, None]) % $(36)) * $(192)) + (((((r243_group % $(8)) * $(6912)) + (r243_col[None, :] * $(36)) + r243_lane[:, None]) / $(36)) % $(192))), dtype=tl.float32)
      r243_value = tl.dot(r243_left, r243_right)
      r243_value = r243_value.to(tl.float32)
      tl.store($(candidate_vbmm_45_ptr) + ((r243_group * $(2304)) + (r243_row[:, None] * $(192)) + r243_col[None, :]), r243_value)
    }
    r244_z = tl.arange($(131072))
    r244_value = tl.where(((((r244_z / $(1024)) % $(128)) >= $(0)) & (((r244_z / $(1024)) % $(128)) < $(36)) & ((((r244_z / $(1024)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vbmm_42_ptr) + ((((((((r244_z / $(128)) % $(8)) * $(4608)) + ((((r244_z / $(1024)) % $(128)) - $(0)) * $(128)) + (r244_z % $(128))) / $(4608)) % $(8)) * $(4608)) + ((((((r244_z / $(128)) % $(8)) * $(36)) + (((r244_z / $(1024)) % $(128)) - $(0))) % $(36)) * $(128)) + (r244_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r244_value = r244_value.to(tl.bfloat16)
    r244_value = tl.broadcast_to(r244_value, [$(131072)])
    tl.store($(candidate_vslice_backward_6_ptr) + r244_z, r244_value)
    r245_z = tl.arange($(131072))
    r245_value = (tl.load($(candidate_vadd_20_ptr) + ((((r245_z / $(1024)) % $(128)) * $(1024)) + (((r245_z / $(128)) % $(8)) * $(128)) + (r245_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_6_ptr) + ((((r245_z / $(1024)) % $(128)) * $(1024)) + (((r245_z / $(128)) % $(8)) * $(128)) + (r245_z % $(128))), dtype=tl.bfloat16))
    r245_value = r245_value.to(tl.bfloat16)
    r245_value = tl.broadcast_to(r245_value, [$(131072)])
    tl.store($(candidate_vadd_23_ptr) + r245_z, r245_value)
    r246_z = tl.arange($(55296))
    r246_value = tl.load($(candidate_vbmm_44_ptr) + ((((r246_z / $(6912)) % $(8)) * $(6912)) + ((r246_z % $(192)) * $(36)) + ((r246_z / $(192)) % $(36))), dtype=tl.float32)
    r246_value = r246_value.to(tl.bfloat16)
    r246_value = tl.broadcast_to(r246_value, [$(55296)])
    tl.store($(candidate_vto_copy_72_ptr) + r246_z, r246_value)
    r247_z = tl.arange($(196608))
    r247_value = tl.where(((((r247_z / $(1536)) % $(128)) >= $(0)) & (((r247_z / $(1536)) % $(128)) < $(36)) & ((((r247_z / $(1536)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vto_copy_72_ptr) + ((((r247_z / $(192)) % $(8)) * $(6912)) + (((((r247_z / $(1536)) % $(128)) - $(0)) % $(36)) * $(192)) + (r247_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r247_value = r247_value.to(tl.bfloat16)
    r247_value = tl.broadcast_to(r247_value, [$(196608)])
    tl.store($(candidate_vslice_backward_7_ptr) + r247_z, r247_value)
    r248_z = tl.arange($(196608))
    r248_value = (tl.load($(candidate_vadd_21_ptr) + ((((r248_z / $(1536)) % $(128)) * $(1536)) + (((r248_z / $(192)) % $(8)) * $(192)) + (r248_z % $(192))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_7_ptr) + ((((r248_z / $(1536)) % $(128)) * $(1536)) + (((r248_z / $(192)) % $(8)) * $(192)) + (r248_z % $(192))), dtype=tl.bfloat16))
    r248_value = r248_value.to(tl.bfloat16)
    r248_value = tl.broadcast_to(r248_value, [$(196608)])
    tl.store($(candidate_vadd_24_ptr) + r248_z, r248_value)
    r249_z = tl.arange($(18432))
    r249_value = tl.load($(candidate_vbmm_45_ptr) + ((((r249_z / $(2304)) % $(8)) * $(2304)) + (((r249_z / $(192)) % $(12)) * $(192)) + (r249_z % $(192))), dtype=tl.float32)
    r249_value = r249_value.to(tl.bfloat16)
    r249_value = tl.broadcast_to(r249_value, [$(18432)])
    tl.store($(candidate_vto_copy_73_ptr) + r249_z, r249_value)
    r250_z = tl.arange($(49152))
    r250_value = tl.where(((((r250_z / $(1536)) % $(32)) >= $(0)) & (((r250_z / $(1536)) % $(32)) < $(12)) & ((((r250_z / $(1536)) % $(32)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vto_copy_73_ptr) + ((((r250_z / $(192)) % $(8)) * $(2304)) + (((((r250_z / $(1536)) % $(32)) - $(0)) % $(12)) * $(192)) + (r250_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r250_value = r250_value.to(tl.bfloat16)
    r250_value = tl.broadcast_to(r250_value, [$(49152)])
    tl.store($(candidate_vslice_backward_8_ptr) + r250_z, r250_value)
    r251_z = tl.arange($(49152))
    r251_value = (tl.load($(candidate_vadd_22_ptr) + ((((r251_z / $(1536)) % $(32)) * $(1536)) + (((r251_z / $(192)) % $(8)) * $(192)) + (r251_z % $(192))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_8_ptr) + ((((r251_z / $(1536)) % $(32)) * $(1536)) + (((r251_z / $(192)) % $(8)) * $(192)) + (r251_z % $(192))), dtype=tl.bfloat16))
    r251_value = r251_value.to(tl.bfloat16)
    r251_value = tl.broadcast_to(r251_value, [$(49152)])
    tl.store($(candidate_vadd_25_ptr) + r251_z, r251_value)
    r252_z = tl.arange($(12288))
    r252_value = tl.cast(0.0, tl.bfloat16)
    r252_value = r252_value.to(tl.bfloat16)
    r252_value = tl.broadcast_to(r252_value, [$(12288)])
    tl.store($(candidate_vfill_ptr) + r252_z, r252_value)
    r253_z = tl.arange($(131072))
    r253_value = tl.where(((((r253_z / $(1024)) % $(128)) >= $(36)) & (((r253_z / $(1024)) % $(128)) < $(48)) & ((((r253_z / $(1024)) % $(128)) - $(36)) % $(1) == $(0))), tl.load($(candidate_vfill_ptr) + ((((((r253_z / $(1024)) % $(128)) - $(36)) % $(12)) * $(1024)) + (((r253_z / $(128)) % $(8)) * $(128)) + (r253_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vadd_23_ptr) + ((((r253_z / $(1024)) % $(128)) * $(1024)) + (((r253_z / $(128)) % $(8)) * $(128)) + (r253_z % $(128))), dtype=tl.bfloat16))
    r253_value = r253_value.to(tl.bfloat16)
    r253_value = tl.broadcast_to(r253_value, [$(131072)])
    tl.store($(candidate_vslice_scatter_ptr) + r253_z, r253_value)
    r254_z = tl.arange($(4096))
    r254_value = tl.cast(0.0, tl.bfloat16)
    r254_value = r254_value.to(tl.bfloat16)
    r254_value = tl.broadcast_to(r254_value, [$(4096)])
    tl.store($(candidate_vfill_1_ptr) + r254_z, r254_value)
    r255_z = tl.arange($(131072))
    r255_value = tl.where(((((r255_z / $(1024)) % $(128)) >= $(60)) & (((r255_z / $(1024)) % $(128)) < $(64)) & ((((r255_z / $(1024)) % $(128)) - $(60)) % $(1) == $(0))), tl.load($(candidate_vfill_1_ptr) + ((((((r255_z / $(1024)) % $(128)) - $(60)) % $(4)) * $(1024)) + (((r255_z / $(128)) % $(8)) * $(128)) + (r255_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vslice_scatter_ptr) + ((((r255_z / $(1024)) % $(128)) * $(1024)) + (((r255_z / $(128)) % $(8)) * $(128)) + (r255_z % $(128))), dtype=tl.bfloat16))
    r255_value = r255_value.to(tl.bfloat16)
    r255_value = tl.broadcast_to(r255_value, [$(131072)])
    tl.store($(candidate_vslice_scatter_1_ptr) + r255_z, r255_value)
    r256_z = tl.arange($(16384))
    r256_value = tl.cast(0.0, tl.bfloat16)
    r256_value = r256_value.to(tl.bfloat16)
    r256_value = tl.broadcast_to(r256_value, [$(16384)])
    tl.store($(candidate_vfill_2_ptr) + r256_z, r256_value)
    r257_z = tl.arange($(131072))
    r257_value = tl.where(((((r257_z / $(1024)) % $(128)) >= $(112)) & (((r257_z / $(1024)) % $(128)) < $(128)) & ((((r257_z / $(1024)) % $(128)) - $(112)) % $(1) == $(0))), tl.load($(candidate_vfill_2_ptr) + ((((((r257_z / $(1024)) % $(128)) - $(112)) % $(16)) * $(1024)) + (((r257_z / $(128)) % $(8)) * $(128)) + (r257_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vslice_scatter_1_ptr) + ((((r257_z / $(1024)) % $(128)) * $(1024)) + (((r257_z / $(128)) % $(8)) * $(128)) + (r257_z % $(128))), dtype=tl.bfloat16))
    r257_value = r257_value.to(tl.bfloat16)
    r257_value = tl.broadcast_to(r257_value, [$(131072)])
    tl.store($(candidate_vslice_scatter_2_ptr) + r257_z, r257_value)
    r258_z = tl.arange($(18432))
    r258_value = tl.cast(0.0, tl.bfloat16)
    r258_value = r258_value.to(tl.bfloat16)
    r258_value = tl.broadcast_to(r258_value, [$(18432)])
    tl.store($(candidate_vfill_3_ptr) + r258_z, r258_value)
    r259_z = tl.arange($(196608))
    r259_value = tl.where(((((r259_z / $(1536)) % $(128)) >= $(36)) & (((r259_z / $(1536)) % $(128)) < $(48)) & ((((r259_z / $(1536)) % $(128)) - $(36)) % $(1) == $(0))), tl.load($(candidate_vfill_3_ptr) + ((((((r259_z / $(1536)) % $(128)) - $(36)) % $(12)) * $(1536)) + (((r259_z / $(192)) % $(8)) * $(192)) + (r259_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vadd_24_ptr) + ((((r259_z / $(1536)) % $(128)) * $(1536)) + (((r259_z / $(192)) % $(8)) * $(192)) + (r259_z % $(192))), dtype=tl.bfloat16))
    r259_value = r259_value.to(tl.bfloat16)
    r259_value = tl.broadcast_to(r259_value, [$(196608)])
    tl.store($(candidate_vslice_scatter_3_ptr) + r259_z, r259_value)
    r260_z = tl.arange($(6144))
    r260_value = tl.cast(0.0, tl.bfloat16)
    r260_value = r260_value.to(tl.bfloat16)
    r260_value = tl.broadcast_to(r260_value, [$(6144)])
    tl.store($(candidate_vfill_4_ptr) + r260_z, r260_value)
    r261_z = tl.arange($(196608))
    r261_value = tl.where(((((r261_z / $(1536)) % $(128)) >= $(60)) & (((r261_z / $(1536)) % $(128)) < $(64)) & ((((r261_z / $(1536)) % $(128)) - $(60)) % $(1) == $(0))), tl.load($(candidate_vfill_4_ptr) + ((((((r261_z / $(1536)) % $(128)) - $(60)) % $(4)) * $(1536)) + (((r261_z / $(192)) % $(8)) * $(192)) + (r261_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vslice_scatter_3_ptr) + ((((r261_z / $(1536)) % $(128)) * $(1536)) + (((r261_z / $(192)) % $(8)) * $(192)) + (r261_z % $(192))), dtype=tl.bfloat16))
    r261_value = r261_value.to(tl.bfloat16)
    r261_value = tl.broadcast_to(r261_value, [$(196608)])
    tl.store($(candidate_vslice_scatter_4_ptr) + r261_z, r261_value)
    r262_z = tl.arange($(24576))
    r262_value = tl.cast(0.0, tl.bfloat16)
    r262_value = r262_value.to(tl.bfloat16)
    r262_value = tl.broadcast_to(r262_value, [$(24576)])
    tl.store($(candidate_vfill_5_ptr) + r262_z, r262_value)
    r263_z = tl.arange($(196608))
    r263_value = tl.where(((((r263_z / $(1536)) % $(128)) >= $(112)) & (((r263_z / $(1536)) % $(128)) < $(128)) & ((((r263_z / $(1536)) % $(128)) - $(112)) % $(1) == $(0))), tl.load($(candidate_vfill_5_ptr) + ((((((r263_z / $(1536)) % $(128)) - $(112)) % $(16)) * $(1536)) + (((r263_z / $(192)) % $(8)) * $(192)) + (r263_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vslice_scatter_4_ptr) + ((((r263_z / $(1536)) % $(128)) * $(1536)) + (((r263_z / $(192)) % $(8)) * $(192)) + (r263_z % $(192))), dtype=tl.bfloat16))
    r263_value = r263_value.to(tl.bfloat16)
    r263_value = tl.broadcast_to(r263_value, [$(196608)])
    tl.store($(candidate_vslice_scatter_5_ptr) + r263_z, r263_value)
    r264_z = tl.arange($(8192))
    r264_lane = tl.arange($(8))
    r264_value = tl.sum(tl.load($(candidate_vslice_scatter_5_ptr) + ((((r264_z[:, None] / $(64)) % $(128)) * $(1536)) + (r264_lane[None, :] * $(192)) + (r264_z[:, None] % $(64)) + $(128)), dtype=tl.bfloat16), axis=1)
    r264_value = r264_value.to(tl.bfloat16)
    r264_value = tl.broadcast_to(r264_value, [$(8192)])
    tl.store($(candidate_vsum_1_ptr) + r264_z, r264_value)
    r265_row = tl.arange($(128))
    r265_col = tl.arange($(256))
    r265_lane = tl.arange($(1024))
    for r265_group in range($(1)) {
      r265_left = tl.load($(candidate_vslice_scatter_2_ptr) + ((((((r265_row[:, None] * $(1024)) + r265_lane[None, :]) / $(1024)) % $(128)) * $(1024)) + (((((r265_row[:, None] * $(1024)) + r265_lane[None, :]) / $(128)) % $(8)) * $(128)) + (((r265_row[:, None] * $(1024)) + r265_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r265_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r265_col[None, :] * $(1024)) + r265_lane[:, None]) / $(128)) % $(8)) * $(65536)) + ((((r265_col[None, :] * $(1024)) + r265_lane[:, None]) % $(128)) * $(256)) + ((((r265_col[None, :] * $(1024)) + r265_lane[:, None]) / $(1024)) % $(256)) + $(32768)), dtype=tl.bfloat16)
      r265_value = tl.dot(r265_left, r265_right)
      r265_value = r265_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_46_ptr) + ((r265_group * $(32768)) + (r265_row[:, None] * $(256)) + r265_col[None, :]), r265_value)
    }
    r266_row = tl.arange($(128))
    r266_col = tl.arange($(256))
    r266_lane = tl.arange($(1024))
    for r266_group in range($(1)) {
      r266_left = tl.load($(candidate_vslice_scatter_5_ptr) + ((((((r266_row[:, None] * $(1024)) + r266_lane[None, :]) / $(1024)) % $(128)) * $(1536)) + (((((r266_row[:, None] * $(1024)) + r266_lane[None, :]) / $(128)) % $(8)) * $(192)) + (((r266_row[:, None] * $(1024)) + r266_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r266_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r266_col[None, :] * $(1024)) + r266_lane[:, None]) / $(128)) % $(8)) * $(65536)) + ((((r266_col[None, :] * $(1024)) + r266_lane[:, None]) % $(128)) * $(256)) + ((((r266_col[None, :] * $(1024)) + r266_lane[:, None]) / $(1024)) % $(256))), dtype=tl.bfloat16)
      r266_value = tl.dot(r266_left, r266_right)
      r266_value = r266_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_47_ptr) + ((r266_group * $(32768)) + (r266_row[:, None] * $(256)) + r266_col[None, :]), r266_value)
    }
    r267_z = tl.arange($(32768))
    r267_value = (tl.load($(candidate_vbmm_46_ptr) + ((((r267_z / $(256)) % $(128)) * $(256)) + (r267_z % $(256))), dtype=tl.bfloat16) + tl.load($(candidate_vbmm_47_ptr) + ((((r267_z / $(256)) % $(128)) * $(256)) + (r267_z % $(256))), dtype=tl.bfloat16))
    r267_value = r267_value.to(tl.bfloat16)
    r267_value = tl.broadcast_to(r267_value, [$(32768)])
    tl.store($(candidate_vadd_26_ptr) + r267_z, r267_value)
    r268_z = tl.arange($(32768))
    r268_value = tl.where(($(0) == $(0)), tl.load($(candidate_vadd_26_ptr) + ((((r268_z / $(256)) % $(128)) * $(256)) + (r268_z % $(256))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r268_value = r268_value.to(tl.bfloat16)
    r268_value = tl.broadcast_to(r268_value, [$(32768)])
    tl.store($(candidate_vselect_backward_ptr) + r268_z, r268_value)
    r269_z = tl.arange($(32768))
    r269_value = tl.where(((((r269_z / $(256)) % $(128)) >= $(0)) & (((r269_z / $(256)) % $(128)) < $(128)) & ((((r269_z / $(256)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vselect_backward_ptr) + ((((((r269_z / $(256)) % $(128)) - $(0)) % $(128)) * $(256)) + (r269_z % $(256))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r269_value = r269_value.to(tl.bfloat16)
    r269_value = tl.broadcast_to(r269_value, [$(32768)])
    tl.store($(candidate_vslice_backward_9_ptr) + r269_z, r269_value)
    r270_z = tl.arange($(8192))
    r270_value = tl.where(((((r270_z / $(64)) % $(128)) >= $(0)) & (((r270_z / $(64)) % $(128)) < $(128)) & ((((r270_z / $(64)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vsum_1_ptr) + ((((((r270_z / $(64)) % $(128)) - $(0)) % $(128)) * $(64)) + (r270_z % $(64))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r270_value = r270_value.to(tl.bfloat16)
    r270_value = tl.broadcast_to(r270_value, [$(8192)])
    tl.store($(candidate_vslice_backward_10_ptr) + r270_z, r270_value)
    r271_z = tl.arange($(131072))
    r271_value = tl.where(((((r271_z / $(1024)) % $(128)) >= $(96)) & (((r271_z / $(1024)) % $(128)) < $(128)) & ((((r271_z / $(1024)) % $(128)) - $(96)) % $(1) == $(0))), tl.load($(candidate_vadd_25_ptr) + ((((((r271_z / $(1024)) % $(128)) - $(96)) % $(32)) * $(1536)) + (((r271_z / $(128)) % $(8)) * $(192)) + (r271_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r271_value = r271_value.to(tl.bfloat16)
    r271_value = tl.broadcast_to(r271_value, [$(131072)])
    tl.store($(candidate_vslice_backward_11_ptr) + r271_z, r271_value)
    r272_z = tl.arange($(65536))
    r272_value = tl.where(((((r272_z / $(512)) % $(128)) >= $(96)) & (((r272_z / $(512)) % $(128)) < $(128)) & ((((r272_z / $(512)) % $(128)) - $(96)) % $(1) == $(0))), tl.load($(candidate_vadd_25_ptr) + ((((((r272_z / $(512)) % $(128)) - $(96)) % $(32)) * $(1536)) + (((r272_z / $(64)) % $(8)) * $(192)) + (r272_z % $(64)) + $(128)), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r272_value = r272_value.to(tl.bfloat16)
    r272_value = tl.broadcast_to(r272_value, [$(65536)])
    tl.store($(candidate_vslice_backward_12_ptr) + r272_z, r272_value)
    r273_row = tl.arange($(32))
    r273_col = tl.arange($(128))
    r273_lane = tl.arange($(16))
    for r273_group in range($(8)) {
      r273_left = tl.load($(candidate_vto_copy_55_ptr) + (((((((r273_group % $(8)) * $(512)) + (r273_lane[None, :] * $(32)) + r273_row[:, None]) / $(512)) % $(8)) * $(512)) + ((((((r273_group % $(8)) * $(512)) + (r273_lane[None, :] * $(32)) + r273_row[:, None]) / $(32)) % $(16)) * $(32)) + ((((r273_group % $(8)) * $(512)) + (r273_lane[None, :] * $(32)) + r273_row[:, None]) % $(32))), dtype=tl.bfloat16)
      r273_right = tl.load($(candidate_vmm_4_ptr) + (((((((r273_group % $(8)) * $(2048)) + (r273_lane[:, None] * $(128)) + r273_col[None, :]) / $(128)) % $(16)) * $(1024)) + ((((((r273_group % $(8)) * $(2048)) + (r273_lane[:, None] * $(128)) + r273_col[None, :]) / $(2048)) % $(8)) * $(128)) + ((((r273_group % $(8)) * $(2048)) + (r273_lane[:, None] * $(128)) + r273_col[None, :]) % $(128)) + $(81920)), dtype=tl.bfloat16)
      r273_value = tl.dot(r273_left, r273_right)
      r273_value = r273_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_48_ptr) + ((r273_group * $(4096)) + (r273_row[:, None] * $(128)) + r273_col[None, :]), r273_value)
    }
    r274_row = tl.arange($(16))
    r274_col = tl.arange($(32))
    r274_lane = tl.arange($(128))
    for r274_group in range($(8)) {
      r274_left = tl.load($(candidate_vmm_4_ptr) + (((((((r274_group % $(8)) * $(2048)) + (r274_row[:, None] * $(128)) + r274_lane[None, :]) / $(128)) % $(16)) * $(1024)) + ((((((r274_group % $(8)) * $(2048)) + (r274_row[:, None] * $(128)) + r274_lane[None, :]) / $(2048)) % $(8)) * $(128)) + ((((r274_group % $(8)) * $(2048)) + (r274_row[:, None] * $(128)) + r274_lane[None, :]) % $(128)) + $(81920)), dtype=tl.bfloat16)
      r274_right = tl.load($(candidate_vbmm_19_ptr) + (((((((r274_group % $(8)) * $(4096)) + (r274_col[None, :] * $(128)) + r274_lane[:, None]) / $(128)) % $(32)) * $(1024)) + ((((((r274_group % $(8)) * $(4096)) + (r274_col[None, :] * $(128)) + r274_lane[:, None]) / $(4096)) % $(8)) * $(128)) + ((((r274_group % $(8)) * $(4096)) + (r274_col[None, :] * $(128)) + r274_lane[:, None]) % $(128)) + $(65536)), dtype=tl.bfloat16)
      r274_value = tl.dot(r274_left, r274_right)
      r274_value = r274_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_49_ptr) + ((r274_group * $(512)) + (r274_row[:, None] * $(32)) + r274_col[None, :]), r274_value)
    }
    r275_z = tl.arange($(4096))
    r275_value = tl.load($(candidate_vbmm_49_ptr) + ((((r275_z / $(512)) % $(8)) * $(512)) + (((r275_z / $(32)) % $(16)) * $(32)) + (r275_z % $(32))), dtype=tl.bfloat16)
    r275_value = r275_value.to(tl.float32)
    r275_value = tl.broadcast_to(r275_value, [$(4096)])
    tl.store($(candidate_vto_copy_74_ptr) + r275_z, r275_value)
    r276_z = tl.arange($(4096))
    r276_lane = tl.arange($(32))
    r276_row_values = tl.load($(candidate_vsoftmax_8_ptr) + ((((r276_z[:, None] / $(512)) % $(8)) * $(512)) + (((r276_z[:, None] / $(32)) % $(16)) * $(32)) + (r276_lane[None, :] % $(32))), dtype=tl.float32)
    r276_row_grad = tl.load($(candidate_vto_copy_74_ptr) + ((((r276_z[:, None] / $(512)) % $(8)) * $(512)) + (((r276_z[:, None] / $(32)) % $(16)) * $(32)) + (r276_lane[None, :] % $(32))), dtype=tl.float32)
    r276_weighted = r276_row_grad * r276_row_values
    r276_dot_grad = tl.sum(r276_weighted, axis=1)
    r276_value = (tl.load($(candidate_vto_copy_74_ptr) + ((((r276_z / $(512)) % $(8)) * $(512)) + (((r276_z / $(32)) % $(16)) * $(32)) + (r276_z % $(32))), dtype=tl.float32) - r276_dot_grad) * tl.load($(candidate_vsoftmax_8_ptr) + ((((r276_z / $(512)) % $(8)) * $(512)) + (((r276_z / $(32)) % $(16)) * $(32)) + (r276_z % $(32))), dtype=tl.float32)
    r276_value = r276_value.to(tl.float32)
    r276_value = tl.broadcast_to(r276_value, [$(4096)])
    tl.store($(candidate_vsoftmax_backward_data_3_ptr) + r276_z, r276_value)
    r277_z = tl.arange($(4096))
    r277_value = tl.where(tl.load($(candidate_vgt_8_ptr) + ((((r277_z / $(32)) % $(16)) * $(32)) + (r277_z % $(32))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_3_ptr) + ((((r277_z / $(512)) % $(8)) * $(512)) + (((r277_z / $(32)) % $(16)) * $(32)) + (r277_z % $(32))), dtype=tl.float32))
    r277_value = r277_value.to(tl.float32)
    r277_value = tl.broadcast_to(r277_value, [$(4096)])
    tl.store($(candidate_vmasked_fill_15_ptr) + r277_z, r277_value)
    r278_z = tl.arange($(4096))
    r278_value = (tl.load($(candidate_vmasked_fill_15_ptr) + ((((r278_z / $(512)) % $(8)) * $(512)) + (((r278_z / $(32)) % $(16)) * $(32)) + (r278_z % $(32))), dtype=tl.float32) * 0.07216878364870322)
    r278_value = r278_value.to(tl.float32)
    r278_value = tl.broadcast_to(r278_value, [$(4096)])
    tl.store($(candidate_vmul_38_ptr) + r278_z, r278_value)
    r279_row = tl.arange($(192))
    r279_col = tl.arange($(32))
    r279_lane = tl.arange($(16))
    for r279_group in range($(8)) {
      r279_left = tl.load($(candidate_vto_copy_53_ptr) + (((((((r279_group % $(8)) * $(3072)) + (r279_lane[None, :] * $(192)) + r279_row[:, None]) / $(3072)) % $(8)) * $(3072)) + ((((((r279_group % $(8)) * $(3072)) + (r279_lane[None, :] * $(192)) + r279_row[:, None]) / $(192)) % $(16)) * $(192)) + ((((r279_group % $(8)) * $(3072)) + (r279_lane[None, :] * $(192)) + r279_row[:, None]) % $(192))), dtype=tl.float32)
      r279_right = tl.load($(candidate_vmul_38_ptr) + (((((((r279_group % $(8)) * $(512)) + (r279_lane[:, None] * $(32)) + r279_col[None, :]) / $(512)) % $(8)) * $(512)) + ((((((r279_group % $(8)) * $(512)) + (r279_lane[:, None] * $(32)) + r279_col[None, :]) / $(32)) % $(16)) * $(32)) + ((((r279_group % $(8)) * $(512)) + (r279_lane[:, None] * $(32)) + r279_col[None, :]) % $(32))), dtype=tl.float32)
      r279_value = tl.dot(r279_left, r279_right)
      r279_value = r279_value.to(tl.float32)
      tl.store($(candidate_vbmm_50_ptr) + ((r279_group * $(6144)) + (r279_row[:, None] * $(32)) + r279_col[None, :]), r279_value)
    }
    r280_row = tl.arange($(16))
    r280_col = tl.arange($(192))
    r280_lane = tl.arange($(32))
    for r280_group in range($(8)) {
      r280_left = tl.load($(candidate_vmul_38_ptr) + (((((((r280_group % $(8)) * $(512)) + (r280_row[:, None] * $(32)) + r280_lane[None, :]) / $(512)) % $(8)) * $(512)) + ((((((r280_group % $(8)) * $(512)) + (r280_row[:, None] * $(32)) + r280_lane[None, :]) / $(32)) % $(16)) * $(32)) + ((((r280_group % $(8)) * $(512)) + (r280_row[:, None] * $(32)) + r280_lane[None, :]) % $(32))), dtype=tl.float32)
      r280_right = tl.load($(candidate_vto_copy_54_ptr) + (((((((r280_group % $(8)) * $(6144)) + (r280_col[None, :] * $(32)) + r280_lane[:, None]) / $(6144)) % $(8)) * $(6144)) + (((((r280_group % $(8)) * $(6144)) + (r280_col[None, :] * $(32)) + r280_lane[:, None]) % $(32)) * $(192)) + (((((r280_group % $(8)) * $(6144)) + (r280_col[None, :] * $(32)) + r280_lane[:, None]) / $(32)) % $(192))), dtype=tl.float32)
      r280_value = tl.dot(r280_left, r280_right)
      r280_value = r280_value.to(tl.float32)
      tl.store($(candidate_vbmm_51_ptr) + ((r280_group * $(3072)) + (r280_row[:, None] * $(192)) + r280_col[None, :]), r280_value)
    }
    r281_z = tl.arange($(131072))
    r281_value = tl.where(((((r281_z / $(1024)) % $(128)) >= $(64)) & (((r281_z / $(1024)) % $(128)) < $(96)) & ((((r281_z / $(1024)) % $(128)) - $(64)) % $(1) == $(0))), tl.load($(candidate_vbmm_48_ptr) + ((((((((r281_z / $(128)) % $(8)) * $(4096)) + ((((r281_z / $(1024)) % $(128)) - $(64)) * $(128)) + (r281_z % $(128))) / $(4096)) % $(8)) * $(4096)) + ((((((r281_z / $(128)) % $(8)) * $(32)) + (((r281_z / $(1024)) % $(128)) - $(64))) % $(32)) * $(128)) + (r281_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r281_value = r281_value.to(tl.bfloat16)
    r281_value = tl.broadcast_to(r281_value, [$(131072)])
    tl.store($(candidate_vslice_backward_13_ptr) + r281_z, r281_value)
    r282_z = tl.arange($(49152))
    r282_value = tl.load($(candidate_vbmm_50_ptr) + ((((r282_z / $(6144)) % $(8)) * $(6144)) + ((r282_z % $(192)) * $(32)) + ((r282_z / $(192)) % $(32))), dtype=tl.float32)
    r282_value = r282_value.to(tl.bfloat16)
    r282_value = tl.broadcast_to(r282_value, [$(49152)])
    tl.store($(candidate_vto_copy_75_ptr) + r282_z, r282_value)
    r283_z = tl.arange($(196608))
    r283_value = tl.where(((((r283_z / $(1536)) % $(128)) >= $(64)) & (((r283_z / $(1536)) % $(128)) < $(96)) & ((((r283_z / $(1536)) % $(128)) - $(64)) % $(1) == $(0))), tl.load($(candidate_vto_copy_75_ptr) + ((((r283_z / $(192)) % $(8)) * $(6144)) + (((((r283_z / $(1536)) % $(128)) - $(64)) % $(32)) * $(192)) + (r283_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r283_value = r283_value.to(tl.bfloat16)
    r283_value = tl.broadcast_to(r283_value, [$(196608)])
    tl.store($(candidate_vslice_backward_14_ptr) + r283_z, r283_value)
    r284_z = tl.arange($(24576))
    r284_value = tl.load($(candidate_vbmm_51_ptr) + ((((r284_z / $(3072)) % $(8)) * $(3072)) + (((r284_z / $(192)) % $(16)) * $(192)) + (r284_z % $(192))), dtype=tl.float32)
    r284_value = r284_value.to(tl.bfloat16)
    r284_value = tl.broadcast_to(r284_value, [$(24576)])
    tl.store($(candidate_vto_copy_76_ptr) + r284_z, r284_value)
    r285_z = tl.arange($(49152))
    r285_value = tl.where(((((r285_z / $(1536)) % $(32)) >= $(16)) & (((r285_z / $(1536)) % $(32)) < $(32)) & ((((r285_z / $(1536)) % $(32)) - $(16)) % $(1) == $(0))), tl.load($(candidate_vto_copy_76_ptr) + ((((r285_z / $(192)) % $(8)) * $(3072)) + (((((r285_z / $(1536)) % $(32)) - $(16)) % $(16)) * $(192)) + (r285_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r285_value = r285_value.to(tl.bfloat16)
    r285_value = tl.broadcast_to(r285_value, [$(49152)])
    tl.store($(candidate_vslice_backward_15_ptr) + r285_z, r285_value)
    r286_row = tl.arange($(8))
    r286_col = tl.arange($(128))
    r286_lane = tl.arange($(4))
    for r286_group in range($(8)) {
      r286_left = tl.load($(candidate_vto_copy_52_ptr) + (((((((r286_group % $(8)) * $(32)) + (r286_lane[None, :] * $(8)) + r286_row[:, None]) / $(32)) % $(8)) * $(32)) + ((((((r286_group % $(8)) * $(32)) + (r286_lane[None, :] * $(8)) + r286_row[:, None]) / $(8)) % $(4)) * $(8)) + ((((r286_group % $(8)) * $(32)) + (r286_lane[None, :] * $(8)) + r286_row[:, None]) % $(8))), dtype=tl.bfloat16)
      r286_right = tl.load($(candidate_vmm_4_ptr) + (((((((r286_group % $(8)) * $(512)) + (r286_lane[:, None] * $(128)) + r286_col[None, :]) / $(128)) % $(4)) * $(1024)) + ((((((r286_group % $(8)) * $(512)) + (r286_lane[:, None] * $(128)) + r286_col[None, :]) / $(512)) % $(8)) * $(128)) + ((((r286_group % $(8)) * $(512)) + (r286_lane[:, None] * $(128)) + r286_col[None, :]) % $(128)) + $(77824)), dtype=tl.bfloat16)
      r286_value = tl.dot(r286_left, r286_right)
      r286_value = r286_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_52_ptr) + ((r286_group * $(1024)) + (r286_row[:, None] * $(128)) + r286_col[None, :]), r286_value)
    }
    r287_row = tl.arange($(4))
    r287_col = tl.arange($(8))
    r287_lane = tl.arange($(128))
    for r287_group in range($(8)) {
      r287_left = tl.load($(candidate_vmm_4_ptr) + (((((((r287_group % $(8)) * $(512)) + (r287_row[:, None] * $(128)) + r287_lane[None, :]) / $(128)) % $(4)) * $(1024)) + ((((((r287_group % $(8)) * $(512)) + (r287_row[:, None] * $(128)) + r287_lane[None, :]) / $(512)) % $(8)) * $(128)) + ((((r287_group % $(8)) * $(512)) + (r287_row[:, None] * $(128)) + r287_lane[None, :]) % $(128)) + $(77824)), dtype=tl.bfloat16)
      r287_right = tl.load($(candidate_vbmm_19_ptr) + (((((((r287_group % $(8)) * $(1024)) + (r287_col[None, :] * $(128)) + r287_lane[:, None]) / $(128)) % $(8)) * $(1024)) + ((((((r287_group % $(8)) * $(1024)) + (r287_col[None, :] * $(128)) + r287_lane[:, None]) / $(1024)) % $(8)) * $(128)) + ((((r287_group % $(8)) * $(1024)) + (r287_col[None, :] * $(128)) + r287_lane[:, None]) % $(128)) + $(49152)), dtype=tl.bfloat16)
      r287_value = tl.dot(r287_left, r287_right)
      r287_value = r287_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_53_ptr) + ((r287_group * $(32)) + (r287_row[:, None] * $(8)) + r287_col[None, :]), r287_value)
    }
    r288_z = tl.arange($(256))
    r288_value = tl.load($(candidate_vbmm_53_ptr) + ((((r288_z / $(32)) % $(8)) * $(32)) + (((r288_z / $(8)) % $(4)) * $(8)) + (r288_z % $(8))), dtype=tl.bfloat16)
    r288_value = r288_value.to(tl.float32)
    r288_value = tl.broadcast_to(r288_value, [$(256)])
    tl.store($(candidate_vto_copy_77_ptr) + r288_z, r288_value)
    r289_z = tl.arange($(256))
    r289_lane = tl.arange($(8))
    r289_row_values = tl.load($(candidate_vsoftmax_7_ptr) + ((((r289_z[:, None] / $(32)) % $(8)) * $(32)) + (((r289_z[:, None] / $(8)) % $(4)) * $(8)) + (r289_lane[None, :] % $(8))), dtype=tl.float32)
    r289_row_grad = tl.load($(candidate_vto_copy_77_ptr) + ((((r289_z[:, None] / $(32)) % $(8)) * $(32)) + (((r289_z[:, None] / $(8)) % $(4)) * $(8)) + (r289_lane[None, :] % $(8))), dtype=tl.float32)
    r289_weighted = r289_row_grad * r289_row_values
    r289_dot_grad = tl.sum(r289_weighted, axis=1)
    r289_value = (tl.load($(candidate_vto_copy_77_ptr) + ((((r289_z / $(32)) % $(8)) * $(32)) + (((r289_z / $(8)) % $(4)) * $(8)) + (r289_z % $(8))), dtype=tl.float32) - r289_dot_grad) * tl.load($(candidate_vsoftmax_7_ptr) + ((((r289_z / $(32)) % $(8)) * $(32)) + (((r289_z / $(8)) % $(4)) * $(8)) + (r289_z % $(8))), dtype=tl.float32)
    r289_value = r289_value.to(tl.float32)
    r289_value = tl.broadcast_to(r289_value, [$(256)])
    tl.store($(candidate_vsoftmax_backward_data_4_ptr) + r289_z, r289_value)
    r290_z = tl.arange($(256))
    r290_value = tl.where(tl.load($(candidate_vgt_7_ptr) + ((((r290_z / $(8)) % $(4)) * $(8)) + (r290_z % $(8))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_4_ptr) + ((((r290_z / $(32)) % $(8)) * $(32)) + (((r290_z / $(8)) % $(4)) * $(8)) + (r290_z % $(8))), dtype=tl.float32))
    r290_value = r290_value.to(tl.float32)
    r290_value = tl.broadcast_to(r290_value, [$(256)])
    tl.store($(candidate_vmasked_fill_16_ptr) + r290_z, r290_value)
    r291_z = tl.arange($(256))
    r291_value = (tl.load($(candidate_vmasked_fill_16_ptr) + ((((r291_z / $(32)) % $(8)) * $(32)) + (((r291_z / $(8)) % $(4)) * $(8)) + (r291_z % $(8))), dtype=tl.float32) * 0.07216878364870322)
    r291_value = r291_value.to(tl.float32)
    r291_value = tl.broadcast_to(r291_value, [$(256)])
    tl.store($(candidate_vmul_39_ptr) + r291_z, r291_value)
    r292_row = tl.arange($(192))
    r292_col = tl.arange($(8))
    r292_lane = tl.arange($(4))
    for r292_group in range($(8)) {
      r292_left = tl.load($(candidate_vto_copy_50_ptr) + (((((((r292_group % $(8)) * $(768)) + (r292_lane[None, :] * $(192)) + r292_row[:, None]) / $(768)) % $(8)) * $(768)) + ((((((r292_group % $(8)) * $(768)) + (r292_lane[None, :] * $(192)) + r292_row[:, None]) / $(192)) % $(4)) * $(192)) + ((((r292_group % $(8)) * $(768)) + (r292_lane[None, :] * $(192)) + r292_row[:, None]) % $(192))), dtype=tl.float32)
      r292_right = tl.load($(candidate_vmul_39_ptr) + (((((((r292_group % $(8)) * $(32)) + (r292_lane[:, None] * $(8)) + r292_col[None, :]) / $(32)) % $(8)) * $(32)) + ((((((r292_group % $(8)) * $(32)) + (r292_lane[:, None] * $(8)) + r292_col[None, :]) / $(8)) % $(4)) * $(8)) + ((((r292_group % $(8)) * $(32)) + (r292_lane[:, None] * $(8)) + r292_col[None, :]) % $(8))), dtype=tl.float32)
      r292_value = tl.dot(r292_left, r292_right)
      r292_value = r292_value.to(tl.float32)
      tl.store($(candidate_vbmm_54_ptr) + ((r292_group * $(1536)) + (r292_row[:, None] * $(8)) + r292_col[None, :]), r292_value)
    }
    r293_row = tl.arange($(4))
    r293_col = tl.arange($(192))
    r293_lane = tl.arange($(8))
    for r293_group in range($(8)) {
      r293_left = tl.load($(candidate_vmul_39_ptr) + (((((((r293_group % $(8)) * $(32)) + (r293_row[:, None] * $(8)) + r293_lane[None, :]) / $(32)) % $(8)) * $(32)) + ((((((r293_group % $(8)) * $(32)) + (r293_row[:, None] * $(8)) + r293_lane[None, :]) / $(8)) % $(4)) * $(8)) + ((((r293_group % $(8)) * $(32)) + (r293_row[:, None] * $(8)) + r293_lane[None, :]) % $(8))), dtype=tl.float32)
      r293_right = tl.load($(candidate_vto_copy_51_ptr) + (((((((r293_group % $(8)) * $(1536)) + (r293_col[None, :] * $(8)) + r293_lane[:, None]) / $(1536)) % $(8)) * $(1536)) + (((((r293_group % $(8)) * $(1536)) + (r293_col[None, :] * $(8)) + r293_lane[:, None]) % $(8)) * $(192)) + (((((r293_group % $(8)) * $(1536)) + (r293_col[None, :] * $(8)) + r293_lane[:, None]) / $(8)) % $(192))), dtype=tl.float32)
      r293_value = tl.dot(r293_left, r293_right)
      r293_value = r293_value.to(tl.float32)
      tl.store($(candidate_vbmm_55_ptr) + ((r293_group * $(768)) + (r293_row[:, None] * $(192)) + r293_col[None, :]), r293_value)
    }
    r294_z = tl.arange($(131072))
    r294_value = tl.where(((((r294_z / $(1024)) % $(128)) >= $(48)) & (((r294_z / $(1024)) % $(128)) < $(56)) & ((((r294_z / $(1024)) % $(128)) - $(48)) % $(1) == $(0))), tl.load($(candidate_vbmm_52_ptr) + ((((((((r294_z / $(128)) % $(8)) * $(1024)) + ((((r294_z / $(1024)) % $(128)) - $(48)) * $(128)) + (r294_z % $(128))) / $(1024)) % $(8)) * $(1024)) + ((((((r294_z / $(128)) % $(8)) * $(8)) + (((r294_z / $(1024)) % $(128)) - $(48))) % $(8)) * $(128)) + (r294_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r294_value = r294_value.to(tl.bfloat16)
    r294_value = tl.broadcast_to(r294_value, [$(131072)])
    tl.store($(candidate_vslice_backward_16_ptr) + r294_z, r294_value)
    r295_z = tl.arange($(131072))
    r295_value = (tl.load($(candidate_vslice_backward_13_ptr) + ((((r295_z / $(1024)) % $(128)) * $(1024)) + (((r295_z / $(128)) % $(8)) * $(128)) + (r295_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_16_ptr) + ((((r295_z / $(1024)) % $(128)) * $(1024)) + (((r295_z / $(128)) % $(8)) * $(128)) + (r295_z % $(128))), dtype=tl.bfloat16))
    r295_value = r295_value.to(tl.bfloat16)
    r295_value = tl.broadcast_to(r295_value, [$(131072)])
    tl.store($(candidate_vadd_27_ptr) + r295_z, r295_value)
    r296_z = tl.arange($(12288))
    r296_value = tl.load($(candidate_vbmm_54_ptr) + ((((r296_z / $(1536)) % $(8)) * $(1536)) + ((r296_z % $(192)) * $(8)) + ((r296_z / $(192)) % $(8))), dtype=tl.float32)
    r296_value = r296_value.to(tl.bfloat16)
    r296_value = tl.broadcast_to(r296_value, [$(12288)])
    tl.store($(candidate_vto_copy_78_ptr) + r296_z, r296_value)
    r297_z = tl.arange($(196608))
    r297_value = tl.where(((((r297_z / $(1536)) % $(128)) >= $(48)) & (((r297_z / $(1536)) % $(128)) < $(56)) & ((((r297_z / $(1536)) % $(128)) - $(48)) % $(1) == $(0))), tl.load($(candidate_vto_copy_78_ptr) + ((((r297_z / $(192)) % $(8)) * $(1536)) + (((((r297_z / $(1536)) % $(128)) - $(48)) % $(8)) * $(192)) + (r297_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r297_value = r297_value.to(tl.bfloat16)
    r297_value = tl.broadcast_to(r297_value, [$(196608)])
    tl.store($(candidate_vslice_backward_17_ptr) + r297_z, r297_value)
    r298_z = tl.arange($(196608))
    r298_value = (tl.load($(candidate_vslice_backward_14_ptr) + ((((r298_z / $(1536)) % $(128)) * $(1536)) + (((r298_z / $(192)) % $(8)) * $(192)) + (r298_z % $(192))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_17_ptr) + ((((r298_z / $(1536)) % $(128)) * $(1536)) + (((r298_z / $(192)) % $(8)) * $(192)) + (r298_z % $(192))), dtype=tl.bfloat16))
    r298_value = r298_value.to(tl.bfloat16)
    r298_value = tl.broadcast_to(r298_value, [$(196608)])
    tl.store($(candidate_vadd_28_ptr) + r298_z, r298_value)
    r299_z = tl.arange($(6144))
    r299_value = tl.load($(candidate_vbmm_55_ptr) + ((((r299_z / $(768)) % $(8)) * $(768)) + (((r299_z / $(192)) % $(4)) * $(192)) + (r299_z % $(192))), dtype=tl.float32)
    r299_value = r299_value.to(tl.bfloat16)
    r299_value = tl.broadcast_to(r299_value, [$(6144)])
    tl.store($(candidate_vto_copy_79_ptr) + r299_z, r299_value)
    r300_z = tl.arange($(49152))
    r300_value = tl.where(((((r300_z / $(1536)) % $(32)) >= $(12)) & (((r300_z / $(1536)) % $(32)) < $(16)) & ((((r300_z / $(1536)) % $(32)) - $(12)) % $(1) == $(0))), tl.load($(candidate_vto_copy_79_ptr) + ((((r300_z / $(192)) % $(8)) * $(768)) + (((((r300_z / $(1536)) % $(32)) - $(12)) % $(4)) * $(192)) + (r300_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r300_value = r300_value.to(tl.bfloat16)
    r300_value = tl.broadcast_to(r300_value, [$(49152)])
    tl.store($(candidate_vslice_backward_18_ptr) + r300_z, r300_value)
    r301_z = tl.arange($(49152))
    r301_value = (tl.load($(candidate_vslice_backward_15_ptr) + ((((r301_z / $(1536)) % $(32)) * $(1536)) + (((r301_z / $(192)) % $(8)) * $(192)) + (r301_z % $(192))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_18_ptr) + ((((r301_z / $(1536)) % $(32)) * $(1536)) + (((r301_z / $(192)) % $(8)) * $(192)) + (r301_z % $(192))), dtype=tl.bfloat16))
    r301_value = r301_value.to(tl.bfloat16)
    r301_value = tl.broadcast_to(r301_value, [$(49152)])
    tl.store($(candidate_vadd_29_ptr) + r301_z, r301_value)
    r302_row = tl.arange($(24))
    r302_col = tl.arange($(128))
    r302_lane = tl.arange($(12))
    for r302_group in range($(8)) {
      r302_left = tl.load($(candidate_vto_copy_49_ptr) + (((((((r302_group % $(8)) * $(288)) + (r302_lane[None, :] * $(24)) + r302_row[:, None]) / $(288)) % $(8)) * $(288)) + ((((((r302_group % $(8)) * $(288)) + (r302_lane[None, :] * $(24)) + r302_row[:, None]) / $(24)) % $(12)) * $(24)) + ((((r302_group % $(8)) * $(288)) + (r302_lane[None, :] * $(24)) + r302_row[:, None]) % $(24))), dtype=tl.bfloat16)
      r302_right = tl.load($(candidate_vmm_4_ptr) + (((((((r302_group % $(8)) * $(1536)) + (r302_lane[:, None] * $(128)) + r302_col[None, :]) / $(128)) % $(12)) * $(1024)) + ((((((r302_group % $(8)) * $(1536)) + (r302_lane[:, None] * $(128)) + r302_col[None, :]) / $(1536)) % $(8)) * $(128)) + ((((r302_group % $(8)) * $(1536)) + (r302_lane[:, None] * $(128)) + r302_col[None, :]) % $(128)) + $(65536)), dtype=tl.bfloat16)
      r302_value = tl.dot(r302_left, r302_right)
      r302_value = r302_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_56_ptr) + ((r302_group * $(3072)) + (r302_row[:, None] * $(128)) + r302_col[None, :]), r302_value)
    }
    r303_row = tl.arange($(12))
    r303_col = tl.arange($(24))
    r303_lane = tl.arange($(128))
    for r303_group in range($(8)) {
      r303_left = tl.load($(candidate_vmm_4_ptr) + (((((((r303_group % $(8)) * $(1536)) + (r303_row[:, None] * $(128)) + r303_lane[None, :]) / $(128)) % $(12)) * $(1024)) + ((((((r303_group % $(8)) * $(1536)) + (r303_row[:, None] * $(128)) + r303_lane[None, :]) / $(1536)) % $(8)) * $(128)) + ((((r303_group % $(8)) * $(1536)) + (r303_row[:, None] * $(128)) + r303_lane[None, :]) % $(128)) + $(65536)), dtype=tl.bfloat16)
      r303_right = tl.load($(candidate_vbmm_19_ptr) + (((((((r303_group % $(8)) * $(3072)) + (r303_col[None, :] * $(128)) + r303_lane[:, None]) / $(128)) % $(24)) * $(1024)) + ((((((r303_group % $(8)) * $(3072)) + (r303_col[None, :] * $(128)) + r303_lane[:, None]) / $(3072)) % $(8)) * $(128)) + ((((r303_group % $(8)) * $(3072)) + (r303_col[None, :] * $(128)) + r303_lane[:, None]) % $(128))), dtype=tl.bfloat16)
      r303_value = tl.dot(r303_left, r303_right)
      r303_value = r303_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_57_ptr) + ((r303_group * $(288)) + (r303_row[:, None] * $(24)) + r303_col[None, :]), r303_value)
    }
    r304_z = tl.arange($(2304))
    r304_value = tl.load($(candidate_vbmm_57_ptr) + ((((r304_z / $(288)) % $(8)) * $(288)) + (((r304_z / $(24)) % $(12)) * $(24)) + (r304_z % $(24))), dtype=tl.bfloat16)
    r304_value = r304_value.to(tl.float32)
    r304_value = tl.broadcast_to(r304_value, [$(2304)])
    tl.store($(candidate_vto_copy_80_ptr) + r304_z, r304_value)
    r305_z = tl.arange($(2304))
    r305_lane = tl.arange($(24))
    r305_row_values = tl.load($(candidate_vsoftmax_6_ptr) + ((((r305_z[:, None] / $(288)) % $(8)) * $(288)) + (((r305_z[:, None] / $(24)) % $(12)) * $(24)) + (r305_lane[None, :] % $(24))), dtype=tl.float32)
    r305_row_grad = tl.load($(candidate_vto_copy_80_ptr) + ((((r305_z[:, None] / $(288)) % $(8)) * $(288)) + (((r305_z[:, None] / $(24)) % $(12)) * $(24)) + (r305_lane[None, :] % $(24))), dtype=tl.float32)
    r305_weighted = r305_row_grad * r305_row_values
    r305_dot_grad = tl.sum(r305_weighted, axis=1)
    r305_value = (tl.load($(candidate_vto_copy_80_ptr) + ((((r305_z / $(288)) % $(8)) * $(288)) + (((r305_z / $(24)) % $(12)) * $(24)) + (r305_z % $(24))), dtype=tl.float32) - r305_dot_grad) * tl.load($(candidate_vsoftmax_6_ptr) + ((((r305_z / $(288)) % $(8)) * $(288)) + (((r305_z / $(24)) % $(12)) * $(24)) + (r305_z % $(24))), dtype=tl.float32)
    r305_value = r305_value.to(tl.float32)
    r305_value = tl.broadcast_to(r305_value, [$(2304)])
    tl.store($(candidate_vsoftmax_backward_data_5_ptr) + r305_z, r305_value)
    r306_z = tl.arange($(2304))
    r306_value = tl.where(tl.load($(candidate_vgt_6_ptr) + ((((r306_z / $(24)) % $(12)) * $(24)) + (r306_z % $(24))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_5_ptr) + ((((r306_z / $(288)) % $(8)) * $(288)) + (((r306_z / $(24)) % $(12)) * $(24)) + (r306_z % $(24))), dtype=tl.float32))
    r306_value = r306_value.to(tl.float32)
    r306_value = tl.broadcast_to(r306_value, [$(2304)])
    tl.store($(candidate_vmasked_fill_17_ptr) + r306_z, r306_value)
    r307_z = tl.arange($(2304))
    r307_value = (tl.load($(candidate_vmasked_fill_17_ptr) + ((((r307_z / $(288)) % $(8)) * $(288)) + (((r307_z / $(24)) % $(12)) * $(24)) + (r307_z % $(24))), dtype=tl.float32) * 0.07216878364870322)
    r307_value = r307_value.to(tl.float32)
    r307_value = tl.broadcast_to(r307_value, [$(2304)])
    tl.store($(candidate_vmul_40_ptr) + r307_z, r307_value)
    r308_row = tl.arange($(192))
    r308_col = tl.arange($(24))
    r308_lane = tl.arange($(12))
    for r308_group in range($(8)) {
      r308_left = tl.load($(candidate_vto_copy_47_ptr) + (((((((r308_group % $(8)) * $(2304)) + (r308_lane[None, :] * $(192)) + r308_row[:, None]) / $(2304)) % $(8)) * $(2304)) + ((((((r308_group % $(8)) * $(2304)) + (r308_lane[None, :] * $(192)) + r308_row[:, None]) / $(192)) % $(12)) * $(192)) + ((((r308_group % $(8)) * $(2304)) + (r308_lane[None, :] * $(192)) + r308_row[:, None]) % $(192))), dtype=tl.float32)
      r308_right = tl.load($(candidate_vmul_40_ptr) + (((((((r308_group % $(8)) * $(288)) + (r308_lane[:, None] * $(24)) + r308_col[None, :]) / $(288)) % $(8)) * $(288)) + ((((((r308_group % $(8)) * $(288)) + (r308_lane[:, None] * $(24)) + r308_col[None, :]) / $(24)) % $(12)) * $(24)) + ((((r308_group % $(8)) * $(288)) + (r308_lane[:, None] * $(24)) + r308_col[None, :]) % $(24))), dtype=tl.float32)
      r308_value = tl.dot(r308_left, r308_right)
      r308_value = r308_value.to(tl.float32)
      tl.store($(candidate_vbmm_58_ptr) + ((r308_group * $(4608)) + (r308_row[:, None] * $(24)) + r308_col[None, :]), r308_value)
    }
    r309_row = tl.arange($(12))
    r309_col = tl.arange($(192))
    r309_lane = tl.arange($(24))
    for r309_group in range($(8)) {
      r309_left = tl.load($(candidate_vmul_40_ptr) + (((((((r309_group % $(8)) * $(288)) + (r309_row[:, None] * $(24)) + r309_lane[None, :]) / $(288)) % $(8)) * $(288)) + ((((((r309_group % $(8)) * $(288)) + (r309_row[:, None] * $(24)) + r309_lane[None, :]) / $(24)) % $(12)) * $(24)) + ((((r309_group % $(8)) * $(288)) + (r309_row[:, None] * $(24)) + r309_lane[None, :]) % $(24))), dtype=tl.float32)
      r309_right = tl.load($(candidate_vto_copy_48_ptr) + (((((((r309_group % $(8)) * $(4608)) + (r309_col[None, :] * $(24)) + r309_lane[:, None]) / $(4608)) % $(8)) * $(4608)) + (((((r309_group % $(8)) * $(4608)) + (r309_col[None, :] * $(24)) + r309_lane[:, None]) % $(24)) * $(192)) + (((((r309_group % $(8)) * $(4608)) + (r309_col[None, :] * $(24)) + r309_lane[:, None]) / $(24)) % $(192))), dtype=tl.float32)
      r309_value = tl.dot(r309_left, r309_right)
      r309_value = r309_value.to(tl.float32)
      tl.store($(candidate_vbmm_59_ptr) + ((r309_group * $(2304)) + (r309_row[:, None] * $(192)) + r309_col[None, :]), r309_value)
    }
    r310_z = tl.arange($(131072))
    r310_value = tl.where(((((r310_z / $(1024)) % $(128)) >= $(0)) & (((r310_z / $(1024)) % $(128)) < $(24)) & ((((r310_z / $(1024)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vbmm_56_ptr) + ((((((((r310_z / $(128)) % $(8)) * $(3072)) + ((((r310_z / $(1024)) % $(128)) - $(0)) * $(128)) + (r310_z % $(128))) / $(3072)) % $(8)) * $(3072)) + ((((((r310_z / $(128)) % $(8)) * $(24)) + (((r310_z / $(1024)) % $(128)) - $(0))) % $(24)) * $(128)) + (r310_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r310_value = r310_value.to(tl.bfloat16)
    r310_value = tl.broadcast_to(r310_value, [$(131072)])
    tl.store($(candidate_vslice_backward_19_ptr) + r310_z, r310_value)
    r311_z = tl.arange($(131072))
    r311_value = (tl.load($(candidate_vadd_27_ptr) + ((((r311_z / $(1024)) % $(128)) * $(1024)) + (((r311_z / $(128)) % $(8)) * $(128)) + (r311_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_19_ptr) + ((((r311_z / $(1024)) % $(128)) * $(1024)) + (((r311_z / $(128)) % $(8)) * $(128)) + (r311_z % $(128))), dtype=tl.bfloat16))
    r311_value = r311_value.to(tl.bfloat16)
    r311_value = tl.broadcast_to(r311_value, [$(131072)])
    tl.store($(candidate_vadd_30_ptr) + r311_z, r311_value)
    r312_z = tl.arange($(36864))
    r312_value = tl.load($(candidate_vbmm_58_ptr) + ((((r312_z / $(4608)) % $(8)) * $(4608)) + ((r312_z % $(192)) * $(24)) + ((r312_z / $(192)) % $(24))), dtype=tl.float32)
    r312_value = r312_value.to(tl.bfloat16)
    r312_value = tl.broadcast_to(r312_value, [$(36864)])
    tl.store($(candidate_vto_copy_81_ptr) + r312_z, r312_value)
    r313_z = tl.arange($(196608))
    r313_value = tl.where(((((r313_z / $(1536)) % $(128)) >= $(0)) & (((r313_z / $(1536)) % $(128)) < $(24)) & ((((r313_z / $(1536)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vto_copy_81_ptr) + ((((r313_z / $(192)) % $(8)) * $(4608)) + (((((r313_z / $(1536)) % $(128)) - $(0)) % $(24)) * $(192)) + (r313_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r313_value = r313_value.to(tl.bfloat16)
    r313_value = tl.broadcast_to(r313_value, [$(196608)])
    tl.store($(candidate_vslice_backward_20_ptr) + r313_z, r313_value)
    r314_z = tl.arange($(196608))
    r314_value = (tl.load($(candidate_vadd_28_ptr) + ((((r314_z / $(1536)) % $(128)) * $(1536)) + (((r314_z / $(192)) % $(8)) * $(192)) + (r314_z % $(192))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_20_ptr) + ((((r314_z / $(1536)) % $(128)) * $(1536)) + (((r314_z / $(192)) % $(8)) * $(192)) + (r314_z % $(192))), dtype=tl.bfloat16))
    r314_value = r314_value.to(tl.bfloat16)
    r314_value = tl.broadcast_to(r314_value, [$(196608)])
    tl.store($(candidate_vadd_31_ptr) + r314_z, r314_value)
    r315_z = tl.arange($(18432))
    r315_value = tl.load($(candidate_vbmm_59_ptr) + ((((r315_z / $(2304)) % $(8)) * $(2304)) + (((r315_z / $(192)) % $(12)) * $(192)) + (r315_z % $(192))), dtype=tl.float32)
    r315_value = r315_value.to(tl.bfloat16)
    r315_value = tl.broadcast_to(r315_value, [$(18432)])
    tl.store($(candidate_vto_copy_82_ptr) + r315_z, r315_value)
    r316_z = tl.arange($(49152))
    r316_value = tl.where(((((r316_z / $(1536)) % $(32)) >= $(0)) & (((r316_z / $(1536)) % $(32)) < $(12)) & ((((r316_z / $(1536)) % $(32)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vto_copy_82_ptr) + ((((r316_z / $(192)) % $(8)) * $(2304)) + (((((r316_z / $(1536)) % $(32)) - $(0)) % $(12)) * $(192)) + (r316_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r316_value = r316_value.to(tl.bfloat16)
    r316_value = tl.broadcast_to(r316_value, [$(49152)])
    tl.store($(candidate_vslice_backward_21_ptr) + r316_z, r316_value)
    r317_z = tl.arange($(49152))
    r317_value = (tl.load($(candidate_vadd_29_ptr) + ((((r317_z / $(1536)) % $(32)) * $(1536)) + (((r317_z / $(192)) % $(8)) * $(192)) + (r317_z % $(192))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_21_ptr) + ((((r317_z / $(1536)) % $(32)) * $(1536)) + (((r317_z / $(192)) % $(8)) * $(192)) + (r317_z % $(192))), dtype=tl.bfloat16))
    r317_value = r317_value.to(tl.bfloat16)
    r317_value = tl.broadcast_to(r317_value, [$(49152)])
    tl.store($(candidate_vadd_32_ptr) + r317_z, r317_value)
    r318_z = tl.arange($(24576))
    r318_value = tl.cast(0.0, tl.bfloat16)
    r318_value = r318_value.to(tl.bfloat16)
    r318_value = tl.broadcast_to(r318_value, [$(24576)])
    tl.store($(candidate_vfill_6_ptr) + r318_z, r318_value)
    r319_z = tl.arange($(131072))
    r319_value = tl.where(((((r319_z / $(1024)) % $(128)) >= $(24)) & (((r319_z / $(1024)) % $(128)) < $(48)) & ((((r319_z / $(1024)) % $(128)) - $(24)) % $(1) == $(0))), tl.load($(candidate_vfill_6_ptr) + ((((((r319_z / $(1024)) % $(128)) - $(24)) % $(24)) * $(1024)) + (((r319_z / $(128)) % $(8)) * $(128)) + (r319_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vadd_30_ptr) + ((((r319_z / $(1024)) % $(128)) * $(1024)) + (((r319_z / $(128)) % $(8)) * $(128)) + (r319_z % $(128))), dtype=tl.bfloat16))
    r319_value = r319_value.to(tl.bfloat16)
    r319_value = tl.broadcast_to(r319_value, [$(131072)])
    tl.store($(candidate_vslice_scatter_6_ptr) + r319_z, r319_value)
    r320_z = tl.arange($(8192))
    r320_value = tl.cast(0.0, tl.bfloat16)
    r320_value = r320_value.to(tl.bfloat16)
    r320_value = tl.broadcast_to(r320_value, [$(8192)])
    tl.store($(candidate_vfill_7_ptr) + r320_z, r320_value)
    r321_z = tl.arange($(131072))
    r321_value = tl.where(((((r321_z / $(1024)) % $(128)) >= $(56)) & (((r321_z / $(1024)) % $(128)) < $(64)) & ((((r321_z / $(1024)) % $(128)) - $(56)) % $(1) == $(0))), tl.load($(candidate_vfill_7_ptr) + ((((((r321_z / $(1024)) % $(128)) - $(56)) % $(8)) * $(1024)) + (((r321_z / $(128)) % $(8)) * $(128)) + (r321_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vslice_scatter_6_ptr) + ((((r321_z / $(1024)) % $(128)) * $(1024)) + (((r321_z / $(128)) % $(8)) * $(128)) + (r321_z % $(128))), dtype=tl.bfloat16))
    r321_value = r321_value.to(tl.bfloat16)
    r321_value = tl.broadcast_to(r321_value, [$(131072)])
    tl.store($(candidate_vslice_scatter_7_ptr) + r321_z, r321_value)
    r322_z = tl.arange($(32768))
    r322_value = tl.cast(0.0, tl.bfloat16)
    r322_value = r322_value.to(tl.bfloat16)
    r322_value = tl.broadcast_to(r322_value, [$(32768)])
    tl.store($(candidate_vfill_8_ptr) + r322_z, r322_value)
    r323_z = tl.arange($(131072))
    r323_value = tl.where(((((r323_z / $(1024)) % $(128)) >= $(96)) & (((r323_z / $(1024)) % $(128)) < $(128)) & ((((r323_z / $(1024)) % $(128)) - $(96)) % $(1) == $(0))), tl.load($(candidate_vfill_8_ptr) + ((((((r323_z / $(1024)) % $(128)) - $(96)) % $(32)) * $(1024)) + (((r323_z / $(128)) % $(8)) * $(128)) + (r323_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vslice_scatter_7_ptr) + ((((r323_z / $(1024)) % $(128)) * $(1024)) + (((r323_z / $(128)) % $(8)) * $(128)) + (r323_z % $(128))), dtype=tl.bfloat16))
    r323_value = r323_value.to(tl.bfloat16)
    r323_value = tl.broadcast_to(r323_value, [$(131072)])
    tl.store($(candidate_vslice_scatter_8_ptr) + r323_z, r323_value)
    r324_z = tl.arange($(36864))
    r324_value = tl.cast(0.0, tl.bfloat16)
    r324_value = r324_value.to(tl.bfloat16)
    r324_value = tl.broadcast_to(r324_value, [$(36864)])
    tl.store($(candidate_vfill_9_ptr) + r324_z, r324_value)
    r325_z = tl.arange($(196608))
    r325_value = tl.where(((((r325_z / $(1536)) % $(128)) >= $(24)) & (((r325_z / $(1536)) % $(128)) < $(48)) & ((((r325_z / $(1536)) % $(128)) - $(24)) % $(1) == $(0))), tl.load($(candidate_vfill_9_ptr) + ((((((r325_z / $(1536)) % $(128)) - $(24)) % $(24)) * $(1536)) + (((r325_z / $(192)) % $(8)) * $(192)) + (r325_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vadd_31_ptr) + ((((r325_z / $(1536)) % $(128)) * $(1536)) + (((r325_z / $(192)) % $(8)) * $(192)) + (r325_z % $(192))), dtype=tl.bfloat16))
    r325_value = r325_value.to(tl.bfloat16)
    r325_value = tl.broadcast_to(r325_value, [$(196608)])
    tl.store($(candidate_vslice_scatter_9_ptr) + r325_z, r325_value)
    r326_z = tl.arange($(12288))
    r326_value = tl.cast(0.0, tl.bfloat16)
    r326_value = r326_value.to(tl.bfloat16)
    r326_value = tl.broadcast_to(r326_value, [$(12288)])
    tl.store($(candidate_vfill_10_ptr) + r326_z, r326_value)
    r327_z = tl.arange($(196608))
    r327_value = tl.where(((((r327_z / $(1536)) % $(128)) >= $(56)) & (((r327_z / $(1536)) % $(128)) < $(64)) & ((((r327_z / $(1536)) % $(128)) - $(56)) % $(1) == $(0))), tl.load($(candidate_vfill_10_ptr) + ((((((r327_z / $(1536)) % $(128)) - $(56)) % $(8)) * $(1536)) + (((r327_z / $(192)) % $(8)) * $(192)) + (r327_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vslice_scatter_9_ptr) + ((((r327_z / $(1536)) % $(128)) * $(1536)) + (((r327_z / $(192)) % $(8)) * $(192)) + (r327_z % $(192))), dtype=tl.bfloat16))
    r327_value = r327_value.to(tl.bfloat16)
    r327_value = tl.broadcast_to(r327_value, [$(196608)])
    tl.store($(candidate_vslice_scatter_10_ptr) + r327_z, r327_value)
    r328_z = tl.arange($(49152))
    r328_value = tl.cast(0.0, tl.bfloat16)
    r328_value = r328_value.to(tl.bfloat16)
    r328_value = tl.broadcast_to(r328_value, [$(49152)])
    tl.store($(candidate_vfill_11_ptr) + r328_z, r328_value)
    r329_z = tl.arange($(196608))
    r329_value = tl.where(((((r329_z / $(1536)) % $(128)) >= $(96)) & (((r329_z / $(1536)) % $(128)) < $(128)) & ((((r329_z / $(1536)) % $(128)) - $(96)) % $(1) == $(0))), tl.load($(candidate_vfill_11_ptr) + ((((((r329_z / $(1536)) % $(128)) - $(96)) % $(32)) * $(1536)) + (((r329_z / $(192)) % $(8)) * $(192)) + (r329_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vslice_scatter_10_ptr) + ((((r329_z / $(1536)) % $(128)) * $(1536)) + (((r329_z / $(192)) % $(8)) * $(192)) + (r329_z % $(192))), dtype=tl.bfloat16))
    r329_value = r329_value.to(tl.bfloat16)
    r329_value = tl.broadcast_to(r329_value, [$(196608)])
    tl.store($(candidate_vslice_scatter_11_ptr) + r329_z, r329_value)
    r330_z = tl.arange($(8192))
    r330_lane = tl.arange($(8))
    r330_value = tl.sum(tl.load($(candidate_vslice_scatter_11_ptr) + ((((r330_z[:, None] / $(64)) % $(128)) * $(1536)) + (r330_lane[None, :] * $(192)) + (r330_z[:, None] % $(64)) + $(128)), dtype=tl.bfloat16), axis=1)
    r330_value = r330_value.to(tl.bfloat16)
    r330_value = tl.broadcast_to(r330_value, [$(8192)])
    tl.store($(candidate_vsum_2_ptr) + r330_z, r330_value)
    r331_row = tl.arange($(128))
    r331_col = tl.arange($(256))
    r331_lane = tl.arange($(1024))
    for r331_group in range($(1)) {
      r331_left = tl.load($(candidate_vslice_scatter_8_ptr) + ((((((r331_row[:, None] * $(1024)) + r331_lane[None, :]) / $(1024)) % $(128)) * $(1024)) + (((((r331_row[:, None] * $(1024)) + r331_lane[None, :]) / $(128)) % $(8)) * $(128)) + (((r331_row[:, None] * $(1024)) + r331_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r331_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r331_col[None, :] * $(1024)) + r331_lane[:, None]) / $(128)) % $(8)) * $(65536)) + ((((r331_col[None, :] * $(1024)) + r331_lane[:, None]) % $(128)) * $(256)) + ((((r331_col[None, :] * $(1024)) + r331_lane[:, None]) / $(1024)) % $(256)) + $(32768)), dtype=tl.bfloat16)
      r331_value = tl.dot(r331_left, r331_right)
      r331_value = r331_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_60_ptr) + ((r331_group * $(32768)) + (r331_row[:, None] * $(256)) + r331_col[None, :]), r331_value)
    }
    r332_row = tl.arange($(128))
    r332_col = tl.arange($(256))
    r332_lane = tl.arange($(1024))
    for r332_group in range($(1)) {
      r332_left = tl.load($(candidate_vslice_scatter_11_ptr) + ((((((r332_row[:, None] * $(1024)) + r332_lane[None, :]) / $(1024)) % $(128)) * $(1536)) + (((((r332_row[:, None] * $(1024)) + r332_lane[None, :]) / $(128)) % $(8)) * $(192)) + (((r332_row[:, None] * $(1024)) + r332_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r332_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r332_col[None, :] * $(1024)) + r332_lane[:, None]) / $(128)) % $(8)) * $(65536)) + ((((r332_col[None, :] * $(1024)) + r332_lane[:, None]) % $(128)) * $(256)) + ((((r332_col[None, :] * $(1024)) + r332_lane[:, None]) / $(1024)) % $(256))), dtype=tl.bfloat16)
      r332_value = tl.dot(r332_left, r332_right)
      r332_value = r332_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_61_ptr) + ((r332_group * $(32768)) + (r332_row[:, None] * $(256)) + r332_col[None, :]), r332_value)
    }
    r333_z = tl.arange($(32768))
    r333_value = (tl.load($(candidate_vbmm_60_ptr) + ((((r333_z / $(256)) % $(128)) * $(256)) + (r333_z % $(256))), dtype=tl.bfloat16) + tl.load($(candidate_vbmm_61_ptr) + ((((r333_z / $(256)) % $(128)) * $(256)) + (r333_z % $(256))), dtype=tl.bfloat16))
    r333_value = r333_value.to(tl.bfloat16)
    r333_value = tl.broadcast_to(r333_value, [$(32768)])
    tl.store($(candidate_vadd_33_ptr) + r333_z, r333_value)
    r334_z = tl.arange($(32768))
    r334_value = tl.where(($(0) == $(0)), tl.load($(candidate_vadd_33_ptr) + ((((r334_z / $(256)) % $(128)) * $(256)) + (r334_z % $(256))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r334_value = r334_value.to(tl.bfloat16)
    r334_value = tl.broadcast_to(r334_value, [$(32768)])
    tl.store($(candidate_vselect_backward_1_ptr) + r334_z, r334_value)
    r335_z = tl.arange($(32768))
    r335_value = tl.where(((((r335_z / $(256)) % $(128)) >= $(0)) & (((r335_z / $(256)) % $(128)) < $(128)) & ((((r335_z / $(256)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vselect_backward_1_ptr) + ((((((r335_z / $(256)) % $(128)) - $(0)) % $(128)) * $(256)) + (r335_z % $(256))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r335_value = r335_value.to(tl.bfloat16)
    r335_value = tl.broadcast_to(r335_value, [$(32768)])
    tl.store($(candidate_vslice_backward_22_ptr) + r335_z, r335_value)
    r336_z = tl.arange($(32768))
    r336_value = (tl.load($(candidate_vslice_backward_9_ptr) + ((((r336_z / $(256)) % $(128)) * $(256)) + (r336_z % $(256))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_22_ptr) + ((((r336_z / $(256)) % $(128)) * $(256)) + (r336_z % $(256))), dtype=tl.bfloat16))
    r336_value = r336_value.to(tl.bfloat16)
    r336_value = tl.broadcast_to(r336_value, [$(32768)])
    tl.store($(candidate_vadd_34_ptr) + r336_z, r336_value)
    r337_z = tl.arange($(8192))
    r337_value = tl.where(((((r337_z / $(64)) % $(128)) >= $(0)) & (((r337_z / $(64)) % $(128)) < $(128)) & ((((r337_z / $(64)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vsum_2_ptr) + ((((((r337_z / $(64)) % $(128)) - $(0)) % $(128)) * $(64)) + (r337_z % $(64))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r337_value = r337_value.to(tl.bfloat16)
    r337_value = tl.broadcast_to(r337_value, [$(8192)])
    tl.store($(candidate_vslice_backward_23_ptr) + r337_z, r337_value)
    r338_z = tl.arange($(8192))
    r338_value = (tl.load($(candidate_vslice_backward_10_ptr) + ((((r338_z / $(64)) % $(128)) * $(64)) + (r338_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_23_ptr) + ((((r338_z / $(64)) % $(128)) * $(64)) + (r338_z % $(64))), dtype=tl.bfloat16))
    r338_value = r338_value.to(tl.bfloat16)
    r338_value = tl.broadcast_to(r338_value, [$(8192)])
    tl.store($(candidate_vadd_35_ptr) + r338_z, r338_value)
    r339_z = tl.arange($(131072))
    r339_value = tl.where(((((r339_z / $(1024)) % $(128)) >= $(64)) & (((r339_z / $(1024)) % $(128)) < $(96)) & ((((r339_z / $(1024)) % $(128)) - $(64)) % $(1) == $(0))), tl.load($(candidate_vadd_32_ptr) + ((((((r339_z / $(1024)) % $(128)) - $(64)) % $(32)) * $(1536)) + (((r339_z / $(128)) % $(8)) * $(192)) + (r339_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r339_value = r339_value.to(tl.bfloat16)
    r339_value = tl.broadcast_to(r339_value, [$(131072)])
    tl.store($(candidate_vslice_backward_24_ptr) + r339_z, r339_value)
    r340_z = tl.arange($(131072))
    r340_value = (tl.load($(candidate_vslice_backward_11_ptr) + ((((r340_z / $(1024)) % $(128)) * $(1024)) + (((r340_z / $(128)) % $(8)) * $(128)) + (r340_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_24_ptr) + ((((r340_z / $(1024)) % $(128)) * $(1024)) + (((r340_z / $(128)) % $(8)) * $(128)) + (r340_z % $(128))), dtype=tl.bfloat16))
    r340_value = r340_value.to(tl.bfloat16)
    r340_value = tl.broadcast_to(r340_value, [$(131072)])
    tl.store($(candidate_vadd_36_ptr) + r340_z, r340_value)
    r341_z = tl.arange($(65536))
    r341_value = tl.where(((((r341_z / $(512)) % $(128)) >= $(64)) & (((r341_z / $(512)) % $(128)) < $(96)) & ((((r341_z / $(512)) % $(128)) - $(64)) % $(1) == $(0))), tl.load($(candidate_vadd_32_ptr) + ((((((r341_z / $(512)) % $(128)) - $(64)) % $(32)) * $(1536)) + (((r341_z / $(64)) % $(8)) * $(192)) + (r341_z % $(64)) + $(128)), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r341_value = r341_value.to(tl.bfloat16)
    r341_value = tl.broadcast_to(r341_value, [$(65536)])
    tl.store($(candidate_vslice_backward_25_ptr) + r341_z, r341_value)
    r342_z = tl.arange($(65536))
    r342_value = (tl.load($(candidate_vslice_backward_12_ptr) + ((((r342_z / $(512)) % $(128)) * $(512)) + (((r342_z / $(64)) % $(8)) * $(64)) + (r342_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_25_ptr) + ((((r342_z / $(512)) % $(128)) * $(512)) + (((r342_z / $(64)) % $(8)) * $(64)) + (r342_z % $(64))), dtype=tl.bfloat16))
    r342_value = r342_value.to(tl.bfloat16)
    r342_value = tl.broadcast_to(r342_value, [$(65536)])
    tl.store($(candidate_vadd_37_ptr) + r342_z, r342_value)
    r343_row = tl.arange($(64))
    r343_col = tl.arange($(128))
    r343_lane = tl.arange($(16))
    for r343_group in range($(8)) {
      r343_left = tl.load($(candidate_vto_copy_46_ptr) + (((((((r343_group % $(8)) * $(1024)) + (r343_lane[None, :] * $(64)) + r343_row[:, None]) / $(1024)) % $(8)) * $(1024)) + ((((((r343_group % $(8)) * $(1024)) + (r343_lane[None, :] * $(64)) + r343_row[:, None]) / $(64)) % $(16)) * $(64)) + ((((r343_group % $(8)) * $(1024)) + (r343_lane[None, :] * $(64)) + r343_row[:, None]) % $(64))), dtype=tl.bfloat16)
      r343_right = tl.load($(candidate_vmm_4_ptr) + (((((((r343_group % $(8)) * $(2048)) + (r343_lane[:, None] * $(128)) + r343_col[None, :]) / $(128)) % $(16)) * $(1024)) + ((((((r343_group % $(8)) * $(2048)) + (r343_lane[:, None] * $(128)) + r343_col[None, :]) / $(2048)) % $(8)) * $(128)) + ((((r343_group % $(8)) * $(2048)) + (r343_lane[:, None] * $(128)) + r343_col[None, :]) % $(128)) + $(49152)), dtype=tl.bfloat16)
      r343_value = tl.dot(r343_left, r343_right)
      r343_value = r343_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_62_ptr) + ((r343_group * $(8192)) + (r343_row[:, None] * $(128)) + r343_col[None, :]), r343_value)
    }
    r344_row = tl.arange($(16))
    r344_col = tl.arange($(64))
    r344_lane = tl.arange($(128))
    for r344_group in range($(8)) {
      r344_left = tl.load($(candidate_vmm_4_ptr) + (((((((r344_group % $(8)) * $(2048)) + (r344_row[:, None] * $(128)) + r344_lane[None, :]) / $(128)) % $(16)) * $(1024)) + ((((((r344_group % $(8)) * $(2048)) + (r344_row[:, None] * $(128)) + r344_lane[None, :]) / $(2048)) % $(8)) * $(128)) + ((((r344_group % $(8)) * $(2048)) + (r344_row[:, None] * $(128)) + r344_lane[None, :]) % $(128)) + $(49152)), dtype=tl.bfloat16)
      r344_right = tl.load($(candidate_vbmm_11_ptr) + (((((((r344_group % $(8)) * $(8192)) + (r344_col[None, :] * $(128)) + r344_lane[:, None]) / $(128)) % $(64)) * $(1024)) + ((((((r344_group % $(8)) * $(8192)) + (r344_col[None, :] * $(128)) + r344_lane[:, None]) / $(8192)) % $(8)) * $(128)) + ((((r344_group % $(8)) * $(8192)) + (r344_col[None, :] * $(128)) + r344_lane[:, None]) % $(128)) + $(65536)), dtype=tl.bfloat16)
      r344_value = tl.dot(r344_left, r344_right)
      r344_value = r344_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_63_ptr) + ((r344_group * $(1024)) + (r344_row[:, None] * $(64)) + r344_col[None, :]), r344_value)
    }
    r345_z = tl.arange($(8192))
    r345_value = tl.load($(candidate_vbmm_63_ptr) + ((((r345_z / $(1024)) % $(8)) * $(1024)) + (((r345_z / $(64)) % $(16)) * $(64)) + (r345_z % $(64))), dtype=tl.bfloat16)
    r345_value = r345_value.to(tl.float32)
    r345_value = tl.broadcast_to(r345_value, [$(8192)])
    tl.store($(candidate_vto_copy_83_ptr) + r345_z, r345_value)
    r346_z = tl.arange($(8192))
    r346_lane = tl.arange($(64))
    r346_row_values = tl.load($(candidate_vsoftmax_5_ptr) + ((((r346_z[:, None] / $(1024)) % $(8)) * $(1024)) + (((r346_z[:, None] / $(64)) % $(16)) * $(64)) + (r346_lane[None, :] % $(64))), dtype=tl.float32)
    r346_row_grad = tl.load($(candidate_vto_copy_83_ptr) + ((((r346_z[:, None] / $(1024)) % $(8)) * $(1024)) + (((r346_z[:, None] / $(64)) % $(16)) * $(64)) + (r346_lane[None, :] % $(64))), dtype=tl.float32)
    r346_weighted = r346_row_grad * r346_row_values
    r346_dot_grad = tl.sum(r346_weighted, axis=1)
    r346_value = (tl.load($(candidate_vto_copy_83_ptr) + ((((r346_z / $(1024)) % $(8)) * $(1024)) + (((r346_z / $(64)) % $(16)) * $(64)) + (r346_z % $(64))), dtype=tl.float32) - r346_dot_grad) * tl.load($(candidate_vsoftmax_5_ptr) + ((((r346_z / $(1024)) % $(8)) * $(1024)) + (((r346_z / $(64)) % $(16)) * $(64)) + (r346_z % $(64))), dtype=tl.float32)
    r346_value = r346_value.to(tl.float32)
    r346_value = tl.broadcast_to(r346_value, [$(8192)])
    tl.store($(candidate_vsoftmax_backward_data_6_ptr) + r346_z, r346_value)
    r347_z = tl.arange($(8192))
    r347_value = tl.where(tl.load($(candidate_vgt_5_ptr) + ((((r347_z / $(64)) % $(16)) * $(64)) + (r347_z % $(64))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_6_ptr) + ((((r347_z / $(1024)) % $(8)) * $(1024)) + (((r347_z / $(64)) % $(16)) * $(64)) + (r347_z % $(64))), dtype=tl.float32))
    r347_value = r347_value.to(tl.float32)
    r347_value = tl.broadcast_to(r347_value, [$(8192)])
    tl.store($(candidate_vmasked_fill_18_ptr) + r347_z, r347_value)
    r348_z = tl.arange($(8192))
    r348_value = (tl.load($(candidate_vmasked_fill_18_ptr) + ((((r348_z / $(1024)) % $(8)) * $(1024)) + (((r348_z / $(64)) % $(16)) * $(64)) + (r348_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r348_value = r348_value.to(tl.float32)
    r348_value = tl.broadcast_to(r348_value, [$(8192)])
    tl.store($(candidate_vmul_41_ptr) + r348_z, r348_value)
    r349_row = tl.arange($(192))
    r349_col = tl.arange($(64))
    r349_lane = tl.arange($(16))
    for r349_group in range($(8)) {
      r349_left = tl.load($(candidate_vto_copy_44_ptr) + (((((((r349_group % $(8)) * $(3072)) + (r349_lane[None, :] * $(192)) + r349_row[:, None]) / $(3072)) % $(8)) * $(3072)) + ((((((r349_group % $(8)) * $(3072)) + (r349_lane[None, :] * $(192)) + r349_row[:, None]) / $(192)) % $(16)) * $(192)) + ((((r349_group % $(8)) * $(3072)) + (r349_lane[None, :] * $(192)) + r349_row[:, None]) % $(192))), dtype=tl.float32)
      r349_right = tl.load($(candidate_vmul_41_ptr) + (((((((r349_group % $(8)) * $(1024)) + (r349_lane[:, None] * $(64)) + r349_col[None, :]) / $(1024)) % $(8)) * $(1024)) + ((((((r349_group % $(8)) * $(1024)) + (r349_lane[:, None] * $(64)) + r349_col[None, :]) / $(64)) % $(16)) * $(64)) + ((((r349_group % $(8)) * $(1024)) + (r349_lane[:, None] * $(64)) + r349_col[None, :]) % $(64))), dtype=tl.float32)
      r349_value = tl.dot(r349_left, r349_right)
      r349_value = r349_value.to(tl.float32)
      tl.store($(candidate_vbmm_64_ptr) + ((r349_group * $(12288)) + (r349_row[:, None] * $(64)) + r349_col[None, :]), r349_value)
    }
    r350_row = tl.arange($(16))
    r350_col = tl.arange($(192))
    r350_lane = tl.arange($(64))
    for r350_group in range($(8)) {
      r350_left = tl.load($(candidate_vmul_41_ptr) + (((((((r350_group % $(8)) * $(1024)) + (r350_row[:, None] * $(64)) + r350_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + ((((((r350_group % $(8)) * $(1024)) + (r350_row[:, None] * $(64)) + r350_lane[None, :]) / $(64)) % $(16)) * $(64)) + ((((r350_group % $(8)) * $(1024)) + (r350_row[:, None] * $(64)) + r350_lane[None, :]) % $(64))), dtype=tl.float32)
      r350_right = tl.load($(candidate_vto_copy_45_ptr) + (((((((r350_group % $(8)) * $(12288)) + (r350_col[None, :] * $(64)) + r350_lane[:, None]) / $(12288)) % $(8)) * $(12288)) + (((((r350_group % $(8)) * $(12288)) + (r350_col[None, :] * $(64)) + r350_lane[:, None]) % $(64)) * $(192)) + (((((r350_group % $(8)) * $(12288)) + (r350_col[None, :] * $(64)) + r350_lane[:, None]) / $(64)) % $(192))), dtype=tl.float32)
      r350_value = tl.dot(r350_left, r350_right)
      r350_value = r350_value.to(tl.float32)
      tl.store($(candidate_vbmm_65_ptr) + ((r350_group * $(3072)) + (r350_row[:, None] * $(192)) + r350_col[None, :]), r350_value)
    }
    r351_z = tl.arange($(131072))
    r351_value = tl.where(((((r351_z / $(1024)) % $(128)) >= $(64)) & (((r351_z / $(1024)) % $(128)) < $(128)) & ((((r351_z / $(1024)) % $(128)) - $(64)) % $(1) == $(0))), tl.load($(candidate_vbmm_62_ptr) + ((((((((r351_z / $(128)) % $(8)) * $(8192)) + ((((r351_z / $(1024)) % $(128)) - $(64)) * $(128)) + (r351_z % $(128))) / $(8192)) % $(8)) * $(8192)) + ((((((r351_z / $(128)) % $(8)) * $(64)) + (((r351_z / $(1024)) % $(128)) - $(64))) % $(64)) * $(128)) + (r351_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r351_value = r351_value.to(tl.bfloat16)
    r351_value = tl.broadcast_to(r351_value, [$(131072)])
    tl.store($(candidate_vslice_backward_26_ptr) + r351_z, r351_value)
    r352_z = tl.arange($(98304))
    r352_value = tl.load($(candidate_vbmm_64_ptr) + ((((r352_z / $(12288)) % $(8)) * $(12288)) + ((r352_z % $(192)) * $(64)) + ((r352_z / $(192)) % $(64))), dtype=tl.float32)
    r352_value = r352_value.to(tl.bfloat16)
    r352_value = tl.broadcast_to(r352_value, [$(98304)])
    tl.store($(candidate_vto_copy_84_ptr) + r352_z, r352_value)
    r353_z = tl.arange($(196608))
    r353_value = tl.where(((((r353_z / $(1536)) % $(128)) >= $(64)) & (((r353_z / $(1536)) % $(128)) < $(128)) & ((((r353_z / $(1536)) % $(128)) - $(64)) % $(1) == $(0))), tl.load($(candidate_vto_copy_84_ptr) + ((((r353_z / $(192)) % $(8)) * $(12288)) + (((((r353_z / $(1536)) % $(128)) - $(64)) % $(64)) * $(192)) + (r353_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r353_value = r353_value.to(tl.bfloat16)
    r353_value = tl.broadcast_to(r353_value, [$(196608)])
    tl.store($(candidate_vslice_backward_27_ptr) + r353_z, r353_value)
    r354_z = tl.arange($(24576))
    r354_value = tl.load($(candidate_vbmm_65_ptr) + ((((r354_z / $(3072)) % $(8)) * $(3072)) + (((r354_z / $(192)) % $(16)) * $(192)) + (r354_z % $(192))), dtype=tl.float32)
    r354_value = r354_value.to(tl.bfloat16)
    r354_value = tl.broadcast_to(r354_value, [$(24576)])
    tl.store($(candidate_vto_copy_85_ptr) + r354_z, r354_value)
    r355_z = tl.arange($(49152))
    r355_value = tl.where(((((r355_z / $(1536)) % $(32)) >= $(16)) & (((r355_z / $(1536)) % $(32)) < $(32)) & ((((r355_z / $(1536)) % $(32)) - $(16)) % $(1) == $(0))), tl.load($(candidate_vto_copy_85_ptr) + ((((r355_z / $(192)) % $(8)) * $(3072)) + (((((r355_z / $(1536)) % $(32)) - $(16)) % $(16)) * $(192)) + (r355_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r355_value = r355_value.to(tl.bfloat16)
    r355_value = tl.broadcast_to(r355_value, [$(49152)])
    tl.store($(candidate_vslice_backward_28_ptr) + r355_z, r355_value)
    r356_row = tl.arange($(16))
    r356_col = tl.arange($(128))
    r356_lane = tl.arange($(4))
    for r356_group in range($(8)) {
      r356_left = tl.load($(candidate_vto_copy_43_ptr) + (((((((r356_group % $(8)) * $(64)) + (r356_lane[None, :] * $(16)) + r356_row[:, None]) / $(64)) % $(8)) * $(64)) + ((((((r356_group % $(8)) * $(64)) + (r356_lane[None, :] * $(16)) + r356_row[:, None]) / $(16)) % $(4)) * $(16)) + ((((r356_group % $(8)) * $(64)) + (r356_lane[None, :] * $(16)) + r356_row[:, None]) % $(16))), dtype=tl.bfloat16)
      r356_right = tl.load($(candidate_vmm_4_ptr) + (((((((r356_group % $(8)) * $(512)) + (r356_lane[:, None] * $(128)) + r356_col[None, :]) / $(128)) % $(4)) * $(1024)) + ((((((r356_group % $(8)) * $(512)) + (r356_lane[:, None] * $(128)) + r356_col[None, :]) / $(512)) % $(8)) * $(128)) + ((((r356_group % $(8)) * $(512)) + (r356_lane[:, None] * $(128)) + r356_col[None, :]) % $(128)) + $(45056)), dtype=tl.bfloat16)
      r356_value = tl.dot(r356_left, r356_right)
      r356_value = r356_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_66_ptr) + ((r356_group * $(2048)) + (r356_row[:, None] * $(128)) + r356_col[None, :]), r356_value)
    }
    r357_row = tl.arange($(4))
    r357_col = tl.arange($(16))
    r357_lane = tl.arange($(128))
    for r357_group in range($(8)) {
      r357_left = tl.load($(candidate_vmm_4_ptr) + (((((((r357_group % $(8)) * $(512)) + (r357_row[:, None] * $(128)) + r357_lane[None, :]) / $(128)) % $(4)) * $(1024)) + ((((((r357_group % $(8)) * $(512)) + (r357_row[:, None] * $(128)) + r357_lane[None, :]) / $(512)) % $(8)) * $(128)) + ((((r357_group % $(8)) * $(512)) + (r357_row[:, None] * $(128)) + r357_lane[None, :]) % $(128)) + $(45056)), dtype=tl.bfloat16)
      r357_right = tl.load($(candidate_vbmm_11_ptr) + (((((((r357_group % $(8)) * $(2048)) + (r357_col[None, :] * $(128)) + r357_lane[:, None]) / $(128)) % $(16)) * $(1024)) + ((((((r357_group % $(8)) * $(2048)) + (r357_col[None, :] * $(128)) + r357_lane[:, None]) / $(2048)) % $(8)) * $(128)) + ((((r357_group % $(8)) * $(2048)) + (r357_col[None, :] * $(128)) + r357_lane[:, None]) % $(128)) + $(49152)), dtype=tl.bfloat16)
      r357_value = tl.dot(r357_left, r357_right)
      r357_value = r357_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_67_ptr) + ((r357_group * $(64)) + (r357_row[:, None] * $(16)) + r357_col[None, :]), r357_value)
    }
    r358_z = tl.arange($(512))
    r358_value = tl.load($(candidate_vbmm_67_ptr) + ((((r358_z / $(64)) % $(8)) * $(64)) + (((r358_z / $(16)) % $(4)) * $(16)) + (r358_z % $(16))), dtype=tl.bfloat16)
    r358_value = r358_value.to(tl.float32)
    r358_value = tl.broadcast_to(r358_value, [$(512)])
    tl.store($(candidate_vto_copy_86_ptr) + r358_z, r358_value)
    r359_z = tl.arange($(512))
    r359_lane = tl.arange($(16))
    r359_row_values = tl.load($(candidate_vsoftmax_4_ptr) + ((((r359_z[:, None] / $(64)) % $(8)) * $(64)) + (((r359_z[:, None] / $(16)) % $(4)) * $(16)) + (r359_lane[None, :] % $(16))), dtype=tl.float32)
    r359_row_grad = tl.load($(candidate_vto_copy_86_ptr) + ((((r359_z[:, None] / $(64)) % $(8)) * $(64)) + (((r359_z[:, None] / $(16)) % $(4)) * $(16)) + (r359_lane[None, :] % $(16))), dtype=tl.float32)
    r359_weighted = r359_row_grad * r359_row_values
    r359_dot_grad = tl.sum(r359_weighted, axis=1)
    r359_value = (tl.load($(candidate_vto_copy_86_ptr) + ((((r359_z / $(64)) % $(8)) * $(64)) + (((r359_z / $(16)) % $(4)) * $(16)) + (r359_z % $(16))), dtype=tl.float32) - r359_dot_grad) * tl.load($(candidate_vsoftmax_4_ptr) + ((((r359_z / $(64)) % $(8)) * $(64)) + (((r359_z / $(16)) % $(4)) * $(16)) + (r359_z % $(16))), dtype=tl.float32)
    r359_value = r359_value.to(tl.float32)
    r359_value = tl.broadcast_to(r359_value, [$(512)])
    tl.store($(candidate_vsoftmax_backward_data_7_ptr) + r359_z, r359_value)
    r360_z = tl.arange($(512))
    r360_value = tl.where(tl.load($(candidate_vgt_4_ptr) + ((((r360_z / $(16)) % $(4)) * $(16)) + (r360_z % $(16))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_7_ptr) + ((((r360_z / $(64)) % $(8)) * $(64)) + (((r360_z / $(16)) % $(4)) * $(16)) + (r360_z % $(16))), dtype=tl.float32))
    r360_value = r360_value.to(tl.float32)
    r360_value = tl.broadcast_to(r360_value, [$(512)])
    tl.store($(candidate_vmasked_fill_19_ptr) + r360_z, r360_value)
    r361_z = tl.arange($(512))
    r361_value = (tl.load($(candidate_vmasked_fill_19_ptr) + ((((r361_z / $(64)) % $(8)) * $(64)) + (((r361_z / $(16)) % $(4)) * $(16)) + (r361_z % $(16))), dtype=tl.float32) * 0.07216878364870322)
    r361_value = r361_value.to(tl.float32)
    r361_value = tl.broadcast_to(r361_value, [$(512)])
    tl.store($(candidate_vmul_42_ptr) + r361_z, r361_value)
    r362_row = tl.arange($(192))
    r362_col = tl.arange($(16))
    r362_lane = tl.arange($(4))
    for r362_group in range($(8)) {
      r362_left = tl.load($(candidate_vto_copy_41_ptr) + (((((((r362_group % $(8)) * $(768)) + (r362_lane[None, :] * $(192)) + r362_row[:, None]) / $(768)) % $(8)) * $(768)) + ((((((r362_group % $(8)) * $(768)) + (r362_lane[None, :] * $(192)) + r362_row[:, None]) / $(192)) % $(4)) * $(192)) + ((((r362_group % $(8)) * $(768)) + (r362_lane[None, :] * $(192)) + r362_row[:, None]) % $(192))), dtype=tl.float32)
      r362_right = tl.load($(candidate_vmul_42_ptr) + (((((((r362_group % $(8)) * $(64)) + (r362_lane[:, None] * $(16)) + r362_col[None, :]) / $(64)) % $(8)) * $(64)) + ((((((r362_group % $(8)) * $(64)) + (r362_lane[:, None] * $(16)) + r362_col[None, :]) / $(16)) % $(4)) * $(16)) + ((((r362_group % $(8)) * $(64)) + (r362_lane[:, None] * $(16)) + r362_col[None, :]) % $(16))), dtype=tl.float32)
      r362_value = tl.dot(r362_left, r362_right)
      r362_value = r362_value.to(tl.float32)
      tl.store($(candidate_vbmm_68_ptr) + ((r362_group * $(3072)) + (r362_row[:, None] * $(16)) + r362_col[None, :]), r362_value)
    }
    r363_row = tl.arange($(4))
    r363_col = tl.arange($(192))
    r363_lane = tl.arange($(16))
    for r363_group in range($(8)) {
      r363_left = tl.load($(candidate_vmul_42_ptr) + (((((((r363_group % $(8)) * $(64)) + (r363_row[:, None] * $(16)) + r363_lane[None, :]) / $(64)) % $(8)) * $(64)) + ((((((r363_group % $(8)) * $(64)) + (r363_row[:, None] * $(16)) + r363_lane[None, :]) / $(16)) % $(4)) * $(16)) + ((((r363_group % $(8)) * $(64)) + (r363_row[:, None] * $(16)) + r363_lane[None, :]) % $(16))), dtype=tl.float32)
      r363_right = tl.load($(candidate_vto_copy_42_ptr) + (((((((r363_group % $(8)) * $(3072)) + (r363_col[None, :] * $(16)) + r363_lane[:, None]) / $(3072)) % $(8)) * $(3072)) + (((((r363_group % $(8)) * $(3072)) + (r363_col[None, :] * $(16)) + r363_lane[:, None]) % $(16)) * $(192)) + (((((r363_group % $(8)) * $(3072)) + (r363_col[None, :] * $(16)) + r363_lane[:, None]) / $(16)) % $(192))), dtype=tl.float32)
      r363_value = tl.dot(r363_left, r363_right)
      r363_value = r363_value.to(tl.float32)
      tl.store($(candidate_vbmm_69_ptr) + ((r363_group * $(768)) + (r363_row[:, None] * $(192)) + r363_col[None, :]), r363_value)
    }
    r364_z = tl.arange($(131072))
    r364_value = tl.where(((((r364_z / $(1024)) % $(128)) >= $(48)) & (((r364_z / $(1024)) % $(128)) < $(64)) & ((((r364_z / $(1024)) % $(128)) - $(48)) % $(1) == $(0))), tl.load($(candidate_vbmm_66_ptr) + ((((((((r364_z / $(128)) % $(8)) * $(2048)) + ((((r364_z / $(1024)) % $(128)) - $(48)) * $(128)) + (r364_z % $(128))) / $(2048)) % $(8)) * $(2048)) + ((((((r364_z / $(128)) % $(8)) * $(16)) + (((r364_z / $(1024)) % $(128)) - $(48))) % $(16)) * $(128)) + (r364_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r364_value = r364_value.to(tl.bfloat16)
    r364_value = tl.broadcast_to(r364_value, [$(131072)])
    tl.store($(candidate_vslice_backward_29_ptr) + r364_z, r364_value)
    r365_z = tl.arange($(131072))
    r365_value = (tl.load($(candidate_vslice_backward_26_ptr) + ((((r365_z / $(1024)) % $(128)) * $(1024)) + (((r365_z / $(128)) % $(8)) * $(128)) + (r365_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_29_ptr) + ((((r365_z / $(1024)) % $(128)) * $(1024)) + (((r365_z / $(128)) % $(8)) * $(128)) + (r365_z % $(128))), dtype=tl.bfloat16))
    r365_value = r365_value.to(tl.bfloat16)
    r365_value = tl.broadcast_to(r365_value, [$(131072)])
    tl.store($(candidate_vadd_38_ptr) + r365_z, r365_value)
    r366_z = tl.arange($(24576))
    r366_value = tl.load($(candidate_vbmm_68_ptr) + ((((r366_z / $(3072)) % $(8)) * $(3072)) + ((r366_z % $(192)) * $(16)) + ((r366_z / $(192)) % $(16))), dtype=tl.float32)
    r366_value = r366_value.to(tl.bfloat16)
    r366_value = tl.broadcast_to(r366_value, [$(24576)])
    tl.store($(candidate_vto_copy_87_ptr) + r366_z, r366_value)
    r367_z = tl.arange($(196608))
    r367_value = tl.where(((((r367_z / $(1536)) % $(128)) >= $(48)) & (((r367_z / $(1536)) % $(128)) < $(64)) & ((((r367_z / $(1536)) % $(128)) - $(48)) % $(1) == $(0))), tl.load($(candidate_vto_copy_87_ptr) + ((((r367_z / $(192)) % $(8)) * $(3072)) + (((((r367_z / $(1536)) % $(128)) - $(48)) % $(16)) * $(192)) + (r367_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r367_value = r367_value.to(tl.bfloat16)
    r367_value = tl.broadcast_to(r367_value, [$(196608)])
    tl.store($(candidate_vslice_backward_30_ptr) + r367_z, r367_value)
    r368_z = tl.arange($(196608))
    r368_value = (tl.load($(candidate_vslice_backward_27_ptr) + ((((r368_z / $(1536)) % $(128)) * $(1536)) + (((r368_z / $(192)) % $(8)) * $(192)) + (r368_z % $(192))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_30_ptr) + ((((r368_z / $(1536)) % $(128)) * $(1536)) + (((r368_z / $(192)) % $(8)) * $(192)) + (r368_z % $(192))), dtype=tl.bfloat16))
    r368_value = r368_value.to(tl.bfloat16)
    r368_value = tl.broadcast_to(r368_value, [$(196608)])
    tl.store($(candidate_vadd_39_ptr) + r368_z, r368_value)
    r369_z = tl.arange($(6144))
    r369_value = tl.load($(candidate_vbmm_69_ptr) + ((((r369_z / $(768)) % $(8)) * $(768)) + (((r369_z / $(192)) % $(4)) * $(192)) + (r369_z % $(192))), dtype=tl.float32)
    r369_value = r369_value.to(tl.bfloat16)
    r369_value = tl.broadcast_to(r369_value, [$(6144)])
    tl.store($(candidate_vto_copy_88_ptr) + r369_z, r369_value)
    r370_z = tl.arange($(49152))
    r370_value = tl.where(((((r370_z / $(1536)) % $(32)) >= $(12)) & (((r370_z / $(1536)) % $(32)) < $(16)) & ((((r370_z / $(1536)) % $(32)) - $(12)) % $(1) == $(0))), tl.load($(candidate_vto_copy_88_ptr) + ((((r370_z / $(192)) % $(8)) * $(768)) + (((((r370_z / $(1536)) % $(32)) - $(12)) % $(4)) * $(192)) + (r370_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r370_value = r370_value.to(tl.bfloat16)
    r370_value = tl.broadcast_to(r370_value, [$(49152)])
    tl.store($(candidate_vslice_backward_31_ptr) + r370_z, r370_value)
    r371_z = tl.arange($(49152))
    r371_value = (tl.load($(candidate_vslice_backward_28_ptr) + ((((r371_z / $(1536)) % $(32)) * $(1536)) + (((r371_z / $(192)) % $(8)) * $(192)) + (r371_z % $(192))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_31_ptr) + ((((r371_z / $(1536)) % $(32)) * $(1536)) + (((r371_z / $(192)) % $(8)) * $(192)) + (r371_z % $(192))), dtype=tl.bfloat16))
    r371_value = r371_value.to(tl.bfloat16)
    r371_value = tl.broadcast_to(r371_value, [$(49152)])
    tl.store($(candidate_vadd_40_ptr) + r371_z, r371_value)
    r372_row = tl.arange($(48))
    r372_col = tl.arange($(128))
    r372_lane = tl.arange($(12))
    for r372_group in range($(8)) {
      r372_left = tl.load($(candidate_vto_copy_40_ptr) + (((((((r372_group % $(8)) * $(576)) + (r372_lane[None, :] * $(48)) + r372_row[:, None]) / $(576)) % $(8)) * $(576)) + ((((((r372_group % $(8)) * $(576)) + (r372_lane[None, :] * $(48)) + r372_row[:, None]) / $(48)) % $(12)) * $(48)) + ((((r372_group % $(8)) * $(576)) + (r372_lane[None, :] * $(48)) + r372_row[:, None]) % $(48))), dtype=tl.bfloat16)
      r372_right = tl.load($(candidate_vmm_4_ptr) + (((((((r372_group % $(8)) * $(1536)) + (r372_lane[:, None] * $(128)) + r372_col[None, :]) / $(128)) % $(12)) * $(1024)) + ((((((r372_group % $(8)) * $(1536)) + (r372_lane[:, None] * $(128)) + r372_col[None, :]) / $(1536)) % $(8)) * $(128)) + ((((r372_group % $(8)) * $(1536)) + (r372_lane[:, None] * $(128)) + r372_col[None, :]) % $(128)) + $(32768)), dtype=tl.bfloat16)
      r372_value = tl.dot(r372_left, r372_right)
      r372_value = r372_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_70_ptr) + ((r372_group * $(6144)) + (r372_row[:, None] * $(128)) + r372_col[None, :]), r372_value)
    }
    r373_row = tl.arange($(12))
    r373_col = tl.arange($(48))
    r373_lane = tl.arange($(128))
    for r373_group in range($(8)) {
      r373_left = tl.load($(candidate_vmm_4_ptr) + (((((((r373_group % $(8)) * $(1536)) + (r373_row[:, None] * $(128)) + r373_lane[None, :]) / $(128)) % $(12)) * $(1024)) + ((((((r373_group % $(8)) * $(1536)) + (r373_row[:, None] * $(128)) + r373_lane[None, :]) / $(1536)) % $(8)) * $(128)) + ((((r373_group % $(8)) * $(1536)) + (r373_row[:, None] * $(128)) + r373_lane[None, :]) % $(128)) + $(32768)), dtype=tl.bfloat16)
      r373_right = tl.load($(candidate_vbmm_11_ptr) + (((((((r373_group % $(8)) * $(6144)) + (r373_col[None, :] * $(128)) + r373_lane[:, None]) / $(128)) % $(48)) * $(1024)) + ((((((r373_group % $(8)) * $(6144)) + (r373_col[None, :] * $(128)) + r373_lane[:, None]) / $(6144)) % $(8)) * $(128)) + ((((r373_group % $(8)) * $(6144)) + (r373_col[None, :] * $(128)) + r373_lane[:, None]) % $(128))), dtype=tl.bfloat16)
      r373_value = tl.dot(r373_left, r373_right)
      r373_value = r373_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_71_ptr) + ((r373_group * $(576)) + (r373_row[:, None] * $(48)) + r373_col[None, :]), r373_value)
    }
    r374_z = tl.arange($(4608))
    r374_value = tl.load($(candidate_vbmm_71_ptr) + ((((r374_z / $(576)) % $(8)) * $(576)) + (((r374_z / $(48)) % $(12)) * $(48)) + (r374_z % $(48))), dtype=tl.bfloat16)
    r374_value = r374_value.to(tl.float32)
    r374_value = tl.broadcast_to(r374_value, [$(4608)])
    tl.store($(candidate_vto_copy_89_ptr) + r374_z, r374_value)
    r375_z = tl.arange($(4608))
    r375_lane = tl.arange($(48))
    r375_row_values = tl.load($(candidate_vsoftmax_3_ptr) + ((((r375_z[:, None] / $(576)) % $(8)) * $(576)) + (((r375_z[:, None] / $(48)) % $(12)) * $(48)) + (r375_lane[None, :] % $(48))), dtype=tl.float32)
    r375_row_grad = tl.load($(candidate_vto_copy_89_ptr) + ((((r375_z[:, None] / $(576)) % $(8)) * $(576)) + (((r375_z[:, None] / $(48)) % $(12)) * $(48)) + (r375_lane[None, :] % $(48))), dtype=tl.float32)
    r375_weighted = r375_row_grad * r375_row_values
    r375_dot_grad = tl.sum(r375_weighted, axis=1)
    r375_value = (tl.load($(candidate_vto_copy_89_ptr) + ((((r375_z / $(576)) % $(8)) * $(576)) + (((r375_z / $(48)) % $(12)) * $(48)) + (r375_z % $(48))), dtype=tl.float32) - r375_dot_grad) * tl.load($(candidate_vsoftmax_3_ptr) + ((((r375_z / $(576)) % $(8)) * $(576)) + (((r375_z / $(48)) % $(12)) * $(48)) + (r375_z % $(48))), dtype=tl.float32)
    r375_value = r375_value.to(tl.float32)
    r375_value = tl.broadcast_to(r375_value, [$(4608)])
    tl.store($(candidate_vsoftmax_backward_data_8_ptr) + r375_z, r375_value)
    r376_z = tl.arange($(4608))
    r376_value = tl.where(tl.load($(candidate_vgt_3_ptr) + ((((r376_z / $(48)) % $(12)) * $(48)) + (r376_z % $(48))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_8_ptr) + ((((r376_z / $(576)) % $(8)) * $(576)) + (((r376_z / $(48)) % $(12)) * $(48)) + (r376_z % $(48))), dtype=tl.float32))
    r376_value = r376_value.to(tl.float32)
    r376_value = tl.broadcast_to(r376_value, [$(4608)])
    tl.store($(candidate_vmasked_fill_20_ptr) + r376_z, r376_value)
    r377_z = tl.arange($(4608))
    r377_value = (tl.load($(candidate_vmasked_fill_20_ptr) + ((((r377_z / $(576)) % $(8)) * $(576)) + (((r377_z / $(48)) % $(12)) * $(48)) + (r377_z % $(48))), dtype=tl.float32) * 0.07216878364870322)
    r377_value = r377_value.to(tl.float32)
    r377_value = tl.broadcast_to(r377_value, [$(4608)])
    tl.store($(candidate_vmul_43_ptr) + r377_z, r377_value)
    r378_row = tl.arange($(192))
    r378_col = tl.arange($(48))
    r378_lane = tl.arange($(12))
    for r378_group in range($(8)) {
      r378_left = tl.load($(candidate_vto_copy_38_ptr) + (((((((r378_group % $(8)) * $(2304)) + (r378_lane[None, :] * $(192)) + r378_row[:, None]) / $(2304)) % $(8)) * $(2304)) + ((((((r378_group % $(8)) * $(2304)) + (r378_lane[None, :] * $(192)) + r378_row[:, None]) / $(192)) % $(12)) * $(192)) + ((((r378_group % $(8)) * $(2304)) + (r378_lane[None, :] * $(192)) + r378_row[:, None]) % $(192))), dtype=tl.float32)
      r378_right = tl.load($(candidate_vmul_43_ptr) + (((((((r378_group % $(8)) * $(576)) + (r378_lane[:, None] * $(48)) + r378_col[None, :]) / $(576)) % $(8)) * $(576)) + ((((((r378_group % $(8)) * $(576)) + (r378_lane[:, None] * $(48)) + r378_col[None, :]) / $(48)) % $(12)) * $(48)) + ((((r378_group % $(8)) * $(576)) + (r378_lane[:, None] * $(48)) + r378_col[None, :]) % $(48))), dtype=tl.float32)
      r378_value = tl.dot(r378_left, r378_right)
      r378_value = r378_value.to(tl.float32)
      tl.store($(candidate_vbmm_72_ptr) + ((r378_group * $(9216)) + (r378_row[:, None] * $(48)) + r378_col[None, :]), r378_value)
    }
    r379_row = tl.arange($(12))
    r379_col = tl.arange($(192))
    r379_lane = tl.arange($(48))
    for r379_group in range($(8)) {
      r379_left = tl.load($(candidate_vmul_43_ptr) + (((((((r379_group % $(8)) * $(576)) + (r379_row[:, None] * $(48)) + r379_lane[None, :]) / $(576)) % $(8)) * $(576)) + ((((((r379_group % $(8)) * $(576)) + (r379_row[:, None] * $(48)) + r379_lane[None, :]) / $(48)) % $(12)) * $(48)) + ((((r379_group % $(8)) * $(576)) + (r379_row[:, None] * $(48)) + r379_lane[None, :]) % $(48))), dtype=tl.float32)
      r379_right = tl.load($(candidate_vto_copy_39_ptr) + (((((((r379_group % $(8)) * $(9216)) + (r379_col[None, :] * $(48)) + r379_lane[:, None]) / $(9216)) % $(8)) * $(9216)) + (((((r379_group % $(8)) * $(9216)) + (r379_col[None, :] * $(48)) + r379_lane[:, None]) % $(48)) * $(192)) + (((((r379_group % $(8)) * $(9216)) + (r379_col[None, :] * $(48)) + r379_lane[:, None]) / $(48)) % $(192))), dtype=tl.float32)
      r379_value = tl.dot(r379_left, r379_right)
      r379_value = r379_value.to(tl.float32)
      tl.store($(candidate_vbmm_73_ptr) + ((r379_group * $(2304)) + (r379_row[:, None] * $(192)) + r379_col[None, :]), r379_value)
    }
    r380_z = tl.arange($(131072))
    r380_value = tl.where(((((r380_z / $(1024)) % $(128)) >= $(0)) & (((r380_z / $(1024)) % $(128)) < $(48)) & ((((r380_z / $(1024)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vbmm_70_ptr) + ((((((((r380_z / $(128)) % $(8)) * $(6144)) + ((((r380_z / $(1024)) % $(128)) - $(0)) * $(128)) + (r380_z % $(128))) / $(6144)) % $(8)) * $(6144)) + ((((((r380_z / $(128)) % $(8)) * $(48)) + (((r380_z / $(1024)) % $(128)) - $(0))) % $(48)) * $(128)) + (r380_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r380_value = r380_value.to(tl.bfloat16)
    r380_value = tl.broadcast_to(r380_value, [$(131072)])
    tl.store($(candidate_vslice_backward_32_ptr) + r380_z, r380_value)
    r381_z = tl.arange($(131072))
    r381_value = (tl.load($(candidate_vadd_38_ptr) + ((((r381_z / $(1024)) % $(128)) * $(1024)) + (((r381_z / $(128)) % $(8)) * $(128)) + (r381_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_32_ptr) + ((((r381_z / $(1024)) % $(128)) * $(1024)) + (((r381_z / $(128)) % $(8)) * $(128)) + (r381_z % $(128))), dtype=tl.bfloat16))
    r381_value = r381_value.to(tl.bfloat16)
    r381_value = tl.broadcast_to(r381_value, [$(131072)])
    tl.store($(candidate_vadd_41_ptr) + r381_z, r381_value)
    r382_z = tl.arange($(73728))
    r382_value = tl.load($(candidate_vbmm_72_ptr) + ((((r382_z / $(9216)) % $(8)) * $(9216)) + ((r382_z % $(192)) * $(48)) + ((r382_z / $(192)) % $(48))), dtype=tl.float32)
    r382_value = r382_value.to(tl.bfloat16)
    r382_value = tl.broadcast_to(r382_value, [$(73728)])
    tl.store($(candidate_vto_copy_90_ptr) + r382_z, r382_value)
    r383_z = tl.arange($(196608))
    r383_value = tl.where(((((r383_z / $(1536)) % $(128)) >= $(0)) & (((r383_z / $(1536)) % $(128)) < $(48)) & ((((r383_z / $(1536)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vto_copy_90_ptr) + ((((r383_z / $(192)) % $(8)) * $(9216)) + (((((r383_z / $(1536)) % $(128)) - $(0)) % $(48)) * $(192)) + (r383_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r383_value = r383_value.to(tl.bfloat16)
    r383_value = tl.broadcast_to(r383_value, [$(196608)])
    tl.store($(candidate_vslice_backward_33_ptr) + r383_z, r383_value)
    r384_z = tl.arange($(196608))
    r384_value = (tl.load($(candidate_vadd_39_ptr) + ((((r384_z / $(1536)) % $(128)) * $(1536)) + (((r384_z / $(192)) % $(8)) * $(192)) + (r384_z % $(192))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_33_ptr) + ((((r384_z / $(1536)) % $(128)) * $(1536)) + (((r384_z / $(192)) % $(8)) * $(192)) + (r384_z % $(192))), dtype=tl.bfloat16))
    r384_value = r384_value.to(tl.bfloat16)
    r384_value = tl.broadcast_to(r384_value, [$(196608)])
    tl.store($(candidate_vadd_42_ptr) + r384_z, r384_value)
    r385_z = tl.arange($(18432))
    r385_value = tl.load($(candidate_vbmm_73_ptr) + ((((r385_z / $(2304)) % $(8)) * $(2304)) + (((r385_z / $(192)) % $(12)) * $(192)) + (r385_z % $(192))), dtype=tl.float32)
    r385_value = r385_value.to(tl.bfloat16)
    r385_value = tl.broadcast_to(r385_value, [$(18432)])
    tl.store($(candidate_vto_copy_91_ptr) + r385_z, r385_value)
    r386_z = tl.arange($(49152))
    r386_value = tl.where(((((r386_z / $(1536)) % $(32)) >= $(0)) & (((r386_z / $(1536)) % $(32)) < $(12)) & ((((r386_z / $(1536)) % $(32)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vto_copy_91_ptr) + ((((r386_z / $(192)) % $(8)) * $(2304)) + (((((r386_z / $(1536)) % $(32)) - $(0)) % $(12)) * $(192)) + (r386_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r386_value = r386_value.to(tl.bfloat16)
    r386_value = tl.broadcast_to(r386_value, [$(49152)])
    tl.store($(candidate_vslice_backward_34_ptr) + r386_z, r386_value)
    r387_z = tl.arange($(49152))
    r387_value = (tl.load($(candidate_vadd_40_ptr) + ((((r387_z / $(1536)) % $(32)) * $(1536)) + (((r387_z / $(192)) % $(8)) * $(192)) + (r387_z % $(192))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_34_ptr) + ((((r387_z / $(1536)) % $(32)) * $(1536)) + (((r387_z / $(192)) % $(8)) * $(192)) + (r387_z % $(192))), dtype=tl.bfloat16))
    r387_value = r387_value.to(tl.bfloat16)
    r387_value = tl.broadcast_to(r387_value, [$(49152)])
    tl.store($(candidate_vadd_43_ptr) + r387_z, r387_value)
    r388_z = tl.arange($(0))
    r388_value = tl.cast(0.0, tl.bfloat16)
    r388_value = r388_value.to(tl.bfloat16)
    r388_value = tl.broadcast_to(r388_value, [$(0)])
    tl.store($(candidate_vfill_12_ptr) + r388_z, r388_value)
    r389_z = tl.arange($(131072))
    r389_value = tl.where(((((r389_z / $(1024)) % $(128)) >= $(48)) & (((r389_z / $(1024)) % $(128)) < $(48)) & ((((r389_z / $(1024)) % $(128)) - $(48)) % $(1) == $(0))), tl.load($(candidate_vfill_12_ptr) + ((((((r389_z / $(1024)) % $(128)) - $(48)) % $(0)) * $(1024)) + (((r389_z / $(128)) % $(8)) * $(128)) + (r389_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vadd_41_ptr) + ((((r389_z / $(1024)) % $(128)) * $(1024)) + (((r389_z / $(128)) % $(8)) * $(128)) + (r389_z % $(128))), dtype=tl.bfloat16))
    r389_value = r389_value.to(tl.bfloat16)
    r389_value = tl.broadcast_to(r389_value, [$(131072)])
    tl.store($(candidate_vslice_scatter_12_ptr) + r389_z, r389_value)
    r390_z = tl.arange($(0))
    r390_value = tl.cast(0.0, tl.bfloat16)
    r390_value = r390_value.to(tl.bfloat16)
    r390_value = tl.broadcast_to(r390_value, [$(0)])
    tl.store($(candidate_vfill_13_ptr) + r390_z, r390_value)
    r391_z = tl.arange($(131072))
    r391_value = tl.where(((((r391_z / $(1024)) % $(128)) >= $(64)) & (((r391_z / $(1024)) % $(128)) < $(64)) & ((((r391_z / $(1024)) % $(128)) - $(64)) % $(1) == $(0))), tl.load($(candidate_vfill_13_ptr) + ((((((r391_z / $(1024)) % $(128)) - $(64)) % $(0)) * $(1024)) + (((r391_z / $(128)) % $(8)) * $(128)) + (r391_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vslice_scatter_12_ptr) + ((((r391_z / $(1024)) % $(128)) * $(1024)) + (((r391_z / $(128)) % $(8)) * $(128)) + (r391_z % $(128))), dtype=tl.bfloat16))
    r391_value = r391_value.to(tl.bfloat16)
    r391_value = tl.broadcast_to(r391_value, [$(131072)])
    tl.store($(candidate_vslice_scatter_13_ptr) + r391_z, r391_value)
    r392_z = tl.arange($(0))
    r392_value = tl.cast(0.0, tl.bfloat16)
    r392_value = r392_value.to(tl.bfloat16)
    r392_value = tl.broadcast_to(r392_value, [$(0)])
    tl.store($(candidate_vfill_14_ptr) + r392_z, r392_value)
    r393_z = tl.arange($(131072))
    r393_value = tl.where(((((r393_z / $(1024)) % $(128)) >= $(128)) & (((r393_z / $(1024)) % $(128)) < $(128)) & ((((r393_z / $(1024)) % $(128)) - $(128)) % $(1) == $(0))), tl.load($(candidate_vfill_14_ptr) + ((((((r393_z / $(1024)) % $(128)) - $(128)) % $(0)) * $(1024)) + (((r393_z / $(128)) % $(8)) * $(128)) + (r393_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vslice_scatter_13_ptr) + ((((r393_z / $(1024)) % $(128)) * $(1024)) + (((r393_z / $(128)) % $(8)) * $(128)) + (r393_z % $(128))), dtype=tl.bfloat16))
    r393_value = r393_value.to(tl.bfloat16)
    r393_value = tl.broadcast_to(r393_value, [$(131072)])
    tl.store($(candidate_vslice_scatter_14_ptr) + r393_z, r393_value)
    r394_z = tl.arange($(0))
    r394_value = tl.cast(0.0, tl.bfloat16)
    r394_value = r394_value.to(tl.bfloat16)
    r394_value = tl.broadcast_to(r394_value, [$(0)])
    tl.store($(candidate_vfill_15_ptr) + r394_z, r394_value)
    r395_z = tl.arange($(196608))
    r395_value = tl.where(((((r395_z / $(1536)) % $(128)) >= $(48)) & (((r395_z / $(1536)) % $(128)) < $(48)) & ((((r395_z / $(1536)) % $(128)) - $(48)) % $(1) == $(0))), tl.load($(candidate_vfill_15_ptr) + ((((((r395_z / $(1536)) % $(128)) - $(48)) % $(0)) * $(1536)) + (((r395_z / $(192)) % $(8)) * $(192)) + (r395_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vadd_42_ptr) + ((((r395_z / $(1536)) % $(128)) * $(1536)) + (((r395_z / $(192)) % $(8)) * $(192)) + (r395_z % $(192))), dtype=tl.bfloat16))
    r395_value = r395_value.to(tl.bfloat16)
    r395_value = tl.broadcast_to(r395_value, [$(196608)])
    tl.store($(candidate_vslice_scatter_15_ptr) + r395_z, r395_value)
    r396_z = tl.arange($(0))
    r396_value = tl.cast(0.0, tl.bfloat16)
    r396_value = r396_value.to(tl.bfloat16)
    r396_value = tl.broadcast_to(r396_value, [$(0)])
    tl.store($(candidate_vfill_16_ptr) + r396_z, r396_value)
    r397_z = tl.arange($(196608))
    r397_value = tl.where(((((r397_z / $(1536)) % $(128)) >= $(64)) & (((r397_z / $(1536)) % $(128)) < $(64)) & ((((r397_z / $(1536)) % $(128)) - $(64)) % $(1) == $(0))), tl.load($(candidate_vfill_16_ptr) + ((((((r397_z / $(1536)) % $(128)) - $(64)) % $(0)) * $(1536)) + (((r397_z / $(192)) % $(8)) * $(192)) + (r397_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vslice_scatter_15_ptr) + ((((r397_z / $(1536)) % $(128)) * $(1536)) + (((r397_z / $(192)) % $(8)) * $(192)) + (r397_z % $(192))), dtype=tl.bfloat16))
    r397_value = r397_value.to(tl.bfloat16)
    r397_value = tl.broadcast_to(r397_value, [$(196608)])
    tl.store($(candidate_vslice_scatter_16_ptr) + r397_z, r397_value)
    r398_z = tl.arange($(0))
    r398_value = tl.cast(0.0, tl.bfloat16)
    r398_value = r398_value.to(tl.bfloat16)
    r398_value = tl.broadcast_to(r398_value, [$(0)])
    tl.store($(candidate_vfill_17_ptr) + r398_z, r398_value)
    r399_z = tl.arange($(196608))
    r399_value = tl.where(((((r399_z / $(1536)) % $(128)) >= $(128)) & (((r399_z / $(1536)) % $(128)) < $(128)) & ((((r399_z / $(1536)) % $(128)) - $(128)) % $(1) == $(0))), tl.load($(candidate_vfill_17_ptr) + ((((((r399_z / $(1536)) % $(128)) - $(128)) % $(0)) * $(1536)) + (((r399_z / $(192)) % $(8)) * $(192)) + (r399_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vslice_scatter_16_ptr) + ((((r399_z / $(1536)) % $(128)) * $(1536)) + (((r399_z / $(192)) % $(8)) * $(192)) + (r399_z % $(192))), dtype=tl.bfloat16))
    r399_value = r399_value.to(tl.bfloat16)
    r399_value = tl.broadcast_to(r399_value, [$(196608)])
    tl.store($(candidate_vslice_scatter_17_ptr) + r399_z, r399_value)
    r400_z = tl.arange($(8192))
    r400_lane = tl.arange($(8))
    r400_value = tl.sum(tl.load($(candidate_vslice_scatter_17_ptr) + ((((r400_z[:, None] / $(64)) % $(128)) * $(1536)) + (r400_lane[None, :] * $(192)) + (r400_z[:, None] % $(64)) + $(128)), dtype=tl.bfloat16), axis=1)
    r400_value = r400_value.to(tl.bfloat16)
    r400_value = tl.broadcast_to(r400_value, [$(8192)])
    tl.store($(candidate_vsum_3_ptr) + r400_z, r400_value)
    r401_row = tl.arange($(128))
    r401_col = tl.arange($(256))
    r401_lane = tl.arange($(1024))
    for r401_group in range($(1)) {
      r401_left = tl.load($(candidate_vslice_scatter_14_ptr) + ((((((r401_row[:, None] * $(1024)) + r401_lane[None, :]) / $(1024)) % $(128)) * $(1024)) + (((((r401_row[:, None] * $(1024)) + r401_lane[None, :]) / $(128)) % $(8)) * $(128)) + (((r401_row[:, None] * $(1024)) + r401_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r401_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r401_col[None, :] * $(1024)) + r401_lane[:, None]) / $(128)) % $(8)) * $(65536)) + ((((r401_col[None, :] * $(1024)) + r401_lane[:, None]) % $(128)) * $(256)) + ((((r401_col[None, :] * $(1024)) + r401_lane[:, None]) / $(1024)) % $(256)) + $(32768)), dtype=tl.bfloat16)
      r401_value = tl.dot(r401_left, r401_right)
      r401_value = r401_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_74_ptr) + ((r401_group * $(32768)) + (r401_row[:, None] * $(256)) + r401_col[None, :]), r401_value)
    }
    r402_row = tl.arange($(128))
    r402_col = tl.arange($(256))
    r402_lane = tl.arange($(1024))
    for r402_group in range($(1)) {
      r402_left = tl.load($(candidate_vslice_scatter_17_ptr) + ((((((r402_row[:, None] * $(1024)) + r402_lane[None, :]) / $(1024)) % $(128)) * $(1536)) + (((((r402_row[:, None] * $(1024)) + r402_lane[None, :]) / $(128)) % $(8)) * $(192)) + (((r402_row[:, None] * $(1024)) + r402_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r402_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r402_col[None, :] * $(1024)) + r402_lane[:, None]) / $(128)) % $(8)) * $(65536)) + ((((r402_col[None, :] * $(1024)) + r402_lane[:, None]) % $(128)) * $(256)) + ((((r402_col[None, :] * $(1024)) + r402_lane[:, None]) / $(1024)) % $(256))), dtype=tl.bfloat16)
      r402_value = tl.dot(r402_left, r402_right)
      r402_value = r402_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_75_ptr) + ((r402_group * $(32768)) + (r402_row[:, None] * $(256)) + r402_col[None, :]), r402_value)
    }
    r403_z = tl.arange($(32768))
    r403_value = (tl.load($(candidate_vbmm_74_ptr) + ((((r403_z / $(256)) % $(128)) * $(256)) + (r403_z % $(256))), dtype=tl.bfloat16) + tl.load($(candidate_vbmm_75_ptr) + ((((r403_z / $(256)) % $(128)) * $(256)) + (r403_z % $(256))), dtype=tl.bfloat16))
    r403_value = r403_value.to(tl.bfloat16)
    r403_value = tl.broadcast_to(r403_value, [$(32768)])
    tl.store($(candidate_vadd_44_ptr) + r403_z, r403_value)
    r404_z = tl.arange($(32768))
    r404_value = tl.where(($(0) == $(0)), tl.load($(candidate_vadd_44_ptr) + ((((r404_z / $(256)) % $(128)) * $(256)) + (r404_z % $(256))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r404_value = r404_value.to(tl.bfloat16)
    r404_value = tl.broadcast_to(r404_value, [$(32768)])
    tl.store($(candidate_vselect_backward_2_ptr) + r404_z, r404_value)
    r405_z = tl.arange($(32768))
    r405_value = tl.where(((((r405_z / $(256)) % $(128)) >= $(0)) & (((r405_z / $(256)) % $(128)) < $(128)) & ((((r405_z / $(256)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vselect_backward_2_ptr) + ((((((r405_z / $(256)) % $(128)) - $(0)) % $(128)) * $(256)) + (r405_z % $(256))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r405_value = r405_value.to(tl.bfloat16)
    r405_value = tl.broadcast_to(r405_value, [$(32768)])
    tl.store($(candidate_vslice_backward_35_ptr) + r405_z, r405_value)
    r406_z = tl.arange($(32768))
    r406_value = (tl.load($(candidate_vadd_34_ptr) + ((((r406_z / $(256)) % $(128)) * $(256)) + (r406_z % $(256))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_35_ptr) + ((((r406_z / $(256)) % $(128)) * $(256)) + (r406_z % $(256))), dtype=tl.bfloat16))
    r406_value = r406_value.to(tl.bfloat16)
    r406_value = tl.broadcast_to(r406_value, [$(32768)])
    tl.store($(candidate_vadd_45_ptr) + r406_z, r406_value)
    r407_z = tl.arange($(8192))
    r407_value = tl.where(((((r407_z / $(64)) % $(128)) >= $(0)) & (((r407_z / $(64)) % $(128)) < $(128)) & ((((r407_z / $(64)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vsum_3_ptr) + ((((((r407_z / $(64)) % $(128)) - $(0)) % $(128)) * $(64)) + (r407_z % $(64))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r407_value = r407_value.to(tl.bfloat16)
    r407_value = tl.broadcast_to(r407_value, [$(8192)])
    tl.store($(candidate_vslice_backward_36_ptr) + r407_z, r407_value)
    r408_z = tl.arange($(8192))
    r408_value = (tl.load($(candidate_vadd_35_ptr) + ((((r408_z / $(64)) % $(128)) * $(64)) + (r408_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_36_ptr) + ((((r408_z / $(64)) % $(128)) * $(64)) + (r408_z % $(64))), dtype=tl.bfloat16))
    r408_value = r408_value.to(tl.bfloat16)
    r408_value = tl.broadcast_to(r408_value, [$(8192)])
    tl.store($(candidate_vadd_46_ptr) + r408_z, r408_value)
    r409_z = tl.arange($(131072))
    r409_value = tl.where(((((r409_z / $(1024)) % $(128)) >= $(32)) & (((r409_z / $(1024)) % $(128)) < $(64)) & ((((r409_z / $(1024)) % $(128)) - $(32)) % $(1) == $(0))), tl.load($(candidate_vadd_43_ptr) + ((((((r409_z / $(1024)) % $(128)) - $(32)) % $(32)) * $(1536)) + (((r409_z / $(128)) % $(8)) * $(192)) + (r409_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r409_value = r409_value.to(tl.bfloat16)
    r409_value = tl.broadcast_to(r409_value, [$(131072)])
    tl.store($(candidate_vslice_backward_37_ptr) + r409_z, r409_value)
    r410_z = tl.arange($(131072))
    r410_value = (tl.load($(candidate_vadd_36_ptr) + ((((r410_z / $(1024)) % $(128)) * $(1024)) + (((r410_z / $(128)) % $(8)) * $(128)) + (r410_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_37_ptr) + ((((r410_z / $(1024)) % $(128)) * $(1024)) + (((r410_z / $(128)) % $(8)) * $(128)) + (r410_z % $(128))), dtype=tl.bfloat16))
    r410_value = r410_value.to(tl.bfloat16)
    r410_value = tl.broadcast_to(r410_value, [$(131072)])
    tl.store($(candidate_vadd_47_ptr) + r410_z, r410_value)
    r411_z = tl.arange($(65536))
    r411_value = tl.where(((((r411_z / $(512)) % $(128)) >= $(32)) & (((r411_z / $(512)) % $(128)) < $(64)) & ((((r411_z / $(512)) % $(128)) - $(32)) % $(1) == $(0))), tl.load($(candidate_vadd_43_ptr) + ((((((r411_z / $(512)) % $(128)) - $(32)) % $(32)) * $(1536)) + (((r411_z / $(64)) % $(8)) * $(192)) + (r411_z % $(64)) + $(128)), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r411_value = r411_value.to(tl.bfloat16)
    r411_value = tl.broadcast_to(r411_value, [$(65536)])
    tl.store($(candidate_vslice_backward_38_ptr) + r411_z, r411_value)
    r412_z = tl.arange($(65536))
    r412_value = (tl.load($(candidate_vadd_37_ptr) + ((((r412_z / $(512)) % $(128)) * $(512)) + (((r412_z / $(64)) % $(8)) * $(64)) + (r412_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_38_ptr) + ((((r412_z / $(512)) % $(128)) * $(512)) + (((r412_z / $(64)) % $(8)) * $(64)) + (r412_z % $(64))), dtype=tl.bfloat16))
    r412_value = r412_value.to(tl.bfloat16)
    r412_value = tl.broadcast_to(r412_value, [$(65536)])
    tl.store($(candidate_vadd_48_ptr) + r412_z, r412_value)
    r413_row = tl.arange($(16))
    r413_col = tl.arange($(128))
    r413_lane = tl.arange($(16))
    for r413_group in range($(8)) {
      r413_left = tl.load($(candidate_vto_copy_37_ptr) + (((((((r413_group % $(8)) * $(256)) + (r413_lane[None, :] * $(16)) + r413_row[:, None]) / $(256)) % $(8)) * $(256)) + ((((((r413_group % $(8)) * $(256)) + (r413_lane[None, :] * $(16)) + r413_row[:, None]) / $(16)) % $(16)) * $(16)) + ((((r413_group % $(8)) * $(256)) + (r413_lane[None, :] * $(16)) + r413_row[:, None]) % $(16))), dtype=tl.bfloat16)
      r413_right = tl.load($(candidate_vmm_4_ptr) + (((((((r413_group % $(8)) * $(2048)) + (r413_lane[:, None] * $(128)) + r413_col[None, :]) / $(128)) % $(16)) * $(1024)) + ((((((r413_group % $(8)) * $(2048)) + (r413_lane[:, None] * $(128)) + r413_col[None, :]) / $(2048)) % $(8)) * $(128)) + ((((r413_group % $(8)) * $(2048)) + (r413_lane[:, None] * $(128)) + r413_col[None, :]) % $(128)) + $(16384)), dtype=tl.bfloat16)
      r413_value = tl.dot(r413_left, r413_right)
      r413_value = r413_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_76_ptr) + ((r413_group * $(2048)) + (r413_row[:, None] * $(128)) + r413_col[None, :]), r413_value)
    }
    r414_row = tl.arange($(16))
    r414_col = tl.arange($(16))
    r414_lane = tl.arange($(128))
    for r414_group in range($(8)) {
      r414_left = tl.load($(candidate_vmm_4_ptr) + (((((((r414_group % $(8)) * $(2048)) + (r414_row[:, None] * $(128)) + r414_lane[None, :]) / $(128)) % $(16)) * $(1024)) + ((((((r414_group % $(8)) * $(2048)) + (r414_row[:, None] * $(128)) + r414_lane[None, :]) / $(2048)) % $(8)) * $(128)) + ((((r414_group % $(8)) * $(2048)) + (r414_row[:, None] * $(128)) + r414_lane[None, :]) % $(128)) + $(16384)), dtype=tl.bfloat16)
      r414_right = tl.load($(candidate_vbmm_3_ptr) + (((((((r414_group % $(8)) * $(2048)) + (r414_col[None, :] * $(128)) + r414_lane[:, None]) / $(128)) % $(16)) * $(1024)) + ((((((r414_group % $(8)) * $(2048)) + (r414_col[None, :] * $(128)) + r414_lane[:, None]) / $(2048)) % $(8)) * $(128)) + ((((r414_group % $(8)) * $(2048)) + (r414_col[None, :] * $(128)) + r414_lane[:, None]) % $(128)) + $(65536)), dtype=tl.bfloat16)
      r414_value = tl.dot(r414_left, r414_right)
      r414_value = r414_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_77_ptr) + ((r414_group * $(256)) + (r414_row[:, None] * $(16)) + r414_col[None, :]), r414_value)
    }
    r415_z = tl.arange($(2048))
    r415_value = tl.load($(candidate_vbmm_77_ptr) + ((((r415_z / $(256)) % $(8)) * $(256)) + (((r415_z / $(16)) % $(16)) * $(16)) + (r415_z % $(16))), dtype=tl.bfloat16)
    r415_value = r415_value.to(tl.float32)
    r415_value = tl.broadcast_to(r415_value, [$(2048)])
    tl.store($(candidate_vto_copy_92_ptr) + r415_z, r415_value)
    r416_z = tl.arange($(2048))
    r416_lane = tl.arange($(16))
    r416_row_values = tl.load($(candidate_vsoftmax_2_ptr) + ((((r416_z[:, None] / $(256)) % $(8)) * $(256)) + (((r416_z[:, None] / $(16)) % $(16)) * $(16)) + (r416_lane[None, :] % $(16))), dtype=tl.float32)
    r416_row_grad = tl.load($(candidate_vto_copy_92_ptr) + ((((r416_z[:, None] / $(256)) % $(8)) * $(256)) + (((r416_z[:, None] / $(16)) % $(16)) * $(16)) + (r416_lane[None, :] % $(16))), dtype=tl.float32)
    r416_weighted = r416_row_grad * r416_row_values
    r416_dot_grad = tl.sum(r416_weighted, axis=1)
    r416_value = (tl.load($(candidate_vto_copy_92_ptr) + ((((r416_z / $(256)) % $(8)) * $(256)) + (((r416_z / $(16)) % $(16)) * $(16)) + (r416_z % $(16))), dtype=tl.float32) - r416_dot_grad) * tl.load($(candidate_vsoftmax_2_ptr) + ((((r416_z / $(256)) % $(8)) * $(256)) + (((r416_z / $(16)) % $(16)) * $(16)) + (r416_z % $(16))), dtype=tl.float32)
    r416_value = r416_value.to(tl.float32)
    r416_value = tl.broadcast_to(r416_value, [$(2048)])
    tl.store($(candidate_vsoftmax_backward_data_9_ptr) + r416_z, r416_value)
    r417_z = tl.arange($(2048))
    r417_value = tl.where(tl.load($(candidate_vgt_2_ptr) + ((((r417_z / $(16)) % $(16)) * $(16)) + (r417_z % $(16))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_9_ptr) + ((((r417_z / $(256)) % $(8)) * $(256)) + (((r417_z / $(16)) % $(16)) * $(16)) + (r417_z % $(16))), dtype=tl.float32))
    r417_value = r417_value.to(tl.float32)
    r417_value = tl.broadcast_to(r417_value, [$(2048)])
    tl.store($(candidate_vmasked_fill_21_ptr) + r417_z, r417_value)
    r418_z = tl.arange($(2048))
    r418_value = (tl.load($(candidate_vmasked_fill_21_ptr) + ((((r418_z / $(256)) % $(8)) * $(256)) + (((r418_z / $(16)) % $(16)) * $(16)) + (r418_z % $(16))), dtype=tl.float32) * 0.07216878364870322)
    r418_value = r418_value.to(tl.float32)
    r418_value = tl.broadcast_to(r418_value, [$(2048)])
    tl.store($(candidate_vmul_44_ptr) + r418_z, r418_value)
    r419_row = tl.arange($(192))
    r419_col = tl.arange($(16))
    r419_lane = tl.arange($(16))
    for r419_group in range($(8)) {
      r419_left = tl.load($(candidate_vto_copy_35_ptr) + (((((((r419_group % $(8)) * $(3072)) + (r419_lane[None, :] * $(192)) + r419_row[:, None]) / $(3072)) % $(8)) * $(3072)) + ((((((r419_group % $(8)) * $(3072)) + (r419_lane[None, :] * $(192)) + r419_row[:, None]) / $(192)) % $(16)) * $(192)) + ((((r419_group % $(8)) * $(3072)) + (r419_lane[None, :] * $(192)) + r419_row[:, None]) % $(192))), dtype=tl.float32)
      r419_right = tl.load($(candidate_vmul_44_ptr) + (((((((r419_group % $(8)) * $(256)) + (r419_lane[:, None] * $(16)) + r419_col[None, :]) / $(256)) % $(8)) * $(256)) + ((((((r419_group % $(8)) * $(256)) + (r419_lane[:, None] * $(16)) + r419_col[None, :]) / $(16)) % $(16)) * $(16)) + ((((r419_group % $(8)) * $(256)) + (r419_lane[:, None] * $(16)) + r419_col[None, :]) % $(16))), dtype=tl.float32)
      r419_value = tl.dot(r419_left, r419_right)
      r419_value = r419_value.to(tl.float32)
      tl.store($(candidate_vbmm_78_ptr) + ((r419_group * $(3072)) + (r419_row[:, None] * $(16)) + r419_col[None, :]), r419_value)
    }
    r420_row = tl.arange($(16))
    r420_col = tl.arange($(192))
    r420_lane = tl.arange($(16))
    for r420_group in range($(8)) {
      r420_left = tl.load($(candidate_vmul_44_ptr) + (((((((r420_group % $(8)) * $(256)) + (r420_row[:, None] * $(16)) + r420_lane[None, :]) / $(256)) % $(8)) * $(256)) + ((((((r420_group % $(8)) * $(256)) + (r420_row[:, None] * $(16)) + r420_lane[None, :]) / $(16)) % $(16)) * $(16)) + ((((r420_group % $(8)) * $(256)) + (r420_row[:, None] * $(16)) + r420_lane[None, :]) % $(16))), dtype=tl.float32)
      r420_right = tl.load($(candidate_vto_copy_36_ptr) + (((((((r420_group % $(8)) * $(3072)) + (r420_col[None, :] * $(16)) + r420_lane[:, None]) / $(3072)) % $(8)) * $(3072)) + (((((r420_group % $(8)) * $(3072)) + (r420_col[None, :] * $(16)) + r420_lane[:, None]) % $(16)) * $(192)) + (((((r420_group % $(8)) * $(3072)) + (r420_col[None, :] * $(16)) + r420_lane[:, None]) / $(16)) % $(192))), dtype=tl.float32)
      r420_value = tl.dot(r420_left, r420_right)
      r420_value = r420_value.to(tl.float32)
      tl.store($(candidate_vbmm_79_ptr) + ((r420_group * $(3072)) + (r420_row[:, None] * $(192)) + r420_col[None, :]), r420_value)
    }
    r421_z = tl.arange($(131072))
    r421_value = tl.where(((((r421_z / $(1024)) % $(128)) >= $(64)) & (((r421_z / $(1024)) % $(128)) < $(80)) & ((((r421_z / $(1024)) % $(128)) - $(64)) % $(1) == $(0))), tl.load($(candidate_vbmm_76_ptr) + ((((((((r421_z / $(128)) % $(8)) * $(2048)) + ((((r421_z / $(1024)) % $(128)) - $(64)) * $(128)) + (r421_z % $(128))) / $(2048)) % $(8)) * $(2048)) + ((((((r421_z / $(128)) % $(8)) * $(16)) + (((r421_z / $(1024)) % $(128)) - $(64))) % $(16)) * $(128)) + (r421_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r421_value = r421_value.to(tl.bfloat16)
    r421_value = tl.broadcast_to(r421_value, [$(131072)])
    tl.store($(candidate_vslice_backward_39_ptr) + r421_z, r421_value)
    r422_z = tl.arange($(24576))
    r422_value = tl.load($(candidate_vbmm_78_ptr) + ((((r422_z / $(3072)) % $(8)) * $(3072)) + ((r422_z % $(192)) * $(16)) + ((r422_z / $(192)) % $(16))), dtype=tl.float32)
    r422_value = r422_value.to(tl.bfloat16)
    r422_value = tl.broadcast_to(r422_value, [$(24576)])
    tl.store($(candidate_vto_copy_93_ptr) + r422_z, r422_value)
    r423_z = tl.arange($(196608))
    r423_value = tl.where(((((r423_z / $(1536)) % $(128)) >= $(64)) & (((r423_z / $(1536)) % $(128)) < $(80)) & ((((r423_z / $(1536)) % $(128)) - $(64)) % $(1) == $(0))), tl.load($(candidate_vto_copy_93_ptr) + ((((r423_z / $(192)) % $(8)) * $(3072)) + (((((r423_z / $(1536)) % $(128)) - $(64)) % $(16)) * $(192)) + (r423_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r423_value = r423_value.to(tl.bfloat16)
    r423_value = tl.broadcast_to(r423_value, [$(196608)])
    tl.store($(candidate_vslice_backward_40_ptr) + r423_z, r423_value)
    r424_z = tl.arange($(24576))
    r424_value = tl.load($(candidate_vbmm_79_ptr) + ((((r424_z / $(3072)) % $(8)) * $(3072)) + (((r424_z / $(192)) % $(16)) * $(192)) + (r424_z % $(192))), dtype=tl.float32)
    r424_value = r424_value.to(tl.bfloat16)
    r424_value = tl.broadcast_to(r424_value, [$(24576)])
    tl.store($(candidate_vto_copy_94_ptr) + r424_z, r424_value)
    r425_z = tl.arange($(49152))
    r425_value = tl.where(((((r425_z / $(1536)) % $(32)) >= $(16)) & (((r425_z / $(1536)) % $(32)) < $(32)) & ((((r425_z / $(1536)) % $(32)) - $(16)) % $(1) == $(0))), tl.load($(candidate_vto_copy_94_ptr) + ((((r425_z / $(192)) % $(8)) * $(3072)) + (((((r425_z / $(1536)) % $(32)) - $(16)) % $(16)) * $(192)) + (r425_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r425_value = r425_value.to(tl.bfloat16)
    r425_value = tl.broadcast_to(r425_value, [$(49152)])
    tl.store($(candidate_vslice_backward_41_ptr) + r425_z, r425_value)
    r426_row = tl.arange($(4))
    r426_col = tl.arange($(128))
    r426_lane = tl.arange($(4))
    for r426_group in range($(8)) {
      r426_left = tl.load($(candidate_vto_copy_34_ptr) + (((((((r426_group % $(8)) * $(16)) + (r426_lane[None, :] * $(4)) + r426_row[:, None]) / $(16)) % $(8)) * $(16)) + ((((((r426_group % $(8)) * $(16)) + (r426_lane[None, :] * $(4)) + r426_row[:, None]) / $(4)) % $(4)) * $(4)) + ((((r426_group % $(8)) * $(16)) + (r426_lane[None, :] * $(4)) + r426_row[:, None]) % $(4))), dtype=tl.bfloat16)
      r426_right = tl.load($(candidate_vmm_4_ptr) + (((((((r426_group % $(8)) * $(512)) + (r426_lane[:, None] * $(128)) + r426_col[None, :]) / $(128)) % $(4)) * $(1024)) + ((((((r426_group % $(8)) * $(512)) + (r426_lane[:, None] * $(128)) + r426_col[None, :]) / $(512)) % $(8)) * $(128)) + ((((r426_group % $(8)) * $(512)) + (r426_lane[:, None] * $(128)) + r426_col[None, :]) % $(128)) + $(12288)), dtype=tl.bfloat16)
      r426_value = tl.dot(r426_left, r426_right)
      r426_value = r426_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_80_ptr) + ((r426_group * $(512)) + (r426_row[:, None] * $(128)) + r426_col[None, :]), r426_value)
    }
    r427_row = tl.arange($(4))
    r427_col = tl.arange($(4))
    r427_lane = tl.arange($(128))
    for r427_group in range($(8)) {
      r427_left = tl.load($(candidate_vmm_4_ptr) + (((((((r427_group % $(8)) * $(512)) + (r427_row[:, None] * $(128)) + r427_lane[None, :]) / $(128)) % $(4)) * $(1024)) + ((((((r427_group % $(8)) * $(512)) + (r427_row[:, None] * $(128)) + r427_lane[None, :]) / $(512)) % $(8)) * $(128)) + ((((r427_group % $(8)) * $(512)) + (r427_row[:, None] * $(128)) + r427_lane[None, :]) % $(128)) + $(12288)), dtype=tl.bfloat16)
      r427_right = tl.load($(candidate_vbmm_3_ptr) + (((((((r427_group % $(8)) * $(512)) + (r427_col[None, :] * $(128)) + r427_lane[:, None]) / $(128)) % $(4)) * $(1024)) + ((((((r427_group % $(8)) * $(512)) + (r427_col[None, :] * $(128)) + r427_lane[:, None]) / $(512)) % $(8)) * $(128)) + ((((r427_group % $(8)) * $(512)) + (r427_col[None, :] * $(128)) + r427_lane[:, None]) % $(128)) + $(49152)), dtype=tl.bfloat16)
      r427_value = tl.dot(r427_left, r427_right)
      r427_value = r427_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_81_ptr) + ((r427_group * $(16)) + (r427_row[:, None] * $(4)) + r427_col[None, :]), r427_value)
    }
    r428_z = tl.arange($(128))
    r428_value = tl.load($(candidate_vbmm_81_ptr) + ((((r428_z / $(16)) % $(8)) * $(16)) + (((r428_z / $(4)) % $(4)) * $(4)) + (r428_z % $(4))), dtype=tl.bfloat16)
    r428_value = r428_value.to(tl.float32)
    r428_value = tl.broadcast_to(r428_value, [$(128)])
    tl.store($(candidate_vto_copy_95_ptr) + r428_z, r428_value)
    r429_z = tl.arange($(128))
    r429_lane = tl.arange($(4))
    r429_row_values = tl.load($(candidate_vsoftmax_1_ptr) + ((((r429_z[:, None] / $(16)) % $(8)) * $(16)) + (((r429_z[:, None] / $(4)) % $(4)) * $(4)) + (r429_lane[None, :] % $(4))), dtype=tl.float32)
    r429_row_grad = tl.load($(candidate_vto_copy_95_ptr) + ((((r429_z[:, None] / $(16)) % $(8)) * $(16)) + (((r429_z[:, None] / $(4)) % $(4)) * $(4)) + (r429_lane[None, :] % $(4))), dtype=tl.float32)
    r429_weighted = r429_row_grad * r429_row_values
    r429_dot_grad = tl.sum(r429_weighted, axis=1)
    r429_value = (tl.load($(candidate_vto_copy_95_ptr) + ((((r429_z / $(16)) % $(8)) * $(16)) + (((r429_z / $(4)) % $(4)) * $(4)) + (r429_z % $(4))), dtype=tl.float32) - r429_dot_grad) * tl.load($(candidate_vsoftmax_1_ptr) + ((((r429_z / $(16)) % $(8)) * $(16)) + (((r429_z / $(4)) % $(4)) * $(4)) + (r429_z % $(4))), dtype=tl.float32)
    r429_value = r429_value.to(tl.float32)
    r429_value = tl.broadcast_to(r429_value, [$(128)])
    tl.store($(candidate_vsoftmax_backward_data_10_ptr) + r429_z, r429_value)
    r430_z = tl.arange($(128))
    r430_value = tl.where(tl.load($(candidate_vgt_1_ptr) + ((((r430_z / $(4)) % $(4)) * $(4)) + (r430_z % $(4))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_10_ptr) + ((((r430_z / $(16)) % $(8)) * $(16)) + (((r430_z / $(4)) % $(4)) * $(4)) + (r430_z % $(4))), dtype=tl.float32))
    r430_value = r430_value.to(tl.float32)
    r430_value = tl.broadcast_to(r430_value, [$(128)])
    tl.store($(candidate_vmasked_fill_22_ptr) + r430_z, r430_value)
    r431_z = tl.arange($(128))
    r431_value = (tl.load($(candidate_vmasked_fill_22_ptr) + ((((r431_z / $(16)) % $(8)) * $(16)) + (((r431_z / $(4)) % $(4)) * $(4)) + (r431_z % $(4))), dtype=tl.float32) * 0.07216878364870322)
    r431_value = r431_value.to(tl.float32)
    r431_value = tl.broadcast_to(r431_value, [$(128)])
    tl.store($(candidate_vmul_45_ptr) + r431_z, r431_value)
    r432_row = tl.arange($(192))
    r432_col = tl.arange($(4))
    r432_lane = tl.arange($(4))
    for r432_group in range($(8)) {
      r432_left = tl.load($(candidate_vto_copy_32_ptr) + (((((((r432_group % $(8)) * $(768)) + (r432_lane[None, :] * $(192)) + r432_row[:, None]) / $(768)) % $(8)) * $(768)) + ((((((r432_group % $(8)) * $(768)) + (r432_lane[None, :] * $(192)) + r432_row[:, None]) / $(192)) % $(4)) * $(192)) + ((((r432_group % $(8)) * $(768)) + (r432_lane[None, :] * $(192)) + r432_row[:, None]) % $(192))), dtype=tl.float32)
      r432_right = tl.load($(candidate_vmul_45_ptr) + (((((((r432_group % $(8)) * $(16)) + (r432_lane[:, None] * $(4)) + r432_col[None, :]) / $(16)) % $(8)) * $(16)) + ((((((r432_group % $(8)) * $(16)) + (r432_lane[:, None] * $(4)) + r432_col[None, :]) / $(4)) % $(4)) * $(4)) + ((((r432_group % $(8)) * $(16)) + (r432_lane[:, None] * $(4)) + r432_col[None, :]) % $(4))), dtype=tl.float32)
      r432_value = tl.dot(r432_left, r432_right)
      r432_value = r432_value.to(tl.float32)
      tl.store($(candidate_vbmm_82_ptr) + ((r432_group * $(768)) + (r432_row[:, None] * $(4)) + r432_col[None, :]), r432_value)
    }
    r433_row = tl.arange($(4))
    r433_col = tl.arange($(192))
    r433_lane = tl.arange($(4))
    for r433_group in range($(8)) {
      r433_left = tl.load($(candidate_vmul_45_ptr) + (((((((r433_group % $(8)) * $(16)) + (r433_row[:, None] * $(4)) + r433_lane[None, :]) / $(16)) % $(8)) * $(16)) + ((((((r433_group % $(8)) * $(16)) + (r433_row[:, None] * $(4)) + r433_lane[None, :]) / $(4)) % $(4)) * $(4)) + ((((r433_group % $(8)) * $(16)) + (r433_row[:, None] * $(4)) + r433_lane[None, :]) % $(4))), dtype=tl.float32)
      r433_right = tl.load($(candidate_vto_copy_33_ptr) + (((((((r433_group % $(8)) * $(768)) + (r433_col[None, :] * $(4)) + r433_lane[:, None]) / $(768)) % $(8)) * $(768)) + (((((r433_group % $(8)) * $(768)) + (r433_col[None, :] * $(4)) + r433_lane[:, None]) % $(4)) * $(192)) + (((((r433_group % $(8)) * $(768)) + (r433_col[None, :] * $(4)) + r433_lane[:, None]) / $(4)) % $(192))), dtype=tl.float32)
      r433_value = tl.dot(r433_left, r433_right)
      r433_value = r433_value.to(tl.float32)
      tl.store($(candidate_vbmm_83_ptr) + ((r433_group * $(768)) + (r433_row[:, None] * $(192)) + r433_col[None, :]), r433_value)
    }
    r434_z = tl.arange($(131072))
    r434_value = tl.where(((((r434_z / $(1024)) % $(128)) >= $(48)) & (((r434_z / $(1024)) % $(128)) < $(52)) & ((((r434_z / $(1024)) % $(128)) - $(48)) % $(1) == $(0))), tl.load($(candidate_vbmm_80_ptr) + ((((((((r434_z / $(128)) % $(8)) * $(512)) + ((((r434_z / $(1024)) % $(128)) - $(48)) * $(128)) + (r434_z % $(128))) / $(512)) % $(8)) * $(512)) + ((((((r434_z / $(128)) % $(8)) * $(4)) + (((r434_z / $(1024)) % $(128)) - $(48))) % $(4)) * $(128)) + (r434_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r434_value = r434_value.to(tl.bfloat16)
    r434_value = tl.broadcast_to(r434_value, [$(131072)])
    tl.store($(candidate_vslice_backward_42_ptr) + r434_z, r434_value)
    r435_z = tl.arange($(131072))
    r435_value = (tl.load($(candidate_vslice_backward_39_ptr) + ((((r435_z / $(1024)) % $(128)) * $(1024)) + (((r435_z / $(128)) % $(8)) * $(128)) + (r435_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_42_ptr) + ((((r435_z / $(1024)) % $(128)) * $(1024)) + (((r435_z / $(128)) % $(8)) * $(128)) + (r435_z % $(128))), dtype=tl.bfloat16))
    r435_value = r435_value.to(tl.bfloat16)
    r435_value = tl.broadcast_to(r435_value, [$(131072)])
    tl.store($(candidate_vadd_49_ptr) + r435_z, r435_value)
    r436_z = tl.arange($(6144))
    r436_value = tl.load($(candidate_vbmm_82_ptr) + ((((r436_z / $(768)) % $(8)) * $(768)) + ((r436_z % $(192)) * $(4)) + ((r436_z / $(192)) % $(4))), dtype=tl.float32)
    r436_value = r436_value.to(tl.bfloat16)
    r436_value = tl.broadcast_to(r436_value, [$(6144)])
    tl.store($(candidate_vto_copy_96_ptr) + r436_z, r436_value)
    r437_z = tl.arange($(196608))
    r437_value = tl.where(((((r437_z / $(1536)) % $(128)) >= $(48)) & (((r437_z / $(1536)) % $(128)) < $(52)) & ((((r437_z / $(1536)) % $(128)) - $(48)) % $(1) == $(0))), tl.load($(candidate_vto_copy_96_ptr) + ((((r437_z / $(192)) % $(8)) * $(768)) + (((((r437_z / $(1536)) % $(128)) - $(48)) % $(4)) * $(192)) + (r437_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r437_value = r437_value.to(tl.bfloat16)
    r437_value = tl.broadcast_to(r437_value, [$(196608)])
    tl.store($(candidate_vslice_backward_43_ptr) + r437_z, r437_value)
    r438_z = tl.arange($(196608))
    r438_value = (tl.load($(candidate_vslice_backward_40_ptr) + ((((r438_z / $(1536)) % $(128)) * $(1536)) + (((r438_z / $(192)) % $(8)) * $(192)) + (r438_z % $(192))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_43_ptr) + ((((r438_z / $(1536)) % $(128)) * $(1536)) + (((r438_z / $(192)) % $(8)) * $(192)) + (r438_z % $(192))), dtype=tl.bfloat16))
    r438_value = r438_value.to(tl.bfloat16)
    r438_value = tl.broadcast_to(r438_value, [$(196608)])
    tl.store($(candidate_vadd_50_ptr) + r438_z, r438_value)
    r439_z = tl.arange($(6144))
    r439_value = tl.load($(candidate_vbmm_83_ptr) + ((((r439_z / $(768)) % $(8)) * $(768)) + (((r439_z / $(192)) % $(4)) * $(192)) + (r439_z % $(192))), dtype=tl.float32)
    r439_value = r439_value.to(tl.bfloat16)
    r439_value = tl.broadcast_to(r439_value, [$(6144)])
    tl.store($(candidate_vto_copy_97_ptr) + r439_z, r439_value)
    r440_z = tl.arange($(49152))
    r440_value = tl.where(((((r440_z / $(1536)) % $(32)) >= $(12)) & (((r440_z / $(1536)) % $(32)) < $(16)) & ((((r440_z / $(1536)) % $(32)) - $(12)) % $(1) == $(0))), tl.load($(candidate_vto_copy_97_ptr) + ((((r440_z / $(192)) % $(8)) * $(768)) + (((((r440_z / $(1536)) % $(32)) - $(12)) % $(4)) * $(192)) + (r440_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r440_value = r440_value.to(tl.bfloat16)
    r440_value = tl.broadcast_to(r440_value, [$(49152)])
    tl.store($(candidate_vslice_backward_44_ptr) + r440_z, r440_value)
    r441_z = tl.arange($(49152))
    r441_value = (tl.load($(candidate_vslice_backward_41_ptr) + ((((r441_z / $(1536)) % $(32)) * $(1536)) + (((r441_z / $(192)) % $(8)) * $(192)) + (r441_z % $(192))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_44_ptr) + ((((r441_z / $(1536)) % $(32)) * $(1536)) + (((r441_z / $(192)) % $(8)) * $(192)) + (r441_z % $(192))), dtype=tl.bfloat16))
    r441_value = r441_value.to(tl.bfloat16)
    r441_value = tl.broadcast_to(r441_value, [$(49152)])
    tl.store($(candidate_vadd_51_ptr) + r441_z, r441_value)
    r442_row = tl.arange($(12))
    r442_col = tl.arange($(128))
    r442_lane = tl.arange($(12))
    for r442_group in range($(8)) {
      r442_left = tl.load($(candidate_vto_copy_31_ptr) + (((((((r442_group % $(8)) * $(144)) + (r442_lane[None, :] * $(12)) + r442_row[:, None]) / $(144)) % $(8)) * $(144)) + ((((((r442_group % $(8)) * $(144)) + (r442_lane[None, :] * $(12)) + r442_row[:, None]) / $(12)) % $(12)) * $(12)) + ((((r442_group % $(8)) * $(144)) + (r442_lane[None, :] * $(12)) + r442_row[:, None]) % $(12))), dtype=tl.bfloat16)
      r442_right = tl.load($(candidate_vmm_4_ptr) + (((((((r442_group % $(8)) * $(1536)) + (r442_lane[:, None] * $(128)) + r442_col[None, :]) / $(128)) % $(12)) * $(1024)) + ((((((r442_group % $(8)) * $(1536)) + (r442_lane[:, None] * $(128)) + r442_col[None, :]) / $(1536)) % $(8)) * $(128)) + ((((r442_group % $(8)) * $(1536)) + (r442_lane[:, None] * $(128)) + r442_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r442_value = tl.dot(r442_left, r442_right)
      r442_value = r442_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_84_ptr) + ((r442_group * $(1536)) + (r442_row[:, None] * $(128)) + r442_col[None, :]), r442_value)
    }
    r443_row = tl.arange($(12))
    r443_col = tl.arange($(12))
    r443_lane = tl.arange($(128))
    for r443_group in range($(8)) {
      r443_left = tl.load($(candidate_vmm_4_ptr) + (((((((r443_group % $(8)) * $(1536)) + (r443_row[:, None] * $(128)) + r443_lane[None, :]) / $(128)) % $(12)) * $(1024)) + ((((((r443_group % $(8)) * $(1536)) + (r443_row[:, None] * $(128)) + r443_lane[None, :]) / $(1536)) % $(8)) * $(128)) + ((((r443_group % $(8)) * $(1536)) + (r443_row[:, None] * $(128)) + r443_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r443_right = tl.load($(candidate_vbmm_3_ptr) + (((((((r443_group % $(8)) * $(1536)) + (r443_col[None, :] * $(128)) + r443_lane[:, None]) / $(128)) % $(12)) * $(1024)) + ((((((r443_group % $(8)) * $(1536)) + (r443_col[None, :] * $(128)) + r443_lane[:, None]) / $(1536)) % $(8)) * $(128)) + ((((r443_group % $(8)) * $(1536)) + (r443_col[None, :] * $(128)) + r443_lane[:, None]) % $(128))), dtype=tl.bfloat16)
      r443_value = tl.dot(r443_left, r443_right)
      r443_value = r443_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_85_ptr) + ((r443_group * $(144)) + (r443_row[:, None] * $(12)) + r443_col[None, :]), r443_value)
    }
    r444_z = tl.arange($(1152))
    r444_value = tl.load($(candidate_vbmm_85_ptr) + ((((r444_z / $(144)) % $(8)) * $(144)) + (((r444_z / $(12)) % $(12)) * $(12)) + (r444_z % $(12))), dtype=tl.bfloat16)
    r444_value = r444_value.to(tl.float32)
    r444_value = tl.broadcast_to(r444_value, [$(1152)])
    tl.store($(candidate_vto_copy_98_ptr) + r444_z, r444_value)
    r445_z = tl.arange($(1152))
    r445_lane = tl.arange($(12))
    r445_row_values = tl.load($(candidate_vsoftmax_ptr) + ((((r445_z[:, None] / $(144)) % $(8)) * $(144)) + (((r445_z[:, None] / $(12)) % $(12)) * $(12)) + (r445_lane[None, :] % $(12))), dtype=tl.float32)
    r445_row_grad = tl.load($(candidate_vto_copy_98_ptr) + ((((r445_z[:, None] / $(144)) % $(8)) * $(144)) + (((r445_z[:, None] / $(12)) % $(12)) * $(12)) + (r445_lane[None, :] % $(12))), dtype=tl.float32)
    r445_weighted = r445_row_grad * r445_row_values
    r445_dot_grad = tl.sum(r445_weighted, axis=1)
    r445_value = (tl.load($(candidate_vto_copy_98_ptr) + ((((r445_z / $(144)) % $(8)) * $(144)) + (((r445_z / $(12)) % $(12)) * $(12)) + (r445_z % $(12))), dtype=tl.float32) - r445_dot_grad) * tl.load($(candidate_vsoftmax_ptr) + ((((r445_z / $(144)) % $(8)) * $(144)) + (((r445_z / $(12)) % $(12)) * $(12)) + (r445_z % $(12))), dtype=tl.float32)
    r445_value = r445_value.to(tl.float32)
    r445_value = tl.broadcast_to(r445_value, [$(1152)])
    tl.store($(candidate_vsoftmax_backward_data_11_ptr) + r445_z, r445_value)
    r446_z = tl.arange($(1152))
    r446_value = tl.where(tl.load($(candidate_vgt_ptr) + ((((r446_z / $(12)) % $(12)) * $(12)) + (r446_z % $(12))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_11_ptr) + ((((r446_z / $(144)) % $(8)) * $(144)) + (((r446_z / $(12)) % $(12)) * $(12)) + (r446_z % $(12))), dtype=tl.float32))
    r446_value = r446_value.to(tl.float32)
    r446_value = tl.broadcast_to(r446_value, [$(1152)])
    tl.store($(candidate_vmasked_fill_23_ptr) + r446_z, r446_value)
    r447_z = tl.arange($(1152))
    r447_value = (tl.load($(candidate_vmasked_fill_23_ptr) + ((((r447_z / $(144)) % $(8)) * $(144)) + (((r447_z / $(12)) % $(12)) * $(12)) + (r447_z % $(12))), dtype=tl.float32) * 0.07216878364870322)
    r447_value = r447_value.to(tl.float32)
    r447_value = tl.broadcast_to(r447_value, [$(1152)])
    tl.store($(candidate_vmul_46_ptr) + r447_z, r447_value)
    r448_row = tl.arange($(192))
    r448_col = tl.arange($(12))
    r448_lane = tl.arange($(12))
    for r448_group in range($(8)) {
      r448_left = tl.load($(candidate_vto_copy_29_ptr) + (((((((r448_group % $(8)) * $(2304)) + (r448_lane[None, :] * $(192)) + r448_row[:, None]) / $(2304)) % $(8)) * $(2304)) + ((((((r448_group % $(8)) * $(2304)) + (r448_lane[None, :] * $(192)) + r448_row[:, None]) / $(192)) % $(12)) * $(192)) + ((((r448_group % $(8)) * $(2304)) + (r448_lane[None, :] * $(192)) + r448_row[:, None]) % $(192))), dtype=tl.float32)
      r448_right = tl.load($(candidate_vmul_46_ptr) + (((((((r448_group % $(8)) * $(144)) + (r448_lane[:, None] * $(12)) + r448_col[None, :]) / $(144)) % $(8)) * $(144)) + ((((((r448_group % $(8)) * $(144)) + (r448_lane[:, None] * $(12)) + r448_col[None, :]) / $(12)) % $(12)) * $(12)) + ((((r448_group % $(8)) * $(144)) + (r448_lane[:, None] * $(12)) + r448_col[None, :]) % $(12))), dtype=tl.float32)
      r448_value = tl.dot(r448_left, r448_right)
      r448_value = r448_value.to(tl.float32)
      tl.store($(candidate_vbmm_86_ptr) + ((r448_group * $(2304)) + (r448_row[:, None] * $(12)) + r448_col[None, :]), r448_value)
    }
    r449_row = tl.arange($(12))
    r449_col = tl.arange($(192))
    r449_lane = tl.arange($(12))
    for r449_group in range($(8)) {
      r449_left = tl.load($(candidate_vmul_46_ptr) + (((((((r449_group % $(8)) * $(144)) + (r449_row[:, None] * $(12)) + r449_lane[None, :]) / $(144)) % $(8)) * $(144)) + ((((((r449_group % $(8)) * $(144)) + (r449_row[:, None] * $(12)) + r449_lane[None, :]) / $(12)) % $(12)) * $(12)) + ((((r449_group % $(8)) * $(144)) + (r449_row[:, None] * $(12)) + r449_lane[None, :]) % $(12))), dtype=tl.float32)
      r449_right = tl.load($(candidate_vto_copy_30_ptr) + (((((((r449_group % $(8)) * $(2304)) + (r449_col[None, :] * $(12)) + r449_lane[:, None]) / $(2304)) % $(8)) * $(2304)) + (((((r449_group % $(8)) * $(2304)) + (r449_col[None, :] * $(12)) + r449_lane[:, None]) % $(12)) * $(192)) + (((((r449_group % $(8)) * $(2304)) + (r449_col[None, :] * $(12)) + r449_lane[:, None]) / $(12)) % $(192))), dtype=tl.float32)
      r449_value = tl.dot(r449_left, r449_right)
      r449_value = r449_value.to(tl.float32)
      tl.store($(candidate_vbmm_87_ptr) + ((r449_group * $(2304)) + (r449_row[:, None] * $(192)) + r449_col[None, :]), r449_value)
    }
    r450_z = tl.arange($(131072))
    r450_value = tl.where(((((r450_z / $(1024)) % $(128)) >= $(0)) & (((r450_z / $(1024)) % $(128)) < $(12)) & ((((r450_z / $(1024)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vbmm_84_ptr) + ((((((((r450_z / $(128)) % $(8)) * $(1536)) + ((((r450_z / $(1024)) % $(128)) - $(0)) * $(128)) + (r450_z % $(128))) / $(1536)) % $(8)) * $(1536)) + ((((((r450_z / $(128)) % $(8)) * $(12)) + (((r450_z / $(1024)) % $(128)) - $(0))) % $(12)) * $(128)) + (r450_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r450_value = r450_value.to(tl.bfloat16)
    r450_value = tl.broadcast_to(r450_value, [$(131072)])
    tl.store($(candidate_vslice_backward_45_ptr) + r450_z, r450_value)
    r451_z = tl.arange($(131072))
    r451_value = (tl.load($(candidate_vadd_49_ptr) + ((((r451_z / $(1024)) % $(128)) * $(1024)) + (((r451_z / $(128)) % $(8)) * $(128)) + (r451_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_45_ptr) + ((((r451_z / $(1024)) % $(128)) * $(1024)) + (((r451_z / $(128)) % $(8)) * $(128)) + (r451_z % $(128))), dtype=tl.bfloat16))
    r451_value = r451_value.to(tl.bfloat16)
    r451_value = tl.broadcast_to(r451_value, [$(131072)])
    tl.store($(candidate_vadd_52_ptr) + r451_z, r451_value)
    r452_z = tl.arange($(18432))
    r452_value = tl.load($(candidate_vbmm_86_ptr) + ((((r452_z / $(2304)) % $(8)) * $(2304)) + ((r452_z % $(192)) * $(12)) + ((r452_z / $(192)) % $(12))), dtype=tl.float32)
    r452_value = r452_value.to(tl.bfloat16)
    r452_value = tl.broadcast_to(r452_value, [$(18432)])
    tl.store($(candidate_vto_copy_99_ptr) + r452_z, r452_value)
    r453_z = tl.arange($(196608))
    r453_value = tl.where(((((r453_z / $(1536)) % $(128)) >= $(0)) & (((r453_z / $(1536)) % $(128)) < $(12)) & ((((r453_z / $(1536)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vto_copy_99_ptr) + ((((r453_z / $(192)) % $(8)) * $(2304)) + (((((r453_z / $(1536)) % $(128)) - $(0)) % $(12)) * $(192)) + (r453_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r453_value = r453_value.to(tl.bfloat16)
    r453_value = tl.broadcast_to(r453_value, [$(196608)])
    tl.store($(candidate_vslice_backward_46_ptr) + r453_z, r453_value)
    r454_z = tl.arange($(196608))
    r454_value = (tl.load($(candidate_vadd_50_ptr) + ((((r454_z / $(1536)) % $(128)) * $(1536)) + (((r454_z / $(192)) % $(8)) * $(192)) + (r454_z % $(192))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_46_ptr) + ((((r454_z / $(1536)) % $(128)) * $(1536)) + (((r454_z / $(192)) % $(8)) * $(192)) + (r454_z % $(192))), dtype=tl.bfloat16))
    r454_value = r454_value.to(tl.bfloat16)
    r454_value = tl.broadcast_to(r454_value, [$(196608)])
    tl.store($(candidate_vadd_53_ptr) + r454_z, r454_value)
    r455_z = tl.arange($(18432))
    r455_value = tl.load($(candidate_vbmm_87_ptr) + ((((r455_z / $(2304)) % $(8)) * $(2304)) + (((r455_z / $(192)) % $(12)) * $(192)) + (r455_z % $(192))), dtype=tl.float32)
    r455_value = r455_value.to(tl.bfloat16)
    r455_value = tl.broadcast_to(r455_value, [$(18432)])
    tl.store($(candidate_vto_copy_100_ptr) + r455_z, r455_value)
    r456_z = tl.arange($(49152))
    r456_value = tl.where(((((r456_z / $(1536)) % $(32)) >= $(0)) & (((r456_z / $(1536)) % $(32)) < $(12)) & ((((r456_z / $(1536)) % $(32)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vto_copy_100_ptr) + ((((r456_z / $(192)) % $(8)) * $(2304)) + (((((r456_z / $(1536)) % $(32)) - $(0)) % $(12)) * $(192)) + (r456_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r456_value = r456_value.to(tl.bfloat16)
    r456_value = tl.broadcast_to(r456_value, [$(49152)])
    tl.store($(candidate_vslice_backward_47_ptr) + r456_z, r456_value)
    r457_z = tl.arange($(49152))
    r457_value = (tl.load($(candidate_vadd_51_ptr) + ((((r457_z / $(1536)) % $(32)) * $(1536)) + (((r457_z / $(192)) % $(8)) * $(192)) + (r457_z % $(192))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_47_ptr) + ((((r457_z / $(1536)) % $(32)) * $(1536)) + (((r457_z / $(192)) % $(8)) * $(192)) + (r457_z % $(192))), dtype=tl.bfloat16))
    r457_value = r457_value.to(tl.bfloat16)
    r457_value = tl.broadcast_to(r457_value, [$(49152)])
    tl.store($(candidate_vadd_54_ptr) + r457_z, r457_value)
    r458_z = tl.arange($(36864))
    r458_value = tl.cast(0.0, tl.bfloat16)
    r458_value = r458_value.to(tl.bfloat16)
    r458_value = tl.broadcast_to(r458_value, [$(36864)])
    tl.store($(candidate_vfill_18_ptr) + r458_z, r458_value)
    r459_z = tl.arange($(131072))
    r459_value = tl.where(((((r459_z / $(1024)) % $(128)) >= $(12)) & (((r459_z / $(1024)) % $(128)) < $(48)) & ((((r459_z / $(1024)) % $(128)) - $(12)) % $(1) == $(0))), tl.load($(candidate_vfill_18_ptr) + ((((((r459_z / $(1024)) % $(128)) - $(12)) % $(36)) * $(1024)) + (((r459_z / $(128)) % $(8)) * $(128)) + (r459_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vadd_52_ptr) + ((((r459_z / $(1024)) % $(128)) * $(1024)) + (((r459_z / $(128)) % $(8)) * $(128)) + (r459_z % $(128))), dtype=tl.bfloat16))
    r459_value = r459_value.to(tl.bfloat16)
    r459_value = tl.broadcast_to(r459_value, [$(131072)])
    tl.store($(candidate_vslice_scatter_18_ptr) + r459_z, r459_value)
    r460_z = tl.arange($(12288))
    r460_value = tl.cast(0.0, tl.bfloat16)
    r460_value = r460_value.to(tl.bfloat16)
    r460_value = tl.broadcast_to(r460_value, [$(12288)])
    tl.store($(candidate_vfill_19_ptr) + r460_z, r460_value)
    r461_z = tl.arange($(131072))
    r461_value = tl.where(((((r461_z / $(1024)) % $(128)) >= $(52)) & (((r461_z / $(1024)) % $(128)) < $(64)) & ((((r461_z / $(1024)) % $(128)) - $(52)) % $(1) == $(0))), tl.load($(candidate_vfill_19_ptr) + ((((((r461_z / $(1024)) % $(128)) - $(52)) % $(12)) * $(1024)) + (((r461_z / $(128)) % $(8)) * $(128)) + (r461_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vslice_scatter_18_ptr) + ((((r461_z / $(1024)) % $(128)) * $(1024)) + (((r461_z / $(128)) % $(8)) * $(128)) + (r461_z % $(128))), dtype=tl.bfloat16))
    r461_value = r461_value.to(tl.bfloat16)
    r461_value = tl.broadcast_to(r461_value, [$(131072)])
    tl.store($(candidate_vslice_scatter_19_ptr) + r461_z, r461_value)
    r462_z = tl.arange($(49152))
    r462_value = tl.cast(0.0, tl.bfloat16)
    r462_value = r462_value.to(tl.bfloat16)
    r462_value = tl.broadcast_to(r462_value, [$(49152)])
    tl.store($(candidate_vfill_20_ptr) + r462_z, r462_value)
    r463_z = tl.arange($(131072))
    r463_value = tl.where(((((r463_z / $(1024)) % $(128)) >= $(80)) & (((r463_z / $(1024)) % $(128)) < $(128)) & ((((r463_z / $(1024)) % $(128)) - $(80)) % $(1) == $(0))), tl.load($(candidate_vfill_20_ptr) + ((((((r463_z / $(1024)) % $(128)) - $(80)) % $(48)) * $(1024)) + (((r463_z / $(128)) % $(8)) * $(128)) + (r463_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vslice_scatter_19_ptr) + ((((r463_z / $(1024)) % $(128)) * $(1024)) + (((r463_z / $(128)) % $(8)) * $(128)) + (r463_z % $(128))), dtype=tl.bfloat16))
    r463_value = r463_value.to(tl.bfloat16)
    r463_value = tl.broadcast_to(r463_value, [$(131072)])
    tl.store($(candidate_vslice_scatter_20_ptr) + r463_z, r463_value)
    r464_z = tl.arange($(55296))
    r464_value = tl.cast(0.0, tl.bfloat16)
    r464_value = r464_value.to(tl.bfloat16)
    r464_value = tl.broadcast_to(r464_value, [$(55296)])
    tl.store($(candidate_vfill_21_ptr) + r464_z, r464_value)
    r465_z = tl.arange($(196608))
    r465_value = tl.where(((((r465_z / $(1536)) % $(128)) >= $(12)) & (((r465_z / $(1536)) % $(128)) < $(48)) & ((((r465_z / $(1536)) % $(128)) - $(12)) % $(1) == $(0))), tl.load($(candidate_vfill_21_ptr) + ((((((r465_z / $(1536)) % $(128)) - $(12)) % $(36)) * $(1536)) + (((r465_z / $(192)) % $(8)) * $(192)) + (r465_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vadd_53_ptr) + ((((r465_z / $(1536)) % $(128)) * $(1536)) + (((r465_z / $(192)) % $(8)) * $(192)) + (r465_z % $(192))), dtype=tl.bfloat16))
    r465_value = r465_value.to(tl.bfloat16)
    r465_value = tl.broadcast_to(r465_value, [$(196608)])
    tl.store($(candidate_vslice_scatter_21_ptr) + r465_z, r465_value)
    r466_z = tl.arange($(18432))
    r466_value = tl.cast(0.0, tl.bfloat16)
    r466_value = r466_value.to(tl.bfloat16)
    r466_value = tl.broadcast_to(r466_value, [$(18432)])
    tl.store($(candidate_vfill_22_ptr) + r466_z, r466_value)
    r467_z = tl.arange($(196608))
    r467_value = tl.where(((((r467_z / $(1536)) % $(128)) >= $(52)) & (((r467_z / $(1536)) % $(128)) < $(64)) & ((((r467_z / $(1536)) % $(128)) - $(52)) % $(1) == $(0))), tl.load($(candidate_vfill_22_ptr) + ((((((r467_z / $(1536)) % $(128)) - $(52)) % $(12)) * $(1536)) + (((r467_z / $(192)) % $(8)) * $(192)) + (r467_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vslice_scatter_21_ptr) + ((((r467_z / $(1536)) % $(128)) * $(1536)) + (((r467_z / $(192)) % $(8)) * $(192)) + (r467_z % $(192))), dtype=tl.bfloat16))
    r467_value = r467_value.to(tl.bfloat16)
    r467_value = tl.broadcast_to(r467_value, [$(196608)])
    tl.store($(candidate_vslice_scatter_22_ptr) + r467_z, r467_value)
    r468_z = tl.arange($(73728))
    r468_value = tl.cast(0.0, tl.bfloat16)
    r468_value = r468_value.to(tl.bfloat16)
    r468_value = tl.broadcast_to(r468_value, [$(73728)])
    tl.store($(candidate_vfill_23_ptr) + r468_z, r468_value)
    r469_z = tl.arange($(196608))
    r469_value = tl.where(((((r469_z / $(1536)) % $(128)) >= $(80)) & (((r469_z / $(1536)) % $(128)) < $(128)) & ((((r469_z / $(1536)) % $(128)) - $(80)) % $(1) == $(0))), tl.load($(candidate_vfill_23_ptr) + ((((((r469_z / $(1536)) % $(128)) - $(80)) % $(48)) * $(1536)) + (((r469_z / $(192)) % $(8)) * $(192)) + (r469_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vslice_scatter_22_ptr) + ((((r469_z / $(1536)) % $(128)) * $(1536)) + (((r469_z / $(192)) % $(8)) * $(192)) + (r469_z % $(192))), dtype=tl.bfloat16))
    r469_value = r469_value.to(tl.bfloat16)
    r469_value = tl.broadcast_to(r469_value, [$(196608)])
    tl.store($(candidate_vslice_scatter_23_ptr) + r469_z, r469_value)
    r470_z = tl.arange($(8192))
    r470_lane = tl.arange($(8))
    r470_value = tl.sum(tl.load($(candidate_vslice_scatter_23_ptr) + ((((r470_z[:, None] / $(64)) % $(128)) * $(1536)) + (r470_lane[None, :] * $(192)) + (r470_z[:, None] % $(64)) + $(128)), dtype=tl.bfloat16), axis=1)
    r470_value = r470_value.to(tl.bfloat16)
    r470_value = tl.broadcast_to(r470_value, [$(8192)])
    tl.store($(candidate_vsum_4_ptr) + r470_z, r470_value)
    r471_row = tl.arange($(128))
    r471_col = tl.arange($(256))
    r471_lane = tl.arange($(1024))
    for r471_group in range($(1)) {
      r471_left = tl.load($(candidate_vslice_scatter_20_ptr) + ((((((r471_row[:, None] * $(1024)) + r471_lane[None, :]) / $(1024)) % $(128)) * $(1024)) + (((((r471_row[:, None] * $(1024)) + r471_lane[None, :]) / $(128)) % $(8)) * $(128)) + (((r471_row[:, None] * $(1024)) + r471_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r471_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r471_col[None, :] * $(1024)) + r471_lane[:, None]) / $(128)) % $(8)) * $(65536)) + ((((r471_col[None, :] * $(1024)) + r471_lane[:, None]) % $(128)) * $(256)) + ((((r471_col[None, :] * $(1024)) + r471_lane[:, None]) / $(1024)) % $(256)) + $(32768)), dtype=tl.bfloat16)
      r471_value = tl.dot(r471_left, r471_right)
      r471_value = r471_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_88_ptr) + ((r471_group * $(32768)) + (r471_row[:, None] * $(256)) + r471_col[None, :]), r471_value)
    }
    r472_row = tl.arange($(128))
    r472_col = tl.arange($(256))
    r472_lane = tl.arange($(1024))
    for r472_group in range($(1)) {
      r472_left = tl.load($(candidate_vslice_scatter_23_ptr) + ((((((r472_row[:, None] * $(1024)) + r472_lane[None, :]) / $(1024)) % $(128)) * $(1536)) + (((((r472_row[:, None] * $(1024)) + r472_lane[None, :]) / $(128)) % $(8)) * $(192)) + (((r472_row[:, None] * $(1024)) + r472_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r472_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r472_col[None, :] * $(1024)) + r472_lane[:, None]) / $(128)) % $(8)) * $(65536)) + ((((r472_col[None, :] * $(1024)) + r472_lane[:, None]) % $(128)) * $(256)) + ((((r472_col[None, :] * $(1024)) + r472_lane[:, None]) / $(1024)) % $(256))), dtype=tl.bfloat16)
      r472_value = tl.dot(r472_left, r472_right)
      r472_value = r472_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_89_ptr) + ((r472_group * $(32768)) + (r472_row[:, None] * $(256)) + r472_col[None, :]), r472_value)
    }
    r473_z = tl.arange($(32768))
    r473_value = (tl.load($(candidate_vbmm_88_ptr) + ((((r473_z / $(256)) % $(128)) * $(256)) + (r473_z % $(256))), dtype=tl.bfloat16) + tl.load($(candidate_vbmm_89_ptr) + ((((r473_z / $(256)) % $(128)) * $(256)) + (r473_z % $(256))), dtype=tl.bfloat16))
    r473_value = r473_value.to(tl.bfloat16)
    r473_value = tl.broadcast_to(r473_value, [$(32768)])
    tl.store($(candidate_vadd_55_ptr) + r473_z, r473_value)
    r474_z = tl.arange($(32768))
    r474_value = tl.where(($(0) == $(0)), tl.load($(candidate_vadd_55_ptr) + ((((r474_z / $(256)) % $(128)) * $(256)) + (r474_z % $(256))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r474_value = r474_value.to(tl.bfloat16)
    r474_value = tl.broadcast_to(r474_value, [$(32768)])
    tl.store($(candidate_vselect_backward_3_ptr) + r474_z, r474_value)
    r475_z = tl.arange($(32768))
    r475_value = tl.where(((((r475_z / $(256)) % $(128)) >= $(0)) & (((r475_z / $(256)) % $(128)) < $(128)) & ((((r475_z / $(256)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vselect_backward_3_ptr) + ((((((r475_z / $(256)) % $(128)) - $(0)) % $(128)) * $(256)) + (r475_z % $(256))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r475_value = r475_value.to(tl.bfloat16)
    r475_value = tl.broadcast_to(r475_value, [$(32768)])
    tl.store($(candidate_vslice_backward_48_ptr) + r475_z, r475_value)
    r476_z = tl.arange($(32768))
    r476_value = (tl.load($(candidate_vadd_45_ptr) + ((((r476_z / $(256)) % $(128)) * $(256)) + (r476_z % $(256))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_48_ptr) + ((((r476_z / $(256)) % $(128)) * $(256)) + (r476_z % $(256))), dtype=tl.bfloat16))
    r476_value = r476_value.to(tl.bfloat16)
    r476_value = tl.broadcast_to(r476_value, [$(32768)])
    tl.store($(candidate_vadd_56_ptr) + r476_z, r476_value)
    r477_z = tl.arange($(8192))
    r477_value = tl.where(((((r477_z / $(64)) % $(128)) >= $(0)) & (((r477_z / $(64)) % $(128)) < $(128)) & ((((r477_z / $(64)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vsum_4_ptr) + ((((((r477_z / $(64)) % $(128)) - $(0)) % $(128)) * $(64)) + (r477_z % $(64))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r477_value = r477_value.to(tl.bfloat16)
    r477_value = tl.broadcast_to(r477_value, [$(8192)])
    tl.store($(candidate_vslice_backward_49_ptr) + r477_z, r477_value)
    r478_z = tl.arange($(8192))
    r478_value = (tl.load($(candidate_vadd_46_ptr) + ((((r478_z / $(64)) % $(128)) * $(64)) + (r478_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_49_ptr) + ((((r478_z / $(64)) % $(128)) * $(64)) + (r478_z % $(64))), dtype=tl.bfloat16))
    r478_value = r478_value.to(tl.bfloat16)
    r478_value = tl.broadcast_to(r478_value, [$(8192)])
    tl.store($(candidate_vadd_57_ptr) + r478_z, r478_value)
    r479_z = tl.arange($(131072))
    r479_value = tl.where(((((r479_z / $(1024)) % $(128)) >= $(0)) & (((r479_z / $(1024)) % $(128)) < $(32)) & ((((r479_z / $(1024)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vadd_54_ptr) + ((((((r479_z / $(1024)) % $(128)) - $(0)) % $(32)) * $(1536)) + (((r479_z / $(128)) % $(8)) * $(192)) + (r479_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r479_value = r479_value.to(tl.bfloat16)
    r479_value = tl.broadcast_to(r479_value, [$(131072)])
    tl.store($(candidate_vslice_backward_50_ptr) + r479_z, r479_value)
    r480_z = tl.arange($(131072))
    r480_value = (tl.load($(candidate_vadd_47_ptr) + ((((r480_z / $(1024)) % $(128)) * $(1024)) + (((r480_z / $(128)) % $(8)) * $(128)) + (r480_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_50_ptr) + ((((r480_z / $(1024)) % $(128)) * $(1024)) + (((r480_z / $(128)) % $(8)) * $(128)) + (r480_z % $(128))), dtype=tl.bfloat16))
    r480_value = r480_value.to(tl.bfloat16)
    r480_value = tl.broadcast_to(r480_value, [$(131072)])
    tl.store($(candidate_vadd_58_ptr) + r480_z, r480_value)
    r481_z = tl.arange($(65536))
    r481_value = tl.where(((((r481_z / $(512)) % $(128)) >= $(0)) & (((r481_z / $(512)) % $(128)) < $(32)) & ((((r481_z / $(512)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vadd_54_ptr) + ((((((r481_z / $(512)) % $(128)) - $(0)) % $(32)) * $(1536)) + (((r481_z / $(64)) % $(8)) * $(192)) + (r481_z % $(64)) + $(128)), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r481_value = r481_value.to(tl.bfloat16)
    r481_value = tl.broadcast_to(r481_value, [$(65536)])
    tl.store($(candidate_vslice_backward_51_ptr) + r481_z, r481_value)
    r482_z = tl.arange($(65536))
    r482_value = (tl.load($(candidate_vadd_48_ptr) + ((((r482_z / $(512)) % $(128)) * $(512)) + (((r482_z / $(64)) % $(8)) * $(64)) + (r482_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_51_ptr) + ((((r482_z / $(512)) % $(128)) * $(512)) + (((r482_z / $(64)) % $(8)) * $(64)) + (r482_z % $(64))), dtype=tl.bfloat16))
    r482_value = r482_value.to(tl.bfloat16)
    r482_value = tl.broadcast_to(r482_value, [$(65536)])
    tl.store($(candidate_vadd_59_ptr) + r482_z, r482_value)
    r483_z = tl.arange($(40960))
    r483_value = tl.where((r483_z % $(320)) < $(256), tl.load($(candidate_vadd_56_ptr) + ((((r483_z / $(320)) % $(128)) * $(256)) + (((r483_z % $(320)) - $(0)) % $(256))), dtype=tl.bfloat16), tl.load($(candidate_vadd_57_ptr) + ((((r483_z / $(320)) % $(128)) * $(64)) + (((r483_z % $(320)) - $(256)) % $(64))), dtype=tl.bfloat16))
    r483_value = r483_value.to(tl.bfloat16)
    r483_value = tl.broadcast_to(r483_value, [$(40960)])
    tl.store($(candidate_vcat_19_ptr) + r483_z, r483_value)
    r484_z = tl.arange($(40960))
    r484_value = tl.cast(0.0, tl.bfloat16)
    for r484_lane in range($(128)) {
      r484_value = tl.where(tl.load($(candidate_vsort_index_ptr) + (r484_lane % $(128)), dtype=tl.int64) == ((r484_z / $(320)) % $(128)), r484_value + tl.load($(candidate_vcat_19_ptr) + (((r484_lane % $(128)) * $(320)) + (r484_z % $(320))), dtype=tl.bfloat16), r484_value)
    }
    r484_value = r484_value.to(tl.bfloat16)
    r484_value = tl.broadcast_to(r484_value, [$(40960)])
    tl.store($(candidate_vindex_put_1_ptr) + r484_z, r484_value)
    r485_z = tl.arange($(8192))
    r485_value = (tl.load($(candidate_vindex_put_1_ptr) + ((((r485_z / $(64)) % $(128)) * $(320)) + (r485_z % $(64)) + $(256)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_28_ptr) + ((((r485_z / $(64)) % $(128)) * $(64)) + (r485_z % $(64))), dtype=tl.bfloat16))
    r485_value = r485_value.to(tl.bfloat16)
    r485_value = tl.broadcast_to(r485_value, [$(8192)])
    tl.store($(candidate_vmul_47_ptr) + r485_z, r485_value)
    r486_z = tl.arange($(8192))
    r486_value = (tl.load($(candidate_vindex_put_1_ptr) + ((((r486_z / $(64)) % $(128)) * $(320)) + (r486_z % $(64)) + $(256)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_27_ptr) + ((((r486_z / $(64)) % $(128)) * $(64)) + (r486_z % $(64))), dtype=tl.bfloat16))
    r486_value = r486_value.to(tl.bfloat16)
    r486_value = tl.broadcast_to(r486_value, [$(8192)])
    tl.store($(candidate_vmul_48_ptr) + r486_z, r486_value)
    r487_z = tl.arange($(8192))
    r487_value = tl.where(((r487_z % $(2)) == $(0)), tl.load($(candidate_vmul_47_ptr) + ((((r487_z / $(64)) % $(128)) * $(64)) + (((r487_z / $(2)) % $(32)) * $(2)) + $(1)), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r487_value = r487_value.to(tl.bfloat16)
    r487_value = tl.broadcast_to(r487_value, [$(8192)])
    tl.store($(candidate_vselect_backward_4_ptr) + r487_z, r487_value)
    r488_z = tl.arange($(4096))
    r488_value = -tl.load($(candidate_vmul_47_ptr) + ((((r488_z / $(32)) % $(128)) * $(64)) + ((r488_z % $(32)) * $(2))), dtype=tl.bfloat16)
    r488_value = r488_value.to(tl.bfloat16)
    r488_value = tl.broadcast_to(r488_value, [$(4096)])
    tl.store($(candidate_vneg_2_ptr) + r488_z, r488_value)
    r489_z = tl.arange($(8192))
    r489_value = tl.where(((r489_z % $(2)) == $(1)), tl.load($(candidate_vneg_2_ptr) + ((((r489_z / $(64)) % $(128)) * $(32)) + ((r489_z / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r489_value = r489_value.to(tl.bfloat16)
    r489_value = tl.broadcast_to(r489_value, [$(8192)])
    tl.store($(candidate_vselect_backward_5_ptr) + r489_z, r489_value)
    r490_z = tl.arange($(8192))
    r490_value = (tl.load($(candidate_vselect_backward_4_ptr) + ((((r490_z / $(64)) % $(128)) * $(64)) + (((r490_z / $(2)) % $(32)) * $(2)) + (r490_z % $(2))), dtype=tl.bfloat16) + tl.load($(candidate_vselect_backward_5_ptr) + ((((r490_z / $(64)) % $(128)) * $(64)) + (((r490_z / $(2)) % $(32)) * $(2)) + (r490_z % $(2))), dtype=tl.bfloat16))
    r490_value = r490_value.to(tl.bfloat16)
    r490_value = tl.broadcast_to(r490_value, [$(8192)])
    tl.store($(candidate_vadd_60_ptr) + r490_z, r490_value)
    r491_z = tl.arange($(8192))
    r491_value = (tl.load($(candidate_vmul_48_ptr) + ((((r491_z / $(64)) % $(128)) * $(64)) + (r491_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vadd_60_ptr) + ((((r491_z / $(64)) % $(128)) * $(64)) + (((((((r491_z / $(64)) % $(128)) * $(64)) + (r491_z % $(64))) / $(2)) % $(32)) * $(2)) + (((((r491_z / $(64)) % $(128)) * $(64)) + (r491_z % $(64))) % $(2))), dtype=tl.bfloat16))
    r491_value = r491_value.to(tl.bfloat16)
    r491_value = tl.broadcast_to(r491_value, [$(8192)])
    tl.store($(candidate_vadd_61_ptr) + r491_z, r491_value)
    r492_z = tl.arange($(65536))
    r492_value = (tl.load($(candidate_vadd_59_ptr) + ((((r492_z / $(512)) % $(128)) * $(512)) + (((r492_z / $(64)) % $(8)) * $(64)) + (r492_z % $(64))), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_26_ptr) + ((((r492_z / $(512)) % $(128)) * $(64)) + (r492_z % $(64))), dtype=tl.bfloat16))
    r492_value = r492_value.to(tl.bfloat16)
    r492_value = tl.broadcast_to(r492_value, [$(65536)])
    tl.store($(candidate_vmul_49_ptr) + r492_z, r492_value)
    r493_z = tl.arange($(65536))
    r493_value = (tl.load($(candidate_vadd_59_ptr) + ((((r493_z / $(512)) % $(128)) * $(512)) + (((r493_z / $(64)) % $(8)) * $(64)) + (r493_z % $(64))), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_25_ptr) + ((((r493_z / $(512)) % $(128)) * $(64)) + (r493_z % $(64))), dtype=tl.bfloat16))
    r493_value = r493_value.to(tl.bfloat16)
    r493_value = tl.broadcast_to(r493_value, [$(65536)])
    tl.store($(candidate_vmul_50_ptr) + r493_z, r493_value)
    r494_z = tl.arange($(65536))
    r494_value = tl.where(((r494_z % $(2)) == $(0)), tl.load($(candidate_vmul_49_ptr) + ((((r494_z / $(512)) % $(128)) * $(512)) + (((r494_z / $(64)) % $(8)) * $(64)) + (((r494_z / $(2)) % $(32)) * $(2)) + $(1)), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r494_value = r494_value.to(tl.bfloat16)
    r494_value = tl.broadcast_to(r494_value, [$(65536)])
    tl.store($(candidate_vselect_backward_6_ptr) + r494_z, r494_value)
    r495_z = tl.arange($(32768))
    r495_value = -tl.load($(candidate_vmul_49_ptr) + ((((r495_z / $(256)) % $(128)) * $(512)) + (((r495_z / $(32)) % $(8)) * $(64)) + ((r495_z % $(32)) * $(2))), dtype=tl.bfloat16)
    r495_value = r495_value.to(tl.bfloat16)
    r495_value = tl.broadcast_to(r495_value, [$(32768)])
    tl.store($(candidate_vneg_3_ptr) + r495_z, r495_value)
    r496_z = tl.arange($(65536))
    r496_value = tl.where(((r496_z % $(2)) == $(1)), tl.load($(candidate_vneg_3_ptr) + ((((r496_z / $(512)) % $(128)) * $(256)) + (((r496_z / $(64)) % $(8)) * $(32)) + ((r496_z / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r496_value = r496_value.to(tl.bfloat16)
    r496_value = tl.broadcast_to(r496_value, [$(65536)])
    tl.store($(candidate_vselect_backward_7_ptr) + r496_z, r496_value)
    r497_z = tl.arange($(65536))
    r497_value = (tl.load($(candidate_vselect_backward_6_ptr) + ((((r497_z / $(512)) % $(128)) * $(512)) + (((r497_z / $(64)) % $(8)) * $(64)) + (((r497_z / $(2)) % $(32)) * $(2)) + (r497_z % $(2))), dtype=tl.bfloat16) + tl.load($(candidate_vselect_backward_7_ptr) + ((((r497_z / $(512)) % $(128)) * $(512)) + (((r497_z / $(64)) % $(8)) * $(64)) + (((r497_z / $(2)) % $(32)) * $(2)) + (r497_z % $(2))), dtype=tl.bfloat16))
    r497_value = r497_value.to(tl.bfloat16)
    r497_value = tl.broadcast_to(r497_value, [$(65536)])
    tl.store($(candidate_vadd_62_ptr) + r497_z, r497_value)
    r498_z = tl.arange($(65536))
    r498_value = (tl.load($(candidate_vmul_50_ptr) + ((((r498_z / $(512)) % $(128)) * $(512)) + (((r498_z / $(64)) % $(8)) * $(64)) + (r498_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vadd_62_ptr) + ((((r498_z / $(512)) % $(128)) * $(512)) + (((r498_z / $(64)) % $(8)) * $(64)) + (((((((r498_z / $(512)) % $(128)) * $(512)) + (((r498_z / $(64)) % $(8)) * $(64)) + (r498_z % $(64))) / $(2)) % $(32)) * $(2)) + (((((r498_z / $(512)) % $(128)) * $(512)) + (((r498_z / $(64)) % $(8)) * $(64)) + (r498_z % $(64))) % $(2))), dtype=tl.bfloat16))
    r498_value = r498_value.to(tl.bfloat16)
    r498_value = tl.broadcast_to(r498_value, [$(65536)])
    tl.store($(candidate_vadd_63_ptr) + r498_z, r498_value)
    r499_z = tl.arange($(196608))
    r499_value = tl.where((r499_z % $(192)) < $(128), tl.load($(candidate_vadd_58_ptr) + ((((((((r499_z / $(192)) % $(128)) * $(1024)) + (((r499_z / $(24576)) % $(8)) * $(128)) + ((r499_z % $(192)) - $(0))) / $(1024)) % $(128)) * $(1024)) + (((((((r499_z / $(192)) % $(128)) * $(1024)) + (((r499_z / $(24576)) % $(8)) * $(128)) + ((r499_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r499_z / $(192)) % $(128)) * $(1024)) + (((r499_z / $(24576)) % $(8)) * $(128)) + ((r499_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vadd_63_ptr) + ((((r499_z / $(192)) % $(128)) * $(512)) + (((r499_z / $(24576)) % $(8)) * $(64)) + (((r499_z % $(192)) - $(128)) % $(64))), dtype=tl.bfloat16))
    r499_value = r499_value.to(tl.bfloat16)
    r499_value = tl.broadcast_to(r499_value, [$(196608)])
    tl.store($(candidate_vcat_20_ptr) + r499_z, r499_value)
    r500_z = tl.arange($(32768))
    r500_value = tl.load($(candidate_vindex_put_1_ptr) + ((((r500_z / $(256)) % $(128)) * $(320)) + (r500_z % $(256))), dtype=tl.bfloat16)
    r500_value = r500_value.to(tl.float32)
    r500_value = tl.broadcast_to(r500_value, [$(32768)])
    tl.store($(candidate_vto_copy_101_ptr) + r500_z, r500_value)
    r501_z = tl.arange($(32768))
    r501_value = (tl.load($(candidate_vto_copy_101_ptr) + ((((r501_z / $(256)) % $(128)) * $(256)) + (r501_z % $(256))), dtype=tl.float32) * tl.cast(tl.load($(candidate_vto_copy_4_ptr) + (r501_z % $(256)), dtype=tl.bfloat16), tl.float32))
    r501_value = r501_value.to(tl.float32)
    r501_value = tl.broadcast_to(r501_value, [$(32768)])
    tl.store($(candidate_vmul_51_ptr) + r501_z, r501_value)
    r502_z = tl.arange($(32768))
    r502_value = (tl.load($(candidate_vmul_51_ptr) + ((((r502_z / $(256)) % $(128)) * $(256)) + (r502_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vto_copy_22_ptr) + ((((r502_z / $(256)) % $(128)) * $(256)) + (r502_z % $(256))), dtype=tl.float32))
    r502_value = r502_value.to(tl.float32)
    r502_value = tl.broadcast_to(r502_value, [$(32768)])
    tl.store($(candidate_vmul_52_ptr) + r502_z, r502_value)
    r503_z = tl.arange($(32768))
    r503_value = (tl.load($(candidate_vmul_51_ptr) + ((((r503_z / $(256)) % $(128)) * $(256)) + (r503_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_1_ptr) + ((r503_z / $(256)) % $(128)), dtype=tl.float32))
    r503_value = r503_value.to(tl.float32)
    r503_value = tl.broadcast_to(r503_value, [$(32768)])
    tl.store($(candidate_vmul_53_ptr) + r503_z, r503_value)
    r504_z = tl.arange($(128))
    r504_lane = tl.arange($(256))
    r504_value = tl.sum(tl.load($(candidate_vmul_52_ptr) + ((r504_z[:, None] * $(256)) + r504_lane[None, :]), dtype=tl.float32), axis=1)
    r504_value = r504_value.to(tl.float32)
    r504_value = tl.broadcast_to(r504_value, [$(128)])
    tl.store($(candidate_vsum_5_ptr) + r504_z, r504_value)
    r505_z = tl.arange($(128))
    r505_value = ((tl.load($(candidate_vrsqrt_1_ptr) + r505_z, dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_1_ptr) + r505_z, dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_1_ptr) + r505_z, dtype=tl.float32)))
    r505_value = r505_value.to(tl.float32)
    r505_value = tl.broadcast_to(r505_value, [$(128)])
    tl.store($(candidate_vpow_6_ptr) + r505_z, r505_value)
    r506_z = tl.arange($(128))
    r506_value = (tl.load($(candidate_vsum_5_ptr) + r506_z, dtype=tl.float32) * -0.5)
    r506_value = r506_value.to(tl.float32)
    r506_value = tl.broadcast_to(r506_value, [$(128)])
    tl.store($(candidate_vmul_54_ptr) + r506_z, r506_value)
    r507_z = tl.arange($(128))
    r507_value = (tl.load($(candidate_vmul_54_ptr) + r507_z, dtype=tl.float32) * tl.load($(candidate_vpow_6_ptr) + r507_z, dtype=tl.float32))
    r507_value = r507_value.to(tl.float32)
    r507_value = tl.broadcast_to(r507_value, [$(128)])
    tl.store($(candidate_vmul_55_ptr) + r507_z, r507_value)
    r508_z = tl.arange($(32768))
    r508_value = (tl.load($(candidate_vmul_55_ptr) + ((r508_z / $(256)) % $(128)), dtype=tl.float32) / 256.0)
    r508_value = r508_value.to(tl.float32)
    r508_value = tl.broadcast_to(r508_value, [$(32768)])
    tl.store($(candidate_vdiv_3_ptr) + r508_z, r508_value)
    r509_z = tl.arange($(32768))
    r509_value = ((tl.load($(candidate_vto_copy_22_ptr) + ((((r509_z / $(256)) % $(128)) * $(256)) + (r509_z % $(256))), dtype=tl.float32)))
    r509_value = r509_value.to(tl.float32)
    r509_value = tl.broadcast_to(r509_value, [$(32768)])
    tl.store($(candidate_vpow_7_ptr) + r509_z, r509_value)
    r510_z = tl.arange($(32768))
    r510_value = (tl.load($(candidate_vpow_7_ptr) + ((((r510_z / $(256)) % $(128)) * $(256)) + (r510_z % $(256))), dtype=tl.float32) * 2.0)
    r510_value = r510_value.to(tl.float32)
    r510_value = tl.broadcast_to(r510_value, [$(32768)])
    tl.store($(candidate_vmul_56_ptr) + r510_z, r510_value)
    r511_z = tl.arange($(32768))
    r511_value = (tl.load($(candidate_vdiv_3_ptr) + ((((r511_z / $(256)) % $(128)) * $(256)) + (r511_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vmul_56_ptr) + ((((r511_z / $(256)) % $(128)) * $(256)) + (r511_z % $(256))), dtype=tl.float32))
    r511_value = r511_value.to(tl.float32)
    r511_value = tl.broadcast_to(r511_value, [$(32768)])
    tl.store($(candidate_vmul_57_ptr) + r511_z, r511_value)
    r512_z = tl.arange($(32768))
    r512_value = (tl.load($(candidate_vmul_53_ptr) + ((((r512_z / $(256)) % $(128)) * $(256)) + (r512_z % $(256))), dtype=tl.float32) + tl.load($(candidate_vmul_57_ptr) + ((((r512_z / $(256)) % $(128)) * $(256)) + (r512_z % $(256))), dtype=tl.float32))
    r512_value = r512_value.to(tl.float32)
    r512_value = tl.broadcast_to(r512_value, [$(32768)])
    tl.store($(candidate_vadd_64_ptr) + r512_z, r512_value)
    r513_z = tl.arange($(32768))
    r513_value = tl.load($(candidate_vadd_64_ptr) + ((((r513_z / $(256)) % $(128)) * $(256)) + (r513_z % $(256))), dtype=tl.float32)
    r513_value = r513_value.to(tl.bfloat16)
    r513_value = tl.broadcast_to(r513_value, [$(32768)])
    tl.store($(candidate_vto_copy_102_ptr) + r513_z, r513_value)
    r514_z = tl.arange($(40960))
    r514_value = tl.where((r514_z % $(320)) < $(256), tl.load($(candidate_vto_copy_102_ptr) + ((((r514_z / $(320)) % $(128)) * $(256)) + (((r514_z % $(320)) - $(0)) % $(256))), dtype=tl.bfloat16), tl.load($(candidate_vadd_61_ptr) + ((((((((r514_z / $(320)) % $(128)) * $(64)) + ((r514_z % $(320)) - $(256))) / $(64)) % $(128)) * $(64)) + (((((r514_z / $(320)) % $(128)) * $(64)) + ((r514_z % $(320)) - $(256))) % $(64))), dtype=tl.bfloat16))
    r514_value = r514_value.to(tl.bfloat16)
    r514_value = tl.broadcast_to(r514_value, [$(40960)])
    tl.store($(candidate_vcat_21_ptr) + r514_z, r514_value)
    r515_row = tl.arange($(128))
    r515_col = tl.arange($(512))
    r515_lane = tl.arange($(320))
    r515_left = tl.load($(candidate_vcat_21_ptr) + ((((((r515_row[:, None] * $(320)) + r515_lane[None, :]) / $(320)) % $(128)) * $(320)) + (((r515_row[:, None] * $(320)) + r515_lane[None, :]) % $(320))), dtype=tl.bfloat16)
    r515_right = tl.load($(candidate_vto_copy_3_ptr) + (((r515_lane[:, None] % $(320)) * $(512)) + (r515_col[None, :] % $(512))), dtype=tl.bfloat16)
    r515_value = tl.dot(r515_left, r515_right)
    r515_value = r515_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_5_ptr) + ((r515_row[:, None] * $(512)) + r515_col[None, :]), r515_value)
    r516_row = tl.arange($(128))
    r516_col = tl.arange($(256))
    r516_lane = tl.arange($(1536))
    r516_left = tl.load($(candidate_vcat_20_ptr) + ((((((((((r516_row[:, None] * $(1536)) + r516_lane[None, :]) / $(1536)) % $(128)) * $(1536)) + (((r516_row[:, None] * $(1536)) + r516_lane[None, :]) % $(1536))) / $(192)) % $(8)) * $(24576)) + (((((r516_row[:, None] * $(1536)) + r516_lane[None, :]) / $(1536)) % $(128)) * $(192)) + (((((((r516_row[:, None] * $(1536)) + r516_lane[None, :]) / $(1536)) % $(128)) * $(1536)) + (((r516_row[:, None] * $(1536)) + r516_lane[None, :]) % $(1536))) % $(192))), dtype=tl.bfloat16)
    r516_right = tl.load($(candidate_vto_copy_2_ptr) + (((r516_lane[:, None] % $(1536)) * $(256)) + (r516_col[None, :] % $(256))), dtype=tl.bfloat16)
    r516_value = tl.dot(r516_left, r516_right)
    r516_value = r516_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_6_ptr) + ((r516_row[:, None] * $(256)) + r516_col[None, :]), r516_value)
    r517_z = tl.arange($(32768))
    r517_value = tl.load($(candidate_vmm_6_ptr) + ((((r517_z / $(256)) % $(128)) * $(256)) + (r517_z % $(256))), dtype=tl.bfloat16)
    r517_value = r517_value.to(tl.float32)
    r517_value = tl.broadcast_to(r517_value, [$(32768)])
    tl.store($(candidate_vto_copy_103_ptr) + r517_z, r517_value)
    r518_z = tl.arange($(32768))
    r518_value = (tl.load($(candidate_vto_copy_103_ptr) + ((((r518_z / $(256)) % $(128)) * $(256)) + (r518_z % $(256))), dtype=tl.float32) * tl.cast(tl.load($(candidate_vto_copy_1_ptr) + (r518_z % $(256)), dtype=tl.bfloat16), tl.float32))
    r518_value = r518_value.to(tl.float32)
    r518_value = tl.broadcast_to(r518_value, [$(32768)])
    tl.store($(candidate_vmul_58_ptr) + r518_z, r518_value)
    r519_z = tl.arange($(32768))
    r519_value = (tl.load($(candidate_vmul_58_ptr) + ((((r519_z / $(256)) % $(128)) * $(256)) + (r519_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vto_copy_20_ptr) + ((((r519_z / $(256)) % $(128)) * $(256)) + (r519_z % $(256))), dtype=tl.float32))
    r519_value = r519_value.to(tl.float32)
    r519_value = tl.broadcast_to(r519_value, [$(32768)])
    tl.store($(candidate_vmul_59_ptr) + r519_z, r519_value)
    r520_z = tl.arange($(32768))
    r520_value = (tl.load($(candidate_vmul_58_ptr) + ((((r520_z / $(256)) % $(128)) * $(256)) + (r520_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_ptr) + ((r520_z / $(256)) % $(128)), dtype=tl.float32))
    r520_value = r520_value.to(tl.float32)
    r520_value = tl.broadcast_to(r520_value, [$(32768)])
    tl.store($(candidate_vmul_60_ptr) + r520_z, r520_value)
    r521_z = tl.arange($(128))
    r521_lane = tl.arange($(256))
    r521_value = tl.sum(tl.load($(candidate_vmul_59_ptr) + ((r521_z[:, None] * $(256)) + r521_lane[None, :]), dtype=tl.float32), axis=1)
    r521_value = r521_value.to(tl.float32)
    r521_value = tl.broadcast_to(r521_value, [$(128)])
    tl.store($(candidate_vsum_6_ptr) + r521_z, r521_value)
    r522_z = tl.arange($(128))
    r522_value = ((tl.load($(candidate_vrsqrt_ptr) + r522_z, dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_ptr) + r522_z, dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_ptr) + r522_z, dtype=tl.float32)))
    r522_value = r522_value.to(tl.float32)
    r522_value = tl.broadcast_to(r522_value, [$(128)])
    tl.store($(candidate_vpow_8_ptr) + r522_z, r522_value)
    r523_z = tl.arange($(128))
    r523_value = (tl.load($(candidate_vsum_6_ptr) + r523_z, dtype=tl.float32) * -0.5)
    r523_value = r523_value.to(tl.float32)
    r523_value = tl.broadcast_to(r523_value, [$(128)])
    tl.store($(candidate_vmul_61_ptr) + r523_z, r523_value)
    r524_z = tl.arange($(128))
    r524_value = (tl.load($(candidate_vmul_61_ptr) + r524_z, dtype=tl.float32) * tl.load($(candidate_vpow_8_ptr) + r524_z, dtype=tl.float32))
    r524_value = r524_value.to(tl.float32)
    r524_value = tl.broadcast_to(r524_value, [$(128)])
    tl.store($(candidate_vmul_62_ptr) + r524_z, r524_value)
    r525_z = tl.arange($(32768))
    r525_value = (tl.load($(candidate_vmul_62_ptr) + ((r525_z / $(256)) % $(128)), dtype=tl.float32) / 256.0)
    r525_value = r525_value.to(tl.float32)
    r525_value = tl.broadcast_to(r525_value, [$(32768)])
    tl.store($(candidate_vdiv_4_ptr) + r525_z, r525_value)
    r526_z = tl.arange($(32768))
    r526_value = ((tl.load($(candidate_vto_copy_20_ptr) + ((((r526_z / $(256)) % $(128)) * $(256)) + (r526_z % $(256))), dtype=tl.float32)))
    r526_value = r526_value.to(tl.float32)
    r526_value = tl.broadcast_to(r526_value, [$(32768)])
    tl.store($(candidate_vpow_9_ptr) + r526_z, r526_value)
    r527_z = tl.arange($(32768))
    r527_value = (tl.load($(candidate_vpow_9_ptr) + ((((r527_z / $(256)) % $(128)) * $(256)) + (r527_z % $(256))), dtype=tl.float32) * 2.0)
    r527_value = r527_value.to(tl.float32)
    r527_value = tl.broadcast_to(r527_value, [$(32768)])
    tl.store($(candidate_vmul_63_ptr) + r527_z, r527_value)
    r528_z = tl.arange($(32768))
    r528_value = (tl.load($(candidate_vdiv_4_ptr) + ((((r528_z / $(256)) % $(128)) * $(256)) + (r528_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vmul_63_ptr) + ((((r528_z / $(256)) % $(128)) * $(256)) + (r528_z % $(256))), dtype=tl.float32))
    r528_value = r528_value.to(tl.float32)
    r528_value = tl.broadcast_to(r528_value, [$(32768)])
    tl.store($(candidate_vmul_64_ptr) + r528_z, r528_value)
    r529_z = tl.arange($(32768))
    r529_value = (tl.load($(candidate_vmul_60_ptr) + ((((r529_z / $(256)) % $(128)) * $(256)) + (r529_z % $(256))), dtype=tl.float32) + tl.load($(candidate_vmul_64_ptr) + ((((r529_z / $(256)) % $(128)) * $(256)) + (r529_z % $(256))), dtype=tl.float32))
    r529_value = r529_value.to(tl.float32)
    r529_value = tl.broadcast_to(r529_value, [$(32768)])
    tl.store($(candidate_vadd_65_ptr) + r529_z, r529_value)
    r530_z = tl.arange($(32768))
    r530_value = tl.load($(candidate_vadd_65_ptr) + ((((r530_z / $(256)) % $(128)) * $(256)) + (r530_z % $(256))), dtype=tl.float32)
    r530_value = r530_value.to(tl.bfloat16)
    r530_value = tl.broadcast_to(r530_value, [$(32768)])
    tl.store($(candidate_vto_copy_104_ptr) + r530_z, r530_value)
    r531_row = tl.arange($(128))
    r531_col = tl.arange($(512))
    r531_lane = tl.arange($(256))
    r531_left = tl.load($(candidate_vto_copy_104_ptr) + ((((((r531_row[:, None] * $(256)) + r531_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r531_row[:, None] * $(256)) + r531_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r531_right = tl.load($(candidate_vto_copy_ptr) + (((r531_lane[:, None] % $(256)) * $(512)) + (r531_col[None, :] % $(512))), dtype=tl.bfloat16)
    r531_value = tl.dot(r531_left, r531_right)
    r531_value = r531_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_7_ptr) + ((r531_row[:, None] * $(512)) + r531_col[None, :]), r531_value)
    r532_z = tl.arange($(65536))
    r532_value = (tl.load($(candidate_vmm_5_ptr) + ((((r532_z / $(512)) % $(128)) * $(512)) + (r532_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vmm_7_ptr) + ((((r532_z / $(512)) % $(128)) * $(512)) + (r532_z % $(512))), dtype=tl.bfloat16))
    r532_value = r532_value.to(tl.bfloat16)
    r532_value = tl.broadcast_to(r532_value, [$(65536)])
    tl.store($(candidate_vadd_66_ptr) + r532_z, r532_value)
    r533_z = tl.arange($(65536))
    r533_value = tl.cast(0.0, tl.bfloat16)
    for r533_lane in range($(128)) {
      r533_value = tl.where(tl.load($(constantee33028f0f_ptr) + r533_lane, dtype=tl.int64) == ((r533_z / $(512)) % $(128)), r533_value + tl.load($(candidate_vadd_66_ptr) + (((r533_lane % $(128)) * $(512)) + (r533_z % $(512))), dtype=tl.bfloat16), r533_value)
    }
    r533_value = r533_value.to(tl.bfloat16)
    r533_value = tl.broadcast_to(r533_value, [$(65536)])
    tl.store($(candidate_vindex_put_2_ptr) + r533_z, r533_value)
    z = tl.arange($(65536))
    value = tl.load($(candidate_vindex_put_2_ptr) + ((((z / $(512)) % $(128)) * $(512)) + (z % $(512))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(65536)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_mla_fa4_te_rope_cp_nonabsorb_dx
