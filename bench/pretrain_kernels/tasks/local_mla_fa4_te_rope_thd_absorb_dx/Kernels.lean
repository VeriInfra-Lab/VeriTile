import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_mla_fa4_te_rope_thd_absorb_dx
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
    r51_col = tl.arange($(576))
    r51_lane = tl.arange($(512))
    r51_left = tl.load($(input0_ptr) + ((((((r51_row[:, None] * $(512)) + r51_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r51_row[:, None] * $(512)) + r51_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r51_right = tl.load($(vto_copy_3_ptr) + (((r51_col[None, :] % $(576)) * $(512)) + (r51_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r51_value = tl.dot(r51_left, r51_right)
    r51_value = r51_value.to(tl.bfloat16)
    tl.store($(vmm_2_ptr) + ((r51_row[:, None] * $(576)) + r51_col[None, :]), r51_value)
    r52_z = tl.arange($(65536))
    r52_value = tl.load($(vmm_2_ptr) + ((((r52_z / $(512)) % $(128)) * $(576)) + (r52_z % $(512))), dtype=tl.bfloat16)
    r52_value = r52_value.to(tl.float32)
    r52_value = tl.broadcast_to(r52_value, [$(65536)])
    tl.store($(vto_copy_22_ptr) + r52_z, r52_value)
    r53_z = tl.arange($(65536))
    r53_value = ((tl.load($(vto_copy_22_ptr) + ((((r53_z / $(512)) % $(128)) * $(512)) + (r53_z % $(512))), dtype=tl.float32)) * (tl.load($(vto_copy_22_ptr) + ((((r53_z / $(512)) % $(128)) * $(512)) + (r53_z % $(512))), dtype=tl.float32)))
    r53_value = r53_value.to(tl.float32)
    r53_value = tl.broadcast_to(r53_value, [$(65536)])
    tl.store($(vpow_4_ptr) + r53_z, r53_value)
    r54_z = tl.arange($(128))
    r54_lane = tl.arange($(512))
    r54_value = (tl.sum(tl.load($(vpow_4_ptr) + ((r54_z[:, None] * $(512)) + r54_lane[None, :]), dtype=tl.float32), axis=1) / 512.0)
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
    r57_z = tl.arange($(65536))
    r57_value = (tl.load($(vto_copy_22_ptr) + ((((r57_z / $(512)) % $(128)) * $(512)) + (r57_z % $(512))), dtype=tl.float32) * tl.load($(vrsqrt_1_ptr) + ((r57_z / $(512)) % $(128)), dtype=tl.float32))
    r57_value = r57_value.to(tl.float32)
    r57_value = tl.broadcast_to(r57_value, [$(65536)])
    tl.store($(vmul_15_ptr) + r57_z, r57_value)
    r58_z = tl.arange($(65536))
    r58_value = (tl.cast(tl.load($(vto_copy_4_ptr) + (r58_z % $(512)), dtype=tl.bfloat16), tl.float32) * tl.load($(vmul_15_ptr) + ((((r58_z / $(512)) % $(128)) * $(512)) + (r58_z % $(512))), dtype=tl.float32))
    r58_value = r58_value.to(tl.float32)
    r58_value = tl.broadcast_to(r58_value, [$(65536)])
    tl.store($(vmul_16_ptr) + r58_z, r58_value)
    r59_z = tl.arange($(65536))
    r59_value = tl.load($(vmul_16_ptr) + ((((r59_z / $(512)) % $(128)) * $(512)) + (r59_z % $(512))), dtype=tl.float32)
    r59_value = r59_value.to(tl.bfloat16)
    r59_value = tl.broadcast_to(r59_value, [$(65536)])
    tl.store($(vto_copy_23_ptr) + r59_z, r59_value)
    r60_row = tl.arange($(128))
    r60_col = tl.arange($(2048))
    r60_lane = tl.arange($(512))
    r60_left = tl.load($(vto_copy_23_ptr) + ((((((r60_row[:, None] * $(512)) + r60_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r60_row[:, None] * $(512)) + r60_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r60_right = tl.load($(vto_copy_5_ptr) + (((r60_col[None, :] % $(2048)) * $(512)) + (r60_lane[:, None] % $(512))), dtype=tl.bfloat16)
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
    r68_value = (tl.load($(vmm_2_ptr) + ((((r68_z / $(32)) % $(128)) * $(576)) + ((r68_z % $(32)) * $(2)) + $(512)), dtype=tl.bfloat16) * tl.load($(vto_copy_17_ptr) + ((((r68_z / $(32)) % $(128)) * $(64)) + (r68_z % $(32))), dtype=tl.bfloat16))
    r68_value = r68_value.to(tl.bfloat16)
    r68_value = tl.broadcast_to(r68_value, [$(4096)])
    tl.store($(vmul_21_ptr) + r68_z, r68_value)
    r69_z = tl.arange($(4096))
    r69_value = (tl.load($(vmm_2_ptr) + ((((r69_z / $(32)) % $(128)) * $(576)) + ((r69_z % $(32)) * $(2)) + $(513)), dtype=tl.bfloat16) * tl.load($(vto_copy_18_ptr) + ((((r69_z / $(32)) % $(128)) * $(64)) + (r69_z % $(32))), dtype=tl.bfloat16))
    r69_value = r69_value.to(tl.bfloat16)
    r69_value = tl.broadcast_to(r69_value, [$(4096)])
    tl.store($(vmul_22_ptr) + r69_z, r69_value)
    r70_z = tl.arange($(4096))
    r70_value = (tl.load($(vmul_21_ptr) + ((((r70_z / $(32)) % $(128)) * $(32)) + (r70_z % $(32))), dtype=tl.bfloat16) - tl.load($(vmul_22_ptr) + ((((r70_z / $(32)) % $(128)) * $(32)) + (r70_z % $(32))), dtype=tl.bfloat16))
    r70_value = r70_value.to(tl.bfloat16)
    r70_value = tl.broadcast_to(r70_value, [$(4096)])
    tl.store($(vsub_5_ptr) + r70_z, r70_value)
    r71_z = tl.arange($(4096))
    r71_value = (tl.load($(vmm_2_ptr) + ((((r71_z / $(32)) % $(128)) * $(576)) + ((r71_z % $(32)) * $(2)) + $(513)), dtype=tl.bfloat16) * tl.load($(vto_copy_17_ptr) + ((((r71_z / $(32)) % $(128)) * $(64)) + (r71_z % $(32))), dtype=tl.bfloat16))
    r71_value = r71_value.to(tl.bfloat16)
    r71_value = tl.broadcast_to(r71_value, [$(4096)])
    tl.store($(vmul_23_ptr) + r71_z, r71_value)
    r72_z = tl.arange($(4096))
    r72_value = (tl.load($(vmm_2_ptr) + ((((r72_z / $(32)) % $(128)) * $(576)) + ((r72_z % $(32)) * $(2)) + $(512)), dtype=tl.bfloat16) * tl.load($(vto_copy_18_ptr) + ((((r72_z / $(32)) % $(128)) * $(64)) + (r72_z % $(32))), dtype=tl.bfloat16))
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
    r115_col = tl.arange($(512))
    r115_lane = tl.arange($(2048))
    r115_left = tl.load($(vcat_9_ptr) + ((((((((((r115_row[:, None] * $(2048)) + r115_lane[None, :]) / $(2048)) % $(128)) * $(2048)) + (((r115_row[:, None] * $(2048)) + r115_lane[None, :]) % $(2048))) / $(256)) % $(8)) * $(32768)) + (((((r115_row[:, None] * $(2048)) + r115_lane[None, :]) / $(2048)) % $(128)) * $(256)) + (((((((r115_row[:, None] * $(2048)) + r115_lane[None, :]) / $(2048)) % $(128)) * $(2048)) + (((r115_row[:, None] * $(2048)) + r115_lane[None, :]) % $(2048))) % $(256))), dtype=tl.bfloat16)
    r115_right = tl.load($(vto_copy_5_ptr) + (((r115_lane[:, None] % $(2048)) * $(512)) + (r115_col[None, :] % $(512))), dtype=tl.bfloat16)
    r115_value = tl.dot(r115_left, r115_right)
    r115_value = r115_value.to(tl.bfloat16)
    tl.store($(vmm_6_ptr) + ((r115_row[:, None] * $(512)) + r115_col[None, :]), r115_value)
    r116_z = tl.arange($(65536))
    r116_value = tl.load($(vmm_6_ptr) + ((((r116_z / $(512)) % $(128)) * $(512)) + (r116_z % $(512))), dtype=tl.bfloat16)
    r116_value = r116_value.to(tl.float32)
    r116_value = tl.broadcast_to(r116_value, [$(65536)])
    tl.store($(vto_copy_28_ptr) + r116_z, r116_value)
    r117_z = tl.arange($(65536))
    r117_value = (tl.load($(vto_copy_28_ptr) + ((((r117_z / $(512)) % $(128)) * $(512)) + (r117_z % $(512))), dtype=tl.float32) * tl.cast(tl.load($(vto_copy_4_ptr) + (r117_z % $(512)), dtype=tl.bfloat16), tl.float32))
    r117_value = r117_value.to(tl.float32)
    r117_value = tl.broadcast_to(r117_value, [$(65536)])
    tl.store($(vmul_35_ptr) + r117_z, r117_value)
    r118_z = tl.arange($(65536))
    r118_value = (tl.load($(vmul_35_ptr) + ((((r118_z / $(512)) % $(128)) * $(512)) + (r118_z % $(512))), dtype=tl.float32) * tl.load($(vto_copy_22_ptr) + ((((r118_z / $(512)) % $(128)) * $(512)) + (r118_z % $(512))), dtype=tl.float32))
    r118_value = r118_value.to(tl.float32)
    r118_value = tl.broadcast_to(r118_value, [$(65536)])
    tl.store($(vmul_36_ptr) + r118_z, r118_value)
    r119_z = tl.arange($(65536))
    r119_value = (tl.load($(vmul_35_ptr) + ((((r119_z / $(512)) % $(128)) * $(512)) + (r119_z % $(512))), dtype=tl.float32) * tl.load($(vrsqrt_1_ptr) + ((r119_z / $(512)) % $(128)), dtype=tl.float32))
    r119_value = r119_value.to(tl.float32)
    r119_value = tl.broadcast_to(r119_value, [$(65536)])
    tl.store($(vmul_37_ptr) + r119_z, r119_value)
    r120_z = tl.arange($(128))
    r120_lane = tl.arange($(512))
    r120_value = tl.sum(tl.load($(vmul_36_ptr) + ((r120_z[:, None] * $(512)) + r120_lane[None, :]), dtype=tl.float32), axis=1)
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
    r124_z = tl.arange($(65536))
    r124_value = (tl.load($(vmul_39_ptr) + ((r124_z / $(512)) % $(128)), dtype=tl.float32) / 512.0)
    r124_value = r124_value.to(tl.float32)
    r124_value = tl.broadcast_to(r124_value, [$(65536)])
    tl.store($(vdiv_2_ptr) + r124_z, r124_value)
    r125_z = tl.arange($(65536))
    r125_value = ((tl.load($(vto_copy_22_ptr) + ((((r125_z / $(512)) % $(128)) * $(512)) + (r125_z % $(512))), dtype=tl.float32)))
    r125_value = r125_value.to(tl.float32)
    r125_value = tl.broadcast_to(r125_value, [$(65536)])
    tl.store($(vpow_6_ptr) + r125_z, r125_value)
    r126_z = tl.arange($(65536))
    r126_value = (tl.load($(vpow_6_ptr) + ((((r126_z / $(512)) % $(128)) * $(512)) + (r126_z % $(512))), dtype=tl.float32) * 2.0)
    r126_value = r126_value.to(tl.float32)
    r126_value = tl.broadcast_to(r126_value, [$(65536)])
    tl.store($(vmul_40_ptr) + r126_z, r126_value)
    r127_z = tl.arange($(65536))
    r127_value = (tl.load($(vdiv_2_ptr) + ((((r127_z / $(512)) % $(128)) * $(512)) + (r127_z % $(512))), dtype=tl.float32) * tl.load($(vmul_40_ptr) + ((((r127_z / $(512)) % $(128)) * $(512)) + (r127_z % $(512))), dtype=tl.float32))
    r127_value = r127_value.to(tl.float32)
    r127_value = tl.broadcast_to(r127_value, [$(65536)])
    tl.store($(vmul_41_ptr) + r127_z, r127_value)
    r128_z = tl.arange($(65536))
    r128_value = (tl.load($(vmul_37_ptr) + ((((r128_z / $(512)) % $(128)) * $(512)) + (r128_z % $(512))), dtype=tl.float32) + tl.load($(vmul_41_ptr) + ((((r128_z / $(512)) % $(128)) * $(512)) + (r128_z % $(512))), dtype=tl.float32))
    r128_value = r128_value.to(tl.float32)
    r128_value = tl.broadcast_to(r128_value, [$(65536)])
    tl.store($(vadd_14_ptr) + r128_z, r128_value)
    r129_z = tl.arange($(65536))
    r129_value = tl.load($(vadd_14_ptr) + ((((r129_z / $(512)) % $(128)) * $(512)) + (r129_z % $(512))), dtype=tl.float32)
    r129_value = r129_value.to(tl.bfloat16)
    r129_value = tl.broadcast_to(r129_value, [$(65536)])
    tl.store($(vto_copy_29_ptr) + r129_z, r129_value)
    r130_z = tl.arange($(73728))
    r130_value = tl.where((r130_z % $(576)) < $(512), tl.load($(vto_copy_29_ptr) + ((((r130_z / $(576)) % $(128)) * $(512)) + (((r130_z % $(576)) - $(0)) % $(512))), dtype=tl.bfloat16), tl.load($(vadd_12_ptr) + ((((((((r130_z / $(576)) % $(128)) * $(64)) + ((r130_z % $(576)) - $(512))) / $(64)) % $(128)) * $(64)) + (((((r130_z / $(576)) % $(128)) * $(64)) + ((r130_z % $(576)) - $(512))) % $(64))), dtype=tl.bfloat16))
    r130_value = r130_value.to(tl.bfloat16)
    r130_value = tl.broadcast_to(r130_value, [$(73728)])
    tl.store($(vcat_11_ptr) + r130_z, r130_value)
    r131_row = tl.arange($(128))
    r131_col = tl.arange($(512))
    r131_lane = tl.arange($(576))
    r131_left = tl.load($(vcat_11_ptr) + ((((((r131_row[:, None] * $(576)) + r131_lane[None, :]) / $(576)) % $(128)) * $(576)) + (((r131_row[:, None] * $(576)) + r131_lane[None, :]) % $(576))), dtype=tl.bfloat16)
    r131_right = tl.load($(vto_copy_3_ptr) + (((r131_lane[:, None] % $(576)) * $(512)) + (r131_col[None, :] % $(512))), dtype=tl.bfloat16)
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
  let vmul_19_ptr : RegionName := "vmul_19"
  let vmul_20_ptr : RegionName := "vmul_20"
  let vadd_5_ptr : RegionName := "vadd_5"
  let vcat_5_ptr : RegionName := "vcat_5"
  let vmul_21_ptr : RegionName := "vmul_21"
  let vmul_22_ptr : RegionName := "vmul_22"
  let vadd_6_ptr : RegionName := "vadd_6"
  let vcat_6_ptr : RegionName := "vcat_6"
  let vcat_8_ptr : RegionName := "vcat_8"
  let vto_copy_24_ptr : RegionName := "vto_copy_24"
  let vto_copy_25_ptr : RegionName := "vto_copy_25"
  let vto_copy_26_ptr : RegionName := "vto_copy_26"
  let vto_copy_27_ptr : RegionName := "vto_copy_27"
  let vmul_26_ptr : RegionName := "vmul_26"
  let vbmm_7_ptr : RegionName := "vbmm_7"
  let vsum_1_ptr : RegionName := "vsum_1"
  let vmul_27_ptr : RegionName := "vmul_27"
  let vmul_28_ptr : RegionName := "vmul_28"
  let vneg_ptr : RegionName := "vneg"
  let vmul_29_ptr : RegionName := "vmul_29"
  let vmul_30_ptr : RegionName := "vmul_30"
  let vmul_31_ptr : RegionName := "vmul_31"
  let vmul_32_ptr : RegionName := "vmul_32"
  let vneg_1_ptr : RegionName := "vneg_1"
  let vmul_33_ptr : RegionName := "vmul_33"
  let vmul_34_ptr : RegionName := "vmul_34"
  let vadd_13_ptr : RegionName := "vadd_13"
  let vcat_9_ptr : RegionName := "vcat_9"
  let vcat_10_ptr : RegionName := "vcat_10"
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
  let vcat_11_ptr : RegionName := "vcat_11"
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
  let candidate_vmm_10_ptr : RegionName := "candidate_vmm_10"
  let candidate_vbmm_7_ptr : RegionName := "candidate_vbmm_7"
  let candidate_vbmm_8_ptr : RegionName := "candidate_vbmm_8"
  let candidate_vcat_8_ptr : RegionName := "candidate_vcat_8"
  let candidate_vcat_9_ptr : RegionName := "candidate_vcat_9"
  let candidate_vto_copy_43_ptr : RegionName := "candidate_vto_copy_43"
  let candidate_vto_copy_44_ptr : RegionName := "candidate_vto_copy_44"
  let candidate_vbmm_9_ptr : RegionName := "candidate_vbmm_9"
  let candidate_vmul_26_ptr : RegionName := "candidate_vmul_26"
  let candidate_vadd_13_ptr : RegionName := "candidate_vadd_13"
  let candidate_vsub_7_ptr : RegionName := "candidate_vsub_7"
  let candidate_vgt_3_ptr : RegionName := "candidate_vgt_3"
  let candidate_vmasked_fill_3_ptr : RegionName := "candidate_vmasked_fill_3"
  let candidate_vsoftmax_3_ptr : RegionName := "candidate_vsoftmax_3"
  let candidate_vto_copy_45_ptr : RegionName := "candidate_vto_copy_45"
  let candidate_vbmm_10_ptr : RegionName := "candidate_vbmm_10"
  let candidate_vlogsumexp_3_ptr : RegionName := "candidate_vlogsumexp_3"
  let candidate_vto_copy_46_ptr : RegionName := "candidate_vto_copy_46"
  let candidate_vto_copy_47_ptr : RegionName := "candidate_vto_copy_47"
  let candidate_vbmm_11_ptr : RegionName := "candidate_vbmm_11"
  let candidate_vmul_27_ptr : RegionName := "candidate_vmul_27"
  let candidate_vadd_14_ptr : RegionName := "candidate_vadd_14"
  let candidate_vsub_8_ptr : RegionName := "candidate_vsub_8"
  let candidate_vgt_4_ptr : RegionName := "candidate_vgt_4"
  let candidate_vmasked_fill_4_ptr : RegionName := "candidate_vmasked_fill_4"
  let candidate_vsoftmax_4_ptr : RegionName := "candidate_vsoftmax_4"
  let candidate_vto_copy_48_ptr : RegionName := "candidate_vto_copy_48"
  let candidate_vbmm_12_ptr : RegionName := "candidate_vbmm_12"
  let candidate_vlogsumexp_4_ptr : RegionName := "candidate_vlogsumexp_4"
  let candidate_vto_copy_49_ptr : RegionName := "candidate_vto_copy_49"
  let candidate_vto_copy_50_ptr : RegionName := "candidate_vto_copy_50"
  let candidate_vbmm_13_ptr : RegionName := "candidate_vbmm_13"
  let candidate_vmul_28_ptr : RegionName := "candidate_vmul_28"
  let candidate_vadd_15_ptr : RegionName := "candidate_vadd_15"
  let candidate_vsub_9_ptr : RegionName := "candidate_vsub_9"
  let candidate_vgt_5_ptr : RegionName := "candidate_vgt_5"
  let candidate_vmasked_fill_5_ptr : RegionName := "candidate_vmasked_fill_5"
  let candidate_vsoftmax_5_ptr : RegionName := "candidate_vsoftmax_5"
  let candidate_vto_copy_51_ptr : RegionName := "candidate_vto_copy_51"
  let candidate_vbmm_14_ptr : RegionName := "candidate_vbmm_14"
  let candidate_vlogsumexp_5_ptr : RegionName := "candidate_vlogsumexp_5"
  let candidate_vcat_10_ptr : RegionName := "candidate_vcat_10"
  let candidate_vcat_11_ptr : RegionName := "candidate_vcat_11"
  let candidate_vto_copy_52_ptr : RegionName := "candidate_vto_copy_52"
  let candidate_vto_copy_53_ptr : RegionName := "candidate_vto_copy_53"
  let candidate_vto_copy_54_ptr : RegionName := "candidate_vto_copy_54"
  let candidate_vto_copy_55_ptr : RegionName := "candidate_vto_copy_55"
  let candidate_vto_copy_56_ptr : RegionName := "candidate_vto_copy_56"
  let candidate_vbmm_15_ptr : RegionName := "candidate_vbmm_15"
  let candidate_vmul_29_ptr : RegionName := "candidate_vmul_29"
  let candidate_vadd_16_ptr : RegionName := "candidate_vadd_16"
  let candidate_vsub_10_ptr : RegionName := "candidate_vsub_10"
  let candidate_vgt_6_ptr : RegionName := "candidate_vgt_6"
  let candidate_vmasked_fill_6_ptr : RegionName := "candidate_vmasked_fill_6"
  let candidate_vsub_11_ptr : RegionName := "candidate_vsub_11"
  let candidate_vexp_ptr : RegionName := "candidate_vexp"
  let candidate_vmul_30_ptr : RegionName := "candidate_vmul_30"
  let candidate_vsum_1_ptr : RegionName := "candidate_vsum_1"
  let candidate_vbmm_16_ptr : RegionName := "candidate_vbmm_16"
  let candidate_vsub_12_ptr : RegionName := "candidate_vsub_12"
  let candidate_vmul_31_ptr : RegionName := "candidate_vmul_31"
  let candidate_vbmm_17_ptr : RegionName := "candidate_vbmm_17"
  let candidate_vmul_32_ptr : RegionName := "candidate_vmul_32"
  let candidate_vto_copy_57_ptr : RegionName := "candidate_vto_copy_57"
  let candidate_vbmm_18_ptr : RegionName := "candidate_vbmm_18"
  let candidate_vmul_33_ptr : RegionName := "candidate_vmul_33"
  let candidate_vto_copy_58_ptr : RegionName := "candidate_vto_copy_58"
  let candidate_vbmm_19_ptr : RegionName := "candidate_vbmm_19"
  let candidate_vto_copy_59_ptr : RegionName := "candidate_vto_copy_59"
  let candidate_vto_copy_60_ptr : RegionName := "candidate_vto_copy_60"
  let candidate_vto_copy_61_ptr : RegionName := "candidate_vto_copy_61"
  let candidate_vto_copy_62_ptr : RegionName := "candidate_vto_copy_62"
  let candidate_vto_copy_63_ptr : RegionName := "candidate_vto_copy_63"
  let candidate_vto_copy_64_ptr : RegionName := "candidate_vto_copy_64"
  let candidate_vbmm_20_ptr : RegionName := "candidate_vbmm_20"
  let candidate_vmul_34_ptr : RegionName := "candidate_vmul_34"
  let candidate_vadd_17_ptr : RegionName := "candidate_vadd_17"
  let candidate_vsub_13_ptr : RegionName := "candidate_vsub_13"
  let candidate_vgt_7_ptr : RegionName := "candidate_vgt_7"
  let candidate_vmasked_fill_7_ptr : RegionName := "candidate_vmasked_fill_7"
  let candidate_vsub_14_ptr : RegionName := "candidate_vsub_14"
  let candidate_vexp_1_ptr : RegionName := "candidate_vexp_1"
  let candidate_vmul_35_ptr : RegionName := "candidate_vmul_35"
  let candidate_vsum_2_ptr : RegionName := "candidate_vsum_2"
  let candidate_vbmm_21_ptr : RegionName := "candidate_vbmm_21"
  let candidate_vsub_15_ptr : RegionName := "candidate_vsub_15"
  let candidate_vmul_36_ptr : RegionName := "candidate_vmul_36"
  let candidate_vbmm_22_ptr : RegionName := "candidate_vbmm_22"
  let candidate_vmul_37_ptr : RegionName := "candidate_vmul_37"
  let candidate_vto_copy_65_ptr : RegionName := "candidate_vto_copy_65"
  let candidate_vbmm_23_ptr : RegionName := "candidate_vbmm_23"
  let candidate_vmul_38_ptr : RegionName := "candidate_vmul_38"
  let candidate_vto_copy_66_ptr : RegionName := "candidate_vto_copy_66"
  let candidate_vbmm_24_ptr : RegionName := "candidate_vbmm_24"
  let candidate_vto_copy_67_ptr : RegionName := "candidate_vto_copy_67"
  let candidate_vto_copy_68_ptr : RegionName := "candidate_vto_copy_68"
  let candidate_vto_copy_69_ptr : RegionName := "candidate_vto_copy_69"
  let candidate_vto_copy_70_ptr : RegionName := "candidate_vto_copy_70"
  let candidate_vto_copy_71_ptr : RegionName := "candidate_vto_copy_71"
  let candidate_vto_copy_72_ptr : RegionName := "candidate_vto_copy_72"
  let candidate_vbmm_25_ptr : RegionName := "candidate_vbmm_25"
  let candidate_vmul_39_ptr : RegionName := "candidate_vmul_39"
  let candidate_vadd_18_ptr : RegionName := "candidate_vadd_18"
  let candidate_vsub_16_ptr : RegionName := "candidate_vsub_16"
  let candidate_vgt_8_ptr : RegionName := "candidate_vgt_8"
  let candidate_vmasked_fill_8_ptr : RegionName := "candidate_vmasked_fill_8"
  let candidate_vsub_17_ptr : RegionName := "candidate_vsub_17"
  let candidate_vexp_2_ptr : RegionName := "candidate_vexp_2"
  let candidate_vmul_40_ptr : RegionName := "candidate_vmul_40"
  let candidate_vsum_3_ptr : RegionName := "candidate_vsum_3"
  let candidate_vbmm_26_ptr : RegionName := "candidate_vbmm_26"
  let candidate_vsub_18_ptr : RegionName := "candidate_vsub_18"
  let candidate_vmul_41_ptr : RegionName := "candidate_vmul_41"
  let candidate_vbmm_27_ptr : RegionName := "candidate_vbmm_27"
  let candidate_vmul_42_ptr : RegionName := "candidate_vmul_42"
  let candidate_vto_copy_73_ptr : RegionName := "candidate_vto_copy_73"
  let candidate_vbmm_28_ptr : RegionName := "candidate_vbmm_28"
  let candidate_vmul_43_ptr : RegionName := "candidate_vmul_43"
  let candidate_vto_copy_74_ptr : RegionName := "candidate_vto_copy_74"
  let candidate_vbmm_29_ptr : RegionName := "candidate_vbmm_29"
  let candidate_vto_copy_75_ptr : RegionName := "candidate_vto_copy_75"
  let candidate_vcat_12_ptr : RegionName := "candidate_vcat_12"
  let candidate_vcat_13_ptr : RegionName := "candidate_vcat_13"
  let candidate_vcat_14_ptr : RegionName := "candidate_vcat_14"
  let candidate_vsum_4_ptr : RegionName := "candidate_vsum_4"
  let candidate_vbmm_30_ptr : RegionName := "candidate_vbmm_30"
  let candidate_vbmm_31_ptr : RegionName := "candidate_vbmm_31"
  let candidate_vadd_19_ptr : RegionName := "candidate_vadd_19"
  let candidate_vmul_44_ptr : RegionName := "candidate_vmul_44"
  let candidate_vmul_45_ptr : RegionName := "candidate_vmul_45"
  let candidate_vselect_backward_ptr : RegionName := "candidate_vselect_backward"
  let candidate_vneg_2_ptr : RegionName := "candidate_vneg_2"
  let candidate_vselect_backward_1_ptr : RegionName := "candidate_vselect_backward_1"
  let candidate_vadd_20_ptr : RegionName := "candidate_vadd_20"
  let candidate_vadd_21_ptr : RegionName := "candidate_vadd_21"
  let candidate_vmul_46_ptr : RegionName := "candidate_vmul_46"
  let candidate_vmul_47_ptr : RegionName := "candidate_vmul_47"
  let candidate_vselect_backward_2_ptr : RegionName := "candidate_vselect_backward_2"
  let candidate_vneg_3_ptr : RegionName := "candidate_vneg_3"
  let candidate_vselect_backward_3_ptr : RegionName := "candidate_vselect_backward_3"
  let candidate_vadd_22_ptr : RegionName := "candidate_vadd_22"
  let candidate_vadd_23_ptr : RegionName := "candidate_vadd_23"
  let candidate_vcat_16_ptr : RegionName := "candidate_vcat_16"
  let candidate_vto_copy_76_ptr : RegionName := "candidate_vto_copy_76"
  let candidate_vmul_48_ptr : RegionName := "candidate_vmul_48"
  let candidate_vmul_49_ptr : RegionName := "candidate_vmul_49"
  let candidate_vmul_50_ptr : RegionName := "candidate_vmul_50"
  let candidate_vsum_5_ptr : RegionName := "candidate_vsum_5"
  let candidate_vpow_6_ptr : RegionName := "candidate_vpow_6"
  let candidate_vmul_51_ptr : RegionName := "candidate_vmul_51"
  let candidate_vmul_52_ptr : RegionName := "candidate_vmul_52"
  let candidate_vdiv_3_ptr : RegionName := "candidate_vdiv_3"
  let candidate_vpow_7_ptr : RegionName := "candidate_vpow_7"
  let candidate_vmul_53_ptr : RegionName := "candidate_vmul_53"
  let candidate_vmul_54_ptr : RegionName := "candidate_vmul_54"
  let candidate_vadd_24_ptr : RegionName := "candidate_vadd_24"
  let candidate_vto_copy_77_ptr : RegionName := "candidate_vto_copy_77"
  let candidate_vcat_17_ptr : RegionName := "candidate_vcat_17"
  let candidate_vmm_11_ptr : RegionName := "candidate_vmm_11"
  let candidate_vmm_12_ptr : RegionName := "candidate_vmm_12"
  let candidate_vto_copy_78_ptr : RegionName := "candidate_vto_copy_78"
  let candidate_vmul_55_ptr : RegionName := "candidate_vmul_55"
  let candidate_vmul_56_ptr : RegionName := "candidate_vmul_56"
  let candidate_vmul_57_ptr : RegionName := "candidate_vmul_57"
  let candidate_vsum_6_ptr : RegionName := "candidate_vsum_6"
  let candidate_vpow_8_ptr : RegionName := "candidate_vpow_8"
  let candidate_vmul_58_ptr : RegionName := "candidate_vmul_58"
  let candidate_vmul_59_ptr : RegionName := "candidate_vmul_59"
  let candidate_vdiv_4_ptr : RegionName := "candidate_vdiv_4"
  let candidate_vpow_9_ptr : RegionName := "candidate_vpow_9"
  let candidate_vmul_60_ptr : RegionName := "candidate_vmul_60"
  let candidate_vmul_61_ptr : RegionName := "candidate_vmul_61"
  let candidate_vadd_25_ptr : RegionName := "candidate_vadd_25"
  let candidate_vto_copy_79_ptr : RegionName := "candidate_vto_copy_79"
  let candidate_vmm_13_ptr : RegionName := "candidate_vmm_13"
  let candidate_vadd_26_ptr : RegionName := "candidate_vadd_26"
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
    r26_col = tl.arange($(576))
    r26_lane = tl.arange($(512))
    r26_left = tl.load($(input0_ptr) + ((((((r26_row[:, None] * $(512)) + r26_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r26_row[:, None] * $(512)) + r26_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r26_right = tl.load($(candidate_vto_copy_3_ptr) + (((r26_col[None, :] % $(576)) * $(512)) + (r26_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r26_value = tl.dot(r26_left, r26_right)
    r26_value = r26_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_2_ptr) + ((r26_row[:, None] * $(576)) + r26_col[None, :]), r26_value)
    r27_z = tl.arange($(65536))
    r27_value = tl.load($(candidate_vmm_2_ptr) + ((((r27_z / $(512)) % $(128)) * $(576)) + (r27_z % $(512))), dtype=tl.bfloat16)
    r27_value = r27_value.to(tl.float32)
    r27_value = tl.broadcast_to(r27_value, [$(65536)])
    tl.store($(candidate_vto_copy_22_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(65536))
    r28_value = ((tl.load($(candidate_vto_copy_22_ptr) + ((((r28_z / $(512)) % $(128)) * $(512)) + (r28_z % $(512))), dtype=tl.float32)) * (tl.load($(candidate_vto_copy_22_ptr) + ((((r28_z / $(512)) % $(128)) * $(512)) + (r28_z % $(512))), dtype=tl.float32)))
    r28_value = r28_value.to(tl.float32)
    r28_value = tl.broadcast_to(r28_value, [$(65536)])
    tl.store($(candidate_vpow_4_ptr) + r28_z, r28_value)
    r29_z = tl.arange($(128))
    r29_lane = tl.arange($(512))
    r29_value = (tl.sum(tl.load($(candidate_vpow_4_ptr) + ((r29_z[:, None] * $(512)) + r29_lane[None, :]), dtype=tl.float32), axis=1) / 512.0)
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
    r32_z = tl.arange($(65536))
    r32_value = (tl.load($(candidate_vto_copy_22_ptr) + ((((r32_z / $(512)) % $(128)) * $(512)) + (r32_z % $(512))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_1_ptr) + ((r32_z / $(512)) % $(128)), dtype=tl.float32))
    r32_value = r32_value.to(tl.float32)
    r32_value = tl.broadcast_to(r32_value, [$(65536)])
    tl.store($(candidate_vmul_15_ptr) + r32_z, r32_value)
    r33_z = tl.arange($(65536))
    r33_value = (tl.cast(tl.load($(candidate_vto_copy_4_ptr) + (r33_z % $(512)), dtype=tl.bfloat16), tl.float32) * tl.load($(candidate_vmul_15_ptr) + ((((r33_z / $(512)) % $(128)) * $(512)) + (r33_z % $(512))), dtype=tl.float32))
    r33_value = r33_value.to(tl.float32)
    r33_value = tl.broadcast_to(r33_value, [$(65536)])
    tl.store($(candidate_vmul_16_ptr) + r33_z, r33_value)
    r34_z = tl.arange($(65536))
    r34_value = tl.load($(candidate_vmul_16_ptr) + ((((r34_z / $(512)) % $(128)) * $(512)) + (r34_z % $(512))), dtype=tl.float32)
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
    r55_value = -tl.load($(candidate_vmm_2_ptr) + ((((r55_z / $(32)) % $(128)) * $(576)) + ((r55_z % $(32)) * $(2)) + $(513)), dtype=tl.bfloat16)
    r55_value = r55_value.to(tl.bfloat16)
    r55_value = tl.broadcast_to(r55_value, [$(4096)])
    tl.store($(candidate_vneg_1_ptr) + r55_z, r55_value)
    r56_z = tl.arange($(8192))
    r56_value = tl.where((r56_z % $(2)) < $(1), tl.load($(candidate_vneg_1_ptr) + ((((r56_z / $(64)) % $(128)) * $(32)) + ((r56_z / $(2)) % $(32))), dtype=tl.bfloat16), tl.load($(candidate_vmm_2_ptr) + ((((r56_z / $(64)) % $(128)) * $(576)) + (((r56_z / $(2)) % $(32)) * $(2)) + $(512)), dtype=tl.bfloat16))
    r56_value = r56_value.to(tl.bfloat16)
    r56_value = tl.broadcast_to(r56_value, [$(8192)])
    tl.store($(candidate_vstack_1_ptr) + r56_z, r56_value)
    r57_z = tl.arange($(8192))
    r57_value = (tl.load($(candidate_vmm_2_ptr) + ((((r57_z / $(64)) % $(128)) * $(576)) + (r57_z % $(64)) + $(512)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_26_ptr) + ((((r57_z / $(64)) % $(128)) * $(64)) + (r57_z % $(64))), dtype=tl.bfloat16))
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
    r60_lane = tl.arange($(512))
    r60_left = tl.load($(input1_ptr) + ((((((r60_row[:, None] * $(512)) + r60_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r60_row[:, None] * $(512)) + r60_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r60_right = tl.load($(candidate_vto_copy_6_ptr) + (((r60_lane[:, None] % $(512)) * $(1024)) + (r60_col[None, :] % $(1024))), dtype=tl.bfloat16)
    r60_value = tl.dot(r60_left, r60_right)
    r60_value = r60_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_10_ptr) + ((r60_row[:, None] * $(1024)) + r60_col[None, :]), r60_value)
    r61_row = tl.arange($(128))
    r61_col = tl.arange($(1024))
    r61_lane = tl.arange($(512))
    for r61_group in range($(1)) {
      r61_left = tl.load($(candidate_vto_copy_23_ptr) + ((((((r61_row[:, None] * $(512)) + r61_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r61_row[:, None] * $(512)) + r61_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r61_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r61_lane[:, None] * $(1024)) + r61_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r61_lane[:, None] * $(1024)) + r61_col[None, :]) % $(128)) * $(512)) + ((((r61_lane[:, None] * $(1024)) + r61_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r61_value = tl.dot(r61_left, r61_right)
      r61_value = r61_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_7_ptr) + ((r61_group * $(131072)) + (r61_row[:, None] * $(1024)) + r61_col[None, :]), r61_value)
    }
    r62_row = tl.arange($(128))
    r62_col = tl.arange($(1024))
    r62_lane = tl.arange($(512))
    for r62_group in range($(1)) {
      r62_left = tl.load($(candidate_vto_copy_23_ptr) + ((((((r62_row[:, None] * $(512)) + r62_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r62_row[:, None] * $(512)) + r62_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r62_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r62_lane[:, None] * $(1024)) + r62_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r62_lane[:, None] * $(1024)) + r62_col[None, :]) % $(128)) * $(512)) + ((((r62_lane[:, None] * $(1024)) + r62_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r62_value = tl.dot(r62_left, r62_right)
      r62_value = r62_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_8_ptr) + ((r62_group * $(131072)) + (r62_row[:, None] * $(1024)) + r62_col[None, :]), r62_value)
    }
    r63_z = tl.arange($(196608))
    r63_value = tl.where((r63_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r63_z / $(1536)) % $(128)) * $(1024)) + (((r63_z / $(192)) % $(8)) * $(128)) + ((r63_z % $(192)) - $(0))) / $(1024)) % $(128)) * $(1536)) + (((((((r63_z / $(1536)) % $(128)) * $(1024)) + (((r63_z / $(192)) % $(8)) * $(128)) + ((r63_z % $(192)) - $(0))) / $(128)) % $(8)) * $(192)) + (((((r63_z / $(1536)) % $(128)) * $(1024)) + (((r63_z / $(192)) % $(8)) * $(128)) + ((r63_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vadd_5_ptr) + ((((r63_z / $(1536)) % $(128)) * $(512)) + (((r63_z / $(192)) % $(8)) * $(64)) + (((r63_z % $(192)) - $(128)) % $(64))), dtype=tl.bfloat16))
    r63_value = r63_value.to(tl.bfloat16)
    r63_value = tl.broadcast_to(r63_value, [$(196608)])
    tl.store($(candidate_vcat_8_ptr) + r63_z, r63_value)
    r64_z = tl.arange($(196608))
    r64_value = tl.where((r64_z % $(192)) < $(128), tl.load($(candidate_vbmm_7_ptr) + ((((((((r64_z / $(1536)) % $(128)) * $(1024)) + (((r64_z / $(192)) % $(8)) * $(128)) + ((r64_z % $(192)) - $(0))) / $(1024)) % $(128)) * $(1024)) + (((((((r64_z / $(1536)) % $(128)) * $(1024)) + (((r64_z / $(192)) % $(8)) * $(128)) + ((r64_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r64_z / $(1536)) % $(128)) * $(1024)) + (((r64_z / $(192)) % $(8)) * $(128)) + ((r64_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vadd_6_ptr) + ((((r64_z / $(1536)) % $(128)) * $(64)) + (((r64_z % $(192)) - $(128)) % $(64))), dtype=tl.bfloat16))
    r64_value = r64_value.to(tl.bfloat16)
    r64_value = tl.broadcast_to(r64_value, [$(196608)])
    tl.store($(candidate_vcat_9_ptr) + r64_z, r64_value)
    r65_z = tl.arange($(73728))
    r65_value = tl.load($(candidate_vcat_8_ptr) + ((((r65_z / $(192)) % $(48)) * $(1536)) + (((r65_z / $(9216)) % $(8)) * $(192)) + (r65_z % $(192))), dtype=tl.bfloat16)
    r65_value = r65_value.to(tl.float32)
    r65_value = tl.broadcast_to(r65_value, [$(73728)])
    tl.store($(candidate_vto_copy_43_ptr) + r65_z, r65_value)
    r66_z = tl.arange($(73728))
    r66_value = tl.load($(candidate_vcat_9_ptr) + ((((r66_z / $(192)) % $(48)) * $(1536)) + (((r66_z / $(9216)) % $(8)) * $(192)) + (r66_z % $(192))), dtype=tl.bfloat16)
    r66_value = r66_value.to(tl.float32)
    r66_value = tl.broadcast_to(r66_value, [$(73728)])
    tl.store($(candidate_vto_copy_44_ptr) + r66_z, r66_value)
    r67_row = tl.arange($(48))
    r67_col = tl.arange($(48))
    r67_lane = tl.arange($(192))
    for r67_group in range($(8)) {
      r67_left = tl.load($(candidate_vto_copy_43_ptr) + (((((((r67_group % $(8)) * $(9216)) + (r67_row[:, None] * $(192)) + r67_lane[None, :]) / $(9216)) % $(8)) * $(9216)) + ((((((r67_group % $(8)) * $(9216)) + (r67_row[:, None] * $(192)) + r67_lane[None, :]) / $(192)) % $(48)) * $(192)) + ((((r67_group % $(8)) * $(9216)) + (r67_row[:, None] * $(192)) + r67_lane[None, :]) % $(192))), dtype=tl.float32)
      r67_right = tl.load($(candidate_vto_copy_44_ptr) + (((((((r67_group % $(8)) * $(9216)) + (r67_lane[:, None] * $(48)) + r67_col[None, :]) / $(9216)) % $(8)) * $(9216)) + (((((r67_group % $(8)) * $(9216)) + (r67_lane[:, None] * $(48)) + r67_col[None, :]) % $(48)) * $(192)) + (((((r67_group % $(8)) * $(9216)) + (r67_lane[:, None] * $(48)) + r67_col[None, :]) / $(48)) % $(192))), dtype=tl.float32)
      r67_value = tl.dot(r67_left, r67_right)
      r67_value = r67_value.to(tl.float32)
      tl.store($(candidate_vbmm_9_ptr) + ((r67_group * $(2304)) + (r67_row[:, None] * $(48)) + r67_col[None, :]), r67_value)
    }
    r68_z = tl.arange($(18432))
    r68_value = (tl.load($(candidate_vbmm_9_ptr) + ((((r68_z / $(2304)) % $(8)) * $(2304)) + (((r68_z / $(48)) % $(48)) * $(48)) + (r68_z % $(48))), dtype=tl.float32) * 0.07216878364870322)
    r68_value = r68_value.to(tl.float32)
    r68_value = tl.broadcast_to(r68_value, [$(18432)])
    tl.store($(candidate_vmul_26_ptr) + r68_z, r68_value)
    r69_z = tl.arange($(48))
    r69_value = (tl.cast(r69_z, tl.int64) + tl.full([], $(48), dtype=tl.int64))
    r69_value = tl.broadcast_to(r69_value, [$(48)])
    tl.store($(candidate_vadd_13_ptr) + r69_z, r69_value)
    r70_z = tl.arange($(48))
    r70_value = (tl.load($(candidate_vadd_13_ptr) + r70_z, dtype=tl.int64) - tl.full([], $(48), dtype=tl.int64))
    r70_value = tl.broadcast_to(r70_value, [$(48)])
    tl.store($(candidate_vsub_7_ptr) + r70_z, r70_value)
    r71_z = tl.arange($(2304))
    r71_value = (tl.cast((r71_z % $(48)), tl.int64) > tl.load($(candidate_vsub_7_ptr) + ((r71_z / $(48)) % $(48)), dtype=tl.int64))
    r71_value = tl.broadcast_to(r71_value, [$(2304)])
    tl.store($(candidate_vgt_3_ptr) + r71_z, r71_value)
    r72_z = tl.arange($(18432))
    r72_value = tl.where(tl.load($(candidate_vgt_3_ptr) + ((((r72_z / $(48)) % $(48)) * $(48)) + (r72_z % $(48))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_26_ptr) + ((((r72_z / $(2304)) % $(8)) * $(2304)) + (((r72_z / $(48)) % $(48)) * $(48)) + (r72_z % $(48))), dtype=tl.float32))
    r72_value = r72_value.to(tl.float32)
    r72_value = tl.broadcast_to(r72_value, [$(18432)])
    tl.store($(candidate_vmasked_fill_3_ptr) + r72_z, r72_value)
    r73_z = tl.arange($(18432))
    r73_lane = tl.arange($(48))
    r73_row_values = tl.load($(candidate_vmasked_fill_3_ptr) + ((((r73_z[:, None] / $(2304)) % $(8)) * $(2304)) + (((r73_z[:, None] / $(48)) % $(48)) * $(48)) + (r73_lane[None, :] % $(48))), dtype=tl.float32)
    r73_maximum = tl.max(r73_row_values, axis=1, keep_dims=true)
    r73_weights = tl.exp(r73_row_values - r73_maximum)
    r73_total = tl.sum(r73_weights, axis=1)
    r73_maximum = tl.max(r73_row_values, axis=1)
    r73_value = tl.exp(tl.load($(candidate_vmasked_fill_3_ptr) + ((((r73_z / $(2304)) % $(8)) * $(2304)) + (((r73_z / $(48)) % $(48)) * $(48)) + (r73_z % $(48))), dtype=tl.float32) - r73_maximum) / r73_total
    r73_value = r73_value.to(tl.float32)
    r73_value = tl.broadcast_to(r73_value, [$(18432)])
    tl.store($(candidate_vsoftmax_3_ptr) + r73_z, r73_value)
    r74_z = tl.arange($(18432))
    r74_value = tl.load($(candidate_vsoftmax_3_ptr) + ((((r74_z / $(2304)) % $(8)) * $(2304)) + (((r74_z / $(48)) % $(48)) * $(48)) + (r74_z % $(48))), dtype=tl.float32)
    r74_value = r74_value.to(tl.bfloat16)
    r74_value = tl.broadcast_to(r74_value, [$(18432)])
    tl.store($(candidate_vto_copy_45_ptr) + r74_z, r74_value)
    r75_row = tl.arange($(48))
    r75_col = tl.arange($(128))
    r75_lane = tl.arange($(48))
    for r75_group in range($(8)) {
      r75_left = tl.load($(candidate_vto_copy_45_ptr) + (((((((r75_group % $(8)) * $(2304)) + (r75_row[:, None] * $(48)) + r75_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r75_group % $(8)) * $(2304)) + (r75_row[:, None] * $(48)) + r75_lane[None, :]) / $(48)) % $(48)) * $(48)) + ((((r75_group % $(8)) * $(2304)) + (r75_row[:, None] * $(48)) + r75_lane[None, :]) % $(48))), dtype=tl.bfloat16)
      r75_right = tl.load($(candidate_vbmm_8_ptr) + (((((((r75_group % $(8)) * $(6144)) + (r75_lane[:, None] * $(128)) + r75_col[None, :]) / $(128)) % $(48)) * $(1024)) + ((((((r75_group % $(8)) * $(6144)) + (r75_lane[:, None] * $(128)) + r75_col[None, :]) / $(6144)) % $(8)) * $(128)) + ((((r75_group % $(8)) * $(6144)) + (r75_lane[:, None] * $(128)) + r75_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r75_value = tl.dot(r75_left, r75_right)
      r75_value = r75_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_10_ptr) + ((r75_group * $(6144)) + (r75_row[:, None] * $(128)) + r75_col[None, :]), r75_value)
    }
    r76_z = tl.arange($(384))
    r76_lane = tl.arange($(48))
    r76_row_values = tl.load($(candidate_vmasked_fill_3_ptr) + ((((r76_z[:, None] / $(48)) % $(8)) * $(2304)) + ((r76_z[:, None] % $(48)) * $(48)) + (r76_lane[None, :] % $(48))), dtype=tl.float32)
    r76_maximum = tl.max(r76_row_values, axis=1, keep_dims=true)
    r76_weights = tl.exp(r76_row_values - r76_maximum)
    r76_total = tl.sum(r76_weights, axis=1)
    r76_maximum = tl.max(r76_row_values, axis=1)
    r76_value = r76_maximum + tl.log(r76_total)
    r76_value = r76_value.to(tl.float32)
    r76_value = tl.broadcast_to(r76_value, [$(384)])
    tl.store($(candidate_vlogsumexp_3_ptr) + r76_z, r76_value)
    r77_z = tl.arange($(24576))
    r77_value = tl.load($(candidate_vcat_8_ptr) + ((((r77_z / $(192)) % $(16)) * $(1536)) + (((r77_z / $(3072)) % $(8)) * $(192)) + (r77_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r77_value = r77_value.to(tl.float32)
    r77_value = tl.broadcast_to(r77_value, [$(24576)])
    tl.store($(candidate_vto_copy_46_ptr) + r77_z, r77_value)
    r78_z = tl.arange($(24576))
    r78_value = tl.load($(candidate_vcat_9_ptr) + ((((r78_z / $(192)) % $(16)) * $(1536)) + (((r78_z / $(3072)) % $(8)) * $(192)) + (r78_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r78_value = r78_value.to(tl.float32)
    r78_value = tl.broadcast_to(r78_value, [$(24576)])
    tl.store($(candidate_vto_copy_47_ptr) + r78_z, r78_value)
    r79_row = tl.arange($(16))
    r79_col = tl.arange($(16))
    r79_lane = tl.arange($(192))
    for r79_group in range($(8)) {
      r79_left = tl.load($(candidate_vto_copy_46_ptr) + (((((((r79_group % $(8)) * $(3072)) + (r79_row[:, None] * $(192)) + r79_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r79_group % $(8)) * $(3072)) + (r79_row[:, None] * $(192)) + r79_lane[None, :]) / $(192)) % $(16)) * $(192)) + ((((r79_group % $(8)) * $(3072)) + (r79_row[:, None] * $(192)) + r79_lane[None, :]) % $(192))), dtype=tl.float32)
      r79_right = tl.load($(candidate_vto_copy_47_ptr) + (((((((r79_group % $(8)) * $(3072)) + (r79_lane[:, None] * $(16)) + r79_col[None, :]) / $(3072)) % $(8)) * $(3072)) + (((((r79_group % $(8)) * $(3072)) + (r79_lane[:, None] * $(16)) + r79_col[None, :]) % $(16)) * $(192)) + (((((r79_group % $(8)) * $(3072)) + (r79_lane[:, None] * $(16)) + r79_col[None, :]) / $(16)) % $(192))), dtype=tl.float32)
      r79_value = tl.dot(r79_left, r79_right)
      r79_value = r79_value.to(tl.float32)
      tl.store($(candidate_vbmm_11_ptr) + ((r79_group * $(256)) + (r79_row[:, None] * $(16)) + r79_col[None, :]), r79_value)
    }
    r80_z = tl.arange($(2048))
    r80_value = (tl.load($(candidate_vbmm_11_ptr) + ((((r80_z / $(256)) % $(8)) * $(256)) + (((r80_z / $(16)) % $(16)) * $(16)) + (r80_z % $(16))), dtype=tl.float32) * 0.07216878364870322)
    r80_value = r80_value.to(tl.float32)
    r80_value = tl.broadcast_to(r80_value, [$(2048)])
    tl.store($(candidate_vmul_27_ptr) + r80_z, r80_value)
    r81_z = tl.arange($(16))
    r81_value = (tl.cast(r81_z, tl.int64) + tl.full([], $(16), dtype=tl.int64))
    r81_value = tl.broadcast_to(r81_value, [$(16)])
    tl.store($(candidate_vadd_14_ptr) + r81_z, r81_value)
    r82_z = tl.arange($(16))
    r82_value = (tl.load($(candidate_vadd_14_ptr) + r82_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r82_value = tl.broadcast_to(r82_value, [$(16)])
    tl.store($(candidate_vsub_8_ptr) + r82_z, r82_value)
    r83_z = tl.arange($(256))
    r83_value = (tl.cast((r83_z % $(16)), tl.int64) > tl.load($(candidate_vsub_8_ptr) + ((r83_z / $(16)) % $(16)), dtype=tl.int64))
    r83_value = tl.broadcast_to(r83_value, [$(256)])
    tl.store($(candidate_vgt_4_ptr) + r83_z, r83_value)
    r84_z = tl.arange($(2048))
    r84_value = tl.where(tl.load($(candidate_vgt_4_ptr) + ((((r84_z / $(16)) % $(16)) * $(16)) + (r84_z % $(16))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_27_ptr) + ((((r84_z / $(256)) % $(8)) * $(256)) + (((r84_z / $(16)) % $(16)) * $(16)) + (r84_z % $(16))), dtype=tl.float32))
    r84_value = r84_value.to(tl.float32)
    r84_value = tl.broadcast_to(r84_value, [$(2048)])
    tl.store($(candidate_vmasked_fill_4_ptr) + r84_z, r84_value)
    r85_z = tl.arange($(2048))
    r85_lane = tl.arange($(16))
    r85_row_values = tl.load($(candidate_vmasked_fill_4_ptr) + ((((r85_z[:, None] / $(256)) % $(8)) * $(256)) + (((r85_z[:, None] / $(16)) % $(16)) * $(16)) + (r85_lane[None, :] % $(16))), dtype=tl.float32)
    r85_maximum = tl.max(r85_row_values, axis=1, keep_dims=true)
    r85_weights = tl.exp(r85_row_values - r85_maximum)
    r85_total = tl.sum(r85_weights, axis=1)
    r85_maximum = tl.max(r85_row_values, axis=1)
    r85_value = tl.exp(tl.load($(candidate_vmasked_fill_4_ptr) + ((((r85_z / $(256)) % $(8)) * $(256)) + (((r85_z / $(16)) % $(16)) * $(16)) + (r85_z % $(16))), dtype=tl.float32) - r85_maximum) / r85_total
    r85_value = r85_value.to(tl.float32)
    r85_value = tl.broadcast_to(r85_value, [$(2048)])
    tl.store($(candidate_vsoftmax_4_ptr) + r85_z, r85_value)
    r86_z = tl.arange($(2048))
    r86_value = tl.load($(candidate_vsoftmax_4_ptr) + ((((r86_z / $(256)) % $(8)) * $(256)) + (((r86_z / $(16)) % $(16)) * $(16)) + (r86_z % $(16))), dtype=tl.float32)
    r86_value = r86_value.to(tl.bfloat16)
    r86_value = tl.broadcast_to(r86_value, [$(2048)])
    tl.store($(candidate_vto_copy_48_ptr) + r86_z, r86_value)
    r87_row = tl.arange($(16))
    r87_col = tl.arange($(128))
    r87_lane = tl.arange($(16))
    for r87_group in range($(8)) {
      r87_left = tl.load($(candidate_vto_copy_48_ptr) + (((((((r87_group % $(8)) * $(256)) + (r87_row[:, None] * $(16)) + r87_lane[None, :]) / $(256)) % $(8)) * $(256)) + ((((((r87_group % $(8)) * $(256)) + (r87_row[:, None] * $(16)) + r87_lane[None, :]) / $(16)) % $(16)) * $(16)) + ((((r87_group % $(8)) * $(256)) + (r87_row[:, None] * $(16)) + r87_lane[None, :]) % $(16))), dtype=tl.bfloat16)
      r87_right = tl.load($(candidate_vbmm_8_ptr) + (((((((r87_group % $(8)) * $(2048)) + (r87_lane[:, None] * $(128)) + r87_col[None, :]) / $(128)) % $(16)) * $(1024)) + ((((((r87_group % $(8)) * $(2048)) + (r87_lane[:, None] * $(128)) + r87_col[None, :]) / $(2048)) % $(8)) * $(128)) + ((((r87_group % $(8)) * $(2048)) + (r87_lane[:, None] * $(128)) + r87_col[None, :]) % $(128)) + $(49152)), dtype=tl.bfloat16)
      r87_value = tl.dot(r87_left, r87_right)
      r87_value = r87_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_12_ptr) + ((r87_group * $(2048)) + (r87_row[:, None] * $(128)) + r87_col[None, :]), r87_value)
    }
    r88_z = tl.arange($(128))
    r88_lane = tl.arange($(16))
    r88_row_values = tl.load($(candidate_vmasked_fill_4_ptr) + ((((r88_z[:, None] / $(16)) % $(8)) * $(256)) + ((r88_z[:, None] % $(16)) * $(16)) + (r88_lane[None, :] % $(16))), dtype=tl.float32)
    r88_maximum = tl.max(r88_row_values, axis=1, keep_dims=true)
    r88_weights = tl.exp(r88_row_values - r88_maximum)
    r88_total = tl.sum(r88_weights, axis=1)
    r88_maximum = tl.max(r88_row_values, axis=1)
    r88_value = r88_maximum + tl.log(r88_total)
    r88_value = r88_value.to(tl.float32)
    r88_value = tl.broadcast_to(r88_value, [$(128)])
    tl.store($(candidate_vlogsumexp_4_ptr) + r88_z, r88_value)
    r89_z = tl.arange($(98304))
    r89_value = tl.load($(candidate_vcat_8_ptr) + ((((r89_z / $(192)) % $(64)) * $(1536)) + (((r89_z / $(12288)) % $(8)) * $(192)) + (r89_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r89_value = r89_value.to(tl.float32)
    r89_value = tl.broadcast_to(r89_value, [$(98304)])
    tl.store($(candidate_vto_copy_49_ptr) + r89_z, r89_value)
    r90_z = tl.arange($(98304))
    r90_value = tl.load($(candidate_vcat_9_ptr) + ((((r90_z / $(192)) % $(64)) * $(1536)) + (((r90_z / $(12288)) % $(8)) * $(192)) + (r90_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r90_value = r90_value.to(tl.float32)
    r90_value = tl.broadcast_to(r90_value, [$(98304)])
    tl.store($(candidate_vto_copy_50_ptr) + r90_z, r90_value)
    r91_row = tl.arange($(64))
    r91_col = tl.arange($(64))
    r91_lane = tl.arange($(192))
    for r91_group in range($(8)) {
      r91_left = tl.load($(candidate_vto_copy_49_ptr) + (((((((r91_group % $(8)) * $(12288)) + (r91_row[:, None] * $(192)) + r91_lane[None, :]) / $(12288)) % $(8)) * $(12288)) + ((((((r91_group % $(8)) * $(12288)) + (r91_row[:, None] * $(192)) + r91_lane[None, :]) / $(192)) % $(64)) * $(192)) + ((((r91_group % $(8)) * $(12288)) + (r91_row[:, None] * $(192)) + r91_lane[None, :]) % $(192))), dtype=tl.float32)
      r91_right = tl.load($(candidate_vto_copy_50_ptr) + (((((((r91_group % $(8)) * $(12288)) + (r91_lane[:, None] * $(64)) + r91_col[None, :]) / $(12288)) % $(8)) * $(12288)) + (((((r91_group % $(8)) * $(12288)) + (r91_lane[:, None] * $(64)) + r91_col[None, :]) % $(64)) * $(192)) + (((((r91_group % $(8)) * $(12288)) + (r91_lane[:, None] * $(64)) + r91_col[None, :]) / $(64)) % $(192))), dtype=tl.float32)
      r91_value = tl.dot(r91_left, r91_right)
      r91_value = r91_value.to(tl.float32)
      tl.store($(candidate_vbmm_13_ptr) + ((r91_group * $(4096)) + (r91_row[:, None] * $(64)) + r91_col[None, :]), r91_value)
    }
    r92_z = tl.arange($(32768))
    r92_value = (tl.load($(candidate_vbmm_13_ptr) + ((((r92_z / $(4096)) % $(8)) * $(4096)) + (((r92_z / $(64)) % $(64)) * $(64)) + (r92_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r92_value = r92_value.to(tl.float32)
    r92_value = tl.broadcast_to(r92_value, [$(32768)])
    tl.store($(candidate_vmul_28_ptr) + r92_z, r92_value)
    r93_z = tl.arange($(64))
    r93_value = (tl.cast(r93_z, tl.int64) + tl.full([], $(64), dtype=tl.int64))
    r93_value = tl.broadcast_to(r93_value, [$(64)])
    tl.store($(candidate_vadd_15_ptr) + r93_z, r93_value)
    r94_z = tl.arange($(64))
    r94_value = (tl.load($(candidate_vadd_15_ptr) + r94_z, dtype=tl.int64) - tl.full([], $(64), dtype=tl.int64))
    r94_value = tl.broadcast_to(r94_value, [$(64)])
    tl.store($(candidate_vsub_9_ptr) + r94_z, r94_value)
    r95_z = tl.arange($(4096))
    r95_value = (tl.cast((r95_z % $(64)), tl.int64) > tl.load($(candidate_vsub_9_ptr) + ((r95_z / $(64)) % $(64)), dtype=tl.int64))
    r95_value = tl.broadcast_to(r95_value, [$(4096)])
    tl.store($(candidate_vgt_5_ptr) + r95_z, r95_value)
    r96_z = tl.arange($(32768))
    r96_value = tl.where(tl.load($(candidate_vgt_5_ptr) + ((((r96_z / $(64)) % $(64)) * $(64)) + (r96_z % $(64))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_28_ptr) + ((((r96_z / $(4096)) % $(8)) * $(4096)) + (((r96_z / $(64)) % $(64)) * $(64)) + (r96_z % $(64))), dtype=tl.float32))
    r96_value = r96_value.to(tl.float32)
    r96_value = tl.broadcast_to(r96_value, [$(32768)])
    tl.store($(candidate_vmasked_fill_5_ptr) + r96_z, r96_value)
    r97_z = tl.arange($(32768))
    r97_lane = tl.arange($(64))
    r97_row_values = tl.load($(candidate_vmasked_fill_5_ptr) + ((((r97_z[:, None] / $(4096)) % $(8)) * $(4096)) + (((r97_z[:, None] / $(64)) % $(64)) * $(64)) + (r97_lane[None, :] % $(64))), dtype=tl.float32)
    r97_maximum = tl.max(r97_row_values, axis=1, keep_dims=true)
    r97_weights = tl.exp(r97_row_values - r97_maximum)
    r97_total = tl.sum(r97_weights, axis=1)
    r97_maximum = tl.max(r97_row_values, axis=1)
    r97_value = tl.exp(tl.load($(candidate_vmasked_fill_5_ptr) + ((((r97_z / $(4096)) % $(8)) * $(4096)) + (((r97_z / $(64)) % $(64)) * $(64)) + (r97_z % $(64))), dtype=tl.float32) - r97_maximum) / r97_total
    r97_value = r97_value.to(tl.float32)
    r97_value = tl.broadcast_to(r97_value, [$(32768)])
    tl.store($(candidate_vsoftmax_5_ptr) + r97_z, r97_value)
    r98_z = tl.arange($(32768))
    r98_value = tl.load($(candidate_vsoftmax_5_ptr) + ((((r98_z / $(4096)) % $(8)) * $(4096)) + (((r98_z / $(64)) % $(64)) * $(64)) + (r98_z % $(64))), dtype=tl.float32)
    r98_value = r98_value.to(tl.bfloat16)
    r98_value = tl.broadcast_to(r98_value, [$(32768)])
    tl.store($(candidate_vto_copy_51_ptr) + r98_z, r98_value)
    r99_row = tl.arange($(64))
    r99_col = tl.arange($(128))
    r99_lane = tl.arange($(64))
    for r99_group in range($(8)) {
      r99_left = tl.load($(candidate_vto_copy_51_ptr) + (((((((r99_group % $(8)) * $(4096)) + (r99_row[:, None] * $(64)) + r99_lane[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r99_group % $(8)) * $(4096)) + (r99_row[:, None] * $(64)) + r99_lane[None, :]) / $(64)) % $(64)) * $(64)) + ((((r99_group % $(8)) * $(4096)) + (r99_row[:, None] * $(64)) + r99_lane[None, :]) % $(64))), dtype=tl.bfloat16)
      r99_right = tl.load($(candidate_vbmm_8_ptr) + (((((((r99_group % $(8)) * $(8192)) + (r99_lane[:, None] * $(128)) + r99_col[None, :]) / $(128)) % $(64)) * $(1024)) + ((((((r99_group % $(8)) * $(8192)) + (r99_lane[:, None] * $(128)) + r99_col[None, :]) / $(8192)) % $(8)) * $(128)) + ((((r99_group % $(8)) * $(8192)) + (r99_lane[:, None] * $(128)) + r99_col[None, :]) % $(128)) + $(65536)), dtype=tl.bfloat16)
      r99_value = tl.dot(r99_left, r99_right)
      r99_value = r99_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_14_ptr) + ((r99_group * $(8192)) + (r99_row[:, None] * $(128)) + r99_col[None, :]), r99_value)
    }
    r100_z = tl.arange($(512))
    r100_lane = tl.arange($(64))
    r100_row_values = tl.load($(candidate_vmasked_fill_5_ptr) + ((((r100_z[:, None] / $(64)) % $(8)) * $(4096)) + ((r100_z[:, None] % $(64)) * $(64)) + (r100_lane[None, :] % $(64))), dtype=tl.float32)
    r100_maximum = tl.max(r100_row_values, axis=1, keep_dims=true)
    r100_weights = tl.exp(r100_row_values - r100_maximum)
    r100_total = tl.sum(r100_weights, axis=1)
    r100_maximum = tl.max(r100_row_values, axis=1)
    r100_value = r100_maximum + tl.log(r100_total)
    r100_value = r100_value.to(tl.float32)
    r100_value = tl.broadcast_to(r100_value, [$(512)])
    tl.store($(candidate_vlogsumexp_5_ptr) + r100_z, r100_value)
    r101_z = tl.arange($(131072))
    r101_value = tl.where(((r101_z / $(1024)) % $(128)) < $(48), tl.load($(candidate_vbmm_10_ptr) + ((((((((r101_z / $(128)) % $(8)) * $(6144)) + ((((r101_z / $(1024)) % $(128)) - $(0)) * $(128)) + (r101_z % $(128))) / $(6144)) % $(8)) * $(6144)) + ((((((r101_z / $(128)) % $(8)) * $(48)) + (((r101_z / $(1024)) % $(128)) - $(0))) % $(48)) * $(128)) + (r101_z % $(128))), dtype=tl.bfloat16), tl.where(((r101_z / $(1024)) % $(128)) < $(64), tl.load($(candidate_vbmm_12_ptr) + ((((((((r101_z / $(128)) % $(8)) * $(2048)) + ((((r101_z / $(1024)) % $(128)) - $(48)) * $(128)) + (r101_z % $(128))) / $(2048)) % $(8)) * $(2048)) + ((((((r101_z / $(128)) % $(8)) * $(16)) + (((r101_z / $(1024)) % $(128)) - $(48))) % $(16)) * $(128)) + (r101_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vbmm_14_ptr) + ((((((((r101_z / $(128)) % $(8)) * $(8192)) + ((((r101_z / $(1024)) % $(128)) - $(64)) * $(128)) + (r101_z % $(128))) / $(8192)) % $(8)) * $(8192)) + ((((((r101_z / $(128)) % $(8)) * $(64)) + (((r101_z / $(1024)) % $(128)) - $(64))) % $(64)) * $(128)) + (r101_z % $(128))), dtype=tl.bfloat16)))
    r101_value = r101_value.to(tl.bfloat16)
    r101_value = tl.broadcast_to(r101_value, [$(131072)])
    tl.store($(candidate_vcat_10_ptr) + r101_z, r101_value)
    r102_z = tl.arange($(1024))
    r102_value = tl.where(((r102_z / $(8)) % $(128)) < $(48), tl.load($(candidate_vlogsumexp_3_ptr) + (((r102_z % $(8)) * $(48)) + ((((r102_z / $(8)) % $(128)) - $(0)) % $(48))), dtype=tl.float32), tl.where(((r102_z / $(8)) % $(128)) < $(64), tl.load($(candidate_vlogsumexp_4_ptr) + (((r102_z % $(8)) * $(16)) + ((((r102_z / $(8)) % $(128)) - $(48)) % $(16))), dtype=tl.float32), tl.load($(candidate_vlogsumexp_5_ptr) + (((r102_z % $(8)) * $(64)) + ((((r102_z / $(8)) % $(128)) - $(64)) % $(64))), dtype=tl.float32)))
    r102_value = r102_value.to(tl.float32)
    r102_value = tl.broadcast_to(r102_value, [$(1024)])
    tl.store($(candidate_vcat_11_ptr) + r102_z, r102_value)
    r103_z = tl.arange($(73728))
    r103_value = tl.load($(candidate_vcat_8_ptr) + ((((r103_z / $(192)) % $(48)) * $(1536)) + (((r103_z / $(9216)) % $(8)) * $(192)) + (r103_z % $(192))), dtype=tl.bfloat16)
    r103_value = r103_value.to(tl.float32)
    r103_value = tl.broadcast_to(r103_value, [$(73728)])
    tl.store($(candidate_vto_copy_52_ptr) + r103_z, r103_value)
    r104_z = tl.arange($(73728))
    r104_value = tl.load($(candidate_vcat_9_ptr) + ((((r104_z / $(192)) % $(48)) * $(1536)) + (((r104_z / $(9216)) % $(8)) * $(192)) + (r104_z % $(192))), dtype=tl.bfloat16)
    r104_value = r104_value.to(tl.float32)
    r104_value = tl.broadcast_to(r104_value, [$(73728)])
    tl.store($(candidate_vto_copy_53_ptr) + r104_z, r104_value)
    r105_z = tl.arange($(49152))
    r105_value = tl.load($(candidate_vbmm_8_ptr) + ((((r105_z / $(128)) % $(48)) * $(1024)) + (((r105_z / $(6144)) % $(8)) * $(128)) + (r105_z % $(128))), dtype=tl.bfloat16)
    r105_value = r105_value.to(tl.float32)
    r105_value = tl.broadcast_to(r105_value, [$(49152)])
    tl.store($(candidate_vto_copy_54_ptr) + r105_z, r105_value)
    r106_z = tl.arange($(49152))
    r106_value = tl.load($(candidate_vmm_10_ptr) + ((((r106_z / $(128)) % $(48)) * $(1024)) + (((r106_z / $(6144)) % $(8)) * $(128)) + (r106_z % $(128))), dtype=tl.bfloat16)
    r106_value = r106_value.to(tl.float32)
    r106_value = tl.broadcast_to(r106_value, [$(49152)])
    tl.store($(candidate_vto_copy_55_ptr) + r106_z, r106_value)
    r107_z = tl.arange($(49152))
    r107_value = tl.load($(candidate_vcat_10_ptr) + ((((r107_z / $(128)) % $(48)) * $(1024)) + (((r107_z / $(6144)) % $(8)) * $(128)) + (r107_z % $(128))), dtype=tl.bfloat16)
    r107_value = r107_value.to(tl.float32)
    r107_value = tl.broadcast_to(r107_value, [$(49152)])
    tl.store($(candidate_vto_copy_56_ptr) + r107_z, r107_value)
    r108_row = tl.arange($(48))
    r108_col = tl.arange($(48))
    r108_lane = tl.arange($(192))
    for r108_group in range($(8)) {
      r108_left = tl.load($(candidate_vto_copy_52_ptr) + (((((((r108_group % $(8)) * $(9216)) + (r108_row[:, None] * $(192)) + r108_lane[None, :]) / $(9216)) % $(8)) * $(9216)) + ((((((r108_group % $(8)) * $(9216)) + (r108_row[:, None] * $(192)) + r108_lane[None, :]) / $(192)) % $(48)) * $(192)) + ((((r108_group % $(8)) * $(9216)) + (r108_row[:, None] * $(192)) + r108_lane[None, :]) % $(192))), dtype=tl.float32)
      r108_right = tl.load($(candidate_vto_copy_53_ptr) + (((((((r108_group % $(8)) * $(9216)) + (r108_lane[:, None] * $(48)) + r108_col[None, :]) / $(9216)) % $(8)) * $(9216)) + (((((r108_group % $(8)) * $(9216)) + (r108_lane[:, None] * $(48)) + r108_col[None, :]) % $(48)) * $(192)) + (((((r108_group % $(8)) * $(9216)) + (r108_lane[:, None] * $(48)) + r108_col[None, :]) / $(48)) % $(192))), dtype=tl.float32)
      r108_value = tl.dot(r108_left, r108_right)
      r108_value = r108_value.to(tl.float32)
      tl.store($(candidate_vbmm_15_ptr) + ((r108_group * $(2304)) + (r108_row[:, None] * $(48)) + r108_col[None, :]), r108_value)
    }
    r109_z = tl.arange($(18432))
    r109_value = (tl.load($(candidate_vbmm_15_ptr) + ((((r109_z / $(2304)) % $(8)) * $(2304)) + (((r109_z / $(48)) % $(48)) * $(48)) + (r109_z % $(48))), dtype=tl.float32) * 0.07216878364870322)
    r109_value = r109_value.to(tl.float32)
    r109_value = tl.broadcast_to(r109_value, [$(18432)])
    tl.store($(candidate_vmul_29_ptr) + r109_z, r109_value)
    r110_z = tl.arange($(48))
    r110_value = (tl.cast(r110_z, tl.int64) + tl.full([], $(48), dtype=tl.int64))
    r110_value = tl.broadcast_to(r110_value, [$(48)])
    tl.store($(candidate_vadd_16_ptr) + r110_z, r110_value)
    r111_z = tl.arange($(48))
    r111_value = (tl.load($(candidate_vadd_16_ptr) + r111_z, dtype=tl.int64) - tl.full([], $(48), dtype=tl.int64))
    r111_value = tl.broadcast_to(r111_value, [$(48)])
    tl.store($(candidate_vsub_10_ptr) + r111_z, r111_value)
    r112_z = tl.arange($(2304))
    r112_value = (tl.cast((r112_z % $(48)), tl.int64) > tl.load($(candidate_vsub_10_ptr) + ((r112_z / $(48)) % $(48)), dtype=tl.int64))
    r112_value = tl.broadcast_to(r112_value, [$(2304)])
    tl.store($(candidate_vgt_6_ptr) + r112_z, r112_value)
    r113_z = tl.arange($(18432))
    r113_value = tl.where(tl.load($(candidate_vgt_6_ptr) + ((((r113_z / $(48)) % $(48)) * $(48)) + (r113_z % $(48))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_29_ptr) + ((((r113_z / $(2304)) % $(8)) * $(2304)) + (((r113_z / $(48)) % $(48)) * $(48)) + (r113_z % $(48))), dtype=tl.float32))
    r113_value = r113_value.to(tl.float32)
    r113_value = tl.broadcast_to(r113_value, [$(18432)])
    tl.store($(candidate_vmasked_fill_6_ptr) + r113_z, r113_value)
    r114_z = tl.arange($(18432))
    r114_value = (tl.load($(candidate_vmasked_fill_6_ptr) + ((((r114_z / $(2304)) % $(8)) * $(2304)) + (((r114_z / $(48)) % $(48)) * $(48)) + (r114_z % $(48))), dtype=tl.float32) - tl.load($(candidate_vcat_11_ptr) + ((((r114_z / $(48)) % $(48)) * $(8)) + ((r114_z / $(2304)) % $(8))), dtype=tl.float32))
    r114_value = r114_value.to(tl.float32)
    r114_value = tl.broadcast_to(r114_value, [$(18432)])
    tl.store($(candidate_vsub_11_ptr) + r114_z, r114_value)
    r115_z = tl.arange($(18432))
    r115_value = tl.exp(tl.load($(candidate_vsub_11_ptr) + ((((r115_z / $(2304)) % $(8)) * $(2304)) + (((r115_z / $(48)) % $(48)) * $(48)) + (r115_z % $(48))), dtype=tl.float32))
    r115_value = r115_value.to(tl.float32)
    r115_value = tl.broadcast_to(r115_value, [$(18432)])
    tl.store($(candidate_vexp_ptr) + r115_z, r115_value)
    r116_z = tl.arange($(49152))
    r116_value = (tl.load($(candidate_vto_copy_55_ptr) + ((((r116_z / $(6144)) % $(8)) * $(6144)) + (((r116_z / $(128)) % $(48)) * $(128)) + (r116_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_56_ptr) + ((((r116_z / $(6144)) % $(8)) * $(6144)) + (((r116_z / $(128)) % $(48)) * $(128)) + (r116_z % $(128))), dtype=tl.float32))
    r116_value = r116_value.to(tl.float32)
    r116_value = tl.broadcast_to(r116_value, [$(49152)])
    tl.store($(candidate_vmul_30_ptr) + r116_z, r116_value)
    r117_z = tl.arange($(384))
    r117_lane = tl.arange($(128))
    r117_value = tl.sum(tl.load($(candidate_vmul_30_ptr) + ((((r117_z[:, None] / $(48)) % $(8)) * $(6144)) + ((r117_z[:, None] % $(48)) * $(128)) + r117_lane[None, :]), dtype=tl.float32), axis=1)
    r117_value = r117_value.to(tl.float32)
    r117_value = tl.broadcast_to(r117_value, [$(384)])
    tl.store($(candidate_vsum_1_ptr) + r117_z, r117_value)
    r118_row = tl.arange($(48))
    r118_col = tl.arange($(48))
    r118_lane = tl.arange($(128))
    for r118_group in range($(8)) {
      r118_left = tl.load($(candidate_vto_copy_55_ptr) + (((((((r118_group % $(8)) * $(6144)) + (r118_row[:, None] * $(128)) + r118_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r118_group % $(8)) * $(6144)) + (r118_row[:, None] * $(128)) + r118_lane[None, :]) / $(128)) % $(48)) * $(128)) + ((((r118_group % $(8)) * $(6144)) + (r118_row[:, None] * $(128)) + r118_lane[None, :]) % $(128))), dtype=tl.float32)
      r118_right = tl.load($(candidate_vto_copy_54_ptr) + (((((((r118_group % $(8)) * $(6144)) + (r118_lane[:, None] * $(48)) + r118_col[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r118_group % $(8)) * $(6144)) + (r118_lane[:, None] * $(48)) + r118_col[None, :]) % $(48)) * $(128)) + (((((r118_group % $(8)) * $(6144)) + (r118_lane[:, None] * $(48)) + r118_col[None, :]) / $(48)) % $(128))), dtype=tl.float32)
      r118_value = tl.dot(r118_left, r118_right)
      r118_value = r118_value.to(tl.float32)
      tl.store($(candidate_vbmm_16_ptr) + ((r118_group * $(2304)) + (r118_row[:, None] * $(48)) + r118_col[None, :]), r118_value)
    }
    r119_z = tl.arange($(18432))
    r119_value = (tl.load($(candidate_vbmm_16_ptr) + ((((r119_z / $(2304)) % $(8)) * $(2304)) + (((r119_z / $(48)) % $(48)) * $(48)) + (r119_z % $(48))), dtype=tl.float32) - tl.load($(candidate_vsum_1_ptr) + ((((r119_z / $(2304)) % $(8)) * $(48)) + ((r119_z / $(48)) % $(48))), dtype=tl.float32))
    r119_value = r119_value.to(tl.float32)
    r119_value = tl.broadcast_to(r119_value, [$(18432)])
    tl.store($(candidate_vsub_12_ptr) + r119_z, r119_value)
    r120_z = tl.arange($(18432))
    r120_value = (tl.load($(candidate_vexp_ptr) + ((((r120_z / $(2304)) % $(8)) * $(2304)) + (((r120_z / $(48)) % $(48)) * $(48)) + (r120_z % $(48))), dtype=tl.float32) * tl.load($(candidate_vsub_12_ptr) + ((((r120_z / $(2304)) % $(8)) * $(2304)) + (((r120_z / $(48)) % $(48)) * $(48)) + (r120_z % $(48))), dtype=tl.float32))
    r120_value = r120_value.to(tl.float32)
    r120_value = tl.broadcast_to(r120_value, [$(18432)])
    tl.store($(candidate_vmul_31_ptr) + r120_z, r120_value)
    r121_row = tl.arange($(48))
    r121_col = tl.arange($(192))
    r121_lane = tl.arange($(48))
    for r121_group in range($(8)) {
      r121_left = tl.load($(candidate_vmul_31_ptr) + (((((((r121_group % $(8)) * $(2304)) + (r121_row[:, None] * $(48)) + r121_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r121_group % $(8)) * $(2304)) + (r121_row[:, None] * $(48)) + r121_lane[None, :]) / $(48)) % $(48)) * $(48)) + ((((r121_group % $(8)) * $(2304)) + (r121_row[:, None] * $(48)) + r121_lane[None, :]) % $(48))), dtype=tl.float32)
      r121_right = tl.load($(candidate_vto_copy_53_ptr) + (((((((r121_group % $(8)) * $(9216)) + (r121_lane[:, None] * $(192)) + r121_col[None, :]) / $(9216)) % $(8)) * $(9216)) + ((((((r121_group % $(8)) * $(9216)) + (r121_lane[:, None] * $(192)) + r121_col[None, :]) / $(192)) % $(48)) * $(192)) + ((((r121_group % $(8)) * $(9216)) + (r121_lane[:, None] * $(192)) + r121_col[None, :]) % $(192))), dtype=tl.float32)
      r121_value = tl.dot(r121_left, r121_right)
      r121_value = r121_value.to(tl.float32)
      tl.store($(candidate_vbmm_17_ptr) + ((r121_group * $(9216)) + (r121_row[:, None] * $(192)) + r121_col[None, :]), r121_value)
    }
    r122_z = tl.arange($(73728))
    r122_value = (tl.load($(candidate_vbmm_17_ptr) + ((((r122_z / $(9216)) % $(8)) * $(9216)) + (((r122_z / $(192)) % $(48)) * $(192)) + (r122_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r122_value = r122_value.to(tl.float32)
    r122_value = tl.broadcast_to(r122_value, [$(73728)])
    tl.store($(candidate_vmul_32_ptr) + r122_z, r122_value)
    r123_z = tl.arange($(73728))
    r123_value = tl.load($(candidate_vmul_32_ptr) + ((((r123_z / $(192)) % $(8)) * $(9216)) + (((r123_z / $(1536)) % $(48)) * $(192)) + (r123_z % $(192))), dtype=tl.float32)
    r123_value = r123_value.to(tl.bfloat16)
    r123_value = tl.broadcast_to(r123_value, [$(73728)])
    tl.store($(candidate_vto_copy_57_ptr) + r123_z, r123_value)
    r124_row = tl.arange($(48))
    r124_col = tl.arange($(192))
    r124_lane = tl.arange($(48))
    for r124_group in range($(8)) {
      r124_left = tl.load($(candidate_vmul_31_ptr) + (((((((r124_group % $(8)) * $(2304)) + (r124_row[:, None] * $(48)) + r124_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + (((((r124_group % $(8)) * $(2304)) + (r124_row[:, None] * $(48)) + r124_lane[None, :]) % $(48)) * $(48)) + (((((r124_group % $(8)) * $(2304)) + (r124_row[:, None] * $(48)) + r124_lane[None, :]) / $(48)) % $(48))), dtype=tl.float32)
      r124_right = tl.load($(candidate_vto_copy_52_ptr) + (((((((r124_group % $(8)) * $(9216)) + (r124_lane[:, None] * $(192)) + r124_col[None, :]) / $(9216)) % $(8)) * $(9216)) + ((((((r124_group % $(8)) * $(9216)) + (r124_lane[:, None] * $(192)) + r124_col[None, :]) / $(192)) % $(48)) * $(192)) + ((((r124_group % $(8)) * $(9216)) + (r124_lane[:, None] * $(192)) + r124_col[None, :]) % $(192))), dtype=tl.float32)
      r124_value = tl.dot(r124_left, r124_right)
      r124_value = r124_value.to(tl.float32)
      tl.store($(candidate_vbmm_18_ptr) + ((r124_group * $(9216)) + (r124_row[:, None] * $(192)) + r124_col[None, :]), r124_value)
    }
    r125_z = tl.arange($(73728))
    r125_value = (tl.load($(candidate_vbmm_18_ptr) + ((((r125_z / $(9216)) % $(8)) * $(9216)) + (((r125_z / $(192)) % $(48)) * $(192)) + (r125_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r125_value = r125_value.to(tl.float32)
    r125_value = tl.broadcast_to(r125_value, [$(73728)])
    tl.store($(candidate_vmul_33_ptr) + r125_z, r125_value)
    r126_z = tl.arange($(73728))
    r126_value = tl.load($(candidate_vmul_33_ptr) + ((((r126_z / $(192)) % $(8)) * $(9216)) + (((r126_z / $(1536)) % $(48)) * $(192)) + (r126_z % $(192))), dtype=tl.float32)
    r126_value = r126_value.to(tl.bfloat16)
    r126_value = tl.broadcast_to(r126_value, [$(73728)])
    tl.store($(candidate_vto_copy_58_ptr) + r126_z, r126_value)
    r127_row = tl.arange($(48))
    r127_col = tl.arange($(128))
    r127_lane = tl.arange($(48))
    for r127_group in range($(8)) {
      r127_left = tl.load($(candidate_vexp_ptr) + (((((((r127_group % $(8)) * $(2304)) + (r127_row[:, None] * $(48)) + r127_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + (((((r127_group % $(8)) * $(2304)) + (r127_row[:, None] * $(48)) + r127_lane[None, :]) % $(48)) * $(48)) + (((((r127_group % $(8)) * $(2304)) + (r127_row[:, None] * $(48)) + r127_lane[None, :]) / $(48)) % $(48))), dtype=tl.float32)
      r127_right = tl.load($(candidate_vto_copy_55_ptr) + (((((((r127_group % $(8)) * $(6144)) + (r127_lane[:, None] * $(128)) + r127_col[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r127_group % $(8)) * $(6144)) + (r127_lane[:, None] * $(128)) + r127_col[None, :]) / $(128)) % $(48)) * $(128)) + ((((r127_group % $(8)) * $(6144)) + (r127_lane[:, None] * $(128)) + r127_col[None, :]) % $(128))), dtype=tl.float32)
      r127_value = tl.dot(r127_left, r127_right)
      r127_value = r127_value.to(tl.float32)
      tl.store($(candidate_vbmm_19_ptr) + ((r127_group * $(6144)) + (r127_row[:, None] * $(128)) + r127_col[None, :]), r127_value)
    }
    r128_z = tl.arange($(49152))
    r128_value = tl.load($(candidate_vbmm_19_ptr) + ((((r128_z / $(128)) % $(8)) * $(6144)) + (((r128_z / $(1024)) % $(48)) * $(128)) + (r128_z % $(128))), dtype=tl.float32)
    r128_value = r128_value.to(tl.bfloat16)
    r128_value = tl.broadcast_to(r128_value, [$(49152)])
    tl.store($(candidate_vto_copy_59_ptr) + r128_z, r128_value)
    r129_z = tl.arange($(24576))
    r129_value = tl.load($(candidate_vcat_8_ptr) + ((((r129_z / $(192)) % $(16)) * $(1536)) + (((r129_z / $(3072)) % $(8)) * $(192)) + (r129_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r129_value = r129_value.to(tl.float32)
    r129_value = tl.broadcast_to(r129_value, [$(24576)])
    tl.store($(candidate_vto_copy_60_ptr) + r129_z, r129_value)
    r130_z = tl.arange($(24576))
    r130_value = tl.load($(candidate_vcat_9_ptr) + ((((r130_z / $(192)) % $(16)) * $(1536)) + (((r130_z / $(3072)) % $(8)) * $(192)) + (r130_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r130_value = r130_value.to(tl.float32)
    r130_value = tl.broadcast_to(r130_value, [$(24576)])
    tl.store($(candidate_vto_copy_61_ptr) + r130_z, r130_value)
    r131_z = tl.arange($(16384))
    r131_value = tl.load($(candidate_vbmm_8_ptr) + ((((r131_z / $(128)) % $(16)) * $(1024)) + (((r131_z / $(2048)) % $(8)) * $(128)) + (r131_z % $(128)) + $(49152)), dtype=tl.bfloat16)
    r131_value = r131_value.to(tl.float32)
    r131_value = tl.broadcast_to(r131_value, [$(16384)])
    tl.store($(candidate_vto_copy_62_ptr) + r131_z, r131_value)
    r132_z = tl.arange($(16384))
    r132_value = tl.load($(candidate_vmm_10_ptr) + ((((r132_z / $(128)) % $(16)) * $(1024)) + (((r132_z / $(2048)) % $(8)) * $(128)) + (r132_z % $(128)) + $(49152)), dtype=tl.bfloat16)
    r132_value = r132_value.to(tl.float32)
    r132_value = tl.broadcast_to(r132_value, [$(16384)])
    tl.store($(candidate_vto_copy_63_ptr) + r132_z, r132_value)
    r133_z = tl.arange($(16384))
    r133_value = tl.load($(candidate_vcat_10_ptr) + ((((r133_z / $(128)) % $(16)) * $(1024)) + (((r133_z / $(2048)) % $(8)) * $(128)) + (r133_z % $(128)) + $(49152)), dtype=tl.bfloat16)
    r133_value = r133_value.to(tl.float32)
    r133_value = tl.broadcast_to(r133_value, [$(16384)])
    tl.store($(candidate_vto_copy_64_ptr) + r133_z, r133_value)
    r134_row = tl.arange($(16))
    r134_col = tl.arange($(16))
    r134_lane = tl.arange($(192))
    for r134_group in range($(8)) {
      r134_left = tl.load($(candidate_vto_copy_60_ptr) + (((((((r134_group % $(8)) * $(3072)) + (r134_row[:, None] * $(192)) + r134_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r134_group % $(8)) * $(3072)) + (r134_row[:, None] * $(192)) + r134_lane[None, :]) / $(192)) % $(16)) * $(192)) + ((((r134_group % $(8)) * $(3072)) + (r134_row[:, None] * $(192)) + r134_lane[None, :]) % $(192))), dtype=tl.float32)
      r134_right = tl.load($(candidate_vto_copy_61_ptr) + (((((((r134_group % $(8)) * $(3072)) + (r134_lane[:, None] * $(16)) + r134_col[None, :]) / $(3072)) % $(8)) * $(3072)) + (((((r134_group % $(8)) * $(3072)) + (r134_lane[:, None] * $(16)) + r134_col[None, :]) % $(16)) * $(192)) + (((((r134_group % $(8)) * $(3072)) + (r134_lane[:, None] * $(16)) + r134_col[None, :]) / $(16)) % $(192))), dtype=tl.float32)
      r134_value = tl.dot(r134_left, r134_right)
      r134_value = r134_value.to(tl.float32)
      tl.store($(candidate_vbmm_20_ptr) + ((r134_group * $(256)) + (r134_row[:, None] * $(16)) + r134_col[None, :]), r134_value)
    }
    r135_z = tl.arange($(2048))
    r135_value = (tl.load($(candidate_vbmm_20_ptr) + ((((r135_z / $(256)) % $(8)) * $(256)) + (((r135_z / $(16)) % $(16)) * $(16)) + (r135_z % $(16))), dtype=tl.float32) * 0.07216878364870322)
    r135_value = r135_value.to(tl.float32)
    r135_value = tl.broadcast_to(r135_value, [$(2048)])
    tl.store($(candidate_vmul_34_ptr) + r135_z, r135_value)
    r136_z = tl.arange($(16))
    r136_value = (tl.cast(r136_z, tl.int64) + tl.full([], $(16), dtype=tl.int64))
    r136_value = tl.broadcast_to(r136_value, [$(16)])
    tl.store($(candidate_vadd_17_ptr) + r136_z, r136_value)
    r137_z = tl.arange($(16))
    r137_value = (tl.load($(candidate_vadd_17_ptr) + r137_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r137_value = tl.broadcast_to(r137_value, [$(16)])
    tl.store($(candidate_vsub_13_ptr) + r137_z, r137_value)
    r138_z = tl.arange($(256))
    r138_value = (tl.cast((r138_z % $(16)), tl.int64) > tl.load($(candidate_vsub_13_ptr) + ((r138_z / $(16)) % $(16)), dtype=tl.int64))
    r138_value = tl.broadcast_to(r138_value, [$(256)])
    tl.store($(candidate_vgt_7_ptr) + r138_z, r138_value)
    r139_z = tl.arange($(2048))
    r139_value = tl.where(tl.load($(candidate_vgt_7_ptr) + ((((r139_z / $(16)) % $(16)) * $(16)) + (r139_z % $(16))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_34_ptr) + ((((r139_z / $(256)) % $(8)) * $(256)) + (((r139_z / $(16)) % $(16)) * $(16)) + (r139_z % $(16))), dtype=tl.float32))
    r139_value = r139_value.to(tl.float32)
    r139_value = tl.broadcast_to(r139_value, [$(2048)])
    tl.store($(candidate_vmasked_fill_7_ptr) + r139_z, r139_value)
    r140_z = tl.arange($(2048))
    r140_value = (tl.load($(candidate_vmasked_fill_7_ptr) + ((((r140_z / $(256)) % $(8)) * $(256)) + (((r140_z / $(16)) % $(16)) * $(16)) + (r140_z % $(16))), dtype=tl.float32) - tl.load($(candidate_vcat_11_ptr) + ((((r140_z / $(16)) % $(16)) * $(8)) + ((r140_z / $(256)) % $(8)) + $(384)), dtype=tl.float32))
    r140_value = r140_value.to(tl.float32)
    r140_value = tl.broadcast_to(r140_value, [$(2048)])
    tl.store($(candidate_vsub_14_ptr) + r140_z, r140_value)
    r141_z = tl.arange($(2048))
    r141_value = tl.exp(tl.load($(candidate_vsub_14_ptr) + ((((r141_z / $(256)) % $(8)) * $(256)) + (((r141_z / $(16)) % $(16)) * $(16)) + (r141_z % $(16))), dtype=tl.float32))
    r141_value = r141_value.to(tl.float32)
    r141_value = tl.broadcast_to(r141_value, [$(2048)])
    tl.store($(candidate_vexp_1_ptr) + r141_z, r141_value)
    r142_z = tl.arange($(16384))
    r142_value = (tl.load($(candidate_vto_copy_63_ptr) + ((((r142_z / $(2048)) % $(8)) * $(2048)) + (((r142_z / $(128)) % $(16)) * $(128)) + (r142_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_64_ptr) + ((((r142_z / $(2048)) % $(8)) * $(2048)) + (((r142_z / $(128)) % $(16)) * $(128)) + (r142_z % $(128))), dtype=tl.float32))
    r142_value = r142_value.to(tl.float32)
    r142_value = tl.broadcast_to(r142_value, [$(16384)])
    tl.store($(candidate_vmul_35_ptr) + r142_z, r142_value)
    r143_z = tl.arange($(128))
    r143_lane = tl.arange($(128))
    r143_value = tl.sum(tl.load($(candidate_vmul_35_ptr) + ((((r143_z[:, None] / $(16)) % $(8)) * $(2048)) + ((r143_z[:, None] % $(16)) * $(128)) + r143_lane[None, :]), dtype=tl.float32), axis=1)
    r143_value = r143_value.to(tl.float32)
    r143_value = tl.broadcast_to(r143_value, [$(128)])
    tl.store($(candidate_vsum_2_ptr) + r143_z, r143_value)
    r144_row = tl.arange($(16))
    r144_col = tl.arange($(16))
    r144_lane = tl.arange($(128))
    for r144_group in range($(8)) {
      r144_left = tl.load($(candidate_vto_copy_63_ptr) + (((((((r144_group % $(8)) * $(2048)) + (r144_row[:, None] * $(128)) + r144_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r144_group % $(8)) * $(2048)) + (r144_row[:, None] * $(128)) + r144_lane[None, :]) / $(128)) % $(16)) * $(128)) + ((((r144_group % $(8)) * $(2048)) + (r144_row[:, None] * $(128)) + r144_lane[None, :]) % $(128))), dtype=tl.float32)
      r144_right = tl.load($(candidate_vto_copy_62_ptr) + (((((((r144_group % $(8)) * $(2048)) + (r144_lane[:, None] * $(16)) + r144_col[None, :]) / $(2048)) % $(8)) * $(2048)) + (((((r144_group % $(8)) * $(2048)) + (r144_lane[:, None] * $(16)) + r144_col[None, :]) % $(16)) * $(128)) + (((((r144_group % $(8)) * $(2048)) + (r144_lane[:, None] * $(16)) + r144_col[None, :]) / $(16)) % $(128))), dtype=tl.float32)
      r144_value = tl.dot(r144_left, r144_right)
      r144_value = r144_value.to(tl.float32)
      tl.store($(candidate_vbmm_21_ptr) + ((r144_group * $(256)) + (r144_row[:, None] * $(16)) + r144_col[None, :]), r144_value)
    }
    r145_z = tl.arange($(2048))
    r145_value = (tl.load($(candidate_vbmm_21_ptr) + ((((r145_z / $(256)) % $(8)) * $(256)) + (((r145_z / $(16)) % $(16)) * $(16)) + (r145_z % $(16))), dtype=tl.float32) - tl.load($(candidate_vsum_2_ptr) + ((((r145_z / $(256)) % $(8)) * $(16)) + ((r145_z / $(16)) % $(16))), dtype=tl.float32))
    r145_value = r145_value.to(tl.float32)
    r145_value = tl.broadcast_to(r145_value, [$(2048)])
    tl.store($(candidate_vsub_15_ptr) + r145_z, r145_value)
    r146_z = tl.arange($(2048))
    r146_value = (tl.load($(candidate_vexp_1_ptr) + ((((r146_z / $(256)) % $(8)) * $(256)) + (((r146_z / $(16)) % $(16)) * $(16)) + (r146_z % $(16))), dtype=tl.float32) * tl.load($(candidate_vsub_15_ptr) + ((((r146_z / $(256)) % $(8)) * $(256)) + (((r146_z / $(16)) % $(16)) * $(16)) + (r146_z % $(16))), dtype=tl.float32))
    r146_value = r146_value.to(tl.float32)
    r146_value = tl.broadcast_to(r146_value, [$(2048)])
    tl.store($(candidate_vmul_36_ptr) + r146_z, r146_value)
    r147_row = tl.arange($(16))
    r147_col = tl.arange($(192))
    r147_lane = tl.arange($(16))
    for r147_group in range($(8)) {
      r147_left = tl.load($(candidate_vmul_36_ptr) + (((((((r147_group % $(8)) * $(256)) + (r147_row[:, None] * $(16)) + r147_lane[None, :]) / $(256)) % $(8)) * $(256)) + ((((((r147_group % $(8)) * $(256)) + (r147_row[:, None] * $(16)) + r147_lane[None, :]) / $(16)) % $(16)) * $(16)) + ((((r147_group % $(8)) * $(256)) + (r147_row[:, None] * $(16)) + r147_lane[None, :]) % $(16))), dtype=tl.float32)
      r147_right = tl.load($(candidate_vto_copy_61_ptr) + (((((((r147_group % $(8)) * $(3072)) + (r147_lane[:, None] * $(192)) + r147_col[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r147_group % $(8)) * $(3072)) + (r147_lane[:, None] * $(192)) + r147_col[None, :]) / $(192)) % $(16)) * $(192)) + ((((r147_group % $(8)) * $(3072)) + (r147_lane[:, None] * $(192)) + r147_col[None, :]) % $(192))), dtype=tl.float32)
      r147_value = tl.dot(r147_left, r147_right)
      r147_value = r147_value.to(tl.float32)
      tl.store($(candidate_vbmm_22_ptr) + ((r147_group * $(3072)) + (r147_row[:, None] * $(192)) + r147_col[None, :]), r147_value)
    }
    r148_z = tl.arange($(24576))
    r148_value = (tl.load($(candidate_vbmm_22_ptr) + ((((r148_z / $(3072)) % $(8)) * $(3072)) + (((r148_z / $(192)) % $(16)) * $(192)) + (r148_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r148_value = r148_value.to(tl.float32)
    r148_value = tl.broadcast_to(r148_value, [$(24576)])
    tl.store($(candidate_vmul_37_ptr) + r148_z, r148_value)
    r149_z = tl.arange($(24576))
    r149_value = tl.load($(candidate_vmul_37_ptr) + ((((r149_z / $(192)) % $(8)) * $(3072)) + (((r149_z / $(1536)) % $(16)) * $(192)) + (r149_z % $(192))), dtype=tl.float32)
    r149_value = r149_value.to(tl.bfloat16)
    r149_value = tl.broadcast_to(r149_value, [$(24576)])
    tl.store($(candidate_vto_copy_65_ptr) + r149_z, r149_value)
    r150_row = tl.arange($(16))
    r150_col = tl.arange($(192))
    r150_lane = tl.arange($(16))
    for r150_group in range($(8)) {
      r150_left = tl.load($(candidate_vmul_36_ptr) + (((((((r150_group % $(8)) * $(256)) + (r150_row[:, None] * $(16)) + r150_lane[None, :]) / $(256)) % $(8)) * $(256)) + (((((r150_group % $(8)) * $(256)) + (r150_row[:, None] * $(16)) + r150_lane[None, :]) % $(16)) * $(16)) + (((((r150_group % $(8)) * $(256)) + (r150_row[:, None] * $(16)) + r150_lane[None, :]) / $(16)) % $(16))), dtype=tl.float32)
      r150_right = tl.load($(candidate_vto_copy_60_ptr) + (((((((r150_group % $(8)) * $(3072)) + (r150_lane[:, None] * $(192)) + r150_col[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r150_group % $(8)) * $(3072)) + (r150_lane[:, None] * $(192)) + r150_col[None, :]) / $(192)) % $(16)) * $(192)) + ((((r150_group % $(8)) * $(3072)) + (r150_lane[:, None] * $(192)) + r150_col[None, :]) % $(192))), dtype=tl.float32)
      r150_value = tl.dot(r150_left, r150_right)
      r150_value = r150_value.to(tl.float32)
      tl.store($(candidate_vbmm_23_ptr) + ((r150_group * $(3072)) + (r150_row[:, None] * $(192)) + r150_col[None, :]), r150_value)
    }
    r151_z = tl.arange($(24576))
    r151_value = (tl.load($(candidate_vbmm_23_ptr) + ((((r151_z / $(3072)) % $(8)) * $(3072)) + (((r151_z / $(192)) % $(16)) * $(192)) + (r151_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r151_value = r151_value.to(tl.float32)
    r151_value = tl.broadcast_to(r151_value, [$(24576)])
    tl.store($(candidate_vmul_38_ptr) + r151_z, r151_value)
    r152_z = tl.arange($(24576))
    r152_value = tl.load($(candidate_vmul_38_ptr) + ((((r152_z / $(192)) % $(8)) * $(3072)) + (((r152_z / $(1536)) % $(16)) * $(192)) + (r152_z % $(192))), dtype=tl.float32)
    r152_value = r152_value.to(tl.bfloat16)
    r152_value = tl.broadcast_to(r152_value, [$(24576)])
    tl.store($(candidate_vto_copy_66_ptr) + r152_z, r152_value)
    r153_row = tl.arange($(16))
    r153_col = tl.arange($(128))
    r153_lane = tl.arange($(16))
    for r153_group in range($(8)) {
      r153_left = tl.load($(candidate_vexp_1_ptr) + (((((((r153_group % $(8)) * $(256)) + (r153_row[:, None] * $(16)) + r153_lane[None, :]) / $(256)) % $(8)) * $(256)) + (((((r153_group % $(8)) * $(256)) + (r153_row[:, None] * $(16)) + r153_lane[None, :]) % $(16)) * $(16)) + (((((r153_group % $(8)) * $(256)) + (r153_row[:, None] * $(16)) + r153_lane[None, :]) / $(16)) % $(16))), dtype=tl.float32)
      r153_right = tl.load($(candidate_vto_copy_63_ptr) + (((((((r153_group % $(8)) * $(2048)) + (r153_lane[:, None] * $(128)) + r153_col[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r153_group % $(8)) * $(2048)) + (r153_lane[:, None] * $(128)) + r153_col[None, :]) / $(128)) % $(16)) * $(128)) + ((((r153_group % $(8)) * $(2048)) + (r153_lane[:, None] * $(128)) + r153_col[None, :]) % $(128))), dtype=tl.float32)
      r153_value = tl.dot(r153_left, r153_right)
      r153_value = r153_value.to(tl.float32)
      tl.store($(candidate_vbmm_24_ptr) + ((r153_group * $(2048)) + (r153_row[:, None] * $(128)) + r153_col[None, :]), r153_value)
    }
    r154_z = tl.arange($(16384))
    r154_value = tl.load($(candidate_vbmm_24_ptr) + ((((r154_z / $(128)) % $(8)) * $(2048)) + (((r154_z / $(1024)) % $(16)) * $(128)) + (r154_z % $(128))), dtype=tl.float32)
    r154_value = r154_value.to(tl.bfloat16)
    r154_value = tl.broadcast_to(r154_value, [$(16384)])
    tl.store($(candidate_vto_copy_67_ptr) + r154_z, r154_value)
    r155_z = tl.arange($(98304))
    r155_value = tl.load($(candidate_vcat_8_ptr) + ((((r155_z / $(192)) % $(64)) * $(1536)) + (((r155_z / $(12288)) % $(8)) * $(192)) + (r155_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r155_value = r155_value.to(tl.float32)
    r155_value = tl.broadcast_to(r155_value, [$(98304)])
    tl.store($(candidate_vto_copy_68_ptr) + r155_z, r155_value)
    r156_z = tl.arange($(98304))
    r156_value = tl.load($(candidate_vcat_9_ptr) + ((((r156_z / $(192)) % $(64)) * $(1536)) + (((r156_z / $(12288)) % $(8)) * $(192)) + (r156_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r156_value = r156_value.to(tl.float32)
    r156_value = tl.broadcast_to(r156_value, [$(98304)])
    tl.store($(candidate_vto_copy_69_ptr) + r156_z, r156_value)
    r157_z = tl.arange($(65536))
    r157_value = tl.load($(candidate_vbmm_8_ptr) + ((((r157_z / $(128)) % $(64)) * $(1024)) + (((r157_z / $(8192)) % $(8)) * $(128)) + (r157_z % $(128)) + $(65536)), dtype=tl.bfloat16)
    r157_value = r157_value.to(tl.float32)
    r157_value = tl.broadcast_to(r157_value, [$(65536)])
    tl.store($(candidate_vto_copy_70_ptr) + r157_z, r157_value)
    r158_z = tl.arange($(65536))
    r158_value = tl.load($(candidate_vmm_10_ptr) + ((((r158_z / $(128)) % $(64)) * $(1024)) + (((r158_z / $(8192)) % $(8)) * $(128)) + (r158_z % $(128)) + $(65536)), dtype=tl.bfloat16)
    r158_value = r158_value.to(tl.float32)
    r158_value = tl.broadcast_to(r158_value, [$(65536)])
    tl.store($(candidate_vto_copy_71_ptr) + r158_z, r158_value)
    r159_z = tl.arange($(65536))
    r159_value = tl.load($(candidate_vcat_10_ptr) + ((((r159_z / $(128)) % $(64)) * $(1024)) + (((r159_z / $(8192)) % $(8)) * $(128)) + (r159_z % $(128)) + $(65536)), dtype=tl.bfloat16)
    r159_value = r159_value.to(tl.float32)
    r159_value = tl.broadcast_to(r159_value, [$(65536)])
    tl.store($(candidate_vto_copy_72_ptr) + r159_z, r159_value)
    r160_row = tl.arange($(64))
    r160_col = tl.arange($(64))
    r160_lane = tl.arange($(192))
    for r160_group in range($(8)) {
      r160_left = tl.load($(candidate_vto_copy_68_ptr) + (((((((r160_group % $(8)) * $(12288)) + (r160_row[:, None] * $(192)) + r160_lane[None, :]) / $(12288)) % $(8)) * $(12288)) + ((((((r160_group % $(8)) * $(12288)) + (r160_row[:, None] * $(192)) + r160_lane[None, :]) / $(192)) % $(64)) * $(192)) + ((((r160_group % $(8)) * $(12288)) + (r160_row[:, None] * $(192)) + r160_lane[None, :]) % $(192))), dtype=tl.float32)
      r160_right = tl.load($(candidate_vto_copy_69_ptr) + (((((((r160_group % $(8)) * $(12288)) + (r160_lane[:, None] * $(64)) + r160_col[None, :]) / $(12288)) % $(8)) * $(12288)) + (((((r160_group % $(8)) * $(12288)) + (r160_lane[:, None] * $(64)) + r160_col[None, :]) % $(64)) * $(192)) + (((((r160_group % $(8)) * $(12288)) + (r160_lane[:, None] * $(64)) + r160_col[None, :]) / $(64)) % $(192))), dtype=tl.float32)
      r160_value = tl.dot(r160_left, r160_right)
      r160_value = r160_value.to(tl.float32)
      tl.store($(candidate_vbmm_25_ptr) + ((r160_group * $(4096)) + (r160_row[:, None] * $(64)) + r160_col[None, :]), r160_value)
    }
    r161_z = tl.arange($(32768))
    r161_value = (tl.load($(candidate_vbmm_25_ptr) + ((((r161_z / $(4096)) % $(8)) * $(4096)) + (((r161_z / $(64)) % $(64)) * $(64)) + (r161_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r161_value = r161_value.to(tl.float32)
    r161_value = tl.broadcast_to(r161_value, [$(32768)])
    tl.store($(candidate_vmul_39_ptr) + r161_z, r161_value)
    r162_z = tl.arange($(64))
    r162_value = (tl.cast(r162_z, tl.int64) + tl.full([], $(64), dtype=tl.int64))
    r162_value = tl.broadcast_to(r162_value, [$(64)])
    tl.store($(candidate_vadd_18_ptr) + r162_z, r162_value)
    r163_z = tl.arange($(64))
    r163_value = (tl.load($(candidate_vadd_18_ptr) + r163_z, dtype=tl.int64) - tl.full([], $(64), dtype=tl.int64))
    r163_value = tl.broadcast_to(r163_value, [$(64)])
    tl.store($(candidate_vsub_16_ptr) + r163_z, r163_value)
    r164_z = tl.arange($(4096))
    r164_value = (tl.cast((r164_z % $(64)), tl.int64) > tl.load($(candidate_vsub_16_ptr) + ((r164_z / $(64)) % $(64)), dtype=tl.int64))
    r164_value = tl.broadcast_to(r164_value, [$(4096)])
    tl.store($(candidate_vgt_8_ptr) + r164_z, r164_value)
    r165_z = tl.arange($(32768))
    r165_value = tl.where(tl.load($(candidate_vgt_8_ptr) + ((((r165_z / $(64)) % $(64)) * $(64)) + (r165_z % $(64))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_39_ptr) + ((((r165_z / $(4096)) % $(8)) * $(4096)) + (((r165_z / $(64)) % $(64)) * $(64)) + (r165_z % $(64))), dtype=tl.float32))
    r165_value = r165_value.to(tl.float32)
    r165_value = tl.broadcast_to(r165_value, [$(32768)])
    tl.store($(candidate_vmasked_fill_8_ptr) + r165_z, r165_value)
    r166_z = tl.arange($(32768))
    r166_value = (tl.load($(candidate_vmasked_fill_8_ptr) + ((((r166_z / $(4096)) % $(8)) * $(4096)) + (((r166_z / $(64)) % $(64)) * $(64)) + (r166_z % $(64))), dtype=tl.float32) - tl.load($(candidate_vcat_11_ptr) + ((((r166_z / $(64)) % $(64)) * $(8)) + ((r166_z / $(4096)) % $(8)) + $(512)), dtype=tl.float32))
    r166_value = r166_value.to(tl.float32)
    r166_value = tl.broadcast_to(r166_value, [$(32768)])
    tl.store($(candidate_vsub_17_ptr) + r166_z, r166_value)
    r167_z = tl.arange($(32768))
    r167_value = tl.exp(tl.load($(candidate_vsub_17_ptr) + ((((r167_z / $(4096)) % $(8)) * $(4096)) + (((r167_z / $(64)) % $(64)) * $(64)) + (r167_z % $(64))), dtype=tl.float32))
    r167_value = r167_value.to(tl.float32)
    r167_value = tl.broadcast_to(r167_value, [$(32768)])
    tl.store($(candidate_vexp_2_ptr) + r167_z, r167_value)
    r168_z = tl.arange($(65536))
    r168_value = (tl.load($(candidate_vto_copy_71_ptr) + ((((r168_z / $(8192)) % $(8)) * $(8192)) + (((r168_z / $(128)) % $(64)) * $(128)) + (r168_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_72_ptr) + ((((r168_z / $(8192)) % $(8)) * $(8192)) + (((r168_z / $(128)) % $(64)) * $(128)) + (r168_z % $(128))), dtype=tl.float32))
    r168_value = r168_value.to(tl.float32)
    r168_value = tl.broadcast_to(r168_value, [$(65536)])
    tl.store($(candidate_vmul_40_ptr) + r168_z, r168_value)
    r169_z = tl.arange($(512))
    r169_lane = tl.arange($(128))
    r169_value = tl.sum(tl.load($(candidate_vmul_40_ptr) + ((((r169_z[:, None] / $(64)) % $(8)) * $(8192)) + ((r169_z[:, None] % $(64)) * $(128)) + r169_lane[None, :]), dtype=tl.float32), axis=1)
    r169_value = r169_value.to(tl.float32)
    r169_value = tl.broadcast_to(r169_value, [$(512)])
    tl.store($(candidate_vsum_3_ptr) + r169_z, r169_value)
    r170_row = tl.arange($(64))
    r170_col = tl.arange($(64))
    r170_lane = tl.arange($(128))
    for r170_group in range($(8)) {
      r170_left = tl.load($(candidate_vto_copy_71_ptr) + (((((((r170_group % $(8)) * $(8192)) + (r170_row[:, None] * $(128)) + r170_lane[None, :]) / $(8192)) % $(8)) * $(8192)) + ((((((r170_group % $(8)) * $(8192)) + (r170_row[:, None] * $(128)) + r170_lane[None, :]) / $(128)) % $(64)) * $(128)) + ((((r170_group % $(8)) * $(8192)) + (r170_row[:, None] * $(128)) + r170_lane[None, :]) % $(128))), dtype=tl.float32)
      r170_right = tl.load($(candidate_vto_copy_70_ptr) + (((((((r170_group % $(8)) * $(8192)) + (r170_lane[:, None] * $(64)) + r170_col[None, :]) / $(8192)) % $(8)) * $(8192)) + (((((r170_group % $(8)) * $(8192)) + (r170_lane[:, None] * $(64)) + r170_col[None, :]) % $(64)) * $(128)) + (((((r170_group % $(8)) * $(8192)) + (r170_lane[:, None] * $(64)) + r170_col[None, :]) / $(64)) % $(128))), dtype=tl.float32)
      r170_value = tl.dot(r170_left, r170_right)
      r170_value = r170_value.to(tl.float32)
      tl.store($(candidate_vbmm_26_ptr) + ((r170_group * $(4096)) + (r170_row[:, None] * $(64)) + r170_col[None, :]), r170_value)
    }
    r171_z = tl.arange($(32768))
    r171_value = (tl.load($(candidate_vbmm_26_ptr) + ((((r171_z / $(4096)) % $(8)) * $(4096)) + (((r171_z / $(64)) % $(64)) * $(64)) + (r171_z % $(64))), dtype=tl.float32) - tl.load($(candidate_vsum_3_ptr) + ((((r171_z / $(4096)) % $(8)) * $(64)) + ((r171_z / $(64)) % $(64))), dtype=tl.float32))
    r171_value = r171_value.to(tl.float32)
    r171_value = tl.broadcast_to(r171_value, [$(32768)])
    tl.store($(candidate_vsub_18_ptr) + r171_z, r171_value)
    r172_z = tl.arange($(32768))
    r172_value = (tl.load($(candidate_vexp_2_ptr) + ((((r172_z / $(4096)) % $(8)) * $(4096)) + (((r172_z / $(64)) % $(64)) * $(64)) + (r172_z % $(64))), dtype=tl.float32) * tl.load($(candidate_vsub_18_ptr) + ((((r172_z / $(4096)) % $(8)) * $(4096)) + (((r172_z / $(64)) % $(64)) * $(64)) + (r172_z % $(64))), dtype=tl.float32))
    r172_value = r172_value.to(tl.float32)
    r172_value = tl.broadcast_to(r172_value, [$(32768)])
    tl.store($(candidate_vmul_41_ptr) + r172_z, r172_value)
    r173_row = tl.arange($(64))
    r173_col = tl.arange($(192))
    r173_lane = tl.arange($(64))
    for r173_group in range($(8)) {
      r173_left = tl.load($(candidate_vmul_41_ptr) + (((((((r173_group % $(8)) * $(4096)) + (r173_row[:, None] * $(64)) + r173_lane[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r173_group % $(8)) * $(4096)) + (r173_row[:, None] * $(64)) + r173_lane[None, :]) / $(64)) % $(64)) * $(64)) + ((((r173_group % $(8)) * $(4096)) + (r173_row[:, None] * $(64)) + r173_lane[None, :]) % $(64))), dtype=tl.float32)
      r173_right = tl.load($(candidate_vto_copy_69_ptr) + (((((((r173_group % $(8)) * $(12288)) + (r173_lane[:, None] * $(192)) + r173_col[None, :]) / $(12288)) % $(8)) * $(12288)) + ((((((r173_group % $(8)) * $(12288)) + (r173_lane[:, None] * $(192)) + r173_col[None, :]) / $(192)) % $(64)) * $(192)) + ((((r173_group % $(8)) * $(12288)) + (r173_lane[:, None] * $(192)) + r173_col[None, :]) % $(192))), dtype=tl.float32)
      r173_value = tl.dot(r173_left, r173_right)
      r173_value = r173_value.to(tl.float32)
      tl.store($(candidate_vbmm_27_ptr) + ((r173_group * $(12288)) + (r173_row[:, None] * $(192)) + r173_col[None, :]), r173_value)
    }
    r174_z = tl.arange($(98304))
    r174_value = (tl.load($(candidate_vbmm_27_ptr) + ((((r174_z / $(12288)) % $(8)) * $(12288)) + (((r174_z / $(192)) % $(64)) * $(192)) + (r174_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r174_value = r174_value.to(tl.float32)
    r174_value = tl.broadcast_to(r174_value, [$(98304)])
    tl.store($(candidate_vmul_42_ptr) + r174_z, r174_value)
    r175_z = tl.arange($(98304))
    r175_value = tl.load($(candidate_vmul_42_ptr) + ((((r175_z / $(192)) % $(8)) * $(12288)) + (((r175_z / $(1536)) % $(64)) * $(192)) + (r175_z % $(192))), dtype=tl.float32)
    r175_value = r175_value.to(tl.bfloat16)
    r175_value = tl.broadcast_to(r175_value, [$(98304)])
    tl.store($(candidate_vto_copy_73_ptr) + r175_z, r175_value)
    r176_row = tl.arange($(64))
    r176_col = tl.arange($(192))
    r176_lane = tl.arange($(64))
    for r176_group in range($(8)) {
      r176_left = tl.load($(candidate_vmul_41_ptr) + (((((((r176_group % $(8)) * $(4096)) + (r176_row[:, None] * $(64)) + r176_lane[None, :]) / $(4096)) % $(8)) * $(4096)) + (((((r176_group % $(8)) * $(4096)) + (r176_row[:, None] * $(64)) + r176_lane[None, :]) % $(64)) * $(64)) + (((((r176_group % $(8)) * $(4096)) + (r176_row[:, None] * $(64)) + r176_lane[None, :]) / $(64)) % $(64))), dtype=tl.float32)
      r176_right = tl.load($(candidate_vto_copy_68_ptr) + (((((((r176_group % $(8)) * $(12288)) + (r176_lane[:, None] * $(192)) + r176_col[None, :]) / $(12288)) % $(8)) * $(12288)) + ((((((r176_group % $(8)) * $(12288)) + (r176_lane[:, None] * $(192)) + r176_col[None, :]) / $(192)) % $(64)) * $(192)) + ((((r176_group % $(8)) * $(12288)) + (r176_lane[:, None] * $(192)) + r176_col[None, :]) % $(192))), dtype=tl.float32)
      r176_value = tl.dot(r176_left, r176_right)
      r176_value = r176_value.to(tl.float32)
      tl.store($(candidate_vbmm_28_ptr) + ((r176_group * $(12288)) + (r176_row[:, None] * $(192)) + r176_col[None, :]), r176_value)
    }
    r177_z = tl.arange($(98304))
    r177_value = (tl.load($(candidate_vbmm_28_ptr) + ((((r177_z / $(12288)) % $(8)) * $(12288)) + (((r177_z / $(192)) % $(64)) * $(192)) + (r177_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r177_value = r177_value.to(tl.float32)
    r177_value = tl.broadcast_to(r177_value, [$(98304)])
    tl.store($(candidate_vmul_43_ptr) + r177_z, r177_value)
    r178_z = tl.arange($(98304))
    r178_value = tl.load($(candidate_vmul_43_ptr) + ((((r178_z / $(192)) % $(8)) * $(12288)) + (((r178_z / $(1536)) % $(64)) * $(192)) + (r178_z % $(192))), dtype=tl.float32)
    r178_value = r178_value.to(tl.bfloat16)
    r178_value = tl.broadcast_to(r178_value, [$(98304)])
    tl.store($(candidate_vto_copy_74_ptr) + r178_z, r178_value)
    r179_row = tl.arange($(64))
    r179_col = tl.arange($(128))
    r179_lane = tl.arange($(64))
    for r179_group in range($(8)) {
      r179_left = tl.load($(candidate_vexp_2_ptr) + (((((((r179_group % $(8)) * $(4096)) + (r179_row[:, None] * $(64)) + r179_lane[None, :]) / $(4096)) % $(8)) * $(4096)) + (((((r179_group % $(8)) * $(4096)) + (r179_row[:, None] * $(64)) + r179_lane[None, :]) % $(64)) * $(64)) + (((((r179_group % $(8)) * $(4096)) + (r179_row[:, None] * $(64)) + r179_lane[None, :]) / $(64)) % $(64))), dtype=tl.float32)
      r179_right = tl.load($(candidate_vto_copy_71_ptr) + (((((((r179_group % $(8)) * $(8192)) + (r179_lane[:, None] * $(128)) + r179_col[None, :]) / $(8192)) % $(8)) * $(8192)) + ((((((r179_group % $(8)) * $(8192)) + (r179_lane[:, None] * $(128)) + r179_col[None, :]) / $(128)) % $(64)) * $(128)) + ((((r179_group % $(8)) * $(8192)) + (r179_lane[:, None] * $(128)) + r179_col[None, :]) % $(128))), dtype=tl.float32)
      r179_value = tl.dot(r179_left, r179_right)
      r179_value = r179_value.to(tl.float32)
      tl.store($(candidate_vbmm_29_ptr) + ((r179_group * $(8192)) + (r179_row[:, None] * $(128)) + r179_col[None, :]), r179_value)
    }
    r180_z = tl.arange($(65536))
    r180_value = tl.load($(candidate_vbmm_29_ptr) + ((((r180_z / $(128)) % $(8)) * $(8192)) + (((r180_z / $(1024)) % $(64)) * $(128)) + (r180_z % $(128))), dtype=tl.float32)
    r180_value = r180_value.to(tl.bfloat16)
    r180_value = tl.broadcast_to(r180_value, [$(65536)])
    tl.store($(candidate_vto_copy_75_ptr) + r180_z, r180_value)
    r181_z = tl.arange($(196608))
    r181_value = tl.where(((r181_z / $(1536)) % $(128)) < $(48), tl.load($(candidate_vto_copy_57_ptr) + ((((((r181_z / $(1536)) % $(128)) - $(0)) % $(48)) * $(1536)) + (((r181_z / $(192)) % $(8)) * $(192)) + (r181_z % $(192))), dtype=tl.bfloat16), tl.where(((r181_z / $(1536)) % $(128)) < $(64), tl.load($(candidate_vto_copy_65_ptr) + ((((((r181_z / $(1536)) % $(128)) - $(48)) % $(16)) * $(1536)) + (((r181_z / $(192)) % $(8)) * $(192)) + (r181_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vto_copy_73_ptr) + ((((((r181_z / $(1536)) % $(128)) - $(64)) % $(64)) * $(1536)) + (((r181_z / $(192)) % $(8)) * $(192)) + (r181_z % $(192))), dtype=tl.bfloat16)))
    r181_value = r181_value.to(tl.bfloat16)
    r181_value = tl.broadcast_to(r181_value, [$(196608)])
    tl.store($(candidate_vcat_12_ptr) + r181_z, r181_value)
    r182_z = tl.arange($(196608))
    r182_value = tl.where(((r182_z / $(1536)) % $(128)) < $(48), tl.load($(candidate_vto_copy_58_ptr) + ((((((r182_z / $(1536)) % $(128)) - $(0)) % $(48)) * $(1536)) + (((r182_z / $(192)) % $(8)) * $(192)) + (r182_z % $(192))), dtype=tl.bfloat16), tl.where(((r182_z / $(1536)) % $(128)) < $(64), tl.load($(candidate_vto_copy_66_ptr) + ((((((r182_z / $(1536)) % $(128)) - $(48)) % $(16)) * $(1536)) + (((r182_z / $(192)) % $(8)) * $(192)) + (r182_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vto_copy_74_ptr) + ((((((r182_z / $(1536)) % $(128)) - $(64)) % $(64)) * $(1536)) + (((r182_z / $(192)) % $(8)) * $(192)) + (r182_z % $(192))), dtype=tl.bfloat16)))
    r182_value = r182_value.to(tl.bfloat16)
    r182_value = tl.broadcast_to(r182_value, [$(196608)])
    tl.store($(candidate_vcat_13_ptr) + r182_z, r182_value)
    r183_z = tl.arange($(131072))
    r183_value = tl.where(((r183_z / $(1024)) % $(128)) < $(48), tl.load($(candidate_vto_copy_59_ptr) + ((((((r183_z / $(1024)) % $(128)) - $(0)) % $(48)) * $(1024)) + (((r183_z / $(128)) % $(8)) * $(128)) + (r183_z % $(128))), dtype=tl.bfloat16), tl.where(((r183_z / $(1024)) % $(128)) < $(64), tl.load($(candidate_vto_copy_67_ptr) + ((((((r183_z / $(1024)) % $(128)) - $(48)) % $(16)) * $(1024)) + (((r183_z / $(128)) % $(8)) * $(128)) + (r183_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vto_copy_75_ptr) + ((((((r183_z / $(1024)) % $(128)) - $(64)) % $(64)) * $(1024)) + (((r183_z / $(128)) % $(8)) * $(128)) + (r183_z % $(128))), dtype=tl.bfloat16)))
    r183_value = r183_value.to(tl.bfloat16)
    r183_value = tl.broadcast_to(r183_value, [$(131072)])
    tl.store($(candidate_vcat_14_ptr) + r183_z, r183_value)
    r184_z = tl.arange($(8192))
    r184_lane = tl.arange($(8))
    r184_value = tl.sum(tl.load($(candidate_vcat_13_ptr) + ((((r184_z[:, None] / $(64)) % $(128)) * $(1536)) + (r184_lane[None, :] * $(192)) + (r184_z[:, None] % $(64)) + $(128)), dtype=tl.bfloat16), axis=1)
    r184_value = r184_value.to(tl.bfloat16)
    r184_value = tl.broadcast_to(r184_value, [$(8192)])
    tl.store($(candidate_vsum_4_ptr) + r184_z, r184_value)
    r185_row = tl.arange($(128))
    r185_col = tl.arange($(512))
    r185_lane = tl.arange($(1024))
    for r185_group in range($(1)) {
      r185_left = tl.load($(candidate_vcat_13_ptr) + ((((((r185_row[:, None] * $(1024)) + r185_lane[None, :]) / $(1024)) % $(128)) * $(1536)) + ((((r185_row[:, None] * $(1024)) + r185_lane[None, :]) % $(8)) * $(192)) + ((((r185_row[:, None] * $(1024)) + r185_lane[None, :]) / $(8)) % $(128))), dtype=tl.bfloat16)
      r185_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r185_lane[:, None] * $(512)) + r185_col[None, :]) / $(512)) % $(8)) * $(131072)) + (((((r185_lane[:, None] * $(512)) + r185_col[None, :]) / $(4096)) % $(128)) * $(512)) + (((r185_lane[:, None] * $(512)) + r185_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r185_value = tl.dot(r185_left, r185_right)
      r185_value = r185_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_30_ptr) + ((r185_group * $(65536)) + (r185_row[:, None] * $(512)) + r185_col[None, :]), r185_value)
    }
    r186_row = tl.arange($(128))
    r186_col = tl.arange($(512))
    r186_lane = tl.arange($(1024))
    for r186_group in range($(1)) {
      r186_left = tl.load($(candidate_vcat_14_ptr) + ((((((r186_row[:, None] * $(1024)) + r186_lane[None, :]) / $(1024)) % $(128)) * $(1024)) + ((((r186_row[:, None] * $(1024)) + r186_lane[None, :]) % $(8)) * $(128)) + ((((r186_row[:, None] * $(1024)) + r186_lane[None, :]) / $(8)) % $(128))), dtype=tl.bfloat16)
      r186_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r186_lane[:, None] * $(512)) + r186_col[None, :]) / $(512)) % $(8)) * $(131072)) + (((((r186_lane[:, None] * $(512)) + r186_col[None, :]) / $(4096)) % $(128)) * $(512)) + (((r186_lane[:, None] * $(512)) + r186_col[None, :]) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r186_value = tl.dot(r186_left, r186_right)
      r186_value = r186_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_31_ptr) + ((r186_group * $(65536)) + (r186_row[:, None] * $(512)) + r186_col[None, :]), r186_value)
    }
    r187_z = tl.arange($(65536))
    r187_value = (tl.load($(candidate_vbmm_30_ptr) + ((((r187_z / $(512)) % $(128)) * $(512)) + (r187_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vbmm_31_ptr) + ((((r187_z / $(512)) % $(128)) * $(512)) + (r187_z % $(512))), dtype=tl.bfloat16))
    r187_value = r187_value.to(tl.bfloat16)
    r187_value = tl.broadcast_to(r187_value, [$(65536)])
    tl.store($(candidate_vadd_19_ptr) + r187_z, r187_value)
    r188_z = tl.arange($(8192))
    r188_value = (tl.load($(candidate_vsum_4_ptr) + ((((r188_z / $(64)) % $(128)) * $(64)) + (r188_z % $(64))), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_27_ptr) + ((((r188_z / $(64)) % $(128)) * $(64)) + (r188_z % $(64))), dtype=tl.bfloat16))
    r188_value = r188_value.to(tl.bfloat16)
    r188_value = tl.broadcast_to(r188_value, [$(8192)])
    tl.store($(candidate_vmul_44_ptr) + r188_z, r188_value)
    r189_z = tl.arange($(8192))
    r189_value = (tl.load($(candidate_vsum_4_ptr) + ((((r189_z / $(64)) % $(128)) * $(64)) + (r189_z % $(64))), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_26_ptr) + ((((r189_z / $(64)) % $(128)) * $(64)) + (r189_z % $(64))), dtype=tl.bfloat16))
    r189_value = r189_value.to(tl.bfloat16)
    r189_value = tl.broadcast_to(r189_value, [$(8192)])
    tl.store($(candidate_vmul_45_ptr) + r189_z, r189_value)
    r190_z = tl.arange($(8192))
    r190_value = tl.where(((r190_z % $(2)) == $(0)), tl.load($(candidate_vmul_44_ptr) + ((((r190_z / $(64)) % $(128)) * $(64)) + (((r190_z / $(2)) % $(32)) * $(2)) + $(1)), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r190_value = r190_value.to(tl.bfloat16)
    r190_value = tl.broadcast_to(r190_value, [$(8192)])
    tl.store($(candidate_vselect_backward_ptr) + r190_z, r190_value)
    r191_z = tl.arange($(4096))
    r191_value = -tl.load($(candidate_vmul_44_ptr) + ((((r191_z / $(32)) % $(128)) * $(64)) + ((r191_z % $(32)) * $(2))), dtype=tl.bfloat16)
    r191_value = r191_value.to(tl.bfloat16)
    r191_value = tl.broadcast_to(r191_value, [$(4096)])
    tl.store($(candidate_vneg_2_ptr) + r191_z, r191_value)
    r192_z = tl.arange($(8192))
    r192_value = tl.where(((r192_z % $(2)) == $(1)), tl.load($(candidate_vneg_2_ptr) + ((((r192_z / $(64)) % $(128)) * $(32)) + ((r192_z / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r192_value = r192_value.to(tl.bfloat16)
    r192_value = tl.broadcast_to(r192_value, [$(8192)])
    tl.store($(candidate_vselect_backward_1_ptr) + r192_z, r192_value)
    r193_z = tl.arange($(8192))
    r193_value = (tl.load($(candidate_vselect_backward_ptr) + ((((r193_z / $(64)) % $(128)) * $(64)) + (((r193_z / $(2)) % $(32)) * $(2)) + (r193_z % $(2))), dtype=tl.bfloat16) + tl.load($(candidate_vselect_backward_1_ptr) + ((((r193_z / $(64)) % $(128)) * $(64)) + (((r193_z / $(2)) % $(32)) * $(2)) + (r193_z % $(2))), dtype=tl.bfloat16))
    r193_value = r193_value.to(tl.bfloat16)
    r193_value = tl.broadcast_to(r193_value, [$(8192)])
    tl.store($(candidate_vadd_20_ptr) + r193_z, r193_value)
    r194_z = tl.arange($(8192))
    r194_value = (tl.load($(candidate_vmul_45_ptr) + ((((r194_z / $(64)) % $(128)) * $(64)) + (r194_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vadd_20_ptr) + ((((r194_z / $(64)) % $(128)) * $(64)) + (((((((r194_z / $(64)) % $(128)) * $(64)) + (r194_z % $(64))) / $(2)) % $(32)) * $(2)) + (((((r194_z / $(64)) % $(128)) * $(64)) + (r194_z % $(64))) % $(2))), dtype=tl.bfloat16))
    r194_value = r194_value.to(tl.bfloat16)
    r194_value = tl.broadcast_to(r194_value, [$(8192)])
    tl.store($(candidate_vadd_21_ptr) + r194_z, r194_value)
    r195_z = tl.arange($(65536))
    r195_value = (tl.load($(candidate_vcat_12_ptr) + ((((r195_z / $(512)) % $(128)) * $(1536)) + (((r195_z / $(64)) % $(8)) * $(192)) + (r195_z % $(64)) + $(128)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_25_ptr) + ((((r195_z / $(512)) % $(128)) * $(64)) + (r195_z % $(64))), dtype=tl.bfloat16))
    r195_value = r195_value.to(tl.bfloat16)
    r195_value = tl.broadcast_to(r195_value, [$(65536)])
    tl.store($(candidate_vmul_46_ptr) + r195_z, r195_value)
    r196_z = tl.arange($(65536))
    r196_value = (tl.load($(candidate_vcat_12_ptr) + ((((r196_z / $(512)) % $(128)) * $(1536)) + (((r196_z / $(64)) % $(8)) * $(192)) + (r196_z % $(64)) + $(128)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_24_ptr) + ((((r196_z / $(512)) % $(128)) * $(64)) + (r196_z % $(64))), dtype=tl.bfloat16))
    r196_value = r196_value.to(tl.bfloat16)
    r196_value = tl.broadcast_to(r196_value, [$(65536)])
    tl.store($(candidate_vmul_47_ptr) + r196_z, r196_value)
    r197_z = tl.arange($(65536))
    r197_value = tl.where(((r197_z % $(2)) == $(0)), tl.load($(candidate_vmul_46_ptr) + ((((r197_z / $(512)) % $(128)) * $(512)) + (((r197_z / $(64)) % $(8)) * $(64)) + (((r197_z / $(2)) % $(32)) * $(2)) + $(1)), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r197_value = r197_value.to(tl.bfloat16)
    r197_value = tl.broadcast_to(r197_value, [$(65536)])
    tl.store($(candidate_vselect_backward_2_ptr) + r197_z, r197_value)
    r198_z = tl.arange($(32768))
    r198_value = -tl.load($(candidate_vmul_46_ptr) + ((((r198_z / $(256)) % $(128)) * $(512)) + (((r198_z / $(32)) % $(8)) * $(64)) + ((r198_z % $(32)) * $(2))), dtype=tl.bfloat16)
    r198_value = r198_value.to(tl.bfloat16)
    r198_value = tl.broadcast_to(r198_value, [$(32768)])
    tl.store($(candidate_vneg_3_ptr) + r198_z, r198_value)
    r199_z = tl.arange($(65536))
    r199_value = tl.where(((r199_z % $(2)) == $(1)), tl.load($(candidate_vneg_3_ptr) + ((((r199_z / $(512)) % $(128)) * $(256)) + (((r199_z / $(64)) % $(8)) * $(32)) + ((r199_z / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r199_value = r199_value.to(tl.bfloat16)
    r199_value = tl.broadcast_to(r199_value, [$(65536)])
    tl.store($(candidate_vselect_backward_3_ptr) + r199_z, r199_value)
    r200_z = tl.arange($(65536))
    r200_value = (tl.load($(candidate_vselect_backward_2_ptr) + ((((r200_z / $(512)) % $(128)) * $(512)) + (((r200_z / $(64)) % $(8)) * $(64)) + (((r200_z / $(2)) % $(32)) * $(2)) + (r200_z % $(2))), dtype=tl.bfloat16) + tl.load($(candidate_vselect_backward_3_ptr) + ((((r200_z / $(512)) % $(128)) * $(512)) + (((r200_z / $(64)) % $(8)) * $(64)) + (((r200_z / $(2)) % $(32)) * $(2)) + (r200_z % $(2))), dtype=tl.bfloat16))
    r200_value = r200_value.to(tl.bfloat16)
    r200_value = tl.broadcast_to(r200_value, [$(65536)])
    tl.store($(candidate_vadd_22_ptr) + r200_z, r200_value)
    r201_z = tl.arange($(65536))
    r201_value = (tl.load($(candidate_vmul_47_ptr) + ((((r201_z / $(512)) % $(128)) * $(512)) + (((r201_z / $(64)) % $(8)) * $(64)) + (r201_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vadd_22_ptr) + ((((r201_z / $(512)) % $(128)) * $(512)) + (((r201_z / $(64)) % $(8)) * $(64)) + (((((((r201_z / $(512)) % $(128)) * $(512)) + (((r201_z / $(64)) % $(8)) * $(64)) + (r201_z % $(64))) / $(2)) % $(32)) * $(2)) + (((((r201_z / $(512)) % $(128)) * $(512)) + (((r201_z / $(64)) % $(8)) * $(64)) + (r201_z % $(64))) % $(2))), dtype=tl.bfloat16))
    r201_value = r201_value.to(tl.bfloat16)
    r201_value = tl.broadcast_to(r201_value, [$(65536)])
    tl.store($(candidate_vadd_23_ptr) + r201_z, r201_value)
    r202_z = tl.arange($(196608))
    r202_value = tl.where((r202_z % $(192)) < $(128), tl.load($(candidate_vcat_12_ptr) + ((((((((r202_z / $(192)) % $(128)) * $(1024)) + (((r202_z / $(24576)) % $(8)) * $(128)) + ((r202_z % $(192)) - $(0))) / $(1024)) % $(128)) * $(1536)) + (((((((r202_z / $(192)) % $(128)) * $(1024)) + (((r202_z / $(24576)) % $(8)) * $(128)) + ((r202_z % $(192)) - $(0))) / $(128)) % $(8)) * $(192)) + (((((r202_z / $(192)) % $(128)) * $(1024)) + (((r202_z / $(24576)) % $(8)) * $(128)) + ((r202_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vadd_23_ptr) + ((((((((r202_z / $(192)) % $(128)) * $(512)) + (((r202_z / $(24576)) % $(8)) * $(64)) + ((r202_z % $(192)) - $(128))) / $(512)) % $(128)) * $(512)) + (((((((r202_z / $(192)) % $(128)) * $(512)) + (((r202_z / $(24576)) % $(8)) * $(64)) + ((r202_z % $(192)) - $(128))) / $(64)) % $(8)) * $(64)) + (((((r202_z / $(192)) % $(128)) * $(512)) + (((r202_z / $(24576)) % $(8)) * $(64)) + ((r202_z % $(192)) - $(128))) % $(64))), dtype=tl.bfloat16))
    r202_value = r202_value.to(tl.bfloat16)
    r202_value = tl.broadcast_to(r202_value, [$(196608)])
    tl.store($(candidate_vcat_16_ptr) + r202_z, r202_value)
    r203_z = tl.arange($(65536))
    r203_value = tl.load($(candidate_vadd_19_ptr) + ((((r203_z / $(512)) % $(128)) * $(512)) + (r203_z % $(512))), dtype=tl.bfloat16)
    r203_value = r203_value.to(tl.float32)
    r203_value = tl.broadcast_to(r203_value, [$(65536)])
    tl.store($(candidate_vto_copy_76_ptr) + r203_z, r203_value)
    r204_z = tl.arange($(65536))
    r204_value = (tl.load($(candidate_vto_copy_76_ptr) + ((((r204_z / $(512)) % $(128)) * $(512)) + (r204_z % $(512))), dtype=tl.float32) * tl.cast(tl.load($(candidate_vto_copy_4_ptr) + (r204_z % $(512)), dtype=tl.bfloat16), tl.float32))
    r204_value = r204_value.to(tl.float32)
    r204_value = tl.broadcast_to(r204_value, [$(65536)])
    tl.store($(candidate_vmul_48_ptr) + r204_z, r204_value)
    r205_z = tl.arange($(65536))
    r205_value = (tl.load($(candidate_vmul_48_ptr) + ((((r205_z / $(512)) % $(128)) * $(512)) + (r205_z % $(512))), dtype=tl.float32) * tl.load($(candidate_vto_copy_22_ptr) + ((((r205_z / $(512)) % $(128)) * $(512)) + (r205_z % $(512))), dtype=tl.float32))
    r205_value = r205_value.to(tl.float32)
    r205_value = tl.broadcast_to(r205_value, [$(65536)])
    tl.store($(candidate_vmul_49_ptr) + r205_z, r205_value)
    r206_z = tl.arange($(65536))
    r206_value = (tl.load($(candidate_vmul_48_ptr) + ((((r206_z / $(512)) % $(128)) * $(512)) + (r206_z % $(512))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_1_ptr) + ((r206_z / $(512)) % $(128)), dtype=tl.float32))
    r206_value = r206_value.to(tl.float32)
    r206_value = tl.broadcast_to(r206_value, [$(65536)])
    tl.store($(candidate_vmul_50_ptr) + r206_z, r206_value)
    r207_z = tl.arange($(128))
    r207_lane = tl.arange($(512))
    r207_value = tl.sum(tl.load($(candidate_vmul_49_ptr) + ((r207_z[:, None] * $(512)) + r207_lane[None, :]), dtype=tl.float32), axis=1)
    r207_value = r207_value.to(tl.float32)
    r207_value = tl.broadcast_to(r207_value, [$(128)])
    tl.store($(candidate_vsum_5_ptr) + r207_z, r207_value)
    r208_z = tl.arange($(128))
    r208_value = ((tl.load($(candidate_vrsqrt_1_ptr) + r208_z, dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_1_ptr) + r208_z, dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_1_ptr) + r208_z, dtype=tl.float32)))
    r208_value = r208_value.to(tl.float32)
    r208_value = tl.broadcast_to(r208_value, [$(128)])
    tl.store($(candidate_vpow_6_ptr) + r208_z, r208_value)
    r209_z = tl.arange($(128))
    r209_value = (tl.load($(candidate_vsum_5_ptr) + r209_z, dtype=tl.float32) * -0.5)
    r209_value = r209_value.to(tl.float32)
    r209_value = tl.broadcast_to(r209_value, [$(128)])
    tl.store($(candidate_vmul_51_ptr) + r209_z, r209_value)
    r210_z = tl.arange($(128))
    r210_value = (tl.load($(candidate_vmul_51_ptr) + r210_z, dtype=tl.float32) * tl.load($(candidate_vpow_6_ptr) + r210_z, dtype=tl.float32))
    r210_value = r210_value.to(tl.float32)
    r210_value = tl.broadcast_to(r210_value, [$(128)])
    tl.store($(candidate_vmul_52_ptr) + r210_z, r210_value)
    r211_z = tl.arange($(65536))
    r211_value = (tl.load($(candidate_vmul_52_ptr) + ((r211_z / $(512)) % $(128)), dtype=tl.float32) / 512.0)
    r211_value = r211_value.to(tl.float32)
    r211_value = tl.broadcast_to(r211_value, [$(65536)])
    tl.store($(candidate_vdiv_3_ptr) + r211_z, r211_value)
    r212_z = tl.arange($(65536))
    r212_value = ((tl.load($(candidate_vto_copy_22_ptr) + ((((r212_z / $(512)) % $(128)) * $(512)) + (r212_z % $(512))), dtype=tl.float32)))
    r212_value = r212_value.to(tl.float32)
    r212_value = tl.broadcast_to(r212_value, [$(65536)])
    tl.store($(candidate_vpow_7_ptr) + r212_z, r212_value)
    r213_z = tl.arange($(65536))
    r213_value = (tl.load($(candidate_vpow_7_ptr) + ((((r213_z / $(512)) % $(128)) * $(512)) + (r213_z % $(512))), dtype=tl.float32) * 2.0)
    r213_value = r213_value.to(tl.float32)
    r213_value = tl.broadcast_to(r213_value, [$(65536)])
    tl.store($(candidate_vmul_53_ptr) + r213_z, r213_value)
    r214_z = tl.arange($(65536))
    r214_value = (tl.load($(candidate_vdiv_3_ptr) + ((((r214_z / $(512)) % $(128)) * $(512)) + (r214_z % $(512))), dtype=tl.float32) * tl.load($(candidate_vmul_53_ptr) + ((((r214_z / $(512)) % $(128)) * $(512)) + (r214_z % $(512))), dtype=tl.float32))
    r214_value = r214_value.to(tl.float32)
    r214_value = tl.broadcast_to(r214_value, [$(65536)])
    tl.store($(candidate_vmul_54_ptr) + r214_z, r214_value)
    r215_z = tl.arange($(65536))
    r215_value = (tl.load($(candidate_vmul_50_ptr) + ((((r215_z / $(512)) % $(128)) * $(512)) + (r215_z % $(512))), dtype=tl.float32) + tl.load($(candidate_vmul_54_ptr) + ((((r215_z / $(512)) % $(128)) * $(512)) + (r215_z % $(512))), dtype=tl.float32))
    r215_value = r215_value.to(tl.float32)
    r215_value = tl.broadcast_to(r215_value, [$(65536)])
    tl.store($(candidate_vadd_24_ptr) + r215_z, r215_value)
    r216_z = tl.arange($(65536))
    r216_value = tl.load($(candidate_vadd_24_ptr) + ((((r216_z / $(512)) % $(128)) * $(512)) + (r216_z % $(512))), dtype=tl.float32)
    r216_value = r216_value.to(tl.bfloat16)
    r216_value = tl.broadcast_to(r216_value, [$(65536)])
    tl.store($(candidate_vto_copy_77_ptr) + r216_z, r216_value)
    r217_z = tl.arange($(73728))
    r217_value = tl.where((r217_z % $(576)) < $(512), tl.load($(candidate_vto_copy_77_ptr) + ((((r217_z / $(576)) % $(128)) * $(512)) + (((r217_z % $(576)) - $(0)) % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vadd_21_ptr) + ((((((((r217_z / $(576)) % $(128)) * $(64)) + ((r217_z % $(576)) - $(512))) / $(64)) % $(128)) * $(64)) + (((((r217_z / $(576)) % $(128)) * $(64)) + ((r217_z % $(576)) - $(512))) % $(64))), dtype=tl.bfloat16))
    r217_value = r217_value.to(tl.bfloat16)
    r217_value = tl.broadcast_to(r217_value, [$(73728)])
    tl.store($(candidate_vcat_17_ptr) + r217_z, r217_value)
    r218_row = tl.arange($(128))
    r218_col = tl.arange($(512))
    r218_lane = tl.arange($(576))
    r218_left = tl.load($(candidate_vcat_17_ptr) + ((((((r218_row[:, None] * $(576)) + r218_lane[None, :]) / $(576)) % $(128)) * $(576)) + (((r218_row[:, None] * $(576)) + r218_lane[None, :]) % $(576))), dtype=tl.bfloat16)
    r218_right = tl.load($(candidate_vto_copy_3_ptr) + (((r218_lane[:, None] % $(576)) * $(512)) + (r218_col[None, :] % $(512))), dtype=tl.bfloat16)
    r218_value = tl.dot(r218_left, r218_right)
    r218_value = r218_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_11_ptr) + ((r218_row[:, None] * $(512)) + r218_col[None, :]), r218_value)
    r219_row = tl.arange($(128))
    r219_col = tl.arange($(256))
    r219_lane = tl.arange($(1536))
    r219_left = tl.load($(candidate_vcat_16_ptr) + ((((((((((r219_row[:, None] * $(1536)) + r219_lane[None, :]) / $(1536)) % $(128)) * $(1536)) + (((r219_row[:, None] * $(1536)) + r219_lane[None, :]) % $(1536))) / $(192)) % $(8)) * $(24576)) + (((((r219_row[:, None] * $(1536)) + r219_lane[None, :]) / $(1536)) % $(128)) * $(192)) + (((((((r219_row[:, None] * $(1536)) + r219_lane[None, :]) / $(1536)) % $(128)) * $(1536)) + (((r219_row[:, None] * $(1536)) + r219_lane[None, :]) % $(1536))) % $(192))), dtype=tl.bfloat16)
    r219_right = tl.load($(candidate_vto_copy_2_ptr) + (((r219_lane[:, None] % $(1536)) * $(256)) + (r219_col[None, :] % $(256))), dtype=tl.bfloat16)
    r219_value = tl.dot(r219_left, r219_right)
    r219_value = r219_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_12_ptr) + ((r219_row[:, None] * $(256)) + r219_col[None, :]), r219_value)
    r220_z = tl.arange($(32768))
    r220_value = tl.load($(candidate_vmm_12_ptr) + ((((r220_z / $(256)) % $(128)) * $(256)) + (r220_z % $(256))), dtype=tl.bfloat16)
    r220_value = r220_value.to(tl.float32)
    r220_value = tl.broadcast_to(r220_value, [$(32768)])
    tl.store($(candidate_vto_copy_78_ptr) + r220_z, r220_value)
    r221_z = tl.arange($(32768))
    r221_value = (tl.load($(candidate_vto_copy_78_ptr) + ((((r221_z / $(256)) % $(128)) * $(256)) + (r221_z % $(256))), dtype=tl.float32) * tl.cast(tl.load($(candidate_vto_copy_1_ptr) + (r221_z % $(256)), dtype=tl.bfloat16), tl.float32))
    r221_value = r221_value.to(tl.float32)
    r221_value = tl.broadcast_to(r221_value, [$(32768)])
    tl.store($(candidate_vmul_55_ptr) + r221_z, r221_value)
    r222_z = tl.arange($(32768))
    r222_value = (tl.load($(candidate_vmul_55_ptr) + ((((r222_z / $(256)) % $(128)) * $(256)) + (r222_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vto_copy_20_ptr) + ((((r222_z / $(256)) % $(128)) * $(256)) + (r222_z % $(256))), dtype=tl.float32))
    r222_value = r222_value.to(tl.float32)
    r222_value = tl.broadcast_to(r222_value, [$(32768)])
    tl.store($(candidate_vmul_56_ptr) + r222_z, r222_value)
    r223_z = tl.arange($(32768))
    r223_value = (tl.load($(candidate_vmul_55_ptr) + ((((r223_z / $(256)) % $(128)) * $(256)) + (r223_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_ptr) + ((r223_z / $(256)) % $(128)), dtype=tl.float32))
    r223_value = r223_value.to(tl.float32)
    r223_value = tl.broadcast_to(r223_value, [$(32768)])
    tl.store($(candidate_vmul_57_ptr) + r223_z, r223_value)
    r224_z = tl.arange($(128))
    r224_lane = tl.arange($(256))
    r224_value = tl.sum(tl.load($(candidate_vmul_56_ptr) + ((r224_z[:, None] * $(256)) + r224_lane[None, :]), dtype=tl.float32), axis=1)
    r224_value = r224_value.to(tl.float32)
    r224_value = tl.broadcast_to(r224_value, [$(128)])
    tl.store($(candidate_vsum_6_ptr) + r224_z, r224_value)
    r225_z = tl.arange($(128))
    r225_value = ((tl.load($(candidate_vrsqrt_ptr) + r225_z, dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_ptr) + r225_z, dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_ptr) + r225_z, dtype=tl.float32)))
    r225_value = r225_value.to(tl.float32)
    r225_value = tl.broadcast_to(r225_value, [$(128)])
    tl.store($(candidate_vpow_8_ptr) + r225_z, r225_value)
    r226_z = tl.arange($(128))
    r226_value = (tl.load($(candidate_vsum_6_ptr) + r226_z, dtype=tl.float32) * -0.5)
    r226_value = r226_value.to(tl.float32)
    r226_value = tl.broadcast_to(r226_value, [$(128)])
    tl.store($(candidate_vmul_58_ptr) + r226_z, r226_value)
    r227_z = tl.arange($(128))
    r227_value = (tl.load($(candidate_vmul_58_ptr) + r227_z, dtype=tl.float32) * tl.load($(candidate_vpow_8_ptr) + r227_z, dtype=tl.float32))
    r227_value = r227_value.to(tl.float32)
    r227_value = tl.broadcast_to(r227_value, [$(128)])
    tl.store($(candidate_vmul_59_ptr) + r227_z, r227_value)
    r228_z = tl.arange($(32768))
    r228_value = (tl.load($(candidate_vmul_59_ptr) + ((r228_z / $(256)) % $(128)), dtype=tl.float32) / 256.0)
    r228_value = r228_value.to(tl.float32)
    r228_value = tl.broadcast_to(r228_value, [$(32768)])
    tl.store($(candidate_vdiv_4_ptr) + r228_z, r228_value)
    r229_z = tl.arange($(32768))
    r229_value = ((tl.load($(candidate_vto_copy_20_ptr) + ((((r229_z / $(256)) % $(128)) * $(256)) + (r229_z % $(256))), dtype=tl.float32)))
    r229_value = r229_value.to(tl.float32)
    r229_value = tl.broadcast_to(r229_value, [$(32768)])
    tl.store($(candidate_vpow_9_ptr) + r229_z, r229_value)
    r230_z = tl.arange($(32768))
    r230_value = (tl.load($(candidate_vpow_9_ptr) + ((((r230_z / $(256)) % $(128)) * $(256)) + (r230_z % $(256))), dtype=tl.float32) * 2.0)
    r230_value = r230_value.to(tl.float32)
    r230_value = tl.broadcast_to(r230_value, [$(32768)])
    tl.store($(candidate_vmul_60_ptr) + r230_z, r230_value)
    r231_z = tl.arange($(32768))
    r231_value = (tl.load($(candidate_vdiv_4_ptr) + ((((r231_z / $(256)) % $(128)) * $(256)) + (r231_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vmul_60_ptr) + ((((r231_z / $(256)) % $(128)) * $(256)) + (r231_z % $(256))), dtype=tl.float32))
    r231_value = r231_value.to(tl.float32)
    r231_value = tl.broadcast_to(r231_value, [$(32768)])
    tl.store($(candidate_vmul_61_ptr) + r231_z, r231_value)
    r232_z = tl.arange($(32768))
    r232_value = (tl.load($(candidate_vmul_57_ptr) + ((((r232_z / $(256)) % $(128)) * $(256)) + (r232_z % $(256))), dtype=tl.float32) + tl.load($(candidate_vmul_61_ptr) + ((((r232_z / $(256)) % $(128)) * $(256)) + (r232_z % $(256))), dtype=tl.float32))
    r232_value = r232_value.to(tl.float32)
    r232_value = tl.broadcast_to(r232_value, [$(32768)])
    tl.store($(candidate_vadd_25_ptr) + r232_z, r232_value)
    r233_z = tl.arange($(32768))
    r233_value = tl.load($(candidate_vadd_25_ptr) + ((((r233_z / $(256)) % $(128)) * $(256)) + (r233_z % $(256))), dtype=tl.float32)
    r233_value = r233_value.to(tl.bfloat16)
    r233_value = tl.broadcast_to(r233_value, [$(32768)])
    tl.store($(candidate_vto_copy_79_ptr) + r233_z, r233_value)
    r234_row = tl.arange($(128))
    r234_col = tl.arange($(512))
    r234_lane = tl.arange($(256))
    r234_left = tl.load($(candidate_vto_copy_79_ptr) + ((((((r234_row[:, None] * $(256)) + r234_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r234_row[:, None] * $(256)) + r234_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r234_right = tl.load($(candidate_vto_copy_ptr) + (((r234_lane[:, None] % $(256)) * $(512)) + (r234_col[None, :] % $(512))), dtype=tl.bfloat16)
    r234_value = tl.dot(r234_left, r234_right)
    r234_value = r234_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_13_ptr) + ((r234_row[:, None] * $(512)) + r234_col[None, :]), r234_value)
    r235_z = tl.arange($(65536))
    r235_value = (tl.load($(candidate_vmm_11_ptr) + ((((r235_z / $(512)) % $(128)) * $(512)) + (r235_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vmm_13_ptr) + ((((r235_z / $(512)) % $(128)) * $(512)) + (r235_z % $(512))), dtype=tl.bfloat16))
    r235_value = r235_value.to(tl.bfloat16)
    r235_value = tl.broadcast_to(r235_value, [$(65536)])
    tl.store($(candidate_vadd_26_ptr) + r235_z, r235_value)
    z = tl.arange($(65536))
    value = tl.load($(candidate_vadd_26_ptr) + ((((z / $(512)) % $(128)) * $(512)) + (z % $(512))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(65536)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_mla_fa4_te_rope_thd_absorb_dx
