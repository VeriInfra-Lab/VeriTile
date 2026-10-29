import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_mla_fa4_te_rope_cp_absorb_dx
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
    tl.store($(vsub_10_ptr) + r70_z, r70_value)
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
  let vmul_19_ptr : RegionName := "vmul_19"
  let vmul_20_ptr : RegionName := "vmul_20"
  let vadd_5_ptr : RegionName := "vadd_5"
  let vcat_5_ptr : RegionName := "vcat_5"
  let vmul_21_ptr : RegionName := "vmul_21"
  let vmul_22_ptr : RegionName := "vmul_22"
  let vadd_6_ptr : RegionName := "vadd_6"
  let vadd_7_ptr : RegionName := "vadd_7"
  let vto_copy_24_ptr : RegionName := "vto_copy_24"
  let vto_copy_25_ptr : RegionName := "vto_copy_25"
  let vto_copy_26_ptr : RegionName := "vto_copy_26"
  let vto_copy_27_ptr : RegionName := "vto_copy_27"
  let vsum_1_ptr : RegionName := "vsum_1"
  let vneg_ptr : RegionName := "vneg"
  let vneg_1_ptr : RegionName := "vneg_1"
  let vslice_backward_ptr : RegionName := "vslice_backward"
  let vslice_backward_1_ptr : RegionName := "vslice_backward_1"
  let vslice_backward_2_ptr : RegionName := "vslice_backward_2"
  let vslice_backward_3_ptr : RegionName := "vslice_backward_3"
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
  let candidate_vindex_put_ptr : RegionName := "candidate_vindex_put"
  let candidate_vmm_28_ptr : RegionName := "candidate_vmm_28"
  let candidate_vbmm_22_ptr : RegionName := "candidate_vbmm_22"
  let candidate_vbmm_23_ptr : RegionName := "candidate_vbmm_23"
  let candidate_vcat_11_ptr : RegionName := "candidate_vcat_11"
  let candidate_vcat_12_ptr : RegionName := "candidate_vcat_12"
  let candidate_vto_copy_89_ptr : RegionName := "candidate_vto_copy_89"
  let candidate_vto_copy_90_ptr : RegionName := "candidate_vto_copy_90"
  let candidate_vbmm_24_ptr : RegionName := "candidate_vbmm_24"
  let candidate_vmul_35_ptr : RegionName := "candidate_vmul_35"
  let candidate_vadd_32_ptr : RegionName := "candidate_vadd_32"
  let candidate_vsub_21_ptr : RegionName := "candidate_vsub_21"
  let candidate_vgt_12_ptr : RegionName := "candidate_vgt_12"
  let candidate_vmasked_fill_12_ptr : RegionName := "candidate_vmasked_fill_12"
  let candidate_vsoftmax_12_ptr : RegionName := "candidate_vsoftmax_12"
  let candidate_vto_copy_91_ptr : RegionName := "candidate_vto_copy_91"
  let candidate_vbmm_25_ptr : RegionName := "candidate_vbmm_25"
  let candidate_vlogsumexp_12_ptr : RegionName := "candidate_vlogsumexp_12"
  let candidate_vto_copy_92_ptr : RegionName := "candidate_vto_copy_92"
  let candidate_vto_copy_93_ptr : RegionName := "candidate_vto_copy_93"
  let candidate_vbmm_26_ptr : RegionName := "candidate_vbmm_26"
  let candidate_vmul_36_ptr : RegionName := "candidate_vmul_36"
  let candidate_vadd_33_ptr : RegionName := "candidate_vadd_33"
  let candidate_vsub_22_ptr : RegionName := "candidate_vsub_22"
  let candidate_vgt_13_ptr : RegionName := "candidate_vgt_13"
  let candidate_vmasked_fill_13_ptr : RegionName := "candidate_vmasked_fill_13"
  let candidate_vsoftmax_13_ptr : RegionName := "candidate_vsoftmax_13"
  let candidate_vto_copy_94_ptr : RegionName := "candidate_vto_copy_94"
  let candidate_vbmm_27_ptr : RegionName := "candidate_vbmm_27"
  let candidate_vlogsumexp_13_ptr : RegionName := "candidate_vlogsumexp_13"
  let candidate_vto_copy_95_ptr : RegionName := "candidate_vto_copy_95"
  let candidate_vto_copy_96_ptr : RegionName := "candidate_vto_copy_96"
  let candidate_vbmm_28_ptr : RegionName := "candidate_vbmm_28"
  let candidate_vmul_37_ptr : RegionName := "candidate_vmul_37"
  let candidate_vadd_34_ptr : RegionName := "candidate_vadd_34"
  let candidate_vsub_23_ptr : RegionName := "candidate_vsub_23"
  let candidate_vgt_14_ptr : RegionName := "candidate_vgt_14"
  let candidate_vmasked_fill_14_ptr : RegionName := "candidate_vmasked_fill_14"
  let candidate_vsoftmax_14_ptr : RegionName := "candidate_vsoftmax_14"
  let candidate_vto_copy_97_ptr : RegionName := "candidate_vto_copy_97"
  let candidate_vbmm_29_ptr : RegionName := "candidate_vbmm_29"
  let candidate_vlogsumexp_14_ptr : RegionName := "candidate_vlogsumexp_14"
  let candidate_vcat_13_ptr : RegionName := "candidate_vcat_13"
  let candidate_vcat_14_ptr : RegionName := "candidate_vcat_14"
  let candidate_vto_copy_98_ptr : RegionName := "candidate_vto_copy_98"
  let candidate_vto_copy_99_ptr : RegionName := "candidate_vto_copy_99"
  let candidate_vto_copy_100_ptr : RegionName := "candidate_vto_copy_100"
  let candidate_vto_copy_101_ptr : RegionName := "candidate_vto_copy_101"
  let candidate_vto_copy_102_ptr : RegionName := "candidate_vto_copy_102"
  let candidate_vbmm_30_ptr : RegionName := "candidate_vbmm_30"
  let candidate_vmul_38_ptr : RegionName := "candidate_vmul_38"
  let candidate_vadd_35_ptr : RegionName := "candidate_vadd_35"
  let candidate_vsub_24_ptr : RegionName := "candidate_vsub_24"
  let candidate_vgt_15_ptr : RegionName := "candidate_vgt_15"
  let candidate_vmasked_fill_15_ptr : RegionName := "candidate_vmasked_fill_15"
  let candidate_vsub_25_ptr : RegionName := "candidate_vsub_25"
  let candidate_vexp_ptr : RegionName := "candidate_vexp"
  let candidate_vmul_39_ptr : RegionName := "candidate_vmul_39"
  let candidate_vsum_1_ptr : RegionName := "candidate_vsum_1"
  let candidate_vbmm_31_ptr : RegionName := "candidate_vbmm_31"
  let candidate_vsub_26_ptr : RegionName := "candidate_vsub_26"
  let candidate_vmul_40_ptr : RegionName := "candidate_vmul_40"
  let candidate_vbmm_32_ptr : RegionName := "candidate_vbmm_32"
  let candidate_vmul_41_ptr : RegionName := "candidate_vmul_41"
  let candidate_vto_copy_103_ptr : RegionName := "candidate_vto_copy_103"
  let candidate_vbmm_33_ptr : RegionName := "candidate_vbmm_33"
  let candidate_vmul_42_ptr : RegionName := "candidate_vmul_42"
  let candidate_vto_copy_104_ptr : RegionName := "candidate_vto_copy_104"
  let candidate_vbmm_34_ptr : RegionName := "candidate_vbmm_34"
  let candidate_vto_copy_105_ptr : RegionName := "candidate_vto_copy_105"
  let candidate_vcat_15_ptr : RegionName := "candidate_vcat_15"
  let candidate_vcat_16_ptr : RegionName := "candidate_vcat_16"
  let candidate_vto_copy_106_ptr : RegionName := "candidate_vto_copy_106"
  let candidate_vto_copy_107_ptr : RegionName := "candidate_vto_copy_107"
  let candidate_vto_copy_108_ptr : RegionName := "candidate_vto_copy_108"
  let candidate_vto_copy_109_ptr : RegionName := "candidate_vto_copy_109"
  let candidate_vto_copy_110_ptr : RegionName := "candidate_vto_copy_110"
  let candidate_vbmm_35_ptr : RegionName := "candidate_vbmm_35"
  let candidate_vmul_43_ptr : RegionName := "candidate_vmul_43"
  let candidate_vadd_36_ptr : RegionName := "candidate_vadd_36"
  let candidate_vsub_27_ptr : RegionName := "candidate_vsub_27"
  let candidate_vgt_16_ptr : RegionName := "candidate_vgt_16"
  let candidate_vmasked_fill_16_ptr : RegionName := "candidate_vmasked_fill_16"
  let candidate_vsub_28_ptr : RegionName := "candidate_vsub_28"
  let candidate_vexp_1_ptr : RegionName := "candidate_vexp_1"
  let candidate_vmul_44_ptr : RegionName := "candidate_vmul_44"
  let candidate_vsum_2_ptr : RegionName := "candidate_vsum_2"
  let candidate_vbmm_36_ptr : RegionName := "candidate_vbmm_36"
  let candidate_vsub_29_ptr : RegionName := "candidate_vsub_29"
  let candidate_vmul_45_ptr : RegionName := "candidate_vmul_45"
  let candidate_vbmm_37_ptr : RegionName := "candidate_vbmm_37"
  let candidate_vmul_46_ptr : RegionName := "candidate_vmul_46"
  let candidate_vto_copy_111_ptr : RegionName := "candidate_vto_copy_111"
  let candidate_vbmm_38_ptr : RegionName := "candidate_vbmm_38"
  let candidate_vmul_47_ptr : RegionName := "candidate_vmul_47"
  let candidate_vto_copy_112_ptr : RegionName := "candidate_vto_copy_112"
  let candidate_vbmm_39_ptr : RegionName := "candidate_vbmm_39"
  let candidate_vto_copy_113_ptr : RegionName := "candidate_vto_copy_113"
  let candidate_vcat_17_ptr : RegionName := "candidate_vcat_17"
  let candidate_vcat_18_ptr : RegionName := "candidate_vcat_18"
  let candidate_vto_copy_114_ptr : RegionName := "candidate_vto_copy_114"
  let candidate_vto_copy_115_ptr : RegionName := "candidate_vto_copy_115"
  let candidate_vto_copy_116_ptr : RegionName := "candidate_vto_copy_116"
  let candidate_vto_copy_117_ptr : RegionName := "candidate_vto_copy_117"
  let candidate_vto_copy_118_ptr : RegionName := "candidate_vto_copy_118"
  let candidate_vbmm_40_ptr : RegionName := "candidate_vbmm_40"
  let candidate_vmul_48_ptr : RegionName := "candidate_vmul_48"
  let candidate_vadd_37_ptr : RegionName := "candidate_vadd_37"
  let candidate_vsub_30_ptr : RegionName := "candidate_vsub_30"
  let candidate_vgt_17_ptr : RegionName := "candidate_vgt_17"
  let candidate_vmasked_fill_17_ptr : RegionName := "candidate_vmasked_fill_17"
  let candidate_vsub_31_ptr : RegionName := "candidate_vsub_31"
  let candidate_vexp_2_ptr : RegionName := "candidate_vexp_2"
  let candidate_vmul_49_ptr : RegionName := "candidate_vmul_49"
  let candidate_vsum_3_ptr : RegionName := "candidate_vsum_3"
  let candidate_vbmm_41_ptr : RegionName := "candidate_vbmm_41"
  let candidate_vsub_32_ptr : RegionName := "candidate_vsub_32"
  let candidate_vmul_50_ptr : RegionName := "candidate_vmul_50"
  let candidate_vbmm_42_ptr : RegionName := "candidate_vbmm_42"
  let candidate_vmul_51_ptr : RegionName := "candidate_vmul_51"
  let candidate_vto_copy_119_ptr : RegionName := "candidate_vto_copy_119"
  let candidate_vbmm_43_ptr : RegionName := "candidate_vbmm_43"
  let candidate_vmul_52_ptr : RegionName := "candidate_vmul_52"
  let candidate_vto_copy_120_ptr : RegionName := "candidate_vto_copy_120"
  let candidate_vbmm_44_ptr : RegionName := "candidate_vbmm_44"
  let candidate_vto_copy_121_ptr : RegionName := "candidate_vto_copy_121"
  let candidate_vcat_19_ptr : RegionName := "candidate_vcat_19"
  let candidate_vcat_20_ptr : RegionName := "candidate_vcat_20"
  let candidate_vcat_21_ptr : RegionName := "candidate_vcat_21"
  let candidate_vcat_22_ptr : RegionName := "candidate_vcat_22"
  let candidate_vcat_23_ptr : RegionName := "candidate_vcat_23"
  let candidate_vsum_4_ptr : RegionName := "candidate_vsum_4"
  let candidate_vbmm_45_ptr : RegionName := "candidate_vbmm_45"
  let candidate_vbmm_46_ptr : RegionName := "candidate_vbmm_46"
  let candidate_vadd_38_ptr : RegionName := "candidate_vadd_38"
  let candidate_vslice_backward_ptr : RegionName := "candidate_vslice_backward"
  let candidate_vslice_backward_1_ptr : RegionName := "candidate_vslice_backward_1"
  let candidate_vslice_backward_2_ptr : RegionName := "candidate_vslice_backward_2"
  let candidate_vslice_backward_3_ptr : RegionName := "candidate_vslice_backward_3"
  let candidate_vbmm_49_ptr : RegionName := "candidate_vbmm_49"
  let candidate_vbmm_50_ptr : RegionName := "candidate_vbmm_50"
  let candidate_vcat_25_ptr : RegionName := "candidate_vcat_25"
  let candidate_vcat_26_ptr : RegionName := "candidate_vcat_26"
  let candidate_vto_copy_122_ptr : RegionName := "candidate_vto_copy_122"
  let candidate_vto_copy_123_ptr : RegionName := "candidate_vto_copy_123"
  let candidate_vbmm_51_ptr : RegionName := "candidate_vbmm_51"
  let candidate_vmul_53_ptr : RegionName := "candidate_vmul_53"
  let candidate_vadd_39_ptr : RegionName := "candidate_vadd_39"
  let candidate_vsub_33_ptr : RegionName := "candidate_vsub_33"
  let candidate_vgt_18_ptr : RegionName := "candidate_vgt_18"
  let candidate_vmasked_fill_18_ptr : RegionName := "candidate_vmasked_fill_18"
  let candidate_vsoftmax_15_ptr : RegionName := "candidate_vsoftmax_15"
  let candidate_vto_copy_124_ptr : RegionName := "candidate_vto_copy_124"
  let candidate_vbmm_52_ptr : RegionName := "candidate_vbmm_52"
  let candidate_vlogsumexp_15_ptr : RegionName := "candidate_vlogsumexp_15"
  let candidate_vto_copy_125_ptr : RegionName := "candidate_vto_copy_125"
  let candidate_vto_copy_126_ptr : RegionName := "candidate_vto_copy_126"
  let candidate_vbmm_53_ptr : RegionName := "candidate_vbmm_53"
  let candidate_vmul_54_ptr : RegionName := "candidate_vmul_54"
  let candidate_vadd_40_ptr : RegionName := "candidate_vadd_40"
  let candidate_vsub_34_ptr : RegionName := "candidate_vsub_34"
  let candidate_vgt_19_ptr : RegionName := "candidate_vgt_19"
  let candidate_vmasked_fill_19_ptr : RegionName := "candidate_vmasked_fill_19"
  let candidate_vsoftmax_16_ptr : RegionName := "candidate_vsoftmax_16"
  let candidate_vto_copy_127_ptr : RegionName := "candidate_vto_copy_127"
  let candidate_vbmm_54_ptr : RegionName := "candidate_vbmm_54"
  let candidate_vlogsumexp_16_ptr : RegionName := "candidate_vlogsumexp_16"
  let candidate_vto_copy_128_ptr : RegionName := "candidate_vto_copy_128"
  let candidate_vto_copy_129_ptr : RegionName := "candidate_vto_copy_129"
  let candidate_vbmm_55_ptr : RegionName := "candidate_vbmm_55"
  let candidate_vmul_55_ptr : RegionName := "candidate_vmul_55"
  let candidate_vadd_41_ptr : RegionName := "candidate_vadd_41"
  let candidate_vsub_35_ptr : RegionName := "candidate_vsub_35"
  let candidate_vgt_20_ptr : RegionName := "candidate_vgt_20"
  let candidate_vmasked_fill_20_ptr : RegionName := "candidate_vmasked_fill_20"
  let candidate_vsoftmax_17_ptr : RegionName := "candidate_vsoftmax_17"
  let candidate_vto_copy_130_ptr : RegionName := "candidate_vto_copy_130"
  let candidate_vbmm_56_ptr : RegionName := "candidate_vbmm_56"
  let candidate_vlogsumexp_17_ptr : RegionName := "candidate_vlogsumexp_17"
  let candidate_vcat_27_ptr : RegionName := "candidate_vcat_27"
  let candidate_vcat_28_ptr : RegionName := "candidate_vcat_28"
  let candidate_vto_copy_131_ptr : RegionName := "candidate_vto_copy_131"
  let candidate_vto_copy_132_ptr : RegionName := "candidate_vto_copy_132"
  let candidate_vto_copy_133_ptr : RegionName := "candidate_vto_copy_133"
  let candidate_vto_copy_134_ptr : RegionName := "candidate_vto_copy_134"
  let candidate_vto_copy_135_ptr : RegionName := "candidate_vto_copy_135"
  let candidate_vbmm_57_ptr : RegionName := "candidate_vbmm_57"
  let candidate_vmul_56_ptr : RegionName := "candidate_vmul_56"
  let candidate_vadd_42_ptr : RegionName := "candidate_vadd_42"
  let candidate_vsub_36_ptr : RegionName := "candidate_vsub_36"
  let candidate_vgt_21_ptr : RegionName := "candidate_vgt_21"
  let candidate_vmasked_fill_21_ptr : RegionName := "candidate_vmasked_fill_21"
  let candidate_vsub_37_ptr : RegionName := "candidate_vsub_37"
  let candidate_vexp_3_ptr : RegionName := "candidate_vexp_3"
  let candidate_vmul_57_ptr : RegionName := "candidate_vmul_57"
  let candidate_vsum_5_ptr : RegionName := "candidate_vsum_5"
  let candidate_vbmm_58_ptr : RegionName := "candidate_vbmm_58"
  let candidate_vsub_38_ptr : RegionName := "candidate_vsub_38"
  let candidate_vmul_58_ptr : RegionName := "candidate_vmul_58"
  let candidate_vbmm_59_ptr : RegionName := "candidate_vbmm_59"
  let candidate_vmul_59_ptr : RegionName := "candidate_vmul_59"
  let candidate_vto_copy_136_ptr : RegionName := "candidate_vto_copy_136"
  let candidate_vbmm_60_ptr : RegionName := "candidate_vbmm_60"
  let candidate_vmul_60_ptr : RegionName := "candidate_vmul_60"
  let candidate_vto_copy_137_ptr : RegionName := "candidate_vto_copy_137"
  let candidate_vbmm_61_ptr : RegionName := "candidate_vbmm_61"
  let candidate_vto_copy_138_ptr : RegionName := "candidate_vto_copy_138"
  let candidate_vcat_29_ptr : RegionName := "candidate_vcat_29"
  let candidate_vcat_30_ptr : RegionName := "candidate_vcat_30"
  let candidate_vto_copy_139_ptr : RegionName := "candidate_vto_copy_139"
  let candidate_vto_copy_140_ptr : RegionName := "candidate_vto_copy_140"
  let candidate_vto_copy_141_ptr : RegionName := "candidate_vto_copy_141"
  let candidate_vto_copy_142_ptr : RegionName := "candidate_vto_copy_142"
  let candidate_vto_copy_143_ptr : RegionName := "candidate_vto_copy_143"
  let candidate_vbmm_62_ptr : RegionName := "candidate_vbmm_62"
  let candidate_vmul_61_ptr : RegionName := "candidate_vmul_61"
  let candidate_vadd_43_ptr : RegionName := "candidate_vadd_43"
  let candidate_vsub_39_ptr : RegionName := "candidate_vsub_39"
  let candidate_vgt_22_ptr : RegionName := "candidate_vgt_22"
  let candidate_vmasked_fill_22_ptr : RegionName := "candidate_vmasked_fill_22"
  let candidate_vsub_40_ptr : RegionName := "candidate_vsub_40"
  let candidate_vexp_4_ptr : RegionName := "candidate_vexp_4"
  let candidate_vmul_62_ptr : RegionName := "candidate_vmul_62"
  let candidate_vsum_6_ptr : RegionName := "candidate_vsum_6"
  let candidate_vbmm_63_ptr : RegionName := "candidate_vbmm_63"
  let candidate_vsub_41_ptr : RegionName := "candidate_vsub_41"
  let candidate_vmul_63_ptr : RegionName := "candidate_vmul_63"
  let candidate_vbmm_64_ptr : RegionName := "candidate_vbmm_64"
  let candidate_vmul_64_ptr : RegionName := "candidate_vmul_64"
  let candidate_vto_copy_144_ptr : RegionName := "candidate_vto_copy_144"
  let candidate_vbmm_65_ptr : RegionName := "candidate_vbmm_65"
  let candidate_vmul_65_ptr : RegionName := "candidate_vmul_65"
  let candidate_vto_copy_145_ptr : RegionName := "candidate_vto_copy_145"
  let candidate_vbmm_66_ptr : RegionName := "candidate_vbmm_66"
  let candidate_vto_copy_146_ptr : RegionName := "candidate_vto_copy_146"
  let candidate_vcat_31_ptr : RegionName := "candidate_vcat_31"
  let candidate_vcat_32_ptr : RegionName := "candidate_vcat_32"
  let candidate_vto_copy_147_ptr : RegionName := "candidate_vto_copy_147"
  let candidate_vto_copy_148_ptr : RegionName := "candidate_vto_copy_148"
  let candidate_vto_copy_149_ptr : RegionName := "candidate_vto_copy_149"
  let candidate_vto_copy_150_ptr : RegionName := "candidate_vto_copy_150"
  let candidate_vto_copy_151_ptr : RegionName := "candidate_vto_copy_151"
  let candidate_vbmm_67_ptr : RegionName := "candidate_vbmm_67"
  let candidate_vmul_66_ptr : RegionName := "candidate_vmul_66"
  let candidate_vadd_44_ptr : RegionName := "candidate_vadd_44"
  let candidate_vsub_42_ptr : RegionName := "candidate_vsub_42"
  let candidate_vgt_23_ptr : RegionName := "candidate_vgt_23"
  let candidate_vmasked_fill_23_ptr : RegionName := "candidate_vmasked_fill_23"
  let candidate_vsub_43_ptr : RegionName := "candidate_vsub_43"
  let candidate_vexp_5_ptr : RegionName := "candidate_vexp_5"
  let candidate_vmul_67_ptr : RegionName := "candidate_vmul_67"
  let candidate_vsum_7_ptr : RegionName := "candidate_vsum_7"
  let candidate_vbmm_68_ptr : RegionName := "candidate_vbmm_68"
  let candidate_vsub_44_ptr : RegionName := "candidate_vsub_44"
  let candidate_vmul_68_ptr : RegionName := "candidate_vmul_68"
  let candidate_vbmm_69_ptr : RegionName := "candidate_vbmm_69"
  let candidate_vmul_69_ptr : RegionName := "candidate_vmul_69"
  let candidate_vto_copy_152_ptr : RegionName := "candidate_vto_copy_152"
  let candidate_vbmm_70_ptr : RegionName := "candidate_vbmm_70"
  let candidate_vmul_70_ptr : RegionName := "candidate_vmul_70"
  let candidate_vto_copy_153_ptr : RegionName := "candidate_vto_copy_153"
  let candidate_vbmm_71_ptr : RegionName := "candidate_vbmm_71"
  let candidate_vto_copy_154_ptr : RegionName := "candidate_vto_copy_154"
  let candidate_vcat_33_ptr : RegionName := "candidate_vcat_33"
  let candidate_vcat_34_ptr : RegionName := "candidate_vcat_34"
  let candidate_vcat_35_ptr : RegionName := "candidate_vcat_35"
  let candidate_vcat_36_ptr : RegionName := "candidate_vcat_36"
  let candidate_vcat_37_ptr : RegionName := "candidate_vcat_37"
  let candidate_vsum_8_ptr : RegionName := "candidate_vsum_8"
  let candidate_vbmm_72_ptr : RegionName := "candidate_vbmm_72"
  let candidate_vbmm_73_ptr : RegionName := "candidate_vbmm_73"
  let candidate_vadd_45_ptr : RegionName := "candidate_vadd_45"
  let candidate_vslice_backward_4_ptr : RegionName := "candidate_vslice_backward_4"
  let candidate_vadd_46_ptr : RegionName := "candidate_vadd_46"
  let candidate_vslice_backward_5_ptr : RegionName := "candidate_vslice_backward_5"
  let candidate_vadd_47_ptr : RegionName := "candidate_vadd_47"
  let candidate_vslice_backward_6_ptr : RegionName := "candidate_vslice_backward_6"
  let candidate_vadd_48_ptr : RegionName := "candidate_vadd_48"
  let candidate_vslice_backward_7_ptr : RegionName := "candidate_vslice_backward_7"
  let candidate_vadd_49_ptr : RegionName := "candidate_vadd_49"
  let candidate_vbmm_76_ptr : RegionName := "candidate_vbmm_76"
  let candidate_vbmm_77_ptr : RegionName := "candidate_vbmm_77"
  let candidate_vcat_39_ptr : RegionName := "candidate_vcat_39"
  let candidate_vcat_40_ptr : RegionName := "candidate_vcat_40"
  let candidate_vto_copy_155_ptr : RegionName := "candidate_vto_copy_155"
  let candidate_vto_copy_156_ptr : RegionName := "candidate_vto_copy_156"
  let candidate_vbmm_78_ptr : RegionName := "candidate_vbmm_78"
  let candidate_vmul_71_ptr : RegionName := "candidate_vmul_71"
  let candidate_vadd_50_ptr : RegionName := "candidate_vadd_50"
  let candidate_vsub_45_ptr : RegionName := "candidate_vsub_45"
  let candidate_vgt_24_ptr : RegionName := "candidate_vgt_24"
  let candidate_vmasked_fill_24_ptr : RegionName := "candidate_vmasked_fill_24"
  let candidate_vsoftmax_18_ptr : RegionName := "candidate_vsoftmax_18"
  let candidate_vto_copy_157_ptr : RegionName := "candidate_vto_copy_157"
  let candidate_vbmm_79_ptr : RegionName := "candidate_vbmm_79"
  let candidate_vlogsumexp_18_ptr : RegionName := "candidate_vlogsumexp_18"
  let candidate_vto_copy_158_ptr : RegionName := "candidate_vto_copy_158"
  let candidate_vto_copy_159_ptr : RegionName := "candidate_vto_copy_159"
  let candidate_vbmm_80_ptr : RegionName := "candidate_vbmm_80"
  let candidate_vmul_72_ptr : RegionName := "candidate_vmul_72"
  let candidate_vadd_51_ptr : RegionName := "candidate_vadd_51"
  let candidate_vsub_46_ptr : RegionName := "candidate_vsub_46"
  let candidate_vgt_25_ptr : RegionName := "candidate_vgt_25"
  let candidate_vmasked_fill_25_ptr : RegionName := "candidate_vmasked_fill_25"
  let candidate_vsoftmax_19_ptr : RegionName := "candidate_vsoftmax_19"
  let candidate_vto_copy_160_ptr : RegionName := "candidate_vto_copy_160"
  let candidate_vbmm_81_ptr : RegionName := "candidate_vbmm_81"
  let candidate_vlogsumexp_19_ptr : RegionName := "candidate_vlogsumexp_19"
  let candidate_vto_copy_161_ptr : RegionName := "candidate_vto_copy_161"
  let candidate_vto_copy_162_ptr : RegionName := "candidate_vto_copy_162"
  let candidate_vbmm_82_ptr : RegionName := "candidate_vbmm_82"
  let candidate_vmul_73_ptr : RegionName := "candidate_vmul_73"
  let candidate_vadd_52_ptr : RegionName := "candidate_vadd_52"
  let candidate_vsub_47_ptr : RegionName := "candidate_vsub_47"
  let candidate_vgt_26_ptr : RegionName := "candidate_vgt_26"
  let candidate_vmasked_fill_26_ptr : RegionName := "candidate_vmasked_fill_26"
  let candidate_vsoftmax_20_ptr : RegionName := "candidate_vsoftmax_20"
  let candidate_vto_copy_163_ptr : RegionName := "candidate_vto_copy_163"
  let candidate_vbmm_83_ptr : RegionName := "candidate_vbmm_83"
  let candidate_vlogsumexp_20_ptr : RegionName := "candidate_vlogsumexp_20"
  let candidate_vcat_41_ptr : RegionName := "candidate_vcat_41"
  let candidate_vcat_42_ptr : RegionName := "candidate_vcat_42"
  let candidate_vto_copy_164_ptr : RegionName := "candidate_vto_copy_164"
  let candidate_vto_copy_165_ptr : RegionName := "candidate_vto_copy_165"
  let candidate_vto_copy_166_ptr : RegionName := "candidate_vto_copy_166"
  let candidate_vto_copy_167_ptr : RegionName := "candidate_vto_copy_167"
  let candidate_vto_copy_168_ptr : RegionName := "candidate_vto_copy_168"
  let candidate_vbmm_84_ptr : RegionName := "candidate_vbmm_84"
  let candidate_vmul_74_ptr : RegionName := "candidate_vmul_74"
  let candidate_vadd_53_ptr : RegionName := "candidate_vadd_53"
  let candidate_vsub_48_ptr : RegionName := "candidate_vsub_48"
  let candidate_vgt_27_ptr : RegionName := "candidate_vgt_27"
  let candidate_vmasked_fill_27_ptr : RegionName := "candidate_vmasked_fill_27"
  let candidate_vsub_49_ptr : RegionName := "candidate_vsub_49"
  let candidate_vexp_6_ptr : RegionName := "candidate_vexp_6"
  let candidate_vmul_75_ptr : RegionName := "candidate_vmul_75"
  let candidate_vsum_9_ptr : RegionName := "candidate_vsum_9"
  let candidate_vbmm_85_ptr : RegionName := "candidate_vbmm_85"
  let candidate_vsub_50_ptr : RegionName := "candidate_vsub_50"
  let candidate_vmul_76_ptr : RegionName := "candidate_vmul_76"
  let candidate_vbmm_86_ptr : RegionName := "candidate_vbmm_86"
  let candidate_vmul_77_ptr : RegionName := "candidate_vmul_77"
  let candidate_vto_copy_169_ptr : RegionName := "candidate_vto_copy_169"
  let candidate_vbmm_87_ptr : RegionName := "candidate_vbmm_87"
  let candidate_vmul_78_ptr : RegionName := "candidate_vmul_78"
  let candidate_vto_copy_170_ptr : RegionName := "candidate_vto_copy_170"
  let candidate_vbmm_88_ptr : RegionName := "candidate_vbmm_88"
  let candidate_vto_copy_171_ptr : RegionName := "candidate_vto_copy_171"
  let candidate_vto_copy_172_ptr : RegionName := "candidate_vto_copy_172"
  let candidate_vto_copy_173_ptr : RegionName := "candidate_vto_copy_173"
  let candidate_vto_copy_174_ptr : RegionName := "candidate_vto_copy_174"
  let candidate_vto_copy_175_ptr : RegionName := "candidate_vto_copy_175"
  let candidate_vto_copy_176_ptr : RegionName := "candidate_vto_copy_176"
  let candidate_vbmm_89_ptr : RegionName := "candidate_vbmm_89"
  let candidate_vmul_79_ptr : RegionName := "candidate_vmul_79"
  let candidate_vadd_54_ptr : RegionName := "candidate_vadd_54"
  let candidate_vsub_51_ptr : RegionName := "candidate_vsub_51"
  let candidate_vgt_28_ptr : RegionName := "candidate_vgt_28"
  let candidate_vmasked_fill_28_ptr : RegionName := "candidate_vmasked_fill_28"
  let candidate_vsub_52_ptr : RegionName := "candidate_vsub_52"
  let candidate_vexp_7_ptr : RegionName := "candidate_vexp_7"
  let candidate_vmul_80_ptr : RegionName := "candidate_vmul_80"
  let candidate_vsum_10_ptr : RegionName := "candidate_vsum_10"
  let candidate_vbmm_90_ptr : RegionName := "candidate_vbmm_90"
  let candidate_vsub_53_ptr : RegionName := "candidate_vsub_53"
  let candidate_vmul_81_ptr : RegionName := "candidate_vmul_81"
  let candidate_vbmm_91_ptr : RegionName := "candidate_vbmm_91"
  let candidate_vmul_82_ptr : RegionName := "candidate_vmul_82"
  let candidate_vto_copy_177_ptr : RegionName := "candidate_vto_copy_177"
  let candidate_vbmm_92_ptr : RegionName := "candidate_vbmm_92"
  let candidate_vmul_83_ptr : RegionName := "candidate_vmul_83"
  let candidate_vto_copy_178_ptr : RegionName := "candidate_vto_copy_178"
  let candidate_vbmm_93_ptr : RegionName := "candidate_vbmm_93"
  let candidate_vto_copy_179_ptr : RegionName := "candidate_vto_copy_179"
  let candidate_vto_copy_180_ptr : RegionName := "candidate_vto_copy_180"
  let candidate_vto_copy_181_ptr : RegionName := "candidate_vto_copy_181"
  let candidate_vto_copy_182_ptr : RegionName := "candidate_vto_copy_182"
  let candidate_vto_copy_183_ptr : RegionName := "candidate_vto_copy_183"
  let candidate_vto_copy_184_ptr : RegionName := "candidate_vto_copy_184"
  let candidate_vbmm_94_ptr : RegionName := "candidate_vbmm_94"
  let candidate_vmul_84_ptr : RegionName := "candidate_vmul_84"
  let candidate_vadd_55_ptr : RegionName := "candidate_vadd_55"
  let candidate_vsub_54_ptr : RegionName := "candidate_vsub_54"
  let candidate_vgt_29_ptr : RegionName := "candidate_vgt_29"
  let candidate_vmasked_fill_29_ptr : RegionName := "candidate_vmasked_fill_29"
  let candidate_vsub_55_ptr : RegionName := "candidate_vsub_55"
  let candidate_vexp_8_ptr : RegionName := "candidate_vexp_8"
  let candidate_vmul_85_ptr : RegionName := "candidate_vmul_85"
  let candidate_vsum_11_ptr : RegionName := "candidate_vsum_11"
  let candidate_vbmm_95_ptr : RegionName := "candidate_vbmm_95"
  let candidate_vsub_56_ptr : RegionName := "candidate_vsub_56"
  let candidate_vmul_86_ptr : RegionName := "candidate_vmul_86"
  let candidate_vbmm_96_ptr : RegionName := "candidate_vbmm_96"
  let candidate_vmul_87_ptr : RegionName := "candidate_vmul_87"
  let candidate_vto_copy_185_ptr : RegionName := "candidate_vto_copy_185"
  let candidate_vbmm_97_ptr : RegionName := "candidate_vbmm_97"
  let candidate_vmul_88_ptr : RegionName := "candidate_vmul_88"
  let candidate_vto_copy_186_ptr : RegionName := "candidate_vto_copy_186"
  let candidate_vbmm_98_ptr : RegionName := "candidate_vbmm_98"
  let candidate_vto_copy_187_ptr : RegionName := "candidate_vto_copy_187"
  let candidate_vcat_43_ptr : RegionName := "candidate_vcat_43"
  let candidate_vcat_44_ptr : RegionName := "candidate_vcat_44"
  let candidate_vcat_45_ptr : RegionName := "candidate_vcat_45"
  let candidate_vsum_12_ptr : RegionName := "candidate_vsum_12"
  let candidate_vbmm_99_ptr : RegionName := "candidate_vbmm_99"
  let candidate_vbmm_100_ptr : RegionName := "candidate_vbmm_100"
  let candidate_vadd_56_ptr : RegionName := "candidate_vadd_56"
  let candidate_vslice_backward_8_ptr : RegionName := "candidate_vslice_backward_8"
  let candidate_vadd_57_ptr : RegionName := "candidate_vadd_57"
  let candidate_vslice_backward_9_ptr : RegionName := "candidate_vslice_backward_9"
  let candidate_vadd_58_ptr : RegionName := "candidate_vadd_58"
  let candidate_vslice_backward_10_ptr : RegionName := "candidate_vslice_backward_10"
  let candidate_vadd_59_ptr : RegionName := "candidate_vadd_59"
  let candidate_vslice_backward_11_ptr : RegionName := "candidate_vslice_backward_11"
  let candidate_vadd_60_ptr : RegionName := "candidate_vadd_60"
  let candidate_vbmm_103_ptr : RegionName := "candidate_vbmm_103"
  let candidate_vbmm_104_ptr : RegionName := "candidate_vbmm_104"
  let candidate_vcat_47_ptr : RegionName := "candidate_vcat_47"
  let candidate_vcat_48_ptr : RegionName := "candidate_vcat_48"
  let candidate_vto_copy_188_ptr : RegionName := "candidate_vto_copy_188"
  let candidate_vto_copy_189_ptr : RegionName := "candidate_vto_copy_189"
  let candidate_vbmm_105_ptr : RegionName := "candidate_vbmm_105"
  let candidate_vmul_89_ptr : RegionName := "candidate_vmul_89"
  let candidate_vadd_61_ptr : RegionName := "candidate_vadd_61"
  let candidate_vsub_57_ptr : RegionName := "candidate_vsub_57"
  let candidate_vgt_30_ptr : RegionName := "candidate_vgt_30"
  let candidate_vmasked_fill_30_ptr : RegionName := "candidate_vmasked_fill_30"
  let candidate_vsoftmax_21_ptr : RegionName := "candidate_vsoftmax_21"
  let candidate_vto_copy_190_ptr : RegionName := "candidate_vto_copy_190"
  let candidate_vbmm_106_ptr : RegionName := "candidate_vbmm_106"
  let candidate_vlogsumexp_21_ptr : RegionName := "candidate_vlogsumexp_21"
  let candidate_vto_copy_191_ptr : RegionName := "candidate_vto_copy_191"
  let candidate_vto_copy_192_ptr : RegionName := "candidate_vto_copy_192"
  let candidate_vbmm_107_ptr : RegionName := "candidate_vbmm_107"
  let candidate_vmul_90_ptr : RegionName := "candidate_vmul_90"
  let candidate_vadd_62_ptr : RegionName := "candidate_vadd_62"
  let candidate_vsub_58_ptr : RegionName := "candidate_vsub_58"
  let candidate_vgt_31_ptr : RegionName := "candidate_vgt_31"
  let candidate_vmasked_fill_31_ptr : RegionName := "candidate_vmasked_fill_31"
  let candidate_vsoftmax_22_ptr : RegionName := "candidate_vsoftmax_22"
  let candidate_vto_copy_193_ptr : RegionName := "candidate_vto_copy_193"
  let candidate_vbmm_108_ptr : RegionName := "candidate_vbmm_108"
  let candidate_vlogsumexp_22_ptr : RegionName := "candidate_vlogsumexp_22"
  let candidate_vto_copy_194_ptr : RegionName := "candidate_vto_copy_194"
  let candidate_vto_copy_195_ptr : RegionName := "candidate_vto_copy_195"
  let candidate_vbmm_109_ptr : RegionName := "candidate_vbmm_109"
  let candidate_vmul_91_ptr : RegionName := "candidate_vmul_91"
  let candidate_vadd_63_ptr : RegionName := "candidate_vadd_63"
  let candidate_vsub_59_ptr : RegionName := "candidate_vsub_59"
  let candidate_vgt_32_ptr : RegionName := "candidate_vgt_32"
  let candidate_vmasked_fill_32_ptr : RegionName := "candidate_vmasked_fill_32"
  let candidate_vsoftmax_23_ptr : RegionName := "candidate_vsoftmax_23"
  let candidate_vto_copy_196_ptr : RegionName := "candidate_vto_copy_196"
  let candidate_vbmm_110_ptr : RegionName := "candidate_vbmm_110"
  let candidate_vlogsumexp_23_ptr : RegionName := "candidate_vlogsumexp_23"
  let candidate_vcat_49_ptr : RegionName := "candidate_vcat_49"
  let candidate_vcat_50_ptr : RegionName := "candidate_vcat_50"
  let candidate_vto_copy_197_ptr : RegionName := "candidate_vto_copy_197"
  let candidate_vto_copy_198_ptr : RegionName := "candidate_vto_copy_198"
  let candidate_vto_copy_199_ptr : RegionName := "candidate_vto_copy_199"
  let candidate_vto_copy_200_ptr : RegionName := "candidate_vto_copy_200"
  let candidate_vto_copy_201_ptr : RegionName := "candidate_vto_copy_201"
  let candidate_vbmm_111_ptr : RegionName := "candidate_vbmm_111"
  let candidate_vmul_92_ptr : RegionName := "candidate_vmul_92"
  let candidate_vadd_64_ptr : RegionName := "candidate_vadd_64"
  let candidate_vsub_60_ptr : RegionName := "candidate_vsub_60"
  let candidate_vgt_33_ptr : RegionName := "candidate_vgt_33"
  let candidate_vmasked_fill_33_ptr : RegionName := "candidate_vmasked_fill_33"
  let candidate_vsub_61_ptr : RegionName := "candidate_vsub_61"
  let candidate_vexp_9_ptr : RegionName := "candidate_vexp_9"
  let candidate_vmul_93_ptr : RegionName := "candidate_vmul_93"
  let candidate_vsum_13_ptr : RegionName := "candidate_vsum_13"
  let candidate_vbmm_112_ptr : RegionName := "candidate_vbmm_112"
  let candidate_vsub_62_ptr : RegionName := "candidate_vsub_62"
  let candidate_vmul_94_ptr : RegionName := "candidate_vmul_94"
  let candidate_vbmm_113_ptr : RegionName := "candidate_vbmm_113"
  let candidate_vmul_95_ptr : RegionName := "candidate_vmul_95"
  let candidate_vto_copy_202_ptr : RegionName := "candidate_vto_copy_202"
  let candidate_vbmm_114_ptr : RegionName := "candidate_vbmm_114"
  let candidate_vmul_96_ptr : RegionName := "candidate_vmul_96"
  let candidate_vto_copy_203_ptr : RegionName := "candidate_vto_copy_203"
  let candidate_vbmm_115_ptr : RegionName := "candidate_vbmm_115"
  let candidate_vto_copy_204_ptr : RegionName := "candidate_vto_copy_204"
  let candidate_vcat_51_ptr : RegionName := "candidate_vcat_51"
  let candidate_vcat_52_ptr : RegionName := "candidate_vcat_52"
  let candidate_vto_copy_205_ptr : RegionName := "candidate_vto_copy_205"
  let candidate_vto_copy_206_ptr : RegionName := "candidate_vto_copy_206"
  let candidate_vto_copy_207_ptr : RegionName := "candidate_vto_copy_207"
  let candidate_vto_copy_208_ptr : RegionName := "candidate_vto_copy_208"
  let candidate_vto_copy_209_ptr : RegionName := "candidate_vto_copy_209"
  let candidate_vbmm_116_ptr : RegionName := "candidate_vbmm_116"
  let candidate_vmul_97_ptr : RegionName := "candidate_vmul_97"
  let candidate_vadd_65_ptr : RegionName := "candidate_vadd_65"
  let candidate_vsub_63_ptr : RegionName := "candidate_vsub_63"
  let candidate_vgt_34_ptr : RegionName := "candidate_vgt_34"
  let candidate_vmasked_fill_34_ptr : RegionName := "candidate_vmasked_fill_34"
  let candidate_vsub_64_ptr : RegionName := "candidate_vsub_64"
  let candidate_vexp_10_ptr : RegionName := "candidate_vexp_10"
  let candidate_vmul_98_ptr : RegionName := "candidate_vmul_98"
  let candidate_vsum_14_ptr : RegionName := "candidate_vsum_14"
  let candidate_vbmm_117_ptr : RegionName := "candidate_vbmm_117"
  let candidate_vsub_65_ptr : RegionName := "candidate_vsub_65"
  let candidate_vmul_99_ptr : RegionName := "candidate_vmul_99"
  let candidate_vbmm_118_ptr : RegionName := "candidate_vbmm_118"
  let candidate_vmul_100_ptr : RegionName := "candidate_vmul_100"
  let candidate_vto_copy_210_ptr : RegionName := "candidate_vto_copy_210"
  let candidate_vbmm_119_ptr : RegionName := "candidate_vbmm_119"
  let candidate_vmul_101_ptr : RegionName := "candidate_vmul_101"
  let candidate_vto_copy_211_ptr : RegionName := "candidate_vto_copy_211"
  let candidate_vbmm_120_ptr : RegionName := "candidate_vbmm_120"
  let candidate_vto_copy_212_ptr : RegionName := "candidate_vto_copy_212"
  let candidate_vcat_53_ptr : RegionName := "candidate_vcat_53"
  let candidate_vcat_54_ptr : RegionName := "candidate_vcat_54"
  let candidate_vto_copy_213_ptr : RegionName := "candidate_vto_copy_213"
  let candidate_vto_copy_214_ptr : RegionName := "candidate_vto_copy_214"
  let candidate_vto_copy_215_ptr : RegionName := "candidate_vto_copy_215"
  let candidate_vto_copy_216_ptr : RegionName := "candidate_vto_copy_216"
  let candidate_vto_copy_217_ptr : RegionName := "candidate_vto_copy_217"
  let candidate_vbmm_121_ptr : RegionName := "candidate_vbmm_121"
  let candidate_vmul_102_ptr : RegionName := "candidate_vmul_102"
  let candidate_vadd_66_ptr : RegionName := "candidate_vadd_66"
  let candidate_vsub_66_ptr : RegionName := "candidate_vsub_66"
  let candidate_vgt_35_ptr : RegionName := "candidate_vgt_35"
  let candidate_vmasked_fill_35_ptr : RegionName := "candidate_vmasked_fill_35"
  let candidate_vsub_67_ptr : RegionName := "candidate_vsub_67"
  let candidate_vexp_11_ptr : RegionName := "candidate_vexp_11"
  let candidate_vmul_103_ptr : RegionName := "candidate_vmul_103"
  let candidate_vsum_15_ptr : RegionName := "candidate_vsum_15"
  let candidate_vbmm_122_ptr : RegionName := "candidate_vbmm_122"
  let candidate_vsub_68_ptr : RegionName := "candidate_vsub_68"
  let candidate_vmul_104_ptr : RegionName := "candidate_vmul_104"
  let candidate_vbmm_123_ptr : RegionName := "candidate_vbmm_123"
  let candidate_vmul_105_ptr : RegionName := "candidate_vmul_105"
  let candidate_vto_copy_218_ptr : RegionName := "candidate_vto_copy_218"
  let candidate_vbmm_124_ptr : RegionName := "candidate_vbmm_124"
  let candidate_vmul_106_ptr : RegionName := "candidate_vmul_106"
  let candidate_vto_copy_219_ptr : RegionName := "candidate_vto_copy_219"
  let candidate_vbmm_125_ptr : RegionName := "candidate_vbmm_125"
  let candidate_vto_copy_220_ptr : RegionName := "candidate_vto_copy_220"
  let candidate_vcat_55_ptr : RegionName := "candidate_vcat_55"
  let candidate_vcat_56_ptr : RegionName := "candidate_vcat_56"
  let candidate_vcat_57_ptr : RegionName := "candidate_vcat_57"
  let candidate_vcat_58_ptr : RegionName := "candidate_vcat_58"
  let candidate_vcat_59_ptr : RegionName := "candidate_vcat_59"
  let candidate_vsum_16_ptr : RegionName := "candidate_vsum_16"
  let candidate_vbmm_126_ptr : RegionName := "candidate_vbmm_126"
  let candidate_vbmm_127_ptr : RegionName := "candidate_vbmm_127"
  let candidate_vadd_67_ptr : RegionName := "candidate_vadd_67"
  let candidate_vslice_backward_12_ptr : RegionName := "candidate_vslice_backward_12"
  let candidate_vadd_68_ptr : RegionName := "candidate_vadd_68"
  let candidate_vslice_backward_13_ptr : RegionName := "candidate_vslice_backward_13"
  let candidate_vadd_69_ptr : RegionName := "candidate_vadd_69"
  let candidate_vslice_backward_14_ptr : RegionName := "candidate_vslice_backward_14"
  let candidate_vadd_70_ptr : RegionName := "candidate_vadd_70"
  let candidate_vslice_backward_15_ptr : RegionName := "candidate_vslice_backward_15"
  let candidate_vadd_71_ptr : RegionName := "candidate_vadd_71"
  let candidate_vcat_61_ptr : RegionName := "candidate_vcat_61"
  let candidate_vindex_put_1_ptr : RegionName := "candidate_vindex_put_1"
  let candidate_vmul_107_ptr : RegionName := "candidate_vmul_107"
  let candidate_vmul_108_ptr : RegionName := "candidate_vmul_108"
  let candidate_vselect_backward_ptr : RegionName := "candidate_vselect_backward"
  let candidate_vneg_2_ptr : RegionName := "candidate_vneg_2"
  let candidate_vselect_backward_1_ptr : RegionName := "candidate_vselect_backward_1"
  let candidate_vadd_72_ptr : RegionName := "candidate_vadd_72"
  let candidate_vadd_73_ptr : RegionName := "candidate_vadd_73"
  let candidate_vmul_109_ptr : RegionName := "candidate_vmul_109"
  let candidate_vmul_110_ptr : RegionName := "candidate_vmul_110"
  let candidate_vselect_backward_2_ptr : RegionName := "candidate_vselect_backward_2"
  let candidate_vneg_3_ptr : RegionName := "candidate_vneg_3"
  let candidate_vselect_backward_3_ptr : RegionName := "candidate_vselect_backward_3"
  let candidate_vadd_74_ptr : RegionName := "candidate_vadd_74"
  let candidate_vadd_75_ptr : RegionName := "candidate_vadd_75"
  let candidate_vcat_62_ptr : RegionName := "candidate_vcat_62"
  let candidate_vto_copy_221_ptr : RegionName := "candidate_vto_copy_221"
  let candidate_vmul_111_ptr : RegionName := "candidate_vmul_111"
  let candidate_vmul_112_ptr : RegionName := "candidate_vmul_112"
  let candidate_vmul_113_ptr : RegionName := "candidate_vmul_113"
  let candidate_vsum_17_ptr : RegionName := "candidate_vsum_17"
  let candidate_vpow_6_ptr : RegionName := "candidate_vpow_6"
  let candidate_vmul_114_ptr : RegionName := "candidate_vmul_114"
  let candidate_vmul_115_ptr : RegionName := "candidate_vmul_115"
  let candidate_vdiv_3_ptr : RegionName := "candidate_vdiv_3"
  let candidate_vpow_7_ptr : RegionName := "candidate_vpow_7"
  let candidate_vmul_116_ptr : RegionName := "candidate_vmul_116"
  let candidate_vmul_117_ptr : RegionName := "candidate_vmul_117"
  let candidate_vadd_76_ptr : RegionName := "candidate_vadd_76"
  let candidate_vto_copy_222_ptr : RegionName := "candidate_vto_copy_222"
  let candidate_vcat_63_ptr : RegionName := "candidate_vcat_63"
  let candidate_vmm_29_ptr : RegionName := "candidate_vmm_29"
  let candidate_vmm_30_ptr : RegionName := "candidate_vmm_30"
  let candidate_vto_copy_223_ptr : RegionName := "candidate_vto_copy_223"
  let candidate_vmul_118_ptr : RegionName := "candidate_vmul_118"
  let candidate_vmul_119_ptr : RegionName := "candidate_vmul_119"
  let candidate_vmul_120_ptr : RegionName := "candidate_vmul_120"
  let candidate_vsum_18_ptr : RegionName := "candidate_vsum_18"
  let candidate_vpow_8_ptr : RegionName := "candidate_vpow_8"
  let candidate_vmul_121_ptr : RegionName := "candidate_vmul_121"
  let candidate_vmul_122_ptr : RegionName := "candidate_vmul_122"
  let candidate_vdiv_4_ptr : RegionName := "candidate_vdiv_4"
  let candidate_vpow_9_ptr : RegionName := "candidate_vpow_9"
  let candidate_vmul_123_ptr : RegionName := "candidate_vmul_123"
  let candidate_vmul_124_ptr : RegionName := "candidate_vmul_124"
  let candidate_vadd_77_ptr : RegionName := "candidate_vadd_77"
  let candidate_vto_copy_224_ptr : RegionName := "candidate_vto_copy_224"
  let candidate_vmm_31_ptr : RegionName := "candidate_vmm_31"
  let candidate_vadd_78_ptr : RegionName := "candidate_vadd_78"
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
    r33_col = tl.arange($(576))
    r33_lane = tl.arange($(512))
    r33_left = tl.load($(candidate_vindex_1_ptr) + ((((((r33_row[:, None] * $(512)) + r33_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r33_row[:, None] * $(512)) + r33_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r33_right = tl.load($(candidate_vto_copy_3_ptr) + (((r33_col[None, :] % $(576)) * $(512)) + (r33_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r33_value = tl.dot(r33_left, r33_right)
    r33_value = r33_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_2_ptr) + ((r33_row[:, None] * $(576)) + r33_col[None, :]), r33_value)
    r34_z = tl.arange($(65536))
    r34_value = tl.load($(candidate_vmm_2_ptr) + ((((r34_z / $(512)) % $(128)) * $(576)) + (r34_z % $(512))), dtype=tl.bfloat16)
    r34_value = r34_value.to(tl.float32)
    r34_value = tl.broadcast_to(r34_value, [$(65536)])
    tl.store($(candidate_vto_copy_22_ptr) + r34_z, r34_value)
    r35_z = tl.arange($(65536))
    r35_value = ((tl.load($(candidate_vto_copy_22_ptr) + ((((r35_z / $(512)) % $(128)) * $(512)) + (r35_z % $(512))), dtype=tl.float32)) * (tl.load($(candidate_vto_copy_22_ptr) + ((((r35_z / $(512)) % $(128)) * $(512)) + (r35_z % $(512))), dtype=tl.float32)))
    r35_value = r35_value.to(tl.float32)
    r35_value = tl.broadcast_to(r35_value, [$(65536)])
    tl.store($(candidate_vpow_4_ptr) + r35_z, r35_value)
    r36_z = tl.arange($(128))
    r36_lane = tl.arange($(512))
    r36_value = (tl.sum(tl.load($(candidate_vpow_4_ptr) + ((r36_z[:, None] * $(512)) + r36_lane[None, :]), dtype=tl.float32), axis=1) / 512.0)
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
    r39_z = tl.arange($(65536))
    r39_value = (tl.load($(candidate_vto_copy_22_ptr) + ((((r39_z / $(512)) % $(128)) * $(512)) + (r39_z % $(512))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_1_ptr) + ((r39_z / $(512)) % $(128)), dtype=tl.float32))
    r39_value = r39_value.to(tl.float32)
    r39_value = tl.broadcast_to(r39_value, [$(65536)])
    tl.store($(candidate_vmul_15_ptr) + r39_z, r39_value)
    r40_z = tl.arange($(65536))
    r40_value = (tl.cast(tl.load($(candidate_vto_copy_4_ptr) + (r40_z % $(512)), dtype=tl.bfloat16), tl.float32) * tl.load($(candidate_vmul_15_ptr) + ((((r40_z / $(512)) % $(128)) * $(512)) + (r40_z % $(512))), dtype=tl.float32))
    r40_value = r40_value.to(tl.float32)
    r40_value = tl.broadcast_to(r40_value, [$(65536)])
    tl.store($(candidate_vmul_16_ptr) + r40_z, r40_value)
    r41_z = tl.arange($(65536))
    r41_value = tl.load($(candidate_vmul_16_ptr) + ((((r41_z / $(512)) % $(128)) * $(512)) + (r41_z % $(512))), dtype=tl.float32)
    r41_value = r41_value.to(tl.bfloat16)
    r41_value = tl.broadcast_to(r41_value, [$(65536)])
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
    r62_value = -tl.load($(candidate_vmm_2_ptr) + ((((r62_z / $(32)) % $(128)) * $(576)) + ((r62_z % $(32)) * $(2)) + $(513)), dtype=tl.bfloat16)
    r62_value = r62_value.to(tl.bfloat16)
    r62_value = tl.broadcast_to(r62_value, [$(4096)])
    tl.store($(candidate_vneg_1_ptr) + r62_z, r62_value)
    r63_z = tl.arange($(8192))
    r63_value = tl.where((r63_z % $(2)) < $(1), tl.load($(candidate_vneg_1_ptr) + ((((r63_z / $(64)) % $(128)) * $(32)) + ((r63_z / $(2)) % $(32))), dtype=tl.bfloat16), tl.load($(candidate_vmm_2_ptr) + ((((r63_z / $(64)) % $(128)) * $(576)) + (((r63_z / $(2)) % $(32)) * $(2)) + $(512)), dtype=tl.bfloat16))
    r63_value = r63_value.to(tl.bfloat16)
    r63_value = tl.broadcast_to(r63_value, [$(8192)])
    tl.store($(candidate_vstack_1_ptr) + r63_z, r63_value)
    r64_z = tl.arange($(8192))
    r64_value = (tl.load($(candidate_vmm_2_ptr) + ((((r64_z / $(64)) % $(128)) * $(576)) + (r64_z % $(64)) + $(512)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_27_ptr) + ((((r64_z / $(64)) % $(128)) * $(64)) + (r64_z % $(64))), dtype=tl.bfloat16))
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
    r67_z = tl.arange($(73728))
    r67_value = tl.where((r67_z % $(576)) < $(512), tl.load($(candidate_vto_copy_23_ptr) + ((((((((r67_z / $(576)) % $(128)) * $(512)) + ((r67_z % $(576)) - $(0))) / $(512)) % $(128)) * $(512)) + (((((r67_z / $(576)) % $(128)) * $(512)) + ((r67_z % $(576)) - $(0))) % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vadd_7_ptr) + ((((((((r67_z / $(576)) % $(128)) * $(64)) + ((r67_z % $(576)) - $(512))) / $(64)) % $(128)) * $(64)) + (((((r67_z / $(576)) % $(128)) * $(64)) + ((r67_z % $(576)) - $(512))) % $(64))), dtype=tl.bfloat16))
    r67_value = r67_value.to(tl.bfloat16)
    r67_value = tl.broadcast_to(r67_value, [$(73728)])
    tl.store($(candidate_vcat_5_ptr) + r67_z, r67_value)
    r68_z = tl.arange($(73728))
    r68_value = tl.load($(candidate_vcat_5_ptr) + (((tl.load($(candidate_vsort_index_ptr) + ((r68_z / $(576)) % $(128)), dtype=tl.int64) % $(128)) * $(576)) + (r68_z % $(576))), dtype=tl.bfloat16)
    r68_value = r68_value.to(tl.bfloat16)
    r68_value = tl.broadcast_to(r68_value, [$(73728)])
    tl.store($(candidate_vindex_2_ptr) + r68_z, r68_value)
    r69_z = tl.arange($(65536))
    r69_value = tl.cast(0.0, tl.bfloat16)
    for r69_lane in range($(128)) {
      r69_value = tl.where(tl.load($(candidate_vsort_index_ptr) + (r69_lane % $(128)), dtype=tl.int64) == ((r69_z / $(512)) % $(128)), r69_value + tl.load($(input1_ptr) + (((r69_lane % $(128)) * $(512)) + (r69_z % $(512))), dtype=tl.bfloat16), r69_value)
    }
    r69_value = r69_value.to(tl.bfloat16)
    r69_value = tl.broadcast_to(r69_value, [$(65536)])
    tl.store($(candidate_vindex_put_ptr) + r69_z, r69_value)
    r70_row = tl.arange($(128))
    r70_col = tl.arange($(1024))
    r70_lane = tl.arange($(512))
    r70_left = tl.load($(candidate_vindex_put_ptr) + ((((((r70_row[:, None] * $(512)) + r70_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r70_row[:, None] * $(512)) + r70_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r70_right = tl.load($(candidate_vto_copy_6_ptr) + (((r70_lane[:, None] % $(512)) * $(1024)) + (r70_col[None, :] % $(1024))), dtype=tl.bfloat16)
    r70_value = tl.dot(r70_left, r70_right)
    r70_value = r70_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_28_ptr) + ((r70_row[:, None] * $(1024)) + r70_col[None, :]), r70_value)
    r71_row = tl.arange($(128))
    r71_col = tl.arange($(1024))
    r71_lane = tl.arange($(512))
    for r71_group in range($(1)) {
      r71_left = tl.load($(candidate_vindex_2_ptr) + ((((((r71_row[:, None] * $(512)) + r71_lane[None, :]) / $(512)) % $(128)) * $(576)) + (((r71_row[:, None] * $(512)) + r71_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r71_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r71_lane[:, None] * $(1024)) + r71_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r71_lane[:, None] * $(1024)) + r71_col[None, :]) % $(128)) * $(512)) + ((((r71_lane[:, None] * $(1024)) + r71_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r71_value = tl.dot(r71_left, r71_right)
      r71_value = r71_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_22_ptr) + ((r71_group * $(131072)) + (r71_row[:, None] * $(1024)) + r71_col[None, :]), r71_value)
    }
    r72_row = tl.arange($(128))
    r72_col = tl.arange($(1024))
    r72_lane = tl.arange($(512))
    for r72_group in range($(1)) {
      r72_left = tl.load($(candidate_vindex_2_ptr) + ((((((r72_row[:, None] * $(512)) + r72_lane[None, :]) / $(512)) % $(128)) * $(576)) + (((r72_row[:, None] * $(512)) + r72_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r72_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r72_lane[:, None] * $(1024)) + r72_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r72_lane[:, None] * $(1024)) + r72_col[None, :]) % $(128)) * $(512)) + ((((r72_lane[:, None] * $(1024)) + r72_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r72_value = tl.dot(r72_left, r72_right)
      r72_value = r72_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_23_ptr) + ((r72_group * $(131072)) + (r72_row[:, None] * $(1024)) + r72_col[None, :]), r72_value)
    }
    r73_z = tl.arange($(49152))
    r73_value = tl.where((r73_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r73_z / $(1536)) % $(32)) * $(1024)) + (((r73_z / $(192)) % $(8)) * $(128)) + ((r73_z % $(192)) - $(0)) + $(98304)) / $(1024)) % $(128)) * $(1536)) + (((((((r73_z / $(1536)) % $(32)) * $(1024)) + (((r73_z / $(192)) % $(8)) * $(128)) + ((r73_z % $(192)) - $(0)) + $(98304)) / $(128)) % $(8)) * $(192)) + (((((r73_z / $(1536)) % $(32)) * $(1024)) + (((r73_z / $(192)) % $(8)) * $(128)) + ((r73_z % $(192)) - $(0)) + $(98304)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vadd_6_ptr) + ((((((((r73_z / $(1536)) % $(32)) * $(512)) + (((r73_z / $(192)) % $(8)) * $(64)) + ((r73_z % $(192)) - $(128)) + $(49152)) / $(512)) % $(128)) * $(512)) + (((((((r73_z / $(1536)) % $(32)) * $(512)) + (((r73_z / $(192)) % $(8)) * $(64)) + ((r73_z % $(192)) - $(128)) + $(49152)) / $(64)) % $(8)) * $(64)) + (((((r73_z / $(1536)) % $(32)) * $(512)) + (((r73_z / $(192)) % $(8)) * $(64)) + ((r73_z % $(192)) - $(128)) + $(49152)) % $(64))), dtype=tl.bfloat16))
    r73_value = r73_value.to(tl.bfloat16)
    r73_value = tl.broadcast_to(r73_value, [$(49152)])
    tl.store($(candidate_vcat_11_ptr) + r73_z, r73_value)
    r74_z = tl.arange($(196608))
    r74_value = tl.where((r74_z % $(192)) < $(128), tl.load($(candidate_vbmm_22_ptr) + ((((((((r74_z / $(1536)) % $(128)) * $(1024)) + (((r74_z / $(192)) % $(8)) * $(128)) + ((r74_z % $(192)) - $(0))) / $(1024)) % $(128)) * $(1024)) + (((((((r74_z / $(1536)) % $(128)) * $(1024)) + (((r74_z / $(192)) % $(8)) * $(128)) + ((r74_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r74_z / $(1536)) % $(128)) * $(1024)) + (((r74_z / $(192)) % $(8)) * $(128)) + ((r74_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vindex_2_ptr) + ((((r74_z / $(1536)) % $(128)) * $(576)) + ((((r74_z % $(192)) - $(128)) + $(512)) % $(576))), dtype=tl.bfloat16))
    r74_value = r74_value.to(tl.bfloat16)
    r74_value = tl.broadcast_to(r74_value, [$(196608)])
    tl.store($(candidate_vcat_12_ptr) + r74_z, r74_value)
    r75_z = tl.arange($(18432))
    r75_value = tl.load($(candidate_vcat_11_ptr) + ((((r75_z / $(192)) % $(12)) * $(1536)) + (((r75_z / $(2304)) % $(8)) * $(192)) + (r75_z % $(192))), dtype=tl.bfloat16)
    r75_value = r75_value.to(tl.float32)
    r75_value = tl.broadcast_to(r75_value, [$(18432)])
    tl.store($(candidate_vto_copy_89_ptr) + r75_z, r75_value)
    r76_z = tl.arange($(55296))
    r76_value = tl.load($(candidate_vcat_12_ptr) + ((((r76_z / $(192)) % $(36)) * $(1536)) + (((r76_z / $(6912)) % $(8)) * $(192)) + (r76_z % $(192))), dtype=tl.bfloat16)
    r76_value = r76_value.to(tl.float32)
    r76_value = tl.broadcast_to(r76_value, [$(55296)])
    tl.store($(candidate_vto_copy_90_ptr) + r76_z, r76_value)
    r77_row = tl.arange($(12))
    r77_col = tl.arange($(36))
    r77_lane = tl.arange($(192))
    for r77_group in range($(8)) {
      r77_left = tl.load($(candidate_vto_copy_89_ptr) + (((((((r77_group % $(8)) * $(2304)) + (r77_row[:, None] * $(192)) + r77_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r77_group % $(8)) * $(2304)) + (r77_row[:, None] * $(192)) + r77_lane[None, :]) / $(192)) % $(12)) * $(192)) + ((((r77_group % $(8)) * $(2304)) + (r77_row[:, None] * $(192)) + r77_lane[None, :]) % $(192))), dtype=tl.float32)
      r77_right = tl.load($(candidate_vto_copy_90_ptr) + (((((((r77_group % $(8)) * $(6912)) + (r77_lane[:, None] * $(36)) + r77_col[None, :]) / $(6912)) % $(8)) * $(6912)) + (((((r77_group % $(8)) * $(6912)) + (r77_lane[:, None] * $(36)) + r77_col[None, :]) % $(36)) * $(192)) + (((((r77_group % $(8)) * $(6912)) + (r77_lane[:, None] * $(36)) + r77_col[None, :]) / $(36)) % $(192))), dtype=tl.float32)
      r77_value = tl.dot(r77_left, r77_right)
      r77_value = r77_value.to(tl.float32)
      tl.store($(candidate_vbmm_24_ptr) + ((r77_group * $(432)) + (r77_row[:, None] * $(36)) + r77_col[None, :]), r77_value)
    }
    r78_z = tl.arange($(3456))
    r78_value = (tl.load($(candidate_vbmm_24_ptr) + ((((r78_z / $(432)) % $(8)) * $(432)) + (((r78_z / $(36)) % $(12)) * $(36)) + (r78_z % $(36))), dtype=tl.float32) * 0.07216878364870322)
    r78_value = r78_value.to(tl.float32)
    r78_value = tl.broadcast_to(r78_value, [$(3456)])
    tl.store($(candidate_vmul_35_ptr) + r78_z, r78_value)
    r79_z = tl.arange($(12))
    r79_value = (tl.cast(r79_z, tl.int64) + tl.full([], $(36), dtype=tl.int64))
    r79_value = tl.broadcast_to(r79_value, [$(12)])
    tl.store($(candidate_vadd_32_ptr) + r79_z, r79_value)
    r80_z = tl.arange($(12))
    r80_value = (tl.load($(candidate_vadd_32_ptr) + r80_z, dtype=tl.int64) - tl.full([], $(12), dtype=tl.int64))
    r80_value = tl.broadcast_to(r80_value, [$(12)])
    tl.store($(candidate_vsub_21_ptr) + r80_z, r80_value)
    r81_z = tl.arange($(432))
    r81_value = (tl.cast((r81_z % $(36)), tl.int64) > tl.load($(candidate_vsub_21_ptr) + ((r81_z / $(36)) % $(12)), dtype=tl.int64))
    r81_value = tl.broadcast_to(r81_value, [$(432)])
    tl.store($(candidate_vgt_12_ptr) + r81_z, r81_value)
    r82_z = tl.arange($(3456))
    r82_value = tl.where(tl.load($(candidate_vgt_12_ptr) + ((((r82_z / $(36)) % $(12)) * $(36)) + (r82_z % $(36))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_35_ptr) + ((((r82_z / $(432)) % $(8)) * $(432)) + (((r82_z / $(36)) % $(12)) * $(36)) + (r82_z % $(36))), dtype=tl.float32))
    r82_value = r82_value.to(tl.float32)
    r82_value = tl.broadcast_to(r82_value, [$(3456)])
    tl.store($(candidate_vmasked_fill_12_ptr) + r82_z, r82_value)
    r83_z = tl.arange($(3456))
    r83_lane = tl.arange($(36))
    r83_row_values = tl.load($(candidate_vmasked_fill_12_ptr) + ((((r83_z[:, None] / $(432)) % $(8)) * $(432)) + (((r83_z[:, None] / $(36)) % $(12)) * $(36)) + (r83_lane[None, :] % $(36))), dtype=tl.float32)
    r83_maximum = tl.max(r83_row_values, axis=1, keep_dims=true)
    r83_weights = tl.exp(r83_row_values - r83_maximum)
    r83_total = tl.sum(r83_weights, axis=1)
    r83_maximum = tl.max(r83_row_values, axis=1)
    r83_value = tl.exp(tl.load($(candidate_vmasked_fill_12_ptr) + ((((r83_z / $(432)) % $(8)) * $(432)) + (((r83_z / $(36)) % $(12)) * $(36)) + (r83_z % $(36))), dtype=tl.float32) - r83_maximum) / r83_total
    r83_value = r83_value.to(tl.float32)
    r83_value = tl.broadcast_to(r83_value, [$(3456)])
    tl.store($(candidate_vsoftmax_12_ptr) + r83_z, r83_value)
    r84_z = tl.arange($(3456))
    r84_value = tl.load($(candidate_vsoftmax_12_ptr) + ((((r84_z / $(432)) % $(8)) * $(432)) + (((r84_z / $(36)) % $(12)) * $(36)) + (r84_z % $(36))), dtype=tl.float32)
    r84_value = r84_value.to(tl.bfloat16)
    r84_value = tl.broadcast_to(r84_value, [$(3456)])
    tl.store($(candidate_vto_copy_91_ptr) + r84_z, r84_value)
    r85_row = tl.arange($(12))
    r85_col = tl.arange($(128))
    r85_lane = tl.arange($(36))
    for r85_group in range($(8)) {
      r85_left = tl.load($(candidate_vto_copy_91_ptr) + (((((((r85_group % $(8)) * $(432)) + (r85_row[:, None] * $(36)) + r85_lane[None, :]) / $(432)) % $(8)) * $(432)) + ((((((r85_group % $(8)) * $(432)) + (r85_row[:, None] * $(36)) + r85_lane[None, :]) / $(36)) % $(12)) * $(36)) + ((((r85_group % $(8)) * $(432)) + (r85_row[:, None] * $(36)) + r85_lane[None, :]) % $(36))), dtype=tl.bfloat16)
      r85_right = tl.load($(candidate_vbmm_23_ptr) + (((((((r85_group % $(8)) * $(4608)) + (r85_lane[:, None] * $(128)) + r85_col[None, :]) / $(128)) % $(36)) * $(1024)) + ((((((r85_group % $(8)) * $(4608)) + (r85_lane[:, None] * $(128)) + r85_col[None, :]) / $(4608)) % $(8)) * $(128)) + ((((r85_group % $(8)) * $(4608)) + (r85_lane[:, None] * $(128)) + r85_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r85_value = tl.dot(r85_left, r85_right)
      r85_value = r85_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_25_ptr) + ((r85_group * $(1536)) + (r85_row[:, None] * $(128)) + r85_col[None, :]), r85_value)
    }
    r86_z = tl.arange($(96))
    r86_lane = tl.arange($(36))
    r86_row_values = tl.load($(candidate_vmasked_fill_12_ptr) + ((((r86_z[:, None] / $(12)) % $(8)) * $(432)) + ((r86_z[:, None] % $(12)) * $(36)) + (r86_lane[None, :] % $(36))), dtype=tl.float32)
    r86_maximum = tl.max(r86_row_values, axis=1, keep_dims=true)
    r86_weights = tl.exp(r86_row_values - r86_maximum)
    r86_total = tl.sum(r86_weights, axis=1)
    r86_maximum = tl.max(r86_row_values, axis=1)
    r86_value = r86_maximum + tl.log(r86_total)
    r86_value = r86_value.to(tl.float32)
    r86_value = tl.broadcast_to(r86_value, [$(96)])
    tl.store($(candidate_vlogsumexp_12_ptr) + r86_z, r86_value)
    r87_z = tl.arange($(6144))
    r87_value = tl.load($(candidate_vcat_11_ptr) + ((((r87_z / $(192)) % $(4)) * $(1536)) + (((r87_z / $(768)) % $(8)) * $(192)) + (r87_z % $(192)) + $(18432)), dtype=tl.bfloat16)
    r87_value = r87_value.to(tl.float32)
    r87_value = tl.broadcast_to(r87_value, [$(6144)])
    tl.store($(candidate_vto_copy_92_ptr) + r87_z, r87_value)
    r88_z = tl.arange($(18432))
    r88_value = tl.load($(candidate_vcat_12_ptr) + ((((r88_z / $(192)) % $(12)) * $(1536)) + (((r88_z / $(2304)) % $(8)) * $(192)) + (r88_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r88_value = r88_value.to(tl.float32)
    r88_value = tl.broadcast_to(r88_value, [$(18432)])
    tl.store($(candidate_vto_copy_93_ptr) + r88_z, r88_value)
    r89_row = tl.arange($(4))
    r89_col = tl.arange($(12))
    r89_lane = tl.arange($(192))
    for r89_group in range($(8)) {
      r89_left = tl.load($(candidate_vto_copy_92_ptr) + (((((((r89_group % $(8)) * $(768)) + (r89_row[:, None] * $(192)) + r89_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r89_group % $(8)) * $(768)) + (r89_row[:, None] * $(192)) + r89_lane[None, :]) / $(192)) % $(4)) * $(192)) + ((((r89_group % $(8)) * $(768)) + (r89_row[:, None] * $(192)) + r89_lane[None, :]) % $(192))), dtype=tl.float32)
      r89_right = tl.load($(candidate_vto_copy_93_ptr) + (((((((r89_group % $(8)) * $(2304)) + (r89_lane[:, None] * $(12)) + r89_col[None, :]) / $(2304)) % $(8)) * $(2304)) + (((((r89_group % $(8)) * $(2304)) + (r89_lane[:, None] * $(12)) + r89_col[None, :]) % $(12)) * $(192)) + (((((r89_group % $(8)) * $(2304)) + (r89_lane[:, None] * $(12)) + r89_col[None, :]) / $(12)) % $(192))), dtype=tl.float32)
      r89_value = tl.dot(r89_left, r89_right)
      r89_value = r89_value.to(tl.float32)
      tl.store($(candidate_vbmm_26_ptr) + ((r89_group * $(48)) + (r89_row[:, None] * $(12)) + r89_col[None, :]), r89_value)
    }
    r90_z = tl.arange($(384))
    r90_value = (tl.load($(candidate_vbmm_26_ptr) + ((((r90_z / $(48)) % $(8)) * $(48)) + (((r90_z / $(12)) % $(4)) * $(12)) + (r90_z % $(12))), dtype=tl.float32) * 0.07216878364870322)
    r90_value = r90_value.to(tl.float32)
    r90_value = tl.broadcast_to(r90_value, [$(384)])
    tl.store($(candidate_vmul_36_ptr) + r90_z, r90_value)
    r91_z = tl.arange($(4))
    r91_value = (tl.cast(r91_z, tl.int64) + tl.full([], $(12), dtype=tl.int64))
    r91_value = tl.broadcast_to(r91_value, [$(4)])
    tl.store($(candidate_vadd_33_ptr) + r91_z, r91_value)
    r92_z = tl.arange($(4))
    r92_value = (tl.load($(candidate_vadd_33_ptr) + r92_z, dtype=tl.int64) - tl.full([], $(4), dtype=tl.int64))
    r92_value = tl.broadcast_to(r92_value, [$(4)])
    tl.store($(candidate_vsub_22_ptr) + r92_z, r92_value)
    r93_z = tl.arange($(48))
    r93_value = (tl.cast((r93_z % $(12)), tl.int64) > tl.load($(candidate_vsub_22_ptr) + ((r93_z / $(12)) % $(4)), dtype=tl.int64))
    r93_value = tl.broadcast_to(r93_value, [$(48)])
    tl.store($(candidate_vgt_13_ptr) + r93_z, r93_value)
    r94_z = tl.arange($(384))
    r94_value = tl.where(tl.load($(candidate_vgt_13_ptr) + ((((r94_z / $(12)) % $(4)) * $(12)) + (r94_z % $(12))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_36_ptr) + ((((r94_z / $(48)) % $(8)) * $(48)) + (((r94_z / $(12)) % $(4)) * $(12)) + (r94_z % $(12))), dtype=tl.float32))
    r94_value = r94_value.to(tl.float32)
    r94_value = tl.broadcast_to(r94_value, [$(384)])
    tl.store($(candidate_vmasked_fill_13_ptr) + r94_z, r94_value)
    r95_z = tl.arange($(384))
    r95_lane = tl.arange($(12))
    r95_row_values = tl.load($(candidate_vmasked_fill_13_ptr) + ((((r95_z[:, None] / $(48)) % $(8)) * $(48)) + (((r95_z[:, None] / $(12)) % $(4)) * $(12)) + (r95_lane[None, :] % $(12))), dtype=tl.float32)
    r95_maximum = tl.max(r95_row_values, axis=1, keep_dims=true)
    r95_weights = tl.exp(r95_row_values - r95_maximum)
    r95_total = tl.sum(r95_weights, axis=1)
    r95_maximum = tl.max(r95_row_values, axis=1)
    r95_value = tl.exp(tl.load($(candidate_vmasked_fill_13_ptr) + ((((r95_z / $(48)) % $(8)) * $(48)) + (((r95_z / $(12)) % $(4)) * $(12)) + (r95_z % $(12))), dtype=tl.float32) - r95_maximum) / r95_total
    r95_value = r95_value.to(tl.float32)
    r95_value = tl.broadcast_to(r95_value, [$(384)])
    tl.store($(candidate_vsoftmax_13_ptr) + r95_z, r95_value)
    r96_z = tl.arange($(384))
    r96_value = tl.load($(candidate_vsoftmax_13_ptr) + ((((r96_z / $(48)) % $(8)) * $(48)) + (((r96_z / $(12)) % $(4)) * $(12)) + (r96_z % $(12))), dtype=tl.float32)
    r96_value = r96_value.to(tl.bfloat16)
    r96_value = tl.broadcast_to(r96_value, [$(384)])
    tl.store($(candidate_vto_copy_94_ptr) + r96_z, r96_value)
    r97_row = tl.arange($(4))
    r97_col = tl.arange($(128))
    r97_lane = tl.arange($(12))
    for r97_group in range($(8)) {
      r97_left = tl.load($(candidate_vto_copy_94_ptr) + (((((((r97_group % $(8)) * $(48)) + (r97_row[:, None] * $(12)) + r97_lane[None, :]) / $(48)) % $(8)) * $(48)) + ((((((r97_group % $(8)) * $(48)) + (r97_row[:, None] * $(12)) + r97_lane[None, :]) / $(12)) % $(4)) * $(12)) + ((((r97_group % $(8)) * $(48)) + (r97_row[:, None] * $(12)) + r97_lane[None, :]) % $(12))), dtype=tl.bfloat16)
      r97_right = tl.load($(candidate_vbmm_23_ptr) + (((((((r97_group % $(8)) * $(1536)) + (r97_lane[:, None] * $(128)) + r97_col[None, :]) / $(128)) % $(12)) * $(1024)) + ((((((r97_group % $(8)) * $(1536)) + (r97_lane[:, None] * $(128)) + r97_col[None, :]) / $(1536)) % $(8)) * $(128)) + ((((r97_group % $(8)) * $(1536)) + (r97_lane[:, None] * $(128)) + r97_col[None, :]) % $(128)) + $(49152)), dtype=tl.bfloat16)
      r97_value = tl.dot(r97_left, r97_right)
      r97_value = r97_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_27_ptr) + ((r97_group * $(512)) + (r97_row[:, None] * $(128)) + r97_col[None, :]), r97_value)
    }
    r98_z = tl.arange($(32))
    r98_lane = tl.arange($(12))
    r98_row_values = tl.load($(candidate_vmasked_fill_13_ptr) + ((((r98_z[:, None] / $(4)) % $(8)) * $(48)) + ((r98_z[:, None] % $(4)) * $(12)) + (r98_lane[None, :] % $(12))), dtype=tl.float32)
    r98_maximum = tl.max(r98_row_values, axis=1, keep_dims=true)
    r98_weights = tl.exp(r98_row_values - r98_maximum)
    r98_total = tl.sum(r98_weights, axis=1)
    r98_maximum = tl.max(r98_row_values, axis=1)
    r98_value = r98_maximum + tl.log(r98_total)
    r98_value = r98_value.to(tl.float32)
    r98_value = tl.broadcast_to(r98_value, [$(32)])
    tl.store($(candidate_vlogsumexp_13_ptr) + r98_z, r98_value)
    r99_z = tl.arange($(24576))
    r99_value = tl.load($(candidate_vcat_11_ptr) + ((((r99_z / $(192)) % $(16)) * $(1536)) + (((r99_z / $(3072)) % $(8)) * $(192)) + (r99_z % $(192)) + $(24576)), dtype=tl.bfloat16)
    r99_value = r99_value.to(tl.float32)
    r99_value = tl.broadcast_to(r99_value, [$(24576)])
    tl.store($(candidate_vto_copy_95_ptr) + r99_z, r99_value)
    r100_z = tl.arange($(73728))
    r100_value = tl.load($(candidate_vcat_12_ptr) + ((((r100_z / $(192)) % $(48)) * $(1536)) + (((r100_z / $(9216)) % $(8)) * $(192)) + (r100_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r100_value = r100_value.to(tl.float32)
    r100_value = tl.broadcast_to(r100_value, [$(73728)])
    tl.store($(candidate_vto_copy_96_ptr) + r100_z, r100_value)
    r101_row = tl.arange($(16))
    r101_col = tl.arange($(48))
    r101_lane = tl.arange($(192))
    for r101_group in range($(8)) {
      r101_left = tl.load($(candidate_vto_copy_95_ptr) + (((((((r101_group % $(8)) * $(3072)) + (r101_row[:, None] * $(192)) + r101_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r101_group % $(8)) * $(3072)) + (r101_row[:, None] * $(192)) + r101_lane[None, :]) / $(192)) % $(16)) * $(192)) + ((((r101_group % $(8)) * $(3072)) + (r101_row[:, None] * $(192)) + r101_lane[None, :]) % $(192))), dtype=tl.float32)
      r101_right = tl.load($(candidate_vto_copy_96_ptr) + (((((((r101_group % $(8)) * $(9216)) + (r101_lane[:, None] * $(48)) + r101_col[None, :]) / $(9216)) % $(8)) * $(9216)) + (((((r101_group % $(8)) * $(9216)) + (r101_lane[:, None] * $(48)) + r101_col[None, :]) % $(48)) * $(192)) + (((((r101_group % $(8)) * $(9216)) + (r101_lane[:, None] * $(48)) + r101_col[None, :]) / $(48)) % $(192))), dtype=tl.float32)
      r101_value = tl.dot(r101_left, r101_right)
      r101_value = r101_value.to(tl.float32)
      tl.store($(candidate_vbmm_28_ptr) + ((r101_group * $(768)) + (r101_row[:, None] * $(48)) + r101_col[None, :]), r101_value)
    }
    r102_z = tl.arange($(6144))
    r102_value = (tl.load($(candidate_vbmm_28_ptr) + ((((r102_z / $(768)) % $(8)) * $(768)) + (((r102_z / $(48)) % $(16)) * $(48)) + (r102_z % $(48))), dtype=tl.float32) * 0.07216878364870322)
    r102_value = r102_value.to(tl.float32)
    r102_value = tl.broadcast_to(r102_value, [$(6144)])
    tl.store($(candidate_vmul_37_ptr) + r102_z, r102_value)
    r103_z = tl.arange($(16))
    r103_value = (tl.cast(r103_z, tl.int64) + tl.full([], $(48), dtype=tl.int64))
    r103_value = tl.broadcast_to(r103_value, [$(16)])
    tl.store($(candidate_vadd_34_ptr) + r103_z, r103_value)
    r104_z = tl.arange($(16))
    r104_value = (tl.load($(candidate_vadd_34_ptr) + r104_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r104_value = tl.broadcast_to(r104_value, [$(16)])
    tl.store($(candidate_vsub_23_ptr) + r104_z, r104_value)
    r105_z = tl.arange($(768))
    r105_value = (tl.cast((r105_z % $(48)), tl.int64) > tl.load($(candidate_vsub_23_ptr) + ((r105_z / $(48)) % $(16)), dtype=tl.int64))
    r105_value = tl.broadcast_to(r105_value, [$(768)])
    tl.store($(candidate_vgt_14_ptr) + r105_z, r105_value)
    r106_z = tl.arange($(6144))
    r106_value = tl.where(tl.load($(candidate_vgt_14_ptr) + ((((r106_z / $(48)) % $(16)) * $(48)) + (r106_z % $(48))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_37_ptr) + ((((r106_z / $(768)) % $(8)) * $(768)) + (((r106_z / $(48)) % $(16)) * $(48)) + (r106_z % $(48))), dtype=tl.float32))
    r106_value = r106_value.to(tl.float32)
    r106_value = tl.broadcast_to(r106_value, [$(6144)])
    tl.store($(candidate_vmasked_fill_14_ptr) + r106_z, r106_value)
    r107_z = tl.arange($(6144))
    r107_lane = tl.arange($(48))
    r107_row_values = tl.load($(candidate_vmasked_fill_14_ptr) + ((((r107_z[:, None] / $(768)) % $(8)) * $(768)) + (((r107_z[:, None] / $(48)) % $(16)) * $(48)) + (r107_lane[None, :] % $(48))), dtype=tl.float32)
    r107_maximum = tl.max(r107_row_values, axis=1, keep_dims=true)
    r107_weights = tl.exp(r107_row_values - r107_maximum)
    r107_total = tl.sum(r107_weights, axis=1)
    r107_maximum = tl.max(r107_row_values, axis=1)
    r107_value = tl.exp(tl.load($(candidate_vmasked_fill_14_ptr) + ((((r107_z / $(768)) % $(8)) * $(768)) + (((r107_z / $(48)) % $(16)) * $(48)) + (r107_z % $(48))), dtype=tl.float32) - r107_maximum) / r107_total
    r107_value = r107_value.to(tl.float32)
    r107_value = tl.broadcast_to(r107_value, [$(6144)])
    tl.store($(candidate_vsoftmax_14_ptr) + r107_z, r107_value)
    r108_z = tl.arange($(6144))
    r108_value = tl.load($(candidate_vsoftmax_14_ptr) + ((((r108_z / $(768)) % $(8)) * $(768)) + (((r108_z / $(48)) % $(16)) * $(48)) + (r108_z % $(48))), dtype=tl.float32)
    r108_value = r108_value.to(tl.bfloat16)
    r108_value = tl.broadcast_to(r108_value, [$(6144)])
    tl.store($(candidate_vto_copy_97_ptr) + r108_z, r108_value)
    r109_row = tl.arange($(16))
    r109_col = tl.arange($(128))
    r109_lane = tl.arange($(48))
    for r109_group in range($(8)) {
      r109_left = tl.load($(candidate_vto_copy_97_ptr) + (((((((r109_group % $(8)) * $(768)) + (r109_row[:, None] * $(48)) + r109_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r109_group % $(8)) * $(768)) + (r109_row[:, None] * $(48)) + r109_lane[None, :]) / $(48)) % $(16)) * $(48)) + ((((r109_group % $(8)) * $(768)) + (r109_row[:, None] * $(48)) + r109_lane[None, :]) % $(48))), dtype=tl.bfloat16)
      r109_right = tl.load($(candidate_vbmm_23_ptr) + (((((((r109_group % $(8)) * $(6144)) + (r109_lane[:, None] * $(128)) + r109_col[None, :]) / $(128)) % $(48)) * $(1024)) + ((((((r109_group % $(8)) * $(6144)) + (r109_lane[:, None] * $(128)) + r109_col[None, :]) / $(6144)) % $(8)) * $(128)) + ((((r109_group % $(8)) * $(6144)) + (r109_lane[:, None] * $(128)) + r109_col[None, :]) % $(128)) + $(65536)), dtype=tl.bfloat16)
      r109_value = tl.dot(r109_left, r109_right)
      r109_value = r109_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_29_ptr) + ((r109_group * $(2048)) + (r109_row[:, None] * $(128)) + r109_col[None, :]), r109_value)
    }
    r110_z = tl.arange($(128))
    r110_lane = tl.arange($(48))
    r110_row_values = tl.load($(candidate_vmasked_fill_14_ptr) + ((((r110_z[:, None] / $(16)) % $(8)) * $(768)) + ((r110_z[:, None] % $(16)) * $(48)) + (r110_lane[None, :] % $(48))), dtype=tl.float32)
    r110_maximum = tl.max(r110_row_values, axis=1, keep_dims=true)
    r110_weights = tl.exp(r110_row_values - r110_maximum)
    r110_total = tl.sum(r110_weights, axis=1)
    r110_maximum = tl.max(r110_row_values, axis=1)
    r110_value = r110_maximum + tl.log(r110_total)
    r110_value = r110_value.to(tl.float32)
    r110_value = tl.broadcast_to(r110_value, [$(128)])
    tl.store($(candidate_vlogsumexp_14_ptr) + r110_z, r110_value)
    r111_z = tl.arange($(32768))
    r111_value = tl.where(((r111_z / $(1024)) % $(32)) < $(12), tl.load($(candidate_vbmm_25_ptr) + ((((((((r111_z / $(128)) % $(8)) * $(1536)) + ((((r111_z / $(1024)) % $(32)) - $(0)) * $(128)) + (r111_z % $(128))) / $(1536)) % $(8)) * $(1536)) + ((((((r111_z / $(128)) % $(8)) * $(12)) + (((r111_z / $(1024)) % $(32)) - $(0))) % $(12)) * $(128)) + (r111_z % $(128))), dtype=tl.bfloat16), tl.where(((r111_z / $(1024)) % $(32)) < $(16), tl.load($(candidate_vbmm_27_ptr) + ((((((((r111_z / $(128)) % $(8)) * $(512)) + ((((r111_z / $(1024)) % $(32)) - $(12)) * $(128)) + (r111_z % $(128))) / $(512)) % $(8)) * $(512)) + ((((((r111_z / $(128)) % $(8)) * $(4)) + (((r111_z / $(1024)) % $(32)) - $(12))) % $(4)) * $(128)) + (r111_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vbmm_29_ptr) + ((((((((r111_z / $(128)) % $(8)) * $(2048)) + ((((r111_z / $(1024)) % $(32)) - $(16)) * $(128)) + (r111_z % $(128))) / $(2048)) % $(8)) * $(2048)) + ((((((r111_z / $(128)) % $(8)) * $(16)) + (((r111_z / $(1024)) % $(32)) - $(16))) % $(16)) * $(128)) + (r111_z % $(128))), dtype=tl.bfloat16)))
    r111_value = r111_value.to(tl.bfloat16)
    r111_value = tl.broadcast_to(r111_value, [$(32768)])
    tl.store($(candidate_vcat_13_ptr) + r111_z, r111_value)
    r112_z = tl.arange($(256))
    r112_value = tl.where(((r112_z / $(8)) % $(32)) < $(12), tl.load($(candidate_vlogsumexp_12_ptr) + (((r112_z % $(8)) * $(12)) + ((((r112_z / $(8)) % $(32)) - $(0)) % $(12))), dtype=tl.float32), tl.where(((r112_z / $(8)) % $(32)) < $(16), tl.load($(candidate_vlogsumexp_13_ptr) + (((r112_z % $(8)) * $(4)) + ((((r112_z / $(8)) % $(32)) - $(12)) % $(4))), dtype=tl.float32), tl.load($(candidate_vlogsumexp_14_ptr) + (((r112_z % $(8)) * $(16)) + ((((r112_z / $(8)) % $(32)) - $(16)) % $(16))), dtype=tl.float32)))
    r112_value = r112_value.to(tl.float32)
    r112_value = tl.broadcast_to(r112_value, [$(256)])
    tl.store($(candidate_vcat_14_ptr) + r112_z, r112_value)
    r113_z = tl.arange($(18432))
    r113_value = tl.load($(candidate_vcat_11_ptr) + ((((r113_z / $(192)) % $(12)) * $(1536)) + (((r113_z / $(2304)) % $(8)) * $(192)) + (r113_z % $(192))), dtype=tl.bfloat16)
    r113_value = r113_value.to(tl.float32)
    r113_value = tl.broadcast_to(r113_value, [$(18432)])
    tl.store($(candidate_vto_copy_98_ptr) + r113_z, r113_value)
    r114_z = tl.arange($(55296))
    r114_value = tl.load($(candidate_vcat_12_ptr) + ((((r114_z / $(192)) % $(36)) * $(1536)) + (((r114_z / $(6912)) % $(8)) * $(192)) + (r114_z % $(192))), dtype=tl.bfloat16)
    r114_value = r114_value.to(tl.float32)
    r114_value = tl.broadcast_to(r114_value, [$(55296)])
    tl.store($(candidate_vto_copy_99_ptr) + r114_z, r114_value)
    r115_z = tl.arange($(36864))
    r115_value = tl.load($(candidate_vbmm_23_ptr) + ((((r115_z / $(128)) % $(36)) * $(1024)) + (((r115_z / $(4608)) % $(8)) * $(128)) + (r115_z % $(128))), dtype=tl.bfloat16)
    r115_value = r115_value.to(tl.float32)
    r115_value = tl.broadcast_to(r115_value, [$(36864)])
    tl.store($(candidate_vto_copy_100_ptr) + r115_z, r115_value)
    r116_z = tl.arange($(12288))
    r116_value = tl.load($(candidate_vmm_28_ptr) + ((((r116_z / $(128)) % $(12)) * $(1024)) + (((r116_z / $(1536)) % $(8)) * $(128)) + (r116_z % $(128)) + $(98304)), dtype=tl.bfloat16)
    r116_value = r116_value.to(tl.float32)
    r116_value = tl.broadcast_to(r116_value, [$(12288)])
    tl.store($(candidate_vto_copy_101_ptr) + r116_z, r116_value)
    r117_z = tl.arange($(12288))
    r117_value = tl.load($(candidate_vcat_13_ptr) + ((((r117_z / $(128)) % $(12)) * $(1024)) + (((r117_z / $(1536)) % $(8)) * $(128)) + (r117_z % $(128))), dtype=tl.bfloat16)
    r117_value = r117_value.to(tl.float32)
    r117_value = tl.broadcast_to(r117_value, [$(12288)])
    tl.store($(candidate_vto_copy_102_ptr) + r117_z, r117_value)
    r118_row = tl.arange($(12))
    r118_col = tl.arange($(36))
    r118_lane = tl.arange($(192))
    for r118_group in range($(8)) {
      r118_left = tl.load($(candidate_vto_copy_98_ptr) + (((((((r118_group % $(8)) * $(2304)) + (r118_row[:, None] * $(192)) + r118_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r118_group % $(8)) * $(2304)) + (r118_row[:, None] * $(192)) + r118_lane[None, :]) / $(192)) % $(12)) * $(192)) + ((((r118_group % $(8)) * $(2304)) + (r118_row[:, None] * $(192)) + r118_lane[None, :]) % $(192))), dtype=tl.float32)
      r118_right = tl.load($(candidate_vto_copy_99_ptr) + (((((((r118_group % $(8)) * $(6912)) + (r118_lane[:, None] * $(36)) + r118_col[None, :]) / $(6912)) % $(8)) * $(6912)) + (((((r118_group % $(8)) * $(6912)) + (r118_lane[:, None] * $(36)) + r118_col[None, :]) % $(36)) * $(192)) + (((((r118_group % $(8)) * $(6912)) + (r118_lane[:, None] * $(36)) + r118_col[None, :]) / $(36)) % $(192))), dtype=tl.float32)
      r118_value = tl.dot(r118_left, r118_right)
      r118_value = r118_value.to(tl.float32)
      tl.store($(candidate_vbmm_30_ptr) + ((r118_group * $(432)) + (r118_row[:, None] * $(36)) + r118_col[None, :]), r118_value)
    }
    r119_z = tl.arange($(3456))
    r119_value = (tl.load($(candidate_vbmm_30_ptr) + ((((r119_z / $(432)) % $(8)) * $(432)) + (((r119_z / $(36)) % $(12)) * $(36)) + (r119_z % $(36))), dtype=tl.float32) * 0.07216878364870322)
    r119_value = r119_value.to(tl.float32)
    r119_value = tl.broadcast_to(r119_value, [$(3456)])
    tl.store($(candidate_vmul_38_ptr) + r119_z, r119_value)
    r120_z = tl.arange($(12))
    r120_value = (tl.cast(r120_z, tl.int64) + tl.full([], $(36), dtype=tl.int64))
    r120_value = tl.broadcast_to(r120_value, [$(12)])
    tl.store($(candidate_vadd_35_ptr) + r120_z, r120_value)
    r121_z = tl.arange($(12))
    r121_value = (tl.load($(candidate_vadd_35_ptr) + r121_z, dtype=tl.int64) - tl.full([], $(12), dtype=tl.int64))
    r121_value = tl.broadcast_to(r121_value, [$(12)])
    tl.store($(candidate_vsub_24_ptr) + r121_z, r121_value)
    r122_z = tl.arange($(432))
    r122_value = (tl.cast((r122_z % $(36)), tl.int64) > tl.load($(candidate_vsub_24_ptr) + ((r122_z / $(36)) % $(12)), dtype=tl.int64))
    r122_value = tl.broadcast_to(r122_value, [$(432)])
    tl.store($(candidate_vgt_15_ptr) + r122_z, r122_value)
    r123_z = tl.arange($(3456))
    r123_value = tl.where(tl.load($(candidate_vgt_15_ptr) + ((((r123_z / $(36)) % $(12)) * $(36)) + (r123_z % $(36))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_38_ptr) + ((((r123_z / $(432)) % $(8)) * $(432)) + (((r123_z / $(36)) % $(12)) * $(36)) + (r123_z % $(36))), dtype=tl.float32))
    r123_value = r123_value.to(tl.float32)
    r123_value = tl.broadcast_to(r123_value, [$(3456)])
    tl.store($(candidate_vmasked_fill_15_ptr) + r123_z, r123_value)
    r124_z = tl.arange($(3456))
    r124_value = (tl.load($(candidate_vmasked_fill_15_ptr) + ((((r124_z / $(432)) % $(8)) * $(432)) + (((r124_z / $(36)) % $(12)) * $(36)) + (r124_z % $(36))), dtype=tl.float32) - tl.load($(candidate_vcat_14_ptr) + ((((r124_z / $(36)) % $(12)) * $(8)) + ((r124_z / $(432)) % $(8))), dtype=tl.float32))
    r124_value = r124_value.to(tl.float32)
    r124_value = tl.broadcast_to(r124_value, [$(3456)])
    tl.store($(candidate_vsub_25_ptr) + r124_z, r124_value)
    r125_z = tl.arange($(3456))
    r125_value = tl.exp(tl.load($(candidate_vsub_25_ptr) + ((((r125_z / $(432)) % $(8)) * $(432)) + (((r125_z / $(36)) % $(12)) * $(36)) + (r125_z % $(36))), dtype=tl.float32))
    r125_value = r125_value.to(tl.float32)
    r125_value = tl.broadcast_to(r125_value, [$(3456)])
    tl.store($(candidate_vexp_ptr) + r125_z, r125_value)
    r126_z = tl.arange($(12288))
    r126_value = (tl.load($(candidate_vto_copy_101_ptr) + ((((r126_z / $(1536)) % $(8)) * $(1536)) + (((r126_z / $(128)) % $(12)) * $(128)) + (r126_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_102_ptr) + ((((r126_z / $(1536)) % $(8)) * $(1536)) + (((r126_z / $(128)) % $(12)) * $(128)) + (r126_z % $(128))), dtype=tl.float32))
    r126_value = r126_value.to(tl.float32)
    r126_value = tl.broadcast_to(r126_value, [$(12288)])
    tl.store($(candidate_vmul_39_ptr) + r126_z, r126_value)
    r127_z = tl.arange($(96))
    r127_lane = tl.arange($(128))
    r127_value = tl.sum(tl.load($(candidate_vmul_39_ptr) + ((((r127_z[:, None] / $(12)) % $(8)) * $(1536)) + ((r127_z[:, None] % $(12)) * $(128)) + r127_lane[None, :]), dtype=tl.float32), axis=1)
    r127_value = r127_value.to(tl.float32)
    r127_value = tl.broadcast_to(r127_value, [$(96)])
    tl.store($(candidate_vsum_1_ptr) + r127_z, r127_value)
    r128_row = tl.arange($(12))
    r128_col = tl.arange($(36))
    r128_lane = tl.arange($(128))
    for r128_group in range($(8)) {
      r128_left = tl.load($(candidate_vto_copy_101_ptr) + (((((((r128_group % $(8)) * $(1536)) + (r128_row[:, None] * $(128)) + r128_lane[None, :]) / $(1536)) % $(8)) * $(1536)) + ((((((r128_group % $(8)) * $(1536)) + (r128_row[:, None] * $(128)) + r128_lane[None, :]) / $(128)) % $(12)) * $(128)) + ((((r128_group % $(8)) * $(1536)) + (r128_row[:, None] * $(128)) + r128_lane[None, :]) % $(128))), dtype=tl.float32)
      r128_right = tl.load($(candidate_vto_copy_100_ptr) + (((((((r128_group % $(8)) * $(4608)) + (r128_lane[:, None] * $(36)) + r128_col[None, :]) / $(4608)) % $(8)) * $(4608)) + (((((r128_group % $(8)) * $(4608)) + (r128_lane[:, None] * $(36)) + r128_col[None, :]) % $(36)) * $(128)) + (((((r128_group % $(8)) * $(4608)) + (r128_lane[:, None] * $(36)) + r128_col[None, :]) / $(36)) % $(128))), dtype=tl.float32)
      r128_value = tl.dot(r128_left, r128_right)
      r128_value = r128_value.to(tl.float32)
      tl.store($(candidate_vbmm_31_ptr) + ((r128_group * $(432)) + (r128_row[:, None] * $(36)) + r128_col[None, :]), r128_value)
    }
    r129_z = tl.arange($(3456))
    r129_value = (tl.load($(candidate_vbmm_31_ptr) + ((((r129_z / $(432)) % $(8)) * $(432)) + (((r129_z / $(36)) % $(12)) * $(36)) + (r129_z % $(36))), dtype=tl.float32) - tl.load($(candidate_vsum_1_ptr) + ((((r129_z / $(432)) % $(8)) * $(12)) + ((r129_z / $(36)) % $(12))), dtype=tl.float32))
    r129_value = r129_value.to(tl.float32)
    r129_value = tl.broadcast_to(r129_value, [$(3456)])
    tl.store($(candidate_vsub_26_ptr) + r129_z, r129_value)
    r130_z = tl.arange($(3456))
    r130_value = (tl.load($(candidate_vexp_ptr) + ((((r130_z / $(432)) % $(8)) * $(432)) + (((r130_z / $(36)) % $(12)) * $(36)) + (r130_z % $(36))), dtype=tl.float32) * tl.load($(candidate_vsub_26_ptr) + ((((r130_z / $(432)) % $(8)) * $(432)) + (((r130_z / $(36)) % $(12)) * $(36)) + (r130_z % $(36))), dtype=tl.float32))
    r130_value = r130_value.to(tl.float32)
    r130_value = tl.broadcast_to(r130_value, [$(3456)])
    tl.store($(candidate_vmul_40_ptr) + r130_z, r130_value)
    r131_row = tl.arange($(12))
    r131_col = tl.arange($(192))
    r131_lane = tl.arange($(36))
    for r131_group in range($(8)) {
      r131_left = tl.load($(candidate_vmul_40_ptr) + (((((((r131_group % $(8)) * $(432)) + (r131_row[:, None] * $(36)) + r131_lane[None, :]) / $(432)) % $(8)) * $(432)) + ((((((r131_group % $(8)) * $(432)) + (r131_row[:, None] * $(36)) + r131_lane[None, :]) / $(36)) % $(12)) * $(36)) + ((((r131_group % $(8)) * $(432)) + (r131_row[:, None] * $(36)) + r131_lane[None, :]) % $(36))), dtype=tl.float32)
      r131_right = tl.load($(candidate_vto_copy_99_ptr) + (((((((r131_group % $(8)) * $(6912)) + (r131_lane[:, None] * $(192)) + r131_col[None, :]) / $(6912)) % $(8)) * $(6912)) + ((((((r131_group % $(8)) * $(6912)) + (r131_lane[:, None] * $(192)) + r131_col[None, :]) / $(192)) % $(36)) * $(192)) + ((((r131_group % $(8)) * $(6912)) + (r131_lane[:, None] * $(192)) + r131_col[None, :]) % $(192))), dtype=tl.float32)
      r131_value = tl.dot(r131_left, r131_right)
      r131_value = r131_value.to(tl.float32)
      tl.store($(candidate_vbmm_32_ptr) + ((r131_group * $(2304)) + (r131_row[:, None] * $(192)) + r131_col[None, :]), r131_value)
    }
    r132_z = tl.arange($(18432))
    r132_value = (tl.load($(candidate_vbmm_32_ptr) + ((((r132_z / $(2304)) % $(8)) * $(2304)) + (((r132_z / $(192)) % $(12)) * $(192)) + (r132_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r132_value = r132_value.to(tl.float32)
    r132_value = tl.broadcast_to(r132_value, [$(18432)])
    tl.store($(candidate_vmul_41_ptr) + r132_z, r132_value)
    r133_z = tl.arange($(18432))
    r133_value = tl.load($(candidate_vmul_41_ptr) + ((((r133_z / $(192)) % $(8)) * $(2304)) + (((r133_z / $(1536)) % $(12)) * $(192)) + (r133_z % $(192))), dtype=tl.float32)
    r133_value = r133_value.to(tl.bfloat16)
    r133_value = tl.broadcast_to(r133_value, [$(18432)])
    tl.store($(candidate_vto_copy_103_ptr) + r133_z, r133_value)
    r134_row = tl.arange($(36))
    r134_col = tl.arange($(192))
    r134_lane = tl.arange($(12))
    for r134_group in range($(8)) {
      r134_left = tl.load($(candidate_vmul_40_ptr) + (((((((r134_group % $(8)) * $(432)) + (r134_row[:, None] * $(12)) + r134_lane[None, :]) / $(432)) % $(8)) * $(432)) + (((((r134_group % $(8)) * $(432)) + (r134_row[:, None] * $(12)) + r134_lane[None, :]) % $(12)) * $(36)) + (((((r134_group % $(8)) * $(432)) + (r134_row[:, None] * $(12)) + r134_lane[None, :]) / $(12)) % $(36))), dtype=tl.float32)
      r134_right = tl.load($(candidate_vto_copy_98_ptr) + (((((((r134_group % $(8)) * $(2304)) + (r134_lane[:, None] * $(192)) + r134_col[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r134_group % $(8)) * $(2304)) + (r134_lane[:, None] * $(192)) + r134_col[None, :]) / $(192)) % $(12)) * $(192)) + ((((r134_group % $(8)) * $(2304)) + (r134_lane[:, None] * $(192)) + r134_col[None, :]) % $(192))), dtype=tl.float32)
      r134_value = tl.dot(r134_left, r134_right)
      r134_value = r134_value.to(tl.float32)
      tl.store($(candidate_vbmm_33_ptr) + ((r134_group * $(6912)) + (r134_row[:, None] * $(192)) + r134_col[None, :]), r134_value)
    }
    r135_z = tl.arange($(55296))
    r135_value = (tl.load($(candidate_vbmm_33_ptr) + ((((r135_z / $(6912)) % $(8)) * $(6912)) + (((r135_z / $(192)) % $(36)) * $(192)) + (r135_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r135_value = r135_value.to(tl.float32)
    r135_value = tl.broadcast_to(r135_value, [$(55296)])
    tl.store($(candidate_vmul_42_ptr) + r135_z, r135_value)
    r136_z = tl.arange($(55296))
    r136_value = tl.load($(candidate_vmul_42_ptr) + ((((r136_z / $(192)) % $(8)) * $(6912)) + (((r136_z / $(1536)) % $(36)) * $(192)) + (r136_z % $(192))), dtype=tl.float32)
    r136_value = r136_value.to(tl.bfloat16)
    r136_value = tl.broadcast_to(r136_value, [$(55296)])
    tl.store($(candidate_vto_copy_104_ptr) + r136_z, r136_value)
    r137_row = tl.arange($(36))
    r137_col = tl.arange($(128))
    r137_lane = tl.arange($(12))
    for r137_group in range($(8)) {
      r137_left = tl.load($(candidate_vexp_ptr) + (((((((r137_group % $(8)) * $(432)) + (r137_row[:, None] * $(12)) + r137_lane[None, :]) / $(432)) % $(8)) * $(432)) + (((((r137_group % $(8)) * $(432)) + (r137_row[:, None] * $(12)) + r137_lane[None, :]) % $(12)) * $(36)) + (((((r137_group % $(8)) * $(432)) + (r137_row[:, None] * $(12)) + r137_lane[None, :]) / $(12)) % $(36))), dtype=tl.float32)
      r137_right = tl.load($(candidate_vto_copy_101_ptr) + (((((((r137_group % $(8)) * $(1536)) + (r137_lane[:, None] * $(128)) + r137_col[None, :]) / $(1536)) % $(8)) * $(1536)) + ((((((r137_group % $(8)) * $(1536)) + (r137_lane[:, None] * $(128)) + r137_col[None, :]) / $(128)) % $(12)) * $(128)) + ((((r137_group % $(8)) * $(1536)) + (r137_lane[:, None] * $(128)) + r137_col[None, :]) % $(128))), dtype=tl.float32)
      r137_value = tl.dot(r137_left, r137_right)
      r137_value = r137_value.to(tl.float32)
      tl.store($(candidate_vbmm_34_ptr) + ((r137_group * $(4608)) + (r137_row[:, None] * $(128)) + r137_col[None, :]), r137_value)
    }
    r138_z = tl.arange($(36864))
    r138_value = tl.load($(candidate_vbmm_34_ptr) + ((((r138_z / $(128)) % $(8)) * $(4608)) + (((r138_z / $(1024)) % $(36)) * $(128)) + (r138_z % $(128))), dtype=tl.float32)
    r138_value = r138_value.to(tl.bfloat16)
    r138_value = tl.broadcast_to(r138_value, [$(36864)])
    tl.store($(candidate_vto_copy_105_ptr) + r138_z, r138_value)
    r139_z = tl.arange($(73728))
    r139_value = tl.where(((r139_z / $(1536)) % $(48)) < $(36), tl.load($(candidate_vto_copy_104_ptr) + ((((((r139_z / $(1536)) % $(48)) - $(0)) % $(36)) * $(1536)) + (((r139_z / $(192)) % $(8)) * $(192)) + (r139_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r139_value = r139_value.to(tl.bfloat16)
    r139_value = tl.broadcast_to(r139_value, [$(73728)])
    tl.store($(candidate_vcat_15_ptr) + r139_z, r139_value)
    r140_z = tl.arange($(49152))
    r140_value = tl.where(((r140_z / $(1024)) % $(48)) < $(36), tl.load($(candidate_vto_copy_105_ptr) + ((((((r140_z / $(1024)) % $(48)) - $(0)) % $(36)) * $(1024)) + (((r140_z / $(128)) % $(8)) * $(128)) + (r140_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r140_value = r140_value.to(tl.bfloat16)
    r140_value = tl.broadcast_to(r140_value, [$(49152)])
    tl.store($(candidate_vcat_16_ptr) + r140_z, r140_value)
    r141_z = tl.arange($(6144))
    r141_value = tl.load($(candidate_vcat_11_ptr) + ((((r141_z / $(192)) % $(4)) * $(1536)) + (((r141_z / $(768)) % $(8)) * $(192)) + (r141_z % $(192)) + $(18432)), dtype=tl.bfloat16)
    r141_value = r141_value.to(tl.float32)
    r141_value = tl.broadcast_to(r141_value, [$(6144)])
    tl.store($(candidate_vto_copy_106_ptr) + r141_z, r141_value)
    r142_z = tl.arange($(18432))
    r142_value = tl.load($(candidate_vcat_12_ptr) + ((((r142_z / $(192)) % $(12)) * $(1536)) + (((r142_z / $(2304)) % $(8)) * $(192)) + (r142_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r142_value = r142_value.to(tl.float32)
    r142_value = tl.broadcast_to(r142_value, [$(18432)])
    tl.store($(candidate_vto_copy_107_ptr) + r142_z, r142_value)
    r143_z = tl.arange($(12288))
    r143_value = tl.load($(candidate_vbmm_23_ptr) + ((((r143_z / $(128)) % $(12)) * $(1024)) + (((r143_z / $(1536)) % $(8)) * $(128)) + (r143_z % $(128)) + $(49152)), dtype=tl.bfloat16)
    r143_value = r143_value.to(tl.float32)
    r143_value = tl.broadcast_to(r143_value, [$(12288)])
    tl.store($(candidate_vto_copy_108_ptr) + r143_z, r143_value)
    r144_z = tl.arange($(4096))
    r144_value = tl.load($(candidate_vmm_28_ptr) + ((((r144_z / $(128)) % $(4)) * $(1024)) + (((r144_z / $(512)) % $(8)) * $(128)) + (r144_z % $(128)) + $(110592)), dtype=tl.bfloat16)
    r144_value = r144_value.to(tl.float32)
    r144_value = tl.broadcast_to(r144_value, [$(4096)])
    tl.store($(candidate_vto_copy_109_ptr) + r144_z, r144_value)
    r145_z = tl.arange($(4096))
    r145_value = tl.load($(candidate_vcat_13_ptr) + ((((r145_z / $(128)) % $(4)) * $(1024)) + (((r145_z / $(512)) % $(8)) * $(128)) + (r145_z % $(128)) + $(12288)), dtype=tl.bfloat16)
    r145_value = r145_value.to(tl.float32)
    r145_value = tl.broadcast_to(r145_value, [$(4096)])
    tl.store($(candidate_vto_copy_110_ptr) + r145_z, r145_value)
    r146_row = tl.arange($(4))
    r146_col = tl.arange($(12))
    r146_lane = tl.arange($(192))
    for r146_group in range($(8)) {
      r146_left = tl.load($(candidate_vto_copy_106_ptr) + (((((((r146_group % $(8)) * $(768)) + (r146_row[:, None] * $(192)) + r146_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r146_group % $(8)) * $(768)) + (r146_row[:, None] * $(192)) + r146_lane[None, :]) / $(192)) % $(4)) * $(192)) + ((((r146_group % $(8)) * $(768)) + (r146_row[:, None] * $(192)) + r146_lane[None, :]) % $(192))), dtype=tl.float32)
      r146_right = tl.load($(candidate_vto_copy_107_ptr) + (((((((r146_group % $(8)) * $(2304)) + (r146_lane[:, None] * $(12)) + r146_col[None, :]) / $(2304)) % $(8)) * $(2304)) + (((((r146_group % $(8)) * $(2304)) + (r146_lane[:, None] * $(12)) + r146_col[None, :]) % $(12)) * $(192)) + (((((r146_group % $(8)) * $(2304)) + (r146_lane[:, None] * $(12)) + r146_col[None, :]) / $(12)) % $(192))), dtype=tl.float32)
      r146_value = tl.dot(r146_left, r146_right)
      r146_value = r146_value.to(tl.float32)
      tl.store($(candidate_vbmm_35_ptr) + ((r146_group * $(48)) + (r146_row[:, None] * $(12)) + r146_col[None, :]), r146_value)
    }
    r147_z = tl.arange($(384))
    r147_value = (tl.load($(candidate_vbmm_35_ptr) + ((((r147_z / $(48)) % $(8)) * $(48)) + (((r147_z / $(12)) % $(4)) * $(12)) + (r147_z % $(12))), dtype=tl.float32) * 0.07216878364870322)
    r147_value = r147_value.to(tl.float32)
    r147_value = tl.broadcast_to(r147_value, [$(384)])
    tl.store($(candidate_vmul_43_ptr) + r147_z, r147_value)
    r148_z = tl.arange($(4))
    r148_value = (tl.cast(r148_z, tl.int64) + tl.full([], $(12), dtype=tl.int64))
    r148_value = tl.broadcast_to(r148_value, [$(4)])
    tl.store($(candidate_vadd_36_ptr) + r148_z, r148_value)
    r149_z = tl.arange($(4))
    r149_value = (tl.load($(candidate_vadd_36_ptr) + r149_z, dtype=tl.int64) - tl.full([], $(4), dtype=tl.int64))
    r149_value = tl.broadcast_to(r149_value, [$(4)])
    tl.store($(candidate_vsub_27_ptr) + r149_z, r149_value)
    r150_z = tl.arange($(48))
    r150_value = (tl.cast((r150_z % $(12)), tl.int64) > tl.load($(candidate_vsub_27_ptr) + ((r150_z / $(12)) % $(4)), dtype=tl.int64))
    r150_value = tl.broadcast_to(r150_value, [$(48)])
    tl.store($(candidate_vgt_16_ptr) + r150_z, r150_value)
    r151_z = tl.arange($(384))
    r151_value = tl.where(tl.load($(candidate_vgt_16_ptr) + ((((r151_z / $(12)) % $(4)) * $(12)) + (r151_z % $(12))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_43_ptr) + ((((r151_z / $(48)) % $(8)) * $(48)) + (((r151_z / $(12)) % $(4)) * $(12)) + (r151_z % $(12))), dtype=tl.float32))
    r151_value = r151_value.to(tl.float32)
    r151_value = tl.broadcast_to(r151_value, [$(384)])
    tl.store($(candidate_vmasked_fill_16_ptr) + r151_z, r151_value)
    r152_z = tl.arange($(384))
    r152_value = (tl.load($(candidate_vmasked_fill_16_ptr) + ((((r152_z / $(48)) % $(8)) * $(48)) + (((r152_z / $(12)) % $(4)) * $(12)) + (r152_z % $(12))), dtype=tl.float32) - tl.load($(candidate_vcat_14_ptr) + ((((r152_z / $(12)) % $(4)) * $(8)) + ((r152_z / $(48)) % $(8)) + $(96)), dtype=tl.float32))
    r152_value = r152_value.to(tl.float32)
    r152_value = tl.broadcast_to(r152_value, [$(384)])
    tl.store($(candidate_vsub_28_ptr) + r152_z, r152_value)
    r153_z = tl.arange($(384))
    r153_value = tl.exp(tl.load($(candidate_vsub_28_ptr) + ((((r153_z / $(48)) % $(8)) * $(48)) + (((r153_z / $(12)) % $(4)) * $(12)) + (r153_z % $(12))), dtype=tl.float32))
    r153_value = r153_value.to(tl.float32)
    r153_value = tl.broadcast_to(r153_value, [$(384)])
    tl.store($(candidate_vexp_1_ptr) + r153_z, r153_value)
    r154_z = tl.arange($(4096))
    r154_value = (tl.load($(candidate_vto_copy_109_ptr) + ((((r154_z / $(512)) % $(8)) * $(512)) + (((r154_z / $(128)) % $(4)) * $(128)) + (r154_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_110_ptr) + ((((r154_z / $(512)) % $(8)) * $(512)) + (((r154_z / $(128)) % $(4)) * $(128)) + (r154_z % $(128))), dtype=tl.float32))
    r154_value = r154_value.to(tl.float32)
    r154_value = tl.broadcast_to(r154_value, [$(4096)])
    tl.store($(candidate_vmul_44_ptr) + r154_z, r154_value)
    r155_z = tl.arange($(32))
    r155_lane = tl.arange($(128))
    r155_value = tl.sum(tl.load($(candidate_vmul_44_ptr) + ((((r155_z[:, None] / $(4)) % $(8)) * $(512)) + ((r155_z[:, None] % $(4)) * $(128)) + r155_lane[None, :]), dtype=tl.float32), axis=1)
    r155_value = r155_value.to(tl.float32)
    r155_value = tl.broadcast_to(r155_value, [$(32)])
    tl.store($(candidate_vsum_2_ptr) + r155_z, r155_value)
    r156_row = tl.arange($(4))
    r156_col = tl.arange($(12))
    r156_lane = tl.arange($(128))
    for r156_group in range($(8)) {
      r156_left = tl.load($(candidate_vto_copy_109_ptr) + (((((((r156_group % $(8)) * $(512)) + (r156_row[:, None] * $(128)) + r156_lane[None, :]) / $(512)) % $(8)) * $(512)) + ((((((r156_group % $(8)) * $(512)) + (r156_row[:, None] * $(128)) + r156_lane[None, :]) / $(128)) % $(4)) * $(128)) + ((((r156_group % $(8)) * $(512)) + (r156_row[:, None] * $(128)) + r156_lane[None, :]) % $(128))), dtype=tl.float32)
      r156_right = tl.load($(candidate_vto_copy_108_ptr) + (((((((r156_group % $(8)) * $(1536)) + (r156_lane[:, None] * $(12)) + r156_col[None, :]) / $(1536)) % $(8)) * $(1536)) + (((((r156_group % $(8)) * $(1536)) + (r156_lane[:, None] * $(12)) + r156_col[None, :]) % $(12)) * $(128)) + (((((r156_group % $(8)) * $(1536)) + (r156_lane[:, None] * $(12)) + r156_col[None, :]) / $(12)) % $(128))), dtype=tl.float32)
      r156_value = tl.dot(r156_left, r156_right)
      r156_value = r156_value.to(tl.float32)
      tl.store($(candidate_vbmm_36_ptr) + ((r156_group * $(48)) + (r156_row[:, None] * $(12)) + r156_col[None, :]), r156_value)
    }
    r157_z = tl.arange($(384))
    r157_value = (tl.load($(candidate_vbmm_36_ptr) + ((((r157_z / $(48)) % $(8)) * $(48)) + (((r157_z / $(12)) % $(4)) * $(12)) + (r157_z % $(12))), dtype=tl.float32) - tl.load($(candidate_vsum_2_ptr) + ((((r157_z / $(48)) % $(8)) * $(4)) + ((r157_z / $(12)) % $(4))), dtype=tl.float32))
    r157_value = r157_value.to(tl.float32)
    r157_value = tl.broadcast_to(r157_value, [$(384)])
    tl.store($(candidate_vsub_29_ptr) + r157_z, r157_value)
    r158_z = tl.arange($(384))
    r158_value = (tl.load($(candidate_vexp_1_ptr) + ((((r158_z / $(48)) % $(8)) * $(48)) + (((r158_z / $(12)) % $(4)) * $(12)) + (r158_z % $(12))), dtype=tl.float32) * tl.load($(candidate_vsub_29_ptr) + ((((r158_z / $(48)) % $(8)) * $(48)) + (((r158_z / $(12)) % $(4)) * $(12)) + (r158_z % $(12))), dtype=tl.float32))
    r158_value = r158_value.to(tl.float32)
    r158_value = tl.broadcast_to(r158_value, [$(384)])
    tl.store($(candidate_vmul_45_ptr) + r158_z, r158_value)
    r159_row = tl.arange($(4))
    r159_col = tl.arange($(192))
    r159_lane = tl.arange($(12))
    for r159_group in range($(8)) {
      r159_left = tl.load($(candidate_vmul_45_ptr) + (((((((r159_group % $(8)) * $(48)) + (r159_row[:, None] * $(12)) + r159_lane[None, :]) / $(48)) % $(8)) * $(48)) + ((((((r159_group % $(8)) * $(48)) + (r159_row[:, None] * $(12)) + r159_lane[None, :]) / $(12)) % $(4)) * $(12)) + ((((r159_group % $(8)) * $(48)) + (r159_row[:, None] * $(12)) + r159_lane[None, :]) % $(12))), dtype=tl.float32)
      r159_right = tl.load($(candidate_vto_copy_107_ptr) + (((((((r159_group % $(8)) * $(2304)) + (r159_lane[:, None] * $(192)) + r159_col[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r159_group % $(8)) * $(2304)) + (r159_lane[:, None] * $(192)) + r159_col[None, :]) / $(192)) % $(12)) * $(192)) + ((((r159_group % $(8)) * $(2304)) + (r159_lane[:, None] * $(192)) + r159_col[None, :]) % $(192))), dtype=tl.float32)
      r159_value = tl.dot(r159_left, r159_right)
      r159_value = r159_value.to(tl.float32)
      tl.store($(candidate_vbmm_37_ptr) + ((r159_group * $(768)) + (r159_row[:, None] * $(192)) + r159_col[None, :]), r159_value)
    }
    r160_z = tl.arange($(6144))
    r160_value = (tl.load($(candidate_vbmm_37_ptr) + ((((r160_z / $(768)) % $(8)) * $(768)) + (((r160_z / $(192)) % $(4)) * $(192)) + (r160_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r160_value = r160_value.to(tl.float32)
    r160_value = tl.broadcast_to(r160_value, [$(6144)])
    tl.store($(candidate_vmul_46_ptr) + r160_z, r160_value)
    r161_z = tl.arange($(6144))
    r161_value = tl.load($(candidate_vmul_46_ptr) + ((((r161_z / $(192)) % $(8)) * $(768)) + (((r161_z / $(1536)) % $(4)) * $(192)) + (r161_z % $(192))), dtype=tl.float32)
    r161_value = r161_value.to(tl.bfloat16)
    r161_value = tl.broadcast_to(r161_value, [$(6144)])
    tl.store($(candidate_vto_copy_111_ptr) + r161_z, r161_value)
    r162_row = tl.arange($(12))
    r162_col = tl.arange($(192))
    r162_lane = tl.arange($(4))
    for r162_group in range($(8)) {
      r162_left = tl.load($(candidate_vmul_45_ptr) + (((((((r162_group % $(8)) * $(48)) + (r162_row[:, None] * $(4)) + r162_lane[None, :]) / $(48)) % $(8)) * $(48)) + (((((r162_group % $(8)) * $(48)) + (r162_row[:, None] * $(4)) + r162_lane[None, :]) % $(4)) * $(12)) + (((((r162_group % $(8)) * $(48)) + (r162_row[:, None] * $(4)) + r162_lane[None, :]) / $(4)) % $(12))), dtype=tl.float32)
      r162_right = tl.load($(candidate_vto_copy_106_ptr) + (((((((r162_group % $(8)) * $(768)) + (r162_lane[:, None] * $(192)) + r162_col[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r162_group % $(8)) * $(768)) + (r162_lane[:, None] * $(192)) + r162_col[None, :]) / $(192)) % $(4)) * $(192)) + ((((r162_group % $(8)) * $(768)) + (r162_lane[:, None] * $(192)) + r162_col[None, :]) % $(192))), dtype=tl.float32)
      r162_value = tl.dot(r162_left, r162_right)
      r162_value = r162_value.to(tl.float32)
      tl.store($(candidate_vbmm_38_ptr) + ((r162_group * $(2304)) + (r162_row[:, None] * $(192)) + r162_col[None, :]), r162_value)
    }
    r163_z = tl.arange($(18432))
    r163_value = (tl.load($(candidate_vbmm_38_ptr) + ((((r163_z / $(2304)) % $(8)) * $(2304)) + (((r163_z / $(192)) % $(12)) * $(192)) + (r163_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r163_value = r163_value.to(tl.float32)
    r163_value = tl.broadcast_to(r163_value, [$(18432)])
    tl.store($(candidate_vmul_47_ptr) + r163_z, r163_value)
    r164_z = tl.arange($(18432))
    r164_value = tl.load($(candidate_vmul_47_ptr) + ((((r164_z / $(192)) % $(8)) * $(2304)) + (((r164_z / $(1536)) % $(12)) * $(192)) + (r164_z % $(192))), dtype=tl.float32)
    r164_value = r164_value.to(tl.bfloat16)
    r164_value = tl.broadcast_to(r164_value, [$(18432)])
    tl.store($(candidate_vto_copy_112_ptr) + r164_z, r164_value)
    r165_row = tl.arange($(12))
    r165_col = tl.arange($(128))
    r165_lane = tl.arange($(4))
    for r165_group in range($(8)) {
      r165_left = tl.load($(candidate_vexp_1_ptr) + (((((((r165_group % $(8)) * $(48)) + (r165_row[:, None] * $(4)) + r165_lane[None, :]) / $(48)) % $(8)) * $(48)) + (((((r165_group % $(8)) * $(48)) + (r165_row[:, None] * $(4)) + r165_lane[None, :]) % $(4)) * $(12)) + (((((r165_group % $(8)) * $(48)) + (r165_row[:, None] * $(4)) + r165_lane[None, :]) / $(4)) % $(12))), dtype=tl.float32)
      r165_right = tl.load($(candidate_vto_copy_109_ptr) + (((((((r165_group % $(8)) * $(512)) + (r165_lane[:, None] * $(128)) + r165_col[None, :]) / $(512)) % $(8)) * $(512)) + ((((((r165_group % $(8)) * $(512)) + (r165_lane[:, None] * $(128)) + r165_col[None, :]) / $(128)) % $(4)) * $(128)) + ((((r165_group % $(8)) * $(512)) + (r165_lane[:, None] * $(128)) + r165_col[None, :]) % $(128))), dtype=tl.float32)
      r165_value = tl.dot(r165_left, r165_right)
      r165_value = r165_value.to(tl.float32)
      tl.store($(candidate_vbmm_39_ptr) + ((r165_group * $(1536)) + (r165_row[:, None] * $(128)) + r165_col[None, :]), r165_value)
    }
    r166_z = tl.arange($(12288))
    r166_value = tl.load($(candidate_vbmm_39_ptr) + ((((r166_z / $(128)) % $(8)) * $(1536)) + (((r166_z / $(1024)) % $(12)) * $(128)) + (r166_z % $(128))), dtype=tl.float32)
    r166_value = r166_value.to(tl.bfloat16)
    r166_value = tl.broadcast_to(r166_value, [$(12288)])
    tl.store($(candidate_vto_copy_113_ptr) + r166_z, r166_value)
    r167_z = tl.arange($(24576))
    r167_value = tl.where(((r167_z / $(1536)) % $(16)) < $(12), tl.load($(candidate_vto_copy_112_ptr) + ((((((r167_z / $(1536)) % $(16)) - $(0)) % $(12)) * $(1536)) + (((r167_z / $(192)) % $(8)) * $(192)) + (r167_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r167_value = r167_value.to(tl.bfloat16)
    r167_value = tl.broadcast_to(r167_value, [$(24576)])
    tl.store($(candidate_vcat_17_ptr) + r167_z, r167_value)
    r168_z = tl.arange($(16384))
    r168_value = tl.where(((r168_z / $(1024)) % $(16)) < $(12), tl.load($(candidate_vto_copy_113_ptr) + ((((((r168_z / $(1024)) % $(16)) - $(0)) % $(12)) * $(1024)) + (((r168_z / $(128)) % $(8)) * $(128)) + (r168_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r168_value = r168_value.to(tl.bfloat16)
    r168_value = tl.broadcast_to(r168_value, [$(16384)])
    tl.store($(candidate_vcat_18_ptr) + r168_z, r168_value)
    r169_z = tl.arange($(24576))
    r169_value = tl.load($(candidate_vcat_11_ptr) + ((((r169_z / $(192)) % $(16)) * $(1536)) + (((r169_z / $(3072)) % $(8)) * $(192)) + (r169_z % $(192)) + $(24576)), dtype=tl.bfloat16)
    r169_value = r169_value.to(tl.float32)
    r169_value = tl.broadcast_to(r169_value, [$(24576)])
    tl.store($(candidate_vto_copy_114_ptr) + r169_z, r169_value)
    r170_z = tl.arange($(73728))
    r170_value = tl.load($(candidate_vcat_12_ptr) + ((((r170_z / $(192)) % $(48)) * $(1536)) + (((r170_z / $(9216)) % $(8)) * $(192)) + (r170_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r170_value = r170_value.to(tl.float32)
    r170_value = tl.broadcast_to(r170_value, [$(73728)])
    tl.store($(candidate_vto_copy_115_ptr) + r170_z, r170_value)
    r171_z = tl.arange($(49152))
    r171_value = tl.load($(candidate_vbmm_23_ptr) + ((((r171_z / $(128)) % $(48)) * $(1024)) + (((r171_z / $(6144)) % $(8)) * $(128)) + (r171_z % $(128)) + $(65536)), dtype=tl.bfloat16)
    r171_value = r171_value.to(tl.float32)
    r171_value = tl.broadcast_to(r171_value, [$(49152)])
    tl.store($(candidate_vto_copy_116_ptr) + r171_z, r171_value)
    r172_z = tl.arange($(16384))
    r172_value = tl.load($(candidate_vmm_28_ptr) + ((((r172_z / $(128)) % $(16)) * $(1024)) + (((r172_z / $(2048)) % $(8)) * $(128)) + (r172_z % $(128)) + $(114688)), dtype=tl.bfloat16)
    r172_value = r172_value.to(tl.float32)
    r172_value = tl.broadcast_to(r172_value, [$(16384)])
    tl.store($(candidate_vto_copy_117_ptr) + r172_z, r172_value)
    r173_z = tl.arange($(16384))
    r173_value = tl.load($(candidate_vcat_13_ptr) + ((((r173_z / $(128)) % $(16)) * $(1024)) + (((r173_z / $(2048)) % $(8)) * $(128)) + (r173_z % $(128)) + $(16384)), dtype=tl.bfloat16)
    r173_value = r173_value.to(tl.float32)
    r173_value = tl.broadcast_to(r173_value, [$(16384)])
    tl.store($(candidate_vto_copy_118_ptr) + r173_z, r173_value)
    r174_row = tl.arange($(16))
    r174_col = tl.arange($(48))
    r174_lane = tl.arange($(192))
    for r174_group in range($(8)) {
      r174_left = tl.load($(candidate_vto_copy_114_ptr) + (((((((r174_group % $(8)) * $(3072)) + (r174_row[:, None] * $(192)) + r174_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r174_group % $(8)) * $(3072)) + (r174_row[:, None] * $(192)) + r174_lane[None, :]) / $(192)) % $(16)) * $(192)) + ((((r174_group % $(8)) * $(3072)) + (r174_row[:, None] * $(192)) + r174_lane[None, :]) % $(192))), dtype=tl.float32)
      r174_right = tl.load($(candidate_vto_copy_115_ptr) + (((((((r174_group % $(8)) * $(9216)) + (r174_lane[:, None] * $(48)) + r174_col[None, :]) / $(9216)) % $(8)) * $(9216)) + (((((r174_group % $(8)) * $(9216)) + (r174_lane[:, None] * $(48)) + r174_col[None, :]) % $(48)) * $(192)) + (((((r174_group % $(8)) * $(9216)) + (r174_lane[:, None] * $(48)) + r174_col[None, :]) / $(48)) % $(192))), dtype=tl.float32)
      r174_value = tl.dot(r174_left, r174_right)
      r174_value = r174_value.to(tl.float32)
      tl.store($(candidate_vbmm_40_ptr) + ((r174_group * $(768)) + (r174_row[:, None] * $(48)) + r174_col[None, :]), r174_value)
    }
    r175_z = tl.arange($(6144))
    r175_value = (tl.load($(candidate_vbmm_40_ptr) + ((((r175_z / $(768)) % $(8)) * $(768)) + (((r175_z / $(48)) % $(16)) * $(48)) + (r175_z % $(48))), dtype=tl.float32) * 0.07216878364870322)
    r175_value = r175_value.to(tl.float32)
    r175_value = tl.broadcast_to(r175_value, [$(6144)])
    tl.store($(candidate_vmul_48_ptr) + r175_z, r175_value)
    r176_z = tl.arange($(16))
    r176_value = (tl.cast(r176_z, tl.int64) + tl.full([], $(48), dtype=tl.int64))
    r176_value = tl.broadcast_to(r176_value, [$(16)])
    tl.store($(candidate_vadd_37_ptr) + r176_z, r176_value)
    r177_z = tl.arange($(16))
    r177_value = (tl.load($(candidate_vadd_37_ptr) + r177_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r177_value = tl.broadcast_to(r177_value, [$(16)])
    tl.store($(candidate_vsub_30_ptr) + r177_z, r177_value)
    r178_z = tl.arange($(768))
    r178_value = (tl.cast((r178_z % $(48)), tl.int64) > tl.load($(candidate_vsub_30_ptr) + ((r178_z / $(48)) % $(16)), dtype=tl.int64))
    r178_value = tl.broadcast_to(r178_value, [$(768)])
    tl.store($(candidate_vgt_17_ptr) + r178_z, r178_value)
    r179_z = tl.arange($(6144))
    r179_value = tl.where(tl.load($(candidate_vgt_17_ptr) + ((((r179_z / $(48)) % $(16)) * $(48)) + (r179_z % $(48))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_48_ptr) + ((((r179_z / $(768)) % $(8)) * $(768)) + (((r179_z / $(48)) % $(16)) * $(48)) + (r179_z % $(48))), dtype=tl.float32))
    r179_value = r179_value.to(tl.float32)
    r179_value = tl.broadcast_to(r179_value, [$(6144)])
    tl.store($(candidate_vmasked_fill_17_ptr) + r179_z, r179_value)
    r180_z = tl.arange($(6144))
    r180_value = (tl.load($(candidate_vmasked_fill_17_ptr) + ((((r180_z / $(768)) % $(8)) * $(768)) + (((r180_z / $(48)) % $(16)) * $(48)) + (r180_z % $(48))), dtype=tl.float32) - tl.load($(candidate_vcat_14_ptr) + ((((r180_z / $(48)) % $(16)) * $(8)) + ((r180_z / $(768)) % $(8)) + $(128)), dtype=tl.float32))
    r180_value = r180_value.to(tl.float32)
    r180_value = tl.broadcast_to(r180_value, [$(6144)])
    tl.store($(candidate_vsub_31_ptr) + r180_z, r180_value)
    r181_z = tl.arange($(6144))
    r181_value = tl.exp(tl.load($(candidate_vsub_31_ptr) + ((((r181_z / $(768)) % $(8)) * $(768)) + (((r181_z / $(48)) % $(16)) * $(48)) + (r181_z % $(48))), dtype=tl.float32))
    r181_value = r181_value.to(tl.float32)
    r181_value = tl.broadcast_to(r181_value, [$(6144)])
    tl.store($(candidate_vexp_2_ptr) + r181_z, r181_value)
    r182_z = tl.arange($(16384))
    r182_value = (tl.load($(candidate_vto_copy_117_ptr) + ((((r182_z / $(2048)) % $(8)) * $(2048)) + (((r182_z / $(128)) % $(16)) * $(128)) + (r182_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_118_ptr) + ((((r182_z / $(2048)) % $(8)) * $(2048)) + (((r182_z / $(128)) % $(16)) * $(128)) + (r182_z % $(128))), dtype=tl.float32))
    r182_value = r182_value.to(tl.float32)
    r182_value = tl.broadcast_to(r182_value, [$(16384)])
    tl.store($(candidate_vmul_49_ptr) + r182_z, r182_value)
    r183_z = tl.arange($(128))
    r183_lane = tl.arange($(128))
    r183_value = tl.sum(tl.load($(candidate_vmul_49_ptr) + ((((r183_z[:, None] / $(16)) % $(8)) * $(2048)) + ((r183_z[:, None] % $(16)) * $(128)) + r183_lane[None, :]), dtype=tl.float32), axis=1)
    r183_value = r183_value.to(tl.float32)
    r183_value = tl.broadcast_to(r183_value, [$(128)])
    tl.store($(candidate_vsum_3_ptr) + r183_z, r183_value)
    r184_row = tl.arange($(16))
    r184_col = tl.arange($(48))
    r184_lane = tl.arange($(128))
    for r184_group in range($(8)) {
      r184_left = tl.load($(candidate_vto_copy_117_ptr) + (((((((r184_group % $(8)) * $(2048)) + (r184_row[:, None] * $(128)) + r184_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r184_group % $(8)) * $(2048)) + (r184_row[:, None] * $(128)) + r184_lane[None, :]) / $(128)) % $(16)) * $(128)) + ((((r184_group % $(8)) * $(2048)) + (r184_row[:, None] * $(128)) + r184_lane[None, :]) % $(128))), dtype=tl.float32)
      r184_right = tl.load($(candidate_vto_copy_116_ptr) + (((((((r184_group % $(8)) * $(6144)) + (r184_lane[:, None] * $(48)) + r184_col[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r184_group % $(8)) * $(6144)) + (r184_lane[:, None] * $(48)) + r184_col[None, :]) % $(48)) * $(128)) + (((((r184_group % $(8)) * $(6144)) + (r184_lane[:, None] * $(48)) + r184_col[None, :]) / $(48)) % $(128))), dtype=tl.float32)
      r184_value = tl.dot(r184_left, r184_right)
      r184_value = r184_value.to(tl.float32)
      tl.store($(candidate_vbmm_41_ptr) + ((r184_group * $(768)) + (r184_row[:, None] * $(48)) + r184_col[None, :]), r184_value)
    }
    r185_z = tl.arange($(6144))
    r185_value = (tl.load($(candidate_vbmm_41_ptr) + ((((r185_z / $(768)) % $(8)) * $(768)) + (((r185_z / $(48)) % $(16)) * $(48)) + (r185_z % $(48))), dtype=tl.float32) - tl.load($(candidate_vsum_3_ptr) + ((((r185_z / $(768)) % $(8)) * $(16)) + ((r185_z / $(48)) % $(16))), dtype=tl.float32))
    r185_value = r185_value.to(tl.float32)
    r185_value = tl.broadcast_to(r185_value, [$(6144)])
    tl.store($(candidate_vsub_32_ptr) + r185_z, r185_value)
    r186_z = tl.arange($(6144))
    r186_value = (tl.load($(candidate_vexp_2_ptr) + ((((r186_z / $(768)) % $(8)) * $(768)) + (((r186_z / $(48)) % $(16)) * $(48)) + (r186_z % $(48))), dtype=tl.float32) * tl.load($(candidate_vsub_32_ptr) + ((((r186_z / $(768)) % $(8)) * $(768)) + (((r186_z / $(48)) % $(16)) * $(48)) + (r186_z % $(48))), dtype=tl.float32))
    r186_value = r186_value.to(tl.float32)
    r186_value = tl.broadcast_to(r186_value, [$(6144)])
    tl.store($(candidate_vmul_50_ptr) + r186_z, r186_value)
    r187_row = tl.arange($(16))
    r187_col = tl.arange($(192))
    r187_lane = tl.arange($(48))
    for r187_group in range($(8)) {
      r187_left = tl.load($(candidate_vmul_50_ptr) + (((((((r187_group % $(8)) * $(768)) + (r187_row[:, None] * $(48)) + r187_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r187_group % $(8)) * $(768)) + (r187_row[:, None] * $(48)) + r187_lane[None, :]) / $(48)) % $(16)) * $(48)) + ((((r187_group % $(8)) * $(768)) + (r187_row[:, None] * $(48)) + r187_lane[None, :]) % $(48))), dtype=tl.float32)
      r187_right = tl.load($(candidate_vto_copy_115_ptr) + (((((((r187_group % $(8)) * $(9216)) + (r187_lane[:, None] * $(192)) + r187_col[None, :]) / $(9216)) % $(8)) * $(9216)) + ((((((r187_group % $(8)) * $(9216)) + (r187_lane[:, None] * $(192)) + r187_col[None, :]) / $(192)) % $(48)) * $(192)) + ((((r187_group % $(8)) * $(9216)) + (r187_lane[:, None] * $(192)) + r187_col[None, :]) % $(192))), dtype=tl.float32)
      r187_value = tl.dot(r187_left, r187_right)
      r187_value = r187_value.to(tl.float32)
      tl.store($(candidate_vbmm_42_ptr) + ((r187_group * $(3072)) + (r187_row[:, None] * $(192)) + r187_col[None, :]), r187_value)
    }
    r188_z = tl.arange($(24576))
    r188_value = (tl.load($(candidate_vbmm_42_ptr) + ((((r188_z / $(3072)) % $(8)) * $(3072)) + (((r188_z / $(192)) % $(16)) * $(192)) + (r188_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r188_value = r188_value.to(tl.float32)
    r188_value = tl.broadcast_to(r188_value, [$(24576)])
    tl.store($(candidate_vmul_51_ptr) + r188_z, r188_value)
    r189_z = tl.arange($(24576))
    r189_value = tl.load($(candidate_vmul_51_ptr) + ((((r189_z / $(192)) % $(8)) * $(3072)) + (((r189_z / $(1536)) % $(16)) * $(192)) + (r189_z % $(192))), dtype=tl.float32)
    r189_value = r189_value.to(tl.bfloat16)
    r189_value = tl.broadcast_to(r189_value, [$(24576)])
    tl.store($(candidate_vto_copy_119_ptr) + r189_z, r189_value)
    r190_row = tl.arange($(48))
    r190_col = tl.arange($(192))
    r190_lane = tl.arange($(16))
    for r190_group in range($(8)) {
      r190_left = tl.load($(candidate_vmul_50_ptr) + (((((((r190_group % $(8)) * $(768)) + (r190_row[:, None] * $(16)) + r190_lane[None, :]) / $(768)) % $(8)) * $(768)) + (((((r190_group % $(8)) * $(768)) + (r190_row[:, None] * $(16)) + r190_lane[None, :]) % $(16)) * $(48)) + (((((r190_group % $(8)) * $(768)) + (r190_row[:, None] * $(16)) + r190_lane[None, :]) / $(16)) % $(48))), dtype=tl.float32)
      r190_right = tl.load($(candidate_vto_copy_114_ptr) + (((((((r190_group % $(8)) * $(3072)) + (r190_lane[:, None] * $(192)) + r190_col[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r190_group % $(8)) * $(3072)) + (r190_lane[:, None] * $(192)) + r190_col[None, :]) / $(192)) % $(16)) * $(192)) + ((((r190_group % $(8)) * $(3072)) + (r190_lane[:, None] * $(192)) + r190_col[None, :]) % $(192))), dtype=tl.float32)
      r190_value = tl.dot(r190_left, r190_right)
      r190_value = r190_value.to(tl.float32)
      tl.store($(candidate_vbmm_43_ptr) + ((r190_group * $(9216)) + (r190_row[:, None] * $(192)) + r190_col[None, :]), r190_value)
    }
    r191_z = tl.arange($(73728))
    r191_value = (tl.load($(candidate_vbmm_43_ptr) + ((((r191_z / $(9216)) % $(8)) * $(9216)) + (((r191_z / $(192)) % $(48)) * $(192)) + (r191_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r191_value = r191_value.to(tl.float32)
    r191_value = tl.broadcast_to(r191_value, [$(73728)])
    tl.store($(candidate_vmul_52_ptr) + r191_z, r191_value)
    r192_z = tl.arange($(73728))
    r192_value = tl.load($(candidate_vmul_52_ptr) + ((((r192_z / $(192)) % $(8)) * $(9216)) + (((r192_z / $(1536)) % $(48)) * $(192)) + (r192_z % $(192))), dtype=tl.float32)
    r192_value = r192_value.to(tl.bfloat16)
    r192_value = tl.broadcast_to(r192_value, [$(73728)])
    tl.store($(candidate_vto_copy_120_ptr) + r192_z, r192_value)
    r193_row = tl.arange($(48))
    r193_col = tl.arange($(128))
    r193_lane = tl.arange($(16))
    for r193_group in range($(8)) {
      r193_left = tl.load($(candidate_vexp_2_ptr) + (((((((r193_group % $(8)) * $(768)) + (r193_row[:, None] * $(16)) + r193_lane[None, :]) / $(768)) % $(8)) * $(768)) + (((((r193_group % $(8)) * $(768)) + (r193_row[:, None] * $(16)) + r193_lane[None, :]) % $(16)) * $(48)) + (((((r193_group % $(8)) * $(768)) + (r193_row[:, None] * $(16)) + r193_lane[None, :]) / $(16)) % $(48))), dtype=tl.float32)
      r193_right = tl.load($(candidate_vto_copy_117_ptr) + (((((((r193_group % $(8)) * $(2048)) + (r193_lane[:, None] * $(128)) + r193_col[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r193_group % $(8)) * $(2048)) + (r193_lane[:, None] * $(128)) + r193_col[None, :]) / $(128)) % $(16)) * $(128)) + ((((r193_group % $(8)) * $(2048)) + (r193_lane[:, None] * $(128)) + r193_col[None, :]) % $(128))), dtype=tl.float32)
      r193_value = tl.dot(r193_left, r193_right)
      r193_value = r193_value.to(tl.float32)
      tl.store($(candidate_vbmm_44_ptr) + ((r193_group * $(6144)) + (r193_row[:, None] * $(128)) + r193_col[None, :]), r193_value)
    }
    r194_z = tl.arange($(49152))
    r194_value = tl.load($(candidate_vbmm_44_ptr) + ((((r194_z / $(128)) % $(8)) * $(6144)) + (((r194_z / $(1024)) % $(48)) * $(128)) + (r194_z % $(128))), dtype=tl.float32)
    r194_value = r194_value.to(tl.bfloat16)
    r194_value = tl.broadcast_to(r194_value, [$(49152)])
    tl.store($(candidate_vto_copy_121_ptr) + r194_z, r194_value)
    r195_z = tl.arange($(98304))
    r195_value = tl.where(((r195_z / $(1536)) % $(64)) < $(48), tl.load($(candidate_vto_copy_120_ptr) + ((((((r195_z / $(1536)) % $(64)) - $(0)) % $(48)) * $(1536)) + (((r195_z / $(192)) % $(8)) * $(192)) + (r195_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r195_value = r195_value.to(tl.bfloat16)
    r195_value = tl.broadcast_to(r195_value, [$(98304)])
    tl.store($(candidate_vcat_19_ptr) + r195_z, r195_value)
    r196_z = tl.arange($(65536))
    r196_value = tl.where(((r196_z / $(1024)) % $(64)) < $(48), tl.load($(candidate_vto_copy_121_ptr) + ((((((r196_z / $(1024)) % $(64)) - $(0)) % $(48)) * $(1024)) + (((r196_z / $(128)) % $(8)) * $(128)) + (r196_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r196_value = r196_value.to(tl.bfloat16)
    r196_value = tl.broadcast_to(r196_value, [$(65536)])
    tl.store($(candidate_vcat_20_ptr) + r196_z, r196_value)
    r197_z = tl.arange($(49152))
    r197_value = tl.where(((r197_z / $(1536)) % $(32)) < $(12), tl.load($(candidate_vto_copy_103_ptr) + ((((((r197_z / $(1536)) % $(32)) - $(0)) % $(12)) * $(1536)) + (((r197_z / $(192)) % $(8)) * $(192)) + (r197_z % $(192))), dtype=tl.bfloat16), tl.where(((r197_z / $(1536)) % $(32)) < $(16), tl.load($(candidate_vto_copy_111_ptr) + ((((((r197_z / $(1536)) % $(32)) - $(12)) % $(4)) * $(1536)) + (((r197_z / $(192)) % $(8)) * $(192)) + (r197_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vto_copy_119_ptr) + ((((((r197_z / $(1536)) % $(32)) - $(16)) % $(16)) * $(1536)) + (((r197_z / $(192)) % $(8)) * $(192)) + (r197_z % $(192))), dtype=tl.bfloat16)))
    r197_value = r197_value.to(tl.bfloat16)
    r197_value = tl.broadcast_to(r197_value, [$(49152)])
    tl.store($(candidate_vcat_21_ptr) + r197_z, r197_value)
    r198_z = tl.arange($(196608))
    r198_value = tl.where(((r198_z / $(1536)) % $(128)) < $(48), tl.load($(candidate_vcat_15_ptr) + ((((((r198_z / $(1536)) % $(128)) - $(0)) % $(48)) * $(1536)) + (((r198_z / $(192)) % $(8)) * $(192)) + (r198_z % $(192))), dtype=tl.bfloat16), tl.where(((r198_z / $(1536)) % $(128)) < $(64), tl.load($(candidate_vcat_17_ptr) + ((((((r198_z / $(1536)) % $(128)) - $(48)) % $(16)) * $(1536)) + (((r198_z / $(192)) % $(8)) * $(192)) + (r198_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vcat_19_ptr) + ((((((r198_z / $(1536)) % $(128)) - $(64)) % $(64)) * $(1536)) + (((r198_z / $(192)) % $(8)) * $(192)) + (r198_z % $(192))), dtype=tl.bfloat16)))
    r198_value = r198_value.to(tl.bfloat16)
    r198_value = tl.broadcast_to(r198_value, [$(196608)])
    tl.store($(candidate_vcat_22_ptr) + r198_z, r198_value)
    r199_z = tl.arange($(131072))
    r199_value = tl.where(((r199_z / $(1024)) % $(128)) < $(48), tl.load($(candidate_vcat_16_ptr) + ((((((r199_z / $(1024)) % $(128)) - $(0)) % $(48)) * $(1024)) + (((r199_z / $(128)) % $(8)) * $(128)) + (r199_z % $(128))), dtype=tl.bfloat16), tl.where(((r199_z / $(1024)) % $(128)) < $(64), tl.load($(candidate_vcat_18_ptr) + ((((((r199_z / $(1024)) % $(128)) - $(48)) % $(16)) * $(1024)) + (((r199_z / $(128)) % $(8)) * $(128)) + (r199_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_20_ptr) + ((((((r199_z / $(1024)) % $(128)) - $(64)) % $(64)) * $(1024)) + (((r199_z / $(128)) % $(8)) * $(128)) + (r199_z % $(128))), dtype=tl.bfloat16)))
    r199_value = r199_value.to(tl.bfloat16)
    r199_value = tl.broadcast_to(r199_value, [$(131072)])
    tl.store($(candidate_vcat_23_ptr) + r199_z, r199_value)
    r200_z = tl.arange($(8192))
    r200_lane = tl.arange($(8))
    r200_value = tl.sum(tl.load($(candidate_vcat_22_ptr) + ((((r200_z[:, None] / $(64)) % $(128)) * $(1536)) + (r200_lane[None, :] * $(192)) + (r200_z[:, None] % $(64)) + $(128)), dtype=tl.bfloat16), axis=1)
    r200_value = r200_value.to(tl.bfloat16)
    r200_value = tl.broadcast_to(r200_value, [$(8192)])
    tl.store($(candidate_vsum_4_ptr) + r200_z, r200_value)
    r201_row = tl.arange($(128))
    r201_col = tl.arange($(512))
    r201_lane = tl.arange($(1024))
    for r201_group in range($(1)) {
      r201_left = tl.load($(candidate_vcat_22_ptr) + ((((((r201_row[:, None] * $(1024)) + r201_lane[None, :]) / $(1024)) % $(128)) * $(1536)) + ((((r201_row[:, None] * $(1024)) + r201_lane[None, :]) % $(8)) * $(192)) + ((((r201_row[:, None] * $(1024)) + r201_lane[None, :]) / $(8)) % $(128))), dtype=tl.bfloat16)
      r201_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r201_lane[:, None] * $(512)) + r201_col[None, :]) / $(512)) % $(8)) * $(131072)) + (((((r201_lane[:, None] * $(512)) + r201_col[None, :]) / $(4096)) % $(128)) * $(512)) + (((r201_lane[:, None] * $(512)) + r201_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r201_value = tl.dot(r201_left, r201_right)
      r201_value = r201_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_45_ptr) + ((r201_group * $(65536)) + (r201_row[:, None] * $(512)) + r201_col[None, :]), r201_value)
    }
    r202_row = tl.arange($(128))
    r202_col = tl.arange($(512))
    r202_lane = tl.arange($(1024))
    for r202_group in range($(1)) {
      r202_left = tl.load($(candidate_vcat_23_ptr) + ((((((r202_row[:, None] * $(1024)) + r202_lane[None, :]) / $(1024)) % $(128)) * $(1024)) + ((((r202_row[:, None] * $(1024)) + r202_lane[None, :]) % $(8)) * $(128)) + ((((r202_row[:, None] * $(1024)) + r202_lane[None, :]) / $(8)) % $(128))), dtype=tl.bfloat16)
      r202_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r202_lane[:, None] * $(512)) + r202_col[None, :]) / $(512)) % $(8)) * $(131072)) + (((((r202_lane[:, None] * $(512)) + r202_col[None, :]) / $(4096)) % $(128)) * $(512)) + (((r202_lane[:, None] * $(512)) + r202_col[None, :]) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r202_value = tl.dot(r202_left, r202_right)
      r202_value = r202_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_46_ptr) + ((r202_group * $(65536)) + (r202_row[:, None] * $(512)) + r202_col[None, :]), r202_value)
    }
    r203_z = tl.arange($(65536))
    r203_value = (tl.load($(candidate_vbmm_45_ptr) + ((((r203_z / $(512)) % $(128)) * $(512)) + (r203_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vbmm_46_ptr) + ((((r203_z / $(512)) % $(128)) * $(512)) + (r203_z % $(512))), dtype=tl.bfloat16))
    r203_value = r203_value.to(tl.bfloat16)
    r203_value = tl.broadcast_to(r203_value, [$(65536)])
    tl.store($(candidate_vadd_38_ptr) + r203_z, r203_value)
    r204_z = tl.arange($(65536))
    r204_value = tl.where(((((r204_z / $(512)) % $(128)) >= $(0)) & (((r204_z / $(512)) % $(128)) < $(128)) & ((((r204_z / $(512)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vadd_38_ptr) + ((((((r204_z / $(512)) % $(128)) - $(0)) % $(128)) * $(512)) + (r204_z % $(512))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r204_value = r204_value.to(tl.bfloat16)
    r204_value = tl.broadcast_to(r204_value, [$(65536)])
    tl.store($(candidate_vslice_backward_ptr) + r204_z, r204_value)
    r205_z = tl.arange($(8192))
    r205_value = tl.where(((((r205_z / $(64)) % $(128)) >= $(0)) & (((r205_z / $(64)) % $(128)) < $(128)) & ((((r205_z / $(64)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vsum_4_ptr) + ((((((r205_z / $(64)) % $(128)) - $(0)) % $(128)) * $(64)) + (r205_z % $(64))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r205_value = r205_value.to(tl.bfloat16)
    r205_value = tl.broadcast_to(r205_value, [$(8192)])
    tl.store($(candidate_vslice_backward_1_ptr) + r205_z, r205_value)
    r206_z = tl.arange($(131072))
    r206_value = tl.where(((((r206_z / $(1024)) % $(128)) >= $(96)) & (((r206_z / $(1024)) % $(128)) < $(128)) & ((((r206_z / $(1024)) % $(128)) - $(96)) % $(1) == $(0))), tl.load($(candidate_vcat_21_ptr) + ((((((r206_z / $(1024)) % $(128)) - $(96)) % $(32)) * $(1536)) + (((r206_z / $(128)) % $(8)) * $(192)) + (r206_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r206_value = r206_value.to(tl.bfloat16)
    r206_value = tl.broadcast_to(r206_value, [$(131072)])
    tl.store($(candidate_vslice_backward_2_ptr) + r206_z, r206_value)
    r207_z = tl.arange($(65536))
    r207_value = tl.where(((((r207_z / $(512)) % $(128)) >= $(96)) & (((r207_z / $(512)) % $(128)) < $(128)) & ((((r207_z / $(512)) % $(128)) - $(96)) % $(1) == $(0))), tl.load($(candidate_vcat_21_ptr) + ((((((r207_z / $(512)) % $(128)) - $(96)) % $(32)) * $(1536)) + (((r207_z / $(64)) % $(8)) * $(192)) + (r207_z % $(64)) + $(128)), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r207_value = r207_value.to(tl.bfloat16)
    r207_value = tl.broadcast_to(r207_value, [$(65536)])
    tl.store($(candidate_vslice_backward_3_ptr) + r207_z, r207_value)
    r208_row = tl.arange($(128))
    r208_col = tl.arange($(1024))
    r208_lane = tl.arange($(512))
    for r208_group in range($(1)) {
      r208_left = tl.load($(candidate_vindex_2_ptr) + ((((((r208_row[:, None] * $(512)) + r208_lane[None, :]) / $(512)) % $(128)) * $(576)) + (((r208_row[:, None] * $(512)) + r208_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r208_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r208_lane[:, None] * $(1024)) + r208_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r208_lane[:, None] * $(1024)) + r208_col[None, :]) % $(128)) * $(512)) + ((((r208_lane[:, None] * $(1024)) + r208_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r208_value = tl.dot(r208_left, r208_right)
      r208_value = r208_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_49_ptr) + ((r208_group * $(131072)) + (r208_row[:, None] * $(1024)) + r208_col[None, :]), r208_value)
    }
    r209_row = tl.arange($(128))
    r209_col = tl.arange($(1024))
    r209_lane = tl.arange($(512))
    for r209_group in range($(1)) {
      r209_left = tl.load($(candidate_vindex_2_ptr) + ((((((r209_row[:, None] * $(512)) + r209_lane[None, :]) / $(512)) % $(128)) * $(576)) + (((r209_row[:, None] * $(512)) + r209_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r209_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r209_lane[:, None] * $(1024)) + r209_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r209_lane[:, None] * $(1024)) + r209_col[None, :]) % $(128)) * $(512)) + ((((r209_lane[:, None] * $(1024)) + r209_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r209_value = tl.dot(r209_left, r209_right)
      r209_value = r209_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_50_ptr) + ((r209_group * $(131072)) + (r209_row[:, None] * $(1024)) + r209_col[None, :]), r209_value)
    }
    r210_z = tl.arange($(49152))
    r210_value = tl.where((r210_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r210_z / $(1536)) % $(32)) * $(1024)) + (((r210_z / $(192)) % $(8)) * $(128)) + ((r210_z % $(192)) - $(0)) + $(65536)) / $(1024)) % $(128)) * $(1536)) + (((((((r210_z / $(1536)) % $(32)) * $(1024)) + (((r210_z / $(192)) % $(8)) * $(128)) + ((r210_z % $(192)) - $(0)) + $(65536)) / $(128)) % $(8)) * $(192)) + (((((r210_z / $(1536)) % $(32)) * $(1024)) + (((r210_z / $(192)) % $(8)) * $(128)) + ((r210_z % $(192)) - $(0)) + $(65536)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vadd_6_ptr) + ((((((((r210_z / $(1536)) % $(32)) * $(512)) + (((r210_z / $(192)) % $(8)) * $(64)) + ((r210_z % $(192)) - $(128)) + $(32768)) / $(512)) % $(128)) * $(512)) + (((((((r210_z / $(1536)) % $(32)) * $(512)) + (((r210_z / $(192)) % $(8)) * $(64)) + ((r210_z % $(192)) - $(128)) + $(32768)) / $(64)) % $(8)) * $(64)) + (((((r210_z / $(1536)) % $(32)) * $(512)) + (((r210_z / $(192)) % $(8)) * $(64)) + ((r210_z % $(192)) - $(128)) + $(32768)) % $(64))), dtype=tl.bfloat16))
    r210_value = r210_value.to(tl.bfloat16)
    r210_value = tl.broadcast_to(r210_value, [$(49152)])
    tl.store($(candidate_vcat_25_ptr) + r210_z, r210_value)
    r211_z = tl.arange($(196608))
    r211_value = tl.where((r211_z % $(192)) < $(128), tl.load($(candidate_vbmm_49_ptr) + ((((((((r211_z / $(1536)) % $(128)) * $(1024)) + (((r211_z / $(192)) % $(8)) * $(128)) + ((r211_z % $(192)) - $(0))) / $(1024)) % $(128)) * $(1024)) + (((((((r211_z / $(1536)) % $(128)) * $(1024)) + (((r211_z / $(192)) % $(8)) * $(128)) + ((r211_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r211_z / $(1536)) % $(128)) * $(1024)) + (((r211_z / $(192)) % $(8)) * $(128)) + ((r211_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vindex_2_ptr) + ((((r211_z / $(1536)) % $(128)) * $(576)) + ((((r211_z % $(192)) - $(128)) + $(512)) % $(576))), dtype=tl.bfloat16))
    r211_value = r211_value.to(tl.bfloat16)
    r211_value = tl.broadcast_to(r211_value, [$(196608)])
    tl.store($(candidate_vcat_26_ptr) + r211_z, r211_value)
    r212_z = tl.arange($(18432))
    r212_value = tl.load($(candidate_vcat_25_ptr) + ((((r212_z / $(192)) % $(12)) * $(1536)) + (((r212_z / $(2304)) % $(8)) * $(192)) + (r212_z % $(192))), dtype=tl.bfloat16)
    r212_value = r212_value.to(tl.float32)
    r212_value = tl.broadcast_to(r212_value, [$(18432)])
    tl.store($(candidate_vto_copy_122_ptr) + r212_z, r212_value)
    r213_z = tl.arange($(36864))
    r213_value = tl.load($(candidate_vcat_26_ptr) + ((((r213_z / $(192)) % $(24)) * $(1536)) + (((r213_z / $(4608)) % $(8)) * $(192)) + (r213_z % $(192))), dtype=tl.bfloat16)
    r213_value = r213_value.to(tl.float32)
    r213_value = tl.broadcast_to(r213_value, [$(36864)])
    tl.store($(candidate_vto_copy_123_ptr) + r213_z, r213_value)
    r214_row = tl.arange($(12))
    r214_col = tl.arange($(24))
    r214_lane = tl.arange($(192))
    for r214_group in range($(8)) {
      r214_left = tl.load($(candidate_vto_copy_122_ptr) + (((((((r214_group % $(8)) * $(2304)) + (r214_row[:, None] * $(192)) + r214_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r214_group % $(8)) * $(2304)) + (r214_row[:, None] * $(192)) + r214_lane[None, :]) / $(192)) % $(12)) * $(192)) + ((((r214_group % $(8)) * $(2304)) + (r214_row[:, None] * $(192)) + r214_lane[None, :]) % $(192))), dtype=tl.float32)
      r214_right = tl.load($(candidate_vto_copy_123_ptr) + (((((((r214_group % $(8)) * $(4608)) + (r214_lane[:, None] * $(24)) + r214_col[None, :]) / $(4608)) % $(8)) * $(4608)) + (((((r214_group % $(8)) * $(4608)) + (r214_lane[:, None] * $(24)) + r214_col[None, :]) % $(24)) * $(192)) + (((((r214_group % $(8)) * $(4608)) + (r214_lane[:, None] * $(24)) + r214_col[None, :]) / $(24)) % $(192))), dtype=tl.float32)
      r214_value = tl.dot(r214_left, r214_right)
      r214_value = r214_value.to(tl.float32)
      tl.store($(candidate_vbmm_51_ptr) + ((r214_group * $(288)) + (r214_row[:, None] * $(24)) + r214_col[None, :]), r214_value)
    }
    r215_z = tl.arange($(2304))
    r215_value = (tl.load($(candidate_vbmm_51_ptr) + ((((r215_z / $(288)) % $(8)) * $(288)) + (((r215_z / $(24)) % $(12)) * $(24)) + (r215_z % $(24))), dtype=tl.float32) * 0.07216878364870322)
    r215_value = r215_value.to(tl.float32)
    r215_value = tl.broadcast_to(r215_value, [$(2304)])
    tl.store($(candidate_vmul_53_ptr) + r215_z, r215_value)
    r216_z = tl.arange($(12))
    r216_value = (tl.cast(r216_z, tl.int64) + tl.full([], $(24), dtype=tl.int64))
    r216_value = tl.broadcast_to(r216_value, [$(12)])
    tl.store($(candidate_vadd_39_ptr) + r216_z, r216_value)
    r217_z = tl.arange($(12))
    r217_value = (tl.load($(candidate_vadd_39_ptr) + r217_z, dtype=tl.int64) - tl.full([], $(12), dtype=tl.int64))
    r217_value = tl.broadcast_to(r217_value, [$(12)])
    tl.store($(candidate_vsub_33_ptr) + r217_z, r217_value)
    r218_z = tl.arange($(288))
    r218_value = (tl.cast((r218_z % $(24)), tl.int64) > tl.load($(candidate_vsub_33_ptr) + ((r218_z / $(24)) % $(12)), dtype=tl.int64))
    r218_value = tl.broadcast_to(r218_value, [$(288)])
    tl.store($(candidate_vgt_18_ptr) + r218_z, r218_value)
    r219_z = tl.arange($(2304))
    r219_value = tl.where(tl.load($(candidate_vgt_18_ptr) + ((((r219_z / $(24)) % $(12)) * $(24)) + (r219_z % $(24))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_53_ptr) + ((((r219_z / $(288)) % $(8)) * $(288)) + (((r219_z / $(24)) % $(12)) * $(24)) + (r219_z % $(24))), dtype=tl.float32))
    r219_value = r219_value.to(tl.float32)
    r219_value = tl.broadcast_to(r219_value, [$(2304)])
    tl.store($(candidate_vmasked_fill_18_ptr) + r219_z, r219_value)
    r220_z = tl.arange($(2304))
    r220_lane = tl.arange($(24))
    r220_row_values = tl.load($(candidate_vmasked_fill_18_ptr) + ((((r220_z[:, None] / $(288)) % $(8)) * $(288)) + (((r220_z[:, None] / $(24)) % $(12)) * $(24)) + (r220_lane[None, :] % $(24))), dtype=tl.float32)
    r220_maximum = tl.max(r220_row_values, axis=1, keep_dims=true)
    r220_weights = tl.exp(r220_row_values - r220_maximum)
    r220_total = tl.sum(r220_weights, axis=1)
    r220_maximum = tl.max(r220_row_values, axis=1)
    r220_value = tl.exp(tl.load($(candidate_vmasked_fill_18_ptr) + ((((r220_z / $(288)) % $(8)) * $(288)) + (((r220_z / $(24)) % $(12)) * $(24)) + (r220_z % $(24))), dtype=tl.float32) - r220_maximum) / r220_total
    r220_value = r220_value.to(tl.float32)
    r220_value = tl.broadcast_to(r220_value, [$(2304)])
    tl.store($(candidate_vsoftmax_15_ptr) + r220_z, r220_value)
    r221_z = tl.arange($(2304))
    r221_value = tl.load($(candidate_vsoftmax_15_ptr) + ((((r221_z / $(288)) % $(8)) * $(288)) + (((r221_z / $(24)) % $(12)) * $(24)) + (r221_z % $(24))), dtype=tl.float32)
    r221_value = r221_value.to(tl.bfloat16)
    r221_value = tl.broadcast_to(r221_value, [$(2304)])
    tl.store($(candidate_vto_copy_124_ptr) + r221_z, r221_value)
    r222_row = tl.arange($(12))
    r222_col = tl.arange($(128))
    r222_lane = tl.arange($(24))
    for r222_group in range($(8)) {
      r222_left = tl.load($(candidate_vto_copy_124_ptr) + (((((((r222_group % $(8)) * $(288)) + (r222_row[:, None] * $(24)) + r222_lane[None, :]) / $(288)) % $(8)) * $(288)) + ((((((r222_group % $(8)) * $(288)) + (r222_row[:, None] * $(24)) + r222_lane[None, :]) / $(24)) % $(12)) * $(24)) + ((((r222_group % $(8)) * $(288)) + (r222_row[:, None] * $(24)) + r222_lane[None, :]) % $(24))), dtype=tl.bfloat16)
      r222_right = tl.load($(candidate_vbmm_50_ptr) + (((((((r222_group % $(8)) * $(3072)) + (r222_lane[:, None] * $(128)) + r222_col[None, :]) / $(128)) % $(24)) * $(1024)) + ((((((r222_group % $(8)) * $(3072)) + (r222_lane[:, None] * $(128)) + r222_col[None, :]) / $(3072)) % $(8)) * $(128)) + ((((r222_group % $(8)) * $(3072)) + (r222_lane[:, None] * $(128)) + r222_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r222_value = tl.dot(r222_left, r222_right)
      r222_value = r222_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_52_ptr) + ((r222_group * $(1536)) + (r222_row[:, None] * $(128)) + r222_col[None, :]), r222_value)
    }
    r223_z = tl.arange($(96))
    r223_lane = tl.arange($(24))
    r223_row_values = tl.load($(candidate_vmasked_fill_18_ptr) + ((((r223_z[:, None] / $(12)) % $(8)) * $(288)) + ((r223_z[:, None] % $(12)) * $(24)) + (r223_lane[None, :] % $(24))), dtype=tl.float32)
    r223_maximum = tl.max(r223_row_values, axis=1, keep_dims=true)
    r223_weights = tl.exp(r223_row_values - r223_maximum)
    r223_total = tl.sum(r223_weights, axis=1)
    r223_maximum = tl.max(r223_row_values, axis=1)
    r223_value = r223_maximum + tl.log(r223_total)
    r223_value = r223_value.to(tl.float32)
    r223_value = tl.broadcast_to(r223_value, [$(96)])
    tl.store($(candidate_vlogsumexp_15_ptr) + r223_z, r223_value)
    r224_z = tl.arange($(6144))
    r224_value = tl.load($(candidate_vcat_25_ptr) + ((((r224_z / $(192)) % $(4)) * $(1536)) + (((r224_z / $(768)) % $(8)) * $(192)) + (r224_z % $(192)) + $(18432)), dtype=tl.bfloat16)
    r224_value = r224_value.to(tl.float32)
    r224_value = tl.broadcast_to(r224_value, [$(6144)])
    tl.store($(candidate_vto_copy_125_ptr) + r224_z, r224_value)
    r225_z = tl.arange($(12288))
    r225_value = tl.load($(candidate_vcat_26_ptr) + ((((r225_z / $(192)) % $(8)) * $(1536)) + (((r225_z / $(1536)) % $(8)) * $(192)) + (r225_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r225_value = r225_value.to(tl.float32)
    r225_value = tl.broadcast_to(r225_value, [$(12288)])
    tl.store($(candidate_vto_copy_126_ptr) + r225_z, r225_value)
    r226_row = tl.arange($(4))
    r226_col = tl.arange($(8))
    r226_lane = tl.arange($(192))
    for r226_group in range($(8)) {
      r226_left = tl.load($(candidate_vto_copy_125_ptr) + (((((((r226_group % $(8)) * $(768)) + (r226_row[:, None] * $(192)) + r226_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r226_group % $(8)) * $(768)) + (r226_row[:, None] * $(192)) + r226_lane[None, :]) / $(192)) % $(4)) * $(192)) + ((((r226_group % $(8)) * $(768)) + (r226_row[:, None] * $(192)) + r226_lane[None, :]) % $(192))), dtype=tl.float32)
      r226_right = tl.load($(candidate_vto_copy_126_ptr) + (((((((r226_group % $(8)) * $(1536)) + (r226_lane[:, None] * $(8)) + r226_col[None, :]) / $(1536)) % $(8)) * $(1536)) + (((((r226_group % $(8)) * $(1536)) + (r226_lane[:, None] * $(8)) + r226_col[None, :]) % $(8)) * $(192)) + (((((r226_group % $(8)) * $(1536)) + (r226_lane[:, None] * $(8)) + r226_col[None, :]) / $(8)) % $(192))), dtype=tl.float32)
      r226_value = tl.dot(r226_left, r226_right)
      r226_value = r226_value.to(tl.float32)
      tl.store($(candidate_vbmm_53_ptr) + ((r226_group * $(32)) + (r226_row[:, None] * $(8)) + r226_col[None, :]), r226_value)
    }
    r227_z = tl.arange($(256))
    r227_value = (tl.load($(candidate_vbmm_53_ptr) + ((((r227_z / $(32)) % $(8)) * $(32)) + (((r227_z / $(8)) % $(4)) * $(8)) + (r227_z % $(8))), dtype=tl.float32) * 0.07216878364870322)
    r227_value = r227_value.to(tl.float32)
    r227_value = tl.broadcast_to(r227_value, [$(256)])
    tl.store($(candidate_vmul_54_ptr) + r227_z, r227_value)
    r228_z = tl.arange($(4))
    r228_value = (tl.cast(r228_z, tl.int64) + tl.full([], $(8), dtype=tl.int64))
    r228_value = tl.broadcast_to(r228_value, [$(4)])
    tl.store($(candidate_vadd_40_ptr) + r228_z, r228_value)
    r229_z = tl.arange($(4))
    r229_value = (tl.load($(candidate_vadd_40_ptr) + r229_z, dtype=tl.int64) - tl.full([], $(4), dtype=tl.int64))
    r229_value = tl.broadcast_to(r229_value, [$(4)])
    tl.store($(candidate_vsub_34_ptr) + r229_z, r229_value)
    r230_z = tl.arange($(32))
    r230_value = (tl.cast((r230_z % $(8)), tl.int64) > tl.load($(candidate_vsub_34_ptr) + ((r230_z / $(8)) % $(4)), dtype=tl.int64))
    r230_value = tl.broadcast_to(r230_value, [$(32)])
    tl.store($(candidate_vgt_19_ptr) + r230_z, r230_value)
    r231_z = tl.arange($(256))
    r231_value = tl.where(tl.load($(candidate_vgt_19_ptr) + ((((r231_z / $(8)) % $(4)) * $(8)) + (r231_z % $(8))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_54_ptr) + ((((r231_z / $(32)) % $(8)) * $(32)) + (((r231_z / $(8)) % $(4)) * $(8)) + (r231_z % $(8))), dtype=tl.float32))
    r231_value = r231_value.to(tl.float32)
    r231_value = tl.broadcast_to(r231_value, [$(256)])
    tl.store($(candidate_vmasked_fill_19_ptr) + r231_z, r231_value)
    r232_z = tl.arange($(256))
    r232_lane = tl.arange($(8))
    r232_row_values = tl.load($(candidate_vmasked_fill_19_ptr) + ((((r232_z[:, None] / $(32)) % $(8)) * $(32)) + (((r232_z[:, None] / $(8)) % $(4)) * $(8)) + (r232_lane[None, :] % $(8))), dtype=tl.float32)
    r232_maximum = tl.max(r232_row_values, axis=1, keep_dims=true)
    r232_weights = tl.exp(r232_row_values - r232_maximum)
    r232_total = tl.sum(r232_weights, axis=1)
    r232_maximum = tl.max(r232_row_values, axis=1)
    r232_value = tl.exp(tl.load($(candidate_vmasked_fill_19_ptr) + ((((r232_z / $(32)) % $(8)) * $(32)) + (((r232_z / $(8)) % $(4)) * $(8)) + (r232_z % $(8))), dtype=tl.float32) - r232_maximum) / r232_total
    r232_value = r232_value.to(tl.float32)
    r232_value = tl.broadcast_to(r232_value, [$(256)])
    tl.store($(candidate_vsoftmax_16_ptr) + r232_z, r232_value)
    r233_z = tl.arange($(256))
    r233_value = tl.load($(candidate_vsoftmax_16_ptr) + ((((r233_z / $(32)) % $(8)) * $(32)) + (((r233_z / $(8)) % $(4)) * $(8)) + (r233_z % $(8))), dtype=tl.float32)
    r233_value = r233_value.to(tl.bfloat16)
    r233_value = tl.broadcast_to(r233_value, [$(256)])
    tl.store($(candidate_vto_copy_127_ptr) + r233_z, r233_value)
    r234_row = tl.arange($(4))
    r234_col = tl.arange($(128))
    r234_lane = tl.arange($(8))
    for r234_group in range($(8)) {
      r234_left = tl.load($(candidate_vto_copy_127_ptr) + (((((((r234_group % $(8)) * $(32)) + (r234_row[:, None] * $(8)) + r234_lane[None, :]) / $(32)) % $(8)) * $(32)) + ((((((r234_group % $(8)) * $(32)) + (r234_row[:, None] * $(8)) + r234_lane[None, :]) / $(8)) % $(4)) * $(8)) + ((((r234_group % $(8)) * $(32)) + (r234_row[:, None] * $(8)) + r234_lane[None, :]) % $(8))), dtype=tl.bfloat16)
      r234_right = tl.load($(candidate_vbmm_50_ptr) + (((((((r234_group % $(8)) * $(1024)) + (r234_lane[:, None] * $(128)) + r234_col[None, :]) / $(128)) % $(8)) * $(1024)) + ((((((r234_group % $(8)) * $(1024)) + (r234_lane[:, None] * $(128)) + r234_col[None, :]) / $(1024)) % $(8)) * $(128)) + ((((r234_group % $(8)) * $(1024)) + (r234_lane[:, None] * $(128)) + r234_col[None, :]) % $(128)) + $(49152)), dtype=tl.bfloat16)
      r234_value = tl.dot(r234_left, r234_right)
      r234_value = r234_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_54_ptr) + ((r234_group * $(512)) + (r234_row[:, None] * $(128)) + r234_col[None, :]), r234_value)
    }
    r235_z = tl.arange($(32))
    r235_lane = tl.arange($(8))
    r235_row_values = tl.load($(candidate_vmasked_fill_19_ptr) + ((((r235_z[:, None] / $(4)) % $(8)) * $(32)) + ((r235_z[:, None] % $(4)) * $(8)) + (r235_lane[None, :] % $(8))), dtype=tl.float32)
    r235_maximum = tl.max(r235_row_values, axis=1, keep_dims=true)
    r235_weights = tl.exp(r235_row_values - r235_maximum)
    r235_total = tl.sum(r235_weights, axis=1)
    r235_maximum = tl.max(r235_row_values, axis=1)
    r235_value = r235_maximum + tl.log(r235_total)
    r235_value = r235_value.to(tl.float32)
    r235_value = tl.broadcast_to(r235_value, [$(32)])
    tl.store($(candidate_vlogsumexp_16_ptr) + r235_z, r235_value)
    r236_z = tl.arange($(24576))
    r236_value = tl.load($(candidate_vcat_25_ptr) + ((((r236_z / $(192)) % $(16)) * $(1536)) + (((r236_z / $(3072)) % $(8)) * $(192)) + (r236_z % $(192)) + $(24576)), dtype=tl.bfloat16)
    r236_value = r236_value.to(tl.float32)
    r236_value = tl.broadcast_to(r236_value, [$(24576)])
    tl.store($(candidate_vto_copy_128_ptr) + r236_z, r236_value)
    r237_z = tl.arange($(49152))
    r237_value = tl.load($(candidate_vcat_26_ptr) + ((((r237_z / $(192)) % $(32)) * $(1536)) + (((r237_z / $(6144)) % $(8)) * $(192)) + (r237_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r237_value = r237_value.to(tl.float32)
    r237_value = tl.broadcast_to(r237_value, [$(49152)])
    tl.store($(candidate_vto_copy_129_ptr) + r237_z, r237_value)
    r238_row = tl.arange($(16))
    r238_col = tl.arange($(32))
    r238_lane = tl.arange($(192))
    for r238_group in range($(8)) {
      r238_left = tl.load($(candidate_vto_copy_128_ptr) + (((((((r238_group % $(8)) * $(3072)) + (r238_row[:, None] * $(192)) + r238_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r238_group % $(8)) * $(3072)) + (r238_row[:, None] * $(192)) + r238_lane[None, :]) / $(192)) % $(16)) * $(192)) + ((((r238_group % $(8)) * $(3072)) + (r238_row[:, None] * $(192)) + r238_lane[None, :]) % $(192))), dtype=tl.float32)
      r238_right = tl.load($(candidate_vto_copy_129_ptr) + (((((((r238_group % $(8)) * $(6144)) + (r238_lane[:, None] * $(32)) + r238_col[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r238_group % $(8)) * $(6144)) + (r238_lane[:, None] * $(32)) + r238_col[None, :]) % $(32)) * $(192)) + (((((r238_group % $(8)) * $(6144)) + (r238_lane[:, None] * $(32)) + r238_col[None, :]) / $(32)) % $(192))), dtype=tl.float32)
      r238_value = tl.dot(r238_left, r238_right)
      r238_value = r238_value.to(tl.float32)
      tl.store($(candidate_vbmm_55_ptr) + ((r238_group * $(512)) + (r238_row[:, None] * $(32)) + r238_col[None, :]), r238_value)
    }
    r239_z = tl.arange($(4096))
    r239_value = (tl.load($(candidate_vbmm_55_ptr) + ((((r239_z / $(512)) % $(8)) * $(512)) + (((r239_z / $(32)) % $(16)) * $(32)) + (r239_z % $(32))), dtype=tl.float32) * 0.07216878364870322)
    r239_value = r239_value.to(tl.float32)
    r239_value = tl.broadcast_to(r239_value, [$(4096)])
    tl.store($(candidate_vmul_55_ptr) + r239_z, r239_value)
    r240_z = tl.arange($(16))
    r240_value = (tl.cast(r240_z, tl.int64) + tl.full([], $(32), dtype=tl.int64))
    r240_value = tl.broadcast_to(r240_value, [$(16)])
    tl.store($(candidate_vadd_41_ptr) + r240_z, r240_value)
    r241_z = tl.arange($(16))
    r241_value = (tl.load($(candidate_vadd_41_ptr) + r241_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r241_value = tl.broadcast_to(r241_value, [$(16)])
    tl.store($(candidate_vsub_35_ptr) + r241_z, r241_value)
    r242_z = tl.arange($(512))
    r242_value = (tl.cast((r242_z % $(32)), tl.int64) > tl.load($(candidate_vsub_35_ptr) + ((r242_z / $(32)) % $(16)), dtype=tl.int64))
    r242_value = tl.broadcast_to(r242_value, [$(512)])
    tl.store($(candidate_vgt_20_ptr) + r242_z, r242_value)
    r243_z = tl.arange($(4096))
    r243_value = tl.where(tl.load($(candidate_vgt_20_ptr) + ((((r243_z / $(32)) % $(16)) * $(32)) + (r243_z % $(32))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_55_ptr) + ((((r243_z / $(512)) % $(8)) * $(512)) + (((r243_z / $(32)) % $(16)) * $(32)) + (r243_z % $(32))), dtype=tl.float32))
    r243_value = r243_value.to(tl.float32)
    r243_value = tl.broadcast_to(r243_value, [$(4096)])
    tl.store($(candidate_vmasked_fill_20_ptr) + r243_z, r243_value)
    r244_z = tl.arange($(4096))
    r244_lane = tl.arange($(32))
    r244_row_values = tl.load($(candidate_vmasked_fill_20_ptr) + ((((r244_z[:, None] / $(512)) % $(8)) * $(512)) + (((r244_z[:, None] / $(32)) % $(16)) * $(32)) + (r244_lane[None, :] % $(32))), dtype=tl.float32)
    r244_maximum = tl.max(r244_row_values, axis=1, keep_dims=true)
    r244_weights = tl.exp(r244_row_values - r244_maximum)
    r244_total = tl.sum(r244_weights, axis=1)
    r244_maximum = tl.max(r244_row_values, axis=1)
    r244_value = tl.exp(tl.load($(candidate_vmasked_fill_20_ptr) + ((((r244_z / $(512)) % $(8)) * $(512)) + (((r244_z / $(32)) % $(16)) * $(32)) + (r244_z % $(32))), dtype=tl.float32) - r244_maximum) / r244_total
    r244_value = r244_value.to(tl.float32)
    r244_value = tl.broadcast_to(r244_value, [$(4096)])
    tl.store($(candidate_vsoftmax_17_ptr) + r244_z, r244_value)
    r245_z = tl.arange($(4096))
    r245_value = tl.load($(candidate_vsoftmax_17_ptr) + ((((r245_z / $(512)) % $(8)) * $(512)) + (((r245_z / $(32)) % $(16)) * $(32)) + (r245_z % $(32))), dtype=tl.float32)
    r245_value = r245_value.to(tl.bfloat16)
    r245_value = tl.broadcast_to(r245_value, [$(4096)])
    tl.store($(candidate_vto_copy_130_ptr) + r245_z, r245_value)
    r246_row = tl.arange($(16))
    r246_col = tl.arange($(128))
    r246_lane = tl.arange($(32))
    for r246_group in range($(8)) {
      r246_left = tl.load($(candidate_vto_copy_130_ptr) + (((((((r246_group % $(8)) * $(512)) + (r246_row[:, None] * $(32)) + r246_lane[None, :]) / $(512)) % $(8)) * $(512)) + ((((((r246_group % $(8)) * $(512)) + (r246_row[:, None] * $(32)) + r246_lane[None, :]) / $(32)) % $(16)) * $(32)) + ((((r246_group % $(8)) * $(512)) + (r246_row[:, None] * $(32)) + r246_lane[None, :]) % $(32))), dtype=tl.bfloat16)
      r246_right = tl.load($(candidate_vbmm_50_ptr) + (((((((r246_group % $(8)) * $(4096)) + (r246_lane[:, None] * $(128)) + r246_col[None, :]) / $(128)) % $(32)) * $(1024)) + ((((((r246_group % $(8)) * $(4096)) + (r246_lane[:, None] * $(128)) + r246_col[None, :]) / $(4096)) % $(8)) * $(128)) + ((((r246_group % $(8)) * $(4096)) + (r246_lane[:, None] * $(128)) + r246_col[None, :]) % $(128)) + $(65536)), dtype=tl.bfloat16)
      r246_value = tl.dot(r246_left, r246_right)
      r246_value = r246_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_56_ptr) + ((r246_group * $(2048)) + (r246_row[:, None] * $(128)) + r246_col[None, :]), r246_value)
    }
    r247_z = tl.arange($(128))
    r247_lane = tl.arange($(32))
    r247_row_values = tl.load($(candidate_vmasked_fill_20_ptr) + ((((r247_z[:, None] / $(16)) % $(8)) * $(512)) + ((r247_z[:, None] % $(16)) * $(32)) + (r247_lane[None, :] % $(32))), dtype=tl.float32)
    r247_maximum = tl.max(r247_row_values, axis=1, keep_dims=true)
    r247_weights = tl.exp(r247_row_values - r247_maximum)
    r247_total = tl.sum(r247_weights, axis=1)
    r247_maximum = tl.max(r247_row_values, axis=1)
    r247_value = r247_maximum + tl.log(r247_total)
    r247_value = r247_value.to(tl.float32)
    r247_value = tl.broadcast_to(r247_value, [$(128)])
    tl.store($(candidate_vlogsumexp_17_ptr) + r247_z, r247_value)
    r248_z = tl.arange($(32768))
    r248_value = tl.where(((r248_z / $(1024)) % $(32)) < $(12), tl.load($(candidate_vbmm_52_ptr) + ((((((((r248_z / $(128)) % $(8)) * $(1536)) + ((((r248_z / $(1024)) % $(32)) - $(0)) * $(128)) + (r248_z % $(128))) / $(1536)) % $(8)) * $(1536)) + ((((((r248_z / $(128)) % $(8)) * $(12)) + (((r248_z / $(1024)) % $(32)) - $(0))) % $(12)) * $(128)) + (r248_z % $(128))), dtype=tl.bfloat16), tl.where(((r248_z / $(1024)) % $(32)) < $(16), tl.load($(candidate_vbmm_54_ptr) + ((((((((r248_z / $(128)) % $(8)) * $(512)) + ((((r248_z / $(1024)) % $(32)) - $(12)) * $(128)) + (r248_z % $(128))) / $(512)) % $(8)) * $(512)) + ((((((r248_z / $(128)) % $(8)) * $(4)) + (((r248_z / $(1024)) % $(32)) - $(12))) % $(4)) * $(128)) + (r248_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vbmm_56_ptr) + ((((((((r248_z / $(128)) % $(8)) * $(2048)) + ((((r248_z / $(1024)) % $(32)) - $(16)) * $(128)) + (r248_z % $(128))) / $(2048)) % $(8)) * $(2048)) + ((((((r248_z / $(128)) % $(8)) * $(16)) + (((r248_z / $(1024)) % $(32)) - $(16))) % $(16)) * $(128)) + (r248_z % $(128))), dtype=tl.bfloat16)))
    r248_value = r248_value.to(tl.bfloat16)
    r248_value = tl.broadcast_to(r248_value, [$(32768)])
    tl.store($(candidate_vcat_27_ptr) + r248_z, r248_value)
    r249_z = tl.arange($(256))
    r249_value = tl.where(((r249_z / $(8)) % $(32)) < $(12), tl.load($(candidate_vlogsumexp_15_ptr) + (((r249_z % $(8)) * $(12)) + ((((r249_z / $(8)) % $(32)) - $(0)) % $(12))), dtype=tl.float32), tl.where(((r249_z / $(8)) % $(32)) < $(16), tl.load($(candidate_vlogsumexp_16_ptr) + (((r249_z % $(8)) * $(4)) + ((((r249_z / $(8)) % $(32)) - $(12)) % $(4))), dtype=tl.float32), tl.load($(candidate_vlogsumexp_17_ptr) + (((r249_z % $(8)) * $(16)) + ((((r249_z / $(8)) % $(32)) - $(16)) % $(16))), dtype=tl.float32)))
    r249_value = r249_value.to(tl.float32)
    r249_value = tl.broadcast_to(r249_value, [$(256)])
    tl.store($(candidate_vcat_28_ptr) + r249_z, r249_value)
    r250_z = tl.arange($(18432))
    r250_value = tl.load($(candidate_vcat_25_ptr) + ((((r250_z / $(192)) % $(12)) * $(1536)) + (((r250_z / $(2304)) % $(8)) * $(192)) + (r250_z % $(192))), dtype=tl.bfloat16)
    r250_value = r250_value.to(tl.float32)
    r250_value = tl.broadcast_to(r250_value, [$(18432)])
    tl.store($(candidate_vto_copy_131_ptr) + r250_z, r250_value)
    r251_z = tl.arange($(36864))
    r251_value = tl.load($(candidate_vcat_26_ptr) + ((((r251_z / $(192)) % $(24)) * $(1536)) + (((r251_z / $(4608)) % $(8)) * $(192)) + (r251_z % $(192))), dtype=tl.bfloat16)
    r251_value = r251_value.to(tl.float32)
    r251_value = tl.broadcast_to(r251_value, [$(36864)])
    tl.store($(candidate_vto_copy_132_ptr) + r251_z, r251_value)
    r252_z = tl.arange($(24576))
    r252_value = tl.load($(candidate_vbmm_50_ptr) + ((((r252_z / $(128)) % $(24)) * $(1024)) + (((r252_z / $(3072)) % $(8)) * $(128)) + (r252_z % $(128))), dtype=tl.bfloat16)
    r252_value = r252_value.to(tl.float32)
    r252_value = tl.broadcast_to(r252_value, [$(24576)])
    tl.store($(candidate_vto_copy_133_ptr) + r252_z, r252_value)
    r253_z = tl.arange($(12288))
    r253_value = tl.load($(candidate_vmm_28_ptr) + ((((r253_z / $(128)) % $(12)) * $(1024)) + (((r253_z / $(1536)) % $(8)) * $(128)) + (r253_z % $(128)) + $(65536)), dtype=tl.bfloat16)
    r253_value = r253_value.to(tl.float32)
    r253_value = tl.broadcast_to(r253_value, [$(12288)])
    tl.store($(candidate_vto_copy_134_ptr) + r253_z, r253_value)
    r254_z = tl.arange($(12288))
    r254_value = tl.load($(candidate_vcat_27_ptr) + ((((r254_z / $(128)) % $(12)) * $(1024)) + (((r254_z / $(1536)) % $(8)) * $(128)) + (r254_z % $(128))), dtype=tl.bfloat16)
    r254_value = r254_value.to(tl.float32)
    r254_value = tl.broadcast_to(r254_value, [$(12288)])
    tl.store($(candidate_vto_copy_135_ptr) + r254_z, r254_value)
    r255_row = tl.arange($(12))
    r255_col = tl.arange($(24))
    r255_lane = tl.arange($(192))
    for r255_group in range($(8)) {
      r255_left = tl.load($(candidate_vto_copy_131_ptr) + (((((((r255_group % $(8)) * $(2304)) + (r255_row[:, None] * $(192)) + r255_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r255_group % $(8)) * $(2304)) + (r255_row[:, None] * $(192)) + r255_lane[None, :]) / $(192)) % $(12)) * $(192)) + ((((r255_group % $(8)) * $(2304)) + (r255_row[:, None] * $(192)) + r255_lane[None, :]) % $(192))), dtype=tl.float32)
      r255_right = tl.load($(candidate_vto_copy_132_ptr) + (((((((r255_group % $(8)) * $(4608)) + (r255_lane[:, None] * $(24)) + r255_col[None, :]) / $(4608)) % $(8)) * $(4608)) + (((((r255_group % $(8)) * $(4608)) + (r255_lane[:, None] * $(24)) + r255_col[None, :]) % $(24)) * $(192)) + (((((r255_group % $(8)) * $(4608)) + (r255_lane[:, None] * $(24)) + r255_col[None, :]) / $(24)) % $(192))), dtype=tl.float32)
      r255_value = tl.dot(r255_left, r255_right)
      r255_value = r255_value.to(tl.float32)
      tl.store($(candidate_vbmm_57_ptr) + ((r255_group * $(288)) + (r255_row[:, None] * $(24)) + r255_col[None, :]), r255_value)
    }
    r256_z = tl.arange($(2304))
    r256_value = (tl.load($(candidate_vbmm_57_ptr) + ((((r256_z / $(288)) % $(8)) * $(288)) + (((r256_z / $(24)) % $(12)) * $(24)) + (r256_z % $(24))), dtype=tl.float32) * 0.07216878364870322)
    r256_value = r256_value.to(tl.float32)
    r256_value = tl.broadcast_to(r256_value, [$(2304)])
    tl.store($(candidate_vmul_56_ptr) + r256_z, r256_value)
    r257_z = tl.arange($(12))
    r257_value = (tl.cast(r257_z, tl.int64) + tl.full([], $(24), dtype=tl.int64))
    r257_value = tl.broadcast_to(r257_value, [$(12)])
    tl.store($(candidate_vadd_42_ptr) + r257_z, r257_value)
    r258_z = tl.arange($(12))
    r258_value = (tl.load($(candidate_vadd_42_ptr) + r258_z, dtype=tl.int64) - tl.full([], $(12), dtype=tl.int64))
    r258_value = tl.broadcast_to(r258_value, [$(12)])
    tl.store($(candidate_vsub_36_ptr) + r258_z, r258_value)
    r259_z = tl.arange($(288))
    r259_value = (tl.cast((r259_z % $(24)), tl.int64) > tl.load($(candidate_vsub_36_ptr) + ((r259_z / $(24)) % $(12)), dtype=tl.int64))
    r259_value = tl.broadcast_to(r259_value, [$(288)])
    tl.store($(candidate_vgt_21_ptr) + r259_z, r259_value)
    r260_z = tl.arange($(2304))
    r260_value = tl.where(tl.load($(candidate_vgt_21_ptr) + ((((r260_z / $(24)) % $(12)) * $(24)) + (r260_z % $(24))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_56_ptr) + ((((r260_z / $(288)) % $(8)) * $(288)) + (((r260_z / $(24)) % $(12)) * $(24)) + (r260_z % $(24))), dtype=tl.float32))
    r260_value = r260_value.to(tl.float32)
    r260_value = tl.broadcast_to(r260_value, [$(2304)])
    tl.store($(candidate_vmasked_fill_21_ptr) + r260_z, r260_value)
    r261_z = tl.arange($(2304))
    r261_value = (tl.load($(candidate_vmasked_fill_21_ptr) + ((((r261_z / $(288)) % $(8)) * $(288)) + (((r261_z / $(24)) % $(12)) * $(24)) + (r261_z % $(24))), dtype=tl.float32) - tl.load($(candidate_vcat_28_ptr) + ((((r261_z / $(24)) % $(12)) * $(8)) + ((r261_z / $(288)) % $(8))), dtype=tl.float32))
    r261_value = r261_value.to(tl.float32)
    r261_value = tl.broadcast_to(r261_value, [$(2304)])
    tl.store($(candidate_vsub_37_ptr) + r261_z, r261_value)
    r262_z = tl.arange($(2304))
    r262_value = tl.exp(tl.load($(candidate_vsub_37_ptr) + ((((r262_z / $(288)) % $(8)) * $(288)) + (((r262_z / $(24)) % $(12)) * $(24)) + (r262_z % $(24))), dtype=tl.float32))
    r262_value = r262_value.to(tl.float32)
    r262_value = tl.broadcast_to(r262_value, [$(2304)])
    tl.store($(candidate_vexp_3_ptr) + r262_z, r262_value)
    r263_z = tl.arange($(12288))
    r263_value = (tl.load($(candidate_vto_copy_134_ptr) + ((((r263_z / $(1536)) % $(8)) * $(1536)) + (((r263_z / $(128)) % $(12)) * $(128)) + (r263_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_135_ptr) + ((((r263_z / $(1536)) % $(8)) * $(1536)) + (((r263_z / $(128)) % $(12)) * $(128)) + (r263_z % $(128))), dtype=tl.float32))
    r263_value = r263_value.to(tl.float32)
    r263_value = tl.broadcast_to(r263_value, [$(12288)])
    tl.store($(candidate_vmul_57_ptr) + r263_z, r263_value)
    r264_z = tl.arange($(96))
    r264_lane = tl.arange($(128))
    r264_value = tl.sum(tl.load($(candidate_vmul_57_ptr) + ((((r264_z[:, None] / $(12)) % $(8)) * $(1536)) + ((r264_z[:, None] % $(12)) * $(128)) + r264_lane[None, :]), dtype=tl.float32), axis=1)
    r264_value = r264_value.to(tl.float32)
    r264_value = tl.broadcast_to(r264_value, [$(96)])
    tl.store($(candidate_vsum_5_ptr) + r264_z, r264_value)
    r265_row = tl.arange($(12))
    r265_col = tl.arange($(24))
    r265_lane = tl.arange($(128))
    for r265_group in range($(8)) {
      r265_left = tl.load($(candidate_vto_copy_134_ptr) + (((((((r265_group % $(8)) * $(1536)) + (r265_row[:, None] * $(128)) + r265_lane[None, :]) / $(1536)) % $(8)) * $(1536)) + ((((((r265_group % $(8)) * $(1536)) + (r265_row[:, None] * $(128)) + r265_lane[None, :]) / $(128)) % $(12)) * $(128)) + ((((r265_group % $(8)) * $(1536)) + (r265_row[:, None] * $(128)) + r265_lane[None, :]) % $(128))), dtype=tl.float32)
      r265_right = tl.load($(candidate_vto_copy_133_ptr) + (((((((r265_group % $(8)) * $(3072)) + (r265_lane[:, None] * $(24)) + r265_col[None, :]) / $(3072)) % $(8)) * $(3072)) + (((((r265_group % $(8)) * $(3072)) + (r265_lane[:, None] * $(24)) + r265_col[None, :]) % $(24)) * $(128)) + (((((r265_group % $(8)) * $(3072)) + (r265_lane[:, None] * $(24)) + r265_col[None, :]) / $(24)) % $(128))), dtype=tl.float32)
      r265_value = tl.dot(r265_left, r265_right)
      r265_value = r265_value.to(tl.float32)
      tl.store($(candidate_vbmm_58_ptr) + ((r265_group * $(288)) + (r265_row[:, None] * $(24)) + r265_col[None, :]), r265_value)
    }
    r266_z = tl.arange($(2304))
    r266_value = (tl.load($(candidate_vbmm_58_ptr) + ((((r266_z / $(288)) % $(8)) * $(288)) + (((r266_z / $(24)) % $(12)) * $(24)) + (r266_z % $(24))), dtype=tl.float32) - tl.load($(candidate_vsum_5_ptr) + ((((r266_z / $(288)) % $(8)) * $(12)) + ((r266_z / $(24)) % $(12))), dtype=tl.float32))
    r266_value = r266_value.to(tl.float32)
    r266_value = tl.broadcast_to(r266_value, [$(2304)])
    tl.store($(candidate_vsub_38_ptr) + r266_z, r266_value)
    r267_z = tl.arange($(2304))
    r267_value = (tl.load($(candidate_vexp_3_ptr) + ((((r267_z / $(288)) % $(8)) * $(288)) + (((r267_z / $(24)) % $(12)) * $(24)) + (r267_z % $(24))), dtype=tl.float32) * tl.load($(candidate_vsub_38_ptr) + ((((r267_z / $(288)) % $(8)) * $(288)) + (((r267_z / $(24)) % $(12)) * $(24)) + (r267_z % $(24))), dtype=tl.float32))
    r267_value = r267_value.to(tl.float32)
    r267_value = tl.broadcast_to(r267_value, [$(2304)])
    tl.store($(candidate_vmul_58_ptr) + r267_z, r267_value)
    r268_row = tl.arange($(12))
    r268_col = tl.arange($(192))
    r268_lane = tl.arange($(24))
    for r268_group in range($(8)) {
      r268_left = tl.load($(candidate_vmul_58_ptr) + (((((((r268_group % $(8)) * $(288)) + (r268_row[:, None] * $(24)) + r268_lane[None, :]) / $(288)) % $(8)) * $(288)) + ((((((r268_group % $(8)) * $(288)) + (r268_row[:, None] * $(24)) + r268_lane[None, :]) / $(24)) % $(12)) * $(24)) + ((((r268_group % $(8)) * $(288)) + (r268_row[:, None] * $(24)) + r268_lane[None, :]) % $(24))), dtype=tl.float32)
      r268_right = tl.load($(candidate_vto_copy_132_ptr) + (((((((r268_group % $(8)) * $(4608)) + (r268_lane[:, None] * $(192)) + r268_col[None, :]) / $(4608)) % $(8)) * $(4608)) + ((((((r268_group % $(8)) * $(4608)) + (r268_lane[:, None] * $(192)) + r268_col[None, :]) / $(192)) % $(24)) * $(192)) + ((((r268_group % $(8)) * $(4608)) + (r268_lane[:, None] * $(192)) + r268_col[None, :]) % $(192))), dtype=tl.float32)
      r268_value = tl.dot(r268_left, r268_right)
      r268_value = r268_value.to(tl.float32)
      tl.store($(candidate_vbmm_59_ptr) + ((r268_group * $(2304)) + (r268_row[:, None] * $(192)) + r268_col[None, :]), r268_value)
    }
    r269_z = tl.arange($(18432))
    r269_value = (tl.load($(candidate_vbmm_59_ptr) + ((((r269_z / $(2304)) % $(8)) * $(2304)) + (((r269_z / $(192)) % $(12)) * $(192)) + (r269_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r269_value = r269_value.to(tl.float32)
    r269_value = tl.broadcast_to(r269_value, [$(18432)])
    tl.store($(candidate_vmul_59_ptr) + r269_z, r269_value)
    r270_z = tl.arange($(18432))
    r270_value = tl.load($(candidate_vmul_59_ptr) + ((((r270_z / $(192)) % $(8)) * $(2304)) + (((r270_z / $(1536)) % $(12)) * $(192)) + (r270_z % $(192))), dtype=tl.float32)
    r270_value = r270_value.to(tl.bfloat16)
    r270_value = tl.broadcast_to(r270_value, [$(18432)])
    tl.store($(candidate_vto_copy_136_ptr) + r270_z, r270_value)
    r271_row = tl.arange($(24))
    r271_col = tl.arange($(192))
    r271_lane = tl.arange($(12))
    for r271_group in range($(8)) {
      r271_left = tl.load($(candidate_vmul_58_ptr) + (((((((r271_group % $(8)) * $(288)) + (r271_row[:, None] * $(12)) + r271_lane[None, :]) / $(288)) % $(8)) * $(288)) + (((((r271_group % $(8)) * $(288)) + (r271_row[:, None] * $(12)) + r271_lane[None, :]) % $(12)) * $(24)) + (((((r271_group % $(8)) * $(288)) + (r271_row[:, None] * $(12)) + r271_lane[None, :]) / $(12)) % $(24))), dtype=tl.float32)
      r271_right = tl.load($(candidate_vto_copy_131_ptr) + (((((((r271_group % $(8)) * $(2304)) + (r271_lane[:, None] * $(192)) + r271_col[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r271_group % $(8)) * $(2304)) + (r271_lane[:, None] * $(192)) + r271_col[None, :]) / $(192)) % $(12)) * $(192)) + ((((r271_group % $(8)) * $(2304)) + (r271_lane[:, None] * $(192)) + r271_col[None, :]) % $(192))), dtype=tl.float32)
      r271_value = tl.dot(r271_left, r271_right)
      r271_value = r271_value.to(tl.float32)
      tl.store($(candidate_vbmm_60_ptr) + ((r271_group * $(4608)) + (r271_row[:, None] * $(192)) + r271_col[None, :]), r271_value)
    }
    r272_z = tl.arange($(36864))
    r272_value = (tl.load($(candidate_vbmm_60_ptr) + ((((r272_z / $(4608)) % $(8)) * $(4608)) + (((r272_z / $(192)) % $(24)) * $(192)) + (r272_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r272_value = r272_value.to(tl.float32)
    r272_value = tl.broadcast_to(r272_value, [$(36864)])
    tl.store($(candidate_vmul_60_ptr) + r272_z, r272_value)
    r273_z = tl.arange($(36864))
    r273_value = tl.load($(candidate_vmul_60_ptr) + ((((r273_z / $(192)) % $(8)) * $(4608)) + (((r273_z / $(1536)) % $(24)) * $(192)) + (r273_z % $(192))), dtype=tl.float32)
    r273_value = r273_value.to(tl.bfloat16)
    r273_value = tl.broadcast_to(r273_value, [$(36864)])
    tl.store($(candidate_vto_copy_137_ptr) + r273_z, r273_value)
    r274_row = tl.arange($(24))
    r274_col = tl.arange($(128))
    r274_lane = tl.arange($(12))
    for r274_group in range($(8)) {
      r274_left = tl.load($(candidate_vexp_3_ptr) + (((((((r274_group % $(8)) * $(288)) + (r274_row[:, None] * $(12)) + r274_lane[None, :]) / $(288)) % $(8)) * $(288)) + (((((r274_group % $(8)) * $(288)) + (r274_row[:, None] * $(12)) + r274_lane[None, :]) % $(12)) * $(24)) + (((((r274_group % $(8)) * $(288)) + (r274_row[:, None] * $(12)) + r274_lane[None, :]) / $(12)) % $(24))), dtype=tl.float32)
      r274_right = tl.load($(candidate_vto_copy_134_ptr) + (((((((r274_group % $(8)) * $(1536)) + (r274_lane[:, None] * $(128)) + r274_col[None, :]) / $(1536)) % $(8)) * $(1536)) + ((((((r274_group % $(8)) * $(1536)) + (r274_lane[:, None] * $(128)) + r274_col[None, :]) / $(128)) % $(12)) * $(128)) + ((((r274_group % $(8)) * $(1536)) + (r274_lane[:, None] * $(128)) + r274_col[None, :]) % $(128))), dtype=tl.float32)
      r274_value = tl.dot(r274_left, r274_right)
      r274_value = r274_value.to(tl.float32)
      tl.store($(candidate_vbmm_61_ptr) + ((r274_group * $(3072)) + (r274_row[:, None] * $(128)) + r274_col[None, :]), r274_value)
    }
    r275_z = tl.arange($(24576))
    r275_value = tl.load($(candidate_vbmm_61_ptr) + ((((r275_z / $(128)) % $(8)) * $(3072)) + (((r275_z / $(1024)) % $(24)) * $(128)) + (r275_z % $(128))), dtype=tl.float32)
    r275_value = r275_value.to(tl.bfloat16)
    r275_value = tl.broadcast_to(r275_value, [$(24576)])
    tl.store($(candidate_vto_copy_138_ptr) + r275_z, r275_value)
    r276_z = tl.arange($(73728))
    r276_value = tl.where(((r276_z / $(1536)) % $(48)) < $(24), tl.load($(candidate_vto_copy_137_ptr) + ((((((r276_z / $(1536)) % $(48)) - $(0)) % $(24)) * $(1536)) + (((r276_z / $(192)) % $(8)) * $(192)) + (r276_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r276_value = r276_value.to(tl.bfloat16)
    r276_value = tl.broadcast_to(r276_value, [$(73728)])
    tl.store($(candidate_vcat_29_ptr) + r276_z, r276_value)
    r277_z = tl.arange($(49152))
    r277_value = tl.where(((r277_z / $(1024)) % $(48)) < $(24), tl.load($(candidate_vto_copy_138_ptr) + ((((((r277_z / $(1024)) % $(48)) - $(0)) % $(24)) * $(1024)) + (((r277_z / $(128)) % $(8)) * $(128)) + (r277_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r277_value = r277_value.to(tl.bfloat16)
    r277_value = tl.broadcast_to(r277_value, [$(49152)])
    tl.store($(candidate_vcat_30_ptr) + r277_z, r277_value)
    r278_z = tl.arange($(6144))
    r278_value = tl.load($(candidate_vcat_25_ptr) + ((((r278_z / $(192)) % $(4)) * $(1536)) + (((r278_z / $(768)) % $(8)) * $(192)) + (r278_z % $(192)) + $(18432)), dtype=tl.bfloat16)
    r278_value = r278_value.to(tl.float32)
    r278_value = tl.broadcast_to(r278_value, [$(6144)])
    tl.store($(candidate_vto_copy_139_ptr) + r278_z, r278_value)
    r279_z = tl.arange($(12288))
    r279_value = tl.load($(candidate_vcat_26_ptr) + ((((r279_z / $(192)) % $(8)) * $(1536)) + (((r279_z / $(1536)) % $(8)) * $(192)) + (r279_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r279_value = r279_value.to(tl.float32)
    r279_value = tl.broadcast_to(r279_value, [$(12288)])
    tl.store($(candidate_vto_copy_140_ptr) + r279_z, r279_value)
    r280_z = tl.arange($(8192))
    r280_value = tl.load($(candidate_vbmm_50_ptr) + ((((r280_z / $(128)) % $(8)) * $(1024)) + (((r280_z / $(1024)) % $(8)) * $(128)) + (r280_z % $(128)) + $(49152)), dtype=tl.bfloat16)
    r280_value = r280_value.to(tl.float32)
    r280_value = tl.broadcast_to(r280_value, [$(8192)])
    tl.store($(candidate_vto_copy_141_ptr) + r280_z, r280_value)
    r281_z = tl.arange($(4096))
    r281_value = tl.load($(candidate_vmm_28_ptr) + ((((r281_z / $(128)) % $(4)) * $(1024)) + (((r281_z / $(512)) % $(8)) * $(128)) + (r281_z % $(128)) + $(77824)), dtype=tl.bfloat16)
    r281_value = r281_value.to(tl.float32)
    r281_value = tl.broadcast_to(r281_value, [$(4096)])
    tl.store($(candidate_vto_copy_142_ptr) + r281_z, r281_value)
    r282_z = tl.arange($(4096))
    r282_value = tl.load($(candidate_vcat_27_ptr) + ((((r282_z / $(128)) % $(4)) * $(1024)) + (((r282_z / $(512)) % $(8)) * $(128)) + (r282_z % $(128)) + $(12288)), dtype=tl.bfloat16)
    r282_value = r282_value.to(tl.float32)
    r282_value = tl.broadcast_to(r282_value, [$(4096)])
    tl.store($(candidate_vto_copy_143_ptr) + r282_z, r282_value)
    r283_row = tl.arange($(4))
    r283_col = tl.arange($(8))
    r283_lane = tl.arange($(192))
    for r283_group in range($(8)) {
      r283_left = tl.load($(candidate_vto_copy_139_ptr) + (((((((r283_group % $(8)) * $(768)) + (r283_row[:, None] * $(192)) + r283_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r283_group % $(8)) * $(768)) + (r283_row[:, None] * $(192)) + r283_lane[None, :]) / $(192)) % $(4)) * $(192)) + ((((r283_group % $(8)) * $(768)) + (r283_row[:, None] * $(192)) + r283_lane[None, :]) % $(192))), dtype=tl.float32)
      r283_right = tl.load($(candidate_vto_copy_140_ptr) + (((((((r283_group % $(8)) * $(1536)) + (r283_lane[:, None] * $(8)) + r283_col[None, :]) / $(1536)) % $(8)) * $(1536)) + (((((r283_group % $(8)) * $(1536)) + (r283_lane[:, None] * $(8)) + r283_col[None, :]) % $(8)) * $(192)) + (((((r283_group % $(8)) * $(1536)) + (r283_lane[:, None] * $(8)) + r283_col[None, :]) / $(8)) % $(192))), dtype=tl.float32)
      r283_value = tl.dot(r283_left, r283_right)
      r283_value = r283_value.to(tl.float32)
      tl.store($(candidate_vbmm_62_ptr) + ((r283_group * $(32)) + (r283_row[:, None] * $(8)) + r283_col[None, :]), r283_value)
    }
    r284_z = tl.arange($(256))
    r284_value = (tl.load($(candidate_vbmm_62_ptr) + ((((r284_z / $(32)) % $(8)) * $(32)) + (((r284_z / $(8)) % $(4)) * $(8)) + (r284_z % $(8))), dtype=tl.float32) * 0.07216878364870322)
    r284_value = r284_value.to(tl.float32)
    r284_value = tl.broadcast_to(r284_value, [$(256)])
    tl.store($(candidate_vmul_61_ptr) + r284_z, r284_value)
    r285_z = tl.arange($(4))
    r285_value = (tl.cast(r285_z, tl.int64) + tl.full([], $(8), dtype=tl.int64))
    r285_value = tl.broadcast_to(r285_value, [$(4)])
    tl.store($(candidate_vadd_43_ptr) + r285_z, r285_value)
    r286_z = tl.arange($(4))
    r286_value = (tl.load($(candidate_vadd_43_ptr) + r286_z, dtype=tl.int64) - tl.full([], $(4), dtype=tl.int64))
    r286_value = tl.broadcast_to(r286_value, [$(4)])
    tl.store($(candidate_vsub_39_ptr) + r286_z, r286_value)
    r287_z = tl.arange($(32))
    r287_value = (tl.cast((r287_z % $(8)), tl.int64) > tl.load($(candidate_vsub_39_ptr) + ((r287_z / $(8)) % $(4)), dtype=tl.int64))
    r287_value = tl.broadcast_to(r287_value, [$(32)])
    tl.store($(candidate_vgt_22_ptr) + r287_z, r287_value)
    r288_z = tl.arange($(256))
    r288_value = tl.where(tl.load($(candidate_vgt_22_ptr) + ((((r288_z / $(8)) % $(4)) * $(8)) + (r288_z % $(8))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_61_ptr) + ((((r288_z / $(32)) % $(8)) * $(32)) + (((r288_z / $(8)) % $(4)) * $(8)) + (r288_z % $(8))), dtype=tl.float32))
    r288_value = r288_value.to(tl.float32)
    r288_value = tl.broadcast_to(r288_value, [$(256)])
    tl.store($(candidate_vmasked_fill_22_ptr) + r288_z, r288_value)
    r289_z = tl.arange($(256))
    r289_value = (tl.load($(candidate_vmasked_fill_22_ptr) + ((((r289_z / $(32)) % $(8)) * $(32)) + (((r289_z / $(8)) % $(4)) * $(8)) + (r289_z % $(8))), dtype=tl.float32) - tl.load($(candidate_vcat_28_ptr) + ((((r289_z / $(8)) % $(4)) * $(8)) + ((r289_z / $(32)) % $(8)) + $(96)), dtype=tl.float32))
    r289_value = r289_value.to(tl.float32)
    r289_value = tl.broadcast_to(r289_value, [$(256)])
    tl.store($(candidate_vsub_40_ptr) + r289_z, r289_value)
    r290_z = tl.arange($(256))
    r290_value = tl.exp(tl.load($(candidate_vsub_40_ptr) + ((((r290_z / $(32)) % $(8)) * $(32)) + (((r290_z / $(8)) % $(4)) * $(8)) + (r290_z % $(8))), dtype=tl.float32))
    r290_value = r290_value.to(tl.float32)
    r290_value = tl.broadcast_to(r290_value, [$(256)])
    tl.store($(candidate_vexp_4_ptr) + r290_z, r290_value)
    r291_z = tl.arange($(4096))
    r291_value = (tl.load($(candidate_vto_copy_142_ptr) + ((((r291_z / $(512)) % $(8)) * $(512)) + (((r291_z / $(128)) % $(4)) * $(128)) + (r291_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_143_ptr) + ((((r291_z / $(512)) % $(8)) * $(512)) + (((r291_z / $(128)) % $(4)) * $(128)) + (r291_z % $(128))), dtype=tl.float32))
    r291_value = r291_value.to(tl.float32)
    r291_value = tl.broadcast_to(r291_value, [$(4096)])
    tl.store($(candidate_vmul_62_ptr) + r291_z, r291_value)
    r292_z = tl.arange($(32))
    r292_lane = tl.arange($(128))
    r292_value = tl.sum(tl.load($(candidate_vmul_62_ptr) + ((((r292_z[:, None] / $(4)) % $(8)) * $(512)) + ((r292_z[:, None] % $(4)) * $(128)) + r292_lane[None, :]), dtype=tl.float32), axis=1)
    r292_value = r292_value.to(tl.float32)
    r292_value = tl.broadcast_to(r292_value, [$(32)])
    tl.store($(candidate_vsum_6_ptr) + r292_z, r292_value)
    r293_row = tl.arange($(4))
    r293_col = tl.arange($(8))
    r293_lane = tl.arange($(128))
    for r293_group in range($(8)) {
      r293_left = tl.load($(candidate_vto_copy_142_ptr) + (((((((r293_group % $(8)) * $(512)) + (r293_row[:, None] * $(128)) + r293_lane[None, :]) / $(512)) % $(8)) * $(512)) + ((((((r293_group % $(8)) * $(512)) + (r293_row[:, None] * $(128)) + r293_lane[None, :]) / $(128)) % $(4)) * $(128)) + ((((r293_group % $(8)) * $(512)) + (r293_row[:, None] * $(128)) + r293_lane[None, :]) % $(128))), dtype=tl.float32)
      r293_right = tl.load($(candidate_vto_copy_141_ptr) + (((((((r293_group % $(8)) * $(1024)) + (r293_lane[:, None] * $(8)) + r293_col[None, :]) / $(1024)) % $(8)) * $(1024)) + (((((r293_group % $(8)) * $(1024)) + (r293_lane[:, None] * $(8)) + r293_col[None, :]) % $(8)) * $(128)) + (((((r293_group % $(8)) * $(1024)) + (r293_lane[:, None] * $(8)) + r293_col[None, :]) / $(8)) % $(128))), dtype=tl.float32)
      r293_value = tl.dot(r293_left, r293_right)
      r293_value = r293_value.to(tl.float32)
      tl.store($(candidate_vbmm_63_ptr) + ((r293_group * $(32)) + (r293_row[:, None] * $(8)) + r293_col[None, :]), r293_value)
    }
    r294_z = tl.arange($(256))
    r294_value = (tl.load($(candidate_vbmm_63_ptr) + ((((r294_z / $(32)) % $(8)) * $(32)) + (((r294_z / $(8)) % $(4)) * $(8)) + (r294_z % $(8))), dtype=tl.float32) - tl.load($(candidate_vsum_6_ptr) + ((((r294_z / $(32)) % $(8)) * $(4)) + ((r294_z / $(8)) % $(4))), dtype=tl.float32))
    r294_value = r294_value.to(tl.float32)
    r294_value = tl.broadcast_to(r294_value, [$(256)])
    tl.store($(candidate_vsub_41_ptr) + r294_z, r294_value)
    r295_z = tl.arange($(256))
    r295_value = (tl.load($(candidate_vexp_4_ptr) + ((((r295_z / $(32)) % $(8)) * $(32)) + (((r295_z / $(8)) % $(4)) * $(8)) + (r295_z % $(8))), dtype=tl.float32) * tl.load($(candidate_vsub_41_ptr) + ((((r295_z / $(32)) % $(8)) * $(32)) + (((r295_z / $(8)) % $(4)) * $(8)) + (r295_z % $(8))), dtype=tl.float32))
    r295_value = r295_value.to(tl.float32)
    r295_value = tl.broadcast_to(r295_value, [$(256)])
    tl.store($(candidate_vmul_63_ptr) + r295_z, r295_value)
    r296_row = tl.arange($(4))
    r296_col = tl.arange($(192))
    r296_lane = tl.arange($(8))
    for r296_group in range($(8)) {
      r296_left = tl.load($(candidate_vmul_63_ptr) + (((((((r296_group % $(8)) * $(32)) + (r296_row[:, None] * $(8)) + r296_lane[None, :]) / $(32)) % $(8)) * $(32)) + ((((((r296_group % $(8)) * $(32)) + (r296_row[:, None] * $(8)) + r296_lane[None, :]) / $(8)) % $(4)) * $(8)) + ((((r296_group % $(8)) * $(32)) + (r296_row[:, None] * $(8)) + r296_lane[None, :]) % $(8))), dtype=tl.float32)
      r296_right = tl.load($(candidate_vto_copy_140_ptr) + (((((((r296_group % $(8)) * $(1536)) + (r296_lane[:, None] * $(192)) + r296_col[None, :]) / $(1536)) % $(8)) * $(1536)) + ((((((r296_group % $(8)) * $(1536)) + (r296_lane[:, None] * $(192)) + r296_col[None, :]) / $(192)) % $(8)) * $(192)) + ((((r296_group % $(8)) * $(1536)) + (r296_lane[:, None] * $(192)) + r296_col[None, :]) % $(192))), dtype=tl.float32)
      r296_value = tl.dot(r296_left, r296_right)
      r296_value = r296_value.to(tl.float32)
      tl.store($(candidate_vbmm_64_ptr) + ((r296_group * $(768)) + (r296_row[:, None] * $(192)) + r296_col[None, :]), r296_value)
    }
    r297_z = tl.arange($(6144))
    r297_value = (tl.load($(candidate_vbmm_64_ptr) + ((((r297_z / $(768)) % $(8)) * $(768)) + (((r297_z / $(192)) % $(4)) * $(192)) + (r297_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r297_value = r297_value.to(tl.float32)
    r297_value = tl.broadcast_to(r297_value, [$(6144)])
    tl.store($(candidate_vmul_64_ptr) + r297_z, r297_value)
    r298_z = tl.arange($(6144))
    r298_value = tl.load($(candidate_vmul_64_ptr) + ((((r298_z / $(192)) % $(8)) * $(768)) + (((r298_z / $(1536)) % $(4)) * $(192)) + (r298_z % $(192))), dtype=tl.float32)
    r298_value = r298_value.to(tl.bfloat16)
    r298_value = tl.broadcast_to(r298_value, [$(6144)])
    tl.store($(candidate_vto_copy_144_ptr) + r298_z, r298_value)
    r299_row = tl.arange($(8))
    r299_col = tl.arange($(192))
    r299_lane = tl.arange($(4))
    for r299_group in range($(8)) {
      r299_left = tl.load($(candidate_vmul_63_ptr) + (((((((r299_group % $(8)) * $(32)) + (r299_row[:, None] * $(4)) + r299_lane[None, :]) / $(32)) % $(8)) * $(32)) + (((((r299_group % $(8)) * $(32)) + (r299_row[:, None] * $(4)) + r299_lane[None, :]) % $(4)) * $(8)) + (((((r299_group % $(8)) * $(32)) + (r299_row[:, None] * $(4)) + r299_lane[None, :]) / $(4)) % $(8))), dtype=tl.float32)
      r299_right = tl.load($(candidate_vto_copy_139_ptr) + (((((((r299_group % $(8)) * $(768)) + (r299_lane[:, None] * $(192)) + r299_col[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r299_group % $(8)) * $(768)) + (r299_lane[:, None] * $(192)) + r299_col[None, :]) / $(192)) % $(4)) * $(192)) + ((((r299_group % $(8)) * $(768)) + (r299_lane[:, None] * $(192)) + r299_col[None, :]) % $(192))), dtype=tl.float32)
      r299_value = tl.dot(r299_left, r299_right)
      r299_value = r299_value.to(tl.float32)
      tl.store($(candidate_vbmm_65_ptr) + ((r299_group * $(1536)) + (r299_row[:, None] * $(192)) + r299_col[None, :]), r299_value)
    }
    r300_z = tl.arange($(12288))
    r300_value = (tl.load($(candidate_vbmm_65_ptr) + ((((r300_z / $(1536)) % $(8)) * $(1536)) + (((r300_z / $(192)) % $(8)) * $(192)) + (r300_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r300_value = r300_value.to(tl.float32)
    r300_value = tl.broadcast_to(r300_value, [$(12288)])
    tl.store($(candidate_vmul_65_ptr) + r300_z, r300_value)
    r301_z = tl.arange($(12288))
    r301_value = tl.load($(candidate_vmul_65_ptr) + ((((r301_z / $(192)) % $(8)) * $(1536)) + (((r301_z / $(1536)) % $(8)) * $(192)) + (r301_z % $(192))), dtype=tl.float32)
    r301_value = r301_value.to(tl.bfloat16)
    r301_value = tl.broadcast_to(r301_value, [$(12288)])
    tl.store($(candidate_vto_copy_145_ptr) + r301_z, r301_value)
    r302_row = tl.arange($(8))
    r302_col = tl.arange($(128))
    r302_lane = tl.arange($(4))
    for r302_group in range($(8)) {
      r302_left = tl.load($(candidate_vexp_4_ptr) + (((((((r302_group % $(8)) * $(32)) + (r302_row[:, None] * $(4)) + r302_lane[None, :]) / $(32)) % $(8)) * $(32)) + (((((r302_group % $(8)) * $(32)) + (r302_row[:, None] * $(4)) + r302_lane[None, :]) % $(4)) * $(8)) + (((((r302_group % $(8)) * $(32)) + (r302_row[:, None] * $(4)) + r302_lane[None, :]) / $(4)) % $(8))), dtype=tl.float32)
      r302_right = tl.load($(candidate_vto_copy_142_ptr) + (((((((r302_group % $(8)) * $(512)) + (r302_lane[:, None] * $(128)) + r302_col[None, :]) / $(512)) % $(8)) * $(512)) + ((((((r302_group % $(8)) * $(512)) + (r302_lane[:, None] * $(128)) + r302_col[None, :]) / $(128)) % $(4)) * $(128)) + ((((r302_group % $(8)) * $(512)) + (r302_lane[:, None] * $(128)) + r302_col[None, :]) % $(128))), dtype=tl.float32)
      r302_value = tl.dot(r302_left, r302_right)
      r302_value = r302_value.to(tl.float32)
      tl.store($(candidate_vbmm_66_ptr) + ((r302_group * $(1024)) + (r302_row[:, None] * $(128)) + r302_col[None, :]), r302_value)
    }
    r303_z = tl.arange($(8192))
    r303_value = tl.load($(candidate_vbmm_66_ptr) + ((((r303_z / $(128)) % $(8)) * $(1024)) + (((r303_z / $(1024)) % $(8)) * $(128)) + (r303_z % $(128))), dtype=tl.float32)
    r303_value = r303_value.to(tl.bfloat16)
    r303_value = tl.broadcast_to(r303_value, [$(8192)])
    tl.store($(candidate_vto_copy_146_ptr) + r303_z, r303_value)
    r304_z = tl.arange($(24576))
    r304_value = tl.where(((r304_z / $(1536)) % $(16)) < $(8), tl.load($(candidate_vto_copy_145_ptr) + ((((((r304_z / $(1536)) % $(16)) - $(0)) % $(8)) * $(1536)) + (((r304_z / $(192)) % $(8)) * $(192)) + (r304_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r304_value = r304_value.to(tl.bfloat16)
    r304_value = tl.broadcast_to(r304_value, [$(24576)])
    tl.store($(candidate_vcat_31_ptr) + r304_z, r304_value)
    r305_z = tl.arange($(16384))
    r305_value = tl.where(((r305_z / $(1024)) % $(16)) < $(8), tl.load($(candidate_vto_copy_146_ptr) + ((((((r305_z / $(1024)) % $(16)) - $(0)) % $(8)) * $(1024)) + (((r305_z / $(128)) % $(8)) * $(128)) + (r305_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r305_value = r305_value.to(tl.bfloat16)
    r305_value = tl.broadcast_to(r305_value, [$(16384)])
    tl.store($(candidate_vcat_32_ptr) + r305_z, r305_value)
    r306_z = tl.arange($(24576))
    r306_value = tl.load($(candidate_vcat_25_ptr) + ((((r306_z / $(192)) % $(16)) * $(1536)) + (((r306_z / $(3072)) % $(8)) * $(192)) + (r306_z % $(192)) + $(24576)), dtype=tl.bfloat16)
    r306_value = r306_value.to(tl.float32)
    r306_value = tl.broadcast_to(r306_value, [$(24576)])
    tl.store($(candidate_vto_copy_147_ptr) + r306_z, r306_value)
    r307_z = tl.arange($(49152))
    r307_value = tl.load($(candidate_vcat_26_ptr) + ((((r307_z / $(192)) % $(32)) * $(1536)) + (((r307_z / $(6144)) % $(8)) * $(192)) + (r307_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r307_value = r307_value.to(tl.float32)
    r307_value = tl.broadcast_to(r307_value, [$(49152)])
    tl.store($(candidate_vto_copy_148_ptr) + r307_z, r307_value)
    r308_z = tl.arange($(32768))
    r308_value = tl.load($(candidate_vbmm_50_ptr) + ((((r308_z / $(128)) % $(32)) * $(1024)) + (((r308_z / $(4096)) % $(8)) * $(128)) + (r308_z % $(128)) + $(65536)), dtype=tl.bfloat16)
    r308_value = r308_value.to(tl.float32)
    r308_value = tl.broadcast_to(r308_value, [$(32768)])
    tl.store($(candidate_vto_copy_149_ptr) + r308_z, r308_value)
    r309_z = tl.arange($(16384))
    r309_value = tl.load($(candidate_vmm_28_ptr) + ((((r309_z / $(128)) % $(16)) * $(1024)) + (((r309_z / $(2048)) % $(8)) * $(128)) + (r309_z % $(128)) + $(81920)), dtype=tl.bfloat16)
    r309_value = r309_value.to(tl.float32)
    r309_value = tl.broadcast_to(r309_value, [$(16384)])
    tl.store($(candidate_vto_copy_150_ptr) + r309_z, r309_value)
    r310_z = tl.arange($(16384))
    r310_value = tl.load($(candidate_vcat_27_ptr) + ((((r310_z / $(128)) % $(16)) * $(1024)) + (((r310_z / $(2048)) % $(8)) * $(128)) + (r310_z % $(128)) + $(16384)), dtype=tl.bfloat16)
    r310_value = r310_value.to(tl.float32)
    r310_value = tl.broadcast_to(r310_value, [$(16384)])
    tl.store($(candidate_vto_copy_151_ptr) + r310_z, r310_value)
    r311_row = tl.arange($(16))
    r311_col = tl.arange($(32))
    r311_lane = tl.arange($(192))
    for r311_group in range($(8)) {
      r311_left = tl.load($(candidate_vto_copy_147_ptr) + (((((((r311_group % $(8)) * $(3072)) + (r311_row[:, None] * $(192)) + r311_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r311_group % $(8)) * $(3072)) + (r311_row[:, None] * $(192)) + r311_lane[None, :]) / $(192)) % $(16)) * $(192)) + ((((r311_group % $(8)) * $(3072)) + (r311_row[:, None] * $(192)) + r311_lane[None, :]) % $(192))), dtype=tl.float32)
      r311_right = tl.load($(candidate_vto_copy_148_ptr) + (((((((r311_group % $(8)) * $(6144)) + (r311_lane[:, None] * $(32)) + r311_col[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r311_group % $(8)) * $(6144)) + (r311_lane[:, None] * $(32)) + r311_col[None, :]) % $(32)) * $(192)) + (((((r311_group % $(8)) * $(6144)) + (r311_lane[:, None] * $(32)) + r311_col[None, :]) / $(32)) % $(192))), dtype=tl.float32)
      r311_value = tl.dot(r311_left, r311_right)
      r311_value = r311_value.to(tl.float32)
      tl.store($(candidate_vbmm_67_ptr) + ((r311_group * $(512)) + (r311_row[:, None] * $(32)) + r311_col[None, :]), r311_value)
    }
    r312_z = tl.arange($(4096))
    r312_value = (tl.load($(candidate_vbmm_67_ptr) + ((((r312_z / $(512)) % $(8)) * $(512)) + (((r312_z / $(32)) % $(16)) * $(32)) + (r312_z % $(32))), dtype=tl.float32) * 0.07216878364870322)
    r312_value = r312_value.to(tl.float32)
    r312_value = tl.broadcast_to(r312_value, [$(4096)])
    tl.store($(candidate_vmul_66_ptr) + r312_z, r312_value)
    r313_z = tl.arange($(16))
    r313_value = (tl.cast(r313_z, tl.int64) + tl.full([], $(32), dtype=tl.int64))
    r313_value = tl.broadcast_to(r313_value, [$(16)])
    tl.store($(candidate_vadd_44_ptr) + r313_z, r313_value)
    r314_z = tl.arange($(16))
    r314_value = (tl.load($(candidate_vadd_44_ptr) + r314_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r314_value = tl.broadcast_to(r314_value, [$(16)])
    tl.store($(candidate_vsub_42_ptr) + r314_z, r314_value)
    r315_z = tl.arange($(512))
    r315_value = (tl.cast((r315_z % $(32)), tl.int64) > tl.load($(candidate_vsub_42_ptr) + ((r315_z / $(32)) % $(16)), dtype=tl.int64))
    r315_value = tl.broadcast_to(r315_value, [$(512)])
    tl.store($(candidate_vgt_23_ptr) + r315_z, r315_value)
    r316_z = tl.arange($(4096))
    r316_value = tl.where(tl.load($(candidate_vgt_23_ptr) + ((((r316_z / $(32)) % $(16)) * $(32)) + (r316_z % $(32))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_66_ptr) + ((((r316_z / $(512)) % $(8)) * $(512)) + (((r316_z / $(32)) % $(16)) * $(32)) + (r316_z % $(32))), dtype=tl.float32))
    r316_value = r316_value.to(tl.float32)
    r316_value = tl.broadcast_to(r316_value, [$(4096)])
    tl.store($(candidate_vmasked_fill_23_ptr) + r316_z, r316_value)
    r317_z = tl.arange($(4096))
    r317_value = (tl.load($(candidate_vmasked_fill_23_ptr) + ((((r317_z / $(512)) % $(8)) * $(512)) + (((r317_z / $(32)) % $(16)) * $(32)) + (r317_z % $(32))), dtype=tl.float32) - tl.load($(candidate_vcat_28_ptr) + ((((r317_z / $(32)) % $(16)) * $(8)) + ((r317_z / $(512)) % $(8)) + $(128)), dtype=tl.float32))
    r317_value = r317_value.to(tl.float32)
    r317_value = tl.broadcast_to(r317_value, [$(4096)])
    tl.store($(candidate_vsub_43_ptr) + r317_z, r317_value)
    r318_z = tl.arange($(4096))
    r318_value = tl.exp(tl.load($(candidate_vsub_43_ptr) + ((((r318_z / $(512)) % $(8)) * $(512)) + (((r318_z / $(32)) % $(16)) * $(32)) + (r318_z % $(32))), dtype=tl.float32))
    r318_value = r318_value.to(tl.float32)
    r318_value = tl.broadcast_to(r318_value, [$(4096)])
    tl.store($(candidate_vexp_5_ptr) + r318_z, r318_value)
    r319_z = tl.arange($(16384))
    r319_value = (tl.load($(candidate_vto_copy_150_ptr) + ((((r319_z / $(2048)) % $(8)) * $(2048)) + (((r319_z / $(128)) % $(16)) * $(128)) + (r319_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_151_ptr) + ((((r319_z / $(2048)) % $(8)) * $(2048)) + (((r319_z / $(128)) % $(16)) * $(128)) + (r319_z % $(128))), dtype=tl.float32))
    r319_value = r319_value.to(tl.float32)
    r319_value = tl.broadcast_to(r319_value, [$(16384)])
    tl.store($(candidate_vmul_67_ptr) + r319_z, r319_value)
    r320_z = tl.arange($(128))
    r320_lane = tl.arange($(128))
    r320_value = tl.sum(tl.load($(candidate_vmul_67_ptr) + ((((r320_z[:, None] / $(16)) % $(8)) * $(2048)) + ((r320_z[:, None] % $(16)) * $(128)) + r320_lane[None, :]), dtype=tl.float32), axis=1)
    r320_value = r320_value.to(tl.float32)
    r320_value = tl.broadcast_to(r320_value, [$(128)])
    tl.store($(candidate_vsum_7_ptr) + r320_z, r320_value)
    r321_row = tl.arange($(16))
    r321_col = tl.arange($(32))
    r321_lane = tl.arange($(128))
    for r321_group in range($(8)) {
      r321_left = tl.load($(candidate_vto_copy_150_ptr) + (((((((r321_group % $(8)) * $(2048)) + (r321_row[:, None] * $(128)) + r321_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r321_group % $(8)) * $(2048)) + (r321_row[:, None] * $(128)) + r321_lane[None, :]) / $(128)) % $(16)) * $(128)) + ((((r321_group % $(8)) * $(2048)) + (r321_row[:, None] * $(128)) + r321_lane[None, :]) % $(128))), dtype=tl.float32)
      r321_right = tl.load($(candidate_vto_copy_149_ptr) + (((((((r321_group % $(8)) * $(4096)) + (r321_lane[:, None] * $(32)) + r321_col[None, :]) / $(4096)) % $(8)) * $(4096)) + (((((r321_group % $(8)) * $(4096)) + (r321_lane[:, None] * $(32)) + r321_col[None, :]) % $(32)) * $(128)) + (((((r321_group % $(8)) * $(4096)) + (r321_lane[:, None] * $(32)) + r321_col[None, :]) / $(32)) % $(128))), dtype=tl.float32)
      r321_value = tl.dot(r321_left, r321_right)
      r321_value = r321_value.to(tl.float32)
      tl.store($(candidate_vbmm_68_ptr) + ((r321_group * $(512)) + (r321_row[:, None] * $(32)) + r321_col[None, :]), r321_value)
    }
    r322_z = tl.arange($(4096))
    r322_value = (tl.load($(candidate_vbmm_68_ptr) + ((((r322_z / $(512)) % $(8)) * $(512)) + (((r322_z / $(32)) % $(16)) * $(32)) + (r322_z % $(32))), dtype=tl.float32) - tl.load($(candidate_vsum_7_ptr) + ((((r322_z / $(512)) % $(8)) * $(16)) + ((r322_z / $(32)) % $(16))), dtype=tl.float32))
    r322_value = r322_value.to(tl.float32)
    r322_value = tl.broadcast_to(r322_value, [$(4096)])
    tl.store($(candidate_vsub_44_ptr) + r322_z, r322_value)
    r323_z = tl.arange($(4096))
    r323_value = (tl.load($(candidate_vexp_5_ptr) + ((((r323_z / $(512)) % $(8)) * $(512)) + (((r323_z / $(32)) % $(16)) * $(32)) + (r323_z % $(32))), dtype=tl.float32) * tl.load($(candidate_vsub_44_ptr) + ((((r323_z / $(512)) % $(8)) * $(512)) + (((r323_z / $(32)) % $(16)) * $(32)) + (r323_z % $(32))), dtype=tl.float32))
    r323_value = r323_value.to(tl.float32)
    r323_value = tl.broadcast_to(r323_value, [$(4096)])
    tl.store($(candidate_vmul_68_ptr) + r323_z, r323_value)
    r324_row = tl.arange($(16))
    r324_col = tl.arange($(192))
    r324_lane = tl.arange($(32))
    for r324_group in range($(8)) {
      r324_left = tl.load($(candidate_vmul_68_ptr) + (((((((r324_group % $(8)) * $(512)) + (r324_row[:, None] * $(32)) + r324_lane[None, :]) / $(512)) % $(8)) * $(512)) + ((((((r324_group % $(8)) * $(512)) + (r324_row[:, None] * $(32)) + r324_lane[None, :]) / $(32)) % $(16)) * $(32)) + ((((r324_group % $(8)) * $(512)) + (r324_row[:, None] * $(32)) + r324_lane[None, :]) % $(32))), dtype=tl.float32)
      r324_right = tl.load($(candidate_vto_copy_148_ptr) + (((((((r324_group % $(8)) * $(6144)) + (r324_lane[:, None] * $(192)) + r324_col[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r324_group % $(8)) * $(6144)) + (r324_lane[:, None] * $(192)) + r324_col[None, :]) / $(192)) % $(32)) * $(192)) + ((((r324_group % $(8)) * $(6144)) + (r324_lane[:, None] * $(192)) + r324_col[None, :]) % $(192))), dtype=tl.float32)
      r324_value = tl.dot(r324_left, r324_right)
      r324_value = r324_value.to(tl.float32)
      tl.store($(candidate_vbmm_69_ptr) + ((r324_group * $(3072)) + (r324_row[:, None] * $(192)) + r324_col[None, :]), r324_value)
    }
    r325_z = tl.arange($(24576))
    r325_value = (tl.load($(candidate_vbmm_69_ptr) + ((((r325_z / $(3072)) % $(8)) * $(3072)) + (((r325_z / $(192)) % $(16)) * $(192)) + (r325_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r325_value = r325_value.to(tl.float32)
    r325_value = tl.broadcast_to(r325_value, [$(24576)])
    tl.store($(candidate_vmul_69_ptr) + r325_z, r325_value)
    r326_z = tl.arange($(24576))
    r326_value = tl.load($(candidate_vmul_69_ptr) + ((((r326_z / $(192)) % $(8)) * $(3072)) + (((r326_z / $(1536)) % $(16)) * $(192)) + (r326_z % $(192))), dtype=tl.float32)
    r326_value = r326_value.to(tl.bfloat16)
    r326_value = tl.broadcast_to(r326_value, [$(24576)])
    tl.store($(candidate_vto_copy_152_ptr) + r326_z, r326_value)
    r327_row = tl.arange($(32))
    r327_col = tl.arange($(192))
    r327_lane = tl.arange($(16))
    for r327_group in range($(8)) {
      r327_left = tl.load($(candidate_vmul_68_ptr) + (((((((r327_group % $(8)) * $(512)) + (r327_row[:, None] * $(16)) + r327_lane[None, :]) / $(512)) % $(8)) * $(512)) + (((((r327_group % $(8)) * $(512)) + (r327_row[:, None] * $(16)) + r327_lane[None, :]) % $(16)) * $(32)) + (((((r327_group % $(8)) * $(512)) + (r327_row[:, None] * $(16)) + r327_lane[None, :]) / $(16)) % $(32))), dtype=tl.float32)
      r327_right = tl.load($(candidate_vto_copy_147_ptr) + (((((((r327_group % $(8)) * $(3072)) + (r327_lane[:, None] * $(192)) + r327_col[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r327_group % $(8)) * $(3072)) + (r327_lane[:, None] * $(192)) + r327_col[None, :]) / $(192)) % $(16)) * $(192)) + ((((r327_group % $(8)) * $(3072)) + (r327_lane[:, None] * $(192)) + r327_col[None, :]) % $(192))), dtype=tl.float32)
      r327_value = tl.dot(r327_left, r327_right)
      r327_value = r327_value.to(tl.float32)
      tl.store($(candidate_vbmm_70_ptr) + ((r327_group * $(6144)) + (r327_row[:, None] * $(192)) + r327_col[None, :]), r327_value)
    }
    r328_z = tl.arange($(49152))
    r328_value = (tl.load($(candidate_vbmm_70_ptr) + ((((r328_z / $(6144)) % $(8)) * $(6144)) + (((r328_z / $(192)) % $(32)) * $(192)) + (r328_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r328_value = r328_value.to(tl.float32)
    r328_value = tl.broadcast_to(r328_value, [$(49152)])
    tl.store($(candidate_vmul_70_ptr) + r328_z, r328_value)
    r329_z = tl.arange($(49152))
    r329_value = tl.load($(candidate_vmul_70_ptr) + ((((r329_z / $(192)) % $(8)) * $(6144)) + (((r329_z / $(1536)) % $(32)) * $(192)) + (r329_z % $(192))), dtype=tl.float32)
    r329_value = r329_value.to(tl.bfloat16)
    r329_value = tl.broadcast_to(r329_value, [$(49152)])
    tl.store($(candidate_vto_copy_153_ptr) + r329_z, r329_value)
    r330_row = tl.arange($(32))
    r330_col = tl.arange($(128))
    r330_lane = tl.arange($(16))
    for r330_group in range($(8)) {
      r330_left = tl.load($(candidate_vexp_5_ptr) + (((((((r330_group % $(8)) * $(512)) + (r330_row[:, None] * $(16)) + r330_lane[None, :]) / $(512)) % $(8)) * $(512)) + (((((r330_group % $(8)) * $(512)) + (r330_row[:, None] * $(16)) + r330_lane[None, :]) % $(16)) * $(32)) + (((((r330_group % $(8)) * $(512)) + (r330_row[:, None] * $(16)) + r330_lane[None, :]) / $(16)) % $(32))), dtype=tl.float32)
      r330_right = tl.load($(candidate_vto_copy_150_ptr) + (((((((r330_group % $(8)) * $(2048)) + (r330_lane[:, None] * $(128)) + r330_col[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r330_group % $(8)) * $(2048)) + (r330_lane[:, None] * $(128)) + r330_col[None, :]) / $(128)) % $(16)) * $(128)) + ((((r330_group % $(8)) * $(2048)) + (r330_lane[:, None] * $(128)) + r330_col[None, :]) % $(128))), dtype=tl.float32)
      r330_value = tl.dot(r330_left, r330_right)
      r330_value = r330_value.to(tl.float32)
      tl.store($(candidate_vbmm_71_ptr) + ((r330_group * $(4096)) + (r330_row[:, None] * $(128)) + r330_col[None, :]), r330_value)
    }
    r331_z = tl.arange($(32768))
    r331_value = tl.load($(candidate_vbmm_71_ptr) + ((((r331_z / $(128)) % $(8)) * $(4096)) + (((r331_z / $(1024)) % $(32)) * $(128)) + (r331_z % $(128))), dtype=tl.float32)
    r331_value = r331_value.to(tl.bfloat16)
    r331_value = tl.broadcast_to(r331_value, [$(32768)])
    tl.store($(candidate_vto_copy_154_ptr) + r331_z, r331_value)
    r332_z = tl.arange($(98304))
    r332_value = tl.where(((r332_z / $(1536)) % $(64)) < $(32), tl.load($(candidate_vto_copy_153_ptr) + ((((((r332_z / $(1536)) % $(64)) - $(0)) % $(32)) * $(1536)) + (((r332_z / $(192)) % $(8)) * $(192)) + (r332_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r332_value = r332_value.to(tl.bfloat16)
    r332_value = tl.broadcast_to(r332_value, [$(98304)])
    tl.store($(candidate_vcat_33_ptr) + r332_z, r332_value)
    r333_z = tl.arange($(65536))
    r333_value = tl.where(((r333_z / $(1024)) % $(64)) < $(32), tl.load($(candidate_vto_copy_154_ptr) + ((((((r333_z / $(1024)) % $(64)) - $(0)) % $(32)) * $(1024)) + (((r333_z / $(128)) % $(8)) * $(128)) + (r333_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r333_value = r333_value.to(tl.bfloat16)
    r333_value = tl.broadcast_to(r333_value, [$(65536)])
    tl.store($(candidate_vcat_34_ptr) + r333_z, r333_value)
    r334_z = tl.arange($(49152))
    r334_value = tl.where(((r334_z / $(1536)) % $(32)) < $(12), tl.load($(candidate_vto_copy_136_ptr) + ((((((r334_z / $(1536)) % $(32)) - $(0)) % $(12)) * $(1536)) + (((r334_z / $(192)) % $(8)) * $(192)) + (r334_z % $(192))), dtype=tl.bfloat16), tl.where(((r334_z / $(1536)) % $(32)) < $(16), tl.load($(candidate_vto_copy_144_ptr) + ((((((r334_z / $(1536)) % $(32)) - $(12)) % $(4)) * $(1536)) + (((r334_z / $(192)) % $(8)) * $(192)) + (r334_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vto_copy_152_ptr) + ((((((r334_z / $(1536)) % $(32)) - $(16)) % $(16)) * $(1536)) + (((r334_z / $(192)) % $(8)) * $(192)) + (r334_z % $(192))), dtype=tl.bfloat16)))
    r334_value = r334_value.to(tl.bfloat16)
    r334_value = tl.broadcast_to(r334_value, [$(49152)])
    tl.store($(candidate_vcat_35_ptr) + r334_z, r334_value)
    r335_z = tl.arange($(196608))
    r335_value = tl.where(((r335_z / $(1536)) % $(128)) < $(48), tl.load($(candidate_vcat_29_ptr) + ((((((r335_z / $(1536)) % $(128)) - $(0)) % $(48)) * $(1536)) + (((r335_z / $(192)) % $(8)) * $(192)) + (r335_z % $(192))), dtype=tl.bfloat16), tl.where(((r335_z / $(1536)) % $(128)) < $(64), tl.load($(candidate_vcat_31_ptr) + ((((((r335_z / $(1536)) % $(128)) - $(48)) % $(16)) * $(1536)) + (((r335_z / $(192)) % $(8)) * $(192)) + (r335_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vcat_33_ptr) + ((((((r335_z / $(1536)) % $(128)) - $(64)) % $(64)) * $(1536)) + (((r335_z / $(192)) % $(8)) * $(192)) + (r335_z % $(192))), dtype=tl.bfloat16)))
    r335_value = r335_value.to(tl.bfloat16)
    r335_value = tl.broadcast_to(r335_value, [$(196608)])
    tl.store($(candidate_vcat_36_ptr) + r335_z, r335_value)
    r336_z = tl.arange($(131072))
    r336_value = tl.where(((r336_z / $(1024)) % $(128)) < $(48), tl.load($(candidate_vcat_30_ptr) + ((((((r336_z / $(1024)) % $(128)) - $(0)) % $(48)) * $(1024)) + (((r336_z / $(128)) % $(8)) * $(128)) + (r336_z % $(128))), dtype=tl.bfloat16), tl.where(((r336_z / $(1024)) % $(128)) < $(64), tl.load($(candidate_vcat_32_ptr) + ((((((r336_z / $(1024)) % $(128)) - $(48)) % $(16)) * $(1024)) + (((r336_z / $(128)) % $(8)) * $(128)) + (r336_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_34_ptr) + ((((((r336_z / $(1024)) % $(128)) - $(64)) % $(64)) * $(1024)) + (((r336_z / $(128)) % $(8)) * $(128)) + (r336_z % $(128))), dtype=tl.bfloat16)))
    r336_value = r336_value.to(tl.bfloat16)
    r336_value = tl.broadcast_to(r336_value, [$(131072)])
    tl.store($(candidate_vcat_37_ptr) + r336_z, r336_value)
    r337_z = tl.arange($(8192))
    r337_lane = tl.arange($(8))
    r337_value = tl.sum(tl.load($(candidate_vcat_36_ptr) + ((((r337_z[:, None] / $(64)) % $(128)) * $(1536)) + (r337_lane[None, :] * $(192)) + (r337_z[:, None] % $(64)) + $(128)), dtype=tl.bfloat16), axis=1)
    r337_value = r337_value.to(tl.bfloat16)
    r337_value = tl.broadcast_to(r337_value, [$(8192)])
    tl.store($(candidate_vsum_8_ptr) + r337_z, r337_value)
    r338_row = tl.arange($(128))
    r338_col = tl.arange($(512))
    r338_lane = tl.arange($(1024))
    for r338_group in range($(1)) {
      r338_left = tl.load($(candidate_vcat_36_ptr) + ((((((r338_row[:, None] * $(1024)) + r338_lane[None, :]) / $(1024)) % $(128)) * $(1536)) + ((((r338_row[:, None] * $(1024)) + r338_lane[None, :]) % $(8)) * $(192)) + ((((r338_row[:, None] * $(1024)) + r338_lane[None, :]) / $(8)) % $(128))), dtype=tl.bfloat16)
      r338_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r338_lane[:, None] * $(512)) + r338_col[None, :]) / $(512)) % $(8)) * $(131072)) + (((((r338_lane[:, None] * $(512)) + r338_col[None, :]) / $(4096)) % $(128)) * $(512)) + (((r338_lane[:, None] * $(512)) + r338_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r338_value = tl.dot(r338_left, r338_right)
      r338_value = r338_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_72_ptr) + ((r338_group * $(65536)) + (r338_row[:, None] * $(512)) + r338_col[None, :]), r338_value)
    }
    r339_row = tl.arange($(128))
    r339_col = tl.arange($(512))
    r339_lane = tl.arange($(1024))
    for r339_group in range($(1)) {
      r339_left = tl.load($(candidate_vcat_37_ptr) + ((((((r339_row[:, None] * $(1024)) + r339_lane[None, :]) / $(1024)) % $(128)) * $(1024)) + ((((r339_row[:, None] * $(1024)) + r339_lane[None, :]) % $(8)) * $(128)) + ((((r339_row[:, None] * $(1024)) + r339_lane[None, :]) / $(8)) % $(128))), dtype=tl.bfloat16)
      r339_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r339_lane[:, None] * $(512)) + r339_col[None, :]) / $(512)) % $(8)) * $(131072)) + (((((r339_lane[:, None] * $(512)) + r339_col[None, :]) / $(4096)) % $(128)) * $(512)) + (((r339_lane[:, None] * $(512)) + r339_col[None, :]) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r339_value = tl.dot(r339_left, r339_right)
      r339_value = r339_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_73_ptr) + ((r339_group * $(65536)) + (r339_row[:, None] * $(512)) + r339_col[None, :]), r339_value)
    }
    r340_z = tl.arange($(65536))
    r340_value = (tl.load($(candidate_vbmm_72_ptr) + ((((r340_z / $(512)) % $(128)) * $(512)) + (r340_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vbmm_73_ptr) + ((((r340_z / $(512)) % $(128)) * $(512)) + (r340_z % $(512))), dtype=tl.bfloat16))
    r340_value = r340_value.to(tl.bfloat16)
    r340_value = tl.broadcast_to(r340_value, [$(65536)])
    tl.store($(candidate_vadd_45_ptr) + r340_z, r340_value)
    r341_z = tl.arange($(65536))
    r341_value = tl.where(((((r341_z / $(512)) % $(128)) >= $(0)) & (((r341_z / $(512)) % $(128)) < $(128)) & ((((r341_z / $(512)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vadd_45_ptr) + ((((((r341_z / $(512)) % $(128)) - $(0)) % $(128)) * $(512)) + (r341_z % $(512))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r341_value = r341_value.to(tl.bfloat16)
    r341_value = tl.broadcast_to(r341_value, [$(65536)])
    tl.store($(candidate_vslice_backward_4_ptr) + r341_z, r341_value)
    r342_z = tl.arange($(65536))
    r342_value = (tl.load($(candidate_vslice_backward_ptr) + ((((r342_z / $(512)) % $(128)) * $(512)) + (r342_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_4_ptr) + ((((r342_z / $(512)) % $(128)) * $(512)) + (r342_z % $(512))), dtype=tl.bfloat16))
    r342_value = r342_value.to(tl.bfloat16)
    r342_value = tl.broadcast_to(r342_value, [$(65536)])
    tl.store($(candidate_vadd_46_ptr) + r342_z, r342_value)
    r343_z = tl.arange($(8192))
    r343_value = tl.where(((((r343_z / $(64)) % $(128)) >= $(0)) & (((r343_z / $(64)) % $(128)) < $(128)) & ((((r343_z / $(64)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vsum_8_ptr) + ((((((r343_z / $(64)) % $(128)) - $(0)) % $(128)) * $(64)) + (r343_z % $(64))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r343_value = r343_value.to(tl.bfloat16)
    r343_value = tl.broadcast_to(r343_value, [$(8192)])
    tl.store($(candidate_vslice_backward_5_ptr) + r343_z, r343_value)
    r344_z = tl.arange($(8192))
    r344_value = (tl.load($(candidate_vslice_backward_1_ptr) + ((((r344_z / $(64)) % $(128)) * $(64)) + (r344_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_5_ptr) + ((((r344_z / $(64)) % $(128)) * $(64)) + (r344_z % $(64))), dtype=tl.bfloat16))
    r344_value = r344_value.to(tl.bfloat16)
    r344_value = tl.broadcast_to(r344_value, [$(8192)])
    tl.store($(candidate_vadd_47_ptr) + r344_z, r344_value)
    r345_z = tl.arange($(131072))
    r345_value = tl.where(((((r345_z / $(1024)) % $(128)) >= $(64)) & (((r345_z / $(1024)) % $(128)) < $(96)) & ((((r345_z / $(1024)) % $(128)) - $(64)) % $(1) == $(0))), tl.load($(candidate_vcat_35_ptr) + ((((((r345_z / $(1024)) % $(128)) - $(64)) % $(32)) * $(1536)) + (((r345_z / $(128)) % $(8)) * $(192)) + (r345_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r345_value = r345_value.to(tl.bfloat16)
    r345_value = tl.broadcast_to(r345_value, [$(131072)])
    tl.store($(candidate_vslice_backward_6_ptr) + r345_z, r345_value)
    r346_z = tl.arange($(131072))
    r346_value = (tl.load($(candidate_vslice_backward_2_ptr) + ((((r346_z / $(1024)) % $(128)) * $(1024)) + (((r346_z / $(128)) % $(8)) * $(128)) + (r346_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_6_ptr) + ((((r346_z / $(1024)) % $(128)) * $(1024)) + (((r346_z / $(128)) % $(8)) * $(128)) + (r346_z % $(128))), dtype=tl.bfloat16))
    r346_value = r346_value.to(tl.bfloat16)
    r346_value = tl.broadcast_to(r346_value, [$(131072)])
    tl.store($(candidate_vadd_48_ptr) + r346_z, r346_value)
    r347_z = tl.arange($(65536))
    r347_value = tl.where(((((r347_z / $(512)) % $(128)) >= $(64)) & (((r347_z / $(512)) % $(128)) < $(96)) & ((((r347_z / $(512)) % $(128)) - $(64)) % $(1) == $(0))), tl.load($(candidate_vcat_35_ptr) + ((((((r347_z / $(512)) % $(128)) - $(64)) % $(32)) * $(1536)) + (((r347_z / $(64)) % $(8)) * $(192)) + (r347_z % $(64)) + $(128)), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r347_value = r347_value.to(tl.bfloat16)
    r347_value = tl.broadcast_to(r347_value, [$(65536)])
    tl.store($(candidate_vslice_backward_7_ptr) + r347_z, r347_value)
    r348_z = tl.arange($(65536))
    r348_value = (tl.load($(candidate_vslice_backward_3_ptr) + ((((r348_z / $(512)) % $(128)) * $(512)) + (((r348_z / $(64)) % $(8)) * $(64)) + (r348_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_7_ptr) + ((((r348_z / $(512)) % $(128)) * $(512)) + (((r348_z / $(64)) % $(8)) * $(64)) + (r348_z % $(64))), dtype=tl.bfloat16))
    r348_value = r348_value.to(tl.bfloat16)
    r348_value = tl.broadcast_to(r348_value, [$(65536)])
    tl.store($(candidate_vadd_49_ptr) + r348_z, r348_value)
    r349_row = tl.arange($(128))
    r349_col = tl.arange($(1024))
    r349_lane = tl.arange($(512))
    for r349_group in range($(1)) {
      r349_left = tl.load($(candidate_vindex_2_ptr) + ((((((r349_row[:, None] * $(512)) + r349_lane[None, :]) / $(512)) % $(128)) * $(576)) + (((r349_row[:, None] * $(512)) + r349_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r349_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r349_lane[:, None] * $(1024)) + r349_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r349_lane[:, None] * $(1024)) + r349_col[None, :]) % $(128)) * $(512)) + ((((r349_lane[:, None] * $(1024)) + r349_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r349_value = tl.dot(r349_left, r349_right)
      r349_value = r349_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_76_ptr) + ((r349_group * $(131072)) + (r349_row[:, None] * $(1024)) + r349_col[None, :]), r349_value)
    }
    r350_row = tl.arange($(128))
    r350_col = tl.arange($(1024))
    r350_lane = tl.arange($(512))
    for r350_group in range($(1)) {
      r350_left = tl.load($(candidate_vindex_2_ptr) + ((((((r350_row[:, None] * $(512)) + r350_lane[None, :]) / $(512)) % $(128)) * $(576)) + (((r350_row[:, None] * $(512)) + r350_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r350_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r350_lane[:, None] * $(1024)) + r350_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r350_lane[:, None] * $(1024)) + r350_col[None, :]) % $(128)) * $(512)) + ((((r350_lane[:, None] * $(1024)) + r350_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r350_value = tl.dot(r350_left, r350_right)
      r350_value = r350_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_77_ptr) + ((r350_group * $(131072)) + (r350_row[:, None] * $(1024)) + r350_col[None, :]), r350_value)
    }
    r351_z = tl.arange($(49152))
    r351_value = tl.where((r351_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r351_z / $(1536)) % $(32)) * $(1024)) + (((r351_z / $(192)) % $(8)) * $(128)) + ((r351_z % $(192)) - $(0)) + $(32768)) / $(1024)) % $(128)) * $(1536)) + (((((((r351_z / $(1536)) % $(32)) * $(1024)) + (((r351_z / $(192)) % $(8)) * $(128)) + ((r351_z % $(192)) - $(0)) + $(32768)) / $(128)) % $(8)) * $(192)) + (((((r351_z / $(1536)) % $(32)) * $(1024)) + (((r351_z / $(192)) % $(8)) * $(128)) + ((r351_z % $(192)) - $(0)) + $(32768)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vadd_6_ptr) + ((((((((r351_z / $(1536)) % $(32)) * $(512)) + (((r351_z / $(192)) % $(8)) * $(64)) + ((r351_z % $(192)) - $(128)) + $(16384)) / $(512)) % $(128)) * $(512)) + (((((((r351_z / $(1536)) % $(32)) * $(512)) + (((r351_z / $(192)) % $(8)) * $(64)) + ((r351_z % $(192)) - $(128)) + $(16384)) / $(64)) % $(8)) * $(64)) + (((((r351_z / $(1536)) % $(32)) * $(512)) + (((r351_z / $(192)) % $(8)) * $(64)) + ((r351_z % $(192)) - $(128)) + $(16384)) % $(64))), dtype=tl.bfloat16))
    r351_value = r351_value.to(tl.bfloat16)
    r351_value = tl.broadcast_to(r351_value, [$(49152)])
    tl.store($(candidate_vcat_39_ptr) + r351_z, r351_value)
    r352_z = tl.arange($(196608))
    r352_value = tl.where((r352_z % $(192)) < $(128), tl.load($(candidate_vbmm_76_ptr) + ((((((((r352_z / $(1536)) % $(128)) * $(1024)) + (((r352_z / $(192)) % $(8)) * $(128)) + ((r352_z % $(192)) - $(0))) / $(1024)) % $(128)) * $(1024)) + (((((((r352_z / $(1536)) % $(128)) * $(1024)) + (((r352_z / $(192)) % $(8)) * $(128)) + ((r352_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r352_z / $(1536)) % $(128)) * $(1024)) + (((r352_z / $(192)) % $(8)) * $(128)) + ((r352_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vindex_2_ptr) + ((((r352_z / $(1536)) % $(128)) * $(576)) + ((((r352_z % $(192)) - $(128)) + $(512)) % $(576))), dtype=tl.bfloat16))
    r352_value = r352_value.to(tl.bfloat16)
    r352_value = tl.broadcast_to(r352_value, [$(196608)])
    tl.store($(candidate_vcat_40_ptr) + r352_z, r352_value)
    r353_z = tl.arange($(18432))
    r353_value = tl.load($(candidate_vcat_39_ptr) + ((((r353_z / $(192)) % $(12)) * $(1536)) + (((r353_z / $(2304)) % $(8)) * $(192)) + (r353_z % $(192))), dtype=tl.bfloat16)
    r353_value = r353_value.to(tl.float32)
    r353_value = tl.broadcast_to(r353_value, [$(18432)])
    tl.store($(candidate_vto_copy_155_ptr) + r353_z, r353_value)
    r354_z = tl.arange($(73728))
    r354_value = tl.load($(candidate_vcat_40_ptr) + ((((r354_z / $(192)) % $(48)) * $(1536)) + (((r354_z / $(9216)) % $(8)) * $(192)) + (r354_z % $(192))), dtype=tl.bfloat16)
    r354_value = r354_value.to(tl.float32)
    r354_value = tl.broadcast_to(r354_value, [$(73728)])
    tl.store($(candidate_vto_copy_156_ptr) + r354_z, r354_value)
    r355_row = tl.arange($(12))
    r355_col = tl.arange($(48))
    r355_lane = tl.arange($(192))
    for r355_group in range($(8)) {
      r355_left = tl.load($(candidate_vto_copy_155_ptr) + (((((((r355_group % $(8)) * $(2304)) + (r355_row[:, None] * $(192)) + r355_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r355_group % $(8)) * $(2304)) + (r355_row[:, None] * $(192)) + r355_lane[None, :]) / $(192)) % $(12)) * $(192)) + ((((r355_group % $(8)) * $(2304)) + (r355_row[:, None] * $(192)) + r355_lane[None, :]) % $(192))), dtype=tl.float32)
      r355_right = tl.load($(candidate_vto_copy_156_ptr) + (((((((r355_group % $(8)) * $(9216)) + (r355_lane[:, None] * $(48)) + r355_col[None, :]) / $(9216)) % $(8)) * $(9216)) + (((((r355_group % $(8)) * $(9216)) + (r355_lane[:, None] * $(48)) + r355_col[None, :]) % $(48)) * $(192)) + (((((r355_group % $(8)) * $(9216)) + (r355_lane[:, None] * $(48)) + r355_col[None, :]) / $(48)) % $(192))), dtype=tl.float32)
      r355_value = tl.dot(r355_left, r355_right)
      r355_value = r355_value.to(tl.float32)
      tl.store($(candidate_vbmm_78_ptr) + ((r355_group * $(576)) + (r355_row[:, None] * $(48)) + r355_col[None, :]), r355_value)
    }
    r356_z = tl.arange($(4608))
    r356_value = (tl.load($(candidate_vbmm_78_ptr) + ((((r356_z / $(576)) % $(8)) * $(576)) + (((r356_z / $(48)) % $(12)) * $(48)) + (r356_z % $(48))), dtype=tl.float32) * 0.07216878364870322)
    r356_value = r356_value.to(tl.float32)
    r356_value = tl.broadcast_to(r356_value, [$(4608)])
    tl.store($(candidate_vmul_71_ptr) + r356_z, r356_value)
    r357_z = tl.arange($(12))
    r357_value = (tl.cast(r357_z, tl.int64) + tl.full([], $(48), dtype=tl.int64))
    r357_value = tl.broadcast_to(r357_value, [$(12)])
    tl.store($(candidate_vadd_50_ptr) + r357_z, r357_value)
    r358_z = tl.arange($(12))
    r358_value = (tl.load($(candidate_vadd_50_ptr) + r358_z, dtype=tl.int64) - tl.full([], $(12), dtype=tl.int64))
    r358_value = tl.broadcast_to(r358_value, [$(12)])
    tl.store($(candidate_vsub_45_ptr) + r358_z, r358_value)
    r359_z = tl.arange($(576))
    r359_value = (tl.cast((r359_z % $(48)), tl.int64) > tl.load($(candidate_vsub_45_ptr) + ((r359_z / $(48)) % $(12)), dtype=tl.int64))
    r359_value = tl.broadcast_to(r359_value, [$(576)])
    tl.store($(candidate_vgt_24_ptr) + r359_z, r359_value)
    r360_z = tl.arange($(4608))
    r360_value = tl.where(tl.load($(candidate_vgt_24_ptr) + ((((r360_z / $(48)) % $(12)) * $(48)) + (r360_z % $(48))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_71_ptr) + ((((r360_z / $(576)) % $(8)) * $(576)) + (((r360_z / $(48)) % $(12)) * $(48)) + (r360_z % $(48))), dtype=tl.float32))
    r360_value = r360_value.to(tl.float32)
    r360_value = tl.broadcast_to(r360_value, [$(4608)])
    tl.store($(candidate_vmasked_fill_24_ptr) + r360_z, r360_value)
    r361_z = tl.arange($(4608))
    r361_lane = tl.arange($(48))
    r361_row_values = tl.load($(candidate_vmasked_fill_24_ptr) + ((((r361_z[:, None] / $(576)) % $(8)) * $(576)) + (((r361_z[:, None] / $(48)) % $(12)) * $(48)) + (r361_lane[None, :] % $(48))), dtype=tl.float32)
    r361_maximum = tl.max(r361_row_values, axis=1, keep_dims=true)
    r361_weights = tl.exp(r361_row_values - r361_maximum)
    r361_total = tl.sum(r361_weights, axis=1)
    r361_maximum = tl.max(r361_row_values, axis=1)
    r361_value = tl.exp(tl.load($(candidate_vmasked_fill_24_ptr) + ((((r361_z / $(576)) % $(8)) * $(576)) + (((r361_z / $(48)) % $(12)) * $(48)) + (r361_z % $(48))), dtype=tl.float32) - r361_maximum) / r361_total
    r361_value = r361_value.to(tl.float32)
    r361_value = tl.broadcast_to(r361_value, [$(4608)])
    tl.store($(candidate_vsoftmax_18_ptr) + r361_z, r361_value)
    r362_z = tl.arange($(4608))
    r362_value = tl.load($(candidate_vsoftmax_18_ptr) + ((((r362_z / $(576)) % $(8)) * $(576)) + (((r362_z / $(48)) % $(12)) * $(48)) + (r362_z % $(48))), dtype=tl.float32)
    r362_value = r362_value.to(tl.bfloat16)
    r362_value = tl.broadcast_to(r362_value, [$(4608)])
    tl.store($(candidate_vto_copy_157_ptr) + r362_z, r362_value)
    r363_row = tl.arange($(12))
    r363_col = tl.arange($(128))
    r363_lane = tl.arange($(48))
    for r363_group in range($(8)) {
      r363_left = tl.load($(candidate_vto_copy_157_ptr) + (((((((r363_group % $(8)) * $(576)) + (r363_row[:, None] * $(48)) + r363_lane[None, :]) / $(576)) % $(8)) * $(576)) + ((((((r363_group % $(8)) * $(576)) + (r363_row[:, None] * $(48)) + r363_lane[None, :]) / $(48)) % $(12)) * $(48)) + ((((r363_group % $(8)) * $(576)) + (r363_row[:, None] * $(48)) + r363_lane[None, :]) % $(48))), dtype=tl.bfloat16)
      r363_right = tl.load($(candidate_vbmm_77_ptr) + (((((((r363_group % $(8)) * $(6144)) + (r363_lane[:, None] * $(128)) + r363_col[None, :]) / $(128)) % $(48)) * $(1024)) + ((((((r363_group % $(8)) * $(6144)) + (r363_lane[:, None] * $(128)) + r363_col[None, :]) / $(6144)) % $(8)) * $(128)) + ((((r363_group % $(8)) * $(6144)) + (r363_lane[:, None] * $(128)) + r363_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r363_value = tl.dot(r363_left, r363_right)
      r363_value = r363_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_79_ptr) + ((r363_group * $(1536)) + (r363_row[:, None] * $(128)) + r363_col[None, :]), r363_value)
    }
    r364_z = tl.arange($(96))
    r364_lane = tl.arange($(48))
    r364_row_values = tl.load($(candidate_vmasked_fill_24_ptr) + ((((r364_z[:, None] / $(12)) % $(8)) * $(576)) + ((r364_z[:, None] % $(12)) * $(48)) + (r364_lane[None, :] % $(48))), dtype=tl.float32)
    r364_maximum = tl.max(r364_row_values, axis=1, keep_dims=true)
    r364_weights = tl.exp(r364_row_values - r364_maximum)
    r364_total = tl.sum(r364_weights, axis=1)
    r364_maximum = tl.max(r364_row_values, axis=1)
    r364_value = r364_maximum + tl.log(r364_total)
    r364_value = r364_value.to(tl.float32)
    r364_value = tl.broadcast_to(r364_value, [$(96)])
    tl.store($(candidate_vlogsumexp_18_ptr) + r364_z, r364_value)
    r365_z = tl.arange($(6144))
    r365_value = tl.load($(candidate_vcat_39_ptr) + ((((r365_z / $(192)) % $(4)) * $(1536)) + (((r365_z / $(768)) % $(8)) * $(192)) + (r365_z % $(192)) + $(18432)), dtype=tl.bfloat16)
    r365_value = r365_value.to(tl.float32)
    r365_value = tl.broadcast_to(r365_value, [$(6144)])
    tl.store($(candidate_vto_copy_158_ptr) + r365_z, r365_value)
    r366_z = tl.arange($(24576))
    r366_value = tl.load($(candidate_vcat_40_ptr) + ((((r366_z / $(192)) % $(16)) * $(1536)) + (((r366_z / $(3072)) % $(8)) * $(192)) + (r366_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r366_value = r366_value.to(tl.float32)
    r366_value = tl.broadcast_to(r366_value, [$(24576)])
    tl.store($(candidate_vto_copy_159_ptr) + r366_z, r366_value)
    r367_row = tl.arange($(4))
    r367_col = tl.arange($(16))
    r367_lane = tl.arange($(192))
    for r367_group in range($(8)) {
      r367_left = tl.load($(candidate_vto_copy_158_ptr) + (((((((r367_group % $(8)) * $(768)) + (r367_row[:, None] * $(192)) + r367_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r367_group % $(8)) * $(768)) + (r367_row[:, None] * $(192)) + r367_lane[None, :]) / $(192)) % $(4)) * $(192)) + ((((r367_group % $(8)) * $(768)) + (r367_row[:, None] * $(192)) + r367_lane[None, :]) % $(192))), dtype=tl.float32)
      r367_right = tl.load($(candidate_vto_copy_159_ptr) + (((((((r367_group % $(8)) * $(3072)) + (r367_lane[:, None] * $(16)) + r367_col[None, :]) / $(3072)) % $(8)) * $(3072)) + (((((r367_group % $(8)) * $(3072)) + (r367_lane[:, None] * $(16)) + r367_col[None, :]) % $(16)) * $(192)) + (((((r367_group % $(8)) * $(3072)) + (r367_lane[:, None] * $(16)) + r367_col[None, :]) / $(16)) % $(192))), dtype=tl.float32)
      r367_value = tl.dot(r367_left, r367_right)
      r367_value = r367_value.to(tl.float32)
      tl.store($(candidate_vbmm_80_ptr) + ((r367_group * $(64)) + (r367_row[:, None] * $(16)) + r367_col[None, :]), r367_value)
    }
    r368_z = tl.arange($(512))
    r368_value = (tl.load($(candidate_vbmm_80_ptr) + ((((r368_z / $(64)) % $(8)) * $(64)) + (((r368_z / $(16)) % $(4)) * $(16)) + (r368_z % $(16))), dtype=tl.float32) * 0.07216878364870322)
    r368_value = r368_value.to(tl.float32)
    r368_value = tl.broadcast_to(r368_value, [$(512)])
    tl.store($(candidate_vmul_72_ptr) + r368_z, r368_value)
    r369_z = tl.arange($(4))
    r369_value = (tl.cast(r369_z, tl.int64) + tl.full([], $(16), dtype=tl.int64))
    r369_value = tl.broadcast_to(r369_value, [$(4)])
    tl.store($(candidate_vadd_51_ptr) + r369_z, r369_value)
    r370_z = tl.arange($(4))
    r370_value = (tl.load($(candidate_vadd_51_ptr) + r370_z, dtype=tl.int64) - tl.full([], $(4), dtype=tl.int64))
    r370_value = tl.broadcast_to(r370_value, [$(4)])
    tl.store($(candidate_vsub_46_ptr) + r370_z, r370_value)
    r371_z = tl.arange($(64))
    r371_value = (tl.cast((r371_z % $(16)), tl.int64) > tl.load($(candidate_vsub_46_ptr) + ((r371_z / $(16)) % $(4)), dtype=tl.int64))
    r371_value = tl.broadcast_to(r371_value, [$(64)])
    tl.store($(candidate_vgt_25_ptr) + r371_z, r371_value)
    r372_z = tl.arange($(512))
    r372_value = tl.where(tl.load($(candidate_vgt_25_ptr) + ((((r372_z / $(16)) % $(4)) * $(16)) + (r372_z % $(16))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_72_ptr) + ((((r372_z / $(64)) % $(8)) * $(64)) + (((r372_z / $(16)) % $(4)) * $(16)) + (r372_z % $(16))), dtype=tl.float32))
    r372_value = r372_value.to(tl.float32)
    r372_value = tl.broadcast_to(r372_value, [$(512)])
    tl.store($(candidate_vmasked_fill_25_ptr) + r372_z, r372_value)
    r373_z = tl.arange($(512))
    r373_lane = tl.arange($(16))
    r373_row_values = tl.load($(candidate_vmasked_fill_25_ptr) + ((((r373_z[:, None] / $(64)) % $(8)) * $(64)) + (((r373_z[:, None] / $(16)) % $(4)) * $(16)) + (r373_lane[None, :] % $(16))), dtype=tl.float32)
    r373_maximum = tl.max(r373_row_values, axis=1, keep_dims=true)
    r373_weights = tl.exp(r373_row_values - r373_maximum)
    r373_total = tl.sum(r373_weights, axis=1)
    r373_maximum = tl.max(r373_row_values, axis=1)
    r373_value = tl.exp(tl.load($(candidate_vmasked_fill_25_ptr) + ((((r373_z / $(64)) % $(8)) * $(64)) + (((r373_z / $(16)) % $(4)) * $(16)) + (r373_z % $(16))), dtype=tl.float32) - r373_maximum) / r373_total
    r373_value = r373_value.to(tl.float32)
    r373_value = tl.broadcast_to(r373_value, [$(512)])
    tl.store($(candidate_vsoftmax_19_ptr) + r373_z, r373_value)
    r374_z = tl.arange($(512))
    r374_value = tl.load($(candidate_vsoftmax_19_ptr) + ((((r374_z / $(64)) % $(8)) * $(64)) + (((r374_z / $(16)) % $(4)) * $(16)) + (r374_z % $(16))), dtype=tl.float32)
    r374_value = r374_value.to(tl.bfloat16)
    r374_value = tl.broadcast_to(r374_value, [$(512)])
    tl.store($(candidate_vto_copy_160_ptr) + r374_z, r374_value)
    r375_row = tl.arange($(4))
    r375_col = tl.arange($(128))
    r375_lane = tl.arange($(16))
    for r375_group in range($(8)) {
      r375_left = tl.load($(candidate_vto_copy_160_ptr) + (((((((r375_group % $(8)) * $(64)) + (r375_row[:, None] * $(16)) + r375_lane[None, :]) / $(64)) % $(8)) * $(64)) + ((((((r375_group % $(8)) * $(64)) + (r375_row[:, None] * $(16)) + r375_lane[None, :]) / $(16)) % $(4)) * $(16)) + ((((r375_group % $(8)) * $(64)) + (r375_row[:, None] * $(16)) + r375_lane[None, :]) % $(16))), dtype=tl.bfloat16)
      r375_right = tl.load($(candidate_vbmm_77_ptr) + (((((((r375_group % $(8)) * $(2048)) + (r375_lane[:, None] * $(128)) + r375_col[None, :]) / $(128)) % $(16)) * $(1024)) + ((((((r375_group % $(8)) * $(2048)) + (r375_lane[:, None] * $(128)) + r375_col[None, :]) / $(2048)) % $(8)) * $(128)) + ((((r375_group % $(8)) * $(2048)) + (r375_lane[:, None] * $(128)) + r375_col[None, :]) % $(128)) + $(49152)), dtype=tl.bfloat16)
      r375_value = tl.dot(r375_left, r375_right)
      r375_value = r375_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_81_ptr) + ((r375_group * $(512)) + (r375_row[:, None] * $(128)) + r375_col[None, :]), r375_value)
    }
    r376_z = tl.arange($(32))
    r376_lane = tl.arange($(16))
    r376_row_values = tl.load($(candidate_vmasked_fill_25_ptr) + ((((r376_z[:, None] / $(4)) % $(8)) * $(64)) + ((r376_z[:, None] % $(4)) * $(16)) + (r376_lane[None, :] % $(16))), dtype=tl.float32)
    r376_maximum = tl.max(r376_row_values, axis=1, keep_dims=true)
    r376_weights = tl.exp(r376_row_values - r376_maximum)
    r376_total = tl.sum(r376_weights, axis=1)
    r376_maximum = tl.max(r376_row_values, axis=1)
    r376_value = r376_maximum + tl.log(r376_total)
    r376_value = r376_value.to(tl.float32)
    r376_value = tl.broadcast_to(r376_value, [$(32)])
    tl.store($(candidate_vlogsumexp_19_ptr) + r376_z, r376_value)
    r377_z = tl.arange($(24576))
    r377_value = tl.load($(candidate_vcat_39_ptr) + ((((r377_z / $(192)) % $(16)) * $(1536)) + (((r377_z / $(3072)) % $(8)) * $(192)) + (r377_z % $(192)) + $(24576)), dtype=tl.bfloat16)
    r377_value = r377_value.to(tl.float32)
    r377_value = tl.broadcast_to(r377_value, [$(24576)])
    tl.store($(candidate_vto_copy_161_ptr) + r377_z, r377_value)
    r378_z = tl.arange($(98304))
    r378_value = tl.load($(candidate_vcat_40_ptr) + ((((r378_z / $(192)) % $(64)) * $(1536)) + (((r378_z / $(12288)) % $(8)) * $(192)) + (r378_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r378_value = r378_value.to(tl.float32)
    r378_value = tl.broadcast_to(r378_value, [$(98304)])
    tl.store($(candidate_vto_copy_162_ptr) + r378_z, r378_value)
    r379_row = tl.arange($(16))
    r379_col = tl.arange($(64))
    r379_lane = tl.arange($(192))
    for r379_group in range($(8)) {
      r379_left = tl.load($(candidate_vto_copy_161_ptr) + (((((((r379_group % $(8)) * $(3072)) + (r379_row[:, None] * $(192)) + r379_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r379_group % $(8)) * $(3072)) + (r379_row[:, None] * $(192)) + r379_lane[None, :]) / $(192)) % $(16)) * $(192)) + ((((r379_group % $(8)) * $(3072)) + (r379_row[:, None] * $(192)) + r379_lane[None, :]) % $(192))), dtype=tl.float32)
      r379_right = tl.load($(candidate_vto_copy_162_ptr) + (((((((r379_group % $(8)) * $(12288)) + (r379_lane[:, None] * $(64)) + r379_col[None, :]) / $(12288)) % $(8)) * $(12288)) + (((((r379_group % $(8)) * $(12288)) + (r379_lane[:, None] * $(64)) + r379_col[None, :]) % $(64)) * $(192)) + (((((r379_group % $(8)) * $(12288)) + (r379_lane[:, None] * $(64)) + r379_col[None, :]) / $(64)) % $(192))), dtype=tl.float32)
      r379_value = tl.dot(r379_left, r379_right)
      r379_value = r379_value.to(tl.float32)
      tl.store($(candidate_vbmm_82_ptr) + ((r379_group * $(1024)) + (r379_row[:, None] * $(64)) + r379_col[None, :]), r379_value)
    }
    r380_z = tl.arange($(8192))
    r380_value = (tl.load($(candidate_vbmm_82_ptr) + ((((r380_z / $(1024)) % $(8)) * $(1024)) + (((r380_z / $(64)) % $(16)) * $(64)) + (r380_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r380_value = r380_value.to(tl.float32)
    r380_value = tl.broadcast_to(r380_value, [$(8192)])
    tl.store($(candidate_vmul_73_ptr) + r380_z, r380_value)
    r381_z = tl.arange($(16))
    r381_value = (tl.cast(r381_z, tl.int64) + tl.full([], $(64), dtype=tl.int64))
    r381_value = tl.broadcast_to(r381_value, [$(16)])
    tl.store($(candidate_vadd_52_ptr) + r381_z, r381_value)
    r382_z = tl.arange($(16))
    r382_value = (tl.load($(candidate_vadd_52_ptr) + r382_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r382_value = tl.broadcast_to(r382_value, [$(16)])
    tl.store($(candidate_vsub_47_ptr) + r382_z, r382_value)
    r383_z = tl.arange($(1024))
    r383_value = (tl.cast((r383_z % $(64)), tl.int64) > tl.load($(candidate_vsub_47_ptr) + ((r383_z / $(64)) % $(16)), dtype=tl.int64))
    r383_value = tl.broadcast_to(r383_value, [$(1024)])
    tl.store($(candidate_vgt_26_ptr) + r383_z, r383_value)
    r384_z = tl.arange($(8192))
    r384_value = tl.where(tl.load($(candidate_vgt_26_ptr) + ((((r384_z / $(64)) % $(16)) * $(64)) + (r384_z % $(64))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_73_ptr) + ((((r384_z / $(1024)) % $(8)) * $(1024)) + (((r384_z / $(64)) % $(16)) * $(64)) + (r384_z % $(64))), dtype=tl.float32))
    r384_value = r384_value.to(tl.float32)
    r384_value = tl.broadcast_to(r384_value, [$(8192)])
    tl.store($(candidate_vmasked_fill_26_ptr) + r384_z, r384_value)
    r385_z = tl.arange($(8192))
    r385_lane = tl.arange($(64))
    r385_row_values = tl.load($(candidate_vmasked_fill_26_ptr) + ((((r385_z[:, None] / $(1024)) % $(8)) * $(1024)) + (((r385_z[:, None] / $(64)) % $(16)) * $(64)) + (r385_lane[None, :] % $(64))), dtype=tl.float32)
    r385_maximum = tl.max(r385_row_values, axis=1, keep_dims=true)
    r385_weights = tl.exp(r385_row_values - r385_maximum)
    r385_total = tl.sum(r385_weights, axis=1)
    r385_maximum = tl.max(r385_row_values, axis=1)
    r385_value = tl.exp(tl.load($(candidate_vmasked_fill_26_ptr) + ((((r385_z / $(1024)) % $(8)) * $(1024)) + (((r385_z / $(64)) % $(16)) * $(64)) + (r385_z % $(64))), dtype=tl.float32) - r385_maximum) / r385_total
    r385_value = r385_value.to(tl.float32)
    r385_value = tl.broadcast_to(r385_value, [$(8192)])
    tl.store($(candidate_vsoftmax_20_ptr) + r385_z, r385_value)
    r386_z = tl.arange($(8192))
    r386_value = tl.load($(candidate_vsoftmax_20_ptr) + ((((r386_z / $(1024)) % $(8)) * $(1024)) + (((r386_z / $(64)) % $(16)) * $(64)) + (r386_z % $(64))), dtype=tl.float32)
    r386_value = r386_value.to(tl.bfloat16)
    r386_value = tl.broadcast_to(r386_value, [$(8192)])
    tl.store($(candidate_vto_copy_163_ptr) + r386_z, r386_value)
    r387_row = tl.arange($(16))
    r387_col = tl.arange($(128))
    r387_lane = tl.arange($(64))
    for r387_group in range($(8)) {
      r387_left = tl.load($(candidate_vto_copy_163_ptr) + (((((((r387_group % $(8)) * $(1024)) + (r387_row[:, None] * $(64)) + r387_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + ((((((r387_group % $(8)) * $(1024)) + (r387_row[:, None] * $(64)) + r387_lane[None, :]) / $(64)) % $(16)) * $(64)) + ((((r387_group % $(8)) * $(1024)) + (r387_row[:, None] * $(64)) + r387_lane[None, :]) % $(64))), dtype=tl.bfloat16)
      r387_right = tl.load($(candidate_vbmm_77_ptr) + (((((((r387_group % $(8)) * $(8192)) + (r387_lane[:, None] * $(128)) + r387_col[None, :]) / $(128)) % $(64)) * $(1024)) + ((((((r387_group % $(8)) * $(8192)) + (r387_lane[:, None] * $(128)) + r387_col[None, :]) / $(8192)) % $(8)) * $(128)) + ((((r387_group % $(8)) * $(8192)) + (r387_lane[:, None] * $(128)) + r387_col[None, :]) % $(128)) + $(65536)), dtype=tl.bfloat16)
      r387_value = tl.dot(r387_left, r387_right)
      r387_value = r387_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_83_ptr) + ((r387_group * $(2048)) + (r387_row[:, None] * $(128)) + r387_col[None, :]), r387_value)
    }
    r388_z = tl.arange($(128))
    r388_lane = tl.arange($(64))
    r388_row_values = tl.load($(candidate_vmasked_fill_26_ptr) + ((((r388_z[:, None] / $(16)) % $(8)) * $(1024)) + ((r388_z[:, None] % $(16)) * $(64)) + (r388_lane[None, :] % $(64))), dtype=tl.float32)
    r388_maximum = tl.max(r388_row_values, axis=1, keep_dims=true)
    r388_weights = tl.exp(r388_row_values - r388_maximum)
    r388_total = tl.sum(r388_weights, axis=1)
    r388_maximum = tl.max(r388_row_values, axis=1)
    r388_value = r388_maximum + tl.log(r388_total)
    r388_value = r388_value.to(tl.float32)
    r388_value = tl.broadcast_to(r388_value, [$(128)])
    tl.store($(candidate_vlogsumexp_20_ptr) + r388_z, r388_value)
    r389_z = tl.arange($(32768))
    r389_value = tl.where(((r389_z / $(1024)) % $(32)) < $(12), tl.load($(candidate_vbmm_79_ptr) + ((((((((r389_z / $(128)) % $(8)) * $(1536)) + ((((r389_z / $(1024)) % $(32)) - $(0)) * $(128)) + (r389_z % $(128))) / $(1536)) % $(8)) * $(1536)) + ((((((r389_z / $(128)) % $(8)) * $(12)) + (((r389_z / $(1024)) % $(32)) - $(0))) % $(12)) * $(128)) + (r389_z % $(128))), dtype=tl.bfloat16), tl.where(((r389_z / $(1024)) % $(32)) < $(16), tl.load($(candidate_vbmm_81_ptr) + ((((((((r389_z / $(128)) % $(8)) * $(512)) + ((((r389_z / $(1024)) % $(32)) - $(12)) * $(128)) + (r389_z % $(128))) / $(512)) % $(8)) * $(512)) + ((((((r389_z / $(128)) % $(8)) * $(4)) + (((r389_z / $(1024)) % $(32)) - $(12))) % $(4)) * $(128)) + (r389_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vbmm_83_ptr) + ((((((((r389_z / $(128)) % $(8)) * $(2048)) + ((((r389_z / $(1024)) % $(32)) - $(16)) * $(128)) + (r389_z % $(128))) / $(2048)) % $(8)) * $(2048)) + ((((((r389_z / $(128)) % $(8)) * $(16)) + (((r389_z / $(1024)) % $(32)) - $(16))) % $(16)) * $(128)) + (r389_z % $(128))), dtype=tl.bfloat16)))
    r389_value = r389_value.to(tl.bfloat16)
    r389_value = tl.broadcast_to(r389_value, [$(32768)])
    tl.store($(candidate_vcat_41_ptr) + r389_z, r389_value)
    r390_z = tl.arange($(256))
    r390_value = tl.where(((r390_z / $(8)) % $(32)) < $(12), tl.load($(candidate_vlogsumexp_18_ptr) + (((r390_z % $(8)) * $(12)) + ((((r390_z / $(8)) % $(32)) - $(0)) % $(12))), dtype=tl.float32), tl.where(((r390_z / $(8)) % $(32)) < $(16), tl.load($(candidate_vlogsumexp_19_ptr) + (((r390_z % $(8)) * $(4)) + ((((r390_z / $(8)) % $(32)) - $(12)) % $(4))), dtype=tl.float32), tl.load($(candidate_vlogsumexp_20_ptr) + (((r390_z % $(8)) * $(16)) + ((((r390_z / $(8)) % $(32)) - $(16)) % $(16))), dtype=tl.float32)))
    r390_value = r390_value.to(tl.float32)
    r390_value = tl.broadcast_to(r390_value, [$(256)])
    tl.store($(candidate_vcat_42_ptr) + r390_z, r390_value)
    r391_z = tl.arange($(18432))
    r391_value = tl.load($(candidate_vcat_39_ptr) + ((((r391_z / $(192)) % $(12)) * $(1536)) + (((r391_z / $(2304)) % $(8)) * $(192)) + (r391_z % $(192))), dtype=tl.bfloat16)
    r391_value = r391_value.to(tl.float32)
    r391_value = tl.broadcast_to(r391_value, [$(18432)])
    tl.store($(candidate_vto_copy_164_ptr) + r391_z, r391_value)
    r392_z = tl.arange($(73728))
    r392_value = tl.load($(candidate_vcat_40_ptr) + ((((r392_z / $(192)) % $(48)) * $(1536)) + (((r392_z / $(9216)) % $(8)) * $(192)) + (r392_z % $(192))), dtype=tl.bfloat16)
    r392_value = r392_value.to(tl.float32)
    r392_value = tl.broadcast_to(r392_value, [$(73728)])
    tl.store($(candidate_vto_copy_165_ptr) + r392_z, r392_value)
    r393_z = tl.arange($(49152))
    r393_value = tl.load($(candidate_vbmm_77_ptr) + ((((r393_z / $(128)) % $(48)) * $(1024)) + (((r393_z / $(6144)) % $(8)) * $(128)) + (r393_z % $(128))), dtype=tl.bfloat16)
    r393_value = r393_value.to(tl.float32)
    r393_value = tl.broadcast_to(r393_value, [$(49152)])
    tl.store($(candidate_vto_copy_166_ptr) + r393_z, r393_value)
    r394_z = tl.arange($(12288))
    r394_value = tl.load($(candidate_vmm_28_ptr) + ((((r394_z / $(128)) % $(12)) * $(1024)) + (((r394_z / $(1536)) % $(8)) * $(128)) + (r394_z % $(128)) + $(32768)), dtype=tl.bfloat16)
    r394_value = r394_value.to(tl.float32)
    r394_value = tl.broadcast_to(r394_value, [$(12288)])
    tl.store($(candidate_vto_copy_167_ptr) + r394_z, r394_value)
    r395_z = tl.arange($(12288))
    r395_value = tl.load($(candidate_vcat_41_ptr) + ((((r395_z / $(128)) % $(12)) * $(1024)) + (((r395_z / $(1536)) % $(8)) * $(128)) + (r395_z % $(128))), dtype=tl.bfloat16)
    r395_value = r395_value.to(tl.float32)
    r395_value = tl.broadcast_to(r395_value, [$(12288)])
    tl.store($(candidate_vto_copy_168_ptr) + r395_z, r395_value)
    r396_row = tl.arange($(12))
    r396_col = tl.arange($(48))
    r396_lane = tl.arange($(192))
    for r396_group in range($(8)) {
      r396_left = tl.load($(candidate_vto_copy_164_ptr) + (((((((r396_group % $(8)) * $(2304)) + (r396_row[:, None] * $(192)) + r396_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r396_group % $(8)) * $(2304)) + (r396_row[:, None] * $(192)) + r396_lane[None, :]) / $(192)) % $(12)) * $(192)) + ((((r396_group % $(8)) * $(2304)) + (r396_row[:, None] * $(192)) + r396_lane[None, :]) % $(192))), dtype=tl.float32)
      r396_right = tl.load($(candidate_vto_copy_165_ptr) + (((((((r396_group % $(8)) * $(9216)) + (r396_lane[:, None] * $(48)) + r396_col[None, :]) / $(9216)) % $(8)) * $(9216)) + (((((r396_group % $(8)) * $(9216)) + (r396_lane[:, None] * $(48)) + r396_col[None, :]) % $(48)) * $(192)) + (((((r396_group % $(8)) * $(9216)) + (r396_lane[:, None] * $(48)) + r396_col[None, :]) / $(48)) % $(192))), dtype=tl.float32)
      r396_value = tl.dot(r396_left, r396_right)
      r396_value = r396_value.to(tl.float32)
      tl.store($(candidate_vbmm_84_ptr) + ((r396_group * $(576)) + (r396_row[:, None] * $(48)) + r396_col[None, :]), r396_value)
    }
    r397_z = tl.arange($(4608))
    r397_value = (tl.load($(candidate_vbmm_84_ptr) + ((((r397_z / $(576)) % $(8)) * $(576)) + (((r397_z / $(48)) % $(12)) * $(48)) + (r397_z % $(48))), dtype=tl.float32) * 0.07216878364870322)
    r397_value = r397_value.to(tl.float32)
    r397_value = tl.broadcast_to(r397_value, [$(4608)])
    tl.store($(candidate_vmul_74_ptr) + r397_z, r397_value)
    r398_z = tl.arange($(12))
    r398_value = (tl.cast(r398_z, tl.int64) + tl.full([], $(48), dtype=tl.int64))
    r398_value = tl.broadcast_to(r398_value, [$(12)])
    tl.store($(candidate_vadd_53_ptr) + r398_z, r398_value)
    r399_z = tl.arange($(12))
    r399_value = (tl.load($(candidate_vadd_53_ptr) + r399_z, dtype=tl.int64) - tl.full([], $(12), dtype=tl.int64))
    r399_value = tl.broadcast_to(r399_value, [$(12)])
    tl.store($(candidate_vsub_48_ptr) + r399_z, r399_value)
    r400_z = tl.arange($(576))
    r400_value = (tl.cast((r400_z % $(48)), tl.int64) > tl.load($(candidate_vsub_48_ptr) + ((r400_z / $(48)) % $(12)), dtype=tl.int64))
    r400_value = tl.broadcast_to(r400_value, [$(576)])
    tl.store($(candidate_vgt_27_ptr) + r400_z, r400_value)
    r401_z = tl.arange($(4608))
    r401_value = tl.where(tl.load($(candidate_vgt_27_ptr) + ((((r401_z / $(48)) % $(12)) * $(48)) + (r401_z % $(48))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_74_ptr) + ((((r401_z / $(576)) % $(8)) * $(576)) + (((r401_z / $(48)) % $(12)) * $(48)) + (r401_z % $(48))), dtype=tl.float32))
    r401_value = r401_value.to(tl.float32)
    r401_value = tl.broadcast_to(r401_value, [$(4608)])
    tl.store($(candidate_vmasked_fill_27_ptr) + r401_z, r401_value)
    r402_z = tl.arange($(4608))
    r402_value = (tl.load($(candidate_vmasked_fill_27_ptr) + ((((r402_z / $(576)) % $(8)) * $(576)) + (((r402_z / $(48)) % $(12)) * $(48)) + (r402_z % $(48))), dtype=tl.float32) - tl.load($(candidate_vcat_42_ptr) + ((((r402_z / $(48)) % $(12)) * $(8)) + ((r402_z / $(576)) % $(8))), dtype=tl.float32))
    r402_value = r402_value.to(tl.float32)
    r402_value = tl.broadcast_to(r402_value, [$(4608)])
    tl.store($(candidate_vsub_49_ptr) + r402_z, r402_value)
    r403_z = tl.arange($(4608))
    r403_value = tl.exp(tl.load($(candidate_vsub_49_ptr) + ((((r403_z / $(576)) % $(8)) * $(576)) + (((r403_z / $(48)) % $(12)) * $(48)) + (r403_z % $(48))), dtype=tl.float32))
    r403_value = r403_value.to(tl.float32)
    r403_value = tl.broadcast_to(r403_value, [$(4608)])
    tl.store($(candidate_vexp_6_ptr) + r403_z, r403_value)
    r404_z = tl.arange($(12288))
    r404_value = (tl.load($(candidate_vto_copy_167_ptr) + ((((r404_z / $(1536)) % $(8)) * $(1536)) + (((r404_z / $(128)) % $(12)) * $(128)) + (r404_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_168_ptr) + ((((r404_z / $(1536)) % $(8)) * $(1536)) + (((r404_z / $(128)) % $(12)) * $(128)) + (r404_z % $(128))), dtype=tl.float32))
    r404_value = r404_value.to(tl.float32)
    r404_value = tl.broadcast_to(r404_value, [$(12288)])
    tl.store($(candidate_vmul_75_ptr) + r404_z, r404_value)
    r405_z = tl.arange($(96))
    r405_lane = tl.arange($(128))
    r405_value = tl.sum(tl.load($(candidate_vmul_75_ptr) + ((((r405_z[:, None] / $(12)) % $(8)) * $(1536)) + ((r405_z[:, None] % $(12)) * $(128)) + r405_lane[None, :]), dtype=tl.float32), axis=1)
    r405_value = r405_value.to(tl.float32)
    r405_value = tl.broadcast_to(r405_value, [$(96)])
    tl.store($(candidate_vsum_9_ptr) + r405_z, r405_value)
    r406_row = tl.arange($(12))
    r406_col = tl.arange($(48))
    r406_lane = tl.arange($(128))
    for r406_group in range($(8)) {
      r406_left = tl.load($(candidate_vto_copy_167_ptr) + (((((((r406_group % $(8)) * $(1536)) + (r406_row[:, None] * $(128)) + r406_lane[None, :]) / $(1536)) % $(8)) * $(1536)) + ((((((r406_group % $(8)) * $(1536)) + (r406_row[:, None] * $(128)) + r406_lane[None, :]) / $(128)) % $(12)) * $(128)) + ((((r406_group % $(8)) * $(1536)) + (r406_row[:, None] * $(128)) + r406_lane[None, :]) % $(128))), dtype=tl.float32)
      r406_right = tl.load($(candidate_vto_copy_166_ptr) + (((((((r406_group % $(8)) * $(6144)) + (r406_lane[:, None] * $(48)) + r406_col[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r406_group % $(8)) * $(6144)) + (r406_lane[:, None] * $(48)) + r406_col[None, :]) % $(48)) * $(128)) + (((((r406_group % $(8)) * $(6144)) + (r406_lane[:, None] * $(48)) + r406_col[None, :]) / $(48)) % $(128))), dtype=tl.float32)
      r406_value = tl.dot(r406_left, r406_right)
      r406_value = r406_value.to(tl.float32)
      tl.store($(candidate_vbmm_85_ptr) + ((r406_group * $(576)) + (r406_row[:, None] * $(48)) + r406_col[None, :]), r406_value)
    }
    r407_z = tl.arange($(4608))
    r407_value = (tl.load($(candidate_vbmm_85_ptr) + ((((r407_z / $(576)) % $(8)) * $(576)) + (((r407_z / $(48)) % $(12)) * $(48)) + (r407_z % $(48))), dtype=tl.float32) - tl.load($(candidate_vsum_9_ptr) + ((((r407_z / $(576)) % $(8)) * $(12)) + ((r407_z / $(48)) % $(12))), dtype=tl.float32))
    r407_value = r407_value.to(tl.float32)
    r407_value = tl.broadcast_to(r407_value, [$(4608)])
    tl.store($(candidate_vsub_50_ptr) + r407_z, r407_value)
    r408_z = tl.arange($(4608))
    r408_value = (tl.load($(candidate_vexp_6_ptr) + ((((r408_z / $(576)) % $(8)) * $(576)) + (((r408_z / $(48)) % $(12)) * $(48)) + (r408_z % $(48))), dtype=tl.float32) * tl.load($(candidate_vsub_50_ptr) + ((((r408_z / $(576)) % $(8)) * $(576)) + (((r408_z / $(48)) % $(12)) * $(48)) + (r408_z % $(48))), dtype=tl.float32))
    r408_value = r408_value.to(tl.float32)
    r408_value = tl.broadcast_to(r408_value, [$(4608)])
    tl.store($(candidate_vmul_76_ptr) + r408_z, r408_value)
    r409_row = tl.arange($(12))
    r409_col = tl.arange($(192))
    r409_lane = tl.arange($(48))
    for r409_group in range($(8)) {
      r409_left = tl.load($(candidate_vmul_76_ptr) + (((((((r409_group % $(8)) * $(576)) + (r409_row[:, None] * $(48)) + r409_lane[None, :]) / $(576)) % $(8)) * $(576)) + ((((((r409_group % $(8)) * $(576)) + (r409_row[:, None] * $(48)) + r409_lane[None, :]) / $(48)) % $(12)) * $(48)) + ((((r409_group % $(8)) * $(576)) + (r409_row[:, None] * $(48)) + r409_lane[None, :]) % $(48))), dtype=tl.float32)
      r409_right = tl.load($(candidate_vto_copy_165_ptr) + (((((((r409_group % $(8)) * $(9216)) + (r409_lane[:, None] * $(192)) + r409_col[None, :]) / $(9216)) % $(8)) * $(9216)) + ((((((r409_group % $(8)) * $(9216)) + (r409_lane[:, None] * $(192)) + r409_col[None, :]) / $(192)) % $(48)) * $(192)) + ((((r409_group % $(8)) * $(9216)) + (r409_lane[:, None] * $(192)) + r409_col[None, :]) % $(192))), dtype=tl.float32)
      r409_value = tl.dot(r409_left, r409_right)
      r409_value = r409_value.to(tl.float32)
      tl.store($(candidate_vbmm_86_ptr) + ((r409_group * $(2304)) + (r409_row[:, None] * $(192)) + r409_col[None, :]), r409_value)
    }
    r410_z = tl.arange($(18432))
    r410_value = (tl.load($(candidate_vbmm_86_ptr) + ((((r410_z / $(2304)) % $(8)) * $(2304)) + (((r410_z / $(192)) % $(12)) * $(192)) + (r410_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r410_value = r410_value.to(tl.float32)
    r410_value = tl.broadcast_to(r410_value, [$(18432)])
    tl.store($(candidate_vmul_77_ptr) + r410_z, r410_value)
    r411_z = tl.arange($(18432))
    r411_value = tl.load($(candidate_vmul_77_ptr) + ((((r411_z / $(192)) % $(8)) * $(2304)) + (((r411_z / $(1536)) % $(12)) * $(192)) + (r411_z % $(192))), dtype=tl.float32)
    r411_value = r411_value.to(tl.bfloat16)
    r411_value = tl.broadcast_to(r411_value, [$(18432)])
    tl.store($(candidate_vto_copy_169_ptr) + r411_z, r411_value)
    r412_row = tl.arange($(48))
    r412_col = tl.arange($(192))
    r412_lane = tl.arange($(12))
    for r412_group in range($(8)) {
      r412_left = tl.load($(candidate_vmul_76_ptr) + (((((((r412_group % $(8)) * $(576)) + (r412_row[:, None] * $(12)) + r412_lane[None, :]) / $(576)) % $(8)) * $(576)) + (((((r412_group % $(8)) * $(576)) + (r412_row[:, None] * $(12)) + r412_lane[None, :]) % $(12)) * $(48)) + (((((r412_group % $(8)) * $(576)) + (r412_row[:, None] * $(12)) + r412_lane[None, :]) / $(12)) % $(48))), dtype=tl.float32)
      r412_right = tl.load($(candidate_vto_copy_164_ptr) + (((((((r412_group % $(8)) * $(2304)) + (r412_lane[:, None] * $(192)) + r412_col[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r412_group % $(8)) * $(2304)) + (r412_lane[:, None] * $(192)) + r412_col[None, :]) / $(192)) % $(12)) * $(192)) + ((((r412_group % $(8)) * $(2304)) + (r412_lane[:, None] * $(192)) + r412_col[None, :]) % $(192))), dtype=tl.float32)
      r412_value = tl.dot(r412_left, r412_right)
      r412_value = r412_value.to(tl.float32)
      tl.store($(candidate_vbmm_87_ptr) + ((r412_group * $(9216)) + (r412_row[:, None] * $(192)) + r412_col[None, :]), r412_value)
    }
    r413_z = tl.arange($(73728))
    r413_value = (tl.load($(candidate_vbmm_87_ptr) + ((((r413_z / $(9216)) % $(8)) * $(9216)) + (((r413_z / $(192)) % $(48)) * $(192)) + (r413_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r413_value = r413_value.to(tl.float32)
    r413_value = tl.broadcast_to(r413_value, [$(73728)])
    tl.store($(candidate_vmul_78_ptr) + r413_z, r413_value)
    r414_z = tl.arange($(73728))
    r414_value = tl.load($(candidate_vmul_78_ptr) + ((((r414_z / $(192)) % $(8)) * $(9216)) + (((r414_z / $(1536)) % $(48)) * $(192)) + (r414_z % $(192))), dtype=tl.float32)
    r414_value = r414_value.to(tl.bfloat16)
    r414_value = tl.broadcast_to(r414_value, [$(73728)])
    tl.store($(candidate_vto_copy_170_ptr) + r414_z, r414_value)
    r415_row = tl.arange($(48))
    r415_col = tl.arange($(128))
    r415_lane = tl.arange($(12))
    for r415_group in range($(8)) {
      r415_left = tl.load($(candidate_vexp_6_ptr) + (((((((r415_group % $(8)) * $(576)) + (r415_row[:, None] * $(12)) + r415_lane[None, :]) / $(576)) % $(8)) * $(576)) + (((((r415_group % $(8)) * $(576)) + (r415_row[:, None] * $(12)) + r415_lane[None, :]) % $(12)) * $(48)) + (((((r415_group % $(8)) * $(576)) + (r415_row[:, None] * $(12)) + r415_lane[None, :]) / $(12)) % $(48))), dtype=tl.float32)
      r415_right = tl.load($(candidate_vto_copy_167_ptr) + (((((((r415_group % $(8)) * $(1536)) + (r415_lane[:, None] * $(128)) + r415_col[None, :]) / $(1536)) % $(8)) * $(1536)) + ((((((r415_group % $(8)) * $(1536)) + (r415_lane[:, None] * $(128)) + r415_col[None, :]) / $(128)) % $(12)) * $(128)) + ((((r415_group % $(8)) * $(1536)) + (r415_lane[:, None] * $(128)) + r415_col[None, :]) % $(128))), dtype=tl.float32)
      r415_value = tl.dot(r415_left, r415_right)
      r415_value = r415_value.to(tl.float32)
      tl.store($(candidate_vbmm_88_ptr) + ((r415_group * $(6144)) + (r415_row[:, None] * $(128)) + r415_col[None, :]), r415_value)
    }
    r416_z = tl.arange($(49152))
    r416_value = tl.load($(candidate_vbmm_88_ptr) + ((((r416_z / $(128)) % $(8)) * $(6144)) + (((r416_z / $(1024)) % $(48)) * $(128)) + (r416_z % $(128))), dtype=tl.float32)
    r416_value = r416_value.to(tl.bfloat16)
    r416_value = tl.broadcast_to(r416_value, [$(49152)])
    tl.store($(candidate_vto_copy_171_ptr) + r416_z, r416_value)
    r417_z = tl.arange($(6144))
    r417_value = tl.load($(candidate_vcat_39_ptr) + ((((r417_z / $(192)) % $(4)) * $(1536)) + (((r417_z / $(768)) % $(8)) * $(192)) + (r417_z % $(192)) + $(18432)), dtype=tl.bfloat16)
    r417_value = r417_value.to(tl.float32)
    r417_value = tl.broadcast_to(r417_value, [$(6144)])
    tl.store($(candidate_vto_copy_172_ptr) + r417_z, r417_value)
    r418_z = tl.arange($(24576))
    r418_value = tl.load($(candidate_vcat_40_ptr) + ((((r418_z / $(192)) % $(16)) * $(1536)) + (((r418_z / $(3072)) % $(8)) * $(192)) + (r418_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r418_value = r418_value.to(tl.float32)
    r418_value = tl.broadcast_to(r418_value, [$(24576)])
    tl.store($(candidate_vto_copy_173_ptr) + r418_z, r418_value)
    r419_z = tl.arange($(16384))
    r419_value = tl.load($(candidate_vbmm_77_ptr) + ((((r419_z / $(128)) % $(16)) * $(1024)) + (((r419_z / $(2048)) % $(8)) * $(128)) + (r419_z % $(128)) + $(49152)), dtype=tl.bfloat16)
    r419_value = r419_value.to(tl.float32)
    r419_value = tl.broadcast_to(r419_value, [$(16384)])
    tl.store($(candidate_vto_copy_174_ptr) + r419_z, r419_value)
    r420_z = tl.arange($(4096))
    r420_value = tl.load($(candidate_vmm_28_ptr) + ((((r420_z / $(128)) % $(4)) * $(1024)) + (((r420_z / $(512)) % $(8)) * $(128)) + (r420_z % $(128)) + $(45056)), dtype=tl.bfloat16)
    r420_value = r420_value.to(tl.float32)
    r420_value = tl.broadcast_to(r420_value, [$(4096)])
    tl.store($(candidate_vto_copy_175_ptr) + r420_z, r420_value)
    r421_z = tl.arange($(4096))
    r421_value = tl.load($(candidate_vcat_41_ptr) + ((((r421_z / $(128)) % $(4)) * $(1024)) + (((r421_z / $(512)) % $(8)) * $(128)) + (r421_z % $(128)) + $(12288)), dtype=tl.bfloat16)
    r421_value = r421_value.to(tl.float32)
    r421_value = tl.broadcast_to(r421_value, [$(4096)])
    tl.store($(candidate_vto_copy_176_ptr) + r421_z, r421_value)
    r422_row = tl.arange($(4))
    r422_col = tl.arange($(16))
    r422_lane = tl.arange($(192))
    for r422_group in range($(8)) {
      r422_left = tl.load($(candidate_vto_copy_172_ptr) + (((((((r422_group % $(8)) * $(768)) + (r422_row[:, None] * $(192)) + r422_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r422_group % $(8)) * $(768)) + (r422_row[:, None] * $(192)) + r422_lane[None, :]) / $(192)) % $(4)) * $(192)) + ((((r422_group % $(8)) * $(768)) + (r422_row[:, None] * $(192)) + r422_lane[None, :]) % $(192))), dtype=tl.float32)
      r422_right = tl.load($(candidate_vto_copy_173_ptr) + (((((((r422_group % $(8)) * $(3072)) + (r422_lane[:, None] * $(16)) + r422_col[None, :]) / $(3072)) % $(8)) * $(3072)) + (((((r422_group % $(8)) * $(3072)) + (r422_lane[:, None] * $(16)) + r422_col[None, :]) % $(16)) * $(192)) + (((((r422_group % $(8)) * $(3072)) + (r422_lane[:, None] * $(16)) + r422_col[None, :]) / $(16)) % $(192))), dtype=tl.float32)
      r422_value = tl.dot(r422_left, r422_right)
      r422_value = r422_value.to(tl.float32)
      tl.store($(candidate_vbmm_89_ptr) + ((r422_group * $(64)) + (r422_row[:, None] * $(16)) + r422_col[None, :]), r422_value)
    }
    r423_z = tl.arange($(512))
    r423_value = (tl.load($(candidate_vbmm_89_ptr) + ((((r423_z / $(64)) % $(8)) * $(64)) + (((r423_z / $(16)) % $(4)) * $(16)) + (r423_z % $(16))), dtype=tl.float32) * 0.07216878364870322)
    r423_value = r423_value.to(tl.float32)
    r423_value = tl.broadcast_to(r423_value, [$(512)])
    tl.store($(candidate_vmul_79_ptr) + r423_z, r423_value)
    r424_z = tl.arange($(4))
    r424_value = (tl.cast(r424_z, tl.int64) + tl.full([], $(16), dtype=tl.int64))
    r424_value = tl.broadcast_to(r424_value, [$(4)])
    tl.store($(candidate_vadd_54_ptr) + r424_z, r424_value)
    r425_z = tl.arange($(4))
    r425_value = (tl.load($(candidate_vadd_54_ptr) + r425_z, dtype=tl.int64) - tl.full([], $(4), dtype=tl.int64))
    r425_value = tl.broadcast_to(r425_value, [$(4)])
    tl.store($(candidate_vsub_51_ptr) + r425_z, r425_value)
    r426_z = tl.arange($(64))
    r426_value = (tl.cast((r426_z % $(16)), tl.int64) > tl.load($(candidate_vsub_51_ptr) + ((r426_z / $(16)) % $(4)), dtype=tl.int64))
    r426_value = tl.broadcast_to(r426_value, [$(64)])
    tl.store($(candidate_vgt_28_ptr) + r426_z, r426_value)
    r427_z = tl.arange($(512))
    r427_value = tl.where(tl.load($(candidate_vgt_28_ptr) + ((((r427_z / $(16)) % $(4)) * $(16)) + (r427_z % $(16))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_79_ptr) + ((((r427_z / $(64)) % $(8)) * $(64)) + (((r427_z / $(16)) % $(4)) * $(16)) + (r427_z % $(16))), dtype=tl.float32))
    r427_value = r427_value.to(tl.float32)
    r427_value = tl.broadcast_to(r427_value, [$(512)])
    tl.store($(candidate_vmasked_fill_28_ptr) + r427_z, r427_value)
    r428_z = tl.arange($(512))
    r428_value = (tl.load($(candidate_vmasked_fill_28_ptr) + ((((r428_z / $(64)) % $(8)) * $(64)) + (((r428_z / $(16)) % $(4)) * $(16)) + (r428_z % $(16))), dtype=tl.float32) - tl.load($(candidate_vcat_42_ptr) + ((((r428_z / $(16)) % $(4)) * $(8)) + ((r428_z / $(64)) % $(8)) + $(96)), dtype=tl.float32))
    r428_value = r428_value.to(tl.float32)
    r428_value = tl.broadcast_to(r428_value, [$(512)])
    tl.store($(candidate_vsub_52_ptr) + r428_z, r428_value)
    r429_z = tl.arange($(512))
    r429_value = tl.exp(tl.load($(candidate_vsub_52_ptr) + ((((r429_z / $(64)) % $(8)) * $(64)) + (((r429_z / $(16)) % $(4)) * $(16)) + (r429_z % $(16))), dtype=tl.float32))
    r429_value = r429_value.to(tl.float32)
    r429_value = tl.broadcast_to(r429_value, [$(512)])
    tl.store($(candidate_vexp_7_ptr) + r429_z, r429_value)
    r430_z = tl.arange($(4096))
    r430_value = (tl.load($(candidate_vto_copy_175_ptr) + ((((r430_z / $(512)) % $(8)) * $(512)) + (((r430_z / $(128)) % $(4)) * $(128)) + (r430_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_176_ptr) + ((((r430_z / $(512)) % $(8)) * $(512)) + (((r430_z / $(128)) % $(4)) * $(128)) + (r430_z % $(128))), dtype=tl.float32))
    r430_value = r430_value.to(tl.float32)
    r430_value = tl.broadcast_to(r430_value, [$(4096)])
    tl.store($(candidate_vmul_80_ptr) + r430_z, r430_value)
    r431_z = tl.arange($(32))
    r431_lane = tl.arange($(128))
    r431_value = tl.sum(tl.load($(candidate_vmul_80_ptr) + ((((r431_z[:, None] / $(4)) % $(8)) * $(512)) + ((r431_z[:, None] % $(4)) * $(128)) + r431_lane[None, :]), dtype=tl.float32), axis=1)
    r431_value = r431_value.to(tl.float32)
    r431_value = tl.broadcast_to(r431_value, [$(32)])
    tl.store($(candidate_vsum_10_ptr) + r431_z, r431_value)
    r432_row = tl.arange($(4))
    r432_col = tl.arange($(16))
    r432_lane = tl.arange($(128))
    for r432_group in range($(8)) {
      r432_left = tl.load($(candidate_vto_copy_175_ptr) + (((((((r432_group % $(8)) * $(512)) + (r432_row[:, None] * $(128)) + r432_lane[None, :]) / $(512)) % $(8)) * $(512)) + ((((((r432_group % $(8)) * $(512)) + (r432_row[:, None] * $(128)) + r432_lane[None, :]) / $(128)) % $(4)) * $(128)) + ((((r432_group % $(8)) * $(512)) + (r432_row[:, None] * $(128)) + r432_lane[None, :]) % $(128))), dtype=tl.float32)
      r432_right = tl.load($(candidate_vto_copy_174_ptr) + (((((((r432_group % $(8)) * $(2048)) + (r432_lane[:, None] * $(16)) + r432_col[None, :]) / $(2048)) % $(8)) * $(2048)) + (((((r432_group % $(8)) * $(2048)) + (r432_lane[:, None] * $(16)) + r432_col[None, :]) % $(16)) * $(128)) + (((((r432_group % $(8)) * $(2048)) + (r432_lane[:, None] * $(16)) + r432_col[None, :]) / $(16)) % $(128))), dtype=tl.float32)
      r432_value = tl.dot(r432_left, r432_right)
      r432_value = r432_value.to(tl.float32)
      tl.store($(candidate_vbmm_90_ptr) + ((r432_group * $(64)) + (r432_row[:, None] * $(16)) + r432_col[None, :]), r432_value)
    }
    r433_z = tl.arange($(512))
    r433_value = (tl.load($(candidate_vbmm_90_ptr) + ((((r433_z / $(64)) % $(8)) * $(64)) + (((r433_z / $(16)) % $(4)) * $(16)) + (r433_z % $(16))), dtype=tl.float32) - tl.load($(candidate_vsum_10_ptr) + ((((r433_z / $(64)) % $(8)) * $(4)) + ((r433_z / $(16)) % $(4))), dtype=tl.float32))
    r433_value = r433_value.to(tl.float32)
    r433_value = tl.broadcast_to(r433_value, [$(512)])
    tl.store($(candidate_vsub_53_ptr) + r433_z, r433_value)
    r434_z = tl.arange($(512))
    r434_value = (tl.load($(candidate_vexp_7_ptr) + ((((r434_z / $(64)) % $(8)) * $(64)) + (((r434_z / $(16)) % $(4)) * $(16)) + (r434_z % $(16))), dtype=tl.float32) * tl.load($(candidate_vsub_53_ptr) + ((((r434_z / $(64)) % $(8)) * $(64)) + (((r434_z / $(16)) % $(4)) * $(16)) + (r434_z % $(16))), dtype=tl.float32))
    r434_value = r434_value.to(tl.float32)
    r434_value = tl.broadcast_to(r434_value, [$(512)])
    tl.store($(candidate_vmul_81_ptr) + r434_z, r434_value)
    r435_row = tl.arange($(4))
    r435_col = tl.arange($(192))
    r435_lane = tl.arange($(16))
    for r435_group in range($(8)) {
      r435_left = tl.load($(candidate_vmul_81_ptr) + (((((((r435_group % $(8)) * $(64)) + (r435_row[:, None] * $(16)) + r435_lane[None, :]) / $(64)) % $(8)) * $(64)) + ((((((r435_group % $(8)) * $(64)) + (r435_row[:, None] * $(16)) + r435_lane[None, :]) / $(16)) % $(4)) * $(16)) + ((((r435_group % $(8)) * $(64)) + (r435_row[:, None] * $(16)) + r435_lane[None, :]) % $(16))), dtype=tl.float32)
      r435_right = tl.load($(candidate_vto_copy_173_ptr) + (((((((r435_group % $(8)) * $(3072)) + (r435_lane[:, None] * $(192)) + r435_col[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r435_group % $(8)) * $(3072)) + (r435_lane[:, None] * $(192)) + r435_col[None, :]) / $(192)) % $(16)) * $(192)) + ((((r435_group % $(8)) * $(3072)) + (r435_lane[:, None] * $(192)) + r435_col[None, :]) % $(192))), dtype=tl.float32)
      r435_value = tl.dot(r435_left, r435_right)
      r435_value = r435_value.to(tl.float32)
      tl.store($(candidate_vbmm_91_ptr) + ((r435_group * $(768)) + (r435_row[:, None] * $(192)) + r435_col[None, :]), r435_value)
    }
    r436_z = tl.arange($(6144))
    r436_value = (tl.load($(candidate_vbmm_91_ptr) + ((((r436_z / $(768)) % $(8)) * $(768)) + (((r436_z / $(192)) % $(4)) * $(192)) + (r436_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r436_value = r436_value.to(tl.float32)
    r436_value = tl.broadcast_to(r436_value, [$(6144)])
    tl.store($(candidate_vmul_82_ptr) + r436_z, r436_value)
    r437_z = tl.arange($(6144))
    r437_value = tl.load($(candidate_vmul_82_ptr) + ((((r437_z / $(192)) % $(8)) * $(768)) + (((r437_z / $(1536)) % $(4)) * $(192)) + (r437_z % $(192))), dtype=tl.float32)
    r437_value = r437_value.to(tl.bfloat16)
    r437_value = tl.broadcast_to(r437_value, [$(6144)])
    tl.store($(candidate_vto_copy_177_ptr) + r437_z, r437_value)
    r438_row = tl.arange($(16))
    r438_col = tl.arange($(192))
    r438_lane = tl.arange($(4))
    for r438_group in range($(8)) {
      r438_left = tl.load($(candidate_vmul_81_ptr) + (((((((r438_group % $(8)) * $(64)) + (r438_row[:, None] * $(4)) + r438_lane[None, :]) / $(64)) % $(8)) * $(64)) + (((((r438_group % $(8)) * $(64)) + (r438_row[:, None] * $(4)) + r438_lane[None, :]) % $(4)) * $(16)) + (((((r438_group % $(8)) * $(64)) + (r438_row[:, None] * $(4)) + r438_lane[None, :]) / $(4)) % $(16))), dtype=tl.float32)
      r438_right = tl.load($(candidate_vto_copy_172_ptr) + (((((((r438_group % $(8)) * $(768)) + (r438_lane[:, None] * $(192)) + r438_col[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r438_group % $(8)) * $(768)) + (r438_lane[:, None] * $(192)) + r438_col[None, :]) / $(192)) % $(4)) * $(192)) + ((((r438_group % $(8)) * $(768)) + (r438_lane[:, None] * $(192)) + r438_col[None, :]) % $(192))), dtype=tl.float32)
      r438_value = tl.dot(r438_left, r438_right)
      r438_value = r438_value.to(tl.float32)
      tl.store($(candidate_vbmm_92_ptr) + ((r438_group * $(3072)) + (r438_row[:, None] * $(192)) + r438_col[None, :]), r438_value)
    }
    r439_z = tl.arange($(24576))
    r439_value = (tl.load($(candidate_vbmm_92_ptr) + ((((r439_z / $(3072)) % $(8)) * $(3072)) + (((r439_z / $(192)) % $(16)) * $(192)) + (r439_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r439_value = r439_value.to(tl.float32)
    r439_value = tl.broadcast_to(r439_value, [$(24576)])
    tl.store($(candidate_vmul_83_ptr) + r439_z, r439_value)
    r440_z = tl.arange($(24576))
    r440_value = tl.load($(candidate_vmul_83_ptr) + ((((r440_z / $(192)) % $(8)) * $(3072)) + (((r440_z / $(1536)) % $(16)) * $(192)) + (r440_z % $(192))), dtype=tl.float32)
    r440_value = r440_value.to(tl.bfloat16)
    r440_value = tl.broadcast_to(r440_value, [$(24576)])
    tl.store($(candidate_vto_copy_178_ptr) + r440_z, r440_value)
    r441_row = tl.arange($(16))
    r441_col = tl.arange($(128))
    r441_lane = tl.arange($(4))
    for r441_group in range($(8)) {
      r441_left = tl.load($(candidate_vexp_7_ptr) + (((((((r441_group % $(8)) * $(64)) + (r441_row[:, None] * $(4)) + r441_lane[None, :]) / $(64)) % $(8)) * $(64)) + (((((r441_group % $(8)) * $(64)) + (r441_row[:, None] * $(4)) + r441_lane[None, :]) % $(4)) * $(16)) + (((((r441_group % $(8)) * $(64)) + (r441_row[:, None] * $(4)) + r441_lane[None, :]) / $(4)) % $(16))), dtype=tl.float32)
      r441_right = tl.load($(candidate_vto_copy_175_ptr) + (((((((r441_group % $(8)) * $(512)) + (r441_lane[:, None] * $(128)) + r441_col[None, :]) / $(512)) % $(8)) * $(512)) + ((((((r441_group % $(8)) * $(512)) + (r441_lane[:, None] * $(128)) + r441_col[None, :]) / $(128)) % $(4)) * $(128)) + ((((r441_group % $(8)) * $(512)) + (r441_lane[:, None] * $(128)) + r441_col[None, :]) % $(128))), dtype=tl.float32)
      r441_value = tl.dot(r441_left, r441_right)
      r441_value = r441_value.to(tl.float32)
      tl.store($(candidate_vbmm_93_ptr) + ((r441_group * $(2048)) + (r441_row[:, None] * $(128)) + r441_col[None, :]), r441_value)
    }
    r442_z = tl.arange($(16384))
    r442_value = tl.load($(candidate_vbmm_93_ptr) + ((((r442_z / $(128)) % $(8)) * $(2048)) + (((r442_z / $(1024)) % $(16)) * $(128)) + (r442_z % $(128))), dtype=tl.float32)
    r442_value = r442_value.to(tl.bfloat16)
    r442_value = tl.broadcast_to(r442_value, [$(16384)])
    tl.store($(candidate_vto_copy_179_ptr) + r442_z, r442_value)
    r443_z = tl.arange($(24576))
    r443_value = tl.load($(candidate_vcat_39_ptr) + ((((r443_z / $(192)) % $(16)) * $(1536)) + (((r443_z / $(3072)) % $(8)) * $(192)) + (r443_z % $(192)) + $(24576)), dtype=tl.bfloat16)
    r443_value = r443_value.to(tl.float32)
    r443_value = tl.broadcast_to(r443_value, [$(24576)])
    tl.store($(candidate_vto_copy_180_ptr) + r443_z, r443_value)
    r444_z = tl.arange($(98304))
    r444_value = tl.load($(candidate_vcat_40_ptr) + ((((r444_z / $(192)) % $(64)) * $(1536)) + (((r444_z / $(12288)) % $(8)) * $(192)) + (r444_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r444_value = r444_value.to(tl.float32)
    r444_value = tl.broadcast_to(r444_value, [$(98304)])
    tl.store($(candidate_vto_copy_181_ptr) + r444_z, r444_value)
    r445_z = tl.arange($(65536))
    r445_value = tl.load($(candidate_vbmm_77_ptr) + ((((r445_z / $(128)) % $(64)) * $(1024)) + (((r445_z / $(8192)) % $(8)) * $(128)) + (r445_z % $(128)) + $(65536)), dtype=tl.bfloat16)
    r445_value = r445_value.to(tl.float32)
    r445_value = tl.broadcast_to(r445_value, [$(65536)])
    tl.store($(candidate_vto_copy_182_ptr) + r445_z, r445_value)
    r446_z = tl.arange($(16384))
    r446_value = tl.load($(candidate_vmm_28_ptr) + ((((r446_z / $(128)) % $(16)) * $(1024)) + (((r446_z / $(2048)) % $(8)) * $(128)) + (r446_z % $(128)) + $(49152)), dtype=tl.bfloat16)
    r446_value = r446_value.to(tl.float32)
    r446_value = tl.broadcast_to(r446_value, [$(16384)])
    tl.store($(candidate_vto_copy_183_ptr) + r446_z, r446_value)
    r447_z = tl.arange($(16384))
    r447_value = tl.load($(candidate_vcat_41_ptr) + ((((r447_z / $(128)) % $(16)) * $(1024)) + (((r447_z / $(2048)) % $(8)) * $(128)) + (r447_z % $(128)) + $(16384)), dtype=tl.bfloat16)
    r447_value = r447_value.to(tl.float32)
    r447_value = tl.broadcast_to(r447_value, [$(16384)])
    tl.store($(candidate_vto_copy_184_ptr) + r447_z, r447_value)
    r448_row = tl.arange($(16))
    r448_col = tl.arange($(64))
    r448_lane = tl.arange($(192))
    for r448_group in range($(8)) {
      r448_left = tl.load($(candidate_vto_copy_180_ptr) + (((((((r448_group % $(8)) * $(3072)) + (r448_row[:, None] * $(192)) + r448_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r448_group % $(8)) * $(3072)) + (r448_row[:, None] * $(192)) + r448_lane[None, :]) / $(192)) % $(16)) * $(192)) + ((((r448_group % $(8)) * $(3072)) + (r448_row[:, None] * $(192)) + r448_lane[None, :]) % $(192))), dtype=tl.float32)
      r448_right = tl.load($(candidate_vto_copy_181_ptr) + (((((((r448_group % $(8)) * $(12288)) + (r448_lane[:, None] * $(64)) + r448_col[None, :]) / $(12288)) % $(8)) * $(12288)) + (((((r448_group % $(8)) * $(12288)) + (r448_lane[:, None] * $(64)) + r448_col[None, :]) % $(64)) * $(192)) + (((((r448_group % $(8)) * $(12288)) + (r448_lane[:, None] * $(64)) + r448_col[None, :]) / $(64)) % $(192))), dtype=tl.float32)
      r448_value = tl.dot(r448_left, r448_right)
      r448_value = r448_value.to(tl.float32)
      tl.store($(candidate_vbmm_94_ptr) + ((r448_group * $(1024)) + (r448_row[:, None] * $(64)) + r448_col[None, :]), r448_value)
    }
    r449_z = tl.arange($(8192))
    r449_value = (tl.load($(candidate_vbmm_94_ptr) + ((((r449_z / $(1024)) % $(8)) * $(1024)) + (((r449_z / $(64)) % $(16)) * $(64)) + (r449_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r449_value = r449_value.to(tl.float32)
    r449_value = tl.broadcast_to(r449_value, [$(8192)])
    tl.store($(candidate_vmul_84_ptr) + r449_z, r449_value)
    r450_z = tl.arange($(16))
    r450_value = (tl.cast(r450_z, tl.int64) + tl.full([], $(64), dtype=tl.int64))
    r450_value = tl.broadcast_to(r450_value, [$(16)])
    tl.store($(candidate_vadd_55_ptr) + r450_z, r450_value)
    r451_z = tl.arange($(16))
    r451_value = (tl.load($(candidate_vadd_55_ptr) + r451_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r451_value = tl.broadcast_to(r451_value, [$(16)])
    tl.store($(candidate_vsub_54_ptr) + r451_z, r451_value)
    r452_z = tl.arange($(1024))
    r452_value = (tl.cast((r452_z % $(64)), tl.int64) > tl.load($(candidate_vsub_54_ptr) + ((r452_z / $(64)) % $(16)), dtype=tl.int64))
    r452_value = tl.broadcast_to(r452_value, [$(1024)])
    tl.store($(candidate_vgt_29_ptr) + r452_z, r452_value)
    r453_z = tl.arange($(8192))
    r453_value = tl.where(tl.load($(candidate_vgt_29_ptr) + ((((r453_z / $(64)) % $(16)) * $(64)) + (r453_z % $(64))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_84_ptr) + ((((r453_z / $(1024)) % $(8)) * $(1024)) + (((r453_z / $(64)) % $(16)) * $(64)) + (r453_z % $(64))), dtype=tl.float32))
    r453_value = r453_value.to(tl.float32)
    r453_value = tl.broadcast_to(r453_value, [$(8192)])
    tl.store($(candidate_vmasked_fill_29_ptr) + r453_z, r453_value)
    r454_z = tl.arange($(8192))
    r454_value = (tl.load($(candidate_vmasked_fill_29_ptr) + ((((r454_z / $(1024)) % $(8)) * $(1024)) + (((r454_z / $(64)) % $(16)) * $(64)) + (r454_z % $(64))), dtype=tl.float32) - tl.load($(candidate_vcat_42_ptr) + ((((r454_z / $(64)) % $(16)) * $(8)) + ((r454_z / $(1024)) % $(8)) + $(128)), dtype=tl.float32))
    r454_value = r454_value.to(tl.float32)
    r454_value = tl.broadcast_to(r454_value, [$(8192)])
    tl.store($(candidate_vsub_55_ptr) + r454_z, r454_value)
    r455_z = tl.arange($(8192))
    r455_value = tl.exp(tl.load($(candidate_vsub_55_ptr) + ((((r455_z / $(1024)) % $(8)) * $(1024)) + (((r455_z / $(64)) % $(16)) * $(64)) + (r455_z % $(64))), dtype=tl.float32))
    r455_value = r455_value.to(tl.float32)
    r455_value = tl.broadcast_to(r455_value, [$(8192)])
    tl.store($(candidate_vexp_8_ptr) + r455_z, r455_value)
    r456_z = tl.arange($(16384))
    r456_value = (tl.load($(candidate_vto_copy_183_ptr) + ((((r456_z / $(2048)) % $(8)) * $(2048)) + (((r456_z / $(128)) % $(16)) * $(128)) + (r456_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_184_ptr) + ((((r456_z / $(2048)) % $(8)) * $(2048)) + (((r456_z / $(128)) % $(16)) * $(128)) + (r456_z % $(128))), dtype=tl.float32))
    r456_value = r456_value.to(tl.float32)
    r456_value = tl.broadcast_to(r456_value, [$(16384)])
    tl.store($(candidate_vmul_85_ptr) + r456_z, r456_value)
    r457_z = tl.arange($(128))
    r457_lane = tl.arange($(128))
    r457_value = tl.sum(tl.load($(candidate_vmul_85_ptr) + ((((r457_z[:, None] / $(16)) % $(8)) * $(2048)) + ((r457_z[:, None] % $(16)) * $(128)) + r457_lane[None, :]), dtype=tl.float32), axis=1)
    r457_value = r457_value.to(tl.float32)
    r457_value = tl.broadcast_to(r457_value, [$(128)])
    tl.store($(candidate_vsum_11_ptr) + r457_z, r457_value)
    r458_row = tl.arange($(16))
    r458_col = tl.arange($(64))
    r458_lane = tl.arange($(128))
    for r458_group in range($(8)) {
      r458_left = tl.load($(candidate_vto_copy_183_ptr) + (((((((r458_group % $(8)) * $(2048)) + (r458_row[:, None] * $(128)) + r458_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r458_group % $(8)) * $(2048)) + (r458_row[:, None] * $(128)) + r458_lane[None, :]) / $(128)) % $(16)) * $(128)) + ((((r458_group % $(8)) * $(2048)) + (r458_row[:, None] * $(128)) + r458_lane[None, :]) % $(128))), dtype=tl.float32)
      r458_right = tl.load($(candidate_vto_copy_182_ptr) + (((((((r458_group % $(8)) * $(8192)) + (r458_lane[:, None] * $(64)) + r458_col[None, :]) / $(8192)) % $(8)) * $(8192)) + (((((r458_group % $(8)) * $(8192)) + (r458_lane[:, None] * $(64)) + r458_col[None, :]) % $(64)) * $(128)) + (((((r458_group % $(8)) * $(8192)) + (r458_lane[:, None] * $(64)) + r458_col[None, :]) / $(64)) % $(128))), dtype=tl.float32)
      r458_value = tl.dot(r458_left, r458_right)
      r458_value = r458_value.to(tl.float32)
      tl.store($(candidate_vbmm_95_ptr) + ((r458_group * $(1024)) + (r458_row[:, None] * $(64)) + r458_col[None, :]), r458_value)
    }
    r459_z = tl.arange($(8192))
    r459_value = (tl.load($(candidate_vbmm_95_ptr) + ((((r459_z / $(1024)) % $(8)) * $(1024)) + (((r459_z / $(64)) % $(16)) * $(64)) + (r459_z % $(64))), dtype=tl.float32) - tl.load($(candidate_vsum_11_ptr) + ((((r459_z / $(1024)) % $(8)) * $(16)) + ((r459_z / $(64)) % $(16))), dtype=tl.float32))
    r459_value = r459_value.to(tl.float32)
    r459_value = tl.broadcast_to(r459_value, [$(8192)])
    tl.store($(candidate_vsub_56_ptr) + r459_z, r459_value)
    r460_z = tl.arange($(8192))
    r460_value = (tl.load($(candidate_vexp_8_ptr) + ((((r460_z / $(1024)) % $(8)) * $(1024)) + (((r460_z / $(64)) % $(16)) * $(64)) + (r460_z % $(64))), dtype=tl.float32) * tl.load($(candidate_vsub_56_ptr) + ((((r460_z / $(1024)) % $(8)) * $(1024)) + (((r460_z / $(64)) % $(16)) * $(64)) + (r460_z % $(64))), dtype=tl.float32))
    r460_value = r460_value.to(tl.float32)
    r460_value = tl.broadcast_to(r460_value, [$(8192)])
    tl.store($(candidate_vmul_86_ptr) + r460_z, r460_value)
    r461_row = tl.arange($(16))
    r461_col = tl.arange($(192))
    r461_lane = tl.arange($(64))
    for r461_group in range($(8)) {
      r461_left = tl.load($(candidate_vmul_86_ptr) + (((((((r461_group % $(8)) * $(1024)) + (r461_row[:, None] * $(64)) + r461_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + ((((((r461_group % $(8)) * $(1024)) + (r461_row[:, None] * $(64)) + r461_lane[None, :]) / $(64)) % $(16)) * $(64)) + ((((r461_group % $(8)) * $(1024)) + (r461_row[:, None] * $(64)) + r461_lane[None, :]) % $(64))), dtype=tl.float32)
      r461_right = tl.load($(candidate_vto_copy_181_ptr) + (((((((r461_group % $(8)) * $(12288)) + (r461_lane[:, None] * $(192)) + r461_col[None, :]) / $(12288)) % $(8)) * $(12288)) + ((((((r461_group % $(8)) * $(12288)) + (r461_lane[:, None] * $(192)) + r461_col[None, :]) / $(192)) % $(64)) * $(192)) + ((((r461_group % $(8)) * $(12288)) + (r461_lane[:, None] * $(192)) + r461_col[None, :]) % $(192))), dtype=tl.float32)
      r461_value = tl.dot(r461_left, r461_right)
      r461_value = r461_value.to(tl.float32)
      tl.store($(candidate_vbmm_96_ptr) + ((r461_group * $(3072)) + (r461_row[:, None] * $(192)) + r461_col[None, :]), r461_value)
    }
    r462_z = tl.arange($(24576))
    r462_value = (tl.load($(candidate_vbmm_96_ptr) + ((((r462_z / $(3072)) % $(8)) * $(3072)) + (((r462_z / $(192)) % $(16)) * $(192)) + (r462_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r462_value = r462_value.to(tl.float32)
    r462_value = tl.broadcast_to(r462_value, [$(24576)])
    tl.store($(candidate_vmul_87_ptr) + r462_z, r462_value)
    r463_z = tl.arange($(24576))
    r463_value = tl.load($(candidate_vmul_87_ptr) + ((((r463_z / $(192)) % $(8)) * $(3072)) + (((r463_z / $(1536)) % $(16)) * $(192)) + (r463_z % $(192))), dtype=tl.float32)
    r463_value = r463_value.to(tl.bfloat16)
    r463_value = tl.broadcast_to(r463_value, [$(24576)])
    tl.store($(candidate_vto_copy_185_ptr) + r463_z, r463_value)
    r464_row = tl.arange($(64))
    r464_col = tl.arange($(192))
    r464_lane = tl.arange($(16))
    for r464_group in range($(8)) {
      r464_left = tl.load($(candidate_vmul_86_ptr) + (((((((r464_group % $(8)) * $(1024)) + (r464_row[:, None] * $(16)) + r464_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + (((((r464_group % $(8)) * $(1024)) + (r464_row[:, None] * $(16)) + r464_lane[None, :]) % $(16)) * $(64)) + (((((r464_group % $(8)) * $(1024)) + (r464_row[:, None] * $(16)) + r464_lane[None, :]) / $(16)) % $(64))), dtype=tl.float32)
      r464_right = tl.load($(candidate_vto_copy_180_ptr) + (((((((r464_group % $(8)) * $(3072)) + (r464_lane[:, None] * $(192)) + r464_col[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r464_group % $(8)) * $(3072)) + (r464_lane[:, None] * $(192)) + r464_col[None, :]) / $(192)) % $(16)) * $(192)) + ((((r464_group % $(8)) * $(3072)) + (r464_lane[:, None] * $(192)) + r464_col[None, :]) % $(192))), dtype=tl.float32)
      r464_value = tl.dot(r464_left, r464_right)
      r464_value = r464_value.to(tl.float32)
      tl.store($(candidate_vbmm_97_ptr) + ((r464_group * $(12288)) + (r464_row[:, None] * $(192)) + r464_col[None, :]), r464_value)
    }
    r465_z = tl.arange($(98304))
    r465_value = (tl.load($(candidate_vbmm_97_ptr) + ((((r465_z / $(12288)) % $(8)) * $(12288)) + (((r465_z / $(192)) % $(64)) * $(192)) + (r465_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r465_value = r465_value.to(tl.float32)
    r465_value = tl.broadcast_to(r465_value, [$(98304)])
    tl.store($(candidate_vmul_88_ptr) + r465_z, r465_value)
    r466_z = tl.arange($(98304))
    r466_value = tl.load($(candidate_vmul_88_ptr) + ((((r466_z / $(192)) % $(8)) * $(12288)) + (((r466_z / $(1536)) % $(64)) * $(192)) + (r466_z % $(192))), dtype=tl.float32)
    r466_value = r466_value.to(tl.bfloat16)
    r466_value = tl.broadcast_to(r466_value, [$(98304)])
    tl.store($(candidate_vto_copy_186_ptr) + r466_z, r466_value)
    r467_row = tl.arange($(64))
    r467_col = tl.arange($(128))
    r467_lane = tl.arange($(16))
    for r467_group in range($(8)) {
      r467_left = tl.load($(candidate_vexp_8_ptr) + (((((((r467_group % $(8)) * $(1024)) + (r467_row[:, None] * $(16)) + r467_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + (((((r467_group % $(8)) * $(1024)) + (r467_row[:, None] * $(16)) + r467_lane[None, :]) % $(16)) * $(64)) + (((((r467_group % $(8)) * $(1024)) + (r467_row[:, None] * $(16)) + r467_lane[None, :]) / $(16)) % $(64))), dtype=tl.float32)
      r467_right = tl.load($(candidate_vto_copy_183_ptr) + (((((((r467_group % $(8)) * $(2048)) + (r467_lane[:, None] * $(128)) + r467_col[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r467_group % $(8)) * $(2048)) + (r467_lane[:, None] * $(128)) + r467_col[None, :]) / $(128)) % $(16)) * $(128)) + ((((r467_group % $(8)) * $(2048)) + (r467_lane[:, None] * $(128)) + r467_col[None, :]) % $(128))), dtype=tl.float32)
      r467_value = tl.dot(r467_left, r467_right)
      r467_value = r467_value.to(tl.float32)
      tl.store($(candidate_vbmm_98_ptr) + ((r467_group * $(8192)) + (r467_row[:, None] * $(128)) + r467_col[None, :]), r467_value)
    }
    r468_z = tl.arange($(65536))
    r468_value = tl.load($(candidate_vbmm_98_ptr) + ((((r468_z / $(128)) % $(8)) * $(8192)) + (((r468_z / $(1024)) % $(64)) * $(128)) + (r468_z % $(128))), dtype=tl.float32)
    r468_value = r468_value.to(tl.bfloat16)
    r468_value = tl.broadcast_to(r468_value, [$(65536)])
    tl.store($(candidate_vto_copy_187_ptr) + r468_z, r468_value)
    r469_z = tl.arange($(49152))
    r469_value = tl.where(((r469_z / $(1536)) % $(32)) < $(12), tl.load($(candidate_vto_copy_169_ptr) + ((((((r469_z / $(1536)) % $(32)) - $(0)) % $(12)) * $(1536)) + (((r469_z / $(192)) % $(8)) * $(192)) + (r469_z % $(192))), dtype=tl.bfloat16), tl.where(((r469_z / $(1536)) % $(32)) < $(16), tl.load($(candidate_vto_copy_177_ptr) + ((((((r469_z / $(1536)) % $(32)) - $(12)) % $(4)) * $(1536)) + (((r469_z / $(192)) % $(8)) * $(192)) + (r469_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vto_copy_185_ptr) + ((((((r469_z / $(1536)) % $(32)) - $(16)) % $(16)) * $(1536)) + (((r469_z / $(192)) % $(8)) * $(192)) + (r469_z % $(192))), dtype=tl.bfloat16)))
    r469_value = r469_value.to(tl.bfloat16)
    r469_value = tl.broadcast_to(r469_value, [$(49152)])
    tl.store($(candidate_vcat_43_ptr) + r469_z, r469_value)
    r470_z = tl.arange($(196608))
    r470_value = tl.where(((r470_z / $(1536)) % $(128)) < $(48), tl.load($(candidate_vto_copy_170_ptr) + ((((((r470_z / $(1536)) % $(128)) - $(0)) % $(48)) * $(1536)) + (((r470_z / $(192)) % $(8)) * $(192)) + (r470_z % $(192))), dtype=tl.bfloat16), tl.where(((r470_z / $(1536)) % $(128)) < $(64), tl.load($(candidate_vto_copy_178_ptr) + ((((((r470_z / $(1536)) % $(128)) - $(48)) % $(16)) * $(1536)) + (((r470_z / $(192)) % $(8)) * $(192)) + (r470_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vto_copy_186_ptr) + ((((((r470_z / $(1536)) % $(128)) - $(64)) % $(64)) * $(1536)) + (((r470_z / $(192)) % $(8)) * $(192)) + (r470_z % $(192))), dtype=tl.bfloat16)))
    r470_value = r470_value.to(tl.bfloat16)
    r470_value = tl.broadcast_to(r470_value, [$(196608)])
    tl.store($(candidate_vcat_44_ptr) + r470_z, r470_value)
    r471_z = tl.arange($(131072))
    r471_value = tl.where(((r471_z / $(1024)) % $(128)) < $(48), tl.load($(candidate_vto_copy_171_ptr) + ((((((r471_z / $(1024)) % $(128)) - $(0)) % $(48)) * $(1024)) + (((r471_z / $(128)) % $(8)) * $(128)) + (r471_z % $(128))), dtype=tl.bfloat16), tl.where(((r471_z / $(1024)) % $(128)) < $(64), tl.load($(candidate_vto_copy_179_ptr) + ((((((r471_z / $(1024)) % $(128)) - $(48)) % $(16)) * $(1024)) + (((r471_z / $(128)) % $(8)) * $(128)) + (r471_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vto_copy_187_ptr) + ((((((r471_z / $(1024)) % $(128)) - $(64)) % $(64)) * $(1024)) + (((r471_z / $(128)) % $(8)) * $(128)) + (r471_z % $(128))), dtype=tl.bfloat16)))
    r471_value = r471_value.to(tl.bfloat16)
    r471_value = tl.broadcast_to(r471_value, [$(131072)])
    tl.store($(candidate_vcat_45_ptr) + r471_z, r471_value)
    r472_z = tl.arange($(8192))
    r472_lane = tl.arange($(8))
    r472_value = tl.sum(tl.load($(candidate_vcat_44_ptr) + ((((r472_z[:, None] / $(64)) % $(128)) * $(1536)) + (r472_lane[None, :] * $(192)) + (r472_z[:, None] % $(64)) + $(128)), dtype=tl.bfloat16), axis=1)
    r472_value = r472_value.to(tl.bfloat16)
    r472_value = tl.broadcast_to(r472_value, [$(8192)])
    tl.store($(candidate_vsum_12_ptr) + r472_z, r472_value)
    r473_row = tl.arange($(128))
    r473_col = tl.arange($(512))
    r473_lane = tl.arange($(1024))
    for r473_group in range($(1)) {
      r473_left = tl.load($(candidate_vcat_44_ptr) + ((((((r473_row[:, None] * $(1024)) + r473_lane[None, :]) / $(1024)) % $(128)) * $(1536)) + ((((r473_row[:, None] * $(1024)) + r473_lane[None, :]) % $(8)) * $(192)) + ((((r473_row[:, None] * $(1024)) + r473_lane[None, :]) / $(8)) % $(128))), dtype=tl.bfloat16)
      r473_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r473_lane[:, None] * $(512)) + r473_col[None, :]) / $(512)) % $(8)) * $(131072)) + (((((r473_lane[:, None] * $(512)) + r473_col[None, :]) / $(4096)) % $(128)) * $(512)) + (((r473_lane[:, None] * $(512)) + r473_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r473_value = tl.dot(r473_left, r473_right)
      r473_value = r473_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_99_ptr) + ((r473_group * $(65536)) + (r473_row[:, None] * $(512)) + r473_col[None, :]), r473_value)
    }
    r474_row = tl.arange($(128))
    r474_col = tl.arange($(512))
    r474_lane = tl.arange($(1024))
    for r474_group in range($(1)) {
      r474_left = tl.load($(candidate_vcat_45_ptr) + ((((((r474_row[:, None] * $(1024)) + r474_lane[None, :]) / $(1024)) % $(128)) * $(1024)) + ((((r474_row[:, None] * $(1024)) + r474_lane[None, :]) % $(8)) * $(128)) + ((((r474_row[:, None] * $(1024)) + r474_lane[None, :]) / $(8)) % $(128))), dtype=tl.bfloat16)
      r474_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r474_lane[:, None] * $(512)) + r474_col[None, :]) / $(512)) % $(8)) * $(131072)) + (((((r474_lane[:, None] * $(512)) + r474_col[None, :]) / $(4096)) % $(128)) * $(512)) + (((r474_lane[:, None] * $(512)) + r474_col[None, :]) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r474_value = tl.dot(r474_left, r474_right)
      r474_value = r474_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_100_ptr) + ((r474_group * $(65536)) + (r474_row[:, None] * $(512)) + r474_col[None, :]), r474_value)
    }
    r475_z = tl.arange($(65536))
    r475_value = (tl.load($(candidate_vbmm_99_ptr) + ((((r475_z / $(512)) % $(128)) * $(512)) + (r475_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vbmm_100_ptr) + ((((r475_z / $(512)) % $(128)) * $(512)) + (r475_z % $(512))), dtype=tl.bfloat16))
    r475_value = r475_value.to(tl.bfloat16)
    r475_value = tl.broadcast_to(r475_value, [$(65536)])
    tl.store($(candidate_vadd_56_ptr) + r475_z, r475_value)
    r476_z = tl.arange($(65536))
    r476_value = tl.where(((((r476_z / $(512)) % $(128)) >= $(0)) & (((r476_z / $(512)) % $(128)) < $(128)) & ((((r476_z / $(512)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vadd_56_ptr) + ((((((r476_z / $(512)) % $(128)) - $(0)) % $(128)) * $(512)) + (r476_z % $(512))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r476_value = r476_value.to(tl.bfloat16)
    r476_value = tl.broadcast_to(r476_value, [$(65536)])
    tl.store($(candidate_vslice_backward_8_ptr) + r476_z, r476_value)
    r477_z = tl.arange($(65536))
    r477_value = (tl.load($(candidate_vadd_46_ptr) + ((((r477_z / $(512)) % $(128)) * $(512)) + (r477_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_8_ptr) + ((((r477_z / $(512)) % $(128)) * $(512)) + (r477_z % $(512))), dtype=tl.bfloat16))
    r477_value = r477_value.to(tl.bfloat16)
    r477_value = tl.broadcast_to(r477_value, [$(65536)])
    tl.store($(candidate_vadd_57_ptr) + r477_z, r477_value)
    r478_z = tl.arange($(8192))
    r478_value = tl.where(((((r478_z / $(64)) % $(128)) >= $(0)) & (((r478_z / $(64)) % $(128)) < $(128)) & ((((r478_z / $(64)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vsum_12_ptr) + ((((((r478_z / $(64)) % $(128)) - $(0)) % $(128)) * $(64)) + (r478_z % $(64))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r478_value = r478_value.to(tl.bfloat16)
    r478_value = tl.broadcast_to(r478_value, [$(8192)])
    tl.store($(candidate_vslice_backward_9_ptr) + r478_z, r478_value)
    r479_z = tl.arange($(8192))
    r479_value = (tl.load($(candidate_vadd_47_ptr) + ((((r479_z / $(64)) % $(128)) * $(64)) + (r479_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_9_ptr) + ((((r479_z / $(64)) % $(128)) * $(64)) + (r479_z % $(64))), dtype=tl.bfloat16))
    r479_value = r479_value.to(tl.bfloat16)
    r479_value = tl.broadcast_to(r479_value, [$(8192)])
    tl.store($(candidate_vadd_58_ptr) + r479_z, r479_value)
    r480_z = tl.arange($(131072))
    r480_value = tl.where(((((r480_z / $(1024)) % $(128)) >= $(32)) & (((r480_z / $(1024)) % $(128)) < $(64)) & ((((r480_z / $(1024)) % $(128)) - $(32)) % $(1) == $(0))), tl.load($(candidate_vcat_43_ptr) + ((((((r480_z / $(1024)) % $(128)) - $(32)) % $(32)) * $(1536)) + (((r480_z / $(128)) % $(8)) * $(192)) + (r480_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r480_value = r480_value.to(tl.bfloat16)
    r480_value = tl.broadcast_to(r480_value, [$(131072)])
    tl.store($(candidate_vslice_backward_10_ptr) + r480_z, r480_value)
    r481_z = tl.arange($(131072))
    r481_value = (tl.load($(candidate_vadd_48_ptr) + ((((r481_z / $(1024)) % $(128)) * $(1024)) + (((r481_z / $(128)) % $(8)) * $(128)) + (r481_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_10_ptr) + ((((r481_z / $(1024)) % $(128)) * $(1024)) + (((r481_z / $(128)) % $(8)) * $(128)) + (r481_z % $(128))), dtype=tl.bfloat16))
    r481_value = r481_value.to(tl.bfloat16)
    r481_value = tl.broadcast_to(r481_value, [$(131072)])
    tl.store($(candidate_vadd_59_ptr) + r481_z, r481_value)
    r482_z = tl.arange($(65536))
    r482_value = tl.where(((((r482_z / $(512)) % $(128)) >= $(32)) & (((r482_z / $(512)) % $(128)) < $(64)) & ((((r482_z / $(512)) % $(128)) - $(32)) % $(1) == $(0))), tl.load($(candidate_vcat_43_ptr) + ((((((r482_z / $(512)) % $(128)) - $(32)) % $(32)) * $(1536)) + (((r482_z / $(64)) % $(8)) * $(192)) + (r482_z % $(64)) + $(128)), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r482_value = r482_value.to(tl.bfloat16)
    r482_value = tl.broadcast_to(r482_value, [$(65536)])
    tl.store($(candidate_vslice_backward_11_ptr) + r482_z, r482_value)
    r483_z = tl.arange($(65536))
    r483_value = (tl.load($(candidate_vadd_49_ptr) + ((((r483_z / $(512)) % $(128)) * $(512)) + (((r483_z / $(64)) % $(8)) * $(64)) + (r483_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_11_ptr) + ((((r483_z / $(512)) % $(128)) * $(512)) + (((r483_z / $(64)) % $(8)) * $(64)) + (r483_z % $(64))), dtype=tl.bfloat16))
    r483_value = r483_value.to(tl.bfloat16)
    r483_value = tl.broadcast_to(r483_value, [$(65536)])
    tl.store($(candidate_vadd_60_ptr) + r483_z, r483_value)
    r484_row = tl.arange($(128))
    r484_col = tl.arange($(1024))
    r484_lane = tl.arange($(512))
    for r484_group in range($(1)) {
      r484_left = tl.load($(candidate_vindex_2_ptr) + ((((((r484_row[:, None] * $(512)) + r484_lane[None, :]) / $(512)) % $(128)) * $(576)) + (((r484_row[:, None] * $(512)) + r484_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r484_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r484_lane[:, None] * $(1024)) + r484_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r484_lane[:, None] * $(1024)) + r484_col[None, :]) % $(128)) * $(512)) + ((((r484_lane[:, None] * $(1024)) + r484_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r484_value = tl.dot(r484_left, r484_right)
      r484_value = r484_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_103_ptr) + ((r484_group * $(131072)) + (r484_row[:, None] * $(1024)) + r484_col[None, :]), r484_value)
    }
    r485_row = tl.arange($(128))
    r485_col = tl.arange($(1024))
    r485_lane = tl.arange($(512))
    for r485_group in range($(1)) {
      r485_left = tl.load($(candidate_vindex_2_ptr) + ((((((r485_row[:, None] * $(512)) + r485_lane[None, :]) / $(512)) % $(128)) * $(576)) + (((r485_row[:, None] * $(512)) + r485_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r485_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r485_lane[:, None] * $(1024)) + r485_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r485_lane[:, None] * $(1024)) + r485_col[None, :]) % $(128)) * $(512)) + ((((r485_lane[:, None] * $(1024)) + r485_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r485_value = tl.dot(r485_left, r485_right)
      r485_value = r485_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_104_ptr) + ((r485_group * $(131072)) + (r485_row[:, None] * $(1024)) + r485_col[None, :]), r485_value)
    }
    r486_z = tl.arange($(49152))
    r486_value = tl.where((r486_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r486_z / $(1536)) % $(32)) * $(1024)) + (((r486_z / $(192)) % $(8)) * $(128)) + ((r486_z % $(192)) - $(0))) / $(1024)) % $(128)) * $(1536)) + (((((((r486_z / $(1536)) % $(32)) * $(1024)) + (((r486_z / $(192)) % $(8)) * $(128)) + ((r486_z % $(192)) - $(0))) / $(128)) % $(8)) * $(192)) + (((((r486_z / $(1536)) % $(32)) * $(1024)) + (((r486_z / $(192)) % $(8)) * $(128)) + ((r486_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vadd_6_ptr) + ((((((((r486_z / $(1536)) % $(32)) * $(512)) + (((r486_z / $(192)) % $(8)) * $(64)) + ((r486_z % $(192)) - $(128))) / $(512)) % $(128)) * $(512)) + (((((((r486_z / $(1536)) % $(32)) * $(512)) + (((r486_z / $(192)) % $(8)) * $(64)) + ((r486_z % $(192)) - $(128))) / $(64)) % $(8)) * $(64)) + (((((r486_z / $(1536)) % $(32)) * $(512)) + (((r486_z / $(192)) % $(8)) * $(64)) + ((r486_z % $(192)) - $(128))) % $(64))), dtype=tl.bfloat16))
    r486_value = r486_value.to(tl.bfloat16)
    r486_value = tl.broadcast_to(r486_value, [$(49152)])
    tl.store($(candidate_vcat_47_ptr) + r486_z, r486_value)
    r487_z = tl.arange($(196608))
    r487_value = tl.where((r487_z % $(192)) < $(128), tl.load($(candidate_vbmm_103_ptr) + ((((((((r487_z / $(1536)) % $(128)) * $(1024)) + (((r487_z / $(192)) % $(8)) * $(128)) + ((r487_z % $(192)) - $(0))) / $(1024)) % $(128)) * $(1024)) + (((((((r487_z / $(1536)) % $(128)) * $(1024)) + (((r487_z / $(192)) % $(8)) * $(128)) + ((r487_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r487_z / $(1536)) % $(128)) * $(1024)) + (((r487_z / $(192)) % $(8)) * $(128)) + ((r487_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vindex_2_ptr) + ((((r487_z / $(1536)) % $(128)) * $(576)) + ((((r487_z % $(192)) - $(128)) + $(512)) % $(576))), dtype=tl.bfloat16))
    r487_value = r487_value.to(tl.bfloat16)
    r487_value = tl.broadcast_to(r487_value, [$(196608)])
    tl.store($(candidate_vcat_48_ptr) + r487_z, r487_value)
    r488_z = tl.arange($(18432))
    r488_value = tl.load($(candidate_vcat_47_ptr) + ((((r488_z / $(192)) % $(12)) * $(1536)) + (((r488_z / $(2304)) % $(8)) * $(192)) + (r488_z % $(192))), dtype=tl.bfloat16)
    r488_value = r488_value.to(tl.float32)
    r488_value = tl.broadcast_to(r488_value, [$(18432)])
    tl.store($(candidate_vto_copy_188_ptr) + r488_z, r488_value)
    r489_z = tl.arange($(18432))
    r489_value = tl.load($(candidate_vcat_48_ptr) + ((((r489_z / $(192)) % $(12)) * $(1536)) + (((r489_z / $(2304)) % $(8)) * $(192)) + (r489_z % $(192))), dtype=tl.bfloat16)
    r489_value = r489_value.to(tl.float32)
    r489_value = tl.broadcast_to(r489_value, [$(18432)])
    tl.store($(candidate_vto_copy_189_ptr) + r489_z, r489_value)
    r490_row = tl.arange($(12))
    r490_col = tl.arange($(12))
    r490_lane = tl.arange($(192))
    for r490_group in range($(8)) {
      r490_left = tl.load($(candidate_vto_copy_188_ptr) + (((((((r490_group % $(8)) * $(2304)) + (r490_row[:, None] * $(192)) + r490_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r490_group % $(8)) * $(2304)) + (r490_row[:, None] * $(192)) + r490_lane[None, :]) / $(192)) % $(12)) * $(192)) + ((((r490_group % $(8)) * $(2304)) + (r490_row[:, None] * $(192)) + r490_lane[None, :]) % $(192))), dtype=tl.float32)
      r490_right = tl.load($(candidate_vto_copy_189_ptr) + (((((((r490_group % $(8)) * $(2304)) + (r490_lane[:, None] * $(12)) + r490_col[None, :]) / $(2304)) % $(8)) * $(2304)) + (((((r490_group % $(8)) * $(2304)) + (r490_lane[:, None] * $(12)) + r490_col[None, :]) % $(12)) * $(192)) + (((((r490_group % $(8)) * $(2304)) + (r490_lane[:, None] * $(12)) + r490_col[None, :]) / $(12)) % $(192))), dtype=tl.float32)
      r490_value = tl.dot(r490_left, r490_right)
      r490_value = r490_value.to(tl.float32)
      tl.store($(candidate_vbmm_105_ptr) + ((r490_group * $(144)) + (r490_row[:, None] * $(12)) + r490_col[None, :]), r490_value)
    }
    r491_z = tl.arange($(1152))
    r491_value = (tl.load($(candidate_vbmm_105_ptr) + ((((r491_z / $(144)) % $(8)) * $(144)) + (((r491_z / $(12)) % $(12)) * $(12)) + (r491_z % $(12))), dtype=tl.float32) * 0.07216878364870322)
    r491_value = r491_value.to(tl.float32)
    r491_value = tl.broadcast_to(r491_value, [$(1152)])
    tl.store($(candidate_vmul_89_ptr) + r491_z, r491_value)
    r492_z = tl.arange($(12))
    r492_value = (tl.cast(r492_z, tl.int64) + tl.full([], $(12), dtype=tl.int64))
    r492_value = tl.broadcast_to(r492_value, [$(12)])
    tl.store($(candidate_vadd_61_ptr) + r492_z, r492_value)
    r493_z = tl.arange($(12))
    r493_value = (tl.load($(candidate_vadd_61_ptr) + r493_z, dtype=tl.int64) - tl.full([], $(12), dtype=tl.int64))
    r493_value = tl.broadcast_to(r493_value, [$(12)])
    tl.store($(candidate_vsub_57_ptr) + r493_z, r493_value)
    r494_z = tl.arange($(144))
    r494_value = (tl.cast((r494_z % $(12)), tl.int64) > tl.load($(candidate_vsub_57_ptr) + ((r494_z / $(12)) % $(12)), dtype=tl.int64))
    r494_value = tl.broadcast_to(r494_value, [$(144)])
    tl.store($(candidate_vgt_30_ptr) + r494_z, r494_value)
    r495_z = tl.arange($(1152))
    r495_value = tl.where(tl.load($(candidate_vgt_30_ptr) + ((((r495_z / $(12)) % $(12)) * $(12)) + (r495_z % $(12))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_89_ptr) + ((((r495_z / $(144)) % $(8)) * $(144)) + (((r495_z / $(12)) % $(12)) * $(12)) + (r495_z % $(12))), dtype=tl.float32))
    r495_value = r495_value.to(tl.float32)
    r495_value = tl.broadcast_to(r495_value, [$(1152)])
    tl.store($(candidate_vmasked_fill_30_ptr) + r495_z, r495_value)
    r496_z = tl.arange($(1152))
    r496_lane = tl.arange($(12))
    r496_row_values = tl.load($(candidate_vmasked_fill_30_ptr) + ((((r496_z[:, None] / $(144)) % $(8)) * $(144)) + (((r496_z[:, None] / $(12)) % $(12)) * $(12)) + (r496_lane[None, :] % $(12))), dtype=tl.float32)
    r496_maximum = tl.max(r496_row_values, axis=1, keep_dims=true)
    r496_weights = tl.exp(r496_row_values - r496_maximum)
    r496_total = tl.sum(r496_weights, axis=1)
    r496_maximum = tl.max(r496_row_values, axis=1)
    r496_value = tl.exp(tl.load($(candidate_vmasked_fill_30_ptr) + ((((r496_z / $(144)) % $(8)) * $(144)) + (((r496_z / $(12)) % $(12)) * $(12)) + (r496_z % $(12))), dtype=tl.float32) - r496_maximum) / r496_total
    r496_value = r496_value.to(tl.float32)
    r496_value = tl.broadcast_to(r496_value, [$(1152)])
    tl.store($(candidate_vsoftmax_21_ptr) + r496_z, r496_value)
    r497_z = tl.arange($(1152))
    r497_value = tl.load($(candidate_vsoftmax_21_ptr) + ((((r497_z / $(144)) % $(8)) * $(144)) + (((r497_z / $(12)) % $(12)) * $(12)) + (r497_z % $(12))), dtype=tl.float32)
    r497_value = r497_value.to(tl.bfloat16)
    r497_value = tl.broadcast_to(r497_value, [$(1152)])
    tl.store($(candidate_vto_copy_190_ptr) + r497_z, r497_value)
    r498_row = tl.arange($(12))
    r498_col = tl.arange($(128))
    r498_lane = tl.arange($(12))
    for r498_group in range($(8)) {
      r498_left = tl.load($(candidate_vto_copy_190_ptr) + (((((((r498_group % $(8)) * $(144)) + (r498_row[:, None] * $(12)) + r498_lane[None, :]) / $(144)) % $(8)) * $(144)) + ((((((r498_group % $(8)) * $(144)) + (r498_row[:, None] * $(12)) + r498_lane[None, :]) / $(12)) % $(12)) * $(12)) + ((((r498_group % $(8)) * $(144)) + (r498_row[:, None] * $(12)) + r498_lane[None, :]) % $(12))), dtype=tl.bfloat16)
      r498_right = tl.load($(candidate_vbmm_104_ptr) + (((((((r498_group % $(8)) * $(1536)) + (r498_lane[:, None] * $(128)) + r498_col[None, :]) / $(128)) % $(12)) * $(1024)) + ((((((r498_group % $(8)) * $(1536)) + (r498_lane[:, None] * $(128)) + r498_col[None, :]) / $(1536)) % $(8)) * $(128)) + ((((r498_group % $(8)) * $(1536)) + (r498_lane[:, None] * $(128)) + r498_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r498_value = tl.dot(r498_left, r498_right)
      r498_value = r498_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_106_ptr) + ((r498_group * $(1536)) + (r498_row[:, None] * $(128)) + r498_col[None, :]), r498_value)
    }
    r499_z = tl.arange($(96))
    r499_lane = tl.arange($(12))
    r499_row_values = tl.load($(candidate_vmasked_fill_30_ptr) + ((((r499_z[:, None] / $(12)) % $(8)) * $(144)) + ((r499_z[:, None] % $(12)) * $(12)) + (r499_lane[None, :] % $(12))), dtype=tl.float32)
    r499_maximum = tl.max(r499_row_values, axis=1, keep_dims=true)
    r499_weights = tl.exp(r499_row_values - r499_maximum)
    r499_total = tl.sum(r499_weights, axis=1)
    r499_maximum = tl.max(r499_row_values, axis=1)
    r499_value = r499_maximum + tl.log(r499_total)
    r499_value = r499_value.to(tl.float32)
    r499_value = tl.broadcast_to(r499_value, [$(96)])
    tl.store($(candidate_vlogsumexp_21_ptr) + r499_z, r499_value)
    r500_z = tl.arange($(6144))
    r500_value = tl.load($(candidate_vcat_47_ptr) + ((((r500_z / $(192)) % $(4)) * $(1536)) + (((r500_z / $(768)) % $(8)) * $(192)) + (r500_z % $(192)) + $(18432)), dtype=tl.bfloat16)
    r500_value = r500_value.to(tl.float32)
    r500_value = tl.broadcast_to(r500_value, [$(6144)])
    tl.store($(candidate_vto_copy_191_ptr) + r500_z, r500_value)
    r501_z = tl.arange($(6144))
    r501_value = tl.load($(candidate_vcat_48_ptr) + ((((r501_z / $(192)) % $(4)) * $(1536)) + (((r501_z / $(768)) % $(8)) * $(192)) + (r501_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r501_value = r501_value.to(tl.float32)
    r501_value = tl.broadcast_to(r501_value, [$(6144)])
    tl.store($(candidate_vto_copy_192_ptr) + r501_z, r501_value)
    r502_row = tl.arange($(4))
    r502_col = tl.arange($(4))
    r502_lane = tl.arange($(192))
    for r502_group in range($(8)) {
      r502_left = tl.load($(candidate_vto_copy_191_ptr) + (((((((r502_group % $(8)) * $(768)) + (r502_row[:, None] * $(192)) + r502_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r502_group % $(8)) * $(768)) + (r502_row[:, None] * $(192)) + r502_lane[None, :]) / $(192)) % $(4)) * $(192)) + ((((r502_group % $(8)) * $(768)) + (r502_row[:, None] * $(192)) + r502_lane[None, :]) % $(192))), dtype=tl.float32)
      r502_right = tl.load($(candidate_vto_copy_192_ptr) + (((((((r502_group % $(8)) * $(768)) + (r502_lane[:, None] * $(4)) + r502_col[None, :]) / $(768)) % $(8)) * $(768)) + (((((r502_group % $(8)) * $(768)) + (r502_lane[:, None] * $(4)) + r502_col[None, :]) % $(4)) * $(192)) + (((((r502_group % $(8)) * $(768)) + (r502_lane[:, None] * $(4)) + r502_col[None, :]) / $(4)) % $(192))), dtype=tl.float32)
      r502_value = tl.dot(r502_left, r502_right)
      r502_value = r502_value.to(tl.float32)
      tl.store($(candidate_vbmm_107_ptr) + ((r502_group * $(16)) + (r502_row[:, None] * $(4)) + r502_col[None, :]), r502_value)
    }
    r503_z = tl.arange($(128))
    r503_value = (tl.load($(candidate_vbmm_107_ptr) + ((((r503_z / $(16)) % $(8)) * $(16)) + (((r503_z / $(4)) % $(4)) * $(4)) + (r503_z % $(4))), dtype=tl.float32) * 0.07216878364870322)
    r503_value = r503_value.to(tl.float32)
    r503_value = tl.broadcast_to(r503_value, [$(128)])
    tl.store($(candidate_vmul_90_ptr) + r503_z, r503_value)
    r504_z = tl.arange($(4))
    r504_value = (tl.cast(r504_z, tl.int64) + tl.full([], $(4), dtype=tl.int64))
    r504_value = tl.broadcast_to(r504_value, [$(4)])
    tl.store($(candidate_vadd_62_ptr) + r504_z, r504_value)
    r505_z = tl.arange($(4))
    r505_value = (tl.load($(candidate_vadd_62_ptr) + r505_z, dtype=tl.int64) - tl.full([], $(4), dtype=tl.int64))
    r505_value = tl.broadcast_to(r505_value, [$(4)])
    tl.store($(candidate_vsub_58_ptr) + r505_z, r505_value)
    r506_z = tl.arange($(16))
    r506_value = (tl.cast((r506_z % $(4)), tl.int64) > tl.load($(candidate_vsub_58_ptr) + ((r506_z / $(4)) % $(4)), dtype=tl.int64))
    r506_value = tl.broadcast_to(r506_value, [$(16)])
    tl.store($(candidate_vgt_31_ptr) + r506_z, r506_value)
    r507_z = tl.arange($(128))
    r507_value = tl.where(tl.load($(candidate_vgt_31_ptr) + ((((r507_z / $(4)) % $(4)) * $(4)) + (r507_z % $(4))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_90_ptr) + ((((r507_z / $(16)) % $(8)) * $(16)) + (((r507_z / $(4)) % $(4)) * $(4)) + (r507_z % $(4))), dtype=tl.float32))
    r507_value = r507_value.to(tl.float32)
    r507_value = tl.broadcast_to(r507_value, [$(128)])
    tl.store($(candidate_vmasked_fill_31_ptr) + r507_z, r507_value)
    r508_z = tl.arange($(128))
    r508_lane = tl.arange($(4))
    r508_row_values = tl.load($(candidate_vmasked_fill_31_ptr) + ((((r508_z[:, None] / $(16)) % $(8)) * $(16)) + (((r508_z[:, None] / $(4)) % $(4)) * $(4)) + (r508_lane[None, :] % $(4))), dtype=tl.float32)
    r508_maximum = tl.max(r508_row_values, axis=1, keep_dims=true)
    r508_weights = tl.exp(r508_row_values - r508_maximum)
    r508_total = tl.sum(r508_weights, axis=1)
    r508_maximum = tl.max(r508_row_values, axis=1)
    r508_value = tl.exp(tl.load($(candidate_vmasked_fill_31_ptr) + ((((r508_z / $(16)) % $(8)) * $(16)) + (((r508_z / $(4)) % $(4)) * $(4)) + (r508_z % $(4))), dtype=tl.float32) - r508_maximum) / r508_total
    r508_value = r508_value.to(tl.float32)
    r508_value = tl.broadcast_to(r508_value, [$(128)])
    tl.store($(candidate_vsoftmax_22_ptr) + r508_z, r508_value)
    r509_z = tl.arange($(128))
    r509_value = tl.load($(candidate_vsoftmax_22_ptr) + ((((r509_z / $(16)) % $(8)) * $(16)) + (((r509_z / $(4)) % $(4)) * $(4)) + (r509_z % $(4))), dtype=tl.float32)
    r509_value = r509_value.to(tl.bfloat16)
    r509_value = tl.broadcast_to(r509_value, [$(128)])
    tl.store($(candidate_vto_copy_193_ptr) + r509_z, r509_value)
    r510_row = tl.arange($(4))
    r510_col = tl.arange($(128))
    r510_lane = tl.arange($(4))
    for r510_group in range($(8)) {
      r510_left = tl.load($(candidate_vto_copy_193_ptr) + (((((((r510_group % $(8)) * $(16)) + (r510_row[:, None] * $(4)) + r510_lane[None, :]) / $(16)) % $(8)) * $(16)) + ((((((r510_group % $(8)) * $(16)) + (r510_row[:, None] * $(4)) + r510_lane[None, :]) / $(4)) % $(4)) * $(4)) + ((((r510_group % $(8)) * $(16)) + (r510_row[:, None] * $(4)) + r510_lane[None, :]) % $(4))), dtype=tl.bfloat16)
      r510_right = tl.load($(candidate_vbmm_104_ptr) + (((((((r510_group % $(8)) * $(512)) + (r510_lane[:, None] * $(128)) + r510_col[None, :]) / $(128)) % $(4)) * $(1024)) + ((((((r510_group % $(8)) * $(512)) + (r510_lane[:, None] * $(128)) + r510_col[None, :]) / $(512)) % $(8)) * $(128)) + ((((r510_group % $(8)) * $(512)) + (r510_lane[:, None] * $(128)) + r510_col[None, :]) % $(128)) + $(49152)), dtype=tl.bfloat16)
      r510_value = tl.dot(r510_left, r510_right)
      r510_value = r510_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_108_ptr) + ((r510_group * $(512)) + (r510_row[:, None] * $(128)) + r510_col[None, :]), r510_value)
    }
    r511_z = tl.arange($(32))
    r511_lane = tl.arange($(4))
    r511_row_values = tl.load($(candidate_vmasked_fill_31_ptr) + ((((r511_z[:, None] / $(4)) % $(8)) * $(16)) + ((r511_z[:, None] % $(4)) * $(4)) + (r511_lane[None, :] % $(4))), dtype=tl.float32)
    r511_maximum = tl.max(r511_row_values, axis=1, keep_dims=true)
    r511_weights = tl.exp(r511_row_values - r511_maximum)
    r511_total = tl.sum(r511_weights, axis=1)
    r511_maximum = tl.max(r511_row_values, axis=1)
    r511_value = r511_maximum + tl.log(r511_total)
    r511_value = r511_value.to(tl.float32)
    r511_value = tl.broadcast_to(r511_value, [$(32)])
    tl.store($(candidate_vlogsumexp_22_ptr) + r511_z, r511_value)
    r512_z = tl.arange($(24576))
    r512_value = tl.load($(candidate_vcat_47_ptr) + ((((r512_z / $(192)) % $(16)) * $(1536)) + (((r512_z / $(3072)) % $(8)) * $(192)) + (r512_z % $(192)) + $(24576)), dtype=tl.bfloat16)
    r512_value = r512_value.to(tl.float32)
    r512_value = tl.broadcast_to(r512_value, [$(24576)])
    tl.store($(candidate_vto_copy_194_ptr) + r512_z, r512_value)
    r513_z = tl.arange($(24576))
    r513_value = tl.load($(candidate_vcat_48_ptr) + ((((r513_z / $(192)) % $(16)) * $(1536)) + (((r513_z / $(3072)) % $(8)) * $(192)) + (r513_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r513_value = r513_value.to(tl.float32)
    r513_value = tl.broadcast_to(r513_value, [$(24576)])
    tl.store($(candidate_vto_copy_195_ptr) + r513_z, r513_value)
    r514_row = tl.arange($(16))
    r514_col = tl.arange($(16))
    r514_lane = tl.arange($(192))
    for r514_group in range($(8)) {
      r514_left = tl.load($(candidate_vto_copy_194_ptr) + (((((((r514_group % $(8)) * $(3072)) + (r514_row[:, None] * $(192)) + r514_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r514_group % $(8)) * $(3072)) + (r514_row[:, None] * $(192)) + r514_lane[None, :]) / $(192)) % $(16)) * $(192)) + ((((r514_group % $(8)) * $(3072)) + (r514_row[:, None] * $(192)) + r514_lane[None, :]) % $(192))), dtype=tl.float32)
      r514_right = tl.load($(candidate_vto_copy_195_ptr) + (((((((r514_group % $(8)) * $(3072)) + (r514_lane[:, None] * $(16)) + r514_col[None, :]) / $(3072)) % $(8)) * $(3072)) + (((((r514_group % $(8)) * $(3072)) + (r514_lane[:, None] * $(16)) + r514_col[None, :]) % $(16)) * $(192)) + (((((r514_group % $(8)) * $(3072)) + (r514_lane[:, None] * $(16)) + r514_col[None, :]) / $(16)) % $(192))), dtype=tl.float32)
      r514_value = tl.dot(r514_left, r514_right)
      r514_value = r514_value.to(tl.float32)
      tl.store($(candidate_vbmm_109_ptr) + ((r514_group * $(256)) + (r514_row[:, None] * $(16)) + r514_col[None, :]), r514_value)
    }
    r515_z = tl.arange($(2048))
    r515_value = (tl.load($(candidate_vbmm_109_ptr) + ((((r515_z / $(256)) % $(8)) * $(256)) + (((r515_z / $(16)) % $(16)) * $(16)) + (r515_z % $(16))), dtype=tl.float32) * 0.07216878364870322)
    r515_value = r515_value.to(tl.float32)
    r515_value = tl.broadcast_to(r515_value, [$(2048)])
    tl.store($(candidate_vmul_91_ptr) + r515_z, r515_value)
    r516_z = tl.arange($(16))
    r516_value = (tl.cast(r516_z, tl.int64) + tl.full([], $(16), dtype=tl.int64))
    r516_value = tl.broadcast_to(r516_value, [$(16)])
    tl.store($(candidate_vadd_63_ptr) + r516_z, r516_value)
    r517_z = tl.arange($(16))
    r517_value = (tl.load($(candidate_vadd_63_ptr) + r517_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r517_value = tl.broadcast_to(r517_value, [$(16)])
    tl.store($(candidate_vsub_59_ptr) + r517_z, r517_value)
    r518_z = tl.arange($(256))
    r518_value = (tl.cast((r518_z % $(16)), tl.int64) > tl.load($(candidate_vsub_59_ptr) + ((r518_z / $(16)) % $(16)), dtype=tl.int64))
    r518_value = tl.broadcast_to(r518_value, [$(256)])
    tl.store($(candidate_vgt_32_ptr) + r518_z, r518_value)
    r519_z = tl.arange($(2048))
    r519_value = tl.where(tl.load($(candidate_vgt_32_ptr) + ((((r519_z / $(16)) % $(16)) * $(16)) + (r519_z % $(16))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_91_ptr) + ((((r519_z / $(256)) % $(8)) * $(256)) + (((r519_z / $(16)) % $(16)) * $(16)) + (r519_z % $(16))), dtype=tl.float32))
    r519_value = r519_value.to(tl.float32)
    r519_value = tl.broadcast_to(r519_value, [$(2048)])
    tl.store($(candidate_vmasked_fill_32_ptr) + r519_z, r519_value)
    r520_z = tl.arange($(2048))
    r520_lane = tl.arange($(16))
    r520_row_values = tl.load($(candidate_vmasked_fill_32_ptr) + ((((r520_z[:, None] / $(256)) % $(8)) * $(256)) + (((r520_z[:, None] / $(16)) % $(16)) * $(16)) + (r520_lane[None, :] % $(16))), dtype=tl.float32)
    r520_maximum = tl.max(r520_row_values, axis=1, keep_dims=true)
    r520_weights = tl.exp(r520_row_values - r520_maximum)
    r520_total = tl.sum(r520_weights, axis=1)
    r520_maximum = tl.max(r520_row_values, axis=1)
    r520_value = tl.exp(tl.load($(candidate_vmasked_fill_32_ptr) + ((((r520_z / $(256)) % $(8)) * $(256)) + (((r520_z / $(16)) % $(16)) * $(16)) + (r520_z % $(16))), dtype=tl.float32) - r520_maximum) / r520_total
    r520_value = r520_value.to(tl.float32)
    r520_value = tl.broadcast_to(r520_value, [$(2048)])
    tl.store($(candidate_vsoftmax_23_ptr) + r520_z, r520_value)
    r521_z = tl.arange($(2048))
    r521_value = tl.load($(candidate_vsoftmax_23_ptr) + ((((r521_z / $(256)) % $(8)) * $(256)) + (((r521_z / $(16)) % $(16)) * $(16)) + (r521_z % $(16))), dtype=tl.float32)
    r521_value = r521_value.to(tl.bfloat16)
    r521_value = tl.broadcast_to(r521_value, [$(2048)])
    tl.store($(candidate_vto_copy_196_ptr) + r521_z, r521_value)
    r522_row = tl.arange($(16))
    r522_col = tl.arange($(128))
    r522_lane = tl.arange($(16))
    for r522_group in range($(8)) {
      r522_left = tl.load($(candidate_vto_copy_196_ptr) + (((((((r522_group % $(8)) * $(256)) + (r522_row[:, None] * $(16)) + r522_lane[None, :]) / $(256)) % $(8)) * $(256)) + ((((((r522_group % $(8)) * $(256)) + (r522_row[:, None] * $(16)) + r522_lane[None, :]) / $(16)) % $(16)) * $(16)) + ((((r522_group % $(8)) * $(256)) + (r522_row[:, None] * $(16)) + r522_lane[None, :]) % $(16))), dtype=tl.bfloat16)
      r522_right = tl.load($(candidate_vbmm_104_ptr) + (((((((r522_group % $(8)) * $(2048)) + (r522_lane[:, None] * $(128)) + r522_col[None, :]) / $(128)) % $(16)) * $(1024)) + ((((((r522_group % $(8)) * $(2048)) + (r522_lane[:, None] * $(128)) + r522_col[None, :]) / $(2048)) % $(8)) * $(128)) + ((((r522_group % $(8)) * $(2048)) + (r522_lane[:, None] * $(128)) + r522_col[None, :]) % $(128)) + $(65536)), dtype=tl.bfloat16)
      r522_value = tl.dot(r522_left, r522_right)
      r522_value = r522_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_110_ptr) + ((r522_group * $(2048)) + (r522_row[:, None] * $(128)) + r522_col[None, :]), r522_value)
    }
    r523_z = tl.arange($(128))
    r523_lane = tl.arange($(16))
    r523_row_values = tl.load($(candidate_vmasked_fill_32_ptr) + ((((r523_z[:, None] / $(16)) % $(8)) * $(256)) + ((r523_z[:, None] % $(16)) * $(16)) + (r523_lane[None, :] % $(16))), dtype=tl.float32)
    r523_maximum = tl.max(r523_row_values, axis=1, keep_dims=true)
    r523_weights = tl.exp(r523_row_values - r523_maximum)
    r523_total = tl.sum(r523_weights, axis=1)
    r523_maximum = tl.max(r523_row_values, axis=1)
    r523_value = r523_maximum + tl.log(r523_total)
    r523_value = r523_value.to(tl.float32)
    r523_value = tl.broadcast_to(r523_value, [$(128)])
    tl.store($(candidate_vlogsumexp_23_ptr) + r523_z, r523_value)
    r524_z = tl.arange($(32768))
    r524_value = tl.where(((r524_z / $(1024)) % $(32)) < $(12), tl.load($(candidate_vbmm_106_ptr) + ((((((((r524_z / $(128)) % $(8)) * $(1536)) + ((((r524_z / $(1024)) % $(32)) - $(0)) * $(128)) + (r524_z % $(128))) / $(1536)) % $(8)) * $(1536)) + ((((((r524_z / $(128)) % $(8)) * $(12)) + (((r524_z / $(1024)) % $(32)) - $(0))) % $(12)) * $(128)) + (r524_z % $(128))), dtype=tl.bfloat16), tl.where(((r524_z / $(1024)) % $(32)) < $(16), tl.load($(candidate_vbmm_108_ptr) + ((((((((r524_z / $(128)) % $(8)) * $(512)) + ((((r524_z / $(1024)) % $(32)) - $(12)) * $(128)) + (r524_z % $(128))) / $(512)) % $(8)) * $(512)) + ((((((r524_z / $(128)) % $(8)) * $(4)) + (((r524_z / $(1024)) % $(32)) - $(12))) % $(4)) * $(128)) + (r524_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vbmm_110_ptr) + ((((((((r524_z / $(128)) % $(8)) * $(2048)) + ((((r524_z / $(1024)) % $(32)) - $(16)) * $(128)) + (r524_z % $(128))) / $(2048)) % $(8)) * $(2048)) + ((((((r524_z / $(128)) % $(8)) * $(16)) + (((r524_z / $(1024)) % $(32)) - $(16))) % $(16)) * $(128)) + (r524_z % $(128))), dtype=tl.bfloat16)))
    r524_value = r524_value.to(tl.bfloat16)
    r524_value = tl.broadcast_to(r524_value, [$(32768)])
    tl.store($(candidate_vcat_49_ptr) + r524_z, r524_value)
    r525_z = tl.arange($(256))
    r525_value = tl.where(((r525_z / $(8)) % $(32)) < $(12), tl.load($(candidate_vlogsumexp_21_ptr) + (((r525_z % $(8)) * $(12)) + ((((r525_z / $(8)) % $(32)) - $(0)) % $(12))), dtype=tl.float32), tl.where(((r525_z / $(8)) % $(32)) < $(16), tl.load($(candidate_vlogsumexp_22_ptr) + (((r525_z % $(8)) * $(4)) + ((((r525_z / $(8)) % $(32)) - $(12)) % $(4))), dtype=tl.float32), tl.load($(candidate_vlogsumexp_23_ptr) + (((r525_z % $(8)) * $(16)) + ((((r525_z / $(8)) % $(32)) - $(16)) % $(16))), dtype=tl.float32)))
    r525_value = r525_value.to(tl.float32)
    r525_value = tl.broadcast_to(r525_value, [$(256)])
    tl.store($(candidate_vcat_50_ptr) + r525_z, r525_value)
    r526_z = tl.arange($(18432))
    r526_value = tl.load($(candidate_vcat_47_ptr) + ((((r526_z / $(192)) % $(12)) * $(1536)) + (((r526_z / $(2304)) % $(8)) * $(192)) + (r526_z % $(192))), dtype=tl.bfloat16)
    r526_value = r526_value.to(tl.float32)
    r526_value = tl.broadcast_to(r526_value, [$(18432)])
    tl.store($(candidate_vto_copy_197_ptr) + r526_z, r526_value)
    r527_z = tl.arange($(18432))
    r527_value = tl.load($(candidate_vcat_48_ptr) + ((((r527_z / $(192)) % $(12)) * $(1536)) + (((r527_z / $(2304)) % $(8)) * $(192)) + (r527_z % $(192))), dtype=tl.bfloat16)
    r527_value = r527_value.to(tl.float32)
    r527_value = tl.broadcast_to(r527_value, [$(18432)])
    tl.store($(candidate_vto_copy_198_ptr) + r527_z, r527_value)
    r528_z = tl.arange($(12288))
    r528_value = tl.load($(candidate_vbmm_104_ptr) + ((((r528_z / $(128)) % $(12)) * $(1024)) + (((r528_z / $(1536)) % $(8)) * $(128)) + (r528_z % $(128))), dtype=tl.bfloat16)
    r528_value = r528_value.to(tl.float32)
    r528_value = tl.broadcast_to(r528_value, [$(12288)])
    tl.store($(candidate_vto_copy_199_ptr) + r528_z, r528_value)
    r529_z = tl.arange($(12288))
    r529_value = tl.load($(candidate_vmm_28_ptr) + ((((r529_z / $(128)) % $(12)) * $(1024)) + (((r529_z / $(1536)) % $(8)) * $(128)) + (r529_z % $(128))), dtype=tl.bfloat16)
    r529_value = r529_value.to(tl.float32)
    r529_value = tl.broadcast_to(r529_value, [$(12288)])
    tl.store($(candidate_vto_copy_200_ptr) + r529_z, r529_value)
    r530_z = tl.arange($(12288))
    r530_value = tl.load($(candidate_vcat_49_ptr) + ((((r530_z / $(128)) % $(12)) * $(1024)) + (((r530_z / $(1536)) % $(8)) * $(128)) + (r530_z % $(128))), dtype=tl.bfloat16)
    r530_value = r530_value.to(tl.float32)
    r530_value = tl.broadcast_to(r530_value, [$(12288)])
    tl.store($(candidate_vto_copy_201_ptr) + r530_z, r530_value)
    r531_row = tl.arange($(12))
    r531_col = tl.arange($(12))
    r531_lane = tl.arange($(192))
    for r531_group in range($(8)) {
      r531_left = tl.load($(candidate_vto_copy_197_ptr) + (((((((r531_group % $(8)) * $(2304)) + (r531_row[:, None] * $(192)) + r531_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r531_group % $(8)) * $(2304)) + (r531_row[:, None] * $(192)) + r531_lane[None, :]) / $(192)) % $(12)) * $(192)) + ((((r531_group % $(8)) * $(2304)) + (r531_row[:, None] * $(192)) + r531_lane[None, :]) % $(192))), dtype=tl.float32)
      r531_right = tl.load($(candidate_vto_copy_198_ptr) + (((((((r531_group % $(8)) * $(2304)) + (r531_lane[:, None] * $(12)) + r531_col[None, :]) / $(2304)) % $(8)) * $(2304)) + (((((r531_group % $(8)) * $(2304)) + (r531_lane[:, None] * $(12)) + r531_col[None, :]) % $(12)) * $(192)) + (((((r531_group % $(8)) * $(2304)) + (r531_lane[:, None] * $(12)) + r531_col[None, :]) / $(12)) % $(192))), dtype=tl.float32)
      r531_value = tl.dot(r531_left, r531_right)
      r531_value = r531_value.to(tl.float32)
      tl.store($(candidate_vbmm_111_ptr) + ((r531_group * $(144)) + (r531_row[:, None] * $(12)) + r531_col[None, :]), r531_value)
    }
    r532_z = tl.arange($(1152))
    r532_value = (tl.load($(candidate_vbmm_111_ptr) + ((((r532_z / $(144)) % $(8)) * $(144)) + (((r532_z / $(12)) % $(12)) * $(12)) + (r532_z % $(12))), dtype=tl.float32) * 0.07216878364870322)
    r532_value = r532_value.to(tl.float32)
    r532_value = tl.broadcast_to(r532_value, [$(1152)])
    tl.store($(candidate_vmul_92_ptr) + r532_z, r532_value)
    r533_z = tl.arange($(12))
    r533_value = (tl.cast(r533_z, tl.int64) + tl.full([], $(12), dtype=tl.int64))
    r533_value = tl.broadcast_to(r533_value, [$(12)])
    tl.store($(candidate_vadd_64_ptr) + r533_z, r533_value)
    r534_z = tl.arange($(12))
    r534_value = (tl.load($(candidate_vadd_64_ptr) + r534_z, dtype=tl.int64) - tl.full([], $(12), dtype=tl.int64))
    r534_value = tl.broadcast_to(r534_value, [$(12)])
    tl.store($(candidate_vsub_60_ptr) + r534_z, r534_value)
    r535_z = tl.arange($(144))
    r535_value = (tl.cast((r535_z % $(12)), tl.int64) > tl.load($(candidate_vsub_60_ptr) + ((r535_z / $(12)) % $(12)), dtype=tl.int64))
    r535_value = tl.broadcast_to(r535_value, [$(144)])
    tl.store($(candidate_vgt_33_ptr) + r535_z, r535_value)
    r536_z = tl.arange($(1152))
    r536_value = tl.where(tl.load($(candidate_vgt_33_ptr) + ((((r536_z / $(12)) % $(12)) * $(12)) + (r536_z % $(12))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_92_ptr) + ((((r536_z / $(144)) % $(8)) * $(144)) + (((r536_z / $(12)) % $(12)) * $(12)) + (r536_z % $(12))), dtype=tl.float32))
    r536_value = r536_value.to(tl.float32)
    r536_value = tl.broadcast_to(r536_value, [$(1152)])
    tl.store($(candidate_vmasked_fill_33_ptr) + r536_z, r536_value)
    r537_z = tl.arange($(1152))
    r537_value = (tl.load($(candidate_vmasked_fill_33_ptr) + ((((r537_z / $(144)) % $(8)) * $(144)) + (((r537_z / $(12)) % $(12)) * $(12)) + (r537_z % $(12))), dtype=tl.float32) - tl.load($(candidate_vcat_50_ptr) + ((((r537_z / $(12)) % $(12)) * $(8)) + ((r537_z / $(144)) % $(8))), dtype=tl.float32))
    r537_value = r537_value.to(tl.float32)
    r537_value = tl.broadcast_to(r537_value, [$(1152)])
    tl.store($(candidate_vsub_61_ptr) + r537_z, r537_value)
    r538_z = tl.arange($(1152))
    r538_value = tl.exp(tl.load($(candidate_vsub_61_ptr) + ((((r538_z / $(144)) % $(8)) * $(144)) + (((r538_z / $(12)) % $(12)) * $(12)) + (r538_z % $(12))), dtype=tl.float32))
    r538_value = r538_value.to(tl.float32)
    r538_value = tl.broadcast_to(r538_value, [$(1152)])
    tl.store($(candidate_vexp_9_ptr) + r538_z, r538_value)
    r539_z = tl.arange($(12288))
    r539_value = (tl.load($(candidate_vto_copy_200_ptr) + ((((r539_z / $(1536)) % $(8)) * $(1536)) + (((r539_z / $(128)) % $(12)) * $(128)) + (r539_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_201_ptr) + ((((r539_z / $(1536)) % $(8)) * $(1536)) + (((r539_z / $(128)) % $(12)) * $(128)) + (r539_z % $(128))), dtype=tl.float32))
    r539_value = r539_value.to(tl.float32)
    r539_value = tl.broadcast_to(r539_value, [$(12288)])
    tl.store($(candidate_vmul_93_ptr) + r539_z, r539_value)
    r540_z = tl.arange($(96))
    r540_lane = tl.arange($(128))
    r540_value = tl.sum(tl.load($(candidate_vmul_93_ptr) + ((((r540_z[:, None] / $(12)) % $(8)) * $(1536)) + ((r540_z[:, None] % $(12)) * $(128)) + r540_lane[None, :]), dtype=tl.float32), axis=1)
    r540_value = r540_value.to(tl.float32)
    r540_value = tl.broadcast_to(r540_value, [$(96)])
    tl.store($(candidate_vsum_13_ptr) + r540_z, r540_value)
    r541_row = tl.arange($(12))
    r541_col = tl.arange($(12))
    r541_lane = tl.arange($(128))
    for r541_group in range($(8)) {
      r541_left = tl.load($(candidate_vto_copy_200_ptr) + (((((((r541_group % $(8)) * $(1536)) + (r541_row[:, None] * $(128)) + r541_lane[None, :]) / $(1536)) % $(8)) * $(1536)) + ((((((r541_group % $(8)) * $(1536)) + (r541_row[:, None] * $(128)) + r541_lane[None, :]) / $(128)) % $(12)) * $(128)) + ((((r541_group % $(8)) * $(1536)) + (r541_row[:, None] * $(128)) + r541_lane[None, :]) % $(128))), dtype=tl.float32)
      r541_right = tl.load($(candidate_vto_copy_199_ptr) + (((((((r541_group % $(8)) * $(1536)) + (r541_lane[:, None] * $(12)) + r541_col[None, :]) / $(1536)) % $(8)) * $(1536)) + (((((r541_group % $(8)) * $(1536)) + (r541_lane[:, None] * $(12)) + r541_col[None, :]) % $(12)) * $(128)) + (((((r541_group % $(8)) * $(1536)) + (r541_lane[:, None] * $(12)) + r541_col[None, :]) / $(12)) % $(128))), dtype=tl.float32)
      r541_value = tl.dot(r541_left, r541_right)
      r541_value = r541_value.to(tl.float32)
      tl.store($(candidate_vbmm_112_ptr) + ((r541_group * $(144)) + (r541_row[:, None] * $(12)) + r541_col[None, :]), r541_value)
    }
    r542_z = tl.arange($(1152))
    r542_value = (tl.load($(candidate_vbmm_112_ptr) + ((((r542_z / $(144)) % $(8)) * $(144)) + (((r542_z / $(12)) % $(12)) * $(12)) + (r542_z % $(12))), dtype=tl.float32) - tl.load($(candidate_vsum_13_ptr) + ((((r542_z / $(144)) % $(8)) * $(12)) + ((r542_z / $(12)) % $(12))), dtype=tl.float32))
    r542_value = r542_value.to(tl.float32)
    r542_value = tl.broadcast_to(r542_value, [$(1152)])
    tl.store($(candidate_vsub_62_ptr) + r542_z, r542_value)
    r543_z = tl.arange($(1152))
    r543_value = (tl.load($(candidate_vexp_9_ptr) + ((((r543_z / $(144)) % $(8)) * $(144)) + (((r543_z / $(12)) % $(12)) * $(12)) + (r543_z % $(12))), dtype=tl.float32) * tl.load($(candidate_vsub_62_ptr) + ((((r543_z / $(144)) % $(8)) * $(144)) + (((r543_z / $(12)) % $(12)) * $(12)) + (r543_z % $(12))), dtype=tl.float32))
    r543_value = r543_value.to(tl.float32)
    r543_value = tl.broadcast_to(r543_value, [$(1152)])
    tl.store($(candidate_vmul_94_ptr) + r543_z, r543_value)
    r544_row = tl.arange($(12))
    r544_col = tl.arange($(192))
    r544_lane = tl.arange($(12))
    for r544_group in range($(8)) {
      r544_left = tl.load($(candidate_vmul_94_ptr) + (((((((r544_group % $(8)) * $(144)) + (r544_row[:, None] * $(12)) + r544_lane[None, :]) / $(144)) % $(8)) * $(144)) + ((((((r544_group % $(8)) * $(144)) + (r544_row[:, None] * $(12)) + r544_lane[None, :]) / $(12)) % $(12)) * $(12)) + ((((r544_group % $(8)) * $(144)) + (r544_row[:, None] * $(12)) + r544_lane[None, :]) % $(12))), dtype=tl.float32)
      r544_right = tl.load($(candidate_vto_copy_198_ptr) + (((((((r544_group % $(8)) * $(2304)) + (r544_lane[:, None] * $(192)) + r544_col[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r544_group % $(8)) * $(2304)) + (r544_lane[:, None] * $(192)) + r544_col[None, :]) / $(192)) % $(12)) * $(192)) + ((((r544_group % $(8)) * $(2304)) + (r544_lane[:, None] * $(192)) + r544_col[None, :]) % $(192))), dtype=tl.float32)
      r544_value = tl.dot(r544_left, r544_right)
      r544_value = r544_value.to(tl.float32)
      tl.store($(candidate_vbmm_113_ptr) + ((r544_group * $(2304)) + (r544_row[:, None] * $(192)) + r544_col[None, :]), r544_value)
    }
    r545_z = tl.arange($(18432))
    r545_value = (tl.load($(candidate_vbmm_113_ptr) + ((((r545_z / $(2304)) % $(8)) * $(2304)) + (((r545_z / $(192)) % $(12)) * $(192)) + (r545_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r545_value = r545_value.to(tl.float32)
    r545_value = tl.broadcast_to(r545_value, [$(18432)])
    tl.store($(candidate_vmul_95_ptr) + r545_z, r545_value)
    r546_z = tl.arange($(18432))
    r546_value = tl.load($(candidate_vmul_95_ptr) + ((((r546_z / $(192)) % $(8)) * $(2304)) + (((r546_z / $(1536)) % $(12)) * $(192)) + (r546_z % $(192))), dtype=tl.float32)
    r546_value = r546_value.to(tl.bfloat16)
    r546_value = tl.broadcast_to(r546_value, [$(18432)])
    tl.store($(candidate_vto_copy_202_ptr) + r546_z, r546_value)
    r547_row = tl.arange($(12))
    r547_col = tl.arange($(192))
    r547_lane = tl.arange($(12))
    for r547_group in range($(8)) {
      r547_left = tl.load($(candidate_vmul_94_ptr) + (((((((r547_group % $(8)) * $(144)) + (r547_row[:, None] * $(12)) + r547_lane[None, :]) / $(144)) % $(8)) * $(144)) + (((((r547_group % $(8)) * $(144)) + (r547_row[:, None] * $(12)) + r547_lane[None, :]) % $(12)) * $(12)) + (((((r547_group % $(8)) * $(144)) + (r547_row[:, None] * $(12)) + r547_lane[None, :]) / $(12)) % $(12))), dtype=tl.float32)
      r547_right = tl.load($(candidate_vto_copy_197_ptr) + (((((((r547_group % $(8)) * $(2304)) + (r547_lane[:, None] * $(192)) + r547_col[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r547_group % $(8)) * $(2304)) + (r547_lane[:, None] * $(192)) + r547_col[None, :]) / $(192)) % $(12)) * $(192)) + ((((r547_group % $(8)) * $(2304)) + (r547_lane[:, None] * $(192)) + r547_col[None, :]) % $(192))), dtype=tl.float32)
      r547_value = tl.dot(r547_left, r547_right)
      r547_value = r547_value.to(tl.float32)
      tl.store($(candidate_vbmm_114_ptr) + ((r547_group * $(2304)) + (r547_row[:, None] * $(192)) + r547_col[None, :]), r547_value)
    }
    r548_z = tl.arange($(18432))
    r548_value = (tl.load($(candidate_vbmm_114_ptr) + ((((r548_z / $(2304)) % $(8)) * $(2304)) + (((r548_z / $(192)) % $(12)) * $(192)) + (r548_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r548_value = r548_value.to(tl.float32)
    r548_value = tl.broadcast_to(r548_value, [$(18432)])
    tl.store($(candidate_vmul_96_ptr) + r548_z, r548_value)
    r549_z = tl.arange($(18432))
    r549_value = tl.load($(candidate_vmul_96_ptr) + ((((r549_z / $(192)) % $(8)) * $(2304)) + (((r549_z / $(1536)) % $(12)) * $(192)) + (r549_z % $(192))), dtype=tl.float32)
    r549_value = r549_value.to(tl.bfloat16)
    r549_value = tl.broadcast_to(r549_value, [$(18432)])
    tl.store($(candidate_vto_copy_203_ptr) + r549_z, r549_value)
    r550_row = tl.arange($(12))
    r550_col = tl.arange($(128))
    r550_lane = tl.arange($(12))
    for r550_group in range($(8)) {
      r550_left = tl.load($(candidate_vexp_9_ptr) + (((((((r550_group % $(8)) * $(144)) + (r550_row[:, None] * $(12)) + r550_lane[None, :]) / $(144)) % $(8)) * $(144)) + (((((r550_group % $(8)) * $(144)) + (r550_row[:, None] * $(12)) + r550_lane[None, :]) % $(12)) * $(12)) + (((((r550_group % $(8)) * $(144)) + (r550_row[:, None] * $(12)) + r550_lane[None, :]) / $(12)) % $(12))), dtype=tl.float32)
      r550_right = tl.load($(candidate_vto_copy_200_ptr) + (((((((r550_group % $(8)) * $(1536)) + (r550_lane[:, None] * $(128)) + r550_col[None, :]) / $(1536)) % $(8)) * $(1536)) + ((((((r550_group % $(8)) * $(1536)) + (r550_lane[:, None] * $(128)) + r550_col[None, :]) / $(128)) % $(12)) * $(128)) + ((((r550_group % $(8)) * $(1536)) + (r550_lane[:, None] * $(128)) + r550_col[None, :]) % $(128))), dtype=tl.float32)
      r550_value = tl.dot(r550_left, r550_right)
      r550_value = r550_value.to(tl.float32)
      tl.store($(candidate_vbmm_115_ptr) + ((r550_group * $(1536)) + (r550_row[:, None] * $(128)) + r550_col[None, :]), r550_value)
    }
    r551_z = tl.arange($(12288))
    r551_value = tl.load($(candidate_vbmm_115_ptr) + ((((r551_z / $(128)) % $(8)) * $(1536)) + (((r551_z / $(1024)) % $(12)) * $(128)) + (r551_z % $(128))), dtype=tl.float32)
    r551_value = r551_value.to(tl.bfloat16)
    r551_value = tl.broadcast_to(r551_value, [$(12288)])
    tl.store($(candidate_vto_copy_204_ptr) + r551_z, r551_value)
    r552_z = tl.arange($(73728))
    r552_value = tl.where(((r552_z / $(1536)) % $(48)) < $(12), tl.load($(candidate_vto_copy_203_ptr) + ((((((r552_z / $(1536)) % $(48)) - $(0)) % $(12)) * $(1536)) + (((r552_z / $(192)) % $(8)) * $(192)) + (r552_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r552_value = r552_value.to(tl.bfloat16)
    r552_value = tl.broadcast_to(r552_value, [$(73728)])
    tl.store($(candidate_vcat_51_ptr) + r552_z, r552_value)
    r553_z = tl.arange($(49152))
    r553_value = tl.where(((r553_z / $(1024)) % $(48)) < $(12), tl.load($(candidate_vto_copy_204_ptr) + ((((((r553_z / $(1024)) % $(48)) - $(0)) % $(12)) * $(1024)) + (((r553_z / $(128)) % $(8)) * $(128)) + (r553_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r553_value = r553_value.to(tl.bfloat16)
    r553_value = tl.broadcast_to(r553_value, [$(49152)])
    tl.store($(candidate_vcat_52_ptr) + r553_z, r553_value)
    r554_z = tl.arange($(6144))
    r554_value = tl.load($(candidate_vcat_47_ptr) + ((((r554_z / $(192)) % $(4)) * $(1536)) + (((r554_z / $(768)) % $(8)) * $(192)) + (r554_z % $(192)) + $(18432)), dtype=tl.bfloat16)
    r554_value = r554_value.to(tl.float32)
    r554_value = tl.broadcast_to(r554_value, [$(6144)])
    tl.store($(candidate_vto_copy_205_ptr) + r554_z, r554_value)
    r555_z = tl.arange($(6144))
    r555_value = tl.load($(candidate_vcat_48_ptr) + ((((r555_z / $(192)) % $(4)) * $(1536)) + (((r555_z / $(768)) % $(8)) * $(192)) + (r555_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r555_value = r555_value.to(tl.float32)
    r555_value = tl.broadcast_to(r555_value, [$(6144)])
    tl.store($(candidate_vto_copy_206_ptr) + r555_z, r555_value)
    r556_z = tl.arange($(4096))
    r556_value = tl.load($(candidate_vbmm_104_ptr) + ((((r556_z / $(128)) % $(4)) * $(1024)) + (((r556_z / $(512)) % $(8)) * $(128)) + (r556_z % $(128)) + $(49152)), dtype=tl.bfloat16)
    r556_value = r556_value.to(tl.float32)
    r556_value = tl.broadcast_to(r556_value, [$(4096)])
    tl.store($(candidate_vto_copy_207_ptr) + r556_z, r556_value)
    r557_z = tl.arange($(4096))
    r557_value = tl.load($(candidate_vmm_28_ptr) + ((((r557_z / $(128)) % $(4)) * $(1024)) + (((r557_z / $(512)) % $(8)) * $(128)) + (r557_z % $(128)) + $(12288)), dtype=tl.bfloat16)
    r557_value = r557_value.to(tl.float32)
    r557_value = tl.broadcast_to(r557_value, [$(4096)])
    tl.store($(candidate_vto_copy_208_ptr) + r557_z, r557_value)
    r558_z = tl.arange($(4096))
    r558_value = tl.load($(candidate_vcat_49_ptr) + ((((r558_z / $(128)) % $(4)) * $(1024)) + (((r558_z / $(512)) % $(8)) * $(128)) + (r558_z % $(128)) + $(12288)), dtype=tl.bfloat16)
    r558_value = r558_value.to(tl.float32)
    r558_value = tl.broadcast_to(r558_value, [$(4096)])
    tl.store($(candidate_vto_copy_209_ptr) + r558_z, r558_value)
    r559_row = tl.arange($(4))
    r559_col = tl.arange($(4))
    r559_lane = tl.arange($(192))
    for r559_group in range($(8)) {
      r559_left = tl.load($(candidate_vto_copy_205_ptr) + (((((((r559_group % $(8)) * $(768)) + (r559_row[:, None] * $(192)) + r559_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r559_group % $(8)) * $(768)) + (r559_row[:, None] * $(192)) + r559_lane[None, :]) / $(192)) % $(4)) * $(192)) + ((((r559_group % $(8)) * $(768)) + (r559_row[:, None] * $(192)) + r559_lane[None, :]) % $(192))), dtype=tl.float32)
      r559_right = tl.load($(candidate_vto_copy_206_ptr) + (((((((r559_group % $(8)) * $(768)) + (r559_lane[:, None] * $(4)) + r559_col[None, :]) / $(768)) % $(8)) * $(768)) + (((((r559_group % $(8)) * $(768)) + (r559_lane[:, None] * $(4)) + r559_col[None, :]) % $(4)) * $(192)) + (((((r559_group % $(8)) * $(768)) + (r559_lane[:, None] * $(4)) + r559_col[None, :]) / $(4)) % $(192))), dtype=tl.float32)
      r559_value = tl.dot(r559_left, r559_right)
      r559_value = r559_value.to(tl.float32)
      tl.store($(candidate_vbmm_116_ptr) + ((r559_group * $(16)) + (r559_row[:, None] * $(4)) + r559_col[None, :]), r559_value)
    }
    r560_z = tl.arange($(128))
    r560_value = (tl.load($(candidate_vbmm_116_ptr) + ((((r560_z / $(16)) % $(8)) * $(16)) + (((r560_z / $(4)) % $(4)) * $(4)) + (r560_z % $(4))), dtype=tl.float32) * 0.07216878364870322)
    r560_value = r560_value.to(tl.float32)
    r560_value = tl.broadcast_to(r560_value, [$(128)])
    tl.store($(candidate_vmul_97_ptr) + r560_z, r560_value)
    r561_z = tl.arange($(4))
    r561_value = (tl.cast(r561_z, tl.int64) + tl.full([], $(4), dtype=tl.int64))
    r561_value = tl.broadcast_to(r561_value, [$(4)])
    tl.store($(candidate_vadd_65_ptr) + r561_z, r561_value)
    r562_z = tl.arange($(4))
    r562_value = (tl.load($(candidate_vadd_65_ptr) + r562_z, dtype=tl.int64) - tl.full([], $(4), dtype=tl.int64))
    r562_value = tl.broadcast_to(r562_value, [$(4)])
    tl.store($(candidate_vsub_63_ptr) + r562_z, r562_value)
    r563_z = tl.arange($(16))
    r563_value = (tl.cast((r563_z % $(4)), tl.int64) > tl.load($(candidate_vsub_63_ptr) + ((r563_z / $(4)) % $(4)), dtype=tl.int64))
    r563_value = tl.broadcast_to(r563_value, [$(16)])
    tl.store($(candidate_vgt_34_ptr) + r563_z, r563_value)
    r564_z = tl.arange($(128))
    r564_value = tl.where(tl.load($(candidate_vgt_34_ptr) + ((((r564_z / $(4)) % $(4)) * $(4)) + (r564_z % $(4))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_97_ptr) + ((((r564_z / $(16)) % $(8)) * $(16)) + (((r564_z / $(4)) % $(4)) * $(4)) + (r564_z % $(4))), dtype=tl.float32))
    r564_value = r564_value.to(tl.float32)
    r564_value = tl.broadcast_to(r564_value, [$(128)])
    tl.store($(candidate_vmasked_fill_34_ptr) + r564_z, r564_value)
    r565_z = tl.arange($(128))
    r565_value = (tl.load($(candidate_vmasked_fill_34_ptr) + ((((r565_z / $(16)) % $(8)) * $(16)) + (((r565_z / $(4)) % $(4)) * $(4)) + (r565_z % $(4))), dtype=tl.float32) - tl.load($(candidate_vcat_50_ptr) + ((((r565_z / $(4)) % $(4)) * $(8)) + ((r565_z / $(16)) % $(8)) + $(96)), dtype=tl.float32))
    r565_value = r565_value.to(tl.float32)
    r565_value = tl.broadcast_to(r565_value, [$(128)])
    tl.store($(candidate_vsub_64_ptr) + r565_z, r565_value)
    r566_z = tl.arange($(128))
    r566_value = tl.exp(tl.load($(candidate_vsub_64_ptr) + ((((r566_z / $(16)) % $(8)) * $(16)) + (((r566_z / $(4)) % $(4)) * $(4)) + (r566_z % $(4))), dtype=tl.float32))
    r566_value = r566_value.to(tl.float32)
    r566_value = tl.broadcast_to(r566_value, [$(128)])
    tl.store($(candidate_vexp_10_ptr) + r566_z, r566_value)
    r567_z = tl.arange($(4096))
    r567_value = (tl.load($(candidate_vto_copy_208_ptr) + ((((r567_z / $(512)) % $(8)) * $(512)) + (((r567_z / $(128)) % $(4)) * $(128)) + (r567_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_209_ptr) + ((((r567_z / $(512)) % $(8)) * $(512)) + (((r567_z / $(128)) % $(4)) * $(128)) + (r567_z % $(128))), dtype=tl.float32))
    r567_value = r567_value.to(tl.float32)
    r567_value = tl.broadcast_to(r567_value, [$(4096)])
    tl.store($(candidate_vmul_98_ptr) + r567_z, r567_value)
    r568_z = tl.arange($(32))
    r568_lane = tl.arange($(128))
    r568_value = tl.sum(tl.load($(candidate_vmul_98_ptr) + ((((r568_z[:, None] / $(4)) % $(8)) * $(512)) + ((r568_z[:, None] % $(4)) * $(128)) + r568_lane[None, :]), dtype=tl.float32), axis=1)
    r568_value = r568_value.to(tl.float32)
    r568_value = tl.broadcast_to(r568_value, [$(32)])
    tl.store($(candidate_vsum_14_ptr) + r568_z, r568_value)
    r569_row = tl.arange($(4))
    r569_col = tl.arange($(4))
    r569_lane = tl.arange($(128))
    for r569_group in range($(8)) {
      r569_left = tl.load($(candidate_vto_copy_208_ptr) + (((((((r569_group % $(8)) * $(512)) + (r569_row[:, None] * $(128)) + r569_lane[None, :]) / $(512)) % $(8)) * $(512)) + ((((((r569_group % $(8)) * $(512)) + (r569_row[:, None] * $(128)) + r569_lane[None, :]) / $(128)) % $(4)) * $(128)) + ((((r569_group % $(8)) * $(512)) + (r569_row[:, None] * $(128)) + r569_lane[None, :]) % $(128))), dtype=tl.float32)
      r569_right = tl.load($(candidate_vto_copy_207_ptr) + (((((((r569_group % $(8)) * $(512)) + (r569_lane[:, None] * $(4)) + r569_col[None, :]) / $(512)) % $(8)) * $(512)) + (((((r569_group % $(8)) * $(512)) + (r569_lane[:, None] * $(4)) + r569_col[None, :]) % $(4)) * $(128)) + (((((r569_group % $(8)) * $(512)) + (r569_lane[:, None] * $(4)) + r569_col[None, :]) / $(4)) % $(128))), dtype=tl.float32)
      r569_value = tl.dot(r569_left, r569_right)
      r569_value = r569_value.to(tl.float32)
      tl.store($(candidate_vbmm_117_ptr) + ((r569_group * $(16)) + (r569_row[:, None] * $(4)) + r569_col[None, :]), r569_value)
    }
    r570_z = tl.arange($(128))
    r570_value = (tl.load($(candidate_vbmm_117_ptr) + ((((r570_z / $(16)) % $(8)) * $(16)) + (((r570_z / $(4)) % $(4)) * $(4)) + (r570_z % $(4))), dtype=tl.float32) - tl.load($(candidate_vsum_14_ptr) + ((((r570_z / $(16)) % $(8)) * $(4)) + ((r570_z / $(4)) % $(4))), dtype=tl.float32))
    r570_value = r570_value.to(tl.float32)
    r570_value = tl.broadcast_to(r570_value, [$(128)])
    tl.store($(candidate_vsub_65_ptr) + r570_z, r570_value)
    r571_z = tl.arange($(128))
    r571_value = (tl.load($(candidate_vexp_10_ptr) + ((((r571_z / $(16)) % $(8)) * $(16)) + (((r571_z / $(4)) % $(4)) * $(4)) + (r571_z % $(4))), dtype=tl.float32) * tl.load($(candidate_vsub_65_ptr) + ((((r571_z / $(16)) % $(8)) * $(16)) + (((r571_z / $(4)) % $(4)) * $(4)) + (r571_z % $(4))), dtype=tl.float32))
    r571_value = r571_value.to(tl.float32)
    r571_value = tl.broadcast_to(r571_value, [$(128)])
    tl.store($(candidate_vmul_99_ptr) + r571_z, r571_value)
    r572_row = tl.arange($(4))
    r572_col = tl.arange($(192))
    r572_lane = tl.arange($(4))
    for r572_group in range($(8)) {
      r572_left = tl.load($(candidate_vmul_99_ptr) + (((((((r572_group % $(8)) * $(16)) + (r572_row[:, None] * $(4)) + r572_lane[None, :]) / $(16)) % $(8)) * $(16)) + ((((((r572_group % $(8)) * $(16)) + (r572_row[:, None] * $(4)) + r572_lane[None, :]) / $(4)) % $(4)) * $(4)) + ((((r572_group % $(8)) * $(16)) + (r572_row[:, None] * $(4)) + r572_lane[None, :]) % $(4))), dtype=tl.float32)
      r572_right = tl.load($(candidate_vto_copy_206_ptr) + (((((((r572_group % $(8)) * $(768)) + (r572_lane[:, None] * $(192)) + r572_col[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r572_group % $(8)) * $(768)) + (r572_lane[:, None] * $(192)) + r572_col[None, :]) / $(192)) % $(4)) * $(192)) + ((((r572_group % $(8)) * $(768)) + (r572_lane[:, None] * $(192)) + r572_col[None, :]) % $(192))), dtype=tl.float32)
      r572_value = tl.dot(r572_left, r572_right)
      r572_value = r572_value.to(tl.float32)
      tl.store($(candidate_vbmm_118_ptr) + ((r572_group * $(768)) + (r572_row[:, None] * $(192)) + r572_col[None, :]), r572_value)
    }
    r573_z = tl.arange($(6144))
    r573_value = (tl.load($(candidate_vbmm_118_ptr) + ((((r573_z / $(768)) % $(8)) * $(768)) + (((r573_z / $(192)) % $(4)) * $(192)) + (r573_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r573_value = r573_value.to(tl.float32)
    r573_value = tl.broadcast_to(r573_value, [$(6144)])
    tl.store($(candidate_vmul_100_ptr) + r573_z, r573_value)
    r574_z = tl.arange($(6144))
    r574_value = tl.load($(candidate_vmul_100_ptr) + ((((r574_z / $(192)) % $(8)) * $(768)) + (((r574_z / $(1536)) % $(4)) * $(192)) + (r574_z % $(192))), dtype=tl.float32)
    r574_value = r574_value.to(tl.bfloat16)
    r574_value = tl.broadcast_to(r574_value, [$(6144)])
    tl.store($(candidate_vto_copy_210_ptr) + r574_z, r574_value)
    r575_row = tl.arange($(4))
    r575_col = tl.arange($(192))
    r575_lane = tl.arange($(4))
    for r575_group in range($(8)) {
      r575_left = tl.load($(candidate_vmul_99_ptr) + (((((((r575_group % $(8)) * $(16)) + (r575_row[:, None] * $(4)) + r575_lane[None, :]) / $(16)) % $(8)) * $(16)) + (((((r575_group % $(8)) * $(16)) + (r575_row[:, None] * $(4)) + r575_lane[None, :]) % $(4)) * $(4)) + (((((r575_group % $(8)) * $(16)) + (r575_row[:, None] * $(4)) + r575_lane[None, :]) / $(4)) % $(4))), dtype=tl.float32)
      r575_right = tl.load($(candidate_vto_copy_205_ptr) + (((((((r575_group % $(8)) * $(768)) + (r575_lane[:, None] * $(192)) + r575_col[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r575_group % $(8)) * $(768)) + (r575_lane[:, None] * $(192)) + r575_col[None, :]) / $(192)) % $(4)) * $(192)) + ((((r575_group % $(8)) * $(768)) + (r575_lane[:, None] * $(192)) + r575_col[None, :]) % $(192))), dtype=tl.float32)
      r575_value = tl.dot(r575_left, r575_right)
      r575_value = r575_value.to(tl.float32)
      tl.store($(candidate_vbmm_119_ptr) + ((r575_group * $(768)) + (r575_row[:, None] * $(192)) + r575_col[None, :]), r575_value)
    }
    r576_z = tl.arange($(6144))
    r576_value = (tl.load($(candidate_vbmm_119_ptr) + ((((r576_z / $(768)) % $(8)) * $(768)) + (((r576_z / $(192)) % $(4)) * $(192)) + (r576_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r576_value = r576_value.to(tl.float32)
    r576_value = tl.broadcast_to(r576_value, [$(6144)])
    tl.store($(candidate_vmul_101_ptr) + r576_z, r576_value)
    r577_z = tl.arange($(6144))
    r577_value = tl.load($(candidate_vmul_101_ptr) + ((((r577_z / $(192)) % $(8)) * $(768)) + (((r577_z / $(1536)) % $(4)) * $(192)) + (r577_z % $(192))), dtype=tl.float32)
    r577_value = r577_value.to(tl.bfloat16)
    r577_value = tl.broadcast_to(r577_value, [$(6144)])
    tl.store($(candidate_vto_copy_211_ptr) + r577_z, r577_value)
    r578_row = tl.arange($(4))
    r578_col = tl.arange($(128))
    r578_lane = tl.arange($(4))
    for r578_group in range($(8)) {
      r578_left = tl.load($(candidate_vexp_10_ptr) + (((((((r578_group % $(8)) * $(16)) + (r578_row[:, None] * $(4)) + r578_lane[None, :]) / $(16)) % $(8)) * $(16)) + (((((r578_group % $(8)) * $(16)) + (r578_row[:, None] * $(4)) + r578_lane[None, :]) % $(4)) * $(4)) + (((((r578_group % $(8)) * $(16)) + (r578_row[:, None] * $(4)) + r578_lane[None, :]) / $(4)) % $(4))), dtype=tl.float32)
      r578_right = tl.load($(candidate_vto_copy_208_ptr) + (((((((r578_group % $(8)) * $(512)) + (r578_lane[:, None] * $(128)) + r578_col[None, :]) / $(512)) % $(8)) * $(512)) + ((((((r578_group % $(8)) * $(512)) + (r578_lane[:, None] * $(128)) + r578_col[None, :]) / $(128)) % $(4)) * $(128)) + ((((r578_group % $(8)) * $(512)) + (r578_lane[:, None] * $(128)) + r578_col[None, :]) % $(128))), dtype=tl.float32)
      r578_value = tl.dot(r578_left, r578_right)
      r578_value = r578_value.to(tl.float32)
      tl.store($(candidate_vbmm_120_ptr) + ((r578_group * $(512)) + (r578_row[:, None] * $(128)) + r578_col[None, :]), r578_value)
    }
    r579_z = tl.arange($(4096))
    r579_value = tl.load($(candidate_vbmm_120_ptr) + ((((r579_z / $(128)) % $(8)) * $(512)) + (((r579_z / $(1024)) % $(4)) * $(128)) + (r579_z % $(128))), dtype=tl.float32)
    r579_value = r579_value.to(tl.bfloat16)
    r579_value = tl.broadcast_to(r579_value, [$(4096)])
    tl.store($(candidate_vto_copy_212_ptr) + r579_z, r579_value)
    r580_z = tl.arange($(24576))
    r580_value = tl.where(((r580_z / $(1536)) % $(16)) < $(4), tl.load($(candidate_vto_copy_211_ptr) + ((((((r580_z / $(1536)) % $(16)) - $(0)) % $(4)) * $(1536)) + (((r580_z / $(192)) % $(8)) * $(192)) + (r580_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r580_value = r580_value.to(tl.bfloat16)
    r580_value = tl.broadcast_to(r580_value, [$(24576)])
    tl.store($(candidate_vcat_53_ptr) + r580_z, r580_value)
    r581_z = tl.arange($(16384))
    r581_value = tl.where(((r581_z / $(1024)) % $(16)) < $(4), tl.load($(candidate_vto_copy_212_ptr) + ((((((r581_z / $(1024)) % $(16)) - $(0)) % $(4)) * $(1024)) + (((r581_z / $(128)) % $(8)) * $(128)) + (r581_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r581_value = r581_value.to(tl.bfloat16)
    r581_value = tl.broadcast_to(r581_value, [$(16384)])
    tl.store($(candidate_vcat_54_ptr) + r581_z, r581_value)
    r582_z = tl.arange($(24576))
    r582_value = tl.load($(candidate_vcat_47_ptr) + ((((r582_z / $(192)) % $(16)) * $(1536)) + (((r582_z / $(3072)) % $(8)) * $(192)) + (r582_z % $(192)) + $(24576)), dtype=tl.bfloat16)
    r582_value = r582_value.to(tl.float32)
    r582_value = tl.broadcast_to(r582_value, [$(24576)])
    tl.store($(candidate_vto_copy_213_ptr) + r582_z, r582_value)
    r583_z = tl.arange($(24576))
    r583_value = tl.load($(candidate_vcat_48_ptr) + ((((r583_z / $(192)) % $(16)) * $(1536)) + (((r583_z / $(3072)) % $(8)) * $(192)) + (r583_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r583_value = r583_value.to(tl.float32)
    r583_value = tl.broadcast_to(r583_value, [$(24576)])
    tl.store($(candidate_vto_copy_214_ptr) + r583_z, r583_value)
    r584_z = tl.arange($(16384))
    r584_value = tl.load($(candidate_vbmm_104_ptr) + ((((r584_z / $(128)) % $(16)) * $(1024)) + (((r584_z / $(2048)) % $(8)) * $(128)) + (r584_z % $(128)) + $(65536)), dtype=tl.bfloat16)
    r584_value = r584_value.to(tl.float32)
    r584_value = tl.broadcast_to(r584_value, [$(16384)])
    tl.store($(candidate_vto_copy_215_ptr) + r584_z, r584_value)
    r585_z = tl.arange($(16384))
    r585_value = tl.load($(candidate_vmm_28_ptr) + ((((r585_z / $(128)) % $(16)) * $(1024)) + (((r585_z / $(2048)) % $(8)) * $(128)) + (r585_z % $(128)) + $(16384)), dtype=tl.bfloat16)
    r585_value = r585_value.to(tl.float32)
    r585_value = tl.broadcast_to(r585_value, [$(16384)])
    tl.store($(candidate_vto_copy_216_ptr) + r585_z, r585_value)
    r586_z = tl.arange($(16384))
    r586_value = tl.load($(candidate_vcat_49_ptr) + ((((r586_z / $(128)) % $(16)) * $(1024)) + (((r586_z / $(2048)) % $(8)) * $(128)) + (r586_z % $(128)) + $(16384)), dtype=tl.bfloat16)
    r586_value = r586_value.to(tl.float32)
    r586_value = tl.broadcast_to(r586_value, [$(16384)])
    tl.store($(candidate_vto_copy_217_ptr) + r586_z, r586_value)
    r587_row = tl.arange($(16))
    r587_col = tl.arange($(16))
    r587_lane = tl.arange($(192))
    for r587_group in range($(8)) {
      r587_left = tl.load($(candidate_vto_copy_213_ptr) + (((((((r587_group % $(8)) * $(3072)) + (r587_row[:, None] * $(192)) + r587_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r587_group % $(8)) * $(3072)) + (r587_row[:, None] * $(192)) + r587_lane[None, :]) / $(192)) % $(16)) * $(192)) + ((((r587_group % $(8)) * $(3072)) + (r587_row[:, None] * $(192)) + r587_lane[None, :]) % $(192))), dtype=tl.float32)
      r587_right = tl.load($(candidate_vto_copy_214_ptr) + (((((((r587_group % $(8)) * $(3072)) + (r587_lane[:, None] * $(16)) + r587_col[None, :]) / $(3072)) % $(8)) * $(3072)) + (((((r587_group % $(8)) * $(3072)) + (r587_lane[:, None] * $(16)) + r587_col[None, :]) % $(16)) * $(192)) + (((((r587_group % $(8)) * $(3072)) + (r587_lane[:, None] * $(16)) + r587_col[None, :]) / $(16)) % $(192))), dtype=tl.float32)
      r587_value = tl.dot(r587_left, r587_right)
      r587_value = r587_value.to(tl.float32)
      tl.store($(candidate_vbmm_121_ptr) + ((r587_group * $(256)) + (r587_row[:, None] * $(16)) + r587_col[None, :]), r587_value)
    }
    r588_z = tl.arange($(2048))
    r588_value = (tl.load($(candidate_vbmm_121_ptr) + ((((r588_z / $(256)) % $(8)) * $(256)) + (((r588_z / $(16)) % $(16)) * $(16)) + (r588_z % $(16))), dtype=tl.float32) * 0.07216878364870322)
    r588_value = r588_value.to(tl.float32)
    r588_value = tl.broadcast_to(r588_value, [$(2048)])
    tl.store($(candidate_vmul_102_ptr) + r588_z, r588_value)
    r589_z = tl.arange($(16))
    r589_value = (tl.cast(r589_z, tl.int64) + tl.full([], $(16), dtype=tl.int64))
    r589_value = tl.broadcast_to(r589_value, [$(16)])
    tl.store($(candidate_vadd_66_ptr) + r589_z, r589_value)
    r590_z = tl.arange($(16))
    r590_value = (tl.load($(candidate_vadd_66_ptr) + r590_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r590_value = tl.broadcast_to(r590_value, [$(16)])
    tl.store($(candidate_vsub_66_ptr) + r590_z, r590_value)
    r591_z = tl.arange($(256))
    r591_value = (tl.cast((r591_z % $(16)), tl.int64) > tl.load($(candidate_vsub_66_ptr) + ((r591_z / $(16)) % $(16)), dtype=tl.int64))
    r591_value = tl.broadcast_to(r591_value, [$(256)])
    tl.store($(candidate_vgt_35_ptr) + r591_z, r591_value)
    r592_z = tl.arange($(2048))
    r592_value = tl.where(tl.load($(candidate_vgt_35_ptr) + ((((r592_z / $(16)) % $(16)) * $(16)) + (r592_z % $(16))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_102_ptr) + ((((r592_z / $(256)) % $(8)) * $(256)) + (((r592_z / $(16)) % $(16)) * $(16)) + (r592_z % $(16))), dtype=tl.float32))
    r592_value = r592_value.to(tl.float32)
    r592_value = tl.broadcast_to(r592_value, [$(2048)])
    tl.store($(candidate_vmasked_fill_35_ptr) + r592_z, r592_value)
    r593_z = tl.arange($(2048))
    r593_value = (tl.load($(candidate_vmasked_fill_35_ptr) + ((((r593_z / $(256)) % $(8)) * $(256)) + (((r593_z / $(16)) % $(16)) * $(16)) + (r593_z % $(16))), dtype=tl.float32) - tl.load($(candidate_vcat_50_ptr) + ((((r593_z / $(16)) % $(16)) * $(8)) + ((r593_z / $(256)) % $(8)) + $(128)), dtype=tl.float32))
    r593_value = r593_value.to(tl.float32)
    r593_value = tl.broadcast_to(r593_value, [$(2048)])
    tl.store($(candidate_vsub_67_ptr) + r593_z, r593_value)
    r594_z = tl.arange($(2048))
    r594_value = tl.exp(tl.load($(candidate_vsub_67_ptr) + ((((r594_z / $(256)) % $(8)) * $(256)) + (((r594_z / $(16)) % $(16)) * $(16)) + (r594_z % $(16))), dtype=tl.float32))
    r594_value = r594_value.to(tl.float32)
    r594_value = tl.broadcast_to(r594_value, [$(2048)])
    tl.store($(candidate_vexp_11_ptr) + r594_z, r594_value)
    r595_z = tl.arange($(16384))
    r595_value = (tl.load($(candidate_vto_copy_216_ptr) + ((((r595_z / $(2048)) % $(8)) * $(2048)) + (((r595_z / $(128)) % $(16)) * $(128)) + (r595_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_217_ptr) + ((((r595_z / $(2048)) % $(8)) * $(2048)) + (((r595_z / $(128)) % $(16)) * $(128)) + (r595_z % $(128))), dtype=tl.float32))
    r595_value = r595_value.to(tl.float32)
    r595_value = tl.broadcast_to(r595_value, [$(16384)])
    tl.store($(candidate_vmul_103_ptr) + r595_z, r595_value)
    r596_z = tl.arange($(128))
    r596_lane = tl.arange($(128))
    r596_value = tl.sum(tl.load($(candidate_vmul_103_ptr) + ((((r596_z[:, None] / $(16)) % $(8)) * $(2048)) + ((r596_z[:, None] % $(16)) * $(128)) + r596_lane[None, :]), dtype=tl.float32), axis=1)
    r596_value = r596_value.to(tl.float32)
    r596_value = tl.broadcast_to(r596_value, [$(128)])
    tl.store($(candidate_vsum_15_ptr) + r596_z, r596_value)
    r597_row = tl.arange($(16))
    r597_col = tl.arange($(16))
    r597_lane = tl.arange($(128))
    for r597_group in range($(8)) {
      r597_left = tl.load($(candidate_vto_copy_216_ptr) + (((((((r597_group % $(8)) * $(2048)) + (r597_row[:, None] * $(128)) + r597_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r597_group % $(8)) * $(2048)) + (r597_row[:, None] * $(128)) + r597_lane[None, :]) / $(128)) % $(16)) * $(128)) + ((((r597_group % $(8)) * $(2048)) + (r597_row[:, None] * $(128)) + r597_lane[None, :]) % $(128))), dtype=tl.float32)
      r597_right = tl.load($(candidate_vto_copy_215_ptr) + (((((((r597_group % $(8)) * $(2048)) + (r597_lane[:, None] * $(16)) + r597_col[None, :]) / $(2048)) % $(8)) * $(2048)) + (((((r597_group % $(8)) * $(2048)) + (r597_lane[:, None] * $(16)) + r597_col[None, :]) % $(16)) * $(128)) + (((((r597_group % $(8)) * $(2048)) + (r597_lane[:, None] * $(16)) + r597_col[None, :]) / $(16)) % $(128))), dtype=tl.float32)
      r597_value = tl.dot(r597_left, r597_right)
      r597_value = r597_value.to(tl.float32)
      tl.store($(candidate_vbmm_122_ptr) + ((r597_group * $(256)) + (r597_row[:, None] * $(16)) + r597_col[None, :]), r597_value)
    }
    r598_z = tl.arange($(2048))
    r598_value = (tl.load($(candidate_vbmm_122_ptr) + ((((r598_z / $(256)) % $(8)) * $(256)) + (((r598_z / $(16)) % $(16)) * $(16)) + (r598_z % $(16))), dtype=tl.float32) - tl.load($(candidate_vsum_15_ptr) + ((((r598_z / $(256)) % $(8)) * $(16)) + ((r598_z / $(16)) % $(16))), dtype=tl.float32))
    r598_value = r598_value.to(tl.float32)
    r598_value = tl.broadcast_to(r598_value, [$(2048)])
    tl.store($(candidate_vsub_68_ptr) + r598_z, r598_value)
    r599_z = tl.arange($(2048))
    r599_value = (tl.load($(candidate_vexp_11_ptr) + ((((r599_z / $(256)) % $(8)) * $(256)) + (((r599_z / $(16)) % $(16)) * $(16)) + (r599_z % $(16))), dtype=tl.float32) * tl.load($(candidate_vsub_68_ptr) + ((((r599_z / $(256)) % $(8)) * $(256)) + (((r599_z / $(16)) % $(16)) * $(16)) + (r599_z % $(16))), dtype=tl.float32))
    r599_value = r599_value.to(tl.float32)
    r599_value = tl.broadcast_to(r599_value, [$(2048)])
    tl.store($(candidate_vmul_104_ptr) + r599_z, r599_value)
    r600_row = tl.arange($(16))
    r600_col = tl.arange($(192))
    r600_lane = tl.arange($(16))
    for r600_group in range($(8)) {
      r600_left = tl.load($(candidate_vmul_104_ptr) + (((((((r600_group % $(8)) * $(256)) + (r600_row[:, None] * $(16)) + r600_lane[None, :]) / $(256)) % $(8)) * $(256)) + ((((((r600_group % $(8)) * $(256)) + (r600_row[:, None] * $(16)) + r600_lane[None, :]) / $(16)) % $(16)) * $(16)) + ((((r600_group % $(8)) * $(256)) + (r600_row[:, None] * $(16)) + r600_lane[None, :]) % $(16))), dtype=tl.float32)
      r600_right = tl.load($(candidate_vto_copy_214_ptr) + (((((((r600_group % $(8)) * $(3072)) + (r600_lane[:, None] * $(192)) + r600_col[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r600_group % $(8)) * $(3072)) + (r600_lane[:, None] * $(192)) + r600_col[None, :]) / $(192)) % $(16)) * $(192)) + ((((r600_group % $(8)) * $(3072)) + (r600_lane[:, None] * $(192)) + r600_col[None, :]) % $(192))), dtype=tl.float32)
      r600_value = tl.dot(r600_left, r600_right)
      r600_value = r600_value.to(tl.float32)
      tl.store($(candidate_vbmm_123_ptr) + ((r600_group * $(3072)) + (r600_row[:, None] * $(192)) + r600_col[None, :]), r600_value)
    }
    r601_z = tl.arange($(24576))
    r601_value = (tl.load($(candidate_vbmm_123_ptr) + ((((r601_z / $(3072)) % $(8)) * $(3072)) + (((r601_z / $(192)) % $(16)) * $(192)) + (r601_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r601_value = r601_value.to(tl.float32)
    r601_value = tl.broadcast_to(r601_value, [$(24576)])
    tl.store($(candidate_vmul_105_ptr) + r601_z, r601_value)
    r602_z = tl.arange($(24576))
    r602_value = tl.load($(candidate_vmul_105_ptr) + ((((r602_z / $(192)) % $(8)) * $(3072)) + (((r602_z / $(1536)) % $(16)) * $(192)) + (r602_z % $(192))), dtype=tl.float32)
    r602_value = r602_value.to(tl.bfloat16)
    r602_value = tl.broadcast_to(r602_value, [$(24576)])
    tl.store($(candidate_vto_copy_218_ptr) + r602_z, r602_value)
    r603_row = tl.arange($(16))
    r603_col = tl.arange($(192))
    r603_lane = tl.arange($(16))
    for r603_group in range($(8)) {
      r603_left = tl.load($(candidate_vmul_104_ptr) + (((((((r603_group % $(8)) * $(256)) + (r603_row[:, None] * $(16)) + r603_lane[None, :]) / $(256)) % $(8)) * $(256)) + (((((r603_group % $(8)) * $(256)) + (r603_row[:, None] * $(16)) + r603_lane[None, :]) % $(16)) * $(16)) + (((((r603_group % $(8)) * $(256)) + (r603_row[:, None] * $(16)) + r603_lane[None, :]) / $(16)) % $(16))), dtype=tl.float32)
      r603_right = tl.load($(candidate_vto_copy_213_ptr) + (((((((r603_group % $(8)) * $(3072)) + (r603_lane[:, None] * $(192)) + r603_col[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r603_group % $(8)) * $(3072)) + (r603_lane[:, None] * $(192)) + r603_col[None, :]) / $(192)) % $(16)) * $(192)) + ((((r603_group % $(8)) * $(3072)) + (r603_lane[:, None] * $(192)) + r603_col[None, :]) % $(192))), dtype=tl.float32)
      r603_value = tl.dot(r603_left, r603_right)
      r603_value = r603_value.to(tl.float32)
      tl.store($(candidate_vbmm_124_ptr) + ((r603_group * $(3072)) + (r603_row[:, None] * $(192)) + r603_col[None, :]), r603_value)
    }
    r604_z = tl.arange($(24576))
    r604_value = (tl.load($(candidate_vbmm_124_ptr) + ((((r604_z / $(3072)) % $(8)) * $(3072)) + (((r604_z / $(192)) % $(16)) * $(192)) + (r604_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r604_value = r604_value.to(tl.float32)
    r604_value = tl.broadcast_to(r604_value, [$(24576)])
    tl.store($(candidate_vmul_106_ptr) + r604_z, r604_value)
    r605_z = tl.arange($(24576))
    r605_value = tl.load($(candidate_vmul_106_ptr) + ((((r605_z / $(192)) % $(8)) * $(3072)) + (((r605_z / $(1536)) % $(16)) * $(192)) + (r605_z % $(192))), dtype=tl.float32)
    r605_value = r605_value.to(tl.bfloat16)
    r605_value = tl.broadcast_to(r605_value, [$(24576)])
    tl.store($(candidate_vto_copy_219_ptr) + r605_z, r605_value)
    r606_row = tl.arange($(16))
    r606_col = tl.arange($(128))
    r606_lane = tl.arange($(16))
    for r606_group in range($(8)) {
      r606_left = tl.load($(candidate_vexp_11_ptr) + (((((((r606_group % $(8)) * $(256)) + (r606_row[:, None] * $(16)) + r606_lane[None, :]) / $(256)) % $(8)) * $(256)) + (((((r606_group % $(8)) * $(256)) + (r606_row[:, None] * $(16)) + r606_lane[None, :]) % $(16)) * $(16)) + (((((r606_group % $(8)) * $(256)) + (r606_row[:, None] * $(16)) + r606_lane[None, :]) / $(16)) % $(16))), dtype=tl.float32)
      r606_right = tl.load($(candidate_vto_copy_216_ptr) + (((((((r606_group % $(8)) * $(2048)) + (r606_lane[:, None] * $(128)) + r606_col[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r606_group % $(8)) * $(2048)) + (r606_lane[:, None] * $(128)) + r606_col[None, :]) / $(128)) % $(16)) * $(128)) + ((((r606_group % $(8)) * $(2048)) + (r606_lane[:, None] * $(128)) + r606_col[None, :]) % $(128))), dtype=tl.float32)
      r606_value = tl.dot(r606_left, r606_right)
      r606_value = r606_value.to(tl.float32)
      tl.store($(candidate_vbmm_125_ptr) + ((r606_group * $(2048)) + (r606_row[:, None] * $(128)) + r606_col[None, :]), r606_value)
    }
    r607_z = tl.arange($(16384))
    r607_value = tl.load($(candidate_vbmm_125_ptr) + ((((r607_z / $(128)) % $(8)) * $(2048)) + (((r607_z / $(1024)) % $(16)) * $(128)) + (r607_z % $(128))), dtype=tl.float32)
    r607_value = r607_value.to(tl.bfloat16)
    r607_value = tl.broadcast_to(r607_value, [$(16384)])
    tl.store($(candidate_vto_copy_220_ptr) + r607_z, r607_value)
    r608_z = tl.arange($(98304))
    r608_value = tl.where(((r608_z / $(1536)) % $(64)) < $(16), tl.load($(candidate_vto_copy_219_ptr) + ((((((r608_z / $(1536)) % $(64)) - $(0)) % $(16)) * $(1536)) + (((r608_z / $(192)) % $(8)) * $(192)) + (r608_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r608_value = r608_value.to(tl.bfloat16)
    r608_value = tl.broadcast_to(r608_value, [$(98304)])
    tl.store($(candidate_vcat_55_ptr) + r608_z, r608_value)
    r609_z = tl.arange($(65536))
    r609_value = tl.where(((r609_z / $(1024)) % $(64)) < $(16), tl.load($(candidate_vto_copy_220_ptr) + ((((((r609_z / $(1024)) % $(64)) - $(0)) % $(16)) * $(1024)) + (((r609_z / $(128)) % $(8)) * $(128)) + (r609_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r609_value = r609_value.to(tl.bfloat16)
    r609_value = tl.broadcast_to(r609_value, [$(65536)])
    tl.store($(candidate_vcat_56_ptr) + r609_z, r609_value)
    r610_z = tl.arange($(49152))
    r610_value = tl.where(((r610_z / $(1536)) % $(32)) < $(12), tl.load($(candidate_vto_copy_202_ptr) + ((((((r610_z / $(1536)) % $(32)) - $(0)) % $(12)) * $(1536)) + (((r610_z / $(192)) % $(8)) * $(192)) + (r610_z % $(192))), dtype=tl.bfloat16), tl.where(((r610_z / $(1536)) % $(32)) < $(16), tl.load($(candidate_vto_copy_210_ptr) + ((((((r610_z / $(1536)) % $(32)) - $(12)) % $(4)) * $(1536)) + (((r610_z / $(192)) % $(8)) * $(192)) + (r610_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vto_copy_218_ptr) + ((((((r610_z / $(1536)) % $(32)) - $(16)) % $(16)) * $(1536)) + (((r610_z / $(192)) % $(8)) * $(192)) + (r610_z % $(192))), dtype=tl.bfloat16)))
    r610_value = r610_value.to(tl.bfloat16)
    r610_value = tl.broadcast_to(r610_value, [$(49152)])
    tl.store($(candidate_vcat_57_ptr) + r610_z, r610_value)
    r611_z = tl.arange($(196608))
    r611_value = tl.where(((r611_z / $(1536)) % $(128)) < $(48), tl.load($(candidate_vcat_51_ptr) + ((((((r611_z / $(1536)) % $(128)) - $(0)) % $(48)) * $(1536)) + (((r611_z / $(192)) % $(8)) * $(192)) + (r611_z % $(192))), dtype=tl.bfloat16), tl.where(((r611_z / $(1536)) % $(128)) < $(64), tl.load($(candidate_vcat_53_ptr) + ((((((r611_z / $(1536)) % $(128)) - $(48)) % $(16)) * $(1536)) + (((r611_z / $(192)) % $(8)) * $(192)) + (r611_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vcat_55_ptr) + ((((((r611_z / $(1536)) % $(128)) - $(64)) % $(64)) * $(1536)) + (((r611_z / $(192)) % $(8)) * $(192)) + (r611_z % $(192))), dtype=tl.bfloat16)))
    r611_value = r611_value.to(tl.bfloat16)
    r611_value = tl.broadcast_to(r611_value, [$(196608)])
    tl.store($(candidate_vcat_58_ptr) + r611_z, r611_value)
    r612_z = tl.arange($(131072))
    r612_value = tl.where(((r612_z / $(1024)) % $(128)) < $(48), tl.load($(candidate_vcat_52_ptr) + ((((((r612_z / $(1024)) % $(128)) - $(0)) % $(48)) * $(1024)) + (((r612_z / $(128)) % $(8)) * $(128)) + (r612_z % $(128))), dtype=tl.bfloat16), tl.where(((r612_z / $(1024)) % $(128)) < $(64), tl.load($(candidate_vcat_54_ptr) + ((((((r612_z / $(1024)) % $(128)) - $(48)) % $(16)) * $(1024)) + (((r612_z / $(128)) % $(8)) * $(128)) + (r612_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_56_ptr) + ((((((r612_z / $(1024)) % $(128)) - $(64)) % $(64)) * $(1024)) + (((r612_z / $(128)) % $(8)) * $(128)) + (r612_z % $(128))), dtype=tl.bfloat16)))
    r612_value = r612_value.to(tl.bfloat16)
    r612_value = tl.broadcast_to(r612_value, [$(131072)])
    tl.store($(candidate_vcat_59_ptr) + r612_z, r612_value)
    r613_z = tl.arange($(8192))
    r613_lane = tl.arange($(8))
    r613_value = tl.sum(tl.load($(candidate_vcat_58_ptr) + ((((r613_z[:, None] / $(64)) % $(128)) * $(1536)) + (r613_lane[None, :] * $(192)) + (r613_z[:, None] % $(64)) + $(128)), dtype=tl.bfloat16), axis=1)
    r613_value = r613_value.to(tl.bfloat16)
    r613_value = tl.broadcast_to(r613_value, [$(8192)])
    tl.store($(candidate_vsum_16_ptr) + r613_z, r613_value)
    r614_row = tl.arange($(128))
    r614_col = tl.arange($(512))
    r614_lane = tl.arange($(1024))
    for r614_group in range($(1)) {
      r614_left = tl.load($(candidate_vcat_58_ptr) + ((((((r614_row[:, None] * $(1024)) + r614_lane[None, :]) / $(1024)) % $(128)) * $(1536)) + ((((r614_row[:, None] * $(1024)) + r614_lane[None, :]) % $(8)) * $(192)) + ((((r614_row[:, None] * $(1024)) + r614_lane[None, :]) / $(8)) % $(128))), dtype=tl.bfloat16)
      r614_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r614_lane[:, None] * $(512)) + r614_col[None, :]) / $(512)) % $(8)) * $(131072)) + (((((r614_lane[:, None] * $(512)) + r614_col[None, :]) / $(4096)) % $(128)) * $(512)) + (((r614_lane[:, None] * $(512)) + r614_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r614_value = tl.dot(r614_left, r614_right)
      r614_value = r614_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_126_ptr) + ((r614_group * $(65536)) + (r614_row[:, None] * $(512)) + r614_col[None, :]), r614_value)
    }
    r615_row = tl.arange($(128))
    r615_col = tl.arange($(512))
    r615_lane = tl.arange($(1024))
    for r615_group in range($(1)) {
      r615_left = tl.load($(candidate_vcat_59_ptr) + ((((((r615_row[:, None] * $(1024)) + r615_lane[None, :]) / $(1024)) % $(128)) * $(1024)) + ((((r615_row[:, None] * $(1024)) + r615_lane[None, :]) % $(8)) * $(128)) + ((((r615_row[:, None] * $(1024)) + r615_lane[None, :]) / $(8)) % $(128))), dtype=tl.bfloat16)
      r615_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r615_lane[:, None] * $(512)) + r615_col[None, :]) / $(512)) % $(8)) * $(131072)) + (((((r615_lane[:, None] * $(512)) + r615_col[None, :]) / $(4096)) % $(128)) * $(512)) + (((r615_lane[:, None] * $(512)) + r615_col[None, :]) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r615_value = tl.dot(r615_left, r615_right)
      r615_value = r615_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_127_ptr) + ((r615_group * $(65536)) + (r615_row[:, None] * $(512)) + r615_col[None, :]), r615_value)
    }
    r616_z = tl.arange($(65536))
    r616_value = (tl.load($(candidate_vbmm_126_ptr) + ((((r616_z / $(512)) % $(128)) * $(512)) + (r616_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vbmm_127_ptr) + ((((r616_z / $(512)) % $(128)) * $(512)) + (r616_z % $(512))), dtype=tl.bfloat16))
    r616_value = r616_value.to(tl.bfloat16)
    r616_value = tl.broadcast_to(r616_value, [$(65536)])
    tl.store($(candidate_vadd_67_ptr) + r616_z, r616_value)
    r617_z = tl.arange($(65536))
    r617_value = tl.where(((((r617_z / $(512)) % $(128)) >= $(0)) & (((r617_z / $(512)) % $(128)) < $(128)) & ((((r617_z / $(512)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vadd_67_ptr) + ((((((r617_z / $(512)) % $(128)) - $(0)) % $(128)) * $(512)) + (r617_z % $(512))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r617_value = r617_value.to(tl.bfloat16)
    r617_value = tl.broadcast_to(r617_value, [$(65536)])
    tl.store($(candidate_vslice_backward_12_ptr) + r617_z, r617_value)
    r618_z = tl.arange($(65536))
    r618_value = (tl.load($(candidate_vadd_57_ptr) + ((((r618_z / $(512)) % $(128)) * $(512)) + (r618_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_12_ptr) + ((((r618_z / $(512)) % $(128)) * $(512)) + (r618_z % $(512))), dtype=tl.bfloat16))
    r618_value = r618_value.to(tl.bfloat16)
    r618_value = tl.broadcast_to(r618_value, [$(65536)])
    tl.store($(candidate_vadd_68_ptr) + r618_z, r618_value)
    r619_z = tl.arange($(8192))
    r619_value = tl.where(((((r619_z / $(64)) % $(128)) >= $(0)) & (((r619_z / $(64)) % $(128)) < $(128)) & ((((r619_z / $(64)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vsum_16_ptr) + ((((((r619_z / $(64)) % $(128)) - $(0)) % $(128)) * $(64)) + (r619_z % $(64))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r619_value = r619_value.to(tl.bfloat16)
    r619_value = tl.broadcast_to(r619_value, [$(8192)])
    tl.store($(candidate_vslice_backward_13_ptr) + r619_z, r619_value)
    r620_z = tl.arange($(8192))
    r620_value = (tl.load($(candidate_vadd_58_ptr) + ((((r620_z / $(64)) % $(128)) * $(64)) + (r620_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_13_ptr) + ((((r620_z / $(64)) % $(128)) * $(64)) + (r620_z % $(64))), dtype=tl.bfloat16))
    r620_value = r620_value.to(tl.bfloat16)
    r620_value = tl.broadcast_to(r620_value, [$(8192)])
    tl.store($(candidate_vadd_69_ptr) + r620_z, r620_value)
    r621_z = tl.arange($(131072))
    r621_value = tl.where(((((r621_z / $(1024)) % $(128)) >= $(0)) & (((r621_z / $(1024)) % $(128)) < $(32)) & ((((r621_z / $(1024)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vcat_57_ptr) + ((((((r621_z / $(1024)) % $(128)) - $(0)) % $(32)) * $(1536)) + (((r621_z / $(128)) % $(8)) * $(192)) + (r621_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r621_value = r621_value.to(tl.bfloat16)
    r621_value = tl.broadcast_to(r621_value, [$(131072)])
    tl.store($(candidate_vslice_backward_14_ptr) + r621_z, r621_value)
    r622_z = tl.arange($(131072))
    r622_value = (tl.load($(candidate_vadd_59_ptr) + ((((r622_z / $(1024)) % $(128)) * $(1024)) + (((r622_z / $(128)) % $(8)) * $(128)) + (r622_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_14_ptr) + ((((r622_z / $(1024)) % $(128)) * $(1024)) + (((r622_z / $(128)) % $(8)) * $(128)) + (r622_z % $(128))), dtype=tl.bfloat16))
    r622_value = r622_value.to(tl.bfloat16)
    r622_value = tl.broadcast_to(r622_value, [$(131072)])
    tl.store($(candidate_vadd_70_ptr) + r622_z, r622_value)
    r623_z = tl.arange($(65536))
    r623_value = tl.where(((((r623_z / $(512)) % $(128)) >= $(0)) & (((r623_z / $(512)) % $(128)) < $(32)) & ((((r623_z / $(512)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vcat_57_ptr) + ((((((r623_z / $(512)) % $(128)) - $(0)) % $(32)) * $(1536)) + (((r623_z / $(64)) % $(8)) * $(192)) + (r623_z % $(64)) + $(128)), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r623_value = r623_value.to(tl.bfloat16)
    r623_value = tl.broadcast_to(r623_value, [$(65536)])
    tl.store($(candidate_vslice_backward_15_ptr) + r623_z, r623_value)
    r624_z = tl.arange($(65536))
    r624_value = (tl.load($(candidate_vadd_60_ptr) + ((((r624_z / $(512)) % $(128)) * $(512)) + (((r624_z / $(64)) % $(8)) * $(64)) + (r624_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_15_ptr) + ((((r624_z / $(512)) % $(128)) * $(512)) + (((r624_z / $(64)) % $(8)) * $(64)) + (r624_z % $(64))), dtype=tl.bfloat16))
    r624_value = r624_value.to(tl.bfloat16)
    r624_value = tl.broadcast_to(r624_value, [$(65536)])
    tl.store($(candidate_vadd_71_ptr) + r624_z, r624_value)
    r625_z = tl.arange($(73728))
    r625_value = tl.where((r625_z % $(576)) < $(512), tl.load($(candidate_vadd_68_ptr) + ((((r625_z / $(576)) % $(128)) * $(512)) + (((r625_z % $(576)) - $(0)) % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vadd_69_ptr) + ((((r625_z / $(576)) % $(128)) * $(64)) + (((r625_z % $(576)) - $(512)) % $(64))), dtype=tl.bfloat16))
    r625_value = r625_value.to(tl.bfloat16)
    r625_value = tl.broadcast_to(r625_value, [$(73728)])
    tl.store($(candidate_vcat_61_ptr) + r625_z, r625_value)
    r626_z = tl.arange($(73728))
    r626_value = tl.cast(0.0, tl.bfloat16)
    for r626_lane in range($(128)) {
      r626_value = tl.where(tl.load($(candidate_vsort_index_ptr) + (r626_lane % $(128)), dtype=tl.int64) == ((r626_z / $(576)) % $(128)), r626_value + tl.load($(candidate_vcat_61_ptr) + (((r626_lane % $(128)) * $(576)) + (r626_z % $(576))), dtype=tl.bfloat16), r626_value)
    }
    r626_value = r626_value.to(tl.bfloat16)
    r626_value = tl.broadcast_to(r626_value, [$(73728)])
    tl.store($(candidate_vindex_put_1_ptr) + r626_z, r626_value)
    r627_z = tl.arange($(8192))
    r627_value = (tl.load($(candidate_vindex_put_1_ptr) + ((((r627_z / $(64)) % $(128)) * $(576)) + (r627_z % $(64)) + $(512)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_28_ptr) + ((((r627_z / $(64)) % $(128)) * $(64)) + (r627_z % $(64))), dtype=tl.bfloat16))
    r627_value = r627_value.to(tl.bfloat16)
    r627_value = tl.broadcast_to(r627_value, [$(8192)])
    tl.store($(candidate_vmul_107_ptr) + r627_z, r627_value)
    r628_z = tl.arange($(8192))
    r628_value = (tl.load($(candidate_vindex_put_1_ptr) + ((((r628_z / $(64)) % $(128)) * $(576)) + (r628_z % $(64)) + $(512)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_27_ptr) + ((((r628_z / $(64)) % $(128)) * $(64)) + (r628_z % $(64))), dtype=tl.bfloat16))
    r628_value = r628_value.to(tl.bfloat16)
    r628_value = tl.broadcast_to(r628_value, [$(8192)])
    tl.store($(candidate_vmul_108_ptr) + r628_z, r628_value)
    r629_z = tl.arange($(8192))
    r629_value = tl.where(((r629_z % $(2)) == $(0)), tl.load($(candidate_vmul_107_ptr) + ((((r629_z / $(64)) % $(128)) * $(64)) + (((r629_z / $(2)) % $(32)) * $(2)) + $(1)), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r629_value = r629_value.to(tl.bfloat16)
    r629_value = tl.broadcast_to(r629_value, [$(8192)])
    tl.store($(candidate_vselect_backward_ptr) + r629_z, r629_value)
    r630_z = tl.arange($(4096))
    r630_value = -tl.load($(candidate_vmul_107_ptr) + ((((r630_z / $(32)) % $(128)) * $(64)) + ((r630_z % $(32)) * $(2))), dtype=tl.bfloat16)
    r630_value = r630_value.to(tl.bfloat16)
    r630_value = tl.broadcast_to(r630_value, [$(4096)])
    tl.store($(candidate_vneg_2_ptr) + r630_z, r630_value)
    r631_z = tl.arange($(8192))
    r631_value = tl.where(((r631_z % $(2)) == $(1)), tl.load($(candidate_vneg_2_ptr) + ((((r631_z / $(64)) % $(128)) * $(32)) + ((r631_z / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r631_value = r631_value.to(tl.bfloat16)
    r631_value = tl.broadcast_to(r631_value, [$(8192)])
    tl.store($(candidate_vselect_backward_1_ptr) + r631_z, r631_value)
    r632_z = tl.arange($(8192))
    r632_value = (tl.load($(candidate_vselect_backward_ptr) + ((((r632_z / $(64)) % $(128)) * $(64)) + (((r632_z / $(2)) % $(32)) * $(2)) + (r632_z % $(2))), dtype=tl.bfloat16) + tl.load($(candidate_vselect_backward_1_ptr) + ((((r632_z / $(64)) % $(128)) * $(64)) + (((r632_z / $(2)) % $(32)) * $(2)) + (r632_z % $(2))), dtype=tl.bfloat16))
    r632_value = r632_value.to(tl.bfloat16)
    r632_value = tl.broadcast_to(r632_value, [$(8192)])
    tl.store($(candidate_vadd_72_ptr) + r632_z, r632_value)
    r633_z = tl.arange($(8192))
    r633_value = (tl.load($(candidate_vmul_108_ptr) + ((((r633_z / $(64)) % $(128)) * $(64)) + (r633_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vadd_72_ptr) + ((((r633_z / $(64)) % $(128)) * $(64)) + (((((((r633_z / $(64)) % $(128)) * $(64)) + (r633_z % $(64))) / $(2)) % $(32)) * $(2)) + (((((r633_z / $(64)) % $(128)) * $(64)) + (r633_z % $(64))) % $(2))), dtype=tl.bfloat16))
    r633_value = r633_value.to(tl.bfloat16)
    r633_value = tl.broadcast_to(r633_value, [$(8192)])
    tl.store($(candidate_vadd_73_ptr) + r633_z, r633_value)
    r634_z = tl.arange($(65536))
    r634_value = (tl.load($(candidate_vadd_71_ptr) + ((((r634_z / $(512)) % $(128)) * $(512)) + (((r634_z / $(64)) % $(8)) * $(64)) + (r634_z % $(64))), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_26_ptr) + ((((r634_z / $(512)) % $(128)) * $(64)) + (r634_z % $(64))), dtype=tl.bfloat16))
    r634_value = r634_value.to(tl.bfloat16)
    r634_value = tl.broadcast_to(r634_value, [$(65536)])
    tl.store($(candidate_vmul_109_ptr) + r634_z, r634_value)
    r635_z = tl.arange($(65536))
    r635_value = (tl.load($(candidate_vadd_71_ptr) + ((((r635_z / $(512)) % $(128)) * $(512)) + (((r635_z / $(64)) % $(8)) * $(64)) + (r635_z % $(64))), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_25_ptr) + ((((r635_z / $(512)) % $(128)) * $(64)) + (r635_z % $(64))), dtype=tl.bfloat16))
    r635_value = r635_value.to(tl.bfloat16)
    r635_value = tl.broadcast_to(r635_value, [$(65536)])
    tl.store($(candidate_vmul_110_ptr) + r635_z, r635_value)
    r636_z = tl.arange($(65536))
    r636_value = tl.where(((r636_z % $(2)) == $(0)), tl.load($(candidate_vmul_109_ptr) + ((((r636_z / $(512)) % $(128)) * $(512)) + (((r636_z / $(64)) % $(8)) * $(64)) + (((r636_z / $(2)) % $(32)) * $(2)) + $(1)), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r636_value = r636_value.to(tl.bfloat16)
    r636_value = tl.broadcast_to(r636_value, [$(65536)])
    tl.store($(candidate_vselect_backward_2_ptr) + r636_z, r636_value)
    r637_z = tl.arange($(32768))
    r637_value = -tl.load($(candidate_vmul_109_ptr) + ((((r637_z / $(256)) % $(128)) * $(512)) + (((r637_z / $(32)) % $(8)) * $(64)) + ((r637_z % $(32)) * $(2))), dtype=tl.bfloat16)
    r637_value = r637_value.to(tl.bfloat16)
    r637_value = tl.broadcast_to(r637_value, [$(32768)])
    tl.store($(candidate_vneg_3_ptr) + r637_z, r637_value)
    r638_z = tl.arange($(65536))
    r638_value = tl.where(((r638_z % $(2)) == $(1)), tl.load($(candidate_vneg_3_ptr) + ((((r638_z / $(512)) % $(128)) * $(256)) + (((r638_z / $(64)) % $(8)) * $(32)) + ((r638_z / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r638_value = r638_value.to(tl.bfloat16)
    r638_value = tl.broadcast_to(r638_value, [$(65536)])
    tl.store($(candidate_vselect_backward_3_ptr) + r638_z, r638_value)
    r639_z = tl.arange($(65536))
    r639_value = (tl.load($(candidate_vselect_backward_2_ptr) + ((((r639_z / $(512)) % $(128)) * $(512)) + (((r639_z / $(64)) % $(8)) * $(64)) + (((r639_z / $(2)) % $(32)) * $(2)) + (r639_z % $(2))), dtype=tl.bfloat16) + tl.load($(candidate_vselect_backward_3_ptr) + ((((r639_z / $(512)) % $(128)) * $(512)) + (((r639_z / $(64)) % $(8)) * $(64)) + (((r639_z / $(2)) % $(32)) * $(2)) + (r639_z % $(2))), dtype=tl.bfloat16))
    r639_value = r639_value.to(tl.bfloat16)
    r639_value = tl.broadcast_to(r639_value, [$(65536)])
    tl.store($(candidate_vadd_74_ptr) + r639_z, r639_value)
    r640_z = tl.arange($(65536))
    r640_value = (tl.load($(candidate_vmul_110_ptr) + ((((r640_z / $(512)) % $(128)) * $(512)) + (((r640_z / $(64)) % $(8)) * $(64)) + (r640_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vadd_74_ptr) + ((((r640_z / $(512)) % $(128)) * $(512)) + (((r640_z / $(64)) % $(8)) * $(64)) + (((((((r640_z / $(512)) % $(128)) * $(512)) + (((r640_z / $(64)) % $(8)) * $(64)) + (r640_z % $(64))) / $(2)) % $(32)) * $(2)) + (((((r640_z / $(512)) % $(128)) * $(512)) + (((r640_z / $(64)) % $(8)) * $(64)) + (r640_z % $(64))) % $(2))), dtype=tl.bfloat16))
    r640_value = r640_value.to(tl.bfloat16)
    r640_value = tl.broadcast_to(r640_value, [$(65536)])
    tl.store($(candidate_vadd_75_ptr) + r640_z, r640_value)
    r641_z = tl.arange($(196608))
    r641_value = tl.where((r641_z % $(192)) < $(128), tl.load($(candidate_vadd_70_ptr) + ((((((((r641_z / $(192)) % $(128)) * $(1024)) + (((r641_z / $(24576)) % $(8)) * $(128)) + ((r641_z % $(192)) - $(0))) / $(1024)) % $(128)) * $(1024)) + (((((((r641_z / $(192)) % $(128)) * $(1024)) + (((r641_z / $(24576)) % $(8)) * $(128)) + ((r641_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r641_z / $(192)) % $(128)) * $(1024)) + (((r641_z / $(24576)) % $(8)) * $(128)) + ((r641_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vadd_75_ptr) + ((((r641_z / $(192)) % $(128)) * $(512)) + (((r641_z / $(24576)) % $(8)) * $(64)) + (((r641_z % $(192)) - $(128)) % $(64))), dtype=tl.bfloat16))
    r641_value = r641_value.to(tl.bfloat16)
    r641_value = tl.broadcast_to(r641_value, [$(196608)])
    tl.store($(candidate_vcat_62_ptr) + r641_z, r641_value)
    r642_z = tl.arange($(65536))
    r642_value = tl.load($(candidate_vindex_put_1_ptr) + ((((r642_z / $(512)) % $(128)) * $(576)) + (r642_z % $(512))), dtype=tl.bfloat16)
    r642_value = r642_value.to(tl.float32)
    r642_value = tl.broadcast_to(r642_value, [$(65536)])
    tl.store($(candidate_vto_copy_221_ptr) + r642_z, r642_value)
    r643_z = tl.arange($(65536))
    r643_value = (tl.load($(candidate_vto_copy_221_ptr) + ((((r643_z / $(512)) % $(128)) * $(512)) + (r643_z % $(512))), dtype=tl.float32) * tl.cast(tl.load($(candidate_vto_copy_4_ptr) + (r643_z % $(512)), dtype=tl.bfloat16), tl.float32))
    r643_value = r643_value.to(tl.float32)
    r643_value = tl.broadcast_to(r643_value, [$(65536)])
    tl.store($(candidate_vmul_111_ptr) + r643_z, r643_value)
    r644_z = tl.arange($(65536))
    r644_value = (tl.load($(candidate_vmul_111_ptr) + ((((r644_z / $(512)) % $(128)) * $(512)) + (r644_z % $(512))), dtype=tl.float32) * tl.load($(candidate_vto_copy_22_ptr) + ((((r644_z / $(512)) % $(128)) * $(512)) + (r644_z % $(512))), dtype=tl.float32))
    r644_value = r644_value.to(tl.float32)
    r644_value = tl.broadcast_to(r644_value, [$(65536)])
    tl.store($(candidate_vmul_112_ptr) + r644_z, r644_value)
    r645_z = tl.arange($(65536))
    r645_value = (tl.load($(candidate_vmul_111_ptr) + ((((r645_z / $(512)) % $(128)) * $(512)) + (r645_z % $(512))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_1_ptr) + ((r645_z / $(512)) % $(128)), dtype=tl.float32))
    r645_value = r645_value.to(tl.float32)
    r645_value = tl.broadcast_to(r645_value, [$(65536)])
    tl.store($(candidate_vmul_113_ptr) + r645_z, r645_value)
    r646_z = tl.arange($(128))
    r646_lane = tl.arange($(512))
    r646_value = tl.sum(tl.load($(candidate_vmul_112_ptr) + ((r646_z[:, None] * $(512)) + r646_lane[None, :]), dtype=tl.float32), axis=1)
    r646_value = r646_value.to(tl.float32)
    r646_value = tl.broadcast_to(r646_value, [$(128)])
    tl.store($(candidate_vsum_17_ptr) + r646_z, r646_value)
    r647_z = tl.arange($(128))
    r647_value = ((tl.load($(candidate_vrsqrt_1_ptr) + r647_z, dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_1_ptr) + r647_z, dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_1_ptr) + r647_z, dtype=tl.float32)))
    r647_value = r647_value.to(tl.float32)
    r647_value = tl.broadcast_to(r647_value, [$(128)])
    tl.store($(candidate_vpow_6_ptr) + r647_z, r647_value)
    r648_z = tl.arange($(128))
    r648_value = (tl.load($(candidate_vsum_17_ptr) + r648_z, dtype=tl.float32) * -0.5)
    r648_value = r648_value.to(tl.float32)
    r648_value = tl.broadcast_to(r648_value, [$(128)])
    tl.store($(candidate_vmul_114_ptr) + r648_z, r648_value)
    r649_z = tl.arange($(128))
    r649_value = (tl.load($(candidate_vmul_114_ptr) + r649_z, dtype=tl.float32) * tl.load($(candidate_vpow_6_ptr) + r649_z, dtype=tl.float32))
    r649_value = r649_value.to(tl.float32)
    r649_value = tl.broadcast_to(r649_value, [$(128)])
    tl.store($(candidate_vmul_115_ptr) + r649_z, r649_value)
    r650_z = tl.arange($(65536))
    r650_value = (tl.load($(candidate_vmul_115_ptr) + ((r650_z / $(512)) % $(128)), dtype=tl.float32) / 512.0)
    r650_value = r650_value.to(tl.float32)
    r650_value = tl.broadcast_to(r650_value, [$(65536)])
    tl.store($(candidate_vdiv_3_ptr) + r650_z, r650_value)
    r651_z = tl.arange($(65536))
    r651_value = ((tl.load($(candidate_vto_copy_22_ptr) + ((((r651_z / $(512)) % $(128)) * $(512)) + (r651_z % $(512))), dtype=tl.float32)))
    r651_value = r651_value.to(tl.float32)
    r651_value = tl.broadcast_to(r651_value, [$(65536)])
    tl.store($(candidate_vpow_7_ptr) + r651_z, r651_value)
    r652_z = tl.arange($(65536))
    r652_value = (tl.load($(candidate_vpow_7_ptr) + ((((r652_z / $(512)) % $(128)) * $(512)) + (r652_z % $(512))), dtype=tl.float32) * 2.0)
    r652_value = r652_value.to(tl.float32)
    r652_value = tl.broadcast_to(r652_value, [$(65536)])
    tl.store($(candidate_vmul_116_ptr) + r652_z, r652_value)
    r653_z = tl.arange($(65536))
    r653_value = (tl.load($(candidate_vdiv_3_ptr) + ((((r653_z / $(512)) % $(128)) * $(512)) + (r653_z % $(512))), dtype=tl.float32) * tl.load($(candidate_vmul_116_ptr) + ((((r653_z / $(512)) % $(128)) * $(512)) + (r653_z % $(512))), dtype=tl.float32))
    r653_value = r653_value.to(tl.float32)
    r653_value = tl.broadcast_to(r653_value, [$(65536)])
    tl.store($(candidate_vmul_117_ptr) + r653_z, r653_value)
    r654_z = tl.arange($(65536))
    r654_value = (tl.load($(candidate_vmul_113_ptr) + ((((r654_z / $(512)) % $(128)) * $(512)) + (r654_z % $(512))), dtype=tl.float32) + tl.load($(candidate_vmul_117_ptr) + ((((r654_z / $(512)) % $(128)) * $(512)) + (r654_z % $(512))), dtype=tl.float32))
    r654_value = r654_value.to(tl.float32)
    r654_value = tl.broadcast_to(r654_value, [$(65536)])
    tl.store($(candidate_vadd_76_ptr) + r654_z, r654_value)
    r655_z = tl.arange($(65536))
    r655_value = tl.load($(candidate_vadd_76_ptr) + ((((r655_z / $(512)) % $(128)) * $(512)) + (r655_z % $(512))), dtype=tl.float32)
    r655_value = r655_value.to(tl.bfloat16)
    r655_value = tl.broadcast_to(r655_value, [$(65536)])
    tl.store($(candidate_vto_copy_222_ptr) + r655_z, r655_value)
    r656_z = tl.arange($(73728))
    r656_value = tl.where((r656_z % $(576)) < $(512), tl.load($(candidate_vto_copy_222_ptr) + ((((r656_z / $(576)) % $(128)) * $(512)) + (((r656_z % $(576)) - $(0)) % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vadd_73_ptr) + ((((((((r656_z / $(576)) % $(128)) * $(64)) + ((r656_z % $(576)) - $(512))) / $(64)) % $(128)) * $(64)) + (((((r656_z / $(576)) % $(128)) * $(64)) + ((r656_z % $(576)) - $(512))) % $(64))), dtype=tl.bfloat16))
    r656_value = r656_value.to(tl.bfloat16)
    r656_value = tl.broadcast_to(r656_value, [$(73728)])
    tl.store($(candidate_vcat_63_ptr) + r656_z, r656_value)
    r657_row = tl.arange($(128))
    r657_col = tl.arange($(512))
    r657_lane = tl.arange($(576))
    r657_left = tl.load($(candidate_vcat_63_ptr) + ((((((r657_row[:, None] * $(576)) + r657_lane[None, :]) / $(576)) % $(128)) * $(576)) + (((r657_row[:, None] * $(576)) + r657_lane[None, :]) % $(576))), dtype=tl.bfloat16)
    r657_right = tl.load($(candidate_vto_copy_3_ptr) + (((r657_lane[:, None] % $(576)) * $(512)) + (r657_col[None, :] % $(512))), dtype=tl.bfloat16)
    r657_value = tl.dot(r657_left, r657_right)
    r657_value = r657_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_29_ptr) + ((r657_row[:, None] * $(512)) + r657_col[None, :]), r657_value)
    r658_row = tl.arange($(128))
    r658_col = tl.arange($(256))
    r658_lane = tl.arange($(1536))
    r658_left = tl.load($(candidate_vcat_62_ptr) + ((((((((((r658_row[:, None] * $(1536)) + r658_lane[None, :]) / $(1536)) % $(128)) * $(1536)) + (((r658_row[:, None] * $(1536)) + r658_lane[None, :]) % $(1536))) / $(192)) % $(8)) * $(24576)) + (((((r658_row[:, None] * $(1536)) + r658_lane[None, :]) / $(1536)) % $(128)) * $(192)) + (((((((r658_row[:, None] * $(1536)) + r658_lane[None, :]) / $(1536)) % $(128)) * $(1536)) + (((r658_row[:, None] * $(1536)) + r658_lane[None, :]) % $(1536))) % $(192))), dtype=tl.bfloat16)
    r658_right = tl.load($(candidate_vto_copy_2_ptr) + (((r658_lane[:, None] % $(1536)) * $(256)) + (r658_col[None, :] % $(256))), dtype=tl.bfloat16)
    r658_value = tl.dot(r658_left, r658_right)
    r658_value = r658_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_30_ptr) + ((r658_row[:, None] * $(256)) + r658_col[None, :]), r658_value)
    r659_z = tl.arange($(32768))
    r659_value = tl.load($(candidate_vmm_30_ptr) + ((((r659_z / $(256)) % $(128)) * $(256)) + (r659_z % $(256))), dtype=tl.bfloat16)
    r659_value = r659_value.to(tl.float32)
    r659_value = tl.broadcast_to(r659_value, [$(32768)])
    tl.store($(candidate_vto_copy_223_ptr) + r659_z, r659_value)
    r660_z = tl.arange($(32768))
    r660_value = (tl.load($(candidate_vto_copy_223_ptr) + ((((r660_z / $(256)) % $(128)) * $(256)) + (r660_z % $(256))), dtype=tl.float32) * tl.cast(tl.load($(candidate_vto_copy_1_ptr) + (r660_z % $(256)), dtype=tl.bfloat16), tl.float32))
    r660_value = r660_value.to(tl.float32)
    r660_value = tl.broadcast_to(r660_value, [$(32768)])
    tl.store($(candidate_vmul_118_ptr) + r660_z, r660_value)
    r661_z = tl.arange($(32768))
    r661_value = (tl.load($(candidate_vmul_118_ptr) + ((((r661_z / $(256)) % $(128)) * $(256)) + (r661_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vto_copy_20_ptr) + ((((r661_z / $(256)) % $(128)) * $(256)) + (r661_z % $(256))), dtype=tl.float32))
    r661_value = r661_value.to(tl.float32)
    r661_value = tl.broadcast_to(r661_value, [$(32768)])
    tl.store($(candidate_vmul_119_ptr) + r661_z, r661_value)
    r662_z = tl.arange($(32768))
    r662_value = (tl.load($(candidate_vmul_118_ptr) + ((((r662_z / $(256)) % $(128)) * $(256)) + (r662_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_ptr) + ((r662_z / $(256)) % $(128)), dtype=tl.float32))
    r662_value = r662_value.to(tl.float32)
    r662_value = tl.broadcast_to(r662_value, [$(32768)])
    tl.store($(candidate_vmul_120_ptr) + r662_z, r662_value)
    r663_z = tl.arange($(128))
    r663_lane = tl.arange($(256))
    r663_value = tl.sum(tl.load($(candidate_vmul_119_ptr) + ((r663_z[:, None] * $(256)) + r663_lane[None, :]), dtype=tl.float32), axis=1)
    r663_value = r663_value.to(tl.float32)
    r663_value = tl.broadcast_to(r663_value, [$(128)])
    tl.store($(candidate_vsum_18_ptr) + r663_z, r663_value)
    r664_z = tl.arange($(128))
    r664_value = ((tl.load($(candidate_vrsqrt_ptr) + r664_z, dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_ptr) + r664_z, dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_ptr) + r664_z, dtype=tl.float32)))
    r664_value = r664_value.to(tl.float32)
    r664_value = tl.broadcast_to(r664_value, [$(128)])
    tl.store($(candidate_vpow_8_ptr) + r664_z, r664_value)
    r665_z = tl.arange($(128))
    r665_value = (tl.load($(candidate_vsum_18_ptr) + r665_z, dtype=tl.float32) * -0.5)
    r665_value = r665_value.to(tl.float32)
    r665_value = tl.broadcast_to(r665_value, [$(128)])
    tl.store($(candidate_vmul_121_ptr) + r665_z, r665_value)
    r666_z = tl.arange($(128))
    r666_value = (tl.load($(candidate_vmul_121_ptr) + r666_z, dtype=tl.float32) * tl.load($(candidate_vpow_8_ptr) + r666_z, dtype=tl.float32))
    r666_value = r666_value.to(tl.float32)
    r666_value = tl.broadcast_to(r666_value, [$(128)])
    tl.store($(candidate_vmul_122_ptr) + r666_z, r666_value)
    r667_z = tl.arange($(32768))
    r667_value = (tl.load($(candidate_vmul_122_ptr) + ((r667_z / $(256)) % $(128)), dtype=tl.float32) / 256.0)
    r667_value = r667_value.to(tl.float32)
    r667_value = tl.broadcast_to(r667_value, [$(32768)])
    tl.store($(candidate_vdiv_4_ptr) + r667_z, r667_value)
    r668_z = tl.arange($(32768))
    r668_value = ((tl.load($(candidate_vto_copy_20_ptr) + ((((r668_z / $(256)) % $(128)) * $(256)) + (r668_z % $(256))), dtype=tl.float32)))
    r668_value = r668_value.to(tl.float32)
    r668_value = tl.broadcast_to(r668_value, [$(32768)])
    tl.store($(candidate_vpow_9_ptr) + r668_z, r668_value)
    r669_z = tl.arange($(32768))
    r669_value = (tl.load($(candidate_vpow_9_ptr) + ((((r669_z / $(256)) % $(128)) * $(256)) + (r669_z % $(256))), dtype=tl.float32) * 2.0)
    r669_value = r669_value.to(tl.float32)
    r669_value = tl.broadcast_to(r669_value, [$(32768)])
    tl.store($(candidate_vmul_123_ptr) + r669_z, r669_value)
    r670_z = tl.arange($(32768))
    r670_value = (tl.load($(candidate_vdiv_4_ptr) + ((((r670_z / $(256)) % $(128)) * $(256)) + (r670_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vmul_123_ptr) + ((((r670_z / $(256)) % $(128)) * $(256)) + (r670_z % $(256))), dtype=tl.float32))
    r670_value = r670_value.to(tl.float32)
    r670_value = tl.broadcast_to(r670_value, [$(32768)])
    tl.store($(candidate_vmul_124_ptr) + r670_z, r670_value)
    r671_z = tl.arange($(32768))
    r671_value = (tl.load($(candidate_vmul_120_ptr) + ((((r671_z / $(256)) % $(128)) * $(256)) + (r671_z % $(256))), dtype=tl.float32) + tl.load($(candidate_vmul_124_ptr) + ((((r671_z / $(256)) % $(128)) * $(256)) + (r671_z % $(256))), dtype=tl.float32))
    r671_value = r671_value.to(tl.float32)
    r671_value = tl.broadcast_to(r671_value, [$(32768)])
    tl.store($(candidate_vadd_77_ptr) + r671_z, r671_value)
    r672_z = tl.arange($(32768))
    r672_value = tl.load($(candidate_vadd_77_ptr) + ((((r672_z / $(256)) % $(128)) * $(256)) + (r672_z % $(256))), dtype=tl.float32)
    r672_value = r672_value.to(tl.bfloat16)
    r672_value = tl.broadcast_to(r672_value, [$(32768)])
    tl.store($(candidate_vto_copy_224_ptr) + r672_z, r672_value)
    r673_row = tl.arange($(128))
    r673_col = tl.arange($(512))
    r673_lane = tl.arange($(256))
    r673_left = tl.load($(candidate_vto_copy_224_ptr) + ((((((r673_row[:, None] * $(256)) + r673_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r673_row[:, None] * $(256)) + r673_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r673_right = tl.load($(candidate_vto_copy_ptr) + (((r673_lane[:, None] % $(256)) * $(512)) + (r673_col[None, :] % $(512))), dtype=tl.bfloat16)
    r673_value = tl.dot(r673_left, r673_right)
    r673_value = r673_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_31_ptr) + ((r673_row[:, None] * $(512)) + r673_col[None, :]), r673_value)
    r674_z = tl.arange($(65536))
    r674_value = (tl.load($(candidate_vmm_29_ptr) + ((((r674_z / $(512)) % $(128)) * $(512)) + (r674_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vmm_31_ptr) + ((((r674_z / $(512)) % $(128)) * $(512)) + (r674_z % $(512))), dtype=tl.bfloat16))
    r674_value = r674_value.to(tl.bfloat16)
    r674_value = tl.broadcast_to(r674_value, [$(65536)])
    tl.store($(candidate_vadd_78_ptr) + r674_z, r674_value)
    r675_z = tl.arange($(65536))
    r675_value = tl.cast(0.0, tl.bfloat16)
    for r675_lane in range($(128)) {
      r675_value = tl.where(tl.load($(constantee33028f0f_ptr) + r675_lane, dtype=tl.int64) == ((r675_z / $(512)) % $(128)), r675_value + tl.load($(candidate_vadd_78_ptr) + (((r675_lane % $(128)) * $(512)) + (r675_z % $(512))), dtype=tl.bfloat16), r675_value)
    }
    r675_value = r675_value.to(tl.bfloat16)
    r675_value = tl.broadcast_to(r675_value, [$(65536)])
    tl.store($(candidate_vindex_put_2_ptr) + r675_z, r675_value)
    z = tl.arange($(65536))
    value = tl.load($(candidate_vindex_put_2_ptr) + ((((z / $(512)) % $(128)) * $(512)) + (z % $(512))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(65536)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_mla_fa4_te_rope_cp_absorb_dx
