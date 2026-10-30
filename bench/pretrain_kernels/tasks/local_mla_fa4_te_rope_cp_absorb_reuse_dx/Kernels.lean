import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_mla_fa4_te_rope_cp_absorb_reuse_dx
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
  let candidate_vto_copy_29_ptr : RegionName := "candidate_vto_copy_29"
  let candidate_vto_copy_30_ptr : RegionName := "candidate_vto_copy_30"
  let candidate_vbmm_3_ptr : RegionName := "candidate_vbmm_3"
  let candidate_vto_copy_31_ptr : RegionName := "candidate_vto_copy_31"
  let candidate_vto_copy_32_ptr : RegionName := "candidate_vto_copy_32"
  let candidate_vmm_3_ptr : RegionName := "candidate_vmm_3"
  let candidate_vadd_8_ptr : RegionName := "candidate_vadd_8"
  let candidate_vmul_23_ptr : RegionName := "candidate_vmul_23"
  let candidate_vadd_9_ptr : RegionName := "candidate_vadd_9"
  let candidate_vsub_9_ptr : RegionName := "candidate_vsub_9"
  let candidate_vgt_ptr : RegionName := "candidate_vgt"
  let candidate_vmasked_fill_ptr : RegionName := "candidate_vmasked_fill"
  let candidate_vsoftmax_ptr : RegionName := "candidate_vsoftmax"
  let candidate_vto_copy_33_ptr : RegionName := "candidate_vto_copy_33"
  let candidate_vmm_4_ptr : RegionName := "candidate_vmm_4"
  let candidate_vlogsumexp_ptr : RegionName := "candidate_vlogsumexp"
  let candidate_vto_copy_34_ptr : RegionName := "candidate_vto_copy_34"
  let candidate_vto_copy_35_ptr : RegionName := "candidate_vto_copy_35"
  let candidate_vbmm_4_ptr : RegionName := "candidate_vbmm_4"
  let candidate_vto_copy_36_ptr : RegionName := "candidate_vto_copy_36"
  let candidate_vto_copy_37_ptr : RegionName := "candidate_vto_copy_37"
  let candidate_vmm_5_ptr : RegionName := "candidate_vmm_5"
  let candidate_vadd_10_ptr : RegionName := "candidate_vadd_10"
  let candidate_vmul_24_ptr : RegionName := "candidate_vmul_24"
  let candidate_vadd_11_ptr : RegionName := "candidate_vadd_11"
  let candidate_vsub_10_ptr : RegionName := "candidate_vsub_10"
  let candidate_vgt_1_ptr : RegionName := "candidate_vgt_1"
  let candidate_vmasked_fill_1_ptr : RegionName := "candidate_vmasked_fill_1"
  let candidate_vsoftmax_1_ptr : RegionName := "candidate_vsoftmax_1"
  let candidate_vto_copy_38_ptr : RegionName := "candidate_vto_copy_38"
  let candidate_vmm_6_ptr : RegionName := "candidate_vmm_6"
  let candidate_vlogsumexp_1_ptr : RegionName := "candidate_vlogsumexp_1"
  let candidate_vto_copy_39_ptr : RegionName := "candidate_vto_copy_39"
  let candidate_vto_copy_40_ptr : RegionName := "candidate_vto_copy_40"
  let candidate_vbmm_5_ptr : RegionName := "candidate_vbmm_5"
  let candidate_vto_copy_41_ptr : RegionName := "candidate_vto_copy_41"
  let candidate_vto_copy_42_ptr : RegionName := "candidate_vto_copy_42"
  let candidate_vmm_7_ptr : RegionName := "candidate_vmm_7"
  let candidate_vadd_12_ptr : RegionName := "candidate_vadd_12"
  let candidate_vmul_25_ptr : RegionName := "candidate_vmul_25"
  let candidate_vadd_13_ptr : RegionName := "candidate_vadd_13"
  let candidate_vsub_11_ptr : RegionName := "candidate_vsub_11"
  let candidate_vgt_2_ptr : RegionName := "candidate_vgt_2"
  let candidate_vmasked_fill_2_ptr : RegionName := "candidate_vmasked_fill_2"
  let candidate_vsoftmax_2_ptr : RegionName := "candidate_vsoftmax_2"
  let candidate_vto_copy_43_ptr : RegionName := "candidate_vto_copy_43"
  let candidate_vmm_8_ptr : RegionName := "candidate_vmm_8"
  let candidate_vlogsumexp_2_ptr : RegionName := "candidate_vlogsumexp_2"
  let candidate_vcat_6_ptr : RegionName := "candidate_vcat_6"
  let candidate_vcat_7_ptr : RegionName := "candidate_vcat_7"
  let candidate_vbmm_6_ptr : RegionName := "candidate_vbmm_6"
  let candidate_vbmm_7_ptr : RegionName := "candidate_vbmm_7"
  let candidate_vto_copy_44_ptr : RegionName := "candidate_vto_copy_44"
  let candidate_vto_copy_45_ptr : RegionName := "candidate_vto_copy_45"
  let candidate_vbmm_8_ptr : RegionName := "candidate_vbmm_8"
  let candidate_vto_copy_46_ptr : RegionName := "candidate_vto_copy_46"
  let candidate_vto_copy_47_ptr : RegionName := "candidate_vto_copy_47"
  let candidate_vmm_9_ptr : RegionName := "candidate_vmm_9"
  let candidate_vadd_14_ptr : RegionName := "candidate_vadd_14"
  let candidate_vmul_26_ptr : RegionName := "candidate_vmul_26"
  let candidate_vadd_15_ptr : RegionName := "candidate_vadd_15"
  let candidate_vsub_12_ptr : RegionName := "candidate_vsub_12"
  let candidate_vgt_3_ptr : RegionName := "candidate_vgt_3"
  let candidate_vmasked_fill_3_ptr : RegionName := "candidate_vmasked_fill_3"
  let candidate_vsoftmax_3_ptr : RegionName := "candidate_vsoftmax_3"
  let candidate_vto_copy_48_ptr : RegionName := "candidate_vto_copy_48"
  let candidate_vmm_10_ptr : RegionName := "candidate_vmm_10"
  let candidate_vlogsumexp_3_ptr : RegionName := "candidate_vlogsumexp_3"
  let candidate_vto_copy_49_ptr : RegionName := "candidate_vto_copy_49"
  let candidate_vto_copy_50_ptr : RegionName := "candidate_vto_copy_50"
  let candidate_vbmm_9_ptr : RegionName := "candidate_vbmm_9"
  let candidate_vto_copy_51_ptr : RegionName := "candidate_vto_copy_51"
  let candidate_vto_copy_52_ptr : RegionName := "candidate_vto_copy_52"
  let candidate_vmm_11_ptr : RegionName := "candidate_vmm_11"
  let candidate_vadd_16_ptr : RegionName := "candidate_vadd_16"
  let candidate_vmul_27_ptr : RegionName := "candidate_vmul_27"
  let candidate_vadd_17_ptr : RegionName := "candidate_vadd_17"
  let candidate_vsub_13_ptr : RegionName := "candidate_vsub_13"
  let candidate_vgt_4_ptr : RegionName := "candidate_vgt_4"
  let candidate_vmasked_fill_4_ptr : RegionName := "candidate_vmasked_fill_4"
  let candidate_vsoftmax_4_ptr : RegionName := "candidate_vsoftmax_4"
  let candidate_vto_copy_53_ptr : RegionName := "candidate_vto_copy_53"
  let candidate_vmm_12_ptr : RegionName := "candidate_vmm_12"
  let candidate_vlogsumexp_4_ptr : RegionName := "candidate_vlogsumexp_4"
  let candidate_vto_copy_54_ptr : RegionName := "candidate_vto_copy_54"
  let candidate_vto_copy_55_ptr : RegionName := "candidate_vto_copy_55"
  let candidate_vbmm_10_ptr : RegionName := "candidate_vbmm_10"
  let candidate_vto_copy_56_ptr : RegionName := "candidate_vto_copy_56"
  let candidate_vto_copy_57_ptr : RegionName := "candidate_vto_copy_57"
  let candidate_vmm_13_ptr : RegionName := "candidate_vmm_13"
  let candidate_vadd_18_ptr : RegionName := "candidate_vadd_18"
  let candidate_vmul_28_ptr : RegionName := "candidate_vmul_28"
  let candidate_vadd_19_ptr : RegionName := "candidate_vadd_19"
  let candidate_vsub_14_ptr : RegionName := "candidate_vsub_14"
  let candidate_vgt_5_ptr : RegionName := "candidate_vgt_5"
  let candidate_vmasked_fill_5_ptr : RegionName := "candidate_vmasked_fill_5"
  let candidate_vsoftmax_5_ptr : RegionName := "candidate_vsoftmax_5"
  let candidate_vto_copy_58_ptr : RegionName := "candidate_vto_copy_58"
  let candidate_vmm_14_ptr : RegionName := "candidate_vmm_14"
  let candidate_vlogsumexp_5_ptr : RegionName := "candidate_vlogsumexp_5"
  let candidate_vcat_8_ptr : RegionName := "candidate_vcat_8"
  let candidate_vcat_9_ptr : RegionName := "candidate_vcat_9"
  let candidate_vbmm_11_ptr : RegionName := "candidate_vbmm_11"
  let candidate_vbmm_12_ptr : RegionName := "candidate_vbmm_12"
  let candidate_vto_copy_59_ptr : RegionName := "candidate_vto_copy_59"
  let candidate_vto_copy_60_ptr : RegionName := "candidate_vto_copy_60"
  let candidate_vbmm_13_ptr : RegionName := "candidate_vbmm_13"
  let candidate_vto_copy_61_ptr : RegionName := "candidate_vto_copy_61"
  let candidate_vto_copy_62_ptr : RegionName := "candidate_vto_copy_62"
  let candidate_vmm_15_ptr : RegionName := "candidate_vmm_15"
  let candidate_vadd_20_ptr : RegionName := "candidate_vadd_20"
  let candidate_vmul_29_ptr : RegionName := "candidate_vmul_29"
  let candidate_vadd_21_ptr : RegionName := "candidate_vadd_21"
  let candidate_vsub_15_ptr : RegionName := "candidate_vsub_15"
  let candidate_vgt_6_ptr : RegionName := "candidate_vgt_6"
  let candidate_vmasked_fill_6_ptr : RegionName := "candidate_vmasked_fill_6"
  let candidate_vsoftmax_6_ptr : RegionName := "candidate_vsoftmax_6"
  let candidate_vto_copy_63_ptr : RegionName := "candidate_vto_copy_63"
  let candidate_vmm_16_ptr : RegionName := "candidate_vmm_16"
  let candidate_vlogsumexp_6_ptr : RegionName := "candidate_vlogsumexp_6"
  let candidate_vto_copy_64_ptr : RegionName := "candidate_vto_copy_64"
  let candidate_vto_copy_65_ptr : RegionName := "candidate_vto_copy_65"
  let candidate_vbmm_14_ptr : RegionName := "candidate_vbmm_14"
  let candidate_vto_copy_66_ptr : RegionName := "candidate_vto_copy_66"
  let candidate_vto_copy_67_ptr : RegionName := "candidate_vto_copy_67"
  let candidate_vmm_17_ptr : RegionName := "candidate_vmm_17"
  let candidate_vadd_22_ptr : RegionName := "candidate_vadd_22"
  let candidate_vmul_30_ptr : RegionName := "candidate_vmul_30"
  let candidate_vadd_23_ptr : RegionName := "candidate_vadd_23"
  let candidate_vsub_16_ptr : RegionName := "candidate_vsub_16"
  let candidate_vgt_7_ptr : RegionName := "candidate_vgt_7"
  let candidate_vmasked_fill_7_ptr : RegionName := "candidate_vmasked_fill_7"
  let candidate_vsoftmax_7_ptr : RegionName := "candidate_vsoftmax_7"
  let candidate_vto_copy_68_ptr : RegionName := "candidate_vto_copy_68"
  let candidate_vmm_18_ptr : RegionName := "candidate_vmm_18"
  let candidate_vlogsumexp_7_ptr : RegionName := "candidate_vlogsumexp_7"
  let candidate_vto_copy_69_ptr : RegionName := "candidate_vto_copy_69"
  let candidate_vto_copy_70_ptr : RegionName := "candidate_vto_copy_70"
  let candidate_vbmm_15_ptr : RegionName := "candidate_vbmm_15"
  let candidate_vto_copy_71_ptr : RegionName := "candidate_vto_copy_71"
  let candidate_vto_copy_72_ptr : RegionName := "candidate_vto_copy_72"
  let candidate_vmm_19_ptr : RegionName := "candidate_vmm_19"
  let candidate_vadd_24_ptr : RegionName := "candidate_vadd_24"
  let candidate_vmul_31_ptr : RegionName := "candidate_vmul_31"
  let candidate_vadd_25_ptr : RegionName := "candidate_vadd_25"
  let candidate_vsub_17_ptr : RegionName := "candidate_vsub_17"
  let candidate_vgt_8_ptr : RegionName := "candidate_vgt_8"
  let candidate_vmasked_fill_8_ptr : RegionName := "candidate_vmasked_fill_8"
  let candidate_vsoftmax_8_ptr : RegionName := "candidate_vsoftmax_8"
  let candidate_vto_copy_73_ptr : RegionName := "candidate_vto_copy_73"
  let candidate_vmm_20_ptr : RegionName := "candidate_vmm_20"
  let candidate_vlogsumexp_8_ptr : RegionName := "candidate_vlogsumexp_8"
  let candidate_vcat_10_ptr : RegionName := "candidate_vcat_10"
  let candidate_vcat_11_ptr : RegionName := "candidate_vcat_11"
  let candidate_vbmm_16_ptr : RegionName := "candidate_vbmm_16"
  let candidate_vbmm_17_ptr : RegionName := "candidate_vbmm_17"
  let candidate_vto_copy_74_ptr : RegionName := "candidate_vto_copy_74"
  let candidate_vto_copy_75_ptr : RegionName := "candidate_vto_copy_75"
  let candidate_vbmm_18_ptr : RegionName := "candidate_vbmm_18"
  let candidate_vto_copy_76_ptr : RegionName := "candidate_vto_copy_76"
  let candidate_vto_copy_77_ptr : RegionName := "candidate_vto_copy_77"
  let candidate_vmm_21_ptr : RegionName := "candidate_vmm_21"
  let candidate_vadd_26_ptr : RegionName := "candidate_vadd_26"
  let candidate_vmul_32_ptr : RegionName := "candidate_vmul_32"
  let candidate_vadd_27_ptr : RegionName := "candidate_vadd_27"
  let candidate_vsub_18_ptr : RegionName := "candidate_vsub_18"
  let candidate_vgt_9_ptr : RegionName := "candidate_vgt_9"
  let candidate_vmasked_fill_9_ptr : RegionName := "candidate_vmasked_fill_9"
  let candidate_vsoftmax_9_ptr : RegionName := "candidate_vsoftmax_9"
  let candidate_vto_copy_78_ptr : RegionName := "candidate_vto_copy_78"
  let candidate_vmm_22_ptr : RegionName := "candidate_vmm_22"
  let candidate_vlogsumexp_9_ptr : RegionName := "candidate_vlogsumexp_9"
  let candidate_vto_copy_79_ptr : RegionName := "candidate_vto_copy_79"
  let candidate_vto_copy_80_ptr : RegionName := "candidate_vto_copy_80"
  let candidate_vbmm_19_ptr : RegionName := "candidate_vbmm_19"
  let candidate_vto_copy_81_ptr : RegionName := "candidate_vto_copy_81"
  let candidate_vto_copy_82_ptr : RegionName := "candidate_vto_copy_82"
  let candidate_vmm_23_ptr : RegionName := "candidate_vmm_23"
  let candidate_vadd_28_ptr : RegionName := "candidate_vadd_28"
  let candidate_vmul_33_ptr : RegionName := "candidate_vmul_33"
  let candidate_vadd_29_ptr : RegionName := "candidate_vadd_29"
  let candidate_vsub_19_ptr : RegionName := "candidate_vsub_19"
  let candidate_vgt_10_ptr : RegionName := "candidate_vgt_10"
  let candidate_vmasked_fill_10_ptr : RegionName := "candidate_vmasked_fill_10"
  let candidate_vsoftmax_10_ptr : RegionName := "candidate_vsoftmax_10"
  let candidate_vto_copy_83_ptr : RegionName := "candidate_vto_copy_83"
  let candidate_vmm_24_ptr : RegionName := "candidate_vmm_24"
  let candidate_vlogsumexp_10_ptr : RegionName := "candidate_vlogsumexp_10"
  let candidate_vto_copy_84_ptr : RegionName := "candidate_vto_copy_84"
  let candidate_vto_copy_85_ptr : RegionName := "candidate_vto_copy_85"
  let candidate_vbmm_20_ptr : RegionName := "candidate_vbmm_20"
  let candidate_vto_copy_86_ptr : RegionName := "candidate_vto_copy_86"
  let candidate_vto_copy_87_ptr : RegionName := "candidate_vto_copy_87"
  let candidate_vmm_25_ptr : RegionName := "candidate_vmm_25"
  let candidate_vadd_30_ptr : RegionName := "candidate_vadd_30"
  let candidate_vmul_34_ptr : RegionName := "candidate_vmul_34"
  let candidate_vadd_31_ptr : RegionName := "candidate_vadd_31"
  let candidate_vsub_20_ptr : RegionName := "candidate_vsub_20"
  let candidate_vgt_11_ptr : RegionName := "candidate_vgt_11"
  let candidate_vmasked_fill_11_ptr : RegionName := "candidate_vmasked_fill_11"
  let candidate_vsoftmax_11_ptr : RegionName := "candidate_vsoftmax_11"
  let candidate_vto_copy_88_ptr : RegionName := "candidate_vto_copy_88"
  let candidate_vmm_26_ptr : RegionName := "candidate_vmm_26"
  let candidate_vlogsumexp_11_ptr : RegionName := "candidate_vlogsumexp_11"
  let candidate_vcat_12_ptr : RegionName := "candidate_vcat_12"
  let candidate_vcat_13_ptr : RegionName := "candidate_vcat_13"
  let candidate_vbmm_21_ptr : RegionName := "candidate_vbmm_21"
  let candidate_vindex_put_ptr : RegionName := "candidate_vindex_put"
  let candidate_vmm_28_ptr : RegionName := "candidate_vmm_28"
  let candidate_vbmm_22_ptr : RegionName := "candidate_vbmm_22"
  let candidate_vbmm_23_ptr : RegionName := "candidate_vbmm_23"
  let candidate_vcat_15_ptr : RegionName := "candidate_vcat_15"
  let candidate_vcat_16_ptr : RegionName := "candidate_vcat_16"
  let candidate_vto_copy_89_ptr : RegionName := "candidate_vto_copy_89"
  let candidate_vto_copy_90_ptr : RegionName := "candidate_vto_copy_90"
  let candidate_vto_copy_91_ptr : RegionName := "candidate_vto_copy_91"
  let candidate_vto_copy_92_ptr : RegionName := "candidate_vto_copy_92"
  let candidate_vto_copy_93_ptr : RegionName := "candidate_vto_copy_93"
  let candidate_vbmm_24_ptr : RegionName := "candidate_vbmm_24"
  let candidate_vmul_35_ptr : RegionName := "candidate_vmul_35"
  let candidate_vadd_32_ptr : RegionName := "candidate_vadd_32"
  let candidate_vsub_21_ptr : RegionName := "candidate_vsub_21"
  let candidate_vgt_12_ptr : RegionName := "candidate_vgt_12"
  let candidate_vmasked_fill_12_ptr : RegionName := "candidate_vmasked_fill_12"
  let candidate_vsub_22_ptr : RegionName := "candidate_vsub_22"
  let candidate_vexp_ptr : RegionName := "candidate_vexp"
  let candidate_vmul_36_ptr : RegionName := "candidate_vmul_36"
  let candidate_vsum_1_ptr : RegionName := "candidate_vsum_1"
  let candidate_vbmm_25_ptr : RegionName := "candidate_vbmm_25"
  let candidate_vsub_23_ptr : RegionName := "candidate_vsub_23"
  let candidate_vmul_37_ptr : RegionName := "candidate_vmul_37"
  let candidate_vbmm_26_ptr : RegionName := "candidate_vbmm_26"
  let candidate_vmul_38_ptr : RegionName := "candidate_vmul_38"
  let candidate_vto_copy_94_ptr : RegionName := "candidate_vto_copy_94"
  let candidate_vbmm_27_ptr : RegionName := "candidate_vbmm_27"
  let candidate_vmul_39_ptr : RegionName := "candidate_vmul_39"
  let candidate_vto_copy_95_ptr : RegionName := "candidate_vto_copy_95"
  let candidate_vbmm_28_ptr : RegionName := "candidate_vbmm_28"
  let candidate_vto_copy_96_ptr : RegionName := "candidate_vto_copy_96"
  let candidate_vcat_17_ptr : RegionName := "candidate_vcat_17"
  let candidate_vcat_18_ptr : RegionName := "candidate_vcat_18"
  let candidate_vto_copy_97_ptr : RegionName := "candidate_vto_copy_97"
  let candidate_vto_copy_98_ptr : RegionName := "candidate_vto_copy_98"
  let candidate_vto_copy_99_ptr : RegionName := "candidate_vto_copy_99"
  let candidate_vto_copy_100_ptr : RegionName := "candidate_vto_copy_100"
  let candidate_vto_copy_101_ptr : RegionName := "candidate_vto_copy_101"
  let candidate_vbmm_29_ptr : RegionName := "candidate_vbmm_29"
  let candidate_vmul_40_ptr : RegionName := "candidate_vmul_40"
  let candidate_vadd_33_ptr : RegionName := "candidate_vadd_33"
  let candidate_vsub_24_ptr : RegionName := "candidate_vsub_24"
  let candidate_vgt_13_ptr : RegionName := "candidate_vgt_13"
  let candidate_vmasked_fill_13_ptr : RegionName := "candidate_vmasked_fill_13"
  let candidate_vsub_25_ptr : RegionName := "candidate_vsub_25"
  let candidate_vexp_1_ptr : RegionName := "candidate_vexp_1"
  let candidate_vmul_41_ptr : RegionName := "candidate_vmul_41"
  let candidate_vsum_2_ptr : RegionName := "candidate_vsum_2"
  let candidate_vbmm_30_ptr : RegionName := "candidate_vbmm_30"
  let candidate_vsub_26_ptr : RegionName := "candidate_vsub_26"
  let candidate_vmul_42_ptr : RegionName := "candidate_vmul_42"
  let candidate_vbmm_31_ptr : RegionName := "candidate_vbmm_31"
  let candidate_vmul_43_ptr : RegionName := "candidate_vmul_43"
  let candidate_vto_copy_102_ptr : RegionName := "candidate_vto_copy_102"
  let candidate_vbmm_32_ptr : RegionName := "candidate_vbmm_32"
  let candidate_vmul_44_ptr : RegionName := "candidate_vmul_44"
  let candidate_vto_copy_103_ptr : RegionName := "candidate_vto_copy_103"
  let candidate_vbmm_33_ptr : RegionName := "candidate_vbmm_33"
  let candidate_vto_copy_104_ptr : RegionName := "candidate_vto_copy_104"
  let candidate_vcat_19_ptr : RegionName := "candidate_vcat_19"
  let candidate_vcat_20_ptr : RegionName := "candidate_vcat_20"
  let candidate_vto_copy_105_ptr : RegionName := "candidate_vto_copy_105"
  let candidate_vto_copy_106_ptr : RegionName := "candidate_vto_copy_106"
  let candidate_vto_copy_107_ptr : RegionName := "candidate_vto_copy_107"
  let candidate_vto_copy_108_ptr : RegionName := "candidate_vto_copy_108"
  let candidate_vto_copy_109_ptr : RegionName := "candidate_vto_copy_109"
  let candidate_vbmm_34_ptr : RegionName := "candidate_vbmm_34"
  let candidate_vmul_45_ptr : RegionName := "candidate_vmul_45"
  let candidate_vadd_34_ptr : RegionName := "candidate_vadd_34"
  let candidate_vsub_27_ptr : RegionName := "candidate_vsub_27"
  let candidate_vgt_14_ptr : RegionName := "candidate_vgt_14"
  let candidate_vmasked_fill_14_ptr : RegionName := "candidate_vmasked_fill_14"
  let candidate_vsub_28_ptr : RegionName := "candidate_vsub_28"
  let candidate_vexp_2_ptr : RegionName := "candidate_vexp_2"
  let candidate_vmul_46_ptr : RegionName := "candidate_vmul_46"
  let candidate_vsum_3_ptr : RegionName := "candidate_vsum_3"
  let candidate_vbmm_35_ptr : RegionName := "candidate_vbmm_35"
  let candidate_vsub_29_ptr : RegionName := "candidate_vsub_29"
  let candidate_vmul_47_ptr : RegionName := "candidate_vmul_47"
  let candidate_vbmm_36_ptr : RegionName := "candidate_vbmm_36"
  let candidate_vmul_48_ptr : RegionName := "candidate_vmul_48"
  let candidate_vto_copy_110_ptr : RegionName := "candidate_vto_copy_110"
  let candidate_vbmm_37_ptr : RegionName := "candidate_vbmm_37"
  let candidate_vmul_49_ptr : RegionName := "candidate_vmul_49"
  let candidate_vto_copy_111_ptr : RegionName := "candidate_vto_copy_111"
  let candidate_vbmm_38_ptr : RegionName := "candidate_vbmm_38"
  let candidate_vto_copy_112_ptr : RegionName := "candidate_vto_copy_112"
  let candidate_vcat_21_ptr : RegionName := "candidate_vcat_21"
  let candidate_vcat_22_ptr : RegionName := "candidate_vcat_22"
  let candidate_vcat_23_ptr : RegionName := "candidate_vcat_23"
  let candidate_vcat_24_ptr : RegionName := "candidate_vcat_24"
  let candidate_vcat_25_ptr : RegionName := "candidate_vcat_25"
  let candidate_vsum_4_ptr : RegionName := "candidate_vsum_4"
  let candidate_vbmm_39_ptr : RegionName := "candidate_vbmm_39"
  let candidate_vbmm_40_ptr : RegionName := "candidate_vbmm_40"
  let candidate_vadd_35_ptr : RegionName := "candidate_vadd_35"
  let candidate_vslice_backward_ptr : RegionName := "candidate_vslice_backward"
  let candidate_vslice_backward_1_ptr : RegionName := "candidate_vslice_backward_1"
  let candidate_vslice_backward_2_ptr : RegionName := "candidate_vslice_backward_2"
  let candidate_vslice_backward_3_ptr : RegionName := "candidate_vslice_backward_3"
  let candidate_vbmm_43_ptr : RegionName := "candidate_vbmm_43"
  let candidate_vbmm_44_ptr : RegionName := "candidate_vbmm_44"
  let candidate_vcat_27_ptr : RegionName := "candidate_vcat_27"
  let candidate_vcat_28_ptr : RegionName := "candidate_vcat_28"
  let candidate_vto_copy_113_ptr : RegionName := "candidate_vto_copy_113"
  let candidate_vto_copy_114_ptr : RegionName := "candidate_vto_copy_114"
  let candidate_vto_copy_115_ptr : RegionName := "candidate_vto_copy_115"
  let candidate_vto_copy_116_ptr : RegionName := "candidate_vto_copy_116"
  let candidate_vto_copy_117_ptr : RegionName := "candidate_vto_copy_117"
  let candidate_vbmm_45_ptr : RegionName := "candidate_vbmm_45"
  let candidate_vmul_50_ptr : RegionName := "candidate_vmul_50"
  let candidate_vadd_36_ptr : RegionName := "candidate_vadd_36"
  let candidate_vsub_30_ptr : RegionName := "candidate_vsub_30"
  let candidate_vgt_15_ptr : RegionName := "candidate_vgt_15"
  let candidate_vmasked_fill_15_ptr : RegionName := "candidate_vmasked_fill_15"
  let candidate_vsub_31_ptr : RegionName := "candidate_vsub_31"
  let candidate_vexp_3_ptr : RegionName := "candidate_vexp_3"
  let candidate_vmul_51_ptr : RegionName := "candidate_vmul_51"
  let candidate_vsum_5_ptr : RegionName := "candidate_vsum_5"
  let candidate_vbmm_46_ptr : RegionName := "candidate_vbmm_46"
  let candidate_vsub_32_ptr : RegionName := "candidate_vsub_32"
  let candidate_vmul_52_ptr : RegionName := "candidate_vmul_52"
  let candidate_vbmm_47_ptr : RegionName := "candidate_vbmm_47"
  let candidate_vmul_53_ptr : RegionName := "candidate_vmul_53"
  let candidate_vto_copy_118_ptr : RegionName := "candidate_vto_copy_118"
  let candidate_vbmm_48_ptr : RegionName := "candidate_vbmm_48"
  let candidate_vmul_54_ptr : RegionName := "candidate_vmul_54"
  let candidate_vto_copy_119_ptr : RegionName := "candidate_vto_copy_119"
  let candidate_vbmm_49_ptr : RegionName := "candidate_vbmm_49"
  let candidate_vto_copy_120_ptr : RegionName := "candidate_vto_copy_120"
  let candidate_vcat_29_ptr : RegionName := "candidate_vcat_29"
  let candidate_vcat_30_ptr : RegionName := "candidate_vcat_30"
  let candidate_vto_copy_121_ptr : RegionName := "candidate_vto_copy_121"
  let candidate_vto_copy_122_ptr : RegionName := "candidate_vto_copy_122"
  let candidate_vto_copy_123_ptr : RegionName := "candidate_vto_copy_123"
  let candidate_vto_copy_124_ptr : RegionName := "candidate_vto_copy_124"
  let candidate_vto_copy_125_ptr : RegionName := "candidate_vto_copy_125"
  let candidate_vbmm_50_ptr : RegionName := "candidate_vbmm_50"
  let candidate_vmul_55_ptr : RegionName := "candidate_vmul_55"
  let candidate_vadd_37_ptr : RegionName := "candidate_vadd_37"
  let candidate_vsub_33_ptr : RegionName := "candidate_vsub_33"
  let candidate_vgt_16_ptr : RegionName := "candidate_vgt_16"
  let candidate_vmasked_fill_16_ptr : RegionName := "candidate_vmasked_fill_16"
  let candidate_vsub_34_ptr : RegionName := "candidate_vsub_34"
  let candidate_vexp_4_ptr : RegionName := "candidate_vexp_4"
  let candidate_vmul_56_ptr : RegionName := "candidate_vmul_56"
  let candidate_vsum_6_ptr : RegionName := "candidate_vsum_6"
  let candidate_vbmm_51_ptr : RegionName := "candidate_vbmm_51"
  let candidate_vsub_35_ptr : RegionName := "candidate_vsub_35"
  let candidate_vmul_57_ptr : RegionName := "candidate_vmul_57"
  let candidate_vbmm_52_ptr : RegionName := "candidate_vbmm_52"
  let candidate_vmul_58_ptr : RegionName := "candidate_vmul_58"
  let candidate_vto_copy_126_ptr : RegionName := "candidate_vto_copy_126"
  let candidate_vbmm_53_ptr : RegionName := "candidate_vbmm_53"
  let candidate_vmul_59_ptr : RegionName := "candidate_vmul_59"
  let candidate_vto_copy_127_ptr : RegionName := "candidate_vto_copy_127"
  let candidate_vbmm_54_ptr : RegionName := "candidate_vbmm_54"
  let candidate_vto_copy_128_ptr : RegionName := "candidate_vto_copy_128"
  let candidate_vcat_31_ptr : RegionName := "candidate_vcat_31"
  let candidate_vcat_32_ptr : RegionName := "candidate_vcat_32"
  let candidate_vto_copy_129_ptr : RegionName := "candidate_vto_copy_129"
  let candidate_vto_copy_130_ptr : RegionName := "candidate_vto_copy_130"
  let candidate_vto_copy_131_ptr : RegionName := "candidate_vto_copy_131"
  let candidate_vto_copy_132_ptr : RegionName := "candidate_vto_copy_132"
  let candidate_vto_copy_133_ptr : RegionName := "candidate_vto_copy_133"
  let candidate_vbmm_55_ptr : RegionName := "candidate_vbmm_55"
  let candidate_vmul_60_ptr : RegionName := "candidate_vmul_60"
  let candidate_vadd_38_ptr : RegionName := "candidate_vadd_38"
  let candidate_vsub_36_ptr : RegionName := "candidate_vsub_36"
  let candidate_vgt_17_ptr : RegionName := "candidate_vgt_17"
  let candidate_vmasked_fill_17_ptr : RegionName := "candidate_vmasked_fill_17"
  let candidate_vsub_37_ptr : RegionName := "candidate_vsub_37"
  let candidate_vexp_5_ptr : RegionName := "candidate_vexp_5"
  let candidate_vmul_61_ptr : RegionName := "candidate_vmul_61"
  let candidate_vsum_7_ptr : RegionName := "candidate_vsum_7"
  let candidate_vbmm_56_ptr : RegionName := "candidate_vbmm_56"
  let candidate_vsub_38_ptr : RegionName := "candidate_vsub_38"
  let candidate_vmul_62_ptr : RegionName := "candidate_vmul_62"
  let candidate_vbmm_57_ptr : RegionName := "candidate_vbmm_57"
  let candidate_vmul_63_ptr : RegionName := "candidate_vmul_63"
  let candidate_vto_copy_134_ptr : RegionName := "candidate_vto_copy_134"
  let candidate_vbmm_58_ptr : RegionName := "candidate_vbmm_58"
  let candidate_vmul_64_ptr : RegionName := "candidate_vmul_64"
  let candidate_vto_copy_135_ptr : RegionName := "candidate_vto_copy_135"
  let candidate_vbmm_59_ptr : RegionName := "candidate_vbmm_59"
  let candidate_vto_copy_136_ptr : RegionName := "candidate_vto_copy_136"
  let candidate_vcat_33_ptr : RegionName := "candidate_vcat_33"
  let candidate_vcat_34_ptr : RegionName := "candidate_vcat_34"
  let candidate_vcat_35_ptr : RegionName := "candidate_vcat_35"
  let candidate_vcat_36_ptr : RegionName := "candidate_vcat_36"
  let candidate_vcat_37_ptr : RegionName := "candidate_vcat_37"
  let candidate_vsum_8_ptr : RegionName := "candidate_vsum_8"
  let candidate_vbmm_60_ptr : RegionName := "candidate_vbmm_60"
  let candidate_vbmm_61_ptr : RegionName := "candidate_vbmm_61"
  let candidate_vadd_39_ptr : RegionName := "candidate_vadd_39"
  let candidate_vslice_backward_4_ptr : RegionName := "candidate_vslice_backward_4"
  let candidate_vadd_40_ptr : RegionName := "candidate_vadd_40"
  let candidate_vslice_backward_5_ptr : RegionName := "candidate_vslice_backward_5"
  let candidate_vadd_41_ptr : RegionName := "candidate_vadd_41"
  let candidate_vslice_backward_6_ptr : RegionName := "candidate_vslice_backward_6"
  let candidate_vadd_42_ptr : RegionName := "candidate_vadd_42"
  let candidate_vslice_backward_7_ptr : RegionName := "candidate_vslice_backward_7"
  let candidate_vadd_43_ptr : RegionName := "candidate_vadd_43"
  let candidate_vbmm_64_ptr : RegionName := "candidate_vbmm_64"
  let candidate_vbmm_65_ptr : RegionName := "candidate_vbmm_65"
  let candidate_vcat_39_ptr : RegionName := "candidate_vcat_39"
  let candidate_vcat_40_ptr : RegionName := "candidate_vcat_40"
  let candidate_vto_copy_137_ptr : RegionName := "candidate_vto_copy_137"
  let candidate_vto_copy_138_ptr : RegionName := "candidate_vto_copy_138"
  let candidate_vto_copy_139_ptr : RegionName := "candidate_vto_copy_139"
  let candidate_vto_copy_140_ptr : RegionName := "candidate_vto_copy_140"
  let candidate_vto_copy_141_ptr : RegionName := "candidate_vto_copy_141"
  let candidate_vbmm_66_ptr : RegionName := "candidate_vbmm_66"
  let candidate_vmul_65_ptr : RegionName := "candidate_vmul_65"
  let candidate_vadd_44_ptr : RegionName := "candidate_vadd_44"
  let candidate_vsub_39_ptr : RegionName := "candidate_vsub_39"
  let candidate_vgt_18_ptr : RegionName := "candidate_vgt_18"
  let candidate_vmasked_fill_18_ptr : RegionName := "candidate_vmasked_fill_18"
  let candidate_vsub_40_ptr : RegionName := "candidate_vsub_40"
  let candidate_vexp_6_ptr : RegionName := "candidate_vexp_6"
  let candidate_vmul_66_ptr : RegionName := "candidate_vmul_66"
  let candidate_vsum_9_ptr : RegionName := "candidate_vsum_9"
  let candidate_vbmm_67_ptr : RegionName := "candidate_vbmm_67"
  let candidate_vsub_41_ptr : RegionName := "candidate_vsub_41"
  let candidate_vmul_67_ptr : RegionName := "candidate_vmul_67"
  let candidate_vbmm_68_ptr : RegionName := "candidate_vbmm_68"
  let candidate_vmul_68_ptr : RegionName := "candidate_vmul_68"
  let candidate_vto_copy_142_ptr : RegionName := "candidate_vto_copy_142"
  let candidate_vbmm_69_ptr : RegionName := "candidate_vbmm_69"
  let candidate_vmul_69_ptr : RegionName := "candidate_vmul_69"
  let candidate_vto_copy_143_ptr : RegionName := "candidate_vto_copy_143"
  let candidate_vbmm_70_ptr : RegionName := "candidate_vbmm_70"
  let candidate_vto_copy_144_ptr : RegionName := "candidate_vto_copy_144"
  let candidate_vto_copy_145_ptr : RegionName := "candidate_vto_copy_145"
  let candidate_vto_copy_146_ptr : RegionName := "candidate_vto_copy_146"
  let candidate_vto_copy_147_ptr : RegionName := "candidate_vto_copy_147"
  let candidate_vto_copy_148_ptr : RegionName := "candidate_vto_copy_148"
  let candidate_vto_copy_149_ptr : RegionName := "candidate_vto_copy_149"
  let candidate_vbmm_71_ptr : RegionName := "candidate_vbmm_71"
  let candidate_vmul_70_ptr : RegionName := "candidate_vmul_70"
  let candidate_vadd_45_ptr : RegionName := "candidate_vadd_45"
  let candidate_vsub_42_ptr : RegionName := "candidate_vsub_42"
  let candidate_vgt_19_ptr : RegionName := "candidate_vgt_19"
  let candidate_vmasked_fill_19_ptr : RegionName := "candidate_vmasked_fill_19"
  let candidate_vsub_43_ptr : RegionName := "candidate_vsub_43"
  let candidate_vexp_7_ptr : RegionName := "candidate_vexp_7"
  let candidate_vmul_71_ptr : RegionName := "candidate_vmul_71"
  let candidate_vsum_10_ptr : RegionName := "candidate_vsum_10"
  let candidate_vbmm_72_ptr : RegionName := "candidate_vbmm_72"
  let candidate_vsub_44_ptr : RegionName := "candidate_vsub_44"
  let candidate_vmul_72_ptr : RegionName := "candidate_vmul_72"
  let candidate_vbmm_73_ptr : RegionName := "candidate_vbmm_73"
  let candidate_vmul_73_ptr : RegionName := "candidate_vmul_73"
  let candidate_vto_copy_150_ptr : RegionName := "candidate_vto_copy_150"
  let candidate_vbmm_74_ptr : RegionName := "candidate_vbmm_74"
  let candidate_vmul_74_ptr : RegionName := "candidate_vmul_74"
  let candidate_vto_copy_151_ptr : RegionName := "candidate_vto_copy_151"
  let candidate_vbmm_75_ptr : RegionName := "candidate_vbmm_75"
  let candidate_vto_copy_152_ptr : RegionName := "candidate_vto_copy_152"
  let candidate_vto_copy_153_ptr : RegionName := "candidate_vto_copy_153"
  let candidate_vto_copy_154_ptr : RegionName := "candidate_vto_copy_154"
  let candidate_vto_copy_155_ptr : RegionName := "candidate_vto_copy_155"
  let candidate_vto_copy_156_ptr : RegionName := "candidate_vto_copy_156"
  let candidate_vto_copy_157_ptr : RegionName := "candidate_vto_copy_157"
  let candidate_vbmm_76_ptr : RegionName := "candidate_vbmm_76"
  let candidate_vmul_75_ptr : RegionName := "candidate_vmul_75"
  let candidate_vadd_46_ptr : RegionName := "candidate_vadd_46"
  let candidate_vsub_45_ptr : RegionName := "candidate_vsub_45"
  let candidate_vgt_20_ptr : RegionName := "candidate_vgt_20"
  let candidate_vmasked_fill_20_ptr : RegionName := "candidate_vmasked_fill_20"
  let candidate_vsub_46_ptr : RegionName := "candidate_vsub_46"
  let candidate_vexp_8_ptr : RegionName := "candidate_vexp_8"
  let candidate_vmul_76_ptr : RegionName := "candidate_vmul_76"
  let candidate_vsum_11_ptr : RegionName := "candidate_vsum_11"
  let candidate_vbmm_77_ptr : RegionName := "candidate_vbmm_77"
  let candidate_vsub_47_ptr : RegionName := "candidate_vsub_47"
  let candidate_vmul_77_ptr : RegionName := "candidate_vmul_77"
  let candidate_vbmm_78_ptr : RegionName := "candidate_vbmm_78"
  let candidate_vmul_78_ptr : RegionName := "candidate_vmul_78"
  let candidate_vto_copy_158_ptr : RegionName := "candidate_vto_copy_158"
  let candidate_vbmm_79_ptr : RegionName := "candidate_vbmm_79"
  let candidate_vmul_79_ptr : RegionName := "candidate_vmul_79"
  let candidate_vto_copy_159_ptr : RegionName := "candidate_vto_copy_159"
  let candidate_vbmm_80_ptr : RegionName := "candidate_vbmm_80"
  let candidate_vto_copy_160_ptr : RegionName := "candidate_vto_copy_160"
  let candidate_vcat_41_ptr : RegionName := "candidate_vcat_41"
  let candidate_vcat_42_ptr : RegionName := "candidate_vcat_42"
  let candidate_vcat_43_ptr : RegionName := "candidate_vcat_43"
  let candidate_vsum_12_ptr : RegionName := "candidate_vsum_12"
  let candidate_vbmm_81_ptr : RegionName := "candidate_vbmm_81"
  let candidate_vbmm_82_ptr : RegionName := "candidate_vbmm_82"
  let candidate_vadd_47_ptr : RegionName := "candidate_vadd_47"
  let candidate_vslice_backward_8_ptr : RegionName := "candidate_vslice_backward_8"
  let candidate_vadd_48_ptr : RegionName := "candidate_vadd_48"
  let candidate_vslice_backward_9_ptr : RegionName := "candidate_vslice_backward_9"
  let candidate_vadd_49_ptr : RegionName := "candidate_vadd_49"
  let candidate_vslice_backward_10_ptr : RegionName := "candidate_vslice_backward_10"
  let candidate_vadd_50_ptr : RegionName := "candidate_vadd_50"
  let candidate_vslice_backward_11_ptr : RegionName := "candidate_vslice_backward_11"
  let candidate_vadd_51_ptr : RegionName := "candidate_vadd_51"
  let candidate_vbmm_85_ptr : RegionName := "candidate_vbmm_85"
  let candidate_vbmm_86_ptr : RegionName := "candidate_vbmm_86"
  let candidate_vcat_45_ptr : RegionName := "candidate_vcat_45"
  let candidate_vcat_46_ptr : RegionName := "candidate_vcat_46"
  let candidate_vto_copy_161_ptr : RegionName := "candidate_vto_copy_161"
  let candidate_vto_copy_162_ptr : RegionName := "candidate_vto_copy_162"
  let candidate_vto_copy_163_ptr : RegionName := "candidate_vto_copy_163"
  let candidate_vto_copy_164_ptr : RegionName := "candidate_vto_copy_164"
  let candidate_vto_copy_165_ptr : RegionName := "candidate_vto_copy_165"
  let candidate_vbmm_87_ptr : RegionName := "candidate_vbmm_87"
  let candidate_vmul_80_ptr : RegionName := "candidate_vmul_80"
  let candidate_vadd_52_ptr : RegionName := "candidate_vadd_52"
  let candidate_vsub_48_ptr : RegionName := "candidate_vsub_48"
  let candidate_vgt_21_ptr : RegionName := "candidate_vgt_21"
  let candidate_vmasked_fill_21_ptr : RegionName := "candidate_vmasked_fill_21"
  let candidate_vsub_49_ptr : RegionName := "candidate_vsub_49"
  let candidate_vexp_9_ptr : RegionName := "candidate_vexp_9"
  let candidate_vmul_81_ptr : RegionName := "candidate_vmul_81"
  let candidate_vsum_13_ptr : RegionName := "candidate_vsum_13"
  let candidate_vbmm_88_ptr : RegionName := "candidate_vbmm_88"
  let candidate_vsub_50_ptr : RegionName := "candidate_vsub_50"
  let candidate_vmul_82_ptr : RegionName := "candidate_vmul_82"
  let candidate_vbmm_89_ptr : RegionName := "candidate_vbmm_89"
  let candidate_vmul_83_ptr : RegionName := "candidate_vmul_83"
  let candidate_vto_copy_166_ptr : RegionName := "candidate_vto_copy_166"
  let candidate_vbmm_90_ptr : RegionName := "candidate_vbmm_90"
  let candidate_vmul_84_ptr : RegionName := "candidate_vmul_84"
  let candidate_vto_copy_167_ptr : RegionName := "candidate_vto_copy_167"
  let candidate_vbmm_91_ptr : RegionName := "candidate_vbmm_91"
  let candidate_vto_copy_168_ptr : RegionName := "candidate_vto_copy_168"
  let candidate_vcat_47_ptr : RegionName := "candidate_vcat_47"
  let candidate_vcat_48_ptr : RegionName := "candidate_vcat_48"
  let candidate_vto_copy_169_ptr : RegionName := "candidate_vto_copy_169"
  let candidate_vto_copy_170_ptr : RegionName := "candidate_vto_copy_170"
  let candidate_vto_copy_171_ptr : RegionName := "candidate_vto_copy_171"
  let candidate_vto_copy_172_ptr : RegionName := "candidate_vto_copy_172"
  let candidate_vto_copy_173_ptr : RegionName := "candidate_vto_copy_173"
  let candidate_vbmm_92_ptr : RegionName := "candidate_vbmm_92"
  let candidate_vmul_85_ptr : RegionName := "candidate_vmul_85"
  let candidate_vadd_53_ptr : RegionName := "candidate_vadd_53"
  let candidate_vsub_51_ptr : RegionName := "candidate_vsub_51"
  let candidate_vgt_22_ptr : RegionName := "candidate_vgt_22"
  let candidate_vmasked_fill_22_ptr : RegionName := "candidate_vmasked_fill_22"
  let candidate_vsub_52_ptr : RegionName := "candidate_vsub_52"
  let candidate_vexp_10_ptr : RegionName := "candidate_vexp_10"
  let candidate_vmul_86_ptr : RegionName := "candidate_vmul_86"
  let candidate_vsum_14_ptr : RegionName := "candidate_vsum_14"
  let candidate_vbmm_93_ptr : RegionName := "candidate_vbmm_93"
  let candidate_vsub_53_ptr : RegionName := "candidate_vsub_53"
  let candidate_vmul_87_ptr : RegionName := "candidate_vmul_87"
  let candidate_vbmm_94_ptr : RegionName := "candidate_vbmm_94"
  let candidate_vmul_88_ptr : RegionName := "candidate_vmul_88"
  let candidate_vto_copy_174_ptr : RegionName := "candidate_vto_copy_174"
  let candidate_vbmm_95_ptr : RegionName := "candidate_vbmm_95"
  let candidate_vmul_89_ptr : RegionName := "candidate_vmul_89"
  let candidate_vto_copy_175_ptr : RegionName := "candidate_vto_copy_175"
  let candidate_vbmm_96_ptr : RegionName := "candidate_vbmm_96"
  let candidate_vto_copy_176_ptr : RegionName := "candidate_vto_copy_176"
  let candidate_vcat_49_ptr : RegionName := "candidate_vcat_49"
  let candidate_vcat_50_ptr : RegionName := "candidate_vcat_50"
  let candidate_vto_copy_177_ptr : RegionName := "candidate_vto_copy_177"
  let candidate_vto_copy_178_ptr : RegionName := "candidate_vto_copy_178"
  let candidate_vto_copy_179_ptr : RegionName := "candidate_vto_copy_179"
  let candidate_vto_copy_180_ptr : RegionName := "candidate_vto_copy_180"
  let candidate_vto_copy_181_ptr : RegionName := "candidate_vto_copy_181"
  let candidate_vbmm_97_ptr : RegionName := "candidate_vbmm_97"
  let candidate_vmul_90_ptr : RegionName := "candidate_vmul_90"
  let candidate_vadd_54_ptr : RegionName := "candidate_vadd_54"
  let candidate_vsub_54_ptr : RegionName := "candidate_vsub_54"
  let candidate_vgt_23_ptr : RegionName := "candidate_vgt_23"
  let candidate_vmasked_fill_23_ptr : RegionName := "candidate_vmasked_fill_23"
  let candidate_vsub_55_ptr : RegionName := "candidate_vsub_55"
  let candidate_vexp_11_ptr : RegionName := "candidate_vexp_11"
  let candidate_vmul_91_ptr : RegionName := "candidate_vmul_91"
  let candidate_vsum_15_ptr : RegionName := "candidate_vsum_15"
  let candidate_vbmm_98_ptr : RegionName := "candidate_vbmm_98"
  let candidate_vsub_56_ptr : RegionName := "candidate_vsub_56"
  let candidate_vmul_92_ptr : RegionName := "candidate_vmul_92"
  let candidate_vbmm_99_ptr : RegionName := "candidate_vbmm_99"
  let candidate_vmul_93_ptr : RegionName := "candidate_vmul_93"
  let candidate_vto_copy_182_ptr : RegionName := "candidate_vto_copy_182"
  let candidate_vbmm_100_ptr : RegionName := "candidate_vbmm_100"
  let candidate_vmul_94_ptr : RegionName := "candidate_vmul_94"
  let candidate_vto_copy_183_ptr : RegionName := "candidate_vto_copy_183"
  let candidate_vbmm_101_ptr : RegionName := "candidate_vbmm_101"
  let candidate_vto_copy_184_ptr : RegionName := "candidate_vto_copy_184"
  let candidate_vcat_51_ptr : RegionName := "candidate_vcat_51"
  let candidate_vcat_52_ptr : RegionName := "candidate_vcat_52"
  let candidate_vcat_53_ptr : RegionName := "candidate_vcat_53"
  let candidate_vcat_54_ptr : RegionName := "candidate_vcat_54"
  let candidate_vcat_55_ptr : RegionName := "candidate_vcat_55"
  let candidate_vsum_16_ptr : RegionName := "candidate_vsum_16"
  let candidate_vbmm_102_ptr : RegionName := "candidate_vbmm_102"
  let candidate_vbmm_103_ptr : RegionName := "candidate_vbmm_103"
  let candidate_vadd_55_ptr : RegionName := "candidate_vadd_55"
  let candidate_vslice_backward_12_ptr : RegionName := "candidate_vslice_backward_12"
  let candidate_vadd_56_ptr : RegionName := "candidate_vadd_56"
  let candidate_vslice_backward_13_ptr : RegionName := "candidate_vslice_backward_13"
  let candidate_vadd_57_ptr : RegionName := "candidate_vadd_57"
  let candidate_vslice_backward_14_ptr : RegionName := "candidate_vslice_backward_14"
  let candidate_vadd_58_ptr : RegionName := "candidate_vadd_58"
  let candidate_vslice_backward_15_ptr : RegionName := "candidate_vslice_backward_15"
  let candidate_vadd_59_ptr : RegionName := "candidate_vadd_59"
  let candidate_vcat_57_ptr : RegionName := "candidate_vcat_57"
  let candidate_vindex_put_1_ptr : RegionName := "candidate_vindex_put_1"
  let candidate_vmul_95_ptr : RegionName := "candidate_vmul_95"
  let candidate_vmul_96_ptr : RegionName := "candidate_vmul_96"
  let candidate_vselect_backward_ptr : RegionName := "candidate_vselect_backward"
  let candidate_vneg_2_ptr : RegionName := "candidate_vneg_2"
  let candidate_vselect_backward_1_ptr : RegionName := "candidate_vselect_backward_1"
  let candidate_vadd_60_ptr : RegionName := "candidate_vadd_60"
  let candidate_vadd_61_ptr : RegionName := "candidate_vadd_61"
  let candidate_vmul_97_ptr : RegionName := "candidate_vmul_97"
  let candidate_vmul_98_ptr : RegionName := "candidate_vmul_98"
  let candidate_vselect_backward_2_ptr : RegionName := "candidate_vselect_backward_2"
  let candidate_vneg_3_ptr : RegionName := "candidate_vneg_3"
  let candidate_vselect_backward_3_ptr : RegionName := "candidate_vselect_backward_3"
  let candidate_vadd_62_ptr : RegionName := "candidate_vadd_62"
  let candidate_vadd_63_ptr : RegionName := "candidate_vadd_63"
  let candidate_vcat_58_ptr : RegionName := "candidate_vcat_58"
  let candidate_vto_copy_185_ptr : RegionName := "candidate_vto_copy_185"
  let candidate_vmul_99_ptr : RegionName := "candidate_vmul_99"
  let candidate_vmul_100_ptr : RegionName := "candidate_vmul_100"
  let candidate_vmul_101_ptr : RegionName := "candidate_vmul_101"
  let candidate_vsum_17_ptr : RegionName := "candidate_vsum_17"
  let candidate_vpow_6_ptr : RegionName := "candidate_vpow_6"
  let candidate_vmul_102_ptr : RegionName := "candidate_vmul_102"
  let candidate_vmul_103_ptr : RegionName := "candidate_vmul_103"
  let candidate_vdiv_3_ptr : RegionName := "candidate_vdiv_3"
  let candidate_vpow_7_ptr : RegionName := "candidate_vpow_7"
  let candidate_vmul_104_ptr : RegionName := "candidate_vmul_104"
  let candidate_vmul_105_ptr : RegionName := "candidate_vmul_105"
  let candidate_vadd_64_ptr : RegionName := "candidate_vadd_64"
  let candidate_vto_copy_186_ptr : RegionName := "candidate_vto_copy_186"
  let candidate_vcat_59_ptr : RegionName := "candidate_vcat_59"
  let candidate_vmm_29_ptr : RegionName := "candidate_vmm_29"
  let candidate_vmm_30_ptr : RegionName := "candidate_vmm_30"
  let candidate_vto_copy_187_ptr : RegionName := "candidate_vto_copy_187"
  let candidate_vmul_106_ptr : RegionName := "candidate_vmul_106"
  let candidate_vmul_107_ptr : RegionName := "candidate_vmul_107"
  let candidate_vmul_108_ptr : RegionName := "candidate_vmul_108"
  let candidate_vsum_18_ptr : RegionName := "candidate_vsum_18"
  let candidate_vpow_8_ptr : RegionName := "candidate_vpow_8"
  let candidate_vmul_109_ptr : RegionName := "candidate_vmul_109"
  let candidate_vmul_110_ptr : RegionName := "candidate_vmul_110"
  let candidate_vdiv_4_ptr : RegionName := "candidate_vdiv_4"
  let candidate_vpow_9_ptr : RegionName := "candidate_vpow_9"
  let candidate_vmul_111_ptr : RegionName := "candidate_vmul_111"
  let candidate_vmul_112_ptr : RegionName := "candidate_vmul_112"
  let candidate_vadd_65_ptr : RegionName := "candidate_vadd_65"
  let candidate_vto_copy_188_ptr : RegionName := "candidate_vto_copy_188"
  let candidate_vmm_31_ptr : RegionName := "candidate_vmm_31"
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
    r69_row = tl.arange($(32))
    r69_col = tl.arange($(512))
    r69_lane = tl.arange($(128))
    for r69_group in range($(8)) {
      r69_left = tl.load($(candidate_vmm_1_ptr) + (((((((r69_group % $(8)) * $(4096)) + (r69_row[:, None] * $(128)) + r69_lane[None, :]) / $(128)) % $(32)) * $(1536)) + ((((((r69_group % $(8)) * $(4096)) + (r69_row[:, None] * $(128)) + r69_lane[None, :]) / $(4096)) % $(8)) * $(192)) + ((((r69_group % $(8)) * $(4096)) + (r69_row[:, None] * $(128)) + r69_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r69_right = tl.load($(candidate_vto_copy_5_ptr) + (((((((r69_group % $(8)) * $(65536)) + (r69_lane[:, None] * $(512)) + r69_col[None, :]) / $(65536)) % $(8)) * $(131072)) + ((((((r69_group % $(8)) * $(65536)) + (r69_lane[:, None] * $(512)) + r69_col[None, :]) / $(512)) % $(128)) * $(512)) + ((((r69_group % $(8)) * $(65536)) + (r69_lane[:, None] * $(512)) + r69_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r69_value = tl.dot(r69_left, r69_right)
      r69_value = r69_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_2_ptr) + ((r69_group * $(16384)) + (r69_row[:, None] * $(512)) + r69_col[None, :]), r69_value)
    }
    r70_z = tl.arange($(6144))
    r70_value = tl.load($(candidate_vadd_6_ptr) + ((((r70_z / $(64)) % $(12)) * $(512)) + (((r70_z / $(768)) % $(8)) * $(64)) + (r70_z % $(64))), dtype=tl.bfloat16)
    r70_value = r70_value.to(tl.float32)
    r70_value = tl.broadcast_to(r70_value, [$(6144)])
    tl.store($(candidate_vto_copy_29_ptr) + r70_z, r70_value)
    r71_z = tl.arange($(768))
    r71_value = tl.load($(candidate_vindex_2_ptr) + ((((r71_z / $(64)) % $(12)) * $(576)) + (r71_z % $(64)) + $(512)), dtype=tl.bfloat16)
    r71_value = r71_value.to(tl.float32)
    r71_value = tl.broadcast_to(r71_value, [$(768)])
    tl.store($(candidate_vto_copy_30_ptr) + r71_z, r71_value)
    r72_row = tl.arange($(12))
    r72_col = tl.arange($(12))
    r72_lane = tl.arange($(64))
    for r72_group in range($(8)) {
      r72_left = tl.load($(candidate_vto_copy_29_ptr) + (((((((r72_group % $(8)) * $(768)) + (r72_row[:, None] * $(64)) + r72_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r72_group % $(8)) * $(768)) + (r72_row[:, None] * $(64)) + r72_lane[None, :]) / $(64)) % $(12)) * $(64)) + ((((r72_group % $(8)) * $(768)) + (r72_row[:, None] * $(64)) + r72_lane[None, :]) % $(64))), dtype=tl.float32)
      r72_right = tl.load($(candidate_vto_copy_30_ptr) + ((((((r72_group % $(8)) * $(768)) + (r72_lane[:, None] * $(12)) + r72_col[None, :]) % $(12)) * $(64)) + (((((r72_group % $(8)) * $(768)) + (r72_lane[:, None] * $(12)) + r72_col[None, :]) / $(12)) % $(64))), dtype=tl.float32)
      r72_value = tl.dot(r72_left, r72_right)
      r72_value = r72_value.to(tl.float32)
      tl.store($(candidate_vbmm_3_ptr) + ((r72_group * $(144)) + (r72_row[:, None] * $(12)) + r72_col[None, :]), r72_value)
    }
    r73_z = tl.arange($(49152))
    r73_value = tl.load($(candidate_vbmm_2_ptr) + ((((r73_z / $(6144)) % $(8)) * $(16384)) + (((r73_z / $(512)) % $(12)) * $(512)) + (r73_z % $(512))), dtype=tl.bfloat16)
    r73_value = r73_value.to(tl.float32)
    r73_value = tl.broadcast_to(r73_value, [$(49152)])
    tl.store($(candidate_vto_copy_31_ptr) + r73_z, r73_value)
    r74_z = tl.arange($(6144))
    r74_value = tl.load($(candidate_vindex_2_ptr) + ((((r74_z / $(512)) % $(12)) * $(576)) + (r74_z % $(512))), dtype=tl.bfloat16)
    r74_value = r74_value.to(tl.float32)
    r74_value = tl.broadcast_to(r74_value, [$(6144)])
    tl.store($(candidate_vto_copy_32_ptr) + r74_z, r74_value)
    r75_row = tl.arange($(96))
    r75_col = tl.arange($(12))
    r75_lane = tl.arange($(512))
    r75_left = tl.load($(candidate_vto_copy_31_ptr) + ((((((r75_row[:, None] * $(512)) + r75_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r75_row[:, None] * $(512)) + r75_lane[None, :]) / $(512)) % $(12)) * $(512)) + (((r75_row[:, None] * $(512)) + r75_lane[None, :]) % $(512))), dtype=tl.float32)
    r75_right = tl.load($(candidate_vto_copy_32_ptr) + (((((r75_lane[:, None] * $(12)) + r75_col[None, :]) % $(12)) * $(512)) + ((((r75_lane[:, None] * $(12)) + r75_col[None, :]) / $(12)) % $(512))), dtype=tl.float32)
    r75_value = tl.dot(r75_left, r75_right)
    r75_value = r75_value.to(tl.float32)
    tl.store($(candidate_vmm_3_ptr) + ((r75_row[:, None] * $(12)) + r75_col[None, :]), r75_value)
    r76_z = tl.arange($(1152))
    r76_value = (tl.load($(candidate_vbmm_3_ptr) + ((((r76_z / $(144)) % $(8)) * $(144)) + (((r76_z / $(12)) % $(12)) * $(12)) + (r76_z % $(12))), dtype=tl.float32) + tl.load($(candidate_vmm_3_ptr) + ((((r76_z / $(144)) % $(8)) * $(144)) + (((r76_z / $(12)) % $(12)) * $(12)) + (r76_z % $(12))), dtype=tl.float32))
    r76_value = r76_value.to(tl.float32)
    r76_value = tl.broadcast_to(r76_value, [$(1152)])
    tl.store($(candidate_vadd_8_ptr) + r76_z, r76_value)
    r77_z = tl.arange($(1152))
    r77_value = (tl.load($(candidate_vadd_8_ptr) + ((((r77_z / $(144)) % $(8)) * $(144)) + (((r77_z / $(12)) % $(12)) * $(12)) + (r77_z % $(12))), dtype=tl.float32) * 0.07216878364870322)
    r77_value = r77_value.to(tl.float32)
    r77_value = tl.broadcast_to(r77_value, [$(1152)])
    tl.store($(candidate_vmul_23_ptr) + r77_z, r77_value)
    r78_z = tl.arange($(12))
    r78_value = (tl.cast(r78_z, tl.int64) + tl.full([], $(12), dtype=tl.int64))
    r78_value = tl.broadcast_to(r78_value, [$(12)])
    tl.store($(candidate_vadd_9_ptr) + r78_z, r78_value)
    r79_z = tl.arange($(12))
    r79_value = (tl.load($(candidate_vadd_9_ptr) + r79_z, dtype=tl.int64) - tl.full([], $(12), dtype=tl.int64))
    r79_value = tl.broadcast_to(r79_value, [$(12)])
    tl.store($(candidate_vsub_9_ptr) + r79_z, r79_value)
    r80_z = tl.arange($(144))
    r80_value = (tl.cast((r80_z % $(12)), tl.int64) > tl.load($(candidate_vsub_9_ptr) + ((r80_z / $(12)) % $(12)), dtype=tl.int64))
    r80_value = tl.broadcast_to(r80_value, [$(144)])
    tl.store($(candidate_vgt_ptr) + r80_z, r80_value)
    r81_z = tl.arange($(1152))
    r81_value = tl.where(tl.load($(candidate_vgt_ptr) + ((((r81_z / $(12)) % $(12)) * $(12)) + (r81_z % $(12))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_23_ptr) + ((((r81_z / $(144)) % $(8)) * $(144)) + (((r81_z / $(12)) % $(12)) * $(12)) + (r81_z % $(12))), dtype=tl.float32))
    r81_value = r81_value.to(tl.float32)
    r81_value = tl.broadcast_to(r81_value, [$(1152)])
    tl.store($(candidate_vmasked_fill_ptr) + r81_z, r81_value)
    r82_z = tl.arange($(1152))
    r82_lane = tl.arange($(12))
    r82_row_values = tl.load($(candidate_vmasked_fill_ptr) + ((((r82_z[:, None] / $(144)) % $(8)) * $(144)) + (((r82_z[:, None] / $(12)) % $(12)) * $(12)) + (r82_lane[None, :] % $(12))), dtype=tl.float32)
    r82_maximum = tl.max(r82_row_values, axis=1, keep_dims=true)
    r82_weights = tl.exp(r82_row_values - r82_maximum)
    r82_total = tl.sum(r82_weights, axis=1)
    r82_maximum = tl.max(r82_row_values, axis=1)
    r82_value = tl.exp(tl.load($(candidate_vmasked_fill_ptr) + ((((r82_z / $(144)) % $(8)) * $(144)) + (((r82_z / $(12)) % $(12)) * $(12)) + (r82_z % $(12))), dtype=tl.float32) - r82_maximum) / r82_total
    r82_value = r82_value.to(tl.float32)
    r82_value = tl.broadcast_to(r82_value, [$(1152)])
    tl.store($(candidate_vsoftmax_ptr) + r82_z, r82_value)
    r83_z = tl.arange($(1152))
    r83_value = tl.load($(candidate_vsoftmax_ptr) + ((((r83_z / $(144)) % $(8)) * $(144)) + (((r83_z / $(12)) % $(12)) * $(12)) + (r83_z % $(12))), dtype=tl.float32)
    r83_value = r83_value.to(tl.bfloat16)
    r83_value = tl.broadcast_to(r83_value, [$(1152)])
    tl.store($(candidate_vto_copy_33_ptr) + r83_z, r83_value)
    r84_row = tl.arange($(96))
    r84_col = tl.arange($(512))
    r84_lane = tl.arange($(12))
    r84_left = tl.load($(candidate_vto_copy_33_ptr) + ((((((r84_row[:, None] * $(12)) + r84_lane[None, :]) / $(144)) % $(8)) * $(144)) + (((((r84_row[:, None] * $(12)) + r84_lane[None, :]) / $(12)) % $(12)) * $(12)) + (((r84_row[:, None] * $(12)) + r84_lane[None, :]) % $(12))), dtype=tl.bfloat16)
    r84_right = tl.load($(candidate_vindex_2_ptr) + ((((((r84_lane[:, None] * $(512)) + r84_col[None, :]) / $(512)) % $(12)) * $(576)) + (((r84_lane[:, None] * $(512)) + r84_col[None, :]) % $(512))), dtype=tl.bfloat16)
    r84_value = tl.dot(r84_left, r84_right)
    r84_value = r84_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_4_ptr) + ((r84_row[:, None] * $(512)) + r84_col[None, :]), r84_value)
    r85_z = tl.arange($(96))
    r85_lane = tl.arange($(12))
    r85_row_values = tl.load($(candidate_vmasked_fill_ptr) + ((((r85_z[:, None] / $(12)) % $(8)) * $(144)) + ((r85_z[:, None] % $(12)) * $(12)) + (r85_lane[None, :] % $(12))), dtype=tl.float32)
    r85_maximum = tl.max(r85_row_values, axis=1, keep_dims=true)
    r85_weights = tl.exp(r85_row_values - r85_maximum)
    r85_total = tl.sum(r85_weights, axis=1)
    r85_maximum = tl.max(r85_row_values, axis=1)
    r85_value = r85_maximum + tl.log(r85_total)
    r85_value = r85_value.to(tl.float32)
    r85_value = tl.broadcast_to(r85_value, [$(96)])
    tl.store($(candidate_vlogsumexp_ptr) + r85_z, r85_value)
    r86_z = tl.arange($(2048))
    r86_value = tl.load($(candidate_vadd_6_ptr) + ((((r86_z / $(64)) % $(4)) * $(512)) + (((r86_z / $(256)) % $(8)) * $(64)) + (r86_z % $(64)) + $(6144)), dtype=tl.bfloat16)
    r86_value = r86_value.to(tl.float32)
    r86_value = tl.broadcast_to(r86_value, [$(2048)])
    tl.store($(candidate_vto_copy_34_ptr) + r86_z, r86_value)
    r87_z = tl.arange($(256))
    r87_value = tl.load($(candidate_vindex_2_ptr) + ((((r87_z / $(64)) % $(4)) * $(576)) + (r87_z % $(64)) + $(28160)), dtype=tl.bfloat16)
    r87_value = r87_value.to(tl.float32)
    r87_value = tl.broadcast_to(r87_value, [$(256)])
    tl.store($(candidate_vto_copy_35_ptr) + r87_z, r87_value)
    r88_row = tl.arange($(4))
    r88_col = tl.arange($(4))
    r88_lane = tl.arange($(64))
    for r88_group in range($(8)) {
      r88_left = tl.load($(candidate_vto_copy_34_ptr) + (((((((r88_group % $(8)) * $(256)) + (r88_row[:, None] * $(64)) + r88_lane[None, :]) / $(256)) % $(8)) * $(256)) + ((((((r88_group % $(8)) * $(256)) + (r88_row[:, None] * $(64)) + r88_lane[None, :]) / $(64)) % $(4)) * $(64)) + ((((r88_group % $(8)) * $(256)) + (r88_row[:, None] * $(64)) + r88_lane[None, :]) % $(64))), dtype=tl.float32)
      r88_right = tl.load($(candidate_vto_copy_35_ptr) + ((((((r88_group % $(8)) * $(256)) + (r88_lane[:, None] * $(4)) + r88_col[None, :]) % $(4)) * $(64)) + (((((r88_group % $(8)) * $(256)) + (r88_lane[:, None] * $(4)) + r88_col[None, :]) / $(4)) % $(64))), dtype=tl.float32)
      r88_value = tl.dot(r88_left, r88_right)
      r88_value = r88_value.to(tl.float32)
      tl.store($(candidate_vbmm_4_ptr) + ((r88_group * $(16)) + (r88_row[:, None] * $(4)) + r88_col[None, :]), r88_value)
    }
    r89_z = tl.arange($(16384))
    r89_value = tl.load($(candidate_vbmm_2_ptr) + ((((r89_z / $(2048)) % $(8)) * $(16384)) + (((r89_z / $(512)) % $(4)) * $(512)) + (r89_z % $(512)) + $(6144)), dtype=tl.bfloat16)
    r89_value = r89_value.to(tl.float32)
    r89_value = tl.broadcast_to(r89_value, [$(16384)])
    tl.store($(candidate_vto_copy_36_ptr) + r89_z, r89_value)
    r90_z = tl.arange($(2048))
    r90_value = tl.load($(candidate_vindex_2_ptr) + ((((r90_z / $(512)) % $(4)) * $(576)) + (r90_z % $(512)) + $(27648)), dtype=tl.bfloat16)
    r90_value = r90_value.to(tl.float32)
    r90_value = tl.broadcast_to(r90_value, [$(2048)])
    tl.store($(candidate_vto_copy_37_ptr) + r90_z, r90_value)
    r91_row = tl.arange($(32))
    r91_col = tl.arange($(4))
    r91_lane = tl.arange($(512))
    r91_left = tl.load($(candidate_vto_copy_36_ptr) + ((((((r91_row[:, None] * $(512)) + r91_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + (((((r91_row[:, None] * $(512)) + r91_lane[None, :]) / $(512)) % $(4)) * $(512)) + (((r91_row[:, None] * $(512)) + r91_lane[None, :]) % $(512))), dtype=tl.float32)
    r91_right = tl.load($(candidate_vto_copy_37_ptr) + (((((r91_lane[:, None] * $(4)) + r91_col[None, :]) % $(4)) * $(512)) + ((((r91_lane[:, None] * $(4)) + r91_col[None, :]) / $(4)) % $(512))), dtype=tl.float32)
    r91_value = tl.dot(r91_left, r91_right)
    r91_value = r91_value.to(tl.float32)
    tl.store($(candidate_vmm_5_ptr) + ((r91_row[:, None] * $(4)) + r91_col[None, :]), r91_value)
    r92_z = tl.arange($(128))
    r92_value = (tl.load($(candidate_vbmm_4_ptr) + ((((r92_z / $(16)) % $(8)) * $(16)) + (((r92_z / $(4)) % $(4)) * $(4)) + (r92_z % $(4))), dtype=tl.float32) + tl.load($(candidate_vmm_5_ptr) + ((((r92_z / $(16)) % $(8)) * $(16)) + (((r92_z / $(4)) % $(4)) * $(4)) + (r92_z % $(4))), dtype=tl.float32))
    r92_value = r92_value.to(tl.float32)
    r92_value = tl.broadcast_to(r92_value, [$(128)])
    tl.store($(candidate_vadd_10_ptr) + r92_z, r92_value)
    r93_z = tl.arange($(128))
    r93_value = (tl.load($(candidate_vadd_10_ptr) + ((((r93_z / $(16)) % $(8)) * $(16)) + (((r93_z / $(4)) % $(4)) * $(4)) + (r93_z % $(4))), dtype=tl.float32) * 0.07216878364870322)
    r93_value = r93_value.to(tl.float32)
    r93_value = tl.broadcast_to(r93_value, [$(128)])
    tl.store($(candidate_vmul_24_ptr) + r93_z, r93_value)
    r94_z = tl.arange($(4))
    r94_value = (tl.cast(r94_z, tl.int64) + tl.full([], $(4), dtype=tl.int64))
    r94_value = tl.broadcast_to(r94_value, [$(4)])
    tl.store($(candidate_vadd_11_ptr) + r94_z, r94_value)
    r95_z = tl.arange($(4))
    r95_value = (tl.load($(candidate_vadd_11_ptr) + r95_z, dtype=tl.int64) - tl.full([], $(4), dtype=tl.int64))
    r95_value = tl.broadcast_to(r95_value, [$(4)])
    tl.store($(candidate_vsub_10_ptr) + r95_z, r95_value)
    r96_z = tl.arange($(16))
    r96_value = (tl.cast((r96_z % $(4)), tl.int64) > tl.load($(candidate_vsub_10_ptr) + ((r96_z / $(4)) % $(4)), dtype=tl.int64))
    r96_value = tl.broadcast_to(r96_value, [$(16)])
    tl.store($(candidate_vgt_1_ptr) + r96_z, r96_value)
    r97_z = tl.arange($(128))
    r97_value = tl.where(tl.load($(candidate_vgt_1_ptr) + ((((r97_z / $(4)) % $(4)) * $(4)) + (r97_z % $(4))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_24_ptr) + ((((r97_z / $(16)) % $(8)) * $(16)) + (((r97_z / $(4)) % $(4)) * $(4)) + (r97_z % $(4))), dtype=tl.float32))
    r97_value = r97_value.to(tl.float32)
    r97_value = tl.broadcast_to(r97_value, [$(128)])
    tl.store($(candidate_vmasked_fill_1_ptr) + r97_z, r97_value)
    r98_z = tl.arange($(128))
    r98_lane = tl.arange($(4))
    r98_row_values = tl.load($(candidate_vmasked_fill_1_ptr) + ((((r98_z[:, None] / $(16)) % $(8)) * $(16)) + (((r98_z[:, None] / $(4)) % $(4)) * $(4)) + (r98_lane[None, :] % $(4))), dtype=tl.float32)
    r98_maximum = tl.max(r98_row_values, axis=1, keep_dims=true)
    r98_weights = tl.exp(r98_row_values - r98_maximum)
    r98_total = tl.sum(r98_weights, axis=1)
    r98_maximum = tl.max(r98_row_values, axis=1)
    r98_value = tl.exp(tl.load($(candidate_vmasked_fill_1_ptr) + ((((r98_z / $(16)) % $(8)) * $(16)) + (((r98_z / $(4)) % $(4)) * $(4)) + (r98_z % $(4))), dtype=tl.float32) - r98_maximum) / r98_total
    r98_value = r98_value.to(tl.float32)
    r98_value = tl.broadcast_to(r98_value, [$(128)])
    tl.store($(candidate_vsoftmax_1_ptr) + r98_z, r98_value)
    r99_z = tl.arange($(128))
    r99_value = tl.load($(candidate_vsoftmax_1_ptr) + ((((r99_z / $(16)) % $(8)) * $(16)) + (((r99_z / $(4)) % $(4)) * $(4)) + (r99_z % $(4))), dtype=tl.float32)
    r99_value = r99_value.to(tl.bfloat16)
    r99_value = tl.broadcast_to(r99_value, [$(128)])
    tl.store($(candidate_vto_copy_38_ptr) + r99_z, r99_value)
    r100_row = tl.arange($(32))
    r100_col = tl.arange($(512))
    r100_lane = tl.arange($(4))
    r100_left = tl.load($(candidate_vto_copy_38_ptr) + ((((((r100_row[:, None] * $(4)) + r100_lane[None, :]) / $(16)) % $(8)) * $(16)) + (((((r100_row[:, None] * $(4)) + r100_lane[None, :]) / $(4)) % $(4)) * $(4)) + (((r100_row[:, None] * $(4)) + r100_lane[None, :]) % $(4))), dtype=tl.bfloat16)
    r100_right = tl.load($(candidate_vindex_2_ptr) + ((((((r100_lane[:, None] * $(512)) + r100_col[None, :]) / $(512)) % $(4)) * $(576)) + (((r100_lane[:, None] * $(512)) + r100_col[None, :]) % $(512)) + $(27648)), dtype=tl.bfloat16)
    r100_value = tl.dot(r100_left, r100_right)
    r100_value = r100_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_6_ptr) + ((r100_row[:, None] * $(512)) + r100_col[None, :]), r100_value)
    r101_z = tl.arange($(32))
    r101_lane = tl.arange($(4))
    r101_row_values = tl.load($(candidate_vmasked_fill_1_ptr) + ((((r101_z[:, None] / $(4)) % $(8)) * $(16)) + ((r101_z[:, None] % $(4)) * $(4)) + (r101_lane[None, :] % $(4))), dtype=tl.float32)
    r101_maximum = tl.max(r101_row_values, axis=1, keep_dims=true)
    r101_weights = tl.exp(r101_row_values - r101_maximum)
    r101_total = tl.sum(r101_weights, axis=1)
    r101_maximum = tl.max(r101_row_values, axis=1)
    r101_value = r101_maximum + tl.log(r101_total)
    r101_value = r101_value.to(tl.float32)
    r101_value = tl.broadcast_to(r101_value, [$(32)])
    tl.store($(candidate_vlogsumexp_1_ptr) + r101_z, r101_value)
    r102_z = tl.arange($(8192))
    r102_value = tl.load($(candidate_vadd_6_ptr) + ((((r102_z / $(64)) % $(16)) * $(512)) + (((r102_z / $(1024)) % $(8)) * $(64)) + (r102_z % $(64)) + $(8192)), dtype=tl.bfloat16)
    r102_value = r102_value.to(tl.float32)
    r102_value = tl.broadcast_to(r102_value, [$(8192)])
    tl.store($(candidate_vto_copy_39_ptr) + r102_z, r102_value)
    r103_z = tl.arange($(1024))
    r103_value = tl.load($(candidate_vindex_2_ptr) + ((((r103_z / $(64)) % $(16)) * $(576)) + (r103_z % $(64)) + $(37376)), dtype=tl.bfloat16)
    r103_value = r103_value.to(tl.float32)
    r103_value = tl.broadcast_to(r103_value, [$(1024)])
    tl.store($(candidate_vto_copy_40_ptr) + r103_z, r103_value)
    r104_row = tl.arange($(16))
    r104_col = tl.arange($(16))
    r104_lane = tl.arange($(64))
    for r104_group in range($(8)) {
      r104_left = tl.load($(candidate_vto_copy_39_ptr) + (((((((r104_group % $(8)) * $(1024)) + (r104_row[:, None] * $(64)) + r104_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + ((((((r104_group % $(8)) * $(1024)) + (r104_row[:, None] * $(64)) + r104_lane[None, :]) / $(64)) % $(16)) * $(64)) + ((((r104_group % $(8)) * $(1024)) + (r104_row[:, None] * $(64)) + r104_lane[None, :]) % $(64))), dtype=tl.float32)
      r104_right = tl.load($(candidate_vto_copy_40_ptr) + ((((((r104_group % $(8)) * $(1024)) + (r104_lane[:, None] * $(16)) + r104_col[None, :]) % $(16)) * $(64)) + (((((r104_group % $(8)) * $(1024)) + (r104_lane[:, None] * $(16)) + r104_col[None, :]) / $(16)) % $(64))), dtype=tl.float32)
      r104_value = tl.dot(r104_left, r104_right)
      r104_value = r104_value.to(tl.float32)
      tl.store($(candidate_vbmm_5_ptr) + ((r104_group * $(256)) + (r104_row[:, None] * $(16)) + r104_col[None, :]), r104_value)
    }
    r105_z = tl.arange($(65536))
    r105_value = tl.load($(candidate_vbmm_2_ptr) + ((((r105_z / $(8192)) % $(8)) * $(16384)) + (((r105_z / $(512)) % $(16)) * $(512)) + (r105_z % $(512)) + $(8192)), dtype=tl.bfloat16)
    r105_value = r105_value.to(tl.float32)
    r105_value = tl.broadcast_to(r105_value, [$(65536)])
    tl.store($(candidate_vto_copy_41_ptr) + r105_z, r105_value)
    r106_z = tl.arange($(8192))
    r106_value = tl.load($(candidate_vindex_2_ptr) + ((((r106_z / $(512)) % $(16)) * $(576)) + (r106_z % $(512)) + $(36864)), dtype=tl.bfloat16)
    r106_value = r106_value.to(tl.float32)
    r106_value = tl.broadcast_to(r106_value, [$(8192)])
    tl.store($(candidate_vto_copy_42_ptr) + r106_z, r106_value)
    r107_row = tl.arange($(128))
    r107_col = tl.arange($(16))
    r107_lane = tl.arange($(512))
    r107_left = tl.load($(candidate_vto_copy_41_ptr) + ((((((r107_row[:, None] * $(512)) + r107_lane[None, :]) / $(8192)) % $(8)) * $(8192)) + (((((r107_row[:, None] * $(512)) + r107_lane[None, :]) / $(512)) % $(16)) * $(512)) + (((r107_row[:, None] * $(512)) + r107_lane[None, :]) % $(512))), dtype=tl.float32)
    r107_right = tl.load($(candidate_vto_copy_42_ptr) + (((((r107_lane[:, None] * $(16)) + r107_col[None, :]) % $(16)) * $(512)) + ((((r107_lane[:, None] * $(16)) + r107_col[None, :]) / $(16)) % $(512))), dtype=tl.float32)
    r107_value = tl.dot(r107_left, r107_right)
    r107_value = r107_value.to(tl.float32)
    tl.store($(candidate_vmm_7_ptr) + ((r107_row[:, None] * $(16)) + r107_col[None, :]), r107_value)
    r108_z = tl.arange($(2048))
    r108_value = (tl.load($(candidate_vbmm_5_ptr) + ((((r108_z / $(256)) % $(8)) * $(256)) + (((r108_z / $(16)) % $(16)) * $(16)) + (r108_z % $(16))), dtype=tl.float32) + tl.load($(candidate_vmm_7_ptr) + ((((r108_z / $(256)) % $(8)) * $(256)) + (((r108_z / $(16)) % $(16)) * $(16)) + (r108_z % $(16))), dtype=tl.float32))
    r108_value = r108_value.to(tl.float32)
    r108_value = tl.broadcast_to(r108_value, [$(2048)])
    tl.store($(candidate_vadd_12_ptr) + r108_z, r108_value)
    r109_z = tl.arange($(2048))
    r109_value = (tl.load($(candidate_vadd_12_ptr) + ((((r109_z / $(256)) % $(8)) * $(256)) + (((r109_z / $(16)) % $(16)) * $(16)) + (r109_z % $(16))), dtype=tl.float32) * 0.07216878364870322)
    r109_value = r109_value.to(tl.float32)
    r109_value = tl.broadcast_to(r109_value, [$(2048)])
    tl.store($(candidate_vmul_25_ptr) + r109_z, r109_value)
    r110_z = tl.arange($(16))
    r110_value = (tl.cast(r110_z, tl.int64) + tl.full([], $(16), dtype=tl.int64))
    r110_value = tl.broadcast_to(r110_value, [$(16)])
    tl.store($(candidate_vadd_13_ptr) + r110_z, r110_value)
    r111_z = tl.arange($(16))
    r111_value = (tl.load($(candidate_vadd_13_ptr) + r111_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r111_value = tl.broadcast_to(r111_value, [$(16)])
    tl.store($(candidate_vsub_11_ptr) + r111_z, r111_value)
    r112_z = tl.arange($(256))
    r112_value = (tl.cast((r112_z % $(16)), tl.int64) > tl.load($(candidate_vsub_11_ptr) + ((r112_z / $(16)) % $(16)), dtype=tl.int64))
    r112_value = tl.broadcast_to(r112_value, [$(256)])
    tl.store($(candidate_vgt_2_ptr) + r112_z, r112_value)
    r113_z = tl.arange($(2048))
    r113_value = tl.where(tl.load($(candidate_vgt_2_ptr) + ((((r113_z / $(16)) % $(16)) * $(16)) + (r113_z % $(16))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_25_ptr) + ((((r113_z / $(256)) % $(8)) * $(256)) + (((r113_z / $(16)) % $(16)) * $(16)) + (r113_z % $(16))), dtype=tl.float32))
    r113_value = r113_value.to(tl.float32)
    r113_value = tl.broadcast_to(r113_value, [$(2048)])
    tl.store($(candidate_vmasked_fill_2_ptr) + r113_z, r113_value)
    r114_z = tl.arange($(2048))
    r114_lane = tl.arange($(16))
    r114_row_values = tl.load($(candidate_vmasked_fill_2_ptr) + ((((r114_z[:, None] / $(256)) % $(8)) * $(256)) + (((r114_z[:, None] / $(16)) % $(16)) * $(16)) + (r114_lane[None, :] % $(16))), dtype=tl.float32)
    r114_maximum = tl.max(r114_row_values, axis=1, keep_dims=true)
    r114_weights = tl.exp(r114_row_values - r114_maximum)
    r114_total = tl.sum(r114_weights, axis=1)
    r114_maximum = tl.max(r114_row_values, axis=1)
    r114_value = tl.exp(tl.load($(candidate_vmasked_fill_2_ptr) + ((((r114_z / $(256)) % $(8)) * $(256)) + (((r114_z / $(16)) % $(16)) * $(16)) + (r114_z % $(16))), dtype=tl.float32) - r114_maximum) / r114_total
    r114_value = r114_value.to(tl.float32)
    r114_value = tl.broadcast_to(r114_value, [$(2048)])
    tl.store($(candidate_vsoftmax_2_ptr) + r114_z, r114_value)
    r115_z = tl.arange($(2048))
    r115_value = tl.load($(candidate_vsoftmax_2_ptr) + ((((r115_z / $(256)) % $(8)) * $(256)) + (((r115_z / $(16)) % $(16)) * $(16)) + (r115_z % $(16))), dtype=tl.float32)
    r115_value = r115_value.to(tl.bfloat16)
    r115_value = tl.broadcast_to(r115_value, [$(2048)])
    tl.store($(candidate_vto_copy_43_ptr) + r115_z, r115_value)
    r116_row = tl.arange($(128))
    r116_col = tl.arange($(512))
    r116_lane = tl.arange($(16))
    r116_left = tl.load($(candidate_vto_copy_43_ptr) + ((((((r116_row[:, None] * $(16)) + r116_lane[None, :]) / $(256)) % $(8)) * $(256)) + (((((r116_row[:, None] * $(16)) + r116_lane[None, :]) / $(16)) % $(16)) * $(16)) + (((r116_row[:, None] * $(16)) + r116_lane[None, :]) % $(16))), dtype=tl.bfloat16)
    r116_right = tl.load($(candidate_vindex_2_ptr) + ((((((r116_lane[:, None] * $(512)) + r116_col[None, :]) / $(512)) % $(16)) * $(576)) + (((r116_lane[:, None] * $(512)) + r116_col[None, :]) % $(512)) + $(36864)), dtype=tl.bfloat16)
    r116_value = tl.dot(r116_left, r116_right)
    r116_value = r116_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_8_ptr) + ((r116_row[:, None] * $(512)) + r116_col[None, :]), r116_value)
    r117_z = tl.arange($(128))
    r117_lane = tl.arange($(16))
    r117_row_values = tl.load($(candidate_vmasked_fill_2_ptr) + ((((r117_z[:, None] / $(16)) % $(8)) * $(256)) + ((r117_z[:, None] % $(16)) * $(16)) + (r117_lane[None, :] % $(16))), dtype=tl.float32)
    r117_maximum = tl.max(r117_row_values, axis=1, keep_dims=true)
    r117_weights = tl.exp(r117_row_values - r117_maximum)
    r117_total = tl.sum(r117_weights, axis=1)
    r117_maximum = tl.max(r117_row_values, axis=1)
    r117_value = r117_maximum + tl.log(r117_total)
    r117_value = r117_value.to(tl.float32)
    r117_value = tl.broadcast_to(r117_value, [$(128)])
    tl.store($(candidate_vlogsumexp_2_ptr) + r117_z, r117_value)
    r118_z = tl.arange($(131072))
    r118_value = tl.where(((r118_z / $(4096)) % $(32)) < $(12), tl.load($(candidate_vmm_4_ptr) + (((((((r118_z / $(512)) % $(8)) * $(12)) + (((r118_z / $(4096)) % $(32)) - $(0))) % $(96)) * $(512)) + (r118_z % $(512))), dtype=tl.bfloat16), tl.where(((r118_z / $(4096)) % $(32)) < $(16), tl.load($(candidate_vmm_6_ptr) + (((((((r118_z / $(512)) % $(8)) * $(4)) + (((r118_z / $(4096)) % $(32)) - $(12))) % $(32)) * $(512)) + (r118_z % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vmm_8_ptr) + (((((((r118_z / $(512)) % $(8)) * $(16)) + (((r118_z / $(4096)) % $(32)) - $(16))) % $(128)) * $(512)) + (r118_z % $(512))), dtype=tl.bfloat16)))
    r118_value = r118_value.to(tl.bfloat16)
    r118_value = tl.broadcast_to(r118_value, [$(131072)])
    tl.store($(candidate_vcat_6_ptr) + r118_z, r118_value)
    r119_z = tl.arange($(256))
    r119_value = tl.where(((r119_z / $(8)) % $(32)) < $(12), tl.load($(candidate_vlogsumexp_ptr) + (((r119_z % $(8)) * $(12)) + ((((r119_z / $(8)) % $(32)) - $(0)) % $(12))), dtype=tl.float32), tl.where(((r119_z / $(8)) % $(32)) < $(16), tl.load($(candidate_vlogsumexp_1_ptr) + (((r119_z % $(8)) * $(4)) + ((((r119_z / $(8)) % $(32)) - $(12)) % $(4))), dtype=tl.float32), tl.load($(candidate_vlogsumexp_2_ptr) + (((r119_z % $(8)) * $(16)) + ((((r119_z / $(8)) % $(32)) - $(16)) % $(16))), dtype=tl.float32)))
    r119_value = r119_value.to(tl.float32)
    r119_value = tl.broadcast_to(r119_value, [$(256)])
    tl.store($(candidate_vcat_7_ptr) + r119_z, r119_value)
    r120_row = tl.arange($(32))
    r120_col = tl.arange($(128))
    r120_lane = tl.arange($(512))
    for r120_group in range($(8)) {
      r120_left = tl.load($(candidate_vcat_6_ptr) + (((((((r120_group % $(8)) * $(16384)) + (r120_row[:, None] * $(512)) + r120_lane[None, :]) / $(512)) % $(32)) * $(4096)) + ((((((r120_group % $(8)) * $(16384)) + (r120_row[:, None] * $(512)) + r120_lane[None, :]) / $(16384)) % $(8)) * $(512)) + ((((r120_group % $(8)) * $(16384)) + (r120_row[:, None] * $(512)) + r120_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r120_right = tl.load($(candidate_vto_copy_5_ptr) + (((((((r120_group % $(8)) * $(65536)) + (r120_lane[:, None] * $(128)) + r120_col[None, :]) / $(65536)) % $(8)) * $(131072)) + (((((r120_group % $(8)) * $(65536)) + (r120_lane[:, None] * $(128)) + r120_col[None, :]) % $(128)) * $(512)) + (((((r120_group % $(8)) * $(65536)) + (r120_lane[:, None] * $(128)) + r120_col[None, :]) / $(128)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r120_value = tl.dot(r120_left, r120_right)
      r120_value = r120_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_6_ptr) + ((r120_group * $(4096)) + (r120_row[:, None] * $(128)) + r120_col[None, :]), r120_value)
    }
    r121_row = tl.arange($(32))
    r121_col = tl.arange($(512))
    r121_lane = tl.arange($(128))
    for r121_group in range($(8)) {
      r121_left = tl.load($(candidate_vmm_1_ptr) + (((((((r121_group % $(8)) * $(4096)) + (r121_row[:, None] * $(128)) + r121_lane[None, :]) / $(128)) % $(32)) * $(1536)) + ((((((r121_group % $(8)) * $(4096)) + (r121_row[:, None] * $(128)) + r121_lane[None, :]) / $(4096)) % $(8)) * $(192)) + ((((r121_group % $(8)) * $(4096)) + (r121_row[:, None] * $(128)) + r121_lane[None, :]) % $(128)) + $(49152)), dtype=tl.bfloat16)
      r121_right = tl.load($(candidate_vto_copy_5_ptr) + (((((((r121_group % $(8)) * $(65536)) + (r121_lane[:, None] * $(512)) + r121_col[None, :]) / $(65536)) % $(8)) * $(131072)) + ((((((r121_group % $(8)) * $(65536)) + (r121_lane[:, None] * $(512)) + r121_col[None, :]) / $(512)) % $(128)) * $(512)) + ((((r121_group % $(8)) * $(65536)) + (r121_lane[:, None] * $(512)) + r121_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r121_value = tl.dot(r121_left, r121_right)
      r121_value = r121_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_7_ptr) + ((r121_group * $(16384)) + (r121_row[:, None] * $(512)) + r121_col[None, :]), r121_value)
    }
    r122_z = tl.arange($(6144))
    r122_value = tl.load($(candidate_vadd_6_ptr) + ((((r122_z / $(64)) % $(12)) * $(512)) + (((r122_z / $(768)) % $(8)) * $(64)) + (r122_z % $(64)) + $(16384)), dtype=tl.bfloat16)
    r122_value = r122_value.to(tl.float32)
    r122_value = tl.broadcast_to(r122_value, [$(6144)])
    tl.store($(candidate_vto_copy_44_ptr) + r122_z, r122_value)
    r123_z = tl.arange($(3072))
    r123_value = tl.load($(candidate_vindex_2_ptr) + ((((r123_z / $(64)) % $(48)) * $(576)) + (r123_z % $(64)) + $(512)), dtype=tl.bfloat16)
    r123_value = r123_value.to(tl.float32)
    r123_value = tl.broadcast_to(r123_value, [$(3072)])
    tl.store($(candidate_vto_copy_45_ptr) + r123_z, r123_value)
    r124_row = tl.arange($(12))
    r124_col = tl.arange($(48))
    r124_lane = tl.arange($(64))
    for r124_group in range($(8)) {
      r124_left = tl.load($(candidate_vto_copy_44_ptr) + (((((((r124_group % $(8)) * $(768)) + (r124_row[:, None] * $(64)) + r124_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r124_group % $(8)) * $(768)) + (r124_row[:, None] * $(64)) + r124_lane[None, :]) / $(64)) % $(12)) * $(64)) + ((((r124_group % $(8)) * $(768)) + (r124_row[:, None] * $(64)) + r124_lane[None, :]) % $(64))), dtype=tl.float32)
      r124_right = tl.load($(candidate_vto_copy_45_ptr) + ((((((r124_group % $(8)) * $(3072)) + (r124_lane[:, None] * $(48)) + r124_col[None, :]) % $(48)) * $(64)) + (((((r124_group % $(8)) * $(3072)) + (r124_lane[:, None] * $(48)) + r124_col[None, :]) / $(48)) % $(64))), dtype=tl.float32)
      r124_value = tl.dot(r124_left, r124_right)
      r124_value = r124_value.to(tl.float32)
      tl.store($(candidate_vbmm_8_ptr) + ((r124_group * $(576)) + (r124_row[:, None] * $(48)) + r124_col[None, :]), r124_value)
    }
    r125_z = tl.arange($(49152))
    r125_value = tl.load($(candidate_vbmm_7_ptr) + ((((r125_z / $(6144)) % $(8)) * $(16384)) + (((r125_z / $(512)) % $(12)) * $(512)) + (r125_z % $(512))), dtype=tl.bfloat16)
    r125_value = r125_value.to(tl.float32)
    r125_value = tl.broadcast_to(r125_value, [$(49152)])
    tl.store($(candidate_vto_copy_46_ptr) + r125_z, r125_value)
    r126_z = tl.arange($(24576))
    r126_value = tl.load($(candidate_vindex_2_ptr) + ((((r126_z / $(512)) % $(48)) * $(576)) + (r126_z % $(512))), dtype=tl.bfloat16)
    r126_value = r126_value.to(tl.float32)
    r126_value = tl.broadcast_to(r126_value, [$(24576)])
    tl.store($(candidate_vto_copy_47_ptr) + r126_z, r126_value)
    r127_row = tl.arange($(96))
    r127_col = tl.arange($(48))
    r127_lane = tl.arange($(512))
    r127_left = tl.load($(candidate_vto_copy_46_ptr) + ((((((r127_row[:, None] * $(512)) + r127_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r127_row[:, None] * $(512)) + r127_lane[None, :]) / $(512)) % $(12)) * $(512)) + (((r127_row[:, None] * $(512)) + r127_lane[None, :]) % $(512))), dtype=tl.float32)
    r127_right = tl.load($(candidate_vto_copy_47_ptr) + (((((r127_lane[:, None] * $(48)) + r127_col[None, :]) % $(48)) * $(512)) + ((((r127_lane[:, None] * $(48)) + r127_col[None, :]) / $(48)) % $(512))), dtype=tl.float32)
    r127_value = tl.dot(r127_left, r127_right)
    r127_value = r127_value.to(tl.float32)
    tl.store($(candidate_vmm_9_ptr) + ((r127_row[:, None] * $(48)) + r127_col[None, :]), r127_value)
    r128_z = tl.arange($(4608))
    r128_value = (tl.load($(candidate_vbmm_8_ptr) + ((((r128_z / $(576)) % $(8)) * $(576)) + (((r128_z / $(48)) % $(12)) * $(48)) + (r128_z % $(48))), dtype=tl.float32) + tl.load($(candidate_vmm_9_ptr) + ((((r128_z / $(576)) % $(8)) * $(576)) + (((r128_z / $(48)) % $(12)) * $(48)) + (r128_z % $(48))), dtype=tl.float32))
    r128_value = r128_value.to(tl.float32)
    r128_value = tl.broadcast_to(r128_value, [$(4608)])
    tl.store($(candidate_vadd_14_ptr) + r128_z, r128_value)
    r129_z = tl.arange($(4608))
    r129_value = (tl.load($(candidate_vadd_14_ptr) + ((((r129_z / $(576)) % $(8)) * $(576)) + (((r129_z / $(48)) % $(12)) * $(48)) + (r129_z % $(48))), dtype=tl.float32) * 0.07216878364870322)
    r129_value = r129_value.to(tl.float32)
    r129_value = tl.broadcast_to(r129_value, [$(4608)])
    tl.store($(candidate_vmul_26_ptr) + r129_z, r129_value)
    r130_z = tl.arange($(12))
    r130_value = (tl.cast(r130_z, tl.int64) + tl.full([], $(48), dtype=tl.int64))
    r130_value = tl.broadcast_to(r130_value, [$(12)])
    tl.store($(candidate_vadd_15_ptr) + r130_z, r130_value)
    r131_z = tl.arange($(12))
    r131_value = (tl.load($(candidate_vadd_15_ptr) + r131_z, dtype=tl.int64) - tl.full([], $(12), dtype=tl.int64))
    r131_value = tl.broadcast_to(r131_value, [$(12)])
    tl.store($(candidate_vsub_12_ptr) + r131_z, r131_value)
    r132_z = tl.arange($(576))
    r132_value = (tl.cast((r132_z % $(48)), tl.int64) > tl.load($(candidate_vsub_12_ptr) + ((r132_z / $(48)) % $(12)), dtype=tl.int64))
    r132_value = tl.broadcast_to(r132_value, [$(576)])
    tl.store($(candidate_vgt_3_ptr) + r132_z, r132_value)
    r133_z = tl.arange($(4608))
    r133_value = tl.where(tl.load($(candidate_vgt_3_ptr) + ((((r133_z / $(48)) % $(12)) * $(48)) + (r133_z % $(48))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_26_ptr) + ((((r133_z / $(576)) % $(8)) * $(576)) + (((r133_z / $(48)) % $(12)) * $(48)) + (r133_z % $(48))), dtype=tl.float32))
    r133_value = r133_value.to(tl.float32)
    r133_value = tl.broadcast_to(r133_value, [$(4608)])
    tl.store($(candidate_vmasked_fill_3_ptr) + r133_z, r133_value)
    r134_z = tl.arange($(4608))
    r134_lane = tl.arange($(48))
    r134_row_values = tl.load($(candidate_vmasked_fill_3_ptr) + ((((r134_z[:, None] / $(576)) % $(8)) * $(576)) + (((r134_z[:, None] / $(48)) % $(12)) * $(48)) + (r134_lane[None, :] % $(48))), dtype=tl.float32)
    r134_maximum = tl.max(r134_row_values, axis=1, keep_dims=true)
    r134_weights = tl.exp(r134_row_values - r134_maximum)
    r134_total = tl.sum(r134_weights, axis=1)
    r134_maximum = tl.max(r134_row_values, axis=1)
    r134_value = tl.exp(tl.load($(candidate_vmasked_fill_3_ptr) + ((((r134_z / $(576)) % $(8)) * $(576)) + (((r134_z / $(48)) % $(12)) * $(48)) + (r134_z % $(48))), dtype=tl.float32) - r134_maximum) / r134_total
    r134_value = r134_value.to(tl.float32)
    r134_value = tl.broadcast_to(r134_value, [$(4608)])
    tl.store($(candidate_vsoftmax_3_ptr) + r134_z, r134_value)
    r135_z = tl.arange($(4608))
    r135_value = tl.load($(candidate_vsoftmax_3_ptr) + ((((r135_z / $(576)) % $(8)) * $(576)) + (((r135_z / $(48)) % $(12)) * $(48)) + (r135_z % $(48))), dtype=tl.float32)
    r135_value = r135_value.to(tl.bfloat16)
    r135_value = tl.broadcast_to(r135_value, [$(4608)])
    tl.store($(candidate_vto_copy_48_ptr) + r135_z, r135_value)
    r136_row = tl.arange($(96))
    r136_col = tl.arange($(512))
    r136_lane = tl.arange($(48))
    r136_left = tl.load($(candidate_vto_copy_48_ptr) + ((((((r136_row[:, None] * $(48)) + r136_lane[None, :]) / $(576)) % $(8)) * $(576)) + (((((r136_row[:, None] * $(48)) + r136_lane[None, :]) / $(48)) % $(12)) * $(48)) + (((r136_row[:, None] * $(48)) + r136_lane[None, :]) % $(48))), dtype=tl.bfloat16)
    r136_right = tl.load($(candidate_vindex_2_ptr) + ((((((r136_lane[:, None] * $(512)) + r136_col[None, :]) / $(512)) % $(48)) * $(576)) + (((r136_lane[:, None] * $(512)) + r136_col[None, :]) % $(512))), dtype=tl.bfloat16)
    r136_value = tl.dot(r136_left, r136_right)
    r136_value = r136_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_10_ptr) + ((r136_row[:, None] * $(512)) + r136_col[None, :]), r136_value)
    r137_z = tl.arange($(96))
    r137_lane = tl.arange($(48))
    r137_row_values = tl.load($(candidate_vmasked_fill_3_ptr) + ((((r137_z[:, None] / $(12)) % $(8)) * $(576)) + ((r137_z[:, None] % $(12)) * $(48)) + (r137_lane[None, :] % $(48))), dtype=tl.float32)
    r137_maximum = tl.max(r137_row_values, axis=1, keep_dims=true)
    r137_weights = tl.exp(r137_row_values - r137_maximum)
    r137_total = tl.sum(r137_weights, axis=1)
    r137_maximum = tl.max(r137_row_values, axis=1)
    r137_value = r137_maximum + tl.log(r137_total)
    r137_value = r137_value.to(tl.float32)
    r137_value = tl.broadcast_to(r137_value, [$(96)])
    tl.store($(candidate_vlogsumexp_3_ptr) + r137_z, r137_value)
    r138_z = tl.arange($(2048))
    r138_value = tl.load($(candidate_vadd_6_ptr) + ((((r138_z / $(64)) % $(4)) * $(512)) + (((r138_z / $(256)) % $(8)) * $(64)) + (r138_z % $(64)) + $(22528)), dtype=tl.bfloat16)
    r138_value = r138_value.to(tl.float32)
    r138_value = tl.broadcast_to(r138_value, [$(2048)])
    tl.store($(candidate_vto_copy_49_ptr) + r138_z, r138_value)
    r139_z = tl.arange($(1024))
    r139_value = tl.load($(candidate_vindex_2_ptr) + ((((r139_z / $(64)) % $(16)) * $(576)) + (r139_z % $(64)) + $(28160)), dtype=tl.bfloat16)
    r139_value = r139_value.to(tl.float32)
    r139_value = tl.broadcast_to(r139_value, [$(1024)])
    tl.store($(candidate_vto_copy_50_ptr) + r139_z, r139_value)
    r140_row = tl.arange($(4))
    r140_col = tl.arange($(16))
    r140_lane = tl.arange($(64))
    for r140_group in range($(8)) {
      r140_left = tl.load($(candidate_vto_copy_49_ptr) + (((((((r140_group % $(8)) * $(256)) + (r140_row[:, None] * $(64)) + r140_lane[None, :]) / $(256)) % $(8)) * $(256)) + ((((((r140_group % $(8)) * $(256)) + (r140_row[:, None] * $(64)) + r140_lane[None, :]) / $(64)) % $(4)) * $(64)) + ((((r140_group % $(8)) * $(256)) + (r140_row[:, None] * $(64)) + r140_lane[None, :]) % $(64))), dtype=tl.float32)
      r140_right = tl.load($(candidate_vto_copy_50_ptr) + ((((((r140_group % $(8)) * $(1024)) + (r140_lane[:, None] * $(16)) + r140_col[None, :]) % $(16)) * $(64)) + (((((r140_group % $(8)) * $(1024)) + (r140_lane[:, None] * $(16)) + r140_col[None, :]) / $(16)) % $(64))), dtype=tl.float32)
      r140_value = tl.dot(r140_left, r140_right)
      r140_value = r140_value.to(tl.float32)
      tl.store($(candidate_vbmm_9_ptr) + ((r140_group * $(64)) + (r140_row[:, None] * $(16)) + r140_col[None, :]), r140_value)
    }
    r141_z = tl.arange($(16384))
    r141_value = tl.load($(candidate_vbmm_7_ptr) + ((((r141_z / $(2048)) % $(8)) * $(16384)) + (((r141_z / $(512)) % $(4)) * $(512)) + (r141_z % $(512)) + $(6144)), dtype=tl.bfloat16)
    r141_value = r141_value.to(tl.float32)
    r141_value = tl.broadcast_to(r141_value, [$(16384)])
    tl.store($(candidate_vto_copy_51_ptr) + r141_z, r141_value)
    r142_z = tl.arange($(8192))
    r142_value = tl.load($(candidate_vindex_2_ptr) + ((((r142_z / $(512)) % $(16)) * $(576)) + (r142_z % $(512)) + $(27648)), dtype=tl.bfloat16)
    r142_value = r142_value.to(tl.float32)
    r142_value = tl.broadcast_to(r142_value, [$(8192)])
    tl.store($(candidate_vto_copy_52_ptr) + r142_z, r142_value)
    r143_row = tl.arange($(32))
    r143_col = tl.arange($(16))
    r143_lane = tl.arange($(512))
    r143_left = tl.load($(candidate_vto_copy_51_ptr) + ((((((r143_row[:, None] * $(512)) + r143_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + (((((r143_row[:, None] * $(512)) + r143_lane[None, :]) / $(512)) % $(4)) * $(512)) + (((r143_row[:, None] * $(512)) + r143_lane[None, :]) % $(512))), dtype=tl.float32)
    r143_right = tl.load($(candidate_vto_copy_52_ptr) + (((((r143_lane[:, None] * $(16)) + r143_col[None, :]) % $(16)) * $(512)) + ((((r143_lane[:, None] * $(16)) + r143_col[None, :]) / $(16)) % $(512))), dtype=tl.float32)
    r143_value = tl.dot(r143_left, r143_right)
    r143_value = r143_value.to(tl.float32)
    tl.store($(candidate_vmm_11_ptr) + ((r143_row[:, None] * $(16)) + r143_col[None, :]), r143_value)
    r144_z = tl.arange($(512))
    r144_value = (tl.load($(candidate_vbmm_9_ptr) + ((((r144_z / $(64)) % $(8)) * $(64)) + (((r144_z / $(16)) % $(4)) * $(16)) + (r144_z % $(16))), dtype=tl.float32) + tl.load($(candidate_vmm_11_ptr) + ((((r144_z / $(64)) % $(8)) * $(64)) + (((r144_z / $(16)) % $(4)) * $(16)) + (r144_z % $(16))), dtype=tl.float32))
    r144_value = r144_value.to(tl.float32)
    r144_value = tl.broadcast_to(r144_value, [$(512)])
    tl.store($(candidate_vadd_16_ptr) + r144_z, r144_value)
    r145_z = tl.arange($(512))
    r145_value = (tl.load($(candidate_vadd_16_ptr) + ((((r145_z / $(64)) % $(8)) * $(64)) + (((r145_z / $(16)) % $(4)) * $(16)) + (r145_z % $(16))), dtype=tl.float32) * 0.07216878364870322)
    r145_value = r145_value.to(tl.float32)
    r145_value = tl.broadcast_to(r145_value, [$(512)])
    tl.store($(candidate_vmul_27_ptr) + r145_z, r145_value)
    r146_z = tl.arange($(4))
    r146_value = (tl.cast(r146_z, tl.int64) + tl.full([], $(16), dtype=tl.int64))
    r146_value = tl.broadcast_to(r146_value, [$(4)])
    tl.store($(candidate_vadd_17_ptr) + r146_z, r146_value)
    r147_z = tl.arange($(4))
    r147_value = (tl.load($(candidate_vadd_17_ptr) + r147_z, dtype=tl.int64) - tl.full([], $(4), dtype=tl.int64))
    r147_value = tl.broadcast_to(r147_value, [$(4)])
    tl.store($(candidate_vsub_13_ptr) + r147_z, r147_value)
    r148_z = tl.arange($(64))
    r148_value = (tl.cast((r148_z % $(16)), tl.int64) > tl.load($(candidate_vsub_13_ptr) + ((r148_z / $(16)) % $(4)), dtype=tl.int64))
    r148_value = tl.broadcast_to(r148_value, [$(64)])
    tl.store($(candidate_vgt_4_ptr) + r148_z, r148_value)
    r149_z = tl.arange($(512))
    r149_value = tl.where(tl.load($(candidate_vgt_4_ptr) + ((((r149_z / $(16)) % $(4)) * $(16)) + (r149_z % $(16))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_27_ptr) + ((((r149_z / $(64)) % $(8)) * $(64)) + (((r149_z / $(16)) % $(4)) * $(16)) + (r149_z % $(16))), dtype=tl.float32))
    r149_value = r149_value.to(tl.float32)
    r149_value = tl.broadcast_to(r149_value, [$(512)])
    tl.store($(candidate_vmasked_fill_4_ptr) + r149_z, r149_value)
    r150_z = tl.arange($(512))
    r150_lane = tl.arange($(16))
    r150_row_values = tl.load($(candidate_vmasked_fill_4_ptr) + ((((r150_z[:, None] / $(64)) % $(8)) * $(64)) + (((r150_z[:, None] / $(16)) % $(4)) * $(16)) + (r150_lane[None, :] % $(16))), dtype=tl.float32)
    r150_maximum = tl.max(r150_row_values, axis=1, keep_dims=true)
    r150_weights = tl.exp(r150_row_values - r150_maximum)
    r150_total = tl.sum(r150_weights, axis=1)
    r150_maximum = tl.max(r150_row_values, axis=1)
    r150_value = tl.exp(tl.load($(candidate_vmasked_fill_4_ptr) + ((((r150_z / $(64)) % $(8)) * $(64)) + (((r150_z / $(16)) % $(4)) * $(16)) + (r150_z % $(16))), dtype=tl.float32) - r150_maximum) / r150_total
    r150_value = r150_value.to(tl.float32)
    r150_value = tl.broadcast_to(r150_value, [$(512)])
    tl.store($(candidate_vsoftmax_4_ptr) + r150_z, r150_value)
    r151_z = tl.arange($(512))
    r151_value = tl.load($(candidate_vsoftmax_4_ptr) + ((((r151_z / $(64)) % $(8)) * $(64)) + (((r151_z / $(16)) % $(4)) * $(16)) + (r151_z % $(16))), dtype=tl.float32)
    r151_value = r151_value.to(tl.bfloat16)
    r151_value = tl.broadcast_to(r151_value, [$(512)])
    tl.store($(candidate_vto_copy_53_ptr) + r151_z, r151_value)
    r152_row = tl.arange($(32))
    r152_col = tl.arange($(512))
    r152_lane = tl.arange($(16))
    r152_left = tl.load($(candidate_vto_copy_53_ptr) + ((((((r152_row[:, None] * $(16)) + r152_lane[None, :]) / $(64)) % $(8)) * $(64)) + (((((r152_row[:, None] * $(16)) + r152_lane[None, :]) / $(16)) % $(4)) * $(16)) + (((r152_row[:, None] * $(16)) + r152_lane[None, :]) % $(16))), dtype=tl.bfloat16)
    r152_right = tl.load($(candidate_vindex_2_ptr) + ((((((r152_lane[:, None] * $(512)) + r152_col[None, :]) / $(512)) % $(16)) * $(576)) + (((r152_lane[:, None] * $(512)) + r152_col[None, :]) % $(512)) + $(27648)), dtype=tl.bfloat16)
    r152_value = tl.dot(r152_left, r152_right)
    r152_value = r152_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_12_ptr) + ((r152_row[:, None] * $(512)) + r152_col[None, :]), r152_value)
    r153_z = tl.arange($(32))
    r153_lane = tl.arange($(16))
    r153_row_values = tl.load($(candidate_vmasked_fill_4_ptr) + ((((r153_z[:, None] / $(4)) % $(8)) * $(64)) + ((r153_z[:, None] % $(4)) * $(16)) + (r153_lane[None, :] % $(16))), dtype=tl.float32)
    r153_maximum = tl.max(r153_row_values, axis=1, keep_dims=true)
    r153_weights = tl.exp(r153_row_values - r153_maximum)
    r153_total = tl.sum(r153_weights, axis=1)
    r153_maximum = tl.max(r153_row_values, axis=1)
    r153_value = r153_maximum + tl.log(r153_total)
    r153_value = r153_value.to(tl.float32)
    r153_value = tl.broadcast_to(r153_value, [$(32)])
    tl.store($(candidate_vlogsumexp_4_ptr) + r153_z, r153_value)
    r154_z = tl.arange($(8192))
    r154_value = tl.load($(candidate_vadd_6_ptr) + ((((r154_z / $(64)) % $(16)) * $(512)) + (((r154_z / $(1024)) % $(8)) * $(64)) + (r154_z % $(64)) + $(24576)), dtype=tl.bfloat16)
    r154_value = r154_value.to(tl.float32)
    r154_value = tl.broadcast_to(r154_value, [$(8192)])
    tl.store($(candidate_vto_copy_54_ptr) + r154_z, r154_value)
    r155_z = tl.arange($(4096))
    r155_value = tl.load($(candidate_vindex_2_ptr) + ((((r155_z / $(64)) % $(64)) * $(576)) + (r155_z % $(64)) + $(37376)), dtype=tl.bfloat16)
    r155_value = r155_value.to(tl.float32)
    r155_value = tl.broadcast_to(r155_value, [$(4096)])
    tl.store($(candidate_vto_copy_55_ptr) + r155_z, r155_value)
    r156_row = tl.arange($(16))
    r156_col = tl.arange($(64))
    r156_lane = tl.arange($(64))
    for r156_group in range($(8)) {
      r156_left = tl.load($(candidate_vto_copy_54_ptr) + (((((((r156_group % $(8)) * $(1024)) + (r156_row[:, None] * $(64)) + r156_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + ((((((r156_group % $(8)) * $(1024)) + (r156_row[:, None] * $(64)) + r156_lane[None, :]) / $(64)) % $(16)) * $(64)) + ((((r156_group % $(8)) * $(1024)) + (r156_row[:, None] * $(64)) + r156_lane[None, :]) % $(64))), dtype=tl.float32)
      r156_right = tl.load($(candidate_vto_copy_55_ptr) + ((((((r156_group % $(8)) * $(4096)) + (r156_lane[:, None] * $(64)) + r156_col[None, :]) % $(64)) * $(64)) + (((((r156_group % $(8)) * $(4096)) + (r156_lane[:, None] * $(64)) + r156_col[None, :]) / $(64)) % $(64))), dtype=tl.float32)
      r156_value = tl.dot(r156_left, r156_right)
      r156_value = r156_value.to(tl.float32)
      tl.store($(candidate_vbmm_10_ptr) + ((r156_group * $(1024)) + (r156_row[:, None] * $(64)) + r156_col[None, :]), r156_value)
    }
    r157_z = tl.arange($(65536))
    r157_value = tl.load($(candidate_vbmm_7_ptr) + ((((r157_z / $(8192)) % $(8)) * $(16384)) + (((r157_z / $(512)) % $(16)) * $(512)) + (r157_z % $(512)) + $(8192)), dtype=tl.bfloat16)
    r157_value = r157_value.to(tl.float32)
    r157_value = tl.broadcast_to(r157_value, [$(65536)])
    tl.store($(candidate_vto_copy_56_ptr) + r157_z, r157_value)
    r158_z = tl.arange($(32768))
    r158_value = tl.load($(candidate_vindex_2_ptr) + ((((r158_z / $(512)) % $(64)) * $(576)) + (r158_z % $(512)) + $(36864)), dtype=tl.bfloat16)
    r158_value = r158_value.to(tl.float32)
    r158_value = tl.broadcast_to(r158_value, [$(32768)])
    tl.store($(candidate_vto_copy_57_ptr) + r158_z, r158_value)
    r159_row = tl.arange($(128))
    r159_col = tl.arange($(64))
    r159_lane = tl.arange($(512))
    r159_left = tl.load($(candidate_vto_copy_56_ptr) + ((((((r159_row[:, None] * $(512)) + r159_lane[None, :]) / $(8192)) % $(8)) * $(8192)) + (((((r159_row[:, None] * $(512)) + r159_lane[None, :]) / $(512)) % $(16)) * $(512)) + (((r159_row[:, None] * $(512)) + r159_lane[None, :]) % $(512))), dtype=tl.float32)
    r159_right = tl.load($(candidate_vto_copy_57_ptr) + (((((r159_lane[:, None] * $(64)) + r159_col[None, :]) % $(64)) * $(512)) + ((((r159_lane[:, None] * $(64)) + r159_col[None, :]) / $(64)) % $(512))), dtype=tl.float32)
    r159_value = tl.dot(r159_left, r159_right)
    r159_value = r159_value.to(tl.float32)
    tl.store($(candidate_vmm_13_ptr) + ((r159_row[:, None] * $(64)) + r159_col[None, :]), r159_value)
    r160_z = tl.arange($(8192))
    r160_value = (tl.load($(candidate_vbmm_10_ptr) + ((((r160_z / $(1024)) % $(8)) * $(1024)) + (((r160_z / $(64)) % $(16)) * $(64)) + (r160_z % $(64))), dtype=tl.float32) + tl.load($(candidate_vmm_13_ptr) + ((((r160_z / $(1024)) % $(8)) * $(1024)) + (((r160_z / $(64)) % $(16)) * $(64)) + (r160_z % $(64))), dtype=tl.float32))
    r160_value = r160_value.to(tl.float32)
    r160_value = tl.broadcast_to(r160_value, [$(8192)])
    tl.store($(candidate_vadd_18_ptr) + r160_z, r160_value)
    r161_z = tl.arange($(8192))
    r161_value = (tl.load($(candidate_vadd_18_ptr) + ((((r161_z / $(1024)) % $(8)) * $(1024)) + (((r161_z / $(64)) % $(16)) * $(64)) + (r161_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r161_value = r161_value.to(tl.float32)
    r161_value = tl.broadcast_to(r161_value, [$(8192)])
    tl.store($(candidate_vmul_28_ptr) + r161_z, r161_value)
    r162_z = tl.arange($(16))
    r162_value = (tl.cast(r162_z, tl.int64) + tl.full([], $(64), dtype=tl.int64))
    r162_value = tl.broadcast_to(r162_value, [$(16)])
    tl.store($(candidate_vadd_19_ptr) + r162_z, r162_value)
    r163_z = tl.arange($(16))
    r163_value = (tl.load($(candidate_vadd_19_ptr) + r163_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r163_value = tl.broadcast_to(r163_value, [$(16)])
    tl.store($(candidate_vsub_14_ptr) + r163_z, r163_value)
    r164_z = tl.arange($(1024))
    r164_value = (tl.cast((r164_z % $(64)), tl.int64) > tl.load($(candidate_vsub_14_ptr) + ((r164_z / $(64)) % $(16)), dtype=tl.int64))
    r164_value = tl.broadcast_to(r164_value, [$(1024)])
    tl.store($(candidate_vgt_5_ptr) + r164_z, r164_value)
    r165_z = tl.arange($(8192))
    r165_value = tl.where(tl.load($(candidate_vgt_5_ptr) + ((((r165_z / $(64)) % $(16)) * $(64)) + (r165_z % $(64))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_28_ptr) + ((((r165_z / $(1024)) % $(8)) * $(1024)) + (((r165_z / $(64)) % $(16)) * $(64)) + (r165_z % $(64))), dtype=tl.float32))
    r165_value = r165_value.to(tl.float32)
    r165_value = tl.broadcast_to(r165_value, [$(8192)])
    tl.store($(candidate_vmasked_fill_5_ptr) + r165_z, r165_value)
    r166_z = tl.arange($(8192))
    r166_lane = tl.arange($(64))
    r166_row_values = tl.load($(candidate_vmasked_fill_5_ptr) + ((((r166_z[:, None] / $(1024)) % $(8)) * $(1024)) + (((r166_z[:, None] / $(64)) % $(16)) * $(64)) + (r166_lane[None, :] % $(64))), dtype=tl.float32)
    r166_maximum = tl.max(r166_row_values, axis=1, keep_dims=true)
    r166_weights = tl.exp(r166_row_values - r166_maximum)
    r166_total = tl.sum(r166_weights, axis=1)
    r166_maximum = tl.max(r166_row_values, axis=1)
    r166_value = tl.exp(tl.load($(candidate_vmasked_fill_5_ptr) + ((((r166_z / $(1024)) % $(8)) * $(1024)) + (((r166_z / $(64)) % $(16)) * $(64)) + (r166_z % $(64))), dtype=tl.float32) - r166_maximum) / r166_total
    r166_value = r166_value.to(tl.float32)
    r166_value = tl.broadcast_to(r166_value, [$(8192)])
    tl.store($(candidate_vsoftmax_5_ptr) + r166_z, r166_value)
    r167_z = tl.arange($(8192))
    r167_value = tl.load($(candidate_vsoftmax_5_ptr) + ((((r167_z / $(1024)) % $(8)) * $(1024)) + (((r167_z / $(64)) % $(16)) * $(64)) + (r167_z % $(64))), dtype=tl.float32)
    r167_value = r167_value.to(tl.bfloat16)
    r167_value = tl.broadcast_to(r167_value, [$(8192)])
    tl.store($(candidate_vto_copy_58_ptr) + r167_z, r167_value)
    r168_row = tl.arange($(128))
    r168_col = tl.arange($(512))
    r168_lane = tl.arange($(64))
    r168_left = tl.load($(candidate_vto_copy_58_ptr) + ((((((r168_row[:, None] * $(64)) + r168_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + (((((r168_row[:, None] * $(64)) + r168_lane[None, :]) / $(64)) % $(16)) * $(64)) + (((r168_row[:, None] * $(64)) + r168_lane[None, :]) % $(64))), dtype=tl.bfloat16)
    r168_right = tl.load($(candidate_vindex_2_ptr) + ((((((r168_lane[:, None] * $(512)) + r168_col[None, :]) / $(512)) % $(64)) * $(576)) + (((r168_lane[:, None] * $(512)) + r168_col[None, :]) % $(512)) + $(36864)), dtype=tl.bfloat16)
    r168_value = tl.dot(r168_left, r168_right)
    r168_value = r168_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_14_ptr) + ((r168_row[:, None] * $(512)) + r168_col[None, :]), r168_value)
    r169_z = tl.arange($(128))
    r169_lane = tl.arange($(64))
    r169_row_values = tl.load($(candidate_vmasked_fill_5_ptr) + ((((r169_z[:, None] / $(16)) % $(8)) * $(1024)) + ((r169_z[:, None] % $(16)) * $(64)) + (r169_lane[None, :] % $(64))), dtype=tl.float32)
    r169_maximum = tl.max(r169_row_values, axis=1, keep_dims=true)
    r169_weights = tl.exp(r169_row_values - r169_maximum)
    r169_total = tl.sum(r169_weights, axis=1)
    r169_maximum = tl.max(r169_row_values, axis=1)
    r169_value = r169_maximum + tl.log(r169_total)
    r169_value = r169_value.to(tl.float32)
    r169_value = tl.broadcast_to(r169_value, [$(128)])
    tl.store($(candidate_vlogsumexp_5_ptr) + r169_z, r169_value)
    r170_z = tl.arange($(131072))
    r170_value = tl.where(((r170_z / $(4096)) % $(32)) < $(12), tl.load($(candidate_vmm_10_ptr) + (((((((r170_z / $(512)) % $(8)) * $(12)) + (((r170_z / $(4096)) % $(32)) - $(0))) % $(96)) * $(512)) + (r170_z % $(512))), dtype=tl.bfloat16), tl.where(((r170_z / $(4096)) % $(32)) < $(16), tl.load($(candidate_vmm_12_ptr) + (((((((r170_z / $(512)) % $(8)) * $(4)) + (((r170_z / $(4096)) % $(32)) - $(12))) % $(32)) * $(512)) + (r170_z % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vmm_14_ptr) + (((((((r170_z / $(512)) % $(8)) * $(16)) + (((r170_z / $(4096)) % $(32)) - $(16))) % $(128)) * $(512)) + (r170_z % $(512))), dtype=tl.bfloat16)))
    r170_value = r170_value.to(tl.bfloat16)
    r170_value = tl.broadcast_to(r170_value, [$(131072)])
    tl.store($(candidate_vcat_8_ptr) + r170_z, r170_value)
    r171_z = tl.arange($(256))
    r171_value = tl.where(((r171_z / $(8)) % $(32)) < $(12), tl.load($(candidate_vlogsumexp_3_ptr) + (((r171_z % $(8)) * $(12)) + ((((r171_z / $(8)) % $(32)) - $(0)) % $(12))), dtype=tl.float32), tl.where(((r171_z / $(8)) % $(32)) < $(16), tl.load($(candidate_vlogsumexp_4_ptr) + (((r171_z % $(8)) * $(4)) + ((((r171_z / $(8)) % $(32)) - $(12)) % $(4))), dtype=tl.float32), tl.load($(candidate_vlogsumexp_5_ptr) + (((r171_z % $(8)) * $(16)) + ((((r171_z / $(8)) % $(32)) - $(16)) % $(16))), dtype=tl.float32)))
    r171_value = r171_value.to(tl.float32)
    r171_value = tl.broadcast_to(r171_value, [$(256)])
    tl.store($(candidate_vcat_9_ptr) + r171_z, r171_value)
    r172_row = tl.arange($(32))
    r172_col = tl.arange($(128))
    r172_lane = tl.arange($(512))
    for r172_group in range($(8)) {
      r172_left = tl.load($(candidate_vcat_8_ptr) + (((((((r172_group % $(8)) * $(16384)) + (r172_row[:, None] * $(512)) + r172_lane[None, :]) / $(512)) % $(32)) * $(4096)) + ((((((r172_group % $(8)) * $(16384)) + (r172_row[:, None] * $(512)) + r172_lane[None, :]) / $(16384)) % $(8)) * $(512)) + ((((r172_group % $(8)) * $(16384)) + (r172_row[:, None] * $(512)) + r172_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r172_right = tl.load($(candidate_vto_copy_5_ptr) + (((((((r172_group % $(8)) * $(65536)) + (r172_lane[:, None] * $(128)) + r172_col[None, :]) / $(65536)) % $(8)) * $(131072)) + (((((r172_group % $(8)) * $(65536)) + (r172_lane[:, None] * $(128)) + r172_col[None, :]) % $(128)) * $(512)) + (((((r172_group % $(8)) * $(65536)) + (r172_lane[:, None] * $(128)) + r172_col[None, :]) / $(128)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r172_value = tl.dot(r172_left, r172_right)
      r172_value = r172_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_11_ptr) + ((r172_group * $(4096)) + (r172_row[:, None] * $(128)) + r172_col[None, :]), r172_value)
    }
    r173_row = tl.arange($(32))
    r173_col = tl.arange($(512))
    r173_lane = tl.arange($(128))
    for r173_group in range($(8)) {
      r173_left = tl.load($(candidate_vmm_1_ptr) + (((((((r173_group % $(8)) * $(4096)) + (r173_row[:, None] * $(128)) + r173_lane[None, :]) / $(128)) % $(32)) * $(1536)) + ((((((r173_group % $(8)) * $(4096)) + (r173_row[:, None] * $(128)) + r173_lane[None, :]) / $(4096)) % $(8)) * $(192)) + ((((r173_group % $(8)) * $(4096)) + (r173_row[:, None] * $(128)) + r173_lane[None, :]) % $(128)) + $(98304)), dtype=tl.bfloat16)
      r173_right = tl.load($(candidate_vto_copy_5_ptr) + (((((((r173_group % $(8)) * $(65536)) + (r173_lane[:, None] * $(512)) + r173_col[None, :]) / $(65536)) % $(8)) * $(131072)) + ((((((r173_group % $(8)) * $(65536)) + (r173_lane[:, None] * $(512)) + r173_col[None, :]) / $(512)) % $(128)) * $(512)) + ((((r173_group % $(8)) * $(65536)) + (r173_lane[:, None] * $(512)) + r173_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r173_value = tl.dot(r173_left, r173_right)
      r173_value = r173_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_12_ptr) + ((r173_group * $(16384)) + (r173_row[:, None] * $(512)) + r173_col[None, :]), r173_value)
    }
    r174_z = tl.arange($(6144))
    r174_value = tl.load($(candidate_vadd_6_ptr) + ((((r174_z / $(64)) % $(12)) * $(512)) + (((r174_z / $(768)) % $(8)) * $(64)) + (r174_z % $(64)) + $(32768)), dtype=tl.bfloat16)
    r174_value = r174_value.to(tl.float32)
    r174_value = tl.broadcast_to(r174_value, [$(6144)])
    tl.store($(candidate_vto_copy_59_ptr) + r174_z, r174_value)
    r175_z = tl.arange($(1536))
    r175_value = tl.load($(candidate_vindex_2_ptr) + ((((r175_z / $(64)) % $(24)) * $(576)) + (r175_z % $(64)) + $(512)), dtype=tl.bfloat16)
    r175_value = r175_value.to(tl.float32)
    r175_value = tl.broadcast_to(r175_value, [$(1536)])
    tl.store($(candidate_vto_copy_60_ptr) + r175_z, r175_value)
    r176_row = tl.arange($(12))
    r176_col = tl.arange($(24))
    r176_lane = tl.arange($(64))
    for r176_group in range($(8)) {
      r176_left = tl.load($(candidate_vto_copy_59_ptr) + (((((((r176_group % $(8)) * $(768)) + (r176_row[:, None] * $(64)) + r176_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r176_group % $(8)) * $(768)) + (r176_row[:, None] * $(64)) + r176_lane[None, :]) / $(64)) % $(12)) * $(64)) + ((((r176_group % $(8)) * $(768)) + (r176_row[:, None] * $(64)) + r176_lane[None, :]) % $(64))), dtype=tl.float32)
      r176_right = tl.load($(candidate_vto_copy_60_ptr) + ((((((r176_group % $(8)) * $(1536)) + (r176_lane[:, None] * $(24)) + r176_col[None, :]) % $(24)) * $(64)) + (((((r176_group % $(8)) * $(1536)) + (r176_lane[:, None] * $(24)) + r176_col[None, :]) / $(24)) % $(64))), dtype=tl.float32)
      r176_value = tl.dot(r176_left, r176_right)
      r176_value = r176_value.to(tl.float32)
      tl.store($(candidate_vbmm_13_ptr) + ((r176_group * $(288)) + (r176_row[:, None] * $(24)) + r176_col[None, :]), r176_value)
    }
    r177_z = tl.arange($(49152))
    r177_value = tl.load($(candidate_vbmm_12_ptr) + ((((r177_z / $(6144)) % $(8)) * $(16384)) + (((r177_z / $(512)) % $(12)) * $(512)) + (r177_z % $(512))), dtype=tl.bfloat16)
    r177_value = r177_value.to(tl.float32)
    r177_value = tl.broadcast_to(r177_value, [$(49152)])
    tl.store($(candidate_vto_copy_61_ptr) + r177_z, r177_value)
    r178_z = tl.arange($(12288))
    r178_value = tl.load($(candidate_vindex_2_ptr) + ((((r178_z / $(512)) % $(24)) * $(576)) + (r178_z % $(512))), dtype=tl.bfloat16)
    r178_value = r178_value.to(tl.float32)
    r178_value = tl.broadcast_to(r178_value, [$(12288)])
    tl.store($(candidate_vto_copy_62_ptr) + r178_z, r178_value)
    r179_row = tl.arange($(96))
    r179_col = tl.arange($(24))
    r179_lane = tl.arange($(512))
    r179_left = tl.load($(candidate_vto_copy_61_ptr) + ((((((r179_row[:, None] * $(512)) + r179_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r179_row[:, None] * $(512)) + r179_lane[None, :]) / $(512)) % $(12)) * $(512)) + (((r179_row[:, None] * $(512)) + r179_lane[None, :]) % $(512))), dtype=tl.float32)
    r179_right = tl.load($(candidate_vto_copy_62_ptr) + (((((r179_lane[:, None] * $(24)) + r179_col[None, :]) % $(24)) * $(512)) + ((((r179_lane[:, None] * $(24)) + r179_col[None, :]) / $(24)) % $(512))), dtype=tl.float32)
    r179_value = tl.dot(r179_left, r179_right)
    r179_value = r179_value.to(tl.float32)
    tl.store($(candidate_vmm_15_ptr) + ((r179_row[:, None] * $(24)) + r179_col[None, :]), r179_value)
    r180_z = tl.arange($(2304))
    r180_value = (tl.load($(candidate_vbmm_13_ptr) + ((((r180_z / $(288)) % $(8)) * $(288)) + (((r180_z / $(24)) % $(12)) * $(24)) + (r180_z % $(24))), dtype=tl.float32) + tl.load($(candidate_vmm_15_ptr) + ((((r180_z / $(288)) % $(8)) * $(288)) + (((r180_z / $(24)) % $(12)) * $(24)) + (r180_z % $(24))), dtype=tl.float32))
    r180_value = r180_value.to(tl.float32)
    r180_value = tl.broadcast_to(r180_value, [$(2304)])
    tl.store($(candidate_vadd_20_ptr) + r180_z, r180_value)
    r181_z = tl.arange($(2304))
    r181_value = (tl.load($(candidate_vadd_20_ptr) + ((((r181_z / $(288)) % $(8)) * $(288)) + (((r181_z / $(24)) % $(12)) * $(24)) + (r181_z % $(24))), dtype=tl.float32) * 0.07216878364870322)
    r181_value = r181_value.to(tl.float32)
    r181_value = tl.broadcast_to(r181_value, [$(2304)])
    tl.store($(candidate_vmul_29_ptr) + r181_z, r181_value)
    r182_z = tl.arange($(12))
    r182_value = (tl.cast(r182_z, tl.int64) + tl.full([], $(24), dtype=tl.int64))
    r182_value = tl.broadcast_to(r182_value, [$(12)])
    tl.store($(candidate_vadd_21_ptr) + r182_z, r182_value)
    r183_z = tl.arange($(12))
    r183_value = (tl.load($(candidate_vadd_21_ptr) + r183_z, dtype=tl.int64) - tl.full([], $(12), dtype=tl.int64))
    r183_value = tl.broadcast_to(r183_value, [$(12)])
    tl.store($(candidate_vsub_15_ptr) + r183_z, r183_value)
    r184_z = tl.arange($(288))
    r184_value = (tl.cast((r184_z % $(24)), tl.int64) > tl.load($(candidate_vsub_15_ptr) + ((r184_z / $(24)) % $(12)), dtype=tl.int64))
    r184_value = tl.broadcast_to(r184_value, [$(288)])
    tl.store($(candidate_vgt_6_ptr) + r184_z, r184_value)
    r185_z = tl.arange($(2304))
    r185_value = tl.where(tl.load($(candidate_vgt_6_ptr) + ((((r185_z / $(24)) % $(12)) * $(24)) + (r185_z % $(24))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_29_ptr) + ((((r185_z / $(288)) % $(8)) * $(288)) + (((r185_z / $(24)) % $(12)) * $(24)) + (r185_z % $(24))), dtype=tl.float32))
    r185_value = r185_value.to(tl.float32)
    r185_value = tl.broadcast_to(r185_value, [$(2304)])
    tl.store($(candidate_vmasked_fill_6_ptr) + r185_z, r185_value)
    r186_z = tl.arange($(2304))
    r186_lane = tl.arange($(24))
    r186_row_values = tl.load($(candidate_vmasked_fill_6_ptr) + ((((r186_z[:, None] / $(288)) % $(8)) * $(288)) + (((r186_z[:, None] / $(24)) % $(12)) * $(24)) + (r186_lane[None, :] % $(24))), dtype=tl.float32)
    r186_maximum = tl.max(r186_row_values, axis=1, keep_dims=true)
    r186_weights = tl.exp(r186_row_values - r186_maximum)
    r186_total = tl.sum(r186_weights, axis=1)
    r186_maximum = tl.max(r186_row_values, axis=1)
    r186_value = tl.exp(tl.load($(candidate_vmasked_fill_6_ptr) + ((((r186_z / $(288)) % $(8)) * $(288)) + (((r186_z / $(24)) % $(12)) * $(24)) + (r186_z % $(24))), dtype=tl.float32) - r186_maximum) / r186_total
    r186_value = r186_value.to(tl.float32)
    r186_value = tl.broadcast_to(r186_value, [$(2304)])
    tl.store($(candidate_vsoftmax_6_ptr) + r186_z, r186_value)
    r187_z = tl.arange($(2304))
    r187_value = tl.load($(candidate_vsoftmax_6_ptr) + ((((r187_z / $(288)) % $(8)) * $(288)) + (((r187_z / $(24)) % $(12)) * $(24)) + (r187_z % $(24))), dtype=tl.float32)
    r187_value = r187_value.to(tl.bfloat16)
    r187_value = tl.broadcast_to(r187_value, [$(2304)])
    tl.store($(candidate_vto_copy_63_ptr) + r187_z, r187_value)
    r188_row = tl.arange($(96))
    r188_col = tl.arange($(512))
    r188_lane = tl.arange($(24))
    r188_left = tl.load($(candidate_vto_copy_63_ptr) + ((((((r188_row[:, None] * $(24)) + r188_lane[None, :]) / $(288)) % $(8)) * $(288)) + (((((r188_row[:, None] * $(24)) + r188_lane[None, :]) / $(24)) % $(12)) * $(24)) + (((r188_row[:, None] * $(24)) + r188_lane[None, :]) % $(24))), dtype=tl.bfloat16)
    r188_right = tl.load($(candidate_vindex_2_ptr) + ((((((r188_lane[:, None] * $(512)) + r188_col[None, :]) / $(512)) % $(24)) * $(576)) + (((r188_lane[:, None] * $(512)) + r188_col[None, :]) % $(512))), dtype=tl.bfloat16)
    r188_value = tl.dot(r188_left, r188_right)
    r188_value = r188_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_16_ptr) + ((r188_row[:, None] * $(512)) + r188_col[None, :]), r188_value)
    r189_z = tl.arange($(96))
    r189_lane = tl.arange($(24))
    r189_row_values = tl.load($(candidate_vmasked_fill_6_ptr) + ((((r189_z[:, None] / $(12)) % $(8)) * $(288)) + ((r189_z[:, None] % $(12)) * $(24)) + (r189_lane[None, :] % $(24))), dtype=tl.float32)
    r189_maximum = tl.max(r189_row_values, axis=1, keep_dims=true)
    r189_weights = tl.exp(r189_row_values - r189_maximum)
    r189_total = tl.sum(r189_weights, axis=1)
    r189_maximum = tl.max(r189_row_values, axis=1)
    r189_value = r189_maximum + tl.log(r189_total)
    r189_value = r189_value.to(tl.float32)
    r189_value = tl.broadcast_to(r189_value, [$(96)])
    tl.store($(candidate_vlogsumexp_6_ptr) + r189_z, r189_value)
    r190_z = tl.arange($(2048))
    r190_value = tl.load($(candidate_vadd_6_ptr) + ((((r190_z / $(64)) % $(4)) * $(512)) + (((r190_z / $(256)) % $(8)) * $(64)) + (r190_z % $(64)) + $(38912)), dtype=tl.bfloat16)
    r190_value = r190_value.to(tl.float32)
    r190_value = tl.broadcast_to(r190_value, [$(2048)])
    tl.store($(candidate_vto_copy_64_ptr) + r190_z, r190_value)
    r191_z = tl.arange($(512))
    r191_value = tl.load($(candidate_vindex_2_ptr) + ((((r191_z / $(64)) % $(8)) * $(576)) + (r191_z % $(64)) + $(28160)), dtype=tl.bfloat16)
    r191_value = r191_value.to(tl.float32)
    r191_value = tl.broadcast_to(r191_value, [$(512)])
    tl.store($(candidate_vto_copy_65_ptr) + r191_z, r191_value)
    r192_row = tl.arange($(4))
    r192_col = tl.arange($(8))
    r192_lane = tl.arange($(64))
    for r192_group in range($(8)) {
      r192_left = tl.load($(candidate_vto_copy_64_ptr) + (((((((r192_group % $(8)) * $(256)) + (r192_row[:, None] * $(64)) + r192_lane[None, :]) / $(256)) % $(8)) * $(256)) + ((((((r192_group % $(8)) * $(256)) + (r192_row[:, None] * $(64)) + r192_lane[None, :]) / $(64)) % $(4)) * $(64)) + ((((r192_group % $(8)) * $(256)) + (r192_row[:, None] * $(64)) + r192_lane[None, :]) % $(64))), dtype=tl.float32)
      r192_right = tl.load($(candidate_vto_copy_65_ptr) + ((((((r192_group % $(8)) * $(512)) + (r192_lane[:, None] * $(8)) + r192_col[None, :]) % $(8)) * $(64)) + (((((r192_group % $(8)) * $(512)) + (r192_lane[:, None] * $(8)) + r192_col[None, :]) / $(8)) % $(64))), dtype=tl.float32)
      r192_value = tl.dot(r192_left, r192_right)
      r192_value = r192_value.to(tl.float32)
      tl.store($(candidate_vbmm_14_ptr) + ((r192_group * $(32)) + (r192_row[:, None] * $(8)) + r192_col[None, :]), r192_value)
    }
    r193_z = tl.arange($(16384))
    r193_value = tl.load($(candidate_vbmm_12_ptr) + ((((r193_z / $(2048)) % $(8)) * $(16384)) + (((r193_z / $(512)) % $(4)) * $(512)) + (r193_z % $(512)) + $(6144)), dtype=tl.bfloat16)
    r193_value = r193_value.to(tl.float32)
    r193_value = tl.broadcast_to(r193_value, [$(16384)])
    tl.store($(candidate_vto_copy_66_ptr) + r193_z, r193_value)
    r194_z = tl.arange($(4096))
    r194_value = tl.load($(candidate_vindex_2_ptr) + ((((r194_z / $(512)) % $(8)) * $(576)) + (r194_z % $(512)) + $(27648)), dtype=tl.bfloat16)
    r194_value = r194_value.to(tl.float32)
    r194_value = tl.broadcast_to(r194_value, [$(4096)])
    tl.store($(candidate_vto_copy_67_ptr) + r194_z, r194_value)
    r195_row = tl.arange($(32))
    r195_col = tl.arange($(8))
    r195_lane = tl.arange($(512))
    r195_left = tl.load($(candidate_vto_copy_66_ptr) + ((((((r195_row[:, None] * $(512)) + r195_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + (((((r195_row[:, None] * $(512)) + r195_lane[None, :]) / $(512)) % $(4)) * $(512)) + (((r195_row[:, None] * $(512)) + r195_lane[None, :]) % $(512))), dtype=tl.float32)
    r195_right = tl.load($(candidate_vto_copy_67_ptr) + (((((r195_lane[:, None] * $(8)) + r195_col[None, :]) % $(8)) * $(512)) + ((((r195_lane[:, None] * $(8)) + r195_col[None, :]) / $(8)) % $(512))), dtype=tl.float32)
    r195_value = tl.dot(r195_left, r195_right)
    r195_value = r195_value.to(tl.float32)
    tl.store($(candidate_vmm_17_ptr) + ((r195_row[:, None] * $(8)) + r195_col[None, :]), r195_value)
    r196_z = tl.arange($(256))
    r196_value = (tl.load($(candidate_vbmm_14_ptr) + ((((r196_z / $(32)) % $(8)) * $(32)) + (((r196_z / $(8)) % $(4)) * $(8)) + (r196_z % $(8))), dtype=tl.float32) + tl.load($(candidate_vmm_17_ptr) + ((((r196_z / $(32)) % $(8)) * $(32)) + (((r196_z / $(8)) % $(4)) * $(8)) + (r196_z % $(8))), dtype=tl.float32))
    r196_value = r196_value.to(tl.float32)
    r196_value = tl.broadcast_to(r196_value, [$(256)])
    tl.store($(candidate_vadd_22_ptr) + r196_z, r196_value)
    r197_z = tl.arange($(256))
    r197_value = (tl.load($(candidate_vadd_22_ptr) + ((((r197_z / $(32)) % $(8)) * $(32)) + (((r197_z / $(8)) % $(4)) * $(8)) + (r197_z % $(8))), dtype=tl.float32) * 0.07216878364870322)
    r197_value = r197_value.to(tl.float32)
    r197_value = tl.broadcast_to(r197_value, [$(256)])
    tl.store($(candidate_vmul_30_ptr) + r197_z, r197_value)
    r198_z = tl.arange($(4))
    r198_value = (tl.cast(r198_z, tl.int64) + tl.full([], $(8), dtype=tl.int64))
    r198_value = tl.broadcast_to(r198_value, [$(4)])
    tl.store($(candidate_vadd_23_ptr) + r198_z, r198_value)
    r199_z = tl.arange($(4))
    r199_value = (tl.load($(candidate_vadd_23_ptr) + r199_z, dtype=tl.int64) - tl.full([], $(4), dtype=tl.int64))
    r199_value = tl.broadcast_to(r199_value, [$(4)])
    tl.store($(candidate_vsub_16_ptr) + r199_z, r199_value)
    r200_z = tl.arange($(32))
    r200_value = (tl.cast((r200_z % $(8)), tl.int64) > tl.load($(candidate_vsub_16_ptr) + ((r200_z / $(8)) % $(4)), dtype=tl.int64))
    r200_value = tl.broadcast_to(r200_value, [$(32)])
    tl.store($(candidate_vgt_7_ptr) + r200_z, r200_value)
    r201_z = tl.arange($(256))
    r201_value = tl.where(tl.load($(candidate_vgt_7_ptr) + ((((r201_z / $(8)) % $(4)) * $(8)) + (r201_z % $(8))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_30_ptr) + ((((r201_z / $(32)) % $(8)) * $(32)) + (((r201_z / $(8)) % $(4)) * $(8)) + (r201_z % $(8))), dtype=tl.float32))
    r201_value = r201_value.to(tl.float32)
    r201_value = tl.broadcast_to(r201_value, [$(256)])
    tl.store($(candidate_vmasked_fill_7_ptr) + r201_z, r201_value)
    r202_z = tl.arange($(256))
    r202_lane = tl.arange($(8))
    r202_row_values = tl.load($(candidate_vmasked_fill_7_ptr) + ((((r202_z[:, None] / $(32)) % $(8)) * $(32)) + (((r202_z[:, None] / $(8)) % $(4)) * $(8)) + (r202_lane[None, :] % $(8))), dtype=tl.float32)
    r202_maximum = tl.max(r202_row_values, axis=1, keep_dims=true)
    r202_weights = tl.exp(r202_row_values - r202_maximum)
    r202_total = tl.sum(r202_weights, axis=1)
    r202_maximum = tl.max(r202_row_values, axis=1)
    r202_value = tl.exp(tl.load($(candidate_vmasked_fill_7_ptr) + ((((r202_z / $(32)) % $(8)) * $(32)) + (((r202_z / $(8)) % $(4)) * $(8)) + (r202_z % $(8))), dtype=tl.float32) - r202_maximum) / r202_total
    r202_value = r202_value.to(tl.float32)
    r202_value = tl.broadcast_to(r202_value, [$(256)])
    tl.store($(candidate_vsoftmax_7_ptr) + r202_z, r202_value)
    r203_z = tl.arange($(256))
    r203_value = tl.load($(candidate_vsoftmax_7_ptr) + ((((r203_z / $(32)) % $(8)) * $(32)) + (((r203_z / $(8)) % $(4)) * $(8)) + (r203_z % $(8))), dtype=tl.float32)
    r203_value = r203_value.to(tl.bfloat16)
    r203_value = tl.broadcast_to(r203_value, [$(256)])
    tl.store($(candidate_vto_copy_68_ptr) + r203_z, r203_value)
    r204_row = tl.arange($(32))
    r204_col = tl.arange($(512))
    r204_lane = tl.arange($(8))
    r204_left = tl.load($(candidate_vto_copy_68_ptr) + ((((((r204_row[:, None] * $(8)) + r204_lane[None, :]) / $(32)) % $(8)) * $(32)) + (((((r204_row[:, None] * $(8)) + r204_lane[None, :]) / $(8)) % $(4)) * $(8)) + (((r204_row[:, None] * $(8)) + r204_lane[None, :]) % $(8))), dtype=tl.bfloat16)
    r204_right = tl.load($(candidate_vindex_2_ptr) + ((((((r204_lane[:, None] * $(512)) + r204_col[None, :]) / $(512)) % $(8)) * $(576)) + (((r204_lane[:, None] * $(512)) + r204_col[None, :]) % $(512)) + $(27648)), dtype=tl.bfloat16)
    r204_value = tl.dot(r204_left, r204_right)
    r204_value = r204_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_18_ptr) + ((r204_row[:, None] * $(512)) + r204_col[None, :]), r204_value)
    r205_z = tl.arange($(32))
    r205_lane = tl.arange($(8))
    r205_row_values = tl.load($(candidate_vmasked_fill_7_ptr) + ((((r205_z[:, None] / $(4)) % $(8)) * $(32)) + ((r205_z[:, None] % $(4)) * $(8)) + (r205_lane[None, :] % $(8))), dtype=tl.float32)
    r205_maximum = tl.max(r205_row_values, axis=1, keep_dims=true)
    r205_weights = tl.exp(r205_row_values - r205_maximum)
    r205_total = tl.sum(r205_weights, axis=1)
    r205_maximum = tl.max(r205_row_values, axis=1)
    r205_value = r205_maximum + tl.log(r205_total)
    r205_value = r205_value.to(tl.float32)
    r205_value = tl.broadcast_to(r205_value, [$(32)])
    tl.store($(candidate_vlogsumexp_7_ptr) + r205_z, r205_value)
    r206_z = tl.arange($(8192))
    r206_value = tl.load($(candidate_vadd_6_ptr) + ((((r206_z / $(64)) % $(16)) * $(512)) + (((r206_z / $(1024)) % $(8)) * $(64)) + (r206_z % $(64)) + $(40960)), dtype=tl.bfloat16)
    r206_value = r206_value.to(tl.float32)
    r206_value = tl.broadcast_to(r206_value, [$(8192)])
    tl.store($(candidate_vto_copy_69_ptr) + r206_z, r206_value)
    r207_z = tl.arange($(2048))
    r207_value = tl.load($(candidate_vindex_2_ptr) + ((((r207_z / $(64)) % $(32)) * $(576)) + (r207_z % $(64)) + $(37376)), dtype=tl.bfloat16)
    r207_value = r207_value.to(tl.float32)
    r207_value = tl.broadcast_to(r207_value, [$(2048)])
    tl.store($(candidate_vto_copy_70_ptr) + r207_z, r207_value)
    r208_row = tl.arange($(16))
    r208_col = tl.arange($(32))
    r208_lane = tl.arange($(64))
    for r208_group in range($(8)) {
      r208_left = tl.load($(candidate_vto_copy_69_ptr) + (((((((r208_group % $(8)) * $(1024)) + (r208_row[:, None] * $(64)) + r208_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + ((((((r208_group % $(8)) * $(1024)) + (r208_row[:, None] * $(64)) + r208_lane[None, :]) / $(64)) % $(16)) * $(64)) + ((((r208_group % $(8)) * $(1024)) + (r208_row[:, None] * $(64)) + r208_lane[None, :]) % $(64))), dtype=tl.float32)
      r208_right = tl.load($(candidate_vto_copy_70_ptr) + ((((((r208_group % $(8)) * $(2048)) + (r208_lane[:, None] * $(32)) + r208_col[None, :]) % $(32)) * $(64)) + (((((r208_group % $(8)) * $(2048)) + (r208_lane[:, None] * $(32)) + r208_col[None, :]) / $(32)) % $(64))), dtype=tl.float32)
      r208_value = tl.dot(r208_left, r208_right)
      r208_value = r208_value.to(tl.float32)
      tl.store($(candidate_vbmm_15_ptr) + ((r208_group * $(512)) + (r208_row[:, None] * $(32)) + r208_col[None, :]), r208_value)
    }
    r209_z = tl.arange($(65536))
    r209_value = tl.load($(candidate_vbmm_12_ptr) + ((((r209_z / $(8192)) % $(8)) * $(16384)) + (((r209_z / $(512)) % $(16)) * $(512)) + (r209_z % $(512)) + $(8192)), dtype=tl.bfloat16)
    r209_value = r209_value.to(tl.float32)
    r209_value = tl.broadcast_to(r209_value, [$(65536)])
    tl.store($(candidate_vto_copy_71_ptr) + r209_z, r209_value)
    r210_z = tl.arange($(16384))
    r210_value = tl.load($(candidate_vindex_2_ptr) + ((((r210_z / $(512)) % $(32)) * $(576)) + (r210_z % $(512)) + $(36864)), dtype=tl.bfloat16)
    r210_value = r210_value.to(tl.float32)
    r210_value = tl.broadcast_to(r210_value, [$(16384)])
    tl.store($(candidate_vto_copy_72_ptr) + r210_z, r210_value)
    r211_row = tl.arange($(128))
    r211_col = tl.arange($(32))
    r211_lane = tl.arange($(512))
    r211_left = tl.load($(candidate_vto_copy_71_ptr) + ((((((r211_row[:, None] * $(512)) + r211_lane[None, :]) / $(8192)) % $(8)) * $(8192)) + (((((r211_row[:, None] * $(512)) + r211_lane[None, :]) / $(512)) % $(16)) * $(512)) + (((r211_row[:, None] * $(512)) + r211_lane[None, :]) % $(512))), dtype=tl.float32)
    r211_right = tl.load($(candidate_vto_copy_72_ptr) + (((((r211_lane[:, None] * $(32)) + r211_col[None, :]) % $(32)) * $(512)) + ((((r211_lane[:, None] * $(32)) + r211_col[None, :]) / $(32)) % $(512))), dtype=tl.float32)
    r211_value = tl.dot(r211_left, r211_right)
    r211_value = r211_value.to(tl.float32)
    tl.store($(candidate_vmm_19_ptr) + ((r211_row[:, None] * $(32)) + r211_col[None, :]), r211_value)
    r212_z = tl.arange($(4096))
    r212_value = (tl.load($(candidate_vbmm_15_ptr) + ((((r212_z / $(512)) % $(8)) * $(512)) + (((r212_z / $(32)) % $(16)) * $(32)) + (r212_z % $(32))), dtype=tl.float32) + tl.load($(candidate_vmm_19_ptr) + ((((r212_z / $(512)) % $(8)) * $(512)) + (((r212_z / $(32)) % $(16)) * $(32)) + (r212_z % $(32))), dtype=tl.float32))
    r212_value = r212_value.to(tl.float32)
    r212_value = tl.broadcast_to(r212_value, [$(4096)])
    tl.store($(candidate_vadd_24_ptr) + r212_z, r212_value)
    r213_z = tl.arange($(4096))
    r213_value = (tl.load($(candidate_vadd_24_ptr) + ((((r213_z / $(512)) % $(8)) * $(512)) + (((r213_z / $(32)) % $(16)) * $(32)) + (r213_z % $(32))), dtype=tl.float32) * 0.07216878364870322)
    r213_value = r213_value.to(tl.float32)
    r213_value = tl.broadcast_to(r213_value, [$(4096)])
    tl.store($(candidate_vmul_31_ptr) + r213_z, r213_value)
    r214_z = tl.arange($(16))
    r214_value = (tl.cast(r214_z, tl.int64) + tl.full([], $(32), dtype=tl.int64))
    r214_value = tl.broadcast_to(r214_value, [$(16)])
    tl.store($(candidate_vadd_25_ptr) + r214_z, r214_value)
    r215_z = tl.arange($(16))
    r215_value = (tl.load($(candidate_vadd_25_ptr) + r215_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r215_value = tl.broadcast_to(r215_value, [$(16)])
    tl.store($(candidate_vsub_17_ptr) + r215_z, r215_value)
    r216_z = tl.arange($(512))
    r216_value = (tl.cast((r216_z % $(32)), tl.int64) > tl.load($(candidate_vsub_17_ptr) + ((r216_z / $(32)) % $(16)), dtype=tl.int64))
    r216_value = tl.broadcast_to(r216_value, [$(512)])
    tl.store($(candidate_vgt_8_ptr) + r216_z, r216_value)
    r217_z = tl.arange($(4096))
    r217_value = tl.where(tl.load($(candidate_vgt_8_ptr) + ((((r217_z / $(32)) % $(16)) * $(32)) + (r217_z % $(32))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_31_ptr) + ((((r217_z / $(512)) % $(8)) * $(512)) + (((r217_z / $(32)) % $(16)) * $(32)) + (r217_z % $(32))), dtype=tl.float32))
    r217_value = r217_value.to(tl.float32)
    r217_value = tl.broadcast_to(r217_value, [$(4096)])
    tl.store($(candidate_vmasked_fill_8_ptr) + r217_z, r217_value)
    r218_z = tl.arange($(4096))
    r218_lane = tl.arange($(32))
    r218_row_values = tl.load($(candidate_vmasked_fill_8_ptr) + ((((r218_z[:, None] / $(512)) % $(8)) * $(512)) + (((r218_z[:, None] / $(32)) % $(16)) * $(32)) + (r218_lane[None, :] % $(32))), dtype=tl.float32)
    r218_maximum = tl.max(r218_row_values, axis=1, keep_dims=true)
    r218_weights = tl.exp(r218_row_values - r218_maximum)
    r218_total = tl.sum(r218_weights, axis=1)
    r218_maximum = tl.max(r218_row_values, axis=1)
    r218_value = tl.exp(tl.load($(candidate_vmasked_fill_8_ptr) + ((((r218_z / $(512)) % $(8)) * $(512)) + (((r218_z / $(32)) % $(16)) * $(32)) + (r218_z % $(32))), dtype=tl.float32) - r218_maximum) / r218_total
    r218_value = r218_value.to(tl.float32)
    r218_value = tl.broadcast_to(r218_value, [$(4096)])
    tl.store($(candidate_vsoftmax_8_ptr) + r218_z, r218_value)
    r219_z = tl.arange($(4096))
    r219_value = tl.load($(candidate_vsoftmax_8_ptr) + ((((r219_z / $(512)) % $(8)) * $(512)) + (((r219_z / $(32)) % $(16)) * $(32)) + (r219_z % $(32))), dtype=tl.float32)
    r219_value = r219_value.to(tl.bfloat16)
    r219_value = tl.broadcast_to(r219_value, [$(4096)])
    tl.store($(candidate_vto_copy_73_ptr) + r219_z, r219_value)
    r220_row = tl.arange($(128))
    r220_col = tl.arange($(512))
    r220_lane = tl.arange($(32))
    r220_left = tl.load($(candidate_vto_copy_73_ptr) + ((((((r220_row[:, None] * $(32)) + r220_lane[None, :]) / $(512)) % $(8)) * $(512)) + (((((r220_row[:, None] * $(32)) + r220_lane[None, :]) / $(32)) % $(16)) * $(32)) + (((r220_row[:, None] * $(32)) + r220_lane[None, :]) % $(32))), dtype=tl.bfloat16)
    r220_right = tl.load($(candidate_vindex_2_ptr) + ((((((r220_lane[:, None] * $(512)) + r220_col[None, :]) / $(512)) % $(32)) * $(576)) + (((r220_lane[:, None] * $(512)) + r220_col[None, :]) % $(512)) + $(36864)), dtype=tl.bfloat16)
    r220_value = tl.dot(r220_left, r220_right)
    r220_value = r220_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_20_ptr) + ((r220_row[:, None] * $(512)) + r220_col[None, :]), r220_value)
    r221_z = tl.arange($(128))
    r221_lane = tl.arange($(32))
    r221_row_values = tl.load($(candidate_vmasked_fill_8_ptr) + ((((r221_z[:, None] / $(16)) % $(8)) * $(512)) + ((r221_z[:, None] % $(16)) * $(32)) + (r221_lane[None, :] % $(32))), dtype=tl.float32)
    r221_maximum = tl.max(r221_row_values, axis=1, keep_dims=true)
    r221_weights = tl.exp(r221_row_values - r221_maximum)
    r221_total = tl.sum(r221_weights, axis=1)
    r221_maximum = tl.max(r221_row_values, axis=1)
    r221_value = r221_maximum + tl.log(r221_total)
    r221_value = r221_value.to(tl.float32)
    r221_value = tl.broadcast_to(r221_value, [$(128)])
    tl.store($(candidate_vlogsumexp_8_ptr) + r221_z, r221_value)
    r222_z = tl.arange($(131072))
    r222_value = tl.where(((r222_z / $(4096)) % $(32)) < $(12), tl.load($(candidate_vmm_16_ptr) + (((((((r222_z / $(512)) % $(8)) * $(12)) + (((r222_z / $(4096)) % $(32)) - $(0))) % $(96)) * $(512)) + (r222_z % $(512))), dtype=tl.bfloat16), tl.where(((r222_z / $(4096)) % $(32)) < $(16), tl.load($(candidate_vmm_18_ptr) + (((((((r222_z / $(512)) % $(8)) * $(4)) + (((r222_z / $(4096)) % $(32)) - $(12))) % $(32)) * $(512)) + (r222_z % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vmm_20_ptr) + (((((((r222_z / $(512)) % $(8)) * $(16)) + (((r222_z / $(4096)) % $(32)) - $(16))) % $(128)) * $(512)) + (r222_z % $(512))), dtype=tl.bfloat16)))
    r222_value = r222_value.to(tl.bfloat16)
    r222_value = tl.broadcast_to(r222_value, [$(131072)])
    tl.store($(candidate_vcat_10_ptr) + r222_z, r222_value)
    r223_z = tl.arange($(256))
    r223_value = tl.where(((r223_z / $(8)) % $(32)) < $(12), tl.load($(candidate_vlogsumexp_6_ptr) + (((r223_z % $(8)) * $(12)) + ((((r223_z / $(8)) % $(32)) - $(0)) % $(12))), dtype=tl.float32), tl.where(((r223_z / $(8)) % $(32)) < $(16), tl.load($(candidate_vlogsumexp_7_ptr) + (((r223_z % $(8)) * $(4)) + ((((r223_z / $(8)) % $(32)) - $(12)) % $(4))), dtype=tl.float32), tl.load($(candidate_vlogsumexp_8_ptr) + (((r223_z % $(8)) * $(16)) + ((((r223_z / $(8)) % $(32)) - $(16)) % $(16))), dtype=tl.float32)))
    r223_value = r223_value.to(tl.float32)
    r223_value = tl.broadcast_to(r223_value, [$(256)])
    tl.store($(candidate_vcat_11_ptr) + r223_z, r223_value)
    r224_row = tl.arange($(32))
    r224_col = tl.arange($(128))
    r224_lane = tl.arange($(512))
    for r224_group in range($(8)) {
      r224_left = tl.load($(candidate_vcat_10_ptr) + (((((((r224_group % $(8)) * $(16384)) + (r224_row[:, None] * $(512)) + r224_lane[None, :]) / $(512)) % $(32)) * $(4096)) + ((((((r224_group % $(8)) * $(16384)) + (r224_row[:, None] * $(512)) + r224_lane[None, :]) / $(16384)) % $(8)) * $(512)) + ((((r224_group % $(8)) * $(16384)) + (r224_row[:, None] * $(512)) + r224_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r224_right = tl.load($(candidate_vto_copy_5_ptr) + (((((((r224_group % $(8)) * $(65536)) + (r224_lane[:, None] * $(128)) + r224_col[None, :]) / $(65536)) % $(8)) * $(131072)) + (((((r224_group % $(8)) * $(65536)) + (r224_lane[:, None] * $(128)) + r224_col[None, :]) % $(128)) * $(512)) + (((((r224_group % $(8)) * $(65536)) + (r224_lane[:, None] * $(128)) + r224_col[None, :]) / $(128)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r224_value = tl.dot(r224_left, r224_right)
      r224_value = r224_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_16_ptr) + ((r224_group * $(4096)) + (r224_row[:, None] * $(128)) + r224_col[None, :]), r224_value)
    }
    r225_row = tl.arange($(32))
    r225_col = tl.arange($(512))
    r225_lane = tl.arange($(128))
    for r225_group in range($(8)) {
      r225_left = tl.load($(candidate_vmm_1_ptr) + (((((((r225_group % $(8)) * $(4096)) + (r225_row[:, None] * $(128)) + r225_lane[None, :]) / $(128)) % $(32)) * $(1536)) + ((((((r225_group % $(8)) * $(4096)) + (r225_row[:, None] * $(128)) + r225_lane[None, :]) / $(4096)) % $(8)) * $(192)) + ((((r225_group % $(8)) * $(4096)) + (r225_row[:, None] * $(128)) + r225_lane[None, :]) % $(128)) + $(147456)), dtype=tl.bfloat16)
      r225_right = tl.load($(candidate_vto_copy_5_ptr) + (((((((r225_group % $(8)) * $(65536)) + (r225_lane[:, None] * $(512)) + r225_col[None, :]) / $(65536)) % $(8)) * $(131072)) + ((((((r225_group % $(8)) * $(65536)) + (r225_lane[:, None] * $(512)) + r225_col[None, :]) / $(512)) % $(128)) * $(512)) + ((((r225_group % $(8)) * $(65536)) + (r225_lane[:, None] * $(512)) + r225_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r225_value = tl.dot(r225_left, r225_right)
      r225_value = r225_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_17_ptr) + ((r225_group * $(16384)) + (r225_row[:, None] * $(512)) + r225_col[None, :]), r225_value)
    }
    r226_z = tl.arange($(6144))
    r226_value = tl.load($(candidate_vadd_6_ptr) + ((((r226_z / $(64)) % $(12)) * $(512)) + (((r226_z / $(768)) % $(8)) * $(64)) + (r226_z % $(64)) + $(49152)), dtype=tl.bfloat16)
    r226_value = r226_value.to(tl.float32)
    r226_value = tl.broadcast_to(r226_value, [$(6144)])
    tl.store($(candidate_vto_copy_74_ptr) + r226_z, r226_value)
    r227_z = tl.arange($(2304))
    r227_value = tl.load($(candidate_vindex_2_ptr) + ((((r227_z / $(64)) % $(36)) * $(576)) + (r227_z % $(64)) + $(512)), dtype=tl.bfloat16)
    r227_value = r227_value.to(tl.float32)
    r227_value = tl.broadcast_to(r227_value, [$(2304)])
    tl.store($(candidate_vto_copy_75_ptr) + r227_z, r227_value)
    r228_row = tl.arange($(12))
    r228_col = tl.arange($(36))
    r228_lane = tl.arange($(64))
    for r228_group in range($(8)) {
      r228_left = tl.load($(candidate_vto_copy_74_ptr) + (((((((r228_group % $(8)) * $(768)) + (r228_row[:, None] * $(64)) + r228_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r228_group % $(8)) * $(768)) + (r228_row[:, None] * $(64)) + r228_lane[None, :]) / $(64)) % $(12)) * $(64)) + ((((r228_group % $(8)) * $(768)) + (r228_row[:, None] * $(64)) + r228_lane[None, :]) % $(64))), dtype=tl.float32)
      r228_right = tl.load($(candidate_vto_copy_75_ptr) + ((((((r228_group % $(8)) * $(2304)) + (r228_lane[:, None] * $(36)) + r228_col[None, :]) % $(36)) * $(64)) + (((((r228_group % $(8)) * $(2304)) + (r228_lane[:, None] * $(36)) + r228_col[None, :]) / $(36)) % $(64))), dtype=tl.float32)
      r228_value = tl.dot(r228_left, r228_right)
      r228_value = r228_value.to(tl.float32)
      tl.store($(candidate_vbmm_18_ptr) + ((r228_group * $(432)) + (r228_row[:, None] * $(36)) + r228_col[None, :]), r228_value)
    }
    r229_z = tl.arange($(49152))
    r229_value = tl.load($(candidate_vbmm_17_ptr) + ((((r229_z / $(6144)) % $(8)) * $(16384)) + (((r229_z / $(512)) % $(12)) * $(512)) + (r229_z % $(512))), dtype=tl.bfloat16)
    r229_value = r229_value.to(tl.float32)
    r229_value = tl.broadcast_to(r229_value, [$(49152)])
    tl.store($(candidate_vto_copy_76_ptr) + r229_z, r229_value)
    r230_z = tl.arange($(18432))
    r230_value = tl.load($(candidate_vindex_2_ptr) + ((((r230_z / $(512)) % $(36)) * $(576)) + (r230_z % $(512))), dtype=tl.bfloat16)
    r230_value = r230_value.to(tl.float32)
    r230_value = tl.broadcast_to(r230_value, [$(18432)])
    tl.store($(candidate_vto_copy_77_ptr) + r230_z, r230_value)
    r231_row = tl.arange($(96))
    r231_col = tl.arange($(36))
    r231_lane = tl.arange($(512))
    r231_left = tl.load($(candidate_vto_copy_76_ptr) + ((((((r231_row[:, None] * $(512)) + r231_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r231_row[:, None] * $(512)) + r231_lane[None, :]) / $(512)) % $(12)) * $(512)) + (((r231_row[:, None] * $(512)) + r231_lane[None, :]) % $(512))), dtype=tl.float32)
    r231_right = tl.load($(candidate_vto_copy_77_ptr) + (((((r231_lane[:, None] * $(36)) + r231_col[None, :]) % $(36)) * $(512)) + ((((r231_lane[:, None] * $(36)) + r231_col[None, :]) / $(36)) % $(512))), dtype=tl.float32)
    r231_value = tl.dot(r231_left, r231_right)
    r231_value = r231_value.to(tl.float32)
    tl.store($(candidate_vmm_21_ptr) + ((r231_row[:, None] * $(36)) + r231_col[None, :]), r231_value)
    r232_z = tl.arange($(3456))
    r232_value = (tl.load($(candidate_vbmm_18_ptr) + ((((r232_z / $(432)) % $(8)) * $(432)) + (((r232_z / $(36)) % $(12)) * $(36)) + (r232_z % $(36))), dtype=tl.float32) + tl.load($(candidate_vmm_21_ptr) + ((((r232_z / $(432)) % $(8)) * $(432)) + (((r232_z / $(36)) % $(12)) * $(36)) + (r232_z % $(36))), dtype=tl.float32))
    r232_value = r232_value.to(tl.float32)
    r232_value = tl.broadcast_to(r232_value, [$(3456)])
    tl.store($(candidate_vadd_26_ptr) + r232_z, r232_value)
    r233_z = tl.arange($(3456))
    r233_value = (tl.load($(candidate_vadd_26_ptr) + ((((r233_z / $(432)) % $(8)) * $(432)) + (((r233_z / $(36)) % $(12)) * $(36)) + (r233_z % $(36))), dtype=tl.float32) * 0.07216878364870322)
    r233_value = r233_value.to(tl.float32)
    r233_value = tl.broadcast_to(r233_value, [$(3456)])
    tl.store($(candidate_vmul_32_ptr) + r233_z, r233_value)
    r234_z = tl.arange($(12))
    r234_value = (tl.cast(r234_z, tl.int64) + tl.full([], $(36), dtype=tl.int64))
    r234_value = tl.broadcast_to(r234_value, [$(12)])
    tl.store($(candidate_vadd_27_ptr) + r234_z, r234_value)
    r235_z = tl.arange($(12))
    r235_value = (tl.load($(candidate_vadd_27_ptr) + r235_z, dtype=tl.int64) - tl.full([], $(12), dtype=tl.int64))
    r235_value = tl.broadcast_to(r235_value, [$(12)])
    tl.store($(candidate_vsub_18_ptr) + r235_z, r235_value)
    r236_z = tl.arange($(432))
    r236_value = (tl.cast((r236_z % $(36)), tl.int64) > tl.load($(candidate_vsub_18_ptr) + ((r236_z / $(36)) % $(12)), dtype=tl.int64))
    r236_value = tl.broadcast_to(r236_value, [$(432)])
    tl.store($(candidate_vgt_9_ptr) + r236_z, r236_value)
    r237_z = tl.arange($(3456))
    r237_value = tl.where(tl.load($(candidate_vgt_9_ptr) + ((((r237_z / $(36)) % $(12)) * $(36)) + (r237_z % $(36))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_32_ptr) + ((((r237_z / $(432)) % $(8)) * $(432)) + (((r237_z / $(36)) % $(12)) * $(36)) + (r237_z % $(36))), dtype=tl.float32))
    r237_value = r237_value.to(tl.float32)
    r237_value = tl.broadcast_to(r237_value, [$(3456)])
    tl.store($(candidate_vmasked_fill_9_ptr) + r237_z, r237_value)
    r238_z = tl.arange($(3456))
    r238_lane = tl.arange($(36))
    r238_row_values = tl.load($(candidate_vmasked_fill_9_ptr) + ((((r238_z[:, None] / $(432)) % $(8)) * $(432)) + (((r238_z[:, None] / $(36)) % $(12)) * $(36)) + (r238_lane[None, :] % $(36))), dtype=tl.float32)
    r238_maximum = tl.max(r238_row_values, axis=1, keep_dims=true)
    r238_weights = tl.exp(r238_row_values - r238_maximum)
    r238_total = tl.sum(r238_weights, axis=1)
    r238_maximum = tl.max(r238_row_values, axis=1)
    r238_value = tl.exp(tl.load($(candidate_vmasked_fill_9_ptr) + ((((r238_z / $(432)) % $(8)) * $(432)) + (((r238_z / $(36)) % $(12)) * $(36)) + (r238_z % $(36))), dtype=tl.float32) - r238_maximum) / r238_total
    r238_value = r238_value.to(tl.float32)
    r238_value = tl.broadcast_to(r238_value, [$(3456)])
    tl.store($(candidate_vsoftmax_9_ptr) + r238_z, r238_value)
    r239_z = tl.arange($(3456))
    r239_value = tl.load($(candidate_vsoftmax_9_ptr) + ((((r239_z / $(432)) % $(8)) * $(432)) + (((r239_z / $(36)) % $(12)) * $(36)) + (r239_z % $(36))), dtype=tl.float32)
    r239_value = r239_value.to(tl.bfloat16)
    r239_value = tl.broadcast_to(r239_value, [$(3456)])
    tl.store($(candidate_vto_copy_78_ptr) + r239_z, r239_value)
    r240_row = tl.arange($(96))
    r240_col = tl.arange($(512))
    r240_lane = tl.arange($(36))
    r240_left = tl.load($(candidate_vto_copy_78_ptr) + ((((((r240_row[:, None] * $(36)) + r240_lane[None, :]) / $(432)) % $(8)) * $(432)) + (((((r240_row[:, None] * $(36)) + r240_lane[None, :]) / $(36)) % $(12)) * $(36)) + (((r240_row[:, None] * $(36)) + r240_lane[None, :]) % $(36))), dtype=tl.bfloat16)
    r240_right = tl.load($(candidate_vindex_2_ptr) + ((((((r240_lane[:, None] * $(512)) + r240_col[None, :]) / $(512)) % $(36)) * $(576)) + (((r240_lane[:, None] * $(512)) + r240_col[None, :]) % $(512))), dtype=tl.bfloat16)
    r240_value = tl.dot(r240_left, r240_right)
    r240_value = r240_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_22_ptr) + ((r240_row[:, None] * $(512)) + r240_col[None, :]), r240_value)
    r241_z = tl.arange($(96))
    r241_lane = tl.arange($(36))
    r241_row_values = tl.load($(candidate_vmasked_fill_9_ptr) + ((((r241_z[:, None] / $(12)) % $(8)) * $(432)) + ((r241_z[:, None] % $(12)) * $(36)) + (r241_lane[None, :] % $(36))), dtype=tl.float32)
    r241_maximum = tl.max(r241_row_values, axis=1, keep_dims=true)
    r241_weights = tl.exp(r241_row_values - r241_maximum)
    r241_total = tl.sum(r241_weights, axis=1)
    r241_maximum = tl.max(r241_row_values, axis=1)
    r241_value = r241_maximum + tl.log(r241_total)
    r241_value = r241_value.to(tl.float32)
    r241_value = tl.broadcast_to(r241_value, [$(96)])
    tl.store($(candidate_vlogsumexp_9_ptr) + r241_z, r241_value)
    r242_z = tl.arange($(2048))
    r242_value = tl.load($(candidate_vadd_6_ptr) + ((((r242_z / $(64)) % $(4)) * $(512)) + (((r242_z / $(256)) % $(8)) * $(64)) + (r242_z % $(64)) + $(55296)), dtype=tl.bfloat16)
    r242_value = r242_value.to(tl.float32)
    r242_value = tl.broadcast_to(r242_value, [$(2048)])
    tl.store($(candidate_vto_copy_79_ptr) + r242_z, r242_value)
    r243_z = tl.arange($(768))
    r243_value = tl.load($(candidate_vindex_2_ptr) + ((((r243_z / $(64)) % $(12)) * $(576)) + (r243_z % $(64)) + $(28160)), dtype=tl.bfloat16)
    r243_value = r243_value.to(tl.float32)
    r243_value = tl.broadcast_to(r243_value, [$(768)])
    tl.store($(candidate_vto_copy_80_ptr) + r243_z, r243_value)
    r244_row = tl.arange($(4))
    r244_col = tl.arange($(12))
    r244_lane = tl.arange($(64))
    for r244_group in range($(8)) {
      r244_left = tl.load($(candidate_vto_copy_79_ptr) + (((((((r244_group % $(8)) * $(256)) + (r244_row[:, None] * $(64)) + r244_lane[None, :]) / $(256)) % $(8)) * $(256)) + ((((((r244_group % $(8)) * $(256)) + (r244_row[:, None] * $(64)) + r244_lane[None, :]) / $(64)) % $(4)) * $(64)) + ((((r244_group % $(8)) * $(256)) + (r244_row[:, None] * $(64)) + r244_lane[None, :]) % $(64))), dtype=tl.float32)
      r244_right = tl.load($(candidate_vto_copy_80_ptr) + ((((((r244_group % $(8)) * $(768)) + (r244_lane[:, None] * $(12)) + r244_col[None, :]) % $(12)) * $(64)) + (((((r244_group % $(8)) * $(768)) + (r244_lane[:, None] * $(12)) + r244_col[None, :]) / $(12)) % $(64))), dtype=tl.float32)
      r244_value = tl.dot(r244_left, r244_right)
      r244_value = r244_value.to(tl.float32)
      tl.store($(candidate_vbmm_19_ptr) + ((r244_group * $(48)) + (r244_row[:, None] * $(12)) + r244_col[None, :]), r244_value)
    }
    r245_z = tl.arange($(16384))
    r245_value = tl.load($(candidate_vbmm_17_ptr) + ((((r245_z / $(2048)) % $(8)) * $(16384)) + (((r245_z / $(512)) % $(4)) * $(512)) + (r245_z % $(512)) + $(6144)), dtype=tl.bfloat16)
    r245_value = r245_value.to(tl.float32)
    r245_value = tl.broadcast_to(r245_value, [$(16384)])
    tl.store($(candidate_vto_copy_81_ptr) + r245_z, r245_value)
    r246_z = tl.arange($(6144))
    r246_value = tl.load($(candidate_vindex_2_ptr) + ((((r246_z / $(512)) % $(12)) * $(576)) + (r246_z % $(512)) + $(27648)), dtype=tl.bfloat16)
    r246_value = r246_value.to(tl.float32)
    r246_value = tl.broadcast_to(r246_value, [$(6144)])
    tl.store($(candidate_vto_copy_82_ptr) + r246_z, r246_value)
    r247_row = tl.arange($(32))
    r247_col = tl.arange($(12))
    r247_lane = tl.arange($(512))
    r247_left = tl.load($(candidate_vto_copy_81_ptr) + ((((((r247_row[:, None] * $(512)) + r247_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + (((((r247_row[:, None] * $(512)) + r247_lane[None, :]) / $(512)) % $(4)) * $(512)) + (((r247_row[:, None] * $(512)) + r247_lane[None, :]) % $(512))), dtype=tl.float32)
    r247_right = tl.load($(candidate_vto_copy_82_ptr) + (((((r247_lane[:, None] * $(12)) + r247_col[None, :]) % $(12)) * $(512)) + ((((r247_lane[:, None] * $(12)) + r247_col[None, :]) / $(12)) % $(512))), dtype=tl.float32)
    r247_value = tl.dot(r247_left, r247_right)
    r247_value = r247_value.to(tl.float32)
    tl.store($(candidate_vmm_23_ptr) + ((r247_row[:, None] * $(12)) + r247_col[None, :]), r247_value)
    r248_z = tl.arange($(384))
    r248_value = (tl.load($(candidate_vbmm_19_ptr) + ((((r248_z / $(48)) % $(8)) * $(48)) + (((r248_z / $(12)) % $(4)) * $(12)) + (r248_z % $(12))), dtype=tl.float32) + tl.load($(candidate_vmm_23_ptr) + ((((r248_z / $(48)) % $(8)) * $(48)) + (((r248_z / $(12)) % $(4)) * $(12)) + (r248_z % $(12))), dtype=tl.float32))
    r248_value = r248_value.to(tl.float32)
    r248_value = tl.broadcast_to(r248_value, [$(384)])
    tl.store($(candidate_vadd_28_ptr) + r248_z, r248_value)
    r249_z = tl.arange($(384))
    r249_value = (tl.load($(candidate_vadd_28_ptr) + ((((r249_z / $(48)) % $(8)) * $(48)) + (((r249_z / $(12)) % $(4)) * $(12)) + (r249_z % $(12))), dtype=tl.float32) * 0.07216878364870322)
    r249_value = r249_value.to(tl.float32)
    r249_value = tl.broadcast_to(r249_value, [$(384)])
    tl.store($(candidate_vmul_33_ptr) + r249_z, r249_value)
    r250_z = tl.arange($(4))
    r250_value = (tl.cast(r250_z, tl.int64) + tl.full([], $(12), dtype=tl.int64))
    r250_value = tl.broadcast_to(r250_value, [$(4)])
    tl.store($(candidate_vadd_29_ptr) + r250_z, r250_value)
    r251_z = tl.arange($(4))
    r251_value = (tl.load($(candidate_vadd_29_ptr) + r251_z, dtype=tl.int64) - tl.full([], $(4), dtype=tl.int64))
    r251_value = tl.broadcast_to(r251_value, [$(4)])
    tl.store($(candidate_vsub_19_ptr) + r251_z, r251_value)
    r252_z = tl.arange($(48))
    r252_value = (tl.cast((r252_z % $(12)), tl.int64) > tl.load($(candidate_vsub_19_ptr) + ((r252_z / $(12)) % $(4)), dtype=tl.int64))
    r252_value = tl.broadcast_to(r252_value, [$(48)])
    tl.store($(candidate_vgt_10_ptr) + r252_z, r252_value)
    r253_z = tl.arange($(384))
    r253_value = tl.where(tl.load($(candidate_vgt_10_ptr) + ((((r253_z / $(12)) % $(4)) * $(12)) + (r253_z % $(12))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_33_ptr) + ((((r253_z / $(48)) % $(8)) * $(48)) + (((r253_z / $(12)) % $(4)) * $(12)) + (r253_z % $(12))), dtype=tl.float32))
    r253_value = r253_value.to(tl.float32)
    r253_value = tl.broadcast_to(r253_value, [$(384)])
    tl.store($(candidate_vmasked_fill_10_ptr) + r253_z, r253_value)
    r254_z = tl.arange($(384))
    r254_lane = tl.arange($(12))
    r254_row_values = tl.load($(candidate_vmasked_fill_10_ptr) + ((((r254_z[:, None] / $(48)) % $(8)) * $(48)) + (((r254_z[:, None] / $(12)) % $(4)) * $(12)) + (r254_lane[None, :] % $(12))), dtype=tl.float32)
    r254_maximum = tl.max(r254_row_values, axis=1, keep_dims=true)
    r254_weights = tl.exp(r254_row_values - r254_maximum)
    r254_total = tl.sum(r254_weights, axis=1)
    r254_maximum = tl.max(r254_row_values, axis=1)
    r254_value = tl.exp(tl.load($(candidate_vmasked_fill_10_ptr) + ((((r254_z / $(48)) % $(8)) * $(48)) + (((r254_z / $(12)) % $(4)) * $(12)) + (r254_z % $(12))), dtype=tl.float32) - r254_maximum) / r254_total
    r254_value = r254_value.to(tl.float32)
    r254_value = tl.broadcast_to(r254_value, [$(384)])
    tl.store($(candidate_vsoftmax_10_ptr) + r254_z, r254_value)
    r255_z = tl.arange($(384))
    r255_value = tl.load($(candidate_vsoftmax_10_ptr) + ((((r255_z / $(48)) % $(8)) * $(48)) + (((r255_z / $(12)) % $(4)) * $(12)) + (r255_z % $(12))), dtype=tl.float32)
    r255_value = r255_value.to(tl.bfloat16)
    r255_value = tl.broadcast_to(r255_value, [$(384)])
    tl.store($(candidate_vto_copy_83_ptr) + r255_z, r255_value)
    r256_row = tl.arange($(32))
    r256_col = tl.arange($(512))
    r256_lane = tl.arange($(12))
    r256_left = tl.load($(candidate_vto_copy_83_ptr) + ((((((r256_row[:, None] * $(12)) + r256_lane[None, :]) / $(48)) % $(8)) * $(48)) + (((((r256_row[:, None] * $(12)) + r256_lane[None, :]) / $(12)) % $(4)) * $(12)) + (((r256_row[:, None] * $(12)) + r256_lane[None, :]) % $(12))), dtype=tl.bfloat16)
    r256_right = tl.load($(candidate_vindex_2_ptr) + ((((((r256_lane[:, None] * $(512)) + r256_col[None, :]) / $(512)) % $(12)) * $(576)) + (((r256_lane[:, None] * $(512)) + r256_col[None, :]) % $(512)) + $(27648)), dtype=tl.bfloat16)
    r256_value = tl.dot(r256_left, r256_right)
    r256_value = r256_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_24_ptr) + ((r256_row[:, None] * $(512)) + r256_col[None, :]), r256_value)
    r257_z = tl.arange($(32))
    r257_lane = tl.arange($(12))
    r257_row_values = tl.load($(candidate_vmasked_fill_10_ptr) + ((((r257_z[:, None] / $(4)) % $(8)) * $(48)) + ((r257_z[:, None] % $(4)) * $(12)) + (r257_lane[None, :] % $(12))), dtype=tl.float32)
    r257_maximum = tl.max(r257_row_values, axis=1, keep_dims=true)
    r257_weights = tl.exp(r257_row_values - r257_maximum)
    r257_total = tl.sum(r257_weights, axis=1)
    r257_maximum = tl.max(r257_row_values, axis=1)
    r257_value = r257_maximum + tl.log(r257_total)
    r257_value = r257_value.to(tl.float32)
    r257_value = tl.broadcast_to(r257_value, [$(32)])
    tl.store($(candidate_vlogsumexp_10_ptr) + r257_z, r257_value)
    r258_z = tl.arange($(8192))
    r258_value = tl.load($(candidate_vadd_6_ptr) + ((((r258_z / $(64)) % $(16)) * $(512)) + (((r258_z / $(1024)) % $(8)) * $(64)) + (r258_z % $(64)) + $(57344)), dtype=tl.bfloat16)
    r258_value = r258_value.to(tl.float32)
    r258_value = tl.broadcast_to(r258_value, [$(8192)])
    tl.store($(candidate_vto_copy_84_ptr) + r258_z, r258_value)
    r259_z = tl.arange($(3072))
    r259_value = tl.load($(candidate_vindex_2_ptr) + ((((r259_z / $(64)) % $(48)) * $(576)) + (r259_z % $(64)) + $(37376)), dtype=tl.bfloat16)
    r259_value = r259_value.to(tl.float32)
    r259_value = tl.broadcast_to(r259_value, [$(3072)])
    tl.store($(candidate_vto_copy_85_ptr) + r259_z, r259_value)
    r260_row = tl.arange($(16))
    r260_col = tl.arange($(48))
    r260_lane = tl.arange($(64))
    for r260_group in range($(8)) {
      r260_left = tl.load($(candidate_vto_copy_84_ptr) + (((((((r260_group % $(8)) * $(1024)) + (r260_row[:, None] * $(64)) + r260_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + ((((((r260_group % $(8)) * $(1024)) + (r260_row[:, None] * $(64)) + r260_lane[None, :]) / $(64)) % $(16)) * $(64)) + ((((r260_group % $(8)) * $(1024)) + (r260_row[:, None] * $(64)) + r260_lane[None, :]) % $(64))), dtype=tl.float32)
      r260_right = tl.load($(candidate_vto_copy_85_ptr) + ((((((r260_group % $(8)) * $(3072)) + (r260_lane[:, None] * $(48)) + r260_col[None, :]) % $(48)) * $(64)) + (((((r260_group % $(8)) * $(3072)) + (r260_lane[:, None] * $(48)) + r260_col[None, :]) / $(48)) % $(64))), dtype=tl.float32)
      r260_value = tl.dot(r260_left, r260_right)
      r260_value = r260_value.to(tl.float32)
      tl.store($(candidate_vbmm_20_ptr) + ((r260_group * $(768)) + (r260_row[:, None] * $(48)) + r260_col[None, :]), r260_value)
    }
    r261_z = tl.arange($(65536))
    r261_value = tl.load($(candidate_vbmm_17_ptr) + ((((r261_z / $(8192)) % $(8)) * $(16384)) + (((r261_z / $(512)) % $(16)) * $(512)) + (r261_z % $(512)) + $(8192)), dtype=tl.bfloat16)
    r261_value = r261_value.to(tl.float32)
    r261_value = tl.broadcast_to(r261_value, [$(65536)])
    tl.store($(candidate_vto_copy_86_ptr) + r261_z, r261_value)
    r262_z = tl.arange($(24576))
    r262_value = tl.load($(candidate_vindex_2_ptr) + ((((r262_z / $(512)) % $(48)) * $(576)) + (r262_z % $(512)) + $(36864)), dtype=tl.bfloat16)
    r262_value = r262_value.to(tl.float32)
    r262_value = tl.broadcast_to(r262_value, [$(24576)])
    tl.store($(candidate_vto_copy_87_ptr) + r262_z, r262_value)
    r263_row = tl.arange($(128))
    r263_col = tl.arange($(48))
    r263_lane = tl.arange($(512))
    r263_left = tl.load($(candidate_vto_copy_86_ptr) + ((((((r263_row[:, None] * $(512)) + r263_lane[None, :]) / $(8192)) % $(8)) * $(8192)) + (((((r263_row[:, None] * $(512)) + r263_lane[None, :]) / $(512)) % $(16)) * $(512)) + (((r263_row[:, None] * $(512)) + r263_lane[None, :]) % $(512))), dtype=tl.float32)
    r263_right = tl.load($(candidate_vto_copy_87_ptr) + (((((r263_lane[:, None] * $(48)) + r263_col[None, :]) % $(48)) * $(512)) + ((((r263_lane[:, None] * $(48)) + r263_col[None, :]) / $(48)) % $(512))), dtype=tl.float32)
    r263_value = tl.dot(r263_left, r263_right)
    r263_value = r263_value.to(tl.float32)
    tl.store($(candidate_vmm_25_ptr) + ((r263_row[:, None] * $(48)) + r263_col[None, :]), r263_value)
    r264_z = tl.arange($(6144))
    r264_value = (tl.load($(candidate_vbmm_20_ptr) + ((((r264_z / $(768)) % $(8)) * $(768)) + (((r264_z / $(48)) % $(16)) * $(48)) + (r264_z % $(48))), dtype=tl.float32) + tl.load($(candidate_vmm_25_ptr) + ((((r264_z / $(768)) % $(8)) * $(768)) + (((r264_z / $(48)) % $(16)) * $(48)) + (r264_z % $(48))), dtype=tl.float32))
    r264_value = r264_value.to(tl.float32)
    r264_value = tl.broadcast_to(r264_value, [$(6144)])
    tl.store($(candidate_vadd_30_ptr) + r264_z, r264_value)
    r265_z = tl.arange($(6144))
    r265_value = (tl.load($(candidate_vadd_30_ptr) + ((((r265_z / $(768)) % $(8)) * $(768)) + (((r265_z / $(48)) % $(16)) * $(48)) + (r265_z % $(48))), dtype=tl.float32) * 0.07216878364870322)
    r265_value = r265_value.to(tl.float32)
    r265_value = tl.broadcast_to(r265_value, [$(6144)])
    tl.store($(candidate_vmul_34_ptr) + r265_z, r265_value)
    r266_z = tl.arange($(16))
    r266_value = (tl.cast(r266_z, tl.int64) + tl.full([], $(48), dtype=tl.int64))
    r266_value = tl.broadcast_to(r266_value, [$(16)])
    tl.store($(candidate_vadd_31_ptr) + r266_z, r266_value)
    r267_z = tl.arange($(16))
    r267_value = (tl.load($(candidate_vadd_31_ptr) + r267_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r267_value = tl.broadcast_to(r267_value, [$(16)])
    tl.store($(candidate_vsub_20_ptr) + r267_z, r267_value)
    r268_z = tl.arange($(768))
    r268_value = (tl.cast((r268_z % $(48)), tl.int64) > tl.load($(candidate_vsub_20_ptr) + ((r268_z / $(48)) % $(16)), dtype=tl.int64))
    r268_value = tl.broadcast_to(r268_value, [$(768)])
    tl.store($(candidate_vgt_11_ptr) + r268_z, r268_value)
    r269_z = tl.arange($(6144))
    r269_value = tl.where(tl.load($(candidate_vgt_11_ptr) + ((((r269_z / $(48)) % $(16)) * $(48)) + (r269_z % $(48))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_34_ptr) + ((((r269_z / $(768)) % $(8)) * $(768)) + (((r269_z / $(48)) % $(16)) * $(48)) + (r269_z % $(48))), dtype=tl.float32))
    r269_value = r269_value.to(tl.float32)
    r269_value = tl.broadcast_to(r269_value, [$(6144)])
    tl.store($(candidate_vmasked_fill_11_ptr) + r269_z, r269_value)
    r270_z = tl.arange($(6144))
    r270_lane = tl.arange($(48))
    r270_row_values = tl.load($(candidate_vmasked_fill_11_ptr) + ((((r270_z[:, None] / $(768)) % $(8)) * $(768)) + (((r270_z[:, None] / $(48)) % $(16)) * $(48)) + (r270_lane[None, :] % $(48))), dtype=tl.float32)
    r270_maximum = tl.max(r270_row_values, axis=1, keep_dims=true)
    r270_weights = tl.exp(r270_row_values - r270_maximum)
    r270_total = tl.sum(r270_weights, axis=1)
    r270_maximum = tl.max(r270_row_values, axis=1)
    r270_value = tl.exp(tl.load($(candidate_vmasked_fill_11_ptr) + ((((r270_z / $(768)) % $(8)) * $(768)) + (((r270_z / $(48)) % $(16)) * $(48)) + (r270_z % $(48))), dtype=tl.float32) - r270_maximum) / r270_total
    r270_value = r270_value.to(tl.float32)
    r270_value = tl.broadcast_to(r270_value, [$(6144)])
    tl.store($(candidate_vsoftmax_11_ptr) + r270_z, r270_value)
    r271_z = tl.arange($(6144))
    r271_value = tl.load($(candidate_vsoftmax_11_ptr) + ((((r271_z / $(768)) % $(8)) * $(768)) + (((r271_z / $(48)) % $(16)) * $(48)) + (r271_z % $(48))), dtype=tl.float32)
    r271_value = r271_value.to(tl.bfloat16)
    r271_value = tl.broadcast_to(r271_value, [$(6144)])
    tl.store($(candidate_vto_copy_88_ptr) + r271_z, r271_value)
    r272_row = tl.arange($(128))
    r272_col = tl.arange($(512))
    r272_lane = tl.arange($(48))
    r272_left = tl.load($(candidate_vto_copy_88_ptr) + ((((((r272_row[:, None] * $(48)) + r272_lane[None, :]) / $(768)) % $(8)) * $(768)) + (((((r272_row[:, None] * $(48)) + r272_lane[None, :]) / $(48)) % $(16)) * $(48)) + (((r272_row[:, None] * $(48)) + r272_lane[None, :]) % $(48))), dtype=tl.bfloat16)
    r272_right = tl.load($(candidate_vindex_2_ptr) + ((((((r272_lane[:, None] * $(512)) + r272_col[None, :]) / $(512)) % $(48)) * $(576)) + (((r272_lane[:, None] * $(512)) + r272_col[None, :]) % $(512)) + $(36864)), dtype=tl.bfloat16)
    r272_value = tl.dot(r272_left, r272_right)
    r272_value = r272_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_26_ptr) + ((r272_row[:, None] * $(512)) + r272_col[None, :]), r272_value)
    r273_z = tl.arange($(128))
    r273_lane = tl.arange($(48))
    r273_row_values = tl.load($(candidate_vmasked_fill_11_ptr) + ((((r273_z[:, None] / $(16)) % $(8)) * $(768)) + ((r273_z[:, None] % $(16)) * $(48)) + (r273_lane[None, :] % $(48))), dtype=tl.float32)
    r273_maximum = tl.max(r273_row_values, axis=1, keep_dims=true)
    r273_weights = tl.exp(r273_row_values - r273_maximum)
    r273_total = tl.sum(r273_weights, axis=1)
    r273_maximum = tl.max(r273_row_values, axis=1)
    r273_value = r273_maximum + tl.log(r273_total)
    r273_value = r273_value.to(tl.float32)
    r273_value = tl.broadcast_to(r273_value, [$(128)])
    tl.store($(candidate_vlogsumexp_11_ptr) + r273_z, r273_value)
    r274_z = tl.arange($(131072))
    r274_value = tl.where(((r274_z / $(4096)) % $(32)) < $(12), tl.load($(candidate_vmm_22_ptr) + (((((((r274_z / $(512)) % $(8)) * $(12)) + (((r274_z / $(4096)) % $(32)) - $(0))) % $(96)) * $(512)) + (r274_z % $(512))), dtype=tl.bfloat16), tl.where(((r274_z / $(4096)) % $(32)) < $(16), tl.load($(candidate_vmm_24_ptr) + (((((((r274_z / $(512)) % $(8)) * $(4)) + (((r274_z / $(4096)) % $(32)) - $(12))) % $(32)) * $(512)) + (r274_z % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vmm_26_ptr) + (((((((r274_z / $(512)) % $(8)) * $(16)) + (((r274_z / $(4096)) % $(32)) - $(16))) % $(128)) * $(512)) + (r274_z % $(512))), dtype=tl.bfloat16)))
    r274_value = r274_value.to(tl.bfloat16)
    r274_value = tl.broadcast_to(r274_value, [$(131072)])
    tl.store($(candidate_vcat_12_ptr) + r274_z, r274_value)
    r275_z = tl.arange($(256))
    r275_value = tl.where(((r275_z / $(8)) % $(32)) < $(12), tl.load($(candidate_vlogsumexp_9_ptr) + (((r275_z % $(8)) * $(12)) + ((((r275_z / $(8)) % $(32)) - $(0)) % $(12))), dtype=tl.float32), tl.where(((r275_z / $(8)) % $(32)) < $(16), tl.load($(candidate_vlogsumexp_10_ptr) + (((r275_z % $(8)) * $(4)) + ((((r275_z / $(8)) % $(32)) - $(12)) % $(4))), dtype=tl.float32), tl.load($(candidate_vlogsumexp_11_ptr) + (((r275_z % $(8)) * $(16)) + ((((r275_z / $(8)) % $(32)) - $(16)) % $(16))), dtype=tl.float32)))
    r275_value = r275_value.to(tl.float32)
    r275_value = tl.broadcast_to(r275_value, [$(256)])
    tl.store($(candidate_vcat_13_ptr) + r275_z, r275_value)
    r276_row = tl.arange($(32))
    r276_col = tl.arange($(128))
    r276_lane = tl.arange($(512))
    for r276_group in range($(8)) {
      r276_left = tl.load($(candidate_vcat_12_ptr) + (((((((r276_group % $(8)) * $(16384)) + (r276_row[:, None] * $(512)) + r276_lane[None, :]) / $(512)) % $(32)) * $(4096)) + ((((((r276_group % $(8)) * $(16384)) + (r276_row[:, None] * $(512)) + r276_lane[None, :]) / $(16384)) % $(8)) * $(512)) + ((((r276_group % $(8)) * $(16384)) + (r276_row[:, None] * $(512)) + r276_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r276_right = tl.load($(candidate_vto_copy_5_ptr) + (((((((r276_group % $(8)) * $(65536)) + (r276_lane[:, None] * $(128)) + r276_col[None, :]) / $(65536)) % $(8)) * $(131072)) + (((((r276_group % $(8)) * $(65536)) + (r276_lane[:, None] * $(128)) + r276_col[None, :]) % $(128)) * $(512)) + (((((r276_group % $(8)) * $(65536)) + (r276_lane[:, None] * $(128)) + r276_col[None, :]) / $(128)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r276_value = tl.dot(r276_left, r276_right)
      r276_value = r276_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_21_ptr) + ((r276_group * $(4096)) + (r276_row[:, None] * $(128)) + r276_col[None, :]), r276_value)
    }
    r277_z = tl.arange($(65536))
    r277_value = tl.cast(0.0, tl.bfloat16)
    for r277_lane in range($(128)) {
      r277_value = tl.where(tl.load($(candidate_vsort_index_ptr) + (r277_lane % $(128)), dtype=tl.int64) == ((r277_z / $(512)) % $(128)), r277_value + tl.load($(input1_ptr) + (((r277_lane % $(128)) * $(512)) + (r277_z % $(512))), dtype=tl.bfloat16), r277_value)
    }
    r277_value = r277_value.to(tl.bfloat16)
    r277_value = tl.broadcast_to(r277_value, [$(65536)])
    tl.store($(candidate_vindex_put_ptr) + r277_z, r277_value)
    r278_row = tl.arange($(128))
    r278_col = tl.arange($(1024))
    r278_lane = tl.arange($(512))
    r278_left = tl.load($(candidate_vindex_put_ptr) + ((((((r278_row[:, None] * $(512)) + r278_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r278_row[:, None] * $(512)) + r278_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r278_right = tl.load($(candidate_vto_copy_6_ptr) + (((r278_lane[:, None] % $(512)) * $(1024)) + (r278_col[None, :] % $(1024))), dtype=tl.bfloat16)
    r278_value = tl.dot(r278_left, r278_right)
    r278_value = r278_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_28_ptr) + ((r278_row[:, None] * $(1024)) + r278_col[None, :]), r278_value)
    r279_row = tl.arange($(128))
    r279_col = tl.arange($(1024))
    r279_lane = tl.arange($(512))
    for r279_group in range($(1)) {
      r279_left = tl.load($(candidate_vindex_2_ptr) + ((((((r279_row[:, None] * $(512)) + r279_lane[None, :]) / $(512)) % $(128)) * $(576)) + (((r279_row[:, None] * $(512)) + r279_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r279_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r279_lane[:, None] * $(1024)) + r279_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r279_lane[:, None] * $(1024)) + r279_col[None, :]) % $(128)) * $(512)) + ((((r279_lane[:, None] * $(1024)) + r279_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r279_value = tl.dot(r279_left, r279_right)
      r279_value = r279_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_22_ptr) + ((r279_group * $(131072)) + (r279_row[:, None] * $(1024)) + r279_col[None, :]), r279_value)
    }
    r280_row = tl.arange($(128))
    r280_col = tl.arange($(1024))
    r280_lane = tl.arange($(512))
    for r280_group in range($(1)) {
      r280_left = tl.load($(candidate_vindex_2_ptr) + ((((((r280_row[:, None] * $(512)) + r280_lane[None, :]) / $(512)) % $(128)) * $(576)) + (((r280_row[:, None] * $(512)) + r280_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r280_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r280_lane[:, None] * $(1024)) + r280_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r280_lane[:, None] * $(1024)) + r280_col[None, :]) % $(128)) * $(512)) + ((((r280_lane[:, None] * $(1024)) + r280_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r280_value = tl.dot(r280_left, r280_right)
      r280_value = r280_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_23_ptr) + ((r280_group * $(131072)) + (r280_row[:, None] * $(1024)) + r280_col[None, :]), r280_value)
    }
    r281_z = tl.arange($(49152))
    r281_value = tl.where((r281_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r281_z / $(1536)) % $(32)) * $(1024)) + (((r281_z / $(192)) % $(8)) * $(128)) + ((r281_z % $(192)) - $(0)) + $(98304)) / $(1024)) % $(128)) * $(1536)) + (((((((r281_z / $(1536)) % $(32)) * $(1024)) + (((r281_z / $(192)) % $(8)) * $(128)) + ((r281_z % $(192)) - $(0)) + $(98304)) / $(128)) % $(8)) * $(192)) + (((((r281_z / $(1536)) % $(32)) * $(1024)) + (((r281_z / $(192)) % $(8)) * $(128)) + ((r281_z % $(192)) - $(0)) + $(98304)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vadd_6_ptr) + ((((((((r281_z / $(1536)) % $(32)) * $(512)) + (((r281_z / $(192)) % $(8)) * $(64)) + ((r281_z % $(192)) - $(128)) + $(49152)) / $(512)) % $(128)) * $(512)) + (((((((r281_z / $(1536)) % $(32)) * $(512)) + (((r281_z / $(192)) % $(8)) * $(64)) + ((r281_z % $(192)) - $(128)) + $(49152)) / $(64)) % $(8)) * $(64)) + (((((r281_z / $(1536)) % $(32)) * $(512)) + (((r281_z / $(192)) % $(8)) * $(64)) + ((r281_z % $(192)) - $(128)) + $(49152)) % $(64))), dtype=tl.bfloat16))
    r281_value = r281_value.to(tl.bfloat16)
    r281_value = tl.broadcast_to(r281_value, [$(49152)])
    tl.store($(candidate_vcat_15_ptr) + r281_z, r281_value)
    r282_z = tl.arange($(196608))
    r282_value = tl.where((r282_z % $(192)) < $(128), tl.load($(candidate_vbmm_22_ptr) + ((((((((r282_z / $(1536)) % $(128)) * $(1024)) + (((r282_z / $(192)) % $(8)) * $(128)) + ((r282_z % $(192)) - $(0))) / $(1024)) % $(128)) * $(1024)) + (((((((r282_z / $(1536)) % $(128)) * $(1024)) + (((r282_z / $(192)) % $(8)) * $(128)) + ((r282_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r282_z / $(1536)) % $(128)) * $(1024)) + (((r282_z / $(192)) % $(8)) * $(128)) + ((r282_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vindex_2_ptr) + ((((r282_z / $(1536)) % $(128)) * $(576)) + ((((r282_z % $(192)) - $(128)) + $(512)) % $(576))), dtype=tl.bfloat16))
    r282_value = r282_value.to(tl.bfloat16)
    r282_value = tl.broadcast_to(r282_value, [$(196608)])
    tl.store($(candidate_vcat_16_ptr) + r282_z, r282_value)
    r283_z = tl.arange($(18432))
    r283_value = tl.load($(candidate_vcat_15_ptr) + ((((r283_z / $(192)) % $(12)) * $(1536)) + (((r283_z / $(2304)) % $(8)) * $(192)) + (r283_z % $(192))), dtype=tl.bfloat16)
    r283_value = r283_value.to(tl.float32)
    r283_value = tl.broadcast_to(r283_value, [$(18432)])
    tl.store($(candidate_vto_copy_89_ptr) + r283_z, r283_value)
    r284_z = tl.arange($(55296))
    r284_value = tl.load($(candidate_vcat_16_ptr) + ((((r284_z / $(192)) % $(36)) * $(1536)) + (((r284_z / $(6912)) % $(8)) * $(192)) + (r284_z % $(192))), dtype=tl.bfloat16)
    r284_value = r284_value.to(tl.float32)
    r284_value = tl.broadcast_to(r284_value, [$(55296)])
    tl.store($(candidate_vto_copy_90_ptr) + r284_z, r284_value)
    r285_z = tl.arange($(36864))
    r285_value = tl.load($(candidate_vbmm_23_ptr) + ((((r285_z / $(128)) % $(36)) * $(1024)) + (((r285_z / $(4608)) % $(8)) * $(128)) + (r285_z % $(128))), dtype=tl.bfloat16)
    r285_value = r285_value.to(tl.float32)
    r285_value = tl.broadcast_to(r285_value, [$(36864)])
    tl.store($(candidate_vto_copy_91_ptr) + r285_z, r285_value)
    r286_z = tl.arange($(12288))
    r286_value = tl.load($(candidate_vmm_28_ptr) + ((((r286_z / $(128)) % $(12)) * $(1024)) + (((r286_z / $(1536)) % $(8)) * $(128)) + (r286_z % $(128)) + $(98304)), dtype=tl.bfloat16)
    r286_value = r286_value.to(tl.float32)
    r286_value = tl.broadcast_to(r286_value, [$(12288)])
    tl.store($(candidate_vto_copy_92_ptr) + r286_z, r286_value)
    r287_z = tl.arange($(12288))
    r287_value = tl.load($(candidate_vbmm_21_ptr) + ((((r287_z / $(1536)) % $(8)) * $(4096)) + (((r287_z / $(128)) % $(12)) * $(128)) + (r287_z % $(128))), dtype=tl.bfloat16)
    r287_value = r287_value.to(tl.float32)
    r287_value = tl.broadcast_to(r287_value, [$(12288)])
    tl.store($(candidate_vto_copy_93_ptr) + r287_z, r287_value)
    r288_row = tl.arange($(12))
    r288_col = tl.arange($(36))
    r288_lane = tl.arange($(192))
    for r288_group in range($(8)) {
      r288_left = tl.load($(candidate_vto_copy_89_ptr) + (((((((r288_group % $(8)) * $(2304)) + (r288_row[:, None] * $(192)) + r288_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r288_group % $(8)) * $(2304)) + (r288_row[:, None] * $(192)) + r288_lane[None, :]) / $(192)) % $(12)) * $(192)) + ((((r288_group % $(8)) * $(2304)) + (r288_row[:, None] * $(192)) + r288_lane[None, :]) % $(192))), dtype=tl.float32)
      r288_right = tl.load($(candidate_vto_copy_90_ptr) + (((((((r288_group % $(8)) * $(6912)) + (r288_lane[:, None] * $(36)) + r288_col[None, :]) / $(6912)) % $(8)) * $(6912)) + (((((r288_group % $(8)) * $(6912)) + (r288_lane[:, None] * $(36)) + r288_col[None, :]) % $(36)) * $(192)) + (((((r288_group % $(8)) * $(6912)) + (r288_lane[:, None] * $(36)) + r288_col[None, :]) / $(36)) % $(192))), dtype=tl.float32)
      r288_value = tl.dot(r288_left, r288_right)
      r288_value = r288_value.to(tl.float32)
      tl.store($(candidate_vbmm_24_ptr) + ((r288_group * $(432)) + (r288_row[:, None] * $(36)) + r288_col[None, :]), r288_value)
    }
    r289_z = tl.arange($(3456))
    r289_value = (tl.load($(candidate_vbmm_24_ptr) + ((((r289_z / $(432)) % $(8)) * $(432)) + (((r289_z / $(36)) % $(12)) * $(36)) + (r289_z % $(36))), dtype=tl.float32) * 0.07216878364870322)
    r289_value = r289_value.to(tl.float32)
    r289_value = tl.broadcast_to(r289_value, [$(3456)])
    tl.store($(candidate_vmul_35_ptr) + r289_z, r289_value)
    r290_z = tl.arange($(12))
    r290_value = (tl.cast(r290_z, tl.int64) + tl.full([], $(36), dtype=tl.int64))
    r290_value = tl.broadcast_to(r290_value, [$(12)])
    tl.store($(candidate_vadd_32_ptr) + r290_z, r290_value)
    r291_z = tl.arange($(12))
    r291_value = (tl.load($(candidate_vadd_32_ptr) + r291_z, dtype=tl.int64) - tl.full([], $(12), dtype=tl.int64))
    r291_value = tl.broadcast_to(r291_value, [$(12)])
    tl.store($(candidate_vsub_21_ptr) + r291_z, r291_value)
    r292_z = tl.arange($(432))
    r292_value = (tl.cast((r292_z % $(36)), tl.int64) > tl.load($(candidate_vsub_21_ptr) + ((r292_z / $(36)) % $(12)), dtype=tl.int64))
    r292_value = tl.broadcast_to(r292_value, [$(432)])
    tl.store($(candidate_vgt_12_ptr) + r292_z, r292_value)
    r293_z = tl.arange($(3456))
    r293_value = tl.where(tl.load($(candidate_vgt_12_ptr) + ((((r293_z / $(36)) % $(12)) * $(36)) + (r293_z % $(36))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_35_ptr) + ((((r293_z / $(432)) % $(8)) * $(432)) + (((r293_z / $(36)) % $(12)) * $(36)) + (r293_z % $(36))), dtype=tl.float32))
    r293_value = r293_value.to(tl.float32)
    r293_value = tl.broadcast_to(r293_value, [$(3456)])
    tl.store($(candidate_vmasked_fill_12_ptr) + r293_z, r293_value)
    r294_z = tl.arange($(3456))
    r294_value = (tl.load($(candidate_vmasked_fill_12_ptr) + ((((r294_z / $(432)) % $(8)) * $(432)) + (((r294_z / $(36)) % $(12)) * $(36)) + (r294_z % $(36))), dtype=tl.float32) - tl.load($(candidate_vcat_13_ptr) + ((((r294_z / $(36)) % $(12)) * $(8)) + ((r294_z / $(432)) % $(8))), dtype=tl.float32))
    r294_value = r294_value.to(tl.float32)
    r294_value = tl.broadcast_to(r294_value, [$(3456)])
    tl.store($(candidate_vsub_22_ptr) + r294_z, r294_value)
    r295_z = tl.arange($(3456))
    r295_value = tl.exp(tl.load($(candidate_vsub_22_ptr) + ((((r295_z / $(432)) % $(8)) * $(432)) + (((r295_z / $(36)) % $(12)) * $(36)) + (r295_z % $(36))), dtype=tl.float32))
    r295_value = r295_value.to(tl.float32)
    r295_value = tl.broadcast_to(r295_value, [$(3456)])
    tl.store($(candidate_vexp_ptr) + r295_z, r295_value)
    r296_z = tl.arange($(12288))
    r296_value = (tl.load($(candidate_vto_copy_92_ptr) + ((((r296_z / $(1536)) % $(8)) * $(1536)) + (((r296_z / $(128)) % $(12)) * $(128)) + (r296_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_93_ptr) + ((((r296_z / $(1536)) % $(8)) * $(1536)) + (((r296_z / $(128)) % $(12)) * $(128)) + (r296_z % $(128))), dtype=tl.float32))
    r296_value = r296_value.to(tl.float32)
    r296_value = tl.broadcast_to(r296_value, [$(12288)])
    tl.store($(candidate_vmul_36_ptr) + r296_z, r296_value)
    r297_z = tl.arange($(96))
    r297_lane = tl.arange($(128))
    r297_value = tl.sum(tl.load($(candidate_vmul_36_ptr) + ((((r297_z[:, None] / $(12)) % $(8)) * $(1536)) + ((r297_z[:, None] % $(12)) * $(128)) + r297_lane[None, :]), dtype=tl.float32), axis=1)
    r297_value = r297_value.to(tl.float32)
    r297_value = tl.broadcast_to(r297_value, [$(96)])
    tl.store($(candidate_vsum_1_ptr) + r297_z, r297_value)
    r298_row = tl.arange($(12))
    r298_col = tl.arange($(36))
    r298_lane = tl.arange($(128))
    for r298_group in range($(8)) {
      r298_left = tl.load($(candidate_vto_copy_92_ptr) + (((((((r298_group % $(8)) * $(1536)) + (r298_row[:, None] * $(128)) + r298_lane[None, :]) / $(1536)) % $(8)) * $(1536)) + ((((((r298_group % $(8)) * $(1536)) + (r298_row[:, None] * $(128)) + r298_lane[None, :]) / $(128)) % $(12)) * $(128)) + ((((r298_group % $(8)) * $(1536)) + (r298_row[:, None] * $(128)) + r298_lane[None, :]) % $(128))), dtype=tl.float32)
      r298_right = tl.load($(candidate_vto_copy_91_ptr) + (((((((r298_group % $(8)) * $(4608)) + (r298_lane[:, None] * $(36)) + r298_col[None, :]) / $(4608)) % $(8)) * $(4608)) + (((((r298_group % $(8)) * $(4608)) + (r298_lane[:, None] * $(36)) + r298_col[None, :]) % $(36)) * $(128)) + (((((r298_group % $(8)) * $(4608)) + (r298_lane[:, None] * $(36)) + r298_col[None, :]) / $(36)) % $(128))), dtype=tl.float32)
      r298_value = tl.dot(r298_left, r298_right)
      r298_value = r298_value.to(tl.float32)
      tl.store($(candidate_vbmm_25_ptr) + ((r298_group * $(432)) + (r298_row[:, None] * $(36)) + r298_col[None, :]), r298_value)
    }
    r299_z = tl.arange($(3456))
    r299_value = (tl.load($(candidate_vbmm_25_ptr) + ((((r299_z / $(432)) % $(8)) * $(432)) + (((r299_z / $(36)) % $(12)) * $(36)) + (r299_z % $(36))), dtype=tl.float32) - tl.load($(candidate_vsum_1_ptr) + ((((r299_z / $(432)) % $(8)) * $(12)) + ((r299_z / $(36)) % $(12))), dtype=tl.float32))
    r299_value = r299_value.to(tl.float32)
    r299_value = tl.broadcast_to(r299_value, [$(3456)])
    tl.store($(candidate_vsub_23_ptr) + r299_z, r299_value)
    r300_z = tl.arange($(3456))
    r300_value = (tl.load($(candidate_vexp_ptr) + ((((r300_z / $(432)) % $(8)) * $(432)) + (((r300_z / $(36)) % $(12)) * $(36)) + (r300_z % $(36))), dtype=tl.float32) * tl.load($(candidate_vsub_23_ptr) + ((((r300_z / $(432)) % $(8)) * $(432)) + (((r300_z / $(36)) % $(12)) * $(36)) + (r300_z % $(36))), dtype=tl.float32))
    r300_value = r300_value.to(tl.float32)
    r300_value = tl.broadcast_to(r300_value, [$(3456)])
    tl.store($(candidate_vmul_37_ptr) + r300_z, r300_value)
    r301_row = tl.arange($(12))
    r301_col = tl.arange($(192))
    r301_lane = tl.arange($(36))
    for r301_group in range($(8)) {
      r301_left = tl.load($(candidate_vmul_37_ptr) + (((((((r301_group % $(8)) * $(432)) + (r301_row[:, None] * $(36)) + r301_lane[None, :]) / $(432)) % $(8)) * $(432)) + ((((((r301_group % $(8)) * $(432)) + (r301_row[:, None] * $(36)) + r301_lane[None, :]) / $(36)) % $(12)) * $(36)) + ((((r301_group % $(8)) * $(432)) + (r301_row[:, None] * $(36)) + r301_lane[None, :]) % $(36))), dtype=tl.float32)
      r301_right = tl.load($(candidate_vto_copy_90_ptr) + (((((((r301_group % $(8)) * $(6912)) + (r301_lane[:, None] * $(192)) + r301_col[None, :]) / $(6912)) % $(8)) * $(6912)) + ((((((r301_group % $(8)) * $(6912)) + (r301_lane[:, None] * $(192)) + r301_col[None, :]) / $(192)) % $(36)) * $(192)) + ((((r301_group % $(8)) * $(6912)) + (r301_lane[:, None] * $(192)) + r301_col[None, :]) % $(192))), dtype=tl.float32)
      r301_value = tl.dot(r301_left, r301_right)
      r301_value = r301_value.to(tl.float32)
      tl.store($(candidate_vbmm_26_ptr) + ((r301_group * $(2304)) + (r301_row[:, None] * $(192)) + r301_col[None, :]), r301_value)
    }
    r302_z = tl.arange($(18432))
    r302_value = (tl.load($(candidate_vbmm_26_ptr) + ((((r302_z / $(2304)) % $(8)) * $(2304)) + (((r302_z / $(192)) % $(12)) * $(192)) + (r302_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r302_value = r302_value.to(tl.float32)
    r302_value = tl.broadcast_to(r302_value, [$(18432)])
    tl.store($(candidate_vmul_38_ptr) + r302_z, r302_value)
    r303_z = tl.arange($(18432))
    r303_value = tl.load($(candidate_vmul_38_ptr) + ((((r303_z / $(192)) % $(8)) * $(2304)) + (((r303_z / $(1536)) % $(12)) * $(192)) + (r303_z % $(192))), dtype=tl.float32)
    r303_value = r303_value.to(tl.bfloat16)
    r303_value = tl.broadcast_to(r303_value, [$(18432)])
    tl.store($(candidate_vto_copy_94_ptr) + r303_z, r303_value)
    r304_row = tl.arange($(36))
    r304_col = tl.arange($(192))
    r304_lane = tl.arange($(12))
    for r304_group in range($(8)) {
      r304_left = tl.load($(candidate_vmul_37_ptr) + (((((((r304_group % $(8)) * $(432)) + (r304_row[:, None] * $(12)) + r304_lane[None, :]) / $(432)) % $(8)) * $(432)) + (((((r304_group % $(8)) * $(432)) + (r304_row[:, None] * $(12)) + r304_lane[None, :]) % $(12)) * $(36)) + (((((r304_group % $(8)) * $(432)) + (r304_row[:, None] * $(12)) + r304_lane[None, :]) / $(12)) % $(36))), dtype=tl.float32)
      r304_right = tl.load($(candidate_vto_copy_89_ptr) + (((((((r304_group % $(8)) * $(2304)) + (r304_lane[:, None] * $(192)) + r304_col[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r304_group % $(8)) * $(2304)) + (r304_lane[:, None] * $(192)) + r304_col[None, :]) / $(192)) % $(12)) * $(192)) + ((((r304_group % $(8)) * $(2304)) + (r304_lane[:, None] * $(192)) + r304_col[None, :]) % $(192))), dtype=tl.float32)
      r304_value = tl.dot(r304_left, r304_right)
      r304_value = r304_value.to(tl.float32)
      tl.store($(candidate_vbmm_27_ptr) + ((r304_group * $(6912)) + (r304_row[:, None] * $(192)) + r304_col[None, :]), r304_value)
    }
    r305_z = tl.arange($(55296))
    r305_value = (tl.load($(candidate_vbmm_27_ptr) + ((((r305_z / $(6912)) % $(8)) * $(6912)) + (((r305_z / $(192)) % $(36)) * $(192)) + (r305_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r305_value = r305_value.to(tl.float32)
    r305_value = tl.broadcast_to(r305_value, [$(55296)])
    tl.store($(candidate_vmul_39_ptr) + r305_z, r305_value)
    r306_z = tl.arange($(55296))
    r306_value = tl.load($(candidate_vmul_39_ptr) + ((((r306_z / $(192)) % $(8)) * $(6912)) + (((r306_z / $(1536)) % $(36)) * $(192)) + (r306_z % $(192))), dtype=tl.float32)
    r306_value = r306_value.to(tl.bfloat16)
    r306_value = tl.broadcast_to(r306_value, [$(55296)])
    tl.store($(candidate_vto_copy_95_ptr) + r306_z, r306_value)
    r307_row = tl.arange($(36))
    r307_col = tl.arange($(128))
    r307_lane = tl.arange($(12))
    for r307_group in range($(8)) {
      r307_left = tl.load($(candidate_vexp_ptr) + (((((((r307_group % $(8)) * $(432)) + (r307_row[:, None] * $(12)) + r307_lane[None, :]) / $(432)) % $(8)) * $(432)) + (((((r307_group % $(8)) * $(432)) + (r307_row[:, None] * $(12)) + r307_lane[None, :]) % $(12)) * $(36)) + (((((r307_group % $(8)) * $(432)) + (r307_row[:, None] * $(12)) + r307_lane[None, :]) / $(12)) % $(36))), dtype=tl.float32)
      r307_right = tl.load($(candidate_vto_copy_92_ptr) + (((((((r307_group % $(8)) * $(1536)) + (r307_lane[:, None] * $(128)) + r307_col[None, :]) / $(1536)) % $(8)) * $(1536)) + ((((((r307_group % $(8)) * $(1536)) + (r307_lane[:, None] * $(128)) + r307_col[None, :]) / $(128)) % $(12)) * $(128)) + ((((r307_group % $(8)) * $(1536)) + (r307_lane[:, None] * $(128)) + r307_col[None, :]) % $(128))), dtype=tl.float32)
      r307_value = tl.dot(r307_left, r307_right)
      r307_value = r307_value.to(tl.float32)
      tl.store($(candidate_vbmm_28_ptr) + ((r307_group * $(4608)) + (r307_row[:, None] * $(128)) + r307_col[None, :]), r307_value)
    }
    r308_z = tl.arange($(36864))
    r308_value = tl.load($(candidate_vbmm_28_ptr) + ((((r308_z / $(128)) % $(8)) * $(4608)) + (((r308_z / $(1024)) % $(36)) * $(128)) + (r308_z % $(128))), dtype=tl.float32)
    r308_value = r308_value.to(tl.bfloat16)
    r308_value = tl.broadcast_to(r308_value, [$(36864)])
    tl.store($(candidate_vto_copy_96_ptr) + r308_z, r308_value)
    r309_z = tl.arange($(73728))
    r309_value = tl.where(((r309_z / $(1536)) % $(48)) < $(36), tl.load($(candidate_vto_copy_95_ptr) + ((((((r309_z / $(1536)) % $(48)) - $(0)) % $(36)) * $(1536)) + (((r309_z / $(192)) % $(8)) * $(192)) + (r309_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r309_value = r309_value.to(tl.bfloat16)
    r309_value = tl.broadcast_to(r309_value, [$(73728)])
    tl.store($(candidate_vcat_17_ptr) + r309_z, r309_value)
    r310_z = tl.arange($(49152))
    r310_value = tl.where(((r310_z / $(1024)) % $(48)) < $(36), tl.load($(candidate_vto_copy_96_ptr) + ((((((r310_z / $(1024)) % $(48)) - $(0)) % $(36)) * $(1024)) + (((r310_z / $(128)) % $(8)) * $(128)) + (r310_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r310_value = r310_value.to(tl.bfloat16)
    r310_value = tl.broadcast_to(r310_value, [$(49152)])
    tl.store($(candidate_vcat_18_ptr) + r310_z, r310_value)
    r311_z = tl.arange($(6144))
    r311_value = tl.load($(candidate_vcat_15_ptr) + ((((r311_z / $(192)) % $(4)) * $(1536)) + (((r311_z / $(768)) % $(8)) * $(192)) + (r311_z % $(192)) + $(18432)), dtype=tl.bfloat16)
    r311_value = r311_value.to(tl.float32)
    r311_value = tl.broadcast_to(r311_value, [$(6144)])
    tl.store($(candidate_vto_copy_97_ptr) + r311_z, r311_value)
    r312_z = tl.arange($(18432))
    r312_value = tl.load($(candidate_vcat_16_ptr) + ((((r312_z / $(192)) % $(12)) * $(1536)) + (((r312_z / $(2304)) % $(8)) * $(192)) + (r312_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r312_value = r312_value.to(tl.float32)
    r312_value = tl.broadcast_to(r312_value, [$(18432)])
    tl.store($(candidate_vto_copy_98_ptr) + r312_z, r312_value)
    r313_z = tl.arange($(12288))
    r313_value = tl.load($(candidate_vbmm_23_ptr) + ((((r313_z / $(128)) % $(12)) * $(1024)) + (((r313_z / $(1536)) % $(8)) * $(128)) + (r313_z % $(128)) + $(49152)), dtype=tl.bfloat16)
    r313_value = r313_value.to(tl.float32)
    r313_value = tl.broadcast_to(r313_value, [$(12288)])
    tl.store($(candidate_vto_copy_99_ptr) + r313_z, r313_value)
    r314_z = tl.arange($(4096))
    r314_value = tl.load($(candidate_vmm_28_ptr) + ((((r314_z / $(128)) % $(4)) * $(1024)) + (((r314_z / $(512)) % $(8)) * $(128)) + (r314_z % $(128)) + $(110592)), dtype=tl.bfloat16)
    r314_value = r314_value.to(tl.float32)
    r314_value = tl.broadcast_to(r314_value, [$(4096)])
    tl.store($(candidate_vto_copy_100_ptr) + r314_z, r314_value)
    r315_z = tl.arange($(4096))
    r315_value = tl.load($(candidate_vbmm_21_ptr) + ((((r315_z / $(512)) % $(8)) * $(4096)) + (((r315_z / $(128)) % $(4)) * $(128)) + (r315_z % $(128)) + $(1536)), dtype=tl.bfloat16)
    r315_value = r315_value.to(tl.float32)
    r315_value = tl.broadcast_to(r315_value, [$(4096)])
    tl.store($(candidate_vto_copy_101_ptr) + r315_z, r315_value)
    r316_row = tl.arange($(4))
    r316_col = tl.arange($(12))
    r316_lane = tl.arange($(192))
    for r316_group in range($(8)) {
      r316_left = tl.load($(candidate_vto_copy_97_ptr) + (((((((r316_group % $(8)) * $(768)) + (r316_row[:, None] * $(192)) + r316_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r316_group % $(8)) * $(768)) + (r316_row[:, None] * $(192)) + r316_lane[None, :]) / $(192)) % $(4)) * $(192)) + ((((r316_group % $(8)) * $(768)) + (r316_row[:, None] * $(192)) + r316_lane[None, :]) % $(192))), dtype=tl.float32)
      r316_right = tl.load($(candidate_vto_copy_98_ptr) + (((((((r316_group % $(8)) * $(2304)) + (r316_lane[:, None] * $(12)) + r316_col[None, :]) / $(2304)) % $(8)) * $(2304)) + (((((r316_group % $(8)) * $(2304)) + (r316_lane[:, None] * $(12)) + r316_col[None, :]) % $(12)) * $(192)) + (((((r316_group % $(8)) * $(2304)) + (r316_lane[:, None] * $(12)) + r316_col[None, :]) / $(12)) % $(192))), dtype=tl.float32)
      r316_value = tl.dot(r316_left, r316_right)
      r316_value = r316_value.to(tl.float32)
      tl.store($(candidate_vbmm_29_ptr) + ((r316_group * $(48)) + (r316_row[:, None] * $(12)) + r316_col[None, :]), r316_value)
    }
    r317_z = tl.arange($(384))
    r317_value = (tl.load($(candidate_vbmm_29_ptr) + ((((r317_z / $(48)) % $(8)) * $(48)) + (((r317_z / $(12)) % $(4)) * $(12)) + (r317_z % $(12))), dtype=tl.float32) * 0.07216878364870322)
    r317_value = r317_value.to(tl.float32)
    r317_value = tl.broadcast_to(r317_value, [$(384)])
    tl.store($(candidate_vmul_40_ptr) + r317_z, r317_value)
    r318_z = tl.arange($(4))
    r318_value = (tl.cast(r318_z, tl.int64) + tl.full([], $(12), dtype=tl.int64))
    r318_value = tl.broadcast_to(r318_value, [$(4)])
    tl.store($(candidate_vadd_33_ptr) + r318_z, r318_value)
    r319_z = tl.arange($(4))
    r319_value = (tl.load($(candidate_vadd_33_ptr) + r319_z, dtype=tl.int64) - tl.full([], $(4), dtype=tl.int64))
    r319_value = tl.broadcast_to(r319_value, [$(4)])
    tl.store($(candidate_vsub_24_ptr) + r319_z, r319_value)
    r320_z = tl.arange($(48))
    r320_value = (tl.cast((r320_z % $(12)), tl.int64) > tl.load($(candidate_vsub_24_ptr) + ((r320_z / $(12)) % $(4)), dtype=tl.int64))
    r320_value = tl.broadcast_to(r320_value, [$(48)])
    tl.store($(candidate_vgt_13_ptr) + r320_z, r320_value)
    r321_z = tl.arange($(384))
    r321_value = tl.where(tl.load($(candidate_vgt_13_ptr) + ((((r321_z / $(12)) % $(4)) * $(12)) + (r321_z % $(12))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_40_ptr) + ((((r321_z / $(48)) % $(8)) * $(48)) + (((r321_z / $(12)) % $(4)) * $(12)) + (r321_z % $(12))), dtype=tl.float32))
    r321_value = r321_value.to(tl.float32)
    r321_value = tl.broadcast_to(r321_value, [$(384)])
    tl.store($(candidate_vmasked_fill_13_ptr) + r321_z, r321_value)
    r322_z = tl.arange($(384))
    r322_value = (tl.load($(candidate_vmasked_fill_13_ptr) + ((((r322_z / $(48)) % $(8)) * $(48)) + (((r322_z / $(12)) % $(4)) * $(12)) + (r322_z % $(12))), dtype=tl.float32) - tl.load($(candidate_vcat_13_ptr) + ((((r322_z / $(12)) % $(4)) * $(8)) + ((r322_z / $(48)) % $(8)) + $(96)), dtype=tl.float32))
    r322_value = r322_value.to(tl.float32)
    r322_value = tl.broadcast_to(r322_value, [$(384)])
    tl.store($(candidate_vsub_25_ptr) + r322_z, r322_value)
    r323_z = tl.arange($(384))
    r323_value = tl.exp(tl.load($(candidate_vsub_25_ptr) + ((((r323_z / $(48)) % $(8)) * $(48)) + (((r323_z / $(12)) % $(4)) * $(12)) + (r323_z % $(12))), dtype=tl.float32))
    r323_value = r323_value.to(tl.float32)
    r323_value = tl.broadcast_to(r323_value, [$(384)])
    tl.store($(candidate_vexp_1_ptr) + r323_z, r323_value)
    r324_z = tl.arange($(4096))
    r324_value = (tl.load($(candidate_vto_copy_100_ptr) + ((((r324_z / $(512)) % $(8)) * $(512)) + (((r324_z / $(128)) % $(4)) * $(128)) + (r324_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_101_ptr) + ((((r324_z / $(512)) % $(8)) * $(512)) + (((r324_z / $(128)) % $(4)) * $(128)) + (r324_z % $(128))), dtype=tl.float32))
    r324_value = r324_value.to(tl.float32)
    r324_value = tl.broadcast_to(r324_value, [$(4096)])
    tl.store($(candidate_vmul_41_ptr) + r324_z, r324_value)
    r325_z = tl.arange($(32))
    r325_lane = tl.arange($(128))
    r325_value = tl.sum(tl.load($(candidate_vmul_41_ptr) + ((((r325_z[:, None] / $(4)) % $(8)) * $(512)) + ((r325_z[:, None] % $(4)) * $(128)) + r325_lane[None, :]), dtype=tl.float32), axis=1)
    r325_value = r325_value.to(tl.float32)
    r325_value = tl.broadcast_to(r325_value, [$(32)])
    tl.store($(candidate_vsum_2_ptr) + r325_z, r325_value)
    r326_row = tl.arange($(4))
    r326_col = tl.arange($(12))
    r326_lane = tl.arange($(128))
    for r326_group in range($(8)) {
      r326_left = tl.load($(candidate_vto_copy_100_ptr) + (((((((r326_group % $(8)) * $(512)) + (r326_row[:, None] * $(128)) + r326_lane[None, :]) / $(512)) % $(8)) * $(512)) + ((((((r326_group % $(8)) * $(512)) + (r326_row[:, None] * $(128)) + r326_lane[None, :]) / $(128)) % $(4)) * $(128)) + ((((r326_group % $(8)) * $(512)) + (r326_row[:, None] * $(128)) + r326_lane[None, :]) % $(128))), dtype=tl.float32)
      r326_right = tl.load($(candidate_vto_copy_99_ptr) + (((((((r326_group % $(8)) * $(1536)) + (r326_lane[:, None] * $(12)) + r326_col[None, :]) / $(1536)) % $(8)) * $(1536)) + (((((r326_group % $(8)) * $(1536)) + (r326_lane[:, None] * $(12)) + r326_col[None, :]) % $(12)) * $(128)) + (((((r326_group % $(8)) * $(1536)) + (r326_lane[:, None] * $(12)) + r326_col[None, :]) / $(12)) % $(128))), dtype=tl.float32)
      r326_value = tl.dot(r326_left, r326_right)
      r326_value = r326_value.to(tl.float32)
      tl.store($(candidate_vbmm_30_ptr) + ((r326_group * $(48)) + (r326_row[:, None] * $(12)) + r326_col[None, :]), r326_value)
    }
    r327_z = tl.arange($(384))
    r327_value = (tl.load($(candidate_vbmm_30_ptr) + ((((r327_z / $(48)) % $(8)) * $(48)) + (((r327_z / $(12)) % $(4)) * $(12)) + (r327_z % $(12))), dtype=tl.float32) - tl.load($(candidate_vsum_2_ptr) + ((((r327_z / $(48)) % $(8)) * $(4)) + ((r327_z / $(12)) % $(4))), dtype=tl.float32))
    r327_value = r327_value.to(tl.float32)
    r327_value = tl.broadcast_to(r327_value, [$(384)])
    tl.store($(candidate_vsub_26_ptr) + r327_z, r327_value)
    r328_z = tl.arange($(384))
    r328_value = (tl.load($(candidate_vexp_1_ptr) + ((((r328_z / $(48)) % $(8)) * $(48)) + (((r328_z / $(12)) % $(4)) * $(12)) + (r328_z % $(12))), dtype=tl.float32) * tl.load($(candidate_vsub_26_ptr) + ((((r328_z / $(48)) % $(8)) * $(48)) + (((r328_z / $(12)) % $(4)) * $(12)) + (r328_z % $(12))), dtype=tl.float32))
    r328_value = r328_value.to(tl.float32)
    r328_value = tl.broadcast_to(r328_value, [$(384)])
    tl.store($(candidate_vmul_42_ptr) + r328_z, r328_value)
    r329_row = tl.arange($(4))
    r329_col = tl.arange($(192))
    r329_lane = tl.arange($(12))
    for r329_group in range($(8)) {
      r329_left = tl.load($(candidate_vmul_42_ptr) + (((((((r329_group % $(8)) * $(48)) + (r329_row[:, None] * $(12)) + r329_lane[None, :]) / $(48)) % $(8)) * $(48)) + ((((((r329_group % $(8)) * $(48)) + (r329_row[:, None] * $(12)) + r329_lane[None, :]) / $(12)) % $(4)) * $(12)) + ((((r329_group % $(8)) * $(48)) + (r329_row[:, None] * $(12)) + r329_lane[None, :]) % $(12))), dtype=tl.float32)
      r329_right = tl.load($(candidate_vto_copy_98_ptr) + (((((((r329_group % $(8)) * $(2304)) + (r329_lane[:, None] * $(192)) + r329_col[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r329_group % $(8)) * $(2304)) + (r329_lane[:, None] * $(192)) + r329_col[None, :]) / $(192)) % $(12)) * $(192)) + ((((r329_group % $(8)) * $(2304)) + (r329_lane[:, None] * $(192)) + r329_col[None, :]) % $(192))), dtype=tl.float32)
      r329_value = tl.dot(r329_left, r329_right)
      r329_value = r329_value.to(tl.float32)
      tl.store($(candidate_vbmm_31_ptr) + ((r329_group * $(768)) + (r329_row[:, None] * $(192)) + r329_col[None, :]), r329_value)
    }
    r330_z = tl.arange($(6144))
    r330_value = (tl.load($(candidate_vbmm_31_ptr) + ((((r330_z / $(768)) % $(8)) * $(768)) + (((r330_z / $(192)) % $(4)) * $(192)) + (r330_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r330_value = r330_value.to(tl.float32)
    r330_value = tl.broadcast_to(r330_value, [$(6144)])
    tl.store($(candidate_vmul_43_ptr) + r330_z, r330_value)
    r331_z = tl.arange($(6144))
    r331_value = tl.load($(candidate_vmul_43_ptr) + ((((r331_z / $(192)) % $(8)) * $(768)) + (((r331_z / $(1536)) % $(4)) * $(192)) + (r331_z % $(192))), dtype=tl.float32)
    r331_value = r331_value.to(tl.bfloat16)
    r331_value = tl.broadcast_to(r331_value, [$(6144)])
    tl.store($(candidate_vto_copy_102_ptr) + r331_z, r331_value)
    r332_row = tl.arange($(12))
    r332_col = tl.arange($(192))
    r332_lane = tl.arange($(4))
    for r332_group in range($(8)) {
      r332_left = tl.load($(candidate_vmul_42_ptr) + (((((((r332_group % $(8)) * $(48)) + (r332_row[:, None] * $(4)) + r332_lane[None, :]) / $(48)) % $(8)) * $(48)) + (((((r332_group % $(8)) * $(48)) + (r332_row[:, None] * $(4)) + r332_lane[None, :]) % $(4)) * $(12)) + (((((r332_group % $(8)) * $(48)) + (r332_row[:, None] * $(4)) + r332_lane[None, :]) / $(4)) % $(12))), dtype=tl.float32)
      r332_right = tl.load($(candidate_vto_copy_97_ptr) + (((((((r332_group % $(8)) * $(768)) + (r332_lane[:, None] * $(192)) + r332_col[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r332_group % $(8)) * $(768)) + (r332_lane[:, None] * $(192)) + r332_col[None, :]) / $(192)) % $(4)) * $(192)) + ((((r332_group % $(8)) * $(768)) + (r332_lane[:, None] * $(192)) + r332_col[None, :]) % $(192))), dtype=tl.float32)
      r332_value = tl.dot(r332_left, r332_right)
      r332_value = r332_value.to(tl.float32)
      tl.store($(candidate_vbmm_32_ptr) + ((r332_group * $(2304)) + (r332_row[:, None] * $(192)) + r332_col[None, :]), r332_value)
    }
    r333_z = tl.arange($(18432))
    r333_value = (tl.load($(candidate_vbmm_32_ptr) + ((((r333_z / $(2304)) % $(8)) * $(2304)) + (((r333_z / $(192)) % $(12)) * $(192)) + (r333_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r333_value = r333_value.to(tl.float32)
    r333_value = tl.broadcast_to(r333_value, [$(18432)])
    tl.store($(candidate_vmul_44_ptr) + r333_z, r333_value)
    r334_z = tl.arange($(18432))
    r334_value = tl.load($(candidate_vmul_44_ptr) + ((((r334_z / $(192)) % $(8)) * $(2304)) + (((r334_z / $(1536)) % $(12)) * $(192)) + (r334_z % $(192))), dtype=tl.float32)
    r334_value = r334_value.to(tl.bfloat16)
    r334_value = tl.broadcast_to(r334_value, [$(18432)])
    tl.store($(candidate_vto_copy_103_ptr) + r334_z, r334_value)
    r335_row = tl.arange($(12))
    r335_col = tl.arange($(128))
    r335_lane = tl.arange($(4))
    for r335_group in range($(8)) {
      r335_left = tl.load($(candidate_vexp_1_ptr) + (((((((r335_group % $(8)) * $(48)) + (r335_row[:, None] * $(4)) + r335_lane[None, :]) / $(48)) % $(8)) * $(48)) + (((((r335_group % $(8)) * $(48)) + (r335_row[:, None] * $(4)) + r335_lane[None, :]) % $(4)) * $(12)) + (((((r335_group % $(8)) * $(48)) + (r335_row[:, None] * $(4)) + r335_lane[None, :]) / $(4)) % $(12))), dtype=tl.float32)
      r335_right = tl.load($(candidate_vto_copy_100_ptr) + (((((((r335_group % $(8)) * $(512)) + (r335_lane[:, None] * $(128)) + r335_col[None, :]) / $(512)) % $(8)) * $(512)) + ((((((r335_group % $(8)) * $(512)) + (r335_lane[:, None] * $(128)) + r335_col[None, :]) / $(128)) % $(4)) * $(128)) + ((((r335_group % $(8)) * $(512)) + (r335_lane[:, None] * $(128)) + r335_col[None, :]) % $(128))), dtype=tl.float32)
      r335_value = tl.dot(r335_left, r335_right)
      r335_value = r335_value.to(tl.float32)
      tl.store($(candidate_vbmm_33_ptr) + ((r335_group * $(1536)) + (r335_row[:, None] * $(128)) + r335_col[None, :]), r335_value)
    }
    r336_z = tl.arange($(12288))
    r336_value = tl.load($(candidate_vbmm_33_ptr) + ((((r336_z / $(128)) % $(8)) * $(1536)) + (((r336_z / $(1024)) % $(12)) * $(128)) + (r336_z % $(128))), dtype=tl.float32)
    r336_value = r336_value.to(tl.bfloat16)
    r336_value = tl.broadcast_to(r336_value, [$(12288)])
    tl.store($(candidate_vto_copy_104_ptr) + r336_z, r336_value)
    r337_z = tl.arange($(24576))
    r337_value = tl.where(((r337_z / $(1536)) % $(16)) < $(12), tl.load($(candidate_vto_copy_103_ptr) + ((((((r337_z / $(1536)) % $(16)) - $(0)) % $(12)) * $(1536)) + (((r337_z / $(192)) % $(8)) * $(192)) + (r337_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r337_value = r337_value.to(tl.bfloat16)
    r337_value = tl.broadcast_to(r337_value, [$(24576)])
    tl.store($(candidate_vcat_19_ptr) + r337_z, r337_value)
    r338_z = tl.arange($(16384))
    r338_value = tl.where(((r338_z / $(1024)) % $(16)) < $(12), tl.load($(candidate_vto_copy_104_ptr) + ((((((r338_z / $(1024)) % $(16)) - $(0)) % $(12)) * $(1024)) + (((r338_z / $(128)) % $(8)) * $(128)) + (r338_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r338_value = r338_value.to(tl.bfloat16)
    r338_value = tl.broadcast_to(r338_value, [$(16384)])
    tl.store($(candidate_vcat_20_ptr) + r338_z, r338_value)
    r339_z = tl.arange($(24576))
    r339_value = tl.load($(candidate_vcat_15_ptr) + ((((r339_z / $(192)) % $(16)) * $(1536)) + (((r339_z / $(3072)) % $(8)) * $(192)) + (r339_z % $(192)) + $(24576)), dtype=tl.bfloat16)
    r339_value = r339_value.to(tl.float32)
    r339_value = tl.broadcast_to(r339_value, [$(24576)])
    tl.store($(candidate_vto_copy_105_ptr) + r339_z, r339_value)
    r340_z = tl.arange($(73728))
    r340_value = tl.load($(candidate_vcat_16_ptr) + ((((r340_z / $(192)) % $(48)) * $(1536)) + (((r340_z / $(9216)) % $(8)) * $(192)) + (r340_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r340_value = r340_value.to(tl.float32)
    r340_value = tl.broadcast_to(r340_value, [$(73728)])
    tl.store($(candidate_vto_copy_106_ptr) + r340_z, r340_value)
    r341_z = tl.arange($(49152))
    r341_value = tl.load($(candidate_vbmm_23_ptr) + ((((r341_z / $(128)) % $(48)) * $(1024)) + (((r341_z / $(6144)) % $(8)) * $(128)) + (r341_z % $(128)) + $(65536)), dtype=tl.bfloat16)
    r341_value = r341_value.to(tl.float32)
    r341_value = tl.broadcast_to(r341_value, [$(49152)])
    tl.store($(candidate_vto_copy_107_ptr) + r341_z, r341_value)
    r342_z = tl.arange($(16384))
    r342_value = tl.load($(candidate_vmm_28_ptr) + ((((r342_z / $(128)) % $(16)) * $(1024)) + (((r342_z / $(2048)) % $(8)) * $(128)) + (r342_z % $(128)) + $(114688)), dtype=tl.bfloat16)
    r342_value = r342_value.to(tl.float32)
    r342_value = tl.broadcast_to(r342_value, [$(16384)])
    tl.store($(candidate_vto_copy_108_ptr) + r342_z, r342_value)
    r343_z = tl.arange($(16384))
    r343_value = tl.load($(candidate_vbmm_21_ptr) + ((((r343_z / $(2048)) % $(8)) * $(4096)) + (((r343_z / $(128)) % $(16)) * $(128)) + (r343_z % $(128)) + $(2048)), dtype=tl.bfloat16)
    r343_value = r343_value.to(tl.float32)
    r343_value = tl.broadcast_to(r343_value, [$(16384)])
    tl.store($(candidate_vto_copy_109_ptr) + r343_z, r343_value)
    r344_row = tl.arange($(16))
    r344_col = tl.arange($(48))
    r344_lane = tl.arange($(192))
    for r344_group in range($(8)) {
      r344_left = tl.load($(candidate_vto_copy_105_ptr) + (((((((r344_group % $(8)) * $(3072)) + (r344_row[:, None] * $(192)) + r344_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r344_group % $(8)) * $(3072)) + (r344_row[:, None] * $(192)) + r344_lane[None, :]) / $(192)) % $(16)) * $(192)) + ((((r344_group % $(8)) * $(3072)) + (r344_row[:, None] * $(192)) + r344_lane[None, :]) % $(192))), dtype=tl.float32)
      r344_right = tl.load($(candidate_vto_copy_106_ptr) + (((((((r344_group % $(8)) * $(9216)) + (r344_lane[:, None] * $(48)) + r344_col[None, :]) / $(9216)) % $(8)) * $(9216)) + (((((r344_group % $(8)) * $(9216)) + (r344_lane[:, None] * $(48)) + r344_col[None, :]) % $(48)) * $(192)) + (((((r344_group % $(8)) * $(9216)) + (r344_lane[:, None] * $(48)) + r344_col[None, :]) / $(48)) % $(192))), dtype=tl.float32)
      r344_value = tl.dot(r344_left, r344_right)
      r344_value = r344_value.to(tl.float32)
      tl.store($(candidate_vbmm_34_ptr) + ((r344_group * $(768)) + (r344_row[:, None] * $(48)) + r344_col[None, :]), r344_value)
    }
    r345_z = tl.arange($(6144))
    r345_value = (tl.load($(candidate_vbmm_34_ptr) + ((((r345_z / $(768)) % $(8)) * $(768)) + (((r345_z / $(48)) % $(16)) * $(48)) + (r345_z % $(48))), dtype=tl.float32) * 0.07216878364870322)
    r345_value = r345_value.to(tl.float32)
    r345_value = tl.broadcast_to(r345_value, [$(6144)])
    tl.store($(candidate_vmul_45_ptr) + r345_z, r345_value)
    r346_z = tl.arange($(16))
    r346_value = (tl.cast(r346_z, tl.int64) + tl.full([], $(48), dtype=tl.int64))
    r346_value = tl.broadcast_to(r346_value, [$(16)])
    tl.store($(candidate_vadd_34_ptr) + r346_z, r346_value)
    r347_z = tl.arange($(16))
    r347_value = (tl.load($(candidate_vadd_34_ptr) + r347_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r347_value = tl.broadcast_to(r347_value, [$(16)])
    tl.store($(candidate_vsub_27_ptr) + r347_z, r347_value)
    r348_z = tl.arange($(768))
    r348_value = (tl.cast((r348_z % $(48)), tl.int64) > tl.load($(candidate_vsub_27_ptr) + ((r348_z / $(48)) % $(16)), dtype=tl.int64))
    r348_value = tl.broadcast_to(r348_value, [$(768)])
    tl.store($(candidate_vgt_14_ptr) + r348_z, r348_value)
    r349_z = tl.arange($(6144))
    r349_value = tl.where(tl.load($(candidate_vgt_14_ptr) + ((((r349_z / $(48)) % $(16)) * $(48)) + (r349_z % $(48))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_45_ptr) + ((((r349_z / $(768)) % $(8)) * $(768)) + (((r349_z / $(48)) % $(16)) * $(48)) + (r349_z % $(48))), dtype=tl.float32))
    r349_value = r349_value.to(tl.float32)
    r349_value = tl.broadcast_to(r349_value, [$(6144)])
    tl.store($(candidate_vmasked_fill_14_ptr) + r349_z, r349_value)
    r350_z = tl.arange($(6144))
    r350_value = (tl.load($(candidate_vmasked_fill_14_ptr) + ((((r350_z / $(768)) % $(8)) * $(768)) + (((r350_z / $(48)) % $(16)) * $(48)) + (r350_z % $(48))), dtype=tl.float32) - tl.load($(candidate_vcat_13_ptr) + ((((r350_z / $(48)) % $(16)) * $(8)) + ((r350_z / $(768)) % $(8)) + $(128)), dtype=tl.float32))
    r350_value = r350_value.to(tl.float32)
    r350_value = tl.broadcast_to(r350_value, [$(6144)])
    tl.store($(candidate_vsub_28_ptr) + r350_z, r350_value)
    r351_z = tl.arange($(6144))
    r351_value = tl.exp(tl.load($(candidate_vsub_28_ptr) + ((((r351_z / $(768)) % $(8)) * $(768)) + (((r351_z / $(48)) % $(16)) * $(48)) + (r351_z % $(48))), dtype=tl.float32))
    r351_value = r351_value.to(tl.float32)
    r351_value = tl.broadcast_to(r351_value, [$(6144)])
    tl.store($(candidate_vexp_2_ptr) + r351_z, r351_value)
    r352_z = tl.arange($(16384))
    r352_value = (tl.load($(candidate_vto_copy_108_ptr) + ((((r352_z / $(2048)) % $(8)) * $(2048)) + (((r352_z / $(128)) % $(16)) * $(128)) + (r352_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_109_ptr) + ((((r352_z / $(2048)) % $(8)) * $(2048)) + (((r352_z / $(128)) % $(16)) * $(128)) + (r352_z % $(128))), dtype=tl.float32))
    r352_value = r352_value.to(tl.float32)
    r352_value = tl.broadcast_to(r352_value, [$(16384)])
    tl.store($(candidate_vmul_46_ptr) + r352_z, r352_value)
    r353_z = tl.arange($(128))
    r353_lane = tl.arange($(128))
    r353_value = tl.sum(tl.load($(candidate_vmul_46_ptr) + ((((r353_z[:, None] / $(16)) % $(8)) * $(2048)) + ((r353_z[:, None] % $(16)) * $(128)) + r353_lane[None, :]), dtype=tl.float32), axis=1)
    r353_value = r353_value.to(tl.float32)
    r353_value = tl.broadcast_to(r353_value, [$(128)])
    tl.store($(candidate_vsum_3_ptr) + r353_z, r353_value)
    r354_row = tl.arange($(16))
    r354_col = tl.arange($(48))
    r354_lane = tl.arange($(128))
    for r354_group in range($(8)) {
      r354_left = tl.load($(candidate_vto_copy_108_ptr) + (((((((r354_group % $(8)) * $(2048)) + (r354_row[:, None] * $(128)) + r354_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r354_group % $(8)) * $(2048)) + (r354_row[:, None] * $(128)) + r354_lane[None, :]) / $(128)) % $(16)) * $(128)) + ((((r354_group % $(8)) * $(2048)) + (r354_row[:, None] * $(128)) + r354_lane[None, :]) % $(128))), dtype=tl.float32)
      r354_right = tl.load($(candidate_vto_copy_107_ptr) + (((((((r354_group % $(8)) * $(6144)) + (r354_lane[:, None] * $(48)) + r354_col[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r354_group % $(8)) * $(6144)) + (r354_lane[:, None] * $(48)) + r354_col[None, :]) % $(48)) * $(128)) + (((((r354_group % $(8)) * $(6144)) + (r354_lane[:, None] * $(48)) + r354_col[None, :]) / $(48)) % $(128))), dtype=tl.float32)
      r354_value = tl.dot(r354_left, r354_right)
      r354_value = r354_value.to(tl.float32)
      tl.store($(candidate_vbmm_35_ptr) + ((r354_group * $(768)) + (r354_row[:, None] * $(48)) + r354_col[None, :]), r354_value)
    }
    r355_z = tl.arange($(6144))
    r355_value = (tl.load($(candidate_vbmm_35_ptr) + ((((r355_z / $(768)) % $(8)) * $(768)) + (((r355_z / $(48)) % $(16)) * $(48)) + (r355_z % $(48))), dtype=tl.float32) - tl.load($(candidate_vsum_3_ptr) + ((((r355_z / $(768)) % $(8)) * $(16)) + ((r355_z / $(48)) % $(16))), dtype=tl.float32))
    r355_value = r355_value.to(tl.float32)
    r355_value = tl.broadcast_to(r355_value, [$(6144)])
    tl.store($(candidate_vsub_29_ptr) + r355_z, r355_value)
    r356_z = tl.arange($(6144))
    r356_value = (tl.load($(candidate_vexp_2_ptr) + ((((r356_z / $(768)) % $(8)) * $(768)) + (((r356_z / $(48)) % $(16)) * $(48)) + (r356_z % $(48))), dtype=tl.float32) * tl.load($(candidate_vsub_29_ptr) + ((((r356_z / $(768)) % $(8)) * $(768)) + (((r356_z / $(48)) % $(16)) * $(48)) + (r356_z % $(48))), dtype=tl.float32))
    r356_value = r356_value.to(tl.float32)
    r356_value = tl.broadcast_to(r356_value, [$(6144)])
    tl.store($(candidate_vmul_47_ptr) + r356_z, r356_value)
    r357_row = tl.arange($(16))
    r357_col = tl.arange($(192))
    r357_lane = tl.arange($(48))
    for r357_group in range($(8)) {
      r357_left = tl.load($(candidate_vmul_47_ptr) + (((((((r357_group % $(8)) * $(768)) + (r357_row[:, None] * $(48)) + r357_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r357_group % $(8)) * $(768)) + (r357_row[:, None] * $(48)) + r357_lane[None, :]) / $(48)) % $(16)) * $(48)) + ((((r357_group % $(8)) * $(768)) + (r357_row[:, None] * $(48)) + r357_lane[None, :]) % $(48))), dtype=tl.float32)
      r357_right = tl.load($(candidate_vto_copy_106_ptr) + (((((((r357_group % $(8)) * $(9216)) + (r357_lane[:, None] * $(192)) + r357_col[None, :]) / $(9216)) % $(8)) * $(9216)) + ((((((r357_group % $(8)) * $(9216)) + (r357_lane[:, None] * $(192)) + r357_col[None, :]) / $(192)) % $(48)) * $(192)) + ((((r357_group % $(8)) * $(9216)) + (r357_lane[:, None] * $(192)) + r357_col[None, :]) % $(192))), dtype=tl.float32)
      r357_value = tl.dot(r357_left, r357_right)
      r357_value = r357_value.to(tl.float32)
      tl.store($(candidate_vbmm_36_ptr) + ((r357_group * $(3072)) + (r357_row[:, None] * $(192)) + r357_col[None, :]), r357_value)
    }
    r358_z = tl.arange($(24576))
    r358_value = (tl.load($(candidate_vbmm_36_ptr) + ((((r358_z / $(3072)) % $(8)) * $(3072)) + (((r358_z / $(192)) % $(16)) * $(192)) + (r358_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r358_value = r358_value.to(tl.float32)
    r358_value = tl.broadcast_to(r358_value, [$(24576)])
    tl.store($(candidate_vmul_48_ptr) + r358_z, r358_value)
    r359_z = tl.arange($(24576))
    r359_value = tl.load($(candidate_vmul_48_ptr) + ((((r359_z / $(192)) % $(8)) * $(3072)) + (((r359_z / $(1536)) % $(16)) * $(192)) + (r359_z % $(192))), dtype=tl.float32)
    r359_value = r359_value.to(tl.bfloat16)
    r359_value = tl.broadcast_to(r359_value, [$(24576)])
    tl.store($(candidate_vto_copy_110_ptr) + r359_z, r359_value)
    r360_row = tl.arange($(48))
    r360_col = tl.arange($(192))
    r360_lane = tl.arange($(16))
    for r360_group in range($(8)) {
      r360_left = tl.load($(candidate_vmul_47_ptr) + (((((((r360_group % $(8)) * $(768)) + (r360_row[:, None] * $(16)) + r360_lane[None, :]) / $(768)) % $(8)) * $(768)) + (((((r360_group % $(8)) * $(768)) + (r360_row[:, None] * $(16)) + r360_lane[None, :]) % $(16)) * $(48)) + (((((r360_group % $(8)) * $(768)) + (r360_row[:, None] * $(16)) + r360_lane[None, :]) / $(16)) % $(48))), dtype=tl.float32)
      r360_right = tl.load($(candidate_vto_copy_105_ptr) + (((((((r360_group % $(8)) * $(3072)) + (r360_lane[:, None] * $(192)) + r360_col[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r360_group % $(8)) * $(3072)) + (r360_lane[:, None] * $(192)) + r360_col[None, :]) / $(192)) % $(16)) * $(192)) + ((((r360_group % $(8)) * $(3072)) + (r360_lane[:, None] * $(192)) + r360_col[None, :]) % $(192))), dtype=tl.float32)
      r360_value = tl.dot(r360_left, r360_right)
      r360_value = r360_value.to(tl.float32)
      tl.store($(candidate_vbmm_37_ptr) + ((r360_group * $(9216)) + (r360_row[:, None] * $(192)) + r360_col[None, :]), r360_value)
    }
    r361_z = tl.arange($(73728))
    r361_value = (tl.load($(candidate_vbmm_37_ptr) + ((((r361_z / $(9216)) % $(8)) * $(9216)) + (((r361_z / $(192)) % $(48)) * $(192)) + (r361_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r361_value = r361_value.to(tl.float32)
    r361_value = tl.broadcast_to(r361_value, [$(73728)])
    tl.store($(candidate_vmul_49_ptr) + r361_z, r361_value)
    r362_z = tl.arange($(73728))
    r362_value = tl.load($(candidate_vmul_49_ptr) + ((((r362_z / $(192)) % $(8)) * $(9216)) + (((r362_z / $(1536)) % $(48)) * $(192)) + (r362_z % $(192))), dtype=tl.float32)
    r362_value = r362_value.to(tl.bfloat16)
    r362_value = tl.broadcast_to(r362_value, [$(73728)])
    tl.store($(candidate_vto_copy_111_ptr) + r362_z, r362_value)
    r363_row = tl.arange($(48))
    r363_col = tl.arange($(128))
    r363_lane = tl.arange($(16))
    for r363_group in range($(8)) {
      r363_left = tl.load($(candidate_vexp_2_ptr) + (((((((r363_group % $(8)) * $(768)) + (r363_row[:, None] * $(16)) + r363_lane[None, :]) / $(768)) % $(8)) * $(768)) + (((((r363_group % $(8)) * $(768)) + (r363_row[:, None] * $(16)) + r363_lane[None, :]) % $(16)) * $(48)) + (((((r363_group % $(8)) * $(768)) + (r363_row[:, None] * $(16)) + r363_lane[None, :]) / $(16)) % $(48))), dtype=tl.float32)
      r363_right = tl.load($(candidate_vto_copy_108_ptr) + (((((((r363_group % $(8)) * $(2048)) + (r363_lane[:, None] * $(128)) + r363_col[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r363_group % $(8)) * $(2048)) + (r363_lane[:, None] * $(128)) + r363_col[None, :]) / $(128)) % $(16)) * $(128)) + ((((r363_group % $(8)) * $(2048)) + (r363_lane[:, None] * $(128)) + r363_col[None, :]) % $(128))), dtype=tl.float32)
      r363_value = tl.dot(r363_left, r363_right)
      r363_value = r363_value.to(tl.float32)
      tl.store($(candidate_vbmm_38_ptr) + ((r363_group * $(6144)) + (r363_row[:, None] * $(128)) + r363_col[None, :]), r363_value)
    }
    r364_z = tl.arange($(49152))
    r364_value = tl.load($(candidate_vbmm_38_ptr) + ((((r364_z / $(128)) % $(8)) * $(6144)) + (((r364_z / $(1024)) % $(48)) * $(128)) + (r364_z % $(128))), dtype=tl.float32)
    r364_value = r364_value.to(tl.bfloat16)
    r364_value = tl.broadcast_to(r364_value, [$(49152)])
    tl.store($(candidate_vto_copy_112_ptr) + r364_z, r364_value)
    r365_z = tl.arange($(98304))
    r365_value = tl.where(((r365_z / $(1536)) % $(64)) < $(48), tl.load($(candidate_vto_copy_111_ptr) + ((((((r365_z / $(1536)) % $(64)) - $(0)) % $(48)) * $(1536)) + (((r365_z / $(192)) % $(8)) * $(192)) + (r365_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r365_value = r365_value.to(tl.bfloat16)
    r365_value = tl.broadcast_to(r365_value, [$(98304)])
    tl.store($(candidate_vcat_21_ptr) + r365_z, r365_value)
    r366_z = tl.arange($(65536))
    r366_value = tl.where(((r366_z / $(1024)) % $(64)) < $(48), tl.load($(candidate_vto_copy_112_ptr) + ((((((r366_z / $(1024)) % $(64)) - $(0)) % $(48)) * $(1024)) + (((r366_z / $(128)) % $(8)) * $(128)) + (r366_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r366_value = r366_value.to(tl.bfloat16)
    r366_value = tl.broadcast_to(r366_value, [$(65536)])
    tl.store($(candidate_vcat_22_ptr) + r366_z, r366_value)
    r367_z = tl.arange($(49152))
    r367_value = tl.where(((r367_z / $(1536)) % $(32)) < $(12), tl.load($(candidate_vto_copy_94_ptr) + ((((((r367_z / $(1536)) % $(32)) - $(0)) % $(12)) * $(1536)) + (((r367_z / $(192)) % $(8)) * $(192)) + (r367_z % $(192))), dtype=tl.bfloat16), tl.where(((r367_z / $(1536)) % $(32)) < $(16), tl.load($(candidate_vto_copy_102_ptr) + ((((((r367_z / $(1536)) % $(32)) - $(12)) % $(4)) * $(1536)) + (((r367_z / $(192)) % $(8)) * $(192)) + (r367_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vto_copy_110_ptr) + ((((((r367_z / $(1536)) % $(32)) - $(16)) % $(16)) * $(1536)) + (((r367_z / $(192)) % $(8)) * $(192)) + (r367_z % $(192))), dtype=tl.bfloat16)))
    r367_value = r367_value.to(tl.bfloat16)
    r367_value = tl.broadcast_to(r367_value, [$(49152)])
    tl.store($(candidate_vcat_23_ptr) + r367_z, r367_value)
    r368_z = tl.arange($(196608))
    r368_value = tl.where(((r368_z / $(1536)) % $(128)) < $(48), tl.load($(candidate_vcat_17_ptr) + ((((((r368_z / $(1536)) % $(128)) - $(0)) % $(48)) * $(1536)) + (((r368_z / $(192)) % $(8)) * $(192)) + (r368_z % $(192))), dtype=tl.bfloat16), tl.where(((r368_z / $(1536)) % $(128)) < $(64), tl.load($(candidate_vcat_19_ptr) + ((((((r368_z / $(1536)) % $(128)) - $(48)) % $(16)) * $(1536)) + (((r368_z / $(192)) % $(8)) * $(192)) + (r368_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vcat_21_ptr) + ((((((r368_z / $(1536)) % $(128)) - $(64)) % $(64)) * $(1536)) + (((r368_z / $(192)) % $(8)) * $(192)) + (r368_z % $(192))), dtype=tl.bfloat16)))
    r368_value = r368_value.to(tl.bfloat16)
    r368_value = tl.broadcast_to(r368_value, [$(196608)])
    tl.store($(candidate_vcat_24_ptr) + r368_z, r368_value)
    r369_z = tl.arange($(131072))
    r369_value = tl.where(((r369_z / $(1024)) % $(128)) < $(48), tl.load($(candidate_vcat_18_ptr) + ((((((r369_z / $(1024)) % $(128)) - $(0)) % $(48)) * $(1024)) + (((r369_z / $(128)) % $(8)) * $(128)) + (r369_z % $(128))), dtype=tl.bfloat16), tl.where(((r369_z / $(1024)) % $(128)) < $(64), tl.load($(candidate_vcat_20_ptr) + ((((((r369_z / $(1024)) % $(128)) - $(48)) % $(16)) * $(1024)) + (((r369_z / $(128)) % $(8)) * $(128)) + (r369_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_22_ptr) + ((((((r369_z / $(1024)) % $(128)) - $(64)) % $(64)) * $(1024)) + (((r369_z / $(128)) % $(8)) * $(128)) + (r369_z % $(128))), dtype=tl.bfloat16)))
    r369_value = r369_value.to(tl.bfloat16)
    r369_value = tl.broadcast_to(r369_value, [$(131072)])
    tl.store($(candidate_vcat_25_ptr) + r369_z, r369_value)
    r370_z = tl.arange($(8192))
    r370_lane = tl.arange($(8))
    r370_value = tl.sum(tl.load($(candidate_vcat_24_ptr) + ((((r370_z[:, None] / $(64)) % $(128)) * $(1536)) + (r370_lane[None, :] * $(192)) + (r370_z[:, None] % $(64)) + $(128)), dtype=tl.bfloat16), axis=1)
    r370_value = r370_value.to(tl.bfloat16)
    r370_value = tl.broadcast_to(r370_value, [$(8192)])
    tl.store($(candidate_vsum_4_ptr) + r370_z, r370_value)
    r371_row = tl.arange($(128))
    r371_col = tl.arange($(512))
    r371_lane = tl.arange($(1024))
    for r371_group in range($(1)) {
      r371_left = tl.load($(candidate_vcat_24_ptr) + ((((((r371_row[:, None] * $(1024)) + r371_lane[None, :]) / $(1024)) % $(128)) * $(1536)) + ((((r371_row[:, None] * $(1024)) + r371_lane[None, :]) % $(8)) * $(192)) + ((((r371_row[:, None] * $(1024)) + r371_lane[None, :]) / $(8)) % $(128))), dtype=tl.bfloat16)
      r371_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r371_lane[:, None] * $(512)) + r371_col[None, :]) / $(512)) % $(8)) * $(131072)) + (((((r371_lane[:, None] * $(512)) + r371_col[None, :]) / $(4096)) % $(128)) * $(512)) + (((r371_lane[:, None] * $(512)) + r371_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r371_value = tl.dot(r371_left, r371_right)
      r371_value = r371_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_39_ptr) + ((r371_group * $(65536)) + (r371_row[:, None] * $(512)) + r371_col[None, :]), r371_value)
    }
    r372_row = tl.arange($(128))
    r372_col = tl.arange($(512))
    r372_lane = tl.arange($(1024))
    for r372_group in range($(1)) {
      r372_left = tl.load($(candidate_vcat_25_ptr) + ((((((r372_row[:, None] * $(1024)) + r372_lane[None, :]) / $(1024)) % $(128)) * $(1024)) + ((((r372_row[:, None] * $(1024)) + r372_lane[None, :]) % $(8)) * $(128)) + ((((r372_row[:, None] * $(1024)) + r372_lane[None, :]) / $(8)) % $(128))), dtype=tl.bfloat16)
      r372_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r372_lane[:, None] * $(512)) + r372_col[None, :]) / $(512)) % $(8)) * $(131072)) + (((((r372_lane[:, None] * $(512)) + r372_col[None, :]) / $(4096)) % $(128)) * $(512)) + (((r372_lane[:, None] * $(512)) + r372_col[None, :]) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r372_value = tl.dot(r372_left, r372_right)
      r372_value = r372_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_40_ptr) + ((r372_group * $(65536)) + (r372_row[:, None] * $(512)) + r372_col[None, :]), r372_value)
    }
    r373_z = tl.arange($(65536))
    r373_value = (tl.load($(candidate_vbmm_39_ptr) + ((((r373_z / $(512)) % $(128)) * $(512)) + (r373_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vbmm_40_ptr) + ((((r373_z / $(512)) % $(128)) * $(512)) + (r373_z % $(512))), dtype=tl.bfloat16))
    r373_value = r373_value.to(tl.bfloat16)
    r373_value = tl.broadcast_to(r373_value, [$(65536)])
    tl.store($(candidate_vadd_35_ptr) + r373_z, r373_value)
    r374_z = tl.arange($(65536))
    r374_value = tl.where(((((r374_z / $(512)) % $(128)) >= $(0)) & (((r374_z / $(512)) % $(128)) < $(128)) & ((((r374_z / $(512)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vadd_35_ptr) + ((((((r374_z / $(512)) % $(128)) - $(0)) % $(128)) * $(512)) + (r374_z % $(512))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r374_value = r374_value.to(tl.bfloat16)
    r374_value = tl.broadcast_to(r374_value, [$(65536)])
    tl.store($(candidate_vslice_backward_ptr) + r374_z, r374_value)
    r375_z = tl.arange($(8192))
    r375_value = tl.where(((((r375_z / $(64)) % $(128)) >= $(0)) & (((r375_z / $(64)) % $(128)) < $(128)) & ((((r375_z / $(64)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vsum_4_ptr) + ((((((r375_z / $(64)) % $(128)) - $(0)) % $(128)) * $(64)) + (r375_z % $(64))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r375_value = r375_value.to(tl.bfloat16)
    r375_value = tl.broadcast_to(r375_value, [$(8192)])
    tl.store($(candidate_vslice_backward_1_ptr) + r375_z, r375_value)
    r376_z = tl.arange($(131072))
    r376_value = tl.where(((((r376_z / $(1024)) % $(128)) >= $(96)) & (((r376_z / $(1024)) % $(128)) < $(128)) & ((((r376_z / $(1024)) % $(128)) - $(96)) % $(1) == $(0))), tl.load($(candidate_vcat_23_ptr) + ((((((r376_z / $(1024)) % $(128)) - $(96)) % $(32)) * $(1536)) + (((r376_z / $(128)) % $(8)) * $(192)) + (r376_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r376_value = r376_value.to(tl.bfloat16)
    r376_value = tl.broadcast_to(r376_value, [$(131072)])
    tl.store($(candidate_vslice_backward_2_ptr) + r376_z, r376_value)
    r377_z = tl.arange($(65536))
    r377_value = tl.where(((((r377_z / $(512)) % $(128)) >= $(96)) & (((r377_z / $(512)) % $(128)) < $(128)) & ((((r377_z / $(512)) % $(128)) - $(96)) % $(1) == $(0))), tl.load($(candidate_vcat_23_ptr) + ((((((r377_z / $(512)) % $(128)) - $(96)) % $(32)) * $(1536)) + (((r377_z / $(64)) % $(8)) * $(192)) + (r377_z % $(64)) + $(128)), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r377_value = r377_value.to(tl.bfloat16)
    r377_value = tl.broadcast_to(r377_value, [$(65536)])
    tl.store($(candidate_vslice_backward_3_ptr) + r377_z, r377_value)
    r378_row = tl.arange($(128))
    r378_col = tl.arange($(1024))
    r378_lane = tl.arange($(512))
    for r378_group in range($(1)) {
      r378_left = tl.load($(candidate_vindex_2_ptr) + ((((((r378_row[:, None] * $(512)) + r378_lane[None, :]) / $(512)) % $(128)) * $(576)) + (((r378_row[:, None] * $(512)) + r378_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r378_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r378_lane[:, None] * $(1024)) + r378_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r378_lane[:, None] * $(1024)) + r378_col[None, :]) % $(128)) * $(512)) + ((((r378_lane[:, None] * $(1024)) + r378_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r378_value = tl.dot(r378_left, r378_right)
      r378_value = r378_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_43_ptr) + ((r378_group * $(131072)) + (r378_row[:, None] * $(1024)) + r378_col[None, :]), r378_value)
    }
    r379_row = tl.arange($(128))
    r379_col = tl.arange($(1024))
    r379_lane = tl.arange($(512))
    for r379_group in range($(1)) {
      r379_left = tl.load($(candidate_vindex_2_ptr) + ((((((r379_row[:, None] * $(512)) + r379_lane[None, :]) / $(512)) % $(128)) * $(576)) + (((r379_row[:, None] * $(512)) + r379_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r379_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r379_lane[:, None] * $(1024)) + r379_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r379_lane[:, None] * $(1024)) + r379_col[None, :]) % $(128)) * $(512)) + ((((r379_lane[:, None] * $(1024)) + r379_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r379_value = tl.dot(r379_left, r379_right)
      r379_value = r379_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_44_ptr) + ((r379_group * $(131072)) + (r379_row[:, None] * $(1024)) + r379_col[None, :]), r379_value)
    }
    r380_z = tl.arange($(49152))
    r380_value = tl.where((r380_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r380_z / $(1536)) % $(32)) * $(1024)) + (((r380_z / $(192)) % $(8)) * $(128)) + ((r380_z % $(192)) - $(0)) + $(65536)) / $(1024)) % $(128)) * $(1536)) + (((((((r380_z / $(1536)) % $(32)) * $(1024)) + (((r380_z / $(192)) % $(8)) * $(128)) + ((r380_z % $(192)) - $(0)) + $(65536)) / $(128)) % $(8)) * $(192)) + (((((r380_z / $(1536)) % $(32)) * $(1024)) + (((r380_z / $(192)) % $(8)) * $(128)) + ((r380_z % $(192)) - $(0)) + $(65536)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vadd_6_ptr) + ((((((((r380_z / $(1536)) % $(32)) * $(512)) + (((r380_z / $(192)) % $(8)) * $(64)) + ((r380_z % $(192)) - $(128)) + $(32768)) / $(512)) % $(128)) * $(512)) + (((((((r380_z / $(1536)) % $(32)) * $(512)) + (((r380_z / $(192)) % $(8)) * $(64)) + ((r380_z % $(192)) - $(128)) + $(32768)) / $(64)) % $(8)) * $(64)) + (((((r380_z / $(1536)) % $(32)) * $(512)) + (((r380_z / $(192)) % $(8)) * $(64)) + ((r380_z % $(192)) - $(128)) + $(32768)) % $(64))), dtype=tl.bfloat16))
    r380_value = r380_value.to(tl.bfloat16)
    r380_value = tl.broadcast_to(r380_value, [$(49152)])
    tl.store($(candidate_vcat_27_ptr) + r380_z, r380_value)
    r381_z = tl.arange($(196608))
    r381_value = tl.where((r381_z % $(192)) < $(128), tl.load($(candidate_vbmm_43_ptr) + ((((((((r381_z / $(1536)) % $(128)) * $(1024)) + (((r381_z / $(192)) % $(8)) * $(128)) + ((r381_z % $(192)) - $(0))) / $(1024)) % $(128)) * $(1024)) + (((((((r381_z / $(1536)) % $(128)) * $(1024)) + (((r381_z / $(192)) % $(8)) * $(128)) + ((r381_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r381_z / $(1536)) % $(128)) * $(1024)) + (((r381_z / $(192)) % $(8)) * $(128)) + ((r381_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vindex_2_ptr) + ((((r381_z / $(1536)) % $(128)) * $(576)) + ((((r381_z % $(192)) - $(128)) + $(512)) % $(576))), dtype=tl.bfloat16))
    r381_value = r381_value.to(tl.bfloat16)
    r381_value = tl.broadcast_to(r381_value, [$(196608)])
    tl.store($(candidate_vcat_28_ptr) + r381_z, r381_value)
    r382_z = tl.arange($(18432))
    r382_value = tl.load($(candidate_vcat_27_ptr) + ((((r382_z / $(192)) % $(12)) * $(1536)) + (((r382_z / $(2304)) % $(8)) * $(192)) + (r382_z % $(192))), dtype=tl.bfloat16)
    r382_value = r382_value.to(tl.float32)
    r382_value = tl.broadcast_to(r382_value, [$(18432)])
    tl.store($(candidate_vto_copy_113_ptr) + r382_z, r382_value)
    r383_z = tl.arange($(36864))
    r383_value = tl.load($(candidate_vcat_28_ptr) + ((((r383_z / $(192)) % $(24)) * $(1536)) + (((r383_z / $(4608)) % $(8)) * $(192)) + (r383_z % $(192))), dtype=tl.bfloat16)
    r383_value = r383_value.to(tl.float32)
    r383_value = tl.broadcast_to(r383_value, [$(36864)])
    tl.store($(candidate_vto_copy_114_ptr) + r383_z, r383_value)
    r384_z = tl.arange($(24576))
    r384_value = tl.load($(candidate_vbmm_44_ptr) + ((((r384_z / $(128)) % $(24)) * $(1024)) + (((r384_z / $(3072)) % $(8)) * $(128)) + (r384_z % $(128))), dtype=tl.bfloat16)
    r384_value = r384_value.to(tl.float32)
    r384_value = tl.broadcast_to(r384_value, [$(24576)])
    tl.store($(candidate_vto_copy_115_ptr) + r384_z, r384_value)
    r385_z = tl.arange($(12288))
    r385_value = tl.load($(candidate_vmm_28_ptr) + ((((r385_z / $(128)) % $(12)) * $(1024)) + (((r385_z / $(1536)) % $(8)) * $(128)) + (r385_z % $(128)) + $(65536)), dtype=tl.bfloat16)
    r385_value = r385_value.to(tl.float32)
    r385_value = tl.broadcast_to(r385_value, [$(12288)])
    tl.store($(candidate_vto_copy_116_ptr) + r385_z, r385_value)
    r386_z = tl.arange($(12288))
    r386_value = tl.load($(candidate_vbmm_16_ptr) + ((((r386_z / $(1536)) % $(8)) * $(4096)) + (((r386_z / $(128)) % $(12)) * $(128)) + (r386_z % $(128))), dtype=tl.bfloat16)
    r386_value = r386_value.to(tl.float32)
    r386_value = tl.broadcast_to(r386_value, [$(12288)])
    tl.store($(candidate_vto_copy_117_ptr) + r386_z, r386_value)
    r387_row = tl.arange($(12))
    r387_col = tl.arange($(24))
    r387_lane = tl.arange($(192))
    for r387_group in range($(8)) {
      r387_left = tl.load($(candidate_vto_copy_113_ptr) + (((((((r387_group % $(8)) * $(2304)) + (r387_row[:, None] * $(192)) + r387_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r387_group % $(8)) * $(2304)) + (r387_row[:, None] * $(192)) + r387_lane[None, :]) / $(192)) % $(12)) * $(192)) + ((((r387_group % $(8)) * $(2304)) + (r387_row[:, None] * $(192)) + r387_lane[None, :]) % $(192))), dtype=tl.float32)
      r387_right = tl.load($(candidate_vto_copy_114_ptr) + (((((((r387_group % $(8)) * $(4608)) + (r387_lane[:, None] * $(24)) + r387_col[None, :]) / $(4608)) % $(8)) * $(4608)) + (((((r387_group % $(8)) * $(4608)) + (r387_lane[:, None] * $(24)) + r387_col[None, :]) % $(24)) * $(192)) + (((((r387_group % $(8)) * $(4608)) + (r387_lane[:, None] * $(24)) + r387_col[None, :]) / $(24)) % $(192))), dtype=tl.float32)
      r387_value = tl.dot(r387_left, r387_right)
      r387_value = r387_value.to(tl.float32)
      tl.store($(candidate_vbmm_45_ptr) + ((r387_group * $(288)) + (r387_row[:, None] * $(24)) + r387_col[None, :]), r387_value)
    }
    r388_z = tl.arange($(2304))
    r388_value = (tl.load($(candidate_vbmm_45_ptr) + ((((r388_z / $(288)) % $(8)) * $(288)) + (((r388_z / $(24)) % $(12)) * $(24)) + (r388_z % $(24))), dtype=tl.float32) * 0.07216878364870322)
    r388_value = r388_value.to(tl.float32)
    r388_value = tl.broadcast_to(r388_value, [$(2304)])
    tl.store($(candidate_vmul_50_ptr) + r388_z, r388_value)
    r389_z = tl.arange($(12))
    r389_value = (tl.cast(r389_z, tl.int64) + tl.full([], $(24), dtype=tl.int64))
    r389_value = tl.broadcast_to(r389_value, [$(12)])
    tl.store($(candidate_vadd_36_ptr) + r389_z, r389_value)
    r390_z = tl.arange($(12))
    r390_value = (tl.load($(candidate_vadd_36_ptr) + r390_z, dtype=tl.int64) - tl.full([], $(12), dtype=tl.int64))
    r390_value = tl.broadcast_to(r390_value, [$(12)])
    tl.store($(candidate_vsub_30_ptr) + r390_z, r390_value)
    r391_z = tl.arange($(288))
    r391_value = (tl.cast((r391_z % $(24)), tl.int64) > tl.load($(candidate_vsub_30_ptr) + ((r391_z / $(24)) % $(12)), dtype=tl.int64))
    r391_value = tl.broadcast_to(r391_value, [$(288)])
    tl.store($(candidate_vgt_15_ptr) + r391_z, r391_value)
    r392_z = tl.arange($(2304))
    r392_value = tl.where(tl.load($(candidate_vgt_15_ptr) + ((((r392_z / $(24)) % $(12)) * $(24)) + (r392_z % $(24))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_50_ptr) + ((((r392_z / $(288)) % $(8)) * $(288)) + (((r392_z / $(24)) % $(12)) * $(24)) + (r392_z % $(24))), dtype=tl.float32))
    r392_value = r392_value.to(tl.float32)
    r392_value = tl.broadcast_to(r392_value, [$(2304)])
    tl.store($(candidate_vmasked_fill_15_ptr) + r392_z, r392_value)
    r393_z = tl.arange($(2304))
    r393_value = (tl.load($(candidate_vmasked_fill_15_ptr) + ((((r393_z / $(288)) % $(8)) * $(288)) + (((r393_z / $(24)) % $(12)) * $(24)) + (r393_z % $(24))), dtype=tl.float32) - tl.load($(candidate_vcat_11_ptr) + ((((r393_z / $(24)) % $(12)) * $(8)) + ((r393_z / $(288)) % $(8))), dtype=tl.float32))
    r393_value = r393_value.to(tl.float32)
    r393_value = tl.broadcast_to(r393_value, [$(2304)])
    tl.store($(candidate_vsub_31_ptr) + r393_z, r393_value)
    r394_z = tl.arange($(2304))
    r394_value = tl.exp(tl.load($(candidate_vsub_31_ptr) + ((((r394_z / $(288)) % $(8)) * $(288)) + (((r394_z / $(24)) % $(12)) * $(24)) + (r394_z % $(24))), dtype=tl.float32))
    r394_value = r394_value.to(tl.float32)
    r394_value = tl.broadcast_to(r394_value, [$(2304)])
    tl.store($(candidate_vexp_3_ptr) + r394_z, r394_value)
    r395_z = tl.arange($(12288))
    r395_value = (tl.load($(candidate_vto_copy_116_ptr) + ((((r395_z / $(1536)) % $(8)) * $(1536)) + (((r395_z / $(128)) % $(12)) * $(128)) + (r395_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_117_ptr) + ((((r395_z / $(1536)) % $(8)) * $(1536)) + (((r395_z / $(128)) % $(12)) * $(128)) + (r395_z % $(128))), dtype=tl.float32))
    r395_value = r395_value.to(tl.float32)
    r395_value = tl.broadcast_to(r395_value, [$(12288)])
    tl.store($(candidate_vmul_51_ptr) + r395_z, r395_value)
    r396_z = tl.arange($(96))
    r396_lane = tl.arange($(128))
    r396_value = tl.sum(tl.load($(candidate_vmul_51_ptr) + ((((r396_z[:, None] / $(12)) % $(8)) * $(1536)) + ((r396_z[:, None] % $(12)) * $(128)) + r396_lane[None, :]), dtype=tl.float32), axis=1)
    r396_value = r396_value.to(tl.float32)
    r396_value = tl.broadcast_to(r396_value, [$(96)])
    tl.store($(candidate_vsum_5_ptr) + r396_z, r396_value)
    r397_row = tl.arange($(12))
    r397_col = tl.arange($(24))
    r397_lane = tl.arange($(128))
    for r397_group in range($(8)) {
      r397_left = tl.load($(candidate_vto_copy_116_ptr) + (((((((r397_group % $(8)) * $(1536)) + (r397_row[:, None] * $(128)) + r397_lane[None, :]) / $(1536)) % $(8)) * $(1536)) + ((((((r397_group % $(8)) * $(1536)) + (r397_row[:, None] * $(128)) + r397_lane[None, :]) / $(128)) % $(12)) * $(128)) + ((((r397_group % $(8)) * $(1536)) + (r397_row[:, None] * $(128)) + r397_lane[None, :]) % $(128))), dtype=tl.float32)
      r397_right = tl.load($(candidate_vto_copy_115_ptr) + (((((((r397_group % $(8)) * $(3072)) + (r397_lane[:, None] * $(24)) + r397_col[None, :]) / $(3072)) % $(8)) * $(3072)) + (((((r397_group % $(8)) * $(3072)) + (r397_lane[:, None] * $(24)) + r397_col[None, :]) % $(24)) * $(128)) + (((((r397_group % $(8)) * $(3072)) + (r397_lane[:, None] * $(24)) + r397_col[None, :]) / $(24)) % $(128))), dtype=tl.float32)
      r397_value = tl.dot(r397_left, r397_right)
      r397_value = r397_value.to(tl.float32)
      tl.store($(candidate_vbmm_46_ptr) + ((r397_group * $(288)) + (r397_row[:, None] * $(24)) + r397_col[None, :]), r397_value)
    }
    r398_z = tl.arange($(2304))
    r398_value = (tl.load($(candidate_vbmm_46_ptr) + ((((r398_z / $(288)) % $(8)) * $(288)) + (((r398_z / $(24)) % $(12)) * $(24)) + (r398_z % $(24))), dtype=tl.float32) - tl.load($(candidate_vsum_5_ptr) + ((((r398_z / $(288)) % $(8)) * $(12)) + ((r398_z / $(24)) % $(12))), dtype=tl.float32))
    r398_value = r398_value.to(tl.float32)
    r398_value = tl.broadcast_to(r398_value, [$(2304)])
    tl.store($(candidate_vsub_32_ptr) + r398_z, r398_value)
    r399_z = tl.arange($(2304))
    r399_value = (tl.load($(candidate_vexp_3_ptr) + ((((r399_z / $(288)) % $(8)) * $(288)) + (((r399_z / $(24)) % $(12)) * $(24)) + (r399_z % $(24))), dtype=tl.float32) * tl.load($(candidate_vsub_32_ptr) + ((((r399_z / $(288)) % $(8)) * $(288)) + (((r399_z / $(24)) % $(12)) * $(24)) + (r399_z % $(24))), dtype=tl.float32))
    r399_value = r399_value.to(tl.float32)
    r399_value = tl.broadcast_to(r399_value, [$(2304)])
    tl.store($(candidate_vmul_52_ptr) + r399_z, r399_value)
    r400_row = tl.arange($(12))
    r400_col = tl.arange($(192))
    r400_lane = tl.arange($(24))
    for r400_group in range($(8)) {
      r400_left = tl.load($(candidate_vmul_52_ptr) + (((((((r400_group % $(8)) * $(288)) + (r400_row[:, None] * $(24)) + r400_lane[None, :]) / $(288)) % $(8)) * $(288)) + ((((((r400_group % $(8)) * $(288)) + (r400_row[:, None] * $(24)) + r400_lane[None, :]) / $(24)) % $(12)) * $(24)) + ((((r400_group % $(8)) * $(288)) + (r400_row[:, None] * $(24)) + r400_lane[None, :]) % $(24))), dtype=tl.float32)
      r400_right = tl.load($(candidate_vto_copy_114_ptr) + (((((((r400_group % $(8)) * $(4608)) + (r400_lane[:, None] * $(192)) + r400_col[None, :]) / $(4608)) % $(8)) * $(4608)) + ((((((r400_group % $(8)) * $(4608)) + (r400_lane[:, None] * $(192)) + r400_col[None, :]) / $(192)) % $(24)) * $(192)) + ((((r400_group % $(8)) * $(4608)) + (r400_lane[:, None] * $(192)) + r400_col[None, :]) % $(192))), dtype=tl.float32)
      r400_value = tl.dot(r400_left, r400_right)
      r400_value = r400_value.to(tl.float32)
      tl.store($(candidate_vbmm_47_ptr) + ((r400_group * $(2304)) + (r400_row[:, None] * $(192)) + r400_col[None, :]), r400_value)
    }
    r401_z = tl.arange($(18432))
    r401_value = (tl.load($(candidate_vbmm_47_ptr) + ((((r401_z / $(2304)) % $(8)) * $(2304)) + (((r401_z / $(192)) % $(12)) * $(192)) + (r401_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r401_value = r401_value.to(tl.float32)
    r401_value = tl.broadcast_to(r401_value, [$(18432)])
    tl.store($(candidate_vmul_53_ptr) + r401_z, r401_value)
    r402_z = tl.arange($(18432))
    r402_value = tl.load($(candidate_vmul_53_ptr) + ((((r402_z / $(192)) % $(8)) * $(2304)) + (((r402_z / $(1536)) % $(12)) * $(192)) + (r402_z % $(192))), dtype=tl.float32)
    r402_value = r402_value.to(tl.bfloat16)
    r402_value = tl.broadcast_to(r402_value, [$(18432)])
    tl.store($(candidate_vto_copy_118_ptr) + r402_z, r402_value)
    r403_row = tl.arange($(24))
    r403_col = tl.arange($(192))
    r403_lane = tl.arange($(12))
    for r403_group in range($(8)) {
      r403_left = tl.load($(candidate_vmul_52_ptr) + (((((((r403_group % $(8)) * $(288)) + (r403_row[:, None] * $(12)) + r403_lane[None, :]) / $(288)) % $(8)) * $(288)) + (((((r403_group % $(8)) * $(288)) + (r403_row[:, None] * $(12)) + r403_lane[None, :]) % $(12)) * $(24)) + (((((r403_group % $(8)) * $(288)) + (r403_row[:, None] * $(12)) + r403_lane[None, :]) / $(12)) % $(24))), dtype=tl.float32)
      r403_right = tl.load($(candidate_vto_copy_113_ptr) + (((((((r403_group % $(8)) * $(2304)) + (r403_lane[:, None] * $(192)) + r403_col[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r403_group % $(8)) * $(2304)) + (r403_lane[:, None] * $(192)) + r403_col[None, :]) / $(192)) % $(12)) * $(192)) + ((((r403_group % $(8)) * $(2304)) + (r403_lane[:, None] * $(192)) + r403_col[None, :]) % $(192))), dtype=tl.float32)
      r403_value = tl.dot(r403_left, r403_right)
      r403_value = r403_value.to(tl.float32)
      tl.store($(candidate_vbmm_48_ptr) + ((r403_group * $(4608)) + (r403_row[:, None] * $(192)) + r403_col[None, :]), r403_value)
    }
    r404_z = tl.arange($(36864))
    r404_value = (tl.load($(candidate_vbmm_48_ptr) + ((((r404_z / $(4608)) % $(8)) * $(4608)) + (((r404_z / $(192)) % $(24)) * $(192)) + (r404_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r404_value = r404_value.to(tl.float32)
    r404_value = tl.broadcast_to(r404_value, [$(36864)])
    tl.store($(candidate_vmul_54_ptr) + r404_z, r404_value)
    r405_z = tl.arange($(36864))
    r405_value = tl.load($(candidate_vmul_54_ptr) + ((((r405_z / $(192)) % $(8)) * $(4608)) + (((r405_z / $(1536)) % $(24)) * $(192)) + (r405_z % $(192))), dtype=tl.float32)
    r405_value = r405_value.to(tl.bfloat16)
    r405_value = tl.broadcast_to(r405_value, [$(36864)])
    tl.store($(candidate_vto_copy_119_ptr) + r405_z, r405_value)
    r406_row = tl.arange($(24))
    r406_col = tl.arange($(128))
    r406_lane = tl.arange($(12))
    for r406_group in range($(8)) {
      r406_left = tl.load($(candidate_vexp_3_ptr) + (((((((r406_group % $(8)) * $(288)) + (r406_row[:, None] * $(12)) + r406_lane[None, :]) / $(288)) % $(8)) * $(288)) + (((((r406_group % $(8)) * $(288)) + (r406_row[:, None] * $(12)) + r406_lane[None, :]) % $(12)) * $(24)) + (((((r406_group % $(8)) * $(288)) + (r406_row[:, None] * $(12)) + r406_lane[None, :]) / $(12)) % $(24))), dtype=tl.float32)
      r406_right = tl.load($(candidate_vto_copy_116_ptr) + (((((((r406_group % $(8)) * $(1536)) + (r406_lane[:, None] * $(128)) + r406_col[None, :]) / $(1536)) % $(8)) * $(1536)) + ((((((r406_group % $(8)) * $(1536)) + (r406_lane[:, None] * $(128)) + r406_col[None, :]) / $(128)) % $(12)) * $(128)) + ((((r406_group % $(8)) * $(1536)) + (r406_lane[:, None] * $(128)) + r406_col[None, :]) % $(128))), dtype=tl.float32)
      r406_value = tl.dot(r406_left, r406_right)
      r406_value = r406_value.to(tl.float32)
      tl.store($(candidate_vbmm_49_ptr) + ((r406_group * $(3072)) + (r406_row[:, None] * $(128)) + r406_col[None, :]), r406_value)
    }
    r407_z = tl.arange($(24576))
    r407_value = tl.load($(candidate_vbmm_49_ptr) + ((((r407_z / $(128)) % $(8)) * $(3072)) + (((r407_z / $(1024)) % $(24)) * $(128)) + (r407_z % $(128))), dtype=tl.float32)
    r407_value = r407_value.to(tl.bfloat16)
    r407_value = tl.broadcast_to(r407_value, [$(24576)])
    tl.store($(candidate_vto_copy_120_ptr) + r407_z, r407_value)
    r408_z = tl.arange($(73728))
    r408_value = tl.where(((r408_z / $(1536)) % $(48)) < $(24), tl.load($(candidate_vto_copy_119_ptr) + ((((((r408_z / $(1536)) % $(48)) - $(0)) % $(24)) * $(1536)) + (((r408_z / $(192)) % $(8)) * $(192)) + (r408_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r408_value = r408_value.to(tl.bfloat16)
    r408_value = tl.broadcast_to(r408_value, [$(73728)])
    tl.store($(candidate_vcat_29_ptr) + r408_z, r408_value)
    r409_z = tl.arange($(49152))
    r409_value = tl.where(((r409_z / $(1024)) % $(48)) < $(24), tl.load($(candidate_vto_copy_120_ptr) + ((((((r409_z / $(1024)) % $(48)) - $(0)) % $(24)) * $(1024)) + (((r409_z / $(128)) % $(8)) * $(128)) + (r409_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r409_value = r409_value.to(tl.bfloat16)
    r409_value = tl.broadcast_to(r409_value, [$(49152)])
    tl.store($(candidate_vcat_30_ptr) + r409_z, r409_value)
    r410_z = tl.arange($(6144))
    r410_value = tl.load($(candidate_vcat_27_ptr) + ((((r410_z / $(192)) % $(4)) * $(1536)) + (((r410_z / $(768)) % $(8)) * $(192)) + (r410_z % $(192)) + $(18432)), dtype=tl.bfloat16)
    r410_value = r410_value.to(tl.float32)
    r410_value = tl.broadcast_to(r410_value, [$(6144)])
    tl.store($(candidate_vto_copy_121_ptr) + r410_z, r410_value)
    r411_z = tl.arange($(12288))
    r411_value = tl.load($(candidate_vcat_28_ptr) + ((((r411_z / $(192)) % $(8)) * $(1536)) + (((r411_z / $(1536)) % $(8)) * $(192)) + (r411_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r411_value = r411_value.to(tl.float32)
    r411_value = tl.broadcast_to(r411_value, [$(12288)])
    tl.store($(candidate_vto_copy_122_ptr) + r411_z, r411_value)
    r412_z = tl.arange($(8192))
    r412_value = tl.load($(candidate_vbmm_44_ptr) + ((((r412_z / $(128)) % $(8)) * $(1024)) + (((r412_z / $(1024)) % $(8)) * $(128)) + (r412_z % $(128)) + $(49152)), dtype=tl.bfloat16)
    r412_value = r412_value.to(tl.float32)
    r412_value = tl.broadcast_to(r412_value, [$(8192)])
    tl.store($(candidate_vto_copy_123_ptr) + r412_z, r412_value)
    r413_z = tl.arange($(4096))
    r413_value = tl.load($(candidate_vmm_28_ptr) + ((((r413_z / $(128)) % $(4)) * $(1024)) + (((r413_z / $(512)) % $(8)) * $(128)) + (r413_z % $(128)) + $(77824)), dtype=tl.bfloat16)
    r413_value = r413_value.to(tl.float32)
    r413_value = tl.broadcast_to(r413_value, [$(4096)])
    tl.store($(candidate_vto_copy_124_ptr) + r413_z, r413_value)
    r414_z = tl.arange($(4096))
    r414_value = tl.load($(candidate_vbmm_16_ptr) + ((((r414_z / $(512)) % $(8)) * $(4096)) + (((r414_z / $(128)) % $(4)) * $(128)) + (r414_z % $(128)) + $(1536)), dtype=tl.bfloat16)
    r414_value = r414_value.to(tl.float32)
    r414_value = tl.broadcast_to(r414_value, [$(4096)])
    tl.store($(candidate_vto_copy_125_ptr) + r414_z, r414_value)
    r415_row = tl.arange($(4))
    r415_col = tl.arange($(8))
    r415_lane = tl.arange($(192))
    for r415_group in range($(8)) {
      r415_left = tl.load($(candidate_vto_copy_121_ptr) + (((((((r415_group % $(8)) * $(768)) + (r415_row[:, None] * $(192)) + r415_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r415_group % $(8)) * $(768)) + (r415_row[:, None] * $(192)) + r415_lane[None, :]) / $(192)) % $(4)) * $(192)) + ((((r415_group % $(8)) * $(768)) + (r415_row[:, None] * $(192)) + r415_lane[None, :]) % $(192))), dtype=tl.float32)
      r415_right = tl.load($(candidate_vto_copy_122_ptr) + (((((((r415_group % $(8)) * $(1536)) + (r415_lane[:, None] * $(8)) + r415_col[None, :]) / $(1536)) % $(8)) * $(1536)) + (((((r415_group % $(8)) * $(1536)) + (r415_lane[:, None] * $(8)) + r415_col[None, :]) % $(8)) * $(192)) + (((((r415_group % $(8)) * $(1536)) + (r415_lane[:, None] * $(8)) + r415_col[None, :]) / $(8)) % $(192))), dtype=tl.float32)
      r415_value = tl.dot(r415_left, r415_right)
      r415_value = r415_value.to(tl.float32)
      tl.store($(candidate_vbmm_50_ptr) + ((r415_group * $(32)) + (r415_row[:, None] * $(8)) + r415_col[None, :]), r415_value)
    }
    r416_z = tl.arange($(256))
    r416_value = (tl.load($(candidate_vbmm_50_ptr) + ((((r416_z / $(32)) % $(8)) * $(32)) + (((r416_z / $(8)) % $(4)) * $(8)) + (r416_z % $(8))), dtype=tl.float32) * 0.07216878364870322)
    r416_value = r416_value.to(tl.float32)
    r416_value = tl.broadcast_to(r416_value, [$(256)])
    tl.store($(candidate_vmul_55_ptr) + r416_z, r416_value)
    r417_z = tl.arange($(4))
    r417_value = (tl.cast(r417_z, tl.int64) + tl.full([], $(8), dtype=tl.int64))
    r417_value = tl.broadcast_to(r417_value, [$(4)])
    tl.store($(candidate_vadd_37_ptr) + r417_z, r417_value)
    r418_z = tl.arange($(4))
    r418_value = (tl.load($(candidate_vadd_37_ptr) + r418_z, dtype=tl.int64) - tl.full([], $(4), dtype=tl.int64))
    r418_value = tl.broadcast_to(r418_value, [$(4)])
    tl.store($(candidate_vsub_33_ptr) + r418_z, r418_value)
    r419_z = tl.arange($(32))
    r419_value = (tl.cast((r419_z % $(8)), tl.int64) > tl.load($(candidate_vsub_33_ptr) + ((r419_z / $(8)) % $(4)), dtype=tl.int64))
    r419_value = tl.broadcast_to(r419_value, [$(32)])
    tl.store($(candidate_vgt_16_ptr) + r419_z, r419_value)
    r420_z = tl.arange($(256))
    r420_value = tl.where(tl.load($(candidate_vgt_16_ptr) + ((((r420_z / $(8)) % $(4)) * $(8)) + (r420_z % $(8))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_55_ptr) + ((((r420_z / $(32)) % $(8)) * $(32)) + (((r420_z / $(8)) % $(4)) * $(8)) + (r420_z % $(8))), dtype=tl.float32))
    r420_value = r420_value.to(tl.float32)
    r420_value = tl.broadcast_to(r420_value, [$(256)])
    tl.store($(candidate_vmasked_fill_16_ptr) + r420_z, r420_value)
    r421_z = tl.arange($(256))
    r421_value = (tl.load($(candidate_vmasked_fill_16_ptr) + ((((r421_z / $(32)) % $(8)) * $(32)) + (((r421_z / $(8)) % $(4)) * $(8)) + (r421_z % $(8))), dtype=tl.float32) - tl.load($(candidate_vcat_11_ptr) + ((((r421_z / $(8)) % $(4)) * $(8)) + ((r421_z / $(32)) % $(8)) + $(96)), dtype=tl.float32))
    r421_value = r421_value.to(tl.float32)
    r421_value = tl.broadcast_to(r421_value, [$(256)])
    tl.store($(candidate_vsub_34_ptr) + r421_z, r421_value)
    r422_z = tl.arange($(256))
    r422_value = tl.exp(tl.load($(candidate_vsub_34_ptr) + ((((r422_z / $(32)) % $(8)) * $(32)) + (((r422_z / $(8)) % $(4)) * $(8)) + (r422_z % $(8))), dtype=tl.float32))
    r422_value = r422_value.to(tl.float32)
    r422_value = tl.broadcast_to(r422_value, [$(256)])
    tl.store($(candidate_vexp_4_ptr) + r422_z, r422_value)
    r423_z = tl.arange($(4096))
    r423_value = (tl.load($(candidate_vto_copy_124_ptr) + ((((r423_z / $(512)) % $(8)) * $(512)) + (((r423_z / $(128)) % $(4)) * $(128)) + (r423_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_125_ptr) + ((((r423_z / $(512)) % $(8)) * $(512)) + (((r423_z / $(128)) % $(4)) * $(128)) + (r423_z % $(128))), dtype=tl.float32))
    r423_value = r423_value.to(tl.float32)
    r423_value = tl.broadcast_to(r423_value, [$(4096)])
    tl.store($(candidate_vmul_56_ptr) + r423_z, r423_value)
    r424_z = tl.arange($(32))
    r424_lane = tl.arange($(128))
    r424_value = tl.sum(tl.load($(candidate_vmul_56_ptr) + ((((r424_z[:, None] / $(4)) % $(8)) * $(512)) + ((r424_z[:, None] % $(4)) * $(128)) + r424_lane[None, :]), dtype=tl.float32), axis=1)
    r424_value = r424_value.to(tl.float32)
    r424_value = tl.broadcast_to(r424_value, [$(32)])
    tl.store($(candidate_vsum_6_ptr) + r424_z, r424_value)
    r425_row = tl.arange($(4))
    r425_col = tl.arange($(8))
    r425_lane = tl.arange($(128))
    for r425_group in range($(8)) {
      r425_left = tl.load($(candidate_vto_copy_124_ptr) + (((((((r425_group % $(8)) * $(512)) + (r425_row[:, None] * $(128)) + r425_lane[None, :]) / $(512)) % $(8)) * $(512)) + ((((((r425_group % $(8)) * $(512)) + (r425_row[:, None] * $(128)) + r425_lane[None, :]) / $(128)) % $(4)) * $(128)) + ((((r425_group % $(8)) * $(512)) + (r425_row[:, None] * $(128)) + r425_lane[None, :]) % $(128))), dtype=tl.float32)
      r425_right = tl.load($(candidate_vto_copy_123_ptr) + (((((((r425_group % $(8)) * $(1024)) + (r425_lane[:, None] * $(8)) + r425_col[None, :]) / $(1024)) % $(8)) * $(1024)) + (((((r425_group % $(8)) * $(1024)) + (r425_lane[:, None] * $(8)) + r425_col[None, :]) % $(8)) * $(128)) + (((((r425_group % $(8)) * $(1024)) + (r425_lane[:, None] * $(8)) + r425_col[None, :]) / $(8)) % $(128))), dtype=tl.float32)
      r425_value = tl.dot(r425_left, r425_right)
      r425_value = r425_value.to(tl.float32)
      tl.store($(candidate_vbmm_51_ptr) + ((r425_group * $(32)) + (r425_row[:, None] * $(8)) + r425_col[None, :]), r425_value)
    }
    r426_z = tl.arange($(256))
    r426_value = (tl.load($(candidate_vbmm_51_ptr) + ((((r426_z / $(32)) % $(8)) * $(32)) + (((r426_z / $(8)) % $(4)) * $(8)) + (r426_z % $(8))), dtype=tl.float32) - tl.load($(candidate_vsum_6_ptr) + ((((r426_z / $(32)) % $(8)) * $(4)) + ((r426_z / $(8)) % $(4))), dtype=tl.float32))
    r426_value = r426_value.to(tl.float32)
    r426_value = tl.broadcast_to(r426_value, [$(256)])
    tl.store($(candidate_vsub_35_ptr) + r426_z, r426_value)
    r427_z = tl.arange($(256))
    r427_value = (tl.load($(candidate_vexp_4_ptr) + ((((r427_z / $(32)) % $(8)) * $(32)) + (((r427_z / $(8)) % $(4)) * $(8)) + (r427_z % $(8))), dtype=tl.float32) * tl.load($(candidate_vsub_35_ptr) + ((((r427_z / $(32)) % $(8)) * $(32)) + (((r427_z / $(8)) % $(4)) * $(8)) + (r427_z % $(8))), dtype=tl.float32))
    r427_value = r427_value.to(tl.float32)
    r427_value = tl.broadcast_to(r427_value, [$(256)])
    tl.store($(candidate_vmul_57_ptr) + r427_z, r427_value)
    r428_row = tl.arange($(4))
    r428_col = tl.arange($(192))
    r428_lane = tl.arange($(8))
    for r428_group in range($(8)) {
      r428_left = tl.load($(candidate_vmul_57_ptr) + (((((((r428_group % $(8)) * $(32)) + (r428_row[:, None] * $(8)) + r428_lane[None, :]) / $(32)) % $(8)) * $(32)) + ((((((r428_group % $(8)) * $(32)) + (r428_row[:, None] * $(8)) + r428_lane[None, :]) / $(8)) % $(4)) * $(8)) + ((((r428_group % $(8)) * $(32)) + (r428_row[:, None] * $(8)) + r428_lane[None, :]) % $(8))), dtype=tl.float32)
      r428_right = tl.load($(candidate_vto_copy_122_ptr) + (((((((r428_group % $(8)) * $(1536)) + (r428_lane[:, None] * $(192)) + r428_col[None, :]) / $(1536)) % $(8)) * $(1536)) + ((((((r428_group % $(8)) * $(1536)) + (r428_lane[:, None] * $(192)) + r428_col[None, :]) / $(192)) % $(8)) * $(192)) + ((((r428_group % $(8)) * $(1536)) + (r428_lane[:, None] * $(192)) + r428_col[None, :]) % $(192))), dtype=tl.float32)
      r428_value = tl.dot(r428_left, r428_right)
      r428_value = r428_value.to(tl.float32)
      tl.store($(candidate_vbmm_52_ptr) + ((r428_group * $(768)) + (r428_row[:, None] * $(192)) + r428_col[None, :]), r428_value)
    }
    r429_z = tl.arange($(6144))
    r429_value = (tl.load($(candidate_vbmm_52_ptr) + ((((r429_z / $(768)) % $(8)) * $(768)) + (((r429_z / $(192)) % $(4)) * $(192)) + (r429_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r429_value = r429_value.to(tl.float32)
    r429_value = tl.broadcast_to(r429_value, [$(6144)])
    tl.store($(candidate_vmul_58_ptr) + r429_z, r429_value)
    r430_z = tl.arange($(6144))
    r430_value = tl.load($(candidate_vmul_58_ptr) + ((((r430_z / $(192)) % $(8)) * $(768)) + (((r430_z / $(1536)) % $(4)) * $(192)) + (r430_z % $(192))), dtype=tl.float32)
    r430_value = r430_value.to(tl.bfloat16)
    r430_value = tl.broadcast_to(r430_value, [$(6144)])
    tl.store($(candidate_vto_copy_126_ptr) + r430_z, r430_value)
    r431_row = tl.arange($(8))
    r431_col = tl.arange($(192))
    r431_lane = tl.arange($(4))
    for r431_group in range($(8)) {
      r431_left = tl.load($(candidate_vmul_57_ptr) + (((((((r431_group % $(8)) * $(32)) + (r431_row[:, None] * $(4)) + r431_lane[None, :]) / $(32)) % $(8)) * $(32)) + (((((r431_group % $(8)) * $(32)) + (r431_row[:, None] * $(4)) + r431_lane[None, :]) % $(4)) * $(8)) + (((((r431_group % $(8)) * $(32)) + (r431_row[:, None] * $(4)) + r431_lane[None, :]) / $(4)) % $(8))), dtype=tl.float32)
      r431_right = tl.load($(candidate_vto_copy_121_ptr) + (((((((r431_group % $(8)) * $(768)) + (r431_lane[:, None] * $(192)) + r431_col[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r431_group % $(8)) * $(768)) + (r431_lane[:, None] * $(192)) + r431_col[None, :]) / $(192)) % $(4)) * $(192)) + ((((r431_group % $(8)) * $(768)) + (r431_lane[:, None] * $(192)) + r431_col[None, :]) % $(192))), dtype=tl.float32)
      r431_value = tl.dot(r431_left, r431_right)
      r431_value = r431_value.to(tl.float32)
      tl.store($(candidate_vbmm_53_ptr) + ((r431_group * $(1536)) + (r431_row[:, None] * $(192)) + r431_col[None, :]), r431_value)
    }
    r432_z = tl.arange($(12288))
    r432_value = (tl.load($(candidate_vbmm_53_ptr) + ((((r432_z / $(1536)) % $(8)) * $(1536)) + (((r432_z / $(192)) % $(8)) * $(192)) + (r432_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r432_value = r432_value.to(tl.float32)
    r432_value = tl.broadcast_to(r432_value, [$(12288)])
    tl.store($(candidate_vmul_59_ptr) + r432_z, r432_value)
    r433_z = tl.arange($(12288))
    r433_value = tl.load($(candidate_vmul_59_ptr) + ((((r433_z / $(192)) % $(8)) * $(1536)) + (((r433_z / $(1536)) % $(8)) * $(192)) + (r433_z % $(192))), dtype=tl.float32)
    r433_value = r433_value.to(tl.bfloat16)
    r433_value = tl.broadcast_to(r433_value, [$(12288)])
    tl.store($(candidate_vto_copy_127_ptr) + r433_z, r433_value)
    r434_row = tl.arange($(8))
    r434_col = tl.arange($(128))
    r434_lane = tl.arange($(4))
    for r434_group in range($(8)) {
      r434_left = tl.load($(candidate_vexp_4_ptr) + (((((((r434_group % $(8)) * $(32)) + (r434_row[:, None] * $(4)) + r434_lane[None, :]) / $(32)) % $(8)) * $(32)) + (((((r434_group % $(8)) * $(32)) + (r434_row[:, None] * $(4)) + r434_lane[None, :]) % $(4)) * $(8)) + (((((r434_group % $(8)) * $(32)) + (r434_row[:, None] * $(4)) + r434_lane[None, :]) / $(4)) % $(8))), dtype=tl.float32)
      r434_right = tl.load($(candidate_vto_copy_124_ptr) + (((((((r434_group % $(8)) * $(512)) + (r434_lane[:, None] * $(128)) + r434_col[None, :]) / $(512)) % $(8)) * $(512)) + ((((((r434_group % $(8)) * $(512)) + (r434_lane[:, None] * $(128)) + r434_col[None, :]) / $(128)) % $(4)) * $(128)) + ((((r434_group % $(8)) * $(512)) + (r434_lane[:, None] * $(128)) + r434_col[None, :]) % $(128))), dtype=tl.float32)
      r434_value = tl.dot(r434_left, r434_right)
      r434_value = r434_value.to(tl.float32)
      tl.store($(candidate_vbmm_54_ptr) + ((r434_group * $(1024)) + (r434_row[:, None] * $(128)) + r434_col[None, :]), r434_value)
    }
    r435_z = tl.arange($(8192))
    r435_value = tl.load($(candidate_vbmm_54_ptr) + ((((r435_z / $(128)) % $(8)) * $(1024)) + (((r435_z / $(1024)) % $(8)) * $(128)) + (r435_z % $(128))), dtype=tl.float32)
    r435_value = r435_value.to(tl.bfloat16)
    r435_value = tl.broadcast_to(r435_value, [$(8192)])
    tl.store($(candidate_vto_copy_128_ptr) + r435_z, r435_value)
    r436_z = tl.arange($(24576))
    r436_value = tl.where(((r436_z / $(1536)) % $(16)) < $(8), tl.load($(candidate_vto_copy_127_ptr) + ((((((r436_z / $(1536)) % $(16)) - $(0)) % $(8)) * $(1536)) + (((r436_z / $(192)) % $(8)) * $(192)) + (r436_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r436_value = r436_value.to(tl.bfloat16)
    r436_value = tl.broadcast_to(r436_value, [$(24576)])
    tl.store($(candidate_vcat_31_ptr) + r436_z, r436_value)
    r437_z = tl.arange($(16384))
    r437_value = tl.where(((r437_z / $(1024)) % $(16)) < $(8), tl.load($(candidate_vto_copy_128_ptr) + ((((((r437_z / $(1024)) % $(16)) - $(0)) % $(8)) * $(1024)) + (((r437_z / $(128)) % $(8)) * $(128)) + (r437_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r437_value = r437_value.to(tl.bfloat16)
    r437_value = tl.broadcast_to(r437_value, [$(16384)])
    tl.store($(candidate_vcat_32_ptr) + r437_z, r437_value)
    r438_z = tl.arange($(24576))
    r438_value = tl.load($(candidate_vcat_27_ptr) + ((((r438_z / $(192)) % $(16)) * $(1536)) + (((r438_z / $(3072)) % $(8)) * $(192)) + (r438_z % $(192)) + $(24576)), dtype=tl.bfloat16)
    r438_value = r438_value.to(tl.float32)
    r438_value = tl.broadcast_to(r438_value, [$(24576)])
    tl.store($(candidate_vto_copy_129_ptr) + r438_z, r438_value)
    r439_z = tl.arange($(49152))
    r439_value = tl.load($(candidate_vcat_28_ptr) + ((((r439_z / $(192)) % $(32)) * $(1536)) + (((r439_z / $(6144)) % $(8)) * $(192)) + (r439_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r439_value = r439_value.to(tl.float32)
    r439_value = tl.broadcast_to(r439_value, [$(49152)])
    tl.store($(candidate_vto_copy_130_ptr) + r439_z, r439_value)
    r440_z = tl.arange($(32768))
    r440_value = tl.load($(candidate_vbmm_44_ptr) + ((((r440_z / $(128)) % $(32)) * $(1024)) + (((r440_z / $(4096)) % $(8)) * $(128)) + (r440_z % $(128)) + $(65536)), dtype=tl.bfloat16)
    r440_value = r440_value.to(tl.float32)
    r440_value = tl.broadcast_to(r440_value, [$(32768)])
    tl.store($(candidate_vto_copy_131_ptr) + r440_z, r440_value)
    r441_z = tl.arange($(16384))
    r441_value = tl.load($(candidate_vmm_28_ptr) + ((((r441_z / $(128)) % $(16)) * $(1024)) + (((r441_z / $(2048)) % $(8)) * $(128)) + (r441_z % $(128)) + $(81920)), dtype=tl.bfloat16)
    r441_value = r441_value.to(tl.float32)
    r441_value = tl.broadcast_to(r441_value, [$(16384)])
    tl.store($(candidate_vto_copy_132_ptr) + r441_z, r441_value)
    r442_z = tl.arange($(16384))
    r442_value = tl.load($(candidate_vbmm_16_ptr) + ((((r442_z / $(2048)) % $(8)) * $(4096)) + (((r442_z / $(128)) % $(16)) * $(128)) + (r442_z % $(128)) + $(2048)), dtype=tl.bfloat16)
    r442_value = r442_value.to(tl.float32)
    r442_value = tl.broadcast_to(r442_value, [$(16384)])
    tl.store($(candidate_vto_copy_133_ptr) + r442_z, r442_value)
    r443_row = tl.arange($(16))
    r443_col = tl.arange($(32))
    r443_lane = tl.arange($(192))
    for r443_group in range($(8)) {
      r443_left = tl.load($(candidate_vto_copy_129_ptr) + (((((((r443_group % $(8)) * $(3072)) + (r443_row[:, None] * $(192)) + r443_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r443_group % $(8)) * $(3072)) + (r443_row[:, None] * $(192)) + r443_lane[None, :]) / $(192)) % $(16)) * $(192)) + ((((r443_group % $(8)) * $(3072)) + (r443_row[:, None] * $(192)) + r443_lane[None, :]) % $(192))), dtype=tl.float32)
      r443_right = tl.load($(candidate_vto_copy_130_ptr) + (((((((r443_group % $(8)) * $(6144)) + (r443_lane[:, None] * $(32)) + r443_col[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r443_group % $(8)) * $(6144)) + (r443_lane[:, None] * $(32)) + r443_col[None, :]) % $(32)) * $(192)) + (((((r443_group % $(8)) * $(6144)) + (r443_lane[:, None] * $(32)) + r443_col[None, :]) / $(32)) % $(192))), dtype=tl.float32)
      r443_value = tl.dot(r443_left, r443_right)
      r443_value = r443_value.to(tl.float32)
      tl.store($(candidate_vbmm_55_ptr) + ((r443_group * $(512)) + (r443_row[:, None] * $(32)) + r443_col[None, :]), r443_value)
    }
    r444_z = tl.arange($(4096))
    r444_value = (tl.load($(candidate_vbmm_55_ptr) + ((((r444_z / $(512)) % $(8)) * $(512)) + (((r444_z / $(32)) % $(16)) * $(32)) + (r444_z % $(32))), dtype=tl.float32) * 0.07216878364870322)
    r444_value = r444_value.to(tl.float32)
    r444_value = tl.broadcast_to(r444_value, [$(4096)])
    tl.store($(candidate_vmul_60_ptr) + r444_z, r444_value)
    r445_z = tl.arange($(16))
    r445_value = (tl.cast(r445_z, tl.int64) + tl.full([], $(32), dtype=tl.int64))
    r445_value = tl.broadcast_to(r445_value, [$(16)])
    tl.store($(candidate_vadd_38_ptr) + r445_z, r445_value)
    r446_z = tl.arange($(16))
    r446_value = (tl.load($(candidate_vadd_38_ptr) + r446_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r446_value = tl.broadcast_to(r446_value, [$(16)])
    tl.store($(candidate_vsub_36_ptr) + r446_z, r446_value)
    r447_z = tl.arange($(512))
    r447_value = (tl.cast((r447_z % $(32)), tl.int64) > tl.load($(candidate_vsub_36_ptr) + ((r447_z / $(32)) % $(16)), dtype=tl.int64))
    r447_value = tl.broadcast_to(r447_value, [$(512)])
    tl.store($(candidate_vgt_17_ptr) + r447_z, r447_value)
    r448_z = tl.arange($(4096))
    r448_value = tl.where(tl.load($(candidate_vgt_17_ptr) + ((((r448_z / $(32)) % $(16)) * $(32)) + (r448_z % $(32))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_60_ptr) + ((((r448_z / $(512)) % $(8)) * $(512)) + (((r448_z / $(32)) % $(16)) * $(32)) + (r448_z % $(32))), dtype=tl.float32))
    r448_value = r448_value.to(tl.float32)
    r448_value = tl.broadcast_to(r448_value, [$(4096)])
    tl.store($(candidate_vmasked_fill_17_ptr) + r448_z, r448_value)
    r449_z = tl.arange($(4096))
    r449_value = (tl.load($(candidate_vmasked_fill_17_ptr) + ((((r449_z / $(512)) % $(8)) * $(512)) + (((r449_z / $(32)) % $(16)) * $(32)) + (r449_z % $(32))), dtype=tl.float32) - tl.load($(candidate_vcat_11_ptr) + ((((r449_z / $(32)) % $(16)) * $(8)) + ((r449_z / $(512)) % $(8)) + $(128)), dtype=tl.float32))
    r449_value = r449_value.to(tl.float32)
    r449_value = tl.broadcast_to(r449_value, [$(4096)])
    tl.store($(candidate_vsub_37_ptr) + r449_z, r449_value)
    r450_z = tl.arange($(4096))
    r450_value = tl.exp(tl.load($(candidate_vsub_37_ptr) + ((((r450_z / $(512)) % $(8)) * $(512)) + (((r450_z / $(32)) % $(16)) * $(32)) + (r450_z % $(32))), dtype=tl.float32))
    r450_value = r450_value.to(tl.float32)
    r450_value = tl.broadcast_to(r450_value, [$(4096)])
    tl.store($(candidate_vexp_5_ptr) + r450_z, r450_value)
    r451_z = tl.arange($(16384))
    r451_value = (tl.load($(candidate_vto_copy_132_ptr) + ((((r451_z / $(2048)) % $(8)) * $(2048)) + (((r451_z / $(128)) % $(16)) * $(128)) + (r451_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_133_ptr) + ((((r451_z / $(2048)) % $(8)) * $(2048)) + (((r451_z / $(128)) % $(16)) * $(128)) + (r451_z % $(128))), dtype=tl.float32))
    r451_value = r451_value.to(tl.float32)
    r451_value = tl.broadcast_to(r451_value, [$(16384)])
    tl.store($(candidate_vmul_61_ptr) + r451_z, r451_value)
    r452_z = tl.arange($(128))
    r452_lane = tl.arange($(128))
    r452_value = tl.sum(tl.load($(candidate_vmul_61_ptr) + ((((r452_z[:, None] / $(16)) % $(8)) * $(2048)) + ((r452_z[:, None] % $(16)) * $(128)) + r452_lane[None, :]), dtype=tl.float32), axis=1)
    r452_value = r452_value.to(tl.float32)
    r452_value = tl.broadcast_to(r452_value, [$(128)])
    tl.store($(candidate_vsum_7_ptr) + r452_z, r452_value)
    r453_row = tl.arange($(16))
    r453_col = tl.arange($(32))
    r453_lane = tl.arange($(128))
    for r453_group in range($(8)) {
      r453_left = tl.load($(candidate_vto_copy_132_ptr) + (((((((r453_group % $(8)) * $(2048)) + (r453_row[:, None] * $(128)) + r453_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r453_group % $(8)) * $(2048)) + (r453_row[:, None] * $(128)) + r453_lane[None, :]) / $(128)) % $(16)) * $(128)) + ((((r453_group % $(8)) * $(2048)) + (r453_row[:, None] * $(128)) + r453_lane[None, :]) % $(128))), dtype=tl.float32)
      r453_right = tl.load($(candidate_vto_copy_131_ptr) + (((((((r453_group % $(8)) * $(4096)) + (r453_lane[:, None] * $(32)) + r453_col[None, :]) / $(4096)) % $(8)) * $(4096)) + (((((r453_group % $(8)) * $(4096)) + (r453_lane[:, None] * $(32)) + r453_col[None, :]) % $(32)) * $(128)) + (((((r453_group % $(8)) * $(4096)) + (r453_lane[:, None] * $(32)) + r453_col[None, :]) / $(32)) % $(128))), dtype=tl.float32)
      r453_value = tl.dot(r453_left, r453_right)
      r453_value = r453_value.to(tl.float32)
      tl.store($(candidate_vbmm_56_ptr) + ((r453_group * $(512)) + (r453_row[:, None] * $(32)) + r453_col[None, :]), r453_value)
    }
    r454_z = tl.arange($(4096))
    r454_value = (tl.load($(candidate_vbmm_56_ptr) + ((((r454_z / $(512)) % $(8)) * $(512)) + (((r454_z / $(32)) % $(16)) * $(32)) + (r454_z % $(32))), dtype=tl.float32) - tl.load($(candidate_vsum_7_ptr) + ((((r454_z / $(512)) % $(8)) * $(16)) + ((r454_z / $(32)) % $(16))), dtype=tl.float32))
    r454_value = r454_value.to(tl.float32)
    r454_value = tl.broadcast_to(r454_value, [$(4096)])
    tl.store($(candidate_vsub_38_ptr) + r454_z, r454_value)
    r455_z = tl.arange($(4096))
    r455_value = (tl.load($(candidate_vexp_5_ptr) + ((((r455_z / $(512)) % $(8)) * $(512)) + (((r455_z / $(32)) % $(16)) * $(32)) + (r455_z % $(32))), dtype=tl.float32) * tl.load($(candidate_vsub_38_ptr) + ((((r455_z / $(512)) % $(8)) * $(512)) + (((r455_z / $(32)) % $(16)) * $(32)) + (r455_z % $(32))), dtype=tl.float32))
    r455_value = r455_value.to(tl.float32)
    r455_value = tl.broadcast_to(r455_value, [$(4096)])
    tl.store($(candidate_vmul_62_ptr) + r455_z, r455_value)
    r456_row = tl.arange($(16))
    r456_col = tl.arange($(192))
    r456_lane = tl.arange($(32))
    for r456_group in range($(8)) {
      r456_left = tl.load($(candidate_vmul_62_ptr) + (((((((r456_group % $(8)) * $(512)) + (r456_row[:, None] * $(32)) + r456_lane[None, :]) / $(512)) % $(8)) * $(512)) + ((((((r456_group % $(8)) * $(512)) + (r456_row[:, None] * $(32)) + r456_lane[None, :]) / $(32)) % $(16)) * $(32)) + ((((r456_group % $(8)) * $(512)) + (r456_row[:, None] * $(32)) + r456_lane[None, :]) % $(32))), dtype=tl.float32)
      r456_right = tl.load($(candidate_vto_copy_130_ptr) + (((((((r456_group % $(8)) * $(6144)) + (r456_lane[:, None] * $(192)) + r456_col[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r456_group % $(8)) * $(6144)) + (r456_lane[:, None] * $(192)) + r456_col[None, :]) / $(192)) % $(32)) * $(192)) + ((((r456_group % $(8)) * $(6144)) + (r456_lane[:, None] * $(192)) + r456_col[None, :]) % $(192))), dtype=tl.float32)
      r456_value = tl.dot(r456_left, r456_right)
      r456_value = r456_value.to(tl.float32)
      tl.store($(candidate_vbmm_57_ptr) + ((r456_group * $(3072)) + (r456_row[:, None] * $(192)) + r456_col[None, :]), r456_value)
    }
    r457_z = tl.arange($(24576))
    r457_value = (tl.load($(candidate_vbmm_57_ptr) + ((((r457_z / $(3072)) % $(8)) * $(3072)) + (((r457_z / $(192)) % $(16)) * $(192)) + (r457_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r457_value = r457_value.to(tl.float32)
    r457_value = tl.broadcast_to(r457_value, [$(24576)])
    tl.store($(candidate_vmul_63_ptr) + r457_z, r457_value)
    r458_z = tl.arange($(24576))
    r458_value = tl.load($(candidate_vmul_63_ptr) + ((((r458_z / $(192)) % $(8)) * $(3072)) + (((r458_z / $(1536)) % $(16)) * $(192)) + (r458_z % $(192))), dtype=tl.float32)
    r458_value = r458_value.to(tl.bfloat16)
    r458_value = tl.broadcast_to(r458_value, [$(24576)])
    tl.store($(candidate_vto_copy_134_ptr) + r458_z, r458_value)
    r459_row = tl.arange($(32))
    r459_col = tl.arange($(192))
    r459_lane = tl.arange($(16))
    for r459_group in range($(8)) {
      r459_left = tl.load($(candidate_vmul_62_ptr) + (((((((r459_group % $(8)) * $(512)) + (r459_row[:, None] * $(16)) + r459_lane[None, :]) / $(512)) % $(8)) * $(512)) + (((((r459_group % $(8)) * $(512)) + (r459_row[:, None] * $(16)) + r459_lane[None, :]) % $(16)) * $(32)) + (((((r459_group % $(8)) * $(512)) + (r459_row[:, None] * $(16)) + r459_lane[None, :]) / $(16)) % $(32))), dtype=tl.float32)
      r459_right = tl.load($(candidate_vto_copy_129_ptr) + (((((((r459_group % $(8)) * $(3072)) + (r459_lane[:, None] * $(192)) + r459_col[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r459_group % $(8)) * $(3072)) + (r459_lane[:, None] * $(192)) + r459_col[None, :]) / $(192)) % $(16)) * $(192)) + ((((r459_group % $(8)) * $(3072)) + (r459_lane[:, None] * $(192)) + r459_col[None, :]) % $(192))), dtype=tl.float32)
      r459_value = tl.dot(r459_left, r459_right)
      r459_value = r459_value.to(tl.float32)
      tl.store($(candidate_vbmm_58_ptr) + ((r459_group * $(6144)) + (r459_row[:, None] * $(192)) + r459_col[None, :]), r459_value)
    }
    r460_z = tl.arange($(49152))
    r460_value = (tl.load($(candidate_vbmm_58_ptr) + ((((r460_z / $(6144)) % $(8)) * $(6144)) + (((r460_z / $(192)) % $(32)) * $(192)) + (r460_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r460_value = r460_value.to(tl.float32)
    r460_value = tl.broadcast_to(r460_value, [$(49152)])
    tl.store($(candidate_vmul_64_ptr) + r460_z, r460_value)
    r461_z = tl.arange($(49152))
    r461_value = tl.load($(candidate_vmul_64_ptr) + ((((r461_z / $(192)) % $(8)) * $(6144)) + (((r461_z / $(1536)) % $(32)) * $(192)) + (r461_z % $(192))), dtype=tl.float32)
    r461_value = r461_value.to(tl.bfloat16)
    r461_value = tl.broadcast_to(r461_value, [$(49152)])
    tl.store($(candidate_vto_copy_135_ptr) + r461_z, r461_value)
    r462_row = tl.arange($(32))
    r462_col = tl.arange($(128))
    r462_lane = tl.arange($(16))
    for r462_group in range($(8)) {
      r462_left = tl.load($(candidate_vexp_5_ptr) + (((((((r462_group % $(8)) * $(512)) + (r462_row[:, None] * $(16)) + r462_lane[None, :]) / $(512)) % $(8)) * $(512)) + (((((r462_group % $(8)) * $(512)) + (r462_row[:, None] * $(16)) + r462_lane[None, :]) % $(16)) * $(32)) + (((((r462_group % $(8)) * $(512)) + (r462_row[:, None] * $(16)) + r462_lane[None, :]) / $(16)) % $(32))), dtype=tl.float32)
      r462_right = tl.load($(candidate_vto_copy_132_ptr) + (((((((r462_group % $(8)) * $(2048)) + (r462_lane[:, None] * $(128)) + r462_col[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r462_group % $(8)) * $(2048)) + (r462_lane[:, None] * $(128)) + r462_col[None, :]) / $(128)) % $(16)) * $(128)) + ((((r462_group % $(8)) * $(2048)) + (r462_lane[:, None] * $(128)) + r462_col[None, :]) % $(128))), dtype=tl.float32)
      r462_value = tl.dot(r462_left, r462_right)
      r462_value = r462_value.to(tl.float32)
      tl.store($(candidate_vbmm_59_ptr) + ((r462_group * $(4096)) + (r462_row[:, None] * $(128)) + r462_col[None, :]), r462_value)
    }
    r463_z = tl.arange($(32768))
    r463_value = tl.load($(candidate_vbmm_59_ptr) + ((((r463_z / $(128)) % $(8)) * $(4096)) + (((r463_z / $(1024)) % $(32)) * $(128)) + (r463_z % $(128))), dtype=tl.float32)
    r463_value = r463_value.to(tl.bfloat16)
    r463_value = tl.broadcast_to(r463_value, [$(32768)])
    tl.store($(candidate_vto_copy_136_ptr) + r463_z, r463_value)
    r464_z = tl.arange($(98304))
    r464_value = tl.where(((r464_z / $(1536)) % $(64)) < $(32), tl.load($(candidate_vto_copy_135_ptr) + ((((((r464_z / $(1536)) % $(64)) - $(0)) % $(32)) * $(1536)) + (((r464_z / $(192)) % $(8)) * $(192)) + (r464_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r464_value = r464_value.to(tl.bfloat16)
    r464_value = tl.broadcast_to(r464_value, [$(98304)])
    tl.store($(candidate_vcat_33_ptr) + r464_z, r464_value)
    r465_z = tl.arange($(65536))
    r465_value = tl.where(((r465_z / $(1024)) % $(64)) < $(32), tl.load($(candidate_vto_copy_136_ptr) + ((((((r465_z / $(1024)) % $(64)) - $(0)) % $(32)) * $(1024)) + (((r465_z / $(128)) % $(8)) * $(128)) + (r465_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r465_value = r465_value.to(tl.bfloat16)
    r465_value = tl.broadcast_to(r465_value, [$(65536)])
    tl.store($(candidate_vcat_34_ptr) + r465_z, r465_value)
    r466_z = tl.arange($(49152))
    r466_value = tl.where(((r466_z / $(1536)) % $(32)) < $(12), tl.load($(candidate_vto_copy_118_ptr) + ((((((r466_z / $(1536)) % $(32)) - $(0)) % $(12)) * $(1536)) + (((r466_z / $(192)) % $(8)) * $(192)) + (r466_z % $(192))), dtype=tl.bfloat16), tl.where(((r466_z / $(1536)) % $(32)) < $(16), tl.load($(candidate_vto_copy_126_ptr) + ((((((r466_z / $(1536)) % $(32)) - $(12)) % $(4)) * $(1536)) + (((r466_z / $(192)) % $(8)) * $(192)) + (r466_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vto_copy_134_ptr) + ((((((r466_z / $(1536)) % $(32)) - $(16)) % $(16)) * $(1536)) + (((r466_z / $(192)) % $(8)) * $(192)) + (r466_z % $(192))), dtype=tl.bfloat16)))
    r466_value = r466_value.to(tl.bfloat16)
    r466_value = tl.broadcast_to(r466_value, [$(49152)])
    tl.store($(candidate_vcat_35_ptr) + r466_z, r466_value)
    r467_z = tl.arange($(196608))
    r467_value = tl.where(((r467_z / $(1536)) % $(128)) < $(48), tl.load($(candidate_vcat_29_ptr) + ((((((r467_z / $(1536)) % $(128)) - $(0)) % $(48)) * $(1536)) + (((r467_z / $(192)) % $(8)) * $(192)) + (r467_z % $(192))), dtype=tl.bfloat16), tl.where(((r467_z / $(1536)) % $(128)) < $(64), tl.load($(candidate_vcat_31_ptr) + ((((((r467_z / $(1536)) % $(128)) - $(48)) % $(16)) * $(1536)) + (((r467_z / $(192)) % $(8)) * $(192)) + (r467_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vcat_33_ptr) + ((((((r467_z / $(1536)) % $(128)) - $(64)) % $(64)) * $(1536)) + (((r467_z / $(192)) % $(8)) * $(192)) + (r467_z % $(192))), dtype=tl.bfloat16)))
    r467_value = r467_value.to(tl.bfloat16)
    r467_value = tl.broadcast_to(r467_value, [$(196608)])
    tl.store($(candidate_vcat_36_ptr) + r467_z, r467_value)
    r468_z = tl.arange($(131072))
    r468_value = tl.where(((r468_z / $(1024)) % $(128)) < $(48), tl.load($(candidate_vcat_30_ptr) + ((((((r468_z / $(1024)) % $(128)) - $(0)) % $(48)) * $(1024)) + (((r468_z / $(128)) % $(8)) * $(128)) + (r468_z % $(128))), dtype=tl.bfloat16), tl.where(((r468_z / $(1024)) % $(128)) < $(64), tl.load($(candidate_vcat_32_ptr) + ((((((r468_z / $(1024)) % $(128)) - $(48)) % $(16)) * $(1024)) + (((r468_z / $(128)) % $(8)) * $(128)) + (r468_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_34_ptr) + ((((((r468_z / $(1024)) % $(128)) - $(64)) % $(64)) * $(1024)) + (((r468_z / $(128)) % $(8)) * $(128)) + (r468_z % $(128))), dtype=tl.bfloat16)))
    r468_value = r468_value.to(tl.bfloat16)
    r468_value = tl.broadcast_to(r468_value, [$(131072)])
    tl.store($(candidate_vcat_37_ptr) + r468_z, r468_value)
    r469_z = tl.arange($(8192))
    r469_lane = tl.arange($(8))
    r469_value = tl.sum(tl.load($(candidate_vcat_36_ptr) + ((((r469_z[:, None] / $(64)) % $(128)) * $(1536)) + (r469_lane[None, :] * $(192)) + (r469_z[:, None] % $(64)) + $(128)), dtype=tl.bfloat16), axis=1)
    r469_value = r469_value.to(tl.bfloat16)
    r469_value = tl.broadcast_to(r469_value, [$(8192)])
    tl.store($(candidate_vsum_8_ptr) + r469_z, r469_value)
    r470_row = tl.arange($(128))
    r470_col = tl.arange($(512))
    r470_lane = tl.arange($(1024))
    for r470_group in range($(1)) {
      r470_left = tl.load($(candidate_vcat_36_ptr) + ((((((r470_row[:, None] * $(1024)) + r470_lane[None, :]) / $(1024)) % $(128)) * $(1536)) + ((((r470_row[:, None] * $(1024)) + r470_lane[None, :]) % $(8)) * $(192)) + ((((r470_row[:, None] * $(1024)) + r470_lane[None, :]) / $(8)) % $(128))), dtype=tl.bfloat16)
      r470_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r470_lane[:, None] * $(512)) + r470_col[None, :]) / $(512)) % $(8)) * $(131072)) + (((((r470_lane[:, None] * $(512)) + r470_col[None, :]) / $(4096)) % $(128)) * $(512)) + (((r470_lane[:, None] * $(512)) + r470_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r470_value = tl.dot(r470_left, r470_right)
      r470_value = r470_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_60_ptr) + ((r470_group * $(65536)) + (r470_row[:, None] * $(512)) + r470_col[None, :]), r470_value)
    }
    r471_row = tl.arange($(128))
    r471_col = tl.arange($(512))
    r471_lane = tl.arange($(1024))
    for r471_group in range($(1)) {
      r471_left = tl.load($(candidate_vcat_37_ptr) + ((((((r471_row[:, None] * $(1024)) + r471_lane[None, :]) / $(1024)) % $(128)) * $(1024)) + ((((r471_row[:, None] * $(1024)) + r471_lane[None, :]) % $(8)) * $(128)) + ((((r471_row[:, None] * $(1024)) + r471_lane[None, :]) / $(8)) % $(128))), dtype=tl.bfloat16)
      r471_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r471_lane[:, None] * $(512)) + r471_col[None, :]) / $(512)) % $(8)) * $(131072)) + (((((r471_lane[:, None] * $(512)) + r471_col[None, :]) / $(4096)) % $(128)) * $(512)) + (((r471_lane[:, None] * $(512)) + r471_col[None, :]) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r471_value = tl.dot(r471_left, r471_right)
      r471_value = r471_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_61_ptr) + ((r471_group * $(65536)) + (r471_row[:, None] * $(512)) + r471_col[None, :]), r471_value)
    }
    r472_z = tl.arange($(65536))
    r472_value = (tl.load($(candidate_vbmm_60_ptr) + ((((r472_z / $(512)) % $(128)) * $(512)) + (r472_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vbmm_61_ptr) + ((((r472_z / $(512)) % $(128)) * $(512)) + (r472_z % $(512))), dtype=tl.bfloat16))
    r472_value = r472_value.to(tl.bfloat16)
    r472_value = tl.broadcast_to(r472_value, [$(65536)])
    tl.store($(candidate_vadd_39_ptr) + r472_z, r472_value)
    r473_z = tl.arange($(65536))
    r473_value = tl.where(((((r473_z / $(512)) % $(128)) >= $(0)) & (((r473_z / $(512)) % $(128)) < $(128)) & ((((r473_z / $(512)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vadd_39_ptr) + ((((((r473_z / $(512)) % $(128)) - $(0)) % $(128)) * $(512)) + (r473_z % $(512))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r473_value = r473_value.to(tl.bfloat16)
    r473_value = tl.broadcast_to(r473_value, [$(65536)])
    tl.store($(candidate_vslice_backward_4_ptr) + r473_z, r473_value)
    r474_z = tl.arange($(65536))
    r474_value = (tl.load($(candidate_vslice_backward_ptr) + ((((r474_z / $(512)) % $(128)) * $(512)) + (r474_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_4_ptr) + ((((r474_z / $(512)) % $(128)) * $(512)) + (r474_z % $(512))), dtype=tl.bfloat16))
    r474_value = r474_value.to(tl.bfloat16)
    r474_value = tl.broadcast_to(r474_value, [$(65536)])
    tl.store($(candidate_vadd_40_ptr) + r474_z, r474_value)
    r475_z = tl.arange($(8192))
    r475_value = tl.where(((((r475_z / $(64)) % $(128)) >= $(0)) & (((r475_z / $(64)) % $(128)) < $(128)) & ((((r475_z / $(64)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vsum_8_ptr) + ((((((r475_z / $(64)) % $(128)) - $(0)) % $(128)) * $(64)) + (r475_z % $(64))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r475_value = r475_value.to(tl.bfloat16)
    r475_value = tl.broadcast_to(r475_value, [$(8192)])
    tl.store($(candidate_vslice_backward_5_ptr) + r475_z, r475_value)
    r476_z = tl.arange($(8192))
    r476_value = (tl.load($(candidate_vslice_backward_1_ptr) + ((((r476_z / $(64)) % $(128)) * $(64)) + (r476_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_5_ptr) + ((((r476_z / $(64)) % $(128)) * $(64)) + (r476_z % $(64))), dtype=tl.bfloat16))
    r476_value = r476_value.to(tl.bfloat16)
    r476_value = tl.broadcast_to(r476_value, [$(8192)])
    tl.store($(candidate_vadd_41_ptr) + r476_z, r476_value)
    r477_z = tl.arange($(131072))
    r477_value = tl.where(((((r477_z / $(1024)) % $(128)) >= $(64)) & (((r477_z / $(1024)) % $(128)) < $(96)) & ((((r477_z / $(1024)) % $(128)) - $(64)) % $(1) == $(0))), tl.load($(candidate_vcat_35_ptr) + ((((((r477_z / $(1024)) % $(128)) - $(64)) % $(32)) * $(1536)) + (((r477_z / $(128)) % $(8)) * $(192)) + (r477_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r477_value = r477_value.to(tl.bfloat16)
    r477_value = tl.broadcast_to(r477_value, [$(131072)])
    tl.store($(candidate_vslice_backward_6_ptr) + r477_z, r477_value)
    r478_z = tl.arange($(131072))
    r478_value = (tl.load($(candidate_vslice_backward_2_ptr) + ((((r478_z / $(1024)) % $(128)) * $(1024)) + (((r478_z / $(128)) % $(8)) * $(128)) + (r478_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_6_ptr) + ((((r478_z / $(1024)) % $(128)) * $(1024)) + (((r478_z / $(128)) % $(8)) * $(128)) + (r478_z % $(128))), dtype=tl.bfloat16))
    r478_value = r478_value.to(tl.bfloat16)
    r478_value = tl.broadcast_to(r478_value, [$(131072)])
    tl.store($(candidate_vadd_42_ptr) + r478_z, r478_value)
    r479_z = tl.arange($(65536))
    r479_value = tl.where(((((r479_z / $(512)) % $(128)) >= $(64)) & (((r479_z / $(512)) % $(128)) < $(96)) & ((((r479_z / $(512)) % $(128)) - $(64)) % $(1) == $(0))), tl.load($(candidate_vcat_35_ptr) + ((((((r479_z / $(512)) % $(128)) - $(64)) % $(32)) * $(1536)) + (((r479_z / $(64)) % $(8)) * $(192)) + (r479_z % $(64)) + $(128)), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r479_value = r479_value.to(tl.bfloat16)
    r479_value = tl.broadcast_to(r479_value, [$(65536)])
    tl.store($(candidate_vslice_backward_7_ptr) + r479_z, r479_value)
    r480_z = tl.arange($(65536))
    r480_value = (tl.load($(candidate_vslice_backward_3_ptr) + ((((r480_z / $(512)) % $(128)) * $(512)) + (((r480_z / $(64)) % $(8)) * $(64)) + (r480_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_7_ptr) + ((((r480_z / $(512)) % $(128)) * $(512)) + (((r480_z / $(64)) % $(8)) * $(64)) + (r480_z % $(64))), dtype=tl.bfloat16))
    r480_value = r480_value.to(tl.bfloat16)
    r480_value = tl.broadcast_to(r480_value, [$(65536)])
    tl.store($(candidate_vadd_43_ptr) + r480_z, r480_value)
    r481_row = tl.arange($(128))
    r481_col = tl.arange($(1024))
    r481_lane = tl.arange($(512))
    for r481_group in range($(1)) {
      r481_left = tl.load($(candidate_vindex_2_ptr) + ((((((r481_row[:, None] * $(512)) + r481_lane[None, :]) / $(512)) % $(128)) * $(576)) + (((r481_row[:, None] * $(512)) + r481_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r481_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r481_lane[:, None] * $(1024)) + r481_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r481_lane[:, None] * $(1024)) + r481_col[None, :]) % $(128)) * $(512)) + ((((r481_lane[:, None] * $(1024)) + r481_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r481_value = tl.dot(r481_left, r481_right)
      r481_value = r481_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_64_ptr) + ((r481_group * $(131072)) + (r481_row[:, None] * $(1024)) + r481_col[None, :]), r481_value)
    }
    r482_row = tl.arange($(128))
    r482_col = tl.arange($(1024))
    r482_lane = tl.arange($(512))
    for r482_group in range($(1)) {
      r482_left = tl.load($(candidate_vindex_2_ptr) + ((((((r482_row[:, None] * $(512)) + r482_lane[None, :]) / $(512)) % $(128)) * $(576)) + (((r482_row[:, None] * $(512)) + r482_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r482_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r482_lane[:, None] * $(1024)) + r482_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r482_lane[:, None] * $(1024)) + r482_col[None, :]) % $(128)) * $(512)) + ((((r482_lane[:, None] * $(1024)) + r482_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r482_value = tl.dot(r482_left, r482_right)
      r482_value = r482_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_65_ptr) + ((r482_group * $(131072)) + (r482_row[:, None] * $(1024)) + r482_col[None, :]), r482_value)
    }
    r483_z = tl.arange($(49152))
    r483_value = tl.where((r483_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r483_z / $(1536)) % $(32)) * $(1024)) + (((r483_z / $(192)) % $(8)) * $(128)) + ((r483_z % $(192)) - $(0)) + $(32768)) / $(1024)) % $(128)) * $(1536)) + (((((((r483_z / $(1536)) % $(32)) * $(1024)) + (((r483_z / $(192)) % $(8)) * $(128)) + ((r483_z % $(192)) - $(0)) + $(32768)) / $(128)) % $(8)) * $(192)) + (((((r483_z / $(1536)) % $(32)) * $(1024)) + (((r483_z / $(192)) % $(8)) * $(128)) + ((r483_z % $(192)) - $(0)) + $(32768)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vadd_6_ptr) + ((((((((r483_z / $(1536)) % $(32)) * $(512)) + (((r483_z / $(192)) % $(8)) * $(64)) + ((r483_z % $(192)) - $(128)) + $(16384)) / $(512)) % $(128)) * $(512)) + (((((((r483_z / $(1536)) % $(32)) * $(512)) + (((r483_z / $(192)) % $(8)) * $(64)) + ((r483_z % $(192)) - $(128)) + $(16384)) / $(64)) % $(8)) * $(64)) + (((((r483_z / $(1536)) % $(32)) * $(512)) + (((r483_z / $(192)) % $(8)) * $(64)) + ((r483_z % $(192)) - $(128)) + $(16384)) % $(64))), dtype=tl.bfloat16))
    r483_value = r483_value.to(tl.bfloat16)
    r483_value = tl.broadcast_to(r483_value, [$(49152)])
    tl.store($(candidate_vcat_39_ptr) + r483_z, r483_value)
    r484_z = tl.arange($(196608))
    r484_value = tl.where((r484_z % $(192)) < $(128), tl.load($(candidate_vbmm_64_ptr) + ((((((((r484_z / $(1536)) % $(128)) * $(1024)) + (((r484_z / $(192)) % $(8)) * $(128)) + ((r484_z % $(192)) - $(0))) / $(1024)) % $(128)) * $(1024)) + (((((((r484_z / $(1536)) % $(128)) * $(1024)) + (((r484_z / $(192)) % $(8)) * $(128)) + ((r484_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r484_z / $(1536)) % $(128)) * $(1024)) + (((r484_z / $(192)) % $(8)) * $(128)) + ((r484_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vindex_2_ptr) + ((((r484_z / $(1536)) % $(128)) * $(576)) + ((((r484_z % $(192)) - $(128)) + $(512)) % $(576))), dtype=tl.bfloat16))
    r484_value = r484_value.to(tl.bfloat16)
    r484_value = tl.broadcast_to(r484_value, [$(196608)])
    tl.store($(candidate_vcat_40_ptr) + r484_z, r484_value)
    r485_z = tl.arange($(18432))
    r485_value = tl.load($(candidate_vcat_39_ptr) + ((((r485_z / $(192)) % $(12)) * $(1536)) + (((r485_z / $(2304)) % $(8)) * $(192)) + (r485_z % $(192))), dtype=tl.bfloat16)
    r485_value = r485_value.to(tl.float32)
    r485_value = tl.broadcast_to(r485_value, [$(18432)])
    tl.store($(candidate_vto_copy_137_ptr) + r485_z, r485_value)
    r486_z = tl.arange($(73728))
    r486_value = tl.load($(candidate_vcat_40_ptr) + ((((r486_z / $(192)) % $(48)) * $(1536)) + (((r486_z / $(9216)) % $(8)) * $(192)) + (r486_z % $(192))), dtype=tl.bfloat16)
    r486_value = r486_value.to(tl.float32)
    r486_value = tl.broadcast_to(r486_value, [$(73728)])
    tl.store($(candidate_vto_copy_138_ptr) + r486_z, r486_value)
    r487_z = tl.arange($(49152))
    r487_value = tl.load($(candidate_vbmm_65_ptr) + ((((r487_z / $(128)) % $(48)) * $(1024)) + (((r487_z / $(6144)) % $(8)) * $(128)) + (r487_z % $(128))), dtype=tl.bfloat16)
    r487_value = r487_value.to(tl.float32)
    r487_value = tl.broadcast_to(r487_value, [$(49152)])
    tl.store($(candidate_vto_copy_139_ptr) + r487_z, r487_value)
    r488_z = tl.arange($(12288))
    r488_value = tl.load($(candidate_vmm_28_ptr) + ((((r488_z / $(128)) % $(12)) * $(1024)) + (((r488_z / $(1536)) % $(8)) * $(128)) + (r488_z % $(128)) + $(32768)), dtype=tl.bfloat16)
    r488_value = r488_value.to(tl.float32)
    r488_value = tl.broadcast_to(r488_value, [$(12288)])
    tl.store($(candidate_vto_copy_140_ptr) + r488_z, r488_value)
    r489_z = tl.arange($(12288))
    r489_value = tl.load($(candidate_vbmm_11_ptr) + ((((r489_z / $(1536)) % $(8)) * $(4096)) + (((r489_z / $(128)) % $(12)) * $(128)) + (r489_z % $(128))), dtype=tl.bfloat16)
    r489_value = r489_value.to(tl.float32)
    r489_value = tl.broadcast_to(r489_value, [$(12288)])
    tl.store($(candidate_vto_copy_141_ptr) + r489_z, r489_value)
    r490_row = tl.arange($(12))
    r490_col = tl.arange($(48))
    r490_lane = tl.arange($(192))
    for r490_group in range($(8)) {
      r490_left = tl.load($(candidate_vto_copy_137_ptr) + (((((((r490_group % $(8)) * $(2304)) + (r490_row[:, None] * $(192)) + r490_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r490_group % $(8)) * $(2304)) + (r490_row[:, None] * $(192)) + r490_lane[None, :]) / $(192)) % $(12)) * $(192)) + ((((r490_group % $(8)) * $(2304)) + (r490_row[:, None] * $(192)) + r490_lane[None, :]) % $(192))), dtype=tl.float32)
      r490_right = tl.load($(candidate_vto_copy_138_ptr) + (((((((r490_group % $(8)) * $(9216)) + (r490_lane[:, None] * $(48)) + r490_col[None, :]) / $(9216)) % $(8)) * $(9216)) + (((((r490_group % $(8)) * $(9216)) + (r490_lane[:, None] * $(48)) + r490_col[None, :]) % $(48)) * $(192)) + (((((r490_group % $(8)) * $(9216)) + (r490_lane[:, None] * $(48)) + r490_col[None, :]) / $(48)) % $(192))), dtype=tl.float32)
      r490_value = tl.dot(r490_left, r490_right)
      r490_value = r490_value.to(tl.float32)
      tl.store($(candidate_vbmm_66_ptr) + ((r490_group * $(576)) + (r490_row[:, None] * $(48)) + r490_col[None, :]), r490_value)
    }
    r491_z = tl.arange($(4608))
    r491_value = (tl.load($(candidate_vbmm_66_ptr) + ((((r491_z / $(576)) % $(8)) * $(576)) + (((r491_z / $(48)) % $(12)) * $(48)) + (r491_z % $(48))), dtype=tl.float32) * 0.07216878364870322)
    r491_value = r491_value.to(tl.float32)
    r491_value = tl.broadcast_to(r491_value, [$(4608)])
    tl.store($(candidate_vmul_65_ptr) + r491_z, r491_value)
    r492_z = tl.arange($(12))
    r492_value = (tl.cast(r492_z, tl.int64) + tl.full([], $(48), dtype=tl.int64))
    r492_value = tl.broadcast_to(r492_value, [$(12)])
    tl.store($(candidate_vadd_44_ptr) + r492_z, r492_value)
    r493_z = tl.arange($(12))
    r493_value = (tl.load($(candidate_vadd_44_ptr) + r493_z, dtype=tl.int64) - tl.full([], $(12), dtype=tl.int64))
    r493_value = tl.broadcast_to(r493_value, [$(12)])
    tl.store($(candidate_vsub_39_ptr) + r493_z, r493_value)
    r494_z = tl.arange($(576))
    r494_value = (tl.cast((r494_z % $(48)), tl.int64) > tl.load($(candidate_vsub_39_ptr) + ((r494_z / $(48)) % $(12)), dtype=tl.int64))
    r494_value = tl.broadcast_to(r494_value, [$(576)])
    tl.store($(candidate_vgt_18_ptr) + r494_z, r494_value)
    r495_z = tl.arange($(4608))
    r495_value = tl.where(tl.load($(candidate_vgt_18_ptr) + ((((r495_z / $(48)) % $(12)) * $(48)) + (r495_z % $(48))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_65_ptr) + ((((r495_z / $(576)) % $(8)) * $(576)) + (((r495_z / $(48)) % $(12)) * $(48)) + (r495_z % $(48))), dtype=tl.float32))
    r495_value = r495_value.to(tl.float32)
    r495_value = tl.broadcast_to(r495_value, [$(4608)])
    tl.store($(candidate_vmasked_fill_18_ptr) + r495_z, r495_value)
    r496_z = tl.arange($(4608))
    r496_value = (tl.load($(candidate_vmasked_fill_18_ptr) + ((((r496_z / $(576)) % $(8)) * $(576)) + (((r496_z / $(48)) % $(12)) * $(48)) + (r496_z % $(48))), dtype=tl.float32) - tl.load($(candidate_vcat_9_ptr) + ((((r496_z / $(48)) % $(12)) * $(8)) + ((r496_z / $(576)) % $(8))), dtype=tl.float32))
    r496_value = r496_value.to(tl.float32)
    r496_value = tl.broadcast_to(r496_value, [$(4608)])
    tl.store($(candidate_vsub_40_ptr) + r496_z, r496_value)
    r497_z = tl.arange($(4608))
    r497_value = tl.exp(tl.load($(candidate_vsub_40_ptr) + ((((r497_z / $(576)) % $(8)) * $(576)) + (((r497_z / $(48)) % $(12)) * $(48)) + (r497_z % $(48))), dtype=tl.float32))
    r497_value = r497_value.to(tl.float32)
    r497_value = tl.broadcast_to(r497_value, [$(4608)])
    tl.store($(candidate_vexp_6_ptr) + r497_z, r497_value)
    r498_z = tl.arange($(12288))
    r498_value = (tl.load($(candidate_vto_copy_140_ptr) + ((((r498_z / $(1536)) % $(8)) * $(1536)) + (((r498_z / $(128)) % $(12)) * $(128)) + (r498_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_141_ptr) + ((((r498_z / $(1536)) % $(8)) * $(1536)) + (((r498_z / $(128)) % $(12)) * $(128)) + (r498_z % $(128))), dtype=tl.float32))
    r498_value = r498_value.to(tl.float32)
    r498_value = tl.broadcast_to(r498_value, [$(12288)])
    tl.store($(candidate_vmul_66_ptr) + r498_z, r498_value)
    r499_z = tl.arange($(96))
    r499_lane = tl.arange($(128))
    r499_value = tl.sum(tl.load($(candidate_vmul_66_ptr) + ((((r499_z[:, None] / $(12)) % $(8)) * $(1536)) + ((r499_z[:, None] % $(12)) * $(128)) + r499_lane[None, :]), dtype=tl.float32), axis=1)
    r499_value = r499_value.to(tl.float32)
    r499_value = tl.broadcast_to(r499_value, [$(96)])
    tl.store($(candidate_vsum_9_ptr) + r499_z, r499_value)
    r500_row = tl.arange($(12))
    r500_col = tl.arange($(48))
    r500_lane = tl.arange($(128))
    for r500_group in range($(8)) {
      r500_left = tl.load($(candidate_vto_copy_140_ptr) + (((((((r500_group % $(8)) * $(1536)) + (r500_row[:, None] * $(128)) + r500_lane[None, :]) / $(1536)) % $(8)) * $(1536)) + ((((((r500_group % $(8)) * $(1536)) + (r500_row[:, None] * $(128)) + r500_lane[None, :]) / $(128)) % $(12)) * $(128)) + ((((r500_group % $(8)) * $(1536)) + (r500_row[:, None] * $(128)) + r500_lane[None, :]) % $(128))), dtype=tl.float32)
      r500_right = tl.load($(candidate_vto_copy_139_ptr) + (((((((r500_group % $(8)) * $(6144)) + (r500_lane[:, None] * $(48)) + r500_col[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r500_group % $(8)) * $(6144)) + (r500_lane[:, None] * $(48)) + r500_col[None, :]) % $(48)) * $(128)) + (((((r500_group % $(8)) * $(6144)) + (r500_lane[:, None] * $(48)) + r500_col[None, :]) / $(48)) % $(128))), dtype=tl.float32)
      r500_value = tl.dot(r500_left, r500_right)
      r500_value = r500_value.to(tl.float32)
      tl.store($(candidate_vbmm_67_ptr) + ((r500_group * $(576)) + (r500_row[:, None] * $(48)) + r500_col[None, :]), r500_value)
    }
    r501_z = tl.arange($(4608))
    r501_value = (tl.load($(candidate_vbmm_67_ptr) + ((((r501_z / $(576)) % $(8)) * $(576)) + (((r501_z / $(48)) % $(12)) * $(48)) + (r501_z % $(48))), dtype=tl.float32) - tl.load($(candidate_vsum_9_ptr) + ((((r501_z / $(576)) % $(8)) * $(12)) + ((r501_z / $(48)) % $(12))), dtype=tl.float32))
    r501_value = r501_value.to(tl.float32)
    r501_value = tl.broadcast_to(r501_value, [$(4608)])
    tl.store($(candidate_vsub_41_ptr) + r501_z, r501_value)
    r502_z = tl.arange($(4608))
    r502_value = (tl.load($(candidate_vexp_6_ptr) + ((((r502_z / $(576)) % $(8)) * $(576)) + (((r502_z / $(48)) % $(12)) * $(48)) + (r502_z % $(48))), dtype=tl.float32) * tl.load($(candidate_vsub_41_ptr) + ((((r502_z / $(576)) % $(8)) * $(576)) + (((r502_z / $(48)) % $(12)) * $(48)) + (r502_z % $(48))), dtype=tl.float32))
    r502_value = r502_value.to(tl.float32)
    r502_value = tl.broadcast_to(r502_value, [$(4608)])
    tl.store($(candidate_vmul_67_ptr) + r502_z, r502_value)
    r503_row = tl.arange($(12))
    r503_col = tl.arange($(192))
    r503_lane = tl.arange($(48))
    for r503_group in range($(8)) {
      r503_left = tl.load($(candidate_vmul_67_ptr) + (((((((r503_group % $(8)) * $(576)) + (r503_row[:, None] * $(48)) + r503_lane[None, :]) / $(576)) % $(8)) * $(576)) + ((((((r503_group % $(8)) * $(576)) + (r503_row[:, None] * $(48)) + r503_lane[None, :]) / $(48)) % $(12)) * $(48)) + ((((r503_group % $(8)) * $(576)) + (r503_row[:, None] * $(48)) + r503_lane[None, :]) % $(48))), dtype=tl.float32)
      r503_right = tl.load($(candidate_vto_copy_138_ptr) + (((((((r503_group % $(8)) * $(9216)) + (r503_lane[:, None] * $(192)) + r503_col[None, :]) / $(9216)) % $(8)) * $(9216)) + ((((((r503_group % $(8)) * $(9216)) + (r503_lane[:, None] * $(192)) + r503_col[None, :]) / $(192)) % $(48)) * $(192)) + ((((r503_group % $(8)) * $(9216)) + (r503_lane[:, None] * $(192)) + r503_col[None, :]) % $(192))), dtype=tl.float32)
      r503_value = tl.dot(r503_left, r503_right)
      r503_value = r503_value.to(tl.float32)
      tl.store($(candidate_vbmm_68_ptr) + ((r503_group * $(2304)) + (r503_row[:, None] * $(192)) + r503_col[None, :]), r503_value)
    }
    r504_z = tl.arange($(18432))
    r504_value = (tl.load($(candidate_vbmm_68_ptr) + ((((r504_z / $(2304)) % $(8)) * $(2304)) + (((r504_z / $(192)) % $(12)) * $(192)) + (r504_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r504_value = r504_value.to(tl.float32)
    r504_value = tl.broadcast_to(r504_value, [$(18432)])
    tl.store($(candidate_vmul_68_ptr) + r504_z, r504_value)
    r505_z = tl.arange($(18432))
    r505_value = tl.load($(candidate_vmul_68_ptr) + ((((r505_z / $(192)) % $(8)) * $(2304)) + (((r505_z / $(1536)) % $(12)) * $(192)) + (r505_z % $(192))), dtype=tl.float32)
    r505_value = r505_value.to(tl.bfloat16)
    r505_value = tl.broadcast_to(r505_value, [$(18432)])
    tl.store($(candidate_vto_copy_142_ptr) + r505_z, r505_value)
    r506_row = tl.arange($(48))
    r506_col = tl.arange($(192))
    r506_lane = tl.arange($(12))
    for r506_group in range($(8)) {
      r506_left = tl.load($(candidate_vmul_67_ptr) + (((((((r506_group % $(8)) * $(576)) + (r506_row[:, None] * $(12)) + r506_lane[None, :]) / $(576)) % $(8)) * $(576)) + (((((r506_group % $(8)) * $(576)) + (r506_row[:, None] * $(12)) + r506_lane[None, :]) % $(12)) * $(48)) + (((((r506_group % $(8)) * $(576)) + (r506_row[:, None] * $(12)) + r506_lane[None, :]) / $(12)) % $(48))), dtype=tl.float32)
      r506_right = tl.load($(candidate_vto_copy_137_ptr) + (((((((r506_group % $(8)) * $(2304)) + (r506_lane[:, None] * $(192)) + r506_col[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r506_group % $(8)) * $(2304)) + (r506_lane[:, None] * $(192)) + r506_col[None, :]) / $(192)) % $(12)) * $(192)) + ((((r506_group % $(8)) * $(2304)) + (r506_lane[:, None] * $(192)) + r506_col[None, :]) % $(192))), dtype=tl.float32)
      r506_value = tl.dot(r506_left, r506_right)
      r506_value = r506_value.to(tl.float32)
      tl.store($(candidate_vbmm_69_ptr) + ((r506_group * $(9216)) + (r506_row[:, None] * $(192)) + r506_col[None, :]), r506_value)
    }
    r507_z = tl.arange($(73728))
    r507_value = (tl.load($(candidate_vbmm_69_ptr) + ((((r507_z / $(9216)) % $(8)) * $(9216)) + (((r507_z / $(192)) % $(48)) * $(192)) + (r507_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r507_value = r507_value.to(tl.float32)
    r507_value = tl.broadcast_to(r507_value, [$(73728)])
    tl.store($(candidate_vmul_69_ptr) + r507_z, r507_value)
    r508_z = tl.arange($(73728))
    r508_value = tl.load($(candidate_vmul_69_ptr) + ((((r508_z / $(192)) % $(8)) * $(9216)) + (((r508_z / $(1536)) % $(48)) * $(192)) + (r508_z % $(192))), dtype=tl.float32)
    r508_value = r508_value.to(tl.bfloat16)
    r508_value = tl.broadcast_to(r508_value, [$(73728)])
    tl.store($(candidate_vto_copy_143_ptr) + r508_z, r508_value)
    r509_row = tl.arange($(48))
    r509_col = tl.arange($(128))
    r509_lane = tl.arange($(12))
    for r509_group in range($(8)) {
      r509_left = tl.load($(candidate_vexp_6_ptr) + (((((((r509_group % $(8)) * $(576)) + (r509_row[:, None] * $(12)) + r509_lane[None, :]) / $(576)) % $(8)) * $(576)) + (((((r509_group % $(8)) * $(576)) + (r509_row[:, None] * $(12)) + r509_lane[None, :]) % $(12)) * $(48)) + (((((r509_group % $(8)) * $(576)) + (r509_row[:, None] * $(12)) + r509_lane[None, :]) / $(12)) % $(48))), dtype=tl.float32)
      r509_right = tl.load($(candidate_vto_copy_140_ptr) + (((((((r509_group % $(8)) * $(1536)) + (r509_lane[:, None] * $(128)) + r509_col[None, :]) / $(1536)) % $(8)) * $(1536)) + ((((((r509_group % $(8)) * $(1536)) + (r509_lane[:, None] * $(128)) + r509_col[None, :]) / $(128)) % $(12)) * $(128)) + ((((r509_group % $(8)) * $(1536)) + (r509_lane[:, None] * $(128)) + r509_col[None, :]) % $(128))), dtype=tl.float32)
      r509_value = tl.dot(r509_left, r509_right)
      r509_value = r509_value.to(tl.float32)
      tl.store($(candidate_vbmm_70_ptr) + ((r509_group * $(6144)) + (r509_row[:, None] * $(128)) + r509_col[None, :]), r509_value)
    }
    r510_z = tl.arange($(49152))
    r510_value = tl.load($(candidate_vbmm_70_ptr) + ((((r510_z / $(128)) % $(8)) * $(6144)) + (((r510_z / $(1024)) % $(48)) * $(128)) + (r510_z % $(128))), dtype=tl.float32)
    r510_value = r510_value.to(tl.bfloat16)
    r510_value = tl.broadcast_to(r510_value, [$(49152)])
    tl.store($(candidate_vto_copy_144_ptr) + r510_z, r510_value)
    r511_z = tl.arange($(6144))
    r511_value = tl.load($(candidate_vcat_39_ptr) + ((((r511_z / $(192)) % $(4)) * $(1536)) + (((r511_z / $(768)) % $(8)) * $(192)) + (r511_z % $(192)) + $(18432)), dtype=tl.bfloat16)
    r511_value = r511_value.to(tl.float32)
    r511_value = tl.broadcast_to(r511_value, [$(6144)])
    tl.store($(candidate_vto_copy_145_ptr) + r511_z, r511_value)
    r512_z = tl.arange($(24576))
    r512_value = tl.load($(candidate_vcat_40_ptr) + ((((r512_z / $(192)) % $(16)) * $(1536)) + (((r512_z / $(3072)) % $(8)) * $(192)) + (r512_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r512_value = r512_value.to(tl.float32)
    r512_value = tl.broadcast_to(r512_value, [$(24576)])
    tl.store($(candidate_vto_copy_146_ptr) + r512_z, r512_value)
    r513_z = tl.arange($(16384))
    r513_value = tl.load($(candidate_vbmm_65_ptr) + ((((r513_z / $(128)) % $(16)) * $(1024)) + (((r513_z / $(2048)) % $(8)) * $(128)) + (r513_z % $(128)) + $(49152)), dtype=tl.bfloat16)
    r513_value = r513_value.to(tl.float32)
    r513_value = tl.broadcast_to(r513_value, [$(16384)])
    tl.store($(candidate_vto_copy_147_ptr) + r513_z, r513_value)
    r514_z = tl.arange($(4096))
    r514_value = tl.load($(candidate_vmm_28_ptr) + ((((r514_z / $(128)) % $(4)) * $(1024)) + (((r514_z / $(512)) % $(8)) * $(128)) + (r514_z % $(128)) + $(45056)), dtype=tl.bfloat16)
    r514_value = r514_value.to(tl.float32)
    r514_value = tl.broadcast_to(r514_value, [$(4096)])
    tl.store($(candidate_vto_copy_148_ptr) + r514_z, r514_value)
    r515_z = tl.arange($(4096))
    r515_value = tl.load($(candidate_vbmm_11_ptr) + ((((r515_z / $(512)) % $(8)) * $(4096)) + (((r515_z / $(128)) % $(4)) * $(128)) + (r515_z % $(128)) + $(1536)), dtype=tl.bfloat16)
    r515_value = r515_value.to(tl.float32)
    r515_value = tl.broadcast_to(r515_value, [$(4096)])
    tl.store($(candidate_vto_copy_149_ptr) + r515_z, r515_value)
    r516_row = tl.arange($(4))
    r516_col = tl.arange($(16))
    r516_lane = tl.arange($(192))
    for r516_group in range($(8)) {
      r516_left = tl.load($(candidate_vto_copy_145_ptr) + (((((((r516_group % $(8)) * $(768)) + (r516_row[:, None] * $(192)) + r516_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r516_group % $(8)) * $(768)) + (r516_row[:, None] * $(192)) + r516_lane[None, :]) / $(192)) % $(4)) * $(192)) + ((((r516_group % $(8)) * $(768)) + (r516_row[:, None] * $(192)) + r516_lane[None, :]) % $(192))), dtype=tl.float32)
      r516_right = tl.load($(candidate_vto_copy_146_ptr) + (((((((r516_group % $(8)) * $(3072)) + (r516_lane[:, None] * $(16)) + r516_col[None, :]) / $(3072)) % $(8)) * $(3072)) + (((((r516_group % $(8)) * $(3072)) + (r516_lane[:, None] * $(16)) + r516_col[None, :]) % $(16)) * $(192)) + (((((r516_group % $(8)) * $(3072)) + (r516_lane[:, None] * $(16)) + r516_col[None, :]) / $(16)) % $(192))), dtype=tl.float32)
      r516_value = tl.dot(r516_left, r516_right)
      r516_value = r516_value.to(tl.float32)
      tl.store($(candidate_vbmm_71_ptr) + ((r516_group * $(64)) + (r516_row[:, None] * $(16)) + r516_col[None, :]), r516_value)
    }
    r517_z = tl.arange($(512))
    r517_value = (tl.load($(candidate_vbmm_71_ptr) + ((((r517_z / $(64)) % $(8)) * $(64)) + (((r517_z / $(16)) % $(4)) * $(16)) + (r517_z % $(16))), dtype=tl.float32) * 0.07216878364870322)
    r517_value = r517_value.to(tl.float32)
    r517_value = tl.broadcast_to(r517_value, [$(512)])
    tl.store($(candidate_vmul_70_ptr) + r517_z, r517_value)
    r518_z = tl.arange($(4))
    r518_value = (tl.cast(r518_z, tl.int64) + tl.full([], $(16), dtype=tl.int64))
    r518_value = tl.broadcast_to(r518_value, [$(4)])
    tl.store($(candidate_vadd_45_ptr) + r518_z, r518_value)
    r519_z = tl.arange($(4))
    r519_value = (tl.load($(candidate_vadd_45_ptr) + r519_z, dtype=tl.int64) - tl.full([], $(4), dtype=tl.int64))
    r519_value = tl.broadcast_to(r519_value, [$(4)])
    tl.store($(candidate_vsub_42_ptr) + r519_z, r519_value)
    r520_z = tl.arange($(64))
    r520_value = (tl.cast((r520_z % $(16)), tl.int64) > tl.load($(candidate_vsub_42_ptr) + ((r520_z / $(16)) % $(4)), dtype=tl.int64))
    r520_value = tl.broadcast_to(r520_value, [$(64)])
    tl.store($(candidate_vgt_19_ptr) + r520_z, r520_value)
    r521_z = tl.arange($(512))
    r521_value = tl.where(tl.load($(candidate_vgt_19_ptr) + ((((r521_z / $(16)) % $(4)) * $(16)) + (r521_z % $(16))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_70_ptr) + ((((r521_z / $(64)) % $(8)) * $(64)) + (((r521_z / $(16)) % $(4)) * $(16)) + (r521_z % $(16))), dtype=tl.float32))
    r521_value = r521_value.to(tl.float32)
    r521_value = tl.broadcast_to(r521_value, [$(512)])
    tl.store($(candidate_vmasked_fill_19_ptr) + r521_z, r521_value)
    r522_z = tl.arange($(512))
    r522_value = (tl.load($(candidate_vmasked_fill_19_ptr) + ((((r522_z / $(64)) % $(8)) * $(64)) + (((r522_z / $(16)) % $(4)) * $(16)) + (r522_z % $(16))), dtype=tl.float32) - tl.load($(candidate_vcat_9_ptr) + ((((r522_z / $(16)) % $(4)) * $(8)) + ((r522_z / $(64)) % $(8)) + $(96)), dtype=tl.float32))
    r522_value = r522_value.to(tl.float32)
    r522_value = tl.broadcast_to(r522_value, [$(512)])
    tl.store($(candidate_vsub_43_ptr) + r522_z, r522_value)
    r523_z = tl.arange($(512))
    r523_value = tl.exp(tl.load($(candidate_vsub_43_ptr) + ((((r523_z / $(64)) % $(8)) * $(64)) + (((r523_z / $(16)) % $(4)) * $(16)) + (r523_z % $(16))), dtype=tl.float32))
    r523_value = r523_value.to(tl.float32)
    r523_value = tl.broadcast_to(r523_value, [$(512)])
    tl.store($(candidate_vexp_7_ptr) + r523_z, r523_value)
    r524_z = tl.arange($(4096))
    r524_value = (tl.load($(candidate_vto_copy_148_ptr) + ((((r524_z / $(512)) % $(8)) * $(512)) + (((r524_z / $(128)) % $(4)) * $(128)) + (r524_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_149_ptr) + ((((r524_z / $(512)) % $(8)) * $(512)) + (((r524_z / $(128)) % $(4)) * $(128)) + (r524_z % $(128))), dtype=tl.float32))
    r524_value = r524_value.to(tl.float32)
    r524_value = tl.broadcast_to(r524_value, [$(4096)])
    tl.store($(candidate_vmul_71_ptr) + r524_z, r524_value)
    r525_z = tl.arange($(32))
    r525_lane = tl.arange($(128))
    r525_value = tl.sum(tl.load($(candidate_vmul_71_ptr) + ((((r525_z[:, None] / $(4)) % $(8)) * $(512)) + ((r525_z[:, None] % $(4)) * $(128)) + r525_lane[None, :]), dtype=tl.float32), axis=1)
    r525_value = r525_value.to(tl.float32)
    r525_value = tl.broadcast_to(r525_value, [$(32)])
    tl.store($(candidate_vsum_10_ptr) + r525_z, r525_value)
    r526_row = tl.arange($(4))
    r526_col = tl.arange($(16))
    r526_lane = tl.arange($(128))
    for r526_group in range($(8)) {
      r526_left = tl.load($(candidate_vto_copy_148_ptr) + (((((((r526_group % $(8)) * $(512)) + (r526_row[:, None] * $(128)) + r526_lane[None, :]) / $(512)) % $(8)) * $(512)) + ((((((r526_group % $(8)) * $(512)) + (r526_row[:, None] * $(128)) + r526_lane[None, :]) / $(128)) % $(4)) * $(128)) + ((((r526_group % $(8)) * $(512)) + (r526_row[:, None] * $(128)) + r526_lane[None, :]) % $(128))), dtype=tl.float32)
      r526_right = tl.load($(candidate_vto_copy_147_ptr) + (((((((r526_group % $(8)) * $(2048)) + (r526_lane[:, None] * $(16)) + r526_col[None, :]) / $(2048)) % $(8)) * $(2048)) + (((((r526_group % $(8)) * $(2048)) + (r526_lane[:, None] * $(16)) + r526_col[None, :]) % $(16)) * $(128)) + (((((r526_group % $(8)) * $(2048)) + (r526_lane[:, None] * $(16)) + r526_col[None, :]) / $(16)) % $(128))), dtype=tl.float32)
      r526_value = tl.dot(r526_left, r526_right)
      r526_value = r526_value.to(tl.float32)
      tl.store($(candidate_vbmm_72_ptr) + ((r526_group * $(64)) + (r526_row[:, None] * $(16)) + r526_col[None, :]), r526_value)
    }
    r527_z = tl.arange($(512))
    r527_value = (tl.load($(candidate_vbmm_72_ptr) + ((((r527_z / $(64)) % $(8)) * $(64)) + (((r527_z / $(16)) % $(4)) * $(16)) + (r527_z % $(16))), dtype=tl.float32) - tl.load($(candidate_vsum_10_ptr) + ((((r527_z / $(64)) % $(8)) * $(4)) + ((r527_z / $(16)) % $(4))), dtype=tl.float32))
    r527_value = r527_value.to(tl.float32)
    r527_value = tl.broadcast_to(r527_value, [$(512)])
    tl.store($(candidate_vsub_44_ptr) + r527_z, r527_value)
    r528_z = tl.arange($(512))
    r528_value = (tl.load($(candidate_vexp_7_ptr) + ((((r528_z / $(64)) % $(8)) * $(64)) + (((r528_z / $(16)) % $(4)) * $(16)) + (r528_z % $(16))), dtype=tl.float32) * tl.load($(candidate_vsub_44_ptr) + ((((r528_z / $(64)) % $(8)) * $(64)) + (((r528_z / $(16)) % $(4)) * $(16)) + (r528_z % $(16))), dtype=tl.float32))
    r528_value = r528_value.to(tl.float32)
    r528_value = tl.broadcast_to(r528_value, [$(512)])
    tl.store($(candidate_vmul_72_ptr) + r528_z, r528_value)
    r529_row = tl.arange($(4))
    r529_col = tl.arange($(192))
    r529_lane = tl.arange($(16))
    for r529_group in range($(8)) {
      r529_left = tl.load($(candidate_vmul_72_ptr) + (((((((r529_group % $(8)) * $(64)) + (r529_row[:, None] * $(16)) + r529_lane[None, :]) / $(64)) % $(8)) * $(64)) + ((((((r529_group % $(8)) * $(64)) + (r529_row[:, None] * $(16)) + r529_lane[None, :]) / $(16)) % $(4)) * $(16)) + ((((r529_group % $(8)) * $(64)) + (r529_row[:, None] * $(16)) + r529_lane[None, :]) % $(16))), dtype=tl.float32)
      r529_right = tl.load($(candidate_vto_copy_146_ptr) + (((((((r529_group % $(8)) * $(3072)) + (r529_lane[:, None] * $(192)) + r529_col[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r529_group % $(8)) * $(3072)) + (r529_lane[:, None] * $(192)) + r529_col[None, :]) / $(192)) % $(16)) * $(192)) + ((((r529_group % $(8)) * $(3072)) + (r529_lane[:, None] * $(192)) + r529_col[None, :]) % $(192))), dtype=tl.float32)
      r529_value = tl.dot(r529_left, r529_right)
      r529_value = r529_value.to(tl.float32)
      tl.store($(candidate_vbmm_73_ptr) + ((r529_group * $(768)) + (r529_row[:, None] * $(192)) + r529_col[None, :]), r529_value)
    }
    r530_z = tl.arange($(6144))
    r530_value = (tl.load($(candidate_vbmm_73_ptr) + ((((r530_z / $(768)) % $(8)) * $(768)) + (((r530_z / $(192)) % $(4)) * $(192)) + (r530_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r530_value = r530_value.to(tl.float32)
    r530_value = tl.broadcast_to(r530_value, [$(6144)])
    tl.store($(candidate_vmul_73_ptr) + r530_z, r530_value)
    r531_z = tl.arange($(6144))
    r531_value = tl.load($(candidate_vmul_73_ptr) + ((((r531_z / $(192)) % $(8)) * $(768)) + (((r531_z / $(1536)) % $(4)) * $(192)) + (r531_z % $(192))), dtype=tl.float32)
    r531_value = r531_value.to(tl.bfloat16)
    r531_value = tl.broadcast_to(r531_value, [$(6144)])
    tl.store($(candidate_vto_copy_150_ptr) + r531_z, r531_value)
    r532_row = tl.arange($(16))
    r532_col = tl.arange($(192))
    r532_lane = tl.arange($(4))
    for r532_group in range($(8)) {
      r532_left = tl.load($(candidate_vmul_72_ptr) + (((((((r532_group % $(8)) * $(64)) + (r532_row[:, None] * $(4)) + r532_lane[None, :]) / $(64)) % $(8)) * $(64)) + (((((r532_group % $(8)) * $(64)) + (r532_row[:, None] * $(4)) + r532_lane[None, :]) % $(4)) * $(16)) + (((((r532_group % $(8)) * $(64)) + (r532_row[:, None] * $(4)) + r532_lane[None, :]) / $(4)) % $(16))), dtype=tl.float32)
      r532_right = tl.load($(candidate_vto_copy_145_ptr) + (((((((r532_group % $(8)) * $(768)) + (r532_lane[:, None] * $(192)) + r532_col[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r532_group % $(8)) * $(768)) + (r532_lane[:, None] * $(192)) + r532_col[None, :]) / $(192)) % $(4)) * $(192)) + ((((r532_group % $(8)) * $(768)) + (r532_lane[:, None] * $(192)) + r532_col[None, :]) % $(192))), dtype=tl.float32)
      r532_value = tl.dot(r532_left, r532_right)
      r532_value = r532_value.to(tl.float32)
      tl.store($(candidate_vbmm_74_ptr) + ((r532_group * $(3072)) + (r532_row[:, None] * $(192)) + r532_col[None, :]), r532_value)
    }
    r533_z = tl.arange($(24576))
    r533_value = (tl.load($(candidate_vbmm_74_ptr) + ((((r533_z / $(3072)) % $(8)) * $(3072)) + (((r533_z / $(192)) % $(16)) * $(192)) + (r533_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r533_value = r533_value.to(tl.float32)
    r533_value = tl.broadcast_to(r533_value, [$(24576)])
    tl.store($(candidate_vmul_74_ptr) + r533_z, r533_value)
    r534_z = tl.arange($(24576))
    r534_value = tl.load($(candidate_vmul_74_ptr) + ((((r534_z / $(192)) % $(8)) * $(3072)) + (((r534_z / $(1536)) % $(16)) * $(192)) + (r534_z % $(192))), dtype=tl.float32)
    r534_value = r534_value.to(tl.bfloat16)
    r534_value = tl.broadcast_to(r534_value, [$(24576)])
    tl.store($(candidate_vto_copy_151_ptr) + r534_z, r534_value)
    r535_row = tl.arange($(16))
    r535_col = tl.arange($(128))
    r535_lane = tl.arange($(4))
    for r535_group in range($(8)) {
      r535_left = tl.load($(candidate_vexp_7_ptr) + (((((((r535_group % $(8)) * $(64)) + (r535_row[:, None] * $(4)) + r535_lane[None, :]) / $(64)) % $(8)) * $(64)) + (((((r535_group % $(8)) * $(64)) + (r535_row[:, None] * $(4)) + r535_lane[None, :]) % $(4)) * $(16)) + (((((r535_group % $(8)) * $(64)) + (r535_row[:, None] * $(4)) + r535_lane[None, :]) / $(4)) % $(16))), dtype=tl.float32)
      r535_right = tl.load($(candidate_vto_copy_148_ptr) + (((((((r535_group % $(8)) * $(512)) + (r535_lane[:, None] * $(128)) + r535_col[None, :]) / $(512)) % $(8)) * $(512)) + ((((((r535_group % $(8)) * $(512)) + (r535_lane[:, None] * $(128)) + r535_col[None, :]) / $(128)) % $(4)) * $(128)) + ((((r535_group % $(8)) * $(512)) + (r535_lane[:, None] * $(128)) + r535_col[None, :]) % $(128))), dtype=tl.float32)
      r535_value = tl.dot(r535_left, r535_right)
      r535_value = r535_value.to(tl.float32)
      tl.store($(candidate_vbmm_75_ptr) + ((r535_group * $(2048)) + (r535_row[:, None] * $(128)) + r535_col[None, :]), r535_value)
    }
    r536_z = tl.arange($(16384))
    r536_value = tl.load($(candidate_vbmm_75_ptr) + ((((r536_z / $(128)) % $(8)) * $(2048)) + (((r536_z / $(1024)) % $(16)) * $(128)) + (r536_z % $(128))), dtype=tl.float32)
    r536_value = r536_value.to(tl.bfloat16)
    r536_value = tl.broadcast_to(r536_value, [$(16384)])
    tl.store($(candidate_vto_copy_152_ptr) + r536_z, r536_value)
    r537_z = tl.arange($(24576))
    r537_value = tl.load($(candidate_vcat_39_ptr) + ((((r537_z / $(192)) % $(16)) * $(1536)) + (((r537_z / $(3072)) % $(8)) * $(192)) + (r537_z % $(192)) + $(24576)), dtype=tl.bfloat16)
    r537_value = r537_value.to(tl.float32)
    r537_value = tl.broadcast_to(r537_value, [$(24576)])
    tl.store($(candidate_vto_copy_153_ptr) + r537_z, r537_value)
    r538_z = tl.arange($(98304))
    r538_value = tl.load($(candidate_vcat_40_ptr) + ((((r538_z / $(192)) % $(64)) * $(1536)) + (((r538_z / $(12288)) % $(8)) * $(192)) + (r538_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r538_value = r538_value.to(tl.float32)
    r538_value = tl.broadcast_to(r538_value, [$(98304)])
    tl.store($(candidate_vto_copy_154_ptr) + r538_z, r538_value)
    r539_z = tl.arange($(65536))
    r539_value = tl.load($(candidate_vbmm_65_ptr) + ((((r539_z / $(128)) % $(64)) * $(1024)) + (((r539_z / $(8192)) % $(8)) * $(128)) + (r539_z % $(128)) + $(65536)), dtype=tl.bfloat16)
    r539_value = r539_value.to(tl.float32)
    r539_value = tl.broadcast_to(r539_value, [$(65536)])
    tl.store($(candidate_vto_copy_155_ptr) + r539_z, r539_value)
    r540_z = tl.arange($(16384))
    r540_value = tl.load($(candidate_vmm_28_ptr) + ((((r540_z / $(128)) % $(16)) * $(1024)) + (((r540_z / $(2048)) % $(8)) * $(128)) + (r540_z % $(128)) + $(49152)), dtype=tl.bfloat16)
    r540_value = r540_value.to(tl.float32)
    r540_value = tl.broadcast_to(r540_value, [$(16384)])
    tl.store($(candidate_vto_copy_156_ptr) + r540_z, r540_value)
    r541_z = tl.arange($(16384))
    r541_value = tl.load($(candidate_vbmm_11_ptr) + ((((r541_z / $(2048)) % $(8)) * $(4096)) + (((r541_z / $(128)) % $(16)) * $(128)) + (r541_z % $(128)) + $(2048)), dtype=tl.bfloat16)
    r541_value = r541_value.to(tl.float32)
    r541_value = tl.broadcast_to(r541_value, [$(16384)])
    tl.store($(candidate_vto_copy_157_ptr) + r541_z, r541_value)
    r542_row = tl.arange($(16))
    r542_col = tl.arange($(64))
    r542_lane = tl.arange($(192))
    for r542_group in range($(8)) {
      r542_left = tl.load($(candidate_vto_copy_153_ptr) + (((((((r542_group % $(8)) * $(3072)) + (r542_row[:, None] * $(192)) + r542_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r542_group % $(8)) * $(3072)) + (r542_row[:, None] * $(192)) + r542_lane[None, :]) / $(192)) % $(16)) * $(192)) + ((((r542_group % $(8)) * $(3072)) + (r542_row[:, None] * $(192)) + r542_lane[None, :]) % $(192))), dtype=tl.float32)
      r542_right = tl.load($(candidate_vto_copy_154_ptr) + (((((((r542_group % $(8)) * $(12288)) + (r542_lane[:, None] * $(64)) + r542_col[None, :]) / $(12288)) % $(8)) * $(12288)) + (((((r542_group % $(8)) * $(12288)) + (r542_lane[:, None] * $(64)) + r542_col[None, :]) % $(64)) * $(192)) + (((((r542_group % $(8)) * $(12288)) + (r542_lane[:, None] * $(64)) + r542_col[None, :]) / $(64)) % $(192))), dtype=tl.float32)
      r542_value = tl.dot(r542_left, r542_right)
      r542_value = r542_value.to(tl.float32)
      tl.store($(candidate_vbmm_76_ptr) + ((r542_group * $(1024)) + (r542_row[:, None] * $(64)) + r542_col[None, :]), r542_value)
    }
    r543_z = tl.arange($(8192))
    r543_value = (tl.load($(candidate_vbmm_76_ptr) + ((((r543_z / $(1024)) % $(8)) * $(1024)) + (((r543_z / $(64)) % $(16)) * $(64)) + (r543_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r543_value = r543_value.to(tl.float32)
    r543_value = tl.broadcast_to(r543_value, [$(8192)])
    tl.store($(candidate_vmul_75_ptr) + r543_z, r543_value)
    r544_z = tl.arange($(16))
    r544_value = (tl.cast(r544_z, tl.int64) + tl.full([], $(64), dtype=tl.int64))
    r544_value = tl.broadcast_to(r544_value, [$(16)])
    tl.store($(candidate_vadd_46_ptr) + r544_z, r544_value)
    r545_z = tl.arange($(16))
    r545_value = (tl.load($(candidate_vadd_46_ptr) + r545_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r545_value = tl.broadcast_to(r545_value, [$(16)])
    tl.store($(candidate_vsub_45_ptr) + r545_z, r545_value)
    r546_z = tl.arange($(1024))
    r546_value = (tl.cast((r546_z % $(64)), tl.int64) > tl.load($(candidate_vsub_45_ptr) + ((r546_z / $(64)) % $(16)), dtype=tl.int64))
    r546_value = tl.broadcast_to(r546_value, [$(1024)])
    tl.store($(candidate_vgt_20_ptr) + r546_z, r546_value)
    r547_z = tl.arange($(8192))
    r547_value = tl.where(tl.load($(candidate_vgt_20_ptr) + ((((r547_z / $(64)) % $(16)) * $(64)) + (r547_z % $(64))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_75_ptr) + ((((r547_z / $(1024)) % $(8)) * $(1024)) + (((r547_z / $(64)) % $(16)) * $(64)) + (r547_z % $(64))), dtype=tl.float32))
    r547_value = r547_value.to(tl.float32)
    r547_value = tl.broadcast_to(r547_value, [$(8192)])
    tl.store($(candidate_vmasked_fill_20_ptr) + r547_z, r547_value)
    r548_z = tl.arange($(8192))
    r548_value = (tl.load($(candidate_vmasked_fill_20_ptr) + ((((r548_z / $(1024)) % $(8)) * $(1024)) + (((r548_z / $(64)) % $(16)) * $(64)) + (r548_z % $(64))), dtype=tl.float32) - tl.load($(candidate_vcat_9_ptr) + ((((r548_z / $(64)) % $(16)) * $(8)) + ((r548_z / $(1024)) % $(8)) + $(128)), dtype=tl.float32))
    r548_value = r548_value.to(tl.float32)
    r548_value = tl.broadcast_to(r548_value, [$(8192)])
    tl.store($(candidate_vsub_46_ptr) + r548_z, r548_value)
    r549_z = tl.arange($(8192))
    r549_value = tl.exp(tl.load($(candidate_vsub_46_ptr) + ((((r549_z / $(1024)) % $(8)) * $(1024)) + (((r549_z / $(64)) % $(16)) * $(64)) + (r549_z % $(64))), dtype=tl.float32))
    r549_value = r549_value.to(tl.float32)
    r549_value = tl.broadcast_to(r549_value, [$(8192)])
    tl.store($(candidate_vexp_8_ptr) + r549_z, r549_value)
    r550_z = tl.arange($(16384))
    r550_value = (tl.load($(candidate_vto_copy_156_ptr) + ((((r550_z / $(2048)) % $(8)) * $(2048)) + (((r550_z / $(128)) % $(16)) * $(128)) + (r550_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_157_ptr) + ((((r550_z / $(2048)) % $(8)) * $(2048)) + (((r550_z / $(128)) % $(16)) * $(128)) + (r550_z % $(128))), dtype=tl.float32))
    r550_value = r550_value.to(tl.float32)
    r550_value = tl.broadcast_to(r550_value, [$(16384)])
    tl.store($(candidate_vmul_76_ptr) + r550_z, r550_value)
    r551_z = tl.arange($(128))
    r551_lane = tl.arange($(128))
    r551_value = tl.sum(tl.load($(candidate_vmul_76_ptr) + ((((r551_z[:, None] / $(16)) % $(8)) * $(2048)) + ((r551_z[:, None] % $(16)) * $(128)) + r551_lane[None, :]), dtype=tl.float32), axis=1)
    r551_value = r551_value.to(tl.float32)
    r551_value = tl.broadcast_to(r551_value, [$(128)])
    tl.store($(candidate_vsum_11_ptr) + r551_z, r551_value)
    r552_row = tl.arange($(16))
    r552_col = tl.arange($(64))
    r552_lane = tl.arange($(128))
    for r552_group in range($(8)) {
      r552_left = tl.load($(candidate_vto_copy_156_ptr) + (((((((r552_group % $(8)) * $(2048)) + (r552_row[:, None] * $(128)) + r552_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r552_group % $(8)) * $(2048)) + (r552_row[:, None] * $(128)) + r552_lane[None, :]) / $(128)) % $(16)) * $(128)) + ((((r552_group % $(8)) * $(2048)) + (r552_row[:, None] * $(128)) + r552_lane[None, :]) % $(128))), dtype=tl.float32)
      r552_right = tl.load($(candidate_vto_copy_155_ptr) + (((((((r552_group % $(8)) * $(8192)) + (r552_lane[:, None] * $(64)) + r552_col[None, :]) / $(8192)) % $(8)) * $(8192)) + (((((r552_group % $(8)) * $(8192)) + (r552_lane[:, None] * $(64)) + r552_col[None, :]) % $(64)) * $(128)) + (((((r552_group % $(8)) * $(8192)) + (r552_lane[:, None] * $(64)) + r552_col[None, :]) / $(64)) % $(128))), dtype=tl.float32)
      r552_value = tl.dot(r552_left, r552_right)
      r552_value = r552_value.to(tl.float32)
      tl.store($(candidate_vbmm_77_ptr) + ((r552_group * $(1024)) + (r552_row[:, None] * $(64)) + r552_col[None, :]), r552_value)
    }
    r553_z = tl.arange($(8192))
    r553_value = (tl.load($(candidate_vbmm_77_ptr) + ((((r553_z / $(1024)) % $(8)) * $(1024)) + (((r553_z / $(64)) % $(16)) * $(64)) + (r553_z % $(64))), dtype=tl.float32) - tl.load($(candidate_vsum_11_ptr) + ((((r553_z / $(1024)) % $(8)) * $(16)) + ((r553_z / $(64)) % $(16))), dtype=tl.float32))
    r553_value = r553_value.to(tl.float32)
    r553_value = tl.broadcast_to(r553_value, [$(8192)])
    tl.store($(candidate_vsub_47_ptr) + r553_z, r553_value)
    r554_z = tl.arange($(8192))
    r554_value = (tl.load($(candidate_vexp_8_ptr) + ((((r554_z / $(1024)) % $(8)) * $(1024)) + (((r554_z / $(64)) % $(16)) * $(64)) + (r554_z % $(64))), dtype=tl.float32) * tl.load($(candidate_vsub_47_ptr) + ((((r554_z / $(1024)) % $(8)) * $(1024)) + (((r554_z / $(64)) % $(16)) * $(64)) + (r554_z % $(64))), dtype=tl.float32))
    r554_value = r554_value.to(tl.float32)
    r554_value = tl.broadcast_to(r554_value, [$(8192)])
    tl.store($(candidate_vmul_77_ptr) + r554_z, r554_value)
    r555_row = tl.arange($(16))
    r555_col = tl.arange($(192))
    r555_lane = tl.arange($(64))
    for r555_group in range($(8)) {
      r555_left = tl.load($(candidate_vmul_77_ptr) + (((((((r555_group % $(8)) * $(1024)) + (r555_row[:, None] * $(64)) + r555_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + ((((((r555_group % $(8)) * $(1024)) + (r555_row[:, None] * $(64)) + r555_lane[None, :]) / $(64)) % $(16)) * $(64)) + ((((r555_group % $(8)) * $(1024)) + (r555_row[:, None] * $(64)) + r555_lane[None, :]) % $(64))), dtype=tl.float32)
      r555_right = tl.load($(candidate_vto_copy_154_ptr) + (((((((r555_group % $(8)) * $(12288)) + (r555_lane[:, None] * $(192)) + r555_col[None, :]) / $(12288)) % $(8)) * $(12288)) + ((((((r555_group % $(8)) * $(12288)) + (r555_lane[:, None] * $(192)) + r555_col[None, :]) / $(192)) % $(64)) * $(192)) + ((((r555_group % $(8)) * $(12288)) + (r555_lane[:, None] * $(192)) + r555_col[None, :]) % $(192))), dtype=tl.float32)
      r555_value = tl.dot(r555_left, r555_right)
      r555_value = r555_value.to(tl.float32)
      tl.store($(candidate_vbmm_78_ptr) + ((r555_group * $(3072)) + (r555_row[:, None] * $(192)) + r555_col[None, :]), r555_value)
    }
    r556_z = tl.arange($(24576))
    r556_value = (tl.load($(candidate_vbmm_78_ptr) + ((((r556_z / $(3072)) % $(8)) * $(3072)) + (((r556_z / $(192)) % $(16)) * $(192)) + (r556_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r556_value = r556_value.to(tl.float32)
    r556_value = tl.broadcast_to(r556_value, [$(24576)])
    tl.store($(candidate_vmul_78_ptr) + r556_z, r556_value)
    r557_z = tl.arange($(24576))
    r557_value = tl.load($(candidate_vmul_78_ptr) + ((((r557_z / $(192)) % $(8)) * $(3072)) + (((r557_z / $(1536)) % $(16)) * $(192)) + (r557_z % $(192))), dtype=tl.float32)
    r557_value = r557_value.to(tl.bfloat16)
    r557_value = tl.broadcast_to(r557_value, [$(24576)])
    tl.store($(candidate_vto_copy_158_ptr) + r557_z, r557_value)
    r558_row = tl.arange($(64))
    r558_col = tl.arange($(192))
    r558_lane = tl.arange($(16))
    for r558_group in range($(8)) {
      r558_left = tl.load($(candidate_vmul_77_ptr) + (((((((r558_group % $(8)) * $(1024)) + (r558_row[:, None] * $(16)) + r558_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + (((((r558_group % $(8)) * $(1024)) + (r558_row[:, None] * $(16)) + r558_lane[None, :]) % $(16)) * $(64)) + (((((r558_group % $(8)) * $(1024)) + (r558_row[:, None] * $(16)) + r558_lane[None, :]) / $(16)) % $(64))), dtype=tl.float32)
      r558_right = tl.load($(candidate_vto_copy_153_ptr) + (((((((r558_group % $(8)) * $(3072)) + (r558_lane[:, None] * $(192)) + r558_col[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r558_group % $(8)) * $(3072)) + (r558_lane[:, None] * $(192)) + r558_col[None, :]) / $(192)) % $(16)) * $(192)) + ((((r558_group % $(8)) * $(3072)) + (r558_lane[:, None] * $(192)) + r558_col[None, :]) % $(192))), dtype=tl.float32)
      r558_value = tl.dot(r558_left, r558_right)
      r558_value = r558_value.to(tl.float32)
      tl.store($(candidate_vbmm_79_ptr) + ((r558_group * $(12288)) + (r558_row[:, None] * $(192)) + r558_col[None, :]), r558_value)
    }
    r559_z = tl.arange($(98304))
    r559_value = (tl.load($(candidate_vbmm_79_ptr) + ((((r559_z / $(12288)) % $(8)) * $(12288)) + (((r559_z / $(192)) % $(64)) * $(192)) + (r559_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r559_value = r559_value.to(tl.float32)
    r559_value = tl.broadcast_to(r559_value, [$(98304)])
    tl.store($(candidate_vmul_79_ptr) + r559_z, r559_value)
    r560_z = tl.arange($(98304))
    r560_value = tl.load($(candidate_vmul_79_ptr) + ((((r560_z / $(192)) % $(8)) * $(12288)) + (((r560_z / $(1536)) % $(64)) * $(192)) + (r560_z % $(192))), dtype=tl.float32)
    r560_value = r560_value.to(tl.bfloat16)
    r560_value = tl.broadcast_to(r560_value, [$(98304)])
    tl.store($(candidate_vto_copy_159_ptr) + r560_z, r560_value)
    r561_row = tl.arange($(64))
    r561_col = tl.arange($(128))
    r561_lane = tl.arange($(16))
    for r561_group in range($(8)) {
      r561_left = tl.load($(candidate_vexp_8_ptr) + (((((((r561_group % $(8)) * $(1024)) + (r561_row[:, None] * $(16)) + r561_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + (((((r561_group % $(8)) * $(1024)) + (r561_row[:, None] * $(16)) + r561_lane[None, :]) % $(16)) * $(64)) + (((((r561_group % $(8)) * $(1024)) + (r561_row[:, None] * $(16)) + r561_lane[None, :]) / $(16)) % $(64))), dtype=tl.float32)
      r561_right = tl.load($(candidate_vto_copy_156_ptr) + (((((((r561_group % $(8)) * $(2048)) + (r561_lane[:, None] * $(128)) + r561_col[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r561_group % $(8)) * $(2048)) + (r561_lane[:, None] * $(128)) + r561_col[None, :]) / $(128)) % $(16)) * $(128)) + ((((r561_group % $(8)) * $(2048)) + (r561_lane[:, None] * $(128)) + r561_col[None, :]) % $(128))), dtype=tl.float32)
      r561_value = tl.dot(r561_left, r561_right)
      r561_value = r561_value.to(tl.float32)
      tl.store($(candidate_vbmm_80_ptr) + ((r561_group * $(8192)) + (r561_row[:, None] * $(128)) + r561_col[None, :]), r561_value)
    }
    r562_z = tl.arange($(65536))
    r562_value = tl.load($(candidate_vbmm_80_ptr) + ((((r562_z / $(128)) % $(8)) * $(8192)) + (((r562_z / $(1024)) % $(64)) * $(128)) + (r562_z % $(128))), dtype=tl.float32)
    r562_value = r562_value.to(tl.bfloat16)
    r562_value = tl.broadcast_to(r562_value, [$(65536)])
    tl.store($(candidate_vto_copy_160_ptr) + r562_z, r562_value)
    r563_z = tl.arange($(49152))
    r563_value = tl.where(((r563_z / $(1536)) % $(32)) < $(12), tl.load($(candidate_vto_copy_142_ptr) + ((((((r563_z / $(1536)) % $(32)) - $(0)) % $(12)) * $(1536)) + (((r563_z / $(192)) % $(8)) * $(192)) + (r563_z % $(192))), dtype=tl.bfloat16), tl.where(((r563_z / $(1536)) % $(32)) < $(16), tl.load($(candidate_vto_copy_150_ptr) + ((((((r563_z / $(1536)) % $(32)) - $(12)) % $(4)) * $(1536)) + (((r563_z / $(192)) % $(8)) * $(192)) + (r563_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vto_copy_158_ptr) + ((((((r563_z / $(1536)) % $(32)) - $(16)) % $(16)) * $(1536)) + (((r563_z / $(192)) % $(8)) * $(192)) + (r563_z % $(192))), dtype=tl.bfloat16)))
    r563_value = r563_value.to(tl.bfloat16)
    r563_value = tl.broadcast_to(r563_value, [$(49152)])
    tl.store($(candidate_vcat_41_ptr) + r563_z, r563_value)
    r564_z = tl.arange($(196608))
    r564_value = tl.where(((r564_z / $(1536)) % $(128)) < $(48), tl.load($(candidate_vto_copy_143_ptr) + ((((((r564_z / $(1536)) % $(128)) - $(0)) % $(48)) * $(1536)) + (((r564_z / $(192)) % $(8)) * $(192)) + (r564_z % $(192))), dtype=tl.bfloat16), tl.where(((r564_z / $(1536)) % $(128)) < $(64), tl.load($(candidate_vto_copy_151_ptr) + ((((((r564_z / $(1536)) % $(128)) - $(48)) % $(16)) * $(1536)) + (((r564_z / $(192)) % $(8)) * $(192)) + (r564_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vto_copy_159_ptr) + ((((((r564_z / $(1536)) % $(128)) - $(64)) % $(64)) * $(1536)) + (((r564_z / $(192)) % $(8)) * $(192)) + (r564_z % $(192))), dtype=tl.bfloat16)))
    r564_value = r564_value.to(tl.bfloat16)
    r564_value = tl.broadcast_to(r564_value, [$(196608)])
    tl.store($(candidate_vcat_42_ptr) + r564_z, r564_value)
    r565_z = tl.arange($(131072))
    r565_value = tl.where(((r565_z / $(1024)) % $(128)) < $(48), tl.load($(candidate_vto_copy_144_ptr) + ((((((r565_z / $(1024)) % $(128)) - $(0)) % $(48)) * $(1024)) + (((r565_z / $(128)) % $(8)) * $(128)) + (r565_z % $(128))), dtype=tl.bfloat16), tl.where(((r565_z / $(1024)) % $(128)) < $(64), tl.load($(candidate_vto_copy_152_ptr) + ((((((r565_z / $(1024)) % $(128)) - $(48)) % $(16)) * $(1024)) + (((r565_z / $(128)) % $(8)) * $(128)) + (r565_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vto_copy_160_ptr) + ((((((r565_z / $(1024)) % $(128)) - $(64)) % $(64)) * $(1024)) + (((r565_z / $(128)) % $(8)) * $(128)) + (r565_z % $(128))), dtype=tl.bfloat16)))
    r565_value = r565_value.to(tl.bfloat16)
    r565_value = tl.broadcast_to(r565_value, [$(131072)])
    tl.store($(candidate_vcat_43_ptr) + r565_z, r565_value)
    r566_z = tl.arange($(8192))
    r566_lane = tl.arange($(8))
    r566_value = tl.sum(tl.load($(candidate_vcat_42_ptr) + ((((r566_z[:, None] / $(64)) % $(128)) * $(1536)) + (r566_lane[None, :] * $(192)) + (r566_z[:, None] % $(64)) + $(128)), dtype=tl.bfloat16), axis=1)
    r566_value = r566_value.to(tl.bfloat16)
    r566_value = tl.broadcast_to(r566_value, [$(8192)])
    tl.store($(candidate_vsum_12_ptr) + r566_z, r566_value)
    r567_row = tl.arange($(128))
    r567_col = tl.arange($(512))
    r567_lane = tl.arange($(1024))
    for r567_group in range($(1)) {
      r567_left = tl.load($(candidate_vcat_42_ptr) + ((((((r567_row[:, None] * $(1024)) + r567_lane[None, :]) / $(1024)) % $(128)) * $(1536)) + ((((r567_row[:, None] * $(1024)) + r567_lane[None, :]) % $(8)) * $(192)) + ((((r567_row[:, None] * $(1024)) + r567_lane[None, :]) / $(8)) % $(128))), dtype=tl.bfloat16)
      r567_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r567_lane[:, None] * $(512)) + r567_col[None, :]) / $(512)) % $(8)) * $(131072)) + (((((r567_lane[:, None] * $(512)) + r567_col[None, :]) / $(4096)) % $(128)) * $(512)) + (((r567_lane[:, None] * $(512)) + r567_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r567_value = tl.dot(r567_left, r567_right)
      r567_value = r567_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_81_ptr) + ((r567_group * $(65536)) + (r567_row[:, None] * $(512)) + r567_col[None, :]), r567_value)
    }
    r568_row = tl.arange($(128))
    r568_col = tl.arange($(512))
    r568_lane = tl.arange($(1024))
    for r568_group in range($(1)) {
      r568_left = tl.load($(candidate_vcat_43_ptr) + ((((((r568_row[:, None] * $(1024)) + r568_lane[None, :]) / $(1024)) % $(128)) * $(1024)) + ((((r568_row[:, None] * $(1024)) + r568_lane[None, :]) % $(8)) * $(128)) + ((((r568_row[:, None] * $(1024)) + r568_lane[None, :]) / $(8)) % $(128))), dtype=tl.bfloat16)
      r568_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r568_lane[:, None] * $(512)) + r568_col[None, :]) / $(512)) % $(8)) * $(131072)) + (((((r568_lane[:, None] * $(512)) + r568_col[None, :]) / $(4096)) % $(128)) * $(512)) + (((r568_lane[:, None] * $(512)) + r568_col[None, :]) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r568_value = tl.dot(r568_left, r568_right)
      r568_value = r568_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_82_ptr) + ((r568_group * $(65536)) + (r568_row[:, None] * $(512)) + r568_col[None, :]), r568_value)
    }
    r569_z = tl.arange($(65536))
    r569_value = (tl.load($(candidate_vbmm_81_ptr) + ((((r569_z / $(512)) % $(128)) * $(512)) + (r569_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vbmm_82_ptr) + ((((r569_z / $(512)) % $(128)) * $(512)) + (r569_z % $(512))), dtype=tl.bfloat16))
    r569_value = r569_value.to(tl.bfloat16)
    r569_value = tl.broadcast_to(r569_value, [$(65536)])
    tl.store($(candidate_vadd_47_ptr) + r569_z, r569_value)
    r570_z = tl.arange($(65536))
    r570_value = tl.where(((((r570_z / $(512)) % $(128)) >= $(0)) & (((r570_z / $(512)) % $(128)) < $(128)) & ((((r570_z / $(512)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vadd_47_ptr) + ((((((r570_z / $(512)) % $(128)) - $(0)) % $(128)) * $(512)) + (r570_z % $(512))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r570_value = r570_value.to(tl.bfloat16)
    r570_value = tl.broadcast_to(r570_value, [$(65536)])
    tl.store($(candidate_vslice_backward_8_ptr) + r570_z, r570_value)
    r571_z = tl.arange($(65536))
    r571_value = (tl.load($(candidate_vadd_40_ptr) + ((((r571_z / $(512)) % $(128)) * $(512)) + (r571_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_8_ptr) + ((((r571_z / $(512)) % $(128)) * $(512)) + (r571_z % $(512))), dtype=tl.bfloat16))
    r571_value = r571_value.to(tl.bfloat16)
    r571_value = tl.broadcast_to(r571_value, [$(65536)])
    tl.store($(candidate_vadd_48_ptr) + r571_z, r571_value)
    r572_z = tl.arange($(8192))
    r572_value = tl.where(((((r572_z / $(64)) % $(128)) >= $(0)) & (((r572_z / $(64)) % $(128)) < $(128)) & ((((r572_z / $(64)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vsum_12_ptr) + ((((((r572_z / $(64)) % $(128)) - $(0)) % $(128)) * $(64)) + (r572_z % $(64))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r572_value = r572_value.to(tl.bfloat16)
    r572_value = tl.broadcast_to(r572_value, [$(8192)])
    tl.store($(candidate_vslice_backward_9_ptr) + r572_z, r572_value)
    r573_z = tl.arange($(8192))
    r573_value = (tl.load($(candidate_vadd_41_ptr) + ((((r573_z / $(64)) % $(128)) * $(64)) + (r573_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_9_ptr) + ((((r573_z / $(64)) % $(128)) * $(64)) + (r573_z % $(64))), dtype=tl.bfloat16))
    r573_value = r573_value.to(tl.bfloat16)
    r573_value = tl.broadcast_to(r573_value, [$(8192)])
    tl.store($(candidate_vadd_49_ptr) + r573_z, r573_value)
    r574_z = tl.arange($(131072))
    r574_value = tl.where(((((r574_z / $(1024)) % $(128)) >= $(32)) & (((r574_z / $(1024)) % $(128)) < $(64)) & ((((r574_z / $(1024)) % $(128)) - $(32)) % $(1) == $(0))), tl.load($(candidate_vcat_41_ptr) + ((((((r574_z / $(1024)) % $(128)) - $(32)) % $(32)) * $(1536)) + (((r574_z / $(128)) % $(8)) * $(192)) + (r574_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r574_value = r574_value.to(tl.bfloat16)
    r574_value = tl.broadcast_to(r574_value, [$(131072)])
    tl.store($(candidate_vslice_backward_10_ptr) + r574_z, r574_value)
    r575_z = tl.arange($(131072))
    r575_value = (tl.load($(candidate_vadd_42_ptr) + ((((r575_z / $(1024)) % $(128)) * $(1024)) + (((r575_z / $(128)) % $(8)) * $(128)) + (r575_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_10_ptr) + ((((r575_z / $(1024)) % $(128)) * $(1024)) + (((r575_z / $(128)) % $(8)) * $(128)) + (r575_z % $(128))), dtype=tl.bfloat16))
    r575_value = r575_value.to(tl.bfloat16)
    r575_value = tl.broadcast_to(r575_value, [$(131072)])
    tl.store($(candidate_vadd_50_ptr) + r575_z, r575_value)
    r576_z = tl.arange($(65536))
    r576_value = tl.where(((((r576_z / $(512)) % $(128)) >= $(32)) & (((r576_z / $(512)) % $(128)) < $(64)) & ((((r576_z / $(512)) % $(128)) - $(32)) % $(1) == $(0))), tl.load($(candidate_vcat_41_ptr) + ((((((r576_z / $(512)) % $(128)) - $(32)) % $(32)) * $(1536)) + (((r576_z / $(64)) % $(8)) * $(192)) + (r576_z % $(64)) + $(128)), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r576_value = r576_value.to(tl.bfloat16)
    r576_value = tl.broadcast_to(r576_value, [$(65536)])
    tl.store($(candidate_vslice_backward_11_ptr) + r576_z, r576_value)
    r577_z = tl.arange($(65536))
    r577_value = (tl.load($(candidate_vadd_43_ptr) + ((((r577_z / $(512)) % $(128)) * $(512)) + (((r577_z / $(64)) % $(8)) * $(64)) + (r577_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_11_ptr) + ((((r577_z / $(512)) % $(128)) * $(512)) + (((r577_z / $(64)) % $(8)) * $(64)) + (r577_z % $(64))), dtype=tl.bfloat16))
    r577_value = r577_value.to(tl.bfloat16)
    r577_value = tl.broadcast_to(r577_value, [$(65536)])
    tl.store($(candidate_vadd_51_ptr) + r577_z, r577_value)
    r578_row = tl.arange($(128))
    r578_col = tl.arange($(1024))
    r578_lane = tl.arange($(512))
    for r578_group in range($(1)) {
      r578_left = tl.load($(candidate_vindex_2_ptr) + ((((((r578_row[:, None] * $(512)) + r578_lane[None, :]) / $(512)) % $(128)) * $(576)) + (((r578_row[:, None] * $(512)) + r578_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r578_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r578_lane[:, None] * $(1024)) + r578_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r578_lane[:, None] * $(1024)) + r578_col[None, :]) % $(128)) * $(512)) + ((((r578_lane[:, None] * $(1024)) + r578_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r578_value = tl.dot(r578_left, r578_right)
      r578_value = r578_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_85_ptr) + ((r578_group * $(131072)) + (r578_row[:, None] * $(1024)) + r578_col[None, :]), r578_value)
    }
    r579_row = tl.arange($(128))
    r579_col = tl.arange($(1024))
    r579_lane = tl.arange($(512))
    for r579_group in range($(1)) {
      r579_left = tl.load($(candidate_vindex_2_ptr) + ((((((r579_row[:, None] * $(512)) + r579_lane[None, :]) / $(512)) % $(128)) * $(576)) + (((r579_row[:, None] * $(512)) + r579_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r579_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r579_lane[:, None] * $(1024)) + r579_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r579_lane[:, None] * $(1024)) + r579_col[None, :]) % $(128)) * $(512)) + ((((r579_lane[:, None] * $(1024)) + r579_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r579_value = tl.dot(r579_left, r579_right)
      r579_value = r579_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_86_ptr) + ((r579_group * $(131072)) + (r579_row[:, None] * $(1024)) + r579_col[None, :]), r579_value)
    }
    r580_z = tl.arange($(49152))
    r580_value = tl.where((r580_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r580_z / $(1536)) % $(32)) * $(1024)) + (((r580_z / $(192)) % $(8)) * $(128)) + ((r580_z % $(192)) - $(0))) / $(1024)) % $(128)) * $(1536)) + (((((((r580_z / $(1536)) % $(32)) * $(1024)) + (((r580_z / $(192)) % $(8)) * $(128)) + ((r580_z % $(192)) - $(0))) / $(128)) % $(8)) * $(192)) + (((((r580_z / $(1536)) % $(32)) * $(1024)) + (((r580_z / $(192)) % $(8)) * $(128)) + ((r580_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vadd_6_ptr) + ((((((((r580_z / $(1536)) % $(32)) * $(512)) + (((r580_z / $(192)) % $(8)) * $(64)) + ((r580_z % $(192)) - $(128))) / $(512)) % $(128)) * $(512)) + (((((((r580_z / $(1536)) % $(32)) * $(512)) + (((r580_z / $(192)) % $(8)) * $(64)) + ((r580_z % $(192)) - $(128))) / $(64)) % $(8)) * $(64)) + (((((r580_z / $(1536)) % $(32)) * $(512)) + (((r580_z / $(192)) % $(8)) * $(64)) + ((r580_z % $(192)) - $(128))) % $(64))), dtype=tl.bfloat16))
    r580_value = r580_value.to(tl.bfloat16)
    r580_value = tl.broadcast_to(r580_value, [$(49152)])
    tl.store($(candidate_vcat_45_ptr) + r580_z, r580_value)
    r581_z = tl.arange($(196608))
    r581_value = tl.where((r581_z % $(192)) < $(128), tl.load($(candidate_vbmm_85_ptr) + ((((((((r581_z / $(1536)) % $(128)) * $(1024)) + (((r581_z / $(192)) % $(8)) * $(128)) + ((r581_z % $(192)) - $(0))) / $(1024)) % $(128)) * $(1024)) + (((((((r581_z / $(1536)) % $(128)) * $(1024)) + (((r581_z / $(192)) % $(8)) * $(128)) + ((r581_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r581_z / $(1536)) % $(128)) * $(1024)) + (((r581_z / $(192)) % $(8)) * $(128)) + ((r581_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vindex_2_ptr) + ((((r581_z / $(1536)) % $(128)) * $(576)) + ((((r581_z % $(192)) - $(128)) + $(512)) % $(576))), dtype=tl.bfloat16))
    r581_value = r581_value.to(tl.bfloat16)
    r581_value = tl.broadcast_to(r581_value, [$(196608)])
    tl.store($(candidate_vcat_46_ptr) + r581_z, r581_value)
    r582_z = tl.arange($(18432))
    r582_value = tl.load($(candidate_vcat_45_ptr) + ((((r582_z / $(192)) % $(12)) * $(1536)) + (((r582_z / $(2304)) % $(8)) * $(192)) + (r582_z % $(192))), dtype=tl.bfloat16)
    r582_value = r582_value.to(tl.float32)
    r582_value = tl.broadcast_to(r582_value, [$(18432)])
    tl.store($(candidate_vto_copy_161_ptr) + r582_z, r582_value)
    r583_z = tl.arange($(18432))
    r583_value = tl.load($(candidate_vcat_46_ptr) + ((((r583_z / $(192)) % $(12)) * $(1536)) + (((r583_z / $(2304)) % $(8)) * $(192)) + (r583_z % $(192))), dtype=tl.bfloat16)
    r583_value = r583_value.to(tl.float32)
    r583_value = tl.broadcast_to(r583_value, [$(18432)])
    tl.store($(candidate_vto_copy_162_ptr) + r583_z, r583_value)
    r584_z = tl.arange($(12288))
    r584_value = tl.load($(candidate_vbmm_86_ptr) + ((((r584_z / $(128)) % $(12)) * $(1024)) + (((r584_z / $(1536)) % $(8)) * $(128)) + (r584_z % $(128))), dtype=tl.bfloat16)
    r584_value = r584_value.to(tl.float32)
    r584_value = tl.broadcast_to(r584_value, [$(12288)])
    tl.store($(candidate_vto_copy_163_ptr) + r584_z, r584_value)
    r585_z = tl.arange($(12288))
    r585_value = tl.load($(candidate_vmm_28_ptr) + ((((r585_z / $(128)) % $(12)) * $(1024)) + (((r585_z / $(1536)) % $(8)) * $(128)) + (r585_z % $(128))), dtype=tl.bfloat16)
    r585_value = r585_value.to(tl.float32)
    r585_value = tl.broadcast_to(r585_value, [$(12288)])
    tl.store($(candidate_vto_copy_164_ptr) + r585_z, r585_value)
    r586_z = tl.arange($(12288))
    r586_value = tl.load($(candidate_vbmm_6_ptr) + ((((r586_z / $(1536)) % $(8)) * $(4096)) + (((r586_z / $(128)) % $(12)) * $(128)) + (r586_z % $(128))), dtype=tl.bfloat16)
    r586_value = r586_value.to(tl.float32)
    r586_value = tl.broadcast_to(r586_value, [$(12288)])
    tl.store($(candidate_vto_copy_165_ptr) + r586_z, r586_value)
    r587_row = tl.arange($(12))
    r587_col = tl.arange($(12))
    r587_lane = tl.arange($(192))
    for r587_group in range($(8)) {
      r587_left = tl.load($(candidate_vto_copy_161_ptr) + (((((((r587_group % $(8)) * $(2304)) + (r587_row[:, None] * $(192)) + r587_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r587_group % $(8)) * $(2304)) + (r587_row[:, None] * $(192)) + r587_lane[None, :]) / $(192)) % $(12)) * $(192)) + ((((r587_group % $(8)) * $(2304)) + (r587_row[:, None] * $(192)) + r587_lane[None, :]) % $(192))), dtype=tl.float32)
      r587_right = tl.load($(candidate_vto_copy_162_ptr) + (((((((r587_group % $(8)) * $(2304)) + (r587_lane[:, None] * $(12)) + r587_col[None, :]) / $(2304)) % $(8)) * $(2304)) + (((((r587_group % $(8)) * $(2304)) + (r587_lane[:, None] * $(12)) + r587_col[None, :]) % $(12)) * $(192)) + (((((r587_group % $(8)) * $(2304)) + (r587_lane[:, None] * $(12)) + r587_col[None, :]) / $(12)) % $(192))), dtype=tl.float32)
      r587_value = tl.dot(r587_left, r587_right)
      r587_value = r587_value.to(tl.float32)
      tl.store($(candidate_vbmm_87_ptr) + ((r587_group * $(144)) + (r587_row[:, None] * $(12)) + r587_col[None, :]), r587_value)
    }
    r588_z = tl.arange($(1152))
    r588_value = (tl.load($(candidate_vbmm_87_ptr) + ((((r588_z / $(144)) % $(8)) * $(144)) + (((r588_z / $(12)) % $(12)) * $(12)) + (r588_z % $(12))), dtype=tl.float32) * 0.07216878364870322)
    r588_value = r588_value.to(tl.float32)
    r588_value = tl.broadcast_to(r588_value, [$(1152)])
    tl.store($(candidate_vmul_80_ptr) + r588_z, r588_value)
    r589_z = tl.arange($(12))
    r589_value = (tl.cast(r589_z, tl.int64) + tl.full([], $(12), dtype=tl.int64))
    r589_value = tl.broadcast_to(r589_value, [$(12)])
    tl.store($(candidate_vadd_52_ptr) + r589_z, r589_value)
    r590_z = tl.arange($(12))
    r590_value = (tl.load($(candidate_vadd_52_ptr) + r590_z, dtype=tl.int64) - tl.full([], $(12), dtype=tl.int64))
    r590_value = tl.broadcast_to(r590_value, [$(12)])
    tl.store($(candidate_vsub_48_ptr) + r590_z, r590_value)
    r591_z = tl.arange($(144))
    r591_value = (tl.cast((r591_z % $(12)), tl.int64) > tl.load($(candidate_vsub_48_ptr) + ((r591_z / $(12)) % $(12)), dtype=tl.int64))
    r591_value = tl.broadcast_to(r591_value, [$(144)])
    tl.store($(candidate_vgt_21_ptr) + r591_z, r591_value)
    r592_z = tl.arange($(1152))
    r592_value = tl.where(tl.load($(candidate_vgt_21_ptr) + ((((r592_z / $(12)) % $(12)) * $(12)) + (r592_z % $(12))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_80_ptr) + ((((r592_z / $(144)) % $(8)) * $(144)) + (((r592_z / $(12)) % $(12)) * $(12)) + (r592_z % $(12))), dtype=tl.float32))
    r592_value = r592_value.to(tl.float32)
    r592_value = tl.broadcast_to(r592_value, [$(1152)])
    tl.store($(candidate_vmasked_fill_21_ptr) + r592_z, r592_value)
    r593_z = tl.arange($(1152))
    r593_value = (tl.load($(candidate_vmasked_fill_21_ptr) + ((((r593_z / $(144)) % $(8)) * $(144)) + (((r593_z / $(12)) % $(12)) * $(12)) + (r593_z % $(12))), dtype=tl.float32) - tl.load($(candidate_vcat_7_ptr) + ((((r593_z / $(12)) % $(12)) * $(8)) + ((r593_z / $(144)) % $(8))), dtype=tl.float32))
    r593_value = r593_value.to(tl.float32)
    r593_value = tl.broadcast_to(r593_value, [$(1152)])
    tl.store($(candidate_vsub_49_ptr) + r593_z, r593_value)
    r594_z = tl.arange($(1152))
    r594_value = tl.exp(tl.load($(candidate_vsub_49_ptr) + ((((r594_z / $(144)) % $(8)) * $(144)) + (((r594_z / $(12)) % $(12)) * $(12)) + (r594_z % $(12))), dtype=tl.float32))
    r594_value = r594_value.to(tl.float32)
    r594_value = tl.broadcast_to(r594_value, [$(1152)])
    tl.store($(candidate_vexp_9_ptr) + r594_z, r594_value)
    r595_z = tl.arange($(12288))
    r595_value = (tl.load($(candidate_vto_copy_164_ptr) + ((((r595_z / $(1536)) % $(8)) * $(1536)) + (((r595_z / $(128)) % $(12)) * $(128)) + (r595_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_165_ptr) + ((((r595_z / $(1536)) % $(8)) * $(1536)) + (((r595_z / $(128)) % $(12)) * $(128)) + (r595_z % $(128))), dtype=tl.float32))
    r595_value = r595_value.to(tl.float32)
    r595_value = tl.broadcast_to(r595_value, [$(12288)])
    tl.store($(candidate_vmul_81_ptr) + r595_z, r595_value)
    r596_z = tl.arange($(96))
    r596_lane = tl.arange($(128))
    r596_value = tl.sum(tl.load($(candidate_vmul_81_ptr) + ((((r596_z[:, None] / $(12)) % $(8)) * $(1536)) + ((r596_z[:, None] % $(12)) * $(128)) + r596_lane[None, :]), dtype=tl.float32), axis=1)
    r596_value = r596_value.to(tl.float32)
    r596_value = tl.broadcast_to(r596_value, [$(96)])
    tl.store($(candidate_vsum_13_ptr) + r596_z, r596_value)
    r597_row = tl.arange($(12))
    r597_col = tl.arange($(12))
    r597_lane = tl.arange($(128))
    for r597_group in range($(8)) {
      r597_left = tl.load($(candidate_vto_copy_164_ptr) + (((((((r597_group % $(8)) * $(1536)) + (r597_row[:, None] * $(128)) + r597_lane[None, :]) / $(1536)) % $(8)) * $(1536)) + ((((((r597_group % $(8)) * $(1536)) + (r597_row[:, None] * $(128)) + r597_lane[None, :]) / $(128)) % $(12)) * $(128)) + ((((r597_group % $(8)) * $(1536)) + (r597_row[:, None] * $(128)) + r597_lane[None, :]) % $(128))), dtype=tl.float32)
      r597_right = tl.load($(candidate_vto_copy_163_ptr) + (((((((r597_group % $(8)) * $(1536)) + (r597_lane[:, None] * $(12)) + r597_col[None, :]) / $(1536)) % $(8)) * $(1536)) + (((((r597_group % $(8)) * $(1536)) + (r597_lane[:, None] * $(12)) + r597_col[None, :]) % $(12)) * $(128)) + (((((r597_group % $(8)) * $(1536)) + (r597_lane[:, None] * $(12)) + r597_col[None, :]) / $(12)) % $(128))), dtype=tl.float32)
      r597_value = tl.dot(r597_left, r597_right)
      r597_value = r597_value.to(tl.float32)
      tl.store($(candidate_vbmm_88_ptr) + ((r597_group * $(144)) + (r597_row[:, None] * $(12)) + r597_col[None, :]), r597_value)
    }
    r598_z = tl.arange($(1152))
    r598_value = (tl.load($(candidate_vbmm_88_ptr) + ((((r598_z / $(144)) % $(8)) * $(144)) + (((r598_z / $(12)) % $(12)) * $(12)) + (r598_z % $(12))), dtype=tl.float32) - tl.load($(candidate_vsum_13_ptr) + ((((r598_z / $(144)) % $(8)) * $(12)) + ((r598_z / $(12)) % $(12))), dtype=tl.float32))
    r598_value = r598_value.to(tl.float32)
    r598_value = tl.broadcast_to(r598_value, [$(1152)])
    tl.store($(candidate_vsub_50_ptr) + r598_z, r598_value)
    r599_z = tl.arange($(1152))
    r599_value = (tl.load($(candidate_vexp_9_ptr) + ((((r599_z / $(144)) % $(8)) * $(144)) + (((r599_z / $(12)) % $(12)) * $(12)) + (r599_z % $(12))), dtype=tl.float32) * tl.load($(candidate_vsub_50_ptr) + ((((r599_z / $(144)) % $(8)) * $(144)) + (((r599_z / $(12)) % $(12)) * $(12)) + (r599_z % $(12))), dtype=tl.float32))
    r599_value = r599_value.to(tl.float32)
    r599_value = tl.broadcast_to(r599_value, [$(1152)])
    tl.store($(candidate_vmul_82_ptr) + r599_z, r599_value)
    r600_row = tl.arange($(12))
    r600_col = tl.arange($(192))
    r600_lane = tl.arange($(12))
    for r600_group in range($(8)) {
      r600_left = tl.load($(candidate_vmul_82_ptr) + (((((((r600_group % $(8)) * $(144)) + (r600_row[:, None] * $(12)) + r600_lane[None, :]) / $(144)) % $(8)) * $(144)) + ((((((r600_group % $(8)) * $(144)) + (r600_row[:, None] * $(12)) + r600_lane[None, :]) / $(12)) % $(12)) * $(12)) + ((((r600_group % $(8)) * $(144)) + (r600_row[:, None] * $(12)) + r600_lane[None, :]) % $(12))), dtype=tl.float32)
      r600_right = tl.load($(candidate_vto_copy_162_ptr) + (((((((r600_group % $(8)) * $(2304)) + (r600_lane[:, None] * $(192)) + r600_col[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r600_group % $(8)) * $(2304)) + (r600_lane[:, None] * $(192)) + r600_col[None, :]) / $(192)) % $(12)) * $(192)) + ((((r600_group % $(8)) * $(2304)) + (r600_lane[:, None] * $(192)) + r600_col[None, :]) % $(192))), dtype=tl.float32)
      r600_value = tl.dot(r600_left, r600_right)
      r600_value = r600_value.to(tl.float32)
      tl.store($(candidate_vbmm_89_ptr) + ((r600_group * $(2304)) + (r600_row[:, None] * $(192)) + r600_col[None, :]), r600_value)
    }
    r601_z = tl.arange($(18432))
    r601_value = (tl.load($(candidate_vbmm_89_ptr) + ((((r601_z / $(2304)) % $(8)) * $(2304)) + (((r601_z / $(192)) % $(12)) * $(192)) + (r601_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r601_value = r601_value.to(tl.float32)
    r601_value = tl.broadcast_to(r601_value, [$(18432)])
    tl.store($(candidate_vmul_83_ptr) + r601_z, r601_value)
    r602_z = tl.arange($(18432))
    r602_value = tl.load($(candidate_vmul_83_ptr) + ((((r602_z / $(192)) % $(8)) * $(2304)) + (((r602_z / $(1536)) % $(12)) * $(192)) + (r602_z % $(192))), dtype=tl.float32)
    r602_value = r602_value.to(tl.bfloat16)
    r602_value = tl.broadcast_to(r602_value, [$(18432)])
    tl.store($(candidate_vto_copy_166_ptr) + r602_z, r602_value)
    r603_row = tl.arange($(12))
    r603_col = tl.arange($(192))
    r603_lane = tl.arange($(12))
    for r603_group in range($(8)) {
      r603_left = tl.load($(candidate_vmul_82_ptr) + (((((((r603_group % $(8)) * $(144)) + (r603_row[:, None] * $(12)) + r603_lane[None, :]) / $(144)) % $(8)) * $(144)) + (((((r603_group % $(8)) * $(144)) + (r603_row[:, None] * $(12)) + r603_lane[None, :]) % $(12)) * $(12)) + (((((r603_group % $(8)) * $(144)) + (r603_row[:, None] * $(12)) + r603_lane[None, :]) / $(12)) % $(12))), dtype=tl.float32)
      r603_right = tl.load($(candidate_vto_copy_161_ptr) + (((((((r603_group % $(8)) * $(2304)) + (r603_lane[:, None] * $(192)) + r603_col[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r603_group % $(8)) * $(2304)) + (r603_lane[:, None] * $(192)) + r603_col[None, :]) / $(192)) % $(12)) * $(192)) + ((((r603_group % $(8)) * $(2304)) + (r603_lane[:, None] * $(192)) + r603_col[None, :]) % $(192))), dtype=tl.float32)
      r603_value = tl.dot(r603_left, r603_right)
      r603_value = r603_value.to(tl.float32)
      tl.store($(candidate_vbmm_90_ptr) + ((r603_group * $(2304)) + (r603_row[:, None] * $(192)) + r603_col[None, :]), r603_value)
    }
    r604_z = tl.arange($(18432))
    r604_value = (tl.load($(candidate_vbmm_90_ptr) + ((((r604_z / $(2304)) % $(8)) * $(2304)) + (((r604_z / $(192)) % $(12)) * $(192)) + (r604_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r604_value = r604_value.to(tl.float32)
    r604_value = tl.broadcast_to(r604_value, [$(18432)])
    tl.store($(candidate_vmul_84_ptr) + r604_z, r604_value)
    r605_z = tl.arange($(18432))
    r605_value = tl.load($(candidate_vmul_84_ptr) + ((((r605_z / $(192)) % $(8)) * $(2304)) + (((r605_z / $(1536)) % $(12)) * $(192)) + (r605_z % $(192))), dtype=tl.float32)
    r605_value = r605_value.to(tl.bfloat16)
    r605_value = tl.broadcast_to(r605_value, [$(18432)])
    tl.store($(candidate_vto_copy_167_ptr) + r605_z, r605_value)
    r606_row = tl.arange($(12))
    r606_col = tl.arange($(128))
    r606_lane = tl.arange($(12))
    for r606_group in range($(8)) {
      r606_left = tl.load($(candidate_vexp_9_ptr) + (((((((r606_group % $(8)) * $(144)) + (r606_row[:, None] * $(12)) + r606_lane[None, :]) / $(144)) % $(8)) * $(144)) + (((((r606_group % $(8)) * $(144)) + (r606_row[:, None] * $(12)) + r606_lane[None, :]) % $(12)) * $(12)) + (((((r606_group % $(8)) * $(144)) + (r606_row[:, None] * $(12)) + r606_lane[None, :]) / $(12)) % $(12))), dtype=tl.float32)
      r606_right = tl.load($(candidate_vto_copy_164_ptr) + (((((((r606_group % $(8)) * $(1536)) + (r606_lane[:, None] * $(128)) + r606_col[None, :]) / $(1536)) % $(8)) * $(1536)) + ((((((r606_group % $(8)) * $(1536)) + (r606_lane[:, None] * $(128)) + r606_col[None, :]) / $(128)) % $(12)) * $(128)) + ((((r606_group % $(8)) * $(1536)) + (r606_lane[:, None] * $(128)) + r606_col[None, :]) % $(128))), dtype=tl.float32)
      r606_value = tl.dot(r606_left, r606_right)
      r606_value = r606_value.to(tl.float32)
      tl.store($(candidate_vbmm_91_ptr) + ((r606_group * $(1536)) + (r606_row[:, None] * $(128)) + r606_col[None, :]), r606_value)
    }
    r607_z = tl.arange($(12288))
    r607_value = tl.load($(candidate_vbmm_91_ptr) + ((((r607_z / $(128)) % $(8)) * $(1536)) + (((r607_z / $(1024)) % $(12)) * $(128)) + (r607_z % $(128))), dtype=tl.float32)
    r607_value = r607_value.to(tl.bfloat16)
    r607_value = tl.broadcast_to(r607_value, [$(12288)])
    tl.store($(candidate_vto_copy_168_ptr) + r607_z, r607_value)
    r608_z = tl.arange($(73728))
    r608_value = tl.where(((r608_z / $(1536)) % $(48)) < $(12), tl.load($(candidate_vto_copy_167_ptr) + ((((((r608_z / $(1536)) % $(48)) - $(0)) % $(12)) * $(1536)) + (((r608_z / $(192)) % $(8)) * $(192)) + (r608_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r608_value = r608_value.to(tl.bfloat16)
    r608_value = tl.broadcast_to(r608_value, [$(73728)])
    tl.store($(candidate_vcat_47_ptr) + r608_z, r608_value)
    r609_z = tl.arange($(49152))
    r609_value = tl.where(((r609_z / $(1024)) % $(48)) < $(12), tl.load($(candidate_vto_copy_168_ptr) + ((((((r609_z / $(1024)) % $(48)) - $(0)) % $(12)) * $(1024)) + (((r609_z / $(128)) % $(8)) * $(128)) + (r609_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r609_value = r609_value.to(tl.bfloat16)
    r609_value = tl.broadcast_to(r609_value, [$(49152)])
    tl.store($(candidate_vcat_48_ptr) + r609_z, r609_value)
    r610_z = tl.arange($(6144))
    r610_value = tl.load($(candidate_vcat_45_ptr) + ((((r610_z / $(192)) % $(4)) * $(1536)) + (((r610_z / $(768)) % $(8)) * $(192)) + (r610_z % $(192)) + $(18432)), dtype=tl.bfloat16)
    r610_value = r610_value.to(tl.float32)
    r610_value = tl.broadcast_to(r610_value, [$(6144)])
    tl.store($(candidate_vto_copy_169_ptr) + r610_z, r610_value)
    r611_z = tl.arange($(6144))
    r611_value = tl.load($(candidate_vcat_46_ptr) + ((((r611_z / $(192)) % $(4)) * $(1536)) + (((r611_z / $(768)) % $(8)) * $(192)) + (r611_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r611_value = r611_value.to(tl.float32)
    r611_value = tl.broadcast_to(r611_value, [$(6144)])
    tl.store($(candidate_vto_copy_170_ptr) + r611_z, r611_value)
    r612_z = tl.arange($(4096))
    r612_value = tl.load($(candidate_vbmm_86_ptr) + ((((r612_z / $(128)) % $(4)) * $(1024)) + (((r612_z / $(512)) % $(8)) * $(128)) + (r612_z % $(128)) + $(49152)), dtype=tl.bfloat16)
    r612_value = r612_value.to(tl.float32)
    r612_value = tl.broadcast_to(r612_value, [$(4096)])
    tl.store($(candidate_vto_copy_171_ptr) + r612_z, r612_value)
    r613_z = tl.arange($(4096))
    r613_value = tl.load($(candidate_vmm_28_ptr) + ((((r613_z / $(128)) % $(4)) * $(1024)) + (((r613_z / $(512)) % $(8)) * $(128)) + (r613_z % $(128)) + $(12288)), dtype=tl.bfloat16)
    r613_value = r613_value.to(tl.float32)
    r613_value = tl.broadcast_to(r613_value, [$(4096)])
    tl.store($(candidate_vto_copy_172_ptr) + r613_z, r613_value)
    r614_z = tl.arange($(4096))
    r614_value = tl.load($(candidate_vbmm_6_ptr) + ((((r614_z / $(512)) % $(8)) * $(4096)) + (((r614_z / $(128)) % $(4)) * $(128)) + (r614_z % $(128)) + $(1536)), dtype=tl.bfloat16)
    r614_value = r614_value.to(tl.float32)
    r614_value = tl.broadcast_to(r614_value, [$(4096)])
    tl.store($(candidate_vto_copy_173_ptr) + r614_z, r614_value)
    r615_row = tl.arange($(4))
    r615_col = tl.arange($(4))
    r615_lane = tl.arange($(192))
    for r615_group in range($(8)) {
      r615_left = tl.load($(candidate_vto_copy_169_ptr) + (((((((r615_group % $(8)) * $(768)) + (r615_row[:, None] * $(192)) + r615_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r615_group % $(8)) * $(768)) + (r615_row[:, None] * $(192)) + r615_lane[None, :]) / $(192)) % $(4)) * $(192)) + ((((r615_group % $(8)) * $(768)) + (r615_row[:, None] * $(192)) + r615_lane[None, :]) % $(192))), dtype=tl.float32)
      r615_right = tl.load($(candidate_vto_copy_170_ptr) + (((((((r615_group % $(8)) * $(768)) + (r615_lane[:, None] * $(4)) + r615_col[None, :]) / $(768)) % $(8)) * $(768)) + (((((r615_group % $(8)) * $(768)) + (r615_lane[:, None] * $(4)) + r615_col[None, :]) % $(4)) * $(192)) + (((((r615_group % $(8)) * $(768)) + (r615_lane[:, None] * $(4)) + r615_col[None, :]) / $(4)) % $(192))), dtype=tl.float32)
      r615_value = tl.dot(r615_left, r615_right)
      r615_value = r615_value.to(tl.float32)
      tl.store($(candidate_vbmm_92_ptr) + ((r615_group * $(16)) + (r615_row[:, None] * $(4)) + r615_col[None, :]), r615_value)
    }
    r616_z = tl.arange($(128))
    r616_value = (tl.load($(candidate_vbmm_92_ptr) + ((((r616_z / $(16)) % $(8)) * $(16)) + (((r616_z / $(4)) % $(4)) * $(4)) + (r616_z % $(4))), dtype=tl.float32) * 0.07216878364870322)
    r616_value = r616_value.to(tl.float32)
    r616_value = tl.broadcast_to(r616_value, [$(128)])
    tl.store($(candidate_vmul_85_ptr) + r616_z, r616_value)
    r617_z = tl.arange($(4))
    r617_value = (tl.cast(r617_z, tl.int64) + tl.full([], $(4), dtype=tl.int64))
    r617_value = tl.broadcast_to(r617_value, [$(4)])
    tl.store($(candidate_vadd_53_ptr) + r617_z, r617_value)
    r618_z = tl.arange($(4))
    r618_value = (tl.load($(candidate_vadd_53_ptr) + r618_z, dtype=tl.int64) - tl.full([], $(4), dtype=tl.int64))
    r618_value = tl.broadcast_to(r618_value, [$(4)])
    tl.store($(candidate_vsub_51_ptr) + r618_z, r618_value)
    r619_z = tl.arange($(16))
    r619_value = (tl.cast((r619_z % $(4)), tl.int64) > tl.load($(candidate_vsub_51_ptr) + ((r619_z / $(4)) % $(4)), dtype=tl.int64))
    r619_value = tl.broadcast_to(r619_value, [$(16)])
    tl.store($(candidate_vgt_22_ptr) + r619_z, r619_value)
    r620_z = tl.arange($(128))
    r620_value = tl.where(tl.load($(candidate_vgt_22_ptr) + ((((r620_z / $(4)) % $(4)) * $(4)) + (r620_z % $(4))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_85_ptr) + ((((r620_z / $(16)) % $(8)) * $(16)) + (((r620_z / $(4)) % $(4)) * $(4)) + (r620_z % $(4))), dtype=tl.float32))
    r620_value = r620_value.to(tl.float32)
    r620_value = tl.broadcast_to(r620_value, [$(128)])
    tl.store($(candidate_vmasked_fill_22_ptr) + r620_z, r620_value)
    r621_z = tl.arange($(128))
    r621_value = (tl.load($(candidate_vmasked_fill_22_ptr) + ((((r621_z / $(16)) % $(8)) * $(16)) + (((r621_z / $(4)) % $(4)) * $(4)) + (r621_z % $(4))), dtype=tl.float32) - tl.load($(candidate_vcat_7_ptr) + ((((r621_z / $(4)) % $(4)) * $(8)) + ((r621_z / $(16)) % $(8)) + $(96)), dtype=tl.float32))
    r621_value = r621_value.to(tl.float32)
    r621_value = tl.broadcast_to(r621_value, [$(128)])
    tl.store($(candidate_vsub_52_ptr) + r621_z, r621_value)
    r622_z = tl.arange($(128))
    r622_value = tl.exp(tl.load($(candidate_vsub_52_ptr) + ((((r622_z / $(16)) % $(8)) * $(16)) + (((r622_z / $(4)) % $(4)) * $(4)) + (r622_z % $(4))), dtype=tl.float32))
    r622_value = r622_value.to(tl.float32)
    r622_value = tl.broadcast_to(r622_value, [$(128)])
    tl.store($(candidate_vexp_10_ptr) + r622_z, r622_value)
    r623_z = tl.arange($(4096))
    r623_value = (tl.load($(candidate_vto_copy_172_ptr) + ((((r623_z / $(512)) % $(8)) * $(512)) + (((r623_z / $(128)) % $(4)) * $(128)) + (r623_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_173_ptr) + ((((r623_z / $(512)) % $(8)) * $(512)) + (((r623_z / $(128)) % $(4)) * $(128)) + (r623_z % $(128))), dtype=tl.float32))
    r623_value = r623_value.to(tl.float32)
    r623_value = tl.broadcast_to(r623_value, [$(4096)])
    tl.store($(candidate_vmul_86_ptr) + r623_z, r623_value)
    r624_z = tl.arange($(32))
    r624_lane = tl.arange($(128))
    r624_value = tl.sum(tl.load($(candidate_vmul_86_ptr) + ((((r624_z[:, None] / $(4)) % $(8)) * $(512)) + ((r624_z[:, None] % $(4)) * $(128)) + r624_lane[None, :]), dtype=tl.float32), axis=1)
    r624_value = r624_value.to(tl.float32)
    r624_value = tl.broadcast_to(r624_value, [$(32)])
    tl.store($(candidate_vsum_14_ptr) + r624_z, r624_value)
    r625_row = tl.arange($(4))
    r625_col = tl.arange($(4))
    r625_lane = tl.arange($(128))
    for r625_group in range($(8)) {
      r625_left = tl.load($(candidate_vto_copy_172_ptr) + (((((((r625_group % $(8)) * $(512)) + (r625_row[:, None] * $(128)) + r625_lane[None, :]) / $(512)) % $(8)) * $(512)) + ((((((r625_group % $(8)) * $(512)) + (r625_row[:, None] * $(128)) + r625_lane[None, :]) / $(128)) % $(4)) * $(128)) + ((((r625_group % $(8)) * $(512)) + (r625_row[:, None] * $(128)) + r625_lane[None, :]) % $(128))), dtype=tl.float32)
      r625_right = tl.load($(candidate_vto_copy_171_ptr) + (((((((r625_group % $(8)) * $(512)) + (r625_lane[:, None] * $(4)) + r625_col[None, :]) / $(512)) % $(8)) * $(512)) + (((((r625_group % $(8)) * $(512)) + (r625_lane[:, None] * $(4)) + r625_col[None, :]) % $(4)) * $(128)) + (((((r625_group % $(8)) * $(512)) + (r625_lane[:, None] * $(4)) + r625_col[None, :]) / $(4)) % $(128))), dtype=tl.float32)
      r625_value = tl.dot(r625_left, r625_right)
      r625_value = r625_value.to(tl.float32)
      tl.store($(candidate_vbmm_93_ptr) + ((r625_group * $(16)) + (r625_row[:, None] * $(4)) + r625_col[None, :]), r625_value)
    }
    r626_z = tl.arange($(128))
    r626_value = (tl.load($(candidate_vbmm_93_ptr) + ((((r626_z / $(16)) % $(8)) * $(16)) + (((r626_z / $(4)) % $(4)) * $(4)) + (r626_z % $(4))), dtype=tl.float32) - tl.load($(candidate_vsum_14_ptr) + ((((r626_z / $(16)) % $(8)) * $(4)) + ((r626_z / $(4)) % $(4))), dtype=tl.float32))
    r626_value = r626_value.to(tl.float32)
    r626_value = tl.broadcast_to(r626_value, [$(128)])
    tl.store($(candidate_vsub_53_ptr) + r626_z, r626_value)
    r627_z = tl.arange($(128))
    r627_value = (tl.load($(candidate_vexp_10_ptr) + ((((r627_z / $(16)) % $(8)) * $(16)) + (((r627_z / $(4)) % $(4)) * $(4)) + (r627_z % $(4))), dtype=tl.float32) * tl.load($(candidate_vsub_53_ptr) + ((((r627_z / $(16)) % $(8)) * $(16)) + (((r627_z / $(4)) % $(4)) * $(4)) + (r627_z % $(4))), dtype=tl.float32))
    r627_value = r627_value.to(tl.float32)
    r627_value = tl.broadcast_to(r627_value, [$(128)])
    tl.store($(candidate_vmul_87_ptr) + r627_z, r627_value)
    r628_row = tl.arange($(4))
    r628_col = tl.arange($(192))
    r628_lane = tl.arange($(4))
    for r628_group in range($(8)) {
      r628_left = tl.load($(candidate_vmul_87_ptr) + (((((((r628_group % $(8)) * $(16)) + (r628_row[:, None] * $(4)) + r628_lane[None, :]) / $(16)) % $(8)) * $(16)) + ((((((r628_group % $(8)) * $(16)) + (r628_row[:, None] * $(4)) + r628_lane[None, :]) / $(4)) % $(4)) * $(4)) + ((((r628_group % $(8)) * $(16)) + (r628_row[:, None] * $(4)) + r628_lane[None, :]) % $(4))), dtype=tl.float32)
      r628_right = tl.load($(candidate_vto_copy_170_ptr) + (((((((r628_group % $(8)) * $(768)) + (r628_lane[:, None] * $(192)) + r628_col[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r628_group % $(8)) * $(768)) + (r628_lane[:, None] * $(192)) + r628_col[None, :]) / $(192)) % $(4)) * $(192)) + ((((r628_group % $(8)) * $(768)) + (r628_lane[:, None] * $(192)) + r628_col[None, :]) % $(192))), dtype=tl.float32)
      r628_value = tl.dot(r628_left, r628_right)
      r628_value = r628_value.to(tl.float32)
      tl.store($(candidate_vbmm_94_ptr) + ((r628_group * $(768)) + (r628_row[:, None] * $(192)) + r628_col[None, :]), r628_value)
    }
    r629_z = tl.arange($(6144))
    r629_value = (tl.load($(candidate_vbmm_94_ptr) + ((((r629_z / $(768)) % $(8)) * $(768)) + (((r629_z / $(192)) % $(4)) * $(192)) + (r629_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r629_value = r629_value.to(tl.float32)
    r629_value = tl.broadcast_to(r629_value, [$(6144)])
    tl.store($(candidate_vmul_88_ptr) + r629_z, r629_value)
    r630_z = tl.arange($(6144))
    r630_value = tl.load($(candidate_vmul_88_ptr) + ((((r630_z / $(192)) % $(8)) * $(768)) + (((r630_z / $(1536)) % $(4)) * $(192)) + (r630_z % $(192))), dtype=tl.float32)
    r630_value = r630_value.to(tl.bfloat16)
    r630_value = tl.broadcast_to(r630_value, [$(6144)])
    tl.store($(candidate_vto_copy_174_ptr) + r630_z, r630_value)
    r631_row = tl.arange($(4))
    r631_col = tl.arange($(192))
    r631_lane = tl.arange($(4))
    for r631_group in range($(8)) {
      r631_left = tl.load($(candidate_vmul_87_ptr) + (((((((r631_group % $(8)) * $(16)) + (r631_row[:, None] * $(4)) + r631_lane[None, :]) / $(16)) % $(8)) * $(16)) + (((((r631_group % $(8)) * $(16)) + (r631_row[:, None] * $(4)) + r631_lane[None, :]) % $(4)) * $(4)) + (((((r631_group % $(8)) * $(16)) + (r631_row[:, None] * $(4)) + r631_lane[None, :]) / $(4)) % $(4))), dtype=tl.float32)
      r631_right = tl.load($(candidate_vto_copy_169_ptr) + (((((((r631_group % $(8)) * $(768)) + (r631_lane[:, None] * $(192)) + r631_col[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r631_group % $(8)) * $(768)) + (r631_lane[:, None] * $(192)) + r631_col[None, :]) / $(192)) % $(4)) * $(192)) + ((((r631_group % $(8)) * $(768)) + (r631_lane[:, None] * $(192)) + r631_col[None, :]) % $(192))), dtype=tl.float32)
      r631_value = tl.dot(r631_left, r631_right)
      r631_value = r631_value.to(tl.float32)
      tl.store($(candidate_vbmm_95_ptr) + ((r631_group * $(768)) + (r631_row[:, None] * $(192)) + r631_col[None, :]), r631_value)
    }
    r632_z = tl.arange($(6144))
    r632_value = (tl.load($(candidate_vbmm_95_ptr) + ((((r632_z / $(768)) % $(8)) * $(768)) + (((r632_z / $(192)) % $(4)) * $(192)) + (r632_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r632_value = r632_value.to(tl.float32)
    r632_value = tl.broadcast_to(r632_value, [$(6144)])
    tl.store($(candidate_vmul_89_ptr) + r632_z, r632_value)
    r633_z = tl.arange($(6144))
    r633_value = tl.load($(candidate_vmul_89_ptr) + ((((r633_z / $(192)) % $(8)) * $(768)) + (((r633_z / $(1536)) % $(4)) * $(192)) + (r633_z % $(192))), dtype=tl.float32)
    r633_value = r633_value.to(tl.bfloat16)
    r633_value = tl.broadcast_to(r633_value, [$(6144)])
    tl.store($(candidate_vto_copy_175_ptr) + r633_z, r633_value)
    r634_row = tl.arange($(4))
    r634_col = tl.arange($(128))
    r634_lane = tl.arange($(4))
    for r634_group in range($(8)) {
      r634_left = tl.load($(candidate_vexp_10_ptr) + (((((((r634_group % $(8)) * $(16)) + (r634_row[:, None] * $(4)) + r634_lane[None, :]) / $(16)) % $(8)) * $(16)) + (((((r634_group % $(8)) * $(16)) + (r634_row[:, None] * $(4)) + r634_lane[None, :]) % $(4)) * $(4)) + (((((r634_group % $(8)) * $(16)) + (r634_row[:, None] * $(4)) + r634_lane[None, :]) / $(4)) % $(4))), dtype=tl.float32)
      r634_right = tl.load($(candidate_vto_copy_172_ptr) + (((((((r634_group % $(8)) * $(512)) + (r634_lane[:, None] * $(128)) + r634_col[None, :]) / $(512)) % $(8)) * $(512)) + ((((((r634_group % $(8)) * $(512)) + (r634_lane[:, None] * $(128)) + r634_col[None, :]) / $(128)) % $(4)) * $(128)) + ((((r634_group % $(8)) * $(512)) + (r634_lane[:, None] * $(128)) + r634_col[None, :]) % $(128))), dtype=tl.float32)
      r634_value = tl.dot(r634_left, r634_right)
      r634_value = r634_value.to(tl.float32)
      tl.store($(candidate_vbmm_96_ptr) + ((r634_group * $(512)) + (r634_row[:, None] * $(128)) + r634_col[None, :]), r634_value)
    }
    r635_z = tl.arange($(4096))
    r635_value = tl.load($(candidate_vbmm_96_ptr) + ((((r635_z / $(128)) % $(8)) * $(512)) + (((r635_z / $(1024)) % $(4)) * $(128)) + (r635_z % $(128))), dtype=tl.float32)
    r635_value = r635_value.to(tl.bfloat16)
    r635_value = tl.broadcast_to(r635_value, [$(4096)])
    tl.store($(candidate_vto_copy_176_ptr) + r635_z, r635_value)
    r636_z = tl.arange($(24576))
    r636_value = tl.where(((r636_z / $(1536)) % $(16)) < $(4), tl.load($(candidate_vto_copy_175_ptr) + ((((((r636_z / $(1536)) % $(16)) - $(0)) % $(4)) * $(1536)) + (((r636_z / $(192)) % $(8)) * $(192)) + (r636_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r636_value = r636_value.to(tl.bfloat16)
    r636_value = tl.broadcast_to(r636_value, [$(24576)])
    tl.store($(candidate_vcat_49_ptr) + r636_z, r636_value)
    r637_z = tl.arange($(16384))
    r637_value = tl.where(((r637_z / $(1024)) % $(16)) < $(4), tl.load($(candidate_vto_copy_176_ptr) + ((((((r637_z / $(1024)) % $(16)) - $(0)) % $(4)) * $(1024)) + (((r637_z / $(128)) % $(8)) * $(128)) + (r637_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r637_value = r637_value.to(tl.bfloat16)
    r637_value = tl.broadcast_to(r637_value, [$(16384)])
    tl.store($(candidate_vcat_50_ptr) + r637_z, r637_value)
    r638_z = tl.arange($(24576))
    r638_value = tl.load($(candidate_vcat_45_ptr) + ((((r638_z / $(192)) % $(16)) * $(1536)) + (((r638_z / $(3072)) % $(8)) * $(192)) + (r638_z % $(192)) + $(24576)), dtype=tl.bfloat16)
    r638_value = r638_value.to(tl.float32)
    r638_value = tl.broadcast_to(r638_value, [$(24576)])
    tl.store($(candidate_vto_copy_177_ptr) + r638_z, r638_value)
    r639_z = tl.arange($(24576))
    r639_value = tl.load($(candidate_vcat_46_ptr) + ((((r639_z / $(192)) % $(16)) * $(1536)) + (((r639_z / $(3072)) % $(8)) * $(192)) + (r639_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r639_value = r639_value.to(tl.float32)
    r639_value = tl.broadcast_to(r639_value, [$(24576)])
    tl.store($(candidate_vto_copy_178_ptr) + r639_z, r639_value)
    r640_z = tl.arange($(16384))
    r640_value = tl.load($(candidate_vbmm_86_ptr) + ((((r640_z / $(128)) % $(16)) * $(1024)) + (((r640_z / $(2048)) % $(8)) * $(128)) + (r640_z % $(128)) + $(65536)), dtype=tl.bfloat16)
    r640_value = r640_value.to(tl.float32)
    r640_value = tl.broadcast_to(r640_value, [$(16384)])
    tl.store($(candidate_vto_copy_179_ptr) + r640_z, r640_value)
    r641_z = tl.arange($(16384))
    r641_value = tl.load($(candidate_vmm_28_ptr) + ((((r641_z / $(128)) % $(16)) * $(1024)) + (((r641_z / $(2048)) % $(8)) * $(128)) + (r641_z % $(128)) + $(16384)), dtype=tl.bfloat16)
    r641_value = r641_value.to(tl.float32)
    r641_value = tl.broadcast_to(r641_value, [$(16384)])
    tl.store($(candidate_vto_copy_180_ptr) + r641_z, r641_value)
    r642_z = tl.arange($(16384))
    r642_value = tl.load($(candidate_vbmm_6_ptr) + ((((r642_z / $(2048)) % $(8)) * $(4096)) + (((r642_z / $(128)) % $(16)) * $(128)) + (r642_z % $(128)) + $(2048)), dtype=tl.bfloat16)
    r642_value = r642_value.to(tl.float32)
    r642_value = tl.broadcast_to(r642_value, [$(16384)])
    tl.store($(candidate_vto_copy_181_ptr) + r642_z, r642_value)
    r643_row = tl.arange($(16))
    r643_col = tl.arange($(16))
    r643_lane = tl.arange($(192))
    for r643_group in range($(8)) {
      r643_left = tl.load($(candidate_vto_copy_177_ptr) + (((((((r643_group % $(8)) * $(3072)) + (r643_row[:, None] * $(192)) + r643_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r643_group % $(8)) * $(3072)) + (r643_row[:, None] * $(192)) + r643_lane[None, :]) / $(192)) % $(16)) * $(192)) + ((((r643_group % $(8)) * $(3072)) + (r643_row[:, None] * $(192)) + r643_lane[None, :]) % $(192))), dtype=tl.float32)
      r643_right = tl.load($(candidate_vto_copy_178_ptr) + (((((((r643_group % $(8)) * $(3072)) + (r643_lane[:, None] * $(16)) + r643_col[None, :]) / $(3072)) % $(8)) * $(3072)) + (((((r643_group % $(8)) * $(3072)) + (r643_lane[:, None] * $(16)) + r643_col[None, :]) % $(16)) * $(192)) + (((((r643_group % $(8)) * $(3072)) + (r643_lane[:, None] * $(16)) + r643_col[None, :]) / $(16)) % $(192))), dtype=tl.float32)
      r643_value = tl.dot(r643_left, r643_right)
      r643_value = r643_value.to(tl.float32)
      tl.store($(candidate_vbmm_97_ptr) + ((r643_group * $(256)) + (r643_row[:, None] * $(16)) + r643_col[None, :]), r643_value)
    }
    r644_z = tl.arange($(2048))
    r644_value = (tl.load($(candidate_vbmm_97_ptr) + ((((r644_z / $(256)) % $(8)) * $(256)) + (((r644_z / $(16)) % $(16)) * $(16)) + (r644_z % $(16))), dtype=tl.float32) * 0.07216878364870322)
    r644_value = r644_value.to(tl.float32)
    r644_value = tl.broadcast_to(r644_value, [$(2048)])
    tl.store($(candidate_vmul_90_ptr) + r644_z, r644_value)
    r645_z = tl.arange($(16))
    r645_value = (tl.cast(r645_z, tl.int64) + tl.full([], $(16), dtype=tl.int64))
    r645_value = tl.broadcast_to(r645_value, [$(16)])
    tl.store($(candidate_vadd_54_ptr) + r645_z, r645_value)
    r646_z = tl.arange($(16))
    r646_value = (tl.load($(candidate_vadd_54_ptr) + r646_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r646_value = tl.broadcast_to(r646_value, [$(16)])
    tl.store($(candidate_vsub_54_ptr) + r646_z, r646_value)
    r647_z = tl.arange($(256))
    r647_value = (tl.cast((r647_z % $(16)), tl.int64) > tl.load($(candidate_vsub_54_ptr) + ((r647_z / $(16)) % $(16)), dtype=tl.int64))
    r647_value = tl.broadcast_to(r647_value, [$(256)])
    tl.store($(candidate_vgt_23_ptr) + r647_z, r647_value)
    r648_z = tl.arange($(2048))
    r648_value = tl.where(tl.load($(candidate_vgt_23_ptr) + ((((r648_z / $(16)) % $(16)) * $(16)) + (r648_z % $(16))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_90_ptr) + ((((r648_z / $(256)) % $(8)) * $(256)) + (((r648_z / $(16)) % $(16)) * $(16)) + (r648_z % $(16))), dtype=tl.float32))
    r648_value = r648_value.to(tl.float32)
    r648_value = tl.broadcast_to(r648_value, [$(2048)])
    tl.store($(candidate_vmasked_fill_23_ptr) + r648_z, r648_value)
    r649_z = tl.arange($(2048))
    r649_value = (tl.load($(candidate_vmasked_fill_23_ptr) + ((((r649_z / $(256)) % $(8)) * $(256)) + (((r649_z / $(16)) % $(16)) * $(16)) + (r649_z % $(16))), dtype=tl.float32) - tl.load($(candidate_vcat_7_ptr) + ((((r649_z / $(16)) % $(16)) * $(8)) + ((r649_z / $(256)) % $(8)) + $(128)), dtype=tl.float32))
    r649_value = r649_value.to(tl.float32)
    r649_value = tl.broadcast_to(r649_value, [$(2048)])
    tl.store($(candidate_vsub_55_ptr) + r649_z, r649_value)
    r650_z = tl.arange($(2048))
    r650_value = tl.exp(tl.load($(candidate_vsub_55_ptr) + ((((r650_z / $(256)) % $(8)) * $(256)) + (((r650_z / $(16)) % $(16)) * $(16)) + (r650_z % $(16))), dtype=tl.float32))
    r650_value = r650_value.to(tl.float32)
    r650_value = tl.broadcast_to(r650_value, [$(2048)])
    tl.store($(candidate_vexp_11_ptr) + r650_z, r650_value)
    r651_z = tl.arange($(16384))
    r651_value = (tl.load($(candidate_vto_copy_180_ptr) + ((((r651_z / $(2048)) % $(8)) * $(2048)) + (((r651_z / $(128)) % $(16)) * $(128)) + (r651_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_181_ptr) + ((((r651_z / $(2048)) % $(8)) * $(2048)) + (((r651_z / $(128)) % $(16)) * $(128)) + (r651_z % $(128))), dtype=tl.float32))
    r651_value = r651_value.to(tl.float32)
    r651_value = tl.broadcast_to(r651_value, [$(16384)])
    tl.store($(candidate_vmul_91_ptr) + r651_z, r651_value)
    r652_z = tl.arange($(128))
    r652_lane = tl.arange($(128))
    r652_value = tl.sum(tl.load($(candidate_vmul_91_ptr) + ((((r652_z[:, None] / $(16)) % $(8)) * $(2048)) + ((r652_z[:, None] % $(16)) * $(128)) + r652_lane[None, :]), dtype=tl.float32), axis=1)
    r652_value = r652_value.to(tl.float32)
    r652_value = tl.broadcast_to(r652_value, [$(128)])
    tl.store($(candidate_vsum_15_ptr) + r652_z, r652_value)
    r653_row = tl.arange($(16))
    r653_col = tl.arange($(16))
    r653_lane = tl.arange($(128))
    for r653_group in range($(8)) {
      r653_left = tl.load($(candidate_vto_copy_180_ptr) + (((((((r653_group % $(8)) * $(2048)) + (r653_row[:, None] * $(128)) + r653_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r653_group % $(8)) * $(2048)) + (r653_row[:, None] * $(128)) + r653_lane[None, :]) / $(128)) % $(16)) * $(128)) + ((((r653_group % $(8)) * $(2048)) + (r653_row[:, None] * $(128)) + r653_lane[None, :]) % $(128))), dtype=tl.float32)
      r653_right = tl.load($(candidate_vto_copy_179_ptr) + (((((((r653_group % $(8)) * $(2048)) + (r653_lane[:, None] * $(16)) + r653_col[None, :]) / $(2048)) % $(8)) * $(2048)) + (((((r653_group % $(8)) * $(2048)) + (r653_lane[:, None] * $(16)) + r653_col[None, :]) % $(16)) * $(128)) + (((((r653_group % $(8)) * $(2048)) + (r653_lane[:, None] * $(16)) + r653_col[None, :]) / $(16)) % $(128))), dtype=tl.float32)
      r653_value = tl.dot(r653_left, r653_right)
      r653_value = r653_value.to(tl.float32)
      tl.store($(candidate_vbmm_98_ptr) + ((r653_group * $(256)) + (r653_row[:, None] * $(16)) + r653_col[None, :]), r653_value)
    }
    r654_z = tl.arange($(2048))
    r654_value = (tl.load($(candidate_vbmm_98_ptr) + ((((r654_z / $(256)) % $(8)) * $(256)) + (((r654_z / $(16)) % $(16)) * $(16)) + (r654_z % $(16))), dtype=tl.float32) - tl.load($(candidate_vsum_15_ptr) + ((((r654_z / $(256)) % $(8)) * $(16)) + ((r654_z / $(16)) % $(16))), dtype=tl.float32))
    r654_value = r654_value.to(tl.float32)
    r654_value = tl.broadcast_to(r654_value, [$(2048)])
    tl.store($(candidate_vsub_56_ptr) + r654_z, r654_value)
    r655_z = tl.arange($(2048))
    r655_value = (tl.load($(candidate_vexp_11_ptr) + ((((r655_z / $(256)) % $(8)) * $(256)) + (((r655_z / $(16)) % $(16)) * $(16)) + (r655_z % $(16))), dtype=tl.float32) * tl.load($(candidate_vsub_56_ptr) + ((((r655_z / $(256)) % $(8)) * $(256)) + (((r655_z / $(16)) % $(16)) * $(16)) + (r655_z % $(16))), dtype=tl.float32))
    r655_value = r655_value.to(tl.float32)
    r655_value = tl.broadcast_to(r655_value, [$(2048)])
    tl.store($(candidate_vmul_92_ptr) + r655_z, r655_value)
    r656_row = tl.arange($(16))
    r656_col = tl.arange($(192))
    r656_lane = tl.arange($(16))
    for r656_group in range($(8)) {
      r656_left = tl.load($(candidate_vmul_92_ptr) + (((((((r656_group % $(8)) * $(256)) + (r656_row[:, None] * $(16)) + r656_lane[None, :]) / $(256)) % $(8)) * $(256)) + ((((((r656_group % $(8)) * $(256)) + (r656_row[:, None] * $(16)) + r656_lane[None, :]) / $(16)) % $(16)) * $(16)) + ((((r656_group % $(8)) * $(256)) + (r656_row[:, None] * $(16)) + r656_lane[None, :]) % $(16))), dtype=tl.float32)
      r656_right = tl.load($(candidate_vto_copy_178_ptr) + (((((((r656_group % $(8)) * $(3072)) + (r656_lane[:, None] * $(192)) + r656_col[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r656_group % $(8)) * $(3072)) + (r656_lane[:, None] * $(192)) + r656_col[None, :]) / $(192)) % $(16)) * $(192)) + ((((r656_group % $(8)) * $(3072)) + (r656_lane[:, None] * $(192)) + r656_col[None, :]) % $(192))), dtype=tl.float32)
      r656_value = tl.dot(r656_left, r656_right)
      r656_value = r656_value.to(tl.float32)
      tl.store($(candidate_vbmm_99_ptr) + ((r656_group * $(3072)) + (r656_row[:, None] * $(192)) + r656_col[None, :]), r656_value)
    }
    r657_z = tl.arange($(24576))
    r657_value = (tl.load($(candidate_vbmm_99_ptr) + ((((r657_z / $(3072)) % $(8)) * $(3072)) + (((r657_z / $(192)) % $(16)) * $(192)) + (r657_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r657_value = r657_value.to(tl.float32)
    r657_value = tl.broadcast_to(r657_value, [$(24576)])
    tl.store($(candidate_vmul_93_ptr) + r657_z, r657_value)
    r658_z = tl.arange($(24576))
    r658_value = tl.load($(candidate_vmul_93_ptr) + ((((r658_z / $(192)) % $(8)) * $(3072)) + (((r658_z / $(1536)) % $(16)) * $(192)) + (r658_z % $(192))), dtype=tl.float32)
    r658_value = r658_value.to(tl.bfloat16)
    r658_value = tl.broadcast_to(r658_value, [$(24576)])
    tl.store($(candidate_vto_copy_182_ptr) + r658_z, r658_value)
    r659_row = tl.arange($(16))
    r659_col = tl.arange($(192))
    r659_lane = tl.arange($(16))
    for r659_group in range($(8)) {
      r659_left = tl.load($(candidate_vmul_92_ptr) + (((((((r659_group % $(8)) * $(256)) + (r659_row[:, None] * $(16)) + r659_lane[None, :]) / $(256)) % $(8)) * $(256)) + (((((r659_group % $(8)) * $(256)) + (r659_row[:, None] * $(16)) + r659_lane[None, :]) % $(16)) * $(16)) + (((((r659_group % $(8)) * $(256)) + (r659_row[:, None] * $(16)) + r659_lane[None, :]) / $(16)) % $(16))), dtype=tl.float32)
      r659_right = tl.load($(candidate_vto_copy_177_ptr) + (((((((r659_group % $(8)) * $(3072)) + (r659_lane[:, None] * $(192)) + r659_col[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r659_group % $(8)) * $(3072)) + (r659_lane[:, None] * $(192)) + r659_col[None, :]) / $(192)) % $(16)) * $(192)) + ((((r659_group % $(8)) * $(3072)) + (r659_lane[:, None] * $(192)) + r659_col[None, :]) % $(192))), dtype=tl.float32)
      r659_value = tl.dot(r659_left, r659_right)
      r659_value = r659_value.to(tl.float32)
      tl.store($(candidate_vbmm_100_ptr) + ((r659_group * $(3072)) + (r659_row[:, None] * $(192)) + r659_col[None, :]), r659_value)
    }
    r660_z = tl.arange($(24576))
    r660_value = (tl.load($(candidate_vbmm_100_ptr) + ((((r660_z / $(3072)) % $(8)) * $(3072)) + (((r660_z / $(192)) % $(16)) * $(192)) + (r660_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r660_value = r660_value.to(tl.float32)
    r660_value = tl.broadcast_to(r660_value, [$(24576)])
    tl.store($(candidate_vmul_94_ptr) + r660_z, r660_value)
    r661_z = tl.arange($(24576))
    r661_value = tl.load($(candidate_vmul_94_ptr) + ((((r661_z / $(192)) % $(8)) * $(3072)) + (((r661_z / $(1536)) % $(16)) * $(192)) + (r661_z % $(192))), dtype=tl.float32)
    r661_value = r661_value.to(tl.bfloat16)
    r661_value = tl.broadcast_to(r661_value, [$(24576)])
    tl.store($(candidate_vto_copy_183_ptr) + r661_z, r661_value)
    r662_row = tl.arange($(16))
    r662_col = tl.arange($(128))
    r662_lane = tl.arange($(16))
    for r662_group in range($(8)) {
      r662_left = tl.load($(candidate_vexp_11_ptr) + (((((((r662_group % $(8)) * $(256)) + (r662_row[:, None] * $(16)) + r662_lane[None, :]) / $(256)) % $(8)) * $(256)) + (((((r662_group % $(8)) * $(256)) + (r662_row[:, None] * $(16)) + r662_lane[None, :]) % $(16)) * $(16)) + (((((r662_group % $(8)) * $(256)) + (r662_row[:, None] * $(16)) + r662_lane[None, :]) / $(16)) % $(16))), dtype=tl.float32)
      r662_right = tl.load($(candidate_vto_copy_180_ptr) + (((((((r662_group % $(8)) * $(2048)) + (r662_lane[:, None] * $(128)) + r662_col[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r662_group % $(8)) * $(2048)) + (r662_lane[:, None] * $(128)) + r662_col[None, :]) / $(128)) % $(16)) * $(128)) + ((((r662_group % $(8)) * $(2048)) + (r662_lane[:, None] * $(128)) + r662_col[None, :]) % $(128))), dtype=tl.float32)
      r662_value = tl.dot(r662_left, r662_right)
      r662_value = r662_value.to(tl.float32)
      tl.store($(candidate_vbmm_101_ptr) + ((r662_group * $(2048)) + (r662_row[:, None] * $(128)) + r662_col[None, :]), r662_value)
    }
    r663_z = tl.arange($(16384))
    r663_value = tl.load($(candidate_vbmm_101_ptr) + ((((r663_z / $(128)) % $(8)) * $(2048)) + (((r663_z / $(1024)) % $(16)) * $(128)) + (r663_z % $(128))), dtype=tl.float32)
    r663_value = r663_value.to(tl.bfloat16)
    r663_value = tl.broadcast_to(r663_value, [$(16384)])
    tl.store($(candidate_vto_copy_184_ptr) + r663_z, r663_value)
    r664_z = tl.arange($(98304))
    r664_value = tl.where(((r664_z / $(1536)) % $(64)) < $(16), tl.load($(candidate_vto_copy_183_ptr) + ((((((r664_z / $(1536)) % $(64)) - $(0)) % $(16)) * $(1536)) + (((r664_z / $(192)) % $(8)) * $(192)) + (r664_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r664_value = r664_value.to(tl.bfloat16)
    r664_value = tl.broadcast_to(r664_value, [$(98304)])
    tl.store($(candidate_vcat_51_ptr) + r664_z, r664_value)
    r665_z = tl.arange($(65536))
    r665_value = tl.where(((r665_z / $(1024)) % $(64)) < $(16), tl.load($(candidate_vto_copy_184_ptr) + ((((((r665_z / $(1024)) % $(64)) - $(0)) % $(16)) * $(1024)) + (((r665_z / $(128)) % $(8)) * $(128)) + (r665_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r665_value = r665_value.to(tl.bfloat16)
    r665_value = tl.broadcast_to(r665_value, [$(65536)])
    tl.store($(candidate_vcat_52_ptr) + r665_z, r665_value)
    r666_z = tl.arange($(49152))
    r666_value = tl.where(((r666_z / $(1536)) % $(32)) < $(12), tl.load($(candidate_vto_copy_166_ptr) + ((((((r666_z / $(1536)) % $(32)) - $(0)) % $(12)) * $(1536)) + (((r666_z / $(192)) % $(8)) * $(192)) + (r666_z % $(192))), dtype=tl.bfloat16), tl.where(((r666_z / $(1536)) % $(32)) < $(16), tl.load($(candidate_vto_copy_174_ptr) + ((((((r666_z / $(1536)) % $(32)) - $(12)) % $(4)) * $(1536)) + (((r666_z / $(192)) % $(8)) * $(192)) + (r666_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vto_copy_182_ptr) + ((((((r666_z / $(1536)) % $(32)) - $(16)) % $(16)) * $(1536)) + (((r666_z / $(192)) % $(8)) * $(192)) + (r666_z % $(192))), dtype=tl.bfloat16)))
    r666_value = r666_value.to(tl.bfloat16)
    r666_value = tl.broadcast_to(r666_value, [$(49152)])
    tl.store($(candidate_vcat_53_ptr) + r666_z, r666_value)
    r667_z = tl.arange($(196608))
    r667_value = tl.where(((r667_z / $(1536)) % $(128)) < $(48), tl.load($(candidate_vcat_47_ptr) + ((((((r667_z / $(1536)) % $(128)) - $(0)) % $(48)) * $(1536)) + (((r667_z / $(192)) % $(8)) * $(192)) + (r667_z % $(192))), dtype=tl.bfloat16), tl.where(((r667_z / $(1536)) % $(128)) < $(64), tl.load($(candidate_vcat_49_ptr) + ((((((r667_z / $(1536)) % $(128)) - $(48)) % $(16)) * $(1536)) + (((r667_z / $(192)) % $(8)) * $(192)) + (r667_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vcat_51_ptr) + ((((((r667_z / $(1536)) % $(128)) - $(64)) % $(64)) * $(1536)) + (((r667_z / $(192)) % $(8)) * $(192)) + (r667_z % $(192))), dtype=tl.bfloat16)))
    r667_value = r667_value.to(tl.bfloat16)
    r667_value = tl.broadcast_to(r667_value, [$(196608)])
    tl.store($(candidate_vcat_54_ptr) + r667_z, r667_value)
    r668_z = tl.arange($(131072))
    r668_value = tl.where(((r668_z / $(1024)) % $(128)) < $(48), tl.load($(candidate_vcat_48_ptr) + ((((((r668_z / $(1024)) % $(128)) - $(0)) % $(48)) * $(1024)) + (((r668_z / $(128)) % $(8)) * $(128)) + (r668_z % $(128))), dtype=tl.bfloat16), tl.where(((r668_z / $(1024)) % $(128)) < $(64), tl.load($(candidate_vcat_50_ptr) + ((((((r668_z / $(1024)) % $(128)) - $(48)) % $(16)) * $(1024)) + (((r668_z / $(128)) % $(8)) * $(128)) + (r668_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_52_ptr) + ((((((r668_z / $(1024)) % $(128)) - $(64)) % $(64)) * $(1024)) + (((r668_z / $(128)) % $(8)) * $(128)) + (r668_z % $(128))), dtype=tl.bfloat16)))
    r668_value = r668_value.to(tl.bfloat16)
    r668_value = tl.broadcast_to(r668_value, [$(131072)])
    tl.store($(candidate_vcat_55_ptr) + r668_z, r668_value)
    r669_z = tl.arange($(8192))
    r669_lane = tl.arange($(8))
    r669_value = tl.sum(tl.load($(candidate_vcat_54_ptr) + ((((r669_z[:, None] / $(64)) % $(128)) * $(1536)) + (r669_lane[None, :] * $(192)) + (r669_z[:, None] % $(64)) + $(128)), dtype=tl.bfloat16), axis=1)
    r669_value = r669_value.to(tl.bfloat16)
    r669_value = tl.broadcast_to(r669_value, [$(8192)])
    tl.store($(candidate_vsum_16_ptr) + r669_z, r669_value)
    r670_row = tl.arange($(128))
    r670_col = tl.arange($(512))
    r670_lane = tl.arange($(1024))
    for r670_group in range($(1)) {
      r670_left = tl.load($(candidate_vcat_54_ptr) + ((((((r670_row[:, None] * $(1024)) + r670_lane[None, :]) / $(1024)) % $(128)) * $(1536)) + ((((r670_row[:, None] * $(1024)) + r670_lane[None, :]) % $(8)) * $(192)) + ((((r670_row[:, None] * $(1024)) + r670_lane[None, :]) / $(8)) % $(128))), dtype=tl.bfloat16)
      r670_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r670_lane[:, None] * $(512)) + r670_col[None, :]) / $(512)) % $(8)) * $(131072)) + (((((r670_lane[:, None] * $(512)) + r670_col[None, :]) / $(4096)) % $(128)) * $(512)) + (((r670_lane[:, None] * $(512)) + r670_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r670_value = tl.dot(r670_left, r670_right)
      r670_value = r670_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_102_ptr) + ((r670_group * $(65536)) + (r670_row[:, None] * $(512)) + r670_col[None, :]), r670_value)
    }
    r671_row = tl.arange($(128))
    r671_col = tl.arange($(512))
    r671_lane = tl.arange($(1024))
    for r671_group in range($(1)) {
      r671_left = tl.load($(candidate_vcat_55_ptr) + ((((((r671_row[:, None] * $(1024)) + r671_lane[None, :]) / $(1024)) % $(128)) * $(1024)) + ((((r671_row[:, None] * $(1024)) + r671_lane[None, :]) % $(8)) * $(128)) + ((((r671_row[:, None] * $(1024)) + r671_lane[None, :]) / $(8)) % $(128))), dtype=tl.bfloat16)
      r671_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r671_lane[:, None] * $(512)) + r671_col[None, :]) / $(512)) % $(8)) * $(131072)) + (((((r671_lane[:, None] * $(512)) + r671_col[None, :]) / $(4096)) % $(128)) * $(512)) + (((r671_lane[:, None] * $(512)) + r671_col[None, :]) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r671_value = tl.dot(r671_left, r671_right)
      r671_value = r671_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_103_ptr) + ((r671_group * $(65536)) + (r671_row[:, None] * $(512)) + r671_col[None, :]), r671_value)
    }
    r672_z = tl.arange($(65536))
    r672_value = (tl.load($(candidate_vbmm_102_ptr) + ((((r672_z / $(512)) % $(128)) * $(512)) + (r672_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vbmm_103_ptr) + ((((r672_z / $(512)) % $(128)) * $(512)) + (r672_z % $(512))), dtype=tl.bfloat16))
    r672_value = r672_value.to(tl.bfloat16)
    r672_value = tl.broadcast_to(r672_value, [$(65536)])
    tl.store($(candidate_vadd_55_ptr) + r672_z, r672_value)
    r673_z = tl.arange($(65536))
    r673_value = tl.where(((((r673_z / $(512)) % $(128)) >= $(0)) & (((r673_z / $(512)) % $(128)) < $(128)) & ((((r673_z / $(512)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vadd_55_ptr) + ((((((r673_z / $(512)) % $(128)) - $(0)) % $(128)) * $(512)) + (r673_z % $(512))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r673_value = r673_value.to(tl.bfloat16)
    r673_value = tl.broadcast_to(r673_value, [$(65536)])
    tl.store($(candidate_vslice_backward_12_ptr) + r673_z, r673_value)
    r674_z = tl.arange($(65536))
    r674_value = (tl.load($(candidate_vadd_48_ptr) + ((((r674_z / $(512)) % $(128)) * $(512)) + (r674_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_12_ptr) + ((((r674_z / $(512)) % $(128)) * $(512)) + (r674_z % $(512))), dtype=tl.bfloat16))
    r674_value = r674_value.to(tl.bfloat16)
    r674_value = tl.broadcast_to(r674_value, [$(65536)])
    tl.store($(candidate_vadd_56_ptr) + r674_z, r674_value)
    r675_z = tl.arange($(8192))
    r675_value = tl.where(((((r675_z / $(64)) % $(128)) >= $(0)) & (((r675_z / $(64)) % $(128)) < $(128)) & ((((r675_z / $(64)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vsum_16_ptr) + ((((((r675_z / $(64)) % $(128)) - $(0)) % $(128)) * $(64)) + (r675_z % $(64))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r675_value = r675_value.to(tl.bfloat16)
    r675_value = tl.broadcast_to(r675_value, [$(8192)])
    tl.store($(candidate_vslice_backward_13_ptr) + r675_z, r675_value)
    r676_z = tl.arange($(8192))
    r676_value = (tl.load($(candidate_vadd_49_ptr) + ((((r676_z / $(64)) % $(128)) * $(64)) + (r676_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_13_ptr) + ((((r676_z / $(64)) % $(128)) * $(64)) + (r676_z % $(64))), dtype=tl.bfloat16))
    r676_value = r676_value.to(tl.bfloat16)
    r676_value = tl.broadcast_to(r676_value, [$(8192)])
    tl.store($(candidate_vadd_57_ptr) + r676_z, r676_value)
    r677_z = tl.arange($(131072))
    r677_value = tl.where(((((r677_z / $(1024)) % $(128)) >= $(0)) & (((r677_z / $(1024)) % $(128)) < $(32)) & ((((r677_z / $(1024)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vcat_53_ptr) + ((((((r677_z / $(1024)) % $(128)) - $(0)) % $(32)) * $(1536)) + (((r677_z / $(128)) % $(8)) * $(192)) + (r677_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r677_value = r677_value.to(tl.bfloat16)
    r677_value = tl.broadcast_to(r677_value, [$(131072)])
    tl.store($(candidate_vslice_backward_14_ptr) + r677_z, r677_value)
    r678_z = tl.arange($(131072))
    r678_value = (tl.load($(candidate_vadd_50_ptr) + ((((r678_z / $(1024)) % $(128)) * $(1024)) + (((r678_z / $(128)) % $(8)) * $(128)) + (r678_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_14_ptr) + ((((r678_z / $(1024)) % $(128)) * $(1024)) + (((r678_z / $(128)) % $(8)) * $(128)) + (r678_z % $(128))), dtype=tl.bfloat16))
    r678_value = r678_value.to(tl.bfloat16)
    r678_value = tl.broadcast_to(r678_value, [$(131072)])
    tl.store($(candidate_vadd_58_ptr) + r678_z, r678_value)
    r679_z = tl.arange($(65536))
    r679_value = tl.where(((((r679_z / $(512)) % $(128)) >= $(0)) & (((r679_z / $(512)) % $(128)) < $(32)) & ((((r679_z / $(512)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vcat_53_ptr) + ((((((r679_z / $(512)) % $(128)) - $(0)) % $(32)) * $(1536)) + (((r679_z / $(64)) % $(8)) * $(192)) + (r679_z % $(64)) + $(128)), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r679_value = r679_value.to(tl.bfloat16)
    r679_value = tl.broadcast_to(r679_value, [$(65536)])
    tl.store($(candidate_vslice_backward_15_ptr) + r679_z, r679_value)
    r680_z = tl.arange($(65536))
    r680_value = (tl.load($(candidate_vadd_51_ptr) + ((((r680_z / $(512)) % $(128)) * $(512)) + (((r680_z / $(64)) % $(8)) * $(64)) + (r680_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_15_ptr) + ((((r680_z / $(512)) % $(128)) * $(512)) + (((r680_z / $(64)) % $(8)) * $(64)) + (r680_z % $(64))), dtype=tl.bfloat16))
    r680_value = r680_value.to(tl.bfloat16)
    r680_value = tl.broadcast_to(r680_value, [$(65536)])
    tl.store($(candidate_vadd_59_ptr) + r680_z, r680_value)
    r681_z = tl.arange($(73728))
    r681_value = tl.where((r681_z % $(576)) < $(512), tl.load($(candidate_vadd_56_ptr) + ((((r681_z / $(576)) % $(128)) * $(512)) + (((r681_z % $(576)) - $(0)) % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vadd_57_ptr) + ((((r681_z / $(576)) % $(128)) * $(64)) + (((r681_z % $(576)) - $(512)) % $(64))), dtype=tl.bfloat16))
    r681_value = r681_value.to(tl.bfloat16)
    r681_value = tl.broadcast_to(r681_value, [$(73728)])
    tl.store($(candidate_vcat_57_ptr) + r681_z, r681_value)
    r682_z = tl.arange($(73728))
    r682_value = tl.cast(0.0, tl.bfloat16)
    for r682_lane in range($(128)) {
      r682_value = tl.where(tl.load($(candidate_vsort_index_ptr) + (r682_lane % $(128)), dtype=tl.int64) == ((r682_z / $(576)) % $(128)), r682_value + tl.load($(candidate_vcat_57_ptr) + (((r682_lane % $(128)) * $(576)) + (r682_z % $(576))), dtype=tl.bfloat16), r682_value)
    }
    r682_value = r682_value.to(tl.bfloat16)
    r682_value = tl.broadcast_to(r682_value, [$(73728)])
    tl.store($(candidate_vindex_put_1_ptr) + r682_z, r682_value)
    r683_z = tl.arange($(8192))
    r683_value = (tl.load($(candidate_vindex_put_1_ptr) + ((((r683_z / $(64)) % $(128)) * $(576)) + (r683_z % $(64)) + $(512)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_28_ptr) + ((((r683_z / $(64)) % $(128)) * $(64)) + (r683_z % $(64))), dtype=tl.bfloat16))
    r683_value = r683_value.to(tl.bfloat16)
    r683_value = tl.broadcast_to(r683_value, [$(8192)])
    tl.store($(candidate_vmul_95_ptr) + r683_z, r683_value)
    r684_z = tl.arange($(8192))
    r684_value = (tl.load($(candidate_vindex_put_1_ptr) + ((((r684_z / $(64)) % $(128)) * $(576)) + (r684_z % $(64)) + $(512)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_27_ptr) + ((((r684_z / $(64)) % $(128)) * $(64)) + (r684_z % $(64))), dtype=tl.bfloat16))
    r684_value = r684_value.to(tl.bfloat16)
    r684_value = tl.broadcast_to(r684_value, [$(8192)])
    tl.store($(candidate_vmul_96_ptr) + r684_z, r684_value)
    r685_z = tl.arange($(8192))
    r685_value = tl.where(((r685_z % $(2)) == $(0)), tl.load($(candidate_vmul_95_ptr) + ((((r685_z / $(64)) % $(128)) * $(64)) + (((r685_z / $(2)) % $(32)) * $(2)) + $(1)), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r685_value = r685_value.to(tl.bfloat16)
    r685_value = tl.broadcast_to(r685_value, [$(8192)])
    tl.store($(candidate_vselect_backward_ptr) + r685_z, r685_value)
    r686_z = tl.arange($(4096))
    r686_value = -tl.load($(candidate_vmul_95_ptr) + ((((r686_z / $(32)) % $(128)) * $(64)) + ((r686_z % $(32)) * $(2))), dtype=tl.bfloat16)
    r686_value = r686_value.to(tl.bfloat16)
    r686_value = tl.broadcast_to(r686_value, [$(4096)])
    tl.store($(candidate_vneg_2_ptr) + r686_z, r686_value)
    r687_z = tl.arange($(8192))
    r687_value = tl.where(((r687_z % $(2)) == $(1)), tl.load($(candidate_vneg_2_ptr) + ((((r687_z / $(64)) % $(128)) * $(32)) + ((r687_z / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r687_value = r687_value.to(tl.bfloat16)
    r687_value = tl.broadcast_to(r687_value, [$(8192)])
    tl.store($(candidate_vselect_backward_1_ptr) + r687_z, r687_value)
    r688_z = tl.arange($(8192))
    r688_value = (tl.load($(candidate_vselect_backward_ptr) + ((((r688_z / $(64)) % $(128)) * $(64)) + (((r688_z / $(2)) % $(32)) * $(2)) + (r688_z % $(2))), dtype=tl.bfloat16) + tl.load($(candidate_vselect_backward_1_ptr) + ((((r688_z / $(64)) % $(128)) * $(64)) + (((r688_z / $(2)) % $(32)) * $(2)) + (r688_z % $(2))), dtype=tl.bfloat16))
    r688_value = r688_value.to(tl.bfloat16)
    r688_value = tl.broadcast_to(r688_value, [$(8192)])
    tl.store($(candidate_vadd_60_ptr) + r688_z, r688_value)
    r689_z = tl.arange($(8192))
    r689_value = (tl.load($(candidate_vmul_96_ptr) + ((((r689_z / $(64)) % $(128)) * $(64)) + (r689_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vadd_60_ptr) + ((((r689_z / $(64)) % $(128)) * $(64)) + (((((((r689_z / $(64)) % $(128)) * $(64)) + (r689_z % $(64))) / $(2)) % $(32)) * $(2)) + (((((r689_z / $(64)) % $(128)) * $(64)) + (r689_z % $(64))) % $(2))), dtype=tl.bfloat16))
    r689_value = r689_value.to(tl.bfloat16)
    r689_value = tl.broadcast_to(r689_value, [$(8192)])
    tl.store($(candidate_vadd_61_ptr) + r689_z, r689_value)
    r690_z = tl.arange($(65536))
    r690_value = (tl.load($(candidate_vadd_59_ptr) + ((((r690_z / $(512)) % $(128)) * $(512)) + (((r690_z / $(64)) % $(8)) * $(64)) + (r690_z % $(64))), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_26_ptr) + ((((r690_z / $(512)) % $(128)) * $(64)) + (r690_z % $(64))), dtype=tl.bfloat16))
    r690_value = r690_value.to(tl.bfloat16)
    r690_value = tl.broadcast_to(r690_value, [$(65536)])
    tl.store($(candidate_vmul_97_ptr) + r690_z, r690_value)
    r691_z = tl.arange($(65536))
    r691_value = (tl.load($(candidate_vadd_59_ptr) + ((((r691_z / $(512)) % $(128)) * $(512)) + (((r691_z / $(64)) % $(8)) * $(64)) + (r691_z % $(64))), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_25_ptr) + ((((r691_z / $(512)) % $(128)) * $(64)) + (r691_z % $(64))), dtype=tl.bfloat16))
    r691_value = r691_value.to(tl.bfloat16)
    r691_value = tl.broadcast_to(r691_value, [$(65536)])
    tl.store($(candidate_vmul_98_ptr) + r691_z, r691_value)
    r692_z = tl.arange($(65536))
    r692_value = tl.where(((r692_z % $(2)) == $(0)), tl.load($(candidate_vmul_97_ptr) + ((((r692_z / $(512)) % $(128)) * $(512)) + (((r692_z / $(64)) % $(8)) * $(64)) + (((r692_z / $(2)) % $(32)) * $(2)) + $(1)), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r692_value = r692_value.to(tl.bfloat16)
    r692_value = tl.broadcast_to(r692_value, [$(65536)])
    tl.store($(candidate_vselect_backward_2_ptr) + r692_z, r692_value)
    r693_z = tl.arange($(32768))
    r693_value = -tl.load($(candidate_vmul_97_ptr) + ((((r693_z / $(256)) % $(128)) * $(512)) + (((r693_z / $(32)) % $(8)) * $(64)) + ((r693_z % $(32)) * $(2))), dtype=tl.bfloat16)
    r693_value = r693_value.to(tl.bfloat16)
    r693_value = tl.broadcast_to(r693_value, [$(32768)])
    tl.store($(candidate_vneg_3_ptr) + r693_z, r693_value)
    r694_z = tl.arange($(65536))
    r694_value = tl.where(((r694_z % $(2)) == $(1)), tl.load($(candidate_vneg_3_ptr) + ((((r694_z / $(512)) % $(128)) * $(256)) + (((r694_z / $(64)) % $(8)) * $(32)) + ((r694_z / $(2)) % $(32))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r694_value = r694_value.to(tl.bfloat16)
    r694_value = tl.broadcast_to(r694_value, [$(65536)])
    tl.store($(candidate_vselect_backward_3_ptr) + r694_z, r694_value)
    r695_z = tl.arange($(65536))
    r695_value = (tl.load($(candidate_vselect_backward_2_ptr) + ((((r695_z / $(512)) % $(128)) * $(512)) + (((r695_z / $(64)) % $(8)) * $(64)) + (((r695_z / $(2)) % $(32)) * $(2)) + (r695_z % $(2))), dtype=tl.bfloat16) + tl.load($(candidate_vselect_backward_3_ptr) + ((((r695_z / $(512)) % $(128)) * $(512)) + (((r695_z / $(64)) % $(8)) * $(64)) + (((r695_z / $(2)) % $(32)) * $(2)) + (r695_z % $(2))), dtype=tl.bfloat16))
    r695_value = r695_value.to(tl.bfloat16)
    r695_value = tl.broadcast_to(r695_value, [$(65536)])
    tl.store($(candidate_vadd_62_ptr) + r695_z, r695_value)
    r696_z = tl.arange($(65536))
    r696_value = (tl.load($(candidate_vmul_98_ptr) + ((((r696_z / $(512)) % $(128)) * $(512)) + (((r696_z / $(64)) % $(8)) * $(64)) + (r696_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vadd_62_ptr) + ((((r696_z / $(512)) % $(128)) * $(512)) + (((r696_z / $(64)) % $(8)) * $(64)) + (((((((r696_z / $(512)) % $(128)) * $(512)) + (((r696_z / $(64)) % $(8)) * $(64)) + (r696_z % $(64))) / $(2)) % $(32)) * $(2)) + (((((r696_z / $(512)) % $(128)) * $(512)) + (((r696_z / $(64)) % $(8)) * $(64)) + (r696_z % $(64))) % $(2))), dtype=tl.bfloat16))
    r696_value = r696_value.to(tl.bfloat16)
    r696_value = tl.broadcast_to(r696_value, [$(65536)])
    tl.store($(candidate_vadd_63_ptr) + r696_z, r696_value)
    r697_z = tl.arange($(196608))
    r697_value = tl.where((r697_z % $(192)) < $(128), tl.load($(candidate_vadd_58_ptr) + ((((((((r697_z / $(192)) % $(128)) * $(1024)) + (((r697_z / $(24576)) % $(8)) * $(128)) + ((r697_z % $(192)) - $(0))) / $(1024)) % $(128)) * $(1024)) + (((((((r697_z / $(192)) % $(128)) * $(1024)) + (((r697_z / $(24576)) % $(8)) * $(128)) + ((r697_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r697_z / $(192)) % $(128)) * $(1024)) + (((r697_z / $(24576)) % $(8)) * $(128)) + ((r697_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vadd_63_ptr) + ((((r697_z / $(192)) % $(128)) * $(512)) + (((r697_z / $(24576)) % $(8)) * $(64)) + (((r697_z % $(192)) - $(128)) % $(64))), dtype=tl.bfloat16))
    r697_value = r697_value.to(tl.bfloat16)
    r697_value = tl.broadcast_to(r697_value, [$(196608)])
    tl.store($(candidate_vcat_58_ptr) + r697_z, r697_value)
    r698_z = tl.arange($(65536))
    r698_value = tl.load($(candidate_vindex_put_1_ptr) + ((((r698_z / $(512)) % $(128)) * $(576)) + (r698_z % $(512))), dtype=tl.bfloat16)
    r698_value = r698_value.to(tl.float32)
    r698_value = tl.broadcast_to(r698_value, [$(65536)])
    tl.store($(candidate_vto_copy_185_ptr) + r698_z, r698_value)
    r699_z = tl.arange($(65536))
    r699_value = (tl.load($(candidate_vto_copy_185_ptr) + ((((r699_z / $(512)) % $(128)) * $(512)) + (r699_z % $(512))), dtype=tl.float32) * tl.cast(tl.load($(candidate_vto_copy_4_ptr) + (r699_z % $(512)), dtype=tl.bfloat16), tl.float32))
    r699_value = r699_value.to(tl.float32)
    r699_value = tl.broadcast_to(r699_value, [$(65536)])
    tl.store($(candidate_vmul_99_ptr) + r699_z, r699_value)
    r700_z = tl.arange($(65536))
    r700_value = (tl.load($(candidate_vmul_99_ptr) + ((((r700_z / $(512)) % $(128)) * $(512)) + (r700_z % $(512))), dtype=tl.float32) * tl.load($(candidate_vto_copy_22_ptr) + ((((r700_z / $(512)) % $(128)) * $(512)) + (r700_z % $(512))), dtype=tl.float32))
    r700_value = r700_value.to(tl.float32)
    r700_value = tl.broadcast_to(r700_value, [$(65536)])
    tl.store($(candidate_vmul_100_ptr) + r700_z, r700_value)
    r701_z = tl.arange($(65536))
    r701_value = (tl.load($(candidate_vmul_99_ptr) + ((((r701_z / $(512)) % $(128)) * $(512)) + (r701_z % $(512))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_1_ptr) + ((r701_z / $(512)) % $(128)), dtype=tl.float32))
    r701_value = r701_value.to(tl.float32)
    r701_value = tl.broadcast_to(r701_value, [$(65536)])
    tl.store($(candidate_vmul_101_ptr) + r701_z, r701_value)
    r702_z = tl.arange($(128))
    r702_lane = tl.arange($(512))
    r702_value = tl.sum(tl.load($(candidate_vmul_100_ptr) + ((r702_z[:, None] * $(512)) + r702_lane[None, :]), dtype=tl.float32), axis=1)
    r702_value = r702_value.to(tl.float32)
    r702_value = tl.broadcast_to(r702_value, [$(128)])
    tl.store($(candidate_vsum_17_ptr) + r702_z, r702_value)
    r703_z = tl.arange($(128))
    r703_value = ((tl.load($(candidate_vrsqrt_1_ptr) + r703_z, dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_1_ptr) + r703_z, dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_1_ptr) + r703_z, dtype=tl.float32)))
    r703_value = r703_value.to(tl.float32)
    r703_value = tl.broadcast_to(r703_value, [$(128)])
    tl.store($(candidate_vpow_6_ptr) + r703_z, r703_value)
    r704_z = tl.arange($(128))
    r704_value = (tl.load($(candidate_vsum_17_ptr) + r704_z, dtype=tl.float32) * -0.5)
    r704_value = r704_value.to(tl.float32)
    r704_value = tl.broadcast_to(r704_value, [$(128)])
    tl.store($(candidate_vmul_102_ptr) + r704_z, r704_value)
    r705_z = tl.arange($(128))
    r705_value = (tl.load($(candidate_vmul_102_ptr) + r705_z, dtype=tl.float32) * tl.load($(candidate_vpow_6_ptr) + r705_z, dtype=tl.float32))
    r705_value = r705_value.to(tl.float32)
    r705_value = tl.broadcast_to(r705_value, [$(128)])
    tl.store($(candidate_vmul_103_ptr) + r705_z, r705_value)
    r706_z = tl.arange($(65536))
    r706_value = (tl.load($(candidate_vmul_103_ptr) + ((r706_z / $(512)) % $(128)), dtype=tl.float32) / 512.0)
    r706_value = r706_value.to(tl.float32)
    r706_value = tl.broadcast_to(r706_value, [$(65536)])
    tl.store($(candidate_vdiv_3_ptr) + r706_z, r706_value)
    r707_z = tl.arange($(65536))
    r707_value = ((tl.load($(candidate_vto_copy_22_ptr) + ((((r707_z / $(512)) % $(128)) * $(512)) + (r707_z % $(512))), dtype=tl.float32)))
    r707_value = r707_value.to(tl.float32)
    r707_value = tl.broadcast_to(r707_value, [$(65536)])
    tl.store($(candidate_vpow_7_ptr) + r707_z, r707_value)
    r708_z = tl.arange($(65536))
    r708_value = (tl.load($(candidate_vpow_7_ptr) + ((((r708_z / $(512)) % $(128)) * $(512)) + (r708_z % $(512))), dtype=tl.float32) * 2.0)
    r708_value = r708_value.to(tl.float32)
    r708_value = tl.broadcast_to(r708_value, [$(65536)])
    tl.store($(candidate_vmul_104_ptr) + r708_z, r708_value)
    r709_z = tl.arange($(65536))
    r709_value = (tl.load($(candidate_vdiv_3_ptr) + ((((r709_z / $(512)) % $(128)) * $(512)) + (r709_z % $(512))), dtype=tl.float32) * tl.load($(candidate_vmul_104_ptr) + ((((r709_z / $(512)) % $(128)) * $(512)) + (r709_z % $(512))), dtype=tl.float32))
    r709_value = r709_value.to(tl.float32)
    r709_value = tl.broadcast_to(r709_value, [$(65536)])
    tl.store($(candidate_vmul_105_ptr) + r709_z, r709_value)
    r710_z = tl.arange($(65536))
    r710_value = (tl.load($(candidate_vmul_101_ptr) + ((((r710_z / $(512)) % $(128)) * $(512)) + (r710_z % $(512))), dtype=tl.float32) + tl.load($(candidate_vmul_105_ptr) + ((((r710_z / $(512)) % $(128)) * $(512)) + (r710_z % $(512))), dtype=tl.float32))
    r710_value = r710_value.to(tl.float32)
    r710_value = tl.broadcast_to(r710_value, [$(65536)])
    tl.store($(candidate_vadd_64_ptr) + r710_z, r710_value)
    r711_z = tl.arange($(65536))
    r711_value = tl.load($(candidate_vadd_64_ptr) + ((((r711_z / $(512)) % $(128)) * $(512)) + (r711_z % $(512))), dtype=tl.float32)
    r711_value = r711_value.to(tl.bfloat16)
    r711_value = tl.broadcast_to(r711_value, [$(65536)])
    tl.store($(candidate_vto_copy_186_ptr) + r711_z, r711_value)
    r712_z = tl.arange($(73728))
    r712_value = tl.where((r712_z % $(576)) < $(512), tl.load($(candidate_vto_copy_186_ptr) + ((((r712_z / $(576)) % $(128)) * $(512)) + (((r712_z % $(576)) - $(0)) % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vadd_61_ptr) + ((((((((r712_z / $(576)) % $(128)) * $(64)) + ((r712_z % $(576)) - $(512))) / $(64)) % $(128)) * $(64)) + (((((r712_z / $(576)) % $(128)) * $(64)) + ((r712_z % $(576)) - $(512))) % $(64))), dtype=tl.bfloat16))
    r712_value = r712_value.to(tl.bfloat16)
    r712_value = tl.broadcast_to(r712_value, [$(73728)])
    tl.store($(candidate_vcat_59_ptr) + r712_z, r712_value)
    r713_row = tl.arange($(128))
    r713_col = tl.arange($(512))
    r713_lane = tl.arange($(576))
    r713_left = tl.load($(candidate_vcat_59_ptr) + ((((((r713_row[:, None] * $(576)) + r713_lane[None, :]) / $(576)) % $(128)) * $(576)) + (((r713_row[:, None] * $(576)) + r713_lane[None, :]) % $(576))), dtype=tl.bfloat16)
    r713_right = tl.load($(candidate_vto_copy_3_ptr) + (((r713_lane[:, None] % $(576)) * $(512)) + (r713_col[None, :] % $(512))), dtype=tl.bfloat16)
    r713_value = tl.dot(r713_left, r713_right)
    r713_value = r713_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_29_ptr) + ((r713_row[:, None] * $(512)) + r713_col[None, :]), r713_value)
    r714_row = tl.arange($(128))
    r714_col = tl.arange($(256))
    r714_lane = tl.arange($(1536))
    r714_left = tl.load($(candidate_vcat_58_ptr) + ((((((((((r714_row[:, None] * $(1536)) + r714_lane[None, :]) / $(1536)) % $(128)) * $(1536)) + (((r714_row[:, None] * $(1536)) + r714_lane[None, :]) % $(1536))) / $(192)) % $(8)) * $(24576)) + (((((r714_row[:, None] * $(1536)) + r714_lane[None, :]) / $(1536)) % $(128)) * $(192)) + (((((((r714_row[:, None] * $(1536)) + r714_lane[None, :]) / $(1536)) % $(128)) * $(1536)) + (((r714_row[:, None] * $(1536)) + r714_lane[None, :]) % $(1536))) % $(192))), dtype=tl.bfloat16)
    r714_right = tl.load($(candidate_vto_copy_2_ptr) + (((r714_lane[:, None] % $(1536)) * $(256)) + (r714_col[None, :] % $(256))), dtype=tl.bfloat16)
    r714_value = tl.dot(r714_left, r714_right)
    r714_value = r714_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_30_ptr) + ((r714_row[:, None] * $(256)) + r714_col[None, :]), r714_value)
    r715_z = tl.arange($(32768))
    r715_value = tl.load($(candidate_vmm_30_ptr) + ((((r715_z / $(256)) % $(128)) * $(256)) + (r715_z % $(256))), dtype=tl.bfloat16)
    r715_value = r715_value.to(tl.float32)
    r715_value = tl.broadcast_to(r715_value, [$(32768)])
    tl.store($(candidate_vto_copy_187_ptr) + r715_z, r715_value)
    r716_z = tl.arange($(32768))
    r716_value = (tl.load($(candidate_vto_copy_187_ptr) + ((((r716_z / $(256)) % $(128)) * $(256)) + (r716_z % $(256))), dtype=tl.float32) * tl.cast(tl.load($(candidate_vto_copy_1_ptr) + (r716_z % $(256)), dtype=tl.bfloat16), tl.float32))
    r716_value = r716_value.to(tl.float32)
    r716_value = tl.broadcast_to(r716_value, [$(32768)])
    tl.store($(candidate_vmul_106_ptr) + r716_z, r716_value)
    r717_z = tl.arange($(32768))
    r717_value = (tl.load($(candidate_vmul_106_ptr) + ((((r717_z / $(256)) % $(128)) * $(256)) + (r717_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vto_copy_20_ptr) + ((((r717_z / $(256)) % $(128)) * $(256)) + (r717_z % $(256))), dtype=tl.float32))
    r717_value = r717_value.to(tl.float32)
    r717_value = tl.broadcast_to(r717_value, [$(32768)])
    tl.store($(candidate_vmul_107_ptr) + r717_z, r717_value)
    r718_z = tl.arange($(32768))
    r718_value = (tl.load($(candidate_vmul_106_ptr) + ((((r718_z / $(256)) % $(128)) * $(256)) + (r718_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_ptr) + ((r718_z / $(256)) % $(128)), dtype=tl.float32))
    r718_value = r718_value.to(tl.float32)
    r718_value = tl.broadcast_to(r718_value, [$(32768)])
    tl.store($(candidate_vmul_108_ptr) + r718_z, r718_value)
    r719_z = tl.arange($(128))
    r719_lane = tl.arange($(256))
    r719_value = tl.sum(tl.load($(candidate_vmul_107_ptr) + ((r719_z[:, None] * $(256)) + r719_lane[None, :]), dtype=tl.float32), axis=1)
    r719_value = r719_value.to(tl.float32)
    r719_value = tl.broadcast_to(r719_value, [$(128)])
    tl.store($(candidate_vsum_18_ptr) + r719_z, r719_value)
    r720_z = tl.arange($(128))
    r720_value = ((tl.load($(candidate_vrsqrt_ptr) + r720_z, dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_ptr) + r720_z, dtype=tl.float32)) * (tl.load($(candidate_vrsqrt_ptr) + r720_z, dtype=tl.float32)))
    r720_value = r720_value.to(tl.float32)
    r720_value = tl.broadcast_to(r720_value, [$(128)])
    tl.store($(candidate_vpow_8_ptr) + r720_z, r720_value)
    r721_z = tl.arange($(128))
    r721_value = (tl.load($(candidate_vsum_18_ptr) + r721_z, dtype=tl.float32) * -0.5)
    r721_value = r721_value.to(tl.float32)
    r721_value = tl.broadcast_to(r721_value, [$(128)])
    tl.store($(candidate_vmul_109_ptr) + r721_z, r721_value)
    r722_z = tl.arange($(128))
    r722_value = (tl.load($(candidate_vmul_109_ptr) + r722_z, dtype=tl.float32) * tl.load($(candidate_vpow_8_ptr) + r722_z, dtype=tl.float32))
    r722_value = r722_value.to(tl.float32)
    r722_value = tl.broadcast_to(r722_value, [$(128)])
    tl.store($(candidate_vmul_110_ptr) + r722_z, r722_value)
    r723_z = tl.arange($(32768))
    r723_value = (tl.load($(candidate_vmul_110_ptr) + ((r723_z / $(256)) % $(128)), dtype=tl.float32) / 256.0)
    r723_value = r723_value.to(tl.float32)
    r723_value = tl.broadcast_to(r723_value, [$(32768)])
    tl.store($(candidate_vdiv_4_ptr) + r723_z, r723_value)
    r724_z = tl.arange($(32768))
    r724_value = ((tl.load($(candidate_vto_copy_20_ptr) + ((((r724_z / $(256)) % $(128)) * $(256)) + (r724_z % $(256))), dtype=tl.float32)))
    r724_value = r724_value.to(tl.float32)
    r724_value = tl.broadcast_to(r724_value, [$(32768)])
    tl.store($(candidate_vpow_9_ptr) + r724_z, r724_value)
    r725_z = tl.arange($(32768))
    r725_value = (tl.load($(candidate_vpow_9_ptr) + ((((r725_z / $(256)) % $(128)) * $(256)) + (r725_z % $(256))), dtype=tl.float32) * 2.0)
    r725_value = r725_value.to(tl.float32)
    r725_value = tl.broadcast_to(r725_value, [$(32768)])
    tl.store($(candidate_vmul_111_ptr) + r725_z, r725_value)
    r726_z = tl.arange($(32768))
    r726_value = (tl.load($(candidate_vdiv_4_ptr) + ((((r726_z / $(256)) % $(128)) * $(256)) + (r726_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vmul_111_ptr) + ((((r726_z / $(256)) % $(128)) * $(256)) + (r726_z % $(256))), dtype=tl.float32))
    r726_value = r726_value.to(tl.float32)
    r726_value = tl.broadcast_to(r726_value, [$(32768)])
    tl.store($(candidate_vmul_112_ptr) + r726_z, r726_value)
    r727_z = tl.arange($(32768))
    r727_value = (tl.load($(candidate_vmul_108_ptr) + ((((r727_z / $(256)) % $(128)) * $(256)) + (r727_z % $(256))), dtype=tl.float32) + tl.load($(candidate_vmul_112_ptr) + ((((r727_z / $(256)) % $(128)) * $(256)) + (r727_z % $(256))), dtype=tl.float32))
    r727_value = r727_value.to(tl.float32)
    r727_value = tl.broadcast_to(r727_value, [$(32768)])
    tl.store($(candidate_vadd_65_ptr) + r727_z, r727_value)
    r728_z = tl.arange($(32768))
    r728_value = tl.load($(candidate_vadd_65_ptr) + ((((r728_z / $(256)) % $(128)) * $(256)) + (r728_z % $(256))), dtype=tl.float32)
    r728_value = r728_value.to(tl.bfloat16)
    r728_value = tl.broadcast_to(r728_value, [$(32768)])
    tl.store($(candidate_vto_copy_188_ptr) + r728_z, r728_value)
    r729_row = tl.arange($(128))
    r729_col = tl.arange($(512))
    r729_lane = tl.arange($(256))
    r729_left = tl.load($(candidate_vto_copy_188_ptr) + ((((((r729_row[:, None] * $(256)) + r729_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r729_row[:, None] * $(256)) + r729_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r729_right = tl.load($(candidate_vto_copy_ptr) + (((r729_lane[:, None] % $(256)) * $(512)) + (r729_col[None, :] % $(512))), dtype=tl.bfloat16)
    r729_value = tl.dot(r729_left, r729_right)
    r729_value = r729_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_31_ptr) + ((r729_row[:, None] * $(512)) + r729_col[None, :]), r729_value)
    r730_z = tl.arange($(65536))
    r730_value = (tl.load($(candidate_vmm_29_ptr) + ((((r730_z / $(512)) % $(128)) * $(512)) + (r730_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vmm_31_ptr) + ((((r730_z / $(512)) % $(128)) * $(512)) + (r730_z % $(512))), dtype=tl.bfloat16))
    r730_value = r730_value.to(tl.bfloat16)
    r730_value = tl.broadcast_to(r730_value, [$(65536)])
    tl.store($(candidate_vadd_66_ptr) + r730_z, r730_value)
    r731_z = tl.arange($(65536))
    r731_value = tl.cast(0.0, tl.bfloat16)
    for r731_lane in range($(128)) {
      r731_value = tl.where(tl.load($(constantee33028f0f_ptr) + r731_lane, dtype=tl.int64) == ((r731_z / $(512)) % $(128)), r731_value + tl.load($(candidate_vadd_66_ptr) + (((r731_lane % $(128)) * $(512)) + (r731_z % $(512))), dtype=tl.bfloat16), r731_value)
    }
    r731_value = r731_value.to(tl.bfloat16)
    r731_value = tl.broadcast_to(r731_value, [$(65536)])
    tl.store($(candidate_vindex_put_2_ptr) + r731_z, r731_value)
    z = tl.arange($(65536))
    value = tl.load($(candidate_vindex_put_2_ptr) + ((((z / $(512)) % $(128)) * $(512)) + (z % $(512))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(65536)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_mla_fa4_te_rope_cp_absorb_reuse_dx
