import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_mla_fa4_te_rope_thd_nonabsorb_wgrad
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
  let vto_copy_20_ptr : RegionName := "vto_copy_20"
  let vmm_ptr : RegionName := "vmm"
  let vto_copy_21_ptr : RegionName := "vto_copy_21"
  let vpow_3_ptr : RegionName := "vpow_3"
  let vmean_ptr : RegionName := "vmean"
  let vadd_3_ptr : RegionName := "vadd_3"
  let vrsqrt_ptr : RegionName := "vrsqrt"
  let vmul_13_ptr : RegionName := "vmul_13"
  let vmul_14_ptr : RegionName := "vmul_14"
  let vto_copy_22_ptr : RegionName := "vto_copy_22"
  let vmm_1_ptr : RegionName := "vmm_1"
  let vmm_2_ptr : RegionName := "vmm_2"
  let vto_copy_23_ptr : RegionName := "vto_copy_23"
  let vpow_4_ptr : RegionName := "vpow_4"
  let vmean_1_ptr : RegionName := "vmean_1"
  let vadd_4_ptr : RegionName := "vadd_4"
  let vrsqrt_1_ptr : RegionName := "vrsqrt_1"
  let vmul_15_ptr : RegionName := "vmul_15"
  let vmul_16_ptr : RegionName := "vmul_16"
  let vto_copy_24_ptr : RegionName := "vto_copy_24"
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
  let vto_copy_25_ptr : RegionName := "vto_copy_25"
  let vsoftmax_ptr : RegionName := "vsoftmax"
  let vto_copy_26_ptr : RegionName := "vto_copy_26"
  let vmm_5_ptr : RegionName := "vmm_5"
  let vbmm_4_ptr : RegionName := "vbmm_4"
  let vbmm_5_ptr : RegionName := "vbmm_5"
  let vto_copy_27_ptr : RegionName := "vto_copy_27"
  let vsoftmax_backward_data_ptr : RegionName := "vsoftmax_backward_data"
  let vto_copy_28_ptr : RegionName := "vto_copy_28"
  let vmul_26_ptr : RegionName := "vmul_26"
  let vbmm_6_ptr : RegionName := "vbmm_6"
  let vcat_9_ptr : RegionName := "vcat_9"
  let vmm_6_ptr : RegionName := "vmm_6"
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
    r14_value = tl.load($(vmul_6_ptr) + ((((r14_z / $(1024)) % $(512)) * $(1024)) + (r14_z % $(1024))), dtype=tl.float32)
    r14_value = r14_value.to(tl.bfloat16)
    r14_value = tl.broadcast_to(r14_value, [$(524288)])
    tl.store($(vto_copy_6_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(128))
    r15_value = tl.where(r15_z < $(48), tl.cast((r15_z - $(0)), tl.int64), tl.where(r15_z < $(64), tl.cast((r15_z - $(48)), tl.int64), tl.cast((r15_z - $(64)), tl.int64)))
    r15_value = tl.broadcast_to(r15_value, [$(128)])
    tl.store($(vcat_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(128))
    r16_value = (tl.load($(vcat_ptr) + r16_z, dtype=tl.int64) + tl.full([], $(0), dtype=tl.int64))
    r16_value = tl.broadcast_to(r16_value, [$(128)])
    tl.store($(vadd_2_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(32))
    r17_value = tl.toReal((tl.cast(r17_z, tl.int64) * tl.full([], $(2), dtype=tl.int64)))
    r17_value = r17_value.to(tl.float32)
    r17_value = tl.broadcast_to(r17_value, [$(32)])
    tl.store($(vto_copy_14_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(32))
    r18_value = (tl.load($(vto_copy_14_ptr) + r18_z, dtype=tl.float32) / 64.0)
    r18_value = r18_value.to(tl.float32)
    r18_value = tl.broadcast_to(r18_value, [$(32)])
    tl.store($(vdiv_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(32))
    r19_value = tl.extra.cuda.libdevice.pow(10000.0, tl.load($(vdiv_ptr) + r19_z, dtype=tl.float32))
    r19_value = r19_value.to(tl.float32)
    r19_value = tl.broadcast_to(r19_value, [$(32)])
    tl.store($(vpow_1_ptr) + r19_z, r19_value)
    r20_z = tl.arange($(32))
    r20_value = 1.0 / tl.load($(vpow_1_ptr) + r20_z, dtype=tl.float32)
    r20_value = r20_value.to(tl.float32)
    r20_value = tl.broadcast_to(r20_value, [$(32)])
    tl.store($(vreciprocal_ptr) + r20_z, r20_value)
    r21_z = tl.arange($(32))
    r21_value = (tl.load($(vreciprocal_ptr) + r21_z, dtype=tl.float32) * 1.0)
    r21_value = r21_value.to(tl.float32)
    r21_value = tl.broadcast_to(r21_value, [$(32)])
    tl.store($(vmul_7_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(128))
    r22_value = tl.toReal(tl.load($(vadd_2_ptr) + r22_z, dtype=tl.int64))
    r22_value = r22_value.to(tl.float32)
    r22_value = tl.broadcast_to(r22_value, [$(128)])
    tl.store($(vto_copy_16_ptr) + r22_z, r22_value)
    r23_row = tl.arange($(32))
    r23_col = tl.arange($(128))
    r23_lane = tl.arange($(1))
    for r23_group in range($(1)) {
      r23_left = tl.load($(vmul_7_ptr) + ((r23_row[:, None] + r23_lane[None, :]) % $(32)), dtype=tl.float32)
      r23_right = tl.load($(vto_copy_16_ptr) + (((r23_lane[:, None] * $(128)) + r23_col[None, :]) % $(128)), dtype=tl.float32)
      r23_value = tl.dot(r23_left, r23_right)
      r23_value = r23_value.to(tl.float32)
      tl.store($(vbmm_ptr) + ((r23_group * $(4096)) + (r23_row[:, None] * $(128)) + r23_col[None, :]), r23_value)
    }
    r24_z = tl.arange($(8192))
    r24_value = tl.where((r24_z % $(64)) < $(32), tl.load($(vbmm_ptr) + (((((r24_z % $(64)) - $(0)) % $(32)) * $(128)) + ((r24_z / $(64)) % $(128))), dtype=tl.float32), tl.load($(vbmm_ptr) + (((((r24_z % $(64)) - $(32)) % $(32)) * $(128)) + ((r24_z / $(64)) % $(128))), dtype=tl.float32))
    r24_value = r24_value.to(tl.float32)
    r24_value = tl.broadcast_to(r24_value, [$(8192)])
    tl.store($(vcat_1_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(8192))
    r25_value = tl.cos(tl.load($(vcat_1_ptr) + ((((r25_z / $(64)) % $(128)) * $(64)) + (r25_z % $(64))), dtype=tl.float32))
    r25_value = r25_value.to(tl.float32)
    r25_value = tl.broadcast_to(r25_value, [$(8192)])
    tl.store($(vcos_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(8192))
    r26_value = (tl.load($(vcos_ptr) + ((((r26_z / $(64)) % $(128)) * $(64)) + (r26_z % $(64))), dtype=tl.float32) * 1.0)
    r26_value = r26_value.to(tl.float32)
    r26_value = tl.broadcast_to(r26_value, [$(8192)])
    tl.store($(vmul_9_ptr) + r26_z, r26_value)
    r27_z = tl.arange($(8192))
    r27_value = tl.sin(tl.load($(vcat_1_ptr) + ((((r27_z / $(64)) % $(128)) * $(64)) + (r27_z % $(64))), dtype=tl.float32))
    r27_value = r27_value.to(tl.float32)
    r27_value = tl.broadcast_to(r27_value, [$(8192)])
    tl.store($(vsin_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(8192))
    r28_value = (tl.load($(vsin_ptr) + ((((r28_z / $(64)) % $(128)) * $(64)) + (r28_z % $(64))), dtype=tl.float32) * 1.0)
    r28_value = r28_value.to(tl.float32)
    r28_value = tl.broadcast_to(r28_value, [$(8192)])
    tl.store($(vmul_10_ptr) + r28_z, r28_value)
    r29_z = tl.arange($(8192))
    r29_value = tl.load($(vmul_9_ptr) + ((((r29_z / $(64)) % $(128)) * $(64)) + (r29_z % $(64))), dtype=tl.float32)
    r29_value = r29_value.to(tl.bfloat16)
    r29_value = tl.broadcast_to(r29_value, [$(8192)])
    tl.store($(vto_copy_17_ptr) + r29_z, r29_value)
    r30_z = tl.arange($(8192))
    r30_value = tl.load($(vmul_10_ptr) + ((((r30_z / $(64)) % $(128)) * $(64)) + (r30_z % $(64))), dtype=tl.float32)
    r30_value = r30_value.to(tl.bfloat16)
    r30_value = tl.broadcast_to(r30_value, [$(8192)])
    tl.store($(vto_copy_18_ptr) + r30_z, r30_value)
    r31_z = tl.arange($(1))
    r31_value = (tl.load($(vadd_2_ptr) + $(0), dtype=tl.int64) - tl.full([], $(1), dtype=tl.int64))
    r31_value = tl.broadcast_to(r31_value, [$(1)])
    tl.store($(vsub_ptr) + r31_z, r31_value)
    r32_z = tl.arange($(129))
    r32_value = tl.where(r32_z < $(1), tl.load($(vsub_ptr) + $(0), dtype=tl.int64), tl.load($(vadd_2_ptr) + ((r32_z - $(1)) % $(128)), dtype=tl.int64))
    r32_value = tl.broadcast_to(r32_value, [$(129)])
    tl.store($(vcat_3_ptr) + r32_z, r32_value)
    r33_z = tl.arange($(128))
    r33_value = (tl.load($(vcat_3_ptr) + (r33_z + $(1)), dtype=tl.int64) - tl.load($(vcat_3_ptr) + r33_z, dtype=tl.int64))
    r33_value = tl.broadcast_to(r33_value, [$(128)])
    tl.store($(vsub_1_ptr) + r33_z, r33_value)
    r34_z = tl.arange($(128))
    r34_value = (tl.load($(vsub_1_ptr) + r34_z, dtype=tl.int64) != tl.full([], $(1), dtype=tl.int64))
    r34_value = tl.broadcast_to(r34_value, [$(128)])
    tl.store($(vne_ptr) + r34_z, r34_value)
    r35_z = tl.arange($(128))
    r35_value = tl.broadcast_to(tl.full([], $(0), dtype=tl.int64), [$(128)])
    for r35_lane in range($(128)) {
      r35_value = tl.where(r35_lane <= r35_z, (r35_value + tl.where(tl.load($(vne_ptr) + (r35_lane % $(128)), dtype=tl.int1), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))), r35_value)
    }
    r35_value = tl.broadcast_to(r35_value, [$(128)])
    tl.store($(vcumsum_1_ptr) + r35_z, r35_value)
    r36_z = tl.arange($(16384))
    r36_value = (tl.load($(vcumsum_1_ptr) + ((r36_z / $(128)) % $(128)), dtype=tl.int64) == tl.load($(vcumsum_1_ptr) + (r36_z % $(128)), dtype=tl.int64))
    r36_value = tl.broadcast_to(r36_value, [$(16384)])
    tl.store($(veq_ptr) + r36_z, r36_value)
    r37_z = tl.arange($(16384))
    r37_value = tl.where((r37_z % $(128)) <= ((r37_z / $(128)) % $(128)) + $(0), true, false)
    r37_value = tl.broadcast_to(r37_value, [$(16384)])
    tl.store($(vtril_ptr) + r37_z, r37_value)
    r38_z = tl.arange($(16384))
    r38_value = (tl.load($(veq_ptr) + ((((r38_z / $(128)) % $(128)) * $(128)) + (r38_z % $(128))), dtype=tl.int1) & tl.load($(vtril_ptr) + ((((r38_z / $(128)) % $(128)) * $(128)) + (r38_z % $(128))), dtype=tl.int1))
    r38_value = tl.broadcast_to(r38_value, [$(16384)])
    tl.store($(vbitwise_and_ptr) + r38_z, r38_value)
    r39_z = tl.arange($(16384))
    r39_value = tl.where(tl.load($(vbitwise_and_ptr) + ((((r39_z / $(128)) % $(128)) * $(128)) + (r39_z % $(128))), dtype=tl.int1), tl.cast(0.0, tl.bfloat16), tl.cast(-338953138925153547590470800371487866880.0, tl.bfloat16))
    r39_value = r39_value.to(tl.bfloat16)
    r39_value = tl.broadcast_to(r39_value, [$(16384)])
    tl.store($(vwhere_ptr) + r39_z, r39_value)
    r40_z = tl.arange($(524288))
    r40_value = tl.load($(vmul_5_ptr) + ((((r40_z / $(256)) % $(2048)) * $(256)) + (r40_z % $(256))), dtype=tl.float32)
    r40_value = r40_value.to(tl.bfloat16)
    r40_value = tl.broadcast_to(r40_value, [$(524288)])
    tl.store($(vto_copy_20_ptr) + r40_z, r40_value)
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
    tl.store($(vto_copy_21_ptr) + r42_z, r42_value)
    r43_z = tl.arange($(32768))
    r43_value = ((tl.load($(vto_copy_21_ptr) + ((((r43_z / $(256)) % $(128)) * $(256)) + (r43_z % $(256))), dtype=tl.float32)) * (tl.load($(vto_copy_21_ptr) + ((((r43_z / $(256)) % $(128)) * $(256)) + (r43_z % $(256))), dtype=tl.float32)))
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
    r47_value = (tl.load($(vto_copy_21_ptr) + ((((r47_z / $(256)) % $(128)) * $(256)) + (r47_z % $(256))), dtype=tl.float32) * tl.load($(vrsqrt_ptr) + ((r47_z / $(256)) % $(128)), dtype=tl.float32))
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
    tl.store($(vto_copy_22_ptr) + r49_z, r49_value)
    r50_row = tl.arange($(128))
    r50_col = tl.arange($(1536))
    r50_lane = tl.arange($(256))
    r50_left = tl.load($(vto_copy_22_ptr) + ((((((r50_row[:, None] * $(256)) + r50_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r50_row[:, None] * $(256)) + r50_lane[None, :]) % $(256))), dtype=tl.bfloat16)
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
    tl.store($(vto_copy_23_ptr) + r52_z, r52_value)
    r53_z = tl.arange($(32768))
    r53_value = ((tl.load($(vto_copy_23_ptr) + ((((r53_z / $(256)) % $(128)) * $(256)) + (r53_z % $(256))), dtype=tl.float32)) * (tl.load($(vto_copy_23_ptr) + ((((r53_z / $(256)) % $(128)) * $(256)) + (r53_z % $(256))), dtype=tl.float32)))
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
    r57_value = (tl.load($(vto_copy_23_ptr) + ((((r57_z / $(256)) % $(128)) * $(256)) + (r57_z % $(256))), dtype=tl.float32) * tl.load($(vrsqrt_1_ptr) + ((r57_z / $(256)) % $(128)), dtype=tl.float32))
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
    tl.store($(vto_copy_24_ptr) + r59_z, r59_value)
    r60_row = tl.arange($(128))
    r60_col = tl.arange($(2048))
    r60_lane = tl.arange($(256))
    r60_left = tl.load($(vto_copy_24_ptr) + ((((((r60_row[:, None] * $(256)) + r60_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r60_row[:, None] * $(256)) + r60_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r60_right = tl.load($(vto_copy_20_ptr) + (((r60_col[None, :] % $(2048)) * $(256)) + (r60_lane[:, None] % $(256))), dtype=tl.bfloat16)
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
    tl.store($(vto_copy_25_ptr) + r80_z, r80_value)
    r81_z = tl.arange($(131072))
    r81_lane = tl.arange($(128))
    r81_row_values = tl.load($(vto_copy_25_ptr) + ((((r81_z[:, None] / $(16384)) % $(8)) * $(16384)) + (((r81_z[:, None] / $(128)) % $(128)) * $(128)) + (r81_lane[None, :] % $(128))), dtype=tl.float32)
    r81_maximum = tl.max(r81_row_values, axis=1, keep_dims=true)
    r81_weights = tl.exp(r81_row_values - r81_maximum)
    r81_total = tl.sum(r81_weights, axis=1)
    r81_maximum = tl.max(r81_row_values, axis=1)
    r81_value = tl.exp(tl.load($(vto_copy_25_ptr) + ((((r81_z / $(16384)) % $(8)) * $(16384)) + (((r81_z / $(128)) % $(128)) * $(128)) + (r81_z % $(128))), dtype=tl.float32) - r81_maximum) / r81_total
    r81_value = r81_value.to(tl.float32)
    r81_value = tl.broadcast_to(r81_value, [$(131072)])
    tl.store($(vsoftmax_ptr) + r81_z, r81_value)
    r82_z = tl.arange($(131072))
    r82_value = tl.load($(vsoftmax_ptr) + ((((r82_z / $(16384)) % $(8)) * $(16384)) + (((r82_z / $(128)) % $(128)) * $(128)) + (r82_z % $(128))), dtype=tl.float32)
    r82_value = r82_value.to(tl.bfloat16)
    r82_value = tl.broadcast_to(r82_value, [$(131072)])
    tl.store($(vto_copy_26_ptr) + r82_z, r82_value)
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
      r84_left = tl.load($(vto_copy_26_ptr) + (((((((r84_group % $(8)) * $(16384)) + (r84_lane[None, :] * $(128)) + r84_row[:, None]) / $(16384)) % $(8)) * $(16384)) + ((((((r84_group % $(8)) * $(16384)) + (r84_lane[None, :] * $(128)) + r84_row[:, None]) / $(128)) % $(128)) * $(128)) + ((((r84_group % $(8)) * $(16384)) + (r84_lane[None, :] * $(128)) + r84_row[:, None]) % $(128))), dtype=tl.bfloat16)
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
    tl.store($(vto_copy_27_ptr) + r86_z, r86_value)
    r87_z = tl.arange($(131072))
    r87_lane = tl.arange($(128))
    r87_row_values = tl.load($(vsoftmax_ptr) + ((((r87_z[:, None] / $(16384)) % $(8)) * $(16384)) + (((r87_z[:, None] / $(128)) % $(128)) * $(128)) + (r87_lane[None, :] % $(128))), dtype=tl.float32)
    r87_row_grad = tl.load($(vto_copy_27_ptr) + ((((r87_z[:, None] / $(16384)) % $(8)) * $(16384)) + (((r87_z[:, None] / $(128)) % $(128)) * $(128)) + (r87_lane[None, :] % $(128))), dtype=tl.float32)
    r87_weighted = r87_row_grad * r87_row_values
    r87_dot_grad = tl.sum(r87_weighted, axis=1)
    r87_value = (tl.load($(vto_copy_27_ptr) + ((((r87_z / $(16384)) % $(8)) * $(16384)) + (((r87_z / $(128)) % $(128)) * $(128)) + (r87_z % $(128))), dtype=tl.float32) - r87_dot_grad) * tl.load($(vsoftmax_ptr) + ((((r87_z / $(16384)) % $(8)) * $(16384)) + (((r87_z / $(128)) % $(128)) * $(128)) + (r87_z % $(128))), dtype=tl.float32)
    r87_value = r87_value.to(tl.float32)
    r87_value = tl.broadcast_to(r87_value, [$(131072)])
    tl.store($(vsoftmax_backward_data_ptr) + r87_z, r87_value)
    r88_z = tl.arange($(131072))
    r88_value = tl.load($(vsoftmax_backward_data_ptr) + ((((r88_z / $(16384)) % $(8)) * $(16384)) + (((r88_z / $(128)) % $(128)) * $(128)) + (r88_z % $(128))), dtype=tl.float32)
    r88_value = r88_value.to(tl.bfloat16)
    r88_value = tl.broadcast_to(r88_value, [$(131072)])
    tl.store($(vto_copy_28_ptr) + r88_z, r88_value)
    r89_z = tl.arange($(131072))
    r89_value = (tl.load($(vto_copy_28_ptr) + ((((r89_z / $(16384)) % $(8)) * $(16384)) + (((r89_z / $(128)) % $(128)) * $(128)) + (r89_z % $(128))), dtype=tl.bfloat16) * tl.cast(0.07216878364870322, tl.bfloat16))
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
    r91_z = tl.arange($(262144))
    r91_value = tl.where((r91_z % $(256)) < $(128), tl.load($(vbmm_6_ptr) + ((((((((r91_z / $(32768)) % $(8)) * $(24576)) + (((r91_z % $(256)) - $(0)) * $(128)) + ((r91_z / $(256)) % $(128))) / $(24576)) % $(8)) * $(24576)) + ((((((r91_z / $(32768)) % $(8)) * $(192)) + ((r91_z % $(256)) - $(0))) % $(192)) * $(128)) + ((r91_z / $(256)) % $(128))), dtype=tl.bfloat16), tl.load($(vbmm_4_ptr) + ((((((((r91_z / $(32768)) % $(8)) * $(16384)) + (((r91_z / $(256)) % $(128)) * $(128)) + ((r91_z % $(256)) - $(128))) / $(16384)) % $(8)) * $(16384)) + (((((((r91_z / $(32768)) % $(8)) * $(16384)) + (((r91_z / $(256)) % $(128)) * $(128)) + ((r91_z % $(256)) - $(128))) / $(128)) % $(128)) * $(128)) + (((((r91_z / $(32768)) % $(8)) * $(16384)) + (((r91_z / $(256)) % $(128)) * $(128)) + ((r91_z % $(256)) - $(128))) % $(128))), dtype=tl.bfloat16))
    r91_value = r91_value.to(tl.bfloat16)
    r91_value = tl.broadcast_to(r91_value, [$(262144)])
    tl.store($(vcat_9_ptr) + r91_z, r91_value)
    r92_row = tl.arange($(2048))
    r92_col = tl.arange($(256))
    r92_lane = tl.arange($(128))
    r92_left = tl.load($(vcat_9_ptr) + ((((((((((r92_lane[None, :] * $(2048)) + r92_row[:, None]) / $(2048)) % $(128)) * $(2048)) + (((r92_lane[None, :] * $(2048)) + r92_row[:, None]) % $(2048))) / $(256)) % $(8)) * $(32768)) + (((((r92_lane[None, :] * $(2048)) + r92_row[:, None]) / $(2048)) % $(128)) * $(256)) + (((((((r92_lane[None, :] * $(2048)) + r92_row[:, None]) / $(2048)) % $(128)) * $(2048)) + (((r92_lane[None, :] * $(2048)) + r92_row[:, None]) % $(2048))) % $(256))), dtype=tl.bfloat16)
    r92_right = tl.load($(vto_copy_24_ptr) + ((((((r92_lane[:, None] * $(256)) + r92_col[None, :]) / $(256)) % $(128)) * $(256)) + (((r92_lane[:, None] * $(256)) + r92_col[None, :]) % $(256))), dtype=tl.bfloat16)
    r92_value = tl.dot(r92_left, r92_right)
    r92_value = r92_value.to(tl.bfloat16)
    tl.store($(vmm_6_ptr) + ((r92_row[:, None] * $(256)) + r92_col[None, :]), r92_value)
    z = tl.arange($(524288))
    value = tl.load($(vmm_6_ptr) + ((((z / $(256)) % $(2048)) * $(256)) + (z % $(256))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(524288)])
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
  let vto_copy_6_ptr : RegionName := "vto_copy_6"
  let vto_copy_20_ptr : RegionName := "vto_copy_20"
  let vmm_ptr : RegionName := "vmm"
  let vto_copy_21_ptr : RegionName := "vto_copy_21"
  let vpow_3_ptr : RegionName := "vpow_3"
  let vmean_ptr : RegionName := "vmean"
  let vadd_3_ptr : RegionName := "vadd_3"
  let vrsqrt_ptr : RegionName := "vrsqrt"
  let vmul_13_ptr : RegionName := "vmul_13"
  let vmul_14_ptr : RegionName := "vmul_14"
  let vto_copy_22_ptr : RegionName := "vto_copy_22"
  let vmm_1_ptr : RegionName := "vmm_1"
  let vmm_2_ptr : RegionName := "vmm_2"
  let vto_copy_23_ptr : RegionName := "vto_copy_23"
  let vpow_4_ptr : RegionName := "vpow_4"
  let vmean_1_ptr : RegionName := "vmean_1"
  let vadd_4_ptr : RegionName := "vadd_4"
  let vrsqrt_1_ptr : RegionName := "vrsqrt_1"
  let vmul_15_ptr : RegionName := "vmul_15"
  let vmul_16_ptr : RegionName := "vmul_16"
  let vto_copy_24_ptr : RegionName := "vto_copy_24"
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
  let vto_copy_25_ptr : RegionName := "vto_copy_25"
  let vsoftmax_ptr : RegionName := "vsoftmax"
  let vto_copy_26_ptr : RegionName := "vto_copy_26"
  let vbmm_4_ptr : RegionName := "vbmm_4"
  let vto_copy_27_ptr : RegionName := "vto_copy_27"
  let vsoftmax_backward_data_ptr : RegionName := "vsoftmax_backward_data"
  let vto_copy_28_ptr : RegionName := "vto_copy_28"
  let vmul_26_ptr : RegionName := "vmul_26"
  let vbmm_6_ptr : RegionName := "vbmm_6"
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
  let candidate_vto_copy_6_ptr : RegionName := "candidate_vto_copy_6"
  let candidate_vto_copy_20_ptr : RegionName := "candidate_vto_copy_20"
  let candidate_vmm_ptr : RegionName := "candidate_vmm"
  let candidate_vto_copy_21_ptr : RegionName := "candidate_vto_copy_21"
  let candidate_vpow_3_ptr : RegionName := "candidate_vpow_3"
  let candidate_vmean_ptr : RegionName := "candidate_vmean"
  let candidate_vadd_3_ptr : RegionName := "candidate_vadd_3"
  let candidate_vrsqrt_ptr : RegionName := "candidate_vrsqrt"
  let candidate_vmul_13_ptr : RegionName := "candidate_vmul_13"
  let candidate_vmul_14_ptr : RegionName := "candidate_vmul_14"
  let candidate_vto_copy_22_ptr : RegionName := "candidate_vto_copy_22"
  let candidate_vmm_1_ptr : RegionName := "candidate_vmm_1"
  let candidate_vmm_2_ptr : RegionName := "candidate_vmm_2"
  let candidate_vto_copy_23_ptr : RegionName := "candidate_vto_copy_23"
  let candidate_vpow_4_ptr : RegionName := "candidate_vpow_4"
  let candidate_vmean_1_ptr : RegionName := "candidate_vmean_1"
  let candidate_vadd_4_ptr : RegionName := "candidate_vadd_4"
  let candidate_vrsqrt_1_ptr : RegionName := "candidate_vrsqrt_1"
  let candidate_vmul_15_ptr : RegionName := "candidate_vmul_15"
  let candidate_vmul_16_ptr : RegionName := "candidate_vmul_16"
  let candidate_vto_copy_24_ptr : RegionName := "candidate_vto_copy_24"
  let candidate_vdiv_2_ptr : RegionName := "candidate_vdiv_2"
  let candidate_vpow_5_ptr : RegionName := "candidate_vpow_5"
  let candidate_vreciprocal_2_ptr : RegionName := "candidate_vreciprocal_2"
  let candidate_vmul_17_ptr : RegionName := "candidate_vmul_17"
  let candidate_vmul_18_ptr : RegionName := "candidate_vmul_18"
  let candidate_vcat_5_ptr : RegionName := "candidate_vcat_5"
  let candidate_vcos_2_ptr : RegionName := "candidate_vcos_2"
  let candidate_vto_copy_25_ptr : RegionName := "candidate_vto_copy_25"
  let candidate_vsin_2_ptr : RegionName := "candidate_vsin_2"
  let candidate_vto_copy_26_ptr : RegionName := "candidate_vto_copy_26"
  let candidate_vneg_ptr : RegionName := "candidate_vneg"
  let candidate_vstack_ptr : RegionName := "candidate_vstack"
  let candidate_vmul_19_ptr : RegionName := "candidate_vmul_19"
  let candidate_vmul_20_ptr : RegionName := "candidate_vmul_20"
  let candidate_vadd_5_ptr : RegionName := "candidate_vadd_5"
  let candidate_vcat_6_ptr : RegionName := "candidate_vcat_6"
  let candidate_vcos_3_ptr : RegionName := "candidate_vcos_3"
  let candidate_vto_copy_27_ptr : RegionName := "candidate_vto_copy_27"
  let candidate_vsin_3_ptr : RegionName := "candidate_vsin_3"
  let candidate_vto_copy_28_ptr : RegionName := "candidate_vto_copy_28"
  let candidate_vneg_1_ptr : RegionName := "candidate_vneg_1"
  let candidate_vstack_1_ptr : RegionName := "candidate_vstack_1"
  let candidate_vmul_21_ptr : RegionName := "candidate_vmul_21"
  let candidate_vmul_22_ptr : RegionName := "candidate_vmul_22"
  let candidate_vadd_6_ptr : RegionName := "candidate_vadd_6"
  let candidate_vbmm_2_ptr : RegionName := "candidate_vbmm_2"
  let candidate_vbmm_3_ptr : RegionName := "candidate_vbmm_3"
  let candidate_vcat_7_ptr : RegionName := "candidate_vcat_7"
  let candidate_vcat_8_ptr : RegionName := "candidate_vcat_8"
  let candidate_vto_copy_29_ptr : RegionName := "candidate_vto_copy_29"
  let candidate_vto_copy_30_ptr : RegionName := "candidate_vto_copy_30"
  let candidate_vbmm_4_ptr : RegionName := "candidate_vbmm_4"
  let candidate_vmul_23_ptr : RegionName := "candidate_vmul_23"
  let candidate_vadd_7_ptr : RegionName := "candidate_vadd_7"
  let candidate_vsub_4_ptr : RegionName := "candidate_vsub_4"
  let candidate_vgt_ptr : RegionName := "candidate_vgt"
  let candidate_vmasked_fill_ptr : RegionName := "candidate_vmasked_fill"
  let candidate_vsoftmax_ptr : RegionName := "candidate_vsoftmax"
  let candidate_vto_copy_31_ptr : RegionName := "candidate_vto_copy_31"
  let candidate_vto_copy_32_ptr : RegionName := "candidate_vto_copy_32"
  let candidate_vto_copy_33_ptr : RegionName := "candidate_vto_copy_33"
  let candidate_vbmm_6_ptr : RegionName := "candidate_vbmm_6"
  let candidate_vmul_24_ptr : RegionName := "candidate_vmul_24"
  let candidate_vadd_8_ptr : RegionName := "candidate_vadd_8"
  let candidate_vsub_5_ptr : RegionName := "candidate_vsub_5"
  let candidate_vgt_1_ptr : RegionName := "candidate_vgt_1"
  let candidate_vmasked_fill_1_ptr : RegionName := "candidate_vmasked_fill_1"
  let candidate_vsoftmax_1_ptr : RegionName := "candidate_vsoftmax_1"
  let candidate_vto_copy_34_ptr : RegionName := "candidate_vto_copy_34"
  let candidate_vto_copy_35_ptr : RegionName := "candidate_vto_copy_35"
  let candidate_vto_copy_36_ptr : RegionName := "candidate_vto_copy_36"
  let candidate_vbmm_8_ptr : RegionName := "candidate_vbmm_8"
  let candidate_vmul_25_ptr : RegionName := "candidate_vmul_25"
  let candidate_vadd_9_ptr : RegionName := "candidate_vadd_9"
  let candidate_vsub_6_ptr : RegionName := "candidate_vsub_6"
  let candidate_vgt_2_ptr : RegionName := "candidate_vgt_2"
  let candidate_vmasked_fill_2_ptr : RegionName := "candidate_vmasked_fill_2"
  let candidate_vsoftmax_2_ptr : RegionName := "candidate_vsoftmax_2"
  let candidate_vto_copy_37_ptr : RegionName := "candidate_vto_copy_37"
  let candidate_vmm_4_ptr : RegionName := "candidate_vmm_4"
  let candidate_vbmm_10_ptr : RegionName := "candidate_vbmm_10"
  let candidate_vbmm_11_ptr : RegionName := "candidate_vbmm_11"
  let candidate_vto_copy_38_ptr : RegionName := "candidate_vto_copy_38"
  let candidate_vsoftmax_backward_data_ptr : RegionName := "candidate_vsoftmax_backward_data"
  let candidate_vmasked_fill_3_ptr : RegionName := "candidate_vmasked_fill_3"
  let candidate_vmul_26_ptr : RegionName := "candidate_vmul_26"
  let candidate_vbmm_12_ptr : RegionName := "candidate_vbmm_12"
  let candidate_vslice_backward_ptr : RegionName := "candidate_vslice_backward"
  let candidate_vto_copy_39_ptr : RegionName := "candidate_vto_copy_39"
  let candidate_vslice_backward_1_ptr : RegionName := "candidate_vslice_backward_1"
  let candidate_vbmm_13_ptr : RegionName := "candidate_vbmm_13"
  let candidate_vbmm_14_ptr : RegionName := "candidate_vbmm_14"
  let candidate_vto_copy_40_ptr : RegionName := "candidate_vto_copy_40"
  let candidate_vsoftmax_backward_data_1_ptr : RegionName := "candidate_vsoftmax_backward_data_1"
  let candidate_vmasked_fill_4_ptr : RegionName := "candidate_vmasked_fill_4"
  let candidate_vmul_27_ptr : RegionName := "candidate_vmul_27"
  let candidate_vbmm_15_ptr : RegionName := "candidate_vbmm_15"
  let candidate_vslice_backward_2_ptr : RegionName := "candidate_vslice_backward_2"
  let candidate_vadd_10_ptr : RegionName := "candidate_vadd_10"
  let candidate_vto_copy_41_ptr : RegionName := "candidate_vto_copy_41"
  let candidate_vslice_backward_3_ptr : RegionName := "candidate_vslice_backward_3"
  let candidate_vadd_11_ptr : RegionName := "candidate_vadd_11"
  let candidate_vbmm_16_ptr : RegionName := "candidate_vbmm_16"
  let candidate_vbmm_17_ptr : RegionName := "candidate_vbmm_17"
  let candidate_vto_copy_42_ptr : RegionName := "candidate_vto_copy_42"
  let candidate_vsoftmax_backward_data_2_ptr : RegionName := "candidate_vsoftmax_backward_data_2"
  let candidate_vmasked_fill_5_ptr : RegionName := "candidate_vmasked_fill_5"
  let candidate_vmul_28_ptr : RegionName := "candidate_vmul_28"
  let candidate_vbmm_18_ptr : RegionName := "candidate_vbmm_18"
  let candidate_vslice_backward_4_ptr : RegionName := "candidate_vslice_backward_4"
  let candidate_vadd_12_ptr : RegionName := "candidate_vadd_12"
  let candidate_vto_copy_43_ptr : RegionName := "candidate_vto_copy_43"
  let candidate_vslice_backward_5_ptr : RegionName := "candidate_vslice_backward_5"
  let candidate_vadd_13_ptr : RegionName := "candidate_vadd_13"
  let candidate_vbmm_19_ptr : RegionName := "candidate_vbmm_19"
  let candidate_vbmm_20_ptr : RegionName := "candidate_vbmm_20"
  let candidate_vslice_backward_6_ptr : RegionName := "candidate_vslice_backward_6"
  let candidate_vslice_backward_7_ptr : RegionName := "candidate_vslice_backward_7"
  let candidate_vadd_14_ptr : RegionName := "candidate_vadd_14"
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
    r14_value = tl.load($(candidate_vmul_6_ptr) + ((((r14_z / $(1024)) % $(512)) * $(1024)) + (r14_z % $(1024))), dtype=tl.float32)
    r14_value = r14_value.to(tl.bfloat16)
    r14_value = tl.broadcast_to(r14_value, [$(524288)])
    tl.store($(candidate_vto_copy_6_ptr) + r14_z, r14_value)
    r15_z = tl.arange($(524288))
    r15_value = tl.load($(candidate_vmul_5_ptr) + ((((r15_z / $(256)) % $(2048)) * $(256)) + (r15_z % $(256))), dtype=tl.float32)
    r15_value = r15_value.to(tl.bfloat16)
    r15_value = tl.broadcast_to(r15_value, [$(524288)])
    tl.store($(candidate_vto_copy_20_ptr) + r15_z, r15_value)
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
    tl.store($(candidate_vto_copy_21_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(32768))
    r18_value = ((tl.load($(candidate_vto_copy_21_ptr) + ((((r18_z / $(256)) % $(128)) * $(256)) + (r18_z % $(256))), dtype=tl.float32)) * (tl.load($(candidate_vto_copy_21_ptr) + ((((r18_z / $(256)) % $(128)) * $(256)) + (r18_z % $(256))), dtype=tl.float32)))
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
    r22_value = (tl.load($(candidate_vto_copy_21_ptr) + ((((r22_z / $(256)) % $(128)) * $(256)) + (r22_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_ptr) + ((r22_z / $(256)) % $(128)), dtype=tl.float32))
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
    tl.store($(candidate_vto_copy_22_ptr) + r24_z, r24_value)
    r25_row = tl.arange($(128))
    r25_col = tl.arange($(1536))
    r25_lane = tl.arange($(256))
    r25_left = tl.load($(candidate_vto_copy_22_ptr) + ((((((r25_row[:, None] * $(256)) + r25_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r25_row[:, None] * $(256)) + r25_lane[None, :]) % $(256))), dtype=tl.bfloat16)
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
    tl.store($(candidate_vto_copy_23_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(32768))
    r28_value = ((tl.load($(candidate_vto_copy_23_ptr) + ((((r28_z / $(256)) % $(128)) * $(256)) + (r28_z % $(256))), dtype=tl.float32)) * (tl.load($(candidate_vto_copy_23_ptr) + ((((r28_z / $(256)) % $(128)) * $(256)) + (r28_z % $(256))), dtype=tl.float32)))
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
    r32_value = (tl.load($(candidate_vto_copy_23_ptr) + ((((r32_z / $(256)) % $(128)) * $(256)) + (r32_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_1_ptr) + ((r32_z / $(256)) % $(128)), dtype=tl.float32))
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
    tl.store($(candidate_vto_copy_24_ptr) + r34_z, r34_value)
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
    tl.store($(candidate_vto_copy_25_ptr) + r42_z, r42_value)
    r43_z = tl.arange($(8192))
    r43_value = tl.sin(tl.load($(candidate_vcat_5_ptr) + ((((r43_z / $(64)) % $(128)) * $(64)) + (r43_z % $(64))), dtype=tl.float32))
    r43_value = r43_value.to(tl.float32)
    r43_value = tl.broadcast_to(r43_value, [$(8192)])
    tl.store($(candidate_vsin_2_ptr) + r43_z, r43_value)
    r44_z = tl.arange($(8192))
    r44_value = tl.load($(candidate_vsin_2_ptr) + ((((r44_z / $(64)) % $(128)) * $(64)) + (r44_z % $(64))), dtype=tl.float32)
    r44_value = r44_value.to(tl.bfloat16)
    r44_value = tl.broadcast_to(r44_value, [$(8192)])
    tl.store($(candidate_vto_copy_26_ptr) + r44_z, r44_value)
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
    r47_value = (tl.load($(candidate_vmm_1_ptr) + ((((r47_z / $(512)) % $(128)) * $(1536)) + (((r47_z / $(64)) % $(8)) * $(192)) + (r47_z % $(64)) + $(128)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_25_ptr) + ((((r47_z / $(512)) % $(128)) * $(64)) + (r47_z % $(64))), dtype=tl.bfloat16))
    r47_value = r47_value.to(tl.bfloat16)
    r47_value = tl.broadcast_to(r47_value, [$(65536)])
    tl.store($(candidate_vmul_19_ptr) + r47_z, r47_value)
    r48_z = tl.arange($(65536))
    r48_value = (tl.load($(candidate_vstack_ptr) + ((((r48_z / $(512)) % $(128)) * $(512)) + (((r48_z / $(64)) % $(8)) * $(64)) + (((((((r48_z / $(512)) % $(128)) * $(512)) + (((r48_z / $(64)) % $(8)) * $(64)) + (r48_z % $(64))) / $(2)) % $(32)) * $(2)) + (((((r48_z / $(512)) % $(128)) * $(512)) + (((r48_z / $(64)) % $(8)) * $(64)) + (r48_z % $(64))) % $(2))), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_26_ptr) + ((((r48_z / $(512)) % $(128)) * $(64)) + (r48_z % $(64))), dtype=tl.bfloat16))
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
    tl.store($(candidate_vto_copy_27_ptr) + r52_z, r52_value)
    r53_z = tl.arange($(8192))
    r53_value = tl.sin(tl.load($(candidate_vcat_6_ptr) + ((((r53_z / $(64)) % $(128)) * $(64)) + (r53_z % $(64))), dtype=tl.float32))
    r53_value = r53_value.to(tl.float32)
    r53_value = tl.broadcast_to(r53_value, [$(8192)])
    tl.store($(candidate_vsin_3_ptr) + r53_z, r53_value)
    r54_z = tl.arange($(8192))
    r54_value = tl.load($(candidate_vsin_3_ptr) + ((((r54_z / $(64)) % $(128)) * $(64)) + (r54_z % $(64))), dtype=tl.float32)
    r54_value = r54_value.to(tl.bfloat16)
    r54_value = tl.broadcast_to(r54_value, [$(8192)])
    tl.store($(candidate_vto_copy_28_ptr) + r54_z, r54_value)
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
    r57_value = (tl.load($(candidate_vmm_2_ptr) + ((((r57_z / $(64)) % $(128)) * $(320)) + (r57_z % $(64)) + $(256)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_27_ptr) + ((((r57_z / $(64)) % $(128)) * $(64)) + (r57_z % $(64))), dtype=tl.bfloat16))
    r57_value = r57_value.to(tl.bfloat16)
    r57_value = tl.broadcast_to(r57_value, [$(8192)])
    tl.store($(candidate_vmul_21_ptr) + r57_z, r57_value)
    r58_z = tl.arange($(8192))
    r58_value = (tl.load($(candidate_vstack_1_ptr) + ((((r58_z / $(64)) % $(128)) * $(64)) + (((((((r58_z / $(64)) % $(128)) * $(64)) + (r58_z % $(64))) / $(2)) % $(32)) * $(2)) + (((((r58_z / $(64)) % $(128)) * $(64)) + (r58_z % $(64))) % $(2))), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_28_ptr) + ((((r58_z / $(64)) % $(128)) * $(64)) + (r58_z % $(64))), dtype=tl.bfloat16))
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
      r60_left = tl.load($(candidate_vto_copy_24_ptr) + ((((((r60_row[:, None] * $(256)) + r60_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r60_row[:, None] * $(256)) + r60_lane[None, :]) % $(256))), dtype=tl.bfloat16)
      r60_right = tl.load($(candidate_vto_copy_20_ptr) + ((((((r60_lane[:, None] * $(1024)) + r60_col[None, :]) / $(128)) % $(8)) * $(65536)) + ((((r60_lane[:, None] * $(1024)) + r60_col[None, :]) % $(128)) * $(256)) + ((((r60_lane[:, None] * $(1024)) + r60_col[None, :]) / $(1024)) % $(256))), dtype=tl.bfloat16)
      r60_value = tl.dot(r60_left, r60_right)
      r60_value = r60_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_2_ptr) + ((r60_group * $(131072)) + (r60_row[:, None] * $(1024)) + r60_col[None, :]), r60_value)
    }
    r61_row = tl.arange($(128))
    r61_col = tl.arange($(1024))
    r61_lane = tl.arange($(256))
    for r61_group in range($(1)) {
      r61_left = tl.load($(candidate_vto_copy_24_ptr) + ((((((r61_row[:, None] * $(256)) + r61_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r61_row[:, None] * $(256)) + r61_lane[None, :]) % $(256))), dtype=tl.bfloat16)
      r61_right = tl.load($(candidate_vto_copy_20_ptr) + ((((((r61_lane[:, None] * $(1024)) + r61_col[None, :]) / $(128)) % $(8)) * $(65536)) + ((((r61_lane[:, None] * $(1024)) + r61_col[None, :]) % $(128)) * $(256)) + ((((r61_lane[:, None] * $(1024)) + r61_col[None, :]) / $(1024)) % $(256)) + $(32768)), dtype=tl.bfloat16)
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
    tl.store($(candidate_vto_copy_29_ptr) + r64_z, r64_value)
    r65_z = tl.arange($(73728))
    r65_value = tl.load($(candidate_vcat_8_ptr) + ((((r65_z / $(192)) % $(48)) * $(1536)) + (((r65_z / $(9216)) % $(8)) * $(192)) + (r65_z % $(192))), dtype=tl.bfloat16)
    r65_value = r65_value.to(tl.float32)
    r65_value = tl.broadcast_to(r65_value, [$(73728)])
    tl.store($(candidate_vto_copy_30_ptr) + r65_z, r65_value)
    r66_row = tl.arange($(48))
    r66_col = tl.arange($(48))
    r66_lane = tl.arange($(192))
    for r66_group in range($(8)) {
      r66_left = tl.load($(candidate_vto_copy_29_ptr) + (((((((r66_group % $(8)) * $(9216)) + (r66_row[:, None] * $(192)) + r66_lane[None, :]) / $(9216)) % $(8)) * $(9216)) + ((((((r66_group % $(8)) * $(9216)) + (r66_row[:, None] * $(192)) + r66_lane[None, :]) / $(192)) % $(48)) * $(192)) + ((((r66_group % $(8)) * $(9216)) + (r66_row[:, None] * $(192)) + r66_lane[None, :]) % $(192))), dtype=tl.float32)
      r66_right = tl.load($(candidate_vto_copy_30_ptr) + (((((((r66_group % $(8)) * $(9216)) + (r66_lane[:, None] * $(48)) + r66_col[None, :]) / $(9216)) % $(8)) * $(9216)) + (((((r66_group % $(8)) * $(9216)) + (r66_lane[:, None] * $(48)) + r66_col[None, :]) % $(48)) * $(192)) + (((((r66_group % $(8)) * $(9216)) + (r66_lane[:, None] * $(48)) + r66_col[None, :]) / $(48)) % $(192))), dtype=tl.float32)
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
    tl.store($(candidate_vto_copy_31_ptr) + r73_z, r73_value)
    r74_z = tl.arange($(24576))
    r74_value = tl.load($(candidate_vcat_7_ptr) + ((((r74_z / $(192)) % $(16)) * $(1536)) + (((r74_z / $(3072)) % $(8)) * $(192)) + (r74_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r74_value = r74_value.to(tl.float32)
    r74_value = tl.broadcast_to(r74_value, [$(24576)])
    tl.store($(candidate_vto_copy_32_ptr) + r74_z, r74_value)
    r75_z = tl.arange($(24576))
    r75_value = tl.load($(candidate_vcat_8_ptr) + ((((r75_z / $(192)) % $(16)) * $(1536)) + (((r75_z / $(3072)) % $(8)) * $(192)) + (r75_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r75_value = r75_value.to(tl.float32)
    r75_value = tl.broadcast_to(r75_value, [$(24576)])
    tl.store($(candidate_vto_copy_33_ptr) + r75_z, r75_value)
    r76_row = tl.arange($(16))
    r76_col = tl.arange($(16))
    r76_lane = tl.arange($(192))
    for r76_group in range($(8)) {
      r76_left = tl.load($(candidate_vto_copy_32_ptr) + (((((((r76_group % $(8)) * $(3072)) + (r76_row[:, None] * $(192)) + r76_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r76_group % $(8)) * $(3072)) + (r76_row[:, None] * $(192)) + r76_lane[None, :]) / $(192)) % $(16)) * $(192)) + ((((r76_group % $(8)) * $(3072)) + (r76_row[:, None] * $(192)) + r76_lane[None, :]) % $(192))), dtype=tl.float32)
      r76_right = tl.load($(candidate_vto_copy_33_ptr) + (((((((r76_group % $(8)) * $(3072)) + (r76_lane[:, None] * $(16)) + r76_col[None, :]) / $(3072)) % $(8)) * $(3072)) + (((((r76_group % $(8)) * $(3072)) + (r76_lane[:, None] * $(16)) + r76_col[None, :]) % $(16)) * $(192)) + (((((r76_group % $(8)) * $(3072)) + (r76_lane[:, None] * $(16)) + r76_col[None, :]) / $(16)) % $(192))), dtype=tl.float32)
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
    tl.store($(candidate_vto_copy_34_ptr) + r83_z, r83_value)
    r84_z = tl.arange($(98304))
    r84_value = tl.load($(candidate_vcat_7_ptr) + ((((r84_z / $(192)) % $(64)) * $(1536)) + (((r84_z / $(12288)) % $(8)) * $(192)) + (r84_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r84_value = r84_value.to(tl.float32)
    r84_value = tl.broadcast_to(r84_value, [$(98304)])
    tl.store($(candidate_vto_copy_35_ptr) + r84_z, r84_value)
    r85_z = tl.arange($(98304))
    r85_value = tl.load($(candidate_vcat_8_ptr) + ((((r85_z / $(192)) % $(64)) * $(1536)) + (((r85_z / $(12288)) % $(8)) * $(192)) + (r85_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r85_value = r85_value.to(tl.float32)
    r85_value = tl.broadcast_to(r85_value, [$(98304)])
    tl.store($(candidate_vto_copy_36_ptr) + r85_z, r85_value)
    r86_row = tl.arange($(64))
    r86_col = tl.arange($(64))
    r86_lane = tl.arange($(192))
    for r86_group in range($(8)) {
      r86_left = tl.load($(candidate_vto_copy_35_ptr) + (((((((r86_group % $(8)) * $(12288)) + (r86_row[:, None] * $(192)) + r86_lane[None, :]) / $(12288)) % $(8)) * $(12288)) + ((((((r86_group % $(8)) * $(12288)) + (r86_row[:, None] * $(192)) + r86_lane[None, :]) / $(192)) % $(64)) * $(192)) + ((((r86_group % $(8)) * $(12288)) + (r86_row[:, None] * $(192)) + r86_lane[None, :]) % $(192))), dtype=tl.float32)
      r86_right = tl.load($(candidate_vto_copy_36_ptr) + (((((((r86_group % $(8)) * $(12288)) + (r86_lane[:, None] * $(64)) + r86_col[None, :]) / $(12288)) % $(8)) * $(12288)) + (((((r86_group % $(8)) * $(12288)) + (r86_lane[:, None] * $(64)) + r86_col[None, :]) % $(64)) * $(192)) + (((((r86_group % $(8)) * $(12288)) + (r86_lane[:, None] * $(64)) + r86_col[None, :]) / $(64)) % $(192))), dtype=tl.float32)
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
    tl.store($(candidate_vto_copy_37_ptr) + r93_z, r93_value)
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
      r95_left = tl.load($(candidate_vto_copy_37_ptr) + (((((((r95_group % $(8)) * $(4096)) + (r95_lane[None, :] * $(64)) + r95_row[:, None]) / $(4096)) % $(8)) * $(4096)) + ((((((r95_group % $(8)) * $(4096)) + (r95_lane[None, :] * $(64)) + r95_row[:, None]) / $(64)) % $(64)) * $(64)) + ((((r95_group % $(8)) * $(4096)) + (r95_lane[None, :] * $(64)) + r95_row[:, None]) % $(64))), dtype=tl.bfloat16)
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
    tl.store($(candidate_vto_copy_38_ptr) + r97_z, r97_value)
    r98_z = tl.arange($(32768))
    r98_lane = tl.arange($(64))
    r98_row_values = tl.load($(candidate_vsoftmax_2_ptr) + ((((r98_z[:, None] / $(4096)) % $(8)) * $(4096)) + (((r98_z[:, None] / $(64)) % $(64)) * $(64)) + (r98_lane[None, :] % $(64))), dtype=tl.float32)
    r98_row_grad = tl.load($(candidate_vto_copy_38_ptr) + ((((r98_z[:, None] / $(4096)) % $(8)) * $(4096)) + (((r98_z[:, None] / $(64)) % $(64)) * $(64)) + (r98_lane[None, :] % $(64))), dtype=tl.float32)
    r98_weighted = r98_row_grad * r98_row_values
    r98_dot_grad = tl.sum(r98_weighted, axis=1)
    r98_value = (tl.load($(candidate_vto_copy_38_ptr) + ((((r98_z / $(4096)) % $(8)) * $(4096)) + (((r98_z / $(64)) % $(64)) * $(64)) + (r98_z % $(64))), dtype=tl.float32) - r98_dot_grad) * tl.load($(candidate_vsoftmax_2_ptr) + ((((r98_z / $(4096)) % $(8)) * $(4096)) + (((r98_z / $(64)) % $(64)) * $(64)) + (r98_z % $(64))), dtype=tl.float32)
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
      r101_left = tl.load($(candidate_vto_copy_35_ptr) + (((((((r101_group % $(8)) * $(12288)) + (r101_lane[None, :] * $(192)) + r101_row[:, None]) / $(12288)) % $(8)) * $(12288)) + ((((((r101_group % $(8)) * $(12288)) + (r101_lane[None, :] * $(192)) + r101_row[:, None]) / $(192)) % $(64)) * $(192)) + ((((r101_group % $(8)) * $(12288)) + (r101_lane[None, :] * $(192)) + r101_row[:, None]) % $(192))), dtype=tl.float32)
      r101_right = tl.load($(candidate_vmul_26_ptr) + (((((((r101_group % $(8)) * $(4096)) + (r101_lane[:, None] * $(64)) + r101_col[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r101_group % $(8)) * $(4096)) + (r101_lane[:, None] * $(64)) + r101_col[None, :]) / $(64)) % $(64)) * $(64)) + ((((r101_group % $(8)) * $(4096)) + (r101_lane[:, None] * $(64)) + r101_col[None, :]) % $(64))), dtype=tl.float32)
      r101_value = tl.dot(r101_left, r101_right)
      r101_value = r101_value.to(tl.float32)
      tl.store($(candidate_vbmm_12_ptr) + ((r101_group * $(12288)) + (r101_row[:, None] * $(64)) + r101_col[None, :]), r101_value)
    }
    r102_z = tl.arange($(131072))
    r102_value = tl.where(((((r102_z / $(1024)) % $(128)) >= $(64)) & (((r102_z / $(1024)) % $(128)) < $(128)) & ((((r102_z / $(1024)) % $(128)) - $(64)) % $(1) == $(0))), tl.load($(candidate_vbmm_10_ptr) + ((((((((r102_z / $(128)) % $(8)) * $(8192)) + ((((r102_z / $(1024)) % $(128)) - $(64)) * $(128)) + (r102_z % $(128))) / $(8192)) % $(8)) * $(8192)) + ((((((r102_z / $(128)) % $(8)) * $(64)) + (((r102_z / $(1024)) % $(128)) - $(64))) % $(64)) * $(128)) + (r102_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r102_value = r102_value.to(tl.bfloat16)
    r102_value = tl.broadcast_to(r102_value, [$(131072)])
    tl.store($(candidate_vslice_backward_ptr) + r102_z, r102_value)
    r103_z = tl.arange($(98304))
    r103_value = tl.load($(candidate_vbmm_12_ptr) + ((((r103_z / $(12288)) % $(8)) * $(12288)) + ((r103_z % $(192)) * $(64)) + ((r103_z / $(192)) % $(64))), dtype=tl.float32)
    r103_value = r103_value.to(tl.bfloat16)
    r103_value = tl.broadcast_to(r103_value, [$(98304)])
    tl.store($(candidate_vto_copy_39_ptr) + r103_z, r103_value)
    r104_z = tl.arange($(196608))
    r104_value = tl.where(((((r104_z / $(1536)) % $(128)) >= $(64)) & (((r104_z / $(1536)) % $(128)) < $(128)) & ((((r104_z / $(1536)) % $(128)) - $(64)) % $(1) == $(0))), tl.load($(candidate_vto_copy_39_ptr) + ((((r104_z / $(192)) % $(8)) * $(12288)) + (((((r104_z / $(1536)) % $(128)) - $(64)) % $(64)) * $(192)) + (r104_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r104_value = r104_value.to(tl.bfloat16)
    r104_value = tl.broadcast_to(r104_value, [$(196608)])
    tl.store($(candidate_vslice_backward_1_ptr) + r104_z, r104_value)
    r105_row = tl.arange($(16))
    r105_col = tl.arange($(128))
    r105_lane = tl.arange($(16))
    for r105_group in range($(8)) {
      r105_left = tl.load($(candidate_vto_copy_34_ptr) + (((((((r105_group % $(8)) * $(256)) + (r105_lane[None, :] * $(16)) + r105_row[:, None]) / $(256)) % $(8)) * $(256)) + ((((((r105_group % $(8)) * $(256)) + (r105_lane[None, :] * $(16)) + r105_row[:, None]) / $(16)) % $(16)) * $(16)) + ((((r105_group % $(8)) * $(256)) + (r105_lane[None, :] * $(16)) + r105_row[:, None]) % $(16))), dtype=tl.bfloat16)
      r105_right = tl.load($(candidate_vmm_4_ptr) + (((((((r105_group % $(8)) * $(2048)) + (r105_lane[:, None] * $(128)) + r105_col[None, :]) / $(128)) % $(16)) * $(1024)) + ((((((r105_group % $(8)) * $(2048)) + (r105_lane[:, None] * $(128)) + r105_col[None, :]) / $(2048)) % $(8)) * $(128)) + ((((r105_group % $(8)) * $(2048)) + (r105_lane[:, None] * $(128)) + r105_col[None, :]) % $(128)) + $(49152)), dtype=tl.bfloat16)
      r105_value = tl.dot(r105_left, r105_right)
      r105_value = r105_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_13_ptr) + ((r105_group * $(2048)) + (r105_row[:, None] * $(128)) + r105_col[None, :]), r105_value)
    }
    r106_row = tl.arange($(16))
    r106_col = tl.arange($(16))
    r106_lane = tl.arange($(128))
    for r106_group in range($(8)) {
      r106_left = tl.load($(candidate_vmm_4_ptr) + (((((((r106_group % $(8)) * $(2048)) + (r106_row[:, None] * $(128)) + r106_lane[None, :]) / $(128)) % $(16)) * $(1024)) + ((((((r106_group % $(8)) * $(2048)) + (r106_row[:, None] * $(128)) + r106_lane[None, :]) / $(2048)) % $(8)) * $(128)) + ((((r106_group % $(8)) * $(2048)) + (r106_row[:, None] * $(128)) + r106_lane[None, :]) % $(128)) + $(49152)), dtype=tl.bfloat16)
      r106_right = tl.load($(candidate_vbmm_3_ptr) + (((((((r106_group % $(8)) * $(2048)) + (r106_col[None, :] * $(128)) + r106_lane[:, None]) / $(128)) % $(16)) * $(1024)) + ((((((r106_group % $(8)) * $(2048)) + (r106_col[None, :] * $(128)) + r106_lane[:, None]) / $(2048)) % $(8)) * $(128)) + ((((r106_group % $(8)) * $(2048)) + (r106_col[None, :] * $(128)) + r106_lane[:, None]) % $(128)) + $(49152)), dtype=tl.bfloat16)
      r106_value = tl.dot(r106_left, r106_right)
      r106_value = r106_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_14_ptr) + ((r106_group * $(256)) + (r106_row[:, None] * $(16)) + r106_col[None, :]), r106_value)
    }
    r107_z = tl.arange($(2048))
    r107_value = tl.load($(candidate_vbmm_14_ptr) + ((((r107_z / $(256)) % $(8)) * $(256)) + (((r107_z / $(16)) % $(16)) * $(16)) + (r107_z % $(16))), dtype=tl.bfloat16)
    r107_value = r107_value.to(tl.float32)
    r107_value = tl.broadcast_to(r107_value, [$(2048)])
    tl.store($(candidate_vto_copy_40_ptr) + r107_z, r107_value)
    r108_z = tl.arange($(2048))
    r108_lane = tl.arange($(16))
    r108_row_values = tl.load($(candidate_vsoftmax_1_ptr) + ((((r108_z[:, None] / $(256)) % $(8)) * $(256)) + (((r108_z[:, None] / $(16)) % $(16)) * $(16)) + (r108_lane[None, :] % $(16))), dtype=tl.float32)
    r108_row_grad = tl.load($(candidate_vto_copy_40_ptr) + ((((r108_z[:, None] / $(256)) % $(8)) * $(256)) + (((r108_z[:, None] / $(16)) % $(16)) * $(16)) + (r108_lane[None, :] % $(16))), dtype=tl.float32)
    r108_weighted = r108_row_grad * r108_row_values
    r108_dot_grad = tl.sum(r108_weighted, axis=1)
    r108_value = (tl.load($(candidate_vto_copy_40_ptr) + ((((r108_z / $(256)) % $(8)) * $(256)) + (((r108_z / $(16)) % $(16)) * $(16)) + (r108_z % $(16))), dtype=tl.float32) - r108_dot_grad) * tl.load($(candidate_vsoftmax_1_ptr) + ((((r108_z / $(256)) % $(8)) * $(256)) + (((r108_z / $(16)) % $(16)) * $(16)) + (r108_z % $(16))), dtype=tl.float32)
    r108_value = r108_value.to(tl.float32)
    r108_value = tl.broadcast_to(r108_value, [$(2048)])
    tl.store($(candidate_vsoftmax_backward_data_1_ptr) + r108_z, r108_value)
    r109_z = tl.arange($(2048))
    r109_value = tl.where(tl.load($(candidate_vgt_1_ptr) + ((((r109_z / $(16)) % $(16)) * $(16)) + (r109_z % $(16))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_1_ptr) + ((((r109_z / $(256)) % $(8)) * $(256)) + (((r109_z / $(16)) % $(16)) * $(16)) + (r109_z % $(16))), dtype=tl.float32))
    r109_value = r109_value.to(tl.float32)
    r109_value = tl.broadcast_to(r109_value, [$(2048)])
    tl.store($(candidate_vmasked_fill_4_ptr) + r109_z, r109_value)
    r110_z = tl.arange($(2048))
    r110_value = (tl.load($(candidate_vmasked_fill_4_ptr) + ((((r110_z / $(256)) % $(8)) * $(256)) + (((r110_z / $(16)) % $(16)) * $(16)) + (r110_z % $(16))), dtype=tl.float32) * 0.07216878364870322)
    r110_value = r110_value.to(tl.float32)
    r110_value = tl.broadcast_to(r110_value, [$(2048)])
    tl.store($(candidate_vmul_27_ptr) + r110_z, r110_value)
    r111_row = tl.arange($(192))
    r111_col = tl.arange($(16))
    r111_lane = tl.arange($(16))
    for r111_group in range($(8)) {
      r111_left = tl.load($(candidate_vto_copy_32_ptr) + (((((((r111_group % $(8)) * $(3072)) + (r111_lane[None, :] * $(192)) + r111_row[:, None]) / $(3072)) % $(8)) * $(3072)) + ((((((r111_group % $(8)) * $(3072)) + (r111_lane[None, :] * $(192)) + r111_row[:, None]) / $(192)) % $(16)) * $(192)) + ((((r111_group % $(8)) * $(3072)) + (r111_lane[None, :] * $(192)) + r111_row[:, None]) % $(192))), dtype=tl.float32)
      r111_right = tl.load($(candidate_vmul_27_ptr) + (((((((r111_group % $(8)) * $(256)) + (r111_lane[:, None] * $(16)) + r111_col[None, :]) / $(256)) % $(8)) * $(256)) + ((((((r111_group % $(8)) * $(256)) + (r111_lane[:, None] * $(16)) + r111_col[None, :]) / $(16)) % $(16)) * $(16)) + ((((r111_group % $(8)) * $(256)) + (r111_lane[:, None] * $(16)) + r111_col[None, :]) % $(16))), dtype=tl.float32)
      r111_value = tl.dot(r111_left, r111_right)
      r111_value = r111_value.to(tl.float32)
      tl.store($(candidate_vbmm_15_ptr) + ((r111_group * $(3072)) + (r111_row[:, None] * $(16)) + r111_col[None, :]), r111_value)
    }
    r112_z = tl.arange($(131072))
    r112_value = tl.where(((((r112_z / $(1024)) % $(128)) >= $(48)) & (((r112_z / $(1024)) % $(128)) < $(64)) & ((((r112_z / $(1024)) % $(128)) - $(48)) % $(1) == $(0))), tl.load($(candidate_vbmm_13_ptr) + ((((((((r112_z / $(128)) % $(8)) * $(2048)) + ((((r112_z / $(1024)) % $(128)) - $(48)) * $(128)) + (r112_z % $(128))) / $(2048)) % $(8)) * $(2048)) + ((((((r112_z / $(128)) % $(8)) * $(16)) + (((r112_z / $(1024)) % $(128)) - $(48))) % $(16)) * $(128)) + (r112_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r112_value = r112_value.to(tl.bfloat16)
    r112_value = tl.broadcast_to(r112_value, [$(131072)])
    tl.store($(candidate_vslice_backward_2_ptr) + r112_z, r112_value)
    r113_z = tl.arange($(131072))
    r113_value = (tl.load($(candidate_vslice_backward_ptr) + ((((r113_z / $(1024)) % $(128)) * $(1024)) + (((r113_z / $(128)) % $(8)) * $(128)) + (r113_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_2_ptr) + ((((r113_z / $(1024)) % $(128)) * $(1024)) + (((r113_z / $(128)) % $(8)) * $(128)) + (r113_z % $(128))), dtype=tl.bfloat16))
    r113_value = r113_value.to(tl.bfloat16)
    r113_value = tl.broadcast_to(r113_value, [$(131072)])
    tl.store($(candidate_vadd_10_ptr) + r113_z, r113_value)
    r114_z = tl.arange($(24576))
    r114_value = tl.load($(candidate_vbmm_15_ptr) + ((((r114_z / $(3072)) % $(8)) * $(3072)) + ((r114_z % $(192)) * $(16)) + ((r114_z / $(192)) % $(16))), dtype=tl.float32)
    r114_value = r114_value.to(tl.bfloat16)
    r114_value = tl.broadcast_to(r114_value, [$(24576)])
    tl.store($(candidate_vto_copy_41_ptr) + r114_z, r114_value)
    r115_z = tl.arange($(196608))
    r115_value = tl.where(((((r115_z / $(1536)) % $(128)) >= $(48)) & (((r115_z / $(1536)) % $(128)) < $(64)) & ((((r115_z / $(1536)) % $(128)) - $(48)) % $(1) == $(0))), tl.load($(candidate_vto_copy_41_ptr) + ((((r115_z / $(192)) % $(8)) * $(3072)) + (((((r115_z / $(1536)) % $(128)) - $(48)) % $(16)) * $(192)) + (r115_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r115_value = r115_value.to(tl.bfloat16)
    r115_value = tl.broadcast_to(r115_value, [$(196608)])
    tl.store($(candidate_vslice_backward_3_ptr) + r115_z, r115_value)
    r116_z = tl.arange($(196608))
    r116_value = (tl.load($(candidate_vslice_backward_1_ptr) + ((((r116_z / $(1536)) % $(128)) * $(1536)) + (((r116_z / $(192)) % $(8)) * $(192)) + (r116_z % $(192))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_3_ptr) + ((((r116_z / $(1536)) % $(128)) * $(1536)) + (((r116_z / $(192)) % $(8)) * $(192)) + (r116_z % $(192))), dtype=tl.bfloat16))
    r116_value = r116_value.to(tl.bfloat16)
    r116_value = tl.broadcast_to(r116_value, [$(196608)])
    tl.store($(candidate_vadd_11_ptr) + r116_z, r116_value)
    r117_row = tl.arange($(48))
    r117_col = tl.arange($(128))
    r117_lane = tl.arange($(48))
    for r117_group in range($(8)) {
      r117_left = tl.load($(candidate_vto_copy_31_ptr) + (((((((r117_group % $(8)) * $(2304)) + (r117_lane[None, :] * $(48)) + r117_row[:, None]) / $(2304)) % $(8)) * $(2304)) + ((((((r117_group % $(8)) * $(2304)) + (r117_lane[None, :] * $(48)) + r117_row[:, None]) / $(48)) % $(48)) * $(48)) + ((((r117_group % $(8)) * $(2304)) + (r117_lane[None, :] * $(48)) + r117_row[:, None]) % $(48))), dtype=tl.bfloat16)
      r117_right = tl.load($(candidate_vmm_4_ptr) + (((((((r117_group % $(8)) * $(6144)) + (r117_lane[:, None] * $(128)) + r117_col[None, :]) / $(128)) % $(48)) * $(1024)) + ((((((r117_group % $(8)) * $(6144)) + (r117_lane[:, None] * $(128)) + r117_col[None, :]) / $(6144)) % $(8)) * $(128)) + ((((r117_group % $(8)) * $(6144)) + (r117_lane[:, None] * $(128)) + r117_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r117_value = tl.dot(r117_left, r117_right)
      r117_value = r117_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_16_ptr) + ((r117_group * $(6144)) + (r117_row[:, None] * $(128)) + r117_col[None, :]), r117_value)
    }
    r118_row = tl.arange($(48))
    r118_col = tl.arange($(48))
    r118_lane = tl.arange($(128))
    for r118_group in range($(8)) {
      r118_left = tl.load($(candidate_vmm_4_ptr) + (((((((r118_group % $(8)) * $(6144)) + (r118_row[:, None] * $(128)) + r118_lane[None, :]) / $(128)) % $(48)) * $(1024)) + ((((((r118_group % $(8)) * $(6144)) + (r118_row[:, None] * $(128)) + r118_lane[None, :]) / $(6144)) % $(8)) * $(128)) + ((((r118_group % $(8)) * $(6144)) + (r118_row[:, None] * $(128)) + r118_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r118_right = tl.load($(candidate_vbmm_3_ptr) + (((((((r118_group % $(8)) * $(6144)) + (r118_col[None, :] * $(128)) + r118_lane[:, None]) / $(128)) % $(48)) * $(1024)) + ((((((r118_group % $(8)) * $(6144)) + (r118_col[None, :] * $(128)) + r118_lane[:, None]) / $(6144)) % $(8)) * $(128)) + ((((r118_group % $(8)) * $(6144)) + (r118_col[None, :] * $(128)) + r118_lane[:, None]) % $(128))), dtype=tl.bfloat16)
      r118_value = tl.dot(r118_left, r118_right)
      r118_value = r118_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_17_ptr) + ((r118_group * $(2304)) + (r118_row[:, None] * $(48)) + r118_col[None, :]), r118_value)
    }
    r119_z = tl.arange($(18432))
    r119_value = tl.load($(candidate_vbmm_17_ptr) + ((((r119_z / $(2304)) % $(8)) * $(2304)) + (((r119_z / $(48)) % $(48)) * $(48)) + (r119_z % $(48))), dtype=tl.bfloat16)
    r119_value = r119_value.to(tl.float32)
    r119_value = tl.broadcast_to(r119_value, [$(18432)])
    tl.store($(candidate_vto_copy_42_ptr) + r119_z, r119_value)
    r120_z = tl.arange($(18432))
    r120_lane = tl.arange($(48))
    r120_row_values = tl.load($(candidate_vsoftmax_ptr) + ((((r120_z[:, None] / $(2304)) % $(8)) * $(2304)) + (((r120_z[:, None] / $(48)) % $(48)) * $(48)) + (r120_lane[None, :] % $(48))), dtype=tl.float32)
    r120_row_grad = tl.load($(candidate_vto_copy_42_ptr) + ((((r120_z[:, None] / $(2304)) % $(8)) * $(2304)) + (((r120_z[:, None] / $(48)) % $(48)) * $(48)) + (r120_lane[None, :] % $(48))), dtype=tl.float32)
    r120_weighted = r120_row_grad * r120_row_values
    r120_dot_grad = tl.sum(r120_weighted, axis=1)
    r120_value = (tl.load($(candidate_vto_copy_42_ptr) + ((((r120_z / $(2304)) % $(8)) * $(2304)) + (((r120_z / $(48)) % $(48)) * $(48)) + (r120_z % $(48))), dtype=tl.float32) - r120_dot_grad) * tl.load($(candidate_vsoftmax_ptr) + ((((r120_z / $(2304)) % $(8)) * $(2304)) + (((r120_z / $(48)) % $(48)) * $(48)) + (r120_z % $(48))), dtype=tl.float32)
    r120_value = r120_value.to(tl.float32)
    r120_value = tl.broadcast_to(r120_value, [$(18432)])
    tl.store($(candidate_vsoftmax_backward_data_2_ptr) + r120_z, r120_value)
    r121_z = tl.arange($(18432))
    r121_value = tl.where(tl.load($(candidate_vgt_ptr) + ((((r121_z / $(48)) % $(48)) * $(48)) + (r121_z % $(48))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_2_ptr) + ((((r121_z / $(2304)) % $(8)) * $(2304)) + (((r121_z / $(48)) % $(48)) * $(48)) + (r121_z % $(48))), dtype=tl.float32))
    r121_value = r121_value.to(tl.float32)
    r121_value = tl.broadcast_to(r121_value, [$(18432)])
    tl.store($(candidate_vmasked_fill_5_ptr) + r121_z, r121_value)
    r122_z = tl.arange($(18432))
    r122_value = (tl.load($(candidate_vmasked_fill_5_ptr) + ((((r122_z / $(2304)) % $(8)) * $(2304)) + (((r122_z / $(48)) % $(48)) * $(48)) + (r122_z % $(48))), dtype=tl.float32) * 0.07216878364870322)
    r122_value = r122_value.to(tl.float32)
    r122_value = tl.broadcast_to(r122_value, [$(18432)])
    tl.store($(candidate_vmul_28_ptr) + r122_z, r122_value)
    r123_row = tl.arange($(192))
    r123_col = tl.arange($(48))
    r123_lane = tl.arange($(48))
    for r123_group in range($(8)) {
      r123_left = tl.load($(candidate_vto_copy_29_ptr) + (((((((r123_group % $(8)) * $(9216)) + (r123_lane[None, :] * $(192)) + r123_row[:, None]) / $(9216)) % $(8)) * $(9216)) + ((((((r123_group % $(8)) * $(9216)) + (r123_lane[None, :] * $(192)) + r123_row[:, None]) / $(192)) % $(48)) * $(192)) + ((((r123_group % $(8)) * $(9216)) + (r123_lane[None, :] * $(192)) + r123_row[:, None]) % $(192))), dtype=tl.float32)
      r123_right = tl.load($(candidate_vmul_28_ptr) + (((((((r123_group % $(8)) * $(2304)) + (r123_lane[:, None] * $(48)) + r123_col[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r123_group % $(8)) * $(2304)) + (r123_lane[:, None] * $(48)) + r123_col[None, :]) / $(48)) % $(48)) * $(48)) + ((((r123_group % $(8)) * $(2304)) + (r123_lane[:, None] * $(48)) + r123_col[None, :]) % $(48))), dtype=tl.float32)
      r123_value = tl.dot(r123_left, r123_right)
      r123_value = r123_value.to(tl.float32)
      tl.store($(candidate_vbmm_18_ptr) + ((r123_group * $(9216)) + (r123_row[:, None] * $(48)) + r123_col[None, :]), r123_value)
    }
    r124_z = tl.arange($(131072))
    r124_value = tl.where(((((r124_z / $(1024)) % $(128)) >= $(0)) & (((r124_z / $(1024)) % $(128)) < $(48)) & ((((r124_z / $(1024)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vbmm_16_ptr) + ((((((((r124_z / $(128)) % $(8)) * $(6144)) + ((((r124_z / $(1024)) % $(128)) - $(0)) * $(128)) + (r124_z % $(128))) / $(6144)) % $(8)) * $(6144)) + ((((((r124_z / $(128)) % $(8)) * $(48)) + (((r124_z / $(1024)) % $(128)) - $(0))) % $(48)) * $(128)) + (r124_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r124_value = r124_value.to(tl.bfloat16)
    r124_value = tl.broadcast_to(r124_value, [$(131072)])
    tl.store($(candidate_vslice_backward_4_ptr) + r124_z, r124_value)
    r125_z = tl.arange($(131072))
    r125_value = (tl.load($(candidate_vadd_10_ptr) + ((((r125_z / $(1024)) % $(128)) * $(1024)) + (((r125_z / $(128)) % $(8)) * $(128)) + (r125_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_4_ptr) + ((((r125_z / $(1024)) % $(128)) * $(1024)) + (((r125_z / $(128)) % $(8)) * $(128)) + (r125_z % $(128))), dtype=tl.bfloat16))
    r125_value = r125_value.to(tl.bfloat16)
    r125_value = tl.broadcast_to(r125_value, [$(131072)])
    tl.store($(candidate_vadd_12_ptr) + r125_z, r125_value)
    r126_z = tl.arange($(73728))
    r126_value = tl.load($(candidate_vbmm_18_ptr) + ((((r126_z / $(9216)) % $(8)) * $(9216)) + ((r126_z % $(192)) * $(48)) + ((r126_z / $(192)) % $(48))), dtype=tl.float32)
    r126_value = r126_value.to(tl.bfloat16)
    r126_value = tl.broadcast_to(r126_value, [$(73728)])
    tl.store($(candidate_vto_copy_43_ptr) + r126_z, r126_value)
    r127_z = tl.arange($(196608))
    r127_value = tl.where(((((r127_z / $(1536)) % $(128)) >= $(0)) & (((r127_z / $(1536)) % $(128)) < $(48)) & ((((r127_z / $(1536)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vto_copy_43_ptr) + ((((r127_z / $(192)) % $(8)) * $(9216)) + (((((r127_z / $(1536)) % $(128)) - $(0)) % $(48)) * $(192)) + (r127_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r127_value = r127_value.to(tl.bfloat16)
    r127_value = tl.broadcast_to(r127_value, [$(196608)])
    tl.store($(candidate_vslice_backward_5_ptr) + r127_z, r127_value)
    r128_z = tl.arange($(196608))
    r128_value = (tl.load($(candidate_vadd_11_ptr) + ((((r128_z / $(1536)) % $(128)) * $(1536)) + (((r128_z / $(192)) % $(8)) * $(192)) + (r128_z % $(192))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_5_ptr) + ((((r128_z / $(1536)) % $(128)) * $(1536)) + (((r128_z / $(192)) % $(8)) * $(192)) + (r128_z % $(192))), dtype=tl.bfloat16))
    r128_value = r128_value.to(tl.bfloat16)
    r128_value = tl.broadcast_to(r128_value, [$(196608)])
    tl.store($(candidate_vadd_13_ptr) + r128_z, r128_value)
    r129_row = tl.arange($(256))
    r129_col = tl.arange($(1024))
    r129_lane = tl.arange($(128))
    for r129_group in range($(1)) {
      r129_left = tl.load($(candidate_vto_copy_24_ptr) + ((((((r129_lane[None, :] * $(256)) + r129_row[:, None]) / $(256)) % $(128)) * $(256)) + (((r129_lane[None, :] * $(256)) + r129_row[:, None]) % $(256))), dtype=tl.bfloat16)
      r129_right = tl.load($(candidate_vadd_12_ptr) + ((((((r129_lane[:, None] * $(1024)) + r129_col[None, :]) / $(1024)) % $(128)) * $(1024)) + (((((r129_lane[:, None] * $(1024)) + r129_col[None, :]) / $(128)) % $(8)) * $(128)) + (((r129_lane[:, None] * $(1024)) + r129_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r129_value = tl.dot(r129_left, r129_right)
      r129_value = r129_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_19_ptr) + ((r129_group * $(262144)) + (r129_row[:, None] * $(1024)) + r129_col[None, :]), r129_value)
    }
    r130_row = tl.arange($(256))
    r130_col = tl.arange($(1024))
    r130_lane = tl.arange($(128))
    for r130_group in range($(1)) {
      r130_left = tl.load($(candidate_vto_copy_24_ptr) + ((((((r130_lane[None, :] * $(256)) + r130_row[:, None]) / $(256)) % $(128)) * $(256)) + (((r130_lane[None, :] * $(256)) + r130_row[:, None]) % $(256))), dtype=tl.bfloat16)
      r130_right = tl.load($(candidate_vadd_13_ptr) + ((((((r130_lane[:, None] * $(1024)) + r130_col[None, :]) / $(1024)) % $(128)) * $(1536)) + (((((r130_lane[:, None] * $(1024)) + r130_col[None, :]) / $(128)) % $(8)) * $(192)) + (((r130_lane[:, None] * $(1024)) + r130_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r130_value = tl.dot(r130_left, r130_right)
      r130_value = r130_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_20_ptr) + ((r130_group * $(262144)) + (r130_row[:, None] * $(1024)) + r130_col[None, :]), r130_value)
    }
    r131_z = tl.arange($(524288))
    r131_value = tl.where(((((r131_z / $(256)) % $(256)) >= $(128)) & (((r131_z / $(256)) % $(256)) < $(256)) & ((((r131_z / $(256)) % $(256)) - $(128)) % $(1) == $(0))), tl.load($(candidate_vbmm_19_ptr) + (((r131_z % $(256)) * $(1024)) + (((((((r131_z / $(65536)) % $(8)) * $(32768)) + ((((r131_z / $(256)) % $(256)) - $(128)) * $(256)) + (r131_z % $(256))) / $(32768)) % $(8)) * $(128)) + (((((r131_z / $(65536)) % $(8)) * $(128)) + (((r131_z / $(256)) % $(256)) - $(128))) % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r131_value = r131_value.to(tl.bfloat16)
    r131_value = tl.broadcast_to(r131_value, [$(524288)])
    tl.store($(candidate_vslice_backward_6_ptr) + r131_z, r131_value)
    r132_z = tl.arange($(524288))
    r132_value = tl.where(((((r132_z / $(256)) % $(256)) >= $(0)) & (((r132_z / $(256)) % $(256)) < $(128)) & ((((r132_z / $(256)) % $(256)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vbmm_20_ptr) + (((r132_z % $(256)) * $(1024)) + (((((((r132_z / $(65536)) % $(8)) * $(32768)) + ((((r132_z / $(256)) % $(256)) - $(0)) * $(256)) + (r132_z % $(256))) / $(32768)) % $(8)) * $(128)) + (((((r132_z / $(65536)) % $(8)) * $(128)) + (((r132_z / $(256)) % $(256)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r132_value = r132_value.to(tl.bfloat16)
    r132_value = tl.broadcast_to(r132_value, [$(524288)])
    tl.store($(candidate_vslice_backward_7_ptr) + r132_z, r132_value)
    r133_z = tl.arange($(524288))
    r133_value = (tl.load($(candidate_vslice_backward_6_ptr) + ((((r133_z / $(65536)) % $(8)) * $(65536)) + (((r133_z / $(256)) % $(256)) * $(256)) + (r133_z % $(256))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_7_ptr) + ((((r133_z / $(65536)) % $(8)) * $(65536)) + (((r133_z / $(256)) % $(256)) * $(256)) + (r133_z % $(256))), dtype=tl.bfloat16))
    r133_value = r133_value.to(tl.bfloat16)
    r133_value = tl.broadcast_to(r133_value, [$(524288)])
    tl.store($(candidate_vadd_14_ptr) + r133_z, r133_value)
    z = tl.arange($(524288))
    value = tl.load($(candidate_vadd_14_ptr) + ((((((((z / $(256)) % $(2048)) * $(256)) + (z % $(256))) / $(65536)) % $(8)) * $(65536)) + ((((z / $(256)) % $(2048)) % $(256)) * $(256)) + (z % $(256))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(524288)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_mla_fa4_te_rope_thd_nonabsorb_wgrad
