import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_mla_fa4_te_rope_cp_nonabsorb_wgrad
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
    r16_value = (tl.load($(vcat_ptr) + r16_z, dtype=tl.int64) + tl.full([], $(2), dtype=tl.int64))
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
  let vto_copy_6_ptr : RegionName := "vto_copy_6"
  let vcat_ptr : RegionName := "vcat"
  let vadd_2_ptr : RegionName := "vadd_2"
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
  let vto_copy_25_ptr : RegionName := "vto_copy_25"
  let vsoftmax_ptr : RegionName := "vsoftmax"
  let vto_copy_26_ptr : RegionName := "vto_copy_26"
  let vbmm_4_ptr : RegionName := "vbmm_4"
  let vto_copy_27_ptr : RegionName := "vto_copy_27"
  let vsoftmax_backward_data_ptr : RegionName := "vsoftmax_backward_data"
  let vto_copy_28_ptr : RegionName := "vto_copy_28"
  let vmul_26_ptr : RegionName := "vmul_26"
  let vbmm_6_ptr : RegionName := "vbmm_6"
  let vcat_9_ptr : RegionName := "vcat_9"
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
  let candidate_vcat_ptr : RegionName := "candidate_vcat"
  let candidate_vadd_2_ptr : RegionName := "candidate_vadd_2"
  let candidate_vsort_value_ptr : RegionName := "candidate_vsort_value"
  let candidate_vsort_index_ptr : RegionName := "candidate_vsort_index"
  let candidate_vindex_ptr : RegionName := "candidate_vindex"
  let candidate_vsub_8_ptr : RegionName := "candidate_vsub_8"
  let candidate_vto_copy_20_ptr : RegionName := "candidate_vto_copy_20"
  let candidate_vindex_1_ptr : RegionName := "candidate_vindex_1"
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
  let candidate_vto_copy_25_ptr : RegionName := "candidate_vto_copy_25"
  let candidate_vadd_5_ptr : RegionName := "candidate_vadd_5"
  let candidate_vmul_18_ptr : RegionName := "candidate_vmul_18"
  let candidate_vcos_2_ptr : RegionName := "candidate_vcos_2"
  let candidate_vto_copy_26_ptr : RegionName := "candidate_vto_copy_26"
  let candidate_vsin_2_ptr : RegionName := "candidate_vsin_2"
  let candidate_vto_copy_27_ptr : RegionName := "candidate_vto_copy_27"
  let candidate_vneg_ptr : RegionName := "candidate_vneg"
  let candidate_vstack_ptr : RegionName := "candidate_vstack"
  let candidate_vmul_19_ptr : RegionName := "candidate_vmul_19"
  let candidate_vmul_20_ptr : RegionName := "candidate_vmul_20"
  let candidate_vadd_6_ptr : RegionName := "candidate_vadd_6"
  let candidate_vcos_3_ptr : RegionName := "candidate_vcos_3"
  let candidate_vto_copy_28_ptr : RegionName := "candidate_vto_copy_28"
  let candidate_vsin_3_ptr : RegionName := "candidate_vsin_3"
  let candidate_vto_copy_29_ptr : RegionName := "candidate_vto_copy_29"
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
  let candidate_vto_copy_30_ptr : RegionName := "candidate_vto_copy_30"
  let candidate_vto_copy_31_ptr : RegionName := "candidate_vto_copy_31"
  let candidate_vbmm_4_ptr : RegionName := "candidate_vbmm_4"
  let candidate_vmul_23_ptr : RegionName := "candidate_vmul_23"
  let candidate_vadd_8_ptr : RegionName := "candidate_vadd_8"
  let candidate_vsub_9_ptr : RegionName := "candidate_vsub_9"
  let candidate_vgt_ptr : RegionName := "candidate_vgt"
  let candidate_vmasked_fill_ptr : RegionName := "candidate_vmasked_fill"
  let candidate_vsoftmax_ptr : RegionName := "candidate_vsoftmax"
  let candidate_vto_copy_32_ptr : RegionName := "candidate_vto_copy_32"
  let candidate_vto_copy_33_ptr : RegionName := "candidate_vto_copy_33"
  let candidate_vto_copy_34_ptr : RegionName := "candidate_vto_copy_34"
  let candidate_vbmm_6_ptr : RegionName := "candidate_vbmm_6"
  let candidate_vmul_24_ptr : RegionName := "candidate_vmul_24"
  let candidate_vadd_9_ptr : RegionName := "candidate_vadd_9"
  let candidate_vsub_10_ptr : RegionName := "candidate_vsub_10"
  let candidate_vgt_1_ptr : RegionName := "candidate_vgt_1"
  let candidate_vmasked_fill_1_ptr : RegionName := "candidate_vmasked_fill_1"
  let candidate_vsoftmax_1_ptr : RegionName := "candidate_vsoftmax_1"
  let candidate_vto_copy_35_ptr : RegionName := "candidate_vto_copy_35"
  let candidate_vto_copy_36_ptr : RegionName := "candidate_vto_copy_36"
  let candidate_vto_copy_37_ptr : RegionName := "candidate_vto_copy_37"
  let candidate_vbmm_8_ptr : RegionName := "candidate_vbmm_8"
  let candidate_vmul_25_ptr : RegionName := "candidate_vmul_25"
  let candidate_vadd_10_ptr : RegionName := "candidate_vadd_10"
  let candidate_vsub_11_ptr : RegionName := "candidate_vsub_11"
  let candidate_vgt_2_ptr : RegionName := "candidate_vgt_2"
  let candidate_vmasked_fill_2_ptr : RegionName := "candidate_vmasked_fill_2"
  let candidate_vsoftmax_2_ptr : RegionName := "candidate_vsoftmax_2"
  let candidate_vto_copy_38_ptr : RegionName := "candidate_vto_copy_38"
  let candidate_vbmm_10_ptr : RegionName := "candidate_vbmm_10"
  let candidate_vbmm_11_ptr : RegionName := "candidate_vbmm_11"
  let candidate_vcat_9_ptr : RegionName := "candidate_vcat_9"
  let candidate_vcat_10_ptr : RegionName := "candidate_vcat_10"
  let candidate_vto_copy_39_ptr : RegionName := "candidate_vto_copy_39"
  let candidate_vto_copy_40_ptr : RegionName := "candidate_vto_copy_40"
  let candidate_vbmm_12_ptr : RegionName := "candidate_vbmm_12"
  let candidate_vmul_26_ptr : RegionName := "candidate_vmul_26"
  let candidate_vadd_11_ptr : RegionName := "candidate_vadd_11"
  let candidate_vsub_12_ptr : RegionName := "candidate_vsub_12"
  let candidate_vgt_3_ptr : RegionName := "candidate_vgt_3"
  let candidate_vmasked_fill_3_ptr : RegionName := "candidate_vmasked_fill_3"
  let candidate_vsoftmax_3_ptr : RegionName := "candidate_vsoftmax_3"
  let candidate_vto_copy_41_ptr : RegionName := "candidate_vto_copy_41"
  let candidate_vto_copy_42_ptr : RegionName := "candidate_vto_copy_42"
  let candidate_vto_copy_43_ptr : RegionName := "candidate_vto_copy_43"
  let candidate_vbmm_14_ptr : RegionName := "candidate_vbmm_14"
  let candidate_vmul_27_ptr : RegionName := "candidate_vmul_27"
  let candidate_vadd_12_ptr : RegionName := "candidate_vadd_12"
  let candidate_vsub_13_ptr : RegionName := "candidate_vsub_13"
  let candidate_vgt_4_ptr : RegionName := "candidate_vgt_4"
  let candidate_vmasked_fill_4_ptr : RegionName := "candidate_vmasked_fill_4"
  let candidate_vsoftmax_4_ptr : RegionName := "candidate_vsoftmax_4"
  let candidate_vto_copy_44_ptr : RegionName := "candidate_vto_copy_44"
  let candidate_vto_copy_45_ptr : RegionName := "candidate_vto_copy_45"
  let candidate_vto_copy_46_ptr : RegionName := "candidate_vto_copy_46"
  let candidate_vbmm_16_ptr : RegionName := "candidate_vbmm_16"
  let candidate_vmul_28_ptr : RegionName := "candidate_vmul_28"
  let candidate_vadd_13_ptr : RegionName := "candidate_vadd_13"
  let candidate_vsub_14_ptr : RegionName := "candidate_vsub_14"
  let candidate_vgt_5_ptr : RegionName := "candidate_vgt_5"
  let candidate_vmasked_fill_5_ptr : RegionName := "candidate_vmasked_fill_5"
  let candidate_vsoftmax_5_ptr : RegionName := "candidate_vsoftmax_5"
  let candidate_vto_copy_47_ptr : RegionName := "candidate_vto_copy_47"
  let candidate_vbmm_18_ptr : RegionName := "candidate_vbmm_18"
  let candidate_vbmm_19_ptr : RegionName := "candidate_vbmm_19"
  let candidate_vcat_12_ptr : RegionName := "candidate_vcat_12"
  let candidate_vcat_13_ptr : RegionName := "candidate_vcat_13"
  let candidate_vto_copy_48_ptr : RegionName := "candidate_vto_copy_48"
  let candidate_vto_copy_49_ptr : RegionName := "candidate_vto_copy_49"
  let candidate_vbmm_20_ptr : RegionName := "candidate_vbmm_20"
  let candidate_vmul_29_ptr : RegionName := "candidate_vmul_29"
  let candidate_vadd_14_ptr : RegionName := "candidate_vadd_14"
  let candidate_vsub_15_ptr : RegionName := "candidate_vsub_15"
  let candidate_vgt_6_ptr : RegionName := "candidate_vgt_6"
  let candidate_vmasked_fill_6_ptr : RegionName := "candidate_vmasked_fill_6"
  let candidate_vsoftmax_6_ptr : RegionName := "candidate_vsoftmax_6"
  let candidate_vto_copy_50_ptr : RegionName := "candidate_vto_copy_50"
  let candidate_vto_copy_51_ptr : RegionName := "candidate_vto_copy_51"
  let candidate_vto_copy_52_ptr : RegionName := "candidate_vto_copy_52"
  let candidate_vbmm_22_ptr : RegionName := "candidate_vbmm_22"
  let candidate_vmul_30_ptr : RegionName := "candidate_vmul_30"
  let candidate_vadd_15_ptr : RegionName := "candidate_vadd_15"
  let candidate_vsub_16_ptr : RegionName := "candidate_vsub_16"
  let candidate_vgt_7_ptr : RegionName := "candidate_vgt_7"
  let candidate_vmasked_fill_7_ptr : RegionName := "candidate_vmasked_fill_7"
  let candidate_vsoftmax_7_ptr : RegionName := "candidate_vsoftmax_7"
  let candidate_vto_copy_53_ptr : RegionName := "candidate_vto_copy_53"
  let candidate_vto_copy_54_ptr : RegionName := "candidate_vto_copy_54"
  let candidate_vto_copy_55_ptr : RegionName := "candidate_vto_copy_55"
  let candidate_vbmm_24_ptr : RegionName := "candidate_vbmm_24"
  let candidate_vmul_31_ptr : RegionName := "candidate_vmul_31"
  let candidate_vadd_16_ptr : RegionName := "candidate_vadd_16"
  let candidate_vsub_17_ptr : RegionName := "candidate_vsub_17"
  let candidate_vgt_8_ptr : RegionName := "candidate_vgt_8"
  let candidate_vmasked_fill_8_ptr : RegionName := "candidate_vmasked_fill_8"
  let candidate_vsoftmax_8_ptr : RegionName := "candidate_vsoftmax_8"
  let candidate_vto_copy_56_ptr : RegionName := "candidate_vto_copy_56"
  let candidate_vbmm_26_ptr : RegionName := "candidate_vbmm_26"
  let candidate_vbmm_27_ptr : RegionName := "candidate_vbmm_27"
  let candidate_vcat_15_ptr : RegionName := "candidate_vcat_15"
  let candidate_vcat_16_ptr : RegionName := "candidate_vcat_16"
  let candidate_vto_copy_57_ptr : RegionName := "candidate_vto_copy_57"
  let candidate_vto_copy_58_ptr : RegionName := "candidate_vto_copy_58"
  let candidate_vbmm_28_ptr : RegionName := "candidate_vbmm_28"
  let candidate_vmul_32_ptr : RegionName := "candidate_vmul_32"
  let candidate_vadd_17_ptr : RegionName := "candidate_vadd_17"
  let candidate_vsub_18_ptr : RegionName := "candidate_vsub_18"
  let candidate_vgt_9_ptr : RegionName := "candidate_vgt_9"
  let candidate_vmasked_fill_9_ptr : RegionName := "candidate_vmasked_fill_9"
  let candidate_vsoftmax_9_ptr : RegionName := "candidate_vsoftmax_9"
  let candidate_vto_copy_59_ptr : RegionName := "candidate_vto_copy_59"
  let candidate_vto_copy_60_ptr : RegionName := "candidate_vto_copy_60"
  let candidate_vto_copy_61_ptr : RegionName := "candidate_vto_copy_61"
  let candidate_vbmm_30_ptr : RegionName := "candidate_vbmm_30"
  let candidate_vmul_33_ptr : RegionName := "candidate_vmul_33"
  let candidate_vadd_18_ptr : RegionName := "candidate_vadd_18"
  let candidate_vsub_19_ptr : RegionName := "candidate_vsub_19"
  let candidate_vgt_10_ptr : RegionName := "candidate_vgt_10"
  let candidate_vmasked_fill_10_ptr : RegionName := "candidate_vmasked_fill_10"
  let candidate_vsoftmax_10_ptr : RegionName := "candidate_vsoftmax_10"
  let candidate_vto_copy_62_ptr : RegionName := "candidate_vto_copy_62"
  let candidate_vto_copy_63_ptr : RegionName := "candidate_vto_copy_63"
  let candidate_vto_copy_64_ptr : RegionName := "candidate_vto_copy_64"
  let candidate_vbmm_32_ptr : RegionName := "candidate_vbmm_32"
  let candidate_vmul_34_ptr : RegionName := "candidate_vmul_34"
  let candidate_vadd_19_ptr : RegionName := "candidate_vadd_19"
  let candidate_vsub_20_ptr : RegionName := "candidate_vsub_20"
  let candidate_vgt_11_ptr : RegionName := "candidate_vgt_11"
  let candidate_vmasked_fill_11_ptr : RegionName := "candidate_vmasked_fill_11"
  let candidate_vsoftmax_11_ptr : RegionName := "candidate_vsoftmax_11"
  let candidate_vto_copy_65_ptr : RegionName := "candidate_vto_copy_65"
  let candidate_vindex_put_ptr : RegionName := "candidate_vindex_put"
  let candidate_vmm_4_ptr : RegionName := "candidate_vmm_4"
  let candidate_vbmm_34_ptr : RegionName := "candidate_vbmm_34"
  let candidate_vbmm_35_ptr : RegionName := "candidate_vbmm_35"
  let candidate_vto_copy_66_ptr : RegionName := "candidate_vto_copy_66"
  let candidate_vsoftmax_backward_data_ptr : RegionName := "candidate_vsoftmax_backward_data"
  let candidate_vmasked_fill_12_ptr : RegionName := "candidate_vmasked_fill_12"
  let candidate_vmul_35_ptr : RegionName := "candidate_vmul_35"
  let candidate_vbmm_36_ptr : RegionName := "candidate_vbmm_36"
  let candidate_vslice_backward_ptr : RegionName := "candidate_vslice_backward"
  let candidate_vto_copy_67_ptr : RegionName := "candidate_vto_copy_67"
  let candidate_vslice_backward_1_ptr : RegionName := "candidate_vslice_backward_1"
  let candidate_vbmm_37_ptr : RegionName := "candidate_vbmm_37"
  let candidate_vbmm_38_ptr : RegionName := "candidate_vbmm_38"
  let candidate_vto_copy_68_ptr : RegionName := "candidate_vto_copy_68"
  let candidate_vsoftmax_backward_data_1_ptr : RegionName := "candidate_vsoftmax_backward_data_1"
  let candidate_vmasked_fill_13_ptr : RegionName := "candidate_vmasked_fill_13"
  let candidate_vmul_36_ptr : RegionName := "candidate_vmul_36"
  let candidate_vbmm_39_ptr : RegionName := "candidate_vbmm_39"
  let candidate_vslice_backward_2_ptr : RegionName := "candidate_vslice_backward_2"
  let candidate_vadd_20_ptr : RegionName := "candidate_vadd_20"
  let candidate_vto_copy_69_ptr : RegionName := "candidate_vto_copy_69"
  let candidate_vslice_backward_3_ptr : RegionName := "candidate_vslice_backward_3"
  let candidate_vadd_21_ptr : RegionName := "candidate_vadd_21"
  let candidate_vbmm_40_ptr : RegionName := "candidate_vbmm_40"
  let candidate_vbmm_41_ptr : RegionName := "candidate_vbmm_41"
  let candidate_vto_copy_70_ptr : RegionName := "candidate_vto_copy_70"
  let candidate_vsoftmax_backward_data_2_ptr : RegionName := "candidate_vsoftmax_backward_data_2"
  let candidate_vmasked_fill_14_ptr : RegionName := "candidate_vmasked_fill_14"
  let candidate_vmul_37_ptr : RegionName := "candidate_vmul_37"
  let candidate_vbmm_42_ptr : RegionName := "candidate_vbmm_42"
  let candidate_vslice_backward_4_ptr : RegionName := "candidate_vslice_backward_4"
  let candidate_vadd_22_ptr : RegionName := "candidate_vadd_22"
  let candidate_vto_copy_71_ptr : RegionName := "candidate_vto_copy_71"
  let candidate_vslice_backward_5_ptr : RegionName := "candidate_vslice_backward_5"
  let candidate_vadd_23_ptr : RegionName := "candidate_vadd_23"
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
  let candidate_vbmm_43_ptr : RegionName := "candidate_vbmm_43"
  let candidate_vbmm_44_ptr : RegionName := "candidate_vbmm_44"
  let candidate_vslice_backward_6_ptr : RegionName := "candidate_vslice_backward_6"
  let candidate_vslice_backward_7_ptr : RegionName := "candidate_vslice_backward_7"
  let candidate_vadd_24_ptr : RegionName := "candidate_vadd_24"
  let candidate_vbmm_45_ptr : RegionName := "candidate_vbmm_45"
  let candidate_vbmm_46_ptr : RegionName := "candidate_vbmm_46"
  let candidate_vto_copy_72_ptr : RegionName := "candidate_vto_copy_72"
  let candidate_vsoftmax_backward_data_3_ptr : RegionName := "candidate_vsoftmax_backward_data_3"
  let candidate_vmasked_fill_15_ptr : RegionName := "candidate_vmasked_fill_15"
  let candidate_vmul_38_ptr : RegionName := "candidate_vmul_38"
  let candidate_vbmm_47_ptr : RegionName := "candidate_vbmm_47"
  let candidate_vslice_backward_8_ptr : RegionName := "candidate_vslice_backward_8"
  let candidate_vto_copy_73_ptr : RegionName := "candidate_vto_copy_73"
  let candidate_vslice_backward_9_ptr : RegionName := "candidate_vslice_backward_9"
  let candidate_vbmm_48_ptr : RegionName := "candidate_vbmm_48"
  let candidate_vbmm_49_ptr : RegionName := "candidate_vbmm_49"
  let candidate_vto_copy_74_ptr : RegionName := "candidate_vto_copy_74"
  let candidate_vsoftmax_backward_data_4_ptr : RegionName := "candidate_vsoftmax_backward_data_4"
  let candidate_vmasked_fill_16_ptr : RegionName := "candidate_vmasked_fill_16"
  let candidate_vmul_39_ptr : RegionName := "candidate_vmul_39"
  let candidate_vbmm_50_ptr : RegionName := "candidate_vbmm_50"
  let candidate_vslice_backward_10_ptr : RegionName := "candidate_vslice_backward_10"
  let candidate_vadd_25_ptr : RegionName := "candidate_vadd_25"
  let candidate_vto_copy_75_ptr : RegionName := "candidate_vto_copy_75"
  let candidate_vslice_backward_11_ptr : RegionName := "candidate_vslice_backward_11"
  let candidate_vadd_26_ptr : RegionName := "candidate_vadd_26"
  let candidate_vbmm_51_ptr : RegionName := "candidate_vbmm_51"
  let candidate_vbmm_52_ptr : RegionName := "candidate_vbmm_52"
  let candidate_vto_copy_76_ptr : RegionName := "candidate_vto_copy_76"
  let candidate_vsoftmax_backward_data_5_ptr : RegionName := "candidate_vsoftmax_backward_data_5"
  let candidate_vmasked_fill_17_ptr : RegionName := "candidate_vmasked_fill_17"
  let candidate_vmul_40_ptr : RegionName := "candidate_vmul_40"
  let candidate_vbmm_53_ptr : RegionName := "candidate_vbmm_53"
  let candidate_vslice_backward_12_ptr : RegionName := "candidate_vslice_backward_12"
  let candidate_vadd_27_ptr : RegionName := "candidate_vadd_27"
  let candidate_vto_copy_77_ptr : RegionName := "candidate_vto_copy_77"
  let candidate_vslice_backward_13_ptr : RegionName := "candidate_vslice_backward_13"
  let candidate_vadd_28_ptr : RegionName := "candidate_vadd_28"
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
  let candidate_vbmm_54_ptr : RegionName := "candidate_vbmm_54"
  let candidate_vbmm_55_ptr : RegionName := "candidate_vbmm_55"
  let candidate_vslice_backward_14_ptr : RegionName := "candidate_vslice_backward_14"
  let candidate_vslice_backward_15_ptr : RegionName := "candidate_vslice_backward_15"
  let candidate_vadd_29_ptr : RegionName := "candidate_vadd_29"
  let candidate_vadd_30_ptr : RegionName := "candidate_vadd_30"
  let candidate_vbmm_56_ptr : RegionName := "candidate_vbmm_56"
  let candidate_vbmm_57_ptr : RegionName := "candidate_vbmm_57"
  let candidate_vto_copy_78_ptr : RegionName := "candidate_vto_copy_78"
  let candidate_vsoftmax_backward_data_6_ptr : RegionName := "candidate_vsoftmax_backward_data_6"
  let candidate_vmasked_fill_18_ptr : RegionName := "candidate_vmasked_fill_18"
  let candidate_vmul_41_ptr : RegionName := "candidate_vmul_41"
  let candidate_vbmm_58_ptr : RegionName := "candidate_vbmm_58"
  let candidate_vslice_backward_16_ptr : RegionName := "candidate_vslice_backward_16"
  let candidate_vto_copy_79_ptr : RegionName := "candidate_vto_copy_79"
  let candidate_vslice_backward_17_ptr : RegionName := "candidate_vslice_backward_17"
  let candidate_vbmm_59_ptr : RegionName := "candidate_vbmm_59"
  let candidate_vbmm_60_ptr : RegionName := "candidate_vbmm_60"
  let candidate_vto_copy_80_ptr : RegionName := "candidate_vto_copy_80"
  let candidate_vsoftmax_backward_data_7_ptr : RegionName := "candidate_vsoftmax_backward_data_7"
  let candidate_vmasked_fill_19_ptr : RegionName := "candidate_vmasked_fill_19"
  let candidate_vmul_42_ptr : RegionName := "candidate_vmul_42"
  let candidate_vbmm_61_ptr : RegionName := "candidate_vbmm_61"
  let candidate_vslice_backward_18_ptr : RegionName := "candidate_vslice_backward_18"
  let candidate_vadd_31_ptr : RegionName := "candidate_vadd_31"
  let candidate_vto_copy_81_ptr : RegionName := "candidate_vto_copy_81"
  let candidate_vslice_backward_19_ptr : RegionName := "candidate_vslice_backward_19"
  let candidate_vadd_32_ptr : RegionName := "candidate_vadd_32"
  let candidate_vbmm_62_ptr : RegionName := "candidate_vbmm_62"
  let candidate_vbmm_63_ptr : RegionName := "candidate_vbmm_63"
  let candidate_vto_copy_82_ptr : RegionName := "candidate_vto_copy_82"
  let candidate_vsoftmax_backward_data_8_ptr : RegionName := "candidate_vsoftmax_backward_data_8"
  let candidate_vmasked_fill_20_ptr : RegionName := "candidate_vmasked_fill_20"
  let candidate_vmul_43_ptr : RegionName := "candidate_vmul_43"
  let candidate_vbmm_64_ptr : RegionName := "candidate_vbmm_64"
  let candidate_vslice_backward_20_ptr : RegionName := "candidate_vslice_backward_20"
  let candidate_vadd_33_ptr : RegionName := "candidate_vadd_33"
  let candidate_vto_copy_83_ptr : RegionName := "candidate_vto_copy_83"
  let candidate_vslice_backward_21_ptr : RegionName := "candidate_vslice_backward_21"
  let candidate_vadd_34_ptr : RegionName := "candidate_vadd_34"
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
  let candidate_vbmm_65_ptr : RegionName := "candidate_vbmm_65"
  let candidate_vbmm_66_ptr : RegionName := "candidate_vbmm_66"
  let candidate_vslice_backward_22_ptr : RegionName := "candidate_vslice_backward_22"
  let candidate_vslice_backward_23_ptr : RegionName := "candidate_vslice_backward_23"
  let candidate_vadd_35_ptr : RegionName := "candidate_vadd_35"
  let candidate_vadd_36_ptr : RegionName := "candidate_vadd_36"
  let candidate_vbmm_67_ptr : RegionName := "candidate_vbmm_67"
  let candidate_vbmm_68_ptr : RegionName := "candidate_vbmm_68"
  let candidate_vto_copy_84_ptr : RegionName := "candidate_vto_copy_84"
  let candidate_vsoftmax_backward_data_9_ptr : RegionName := "candidate_vsoftmax_backward_data_9"
  let candidate_vmasked_fill_21_ptr : RegionName := "candidate_vmasked_fill_21"
  let candidate_vmul_44_ptr : RegionName := "candidate_vmul_44"
  let candidate_vbmm_69_ptr : RegionName := "candidate_vbmm_69"
  let candidate_vslice_backward_24_ptr : RegionName := "candidate_vslice_backward_24"
  let candidate_vto_copy_85_ptr : RegionName := "candidate_vto_copy_85"
  let candidate_vslice_backward_25_ptr : RegionName := "candidate_vslice_backward_25"
  let candidate_vbmm_70_ptr : RegionName := "candidate_vbmm_70"
  let candidate_vbmm_71_ptr : RegionName := "candidate_vbmm_71"
  let candidate_vto_copy_86_ptr : RegionName := "candidate_vto_copy_86"
  let candidate_vsoftmax_backward_data_10_ptr : RegionName := "candidate_vsoftmax_backward_data_10"
  let candidate_vmasked_fill_22_ptr : RegionName := "candidate_vmasked_fill_22"
  let candidate_vmul_45_ptr : RegionName := "candidate_vmul_45"
  let candidate_vbmm_72_ptr : RegionName := "candidate_vbmm_72"
  let candidate_vslice_backward_26_ptr : RegionName := "candidate_vslice_backward_26"
  let candidate_vadd_37_ptr : RegionName := "candidate_vadd_37"
  let candidate_vto_copy_87_ptr : RegionName := "candidate_vto_copy_87"
  let candidate_vslice_backward_27_ptr : RegionName := "candidate_vslice_backward_27"
  let candidate_vadd_38_ptr : RegionName := "candidate_vadd_38"
  let candidate_vbmm_73_ptr : RegionName := "candidate_vbmm_73"
  let candidate_vbmm_74_ptr : RegionName := "candidate_vbmm_74"
  let candidate_vto_copy_88_ptr : RegionName := "candidate_vto_copy_88"
  let candidate_vsoftmax_backward_data_11_ptr : RegionName := "candidate_vsoftmax_backward_data_11"
  let candidate_vmasked_fill_23_ptr : RegionName := "candidate_vmasked_fill_23"
  let candidate_vmul_46_ptr : RegionName := "candidate_vmul_46"
  let candidate_vbmm_75_ptr : RegionName := "candidate_vbmm_75"
  let candidate_vslice_backward_28_ptr : RegionName := "candidate_vslice_backward_28"
  let candidate_vadd_39_ptr : RegionName := "candidate_vadd_39"
  let candidate_vto_copy_89_ptr : RegionName := "candidate_vto_copy_89"
  let candidate_vslice_backward_29_ptr : RegionName := "candidate_vslice_backward_29"
  let candidate_vadd_40_ptr : RegionName := "candidate_vadd_40"
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
  let candidate_vbmm_76_ptr : RegionName := "candidate_vbmm_76"
  let candidate_vbmm_77_ptr : RegionName := "candidate_vbmm_77"
  let candidate_vslice_backward_30_ptr : RegionName := "candidate_vslice_backward_30"
  let candidate_vslice_backward_31_ptr : RegionName := "candidate_vslice_backward_31"
  let candidate_vadd_41_ptr : RegionName := "candidate_vadd_41"
  let candidate_vadd_42_ptr : RegionName := "candidate_vadd_42"
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
    r15_z = tl.arange($(128))
    r15_value = tl.where(r15_z < $(48), tl.cast((r15_z - $(0)), tl.int64), tl.where(r15_z < $(64), tl.cast((r15_z - $(48)), tl.int64), tl.cast((r15_z - $(64)), tl.int64)))
    r15_value = tl.broadcast_to(r15_value, [$(128)])
    tl.store($(candidate_vcat_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(128))
    r16_value = (tl.load($(candidate_vcat_ptr) + r16_z, dtype=tl.int64) + tl.full([], $(2), dtype=tl.int64))
    r16_value = tl.broadcast_to(r16_value, [$(128)])
    tl.store($(candidate_vadd_2_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(128))
    r17_lane = tl.arange($(128))
    r17_row_values = tl.load($(constantee33028f0f_ptr) + r17_lane[None, :], dtype=tl.int64)
    r17_rank = tl.full([$(128), $(128)], $(0), dtype=tl.int64)
    for r17_other in range($(128)) {
      r17_rank = r17_rank + tl.where(((tl.load($(constantee33028f0f_ptr) + (r17_other % $(128)), dtype=tl.int64) < r17_row_values) | ((tl.load($(constantee33028f0f_ptr) + (r17_other % $(128)), dtype=tl.int64) == r17_row_values) & (r17_other < r17_lane[None, :]))), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))
    }
    r17_selected = r17_rank == tl.cast(r17_z[:, None], tl.int64)
    r17_value = tl.full([$(128)], $(0), dtype=tl.int64)
    for r17_cutoff in range($(128)) {
      r17_value = r17_value + tl.cast(tl.sum(r17_selected & (r17_lane[None, :] > r17_cutoff), axis=1), tl.int64)
    }
    r17_value = tl.load($(constantee33028f0f_ptr) + tl.cast(r17_value, tl.uint32), dtype=tl.int64)
    r17_value = tl.broadcast_to(r17_value, [$(128)])
    tl.store($(candidate_vsort_value_ptr) + r17_z, r17_value)
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
    r18_value = tl.broadcast_to(r18_value, [$(128)])
    tl.store($(candidate_vsort_index_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(128))
    r19_value = tl.load($(candidate_vadd_2_ptr) + (tl.load($(constantee33028f0f_ptr) + r19_z, dtype=tl.int64) % $(128)), dtype=tl.int64)
    r19_value = tl.broadcast_to(r19_value, [$(128)])
    tl.store($(candidate_vindex_ptr) + r19_z, r19_value)
    r20_z = tl.arange($(128))
    r20_value = (tl.load($(candidate_vindex_ptr) + r20_z, dtype=tl.int64) - tl.full([], $(2), dtype=tl.int64))
    r20_value = tl.broadcast_to(r20_value, [$(128)])
    tl.store($(candidate_vsub_8_ptr) + r20_z, r20_value)
    r21_z = tl.arange($(524288))
    r21_value = tl.load($(candidate_vmul_5_ptr) + ((((r21_z / $(256)) % $(2048)) * $(256)) + (r21_z % $(256))), dtype=tl.float32)
    r21_value = r21_value.to(tl.bfloat16)
    r21_value = tl.broadcast_to(r21_value, [$(524288)])
    tl.store($(candidate_vto_copy_20_ptr) + r21_z, r21_value)
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
    tl.store($(candidate_vto_copy_21_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(32768))
    r25_value = ((tl.load($(candidate_vto_copy_21_ptr) + ((((r25_z / $(256)) % $(128)) * $(256)) + (r25_z % $(256))), dtype=tl.float32)) * (tl.load($(candidate_vto_copy_21_ptr) + ((((r25_z / $(256)) % $(128)) * $(256)) + (r25_z % $(256))), dtype=tl.float32)))
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
    r29_value = (tl.load($(candidate_vto_copy_21_ptr) + ((((r29_z / $(256)) % $(128)) * $(256)) + (r29_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_ptr) + ((r29_z / $(256)) % $(128)), dtype=tl.float32))
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
    tl.store($(candidate_vto_copy_22_ptr) + r31_z, r31_value)
    r32_row = tl.arange($(128))
    r32_col = tl.arange($(1536))
    r32_lane = tl.arange($(256))
    r32_left = tl.load($(candidate_vto_copy_22_ptr) + ((((((r32_row[:, None] * $(256)) + r32_lane[None, :]) / $(256)) % $(128)) * $(256)) + (((r32_row[:, None] * $(256)) + r32_lane[None, :]) % $(256))), dtype=tl.bfloat16)
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
    tl.store($(candidate_vto_copy_23_ptr) + r34_z, r34_value)
    r35_z = tl.arange($(32768))
    r35_value = ((tl.load($(candidate_vto_copy_23_ptr) + ((((r35_z / $(256)) % $(128)) * $(256)) + (r35_z % $(256))), dtype=tl.float32)) * (tl.load($(candidate_vto_copy_23_ptr) + ((((r35_z / $(256)) % $(128)) * $(256)) + (r35_z % $(256))), dtype=tl.float32)))
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
    r39_value = (tl.load($(candidate_vto_copy_23_ptr) + ((((r39_z / $(256)) % $(128)) * $(256)) + (r39_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_1_ptr) + ((r39_z / $(256)) % $(128)), dtype=tl.float32))
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
    tl.store($(candidate_vto_copy_24_ptr) + r41_z, r41_value)
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
    tl.store($(candidate_vto_copy_25_ptr) + r46_z, r46_value)
    r47_z = tl.arange($(128))
    r47_value = (tl.load($(candidate_vto_copy_25_ptr) + r47_z, dtype=tl.float32) + 2.0)
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
    tl.store($(candidate_vto_copy_26_ptr) + r50_z, r50_value)
    r51_z = tl.arange($(8192))
    r51_value = tl.sin(tl.load($(candidate_vmul_18_ptr) + ((((r51_z / $(64)) % $(128)) * $(32)) + ((((((r51_z / $(64)) % $(128)) * $(64)) + (r51_z % $(64))) / $(2)) % $(32))), dtype=tl.float32))
    r51_value = r51_value.to(tl.float32)
    r51_value = tl.broadcast_to(r51_value, [$(8192)])
    tl.store($(candidate_vsin_2_ptr) + r51_z, r51_value)
    r52_z = tl.arange($(8192))
    r52_value = tl.load($(candidate_vsin_2_ptr) + ((((r52_z / $(64)) % $(128)) * $(64)) + (r52_z % $(64))), dtype=tl.float32)
    r52_value = r52_value.to(tl.bfloat16)
    r52_value = tl.broadcast_to(r52_value, [$(8192)])
    tl.store($(candidate_vto_copy_27_ptr) + r52_z, r52_value)
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
    r55_value = (tl.load($(candidate_vmm_1_ptr) + ((((r55_z / $(512)) % $(128)) * $(1536)) + (((r55_z / $(64)) % $(8)) * $(192)) + (r55_z % $(64)) + $(128)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_26_ptr) + ((((r55_z / $(512)) % $(128)) * $(64)) + (r55_z % $(64))), dtype=tl.bfloat16))
    r55_value = r55_value.to(tl.bfloat16)
    r55_value = tl.broadcast_to(r55_value, [$(65536)])
    tl.store($(candidate_vmul_19_ptr) + r55_z, r55_value)
    r56_z = tl.arange($(65536))
    r56_value = (tl.load($(candidate_vstack_ptr) + ((((r56_z / $(512)) % $(128)) * $(512)) + (((r56_z / $(64)) % $(8)) * $(64)) + (((((((r56_z / $(512)) % $(128)) * $(512)) + (((r56_z / $(64)) % $(8)) * $(64)) + (r56_z % $(64))) / $(2)) % $(32)) * $(2)) + (((((r56_z / $(512)) % $(128)) * $(512)) + (((r56_z / $(64)) % $(8)) * $(64)) + (r56_z % $(64))) % $(2))), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_27_ptr) + ((((r56_z / $(512)) % $(128)) * $(64)) + (r56_z % $(64))), dtype=tl.bfloat16))
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
    tl.store($(candidate_vto_copy_28_ptr) + r59_z, r59_value)
    r60_z = tl.arange($(8192))
    r60_value = tl.sin(tl.load($(candidate_vmul_18_ptr) + ((((r60_z / $(64)) % $(128)) * $(32)) + ((((((r60_z / $(64)) % $(128)) * $(64)) + (r60_z % $(64))) / $(2)) % $(32))), dtype=tl.float32))
    r60_value = r60_value.to(tl.float32)
    r60_value = tl.broadcast_to(r60_value, [$(8192)])
    tl.store($(candidate_vsin_3_ptr) + r60_z, r60_value)
    r61_z = tl.arange($(8192))
    r61_value = tl.load($(candidate_vsin_3_ptr) + ((((r61_z / $(64)) % $(128)) * $(64)) + (r61_z % $(64))), dtype=tl.float32)
    r61_value = r61_value.to(tl.bfloat16)
    r61_value = tl.broadcast_to(r61_value, [$(8192)])
    tl.store($(candidate_vto_copy_29_ptr) + r61_z, r61_value)
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
    r64_value = (tl.load($(candidate_vmm_2_ptr) + ((((r64_z / $(64)) % $(128)) * $(320)) + (r64_z % $(64)) + $(256)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_28_ptr) + ((((r64_z / $(64)) % $(128)) * $(64)) + (r64_z % $(64))), dtype=tl.bfloat16))
    r64_value = r64_value.to(tl.bfloat16)
    r64_value = tl.broadcast_to(r64_value, [$(8192)])
    tl.store($(candidate_vmul_21_ptr) + r64_z, r64_value)
    r65_z = tl.arange($(8192))
    r65_value = (tl.load($(candidate_vstack_1_ptr) + ((((r65_z / $(64)) % $(128)) * $(64)) + (((((((r65_z / $(64)) % $(128)) * $(64)) + (r65_z % $(64))) / $(2)) % $(32)) * $(2)) + (((((r65_z / $(64)) % $(128)) * $(64)) + (r65_z % $(64))) % $(2))), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_29_ptr) + ((((r65_z / $(64)) % $(128)) * $(64)) + (r65_z % $(64))), dtype=tl.bfloat16))
    r65_value = r65_value.to(tl.bfloat16)
    r65_value = tl.broadcast_to(r65_value, [$(8192)])
    tl.store($(candidate_vmul_22_ptr) + r65_z, r65_value)
    r66_z = tl.arange($(8192))
    r66_value = (tl.load($(candidate_vmul_21_ptr) + ((((r66_z / $(64)) % $(128)) * $(64)) + (r66_z % $(64))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_22_ptr) + ((((r66_z / $(64)) % $(128)) * $(64)) + (r66_z % $(64))), dtype=tl.bfloat16))
    r66_value = r66_value.to(tl.bfloat16)
    r66_value = tl.broadcast_to(r66_value, [$(8192)])
    tl.store($(candidate_vadd_7_ptr) + r66_z, r66_value)
    r67_z = tl.arange($(40960))
    r67_value = tl.where((r67_z % $(320)) < $(256), tl.load($(candidate_vto_copy_24_ptr) + ((((((((r67_z / $(320)) % $(128)) * $(256)) + ((r67_z % $(320)) - $(0))) / $(256)) % $(128)) * $(256)) + (((((r67_z / $(320)) % $(128)) * $(256)) + ((r67_z % $(320)) - $(0))) % $(256))), dtype=tl.bfloat16), tl.load($(candidate_vadd_7_ptr) + ((((((((r67_z / $(320)) % $(128)) * $(64)) + ((r67_z % $(320)) - $(256))) / $(64)) % $(128)) * $(64)) + (((((r67_z / $(320)) % $(128)) * $(64)) + ((r67_z % $(320)) - $(256))) % $(64))), dtype=tl.bfloat16))
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
      r69_right = tl.load($(candidate_vto_copy_20_ptr) + ((((((r69_lane[:, None] * $(1024)) + r69_col[None, :]) / $(128)) % $(8)) * $(65536)) + ((((r69_lane[:, None] * $(1024)) + r69_col[None, :]) % $(128)) * $(256)) + ((((r69_lane[:, None] * $(1024)) + r69_col[None, :]) / $(1024)) % $(256))), dtype=tl.bfloat16)
      r69_value = tl.dot(r69_left, r69_right)
      r69_value = r69_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_2_ptr) + ((r69_group * $(131072)) + (r69_row[:, None] * $(1024)) + r69_col[None, :]), r69_value)
    }
    r70_row = tl.arange($(128))
    r70_col = tl.arange($(1024))
    r70_lane = tl.arange($(256))
    for r70_group in range($(1)) {
      r70_left = tl.load($(candidate_vindex_2_ptr) + ((((((r70_row[:, None] * $(256)) + r70_lane[None, :]) / $(256)) % $(128)) * $(320)) + (((r70_row[:, None] * $(256)) + r70_lane[None, :]) % $(256))), dtype=tl.bfloat16)
      r70_right = tl.load($(candidate_vto_copy_20_ptr) + ((((((r70_lane[:, None] * $(1024)) + r70_col[None, :]) / $(128)) % $(8)) * $(65536)) + ((((r70_lane[:, None] * $(1024)) + r70_col[None, :]) % $(128)) * $(256)) + ((((r70_lane[:, None] * $(1024)) + r70_col[None, :]) / $(1024)) % $(256)) + $(32768)), dtype=tl.bfloat16)
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
    tl.store($(candidate_vto_copy_30_ptr) + r73_z, r73_value)
    r74_z = tl.arange($(18432))
    r74_value = tl.load($(candidate_vcat_7_ptr) + ((((r74_z / $(192)) % $(12)) * $(1536)) + (((r74_z / $(2304)) % $(8)) * $(192)) + (r74_z % $(192))), dtype=tl.bfloat16)
    r74_value = r74_value.to(tl.float32)
    r74_value = tl.broadcast_to(r74_value, [$(18432)])
    tl.store($(candidate_vto_copy_31_ptr) + r74_z, r74_value)
    r75_row = tl.arange($(12))
    r75_col = tl.arange($(12))
    r75_lane = tl.arange($(192))
    for r75_group in range($(8)) {
      r75_left = tl.load($(candidate_vto_copy_30_ptr) + (((((((r75_group % $(8)) * $(2304)) + (r75_row[:, None] * $(192)) + r75_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r75_group % $(8)) * $(2304)) + (r75_row[:, None] * $(192)) + r75_lane[None, :]) / $(192)) % $(12)) * $(192)) + ((((r75_group % $(8)) * $(2304)) + (r75_row[:, None] * $(192)) + r75_lane[None, :]) % $(192))), dtype=tl.float32)
      r75_right = tl.load($(candidate_vto_copy_31_ptr) + (((((((r75_group % $(8)) * $(2304)) + (r75_lane[:, None] * $(12)) + r75_col[None, :]) / $(2304)) % $(8)) * $(2304)) + (((((r75_group % $(8)) * $(2304)) + (r75_lane[:, None] * $(12)) + r75_col[None, :]) % $(12)) * $(192)) + (((((r75_group % $(8)) * $(2304)) + (r75_lane[:, None] * $(12)) + r75_col[None, :]) / $(12)) % $(192))), dtype=tl.float32)
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
    tl.store($(candidate_vto_copy_32_ptr) + r82_z, r82_value)
    r83_z = tl.arange($(6144))
    r83_value = tl.load($(candidate_vcat_6_ptr) + ((((r83_z / $(192)) % $(4)) * $(1536)) + (((r83_z / $(768)) % $(8)) * $(192)) + (r83_z % $(192)) + $(18432)), dtype=tl.bfloat16)
    r83_value = r83_value.to(tl.float32)
    r83_value = tl.broadcast_to(r83_value, [$(6144)])
    tl.store($(candidate_vto_copy_33_ptr) + r83_z, r83_value)
    r84_z = tl.arange($(6144))
    r84_value = tl.load($(candidate_vcat_7_ptr) + ((((r84_z / $(192)) % $(4)) * $(1536)) + (((r84_z / $(768)) % $(8)) * $(192)) + (r84_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r84_value = r84_value.to(tl.float32)
    r84_value = tl.broadcast_to(r84_value, [$(6144)])
    tl.store($(candidate_vto_copy_34_ptr) + r84_z, r84_value)
    r85_row = tl.arange($(4))
    r85_col = tl.arange($(4))
    r85_lane = tl.arange($(192))
    for r85_group in range($(8)) {
      r85_left = tl.load($(candidate_vto_copy_33_ptr) + (((((((r85_group % $(8)) * $(768)) + (r85_row[:, None] * $(192)) + r85_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r85_group % $(8)) * $(768)) + (r85_row[:, None] * $(192)) + r85_lane[None, :]) / $(192)) % $(4)) * $(192)) + ((((r85_group % $(8)) * $(768)) + (r85_row[:, None] * $(192)) + r85_lane[None, :]) % $(192))), dtype=tl.float32)
      r85_right = tl.load($(candidate_vto_copy_34_ptr) + (((((((r85_group % $(8)) * $(768)) + (r85_lane[:, None] * $(4)) + r85_col[None, :]) / $(768)) % $(8)) * $(768)) + (((((r85_group % $(8)) * $(768)) + (r85_lane[:, None] * $(4)) + r85_col[None, :]) % $(4)) * $(192)) + (((((r85_group % $(8)) * $(768)) + (r85_lane[:, None] * $(4)) + r85_col[None, :]) / $(4)) % $(192))), dtype=tl.float32)
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
    tl.store($(candidate_vto_copy_35_ptr) + r92_z, r92_value)
    r93_z = tl.arange($(24576))
    r93_value = tl.load($(candidate_vcat_6_ptr) + ((((r93_z / $(192)) % $(16)) * $(1536)) + (((r93_z / $(3072)) % $(8)) * $(192)) + (r93_z % $(192)) + $(24576)), dtype=tl.bfloat16)
    r93_value = r93_value.to(tl.float32)
    r93_value = tl.broadcast_to(r93_value, [$(24576)])
    tl.store($(candidate_vto_copy_36_ptr) + r93_z, r93_value)
    r94_z = tl.arange($(24576))
    r94_value = tl.load($(candidate_vcat_7_ptr) + ((((r94_z / $(192)) % $(16)) * $(1536)) + (((r94_z / $(3072)) % $(8)) * $(192)) + (r94_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r94_value = r94_value.to(tl.float32)
    r94_value = tl.broadcast_to(r94_value, [$(24576)])
    tl.store($(candidate_vto_copy_37_ptr) + r94_z, r94_value)
    r95_row = tl.arange($(16))
    r95_col = tl.arange($(16))
    r95_lane = tl.arange($(192))
    for r95_group in range($(8)) {
      r95_left = tl.load($(candidate_vto_copy_36_ptr) + (((((((r95_group % $(8)) * $(3072)) + (r95_row[:, None] * $(192)) + r95_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r95_group % $(8)) * $(3072)) + (r95_row[:, None] * $(192)) + r95_lane[None, :]) / $(192)) % $(16)) * $(192)) + ((((r95_group % $(8)) * $(3072)) + (r95_row[:, None] * $(192)) + r95_lane[None, :]) % $(192))), dtype=tl.float32)
      r95_right = tl.load($(candidate_vto_copy_37_ptr) + (((((((r95_group % $(8)) * $(3072)) + (r95_lane[:, None] * $(16)) + r95_col[None, :]) / $(3072)) % $(8)) * $(3072)) + (((((r95_group % $(8)) * $(3072)) + (r95_lane[:, None] * $(16)) + r95_col[None, :]) % $(16)) * $(192)) + (((((r95_group % $(8)) * $(3072)) + (r95_lane[:, None] * $(16)) + r95_col[None, :]) / $(16)) % $(192))), dtype=tl.float32)
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
    tl.store($(candidate_vto_copy_38_ptr) + r102_z, r102_value)
    r103_row = tl.arange($(128))
    r103_col = tl.arange($(1024))
    r103_lane = tl.arange($(256))
    for r103_group in range($(1)) {
      r103_left = tl.load($(candidate_vindex_2_ptr) + ((((((r103_row[:, None] * $(256)) + r103_lane[None, :]) / $(256)) % $(128)) * $(320)) + (((r103_row[:, None] * $(256)) + r103_lane[None, :]) % $(256))), dtype=tl.bfloat16)
      r103_right = tl.load($(candidate_vto_copy_20_ptr) + ((((((r103_lane[:, None] * $(1024)) + r103_col[None, :]) / $(128)) % $(8)) * $(65536)) + ((((r103_lane[:, None] * $(1024)) + r103_col[None, :]) % $(128)) * $(256)) + ((((r103_lane[:, None] * $(1024)) + r103_col[None, :]) / $(1024)) % $(256))), dtype=tl.bfloat16)
      r103_value = tl.dot(r103_left, r103_right)
      r103_value = r103_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_10_ptr) + ((r103_group * $(131072)) + (r103_row[:, None] * $(1024)) + r103_col[None, :]), r103_value)
    }
    r104_row = tl.arange($(128))
    r104_col = tl.arange($(1024))
    r104_lane = tl.arange($(256))
    for r104_group in range($(1)) {
      r104_left = tl.load($(candidate_vindex_2_ptr) + ((((((r104_row[:, None] * $(256)) + r104_lane[None, :]) / $(256)) % $(128)) * $(320)) + (((r104_row[:, None] * $(256)) + r104_lane[None, :]) % $(256))), dtype=tl.bfloat16)
      r104_right = tl.load($(candidate_vto_copy_20_ptr) + ((((((r104_lane[:, None] * $(1024)) + r104_col[None, :]) / $(128)) % $(8)) * $(65536)) + ((((r104_lane[:, None] * $(1024)) + r104_col[None, :]) % $(128)) * $(256)) + ((((r104_lane[:, None] * $(1024)) + r104_col[None, :]) / $(1024)) % $(256)) + $(32768)), dtype=tl.bfloat16)
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
    tl.store($(candidate_vto_copy_39_ptr) + r107_z, r107_value)
    r108_z = tl.arange($(73728))
    r108_value = tl.load($(candidate_vcat_10_ptr) + ((((r108_z / $(192)) % $(48)) * $(1536)) + (((r108_z / $(9216)) % $(8)) * $(192)) + (r108_z % $(192))), dtype=tl.bfloat16)
    r108_value = r108_value.to(tl.float32)
    r108_value = tl.broadcast_to(r108_value, [$(73728)])
    tl.store($(candidate_vto_copy_40_ptr) + r108_z, r108_value)
    r109_row = tl.arange($(12))
    r109_col = tl.arange($(48))
    r109_lane = tl.arange($(192))
    for r109_group in range($(8)) {
      r109_left = tl.load($(candidate_vto_copy_39_ptr) + (((((((r109_group % $(8)) * $(2304)) + (r109_row[:, None] * $(192)) + r109_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r109_group % $(8)) * $(2304)) + (r109_row[:, None] * $(192)) + r109_lane[None, :]) / $(192)) % $(12)) * $(192)) + ((((r109_group % $(8)) * $(2304)) + (r109_row[:, None] * $(192)) + r109_lane[None, :]) % $(192))), dtype=tl.float32)
      r109_right = tl.load($(candidate_vto_copy_40_ptr) + (((((((r109_group % $(8)) * $(9216)) + (r109_lane[:, None] * $(48)) + r109_col[None, :]) / $(9216)) % $(8)) * $(9216)) + (((((r109_group % $(8)) * $(9216)) + (r109_lane[:, None] * $(48)) + r109_col[None, :]) % $(48)) * $(192)) + (((((r109_group % $(8)) * $(9216)) + (r109_lane[:, None] * $(48)) + r109_col[None, :]) / $(48)) % $(192))), dtype=tl.float32)
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
    tl.store($(candidate_vto_copy_41_ptr) + r116_z, r116_value)
    r117_z = tl.arange($(6144))
    r117_value = tl.load($(candidate_vcat_9_ptr) + ((((r117_z / $(192)) % $(4)) * $(1536)) + (((r117_z / $(768)) % $(8)) * $(192)) + (r117_z % $(192)) + $(18432)), dtype=tl.bfloat16)
    r117_value = r117_value.to(tl.float32)
    r117_value = tl.broadcast_to(r117_value, [$(6144)])
    tl.store($(candidate_vto_copy_42_ptr) + r117_z, r117_value)
    r118_z = tl.arange($(24576))
    r118_value = tl.load($(candidate_vcat_10_ptr) + ((((r118_z / $(192)) % $(16)) * $(1536)) + (((r118_z / $(3072)) % $(8)) * $(192)) + (r118_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r118_value = r118_value.to(tl.float32)
    r118_value = tl.broadcast_to(r118_value, [$(24576)])
    tl.store($(candidate_vto_copy_43_ptr) + r118_z, r118_value)
    r119_row = tl.arange($(4))
    r119_col = tl.arange($(16))
    r119_lane = tl.arange($(192))
    for r119_group in range($(8)) {
      r119_left = tl.load($(candidate_vto_copy_42_ptr) + (((((((r119_group % $(8)) * $(768)) + (r119_row[:, None] * $(192)) + r119_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r119_group % $(8)) * $(768)) + (r119_row[:, None] * $(192)) + r119_lane[None, :]) / $(192)) % $(4)) * $(192)) + ((((r119_group % $(8)) * $(768)) + (r119_row[:, None] * $(192)) + r119_lane[None, :]) % $(192))), dtype=tl.float32)
      r119_right = tl.load($(candidate_vto_copy_43_ptr) + (((((((r119_group % $(8)) * $(3072)) + (r119_lane[:, None] * $(16)) + r119_col[None, :]) / $(3072)) % $(8)) * $(3072)) + (((((r119_group % $(8)) * $(3072)) + (r119_lane[:, None] * $(16)) + r119_col[None, :]) % $(16)) * $(192)) + (((((r119_group % $(8)) * $(3072)) + (r119_lane[:, None] * $(16)) + r119_col[None, :]) / $(16)) % $(192))), dtype=tl.float32)
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
    tl.store($(candidate_vto_copy_44_ptr) + r126_z, r126_value)
    r127_z = tl.arange($(24576))
    r127_value = tl.load($(candidate_vcat_9_ptr) + ((((r127_z / $(192)) % $(16)) * $(1536)) + (((r127_z / $(3072)) % $(8)) * $(192)) + (r127_z % $(192)) + $(24576)), dtype=tl.bfloat16)
    r127_value = r127_value.to(tl.float32)
    r127_value = tl.broadcast_to(r127_value, [$(24576)])
    tl.store($(candidate_vto_copy_45_ptr) + r127_z, r127_value)
    r128_z = tl.arange($(98304))
    r128_value = tl.load($(candidate_vcat_10_ptr) + ((((r128_z / $(192)) % $(64)) * $(1536)) + (((r128_z / $(12288)) % $(8)) * $(192)) + (r128_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r128_value = r128_value.to(tl.float32)
    r128_value = tl.broadcast_to(r128_value, [$(98304)])
    tl.store($(candidate_vto_copy_46_ptr) + r128_z, r128_value)
    r129_row = tl.arange($(16))
    r129_col = tl.arange($(64))
    r129_lane = tl.arange($(192))
    for r129_group in range($(8)) {
      r129_left = tl.load($(candidate_vto_copy_45_ptr) + (((((((r129_group % $(8)) * $(3072)) + (r129_row[:, None] * $(192)) + r129_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r129_group % $(8)) * $(3072)) + (r129_row[:, None] * $(192)) + r129_lane[None, :]) / $(192)) % $(16)) * $(192)) + ((((r129_group % $(8)) * $(3072)) + (r129_row[:, None] * $(192)) + r129_lane[None, :]) % $(192))), dtype=tl.float32)
      r129_right = tl.load($(candidate_vto_copy_46_ptr) + (((((((r129_group % $(8)) * $(12288)) + (r129_lane[:, None] * $(64)) + r129_col[None, :]) / $(12288)) % $(8)) * $(12288)) + (((((r129_group % $(8)) * $(12288)) + (r129_lane[:, None] * $(64)) + r129_col[None, :]) % $(64)) * $(192)) + (((((r129_group % $(8)) * $(12288)) + (r129_lane[:, None] * $(64)) + r129_col[None, :]) / $(64)) % $(192))), dtype=tl.float32)
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
    tl.store($(candidate_vto_copy_47_ptr) + r136_z, r136_value)
    r137_row = tl.arange($(128))
    r137_col = tl.arange($(1024))
    r137_lane = tl.arange($(256))
    for r137_group in range($(1)) {
      r137_left = tl.load($(candidate_vindex_2_ptr) + ((((((r137_row[:, None] * $(256)) + r137_lane[None, :]) / $(256)) % $(128)) * $(320)) + (((r137_row[:, None] * $(256)) + r137_lane[None, :]) % $(256))), dtype=tl.bfloat16)
      r137_right = tl.load($(candidate_vto_copy_20_ptr) + ((((((r137_lane[:, None] * $(1024)) + r137_col[None, :]) / $(128)) % $(8)) * $(65536)) + ((((r137_lane[:, None] * $(1024)) + r137_col[None, :]) % $(128)) * $(256)) + ((((r137_lane[:, None] * $(1024)) + r137_col[None, :]) / $(1024)) % $(256))), dtype=tl.bfloat16)
      r137_value = tl.dot(r137_left, r137_right)
      r137_value = r137_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_18_ptr) + ((r137_group * $(131072)) + (r137_row[:, None] * $(1024)) + r137_col[None, :]), r137_value)
    }
    r138_row = tl.arange($(128))
    r138_col = tl.arange($(1024))
    r138_lane = tl.arange($(256))
    for r138_group in range($(1)) {
      r138_left = tl.load($(candidate_vindex_2_ptr) + ((((((r138_row[:, None] * $(256)) + r138_lane[None, :]) / $(256)) % $(128)) * $(320)) + (((r138_row[:, None] * $(256)) + r138_lane[None, :]) % $(256))), dtype=tl.bfloat16)
      r138_right = tl.load($(candidate_vto_copy_20_ptr) + ((((((r138_lane[:, None] * $(1024)) + r138_col[None, :]) / $(128)) % $(8)) * $(65536)) + ((((r138_lane[:, None] * $(1024)) + r138_col[None, :]) % $(128)) * $(256)) + ((((r138_lane[:, None] * $(1024)) + r138_col[None, :]) / $(1024)) % $(256)) + $(32768)), dtype=tl.bfloat16)
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
    tl.store($(candidate_vto_copy_48_ptr) + r141_z, r141_value)
    r142_z = tl.arange($(36864))
    r142_value = tl.load($(candidate_vcat_13_ptr) + ((((r142_z / $(192)) % $(24)) * $(1536)) + (((r142_z / $(4608)) % $(8)) * $(192)) + (r142_z % $(192))), dtype=tl.bfloat16)
    r142_value = r142_value.to(tl.float32)
    r142_value = tl.broadcast_to(r142_value, [$(36864)])
    tl.store($(candidate_vto_copy_49_ptr) + r142_z, r142_value)
    r143_row = tl.arange($(12))
    r143_col = tl.arange($(24))
    r143_lane = tl.arange($(192))
    for r143_group in range($(8)) {
      r143_left = tl.load($(candidate_vto_copy_48_ptr) + (((((((r143_group % $(8)) * $(2304)) + (r143_row[:, None] * $(192)) + r143_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r143_group % $(8)) * $(2304)) + (r143_row[:, None] * $(192)) + r143_lane[None, :]) / $(192)) % $(12)) * $(192)) + ((((r143_group % $(8)) * $(2304)) + (r143_row[:, None] * $(192)) + r143_lane[None, :]) % $(192))), dtype=tl.float32)
      r143_right = tl.load($(candidate_vto_copy_49_ptr) + (((((((r143_group % $(8)) * $(4608)) + (r143_lane[:, None] * $(24)) + r143_col[None, :]) / $(4608)) % $(8)) * $(4608)) + (((((r143_group % $(8)) * $(4608)) + (r143_lane[:, None] * $(24)) + r143_col[None, :]) % $(24)) * $(192)) + (((((r143_group % $(8)) * $(4608)) + (r143_lane[:, None] * $(24)) + r143_col[None, :]) / $(24)) % $(192))), dtype=tl.float32)
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
    tl.store($(candidate_vto_copy_50_ptr) + r150_z, r150_value)
    r151_z = tl.arange($(6144))
    r151_value = tl.load($(candidate_vcat_12_ptr) + ((((r151_z / $(192)) % $(4)) * $(1536)) + (((r151_z / $(768)) % $(8)) * $(192)) + (r151_z % $(192)) + $(18432)), dtype=tl.bfloat16)
    r151_value = r151_value.to(tl.float32)
    r151_value = tl.broadcast_to(r151_value, [$(6144)])
    tl.store($(candidate_vto_copy_51_ptr) + r151_z, r151_value)
    r152_z = tl.arange($(12288))
    r152_value = tl.load($(candidate_vcat_13_ptr) + ((((r152_z / $(192)) % $(8)) * $(1536)) + (((r152_z / $(1536)) % $(8)) * $(192)) + (r152_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r152_value = r152_value.to(tl.float32)
    r152_value = tl.broadcast_to(r152_value, [$(12288)])
    tl.store($(candidate_vto_copy_52_ptr) + r152_z, r152_value)
    r153_row = tl.arange($(4))
    r153_col = tl.arange($(8))
    r153_lane = tl.arange($(192))
    for r153_group in range($(8)) {
      r153_left = tl.load($(candidate_vto_copy_51_ptr) + (((((((r153_group % $(8)) * $(768)) + (r153_row[:, None] * $(192)) + r153_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r153_group % $(8)) * $(768)) + (r153_row[:, None] * $(192)) + r153_lane[None, :]) / $(192)) % $(4)) * $(192)) + ((((r153_group % $(8)) * $(768)) + (r153_row[:, None] * $(192)) + r153_lane[None, :]) % $(192))), dtype=tl.float32)
      r153_right = tl.load($(candidate_vto_copy_52_ptr) + (((((((r153_group % $(8)) * $(1536)) + (r153_lane[:, None] * $(8)) + r153_col[None, :]) / $(1536)) % $(8)) * $(1536)) + (((((r153_group % $(8)) * $(1536)) + (r153_lane[:, None] * $(8)) + r153_col[None, :]) % $(8)) * $(192)) + (((((r153_group % $(8)) * $(1536)) + (r153_lane[:, None] * $(8)) + r153_col[None, :]) / $(8)) % $(192))), dtype=tl.float32)
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
    tl.store($(candidate_vto_copy_53_ptr) + r160_z, r160_value)
    r161_z = tl.arange($(24576))
    r161_value = tl.load($(candidate_vcat_12_ptr) + ((((r161_z / $(192)) % $(16)) * $(1536)) + (((r161_z / $(3072)) % $(8)) * $(192)) + (r161_z % $(192)) + $(24576)), dtype=tl.bfloat16)
    r161_value = r161_value.to(tl.float32)
    r161_value = tl.broadcast_to(r161_value, [$(24576)])
    tl.store($(candidate_vto_copy_54_ptr) + r161_z, r161_value)
    r162_z = tl.arange($(49152))
    r162_value = tl.load($(candidate_vcat_13_ptr) + ((((r162_z / $(192)) % $(32)) * $(1536)) + (((r162_z / $(6144)) % $(8)) * $(192)) + (r162_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r162_value = r162_value.to(tl.float32)
    r162_value = tl.broadcast_to(r162_value, [$(49152)])
    tl.store($(candidate_vto_copy_55_ptr) + r162_z, r162_value)
    r163_row = tl.arange($(16))
    r163_col = tl.arange($(32))
    r163_lane = tl.arange($(192))
    for r163_group in range($(8)) {
      r163_left = tl.load($(candidate_vto_copy_54_ptr) + (((((((r163_group % $(8)) * $(3072)) + (r163_row[:, None] * $(192)) + r163_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r163_group % $(8)) * $(3072)) + (r163_row[:, None] * $(192)) + r163_lane[None, :]) / $(192)) % $(16)) * $(192)) + ((((r163_group % $(8)) * $(3072)) + (r163_row[:, None] * $(192)) + r163_lane[None, :]) % $(192))), dtype=tl.float32)
      r163_right = tl.load($(candidate_vto_copy_55_ptr) + (((((((r163_group % $(8)) * $(6144)) + (r163_lane[:, None] * $(32)) + r163_col[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r163_group % $(8)) * $(6144)) + (r163_lane[:, None] * $(32)) + r163_col[None, :]) % $(32)) * $(192)) + (((((r163_group % $(8)) * $(6144)) + (r163_lane[:, None] * $(32)) + r163_col[None, :]) / $(32)) % $(192))), dtype=tl.float32)
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
    tl.store($(candidate_vto_copy_56_ptr) + r170_z, r170_value)
    r171_row = tl.arange($(128))
    r171_col = tl.arange($(1024))
    r171_lane = tl.arange($(256))
    for r171_group in range($(1)) {
      r171_left = tl.load($(candidate_vindex_2_ptr) + ((((((r171_row[:, None] * $(256)) + r171_lane[None, :]) / $(256)) % $(128)) * $(320)) + (((r171_row[:, None] * $(256)) + r171_lane[None, :]) % $(256))), dtype=tl.bfloat16)
      r171_right = tl.load($(candidate_vto_copy_20_ptr) + ((((((r171_lane[:, None] * $(1024)) + r171_col[None, :]) / $(128)) % $(8)) * $(65536)) + ((((r171_lane[:, None] * $(1024)) + r171_col[None, :]) % $(128)) * $(256)) + ((((r171_lane[:, None] * $(1024)) + r171_col[None, :]) / $(1024)) % $(256))), dtype=tl.bfloat16)
      r171_value = tl.dot(r171_left, r171_right)
      r171_value = r171_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_26_ptr) + ((r171_group * $(131072)) + (r171_row[:, None] * $(1024)) + r171_col[None, :]), r171_value)
    }
    r172_row = tl.arange($(128))
    r172_col = tl.arange($(1024))
    r172_lane = tl.arange($(256))
    for r172_group in range($(1)) {
      r172_left = tl.load($(candidate_vindex_2_ptr) + ((((((r172_row[:, None] * $(256)) + r172_lane[None, :]) / $(256)) % $(128)) * $(320)) + (((r172_row[:, None] * $(256)) + r172_lane[None, :]) % $(256))), dtype=tl.bfloat16)
      r172_right = tl.load($(candidate_vto_copy_20_ptr) + ((((((r172_lane[:, None] * $(1024)) + r172_col[None, :]) / $(128)) % $(8)) * $(65536)) + ((((r172_lane[:, None] * $(1024)) + r172_col[None, :]) % $(128)) * $(256)) + ((((r172_lane[:, None] * $(1024)) + r172_col[None, :]) / $(1024)) % $(256)) + $(32768)), dtype=tl.bfloat16)
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
    tl.store($(candidate_vto_copy_57_ptr) + r175_z, r175_value)
    r176_z = tl.arange($(55296))
    r176_value = tl.load($(candidate_vcat_16_ptr) + ((((r176_z / $(192)) % $(36)) * $(1536)) + (((r176_z / $(6912)) % $(8)) * $(192)) + (r176_z % $(192))), dtype=tl.bfloat16)
    r176_value = r176_value.to(tl.float32)
    r176_value = tl.broadcast_to(r176_value, [$(55296)])
    tl.store($(candidate_vto_copy_58_ptr) + r176_z, r176_value)
    r177_row = tl.arange($(12))
    r177_col = tl.arange($(36))
    r177_lane = tl.arange($(192))
    for r177_group in range($(8)) {
      r177_left = tl.load($(candidate_vto_copy_57_ptr) + (((((((r177_group % $(8)) * $(2304)) + (r177_row[:, None] * $(192)) + r177_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r177_group % $(8)) * $(2304)) + (r177_row[:, None] * $(192)) + r177_lane[None, :]) / $(192)) % $(12)) * $(192)) + ((((r177_group % $(8)) * $(2304)) + (r177_row[:, None] * $(192)) + r177_lane[None, :]) % $(192))), dtype=tl.float32)
      r177_right = tl.load($(candidate_vto_copy_58_ptr) + (((((((r177_group % $(8)) * $(6912)) + (r177_lane[:, None] * $(36)) + r177_col[None, :]) / $(6912)) % $(8)) * $(6912)) + (((((r177_group % $(8)) * $(6912)) + (r177_lane[:, None] * $(36)) + r177_col[None, :]) % $(36)) * $(192)) + (((((r177_group % $(8)) * $(6912)) + (r177_lane[:, None] * $(36)) + r177_col[None, :]) / $(36)) % $(192))), dtype=tl.float32)
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
    tl.store($(candidate_vto_copy_59_ptr) + r184_z, r184_value)
    r185_z = tl.arange($(6144))
    r185_value = tl.load($(candidate_vcat_15_ptr) + ((((r185_z / $(192)) % $(4)) * $(1536)) + (((r185_z / $(768)) % $(8)) * $(192)) + (r185_z % $(192)) + $(18432)), dtype=tl.bfloat16)
    r185_value = r185_value.to(tl.float32)
    r185_value = tl.broadcast_to(r185_value, [$(6144)])
    tl.store($(candidate_vto_copy_60_ptr) + r185_z, r185_value)
    r186_z = tl.arange($(18432))
    r186_value = tl.load($(candidate_vcat_16_ptr) + ((((r186_z / $(192)) % $(12)) * $(1536)) + (((r186_z / $(2304)) % $(8)) * $(192)) + (r186_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r186_value = r186_value.to(tl.float32)
    r186_value = tl.broadcast_to(r186_value, [$(18432)])
    tl.store($(candidate_vto_copy_61_ptr) + r186_z, r186_value)
    r187_row = tl.arange($(4))
    r187_col = tl.arange($(12))
    r187_lane = tl.arange($(192))
    for r187_group in range($(8)) {
      r187_left = tl.load($(candidate_vto_copy_60_ptr) + (((((((r187_group % $(8)) * $(768)) + (r187_row[:, None] * $(192)) + r187_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r187_group % $(8)) * $(768)) + (r187_row[:, None] * $(192)) + r187_lane[None, :]) / $(192)) % $(4)) * $(192)) + ((((r187_group % $(8)) * $(768)) + (r187_row[:, None] * $(192)) + r187_lane[None, :]) % $(192))), dtype=tl.float32)
      r187_right = tl.load($(candidate_vto_copy_61_ptr) + (((((((r187_group % $(8)) * $(2304)) + (r187_lane[:, None] * $(12)) + r187_col[None, :]) / $(2304)) % $(8)) * $(2304)) + (((((r187_group % $(8)) * $(2304)) + (r187_lane[:, None] * $(12)) + r187_col[None, :]) % $(12)) * $(192)) + (((((r187_group % $(8)) * $(2304)) + (r187_lane[:, None] * $(12)) + r187_col[None, :]) / $(12)) % $(192))), dtype=tl.float32)
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
    tl.store($(candidate_vto_copy_62_ptr) + r194_z, r194_value)
    r195_z = tl.arange($(24576))
    r195_value = tl.load($(candidate_vcat_15_ptr) + ((((r195_z / $(192)) % $(16)) * $(1536)) + (((r195_z / $(3072)) % $(8)) * $(192)) + (r195_z % $(192)) + $(24576)), dtype=tl.bfloat16)
    r195_value = r195_value.to(tl.float32)
    r195_value = tl.broadcast_to(r195_value, [$(24576)])
    tl.store($(candidate_vto_copy_63_ptr) + r195_z, r195_value)
    r196_z = tl.arange($(73728))
    r196_value = tl.load($(candidate_vcat_16_ptr) + ((((r196_z / $(192)) % $(48)) * $(1536)) + (((r196_z / $(9216)) % $(8)) * $(192)) + (r196_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r196_value = r196_value.to(tl.float32)
    r196_value = tl.broadcast_to(r196_value, [$(73728)])
    tl.store($(candidate_vto_copy_64_ptr) + r196_z, r196_value)
    r197_row = tl.arange($(16))
    r197_col = tl.arange($(48))
    r197_lane = tl.arange($(192))
    for r197_group in range($(8)) {
      r197_left = tl.load($(candidate_vto_copy_63_ptr) + (((((((r197_group % $(8)) * $(3072)) + (r197_row[:, None] * $(192)) + r197_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r197_group % $(8)) * $(3072)) + (r197_row[:, None] * $(192)) + r197_lane[None, :]) / $(192)) % $(16)) * $(192)) + ((((r197_group % $(8)) * $(3072)) + (r197_row[:, None] * $(192)) + r197_lane[None, :]) % $(192))), dtype=tl.float32)
      r197_right = tl.load($(candidate_vto_copy_64_ptr) + (((((((r197_group % $(8)) * $(9216)) + (r197_lane[:, None] * $(48)) + r197_col[None, :]) / $(9216)) % $(8)) * $(9216)) + (((((r197_group % $(8)) * $(9216)) + (r197_lane[:, None] * $(48)) + r197_col[None, :]) % $(48)) * $(192)) + (((((r197_group % $(8)) * $(9216)) + (r197_lane[:, None] * $(48)) + r197_col[None, :]) / $(48)) % $(192))), dtype=tl.float32)
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
    tl.store($(candidate_vto_copy_65_ptr) + r204_z, r204_value)
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
      r207_left = tl.load($(candidate_vto_copy_65_ptr) + (((((((r207_group % $(8)) * $(768)) + (r207_lane[None, :] * $(48)) + r207_row[:, None]) / $(768)) % $(8)) * $(768)) + ((((((r207_group % $(8)) * $(768)) + (r207_lane[None, :] * $(48)) + r207_row[:, None]) / $(48)) % $(16)) * $(48)) + ((((r207_group % $(8)) * $(768)) + (r207_lane[None, :] * $(48)) + r207_row[:, None]) % $(48))), dtype=tl.bfloat16)
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
    tl.store($(candidate_vto_copy_66_ptr) + r209_z, r209_value)
    r210_z = tl.arange($(6144))
    r210_lane = tl.arange($(48))
    r210_row_values = tl.load($(candidate_vsoftmax_11_ptr) + ((((r210_z[:, None] / $(768)) % $(8)) * $(768)) + (((r210_z[:, None] / $(48)) % $(16)) * $(48)) + (r210_lane[None, :] % $(48))), dtype=tl.float32)
    r210_row_grad = tl.load($(candidate_vto_copy_66_ptr) + ((((r210_z[:, None] / $(768)) % $(8)) * $(768)) + (((r210_z[:, None] / $(48)) % $(16)) * $(48)) + (r210_lane[None, :] % $(48))), dtype=tl.float32)
    r210_weighted = r210_row_grad * r210_row_values
    r210_dot_grad = tl.sum(r210_weighted, axis=1)
    r210_value = (tl.load($(candidate_vto_copy_66_ptr) + ((((r210_z / $(768)) % $(8)) * $(768)) + (((r210_z / $(48)) % $(16)) * $(48)) + (r210_z % $(48))), dtype=tl.float32) - r210_dot_grad) * tl.load($(candidate_vsoftmax_11_ptr) + ((((r210_z / $(768)) % $(8)) * $(768)) + (((r210_z / $(48)) % $(16)) * $(48)) + (r210_z % $(48))), dtype=tl.float32)
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
      r213_left = tl.load($(candidate_vto_copy_63_ptr) + (((((((r213_group % $(8)) * $(3072)) + (r213_lane[None, :] * $(192)) + r213_row[:, None]) / $(3072)) % $(8)) * $(3072)) + ((((((r213_group % $(8)) * $(3072)) + (r213_lane[None, :] * $(192)) + r213_row[:, None]) / $(192)) % $(16)) * $(192)) + ((((r213_group % $(8)) * $(3072)) + (r213_lane[None, :] * $(192)) + r213_row[:, None]) % $(192))), dtype=tl.float32)
      r213_right = tl.load($(candidate_vmul_35_ptr) + (((((((r213_group % $(8)) * $(768)) + (r213_lane[:, None] * $(48)) + r213_col[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r213_group % $(8)) * $(768)) + (r213_lane[:, None] * $(48)) + r213_col[None, :]) / $(48)) % $(16)) * $(48)) + ((((r213_group % $(8)) * $(768)) + (r213_lane[:, None] * $(48)) + r213_col[None, :]) % $(48))), dtype=tl.float32)
      r213_value = tl.dot(r213_left, r213_right)
      r213_value = r213_value.to(tl.float32)
      tl.store($(candidate_vbmm_36_ptr) + ((r213_group * $(9216)) + (r213_row[:, None] * $(48)) + r213_col[None, :]), r213_value)
    }
    r214_z = tl.arange($(131072))
    r214_value = tl.where(((((r214_z / $(1024)) % $(128)) >= $(64)) & (((r214_z / $(1024)) % $(128)) < $(112)) & ((((r214_z / $(1024)) % $(128)) - $(64)) % $(1) == $(0))), tl.load($(candidate_vbmm_34_ptr) + ((((((((r214_z / $(128)) % $(8)) * $(6144)) + ((((r214_z / $(1024)) % $(128)) - $(64)) * $(128)) + (r214_z % $(128))) / $(6144)) % $(8)) * $(6144)) + ((((((r214_z / $(128)) % $(8)) * $(48)) + (((r214_z / $(1024)) % $(128)) - $(64))) % $(48)) * $(128)) + (r214_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r214_value = r214_value.to(tl.bfloat16)
    r214_value = tl.broadcast_to(r214_value, [$(131072)])
    tl.store($(candidate_vslice_backward_ptr) + r214_z, r214_value)
    r215_z = tl.arange($(73728))
    r215_value = tl.load($(candidate_vbmm_36_ptr) + ((((r215_z / $(9216)) % $(8)) * $(9216)) + ((r215_z % $(192)) * $(48)) + ((r215_z / $(192)) % $(48))), dtype=tl.float32)
    r215_value = r215_value.to(tl.bfloat16)
    r215_value = tl.broadcast_to(r215_value, [$(73728)])
    tl.store($(candidate_vto_copy_67_ptr) + r215_z, r215_value)
    r216_z = tl.arange($(196608))
    r216_value = tl.where(((((r216_z / $(1536)) % $(128)) >= $(64)) & (((r216_z / $(1536)) % $(128)) < $(112)) & ((((r216_z / $(1536)) % $(128)) - $(64)) % $(1) == $(0))), tl.load($(candidate_vto_copy_67_ptr) + ((((r216_z / $(192)) % $(8)) * $(9216)) + (((((r216_z / $(1536)) % $(128)) - $(64)) % $(48)) * $(192)) + (r216_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r216_value = r216_value.to(tl.bfloat16)
    r216_value = tl.broadcast_to(r216_value, [$(196608)])
    tl.store($(candidate_vslice_backward_1_ptr) + r216_z, r216_value)
    r217_row = tl.arange($(12))
    r217_col = tl.arange($(128))
    r217_lane = tl.arange($(4))
    for r217_group in range($(8)) {
      r217_left = tl.load($(candidate_vto_copy_62_ptr) + (((((((r217_group % $(8)) * $(48)) + (r217_lane[None, :] * $(12)) + r217_row[:, None]) / $(48)) % $(8)) * $(48)) + ((((((r217_group % $(8)) * $(48)) + (r217_lane[None, :] * $(12)) + r217_row[:, None]) / $(12)) % $(4)) * $(12)) + ((((r217_group % $(8)) * $(48)) + (r217_lane[None, :] * $(12)) + r217_row[:, None]) % $(12))), dtype=tl.bfloat16)
      r217_right = tl.load($(candidate_vmm_4_ptr) + (((((((r217_group % $(8)) * $(512)) + (r217_lane[:, None] * $(128)) + r217_col[None, :]) / $(128)) % $(4)) * $(1024)) + ((((((r217_group % $(8)) * $(512)) + (r217_lane[:, None] * $(128)) + r217_col[None, :]) / $(512)) % $(8)) * $(128)) + ((((r217_group % $(8)) * $(512)) + (r217_lane[:, None] * $(128)) + r217_col[None, :]) % $(128)) + $(110592)), dtype=tl.bfloat16)
      r217_value = tl.dot(r217_left, r217_right)
      r217_value = r217_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_37_ptr) + ((r217_group * $(1536)) + (r217_row[:, None] * $(128)) + r217_col[None, :]), r217_value)
    }
    r218_row = tl.arange($(4))
    r218_col = tl.arange($(12))
    r218_lane = tl.arange($(128))
    for r218_group in range($(8)) {
      r218_left = tl.load($(candidate_vmm_4_ptr) + (((((((r218_group % $(8)) * $(512)) + (r218_row[:, None] * $(128)) + r218_lane[None, :]) / $(128)) % $(4)) * $(1024)) + ((((((r218_group % $(8)) * $(512)) + (r218_row[:, None] * $(128)) + r218_lane[None, :]) / $(512)) % $(8)) * $(128)) + ((((r218_group % $(8)) * $(512)) + (r218_row[:, None] * $(128)) + r218_lane[None, :]) % $(128)) + $(110592)), dtype=tl.bfloat16)
      r218_right = tl.load($(candidate_vbmm_27_ptr) + (((((((r218_group % $(8)) * $(1536)) + (r218_col[None, :] * $(128)) + r218_lane[:, None]) / $(128)) % $(12)) * $(1024)) + ((((((r218_group % $(8)) * $(1536)) + (r218_col[None, :] * $(128)) + r218_lane[:, None]) / $(1536)) % $(8)) * $(128)) + ((((r218_group % $(8)) * $(1536)) + (r218_col[None, :] * $(128)) + r218_lane[:, None]) % $(128)) + $(49152)), dtype=tl.bfloat16)
      r218_value = tl.dot(r218_left, r218_right)
      r218_value = r218_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_38_ptr) + ((r218_group * $(48)) + (r218_row[:, None] * $(12)) + r218_col[None, :]), r218_value)
    }
    r219_z = tl.arange($(384))
    r219_value = tl.load($(candidate_vbmm_38_ptr) + ((((r219_z / $(48)) % $(8)) * $(48)) + (((r219_z / $(12)) % $(4)) * $(12)) + (r219_z % $(12))), dtype=tl.bfloat16)
    r219_value = r219_value.to(tl.float32)
    r219_value = tl.broadcast_to(r219_value, [$(384)])
    tl.store($(candidate_vto_copy_68_ptr) + r219_z, r219_value)
    r220_z = tl.arange($(384))
    r220_lane = tl.arange($(12))
    r220_row_values = tl.load($(candidate_vsoftmax_10_ptr) + ((((r220_z[:, None] / $(48)) % $(8)) * $(48)) + (((r220_z[:, None] / $(12)) % $(4)) * $(12)) + (r220_lane[None, :] % $(12))), dtype=tl.float32)
    r220_row_grad = tl.load($(candidate_vto_copy_68_ptr) + ((((r220_z[:, None] / $(48)) % $(8)) * $(48)) + (((r220_z[:, None] / $(12)) % $(4)) * $(12)) + (r220_lane[None, :] % $(12))), dtype=tl.float32)
    r220_weighted = r220_row_grad * r220_row_values
    r220_dot_grad = tl.sum(r220_weighted, axis=1)
    r220_value = (tl.load($(candidate_vto_copy_68_ptr) + ((((r220_z / $(48)) % $(8)) * $(48)) + (((r220_z / $(12)) % $(4)) * $(12)) + (r220_z % $(12))), dtype=tl.float32) - r220_dot_grad) * tl.load($(candidate_vsoftmax_10_ptr) + ((((r220_z / $(48)) % $(8)) * $(48)) + (((r220_z / $(12)) % $(4)) * $(12)) + (r220_z % $(12))), dtype=tl.float32)
    r220_value = r220_value.to(tl.float32)
    r220_value = tl.broadcast_to(r220_value, [$(384)])
    tl.store($(candidate_vsoftmax_backward_data_1_ptr) + r220_z, r220_value)
    r221_z = tl.arange($(384))
    r221_value = tl.where(tl.load($(candidate_vgt_10_ptr) + ((((r221_z / $(12)) % $(4)) * $(12)) + (r221_z % $(12))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_1_ptr) + ((((r221_z / $(48)) % $(8)) * $(48)) + (((r221_z / $(12)) % $(4)) * $(12)) + (r221_z % $(12))), dtype=tl.float32))
    r221_value = r221_value.to(tl.float32)
    r221_value = tl.broadcast_to(r221_value, [$(384)])
    tl.store($(candidate_vmasked_fill_13_ptr) + r221_z, r221_value)
    r222_z = tl.arange($(384))
    r222_value = (tl.load($(candidate_vmasked_fill_13_ptr) + ((((r222_z / $(48)) % $(8)) * $(48)) + (((r222_z / $(12)) % $(4)) * $(12)) + (r222_z % $(12))), dtype=tl.float32) * 0.07216878364870322)
    r222_value = r222_value.to(tl.float32)
    r222_value = tl.broadcast_to(r222_value, [$(384)])
    tl.store($(candidate_vmul_36_ptr) + r222_z, r222_value)
    r223_row = tl.arange($(192))
    r223_col = tl.arange($(12))
    r223_lane = tl.arange($(4))
    for r223_group in range($(8)) {
      r223_left = tl.load($(candidate_vto_copy_60_ptr) + (((((((r223_group % $(8)) * $(768)) + (r223_lane[None, :] * $(192)) + r223_row[:, None]) / $(768)) % $(8)) * $(768)) + ((((((r223_group % $(8)) * $(768)) + (r223_lane[None, :] * $(192)) + r223_row[:, None]) / $(192)) % $(4)) * $(192)) + ((((r223_group % $(8)) * $(768)) + (r223_lane[None, :] * $(192)) + r223_row[:, None]) % $(192))), dtype=tl.float32)
      r223_right = tl.load($(candidate_vmul_36_ptr) + (((((((r223_group % $(8)) * $(48)) + (r223_lane[:, None] * $(12)) + r223_col[None, :]) / $(48)) % $(8)) * $(48)) + ((((((r223_group % $(8)) * $(48)) + (r223_lane[:, None] * $(12)) + r223_col[None, :]) / $(12)) % $(4)) * $(12)) + ((((r223_group % $(8)) * $(48)) + (r223_lane[:, None] * $(12)) + r223_col[None, :]) % $(12))), dtype=tl.float32)
      r223_value = tl.dot(r223_left, r223_right)
      r223_value = r223_value.to(tl.float32)
      tl.store($(candidate_vbmm_39_ptr) + ((r223_group * $(2304)) + (r223_row[:, None] * $(12)) + r223_col[None, :]), r223_value)
    }
    r224_z = tl.arange($(131072))
    r224_value = tl.where(((((r224_z / $(1024)) % $(128)) >= $(48)) & (((r224_z / $(1024)) % $(128)) < $(60)) & ((((r224_z / $(1024)) % $(128)) - $(48)) % $(1) == $(0))), tl.load($(candidate_vbmm_37_ptr) + ((((((((r224_z / $(128)) % $(8)) * $(1536)) + ((((r224_z / $(1024)) % $(128)) - $(48)) * $(128)) + (r224_z % $(128))) / $(1536)) % $(8)) * $(1536)) + ((((((r224_z / $(128)) % $(8)) * $(12)) + (((r224_z / $(1024)) % $(128)) - $(48))) % $(12)) * $(128)) + (r224_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r224_value = r224_value.to(tl.bfloat16)
    r224_value = tl.broadcast_to(r224_value, [$(131072)])
    tl.store($(candidate_vslice_backward_2_ptr) + r224_z, r224_value)
    r225_z = tl.arange($(131072))
    r225_value = (tl.load($(candidate_vslice_backward_ptr) + ((((r225_z / $(1024)) % $(128)) * $(1024)) + (((r225_z / $(128)) % $(8)) * $(128)) + (r225_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_2_ptr) + ((((r225_z / $(1024)) % $(128)) * $(1024)) + (((r225_z / $(128)) % $(8)) * $(128)) + (r225_z % $(128))), dtype=tl.bfloat16))
    r225_value = r225_value.to(tl.bfloat16)
    r225_value = tl.broadcast_to(r225_value, [$(131072)])
    tl.store($(candidate_vadd_20_ptr) + r225_z, r225_value)
    r226_z = tl.arange($(18432))
    r226_value = tl.load($(candidate_vbmm_39_ptr) + ((((r226_z / $(2304)) % $(8)) * $(2304)) + ((r226_z % $(192)) * $(12)) + ((r226_z / $(192)) % $(12))), dtype=tl.float32)
    r226_value = r226_value.to(tl.bfloat16)
    r226_value = tl.broadcast_to(r226_value, [$(18432)])
    tl.store($(candidate_vto_copy_69_ptr) + r226_z, r226_value)
    r227_z = tl.arange($(196608))
    r227_value = tl.where(((((r227_z / $(1536)) % $(128)) >= $(48)) & (((r227_z / $(1536)) % $(128)) < $(60)) & ((((r227_z / $(1536)) % $(128)) - $(48)) % $(1) == $(0))), tl.load($(candidate_vto_copy_69_ptr) + ((((r227_z / $(192)) % $(8)) * $(2304)) + (((((r227_z / $(1536)) % $(128)) - $(48)) % $(12)) * $(192)) + (r227_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r227_value = r227_value.to(tl.bfloat16)
    r227_value = tl.broadcast_to(r227_value, [$(196608)])
    tl.store($(candidate_vslice_backward_3_ptr) + r227_z, r227_value)
    r228_z = tl.arange($(196608))
    r228_value = (tl.load($(candidate_vslice_backward_1_ptr) + ((((r228_z / $(1536)) % $(128)) * $(1536)) + (((r228_z / $(192)) % $(8)) * $(192)) + (r228_z % $(192))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_3_ptr) + ((((r228_z / $(1536)) % $(128)) * $(1536)) + (((r228_z / $(192)) % $(8)) * $(192)) + (r228_z % $(192))), dtype=tl.bfloat16))
    r228_value = r228_value.to(tl.bfloat16)
    r228_value = tl.broadcast_to(r228_value, [$(196608)])
    tl.store($(candidate_vadd_21_ptr) + r228_z, r228_value)
    r229_row = tl.arange($(36))
    r229_col = tl.arange($(128))
    r229_lane = tl.arange($(12))
    for r229_group in range($(8)) {
      r229_left = tl.load($(candidate_vto_copy_59_ptr) + (((((((r229_group % $(8)) * $(432)) + (r229_lane[None, :] * $(36)) + r229_row[:, None]) / $(432)) % $(8)) * $(432)) + ((((((r229_group % $(8)) * $(432)) + (r229_lane[None, :] * $(36)) + r229_row[:, None]) / $(36)) % $(12)) * $(36)) + ((((r229_group % $(8)) * $(432)) + (r229_lane[None, :] * $(36)) + r229_row[:, None]) % $(36))), dtype=tl.bfloat16)
      r229_right = tl.load($(candidate_vmm_4_ptr) + (((((((r229_group % $(8)) * $(1536)) + (r229_lane[:, None] * $(128)) + r229_col[None, :]) / $(128)) % $(12)) * $(1024)) + ((((((r229_group % $(8)) * $(1536)) + (r229_lane[:, None] * $(128)) + r229_col[None, :]) / $(1536)) % $(8)) * $(128)) + ((((r229_group % $(8)) * $(1536)) + (r229_lane[:, None] * $(128)) + r229_col[None, :]) % $(128)) + $(98304)), dtype=tl.bfloat16)
      r229_value = tl.dot(r229_left, r229_right)
      r229_value = r229_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_40_ptr) + ((r229_group * $(4608)) + (r229_row[:, None] * $(128)) + r229_col[None, :]), r229_value)
    }
    r230_row = tl.arange($(12))
    r230_col = tl.arange($(36))
    r230_lane = tl.arange($(128))
    for r230_group in range($(8)) {
      r230_left = tl.load($(candidate_vmm_4_ptr) + (((((((r230_group % $(8)) * $(1536)) + (r230_row[:, None] * $(128)) + r230_lane[None, :]) / $(128)) % $(12)) * $(1024)) + ((((((r230_group % $(8)) * $(1536)) + (r230_row[:, None] * $(128)) + r230_lane[None, :]) / $(1536)) % $(8)) * $(128)) + ((((r230_group % $(8)) * $(1536)) + (r230_row[:, None] * $(128)) + r230_lane[None, :]) % $(128)) + $(98304)), dtype=tl.bfloat16)
      r230_right = tl.load($(candidate_vbmm_27_ptr) + (((((((r230_group % $(8)) * $(4608)) + (r230_col[None, :] * $(128)) + r230_lane[:, None]) / $(128)) % $(36)) * $(1024)) + ((((((r230_group % $(8)) * $(4608)) + (r230_col[None, :] * $(128)) + r230_lane[:, None]) / $(4608)) % $(8)) * $(128)) + ((((r230_group % $(8)) * $(4608)) + (r230_col[None, :] * $(128)) + r230_lane[:, None]) % $(128))), dtype=tl.bfloat16)
      r230_value = tl.dot(r230_left, r230_right)
      r230_value = r230_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_41_ptr) + ((r230_group * $(432)) + (r230_row[:, None] * $(36)) + r230_col[None, :]), r230_value)
    }
    r231_z = tl.arange($(3456))
    r231_value = tl.load($(candidate_vbmm_41_ptr) + ((((r231_z / $(432)) % $(8)) * $(432)) + (((r231_z / $(36)) % $(12)) * $(36)) + (r231_z % $(36))), dtype=tl.bfloat16)
    r231_value = r231_value.to(tl.float32)
    r231_value = tl.broadcast_to(r231_value, [$(3456)])
    tl.store($(candidate_vto_copy_70_ptr) + r231_z, r231_value)
    r232_z = tl.arange($(3456))
    r232_lane = tl.arange($(36))
    r232_row_values = tl.load($(candidate_vsoftmax_9_ptr) + ((((r232_z[:, None] / $(432)) % $(8)) * $(432)) + (((r232_z[:, None] / $(36)) % $(12)) * $(36)) + (r232_lane[None, :] % $(36))), dtype=tl.float32)
    r232_row_grad = tl.load($(candidate_vto_copy_70_ptr) + ((((r232_z[:, None] / $(432)) % $(8)) * $(432)) + (((r232_z[:, None] / $(36)) % $(12)) * $(36)) + (r232_lane[None, :] % $(36))), dtype=tl.float32)
    r232_weighted = r232_row_grad * r232_row_values
    r232_dot_grad = tl.sum(r232_weighted, axis=1)
    r232_value = (tl.load($(candidate_vto_copy_70_ptr) + ((((r232_z / $(432)) % $(8)) * $(432)) + (((r232_z / $(36)) % $(12)) * $(36)) + (r232_z % $(36))), dtype=tl.float32) - r232_dot_grad) * tl.load($(candidate_vsoftmax_9_ptr) + ((((r232_z / $(432)) % $(8)) * $(432)) + (((r232_z / $(36)) % $(12)) * $(36)) + (r232_z % $(36))), dtype=tl.float32)
    r232_value = r232_value.to(tl.float32)
    r232_value = tl.broadcast_to(r232_value, [$(3456)])
    tl.store($(candidate_vsoftmax_backward_data_2_ptr) + r232_z, r232_value)
    r233_z = tl.arange($(3456))
    r233_value = tl.where(tl.load($(candidate_vgt_9_ptr) + ((((r233_z / $(36)) % $(12)) * $(36)) + (r233_z % $(36))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_2_ptr) + ((((r233_z / $(432)) % $(8)) * $(432)) + (((r233_z / $(36)) % $(12)) * $(36)) + (r233_z % $(36))), dtype=tl.float32))
    r233_value = r233_value.to(tl.float32)
    r233_value = tl.broadcast_to(r233_value, [$(3456)])
    tl.store($(candidate_vmasked_fill_14_ptr) + r233_z, r233_value)
    r234_z = tl.arange($(3456))
    r234_value = (tl.load($(candidate_vmasked_fill_14_ptr) + ((((r234_z / $(432)) % $(8)) * $(432)) + (((r234_z / $(36)) % $(12)) * $(36)) + (r234_z % $(36))), dtype=tl.float32) * 0.07216878364870322)
    r234_value = r234_value.to(tl.float32)
    r234_value = tl.broadcast_to(r234_value, [$(3456)])
    tl.store($(candidate_vmul_37_ptr) + r234_z, r234_value)
    r235_row = tl.arange($(192))
    r235_col = tl.arange($(36))
    r235_lane = tl.arange($(12))
    for r235_group in range($(8)) {
      r235_left = tl.load($(candidate_vto_copy_57_ptr) + (((((((r235_group % $(8)) * $(2304)) + (r235_lane[None, :] * $(192)) + r235_row[:, None]) / $(2304)) % $(8)) * $(2304)) + ((((((r235_group % $(8)) * $(2304)) + (r235_lane[None, :] * $(192)) + r235_row[:, None]) / $(192)) % $(12)) * $(192)) + ((((r235_group % $(8)) * $(2304)) + (r235_lane[None, :] * $(192)) + r235_row[:, None]) % $(192))), dtype=tl.float32)
      r235_right = tl.load($(candidate_vmul_37_ptr) + (((((((r235_group % $(8)) * $(432)) + (r235_lane[:, None] * $(36)) + r235_col[None, :]) / $(432)) % $(8)) * $(432)) + ((((((r235_group % $(8)) * $(432)) + (r235_lane[:, None] * $(36)) + r235_col[None, :]) / $(36)) % $(12)) * $(36)) + ((((r235_group % $(8)) * $(432)) + (r235_lane[:, None] * $(36)) + r235_col[None, :]) % $(36))), dtype=tl.float32)
      r235_value = tl.dot(r235_left, r235_right)
      r235_value = r235_value.to(tl.float32)
      tl.store($(candidate_vbmm_42_ptr) + ((r235_group * $(6912)) + (r235_row[:, None] * $(36)) + r235_col[None, :]), r235_value)
    }
    r236_z = tl.arange($(131072))
    r236_value = tl.where(((((r236_z / $(1024)) % $(128)) >= $(0)) & (((r236_z / $(1024)) % $(128)) < $(36)) & ((((r236_z / $(1024)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vbmm_40_ptr) + ((((((((r236_z / $(128)) % $(8)) * $(4608)) + ((((r236_z / $(1024)) % $(128)) - $(0)) * $(128)) + (r236_z % $(128))) / $(4608)) % $(8)) * $(4608)) + ((((((r236_z / $(128)) % $(8)) * $(36)) + (((r236_z / $(1024)) % $(128)) - $(0))) % $(36)) * $(128)) + (r236_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r236_value = r236_value.to(tl.bfloat16)
    r236_value = tl.broadcast_to(r236_value, [$(131072)])
    tl.store($(candidate_vslice_backward_4_ptr) + r236_z, r236_value)
    r237_z = tl.arange($(131072))
    r237_value = (tl.load($(candidate_vadd_20_ptr) + ((((r237_z / $(1024)) % $(128)) * $(1024)) + (((r237_z / $(128)) % $(8)) * $(128)) + (r237_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_4_ptr) + ((((r237_z / $(1024)) % $(128)) * $(1024)) + (((r237_z / $(128)) % $(8)) * $(128)) + (r237_z % $(128))), dtype=tl.bfloat16))
    r237_value = r237_value.to(tl.bfloat16)
    r237_value = tl.broadcast_to(r237_value, [$(131072)])
    tl.store($(candidate_vadd_22_ptr) + r237_z, r237_value)
    r238_z = tl.arange($(55296))
    r238_value = tl.load($(candidate_vbmm_42_ptr) + ((((r238_z / $(6912)) % $(8)) * $(6912)) + ((r238_z % $(192)) * $(36)) + ((r238_z / $(192)) % $(36))), dtype=tl.float32)
    r238_value = r238_value.to(tl.bfloat16)
    r238_value = tl.broadcast_to(r238_value, [$(55296)])
    tl.store($(candidate_vto_copy_71_ptr) + r238_z, r238_value)
    r239_z = tl.arange($(196608))
    r239_value = tl.where(((((r239_z / $(1536)) % $(128)) >= $(0)) & (((r239_z / $(1536)) % $(128)) < $(36)) & ((((r239_z / $(1536)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vto_copy_71_ptr) + ((((r239_z / $(192)) % $(8)) * $(6912)) + (((((r239_z / $(1536)) % $(128)) - $(0)) % $(36)) * $(192)) + (r239_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r239_value = r239_value.to(tl.bfloat16)
    r239_value = tl.broadcast_to(r239_value, [$(196608)])
    tl.store($(candidate_vslice_backward_5_ptr) + r239_z, r239_value)
    r240_z = tl.arange($(196608))
    r240_value = (tl.load($(candidate_vadd_21_ptr) + ((((r240_z / $(1536)) % $(128)) * $(1536)) + (((r240_z / $(192)) % $(8)) * $(192)) + (r240_z % $(192))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_5_ptr) + ((((r240_z / $(1536)) % $(128)) * $(1536)) + (((r240_z / $(192)) % $(8)) * $(192)) + (r240_z % $(192))), dtype=tl.bfloat16))
    r240_value = r240_value.to(tl.bfloat16)
    r240_value = tl.broadcast_to(r240_value, [$(196608)])
    tl.store($(candidate_vadd_23_ptr) + r240_z, r240_value)
    r241_z = tl.arange($(12288))
    r241_value = tl.cast(0.0, tl.bfloat16)
    r241_value = r241_value.to(tl.bfloat16)
    r241_value = tl.broadcast_to(r241_value, [$(12288)])
    tl.store($(candidate_vfill_ptr) + r241_z, r241_value)
    r242_z = tl.arange($(131072))
    r242_value = tl.where(((((r242_z / $(1024)) % $(128)) >= $(36)) & (((r242_z / $(1024)) % $(128)) < $(48)) & ((((r242_z / $(1024)) % $(128)) - $(36)) % $(1) == $(0))), tl.load($(candidate_vfill_ptr) + ((((((r242_z / $(1024)) % $(128)) - $(36)) % $(12)) * $(1024)) + (((r242_z / $(128)) % $(8)) * $(128)) + (r242_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vadd_22_ptr) + ((((r242_z / $(1024)) % $(128)) * $(1024)) + (((r242_z / $(128)) % $(8)) * $(128)) + (r242_z % $(128))), dtype=tl.bfloat16))
    r242_value = r242_value.to(tl.bfloat16)
    r242_value = tl.broadcast_to(r242_value, [$(131072)])
    tl.store($(candidate_vslice_scatter_ptr) + r242_z, r242_value)
    r243_z = tl.arange($(4096))
    r243_value = tl.cast(0.0, tl.bfloat16)
    r243_value = r243_value.to(tl.bfloat16)
    r243_value = tl.broadcast_to(r243_value, [$(4096)])
    tl.store($(candidate_vfill_1_ptr) + r243_z, r243_value)
    r244_z = tl.arange($(131072))
    r244_value = tl.where(((((r244_z / $(1024)) % $(128)) >= $(60)) & (((r244_z / $(1024)) % $(128)) < $(64)) & ((((r244_z / $(1024)) % $(128)) - $(60)) % $(1) == $(0))), tl.load($(candidate_vfill_1_ptr) + ((((((r244_z / $(1024)) % $(128)) - $(60)) % $(4)) * $(1024)) + (((r244_z / $(128)) % $(8)) * $(128)) + (r244_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vslice_scatter_ptr) + ((((r244_z / $(1024)) % $(128)) * $(1024)) + (((r244_z / $(128)) % $(8)) * $(128)) + (r244_z % $(128))), dtype=tl.bfloat16))
    r244_value = r244_value.to(tl.bfloat16)
    r244_value = tl.broadcast_to(r244_value, [$(131072)])
    tl.store($(candidate_vslice_scatter_1_ptr) + r244_z, r244_value)
    r245_z = tl.arange($(16384))
    r245_value = tl.cast(0.0, tl.bfloat16)
    r245_value = r245_value.to(tl.bfloat16)
    r245_value = tl.broadcast_to(r245_value, [$(16384)])
    tl.store($(candidate_vfill_2_ptr) + r245_z, r245_value)
    r246_z = tl.arange($(131072))
    r246_value = tl.where(((((r246_z / $(1024)) % $(128)) >= $(112)) & (((r246_z / $(1024)) % $(128)) < $(128)) & ((((r246_z / $(1024)) % $(128)) - $(112)) % $(1) == $(0))), tl.load($(candidate_vfill_2_ptr) + ((((((r246_z / $(1024)) % $(128)) - $(112)) % $(16)) * $(1024)) + (((r246_z / $(128)) % $(8)) * $(128)) + (r246_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vslice_scatter_1_ptr) + ((((r246_z / $(1024)) % $(128)) * $(1024)) + (((r246_z / $(128)) % $(8)) * $(128)) + (r246_z % $(128))), dtype=tl.bfloat16))
    r246_value = r246_value.to(tl.bfloat16)
    r246_value = tl.broadcast_to(r246_value, [$(131072)])
    tl.store($(candidate_vslice_scatter_2_ptr) + r246_z, r246_value)
    r247_z = tl.arange($(18432))
    r247_value = tl.cast(0.0, tl.bfloat16)
    r247_value = r247_value.to(tl.bfloat16)
    r247_value = tl.broadcast_to(r247_value, [$(18432)])
    tl.store($(candidate_vfill_3_ptr) + r247_z, r247_value)
    r248_z = tl.arange($(196608))
    r248_value = tl.where(((((r248_z / $(1536)) % $(128)) >= $(36)) & (((r248_z / $(1536)) % $(128)) < $(48)) & ((((r248_z / $(1536)) % $(128)) - $(36)) % $(1) == $(0))), tl.load($(candidate_vfill_3_ptr) + ((((((r248_z / $(1536)) % $(128)) - $(36)) % $(12)) * $(1536)) + (((r248_z / $(192)) % $(8)) * $(192)) + (r248_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vadd_23_ptr) + ((((r248_z / $(1536)) % $(128)) * $(1536)) + (((r248_z / $(192)) % $(8)) * $(192)) + (r248_z % $(192))), dtype=tl.bfloat16))
    r248_value = r248_value.to(tl.bfloat16)
    r248_value = tl.broadcast_to(r248_value, [$(196608)])
    tl.store($(candidate_vslice_scatter_3_ptr) + r248_z, r248_value)
    r249_z = tl.arange($(6144))
    r249_value = tl.cast(0.0, tl.bfloat16)
    r249_value = r249_value.to(tl.bfloat16)
    r249_value = tl.broadcast_to(r249_value, [$(6144)])
    tl.store($(candidate_vfill_4_ptr) + r249_z, r249_value)
    r250_z = tl.arange($(196608))
    r250_value = tl.where(((((r250_z / $(1536)) % $(128)) >= $(60)) & (((r250_z / $(1536)) % $(128)) < $(64)) & ((((r250_z / $(1536)) % $(128)) - $(60)) % $(1) == $(0))), tl.load($(candidate_vfill_4_ptr) + ((((((r250_z / $(1536)) % $(128)) - $(60)) % $(4)) * $(1536)) + (((r250_z / $(192)) % $(8)) * $(192)) + (r250_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vslice_scatter_3_ptr) + ((((r250_z / $(1536)) % $(128)) * $(1536)) + (((r250_z / $(192)) % $(8)) * $(192)) + (r250_z % $(192))), dtype=tl.bfloat16))
    r250_value = r250_value.to(tl.bfloat16)
    r250_value = tl.broadcast_to(r250_value, [$(196608)])
    tl.store($(candidate_vslice_scatter_4_ptr) + r250_z, r250_value)
    r251_z = tl.arange($(24576))
    r251_value = tl.cast(0.0, tl.bfloat16)
    r251_value = r251_value.to(tl.bfloat16)
    r251_value = tl.broadcast_to(r251_value, [$(24576)])
    tl.store($(candidate_vfill_5_ptr) + r251_z, r251_value)
    r252_z = tl.arange($(196608))
    r252_value = tl.where(((((r252_z / $(1536)) % $(128)) >= $(112)) & (((r252_z / $(1536)) % $(128)) < $(128)) & ((((r252_z / $(1536)) % $(128)) - $(112)) % $(1) == $(0))), tl.load($(candidate_vfill_5_ptr) + ((((((r252_z / $(1536)) % $(128)) - $(112)) % $(16)) * $(1536)) + (((r252_z / $(192)) % $(8)) * $(192)) + (r252_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vslice_scatter_4_ptr) + ((((r252_z / $(1536)) % $(128)) * $(1536)) + (((r252_z / $(192)) % $(8)) * $(192)) + (r252_z % $(192))), dtype=tl.bfloat16))
    r252_value = r252_value.to(tl.bfloat16)
    r252_value = tl.broadcast_to(r252_value, [$(196608)])
    tl.store($(candidate_vslice_scatter_5_ptr) + r252_z, r252_value)
    r253_row = tl.arange($(256))
    r253_col = tl.arange($(1024))
    r253_lane = tl.arange($(128))
    for r253_group in range($(1)) {
      r253_left = tl.load($(candidate_vindex_2_ptr) + ((((((r253_lane[None, :] * $(256)) + r253_row[:, None]) / $(256)) % $(128)) * $(320)) + (((r253_lane[None, :] * $(256)) + r253_row[:, None]) % $(256))), dtype=tl.bfloat16)
      r253_right = tl.load($(candidate_vslice_scatter_2_ptr) + ((((((r253_lane[:, None] * $(1024)) + r253_col[None, :]) / $(1024)) % $(128)) * $(1024)) + (((((r253_lane[:, None] * $(1024)) + r253_col[None, :]) / $(128)) % $(8)) * $(128)) + (((r253_lane[:, None] * $(1024)) + r253_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r253_value = tl.dot(r253_left, r253_right)
      r253_value = r253_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_43_ptr) + ((r253_group * $(262144)) + (r253_row[:, None] * $(1024)) + r253_col[None, :]), r253_value)
    }
    r254_row = tl.arange($(256))
    r254_col = tl.arange($(1024))
    r254_lane = tl.arange($(128))
    for r254_group in range($(1)) {
      r254_left = tl.load($(candidate_vindex_2_ptr) + ((((((r254_lane[None, :] * $(256)) + r254_row[:, None]) / $(256)) % $(128)) * $(320)) + (((r254_lane[None, :] * $(256)) + r254_row[:, None]) % $(256))), dtype=tl.bfloat16)
      r254_right = tl.load($(candidate_vslice_scatter_5_ptr) + ((((((r254_lane[:, None] * $(1024)) + r254_col[None, :]) / $(1024)) % $(128)) * $(1536)) + (((((r254_lane[:, None] * $(1024)) + r254_col[None, :]) / $(128)) % $(8)) * $(192)) + (((r254_lane[:, None] * $(1024)) + r254_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r254_value = tl.dot(r254_left, r254_right)
      r254_value = r254_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_44_ptr) + ((r254_group * $(262144)) + (r254_row[:, None] * $(1024)) + r254_col[None, :]), r254_value)
    }
    r255_z = tl.arange($(524288))
    r255_value = tl.where(((((r255_z / $(256)) % $(256)) >= $(128)) & (((r255_z / $(256)) % $(256)) < $(256)) & ((((r255_z / $(256)) % $(256)) - $(128)) % $(1) == $(0))), tl.load($(candidate_vbmm_43_ptr) + (((r255_z % $(256)) * $(1024)) + (((((((r255_z / $(65536)) % $(8)) * $(32768)) + ((((r255_z / $(256)) % $(256)) - $(128)) * $(256)) + (r255_z % $(256))) / $(32768)) % $(8)) * $(128)) + (((((r255_z / $(65536)) % $(8)) * $(128)) + (((r255_z / $(256)) % $(256)) - $(128))) % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r255_value = r255_value.to(tl.bfloat16)
    r255_value = tl.broadcast_to(r255_value, [$(524288)])
    tl.store($(candidate_vslice_backward_6_ptr) + r255_z, r255_value)
    r256_z = tl.arange($(524288))
    r256_value = tl.where(((((r256_z / $(256)) % $(256)) >= $(0)) & (((r256_z / $(256)) % $(256)) < $(128)) & ((((r256_z / $(256)) % $(256)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vbmm_44_ptr) + (((r256_z % $(256)) * $(1024)) + (((((((r256_z / $(65536)) % $(8)) * $(32768)) + ((((r256_z / $(256)) % $(256)) - $(0)) * $(256)) + (r256_z % $(256))) / $(32768)) % $(8)) * $(128)) + (((((r256_z / $(65536)) % $(8)) * $(128)) + (((r256_z / $(256)) % $(256)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r256_value = r256_value.to(tl.bfloat16)
    r256_value = tl.broadcast_to(r256_value, [$(524288)])
    tl.store($(candidate_vslice_backward_7_ptr) + r256_z, r256_value)
    r257_z = tl.arange($(524288))
    r257_value = (tl.load($(candidate_vslice_backward_6_ptr) + ((((r257_z / $(65536)) % $(8)) * $(65536)) + (((r257_z / $(256)) % $(256)) * $(256)) + (r257_z % $(256))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_7_ptr) + ((((r257_z / $(65536)) % $(8)) * $(65536)) + (((r257_z / $(256)) % $(256)) * $(256)) + (r257_z % $(256))), dtype=tl.bfloat16))
    r257_value = r257_value.to(tl.bfloat16)
    r257_value = tl.broadcast_to(r257_value, [$(524288)])
    tl.store($(candidate_vadd_24_ptr) + r257_z, r257_value)
    r258_row = tl.arange($(32))
    r258_col = tl.arange($(128))
    r258_lane = tl.arange($(16))
    for r258_group in range($(8)) {
      r258_left = tl.load($(candidate_vto_copy_56_ptr) + (((((((r258_group % $(8)) * $(512)) + (r258_lane[None, :] * $(32)) + r258_row[:, None]) / $(512)) % $(8)) * $(512)) + ((((((r258_group % $(8)) * $(512)) + (r258_lane[None, :] * $(32)) + r258_row[:, None]) / $(32)) % $(16)) * $(32)) + ((((r258_group % $(8)) * $(512)) + (r258_lane[None, :] * $(32)) + r258_row[:, None]) % $(32))), dtype=tl.bfloat16)
      r258_right = tl.load($(candidate_vmm_4_ptr) + (((((((r258_group % $(8)) * $(2048)) + (r258_lane[:, None] * $(128)) + r258_col[None, :]) / $(128)) % $(16)) * $(1024)) + ((((((r258_group % $(8)) * $(2048)) + (r258_lane[:, None] * $(128)) + r258_col[None, :]) / $(2048)) % $(8)) * $(128)) + ((((r258_group % $(8)) * $(2048)) + (r258_lane[:, None] * $(128)) + r258_col[None, :]) % $(128)) + $(81920)), dtype=tl.bfloat16)
      r258_value = tl.dot(r258_left, r258_right)
      r258_value = r258_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_45_ptr) + ((r258_group * $(4096)) + (r258_row[:, None] * $(128)) + r258_col[None, :]), r258_value)
    }
    r259_row = tl.arange($(16))
    r259_col = tl.arange($(32))
    r259_lane = tl.arange($(128))
    for r259_group in range($(8)) {
      r259_left = tl.load($(candidate_vmm_4_ptr) + (((((((r259_group % $(8)) * $(2048)) + (r259_row[:, None] * $(128)) + r259_lane[None, :]) / $(128)) % $(16)) * $(1024)) + ((((((r259_group % $(8)) * $(2048)) + (r259_row[:, None] * $(128)) + r259_lane[None, :]) / $(2048)) % $(8)) * $(128)) + ((((r259_group % $(8)) * $(2048)) + (r259_row[:, None] * $(128)) + r259_lane[None, :]) % $(128)) + $(81920)), dtype=tl.bfloat16)
      r259_right = tl.load($(candidate_vbmm_19_ptr) + (((((((r259_group % $(8)) * $(4096)) + (r259_col[None, :] * $(128)) + r259_lane[:, None]) / $(128)) % $(32)) * $(1024)) + ((((((r259_group % $(8)) * $(4096)) + (r259_col[None, :] * $(128)) + r259_lane[:, None]) / $(4096)) % $(8)) * $(128)) + ((((r259_group % $(8)) * $(4096)) + (r259_col[None, :] * $(128)) + r259_lane[:, None]) % $(128)) + $(65536)), dtype=tl.bfloat16)
      r259_value = tl.dot(r259_left, r259_right)
      r259_value = r259_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_46_ptr) + ((r259_group * $(512)) + (r259_row[:, None] * $(32)) + r259_col[None, :]), r259_value)
    }
    r260_z = tl.arange($(4096))
    r260_value = tl.load($(candidate_vbmm_46_ptr) + ((((r260_z / $(512)) % $(8)) * $(512)) + (((r260_z / $(32)) % $(16)) * $(32)) + (r260_z % $(32))), dtype=tl.bfloat16)
    r260_value = r260_value.to(tl.float32)
    r260_value = tl.broadcast_to(r260_value, [$(4096)])
    tl.store($(candidate_vto_copy_72_ptr) + r260_z, r260_value)
    r261_z = tl.arange($(4096))
    r261_lane = tl.arange($(32))
    r261_row_values = tl.load($(candidate_vsoftmax_8_ptr) + ((((r261_z[:, None] / $(512)) % $(8)) * $(512)) + (((r261_z[:, None] / $(32)) % $(16)) * $(32)) + (r261_lane[None, :] % $(32))), dtype=tl.float32)
    r261_row_grad = tl.load($(candidate_vto_copy_72_ptr) + ((((r261_z[:, None] / $(512)) % $(8)) * $(512)) + (((r261_z[:, None] / $(32)) % $(16)) * $(32)) + (r261_lane[None, :] % $(32))), dtype=tl.float32)
    r261_weighted = r261_row_grad * r261_row_values
    r261_dot_grad = tl.sum(r261_weighted, axis=1)
    r261_value = (tl.load($(candidate_vto_copy_72_ptr) + ((((r261_z / $(512)) % $(8)) * $(512)) + (((r261_z / $(32)) % $(16)) * $(32)) + (r261_z % $(32))), dtype=tl.float32) - r261_dot_grad) * tl.load($(candidate_vsoftmax_8_ptr) + ((((r261_z / $(512)) % $(8)) * $(512)) + (((r261_z / $(32)) % $(16)) * $(32)) + (r261_z % $(32))), dtype=tl.float32)
    r261_value = r261_value.to(tl.float32)
    r261_value = tl.broadcast_to(r261_value, [$(4096)])
    tl.store($(candidate_vsoftmax_backward_data_3_ptr) + r261_z, r261_value)
    r262_z = tl.arange($(4096))
    r262_value = tl.where(tl.load($(candidate_vgt_8_ptr) + ((((r262_z / $(32)) % $(16)) * $(32)) + (r262_z % $(32))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_3_ptr) + ((((r262_z / $(512)) % $(8)) * $(512)) + (((r262_z / $(32)) % $(16)) * $(32)) + (r262_z % $(32))), dtype=tl.float32))
    r262_value = r262_value.to(tl.float32)
    r262_value = tl.broadcast_to(r262_value, [$(4096)])
    tl.store($(candidate_vmasked_fill_15_ptr) + r262_z, r262_value)
    r263_z = tl.arange($(4096))
    r263_value = (tl.load($(candidate_vmasked_fill_15_ptr) + ((((r263_z / $(512)) % $(8)) * $(512)) + (((r263_z / $(32)) % $(16)) * $(32)) + (r263_z % $(32))), dtype=tl.float32) * 0.07216878364870322)
    r263_value = r263_value.to(tl.float32)
    r263_value = tl.broadcast_to(r263_value, [$(4096)])
    tl.store($(candidate_vmul_38_ptr) + r263_z, r263_value)
    r264_row = tl.arange($(192))
    r264_col = tl.arange($(32))
    r264_lane = tl.arange($(16))
    for r264_group in range($(8)) {
      r264_left = tl.load($(candidate_vto_copy_54_ptr) + (((((((r264_group % $(8)) * $(3072)) + (r264_lane[None, :] * $(192)) + r264_row[:, None]) / $(3072)) % $(8)) * $(3072)) + ((((((r264_group % $(8)) * $(3072)) + (r264_lane[None, :] * $(192)) + r264_row[:, None]) / $(192)) % $(16)) * $(192)) + ((((r264_group % $(8)) * $(3072)) + (r264_lane[None, :] * $(192)) + r264_row[:, None]) % $(192))), dtype=tl.float32)
      r264_right = tl.load($(candidate_vmul_38_ptr) + (((((((r264_group % $(8)) * $(512)) + (r264_lane[:, None] * $(32)) + r264_col[None, :]) / $(512)) % $(8)) * $(512)) + ((((((r264_group % $(8)) * $(512)) + (r264_lane[:, None] * $(32)) + r264_col[None, :]) / $(32)) % $(16)) * $(32)) + ((((r264_group % $(8)) * $(512)) + (r264_lane[:, None] * $(32)) + r264_col[None, :]) % $(32))), dtype=tl.float32)
      r264_value = tl.dot(r264_left, r264_right)
      r264_value = r264_value.to(tl.float32)
      tl.store($(candidate_vbmm_47_ptr) + ((r264_group * $(6144)) + (r264_row[:, None] * $(32)) + r264_col[None, :]), r264_value)
    }
    r265_z = tl.arange($(131072))
    r265_value = tl.where(((((r265_z / $(1024)) % $(128)) >= $(64)) & (((r265_z / $(1024)) % $(128)) < $(96)) & ((((r265_z / $(1024)) % $(128)) - $(64)) % $(1) == $(0))), tl.load($(candidate_vbmm_45_ptr) + ((((((((r265_z / $(128)) % $(8)) * $(4096)) + ((((r265_z / $(1024)) % $(128)) - $(64)) * $(128)) + (r265_z % $(128))) / $(4096)) % $(8)) * $(4096)) + ((((((r265_z / $(128)) % $(8)) * $(32)) + (((r265_z / $(1024)) % $(128)) - $(64))) % $(32)) * $(128)) + (r265_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r265_value = r265_value.to(tl.bfloat16)
    r265_value = tl.broadcast_to(r265_value, [$(131072)])
    tl.store($(candidate_vslice_backward_8_ptr) + r265_z, r265_value)
    r266_z = tl.arange($(49152))
    r266_value = tl.load($(candidate_vbmm_47_ptr) + ((((r266_z / $(6144)) % $(8)) * $(6144)) + ((r266_z % $(192)) * $(32)) + ((r266_z / $(192)) % $(32))), dtype=tl.float32)
    r266_value = r266_value.to(tl.bfloat16)
    r266_value = tl.broadcast_to(r266_value, [$(49152)])
    tl.store($(candidate_vto_copy_73_ptr) + r266_z, r266_value)
    r267_z = tl.arange($(196608))
    r267_value = tl.where(((((r267_z / $(1536)) % $(128)) >= $(64)) & (((r267_z / $(1536)) % $(128)) < $(96)) & ((((r267_z / $(1536)) % $(128)) - $(64)) % $(1) == $(0))), tl.load($(candidate_vto_copy_73_ptr) + ((((r267_z / $(192)) % $(8)) * $(6144)) + (((((r267_z / $(1536)) % $(128)) - $(64)) % $(32)) * $(192)) + (r267_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r267_value = r267_value.to(tl.bfloat16)
    r267_value = tl.broadcast_to(r267_value, [$(196608)])
    tl.store($(candidate_vslice_backward_9_ptr) + r267_z, r267_value)
    r268_row = tl.arange($(8))
    r268_col = tl.arange($(128))
    r268_lane = tl.arange($(4))
    for r268_group in range($(8)) {
      r268_left = tl.load($(candidate_vto_copy_53_ptr) + (((((((r268_group % $(8)) * $(32)) + (r268_lane[None, :] * $(8)) + r268_row[:, None]) / $(32)) % $(8)) * $(32)) + ((((((r268_group % $(8)) * $(32)) + (r268_lane[None, :] * $(8)) + r268_row[:, None]) / $(8)) % $(4)) * $(8)) + ((((r268_group % $(8)) * $(32)) + (r268_lane[None, :] * $(8)) + r268_row[:, None]) % $(8))), dtype=tl.bfloat16)
      r268_right = tl.load($(candidate_vmm_4_ptr) + (((((((r268_group % $(8)) * $(512)) + (r268_lane[:, None] * $(128)) + r268_col[None, :]) / $(128)) % $(4)) * $(1024)) + ((((((r268_group % $(8)) * $(512)) + (r268_lane[:, None] * $(128)) + r268_col[None, :]) / $(512)) % $(8)) * $(128)) + ((((r268_group % $(8)) * $(512)) + (r268_lane[:, None] * $(128)) + r268_col[None, :]) % $(128)) + $(77824)), dtype=tl.bfloat16)
      r268_value = tl.dot(r268_left, r268_right)
      r268_value = r268_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_48_ptr) + ((r268_group * $(1024)) + (r268_row[:, None] * $(128)) + r268_col[None, :]), r268_value)
    }
    r269_row = tl.arange($(4))
    r269_col = tl.arange($(8))
    r269_lane = tl.arange($(128))
    for r269_group in range($(8)) {
      r269_left = tl.load($(candidate_vmm_4_ptr) + (((((((r269_group % $(8)) * $(512)) + (r269_row[:, None] * $(128)) + r269_lane[None, :]) / $(128)) % $(4)) * $(1024)) + ((((((r269_group % $(8)) * $(512)) + (r269_row[:, None] * $(128)) + r269_lane[None, :]) / $(512)) % $(8)) * $(128)) + ((((r269_group % $(8)) * $(512)) + (r269_row[:, None] * $(128)) + r269_lane[None, :]) % $(128)) + $(77824)), dtype=tl.bfloat16)
      r269_right = tl.load($(candidate_vbmm_19_ptr) + (((((((r269_group % $(8)) * $(1024)) + (r269_col[None, :] * $(128)) + r269_lane[:, None]) / $(128)) % $(8)) * $(1024)) + ((((((r269_group % $(8)) * $(1024)) + (r269_col[None, :] * $(128)) + r269_lane[:, None]) / $(1024)) % $(8)) * $(128)) + ((((r269_group % $(8)) * $(1024)) + (r269_col[None, :] * $(128)) + r269_lane[:, None]) % $(128)) + $(49152)), dtype=tl.bfloat16)
      r269_value = tl.dot(r269_left, r269_right)
      r269_value = r269_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_49_ptr) + ((r269_group * $(32)) + (r269_row[:, None] * $(8)) + r269_col[None, :]), r269_value)
    }
    r270_z = tl.arange($(256))
    r270_value = tl.load($(candidate_vbmm_49_ptr) + ((((r270_z / $(32)) % $(8)) * $(32)) + (((r270_z / $(8)) % $(4)) * $(8)) + (r270_z % $(8))), dtype=tl.bfloat16)
    r270_value = r270_value.to(tl.float32)
    r270_value = tl.broadcast_to(r270_value, [$(256)])
    tl.store($(candidate_vto_copy_74_ptr) + r270_z, r270_value)
    r271_z = tl.arange($(256))
    r271_lane = tl.arange($(8))
    r271_row_values = tl.load($(candidate_vsoftmax_7_ptr) + ((((r271_z[:, None] / $(32)) % $(8)) * $(32)) + (((r271_z[:, None] / $(8)) % $(4)) * $(8)) + (r271_lane[None, :] % $(8))), dtype=tl.float32)
    r271_row_grad = tl.load($(candidate_vto_copy_74_ptr) + ((((r271_z[:, None] / $(32)) % $(8)) * $(32)) + (((r271_z[:, None] / $(8)) % $(4)) * $(8)) + (r271_lane[None, :] % $(8))), dtype=tl.float32)
    r271_weighted = r271_row_grad * r271_row_values
    r271_dot_grad = tl.sum(r271_weighted, axis=1)
    r271_value = (tl.load($(candidate_vto_copy_74_ptr) + ((((r271_z / $(32)) % $(8)) * $(32)) + (((r271_z / $(8)) % $(4)) * $(8)) + (r271_z % $(8))), dtype=tl.float32) - r271_dot_grad) * tl.load($(candidate_vsoftmax_7_ptr) + ((((r271_z / $(32)) % $(8)) * $(32)) + (((r271_z / $(8)) % $(4)) * $(8)) + (r271_z % $(8))), dtype=tl.float32)
    r271_value = r271_value.to(tl.float32)
    r271_value = tl.broadcast_to(r271_value, [$(256)])
    tl.store($(candidate_vsoftmax_backward_data_4_ptr) + r271_z, r271_value)
    r272_z = tl.arange($(256))
    r272_value = tl.where(tl.load($(candidate_vgt_7_ptr) + ((((r272_z / $(8)) % $(4)) * $(8)) + (r272_z % $(8))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_4_ptr) + ((((r272_z / $(32)) % $(8)) * $(32)) + (((r272_z / $(8)) % $(4)) * $(8)) + (r272_z % $(8))), dtype=tl.float32))
    r272_value = r272_value.to(tl.float32)
    r272_value = tl.broadcast_to(r272_value, [$(256)])
    tl.store($(candidate_vmasked_fill_16_ptr) + r272_z, r272_value)
    r273_z = tl.arange($(256))
    r273_value = (tl.load($(candidate_vmasked_fill_16_ptr) + ((((r273_z / $(32)) % $(8)) * $(32)) + (((r273_z / $(8)) % $(4)) * $(8)) + (r273_z % $(8))), dtype=tl.float32) * 0.07216878364870322)
    r273_value = r273_value.to(tl.float32)
    r273_value = tl.broadcast_to(r273_value, [$(256)])
    tl.store($(candidate_vmul_39_ptr) + r273_z, r273_value)
    r274_row = tl.arange($(192))
    r274_col = tl.arange($(8))
    r274_lane = tl.arange($(4))
    for r274_group in range($(8)) {
      r274_left = tl.load($(candidate_vto_copy_51_ptr) + (((((((r274_group % $(8)) * $(768)) + (r274_lane[None, :] * $(192)) + r274_row[:, None]) / $(768)) % $(8)) * $(768)) + ((((((r274_group % $(8)) * $(768)) + (r274_lane[None, :] * $(192)) + r274_row[:, None]) / $(192)) % $(4)) * $(192)) + ((((r274_group % $(8)) * $(768)) + (r274_lane[None, :] * $(192)) + r274_row[:, None]) % $(192))), dtype=tl.float32)
      r274_right = tl.load($(candidate_vmul_39_ptr) + (((((((r274_group % $(8)) * $(32)) + (r274_lane[:, None] * $(8)) + r274_col[None, :]) / $(32)) % $(8)) * $(32)) + ((((((r274_group % $(8)) * $(32)) + (r274_lane[:, None] * $(8)) + r274_col[None, :]) / $(8)) % $(4)) * $(8)) + ((((r274_group % $(8)) * $(32)) + (r274_lane[:, None] * $(8)) + r274_col[None, :]) % $(8))), dtype=tl.float32)
      r274_value = tl.dot(r274_left, r274_right)
      r274_value = r274_value.to(tl.float32)
      tl.store($(candidate_vbmm_50_ptr) + ((r274_group * $(1536)) + (r274_row[:, None] * $(8)) + r274_col[None, :]), r274_value)
    }
    r275_z = tl.arange($(131072))
    r275_value = tl.where(((((r275_z / $(1024)) % $(128)) >= $(48)) & (((r275_z / $(1024)) % $(128)) < $(56)) & ((((r275_z / $(1024)) % $(128)) - $(48)) % $(1) == $(0))), tl.load($(candidate_vbmm_48_ptr) + ((((((((r275_z / $(128)) % $(8)) * $(1024)) + ((((r275_z / $(1024)) % $(128)) - $(48)) * $(128)) + (r275_z % $(128))) / $(1024)) % $(8)) * $(1024)) + ((((((r275_z / $(128)) % $(8)) * $(8)) + (((r275_z / $(1024)) % $(128)) - $(48))) % $(8)) * $(128)) + (r275_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r275_value = r275_value.to(tl.bfloat16)
    r275_value = tl.broadcast_to(r275_value, [$(131072)])
    tl.store($(candidate_vslice_backward_10_ptr) + r275_z, r275_value)
    r276_z = tl.arange($(131072))
    r276_value = (tl.load($(candidate_vslice_backward_8_ptr) + ((((r276_z / $(1024)) % $(128)) * $(1024)) + (((r276_z / $(128)) % $(8)) * $(128)) + (r276_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_10_ptr) + ((((r276_z / $(1024)) % $(128)) * $(1024)) + (((r276_z / $(128)) % $(8)) * $(128)) + (r276_z % $(128))), dtype=tl.bfloat16))
    r276_value = r276_value.to(tl.bfloat16)
    r276_value = tl.broadcast_to(r276_value, [$(131072)])
    tl.store($(candidate_vadd_25_ptr) + r276_z, r276_value)
    r277_z = tl.arange($(12288))
    r277_value = tl.load($(candidate_vbmm_50_ptr) + ((((r277_z / $(1536)) % $(8)) * $(1536)) + ((r277_z % $(192)) * $(8)) + ((r277_z / $(192)) % $(8))), dtype=tl.float32)
    r277_value = r277_value.to(tl.bfloat16)
    r277_value = tl.broadcast_to(r277_value, [$(12288)])
    tl.store($(candidate_vto_copy_75_ptr) + r277_z, r277_value)
    r278_z = tl.arange($(196608))
    r278_value = tl.where(((((r278_z / $(1536)) % $(128)) >= $(48)) & (((r278_z / $(1536)) % $(128)) < $(56)) & ((((r278_z / $(1536)) % $(128)) - $(48)) % $(1) == $(0))), tl.load($(candidate_vto_copy_75_ptr) + ((((r278_z / $(192)) % $(8)) * $(1536)) + (((((r278_z / $(1536)) % $(128)) - $(48)) % $(8)) * $(192)) + (r278_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r278_value = r278_value.to(tl.bfloat16)
    r278_value = tl.broadcast_to(r278_value, [$(196608)])
    tl.store($(candidate_vslice_backward_11_ptr) + r278_z, r278_value)
    r279_z = tl.arange($(196608))
    r279_value = (tl.load($(candidate_vslice_backward_9_ptr) + ((((r279_z / $(1536)) % $(128)) * $(1536)) + (((r279_z / $(192)) % $(8)) * $(192)) + (r279_z % $(192))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_11_ptr) + ((((r279_z / $(1536)) % $(128)) * $(1536)) + (((r279_z / $(192)) % $(8)) * $(192)) + (r279_z % $(192))), dtype=tl.bfloat16))
    r279_value = r279_value.to(tl.bfloat16)
    r279_value = tl.broadcast_to(r279_value, [$(196608)])
    tl.store($(candidate_vadd_26_ptr) + r279_z, r279_value)
    r280_row = tl.arange($(24))
    r280_col = tl.arange($(128))
    r280_lane = tl.arange($(12))
    for r280_group in range($(8)) {
      r280_left = tl.load($(candidate_vto_copy_50_ptr) + (((((((r280_group % $(8)) * $(288)) + (r280_lane[None, :] * $(24)) + r280_row[:, None]) / $(288)) % $(8)) * $(288)) + ((((((r280_group % $(8)) * $(288)) + (r280_lane[None, :] * $(24)) + r280_row[:, None]) / $(24)) % $(12)) * $(24)) + ((((r280_group % $(8)) * $(288)) + (r280_lane[None, :] * $(24)) + r280_row[:, None]) % $(24))), dtype=tl.bfloat16)
      r280_right = tl.load($(candidate_vmm_4_ptr) + (((((((r280_group % $(8)) * $(1536)) + (r280_lane[:, None] * $(128)) + r280_col[None, :]) / $(128)) % $(12)) * $(1024)) + ((((((r280_group % $(8)) * $(1536)) + (r280_lane[:, None] * $(128)) + r280_col[None, :]) / $(1536)) % $(8)) * $(128)) + ((((r280_group % $(8)) * $(1536)) + (r280_lane[:, None] * $(128)) + r280_col[None, :]) % $(128)) + $(65536)), dtype=tl.bfloat16)
      r280_value = tl.dot(r280_left, r280_right)
      r280_value = r280_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_51_ptr) + ((r280_group * $(3072)) + (r280_row[:, None] * $(128)) + r280_col[None, :]), r280_value)
    }
    r281_row = tl.arange($(12))
    r281_col = tl.arange($(24))
    r281_lane = tl.arange($(128))
    for r281_group in range($(8)) {
      r281_left = tl.load($(candidate_vmm_4_ptr) + (((((((r281_group % $(8)) * $(1536)) + (r281_row[:, None] * $(128)) + r281_lane[None, :]) / $(128)) % $(12)) * $(1024)) + ((((((r281_group % $(8)) * $(1536)) + (r281_row[:, None] * $(128)) + r281_lane[None, :]) / $(1536)) % $(8)) * $(128)) + ((((r281_group % $(8)) * $(1536)) + (r281_row[:, None] * $(128)) + r281_lane[None, :]) % $(128)) + $(65536)), dtype=tl.bfloat16)
      r281_right = tl.load($(candidate_vbmm_19_ptr) + (((((((r281_group % $(8)) * $(3072)) + (r281_col[None, :] * $(128)) + r281_lane[:, None]) / $(128)) % $(24)) * $(1024)) + ((((((r281_group % $(8)) * $(3072)) + (r281_col[None, :] * $(128)) + r281_lane[:, None]) / $(3072)) % $(8)) * $(128)) + ((((r281_group % $(8)) * $(3072)) + (r281_col[None, :] * $(128)) + r281_lane[:, None]) % $(128))), dtype=tl.bfloat16)
      r281_value = tl.dot(r281_left, r281_right)
      r281_value = r281_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_52_ptr) + ((r281_group * $(288)) + (r281_row[:, None] * $(24)) + r281_col[None, :]), r281_value)
    }
    r282_z = tl.arange($(2304))
    r282_value = tl.load($(candidate_vbmm_52_ptr) + ((((r282_z / $(288)) % $(8)) * $(288)) + (((r282_z / $(24)) % $(12)) * $(24)) + (r282_z % $(24))), dtype=tl.bfloat16)
    r282_value = r282_value.to(tl.float32)
    r282_value = tl.broadcast_to(r282_value, [$(2304)])
    tl.store($(candidate_vto_copy_76_ptr) + r282_z, r282_value)
    r283_z = tl.arange($(2304))
    r283_lane = tl.arange($(24))
    r283_row_values = tl.load($(candidate_vsoftmax_6_ptr) + ((((r283_z[:, None] / $(288)) % $(8)) * $(288)) + (((r283_z[:, None] / $(24)) % $(12)) * $(24)) + (r283_lane[None, :] % $(24))), dtype=tl.float32)
    r283_row_grad = tl.load($(candidate_vto_copy_76_ptr) + ((((r283_z[:, None] / $(288)) % $(8)) * $(288)) + (((r283_z[:, None] / $(24)) % $(12)) * $(24)) + (r283_lane[None, :] % $(24))), dtype=tl.float32)
    r283_weighted = r283_row_grad * r283_row_values
    r283_dot_grad = tl.sum(r283_weighted, axis=1)
    r283_value = (tl.load($(candidate_vto_copy_76_ptr) + ((((r283_z / $(288)) % $(8)) * $(288)) + (((r283_z / $(24)) % $(12)) * $(24)) + (r283_z % $(24))), dtype=tl.float32) - r283_dot_grad) * tl.load($(candidate_vsoftmax_6_ptr) + ((((r283_z / $(288)) % $(8)) * $(288)) + (((r283_z / $(24)) % $(12)) * $(24)) + (r283_z % $(24))), dtype=tl.float32)
    r283_value = r283_value.to(tl.float32)
    r283_value = tl.broadcast_to(r283_value, [$(2304)])
    tl.store($(candidate_vsoftmax_backward_data_5_ptr) + r283_z, r283_value)
    r284_z = tl.arange($(2304))
    r284_value = tl.where(tl.load($(candidate_vgt_6_ptr) + ((((r284_z / $(24)) % $(12)) * $(24)) + (r284_z % $(24))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_5_ptr) + ((((r284_z / $(288)) % $(8)) * $(288)) + (((r284_z / $(24)) % $(12)) * $(24)) + (r284_z % $(24))), dtype=tl.float32))
    r284_value = r284_value.to(tl.float32)
    r284_value = tl.broadcast_to(r284_value, [$(2304)])
    tl.store($(candidate_vmasked_fill_17_ptr) + r284_z, r284_value)
    r285_z = tl.arange($(2304))
    r285_value = (tl.load($(candidate_vmasked_fill_17_ptr) + ((((r285_z / $(288)) % $(8)) * $(288)) + (((r285_z / $(24)) % $(12)) * $(24)) + (r285_z % $(24))), dtype=tl.float32) * 0.07216878364870322)
    r285_value = r285_value.to(tl.float32)
    r285_value = tl.broadcast_to(r285_value, [$(2304)])
    tl.store($(candidate_vmul_40_ptr) + r285_z, r285_value)
    r286_row = tl.arange($(192))
    r286_col = tl.arange($(24))
    r286_lane = tl.arange($(12))
    for r286_group in range($(8)) {
      r286_left = tl.load($(candidate_vto_copy_48_ptr) + (((((((r286_group % $(8)) * $(2304)) + (r286_lane[None, :] * $(192)) + r286_row[:, None]) / $(2304)) % $(8)) * $(2304)) + ((((((r286_group % $(8)) * $(2304)) + (r286_lane[None, :] * $(192)) + r286_row[:, None]) / $(192)) % $(12)) * $(192)) + ((((r286_group % $(8)) * $(2304)) + (r286_lane[None, :] * $(192)) + r286_row[:, None]) % $(192))), dtype=tl.float32)
      r286_right = tl.load($(candidate_vmul_40_ptr) + (((((((r286_group % $(8)) * $(288)) + (r286_lane[:, None] * $(24)) + r286_col[None, :]) / $(288)) % $(8)) * $(288)) + ((((((r286_group % $(8)) * $(288)) + (r286_lane[:, None] * $(24)) + r286_col[None, :]) / $(24)) % $(12)) * $(24)) + ((((r286_group % $(8)) * $(288)) + (r286_lane[:, None] * $(24)) + r286_col[None, :]) % $(24))), dtype=tl.float32)
      r286_value = tl.dot(r286_left, r286_right)
      r286_value = r286_value.to(tl.float32)
      tl.store($(candidate_vbmm_53_ptr) + ((r286_group * $(4608)) + (r286_row[:, None] * $(24)) + r286_col[None, :]), r286_value)
    }
    r287_z = tl.arange($(131072))
    r287_value = tl.where(((((r287_z / $(1024)) % $(128)) >= $(0)) & (((r287_z / $(1024)) % $(128)) < $(24)) & ((((r287_z / $(1024)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vbmm_51_ptr) + ((((((((r287_z / $(128)) % $(8)) * $(3072)) + ((((r287_z / $(1024)) % $(128)) - $(0)) * $(128)) + (r287_z % $(128))) / $(3072)) % $(8)) * $(3072)) + ((((((r287_z / $(128)) % $(8)) * $(24)) + (((r287_z / $(1024)) % $(128)) - $(0))) % $(24)) * $(128)) + (r287_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r287_value = r287_value.to(tl.bfloat16)
    r287_value = tl.broadcast_to(r287_value, [$(131072)])
    tl.store($(candidate_vslice_backward_12_ptr) + r287_z, r287_value)
    r288_z = tl.arange($(131072))
    r288_value = (tl.load($(candidate_vadd_25_ptr) + ((((r288_z / $(1024)) % $(128)) * $(1024)) + (((r288_z / $(128)) % $(8)) * $(128)) + (r288_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_12_ptr) + ((((r288_z / $(1024)) % $(128)) * $(1024)) + (((r288_z / $(128)) % $(8)) * $(128)) + (r288_z % $(128))), dtype=tl.bfloat16))
    r288_value = r288_value.to(tl.bfloat16)
    r288_value = tl.broadcast_to(r288_value, [$(131072)])
    tl.store($(candidate_vadd_27_ptr) + r288_z, r288_value)
    r289_z = tl.arange($(36864))
    r289_value = tl.load($(candidate_vbmm_53_ptr) + ((((r289_z / $(4608)) % $(8)) * $(4608)) + ((r289_z % $(192)) * $(24)) + ((r289_z / $(192)) % $(24))), dtype=tl.float32)
    r289_value = r289_value.to(tl.bfloat16)
    r289_value = tl.broadcast_to(r289_value, [$(36864)])
    tl.store($(candidate_vto_copy_77_ptr) + r289_z, r289_value)
    r290_z = tl.arange($(196608))
    r290_value = tl.where(((((r290_z / $(1536)) % $(128)) >= $(0)) & (((r290_z / $(1536)) % $(128)) < $(24)) & ((((r290_z / $(1536)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vto_copy_77_ptr) + ((((r290_z / $(192)) % $(8)) * $(4608)) + (((((r290_z / $(1536)) % $(128)) - $(0)) % $(24)) * $(192)) + (r290_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r290_value = r290_value.to(tl.bfloat16)
    r290_value = tl.broadcast_to(r290_value, [$(196608)])
    tl.store($(candidate_vslice_backward_13_ptr) + r290_z, r290_value)
    r291_z = tl.arange($(196608))
    r291_value = (tl.load($(candidate_vadd_26_ptr) + ((((r291_z / $(1536)) % $(128)) * $(1536)) + (((r291_z / $(192)) % $(8)) * $(192)) + (r291_z % $(192))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_13_ptr) + ((((r291_z / $(1536)) % $(128)) * $(1536)) + (((r291_z / $(192)) % $(8)) * $(192)) + (r291_z % $(192))), dtype=tl.bfloat16))
    r291_value = r291_value.to(tl.bfloat16)
    r291_value = tl.broadcast_to(r291_value, [$(196608)])
    tl.store($(candidate_vadd_28_ptr) + r291_z, r291_value)
    r292_z = tl.arange($(24576))
    r292_value = tl.cast(0.0, tl.bfloat16)
    r292_value = r292_value.to(tl.bfloat16)
    r292_value = tl.broadcast_to(r292_value, [$(24576)])
    tl.store($(candidate_vfill_6_ptr) + r292_z, r292_value)
    r293_z = tl.arange($(131072))
    r293_value = tl.where(((((r293_z / $(1024)) % $(128)) >= $(24)) & (((r293_z / $(1024)) % $(128)) < $(48)) & ((((r293_z / $(1024)) % $(128)) - $(24)) % $(1) == $(0))), tl.load($(candidate_vfill_6_ptr) + ((((((r293_z / $(1024)) % $(128)) - $(24)) % $(24)) * $(1024)) + (((r293_z / $(128)) % $(8)) * $(128)) + (r293_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vadd_27_ptr) + ((((r293_z / $(1024)) % $(128)) * $(1024)) + (((r293_z / $(128)) % $(8)) * $(128)) + (r293_z % $(128))), dtype=tl.bfloat16))
    r293_value = r293_value.to(tl.bfloat16)
    r293_value = tl.broadcast_to(r293_value, [$(131072)])
    tl.store($(candidate_vslice_scatter_6_ptr) + r293_z, r293_value)
    r294_z = tl.arange($(8192))
    r294_value = tl.cast(0.0, tl.bfloat16)
    r294_value = r294_value.to(tl.bfloat16)
    r294_value = tl.broadcast_to(r294_value, [$(8192)])
    tl.store($(candidate_vfill_7_ptr) + r294_z, r294_value)
    r295_z = tl.arange($(131072))
    r295_value = tl.where(((((r295_z / $(1024)) % $(128)) >= $(56)) & (((r295_z / $(1024)) % $(128)) < $(64)) & ((((r295_z / $(1024)) % $(128)) - $(56)) % $(1) == $(0))), tl.load($(candidate_vfill_7_ptr) + ((((((r295_z / $(1024)) % $(128)) - $(56)) % $(8)) * $(1024)) + (((r295_z / $(128)) % $(8)) * $(128)) + (r295_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vslice_scatter_6_ptr) + ((((r295_z / $(1024)) % $(128)) * $(1024)) + (((r295_z / $(128)) % $(8)) * $(128)) + (r295_z % $(128))), dtype=tl.bfloat16))
    r295_value = r295_value.to(tl.bfloat16)
    r295_value = tl.broadcast_to(r295_value, [$(131072)])
    tl.store($(candidate_vslice_scatter_7_ptr) + r295_z, r295_value)
    r296_z = tl.arange($(32768))
    r296_value = tl.cast(0.0, tl.bfloat16)
    r296_value = r296_value.to(tl.bfloat16)
    r296_value = tl.broadcast_to(r296_value, [$(32768)])
    tl.store($(candidate_vfill_8_ptr) + r296_z, r296_value)
    r297_z = tl.arange($(131072))
    r297_value = tl.where(((((r297_z / $(1024)) % $(128)) >= $(96)) & (((r297_z / $(1024)) % $(128)) < $(128)) & ((((r297_z / $(1024)) % $(128)) - $(96)) % $(1) == $(0))), tl.load($(candidate_vfill_8_ptr) + ((((((r297_z / $(1024)) % $(128)) - $(96)) % $(32)) * $(1024)) + (((r297_z / $(128)) % $(8)) * $(128)) + (r297_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vslice_scatter_7_ptr) + ((((r297_z / $(1024)) % $(128)) * $(1024)) + (((r297_z / $(128)) % $(8)) * $(128)) + (r297_z % $(128))), dtype=tl.bfloat16))
    r297_value = r297_value.to(tl.bfloat16)
    r297_value = tl.broadcast_to(r297_value, [$(131072)])
    tl.store($(candidate_vslice_scatter_8_ptr) + r297_z, r297_value)
    r298_z = tl.arange($(36864))
    r298_value = tl.cast(0.0, tl.bfloat16)
    r298_value = r298_value.to(tl.bfloat16)
    r298_value = tl.broadcast_to(r298_value, [$(36864)])
    tl.store($(candidate_vfill_9_ptr) + r298_z, r298_value)
    r299_z = tl.arange($(196608))
    r299_value = tl.where(((((r299_z / $(1536)) % $(128)) >= $(24)) & (((r299_z / $(1536)) % $(128)) < $(48)) & ((((r299_z / $(1536)) % $(128)) - $(24)) % $(1) == $(0))), tl.load($(candidate_vfill_9_ptr) + ((((((r299_z / $(1536)) % $(128)) - $(24)) % $(24)) * $(1536)) + (((r299_z / $(192)) % $(8)) * $(192)) + (r299_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vadd_28_ptr) + ((((r299_z / $(1536)) % $(128)) * $(1536)) + (((r299_z / $(192)) % $(8)) * $(192)) + (r299_z % $(192))), dtype=tl.bfloat16))
    r299_value = r299_value.to(tl.bfloat16)
    r299_value = tl.broadcast_to(r299_value, [$(196608)])
    tl.store($(candidate_vslice_scatter_9_ptr) + r299_z, r299_value)
    r300_z = tl.arange($(12288))
    r300_value = tl.cast(0.0, tl.bfloat16)
    r300_value = r300_value.to(tl.bfloat16)
    r300_value = tl.broadcast_to(r300_value, [$(12288)])
    tl.store($(candidate_vfill_10_ptr) + r300_z, r300_value)
    r301_z = tl.arange($(196608))
    r301_value = tl.where(((((r301_z / $(1536)) % $(128)) >= $(56)) & (((r301_z / $(1536)) % $(128)) < $(64)) & ((((r301_z / $(1536)) % $(128)) - $(56)) % $(1) == $(0))), tl.load($(candidate_vfill_10_ptr) + ((((((r301_z / $(1536)) % $(128)) - $(56)) % $(8)) * $(1536)) + (((r301_z / $(192)) % $(8)) * $(192)) + (r301_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vslice_scatter_9_ptr) + ((((r301_z / $(1536)) % $(128)) * $(1536)) + (((r301_z / $(192)) % $(8)) * $(192)) + (r301_z % $(192))), dtype=tl.bfloat16))
    r301_value = r301_value.to(tl.bfloat16)
    r301_value = tl.broadcast_to(r301_value, [$(196608)])
    tl.store($(candidate_vslice_scatter_10_ptr) + r301_z, r301_value)
    r302_z = tl.arange($(49152))
    r302_value = tl.cast(0.0, tl.bfloat16)
    r302_value = r302_value.to(tl.bfloat16)
    r302_value = tl.broadcast_to(r302_value, [$(49152)])
    tl.store($(candidate_vfill_11_ptr) + r302_z, r302_value)
    r303_z = tl.arange($(196608))
    r303_value = tl.where(((((r303_z / $(1536)) % $(128)) >= $(96)) & (((r303_z / $(1536)) % $(128)) < $(128)) & ((((r303_z / $(1536)) % $(128)) - $(96)) % $(1) == $(0))), tl.load($(candidate_vfill_11_ptr) + ((((((r303_z / $(1536)) % $(128)) - $(96)) % $(32)) * $(1536)) + (((r303_z / $(192)) % $(8)) * $(192)) + (r303_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vslice_scatter_10_ptr) + ((((r303_z / $(1536)) % $(128)) * $(1536)) + (((r303_z / $(192)) % $(8)) * $(192)) + (r303_z % $(192))), dtype=tl.bfloat16))
    r303_value = r303_value.to(tl.bfloat16)
    r303_value = tl.broadcast_to(r303_value, [$(196608)])
    tl.store($(candidate_vslice_scatter_11_ptr) + r303_z, r303_value)
    r304_row = tl.arange($(256))
    r304_col = tl.arange($(1024))
    r304_lane = tl.arange($(128))
    for r304_group in range($(1)) {
      r304_left = tl.load($(candidate_vindex_2_ptr) + ((((((r304_lane[None, :] * $(256)) + r304_row[:, None]) / $(256)) % $(128)) * $(320)) + (((r304_lane[None, :] * $(256)) + r304_row[:, None]) % $(256))), dtype=tl.bfloat16)
      r304_right = tl.load($(candidate_vslice_scatter_8_ptr) + ((((((r304_lane[:, None] * $(1024)) + r304_col[None, :]) / $(1024)) % $(128)) * $(1024)) + (((((r304_lane[:, None] * $(1024)) + r304_col[None, :]) / $(128)) % $(8)) * $(128)) + (((r304_lane[:, None] * $(1024)) + r304_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r304_value = tl.dot(r304_left, r304_right)
      r304_value = r304_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_54_ptr) + ((r304_group * $(262144)) + (r304_row[:, None] * $(1024)) + r304_col[None, :]), r304_value)
    }
    r305_row = tl.arange($(256))
    r305_col = tl.arange($(1024))
    r305_lane = tl.arange($(128))
    for r305_group in range($(1)) {
      r305_left = tl.load($(candidate_vindex_2_ptr) + ((((((r305_lane[None, :] * $(256)) + r305_row[:, None]) / $(256)) % $(128)) * $(320)) + (((r305_lane[None, :] * $(256)) + r305_row[:, None]) % $(256))), dtype=tl.bfloat16)
      r305_right = tl.load($(candidate_vslice_scatter_11_ptr) + ((((((r305_lane[:, None] * $(1024)) + r305_col[None, :]) / $(1024)) % $(128)) * $(1536)) + (((((r305_lane[:, None] * $(1024)) + r305_col[None, :]) / $(128)) % $(8)) * $(192)) + (((r305_lane[:, None] * $(1024)) + r305_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r305_value = tl.dot(r305_left, r305_right)
      r305_value = r305_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_55_ptr) + ((r305_group * $(262144)) + (r305_row[:, None] * $(1024)) + r305_col[None, :]), r305_value)
    }
    r306_z = tl.arange($(524288))
    r306_value = tl.where(((((r306_z / $(256)) % $(256)) >= $(128)) & (((r306_z / $(256)) % $(256)) < $(256)) & ((((r306_z / $(256)) % $(256)) - $(128)) % $(1) == $(0))), tl.load($(candidate_vbmm_54_ptr) + (((r306_z % $(256)) * $(1024)) + (((((((r306_z / $(65536)) % $(8)) * $(32768)) + ((((r306_z / $(256)) % $(256)) - $(128)) * $(256)) + (r306_z % $(256))) / $(32768)) % $(8)) * $(128)) + (((((r306_z / $(65536)) % $(8)) * $(128)) + (((r306_z / $(256)) % $(256)) - $(128))) % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r306_value = r306_value.to(tl.bfloat16)
    r306_value = tl.broadcast_to(r306_value, [$(524288)])
    tl.store($(candidate_vslice_backward_14_ptr) + r306_z, r306_value)
    r307_z = tl.arange($(524288))
    r307_value = tl.where(((((r307_z / $(256)) % $(256)) >= $(0)) & (((r307_z / $(256)) % $(256)) < $(128)) & ((((r307_z / $(256)) % $(256)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vbmm_55_ptr) + (((r307_z % $(256)) * $(1024)) + (((((((r307_z / $(65536)) % $(8)) * $(32768)) + ((((r307_z / $(256)) % $(256)) - $(0)) * $(256)) + (r307_z % $(256))) / $(32768)) % $(8)) * $(128)) + (((((r307_z / $(65536)) % $(8)) * $(128)) + (((r307_z / $(256)) % $(256)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r307_value = r307_value.to(tl.bfloat16)
    r307_value = tl.broadcast_to(r307_value, [$(524288)])
    tl.store($(candidate_vslice_backward_15_ptr) + r307_z, r307_value)
    r308_z = tl.arange($(524288))
    r308_value = (tl.load($(candidate_vslice_backward_14_ptr) + ((((r308_z / $(65536)) % $(8)) * $(65536)) + (((r308_z / $(256)) % $(256)) * $(256)) + (r308_z % $(256))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_15_ptr) + ((((r308_z / $(65536)) % $(8)) * $(65536)) + (((r308_z / $(256)) % $(256)) * $(256)) + (r308_z % $(256))), dtype=tl.bfloat16))
    r308_value = r308_value.to(tl.bfloat16)
    r308_value = tl.broadcast_to(r308_value, [$(524288)])
    tl.store($(candidate_vadd_29_ptr) + r308_z, r308_value)
    r309_z = tl.arange($(524288))
    r309_value = (tl.load($(candidate_vadd_24_ptr) + ((((((((r309_z / $(256)) % $(2048)) * $(256)) + (r309_z % $(256))) / $(65536)) % $(8)) * $(65536)) + ((((r309_z / $(256)) % $(2048)) % $(256)) * $(256)) + (r309_z % $(256))), dtype=tl.bfloat16) + tl.load($(candidate_vadd_29_ptr) + ((((((((r309_z / $(256)) % $(2048)) * $(256)) + (r309_z % $(256))) / $(65536)) % $(8)) * $(65536)) + ((((r309_z / $(256)) % $(2048)) % $(256)) * $(256)) + (r309_z % $(256))), dtype=tl.bfloat16))
    r309_value = r309_value.to(tl.bfloat16)
    r309_value = tl.broadcast_to(r309_value, [$(524288)])
    tl.store($(candidate_vadd_30_ptr) + r309_z, r309_value)
    r310_row = tl.arange($(64))
    r310_col = tl.arange($(128))
    r310_lane = tl.arange($(16))
    for r310_group in range($(8)) {
      r310_left = tl.load($(candidate_vto_copy_47_ptr) + (((((((r310_group % $(8)) * $(1024)) + (r310_lane[None, :] * $(64)) + r310_row[:, None]) / $(1024)) % $(8)) * $(1024)) + ((((((r310_group % $(8)) * $(1024)) + (r310_lane[None, :] * $(64)) + r310_row[:, None]) / $(64)) % $(16)) * $(64)) + ((((r310_group % $(8)) * $(1024)) + (r310_lane[None, :] * $(64)) + r310_row[:, None]) % $(64))), dtype=tl.bfloat16)
      r310_right = tl.load($(candidate_vmm_4_ptr) + (((((((r310_group % $(8)) * $(2048)) + (r310_lane[:, None] * $(128)) + r310_col[None, :]) / $(128)) % $(16)) * $(1024)) + ((((((r310_group % $(8)) * $(2048)) + (r310_lane[:, None] * $(128)) + r310_col[None, :]) / $(2048)) % $(8)) * $(128)) + ((((r310_group % $(8)) * $(2048)) + (r310_lane[:, None] * $(128)) + r310_col[None, :]) % $(128)) + $(49152)), dtype=tl.bfloat16)
      r310_value = tl.dot(r310_left, r310_right)
      r310_value = r310_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_56_ptr) + ((r310_group * $(8192)) + (r310_row[:, None] * $(128)) + r310_col[None, :]), r310_value)
    }
    r311_row = tl.arange($(16))
    r311_col = tl.arange($(64))
    r311_lane = tl.arange($(128))
    for r311_group in range($(8)) {
      r311_left = tl.load($(candidate_vmm_4_ptr) + (((((((r311_group % $(8)) * $(2048)) + (r311_row[:, None] * $(128)) + r311_lane[None, :]) / $(128)) % $(16)) * $(1024)) + ((((((r311_group % $(8)) * $(2048)) + (r311_row[:, None] * $(128)) + r311_lane[None, :]) / $(2048)) % $(8)) * $(128)) + ((((r311_group % $(8)) * $(2048)) + (r311_row[:, None] * $(128)) + r311_lane[None, :]) % $(128)) + $(49152)), dtype=tl.bfloat16)
      r311_right = tl.load($(candidate_vbmm_11_ptr) + (((((((r311_group % $(8)) * $(8192)) + (r311_col[None, :] * $(128)) + r311_lane[:, None]) / $(128)) % $(64)) * $(1024)) + ((((((r311_group % $(8)) * $(8192)) + (r311_col[None, :] * $(128)) + r311_lane[:, None]) / $(8192)) % $(8)) * $(128)) + ((((r311_group % $(8)) * $(8192)) + (r311_col[None, :] * $(128)) + r311_lane[:, None]) % $(128)) + $(65536)), dtype=tl.bfloat16)
      r311_value = tl.dot(r311_left, r311_right)
      r311_value = r311_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_57_ptr) + ((r311_group * $(1024)) + (r311_row[:, None] * $(64)) + r311_col[None, :]), r311_value)
    }
    r312_z = tl.arange($(8192))
    r312_value = tl.load($(candidate_vbmm_57_ptr) + ((((r312_z / $(1024)) % $(8)) * $(1024)) + (((r312_z / $(64)) % $(16)) * $(64)) + (r312_z % $(64))), dtype=tl.bfloat16)
    r312_value = r312_value.to(tl.float32)
    r312_value = tl.broadcast_to(r312_value, [$(8192)])
    tl.store($(candidate_vto_copy_78_ptr) + r312_z, r312_value)
    r313_z = tl.arange($(8192))
    r313_lane = tl.arange($(64))
    r313_row_values = tl.load($(candidate_vsoftmax_5_ptr) + ((((r313_z[:, None] / $(1024)) % $(8)) * $(1024)) + (((r313_z[:, None] / $(64)) % $(16)) * $(64)) + (r313_lane[None, :] % $(64))), dtype=tl.float32)
    r313_row_grad = tl.load($(candidate_vto_copy_78_ptr) + ((((r313_z[:, None] / $(1024)) % $(8)) * $(1024)) + (((r313_z[:, None] / $(64)) % $(16)) * $(64)) + (r313_lane[None, :] % $(64))), dtype=tl.float32)
    r313_weighted = r313_row_grad * r313_row_values
    r313_dot_grad = tl.sum(r313_weighted, axis=1)
    r313_value = (tl.load($(candidate_vto_copy_78_ptr) + ((((r313_z / $(1024)) % $(8)) * $(1024)) + (((r313_z / $(64)) % $(16)) * $(64)) + (r313_z % $(64))), dtype=tl.float32) - r313_dot_grad) * tl.load($(candidate_vsoftmax_5_ptr) + ((((r313_z / $(1024)) % $(8)) * $(1024)) + (((r313_z / $(64)) % $(16)) * $(64)) + (r313_z % $(64))), dtype=tl.float32)
    r313_value = r313_value.to(tl.float32)
    r313_value = tl.broadcast_to(r313_value, [$(8192)])
    tl.store($(candidate_vsoftmax_backward_data_6_ptr) + r313_z, r313_value)
    r314_z = tl.arange($(8192))
    r314_value = tl.where(tl.load($(candidate_vgt_5_ptr) + ((((r314_z / $(64)) % $(16)) * $(64)) + (r314_z % $(64))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_6_ptr) + ((((r314_z / $(1024)) % $(8)) * $(1024)) + (((r314_z / $(64)) % $(16)) * $(64)) + (r314_z % $(64))), dtype=tl.float32))
    r314_value = r314_value.to(tl.float32)
    r314_value = tl.broadcast_to(r314_value, [$(8192)])
    tl.store($(candidate_vmasked_fill_18_ptr) + r314_z, r314_value)
    r315_z = tl.arange($(8192))
    r315_value = (tl.load($(candidate_vmasked_fill_18_ptr) + ((((r315_z / $(1024)) % $(8)) * $(1024)) + (((r315_z / $(64)) % $(16)) * $(64)) + (r315_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r315_value = r315_value.to(tl.float32)
    r315_value = tl.broadcast_to(r315_value, [$(8192)])
    tl.store($(candidate_vmul_41_ptr) + r315_z, r315_value)
    r316_row = tl.arange($(192))
    r316_col = tl.arange($(64))
    r316_lane = tl.arange($(16))
    for r316_group in range($(8)) {
      r316_left = tl.load($(candidate_vto_copy_45_ptr) + (((((((r316_group % $(8)) * $(3072)) + (r316_lane[None, :] * $(192)) + r316_row[:, None]) / $(3072)) % $(8)) * $(3072)) + ((((((r316_group % $(8)) * $(3072)) + (r316_lane[None, :] * $(192)) + r316_row[:, None]) / $(192)) % $(16)) * $(192)) + ((((r316_group % $(8)) * $(3072)) + (r316_lane[None, :] * $(192)) + r316_row[:, None]) % $(192))), dtype=tl.float32)
      r316_right = tl.load($(candidate_vmul_41_ptr) + (((((((r316_group % $(8)) * $(1024)) + (r316_lane[:, None] * $(64)) + r316_col[None, :]) / $(1024)) % $(8)) * $(1024)) + ((((((r316_group % $(8)) * $(1024)) + (r316_lane[:, None] * $(64)) + r316_col[None, :]) / $(64)) % $(16)) * $(64)) + ((((r316_group % $(8)) * $(1024)) + (r316_lane[:, None] * $(64)) + r316_col[None, :]) % $(64))), dtype=tl.float32)
      r316_value = tl.dot(r316_left, r316_right)
      r316_value = r316_value.to(tl.float32)
      tl.store($(candidate_vbmm_58_ptr) + ((r316_group * $(12288)) + (r316_row[:, None] * $(64)) + r316_col[None, :]), r316_value)
    }
    r317_z = tl.arange($(131072))
    r317_value = tl.where(((((r317_z / $(1024)) % $(128)) >= $(64)) & (((r317_z / $(1024)) % $(128)) < $(128)) & ((((r317_z / $(1024)) % $(128)) - $(64)) % $(1) == $(0))), tl.load($(candidate_vbmm_56_ptr) + ((((((((r317_z / $(128)) % $(8)) * $(8192)) + ((((r317_z / $(1024)) % $(128)) - $(64)) * $(128)) + (r317_z % $(128))) / $(8192)) % $(8)) * $(8192)) + ((((((r317_z / $(128)) % $(8)) * $(64)) + (((r317_z / $(1024)) % $(128)) - $(64))) % $(64)) * $(128)) + (r317_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r317_value = r317_value.to(tl.bfloat16)
    r317_value = tl.broadcast_to(r317_value, [$(131072)])
    tl.store($(candidate_vslice_backward_16_ptr) + r317_z, r317_value)
    r318_z = tl.arange($(98304))
    r318_value = tl.load($(candidate_vbmm_58_ptr) + ((((r318_z / $(12288)) % $(8)) * $(12288)) + ((r318_z % $(192)) * $(64)) + ((r318_z / $(192)) % $(64))), dtype=tl.float32)
    r318_value = r318_value.to(tl.bfloat16)
    r318_value = tl.broadcast_to(r318_value, [$(98304)])
    tl.store($(candidate_vto_copy_79_ptr) + r318_z, r318_value)
    r319_z = tl.arange($(196608))
    r319_value = tl.where(((((r319_z / $(1536)) % $(128)) >= $(64)) & (((r319_z / $(1536)) % $(128)) < $(128)) & ((((r319_z / $(1536)) % $(128)) - $(64)) % $(1) == $(0))), tl.load($(candidate_vto_copy_79_ptr) + ((((r319_z / $(192)) % $(8)) * $(12288)) + (((((r319_z / $(1536)) % $(128)) - $(64)) % $(64)) * $(192)) + (r319_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r319_value = r319_value.to(tl.bfloat16)
    r319_value = tl.broadcast_to(r319_value, [$(196608)])
    tl.store($(candidate_vslice_backward_17_ptr) + r319_z, r319_value)
    r320_row = tl.arange($(16))
    r320_col = tl.arange($(128))
    r320_lane = tl.arange($(4))
    for r320_group in range($(8)) {
      r320_left = tl.load($(candidate_vto_copy_44_ptr) + (((((((r320_group % $(8)) * $(64)) + (r320_lane[None, :] * $(16)) + r320_row[:, None]) / $(64)) % $(8)) * $(64)) + ((((((r320_group % $(8)) * $(64)) + (r320_lane[None, :] * $(16)) + r320_row[:, None]) / $(16)) % $(4)) * $(16)) + ((((r320_group % $(8)) * $(64)) + (r320_lane[None, :] * $(16)) + r320_row[:, None]) % $(16))), dtype=tl.bfloat16)
      r320_right = tl.load($(candidate_vmm_4_ptr) + (((((((r320_group % $(8)) * $(512)) + (r320_lane[:, None] * $(128)) + r320_col[None, :]) / $(128)) % $(4)) * $(1024)) + ((((((r320_group % $(8)) * $(512)) + (r320_lane[:, None] * $(128)) + r320_col[None, :]) / $(512)) % $(8)) * $(128)) + ((((r320_group % $(8)) * $(512)) + (r320_lane[:, None] * $(128)) + r320_col[None, :]) % $(128)) + $(45056)), dtype=tl.bfloat16)
      r320_value = tl.dot(r320_left, r320_right)
      r320_value = r320_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_59_ptr) + ((r320_group * $(2048)) + (r320_row[:, None] * $(128)) + r320_col[None, :]), r320_value)
    }
    r321_row = tl.arange($(4))
    r321_col = tl.arange($(16))
    r321_lane = tl.arange($(128))
    for r321_group in range($(8)) {
      r321_left = tl.load($(candidate_vmm_4_ptr) + (((((((r321_group % $(8)) * $(512)) + (r321_row[:, None] * $(128)) + r321_lane[None, :]) / $(128)) % $(4)) * $(1024)) + ((((((r321_group % $(8)) * $(512)) + (r321_row[:, None] * $(128)) + r321_lane[None, :]) / $(512)) % $(8)) * $(128)) + ((((r321_group % $(8)) * $(512)) + (r321_row[:, None] * $(128)) + r321_lane[None, :]) % $(128)) + $(45056)), dtype=tl.bfloat16)
      r321_right = tl.load($(candidate_vbmm_11_ptr) + (((((((r321_group % $(8)) * $(2048)) + (r321_col[None, :] * $(128)) + r321_lane[:, None]) / $(128)) % $(16)) * $(1024)) + ((((((r321_group % $(8)) * $(2048)) + (r321_col[None, :] * $(128)) + r321_lane[:, None]) / $(2048)) % $(8)) * $(128)) + ((((r321_group % $(8)) * $(2048)) + (r321_col[None, :] * $(128)) + r321_lane[:, None]) % $(128)) + $(49152)), dtype=tl.bfloat16)
      r321_value = tl.dot(r321_left, r321_right)
      r321_value = r321_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_60_ptr) + ((r321_group * $(64)) + (r321_row[:, None] * $(16)) + r321_col[None, :]), r321_value)
    }
    r322_z = tl.arange($(512))
    r322_value = tl.load($(candidate_vbmm_60_ptr) + ((((r322_z / $(64)) % $(8)) * $(64)) + (((r322_z / $(16)) % $(4)) * $(16)) + (r322_z % $(16))), dtype=tl.bfloat16)
    r322_value = r322_value.to(tl.float32)
    r322_value = tl.broadcast_to(r322_value, [$(512)])
    tl.store($(candidate_vto_copy_80_ptr) + r322_z, r322_value)
    r323_z = tl.arange($(512))
    r323_lane = tl.arange($(16))
    r323_row_values = tl.load($(candidate_vsoftmax_4_ptr) + ((((r323_z[:, None] / $(64)) % $(8)) * $(64)) + (((r323_z[:, None] / $(16)) % $(4)) * $(16)) + (r323_lane[None, :] % $(16))), dtype=tl.float32)
    r323_row_grad = tl.load($(candidate_vto_copy_80_ptr) + ((((r323_z[:, None] / $(64)) % $(8)) * $(64)) + (((r323_z[:, None] / $(16)) % $(4)) * $(16)) + (r323_lane[None, :] % $(16))), dtype=tl.float32)
    r323_weighted = r323_row_grad * r323_row_values
    r323_dot_grad = tl.sum(r323_weighted, axis=1)
    r323_value = (tl.load($(candidate_vto_copy_80_ptr) + ((((r323_z / $(64)) % $(8)) * $(64)) + (((r323_z / $(16)) % $(4)) * $(16)) + (r323_z % $(16))), dtype=tl.float32) - r323_dot_grad) * tl.load($(candidate_vsoftmax_4_ptr) + ((((r323_z / $(64)) % $(8)) * $(64)) + (((r323_z / $(16)) % $(4)) * $(16)) + (r323_z % $(16))), dtype=tl.float32)
    r323_value = r323_value.to(tl.float32)
    r323_value = tl.broadcast_to(r323_value, [$(512)])
    tl.store($(candidate_vsoftmax_backward_data_7_ptr) + r323_z, r323_value)
    r324_z = tl.arange($(512))
    r324_value = tl.where(tl.load($(candidate_vgt_4_ptr) + ((((r324_z / $(16)) % $(4)) * $(16)) + (r324_z % $(16))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_7_ptr) + ((((r324_z / $(64)) % $(8)) * $(64)) + (((r324_z / $(16)) % $(4)) * $(16)) + (r324_z % $(16))), dtype=tl.float32))
    r324_value = r324_value.to(tl.float32)
    r324_value = tl.broadcast_to(r324_value, [$(512)])
    tl.store($(candidate_vmasked_fill_19_ptr) + r324_z, r324_value)
    r325_z = tl.arange($(512))
    r325_value = (tl.load($(candidate_vmasked_fill_19_ptr) + ((((r325_z / $(64)) % $(8)) * $(64)) + (((r325_z / $(16)) % $(4)) * $(16)) + (r325_z % $(16))), dtype=tl.float32) * 0.07216878364870322)
    r325_value = r325_value.to(tl.float32)
    r325_value = tl.broadcast_to(r325_value, [$(512)])
    tl.store($(candidate_vmul_42_ptr) + r325_z, r325_value)
    r326_row = tl.arange($(192))
    r326_col = tl.arange($(16))
    r326_lane = tl.arange($(4))
    for r326_group in range($(8)) {
      r326_left = tl.load($(candidate_vto_copy_42_ptr) + (((((((r326_group % $(8)) * $(768)) + (r326_lane[None, :] * $(192)) + r326_row[:, None]) / $(768)) % $(8)) * $(768)) + ((((((r326_group % $(8)) * $(768)) + (r326_lane[None, :] * $(192)) + r326_row[:, None]) / $(192)) % $(4)) * $(192)) + ((((r326_group % $(8)) * $(768)) + (r326_lane[None, :] * $(192)) + r326_row[:, None]) % $(192))), dtype=tl.float32)
      r326_right = tl.load($(candidate_vmul_42_ptr) + (((((((r326_group % $(8)) * $(64)) + (r326_lane[:, None] * $(16)) + r326_col[None, :]) / $(64)) % $(8)) * $(64)) + ((((((r326_group % $(8)) * $(64)) + (r326_lane[:, None] * $(16)) + r326_col[None, :]) / $(16)) % $(4)) * $(16)) + ((((r326_group % $(8)) * $(64)) + (r326_lane[:, None] * $(16)) + r326_col[None, :]) % $(16))), dtype=tl.float32)
      r326_value = tl.dot(r326_left, r326_right)
      r326_value = r326_value.to(tl.float32)
      tl.store($(candidate_vbmm_61_ptr) + ((r326_group * $(3072)) + (r326_row[:, None] * $(16)) + r326_col[None, :]), r326_value)
    }
    r327_z = tl.arange($(131072))
    r327_value = tl.where(((((r327_z / $(1024)) % $(128)) >= $(48)) & (((r327_z / $(1024)) % $(128)) < $(64)) & ((((r327_z / $(1024)) % $(128)) - $(48)) % $(1) == $(0))), tl.load($(candidate_vbmm_59_ptr) + ((((((((r327_z / $(128)) % $(8)) * $(2048)) + ((((r327_z / $(1024)) % $(128)) - $(48)) * $(128)) + (r327_z % $(128))) / $(2048)) % $(8)) * $(2048)) + ((((((r327_z / $(128)) % $(8)) * $(16)) + (((r327_z / $(1024)) % $(128)) - $(48))) % $(16)) * $(128)) + (r327_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r327_value = r327_value.to(tl.bfloat16)
    r327_value = tl.broadcast_to(r327_value, [$(131072)])
    tl.store($(candidate_vslice_backward_18_ptr) + r327_z, r327_value)
    r328_z = tl.arange($(131072))
    r328_value = (tl.load($(candidate_vslice_backward_16_ptr) + ((((r328_z / $(1024)) % $(128)) * $(1024)) + (((r328_z / $(128)) % $(8)) * $(128)) + (r328_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_18_ptr) + ((((r328_z / $(1024)) % $(128)) * $(1024)) + (((r328_z / $(128)) % $(8)) * $(128)) + (r328_z % $(128))), dtype=tl.bfloat16))
    r328_value = r328_value.to(tl.bfloat16)
    r328_value = tl.broadcast_to(r328_value, [$(131072)])
    tl.store($(candidate_vadd_31_ptr) + r328_z, r328_value)
    r329_z = tl.arange($(24576))
    r329_value = tl.load($(candidate_vbmm_61_ptr) + ((((r329_z / $(3072)) % $(8)) * $(3072)) + ((r329_z % $(192)) * $(16)) + ((r329_z / $(192)) % $(16))), dtype=tl.float32)
    r329_value = r329_value.to(tl.bfloat16)
    r329_value = tl.broadcast_to(r329_value, [$(24576)])
    tl.store($(candidate_vto_copy_81_ptr) + r329_z, r329_value)
    r330_z = tl.arange($(196608))
    r330_value = tl.where(((((r330_z / $(1536)) % $(128)) >= $(48)) & (((r330_z / $(1536)) % $(128)) < $(64)) & ((((r330_z / $(1536)) % $(128)) - $(48)) % $(1) == $(0))), tl.load($(candidate_vto_copy_81_ptr) + ((((r330_z / $(192)) % $(8)) * $(3072)) + (((((r330_z / $(1536)) % $(128)) - $(48)) % $(16)) * $(192)) + (r330_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r330_value = r330_value.to(tl.bfloat16)
    r330_value = tl.broadcast_to(r330_value, [$(196608)])
    tl.store($(candidate_vslice_backward_19_ptr) + r330_z, r330_value)
    r331_z = tl.arange($(196608))
    r331_value = (tl.load($(candidate_vslice_backward_17_ptr) + ((((r331_z / $(1536)) % $(128)) * $(1536)) + (((r331_z / $(192)) % $(8)) * $(192)) + (r331_z % $(192))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_19_ptr) + ((((r331_z / $(1536)) % $(128)) * $(1536)) + (((r331_z / $(192)) % $(8)) * $(192)) + (r331_z % $(192))), dtype=tl.bfloat16))
    r331_value = r331_value.to(tl.bfloat16)
    r331_value = tl.broadcast_to(r331_value, [$(196608)])
    tl.store($(candidate_vadd_32_ptr) + r331_z, r331_value)
    r332_row = tl.arange($(48))
    r332_col = tl.arange($(128))
    r332_lane = tl.arange($(12))
    for r332_group in range($(8)) {
      r332_left = tl.load($(candidate_vto_copy_41_ptr) + (((((((r332_group % $(8)) * $(576)) + (r332_lane[None, :] * $(48)) + r332_row[:, None]) / $(576)) % $(8)) * $(576)) + ((((((r332_group % $(8)) * $(576)) + (r332_lane[None, :] * $(48)) + r332_row[:, None]) / $(48)) % $(12)) * $(48)) + ((((r332_group % $(8)) * $(576)) + (r332_lane[None, :] * $(48)) + r332_row[:, None]) % $(48))), dtype=tl.bfloat16)
      r332_right = tl.load($(candidate_vmm_4_ptr) + (((((((r332_group % $(8)) * $(1536)) + (r332_lane[:, None] * $(128)) + r332_col[None, :]) / $(128)) % $(12)) * $(1024)) + ((((((r332_group % $(8)) * $(1536)) + (r332_lane[:, None] * $(128)) + r332_col[None, :]) / $(1536)) % $(8)) * $(128)) + ((((r332_group % $(8)) * $(1536)) + (r332_lane[:, None] * $(128)) + r332_col[None, :]) % $(128)) + $(32768)), dtype=tl.bfloat16)
      r332_value = tl.dot(r332_left, r332_right)
      r332_value = r332_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_62_ptr) + ((r332_group * $(6144)) + (r332_row[:, None] * $(128)) + r332_col[None, :]), r332_value)
    }
    r333_row = tl.arange($(12))
    r333_col = tl.arange($(48))
    r333_lane = tl.arange($(128))
    for r333_group in range($(8)) {
      r333_left = tl.load($(candidate_vmm_4_ptr) + (((((((r333_group % $(8)) * $(1536)) + (r333_row[:, None] * $(128)) + r333_lane[None, :]) / $(128)) % $(12)) * $(1024)) + ((((((r333_group % $(8)) * $(1536)) + (r333_row[:, None] * $(128)) + r333_lane[None, :]) / $(1536)) % $(8)) * $(128)) + ((((r333_group % $(8)) * $(1536)) + (r333_row[:, None] * $(128)) + r333_lane[None, :]) % $(128)) + $(32768)), dtype=tl.bfloat16)
      r333_right = tl.load($(candidate_vbmm_11_ptr) + (((((((r333_group % $(8)) * $(6144)) + (r333_col[None, :] * $(128)) + r333_lane[:, None]) / $(128)) % $(48)) * $(1024)) + ((((((r333_group % $(8)) * $(6144)) + (r333_col[None, :] * $(128)) + r333_lane[:, None]) / $(6144)) % $(8)) * $(128)) + ((((r333_group % $(8)) * $(6144)) + (r333_col[None, :] * $(128)) + r333_lane[:, None]) % $(128))), dtype=tl.bfloat16)
      r333_value = tl.dot(r333_left, r333_right)
      r333_value = r333_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_63_ptr) + ((r333_group * $(576)) + (r333_row[:, None] * $(48)) + r333_col[None, :]), r333_value)
    }
    r334_z = tl.arange($(4608))
    r334_value = tl.load($(candidate_vbmm_63_ptr) + ((((r334_z / $(576)) % $(8)) * $(576)) + (((r334_z / $(48)) % $(12)) * $(48)) + (r334_z % $(48))), dtype=tl.bfloat16)
    r334_value = r334_value.to(tl.float32)
    r334_value = tl.broadcast_to(r334_value, [$(4608)])
    tl.store($(candidate_vto_copy_82_ptr) + r334_z, r334_value)
    r335_z = tl.arange($(4608))
    r335_lane = tl.arange($(48))
    r335_row_values = tl.load($(candidate_vsoftmax_3_ptr) + ((((r335_z[:, None] / $(576)) % $(8)) * $(576)) + (((r335_z[:, None] / $(48)) % $(12)) * $(48)) + (r335_lane[None, :] % $(48))), dtype=tl.float32)
    r335_row_grad = tl.load($(candidate_vto_copy_82_ptr) + ((((r335_z[:, None] / $(576)) % $(8)) * $(576)) + (((r335_z[:, None] / $(48)) % $(12)) * $(48)) + (r335_lane[None, :] % $(48))), dtype=tl.float32)
    r335_weighted = r335_row_grad * r335_row_values
    r335_dot_grad = tl.sum(r335_weighted, axis=1)
    r335_value = (tl.load($(candidate_vto_copy_82_ptr) + ((((r335_z / $(576)) % $(8)) * $(576)) + (((r335_z / $(48)) % $(12)) * $(48)) + (r335_z % $(48))), dtype=tl.float32) - r335_dot_grad) * tl.load($(candidate_vsoftmax_3_ptr) + ((((r335_z / $(576)) % $(8)) * $(576)) + (((r335_z / $(48)) % $(12)) * $(48)) + (r335_z % $(48))), dtype=tl.float32)
    r335_value = r335_value.to(tl.float32)
    r335_value = tl.broadcast_to(r335_value, [$(4608)])
    tl.store($(candidate_vsoftmax_backward_data_8_ptr) + r335_z, r335_value)
    r336_z = tl.arange($(4608))
    r336_value = tl.where(tl.load($(candidate_vgt_3_ptr) + ((((r336_z / $(48)) % $(12)) * $(48)) + (r336_z % $(48))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_8_ptr) + ((((r336_z / $(576)) % $(8)) * $(576)) + (((r336_z / $(48)) % $(12)) * $(48)) + (r336_z % $(48))), dtype=tl.float32))
    r336_value = r336_value.to(tl.float32)
    r336_value = tl.broadcast_to(r336_value, [$(4608)])
    tl.store($(candidate_vmasked_fill_20_ptr) + r336_z, r336_value)
    r337_z = tl.arange($(4608))
    r337_value = (tl.load($(candidate_vmasked_fill_20_ptr) + ((((r337_z / $(576)) % $(8)) * $(576)) + (((r337_z / $(48)) % $(12)) * $(48)) + (r337_z % $(48))), dtype=tl.float32) * 0.07216878364870322)
    r337_value = r337_value.to(tl.float32)
    r337_value = tl.broadcast_to(r337_value, [$(4608)])
    tl.store($(candidate_vmul_43_ptr) + r337_z, r337_value)
    r338_row = tl.arange($(192))
    r338_col = tl.arange($(48))
    r338_lane = tl.arange($(12))
    for r338_group in range($(8)) {
      r338_left = tl.load($(candidate_vto_copy_39_ptr) + (((((((r338_group % $(8)) * $(2304)) + (r338_lane[None, :] * $(192)) + r338_row[:, None]) / $(2304)) % $(8)) * $(2304)) + ((((((r338_group % $(8)) * $(2304)) + (r338_lane[None, :] * $(192)) + r338_row[:, None]) / $(192)) % $(12)) * $(192)) + ((((r338_group % $(8)) * $(2304)) + (r338_lane[None, :] * $(192)) + r338_row[:, None]) % $(192))), dtype=tl.float32)
      r338_right = tl.load($(candidate_vmul_43_ptr) + (((((((r338_group % $(8)) * $(576)) + (r338_lane[:, None] * $(48)) + r338_col[None, :]) / $(576)) % $(8)) * $(576)) + ((((((r338_group % $(8)) * $(576)) + (r338_lane[:, None] * $(48)) + r338_col[None, :]) / $(48)) % $(12)) * $(48)) + ((((r338_group % $(8)) * $(576)) + (r338_lane[:, None] * $(48)) + r338_col[None, :]) % $(48))), dtype=tl.float32)
      r338_value = tl.dot(r338_left, r338_right)
      r338_value = r338_value.to(tl.float32)
      tl.store($(candidate_vbmm_64_ptr) + ((r338_group * $(9216)) + (r338_row[:, None] * $(48)) + r338_col[None, :]), r338_value)
    }
    r339_z = tl.arange($(131072))
    r339_value = tl.where(((((r339_z / $(1024)) % $(128)) >= $(0)) & (((r339_z / $(1024)) % $(128)) < $(48)) & ((((r339_z / $(1024)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vbmm_62_ptr) + ((((((((r339_z / $(128)) % $(8)) * $(6144)) + ((((r339_z / $(1024)) % $(128)) - $(0)) * $(128)) + (r339_z % $(128))) / $(6144)) % $(8)) * $(6144)) + ((((((r339_z / $(128)) % $(8)) * $(48)) + (((r339_z / $(1024)) % $(128)) - $(0))) % $(48)) * $(128)) + (r339_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r339_value = r339_value.to(tl.bfloat16)
    r339_value = tl.broadcast_to(r339_value, [$(131072)])
    tl.store($(candidate_vslice_backward_20_ptr) + r339_z, r339_value)
    r340_z = tl.arange($(131072))
    r340_value = (tl.load($(candidate_vadd_31_ptr) + ((((r340_z / $(1024)) % $(128)) * $(1024)) + (((r340_z / $(128)) % $(8)) * $(128)) + (r340_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_20_ptr) + ((((r340_z / $(1024)) % $(128)) * $(1024)) + (((r340_z / $(128)) % $(8)) * $(128)) + (r340_z % $(128))), dtype=tl.bfloat16))
    r340_value = r340_value.to(tl.bfloat16)
    r340_value = tl.broadcast_to(r340_value, [$(131072)])
    tl.store($(candidate_vadd_33_ptr) + r340_z, r340_value)
    r341_z = tl.arange($(73728))
    r341_value = tl.load($(candidate_vbmm_64_ptr) + ((((r341_z / $(9216)) % $(8)) * $(9216)) + ((r341_z % $(192)) * $(48)) + ((r341_z / $(192)) % $(48))), dtype=tl.float32)
    r341_value = r341_value.to(tl.bfloat16)
    r341_value = tl.broadcast_to(r341_value, [$(73728)])
    tl.store($(candidate_vto_copy_83_ptr) + r341_z, r341_value)
    r342_z = tl.arange($(196608))
    r342_value = tl.where(((((r342_z / $(1536)) % $(128)) >= $(0)) & (((r342_z / $(1536)) % $(128)) < $(48)) & ((((r342_z / $(1536)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vto_copy_83_ptr) + ((((r342_z / $(192)) % $(8)) * $(9216)) + (((((r342_z / $(1536)) % $(128)) - $(0)) % $(48)) * $(192)) + (r342_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r342_value = r342_value.to(tl.bfloat16)
    r342_value = tl.broadcast_to(r342_value, [$(196608)])
    tl.store($(candidate_vslice_backward_21_ptr) + r342_z, r342_value)
    r343_z = tl.arange($(196608))
    r343_value = (tl.load($(candidate_vadd_32_ptr) + ((((r343_z / $(1536)) % $(128)) * $(1536)) + (((r343_z / $(192)) % $(8)) * $(192)) + (r343_z % $(192))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_21_ptr) + ((((r343_z / $(1536)) % $(128)) * $(1536)) + (((r343_z / $(192)) % $(8)) * $(192)) + (r343_z % $(192))), dtype=tl.bfloat16))
    r343_value = r343_value.to(tl.bfloat16)
    r343_value = tl.broadcast_to(r343_value, [$(196608)])
    tl.store($(candidate_vadd_34_ptr) + r343_z, r343_value)
    r344_z = tl.arange($(0))
    r344_value = tl.cast(0.0, tl.bfloat16)
    r344_value = r344_value.to(tl.bfloat16)
    r344_value = tl.broadcast_to(r344_value, [$(0)])
    tl.store($(candidate_vfill_12_ptr) + r344_z, r344_value)
    r345_z = tl.arange($(131072))
    r345_value = tl.where(((((r345_z / $(1024)) % $(128)) >= $(48)) & (((r345_z / $(1024)) % $(128)) < $(48)) & ((((r345_z / $(1024)) % $(128)) - $(48)) % $(1) == $(0))), tl.load($(candidate_vfill_12_ptr) + ((((((r345_z / $(1024)) % $(128)) - $(48)) % $(0)) * $(1024)) + (((r345_z / $(128)) % $(8)) * $(128)) + (r345_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vadd_33_ptr) + ((((r345_z / $(1024)) % $(128)) * $(1024)) + (((r345_z / $(128)) % $(8)) * $(128)) + (r345_z % $(128))), dtype=tl.bfloat16))
    r345_value = r345_value.to(tl.bfloat16)
    r345_value = tl.broadcast_to(r345_value, [$(131072)])
    tl.store($(candidate_vslice_scatter_12_ptr) + r345_z, r345_value)
    r346_z = tl.arange($(0))
    r346_value = tl.cast(0.0, tl.bfloat16)
    r346_value = r346_value.to(tl.bfloat16)
    r346_value = tl.broadcast_to(r346_value, [$(0)])
    tl.store($(candidate_vfill_13_ptr) + r346_z, r346_value)
    r347_z = tl.arange($(131072))
    r347_value = tl.where(((((r347_z / $(1024)) % $(128)) >= $(64)) & (((r347_z / $(1024)) % $(128)) < $(64)) & ((((r347_z / $(1024)) % $(128)) - $(64)) % $(1) == $(0))), tl.load($(candidate_vfill_13_ptr) + ((((((r347_z / $(1024)) % $(128)) - $(64)) % $(0)) * $(1024)) + (((r347_z / $(128)) % $(8)) * $(128)) + (r347_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vslice_scatter_12_ptr) + ((((r347_z / $(1024)) % $(128)) * $(1024)) + (((r347_z / $(128)) % $(8)) * $(128)) + (r347_z % $(128))), dtype=tl.bfloat16))
    r347_value = r347_value.to(tl.bfloat16)
    r347_value = tl.broadcast_to(r347_value, [$(131072)])
    tl.store($(candidate_vslice_scatter_13_ptr) + r347_z, r347_value)
    r348_z = tl.arange($(0))
    r348_value = tl.cast(0.0, tl.bfloat16)
    r348_value = r348_value.to(tl.bfloat16)
    r348_value = tl.broadcast_to(r348_value, [$(0)])
    tl.store($(candidate_vfill_14_ptr) + r348_z, r348_value)
    r349_z = tl.arange($(131072))
    r349_value = tl.where(((((r349_z / $(1024)) % $(128)) >= $(128)) & (((r349_z / $(1024)) % $(128)) < $(128)) & ((((r349_z / $(1024)) % $(128)) - $(128)) % $(1) == $(0))), tl.load($(candidate_vfill_14_ptr) + ((((((r349_z / $(1024)) % $(128)) - $(128)) % $(0)) * $(1024)) + (((r349_z / $(128)) % $(8)) * $(128)) + (r349_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vslice_scatter_13_ptr) + ((((r349_z / $(1024)) % $(128)) * $(1024)) + (((r349_z / $(128)) % $(8)) * $(128)) + (r349_z % $(128))), dtype=tl.bfloat16))
    r349_value = r349_value.to(tl.bfloat16)
    r349_value = tl.broadcast_to(r349_value, [$(131072)])
    tl.store($(candidate_vslice_scatter_14_ptr) + r349_z, r349_value)
    r350_z = tl.arange($(0))
    r350_value = tl.cast(0.0, tl.bfloat16)
    r350_value = r350_value.to(tl.bfloat16)
    r350_value = tl.broadcast_to(r350_value, [$(0)])
    tl.store($(candidate_vfill_15_ptr) + r350_z, r350_value)
    r351_z = tl.arange($(196608))
    r351_value = tl.where(((((r351_z / $(1536)) % $(128)) >= $(48)) & (((r351_z / $(1536)) % $(128)) < $(48)) & ((((r351_z / $(1536)) % $(128)) - $(48)) % $(1) == $(0))), tl.load($(candidate_vfill_15_ptr) + ((((((r351_z / $(1536)) % $(128)) - $(48)) % $(0)) * $(1536)) + (((r351_z / $(192)) % $(8)) * $(192)) + (r351_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vadd_34_ptr) + ((((r351_z / $(1536)) % $(128)) * $(1536)) + (((r351_z / $(192)) % $(8)) * $(192)) + (r351_z % $(192))), dtype=tl.bfloat16))
    r351_value = r351_value.to(tl.bfloat16)
    r351_value = tl.broadcast_to(r351_value, [$(196608)])
    tl.store($(candidate_vslice_scatter_15_ptr) + r351_z, r351_value)
    r352_z = tl.arange($(0))
    r352_value = tl.cast(0.0, tl.bfloat16)
    r352_value = r352_value.to(tl.bfloat16)
    r352_value = tl.broadcast_to(r352_value, [$(0)])
    tl.store($(candidate_vfill_16_ptr) + r352_z, r352_value)
    r353_z = tl.arange($(196608))
    r353_value = tl.where(((((r353_z / $(1536)) % $(128)) >= $(64)) & (((r353_z / $(1536)) % $(128)) < $(64)) & ((((r353_z / $(1536)) % $(128)) - $(64)) % $(1) == $(0))), tl.load($(candidate_vfill_16_ptr) + ((((((r353_z / $(1536)) % $(128)) - $(64)) % $(0)) * $(1536)) + (((r353_z / $(192)) % $(8)) * $(192)) + (r353_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vslice_scatter_15_ptr) + ((((r353_z / $(1536)) % $(128)) * $(1536)) + (((r353_z / $(192)) % $(8)) * $(192)) + (r353_z % $(192))), dtype=tl.bfloat16))
    r353_value = r353_value.to(tl.bfloat16)
    r353_value = tl.broadcast_to(r353_value, [$(196608)])
    tl.store($(candidate_vslice_scatter_16_ptr) + r353_z, r353_value)
    r354_z = tl.arange($(0))
    r354_value = tl.cast(0.0, tl.bfloat16)
    r354_value = r354_value.to(tl.bfloat16)
    r354_value = tl.broadcast_to(r354_value, [$(0)])
    tl.store($(candidate_vfill_17_ptr) + r354_z, r354_value)
    r355_z = tl.arange($(196608))
    r355_value = tl.where(((((r355_z / $(1536)) % $(128)) >= $(128)) & (((r355_z / $(1536)) % $(128)) < $(128)) & ((((r355_z / $(1536)) % $(128)) - $(128)) % $(1) == $(0))), tl.load($(candidate_vfill_17_ptr) + ((((((r355_z / $(1536)) % $(128)) - $(128)) % $(0)) * $(1536)) + (((r355_z / $(192)) % $(8)) * $(192)) + (r355_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vslice_scatter_16_ptr) + ((((r355_z / $(1536)) % $(128)) * $(1536)) + (((r355_z / $(192)) % $(8)) * $(192)) + (r355_z % $(192))), dtype=tl.bfloat16))
    r355_value = r355_value.to(tl.bfloat16)
    r355_value = tl.broadcast_to(r355_value, [$(196608)])
    tl.store($(candidate_vslice_scatter_17_ptr) + r355_z, r355_value)
    r356_row = tl.arange($(256))
    r356_col = tl.arange($(1024))
    r356_lane = tl.arange($(128))
    for r356_group in range($(1)) {
      r356_left = tl.load($(candidate_vindex_2_ptr) + ((((((r356_lane[None, :] * $(256)) + r356_row[:, None]) / $(256)) % $(128)) * $(320)) + (((r356_lane[None, :] * $(256)) + r356_row[:, None]) % $(256))), dtype=tl.bfloat16)
      r356_right = tl.load($(candidate_vslice_scatter_14_ptr) + ((((((r356_lane[:, None] * $(1024)) + r356_col[None, :]) / $(1024)) % $(128)) * $(1024)) + (((((r356_lane[:, None] * $(1024)) + r356_col[None, :]) / $(128)) % $(8)) * $(128)) + (((r356_lane[:, None] * $(1024)) + r356_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r356_value = tl.dot(r356_left, r356_right)
      r356_value = r356_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_65_ptr) + ((r356_group * $(262144)) + (r356_row[:, None] * $(1024)) + r356_col[None, :]), r356_value)
    }
    r357_row = tl.arange($(256))
    r357_col = tl.arange($(1024))
    r357_lane = tl.arange($(128))
    for r357_group in range($(1)) {
      r357_left = tl.load($(candidate_vindex_2_ptr) + ((((((r357_lane[None, :] * $(256)) + r357_row[:, None]) / $(256)) % $(128)) * $(320)) + (((r357_lane[None, :] * $(256)) + r357_row[:, None]) % $(256))), dtype=tl.bfloat16)
      r357_right = tl.load($(candidate_vslice_scatter_17_ptr) + ((((((r357_lane[:, None] * $(1024)) + r357_col[None, :]) / $(1024)) % $(128)) * $(1536)) + (((((r357_lane[:, None] * $(1024)) + r357_col[None, :]) / $(128)) % $(8)) * $(192)) + (((r357_lane[:, None] * $(1024)) + r357_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r357_value = tl.dot(r357_left, r357_right)
      r357_value = r357_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_66_ptr) + ((r357_group * $(262144)) + (r357_row[:, None] * $(1024)) + r357_col[None, :]), r357_value)
    }
    r358_z = tl.arange($(524288))
    r358_value = tl.where(((((r358_z / $(256)) % $(256)) >= $(128)) & (((r358_z / $(256)) % $(256)) < $(256)) & ((((r358_z / $(256)) % $(256)) - $(128)) % $(1) == $(0))), tl.load($(candidate_vbmm_65_ptr) + (((r358_z % $(256)) * $(1024)) + (((((((r358_z / $(65536)) % $(8)) * $(32768)) + ((((r358_z / $(256)) % $(256)) - $(128)) * $(256)) + (r358_z % $(256))) / $(32768)) % $(8)) * $(128)) + (((((r358_z / $(65536)) % $(8)) * $(128)) + (((r358_z / $(256)) % $(256)) - $(128))) % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r358_value = r358_value.to(tl.bfloat16)
    r358_value = tl.broadcast_to(r358_value, [$(524288)])
    tl.store($(candidate_vslice_backward_22_ptr) + r358_z, r358_value)
    r359_z = tl.arange($(524288))
    r359_value = tl.where(((((r359_z / $(256)) % $(256)) >= $(0)) & (((r359_z / $(256)) % $(256)) < $(128)) & ((((r359_z / $(256)) % $(256)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vbmm_66_ptr) + (((r359_z % $(256)) * $(1024)) + (((((((r359_z / $(65536)) % $(8)) * $(32768)) + ((((r359_z / $(256)) % $(256)) - $(0)) * $(256)) + (r359_z % $(256))) / $(32768)) % $(8)) * $(128)) + (((((r359_z / $(65536)) % $(8)) * $(128)) + (((r359_z / $(256)) % $(256)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r359_value = r359_value.to(tl.bfloat16)
    r359_value = tl.broadcast_to(r359_value, [$(524288)])
    tl.store($(candidate_vslice_backward_23_ptr) + r359_z, r359_value)
    r360_z = tl.arange($(524288))
    r360_value = (tl.load($(candidate_vslice_backward_22_ptr) + ((((r360_z / $(65536)) % $(8)) * $(65536)) + (((r360_z / $(256)) % $(256)) * $(256)) + (r360_z % $(256))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_23_ptr) + ((((r360_z / $(65536)) % $(8)) * $(65536)) + (((r360_z / $(256)) % $(256)) * $(256)) + (r360_z % $(256))), dtype=tl.bfloat16))
    r360_value = r360_value.to(tl.bfloat16)
    r360_value = tl.broadcast_to(r360_value, [$(524288)])
    tl.store($(candidate_vadd_35_ptr) + r360_z, r360_value)
    r361_z = tl.arange($(524288))
    r361_value = (tl.load($(candidate_vadd_30_ptr) + ((((r361_z / $(256)) % $(2048)) * $(256)) + (r361_z % $(256))), dtype=tl.bfloat16) + tl.load($(candidate_vadd_35_ptr) + ((((((((r361_z / $(256)) % $(2048)) * $(256)) + (r361_z % $(256))) / $(65536)) % $(8)) * $(65536)) + ((((r361_z / $(256)) % $(2048)) % $(256)) * $(256)) + (r361_z % $(256))), dtype=tl.bfloat16))
    r361_value = r361_value.to(tl.bfloat16)
    r361_value = tl.broadcast_to(r361_value, [$(524288)])
    tl.store($(candidate_vadd_36_ptr) + r361_z, r361_value)
    r362_row = tl.arange($(16))
    r362_col = tl.arange($(128))
    r362_lane = tl.arange($(16))
    for r362_group in range($(8)) {
      r362_left = tl.load($(candidate_vto_copy_38_ptr) + (((((((r362_group % $(8)) * $(256)) + (r362_lane[None, :] * $(16)) + r362_row[:, None]) / $(256)) % $(8)) * $(256)) + ((((((r362_group % $(8)) * $(256)) + (r362_lane[None, :] * $(16)) + r362_row[:, None]) / $(16)) % $(16)) * $(16)) + ((((r362_group % $(8)) * $(256)) + (r362_lane[None, :] * $(16)) + r362_row[:, None]) % $(16))), dtype=tl.bfloat16)
      r362_right = tl.load($(candidate_vmm_4_ptr) + (((((((r362_group % $(8)) * $(2048)) + (r362_lane[:, None] * $(128)) + r362_col[None, :]) / $(128)) % $(16)) * $(1024)) + ((((((r362_group % $(8)) * $(2048)) + (r362_lane[:, None] * $(128)) + r362_col[None, :]) / $(2048)) % $(8)) * $(128)) + ((((r362_group % $(8)) * $(2048)) + (r362_lane[:, None] * $(128)) + r362_col[None, :]) % $(128)) + $(16384)), dtype=tl.bfloat16)
      r362_value = tl.dot(r362_left, r362_right)
      r362_value = r362_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_67_ptr) + ((r362_group * $(2048)) + (r362_row[:, None] * $(128)) + r362_col[None, :]), r362_value)
    }
    r363_row = tl.arange($(16))
    r363_col = tl.arange($(16))
    r363_lane = tl.arange($(128))
    for r363_group in range($(8)) {
      r363_left = tl.load($(candidate_vmm_4_ptr) + (((((((r363_group % $(8)) * $(2048)) + (r363_row[:, None] * $(128)) + r363_lane[None, :]) / $(128)) % $(16)) * $(1024)) + ((((((r363_group % $(8)) * $(2048)) + (r363_row[:, None] * $(128)) + r363_lane[None, :]) / $(2048)) % $(8)) * $(128)) + ((((r363_group % $(8)) * $(2048)) + (r363_row[:, None] * $(128)) + r363_lane[None, :]) % $(128)) + $(16384)), dtype=tl.bfloat16)
      r363_right = tl.load($(candidate_vbmm_3_ptr) + (((((((r363_group % $(8)) * $(2048)) + (r363_col[None, :] * $(128)) + r363_lane[:, None]) / $(128)) % $(16)) * $(1024)) + ((((((r363_group % $(8)) * $(2048)) + (r363_col[None, :] * $(128)) + r363_lane[:, None]) / $(2048)) % $(8)) * $(128)) + ((((r363_group % $(8)) * $(2048)) + (r363_col[None, :] * $(128)) + r363_lane[:, None]) % $(128)) + $(65536)), dtype=tl.bfloat16)
      r363_value = tl.dot(r363_left, r363_right)
      r363_value = r363_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_68_ptr) + ((r363_group * $(256)) + (r363_row[:, None] * $(16)) + r363_col[None, :]), r363_value)
    }
    r364_z = tl.arange($(2048))
    r364_value = tl.load($(candidate_vbmm_68_ptr) + ((((r364_z / $(256)) % $(8)) * $(256)) + (((r364_z / $(16)) % $(16)) * $(16)) + (r364_z % $(16))), dtype=tl.bfloat16)
    r364_value = r364_value.to(tl.float32)
    r364_value = tl.broadcast_to(r364_value, [$(2048)])
    tl.store($(candidate_vto_copy_84_ptr) + r364_z, r364_value)
    r365_z = tl.arange($(2048))
    r365_lane = tl.arange($(16))
    r365_row_values = tl.load($(candidate_vsoftmax_2_ptr) + ((((r365_z[:, None] / $(256)) % $(8)) * $(256)) + (((r365_z[:, None] / $(16)) % $(16)) * $(16)) + (r365_lane[None, :] % $(16))), dtype=tl.float32)
    r365_row_grad = tl.load($(candidate_vto_copy_84_ptr) + ((((r365_z[:, None] / $(256)) % $(8)) * $(256)) + (((r365_z[:, None] / $(16)) % $(16)) * $(16)) + (r365_lane[None, :] % $(16))), dtype=tl.float32)
    r365_weighted = r365_row_grad * r365_row_values
    r365_dot_grad = tl.sum(r365_weighted, axis=1)
    r365_value = (tl.load($(candidate_vto_copy_84_ptr) + ((((r365_z / $(256)) % $(8)) * $(256)) + (((r365_z / $(16)) % $(16)) * $(16)) + (r365_z % $(16))), dtype=tl.float32) - r365_dot_grad) * tl.load($(candidate_vsoftmax_2_ptr) + ((((r365_z / $(256)) % $(8)) * $(256)) + (((r365_z / $(16)) % $(16)) * $(16)) + (r365_z % $(16))), dtype=tl.float32)
    r365_value = r365_value.to(tl.float32)
    r365_value = tl.broadcast_to(r365_value, [$(2048)])
    tl.store($(candidate_vsoftmax_backward_data_9_ptr) + r365_z, r365_value)
    r366_z = tl.arange($(2048))
    r366_value = tl.where(tl.load($(candidate_vgt_2_ptr) + ((((r366_z / $(16)) % $(16)) * $(16)) + (r366_z % $(16))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_9_ptr) + ((((r366_z / $(256)) % $(8)) * $(256)) + (((r366_z / $(16)) % $(16)) * $(16)) + (r366_z % $(16))), dtype=tl.float32))
    r366_value = r366_value.to(tl.float32)
    r366_value = tl.broadcast_to(r366_value, [$(2048)])
    tl.store($(candidate_vmasked_fill_21_ptr) + r366_z, r366_value)
    r367_z = tl.arange($(2048))
    r367_value = (tl.load($(candidate_vmasked_fill_21_ptr) + ((((r367_z / $(256)) % $(8)) * $(256)) + (((r367_z / $(16)) % $(16)) * $(16)) + (r367_z % $(16))), dtype=tl.float32) * 0.07216878364870322)
    r367_value = r367_value.to(tl.float32)
    r367_value = tl.broadcast_to(r367_value, [$(2048)])
    tl.store($(candidate_vmul_44_ptr) + r367_z, r367_value)
    r368_row = tl.arange($(192))
    r368_col = tl.arange($(16))
    r368_lane = tl.arange($(16))
    for r368_group in range($(8)) {
      r368_left = tl.load($(candidate_vto_copy_36_ptr) + (((((((r368_group % $(8)) * $(3072)) + (r368_lane[None, :] * $(192)) + r368_row[:, None]) / $(3072)) % $(8)) * $(3072)) + ((((((r368_group % $(8)) * $(3072)) + (r368_lane[None, :] * $(192)) + r368_row[:, None]) / $(192)) % $(16)) * $(192)) + ((((r368_group % $(8)) * $(3072)) + (r368_lane[None, :] * $(192)) + r368_row[:, None]) % $(192))), dtype=tl.float32)
      r368_right = tl.load($(candidate_vmul_44_ptr) + (((((((r368_group % $(8)) * $(256)) + (r368_lane[:, None] * $(16)) + r368_col[None, :]) / $(256)) % $(8)) * $(256)) + ((((((r368_group % $(8)) * $(256)) + (r368_lane[:, None] * $(16)) + r368_col[None, :]) / $(16)) % $(16)) * $(16)) + ((((r368_group % $(8)) * $(256)) + (r368_lane[:, None] * $(16)) + r368_col[None, :]) % $(16))), dtype=tl.float32)
      r368_value = tl.dot(r368_left, r368_right)
      r368_value = r368_value.to(tl.float32)
      tl.store($(candidate_vbmm_69_ptr) + ((r368_group * $(3072)) + (r368_row[:, None] * $(16)) + r368_col[None, :]), r368_value)
    }
    r369_z = tl.arange($(131072))
    r369_value = tl.where(((((r369_z / $(1024)) % $(128)) >= $(64)) & (((r369_z / $(1024)) % $(128)) < $(80)) & ((((r369_z / $(1024)) % $(128)) - $(64)) % $(1) == $(0))), tl.load($(candidate_vbmm_67_ptr) + ((((((((r369_z / $(128)) % $(8)) * $(2048)) + ((((r369_z / $(1024)) % $(128)) - $(64)) * $(128)) + (r369_z % $(128))) / $(2048)) % $(8)) * $(2048)) + ((((((r369_z / $(128)) % $(8)) * $(16)) + (((r369_z / $(1024)) % $(128)) - $(64))) % $(16)) * $(128)) + (r369_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r369_value = r369_value.to(tl.bfloat16)
    r369_value = tl.broadcast_to(r369_value, [$(131072)])
    tl.store($(candidate_vslice_backward_24_ptr) + r369_z, r369_value)
    r370_z = tl.arange($(24576))
    r370_value = tl.load($(candidate_vbmm_69_ptr) + ((((r370_z / $(3072)) % $(8)) * $(3072)) + ((r370_z % $(192)) * $(16)) + ((r370_z / $(192)) % $(16))), dtype=tl.float32)
    r370_value = r370_value.to(tl.bfloat16)
    r370_value = tl.broadcast_to(r370_value, [$(24576)])
    tl.store($(candidate_vto_copy_85_ptr) + r370_z, r370_value)
    r371_z = tl.arange($(196608))
    r371_value = tl.where(((((r371_z / $(1536)) % $(128)) >= $(64)) & (((r371_z / $(1536)) % $(128)) < $(80)) & ((((r371_z / $(1536)) % $(128)) - $(64)) % $(1) == $(0))), tl.load($(candidate_vto_copy_85_ptr) + ((((r371_z / $(192)) % $(8)) * $(3072)) + (((((r371_z / $(1536)) % $(128)) - $(64)) % $(16)) * $(192)) + (r371_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r371_value = r371_value.to(tl.bfloat16)
    r371_value = tl.broadcast_to(r371_value, [$(196608)])
    tl.store($(candidate_vslice_backward_25_ptr) + r371_z, r371_value)
    r372_row = tl.arange($(4))
    r372_col = tl.arange($(128))
    r372_lane = tl.arange($(4))
    for r372_group in range($(8)) {
      r372_left = tl.load($(candidate_vto_copy_35_ptr) + (((((((r372_group % $(8)) * $(16)) + (r372_lane[None, :] * $(4)) + r372_row[:, None]) / $(16)) % $(8)) * $(16)) + ((((((r372_group % $(8)) * $(16)) + (r372_lane[None, :] * $(4)) + r372_row[:, None]) / $(4)) % $(4)) * $(4)) + ((((r372_group % $(8)) * $(16)) + (r372_lane[None, :] * $(4)) + r372_row[:, None]) % $(4))), dtype=tl.bfloat16)
      r372_right = tl.load($(candidate_vmm_4_ptr) + (((((((r372_group % $(8)) * $(512)) + (r372_lane[:, None] * $(128)) + r372_col[None, :]) / $(128)) % $(4)) * $(1024)) + ((((((r372_group % $(8)) * $(512)) + (r372_lane[:, None] * $(128)) + r372_col[None, :]) / $(512)) % $(8)) * $(128)) + ((((r372_group % $(8)) * $(512)) + (r372_lane[:, None] * $(128)) + r372_col[None, :]) % $(128)) + $(12288)), dtype=tl.bfloat16)
      r372_value = tl.dot(r372_left, r372_right)
      r372_value = r372_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_70_ptr) + ((r372_group * $(512)) + (r372_row[:, None] * $(128)) + r372_col[None, :]), r372_value)
    }
    r373_row = tl.arange($(4))
    r373_col = tl.arange($(4))
    r373_lane = tl.arange($(128))
    for r373_group in range($(8)) {
      r373_left = tl.load($(candidate_vmm_4_ptr) + (((((((r373_group % $(8)) * $(512)) + (r373_row[:, None] * $(128)) + r373_lane[None, :]) / $(128)) % $(4)) * $(1024)) + ((((((r373_group % $(8)) * $(512)) + (r373_row[:, None] * $(128)) + r373_lane[None, :]) / $(512)) % $(8)) * $(128)) + ((((r373_group % $(8)) * $(512)) + (r373_row[:, None] * $(128)) + r373_lane[None, :]) % $(128)) + $(12288)), dtype=tl.bfloat16)
      r373_right = tl.load($(candidate_vbmm_3_ptr) + (((((((r373_group % $(8)) * $(512)) + (r373_col[None, :] * $(128)) + r373_lane[:, None]) / $(128)) % $(4)) * $(1024)) + ((((((r373_group % $(8)) * $(512)) + (r373_col[None, :] * $(128)) + r373_lane[:, None]) / $(512)) % $(8)) * $(128)) + ((((r373_group % $(8)) * $(512)) + (r373_col[None, :] * $(128)) + r373_lane[:, None]) % $(128)) + $(49152)), dtype=tl.bfloat16)
      r373_value = tl.dot(r373_left, r373_right)
      r373_value = r373_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_71_ptr) + ((r373_group * $(16)) + (r373_row[:, None] * $(4)) + r373_col[None, :]), r373_value)
    }
    r374_z = tl.arange($(128))
    r374_value = tl.load($(candidate_vbmm_71_ptr) + ((((r374_z / $(16)) % $(8)) * $(16)) + (((r374_z / $(4)) % $(4)) * $(4)) + (r374_z % $(4))), dtype=tl.bfloat16)
    r374_value = r374_value.to(tl.float32)
    r374_value = tl.broadcast_to(r374_value, [$(128)])
    tl.store($(candidate_vto_copy_86_ptr) + r374_z, r374_value)
    r375_z = tl.arange($(128))
    r375_lane = tl.arange($(4))
    r375_row_values = tl.load($(candidate_vsoftmax_1_ptr) + ((((r375_z[:, None] / $(16)) % $(8)) * $(16)) + (((r375_z[:, None] / $(4)) % $(4)) * $(4)) + (r375_lane[None, :] % $(4))), dtype=tl.float32)
    r375_row_grad = tl.load($(candidate_vto_copy_86_ptr) + ((((r375_z[:, None] / $(16)) % $(8)) * $(16)) + (((r375_z[:, None] / $(4)) % $(4)) * $(4)) + (r375_lane[None, :] % $(4))), dtype=tl.float32)
    r375_weighted = r375_row_grad * r375_row_values
    r375_dot_grad = tl.sum(r375_weighted, axis=1)
    r375_value = (tl.load($(candidate_vto_copy_86_ptr) + ((((r375_z / $(16)) % $(8)) * $(16)) + (((r375_z / $(4)) % $(4)) * $(4)) + (r375_z % $(4))), dtype=tl.float32) - r375_dot_grad) * tl.load($(candidate_vsoftmax_1_ptr) + ((((r375_z / $(16)) % $(8)) * $(16)) + (((r375_z / $(4)) % $(4)) * $(4)) + (r375_z % $(4))), dtype=tl.float32)
    r375_value = r375_value.to(tl.float32)
    r375_value = tl.broadcast_to(r375_value, [$(128)])
    tl.store($(candidate_vsoftmax_backward_data_10_ptr) + r375_z, r375_value)
    r376_z = tl.arange($(128))
    r376_value = tl.where(tl.load($(candidate_vgt_1_ptr) + ((((r376_z / $(4)) % $(4)) * $(4)) + (r376_z % $(4))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_10_ptr) + ((((r376_z / $(16)) % $(8)) * $(16)) + (((r376_z / $(4)) % $(4)) * $(4)) + (r376_z % $(4))), dtype=tl.float32))
    r376_value = r376_value.to(tl.float32)
    r376_value = tl.broadcast_to(r376_value, [$(128)])
    tl.store($(candidate_vmasked_fill_22_ptr) + r376_z, r376_value)
    r377_z = tl.arange($(128))
    r377_value = (tl.load($(candidate_vmasked_fill_22_ptr) + ((((r377_z / $(16)) % $(8)) * $(16)) + (((r377_z / $(4)) % $(4)) * $(4)) + (r377_z % $(4))), dtype=tl.float32) * 0.07216878364870322)
    r377_value = r377_value.to(tl.float32)
    r377_value = tl.broadcast_to(r377_value, [$(128)])
    tl.store($(candidate_vmul_45_ptr) + r377_z, r377_value)
    r378_row = tl.arange($(192))
    r378_col = tl.arange($(4))
    r378_lane = tl.arange($(4))
    for r378_group in range($(8)) {
      r378_left = tl.load($(candidate_vto_copy_33_ptr) + (((((((r378_group % $(8)) * $(768)) + (r378_lane[None, :] * $(192)) + r378_row[:, None]) / $(768)) % $(8)) * $(768)) + ((((((r378_group % $(8)) * $(768)) + (r378_lane[None, :] * $(192)) + r378_row[:, None]) / $(192)) % $(4)) * $(192)) + ((((r378_group % $(8)) * $(768)) + (r378_lane[None, :] * $(192)) + r378_row[:, None]) % $(192))), dtype=tl.float32)
      r378_right = tl.load($(candidate_vmul_45_ptr) + (((((((r378_group % $(8)) * $(16)) + (r378_lane[:, None] * $(4)) + r378_col[None, :]) / $(16)) % $(8)) * $(16)) + ((((((r378_group % $(8)) * $(16)) + (r378_lane[:, None] * $(4)) + r378_col[None, :]) / $(4)) % $(4)) * $(4)) + ((((r378_group % $(8)) * $(16)) + (r378_lane[:, None] * $(4)) + r378_col[None, :]) % $(4))), dtype=tl.float32)
      r378_value = tl.dot(r378_left, r378_right)
      r378_value = r378_value.to(tl.float32)
      tl.store($(candidate_vbmm_72_ptr) + ((r378_group * $(768)) + (r378_row[:, None] * $(4)) + r378_col[None, :]), r378_value)
    }
    r379_z = tl.arange($(131072))
    r379_value = tl.where(((((r379_z / $(1024)) % $(128)) >= $(48)) & (((r379_z / $(1024)) % $(128)) < $(52)) & ((((r379_z / $(1024)) % $(128)) - $(48)) % $(1) == $(0))), tl.load($(candidate_vbmm_70_ptr) + ((((((((r379_z / $(128)) % $(8)) * $(512)) + ((((r379_z / $(1024)) % $(128)) - $(48)) * $(128)) + (r379_z % $(128))) / $(512)) % $(8)) * $(512)) + ((((((r379_z / $(128)) % $(8)) * $(4)) + (((r379_z / $(1024)) % $(128)) - $(48))) % $(4)) * $(128)) + (r379_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r379_value = r379_value.to(tl.bfloat16)
    r379_value = tl.broadcast_to(r379_value, [$(131072)])
    tl.store($(candidate_vslice_backward_26_ptr) + r379_z, r379_value)
    r380_z = tl.arange($(131072))
    r380_value = (tl.load($(candidate_vslice_backward_24_ptr) + ((((r380_z / $(1024)) % $(128)) * $(1024)) + (((r380_z / $(128)) % $(8)) * $(128)) + (r380_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_26_ptr) + ((((r380_z / $(1024)) % $(128)) * $(1024)) + (((r380_z / $(128)) % $(8)) * $(128)) + (r380_z % $(128))), dtype=tl.bfloat16))
    r380_value = r380_value.to(tl.bfloat16)
    r380_value = tl.broadcast_to(r380_value, [$(131072)])
    tl.store($(candidate_vadd_37_ptr) + r380_z, r380_value)
    r381_z = tl.arange($(6144))
    r381_value = tl.load($(candidate_vbmm_72_ptr) + ((((r381_z / $(768)) % $(8)) * $(768)) + ((r381_z % $(192)) * $(4)) + ((r381_z / $(192)) % $(4))), dtype=tl.float32)
    r381_value = r381_value.to(tl.bfloat16)
    r381_value = tl.broadcast_to(r381_value, [$(6144)])
    tl.store($(candidate_vto_copy_87_ptr) + r381_z, r381_value)
    r382_z = tl.arange($(196608))
    r382_value = tl.where(((((r382_z / $(1536)) % $(128)) >= $(48)) & (((r382_z / $(1536)) % $(128)) < $(52)) & ((((r382_z / $(1536)) % $(128)) - $(48)) % $(1) == $(0))), tl.load($(candidate_vto_copy_87_ptr) + ((((r382_z / $(192)) % $(8)) * $(768)) + (((((r382_z / $(1536)) % $(128)) - $(48)) % $(4)) * $(192)) + (r382_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r382_value = r382_value.to(tl.bfloat16)
    r382_value = tl.broadcast_to(r382_value, [$(196608)])
    tl.store($(candidate_vslice_backward_27_ptr) + r382_z, r382_value)
    r383_z = tl.arange($(196608))
    r383_value = (tl.load($(candidate_vslice_backward_25_ptr) + ((((r383_z / $(1536)) % $(128)) * $(1536)) + (((r383_z / $(192)) % $(8)) * $(192)) + (r383_z % $(192))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_27_ptr) + ((((r383_z / $(1536)) % $(128)) * $(1536)) + (((r383_z / $(192)) % $(8)) * $(192)) + (r383_z % $(192))), dtype=tl.bfloat16))
    r383_value = r383_value.to(tl.bfloat16)
    r383_value = tl.broadcast_to(r383_value, [$(196608)])
    tl.store($(candidate_vadd_38_ptr) + r383_z, r383_value)
    r384_row = tl.arange($(12))
    r384_col = tl.arange($(128))
    r384_lane = tl.arange($(12))
    for r384_group in range($(8)) {
      r384_left = tl.load($(candidate_vto_copy_32_ptr) + (((((((r384_group % $(8)) * $(144)) + (r384_lane[None, :] * $(12)) + r384_row[:, None]) / $(144)) % $(8)) * $(144)) + ((((((r384_group % $(8)) * $(144)) + (r384_lane[None, :] * $(12)) + r384_row[:, None]) / $(12)) % $(12)) * $(12)) + ((((r384_group % $(8)) * $(144)) + (r384_lane[None, :] * $(12)) + r384_row[:, None]) % $(12))), dtype=tl.bfloat16)
      r384_right = tl.load($(candidate_vmm_4_ptr) + (((((((r384_group % $(8)) * $(1536)) + (r384_lane[:, None] * $(128)) + r384_col[None, :]) / $(128)) % $(12)) * $(1024)) + ((((((r384_group % $(8)) * $(1536)) + (r384_lane[:, None] * $(128)) + r384_col[None, :]) / $(1536)) % $(8)) * $(128)) + ((((r384_group % $(8)) * $(1536)) + (r384_lane[:, None] * $(128)) + r384_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r384_value = tl.dot(r384_left, r384_right)
      r384_value = r384_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_73_ptr) + ((r384_group * $(1536)) + (r384_row[:, None] * $(128)) + r384_col[None, :]), r384_value)
    }
    r385_row = tl.arange($(12))
    r385_col = tl.arange($(12))
    r385_lane = tl.arange($(128))
    for r385_group in range($(8)) {
      r385_left = tl.load($(candidate_vmm_4_ptr) + (((((((r385_group % $(8)) * $(1536)) + (r385_row[:, None] * $(128)) + r385_lane[None, :]) / $(128)) % $(12)) * $(1024)) + ((((((r385_group % $(8)) * $(1536)) + (r385_row[:, None] * $(128)) + r385_lane[None, :]) / $(1536)) % $(8)) * $(128)) + ((((r385_group % $(8)) * $(1536)) + (r385_row[:, None] * $(128)) + r385_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r385_right = tl.load($(candidate_vbmm_3_ptr) + (((((((r385_group % $(8)) * $(1536)) + (r385_col[None, :] * $(128)) + r385_lane[:, None]) / $(128)) % $(12)) * $(1024)) + ((((((r385_group % $(8)) * $(1536)) + (r385_col[None, :] * $(128)) + r385_lane[:, None]) / $(1536)) % $(8)) * $(128)) + ((((r385_group % $(8)) * $(1536)) + (r385_col[None, :] * $(128)) + r385_lane[:, None]) % $(128))), dtype=tl.bfloat16)
      r385_value = tl.dot(r385_left, r385_right)
      r385_value = r385_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_74_ptr) + ((r385_group * $(144)) + (r385_row[:, None] * $(12)) + r385_col[None, :]), r385_value)
    }
    r386_z = tl.arange($(1152))
    r386_value = tl.load($(candidate_vbmm_74_ptr) + ((((r386_z / $(144)) % $(8)) * $(144)) + (((r386_z / $(12)) % $(12)) * $(12)) + (r386_z % $(12))), dtype=tl.bfloat16)
    r386_value = r386_value.to(tl.float32)
    r386_value = tl.broadcast_to(r386_value, [$(1152)])
    tl.store($(candidate_vto_copy_88_ptr) + r386_z, r386_value)
    r387_z = tl.arange($(1152))
    r387_lane = tl.arange($(12))
    r387_row_values = tl.load($(candidate_vsoftmax_ptr) + ((((r387_z[:, None] / $(144)) % $(8)) * $(144)) + (((r387_z[:, None] / $(12)) % $(12)) * $(12)) + (r387_lane[None, :] % $(12))), dtype=tl.float32)
    r387_row_grad = tl.load($(candidate_vto_copy_88_ptr) + ((((r387_z[:, None] / $(144)) % $(8)) * $(144)) + (((r387_z[:, None] / $(12)) % $(12)) * $(12)) + (r387_lane[None, :] % $(12))), dtype=tl.float32)
    r387_weighted = r387_row_grad * r387_row_values
    r387_dot_grad = tl.sum(r387_weighted, axis=1)
    r387_value = (tl.load($(candidate_vto_copy_88_ptr) + ((((r387_z / $(144)) % $(8)) * $(144)) + (((r387_z / $(12)) % $(12)) * $(12)) + (r387_z % $(12))), dtype=tl.float32) - r387_dot_grad) * tl.load($(candidate_vsoftmax_ptr) + ((((r387_z / $(144)) % $(8)) * $(144)) + (((r387_z / $(12)) % $(12)) * $(12)) + (r387_z % $(12))), dtype=tl.float32)
    r387_value = r387_value.to(tl.float32)
    r387_value = tl.broadcast_to(r387_value, [$(1152)])
    tl.store($(candidate_vsoftmax_backward_data_11_ptr) + r387_z, r387_value)
    r388_z = tl.arange($(1152))
    r388_value = tl.where(tl.load($(candidate_vgt_ptr) + ((((r388_z / $(12)) % $(12)) * $(12)) + (r388_z % $(12))), dtype=tl.int1), 0.0, tl.load($(candidate_vsoftmax_backward_data_11_ptr) + ((((r388_z / $(144)) % $(8)) * $(144)) + (((r388_z / $(12)) % $(12)) * $(12)) + (r388_z % $(12))), dtype=tl.float32))
    r388_value = r388_value.to(tl.float32)
    r388_value = tl.broadcast_to(r388_value, [$(1152)])
    tl.store($(candidate_vmasked_fill_23_ptr) + r388_z, r388_value)
    r389_z = tl.arange($(1152))
    r389_value = (tl.load($(candidate_vmasked_fill_23_ptr) + ((((r389_z / $(144)) % $(8)) * $(144)) + (((r389_z / $(12)) % $(12)) * $(12)) + (r389_z % $(12))), dtype=tl.float32) * 0.07216878364870322)
    r389_value = r389_value.to(tl.float32)
    r389_value = tl.broadcast_to(r389_value, [$(1152)])
    tl.store($(candidate_vmul_46_ptr) + r389_z, r389_value)
    r390_row = tl.arange($(192))
    r390_col = tl.arange($(12))
    r390_lane = tl.arange($(12))
    for r390_group in range($(8)) {
      r390_left = tl.load($(candidate_vto_copy_30_ptr) + (((((((r390_group % $(8)) * $(2304)) + (r390_lane[None, :] * $(192)) + r390_row[:, None]) / $(2304)) % $(8)) * $(2304)) + ((((((r390_group % $(8)) * $(2304)) + (r390_lane[None, :] * $(192)) + r390_row[:, None]) / $(192)) % $(12)) * $(192)) + ((((r390_group % $(8)) * $(2304)) + (r390_lane[None, :] * $(192)) + r390_row[:, None]) % $(192))), dtype=tl.float32)
      r390_right = tl.load($(candidate_vmul_46_ptr) + (((((((r390_group % $(8)) * $(144)) + (r390_lane[:, None] * $(12)) + r390_col[None, :]) / $(144)) % $(8)) * $(144)) + ((((((r390_group % $(8)) * $(144)) + (r390_lane[:, None] * $(12)) + r390_col[None, :]) / $(12)) % $(12)) * $(12)) + ((((r390_group % $(8)) * $(144)) + (r390_lane[:, None] * $(12)) + r390_col[None, :]) % $(12))), dtype=tl.float32)
      r390_value = tl.dot(r390_left, r390_right)
      r390_value = r390_value.to(tl.float32)
      tl.store($(candidate_vbmm_75_ptr) + ((r390_group * $(2304)) + (r390_row[:, None] * $(12)) + r390_col[None, :]), r390_value)
    }
    r391_z = tl.arange($(131072))
    r391_value = tl.where(((((r391_z / $(1024)) % $(128)) >= $(0)) & (((r391_z / $(1024)) % $(128)) < $(12)) & ((((r391_z / $(1024)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vbmm_73_ptr) + ((((((((r391_z / $(128)) % $(8)) * $(1536)) + ((((r391_z / $(1024)) % $(128)) - $(0)) * $(128)) + (r391_z % $(128))) / $(1536)) % $(8)) * $(1536)) + ((((((r391_z / $(128)) % $(8)) * $(12)) + (((r391_z / $(1024)) % $(128)) - $(0))) % $(12)) * $(128)) + (r391_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r391_value = r391_value.to(tl.bfloat16)
    r391_value = tl.broadcast_to(r391_value, [$(131072)])
    tl.store($(candidate_vslice_backward_28_ptr) + r391_z, r391_value)
    r392_z = tl.arange($(131072))
    r392_value = (tl.load($(candidate_vadd_37_ptr) + ((((r392_z / $(1024)) % $(128)) * $(1024)) + (((r392_z / $(128)) % $(8)) * $(128)) + (r392_z % $(128))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_28_ptr) + ((((r392_z / $(1024)) % $(128)) * $(1024)) + (((r392_z / $(128)) % $(8)) * $(128)) + (r392_z % $(128))), dtype=tl.bfloat16))
    r392_value = r392_value.to(tl.bfloat16)
    r392_value = tl.broadcast_to(r392_value, [$(131072)])
    tl.store($(candidate_vadd_39_ptr) + r392_z, r392_value)
    r393_z = tl.arange($(18432))
    r393_value = tl.load($(candidate_vbmm_75_ptr) + ((((r393_z / $(2304)) % $(8)) * $(2304)) + ((r393_z % $(192)) * $(12)) + ((r393_z / $(192)) % $(12))), dtype=tl.float32)
    r393_value = r393_value.to(tl.bfloat16)
    r393_value = tl.broadcast_to(r393_value, [$(18432)])
    tl.store($(candidate_vto_copy_89_ptr) + r393_z, r393_value)
    r394_z = tl.arange($(196608))
    r394_value = tl.where(((((r394_z / $(1536)) % $(128)) >= $(0)) & (((r394_z / $(1536)) % $(128)) < $(12)) & ((((r394_z / $(1536)) % $(128)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vto_copy_89_ptr) + ((((r394_z / $(192)) % $(8)) * $(2304)) + (((((r394_z / $(1536)) % $(128)) - $(0)) % $(12)) * $(192)) + (r394_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r394_value = r394_value.to(tl.bfloat16)
    r394_value = tl.broadcast_to(r394_value, [$(196608)])
    tl.store($(candidate_vslice_backward_29_ptr) + r394_z, r394_value)
    r395_z = tl.arange($(196608))
    r395_value = (tl.load($(candidate_vadd_38_ptr) + ((((r395_z / $(1536)) % $(128)) * $(1536)) + (((r395_z / $(192)) % $(8)) * $(192)) + (r395_z % $(192))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_29_ptr) + ((((r395_z / $(1536)) % $(128)) * $(1536)) + (((r395_z / $(192)) % $(8)) * $(192)) + (r395_z % $(192))), dtype=tl.bfloat16))
    r395_value = r395_value.to(tl.bfloat16)
    r395_value = tl.broadcast_to(r395_value, [$(196608)])
    tl.store($(candidate_vadd_40_ptr) + r395_z, r395_value)
    r396_z = tl.arange($(36864))
    r396_value = tl.cast(0.0, tl.bfloat16)
    r396_value = r396_value.to(tl.bfloat16)
    r396_value = tl.broadcast_to(r396_value, [$(36864)])
    tl.store($(candidate_vfill_18_ptr) + r396_z, r396_value)
    r397_z = tl.arange($(131072))
    r397_value = tl.where(((((r397_z / $(1024)) % $(128)) >= $(12)) & (((r397_z / $(1024)) % $(128)) < $(48)) & ((((r397_z / $(1024)) % $(128)) - $(12)) % $(1) == $(0))), tl.load($(candidate_vfill_18_ptr) + ((((((r397_z / $(1024)) % $(128)) - $(12)) % $(36)) * $(1024)) + (((r397_z / $(128)) % $(8)) * $(128)) + (r397_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vadd_39_ptr) + ((((r397_z / $(1024)) % $(128)) * $(1024)) + (((r397_z / $(128)) % $(8)) * $(128)) + (r397_z % $(128))), dtype=tl.bfloat16))
    r397_value = r397_value.to(tl.bfloat16)
    r397_value = tl.broadcast_to(r397_value, [$(131072)])
    tl.store($(candidate_vslice_scatter_18_ptr) + r397_z, r397_value)
    r398_z = tl.arange($(12288))
    r398_value = tl.cast(0.0, tl.bfloat16)
    r398_value = r398_value.to(tl.bfloat16)
    r398_value = tl.broadcast_to(r398_value, [$(12288)])
    tl.store($(candidate_vfill_19_ptr) + r398_z, r398_value)
    r399_z = tl.arange($(131072))
    r399_value = tl.where(((((r399_z / $(1024)) % $(128)) >= $(52)) & (((r399_z / $(1024)) % $(128)) < $(64)) & ((((r399_z / $(1024)) % $(128)) - $(52)) % $(1) == $(0))), tl.load($(candidate_vfill_19_ptr) + ((((((r399_z / $(1024)) % $(128)) - $(52)) % $(12)) * $(1024)) + (((r399_z / $(128)) % $(8)) * $(128)) + (r399_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vslice_scatter_18_ptr) + ((((r399_z / $(1024)) % $(128)) * $(1024)) + (((r399_z / $(128)) % $(8)) * $(128)) + (r399_z % $(128))), dtype=tl.bfloat16))
    r399_value = r399_value.to(tl.bfloat16)
    r399_value = tl.broadcast_to(r399_value, [$(131072)])
    tl.store($(candidate_vslice_scatter_19_ptr) + r399_z, r399_value)
    r400_z = tl.arange($(49152))
    r400_value = tl.cast(0.0, tl.bfloat16)
    r400_value = r400_value.to(tl.bfloat16)
    r400_value = tl.broadcast_to(r400_value, [$(49152)])
    tl.store($(candidate_vfill_20_ptr) + r400_z, r400_value)
    r401_z = tl.arange($(131072))
    r401_value = tl.where(((((r401_z / $(1024)) % $(128)) >= $(80)) & (((r401_z / $(1024)) % $(128)) < $(128)) & ((((r401_z / $(1024)) % $(128)) - $(80)) % $(1) == $(0))), tl.load($(candidate_vfill_20_ptr) + ((((((r401_z / $(1024)) % $(128)) - $(80)) % $(48)) * $(1024)) + (((r401_z / $(128)) % $(8)) * $(128)) + (r401_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vslice_scatter_19_ptr) + ((((r401_z / $(1024)) % $(128)) * $(1024)) + (((r401_z / $(128)) % $(8)) * $(128)) + (r401_z % $(128))), dtype=tl.bfloat16))
    r401_value = r401_value.to(tl.bfloat16)
    r401_value = tl.broadcast_to(r401_value, [$(131072)])
    tl.store($(candidate_vslice_scatter_20_ptr) + r401_z, r401_value)
    r402_z = tl.arange($(55296))
    r402_value = tl.cast(0.0, tl.bfloat16)
    r402_value = r402_value.to(tl.bfloat16)
    r402_value = tl.broadcast_to(r402_value, [$(55296)])
    tl.store($(candidate_vfill_21_ptr) + r402_z, r402_value)
    r403_z = tl.arange($(196608))
    r403_value = tl.where(((((r403_z / $(1536)) % $(128)) >= $(12)) & (((r403_z / $(1536)) % $(128)) < $(48)) & ((((r403_z / $(1536)) % $(128)) - $(12)) % $(1) == $(0))), tl.load($(candidate_vfill_21_ptr) + ((((((r403_z / $(1536)) % $(128)) - $(12)) % $(36)) * $(1536)) + (((r403_z / $(192)) % $(8)) * $(192)) + (r403_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vadd_40_ptr) + ((((r403_z / $(1536)) % $(128)) * $(1536)) + (((r403_z / $(192)) % $(8)) * $(192)) + (r403_z % $(192))), dtype=tl.bfloat16))
    r403_value = r403_value.to(tl.bfloat16)
    r403_value = tl.broadcast_to(r403_value, [$(196608)])
    tl.store($(candidate_vslice_scatter_21_ptr) + r403_z, r403_value)
    r404_z = tl.arange($(18432))
    r404_value = tl.cast(0.0, tl.bfloat16)
    r404_value = r404_value.to(tl.bfloat16)
    r404_value = tl.broadcast_to(r404_value, [$(18432)])
    tl.store($(candidate_vfill_22_ptr) + r404_z, r404_value)
    r405_z = tl.arange($(196608))
    r405_value = tl.where(((((r405_z / $(1536)) % $(128)) >= $(52)) & (((r405_z / $(1536)) % $(128)) < $(64)) & ((((r405_z / $(1536)) % $(128)) - $(52)) % $(1) == $(0))), tl.load($(candidate_vfill_22_ptr) + ((((((r405_z / $(1536)) % $(128)) - $(52)) % $(12)) * $(1536)) + (((r405_z / $(192)) % $(8)) * $(192)) + (r405_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vslice_scatter_21_ptr) + ((((r405_z / $(1536)) % $(128)) * $(1536)) + (((r405_z / $(192)) % $(8)) * $(192)) + (r405_z % $(192))), dtype=tl.bfloat16))
    r405_value = r405_value.to(tl.bfloat16)
    r405_value = tl.broadcast_to(r405_value, [$(196608)])
    tl.store($(candidate_vslice_scatter_22_ptr) + r405_z, r405_value)
    r406_z = tl.arange($(73728))
    r406_value = tl.cast(0.0, tl.bfloat16)
    r406_value = r406_value.to(tl.bfloat16)
    r406_value = tl.broadcast_to(r406_value, [$(73728)])
    tl.store($(candidate_vfill_23_ptr) + r406_z, r406_value)
    r407_z = tl.arange($(196608))
    r407_value = tl.where(((((r407_z / $(1536)) % $(128)) >= $(80)) & (((r407_z / $(1536)) % $(128)) < $(128)) & ((((r407_z / $(1536)) % $(128)) - $(80)) % $(1) == $(0))), tl.load($(candidate_vfill_23_ptr) + ((((((r407_z / $(1536)) % $(128)) - $(80)) % $(48)) * $(1536)) + (((r407_z / $(192)) % $(8)) * $(192)) + (r407_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vslice_scatter_22_ptr) + ((((r407_z / $(1536)) % $(128)) * $(1536)) + (((r407_z / $(192)) % $(8)) * $(192)) + (r407_z % $(192))), dtype=tl.bfloat16))
    r407_value = r407_value.to(tl.bfloat16)
    r407_value = tl.broadcast_to(r407_value, [$(196608)])
    tl.store($(candidate_vslice_scatter_23_ptr) + r407_z, r407_value)
    r408_row = tl.arange($(256))
    r408_col = tl.arange($(1024))
    r408_lane = tl.arange($(128))
    for r408_group in range($(1)) {
      r408_left = tl.load($(candidate_vindex_2_ptr) + ((((((r408_lane[None, :] * $(256)) + r408_row[:, None]) / $(256)) % $(128)) * $(320)) + (((r408_lane[None, :] * $(256)) + r408_row[:, None]) % $(256))), dtype=tl.bfloat16)
      r408_right = tl.load($(candidate_vslice_scatter_20_ptr) + ((((((r408_lane[:, None] * $(1024)) + r408_col[None, :]) / $(1024)) % $(128)) * $(1024)) + (((((r408_lane[:, None] * $(1024)) + r408_col[None, :]) / $(128)) % $(8)) * $(128)) + (((r408_lane[:, None] * $(1024)) + r408_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r408_value = tl.dot(r408_left, r408_right)
      r408_value = r408_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_76_ptr) + ((r408_group * $(262144)) + (r408_row[:, None] * $(1024)) + r408_col[None, :]), r408_value)
    }
    r409_row = tl.arange($(256))
    r409_col = tl.arange($(1024))
    r409_lane = tl.arange($(128))
    for r409_group in range($(1)) {
      r409_left = tl.load($(candidate_vindex_2_ptr) + ((((((r409_lane[None, :] * $(256)) + r409_row[:, None]) / $(256)) % $(128)) * $(320)) + (((r409_lane[None, :] * $(256)) + r409_row[:, None]) % $(256))), dtype=tl.bfloat16)
      r409_right = tl.load($(candidate_vslice_scatter_23_ptr) + ((((((r409_lane[:, None] * $(1024)) + r409_col[None, :]) / $(1024)) % $(128)) * $(1536)) + (((((r409_lane[:, None] * $(1024)) + r409_col[None, :]) / $(128)) % $(8)) * $(192)) + (((r409_lane[:, None] * $(1024)) + r409_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r409_value = tl.dot(r409_left, r409_right)
      r409_value = r409_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_77_ptr) + ((r409_group * $(262144)) + (r409_row[:, None] * $(1024)) + r409_col[None, :]), r409_value)
    }
    r410_z = tl.arange($(524288))
    r410_value = tl.where(((((r410_z / $(256)) % $(256)) >= $(128)) & (((r410_z / $(256)) % $(256)) < $(256)) & ((((r410_z / $(256)) % $(256)) - $(128)) % $(1) == $(0))), tl.load($(candidate_vbmm_76_ptr) + (((r410_z % $(256)) * $(1024)) + (((((((r410_z / $(65536)) % $(8)) * $(32768)) + ((((r410_z / $(256)) % $(256)) - $(128)) * $(256)) + (r410_z % $(256))) / $(32768)) % $(8)) * $(128)) + (((((r410_z / $(65536)) % $(8)) * $(128)) + (((r410_z / $(256)) % $(256)) - $(128))) % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r410_value = r410_value.to(tl.bfloat16)
    r410_value = tl.broadcast_to(r410_value, [$(524288)])
    tl.store($(candidate_vslice_backward_30_ptr) + r410_z, r410_value)
    r411_z = tl.arange($(524288))
    r411_value = tl.where(((((r411_z / $(256)) % $(256)) >= $(0)) & (((r411_z / $(256)) % $(256)) < $(128)) & ((((r411_z / $(256)) % $(256)) - $(0)) % $(1) == $(0))), tl.load($(candidate_vbmm_77_ptr) + (((r411_z % $(256)) * $(1024)) + (((((((r411_z / $(65536)) % $(8)) * $(32768)) + ((((r411_z / $(256)) % $(256)) - $(0)) * $(256)) + (r411_z % $(256))) / $(32768)) % $(8)) * $(128)) + (((((r411_z / $(65536)) % $(8)) * $(128)) + (((r411_z / $(256)) % $(256)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r411_value = r411_value.to(tl.bfloat16)
    r411_value = tl.broadcast_to(r411_value, [$(524288)])
    tl.store($(candidate_vslice_backward_31_ptr) + r411_z, r411_value)
    r412_z = tl.arange($(524288))
    r412_value = (tl.load($(candidate_vslice_backward_30_ptr) + ((((r412_z / $(65536)) % $(8)) * $(65536)) + (((r412_z / $(256)) % $(256)) * $(256)) + (r412_z % $(256))), dtype=tl.bfloat16) + tl.load($(candidate_vslice_backward_31_ptr) + ((((r412_z / $(65536)) % $(8)) * $(65536)) + (((r412_z / $(256)) % $(256)) * $(256)) + (r412_z % $(256))), dtype=tl.bfloat16))
    r412_value = r412_value.to(tl.bfloat16)
    r412_value = tl.broadcast_to(r412_value, [$(524288)])
    tl.store($(candidate_vadd_41_ptr) + r412_z, r412_value)
    r413_z = tl.arange($(524288))
    r413_value = (tl.load($(candidate_vadd_36_ptr) + ((((r413_z / $(256)) % $(2048)) * $(256)) + (r413_z % $(256))), dtype=tl.bfloat16) + tl.load($(candidate_vadd_41_ptr) + ((((((((r413_z / $(256)) % $(2048)) * $(256)) + (r413_z % $(256))) / $(65536)) % $(8)) * $(65536)) + ((((r413_z / $(256)) % $(2048)) % $(256)) * $(256)) + (r413_z % $(256))), dtype=tl.bfloat16))
    r413_value = r413_value.to(tl.bfloat16)
    r413_value = tl.broadcast_to(r413_value, [$(524288)])
    tl.store($(candidate_vadd_42_ptr) + r413_z, r413_value)
    z = tl.arange($(524288))
    value = tl.load($(candidate_vadd_42_ptr) + ((((z / $(256)) % $(2048)) * $(256)) + (z % $(256))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(524288)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_mla_fa4_te_rope_cp_nonabsorb_wgrad
