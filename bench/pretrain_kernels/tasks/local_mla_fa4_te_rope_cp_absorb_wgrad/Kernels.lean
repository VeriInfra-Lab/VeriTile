import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_mla_fa4_te_rope_cp_absorb_wgrad
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
    r40_z = tl.arange($(1048576))
    r40_value = tl.load($(vmul_5_ptr) + ((((r40_z / $(512)) % $(2048)) * $(512)) + (r40_z % $(512))), dtype=tl.float32)
    r40_value = r40_value.to(tl.bfloat16)
    r40_value = tl.broadcast_to(r40_value, [$(1048576)])
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
    tl.store($(vto_copy_23_ptr) + r52_z, r52_value)
    r53_z = tl.arange($(65536))
    r53_value = ((tl.load($(vto_copy_23_ptr) + ((((r53_z / $(512)) % $(128)) * $(512)) + (r53_z % $(512))), dtype=tl.float32)) * (tl.load($(vto_copy_23_ptr) + ((((r53_z / $(512)) % $(128)) * $(512)) + (r53_z % $(512))), dtype=tl.float32)))
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
    r57_value = (tl.load($(vto_copy_23_ptr) + ((((r57_z / $(512)) % $(128)) * $(512)) + (r57_z % $(512))), dtype=tl.float32) * tl.load($(vrsqrt_1_ptr) + ((r57_z / $(512)) % $(128)), dtype=tl.float32))
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
    tl.store($(vto_copy_24_ptr) + r59_z, r59_value)
    r60_row = tl.arange($(128))
    r60_col = tl.arange($(2048))
    r60_lane = tl.arange($(512))
    r60_left = tl.load($(vto_copy_24_ptr) + ((((((r60_row[:, None] * $(512)) + r60_lane[None, :]) / $(512)) % $(128)) * $(512)) + (((r60_row[:, None] * $(512)) + r60_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r60_right = tl.load($(vto_copy_20_ptr) + (((r60_col[None, :] % $(2048)) * $(512)) + (r60_lane[:, None] % $(512))), dtype=tl.bfloat16)
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
    r92_col = tl.arange($(512))
    r92_lane = tl.arange($(128))
    r92_left = tl.load($(vcat_9_ptr) + ((((((((((r92_lane[None, :] * $(2048)) + r92_row[:, None]) / $(2048)) % $(128)) * $(2048)) + (((r92_lane[None, :] * $(2048)) + r92_row[:, None]) % $(2048))) / $(256)) % $(8)) * $(32768)) + (((((r92_lane[None, :] * $(2048)) + r92_row[:, None]) / $(2048)) % $(128)) * $(256)) + (((((((r92_lane[None, :] * $(2048)) + r92_row[:, None]) / $(2048)) % $(128)) * $(2048)) + (((r92_lane[None, :] * $(2048)) + r92_row[:, None]) % $(2048))) % $(256))), dtype=tl.bfloat16)
    r92_right = tl.load($(vto_copy_24_ptr) + ((((((r92_lane[:, None] * $(512)) + r92_col[None, :]) / $(512)) % $(128)) * $(512)) + (((r92_lane[:, None] * $(512)) + r92_col[None, :]) % $(512))), dtype=tl.bfloat16)
    r92_value = tl.dot(r92_left, r92_right)
    r92_value = r92_value.to(tl.bfloat16)
    tl.store($(vmm_6_ptr) + ((r92_row[:, None] * $(512)) + r92_col[None, :]), r92_value)
    z = tl.arange($(1048576))
    value = tl.load($(vmm_6_ptr) + ((((z / $(512)) % $(2048)) * $(512)) + (z % $(512))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(1048576)])
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
  let vmul_19_ptr : RegionName := "vmul_19"
  let vmul_20_ptr : RegionName := "vmul_20"
  let vadd_5_ptr : RegionName := "vadd_5"
  let vcat_5_ptr : RegionName := "vcat_5"
  let vmul_21_ptr : RegionName := "vmul_21"
  let vmul_22_ptr : RegionName := "vmul_22"
  let vadd_6_ptr : RegionName := "vadd_6"
  let vadd_7_ptr : RegionName := "vadd_7"
  let vto_copy_25_ptr : RegionName := "vto_copy_25"
  let vto_copy_26_ptr : RegionName := "vto_copy_26"
  let vto_copy_27_ptr : RegionName := "vto_copy_27"
  let vto_copy_28_ptr : RegionName := "vto_copy_28"
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
  let candidate_vindex_put_ptr : RegionName := "candidate_vindex_put"
  let candidate_vmm_28_ptr : RegionName := "candidate_vmm_28"
  let candidate_vbmm_22_ptr : RegionName := "candidate_vbmm_22"
  let candidate_vbmm_23_ptr : RegionName := "candidate_vbmm_23"
  let candidate_vcat_11_ptr : RegionName := "candidate_vcat_11"
  let candidate_vcat_12_ptr : RegionName := "candidate_vcat_12"
  let candidate_vto_copy_90_ptr : RegionName := "candidate_vto_copy_90"
  let candidate_vto_copy_91_ptr : RegionName := "candidate_vto_copy_91"
  let candidate_vbmm_24_ptr : RegionName := "candidate_vbmm_24"
  let candidate_vmul_35_ptr : RegionName := "candidate_vmul_35"
  let candidate_vadd_32_ptr : RegionName := "candidate_vadd_32"
  let candidate_vsub_21_ptr : RegionName := "candidate_vsub_21"
  let candidate_vgt_12_ptr : RegionName := "candidate_vgt_12"
  let candidate_vmasked_fill_12_ptr : RegionName := "candidate_vmasked_fill_12"
  let candidate_vsoftmax_12_ptr : RegionName := "candidate_vsoftmax_12"
  let candidate_vto_copy_92_ptr : RegionName := "candidate_vto_copy_92"
  let candidate_vbmm_25_ptr : RegionName := "candidate_vbmm_25"
  let candidate_vlogsumexp_12_ptr : RegionName := "candidate_vlogsumexp_12"
  let candidate_vto_copy_93_ptr : RegionName := "candidate_vto_copy_93"
  let candidate_vto_copy_94_ptr : RegionName := "candidate_vto_copy_94"
  let candidate_vbmm_26_ptr : RegionName := "candidate_vbmm_26"
  let candidate_vmul_36_ptr : RegionName := "candidate_vmul_36"
  let candidate_vadd_33_ptr : RegionName := "candidate_vadd_33"
  let candidate_vsub_22_ptr : RegionName := "candidate_vsub_22"
  let candidate_vgt_13_ptr : RegionName := "candidate_vgt_13"
  let candidate_vmasked_fill_13_ptr : RegionName := "candidate_vmasked_fill_13"
  let candidate_vsoftmax_13_ptr : RegionName := "candidate_vsoftmax_13"
  let candidate_vto_copy_95_ptr : RegionName := "candidate_vto_copy_95"
  let candidate_vbmm_27_ptr : RegionName := "candidate_vbmm_27"
  let candidate_vlogsumexp_13_ptr : RegionName := "candidate_vlogsumexp_13"
  let candidate_vto_copy_96_ptr : RegionName := "candidate_vto_copy_96"
  let candidate_vto_copy_97_ptr : RegionName := "candidate_vto_copy_97"
  let candidate_vbmm_28_ptr : RegionName := "candidate_vbmm_28"
  let candidate_vmul_37_ptr : RegionName := "candidate_vmul_37"
  let candidate_vadd_34_ptr : RegionName := "candidate_vadd_34"
  let candidate_vsub_23_ptr : RegionName := "candidate_vsub_23"
  let candidate_vgt_14_ptr : RegionName := "candidate_vgt_14"
  let candidate_vmasked_fill_14_ptr : RegionName := "candidate_vmasked_fill_14"
  let candidate_vsoftmax_14_ptr : RegionName := "candidate_vsoftmax_14"
  let candidate_vto_copy_98_ptr : RegionName := "candidate_vto_copy_98"
  let candidate_vbmm_29_ptr : RegionName := "candidate_vbmm_29"
  let candidate_vlogsumexp_14_ptr : RegionName := "candidate_vlogsumexp_14"
  let candidate_vcat_13_ptr : RegionName := "candidate_vcat_13"
  let candidate_vcat_14_ptr : RegionName := "candidate_vcat_14"
  let candidate_vto_copy_99_ptr : RegionName := "candidate_vto_copy_99"
  let candidate_vto_copy_100_ptr : RegionName := "candidate_vto_copy_100"
  let candidate_vto_copy_101_ptr : RegionName := "candidate_vto_copy_101"
  let candidate_vto_copy_102_ptr : RegionName := "candidate_vto_copy_102"
  let candidate_vto_copy_103_ptr : RegionName := "candidate_vto_copy_103"
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
  let candidate_vbmm_33_ptr : RegionName := "candidate_vbmm_33"
  let candidate_vmul_42_ptr : RegionName := "candidate_vmul_42"
  let candidate_vto_copy_105_ptr : RegionName := "candidate_vto_copy_105"
  let candidate_vbmm_34_ptr : RegionName := "candidate_vbmm_34"
  let candidate_vto_copy_106_ptr : RegionName := "candidate_vto_copy_106"
  let candidate_vcat_15_ptr : RegionName := "candidate_vcat_15"
  let candidate_vcat_16_ptr : RegionName := "candidate_vcat_16"
  let candidate_vto_copy_107_ptr : RegionName := "candidate_vto_copy_107"
  let candidate_vto_copy_108_ptr : RegionName := "candidate_vto_copy_108"
  let candidate_vto_copy_109_ptr : RegionName := "candidate_vto_copy_109"
  let candidate_vto_copy_110_ptr : RegionName := "candidate_vto_copy_110"
  let candidate_vto_copy_111_ptr : RegionName := "candidate_vto_copy_111"
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
  let candidate_vbmm_38_ptr : RegionName := "candidate_vbmm_38"
  let candidate_vmul_47_ptr : RegionName := "candidate_vmul_47"
  let candidate_vto_copy_113_ptr : RegionName := "candidate_vto_copy_113"
  let candidate_vbmm_39_ptr : RegionName := "candidate_vbmm_39"
  let candidate_vto_copy_114_ptr : RegionName := "candidate_vto_copy_114"
  let candidate_vcat_17_ptr : RegionName := "candidate_vcat_17"
  let candidate_vcat_18_ptr : RegionName := "candidate_vcat_18"
  let candidate_vto_copy_115_ptr : RegionName := "candidate_vto_copy_115"
  let candidate_vto_copy_116_ptr : RegionName := "candidate_vto_copy_116"
  let candidate_vto_copy_117_ptr : RegionName := "candidate_vto_copy_117"
  let candidate_vto_copy_118_ptr : RegionName := "candidate_vto_copy_118"
  let candidate_vto_copy_119_ptr : RegionName := "candidate_vto_copy_119"
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
  let candidate_vbmm_43_ptr : RegionName := "candidate_vbmm_43"
  let candidate_vmul_52_ptr : RegionName := "candidate_vmul_52"
  let candidate_vto_copy_121_ptr : RegionName := "candidate_vto_copy_121"
  let candidate_vbmm_44_ptr : RegionName := "candidate_vbmm_44"
  let candidate_vto_copy_122_ptr : RegionName := "candidate_vto_copy_122"
  let candidate_vcat_19_ptr : RegionName := "candidate_vcat_19"
  let candidate_vcat_20_ptr : RegionName := "candidate_vcat_20"
  let candidate_vcat_22_ptr : RegionName := "candidate_vcat_22"
  let candidate_vcat_23_ptr : RegionName := "candidate_vcat_23"
  let candidate_vbmm_47_ptr : RegionName := "candidate_vbmm_47"
  let candidate_vbmm_48_ptr : RegionName := "candidate_vbmm_48"
  let candidate_vcat_24_ptr : RegionName := "candidate_vcat_24"
  let candidate_vbmm_49_ptr : RegionName := "candidate_vbmm_49"
  let candidate_vbmm_50_ptr : RegionName := "candidate_vbmm_50"
  let candidate_vcat_25_ptr : RegionName := "candidate_vcat_25"
  let candidate_vcat_26_ptr : RegionName := "candidate_vcat_26"
  let candidate_vto_copy_123_ptr : RegionName := "candidate_vto_copy_123"
  let candidate_vto_copy_124_ptr : RegionName := "candidate_vto_copy_124"
  let candidate_vbmm_51_ptr : RegionName := "candidate_vbmm_51"
  let candidate_vmul_53_ptr : RegionName := "candidate_vmul_53"
  let candidate_vadd_39_ptr : RegionName := "candidate_vadd_39"
  let candidate_vsub_33_ptr : RegionName := "candidate_vsub_33"
  let candidate_vgt_18_ptr : RegionName := "candidate_vgt_18"
  let candidate_vmasked_fill_18_ptr : RegionName := "candidate_vmasked_fill_18"
  let candidate_vsoftmax_15_ptr : RegionName := "candidate_vsoftmax_15"
  let candidate_vto_copy_125_ptr : RegionName := "candidate_vto_copy_125"
  let candidate_vbmm_52_ptr : RegionName := "candidate_vbmm_52"
  let candidate_vlogsumexp_15_ptr : RegionName := "candidate_vlogsumexp_15"
  let candidate_vto_copy_126_ptr : RegionName := "candidate_vto_copy_126"
  let candidate_vto_copy_127_ptr : RegionName := "candidate_vto_copy_127"
  let candidate_vbmm_53_ptr : RegionName := "candidate_vbmm_53"
  let candidate_vmul_54_ptr : RegionName := "candidate_vmul_54"
  let candidate_vadd_40_ptr : RegionName := "candidate_vadd_40"
  let candidate_vsub_34_ptr : RegionName := "candidate_vsub_34"
  let candidate_vgt_19_ptr : RegionName := "candidate_vgt_19"
  let candidate_vmasked_fill_19_ptr : RegionName := "candidate_vmasked_fill_19"
  let candidate_vsoftmax_16_ptr : RegionName := "candidate_vsoftmax_16"
  let candidate_vto_copy_128_ptr : RegionName := "candidate_vto_copy_128"
  let candidate_vbmm_54_ptr : RegionName := "candidate_vbmm_54"
  let candidate_vlogsumexp_16_ptr : RegionName := "candidate_vlogsumexp_16"
  let candidate_vto_copy_129_ptr : RegionName := "candidate_vto_copy_129"
  let candidate_vto_copy_130_ptr : RegionName := "candidate_vto_copy_130"
  let candidate_vbmm_55_ptr : RegionName := "candidate_vbmm_55"
  let candidate_vmul_55_ptr : RegionName := "candidate_vmul_55"
  let candidate_vadd_41_ptr : RegionName := "candidate_vadd_41"
  let candidate_vsub_35_ptr : RegionName := "candidate_vsub_35"
  let candidate_vgt_20_ptr : RegionName := "candidate_vgt_20"
  let candidate_vmasked_fill_20_ptr : RegionName := "candidate_vmasked_fill_20"
  let candidate_vsoftmax_17_ptr : RegionName := "candidate_vsoftmax_17"
  let candidate_vto_copy_131_ptr : RegionName := "candidate_vto_copy_131"
  let candidate_vbmm_56_ptr : RegionName := "candidate_vbmm_56"
  let candidate_vlogsumexp_17_ptr : RegionName := "candidate_vlogsumexp_17"
  let candidate_vcat_27_ptr : RegionName := "candidate_vcat_27"
  let candidate_vcat_28_ptr : RegionName := "candidate_vcat_28"
  let candidate_vto_copy_132_ptr : RegionName := "candidate_vto_copy_132"
  let candidate_vto_copy_133_ptr : RegionName := "candidate_vto_copy_133"
  let candidate_vto_copy_134_ptr : RegionName := "candidate_vto_copy_134"
  let candidate_vto_copy_135_ptr : RegionName := "candidate_vto_copy_135"
  let candidate_vto_copy_136_ptr : RegionName := "candidate_vto_copy_136"
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
  let candidate_vbmm_60_ptr : RegionName := "candidate_vbmm_60"
  let candidate_vmul_60_ptr : RegionName := "candidate_vmul_60"
  let candidate_vto_copy_138_ptr : RegionName := "candidate_vto_copy_138"
  let candidate_vbmm_61_ptr : RegionName := "candidate_vbmm_61"
  let candidate_vto_copy_139_ptr : RegionName := "candidate_vto_copy_139"
  let candidate_vcat_29_ptr : RegionName := "candidate_vcat_29"
  let candidate_vcat_30_ptr : RegionName := "candidate_vcat_30"
  let candidate_vto_copy_140_ptr : RegionName := "candidate_vto_copy_140"
  let candidate_vto_copy_141_ptr : RegionName := "candidate_vto_copy_141"
  let candidate_vto_copy_142_ptr : RegionName := "candidate_vto_copy_142"
  let candidate_vto_copy_143_ptr : RegionName := "candidate_vto_copy_143"
  let candidate_vto_copy_144_ptr : RegionName := "candidate_vto_copy_144"
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
  let candidate_vbmm_65_ptr : RegionName := "candidate_vbmm_65"
  let candidate_vmul_65_ptr : RegionName := "candidate_vmul_65"
  let candidate_vto_copy_146_ptr : RegionName := "candidate_vto_copy_146"
  let candidate_vbmm_66_ptr : RegionName := "candidate_vbmm_66"
  let candidate_vto_copy_147_ptr : RegionName := "candidate_vto_copy_147"
  let candidate_vcat_31_ptr : RegionName := "candidate_vcat_31"
  let candidate_vcat_32_ptr : RegionName := "candidate_vcat_32"
  let candidate_vto_copy_148_ptr : RegionName := "candidate_vto_copy_148"
  let candidate_vto_copy_149_ptr : RegionName := "candidate_vto_copy_149"
  let candidate_vto_copy_150_ptr : RegionName := "candidate_vto_copy_150"
  let candidate_vto_copy_151_ptr : RegionName := "candidate_vto_copy_151"
  let candidate_vto_copy_152_ptr : RegionName := "candidate_vto_copy_152"
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
  let candidate_vbmm_70_ptr : RegionName := "candidate_vbmm_70"
  let candidate_vmul_70_ptr : RegionName := "candidate_vmul_70"
  let candidate_vto_copy_154_ptr : RegionName := "candidate_vto_copy_154"
  let candidate_vbmm_71_ptr : RegionName := "candidate_vbmm_71"
  let candidate_vto_copy_155_ptr : RegionName := "candidate_vto_copy_155"
  let candidate_vcat_33_ptr : RegionName := "candidate_vcat_33"
  let candidate_vcat_34_ptr : RegionName := "candidate_vcat_34"
  let candidate_vcat_36_ptr : RegionName := "candidate_vcat_36"
  let candidate_vcat_37_ptr : RegionName := "candidate_vcat_37"
  let candidate_vbmm_74_ptr : RegionName := "candidate_vbmm_74"
  let candidate_vbmm_75_ptr : RegionName := "candidate_vbmm_75"
  let candidate_vcat_38_ptr : RegionName := "candidate_vcat_38"
  let candidate_vadd_46_ptr : RegionName := "candidate_vadd_46"
  let candidate_vbmm_76_ptr : RegionName := "candidate_vbmm_76"
  let candidate_vbmm_77_ptr : RegionName := "candidate_vbmm_77"
  let candidate_vcat_39_ptr : RegionName := "candidate_vcat_39"
  let candidate_vcat_40_ptr : RegionName := "candidate_vcat_40"
  let candidate_vto_copy_156_ptr : RegionName := "candidate_vto_copy_156"
  let candidate_vto_copy_157_ptr : RegionName := "candidate_vto_copy_157"
  let candidate_vbmm_78_ptr : RegionName := "candidate_vbmm_78"
  let candidate_vmul_71_ptr : RegionName := "candidate_vmul_71"
  let candidate_vadd_47_ptr : RegionName := "candidate_vadd_47"
  let candidate_vsub_45_ptr : RegionName := "candidate_vsub_45"
  let candidate_vgt_24_ptr : RegionName := "candidate_vgt_24"
  let candidate_vmasked_fill_24_ptr : RegionName := "candidate_vmasked_fill_24"
  let candidate_vsoftmax_18_ptr : RegionName := "candidate_vsoftmax_18"
  let candidate_vto_copy_158_ptr : RegionName := "candidate_vto_copy_158"
  let candidate_vbmm_79_ptr : RegionName := "candidate_vbmm_79"
  let candidate_vlogsumexp_18_ptr : RegionName := "candidate_vlogsumexp_18"
  let candidate_vto_copy_159_ptr : RegionName := "candidate_vto_copy_159"
  let candidate_vto_copy_160_ptr : RegionName := "candidate_vto_copy_160"
  let candidate_vbmm_80_ptr : RegionName := "candidate_vbmm_80"
  let candidate_vmul_72_ptr : RegionName := "candidate_vmul_72"
  let candidate_vadd_48_ptr : RegionName := "candidate_vadd_48"
  let candidate_vsub_46_ptr : RegionName := "candidate_vsub_46"
  let candidate_vgt_25_ptr : RegionName := "candidate_vgt_25"
  let candidate_vmasked_fill_25_ptr : RegionName := "candidate_vmasked_fill_25"
  let candidate_vsoftmax_19_ptr : RegionName := "candidate_vsoftmax_19"
  let candidate_vto_copy_161_ptr : RegionName := "candidate_vto_copy_161"
  let candidate_vbmm_81_ptr : RegionName := "candidate_vbmm_81"
  let candidate_vlogsumexp_19_ptr : RegionName := "candidate_vlogsumexp_19"
  let candidate_vto_copy_162_ptr : RegionName := "candidate_vto_copy_162"
  let candidate_vto_copy_163_ptr : RegionName := "candidate_vto_copy_163"
  let candidate_vbmm_82_ptr : RegionName := "candidate_vbmm_82"
  let candidate_vmul_73_ptr : RegionName := "candidate_vmul_73"
  let candidate_vadd_49_ptr : RegionName := "candidate_vadd_49"
  let candidate_vsub_47_ptr : RegionName := "candidate_vsub_47"
  let candidate_vgt_26_ptr : RegionName := "candidate_vgt_26"
  let candidate_vmasked_fill_26_ptr : RegionName := "candidate_vmasked_fill_26"
  let candidate_vsoftmax_20_ptr : RegionName := "candidate_vsoftmax_20"
  let candidate_vto_copy_164_ptr : RegionName := "candidate_vto_copy_164"
  let candidate_vbmm_83_ptr : RegionName := "candidate_vbmm_83"
  let candidate_vlogsumexp_20_ptr : RegionName := "candidate_vlogsumexp_20"
  let candidate_vcat_41_ptr : RegionName := "candidate_vcat_41"
  let candidate_vcat_42_ptr : RegionName := "candidate_vcat_42"
  let candidate_vto_copy_165_ptr : RegionName := "candidate_vto_copy_165"
  let candidate_vto_copy_166_ptr : RegionName := "candidate_vto_copy_166"
  let candidate_vto_copy_167_ptr : RegionName := "candidate_vto_copy_167"
  let candidate_vto_copy_168_ptr : RegionName := "candidate_vto_copy_168"
  let candidate_vto_copy_169_ptr : RegionName := "candidate_vto_copy_169"
  let candidate_vbmm_84_ptr : RegionName := "candidate_vbmm_84"
  let candidate_vmul_74_ptr : RegionName := "candidate_vmul_74"
  let candidate_vadd_50_ptr : RegionName := "candidate_vadd_50"
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
  let candidate_vbmm_87_ptr : RegionName := "candidate_vbmm_87"
  let candidate_vmul_78_ptr : RegionName := "candidate_vmul_78"
  let candidate_vto_copy_171_ptr : RegionName := "candidate_vto_copy_171"
  let candidate_vbmm_88_ptr : RegionName := "candidate_vbmm_88"
  let candidate_vto_copy_172_ptr : RegionName := "candidate_vto_copy_172"
  let candidate_vto_copy_173_ptr : RegionName := "candidate_vto_copy_173"
  let candidate_vto_copy_174_ptr : RegionName := "candidate_vto_copy_174"
  let candidate_vto_copy_175_ptr : RegionName := "candidate_vto_copy_175"
  let candidate_vto_copy_176_ptr : RegionName := "candidate_vto_copy_176"
  let candidate_vto_copy_177_ptr : RegionName := "candidate_vto_copy_177"
  let candidate_vbmm_89_ptr : RegionName := "candidate_vbmm_89"
  let candidate_vmul_79_ptr : RegionName := "candidate_vmul_79"
  let candidate_vadd_51_ptr : RegionName := "candidate_vadd_51"
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
  let candidate_vbmm_92_ptr : RegionName := "candidate_vbmm_92"
  let candidate_vmul_83_ptr : RegionName := "candidate_vmul_83"
  let candidate_vto_copy_179_ptr : RegionName := "candidate_vto_copy_179"
  let candidate_vbmm_93_ptr : RegionName := "candidate_vbmm_93"
  let candidate_vto_copy_180_ptr : RegionName := "candidate_vto_copy_180"
  let candidate_vto_copy_181_ptr : RegionName := "candidate_vto_copy_181"
  let candidate_vto_copy_182_ptr : RegionName := "candidate_vto_copy_182"
  let candidate_vto_copy_183_ptr : RegionName := "candidate_vto_copy_183"
  let candidate_vto_copy_184_ptr : RegionName := "candidate_vto_copy_184"
  let candidate_vto_copy_185_ptr : RegionName := "candidate_vto_copy_185"
  let candidate_vbmm_94_ptr : RegionName := "candidate_vbmm_94"
  let candidate_vmul_84_ptr : RegionName := "candidate_vmul_84"
  let candidate_vadd_52_ptr : RegionName := "candidate_vadd_52"
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
  let candidate_vbmm_97_ptr : RegionName := "candidate_vbmm_97"
  let candidate_vmul_88_ptr : RegionName := "candidate_vmul_88"
  let candidate_vto_copy_187_ptr : RegionName := "candidate_vto_copy_187"
  let candidate_vbmm_98_ptr : RegionName := "candidate_vbmm_98"
  let candidate_vto_copy_188_ptr : RegionName := "candidate_vto_copy_188"
  let candidate_vcat_44_ptr : RegionName := "candidate_vcat_44"
  let candidate_vcat_45_ptr : RegionName := "candidate_vcat_45"
  let candidate_vbmm_101_ptr : RegionName := "candidate_vbmm_101"
  let candidate_vbmm_102_ptr : RegionName := "candidate_vbmm_102"
  let candidate_vcat_46_ptr : RegionName := "candidate_vcat_46"
  let candidate_vadd_54_ptr : RegionName := "candidate_vadd_54"
  let candidate_vbmm_103_ptr : RegionName := "candidate_vbmm_103"
  let candidate_vbmm_104_ptr : RegionName := "candidate_vbmm_104"
  let candidate_vcat_47_ptr : RegionName := "candidate_vcat_47"
  let candidate_vcat_48_ptr : RegionName := "candidate_vcat_48"
  let candidate_vto_copy_189_ptr : RegionName := "candidate_vto_copy_189"
  let candidate_vto_copy_190_ptr : RegionName := "candidate_vto_copy_190"
  let candidate_vbmm_105_ptr : RegionName := "candidate_vbmm_105"
  let candidate_vmul_89_ptr : RegionName := "candidate_vmul_89"
  let candidate_vadd_55_ptr : RegionName := "candidate_vadd_55"
  let candidate_vsub_57_ptr : RegionName := "candidate_vsub_57"
  let candidate_vgt_30_ptr : RegionName := "candidate_vgt_30"
  let candidate_vmasked_fill_30_ptr : RegionName := "candidate_vmasked_fill_30"
  let candidate_vsoftmax_21_ptr : RegionName := "candidate_vsoftmax_21"
  let candidate_vto_copy_191_ptr : RegionName := "candidate_vto_copy_191"
  let candidate_vbmm_106_ptr : RegionName := "candidate_vbmm_106"
  let candidate_vlogsumexp_21_ptr : RegionName := "candidate_vlogsumexp_21"
  let candidate_vto_copy_192_ptr : RegionName := "candidate_vto_copy_192"
  let candidate_vto_copy_193_ptr : RegionName := "candidate_vto_copy_193"
  let candidate_vbmm_107_ptr : RegionName := "candidate_vbmm_107"
  let candidate_vmul_90_ptr : RegionName := "candidate_vmul_90"
  let candidate_vadd_56_ptr : RegionName := "candidate_vadd_56"
  let candidate_vsub_58_ptr : RegionName := "candidate_vsub_58"
  let candidate_vgt_31_ptr : RegionName := "candidate_vgt_31"
  let candidate_vmasked_fill_31_ptr : RegionName := "candidate_vmasked_fill_31"
  let candidate_vsoftmax_22_ptr : RegionName := "candidate_vsoftmax_22"
  let candidate_vto_copy_194_ptr : RegionName := "candidate_vto_copy_194"
  let candidate_vbmm_108_ptr : RegionName := "candidate_vbmm_108"
  let candidate_vlogsumexp_22_ptr : RegionName := "candidate_vlogsumexp_22"
  let candidate_vto_copy_195_ptr : RegionName := "candidate_vto_copy_195"
  let candidate_vto_copy_196_ptr : RegionName := "candidate_vto_copy_196"
  let candidate_vbmm_109_ptr : RegionName := "candidate_vbmm_109"
  let candidate_vmul_91_ptr : RegionName := "candidate_vmul_91"
  let candidate_vadd_57_ptr : RegionName := "candidate_vadd_57"
  let candidate_vsub_59_ptr : RegionName := "candidate_vsub_59"
  let candidate_vgt_32_ptr : RegionName := "candidate_vgt_32"
  let candidate_vmasked_fill_32_ptr : RegionName := "candidate_vmasked_fill_32"
  let candidate_vsoftmax_23_ptr : RegionName := "candidate_vsoftmax_23"
  let candidate_vto_copy_197_ptr : RegionName := "candidate_vto_copy_197"
  let candidate_vbmm_110_ptr : RegionName := "candidate_vbmm_110"
  let candidate_vlogsumexp_23_ptr : RegionName := "candidate_vlogsumexp_23"
  let candidate_vcat_49_ptr : RegionName := "candidate_vcat_49"
  let candidate_vcat_50_ptr : RegionName := "candidate_vcat_50"
  let candidate_vto_copy_198_ptr : RegionName := "candidate_vto_copy_198"
  let candidate_vto_copy_199_ptr : RegionName := "candidate_vto_copy_199"
  let candidate_vto_copy_200_ptr : RegionName := "candidate_vto_copy_200"
  let candidate_vto_copy_201_ptr : RegionName := "candidate_vto_copy_201"
  let candidate_vto_copy_202_ptr : RegionName := "candidate_vto_copy_202"
  let candidate_vbmm_111_ptr : RegionName := "candidate_vbmm_111"
  let candidate_vmul_92_ptr : RegionName := "candidate_vmul_92"
  let candidate_vadd_58_ptr : RegionName := "candidate_vadd_58"
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
  let candidate_vbmm_114_ptr : RegionName := "candidate_vbmm_114"
  let candidate_vmul_96_ptr : RegionName := "candidate_vmul_96"
  let candidate_vto_copy_204_ptr : RegionName := "candidate_vto_copy_204"
  let candidate_vbmm_115_ptr : RegionName := "candidate_vbmm_115"
  let candidate_vto_copy_205_ptr : RegionName := "candidate_vto_copy_205"
  let candidate_vcat_51_ptr : RegionName := "candidate_vcat_51"
  let candidate_vcat_52_ptr : RegionName := "candidate_vcat_52"
  let candidate_vto_copy_206_ptr : RegionName := "candidate_vto_copy_206"
  let candidate_vto_copy_207_ptr : RegionName := "candidate_vto_copy_207"
  let candidate_vto_copy_208_ptr : RegionName := "candidate_vto_copy_208"
  let candidate_vto_copy_209_ptr : RegionName := "candidate_vto_copy_209"
  let candidate_vto_copy_210_ptr : RegionName := "candidate_vto_copy_210"
  let candidate_vbmm_116_ptr : RegionName := "candidate_vbmm_116"
  let candidate_vmul_97_ptr : RegionName := "candidate_vmul_97"
  let candidate_vadd_59_ptr : RegionName := "candidate_vadd_59"
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
  let candidate_vbmm_119_ptr : RegionName := "candidate_vbmm_119"
  let candidate_vmul_101_ptr : RegionName := "candidate_vmul_101"
  let candidate_vto_copy_212_ptr : RegionName := "candidate_vto_copy_212"
  let candidate_vbmm_120_ptr : RegionName := "candidate_vbmm_120"
  let candidate_vto_copy_213_ptr : RegionName := "candidate_vto_copy_213"
  let candidate_vcat_53_ptr : RegionName := "candidate_vcat_53"
  let candidate_vcat_54_ptr : RegionName := "candidate_vcat_54"
  let candidate_vto_copy_214_ptr : RegionName := "candidate_vto_copy_214"
  let candidate_vto_copy_215_ptr : RegionName := "candidate_vto_copy_215"
  let candidate_vto_copy_216_ptr : RegionName := "candidate_vto_copy_216"
  let candidate_vto_copy_217_ptr : RegionName := "candidate_vto_copy_217"
  let candidate_vto_copy_218_ptr : RegionName := "candidate_vto_copy_218"
  let candidate_vbmm_121_ptr : RegionName := "candidate_vbmm_121"
  let candidate_vmul_102_ptr : RegionName := "candidate_vmul_102"
  let candidate_vadd_60_ptr : RegionName := "candidate_vadd_60"
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
  let candidate_vbmm_124_ptr : RegionName := "candidate_vbmm_124"
  let candidate_vmul_106_ptr : RegionName := "candidate_vmul_106"
  let candidate_vto_copy_220_ptr : RegionName := "candidate_vto_copy_220"
  let candidate_vbmm_125_ptr : RegionName := "candidate_vbmm_125"
  let candidate_vto_copy_221_ptr : RegionName := "candidate_vto_copy_221"
  let candidate_vcat_55_ptr : RegionName := "candidate_vcat_55"
  let candidate_vcat_56_ptr : RegionName := "candidate_vcat_56"
  let candidate_vcat_58_ptr : RegionName := "candidate_vcat_58"
  let candidate_vcat_59_ptr : RegionName := "candidate_vcat_59"
  let candidate_vbmm_128_ptr : RegionName := "candidate_vbmm_128"
  let candidate_vbmm_129_ptr : RegionName := "candidate_vbmm_129"
  let candidate_vcat_60_ptr : RegionName := "candidate_vcat_60"
  let candidate_vadd_62_ptr : RegionName := "candidate_vadd_62"
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
    r21_z = tl.arange($(1048576))
    r21_value = tl.load($(candidate_vmul_5_ptr) + ((((r21_z / $(512)) % $(2048)) * $(512)) + (r21_z % $(512))), dtype=tl.float32)
    r21_value = r21_value.to(tl.bfloat16)
    r21_value = tl.broadcast_to(r21_value, [$(1048576)])
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
    tl.store($(candidate_vto_copy_23_ptr) + r34_z, r34_value)
    r35_z = tl.arange($(65536))
    r35_value = ((tl.load($(candidate_vto_copy_23_ptr) + ((((r35_z / $(512)) % $(128)) * $(512)) + (r35_z % $(512))), dtype=tl.float32)) * (tl.load($(candidate_vto_copy_23_ptr) + ((((r35_z / $(512)) % $(128)) * $(512)) + (r35_z % $(512))), dtype=tl.float32)))
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
    r39_value = (tl.load($(candidate_vto_copy_23_ptr) + ((((r39_z / $(512)) % $(128)) * $(512)) + (r39_z % $(512))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_1_ptr) + ((r39_z / $(512)) % $(128)), dtype=tl.float32))
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
    r64_value = (tl.load($(candidate_vmm_2_ptr) + ((((r64_z / $(64)) % $(128)) * $(576)) + (r64_z % $(64)) + $(512)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_28_ptr) + ((((r64_z / $(64)) % $(128)) * $(64)) + (r64_z % $(64))), dtype=tl.bfloat16))
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
    r67_z = tl.arange($(73728))
    r67_value = tl.where((r67_z % $(576)) < $(512), tl.load($(candidate_vto_copy_24_ptr) + ((((((((r67_z / $(576)) % $(128)) * $(512)) + ((r67_z % $(576)) - $(0))) / $(512)) % $(128)) * $(512)) + (((((r67_z / $(576)) % $(128)) * $(512)) + ((r67_z % $(576)) - $(0))) % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vadd_7_ptr) + ((((((((r67_z / $(576)) % $(128)) * $(64)) + ((r67_z % $(576)) - $(512))) / $(64)) % $(128)) * $(64)) + (((((r67_z / $(576)) % $(128)) * $(64)) + ((r67_z % $(576)) - $(512))) % $(64))), dtype=tl.bfloat16))
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
      r71_right = tl.load($(candidate_vto_copy_20_ptr) + ((((((r71_lane[:, None] * $(1024)) + r71_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r71_lane[:, None] * $(1024)) + r71_col[None, :]) % $(128)) * $(512)) + ((((r71_lane[:, None] * $(1024)) + r71_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r71_value = tl.dot(r71_left, r71_right)
      r71_value = r71_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_22_ptr) + ((r71_group * $(131072)) + (r71_row[:, None] * $(1024)) + r71_col[None, :]), r71_value)
    }
    r72_row = tl.arange($(128))
    r72_col = tl.arange($(1024))
    r72_lane = tl.arange($(512))
    for r72_group in range($(1)) {
      r72_left = tl.load($(candidate_vindex_2_ptr) + ((((((r72_row[:, None] * $(512)) + r72_lane[None, :]) / $(512)) % $(128)) * $(576)) + (((r72_row[:, None] * $(512)) + r72_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r72_right = tl.load($(candidate_vto_copy_20_ptr) + ((((((r72_lane[:, None] * $(1024)) + r72_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r72_lane[:, None] * $(1024)) + r72_col[None, :]) % $(128)) * $(512)) + ((((r72_lane[:, None] * $(1024)) + r72_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
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
    tl.store($(candidate_vto_copy_90_ptr) + r75_z, r75_value)
    r76_z = tl.arange($(55296))
    r76_value = tl.load($(candidate_vcat_12_ptr) + ((((r76_z / $(192)) % $(36)) * $(1536)) + (((r76_z / $(6912)) % $(8)) * $(192)) + (r76_z % $(192))), dtype=tl.bfloat16)
    r76_value = r76_value.to(tl.float32)
    r76_value = tl.broadcast_to(r76_value, [$(55296)])
    tl.store($(candidate_vto_copy_91_ptr) + r76_z, r76_value)
    r77_row = tl.arange($(12))
    r77_col = tl.arange($(36))
    r77_lane = tl.arange($(192))
    for r77_group in range($(8)) {
      r77_left = tl.load($(candidate_vto_copy_90_ptr) + (((((((r77_group % $(8)) * $(2304)) + (r77_row[:, None] * $(192)) + r77_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r77_group % $(8)) * $(2304)) + (r77_row[:, None] * $(192)) + r77_lane[None, :]) / $(192)) % $(12)) * $(192)) + ((((r77_group % $(8)) * $(2304)) + (r77_row[:, None] * $(192)) + r77_lane[None, :]) % $(192))), dtype=tl.float32)
      r77_right = tl.load($(candidate_vto_copy_91_ptr) + (((((((r77_group % $(8)) * $(6912)) + (r77_lane[:, None] * $(36)) + r77_col[None, :]) / $(6912)) % $(8)) * $(6912)) + (((((r77_group % $(8)) * $(6912)) + (r77_lane[:, None] * $(36)) + r77_col[None, :]) % $(36)) * $(192)) + (((((r77_group % $(8)) * $(6912)) + (r77_lane[:, None] * $(36)) + r77_col[None, :]) / $(36)) % $(192))), dtype=tl.float32)
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
    tl.store($(candidate_vto_copy_92_ptr) + r84_z, r84_value)
    r85_row = tl.arange($(12))
    r85_col = tl.arange($(128))
    r85_lane = tl.arange($(36))
    for r85_group in range($(8)) {
      r85_left = tl.load($(candidate_vto_copy_92_ptr) + (((((((r85_group % $(8)) * $(432)) + (r85_row[:, None] * $(36)) + r85_lane[None, :]) / $(432)) % $(8)) * $(432)) + ((((((r85_group % $(8)) * $(432)) + (r85_row[:, None] * $(36)) + r85_lane[None, :]) / $(36)) % $(12)) * $(36)) + ((((r85_group % $(8)) * $(432)) + (r85_row[:, None] * $(36)) + r85_lane[None, :]) % $(36))), dtype=tl.bfloat16)
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
    tl.store($(candidate_vto_copy_93_ptr) + r87_z, r87_value)
    r88_z = tl.arange($(18432))
    r88_value = tl.load($(candidate_vcat_12_ptr) + ((((r88_z / $(192)) % $(12)) * $(1536)) + (((r88_z / $(2304)) % $(8)) * $(192)) + (r88_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r88_value = r88_value.to(tl.float32)
    r88_value = tl.broadcast_to(r88_value, [$(18432)])
    tl.store($(candidate_vto_copy_94_ptr) + r88_z, r88_value)
    r89_row = tl.arange($(4))
    r89_col = tl.arange($(12))
    r89_lane = tl.arange($(192))
    for r89_group in range($(8)) {
      r89_left = tl.load($(candidate_vto_copy_93_ptr) + (((((((r89_group % $(8)) * $(768)) + (r89_row[:, None] * $(192)) + r89_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r89_group % $(8)) * $(768)) + (r89_row[:, None] * $(192)) + r89_lane[None, :]) / $(192)) % $(4)) * $(192)) + ((((r89_group % $(8)) * $(768)) + (r89_row[:, None] * $(192)) + r89_lane[None, :]) % $(192))), dtype=tl.float32)
      r89_right = tl.load($(candidate_vto_copy_94_ptr) + (((((((r89_group % $(8)) * $(2304)) + (r89_lane[:, None] * $(12)) + r89_col[None, :]) / $(2304)) % $(8)) * $(2304)) + (((((r89_group % $(8)) * $(2304)) + (r89_lane[:, None] * $(12)) + r89_col[None, :]) % $(12)) * $(192)) + (((((r89_group % $(8)) * $(2304)) + (r89_lane[:, None] * $(12)) + r89_col[None, :]) / $(12)) % $(192))), dtype=tl.float32)
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
    tl.store($(candidate_vto_copy_95_ptr) + r96_z, r96_value)
    r97_row = tl.arange($(4))
    r97_col = tl.arange($(128))
    r97_lane = tl.arange($(12))
    for r97_group in range($(8)) {
      r97_left = tl.load($(candidate_vto_copy_95_ptr) + (((((((r97_group % $(8)) * $(48)) + (r97_row[:, None] * $(12)) + r97_lane[None, :]) / $(48)) % $(8)) * $(48)) + ((((((r97_group % $(8)) * $(48)) + (r97_row[:, None] * $(12)) + r97_lane[None, :]) / $(12)) % $(4)) * $(12)) + ((((r97_group % $(8)) * $(48)) + (r97_row[:, None] * $(12)) + r97_lane[None, :]) % $(12))), dtype=tl.bfloat16)
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
    tl.store($(candidate_vto_copy_96_ptr) + r99_z, r99_value)
    r100_z = tl.arange($(73728))
    r100_value = tl.load($(candidate_vcat_12_ptr) + ((((r100_z / $(192)) % $(48)) * $(1536)) + (((r100_z / $(9216)) % $(8)) * $(192)) + (r100_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r100_value = r100_value.to(tl.float32)
    r100_value = tl.broadcast_to(r100_value, [$(73728)])
    tl.store($(candidate_vto_copy_97_ptr) + r100_z, r100_value)
    r101_row = tl.arange($(16))
    r101_col = tl.arange($(48))
    r101_lane = tl.arange($(192))
    for r101_group in range($(8)) {
      r101_left = tl.load($(candidate_vto_copy_96_ptr) + (((((((r101_group % $(8)) * $(3072)) + (r101_row[:, None] * $(192)) + r101_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r101_group % $(8)) * $(3072)) + (r101_row[:, None] * $(192)) + r101_lane[None, :]) / $(192)) % $(16)) * $(192)) + ((((r101_group % $(8)) * $(3072)) + (r101_row[:, None] * $(192)) + r101_lane[None, :]) % $(192))), dtype=tl.float32)
      r101_right = tl.load($(candidate_vto_copy_97_ptr) + (((((((r101_group % $(8)) * $(9216)) + (r101_lane[:, None] * $(48)) + r101_col[None, :]) / $(9216)) % $(8)) * $(9216)) + (((((r101_group % $(8)) * $(9216)) + (r101_lane[:, None] * $(48)) + r101_col[None, :]) % $(48)) * $(192)) + (((((r101_group % $(8)) * $(9216)) + (r101_lane[:, None] * $(48)) + r101_col[None, :]) / $(48)) % $(192))), dtype=tl.float32)
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
    tl.store($(candidate_vto_copy_98_ptr) + r108_z, r108_value)
    r109_row = tl.arange($(16))
    r109_col = tl.arange($(128))
    r109_lane = tl.arange($(48))
    for r109_group in range($(8)) {
      r109_left = tl.load($(candidate_vto_copy_98_ptr) + (((((((r109_group % $(8)) * $(768)) + (r109_row[:, None] * $(48)) + r109_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r109_group % $(8)) * $(768)) + (r109_row[:, None] * $(48)) + r109_lane[None, :]) / $(48)) % $(16)) * $(48)) + ((((r109_group % $(8)) * $(768)) + (r109_row[:, None] * $(48)) + r109_lane[None, :]) % $(48))), dtype=tl.bfloat16)
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
    tl.store($(candidate_vto_copy_99_ptr) + r113_z, r113_value)
    r114_z = tl.arange($(55296))
    r114_value = tl.load($(candidate_vcat_12_ptr) + ((((r114_z / $(192)) % $(36)) * $(1536)) + (((r114_z / $(6912)) % $(8)) * $(192)) + (r114_z % $(192))), dtype=tl.bfloat16)
    r114_value = r114_value.to(tl.float32)
    r114_value = tl.broadcast_to(r114_value, [$(55296)])
    tl.store($(candidate_vto_copy_100_ptr) + r114_z, r114_value)
    r115_z = tl.arange($(36864))
    r115_value = tl.load($(candidate_vbmm_23_ptr) + ((((r115_z / $(128)) % $(36)) * $(1024)) + (((r115_z / $(4608)) % $(8)) * $(128)) + (r115_z % $(128))), dtype=tl.bfloat16)
    r115_value = r115_value.to(tl.float32)
    r115_value = tl.broadcast_to(r115_value, [$(36864)])
    tl.store($(candidate_vto_copy_101_ptr) + r115_z, r115_value)
    r116_z = tl.arange($(12288))
    r116_value = tl.load($(candidate_vmm_28_ptr) + ((((r116_z / $(128)) % $(12)) * $(1024)) + (((r116_z / $(1536)) % $(8)) * $(128)) + (r116_z % $(128)) + $(98304)), dtype=tl.bfloat16)
    r116_value = r116_value.to(tl.float32)
    r116_value = tl.broadcast_to(r116_value, [$(12288)])
    tl.store($(candidate_vto_copy_102_ptr) + r116_z, r116_value)
    r117_z = tl.arange($(12288))
    r117_value = tl.load($(candidate_vcat_13_ptr) + ((((r117_z / $(128)) % $(12)) * $(1024)) + (((r117_z / $(1536)) % $(8)) * $(128)) + (r117_z % $(128))), dtype=tl.bfloat16)
    r117_value = r117_value.to(tl.float32)
    r117_value = tl.broadcast_to(r117_value, [$(12288)])
    tl.store($(candidate_vto_copy_103_ptr) + r117_z, r117_value)
    r118_row = tl.arange($(12))
    r118_col = tl.arange($(36))
    r118_lane = tl.arange($(192))
    for r118_group in range($(8)) {
      r118_left = tl.load($(candidate_vto_copy_99_ptr) + (((((((r118_group % $(8)) * $(2304)) + (r118_row[:, None] * $(192)) + r118_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r118_group % $(8)) * $(2304)) + (r118_row[:, None] * $(192)) + r118_lane[None, :]) / $(192)) % $(12)) * $(192)) + ((((r118_group % $(8)) * $(2304)) + (r118_row[:, None] * $(192)) + r118_lane[None, :]) % $(192))), dtype=tl.float32)
      r118_right = tl.load($(candidate_vto_copy_100_ptr) + (((((((r118_group % $(8)) * $(6912)) + (r118_lane[:, None] * $(36)) + r118_col[None, :]) / $(6912)) % $(8)) * $(6912)) + (((((r118_group % $(8)) * $(6912)) + (r118_lane[:, None] * $(36)) + r118_col[None, :]) % $(36)) * $(192)) + (((((r118_group % $(8)) * $(6912)) + (r118_lane[:, None] * $(36)) + r118_col[None, :]) / $(36)) % $(192))), dtype=tl.float32)
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
    r126_value = (tl.load($(candidate_vto_copy_102_ptr) + ((((r126_z / $(1536)) % $(8)) * $(1536)) + (((r126_z / $(128)) % $(12)) * $(128)) + (r126_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_103_ptr) + ((((r126_z / $(1536)) % $(8)) * $(1536)) + (((r126_z / $(128)) % $(12)) * $(128)) + (r126_z % $(128))), dtype=tl.float32))
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
      r128_left = tl.load($(candidate_vto_copy_102_ptr) + (((((((r128_group % $(8)) * $(1536)) + (r128_row[:, None] * $(128)) + r128_lane[None, :]) / $(1536)) % $(8)) * $(1536)) + ((((((r128_group % $(8)) * $(1536)) + (r128_row[:, None] * $(128)) + r128_lane[None, :]) / $(128)) % $(12)) * $(128)) + ((((r128_group % $(8)) * $(1536)) + (r128_row[:, None] * $(128)) + r128_lane[None, :]) % $(128))), dtype=tl.float32)
      r128_right = tl.load($(candidate_vto_copy_101_ptr) + (((((((r128_group % $(8)) * $(4608)) + (r128_lane[:, None] * $(36)) + r128_col[None, :]) / $(4608)) % $(8)) * $(4608)) + (((((r128_group % $(8)) * $(4608)) + (r128_lane[:, None] * $(36)) + r128_col[None, :]) % $(36)) * $(128)) + (((((r128_group % $(8)) * $(4608)) + (r128_lane[:, None] * $(36)) + r128_col[None, :]) / $(36)) % $(128))), dtype=tl.float32)
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
    r131_row = tl.arange($(36))
    r131_col = tl.arange($(192))
    r131_lane = tl.arange($(12))
    for r131_group in range($(8)) {
      r131_left = tl.load($(candidate_vmul_40_ptr) + (((((((r131_group % $(8)) * $(432)) + (r131_row[:, None] * $(12)) + r131_lane[None, :]) / $(432)) % $(8)) * $(432)) + (((((r131_group % $(8)) * $(432)) + (r131_row[:, None] * $(12)) + r131_lane[None, :]) % $(12)) * $(36)) + (((((r131_group % $(8)) * $(432)) + (r131_row[:, None] * $(12)) + r131_lane[None, :]) / $(12)) % $(36))), dtype=tl.float32)
      r131_right = tl.load($(candidate_vto_copy_99_ptr) + (((((((r131_group % $(8)) * $(2304)) + (r131_lane[:, None] * $(192)) + r131_col[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r131_group % $(8)) * $(2304)) + (r131_lane[:, None] * $(192)) + r131_col[None, :]) / $(192)) % $(12)) * $(192)) + ((((r131_group % $(8)) * $(2304)) + (r131_lane[:, None] * $(192)) + r131_col[None, :]) % $(192))), dtype=tl.float32)
      r131_value = tl.dot(r131_left, r131_right)
      r131_value = r131_value.to(tl.float32)
      tl.store($(candidate_vbmm_33_ptr) + ((r131_group * $(6912)) + (r131_row[:, None] * $(192)) + r131_col[None, :]), r131_value)
    }
    r132_z = tl.arange($(55296))
    r132_value = (tl.load($(candidate_vbmm_33_ptr) + ((((r132_z / $(6912)) % $(8)) * $(6912)) + (((r132_z / $(192)) % $(36)) * $(192)) + (r132_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r132_value = r132_value.to(tl.float32)
    r132_value = tl.broadcast_to(r132_value, [$(55296)])
    tl.store($(candidate_vmul_42_ptr) + r132_z, r132_value)
    r133_z = tl.arange($(55296))
    r133_value = tl.load($(candidate_vmul_42_ptr) + ((((r133_z / $(192)) % $(8)) * $(6912)) + (((r133_z / $(1536)) % $(36)) * $(192)) + (r133_z % $(192))), dtype=tl.float32)
    r133_value = r133_value.to(tl.bfloat16)
    r133_value = tl.broadcast_to(r133_value, [$(55296)])
    tl.store($(candidate_vto_copy_105_ptr) + r133_z, r133_value)
    r134_row = tl.arange($(36))
    r134_col = tl.arange($(128))
    r134_lane = tl.arange($(12))
    for r134_group in range($(8)) {
      r134_left = tl.load($(candidate_vexp_ptr) + (((((((r134_group % $(8)) * $(432)) + (r134_row[:, None] * $(12)) + r134_lane[None, :]) / $(432)) % $(8)) * $(432)) + (((((r134_group % $(8)) * $(432)) + (r134_row[:, None] * $(12)) + r134_lane[None, :]) % $(12)) * $(36)) + (((((r134_group % $(8)) * $(432)) + (r134_row[:, None] * $(12)) + r134_lane[None, :]) / $(12)) % $(36))), dtype=tl.float32)
      r134_right = tl.load($(candidate_vto_copy_102_ptr) + (((((((r134_group % $(8)) * $(1536)) + (r134_lane[:, None] * $(128)) + r134_col[None, :]) / $(1536)) % $(8)) * $(1536)) + ((((((r134_group % $(8)) * $(1536)) + (r134_lane[:, None] * $(128)) + r134_col[None, :]) / $(128)) % $(12)) * $(128)) + ((((r134_group % $(8)) * $(1536)) + (r134_lane[:, None] * $(128)) + r134_col[None, :]) % $(128))), dtype=tl.float32)
      r134_value = tl.dot(r134_left, r134_right)
      r134_value = r134_value.to(tl.float32)
      tl.store($(candidate_vbmm_34_ptr) + ((r134_group * $(4608)) + (r134_row[:, None] * $(128)) + r134_col[None, :]), r134_value)
    }
    r135_z = tl.arange($(36864))
    r135_value = tl.load($(candidate_vbmm_34_ptr) + ((((r135_z / $(128)) % $(8)) * $(4608)) + (((r135_z / $(1024)) % $(36)) * $(128)) + (r135_z % $(128))), dtype=tl.float32)
    r135_value = r135_value.to(tl.bfloat16)
    r135_value = tl.broadcast_to(r135_value, [$(36864)])
    tl.store($(candidate_vto_copy_106_ptr) + r135_z, r135_value)
    r136_z = tl.arange($(73728))
    r136_value = tl.where(((r136_z / $(1536)) % $(48)) < $(36), tl.load($(candidate_vto_copy_105_ptr) + ((((((r136_z / $(1536)) % $(48)) - $(0)) % $(36)) * $(1536)) + (((r136_z / $(192)) % $(8)) * $(192)) + (r136_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r136_value = r136_value.to(tl.bfloat16)
    r136_value = tl.broadcast_to(r136_value, [$(73728)])
    tl.store($(candidate_vcat_15_ptr) + r136_z, r136_value)
    r137_z = tl.arange($(49152))
    r137_value = tl.where(((r137_z / $(1024)) % $(48)) < $(36), tl.load($(candidate_vto_copy_106_ptr) + ((((((r137_z / $(1024)) % $(48)) - $(0)) % $(36)) * $(1024)) + (((r137_z / $(128)) % $(8)) * $(128)) + (r137_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r137_value = r137_value.to(tl.bfloat16)
    r137_value = tl.broadcast_to(r137_value, [$(49152)])
    tl.store($(candidate_vcat_16_ptr) + r137_z, r137_value)
    r138_z = tl.arange($(6144))
    r138_value = tl.load($(candidate_vcat_11_ptr) + ((((r138_z / $(192)) % $(4)) * $(1536)) + (((r138_z / $(768)) % $(8)) * $(192)) + (r138_z % $(192)) + $(18432)), dtype=tl.bfloat16)
    r138_value = r138_value.to(tl.float32)
    r138_value = tl.broadcast_to(r138_value, [$(6144)])
    tl.store($(candidate_vto_copy_107_ptr) + r138_z, r138_value)
    r139_z = tl.arange($(18432))
    r139_value = tl.load($(candidate_vcat_12_ptr) + ((((r139_z / $(192)) % $(12)) * $(1536)) + (((r139_z / $(2304)) % $(8)) * $(192)) + (r139_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r139_value = r139_value.to(tl.float32)
    r139_value = tl.broadcast_to(r139_value, [$(18432)])
    tl.store($(candidate_vto_copy_108_ptr) + r139_z, r139_value)
    r140_z = tl.arange($(12288))
    r140_value = tl.load($(candidate_vbmm_23_ptr) + ((((r140_z / $(128)) % $(12)) * $(1024)) + (((r140_z / $(1536)) % $(8)) * $(128)) + (r140_z % $(128)) + $(49152)), dtype=tl.bfloat16)
    r140_value = r140_value.to(tl.float32)
    r140_value = tl.broadcast_to(r140_value, [$(12288)])
    tl.store($(candidate_vto_copy_109_ptr) + r140_z, r140_value)
    r141_z = tl.arange($(4096))
    r141_value = tl.load($(candidate_vmm_28_ptr) + ((((r141_z / $(128)) % $(4)) * $(1024)) + (((r141_z / $(512)) % $(8)) * $(128)) + (r141_z % $(128)) + $(110592)), dtype=tl.bfloat16)
    r141_value = r141_value.to(tl.float32)
    r141_value = tl.broadcast_to(r141_value, [$(4096)])
    tl.store($(candidate_vto_copy_110_ptr) + r141_z, r141_value)
    r142_z = tl.arange($(4096))
    r142_value = tl.load($(candidate_vcat_13_ptr) + ((((r142_z / $(128)) % $(4)) * $(1024)) + (((r142_z / $(512)) % $(8)) * $(128)) + (r142_z % $(128)) + $(12288)), dtype=tl.bfloat16)
    r142_value = r142_value.to(tl.float32)
    r142_value = tl.broadcast_to(r142_value, [$(4096)])
    tl.store($(candidate_vto_copy_111_ptr) + r142_z, r142_value)
    r143_row = tl.arange($(4))
    r143_col = tl.arange($(12))
    r143_lane = tl.arange($(192))
    for r143_group in range($(8)) {
      r143_left = tl.load($(candidate_vto_copy_107_ptr) + (((((((r143_group % $(8)) * $(768)) + (r143_row[:, None] * $(192)) + r143_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r143_group % $(8)) * $(768)) + (r143_row[:, None] * $(192)) + r143_lane[None, :]) / $(192)) % $(4)) * $(192)) + ((((r143_group % $(8)) * $(768)) + (r143_row[:, None] * $(192)) + r143_lane[None, :]) % $(192))), dtype=tl.float32)
      r143_right = tl.load($(candidate_vto_copy_108_ptr) + (((((((r143_group % $(8)) * $(2304)) + (r143_lane[:, None] * $(12)) + r143_col[None, :]) / $(2304)) % $(8)) * $(2304)) + (((((r143_group % $(8)) * $(2304)) + (r143_lane[:, None] * $(12)) + r143_col[None, :]) % $(12)) * $(192)) + (((((r143_group % $(8)) * $(2304)) + (r143_lane[:, None] * $(12)) + r143_col[None, :]) / $(12)) % $(192))), dtype=tl.float32)
      r143_value = tl.dot(r143_left, r143_right)
      r143_value = r143_value.to(tl.float32)
      tl.store($(candidate_vbmm_35_ptr) + ((r143_group * $(48)) + (r143_row[:, None] * $(12)) + r143_col[None, :]), r143_value)
    }
    r144_z = tl.arange($(384))
    r144_value = (tl.load($(candidate_vbmm_35_ptr) + ((((r144_z / $(48)) % $(8)) * $(48)) + (((r144_z / $(12)) % $(4)) * $(12)) + (r144_z % $(12))), dtype=tl.float32) * 0.07216878364870322)
    r144_value = r144_value.to(tl.float32)
    r144_value = tl.broadcast_to(r144_value, [$(384)])
    tl.store($(candidate_vmul_43_ptr) + r144_z, r144_value)
    r145_z = tl.arange($(4))
    r145_value = (tl.cast(r145_z, tl.int64) + tl.full([], $(12), dtype=tl.int64))
    r145_value = tl.broadcast_to(r145_value, [$(4)])
    tl.store($(candidate_vadd_36_ptr) + r145_z, r145_value)
    r146_z = tl.arange($(4))
    r146_value = (tl.load($(candidate_vadd_36_ptr) + r146_z, dtype=tl.int64) - tl.full([], $(4), dtype=tl.int64))
    r146_value = tl.broadcast_to(r146_value, [$(4)])
    tl.store($(candidate_vsub_27_ptr) + r146_z, r146_value)
    r147_z = tl.arange($(48))
    r147_value = (tl.cast((r147_z % $(12)), tl.int64) > tl.load($(candidate_vsub_27_ptr) + ((r147_z / $(12)) % $(4)), dtype=tl.int64))
    r147_value = tl.broadcast_to(r147_value, [$(48)])
    tl.store($(candidate_vgt_16_ptr) + r147_z, r147_value)
    r148_z = tl.arange($(384))
    r148_value = tl.where(tl.load($(candidate_vgt_16_ptr) + ((((r148_z / $(12)) % $(4)) * $(12)) + (r148_z % $(12))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_43_ptr) + ((((r148_z / $(48)) % $(8)) * $(48)) + (((r148_z / $(12)) % $(4)) * $(12)) + (r148_z % $(12))), dtype=tl.float32))
    r148_value = r148_value.to(tl.float32)
    r148_value = tl.broadcast_to(r148_value, [$(384)])
    tl.store($(candidate_vmasked_fill_16_ptr) + r148_z, r148_value)
    r149_z = tl.arange($(384))
    r149_value = (tl.load($(candidate_vmasked_fill_16_ptr) + ((((r149_z / $(48)) % $(8)) * $(48)) + (((r149_z / $(12)) % $(4)) * $(12)) + (r149_z % $(12))), dtype=tl.float32) - tl.load($(candidate_vcat_14_ptr) + ((((r149_z / $(12)) % $(4)) * $(8)) + ((r149_z / $(48)) % $(8)) + $(96)), dtype=tl.float32))
    r149_value = r149_value.to(tl.float32)
    r149_value = tl.broadcast_to(r149_value, [$(384)])
    tl.store($(candidate_vsub_28_ptr) + r149_z, r149_value)
    r150_z = tl.arange($(384))
    r150_value = tl.exp(tl.load($(candidate_vsub_28_ptr) + ((((r150_z / $(48)) % $(8)) * $(48)) + (((r150_z / $(12)) % $(4)) * $(12)) + (r150_z % $(12))), dtype=tl.float32))
    r150_value = r150_value.to(tl.float32)
    r150_value = tl.broadcast_to(r150_value, [$(384)])
    tl.store($(candidate_vexp_1_ptr) + r150_z, r150_value)
    r151_z = tl.arange($(4096))
    r151_value = (tl.load($(candidate_vto_copy_110_ptr) + ((((r151_z / $(512)) % $(8)) * $(512)) + (((r151_z / $(128)) % $(4)) * $(128)) + (r151_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_111_ptr) + ((((r151_z / $(512)) % $(8)) * $(512)) + (((r151_z / $(128)) % $(4)) * $(128)) + (r151_z % $(128))), dtype=tl.float32))
    r151_value = r151_value.to(tl.float32)
    r151_value = tl.broadcast_to(r151_value, [$(4096)])
    tl.store($(candidate_vmul_44_ptr) + r151_z, r151_value)
    r152_z = tl.arange($(32))
    r152_lane = tl.arange($(128))
    r152_value = tl.sum(tl.load($(candidate_vmul_44_ptr) + ((((r152_z[:, None] / $(4)) % $(8)) * $(512)) + ((r152_z[:, None] % $(4)) * $(128)) + r152_lane[None, :]), dtype=tl.float32), axis=1)
    r152_value = r152_value.to(tl.float32)
    r152_value = tl.broadcast_to(r152_value, [$(32)])
    tl.store($(candidate_vsum_2_ptr) + r152_z, r152_value)
    r153_row = tl.arange($(4))
    r153_col = tl.arange($(12))
    r153_lane = tl.arange($(128))
    for r153_group in range($(8)) {
      r153_left = tl.load($(candidate_vto_copy_110_ptr) + (((((((r153_group % $(8)) * $(512)) + (r153_row[:, None] * $(128)) + r153_lane[None, :]) / $(512)) % $(8)) * $(512)) + ((((((r153_group % $(8)) * $(512)) + (r153_row[:, None] * $(128)) + r153_lane[None, :]) / $(128)) % $(4)) * $(128)) + ((((r153_group % $(8)) * $(512)) + (r153_row[:, None] * $(128)) + r153_lane[None, :]) % $(128))), dtype=tl.float32)
      r153_right = tl.load($(candidate_vto_copy_109_ptr) + (((((((r153_group % $(8)) * $(1536)) + (r153_lane[:, None] * $(12)) + r153_col[None, :]) / $(1536)) % $(8)) * $(1536)) + (((((r153_group % $(8)) * $(1536)) + (r153_lane[:, None] * $(12)) + r153_col[None, :]) % $(12)) * $(128)) + (((((r153_group % $(8)) * $(1536)) + (r153_lane[:, None] * $(12)) + r153_col[None, :]) / $(12)) % $(128))), dtype=tl.float32)
      r153_value = tl.dot(r153_left, r153_right)
      r153_value = r153_value.to(tl.float32)
      tl.store($(candidate_vbmm_36_ptr) + ((r153_group * $(48)) + (r153_row[:, None] * $(12)) + r153_col[None, :]), r153_value)
    }
    r154_z = tl.arange($(384))
    r154_value = (tl.load($(candidate_vbmm_36_ptr) + ((((r154_z / $(48)) % $(8)) * $(48)) + (((r154_z / $(12)) % $(4)) * $(12)) + (r154_z % $(12))), dtype=tl.float32) - tl.load($(candidate_vsum_2_ptr) + ((((r154_z / $(48)) % $(8)) * $(4)) + ((r154_z / $(12)) % $(4))), dtype=tl.float32))
    r154_value = r154_value.to(tl.float32)
    r154_value = tl.broadcast_to(r154_value, [$(384)])
    tl.store($(candidate_vsub_29_ptr) + r154_z, r154_value)
    r155_z = tl.arange($(384))
    r155_value = (tl.load($(candidate_vexp_1_ptr) + ((((r155_z / $(48)) % $(8)) * $(48)) + (((r155_z / $(12)) % $(4)) * $(12)) + (r155_z % $(12))), dtype=tl.float32) * tl.load($(candidate_vsub_29_ptr) + ((((r155_z / $(48)) % $(8)) * $(48)) + (((r155_z / $(12)) % $(4)) * $(12)) + (r155_z % $(12))), dtype=tl.float32))
    r155_value = r155_value.to(tl.float32)
    r155_value = tl.broadcast_to(r155_value, [$(384)])
    tl.store($(candidate_vmul_45_ptr) + r155_z, r155_value)
    r156_row = tl.arange($(12))
    r156_col = tl.arange($(192))
    r156_lane = tl.arange($(4))
    for r156_group in range($(8)) {
      r156_left = tl.load($(candidate_vmul_45_ptr) + (((((((r156_group % $(8)) * $(48)) + (r156_row[:, None] * $(4)) + r156_lane[None, :]) / $(48)) % $(8)) * $(48)) + (((((r156_group % $(8)) * $(48)) + (r156_row[:, None] * $(4)) + r156_lane[None, :]) % $(4)) * $(12)) + (((((r156_group % $(8)) * $(48)) + (r156_row[:, None] * $(4)) + r156_lane[None, :]) / $(4)) % $(12))), dtype=tl.float32)
      r156_right = tl.load($(candidate_vto_copy_107_ptr) + (((((((r156_group % $(8)) * $(768)) + (r156_lane[:, None] * $(192)) + r156_col[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r156_group % $(8)) * $(768)) + (r156_lane[:, None] * $(192)) + r156_col[None, :]) / $(192)) % $(4)) * $(192)) + ((((r156_group % $(8)) * $(768)) + (r156_lane[:, None] * $(192)) + r156_col[None, :]) % $(192))), dtype=tl.float32)
      r156_value = tl.dot(r156_left, r156_right)
      r156_value = r156_value.to(tl.float32)
      tl.store($(candidate_vbmm_38_ptr) + ((r156_group * $(2304)) + (r156_row[:, None] * $(192)) + r156_col[None, :]), r156_value)
    }
    r157_z = tl.arange($(18432))
    r157_value = (tl.load($(candidate_vbmm_38_ptr) + ((((r157_z / $(2304)) % $(8)) * $(2304)) + (((r157_z / $(192)) % $(12)) * $(192)) + (r157_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r157_value = r157_value.to(tl.float32)
    r157_value = tl.broadcast_to(r157_value, [$(18432)])
    tl.store($(candidate_vmul_47_ptr) + r157_z, r157_value)
    r158_z = tl.arange($(18432))
    r158_value = tl.load($(candidate_vmul_47_ptr) + ((((r158_z / $(192)) % $(8)) * $(2304)) + (((r158_z / $(1536)) % $(12)) * $(192)) + (r158_z % $(192))), dtype=tl.float32)
    r158_value = r158_value.to(tl.bfloat16)
    r158_value = tl.broadcast_to(r158_value, [$(18432)])
    tl.store($(candidate_vto_copy_113_ptr) + r158_z, r158_value)
    r159_row = tl.arange($(12))
    r159_col = tl.arange($(128))
    r159_lane = tl.arange($(4))
    for r159_group in range($(8)) {
      r159_left = tl.load($(candidate_vexp_1_ptr) + (((((((r159_group % $(8)) * $(48)) + (r159_row[:, None] * $(4)) + r159_lane[None, :]) / $(48)) % $(8)) * $(48)) + (((((r159_group % $(8)) * $(48)) + (r159_row[:, None] * $(4)) + r159_lane[None, :]) % $(4)) * $(12)) + (((((r159_group % $(8)) * $(48)) + (r159_row[:, None] * $(4)) + r159_lane[None, :]) / $(4)) % $(12))), dtype=tl.float32)
      r159_right = tl.load($(candidate_vto_copy_110_ptr) + (((((((r159_group % $(8)) * $(512)) + (r159_lane[:, None] * $(128)) + r159_col[None, :]) / $(512)) % $(8)) * $(512)) + ((((((r159_group % $(8)) * $(512)) + (r159_lane[:, None] * $(128)) + r159_col[None, :]) / $(128)) % $(4)) * $(128)) + ((((r159_group % $(8)) * $(512)) + (r159_lane[:, None] * $(128)) + r159_col[None, :]) % $(128))), dtype=tl.float32)
      r159_value = tl.dot(r159_left, r159_right)
      r159_value = r159_value.to(tl.float32)
      tl.store($(candidate_vbmm_39_ptr) + ((r159_group * $(1536)) + (r159_row[:, None] * $(128)) + r159_col[None, :]), r159_value)
    }
    r160_z = tl.arange($(12288))
    r160_value = tl.load($(candidate_vbmm_39_ptr) + ((((r160_z / $(128)) % $(8)) * $(1536)) + (((r160_z / $(1024)) % $(12)) * $(128)) + (r160_z % $(128))), dtype=tl.float32)
    r160_value = r160_value.to(tl.bfloat16)
    r160_value = tl.broadcast_to(r160_value, [$(12288)])
    tl.store($(candidate_vto_copy_114_ptr) + r160_z, r160_value)
    r161_z = tl.arange($(24576))
    r161_value = tl.where(((r161_z / $(1536)) % $(16)) < $(12), tl.load($(candidate_vto_copy_113_ptr) + ((((((r161_z / $(1536)) % $(16)) - $(0)) % $(12)) * $(1536)) + (((r161_z / $(192)) % $(8)) * $(192)) + (r161_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r161_value = r161_value.to(tl.bfloat16)
    r161_value = tl.broadcast_to(r161_value, [$(24576)])
    tl.store($(candidate_vcat_17_ptr) + r161_z, r161_value)
    r162_z = tl.arange($(16384))
    r162_value = tl.where(((r162_z / $(1024)) % $(16)) < $(12), tl.load($(candidate_vto_copy_114_ptr) + ((((((r162_z / $(1024)) % $(16)) - $(0)) % $(12)) * $(1024)) + (((r162_z / $(128)) % $(8)) * $(128)) + (r162_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r162_value = r162_value.to(tl.bfloat16)
    r162_value = tl.broadcast_to(r162_value, [$(16384)])
    tl.store($(candidate_vcat_18_ptr) + r162_z, r162_value)
    r163_z = tl.arange($(24576))
    r163_value = tl.load($(candidate_vcat_11_ptr) + ((((r163_z / $(192)) % $(16)) * $(1536)) + (((r163_z / $(3072)) % $(8)) * $(192)) + (r163_z % $(192)) + $(24576)), dtype=tl.bfloat16)
    r163_value = r163_value.to(tl.float32)
    r163_value = tl.broadcast_to(r163_value, [$(24576)])
    tl.store($(candidate_vto_copy_115_ptr) + r163_z, r163_value)
    r164_z = tl.arange($(73728))
    r164_value = tl.load($(candidate_vcat_12_ptr) + ((((r164_z / $(192)) % $(48)) * $(1536)) + (((r164_z / $(9216)) % $(8)) * $(192)) + (r164_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r164_value = r164_value.to(tl.float32)
    r164_value = tl.broadcast_to(r164_value, [$(73728)])
    tl.store($(candidate_vto_copy_116_ptr) + r164_z, r164_value)
    r165_z = tl.arange($(49152))
    r165_value = tl.load($(candidate_vbmm_23_ptr) + ((((r165_z / $(128)) % $(48)) * $(1024)) + (((r165_z / $(6144)) % $(8)) * $(128)) + (r165_z % $(128)) + $(65536)), dtype=tl.bfloat16)
    r165_value = r165_value.to(tl.float32)
    r165_value = tl.broadcast_to(r165_value, [$(49152)])
    tl.store($(candidate_vto_copy_117_ptr) + r165_z, r165_value)
    r166_z = tl.arange($(16384))
    r166_value = tl.load($(candidate_vmm_28_ptr) + ((((r166_z / $(128)) % $(16)) * $(1024)) + (((r166_z / $(2048)) % $(8)) * $(128)) + (r166_z % $(128)) + $(114688)), dtype=tl.bfloat16)
    r166_value = r166_value.to(tl.float32)
    r166_value = tl.broadcast_to(r166_value, [$(16384)])
    tl.store($(candidate_vto_copy_118_ptr) + r166_z, r166_value)
    r167_z = tl.arange($(16384))
    r167_value = tl.load($(candidate_vcat_13_ptr) + ((((r167_z / $(128)) % $(16)) * $(1024)) + (((r167_z / $(2048)) % $(8)) * $(128)) + (r167_z % $(128)) + $(16384)), dtype=tl.bfloat16)
    r167_value = r167_value.to(tl.float32)
    r167_value = tl.broadcast_to(r167_value, [$(16384)])
    tl.store($(candidate_vto_copy_119_ptr) + r167_z, r167_value)
    r168_row = tl.arange($(16))
    r168_col = tl.arange($(48))
    r168_lane = tl.arange($(192))
    for r168_group in range($(8)) {
      r168_left = tl.load($(candidate_vto_copy_115_ptr) + (((((((r168_group % $(8)) * $(3072)) + (r168_row[:, None] * $(192)) + r168_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r168_group % $(8)) * $(3072)) + (r168_row[:, None] * $(192)) + r168_lane[None, :]) / $(192)) % $(16)) * $(192)) + ((((r168_group % $(8)) * $(3072)) + (r168_row[:, None] * $(192)) + r168_lane[None, :]) % $(192))), dtype=tl.float32)
      r168_right = tl.load($(candidate_vto_copy_116_ptr) + (((((((r168_group % $(8)) * $(9216)) + (r168_lane[:, None] * $(48)) + r168_col[None, :]) / $(9216)) % $(8)) * $(9216)) + (((((r168_group % $(8)) * $(9216)) + (r168_lane[:, None] * $(48)) + r168_col[None, :]) % $(48)) * $(192)) + (((((r168_group % $(8)) * $(9216)) + (r168_lane[:, None] * $(48)) + r168_col[None, :]) / $(48)) % $(192))), dtype=tl.float32)
      r168_value = tl.dot(r168_left, r168_right)
      r168_value = r168_value.to(tl.float32)
      tl.store($(candidate_vbmm_40_ptr) + ((r168_group * $(768)) + (r168_row[:, None] * $(48)) + r168_col[None, :]), r168_value)
    }
    r169_z = tl.arange($(6144))
    r169_value = (tl.load($(candidate_vbmm_40_ptr) + ((((r169_z / $(768)) % $(8)) * $(768)) + (((r169_z / $(48)) % $(16)) * $(48)) + (r169_z % $(48))), dtype=tl.float32) * 0.07216878364870322)
    r169_value = r169_value.to(tl.float32)
    r169_value = tl.broadcast_to(r169_value, [$(6144)])
    tl.store($(candidate_vmul_48_ptr) + r169_z, r169_value)
    r170_z = tl.arange($(16))
    r170_value = (tl.cast(r170_z, tl.int64) + tl.full([], $(48), dtype=tl.int64))
    r170_value = tl.broadcast_to(r170_value, [$(16)])
    tl.store($(candidate_vadd_37_ptr) + r170_z, r170_value)
    r171_z = tl.arange($(16))
    r171_value = (tl.load($(candidate_vadd_37_ptr) + r171_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r171_value = tl.broadcast_to(r171_value, [$(16)])
    tl.store($(candidate_vsub_30_ptr) + r171_z, r171_value)
    r172_z = tl.arange($(768))
    r172_value = (tl.cast((r172_z % $(48)), tl.int64) > tl.load($(candidate_vsub_30_ptr) + ((r172_z / $(48)) % $(16)), dtype=tl.int64))
    r172_value = tl.broadcast_to(r172_value, [$(768)])
    tl.store($(candidate_vgt_17_ptr) + r172_z, r172_value)
    r173_z = tl.arange($(6144))
    r173_value = tl.where(tl.load($(candidate_vgt_17_ptr) + ((((r173_z / $(48)) % $(16)) * $(48)) + (r173_z % $(48))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_48_ptr) + ((((r173_z / $(768)) % $(8)) * $(768)) + (((r173_z / $(48)) % $(16)) * $(48)) + (r173_z % $(48))), dtype=tl.float32))
    r173_value = r173_value.to(tl.float32)
    r173_value = tl.broadcast_to(r173_value, [$(6144)])
    tl.store($(candidate_vmasked_fill_17_ptr) + r173_z, r173_value)
    r174_z = tl.arange($(6144))
    r174_value = (tl.load($(candidate_vmasked_fill_17_ptr) + ((((r174_z / $(768)) % $(8)) * $(768)) + (((r174_z / $(48)) % $(16)) * $(48)) + (r174_z % $(48))), dtype=tl.float32) - tl.load($(candidate_vcat_14_ptr) + ((((r174_z / $(48)) % $(16)) * $(8)) + ((r174_z / $(768)) % $(8)) + $(128)), dtype=tl.float32))
    r174_value = r174_value.to(tl.float32)
    r174_value = tl.broadcast_to(r174_value, [$(6144)])
    tl.store($(candidate_vsub_31_ptr) + r174_z, r174_value)
    r175_z = tl.arange($(6144))
    r175_value = tl.exp(tl.load($(candidate_vsub_31_ptr) + ((((r175_z / $(768)) % $(8)) * $(768)) + (((r175_z / $(48)) % $(16)) * $(48)) + (r175_z % $(48))), dtype=tl.float32))
    r175_value = r175_value.to(tl.float32)
    r175_value = tl.broadcast_to(r175_value, [$(6144)])
    tl.store($(candidate_vexp_2_ptr) + r175_z, r175_value)
    r176_z = tl.arange($(16384))
    r176_value = (tl.load($(candidate_vto_copy_118_ptr) + ((((r176_z / $(2048)) % $(8)) * $(2048)) + (((r176_z / $(128)) % $(16)) * $(128)) + (r176_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_119_ptr) + ((((r176_z / $(2048)) % $(8)) * $(2048)) + (((r176_z / $(128)) % $(16)) * $(128)) + (r176_z % $(128))), dtype=tl.float32))
    r176_value = r176_value.to(tl.float32)
    r176_value = tl.broadcast_to(r176_value, [$(16384)])
    tl.store($(candidate_vmul_49_ptr) + r176_z, r176_value)
    r177_z = tl.arange($(128))
    r177_lane = tl.arange($(128))
    r177_value = tl.sum(tl.load($(candidate_vmul_49_ptr) + ((((r177_z[:, None] / $(16)) % $(8)) * $(2048)) + ((r177_z[:, None] % $(16)) * $(128)) + r177_lane[None, :]), dtype=tl.float32), axis=1)
    r177_value = r177_value.to(tl.float32)
    r177_value = tl.broadcast_to(r177_value, [$(128)])
    tl.store($(candidate_vsum_3_ptr) + r177_z, r177_value)
    r178_row = tl.arange($(16))
    r178_col = tl.arange($(48))
    r178_lane = tl.arange($(128))
    for r178_group in range($(8)) {
      r178_left = tl.load($(candidate_vto_copy_118_ptr) + (((((((r178_group % $(8)) * $(2048)) + (r178_row[:, None] * $(128)) + r178_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r178_group % $(8)) * $(2048)) + (r178_row[:, None] * $(128)) + r178_lane[None, :]) / $(128)) % $(16)) * $(128)) + ((((r178_group % $(8)) * $(2048)) + (r178_row[:, None] * $(128)) + r178_lane[None, :]) % $(128))), dtype=tl.float32)
      r178_right = tl.load($(candidate_vto_copy_117_ptr) + (((((((r178_group % $(8)) * $(6144)) + (r178_lane[:, None] * $(48)) + r178_col[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r178_group % $(8)) * $(6144)) + (r178_lane[:, None] * $(48)) + r178_col[None, :]) % $(48)) * $(128)) + (((((r178_group % $(8)) * $(6144)) + (r178_lane[:, None] * $(48)) + r178_col[None, :]) / $(48)) % $(128))), dtype=tl.float32)
      r178_value = tl.dot(r178_left, r178_right)
      r178_value = r178_value.to(tl.float32)
      tl.store($(candidate_vbmm_41_ptr) + ((r178_group * $(768)) + (r178_row[:, None] * $(48)) + r178_col[None, :]), r178_value)
    }
    r179_z = tl.arange($(6144))
    r179_value = (tl.load($(candidate_vbmm_41_ptr) + ((((r179_z / $(768)) % $(8)) * $(768)) + (((r179_z / $(48)) % $(16)) * $(48)) + (r179_z % $(48))), dtype=tl.float32) - tl.load($(candidate_vsum_3_ptr) + ((((r179_z / $(768)) % $(8)) * $(16)) + ((r179_z / $(48)) % $(16))), dtype=tl.float32))
    r179_value = r179_value.to(tl.float32)
    r179_value = tl.broadcast_to(r179_value, [$(6144)])
    tl.store($(candidate_vsub_32_ptr) + r179_z, r179_value)
    r180_z = tl.arange($(6144))
    r180_value = (tl.load($(candidate_vexp_2_ptr) + ((((r180_z / $(768)) % $(8)) * $(768)) + (((r180_z / $(48)) % $(16)) * $(48)) + (r180_z % $(48))), dtype=tl.float32) * tl.load($(candidate_vsub_32_ptr) + ((((r180_z / $(768)) % $(8)) * $(768)) + (((r180_z / $(48)) % $(16)) * $(48)) + (r180_z % $(48))), dtype=tl.float32))
    r180_value = r180_value.to(tl.float32)
    r180_value = tl.broadcast_to(r180_value, [$(6144)])
    tl.store($(candidate_vmul_50_ptr) + r180_z, r180_value)
    r181_row = tl.arange($(48))
    r181_col = tl.arange($(192))
    r181_lane = tl.arange($(16))
    for r181_group in range($(8)) {
      r181_left = tl.load($(candidate_vmul_50_ptr) + (((((((r181_group % $(8)) * $(768)) + (r181_row[:, None] * $(16)) + r181_lane[None, :]) / $(768)) % $(8)) * $(768)) + (((((r181_group % $(8)) * $(768)) + (r181_row[:, None] * $(16)) + r181_lane[None, :]) % $(16)) * $(48)) + (((((r181_group % $(8)) * $(768)) + (r181_row[:, None] * $(16)) + r181_lane[None, :]) / $(16)) % $(48))), dtype=tl.float32)
      r181_right = tl.load($(candidate_vto_copy_115_ptr) + (((((((r181_group % $(8)) * $(3072)) + (r181_lane[:, None] * $(192)) + r181_col[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r181_group % $(8)) * $(3072)) + (r181_lane[:, None] * $(192)) + r181_col[None, :]) / $(192)) % $(16)) * $(192)) + ((((r181_group % $(8)) * $(3072)) + (r181_lane[:, None] * $(192)) + r181_col[None, :]) % $(192))), dtype=tl.float32)
      r181_value = tl.dot(r181_left, r181_right)
      r181_value = r181_value.to(tl.float32)
      tl.store($(candidate_vbmm_43_ptr) + ((r181_group * $(9216)) + (r181_row[:, None] * $(192)) + r181_col[None, :]), r181_value)
    }
    r182_z = tl.arange($(73728))
    r182_value = (tl.load($(candidate_vbmm_43_ptr) + ((((r182_z / $(9216)) % $(8)) * $(9216)) + (((r182_z / $(192)) % $(48)) * $(192)) + (r182_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r182_value = r182_value.to(tl.float32)
    r182_value = tl.broadcast_to(r182_value, [$(73728)])
    tl.store($(candidate_vmul_52_ptr) + r182_z, r182_value)
    r183_z = tl.arange($(73728))
    r183_value = tl.load($(candidate_vmul_52_ptr) + ((((r183_z / $(192)) % $(8)) * $(9216)) + (((r183_z / $(1536)) % $(48)) * $(192)) + (r183_z % $(192))), dtype=tl.float32)
    r183_value = r183_value.to(tl.bfloat16)
    r183_value = tl.broadcast_to(r183_value, [$(73728)])
    tl.store($(candidate_vto_copy_121_ptr) + r183_z, r183_value)
    r184_row = tl.arange($(48))
    r184_col = tl.arange($(128))
    r184_lane = tl.arange($(16))
    for r184_group in range($(8)) {
      r184_left = tl.load($(candidate_vexp_2_ptr) + (((((((r184_group % $(8)) * $(768)) + (r184_row[:, None] * $(16)) + r184_lane[None, :]) / $(768)) % $(8)) * $(768)) + (((((r184_group % $(8)) * $(768)) + (r184_row[:, None] * $(16)) + r184_lane[None, :]) % $(16)) * $(48)) + (((((r184_group % $(8)) * $(768)) + (r184_row[:, None] * $(16)) + r184_lane[None, :]) / $(16)) % $(48))), dtype=tl.float32)
      r184_right = tl.load($(candidate_vto_copy_118_ptr) + (((((((r184_group % $(8)) * $(2048)) + (r184_lane[:, None] * $(128)) + r184_col[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r184_group % $(8)) * $(2048)) + (r184_lane[:, None] * $(128)) + r184_col[None, :]) / $(128)) % $(16)) * $(128)) + ((((r184_group % $(8)) * $(2048)) + (r184_lane[:, None] * $(128)) + r184_col[None, :]) % $(128))), dtype=tl.float32)
      r184_value = tl.dot(r184_left, r184_right)
      r184_value = r184_value.to(tl.float32)
      tl.store($(candidate_vbmm_44_ptr) + ((r184_group * $(6144)) + (r184_row[:, None] * $(128)) + r184_col[None, :]), r184_value)
    }
    r185_z = tl.arange($(49152))
    r185_value = tl.load($(candidate_vbmm_44_ptr) + ((((r185_z / $(128)) % $(8)) * $(6144)) + (((r185_z / $(1024)) % $(48)) * $(128)) + (r185_z % $(128))), dtype=tl.float32)
    r185_value = r185_value.to(tl.bfloat16)
    r185_value = tl.broadcast_to(r185_value, [$(49152)])
    tl.store($(candidate_vto_copy_122_ptr) + r185_z, r185_value)
    r186_z = tl.arange($(98304))
    r186_value = tl.where(((r186_z / $(1536)) % $(64)) < $(48), tl.load($(candidate_vto_copy_121_ptr) + ((((((r186_z / $(1536)) % $(64)) - $(0)) % $(48)) * $(1536)) + (((r186_z / $(192)) % $(8)) * $(192)) + (r186_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r186_value = r186_value.to(tl.bfloat16)
    r186_value = tl.broadcast_to(r186_value, [$(98304)])
    tl.store($(candidate_vcat_19_ptr) + r186_z, r186_value)
    r187_z = tl.arange($(65536))
    r187_value = tl.where(((r187_z / $(1024)) % $(64)) < $(48), tl.load($(candidate_vto_copy_122_ptr) + ((((((r187_z / $(1024)) % $(64)) - $(0)) % $(48)) * $(1024)) + (((r187_z / $(128)) % $(8)) * $(128)) + (r187_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r187_value = r187_value.to(tl.bfloat16)
    r187_value = tl.broadcast_to(r187_value, [$(65536)])
    tl.store($(candidate_vcat_20_ptr) + r187_z, r187_value)
    r188_z = tl.arange($(196608))
    r188_value = tl.where(((r188_z / $(1536)) % $(128)) < $(48), tl.load($(candidate_vcat_15_ptr) + ((((((r188_z / $(1536)) % $(128)) - $(0)) % $(48)) * $(1536)) + (((r188_z / $(192)) % $(8)) * $(192)) + (r188_z % $(192))), dtype=tl.bfloat16), tl.where(((r188_z / $(1536)) % $(128)) < $(64), tl.load($(candidate_vcat_17_ptr) + ((((((r188_z / $(1536)) % $(128)) - $(48)) % $(16)) * $(1536)) + (((r188_z / $(192)) % $(8)) * $(192)) + (r188_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vcat_19_ptr) + ((((((r188_z / $(1536)) % $(128)) - $(64)) % $(64)) * $(1536)) + (((r188_z / $(192)) % $(8)) * $(192)) + (r188_z % $(192))), dtype=tl.bfloat16)))
    r188_value = r188_value.to(tl.bfloat16)
    r188_value = tl.broadcast_to(r188_value, [$(196608)])
    tl.store($(candidate_vcat_22_ptr) + r188_z, r188_value)
    r189_z = tl.arange($(131072))
    r189_value = tl.where(((r189_z / $(1024)) % $(128)) < $(48), tl.load($(candidate_vcat_16_ptr) + ((((((r189_z / $(1024)) % $(128)) - $(0)) % $(48)) * $(1024)) + (((r189_z / $(128)) % $(8)) * $(128)) + (r189_z % $(128))), dtype=tl.bfloat16), tl.where(((r189_z / $(1024)) % $(128)) < $(64), tl.load($(candidate_vcat_18_ptr) + ((((((r189_z / $(1024)) % $(128)) - $(48)) % $(16)) * $(1024)) + (((r189_z / $(128)) % $(8)) * $(128)) + (r189_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_20_ptr) + ((((((r189_z / $(1024)) % $(128)) - $(64)) % $(64)) * $(1024)) + (((r189_z / $(128)) % $(8)) * $(128)) + (r189_z % $(128))), dtype=tl.bfloat16)))
    r189_value = r189_value.to(tl.bfloat16)
    r189_value = tl.broadcast_to(r189_value, [$(131072)])
    tl.store($(candidate_vcat_23_ptr) + r189_z, r189_value)
    r190_row = tl.arange($(1024))
    r190_col = tl.arange($(512))
    r190_lane = tl.arange($(128))
    for r190_group in range($(1)) {
      r190_left = tl.load($(candidate_vcat_22_ptr) + (((((r190_row[:, None] * $(128)) + r190_lane[None, :]) % $(128)) * $(1536)) + (((((r190_row[:, None] * $(128)) + r190_lane[None, :]) / $(16384)) % $(8)) * $(192)) + ((((r190_row[:, None] * $(128)) + r190_lane[None, :]) / $(128)) % $(128))), dtype=tl.bfloat16)
      r190_right = tl.load($(candidate_vindex_2_ptr) + ((((((r190_lane[:, None] * $(512)) + r190_col[None, :]) / $(512)) % $(128)) * $(576)) + (((r190_lane[:, None] * $(512)) + r190_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r190_value = tl.dot(r190_left, r190_right)
      r190_value = r190_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_47_ptr) + ((r190_group * $(524288)) + (r190_row[:, None] * $(512)) + r190_col[None, :]), r190_value)
    }
    r191_row = tl.arange($(1024))
    r191_col = tl.arange($(512))
    r191_lane = tl.arange($(128))
    for r191_group in range($(1)) {
      r191_left = tl.load($(candidate_vcat_23_ptr) + (((((r191_row[:, None] * $(128)) + r191_lane[None, :]) % $(128)) * $(1024)) + (((((r191_row[:, None] * $(128)) + r191_lane[None, :]) / $(16384)) % $(8)) * $(128)) + ((((r191_row[:, None] * $(128)) + r191_lane[None, :]) / $(128)) % $(128))), dtype=tl.bfloat16)
      r191_right = tl.load($(candidate_vindex_2_ptr) + ((((((r191_lane[:, None] * $(512)) + r191_col[None, :]) / $(512)) % $(128)) * $(576)) + (((r191_lane[:, None] * $(512)) + r191_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r191_value = tl.dot(r191_left, r191_right)
      r191_value = r191_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_48_ptr) + ((r191_group * $(524288)) + (r191_row[:, None] * $(512)) + r191_col[None, :]), r191_value)
    }
    r192_z = tl.arange($(1048576))
    r192_value = tl.where(((r192_z / $(512)) % $(256)) < $(128), tl.load($(candidate_vbmm_47_ptr) + ((((((((r192_z / $(131072)) % $(8)) * $(65536)) + ((((r192_z / $(512)) % $(256)) - $(0)) * $(512)) + (r192_z % $(512))) / $(65536)) % $(8)) * $(65536)) + ((((((r192_z / $(131072)) % $(8)) * $(128)) + (((r192_z / $(512)) % $(256)) - $(0))) % $(128)) * $(512)) + (r192_z % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vbmm_48_ptr) + ((((((((r192_z / $(131072)) % $(8)) * $(65536)) + ((((r192_z / $(512)) % $(256)) - $(128)) * $(512)) + (r192_z % $(512))) / $(65536)) % $(8)) * $(65536)) + ((((((r192_z / $(131072)) % $(8)) * $(128)) + (((r192_z / $(512)) % $(256)) - $(128))) % $(128)) * $(512)) + (r192_z % $(512))), dtype=tl.bfloat16))
    r192_value = r192_value.to(tl.bfloat16)
    r192_value = tl.broadcast_to(r192_value, [$(1048576)])
    tl.store($(candidate_vcat_24_ptr) + r192_z, r192_value)
    r193_row = tl.arange($(128))
    r193_col = tl.arange($(1024))
    r193_lane = tl.arange($(512))
    for r193_group in range($(1)) {
      r193_left = tl.load($(candidate_vindex_2_ptr) + ((((((r193_row[:, None] * $(512)) + r193_lane[None, :]) / $(512)) % $(128)) * $(576)) + (((r193_row[:, None] * $(512)) + r193_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r193_right = tl.load($(candidate_vto_copy_20_ptr) + ((((((r193_lane[:, None] * $(1024)) + r193_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r193_lane[:, None] * $(1024)) + r193_col[None, :]) % $(128)) * $(512)) + ((((r193_lane[:, None] * $(1024)) + r193_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r193_value = tl.dot(r193_left, r193_right)
      r193_value = r193_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_49_ptr) + ((r193_group * $(131072)) + (r193_row[:, None] * $(1024)) + r193_col[None, :]), r193_value)
    }
    r194_row = tl.arange($(128))
    r194_col = tl.arange($(1024))
    r194_lane = tl.arange($(512))
    for r194_group in range($(1)) {
      r194_left = tl.load($(candidate_vindex_2_ptr) + ((((((r194_row[:, None] * $(512)) + r194_lane[None, :]) / $(512)) % $(128)) * $(576)) + (((r194_row[:, None] * $(512)) + r194_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r194_right = tl.load($(candidate_vto_copy_20_ptr) + ((((((r194_lane[:, None] * $(1024)) + r194_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r194_lane[:, None] * $(1024)) + r194_col[None, :]) % $(128)) * $(512)) + ((((r194_lane[:, None] * $(1024)) + r194_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r194_value = tl.dot(r194_left, r194_right)
      r194_value = r194_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_50_ptr) + ((r194_group * $(131072)) + (r194_row[:, None] * $(1024)) + r194_col[None, :]), r194_value)
    }
    r195_z = tl.arange($(49152))
    r195_value = tl.where((r195_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r195_z / $(1536)) % $(32)) * $(1024)) + (((r195_z / $(192)) % $(8)) * $(128)) + ((r195_z % $(192)) - $(0)) + $(65536)) / $(1024)) % $(128)) * $(1536)) + (((((((r195_z / $(1536)) % $(32)) * $(1024)) + (((r195_z / $(192)) % $(8)) * $(128)) + ((r195_z % $(192)) - $(0)) + $(65536)) / $(128)) % $(8)) * $(192)) + (((((r195_z / $(1536)) % $(32)) * $(1024)) + (((r195_z / $(192)) % $(8)) * $(128)) + ((r195_z % $(192)) - $(0)) + $(65536)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vadd_6_ptr) + ((((((((r195_z / $(1536)) % $(32)) * $(512)) + (((r195_z / $(192)) % $(8)) * $(64)) + ((r195_z % $(192)) - $(128)) + $(32768)) / $(512)) % $(128)) * $(512)) + (((((((r195_z / $(1536)) % $(32)) * $(512)) + (((r195_z / $(192)) % $(8)) * $(64)) + ((r195_z % $(192)) - $(128)) + $(32768)) / $(64)) % $(8)) * $(64)) + (((((r195_z / $(1536)) % $(32)) * $(512)) + (((r195_z / $(192)) % $(8)) * $(64)) + ((r195_z % $(192)) - $(128)) + $(32768)) % $(64))), dtype=tl.bfloat16))
    r195_value = r195_value.to(tl.bfloat16)
    r195_value = tl.broadcast_to(r195_value, [$(49152)])
    tl.store($(candidate_vcat_25_ptr) + r195_z, r195_value)
    r196_z = tl.arange($(196608))
    r196_value = tl.where((r196_z % $(192)) < $(128), tl.load($(candidate_vbmm_49_ptr) + ((((((((r196_z / $(1536)) % $(128)) * $(1024)) + (((r196_z / $(192)) % $(8)) * $(128)) + ((r196_z % $(192)) - $(0))) / $(1024)) % $(128)) * $(1024)) + (((((((r196_z / $(1536)) % $(128)) * $(1024)) + (((r196_z / $(192)) % $(8)) * $(128)) + ((r196_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r196_z / $(1536)) % $(128)) * $(1024)) + (((r196_z / $(192)) % $(8)) * $(128)) + ((r196_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vindex_2_ptr) + ((((r196_z / $(1536)) % $(128)) * $(576)) + ((((r196_z % $(192)) - $(128)) + $(512)) % $(576))), dtype=tl.bfloat16))
    r196_value = r196_value.to(tl.bfloat16)
    r196_value = tl.broadcast_to(r196_value, [$(196608)])
    tl.store($(candidate_vcat_26_ptr) + r196_z, r196_value)
    r197_z = tl.arange($(18432))
    r197_value = tl.load($(candidate_vcat_25_ptr) + ((((r197_z / $(192)) % $(12)) * $(1536)) + (((r197_z / $(2304)) % $(8)) * $(192)) + (r197_z % $(192))), dtype=tl.bfloat16)
    r197_value = r197_value.to(tl.float32)
    r197_value = tl.broadcast_to(r197_value, [$(18432)])
    tl.store($(candidate_vto_copy_123_ptr) + r197_z, r197_value)
    r198_z = tl.arange($(36864))
    r198_value = tl.load($(candidate_vcat_26_ptr) + ((((r198_z / $(192)) % $(24)) * $(1536)) + (((r198_z / $(4608)) % $(8)) * $(192)) + (r198_z % $(192))), dtype=tl.bfloat16)
    r198_value = r198_value.to(tl.float32)
    r198_value = tl.broadcast_to(r198_value, [$(36864)])
    tl.store($(candidate_vto_copy_124_ptr) + r198_z, r198_value)
    r199_row = tl.arange($(12))
    r199_col = tl.arange($(24))
    r199_lane = tl.arange($(192))
    for r199_group in range($(8)) {
      r199_left = tl.load($(candidate_vto_copy_123_ptr) + (((((((r199_group % $(8)) * $(2304)) + (r199_row[:, None] * $(192)) + r199_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r199_group % $(8)) * $(2304)) + (r199_row[:, None] * $(192)) + r199_lane[None, :]) / $(192)) % $(12)) * $(192)) + ((((r199_group % $(8)) * $(2304)) + (r199_row[:, None] * $(192)) + r199_lane[None, :]) % $(192))), dtype=tl.float32)
      r199_right = tl.load($(candidate_vto_copy_124_ptr) + (((((((r199_group % $(8)) * $(4608)) + (r199_lane[:, None] * $(24)) + r199_col[None, :]) / $(4608)) % $(8)) * $(4608)) + (((((r199_group % $(8)) * $(4608)) + (r199_lane[:, None] * $(24)) + r199_col[None, :]) % $(24)) * $(192)) + (((((r199_group % $(8)) * $(4608)) + (r199_lane[:, None] * $(24)) + r199_col[None, :]) / $(24)) % $(192))), dtype=tl.float32)
      r199_value = tl.dot(r199_left, r199_right)
      r199_value = r199_value.to(tl.float32)
      tl.store($(candidate_vbmm_51_ptr) + ((r199_group * $(288)) + (r199_row[:, None] * $(24)) + r199_col[None, :]), r199_value)
    }
    r200_z = tl.arange($(2304))
    r200_value = (tl.load($(candidate_vbmm_51_ptr) + ((((r200_z / $(288)) % $(8)) * $(288)) + (((r200_z / $(24)) % $(12)) * $(24)) + (r200_z % $(24))), dtype=tl.float32) * 0.07216878364870322)
    r200_value = r200_value.to(tl.float32)
    r200_value = tl.broadcast_to(r200_value, [$(2304)])
    tl.store($(candidate_vmul_53_ptr) + r200_z, r200_value)
    r201_z = tl.arange($(12))
    r201_value = (tl.cast(r201_z, tl.int64) + tl.full([], $(24), dtype=tl.int64))
    r201_value = tl.broadcast_to(r201_value, [$(12)])
    tl.store($(candidate_vadd_39_ptr) + r201_z, r201_value)
    r202_z = tl.arange($(12))
    r202_value = (tl.load($(candidate_vadd_39_ptr) + r202_z, dtype=tl.int64) - tl.full([], $(12), dtype=tl.int64))
    r202_value = tl.broadcast_to(r202_value, [$(12)])
    tl.store($(candidate_vsub_33_ptr) + r202_z, r202_value)
    r203_z = tl.arange($(288))
    r203_value = (tl.cast((r203_z % $(24)), tl.int64) > tl.load($(candidate_vsub_33_ptr) + ((r203_z / $(24)) % $(12)), dtype=tl.int64))
    r203_value = tl.broadcast_to(r203_value, [$(288)])
    tl.store($(candidate_vgt_18_ptr) + r203_z, r203_value)
    r204_z = tl.arange($(2304))
    r204_value = tl.where(tl.load($(candidate_vgt_18_ptr) + ((((r204_z / $(24)) % $(12)) * $(24)) + (r204_z % $(24))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_53_ptr) + ((((r204_z / $(288)) % $(8)) * $(288)) + (((r204_z / $(24)) % $(12)) * $(24)) + (r204_z % $(24))), dtype=tl.float32))
    r204_value = r204_value.to(tl.float32)
    r204_value = tl.broadcast_to(r204_value, [$(2304)])
    tl.store($(candidate_vmasked_fill_18_ptr) + r204_z, r204_value)
    r205_z = tl.arange($(2304))
    r205_lane = tl.arange($(24))
    r205_row_values = tl.load($(candidate_vmasked_fill_18_ptr) + ((((r205_z[:, None] / $(288)) % $(8)) * $(288)) + (((r205_z[:, None] / $(24)) % $(12)) * $(24)) + (r205_lane[None, :] % $(24))), dtype=tl.float32)
    r205_maximum = tl.max(r205_row_values, axis=1, keep_dims=true)
    r205_weights = tl.exp(r205_row_values - r205_maximum)
    r205_total = tl.sum(r205_weights, axis=1)
    r205_maximum = tl.max(r205_row_values, axis=1)
    r205_value = tl.exp(tl.load($(candidate_vmasked_fill_18_ptr) + ((((r205_z / $(288)) % $(8)) * $(288)) + (((r205_z / $(24)) % $(12)) * $(24)) + (r205_z % $(24))), dtype=tl.float32) - r205_maximum) / r205_total
    r205_value = r205_value.to(tl.float32)
    r205_value = tl.broadcast_to(r205_value, [$(2304)])
    tl.store($(candidate_vsoftmax_15_ptr) + r205_z, r205_value)
    r206_z = tl.arange($(2304))
    r206_value = tl.load($(candidate_vsoftmax_15_ptr) + ((((r206_z / $(288)) % $(8)) * $(288)) + (((r206_z / $(24)) % $(12)) * $(24)) + (r206_z % $(24))), dtype=tl.float32)
    r206_value = r206_value.to(tl.bfloat16)
    r206_value = tl.broadcast_to(r206_value, [$(2304)])
    tl.store($(candidate_vto_copy_125_ptr) + r206_z, r206_value)
    r207_row = tl.arange($(12))
    r207_col = tl.arange($(128))
    r207_lane = tl.arange($(24))
    for r207_group in range($(8)) {
      r207_left = tl.load($(candidate_vto_copy_125_ptr) + (((((((r207_group % $(8)) * $(288)) + (r207_row[:, None] * $(24)) + r207_lane[None, :]) / $(288)) % $(8)) * $(288)) + ((((((r207_group % $(8)) * $(288)) + (r207_row[:, None] * $(24)) + r207_lane[None, :]) / $(24)) % $(12)) * $(24)) + ((((r207_group % $(8)) * $(288)) + (r207_row[:, None] * $(24)) + r207_lane[None, :]) % $(24))), dtype=tl.bfloat16)
      r207_right = tl.load($(candidate_vbmm_50_ptr) + (((((((r207_group % $(8)) * $(3072)) + (r207_lane[:, None] * $(128)) + r207_col[None, :]) / $(128)) % $(24)) * $(1024)) + ((((((r207_group % $(8)) * $(3072)) + (r207_lane[:, None] * $(128)) + r207_col[None, :]) / $(3072)) % $(8)) * $(128)) + ((((r207_group % $(8)) * $(3072)) + (r207_lane[:, None] * $(128)) + r207_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r207_value = tl.dot(r207_left, r207_right)
      r207_value = r207_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_52_ptr) + ((r207_group * $(1536)) + (r207_row[:, None] * $(128)) + r207_col[None, :]), r207_value)
    }
    r208_z = tl.arange($(96))
    r208_lane = tl.arange($(24))
    r208_row_values = tl.load($(candidate_vmasked_fill_18_ptr) + ((((r208_z[:, None] / $(12)) % $(8)) * $(288)) + ((r208_z[:, None] % $(12)) * $(24)) + (r208_lane[None, :] % $(24))), dtype=tl.float32)
    r208_maximum = tl.max(r208_row_values, axis=1, keep_dims=true)
    r208_weights = tl.exp(r208_row_values - r208_maximum)
    r208_total = tl.sum(r208_weights, axis=1)
    r208_maximum = tl.max(r208_row_values, axis=1)
    r208_value = r208_maximum + tl.log(r208_total)
    r208_value = r208_value.to(tl.float32)
    r208_value = tl.broadcast_to(r208_value, [$(96)])
    tl.store($(candidate_vlogsumexp_15_ptr) + r208_z, r208_value)
    r209_z = tl.arange($(6144))
    r209_value = tl.load($(candidate_vcat_25_ptr) + ((((r209_z / $(192)) % $(4)) * $(1536)) + (((r209_z / $(768)) % $(8)) * $(192)) + (r209_z % $(192)) + $(18432)), dtype=tl.bfloat16)
    r209_value = r209_value.to(tl.float32)
    r209_value = tl.broadcast_to(r209_value, [$(6144)])
    tl.store($(candidate_vto_copy_126_ptr) + r209_z, r209_value)
    r210_z = tl.arange($(12288))
    r210_value = tl.load($(candidate_vcat_26_ptr) + ((((r210_z / $(192)) % $(8)) * $(1536)) + (((r210_z / $(1536)) % $(8)) * $(192)) + (r210_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r210_value = r210_value.to(tl.float32)
    r210_value = tl.broadcast_to(r210_value, [$(12288)])
    tl.store($(candidate_vto_copy_127_ptr) + r210_z, r210_value)
    r211_row = tl.arange($(4))
    r211_col = tl.arange($(8))
    r211_lane = tl.arange($(192))
    for r211_group in range($(8)) {
      r211_left = tl.load($(candidate_vto_copy_126_ptr) + (((((((r211_group % $(8)) * $(768)) + (r211_row[:, None] * $(192)) + r211_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r211_group % $(8)) * $(768)) + (r211_row[:, None] * $(192)) + r211_lane[None, :]) / $(192)) % $(4)) * $(192)) + ((((r211_group % $(8)) * $(768)) + (r211_row[:, None] * $(192)) + r211_lane[None, :]) % $(192))), dtype=tl.float32)
      r211_right = tl.load($(candidate_vto_copy_127_ptr) + (((((((r211_group % $(8)) * $(1536)) + (r211_lane[:, None] * $(8)) + r211_col[None, :]) / $(1536)) % $(8)) * $(1536)) + (((((r211_group % $(8)) * $(1536)) + (r211_lane[:, None] * $(8)) + r211_col[None, :]) % $(8)) * $(192)) + (((((r211_group % $(8)) * $(1536)) + (r211_lane[:, None] * $(8)) + r211_col[None, :]) / $(8)) % $(192))), dtype=tl.float32)
      r211_value = tl.dot(r211_left, r211_right)
      r211_value = r211_value.to(tl.float32)
      tl.store($(candidate_vbmm_53_ptr) + ((r211_group * $(32)) + (r211_row[:, None] * $(8)) + r211_col[None, :]), r211_value)
    }
    r212_z = tl.arange($(256))
    r212_value = (tl.load($(candidate_vbmm_53_ptr) + ((((r212_z / $(32)) % $(8)) * $(32)) + (((r212_z / $(8)) % $(4)) * $(8)) + (r212_z % $(8))), dtype=tl.float32) * 0.07216878364870322)
    r212_value = r212_value.to(tl.float32)
    r212_value = tl.broadcast_to(r212_value, [$(256)])
    tl.store($(candidate_vmul_54_ptr) + r212_z, r212_value)
    r213_z = tl.arange($(4))
    r213_value = (tl.cast(r213_z, tl.int64) + tl.full([], $(8), dtype=tl.int64))
    r213_value = tl.broadcast_to(r213_value, [$(4)])
    tl.store($(candidate_vadd_40_ptr) + r213_z, r213_value)
    r214_z = tl.arange($(4))
    r214_value = (tl.load($(candidate_vadd_40_ptr) + r214_z, dtype=tl.int64) - tl.full([], $(4), dtype=tl.int64))
    r214_value = tl.broadcast_to(r214_value, [$(4)])
    tl.store($(candidate_vsub_34_ptr) + r214_z, r214_value)
    r215_z = tl.arange($(32))
    r215_value = (tl.cast((r215_z % $(8)), tl.int64) > tl.load($(candidate_vsub_34_ptr) + ((r215_z / $(8)) % $(4)), dtype=tl.int64))
    r215_value = tl.broadcast_to(r215_value, [$(32)])
    tl.store($(candidate_vgt_19_ptr) + r215_z, r215_value)
    r216_z = tl.arange($(256))
    r216_value = tl.where(tl.load($(candidate_vgt_19_ptr) + ((((r216_z / $(8)) % $(4)) * $(8)) + (r216_z % $(8))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_54_ptr) + ((((r216_z / $(32)) % $(8)) * $(32)) + (((r216_z / $(8)) % $(4)) * $(8)) + (r216_z % $(8))), dtype=tl.float32))
    r216_value = r216_value.to(tl.float32)
    r216_value = tl.broadcast_to(r216_value, [$(256)])
    tl.store($(candidate_vmasked_fill_19_ptr) + r216_z, r216_value)
    r217_z = tl.arange($(256))
    r217_lane = tl.arange($(8))
    r217_row_values = tl.load($(candidate_vmasked_fill_19_ptr) + ((((r217_z[:, None] / $(32)) % $(8)) * $(32)) + (((r217_z[:, None] / $(8)) % $(4)) * $(8)) + (r217_lane[None, :] % $(8))), dtype=tl.float32)
    r217_maximum = tl.max(r217_row_values, axis=1, keep_dims=true)
    r217_weights = tl.exp(r217_row_values - r217_maximum)
    r217_total = tl.sum(r217_weights, axis=1)
    r217_maximum = tl.max(r217_row_values, axis=1)
    r217_value = tl.exp(tl.load($(candidate_vmasked_fill_19_ptr) + ((((r217_z / $(32)) % $(8)) * $(32)) + (((r217_z / $(8)) % $(4)) * $(8)) + (r217_z % $(8))), dtype=tl.float32) - r217_maximum) / r217_total
    r217_value = r217_value.to(tl.float32)
    r217_value = tl.broadcast_to(r217_value, [$(256)])
    tl.store($(candidate_vsoftmax_16_ptr) + r217_z, r217_value)
    r218_z = tl.arange($(256))
    r218_value = tl.load($(candidate_vsoftmax_16_ptr) + ((((r218_z / $(32)) % $(8)) * $(32)) + (((r218_z / $(8)) % $(4)) * $(8)) + (r218_z % $(8))), dtype=tl.float32)
    r218_value = r218_value.to(tl.bfloat16)
    r218_value = tl.broadcast_to(r218_value, [$(256)])
    tl.store($(candidate_vto_copy_128_ptr) + r218_z, r218_value)
    r219_row = tl.arange($(4))
    r219_col = tl.arange($(128))
    r219_lane = tl.arange($(8))
    for r219_group in range($(8)) {
      r219_left = tl.load($(candidate_vto_copy_128_ptr) + (((((((r219_group % $(8)) * $(32)) + (r219_row[:, None] * $(8)) + r219_lane[None, :]) / $(32)) % $(8)) * $(32)) + ((((((r219_group % $(8)) * $(32)) + (r219_row[:, None] * $(8)) + r219_lane[None, :]) / $(8)) % $(4)) * $(8)) + ((((r219_group % $(8)) * $(32)) + (r219_row[:, None] * $(8)) + r219_lane[None, :]) % $(8))), dtype=tl.bfloat16)
      r219_right = tl.load($(candidate_vbmm_50_ptr) + (((((((r219_group % $(8)) * $(1024)) + (r219_lane[:, None] * $(128)) + r219_col[None, :]) / $(128)) % $(8)) * $(1024)) + ((((((r219_group % $(8)) * $(1024)) + (r219_lane[:, None] * $(128)) + r219_col[None, :]) / $(1024)) % $(8)) * $(128)) + ((((r219_group % $(8)) * $(1024)) + (r219_lane[:, None] * $(128)) + r219_col[None, :]) % $(128)) + $(49152)), dtype=tl.bfloat16)
      r219_value = tl.dot(r219_left, r219_right)
      r219_value = r219_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_54_ptr) + ((r219_group * $(512)) + (r219_row[:, None] * $(128)) + r219_col[None, :]), r219_value)
    }
    r220_z = tl.arange($(32))
    r220_lane = tl.arange($(8))
    r220_row_values = tl.load($(candidate_vmasked_fill_19_ptr) + ((((r220_z[:, None] / $(4)) % $(8)) * $(32)) + ((r220_z[:, None] % $(4)) * $(8)) + (r220_lane[None, :] % $(8))), dtype=tl.float32)
    r220_maximum = tl.max(r220_row_values, axis=1, keep_dims=true)
    r220_weights = tl.exp(r220_row_values - r220_maximum)
    r220_total = tl.sum(r220_weights, axis=1)
    r220_maximum = tl.max(r220_row_values, axis=1)
    r220_value = r220_maximum + tl.log(r220_total)
    r220_value = r220_value.to(tl.float32)
    r220_value = tl.broadcast_to(r220_value, [$(32)])
    tl.store($(candidate_vlogsumexp_16_ptr) + r220_z, r220_value)
    r221_z = tl.arange($(24576))
    r221_value = tl.load($(candidate_vcat_25_ptr) + ((((r221_z / $(192)) % $(16)) * $(1536)) + (((r221_z / $(3072)) % $(8)) * $(192)) + (r221_z % $(192)) + $(24576)), dtype=tl.bfloat16)
    r221_value = r221_value.to(tl.float32)
    r221_value = tl.broadcast_to(r221_value, [$(24576)])
    tl.store($(candidate_vto_copy_129_ptr) + r221_z, r221_value)
    r222_z = tl.arange($(49152))
    r222_value = tl.load($(candidate_vcat_26_ptr) + ((((r222_z / $(192)) % $(32)) * $(1536)) + (((r222_z / $(6144)) % $(8)) * $(192)) + (r222_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r222_value = r222_value.to(tl.float32)
    r222_value = tl.broadcast_to(r222_value, [$(49152)])
    tl.store($(candidate_vto_copy_130_ptr) + r222_z, r222_value)
    r223_row = tl.arange($(16))
    r223_col = tl.arange($(32))
    r223_lane = tl.arange($(192))
    for r223_group in range($(8)) {
      r223_left = tl.load($(candidate_vto_copy_129_ptr) + (((((((r223_group % $(8)) * $(3072)) + (r223_row[:, None] * $(192)) + r223_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r223_group % $(8)) * $(3072)) + (r223_row[:, None] * $(192)) + r223_lane[None, :]) / $(192)) % $(16)) * $(192)) + ((((r223_group % $(8)) * $(3072)) + (r223_row[:, None] * $(192)) + r223_lane[None, :]) % $(192))), dtype=tl.float32)
      r223_right = tl.load($(candidate_vto_copy_130_ptr) + (((((((r223_group % $(8)) * $(6144)) + (r223_lane[:, None] * $(32)) + r223_col[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r223_group % $(8)) * $(6144)) + (r223_lane[:, None] * $(32)) + r223_col[None, :]) % $(32)) * $(192)) + (((((r223_group % $(8)) * $(6144)) + (r223_lane[:, None] * $(32)) + r223_col[None, :]) / $(32)) % $(192))), dtype=tl.float32)
      r223_value = tl.dot(r223_left, r223_right)
      r223_value = r223_value.to(tl.float32)
      tl.store($(candidate_vbmm_55_ptr) + ((r223_group * $(512)) + (r223_row[:, None] * $(32)) + r223_col[None, :]), r223_value)
    }
    r224_z = tl.arange($(4096))
    r224_value = (tl.load($(candidate_vbmm_55_ptr) + ((((r224_z / $(512)) % $(8)) * $(512)) + (((r224_z / $(32)) % $(16)) * $(32)) + (r224_z % $(32))), dtype=tl.float32) * 0.07216878364870322)
    r224_value = r224_value.to(tl.float32)
    r224_value = tl.broadcast_to(r224_value, [$(4096)])
    tl.store($(candidate_vmul_55_ptr) + r224_z, r224_value)
    r225_z = tl.arange($(16))
    r225_value = (tl.cast(r225_z, tl.int64) + tl.full([], $(32), dtype=tl.int64))
    r225_value = tl.broadcast_to(r225_value, [$(16)])
    tl.store($(candidate_vadd_41_ptr) + r225_z, r225_value)
    r226_z = tl.arange($(16))
    r226_value = (tl.load($(candidate_vadd_41_ptr) + r226_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r226_value = tl.broadcast_to(r226_value, [$(16)])
    tl.store($(candidate_vsub_35_ptr) + r226_z, r226_value)
    r227_z = tl.arange($(512))
    r227_value = (tl.cast((r227_z % $(32)), tl.int64) > tl.load($(candidate_vsub_35_ptr) + ((r227_z / $(32)) % $(16)), dtype=tl.int64))
    r227_value = tl.broadcast_to(r227_value, [$(512)])
    tl.store($(candidate_vgt_20_ptr) + r227_z, r227_value)
    r228_z = tl.arange($(4096))
    r228_value = tl.where(tl.load($(candidate_vgt_20_ptr) + ((((r228_z / $(32)) % $(16)) * $(32)) + (r228_z % $(32))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_55_ptr) + ((((r228_z / $(512)) % $(8)) * $(512)) + (((r228_z / $(32)) % $(16)) * $(32)) + (r228_z % $(32))), dtype=tl.float32))
    r228_value = r228_value.to(tl.float32)
    r228_value = tl.broadcast_to(r228_value, [$(4096)])
    tl.store($(candidate_vmasked_fill_20_ptr) + r228_z, r228_value)
    r229_z = tl.arange($(4096))
    r229_lane = tl.arange($(32))
    r229_row_values = tl.load($(candidate_vmasked_fill_20_ptr) + ((((r229_z[:, None] / $(512)) % $(8)) * $(512)) + (((r229_z[:, None] / $(32)) % $(16)) * $(32)) + (r229_lane[None, :] % $(32))), dtype=tl.float32)
    r229_maximum = tl.max(r229_row_values, axis=1, keep_dims=true)
    r229_weights = tl.exp(r229_row_values - r229_maximum)
    r229_total = tl.sum(r229_weights, axis=1)
    r229_maximum = tl.max(r229_row_values, axis=1)
    r229_value = tl.exp(tl.load($(candidate_vmasked_fill_20_ptr) + ((((r229_z / $(512)) % $(8)) * $(512)) + (((r229_z / $(32)) % $(16)) * $(32)) + (r229_z % $(32))), dtype=tl.float32) - r229_maximum) / r229_total
    r229_value = r229_value.to(tl.float32)
    r229_value = tl.broadcast_to(r229_value, [$(4096)])
    tl.store($(candidate_vsoftmax_17_ptr) + r229_z, r229_value)
    r230_z = tl.arange($(4096))
    r230_value = tl.load($(candidate_vsoftmax_17_ptr) + ((((r230_z / $(512)) % $(8)) * $(512)) + (((r230_z / $(32)) % $(16)) * $(32)) + (r230_z % $(32))), dtype=tl.float32)
    r230_value = r230_value.to(tl.bfloat16)
    r230_value = tl.broadcast_to(r230_value, [$(4096)])
    tl.store($(candidate_vto_copy_131_ptr) + r230_z, r230_value)
    r231_row = tl.arange($(16))
    r231_col = tl.arange($(128))
    r231_lane = tl.arange($(32))
    for r231_group in range($(8)) {
      r231_left = tl.load($(candidate_vto_copy_131_ptr) + (((((((r231_group % $(8)) * $(512)) + (r231_row[:, None] * $(32)) + r231_lane[None, :]) / $(512)) % $(8)) * $(512)) + ((((((r231_group % $(8)) * $(512)) + (r231_row[:, None] * $(32)) + r231_lane[None, :]) / $(32)) % $(16)) * $(32)) + ((((r231_group % $(8)) * $(512)) + (r231_row[:, None] * $(32)) + r231_lane[None, :]) % $(32))), dtype=tl.bfloat16)
      r231_right = tl.load($(candidate_vbmm_50_ptr) + (((((((r231_group % $(8)) * $(4096)) + (r231_lane[:, None] * $(128)) + r231_col[None, :]) / $(128)) % $(32)) * $(1024)) + ((((((r231_group % $(8)) * $(4096)) + (r231_lane[:, None] * $(128)) + r231_col[None, :]) / $(4096)) % $(8)) * $(128)) + ((((r231_group % $(8)) * $(4096)) + (r231_lane[:, None] * $(128)) + r231_col[None, :]) % $(128)) + $(65536)), dtype=tl.bfloat16)
      r231_value = tl.dot(r231_left, r231_right)
      r231_value = r231_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_56_ptr) + ((r231_group * $(2048)) + (r231_row[:, None] * $(128)) + r231_col[None, :]), r231_value)
    }
    r232_z = tl.arange($(128))
    r232_lane = tl.arange($(32))
    r232_row_values = tl.load($(candidate_vmasked_fill_20_ptr) + ((((r232_z[:, None] / $(16)) % $(8)) * $(512)) + ((r232_z[:, None] % $(16)) * $(32)) + (r232_lane[None, :] % $(32))), dtype=tl.float32)
    r232_maximum = tl.max(r232_row_values, axis=1, keep_dims=true)
    r232_weights = tl.exp(r232_row_values - r232_maximum)
    r232_total = tl.sum(r232_weights, axis=1)
    r232_maximum = tl.max(r232_row_values, axis=1)
    r232_value = r232_maximum + tl.log(r232_total)
    r232_value = r232_value.to(tl.float32)
    r232_value = tl.broadcast_to(r232_value, [$(128)])
    tl.store($(candidate_vlogsumexp_17_ptr) + r232_z, r232_value)
    r233_z = tl.arange($(32768))
    r233_value = tl.where(((r233_z / $(1024)) % $(32)) < $(12), tl.load($(candidate_vbmm_52_ptr) + ((((((((r233_z / $(128)) % $(8)) * $(1536)) + ((((r233_z / $(1024)) % $(32)) - $(0)) * $(128)) + (r233_z % $(128))) / $(1536)) % $(8)) * $(1536)) + ((((((r233_z / $(128)) % $(8)) * $(12)) + (((r233_z / $(1024)) % $(32)) - $(0))) % $(12)) * $(128)) + (r233_z % $(128))), dtype=tl.bfloat16), tl.where(((r233_z / $(1024)) % $(32)) < $(16), tl.load($(candidate_vbmm_54_ptr) + ((((((((r233_z / $(128)) % $(8)) * $(512)) + ((((r233_z / $(1024)) % $(32)) - $(12)) * $(128)) + (r233_z % $(128))) / $(512)) % $(8)) * $(512)) + ((((((r233_z / $(128)) % $(8)) * $(4)) + (((r233_z / $(1024)) % $(32)) - $(12))) % $(4)) * $(128)) + (r233_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vbmm_56_ptr) + ((((((((r233_z / $(128)) % $(8)) * $(2048)) + ((((r233_z / $(1024)) % $(32)) - $(16)) * $(128)) + (r233_z % $(128))) / $(2048)) % $(8)) * $(2048)) + ((((((r233_z / $(128)) % $(8)) * $(16)) + (((r233_z / $(1024)) % $(32)) - $(16))) % $(16)) * $(128)) + (r233_z % $(128))), dtype=tl.bfloat16)))
    r233_value = r233_value.to(tl.bfloat16)
    r233_value = tl.broadcast_to(r233_value, [$(32768)])
    tl.store($(candidate_vcat_27_ptr) + r233_z, r233_value)
    r234_z = tl.arange($(256))
    r234_value = tl.where(((r234_z / $(8)) % $(32)) < $(12), tl.load($(candidate_vlogsumexp_15_ptr) + (((r234_z % $(8)) * $(12)) + ((((r234_z / $(8)) % $(32)) - $(0)) % $(12))), dtype=tl.float32), tl.where(((r234_z / $(8)) % $(32)) < $(16), tl.load($(candidate_vlogsumexp_16_ptr) + (((r234_z % $(8)) * $(4)) + ((((r234_z / $(8)) % $(32)) - $(12)) % $(4))), dtype=tl.float32), tl.load($(candidate_vlogsumexp_17_ptr) + (((r234_z % $(8)) * $(16)) + ((((r234_z / $(8)) % $(32)) - $(16)) % $(16))), dtype=tl.float32)))
    r234_value = r234_value.to(tl.float32)
    r234_value = tl.broadcast_to(r234_value, [$(256)])
    tl.store($(candidate_vcat_28_ptr) + r234_z, r234_value)
    r235_z = tl.arange($(18432))
    r235_value = tl.load($(candidate_vcat_25_ptr) + ((((r235_z / $(192)) % $(12)) * $(1536)) + (((r235_z / $(2304)) % $(8)) * $(192)) + (r235_z % $(192))), dtype=tl.bfloat16)
    r235_value = r235_value.to(tl.float32)
    r235_value = tl.broadcast_to(r235_value, [$(18432)])
    tl.store($(candidate_vto_copy_132_ptr) + r235_z, r235_value)
    r236_z = tl.arange($(36864))
    r236_value = tl.load($(candidate_vcat_26_ptr) + ((((r236_z / $(192)) % $(24)) * $(1536)) + (((r236_z / $(4608)) % $(8)) * $(192)) + (r236_z % $(192))), dtype=tl.bfloat16)
    r236_value = r236_value.to(tl.float32)
    r236_value = tl.broadcast_to(r236_value, [$(36864)])
    tl.store($(candidate_vto_copy_133_ptr) + r236_z, r236_value)
    r237_z = tl.arange($(24576))
    r237_value = tl.load($(candidate_vbmm_50_ptr) + ((((r237_z / $(128)) % $(24)) * $(1024)) + (((r237_z / $(3072)) % $(8)) * $(128)) + (r237_z % $(128))), dtype=tl.bfloat16)
    r237_value = r237_value.to(tl.float32)
    r237_value = tl.broadcast_to(r237_value, [$(24576)])
    tl.store($(candidate_vto_copy_134_ptr) + r237_z, r237_value)
    r238_z = tl.arange($(12288))
    r238_value = tl.load($(candidate_vmm_28_ptr) + ((((r238_z / $(128)) % $(12)) * $(1024)) + (((r238_z / $(1536)) % $(8)) * $(128)) + (r238_z % $(128)) + $(65536)), dtype=tl.bfloat16)
    r238_value = r238_value.to(tl.float32)
    r238_value = tl.broadcast_to(r238_value, [$(12288)])
    tl.store($(candidate_vto_copy_135_ptr) + r238_z, r238_value)
    r239_z = tl.arange($(12288))
    r239_value = tl.load($(candidate_vcat_27_ptr) + ((((r239_z / $(128)) % $(12)) * $(1024)) + (((r239_z / $(1536)) % $(8)) * $(128)) + (r239_z % $(128))), dtype=tl.bfloat16)
    r239_value = r239_value.to(tl.float32)
    r239_value = tl.broadcast_to(r239_value, [$(12288)])
    tl.store($(candidate_vto_copy_136_ptr) + r239_z, r239_value)
    r240_row = tl.arange($(12))
    r240_col = tl.arange($(24))
    r240_lane = tl.arange($(192))
    for r240_group in range($(8)) {
      r240_left = tl.load($(candidate_vto_copy_132_ptr) + (((((((r240_group % $(8)) * $(2304)) + (r240_row[:, None] * $(192)) + r240_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r240_group % $(8)) * $(2304)) + (r240_row[:, None] * $(192)) + r240_lane[None, :]) / $(192)) % $(12)) * $(192)) + ((((r240_group % $(8)) * $(2304)) + (r240_row[:, None] * $(192)) + r240_lane[None, :]) % $(192))), dtype=tl.float32)
      r240_right = tl.load($(candidate_vto_copy_133_ptr) + (((((((r240_group % $(8)) * $(4608)) + (r240_lane[:, None] * $(24)) + r240_col[None, :]) / $(4608)) % $(8)) * $(4608)) + (((((r240_group % $(8)) * $(4608)) + (r240_lane[:, None] * $(24)) + r240_col[None, :]) % $(24)) * $(192)) + (((((r240_group % $(8)) * $(4608)) + (r240_lane[:, None] * $(24)) + r240_col[None, :]) / $(24)) % $(192))), dtype=tl.float32)
      r240_value = tl.dot(r240_left, r240_right)
      r240_value = r240_value.to(tl.float32)
      tl.store($(candidate_vbmm_57_ptr) + ((r240_group * $(288)) + (r240_row[:, None] * $(24)) + r240_col[None, :]), r240_value)
    }
    r241_z = tl.arange($(2304))
    r241_value = (tl.load($(candidate_vbmm_57_ptr) + ((((r241_z / $(288)) % $(8)) * $(288)) + (((r241_z / $(24)) % $(12)) * $(24)) + (r241_z % $(24))), dtype=tl.float32) * 0.07216878364870322)
    r241_value = r241_value.to(tl.float32)
    r241_value = tl.broadcast_to(r241_value, [$(2304)])
    tl.store($(candidate_vmul_56_ptr) + r241_z, r241_value)
    r242_z = tl.arange($(12))
    r242_value = (tl.cast(r242_z, tl.int64) + tl.full([], $(24), dtype=tl.int64))
    r242_value = tl.broadcast_to(r242_value, [$(12)])
    tl.store($(candidate_vadd_42_ptr) + r242_z, r242_value)
    r243_z = tl.arange($(12))
    r243_value = (tl.load($(candidate_vadd_42_ptr) + r243_z, dtype=tl.int64) - tl.full([], $(12), dtype=tl.int64))
    r243_value = tl.broadcast_to(r243_value, [$(12)])
    tl.store($(candidate_vsub_36_ptr) + r243_z, r243_value)
    r244_z = tl.arange($(288))
    r244_value = (tl.cast((r244_z % $(24)), tl.int64) > tl.load($(candidate_vsub_36_ptr) + ((r244_z / $(24)) % $(12)), dtype=tl.int64))
    r244_value = tl.broadcast_to(r244_value, [$(288)])
    tl.store($(candidate_vgt_21_ptr) + r244_z, r244_value)
    r245_z = tl.arange($(2304))
    r245_value = tl.where(tl.load($(candidate_vgt_21_ptr) + ((((r245_z / $(24)) % $(12)) * $(24)) + (r245_z % $(24))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_56_ptr) + ((((r245_z / $(288)) % $(8)) * $(288)) + (((r245_z / $(24)) % $(12)) * $(24)) + (r245_z % $(24))), dtype=tl.float32))
    r245_value = r245_value.to(tl.float32)
    r245_value = tl.broadcast_to(r245_value, [$(2304)])
    tl.store($(candidate_vmasked_fill_21_ptr) + r245_z, r245_value)
    r246_z = tl.arange($(2304))
    r246_value = (tl.load($(candidate_vmasked_fill_21_ptr) + ((((r246_z / $(288)) % $(8)) * $(288)) + (((r246_z / $(24)) % $(12)) * $(24)) + (r246_z % $(24))), dtype=tl.float32) - tl.load($(candidate_vcat_28_ptr) + ((((r246_z / $(24)) % $(12)) * $(8)) + ((r246_z / $(288)) % $(8))), dtype=tl.float32))
    r246_value = r246_value.to(tl.float32)
    r246_value = tl.broadcast_to(r246_value, [$(2304)])
    tl.store($(candidate_vsub_37_ptr) + r246_z, r246_value)
    r247_z = tl.arange($(2304))
    r247_value = tl.exp(tl.load($(candidate_vsub_37_ptr) + ((((r247_z / $(288)) % $(8)) * $(288)) + (((r247_z / $(24)) % $(12)) * $(24)) + (r247_z % $(24))), dtype=tl.float32))
    r247_value = r247_value.to(tl.float32)
    r247_value = tl.broadcast_to(r247_value, [$(2304)])
    tl.store($(candidate_vexp_3_ptr) + r247_z, r247_value)
    r248_z = tl.arange($(12288))
    r248_value = (tl.load($(candidate_vto_copy_135_ptr) + ((((r248_z / $(1536)) % $(8)) * $(1536)) + (((r248_z / $(128)) % $(12)) * $(128)) + (r248_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_136_ptr) + ((((r248_z / $(1536)) % $(8)) * $(1536)) + (((r248_z / $(128)) % $(12)) * $(128)) + (r248_z % $(128))), dtype=tl.float32))
    r248_value = r248_value.to(tl.float32)
    r248_value = tl.broadcast_to(r248_value, [$(12288)])
    tl.store($(candidate_vmul_57_ptr) + r248_z, r248_value)
    r249_z = tl.arange($(96))
    r249_lane = tl.arange($(128))
    r249_value = tl.sum(tl.load($(candidate_vmul_57_ptr) + ((((r249_z[:, None] / $(12)) % $(8)) * $(1536)) + ((r249_z[:, None] % $(12)) * $(128)) + r249_lane[None, :]), dtype=tl.float32), axis=1)
    r249_value = r249_value.to(tl.float32)
    r249_value = tl.broadcast_to(r249_value, [$(96)])
    tl.store($(candidate_vsum_5_ptr) + r249_z, r249_value)
    r250_row = tl.arange($(12))
    r250_col = tl.arange($(24))
    r250_lane = tl.arange($(128))
    for r250_group in range($(8)) {
      r250_left = tl.load($(candidate_vto_copy_135_ptr) + (((((((r250_group % $(8)) * $(1536)) + (r250_row[:, None] * $(128)) + r250_lane[None, :]) / $(1536)) % $(8)) * $(1536)) + ((((((r250_group % $(8)) * $(1536)) + (r250_row[:, None] * $(128)) + r250_lane[None, :]) / $(128)) % $(12)) * $(128)) + ((((r250_group % $(8)) * $(1536)) + (r250_row[:, None] * $(128)) + r250_lane[None, :]) % $(128))), dtype=tl.float32)
      r250_right = tl.load($(candidate_vto_copy_134_ptr) + (((((((r250_group % $(8)) * $(3072)) + (r250_lane[:, None] * $(24)) + r250_col[None, :]) / $(3072)) % $(8)) * $(3072)) + (((((r250_group % $(8)) * $(3072)) + (r250_lane[:, None] * $(24)) + r250_col[None, :]) % $(24)) * $(128)) + (((((r250_group % $(8)) * $(3072)) + (r250_lane[:, None] * $(24)) + r250_col[None, :]) / $(24)) % $(128))), dtype=tl.float32)
      r250_value = tl.dot(r250_left, r250_right)
      r250_value = r250_value.to(tl.float32)
      tl.store($(candidate_vbmm_58_ptr) + ((r250_group * $(288)) + (r250_row[:, None] * $(24)) + r250_col[None, :]), r250_value)
    }
    r251_z = tl.arange($(2304))
    r251_value = (tl.load($(candidate_vbmm_58_ptr) + ((((r251_z / $(288)) % $(8)) * $(288)) + (((r251_z / $(24)) % $(12)) * $(24)) + (r251_z % $(24))), dtype=tl.float32) - tl.load($(candidate_vsum_5_ptr) + ((((r251_z / $(288)) % $(8)) * $(12)) + ((r251_z / $(24)) % $(12))), dtype=tl.float32))
    r251_value = r251_value.to(tl.float32)
    r251_value = tl.broadcast_to(r251_value, [$(2304)])
    tl.store($(candidate_vsub_38_ptr) + r251_z, r251_value)
    r252_z = tl.arange($(2304))
    r252_value = (tl.load($(candidate_vexp_3_ptr) + ((((r252_z / $(288)) % $(8)) * $(288)) + (((r252_z / $(24)) % $(12)) * $(24)) + (r252_z % $(24))), dtype=tl.float32) * tl.load($(candidate_vsub_38_ptr) + ((((r252_z / $(288)) % $(8)) * $(288)) + (((r252_z / $(24)) % $(12)) * $(24)) + (r252_z % $(24))), dtype=tl.float32))
    r252_value = r252_value.to(tl.float32)
    r252_value = tl.broadcast_to(r252_value, [$(2304)])
    tl.store($(candidate_vmul_58_ptr) + r252_z, r252_value)
    r253_row = tl.arange($(24))
    r253_col = tl.arange($(192))
    r253_lane = tl.arange($(12))
    for r253_group in range($(8)) {
      r253_left = tl.load($(candidate_vmul_58_ptr) + (((((((r253_group % $(8)) * $(288)) + (r253_row[:, None] * $(12)) + r253_lane[None, :]) / $(288)) % $(8)) * $(288)) + (((((r253_group % $(8)) * $(288)) + (r253_row[:, None] * $(12)) + r253_lane[None, :]) % $(12)) * $(24)) + (((((r253_group % $(8)) * $(288)) + (r253_row[:, None] * $(12)) + r253_lane[None, :]) / $(12)) % $(24))), dtype=tl.float32)
      r253_right = tl.load($(candidate_vto_copy_132_ptr) + (((((((r253_group % $(8)) * $(2304)) + (r253_lane[:, None] * $(192)) + r253_col[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r253_group % $(8)) * $(2304)) + (r253_lane[:, None] * $(192)) + r253_col[None, :]) / $(192)) % $(12)) * $(192)) + ((((r253_group % $(8)) * $(2304)) + (r253_lane[:, None] * $(192)) + r253_col[None, :]) % $(192))), dtype=tl.float32)
      r253_value = tl.dot(r253_left, r253_right)
      r253_value = r253_value.to(tl.float32)
      tl.store($(candidate_vbmm_60_ptr) + ((r253_group * $(4608)) + (r253_row[:, None] * $(192)) + r253_col[None, :]), r253_value)
    }
    r254_z = tl.arange($(36864))
    r254_value = (tl.load($(candidate_vbmm_60_ptr) + ((((r254_z / $(4608)) % $(8)) * $(4608)) + (((r254_z / $(192)) % $(24)) * $(192)) + (r254_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r254_value = r254_value.to(tl.float32)
    r254_value = tl.broadcast_to(r254_value, [$(36864)])
    tl.store($(candidate_vmul_60_ptr) + r254_z, r254_value)
    r255_z = tl.arange($(36864))
    r255_value = tl.load($(candidate_vmul_60_ptr) + ((((r255_z / $(192)) % $(8)) * $(4608)) + (((r255_z / $(1536)) % $(24)) * $(192)) + (r255_z % $(192))), dtype=tl.float32)
    r255_value = r255_value.to(tl.bfloat16)
    r255_value = tl.broadcast_to(r255_value, [$(36864)])
    tl.store($(candidate_vto_copy_138_ptr) + r255_z, r255_value)
    r256_row = tl.arange($(24))
    r256_col = tl.arange($(128))
    r256_lane = tl.arange($(12))
    for r256_group in range($(8)) {
      r256_left = tl.load($(candidate_vexp_3_ptr) + (((((((r256_group % $(8)) * $(288)) + (r256_row[:, None] * $(12)) + r256_lane[None, :]) / $(288)) % $(8)) * $(288)) + (((((r256_group % $(8)) * $(288)) + (r256_row[:, None] * $(12)) + r256_lane[None, :]) % $(12)) * $(24)) + (((((r256_group % $(8)) * $(288)) + (r256_row[:, None] * $(12)) + r256_lane[None, :]) / $(12)) % $(24))), dtype=tl.float32)
      r256_right = tl.load($(candidate_vto_copy_135_ptr) + (((((((r256_group % $(8)) * $(1536)) + (r256_lane[:, None] * $(128)) + r256_col[None, :]) / $(1536)) % $(8)) * $(1536)) + ((((((r256_group % $(8)) * $(1536)) + (r256_lane[:, None] * $(128)) + r256_col[None, :]) / $(128)) % $(12)) * $(128)) + ((((r256_group % $(8)) * $(1536)) + (r256_lane[:, None] * $(128)) + r256_col[None, :]) % $(128))), dtype=tl.float32)
      r256_value = tl.dot(r256_left, r256_right)
      r256_value = r256_value.to(tl.float32)
      tl.store($(candidate_vbmm_61_ptr) + ((r256_group * $(3072)) + (r256_row[:, None] * $(128)) + r256_col[None, :]), r256_value)
    }
    r257_z = tl.arange($(24576))
    r257_value = tl.load($(candidate_vbmm_61_ptr) + ((((r257_z / $(128)) % $(8)) * $(3072)) + (((r257_z / $(1024)) % $(24)) * $(128)) + (r257_z % $(128))), dtype=tl.float32)
    r257_value = r257_value.to(tl.bfloat16)
    r257_value = tl.broadcast_to(r257_value, [$(24576)])
    tl.store($(candidate_vto_copy_139_ptr) + r257_z, r257_value)
    r258_z = tl.arange($(73728))
    r258_value = tl.where(((r258_z / $(1536)) % $(48)) < $(24), tl.load($(candidate_vto_copy_138_ptr) + ((((((r258_z / $(1536)) % $(48)) - $(0)) % $(24)) * $(1536)) + (((r258_z / $(192)) % $(8)) * $(192)) + (r258_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r258_value = r258_value.to(tl.bfloat16)
    r258_value = tl.broadcast_to(r258_value, [$(73728)])
    tl.store($(candidate_vcat_29_ptr) + r258_z, r258_value)
    r259_z = tl.arange($(49152))
    r259_value = tl.where(((r259_z / $(1024)) % $(48)) < $(24), tl.load($(candidate_vto_copy_139_ptr) + ((((((r259_z / $(1024)) % $(48)) - $(0)) % $(24)) * $(1024)) + (((r259_z / $(128)) % $(8)) * $(128)) + (r259_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r259_value = r259_value.to(tl.bfloat16)
    r259_value = tl.broadcast_to(r259_value, [$(49152)])
    tl.store($(candidate_vcat_30_ptr) + r259_z, r259_value)
    r260_z = tl.arange($(6144))
    r260_value = tl.load($(candidate_vcat_25_ptr) + ((((r260_z / $(192)) % $(4)) * $(1536)) + (((r260_z / $(768)) % $(8)) * $(192)) + (r260_z % $(192)) + $(18432)), dtype=tl.bfloat16)
    r260_value = r260_value.to(tl.float32)
    r260_value = tl.broadcast_to(r260_value, [$(6144)])
    tl.store($(candidate_vto_copy_140_ptr) + r260_z, r260_value)
    r261_z = tl.arange($(12288))
    r261_value = tl.load($(candidate_vcat_26_ptr) + ((((r261_z / $(192)) % $(8)) * $(1536)) + (((r261_z / $(1536)) % $(8)) * $(192)) + (r261_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r261_value = r261_value.to(tl.float32)
    r261_value = tl.broadcast_to(r261_value, [$(12288)])
    tl.store($(candidate_vto_copy_141_ptr) + r261_z, r261_value)
    r262_z = tl.arange($(8192))
    r262_value = tl.load($(candidate_vbmm_50_ptr) + ((((r262_z / $(128)) % $(8)) * $(1024)) + (((r262_z / $(1024)) % $(8)) * $(128)) + (r262_z % $(128)) + $(49152)), dtype=tl.bfloat16)
    r262_value = r262_value.to(tl.float32)
    r262_value = tl.broadcast_to(r262_value, [$(8192)])
    tl.store($(candidate_vto_copy_142_ptr) + r262_z, r262_value)
    r263_z = tl.arange($(4096))
    r263_value = tl.load($(candidate_vmm_28_ptr) + ((((r263_z / $(128)) % $(4)) * $(1024)) + (((r263_z / $(512)) % $(8)) * $(128)) + (r263_z % $(128)) + $(77824)), dtype=tl.bfloat16)
    r263_value = r263_value.to(tl.float32)
    r263_value = tl.broadcast_to(r263_value, [$(4096)])
    tl.store($(candidate_vto_copy_143_ptr) + r263_z, r263_value)
    r264_z = tl.arange($(4096))
    r264_value = tl.load($(candidate_vcat_27_ptr) + ((((r264_z / $(128)) % $(4)) * $(1024)) + (((r264_z / $(512)) % $(8)) * $(128)) + (r264_z % $(128)) + $(12288)), dtype=tl.bfloat16)
    r264_value = r264_value.to(tl.float32)
    r264_value = tl.broadcast_to(r264_value, [$(4096)])
    tl.store($(candidate_vto_copy_144_ptr) + r264_z, r264_value)
    r265_row = tl.arange($(4))
    r265_col = tl.arange($(8))
    r265_lane = tl.arange($(192))
    for r265_group in range($(8)) {
      r265_left = tl.load($(candidate_vto_copy_140_ptr) + (((((((r265_group % $(8)) * $(768)) + (r265_row[:, None] * $(192)) + r265_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r265_group % $(8)) * $(768)) + (r265_row[:, None] * $(192)) + r265_lane[None, :]) / $(192)) % $(4)) * $(192)) + ((((r265_group % $(8)) * $(768)) + (r265_row[:, None] * $(192)) + r265_lane[None, :]) % $(192))), dtype=tl.float32)
      r265_right = tl.load($(candidate_vto_copy_141_ptr) + (((((((r265_group % $(8)) * $(1536)) + (r265_lane[:, None] * $(8)) + r265_col[None, :]) / $(1536)) % $(8)) * $(1536)) + (((((r265_group % $(8)) * $(1536)) + (r265_lane[:, None] * $(8)) + r265_col[None, :]) % $(8)) * $(192)) + (((((r265_group % $(8)) * $(1536)) + (r265_lane[:, None] * $(8)) + r265_col[None, :]) / $(8)) % $(192))), dtype=tl.float32)
      r265_value = tl.dot(r265_left, r265_right)
      r265_value = r265_value.to(tl.float32)
      tl.store($(candidate_vbmm_62_ptr) + ((r265_group * $(32)) + (r265_row[:, None] * $(8)) + r265_col[None, :]), r265_value)
    }
    r266_z = tl.arange($(256))
    r266_value = (tl.load($(candidate_vbmm_62_ptr) + ((((r266_z / $(32)) % $(8)) * $(32)) + (((r266_z / $(8)) % $(4)) * $(8)) + (r266_z % $(8))), dtype=tl.float32) * 0.07216878364870322)
    r266_value = r266_value.to(tl.float32)
    r266_value = tl.broadcast_to(r266_value, [$(256)])
    tl.store($(candidate_vmul_61_ptr) + r266_z, r266_value)
    r267_z = tl.arange($(4))
    r267_value = (tl.cast(r267_z, tl.int64) + tl.full([], $(8), dtype=tl.int64))
    r267_value = tl.broadcast_to(r267_value, [$(4)])
    tl.store($(candidate_vadd_43_ptr) + r267_z, r267_value)
    r268_z = tl.arange($(4))
    r268_value = (tl.load($(candidate_vadd_43_ptr) + r268_z, dtype=tl.int64) - tl.full([], $(4), dtype=tl.int64))
    r268_value = tl.broadcast_to(r268_value, [$(4)])
    tl.store($(candidate_vsub_39_ptr) + r268_z, r268_value)
    r269_z = tl.arange($(32))
    r269_value = (tl.cast((r269_z % $(8)), tl.int64) > tl.load($(candidate_vsub_39_ptr) + ((r269_z / $(8)) % $(4)), dtype=tl.int64))
    r269_value = tl.broadcast_to(r269_value, [$(32)])
    tl.store($(candidate_vgt_22_ptr) + r269_z, r269_value)
    r270_z = tl.arange($(256))
    r270_value = tl.where(tl.load($(candidate_vgt_22_ptr) + ((((r270_z / $(8)) % $(4)) * $(8)) + (r270_z % $(8))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_61_ptr) + ((((r270_z / $(32)) % $(8)) * $(32)) + (((r270_z / $(8)) % $(4)) * $(8)) + (r270_z % $(8))), dtype=tl.float32))
    r270_value = r270_value.to(tl.float32)
    r270_value = tl.broadcast_to(r270_value, [$(256)])
    tl.store($(candidate_vmasked_fill_22_ptr) + r270_z, r270_value)
    r271_z = tl.arange($(256))
    r271_value = (tl.load($(candidate_vmasked_fill_22_ptr) + ((((r271_z / $(32)) % $(8)) * $(32)) + (((r271_z / $(8)) % $(4)) * $(8)) + (r271_z % $(8))), dtype=tl.float32) - tl.load($(candidate_vcat_28_ptr) + ((((r271_z / $(8)) % $(4)) * $(8)) + ((r271_z / $(32)) % $(8)) + $(96)), dtype=tl.float32))
    r271_value = r271_value.to(tl.float32)
    r271_value = tl.broadcast_to(r271_value, [$(256)])
    tl.store($(candidate_vsub_40_ptr) + r271_z, r271_value)
    r272_z = tl.arange($(256))
    r272_value = tl.exp(tl.load($(candidate_vsub_40_ptr) + ((((r272_z / $(32)) % $(8)) * $(32)) + (((r272_z / $(8)) % $(4)) * $(8)) + (r272_z % $(8))), dtype=tl.float32))
    r272_value = r272_value.to(tl.float32)
    r272_value = tl.broadcast_to(r272_value, [$(256)])
    tl.store($(candidate_vexp_4_ptr) + r272_z, r272_value)
    r273_z = tl.arange($(4096))
    r273_value = (tl.load($(candidate_vto_copy_143_ptr) + ((((r273_z / $(512)) % $(8)) * $(512)) + (((r273_z / $(128)) % $(4)) * $(128)) + (r273_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_144_ptr) + ((((r273_z / $(512)) % $(8)) * $(512)) + (((r273_z / $(128)) % $(4)) * $(128)) + (r273_z % $(128))), dtype=tl.float32))
    r273_value = r273_value.to(tl.float32)
    r273_value = tl.broadcast_to(r273_value, [$(4096)])
    tl.store($(candidate_vmul_62_ptr) + r273_z, r273_value)
    r274_z = tl.arange($(32))
    r274_lane = tl.arange($(128))
    r274_value = tl.sum(tl.load($(candidate_vmul_62_ptr) + ((((r274_z[:, None] / $(4)) % $(8)) * $(512)) + ((r274_z[:, None] % $(4)) * $(128)) + r274_lane[None, :]), dtype=tl.float32), axis=1)
    r274_value = r274_value.to(tl.float32)
    r274_value = tl.broadcast_to(r274_value, [$(32)])
    tl.store($(candidate_vsum_6_ptr) + r274_z, r274_value)
    r275_row = tl.arange($(4))
    r275_col = tl.arange($(8))
    r275_lane = tl.arange($(128))
    for r275_group in range($(8)) {
      r275_left = tl.load($(candidate_vto_copy_143_ptr) + (((((((r275_group % $(8)) * $(512)) + (r275_row[:, None] * $(128)) + r275_lane[None, :]) / $(512)) % $(8)) * $(512)) + ((((((r275_group % $(8)) * $(512)) + (r275_row[:, None] * $(128)) + r275_lane[None, :]) / $(128)) % $(4)) * $(128)) + ((((r275_group % $(8)) * $(512)) + (r275_row[:, None] * $(128)) + r275_lane[None, :]) % $(128))), dtype=tl.float32)
      r275_right = tl.load($(candidate_vto_copy_142_ptr) + (((((((r275_group % $(8)) * $(1024)) + (r275_lane[:, None] * $(8)) + r275_col[None, :]) / $(1024)) % $(8)) * $(1024)) + (((((r275_group % $(8)) * $(1024)) + (r275_lane[:, None] * $(8)) + r275_col[None, :]) % $(8)) * $(128)) + (((((r275_group % $(8)) * $(1024)) + (r275_lane[:, None] * $(8)) + r275_col[None, :]) / $(8)) % $(128))), dtype=tl.float32)
      r275_value = tl.dot(r275_left, r275_right)
      r275_value = r275_value.to(tl.float32)
      tl.store($(candidate_vbmm_63_ptr) + ((r275_group * $(32)) + (r275_row[:, None] * $(8)) + r275_col[None, :]), r275_value)
    }
    r276_z = tl.arange($(256))
    r276_value = (tl.load($(candidate_vbmm_63_ptr) + ((((r276_z / $(32)) % $(8)) * $(32)) + (((r276_z / $(8)) % $(4)) * $(8)) + (r276_z % $(8))), dtype=tl.float32) - tl.load($(candidate_vsum_6_ptr) + ((((r276_z / $(32)) % $(8)) * $(4)) + ((r276_z / $(8)) % $(4))), dtype=tl.float32))
    r276_value = r276_value.to(tl.float32)
    r276_value = tl.broadcast_to(r276_value, [$(256)])
    tl.store($(candidate_vsub_41_ptr) + r276_z, r276_value)
    r277_z = tl.arange($(256))
    r277_value = (tl.load($(candidate_vexp_4_ptr) + ((((r277_z / $(32)) % $(8)) * $(32)) + (((r277_z / $(8)) % $(4)) * $(8)) + (r277_z % $(8))), dtype=tl.float32) * tl.load($(candidate_vsub_41_ptr) + ((((r277_z / $(32)) % $(8)) * $(32)) + (((r277_z / $(8)) % $(4)) * $(8)) + (r277_z % $(8))), dtype=tl.float32))
    r277_value = r277_value.to(tl.float32)
    r277_value = tl.broadcast_to(r277_value, [$(256)])
    tl.store($(candidate_vmul_63_ptr) + r277_z, r277_value)
    r278_row = tl.arange($(8))
    r278_col = tl.arange($(192))
    r278_lane = tl.arange($(4))
    for r278_group in range($(8)) {
      r278_left = tl.load($(candidate_vmul_63_ptr) + (((((((r278_group % $(8)) * $(32)) + (r278_row[:, None] * $(4)) + r278_lane[None, :]) / $(32)) % $(8)) * $(32)) + (((((r278_group % $(8)) * $(32)) + (r278_row[:, None] * $(4)) + r278_lane[None, :]) % $(4)) * $(8)) + (((((r278_group % $(8)) * $(32)) + (r278_row[:, None] * $(4)) + r278_lane[None, :]) / $(4)) % $(8))), dtype=tl.float32)
      r278_right = tl.load($(candidate_vto_copy_140_ptr) + (((((((r278_group % $(8)) * $(768)) + (r278_lane[:, None] * $(192)) + r278_col[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r278_group % $(8)) * $(768)) + (r278_lane[:, None] * $(192)) + r278_col[None, :]) / $(192)) % $(4)) * $(192)) + ((((r278_group % $(8)) * $(768)) + (r278_lane[:, None] * $(192)) + r278_col[None, :]) % $(192))), dtype=tl.float32)
      r278_value = tl.dot(r278_left, r278_right)
      r278_value = r278_value.to(tl.float32)
      tl.store($(candidate_vbmm_65_ptr) + ((r278_group * $(1536)) + (r278_row[:, None] * $(192)) + r278_col[None, :]), r278_value)
    }
    r279_z = tl.arange($(12288))
    r279_value = (tl.load($(candidate_vbmm_65_ptr) + ((((r279_z / $(1536)) % $(8)) * $(1536)) + (((r279_z / $(192)) % $(8)) * $(192)) + (r279_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r279_value = r279_value.to(tl.float32)
    r279_value = tl.broadcast_to(r279_value, [$(12288)])
    tl.store($(candidate_vmul_65_ptr) + r279_z, r279_value)
    r280_z = tl.arange($(12288))
    r280_value = tl.load($(candidate_vmul_65_ptr) + ((((r280_z / $(192)) % $(8)) * $(1536)) + (((r280_z / $(1536)) % $(8)) * $(192)) + (r280_z % $(192))), dtype=tl.float32)
    r280_value = r280_value.to(tl.bfloat16)
    r280_value = tl.broadcast_to(r280_value, [$(12288)])
    tl.store($(candidate_vto_copy_146_ptr) + r280_z, r280_value)
    r281_row = tl.arange($(8))
    r281_col = tl.arange($(128))
    r281_lane = tl.arange($(4))
    for r281_group in range($(8)) {
      r281_left = tl.load($(candidate_vexp_4_ptr) + (((((((r281_group % $(8)) * $(32)) + (r281_row[:, None] * $(4)) + r281_lane[None, :]) / $(32)) % $(8)) * $(32)) + (((((r281_group % $(8)) * $(32)) + (r281_row[:, None] * $(4)) + r281_lane[None, :]) % $(4)) * $(8)) + (((((r281_group % $(8)) * $(32)) + (r281_row[:, None] * $(4)) + r281_lane[None, :]) / $(4)) % $(8))), dtype=tl.float32)
      r281_right = tl.load($(candidate_vto_copy_143_ptr) + (((((((r281_group % $(8)) * $(512)) + (r281_lane[:, None] * $(128)) + r281_col[None, :]) / $(512)) % $(8)) * $(512)) + ((((((r281_group % $(8)) * $(512)) + (r281_lane[:, None] * $(128)) + r281_col[None, :]) / $(128)) % $(4)) * $(128)) + ((((r281_group % $(8)) * $(512)) + (r281_lane[:, None] * $(128)) + r281_col[None, :]) % $(128))), dtype=tl.float32)
      r281_value = tl.dot(r281_left, r281_right)
      r281_value = r281_value.to(tl.float32)
      tl.store($(candidate_vbmm_66_ptr) + ((r281_group * $(1024)) + (r281_row[:, None] * $(128)) + r281_col[None, :]), r281_value)
    }
    r282_z = tl.arange($(8192))
    r282_value = tl.load($(candidate_vbmm_66_ptr) + ((((r282_z / $(128)) % $(8)) * $(1024)) + (((r282_z / $(1024)) % $(8)) * $(128)) + (r282_z % $(128))), dtype=tl.float32)
    r282_value = r282_value.to(tl.bfloat16)
    r282_value = tl.broadcast_to(r282_value, [$(8192)])
    tl.store($(candidate_vto_copy_147_ptr) + r282_z, r282_value)
    r283_z = tl.arange($(24576))
    r283_value = tl.where(((r283_z / $(1536)) % $(16)) < $(8), tl.load($(candidate_vto_copy_146_ptr) + ((((((r283_z / $(1536)) % $(16)) - $(0)) % $(8)) * $(1536)) + (((r283_z / $(192)) % $(8)) * $(192)) + (r283_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r283_value = r283_value.to(tl.bfloat16)
    r283_value = tl.broadcast_to(r283_value, [$(24576)])
    tl.store($(candidate_vcat_31_ptr) + r283_z, r283_value)
    r284_z = tl.arange($(16384))
    r284_value = tl.where(((r284_z / $(1024)) % $(16)) < $(8), tl.load($(candidate_vto_copy_147_ptr) + ((((((r284_z / $(1024)) % $(16)) - $(0)) % $(8)) * $(1024)) + (((r284_z / $(128)) % $(8)) * $(128)) + (r284_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r284_value = r284_value.to(tl.bfloat16)
    r284_value = tl.broadcast_to(r284_value, [$(16384)])
    tl.store($(candidate_vcat_32_ptr) + r284_z, r284_value)
    r285_z = tl.arange($(24576))
    r285_value = tl.load($(candidate_vcat_25_ptr) + ((((r285_z / $(192)) % $(16)) * $(1536)) + (((r285_z / $(3072)) % $(8)) * $(192)) + (r285_z % $(192)) + $(24576)), dtype=tl.bfloat16)
    r285_value = r285_value.to(tl.float32)
    r285_value = tl.broadcast_to(r285_value, [$(24576)])
    tl.store($(candidate_vto_copy_148_ptr) + r285_z, r285_value)
    r286_z = tl.arange($(49152))
    r286_value = tl.load($(candidate_vcat_26_ptr) + ((((r286_z / $(192)) % $(32)) * $(1536)) + (((r286_z / $(6144)) % $(8)) * $(192)) + (r286_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r286_value = r286_value.to(tl.float32)
    r286_value = tl.broadcast_to(r286_value, [$(49152)])
    tl.store($(candidate_vto_copy_149_ptr) + r286_z, r286_value)
    r287_z = tl.arange($(32768))
    r287_value = tl.load($(candidate_vbmm_50_ptr) + ((((r287_z / $(128)) % $(32)) * $(1024)) + (((r287_z / $(4096)) % $(8)) * $(128)) + (r287_z % $(128)) + $(65536)), dtype=tl.bfloat16)
    r287_value = r287_value.to(tl.float32)
    r287_value = tl.broadcast_to(r287_value, [$(32768)])
    tl.store($(candidate_vto_copy_150_ptr) + r287_z, r287_value)
    r288_z = tl.arange($(16384))
    r288_value = tl.load($(candidate_vmm_28_ptr) + ((((r288_z / $(128)) % $(16)) * $(1024)) + (((r288_z / $(2048)) % $(8)) * $(128)) + (r288_z % $(128)) + $(81920)), dtype=tl.bfloat16)
    r288_value = r288_value.to(tl.float32)
    r288_value = tl.broadcast_to(r288_value, [$(16384)])
    tl.store($(candidate_vto_copy_151_ptr) + r288_z, r288_value)
    r289_z = tl.arange($(16384))
    r289_value = tl.load($(candidate_vcat_27_ptr) + ((((r289_z / $(128)) % $(16)) * $(1024)) + (((r289_z / $(2048)) % $(8)) * $(128)) + (r289_z % $(128)) + $(16384)), dtype=tl.bfloat16)
    r289_value = r289_value.to(tl.float32)
    r289_value = tl.broadcast_to(r289_value, [$(16384)])
    tl.store($(candidate_vto_copy_152_ptr) + r289_z, r289_value)
    r290_row = tl.arange($(16))
    r290_col = tl.arange($(32))
    r290_lane = tl.arange($(192))
    for r290_group in range($(8)) {
      r290_left = tl.load($(candidate_vto_copy_148_ptr) + (((((((r290_group % $(8)) * $(3072)) + (r290_row[:, None] * $(192)) + r290_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r290_group % $(8)) * $(3072)) + (r290_row[:, None] * $(192)) + r290_lane[None, :]) / $(192)) % $(16)) * $(192)) + ((((r290_group % $(8)) * $(3072)) + (r290_row[:, None] * $(192)) + r290_lane[None, :]) % $(192))), dtype=tl.float32)
      r290_right = tl.load($(candidate_vto_copy_149_ptr) + (((((((r290_group % $(8)) * $(6144)) + (r290_lane[:, None] * $(32)) + r290_col[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r290_group % $(8)) * $(6144)) + (r290_lane[:, None] * $(32)) + r290_col[None, :]) % $(32)) * $(192)) + (((((r290_group % $(8)) * $(6144)) + (r290_lane[:, None] * $(32)) + r290_col[None, :]) / $(32)) % $(192))), dtype=tl.float32)
      r290_value = tl.dot(r290_left, r290_right)
      r290_value = r290_value.to(tl.float32)
      tl.store($(candidate_vbmm_67_ptr) + ((r290_group * $(512)) + (r290_row[:, None] * $(32)) + r290_col[None, :]), r290_value)
    }
    r291_z = tl.arange($(4096))
    r291_value = (tl.load($(candidate_vbmm_67_ptr) + ((((r291_z / $(512)) % $(8)) * $(512)) + (((r291_z / $(32)) % $(16)) * $(32)) + (r291_z % $(32))), dtype=tl.float32) * 0.07216878364870322)
    r291_value = r291_value.to(tl.float32)
    r291_value = tl.broadcast_to(r291_value, [$(4096)])
    tl.store($(candidate_vmul_66_ptr) + r291_z, r291_value)
    r292_z = tl.arange($(16))
    r292_value = (tl.cast(r292_z, tl.int64) + tl.full([], $(32), dtype=tl.int64))
    r292_value = tl.broadcast_to(r292_value, [$(16)])
    tl.store($(candidate_vadd_44_ptr) + r292_z, r292_value)
    r293_z = tl.arange($(16))
    r293_value = (tl.load($(candidate_vadd_44_ptr) + r293_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r293_value = tl.broadcast_to(r293_value, [$(16)])
    tl.store($(candidate_vsub_42_ptr) + r293_z, r293_value)
    r294_z = tl.arange($(512))
    r294_value = (tl.cast((r294_z % $(32)), tl.int64) > tl.load($(candidate_vsub_42_ptr) + ((r294_z / $(32)) % $(16)), dtype=tl.int64))
    r294_value = tl.broadcast_to(r294_value, [$(512)])
    tl.store($(candidate_vgt_23_ptr) + r294_z, r294_value)
    r295_z = tl.arange($(4096))
    r295_value = tl.where(tl.load($(candidate_vgt_23_ptr) + ((((r295_z / $(32)) % $(16)) * $(32)) + (r295_z % $(32))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_66_ptr) + ((((r295_z / $(512)) % $(8)) * $(512)) + (((r295_z / $(32)) % $(16)) * $(32)) + (r295_z % $(32))), dtype=tl.float32))
    r295_value = r295_value.to(tl.float32)
    r295_value = tl.broadcast_to(r295_value, [$(4096)])
    tl.store($(candidate_vmasked_fill_23_ptr) + r295_z, r295_value)
    r296_z = tl.arange($(4096))
    r296_value = (tl.load($(candidate_vmasked_fill_23_ptr) + ((((r296_z / $(512)) % $(8)) * $(512)) + (((r296_z / $(32)) % $(16)) * $(32)) + (r296_z % $(32))), dtype=tl.float32) - tl.load($(candidate_vcat_28_ptr) + ((((r296_z / $(32)) % $(16)) * $(8)) + ((r296_z / $(512)) % $(8)) + $(128)), dtype=tl.float32))
    r296_value = r296_value.to(tl.float32)
    r296_value = tl.broadcast_to(r296_value, [$(4096)])
    tl.store($(candidate_vsub_43_ptr) + r296_z, r296_value)
    r297_z = tl.arange($(4096))
    r297_value = tl.exp(tl.load($(candidate_vsub_43_ptr) + ((((r297_z / $(512)) % $(8)) * $(512)) + (((r297_z / $(32)) % $(16)) * $(32)) + (r297_z % $(32))), dtype=tl.float32))
    r297_value = r297_value.to(tl.float32)
    r297_value = tl.broadcast_to(r297_value, [$(4096)])
    tl.store($(candidate_vexp_5_ptr) + r297_z, r297_value)
    r298_z = tl.arange($(16384))
    r298_value = (tl.load($(candidate_vto_copy_151_ptr) + ((((r298_z / $(2048)) % $(8)) * $(2048)) + (((r298_z / $(128)) % $(16)) * $(128)) + (r298_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_152_ptr) + ((((r298_z / $(2048)) % $(8)) * $(2048)) + (((r298_z / $(128)) % $(16)) * $(128)) + (r298_z % $(128))), dtype=tl.float32))
    r298_value = r298_value.to(tl.float32)
    r298_value = tl.broadcast_to(r298_value, [$(16384)])
    tl.store($(candidate_vmul_67_ptr) + r298_z, r298_value)
    r299_z = tl.arange($(128))
    r299_lane = tl.arange($(128))
    r299_value = tl.sum(tl.load($(candidate_vmul_67_ptr) + ((((r299_z[:, None] / $(16)) % $(8)) * $(2048)) + ((r299_z[:, None] % $(16)) * $(128)) + r299_lane[None, :]), dtype=tl.float32), axis=1)
    r299_value = r299_value.to(tl.float32)
    r299_value = tl.broadcast_to(r299_value, [$(128)])
    tl.store($(candidate_vsum_7_ptr) + r299_z, r299_value)
    r300_row = tl.arange($(16))
    r300_col = tl.arange($(32))
    r300_lane = tl.arange($(128))
    for r300_group in range($(8)) {
      r300_left = tl.load($(candidate_vto_copy_151_ptr) + (((((((r300_group % $(8)) * $(2048)) + (r300_row[:, None] * $(128)) + r300_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r300_group % $(8)) * $(2048)) + (r300_row[:, None] * $(128)) + r300_lane[None, :]) / $(128)) % $(16)) * $(128)) + ((((r300_group % $(8)) * $(2048)) + (r300_row[:, None] * $(128)) + r300_lane[None, :]) % $(128))), dtype=tl.float32)
      r300_right = tl.load($(candidate_vto_copy_150_ptr) + (((((((r300_group % $(8)) * $(4096)) + (r300_lane[:, None] * $(32)) + r300_col[None, :]) / $(4096)) % $(8)) * $(4096)) + (((((r300_group % $(8)) * $(4096)) + (r300_lane[:, None] * $(32)) + r300_col[None, :]) % $(32)) * $(128)) + (((((r300_group % $(8)) * $(4096)) + (r300_lane[:, None] * $(32)) + r300_col[None, :]) / $(32)) % $(128))), dtype=tl.float32)
      r300_value = tl.dot(r300_left, r300_right)
      r300_value = r300_value.to(tl.float32)
      tl.store($(candidate_vbmm_68_ptr) + ((r300_group * $(512)) + (r300_row[:, None] * $(32)) + r300_col[None, :]), r300_value)
    }
    r301_z = tl.arange($(4096))
    r301_value = (tl.load($(candidate_vbmm_68_ptr) + ((((r301_z / $(512)) % $(8)) * $(512)) + (((r301_z / $(32)) % $(16)) * $(32)) + (r301_z % $(32))), dtype=tl.float32) - tl.load($(candidate_vsum_7_ptr) + ((((r301_z / $(512)) % $(8)) * $(16)) + ((r301_z / $(32)) % $(16))), dtype=tl.float32))
    r301_value = r301_value.to(tl.float32)
    r301_value = tl.broadcast_to(r301_value, [$(4096)])
    tl.store($(candidate_vsub_44_ptr) + r301_z, r301_value)
    r302_z = tl.arange($(4096))
    r302_value = (tl.load($(candidate_vexp_5_ptr) + ((((r302_z / $(512)) % $(8)) * $(512)) + (((r302_z / $(32)) % $(16)) * $(32)) + (r302_z % $(32))), dtype=tl.float32) * tl.load($(candidate_vsub_44_ptr) + ((((r302_z / $(512)) % $(8)) * $(512)) + (((r302_z / $(32)) % $(16)) * $(32)) + (r302_z % $(32))), dtype=tl.float32))
    r302_value = r302_value.to(tl.float32)
    r302_value = tl.broadcast_to(r302_value, [$(4096)])
    tl.store($(candidate_vmul_68_ptr) + r302_z, r302_value)
    r303_row = tl.arange($(32))
    r303_col = tl.arange($(192))
    r303_lane = tl.arange($(16))
    for r303_group in range($(8)) {
      r303_left = tl.load($(candidate_vmul_68_ptr) + (((((((r303_group % $(8)) * $(512)) + (r303_row[:, None] * $(16)) + r303_lane[None, :]) / $(512)) % $(8)) * $(512)) + (((((r303_group % $(8)) * $(512)) + (r303_row[:, None] * $(16)) + r303_lane[None, :]) % $(16)) * $(32)) + (((((r303_group % $(8)) * $(512)) + (r303_row[:, None] * $(16)) + r303_lane[None, :]) / $(16)) % $(32))), dtype=tl.float32)
      r303_right = tl.load($(candidate_vto_copy_148_ptr) + (((((((r303_group % $(8)) * $(3072)) + (r303_lane[:, None] * $(192)) + r303_col[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r303_group % $(8)) * $(3072)) + (r303_lane[:, None] * $(192)) + r303_col[None, :]) / $(192)) % $(16)) * $(192)) + ((((r303_group % $(8)) * $(3072)) + (r303_lane[:, None] * $(192)) + r303_col[None, :]) % $(192))), dtype=tl.float32)
      r303_value = tl.dot(r303_left, r303_right)
      r303_value = r303_value.to(tl.float32)
      tl.store($(candidate_vbmm_70_ptr) + ((r303_group * $(6144)) + (r303_row[:, None] * $(192)) + r303_col[None, :]), r303_value)
    }
    r304_z = tl.arange($(49152))
    r304_value = (tl.load($(candidate_vbmm_70_ptr) + ((((r304_z / $(6144)) % $(8)) * $(6144)) + (((r304_z / $(192)) % $(32)) * $(192)) + (r304_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r304_value = r304_value.to(tl.float32)
    r304_value = tl.broadcast_to(r304_value, [$(49152)])
    tl.store($(candidate_vmul_70_ptr) + r304_z, r304_value)
    r305_z = tl.arange($(49152))
    r305_value = tl.load($(candidate_vmul_70_ptr) + ((((r305_z / $(192)) % $(8)) * $(6144)) + (((r305_z / $(1536)) % $(32)) * $(192)) + (r305_z % $(192))), dtype=tl.float32)
    r305_value = r305_value.to(tl.bfloat16)
    r305_value = tl.broadcast_to(r305_value, [$(49152)])
    tl.store($(candidate_vto_copy_154_ptr) + r305_z, r305_value)
    r306_row = tl.arange($(32))
    r306_col = tl.arange($(128))
    r306_lane = tl.arange($(16))
    for r306_group in range($(8)) {
      r306_left = tl.load($(candidate_vexp_5_ptr) + (((((((r306_group % $(8)) * $(512)) + (r306_row[:, None] * $(16)) + r306_lane[None, :]) / $(512)) % $(8)) * $(512)) + (((((r306_group % $(8)) * $(512)) + (r306_row[:, None] * $(16)) + r306_lane[None, :]) % $(16)) * $(32)) + (((((r306_group % $(8)) * $(512)) + (r306_row[:, None] * $(16)) + r306_lane[None, :]) / $(16)) % $(32))), dtype=tl.float32)
      r306_right = tl.load($(candidate_vto_copy_151_ptr) + (((((((r306_group % $(8)) * $(2048)) + (r306_lane[:, None] * $(128)) + r306_col[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r306_group % $(8)) * $(2048)) + (r306_lane[:, None] * $(128)) + r306_col[None, :]) / $(128)) % $(16)) * $(128)) + ((((r306_group % $(8)) * $(2048)) + (r306_lane[:, None] * $(128)) + r306_col[None, :]) % $(128))), dtype=tl.float32)
      r306_value = tl.dot(r306_left, r306_right)
      r306_value = r306_value.to(tl.float32)
      tl.store($(candidate_vbmm_71_ptr) + ((r306_group * $(4096)) + (r306_row[:, None] * $(128)) + r306_col[None, :]), r306_value)
    }
    r307_z = tl.arange($(32768))
    r307_value = tl.load($(candidate_vbmm_71_ptr) + ((((r307_z / $(128)) % $(8)) * $(4096)) + (((r307_z / $(1024)) % $(32)) * $(128)) + (r307_z % $(128))), dtype=tl.float32)
    r307_value = r307_value.to(tl.bfloat16)
    r307_value = tl.broadcast_to(r307_value, [$(32768)])
    tl.store($(candidate_vto_copy_155_ptr) + r307_z, r307_value)
    r308_z = tl.arange($(98304))
    r308_value = tl.where(((r308_z / $(1536)) % $(64)) < $(32), tl.load($(candidate_vto_copy_154_ptr) + ((((((r308_z / $(1536)) % $(64)) - $(0)) % $(32)) * $(1536)) + (((r308_z / $(192)) % $(8)) * $(192)) + (r308_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r308_value = r308_value.to(tl.bfloat16)
    r308_value = tl.broadcast_to(r308_value, [$(98304)])
    tl.store($(candidate_vcat_33_ptr) + r308_z, r308_value)
    r309_z = tl.arange($(65536))
    r309_value = tl.where(((r309_z / $(1024)) % $(64)) < $(32), tl.load($(candidate_vto_copy_155_ptr) + ((((((r309_z / $(1024)) % $(64)) - $(0)) % $(32)) * $(1024)) + (((r309_z / $(128)) % $(8)) * $(128)) + (r309_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r309_value = r309_value.to(tl.bfloat16)
    r309_value = tl.broadcast_to(r309_value, [$(65536)])
    tl.store($(candidate_vcat_34_ptr) + r309_z, r309_value)
    r310_z = tl.arange($(196608))
    r310_value = tl.where(((r310_z / $(1536)) % $(128)) < $(48), tl.load($(candidate_vcat_29_ptr) + ((((((r310_z / $(1536)) % $(128)) - $(0)) % $(48)) * $(1536)) + (((r310_z / $(192)) % $(8)) * $(192)) + (r310_z % $(192))), dtype=tl.bfloat16), tl.where(((r310_z / $(1536)) % $(128)) < $(64), tl.load($(candidate_vcat_31_ptr) + ((((((r310_z / $(1536)) % $(128)) - $(48)) % $(16)) * $(1536)) + (((r310_z / $(192)) % $(8)) * $(192)) + (r310_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vcat_33_ptr) + ((((((r310_z / $(1536)) % $(128)) - $(64)) % $(64)) * $(1536)) + (((r310_z / $(192)) % $(8)) * $(192)) + (r310_z % $(192))), dtype=tl.bfloat16)))
    r310_value = r310_value.to(tl.bfloat16)
    r310_value = tl.broadcast_to(r310_value, [$(196608)])
    tl.store($(candidate_vcat_36_ptr) + r310_z, r310_value)
    r311_z = tl.arange($(131072))
    r311_value = tl.where(((r311_z / $(1024)) % $(128)) < $(48), tl.load($(candidate_vcat_30_ptr) + ((((((r311_z / $(1024)) % $(128)) - $(0)) % $(48)) * $(1024)) + (((r311_z / $(128)) % $(8)) * $(128)) + (r311_z % $(128))), dtype=tl.bfloat16), tl.where(((r311_z / $(1024)) % $(128)) < $(64), tl.load($(candidate_vcat_32_ptr) + ((((((r311_z / $(1024)) % $(128)) - $(48)) % $(16)) * $(1024)) + (((r311_z / $(128)) % $(8)) * $(128)) + (r311_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_34_ptr) + ((((((r311_z / $(1024)) % $(128)) - $(64)) % $(64)) * $(1024)) + (((r311_z / $(128)) % $(8)) * $(128)) + (r311_z % $(128))), dtype=tl.bfloat16)))
    r311_value = r311_value.to(tl.bfloat16)
    r311_value = tl.broadcast_to(r311_value, [$(131072)])
    tl.store($(candidate_vcat_37_ptr) + r311_z, r311_value)
    r312_row = tl.arange($(1024))
    r312_col = tl.arange($(512))
    r312_lane = tl.arange($(128))
    for r312_group in range($(1)) {
      r312_left = tl.load($(candidate_vcat_36_ptr) + (((((r312_row[:, None] * $(128)) + r312_lane[None, :]) % $(128)) * $(1536)) + (((((r312_row[:, None] * $(128)) + r312_lane[None, :]) / $(16384)) % $(8)) * $(192)) + ((((r312_row[:, None] * $(128)) + r312_lane[None, :]) / $(128)) % $(128))), dtype=tl.bfloat16)
      r312_right = tl.load($(candidate_vindex_2_ptr) + ((((((r312_lane[:, None] * $(512)) + r312_col[None, :]) / $(512)) % $(128)) * $(576)) + (((r312_lane[:, None] * $(512)) + r312_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r312_value = tl.dot(r312_left, r312_right)
      r312_value = r312_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_74_ptr) + ((r312_group * $(524288)) + (r312_row[:, None] * $(512)) + r312_col[None, :]), r312_value)
    }
    r313_row = tl.arange($(1024))
    r313_col = tl.arange($(512))
    r313_lane = tl.arange($(128))
    for r313_group in range($(1)) {
      r313_left = tl.load($(candidate_vcat_37_ptr) + (((((r313_row[:, None] * $(128)) + r313_lane[None, :]) % $(128)) * $(1024)) + (((((r313_row[:, None] * $(128)) + r313_lane[None, :]) / $(16384)) % $(8)) * $(128)) + ((((r313_row[:, None] * $(128)) + r313_lane[None, :]) / $(128)) % $(128))), dtype=tl.bfloat16)
      r313_right = tl.load($(candidate_vindex_2_ptr) + ((((((r313_lane[:, None] * $(512)) + r313_col[None, :]) / $(512)) % $(128)) * $(576)) + (((r313_lane[:, None] * $(512)) + r313_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r313_value = tl.dot(r313_left, r313_right)
      r313_value = r313_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_75_ptr) + ((r313_group * $(524288)) + (r313_row[:, None] * $(512)) + r313_col[None, :]), r313_value)
    }
    r314_z = tl.arange($(1048576))
    r314_value = tl.where(((r314_z / $(512)) % $(256)) < $(128), tl.load($(candidate_vbmm_74_ptr) + ((((((((r314_z / $(131072)) % $(8)) * $(65536)) + ((((r314_z / $(512)) % $(256)) - $(0)) * $(512)) + (r314_z % $(512))) / $(65536)) % $(8)) * $(65536)) + ((((((r314_z / $(131072)) % $(8)) * $(128)) + (((r314_z / $(512)) % $(256)) - $(0))) % $(128)) * $(512)) + (r314_z % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vbmm_75_ptr) + ((((((((r314_z / $(131072)) % $(8)) * $(65536)) + ((((r314_z / $(512)) % $(256)) - $(128)) * $(512)) + (r314_z % $(512))) / $(65536)) % $(8)) * $(65536)) + ((((((r314_z / $(131072)) % $(8)) * $(128)) + (((r314_z / $(512)) % $(256)) - $(128))) % $(128)) * $(512)) + (r314_z % $(512))), dtype=tl.bfloat16))
    r314_value = r314_value.to(tl.bfloat16)
    r314_value = tl.broadcast_to(r314_value, [$(1048576)])
    tl.store($(candidate_vcat_38_ptr) + r314_z, r314_value)
    r315_z = tl.arange($(1048576))
    r315_value = (tl.load($(candidate_vcat_24_ptr) + ((((((((r315_z / $(512)) % $(2048)) * $(512)) + (r315_z % $(512))) / $(131072)) % $(8)) * $(131072)) + ((((r315_z / $(512)) % $(2048)) % $(256)) * $(512)) + (r315_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vcat_38_ptr) + ((((((((r315_z / $(512)) % $(2048)) * $(512)) + (r315_z % $(512))) / $(131072)) % $(8)) * $(131072)) + ((((r315_z / $(512)) % $(2048)) % $(256)) * $(512)) + (r315_z % $(512))), dtype=tl.bfloat16))
    r315_value = r315_value.to(tl.bfloat16)
    r315_value = tl.broadcast_to(r315_value, [$(1048576)])
    tl.store($(candidate_vadd_46_ptr) + r315_z, r315_value)
    r316_row = tl.arange($(128))
    r316_col = tl.arange($(1024))
    r316_lane = tl.arange($(512))
    for r316_group in range($(1)) {
      r316_left = tl.load($(candidate_vindex_2_ptr) + ((((((r316_row[:, None] * $(512)) + r316_lane[None, :]) / $(512)) % $(128)) * $(576)) + (((r316_row[:, None] * $(512)) + r316_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r316_right = tl.load($(candidate_vto_copy_20_ptr) + ((((((r316_lane[:, None] * $(1024)) + r316_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r316_lane[:, None] * $(1024)) + r316_col[None, :]) % $(128)) * $(512)) + ((((r316_lane[:, None] * $(1024)) + r316_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r316_value = tl.dot(r316_left, r316_right)
      r316_value = r316_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_76_ptr) + ((r316_group * $(131072)) + (r316_row[:, None] * $(1024)) + r316_col[None, :]), r316_value)
    }
    r317_row = tl.arange($(128))
    r317_col = tl.arange($(1024))
    r317_lane = tl.arange($(512))
    for r317_group in range($(1)) {
      r317_left = tl.load($(candidate_vindex_2_ptr) + ((((((r317_row[:, None] * $(512)) + r317_lane[None, :]) / $(512)) % $(128)) * $(576)) + (((r317_row[:, None] * $(512)) + r317_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r317_right = tl.load($(candidate_vto_copy_20_ptr) + ((((((r317_lane[:, None] * $(1024)) + r317_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r317_lane[:, None] * $(1024)) + r317_col[None, :]) % $(128)) * $(512)) + ((((r317_lane[:, None] * $(1024)) + r317_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r317_value = tl.dot(r317_left, r317_right)
      r317_value = r317_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_77_ptr) + ((r317_group * $(131072)) + (r317_row[:, None] * $(1024)) + r317_col[None, :]), r317_value)
    }
    r318_z = tl.arange($(49152))
    r318_value = tl.where((r318_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r318_z / $(1536)) % $(32)) * $(1024)) + (((r318_z / $(192)) % $(8)) * $(128)) + ((r318_z % $(192)) - $(0)) + $(32768)) / $(1024)) % $(128)) * $(1536)) + (((((((r318_z / $(1536)) % $(32)) * $(1024)) + (((r318_z / $(192)) % $(8)) * $(128)) + ((r318_z % $(192)) - $(0)) + $(32768)) / $(128)) % $(8)) * $(192)) + (((((r318_z / $(1536)) % $(32)) * $(1024)) + (((r318_z / $(192)) % $(8)) * $(128)) + ((r318_z % $(192)) - $(0)) + $(32768)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vadd_6_ptr) + ((((((((r318_z / $(1536)) % $(32)) * $(512)) + (((r318_z / $(192)) % $(8)) * $(64)) + ((r318_z % $(192)) - $(128)) + $(16384)) / $(512)) % $(128)) * $(512)) + (((((((r318_z / $(1536)) % $(32)) * $(512)) + (((r318_z / $(192)) % $(8)) * $(64)) + ((r318_z % $(192)) - $(128)) + $(16384)) / $(64)) % $(8)) * $(64)) + (((((r318_z / $(1536)) % $(32)) * $(512)) + (((r318_z / $(192)) % $(8)) * $(64)) + ((r318_z % $(192)) - $(128)) + $(16384)) % $(64))), dtype=tl.bfloat16))
    r318_value = r318_value.to(tl.bfloat16)
    r318_value = tl.broadcast_to(r318_value, [$(49152)])
    tl.store($(candidate_vcat_39_ptr) + r318_z, r318_value)
    r319_z = tl.arange($(196608))
    r319_value = tl.where((r319_z % $(192)) < $(128), tl.load($(candidate_vbmm_76_ptr) + ((((((((r319_z / $(1536)) % $(128)) * $(1024)) + (((r319_z / $(192)) % $(8)) * $(128)) + ((r319_z % $(192)) - $(0))) / $(1024)) % $(128)) * $(1024)) + (((((((r319_z / $(1536)) % $(128)) * $(1024)) + (((r319_z / $(192)) % $(8)) * $(128)) + ((r319_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r319_z / $(1536)) % $(128)) * $(1024)) + (((r319_z / $(192)) % $(8)) * $(128)) + ((r319_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vindex_2_ptr) + ((((r319_z / $(1536)) % $(128)) * $(576)) + ((((r319_z % $(192)) - $(128)) + $(512)) % $(576))), dtype=tl.bfloat16))
    r319_value = r319_value.to(tl.bfloat16)
    r319_value = tl.broadcast_to(r319_value, [$(196608)])
    tl.store($(candidate_vcat_40_ptr) + r319_z, r319_value)
    r320_z = tl.arange($(18432))
    r320_value = tl.load($(candidate_vcat_39_ptr) + ((((r320_z / $(192)) % $(12)) * $(1536)) + (((r320_z / $(2304)) % $(8)) * $(192)) + (r320_z % $(192))), dtype=tl.bfloat16)
    r320_value = r320_value.to(tl.float32)
    r320_value = tl.broadcast_to(r320_value, [$(18432)])
    tl.store($(candidate_vto_copy_156_ptr) + r320_z, r320_value)
    r321_z = tl.arange($(73728))
    r321_value = tl.load($(candidate_vcat_40_ptr) + ((((r321_z / $(192)) % $(48)) * $(1536)) + (((r321_z / $(9216)) % $(8)) * $(192)) + (r321_z % $(192))), dtype=tl.bfloat16)
    r321_value = r321_value.to(tl.float32)
    r321_value = tl.broadcast_to(r321_value, [$(73728)])
    tl.store($(candidate_vto_copy_157_ptr) + r321_z, r321_value)
    r322_row = tl.arange($(12))
    r322_col = tl.arange($(48))
    r322_lane = tl.arange($(192))
    for r322_group in range($(8)) {
      r322_left = tl.load($(candidate_vto_copy_156_ptr) + (((((((r322_group % $(8)) * $(2304)) + (r322_row[:, None] * $(192)) + r322_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r322_group % $(8)) * $(2304)) + (r322_row[:, None] * $(192)) + r322_lane[None, :]) / $(192)) % $(12)) * $(192)) + ((((r322_group % $(8)) * $(2304)) + (r322_row[:, None] * $(192)) + r322_lane[None, :]) % $(192))), dtype=tl.float32)
      r322_right = tl.load($(candidate_vto_copy_157_ptr) + (((((((r322_group % $(8)) * $(9216)) + (r322_lane[:, None] * $(48)) + r322_col[None, :]) / $(9216)) % $(8)) * $(9216)) + (((((r322_group % $(8)) * $(9216)) + (r322_lane[:, None] * $(48)) + r322_col[None, :]) % $(48)) * $(192)) + (((((r322_group % $(8)) * $(9216)) + (r322_lane[:, None] * $(48)) + r322_col[None, :]) / $(48)) % $(192))), dtype=tl.float32)
      r322_value = tl.dot(r322_left, r322_right)
      r322_value = r322_value.to(tl.float32)
      tl.store($(candidate_vbmm_78_ptr) + ((r322_group * $(576)) + (r322_row[:, None] * $(48)) + r322_col[None, :]), r322_value)
    }
    r323_z = tl.arange($(4608))
    r323_value = (tl.load($(candidate_vbmm_78_ptr) + ((((r323_z / $(576)) % $(8)) * $(576)) + (((r323_z / $(48)) % $(12)) * $(48)) + (r323_z % $(48))), dtype=tl.float32) * 0.07216878364870322)
    r323_value = r323_value.to(tl.float32)
    r323_value = tl.broadcast_to(r323_value, [$(4608)])
    tl.store($(candidate_vmul_71_ptr) + r323_z, r323_value)
    r324_z = tl.arange($(12))
    r324_value = (tl.cast(r324_z, tl.int64) + tl.full([], $(48), dtype=tl.int64))
    r324_value = tl.broadcast_to(r324_value, [$(12)])
    tl.store($(candidate_vadd_47_ptr) + r324_z, r324_value)
    r325_z = tl.arange($(12))
    r325_value = (tl.load($(candidate_vadd_47_ptr) + r325_z, dtype=tl.int64) - tl.full([], $(12), dtype=tl.int64))
    r325_value = tl.broadcast_to(r325_value, [$(12)])
    tl.store($(candidate_vsub_45_ptr) + r325_z, r325_value)
    r326_z = tl.arange($(576))
    r326_value = (tl.cast((r326_z % $(48)), tl.int64) > tl.load($(candidate_vsub_45_ptr) + ((r326_z / $(48)) % $(12)), dtype=tl.int64))
    r326_value = tl.broadcast_to(r326_value, [$(576)])
    tl.store($(candidate_vgt_24_ptr) + r326_z, r326_value)
    r327_z = tl.arange($(4608))
    r327_value = tl.where(tl.load($(candidate_vgt_24_ptr) + ((((r327_z / $(48)) % $(12)) * $(48)) + (r327_z % $(48))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_71_ptr) + ((((r327_z / $(576)) % $(8)) * $(576)) + (((r327_z / $(48)) % $(12)) * $(48)) + (r327_z % $(48))), dtype=tl.float32))
    r327_value = r327_value.to(tl.float32)
    r327_value = tl.broadcast_to(r327_value, [$(4608)])
    tl.store($(candidate_vmasked_fill_24_ptr) + r327_z, r327_value)
    r328_z = tl.arange($(4608))
    r328_lane = tl.arange($(48))
    r328_row_values = tl.load($(candidate_vmasked_fill_24_ptr) + ((((r328_z[:, None] / $(576)) % $(8)) * $(576)) + (((r328_z[:, None] / $(48)) % $(12)) * $(48)) + (r328_lane[None, :] % $(48))), dtype=tl.float32)
    r328_maximum = tl.max(r328_row_values, axis=1, keep_dims=true)
    r328_weights = tl.exp(r328_row_values - r328_maximum)
    r328_total = tl.sum(r328_weights, axis=1)
    r328_maximum = tl.max(r328_row_values, axis=1)
    r328_value = tl.exp(tl.load($(candidate_vmasked_fill_24_ptr) + ((((r328_z / $(576)) % $(8)) * $(576)) + (((r328_z / $(48)) % $(12)) * $(48)) + (r328_z % $(48))), dtype=tl.float32) - r328_maximum) / r328_total
    r328_value = r328_value.to(tl.float32)
    r328_value = tl.broadcast_to(r328_value, [$(4608)])
    tl.store($(candidate_vsoftmax_18_ptr) + r328_z, r328_value)
    r329_z = tl.arange($(4608))
    r329_value = tl.load($(candidate_vsoftmax_18_ptr) + ((((r329_z / $(576)) % $(8)) * $(576)) + (((r329_z / $(48)) % $(12)) * $(48)) + (r329_z % $(48))), dtype=tl.float32)
    r329_value = r329_value.to(tl.bfloat16)
    r329_value = tl.broadcast_to(r329_value, [$(4608)])
    tl.store($(candidate_vto_copy_158_ptr) + r329_z, r329_value)
    r330_row = tl.arange($(12))
    r330_col = tl.arange($(128))
    r330_lane = tl.arange($(48))
    for r330_group in range($(8)) {
      r330_left = tl.load($(candidate_vto_copy_158_ptr) + (((((((r330_group % $(8)) * $(576)) + (r330_row[:, None] * $(48)) + r330_lane[None, :]) / $(576)) % $(8)) * $(576)) + ((((((r330_group % $(8)) * $(576)) + (r330_row[:, None] * $(48)) + r330_lane[None, :]) / $(48)) % $(12)) * $(48)) + ((((r330_group % $(8)) * $(576)) + (r330_row[:, None] * $(48)) + r330_lane[None, :]) % $(48))), dtype=tl.bfloat16)
      r330_right = tl.load($(candidate_vbmm_77_ptr) + (((((((r330_group % $(8)) * $(6144)) + (r330_lane[:, None] * $(128)) + r330_col[None, :]) / $(128)) % $(48)) * $(1024)) + ((((((r330_group % $(8)) * $(6144)) + (r330_lane[:, None] * $(128)) + r330_col[None, :]) / $(6144)) % $(8)) * $(128)) + ((((r330_group % $(8)) * $(6144)) + (r330_lane[:, None] * $(128)) + r330_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r330_value = tl.dot(r330_left, r330_right)
      r330_value = r330_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_79_ptr) + ((r330_group * $(1536)) + (r330_row[:, None] * $(128)) + r330_col[None, :]), r330_value)
    }
    r331_z = tl.arange($(96))
    r331_lane = tl.arange($(48))
    r331_row_values = tl.load($(candidate_vmasked_fill_24_ptr) + ((((r331_z[:, None] / $(12)) % $(8)) * $(576)) + ((r331_z[:, None] % $(12)) * $(48)) + (r331_lane[None, :] % $(48))), dtype=tl.float32)
    r331_maximum = tl.max(r331_row_values, axis=1, keep_dims=true)
    r331_weights = tl.exp(r331_row_values - r331_maximum)
    r331_total = tl.sum(r331_weights, axis=1)
    r331_maximum = tl.max(r331_row_values, axis=1)
    r331_value = r331_maximum + tl.log(r331_total)
    r331_value = r331_value.to(tl.float32)
    r331_value = tl.broadcast_to(r331_value, [$(96)])
    tl.store($(candidate_vlogsumexp_18_ptr) + r331_z, r331_value)
    r332_z = tl.arange($(6144))
    r332_value = tl.load($(candidate_vcat_39_ptr) + ((((r332_z / $(192)) % $(4)) * $(1536)) + (((r332_z / $(768)) % $(8)) * $(192)) + (r332_z % $(192)) + $(18432)), dtype=tl.bfloat16)
    r332_value = r332_value.to(tl.float32)
    r332_value = tl.broadcast_to(r332_value, [$(6144)])
    tl.store($(candidate_vto_copy_159_ptr) + r332_z, r332_value)
    r333_z = tl.arange($(24576))
    r333_value = tl.load($(candidate_vcat_40_ptr) + ((((r333_z / $(192)) % $(16)) * $(1536)) + (((r333_z / $(3072)) % $(8)) * $(192)) + (r333_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r333_value = r333_value.to(tl.float32)
    r333_value = tl.broadcast_to(r333_value, [$(24576)])
    tl.store($(candidate_vto_copy_160_ptr) + r333_z, r333_value)
    r334_row = tl.arange($(4))
    r334_col = tl.arange($(16))
    r334_lane = tl.arange($(192))
    for r334_group in range($(8)) {
      r334_left = tl.load($(candidate_vto_copy_159_ptr) + (((((((r334_group % $(8)) * $(768)) + (r334_row[:, None] * $(192)) + r334_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r334_group % $(8)) * $(768)) + (r334_row[:, None] * $(192)) + r334_lane[None, :]) / $(192)) % $(4)) * $(192)) + ((((r334_group % $(8)) * $(768)) + (r334_row[:, None] * $(192)) + r334_lane[None, :]) % $(192))), dtype=tl.float32)
      r334_right = tl.load($(candidate_vto_copy_160_ptr) + (((((((r334_group % $(8)) * $(3072)) + (r334_lane[:, None] * $(16)) + r334_col[None, :]) / $(3072)) % $(8)) * $(3072)) + (((((r334_group % $(8)) * $(3072)) + (r334_lane[:, None] * $(16)) + r334_col[None, :]) % $(16)) * $(192)) + (((((r334_group % $(8)) * $(3072)) + (r334_lane[:, None] * $(16)) + r334_col[None, :]) / $(16)) % $(192))), dtype=tl.float32)
      r334_value = tl.dot(r334_left, r334_right)
      r334_value = r334_value.to(tl.float32)
      tl.store($(candidate_vbmm_80_ptr) + ((r334_group * $(64)) + (r334_row[:, None] * $(16)) + r334_col[None, :]), r334_value)
    }
    r335_z = tl.arange($(512))
    r335_value = (tl.load($(candidate_vbmm_80_ptr) + ((((r335_z / $(64)) % $(8)) * $(64)) + (((r335_z / $(16)) % $(4)) * $(16)) + (r335_z % $(16))), dtype=tl.float32) * 0.07216878364870322)
    r335_value = r335_value.to(tl.float32)
    r335_value = tl.broadcast_to(r335_value, [$(512)])
    tl.store($(candidate_vmul_72_ptr) + r335_z, r335_value)
    r336_z = tl.arange($(4))
    r336_value = (tl.cast(r336_z, tl.int64) + tl.full([], $(16), dtype=tl.int64))
    r336_value = tl.broadcast_to(r336_value, [$(4)])
    tl.store($(candidate_vadd_48_ptr) + r336_z, r336_value)
    r337_z = tl.arange($(4))
    r337_value = (tl.load($(candidate_vadd_48_ptr) + r337_z, dtype=tl.int64) - tl.full([], $(4), dtype=tl.int64))
    r337_value = tl.broadcast_to(r337_value, [$(4)])
    tl.store($(candidate_vsub_46_ptr) + r337_z, r337_value)
    r338_z = tl.arange($(64))
    r338_value = (tl.cast((r338_z % $(16)), tl.int64) > tl.load($(candidate_vsub_46_ptr) + ((r338_z / $(16)) % $(4)), dtype=tl.int64))
    r338_value = tl.broadcast_to(r338_value, [$(64)])
    tl.store($(candidate_vgt_25_ptr) + r338_z, r338_value)
    r339_z = tl.arange($(512))
    r339_value = tl.where(tl.load($(candidate_vgt_25_ptr) + ((((r339_z / $(16)) % $(4)) * $(16)) + (r339_z % $(16))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_72_ptr) + ((((r339_z / $(64)) % $(8)) * $(64)) + (((r339_z / $(16)) % $(4)) * $(16)) + (r339_z % $(16))), dtype=tl.float32))
    r339_value = r339_value.to(tl.float32)
    r339_value = tl.broadcast_to(r339_value, [$(512)])
    tl.store($(candidate_vmasked_fill_25_ptr) + r339_z, r339_value)
    r340_z = tl.arange($(512))
    r340_lane = tl.arange($(16))
    r340_row_values = tl.load($(candidate_vmasked_fill_25_ptr) + ((((r340_z[:, None] / $(64)) % $(8)) * $(64)) + (((r340_z[:, None] / $(16)) % $(4)) * $(16)) + (r340_lane[None, :] % $(16))), dtype=tl.float32)
    r340_maximum = tl.max(r340_row_values, axis=1, keep_dims=true)
    r340_weights = tl.exp(r340_row_values - r340_maximum)
    r340_total = tl.sum(r340_weights, axis=1)
    r340_maximum = tl.max(r340_row_values, axis=1)
    r340_value = tl.exp(tl.load($(candidate_vmasked_fill_25_ptr) + ((((r340_z / $(64)) % $(8)) * $(64)) + (((r340_z / $(16)) % $(4)) * $(16)) + (r340_z % $(16))), dtype=tl.float32) - r340_maximum) / r340_total
    r340_value = r340_value.to(tl.float32)
    r340_value = tl.broadcast_to(r340_value, [$(512)])
    tl.store($(candidate_vsoftmax_19_ptr) + r340_z, r340_value)
    r341_z = tl.arange($(512))
    r341_value = tl.load($(candidate_vsoftmax_19_ptr) + ((((r341_z / $(64)) % $(8)) * $(64)) + (((r341_z / $(16)) % $(4)) * $(16)) + (r341_z % $(16))), dtype=tl.float32)
    r341_value = r341_value.to(tl.bfloat16)
    r341_value = tl.broadcast_to(r341_value, [$(512)])
    tl.store($(candidate_vto_copy_161_ptr) + r341_z, r341_value)
    r342_row = tl.arange($(4))
    r342_col = tl.arange($(128))
    r342_lane = tl.arange($(16))
    for r342_group in range($(8)) {
      r342_left = tl.load($(candidate_vto_copy_161_ptr) + (((((((r342_group % $(8)) * $(64)) + (r342_row[:, None] * $(16)) + r342_lane[None, :]) / $(64)) % $(8)) * $(64)) + ((((((r342_group % $(8)) * $(64)) + (r342_row[:, None] * $(16)) + r342_lane[None, :]) / $(16)) % $(4)) * $(16)) + ((((r342_group % $(8)) * $(64)) + (r342_row[:, None] * $(16)) + r342_lane[None, :]) % $(16))), dtype=tl.bfloat16)
      r342_right = tl.load($(candidate_vbmm_77_ptr) + (((((((r342_group % $(8)) * $(2048)) + (r342_lane[:, None] * $(128)) + r342_col[None, :]) / $(128)) % $(16)) * $(1024)) + ((((((r342_group % $(8)) * $(2048)) + (r342_lane[:, None] * $(128)) + r342_col[None, :]) / $(2048)) % $(8)) * $(128)) + ((((r342_group % $(8)) * $(2048)) + (r342_lane[:, None] * $(128)) + r342_col[None, :]) % $(128)) + $(49152)), dtype=tl.bfloat16)
      r342_value = tl.dot(r342_left, r342_right)
      r342_value = r342_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_81_ptr) + ((r342_group * $(512)) + (r342_row[:, None] * $(128)) + r342_col[None, :]), r342_value)
    }
    r343_z = tl.arange($(32))
    r343_lane = tl.arange($(16))
    r343_row_values = tl.load($(candidate_vmasked_fill_25_ptr) + ((((r343_z[:, None] / $(4)) % $(8)) * $(64)) + ((r343_z[:, None] % $(4)) * $(16)) + (r343_lane[None, :] % $(16))), dtype=tl.float32)
    r343_maximum = tl.max(r343_row_values, axis=1, keep_dims=true)
    r343_weights = tl.exp(r343_row_values - r343_maximum)
    r343_total = tl.sum(r343_weights, axis=1)
    r343_maximum = tl.max(r343_row_values, axis=1)
    r343_value = r343_maximum + tl.log(r343_total)
    r343_value = r343_value.to(tl.float32)
    r343_value = tl.broadcast_to(r343_value, [$(32)])
    tl.store($(candidate_vlogsumexp_19_ptr) + r343_z, r343_value)
    r344_z = tl.arange($(24576))
    r344_value = tl.load($(candidate_vcat_39_ptr) + ((((r344_z / $(192)) % $(16)) * $(1536)) + (((r344_z / $(3072)) % $(8)) * $(192)) + (r344_z % $(192)) + $(24576)), dtype=tl.bfloat16)
    r344_value = r344_value.to(tl.float32)
    r344_value = tl.broadcast_to(r344_value, [$(24576)])
    tl.store($(candidate_vto_copy_162_ptr) + r344_z, r344_value)
    r345_z = tl.arange($(98304))
    r345_value = tl.load($(candidate_vcat_40_ptr) + ((((r345_z / $(192)) % $(64)) * $(1536)) + (((r345_z / $(12288)) % $(8)) * $(192)) + (r345_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r345_value = r345_value.to(tl.float32)
    r345_value = tl.broadcast_to(r345_value, [$(98304)])
    tl.store($(candidate_vto_copy_163_ptr) + r345_z, r345_value)
    r346_row = tl.arange($(16))
    r346_col = tl.arange($(64))
    r346_lane = tl.arange($(192))
    for r346_group in range($(8)) {
      r346_left = tl.load($(candidate_vto_copy_162_ptr) + (((((((r346_group % $(8)) * $(3072)) + (r346_row[:, None] * $(192)) + r346_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r346_group % $(8)) * $(3072)) + (r346_row[:, None] * $(192)) + r346_lane[None, :]) / $(192)) % $(16)) * $(192)) + ((((r346_group % $(8)) * $(3072)) + (r346_row[:, None] * $(192)) + r346_lane[None, :]) % $(192))), dtype=tl.float32)
      r346_right = tl.load($(candidate_vto_copy_163_ptr) + (((((((r346_group % $(8)) * $(12288)) + (r346_lane[:, None] * $(64)) + r346_col[None, :]) / $(12288)) % $(8)) * $(12288)) + (((((r346_group % $(8)) * $(12288)) + (r346_lane[:, None] * $(64)) + r346_col[None, :]) % $(64)) * $(192)) + (((((r346_group % $(8)) * $(12288)) + (r346_lane[:, None] * $(64)) + r346_col[None, :]) / $(64)) % $(192))), dtype=tl.float32)
      r346_value = tl.dot(r346_left, r346_right)
      r346_value = r346_value.to(tl.float32)
      tl.store($(candidate_vbmm_82_ptr) + ((r346_group * $(1024)) + (r346_row[:, None] * $(64)) + r346_col[None, :]), r346_value)
    }
    r347_z = tl.arange($(8192))
    r347_value = (tl.load($(candidate_vbmm_82_ptr) + ((((r347_z / $(1024)) % $(8)) * $(1024)) + (((r347_z / $(64)) % $(16)) * $(64)) + (r347_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r347_value = r347_value.to(tl.float32)
    r347_value = tl.broadcast_to(r347_value, [$(8192)])
    tl.store($(candidate_vmul_73_ptr) + r347_z, r347_value)
    r348_z = tl.arange($(16))
    r348_value = (tl.cast(r348_z, tl.int64) + tl.full([], $(64), dtype=tl.int64))
    r348_value = tl.broadcast_to(r348_value, [$(16)])
    tl.store($(candidate_vadd_49_ptr) + r348_z, r348_value)
    r349_z = tl.arange($(16))
    r349_value = (tl.load($(candidate_vadd_49_ptr) + r349_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r349_value = tl.broadcast_to(r349_value, [$(16)])
    tl.store($(candidate_vsub_47_ptr) + r349_z, r349_value)
    r350_z = tl.arange($(1024))
    r350_value = (tl.cast((r350_z % $(64)), tl.int64) > tl.load($(candidate_vsub_47_ptr) + ((r350_z / $(64)) % $(16)), dtype=tl.int64))
    r350_value = tl.broadcast_to(r350_value, [$(1024)])
    tl.store($(candidate_vgt_26_ptr) + r350_z, r350_value)
    r351_z = tl.arange($(8192))
    r351_value = tl.where(tl.load($(candidate_vgt_26_ptr) + ((((r351_z / $(64)) % $(16)) * $(64)) + (r351_z % $(64))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_73_ptr) + ((((r351_z / $(1024)) % $(8)) * $(1024)) + (((r351_z / $(64)) % $(16)) * $(64)) + (r351_z % $(64))), dtype=tl.float32))
    r351_value = r351_value.to(tl.float32)
    r351_value = tl.broadcast_to(r351_value, [$(8192)])
    tl.store($(candidate_vmasked_fill_26_ptr) + r351_z, r351_value)
    r352_z = tl.arange($(8192))
    r352_lane = tl.arange($(64))
    r352_row_values = tl.load($(candidate_vmasked_fill_26_ptr) + ((((r352_z[:, None] / $(1024)) % $(8)) * $(1024)) + (((r352_z[:, None] / $(64)) % $(16)) * $(64)) + (r352_lane[None, :] % $(64))), dtype=tl.float32)
    r352_maximum = tl.max(r352_row_values, axis=1, keep_dims=true)
    r352_weights = tl.exp(r352_row_values - r352_maximum)
    r352_total = tl.sum(r352_weights, axis=1)
    r352_maximum = tl.max(r352_row_values, axis=1)
    r352_value = tl.exp(tl.load($(candidate_vmasked_fill_26_ptr) + ((((r352_z / $(1024)) % $(8)) * $(1024)) + (((r352_z / $(64)) % $(16)) * $(64)) + (r352_z % $(64))), dtype=tl.float32) - r352_maximum) / r352_total
    r352_value = r352_value.to(tl.float32)
    r352_value = tl.broadcast_to(r352_value, [$(8192)])
    tl.store($(candidate_vsoftmax_20_ptr) + r352_z, r352_value)
    r353_z = tl.arange($(8192))
    r353_value = tl.load($(candidate_vsoftmax_20_ptr) + ((((r353_z / $(1024)) % $(8)) * $(1024)) + (((r353_z / $(64)) % $(16)) * $(64)) + (r353_z % $(64))), dtype=tl.float32)
    r353_value = r353_value.to(tl.bfloat16)
    r353_value = tl.broadcast_to(r353_value, [$(8192)])
    tl.store($(candidate_vto_copy_164_ptr) + r353_z, r353_value)
    r354_row = tl.arange($(16))
    r354_col = tl.arange($(128))
    r354_lane = tl.arange($(64))
    for r354_group in range($(8)) {
      r354_left = tl.load($(candidate_vto_copy_164_ptr) + (((((((r354_group % $(8)) * $(1024)) + (r354_row[:, None] * $(64)) + r354_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + ((((((r354_group % $(8)) * $(1024)) + (r354_row[:, None] * $(64)) + r354_lane[None, :]) / $(64)) % $(16)) * $(64)) + ((((r354_group % $(8)) * $(1024)) + (r354_row[:, None] * $(64)) + r354_lane[None, :]) % $(64))), dtype=tl.bfloat16)
      r354_right = tl.load($(candidate_vbmm_77_ptr) + (((((((r354_group % $(8)) * $(8192)) + (r354_lane[:, None] * $(128)) + r354_col[None, :]) / $(128)) % $(64)) * $(1024)) + ((((((r354_group % $(8)) * $(8192)) + (r354_lane[:, None] * $(128)) + r354_col[None, :]) / $(8192)) % $(8)) * $(128)) + ((((r354_group % $(8)) * $(8192)) + (r354_lane[:, None] * $(128)) + r354_col[None, :]) % $(128)) + $(65536)), dtype=tl.bfloat16)
      r354_value = tl.dot(r354_left, r354_right)
      r354_value = r354_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_83_ptr) + ((r354_group * $(2048)) + (r354_row[:, None] * $(128)) + r354_col[None, :]), r354_value)
    }
    r355_z = tl.arange($(128))
    r355_lane = tl.arange($(64))
    r355_row_values = tl.load($(candidate_vmasked_fill_26_ptr) + ((((r355_z[:, None] / $(16)) % $(8)) * $(1024)) + ((r355_z[:, None] % $(16)) * $(64)) + (r355_lane[None, :] % $(64))), dtype=tl.float32)
    r355_maximum = tl.max(r355_row_values, axis=1, keep_dims=true)
    r355_weights = tl.exp(r355_row_values - r355_maximum)
    r355_total = tl.sum(r355_weights, axis=1)
    r355_maximum = tl.max(r355_row_values, axis=1)
    r355_value = r355_maximum + tl.log(r355_total)
    r355_value = r355_value.to(tl.float32)
    r355_value = tl.broadcast_to(r355_value, [$(128)])
    tl.store($(candidate_vlogsumexp_20_ptr) + r355_z, r355_value)
    r356_z = tl.arange($(32768))
    r356_value = tl.where(((r356_z / $(1024)) % $(32)) < $(12), tl.load($(candidate_vbmm_79_ptr) + ((((((((r356_z / $(128)) % $(8)) * $(1536)) + ((((r356_z / $(1024)) % $(32)) - $(0)) * $(128)) + (r356_z % $(128))) / $(1536)) % $(8)) * $(1536)) + ((((((r356_z / $(128)) % $(8)) * $(12)) + (((r356_z / $(1024)) % $(32)) - $(0))) % $(12)) * $(128)) + (r356_z % $(128))), dtype=tl.bfloat16), tl.where(((r356_z / $(1024)) % $(32)) < $(16), tl.load($(candidate_vbmm_81_ptr) + ((((((((r356_z / $(128)) % $(8)) * $(512)) + ((((r356_z / $(1024)) % $(32)) - $(12)) * $(128)) + (r356_z % $(128))) / $(512)) % $(8)) * $(512)) + ((((((r356_z / $(128)) % $(8)) * $(4)) + (((r356_z / $(1024)) % $(32)) - $(12))) % $(4)) * $(128)) + (r356_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vbmm_83_ptr) + ((((((((r356_z / $(128)) % $(8)) * $(2048)) + ((((r356_z / $(1024)) % $(32)) - $(16)) * $(128)) + (r356_z % $(128))) / $(2048)) % $(8)) * $(2048)) + ((((((r356_z / $(128)) % $(8)) * $(16)) + (((r356_z / $(1024)) % $(32)) - $(16))) % $(16)) * $(128)) + (r356_z % $(128))), dtype=tl.bfloat16)))
    r356_value = r356_value.to(tl.bfloat16)
    r356_value = tl.broadcast_to(r356_value, [$(32768)])
    tl.store($(candidate_vcat_41_ptr) + r356_z, r356_value)
    r357_z = tl.arange($(256))
    r357_value = tl.where(((r357_z / $(8)) % $(32)) < $(12), tl.load($(candidate_vlogsumexp_18_ptr) + (((r357_z % $(8)) * $(12)) + ((((r357_z / $(8)) % $(32)) - $(0)) % $(12))), dtype=tl.float32), tl.where(((r357_z / $(8)) % $(32)) < $(16), tl.load($(candidate_vlogsumexp_19_ptr) + (((r357_z % $(8)) * $(4)) + ((((r357_z / $(8)) % $(32)) - $(12)) % $(4))), dtype=tl.float32), tl.load($(candidate_vlogsumexp_20_ptr) + (((r357_z % $(8)) * $(16)) + ((((r357_z / $(8)) % $(32)) - $(16)) % $(16))), dtype=tl.float32)))
    r357_value = r357_value.to(tl.float32)
    r357_value = tl.broadcast_to(r357_value, [$(256)])
    tl.store($(candidate_vcat_42_ptr) + r357_z, r357_value)
    r358_z = tl.arange($(18432))
    r358_value = tl.load($(candidate_vcat_39_ptr) + ((((r358_z / $(192)) % $(12)) * $(1536)) + (((r358_z / $(2304)) % $(8)) * $(192)) + (r358_z % $(192))), dtype=tl.bfloat16)
    r358_value = r358_value.to(tl.float32)
    r358_value = tl.broadcast_to(r358_value, [$(18432)])
    tl.store($(candidate_vto_copy_165_ptr) + r358_z, r358_value)
    r359_z = tl.arange($(73728))
    r359_value = tl.load($(candidate_vcat_40_ptr) + ((((r359_z / $(192)) % $(48)) * $(1536)) + (((r359_z / $(9216)) % $(8)) * $(192)) + (r359_z % $(192))), dtype=tl.bfloat16)
    r359_value = r359_value.to(tl.float32)
    r359_value = tl.broadcast_to(r359_value, [$(73728)])
    tl.store($(candidate_vto_copy_166_ptr) + r359_z, r359_value)
    r360_z = tl.arange($(49152))
    r360_value = tl.load($(candidate_vbmm_77_ptr) + ((((r360_z / $(128)) % $(48)) * $(1024)) + (((r360_z / $(6144)) % $(8)) * $(128)) + (r360_z % $(128))), dtype=tl.bfloat16)
    r360_value = r360_value.to(tl.float32)
    r360_value = tl.broadcast_to(r360_value, [$(49152)])
    tl.store($(candidate_vto_copy_167_ptr) + r360_z, r360_value)
    r361_z = tl.arange($(12288))
    r361_value = tl.load($(candidate_vmm_28_ptr) + ((((r361_z / $(128)) % $(12)) * $(1024)) + (((r361_z / $(1536)) % $(8)) * $(128)) + (r361_z % $(128)) + $(32768)), dtype=tl.bfloat16)
    r361_value = r361_value.to(tl.float32)
    r361_value = tl.broadcast_to(r361_value, [$(12288)])
    tl.store($(candidate_vto_copy_168_ptr) + r361_z, r361_value)
    r362_z = tl.arange($(12288))
    r362_value = tl.load($(candidate_vcat_41_ptr) + ((((r362_z / $(128)) % $(12)) * $(1024)) + (((r362_z / $(1536)) % $(8)) * $(128)) + (r362_z % $(128))), dtype=tl.bfloat16)
    r362_value = r362_value.to(tl.float32)
    r362_value = tl.broadcast_to(r362_value, [$(12288)])
    tl.store($(candidate_vto_copy_169_ptr) + r362_z, r362_value)
    r363_row = tl.arange($(12))
    r363_col = tl.arange($(48))
    r363_lane = tl.arange($(192))
    for r363_group in range($(8)) {
      r363_left = tl.load($(candidate_vto_copy_165_ptr) + (((((((r363_group % $(8)) * $(2304)) + (r363_row[:, None] * $(192)) + r363_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r363_group % $(8)) * $(2304)) + (r363_row[:, None] * $(192)) + r363_lane[None, :]) / $(192)) % $(12)) * $(192)) + ((((r363_group % $(8)) * $(2304)) + (r363_row[:, None] * $(192)) + r363_lane[None, :]) % $(192))), dtype=tl.float32)
      r363_right = tl.load($(candidate_vto_copy_166_ptr) + (((((((r363_group % $(8)) * $(9216)) + (r363_lane[:, None] * $(48)) + r363_col[None, :]) / $(9216)) % $(8)) * $(9216)) + (((((r363_group % $(8)) * $(9216)) + (r363_lane[:, None] * $(48)) + r363_col[None, :]) % $(48)) * $(192)) + (((((r363_group % $(8)) * $(9216)) + (r363_lane[:, None] * $(48)) + r363_col[None, :]) / $(48)) % $(192))), dtype=tl.float32)
      r363_value = tl.dot(r363_left, r363_right)
      r363_value = r363_value.to(tl.float32)
      tl.store($(candidate_vbmm_84_ptr) + ((r363_group * $(576)) + (r363_row[:, None] * $(48)) + r363_col[None, :]), r363_value)
    }
    r364_z = tl.arange($(4608))
    r364_value = (tl.load($(candidate_vbmm_84_ptr) + ((((r364_z / $(576)) % $(8)) * $(576)) + (((r364_z / $(48)) % $(12)) * $(48)) + (r364_z % $(48))), dtype=tl.float32) * 0.07216878364870322)
    r364_value = r364_value.to(tl.float32)
    r364_value = tl.broadcast_to(r364_value, [$(4608)])
    tl.store($(candidate_vmul_74_ptr) + r364_z, r364_value)
    r365_z = tl.arange($(12))
    r365_value = (tl.cast(r365_z, tl.int64) + tl.full([], $(48), dtype=tl.int64))
    r365_value = tl.broadcast_to(r365_value, [$(12)])
    tl.store($(candidate_vadd_50_ptr) + r365_z, r365_value)
    r366_z = tl.arange($(12))
    r366_value = (tl.load($(candidate_vadd_50_ptr) + r366_z, dtype=tl.int64) - tl.full([], $(12), dtype=tl.int64))
    r366_value = tl.broadcast_to(r366_value, [$(12)])
    tl.store($(candidate_vsub_48_ptr) + r366_z, r366_value)
    r367_z = tl.arange($(576))
    r367_value = (tl.cast((r367_z % $(48)), tl.int64) > tl.load($(candidate_vsub_48_ptr) + ((r367_z / $(48)) % $(12)), dtype=tl.int64))
    r367_value = tl.broadcast_to(r367_value, [$(576)])
    tl.store($(candidate_vgt_27_ptr) + r367_z, r367_value)
    r368_z = tl.arange($(4608))
    r368_value = tl.where(tl.load($(candidate_vgt_27_ptr) + ((((r368_z / $(48)) % $(12)) * $(48)) + (r368_z % $(48))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_74_ptr) + ((((r368_z / $(576)) % $(8)) * $(576)) + (((r368_z / $(48)) % $(12)) * $(48)) + (r368_z % $(48))), dtype=tl.float32))
    r368_value = r368_value.to(tl.float32)
    r368_value = tl.broadcast_to(r368_value, [$(4608)])
    tl.store($(candidate_vmasked_fill_27_ptr) + r368_z, r368_value)
    r369_z = tl.arange($(4608))
    r369_value = (tl.load($(candidate_vmasked_fill_27_ptr) + ((((r369_z / $(576)) % $(8)) * $(576)) + (((r369_z / $(48)) % $(12)) * $(48)) + (r369_z % $(48))), dtype=tl.float32) - tl.load($(candidate_vcat_42_ptr) + ((((r369_z / $(48)) % $(12)) * $(8)) + ((r369_z / $(576)) % $(8))), dtype=tl.float32))
    r369_value = r369_value.to(tl.float32)
    r369_value = tl.broadcast_to(r369_value, [$(4608)])
    tl.store($(candidate_vsub_49_ptr) + r369_z, r369_value)
    r370_z = tl.arange($(4608))
    r370_value = tl.exp(tl.load($(candidate_vsub_49_ptr) + ((((r370_z / $(576)) % $(8)) * $(576)) + (((r370_z / $(48)) % $(12)) * $(48)) + (r370_z % $(48))), dtype=tl.float32))
    r370_value = r370_value.to(tl.float32)
    r370_value = tl.broadcast_to(r370_value, [$(4608)])
    tl.store($(candidate_vexp_6_ptr) + r370_z, r370_value)
    r371_z = tl.arange($(12288))
    r371_value = (tl.load($(candidate_vto_copy_168_ptr) + ((((r371_z / $(1536)) % $(8)) * $(1536)) + (((r371_z / $(128)) % $(12)) * $(128)) + (r371_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_169_ptr) + ((((r371_z / $(1536)) % $(8)) * $(1536)) + (((r371_z / $(128)) % $(12)) * $(128)) + (r371_z % $(128))), dtype=tl.float32))
    r371_value = r371_value.to(tl.float32)
    r371_value = tl.broadcast_to(r371_value, [$(12288)])
    tl.store($(candidate_vmul_75_ptr) + r371_z, r371_value)
    r372_z = tl.arange($(96))
    r372_lane = tl.arange($(128))
    r372_value = tl.sum(tl.load($(candidate_vmul_75_ptr) + ((((r372_z[:, None] / $(12)) % $(8)) * $(1536)) + ((r372_z[:, None] % $(12)) * $(128)) + r372_lane[None, :]), dtype=tl.float32), axis=1)
    r372_value = r372_value.to(tl.float32)
    r372_value = tl.broadcast_to(r372_value, [$(96)])
    tl.store($(candidate_vsum_9_ptr) + r372_z, r372_value)
    r373_row = tl.arange($(12))
    r373_col = tl.arange($(48))
    r373_lane = tl.arange($(128))
    for r373_group in range($(8)) {
      r373_left = tl.load($(candidate_vto_copy_168_ptr) + (((((((r373_group % $(8)) * $(1536)) + (r373_row[:, None] * $(128)) + r373_lane[None, :]) / $(1536)) % $(8)) * $(1536)) + ((((((r373_group % $(8)) * $(1536)) + (r373_row[:, None] * $(128)) + r373_lane[None, :]) / $(128)) % $(12)) * $(128)) + ((((r373_group % $(8)) * $(1536)) + (r373_row[:, None] * $(128)) + r373_lane[None, :]) % $(128))), dtype=tl.float32)
      r373_right = tl.load($(candidate_vto_copy_167_ptr) + (((((((r373_group % $(8)) * $(6144)) + (r373_lane[:, None] * $(48)) + r373_col[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r373_group % $(8)) * $(6144)) + (r373_lane[:, None] * $(48)) + r373_col[None, :]) % $(48)) * $(128)) + (((((r373_group % $(8)) * $(6144)) + (r373_lane[:, None] * $(48)) + r373_col[None, :]) / $(48)) % $(128))), dtype=tl.float32)
      r373_value = tl.dot(r373_left, r373_right)
      r373_value = r373_value.to(tl.float32)
      tl.store($(candidate_vbmm_85_ptr) + ((r373_group * $(576)) + (r373_row[:, None] * $(48)) + r373_col[None, :]), r373_value)
    }
    r374_z = tl.arange($(4608))
    r374_value = (tl.load($(candidate_vbmm_85_ptr) + ((((r374_z / $(576)) % $(8)) * $(576)) + (((r374_z / $(48)) % $(12)) * $(48)) + (r374_z % $(48))), dtype=tl.float32) - tl.load($(candidate_vsum_9_ptr) + ((((r374_z / $(576)) % $(8)) * $(12)) + ((r374_z / $(48)) % $(12))), dtype=tl.float32))
    r374_value = r374_value.to(tl.float32)
    r374_value = tl.broadcast_to(r374_value, [$(4608)])
    tl.store($(candidate_vsub_50_ptr) + r374_z, r374_value)
    r375_z = tl.arange($(4608))
    r375_value = (tl.load($(candidate_vexp_6_ptr) + ((((r375_z / $(576)) % $(8)) * $(576)) + (((r375_z / $(48)) % $(12)) * $(48)) + (r375_z % $(48))), dtype=tl.float32) * tl.load($(candidate_vsub_50_ptr) + ((((r375_z / $(576)) % $(8)) * $(576)) + (((r375_z / $(48)) % $(12)) * $(48)) + (r375_z % $(48))), dtype=tl.float32))
    r375_value = r375_value.to(tl.float32)
    r375_value = tl.broadcast_to(r375_value, [$(4608)])
    tl.store($(candidate_vmul_76_ptr) + r375_z, r375_value)
    r376_row = tl.arange($(48))
    r376_col = tl.arange($(192))
    r376_lane = tl.arange($(12))
    for r376_group in range($(8)) {
      r376_left = tl.load($(candidate_vmul_76_ptr) + (((((((r376_group % $(8)) * $(576)) + (r376_row[:, None] * $(12)) + r376_lane[None, :]) / $(576)) % $(8)) * $(576)) + (((((r376_group % $(8)) * $(576)) + (r376_row[:, None] * $(12)) + r376_lane[None, :]) % $(12)) * $(48)) + (((((r376_group % $(8)) * $(576)) + (r376_row[:, None] * $(12)) + r376_lane[None, :]) / $(12)) % $(48))), dtype=tl.float32)
      r376_right = tl.load($(candidate_vto_copy_165_ptr) + (((((((r376_group % $(8)) * $(2304)) + (r376_lane[:, None] * $(192)) + r376_col[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r376_group % $(8)) * $(2304)) + (r376_lane[:, None] * $(192)) + r376_col[None, :]) / $(192)) % $(12)) * $(192)) + ((((r376_group % $(8)) * $(2304)) + (r376_lane[:, None] * $(192)) + r376_col[None, :]) % $(192))), dtype=tl.float32)
      r376_value = tl.dot(r376_left, r376_right)
      r376_value = r376_value.to(tl.float32)
      tl.store($(candidate_vbmm_87_ptr) + ((r376_group * $(9216)) + (r376_row[:, None] * $(192)) + r376_col[None, :]), r376_value)
    }
    r377_z = tl.arange($(73728))
    r377_value = (tl.load($(candidate_vbmm_87_ptr) + ((((r377_z / $(9216)) % $(8)) * $(9216)) + (((r377_z / $(192)) % $(48)) * $(192)) + (r377_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r377_value = r377_value.to(tl.float32)
    r377_value = tl.broadcast_to(r377_value, [$(73728)])
    tl.store($(candidate_vmul_78_ptr) + r377_z, r377_value)
    r378_z = tl.arange($(73728))
    r378_value = tl.load($(candidate_vmul_78_ptr) + ((((r378_z / $(192)) % $(8)) * $(9216)) + (((r378_z / $(1536)) % $(48)) * $(192)) + (r378_z % $(192))), dtype=tl.float32)
    r378_value = r378_value.to(tl.bfloat16)
    r378_value = tl.broadcast_to(r378_value, [$(73728)])
    tl.store($(candidate_vto_copy_171_ptr) + r378_z, r378_value)
    r379_row = tl.arange($(48))
    r379_col = tl.arange($(128))
    r379_lane = tl.arange($(12))
    for r379_group in range($(8)) {
      r379_left = tl.load($(candidate_vexp_6_ptr) + (((((((r379_group % $(8)) * $(576)) + (r379_row[:, None] * $(12)) + r379_lane[None, :]) / $(576)) % $(8)) * $(576)) + (((((r379_group % $(8)) * $(576)) + (r379_row[:, None] * $(12)) + r379_lane[None, :]) % $(12)) * $(48)) + (((((r379_group % $(8)) * $(576)) + (r379_row[:, None] * $(12)) + r379_lane[None, :]) / $(12)) % $(48))), dtype=tl.float32)
      r379_right = tl.load($(candidate_vto_copy_168_ptr) + (((((((r379_group % $(8)) * $(1536)) + (r379_lane[:, None] * $(128)) + r379_col[None, :]) / $(1536)) % $(8)) * $(1536)) + ((((((r379_group % $(8)) * $(1536)) + (r379_lane[:, None] * $(128)) + r379_col[None, :]) / $(128)) % $(12)) * $(128)) + ((((r379_group % $(8)) * $(1536)) + (r379_lane[:, None] * $(128)) + r379_col[None, :]) % $(128))), dtype=tl.float32)
      r379_value = tl.dot(r379_left, r379_right)
      r379_value = r379_value.to(tl.float32)
      tl.store($(candidate_vbmm_88_ptr) + ((r379_group * $(6144)) + (r379_row[:, None] * $(128)) + r379_col[None, :]), r379_value)
    }
    r380_z = tl.arange($(49152))
    r380_value = tl.load($(candidate_vbmm_88_ptr) + ((((r380_z / $(128)) % $(8)) * $(6144)) + (((r380_z / $(1024)) % $(48)) * $(128)) + (r380_z % $(128))), dtype=tl.float32)
    r380_value = r380_value.to(tl.bfloat16)
    r380_value = tl.broadcast_to(r380_value, [$(49152)])
    tl.store($(candidate_vto_copy_172_ptr) + r380_z, r380_value)
    r381_z = tl.arange($(6144))
    r381_value = tl.load($(candidate_vcat_39_ptr) + ((((r381_z / $(192)) % $(4)) * $(1536)) + (((r381_z / $(768)) % $(8)) * $(192)) + (r381_z % $(192)) + $(18432)), dtype=tl.bfloat16)
    r381_value = r381_value.to(tl.float32)
    r381_value = tl.broadcast_to(r381_value, [$(6144)])
    tl.store($(candidate_vto_copy_173_ptr) + r381_z, r381_value)
    r382_z = tl.arange($(24576))
    r382_value = tl.load($(candidate_vcat_40_ptr) + ((((r382_z / $(192)) % $(16)) * $(1536)) + (((r382_z / $(3072)) % $(8)) * $(192)) + (r382_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r382_value = r382_value.to(tl.float32)
    r382_value = tl.broadcast_to(r382_value, [$(24576)])
    tl.store($(candidate_vto_copy_174_ptr) + r382_z, r382_value)
    r383_z = tl.arange($(16384))
    r383_value = tl.load($(candidate_vbmm_77_ptr) + ((((r383_z / $(128)) % $(16)) * $(1024)) + (((r383_z / $(2048)) % $(8)) * $(128)) + (r383_z % $(128)) + $(49152)), dtype=tl.bfloat16)
    r383_value = r383_value.to(tl.float32)
    r383_value = tl.broadcast_to(r383_value, [$(16384)])
    tl.store($(candidate_vto_copy_175_ptr) + r383_z, r383_value)
    r384_z = tl.arange($(4096))
    r384_value = tl.load($(candidate_vmm_28_ptr) + ((((r384_z / $(128)) % $(4)) * $(1024)) + (((r384_z / $(512)) % $(8)) * $(128)) + (r384_z % $(128)) + $(45056)), dtype=tl.bfloat16)
    r384_value = r384_value.to(tl.float32)
    r384_value = tl.broadcast_to(r384_value, [$(4096)])
    tl.store($(candidate_vto_copy_176_ptr) + r384_z, r384_value)
    r385_z = tl.arange($(4096))
    r385_value = tl.load($(candidate_vcat_41_ptr) + ((((r385_z / $(128)) % $(4)) * $(1024)) + (((r385_z / $(512)) % $(8)) * $(128)) + (r385_z % $(128)) + $(12288)), dtype=tl.bfloat16)
    r385_value = r385_value.to(tl.float32)
    r385_value = tl.broadcast_to(r385_value, [$(4096)])
    tl.store($(candidate_vto_copy_177_ptr) + r385_z, r385_value)
    r386_row = tl.arange($(4))
    r386_col = tl.arange($(16))
    r386_lane = tl.arange($(192))
    for r386_group in range($(8)) {
      r386_left = tl.load($(candidate_vto_copy_173_ptr) + (((((((r386_group % $(8)) * $(768)) + (r386_row[:, None] * $(192)) + r386_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r386_group % $(8)) * $(768)) + (r386_row[:, None] * $(192)) + r386_lane[None, :]) / $(192)) % $(4)) * $(192)) + ((((r386_group % $(8)) * $(768)) + (r386_row[:, None] * $(192)) + r386_lane[None, :]) % $(192))), dtype=tl.float32)
      r386_right = tl.load($(candidate_vto_copy_174_ptr) + (((((((r386_group % $(8)) * $(3072)) + (r386_lane[:, None] * $(16)) + r386_col[None, :]) / $(3072)) % $(8)) * $(3072)) + (((((r386_group % $(8)) * $(3072)) + (r386_lane[:, None] * $(16)) + r386_col[None, :]) % $(16)) * $(192)) + (((((r386_group % $(8)) * $(3072)) + (r386_lane[:, None] * $(16)) + r386_col[None, :]) / $(16)) % $(192))), dtype=tl.float32)
      r386_value = tl.dot(r386_left, r386_right)
      r386_value = r386_value.to(tl.float32)
      tl.store($(candidate_vbmm_89_ptr) + ((r386_group * $(64)) + (r386_row[:, None] * $(16)) + r386_col[None, :]), r386_value)
    }
    r387_z = tl.arange($(512))
    r387_value = (tl.load($(candidate_vbmm_89_ptr) + ((((r387_z / $(64)) % $(8)) * $(64)) + (((r387_z / $(16)) % $(4)) * $(16)) + (r387_z % $(16))), dtype=tl.float32) * 0.07216878364870322)
    r387_value = r387_value.to(tl.float32)
    r387_value = tl.broadcast_to(r387_value, [$(512)])
    tl.store($(candidate_vmul_79_ptr) + r387_z, r387_value)
    r388_z = tl.arange($(4))
    r388_value = (tl.cast(r388_z, tl.int64) + tl.full([], $(16), dtype=tl.int64))
    r388_value = tl.broadcast_to(r388_value, [$(4)])
    tl.store($(candidate_vadd_51_ptr) + r388_z, r388_value)
    r389_z = tl.arange($(4))
    r389_value = (tl.load($(candidate_vadd_51_ptr) + r389_z, dtype=tl.int64) - tl.full([], $(4), dtype=tl.int64))
    r389_value = tl.broadcast_to(r389_value, [$(4)])
    tl.store($(candidate_vsub_51_ptr) + r389_z, r389_value)
    r390_z = tl.arange($(64))
    r390_value = (tl.cast((r390_z % $(16)), tl.int64) > tl.load($(candidate_vsub_51_ptr) + ((r390_z / $(16)) % $(4)), dtype=tl.int64))
    r390_value = tl.broadcast_to(r390_value, [$(64)])
    tl.store($(candidate_vgt_28_ptr) + r390_z, r390_value)
    r391_z = tl.arange($(512))
    r391_value = tl.where(tl.load($(candidate_vgt_28_ptr) + ((((r391_z / $(16)) % $(4)) * $(16)) + (r391_z % $(16))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_79_ptr) + ((((r391_z / $(64)) % $(8)) * $(64)) + (((r391_z / $(16)) % $(4)) * $(16)) + (r391_z % $(16))), dtype=tl.float32))
    r391_value = r391_value.to(tl.float32)
    r391_value = tl.broadcast_to(r391_value, [$(512)])
    tl.store($(candidate_vmasked_fill_28_ptr) + r391_z, r391_value)
    r392_z = tl.arange($(512))
    r392_value = (tl.load($(candidate_vmasked_fill_28_ptr) + ((((r392_z / $(64)) % $(8)) * $(64)) + (((r392_z / $(16)) % $(4)) * $(16)) + (r392_z % $(16))), dtype=tl.float32) - tl.load($(candidate_vcat_42_ptr) + ((((r392_z / $(16)) % $(4)) * $(8)) + ((r392_z / $(64)) % $(8)) + $(96)), dtype=tl.float32))
    r392_value = r392_value.to(tl.float32)
    r392_value = tl.broadcast_to(r392_value, [$(512)])
    tl.store($(candidate_vsub_52_ptr) + r392_z, r392_value)
    r393_z = tl.arange($(512))
    r393_value = tl.exp(tl.load($(candidate_vsub_52_ptr) + ((((r393_z / $(64)) % $(8)) * $(64)) + (((r393_z / $(16)) % $(4)) * $(16)) + (r393_z % $(16))), dtype=tl.float32))
    r393_value = r393_value.to(tl.float32)
    r393_value = tl.broadcast_to(r393_value, [$(512)])
    tl.store($(candidate_vexp_7_ptr) + r393_z, r393_value)
    r394_z = tl.arange($(4096))
    r394_value = (tl.load($(candidate_vto_copy_176_ptr) + ((((r394_z / $(512)) % $(8)) * $(512)) + (((r394_z / $(128)) % $(4)) * $(128)) + (r394_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_177_ptr) + ((((r394_z / $(512)) % $(8)) * $(512)) + (((r394_z / $(128)) % $(4)) * $(128)) + (r394_z % $(128))), dtype=tl.float32))
    r394_value = r394_value.to(tl.float32)
    r394_value = tl.broadcast_to(r394_value, [$(4096)])
    tl.store($(candidate_vmul_80_ptr) + r394_z, r394_value)
    r395_z = tl.arange($(32))
    r395_lane = tl.arange($(128))
    r395_value = tl.sum(tl.load($(candidate_vmul_80_ptr) + ((((r395_z[:, None] / $(4)) % $(8)) * $(512)) + ((r395_z[:, None] % $(4)) * $(128)) + r395_lane[None, :]), dtype=tl.float32), axis=1)
    r395_value = r395_value.to(tl.float32)
    r395_value = tl.broadcast_to(r395_value, [$(32)])
    tl.store($(candidate_vsum_10_ptr) + r395_z, r395_value)
    r396_row = tl.arange($(4))
    r396_col = tl.arange($(16))
    r396_lane = tl.arange($(128))
    for r396_group in range($(8)) {
      r396_left = tl.load($(candidate_vto_copy_176_ptr) + (((((((r396_group % $(8)) * $(512)) + (r396_row[:, None] * $(128)) + r396_lane[None, :]) / $(512)) % $(8)) * $(512)) + ((((((r396_group % $(8)) * $(512)) + (r396_row[:, None] * $(128)) + r396_lane[None, :]) / $(128)) % $(4)) * $(128)) + ((((r396_group % $(8)) * $(512)) + (r396_row[:, None] * $(128)) + r396_lane[None, :]) % $(128))), dtype=tl.float32)
      r396_right = tl.load($(candidate_vto_copy_175_ptr) + (((((((r396_group % $(8)) * $(2048)) + (r396_lane[:, None] * $(16)) + r396_col[None, :]) / $(2048)) % $(8)) * $(2048)) + (((((r396_group % $(8)) * $(2048)) + (r396_lane[:, None] * $(16)) + r396_col[None, :]) % $(16)) * $(128)) + (((((r396_group % $(8)) * $(2048)) + (r396_lane[:, None] * $(16)) + r396_col[None, :]) / $(16)) % $(128))), dtype=tl.float32)
      r396_value = tl.dot(r396_left, r396_right)
      r396_value = r396_value.to(tl.float32)
      tl.store($(candidate_vbmm_90_ptr) + ((r396_group * $(64)) + (r396_row[:, None] * $(16)) + r396_col[None, :]), r396_value)
    }
    r397_z = tl.arange($(512))
    r397_value = (tl.load($(candidate_vbmm_90_ptr) + ((((r397_z / $(64)) % $(8)) * $(64)) + (((r397_z / $(16)) % $(4)) * $(16)) + (r397_z % $(16))), dtype=tl.float32) - tl.load($(candidate_vsum_10_ptr) + ((((r397_z / $(64)) % $(8)) * $(4)) + ((r397_z / $(16)) % $(4))), dtype=tl.float32))
    r397_value = r397_value.to(tl.float32)
    r397_value = tl.broadcast_to(r397_value, [$(512)])
    tl.store($(candidate_vsub_53_ptr) + r397_z, r397_value)
    r398_z = tl.arange($(512))
    r398_value = (tl.load($(candidate_vexp_7_ptr) + ((((r398_z / $(64)) % $(8)) * $(64)) + (((r398_z / $(16)) % $(4)) * $(16)) + (r398_z % $(16))), dtype=tl.float32) * tl.load($(candidate_vsub_53_ptr) + ((((r398_z / $(64)) % $(8)) * $(64)) + (((r398_z / $(16)) % $(4)) * $(16)) + (r398_z % $(16))), dtype=tl.float32))
    r398_value = r398_value.to(tl.float32)
    r398_value = tl.broadcast_to(r398_value, [$(512)])
    tl.store($(candidate_vmul_81_ptr) + r398_z, r398_value)
    r399_row = tl.arange($(16))
    r399_col = tl.arange($(192))
    r399_lane = tl.arange($(4))
    for r399_group in range($(8)) {
      r399_left = tl.load($(candidate_vmul_81_ptr) + (((((((r399_group % $(8)) * $(64)) + (r399_row[:, None] * $(4)) + r399_lane[None, :]) / $(64)) % $(8)) * $(64)) + (((((r399_group % $(8)) * $(64)) + (r399_row[:, None] * $(4)) + r399_lane[None, :]) % $(4)) * $(16)) + (((((r399_group % $(8)) * $(64)) + (r399_row[:, None] * $(4)) + r399_lane[None, :]) / $(4)) % $(16))), dtype=tl.float32)
      r399_right = tl.load($(candidate_vto_copy_173_ptr) + (((((((r399_group % $(8)) * $(768)) + (r399_lane[:, None] * $(192)) + r399_col[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r399_group % $(8)) * $(768)) + (r399_lane[:, None] * $(192)) + r399_col[None, :]) / $(192)) % $(4)) * $(192)) + ((((r399_group % $(8)) * $(768)) + (r399_lane[:, None] * $(192)) + r399_col[None, :]) % $(192))), dtype=tl.float32)
      r399_value = tl.dot(r399_left, r399_right)
      r399_value = r399_value.to(tl.float32)
      tl.store($(candidate_vbmm_92_ptr) + ((r399_group * $(3072)) + (r399_row[:, None] * $(192)) + r399_col[None, :]), r399_value)
    }
    r400_z = tl.arange($(24576))
    r400_value = (tl.load($(candidate_vbmm_92_ptr) + ((((r400_z / $(3072)) % $(8)) * $(3072)) + (((r400_z / $(192)) % $(16)) * $(192)) + (r400_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r400_value = r400_value.to(tl.float32)
    r400_value = tl.broadcast_to(r400_value, [$(24576)])
    tl.store($(candidate_vmul_83_ptr) + r400_z, r400_value)
    r401_z = tl.arange($(24576))
    r401_value = tl.load($(candidate_vmul_83_ptr) + ((((r401_z / $(192)) % $(8)) * $(3072)) + (((r401_z / $(1536)) % $(16)) * $(192)) + (r401_z % $(192))), dtype=tl.float32)
    r401_value = r401_value.to(tl.bfloat16)
    r401_value = tl.broadcast_to(r401_value, [$(24576)])
    tl.store($(candidate_vto_copy_179_ptr) + r401_z, r401_value)
    r402_row = tl.arange($(16))
    r402_col = tl.arange($(128))
    r402_lane = tl.arange($(4))
    for r402_group in range($(8)) {
      r402_left = tl.load($(candidate_vexp_7_ptr) + (((((((r402_group % $(8)) * $(64)) + (r402_row[:, None] * $(4)) + r402_lane[None, :]) / $(64)) % $(8)) * $(64)) + (((((r402_group % $(8)) * $(64)) + (r402_row[:, None] * $(4)) + r402_lane[None, :]) % $(4)) * $(16)) + (((((r402_group % $(8)) * $(64)) + (r402_row[:, None] * $(4)) + r402_lane[None, :]) / $(4)) % $(16))), dtype=tl.float32)
      r402_right = tl.load($(candidate_vto_copy_176_ptr) + (((((((r402_group % $(8)) * $(512)) + (r402_lane[:, None] * $(128)) + r402_col[None, :]) / $(512)) % $(8)) * $(512)) + ((((((r402_group % $(8)) * $(512)) + (r402_lane[:, None] * $(128)) + r402_col[None, :]) / $(128)) % $(4)) * $(128)) + ((((r402_group % $(8)) * $(512)) + (r402_lane[:, None] * $(128)) + r402_col[None, :]) % $(128))), dtype=tl.float32)
      r402_value = tl.dot(r402_left, r402_right)
      r402_value = r402_value.to(tl.float32)
      tl.store($(candidate_vbmm_93_ptr) + ((r402_group * $(2048)) + (r402_row[:, None] * $(128)) + r402_col[None, :]), r402_value)
    }
    r403_z = tl.arange($(16384))
    r403_value = tl.load($(candidate_vbmm_93_ptr) + ((((r403_z / $(128)) % $(8)) * $(2048)) + (((r403_z / $(1024)) % $(16)) * $(128)) + (r403_z % $(128))), dtype=tl.float32)
    r403_value = r403_value.to(tl.bfloat16)
    r403_value = tl.broadcast_to(r403_value, [$(16384)])
    tl.store($(candidate_vto_copy_180_ptr) + r403_z, r403_value)
    r404_z = tl.arange($(24576))
    r404_value = tl.load($(candidate_vcat_39_ptr) + ((((r404_z / $(192)) % $(16)) * $(1536)) + (((r404_z / $(3072)) % $(8)) * $(192)) + (r404_z % $(192)) + $(24576)), dtype=tl.bfloat16)
    r404_value = r404_value.to(tl.float32)
    r404_value = tl.broadcast_to(r404_value, [$(24576)])
    tl.store($(candidate_vto_copy_181_ptr) + r404_z, r404_value)
    r405_z = tl.arange($(98304))
    r405_value = tl.load($(candidate_vcat_40_ptr) + ((((r405_z / $(192)) % $(64)) * $(1536)) + (((r405_z / $(12288)) % $(8)) * $(192)) + (r405_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r405_value = r405_value.to(tl.float32)
    r405_value = tl.broadcast_to(r405_value, [$(98304)])
    tl.store($(candidate_vto_copy_182_ptr) + r405_z, r405_value)
    r406_z = tl.arange($(65536))
    r406_value = tl.load($(candidate_vbmm_77_ptr) + ((((r406_z / $(128)) % $(64)) * $(1024)) + (((r406_z / $(8192)) % $(8)) * $(128)) + (r406_z % $(128)) + $(65536)), dtype=tl.bfloat16)
    r406_value = r406_value.to(tl.float32)
    r406_value = tl.broadcast_to(r406_value, [$(65536)])
    tl.store($(candidate_vto_copy_183_ptr) + r406_z, r406_value)
    r407_z = tl.arange($(16384))
    r407_value = tl.load($(candidate_vmm_28_ptr) + ((((r407_z / $(128)) % $(16)) * $(1024)) + (((r407_z / $(2048)) % $(8)) * $(128)) + (r407_z % $(128)) + $(49152)), dtype=tl.bfloat16)
    r407_value = r407_value.to(tl.float32)
    r407_value = tl.broadcast_to(r407_value, [$(16384)])
    tl.store($(candidate_vto_copy_184_ptr) + r407_z, r407_value)
    r408_z = tl.arange($(16384))
    r408_value = tl.load($(candidate_vcat_41_ptr) + ((((r408_z / $(128)) % $(16)) * $(1024)) + (((r408_z / $(2048)) % $(8)) * $(128)) + (r408_z % $(128)) + $(16384)), dtype=tl.bfloat16)
    r408_value = r408_value.to(tl.float32)
    r408_value = tl.broadcast_to(r408_value, [$(16384)])
    tl.store($(candidate_vto_copy_185_ptr) + r408_z, r408_value)
    r409_row = tl.arange($(16))
    r409_col = tl.arange($(64))
    r409_lane = tl.arange($(192))
    for r409_group in range($(8)) {
      r409_left = tl.load($(candidate_vto_copy_181_ptr) + (((((((r409_group % $(8)) * $(3072)) + (r409_row[:, None] * $(192)) + r409_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r409_group % $(8)) * $(3072)) + (r409_row[:, None] * $(192)) + r409_lane[None, :]) / $(192)) % $(16)) * $(192)) + ((((r409_group % $(8)) * $(3072)) + (r409_row[:, None] * $(192)) + r409_lane[None, :]) % $(192))), dtype=tl.float32)
      r409_right = tl.load($(candidate_vto_copy_182_ptr) + (((((((r409_group % $(8)) * $(12288)) + (r409_lane[:, None] * $(64)) + r409_col[None, :]) / $(12288)) % $(8)) * $(12288)) + (((((r409_group % $(8)) * $(12288)) + (r409_lane[:, None] * $(64)) + r409_col[None, :]) % $(64)) * $(192)) + (((((r409_group % $(8)) * $(12288)) + (r409_lane[:, None] * $(64)) + r409_col[None, :]) / $(64)) % $(192))), dtype=tl.float32)
      r409_value = tl.dot(r409_left, r409_right)
      r409_value = r409_value.to(tl.float32)
      tl.store($(candidate_vbmm_94_ptr) + ((r409_group * $(1024)) + (r409_row[:, None] * $(64)) + r409_col[None, :]), r409_value)
    }
    r410_z = tl.arange($(8192))
    r410_value = (tl.load($(candidate_vbmm_94_ptr) + ((((r410_z / $(1024)) % $(8)) * $(1024)) + (((r410_z / $(64)) % $(16)) * $(64)) + (r410_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r410_value = r410_value.to(tl.float32)
    r410_value = tl.broadcast_to(r410_value, [$(8192)])
    tl.store($(candidate_vmul_84_ptr) + r410_z, r410_value)
    r411_z = tl.arange($(16))
    r411_value = (tl.cast(r411_z, tl.int64) + tl.full([], $(64), dtype=tl.int64))
    r411_value = tl.broadcast_to(r411_value, [$(16)])
    tl.store($(candidate_vadd_52_ptr) + r411_z, r411_value)
    r412_z = tl.arange($(16))
    r412_value = (tl.load($(candidate_vadd_52_ptr) + r412_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r412_value = tl.broadcast_to(r412_value, [$(16)])
    tl.store($(candidate_vsub_54_ptr) + r412_z, r412_value)
    r413_z = tl.arange($(1024))
    r413_value = (tl.cast((r413_z % $(64)), tl.int64) > tl.load($(candidate_vsub_54_ptr) + ((r413_z / $(64)) % $(16)), dtype=tl.int64))
    r413_value = tl.broadcast_to(r413_value, [$(1024)])
    tl.store($(candidate_vgt_29_ptr) + r413_z, r413_value)
    r414_z = tl.arange($(8192))
    r414_value = tl.where(tl.load($(candidate_vgt_29_ptr) + ((((r414_z / $(64)) % $(16)) * $(64)) + (r414_z % $(64))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_84_ptr) + ((((r414_z / $(1024)) % $(8)) * $(1024)) + (((r414_z / $(64)) % $(16)) * $(64)) + (r414_z % $(64))), dtype=tl.float32))
    r414_value = r414_value.to(tl.float32)
    r414_value = tl.broadcast_to(r414_value, [$(8192)])
    tl.store($(candidate_vmasked_fill_29_ptr) + r414_z, r414_value)
    r415_z = tl.arange($(8192))
    r415_value = (tl.load($(candidate_vmasked_fill_29_ptr) + ((((r415_z / $(1024)) % $(8)) * $(1024)) + (((r415_z / $(64)) % $(16)) * $(64)) + (r415_z % $(64))), dtype=tl.float32) - tl.load($(candidate_vcat_42_ptr) + ((((r415_z / $(64)) % $(16)) * $(8)) + ((r415_z / $(1024)) % $(8)) + $(128)), dtype=tl.float32))
    r415_value = r415_value.to(tl.float32)
    r415_value = tl.broadcast_to(r415_value, [$(8192)])
    tl.store($(candidate_vsub_55_ptr) + r415_z, r415_value)
    r416_z = tl.arange($(8192))
    r416_value = tl.exp(tl.load($(candidate_vsub_55_ptr) + ((((r416_z / $(1024)) % $(8)) * $(1024)) + (((r416_z / $(64)) % $(16)) * $(64)) + (r416_z % $(64))), dtype=tl.float32))
    r416_value = r416_value.to(tl.float32)
    r416_value = tl.broadcast_to(r416_value, [$(8192)])
    tl.store($(candidate_vexp_8_ptr) + r416_z, r416_value)
    r417_z = tl.arange($(16384))
    r417_value = (tl.load($(candidate_vto_copy_184_ptr) + ((((r417_z / $(2048)) % $(8)) * $(2048)) + (((r417_z / $(128)) % $(16)) * $(128)) + (r417_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_185_ptr) + ((((r417_z / $(2048)) % $(8)) * $(2048)) + (((r417_z / $(128)) % $(16)) * $(128)) + (r417_z % $(128))), dtype=tl.float32))
    r417_value = r417_value.to(tl.float32)
    r417_value = tl.broadcast_to(r417_value, [$(16384)])
    tl.store($(candidate_vmul_85_ptr) + r417_z, r417_value)
    r418_z = tl.arange($(128))
    r418_lane = tl.arange($(128))
    r418_value = tl.sum(tl.load($(candidate_vmul_85_ptr) + ((((r418_z[:, None] / $(16)) % $(8)) * $(2048)) + ((r418_z[:, None] % $(16)) * $(128)) + r418_lane[None, :]), dtype=tl.float32), axis=1)
    r418_value = r418_value.to(tl.float32)
    r418_value = tl.broadcast_to(r418_value, [$(128)])
    tl.store($(candidate_vsum_11_ptr) + r418_z, r418_value)
    r419_row = tl.arange($(16))
    r419_col = tl.arange($(64))
    r419_lane = tl.arange($(128))
    for r419_group in range($(8)) {
      r419_left = tl.load($(candidate_vto_copy_184_ptr) + (((((((r419_group % $(8)) * $(2048)) + (r419_row[:, None] * $(128)) + r419_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r419_group % $(8)) * $(2048)) + (r419_row[:, None] * $(128)) + r419_lane[None, :]) / $(128)) % $(16)) * $(128)) + ((((r419_group % $(8)) * $(2048)) + (r419_row[:, None] * $(128)) + r419_lane[None, :]) % $(128))), dtype=tl.float32)
      r419_right = tl.load($(candidate_vto_copy_183_ptr) + (((((((r419_group % $(8)) * $(8192)) + (r419_lane[:, None] * $(64)) + r419_col[None, :]) / $(8192)) % $(8)) * $(8192)) + (((((r419_group % $(8)) * $(8192)) + (r419_lane[:, None] * $(64)) + r419_col[None, :]) % $(64)) * $(128)) + (((((r419_group % $(8)) * $(8192)) + (r419_lane[:, None] * $(64)) + r419_col[None, :]) / $(64)) % $(128))), dtype=tl.float32)
      r419_value = tl.dot(r419_left, r419_right)
      r419_value = r419_value.to(tl.float32)
      tl.store($(candidate_vbmm_95_ptr) + ((r419_group * $(1024)) + (r419_row[:, None] * $(64)) + r419_col[None, :]), r419_value)
    }
    r420_z = tl.arange($(8192))
    r420_value = (tl.load($(candidate_vbmm_95_ptr) + ((((r420_z / $(1024)) % $(8)) * $(1024)) + (((r420_z / $(64)) % $(16)) * $(64)) + (r420_z % $(64))), dtype=tl.float32) - tl.load($(candidate_vsum_11_ptr) + ((((r420_z / $(1024)) % $(8)) * $(16)) + ((r420_z / $(64)) % $(16))), dtype=tl.float32))
    r420_value = r420_value.to(tl.float32)
    r420_value = tl.broadcast_to(r420_value, [$(8192)])
    tl.store($(candidate_vsub_56_ptr) + r420_z, r420_value)
    r421_z = tl.arange($(8192))
    r421_value = (tl.load($(candidate_vexp_8_ptr) + ((((r421_z / $(1024)) % $(8)) * $(1024)) + (((r421_z / $(64)) % $(16)) * $(64)) + (r421_z % $(64))), dtype=tl.float32) * tl.load($(candidate_vsub_56_ptr) + ((((r421_z / $(1024)) % $(8)) * $(1024)) + (((r421_z / $(64)) % $(16)) * $(64)) + (r421_z % $(64))), dtype=tl.float32))
    r421_value = r421_value.to(tl.float32)
    r421_value = tl.broadcast_to(r421_value, [$(8192)])
    tl.store($(candidate_vmul_86_ptr) + r421_z, r421_value)
    r422_row = tl.arange($(64))
    r422_col = tl.arange($(192))
    r422_lane = tl.arange($(16))
    for r422_group in range($(8)) {
      r422_left = tl.load($(candidate_vmul_86_ptr) + (((((((r422_group % $(8)) * $(1024)) + (r422_row[:, None] * $(16)) + r422_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + (((((r422_group % $(8)) * $(1024)) + (r422_row[:, None] * $(16)) + r422_lane[None, :]) % $(16)) * $(64)) + (((((r422_group % $(8)) * $(1024)) + (r422_row[:, None] * $(16)) + r422_lane[None, :]) / $(16)) % $(64))), dtype=tl.float32)
      r422_right = tl.load($(candidate_vto_copy_181_ptr) + (((((((r422_group % $(8)) * $(3072)) + (r422_lane[:, None] * $(192)) + r422_col[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r422_group % $(8)) * $(3072)) + (r422_lane[:, None] * $(192)) + r422_col[None, :]) / $(192)) % $(16)) * $(192)) + ((((r422_group % $(8)) * $(3072)) + (r422_lane[:, None] * $(192)) + r422_col[None, :]) % $(192))), dtype=tl.float32)
      r422_value = tl.dot(r422_left, r422_right)
      r422_value = r422_value.to(tl.float32)
      tl.store($(candidate_vbmm_97_ptr) + ((r422_group * $(12288)) + (r422_row[:, None] * $(192)) + r422_col[None, :]), r422_value)
    }
    r423_z = tl.arange($(98304))
    r423_value = (tl.load($(candidate_vbmm_97_ptr) + ((((r423_z / $(12288)) % $(8)) * $(12288)) + (((r423_z / $(192)) % $(64)) * $(192)) + (r423_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r423_value = r423_value.to(tl.float32)
    r423_value = tl.broadcast_to(r423_value, [$(98304)])
    tl.store($(candidate_vmul_88_ptr) + r423_z, r423_value)
    r424_z = tl.arange($(98304))
    r424_value = tl.load($(candidate_vmul_88_ptr) + ((((r424_z / $(192)) % $(8)) * $(12288)) + (((r424_z / $(1536)) % $(64)) * $(192)) + (r424_z % $(192))), dtype=tl.float32)
    r424_value = r424_value.to(tl.bfloat16)
    r424_value = tl.broadcast_to(r424_value, [$(98304)])
    tl.store($(candidate_vto_copy_187_ptr) + r424_z, r424_value)
    r425_row = tl.arange($(64))
    r425_col = tl.arange($(128))
    r425_lane = tl.arange($(16))
    for r425_group in range($(8)) {
      r425_left = tl.load($(candidate_vexp_8_ptr) + (((((((r425_group % $(8)) * $(1024)) + (r425_row[:, None] * $(16)) + r425_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + (((((r425_group % $(8)) * $(1024)) + (r425_row[:, None] * $(16)) + r425_lane[None, :]) % $(16)) * $(64)) + (((((r425_group % $(8)) * $(1024)) + (r425_row[:, None] * $(16)) + r425_lane[None, :]) / $(16)) % $(64))), dtype=tl.float32)
      r425_right = tl.load($(candidate_vto_copy_184_ptr) + (((((((r425_group % $(8)) * $(2048)) + (r425_lane[:, None] * $(128)) + r425_col[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r425_group % $(8)) * $(2048)) + (r425_lane[:, None] * $(128)) + r425_col[None, :]) / $(128)) % $(16)) * $(128)) + ((((r425_group % $(8)) * $(2048)) + (r425_lane[:, None] * $(128)) + r425_col[None, :]) % $(128))), dtype=tl.float32)
      r425_value = tl.dot(r425_left, r425_right)
      r425_value = r425_value.to(tl.float32)
      tl.store($(candidate_vbmm_98_ptr) + ((r425_group * $(8192)) + (r425_row[:, None] * $(128)) + r425_col[None, :]), r425_value)
    }
    r426_z = tl.arange($(65536))
    r426_value = tl.load($(candidate_vbmm_98_ptr) + ((((r426_z / $(128)) % $(8)) * $(8192)) + (((r426_z / $(1024)) % $(64)) * $(128)) + (r426_z % $(128))), dtype=tl.float32)
    r426_value = r426_value.to(tl.bfloat16)
    r426_value = tl.broadcast_to(r426_value, [$(65536)])
    tl.store($(candidate_vto_copy_188_ptr) + r426_z, r426_value)
    r427_z = tl.arange($(196608))
    r427_value = tl.where(((r427_z / $(1536)) % $(128)) < $(48), tl.load($(candidate_vto_copy_171_ptr) + ((((((r427_z / $(1536)) % $(128)) - $(0)) % $(48)) * $(1536)) + (((r427_z / $(192)) % $(8)) * $(192)) + (r427_z % $(192))), dtype=tl.bfloat16), tl.where(((r427_z / $(1536)) % $(128)) < $(64), tl.load($(candidate_vto_copy_179_ptr) + ((((((r427_z / $(1536)) % $(128)) - $(48)) % $(16)) * $(1536)) + (((r427_z / $(192)) % $(8)) * $(192)) + (r427_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vto_copy_187_ptr) + ((((((r427_z / $(1536)) % $(128)) - $(64)) % $(64)) * $(1536)) + (((r427_z / $(192)) % $(8)) * $(192)) + (r427_z % $(192))), dtype=tl.bfloat16)))
    r427_value = r427_value.to(tl.bfloat16)
    r427_value = tl.broadcast_to(r427_value, [$(196608)])
    tl.store($(candidate_vcat_44_ptr) + r427_z, r427_value)
    r428_z = tl.arange($(131072))
    r428_value = tl.where(((r428_z / $(1024)) % $(128)) < $(48), tl.load($(candidate_vto_copy_172_ptr) + ((((((r428_z / $(1024)) % $(128)) - $(0)) % $(48)) * $(1024)) + (((r428_z / $(128)) % $(8)) * $(128)) + (r428_z % $(128))), dtype=tl.bfloat16), tl.where(((r428_z / $(1024)) % $(128)) < $(64), tl.load($(candidate_vto_copy_180_ptr) + ((((((r428_z / $(1024)) % $(128)) - $(48)) % $(16)) * $(1024)) + (((r428_z / $(128)) % $(8)) * $(128)) + (r428_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vto_copy_188_ptr) + ((((((r428_z / $(1024)) % $(128)) - $(64)) % $(64)) * $(1024)) + (((r428_z / $(128)) % $(8)) * $(128)) + (r428_z % $(128))), dtype=tl.bfloat16)))
    r428_value = r428_value.to(tl.bfloat16)
    r428_value = tl.broadcast_to(r428_value, [$(131072)])
    tl.store($(candidate_vcat_45_ptr) + r428_z, r428_value)
    r429_row = tl.arange($(1024))
    r429_col = tl.arange($(512))
    r429_lane = tl.arange($(128))
    for r429_group in range($(1)) {
      r429_left = tl.load($(candidate_vcat_44_ptr) + (((((r429_row[:, None] * $(128)) + r429_lane[None, :]) % $(128)) * $(1536)) + (((((r429_row[:, None] * $(128)) + r429_lane[None, :]) / $(16384)) % $(8)) * $(192)) + ((((r429_row[:, None] * $(128)) + r429_lane[None, :]) / $(128)) % $(128))), dtype=tl.bfloat16)
      r429_right = tl.load($(candidate_vindex_2_ptr) + ((((((r429_lane[:, None] * $(512)) + r429_col[None, :]) / $(512)) % $(128)) * $(576)) + (((r429_lane[:, None] * $(512)) + r429_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r429_value = tl.dot(r429_left, r429_right)
      r429_value = r429_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_101_ptr) + ((r429_group * $(524288)) + (r429_row[:, None] * $(512)) + r429_col[None, :]), r429_value)
    }
    r430_row = tl.arange($(1024))
    r430_col = tl.arange($(512))
    r430_lane = tl.arange($(128))
    for r430_group in range($(1)) {
      r430_left = tl.load($(candidate_vcat_45_ptr) + (((((r430_row[:, None] * $(128)) + r430_lane[None, :]) % $(128)) * $(1024)) + (((((r430_row[:, None] * $(128)) + r430_lane[None, :]) / $(16384)) % $(8)) * $(128)) + ((((r430_row[:, None] * $(128)) + r430_lane[None, :]) / $(128)) % $(128))), dtype=tl.bfloat16)
      r430_right = tl.load($(candidate_vindex_2_ptr) + ((((((r430_lane[:, None] * $(512)) + r430_col[None, :]) / $(512)) % $(128)) * $(576)) + (((r430_lane[:, None] * $(512)) + r430_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r430_value = tl.dot(r430_left, r430_right)
      r430_value = r430_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_102_ptr) + ((r430_group * $(524288)) + (r430_row[:, None] * $(512)) + r430_col[None, :]), r430_value)
    }
    r431_z = tl.arange($(1048576))
    r431_value = tl.where(((r431_z / $(512)) % $(256)) < $(128), tl.load($(candidate_vbmm_101_ptr) + ((((((((r431_z / $(131072)) % $(8)) * $(65536)) + ((((r431_z / $(512)) % $(256)) - $(0)) * $(512)) + (r431_z % $(512))) / $(65536)) % $(8)) * $(65536)) + ((((((r431_z / $(131072)) % $(8)) * $(128)) + (((r431_z / $(512)) % $(256)) - $(0))) % $(128)) * $(512)) + (r431_z % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vbmm_102_ptr) + ((((((((r431_z / $(131072)) % $(8)) * $(65536)) + ((((r431_z / $(512)) % $(256)) - $(128)) * $(512)) + (r431_z % $(512))) / $(65536)) % $(8)) * $(65536)) + ((((((r431_z / $(131072)) % $(8)) * $(128)) + (((r431_z / $(512)) % $(256)) - $(128))) % $(128)) * $(512)) + (r431_z % $(512))), dtype=tl.bfloat16))
    r431_value = r431_value.to(tl.bfloat16)
    r431_value = tl.broadcast_to(r431_value, [$(1048576)])
    tl.store($(candidate_vcat_46_ptr) + r431_z, r431_value)
    r432_z = tl.arange($(1048576))
    r432_value = (tl.load($(candidate_vadd_46_ptr) + ((((r432_z / $(512)) % $(2048)) * $(512)) + (r432_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vcat_46_ptr) + ((((((((r432_z / $(512)) % $(2048)) * $(512)) + (r432_z % $(512))) / $(131072)) % $(8)) * $(131072)) + ((((r432_z / $(512)) % $(2048)) % $(256)) * $(512)) + (r432_z % $(512))), dtype=tl.bfloat16))
    r432_value = r432_value.to(tl.bfloat16)
    r432_value = tl.broadcast_to(r432_value, [$(1048576)])
    tl.store($(candidate_vadd_54_ptr) + r432_z, r432_value)
    r433_row = tl.arange($(128))
    r433_col = tl.arange($(1024))
    r433_lane = tl.arange($(512))
    for r433_group in range($(1)) {
      r433_left = tl.load($(candidate_vindex_2_ptr) + ((((((r433_row[:, None] * $(512)) + r433_lane[None, :]) / $(512)) % $(128)) * $(576)) + (((r433_row[:, None] * $(512)) + r433_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r433_right = tl.load($(candidate_vto_copy_20_ptr) + ((((((r433_lane[:, None] * $(1024)) + r433_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r433_lane[:, None] * $(1024)) + r433_col[None, :]) % $(128)) * $(512)) + ((((r433_lane[:, None] * $(1024)) + r433_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r433_value = tl.dot(r433_left, r433_right)
      r433_value = r433_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_103_ptr) + ((r433_group * $(131072)) + (r433_row[:, None] * $(1024)) + r433_col[None, :]), r433_value)
    }
    r434_row = tl.arange($(128))
    r434_col = tl.arange($(1024))
    r434_lane = tl.arange($(512))
    for r434_group in range($(1)) {
      r434_left = tl.load($(candidate_vindex_2_ptr) + ((((((r434_row[:, None] * $(512)) + r434_lane[None, :]) / $(512)) % $(128)) * $(576)) + (((r434_row[:, None] * $(512)) + r434_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r434_right = tl.load($(candidate_vto_copy_20_ptr) + ((((((r434_lane[:, None] * $(1024)) + r434_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r434_lane[:, None] * $(1024)) + r434_col[None, :]) % $(128)) * $(512)) + ((((r434_lane[:, None] * $(1024)) + r434_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r434_value = tl.dot(r434_left, r434_right)
      r434_value = r434_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_104_ptr) + ((r434_group * $(131072)) + (r434_row[:, None] * $(1024)) + r434_col[None, :]), r434_value)
    }
    r435_z = tl.arange($(49152))
    r435_value = tl.where((r435_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r435_z / $(1536)) % $(32)) * $(1024)) + (((r435_z / $(192)) % $(8)) * $(128)) + ((r435_z % $(192)) - $(0))) / $(1024)) % $(128)) * $(1536)) + (((((((r435_z / $(1536)) % $(32)) * $(1024)) + (((r435_z / $(192)) % $(8)) * $(128)) + ((r435_z % $(192)) - $(0))) / $(128)) % $(8)) * $(192)) + (((((r435_z / $(1536)) % $(32)) * $(1024)) + (((r435_z / $(192)) % $(8)) * $(128)) + ((r435_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vadd_6_ptr) + ((((((((r435_z / $(1536)) % $(32)) * $(512)) + (((r435_z / $(192)) % $(8)) * $(64)) + ((r435_z % $(192)) - $(128))) / $(512)) % $(128)) * $(512)) + (((((((r435_z / $(1536)) % $(32)) * $(512)) + (((r435_z / $(192)) % $(8)) * $(64)) + ((r435_z % $(192)) - $(128))) / $(64)) % $(8)) * $(64)) + (((((r435_z / $(1536)) % $(32)) * $(512)) + (((r435_z / $(192)) % $(8)) * $(64)) + ((r435_z % $(192)) - $(128))) % $(64))), dtype=tl.bfloat16))
    r435_value = r435_value.to(tl.bfloat16)
    r435_value = tl.broadcast_to(r435_value, [$(49152)])
    tl.store($(candidate_vcat_47_ptr) + r435_z, r435_value)
    r436_z = tl.arange($(196608))
    r436_value = tl.where((r436_z % $(192)) < $(128), tl.load($(candidate_vbmm_103_ptr) + ((((((((r436_z / $(1536)) % $(128)) * $(1024)) + (((r436_z / $(192)) % $(8)) * $(128)) + ((r436_z % $(192)) - $(0))) / $(1024)) % $(128)) * $(1024)) + (((((((r436_z / $(1536)) % $(128)) * $(1024)) + (((r436_z / $(192)) % $(8)) * $(128)) + ((r436_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r436_z / $(1536)) % $(128)) * $(1024)) + (((r436_z / $(192)) % $(8)) * $(128)) + ((r436_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vindex_2_ptr) + ((((r436_z / $(1536)) % $(128)) * $(576)) + ((((r436_z % $(192)) - $(128)) + $(512)) % $(576))), dtype=tl.bfloat16))
    r436_value = r436_value.to(tl.bfloat16)
    r436_value = tl.broadcast_to(r436_value, [$(196608)])
    tl.store($(candidate_vcat_48_ptr) + r436_z, r436_value)
    r437_z = tl.arange($(18432))
    r437_value = tl.load($(candidate_vcat_47_ptr) + ((((r437_z / $(192)) % $(12)) * $(1536)) + (((r437_z / $(2304)) % $(8)) * $(192)) + (r437_z % $(192))), dtype=tl.bfloat16)
    r437_value = r437_value.to(tl.float32)
    r437_value = tl.broadcast_to(r437_value, [$(18432)])
    tl.store($(candidate_vto_copy_189_ptr) + r437_z, r437_value)
    r438_z = tl.arange($(18432))
    r438_value = tl.load($(candidate_vcat_48_ptr) + ((((r438_z / $(192)) % $(12)) * $(1536)) + (((r438_z / $(2304)) % $(8)) * $(192)) + (r438_z % $(192))), dtype=tl.bfloat16)
    r438_value = r438_value.to(tl.float32)
    r438_value = tl.broadcast_to(r438_value, [$(18432)])
    tl.store($(candidate_vto_copy_190_ptr) + r438_z, r438_value)
    r439_row = tl.arange($(12))
    r439_col = tl.arange($(12))
    r439_lane = tl.arange($(192))
    for r439_group in range($(8)) {
      r439_left = tl.load($(candidate_vto_copy_189_ptr) + (((((((r439_group % $(8)) * $(2304)) + (r439_row[:, None] * $(192)) + r439_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r439_group % $(8)) * $(2304)) + (r439_row[:, None] * $(192)) + r439_lane[None, :]) / $(192)) % $(12)) * $(192)) + ((((r439_group % $(8)) * $(2304)) + (r439_row[:, None] * $(192)) + r439_lane[None, :]) % $(192))), dtype=tl.float32)
      r439_right = tl.load($(candidate_vto_copy_190_ptr) + (((((((r439_group % $(8)) * $(2304)) + (r439_lane[:, None] * $(12)) + r439_col[None, :]) / $(2304)) % $(8)) * $(2304)) + (((((r439_group % $(8)) * $(2304)) + (r439_lane[:, None] * $(12)) + r439_col[None, :]) % $(12)) * $(192)) + (((((r439_group % $(8)) * $(2304)) + (r439_lane[:, None] * $(12)) + r439_col[None, :]) / $(12)) % $(192))), dtype=tl.float32)
      r439_value = tl.dot(r439_left, r439_right)
      r439_value = r439_value.to(tl.float32)
      tl.store($(candidate_vbmm_105_ptr) + ((r439_group * $(144)) + (r439_row[:, None] * $(12)) + r439_col[None, :]), r439_value)
    }
    r440_z = tl.arange($(1152))
    r440_value = (tl.load($(candidate_vbmm_105_ptr) + ((((r440_z / $(144)) % $(8)) * $(144)) + (((r440_z / $(12)) % $(12)) * $(12)) + (r440_z % $(12))), dtype=tl.float32) * 0.07216878364870322)
    r440_value = r440_value.to(tl.float32)
    r440_value = tl.broadcast_to(r440_value, [$(1152)])
    tl.store($(candidate_vmul_89_ptr) + r440_z, r440_value)
    r441_z = tl.arange($(12))
    r441_value = (tl.cast(r441_z, tl.int64) + tl.full([], $(12), dtype=tl.int64))
    r441_value = tl.broadcast_to(r441_value, [$(12)])
    tl.store($(candidate_vadd_55_ptr) + r441_z, r441_value)
    r442_z = tl.arange($(12))
    r442_value = (tl.load($(candidate_vadd_55_ptr) + r442_z, dtype=tl.int64) - tl.full([], $(12), dtype=tl.int64))
    r442_value = tl.broadcast_to(r442_value, [$(12)])
    tl.store($(candidate_vsub_57_ptr) + r442_z, r442_value)
    r443_z = tl.arange($(144))
    r443_value = (tl.cast((r443_z % $(12)), tl.int64) > tl.load($(candidate_vsub_57_ptr) + ((r443_z / $(12)) % $(12)), dtype=tl.int64))
    r443_value = tl.broadcast_to(r443_value, [$(144)])
    tl.store($(candidate_vgt_30_ptr) + r443_z, r443_value)
    r444_z = tl.arange($(1152))
    r444_value = tl.where(tl.load($(candidate_vgt_30_ptr) + ((((r444_z / $(12)) % $(12)) * $(12)) + (r444_z % $(12))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_89_ptr) + ((((r444_z / $(144)) % $(8)) * $(144)) + (((r444_z / $(12)) % $(12)) * $(12)) + (r444_z % $(12))), dtype=tl.float32))
    r444_value = r444_value.to(tl.float32)
    r444_value = tl.broadcast_to(r444_value, [$(1152)])
    tl.store($(candidate_vmasked_fill_30_ptr) + r444_z, r444_value)
    r445_z = tl.arange($(1152))
    r445_lane = tl.arange($(12))
    r445_row_values = tl.load($(candidate_vmasked_fill_30_ptr) + ((((r445_z[:, None] / $(144)) % $(8)) * $(144)) + (((r445_z[:, None] / $(12)) % $(12)) * $(12)) + (r445_lane[None, :] % $(12))), dtype=tl.float32)
    r445_maximum = tl.max(r445_row_values, axis=1, keep_dims=true)
    r445_weights = tl.exp(r445_row_values - r445_maximum)
    r445_total = tl.sum(r445_weights, axis=1)
    r445_maximum = tl.max(r445_row_values, axis=1)
    r445_value = tl.exp(tl.load($(candidate_vmasked_fill_30_ptr) + ((((r445_z / $(144)) % $(8)) * $(144)) + (((r445_z / $(12)) % $(12)) * $(12)) + (r445_z % $(12))), dtype=tl.float32) - r445_maximum) / r445_total
    r445_value = r445_value.to(tl.float32)
    r445_value = tl.broadcast_to(r445_value, [$(1152)])
    tl.store($(candidate_vsoftmax_21_ptr) + r445_z, r445_value)
    r446_z = tl.arange($(1152))
    r446_value = tl.load($(candidate_vsoftmax_21_ptr) + ((((r446_z / $(144)) % $(8)) * $(144)) + (((r446_z / $(12)) % $(12)) * $(12)) + (r446_z % $(12))), dtype=tl.float32)
    r446_value = r446_value.to(tl.bfloat16)
    r446_value = tl.broadcast_to(r446_value, [$(1152)])
    tl.store($(candidate_vto_copy_191_ptr) + r446_z, r446_value)
    r447_row = tl.arange($(12))
    r447_col = tl.arange($(128))
    r447_lane = tl.arange($(12))
    for r447_group in range($(8)) {
      r447_left = tl.load($(candidate_vto_copy_191_ptr) + (((((((r447_group % $(8)) * $(144)) + (r447_row[:, None] * $(12)) + r447_lane[None, :]) / $(144)) % $(8)) * $(144)) + ((((((r447_group % $(8)) * $(144)) + (r447_row[:, None] * $(12)) + r447_lane[None, :]) / $(12)) % $(12)) * $(12)) + ((((r447_group % $(8)) * $(144)) + (r447_row[:, None] * $(12)) + r447_lane[None, :]) % $(12))), dtype=tl.bfloat16)
      r447_right = tl.load($(candidate_vbmm_104_ptr) + (((((((r447_group % $(8)) * $(1536)) + (r447_lane[:, None] * $(128)) + r447_col[None, :]) / $(128)) % $(12)) * $(1024)) + ((((((r447_group % $(8)) * $(1536)) + (r447_lane[:, None] * $(128)) + r447_col[None, :]) / $(1536)) % $(8)) * $(128)) + ((((r447_group % $(8)) * $(1536)) + (r447_lane[:, None] * $(128)) + r447_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r447_value = tl.dot(r447_left, r447_right)
      r447_value = r447_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_106_ptr) + ((r447_group * $(1536)) + (r447_row[:, None] * $(128)) + r447_col[None, :]), r447_value)
    }
    r448_z = tl.arange($(96))
    r448_lane = tl.arange($(12))
    r448_row_values = tl.load($(candidate_vmasked_fill_30_ptr) + ((((r448_z[:, None] / $(12)) % $(8)) * $(144)) + ((r448_z[:, None] % $(12)) * $(12)) + (r448_lane[None, :] % $(12))), dtype=tl.float32)
    r448_maximum = tl.max(r448_row_values, axis=1, keep_dims=true)
    r448_weights = tl.exp(r448_row_values - r448_maximum)
    r448_total = tl.sum(r448_weights, axis=1)
    r448_maximum = tl.max(r448_row_values, axis=1)
    r448_value = r448_maximum + tl.log(r448_total)
    r448_value = r448_value.to(tl.float32)
    r448_value = tl.broadcast_to(r448_value, [$(96)])
    tl.store($(candidate_vlogsumexp_21_ptr) + r448_z, r448_value)
    r449_z = tl.arange($(6144))
    r449_value = tl.load($(candidate_vcat_47_ptr) + ((((r449_z / $(192)) % $(4)) * $(1536)) + (((r449_z / $(768)) % $(8)) * $(192)) + (r449_z % $(192)) + $(18432)), dtype=tl.bfloat16)
    r449_value = r449_value.to(tl.float32)
    r449_value = tl.broadcast_to(r449_value, [$(6144)])
    tl.store($(candidate_vto_copy_192_ptr) + r449_z, r449_value)
    r450_z = tl.arange($(6144))
    r450_value = tl.load($(candidate_vcat_48_ptr) + ((((r450_z / $(192)) % $(4)) * $(1536)) + (((r450_z / $(768)) % $(8)) * $(192)) + (r450_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r450_value = r450_value.to(tl.float32)
    r450_value = tl.broadcast_to(r450_value, [$(6144)])
    tl.store($(candidate_vto_copy_193_ptr) + r450_z, r450_value)
    r451_row = tl.arange($(4))
    r451_col = tl.arange($(4))
    r451_lane = tl.arange($(192))
    for r451_group in range($(8)) {
      r451_left = tl.load($(candidate_vto_copy_192_ptr) + (((((((r451_group % $(8)) * $(768)) + (r451_row[:, None] * $(192)) + r451_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r451_group % $(8)) * $(768)) + (r451_row[:, None] * $(192)) + r451_lane[None, :]) / $(192)) % $(4)) * $(192)) + ((((r451_group % $(8)) * $(768)) + (r451_row[:, None] * $(192)) + r451_lane[None, :]) % $(192))), dtype=tl.float32)
      r451_right = tl.load($(candidate_vto_copy_193_ptr) + (((((((r451_group % $(8)) * $(768)) + (r451_lane[:, None] * $(4)) + r451_col[None, :]) / $(768)) % $(8)) * $(768)) + (((((r451_group % $(8)) * $(768)) + (r451_lane[:, None] * $(4)) + r451_col[None, :]) % $(4)) * $(192)) + (((((r451_group % $(8)) * $(768)) + (r451_lane[:, None] * $(4)) + r451_col[None, :]) / $(4)) % $(192))), dtype=tl.float32)
      r451_value = tl.dot(r451_left, r451_right)
      r451_value = r451_value.to(tl.float32)
      tl.store($(candidate_vbmm_107_ptr) + ((r451_group * $(16)) + (r451_row[:, None] * $(4)) + r451_col[None, :]), r451_value)
    }
    r452_z = tl.arange($(128))
    r452_value = (tl.load($(candidate_vbmm_107_ptr) + ((((r452_z / $(16)) % $(8)) * $(16)) + (((r452_z / $(4)) % $(4)) * $(4)) + (r452_z % $(4))), dtype=tl.float32) * 0.07216878364870322)
    r452_value = r452_value.to(tl.float32)
    r452_value = tl.broadcast_to(r452_value, [$(128)])
    tl.store($(candidate_vmul_90_ptr) + r452_z, r452_value)
    r453_z = tl.arange($(4))
    r453_value = (tl.cast(r453_z, tl.int64) + tl.full([], $(4), dtype=tl.int64))
    r453_value = tl.broadcast_to(r453_value, [$(4)])
    tl.store($(candidate_vadd_56_ptr) + r453_z, r453_value)
    r454_z = tl.arange($(4))
    r454_value = (tl.load($(candidate_vadd_56_ptr) + r454_z, dtype=tl.int64) - tl.full([], $(4), dtype=tl.int64))
    r454_value = tl.broadcast_to(r454_value, [$(4)])
    tl.store($(candidate_vsub_58_ptr) + r454_z, r454_value)
    r455_z = tl.arange($(16))
    r455_value = (tl.cast((r455_z % $(4)), tl.int64) > tl.load($(candidate_vsub_58_ptr) + ((r455_z / $(4)) % $(4)), dtype=tl.int64))
    r455_value = tl.broadcast_to(r455_value, [$(16)])
    tl.store($(candidate_vgt_31_ptr) + r455_z, r455_value)
    r456_z = tl.arange($(128))
    r456_value = tl.where(tl.load($(candidate_vgt_31_ptr) + ((((r456_z / $(4)) % $(4)) * $(4)) + (r456_z % $(4))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_90_ptr) + ((((r456_z / $(16)) % $(8)) * $(16)) + (((r456_z / $(4)) % $(4)) * $(4)) + (r456_z % $(4))), dtype=tl.float32))
    r456_value = r456_value.to(tl.float32)
    r456_value = tl.broadcast_to(r456_value, [$(128)])
    tl.store($(candidate_vmasked_fill_31_ptr) + r456_z, r456_value)
    r457_z = tl.arange($(128))
    r457_lane = tl.arange($(4))
    r457_row_values = tl.load($(candidate_vmasked_fill_31_ptr) + ((((r457_z[:, None] / $(16)) % $(8)) * $(16)) + (((r457_z[:, None] / $(4)) % $(4)) * $(4)) + (r457_lane[None, :] % $(4))), dtype=tl.float32)
    r457_maximum = tl.max(r457_row_values, axis=1, keep_dims=true)
    r457_weights = tl.exp(r457_row_values - r457_maximum)
    r457_total = tl.sum(r457_weights, axis=1)
    r457_maximum = tl.max(r457_row_values, axis=1)
    r457_value = tl.exp(tl.load($(candidate_vmasked_fill_31_ptr) + ((((r457_z / $(16)) % $(8)) * $(16)) + (((r457_z / $(4)) % $(4)) * $(4)) + (r457_z % $(4))), dtype=tl.float32) - r457_maximum) / r457_total
    r457_value = r457_value.to(tl.float32)
    r457_value = tl.broadcast_to(r457_value, [$(128)])
    tl.store($(candidate_vsoftmax_22_ptr) + r457_z, r457_value)
    r458_z = tl.arange($(128))
    r458_value = tl.load($(candidate_vsoftmax_22_ptr) + ((((r458_z / $(16)) % $(8)) * $(16)) + (((r458_z / $(4)) % $(4)) * $(4)) + (r458_z % $(4))), dtype=tl.float32)
    r458_value = r458_value.to(tl.bfloat16)
    r458_value = tl.broadcast_to(r458_value, [$(128)])
    tl.store($(candidate_vto_copy_194_ptr) + r458_z, r458_value)
    r459_row = tl.arange($(4))
    r459_col = tl.arange($(128))
    r459_lane = tl.arange($(4))
    for r459_group in range($(8)) {
      r459_left = tl.load($(candidate_vto_copy_194_ptr) + (((((((r459_group % $(8)) * $(16)) + (r459_row[:, None] * $(4)) + r459_lane[None, :]) / $(16)) % $(8)) * $(16)) + ((((((r459_group % $(8)) * $(16)) + (r459_row[:, None] * $(4)) + r459_lane[None, :]) / $(4)) % $(4)) * $(4)) + ((((r459_group % $(8)) * $(16)) + (r459_row[:, None] * $(4)) + r459_lane[None, :]) % $(4))), dtype=tl.bfloat16)
      r459_right = tl.load($(candidate_vbmm_104_ptr) + (((((((r459_group % $(8)) * $(512)) + (r459_lane[:, None] * $(128)) + r459_col[None, :]) / $(128)) % $(4)) * $(1024)) + ((((((r459_group % $(8)) * $(512)) + (r459_lane[:, None] * $(128)) + r459_col[None, :]) / $(512)) % $(8)) * $(128)) + ((((r459_group % $(8)) * $(512)) + (r459_lane[:, None] * $(128)) + r459_col[None, :]) % $(128)) + $(49152)), dtype=tl.bfloat16)
      r459_value = tl.dot(r459_left, r459_right)
      r459_value = r459_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_108_ptr) + ((r459_group * $(512)) + (r459_row[:, None] * $(128)) + r459_col[None, :]), r459_value)
    }
    r460_z = tl.arange($(32))
    r460_lane = tl.arange($(4))
    r460_row_values = tl.load($(candidate_vmasked_fill_31_ptr) + ((((r460_z[:, None] / $(4)) % $(8)) * $(16)) + ((r460_z[:, None] % $(4)) * $(4)) + (r460_lane[None, :] % $(4))), dtype=tl.float32)
    r460_maximum = tl.max(r460_row_values, axis=1, keep_dims=true)
    r460_weights = tl.exp(r460_row_values - r460_maximum)
    r460_total = tl.sum(r460_weights, axis=1)
    r460_maximum = tl.max(r460_row_values, axis=1)
    r460_value = r460_maximum + tl.log(r460_total)
    r460_value = r460_value.to(tl.float32)
    r460_value = tl.broadcast_to(r460_value, [$(32)])
    tl.store($(candidate_vlogsumexp_22_ptr) + r460_z, r460_value)
    r461_z = tl.arange($(24576))
    r461_value = tl.load($(candidate_vcat_47_ptr) + ((((r461_z / $(192)) % $(16)) * $(1536)) + (((r461_z / $(3072)) % $(8)) * $(192)) + (r461_z % $(192)) + $(24576)), dtype=tl.bfloat16)
    r461_value = r461_value.to(tl.float32)
    r461_value = tl.broadcast_to(r461_value, [$(24576)])
    tl.store($(candidate_vto_copy_195_ptr) + r461_z, r461_value)
    r462_z = tl.arange($(24576))
    r462_value = tl.load($(candidate_vcat_48_ptr) + ((((r462_z / $(192)) % $(16)) * $(1536)) + (((r462_z / $(3072)) % $(8)) * $(192)) + (r462_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r462_value = r462_value.to(tl.float32)
    r462_value = tl.broadcast_to(r462_value, [$(24576)])
    tl.store($(candidate_vto_copy_196_ptr) + r462_z, r462_value)
    r463_row = tl.arange($(16))
    r463_col = tl.arange($(16))
    r463_lane = tl.arange($(192))
    for r463_group in range($(8)) {
      r463_left = tl.load($(candidate_vto_copy_195_ptr) + (((((((r463_group % $(8)) * $(3072)) + (r463_row[:, None] * $(192)) + r463_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r463_group % $(8)) * $(3072)) + (r463_row[:, None] * $(192)) + r463_lane[None, :]) / $(192)) % $(16)) * $(192)) + ((((r463_group % $(8)) * $(3072)) + (r463_row[:, None] * $(192)) + r463_lane[None, :]) % $(192))), dtype=tl.float32)
      r463_right = tl.load($(candidate_vto_copy_196_ptr) + (((((((r463_group % $(8)) * $(3072)) + (r463_lane[:, None] * $(16)) + r463_col[None, :]) / $(3072)) % $(8)) * $(3072)) + (((((r463_group % $(8)) * $(3072)) + (r463_lane[:, None] * $(16)) + r463_col[None, :]) % $(16)) * $(192)) + (((((r463_group % $(8)) * $(3072)) + (r463_lane[:, None] * $(16)) + r463_col[None, :]) / $(16)) % $(192))), dtype=tl.float32)
      r463_value = tl.dot(r463_left, r463_right)
      r463_value = r463_value.to(tl.float32)
      tl.store($(candidate_vbmm_109_ptr) + ((r463_group * $(256)) + (r463_row[:, None] * $(16)) + r463_col[None, :]), r463_value)
    }
    r464_z = tl.arange($(2048))
    r464_value = (tl.load($(candidate_vbmm_109_ptr) + ((((r464_z / $(256)) % $(8)) * $(256)) + (((r464_z / $(16)) % $(16)) * $(16)) + (r464_z % $(16))), dtype=tl.float32) * 0.07216878364870322)
    r464_value = r464_value.to(tl.float32)
    r464_value = tl.broadcast_to(r464_value, [$(2048)])
    tl.store($(candidate_vmul_91_ptr) + r464_z, r464_value)
    r465_z = tl.arange($(16))
    r465_value = (tl.cast(r465_z, tl.int64) + tl.full([], $(16), dtype=tl.int64))
    r465_value = tl.broadcast_to(r465_value, [$(16)])
    tl.store($(candidate_vadd_57_ptr) + r465_z, r465_value)
    r466_z = tl.arange($(16))
    r466_value = (tl.load($(candidate_vadd_57_ptr) + r466_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r466_value = tl.broadcast_to(r466_value, [$(16)])
    tl.store($(candidate_vsub_59_ptr) + r466_z, r466_value)
    r467_z = tl.arange($(256))
    r467_value = (tl.cast((r467_z % $(16)), tl.int64) > tl.load($(candidate_vsub_59_ptr) + ((r467_z / $(16)) % $(16)), dtype=tl.int64))
    r467_value = tl.broadcast_to(r467_value, [$(256)])
    tl.store($(candidate_vgt_32_ptr) + r467_z, r467_value)
    r468_z = tl.arange($(2048))
    r468_value = tl.where(tl.load($(candidate_vgt_32_ptr) + ((((r468_z / $(16)) % $(16)) * $(16)) + (r468_z % $(16))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_91_ptr) + ((((r468_z / $(256)) % $(8)) * $(256)) + (((r468_z / $(16)) % $(16)) * $(16)) + (r468_z % $(16))), dtype=tl.float32))
    r468_value = r468_value.to(tl.float32)
    r468_value = tl.broadcast_to(r468_value, [$(2048)])
    tl.store($(candidate_vmasked_fill_32_ptr) + r468_z, r468_value)
    r469_z = tl.arange($(2048))
    r469_lane = tl.arange($(16))
    r469_row_values = tl.load($(candidate_vmasked_fill_32_ptr) + ((((r469_z[:, None] / $(256)) % $(8)) * $(256)) + (((r469_z[:, None] / $(16)) % $(16)) * $(16)) + (r469_lane[None, :] % $(16))), dtype=tl.float32)
    r469_maximum = tl.max(r469_row_values, axis=1, keep_dims=true)
    r469_weights = tl.exp(r469_row_values - r469_maximum)
    r469_total = tl.sum(r469_weights, axis=1)
    r469_maximum = tl.max(r469_row_values, axis=1)
    r469_value = tl.exp(tl.load($(candidate_vmasked_fill_32_ptr) + ((((r469_z / $(256)) % $(8)) * $(256)) + (((r469_z / $(16)) % $(16)) * $(16)) + (r469_z % $(16))), dtype=tl.float32) - r469_maximum) / r469_total
    r469_value = r469_value.to(tl.float32)
    r469_value = tl.broadcast_to(r469_value, [$(2048)])
    tl.store($(candidate_vsoftmax_23_ptr) + r469_z, r469_value)
    r470_z = tl.arange($(2048))
    r470_value = tl.load($(candidate_vsoftmax_23_ptr) + ((((r470_z / $(256)) % $(8)) * $(256)) + (((r470_z / $(16)) % $(16)) * $(16)) + (r470_z % $(16))), dtype=tl.float32)
    r470_value = r470_value.to(tl.bfloat16)
    r470_value = tl.broadcast_to(r470_value, [$(2048)])
    tl.store($(candidate_vto_copy_197_ptr) + r470_z, r470_value)
    r471_row = tl.arange($(16))
    r471_col = tl.arange($(128))
    r471_lane = tl.arange($(16))
    for r471_group in range($(8)) {
      r471_left = tl.load($(candidate_vto_copy_197_ptr) + (((((((r471_group % $(8)) * $(256)) + (r471_row[:, None] * $(16)) + r471_lane[None, :]) / $(256)) % $(8)) * $(256)) + ((((((r471_group % $(8)) * $(256)) + (r471_row[:, None] * $(16)) + r471_lane[None, :]) / $(16)) % $(16)) * $(16)) + ((((r471_group % $(8)) * $(256)) + (r471_row[:, None] * $(16)) + r471_lane[None, :]) % $(16))), dtype=tl.bfloat16)
      r471_right = tl.load($(candidate_vbmm_104_ptr) + (((((((r471_group % $(8)) * $(2048)) + (r471_lane[:, None] * $(128)) + r471_col[None, :]) / $(128)) % $(16)) * $(1024)) + ((((((r471_group % $(8)) * $(2048)) + (r471_lane[:, None] * $(128)) + r471_col[None, :]) / $(2048)) % $(8)) * $(128)) + ((((r471_group % $(8)) * $(2048)) + (r471_lane[:, None] * $(128)) + r471_col[None, :]) % $(128)) + $(65536)), dtype=tl.bfloat16)
      r471_value = tl.dot(r471_left, r471_right)
      r471_value = r471_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_110_ptr) + ((r471_group * $(2048)) + (r471_row[:, None] * $(128)) + r471_col[None, :]), r471_value)
    }
    r472_z = tl.arange($(128))
    r472_lane = tl.arange($(16))
    r472_row_values = tl.load($(candidate_vmasked_fill_32_ptr) + ((((r472_z[:, None] / $(16)) % $(8)) * $(256)) + ((r472_z[:, None] % $(16)) * $(16)) + (r472_lane[None, :] % $(16))), dtype=tl.float32)
    r472_maximum = tl.max(r472_row_values, axis=1, keep_dims=true)
    r472_weights = tl.exp(r472_row_values - r472_maximum)
    r472_total = tl.sum(r472_weights, axis=1)
    r472_maximum = tl.max(r472_row_values, axis=1)
    r472_value = r472_maximum + tl.log(r472_total)
    r472_value = r472_value.to(tl.float32)
    r472_value = tl.broadcast_to(r472_value, [$(128)])
    tl.store($(candidate_vlogsumexp_23_ptr) + r472_z, r472_value)
    r473_z = tl.arange($(32768))
    r473_value = tl.where(((r473_z / $(1024)) % $(32)) < $(12), tl.load($(candidate_vbmm_106_ptr) + ((((((((r473_z / $(128)) % $(8)) * $(1536)) + ((((r473_z / $(1024)) % $(32)) - $(0)) * $(128)) + (r473_z % $(128))) / $(1536)) % $(8)) * $(1536)) + ((((((r473_z / $(128)) % $(8)) * $(12)) + (((r473_z / $(1024)) % $(32)) - $(0))) % $(12)) * $(128)) + (r473_z % $(128))), dtype=tl.bfloat16), tl.where(((r473_z / $(1024)) % $(32)) < $(16), tl.load($(candidate_vbmm_108_ptr) + ((((((((r473_z / $(128)) % $(8)) * $(512)) + ((((r473_z / $(1024)) % $(32)) - $(12)) * $(128)) + (r473_z % $(128))) / $(512)) % $(8)) * $(512)) + ((((((r473_z / $(128)) % $(8)) * $(4)) + (((r473_z / $(1024)) % $(32)) - $(12))) % $(4)) * $(128)) + (r473_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vbmm_110_ptr) + ((((((((r473_z / $(128)) % $(8)) * $(2048)) + ((((r473_z / $(1024)) % $(32)) - $(16)) * $(128)) + (r473_z % $(128))) / $(2048)) % $(8)) * $(2048)) + ((((((r473_z / $(128)) % $(8)) * $(16)) + (((r473_z / $(1024)) % $(32)) - $(16))) % $(16)) * $(128)) + (r473_z % $(128))), dtype=tl.bfloat16)))
    r473_value = r473_value.to(tl.bfloat16)
    r473_value = tl.broadcast_to(r473_value, [$(32768)])
    tl.store($(candidate_vcat_49_ptr) + r473_z, r473_value)
    r474_z = tl.arange($(256))
    r474_value = tl.where(((r474_z / $(8)) % $(32)) < $(12), tl.load($(candidate_vlogsumexp_21_ptr) + (((r474_z % $(8)) * $(12)) + ((((r474_z / $(8)) % $(32)) - $(0)) % $(12))), dtype=tl.float32), tl.where(((r474_z / $(8)) % $(32)) < $(16), tl.load($(candidate_vlogsumexp_22_ptr) + (((r474_z % $(8)) * $(4)) + ((((r474_z / $(8)) % $(32)) - $(12)) % $(4))), dtype=tl.float32), tl.load($(candidate_vlogsumexp_23_ptr) + (((r474_z % $(8)) * $(16)) + ((((r474_z / $(8)) % $(32)) - $(16)) % $(16))), dtype=tl.float32)))
    r474_value = r474_value.to(tl.float32)
    r474_value = tl.broadcast_to(r474_value, [$(256)])
    tl.store($(candidate_vcat_50_ptr) + r474_z, r474_value)
    r475_z = tl.arange($(18432))
    r475_value = tl.load($(candidate_vcat_47_ptr) + ((((r475_z / $(192)) % $(12)) * $(1536)) + (((r475_z / $(2304)) % $(8)) * $(192)) + (r475_z % $(192))), dtype=tl.bfloat16)
    r475_value = r475_value.to(tl.float32)
    r475_value = tl.broadcast_to(r475_value, [$(18432)])
    tl.store($(candidate_vto_copy_198_ptr) + r475_z, r475_value)
    r476_z = tl.arange($(18432))
    r476_value = tl.load($(candidate_vcat_48_ptr) + ((((r476_z / $(192)) % $(12)) * $(1536)) + (((r476_z / $(2304)) % $(8)) * $(192)) + (r476_z % $(192))), dtype=tl.bfloat16)
    r476_value = r476_value.to(tl.float32)
    r476_value = tl.broadcast_to(r476_value, [$(18432)])
    tl.store($(candidate_vto_copy_199_ptr) + r476_z, r476_value)
    r477_z = tl.arange($(12288))
    r477_value = tl.load($(candidate_vbmm_104_ptr) + ((((r477_z / $(128)) % $(12)) * $(1024)) + (((r477_z / $(1536)) % $(8)) * $(128)) + (r477_z % $(128))), dtype=tl.bfloat16)
    r477_value = r477_value.to(tl.float32)
    r477_value = tl.broadcast_to(r477_value, [$(12288)])
    tl.store($(candidate_vto_copy_200_ptr) + r477_z, r477_value)
    r478_z = tl.arange($(12288))
    r478_value = tl.load($(candidate_vmm_28_ptr) + ((((r478_z / $(128)) % $(12)) * $(1024)) + (((r478_z / $(1536)) % $(8)) * $(128)) + (r478_z % $(128))), dtype=tl.bfloat16)
    r478_value = r478_value.to(tl.float32)
    r478_value = tl.broadcast_to(r478_value, [$(12288)])
    tl.store($(candidate_vto_copy_201_ptr) + r478_z, r478_value)
    r479_z = tl.arange($(12288))
    r479_value = tl.load($(candidate_vcat_49_ptr) + ((((r479_z / $(128)) % $(12)) * $(1024)) + (((r479_z / $(1536)) % $(8)) * $(128)) + (r479_z % $(128))), dtype=tl.bfloat16)
    r479_value = r479_value.to(tl.float32)
    r479_value = tl.broadcast_to(r479_value, [$(12288)])
    tl.store($(candidate_vto_copy_202_ptr) + r479_z, r479_value)
    r480_row = tl.arange($(12))
    r480_col = tl.arange($(12))
    r480_lane = tl.arange($(192))
    for r480_group in range($(8)) {
      r480_left = tl.load($(candidate_vto_copy_198_ptr) + (((((((r480_group % $(8)) * $(2304)) + (r480_row[:, None] * $(192)) + r480_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r480_group % $(8)) * $(2304)) + (r480_row[:, None] * $(192)) + r480_lane[None, :]) / $(192)) % $(12)) * $(192)) + ((((r480_group % $(8)) * $(2304)) + (r480_row[:, None] * $(192)) + r480_lane[None, :]) % $(192))), dtype=tl.float32)
      r480_right = tl.load($(candidate_vto_copy_199_ptr) + (((((((r480_group % $(8)) * $(2304)) + (r480_lane[:, None] * $(12)) + r480_col[None, :]) / $(2304)) % $(8)) * $(2304)) + (((((r480_group % $(8)) * $(2304)) + (r480_lane[:, None] * $(12)) + r480_col[None, :]) % $(12)) * $(192)) + (((((r480_group % $(8)) * $(2304)) + (r480_lane[:, None] * $(12)) + r480_col[None, :]) / $(12)) % $(192))), dtype=tl.float32)
      r480_value = tl.dot(r480_left, r480_right)
      r480_value = r480_value.to(tl.float32)
      tl.store($(candidate_vbmm_111_ptr) + ((r480_group * $(144)) + (r480_row[:, None] * $(12)) + r480_col[None, :]), r480_value)
    }
    r481_z = tl.arange($(1152))
    r481_value = (tl.load($(candidate_vbmm_111_ptr) + ((((r481_z / $(144)) % $(8)) * $(144)) + (((r481_z / $(12)) % $(12)) * $(12)) + (r481_z % $(12))), dtype=tl.float32) * 0.07216878364870322)
    r481_value = r481_value.to(tl.float32)
    r481_value = tl.broadcast_to(r481_value, [$(1152)])
    tl.store($(candidate_vmul_92_ptr) + r481_z, r481_value)
    r482_z = tl.arange($(12))
    r482_value = (tl.cast(r482_z, tl.int64) + tl.full([], $(12), dtype=tl.int64))
    r482_value = tl.broadcast_to(r482_value, [$(12)])
    tl.store($(candidate_vadd_58_ptr) + r482_z, r482_value)
    r483_z = tl.arange($(12))
    r483_value = (tl.load($(candidate_vadd_58_ptr) + r483_z, dtype=tl.int64) - tl.full([], $(12), dtype=tl.int64))
    r483_value = tl.broadcast_to(r483_value, [$(12)])
    tl.store($(candidate_vsub_60_ptr) + r483_z, r483_value)
    r484_z = tl.arange($(144))
    r484_value = (tl.cast((r484_z % $(12)), tl.int64) > tl.load($(candidate_vsub_60_ptr) + ((r484_z / $(12)) % $(12)), dtype=tl.int64))
    r484_value = tl.broadcast_to(r484_value, [$(144)])
    tl.store($(candidate_vgt_33_ptr) + r484_z, r484_value)
    r485_z = tl.arange($(1152))
    r485_value = tl.where(tl.load($(candidate_vgt_33_ptr) + ((((r485_z / $(12)) % $(12)) * $(12)) + (r485_z % $(12))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_92_ptr) + ((((r485_z / $(144)) % $(8)) * $(144)) + (((r485_z / $(12)) % $(12)) * $(12)) + (r485_z % $(12))), dtype=tl.float32))
    r485_value = r485_value.to(tl.float32)
    r485_value = tl.broadcast_to(r485_value, [$(1152)])
    tl.store($(candidate_vmasked_fill_33_ptr) + r485_z, r485_value)
    r486_z = tl.arange($(1152))
    r486_value = (tl.load($(candidate_vmasked_fill_33_ptr) + ((((r486_z / $(144)) % $(8)) * $(144)) + (((r486_z / $(12)) % $(12)) * $(12)) + (r486_z % $(12))), dtype=tl.float32) - tl.load($(candidate_vcat_50_ptr) + ((((r486_z / $(12)) % $(12)) * $(8)) + ((r486_z / $(144)) % $(8))), dtype=tl.float32))
    r486_value = r486_value.to(tl.float32)
    r486_value = tl.broadcast_to(r486_value, [$(1152)])
    tl.store($(candidate_vsub_61_ptr) + r486_z, r486_value)
    r487_z = tl.arange($(1152))
    r487_value = tl.exp(tl.load($(candidate_vsub_61_ptr) + ((((r487_z / $(144)) % $(8)) * $(144)) + (((r487_z / $(12)) % $(12)) * $(12)) + (r487_z % $(12))), dtype=tl.float32))
    r487_value = r487_value.to(tl.float32)
    r487_value = tl.broadcast_to(r487_value, [$(1152)])
    tl.store($(candidate_vexp_9_ptr) + r487_z, r487_value)
    r488_z = tl.arange($(12288))
    r488_value = (tl.load($(candidate_vto_copy_201_ptr) + ((((r488_z / $(1536)) % $(8)) * $(1536)) + (((r488_z / $(128)) % $(12)) * $(128)) + (r488_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_202_ptr) + ((((r488_z / $(1536)) % $(8)) * $(1536)) + (((r488_z / $(128)) % $(12)) * $(128)) + (r488_z % $(128))), dtype=tl.float32))
    r488_value = r488_value.to(tl.float32)
    r488_value = tl.broadcast_to(r488_value, [$(12288)])
    tl.store($(candidate_vmul_93_ptr) + r488_z, r488_value)
    r489_z = tl.arange($(96))
    r489_lane = tl.arange($(128))
    r489_value = tl.sum(tl.load($(candidate_vmul_93_ptr) + ((((r489_z[:, None] / $(12)) % $(8)) * $(1536)) + ((r489_z[:, None] % $(12)) * $(128)) + r489_lane[None, :]), dtype=tl.float32), axis=1)
    r489_value = r489_value.to(tl.float32)
    r489_value = tl.broadcast_to(r489_value, [$(96)])
    tl.store($(candidate_vsum_13_ptr) + r489_z, r489_value)
    r490_row = tl.arange($(12))
    r490_col = tl.arange($(12))
    r490_lane = tl.arange($(128))
    for r490_group in range($(8)) {
      r490_left = tl.load($(candidate_vto_copy_201_ptr) + (((((((r490_group % $(8)) * $(1536)) + (r490_row[:, None] * $(128)) + r490_lane[None, :]) / $(1536)) % $(8)) * $(1536)) + ((((((r490_group % $(8)) * $(1536)) + (r490_row[:, None] * $(128)) + r490_lane[None, :]) / $(128)) % $(12)) * $(128)) + ((((r490_group % $(8)) * $(1536)) + (r490_row[:, None] * $(128)) + r490_lane[None, :]) % $(128))), dtype=tl.float32)
      r490_right = tl.load($(candidate_vto_copy_200_ptr) + (((((((r490_group % $(8)) * $(1536)) + (r490_lane[:, None] * $(12)) + r490_col[None, :]) / $(1536)) % $(8)) * $(1536)) + (((((r490_group % $(8)) * $(1536)) + (r490_lane[:, None] * $(12)) + r490_col[None, :]) % $(12)) * $(128)) + (((((r490_group % $(8)) * $(1536)) + (r490_lane[:, None] * $(12)) + r490_col[None, :]) / $(12)) % $(128))), dtype=tl.float32)
      r490_value = tl.dot(r490_left, r490_right)
      r490_value = r490_value.to(tl.float32)
      tl.store($(candidate_vbmm_112_ptr) + ((r490_group * $(144)) + (r490_row[:, None] * $(12)) + r490_col[None, :]), r490_value)
    }
    r491_z = tl.arange($(1152))
    r491_value = (tl.load($(candidate_vbmm_112_ptr) + ((((r491_z / $(144)) % $(8)) * $(144)) + (((r491_z / $(12)) % $(12)) * $(12)) + (r491_z % $(12))), dtype=tl.float32) - tl.load($(candidate_vsum_13_ptr) + ((((r491_z / $(144)) % $(8)) * $(12)) + ((r491_z / $(12)) % $(12))), dtype=tl.float32))
    r491_value = r491_value.to(tl.float32)
    r491_value = tl.broadcast_to(r491_value, [$(1152)])
    tl.store($(candidate_vsub_62_ptr) + r491_z, r491_value)
    r492_z = tl.arange($(1152))
    r492_value = (tl.load($(candidate_vexp_9_ptr) + ((((r492_z / $(144)) % $(8)) * $(144)) + (((r492_z / $(12)) % $(12)) * $(12)) + (r492_z % $(12))), dtype=tl.float32) * tl.load($(candidate_vsub_62_ptr) + ((((r492_z / $(144)) % $(8)) * $(144)) + (((r492_z / $(12)) % $(12)) * $(12)) + (r492_z % $(12))), dtype=tl.float32))
    r492_value = r492_value.to(tl.float32)
    r492_value = tl.broadcast_to(r492_value, [$(1152)])
    tl.store($(candidate_vmul_94_ptr) + r492_z, r492_value)
    r493_row = tl.arange($(12))
    r493_col = tl.arange($(192))
    r493_lane = tl.arange($(12))
    for r493_group in range($(8)) {
      r493_left = tl.load($(candidate_vmul_94_ptr) + (((((((r493_group % $(8)) * $(144)) + (r493_row[:, None] * $(12)) + r493_lane[None, :]) / $(144)) % $(8)) * $(144)) + (((((r493_group % $(8)) * $(144)) + (r493_row[:, None] * $(12)) + r493_lane[None, :]) % $(12)) * $(12)) + (((((r493_group % $(8)) * $(144)) + (r493_row[:, None] * $(12)) + r493_lane[None, :]) / $(12)) % $(12))), dtype=tl.float32)
      r493_right = tl.load($(candidate_vto_copy_198_ptr) + (((((((r493_group % $(8)) * $(2304)) + (r493_lane[:, None] * $(192)) + r493_col[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r493_group % $(8)) * $(2304)) + (r493_lane[:, None] * $(192)) + r493_col[None, :]) / $(192)) % $(12)) * $(192)) + ((((r493_group % $(8)) * $(2304)) + (r493_lane[:, None] * $(192)) + r493_col[None, :]) % $(192))), dtype=tl.float32)
      r493_value = tl.dot(r493_left, r493_right)
      r493_value = r493_value.to(tl.float32)
      tl.store($(candidate_vbmm_114_ptr) + ((r493_group * $(2304)) + (r493_row[:, None] * $(192)) + r493_col[None, :]), r493_value)
    }
    r494_z = tl.arange($(18432))
    r494_value = (tl.load($(candidate_vbmm_114_ptr) + ((((r494_z / $(2304)) % $(8)) * $(2304)) + (((r494_z / $(192)) % $(12)) * $(192)) + (r494_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r494_value = r494_value.to(tl.float32)
    r494_value = tl.broadcast_to(r494_value, [$(18432)])
    tl.store($(candidate_vmul_96_ptr) + r494_z, r494_value)
    r495_z = tl.arange($(18432))
    r495_value = tl.load($(candidate_vmul_96_ptr) + ((((r495_z / $(192)) % $(8)) * $(2304)) + (((r495_z / $(1536)) % $(12)) * $(192)) + (r495_z % $(192))), dtype=tl.float32)
    r495_value = r495_value.to(tl.bfloat16)
    r495_value = tl.broadcast_to(r495_value, [$(18432)])
    tl.store($(candidate_vto_copy_204_ptr) + r495_z, r495_value)
    r496_row = tl.arange($(12))
    r496_col = tl.arange($(128))
    r496_lane = tl.arange($(12))
    for r496_group in range($(8)) {
      r496_left = tl.load($(candidate_vexp_9_ptr) + (((((((r496_group % $(8)) * $(144)) + (r496_row[:, None] * $(12)) + r496_lane[None, :]) / $(144)) % $(8)) * $(144)) + (((((r496_group % $(8)) * $(144)) + (r496_row[:, None] * $(12)) + r496_lane[None, :]) % $(12)) * $(12)) + (((((r496_group % $(8)) * $(144)) + (r496_row[:, None] * $(12)) + r496_lane[None, :]) / $(12)) % $(12))), dtype=tl.float32)
      r496_right = tl.load($(candidate_vto_copy_201_ptr) + (((((((r496_group % $(8)) * $(1536)) + (r496_lane[:, None] * $(128)) + r496_col[None, :]) / $(1536)) % $(8)) * $(1536)) + ((((((r496_group % $(8)) * $(1536)) + (r496_lane[:, None] * $(128)) + r496_col[None, :]) / $(128)) % $(12)) * $(128)) + ((((r496_group % $(8)) * $(1536)) + (r496_lane[:, None] * $(128)) + r496_col[None, :]) % $(128))), dtype=tl.float32)
      r496_value = tl.dot(r496_left, r496_right)
      r496_value = r496_value.to(tl.float32)
      tl.store($(candidate_vbmm_115_ptr) + ((r496_group * $(1536)) + (r496_row[:, None] * $(128)) + r496_col[None, :]), r496_value)
    }
    r497_z = tl.arange($(12288))
    r497_value = tl.load($(candidate_vbmm_115_ptr) + ((((r497_z / $(128)) % $(8)) * $(1536)) + (((r497_z / $(1024)) % $(12)) * $(128)) + (r497_z % $(128))), dtype=tl.float32)
    r497_value = r497_value.to(tl.bfloat16)
    r497_value = tl.broadcast_to(r497_value, [$(12288)])
    tl.store($(candidate_vto_copy_205_ptr) + r497_z, r497_value)
    r498_z = tl.arange($(73728))
    r498_value = tl.where(((r498_z / $(1536)) % $(48)) < $(12), tl.load($(candidate_vto_copy_204_ptr) + ((((((r498_z / $(1536)) % $(48)) - $(0)) % $(12)) * $(1536)) + (((r498_z / $(192)) % $(8)) * $(192)) + (r498_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r498_value = r498_value.to(tl.bfloat16)
    r498_value = tl.broadcast_to(r498_value, [$(73728)])
    tl.store($(candidate_vcat_51_ptr) + r498_z, r498_value)
    r499_z = tl.arange($(49152))
    r499_value = tl.where(((r499_z / $(1024)) % $(48)) < $(12), tl.load($(candidate_vto_copy_205_ptr) + ((((((r499_z / $(1024)) % $(48)) - $(0)) % $(12)) * $(1024)) + (((r499_z / $(128)) % $(8)) * $(128)) + (r499_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r499_value = r499_value.to(tl.bfloat16)
    r499_value = tl.broadcast_to(r499_value, [$(49152)])
    tl.store($(candidate_vcat_52_ptr) + r499_z, r499_value)
    r500_z = tl.arange($(6144))
    r500_value = tl.load($(candidate_vcat_47_ptr) + ((((r500_z / $(192)) % $(4)) * $(1536)) + (((r500_z / $(768)) % $(8)) * $(192)) + (r500_z % $(192)) + $(18432)), dtype=tl.bfloat16)
    r500_value = r500_value.to(tl.float32)
    r500_value = tl.broadcast_to(r500_value, [$(6144)])
    tl.store($(candidate_vto_copy_206_ptr) + r500_z, r500_value)
    r501_z = tl.arange($(6144))
    r501_value = tl.load($(candidate_vcat_48_ptr) + ((((r501_z / $(192)) % $(4)) * $(1536)) + (((r501_z / $(768)) % $(8)) * $(192)) + (r501_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r501_value = r501_value.to(tl.float32)
    r501_value = tl.broadcast_to(r501_value, [$(6144)])
    tl.store($(candidate_vto_copy_207_ptr) + r501_z, r501_value)
    r502_z = tl.arange($(4096))
    r502_value = tl.load($(candidate_vbmm_104_ptr) + ((((r502_z / $(128)) % $(4)) * $(1024)) + (((r502_z / $(512)) % $(8)) * $(128)) + (r502_z % $(128)) + $(49152)), dtype=tl.bfloat16)
    r502_value = r502_value.to(tl.float32)
    r502_value = tl.broadcast_to(r502_value, [$(4096)])
    tl.store($(candidate_vto_copy_208_ptr) + r502_z, r502_value)
    r503_z = tl.arange($(4096))
    r503_value = tl.load($(candidate_vmm_28_ptr) + ((((r503_z / $(128)) % $(4)) * $(1024)) + (((r503_z / $(512)) % $(8)) * $(128)) + (r503_z % $(128)) + $(12288)), dtype=tl.bfloat16)
    r503_value = r503_value.to(tl.float32)
    r503_value = tl.broadcast_to(r503_value, [$(4096)])
    tl.store($(candidate_vto_copy_209_ptr) + r503_z, r503_value)
    r504_z = tl.arange($(4096))
    r504_value = tl.load($(candidate_vcat_49_ptr) + ((((r504_z / $(128)) % $(4)) * $(1024)) + (((r504_z / $(512)) % $(8)) * $(128)) + (r504_z % $(128)) + $(12288)), dtype=tl.bfloat16)
    r504_value = r504_value.to(tl.float32)
    r504_value = tl.broadcast_to(r504_value, [$(4096)])
    tl.store($(candidate_vto_copy_210_ptr) + r504_z, r504_value)
    r505_row = tl.arange($(4))
    r505_col = tl.arange($(4))
    r505_lane = tl.arange($(192))
    for r505_group in range($(8)) {
      r505_left = tl.load($(candidate_vto_copy_206_ptr) + (((((((r505_group % $(8)) * $(768)) + (r505_row[:, None] * $(192)) + r505_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r505_group % $(8)) * $(768)) + (r505_row[:, None] * $(192)) + r505_lane[None, :]) / $(192)) % $(4)) * $(192)) + ((((r505_group % $(8)) * $(768)) + (r505_row[:, None] * $(192)) + r505_lane[None, :]) % $(192))), dtype=tl.float32)
      r505_right = tl.load($(candidate_vto_copy_207_ptr) + (((((((r505_group % $(8)) * $(768)) + (r505_lane[:, None] * $(4)) + r505_col[None, :]) / $(768)) % $(8)) * $(768)) + (((((r505_group % $(8)) * $(768)) + (r505_lane[:, None] * $(4)) + r505_col[None, :]) % $(4)) * $(192)) + (((((r505_group % $(8)) * $(768)) + (r505_lane[:, None] * $(4)) + r505_col[None, :]) / $(4)) % $(192))), dtype=tl.float32)
      r505_value = tl.dot(r505_left, r505_right)
      r505_value = r505_value.to(tl.float32)
      tl.store($(candidate_vbmm_116_ptr) + ((r505_group * $(16)) + (r505_row[:, None] * $(4)) + r505_col[None, :]), r505_value)
    }
    r506_z = tl.arange($(128))
    r506_value = (tl.load($(candidate_vbmm_116_ptr) + ((((r506_z / $(16)) % $(8)) * $(16)) + (((r506_z / $(4)) % $(4)) * $(4)) + (r506_z % $(4))), dtype=tl.float32) * 0.07216878364870322)
    r506_value = r506_value.to(tl.float32)
    r506_value = tl.broadcast_to(r506_value, [$(128)])
    tl.store($(candidate_vmul_97_ptr) + r506_z, r506_value)
    r507_z = tl.arange($(4))
    r507_value = (tl.cast(r507_z, tl.int64) + tl.full([], $(4), dtype=tl.int64))
    r507_value = tl.broadcast_to(r507_value, [$(4)])
    tl.store($(candidate_vadd_59_ptr) + r507_z, r507_value)
    r508_z = tl.arange($(4))
    r508_value = (tl.load($(candidate_vadd_59_ptr) + r508_z, dtype=tl.int64) - tl.full([], $(4), dtype=tl.int64))
    r508_value = tl.broadcast_to(r508_value, [$(4)])
    tl.store($(candidate_vsub_63_ptr) + r508_z, r508_value)
    r509_z = tl.arange($(16))
    r509_value = (tl.cast((r509_z % $(4)), tl.int64) > tl.load($(candidate_vsub_63_ptr) + ((r509_z / $(4)) % $(4)), dtype=tl.int64))
    r509_value = tl.broadcast_to(r509_value, [$(16)])
    tl.store($(candidate_vgt_34_ptr) + r509_z, r509_value)
    r510_z = tl.arange($(128))
    r510_value = tl.where(tl.load($(candidate_vgt_34_ptr) + ((((r510_z / $(4)) % $(4)) * $(4)) + (r510_z % $(4))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_97_ptr) + ((((r510_z / $(16)) % $(8)) * $(16)) + (((r510_z / $(4)) % $(4)) * $(4)) + (r510_z % $(4))), dtype=tl.float32))
    r510_value = r510_value.to(tl.float32)
    r510_value = tl.broadcast_to(r510_value, [$(128)])
    tl.store($(candidate_vmasked_fill_34_ptr) + r510_z, r510_value)
    r511_z = tl.arange($(128))
    r511_value = (tl.load($(candidate_vmasked_fill_34_ptr) + ((((r511_z / $(16)) % $(8)) * $(16)) + (((r511_z / $(4)) % $(4)) * $(4)) + (r511_z % $(4))), dtype=tl.float32) - tl.load($(candidate_vcat_50_ptr) + ((((r511_z / $(4)) % $(4)) * $(8)) + ((r511_z / $(16)) % $(8)) + $(96)), dtype=tl.float32))
    r511_value = r511_value.to(tl.float32)
    r511_value = tl.broadcast_to(r511_value, [$(128)])
    tl.store($(candidate_vsub_64_ptr) + r511_z, r511_value)
    r512_z = tl.arange($(128))
    r512_value = tl.exp(tl.load($(candidate_vsub_64_ptr) + ((((r512_z / $(16)) % $(8)) * $(16)) + (((r512_z / $(4)) % $(4)) * $(4)) + (r512_z % $(4))), dtype=tl.float32))
    r512_value = r512_value.to(tl.float32)
    r512_value = tl.broadcast_to(r512_value, [$(128)])
    tl.store($(candidate_vexp_10_ptr) + r512_z, r512_value)
    r513_z = tl.arange($(4096))
    r513_value = (tl.load($(candidate_vto_copy_209_ptr) + ((((r513_z / $(512)) % $(8)) * $(512)) + (((r513_z / $(128)) % $(4)) * $(128)) + (r513_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_210_ptr) + ((((r513_z / $(512)) % $(8)) * $(512)) + (((r513_z / $(128)) % $(4)) * $(128)) + (r513_z % $(128))), dtype=tl.float32))
    r513_value = r513_value.to(tl.float32)
    r513_value = tl.broadcast_to(r513_value, [$(4096)])
    tl.store($(candidate_vmul_98_ptr) + r513_z, r513_value)
    r514_z = tl.arange($(32))
    r514_lane = tl.arange($(128))
    r514_value = tl.sum(tl.load($(candidate_vmul_98_ptr) + ((((r514_z[:, None] / $(4)) % $(8)) * $(512)) + ((r514_z[:, None] % $(4)) * $(128)) + r514_lane[None, :]), dtype=tl.float32), axis=1)
    r514_value = r514_value.to(tl.float32)
    r514_value = tl.broadcast_to(r514_value, [$(32)])
    tl.store($(candidate_vsum_14_ptr) + r514_z, r514_value)
    r515_row = tl.arange($(4))
    r515_col = tl.arange($(4))
    r515_lane = tl.arange($(128))
    for r515_group in range($(8)) {
      r515_left = tl.load($(candidate_vto_copy_209_ptr) + (((((((r515_group % $(8)) * $(512)) + (r515_row[:, None] * $(128)) + r515_lane[None, :]) / $(512)) % $(8)) * $(512)) + ((((((r515_group % $(8)) * $(512)) + (r515_row[:, None] * $(128)) + r515_lane[None, :]) / $(128)) % $(4)) * $(128)) + ((((r515_group % $(8)) * $(512)) + (r515_row[:, None] * $(128)) + r515_lane[None, :]) % $(128))), dtype=tl.float32)
      r515_right = tl.load($(candidate_vto_copy_208_ptr) + (((((((r515_group % $(8)) * $(512)) + (r515_lane[:, None] * $(4)) + r515_col[None, :]) / $(512)) % $(8)) * $(512)) + (((((r515_group % $(8)) * $(512)) + (r515_lane[:, None] * $(4)) + r515_col[None, :]) % $(4)) * $(128)) + (((((r515_group % $(8)) * $(512)) + (r515_lane[:, None] * $(4)) + r515_col[None, :]) / $(4)) % $(128))), dtype=tl.float32)
      r515_value = tl.dot(r515_left, r515_right)
      r515_value = r515_value.to(tl.float32)
      tl.store($(candidate_vbmm_117_ptr) + ((r515_group * $(16)) + (r515_row[:, None] * $(4)) + r515_col[None, :]), r515_value)
    }
    r516_z = tl.arange($(128))
    r516_value = (tl.load($(candidate_vbmm_117_ptr) + ((((r516_z / $(16)) % $(8)) * $(16)) + (((r516_z / $(4)) % $(4)) * $(4)) + (r516_z % $(4))), dtype=tl.float32) - tl.load($(candidate_vsum_14_ptr) + ((((r516_z / $(16)) % $(8)) * $(4)) + ((r516_z / $(4)) % $(4))), dtype=tl.float32))
    r516_value = r516_value.to(tl.float32)
    r516_value = tl.broadcast_to(r516_value, [$(128)])
    tl.store($(candidate_vsub_65_ptr) + r516_z, r516_value)
    r517_z = tl.arange($(128))
    r517_value = (tl.load($(candidate_vexp_10_ptr) + ((((r517_z / $(16)) % $(8)) * $(16)) + (((r517_z / $(4)) % $(4)) * $(4)) + (r517_z % $(4))), dtype=tl.float32) * tl.load($(candidate_vsub_65_ptr) + ((((r517_z / $(16)) % $(8)) * $(16)) + (((r517_z / $(4)) % $(4)) * $(4)) + (r517_z % $(4))), dtype=tl.float32))
    r517_value = r517_value.to(tl.float32)
    r517_value = tl.broadcast_to(r517_value, [$(128)])
    tl.store($(candidate_vmul_99_ptr) + r517_z, r517_value)
    r518_row = tl.arange($(4))
    r518_col = tl.arange($(192))
    r518_lane = tl.arange($(4))
    for r518_group in range($(8)) {
      r518_left = tl.load($(candidate_vmul_99_ptr) + (((((((r518_group % $(8)) * $(16)) + (r518_row[:, None] * $(4)) + r518_lane[None, :]) / $(16)) % $(8)) * $(16)) + (((((r518_group % $(8)) * $(16)) + (r518_row[:, None] * $(4)) + r518_lane[None, :]) % $(4)) * $(4)) + (((((r518_group % $(8)) * $(16)) + (r518_row[:, None] * $(4)) + r518_lane[None, :]) / $(4)) % $(4))), dtype=tl.float32)
      r518_right = tl.load($(candidate_vto_copy_206_ptr) + (((((((r518_group % $(8)) * $(768)) + (r518_lane[:, None] * $(192)) + r518_col[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r518_group % $(8)) * $(768)) + (r518_lane[:, None] * $(192)) + r518_col[None, :]) / $(192)) % $(4)) * $(192)) + ((((r518_group % $(8)) * $(768)) + (r518_lane[:, None] * $(192)) + r518_col[None, :]) % $(192))), dtype=tl.float32)
      r518_value = tl.dot(r518_left, r518_right)
      r518_value = r518_value.to(tl.float32)
      tl.store($(candidate_vbmm_119_ptr) + ((r518_group * $(768)) + (r518_row[:, None] * $(192)) + r518_col[None, :]), r518_value)
    }
    r519_z = tl.arange($(6144))
    r519_value = (tl.load($(candidate_vbmm_119_ptr) + ((((r519_z / $(768)) % $(8)) * $(768)) + (((r519_z / $(192)) % $(4)) * $(192)) + (r519_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r519_value = r519_value.to(tl.float32)
    r519_value = tl.broadcast_to(r519_value, [$(6144)])
    tl.store($(candidate_vmul_101_ptr) + r519_z, r519_value)
    r520_z = tl.arange($(6144))
    r520_value = tl.load($(candidate_vmul_101_ptr) + ((((r520_z / $(192)) % $(8)) * $(768)) + (((r520_z / $(1536)) % $(4)) * $(192)) + (r520_z % $(192))), dtype=tl.float32)
    r520_value = r520_value.to(tl.bfloat16)
    r520_value = tl.broadcast_to(r520_value, [$(6144)])
    tl.store($(candidate_vto_copy_212_ptr) + r520_z, r520_value)
    r521_row = tl.arange($(4))
    r521_col = tl.arange($(128))
    r521_lane = tl.arange($(4))
    for r521_group in range($(8)) {
      r521_left = tl.load($(candidate_vexp_10_ptr) + (((((((r521_group % $(8)) * $(16)) + (r521_row[:, None] * $(4)) + r521_lane[None, :]) / $(16)) % $(8)) * $(16)) + (((((r521_group % $(8)) * $(16)) + (r521_row[:, None] * $(4)) + r521_lane[None, :]) % $(4)) * $(4)) + (((((r521_group % $(8)) * $(16)) + (r521_row[:, None] * $(4)) + r521_lane[None, :]) / $(4)) % $(4))), dtype=tl.float32)
      r521_right = tl.load($(candidate_vto_copy_209_ptr) + (((((((r521_group % $(8)) * $(512)) + (r521_lane[:, None] * $(128)) + r521_col[None, :]) / $(512)) % $(8)) * $(512)) + ((((((r521_group % $(8)) * $(512)) + (r521_lane[:, None] * $(128)) + r521_col[None, :]) / $(128)) % $(4)) * $(128)) + ((((r521_group % $(8)) * $(512)) + (r521_lane[:, None] * $(128)) + r521_col[None, :]) % $(128))), dtype=tl.float32)
      r521_value = tl.dot(r521_left, r521_right)
      r521_value = r521_value.to(tl.float32)
      tl.store($(candidate_vbmm_120_ptr) + ((r521_group * $(512)) + (r521_row[:, None] * $(128)) + r521_col[None, :]), r521_value)
    }
    r522_z = tl.arange($(4096))
    r522_value = tl.load($(candidate_vbmm_120_ptr) + ((((r522_z / $(128)) % $(8)) * $(512)) + (((r522_z / $(1024)) % $(4)) * $(128)) + (r522_z % $(128))), dtype=tl.float32)
    r522_value = r522_value.to(tl.bfloat16)
    r522_value = tl.broadcast_to(r522_value, [$(4096)])
    tl.store($(candidate_vto_copy_213_ptr) + r522_z, r522_value)
    r523_z = tl.arange($(24576))
    r523_value = tl.where(((r523_z / $(1536)) % $(16)) < $(4), tl.load($(candidate_vto_copy_212_ptr) + ((((((r523_z / $(1536)) % $(16)) - $(0)) % $(4)) * $(1536)) + (((r523_z / $(192)) % $(8)) * $(192)) + (r523_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r523_value = r523_value.to(tl.bfloat16)
    r523_value = tl.broadcast_to(r523_value, [$(24576)])
    tl.store($(candidate_vcat_53_ptr) + r523_z, r523_value)
    r524_z = tl.arange($(16384))
    r524_value = tl.where(((r524_z / $(1024)) % $(16)) < $(4), tl.load($(candidate_vto_copy_213_ptr) + ((((((r524_z / $(1024)) % $(16)) - $(0)) % $(4)) * $(1024)) + (((r524_z / $(128)) % $(8)) * $(128)) + (r524_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r524_value = r524_value.to(tl.bfloat16)
    r524_value = tl.broadcast_to(r524_value, [$(16384)])
    tl.store($(candidate_vcat_54_ptr) + r524_z, r524_value)
    r525_z = tl.arange($(24576))
    r525_value = tl.load($(candidate_vcat_47_ptr) + ((((r525_z / $(192)) % $(16)) * $(1536)) + (((r525_z / $(3072)) % $(8)) * $(192)) + (r525_z % $(192)) + $(24576)), dtype=tl.bfloat16)
    r525_value = r525_value.to(tl.float32)
    r525_value = tl.broadcast_to(r525_value, [$(24576)])
    tl.store($(candidate_vto_copy_214_ptr) + r525_z, r525_value)
    r526_z = tl.arange($(24576))
    r526_value = tl.load($(candidate_vcat_48_ptr) + ((((r526_z / $(192)) % $(16)) * $(1536)) + (((r526_z / $(3072)) % $(8)) * $(192)) + (r526_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r526_value = r526_value.to(tl.float32)
    r526_value = tl.broadcast_to(r526_value, [$(24576)])
    tl.store($(candidate_vto_copy_215_ptr) + r526_z, r526_value)
    r527_z = tl.arange($(16384))
    r527_value = tl.load($(candidate_vbmm_104_ptr) + ((((r527_z / $(128)) % $(16)) * $(1024)) + (((r527_z / $(2048)) % $(8)) * $(128)) + (r527_z % $(128)) + $(65536)), dtype=tl.bfloat16)
    r527_value = r527_value.to(tl.float32)
    r527_value = tl.broadcast_to(r527_value, [$(16384)])
    tl.store($(candidate_vto_copy_216_ptr) + r527_z, r527_value)
    r528_z = tl.arange($(16384))
    r528_value = tl.load($(candidate_vmm_28_ptr) + ((((r528_z / $(128)) % $(16)) * $(1024)) + (((r528_z / $(2048)) % $(8)) * $(128)) + (r528_z % $(128)) + $(16384)), dtype=tl.bfloat16)
    r528_value = r528_value.to(tl.float32)
    r528_value = tl.broadcast_to(r528_value, [$(16384)])
    tl.store($(candidate_vto_copy_217_ptr) + r528_z, r528_value)
    r529_z = tl.arange($(16384))
    r529_value = tl.load($(candidate_vcat_49_ptr) + ((((r529_z / $(128)) % $(16)) * $(1024)) + (((r529_z / $(2048)) % $(8)) * $(128)) + (r529_z % $(128)) + $(16384)), dtype=tl.bfloat16)
    r529_value = r529_value.to(tl.float32)
    r529_value = tl.broadcast_to(r529_value, [$(16384)])
    tl.store($(candidate_vto_copy_218_ptr) + r529_z, r529_value)
    r530_row = tl.arange($(16))
    r530_col = tl.arange($(16))
    r530_lane = tl.arange($(192))
    for r530_group in range($(8)) {
      r530_left = tl.load($(candidate_vto_copy_214_ptr) + (((((((r530_group % $(8)) * $(3072)) + (r530_row[:, None] * $(192)) + r530_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r530_group % $(8)) * $(3072)) + (r530_row[:, None] * $(192)) + r530_lane[None, :]) / $(192)) % $(16)) * $(192)) + ((((r530_group % $(8)) * $(3072)) + (r530_row[:, None] * $(192)) + r530_lane[None, :]) % $(192))), dtype=tl.float32)
      r530_right = tl.load($(candidate_vto_copy_215_ptr) + (((((((r530_group % $(8)) * $(3072)) + (r530_lane[:, None] * $(16)) + r530_col[None, :]) / $(3072)) % $(8)) * $(3072)) + (((((r530_group % $(8)) * $(3072)) + (r530_lane[:, None] * $(16)) + r530_col[None, :]) % $(16)) * $(192)) + (((((r530_group % $(8)) * $(3072)) + (r530_lane[:, None] * $(16)) + r530_col[None, :]) / $(16)) % $(192))), dtype=tl.float32)
      r530_value = tl.dot(r530_left, r530_right)
      r530_value = r530_value.to(tl.float32)
      tl.store($(candidate_vbmm_121_ptr) + ((r530_group * $(256)) + (r530_row[:, None] * $(16)) + r530_col[None, :]), r530_value)
    }
    r531_z = tl.arange($(2048))
    r531_value = (tl.load($(candidate_vbmm_121_ptr) + ((((r531_z / $(256)) % $(8)) * $(256)) + (((r531_z / $(16)) % $(16)) * $(16)) + (r531_z % $(16))), dtype=tl.float32) * 0.07216878364870322)
    r531_value = r531_value.to(tl.float32)
    r531_value = tl.broadcast_to(r531_value, [$(2048)])
    tl.store($(candidate_vmul_102_ptr) + r531_z, r531_value)
    r532_z = tl.arange($(16))
    r532_value = (tl.cast(r532_z, tl.int64) + tl.full([], $(16), dtype=tl.int64))
    r532_value = tl.broadcast_to(r532_value, [$(16)])
    tl.store($(candidate_vadd_60_ptr) + r532_z, r532_value)
    r533_z = tl.arange($(16))
    r533_value = (tl.load($(candidate_vadd_60_ptr) + r533_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r533_value = tl.broadcast_to(r533_value, [$(16)])
    tl.store($(candidate_vsub_66_ptr) + r533_z, r533_value)
    r534_z = tl.arange($(256))
    r534_value = (tl.cast((r534_z % $(16)), tl.int64) > tl.load($(candidate_vsub_66_ptr) + ((r534_z / $(16)) % $(16)), dtype=tl.int64))
    r534_value = tl.broadcast_to(r534_value, [$(256)])
    tl.store($(candidate_vgt_35_ptr) + r534_z, r534_value)
    r535_z = tl.arange($(2048))
    r535_value = tl.where(tl.load($(candidate_vgt_35_ptr) + ((((r535_z / $(16)) % $(16)) * $(16)) + (r535_z % $(16))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_102_ptr) + ((((r535_z / $(256)) % $(8)) * $(256)) + (((r535_z / $(16)) % $(16)) * $(16)) + (r535_z % $(16))), dtype=tl.float32))
    r535_value = r535_value.to(tl.float32)
    r535_value = tl.broadcast_to(r535_value, [$(2048)])
    tl.store($(candidate_vmasked_fill_35_ptr) + r535_z, r535_value)
    r536_z = tl.arange($(2048))
    r536_value = (tl.load($(candidate_vmasked_fill_35_ptr) + ((((r536_z / $(256)) % $(8)) * $(256)) + (((r536_z / $(16)) % $(16)) * $(16)) + (r536_z % $(16))), dtype=tl.float32) - tl.load($(candidate_vcat_50_ptr) + ((((r536_z / $(16)) % $(16)) * $(8)) + ((r536_z / $(256)) % $(8)) + $(128)), dtype=tl.float32))
    r536_value = r536_value.to(tl.float32)
    r536_value = tl.broadcast_to(r536_value, [$(2048)])
    tl.store($(candidate_vsub_67_ptr) + r536_z, r536_value)
    r537_z = tl.arange($(2048))
    r537_value = tl.exp(tl.load($(candidate_vsub_67_ptr) + ((((r537_z / $(256)) % $(8)) * $(256)) + (((r537_z / $(16)) % $(16)) * $(16)) + (r537_z % $(16))), dtype=tl.float32))
    r537_value = r537_value.to(tl.float32)
    r537_value = tl.broadcast_to(r537_value, [$(2048)])
    tl.store($(candidate_vexp_11_ptr) + r537_z, r537_value)
    r538_z = tl.arange($(16384))
    r538_value = (tl.load($(candidate_vto_copy_217_ptr) + ((((r538_z / $(2048)) % $(8)) * $(2048)) + (((r538_z / $(128)) % $(16)) * $(128)) + (r538_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_218_ptr) + ((((r538_z / $(2048)) % $(8)) * $(2048)) + (((r538_z / $(128)) % $(16)) * $(128)) + (r538_z % $(128))), dtype=tl.float32))
    r538_value = r538_value.to(tl.float32)
    r538_value = tl.broadcast_to(r538_value, [$(16384)])
    tl.store($(candidate_vmul_103_ptr) + r538_z, r538_value)
    r539_z = tl.arange($(128))
    r539_lane = tl.arange($(128))
    r539_value = tl.sum(tl.load($(candidate_vmul_103_ptr) + ((((r539_z[:, None] / $(16)) % $(8)) * $(2048)) + ((r539_z[:, None] % $(16)) * $(128)) + r539_lane[None, :]), dtype=tl.float32), axis=1)
    r539_value = r539_value.to(tl.float32)
    r539_value = tl.broadcast_to(r539_value, [$(128)])
    tl.store($(candidate_vsum_15_ptr) + r539_z, r539_value)
    r540_row = tl.arange($(16))
    r540_col = tl.arange($(16))
    r540_lane = tl.arange($(128))
    for r540_group in range($(8)) {
      r540_left = tl.load($(candidate_vto_copy_217_ptr) + (((((((r540_group % $(8)) * $(2048)) + (r540_row[:, None] * $(128)) + r540_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r540_group % $(8)) * $(2048)) + (r540_row[:, None] * $(128)) + r540_lane[None, :]) / $(128)) % $(16)) * $(128)) + ((((r540_group % $(8)) * $(2048)) + (r540_row[:, None] * $(128)) + r540_lane[None, :]) % $(128))), dtype=tl.float32)
      r540_right = tl.load($(candidate_vto_copy_216_ptr) + (((((((r540_group % $(8)) * $(2048)) + (r540_lane[:, None] * $(16)) + r540_col[None, :]) / $(2048)) % $(8)) * $(2048)) + (((((r540_group % $(8)) * $(2048)) + (r540_lane[:, None] * $(16)) + r540_col[None, :]) % $(16)) * $(128)) + (((((r540_group % $(8)) * $(2048)) + (r540_lane[:, None] * $(16)) + r540_col[None, :]) / $(16)) % $(128))), dtype=tl.float32)
      r540_value = tl.dot(r540_left, r540_right)
      r540_value = r540_value.to(tl.float32)
      tl.store($(candidate_vbmm_122_ptr) + ((r540_group * $(256)) + (r540_row[:, None] * $(16)) + r540_col[None, :]), r540_value)
    }
    r541_z = tl.arange($(2048))
    r541_value = (tl.load($(candidate_vbmm_122_ptr) + ((((r541_z / $(256)) % $(8)) * $(256)) + (((r541_z / $(16)) % $(16)) * $(16)) + (r541_z % $(16))), dtype=tl.float32) - tl.load($(candidate_vsum_15_ptr) + ((((r541_z / $(256)) % $(8)) * $(16)) + ((r541_z / $(16)) % $(16))), dtype=tl.float32))
    r541_value = r541_value.to(tl.float32)
    r541_value = tl.broadcast_to(r541_value, [$(2048)])
    tl.store($(candidate_vsub_68_ptr) + r541_z, r541_value)
    r542_z = tl.arange($(2048))
    r542_value = (tl.load($(candidate_vexp_11_ptr) + ((((r542_z / $(256)) % $(8)) * $(256)) + (((r542_z / $(16)) % $(16)) * $(16)) + (r542_z % $(16))), dtype=tl.float32) * tl.load($(candidate_vsub_68_ptr) + ((((r542_z / $(256)) % $(8)) * $(256)) + (((r542_z / $(16)) % $(16)) * $(16)) + (r542_z % $(16))), dtype=tl.float32))
    r542_value = r542_value.to(tl.float32)
    r542_value = tl.broadcast_to(r542_value, [$(2048)])
    tl.store($(candidate_vmul_104_ptr) + r542_z, r542_value)
    r543_row = tl.arange($(16))
    r543_col = tl.arange($(192))
    r543_lane = tl.arange($(16))
    for r543_group in range($(8)) {
      r543_left = tl.load($(candidate_vmul_104_ptr) + (((((((r543_group % $(8)) * $(256)) + (r543_row[:, None] * $(16)) + r543_lane[None, :]) / $(256)) % $(8)) * $(256)) + (((((r543_group % $(8)) * $(256)) + (r543_row[:, None] * $(16)) + r543_lane[None, :]) % $(16)) * $(16)) + (((((r543_group % $(8)) * $(256)) + (r543_row[:, None] * $(16)) + r543_lane[None, :]) / $(16)) % $(16))), dtype=tl.float32)
      r543_right = tl.load($(candidate_vto_copy_214_ptr) + (((((((r543_group % $(8)) * $(3072)) + (r543_lane[:, None] * $(192)) + r543_col[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r543_group % $(8)) * $(3072)) + (r543_lane[:, None] * $(192)) + r543_col[None, :]) / $(192)) % $(16)) * $(192)) + ((((r543_group % $(8)) * $(3072)) + (r543_lane[:, None] * $(192)) + r543_col[None, :]) % $(192))), dtype=tl.float32)
      r543_value = tl.dot(r543_left, r543_right)
      r543_value = r543_value.to(tl.float32)
      tl.store($(candidate_vbmm_124_ptr) + ((r543_group * $(3072)) + (r543_row[:, None] * $(192)) + r543_col[None, :]), r543_value)
    }
    r544_z = tl.arange($(24576))
    r544_value = (tl.load($(candidate_vbmm_124_ptr) + ((((r544_z / $(3072)) % $(8)) * $(3072)) + (((r544_z / $(192)) % $(16)) * $(192)) + (r544_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r544_value = r544_value.to(tl.float32)
    r544_value = tl.broadcast_to(r544_value, [$(24576)])
    tl.store($(candidate_vmul_106_ptr) + r544_z, r544_value)
    r545_z = tl.arange($(24576))
    r545_value = tl.load($(candidate_vmul_106_ptr) + ((((r545_z / $(192)) % $(8)) * $(3072)) + (((r545_z / $(1536)) % $(16)) * $(192)) + (r545_z % $(192))), dtype=tl.float32)
    r545_value = r545_value.to(tl.bfloat16)
    r545_value = tl.broadcast_to(r545_value, [$(24576)])
    tl.store($(candidate_vto_copy_220_ptr) + r545_z, r545_value)
    r546_row = tl.arange($(16))
    r546_col = tl.arange($(128))
    r546_lane = tl.arange($(16))
    for r546_group in range($(8)) {
      r546_left = tl.load($(candidate_vexp_11_ptr) + (((((((r546_group % $(8)) * $(256)) + (r546_row[:, None] * $(16)) + r546_lane[None, :]) / $(256)) % $(8)) * $(256)) + (((((r546_group % $(8)) * $(256)) + (r546_row[:, None] * $(16)) + r546_lane[None, :]) % $(16)) * $(16)) + (((((r546_group % $(8)) * $(256)) + (r546_row[:, None] * $(16)) + r546_lane[None, :]) / $(16)) % $(16))), dtype=tl.float32)
      r546_right = tl.load($(candidate_vto_copy_217_ptr) + (((((((r546_group % $(8)) * $(2048)) + (r546_lane[:, None] * $(128)) + r546_col[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r546_group % $(8)) * $(2048)) + (r546_lane[:, None] * $(128)) + r546_col[None, :]) / $(128)) % $(16)) * $(128)) + ((((r546_group % $(8)) * $(2048)) + (r546_lane[:, None] * $(128)) + r546_col[None, :]) % $(128))), dtype=tl.float32)
      r546_value = tl.dot(r546_left, r546_right)
      r546_value = r546_value.to(tl.float32)
      tl.store($(candidate_vbmm_125_ptr) + ((r546_group * $(2048)) + (r546_row[:, None] * $(128)) + r546_col[None, :]), r546_value)
    }
    r547_z = tl.arange($(16384))
    r547_value = tl.load($(candidate_vbmm_125_ptr) + ((((r547_z / $(128)) % $(8)) * $(2048)) + (((r547_z / $(1024)) % $(16)) * $(128)) + (r547_z % $(128))), dtype=tl.float32)
    r547_value = r547_value.to(tl.bfloat16)
    r547_value = tl.broadcast_to(r547_value, [$(16384)])
    tl.store($(candidate_vto_copy_221_ptr) + r547_z, r547_value)
    r548_z = tl.arange($(98304))
    r548_value = tl.where(((r548_z / $(1536)) % $(64)) < $(16), tl.load($(candidate_vto_copy_220_ptr) + ((((((r548_z / $(1536)) % $(64)) - $(0)) % $(16)) * $(1536)) + (((r548_z / $(192)) % $(8)) * $(192)) + (r548_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r548_value = r548_value.to(tl.bfloat16)
    r548_value = tl.broadcast_to(r548_value, [$(98304)])
    tl.store($(candidate_vcat_55_ptr) + r548_z, r548_value)
    r549_z = tl.arange($(65536))
    r549_value = tl.where(((r549_z / $(1024)) % $(64)) < $(16), tl.load($(candidate_vto_copy_221_ptr) + ((((((r549_z / $(1024)) % $(64)) - $(0)) % $(16)) * $(1024)) + (((r549_z / $(128)) % $(8)) * $(128)) + (r549_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r549_value = r549_value.to(tl.bfloat16)
    r549_value = tl.broadcast_to(r549_value, [$(65536)])
    tl.store($(candidate_vcat_56_ptr) + r549_z, r549_value)
    r550_z = tl.arange($(196608))
    r550_value = tl.where(((r550_z / $(1536)) % $(128)) < $(48), tl.load($(candidate_vcat_51_ptr) + ((((((r550_z / $(1536)) % $(128)) - $(0)) % $(48)) * $(1536)) + (((r550_z / $(192)) % $(8)) * $(192)) + (r550_z % $(192))), dtype=tl.bfloat16), tl.where(((r550_z / $(1536)) % $(128)) < $(64), tl.load($(candidate_vcat_53_ptr) + ((((((r550_z / $(1536)) % $(128)) - $(48)) % $(16)) * $(1536)) + (((r550_z / $(192)) % $(8)) * $(192)) + (r550_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vcat_55_ptr) + ((((((r550_z / $(1536)) % $(128)) - $(64)) % $(64)) * $(1536)) + (((r550_z / $(192)) % $(8)) * $(192)) + (r550_z % $(192))), dtype=tl.bfloat16)))
    r550_value = r550_value.to(tl.bfloat16)
    r550_value = tl.broadcast_to(r550_value, [$(196608)])
    tl.store($(candidate_vcat_58_ptr) + r550_z, r550_value)
    r551_z = tl.arange($(131072))
    r551_value = tl.where(((r551_z / $(1024)) % $(128)) < $(48), tl.load($(candidate_vcat_52_ptr) + ((((((r551_z / $(1024)) % $(128)) - $(0)) % $(48)) * $(1024)) + (((r551_z / $(128)) % $(8)) * $(128)) + (r551_z % $(128))), dtype=tl.bfloat16), tl.where(((r551_z / $(1024)) % $(128)) < $(64), tl.load($(candidate_vcat_54_ptr) + ((((((r551_z / $(1024)) % $(128)) - $(48)) % $(16)) * $(1024)) + (((r551_z / $(128)) % $(8)) * $(128)) + (r551_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_56_ptr) + ((((((r551_z / $(1024)) % $(128)) - $(64)) % $(64)) * $(1024)) + (((r551_z / $(128)) % $(8)) * $(128)) + (r551_z % $(128))), dtype=tl.bfloat16)))
    r551_value = r551_value.to(tl.bfloat16)
    r551_value = tl.broadcast_to(r551_value, [$(131072)])
    tl.store($(candidate_vcat_59_ptr) + r551_z, r551_value)
    r552_row = tl.arange($(1024))
    r552_col = tl.arange($(512))
    r552_lane = tl.arange($(128))
    for r552_group in range($(1)) {
      r552_left = tl.load($(candidate_vcat_58_ptr) + (((((r552_row[:, None] * $(128)) + r552_lane[None, :]) % $(128)) * $(1536)) + (((((r552_row[:, None] * $(128)) + r552_lane[None, :]) / $(16384)) % $(8)) * $(192)) + ((((r552_row[:, None] * $(128)) + r552_lane[None, :]) / $(128)) % $(128))), dtype=tl.bfloat16)
      r552_right = tl.load($(candidate_vindex_2_ptr) + ((((((r552_lane[:, None] * $(512)) + r552_col[None, :]) / $(512)) % $(128)) * $(576)) + (((r552_lane[:, None] * $(512)) + r552_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r552_value = tl.dot(r552_left, r552_right)
      r552_value = r552_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_128_ptr) + ((r552_group * $(524288)) + (r552_row[:, None] * $(512)) + r552_col[None, :]), r552_value)
    }
    r553_row = tl.arange($(1024))
    r553_col = tl.arange($(512))
    r553_lane = tl.arange($(128))
    for r553_group in range($(1)) {
      r553_left = tl.load($(candidate_vcat_59_ptr) + (((((r553_row[:, None] * $(128)) + r553_lane[None, :]) % $(128)) * $(1024)) + (((((r553_row[:, None] * $(128)) + r553_lane[None, :]) / $(16384)) % $(8)) * $(128)) + ((((r553_row[:, None] * $(128)) + r553_lane[None, :]) / $(128)) % $(128))), dtype=tl.bfloat16)
      r553_right = tl.load($(candidate_vindex_2_ptr) + ((((((r553_lane[:, None] * $(512)) + r553_col[None, :]) / $(512)) % $(128)) * $(576)) + (((r553_lane[:, None] * $(512)) + r553_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r553_value = tl.dot(r553_left, r553_right)
      r553_value = r553_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_129_ptr) + ((r553_group * $(524288)) + (r553_row[:, None] * $(512)) + r553_col[None, :]), r553_value)
    }
    r554_z = tl.arange($(1048576))
    r554_value = tl.where(((r554_z / $(512)) % $(256)) < $(128), tl.load($(candidate_vbmm_128_ptr) + ((((((((r554_z / $(131072)) % $(8)) * $(65536)) + ((((r554_z / $(512)) % $(256)) - $(0)) * $(512)) + (r554_z % $(512))) / $(65536)) % $(8)) * $(65536)) + ((((((r554_z / $(131072)) % $(8)) * $(128)) + (((r554_z / $(512)) % $(256)) - $(0))) % $(128)) * $(512)) + (r554_z % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vbmm_129_ptr) + ((((((((r554_z / $(131072)) % $(8)) * $(65536)) + ((((r554_z / $(512)) % $(256)) - $(128)) * $(512)) + (r554_z % $(512))) / $(65536)) % $(8)) * $(65536)) + ((((((r554_z / $(131072)) % $(8)) * $(128)) + (((r554_z / $(512)) % $(256)) - $(128))) % $(128)) * $(512)) + (r554_z % $(512))), dtype=tl.bfloat16))
    r554_value = r554_value.to(tl.bfloat16)
    r554_value = tl.broadcast_to(r554_value, [$(1048576)])
    tl.store($(candidate_vcat_60_ptr) + r554_z, r554_value)
    r555_z = tl.arange($(1048576))
    r555_value = (tl.load($(candidate_vadd_54_ptr) + ((((r555_z / $(512)) % $(2048)) * $(512)) + (r555_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vcat_60_ptr) + ((((((((r555_z / $(512)) % $(2048)) * $(512)) + (r555_z % $(512))) / $(131072)) % $(8)) * $(131072)) + ((((r555_z / $(512)) % $(2048)) % $(256)) * $(512)) + (r555_z % $(512))), dtype=tl.bfloat16))
    r555_value = r555_value.to(tl.bfloat16)
    r555_value = tl.broadcast_to(r555_value, [$(1048576)])
    tl.store($(candidate_vadd_62_ptr) + r555_z, r555_value)
    z = tl.arange($(1048576))
    value = tl.load($(candidate_vadd_62_ptr) + ((((z / $(512)) % $(2048)) * $(512)) + (z % $(512))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(1048576)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_mla_fa4_te_rope_cp_absorb_wgrad
