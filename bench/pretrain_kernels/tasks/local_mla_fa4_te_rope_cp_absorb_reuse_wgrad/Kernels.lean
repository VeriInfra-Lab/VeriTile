import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_mla_fa4_te_rope_cp_absorb_reuse_wgrad
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
  let vto_copy_28_ptr : RegionName := "vto_copy_28"
  let vmul_26_ptr : RegionName := "vmul_26"
  let vbmm_6_ptr : RegionName := "vbmm_6"
  let vcat_9_ptr : RegionName := "vcat_9"
  let vmm_6_ptr : RegionName := "vmm_6"
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
  let candidate_vto_copy_30_ptr : RegionName := "candidate_vto_copy_30"
  let candidate_vto_copy_31_ptr : RegionName := "candidate_vto_copy_31"
  let candidate_vbmm_3_ptr : RegionName := "candidate_vbmm_3"
  let candidate_vto_copy_32_ptr : RegionName := "candidate_vto_copy_32"
  let candidate_vto_copy_33_ptr : RegionName := "candidate_vto_copy_33"
  let candidate_vmm_3_ptr : RegionName := "candidate_vmm_3"
  let candidate_vadd_8_ptr : RegionName := "candidate_vadd_8"
  let candidate_vmul_23_ptr : RegionName := "candidate_vmul_23"
  let candidate_vadd_9_ptr : RegionName := "candidate_vadd_9"
  let candidate_vsub_9_ptr : RegionName := "candidate_vsub_9"
  let candidate_vgt_ptr : RegionName := "candidate_vgt"
  let candidate_vmasked_fill_ptr : RegionName := "candidate_vmasked_fill"
  let candidate_vsoftmax_ptr : RegionName := "candidate_vsoftmax"
  let candidate_vto_copy_34_ptr : RegionName := "candidate_vto_copy_34"
  let candidate_vmm_4_ptr : RegionName := "candidate_vmm_4"
  let candidate_vlogsumexp_ptr : RegionName := "candidate_vlogsumexp"
  let candidate_vto_copy_35_ptr : RegionName := "candidate_vto_copy_35"
  let candidate_vto_copy_36_ptr : RegionName := "candidate_vto_copy_36"
  let candidate_vbmm_4_ptr : RegionName := "candidate_vbmm_4"
  let candidate_vto_copy_37_ptr : RegionName := "candidate_vto_copy_37"
  let candidate_vto_copy_38_ptr : RegionName := "candidate_vto_copy_38"
  let candidate_vmm_5_ptr : RegionName := "candidate_vmm_5"
  let candidate_vadd_10_ptr : RegionName := "candidate_vadd_10"
  let candidate_vmul_24_ptr : RegionName := "candidate_vmul_24"
  let candidate_vadd_11_ptr : RegionName := "candidate_vadd_11"
  let candidate_vsub_10_ptr : RegionName := "candidate_vsub_10"
  let candidate_vgt_1_ptr : RegionName := "candidate_vgt_1"
  let candidate_vmasked_fill_1_ptr : RegionName := "candidate_vmasked_fill_1"
  let candidate_vsoftmax_1_ptr : RegionName := "candidate_vsoftmax_1"
  let candidate_vto_copy_39_ptr : RegionName := "candidate_vto_copy_39"
  let candidate_vmm_6_ptr : RegionName := "candidate_vmm_6"
  let candidate_vlogsumexp_1_ptr : RegionName := "candidate_vlogsumexp_1"
  let candidate_vto_copy_40_ptr : RegionName := "candidate_vto_copy_40"
  let candidate_vto_copy_41_ptr : RegionName := "candidate_vto_copy_41"
  let candidate_vbmm_5_ptr : RegionName := "candidate_vbmm_5"
  let candidate_vto_copy_42_ptr : RegionName := "candidate_vto_copy_42"
  let candidate_vto_copy_43_ptr : RegionName := "candidate_vto_copy_43"
  let candidate_vmm_7_ptr : RegionName := "candidate_vmm_7"
  let candidate_vadd_12_ptr : RegionName := "candidate_vadd_12"
  let candidate_vmul_25_ptr : RegionName := "candidate_vmul_25"
  let candidate_vadd_13_ptr : RegionName := "candidate_vadd_13"
  let candidate_vsub_11_ptr : RegionName := "candidate_vsub_11"
  let candidate_vgt_2_ptr : RegionName := "candidate_vgt_2"
  let candidate_vmasked_fill_2_ptr : RegionName := "candidate_vmasked_fill_2"
  let candidate_vsoftmax_2_ptr : RegionName := "candidate_vsoftmax_2"
  let candidate_vto_copy_44_ptr : RegionName := "candidate_vto_copy_44"
  let candidate_vmm_8_ptr : RegionName := "candidate_vmm_8"
  let candidate_vlogsumexp_2_ptr : RegionName := "candidate_vlogsumexp_2"
  let candidate_vcat_6_ptr : RegionName := "candidate_vcat_6"
  let candidate_vcat_7_ptr : RegionName := "candidate_vcat_7"
  let candidate_vbmm_6_ptr : RegionName := "candidate_vbmm_6"
  let candidate_vbmm_7_ptr : RegionName := "candidate_vbmm_7"
  let candidate_vto_copy_45_ptr : RegionName := "candidate_vto_copy_45"
  let candidate_vto_copy_46_ptr : RegionName := "candidate_vto_copy_46"
  let candidate_vbmm_8_ptr : RegionName := "candidate_vbmm_8"
  let candidate_vto_copy_47_ptr : RegionName := "candidate_vto_copy_47"
  let candidate_vto_copy_48_ptr : RegionName := "candidate_vto_copy_48"
  let candidate_vmm_9_ptr : RegionName := "candidate_vmm_9"
  let candidate_vadd_14_ptr : RegionName := "candidate_vadd_14"
  let candidate_vmul_26_ptr : RegionName := "candidate_vmul_26"
  let candidate_vadd_15_ptr : RegionName := "candidate_vadd_15"
  let candidate_vsub_12_ptr : RegionName := "candidate_vsub_12"
  let candidate_vgt_3_ptr : RegionName := "candidate_vgt_3"
  let candidate_vmasked_fill_3_ptr : RegionName := "candidate_vmasked_fill_3"
  let candidate_vsoftmax_3_ptr : RegionName := "candidate_vsoftmax_3"
  let candidate_vto_copy_49_ptr : RegionName := "candidate_vto_copy_49"
  let candidate_vmm_10_ptr : RegionName := "candidate_vmm_10"
  let candidate_vlogsumexp_3_ptr : RegionName := "candidate_vlogsumexp_3"
  let candidate_vto_copy_50_ptr : RegionName := "candidate_vto_copy_50"
  let candidate_vto_copy_51_ptr : RegionName := "candidate_vto_copy_51"
  let candidate_vbmm_9_ptr : RegionName := "candidate_vbmm_9"
  let candidate_vto_copy_52_ptr : RegionName := "candidate_vto_copy_52"
  let candidate_vto_copy_53_ptr : RegionName := "candidate_vto_copy_53"
  let candidate_vmm_11_ptr : RegionName := "candidate_vmm_11"
  let candidate_vadd_16_ptr : RegionName := "candidate_vadd_16"
  let candidate_vmul_27_ptr : RegionName := "candidate_vmul_27"
  let candidate_vadd_17_ptr : RegionName := "candidate_vadd_17"
  let candidate_vsub_13_ptr : RegionName := "candidate_vsub_13"
  let candidate_vgt_4_ptr : RegionName := "candidate_vgt_4"
  let candidate_vmasked_fill_4_ptr : RegionName := "candidate_vmasked_fill_4"
  let candidate_vsoftmax_4_ptr : RegionName := "candidate_vsoftmax_4"
  let candidate_vto_copy_54_ptr : RegionName := "candidate_vto_copy_54"
  let candidate_vmm_12_ptr : RegionName := "candidate_vmm_12"
  let candidate_vlogsumexp_4_ptr : RegionName := "candidate_vlogsumexp_4"
  let candidate_vto_copy_55_ptr : RegionName := "candidate_vto_copy_55"
  let candidate_vto_copy_56_ptr : RegionName := "candidate_vto_copy_56"
  let candidate_vbmm_10_ptr : RegionName := "candidate_vbmm_10"
  let candidate_vto_copy_57_ptr : RegionName := "candidate_vto_copy_57"
  let candidate_vto_copy_58_ptr : RegionName := "candidate_vto_copy_58"
  let candidate_vmm_13_ptr : RegionName := "candidate_vmm_13"
  let candidate_vadd_18_ptr : RegionName := "candidate_vadd_18"
  let candidate_vmul_28_ptr : RegionName := "candidate_vmul_28"
  let candidate_vadd_19_ptr : RegionName := "candidate_vadd_19"
  let candidate_vsub_14_ptr : RegionName := "candidate_vsub_14"
  let candidate_vgt_5_ptr : RegionName := "candidate_vgt_5"
  let candidate_vmasked_fill_5_ptr : RegionName := "candidate_vmasked_fill_5"
  let candidate_vsoftmax_5_ptr : RegionName := "candidate_vsoftmax_5"
  let candidate_vto_copy_59_ptr : RegionName := "candidate_vto_copy_59"
  let candidate_vmm_14_ptr : RegionName := "candidate_vmm_14"
  let candidate_vlogsumexp_5_ptr : RegionName := "candidate_vlogsumexp_5"
  let candidate_vcat_8_ptr : RegionName := "candidate_vcat_8"
  let candidate_vcat_9_ptr : RegionName := "candidate_vcat_9"
  let candidate_vbmm_11_ptr : RegionName := "candidate_vbmm_11"
  let candidate_vbmm_12_ptr : RegionName := "candidate_vbmm_12"
  let candidate_vto_copy_60_ptr : RegionName := "candidate_vto_copy_60"
  let candidate_vto_copy_61_ptr : RegionName := "candidate_vto_copy_61"
  let candidate_vbmm_13_ptr : RegionName := "candidate_vbmm_13"
  let candidate_vto_copy_62_ptr : RegionName := "candidate_vto_copy_62"
  let candidate_vto_copy_63_ptr : RegionName := "candidate_vto_copy_63"
  let candidate_vmm_15_ptr : RegionName := "candidate_vmm_15"
  let candidate_vadd_20_ptr : RegionName := "candidate_vadd_20"
  let candidate_vmul_29_ptr : RegionName := "candidate_vmul_29"
  let candidate_vadd_21_ptr : RegionName := "candidate_vadd_21"
  let candidate_vsub_15_ptr : RegionName := "candidate_vsub_15"
  let candidate_vgt_6_ptr : RegionName := "candidate_vgt_6"
  let candidate_vmasked_fill_6_ptr : RegionName := "candidate_vmasked_fill_6"
  let candidate_vsoftmax_6_ptr : RegionName := "candidate_vsoftmax_6"
  let candidate_vto_copy_64_ptr : RegionName := "candidate_vto_copy_64"
  let candidate_vmm_16_ptr : RegionName := "candidate_vmm_16"
  let candidate_vlogsumexp_6_ptr : RegionName := "candidate_vlogsumexp_6"
  let candidate_vto_copy_65_ptr : RegionName := "candidate_vto_copy_65"
  let candidate_vto_copy_66_ptr : RegionName := "candidate_vto_copy_66"
  let candidate_vbmm_14_ptr : RegionName := "candidate_vbmm_14"
  let candidate_vto_copy_67_ptr : RegionName := "candidate_vto_copy_67"
  let candidate_vto_copy_68_ptr : RegionName := "candidate_vto_copy_68"
  let candidate_vmm_17_ptr : RegionName := "candidate_vmm_17"
  let candidate_vadd_22_ptr : RegionName := "candidate_vadd_22"
  let candidate_vmul_30_ptr : RegionName := "candidate_vmul_30"
  let candidate_vadd_23_ptr : RegionName := "candidate_vadd_23"
  let candidate_vsub_16_ptr : RegionName := "candidate_vsub_16"
  let candidate_vgt_7_ptr : RegionName := "candidate_vgt_7"
  let candidate_vmasked_fill_7_ptr : RegionName := "candidate_vmasked_fill_7"
  let candidate_vsoftmax_7_ptr : RegionName := "candidate_vsoftmax_7"
  let candidate_vto_copy_69_ptr : RegionName := "candidate_vto_copy_69"
  let candidate_vmm_18_ptr : RegionName := "candidate_vmm_18"
  let candidate_vlogsumexp_7_ptr : RegionName := "candidate_vlogsumexp_7"
  let candidate_vto_copy_70_ptr : RegionName := "candidate_vto_copy_70"
  let candidate_vto_copy_71_ptr : RegionName := "candidate_vto_copy_71"
  let candidate_vbmm_15_ptr : RegionName := "candidate_vbmm_15"
  let candidate_vto_copy_72_ptr : RegionName := "candidate_vto_copy_72"
  let candidate_vto_copy_73_ptr : RegionName := "candidate_vto_copy_73"
  let candidate_vmm_19_ptr : RegionName := "candidate_vmm_19"
  let candidate_vadd_24_ptr : RegionName := "candidate_vadd_24"
  let candidate_vmul_31_ptr : RegionName := "candidate_vmul_31"
  let candidate_vadd_25_ptr : RegionName := "candidate_vadd_25"
  let candidate_vsub_17_ptr : RegionName := "candidate_vsub_17"
  let candidate_vgt_8_ptr : RegionName := "candidate_vgt_8"
  let candidate_vmasked_fill_8_ptr : RegionName := "candidate_vmasked_fill_8"
  let candidate_vsoftmax_8_ptr : RegionName := "candidate_vsoftmax_8"
  let candidate_vto_copy_74_ptr : RegionName := "candidate_vto_copy_74"
  let candidate_vmm_20_ptr : RegionName := "candidate_vmm_20"
  let candidate_vlogsumexp_8_ptr : RegionName := "candidate_vlogsumexp_8"
  let candidate_vcat_10_ptr : RegionName := "candidate_vcat_10"
  let candidate_vcat_11_ptr : RegionName := "candidate_vcat_11"
  let candidate_vbmm_16_ptr : RegionName := "candidate_vbmm_16"
  let candidate_vbmm_17_ptr : RegionName := "candidate_vbmm_17"
  let candidate_vto_copy_75_ptr : RegionName := "candidate_vto_copy_75"
  let candidate_vto_copy_76_ptr : RegionName := "candidate_vto_copy_76"
  let candidate_vbmm_18_ptr : RegionName := "candidate_vbmm_18"
  let candidate_vto_copy_77_ptr : RegionName := "candidate_vto_copy_77"
  let candidate_vto_copy_78_ptr : RegionName := "candidate_vto_copy_78"
  let candidate_vmm_21_ptr : RegionName := "candidate_vmm_21"
  let candidate_vadd_26_ptr : RegionName := "candidate_vadd_26"
  let candidate_vmul_32_ptr : RegionName := "candidate_vmul_32"
  let candidate_vadd_27_ptr : RegionName := "candidate_vadd_27"
  let candidate_vsub_18_ptr : RegionName := "candidate_vsub_18"
  let candidate_vgt_9_ptr : RegionName := "candidate_vgt_9"
  let candidate_vmasked_fill_9_ptr : RegionName := "candidate_vmasked_fill_9"
  let candidate_vsoftmax_9_ptr : RegionName := "candidate_vsoftmax_9"
  let candidate_vto_copy_79_ptr : RegionName := "candidate_vto_copy_79"
  let candidate_vmm_22_ptr : RegionName := "candidate_vmm_22"
  let candidate_vlogsumexp_9_ptr : RegionName := "candidate_vlogsumexp_9"
  let candidate_vto_copy_80_ptr : RegionName := "candidate_vto_copy_80"
  let candidate_vto_copy_81_ptr : RegionName := "candidate_vto_copy_81"
  let candidate_vbmm_19_ptr : RegionName := "candidate_vbmm_19"
  let candidate_vto_copy_82_ptr : RegionName := "candidate_vto_copy_82"
  let candidate_vto_copy_83_ptr : RegionName := "candidate_vto_copy_83"
  let candidate_vmm_23_ptr : RegionName := "candidate_vmm_23"
  let candidate_vadd_28_ptr : RegionName := "candidate_vadd_28"
  let candidate_vmul_33_ptr : RegionName := "candidate_vmul_33"
  let candidate_vadd_29_ptr : RegionName := "candidate_vadd_29"
  let candidate_vsub_19_ptr : RegionName := "candidate_vsub_19"
  let candidate_vgt_10_ptr : RegionName := "candidate_vgt_10"
  let candidate_vmasked_fill_10_ptr : RegionName := "candidate_vmasked_fill_10"
  let candidate_vsoftmax_10_ptr : RegionName := "candidate_vsoftmax_10"
  let candidate_vto_copy_84_ptr : RegionName := "candidate_vto_copy_84"
  let candidate_vmm_24_ptr : RegionName := "candidate_vmm_24"
  let candidate_vlogsumexp_10_ptr : RegionName := "candidate_vlogsumexp_10"
  let candidate_vto_copy_85_ptr : RegionName := "candidate_vto_copy_85"
  let candidate_vto_copy_86_ptr : RegionName := "candidate_vto_copy_86"
  let candidate_vbmm_20_ptr : RegionName := "candidate_vbmm_20"
  let candidate_vto_copy_87_ptr : RegionName := "candidate_vto_copy_87"
  let candidate_vto_copy_88_ptr : RegionName := "candidate_vto_copy_88"
  let candidate_vmm_25_ptr : RegionName := "candidate_vmm_25"
  let candidate_vadd_30_ptr : RegionName := "candidate_vadd_30"
  let candidate_vmul_34_ptr : RegionName := "candidate_vmul_34"
  let candidate_vadd_31_ptr : RegionName := "candidate_vadd_31"
  let candidate_vsub_20_ptr : RegionName := "candidate_vsub_20"
  let candidate_vgt_11_ptr : RegionName := "candidate_vgt_11"
  let candidate_vmasked_fill_11_ptr : RegionName := "candidate_vmasked_fill_11"
  let candidate_vsoftmax_11_ptr : RegionName := "candidate_vsoftmax_11"
  let candidate_vto_copy_89_ptr : RegionName := "candidate_vto_copy_89"
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
  let candidate_vto_copy_90_ptr : RegionName := "candidate_vto_copy_90"
  let candidate_vto_copy_91_ptr : RegionName := "candidate_vto_copy_91"
  let candidate_vto_copy_92_ptr : RegionName := "candidate_vto_copy_92"
  let candidate_vto_copy_93_ptr : RegionName := "candidate_vto_copy_93"
  let candidate_vto_copy_94_ptr : RegionName := "candidate_vto_copy_94"
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
  let candidate_vbmm_27_ptr : RegionName := "candidate_vbmm_27"
  let candidate_vmul_39_ptr : RegionName := "candidate_vmul_39"
  let candidate_vto_copy_96_ptr : RegionName := "candidate_vto_copy_96"
  let candidate_vbmm_28_ptr : RegionName := "candidate_vbmm_28"
  let candidate_vto_copy_97_ptr : RegionName := "candidate_vto_copy_97"
  let candidate_vcat_17_ptr : RegionName := "candidate_vcat_17"
  let candidate_vcat_18_ptr : RegionName := "candidate_vcat_18"
  let candidate_vto_copy_98_ptr : RegionName := "candidate_vto_copy_98"
  let candidate_vto_copy_99_ptr : RegionName := "candidate_vto_copy_99"
  let candidate_vto_copy_100_ptr : RegionName := "candidate_vto_copy_100"
  let candidate_vto_copy_101_ptr : RegionName := "candidate_vto_copy_101"
  let candidate_vto_copy_102_ptr : RegionName := "candidate_vto_copy_102"
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
  let candidate_vbmm_32_ptr : RegionName := "candidate_vbmm_32"
  let candidate_vmul_44_ptr : RegionName := "candidate_vmul_44"
  let candidate_vto_copy_104_ptr : RegionName := "candidate_vto_copy_104"
  let candidate_vbmm_33_ptr : RegionName := "candidate_vbmm_33"
  let candidate_vto_copy_105_ptr : RegionName := "candidate_vto_copy_105"
  let candidate_vcat_19_ptr : RegionName := "candidate_vcat_19"
  let candidate_vcat_20_ptr : RegionName := "candidate_vcat_20"
  let candidate_vto_copy_106_ptr : RegionName := "candidate_vto_copy_106"
  let candidate_vto_copy_107_ptr : RegionName := "candidate_vto_copy_107"
  let candidate_vto_copy_108_ptr : RegionName := "candidate_vto_copy_108"
  let candidate_vto_copy_109_ptr : RegionName := "candidate_vto_copy_109"
  let candidate_vto_copy_110_ptr : RegionName := "candidate_vto_copy_110"
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
  let candidate_vbmm_37_ptr : RegionName := "candidate_vbmm_37"
  let candidate_vmul_49_ptr : RegionName := "candidate_vmul_49"
  let candidate_vto_copy_112_ptr : RegionName := "candidate_vto_copy_112"
  let candidate_vbmm_38_ptr : RegionName := "candidate_vbmm_38"
  let candidate_vto_copy_113_ptr : RegionName := "candidate_vto_copy_113"
  let candidate_vcat_21_ptr : RegionName := "candidate_vcat_21"
  let candidate_vcat_22_ptr : RegionName := "candidate_vcat_22"
  let candidate_vcat_24_ptr : RegionName := "candidate_vcat_24"
  let candidate_vcat_25_ptr : RegionName := "candidate_vcat_25"
  let candidate_vbmm_41_ptr : RegionName := "candidate_vbmm_41"
  let candidate_vbmm_42_ptr : RegionName := "candidate_vbmm_42"
  let candidate_vcat_26_ptr : RegionName := "candidate_vcat_26"
  let candidate_vbmm_43_ptr : RegionName := "candidate_vbmm_43"
  let candidate_vbmm_44_ptr : RegionName := "candidate_vbmm_44"
  let candidate_vcat_27_ptr : RegionName := "candidate_vcat_27"
  let candidate_vcat_28_ptr : RegionName := "candidate_vcat_28"
  let candidate_vto_copy_114_ptr : RegionName := "candidate_vto_copy_114"
  let candidate_vto_copy_115_ptr : RegionName := "candidate_vto_copy_115"
  let candidate_vto_copy_116_ptr : RegionName := "candidate_vto_copy_116"
  let candidate_vto_copy_117_ptr : RegionName := "candidate_vto_copy_117"
  let candidate_vto_copy_118_ptr : RegionName := "candidate_vto_copy_118"
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
  let candidate_vbmm_48_ptr : RegionName := "candidate_vbmm_48"
  let candidate_vmul_54_ptr : RegionName := "candidate_vmul_54"
  let candidate_vto_copy_120_ptr : RegionName := "candidate_vto_copy_120"
  let candidate_vbmm_49_ptr : RegionName := "candidate_vbmm_49"
  let candidate_vto_copy_121_ptr : RegionName := "candidate_vto_copy_121"
  let candidate_vcat_29_ptr : RegionName := "candidate_vcat_29"
  let candidate_vcat_30_ptr : RegionName := "candidate_vcat_30"
  let candidate_vto_copy_122_ptr : RegionName := "candidate_vto_copy_122"
  let candidate_vto_copy_123_ptr : RegionName := "candidate_vto_copy_123"
  let candidate_vto_copy_124_ptr : RegionName := "candidate_vto_copy_124"
  let candidate_vto_copy_125_ptr : RegionName := "candidate_vto_copy_125"
  let candidate_vto_copy_126_ptr : RegionName := "candidate_vto_copy_126"
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
  let candidate_vbmm_53_ptr : RegionName := "candidate_vbmm_53"
  let candidate_vmul_59_ptr : RegionName := "candidate_vmul_59"
  let candidate_vto_copy_128_ptr : RegionName := "candidate_vto_copy_128"
  let candidate_vbmm_54_ptr : RegionName := "candidate_vbmm_54"
  let candidate_vto_copy_129_ptr : RegionName := "candidate_vto_copy_129"
  let candidate_vcat_31_ptr : RegionName := "candidate_vcat_31"
  let candidate_vcat_32_ptr : RegionName := "candidate_vcat_32"
  let candidate_vto_copy_130_ptr : RegionName := "candidate_vto_copy_130"
  let candidate_vto_copy_131_ptr : RegionName := "candidate_vto_copy_131"
  let candidate_vto_copy_132_ptr : RegionName := "candidate_vto_copy_132"
  let candidate_vto_copy_133_ptr : RegionName := "candidate_vto_copy_133"
  let candidate_vto_copy_134_ptr : RegionName := "candidate_vto_copy_134"
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
  let candidate_vbmm_58_ptr : RegionName := "candidate_vbmm_58"
  let candidate_vmul_64_ptr : RegionName := "candidate_vmul_64"
  let candidate_vto_copy_136_ptr : RegionName := "candidate_vto_copy_136"
  let candidate_vbmm_59_ptr : RegionName := "candidate_vbmm_59"
  let candidate_vto_copy_137_ptr : RegionName := "candidate_vto_copy_137"
  let candidate_vcat_33_ptr : RegionName := "candidate_vcat_33"
  let candidate_vcat_34_ptr : RegionName := "candidate_vcat_34"
  let candidate_vcat_36_ptr : RegionName := "candidate_vcat_36"
  let candidate_vcat_37_ptr : RegionName := "candidate_vcat_37"
  let candidate_vbmm_62_ptr : RegionName := "candidate_vbmm_62"
  let candidate_vbmm_63_ptr : RegionName := "candidate_vbmm_63"
  let candidate_vcat_38_ptr : RegionName := "candidate_vcat_38"
  let candidate_vadd_40_ptr : RegionName := "candidate_vadd_40"
  let candidate_vbmm_64_ptr : RegionName := "candidate_vbmm_64"
  let candidate_vbmm_65_ptr : RegionName := "candidate_vbmm_65"
  let candidate_vcat_39_ptr : RegionName := "candidate_vcat_39"
  let candidate_vcat_40_ptr : RegionName := "candidate_vcat_40"
  let candidate_vto_copy_138_ptr : RegionName := "candidate_vto_copy_138"
  let candidate_vto_copy_139_ptr : RegionName := "candidate_vto_copy_139"
  let candidate_vto_copy_140_ptr : RegionName := "candidate_vto_copy_140"
  let candidate_vto_copy_141_ptr : RegionName := "candidate_vto_copy_141"
  let candidate_vto_copy_142_ptr : RegionName := "candidate_vto_copy_142"
  let candidate_vbmm_66_ptr : RegionName := "candidate_vbmm_66"
  let candidate_vmul_65_ptr : RegionName := "candidate_vmul_65"
  let candidate_vadd_41_ptr : RegionName := "candidate_vadd_41"
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
  let candidate_vbmm_69_ptr : RegionName := "candidate_vbmm_69"
  let candidate_vmul_69_ptr : RegionName := "candidate_vmul_69"
  let candidate_vto_copy_144_ptr : RegionName := "candidate_vto_copy_144"
  let candidate_vbmm_70_ptr : RegionName := "candidate_vbmm_70"
  let candidate_vto_copy_145_ptr : RegionName := "candidate_vto_copy_145"
  let candidate_vto_copy_146_ptr : RegionName := "candidate_vto_copy_146"
  let candidate_vto_copy_147_ptr : RegionName := "candidate_vto_copy_147"
  let candidate_vto_copy_148_ptr : RegionName := "candidate_vto_copy_148"
  let candidate_vto_copy_149_ptr : RegionName := "candidate_vto_copy_149"
  let candidate_vto_copy_150_ptr : RegionName := "candidate_vto_copy_150"
  let candidate_vbmm_71_ptr : RegionName := "candidate_vbmm_71"
  let candidate_vmul_70_ptr : RegionName := "candidate_vmul_70"
  let candidate_vadd_42_ptr : RegionName := "candidate_vadd_42"
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
  let candidate_vbmm_74_ptr : RegionName := "candidate_vbmm_74"
  let candidate_vmul_74_ptr : RegionName := "candidate_vmul_74"
  let candidate_vto_copy_152_ptr : RegionName := "candidate_vto_copy_152"
  let candidate_vbmm_75_ptr : RegionName := "candidate_vbmm_75"
  let candidate_vto_copy_153_ptr : RegionName := "candidate_vto_copy_153"
  let candidate_vto_copy_154_ptr : RegionName := "candidate_vto_copy_154"
  let candidate_vto_copy_155_ptr : RegionName := "candidate_vto_copy_155"
  let candidate_vto_copy_156_ptr : RegionName := "candidate_vto_copy_156"
  let candidate_vto_copy_157_ptr : RegionName := "candidate_vto_copy_157"
  let candidate_vto_copy_158_ptr : RegionName := "candidate_vto_copy_158"
  let candidate_vbmm_76_ptr : RegionName := "candidate_vbmm_76"
  let candidate_vmul_75_ptr : RegionName := "candidate_vmul_75"
  let candidate_vadd_43_ptr : RegionName := "candidate_vadd_43"
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
  let candidate_vbmm_79_ptr : RegionName := "candidate_vbmm_79"
  let candidate_vmul_79_ptr : RegionName := "candidate_vmul_79"
  let candidate_vto_copy_160_ptr : RegionName := "candidate_vto_copy_160"
  let candidate_vbmm_80_ptr : RegionName := "candidate_vbmm_80"
  let candidate_vto_copy_161_ptr : RegionName := "candidate_vto_copy_161"
  let candidate_vcat_42_ptr : RegionName := "candidate_vcat_42"
  let candidate_vcat_43_ptr : RegionName := "candidate_vcat_43"
  let candidate_vbmm_83_ptr : RegionName := "candidate_vbmm_83"
  let candidate_vbmm_84_ptr : RegionName := "candidate_vbmm_84"
  let candidate_vcat_44_ptr : RegionName := "candidate_vcat_44"
  let candidate_vadd_45_ptr : RegionName := "candidate_vadd_45"
  let candidate_vbmm_85_ptr : RegionName := "candidate_vbmm_85"
  let candidate_vbmm_86_ptr : RegionName := "candidate_vbmm_86"
  let candidate_vcat_45_ptr : RegionName := "candidate_vcat_45"
  let candidate_vcat_46_ptr : RegionName := "candidate_vcat_46"
  let candidate_vto_copy_162_ptr : RegionName := "candidate_vto_copy_162"
  let candidate_vto_copy_163_ptr : RegionName := "candidate_vto_copy_163"
  let candidate_vto_copy_164_ptr : RegionName := "candidate_vto_copy_164"
  let candidate_vto_copy_165_ptr : RegionName := "candidate_vto_copy_165"
  let candidate_vto_copy_166_ptr : RegionName := "candidate_vto_copy_166"
  let candidate_vbmm_87_ptr : RegionName := "candidate_vbmm_87"
  let candidate_vmul_80_ptr : RegionName := "candidate_vmul_80"
  let candidate_vadd_46_ptr : RegionName := "candidate_vadd_46"
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
  let candidate_vbmm_90_ptr : RegionName := "candidate_vbmm_90"
  let candidate_vmul_84_ptr : RegionName := "candidate_vmul_84"
  let candidate_vto_copy_168_ptr : RegionName := "candidate_vto_copy_168"
  let candidate_vbmm_91_ptr : RegionName := "candidate_vbmm_91"
  let candidate_vto_copy_169_ptr : RegionName := "candidate_vto_copy_169"
  let candidate_vcat_47_ptr : RegionName := "candidate_vcat_47"
  let candidate_vcat_48_ptr : RegionName := "candidate_vcat_48"
  let candidate_vto_copy_170_ptr : RegionName := "candidate_vto_copy_170"
  let candidate_vto_copy_171_ptr : RegionName := "candidate_vto_copy_171"
  let candidate_vto_copy_172_ptr : RegionName := "candidate_vto_copy_172"
  let candidate_vto_copy_173_ptr : RegionName := "candidate_vto_copy_173"
  let candidate_vto_copy_174_ptr : RegionName := "candidate_vto_copy_174"
  let candidate_vbmm_92_ptr : RegionName := "candidate_vbmm_92"
  let candidate_vmul_85_ptr : RegionName := "candidate_vmul_85"
  let candidate_vadd_47_ptr : RegionName := "candidate_vadd_47"
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
  let candidate_vbmm_95_ptr : RegionName := "candidate_vbmm_95"
  let candidate_vmul_89_ptr : RegionName := "candidate_vmul_89"
  let candidate_vto_copy_176_ptr : RegionName := "candidate_vto_copy_176"
  let candidate_vbmm_96_ptr : RegionName := "candidate_vbmm_96"
  let candidate_vto_copy_177_ptr : RegionName := "candidate_vto_copy_177"
  let candidate_vcat_49_ptr : RegionName := "candidate_vcat_49"
  let candidate_vcat_50_ptr : RegionName := "candidate_vcat_50"
  let candidate_vto_copy_178_ptr : RegionName := "candidate_vto_copy_178"
  let candidate_vto_copy_179_ptr : RegionName := "candidate_vto_copy_179"
  let candidate_vto_copy_180_ptr : RegionName := "candidate_vto_copy_180"
  let candidate_vto_copy_181_ptr : RegionName := "candidate_vto_copy_181"
  let candidate_vto_copy_182_ptr : RegionName := "candidate_vto_copy_182"
  let candidate_vbmm_97_ptr : RegionName := "candidate_vbmm_97"
  let candidate_vmul_90_ptr : RegionName := "candidate_vmul_90"
  let candidate_vadd_48_ptr : RegionName := "candidate_vadd_48"
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
  let candidate_vbmm_100_ptr : RegionName := "candidate_vbmm_100"
  let candidate_vmul_94_ptr : RegionName := "candidate_vmul_94"
  let candidate_vto_copy_184_ptr : RegionName := "candidate_vto_copy_184"
  let candidate_vbmm_101_ptr : RegionName := "candidate_vbmm_101"
  let candidate_vto_copy_185_ptr : RegionName := "candidate_vto_copy_185"
  let candidate_vcat_51_ptr : RegionName := "candidate_vcat_51"
  let candidate_vcat_52_ptr : RegionName := "candidate_vcat_52"
  let candidate_vcat_54_ptr : RegionName := "candidate_vcat_54"
  let candidate_vcat_55_ptr : RegionName := "candidate_vcat_55"
  let candidate_vbmm_104_ptr : RegionName := "candidate_vbmm_104"
  let candidate_vbmm_105_ptr : RegionName := "candidate_vbmm_105"
  let candidate_vcat_56_ptr : RegionName := "candidate_vcat_56"
  let candidate_vadd_50_ptr : RegionName := "candidate_vadd_50"
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
    r69_row = tl.arange($(32))
    r69_col = tl.arange($(512))
    r69_lane = tl.arange($(128))
    for r69_group in range($(8)) {
      r69_left = tl.load($(candidate_vmm_1_ptr) + (((((((r69_group % $(8)) * $(4096)) + (r69_row[:, None] * $(128)) + r69_lane[None, :]) / $(128)) % $(32)) * $(1536)) + ((((((r69_group % $(8)) * $(4096)) + (r69_row[:, None] * $(128)) + r69_lane[None, :]) / $(4096)) % $(8)) * $(192)) + ((((r69_group % $(8)) * $(4096)) + (r69_row[:, None] * $(128)) + r69_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r69_right = tl.load($(candidate_vto_copy_20_ptr) + (((((((r69_group % $(8)) * $(65536)) + (r69_lane[:, None] * $(512)) + r69_col[None, :]) / $(65536)) % $(8)) * $(131072)) + ((((((r69_group % $(8)) * $(65536)) + (r69_lane[:, None] * $(512)) + r69_col[None, :]) / $(512)) % $(128)) * $(512)) + ((((r69_group % $(8)) * $(65536)) + (r69_lane[:, None] * $(512)) + r69_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r69_value = tl.dot(r69_left, r69_right)
      r69_value = r69_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_2_ptr) + ((r69_group * $(16384)) + (r69_row[:, None] * $(512)) + r69_col[None, :]), r69_value)
    }
    r70_z = tl.arange($(6144))
    r70_value = tl.load($(candidate_vadd_6_ptr) + ((((r70_z / $(64)) % $(12)) * $(512)) + (((r70_z / $(768)) % $(8)) * $(64)) + (r70_z % $(64))), dtype=tl.bfloat16)
    r70_value = r70_value.to(tl.float32)
    r70_value = tl.broadcast_to(r70_value, [$(6144)])
    tl.store($(candidate_vto_copy_30_ptr) + r70_z, r70_value)
    r71_z = tl.arange($(768))
    r71_value = tl.load($(candidate_vindex_2_ptr) + ((((r71_z / $(64)) % $(12)) * $(576)) + (r71_z % $(64)) + $(512)), dtype=tl.bfloat16)
    r71_value = r71_value.to(tl.float32)
    r71_value = tl.broadcast_to(r71_value, [$(768)])
    tl.store($(candidate_vto_copy_31_ptr) + r71_z, r71_value)
    r72_row = tl.arange($(12))
    r72_col = tl.arange($(12))
    r72_lane = tl.arange($(64))
    for r72_group in range($(8)) {
      r72_left = tl.load($(candidate_vto_copy_30_ptr) + (((((((r72_group % $(8)) * $(768)) + (r72_row[:, None] * $(64)) + r72_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r72_group % $(8)) * $(768)) + (r72_row[:, None] * $(64)) + r72_lane[None, :]) / $(64)) % $(12)) * $(64)) + ((((r72_group % $(8)) * $(768)) + (r72_row[:, None] * $(64)) + r72_lane[None, :]) % $(64))), dtype=tl.float32)
      r72_right = tl.load($(candidate_vto_copy_31_ptr) + ((((((r72_group % $(8)) * $(768)) + (r72_lane[:, None] * $(12)) + r72_col[None, :]) % $(12)) * $(64)) + (((((r72_group % $(8)) * $(768)) + (r72_lane[:, None] * $(12)) + r72_col[None, :]) / $(12)) % $(64))), dtype=tl.float32)
      r72_value = tl.dot(r72_left, r72_right)
      r72_value = r72_value.to(tl.float32)
      tl.store($(candidate_vbmm_3_ptr) + ((r72_group * $(144)) + (r72_row[:, None] * $(12)) + r72_col[None, :]), r72_value)
    }
    r73_z = tl.arange($(49152))
    r73_value = tl.load($(candidate_vbmm_2_ptr) + ((((r73_z / $(6144)) % $(8)) * $(16384)) + (((r73_z / $(512)) % $(12)) * $(512)) + (r73_z % $(512))), dtype=tl.bfloat16)
    r73_value = r73_value.to(tl.float32)
    r73_value = tl.broadcast_to(r73_value, [$(49152)])
    tl.store($(candidate_vto_copy_32_ptr) + r73_z, r73_value)
    r74_z = tl.arange($(6144))
    r74_value = tl.load($(candidate_vindex_2_ptr) + ((((r74_z / $(512)) % $(12)) * $(576)) + (r74_z % $(512))), dtype=tl.bfloat16)
    r74_value = r74_value.to(tl.float32)
    r74_value = tl.broadcast_to(r74_value, [$(6144)])
    tl.store($(candidate_vto_copy_33_ptr) + r74_z, r74_value)
    r75_row = tl.arange($(96))
    r75_col = tl.arange($(12))
    r75_lane = tl.arange($(512))
    r75_left = tl.load($(candidate_vto_copy_32_ptr) + ((((((r75_row[:, None] * $(512)) + r75_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r75_row[:, None] * $(512)) + r75_lane[None, :]) / $(512)) % $(12)) * $(512)) + (((r75_row[:, None] * $(512)) + r75_lane[None, :]) % $(512))), dtype=tl.float32)
    r75_right = tl.load($(candidate_vto_copy_33_ptr) + (((((r75_lane[:, None] * $(12)) + r75_col[None, :]) % $(12)) * $(512)) + ((((r75_lane[:, None] * $(12)) + r75_col[None, :]) / $(12)) % $(512))), dtype=tl.float32)
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
    tl.store($(candidate_vto_copy_34_ptr) + r83_z, r83_value)
    r84_row = tl.arange($(96))
    r84_col = tl.arange($(512))
    r84_lane = tl.arange($(12))
    r84_left = tl.load($(candidate_vto_copy_34_ptr) + ((((((r84_row[:, None] * $(12)) + r84_lane[None, :]) / $(144)) % $(8)) * $(144)) + (((((r84_row[:, None] * $(12)) + r84_lane[None, :]) / $(12)) % $(12)) * $(12)) + (((r84_row[:, None] * $(12)) + r84_lane[None, :]) % $(12))), dtype=tl.bfloat16)
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
    tl.store($(candidate_vto_copy_35_ptr) + r86_z, r86_value)
    r87_z = tl.arange($(256))
    r87_value = tl.load($(candidate_vindex_2_ptr) + ((((r87_z / $(64)) % $(4)) * $(576)) + (r87_z % $(64)) + $(28160)), dtype=tl.bfloat16)
    r87_value = r87_value.to(tl.float32)
    r87_value = tl.broadcast_to(r87_value, [$(256)])
    tl.store($(candidate_vto_copy_36_ptr) + r87_z, r87_value)
    r88_row = tl.arange($(4))
    r88_col = tl.arange($(4))
    r88_lane = tl.arange($(64))
    for r88_group in range($(8)) {
      r88_left = tl.load($(candidate_vto_copy_35_ptr) + (((((((r88_group % $(8)) * $(256)) + (r88_row[:, None] * $(64)) + r88_lane[None, :]) / $(256)) % $(8)) * $(256)) + ((((((r88_group % $(8)) * $(256)) + (r88_row[:, None] * $(64)) + r88_lane[None, :]) / $(64)) % $(4)) * $(64)) + ((((r88_group % $(8)) * $(256)) + (r88_row[:, None] * $(64)) + r88_lane[None, :]) % $(64))), dtype=tl.float32)
      r88_right = tl.load($(candidate_vto_copy_36_ptr) + ((((((r88_group % $(8)) * $(256)) + (r88_lane[:, None] * $(4)) + r88_col[None, :]) % $(4)) * $(64)) + (((((r88_group % $(8)) * $(256)) + (r88_lane[:, None] * $(4)) + r88_col[None, :]) / $(4)) % $(64))), dtype=tl.float32)
      r88_value = tl.dot(r88_left, r88_right)
      r88_value = r88_value.to(tl.float32)
      tl.store($(candidate_vbmm_4_ptr) + ((r88_group * $(16)) + (r88_row[:, None] * $(4)) + r88_col[None, :]), r88_value)
    }
    r89_z = tl.arange($(16384))
    r89_value = tl.load($(candidate_vbmm_2_ptr) + ((((r89_z / $(2048)) % $(8)) * $(16384)) + (((r89_z / $(512)) % $(4)) * $(512)) + (r89_z % $(512)) + $(6144)), dtype=tl.bfloat16)
    r89_value = r89_value.to(tl.float32)
    r89_value = tl.broadcast_to(r89_value, [$(16384)])
    tl.store($(candidate_vto_copy_37_ptr) + r89_z, r89_value)
    r90_z = tl.arange($(2048))
    r90_value = tl.load($(candidate_vindex_2_ptr) + ((((r90_z / $(512)) % $(4)) * $(576)) + (r90_z % $(512)) + $(27648)), dtype=tl.bfloat16)
    r90_value = r90_value.to(tl.float32)
    r90_value = tl.broadcast_to(r90_value, [$(2048)])
    tl.store($(candidate_vto_copy_38_ptr) + r90_z, r90_value)
    r91_row = tl.arange($(32))
    r91_col = tl.arange($(4))
    r91_lane = tl.arange($(512))
    r91_left = tl.load($(candidate_vto_copy_37_ptr) + ((((((r91_row[:, None] * $(512)) + r91_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + (((((r91_row[:, None] * $(512)) + r91_lane[None, :]) / $(512)) % $(4)) * $(512)) + (((r91_row[:, None] * $(512)) + r91_lane[None, :]) % $(512))), dtype=tl.float32)
    r91_right = tl.load($(candidate_vto_copy_38_ptr) + (((((r91_lane[:, None] * $(4)) + r91_col[None, :]) % $(4)) * $(512)) + ((((r91_lane[:, None] * $(4)) + r91_col[None, :]) / $(4)) % $(512))), dtype=tl.float32)
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
    tl.store($(candidate_vto_copy_39_ptr) + r99_z, r99_value)
    r100_row = tl.arange($(32))
    r100_col = tl.arange($(512))
    r100_lane = tl.arange($(4))
    r100_left = tl.load($(candidate_vto_copy_39_ptr) + ((((((r100_row[:, None] * $(4)) + r100_lane[None, :]) / $(16)) % $(8)) * $(16)) + (((((r100_row[:, None] * $(4)) + r100_lane[None, :]) / $(4)) % $(4)) * $(4)) + (((r100_row[:, None] * $(4)) + r100_lane[None, :]) % $(4))), dtype=tl.bfloat16)
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
    tl.store($(candidate_vto_copy_40_ptr) + r102_z, r102_value)
    r103_z = tl.arange($(1024))
    r103_value = tl.load($(candidate_vindex_2_ptr) + ((((r103_z / $(64)) % $(16)) * $(576)) + (r103_z % $(64)) + $(37376)), dtype=tl.bfloat16)
    r103_value = r103_value.to(tl.float32)
    r103_value = tl.broadcast_to(r103_value, [$(1024)])
    tl.store($(candidate_vto_copy_41_ptr) + r103_z, r103_value)
    r104_row = tl.arange($(16))
    r104_col = tl.arange($(16))
    r104_lane = tl.arange($(64))
    for r104_group in range($(8)) {
      r104_left = tl.load($(candidate_vto_copy_40_ptr) + (((((((r104_group % $(8)) * $(1024)) + (r104_row[:, None] * $(64)) + r104_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + ((((((r104_group % $(8)) * $(1024)) + (r104_row[:, None] * $(64)) + r104_lane[None, :]) / $(64)) % $(16)) * $(64)) + ((((r104_group % $(8)) * $(1024)) + (r104_row[:, None] * $(64)) + r104_lane[None, :]) % $(64))), dtype=tl.float32)
      r104_right = tl.load($(candidate_vto_copy_41_ptr) + ((((((r104_group % $(8)) * $(1024)) + (r104_lane[:, None] * $(16)) + r104_col[None, :]) % $(16)) * $(64)) + (((((r104_group % $(8)) * $(1024)) + (r104_lane[:, None] * $(16)) + r104_col[None, :]) / $(16)) % $(64))), dtype=tl.float32)
      r104_value = tl.dot(r104_left, r104_right)
      r104_value = r104_value.to(tl.float32)
      tl.store($(candidate_vbmm_5_ptr) + ((r104_group * $(256)) + (r104_row[:, None] * $(16)) + r104_col[None, :]), r104_value)
    }
    r105_z = tl.arange($(65536))
    r105_value = tl.load($(candidate_vbmm_2_ptr) + ((((r105_z / $(8192)) % $(8)) * $(16384)) + (((r105_z / $(512)) % $(16)) * $(512)) + (r105_z % $(512)) + $(8192)), dtype=tl.bfloat16)
    r105_value = r105_value.to(tl.float32)
    r105_value = tl.broadcast_to(r105_value, [$(65536)])
    tl.store($(candidate_vto_copy_42_ptr) + r105_z, r105_value)
    r106_z = tl.arange($(8192))
    r106_value = tl.load($(candidate_vindex_2_ptr) + ((((r106_z / $(512)) % $(16)) * $(576)) + (r106_z % $(512)) + $(36864)), dtype=tl.bfloat16)
    r106_value = r106_value.to(tl.float32)
    r106_value = tl.broadcast_to(r106_value, [$(8192)])
    tl.store($(candidate_vto_copy_43_ptr) + r106_z, r106_value)
    r107_row = tl.arange($(128))
    r107_col = tl.arange($(16))
    r107_lane = tl.arange($(512))
    r107_left = tl.load($(candidate_vto_copy_42_ptr) + ((((((r107_row[:, None] * $(512)) + r107_lane[None, :]) / $(8192)) % $(8)) * $(8192)) + (((((r107_row[:, None] * $(512)) + r107_lane[None, :]) / $(512)) % $(16)) * $(512)) + (((r107_row[:, None] * $(512)) + r107_lane[None, :]) % $(512))), dtype=tl.float32)
    r107_right = tl.load($(candidate_vto_copy_43_ptr) + (((((r107_lane[:, None] * $(16)) + r107_col[None, :]) % $(16)) * $(512)) + ((((r107_lane[:, None] * $(16)) + r107_col[None, :]) / $(16)) % $(512))), dtype=tl.float32)
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
    tl.store($(candidate_vto_copy_44_ptr) + r115_z, r115_value)
    r116_row = tl.arange($(128))
    r116_col = tl.arange($(512))
    r116_lane = tl.arange($(16))
    r116_left = tl.load($(candidate_vto_copy_44_ptr) + ((((((r116_row[:, None] * $(16)) + r116_lane[None, :]) / $(256)) % $(8)) * $(256)) + (((((r116_row[:, None] * $(16)) + r116_lane[None, :]) / $(16)) % $(16)) * $(16)) + (((r116_row[:, None] * $(16)) + r116_lane[None, :]) % $(16))), dtype=tl.bfloat16)
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
      r120_right = tl.load($(candidate_vto_copy_20_ptr) + (((((((r120_group % $(8)) * $(65536)) + (r120_lane[:, None] * $(128)) + r120_col[None, :]) / $(65536)) % $(8)) * $(131072)) + (((((r120_group % $(8)) * $(65536)) + (r120_lane[:, None] * $(128)) + r120_col[None, :]) % $(128)) * $(512)) + (((((r120_group % $(8)) * $(65536)) + (r120_lane[:, None] * $(128)) + r120_col[None, :]) / $(128)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r120_value = tl.dot(r120_left, r120_right)
      r120_value = r120_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_6_ptr) + ((r120_group * $(4096)) + (r120_row[:, None] * $(128)) + r120_col[None, :]), r120_value)
    }
    r121_row = tl.arange($(32))
    r121_col = tl.arange($(512))
    r121_lane = tl.arange($(128))
    for r121_group in range($(8)) {
      r121_left = tl.load($(candidate_vmm_1_ptr) + (((((((r121_group % $(8)) * $(4096)) + (r121_row[:, None] * $(128)) + r121_lane[None, :]) / $(128)) % $(32)) * $(1536)) + ((((((r121_group % $(8)) * $(4096)) + (r121_row[:, None] * $(128)) + r121_lane[None, :]) / $(4096)) % $(8)) * $(192)) + ((((r121_group % $(8)) * $(4096)) + (r121_row[:, None] * $(128)) + r121_lane[None, :]) % $(128)) + $(49152)), dtype=tl.bfloat16)
      r121_right = tl.load($(candidate_vto_copy_20_ptr) + (((((((r121_group % $(8)) * $(65536)) + (r121_lane[:, None] * $(512)) + r121_col[None, :]) / $(65536)) % $(8)) * $(131072)) + ((((((r121_group % $(8)) * $(65536)) + (r121_lane[:, None] * $(512)) + r121_col[None, :]) / $(512)) % $(128)) * $(512)) + ((((r121_group % $(8)) * $(65536)) + (r121_lane[:, None] * $(512)) + r121_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r121_value = tl.dot(r121_left, r121_right)
      r121_value = r121_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_7_ptr) + ((r121_group * $(16384)) + (r121_row[:, None] * $(512)) + r121_col[None, :]), r121_value)
    }
    r122_z = tl.arange($(6144))
    r122_value = tl.load($(candidate_vadd_6_ptr) + ((((r122_z / $(64)) % $(12)) * $(512)) + (((r122_z / $(768)) % $(8)) * $(64)) + (r122_z % $(64)) + $(16384)), dtype=tl.bfloat16)
    r122_value = r122_value.to(tl.float32)
    r122_value = tl.broadcast_to(r122_value, [$(6144)])
    tl.store($(candidate_vto_copy_45_ptr) + r122_z, r122_value)
    r123_z = tl.arange($(3072))
    r123_value = tl.load($(candidate_vindex_2_ptr) + ((((r123_z / $(64)) % $(48)) * $(576)) + (r123_z % $(64)) + $(512)), dtype=tl.bfloat16)
    r123_value = r123_value.to(tl.float32)
    r123_value = tl.broadcast_to(r123_value, [$(3072)])
    tl.store($(candidate_vto_copy_46_ptr) + r123_z, r123_value)
    r124_row = tl.arange($(12))
    r124_col = tl.arange($(48))
    r124_lane = tl.arange($(64))
    for r124_group in range($(8)) {
      r124_left = tl.load($(candidate_vto_copy_45_ptr) + (((((((r124_group % $(8)) * $(768)) + (r124_row[:, None] * $(64)) + r124_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r124_group % $(8)) * $(768)) + (r124_row[:, None] * $(64)) + r124_lane[None, :]) / $(64)) % $(12)) * $(64)) + ((((r124_group % $(8)) * $(768)) + (r124_row[:, None] * $(64)) + r124_lane[None, :]) % $(64))), dtype=tl.float32)
      r124_right = tl.load($(candidate_vto_copy_46_ptr) + ((((((r124_group % $(8)) * $(3072)) + (r124_lane[:, None] * $(48)) + r124_col[None, :]) % $(48)) * $(64)) + (((((r124_group % $(8)) * $(3072)) + (r124_lane[:, None] * $(48)) + r124_col[None, :]) / $(48)) % $(64))), dtype=tl.float32)
      r124_value = tl.dot(r124_left, r124_right)
      r124_value = r124_value.to(tl.float32)
      tl.store($(candidate_vbmm_8_ptr) + ((r124_group * $(576)) + (r124_row[:, None] * $(48)) + r124_col[None, :]), r124_value)
    }
    r125_z = tl.arange($(49152))
    r125_value = tl.load($(candidate_vbmm_7_ptr) + ((((r125_z / $(6144)) % $(8)) * $(16384)) + (((r125_z / $(512)) % $(12)) * $(512)) + (r125_z % $(512))), dtype=tl.bfloat16)
    r125_value = r125_value.to(tl.float32)
    r125_value = tl.broadcast_to(r125_value, [$(49152)])
    tl.store($(candidate_vto_copy_47_ptr) + r125_z, r125_value)
    r126_z = tl.arange($(24576))
    r126_value = tl.load($(candidate_vindex_2_ptr) + ((((r126_z / $(512)) % $(48)) * $(576)) + (r126_z % $(512))), dtype=tl.bfloat16)
    r126_value = r126_value.to(tl.float32)
    r126_value = tl.broadcast_to(r126_value, [$(24576)])
    tl.store($(candidate_vto_copy_48_ptr) + r126_z, r126_value)
    r127_row = tl.arange($(96))
    r127_col = tl.arange($(48))
    r127_lane = tl.arange($(512))
    r127_left = tl.load($(candidate_vto_copy_47_ptr) + ((((((r127_row[:, None] * $(512)) + r127_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r127_row[:, None] * $(512)) + r127_lane[None, :]) / $(512)) % $(12)) * $(512)) + (((r127_row[:, None] * $(512)) + r127_lane[None, :]) % $(512))), dtype=tl.float32)
    r127_right = tl.load($(candidate_vto_copy_48_ptr) + (((((r127_lane[:, None] * $(48)) + r127_col[None, :]) % $(48)) * $(512)) + ((((r127_lane[:, None] * $(48)) + r127_col[None, :]) / $(48)) % $(512))), dtype=tl.float32)
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
    tl.store($(candidate_vto_copy_49_ptr) + r135_z, r135_value)
    r136_row = tl.arange($(96))
    r136_col = tl.arange($(512))
    r136_lane = tl.arange($(48))
    r136_left = tl.load($(candidate_vto_copy_49_ptr) + ((((((r136_row[:, None] * $(48)) + r136_lane[None, :]) / $(576)) % $(8)) * $(576)) + (((((r136_row[:, None] * $(48)) + r136_lane[None, :]) / $(48)) % $(12)) * $(48)) + (((r136_row[:, None] * $(48)) + r136_lane[None, :]) % $(48))), dtype=tl.bfloat16)
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
    tl.store($(candidate_vto_copy_50_ptr) + r138_z, r138_value)
    r139_z = tl.arange($(1024))
    r139_value = tl.load($(candidate_vindex_2_ptr) + ((((r139_z / $(64)) % $(16)) * $(576)) + (r139_z % $(64)) + $(28160)), dtype=tl.bfloat16)
    r139_value = r139_value.to(tl.float32)
    r139_value = tl.broadcast_to(r139_value, [$(1024)])
    tl.store($(candidate_vto_copy_51_ptr) + r139_z, r139_value)
    r140_row = tl.arange($(4))
    r140_col = tl.arange($(16))
    r140_lane = tl.arange($(64))
    for r140_group in range($(8)) {
      r140_left = tl.load($(candidate_vto_copy_50_ptr) + (((((((r140_group % $(8)) * $(256)) + (r140_row[:, None] * $(64)) + r140_lane[None, :]) / $(256)) % $(8)) * $(256)) + ((((((r140_group % $(8)) * $(256)) + (r140_row[:, None] * $(64)) + r140_lane[None, :]) / $(64)) % $(4)) * $(64)) + ((((r140_group % $(8)) * $(256)) + (r140_row[:, None] * $(64)) + r140_lane[None, :]) % $(64))), dtype=tl.float32)
      r140_right = tl.load($(candidate_vto_copy_51_ptr) + ((((((r140_group % $(8)) * $(1024)) + (r140_lane[:, None] * $(16)) + r140_col[None, :]) % $(16)) * $(64)) + (((((r140_group % $(8)) * $(1024)) + (r140_lane[:, None] * $(16)) + r140_col[None, :]) / $(16)) % $(64))), dtype=tl.float32)
      r140_value = tl.dot(r140_left, r140_right)
      r140_value = r140_value.to(tl.float32)
      tl.store($(candidate_vbmm_9_ptr) + ((r140_group * $(64)) + (r140_row[:, None] * $(16)) + r140_col[None, :]), r140_value)
    }
    r141_z = tl.arange($(16384))
    r141_value = tl.load($(candidate_vbmm_7_ptr) + ((((r141_z / $(2048)) % $(8)) * $(16384)) + (((r141_z / $(512)) % $(4)) * $(512)) + (r141_z % $(512)) + $(6144)), dtype=tl.bfloat16)
    r141_value = r141_value.to(tl.float32)
    r141_value = tl.broadcast_to(r141_value, [$(16384)])
    tl.store($(candidate_vto_copy_52_ptr) + r141_z, r141_value)
    r142_z = tl.arange($(8192))
    r142_value = tl.load($(candidate_vindex_2_ptr) + ((((r142_z / $(512)) % $(16)) * $(576)) + (r142_z % $(512)) + $(27648)), dtype=tl.bfloat16)
    r142_value = r142_value.to(tl.float32)
    r142_value = tl.broadcast_to(r142_value, [$(8192)])
    tl.store($(candidate_vto_copy_53_ptr) + r142_z, r142_value)
    r143_row = tl.arange($(32))
    r143_col = tl.arange($(16))
    r143_lane = tl.arange($(512))
    r143_left = tl.load($(candidate_vto_copy_52_ptr) + ((((((r143_row[:, None] * $(512)) + r143_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + (((((r143_row[:, None] * $(512)) + r143_lane[None, :]) / $(512)) % $(4)) * $(512)) + (((r143_row[:, None] * $(512)) + r143_lane[None, :]) % $(512))), dtype=tl.float32)
    r143_right = tl.load($(candidate_vto_copy_53_ptr) + (((((r143_lane[:, None] * $(16)) + r143_col[None, :]) % $(16)) * $(512)) + ((((r143_lane[:, None] * $(16)) + r143_col[None, :]) / $(16)) % $(512))), dtype=tl.float32)
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
    tl.store($(candidate_vto_copy_54_ptr) + r151_z, r151_value)
    r152_row = tl.arange($(32))
    r152_col = tl.arange($(512))
    r152_lane = tl.arange($(16))
    r152_left = tl.load($(candidate_vto_copy_54_ptr) + ((((((r152_row[:, None] * $(16)) + r152_lane[None, :]) / $(64)) % $(8)) * $(64)) + (((((r152_row[:, None] * $(16)) + r152_lane[None, :]) / $(16)) % $(4)) * $(16)) + (((r152_row[:, None] * $(16)) + r152_lane[None, :]) % $(16))), dtype=tl.bfloat16)
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
    tl.store($(candidate_vto_copy_55_ptr) + r154_z, r154_value)
    r155_z = tl.arange($(4096))
    r155_value = tl.load($(candidate_vindex_2_ptr) + ((((r155_z / $(64)) % $(64)) * $(576)) + (r155_z % $(64)) + $(37376)), dtype=tl.bfloat16)
    r155_value = r155_value.to(tl.float32)
    r155_value = tl.broadcast_to(r155_value, [$(4096)])
    tl.store($(candidate_vto_copy_56_ptr) + r155_z, r155_value)
    r156_row = tl.arange($(16))
    r156_col = tl.arange($(64))
    r156_lane = tl.arange($(64))
    for r156_group in range($(8)) {
      r156_left = tl.load($(candidate_vto_copy_55_ptr) + (((((((r156_group % $(8)) * $(1024)) + (r156_row[:, None] * $(64)) + r156_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + ((((((r156_group % $(8)) * $(1024)) + (r156_row[:, None] * $(64)) + r156_lane[None, :]) / $(64)) % $(16)) * $(64)) + ((((r156_group % $(8)) * $(1024)) + (r156_row[:, None] * $(64)) + r156_lane[None, :]) % $(64))), dtype=tl.float32)
      r156_right = tl.load($(candidate_vto_copy_56_ptr) + ((((((r156_group % $(8)) * $(4096)) + (r156_lane[:, None] * $(64)) + r156_col[None, :]) % $(64)) * $(64)) + (((((r156_group % $(8)) * $(4096)) + (r156_lane[:, None] * $(64)) + r156_col[None, :]) / $(64)) % $(64))), dtype=tl.float32)
      r156_value = tl.dot(r156_left, r156_right)
      r156_value = r156_value.to(tl.float32)
      tl.store($(candidate_vbmm_10_ptr) + ((r156_group * $(1024)) + (r156_row[:, None] * $(64)) + r156_col[None, :]), r156_value)
    }
    r157_z = tl.arange($(65536))
    r157_value = tl.load($(candidate_vbmm_7_ptr) + ((((r157_z / $(8192)) % $(8)) * $(16384)) + (((r157_z / $(512)) % $(16)) * $(512)) + (r157_z % $(512)) + $(8192)), dtype=tl.bfloat16)
    r157_value = r157_value.to(tl.float32)
    r157_value = tl.broadcast_to(r157_value, [$(65536)])
    tl.store($(candidate_vto_copy_57_ptr) + r157_z, r157_value)
    r158_z = tl.arange($(32768))
    r158_value = tl.load($(candidate_vindex_2_ptr) + ((((r158_z / $(512)) % $(64)) * $(576)) + (r158_z % $(512)) + $(36864)), dtype=tl.bfloat16)
    r158_value = r158_value.to(tl.float32)
    r158_value = tl.broadcast_to(r158_value, [$(32768)])
    tl.store($(candidate_vto_copy_58_ptr) + r158_z, r158_value)
    r159_row = tl.arange($(128))
    r159_col = tl.arange($(64))
    r159_lane = tl.arange($(512))
    r159_left = tl.load($(candidate_vto_copy_57_ptr) + ((((((r159_row[:, None] * $(512)) + r159_lane[None, :]) / $(8192)) % $(8)) * $(8192)) + (((((r159_row[:, None] * $(512)) + r159_lane[None, :]) / $(512)) % $(16)) * $(512)) + (((r159_row[:, None] * $(512)) + r159_lane[None, :]) % $(512))), dtype=tl.float32)
    r159_right = tl.load($(candidate_vto_copy_58_ptr) + (((((r159_lane[:, None] * $(64)) + r159_col[None, :]) % $(64)) * $(512)) + ((((r159_lane[:, None] * $(64)) + r159_col[None, :]) / $(64)) % $(512))), dtype=tl.float32)
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
    tl.store($(candidate_vto_copy_59_ptr) + r167_z, r167_value)
    r168_row = tl.arange($(128))
    r168_col = tl.arange($(512))
    r168_lane = tl.arange($(64))
    r168_left = tl.load($(candidate_vto_copy_59_ptr) + ((((((r168_row[:, None] * $(64)) + r168_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + (((((r168_row[:, None] * $(64)) + r168_lane[None, :]) / $(64)) % $(16)) * $(64)) + (((r168_row[:, None] * $(64)) + r168_lane[None, :]) % $(64))), dtype=tl.bfloat16)
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
      r172_right = tl.load($(candidate_vto_copy_20_ptr) + (((((((r172_group % $(8)) * $(65536)) + (r172_lane[:, None] * $(128)) + r172_col[None, :]) / $(65536)) % $(8)) * $(131072)) + (((((r172_group % $(8)) * $(65536)) + (r172_lane[:, None] * $(128)) + r172_col[None, :]) % $(128)) * $(512)) + (((((r172_group % $(8)) * $(65536)) + (r172_lane[:, None] * $(128)) + r172_col[None, :]) / $(128)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r172_value = tl.dot(r172_left, r172_right)
      r172_value = r172_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_11_ptr) + ((r172_group * $(4096)) + (r172_row[:, None] * $(128)) + r172_col[None, :]), r172_value)
    }
    r173_row = tl.arange($(32))
    r173_col = tl.arange($(512))
    r173_lane = tl.arange($(128))
    for r173_group in range($(8)) {
      r173_left = tl.load($(candidate_vmm_1_ptr) + (((((((r173_group % $(8)) * $(4096)) + (r173_row[:, None] * $(128)) + r173_lane[None, :]) / $(128)) % $(32)) * $(1536)) + ((((((r173_group % $(8)) * $(4096)) + (r173_row[:, None] * $(128)) + r173_lane[None, :]) / $(4096)) % $(8)) * $(192)) + ((((r173_group % $(8)) * $(4096)) + (r173_row[:, None] * $(128)) + r173_lane[None, :]) % $(128)) + $(98304)), dtype=tl.bfloat16)
      r173_right = tl.load($(candidate_vto_copy_20_ptr) + (((((((r173_group % $(8)) * $(65536)) + (r173_lane[:, None] * $(512)) + r173_col[None, :]) / $(65536)) % $(8)) * $(131072)) + ((((((r173_group % $(8)) * $(65536)) + (r173_lane[:, None] * $(512)) + r173_col[None, :]) / $(512)) % $(128)) * $(512)) + ((((r173_group % $(8)) * $(65536)) + (r173_lane[:, None] * $(512)) + r173_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r173_value = tl.dot(r173_left, r173_right)
      r173_value = r173_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_12_ptr) + ((r173_group * $(16384)) + (r173_row[:, None] * $(512)) + r173_col[None, :]), r173_value)
    }
    r174_z = tl.arange($(6144))
    r174_value = tl.load($(candidate_vadd_6_ptr) + ((((r174_z / $(64)) % $(12)) * $(512)) + (((r174_z / $(768)) % $(8)) * $(64)) + (r174_z % $(64)) + $(32768)), dtype=tl.bfloat16)
    r174_value = r174_value.to(tl.float32)
    r174_value = tl.broadcast_to(r174_value, [$(6144)])
    tl.store($(candidate_vto_copy_60_ptr) + r174_z, r174_value)
    r175_z = tl.arange($(1536))
    r175_value = tl.load($(candidate_vindex_2_ptr) + ((((r175_z / $(64)) % $(24)) * $(576)) + (r175_z % $(64)) + $(512)), dtype=tl.bfloat16)
    r175_value = r175_value.to(tl.float32)
    r175_value = tl.broadcast_to(r175_value, [$(1536)])
    tl.store($(candidate_vto_copy_61_ptr) + r175_z, r175_value)
    r176_row = tl.arange($(12))
    r176_col = tl.arange($(24))
    r176_lane = tl.arange($(64))
    for r176_group in range($(8)) {
      r176_left = tl.load($(candidate_vto_copy_60_ptr) + (((((((r176_group % $(8)) * $(768)) + (r176_row[:, None] * $(64)) + r176_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r176_group % $(8)) * $(768)) + (r176_row[:, None] * $(64)) + r176_lane[None, :]) / $(64)) % $(12)) * $(64)) + ((((r176_group % $(8)) * $(768)) + (r176_row[:, None] * $(64)) + r176_lane[None, :]) % $(64))), dtype=tl.float32)
      r176_right = tl.load($(candidate_vto_copy_61_ptr) + ((((((r176_group % $(8)) * $(1536)) + (r176_lane[:, None] * $(24)) + r176_col[None, :]) % $(24)) * $(64)) + (((((r176_group % $(8)) * $(1536)) + (r176_lane[:, None] * $(24)) + r176_col[None, :]) / $(24)) % $(64))), dtype=tl.float32)
      r176_value = tl.dot(r176_left, r176_right)
      r176_value = r176_value.to(tl.float32)
      tl.store($(candidate_vbmm_13_ptr) + ((r176_group * $(288)) + (r176_row[:, None] * $(24)) + r176_col[None, :]), r176_value)
    }
    r177_z = tl.arange($(49152))
    r177_value = tl.load($(candidate_vbmm_12_ptr) + ((((r177_z / $(6144)) % $(8)) * $(16384)) + (((r177_z / $(512)) % $(12)) * $(512)) + (r177_z % $(512))), dtype=tl.bfloat16)
    r177_value = r177_value.to(tl.float32)
    r177_value = tl.broadcast_to(r177_value, [$(49152)])
    tl.store($(candidate_vto_copy_62_ptr) + r177_z, r177_value)
    r178_z = tl.arange($(12288))
    r178_value = tl.load($(candidate_vindex_2_ptr) + ((((r178_z / $(512)) % $(24)) * $(576)) + (r178_z % $(512))), dtype=tl.bfloat16)
    r178_value = r178_value.to(tl.float32)
    r178_value = tl.broadcast_to(r178_value, [$(12288)])
    tl.store($(candidate_vto_copy_63_ptr) + r178_z, r178_value)
    r179_row = tl.arange($(96))
    r179_col = tl.arange($(24))
    r179_lane = tl.arange($(512))
    r179_left = tl.load($(candidate_vto_copy_62_ptr) + ((((((r179_row[:, None] * $(512)) + r179_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r179_row[:, None] * $(512)) + r179_lane[None, :]) / $(512)) % $(12)) * $(512)) + (((r179_row[:, None] * $(512)) + r179_lane[None, :]) % $(512))), dtype=tl.float32)
    r179_right = tl.load($(candidate_vto_copy_63_ptr) + (((((r179_lane[:, None] * $(24)) + r179_col[None, :]) % $(24)) * $(512)) + ((((r179_lane[:, None] * $(24)) + r179_col[None, :]) / $(24)) % $(512))), dtype=tl.float32)
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
    tl.store($(candidate_vto_copy_64_ptr) + r187_z, r187_value)
    r188_row = tl.arange($(96))
    r188_col = tl.arange($(512))
    r188_lane = tl.arange($(24))
    r188_left = tl.load($(candidate_vto_copy_64_ptr) + ((((((r188_row[:, None] * $(24)) + r188_lane[None, :]) / $(288)) % $(8)) * $(288)) + (((((r188_row[:, None] * $(24)) + r188_lane[None, :]) / $(24)) % $(12)) * $(24)) + (((r188_row[:, None] * $(24)) + r188_lane[None, :]) % $(24))), dtype=tl.bfloat16)
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
    tl.store($(candidate_vto_copy_65_ptr) + r190_z, r190_value)
    r191_z = tl.arange($(512))
    r191_value = tl.load($(candidate_vindex_2_ptr) + ((((r191_z / $(64)) % $(8)) * $(576)) + (r191_z % $(64)) + $(28160)), dtype=tl.bfloat16)
    r191_value = r191_value.to(tl.float32)
    r191_value = tl.broadcast_to(r191_value, [$(512)])
    tl.store($(candidate_vto_copy_66_ptr) + r191_z, r191_value)
    r192_row = tl.arange($(4))
    r192_col = tl.arange($(8))
    r192_lane = tl.arange($(64))
    for r192_group in range($(8)) {
      r192_left = tl.load($(candidate_vto_copy_65_ptr) + (((((((r192_group % $(8)) * $(256)) + (r192_row[:, None] * $(64)) + r192_lane[None, :]) / $(256)) % $(8)) * $(256)) + ((((((r192_group % $(8)) * $(256)) + (r192_row[:, None] * $(64)) + r192_lane[None, :]) / $(64)) % $(4)) * $(64)) + ((((r192_group % $(8)) * $(256)) + (r192_row[:, None] * $(64)) + r192_lane[None, :]) % $(64))), dtype=tl.float32)
      r192_right = tl.load($(candidate_vto_copy_66_ptr) + ((((((r192_group % $(8)) * $(512)) + (r192_lane[:, None] * $(8)) + r192_col[None, :]) % $(8)) * $(64)) + (((((r192_group % $(8)) * $(512)) + (r192_lane[:, None] * $(8)) + r192_col[None, :]) / $(8)) % $(64))), dtype=tl.float32)
      r192_value = tl.dot(r192_left, r192_right)
      r192_value = r192_value.to(tl.float32)
      tl.store($(candidate_vbmm_14_ptr) + ((r192_group * $(32)) + (r192_row[:, None] * $(8)) + r192_col[None, :]), r192_value)
    }
    r193_z = tl.arange($(16384))
    r193_value = tl.load($(candidate_vbmm_12_ptr) + ((((r193_z / $(2048)) % $(8)) * $(16384)) + (((r193_z / $(512)) % $(4)) * $(512)) + (r193_z % $(512)) + $(6144)), dtype=tl.bfloat16)
    r193_value = r193_value.to(tl.float32)
    r193_value = tl.broadcast_to(r193_value, [$(16384)])
    tl.store($(candidate_vto_copy_67_ptr) + r193_z, r193_value)
    r194_z = tl.arange($(4096))
    r194_value = tl.load($(candidate_vindex_2_ptr) + ((((r194_z / $(512)) % $(8)) * $(576)) + (r194_z % $(512)) + $(27648)), dtype=tl.bfloat16)
    r194_value = r194_value.to(tl.float32)
    r194_value = tl.broadcast_to(r194_value, [$(4096)])
    tl.store($(candidate_vto_copy_68_ptr) + r194_z, r194_value)
    r195_row = tl.arange($(32))
    r195_col = tl.arange($(8))
    r195_lane = tl.arange($(512))
    r195_left = tl.load($(candidate_vto_copy_67_ptr) + ((((((r195_row[:, None] * $(512)) + r195_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + (((((r195_row[:, None] * $(512)) + r195_lane[None, :]) / $(512)) % $(4)) * $(512)) + (((r195_row[:, None] * $(512)) + r195_lane[None, :]) % $(512))), dtype=tl.float32)
    r195_right = tl.load($(candidate_vto_copy_68_ptr) + (((((r195_lane[:, None] * $(8)) + r195_col[None, :]) % $(8)) * $(512)) + ((((r195_lane[:, None] * $(8)) + r195_col[None, :]) / $(8)) % $(512))), dtype=tl.float32)
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
    tl.store($(candidate_vto_copy_69_ptr) + r203_z, r203_value)
    r204_row = tl.arange($(32))
    r204_col = tl.arange($(512))
    r204_lane = tl.arange($(8))
    r204_left = tl.load($(candidate_vto_copy_69_ptr) + ((((((r204_row[:, None] * $(8)) + r204_lane[None, :]) / $(32)) % $(8)) * $(32)) + (((((r204_row[:, None] * $(8)) + r204_lane[None, :]) / $(8)) % $(4)) * $(8)) + (((r204_row[:, None] * $(8)) + r204_lane[None, :]) % $(8))), dtype=tl.bfloat16)
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
    tl.store($(candidate_vto_copy_70_ptr) + r206_z, r206_value)
    r207_z = tl.arange($(2048))
    r207_value = tl.load($(candidate_vindex_2_ptr) + ((((r207_z / $(64)) % $(32)) * $(576)) + (r207_z % $(64)) + $(37376)), dtype=tl.bfloat16)
    r207_value = r207_value.to(tl.float32)
    r207_value = tl.broadcast_to(r207_value, [$(2048)])
    tl.store($(candidate_vto_copy_71_ptr) + r207_z, r207_value)
    r208_row = tl.arange($(16))
    r208_col = tl.arange($(32))
    r208_lane = tl.arange($(64))
    for r208_group in range($(8)) {
      r208_left = tl.load($(candidate_vto_copy_70_ptr) + (((((((r208_group % $(8)) * $(1024)) + (r208_row[:, None] * $(64)) + r208_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + ((((((r208_group % $(8)) * $(1024)) + (r208_row[:, None] * $(64)) + r208_lane[None, :]) / $(64)) % $(16)) * $(64)) + ((((r208_group % $(8)) * $(1024)) + (r208_row[:, None] * $(64)) + r208_lane[None, :]) % $(64))), dtype=tl.float32)
      r208_right = tl.load($(candidate_vto_copy_71_ptr) + ((((((r208_group % $(8)) * $(2048)) + (r208_lane[:, None] * $(32)) + r208_col[None, :]) % $(32)) * $(64)) + (((((r208_group % $(8)) * $(2048)) + (r208_lane[:, None] * $(32)) + r208_col[None, :]) / $(32)) % $(64))), dtype=tl.float32)
      r208_value = tl.dot(r208_left, r208_right)
      r208_value = r208_value.to(tl.float32)
      tl.store($(candidate_vbmm_15_ptr) + ((r208_group * $(512)) + (r208_row[:, None] * $(32)) + r208_col[None, :]), r208_value)
    }
    r209_z = tl.arange($(65536))
    r209_value = tl.load($(candidate_vbmm_12_ptr) + ((((r209_z / $(8192)) % $(8)) * $(16384)) + (((r209_z / $(512)) % $(16)) * $(512)) + (r209_z % $(512)) + $(8192)), dtype=tl.bfloat16)
    r209_value = r209_value.to(tl.float32)
    r209_value = tl.broadcast_to(r209_value, [$(65536)])
    tl.store($(candidate_vto_copy_72_ptr) + r209_z, r209_value)
    r210_z = tl.arange($(16384))
    r210_value = tl.load($(candidate_vindex_2_ptr) + ((((r210_z / $(512)) % $(32)) * $(576)) + (r210_z % $(512)) + $(36864)), dtype=tl.bfloat16)
    r210_value = r210_value.to(tl.float32)
    r210_value = tl.broadcast_to(r210_value, [$(16384)])
    tl.store($(candidate_vto_copy_73_ptr) + r210_z, r210_value)
    r211_row = tl.arange($(128))
    r211_col = tl.arange($(32))
    r211_lane = tl.arange($(512))
    r211_left = tl.load($(candidate_vto_copy_72_ptr) + ((((((r211_row[:, None] * $(512)) + r211_lane[None, :]) / $(8192)) % $(8)) * $(8192)) + (((((r211_row[:, None] * $(512)) + r211_lane[None, :]) / $(512)) % $(16)) * $(512)) + (((r211_row[:, None] * $(512)) + r211_lane[None, :]) % $(512))), dtype=tl.float32)
    r211_right = tl.load($(candidate_vto_copy_73_ptr) + (((((r211_lane[:, None] * $(32)) + r211_col[None, :]) % $(32)) * $(512)) + ((((r211_lane[:, None] * $(32)) + r211_col[None, :]) / $(32)) % $(512))), dtype=tl.float32)
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
    tl.store($(candidate_vto_copy_74_ptr) + r219_z, r219_value)
    r220_row = tl.arange($(128))
    r220_col = tl.arange($(512))
    r220_lane = tl.arange($(32))
    r220_left = tl.load($(candidate_vto_copy_74_ptr) + ((((((r220_row[:, None] * $(32)) + r220_lane[None, :]) / $(512)) % $(8)) * $(512)) + (((((r220_row[:, None] * $(32)) + r220_lane[None, :]) / $(32)) % $(16)) * $(32)) + (((r220_row[:, None] * $(32)) + r220_lane[None, :]) % $(32))), dtype=tl.bfloat16)
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
      r224_right = tl.load($(candidate_vto_copy_20_ptr) + (((((((r224_group % $(8)) * $(65536)) + (r224_lane[:, None] * $(128)) + r224_col[None, :]) / $(65536)) % $(8)) * $(131072)) + (((((r224_group % $(8)) * $(65536)) + (r224_lane[:, None] * $(128)) + r224_col[None, :]) % $(128)) * $(512)) + (((((r224_group % $(8)) * $(65536)) + (r224_lane[:, None] * $(128)) + r224_col[None, :]) / $(128)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r224_value = tl.dot(r224_left, r224_right)
      r224_value = r224_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_16_ptr) + ((r224_group * $(4096)) + (r224_row[:, None] * $(128)) + r224_col[None, :]), r224_value)
    }
    r225_row = tl.arange($(32))
    r225_col = tl.arange($(512))
    r225_lane = tl.arange($(128))
    for r225_group in range($(8)) {
      r225_left = tl.load($(candidate_vmm_1_ptr) + (((((((r225_group % $(8)) * $(4096)) + (r225_row[:, None] * $(128)) + r225_lane[None, :]) / $(128)) % $(32)) * $(1536)) + ((((((r225_group % $(8)) * $(4096)) + (r225_row[:, None] * $(128)) + r225_lane[None, :]) / $(4096)) % $(8)) * $(192)) + ((((r225_group % $(8)) * $(4096)) + (r225_row[:, None] * $(128)) + r225_lane[None, :]) % $(128)) + $(147456)), dtype=tl.bfloat16)
      r225_right = tl.load($(candidate_vto_copy_20_ptr) + (((((((r225_group % $(8)) * $(65536)) + (r225_lane[:, None] * $(512)) + r225_col[None, :]) / $(65536)) % $(8)) * $(131072)) + ((((((r225_group % $(8)) * $(65536)) + (r225_lane[:, None] * $(512)) + r225_col[None, :]) / $(512)) % $(128)) * $(512)) + ((((r225_group % $(8)) * $(65536)) + (r225_lane[:, None] * $(512)) + r225_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r225_value = tl.dot(r225_left, r225_right)
      r225_value = r225_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_17_ptr) + ((r225_group * $(16384)) + (r225_row[:, None] * $(512)) + r225_col[None, :]), r225_value)
    }
    r226_z = tl.arange($(6144))
    r226_value = tl.load($(candidate_vadd_6_ptr) + ((((r226_z / $(64)) % $(12)) * $(512)) + (((r226_z / $(768)) % $(8)) * $(64)) + (r226_z % $(64)) + $(49152)), dtype=tl.bfloat16)
    r226_value = r226_value.to(tl.float32)
    r226_value = tl.broadcast_to(r226_value, [$(6144)])
    tl.store($(candidate_vto_copy_75_ptr) + r226_z, r226_value)
    r227_z = tl.arange($(2304))
    r227_value = tl.load($(candidate_vindex_2_ptr) + ((((r227_z / $(64)) % $(36)) * $(576)) + (r227_z % $(64)) + $(512)), dtype=tl.bfloat16)
    r227_value = r227_value.to(tl.float32)
    r227_value = tl.broadcast_to(r227_value, [$(2304)])
    tl.store($(candidate_vto_copy_76_ptr) + r227_z, r227_value)
    r228_row = tl.arange($(12))
    r228_col = tl.arange($(36))
    r228_lane = tl.arange($(64))
    for r228_group in range($(8)) {
      r228_left = tl.load($(candidate_vto_copy_75_ptr) + (((((((r228_group % $(8)) * $(768)) + (r228_row[:, None] * $(64)) + r228_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r228_group % $(8)) * $(768)) + (r228_row[:, None] * $(64)) + r228_lane[None, :]) / $(64)) % $(12)) * $(64)) + ((((r228_group % $(8)) * $(768)) + (r228_row[:, None] * $(64)) + r228_lane[None, :]) % $(64))), dtype=tl.float32)
      r228_right = tl.load($(candidate_vto_copy_76_ptr) + ((((((r228_group % $(8)) * $(2304)) + (r228_lane[:, None] * $(36)) + r228_col[None, :]) % $(36)) * $(64)) + (((((r228_group % $(8)) * $(2304)) + (r228_lane[:, None] * $(36)) + r228_col[None, :]) / $(36)) % $(64))), dtype=tl.float32)
      r228_value = tl.dot(r228_left, r228_right)
      r228_value = r228_value.to(tl.float32)
      tl.store($(candidate_vbmm_18_ptr) + ((r228_group * $(432)) + (r228_row[:, None] * $(36)) + r228_col[None, :]), r228_value)
    }
    r229_z = tl.arange($(49152))
    r229_value = tl.load($(candidate_vbmm_17_ptr) + ((((r229_z / $(6144)) % $(8)) * $(16384)) + (((r229_z / $(512)) % $(12)) * $(512)) + (r229_z % $(512))), dtype=tl.bfloat16)
    r229_value = r229_value.to(tl.float32)
    r229_value = tl.broadcast_to(r229_value, [$(49152)])
    tl.store($(candidate_vto_copy_77_ptr) + r229_z, r229_value)
    r230_z = tl.arange($(18432))
    r230_value = tl.load($(candidate_vindex_2_ptr) + ((((r230_z / $(512)) % $(36)) * $(576)) + (r230_z % $(512))), dtype=tl.bfloat16)
    r230_value = r230_value.to(tl.float32)
    r230_value = tl.broadcast_to(r230_value, [$(18432)])
    tl.store($(candidate_vto_copy_78_ptr) + r230_z, r230_value)
    r231_row = tl.arange($(96))
    r231_col = tl.arange($(36))
    r231_lane = tl.arange($(512))
    r231_left = tl.load($(candidate_vto_copy_77_ptr) + ((((((r231_row[:, None] * $(512)) + r231_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r231_row[:, None] * $(512)) + r231_lane[None, :]) / $(512)) % $(12)) * $(512)) + (((r231_row[:, None] * $(512)) + r231_lane[None, :]) % $(512))), dtype=tl.float32)
    r231_right = tl.load($(candidate_vto_copy_78_ptr) + (((((r231_lane[:, None] * $(36)) + r231_col[None, :]) % $(36)) * $(512)) + ((((r231_lane[:, None] * $(36)) + r231_col[None, :]) / $(36)) % $(512))), dtype=tl.float32)
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
    tl.store($(candidate_vto_copy_79_ptr) + r239_z, r239_value)
    r240_row = tl.arange($(96))
    r240_col = tl.arange($(512))
    r240_lane = tl.arange($(36))
    r240_left = tl.load($(candidate_vto_copy_79_ptr) + ((((((r240_row[:, None] * $(36)) + r240_lane[None, :]) / $(432)) % $(8)) * $(432)) + (((((r240_row[:, None] * $(36)) + r240_lane[None, :]) / $(36)) % $(12)) * $(36)) + (((r240_row[:, None] * $(36)) + r240_lane[None, :]) % $(36))), dtype=tl.bfloat16)
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
    tl.store($(candidate_vto_copy_80_ptr) + r242_z, r242_value)
    r243_z = tl.arange($(768))
    r243_value = tl.load($(candidate_vindex_2_ptr) + ((((r243_z / $(64)) % $(12)) * $(576)) + (r243_z % $(64)) + $(28160)), dtype=tl.bfloat16)
    r243_value = r243_value.to(tl.float32)
    r243_value = tl.broadcast_to(r243_value, [$(768)])
    tl.store($(candidate_vto_copy_81_ptr) + r243_z, r243_value)
    r244_row = tl.arange($(4))
    r244_col = tl.arange($(12))
    r244_lane = tl.arange($(64))
    for r244_group in range($(8)) {
      r244_left = tl.load($(candidate_vto_copy_80_ptr) + (((((((r244_group % $(8)) * $(256)) + (r244_row[:, None] * $(64)) + r244_lane[None, :]) / $(256)) % $(8)) * $(256)) + ((((((r244_group % $(8)) * $(256)) + (r244_row[:, None] * $(64)) + r244_lane[None, :]) / $(64)) % $(4)) * $(64)) + ((((r244_group % $(8)) * $(256)) + (r244_row[:, None] * $(64)) + r244_lane[None, :]) % $(64))), dtype=tl.float32)
      r244_right = tl.load($(candidate_vto_copy_81_ptr) + ((((((r244_group % $(8)) * $(768)) + (r244_lane[:, None] * $(12)) + r244_col[None, :]) % $(12)) * $(64)) + (((((r244_group % $(8)) * $(768)) + (r244_lane[:, None] * $(12)) + r244_col[None, :]) / $(12)) % $(64))), dtype=tl.float32)
      r244_value = tl.dot(r244_left, r244_right)
      r244_value = r244_value.to(tl.float32)
      tl.store($(candidate_vbmm_19_ptr) + ((r244_group * $(48)) + (r244_row[:, None] * $(12)) + r244_col[None, :]), r244_value)
    }
    r245_z = tl.arange($(16384))
    r245_value = tl.load($(candidate_vbmm_17_ptr) + ((((r245_z / $(2048)) % $(8)) * $(16384)) + (((r245_z / $(512)) % $(4)) * $(512)) + (r245_z % $(512)) + $(6144)), dtype=tl.bfloat16)
    r245_value = r245_value.to(tl.float32)
    r245_value = tl.broadcast_to(r245_value, [$(16384)])
    tl.store($(candidate_vto_copy_82_ptr) + r245_z, r245_value)
    r246_z = tl.arange($(6144))
    r246_value = tl.load($(candidate_vindex_2_ptr) + ((((r246_z / $(512)) % $(12)) * $(576)) + (r246_z % $(512)) + $(27648)), dtype=tl.bfloat16)
    r246_value = r246_value.to(tl.float32)
    r246_value = tl.broadcast_to(r246_value, [$(6144)])
    tl.store($(candidate_vto_copy_83_ptr) + r246_z, r246_value)
    r247_row = tl.arange($(32))
    r247_col = tl.arange($(12))
    r247_lane = tl.arange($(512))
    r247_left = tl.load($(candidate_vto_copy_82_ptr) + ((((((r247_row[:, None] * $(512)) + r247_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + (((((r247_row[:, None] * $(512)) + r247_lane[None, :]) / $(512)) % $(4)) * $(512)) + (((r247_row[:, None] * $(512)) + r247_lane[None, :]) % $(512))), dtype=tl.float32)
    r247_right = tl.load($(candidate_vto_copy_83_ptr) + (((((r247_lane[:, None] * $(12)) + r247_col[None, :]) % $(12)) * $(512)) + ((((r247_lane[:, None] * $(12)) + r247_col[None, :]) / $(12)) % $(512))), dtype=tl.float32)
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
    tl.store($(candidate_vto_copy_84_ptr) + r255_z, r255_value)
    r256_row = tl.arange($(32))
    r256_col = tl.arange($(512))
    r256_lane = tl.arange($(12))
    r256_left = tl.load($(candidate_vto_copy_84_ptr) + ((((((r256_row[:, None] * $(12)) + r256_lane[None, :]) / $(48)) % $(8)) * $(48)) + (((((r256_row[:, None] * $(12)) + r256_lane[None, :]) / $(12)) % $(4)) * $(12)) + (((r256_row[:, None] * $(12)) + r256_lane[None, :]) % $(12))), dtype=tl.bfloat16)
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
    tl.store($(candidate_vto_copy_85_ptr) + r258_z, r258_value)
    r259_z = tl.arange($(3072))
    r259_value = tl.load($(candidate_vindex_2_ptr) + ((((r259_z / $(64)) % $(48)) * $(576)) + (r259_z % $(64)) + $(37376)), dtype=tl.bfloat16)
    r259_value = r259_value.to(tl.float32)
    r259_value = tl.broadcast_to(r259_value, [$(3072)])
    tl.store($(candidate_vto_copy_86_ptr) + r259_z, r259_value)
    r260_row = tl.arange($(16))
    r260_col = tl.arange($(48))
    r260_lane = tl.arange($(64))
    for r260_group in range($(8)) {
      r260_left = tl.load($(candidate_vto_copy_85_ptr) + (((((((r260_group % $(8)) * $(1024)) + (r260_row[:, None] * $(64)) + r260_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + ((((((r260_group % $(8)) * $(1024)) + (r260_row[:, None] * $(64)) + r260_lane[None, :]) / $(64)) % $(16)) * $(64)) + ((((r260_group % $(8)) * $(1024)) + (r260_row[:, None] * $(64)) + r260_lane[None, :]) % $(64))), dtype=tl.float32)
      r260_right = tl.load($(candidate_vto_copy_86_ptr) + ((((((r260_group % $(8)) * $(3072)) + (r260_lane[:, None] * $(48)) + r260_col[None, :]) % $(48)) * $(64)) + (((((r260_group % $(8)) * $(3072)) + (r260_lane[:, None] * $(48)) + r260_col[None, :]) / $(48)) % $(64))), dtype=tl.float32)
      r260_value = tl.dot(r260_left, r260_right)
      r260_value = r260_value.to(tl.float32)
      tl.store($(candidate_vbmm_20_ptr) + ((r260_group * $(768)) + (r260_row[:, None] * $(48)) + r260_col[None, :]), r260_value)
    }
    r261_z = tl.arange($(65536))
    r261_value = tl.load($(candidate_vbmm_17_ptr) + ((((r261_z / $(8192)) % $(8)) * $(16384)) + (((r261_z / $(512)) % $(16)) * $(512)) + (r261_z % $(512)) + $(8192)), dtype=tl.bfloat16)
    r261_value = r261_value.to(tl.float32)
    r261_value = tl.broadcast_to(r261_value, [$(65536)])
    tl.store($(candidate_vto_copy_87_ptr) + r261_z, r261_value)
    r262_z = tl.arange($(24576))
    r262_value = tl.load($(candidate_vindex_2_ptr) + ((((r262_z / $(512)) % $(48)) * $(576)) + (r262_z % $(512)) + $(36864)), dtype=tl.bfloat16)
    r262_value = r262_value.to(tl.float32)
    r262_value = tl.broadcast_to(r262_value, [$(24576)])
    tl.store($(candidate_vto_copy_88_ptr) + r262_z, r262_value)
    r263_row = tl.arange($(128))
    r263_col = tl.arange($(48))
    r263_lane = tl.arange($(512))
    r263_left = tl.load($(candidate_vto_copy_87_ptr) + ((((((r263_row[:, None] * $(512)) + r263_lane[None, :]) / $(8192)) % $(8)) * $(8192)) + (((((r263_row[:, None] * $(512)) + r263_lane[None, :]) / $(512)) % $(16)) * $(512)) + (((r263_row[:, None] * $(512)) + r263_lane[None, :]) % $(512))), dtype=tl.float32)
    r263_right = tl.load($(candidate_vto_copy_88_ptr) + (((((r263_lane[:, None] * $(48)) + r263_col[None, :]) % $(48)) * $(512)) + ((((r263_lane[:, None] * $(48)) + r263_col[None, :]) / $(48)) % $(512))), dtype=tl.float32)
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
    tl.store($(candidate_vto_copy_89_ptr) + r271_z, r271_value)
    r272_row = tl.arange($(128))
    r272_col = tl.arange($(512))
    r272_lane = tl.arange($(48))
    r272_left = tl.load($(candidate_vto_copy_89_ptr) + ((((((r272_row[:, None] * $(48)) + r272_lane[None, :]) / $(768)) % $(8)) * $(768)) + (((((r272_row[:, None] * $(48)) + r272_lane[None, :]) / $(48)) % $(16)) * $(48)) + (((r272_row[:, None] * $(48)) + r272_lane[None, :]) % $(48))), dtype=tl.bfloat16)
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
      r276_right = tl.load($(candidate_vto_copy_20_ptr) + (((((((r276_group % $(8)) * $(65536)) + (r276_lane[:, None] * $(128)) + r276_col[None, :]) / $(65536)) % $(8)) * $(131072)) + (((((r276_group % $(8)) * $(65536)) + (r276_lane[:, None] * $(128)) + r276_col[None, :]) % $(128)) * $(512)) + (((((r276_group % $(8)) * $(65536)) + (r276_lane[:, None] * $(128)) + r276_col[None, :]) / $(128)) % $(512)) + $(65536)), dtype=tl.bfloat16)
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
      r279_right = tl.load($(candidate_vto_copy_20_ptr) + ((((((r279_lane[:, None] * $(1024)) + r279_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r279_lane[:, None] * $(1024)) + r279_col[None, :]) % $(128)) * $(512)) + ((((r279_lane[:, None] * $(1024)) + r279_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r279_value = tl.dot(r279_left, r279_right)
      r279_value = r279_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_22_ptr) + ((r279_group * $(131072)) + (r279_row[:, None] * $(1024)) + r279_col[None, :]), r279_value)
    }
    r280_row = tl.arange($(128))
    r280_col = tl.arange($(1024))
    r280_lane = tl.arange($(512))
    for r280_group in range($(1)) {
      r280_left = tl.load($(candidate_vindex_2_ptr) + ((((((r280_row[:, None] * $(512)) + r280_lane[None, :]) / $(512)) % $(128)) * $(576)) + (((r280_row[:, None] * $(512)) + r280_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r280_right = tl.load($(candidate_vto_copy_20_ptr) + ((((((r280_lane[:, None] * $(1024)) + r280_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r280_lane[:, None] * $(1024)) + r280_col[None, :]) % $(128)) * $(512)) + ((((r280_lane[:, None] * $(1024)) + r280_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
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
    tl.store($(candidate_vto_copy_90_ptr) + r283_z, r283_value)
    r284_z = tl.arange($(55296))
    r284_value = tl.load($(candidate_vcat_16_ptr) + ((((r284_z / $(192)) % $(36)) * $(1536)) + (((r284_z / $(6912)) % $(8)) * $(192)) + (r284_z % $(192))), dtype=tl.bfloat16)
    r284_value = r284_value.to(tl.float32)
    r284_value = tl.broadcast_to(r284_value, [$(55296)])
    tl.store($(candidate_vto_copy_91_ptr) + r284_z, r284_value)
    r285_z = tl.arange($(36864))
    r285_value = tl.load($(candidate_vbmm_23_ptr) + ((((r285_z / $(128)) % $(36)) * $(1024)) + (((r285_z / $(4608)) % $(8)) * $(128)) + (r285_z % $(128))), dtype=tl.bfloat16)
    r285_value = r285_value.to(tl.float32)
    r285_value = tl.broadcast_to(r285_value, [$(36864)])
    tl.store($(candidate_vto_copy_92_ptr) + r285_z, r285_value)
    r286_z = tl.arange($(12288))
    r286_value = tl.load($(candidate_vmm_28_ptr) + ((((r286_z / $(128)) % $(12)) * $(1024)) + (((r286_z / $(1536)) % $(8)) * $(128)) + (r286_z % $(128)) + $(98304)), dtype=tl.bfloat16)
    r286_value = r286_value.to(tl.float32)
    r286_value = tl.broadcast_to(r286_value, [$(12288)])
    tl.store($(candidate_vto_copy_93_ptr) + r286_z, r286_value)
    r287_z = tl.arange($(12288))
    r287_value = tl.load($(candidate_vbmm_21_ptr) + ((((r287_z / $(1536)) % $(8)) * $(4096)) + (((r287_z / $(128)) % $(12)) * $(128)) + (r287_z % $(128))), dtype=tl.bfloat16)
    r287_value = r287_value.to(tl.float32)
    r287_value = tl.broadcast_to(r287_value, [$(12288)])
    tl.store($(candidate_vto_copy_94_ptr) + r287_z, r287_value)
    r288_row = tl.arange($(12))
    r288_col = tl.arange($(36))
    r288_lane = tl.arange($(192))
    for r288_group in range($(8)) {
      r288_left = tl.load($(candidate_vto_copy_90_ptr) + (((((((r288_group % $(8)) * $(2304)) + (r288_row[:, None] * $(192)) + r288_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r288_group % $(8)) * $(2304)) + (r288_row[:, None] * $(192)) + r288_lane[None, :]) / $(192)) % $(12)) * $(192)) + ((((r288_group % $(8)) * $(2304)) + (r288_row[:, None] * $(192)) + r288_lane[None, :]) % $(192))), dtype=tl.float32)
      r288_right = tl.load($(candidate_vto_copy_91_ptr) + (((((((r288_group % $(8)) * $(6912)) + (r288_lane[:, None] * $(36)) + r288_col[None, :]) / $(6912)) % $(8)) * $(6912)) + (((((r288_group % $(8)) * $(6912)) + (r288_lane[:, None] * $(36)) + r288_col[None, :]) % $(36)) * $(192)) + (((((r288_group % $(8)) * $(6912)) + (r288_lane[:, None] * $(36)) + r288_col[None, :]) / $(36)) % $(192))), dtype=tl.float32)
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
    r296_value = (tl.load($(candidate_vto_copy_93_ptr) + ((((r296_z / $(1536)) % $(8)) * $(1536)) + (((r296_z / $(128)) % $(12)) * $(128)) + (r296_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_94_ptr) + ((((r296_z / $(1536)) % $(8)) * $(1536)) + (((r296_z / $(128)) % $(12)) * $(128)) + (r296_z % $(128))), dtype=tl.float32))
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
      r298_left = tl.load($(candidate_vto_copy_93_ptr) + (((((((r298_group % $(8)) * $(1536)) + (r298_row[:, None] * $(128)) + r298_lane[None, :]) / $(1536)) % $(8)) * $(1536)) + ((((((r298_group % $(8)) * $(1536)) + (r298_row[:, None] * $(128)) + r298_lane[None, :]) / $(128)) % $(12)) * $(128)) + ((((r298_group % $(8)) * $(1536)) + (r298_row[:, None] * $(128)) + r298_lane[None, :]) % $(128))), dtype=tl.float32)
      r298_right = tl.load($(candidate_vto_copy_92_ptr) + (((((((r298_group % $(8)) * $(4608)) + (r298_lane[:, None] * $(36)) + r298_col[None, :]) / $(4608)) % $(8)) * $(4608)) + (((((r298_group % $(8)) * $(4608)) + (r298_lane[:, None] * $(36)) + r298_col[None, :]) % $(36)) * $(128)) + (((((r298_group % $(8)) * $(4608)) + (r298_lane[:, None] * $(36)) + r298_col[None, :]) / $(36)) % $(128))), dtype=tl.float32)
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
    r301_row = tl.arange($(36))
    r301_col = tl.arange($(192))
    r301_lane = tl.arange($(12))
    for r301_group in range($(8)) {
      r301_left = tl.load($(candidate_vmul_37_ptr) + (((((((r301_group % $(8)) * $(432)) + (r301_row[:, None] * $(12)) + r301_lane[None, :]) / $(432)) % $(8)) * $(432)) + (((((r301_group % $(8)) * $(432)) + (r301_row[:, None] * $(12)) + r301_lane[None, :]) % $(12)) * $(36)) + (((((r301_group % $(8)) * $(432)) + (r301_row[:, None] * $(12)) + r301_lane[None, :]) / $(12)) % $(36))), dtype=tl.float32)
      r301_right = tl.load($(candidate_vto_copy_90_ptr) + (((((((r301_group % $(8)) * $(2304)) + (r301_lane[:, None] * $(192)) + r301_col[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r301_group % $(8)) * $(2304)) + (r301_lane[:, None] * $(192)) + r301_col[None, :]) / $(192)) % $(12)) * $(192)) + ((((r301_group % $(8)) * $(2304)) + (r301_lane[:, None] * $(192)) + r301_col[None, :]) % $(192))), dtype=tl.float32)
      r301_value = tl.dot(r301_left, r301_right)
      r301_value = r301_value.to(tl.float32)
      tl.store($(candidate_vbmm_27_ptr) + ((r301_group * $(6912)) + (r301_row[:, None] * $(192)) + r301_col[None, :]), r301_value)
    }
    r302_z = tl.arange($(55296))
    r302_value = (tl.load($(candidate_vbmm_27_ptr) + ((((r302_z / $(6912)) % $(8)) * $(6912)) + (((r302_z / $(192)) % $(36)) * $(192)) + (r302_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r302_value = r302_value.to(tl.float32)
    r302_value = tl.broadcast_to(r302_value, [$(55296)])
    tl.store($(candidate_vmul_39_ptr) + r302_z, r302_value)
    r303_z = tl.arange($(55296))
    r303_value = tl.load($(candidate_vmul_39_ptr) + ((((r303_z / $(192)) % $(8)) * $(6912)) + (((r303_z / $(1536)) % $(36)) * $(192)) + (r303_z % $(192))), dtype=tl.float32)
    r303_value = r303_value.to(tl.bfloat16)
    r303_value = tl.broadcast_to(r303_value, [$(55296)])
    tl.store($(candidate_vto_copy_96_ptr) + r303_z, r303_value)
    r304_row = tl.arange($(36))
    r304_col = tl.arange($(128))
    r304_lane = tl.arange($(12))
    for r304_group in range($(8)) {
      r304_left = tl.load($(candidate_vexp_ptr) + (((((((r304_group % $(8)) * $(432)) + (r304_row[:, None] * $(12)) + r304_lane[None, :]) / $(432)) % $(8)) * $(432)) + (((((r304_group % $(8)) * $(432)) + (r304_row[:, None] * $(12)) + r304_lane[None, :]) % $(12)) * $(36)) + (((((r304_group % $(8)) * $(432)) + (r304_row[:, None] * $(12)) + r304_lane[None, :]) / $(12)) % $(36))), dtype=tl.float32)
      r304_right = tl.load($(candidate_vto_copy_93_ptr) + (((((((r304_group % $(8)) * $(1536)) + (r304_lane[:, None] * $(128)) + r304_col[None, :]) / $(1536)) % $(8)) * $(1536)) + ((((((r304_group % $(8)) * $(1536)) + (r304_lane[:, None] * $(128)) + r304_col[None, :]) / $(128)) % $(12)) * $(128)) + ((((r304_group % $(8)) * $(1536)) + (r304_lane[:, None] * $(128)) + r304_col[None, :]) % $(128))), dtype=tl.float32)
      r304_value = tl.dot(r304_left, r304_right)
      r304_value = r304_value.to(tl.float32)
      tl.store($(candidate_vbmm_28_ptr) + ((r304_group * $(4608)) + (r304_row[:, None] * $(128)) + r304_col[None, :]), r304_value)
    }
    r305_z = tl.arange($(36864))
    r305_value = tl.load($(candidate_vbmm_28_ptr) + ((((r305_z / $(128)) % $(8)) * $(4608)) + (((r305_z / $(1024)) % $(36)) * $(128)) + (r305_z % $(128))), dtype=tl.float32)
    r305_value = r305_value.to(tl.bfloat16)
    r305_value = tl.broadcast_to(r305_value, [$(36864)])
    tl.store($(candidate_vto_copy_97_ptr) + r305_z, r305_value)
    r306_z = tl.arange($(73728))
    r306_value = tl.where(((r306_z / $(1536)) % $(48)) < $(36), tl.load($(candidate_vto_copy_96_ptr) + ((((((r306_z / $(1536)) % $(48)) - $(0)) % $(36)) * $(1536)) + (((r306_z / $(192)) % $(8)) * $(192)) + (r306_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r306_value = r306_value.to(tl.bfloat16)
    r306_value = tl.broadcast_to(r306_value, [$(73728)])
    tl.store($(candidate_vcat_17_ptr) + r306_z, r306_value)
    r307_z = tl.arange($(49152))
    r307_value = tl.where(((r307_z / $(1024)) % $(48)) < $(36), tl.load($(candidate_vto_copy_97_ptr) + ((((((r307_z / $(1024)) % $(48)) - $(0)) % $(36)) * $(1024)) + (((r307_z / $(128)) % $(8)) * $(128)) + (r307_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r307_value = r307_value.to(tl.bfloat16)
    r307_value = tl.broadcast_to(r307_value, [$(49152)])
    tl.store($(candidate_vcat_18_ptr) + r307_z, r307_value)
    r308_z = tl.arange($(6144))
    r308_value = tl.load($(candidate_vcat_15_ptr) + ((((r308_z / $(192)) % $(4)) * $(1536)) + (((r308_z / $(768)) % $(8)) * $(192)) + (r308_z % $(192)) + $(18432)), dtype=tl.bfloat16)
    r308_value = r308_value.to(tl.float32)
    r308_value = tl.broadcast_to(r308_value, [$(6144)])
    tl.store($(candidate_vto_copy_98_ptr) + r308_z, r308_value)
    r309_z = tl.arange($(18432))
    r309_value = tl.load($(candidate_vcat_16_ptr) + ((((r309_z / $(192)) % $(12)) * $(1536)) + (((r309_z / $(2304)) % $(8)) * $(192)) + (r309_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r309_value = r309_value.to(tl.float32)
    r309_value = tl.broadcast_to(r309_value, [$(18432)])
    tl.store($(candidate_vto_copy_99_ptr) + r309_z, r309_value)
    r310_z = tl.arange($(12288))
    r310_value = tl.load($(candidate_vbmm_23_ptr) + ((((r310_z / $(128)) % $(12)) * $(1024)) + (((r310_z / $(1536)) % $(8)) * $(128)) + (r310_z % $(128)) + $(49152)), dtype=tl.bfloat16)
    r310_value = r310_value.to(tl.float32)
    r310_value = tl.broadcast_to(r310_value, [$(12288)])
    tl.store($(candidate_vto_copy_100_ptr) + r310_z, r310_value)
    r311_z = tl.arange($(4096))
    r311_value = tl.load($(candidate_vmm_28_ptr) + ((((r311_z / $(128)) % $(4)) * $(1024)) + (((r311_z / $(512)) % $(8)) * $(128)) + (r311_z % $(128)) + $(110592)), dtype=tl.bfloat16)
    r311_value = r311_value.to(tl.float32)
    r311_value = tl.broadcast_to(r311_value, [$(4096)])
    tl.store($(candidate_vto_copy_101_ptr) + r311_z, r311_value)
    r312_z = tl.arange($(4096))
    r312_value = tl.load($(candidate_vbmm_21_ptr) + ((((r312_z / $(512)) % $(8)) * $(4096)) + (((r312_z / $(128)) % $(4)) * $(128)) + (r312_z % $(128)) + $(1536)), dtype=tl.bfloat16)
    r312_value = r312_value.to(tl.float32)
    r312_value = tl.broadcast_to(r312_value, [$(4096)])
    tl.store($(candidate_vto_copy_102_ptr) + r312_z, r312_value)
    r313_row = tl.arange($(4))
    r313_col = tl.arange($(12))
    r313_lane = tl.arange($(192))
    for r313_group in range($(8)) {
      r313_left = tl.load($(candidate_vto_copy_98_ptr) + (((((((r313_group % $(8)) * $(768)) + (r313_row[:, None] * $(192)) + r313_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r313_group % $(8)) * $(768)) + (r313_row[:, None] * $(192)) + r313_lane[None, :]) / $(192)) % $(4)) * $(192)) + ((((r313_group % $(8)) * $(768)) + (r313_row[:, None] * $(192)) + r313_lane[None, :]) % $(192))), dtype=tl.float32)
      r313_right = tl.load($(candidate_vto_copy_99_ptr) + (((((((r313_group % $(8)) * $(2304)) + (r313_lane[:, None] * $(12)) + r313_col[None, :]) / $(2304)) % $(8)) * $(2304)) + (((((r313_group % $(8)) * $(2304)) + (r313_lane[:, None] * $(12)) + r313_col[None, :]) % $(12)) * $(192)) + (((((r313_group % $(8)) * $(2304)) + (r313_lane[:, None] * $(12)) + r313_col[None, :]) / $(12)) % $(192))), dtype=tl.float32)
      r313_value = tl.dot(r313_left, r313_right)
      r313_value = r313_value.to(tl.float32)
      tl.store($(candidate_vbmm_29_ptr) + ((r313_group * $(48)) + (r313_row[:, None] * $(12)) + r313_col[None, :]), r313_value)
    }
    r314_z = tl.arange($(384))
    r314_value = (tl.load($(candidate_vbmm_29_ptr) + ((((r314_z / $(48)) % $(8)) * $(48)) + (((r314_z / $(12)) % $(4)) * $(12)) + (r314_z % $(12))), dtype=tl.float32) * 0.07216878364870322)
    r314_value = r314_value.to(tl.float32)
    r314_value = tl.broadcast_to(r314_value, [$(384)])
    tl.store($(candidate_vmul_40_ptr) + r314_z, r314_value)
    r315_z = tl.arange($(4))
    r315_value = (tl.cast(r315_z, tl.int64) + tl.full([], $(12), dtype=tl.int64))
    r315_value = tl.broadcast_to(r315_value, [$(4)])
    tl.store($(candidate_vadd_33_ptr) + r315_z, r315_value)
    r316_z = tl.arange($(4))
    r316_value = (tl.load($(candidate_vadd_33_ptr) + r316_z, dtype=tl.int64) - tl.full([], $(4), dtype=tl.int64))
    r316_value = tl.broadcast_to(r316_value, [$(4)])
    tl.store($(candidate_vsub_24_ptr) + r316_z, r316_value)
    r317_z = tl.arange($(48))
    r317_value = (tl.cast((r317_z % $(12)), tl.int64) > tl.load($(candidate_vsub_24_ptr) + ((r317_z / $(12)) % $(4)), dtype=tl.int64))
    r317_value = tl.broadcast_to(r317_value, [$(48)])
    tl.store($(candidate_vgt_13_ptr) + r317_z, r317_value)
    r318_z = tl.arange($(384))
    r318_value = tl.where(tl.load($(candidate_vgt_13_ptr) + ((((r318_z / $(12)) % $(4)) * $(12)) + (r318_z % $(12))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_40_ptr) + ((((r318_z / $(48)) % $(8)) * $(48)) + (((r318_z / $(12)) % $(4)) * $(12)) + (r318_z % $(12))), dtype=tl.float32))
    r318_value = r318_value.to(tl.float32)
    r318_value = tl.broadcast_to(r318_value, [$(384)])
    tl.store($(candidate_vmasked_fill_13_ptr) + r318_z, r318_value)
    r319_z = tl.arange($(384))
    r319_value = (tl.load($(candidate_vmasked_fill_13_ptr) + ((((r319_z / $(48)) % $(8)) * $(48)) + (((r319_z / $(12)) % $(4)) * $(12)) + (r319_z % $(12))), dtype=tl.float32) - tl.load($(candidate_vcat_13_ptr) + ((((r319_z / $(12)) % $(4)) * $(8)) + ((r319_z / $(48)) % $(8)) + $(96)), dtype=tl.float32))
    r319_value = r319_value.to(tl.float32)
    r319_value = tl.broadcast_to(r319_value, [$(384)])
    tl.store($(candidate_vsub_25_ptr) + r319_z, r319_value)
    r320_z = tl.arange($(384))
    r320_value = tl.exp(tl.load($(candidate_vsub_25_ptr) + ((((r320_z / $(48)) % $(8)) * $(48)) + (((r320_z / $(12)) % $(4)) * $(12)) + (r320_z % $(12))), dtype=tl.float32))
    r320_value = r320_value.to(tl.float32)
    r320_value = tl.broadcast_to(r320_value, [$(384)])
    tl.store($(candidate_vexp_1_ptr) + r320_z, r320_value)
    r321_z = tl.arange($(4096))
    r321_value = (tl.load($(candidate_vto_copy_101_ptr) + ((((r321_z / $(512)) % $(8)) * $(512)) + (((r321_z / $(128)) % $(4)) * $(128)) + (r321_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_102_ptr) + ((((r321_z / $(512)) % $(8)) * $(512)) + (((r321_z / $(128)) % $(4)) * $(128)) + (r321_z % $(128))), dtype=tl.float32))
    r321_value = r321_value.to(tl.float32)
    r321_value = tl.broadcast_to(r321_value, [$(4096)])
    tl.store($(candidate_vmul_41_ptr) + r321_z, r321_value)
    r322_z = tl.arange($(32))
    r322_lane = tl.arange($(128))
    r322_value = tl.sum(tl.load($(candidate_vmul_41_ptr) + ((((r322_z[:, None] / $(4)) % $(8)) * $(512)) + ((r322_z[:, None] % $(4)) * $(128)) + r322_lane[None, :]), dtype=tl.float32), axis=1)
    r322_value = r322_value.to(tl.float32)
    r322_value = tl.broadcast_to(r322_value, [$(32)])
    tl.store($(candidate_vsum_2_ptr) + r322_z, r322_value)
    r323_row = tl.arange($(4))
    r323_col = tl.arange($(12))
    r323_lane = tl.arange($(128))
    for r323_group in range($(8)) {
      r323_left = tl.load($(candidate_vto_copy_101_ptr) + (((((((r323_group % $(8)) * $(512)) + (r323_row[:, None] * $(128)) + r323_lane[None, :]) / $(512)) % $(8)) * $(512)) + ((((((r323_group % $(8)) * $(512)) + (r323_row[:, None] * $(128)) + r323_lane[None, :]) / $(128)) % $(4)) * $(128)) + ((((r323_group % $(8)) * $(512)) + (r323_row[:, None] * $(128)) + r323_lane[None, :]) % $(128))), dtype=tl.float32)
      r323_right = tl.load($(candidate_vto_copy_100_ptr) + (((((((r323_group % $(8)) * $(1536)) + (r323_lane[:, None] * $(12)) + r323_col[None, :]) / $(1536)) % $(8)) * $(1536)) + (((((r323_group % $(8)) * $(1536)) + (r323_lane[:, None] * $(12)) + r323_col[None, :]) % $(12)) * $(128)) + (((((r323_group % $(8)) * $(1536)) + (r323_lane[:, None] * $(12)) + r323_col[None, :]) / $(12)) % $(128))), dtype=tl.float32)
      r323_value = tl.dot(r323_left, r323_right)
      r323_value = r323_value.to(tl.float32)
      tl.store($(candidate_vbmm_30_ptr) + ((r323_group * $(48)) + (r323_row[:, None] * $(12)) + r323_col[None, :]), r323_value)
    }
    r324_z = tl.arange($(384))
    r324_value = (tl.load($(candidate_vbmm_30_ptr) + ((((r324_z / $(48)) % $(8)) * $(48)) + (((r324_z / $(12)) % $(4)) * $(12)) + (r324_z % $(12))), dtype=tl.float32) - tl.load($(candidate_vsum_2_ptr) + ((((r324_z / $(48)) % $(8)) * $(4)) + ((r324_z / $(12)) % $(4))), dtype=tl.float32))
    r324_value = r324_value.to(tl.float32)
    r324_value = tl.broadcast_to(r324_value, [$(384)])
    tl.store($(candidate_vsub_26_ptr) + r324_z, r324_value)
    r325_z = tl.arange($(384))
    r325_value = (tl.load($(candidate_vexp_1_ptr) + ((((r325_z / $(48)) % $(8)) * $(48)) + (((r325_z / $(12)) % $(4)) * $(12)) + (r325_z % $(12))), dtype=tl.float32) * tl.load($(candidate_vsub_26_ptr) + ((((r325_z / $(48)) % $(8)) * $(48)) + (((r325_z / $(12)) % $(4)) * $(12)) + (r325_z % $(12))), dtype=tl.float32))
    r325_value = r325_value.to(tl.float32)
    r325_value = tl.broadcast_to(r325_value, [$(384)])
    tl.store($(candidate_vmul_42_ptr) + r325_z, r325_value)
    r326_row = tl.arange($(12))
    r326_col = tl.arange($(192))
    r326_lane = tl.arange($(4))
    for r326_group in range($(8)) {
      r326_left = tl.load($(candidate_vmul_42_ptr) + (((((((r326_group % $(8)) * $(48)) + (r326_row[:, None] * $(4)) + r326_lane[None, :]) / $(48)) % $(8)) * $(48)) + (((((r326_group % $(8)) * $(48)) + (r326_row[:, None] * $(4)) + r326_lane[None, :]) % $(4)) * $(12)) + (((((r326_group % $(8)) * $(48)) + (r326_row[:, None] * $(4)) + r326_lane[None, :]) / $(4)) % $(12))), dtype=tl.float32)
      r326_right = tl.load($(candidate_vto_copy_98_ptr) + (((((((r326_group % $(8)) * $(768)) + (r326_lane[:, None] * $(192)) + r326_col[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r326_group % $(8)) * $(768)) + (r326_lane[:, None] * $(192)) + r326_col[None, :]) / $(192)) % $(4)) * $(192)) + ((((r326_group % $(8)) * $(768)) + (r326_lane[:, None] * $(192)) + r326_col[None, :]) % $(192))), dtype=tl.float32)
      r326_value = tl.dot(r326_left, r326_right)
      r326_value = r326_value.to(tl.float32)
      tl.store($(candidate_vbmm_32_ptr) + ((r326_group * $(2304)) + (r326_row[:, None] * $(192)) + r326_col[None, :]), r326_value)
    }
    r327_z = tl.arange($(18432))
    r327_value = (tl.load($(candidate_vbmm_32_ptr) + ((((r327_z / $(2304)) % $(8)) * $(2304)) + (((r327_z / $(192)) % $(12)) * $(192)) + (r327_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r327_value = r327_value.to(tl.float32)
    r327_value = tl.broadcast_to(r327_value, [$(18432)])
    tl.store($(candidate_vmul_44_ptr) + r327_z, r327_value)
    r328_z = tl.arange($(18432))
    r328_value = tl.load($(candidate_vmul_44_ptr) + ((((r328_z / $(192)) % $(8)) * $(2304)) + (((r328_z / $(1536)) % $(12)) * $(192)) + (r328_z % $(192))), dtype=tl.float32)
    r328_value = r328_value.to(tl.bfloat16)
    r328_value = tl.broadcast_to(r328_value, [$(18432)])
    tl.store($(candidate_vto_copy_104_ptr) + r328_z, r328_value)
    r329_row = tl.arange($(12))
    r329_col = tl.arange($(128))
    r329_lane = tl.arange($(4))
    for r329_group in range($(8)) {
      r329_left = tl.load($(candidate_vexp_1_ptr) + (((((((r329_group % $(8)) * $(48)) + (r329_row[:, None] * $(4)) + r329_lane[None, :]) / $(48)) % $(8)) * $(48)) + (((((r329_group % $(8)) * $(48)) + (r329_row[:, None] * $(4)) + r329_lane[None, :]) % $(4)) * $(12)) + (((((r329_group % $(8)) * $(48)) + (r329_row[:, None] * $(4)) + r329_lane[None, :]) / $(4)) % $(12))), dtype=tl.float32)
      r329_right = tl.load($(candidate_vto_copy_101_ptr) + (((((((r329_group % $(8)) * $(512)) + (r329_lane[:, None] * $(128)) + r329_col[None, :]) / $(512)) % $(8)) * $(512)) + ((((((r329_group % $(8)) * $(512)) + (r329_lane[:, None] * $(128)) + r329_col[None, :]) / $(128)) % $(4)) * $(128)) + ((((r329_group % $(8)) * $(512)) + (r329_lane[:, None] * $(128)) + r329_col[None, :]) % $(128))), dtype=tl.float32)
      r329_value = tl.dot(r329_left, r329_right)
      r329_value = r329_value.to(tl.float32)
      tl.store($(candidate_vbmm_33_ptr) + ((r329_group * $(1536)) + (r329_row[:, None] * $(128)) + r329_col[None, :]), r329_value)
    }
    r330_z = tl.arange($(12288))
    r330_value = tl.load($(candidate_vbmm_33_ptr) + ((((r330_z / $(128)) % $(8)) * $(1536)) + (((r330_z / $(1024)) % $(12)) * $(128)) + (r330_z % $(128))), dtype=tl.float32)
    r330_value = r330_value.to(tl.bfloat16)
    r330_value = tl.broadcast_to(r330_value, [$(12288)])
    tl.store($(candidate_vto_copy_105_ptr) + r330_z, r330_value)
    r331_z = tl.arange($(24576))
    r331_value = tl.where(((r331_z / $(1536)) % $(16)) < $(12), tl.load($(candidate_vto_copy_104_ptr) + ((((((r331_z / $(1536)) % $(16)) - $(0)) % $(12)) * $(1536)) + (((r331_z / $(192)) % $(8)) * $(192)) + (r331_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r331_value = r331_value.to(tl.bfloat16)
    r331_value = tl.broadcast_to(r331_value, [$(24576)])
    tl.store($(candidate_vcat_19_ptr) + r331_z, r331_value)
    r332_z = tl.arange($(16384))
    r332_value = tl.where(((r332_z / $(1024)) % $(16)) < $(12), tl.load($(candidate_vto_copy_105_ptr) + ((((((r332_z / $(1024)) % $(16)) - $(0)) % $(12)) * $(1024)) + (((r332_z / $(128)) % $(8)) * $(128)) + (r332_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r332_value = r332_value.to(tl.bfloat16)
    r332_value = tl.broadcast_to(r332_value, [$(16384)])
    tl.store($(candidate_vcat_20_ptr) + r332_z, r332_value)
    r333_z = tl.arange($(24576))
    r333_value = tl.load($(candidate_vcat_15_ptr) + ((((r333_z / $(192)) % $(16)) * $(1536)) + (((r333_z / $(3072)) % $(8)) * $(192)) + (r333_z % $(192)) + $(24576)), dtype=tl.bfloat16)
    r333_value = r333_value.to(tl.float32)
    r333_value = tl.broadcast_to(r333_value, [$(24576)])
    tl.store($(candidate_vto_copy_106_ptr) + r333_z, r333_value)
    r334_z = tl.arange($(73728))
    r334_value = tl.load($(candidate_vcat_16_ptr) + ((((r334_z / $(192)) % $(48)) * $(1536)) + (((r334_z / $(9216)) % $(8)) * $(192)) + (r334_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r334_value = r334_value.to(tl.float32)
    r334_value = tl.broadcast_to(r334_value, [$(73728)])
    tl.store($(candidate_vto_copy_107_ptr) + r334_z, r334_value)
    r335_z = tl.arange($(49152))
    r335_value = tl.load($(candidate_vbmm_23_ptr) + ((((r335_z / $(128)) % $(48)) * $(1024)) + (((r335_z / $(6144)) % $(8)) * $(128)) + (r335_z % $(128)) + $(65536)), dtype=tl.bfloat16)
    r335_value = r335_value.to(tl.float32)
    r335_value = tl.broadcast_to(r335_value, [$(49152)])
    tl.store($(candidate_vto_copy_108_ptr) + r335_z, r335_value)
    r336_z = tl.arange($(16384))
    r336_value = tl.load($(candidate_vmm_28_ptr) + ((((r336_z / $(128)) % $(16)) * $(1024)) + (((r336_z / $(2048)) % $(8)) * $(128)) + (r336_z % $(128)) + $(114688)), dtype=tl.bfloat16)
    r336_value = r336_value.to(tl.float32)
    r336_value = tl.broadcast_to(r336_value, [$(16384)])
    tl.store($(candidate_vto_copy_109_ptr) + r336_z, r336_value)
    r337_z = tl.arange($(16384))
    r337_value = tl.load($(candidate_vbmm_21_ptr) + ((((r337_z / $(2048)) % $(8)) * $(4096)) + (((r337_z / $(128)) % $(16)) * $(128)) + (r337_z % $(128)) + $(2048)), dtype=tl.bfloat16)
    r337_value = r337_value.to(tl.float32)
    r337_value = tl.broadcast_to(r337_value, [$(16384)])
    tl.store($(candidate_vto_copy_110_ptr) + r337_z, r337_value)
    r338_row = tl.arange($(16))
    r338_col = tl.arange($(48))
    r338_lane = tl.arange($(192))
    for r338_group in range($(8)) {
      r338_left = tl.load($(candidate_vto_copy_106_ptr) + (((((((r338_group % $(8)) * $(3072)) + (r338_row[:, None] * $(192)) + r338_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r338_group % $(8)) * $(3072)) + (r338_row[:, None] * $(192)) + r338_lane[None, :]) / $(192)) % $(16)) * $(192)) + ((((r338_group % $(8)) * $(3072)) + (r338_row[:, None] * $(192)) + r338_lane[None, :]) % $(192))), dtype=tl.float32)
      r338_right = tl.load($(candidate_vto_copy_107_ptr) + (((((((r338_group % $(8)) * $(9216)) + (r338_lane[:, None] * $(48)) + r338_col[None, :]) / $(9216)) % $(8)) * $(9216)) + (((((r338_group % $(8)) * $(9216)) + (r338_lane[:, None] * $(48)) + r338_col[None, :]) % $(48)) * $(192)) + (((((r338_group % $(8)) * $(9216)) + (r338_lane[:, None] * $(48)) + r338_col[None, :]) / $(48)) % $(192))), dtype=tl.float32)
      r338_value = tl.dot(r338_left, r338_right)
      r338_value = r338_value.to(tl.float32)
      tl.store($(candidate_vbmm_34_ptr) + ((r338_group * $(768)) + (r338_row[:, None] * $(48)) + r338_col[None, :]), r338_value)
    }
    r339_z = tl.arange($(6144))
    r339_value = (tl.load($(candidate_vbmm_34_ptr) + ((((r339_z / $(768)) % $(8)) * $(768)) + (((r339_z / $(48)) % $(16)) * $(48)) + (r339_z % $(48))), dtype=tl.float32) * 0.07216878364870322)
    r339_value = r339_value.to(tl.float32)
    r339_value = tl.broadcast_to(r339_value, [$(6144)])
    tl.store($(candidate_vmul_45_ptr) + r339_z, r339_value)
    r340_z = tl.arange($(16))
    r340_value = (tl.cast(r340_z, tl.int64) + tl.full([], $(48), dtype=tl.int64))
    r340_value = tl.broadcast_to(r340_value, [$(16)])
    tl.store($(candidate_vadd_34_ptr) + r340_z, r340_value)
    r341_z = tl.arange($(16))
    r341_value = (tl.load($(candidate_vadd_34_ptr) + r341_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r341_value = tl.broadcast_to(r341_value, [$(16)])
    tl.store($(candidate_vsub_27_ptr) + r341_z, r341_value)
    r342_z = tl.arange($(768))
    r342_value = (tl.cast((r342_z % $(48)), tl.int64) > tl.load($(candidate_vsub_27_ptr) + ((r342_z / $(48)) % $(16)), dtype=tl.int64))
    r342_value = tl.broadcast_to(r342_value, [$(768)])
    tl.store($(candidate_vgt_14_ptr) + r342_z, r342_value)
    r343_z = tl.arange($(6144))
    r343_value = tl.where(tl.load($(candidate_vgt_14_ptr) + ((((r343_z / $(48)) % $(16)) * $(48)) + (r343_z % $(48))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_45_ptr) + ((((r343_z / $(768)) % $(8)) * $(768)) + (((r343_z / $(48)) % $(16)) * $(48)) + (r343_z % $(48))), dtype=tl.float32))
    r343_value = r343_value.to(tl.float32)
    r343_value = tl.broadcast_to(r343_value, [$(6144)])
    tl.store($(candidate_vmasked_fill_14_ptr) + r343_z, r343_value)
    r344_z = tl.arange($(6144))
    r344_value = (tl.load($(candidate_vmasked_fill_14_ptr) + ((((r344_z / $(768)) % $(8)) * $(768)) + (((r344_z / $(48)) % $(16)) * $(48)) + (r344_z % $(48))), dtype=tl.float32) - tl.load($(candidate_vcat_13_ptr) + ((((r344_z / $(48)) % $(16)) * $(8)) + ((r344_z / $(768)) % $(8)) + $(128)), dtype=tl.float32))
    r344_value = r344_value.to(tl.float32)
    r344_value = tl.broadcast_to(r344_value, [$(6144)])
    tl.store($(candidate_vsub_28_ptr) + r344_z, r344_value)
    r345_z = tl.arange($(6144))
    r345_value = tl.exp(tl.load($(candidate_vsub_28_ptr) + ((((r345_z / $(768)) % $(8)) * $(768)) + (((r345_z / $(48)) % $(16)) * $(48)) + (r345_z % $(48))), dtype=tl.float32))
    r345_value = r345_value.to(tl.float32)
    r345_value = tl.broadcast_to(r345_value, [$(6144)])
    tl.store($(candidate_vexp_2_ptr) + r345_z, r345_value)
    r346_z = tl.arange($(16384))
    r346_value = (tl.load($(candidate_vto_copy_109_ptr) + ((((r346_z / $(2048)) % $(8)) * $(2048)) + (((r346_z / $(128)) % $(16)) * $(128)) + (r346_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_110_ptr) + ((((r346_z / $(2048)) % $(8)) * $(2048)) + (((r346_z / $(128)) % $(16)) * $(128)) + (r346_z % $(128))), dtype=tl.float32))
    r346_value = r346_value.to(tl.float32)
    r346_value = tl.broadcast_to(r346_value, [$(16384)])
    tl.store($(candidate_vmul_46_ptr) + r346_z, r346_value)
    r347_z = tl.arange($(128))
    r347_lane = tl.arange($(128))
    r347_value = tl.sum(tl.load($(candidate_vmul_46_ptr) + ((((r347_z[:, None] / $(16)) % $(8)) * $(2048)) + ((r347_z[:, None] % $(16)) * $(128)) + r347_lane[None, :]), dtype=tl.float32), axis=1)
    r347_value = r347_value.to(tl.float32)
    r347_value = tl.broadcast_to(r347_value, [$(128)])
    tl.store($(candidate_vsum_3_ptr) + r347_z, r347_value)
    r348_row = tl.arange($(16))
    r348_col = tl.arange($(48))
    r348_lane = tl.arange($(128))
    for r348_group in range($(8)) {
      r348_left = tl.load($(candidate_vto_copy_109_ptr) + (((((((r348_group % $(8)) * $(2048)) + (r348_row[:, None] * $(128)) + r348_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r348_group % $(8)) * $(2048)) + (r348_row[:, None] * $(128)) + r348_lane[None, :]) / $(128)) % $(16)) * $(128)) + ((((r348_group % $(8)) * $(2048)) + (r348_row[:, None] * $(128)) + r348_lane[None, :]) % $(128))), dtype=tl.float32)
      r348_right = tl.load($(candidate_vto_copy_108_ptr) + (((((((r348_group % $(8)) * $(6144)) + (r348_lane[:, None] * $(48)) + r348_col[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r348_group % $(8)) * $(6144)) + (r348_lane[:, None] * $(48)) + r348_col[None, :]) % $(48)) * $(128)) + (((((r348_group % $(8)) * $(6144)) + (r348_lane[:, None] * $(48)) + r348_col[None, :]) / $(48)) % $(128))), dtype=tl.float32)
      r348_value = tl.dot(r348_left, r348_right)
      r348_value = r348_value.to(tl.float32)
      tl.store($(candidate_vbmm_35_ptr) + ((r348_group * $(768)) + (r348_row[:, None] * $(48)) + r348_col[None, :]), r348_value)
    }
    r349_z = tl.arange($(6144))
    r349_value = (tl.load($(candidate_vbmm_35_ptr) + ((((r349_z / $(768)) % $(8)) * $(768)) + (((r349_z / $(48)) % $(16)) * $(48)) + (r349_z % $(48))), dtype=tl.float32) - tl.load($(candidate_vsum_3_ptr) + ((((r349_z / $(768)) % $(8)) * $(16)) + ((r349_z / $(48)) % $(16))), dtype=tl.float32))
    r349_value = r349_value.to(tl.float32)
    r349_value = tl.broadcast_to(r349_value, [$(6144)])
    tl.store($(candidate_vsub_29_ptr) + r349_z, r349_value)
    r350_z = tl.arange($(6144))
    r350_value = (tl.load($(candidate_vexp_2_ptr) + ((((r350_z / $(768)) % $(8)) * $(768)) + (((r350_z / $(48)) % $(16)) * $(48)) + (r350_z % $(48))), dtype=tl.float32) * tl.load($(candidate_vsub_29_ptr) + ((((r350_z / $(768)) % $(8)) * $(768)) + (((r350_z / $(48)) % $(16)) * $(48)) + (r350_z % $(48))), dtype=tl.float32))
    r350_value = r350_value.to(tl.float32)
    r350_value = tl.broadcast_to(r350_value, [$(6144)])
    tl.store($(candidate_vmul_47_ptr) + r350_z, r350_value)
    r351_row = tl.arange($(48))
    r351_col = tl.arange($(192))
    r351_lane = tl.arange($(16))
    for r351_group in range($(8)) {
      r351_left = tl.load($(candidate_vmul_47_ptr) + (((((((r351_group % $(8)) * $(768)) + (r351_row[:, None] * $(16)) + r351_lane[None, :]) / $(768)) % $(8)) * $(768)) + (((((r351_group % $(8)) * $(768)) + (r351_row[:, None] * $(16)) + r351_lane[None, :]) % $(16)) * $(48)) + (((((r351_group % $(8)) * $(768)) + (r351_row[:, None] * $(16)) + r351_lane[None, :]) / $(16)) % $(48))), dtype=tl.float32)
      r351_right = tl.load($(candidate_vto_copy_106_ptr) + (((((((r351_group % $(8)) * $(3072)) + (r351_lane[:, None] * $(192)) + r351_col[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r351_group % $(8)) * $(3072)) + (r351_lane[:, None] * $(192)) + r351_col[None, :]) / $(192)) % $(16)) * $(192)) + ((((r351_group % $(8)) * $(3072)) + (r351_lane[:, None] * $(192)) + r351_col[None, :]) % $(192))), dtype=tl.float32)
      r351_value = tl.dot(r351_left, r351_right)
      r351_value = r351_value.to(tl.float32)
      tl.store($(candidate_vbmm_37_ptr) + ((r351_group * $(9216)) + (r351_row[:, None] * $(192)) + r351_col[None, :]), r351_value)
    }
    r352_z = tl.arange($(73728))
    r352_value = (tl.load($(candidate_vbmm_37_ptr) + ((((r352_z / $(9216)) % $(8)) * $(9216)) + (((r352_z / $(192)) % $(48)) * $(192)) + (r352_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r352_value = r352_value.to(tl.float32)
    r352_value = tl.broadcast_to(r352_value, [$(73728)])
    tl.store($(candidate_vmul_49_ptr) + r352_z, r352_value)
    r353_z = tl.arange($(73728))
    r353_value = tl.load($(candidate_vmul_49_ptr) + ((((r353_z / $(192)) % $(8)) * $(9216)) + (((r353_z / $(1536)) % $(48)) * $(192)) + (r353_z % $(192))), dtype=tl.float32)
    r353_value = r353_value.to(tl.bfloat16)
    r353_value = tl.broadcast_to(r353_value, [$(73728)])
    tl.store($(candidate_vto_copy_112_ptr) + r353_z, r353_value)
    r354_row = tl.arange($(48))
    r354_col = tl.arange($(128))
    r354_lane = tl.arange($(16))
    for r354_group in range($(8)) {
      r354_left = tl.load($(candidate_vexp_2_ptr) + (((((((r354_group % $(8)) * $(768)) + (r354_row[:, None] * $(16)) + r354_lane[None, :]) / $(768)) % $(8)) * $(768)) + (((((r354_group % $(8)) * $(768)) + (r354_row[:, None] * $(16)) + r354_lane[None, :]) % $(16)) * $(48)) + (((((r354_group % $(8)) * $(768)) + (r354_row[:, None] * $(16)) + r354_lane[None, :]) / $(16)) % $(48))), dtype=tl.float32)
      r354_right = tl.load($(candidate_vto_copy_109_ptr) + (((((((r354_group % $(8)) * $(2048)) + (r354_lane[:, None] * $(128)) + r354_col[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r354_group % $(8)) * $(2048)) + (r354_lane[:, None] * $(128)) + r354_col[None, :]) / $(128)) % $(16)) * $(128)) + ((((r354_group % $(8)) * $(2048)) + (r354_lane[:, None] * $(128)) + r354_col[None, :]) % $(128))), dtype=tl.float32)
      r354_value = tl.dot(r354_left, r354_right)
      r354_value = r354_value.to(tl.float32)
      tl.store($(candidate_vbmm_38_ptr) + ((r354_group * $(6144)) + (r354_row[:, None] * $(128)) + r354_col[None, :]), r354_value)
    }
    r355_z = tl.arange($(49152))
    r355_value = tl.load($(candidate_vbmm_38_ptr) + ((((r355_z / $(128)) % $(8)) * $(6144)) + (((r355_z / $(1024)) % $(48)) * $(128)) + (r355_z % $(128))), dtype=tl.float32)
    r355_value = r355_value.to(tl.bfloat16)
    r355_value = tl.broadcast_to(r355_value, [$(49152)])
    tl.store($(candidate_vto_copy_113_ptr) + r355_z, r355_value)
    r356_z = tl.arange($(98304))
    r356_value = tl.where(((r356_z / $(1536)) % $(64)) < $(48), tl.load($(candidate_vto_copy_112_ptr) + ((((((r356_z / $(1536)) % $(64)) - $(0)) % $(48)) * $(1536)) + (((r356_z / $(192)) % $(8)) * $(192)) + (r356_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r356_value = r356_value.to(tl.bfloat16)
    r356_value = tl.broadcast_to(r356_value, [$(98304)])
    tl.store($(candidate_vcat_21_ptr) + r356_z, r356_value)
    r357_z = tl.arange($(65536))
    r357_value = tl.where(((r357_z / $(1024)) % $(64)) < $(48), tl.load($(candidate_vto_copy_113_ptr) + ((((((r357_z / $(1024)) % $(64)) - $(0)) % $(48)) * $(1024)) + (((r357_z / $(128)) % $(8)) * $(128)) + (r357_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r357_value = r357_value.to(tl.bfloat16)
    r357_value = tl.broadcast_to(r357_value, [$(65536)])
    tl.store($(candidate_vcat_22_ptr) + r357_z, r357_value)
    r358_z = tl.arange($(196608))
    r358_value = tl.where(((r358_z / $(1536)) % $(128)) < $(48), tl.load($(candidate_vcat_17_ptr) + ((((((r358_z / $(1536)) % $(128)) - $(0)) % $(48)) * $(1536)) + (((r358_z / $(192)) % $(8)) * $(192)) + (r358_z % $(192))), dtype=tl.bfloat16), tl.where(((r358_z / $(1536)) % $(128)) < $(64), tl.load($(candidate_vcat_19_ptr) + ((((((r358_z / $(1536)) % $(128)) - $(48)) % $(16)) * $(1536)) + (((r358_z / $(192)) % $(8)) * $(192)) + (r358_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vcat_21_ptr) + ((((((r358_z / $(1536)) % $(128)) - $(64)) % $(64)) * $(1536)) + (((r358_z / $(192)) % $(8)) * $(192)) + (r358_z % $(192))), dtype=tl.bfloat16)))
    r358_value = r358_value.to(tl.bfloat16)
    r358_value = tl.broadcast_to(r358_value, [$(196608)])
    tl.store($(candidate_vcat_24_ptr) + r358_z, r358_value)
    r359_z = tl.arange($(131072))
    r359_value = tl.where(((r359_z / $(1024)) % $(128)) < $(48), tl.load($(candidate_vcat_18_ptr) + ((((((r359_z / $(1024)) % $(128)) - $(0)) % $(48)) * $(1024)) + (((r359_z / $(128)) % $(8)) * $(128)) + (r359_z % $(128))), dtype=tl.bfloat16), tl.where(((r359_z / $(1024)) % $(128)) < $(64), tl.load($(candidate_vcat_20_ptr) + ((((((r359_z / $(1024)) % $(128)) - $(48)) % $(16)) * $(1024)) + (((r359_z / $(128)) % $(8)) * $(128)) + (r359_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_22_ptr) + ((((((r359_z / $(1024)) % $(128)) - $(64)) % $(64)) * $(1024)) + (((r359_z / $(128)) % $(8)) * $(128)) + (r359_z % $(128))), dtype=tl.bfloat16)))
    r359_value = r359_value.to(tl.bfloat16)
    r359_value = tl.broadcast_to(r359_value, [$(131072)])
    tl.store($(candidate_vcat_25_ptr) + r359_z, r359_value)
    r360_row = tl.arange($(1024))
    r360_col = tl.arange($(512))
    r360_lane = tl.arange($(128))
    for r360_group in range($(1)) {
      r360_left = tl.load($(candidate_vcat_24_ptr) + (((((r360_row[:, None] * $(128)) + r360_lane[None, :]) % $(128)) * $(1536)) + (((((r360_row[:, None] * $(128)) + r360_lane[None, :]) / $(16384)) % $(8)) * $(192)) + ((((r360_row[:, None] * $(128)) + r360_lane[None, :]) / $(128)) % $(128))), dtype=tl.bfloat16)
      r360_right = tl.load($(candidate_vindex_2_ptr) + ((((((r360_lane[:, None] * $(512)) + r360_col[None, :]) / $(512)) % $(128)) * $(576)) + (((r360_lane[:, None] * $(512)) + r360_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r360_value = tl.dot(r360_left, r360_right)
      r360_value = r360_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_41_ptr) + ((r360_group * $(524288)) + (r360_row[:, None] * $(512)) + r360_col[None, :]), r360_value)
    }
    r361_row = tl.arange($(1024))
    r361_col = tl.arange($(512))
    r361_lane = tl.arange($(128))
    for r361_group in range($(1)) {
      r361_left = tl.load($(candidate_vcat_25_ptr) + (((((r361_row[:, None] * $(128)) + r361_lane[None, :]) % $(128)) * $(1024)) + (((((r361_row[:, None] * $(128)) + r361_lane[None, :]) / $(16384)) % $(8)) * $(128)) + ((((r361_row[:, None] * $(128)) + r361_lane[None, :]) / $(128)) % $(128))), dtype=tl.bfloat16)
      r361_right = tl.load($(candidate_vindex_2_ptr) + ((((((r361_lane[:, None] * $(512)) + r361_col[None, :]) / $(512)) % $(128)) * $(576)) + (((r361_lane[:, None] * $(512)) + r361_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r361_value = tl.dot(r361_left, r361_right)
      r361_value = r361_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_42_ptr) + ((r361_group * $(524288)) + (r361_row[:, None] * $(512)) + r361_col[None, :]), r361_value)
    }
    r362_z = tl.arange($(1048576))
    r362_value = tl.where(((r362_z / $(512)) % $(256)) < $(128), tl.load($(candidate_vbmm_41_ptr) + ((((((((r362_z / $(131072)) % $(8)) * $(65536)) + ((((r362_z / $(512)) % $(256)) - $(0)) * $(512)) + (r362_z % $(512))) / $(65536)) % $(8)) * $(65536)) + ((((((r362_z / $(131072)) % $(8)) * $(128)) + (((r362_z / $(512)) % $(256)) - $(0))) % $(128)) * $(512)) + (r362_z % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vbmm_42_ptr) + ((((((((r362_z / $(131072)) % $(8)) * $(65536)) + ((((r362_z / $(512)) % $(256)) - $(128)) * $(512)) + (r362_z % $(512))) / $(65536)) % $(8)) * $(65536)) + ((((((r362_z / $(131072)) % $(8)) * $(128)) + (((r362_z / $(512)) % $(256)) - $(128))) % $(128)) * $(512)) + (r362_z % $(512))), dtype=tl.bfloat16))
    r362_value = r362_value.to(tl.bfloat16)
    r362_value = tl.broadcast_to(r362_value, [$(1048576)])
    tl.store($(candidate_vcat_26_ptr) + r362_z, r362_value)
    r363_row = tl.arange($(128))
    r363_col = tl.arange($(1024))
    r363_lane = tl.arange($(512))
    for r363_group in range($(1)) {
      r363_left = tl.load($(candidate_vindex_2_ptr) + ((((((r363_row[:, None] * $(512)) + r363_lane[None, :]) / $(512)) % $(128)) * $(576)) + (((r363_row[:, None] * $(512)) + r363_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r363_right = tl.load($(candidate_vto_copy_20_ptr) + ((((((r363_lane[:, None] * $(1024)) + r363_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r363_lane[:, None] * $(1024)) + r363_col[None, :]) % $(128)) * $(512)) + ((((r363_lane[:, None] * $(1024)) + r363_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r363_value = tl.dot(r363_left, r363_right)
      r363_value = r363_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_43_ptr) + ((r363_group * $(131072)) + (r363_row[:, None] * $(1024)) + r363_col[None, :]), r363_value)
    }
    r364_row = tl.arange($(128))
    r364_col = tl.arange($(1024))
    r364_lane = tl.arange($(512))
    for r364_group in range($(1)) {
      r364_left = tl.load($(candidate_vindex_2_ptr) + ((((((r364_row[:, None] * $(512)) + r364_lane[None, :]) / $(512)) % $(128)) * $(576)) + (((r364_row[:, None] * $(512)) + r364_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r364_right = tl.load($(candidate_vto_copy_20_ptr) + ((((((r364_lane[:, None] * $(1024)) + r364_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r364_lane[:, None] * $(1024)) + r364_col[None, :]) % $(128)) * $(512)) + ((((r364_lane[:, None] * $(1024)) + r364_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r364_value = tl.dot(r364_left, r364_right)
      r364_value = r364_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_44_ptr) + ((r364_group * $(131072)) + (r364_row[:, None] * $(1024)) + r364_col[None, :]), r364_value)
    }
    r365_z = tl.arange($(49152))
    r365_value = tl.where((r365_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r365_z / $(1536)) % $(32)) * $(1024)) + (((r365_z / $(192)) % $(8)) * $(128)) + ((r365_z % $(192)) - $(0)) + $(65536)) / $(1024)) % $(128)) * $(1536)) + (((((((r365_z / $(1536)) % $(32)) * $(1024)) + (((r365_z / $(192)) % $(8)) * $(128)) + ((r365_z % $(192)) - $(0)) + $(65536)) / $(128)) % $(8)) * $(192)) + (((((r365_z / $(1536)) % $(32)) * $(1024)) + (((r365_z / $(192)) % $(8)) * $(128)) + ((r365_z % $(192)) - $(0)) + $(65536)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vadd_6_ptr) + ((((((((r365_z / $(1536)) % $(32)) * $(512)) + (((r365_z / $(192)) % $(8)) * $(64)) + ((r365_z % $(192)) - $(128)) + $(32768)) / $(512)) % $(128)) * $(512)) + (((((((r365_z / $(1536)) % $(32)) * $(512)) + (((r365_z / $(192)) % $(8)) * $(64)) + ((r365_z % $(192)) - $(128)) + $(32768)) / $(64)) % $(8)) * $(64)) + (((((r365_z / $(1536)) % $(32)) * $(512)) + (((r365_z / $(192)) % $(8)) * $(64)) + ((r365_z % $(192)) - $(128)) + $(32768)) % $(64))), dtype=tl.bfloat16))
    r365_value = r365_value.to(tl.bfloat16)
    r365_value = tl.broadcast_to(r365_value, [$(49152)])
    tl.store($(candidate_vcat_27_ptr) + r365_z, r365_value)
    r366_z = tl.arange($(196608))
    r366_value = tl.where((r366_z % $(192)) < $(128), tl.load($(candidate_vbmm_43_ptr) + ((((((((r366_z / $(1536)) % $(128)) * $(1024)) + (((r366_z / $(192)) % $(8)) * $(128)) + ((r366_z % $(192)) - $(0))) / $(1024)) % $(128)) * $(1024)) + (((((((r366_z / $(1536)) % $(128)) * $(1024)) + (((r366_z / $(192)) % $(8)) * $(128)) + ((r366_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r366_z / $(1536)) % $(128)) * $(1024)) + (((r366_z / $(192)) % $(8)) * $(128)) + ((r366_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vindex_2_ptr) + ((((r366_z / $(1536)) % $(128)) * $(576)) + ((((r366_z % $(192)) - $(128)) + $(512)) % $(576))), dtype=tl.bfloat16))
    r366_value = r366_value.to(tl.bfloat16)
    r366_value = tl.broadcast_to(r366_value, [$(196608)])
    tl.store($(candidate_vcat_28_ptr) + r366_z, r366_value)
    r367_z = tl.arange($(18432))
    r367_value = tl.load($(candidate_vcat_27_ptr) + ((((r367_z / $(192)) % $(12)) * $(1536)) + (((r367_z / $(2304)) % $(8)) * $(192)) + (r367_z % $(192))), dtype=tl.bfloat16)
    r367_value = r367_value.to(tl.float32)
    r367_value = tl.broadcast_to(r367_value, [$(18432)])
    tl.store($(candidate_vto_copy_114_ptr) + r367_z, r367_value)
    r368_z = tl.arange($(36864))
    r368_value = tl.load($(candidate_vcat_28_ptr) + ((((r368_z / $(192)) % $(24)) * $(1536)) + (((r368_z / $(4608)) % $(8)) * $(192)) + (r368_z % $(192))), dtype=tl.bfloat16)
    r368_value = r368_value.to(tl.float32)
    r368_value = tl.broadcast_to(r368_value, [$(36864)])
    tl.store($(candidate_vto_copy_115_ptr) + r368_z, r368_value)
    r369_z = tl.arange($(24576))
    r369_value = tl.load($(candidate_vbmm_44_ptr) + ((((r369_z / $(128)) % $(24)) * $(1024)) + (((r369_z / $(3072)) % $(8)) * $(128)) + (r369_z % $(128))), dtype=tl.bfloat16)
    r369_value = r369_value.to(tl.float32)
    r369_value = tl.broadcast_to(r369_value, [$(24576)])
    tl.store($(candidate_vto_copy_116_ptr) + r369_z, r369_value)
    r370_z = tl.arange($(12288))
    r370_value = tl.load($(candidate_vmm_28_ptr) + ((((r370_z / $(128)) % $(12)) * $(1024)) + (((r370_z / $(1536)) % $(8)) * $(128)) + (r370_z % $(128)) + $(65536)), dtype=tl.bfloat16)
    r370_value = r370_value.to(tl.float32)
    r370_value = tl.broadcast_to(r370_value, [$(12288)])
    tl.store($(candidate_vto_copy_117_ptr) + r370_z, r370_value)
    r371_z = tl.arange($(12288))
    r371_value = tl.load($(candidate_vbmm_16_ptr) + ((((r371_z / $(1536)) % $(8)) * $(4096)) + (((r371_z / $(128)) % $(12)) * $(128)) + (r371_z % $(128))), dtype=tl.bfloat16)
    r371_value = r371_value.to(tl.float32)
    r371_value = tl.broadcast_to(r371_value, [$(12288)])
    tl.store($(candidate_vto_copy_118_ptr) + r371_z, r371_value)
    r372_row = tl.arange($(12))
    r372_col = tl.arange($(24))
    r372_lane = tl.arange($(192))
    for r372_group in range($(8)) {
      r372_left = tl.load($(candidate_vto_copy_114_ptr) + (((((((r372_group % $(8)) * $(2304)) + (r372_row[:, None] * $(192)) + r372_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r372_group % $(8)) * $(2304)) + (r372_row[:, None] * $(192)) + r372_lane[None, :]) / $(192)) % $(12)) * $(192)) + ((((r372_group % $(8)) * $(2304)) + (r372_row[:, None] * $(192)) + r372_lane[None, :]) % $(192))), dtype=tl.float32)
      r372_right = tl.load($(candidate_vto_copy_115_ptr) + (((((((r372_group % $(8)) * $(4608)) + (r372_lane[:, None] * $(24)) + r372_col[None, :]) / $(4608)) % $(8)) * $(4608)) + (((((r372_group % $(8)) * $(4608)) + (r372_lane[:, None] * $(24)) + r372_col[None, :]) % $(24)) * $(192)) + (((((r372_group % $(8)) * $(4608)) + (r372_lane[:, None] * $(24)) + r372_col[None, :]) / $(24)) % $(192))), dtype=tl.float32)
      r372_value = tl.dot(r372_left, r372_right)
      r372_value = r372_value.to(tl.float32)
      tl.store($(candidate_vbmm_45_ptr) + ((r372_group * $(288)) + (r372_row[:, None] * $(24)) + r372_col[None, :]), r372_value)
    }
    r373_z = tl.arange($(2304))
    r373_value = (tl.load($(candidate_vbmm_45_ptr) + ((((r373_z / $(288)) % $(8)) * $(288)) + (((r373_z / $(24)) % $(12)) * $(24)) + (r373_z % $(24))), dtype=tl.float32) * 0.07216878364870322)
    r373_value = r373_value.to(tl.float32)
    r373_value = tl.broadcast_to(r373_value, [$(2304)])
    tl.store($(candidate_vmul_50_ptr) + r373_z, r373_value)
    r374_z = tl.arange($(12))
    r374_value = (tl.cast(r374_z, tl.int64) + tl.full([], $(24), dtype=tl.int64))
    r374_value = tl.broadcast_to(r374_value, [$(12)])
    tl.store($(candidate_vadd_36_ptr) + r374_z, r374_value)
    r375_z = tl.arange($(12))
    r375_value = (tl.load($(candidate_vadd_36_ptr) + r375_z, dtype=tl.int64) - tl.full([], $(12), dtype=tl.int64))
    r375_value = tl.broadcast_to(r375_value, [$(12)])
    tl.store($(candidate_vsub_30_ptr) + r375_z, r375_value)
    r376_z = tl.arange($(288))
    r376_value = (tl.cast((r376_z % $(24)), tl.int64) > tl.load($(candidate_vsub_30_ptr) + ((r376_z / $(24)) % $(12)), dtype=tl.int64))
    r376_value = tl.broadcast_to(r376_value, [$(288)])
    tl.store($(candidate_vgt_15_ptr) + r376_z, r376_value)
    r377_z = tl.arange($(2304))
    r377_value = tl.where(tl.load($(candidate_vgt_15_ptr) + ((((r377_z / $(24)) % $(12)) * $(24)) + (r377_z % $(24))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_50_ptr) + ((((r377_z / $(288)) % $(8)) * $(288)) + (((r377_z / $(24)) % $(12)) * $(24)) + (r377_z % $(24))), dtype=tl.float32))
    r377_value = r377_value.to(tl.float32)
    r377_value = tl.broadcast_to(r377_value, [$(2304)])
    tl.store($(candidate_vmasked_fill_15_ptr) + r377_z, r377_value)
    r378_z = tl.arange($(2304))
    r378_value = (tl.load($(candidate_vmasked_fill_15_ptr) + ((((r378_z / $(288)) % $(8)) * $(288)) + (((r378_z / $(24)) % $(12)) * $(24)) + (r378_z % $(24))), dtype=tl.float32) - tl.load($(candidate_vcat_11_ptr) + ((((r378_z / $(24)) % $(12)) * $(8)) + ((r378_z / $(288)) % $(8))), dtype=tl.float32))
    r378_value = r378_value.to(tl.float32)
    r378_value = tl.broadcast_to(r378_value, [$(2304)])
    tl.store($(candidate_vsub_31_ptr) + r378_z, r378_value)
    r379_z = tl.arange($(2304))
    r379_value = tl.exp(tl.load($(candidate_vsub_31_ptr) + ((((r379_z / $(288)) % $(8)) * $(288)) + (((r379_z / $(24)) % $(12)) * $(24)) + (r379_z % $(24))), dtype=tl.float32))
    r379_value = r379_value.to(tl.float32)
    r379_value = tl.broadcast_to(r379_value, [$(2304)])
    tl.store($(candidate_vexp_3_ptr) + r379_z, r379_value)
    r380_z = tl.arange($(12288))
    r380_value = (tl.load($(candidate_vto_copy_117_ptr) + ((((r380_z / $(1536)) % $(8)) * $(1536)) + (((r380_z / $(128)) % $(12)) * $(128)) + (r380_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_118_ptr) + ((((r380_z / $(1536)) % $(8)) * $(1536)) + (((r380_z / $(128)) % $(12)) * $(128)) + (r380_z % $(128))), dtype=tl.float32))
    r380_value = r380_value.to(tl.float32)
    r380_value = tl.broadcast_to(r380_value, [$(12288)])
    tl.store($(candidate_vmul_51_ptr) + r380_z, r380_value)
    r381_z = tl.arange($(96))
    r381_lane = tl.arange($(128))
    r381_value = tl.sum(tl.load($(candidate_vmul_51_ptr) + ((((r381_z[:, None] / $(12)) % $(8)) * $(1536)) + ((r381_z[:, None] % $(12)) * $(128)) + r381_lane[None, :]), dtype=tl.float32), axis=1)
    r381_value = r381_value.to(tl.float32)
    r381_value = tl.broadcast_to(r381_value, [$(96)])
    tl.store($(candidate_vsum_5_ptr) + r381_z, r381_value)
    r382_row = tl.arange($(12))
    r382_col = tl.arange($(24))
    r382_lane = tl.arange($(128))
    for r382_group in range($(8)) {
      r382_left = tl.load($(candidate_vto_copy_117_ptr) + (((((((r382_group % $(8)) * $(1536)) + (r382_row[:, None] * $(128)) + r382_lane[None, :]) / $(1536)) % $(8)) * $(1536)) + ((((((r382_group % $(8)) * $(1536)) + (r382_row[:, None] * $(128)) + r382_lane[None, :]) / $(128)) % $(12)) * $(128)) + ((((r382_group % $(8)) * $(1536)) + (r382_row[:, None] * $(128)) + r382_lane[None, :]) % $(128))), dtype=tl.float32)
      r382_right = tl.load($(candidate_vto_copy_116_ptr) + (((((((r382_group % $(8)) * $(3072)) + (r382_lane[:, None] * $(24)) + r382_col[None, :]) / $(3072)) % $(8)) * $(3072)) + (((((r382_group % $(8)) * $(3072)) + (r382_lane[:, None] * $(24)) + r382_col[None, :]) % $(24)) * $(128)) + (((((r382_group % $(8)) * $(3072)) + (r382_lane[:, None] * $(24)) + r382_col[None, :]) / $(24)) % $(128))), dtype=tl.float32)
      r382_value = tl.dot(r382_left, r382_right)
      r382_value = r382_value.to(tl.float32)
      tl.store($(candidate_vbmm_46_ptr) + ((r382_group * $(288)) + (r382_row[:, None] * $(24)) + r382_col[None, :]), r382_value)
    }
    r383_z = tl.arange($(2304))
    r383_value = (tl.load($(candidate_vbmm_46_ptr) + ((((r383_z / $(288)) % $(8)) * $(288)) + (((r383_z / $(24)) % $(12)) * $(24)) + (r383_z % $(24))), dtype=tl.float32) - tl.load($(candidate_vsum_5_ptr) + ((((r383_z / $(288)) % $(8)) * $(12)) + ((r383_z / $(24)) % $(12))), dtype=tl.float32))
    r383_value = r383_value.to(tl.float32)
    r383_value = tl.broadcast_to(r383_value, [$(2304)])
    tl.store($(candidate_vsub_32_ptr) + r383_z, r383_value)
    r384_z = tl.arange($(2304))
    r384_value = (tl.load($(candidate_vexp_3_ptr) + ((((r384_z / $(288)) % $(8)) * $(288)) + (((r384_z / $(24)) % $(12)) * $(24)) + (r384_z % $(24))), dtype=tl.float32) * tl.load($(candidate_vsub_32_ptr) + ((((r384_z / $(288)) % $(8)) * $(288)) + (((r384_z / $(24)) % $(12)) * $(24)) + (r384_z % $(24))), dtype=tl.float32))
    r384_value = r384_value.to(tl.float32)
    r384_value = tl.broadcast_to(r384_value, [$(2304)])
    tl.store($(candidate_vmul_52_ptr) + r384_z, r384_value)
    r385_row = tl.arange($(24))
    r385_col = tl.arange($(192))
    r385_lane = tl.arange($(12))
    for r385_group in range($(8)) {
      r385_left = tl.load($(candidate_vmul_52_ptr) + (((((((r385_group % $(8)) * $(288)) + (r385_row[:, None] * $(12)) + r385_lane[None, :]) / $(288)) % $(8)) * $(288)) + (((((r385_group % $(8)) * $(288)) + (r385_row[:, None] * $(12)) + r385_lane[None, :]) % $(12)) * $(24)) + (((((r385_group % $(8)) * $(288)) + (r385_row[:, None] * $(12)) + r385_lane[None, :]) / $(12)) % $(24))), dtype=tl.float32)
      r385_right = tl.load($(candidate_vto_copy_114_ptr) + (((((((r385_group % $(8)) * $(2304)) + (r385_lane[:, None] * $(192)) + r385_col[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r385_group % $(8)) * $(2304)) + (r385_lane[:, None] * $(192)) + r385_col[None, :]) / $(192)) % $(12)) * $(192)) + ((((r385_group % $(8)) * $(2304)) + (r385_lane[:, None] * $(192)) + r385_col[None, :]) % $(192))), dtype=tl.float32)
      r385_value = tl.dot(r385_left, r385_right)
      r385_value = r385_value.to(tl.float32)
      tl.store($(candidate_vbmm_48_ptr) + ((r385_group * $(4608)) + (r385_row[:, None] * $(192)) + r385_col[None, :]), r385_value)
    }
    r386_z = tl.arange($(36864))
    r386_value = (tl.load($(candidate_vbmm_48_ptr) + ((((r386_z / $(4608)) % $(8)) * $(4608)) + (((r386_z / $(192)) % $(24)) * $(192)) + (r386_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r386_value = r386_value.to(tl.float32)
    r386_value = tl.broadcast_to(r386_value, [$(36864)])
    tl.store($(candidate_vmul_54_ptr) + r386_z, r386_value)
    r387_z = tl.arange($(36864))
    r387_value = tl.load($(candidate_vmul_54_ptr) + ((((r387_z / $(192)) % $(8)) * $(4608)) + (((r387_z / $(1536)) % $(24)) * $(192)) + (r387_z % $(192))), dtype=tl.float32)
    r387_value = r387_value.to(tl.bfloat16)
    r387_value = tl.broadcast_to(r387_value, [$(36864)])
    tl.store($(candidate_vto_copy_120_ptr) + r387_z, r387_value)
    r388_row = tl.arange($(24))
    r388_col = tl.arange($(128))
    r388_lane = tl.arange($(12))
    for r388_group in range($(8)) {
      r388_left = tl.load($(candidate_vexp_3_ptr) + (((((((r388_group % $(8)) * $(288)) + (r388_row[:, None] * $(12)) + r388_lane[None, :]) / $(288)) % $(8)) * $(288)) + (((((r388_group % $(8)) * $(288)) + (r388_row[:, None] * $(12)) + r388_lane[None, :]) % $(12)) * $(24)) + (((((r388_group % $(8)) * $(288)) + (r388_row[:, None] * $(12)) + r388_lane[None, :]) / $(12)) % $(24))), dtype=tl.float32)
      r388_right = tl.load($(candidate_vto_copy_117_ptr) + (((((((r388_group % $(8)) * $(1536)) + (r388_lane[:, None] * $(128)) + r388_col[None, :]) / $(1536)) % $(8)) * $(1536)) + ((((((r388_group % $(8)) * $(1536)) + (r388_lane[:, None] * $(128)) + r388_col[None, :]) / $(128)) % $(12)) * $(128)) + ((((r388_group % $(8)) * $(1536)) + (r388_lane[:, None] * $(128)) + r388_col[None, :]) % $(128))), dtype=tl.float32)
      r388_value = tl.dot(r388_left, r388_right)
      r388_value = r388_value.to(tl.float32)
      tl.store($(candidate_vbmm_49_ptr) + ((r388_group * $(3072)) + (r388_row[:, None] * $(128)) + r388_col[None, :]), r388_value)
    }
    r389_z = tl.arange($(24576))
    r389_value = tl.load($(candidate_vbmm_49_ptr) + ((((r389_z / $(128)) % $(8)) * $(3072)) + (((r389_z / $(1024)) % $(24)) * $(128)) + (r389_z % $(128))), dtype=tl.float32)
    r389_value = r389_value.to(tl.bfloat16)
    r389_value = tl.broadcast_to(r389_value, [$(24576)])
    tl.store($(candidate_vto_copy_121_ptr) + r389_z, r389_value)
    r390_z = tl.arange($(73728))
    r390_value = tl.where(((r390_z / $(1536)) % $(48)) < $(24), tl.load($(candidate_vto_copy_120_ptr) + ((((((r390_z / $(1536)) % $(48)) - $(0)) % $(24)) * $(1536)) + (((r390_z / $(192)) % $(8)) * $(192)) + (r390_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r390_value = r390_value.to(tl.bfloat16)
    r390_value = tl.broadcast_to(r390_value, [$(73728)])
    tl.store($(candidate_vcat_29_ptr) + r390_z, r390_value)
    r391_z = tl.arange($(49152))
    r391_value = tl.where(((r391_z / $(1024)) % $(48)) < $(24), tl.load($(candidate_vto_copy_121_ptr) + ((((((r391_z / $(1024)) % $(48)) - $(0)) % $(24)) * $(1024)) + (((r391_z / $(128)) % $(8)) * $(128)) + (r391_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r391_value = r391_value.to(tl.bfloat16)
    r391_value = tl.broadcast_to(r391_value, [$(49152)])
    tl.store($(candidate_vcat_30_ptr) + r391_z, r391_value)
    r392_z = tl.arange($(6144))
    r392_value = tl.load($(candidate_vcat_27_ptr) + ((((r392_z / $(192)) % $(4)) * $(1536)) + (((r392_z / $(768)) % $(8)) * $(192)) + (r392_z % $(192)) + $(18432)), dtype=tl.bfloat16)
    r392_value = r392_value.to(tl.float32)
    r392_value = tl.broadcast_to(r392_value, [$(6144)])
    tl.store($(candidate_vto_copy_122_ptr) + r392_z, r392_value)
    r393_z = tl.arange($(12288))
    r393_value = tl.load($(candidate_vcat_28_ptr) + ((((r393_z / $(192)) % $(8)) * $(1536)) + (((r393_z / $(1536)) % $(8)) * $(192)) + (r393_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r393_value = r393_value.to(tl.float32)
    r393_value = tl.broadcast_to(r393_value, [$(12288)])
    tl.store($(candidate_vto_copy_123_ptr) + r393_z, r393_value)
    r394_z = tl.arange($(8192))
    r394_value = tl.load($(candidate_vbmm_44_ptr) + ((((r394_z / $(128)) % $(8)) * $(1024)) + (((r394_z / $(1024)) % $(8)) * $(128)) + (r394_z % $(128)) + $(49152)), dtype=tl.bfloat16)
    r394_value = r394_value.to(tl.float32)
    r394_value = tl.broadcast_to(r394_value, [$(8192)])
    tl.store($(candidate_vto_copy_124_ptr) + r394_z, r394_value)
    r395_z = tl.arange($(4096))
    r395_value = tl.load($(candidate_vmm_28_ptr) + ((((r395_z / $(128)) % $(4)) * $(1024)) + (((r395_z / $(512)) % $(8)) * $(128)) + (r395_z % $(128)) + $(77824)), dtype=tl.bfloat16)
    r395_value = r395_value.to(tl.float32)
    r395_value = tl.broadcast_to(r395_value, [$(4096)])
    tl.store($(candidate_vto_copy_125_ptr) + r395_z, r395_value)
    r396_z = tl.arange($(4096))
    r396_value = tl.load($(candidate_vbmm_16_ptr) + ((((r396_z / $(512)) % $(8)) * $(4096)) + (((r396_z / $(128)) % $(4)) * $(128)) + (r396_z % $(128)) + $(1536)), dtype=tl.bfloat16)
    r396_value = r396_value.to(tl.float32)
    r396_value = tl.broadcast_to(r396_value, [$(4096)])
    tl.store($(candidate_vto_copy_126_ptr) + r396_z, r396_value)
    r397_row = tl.arange($(4))
    r397_col = tl.arange($(8))
    r397_lane = tl.arange($(192))
    for r397_group in range($(8)) {
      r397_left = tl.load($(candidate_vto_copy_122_ptr) + (((((((r397_group % $(8)) * $(768)) + (r397_row[:, None] * $(192)) + r397_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r397_group % $(8)) * $(768)) + (r397_row[:, None] * $(192)) + r397_lane[None, :]) / $(192)) % $(4)) * $(192)) + ((((r397_group % $(8)) * $(768)) + (r397_row[:, None] * $(192)) + r397_lane[None, :]) % $(192))), dtype=tl.float32)
      r397_right = tl.load($(candidate_vto_copy_123_ptr) + (((((((r397_group % $(8)) * $(1536)) + (r397_lane[:, None] * $(8)) + r397_col[None, :]) / $(1536)) % $(8)) * $(1536)) + (((((r397_group % $(8)) * $(1536)) + (r397_lane[:, None] * $(8)) + r397_col[None, :]) % $(8)) * $(192)) + (((((r397_group % $(8)) * $(1536)) + (r397_lane[:, None] * $(8)) + r397_col[None, :]) / $(8)) % $(192))), dtype=tl.float32)
      r397_value = tl.dot(r397_left, r397_right)
      r397_value = r397_value.to(tl.float32)
      tl.store($(candidate_vbmm_50_ptr) + ((r397_group * $(32)) + (r397_row[:, None] * $(8)) + r397_col[None, :]), r397_value)
    }
    r398_z = tl.arange($(256))
    r398_value = (tl.load($(candidate_vbmm_50_ptr) + ((((r398_z / $(32)) % $(8)) * $(32)) + (((r398_z / $(8)) % $(4)) * $(8)) + (r398_z % $(8))), dtype=tl.float32) * 0.07216878364870322)
    r398_value = r398_value.to(tl.float32)
    r398_value = tl.broadcast_to(r398_value, [$(256)])
    tl.store($(candidate_vmul_55_ptr) + r398_z, r398_value)
    r399_z = tl.arange($(4))
    r399_value = (tl.cast(r399_z, tl.int64) + tl.full([], $(8), dtype=tl.int64))
    r399_value = tl.broadcast_to(r399_value, [$(4)])
    tl.store($(candidate_vadd_37_ptr) + r399_z, r399_value)
    r400_z = tl.arange($(4))
    r400_value = (tl.load($(candidate_vadd_37_ptr) + r400_z, dtype=tl.int64) - tl.full([], $(4), dtype=tl.int64))
    r400_value = tl.broadcast_to(r400_value, [$(4)])
    tl.store($(candidate_vsub_33_ptr) + r400_z, r400_value)
    r401_z = tl.arange($(32))
    r401_value = (tl.cast((r401_z % $(8)), tl.int64) > tl.load($(candidate_vsub_33_ptr) + ((r401_z / $(8)) % $(4)), dtype=tl.int64))
    r401_value = tl.broadcast_to(r401_value, [$(32)])
    tl.store($(candidate_vgt_16_ptr) + r401_z, r401_value)
    r402_z = tl.arange($(256))
    r402_value = tl.where(tl.load($(candidate_vgt_16_ptr) + ((((r402_z / $(8)) % $(4)) * $(8)) + (r402_z % $(8))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_55_ptr) + ((((r402_z / $(32)) % $(8)) * $(32)) + (((r402_z / $(8)) % $(4)) * $(8)) + (r402_z % $(8))), dtype=tl.float32))
    r402_value = r402_value.to(tl.float32)
    r402_value = tl.broadcast_to(r402_value, [$(256)])
    tl.store($(candidate_vmasked_fill_16_ptr) + r402_z, r402_value)
    r403_z = tl.arange($(256))
    r403_value = (tl.load($(candidate_vmasked_fill_16_ptr) + ((((r403_z / $(32)) % $(8)) * $(32)) + (((r403_z / $(8)) % $(4)) * $(8)) + (r403_z % $(8))), dtype=tl.float32) - tl.load($(candidate_vcat_11_ptr) + ((((r403_z / $(8)) % $(4)) * $(8)) + ((r403_z / $(32)) % $(8)) + $(96)), dtype=tl.float32))
    r403_value = r403_value.to(tl.float32)
    r403_value = tl.broadcast_to(r403_value, [$(256)])
    tl.store($(candidate_vsub_34_ptr) + r403_z, r403_value)
    r404_z = tl.arange($(256))
    r404_value = tl.exp(tl.load($(candidate_vsub_34_ptr) + ((((r404_z / $(32)) % $(8)) * $(32)) + (((r404_z / $(8)) % $(4)) * $(8)) + (r404_z % $(8))), dtype=tl.float32))
    r404_value = r404_value.to(tl.float32)
    r404_value = tl.broadcast_to(r404_value, [$(256)])
    tl.store($(candidate_vexp_4_ptr) + r404_z, r404_value)
    r405_z = tl.arange($(4096))
    r405_value = (tl.load($(candidate_vto_copy_125_ptr) + ((((r405_z / $(512)) % $(8)) * $(512)) + (((r405_z / $(128)) % $(4)) * $(128)) + (r405_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_126_ptr) + ((((r405_z / $(512)) % $(8)) * $(512)) + (((r405_z / $(128)) % $(4)) * $(128)) + (r405_z % $(128))), dtype=tl.float32))
    r405_value = r405_value.to(tl.float32)
    r405_value = tl.broadcast_to(r405_value, [$(4096)])
    tl.store($(candidate_vmul_56_ptr) + r405_z, r405_value)
    r406_z = tl.arange($(32))
    r406_lane = tl.arange($(128))
    r406_value = tl.sum(tl.load($(candidate_vmul_56_ptr) + ((((r406_z[:, None] / $(4)) % $(8)) * $(512)) + ((r406_z[:, None] % $(4)) * $(128)) + r406_lane[None, :]), dtype=tl.float32), axis=1)
    r406_value = r406_value.to(tl.float32)
    r406_value = tl.broadcast_to(r406_value, [$(32)])
    tl.store($(candidate_vsum_6_ptr) + r406_z, r406_value)
    r407_row = tl.arange($(4))
    r407_col = tl.arange($(8))
    r407_lane = tl.arange($(128))
    for r407_group in range($(8)) {
      r407_left = tl.load($(candidate_vto_copy_125_ptr) + (((((((r407_group % $(8)) * $(512)) + (r407_row[:, None] * $(128)) + r407_lane[None, :]) / $(512)) % $(8)) * $(512)) + ((((((r407_group % $(8)) * $(512)) + (r407_row[:, None] * $(128)) + r407_lane[None, :]) / $(128)) % $(4)) * $(128)) + ((((r407_group % $(8)) * $(512)) + (r407_row[:, None] * $(128)) + r407_lane[None, :]) % $(128))), dtype=tl.float32)
      r407_right = tl.load($(candidate_vto_copy_124_ptr) + (((((((r407_group % $(8)) * $(1024)) + (r407_lane[:, None] * $(8)) + r407_col[None, :]) / $(1024)) % $(8)) * $(1024)) + (((((r407_group % $(8)) * $(1024)) + (r407_lane[:, None] * $(8)) + r407_col[None, :]) % $(8)) * $(128)) + (((((r407_group % $(8)) * $(1024)) + (r407_lane[:, None] * $(8)) + r407_col[None, :]) / $(8)) % $(128))), dtype=tl.float32)
      r407_value = tl.dot(r407_left, r407_right)
      r407_value = r407_value.to(tl.float32)
      tl.store($(candidate_vbmm_51_ptr) + ((r407_group * $(32)) + (r407_row[:, None] * $(8)) + r407_col[None, :]), r407_value)
    }
    r408_z = tl.arange($(256))
    r408_value = (tl.load($(candidate_vbmm_51_ptr) + ((((r408_z / $(32)) % $(8)) * $(32)) + (((r408_z / $(8)) % $(4)) * $(8)) + (r408_z % $(8))), dtype=tl.float32) - tl.load($(candidate_vsum_6_ptr) + ((((r408_z / $(32)) % $(8)) * $(4)) + ((r408_z / $(8)) % $(4))), dtype=tl.float32))
    r408_value = r408_value.to(tl.float32)
    r408_value = tl.broadcast_to(r408_value, [$(256)])
    tl.store($(candidate_vsub_35_ptr) + r408_z, r408_value)
    r409_z = tl.arange($(256))
    r409_value = (tl.load($(candidate_vexp_4_ptr) + ((((r409_z / $(32)) % $(8)) * $(32)) + (((r409_z / $(8)) % $(4)) * $(8)) + (r409_z % $(8))), dtype=tl.float32) * tl.load($(candidate_vsub_35_ptr) + ((((r409_z / $(32)) % $(8)) * $(32)) + (((r409_z / $(8)) % $(4)) * $(8)) + (r409_z % $(8))), dtype=tl.float32))
    r409_value = r409_value.to(tl.float32)
    r409_value = tl.broadcast_to(r409_value, [$(256)])
    tl.store($(candidate_vmul_57_ptr) + r409_z, r409_value)
    r410_row = tl.arange($(8))
    r410_col = tl.arange($(192))
    r410_lane = tl.arange($(4))
    for r410_group in range($(8)) {
      r410_left = tl.load($(candidate_vmul_57_ptr) + (((((((r410_group % $(8)) * $(32)) + (r410_row[:, None] * $(4)) + r410_lane[None, :]) / $(32)) % $(8)) * $(32)) + (((((r410_group % $(8)) * $(32)) + (r410_row[:, None] * $(4)) + r410_lane[None, :]) % $(4)) * $(8)) + (((((r410_group % $(8)) * $(32)) + (r410_row[:, None] * $(4)) + r410_lane[None, :]) / $(4)) % $(8))), dtype=tl.float32)
      r410_right = tl.load($(candidate_vto_copy_122_ptr) + (((((((r410_group % $(8)) * $(768)) + (r410_lane[:, None] * $(192)) + r410_col[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r410_group % $(8)) * $(768)) + (r410_lane[:, None] * $(192)) + r410_col[None, :]) / $(192)) % $(4)) * $(192)) + ((((r410_group % $(8)) * $(768)) + (r410_lane[:, None] * $(192)) + r410_col[None, :]) % $(192))), dtype=tl.float32)
      r410_value = tl.dot(r410_left, r410_right)
      r410_value = r410_value.to(tl.float32)
      tl.store($(candidate_vbmm_53_ptr) + ((r410_group * $(1536)) + (r410_row[:, None] * $(192)) + r410_col[None, :]), r410_value)
    }
    r411_z = tl.arange($(12288))
    r411_value = (tl.load($(candidate_vbmm_53_ptr) + ((((r411_z / $(1536)) % $(8)) * $(1536)) + (((r411_z / $(192)) % $(8)) * $(192)) + (r411_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r411_value = r411_value.to(tl.float32)
    r411_value = tl.broadcast_to(r411_value, [$(12288)])
    tl.store($(candidate_vmul_59_ptr) + r411_z, r411_value)
    r412_z = tl.arange($(12288))
    r412_value = tl.load($(candidate_vmul_59_ptr) + ((((r412_z / $(192)) % $(8)) * $(1536)) + (((r412_z / $(1536)) % $(8)) * $(192)) + (r412_z % $(192))), dtype=tl.float32)
    r412_value = r412_value.to(tl.bfloat16)
    r412_value = tl.broadcast_to(r412_value, [$(12288)])
    tl.store($(candidate_vto_copy_128_ptr) + r412_z, r412_value)
    r413_row = tl.arange($(8))
    r413_col = tl.arange($(128))
    r413_lane = tl.arange($(4))
    for r413_group in range($(8)) {
      r413_left = tl.load($(candidate_vexp_4_ptr) + (((((((r413_group % $(8)) * $(32)) + (r413_row[:, None] * $(4)) + r413_lane[None, :]) / $(32)) % $(8)) * $(32)) + (((((r413_group % $(8)) * $(32)) + (r413_row[:, None] * $(4)) + r413_lane[None, :]) % $(4)) * $(8)) + (((((r413_group % $(8)) * $(32)) + (r413_row[:, None] * $(4)) + r413_lane[None, :]) / $(4)) % $(8))), dtype=tl.float32)
      r413_right = tl.load($(candidate_vto_copy_125_ptr) + (((((((r413_group % $(8)) * $(512)) + (r413_lane[:, None] * $(128)) + r413_col[None, :]) / $(512)) % $(8)) * $(512)) + ((((((r413_group % $(8)) * $(512)) + (r413_lane[:, None] * $(128)) + r413_col[None, :]) / $(128)) % $(4)) * $(128)) + ((((r413_group % $(8)) * $(512)) + (r413_lane[:, None] * $(128)) + r413_col[None, :]) % $(128))), dtype=tl.float32)
      r413_value = tl.dot(r413_left, r413_right)
      r413_value = r413_value.to(tl.float32)
      tl.store($(candidate_vbmm_54_ptr) + ((r413_group * $(1024)) + (r413_row[:, None] * $(128)) + r413_col[None, :]), r413_value)
    }
    r414_z = tl.arange($(8192))
    r414_value = tl.load($(candidate_vbmm_54_ptr) + ((((r414_z / $(128)) % $(8)) * $(1024)) + (((r414_z / $(1024)) % $(8)) * $(128)) + (r414_z % $(128))), dtype=tl.float32)
    r414_value = r414_value.to(tl.bfloat16)
    r414_value = tl.broadcast_to(r414_value, [$(8192)])
    tl.store($(candidate_vto_copy_129_ptr) + r414_z, r414_value)
    r415_z = tl.arange($(24576))
    r415_value = tl.where(((r415_z / $(1536)) % $(16)) < $(8), tl.load($(candidate_vto_copy_128_ptr) + ((((((r415_z / $(1536)) % $(16)) - $(0)) % $(8)) * $(1536)) + (((r415_z / $(192)) % $(8)) * $(192)) + (r415_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r415_value = r415_value.to(tl.bfloat16)
    r415_value = tl.broadcast_to(r415_value, [$(24576)])
    tl.store($(candidate_vcat_31_ptr) + r415_z, r415_value)
    r416_z = tl.arange($(16384))
    r416_value = tl.where(((r416_z / $(1024)) % $(16)) < $(8), tl.load($(candidate_vto_copy_129_ptr) + ((((((r416_z / $(1024)) % $(16)) - $(0)) % $(8)) * $(1024)) + (((r416_z / $(128)) % $(8)) * $(128)) + (r416_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r416_value = r416_value.to(tl.bfloat16)
    r416_value = tl.broadcast_to(r416_value, [$(16384)])
    tl.store($(candidate_vcat_32_ptr) + r416_z, r416_value)
    r417_z = tl.arange($(24576))
    r417_value = tl.load($(candidate_vcat_27_ptr) + ((((r417_z / $(192)) % $(16)) * $(1536)) + (((r417_z / $(3072)) % $(8)) * $(192)) + (r417_z % $(192)) + $(24576)), dtype=tl.bfloat16)
    r417_value = r417_value.to(tl.float32)
    r417_value = tl.broadcast_to(r417_value, [$(24576)])
    tl.store($(candidate_vto_copy_130_ptr) + r417_z, r417_value)
    r418_z = tl.arange($(49152))
    r418_value = tl.load($(candidate_vcat_28_ptr) + ((((r418_z / $(192)) % $(32)) * $(1536)) + (((r418_z / $(6144)) % $(8)) * $(192)) + (r418_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r418_value = r418_value.to(tl.float32)
    r418_value = tl.broadcast_to(r418_value, [$(49152)])
    tl.store($(candidate_vto_copy_131_ptr) + r418_z, r418_value)
    r419_z = tl.arange($(32768))
    r419_value = tl.load($(candidate_vbmm_44_ptr) + ((((r419_z / $(128)) % $(32)) * $(1024)) + (((r419_z / $(4096)) % $(8)) * $(128)) + (r419_z % $(128)) + $(65536)), dtype=tl.bfloat16)
    r419_value = r419_value.to(tl.float32)
    r419_value = tl.broadcast_to(r419_value, [$(32768)])
    tl.store($(candidate_vto_copy_132_ptr) + r419_z, r419_value)
    r420_z = tl.arange($(16384))
    r420_value = tl.load($(candidate_vmm_28_ptr) + ((((r420_z / $(128)) % $(16)) * $(1024)) + (((r420_z / $(2048)) % $(8)) * $(128)) + (r420_z % $(128)) + $(81920)), dtype=tl.bfloat16)
    r420_value = r420_value.to(tl.float32)
    r420_value = tl.broadcast_to(r420_value, [$(16384)])
    tl.store($(candidate_vto_copy_133_ptr) + r420_z, r420_value)
    r421_z = tl.arange($(16384))
    r421_value = tl.load($(candidate_vbmm_16_ptr) + ((((r421_z / $(2048)) % $(8)) * $(4096)) + (((r421_z / $(128)) % $(16)) * $(128)) + (r421_z % $(128)) + $(2048)), dtype=tl.bfloat16)
    r421_value = r421_value.to(tl.float32)
    r421_value = tl.broadcast_to(r421_value, [$(16384)])
    tl.store($(candidate_vto_copy_134_ptr) + r421_z, r421_value)
    r422_row = tl.arange($(16))
    r422_col = tl.arange($(32))
    r422_lane = tl.arange($(192))
    for r422_group in range($(8)) {
      r422_left = tl.load($(candidate_vto_copy_130_ptr) + (((((((r422_group % $(8)) * $(3072)) + (r422_row[:, None] * $(192)) + r422_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r422_group % $(8)) * $(3072)) + (r422_row[:, None] * $(192)) + r422_lane[None, :]) / $(192)) % $(16)) * $(192)) + ((((r422_group % $(8)) * $(3072)) + (r422_row[:, None] * $(192)) + r422_lane[None, :]) % $(192))), dtype=tl.float32)
      r422_right = tl.load($(candidate_vto_copy_131_ptr) + (((((((r422_group % $(8)) * $(6144)) + (r422_lane[:, None] * $(32)) + r422_col[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r422_group % $(8)) * $(6144)) + (r422_lane[:, None] * $(32)) + r422_col[None, :]) % $(32)) * $(192)) + (((((r422_group % $(8)) * $(6144)) + (r422_lane[:, None] * $(32)) + r422_col[None, :]) / $(32)) % $(192))), dtype=tl.float32)
      r422_value = tl.dot(r422_left, r422_right)
      r422_value = r422_value.to(tl.float32)
      tl.store($(candidate_vbmm_55_ptr) + ((r422_group * $(512)) + (r422_row[:, None] * $(32)) + r422_col[None, :]), r422_value)
    }
    r423_z = tl.arange($(4096))
    r423_value = (tl.load($(candidate_vbmm_55_ptr) + ((((r423_z / $(512)) % $(8)) * $(512)) + (((r423_z / $(32)) % $(16)) * $(32)) + (r423_z % $(32))), dtype=tl.float32) * 0.07216878364870322)
    r423_value = r423_value.to(tl.float32)
    r423_value = tl.broadcast_to(r423_value, [$(4096)])
    tl.store($(candidate_vmul_60_ptr) + r423_z, r423_value)
    r424_z = tl.arange($(16))
    r424_value = (tl.cast(r424_z, tl.int64) + tl.full([], $(32), dtype=tl.int64))
    r424_value = tl.broadcast_to(r424_value, [$(16)])
    tl.store($(candidate_vadd_38_ptr) + r424_z, r424_value)
    r425_z = tl.arange($(16))
    r425_value = (tl.load($(candidate_vadd_38_ptr) + r425_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r425_value = tl.broadcast_to(r425_value, [$(16)])
    tl.store($(candidate_vsub_36_ptr) + r425_z, r425_value)
    r426_z = tl.arange($(512))
    r426_value = (tl.cast((r426_z % $(32)), tl.int64) > tl.load($(candidate_vsub_36_ptr) + ((r426_z / $(32)) % $(16)), dtype=tl.int64))
    r426_value = tl.broadcast_to(r426_value, [$(512)])
    tl.store($(candidate_vgt_17_ptr) + r426_z, r426_value)
    r427_z = tl.arange($(4096))
    r427_value = tl.where(tl.load($(candidate_vgt_17_ptr) + ((((r427_z / $(32)) % $(16)) * $(32)) + (r427_z % $(32))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_60_ptr) + ((((r427_z / $(512)) % $(8)) * $(512)) + (((r427_z / $(32)) % $(16)) * $(32)) + (r427_z % $(32))), dtype=tl.float32))
    r427_value = r427_value.to(tl.float32)
    r427_value = tl.broadcast_to(r427_value, [$(4096)])
    tl.store($(candidate_vmasked_fill_17_ptr) + r427_z, r427_value)
    r428_z = tl.arange($(4096))
    r428_value = (tl.load($(candidate_vmasked_fill_17_ptr) + ((((r428_z / $(512)) % $(8)) * $(512)) + (((r428_z / $(32)) % $(16)) * $(32)) + (r428_z % $(32))), dtype=tl.float32) - tl.load($(candidate_vcat_11_ptr) + ((((r428_z / $(32)) % $(16)) * $(8)) + ((r428_z / $(512)) % $(8)) + $(128)), dtype=tl.float32))
    r428_value = r428_value.to(tl.float32)
    r428_value = tl.broadcast_to(r428_value, [$(4096)])
    tl.store($(candidate_vsub_37_ptr) + r428_z, r428_value)
    r429_z = tl.arange($(4096))
    r429_value = tl.exp(tl.load($(candidate_vsub_37_ptr) + ((((r429_z / $(512)) % $(8)) * $(512)) + (((r429_z / $(32)) % $(16)) * $(32)) + (r429_z % $(32))), dtype=tl.float32))
    r429_value = r429_value.to(tl.float32)
    r429_value = tl.broadcast_to(r429_value, [$(4096)])
    tl.store($(candidate_vexp_5_ptr) + r429_z, r429_value)
    r430_z = tl.arange($(16384))
    r430_value = (tl.load($(candidate_vto_copy_133_ptr) + ((((r430_z / $(2048)) % $(8)) * $(2048)) + (((r430_z / $(128)) % $(16)) * $(128)) + (r430_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_134_ptr) + ((((r430_z / $(2048)) % $(8)) * $(2048)) + (((r430_z / $(128)) % $(16)) * $(128)) + (r430_z % $(128))), dtype=tl.float32))
    r430_value = r430_value.to(tl.float32)
    r430_value = tl.broadcast_to(r430_value, [$(16384)])
    tl.store($(candidate_vmul_61_ptr) + r430_z, r430_value)
    r431_z = tl.arange($(128))
    r431_lane = tl.arange($(128))
    r431_value = tl.sum(tl.load($(candidate_vmul_61_ptr) + ((((r431_z[:, None] / $(16)) % $(8)) * $(2048)) + ((r431_z[:, None] % $(16)) * $(128)) + r431_lane[None, :]), dtype=tl.float32), axis=1)
    r431_value = r431_value.to(tl.float32)
    r431_value = tl.broadcast_to(r431_value, [$(128)])
    tl.store($(candidate_vsum_7_ptr) + r431_z, r431_value)
    r432_row = tl.arange($(16))
    r432_col = tl.arange($(32))
    r432_lane = tl.arange($(128))
    for r432_group in range($(8)) {
      r432_left = tl.load($(candidate_vto_copy_133_ptr) + (((((((r432_group % $(8)) * $(2048)) + (r432_row[:, None] * $(128)) + r432_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r432_group % $(8)) * $(2048)) + (r432_row[:, None] * $(128)) + r432_lane[None, :]) / $(128)) % $(16)) * $(128)) + ((((r432_group % $(8)) * $(2048)) + (r432_row[:, None] * $(128)) + r432_lane[None, :]) % $(128))), dtype=tl.float32)
      r432_right = tl.load($(candidate_vto_copy_132_ptr) + (((((((r432_group % $(8)) * $(4096)) + (r432_lane[:, None] * $(32)) + r432_col[None, :]) / $(4096)) % $(8)) * $(4096)) + (((((r432_group % $(8)) * $(4096)) + (r432_lane[:, None] * $(32)) + r432_col[None, :]) % $(32)) * $(128)) + (((((r432_group % $(8)) * $(4096)) + (r432_lane[:, None] * $(32)) + r432_col[None, :]) / $(32)) % $(128))), dtype=tl.float32)
      r432_value = tl.dot(r432_left, r432_right)
      r432_value = r432_value.to(tl.float32)
      tl.store($(candidate_vbmm_56_ptr) + ((r432_group * $(512)) + (r432_row[:, None] * $(32)) + r432_col[None, :]), r432_value)
    }
    r433_z = tl.arange($(4096))
    r433_value = (tl.load($(candidate_vbmm_56_ptr) + ((((r433_z / $(512)) % $(8)) * $(512)) + (((r433_z / $(32)) % $(16)) * $(32)) + (r433_z % $(32))), dtype=tl.float32) - tl.load($(candidate_vsum_7_ptr) + ((((r433_z / $(512)) % $(8)) * $(16)) + ((r433_z / $(32)) % $(16))), dtype=tl.float32))
    r433_value = r433_value.to(tl.float32)
    r433_value = tl.broadcast_to(r433_value, [$(4096)])
    tl.store($(candidate_vsub_38_ptr) + r433_z, r433_value)
    r434_z = tl.arange($(4096))
    r434_value = (tl.load($(candidate_vexp_5_ptr) + ((((r434_z / $(512)) % $(8)) * $(512)) + (((r434_z / $(32)) % $(16)) * $(32)) + (r434_z % $(32))), dtype=tl.float32) * tl.load($(candidate_vsub_38_ptr) + ((((r434_z / $(512)) % $(8)) * $(512)) + (((r434_z / $(32)) % $(16)) * $(32)) + (r434_z % $(32))), dtype=tl.float32))
    r434_value = r434_value.to(tl.float32)
    r434_value = tl.broadcast_to(r434_value, [$(4096)])
    tl.store($(candidate_vmul_62_ptr) + r434_z, r434_value)
    r435_row = tl.arange($(32))
    r435_col = tl.arange($(192))
    r435_lane = tl.arange($(16))
    for r435_group in range($(8)) {
      r435_left = tl.load($(candidate_vmul_62_ptr) + (((((((r435_group % $(8)) * $(512)) + (r435_row[:, None] * $(16)) + r435_lane[None, :]) / $(512)) % $(8)) * $(512)) + (((((r435_group % $(8)) * $(512)) + (r435_row[:, None] * $(16)) + r435_lane[None, :]) % $(16)) * $(32)) + (((((r435_group % $(8)) * $(512)) + (r435_row[:, None] * $(16)) + r435_lane[None, :]) / $(16)) % $(32))), dtype=tl.float32)
      r435_right = tl.load($(candidate_vto_copy_130_ptr) + (((((((r435_group % $(8)) * $(3072)) + (r435_lane[:, None] * $(192)) + r435_col[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r435_group % $(8)) * $(3072)) + (r435_lane[:, None] * $(192)) + r435_col[None, :]) / $(192)) % $(16)) * $(192)) + ((((r435_group % $(8)) * $(3072)) + (r435_lane[:, None] * $(192)) + r435_col[None, :]) % $(192))), dtype=tl.float32)
      r435_value = tl.dot(r435_left, r435_right)
      r435_value = r435_value.to(tl.float32)
      tl.store($(candidate_vbmm_58_ptr) + ((r435_group * $(6144)) + (r435_row[:, None] * $(192)) + r435_col[None, :]), r435_value)
    }
    r436_z = tl.arange($(49152))
    r436_value = (tl.load($(candidate_vbmm_58_ptr) + ((((r436_z / $(6144)) % $(8)) * $(6144)) + (((r436_z / $(192)) % $(32)) * $(192)) + (r436_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r436_value = r436_value.to(tl.float32)
    r436_value = tl.broadcast_to(r436_value, [$(49152)])
    tl.store($(candidate_vmul_64_ptr) + r436_z, r436_value)
    r437_z = tl.arange($(49152))
    r437_value = tl.load($(candidate_vmul_64_ptr) + ((((r437_z / $(192)) % $(8)) * $(6144)) + (((r437_z / $(1536)) % $(32)) * $(192)) + (r437_z % $(192))), dtype=tl.float32)
    r437_value = r437_value.to(tl.bfloat16)
    r437_value = tl.broadcast_to(r437_value, [$(49152)])
    tl.store($(candidate_vto_copy_136_ptr) + r437_z, r437_value)
    r438_row = tl.arange($(32))
    r438_col = tl.arange($(128))
    r438_lane = tl.arange($(16))
    for r438_group in range($(8)) {
      r438_left = tl.load($(candidate_vexp_5_ptr) + (((((((r438_group % $(8)) * $(512)) + (r438_row[:, None] * $(16)) + r438_lane[None, :]) / $(512)) % $(8)) * $(512)) + (((((r438_group % $(8)) * $(512)) + (r438_row[:, None] * $(16)) + r438_lane[None, :]) % $(16)) * $(32)) + (((((r438_group % $(8)) * $(512)) + (r438_row[:, None] * $(16)) + r438_lane[None, :]) / $(16)) % $(32))), dtype=tl.float32)
      r438_right = tl.load($(candidate_vto_copy_133_ptr) + (((((((r438_group % $(8)) * $(2048)) + (r438_lane[:, None] * $(128)) + r438_col[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r438_group % $(8)) * $(2048)) + (r438_lane[:, None] * $(128)) + r438_col[None, :]) / $(128)) % $(16)) * $(128)) + ((((r438_group % $(8)) * $(2048)) + (r438_lane[:, None] * $(128)) + r438_col[None, :]) % $(128))), dtype=tl.float32)
      r438_value = tl.dot(r438_left, r438_right)
      r438_value = r438_value.to(tl.float32)
      tl.store($(candidate_vbmm_59_ptr) + ((r438_group * $(4096)) + (r438_row[:, None] * $(128)) + r438_col[None, :]), r438_value)
    }
    r439_z = tl.arange($(32768))
    r439_value = tl.load($(candidate_vbmm_59_ptr) + ((((r439_z / $(128)) % $(8)) * $(4096)) + (((r439_z / $(1024)) % $(32)) * $(128)) + (r439_z % $(128))), dtype=tl.float32)
    r439_value = r439_value.to(tl.bfloat16)
    r439_value = tl.broadcast_to(r439_value, [$(32768)])
    tl.store($(candidate_vto_copy_137_ptr) + r439_z, r439_value)
    r440_z = tl.arange($(98304))
    r440_value = tl.where(((r440_z / $(1536)) % $(64)) < $(32), tl.load($(candidate_vto_copy_136_ptr) + ((((((r440_z / $(1536)) % $(64)) - $(0)) % $(32)) * $(1536)) + (((r440_z / $(192)) % $(8)) * $(192)) + (r440_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r440_value = r440_value.to(tl.bfloat16)
    r440_value = tl.broadcast_to(r440_value, [$(98304)])
    tl.store($(candidate_vcat_33_ptr) + r440_z, r440_value)
    r441_z = tl.arange($(65536))
    r441_value = tl.where(((r441_z / $(1024)) % $(64)) < $(32), tl.load($(candidate_vto_copy_137_ptr) + ((((((r441_z / $(1024)) % $(64)) - $(0)) % $(32)) * $(1024)) + (((r441_z / $(128)) % $(8)) * $(128)) + (r441_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r441_value = r441_value.to(tl.bfloat16)
    r441_value = tl.broadcast_to(r441_value, [$(65536)])
    tl.store($(candidate_vcat_34_ptr) + r441_z, r441_value)
    r442_z = tl.arange($(196608))
    r442_value = tl.where(((r442_z / $(1536)) % $(128)) < $(48), tl.load($(candidate_vcat_29_ptr) + ((((((r442_z / $(1536)) % $(128)) - $(0)) % $(48)) * $(1536)) + (((r442_z / $(192)) % $(8)) * $(192)) + (r442_z % $(192))), dtype=tl.bfloat16), tl.where(((r442_z / $(1536)) % $(128)) < $(64), tl.load($(candidate_vcat_31_ptr) + ((((((r442_z / $(1536)) % $(128)) - $(48)) % $(16)) * $(1536)) + (((r442_z / $(192)) % $(8)) * $(192)) + (r442_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vcat_33_ptr) + ((((((r442_z / $(1536)) % $(128)) - $(64)) % $(64)) * $(1536)) + (((r442_z / $(192)) % $(8)) * $(192)) + (r442_z % $(192))), dtype=tl.bfloat16)))
    r442_value = r442_value.to(tl.bfloat16)
    r442_value = tl.broadcast_to(r442_value, [$(196608)])
    tl.store($(candidate_vcat_36_ptr) + r442_z, r442_value)
    r443_z = tl.arange($(131072))
    r443_value = tl.where(((r443_z / $(1024)) % $(128)) < $(48), tl.load($(candidate_vcat_30_ptr) + ((((((r443_z / $(1024)) % $(128)) - $(0)) % $(48)) * $(1024)) + (((r443_z / $(128)) % $(8)) * $(128)) + (r443_z % $(128))), dtype=tl.bfloat16), tl.where(((r443_z / $(1024)) % $(128)) < $(64), tl.load($(candidate_vcat_32_ptr) + ((((((r443_z / $(1024)) % $(128)) - $(48)) % $(16)) * $(1024)) + (((r443_z / $(128)) % $(8)) * $(128)) + (r443_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_34_ptr) + ((((((r443_z / $(1024)) % $(128)) - $(64)) % $(64)) * $(1024)) + (((r443_z / $(128)) % $(8)) * $(128)) + (r443_z % $(128))), dtype=tl.bfloat16)))
    r443_value = r443_value.to(tl.bfloat16)
    r443_value = tl.broadcast_to(r443_value, [$(131072)])
    tl.store($(candidate_vcat_37_ptr) + r443_z, r443_value)
    r444_row = tl.arange($(1024))
    r444_col = tl.arange($(512))
    r444_lane = tl.arange($(128))
    for r444_group in range($(1)) {
      r444_left = tl.load($(candidate_vcat_36_ptr) + (((((r444_row[:, None] * $(128)) + r444_lane[None, :]) % $(128)) * $(1536)) + (((((r444_row[:, None] * $(128)) + r444_lane[None, :]) / $(16384)) % $(8)) * $(192)) + ((((r444_row[:, None] * $(128)) + r444_lane[None, :]) / $(128)) % $(128))), dtype=tl.bfloat16)
      r444_right = tl.load($(candidate_vindex_2_ptr) + ((((((r444_lane[:, None] * $(512)) + r444_col[None, :]) / $(512)) % $(128)) * $(576)) + (((r444_lane[:, None] * $(512)) + r444_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r444_value = tl.dot(r444_left, r444_right)
      r444_value = r444_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_62_ptr) + ((r444_group * $(524288)) + (r444_row[:, None] * $(512)) + r444_col[None, :]), r444_value)
    }
    r445_row = tl.arange($(1024))
    r445_col = tl.arange($(512))
    r445_lane = tl.arange($(128))
    for r445_group in range($(1)) {
      r445_left = tl.load($(candidate_vcat_37_ptr) + (((((r445_row[:, None] * $(128)) + r445_lane[None, :]) % $(128)) * $(1024)) + (((((r445_row[:, None] * $(128)) + r445_lane[None, :]) / $(16384)) % $(8)) * $(128)) + ((((r445_row[:, None] * $(128)) + r445_lane[None, :]) / $(128)) % $(128))), dtype=tl.bfloat16)
      r445_right = tl.load($(candidate_vindex_2_ptr) + ((((((r445_lane[:, None] * $(512)) + r445_col[None, :]) / $(512)) % $(128)) * $(576)) + (((r445_lane[:, None] * $(512)) + r445_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r445_value = tl.dot(r445_left, r445_right)
      r445_value = r445_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_63_ptr) + ((r445_group * $(524288)) + (r445_row[:, None] * $(512)) + r445_col[None, :]), r445_value)
    }
    r446_z = tl.arange($(1048576))
    r446_value = tl.where(((r446_z / $(512)) % $(256)) < $(128), tl.load($(candidate_vbmm_62_ptr) + ((((((((r446_z / $(131072)) % $(8)) * $(65536)) + ((((r446_z / $(512)) % $(256)) - $(0)) * $(512)) + (r446_z % $(512))) / $(65536)) % $(8)) * $(65536)) + ((((((r446_z / $(131072)) % $(8)) * $(128)) + (((r446_z / $(512)) % $(256)) - $(0))) % $(128)) * $(512)) + (r446_z % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vbmm_63_ptr) + ((((((((r446_z / $(131072)) % $(8)) * $(65536)) + ((((r446_z / $(512)) % $(256)) - $(128)) * $(512)) + (r446_z % $(512))) / $(65536)) % $(8)) * $(65536)) + ((((((r446_z / $(131072)) % $(8)) * $(128)) + (((r446_z / $(512)) % $(256)) - $(128))) % $(128)) * $(512)) + (r446_z % $(512))), dtype=tl.bfloat16))
    r446_value = r446_value.to(tl.bfloat16)
    r446_value = tl.broadcast_to(r446_value, [$(1048576)])
    tl.store($(candidate_vcat_38_ptr) + r446_z, r446_value)
    r447_z = tl.arange($(1048576))
    r447_value = (tl.load($(candidate_vcat_26_ptr) + ((((((((r447_z / $(512)) % $(2048)) * $(512)) + (r447_z % $(512))) / $(131072)) % $(8)) * $(131072)) + ((((r447_z / $(512)) % $(2048)) % $(256)) * $(512)) + (r447_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vcat_38_ptr) + ((((((((r447_z / $(512)) % $(2048)) * $(512)) + (r447_z % $(512))) / $(131072)) % $(8)) * $(131072)) + ((((r447_z / $(512)) % $(2048)) % $(256)) * $(512)) + (r447_z % $(512))), dtype=tl.bfloat16))
    r447_value = r447_value.to(tl.bfloat16)
    r447_value = tl.broadcast_to(r447_value, [$(1048576)])
    tl.store($(candidate_vadd_40_ptr) + r447_z, r447_value)
    r448_row = tl.arange($(128))
    r448_col = tl.arange($(1024))
    r448_lane = tl.arange($(512))
    for r448_group in range($(1)) {
      r448_left = tl.load($(candidate_vindex_2_ptr) + ((((((r448_row[:, None] * $(512)) + r448_lane[None, :]) / $(512)) % $(128)) * $(576)) + (((r448_row[:, None] * $(512)) + r448_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r448_right = tl.load($(candidate_vto_copy_20_ptr) + ((((((r448_lane[:, None] * $(1024)) + r448_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r448_lane[:, None] * $(1024)) + r448_col[None, :]) % $(128)) * $(512)) + ((((r448_lane[:, None] * $(1024)) + r448_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r448_value = tl.dot(r448_left, r448_right)
      r448_value = r448_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_64_ptr) + ((r448_group * $(131072)) + (r448_row[:, None] * $(1024)) + r448_col[None, :]), r448_value)
    }
    r449_row = tl.arange($(128))
    r449_col = tl.arange($(1024))
    r449_lane = tl.arange($(512))
    for r449_group in range($(1)) {
      r449_left = tl.load($(candidate_vindex_2_ptr) + ((((((r449_row[:, None] * $(512)) + r449_lane[None, :]) / $(512)) % $(128)) * $(576)) + (((r449_row[:, None] * $(512)) + r449_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r449_right = tl.load($(candidate_vto_copy_20_ptr) + ((((((r449_lane[:, None] * $(1024)) + r449_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r449_lane[:, None] * $(1024)) + r449_col[None, :]) % $(128)) * $(512)) + ((((r449_lane[:, None] * $(1024)) + r449_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r449_value = tl.dot(r449_left, r449_right)
      r449_value = r449_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_65_ptr) + ((r449_group * $(131072)) + (r449_row[:, None] * $(1024)) + r449_col[None, :]), r449_value)
    }
    r450_z = tl.arange($(49152))
    r450_value = tl.where((r450_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r450_z / $(1536)) % $(32)) * $(1024)) + (((r450_z / $(192)) % $(8)) * $(128)) + ((r450_z % $(192)) - $(0)) + $(32768)) / $(1024)) % $(128)) * $(1536)) + (((((((r450_z / $(1536)) % $(32)) * $(1024)) + (((r450_z / $(192)) % $(8)) * $(128)) + ((r450_z % $(192)) - $(0)) + $(32768)) / $(128)) % $(8)) * $(192)) + (((((r450_z / $(1536)) % $(32)) * $(1024)) + (((r450_z / $(192)) % $(8)) * $(128)) + ((r450_z % $(192)) - $(0)) + $(32768)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vadd_6_ptr) + ((((((((r450_z / $(1536)) % $(32)) * $(512)) + (((r450_z / $(192)) % $(8)) * $(64)) + ((r450_z % $(192)) - $(128)) + $(16384)) / $(512)) % $(128)) * $(512)) + (((((((r450_z / $(1536)) % $(32)) * $(512)) + (((r450_z / $(192)) % $(8)) * $(64)) + ((r450_z % $(192)) - $(128)) + $(16384)) / $(64)) % $(8)) * $(64)) + (((((r450_z / $(1536)) % $(32)) * $(512)) + (((r450_z / $(192)) % $(8)) * $(64)) + ((r450_z % $(192)) - $(128)) + $(16384)) % $(64))), dtype=tl.bfloat16))
    r450_value = r450_value.to(tl.bfloat16)
    r450_value = tl.broadcast_to(r450_value, [$(49152)])
    tl.store($(candidate_vcat_39_ptr) + r450_z, r450_value)
    r451_z = tl.arange($(196608))
    r451_value = tl.where((r451_z % $(192)) < $(128), tl.load($(candidate_vbmm_64_ptr) + ((((((((r451_z / $(1536)) % $(128)) * $(1024)) + (((r451_z / $(192)) % $(8)) * $(128)) + ((r451_z % $(192)) - $(0))) / $(1024)) % $(128)) * $(1024)) + (((((((r451_z / $(1536)) % $(128)) * $(1024)) + (((r451_z / $(192)) % $(8)) * $(128)) + ((r451_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r451_z / $(1536)) % $(128)) * $(1024)) + (((r451_z / $(192)) % $(8)) * $(128)) + ((r451_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vindex_2_ptr) + ((((r451_z / $(1536)) % $(128)) * $(576)) + ((((r451_z % $(192)) - $(128)) + $(512)) % $(576))), dtype=tl.bfloat16))
    r451_value = r451_value.to(tl.bfloat16)
    r451_value = tl.broadcast_to(r451_value, [$(196608)])
    tl.store($(candidate_vcat_40_ptr) + r451_z, r451_value)
    r452_z = tl.arange($(18432))
    r452_value = tl.load($(candidate_vcat_39_ptr) + ((((r452_z / $(192)) % $(12)) * $(1536)) + (((r452_z / $(2304)) % $(8)) * $(192)) + (r452_z % $(192))), dtype=tl.bfloat16)
    r452_value = r452_value.to(tl.float32)
    r452_value = tl.broadcast_to(r452_value, [$(18432)])
    tl.store($(candidate_vto_copy_138_ptr) + r452_z, r452_value)
    r453_z = tl.arange($(73728))
    r453_value = tl.load($(candidate_vcat_40_ptr) + ((((r453_z / $(192)) % $(48)) * $(1536)) + (((r453_z / $(9216)) % $(8)) * $(192)) + (r453_z % $(192))), dtype=tl.bfloat16)
    r453_value = r453_value.to(tl.float32)
    r453_value = tl.broadcast_to(r453_value, [$(73728)])
    tl.store($(candidate_vto_copy_139_ptr) + r453_z, r453_value)
    r454_z = tl.arange($(49152))
    r454_value = tl.load($(candidate_vbmm_65_ptr) + ((((r454_z / $(128)) % $(48)) * $(1024)) + (((r454_z / $(6144)) % $(8)) * $(128)) + (r454_z % $(128))), dtype=tl.bfloat16)
    r454_value = r454_value.to(tl.float32)
    r454_value = tl.broadcast_to(r454_value, [$(49152)])
    tl.store($(candidate_vto_copy_140_ptr) + r454_z, r454_value)
    r455_z = tl.arange($(12288))
    r455_value = tl.load($(candidate_vmm_28_ptr) + ((((r455_z / $(128)) % $(12)) * $(1024)) + (((r455_z / $(1536)) % $(8)) * $(128)) + (r455_z % $(128)) + $(32768)), dtype=tl.bfloat16)
    r455_value = r455_value.to(tl.float32)
    r455_value = tl.broadcast_to(r455_value, [$(12288)])
    tl.store($(candidate_vto_copy_141_ptr) + r455_z, r455_value)
    r456_z = tl.arange($(12288))
    r456_value = tl.load($(candidate_vbmm_11_ptr) + ((((r456_z / $(1536)) % $(8)) * $(4096)) + (((r456_z / $(128)) % $(12)) * $(128)) + (r456_z % $(128))), dtype=tl.bfloat16)
    r456_value = r456_value.to(tl.float32)
    r456_value = tl.broadcast_to(r456_value, [$(12288)])
    tl.store($(candidate_vto_copy_142_ptr) + r456_z, r456_value)
    r457_row = tl.arange($(12))
    r457_col = tl.arange($(48))
    r457_lane = tl.arange($(192))
    for r457_group in range($(8)) {
      r457_left = tl.load($(candidate_vto_copy_138_ptr) + (((((((r457_group % $(8)) * $(2304)) + (r457_row[:, None] * $(192)) + r457_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r457_group % $(8)) * $(2304)) + (r457_row[:, None] * $(192)) + r457_lane[None, :]) / $(192)) % $(12)) * $(192)) + ((((r457_group % $(8)) * $(2304)) + (r457_row[:, None] * $(192)) + r457_lane[None, :]) % $(192))), dtype=tl.float32)
      r457_right = tl.load($(candidate_vto_copy_139_ptr) + (((((((r457_group % $(8)) * $(9216)) + (r457_lane[:, None] * $(48)) + r457_col[None, :]) / $(9216)) % $(8)) * $(9216)) + (((((r457_group % $(8)) * $(9216)) + (r457_lane[:, None] * $(48)) + r457_col[None, :]) % $(48)) * $(192)) + (((((r457_group % $(8)) * $(9216)) + (r457_lane[:, None] * $(48)) + r457_col[None, :]) / $(48)) % $(192))), dtype=tl.float32)
      r457_value = tl.dot(r457_left, r457_right)
      r457_value = r457_value.to(tl.float32)
      tl.store($(candidate_vbmm_66_ptr) + ((r457_group * $(576)) + (r457_row[:, None] * $(48)) + r457_col[None, :]), r457_value)
    }
    r458_z = tl.arange($(4608))
    r458_value = (tl.load($(candidate_vbmm_66_ptr) + ((((r458_z / $(576)) % $(8)) * $(576)) + (((r458_z / $(48)) % $(12)) * $(48)) + (r458_z % $(48))), dtype=tl.float32) * 0.07216878364870322)
    r458_value = r458_value.to(tl.float32)
    r458_value = tl.broadcast_to(r458_value, [$(4608)])
    tl.store($(candidate_vmul_65_ptr) + r458_z, r458_value)
    r459_z = tl.arange($(12))
    r459_value = (tl.cast(r459_z, tl.int64) + tl.full([], $(48), dtype=tl.int64))
    r459_value = tl.broadcast_to(r459_value, [$(12)])
    tl.store($(candidate_vadd_41_ptr) + r459_z, r459_value)
    r460_z = tl.arange($(12))
    r460_value = (tl.load($(candidate_vadd_41_ptr) + r460_z, dtype=tl.int64) - tl.full([], $(12), dtype=tl.int64))
    r460_value = tl.broadcast_to(r460_value, [$(12)])
    tl.store($(candidate_vsub_39_ptr) + r460_z, r460_value)
    r461_z = tl.arange($(576))
    r461_value = (tl.cast((r461_z % $(48)), tl.int64) > tl.load($(candidate_vsub_39_ptr) + ((r461_z / $(48)) % $(12)), dtype=tl.int64))
    r461_value = tl.broadcast_to(r461_value, [$(576)])
    tl.store($(candidate_vgt_18_ptr) + r461_z, r461_value)
    r462_z = tl.arange($(4608))
    r462_value = tl.where(tl.load($(candidate_vgt_18_ptr) + ((((r462_z / $(48)) % $(12)) * $(48)) + (r462_z % $(48))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_65_ptr) + ((((r462_z / $(576)) % $(8)) * $(576)) + (((r462_z / $(48)) % $(12)) * $(48)) + (r462_z % $(48))), dtype=tl.float32))
    r462_value = r462_value.to(tl.float32)
    r462_value = tl.broadcast_to(r462_value, [$(4608)])
    tl.store($(candidate_vmasked_fill_18_ptr) + r462_z, r462_value)
    r463_z = tl.arange($(4608))
    r463_value = (tl.load($(candidate_vmasked_fill_18_ptr) + ((((r463_z / $(576)) % $(8)) * $(576)) + (((r463_z / $(48)) % $(12)) * $(48)) + (r463_z % $(48))), dtype=tl.float32) - tl.load($(candidate_vcat_9_ptr) + ((((r463_z / $(48)) % $(12)) * $(8)) + ((r463_z / $(576)) % $(8))), dtype=tl.float32))
    r463_value = r463_value.to(tl.float32)
    r463_value = tl.broadcast_to(r463_value, [$(4608)])
    tl.store($(candidate_vsub_40_ptr) + r463_z, r463_value)
    r464_z = tl.arange($(4608))
    r464_value = tl.exp(tl.load($(candidate_vsub_40_ptr) + ((((r464_z / $(576)) % $(8)) * $(576)) + (((r464_z / $(48)) % $(12)) * $(48)) + (r464_z % $(48))), dtype=tl.float32))
    r464_value = r464_value.to(tl.float32)
    r464_value = tl.broadcast_to(r464_value, [$(4608)])
    tl.store($(candidate_vexp_6_ptr) + r464_z, r464_value)
    r465_z = tl.arange($(12288))
    r465_value = (tl.load($(candidate_vto_copy_141_ptr) + ((((r465_z / $(1536)) % $(8)) * $(1536)) + (((r465_z / $(128)) % $(12)) * $(128)) + (r465_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_142_ptr) + ((((r465_z / $(1536)) % $(8)) * $(1536)) + (((r465_z / $(128)) % $(12)) * $(128)) + (r465_z % $(128))), dtype=tl.float32))
    r465_value = r465_value.to(tl.float32)
    r465_value = tl.broadcast_to(r465_value, [$(12288)])
    tl.store($(candidate_vmul_66_ptr) + r465_z, r465_value)
    r466_z = tl.arange($(96))
    r466_lane = tl.arange($(128))
    r466_value = tl.sum(tl.load($(candidate_vmul_66_ptr) + ((((r466_z[:, None] / $(12)) % $(8)) * $(1536)) + ((r466_z[:, None] % $(12)) * $(128)) + r466_lane[None, :]), dtype=tl.float32), axis=1)
    r466_value = r466_value.to(tl.float32)
    r466_value = tl.broadcast_to(r466_value, [$(96)])
    tl.store($(candidate_vsum_9_ptr) + r466_z, r466_value)
    r467_row = tl.arange($(12))
    r467_col = tl.arange($(48))
    r467_lane = tl.arange($(128))
    for r467_group in range($(8)) {
      r467_left = tl.load($(candidate_vto_copy_141_ptr) + (((((((r467_group % $(8)) * $(1536)) + (r467_row[:, None] * $(128)) + r467_lane[None, :]) / $(1536)) % $(8)) * $(1536)) + ((((((r467_group % $(8)) * $(1536)) + (r467_row[:, None] * $(128)) + r467_lane[None, :]) / $(128)) % $(12)) * $(128)) + ((((r467_group % $(8)) * $(1536)) + (r467_row[:, None] * $(128)) + r467_lane[None, :]) % $(128))), dtype=tl.float32)
      r467_right = tl.load($(candidate_vto_copy_140_ptr) + (((((((r467_group % $(8)) * $(6144)) + (r467_lane[:, None] * $(48)) + r467_col[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r467_group % $(8)) * $(6144)) + (r467_lane[:, None] * $(48)) + r467_col[None, :]) % $(48)) * $(128)) + (((((r467_group % $(8)) * $(6144)) + (r467_lane[:, None] * $(48)) + r467_col[None, :]) / $(48)) % $(128))), dtype=tl.float32)
      r467_value = tl.dot(r467_left, r467_right)
      r467_value = r467_value.to(tl.float32)
      tl.store($(candidate_vbmm_67_ptr) + ((r467_group * $(576)) + (r467_row[:, None] * $(48)) + r467_col[None, :]), r467_value)
    }
    r468_z = tl.arange($(4608))
    r468_value = (tl.load($(candidate_vbmm_67_ptr) + ((((r468_z / $(576)) % $(8)) * $(576)) + (((r468_z / $(48)) % $(12)) * $(48)) + (r468_z % $(48))), dtype=tl.float32) - tl.load($(candidate_vsum_9_ptr) + ((((r468_z / $(576)) % $(8)) * $(12)) + ((r468_z / $(48)) % $(12))), dtype=tl.float32))
    r468_value = r468_value.to(tl.float32)
    r468_value = tl.broadcast_to(r468_value, [$(4608)])
    tl.store($(candidate_vsub_41_ptr) + r468_z, r468_value)
    r469_z = tl.arange($(4608))
    r469_value = (tl.load($(candidate_vexp_6_ptr) + ((((r469_z / $(576)) % $(8)) * $(576)) + (((r469_z / $(48)) % $(12)) * $(48)) + (r469_z % $(48))), dtype=tl.float32) * tl.load($(candidate_vsub_41_ptr) + ((((r469_z / $(576)) % $(8)) * $(576)) + (((r469_z / $(48)) % $(12)) * $(48)) + (r469_z % $(48))), dtype=tl.float32))
    r469_value = r469_value.to(tl.float32)
    r469_value = tl.broadcast_to(r469_value, [$(4608)])
    tl.store($(candidate_vmul_67_ptr) + r469_z, r469_value)
    r470_row = tl.arange($(48))
    r470_col = tl.arange($(192))
    r470_lane = tl.arange($(12))
    for r470_group in range($(8)) {
      r470_left = tl.load($(candidate_vmul_67_ptr) + (((((((r470_group % $(8)) * $(576)) + (r470_row[:, None] * $(12)) + r470_lane[None, :]) / $(576)) % $(8)) * $(576)) + (((((r470_group % $(8)) * $(576)) + (r470_row[:, None] * $(12)) + r470_lane[None, :]) % $(12)) * $(48)) + (((((r470_group % $(8)) * $(576)) + (r470_row[:, None] * $(12)) + r470_lane[None, :]) / $(12)) % $(48))), dtype=tl.float32)
      r470_right = tl.load($(candidate_vto_copy_138_ptr) + (((((((r470_group % $(8)) * $(2304)) + (r470_lane[:, None] * $(192)) + r470_col[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r470_group % $(8)) * $(2304)) + (r470_lane[:, None] * $(192)) + r470_col[None, :]) / $(192)) % $(12)) * $(192)) + ((((r470_group % $(8)) * $(2304)) + (r470_lane[:, None] * $(192)) + r470_col[None, :]) % $(192))), dtype=tl.float32)
      r470_value = tl.dot(r470_left, r470_right)
      r470_value = r470_value.to(tl.float32)
      tl.store($(candidate_vbmm_69_ptr) + ((r470_group * $(9216)) + (r470_row[:, None] * $(192)) + r470_col[None, :]), r470_value)
    }
    r471_z = tl.arange($(73728))
    r471_value = (tl.load($(candidate_vbmm_69_ptr) + ((((r471_z / $(9216)) % $(8)) * $(9216)) + (((r471_z / $(192)) % $(48)) * $(192)) + (r471_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r471_value = r471_value.to(tl.float32)
    r471_value = tl.broadcast_to(r471_value, [$(73728)])
    tl.store($(candidate_vmul_69_ptr) + r471_z, r471_value)
    r472_z = tl.arange($(73728))
    r472_value = tl.load($(candidate_vmul_69_ptr) + ((((r472_z / $(192)) % $(8)) * $(9216)) + (((r472_z / $(1536)) % $(48)) * $(192)) + (r472_z % $(192))), dtype=tl.float32)
    r472_value = r472_value.to(tl.bfloat16)
    r472_value = tl.broadcast_to(r472_value, [$(73728)])
    tl.store($(candidate_vto_copy_144_ptr) + r472_z, r472_value)
    r473_row = tl.arange($(48))
    r473_col = tl.arange($(128))
    r473_lane = tl.arange($(12))
    for r473_group in range($(8)) {
      r473_left = tl.load($(candidate_vexp_6_ptr) + (((((((r473_group % $(8)) * $(576)) + (r473_row[:, None] * $(12)) + r473_lane[None, :]) / $(576)) % $(8)) * $(576)) + (((((r473_group % $(8)) * $(576)) + (r473_row[:, None] * $(12)) + r473_lane[None, :]) % $(12)) * $(48)) + (((((r473_group % $(8)) * $(576)) + (r473_row[:, None] * $(12)) + r473_lane[None, :]) / $(12)) % $(48))), dtype=tl.float32)
      r473_right = tl.load($(candidate_vto_copy_141_ptr) + (((((((r473_group % $(8)) * $(1536)) + (r473_lane[:, None] * $(128)) + r473_col[None, :]) / $(1536)) % $(8)) * $(1536)) + ((((((r473_group % $(8)) * $(1536)) + (r473_lane[:, None] * $(128)) + r473_col[None, :]) / $(128)) % $(12)) * $(128)) + ((((r473_group % $(8)) * $(1536)) + (r473_lane[:, None] * $(128)) + r473_col[None, :]) % $(128))), dtype=tl.float32)
      r473_value = tl.dot(r473_left, r473_right)
      r473_value = r473_value.to(tl.float32)
      tl.store($(candidate_vbmm_70_ptr) + ((r473_group * $(6144)) + (r473_row[:, None] * $(128)) + r473_col[None, :]), r473_value)
    }
    r474_z = tl.arange($(49152))
    r474_value = tl.load($(candidate_vbmm_70_ptr) + ((((r474_z / $(128)) % $(8)) * $(6144)) + (((r474_z / $(1024)) % $(48)) * $(128)) + (r474_z % $(128))), dtype=tl.float32)
    r474_value = r474_value.to(tl.bfloat16)
    r474_value = tl.broadcast_to(r474_value, [$(49152)])
    tl.store($(candidate_vto_copy_145_ptr) + r474_z, r474_value)
    r475_z = tl.arange($(6144))
    r475_value = tl.load($(candidate_vcat_39_ptr) + ((((r475_z / $(192)) % $(4)) * $(1536)) + (((r475_z / $(768)) % $(8)) * $(192)) + (r475_z % $(192)) + $(18432)), dtype=tl.bfloat16)
    r475_value = r475_value.to(tl.float32)
    r475_value = tl.broadcast_to(r475_value, [$(6144)])
    tl.store($(candidate_vto_copy_146_ptr) + r475_z, r475_value)
    r476_z = tl.arange($(24576))
    r476_value = tl.load($(candidate_vcat_40_ptr) + ((((r476_z / $(192)) % $(16)) * $(1536)) + (((r476_z / $(3072)) % $(8)) * $(192)) + (r476_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r476_value = r476_value.to(tl.float32)
    r476_value = tl.broadcast_to(r476_value, [$(24576)])
    tl.store($(candidate_vto_copy_147_ptr) + r476_z, r476_value)
    r477_z = tl.arange($(16384))
    r477_value = tl.load($(candidate_vbmm_65_ptr) + ((((r477_z / $(128)) % $(16)) * $(1024)) + (((r477_z / $(2048)) % $(8)) * $(128)) + (r477_z % $(128)) + $(49152)), dtype=tl.bfloat16)
    r477_value = r477_value.to(tl.float32)
    r477_value = tl.broadcast_to(r477_value, [$(16384)])
    tl.store($(candidate_vto_copy_148_ptr) + r477_z, r477_value)
    r478_z = tl.arange($(4096))
    r478_value = tl.load($(candidate_vmm_28_ptr) + ((((r478_z / $(128)) % $(4)) * $(1024)) + (((r478_z / $(512)) % $(8)) * $(128)) + (r478_z % $(128)) + $(45056)), dtype=tl.bfloat16)
    r478_value = r478_value.to(tl.float32)
    r478_value = tl.broadcast_to(r478_value, [$(4096)])
    tl.store($(candidate_vto_copy_149_ptr) + r478_z, r478_value)
    r479_z = tl.arange($(4096))
    r479_value = tl.load($(candidate_vbmm_11_ptr) + ((((r479_z / $(512)) % $(8)) * $(4096)) + (((r479_z / $(128)) % $(4)) * $(128)) + (r479_z % $(128)) + $(1536)), dtype=tl.bfloat16)
    r479_value = r479_value.to(tl.float32)
    r479_value = tl.broadcast_to(r479_value, [$(4096)])
    tl.store($(candidate_vto_copy_150_ptr) + r479_z, r479_value)
    r480_row = tl.arange($(4))
    r480_col = tl.arange($(16))
    r480_lane = tl.arange($(192))
    for r480_group in range($(8)) {
      r480_left = tl.load($(candidate_vto_copy_146_ptr) + (((((((r480_group % $(8)) * $(768)) + (r480_row[:, None] * $(192)) + r480_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r480_group % $(8)) * $(768)) + (r480_row[:, None] * $(192)) + r480_lane[None, :]) / $(192)) % $(4)) * $(192)) + ((((r480_group % $(8)) * $(768)) + (r480_row[:, None] * $(192)) + r480_lane[None, :]) % $(192))), dtype=tl.float32)
      r480_right = tl.load($(candidate_vto_copy_147_ptr) + (((((((r480_group % $(8)) * $(3072)) + (r480_lane[:, None] * $(16)) + r480_col[None, :]) / $(3072)) % $(8)) * $(3072)) + (((((r480_group % $(8)) * $(3072)) + (r480_lane[:, None] * $(16)) + r480_col[None, :]) % $(16)) * $(192)) + (((((r480_group % $(8)) * $(3072)) + (r480_lane[:, None] * $(16)) + r480_col[None, :]) / $(16)) % $(192))), dtype=tl.float32)
      r480_value = tl.dot(r480_left, r480_right)
      r480_value = r480_value.to(tl.float32)
      tl.store($(candidate_vbmm_71_ptr) + ((r480_group * $(64)) + (r480_row[:, None] * $(16)) + r480_col[None, :]), r480_value)
    }
    r481_z = tl.arange($(512))
    r481_value = (tl.load($(candidate_vbmm_71_ptr) + ((((r481_z / $(64)) % $(8)) * $(64)) + (((r481_z / $(16)) % $(4)) * $(16)) + (r481_z % $(16))), dtype=tl.float32) * 0.07216878364870322)
    r481_value = r481_value.to(tl.float32)
    r481_value = tl.broadcast_to(r481_value, [$(512)])
    tl.store($(candidate_vmul_70_ptr) + r481_z, r481_value)
    r482_z = tl.arange($(4))
    r482_value = (tl.cast(r482_z, tl.int64) + tl.full([], $(16), dtype=tl.int64))
    r482_value = tl.broadcast_to(r482_value, [$(4)])
    tl.store($(candidate_vadd_42_ptr) + r482_z, r482_value)
    r483_z = tl.arange($(4))
    r483_value = (tl.load($(candidate_vadd_42_ptr) + r483_z, dtype=tl.int64) - tl.full([], $(4), dtype=tl.int64))
    r483_value = tl.broadcast_to(r483_value, [$(4)])
    tl.store($(candidate_vsub_42_ptr) + r483_z, r483_value)
    r484_z = tl.arange($(64))
    r484_value = (tl.cast((r484_z % $(16)), tl.int64) > tl.load($(candidate_vsub_42_ptr) + ((r484_z / $(16)) % $(4)), dtype=tl.int64))
    r484_value = tl.broadcast_to(r484_value, [$(64)])
    tl.store($(candidate_vgt_19_ptr) + r484_z, r484_value)
    r485_z = tl.arange($(512))
    r485_value = tl.where(tl.load($(candidate_vgt_19_ptr) + ((((r485_z / $(16)) % $(4)) * $(16)) + (r485_z % $(16))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_70_ptr) + ((((r485_z / $(64)) % $(8)) * $(64)) + (((r485_z / $(16)) % $(4)) * $(16)) + (r485_z % $(16))), dtype=tl.float32))
    r485_value = r485_value.to(tl.float32)
    r485_value = tl.broadcast_to(r485_value, [$(512)])
    tl.store($(candidate_vmasked_fill_19_ptr) + r485_z, r485_value)
    r486_z = tl.arange($(512))
    r486_value = (tl.load($(candidate_vmasked_fill_19_ptr) + ((((r486_z / $(64)) % $(8)) * $(64)) + (((r486_z / $(16)) % $(4)) * $(16)) + (r486_z % $(16))), dtype=tl.float32) - tl.load($(candidate_vcat_9_ptr) + ((((r486_z / $(16)) % $(4)) * $(8)) + ((r486_z / $(64)) % $(8)) + $(96)), dtype=tl.float32))
    r486_value = r486_value.to(tl.float32)
    r486_value = tl.broadcast_to(r486_value, [$(512)])
    tl.store($(candidate_vsub_43_ptr) + r486_z, r486_value)
    r487_z = tl.arange($(512))
    r487_value = tl.exp(tl.load($(candidate_vsub_43_ptr) + ((((r487_z / $(64)) % $(8)) * $(64)) + (((r487_z / $(16)) % $(4)) * $(16)) + (r487_z % $(16))), dtype=tl.float32))
    r487_value = r487_value.to(tl.float32)
    r487_value = tl.broadcast_to(r487_value, [$(512)])
    tl.store($(candidate_vexp_7_ptr) + r487_z, r487_value)
    r488_z = tl.arange($(4096))
    r488_value = (tl.load($(candidate_vto_copy_149_ptr) + ((((r488_z / $(512)) % $(8)) * $(512)) + (((r488_z / $(128)) % $(4)) * $(128)) + (r488_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_150_ptr) + ((((r488_z / $(512)) % $(8)) * $(512)) + (((r488_z / $(128)) % $(4)) * $(128)) + (r488_z % $(128))), dtype=tl.float32))
    r488_value = r488_value.to(tl.float32)
    r488_value = tl.broadcast_to(r488_value, [$(4096)])
    tl.store($(candidate_vmul_71_ptr) + r488_z, r488_value)
    r489_z = tl.arange($(32))
    r489_lane = tl.arange($(128))
    r489_value = tl.sum(tl.load($(candidate_vmul_71_ptr) + ((((r489_z[:, None] / $(4)) % $(8)) * $(512)) + ((r489_z[:, None] % $(4)) * $(128)) + r489_lane[None, :]), dtype=tl.float32), axis=1)
    r489_value = r489_value.to(tl.float32)
    r489_value = tl.broadcast_to(r489_value, [$(32)])
    tl.store($(candidate_vsum_10_ptr) + r489_z, r489_value)
    r490_row = tl.arange($(4))
    r490_col = tl.arange($(16))
    r490_lane = tl.arange($(128))
    for r490_group in range($(8)) {
      r490_left = tl.load($(candidate_vto_copy_149_ptr) + (((((((r490_group % $(8)) * $(512)) + (r490_row[:, None] * $(128)) + r490_lane[None, :]) / $(512)) % $(8)) * $(512)) + ((((((r490_group % $(8)) * $(512)) + (r490_row[:, None] * $(128)) + r490_lane[None, :]) / $(128)) % $(4)) * $(128)) + ((((r490_group % $(8)) * $(512)) + (r490_row[:, None] * $(128)) + r490_lane[None, :]) % $(128))), dtype=tl.float32)
      r490_right = tl.load($(candidate_vto_copy_148_ptr) + (((((((r490_group % $(8)) * $(2048)) + (r490_lane[:, None] * $(16)) + r490_col[None, :]) / $(2048)) % $(8)) * $(2048)) + (((((r490_group % $(8)) * $(2048)) + (r490_lane[:, None] * $(16)) + r490_col[None, :]) % $(16)) * $(128)) + (((((r490_group % $(8)) * $(2048)) + (r490_lane[:, None] * $(16)) + r490_col[None, :]) / $(16)) % $(128))), dtype=tl.float32)
      r490_value = tl.dot(r490_left, r490_right)
      r490_value = r490_value.to(tl.float32)
      tl.store($(candidate_vbmm_72_ptr) + ((r490_group * $(64)) + (r490_row[:, None] * $(16)) + r490_col[None, :]), r490_value)
    }
    r491_z = tl.arange($(512))
    r491_value = (tl.load($(candidate_vbmm_72_ptr) + ((((r491_z / $(64)) % $(8)) * $(64)) + (((r491_z / $(16)) % $(4)) * $(16)) + (r491_z % $(16))), dtype=tl.float32) - tl.load($(candidate_vsum_10_ptr) + ((((r491_z / $(64)) % $(8)) * $(4)) + ((r491_z / $(16)) % $(4))), dtype=tl.float32))
    r491_value = r491_value.to(tl.float32)
    r491_value = tl.broadcast_to(r491_value, [$(512)])
    tl.store($(candidate_vsub_44_ptr) + r491_z, r491_value)
    r492_z = tl.arange($(512))
    r492_value = (tl.load($(candidate_vexp_7_ptr) + ((((r492_z / $(64)) % $(8)) * $(64)) + (((r492_z / $(16)) % $(4)) * $(16)) + (r492_z % $(16))), dtype=tl.float32) * tl.load($(candidate_vsub_44_ptr) + ((((r492_z / $(64)) % $(8)) * $(64)) + (((r492_z / $(16)) % $(4)) * $(16)) + (r492_z % $(16))), dtype=tl.float32))
    r492_value = r492_value.to(tl.float32)
    r492_value = tl.broadcast_to(r492_value, [$(512)])
    tl.store($(candidate_vmul_72_ptr) + r492_z, r492_value)
    r493_row = tl.arange($(16))
    r493_col = tl.arange($(192))
    r493_lane = tl.arange($(4))
    for r493_group in range($(8)) {
      r493_left = tl.load($(candidate_vmul_72_ptr) + (((((((r493_group % $(8)) * $(64)) + (r493_row[:, None] * $(4)) + r493_lane[None, :]) / $(64)) % $(8)) * $(64)) + (((((r493_group % $(8)) * $(64)) + (r493_row[:, None] * $(4)) + r493_lane[None, :]) % $(4)) * $(16)) + (((((r493_group % $(8)) * $(64)) + (r493_row[:, None] * $(4)) + r493_lane[None, :]) / $(4)) % $(16))), dtype=tl.float32)
      r493_right = tl.load($(candidate_vto_copy_146_ptr) + (((((((r493_group % $(8)) * $(768)) + (r493_lane[:, None] * $(192)) + r493_col[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r493_group % $(8)) * $(768)) + (r493_lane[:, None] * $(192)) + r493_col[None, :]) / $(192)) % $(4)) * $(192)) + ((((r493_group % $(8)) * $(768)) + (r493_lane[:, None] * $(192)) + r493_col[None, :]) % $(192))), dtype=tl.float32)
      r493_value = tl.dot(r493_left, r493_right)
      r493_value = r493_value.to(tl.float32)
      tl.store($(candidate_vbmm_74_ptr) + ((r493_group * $(3072)) + (r493_row[:, None] * $(192)) + r493_col[None, :]), r493_value)
    }
    r494_z = tl.arange($(24576))
    r494_value = (tl.load($(candidate_vbmm_74_ptr) + ((((r494_z / $(3072)) % $(8)) * $(3072)) + (((r494_z / $(192)) % $(16)) * $(192)) + (r494_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r494_value = r494_value.to(tl.float32)
    r494_value = tl.broadcast_to(r494_value, [$(24576)])
    tl.store($(candidate_vmul_74_ptr) + r494_z, r494_value)
    r495_z = tl.arange($(24576))
    r495_value = tl.load($(candidate_vmul_74_ptr) + ((((r495_z / $(192)) % $(8)) * $(3072)) + (((r495_z / $(1536)) % $(16)) * $(192)) + (r495_z % $(192))), dtype=tl.float32)
    r495_value = r495_value.to(tl.bfloat16)
    r495_value = tl.broadcast_to(r495_value, [$(24576)])
    tl.store($(candidate_vto_copy_152_ptr) + r495_z, r495_value)
    r496_row = tl.arange($(16))
    r496_col = tl.arange($(128))
    r496_lane = tl.arange($(4))
    for r496_group in range($(8)) {
      r496_left = tl.load($(candidate_vexp_7_ptr) + (((((((r496_group % $(8)) * $(64)) + (r496_row[:, None] * $(4)) + r496_lane[None, :]) / $(64)) % $(8)) * $(64)) + (((((r496_group % $(8)) * $(64)) + (r496_row[:, None] * $(4)) + r496_lane[None, :]) % $(4)) * $(16)) + (((((r496_group % $(8)) * $(64)) + (r496_row[:, None] * $(4)) + r496_lane[None, :]) / $(4)) % $(16))), dtype=tl.float32)
      r496_right = tl.load($(candidate_vto_copy_149_ptr) + (((((((r496_group % $(8)) * $(512)) + (r496_lane[:, None] * $(128)) + r496_col[None, :]) / $(512)) % $(8)) * $(512)) + ((((((r496_group % $(8)) * $(512)) + (r496_lane[:, None] * $(128)) + r496_col[None, :]) / $(128)) % $(4)) * $(128)) + ((((r496_group % $(8)) * $(512)) + (r496_lane[:, None] * $(128)) + r496_col[None, :]) % $(128))), dtype=tl.float32)
      r496_value = tl.dot(r496_left, r496_right)
      r496_value = r496_value.to(tl.float32)
      tl.store($(candidate_vbmm_75_ptr) + ((r496_group * $(2048)) + (r496_row[:, None] * $(128)) + r496_col[None, :]), r496_value)
    }
    r497_z = tl.arange($(16384))
    r497_value = tl.load($(candidate_vbmm_75_ptr) + ((((r497_z / $(128)) % $(8)) * $(2048)) + (((r497_z / $(1024)) % $(16)) * $(128)) + (r497_z % $(128))), dtype=tl.float32)
    r497_value = r497_value.to(tl.bfloat16)
    r497_value = tl.broadcast_to(r497_value, [$(16384)])
    tl.store($(candidate_vto_copy_153_ptr) + r497_z, r497_value)
    r498_z = tl.arange($(24576))
    r498_value = tl.load($(candidate_vcat_39_ptr) + ((((r498_z / $(192)) % $(16)) * $(1536)) + (((r498_z / $(3072)) % $(8)) * $(192)) + (r498_z % $(192)) + $(24576)), dtype=tl.bfloat16)
    r498_value = r498_value.to(tl.float32)
    r498_value = tl.broadcast_to(r498_value, [$(24576)])
    tl.store($(candidate_vto_copy_154_ptr) + r498_z, r498_value)
    r499_z = tl.arange($(98304))
    r499_value = tl.load($(candidate_vcat_40_ptr) + ((((r499_z / $(192)) % $(64)) * $(1536)) + (((r499_z / $(12288)) % $(8)) * $(192)) + (r499_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r499_value = r499_value.to(tl.float32)
    r499_value = tl.broadcast_to(r499_value, [$(98304)])
    tl.store($(candidate_vto_copy_155_ptr) + r499_z, r499_value)
    r500_z = tl.arange($(65536))
    r500_value = tl.load($(candidate_vbmm_65_ptr) + ((((r500_z / $(128)) % $(64)) * $(1024)) + (((r500_z / $(8192)) % $(8)) * $(128)) + (r500_z % $(128)) + $(65536)), dtype=tl.bfloat16)
    r500_value = r500_value.to(tl.float32)
    r500_value = tl.broadcast_to(r500_value, [$(65536)])
    tl.store($(candidate_vto_copy_156_ptr) + r500_z, r500_value)
    r501_z = tl.arange($(16384))
    r501_value = tl.load($(candidate_vmm_28_ptr) + ((((r501_z / $(128)) % $(16)) * $(1024)) + (((r501_z / $(2048)) % $(8)) * $(128)) + (r501_z % $(128)) + $(49152)), dtype=tl.bfloat16)
    r501_value = r501_value.to(tl.float32)
    r501_value = tl.broadcast_to(r501_value, [$(16384)])
    tl.store($(candidate_vto_copy_157_ptr) + r501_z, r501_value)
    r502_z = tl.arange($(16384))
    r502_value = tl.load($(candidate_vbmm_11_ptr) + ((((r502_z / $(2048)) % $(8)) * $(4096)) + (((r502_z / $(128)) % $(16)) * $(128)) + (r502_z % $(128)) + $(2048)), dtype=tl.bfloat16)
    r502_value = r502_value.to(tl.float32)
    r502_value = tl.broadcast_to(r502_value, [$(16384)])
    tl.store($(candidate_vto_copy_158_ptr) + r502_z, r502_value)
    r503_row = tl.arange($(16))
    r503_col = tl.arange($(64))
    r503_lane = tl.arange($(192))
    for r503_group in range($(8)) {
      r503_left = tl.load($(candidate_vto_copy_154_ptr) + (((((((r503_group % $(8)) * $(3072)) + (r503_row[:, None] * $(192)) + r503_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r503_group % $(8)) * $(3072)) + (r503_row[:, None] * $(192)) + r503_lane[None, :]) / $(192)) % $(16)) * $(192)) + ((((r503_group % $(8)) * $(3072)) + (r503_row[:, None] * $(192)) + r503_lane[None, :]) % $(192))), dtype=tl.float32)
      r503_right = tl.load($(candidate_vto_copy_155_ptr) + (((((((r503_group % $(8)) * $(12288)) + (r503_lane[:, None] * $(64)) + r503_col[None, :]) / $(12288)) % $(8)) * $(12288)) + (((((r503_group % $(8)) * $(12288)) + (r503_lane[:, None] * $(64)) + r503_col[None, :]) % $(64)) * $(192)) + (((((r503_group % $(8)) * $(12288)) + (r503_lane[:, None] * $(64)) + r503_col[None, :]) / $(64)) % $(192))), dtype=tl.float32)
      r503_value = tl.dot(r503_left, r503_right)
      r503_value = r503_value.to(tl.float32)
      tl.store($(candidate_vbmm_76_ptr) + ((r503_group * $(1024)) + (r503_row[:, None] * $(64)) + r503_col[None, :]), r503_value)
    }
    r504_z = tl.arange($(8192))
    r504_value = (tl.load($(candidate_vbmm_76_ptr) + ((((r504_z / $(1024)) % $(8)) * $(1024)) + (((r504_z / $(64)) % $(16)) * $(64)) + (r504_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r504_value = r504_value.to(tl.float32)
    r504_value = tl.broadcast_to(r504_value, [$(8192)])
    tl.store($(candidate_vmul_75_ptr) + r504_z, r504_value)
    r505_z = tl.arange($(16))
    r505_value = (tl.cast(r505_z, tl.int64) + tl.full([], $(64), dtype=tl.int64))
    r505_value = tl.broadcast_to(r505_value, [$(16)])
    tl.store($(candidate_vadd_43_ptr) + r505_z, r505_value)
    r506_z = tl.arange($(16))
    r506_value = (tl.load($(candidate_vadd_43_ptr) + r506_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r506_value = tl.broadcast_to(r506_value, [$(16)])
    tl.store($(candidate_vsub_45_ptr) + r506_z, r506_value)
    r507_z = tl.arange($(1024))
    r507_value = (tl.cast((r507_z % $(64)), tl.int64) > tl.load($(candidate_vsub_45_ptr) + ((r507_z / $(64)) % $(16)), dtype=tl.int64))
    r507_value = tl.broadcast_to(r507_value, [$(1024)])
    tl.store($(candidate_vgt_20_ptr) + r507_z, r507_value)
    r508_z = tl.arange($(8192))
    r508_value = tl.where(tl.load($(candidate_vgt_20_ptr) + ((((r508_z / $(64)) % $(16)) * $(64)) + (r508_z % $(64))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_75_ptr) + ((((r508_z / $(1024)) % $(8)) * $(1024)) + (((r508_z / $(64)) % $(16)) * $(64)) + (r508_z % $(64))), dtype=tl.float32))
    r508_value = r508_value.to(tl.float32)
    r508_value = tl.broadcast_to(r508_value, [$(8192)])
    tl.store($(candidate_vmasked_fill_20_ptr) + r508_z, r508_value)
    r509_z = tl.arange($(8192))
    r509_value = (tl.load($(candidate_vmasked_fill_20_ptr) + ((((r509_z / $(1024)) % $(8)) * $(1024)) + (((r509_z / $(64)) % $(16)) * $(64)) + (r509_z % $(64))), dtype=tl.float32) - tl.load($(candidate_vcat_9_ptr) + ((((r509_z / $(64)) % $(16)) * $(8)) + ((r509_z / $(1024)) % $(8)) + $(128)), dtype=tl.float32))
    r509_value = r509_value.to(tl.float32)
    r509_value = tl.broadcast_to(r509_value, [$(8192)])
    tl.store($(candidate_vsub_46_ptr) + r509_z, r509_value)
    r510_z = tl.arange($(8192))
    r510_value = tl.exp(tl.load($(candidate_vsub_46_ptr) + ((((r510_z / $(1024)) % $(8)) * $(1024)) + (((r510_z / $(64)) % $(16)) * $(64)) + (r510_z % $(64))), dtype=tl.float32))
    r510_value = r510_value.to(tl.float32)
    r510_value = tl.broadcast_to(r510_value, [$(8192)])
    tl.store($(candidate_vexp_8_ptr) + r510_z, r510_value)
    r511_z = tl.arange($(16384))
    r511_value = (tl.load($(candidate_vto_copy_157_ptr) + ((((r511_z / $(2048)) % $(8)) * $(2048)) + (((r511_z / $(128)) % $(16)) * $(128)) + (r511_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_158_ptr) + ((((r511_z / $(2048)) % $(8)) * $(2048)) + (((r511_z / $(128)) % $(16)) * $(128)) + (r511_z % $(128))), dtype=tl.float32))
    r511_value = r511_value.to(tl.float32)
    r511_value = tl.broadcast_to(r511_value, [$(16384)])
    tl.store($(candidate_vmul_76_ptr) + r511_z, r511_value)
    r512_z = tl.arange($(128))
    r512_lane = tl.arange($(128))
    r512_value = tl.sum(tl.load($(candidate_vmul_76_ptr) + ((((r512_z[:, None] / $(16)) % $(8)) * $(2048)) + ((r512_z[:, None] % $(16)) * $(128)) + r512_lane[None, :]), dtype=tl.float32), axis=1)
    r512_value = r512_value.to(tl.float32)
    r512_value = tl.broadcast_to(r512_value, [$(128)])
    tl.store($(candidate_vsum_11_ptr) + r512_z, r512_value)
    r513_row = tl.arange($(16))
    r513_col = tl.arange($(64))
    r513_lane = tl.arange($(128))
    for r513_group in range($(8)) {
      r513_left = tl.load($(candidate_vto_copy_157_ptr) + (((((((r513_group % $(8)) * $(2048)) + (r513_row[:, None] * $(128)) + r513_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r513_group % $(8)) * $(2048)) + (r513_row[:, None] * $(128)) + r513_lane[None, :]) / $(128)) % $(16)) * $(128)) + ((((r513_group % $(8)) * $(2048)) + (r513_row[:, None] * $(128)) + r513_lane[None, :]) % $(128))), dtype=tl.float32)
      r513_right = tl.load($(candidate_vto_copy_156_ptr) + (((((((r513_group % $(8)) * $(8192)) + (r513_lane[:, None] * $(64)) + r513_col[None, :]) / $(8192)) % $(8)) * $(8192)) + (((((r513_group % $(8)) * $(8192)) + (r513_lane[:, None] * $(64)) + r513_col[None, :]) % $(64)) * $(128)) + (((((r513_group % $(8)) * $(8192)) + (r513_lane[:, None] * $(64)) + r513_col[None, :]) / $(64)) % $(128))), dtype=tl.float32)
      r513_value = tl.dot(r513_left, r513_right)
      r513_value = r513_value.to(tl.float32)
      tl.store($(candidate_vbmm_77_ptr) + ((r513_group * $(1024)) + (r513_row[:, None] * $(64)) + r513_col[None, :]), r513_value)
    }
    r514_z = tl.arange($(8192))
    r514_value = (tl.load($(candidate_vbmm_77_ptr) + ((((r514_z / $(1024)) % $(8)) * $(1024)) + (((r514_z / $(64)) % $(16)) * $(64)) + (r514_z % $(64))), dtype=tl.float32) - tl.load($(candidate_vsum_11_ptr) + ((((r514_z / $(1024)) % $(8)) * $(16)) + ((r514_z / $(64)) % $(16))), dtype=tl.float32))
    r514_value = r514_value.to(tl.float32)
    r514_value = tl.broadcast_to(r514_value, [$(8192)])
    tl.store($(candidate_vsub_47_ptr) + r514_z, r514_value)
    r515_z = tl.arange($(8192))
    r515_value = (tl.load($(candidate_vexp_8_ptr) + ((((r515_z / $(1024)) % $(8)) * $(1024)) + (((r515_z / $(64)) % $(16)) * $(64)) + (r515_z % $(64))), dtype=tl.float32) * tl.load($(candidate_vsub_47_ptr) + ((((r515_z / $(1024)) % $(8)) * $(1024)) + (((r515_z / $(64)) % $(16)) * $(64)) + (r515_z % $(64))), dtype=tl.float32))
    r515_value = r515_value.to(tl.float32)
    r515_value = tl.broadcast_to(r515_value, [$(8192)])
    tl.store($(candidate_vmul_77_ptr) + r515_z, r515_value)
    r516_row = tl.arange($(64))
    r516_col = tl.arange($(192))
    r516_lane = tl.arange($(16))
    for r516_group in range($(8)) {
      r516_left = tl.load($(candidate_vmul_77_ptr) + (((((((r516_group % $(8)) * $(1024)) + (r516_row[:, None] * $(16)) + r516_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + (((((r516_group % $(8)) * $(1024)) + (r516_row[:, None] * $(16)) + r516_lane[None, :]) % $(16)) * $(64)) + (((((r516_group % $(8)) * $(1024)) + (r516_row[:, None] * $(16)) + r516_lane[None, :]) / $(16)) % $(64))), dtype=tl.float32)
      r516_right = tl.load($(candidate_vto_copy_154_ptr) + (((((((r516_group % $(8)) * $(3072)) + (r516_lane[:, None] * $(192)) + r516_col[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r516_group % $(8)) * $(3072)) + (r516_lane[:, None] * $(192)) + r516_col[None, :]) / $(192)) % $(16)) * $(192)) + ((((r516_group % $(8)) * $(3072)) + (r516_lane[:, None] * $(192)) + r516_col[None, :]) % $(192))), dtype=tl.float32)
      r516_value = tl.dot(r516_left, r516_right)
      r516_value = r516_value.to(tl.float32)
      tl.store($(candidate_vbmm_79_ptr) + ((r516_group * $(12288)) + (r516_row[:, None] * $(192)) + r516_col[None, :]), r516_value)
    }
    r517_z = tl.arange($(98304))
    r517_value = (tl.load($(candidate_vbmm_79_ptr) + ((((r517_z / $(12288)) % $(8)) * $(12288)) + (((r517_z / $(192)) % $(64)) * $(192)) + (r517_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r517_value = r517_value.to(tl.float32)
    r517_value = tl.broadcast_to(r517_value, [$(98304)])
    tl.store($(candidate_vmul_79_ptr) + r517_z, r517_value)
    r518_z = tl.arange($(98304))
    r518_value = tl.load($(candidate_vmul_79_ptr) + ((((r518_z / $(192)) % $(8)) * $(12288)) + (((r518_z / $(1536)) % $(64)) * $(192)) + (r518_z % $(192))), dtype=tl.float32)
    r518_value = r518_value.to(tl.bfloat16)
    r518_value = tl.broadcast_to(r518_value, [$(98304)])
    tl.store($(candidate_vto_copy_160_ptr) + r518_z, r518_value)
    r519_row = tl.arange($(64))
    r519_col = tl.arange($(128))
    r519_lane = tl.arange($(16))
    for r519_group in range($(8)) {
      r519_left = tl.load($(candidate_vexp_8_ptr) + (((((((r519_group % $(8)) * $(1024)) + (r519_row[:, None] * $(16)) + r519_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + (((((r519_group % $(8)) * $(1024)) + (r519_row[:, None] * $(16)) + r519_lane[None, :]) % $(16)) * $(64)) + (((((r519_group % $(8)) * $(1024)) + (r519_row[:, None] * $(16)) + r519_lane[None, :]) / $(16)) % $(64))), dtype=tl.float32)
      r519_right = tl.load($(candidate_vto_copy_157_ptr) + (((((((r519_group % $(8)) * $(2048)) + (r519_lane[:, None] * $(128)) + r519_col[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r519_group % $(8)) * $(2048)) + (r519_lane[:, None] * $(128)) + r519_col[None, :]) / $(128)) % $(16)) * $(128)) + ((((r519_group % $(8)) * $(2048)) + (r519_lane[:, None] * $(128)) + r519_col[None, :]) % $(128))), dtype=tl.float32)
      r519_value = tl.dot(r519_left, r519_right)
      r519_value = r519_value.to(tl.float32)
      tl.store($(candidate_vbmm_80_ptr) + ((r519_group * $(8192)) + (r519_row[:, None] * $(128)) + r519_col[None, :]), r519_value)
    }
    r520_z = tl.arange($(65536))
    r520_value = tl.load($(candidate_vbmm_80_ptr) + ((((r520_z / $(128)) % $(8)) * $(8192)) + (((r520_z / $(1024)) % $(64)) * $(128)) + (r520_z % $(128))), dtype=tl.float32)
    r520_value = r520_value.to(tl.bfloat16)
    r520_value = tl.broadcast_to(r520_value, [$(65536)])
    tl.store($(candidate_vto_copy_161_ptr) + r520_z, r520_value)
    r521_z = tl.arange($(196608))
    r521_value = tl.where(((r521_z / $(1536)) % $(128)) < $(48), tl.load($(candidate_vto_copy_144_ptr) + ((((((r521_z / $(1536)) % $(128)) - $(0)) % $(48)) * $(1536)) + (((r521_z / $(192)) % $(8)) * $(192)) + (r521_z % $(192))), dtype=tl.bfloat16), tl.where(((r521_z / $(1536)) % $(128)) < $(64), tl.load($(candidate_vto_copy_152_ptr) + ((((((r521_z / $(1536)) % $(128)) - $(48)) % $(16)) * $(1536)) + (((r521_z / $(192)) % $(8)) * $(192)) + (r521_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vto_copy_160_ptr) + ((((((r521_z / $(1536)) % $(128)) - $(64)) % $(64)) * $(1536)) + (((r521_z / $(192)) % $(8)) * $(192)) + (r521_z % $(192))), dtype=tl.bfloat16)))
    r521_value = r521_value.to(tl.bfloat16)
    r521_value = tl.broadcast_to(r521_value, [$(196608)])
    tl.store($(candidate_vcat_42_ptr) + r521_z, r521_value)
    r522_z = tl.arange($(131072))
    r522_value = tl.where(((r522_z / $(1024)) % $(128)) < $(48), tl.load($(candidate_vto_copy_145_ptr) + ((((((r522_z / $(1024)) % $(128)) - $(0)) % $(48)) * $(1024)) + (((r522_z / $(128)) % $(8)) * $(128)) + (r522_z % $(128))), dtype=tl.bfloat16), tl.where(((r522_z / $(1024)) % $(128)) < $(64), tl.load($(candidate_vto_copy_153_ptr) + ((((((r522_z / $(1024)) % $(128)) - $(48)) % $(16)) * $(1024)) + (((r522_z / $(128)) % $(8)) * $(128)) + (r522_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vto_copy_161_ptr) + ((((((r522_z / $(1024)) % $(128)) - $(64)) % $(64)) * $(1024)) + (((r522_z / $(128)) % $(8)) * $(128)) + (r522_z % $(128))), dtype=tl.bfloat16)))
    r522_value = r522_value.to(tl.bfloat16)
    r522_value = tl.broadcast_to(r522_value, [$(131072)])
    tl.store($(candidate_vcat_43_ptr) + r522_z, r522_value)
    r523_row = tl.arange($(1024))
    r523_col = tl.arange($(512))
    r523_lane = tl.arange($(128))
    for r523_group in range($(1)) {
      r523_left = tl.load($(candidate_vcat_42_ptr) + (((((r523_row[:, None] * $(128)) + r523_lane[None, :]) % $(128)) * $(1536)) + (((((r523_row[:, None] * $(128)) + r523_lane[None, :]) / $(16384)) % $(8)) * $(192)) + ((((r523_row[:, None] * $(128)) + r523_lane[None, :]) / $(128)) % $(128))), dtype=tl.bfloat16)
      r523_right = tl.load($(candidate_vindex_2_ptr) + ((((((r523_lane[:, None] * $(512)) + r523_col[None, :]) / $(512)) % $(128)) * $(576)) + (((r523_lane[:, None] * $(512)) + r523_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r523_value = tl.dot(r523_left, r523_right)
      r523_value = r523_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_83_ptr) + ((r523_group * $(524288)) + (r523_row[:, None] * $(512)) + r523_col[None, :]), r523_value)
    }
    r524_row = tl.arange($(1024))
    r524_col = tl.arange($(512))
    r524_lane = tl.arange($(128))
    for r524_group in range($(1)) {
      r524_left = tl.load($(candidate_vcat_43_ptr) + (((((r524_row[:, None] * $(128)) + r524_lane[None, :]) % $(128)) * $(1024)) + (((((r524_row[:, None] * $(128)) + r524_lane[None, :]) / $(16384)) % $(8)) * $(128)) + ((((r524_row[:, None] * $(128)) + r524_lane[None, :]) / $(128)) % $(128))), dtype=tl.bfloat16)
      r524_right = tl.load($(candidate_vindex_2_ptr) + ((((((r524_lane[:, None] * $(512)) + r524_col[None, :]) / $(512)) % $(128)) * $(576)) + (((r524_lane[:, None] * $(512)) + r524_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r524_value = tl.dot(r524_left, r524_right)
      r524_value = r524_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_84_ptr) + ((r524_group * $(524288)) + (r524_row[:, None] * $(512)) + r524_col[None, :]), r524_value)
    }
    r525_z = tl.arange($(1048576))
    r525_value = tl.where(((r525_z / $(512)) % $(256)) < $(128), tl.load($(candidate_vbmm_83_ptr) + ((((((((r525_z / $(131072)) % $(8)) * $(65536)) + ((((r525_z / $(512)) % $(256)) - $(0)) * $(512)) + (r525_z % $(512))) / $(65536)) % $(8)) * $(65536)) + ((((((r525_z / $(131072)) % $(8)) * $(128)) + (((r525_z / $(512)) % $(256)) - $(0))) % $(128)) * $(512)) + (r525_z % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vbmm_84_ptr) + ((((((((r525_z / $(131072)) % $(8)) * $(65536)) + ((((r525_z / $(512)) % $(256)) - $(128)) * $(512)) + (r525_z % $(512))) / $(65536)) % $(8)) * $(65536)) + ((((((r525_z / $(131072)) % $(8)) * $(128)) + (((r525_z / $(512)) % $(256)) - $(128))) % $(128)) * $(512)) + (r525_z % $(512))), dtype=tl.bfloat16))
    r525_value = r525_value.to(tl.bfloat16)
    r525_value = tl.broadcast_to(r525_value, [$(1048576)])
    tl.store($(candidate_vcat_44_ptr) + r525_z, r525_value)
    r526_z = tl.arange($(1048576))
    r526_value = (tl.load($(candidate_vadd_40_ptr) + ((((r526_z / $(512)) % $(2048)) * $(512)) + (r526_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vcat_44_ptr) + ((((((((r526_z / $(512)) % $(2048)) * $(512)) + (r526_z % $(512))) / $(131072)) % $(8)) * $(131072)) + ((((r526_z / $(512)) % $(2048)) % $(256)) * $(512)) + (r526_z % $(512))), dtype=tl.bfloat16))
    r526_value = r526_value.to(tl.bfloat16)
    r526_value = tl.broadcast_to(r526_value, [$(1048576)])
    tl.store($(candidate_vadd_45_ptr) + r526_z, r526_value)
    r527_row = tl.arange($(128))
    r527_col = tl.arange($(1024))
    r527_lane = tl.arange($(512))
    for r527_group in range($(1)) {
      r527_left = tl.load($(candidate_vindex_2_ptr) + ((((((r527_row[:, None] * $(512)) + r527_lane[None, :]) / $(512)) % $(128)) * $(576)) + (((r527_row[:, None] * $(512)) + r527_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r527_right = tl.load($(candidate_vto_copy_20_ptr) + ((((((r527_lane[:, None] * $(1024)) + r527_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r527_lane[:, None] * $(1024)) + r527_col[None, :]) % $(128)) * $(512)) + ((((r527_lane[:, None] * $(1024)) + r527_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r527_value = tl.dot(r527_left, r527_right)
      r527_value = r527_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_85_ptr) + ((r527_group * $(131072)) + (r527_row[:, None] * $(1024)) + r527_col[None, :]), r527_value)
    }
    r528_row = tl.arange($(128))
    r528_col = tl.arange($(1024))
    r528_lane = tl.arange($(512))
    for r528_group in range($(1)) {
      r528_left = tl.load($(candidate_vindex_2_ptr) + ((((((r528_row[:, None] * $(512)) + r528_lane[None, :]) / $(512)) % $(128)) * $(576)) + (((r528_row[:, None] * $(512)) + r528_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r528_right = tl.load($(candidate_vto_copy_20_ptr) + ((((((r528_lane[:, None] * $(1024)) + r528_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r528_lane[:, None] * $(1024)) + r528_col[None, :]) % $(128)) * $(512)) + ((((r528_lane[:, None] * $(1024)) + r528_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r528_value = tl.dot(r528_left, r528_right)
      r528_value = r528_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_86_ptr) + ((r528_group * $(131072)) + (r528_row[:, None] * $(1024)) + r528_col[None, :]), r528_value)
    }
    r529_z = tl.arange($(49152))
    r529_value = tl.where((r529_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r529_z / $(1536)) % $(32)) * $(1024)) + (((r529_z / $(192)) % $(8)) * $(128)) + ((r529_z % $(192)) - $(0))) / $(1024)) % $(128)) * $(1536)) + (((((((r529_z / $(1536)) % $(32)) * $(1024)) + (((r529_z / $(192)) % $(8)) * $(128)) + ((r529_z % $(192)) - $(0))) / $(128)) % $(8)) * $(192)) + (((((r529_z / $(1536)) % $(32)) * $(1024)) + (((r529_z / $(192)) % $(8)) * $(128)) + ((r529_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vadd_6_ptr) + ((((((((r529_z / $(1536)) % $(32)) * $(512)) + (((r529_z / $(192)) % $(8)) * $(64)) + ((r529_z % $(192)) - $(128))) / $(512)) % $(128)) * $(512)) + (((((((r529_z / $(1536)) % $(32)) * $(512)) + (((r529_z / $(192)) % $(8)) * $(64)) + ((r529_z % $(192)) - $(128))) / $(64)) % $(8)) * $(64)) + (((((r529_z / $(1536)) % $(32)) * $(512)) + (((r529_z / $(192)) % $(8)) * $(64)) + ((r529_z % $(192)) - $(128))) % $(64))), dtype=tl.bfloat16))
    r529_value = r529_value.to(tl.bfloat16)
    r529_value = tl.broadcast_to(r529_value, [$(49152)])
    tl.store($(candidate_vcat_45_ptr) + r529_z, r529_value)
    r530_z = tl.arange($(196608))
    r530_value = tl.where((r530_z % $(192)) < $(128), tl.load($(candidate_vbmm_85_ptr) + ((((((((r530_z / $(1536)) % $(128)) * $(1024)) + (((r530_z / $(192)) % $(8)) * $(128)) + ((r530_z % $(192)) - $(0))) / $(1024)) % $(128)) * $(1024)) + (((((((r530_z / $(1536)) % $(128)) * $(1024)) + (((r530_z / $(192)) % $(8)) * $(128)) + ((r530_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r530_z / $(1536)) % $(128)) * $(1024)) + (((r530_z / $(192)) % $(8)) * $(128)) + ((r530_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vindex_2_ptr) + ((((r530_z / $(1536)) % $(128)) * $(576)) + ((((r530_z % $(192)) - $(128)) + $(512)) % $(576))), dtype=tl.bfloat16))
    r530_value = r530_value.to(tl.bfloat16)
    r530_value = tl.broadcast_to(r530_value, [$(196608)])
    tl.store($(candidate_vcat_46_ptr) + r530_z, r530_value)
    r531_z = tl.arange($(18432))
    r531_value = tl.load($(candidate_vcat_45_ptr) + ((((r531_z / $(192)) % $(12)) * $(1536)) + (((r531_z / $(2304)) % $(8)) * $(192)) + (r531_z % $(192))), dtype=tl.bfloat16)
    r531_value = r531_value.to(tl.float32)
    r531_value = tl.broadcast_to(r531_value, [$(18432)])
    tl.store($(candidate_vto_copy_162_ptr) + r531_z, r531_value)
    r532_z = tl.arange($(18432))
    r532_value = tl.load($(candidate_vcat_46_ptr) + ((((r532_z / $(192)) % $(12)) * $(1536)) + (((r532_z / $(2304)) % $(8)) * $(192)) + (r532_z % $(192))), dtype=tl.bfloat16)
    r532_value = r532_value.to(tl.float32)
    r532_value = tl.broadcast_to(r532_value, [$(18432)])
    tl.store($(candidate_vto_copy_163_ptr) + r532_z, r532_value)
    r533_z = tl.arange($(12288))
    r533_value = tl.load($(candidate_vbmm_86_ptr) + ((((r533_z / $(128)) % $(12)) * $(1024)) + (((r533_z / $(1536)) % $(8)) * $(128)) + (r533_z % $(128))), dtype=tl.bfloat16)
    r533_value = r533_value.to(tl.float32)
    r533_value = tl.broadcast_to(r533_value, [$(12288)])
    tl.store($(candidate_vto_copy_164_ptr) + r533_z, r533_value)
    r534_z = tl.arange($(12288))
    r534_value = tl.load($(candidate_vmm_28_ptr) + ((((r534_z / $(128)) % $(12)) * $(1024)) + (((r534_z / $(1536)) % $(8)) * $(128)) + (r534_z % $(128))), dtype=tl.bfloat16)
    r534_value = r534_value.to(tl.float32)
    r534_value = tl.broadcast_to(r534_value, [$(12288)])
    tl.store($(candidate_vto_copy_165_ptr) + r534_z, r534_value)
    r535_z = tl.arange($(12288))
    r535_value = tl.load($(candidate_vbmm_6_ptr) + ((((r535_z / $(1536)) % $(8)) * $(4096)) + (((r535_z / $(128)) % $(12)) * $(128)) + (r535_z % $(128))), dtype=tl.bfloat16)
    r535_value = r535_value.to(tl.float32)
    r535_value = tl.broadcast_to(r535_value, [$(12288)])
    tl.store($(candidate_vto_copy_166_ptr) + r535_z, r535_value)
    r536_row = tl.arange($(12))
    r536_col = tl.arange($(12))
    r536_lane = tl.arange($(192))
    for r536_group in range($(8)) {
      r536_left = tl.load($(candidate_vto_copy_162_ptr) + (((((((r536_group % $(8)) * $(2304)) + (r536_row[:, None] * $(192)) + r536_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r536_group % $(8)) * $(2304)) + (r536_row[:, None] * $(192)) + r536_lane[None, :]) / $(192)) % $(12)) * $(192)) + ((((r536_group % $(8)) * $(2304)) + (r536_row[:, None] * $(192)) + r536_lane[None, :]) % $(192))), dtype=tl.float32)
      r536_right = tl.load($(candidate_vto_copy_163_ptr) + (((((((r536_group % $(8)) * $(2304)) + (r536_lane[:, None] * $(12)) + r536_col[None, :]) / $(2304)) % $(8)) * $(2304)) + (((((r536_group % $(8)) * $(2304)) + (r536_lane[:, None] * $(12)) + r536_col[None, :]) % $(12)) * $(192)) + (((((r536_group % $(8)) * $(2304)) + (r536_lane[:, None] * $(12)) + r536_col[None, :]) / $(12)) % $(192))), dtype=tl.float32)
      r536_value = tl.dot(r536_left, r536_right)
      r536_value = r536_value.to(tl.float32)
      tl.store($(candidate_vbmm_87_ptr) + ((r536_group * $(144)) + (r536_row[:, None] * $(12)) + r536_col[None, :]), r536_value)
    }
    r537_z = tl.arange($(1152))
    r537_value = (tl.load($(candidate_vbmm_87_ptr) + ((((r537_z / $(144)) % $(8)) * $(144)) + (((r537_z / $(12)) % $(12)) * $(12)) + (r537_z % $(12))), dtype=tl.float32) * 0.07216878364870322)
    r537_value = r537_value.to(tl.float32)
    r537_value = tl.broadcast_to(r537_value, [$(1152)])
    tl.store($(candidate_vmul_80_ptr) + r537_z, r537_value)
    r538_z = tl.arange($(12))
    r538_value = (tl.cast(r538_z, tl.int64) + tl.full([], $(12), dtype=tl.int64))
    r538_value = tl.broadcast_to(r538_value, [$(12)])
    tl.store($(candidate_vadd_46_ptr) + r538_z, r538_value)
    r539_z = tl.arange($(12))
    r539_value = (tl.load($(candidate_vadd_46_ptr) + r539_z, dtype=tl.int64) - tl.full([], $(12), dtype=tl.int64))
    r539_value = tl.broadcast_to(r539_value, [$(12)])
    tl.store($(candidate_vsub_48_ptr) + r539_z, r539_value)
    r540_z = tl.arange($(144))
    r540_value = (tl.cast((r540_z % $(12)), tl.int64) > tl.load($(candidate_vsub_48_ptr) + ((r540_z / $(12)) % $(12)), dtype=tl.int64))
    r540_value = tl.broadcast_to(r540_value, [$(144)])
    tl.store($(candidate_vgt_21_ptr) + r540_z, r540_value)
    r541_z = tl.arange($(1152))
    r541_value = tl.where(tl.load($(candidate_vgt_21_ptr) + ((((r541_z / $(12)) % $(12)) * $(12)) + (r541_z % $(12))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_80_ptr) + ((((r541_z / $(144)) % $(8)) * $(144)) + (((r541_z / $(12)) % $(12)) * $(12)) + (r541_z % $(12))), dtype=tl.float32))
    r541_value = r541_value.to(tl.float32)
    r541_value = tl.broadcast_to(r541_value, [$(1152)])
    tl.store($(candidate_vmasked_fill_21_ptr) + r541_z, r541_value)
    r542_z = tl.arange($(1152))
    r542_value = (tl.load($(candidate_vmasked_fill_21_ptr) + ((((r542_z / $(144)) % $(8)) * $(144)) + (((r542_z / $(12)) % $(12)) * $(12)) + (r542_z % $(12))), dtype=tl.float32) - tl.load($(candidate_vcat_7_ptr) + ((((r542_z / $(12)) % $(12)) * $(8)) + ((r542_z / $(144)) % $(8))), dtype=tl.float32))
    r542_value = r542_value.to(tl.float32)
    r542_value = tl.broadcast_to(r542_value, [$(1152)])
    tl.store($(candidate_vsub_49_ptr) + r542_z, r542_value)
    r543_z = tl.arange($(1152))
    r543_value = tl.exp(tl.load($(candidate_vsub_49_ptr) + ((((r543_z / $(144)) % $(8)) * $(144)) + (((r543_z / $(12)) % $(12)) * $(12)) + (r543_z % $(12))), dtype=tl.float32))
    r543_value = r543_value.to(tl.float32)
    r543_value = tl.broadcast_to(r543_value, [$(1152)])
    tl.store($(candidate_vexp_9_ptr) + r543_z, r543_value)
    r544_z = tl.arange($(12288))
    r544_value = (tl.load($(candidate_vto_copy_165_ptr) + ((((r544_z / $(1536)) % $(8)) * $(1536)) + (((r544_z / $(128)) % $(12)) * $(128)) + (r544_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_166_ptr) + ((((r544_z / $(1536)) % $(8)) * $(1536)) + (((r544_z / $(128)) % $(12)) * $(128)) + (r544_z % $(128))), dtype=tl.float32))
    r544_value = r544_value.to(tl.float32)
    r544_value = tl.broadcast_to(r544_value, [$(12288)])
    tl.store($(candidate_vmul_81_ptr) + r544_z, r544_value)
    r545_z = tl.arange($(96))
    r545_lane = tl.arange($(128))
    r545_value = tl.sum(tl.load($(candidate_vmul_81_ptr) + ((((r545_z[:, None] / $(12)) % $(8)) * $(1536)) + ((r545_z[:, None] % $(12)) * $(128)) + r545_lane[None, :]), dtype=tl.float32), axis=1)
    r545_value = r545_value.to(tl.float32)
    r545_value = tl.broadcast_to(r545_value, [$(96)])
    tl.store($(candidate_vsum_13_ptr) + r545_z, r545_value)
    r546_row = tl.arange($(12))
    r546_col = tl.arange($(12))
    r546_lane = tl.arange($(128))
    for r546_group in range($(8)) {
      r546_left = tl.load($(candidate_vto_copy_165_ptr) + (((((((r546_group % $(8)) * $(1536)) + (r546_row[:, None] * $(128)) + r546_lane[None, :]) / $(1536)) % $(8)) * $(1536)) + ((((((r546_group % $(8)) * $(1536)) + (r546_row[:, None] * $(128)) + r546_lane[None, :]) / $(128)) % $(12)) * $(128)) + ((((r546_group % $(8)) * $(1536)) + (r546_row[:, None] * $(128)) + r546_lane[None, :]) % $(128))), dtype=tl.float32)
      r546_right = tl.load($(candidate_vto_copy_164_ptr) + (((((((r546_group % $(8)) * $(1536)) + (r546_lane[:, None] * $(12)) + r546_col[None, :]) / $(1536)) % $(8)) * $(1536)) + (((((r546_group % $(8)) * $(1536)) + (r546_lane[:, None] * $(12)) + r546_col[None, :]) % $(12)) * $(128)) + (((((r546_group % $(8)) * $(1536)) + (r546_lane[:, None] * $(12)) + r546_col[None, :]) / $(12)) % $(128))), dtype=tl.float32)
      r546_value = tl.dot(r546_left, r546_right)
      r546_value = r546_value.to(tl.float32)
      tl.store($(candidate_vbmm_88_ptr) + ((r546_group * $(144)) + (r546_row[:, None] * $(12)) + r546_col[None, :]), r546_value)
    }
    r547_z = tl.arange($(1152))
    r547_value = (tl.load($(candidate_vbmm_88_ptr) + ((((r547_z / $(144)) % $(8)) * $(144)) + (((r547_z / $(12)) % $(12)) * $(12)) + (r547_z % $(12))), dtype=tl.float32) - tl.load($(candidate_vsum_13_ptr) + ((((r547_z / $(144)) % $(8)) * $(12)) + ((r547_z / $(12)) % $(12))), dtype=tl.float32))
    r547_value = r547_value.to(tl.float32)
    r547_value = tl.broadcast_to(r547_value, [$(1152)])
    tl.store($(candidate_vsub_50_ptr) + r547_z, r547_value)
    r548_z = tl.arange($(1152))
    r548_value = (tl.load($(candidate_vexp_9_ptr) + ((((r548_z / $(144)) % $(8)) * $(144)) + (((r548_z / $(12)) % $(12)) * $(12)) + (r548_z % $(12))), dtype=tl.float32) * tl.load($(candidate_vsub_50_ptr) + ((((r548_z / $(144)) % $(8)) * $(144)) + (((r548_z / $(12)) % $(12)) * $(12)) + (r548_z % $(12))), dtype=tl.float32))
    r548_value = r548_value.to(tl.float32)
    r548_value = tl.broadcast_to(r548_value, [$(1152)])
    tl.store($(candidate_vmul_82_ptr) + r548_z, r548_value)
    r549_row = tl.arange($(12))
    r549_col = tl.arange($(192))
    r549_lane = tl.arange($(12))
    for r549_group in range($(8)) {
      r549_left = tl.load($(candidate_vmul_82_ptr) + (((((((r549_group % $(8)) * $(144)) + (r549_row[:, None] * $(12)) + r549_lane[None, :]) / $(144)) % $(8)) * $(144)) + (((((r549_group % $(8)) * $(144)) + (r549_row[:, None] * $(12)) + r549_lane[None, :]) % $(12)) * $(12)) + (((((r549_group % $(8)) * $(144)) + (r549_row[:, None] * $(12)) + r549_lane[None, :]) / $(12)) % $(12))), dtype=tl.float32)
      r549_right = tl.load($(candidate_vto_copy_162_ptr) + (((((((r549_group % $(8)) * $(2304)) + (r549_lane[:, None] * $(192)) + r549_col[None, :]) / $(2304)) % $(8)) * $(2304)) + ((((((r549_group % $(8)) * $(2304)) + (r549_lane[:, None] * $(192)) + r549_col[None, :]) / $(192)) % $(12)) * $(192)) + ((((r549_group % $(8)) * $(2304)) + (r549_lane[:, None] * $(192)) + r549_col[None, :]) % $(192))), dtype=tl.float32)
      r549_value = tl.dot(r549_left, r549_right)
      r549_value = r549_value.to(tl.float32)
      tl.store($(candidate_vbmm_90_ptr) + ((r549_group * $(2304)) + (r549_row[:, None] * $(192)) + r549_col[None, :]), r549_value)
    }
    r550_z = tl.arange($(18432))
    r550_value = (tl.load($(candidate_vbmm_90_ptr) + ((((r550_z / $(2304)) % $(8)) * $(2304)) + (((r550_z / $(192)) % $(12)) * $(192)) + (r550_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r550_value = r550_value.to(tl.float32)
    r550_value = tl.broadcast_to(r550_value, [$(18432)])
    tl.store($(candidate_vmul_84_ptr) + r550_z, r550_value)
    r551_z = tl.arange($(18432))
    r551_value = tl.load($(candidate_vmul_84_ptr) + ((((r551_z / $(192)) % $(8)) * $(2304)) + (((r551_z / $(1536)) % $(12)) * $(192)) + (r551_z % $(192))), dtype=tl.float32)
    r551_value = r551_value.to(tl.bfloat16)
    r551_value = tl.broadcast_to(r551_value, [$(18432)])
    tl.store($(candidate_vto_copy_168_ptr) + r551_z, r551_value)
    r552_row = tl.arange($(12))
    r552_col = tl.arange($(128))
    r552_lane = tl.arange($(12))
    for r552_group in range($(8)) {
      r552_left = tl.load($(candidate_vexp_9_ptr) + (((((((r552_group % $(8)) * $(144)) + (r552_row[:, None] * $(12)) + r552_lane[None, :]) / $(144)) % $(8)) * $(144)) + (((((r552_group % $(8)) * $(144)) + (r552_row[:, None] * $(12)) + r552_lane[None, :]) % $(12)) * $(12)) + (((((r552_group % $(8)) * $(144)) + (r552_row[:, None] * $(12)) + r552_lane[None, :]) / $(12)) % $(12))), dtype=tl.float32)
      r552_right = tl.load($(candidate_vto_copy_165_ptr) + (((((((r552_group % $(8)) * $(1536)) + (r552_lane[:, None] * $(128)) + r552_col[None, :]) / $(1536)) % $(8)) * $(1536)) + ((((((r552_group % $(8)) * $(1536)) + (r552_lane[:, None] * $(128)) + r552_col[None, :]) / $(128)) % $(12)) * $(128)) + ((((r552_group % $(8)) * $(1536)) + (r552_lane[:, None] * $(128)) + r552_col[None, :]) % $(128))), dtype=tl.float32)
      r552_value = tl.dot(r552_left, r552_right)
      r552_value = r552_value.to(tl.float32)
      tl.store($(candidate_vbmm_91_ptr) + ((r552_group * $(1536)) + (r552_row[:, None] * $(128)) + r552_col[None, :]), r552_value)
    }
    r553_z = tl.arange($(12288))
    r553_value = tl.load($(candidate_vbmm_91_ptr) + ((((r553_z / $(128)) % $(8)) * $(1536)) + (((r553_z / $(1024)) % $(12)) * $(128)) + (r553_z % $(128))), dtype=tl.float32)
    r553_value = r553_value.to(tl.bfloat16)
    r553_value = tl.broadcast_to(r553_value, [$(12288)])
    tl.store($(candidate_vto_copy_169_ptr) + r553_z, r553_value)
    r554_z = tl.arange($(73728))
    r554_value = tl.where(((r554_z / $(1536)) % $(48)) < $(12), tl.load($(candidate_vto_copy_168_ptr) + ((((((r554_z / $(1536)) % $(48)) - $(0)) % $(12)) * $(1536)) + (((r554_z / $(192)) % $(8)) * $(192)) + (r554_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r554_value = r554_value.to(tl.bfloat16)
    r554_value = tl.broadcast_to(r554_value, [$(73728)])
    tl.store($(candidate_vcat_47_ptr) + r554_z, r554_value)
    r555_z = tl.arange($(49152))
    r555_value = tl.where(((r555_z / $(1024)) % $(48)) < $(12), tl.load($(candidate_vto_copy_169_ptr) + ((((((r555_z / $(1024)) % $(48)) - $(0)) % $(12)) * $(1024)) + (((r555_z / $(128)) % $(8)) * $(128)) + (r555_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r555_value = r555_value.to(tl.bfloat16)
    r555_value = tl.broadcast_to(r555_value, [$(49152)])
    tl.store($(candidate_vcat_48_ptr) + r555_z, r555_value)
    r556_z = tl.arange($(6144))
    r556_value = tl.load($(candidate_vcat_45_ptr) + ((((r556_z / $(192)) % $(4)) * $(1536)) + (((r556_z / $(768)) % $(8)) * $(192)) + (r556_z % $(192)) + $(18432)), dtype=tl.bfloat16)
    r556_value = r556_value.to(tl.float32)
    r556_value = tl.broadcast_to(r556_value, [$(6144)])
    tl.store($(candidate_vto_copy_170_ptr) + r556_z, r556_value)
    r557_z = tl.arange($(6144))
    r557_value = tl.load($(candidate_vcat_46_ptr) + ((((r557_z / $(192)) % $(4)) * $(1536)) + (((r557_z / $(768)) % $(8)) * $(192)) + (r557_z % $(192)) + $(73728)), dtype=tl.bfloat16)
    r557_value = r557_value.to(tl.float32)
    r557_value = tl.broadcast_to(r557_value, [$(6144)])
    tl.store($(candidate_vto_copy_171_ptr) + r557_z, r557_value)
    r558_z = tl.arange($(4096))
    r558_value = tl.load($(candidate_vbmm_86_ptr) + ((((r558_z / $(128)) % $(4)) * $(1024)) + (((r558_z / $(512)) % $(8)) * $(128)) + (r558_z % $(128)) + $(49152)), dtype=tl.bfloat16)
    r558_value = r558_value.to(tl.float32)
    r558_value = tl.broadcast_to(r558_value, [$(4096)])
    tl.store($(candidate_vto_copy_172_ptr) + r558_z, r558_value)
    r559_z = tl.arange($(4096))
    r559_value = tl.load($(candidate_vmm_28_ptr) + ((((r559_z / $(128)) % $(4)) * $(1024)) + (((r559_z / $(512)) % $(8)) * $(128)) + (r559_z % $(128)) + $(12288)), dtype=tl.bfloat16)
    r559_value = r559_value.to(tl.float32)
    r559_value = tl.broadcast_to(r559_value, [$(4096)])
    tl.store($(candidate_vto_copy_173_ptr) + r559_z, r559_value)
    r560_z = tl.arange($(4096))
    r560_value = tl.load($(candidate_vbmm_6_ptr) + ((((r560_z / $(512)) % $(8)) * $(4096)) + (((r560_z / $(128)) % $(4)) * $(128)) + (r560_z % $(128)) + $(1536)), dtype=tl.bfloat16)
    r560_value = r560_value.to(tl.float32)
    r560_value = tl.broadcast_to(r560_value, [$(4096)])
    tl.store($(candidate_vto_copy_174_ptr) + r560_z, r560_value)
    r561_row = tl.arange($(4))
    r561_col = tl.arange($(4))
    r561_lane = tl.arange($(192))
    for r561_group in range($(8)) {
      r561_left = tl.load($(candidate_vto_copy_170_ptr) + (((((((r561_group % $(8)) * $(768)) + (r561_row[:, None] * $(192)) + r561_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r561_group % $(8)) * $(768)) + (r561_row[:, None] * $(192)) + r561_lane[None, :]) / $(192)) % $(4)) * $(192)) + ((((r561_group % $(8)) * $(768)) + (r561_row[:, None] * $(192)) + r561_lane[None, :]) % $(192))), dtype=tl.float32)
      r561_right = tl.load($(candidate_vto_copy_171_ptr) + (((((((r561_group % $(8)) * $(768)) + (r561_lane[:, None] * $(4)) + r561_col[None, :]) / $(768)) % $(8)) * $(768)) + (((((r561_group % $(8)) * $(768)) + (r561_lane[:, None] * $(4)) + r561_col[None, :]) % $(4)) * $(192)) + (((((r561_group % $(8)) * $(768)) + (r561_lane[:, None] * $(4)) + r561_col[None, :]) / $(4)) % $(192))), dtype=tl.float32)
      r561_value = tl.dot(r561_left, r561_right)
      r561_value = r561_value.to(tl.float32)
      tl.store($(candidate_vbmm_92_ptr) + ((r561_group * $(16)) + (r561_row[:, None] * $(4)) + r561_col[None, :]), r561_value)
    }
    r562_z = tl.arange($(128))
    r562_value = (tl.load($(candidate_vbmm_92_ptr) + ((((r562_z / $(16)) % $(8)) * $(16)) + (((r562_z / $(4)) % $(4)) * $(4)) + (r562_z % $(4))), dtype=tl.float32) * 0.07216878364870322)
    r562_value = r562_value.to(tl.float32)
    r562_value = tl.broadcast_to(r562_value, [$(128)])
    tl.store($(candidate_vmul_85_ptr) + r562_z, r562_value)
    r563_z = tl.arange($(4))
    r563_value = (tl.cast(r563_z, tl.int64) + tl.full([], $(4), dtype=tl.int64))
    r563_value = tl.broadcast_to(r563_value, [$(4)])
    tl.store($(candidate_vadd_47_ptr) + r563_z, r563_value)
    r564_z = tl.arange($(4))
    r564_value = (tl.load($(candidate_vadd_47_ptr) + r564_z, dtype=tl.int64) - tl.full([], $(4), dtype=tl.int64))
    r564_value = tl.broadcast_to(r564_value, [$(4)])
    tl.store($(candidate_vsub_51_ptr) + r564_z, r564_value)
    r565_z = tl.arange($(16))
    r565_value = (tl.cast((r565_z % $(4)), tl.int64) > tl.load($(candidate_vsub_51_ptr) + ((r565_z / $(4)) % $(4)), dtype=tl.int64))
    r565_value = tl.broadcast_to(r565_value, [$(16)])
    tl.store($(candidate_vgt_22_ptr) + r565_z, r565_value)
    r566_z = tl.arange($(128))
    r566_value = tl.where(tl.load($(candidate_vgt_22_ptr) + ((((r566_z / $(4)) % $(4)) * $(4)) + (r566_z % $(4))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_85_ptr) + ((((r566_z / $(16)) % $(8)) * $(16)) + (((r566_z / $(4)) % $(4)) * $(4)) + (r566_z % $(4))), dtype=tl.float32))
    r566_value = r566_value.to(tl.float32)
    r566_value = tl.broadcast_to(r566_value, [$(128)])
    tl.store($(candidate_vmasked_fill_22_ptr) + r566_z, r566_value)
    r567_z = tl.arange($(128))
    r567_value = (tl.load($(candidate_vmasked_fill_22_ptr) + ((((r567_z / $(16)) % $(8)) * $(16)) + (((r567_z / $(4)) % $(4)) * $(4)) + (r567_z % $(4))), dtype=tl.float32) - tl.load($(candidate_vcat_7_ptr) + ((((r567_z / $(4)) % $(4)) * $(8)) + ((r567_z / $(16)) % $(8)) + $(96)), dtype=tl.float32))
    r567_value = r567_value.to(tl.float32)
    r567_value = tl.broadcast_to(r567_value, [$(128)])
    tl.store($(candidate_vsub_52_ptr) + r567_z, r567_value)
    r568_z = tl.arange($(128))
    r568_value = tl.exp(tl.load($(candidate_vsub_52_ptr) + ((((r568_z / $(16)) % $(8)) * $(16)) + (((r568_z / $(4)) % $(4)) * $(4)) + (r568_z % $(4))), dtype=tl.float32))
    r568_value = r568_value.to(tl.float32)
    r568_value = tl.broadcast_to(r568_value, [$(128)])
    tl.store($(candidate_vexp_10_ptr) + r568_z, r568_value)
    r569_z = tl.arange($(4096))
    r569_value = (tl.load($(candidate_vto_copy_173_ptr) + ((((r569_z / $(512)) % $(8)) * $(512)) + (((r569_z / $(128)) % $(4)) * $(128)) + (r569_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_174_ptr) + ((((r569_z / $(512)) % $(8)) * $(512)) + (((r569_z / $(128)) % $(4)) * $(128)) + (r569_z % $(128))), dtype=tl.float32))
    r569_value = r569_value.to(tl.float32)
    r569_value = tl.broadcast_to(r569_value, [$(4096)])
    tl.store($(candidate_vmul_86_ptr) + r569_z, r569_value)
    r570_z = tl.arange($(32))
    r570_lane = tl.arange($(128))
    r570_value = tl.sum(tl.load($(candidate_vmul_86_ptr) + ((((r570_z[:, None] / $(4)) % $(8)) * $(512)) + ((r570_z[:, None] % $(4)) * $(128)) + r570_lane[None, :]), dtype=tl.float32), axis=1)
    r570_value = r570_value.to(tl.float32)
    r570_value = tl.broadcast_to(r570_value, [$(32)])
    tl.store($(candidate_vsum_14_ptr) + r570_z, r570_value)
    r571_row = tl.arange($(4))
    r571_col = tl.arange($(4))
    r571_lane = tl.arange($(128))
    for r571_group in range($(8)) {
      r571_left = tl.load($(candidate_vto_copy_173_ptr) + (((((((r571_group % $(8)) * $(512)) + (r571_row[:, None] * $(128)) + r571_lane[None, :]) / $(512)) % $(8)) * $(512)) + ((((((r571_group % $(8)) * $(512)) + (r571_row[:, None] * $(128)) + r571_lane[None, :]) / $(128)) % $(4)) * $(128)) + ((((r571_group % $(8)) * $(512)) + (r571_row[:, None] * $(128)) + r571_lane[None, :]) % $(128))), dtype=tl.float32)
      r571_right = tl.load($(candidate_vto_copy_172_ptr) + (((((((r571_group % $(8)) * $(512)) + (r571_lane[:, None] * $(4)) + r571_col[None, :]) / $(512)) % $(8)) * $(512)) + (((((r571_group % $(8)) * $(512)) + (r571_lane[:, None] * $(4)) + r571_col[None, :]) % $(4)) * $(128)) + (((((r571_group % $(8)) * $(512)) + (r571_lane[:, None] * $(4)) + r571_col[None, :]) / $(4)) % $(128))), dtype=tl.float32)
      r571_value = tl.dot(r571_left, r571_right)
      r571_value = r571_value.to(tl.float32)
      tl.store($(candidate_vbmm_93_ptr) + ((r571_group * $(16)) + (r571_row[:, None] * $(4)) + r571_col[None, :]), r571_value)
    }
    r572_z = tl.arange($(128))
    r572_value = (tl.load($(candidate_vbmm_93_ptr) + ((((r572_z / $(16)) % $(8)) * $(16)) + (((r572_z / $(4)) % $(4)) * $(4)) + (r572_z % $(4))), dtype=tl.float32) - tl.load($(candidate_vsum_14_ptr) + ((((r572_z / $(16)) % $(8)) * $(4)) + ((r572_z / $(4)) % $(4))), dtype=tl.float32))
    r572_value = r572_value.to(tl.float32)
    r572_value = tl.broadcast_to(r572_value, [$(128)])
    tl.store($(candidate_vsub_53_ptr) + r572_z, r572_value)
    r573_z = tl.arange($(128))
    r573_value = (tl.load($(candidate_vexp_10_ptr) + ((((r573_z / $(16)) % $(8)) * $(16)) + (((r573_z / $(4)) % $(4)) * $(4)) + (r573_z % $(4))), dtype=tl.float32) * tl.load($(candidate_vsub_53_ptr) + ((((r573_z / $(16)) % $(8)) * $(16)) + (((r573_z / $(4)) % $(4)) * $(4)) + (r573_z % $(4))), dtype=tl.float32))
    r573_value = r573_value.to(tl.float32)
    r573_value = tl.broadcast_to(r573_value, [$(128)])
    tl.store($(candidate_vmul_87_ptr) + r573_z, r573_value)
    r574_row = tl.arange($(4))
    r574_col = tl.arange($(192))
    r574_lane = tl.arange($(4))
    for r574_group in range($(8)) {
      r574_left = tl.load($(candidate_vmul_87_ptr) + (((((((r574_group % $(8)) * $(16)) + (r574_row[:, None] * $(4)) + r574_lane[None, :]) / $(16)) % $(8)) * $(16)) + (((((r574_group % $(8)) * $(16)) + (r574_row[:, None] * $(4)) + r574_lane[None, :]) % $(4)) * $(4)) + (((((r574_group % $(8)) * $(16)) + (r574_row[:, None] * $(4)) + r574_lane[None, :]) / $(4)) % $(4))), dtype=tl.float32)
      r574_right = tl.load($(candidate_vto_copy_170_ptr) + (((((((r574_group % $(8)) * $(768)) + (r574_lane[:, None] * $(192)) + r574_col[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r574_group % $(8)) * $(768)) + (r574_lane[:, None] * $(192)) + r574_col[None, :]) / $(192)) % $(4)) * $(192)) + ((((r574_group % $(8)) * $(768)) + (r574_lane[:, None] * $(192)) + r574_col[None, :]) % $(192))), dtype=tl.float32)
      r574_value = tl.dot(r574_left, r574_right)
      r574_value = r574_value.to(tl.float32)
      tl.store($(candidate_vbmm_95_ptr) + ((r574_group * $(768)) + (r574_row[:, None] * $(192)) + r574_col[None, :]), r574_value)
    }
    r575_z = tl.arange($(6144))
    r575_value = (tl.load($(candidate_vbmm_95_ptr) + ((((r575_z / $(768)) % $(8)) * $(768)) + (((r575_z / $(192)) % $(4)) * $(192)) + (r575_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r575_value = r575_value.to(tl.float32)
    r575_value = tl.broadcast_to(r575_value, [$(6144)])
    tl.store($(candidate_vmul_89_ptr) + r575_z, r575_value)
    r576_z = tl.arange($(6144))
    r576_value = tl.load($(candidate_vmul_89_ptr) + ((((r576_z / $(192)) % $(8)) * $(768)) + (((r576_z / $(1536)) % $(4)) * $(192)) + (r576_z % $(192))), dtype=tl.float32)
    r576_value = r576_value.to(tl.bfloat16)
    r576_value = tl.broadcast_to(r576_value, [$(6144)])
    tl.store($(candidate_vto_copy_176_ptr) + r576_z, r576_value)
    r577_row = tl.arange($(4))
    r577_col = tl.arange($(128))
    r577_lane = tl.arange($(4))
    for r577_group in range($(8)) {
      r577_left = tl.load($(candidate_vexp_10_ptr) + (((((((r577_group % $(8)) * $(16)) + (r577_row[:, None] * $(4)) + r577_lane[None, :]) / $(16)) % $(8)) * $(16)) + (((((r577_group % $(8)) * $(16)) + (r577_row[:, None] * $(4)) + r577_lane[None, :]) % $(4)) * $(4)) + (((((r577_group % $(8)) * $(16)) + (r577_row[:, None] * $(4)) + r577_lane[None, :]) / $(4)) % $(4))), dtype=tl.float32)
      r577_right = tl.load($(candidate_vto_copy_173_ptr) + (((((((r577_group % $(8)) * $(512)) + (r577_lane[:, None] * $(128)) + r577_col[None, :]) / $(512)) % $(8)) * $(512)) + ((((((r577_group % $(8)) * $(512)) + (r577_lane[:, None] * $(128)) + r577_col[None, :]) / $(128)) % $(4)) * $(128)) + ((((r577_group % $(8)) * $(512)) + (r577_lane[:, None] * $(128)) + r577_col[None, :]) % $(128))), dtype=tl.float32)
      r577_value = tl.dot(r577_left, r577_right)
      r577_value = r577_value.to(tl.float32)
      tl.store($(candidate_vbmm_96_ptr) + ((r577_group * $(512)) + (r577_row[:, None] * $(128)) + r577_col[None, :]), r577_value)
    }
    r578_z = tl.arange($(4096))
    r578_value = tl.load($(candidate_vbmm_96_ptr) + ((((r578_z / $(128)) % $(8)) * $(512)) + (((r578_z / $(1024)) % $(4)) * $(128)) + (r578_z % $(128))), dtype=tl.float32)
    r578_value = r578_value.to(tl.bfloat16)
    r578_value = tl.broadcast_to(r578_value, [$(4096)])
    tl.store($(candidate_vto_copy_177_ptr) + r578_z, r578_value)
    r579_z = tl.arange($(24576))
    r579_value = tl.where(((r579_z / $(1536)) % $(16)) < $(4), tl.load($(candidate_vto_copy_176_ptr) + ((((((r579_z / $(1536)) % $(16)) - $(0)) % $(4)) * $(1536)) + (((r579_z / $(192)) % $(8)) * $(192)) + (r579_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r579_value = r579_value.to(tl.bfloat16)
    r579_value = tl.broadcast_to(r579_value, [$(24576)])
    tl.store($(candidate_vcat_49_ptr) + r579_z, r579_value)
    r580_z = tl.arange($(16384))
    r580_value = tl.where(((r580_z / $(1024)) % $(16)) < $(4), tl.load($(candidate_vto_copy_177_ptr) + ((((((r580_z / $(1024)) % $(16)) - $(0)) % $(4)) * $(1024)) + (((r580_z / $(128)) % $(8)) * $(128)) + (r580_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r580_value = r580_value.to(tl.bfloat16)
    r580_value = tl.broadcast_to(r580_value, [$(16384)])
    tl.store($(candidate_vcat_50_ptr) + r580_z, r580_value)
    r581_z = tl.arange($(24576))
    r581_value = tl.load($(candidate_vcat_45_ptr) + ((((r581_z / $(192)) % $(16)) * $(1536)) + (((r581_z / $(3072)) % $(8)) * $(192)) + (r581_z % $(192)) + $(24576)), dtype=tl.bfloat16)
    r581_value = r581_value.to(tl.float32)
    r581_value = tl.broadcast_to(r581_value, [$(24576)])
    tl.store($(candidate_vto_copy_178_ptr) + r581_z, r581_value)
    r582_z = tl.arange($(24576))
    r582_value = tl.load($(candidate_vcat_46_ptr) + ((((r582_z / $(192)) % $(16)) * $(1536)) + (((r582_z / $(3072)) % $(8)) * $(192)) + (r582_z % $(192)) + $(98304)), dtype=tl.bfloat16)
    r582_value = r582_value.to(tl.float32)
    r582_value = tl.broadcast_to(r582_value, [$(24576)])
    tl.store($(candidate_vto_copy_179_ptr) + r582_z, r582_value)
    r583_z = tl.arange($(16384))
    r583_value = tl.load($(candidate_vbmm_86_ptr) + ((((r583_z / $(128)) % $(16)) * $(1024)) + (((r583_z / $(2048)) % $(8)) * $(128)) + (r583_z % $(128)) + $(65536)), dtype=tl.bfloat16)
    r583_value = r583_value.to(tl.float32)
    r583_value = tl.broadcast_to(r583_value, [$(16384)])
    tl.store($(candidate_vto_copy_180_ptr) + r583_z, r583_value)
    r584_z = tl.arange($(16384))
    r584_value = tl.load($(candidate_vmm_28_ptr) + ((((r584_z / $(128)) % $(16)) * $(1024)) + (((r584_z / $(2048)) % $(8)) * $(128)) + (r584_z % $(128)) + $(16384)), dtype=tl.bfloat16)
    r584_value = r584_value.to(tl.float32)
    r584_value = tl.broadcast_to(r584_value, [$(16384)])
    tl.store($(candidate_vto_copy_181_ptr) + r584_z, r584_value)
    r585_z = tl.arange($(16384))
    r585_value = tl.load($(candidate_vbmm_6_ptr) + ((((r585_z / $(2048)) % $(8)) * $(4096)) + (((r585_z / $(128)) % $(16)) * $(128)) + (r585_z % $(128)) + $(2048)), dtype=tl.bfloat16)
    r585_value = r585_value.to(tl.float32)
    r585_value = tl.broadcast_to(r585_value, [$(16384)])
    tl.store($(candidate_vto_copy_182_ptr) + r585_z, r585_value)
    r586_row = tl.arange($(16))
    r586_col = tl.arange($(16))
    r586_lane = tl.arange($(192))
    for r586_group in range($(8)) {
      r586_left = tl.load($(candidate_vto_copy_178_ptr) + (((((((r586_group % $(8)) * $(3072)) + (r586_row[:, None] * $(192)) + r586_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r586_group % $(8)) * $(3072)) + (r586_row[:, None] * $(192)) + r586_lane[None, :]) / $(192)) % $(16)) * $(192)) + ((((r586_group % $(8)) * $(3072)) + (r586_row[:, None] * $(192)) + r586_lane[None, :]) % $(192))), dtype=tl.float32)
      r586_right = tl.load($(candidate_vto_copy_179_ptr) + (((((((r586_group % $(8)) * $(3072)) + (r586_lane[:, None] * $(16)) + r586_col[None, :]) / $(3072)) % $(8)) * $(3072)) + (((((r586_group % $(8)) * $(3072)) + (r586_lane[:, None] * $(16)) + r586_col[None, :]) % $(16)) * $(192)) + (((((r586_group % $(8)) * $(3072)) + (r586_lane[:, None] * $(16)) + r586_col[None, :]) / $(16)) % $(192))), dtype=tl.float32)
      r586_value = tl.dot(r586_left, r586_right)
      r586_value = r586_value.to(tl.float32)
      tl.store($(candidate_vbmm_97_ptr) + ((r586_group * $(256)) + (r586_row[:, None] * $(16)) + r586_col[None, :]), r586_value)
    }
    r587_z = tl.arange($(2048))
    r587_value = (tl.load($(candidate_vbmm_97_ptr) + ((((r587_z / $(256)) % $(8)) * $(256)) + (((r587_z / $(16)) % $(16)) * $(16)) + (r587_z % $(16))), dtype=tl.float32) * 0.07216878364870322)
    r587_value = r587_value.to(tl.float32)
    r587_value = tl.broadcast_to(r587_value, [$(2048)])
    tl.store($(candidate_vmul_90_ptr) + r587_z, r587_value)
    r588_z = tl.arange($(16))
    r588_value = (tl.cast(r588_z, tl.int64) + tl.full([], $(16), dtype=tl.int64))
    r588_value = tl.broadcast_to(r588_value, [$(16)])
    tl.store($(candidate_vadd_48_ptr) + r588_z, r588_value)
    r589_z = tl.arange($(16))
    r589_value = (tl.load($(candidate_vadd_48_ptr) + r589_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r589_value = tl.broadcast_to(r589_value, [$(16)])
    tl.store($(candidate_vsub_54_ptr) + r589_z, r589_value)
    r590_z = tl.arange($(256))
    r590_value = (tl.cast((r590_z % $(16)), tl.int64) > tl.load($(candidate_vsub_54_ptr) + ((r590_z / $(16)) % $(16)), dtype=tl.int64))
    r590_value = tl.broadcast_to(r590_value, [$(256)])
    tl.store($(candidate_vgt_23_ptr) + r590_z, r590_value)
    r591_z = tl.arange($(2048))
    r591_value = tl.where(tl.load($(candidate_vgt_23_ptr) + ((((r591_z / $(16)) % $(16)) * $(16)) + (r591_z % $(16))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_90_ptr) + ((((r591_z / $(256)) % $(8)) * $(256)) + (((r591_z / $(16)) % $(16)) * $(16)) + (r591_z % $(16))), dtype=tl.float32))
    r591_value = r591_value.to(tl.float32)
    r591_value = tl.broadcast_to(r591_value, [$(2048)])
    tl.store($(candidate_vmasked_fill_23_ptr) + r591_z, r591_value)
    r592_z = tl.arange($(2048))
    r592_value = (tl.load($(candidate_vmasked_fill_23_ptr) + ((((r592_z / $(256)) % $(8)) * $(256)) + (((r592_z / $(16)) % $(16)) * $(16)) + (r592_z % $(16))), dtype=tl.float32) - tl.load($(candidate_vcat_7_ptr) + ((((r592_z / $(16)) % $(16)) * $(8)) + ((r592_z / $(256)) % $(8)) + $(128)), dtype=tl.float32))
    r592_value = r592_value.to(tl.float32)
    r592_value = tl.broadcast_to(r592_value, [$(2048)])
    tl.store($(candidate_vsub_55_ptr) + r592_z, r592_value)
    r593_z = tl.arange($(2048))
    r593_value = tl.exp(tl.load($(candidate_vsub_55_ptr) + ((((r593_z / $(256)) % $(8)) * $(256)) + (((r593_z / $(16)) % $(16)) * $(16)) + (r593_z % $(16))), dtype=tl.float32))
    r593_value = r593_value.to(tl.float32)
    r593_value = tl.broadcast_to(r593_value, [$(2048)])
    tl.store($(candidate_vexp_11_ptr) + r593_z, r593_value)
    r594_z = tl.arange($(16384))
    r594_value = (tl.load($(candidate_vto_copy_181_ptr) + ((((r594_z / $(2048)) % $(8)) * $(2048)) + (((r594_z / $(128)) % $(16)) * $(128)) + (r594_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_182_ptr) + ((((r594_z / $(2048)) % $(8)) * $(2048)) + (((r594_z / $(128)) % $(16)) * $(128)) + (r594_z % $(128))), dtype=tl.float32))
    r594_value = r594_value.to(tl.float32)
    r594_value = tl.broadcast_to(r594_value, [$(16384)])
    tl.store($(candidate_vmul_91_ptr) + r594_z, r594_value)
    r595_z = tl.arange($(128))
    r595_lane = tl.arange($(128))
    r595_value = tl.sum(tl.load($(candidate_vmul_91_ptr) + ((((r595_z[:, None] / $(16)) % $(8)) * $(2048)) + ((r595_z[:, None] % $(16)) * $(128)) + r595_lane[None, :]), dtype=tl.float32), axis=1)
    r595_value = r595_value.to(tl.float32)
    r595_value = tl.broadcast_to(r595_value, [$(128)])
    tl.store($(candidate_vsum_15_ptr) + r595_z, r595_value)
    r596_row = tl.arange($(16))
    r596_col = tl.arange($(16))
    r596_lane = tl.arange($(128))
    for r596_group in range($(8)) {
      r596_left = tl.load($(candidate_vto_copy_181_ptr) + (((((((r596_group % $(8)) * $(2048)) + (r596_row[:, None] * $(128)) + r596_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r596_group % $(8)) * $(2048)) + (r596_row[:, None] * $(128)) + r596_lane[None, :]) / $(128)) % $(16)) * $(128)) + ((((r596_group % $(8)) * $(2048)) + (r596_row[:, None] * $(128)) + r596_lane[None, :]) % $(128))), dtype=tl.float32)
      r596_right = tl.load($(candidate_vto_copy_180_ptr) + (((((((r596_group % $(8)) * $(2048)) + (r596_lane[:, None] * $(16)) + r596_col[None, :]) / $(2048)) % $(8)) * $(2048)) + (((((r596_group % $(8)) * $(2048)) + (r596_lane[:, None] * $(16)) + r596_col[None, :]) % $(16)) * $(128)) + (((((r596_group % $(8)) * $(2048)) + (r596_lane[:, None] * $(16)) + r596_col[None, :]) / $(16)) % $(128))), dtype=tl.float32)
      r596_value = tl.dot(r596_left, r596_right)
      r596_value = r596_value.to(tl.float32)
      tl.store($(candidate_vbmm_98_ptr) + ((r596_group * $(256)) + (r596_row[:, None] * $(16)) + r596_col[None, :]), r596_value)
    }
    r597_z = tl.arange($(2048))
    r597_value = (tl.load($(candidate_vbmm_98_ptr) + ((((r597_z / $(256)) % $(8)) * $(256)) + (((r597_z / $(16)) % $(16)) * $(16)) + (r597_z % $(16))), dtype=tl.float32) - tl.load($(candidate_vsum_15_ptr) + ((((r597_z / $(256)) % $(8)) * $(16)) + ((r597_z / $(16)) % $(16))), dtype=tl.float32))
    r597_value = r597_value.to(tl.float32)
    r597_value = tl.broadcast_to(r597_value, [$(2048)])
    tl.store($(candidate_vsub_56_ptr) + r597_z, r597_value)
    r598_z = tl.arange($(2048))
    r598_value = (tl.load($(candidate_vexp_11_ptr) + ((((r598_z / $(256)) % $(8)) * $(256)) + (((r598_z / $(16)) % $(16)) * $(16)) + (r598_z % $(16))), dtype=tl.float32) * tl.load($(candidate_vsub_56_ptr) + ((((r598_z / $(256)) % $(8)) * $(256)) + (((r598_z / $(16)) % $(16)) * $(16)) + (r598_z % $(16))), dtype=tl.float32))
    r598_value = r598_value.to(tl.float32)
    r598_value = tl.broadcast_to(r598_value, [$(2048)])
    tl.store($(candidate_vmul_92_ptr) + r598_z, r598_value)
    r599_row = tl.arange($(16))
    r599_col = tl.arange($(192))
    r599_lane = tl.arange($(16))
    for r599_group in range($(8)) {
      r599_left = tl.load($(candidate_vmul_92_ptr) + (((((((r599_group % $(8)) * $(256)) + (r599_row[:, None] * $(16)) + r599_lane[None, :]) / $(256)) % $(8)) * $(256)) + (((((r599_group % $(8)) * $(256)) + (r599_row[:, None] * $(16)) + r599_lane[None, :]) % $(16)) * $(16)) + (((((r599_group % $(8)) * $(256)) + (r599_row[:, None] * $(16)) + r599_lane[None, :]) / $(16)) % $(16))), dtype=tl.float32)
      r599_right = tl.load($(candidate_vto_copy_178_ptr) + (((((((r599_group % $(8)) * $(3072)) + (r599_lane[:, None] * $(192)) + r599_col[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r599_group % $(8)) * $(3072)) + (r599_lane[:, None] * $(192)) + r599_col[None, :]) / $(192)) % $(16)) * $(192)) + ((((r599_group % $(8)) * $(3072)) + (r599_lane[:, None] * $(192)) + r599_col[None, :]) % $(192))), dtype=tl.float32)
      r599_value = tl.dot(r599_left, r599_right)
      r599_value = r599_value.to(tl.float32)
      tl.store($(candidate_vbmm_100_ptr) + ((r599_group * $(3072)) + (r599_row[:, None] * $(192)) + r599_col[None, :]), r599_value)
    }
    r600_z = tl.arange($(24576))
    r600_value = (tl.load($(candidate_vbmm_100_ptr) + ((((r600_z / $(3072)) % $(8)) * $(3072)) + (((r600_z / $(192)) % $(16)) * $(192)) + (r600_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r600_value = r600_value.to(tl.float32)
    r600_value = tl.broadcast_to(r600_value, [$(24576)])
    tl.store($(candidate_vmul_94_ptr) + r600_z, r600_value)
    r601_z = tl.arange($(24576))
    r601_value = tl.load($(candidate_vmul_94_ptr) + ((((r601_z / $(192)) % $(8)) * $(3072)) + (((r601_z / $(1536)) % $(16)) * $(192)) + (r601_z % $(192))), dtype=tl.float32)
    r601_value = r601_value.to(tl.bfloat16)
    r601_value = tl.broadcast_to(r601_value, [$(24576)])
    tl.store($(candidate_vto_copy_184_ptr) + r601_z, r601_value)
    r602_row = tl.arange($(16))
    r602_col = tl.arange($(128))
    r602_lane = tl.arange($(16))
    for r602_group in range($(8)) {
      r602_left = tl.load($(candidate_vexp_11_ptr) + (((((((r602_group % $(8)) * $(256)) + (r602_row[:, None] * $(16)) + r602_lane[None, :]) / $(256)) % $(8)) * $(256)) + (((((r602_group % $(8)) * $(256)) + (r602_row[:, None] * $(16)) + r602_lane[None, :]) % $(16)) * $(16)) + (((((r602_group % $(8)) * $(256)) + (r602_row[:, None] * $(16)) + r602_lane[None, :]) / $(16)) % $(16))), dtype=tl.float32)
      r602_right = tl.load($(candidate_vto_copy_181_ptr) + (((((((r602_group % $(8)) * $(2048)) + (r602_lane[:, None] * $(128)) + r602_col[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r602_group % $(8)) * $(2048)) + (r602_lane[:, None] * $(128)) + r602_col[None, :]) / $(128)) % $(16)) * $(128)) + ((((r602_group % $(8)) * $(2048)) + (r602_lane[:, None] * $(128)) + r602_col[None, :]) % $(128))), dtype=tl.float32)
      r602_value = tl.dot(r602_left, r602_right)
      r602_value = r602_value.to(tl.float32)
      tl.store($(candidate_vbmm_101_ptr) + ((r602_group * $(2048)) + (r602_row[:, None] * $(128)) + r602_col[None, :]), r602_value)
    }
    r603_z = tl.arange($(16384))
    r603_value = tl.load($(candidate_vbmm_101_ptr) + ((((r603_z / $(128)) % $(8)) * $(2048)) + (((r603_z / $(1024)) % $(16)) * $(128)) + (r603_z % $(128))), dtype=tl.float32)
    r603_value = r603_value.to(tl.bfloat16)
    r603_value = tl.broadcast_to(r603_value, [$(16384)])
    tl.store($(candidate_vto_copy_185_ptr) + r603_z, r603_value)
    r604_z = tl.arange($(98304))
    r604_value = tl.where(((r604_z / $(1536)) % $(64)) < $(16), tl.load($(candidate_vto_copy_184_ptr) + ((((((r604_z / $(1536)) % $(64)) - $(0)) % $(16)) * $(1536)) + (((r604_z / $(192)) % $(8)) * $(192)) + (r604_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r604_value = r604_value.to(tl.bfloat16)
    r604_value = tl.broadcast_to(r604_value, [$(98304)])
    tl.store($(candidate_vcat_51_ptr) + r604_z, r604_value)
    r605_z = tl.arange($(65536))
    r605_value = tl.where(((r605_z / $(1024)) % $(64)) < $(16), tl.load($(candidate_vto_copy_185_ptr) + ((((((r605_z / $(1024)) % $(64)) - $(0)) % $(16)) * $(1024)) + (((r605_z / $(128)) % $(8)) * $(128)) + (r605_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r605_value = r605_value.to(tl.bfloat16)
    r605_value = tl.broadcast_to(r605_value, [$(65536)])
    tl.store($(candidate_vcat_52_ptr) + r605_z, r605_value)
    r606_z = tl.arange($(196608))
    r606_value = tl.where(((r606_z / $(1536)) % $(128)) < $(48), tl.load($(candidate_vcat_47_ptr) + ((((((r606_z / $(1536)) % $(128)) - $(0)) % $(48)) * $(1536)) + (((r606_z / $(192)) % $(8)) * $(192)) + (r606_z % $(192))), dtype=tl.bfloat16), tl.where(((r606_z / $(1536)) % $(128)) < $(64), tl.load($(candidate_vcat_49_ptr) + ((((((r606_z / $(1536)) % $(128)) - $(48)) % $(16)) * $(1536)) + (((r606_z / $(192)) % $(8)) * $(192)) + (r606_z % $(192))), dtype=tl.bfloat16), tl.load($(candidate_vcat_51_ptr) + ((((((r606_z / $(1536)) % $(128)) - $(64)) % $(64)) * $(1536)) + (((r606_z / $(192)) % $(8)) * $(192)) + (r606_z % $(192))), dtype=tl.bfloat16)))
    r606_value = r606_value.to(tl.bfloat16)
    r606_value = tl.broadcast_to(r606_value, [$(196608)])
    tl.store($(candidate_vcat_54_ptr) + r606_z, r606_value)
    r607_z = tl.arange($(131072))
    r607_value = tl.where(((r607_z / $(1024)) % $(128)) < $(48), tl.load($(candidate_vcat_48_ptr) + ((((((r607_z / $(1024)) % $(128)) - $(0)) % $(48)) * $(1024)) + (((r607_z / $(128)) % $(8)) * $(128)) + (r607_z % $(128))), dtype=tl.bfloat16), tl.where(((r607_z / $(1024)) % $(128)) < $(64), tl.load($(candidate_vcat_50_ptr) + ((((((r607_z / $(1024)) % $(128)) - $(48)) % $(16)) * $(1024)) + (((r607_z / $(128)) % $(8)) * $(128)) + (r607_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_52_ptr) + ((((((r607_z / $(1024)) % $(128)) - $(64)) % $(64)) * $(1024)) + (((r607_z / $(128)) % $(8)) * $(128)) + (r607_z % $(128))), dtype=tl.bfloat16)))
    r607_value = r607_value.to(tl.bfloat16)
    r607_value = tl.broadcast_to(r607_value, [$(131072)])
    tl.store($(candidate_vcat_55_ptr) + r607_z, r607_value)
    r608_row = tl.arange($(1024))
    r608_col = tl.arange($(512))
    r608_lane = tl.arange($(128))
    for r608_group in range($(1)) {
      r608_left = tl.load($(candidate_vcat_54_ptr) + (((((r608_row[:, None] * $(128)) + r608_lane[None, :]) % $(128)) * $(1536)) + (((((r608_row[:, None] * $(128)) + r608_lane[None, :]) / $(16384)) % $(8)) * $(192)) + ((((r608_row[:, None] * $(128)) + r608_lane[None, :]) / $(128)) % $(128))), dtype=tl.bfloat16)
      r608_right = tl.load($(candidate_vindex_2_ptr) + ((((((r608_lane[:, None] * $(512)) + r608_col[None, :]) / $(512)) % $(128)) * $(576)) + (((r608_lane[:, None] * $(512)) + r608_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r608_value = tl.dot(r608_left, r608_right)
      r608_value = r608_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_104_ptr) + ((r608_group * $(524288)) + (r608_row[:, None] * $(512)) + r608_col[None, :]), r608_value)
    }
    r609_row = tl.arange($(1024))
    r609_col = tl.arange($(512))
    r609_lane = tl.arange($(128))
    for r609_group in range($(1)) {
      r609_left = tl.load($(candidate_vcat_55_ptr) + (((((r609_row[:, None] * $(128)) + r609_lane[None, :]) % $(128)) * $(1024)) + (((((r609_row[:, None] * $(128)) + r609_lane[None, :]) / $(16384)) % $(8)) * $(128)) + ((((r609_row[:, None] * $(128)) + r609_lane[None, :]) / $(128)) % $(128))), dtype=tl.bfloat16)
      r609_right = tl.load($(candidate_vindex_2_ptr) + ((((((r609_lane[:, None] * $(512)) + r609_col[None, :]) / $(512)) % $(128)) * $(576)) + (((r609_lane[:, None] * $(512)) + r609_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r609_value = tl.dot(r609_left, r609_right)
      r609_value = r609_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_105_ptr) + ((r609_group * $(524288)) + (r609_row[:, None] * $(512)) + r609_col[None, :]), r609_value)
    }
    r610_z = tl.arange($(1048576))
    r610_value = tl.where(((r610_z / $(512)) % $(256)) < $(128), tl.load($(candidate_vbmm_104_ptr) + ((((((((r610_z / $(131072)) % $(8)) * $(65536)) + ((((r610_z / $(512)) % $(256)) - $(0)) * $(512)) + (r610_z % $(512))) / $(65536)) % $(8)) * $(65536)) + ((((((r610_z / $(131072)) % $(8)) * $(128)) + (((r610_z / $(512)) % $(256)) - $(0))) % $(128)) * $(512)) + (r610_z % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vbmm_105_ptr) + ((((((((r610_z / $(131072)) % $(8)) * $(65536)) + ((((r610_z / $(512)) % $(256)) - $(128)) * $(512)) + (r610_z % $(512))) / $(65536)) % $(8)) * $(65536)) + ((((((r610_z / $(131072)) % $(8)) * $(128)) + (((r610_z / $(512)) % $(256)) - $(128))) % $(128)) * $(512)) + (r610_z % $(512))), dtype=tl.bfloat16))
    r610_value = r610_value.to(tl.bfloat16)
    r610_value = tl.broadcast_to(r610_value, [$(1048576)])
    tl.store($(candidate_vcat_56_ptr) + r610_z, r610_value)
    r611_z = tl.arange($(1048576))
    r611_value = (tl.load($(candidate_vadd_45_ptr) + ((((r611_z / $(512)) % $(2048)) * $(512)) + (r611_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vcat_56_ptr) + ((((((((r611_z / $(512)) % $(2048)) * $(512)) + (r611_z % $(512))) / $(131072)) % $(8)) * $(131072)) + ((((r611_z / $(512)) % $(2048)) % $(256)) * $(512)) + (r611_z % $(512))), dtype=tl.bfloat16))
    r611_value = r611_value.to(tl.bfloat16)
    r611_value = tl.broadcast_to(r611_value, [$(1048576)])
    tl.store($(candidate_vadd_50_ptr) + r611_z, r611_value)
    z = tl.arange($(1048576))
    value = tl.load($(candidate_vadd_50_ptr) + ((((z / $(512)) % $(2048)) * $(512)) + (z % $(512))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(1048576)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_mla_fa4_te_rope_cp_absorb_reuse_wgrad
