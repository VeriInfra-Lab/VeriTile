import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_mla_fa4_te_rope_thd_absorb_reuse_fwd
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
  let vbmm_3_ptr : RegionName := "vbmm_3"
  let vmm_4_ptr : RegionName := "vmm_4"
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
    r83_col = tl.arange($(128))
    r83_lane = tl.arange($(128))
    for r83_group in range($(8)) {
      r83_left = tl.load($(vto_copy_25_ptr) + (((((((r83_group % $(8)) * $(16384)) + (r83_row[:, None] * $(128)) + r83_lane[None, :]) / $(16384)) % $(8)) * $(16384)) + ((((((r83_group % $(8)) * $(16384)) + (r83_row[:, None] * $(128)) + r83_lane[None, :]) / $(128)) % $(128)) * $(128)) + ((((r83_group % $(8)) * $(16384)) + (r83_row[:, None] * $(128)) + r83_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r83_right = tl.load($(vmm_3_ptr) + (((((((r83_group % $(8)) * $(16384)) + (r83_lane[:, None] * $(128)) + r83_col[None, :]) / $(128)) % $(128)) * $(2048)) + ((((((r83_group % $(8)) * $(16384)) + (r83_lane[:, None] * $(128)) + r83_col[None, :]) / $(16384)) % $(8)) * $(256)) + ((((r83_group % $(8)) * $(16384)) + (r83_lane[:, None] * $(128)) + r83_col[None, :]) % $(128)) + $(128)), dtype=tl.bfloat16)
      r83_value = tl.dot(r83_left, r83_right)
      r83_value = r83_value.to(tl.bfloat16)
      tl.store($(vbmm_3_ptr) + ((r83_group * $(16384)) + (r83_row[:, None] * $(128)) + r83_col[None, :]), r83_value)
    }
    r84_row = tl.arange($(128))
    r84_col = tl.arange($(512))
    r84_lane = tl.arange($(1024))
    r84_left = tl.load($(vbmm_3_ptr) + ((((((((((r84_row[:, None] * $(1024)) + r84_lane[None, :]) / $(1024)) % $(128)) * $(1024)) + (((r84_row[:, None] * $(1024)) + r84_lane[None, :]) % $(1024))) / $(128)) % $(8)) * $(16384)) + (((((r84_row[:, None] * $(1024)) + r84_lane[None, :]) / $(1024)) % $(128)) * $(128)) + (((((((r84_row[:, None] * $(1024)) + r84_lane[None, :]) / $(1024)) % $(128)) * $(1024)) + (((r84_row[:, None] * $(1024)) + r84_lane[None, :]) % $(1024))) % $(128))), dtype=tl.bfloat16)
    r84_right = tl.load($(vto_copy_6_ptr) + (((r84_col[None, :] % $(512)) * $(1024)) + (r84_lane[:, None] % $(1024))), dtype=tl.bfloat16)
    r84_value = tl.dot(r84_left, r84_right)
    r84_value = r84_value.to(tl.bfloat16)
    tl.store($(vmm_4_ptr) + ((r84_row[:, None] * $(512)) + r84_col[None, :]), r84_value)
    z = tl.arange($(65536))
    value = tl.load($(vmm_4_ptr) + ((((z / $(512)) % $(128)) * $(512)) + (z % $(512))), dtype=tl.bfloat16)
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
  let vbmm_2_ptr : RegionName := "vbmm_2"
  let vmul_25_ptr : RegionName := "vmul_25"
  let vadd_7_ptr : RegionName := "vadd_7"
  let vto_copy_24_ptr : RegionName := "vto_copy_24"
  let vsoftmax_ptr : RegionName := "vsoftmax"
  let vto_copy_25_ptr : RegionName := "vto_copy_25"
  let vbmm_3_ptr : RegionName := "vbmm_3"
  let vmm_4_ptr : RegionName := "vmm_4"
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
  let candidate_vto_copy_28_ptr : RegionName := "candidate_vto_copy_28"
  let candidate_vto_copy_29_ptr : RegionName := "candidate_vto_copy_29"
  let candidate_vbmm_3_ptr : RegionName := "candidate_vbmm_3"
  let candidate_vto_copy_30_ptr : RegionName := "candidate_vto_copy_30"
  let candidate_vto_copy_31_ptr : RegionName := "candidate_vto_copy_31"
  let candidate_vmm_3_ptr : RegionName := "candidate_vmm_3"
  let candidate_vadd_7_ptr : RegionName := "candidate_vadd_7"
  let candidate_vmul_23_ptr : RegionName := "candidate_vmul_23"
  let candidate_vadd_8_ptr : RegionName := "candidate_vadd_8"
  let candidate_vsub_4_ptr : RegionName := "candidate_vsub_4"
  let candidate_vgt_ptr : RegionName := "candidate_vgt"
  let candidate_vmasked_fill_ptr : RegionName := "candidate_vmasked_fill"
  let candidate_vsoftmax_ptr : RegionName := "candidate_vsoftmax"
  let candidate_vto_copy_32_ptr : RegionName := "candidate_vto_copy_32"
  let candidate_vmm_4_ptr : RegionName := "candidate_vmm_4"
  let candidate_vto_copy_33_ptr : RegionName := "candidate_vto_copy_33"
  let candidate_vto_copy_34_ptr : RegionName := "candidate_vto_copy_34"
  let candidate_vbmm_4_ptr : RegionName := "candidate_vbmm_4"
  let candidate_vto_copy_35_ptr : RegionName := "candidate_vto_copy_35"
  let candidate_vto_copy_36_ptr : RegionName := "candidate_vto_copy_36"
  let candidate_vmm_5_ptr : RegionName := "candidate_vmm_5"
  let candidate_vadd_9_ptr : RegionName := "candidate_vadd_9"
  let candidate_vmul_24_ptr : RegionName := "candidate_vmul_24"
  let candidate_vadd_10_ptr : RegionName := "candidate_vadd_10"
  let candidate_vsub_5_ptr : RegionName := "candidate_vsub_5"
  let candidate_vgt_1_ptr : RegionName := "candidate_vgt_1"
  let candidate_vmasked_fill_1_ptr : RegionName := "candidate_vmasked_fill_1"
  let candidate_vsoftmax_1_ptr : RegionName := "candidate_vsoftmax_1"
  let candidate_vto_copy_37_ptr : RegionName := "candidate_vto_copy_37"
  let candidate_vmm_6_ptr : RegionName := "candidate_vmm_6"
  let candidate_vto_copy_38_ptr : RegionName := "candidate_vto_copy_38"
  let candidate_vto_copy_39_ptr : RegionName := "candidate_vto_copy_39"
  let candidate_vbmm_5_ptr : RegionName := "candidate_vbmm_5"
  let candidate_vto_copy_40_ptr : RegionName := "candidate_vto_copy_40"
  let candidate_vto_copy_41_ptr : RegionName := "candidate_vto_copy_41"
  let candidate_vmm_7_ptr : RegionName := "candidate_vmm_7"
  let candidate_vadd_11_ptr : RegionName := "candidate_vadd_11"
  let candidate_vmul_25_ptr : RegionName := "candidate_vmul_25"
  let candidate_vadd_12_ptr : RegionName := "candidate_vadd_12"
  let candidate_vsub_6_ptr : RegionName := "candidate_vsub_6"
  let candidate_vgt_2_ptr : RegionName := "candidate_vgt_2"
  let candidate_vmasked_fill_2_ptr : RegionName := "candidate_vmasked_fill_2"
  let candidate_vsoftmax_2_ptr : RegionName := "candidate_vsoftmax_2"
  let candidate_vto_copy_42_ptr : RegionName := "candidate_vto_copy_42"
  let candidate_vmm_8_ptr : RegionName := "candidate_vmm_8"
  let candidate_vcat_7_ptr : RegionName := "candidate_vcat_7"
  let candidate_vbmm_6_ptr : RegionName := "candidate_vbmm_6"
  let candidate_vmm_9_ptr : RegionName := "candidate_vmm_9"
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
    r60_col = tl.arange($(512))
    r60_lane = tl.arange($(128))
    for r60_group in range($(8)) {
      r60_left = tl.load($(candidate_vmm_1_ptr) + (((((((r60_group % $(8)) * $(16384)) + (r60_row[:, None] * $(128)) + r60_lane[None, :]) / $(128)) % $(128)) * $(1536)) + ((((((r60_group % $(8)) * $(16384)) + (r60_row[:, None] * $(128)) + r60_lane[None, :]) / $(16384)) % $(8)) * $(192)) + ((((r60_group % $(8)) * $(16384)) + (r60_row[:, None] * $(128)) + r60_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r60_right = tl.load($(candidate_vto_copy_5_ptr) + (((((((r60_group % $(8)) * $(65536)) + (r60_lane[:, None] * $(512)) + r60_col[None, :]) / $(65536)) % $(8)) * $(131072)) + ((((((r60_group % $(8)) * $(65536)) + (r60_lane[:, None] * $(512)) + r60_col[None, :]) / $(512)) % $(128)) * $(512)) + ((((r60_group % $(8)) * $(65536)) + (r60_lane[:, None] * $(512)) + r60_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r60_value = tl.dot(r60_left, r60_right)
      r60_value = r60_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_2_ptr) + ((r60_group * $(65536)) + (r60_row[:, None] * $(512)) + r60_col[None, :]), r60_value)
    }
    r61_z = tl.arange($(24576))
    r61_value = tl.load($(candidate_vadd_5_ptr) + ((((r61_z / $(64)) % $(48)) * $(512)) + (((r61_z / $(3072)) % $(8)) * $(64)) + (r61_z % $(64))), dtype=tl.bfloat16)
    r61_value = r61_value.to(tl.float32)
    r61_value = tl.broadcast_to(r61_value, [$(24576)])
    tl.store($(candidate_vto_copy_28_ptr) + r61_z, r61_value)
    r62_z = tl.arange($(3072))
    r62_value = tl.load($(candidate_vadd_6_ptr) + ((((r62_z / $(64)) % $(48)) * $(64)) + (r62_z % $(64))), dtype=tl.bfloat16)
    r62_value = r62_value.to(tl.float32)
    r62_value = tl.broadcast_to(r62_value, [$(3072)])
    tl.store($(candidate_vto_copy_29_ptr) + r62_z, r62_value)
    r63_row = tl.arange($(48))
    r63_col = tl.arange($(48))
    r63_lane = tl.arange($(64))
    for r63_group in range($(8)) {
      r63_left = tl.load($(candidate_vto_copy_28_ptr) + (((((((r63_group % $(8)) * $(3072)) + (r63_row[:, None] * $(64)) + r63_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r63_group % $(8)) * $(3072)) + (r63_row[:, None] * $(64)) + r63_lane[None, :]) / $(64)) % $(48)) * $(64)) + ((((r63_group % $(8)) * $(3072)) + (r63_row[:, None] * $(64)) + r63_lane[None, :]) % $(64))), dtype=tl.float32)
      r63_right = tl.load($(candidate_vto_copy_29_ptr) + ((((((r63_group % $(8)) * $(3072)) + (r63_lane[:, None] * $(48)) + r63_col[None, :]) % $(48)) * $(64)) + (((((r63_group % $(8)) * $(3072)) + (r63_lane[:, None] * $(48)) + r63_col[None, :]) / $(48)) % $(64))), dtype=tl.float32)
      r63_value = tl.dot(r63_left, r63_right)
      r63_value = r63_value.to(tl.float32)
      tl.store($(candidate_vbmm_3_ptr) + ((r63_group * $(2304)) + (r63_row[:, None] * $(48)) + r63_col[None, :]), r63_value)
    }
    r64_z = tl.arange($(196608))
    r64_value = tl.load($(candidate_vbmm_2_ptr) + ((((r64_z / $(24576)) % $(8)) * $(65536)) + (((r64_z / $(512)) % $(48)) * $(512)) + (r64_z % $(512))), dtype=tl.bfloat16)
    r64_value = r64_value.to(tl.float32)
    r64_value = tl.broadcast_to(r64_value, [$(196608)])
    tl.store($(candidate_vto_copy_30_ptr) + r64_z, r64_value)
    r65_z = tl.arange($(24576))
    r65_value = tl.load($(candidate_vto_copy_23_ptr) + ((((r65_z / $(512)) % $(48)) * $(512)) + (r65_z % $(512))), dtype=tl.bfloat16)
    r65_value = r65_value.to(tl.float32)
    r65_value = tl.broadcast_to(r65_value, [$(24576)])
    tl.store($(candidate_vto_copy_31_ptr) + r65_z, r65_value)
    r66_row = tl.arange($(384))
    r66_col = tl.arange($(48))
    r66_lane = tl.arange($(512))
    r66_left = tl.load($(candidate_vto_copy_30_ptr) + ((((((r66_row[:, None] * $(512)) + r66_lane[None, :]) / $(24576)) % $(8)) * $(24576)) + (((((r66_row[:, None] * $(512)) + r66_lane[None, :]) / $(512)) % $(48)) * $(512)) + (((r66_row[:, None] * $(512)) + r66_lane[None, :]) % $(512))), dtype=tl.float32)
    r66_right = tl.load($(candidate_vto_copy_31_ptr) + (((((r66_lane[:, None] * $(48)) + r66_col[None, :]) % $(48)) * $(512)) + ((((r66_lane[:, None] * $(48)) + r66_col[None, :]) / $(48)) % $(512))), dtype=tl.float32)
    r66_value = tl.dot(r66_left, r66_right)
    r66_value = r66_value.to(tl.float32)
    tl.store($(candidate_vmm_3_ptr) + ((r66_row[:, None] * $(48)) + r66_col[None, :]), r66_value)
    r67_z = tl.arange($(18432))
    r67_value = (tl.load($(candidate_vbmm_3_ptr) + ((((r67_z / $(2304)) % $(8)) * $(2304)) + (((r67_z / $(48)) % $(48)) * $(48)) + (r67_z % $(48))), dtype=tl.float32) + tl.load($(candidate_vmm_3_ptr) + ((((r67_z / $(2304)) % $(8)) * $(2304)) + (((r67_z / $(48)) % $(48)) * $(48)) + (r67_z % $(48))), dtype=tl.float32))
    r67_value = r67_value.to(tl.float32)
    r67_value = tl.broadcast_to(r67_value, [$(18432)])
    tl.store($(candidate_vadd_7_ptr) + r67_z, r67_value)
    r68_z = tl.arange($(18432))
    r68_value = (tl.load($(candidate_vadd_7_ptr) + ((((r68_z / $(2304)) % $(8)) * $(2304)) + (((r68_z / $(48)) % $(48)) * $(48)) + (r68_z % $(48))), dtype=tl.float32) * 0.07216878364870322)
    r68_value = r68_value.to(tl.float32)
    r68_value = tl.broadcast_to(r68_value, [$(18432)])
    tl.store($(candidate_vmul_23_ptr) + r68_z, r68_value)
    r69_z = tl.arange($(48))
    r69_value = (tl.cast(r69_z, tl.int64) + tl.full([], $(48), dtype=tl.int64))
    r69_value = tl.broadcast_to(r69_value, [$(48)])
    tl.store($(candidate_vadd_8_ptr) + r69_z, r69_value)
    r70_z = tl.arange($(48))
    r70_value = (tl.load($(candidate_vadd_8_ptr) + r70_z, dtype=tl.int64) - tl.full([], $(48), dtype=tl.int64))
    r70_value = tl.broadcast_to(r70_value, [$(48)])
    tl.store($(candidate_vsub_4_ptr) + r70_z, r70_value)
    r71_z = tl.arange($(2304))
    r71_value = (tl.cast((r71_z % $(48)), tl.int64) > tl.load($(candidate_vsub_4_ptr) + ((r71_z / $(48)) % $(48)), dtype=tl.int64))
    r71_value = tl.broadcast_to(r71_value, [$(2304)])
    tl.store($(candidate_vgt_ptr) + r71_z, r71_value)
    r72_z = tl.arange($(18432))
    r72_value = tl.where(tl.load($(candidate_vgt_ptr) + ((((r72_z / $(48)) % $(48)) * $(48)) + (r72_z % $(48))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_23_ptr) + ((((r72_z / $(2304)) % $(8)) * $(2304)) + (((r72_z / $(48)) % $(48)) * $(48)) + (r72_z % $(48))), dtype=tl.float32))
    r72_value = r72_value.to(tl.float32)
    r72_value = tl.broadcast_to(r72_value, [$(18432)])
    tl.store($(candidate_vmasked_fill_ptr) + r72_z, r72_value)
    r73_z = tl.arange($(18432))
    r73_lane = tl.arange($(48))
    r73_row_values = tl.load($(candidate_vmasked_fill_ptr) + ((((r73_z[:, None] / $(2304)) % $(8)) * $(2304)) + (((r73_z[:, None] / $(48)) % $(48)) * $(48)) + (r73_lane[None, :] % $(48))), dtype=tl.float32)
    r73_maximum = tl.max(r73_row_values, axis=1, keep_dims=true)
    r73_weights = tl.exp(r73_row_values - r73_maximum)
    r73_total = tl.sum(r73_weights, axis=1)
    r73_maximum = tl.max(r73_row_values, axis=1)
    r73_value = tl.exp(tl.load($(candidate_vmasked_fill_ptr) + ((((r73_z / $(2304)) % $(8)) * $(2304)) + (((r73_z / $(48)) % $(48)) * $(48)) + (r73_z % $(48))), dtype=tl.float32) - r73_maximum) / r73_total
    r73_value = r73_value.to(tl.float32)
    r73_value = tl.broadcast_to(r73_value, [$(18432)])
    tl.store($(candidate_vsoftmax_ptr) + r73_z, r73_value)
    r74_z = tl.arange($(18432))
    r74_value = tl.load($(candidate_vsoftmax_ptr) + ((((r74_z / $(2304)) % $(8)) * $(2304)) + (((r74_z / $(48)) % $(48)) * $(48)) + (r74_z % $(48))), dtype=tl.float32)
    r74_value = r74_value.to(tl.bfloat16)
    r74_value = tl.broadcast_to(r74_value, [$(18432)])
    tl.store($(candidate_vto_copy_32_ptr) + r74_z, r74_value)
    r75_row = tl.arange($(384))
    r75_col = tl.arange($(512))
    r75_lane = tl.arange($(48))
    r75_left = tl.load($(candidate_vto_copy_32_ptr) + ((((((r75_row[:, None] * $(48)) + r75_lane[None, :]) / $(2304)) % $(8)) * $(2304)) + (((((r75_row[:, None] * $(48)) + r75_lane[None, :]) / $(48)) % $(48)) * $(48)) + (((r75_row[:, None] * $(48)) + r75_lane[None, :]) % $(48))), dtype=tl.bfloat16)
    r75_right = tl.load($(candidate_vto_copy_23_ptr) + ((((((r75_lane[:, None] * $(512)) + r75_col[None, :]) / $(512)) % $(48)) * $(512)) + (((r75_lane[:, None] * $(512)) + r75_col[None, :]) % $(512))), dtype=tl.bfloat16)
    r75_value = tl.dot(r75_left, r75_right)
    r75_value = r75_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_4_ptr) + ((r75_row[:, None] * $(512)) + r75_col[None, :]), r75_value)
    r76_z = tl.arange($(8192))
    r76_value = tl.load($(candidate_vadd_5_ptr) + ((((r76_z / $(64)) % $(16)) * $(512)) + (((r76_z / $(1024)) % $(8)) * $(64)) + (r76_z % $(64)) + $(24576)), dtype=tl.bfloat16)
    r76_value = r76_value.to(tl.float32)
    r76_value = tl.broadcast_to(r76_value, [$(8192)])
    tl.store($(candidate_vto_copy_33_ptr) + r76_z, r76_value)
    r77_z = tl.arange($(1024))
    r77_value = tl.load($(candidate_vadd_6_ptr) + ((((r77_z / $(64)) % $(16)) * $(64)) + (r77_z % $(64)) + $(3072)), dtype=tl.bfloat16)
    r77_value = r77_value.to(tl.float32)
    r77_value = tl.broadcast_to(r77_value, [$(1024)])
    tl.store($(candidate_vto_copy_34_ptr) + r77_z, r77_value)
    r78_row = tl.arange($(16))
    r78_col = tl.arange($(16))
    r78_lane = tl.arange($(64))
    for r78_group in range($(8)) {
      r78_left = tl.load($(candidate_vto_copy_33_ptr) + (((((((r78_group % $(8)) * $(1024)) + (r78_row[:, None] * $(64)) + r78_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + ((((((r78_group % $(8)) * $(1024)) + (r78_row[:, None] * $(64)) + r78_lane[None, :]) / $(64)) % $(16)) * $(64)) + ((((r78_group % $(8)) * $(1024)) + (r78_row[:, None] * $(64)) + r78_lane[None, :]) % $(64))), dtype=tl.float32)
      r78_right = tl.load($(candidate_vto_copy_34_ptr) + ((((((r78_group % $(8)) * $(1024)) + (r78_lane[:, None] * $(16)) + r78_col[None, :]) % $(16)) * $(64)) + (((((r78_group % $(8)) * $(1024)) + (r78_lane[:, None] * $(16)) + r78_col[None, :]) / $(16)) % $(64))), dtype=tl.float32)
      r78_value = tl.dot(r78_left, r78_right)
      r78_value = r78_value.to(tl.float32)
      tl.store($(candidate_vbmm_4_ptr) + ((r78_group * $(256)) + (r78_row[:, None] * $(16)) + r78_col[None, :]), r78_value)
    }
    r79_z = tl.arange($(65536))
    r79_value = tl.load($(candidate_vbmm_2_ptr) + ((((r79_z / $(8192)) % $(8)) * $(65536)) + (((r79_z / $(512)) % $(16)) * $(512)) + (r79_z % $(512)) + $(24576)), dtype=tl.bfloat16)
    r79_value = r79_value.to(tl.float32)
    r79_value = tl.broadcast_to(r79_value, [$(65536)])
    tl.store($(candidate_vto_copy_35_ptr) + r79_z, r79_value)
    r80_z = tl.arange($(8192))
    r80_value = tl.load($(candidate_vto_copy_23_ptr) + ((((r80_z / $(512)) % $(16)) * $(512)) + (r80_z % $(512)) + $(24576)), dtype=tl.bfloat16)
    r80_value = r80_value.to(tl.float32)
    r80_value = tl.broadcast_to(r80_value, [$(8192)])
    tl.store($(candidate_vto_copy_36_ptr) + r80_z, r80_value)
    r81_row = tl.arange($(128))
    r81_col = tl.arange($(16))
    r81_lane = tl.arange($(512))
    r81_left = tl.load($(candidate_vto_copy_35_ptr) + ((((((r81_row[:, None] * $(512)) + r81_lane[None, :]) / $(8192)) % $(8)) * $(8192)) + (((((r81_row[:, None] * $(512)) + r81_lane[None, :]) / $(512)) % $(16)) * $(512)) + (((r81_row[:, None] * $(512)) + r81_lane[None, :]) % $(512))), dtype=tl.float32)
    r81_right = tl.load($(candidate_vto_copy_36_ptr) + (((((r81_lane[:, None] * $(16)) + r81_col[None, :]) % $(16)) * $(512)) + ((((r81_lane[:, None] * $(16)) + r81_col[None, :]) / $(16)) % $(512))), dtype=tl.float32)
    r81_value = tl.dot(r81_left, r81_right)
    r81_value = r81_value.to(tl.float32)
    tl.store($(candidate_vmm_5_ptr) + ((r81_row[:, None] * $(16)) + r81_col[None, :]), r81_value)
    r82_z = tl.arange($(2048))
    r82_value = (tl.load($(candidate_vbmm_4_ptr) + ((((r82_z / $(256)) % $(8)) * $(256)) + (((r82_z / $(16)) % $(16)) * $(16)) + (r82_z % $(16))), dtype=tl.float32) + tl.load($(candidate_vmm_5_ptr) + ((((r82_z / $(256)) % $(8)) * $(256)) + (((r82_z / $(16)) % $(16)) * $(16)) + (r82_z % $(16))), dtype=tl.float32))
    r82_value = r82_value.to(tl.float32)
    r82_value = tl.broadcast_to(r82_value, [$(2048)])
    tl.store($(candidate_vadd_9_ptr) + r82_z, r82_value)
    r83_z = tl.arange($(2048))
    r83_value = (tl.load($(candidate_vadd_9_ptr) + ((((r83_z / $(256)) % $(8)) * $(256)) + (((r83_z / $(16)) % $(16)) * $(16)) + (r83_z % $(16))), dtype=tl.float32) * 0.07216878364870322)
    r83_value = r83_value.to(tl.float32)
    r83_value = tl.broadcast_to(r83_value, [$(2048)])
    tl.store($(candidate_vmul_24_ptr) + r83_z, r83_value)
    r84_z = tl.arange($(16))
    r84_value = (tl.cast(r84_z, tl.int64) + tl.full([], $(16), dtype=tl.int64))
    r84_value = tl.broadcast_to(r84_value, [$(16)])
    tl.store($(candidate_vadd_10_ptr) + r84_z, r84_value)
    r85_z = tl.arange($(16))
    r85_value = (tl.load($(candidate_vadd_10_ptr) + r85_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r85_value = tl.broadcast_to(r85_value, [$(16)])
    tl.store($(candidate_vsub_5_ptr) + r85_z, r85_value)
    r86_z = tl.arange($(256))
    r86_value = (tl.cast((r86_z % $(16)), tl.int64) > tl.load($(candidate_vsub_5_ptr) + ((r86_z / $(16)) % $(16)), dtype=tl.int64))
    r86_value = tl.broadcast_to(r86_value, [$(256)])
    tl.store($(candidate_vgt_1_ptr) + r86_z, r86_value)
    r87_z = tl.arange($(2048))
    r87_value = tl.where(tl.load($(candidate_vgt_1_ptr) + ((((r87_z / $(16)) % $(16)) * $(16)) + (r87_z % $(16))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_24_ptr) + ((((r87_z / $(256)) % $(8)) * $(256)) + (((r87_z / $(16)) % $(16)) * $(16)) + (r87_z % $(16))), dtype=tl.float32))
    r87_value = r87_value.to(tl.float32)
    r87_value = tl.broadcast_to(r87_value, [$(2048)])
    tl.store($(candidate_vmasked_fill_1_ptr) + r87_z, r87_value)
    r88_z = tl.arange($(2048))
    r88_lane = tl.arange($(16))
    r88_row_values = tl.load($(candidate_vmasked_fill_1_ptr) + ((((r88_z[:, None] / $(256)) % $(8)) * $(256)) + (((r88_z[:, None] / $(16)) % $(16)) * $(16)) + (r88_lane[None, :] % $(16))), dtype=tl.float32)
    r88_maximum = tl.max(r88_row_values, axis=1, keep_dims=true)
    r88_weights = tl.exp(r88_row_values - r88_maximum)
    r88_total = tl.sum(r88_weights, axis=1)
    r88_maximum = tl.max(r88_row_values, axis=1)
    r88_value = tl.exp(tl.load($(candidate_vmasked_fill_1_ptr) + ((((r88_z / $(256)) % $(8)) * $(256)) + (((r88_z / $(16)) % $(16)) * $(16)) + (r88_z % $(16))), dtype=tl.float32) - r88_maximum) / r88_total
    r88_value = r88_value.to(tl.float32)
    r88_value = tl.broadcast_to(r88_value, [$(2048)])
    tl.store($(candidate_vsoftmax_1_ptr) + r88_z, r88_value)
    r89_z = tl.arange($(2048))
    r89_value = tl.load($(candidate_vsoftmax_1_ptr) + ((((r89_z / $(256)) % $(8)) * $(256)) + (((r89_z / $(16)) % $(16)) * $(16)) + (r89_z % $(16))), dtype=tl.float32)
    r89_value = r89_value.to(tl.bfloat16)
    r89_value = tl.broadcast_to(r89_value, [$(2048)])
    tl.store($(candidate_vto_copy_37_ptr) + r89_z, r89_value)
    r90_row = tl.arange($(128))
    r90_col = tl.arange($(512))
    r90_lane = tl.arange($(16))
    r90_left = tl.load($(candidate_vto_copy_37_ptr) + ((((((r90_row[:, None] * $(16)) + r90_lane[None, :]) / $(256)) % $(8)) * $(256)) + (((((r90_row[:, None] * $(16)) + r90_lane[None, :]) / $(16)) % $(16)) * $(16)) + (((r90_row[:, None] * $(16)) + r90_lane[None, :]) % $(16))), dtype=tl.bfloat16)
    r90_right = tl.load($(candidate_vto_copy_23_ptr) + ((((((r90_lane[:, None] * $(512)) + r90_col[None, :]) / $(512)) % $(16)) * $(512)) + (((r90_lane[:, None] * $(512)) + r90_col[None, :]) % $(512)) + $(24576)), dtype=tl.bfloat16)
    r90_value = tl.dot(r90_left, r90_right)
    r90_value = r90_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_6_ptr) + ((r90_row[:, None] * $(512)) + r90_col[None, :]), r90_value)
    r91_z = tl.arange($(32768))
    r91_value = tl.load($(candidate_vadd_5_ptr) + ((((r91_z / $(64)) % $(64)) * $(512)) + (((r91_z / $(4096)) % $(8)) * $(64)) + (r91_z % $(64)) + $(32768)), dtype=tl.bfloat16)
    r91_value = r91_value.to(tl.float32)
    r91_value = tl.broadcast_to(r91_value, [$(32768)])
    tl.store($(candidate_vto_copy_38_ptr) + r91_z, r91_value)
    r92_z = tl.arange($(4096))
    r92_value = tl.load($(candidate_vadd_6_ptr) + ((((r92_z / $(64)) % $(64)) * $(64)) + (r92_z % $(64)) + $(4096)), dtype=tl.bfloat16)
    r92_value = r92_value.to(tl.float32)
    r92_value = tl.broadcast_to(r92_value, [$(4096)])
    tl.store($(candidate_vto_copy_39_ptr) + r92_z, r92_value)
    r93_row = tl.arange($(64))
    r93_col = tl.arange($(64))
    r93_lane = tl.arange($(64))
    for r93_group in range($(8)) {
      r93_left = tl.load($(candidate_vto_copy_38_ptr) + (((((((r93_group % $(8)) * $(4096)) + (r93_row[:, None] * $(64)) + r93_lane[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r93_group % $(8)) * $(4096)) + (r93_row[:, None] * $(64)) + r93_lane[None, :]) / $(64)) % $(64)) * $(64)) + ((((r93_group % $(8)) * $(4096)) + (r93_row[:, None] * $(64)) + r93_lane[None, :]) % $(64))), dtype=tl.float32)
      r93_right = tl.load($(candidate_vto_copy_39_ptr) + ((((((r93_group % $(8)) * $(4096)) + (r93_lane[:, None] * $(64)) + r93_col[None, :]) % $(64)) * $(64)) + (((((r93_group % $(8)) * $(4096)) + (r93_lane[:, None] * $(64)) + r93_col[None, :]) / $(64)) % $(64))), dtype=tl.float32)
      r93_value = tl.dot(r93_left, r93_right)
      r93_value = r93_value.to(tl.float32)
      tl.store($(candidate_vbmm_5_ptr) + ((r93_group * $(4096)) + (r93_row[:, None] * $(64)) + r93_col[None, :]), r93_value)
    }
    r94_z = tl.arange($(262144))
    r94_value = tl.load($(candidate_vbmm_2_ptr) + ((((r94_z / $(32768)) % $(8)) * $(65536)) + (((r94_z / $(512)) % $(64)) * $(512)) + (r94_z % $(512)) + $(32768)), dtype=tl.bfloat16)
    r94_value = r94_value.to(tl.float32)
    r94_value = tl.broadcast_to(r94_value, [$(262144)])
    tl.store($(candidate_vto_copy_40_ptr) + r94_z, r94_value)
    r95_z = tl.arange($(32768))
    r95_value = tl.load($(candidate_vto_copy_23_ptr) + ((((r95_z / $(512)) % $(64)) * $(512)) + (r95_z % $(512)) + $(32768)), dtype=tl.bfloat16)
    r95_value = r95_value.to(tl.float32)
    r95_value = tl.broadcast_to(r95_value, [$(32768)])
    tl.store($(candidate_vto_copy_41_ptr) + r95_z, r95_value)
    r96_row = tl.arange($(512))
    r96_col = tl.arange($(64))
    r96_lane = tl.arange($(512))
    r96_left = tl.load($(candidate_vto_copy_40_ptr) + ((((((r96_row[:, None] * $(512)) + r96_lane[None, :]) / $(32768)) % $(8)) * $(32768)) + (((((r96_row[:, None] * $(512)) + r96_lane[None, :]) / $(512)) % $(64)) * $(512)) + (((r96_row[:, None] * $(512)) + r96_lane[None, :]) % $(512))), dtype=tl.float32)
    r96_right = tl.load($(candidate_vto_copy_41_ptr) + (((((r96_lane[:, None] * $(64)) + r96_col[None, :]) % $(64)) * $(512)) + ((((r96_lane[:, None] * $(64)) + r96_col[None, :]) / $(64)) % $(512))), dtype=tl.float32)
    r96_value = tl.dot(r96_left, r96_right)
    r96_value = r96_value.to(tl.float32)
    tl.store($(candidate_vmm_7_ptr) + ((r96_row[:, None] * $(64)) + r96_col[None, :]), r96_value)
    r97_z = tl.arange($(32768))
    r97_value = (tl.load($(candidate_vbmm_5_ptr) + ((((r97_z / $(4096)) % $(8)) * $(4096)) + (((r97_z / $(64)) % $(64)) * $(64)) + (r97_z % $(64))), dtype=tl.float32) + tl.load($(candidate_vmm_7_ptr) + ((((r97_z / $(4096)) % $(8)) * $(4096)) + (((r97_z / $(64)) % $(64)) * $(64)) + (r97_z % $(64))), dtype=tl.float32))
    r97_value = r97_value.to(tl.float32)
    r97_value = tl.broadcast_to(r97_value, [$(32768)])
    tl.store($(candidate_vadd_11_ptr) + r97_z, r97_value)
    r98_z = tl.arange($(32768))
    r98_value = (tl.load($(candidate_vadd_11_ptr) + ((((r98_z / $(4096)) % $(8)) * $(4096)) + (((r98_z / $(64)) % $(64)) * $(64)) + (r98_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r98_value = r98_value.to(tl.float32)
    r98_value = tl.broadcast_to(r98_value, [$(32768)])
    tl.store($(candidate_vmul_25_ptr) + r98_z, r98_value)
    r99_z = tl.arange($(64))
    r99_value = (tl.cast(r99_z, tl.int64) + tl.full([], $(64), dtype=tl.int64))
    r99_value = tl.broadcast_to(r99_value, [$(64)])
    tl.store($(candidate_vadd_12_ptr) + r99_z, r99_value)
    r100_z = tl.arange($(64))
    r100_value = (tl.load($(candidate_vadd_12_ptr) + r100_z, dtype=tl.int64) - tl.full([], $(64), dtype=tl.int64))
    r100_value = tl.broadcast_to(r100_value, [$(64)])
    tl.store($(candidate_vsub_6_ptr) + r100_z, r100_value)
    r101_z = tl.arange($(4096))
    r101_value = (tl.cast((r101_z % $(64)), tl.int64) > tl.load($(candidate_vsub_6_ptr) + ((r101_z / $(64)) % $(64)), dtype=tl.int64))
    r101_value = tl.broadcast_to(r101_value, [$(4096)])
    tl.store($(candidate_vgt_2_ptr) + r101_z, r101_value)
    r102_z = tl.arange($(32768))
    r102_value = tl.where(tl.load($(candidate_vgt_2_ptr) + ((((r102_z / $(64)) % $(64)) * $(64)) + (r102_z % $(64))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_25_ptr) + ((((r102_z / $(4096)) % $(8)) * $(4096)) + (((r102_z / $(64)) % $(64)) * $(64)) + (r102_z % $(64))), dtype=tl.float32))
    r102_value = r102_value.to(tl.float32)
    r102_value = tl.broadcast_to(r102_value, [$(32768)])
    tl.store($(candidate_vmasked_fill_2_ptr) + r102_z, r102_value)
    r103_z = tl.arange($(32768))
    r103_lane = tl.arange($(64))
    r103_row_values = tl.load($(candidate_vmasked_fill_2_ptr) + ((((r103_z[:, None] / $(4096)) % $(8)) * $(4096)) + (((r103_z[:, None] / $(64)) % $(64)) * $(64)) + (r103_lane[None, :] % $(64))), dtype=tl.float32)
    r103_maximum = tl.max(r103_row_values, axis=1, keep_dims=true)
    r103_weights = tl.exp(r103_row_values - r103_maximum)
    r103_total = tl.sum(r103_weights, axis=1)
    r103_maximum = tl.max(r103_row_values, axis=1)
    r103_value = tl.exp(tl.load($(candidate_vmasked_fill_2_ptr) + ((((r103_z / $(4096)) % $(8)) * $(4096)) + (((r103_z / $(64)) % $(64)) * $(64)) + (r103_z % $(64))), dtype=tl.float32) - r103_maximum) / r103_total
    r103_value = r103_value.to(tl.float32)
    r103_value = tl.broadcast_to(r103_value, [$(32768)])
    tl.store($(candidate_vsoftmax_2_ptr) + r103_z, r103_value)
    r104_z = tl.arange($(32768))
    r104_value = tl.load($(candidate_vsoftmax_2_ptr) + ((((r104_z / $(4096)) % $(8)) * $(4096)) + (((r104_z / $(64)) % $(64)) * $(64)) + (r104_z % $(64))), dtype=tl.float32)
    r104_value = r104_value.to(tl.bfloat16)
    r104_value = tl.broadcast_to(r104_value, [$(32768)])
    tl.store($(candidate_vto_copy_42_ptr) + r104_z, r104_value)
    r105_row = tl.arange($(512))
    r105_col = tl.arange($(512))
    r105_lane = tl.arange($(64))
    r105_left = tl.load($(candidate_vto_copy_42_ptr) + ((((((r105_row[:, None] * $(64)) + r105_lane[None, :]) / $(4096)) % $(8)) * $(4096)) + (((((r105_row[:, None] * $(64)) + r105_lane[None, :]) / $(64)) % $(64)) * $(64)) + (((r105_row[:, None] * $(64)) + r105_lane[None, :]) % $(64))), dtype=tl.bfloat16)
    r105_right = tl.load($(candidate_vto_copy_23_ptr) + ((((((r105_lane[:, None] * $(512)) + r105_col[None, :]) / $(512)) % $(64)) * $(512)) + (((r105_lane[:, None] * $(512)) + r105_col[None, :]) % $(512)) + $(32768)), dtype=tl.bfloat16)
    r105_value = tl.dot(r105_left, r105_right)
    r105_value = r105_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_8_ptr) + ((r105_row[:, None] * $(512)) + r105_col[None, :]), r105_value)
    r106_z = tl.arange($(524288))
    r106_value = tl.where(((r106_z / $(4096)) % $(128)) < $(48), tl.load($(candidate_vmm_4_ptr) + (((((((r106_z / $(512)) % $(8)) * $(48)) + (((r106_z / $(4096)) % $(128)) - $(0))) % $(384)) * $(512)) + (r106_z % $(512))), dtype=tl.bfloat16), tl.where(((r106_z / $(4096)) % $(128)) < $(64), tl.load($(candidate_vmm_6_ptr) + (((((((r106_z / $(512)) % $(8)) * $(16)) + (((r106_z / $(4096)) % $(128)) - $(48))) % $(128)) * $(512)) + (r106_z % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vmm_8_ptr) + (((((((r106_z / $(512)) % $(8)) * $(64)) + (((r106_z / $(4096)) % $(128)) - $(64))) % $(512)) * $(512)) + (r106_z % $(512))), dtype=tl.bfloat16)))
    r106_value = r106_value.to(tl.bfloat16)
    r106_value = tl.broadcast_to(r106_value, [$(524288)])
    tl.store($(candidate_vcat_7_ptr) + r106_z, r106_value)
    r107_row = tl.arange($(128))
    r107_col = tl.arange($(128))
    r107_lane = tl.arange($(512))
    for r107_group in range($(8)) {
      r107_left = tl.load($(candidate_vcat_7_ptr) + (((((((r107_group % $(8)) * $(65536)) + (r107_row[:, None] * $(512)) + r107_lane[None, :]) / $(512)) % $(128)) * $(4096)) + ((((((r107_group % $(8)) * $(65536)) + (r107_row[:, None] * $(512)) + r107_lane[None, :]) / $(65536)) % $(8)) * $(512)) + ((((r107_group % $(8)) * $(65536)) + (r107_row[:, None] * $(512)) + r107_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r107_right = tl.load($(candidate_vto_copy_5_ptr) + (((((((r107_group % $(8)) * $(65536)) + (r107_lane[:, None] * $(128)) + r107_col[None, :]) / $(65536)) % $(8)) * $(131072)) + (((((r107_group % $(8)) * $(65536)) + (r107_lane[:, None] * $(128)) + r107_col[None, :]) % $(128)) * $(512)) + (((((r107_group % $(8)) * $(65536)) + (r107_lane[:, None] * $(128)) + r107_col[None, :]) / $(128)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r107_value = tl.dot(r107_left, r107_right)
      r107_value = r107_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_6_ptr) + ((r107_group * $(16384)) + (r107_row[:, None] * $(128)) + r107_col[None, :]), r107_value)
    }
    r108_row = tl.arange($(128))
    r108_col = tl.arange($(512))
    r108_lane = tl.arange($(1024))
    r108_left = tl.load($(candidate_vbmm_6_ptr) + ((((((((((r108_row[:, None] * $(1024)) + r108_lane[None, :]) / $(1024)) % $(128)) * $(1024)) + (((r108_row[:, None] * $(1024)) + r108_lane[None, :]) % $(1024))) / $(128)) % $(8)) * $(16384)) + (((((r108_row[:, None] * $(1024)) + r108_lane[None, :]) / $(1024)) % $(128)) * $(128)) + (((((((r108_row[:, None] * $(1024)) + r108_lane[None, :]) / $(1024)) % $(128)) * $(1024)) + (((r108_row[:, None] * $(1024)) + r108_lane[None, :]) % $(1024))) % $(128))), dtype=tl.bfloat16)
    r108_right = tl.load($(candidate_vto_copy_6_ptr) + (((r108_col[None, :] % $(512)) * $(1024)) + (r108_lane[:, None] % $(1024))), dtype=tl.bfloat16)
    r108_value = tl.dot(r108_left, r108_right)
    r108_value = r108_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_9_ptr) + ((r108_row[:, None] * $(512)) + r108_col[None, :]), r108_value)
    z = tl.arange($(65536))
    value = tl.load($(candidate_vmm_9_ptr) + ((((z / $(512)) % $(128)) * $(512)) + (z % $(512))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(65536)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_mla_fa4_te_rope_thd_absorb_reuse_fwd
