import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_mla_fa4_te_rope_cp_absorb_fwd
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
  let candidate_vcat_6_ptr : RegionName := "candidate_vcat_6"
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
  let candidate_vcat_7_ptr : RegionName := "candidate_vcat_7"
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
  let candidate_vcat_8_ptr : RegionName := "candidate_vcat_8"
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
  let candidate_vcat_9_ptr : RegionName := "candidate_vcat_9"
  let candidate_vbmm_21_ptr : RegionName := "candidate_vbmm_21"
  let candidate_vcat_10_ptr : RegionName := "candidate_vcat_10"
  let candidate_vmm_27_ptr : RegionName := "candidate_vmm_27"
  let candidate_vindex_3_ptr : RegionName := "candidate_vindex_3"
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
    r85_z = tl.arange($(2048))
    r85_value = tl.load($(candidate_vadd_6_ptr) + ((((r85_z / $(64)) % $(4)) * $(512)) + (((r85_z / $(256)) % $(8)) * $(64)) + (r85_z % $(64)) + $(6144)), dtype=tl.bfloat16)
    r85_value = r85_value.to(tl.float32)
    r85_value = tl.broadcast_to(r85_value, [$(2048)])
    tl.store($(candidate_vto_copy_34_ptr) + r85_z, r85_value)
    r86_z = tl.arange($(256))
    r86_value = tl.load($(candidate_vindex_2_ptr) + ((((r86_z / $(64)) % $(4)) * $(576)) + (r86_z % $(64)) + $(28160)), dtype=tl.bfloat16)
    r86_value = r86_value.to(tl.float32)
    r86_value = tl.broadcast_to(r86_value, [$(256)])
    tl.store($(candidate_vto_copy_35_ptr) + r86_z, r86_value)
    r87_row = tl.arange($(4))
    r87_col = tl.arange($(4))
    r87_lane = tl.arange($(64))
    for r87_group in range($(8)) {
      r87_left = tl.load($(candidate_vto_copy_34_ptr) + (((((((r87_group % $(8)) * $(256)) + (r87_row[:, None] * $(64)) + r87_lane[None, :]) / $(256)) % $(8)) * $(256)) + ((((((r87_group % $(8)) * $(256)) + (r87_row[:, None] * $(64)) + r87_lane[None, :]) / $(64)) % $(4)) * $(64)) + ((((r87_group % $(8)) * $(256)) + (r87_row[:, None] * $(64)) + r87_lane[None, :]) % $(64))), dtype=tl.float32)
      r87_right = tl.load($(candidate_vto_copy_35_ptr) + ((((((r87_group % $(8)) * $(256)) + (r87_lane[:, None] * $(4)) + r87_col[None, :]) % $(4)) * $(64)) + (((((r87_group % $(8)) * $(256)) + (r87_lane[:, None] * $(4)) + r87_col[None, :]) / $(4)) % $(64))), dtype=tl.float32)
      r87_value = tl.dot(r87_left, r87_right)
      r87_value = r87_value.to(tl.float32)
      tl.store($(candidate_vbmm_4_ptr) + ((r87_group * $(16)) + (r87_row[:, None] * $(4)) + r87_col[None, :]), r87_value)
    }
    r88_z = tl.arange($(16384))
    r88_value = tl.load($(candidate_vbmm_2_ptr) + ((((r88_z / $(2048)) % $(8)) * $(16384)) + (((r88_z / $(512)) % $(4)) * $(512)) + (r88_z % $(512)) + $(6144)), dtype=tl.bfloat16)
    r88_value = r88_value.to(tl.float32)
    r88_value = tl.broadcast_to(r88_value, [$(16384)])
    tl.store($(candidate_vto_copy_36_ptr) + r88_z, r88_value)
    r89_z = tl.arange($(2048))
    r89_value = tl.load($(candidate_vindex_2_ptr) + ((((r89_z / $(512)) % $(4)) * $(576)) + (r89_z % $(512)) + $(27648)), dtype=tl.bfloat16)
    r89_value = r89_value.to(tl.float32)
    r89_value = tl.broadcast_to(r89_value, [$(2048)])
    tl.store($(candidate_vto_copy_37_ptr) + r89_z, r89_value)
    r90_row = tl.arange($(32))
    r90_col = tl.arange($(4))
    r90_lane = tl.arange($(512))
    r90_left = tl.load($(candidate_vto_copy_36_ptr) + ((((((r90_row[:, None] * $(512)) + r90_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + (((((r90_row[:, None] * $(512)) + r90_lane[None, :]) / $(512)) % $(4)) * $(512)) + (((r90_row[:, None] * $(512)) + r90_lane[None, :]) % $(512))), dtype=tl.float32)
    r90_right = tl.load($(candidate_vto_copy_37_ptr) + (((((r90_lane[:, None] * $(4)) + r90_col[None, :]) % $(4)) * $(512)) + ((((r90_lane[:, None] * $(4)) + r90_col[None, :]) / $(4)) % $(512))), dtype=tl.float32)
    r90_value = tl.dot(r90_left, r90_right)
    r90_value = r90_value.to(tl.float32)
    tl.store($(candidate_vmm_5_ptr) + ((r90_row[:, None] * $(4)) + r90_col[None, :]), r90_value)
    r91_z = tl.arange($(128))
    r91_value = (tl.load($(candidate_vbmm_4_ptr) + ((((r91_z / $(16)) % $(8)) * $(16)) + (((r91_z / $(4)) % $(4)) * $(4)) + (r91_z % $(4))), dtype=tl.float32) + tl.load($(candidate_vmm_5_ptr) + ((((r91_z / $(16)) % $(8)) * $(16)) + (((r91_z / $(4)) % $(4)) * $(4)) + (r91_z % $(4))), dtype=tl.float32))
    r91_value = r91_value.to(tl.float32)
    r91_value = tl.broadcast_to(r91_value, [$(128)])
    tl.store($(candidate_vadd_10_ptr) + r91_z, r91_value)
    r92_z = tl.arange($(128))
    r92_value = (tl.load($(candidate_vadd_10_ptr) + ((((r92_z / $(16)) % $(8)) * $(16)) + (((r92_z / $(4)) % $(4)) * $(4)) + (r92_z % $(4))), dtype=tl.float32) * 0.07216878364870322)
    r92_value = r92_value.to(tl.float32)
    r92_value = tl.broadcast_to(r92_value, [$(128)])
    tl.store($(candidate_vmul_24_ptr) + r92_z, r92_value)
    r93_z = tl.arange($(4))
    r93_value = (tl.cast(r93_z, tl.int64) + tl.full([], $(4), dtype=tl.int64))
    r93_value = tl.broadcast_to(r93_value, [$(4)])
    tl.store($(candidate_vadd_11_ptr) + r93_z, r93_value)
    r94_z = tl.arange($(4))
    r94_value = (tl.load($(candidate_vadd_11_ptr) + r94_z, dtype=tl.int64) - tl.full([], $(4), dtype=tl.int64))
    r94_value = tl.broadcast_to(r94_value, [$(4)])
    tl.store($(candidate_vsub_10_ptr) + r94_z, r94_value)
    r95_z = tl.arange($(16))
    r95_value = (tl.cast((r95_z % $(4)), tl.int64) > tl.load($(candidate_vsub_10_ptr) + ((r95_z / $(4)) % $(4)), dtype=tl.int64))
    r95_value = tl.broadcast_to(r95_value, [$(16)])
    tl.store($(candidate_vgt_1_ptr) + r95_z, r95_value)
    r96_z = tl.arange($(128))
    r96_value = tl.where(tl.load($(candidate_vgt_1_ptr) + ((((r96_z / $(4)) % $(4)) * $(4)) + (r96_z % $(4))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_24_ptr) + ((((r96_z / $(16)) % $(8)) * $(16)) + (((r96_z / $(4)) % $(4)) * $(4)) + (r96_z % $(4))), dtype=tl.float32))
    r96_value = r96_value.to(tl.float32)
    r96_value = tl.broadcast_to(r96_value, [$(128)])
    tl.store($(candidate_vmasked_fill_1_ptr) + r96_z, r96_value)
    r97_z = tl.arange($(128))
    r97_lane = tl.arange($(4))
    r97_row_values = tl.load($(candidate_vmasked_fill_1_ptr) + ((((r97_z[:, None] / $(16)) % $(8)) * $(16)) + (((r97_z[:, None] / $(4)) % $(4)) * $(4)) + (r97_lane[None, :] % $(4))), dtype=tl.float32)
    r97_maximum = tl.max(r97_row_values, axis=1, keep_dims=true)
    r97_weights = tl.exp(r97_row_values - r97_maximum)
    r97_total = tl.sum(r97_weights, axis=1)
    r97_maximum = tl.max(r97_row_values, axis=1)
    r97_value = tl.exp(tl.load($(candidate_vmasked_fill_1_ptr) + ((((r97_z / $(16)) % $(8)) * $(16)) + (((r97_z / $(4)) % $(4)) * $(4)) + (r97_z % $(4))), dtype=tl.float32) - r97_maximum) / r97_total
    r97_value = r97_value.to(tl.float32)
    r97_value = tl.broadcast_to(r97_value, [$(128)])
    tl.store($(candidate_vsoftmax_1_ptr) + r97_z, r97_value)
    r98_z = tl.arange($(128))
    r98_value = tl.load($(candidate_vsoftmax_1_ptr) + ((((r98_z / $(16)) % $(8)) * $(16)) + (((r98_z / $(4)) % $(4)) * $(4)) + (r98_z % $(4))), dtype=tl.float32)
    r98_value = r98_value.to(tl.bfloat16)
    r98_value = tl.broadcast_to(r98_value, [$(128)])
    tl.store($(candidate_vto_copy_38_ptr) + r98_z, r98_value)
    r99_row = tl.arange($(32))
    r99_col = tl.arange($(512))
    r99_lane = tl.arange($(4))
    r99_left = tl.load($(candidate_vto_copy_38_ptr) + ((((((r99_row[:, None] * $(4)) + r99_lane[None, :]) / $(16)) % $(8)) * $(16)) + (((((r99_row[:, None] * $(4)) + r99_lane[None, :]) / $(4)) % $(4)) * $(4)) + (((r99_row[:, None] * $(4)) + r99_lane[None, :]) % $(4))), dtype=tl.bfloat16)
    r99_right = tl.load($(candidate_vindex_2_ptr) + ((((((r99_lane[:, None] * $(512)) + r99_col[None, :]) / $(512)) % $(4)) * $(576)) + (((r99_lane[:, None] * $(512)) + r99_col[None, :]) % $(512)) + $(27648)), dtype=tl.bfloat16)
    r99_value = tl.dot(r99_left, r99_right)
    r99_value = r99_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_6_ptr) + ((r99_row[:, None] * $(512)) + r99_col[None, :]), r99_value)
    r100_z = tl.arange($(8192))
    r100_value = tl.load($(candidate_vadd_6_ptr) + ((((r100_z / $(64)) % $(16)) * $(512)) + (((r100_z / $(1024)) % $(8)) * $(64)) + (r100_z % $(64)) + $(8192)), dtype=tl.bfloat16)
    r100_value = r100_value.to(tl.float32)
    r100_value = tl.broadcast_to(r100_value, [$(8192)])
    tl.store($(candidate_vto_copy_39_ptr) + r100_z, r100_value)
    r101_z = tl.arange($(1024))
    r101_value = tl.load($(candidate_vindex_2_ptr) + ((((r101_z / $(64)) % $(16)) * $(576)) + (r101_z % $(64)) + $(37376)), dtype=tl.bfloat16)
    r101_value = r101_value.to(tl.float32)
    r101_value = tl.broadcast_to(r101_value, [$(1024)])
    tl.store($(candidate_vto_copy_40_ptr) + r101_z, r101_value)
    r102_row = tl.arange($(16))
    r102_col = tl.arange($(16))
    r102_lane = tl.arange($(64))
    for r102_group in range($(8)) {
      r102_left = tl.load($(candidate_vto_copy_39_ptr) + (((((((r102_group % $(8)) * $(1024)) + (r102_row[:, None] * $(64)) + r102_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + ((((((r102_group % $(8)) * $(1024)) + (r102_row[:, None] * $(64)) + r102_lane[None, :]) / $(64)) % $(16)) * $(64)) + ((((r102_group % $(8)) * $(1024)) + (r102_row[:, None] * $(64)) + r102_lane[None, :]) % $(64))), dtype=tl.float32)
      r102_right = tl.load($(candidate_vto_copy_40_ptr) + ((((((r102_group % $(8)) * $(1024)) + (r102_lane[:, None] * $(16)) + r102_col[None, :]) % $(16)) * $(64)) + (((((r102_group % $(8)) * $(1024)) + (r102_lane[:, None] * $(16)) + r102_col[None, :]) / $(16)) % $(64))), dtype=tl.float32)
      r102_value = tl.dot(r102_left, r102_right)
      r102_value = r102_value.to(tl.float32)
      tl.store($(candidate_vbmm_5_ptr) + ((r102_group * $(256)) + (r102_row[:, None] * $(16)) + r102_col[None, :]), r102_value)
    }
    r103_z = tl.arange($(65536))
    r103_value = tl.load($(candidate_vbmm_2_ptr) + ((((r103_z / $(8192)) % $(8)) * $(16384)) + (((r103_z / $(512)) % $(16)) * $(512)) + (r103_z % $(512)) + $(8192)), dtype=tl.bfloat16)
    r103_value = r103_value.to(tl.float32)
    r103_value = tl.broadcast_to(r103_value, [$(65536)])
    tl.store($(candidate_vto_copy_41_ptr) + r103_z, r103_value)
    r104_z = tl.arange($(8192))
    r104_value = tl.load($(candidate_vindex_2_ptr) + ((((r104_z / $(512)) % $(16)) * $(576)) + (r104_z % $(512)) + $(36864)), dtype=tl.bfloat16)
    r104_value = r104_value.to(tl.float32)
    r104_value = tl.broadcast_to(r104_value, [$(8192)])
    tl.store($(candidate_vto_copy_42_ptr) + r104_z, r104_value)
    r105_row = tl.arange($(128))
    r105_col = tl.arange($(16))
    r105_lane = tl.arange($(512))
    r105_left = tl.load($(candidate_vto_copy_41_ptr) + ((((((r105_row[:, None] * $(512)) + r105_lane[None, :]) / $(8192)) % $(8)) * $(8192)) + (((((r105_row[:, None] * $(512)) + r105_lane[None, :]) / $(512)) % $(16)) * $(512)) + (((r105_row[:, None] * $(512)) + r105_lane[None, :]) % $(512))), dtype=tl.float32)
    r105_right = tl.load($(candidate_vto_copy_42_ptr) + (((((r105_lane[:, None] * $(16)) + r105_col[None, :]) % $(16)) * $(512)) + ((((r105_lane[:, None] * $(16)) + r105_col[None, :]) / $(16)) % $(512))), dtype=tl.float32)
    r105_value = tl.dot(r105_left, r105_right)
    r105_value = r105_value.to(tl.float32)
    tl.store($(candidate_vmm_7_ptr) + ((r105_row[:, None] * $(16)) + r105_col[None, :]), r105_value)
    r106_z = tl.arange($(2048))
    r106_value = (tl.load($(candidate_vbmm_5_ptr) + ((((r106_z / $(256)) % $(8)) * $(256)) + (((r106_z / $(16)) % $(16)) * $(16)) + (r106_z % $(16))), dtype=tl.float32) + tl.load($(candidate_vmm_7_ptr) + ((((r106_z / $(256)) % $(8)) * $(256)) + (((r106_z / $(16)) % $(16)) * $(16)) + (r106_z % $(16))), dtype=tl.float32))
    r106_value = r106_value.to(tl.float32)
    r106_value = tl.broadcast_to(r106_value, [$(2048)])
    tl.store($(candidate_vadd_12_ptr) + r106_z, r106_value)
    r107_z = tl.arange($(2048))
    r107_value = (tl.load($(candidate_vadd_12_ptr) + ((((r107_z / $(256)) % $(8)) * $(256)) + (((r107_z / $(16)) % $(16)) * $(16)) + (r107_z % $(16))), dtype=tl.float32) * 0.07216878364870322)
    r107_value = r107_value.to(tl.float32)
    r107_value = tl.broadcast_to(r107_value, [$(2048)])
    tl.store($(candidate_vmul_25_ptr) + r107_z, r107_value)
    r108_z = tl.arange($(16))
    r108_value = (tl.cast(r108_z, tl.int64) + tl.full([], $(16), dtype=tl.int64))
    r108_value = tl.broadcast_to(r108_value, [$(16)])
    tl.store($(candidate_vadd_13_ptr) + r108_z, r108_value)
    r109_z = tl.arange($(16))
    r109_value = (tl.load($(candidate_vadd_13_ptr) + r109_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r109_value = tl.broadcast_to(r109_value, [$(16)])
    tl.store($(candidate_vsub_11_ptr) + r109_z, r109_value)
    r110_z = tl.arange($(256))
    r110_value = (tl.cast((r110_z % $(16)), tl.int64) > tl.load($(candidate_vsub_11_ptr) + ((r110_z / $(16)) % $(16)), dtype=tl.int64))
    r110_value = tl.broadcast_to(r110_value, [$(256)])
    tl.store($(candidate_vgt_2_ptr) + r110_z, r110_value)
    r111_z = tl.arange($(2048))
    r111_value = tl.where(tl.load($(candidate_vgt_2_ptr) + ((((r111_z / $(16)) % $(16)) * $(16)) + (r111_z % $(16))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_25_ptr) + ((((r111_z / $(256)) % $(8)) * $(256)) + (((r111_z / $(16)) % $(16)) * $(16)) + (r111_z % $(16))), dtype=tl.float32))
    r111_value = r111_value.to(tl.float32)
    r111_value = tl.broadcast_to(r111_value, [$(2048)])
    tl.store($(candidate_vmasked_fill_2_ptr) + r111_z, r111_value)
    r112_z = tl.arange($(2048))
    r112_lane = tl.arange($(16))
    r112_row_values = tl.load($(candidate_vmasked_fill_2_ptr) + ((((r112_z[:, None] / $(256)) % $(8)) * $(256)) + (((r112_z[:, None] / $(16)) % $(16)) * $(16)) + (r112_lane[None, :] % $(16))), dtype=tl.float32)
    r112_maximum = tl.max(r112_row_values, axis=1, keep_dims=true)
    r112_weights = tl.exp(r112_row_values - r112_maximum)
    r112_total = tl.sum(r112_weights, axis=1)
    r112_maximum = tl.max(r112_row_values, axis=1)
    r112_value = tl.exp(tl.load($(candidate_vmasked_fill_2_ptr) + ((((r112_z / $(256)) % $(8)) * $(256)) + (((r112_z / $(16)) % $(16)) * $(16)) + (r112_z % $(16))), dtype=tl.float32) - r112_maximum) / r112_total
    r112_value = r112_value.to(tl.float32)
    r112_value = tl.broadcast_to(r112_value, [$(2048)])
    tl.store($(candidate_vsoftmax_2_ptr) + r112_z, r112_value)
    r113_z = tl.arange($(2048))
    r113_value = tl.load($(candidate_vsoftmax_2_ptr) + ((((r113_z / $(256)) % $(8)) * $(256)) + (((r113_z / $(16)) % $(16)) * $(16)) + (r113_z % $(16))), dtype=tl.float32)
    r113_value = r113_value.to(tl.bfloat16)
    r113_value = tl.broadcast_to(r113_value, [$(2048)])
    tl.store($(candidate_vto_copy_43_ptr) + r113_z, r113_value)
    r114_row = tl.arange($(128))
    r114_col = tl.arange($(512))
    r114_lane = tl.arange($(16))
    r114_left = tl.load($(candidate_vto_copy_43_ptr) + ((((((r114_row[:, None] * $(16)) + r114_lane[None, :]) / $(256)) % $(8)) * $(256)) + (((((r114_row[:, None] * $(16)) + r114_lane[None, :]) / $(16)) % $(16)) * $(16)) + (((r114_row[:, None] * $(16)) + r114_lane[None, :]) % $(16))), dtype=tl.bfloat16)
    r114_right = tl.load($(candidate_vindex_2_ptr) + ((((((r114_lane[:, None] * $(512)) + r114_col[None, :]) / $(512)) % $(16)) * $(576)) + (((r114_lane[:, None] * $(512)) + r114_col[None, :]) % $(512)) + $(36864)), dtype=tl.bfloat16)
    r114_value = tl.dot(r114_left, r114_right)
    r114_value = r114_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_8_ptr) + ((r114_row[:, None] * $(512)) + r114_col[None, :]), r114_value)
    r115_z = tl.arange($(131072))
    r115_value = tl.where(((r115_z / $(4096)) % $(32)) < $(12), tl.load($(candidate_vmm_4_ptr) + (((((((r115_z / $(512)) % $(8)) * $(12)) + (((r115_z / $(4096)) % $(32)) - $(0))) % $(96)) * $(512)) + (r115_z % $(512))), dtype=tl.bfloat16), tl.where(((r115_z / $(4096)) % $(32)) < $(16), tl.load($(candidate_vmm_6_ptr) + (((((((r115_z / $(512)) % $(8)) * $(4)) + (((r115_z / $(4096)) % $(32)) - $(12))) % $(32)) * $(512)) + (r115_z % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vmm_8_ptr) + (((((((r115_z / $(512)) % $(8)) * $(16)) + (((r115_z / $(4096)) % $(32)) - $(16))) % $(128)) * $(512)) + (r115_z % $(512))), dtype=tl.bfloat16)))
    r115_value = r115_value.to(tl.bfloat16)
    r115_value = tl.broadcast_to(r115_value, [$(131072)])
    tl.store($(candidate_vcat_6_ptr) + r115_z, r115_value)
    r116_row = tl.arange($(32))
    r116_col = tl.arange($(128))
    r116_lane = tl.arange($(512))
    for r116_group in range($(8)) {
      r116_left = tl.load($(candidate_vcat_6_ptr) + (((((((r116_group % $(8)) * $(16384)) + (r116_row[:, None] * $(512)) + r116_lane[None, :]) / $(512)) % $(32)) * $(4096)) + ((((((r116_group % $(8)) * $(16384)) + (r116_row[:, None] * $(512)) + r116_lane[None, :]) / $(16384)) % $(8)) * $(512)) + ((((r116_group % $(8)) * $(16384)) + (r116_row[:, None] * $(512)) + r116_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r116_right = tl.load($(candidate_vto_copy_5_ptr) + (((((((r116_group % $(8)) * $(65536)) + (r116_lane[:, None] * $(128)) + r116_col[None, :]) / $(65536)) % $(8)) * $(131072)) + (((((r116_group % $(8)) * $(65536)) + (r116_lane[:, None] * $(128)) + r116_col[None, :]) % $(128)) * $(512)) + (((((r116_group % $(8)) * $(65536)) + (r116_lane[:, None] * $(128)) + r116_col[None, :]) / $(128)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r116_value = tl.dot(r116_left, r116_right)
      r116_value = r116_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_6_ptr) + ((r116_group * $(4096)) + (r116_row[:, None] * $(128)) + r116_col[None, :]), r116_value)
    }
    r117_row = tl.arange($(32))
    r117_col = tl.arange($(512))
    r117_lane = tl.arange($(128))
    for r117_group in range($(8)) {
      r117_left = tl.load($(candidate_vmm_1_ptr) + (((((((r117_group % $(8)) * $(4096)) + (r117_row[:, None] * $(128)) + r117_lane[None, :]) / $(128)) % $(32)) * $(1536)) + ((((((r117_group % $(8)) * $(4096)) + (r117_row[:, None] * $(128)) + r117_lane[None, :]) / $(4096)) % $(8)) * $(192)) + ((((r117_group % $(8)) * $(4096)) + (r117_row[:, None] * $(128)) + r117_lane[None, :]) % $(128)) + $(49152)), dtype=tl.bfloat16)
      r117_right = tl.load($(candidate_vto_copy_5_ptr) + (((((((r117_group % $(8)) * $(65536)) + (r117_lane[:, None] * $(512)) + r117_col[None, :]) / $(65536)) % $(8)) * $(131072)) + ((((((r117_group % $(8)) * $(65536)) + (r117_lane[:, None] * $(512)) + r117_col[None, :]) / $(512)) % $(128)) * $(512)) + ((((r117_group % $(8)) * $(65536)) + (r117_lane[:, None] * $(512)) + r117_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r117_value = tl.dot(r117_left, r117_right)
      r117_value = r117_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_7_ptr) + ((r117_group * $(16384)) + (r117_row[:, None] * $(512)) + r117_col[None, :]), r117_value)
    }
    r118_z = tl.arange($(6144))
    r118_value = tl.load($(candidate_vadd_6_ptr) + ((((r118_z / $(64)) % $(12)) * $(512)) + (((r118_z / $(768)) % $(8)) * $(64)) + (r118_z % $(64)) + $(16384)), dtype=tl.bfloat16)
    r118_value = r118_value.to(tl.float32)
    r118_value = tl.broadcast_to(r118_value, [$(6144)])
    tl.store($(candidate_vto_copy_44_ptr) + r118_z, r118_value)
    r119_z = tl.arange($(3072))
    r119_value = tl.load($(candidate_vindex_2_ptr) + ((((r119_z / $(64)) % $(48)) * $(576)) + (r119_z % $(64)) + $(512)), dtype=tl.bfloat16)
    r119_value = r119_value.to(tl.float32)
    r119_value = tl.broadcast_to(r119_value, [$(3072)])
    tl.store($(candidate_vto_copy_45_ptr) + r119_z, r119_value)
    r120_row = tl.arange($(12))
    r120_col = tl.arange($(48))
    r120_lane = tl.arange($(64))
    for r120_group in range($(8)) {
      r120_left = tl.load($(candidate_vto_copy_44_ptr) + (((((((r120_group % $(8)) * $(768)) + (r120_row[:, None] * $(64)) + r120_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r120_group % $(8)) * $(768)) + (r120_row[:, None] * $(64)) + r120_lane[None, :]) / $(64)) % $(12)) * $(64)) + ((((r120_group % $(8)) * $(768)) + (r120_row[:, None] * $(64)) + r120_lane[None, :]) % $(64))), dtype=tl.float32)
      r120_right = tl.load($(candidate_vto_copy_45_ptr) + ((((((r120_group % $(8)) * $(3072)) + (r120_lane[:, None] * $(48)) + r120_col[None, :]) % $(48)) * $(64)) + (((((r120_group % $(8)) * $(3072)) + (r120_lane[:, None] * $(48)) + r120_col[None, :]) / $(48)) % $(64))), dtype=tl.float32)
      r120_value = tl.dot(r120_left, r120_right)
      r120_value = r120_value.to(tl.float32)
      tl.store($(candidate_vbmm_8_ptr) + ((r120_group * $(576)) + (r120_row[:, None] * $(48)) + r120_col[None, :]), r120_value)
    }
    r121_z = tl.arange($(49152))
    r121_value = tl.load($(candidate_vbmm_7_ptr) + ((((r121_z / $(6144)) % $(8)) * $(16384)) + (((r121_z / $(512)) % $(12)) * $(512)) + (r121_z % $(512))), dtype=tl.bfloat16)
    r121_value = r121_value.to(tl.float32)
    r121_value = tl.broadcast_to(r121_value, [$(49152)])
    tl.store($(candidate_vto_copy_46_ptr) + r121_z, r121_value)
    r122_z = tl.arange($(24576))
    r122_value = tl.load($(candidate_vindex_2_ptr) + ((((r122_z / $(512)) % $(48)) * $(576)) + (r122_z % $(512))), dtype=tl.bfloat16)
    r122_value = r122_value.to(tl.float32)
    r122_value = tl.broadcast_to(r122_value, [$(24576)])
    tl.store($(candidate_vto_copy_47_ptr) + r122_z, r122_value)
    r123_row = tl.arange($(96))
    r123_col = tl.arange($(48))
    r123_lane = tl.arange($(512))
    r123_left = tl.load($(candidate_vto_copy_46_ptr) + ((((((r123_row[:, None] * $(512)) + r123_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r123_row[:, None] * $(512)) + r123_lane[None, :]) / $(512)) % $(12)) * $(512)) + (((r123_row[:, None] * $(512)) + r123_lane[None, :]) % $(512))), dtype=tl.float32)
    r123_right = tl.load($(candidate_vto_copy_47_ptr) + (((((r123_lane[:, None] * $(48)) + r123_col[None, :]) % $(48)) * $(512)) + ((((r123_lane[:, None] * $(48)) + r123_col[None, :]) / $(48)) % $(512))), dtype=tl.float32)
    r123_value = tl.dot(r123_left, r123_right)
    r123_value = r123_value.to(tl.float32)
    tl.store($(candidate_vmm_9_ptr) + ((r123_row[:, None] * $(48)) + r123_col[None, :]), r123_value)
    r124_z = tl.arange($(4608))
    r124_value = (tl.load($(candidate_vbmm_8_ptr) + ((((r124_z / $(576)) % $(8)) * $(576)) + (((r124_z / $(48)) % $(12)) * $(48)) + (r124_z % $(48))), dtype=tl.float32) + tl.load($(candidate_vmm_9_ptr) + ((((r124_z / $(576)) % $(8)) * $(576)) + (((r124_z / $(48)) % $(12)) * $(48)) + (r124_z % $(48))), dtype=tl.float32))
    r124_value = r124_value.to(tl.float32)
    r124_value = tl.broadcast_to(r124_value, [$(4608)])
    tl.store($(candidate_vadd_14_ptr) + r124_z, r124_value)
    r125_z = tl.arange($(4608))
    r125_value = (tl.load($(candidate_vadd_14_ptr) + ((((r125_z / $(576)) % $(8)) * $(576)) + (((r125_z / $(48)) % $(12)) * $(48)) + (r125_z % $(48))), dtype=tl.float32) * 0.07216878364870322)
    r125_value = r125_value.to(tl.float32)
    r125_value = tl.broadcast_to(r125_value, [$(4608)])
    tl.store($(candidate_vmul_26_ptr) + r125_z, r125_value)
    r126_z = tl.arange($(12))
    r126_value = (tl.cast(r126_z, tl.int64) + tl.full([], $(48), dtype=tl.int64))
    r126_value = tl.broadcast_to(r126_value, [$(12)])
    tl.store($(candidate_vadd_15_ptr) + r126_z, r126_value)
    r127_z = tl.arange($(12))
    r127_value = (tl.load($(candidate_vadd_15_ptr) + r127_z, dtype=tl.int64) - tl.full([], $(12), dtype=tl.int64))
    r127_value = tl.broadcast_to(r127_value, [$(12)])
    tl.store($(candidate_vsub_12_ptr) + r127_z, r127_value)
    r128_z = tl.arange($(576))
    r128_value = (tl.cast((r128_z % $(48)), tl.int64) > tl.load($(candidate_vsub_12_ptr) + ((r128_z / $(48)) % $(12)), dtype=tl.int64))
    r128_value = tl.broadcast_to(r128_value, [$(576)])
    tl.store($(candidate_vgt_3_ptr) + r128_z, r128_value)
    r129_z = tl.arange($(4608))
    r129_value = tl.where(tl.load($(candidate_vgt_3_ptr) + ((((r129_z / $(48)) % $(12)) * $(48)) + (r129_z % $(48))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_26_ptr) + ((((r129_z / $(576)) % $(8)) * $(576)) + (((r129_z / $(48)) % $(12)) * $(48)) + (r129_z % $(48))), dtype=tl.float32))
    r129_value = r129_value.to(tl.float32)
    r129_value = tl.broadcast_to(r129_value, [$(4608)])
    tl.store($(candidate_vmasked_fill_3_ptr) + r129_z, r129_value)
    r130_z = tl.arange($(4608))
    r130_lane = tl.arange($(48))
    r130_row_values = tl.load($(candidate_vmasked_fill_3_ptr) + ((((r130_z[:, None] / $(576)) % $(8)) * $(576)) + (((r130_z[:, None] / $(48)) % $(12)) * $(48)) + (r130_lane[None, :] % $(48))), dtype=tl.float32)
    r130_maximum = tl.max(r130_row_values, axis=1, keep_dims=true)
    r130_weights = tl.exp(r130_row_values - r130_maximum)
    r130_total = tl.sum(r130_weights, axis=1)
    r130_maximum = tl.max(r130_row_values, axis=1)
    r130_value = tl.exp(tl.load($(candidate_vmasked_fill_3_ptr) + ((((r130_z / $(576)) % $(8)) * $(576)) + (((r130_z / $(48)) % $(12)) * $(48)) + (r130_z % $(48))), dtype=tl.float32) - r130_maximum) / r130_total
    r130_value = r130_value.to(tl.float32)
    r130_value = tl.broadcast_to(r130_value, [$(4608)])
    tl.store($(candidate_vsoftmax_3_ptr) + r130_z, r130_value)
    r131_z = tl.arange($(4608))
    r131_value = tl.load($(candidate_vsoftmax_3_ptr) + ((((r131_z / $(576)) % $(8)) * $(576)) + (((r131_z / $(48)) % $(12)) * $(48)) + (r131_z % $(48))), dtype=tl.float32)
    r131_value = r131_value.to(tl.bfloat16)
    r131_value = tl.broadcast_to(r131_value, [$(4608)])
    tl.store($(candidate_vto_copy_48_ptr) + r131_z, r131_value)
    r132_row = tl.arange($(96))
    r132_col = tl.arange($(512))
    r132_lane = tl.arange($(48))
    r132_left = tl.load($(candidate_vto_copy_48_ptr) + ((((((r132_row[:, None] * $(48)) + r132_lane[None, :]) / $(576)) % $(8)) * $(576)) + (((((r132_row[:, None] * $(48)) + r132_lane[None, :]) / $(48)) % $(12)) * $(48)) + (((r132_row[:, None] * $(48)) + r132_lane[None, :]) % $(48))), dtype=tl.bfloat16)
    r132_right = tl.load($(candidate_vindex_2_ptr) + ((((((r132_lane[:, None] * $(512)) + r132_col[None, :]) / $(512)) % $(48)) * $(576)) + (((r132_lane[:, None] * $(512)) + r132_col[None, :]) % $(512))), dtype=tl.bfloat16)
    r132_value = tl.dot(r132_left, r132_right)
    r132_value = r132_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_10_ptr) + ((r132_row[:, None] * $(512)) + r132_col[None, :]), r132_value)
    r133_z = tl.arange($(2048))
    r133_value = tl.load($(candidate_vadd_6_ptr) + ((((r133_z / $(64)) % $(4)) * $(512)) + (((r133_z / $(256)) % $(8)) * $(64)) + (r133_z % $(64)) + $(22528)), dtype=tl.bfloat16)
    r133_value = r133_value.to(tl.float32)
    r133_value = tl.broadcast_to(r133_value, [$(2048)])
    tl.store($(candidate_vto_copy_49_ptr) + r133_z, r133_value)
    r134_z = tl.arange($(1024))
    r134_value = tl.load($(candidate_vindex_2_ptr) + ((((r134_z / $(64)) % $(16)) * $(576)) + (r134_z % $(64)) + $(28160)), dtype=tl.bfloat16)
    r134_value = r134_value.to(tl.float32)
    r134_value = tl.broadcast_to(r134_value, [$(1024)])
    tl.store($(candidate_vto_copy_50_ptr) + r134_z, r134_value)
    r135_row = tl.arange($(4))
    r135_col = tl.arange($(16))
    r135_lane = tl.arange($(64))
    for r135_group in range($(8)) {
      r135_left = tl.load($(candidate_vto_copy_49_ptr) + (((((((r135_group % $(8)) * $(256)) + (r135_row[:, None] * $(64)) + r135_lane[None, :]) / $(256)) % $(8)) * $(256)) + ((((((r135_group % $(8)) * $(256)) + (r135_row[:, None] * $(64)) + r135_lane[None, :]) / $(64)) % $(4)) * $(64)) + ((((r135_group % $(8)) * $(256)) + (r135_row[:, None] * $(64)) + r135_lane[None, :]) % $(64))), dtype=tl.float32)
      r135_right = tl.load($(candidate_vto_copy_50_ptr) + ((((((r135_group % $(8)) * $(1024)) + (r135_lane[:, None] * $(16)) + r135_col[None, :]) % $(16)) * $(64)) + (((((r135_group % $(8)) * $(1024)) + (r135_lane[:, None] * $(16)) + r135_col[None, :]) / $(16)) % $(64))), dtype=tl.float32)
      r135_value = tl.dot(r135_left, r135_right)
      r135_value = r135_value.to(tl.float32)
      tl.store($(candidate_vbmm_9_ptr) + ((r135_group * $(64)) + (r135_row[:, None] * $(16)) + r135_col[None, :]), r135_value)
    }
    r136_z = tl.arange($(16384))
    r136_value = tl.load($(candidate_vbmm_7_ptr) + ((((r136_z / $(2048)) % $(8)) * $(16384)) + (((r136_z / $(512)) % $(4)) * $(512)) + (r136_z % $(512)) + $(6144)), dtype=tl.bfloat16)
    r136_value = r136_value.to(tl.float32)
    r136_value = tl.broadcast_to(r136_value, [$(16384)])
    tl.store($(candidate_vto_copy_51_ptr) + r136_z, r136_value)
    r137_z = tl.arange($(8192))
    r137_value = tl.load($(candidate_vindex_2_ptr) + ((((r137_z / $(512)) % $(16)) * $(576)) + (r137_z % $(512)) + $(27648)), dtype=tl.bfloat16)
    r137_value = r137_value.to(tl.float32)
    r137_value = tl.broadcast_to(r137_value, [$(8192)])
    tl.store($(candidate_vto_copy_52_ptr) + r137_z, r137_value)
    r138_row = tl.arange($(32))
    r138_col = tl.arange($(16))
    r138_lane = tl.arange($(512))
    r138_left = tl.load($(candidate_vto_copy_51_ptr) + ((((((r138_row[:, None] * $(512)) + r138_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + (((((r138_row[:, None] * $(512)) + r138_lane[None, :]) / $(512)) % $(4)) * $(512)) + (((r138_row[:, None] * $(512)) + r138_lane[None, :]) % $(512))), dtype=tl.float32)
    r138_right = tl.load($(candidate_vto_copy_52_ptr) + (((((r138_lane[:, None] * $(16)) + r138_col[None, :]) % $(16)) * $(512)) + ((((r138_lane[:, None] * $(16)) + r138_col[None, :]) / $(16)) % $(512))), dtype=tl.float32)
    r138_value = tl.dot(r138_left, r138_right)
    r138_value = r138_value.to(tl.float32)
    tl.store($(candidate_vmm_11_ptr) + ((r138_row[:, None] * $(16)) + r138_col[None, :]), r138_value)
    r139_z = tl.arange($(512))
    r139_value = (tl.load($(candidate_vbmm_9_ptr) + ((((r139_z / $(64)) % $(8)) * $(64)) + (((r139_z / $(16)) % $(4)) * $(16)) + (r139_z % $(16))), dtype=tl.float32) + tl.load($(candidate_vmm_11_ptr) + ((((r139_z / $(64)) % $(8)) * $(64)) + (((r139_z / $(16)) % $(4)) * $(16)) + (r139_z % $(16))), dtype=tl.float32))
    r139_value = r139_value.to(tl.float32)
    r139_value = tl.broadcast_to(r139_value, [$(512)])
    tl.store($(candidate_vadd_16_ptr) + r139_z, r139_value)
    r140_z = tl.arange($(512))
    r140_value = (tl.load($(candidate_vadd_16_ptr) + ((((r140_z / $(64)) % $(8)) * $(64)) + (((r140_z / $(16)) % $(4)) * $(16)) + (r140_z % $(16))), dtype=tl.float32) * 0.07216878364870322)
    r140_value = r140_value.to(tl.float32)
    r140_value = tl.broadcast_to(r140_value, [$(512)])
    tl.store($(candidate_vmul_27_ptr) + r140_z, r140_value)
    r141_z = tl.arange($(4))
    r141_value = (tl.cast(r141_z, tl.int64) + tl.full([], $(16), dtype=tl.int64))
    r141_value = tl.broadcast_to(r141_value, [$(4)])
    tl.store($(candidate_vadd_17_ptr) + r141_z, r141_value)
    r142_z = tl.arange($(4))
    r142_value = (tl.load($(candidate_vadd_17_ptr) + r142_z, dtype=tl.int64) - tl.full([], $(4), dtype=tl.int64))
    r142_value = tl.broadcast_to(r142_value, [$(4)])
    tl.store($(candidate_vsub_13_ptr) + r142_z, r142_value)
    r143_z = tl.arange($(64))
    r143_value = (tl.cast((r143_z % $(16)), tl.int64) > tl.load($(candidate_vsub_13_ptr) + ((r143_z / $(16)) % $(4)), dtype=tl.int64))
    r143_value = tl.broadcast_to(r143_value, [$(64)])
    tl.store($(candidate_vgt_4_ptr) + r143_z, r143_value)
    r144_z = tl.arange($(512))
    r144_value = tl.where(tl.load($(candidate_vgt_4_ptr) + ((((r144_z / $(16)) % $(4)) * $(16)) + (r144_z % $(16))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_27_ptr) + ((((r144_z / $(64)) % $(8)) * $(64)) + (((r144_z / $(16)) % $(4)) * $(16)) + (r144_z % $(16))), dtype=tl.float32))
    r144_value = r144_value.to(tl.float32)
    r144_value = tl.broadcast_to(r144_value, [$(512)])
    tl.store($(candidate_vmasked_fill_4_ptr) + r144_z, r144_value)
    r145_z = tl.arange($(512))
    r145_lane = tl.arange($(16))
    r145_row_values = tl.load($(candidate_vmasked_fill_4_ptr) + ((((r145_z[:, None] / $(64)) % $(8)) * $(64)) + (((r145_z[:, None] / $(16)) % $(4)) * $(16)) + (r145_lane[None, :] % $(16))), dtype=tl.float32)
    r145_maximum = tl.max(r145_row_values, axis=1, keep_dims=true)
    r145_weights = tl.exp(r145_row_values - r145_maximum)
    r145_total = tl.sum(r145_weights, axis=1)
    r145_maximum = tl.max(r145_row_values, axis=1)
    r145_value = tl.exp(tl.load($(candidate_vmasked_fill_4_ptr) + ((((r145_z / $(64)) % $(8)) * $(64)) + (((r145_z / $(16)) % $(4)) * $(16)) + (r145_z % $(16))), dtype=tl.float32) - r145_maximum) / r145_total
    r145_value = r145_value.to(tl.float32)
    r145_value = tl.broadcast_to(r145_value, [$(512)])
    tl.store($(candidate_vsoftmax_4_ptr) + r145_z, r145_value)
    r146_z = tl.arange($(512))
    r146_value = tl.load($(candidate_vsoftmax_4_ptr) + ((((r146_z / $(64)) % $(8)) * $(64)) + (((r146_z / $(16)) % $(4)) * $(16)) + (r146_z % $(16))), dtype=tl.float32)
    r146_value = r146_value.to(tl.bfloat16)
    r146_value = tl.broadcast_to(r146_value, [$(512)])
    tl.store($(candidate_vto_copy_53_ptr) + r146_z, r146_value)
    r147_row = tl.arange($(32))
    r147_col = tl.arange($(512))
    r147_lane = tl.arange($(16))
    r147_left = tl.load($(candidate_vto_copy_53_ptr) + ((((((r147_row[:, None] * $(16)) + r147_lane[None, :]) / $(64)) % $(8)) * $(64)) + (((((r147_row[:, None] * $(16)) + r147_lane[None, :]) / $(16)) % $(4)) * $(16)) + (((r147_row[:, None] * $(16)) + r147_lane[None, :]) % $(16))), dtype=tl.bfloat16)
    r147_right = tl.load($(candidate_vindex_2_ptr) + ((((((r147_lane[:, None] * $(512)) + r147_col[None, :]) / $(512)) % $(16)) * $(576)) + (((r147_lane[:, None] * $(512)) + r147_col[None, :]) % $(512)) + $(27648)), dtype=tl.bfloat16)
    r147_value = tl.dot(r147_left, r147_right)
    r147_value = r147_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_12_ptr) + ((r147_row[:, None] * $(512)) + r147_col[None, :]), r147_value)
    r148_z = tl.arange($(8192))
    r148_value = tl.load($(candidate_vadd_6_ptr) + ((((r148_z / $(64)) % $(16)) * $(512)) + (((r148_z / $(1024)) % $(8)) * $(64)) + (r148_z % $(64)) + $(24576)), dtype=tl.bfloat16)
    r148_value = r148_value.to(tl.float32)
    r148_value = tl.broadcast_to(r148_value, [$(8192)])
    tl.store($(candidate_vto_copy_54_ptr) + r148_z, r148_value)
    r149_z = tl.arange($(4096))
    r149_value = tl.load($(candidate_vindex_2_ptr) + ((((r149_z / $(64)) % $(64)) * $(576)) + (r149_z % $(64)) + $(37376)), dtype=tl.bfloat16)
    r149_value = r149_value.to(tl.float32)
    r149_value = tl.broadcast_to(r149_value, [$(4096)])
    tl.store($(candidate_vto_copy_55_ptr) + r149_z, r149_value)
    r150_row = tl.arange($(16))
    r150_col = tl.arange($(64))
    r150_lane = tl.arange($(64))
    for r150_group in range($(8)) {
      r150_left = tl.load($(candidate_vto_copy_54_ptr) + (((((((r150_group % $(8)) * $(1024)) + (r150_row[:, None] * $(64)) + r150_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + ((((((r150_group % $(8)) * $(1024)) + (r150_row[:, None] * $(64)) + r150_lane[None, :]) / $(64)) % $(16)) * $(64)) + ((((r150_group % $(8)) * $(1024)) + (r150_row[:, None] * $(64)) + r150_lane[None, :]) % $(64))), dtype=tl.float32)
      r150_right = tl.load($(candidate_vto_copy_55_ptr) + ((((((r150_group % $(8)) * $(4096)) + (r150_lane[:, None] * $(64)) + r150_col[None, :]) % $(64)) * $(64)) + (((((r150_group % $(8)) * $(4096)) + (r150_lane[:, None] * $(64)) + r150_col[None, :]) / $(64)) % $(64))), dtype=tl.float32)
      r150_value = tl.dot(r150_left, r150_right)
      r150_value = r150_value.to(tl.float32)
      tl.store($(candidate_vbmm_10_ptr) + ((r150_group * $(1024)) + (r150_row[:, None] * $(64)) + r150_col[None, :]), r150_value)
    }
    r151_z = tl.arange($(65536))
    r151_value = tl.load($(candidate_vbmm_7_ptr) + ((((r151_z / $(8192)) % $(8)) * $(16384)) + (((r151_z / $(512)) % $(16)) * $(512)) + (r151_z % $(512)) + $(8192)), dtype=tl.bfloat16)
    r151_value = r151_value.to(tl.float32)
    r151_value = tl.broadcast_to(r151_value, [$(65536)])
    tl.store($(candidate_vto_copy_56_ptr) + r151_z, r151_value)
    r152_z = tl.arange($(32768))
    r152_value = tl.load($(candidate_vindex_2_ptr) + ((((r152_z / $(512)) % $(64)) * $(576)) + (r152_z % $(512)) + $(36864)), dtype=tl.bfloat16)
    r152_value = r152_value.to(tl.float32)
    r152_value = tl.broadcast_to(r152_value, [$(32768)])
    tl.store($(candidate_vto_copy_57_ptr) + r152_z, r152_value)
    r153_row = tl.arange($(128))
    r153_col = tl.arange($(64))
    r153_lane = tl.arange($(512))
    r153_left = tl.load($(candidate_vto_copy_56_ptr) + ((((((r153_row[:, None] * $(512)) + r153_lane[None, :]) / $(8192)) % $(8)) * $(8192)) + (((((r153_row[:, None] * $(512)) + r153_lane[None, :]) / $(512)) % $(16)) * $(512)) + (((r153_row[:, None] * $(512)) + r153_lane[None, :]) % $(512))), dtype=tl.float32)
    r153_right = tl.load($(candidate_vto_copy_57_ptr) + (((((r153_lane[:, None] * $(64)) + r153_col[None, :]) % $(64)) * $(512)) + ((((r153_lane[:, None] * $(64)) + r153_col[None, :]) / $(64)) % $(512))), dtype=tl.float32)
    r153_value = tl.dot(r153_left, r153_right)
    r153_value = r153_value.to(tl.float32)
    tl.store($(candidate_vmm_13_ptr) + ((r153_row[:, None] * $(64)) + r153_col[None, :]), r153_value)
    r154_z = tl.arange($(8192))
    r154_value = (tl.load($(candidate_vbmm_10_ptr) + ((((r154_z / $(1024)) % $(8)) * $(1024)) + (((r154_z / $(64)) % $(16)) * $(64)) + (r154_z % $(64))), dtype=tl.float32) + tl.load($(candidate_vmm_13_ptr) + ((((r154_z / $(1024)) % $(8)) * $(1024)) + (((r154_z / $(64)) % $(16)) * $(64)) + (r154_z % $(64))), dtype=tl.float32))
    r154_value = r154_value.to(tl.float32)
    r154_value = tl.broadcast_to(r154_value, [$(8192)])
    tl.store($(candidate_vadd_18_ptr) + r154_z, r154_value)
    r155_z = tl.arange($(8192))
    r155_value = (tl.load($(candidate_vadd_18_ptr) + ((((r155_z / $(1024)) % $(8)) * $(1024)) + (((r155_z / $(64)) % $(16)) * $(64)) + (r155_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r155_value = r155_value.to(tl.float32)
    r155_value = tl.broadcast_to(r155_value, [$(8192)])
    tl.store($(candidate_vmul_28_ptr) + r155_z, r155_value)
    r156_z = tl.arange($(16))
    r156_value = (tl.cast(r156_z, tl.int64) + tl.full([], $(64), dtype=tl.int64))
    r156_value = tl.broadcast_to(r156_value, [$(16)])
    tl.store($(candidate_vadd_19_ptr) + r156_z, r156_value)
    r157_z = tl.arange($(16))
    r157_value = (tl.load($(candidate_vadd_19_ptr) + r157_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r157_value = tl.broadcast_to(r157_value, [$(16)])
    tl.store($(candidate_vsub_14_ptr) + r157_z, r157_value)
    r158_z = tl.arange($(1024))
    r158_value = (tl.cast((r158_z % $(64)), tl.int64) > tl.load($(candidate_vsub_14_ptr) + ((r158_z / $(64)) % $(16)), dtype=tl.int64))
    r158_value = tl.broadcast_to(r158_value, [$(1024)])
    tl.store($(candidate_vgt_5_ptr) + r158_z, r158_value)
    r159_z = tl.arange($(8192))
    r159_value = tl.where(tl.load($(candidate_vgt_5_ptr) + ((((r159_z / $(64)) % $(16)) * $(64)) + (r159_z % $(64))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_28_ptr) + ((((r159_z / $(1024)) % $(8)) * $(1024)) + (((r159_z / $(64)) % $(16)) * $(64)) + (r159_z % $(64))), dtype=tl.float32))
    r159_value = r159_value.to(tl.float32)
    r159_value = tl.broadcast_to(r159_value, [$(8192)])
    tl.store($(candidate_vmasked_fill_5_ptr) + r159_z, r159_value)
    r160_z = tl.arange($(8192))
    r160_lane = tl.arange($(64))
    r160_row_values = tl.load($(candidate_vmasked_fill_5_ptr) + ((((r160_z[:, None] / $(1024)) % $(8)) * $(1024)) + (((r160_z[:, None] / $(64)) % $(16)) * $(64)) + (r160_lane[None, :] % $(64))), dtype=tl.float32)
    r160_maximum = tl.max(r160_row_values, axis=1, keep_dims=true)
    r160_weights = tl.exp(r160_row_values - r160_maximum)
    r160_total = tl.sum(r160_weights, axis=1)
    r160_maximum = tl.max(r160_row_values, axis=1)
    r160_value = tl.exp(tl.load($(candidate_vmasked_fill_5_ptr) + ((((r160_z / $(1024)) % $(8)) * $(1024)) + (((r160_z / $(64)) % $(16)) * $(64)) + (r160_z % $(64))), dtype=tl.float32) - r160_maximum) / r160_total
    r160_value = r160_value.to(tl.float32)
    r160_value = tl.broadcast_to(r160_value, [$(8192)])
    tl.store($(candidate_vsoftmax_5_ptr) + r160_z, r160_value)
    r161_z = tl.arange($(8192))
    r161_value = tl.load($(candidate_vsoftmax_5_ptr) + ((((r161_z / $(1024)) % $(8)) * $(1024)) + (((r161_z / $(64)) % $(16)) * $(64)) + (r161_z % $(64))), dtype=tl.float32)
    r161_value = r161_value.to(tl.bfloat16)
    r161_value = tl.broadcast_to(r161_value, [$(8192)])
    tl.store($(candidate_vto_copy_58_ptr) + r161_z, r161_value)
    r162_row = tl.arange($(128))
    r162_col = tl.arange($(512))
    r162_lane = tl.arange($(64))
    r162_left = tl.load($(candidate_vto_copy_58_ptr) + ((((((r162_row[:, None] * $(64)) + r162_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + (((((r162_row[:, None] * $(64)) + r162_lane[None, :]) / $(64)) % $(16)) * $(64)) + (((r162_row[:, None] * $(64)) + r162_lane[None, :]) % $(64))), dtype=tl.bfloat16)
    r162_right = tl.load($(candidate_vindex_2_ptr) + ((((((r162_lane[:, None] * $(512)) + r162_col[None, :]) / $(512)) % $(64)) * $(576)) + (((r162_lane[:, None] * $(512)) + r162_col[None, :]) % $(512)) + $(36864)), dtype=tl.bfloat16)
    r162_value = tl.dot(r162_left, r162_right)
    r162_value = r162_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_14_ptr) + ((r162_row[:, None] * $(512)) + r162_col[None, :]), r162_value)
    r163_z = tl.arange($(131072))
    r163_value = tl.where(((r163_z / $(4096)) % $(32)) < $(12), tl.load($(candidate_vmm_10_ptr) + (((((((r163_z / $(512)) % $(8)) * $(12)) + (((r163_z / $(4096)) % $(32)) - $(0))) % $(96)) * $(512)) + (r163_z % $(512))), dtype=tl.bfloat16), tl.where(((r163_z / $(4096)) % $(32)) < $(16), tl.load($(candidate_vmm_12_ptr) + (((((((r163_z / $(512)) % $(8)) * $(4)) + (((r163_z / $(4096)) % $(32)) - $(12))) % $(32)) * $(512)) + (r163_z % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vmm_14_ptr) + (((((((r163_z / $(512)) % $(8)) * $(16)) + (((r163_z / $(4096)) % $(32)) - $(16))) % $(128)) * $(512)) + (r163_z % $(512))), dtype=tl.bfloat16)))
    r163_value = r163_value.to(tl.bfloat16)
    r163_value = tl.broadcast_to(r163_value, [$(131072)])
    tl.store($(candidate_vcat_7_ptr) + r163_z, r163_value)
    r164_row = tl.arange($(32))
    r164_col = tl.arange($(128))
    r164_lane = tl.arange($(512))
    for r164_group in range($(8)) {
      r164_left = tl.load($(candidate_vcat_7_ptr) + (((((((r164_group % $(8)) * $(16384)) + (r164_row[:, None] * $(512)) + r164_lane[None, :]) / $(512)) % $(32)) * $(4096)) + ((((((r164_group % $(8)) * $(16384)) + (r164_row[:, None] * $(512)) + r164_lane[None, :]) / $(16384)) % $(8)) * $(512)) + ((((r164_group % $(8)) * $(16384)) + (r164_row[:, None] * $(512)) + r164_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r164_right = tl.load($(candidate_vto_copy_5_ptr) + (((((((r164_group % $(8)) * $(65536)) + (r164_lane[:, None] * $(128)) + r164_col[None, :]) / $(65536)) % $(8)) * $(131072)) + (((((r164_group % $(8)) * $(65536)) + (r164_lane[:, None] * $(128)) + r164_col[None, :]) % $(128)) * $(512)) + (((((r164_group % $(8)) * $(65536)) + (r164_lane[:, None] * $(128)) + r164_col[None, :]) / $(128)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r164_value = tl.dot(r164_left, r164_right)
      r164_value = r164_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_11_ptr) + ((r164_group * $(4096)) + (r164_row[:, None] * $(128)) + r164_col[None, :]), r164_value)
    }
    r165_row = tl.arange($(32))
    r165_col = tl.arange($(512))
    r165_lane = tl.arange($(128))
    for r165_group in range($(8)) {
      r165_left = tl.load($(candidate_vmm_1_ptr) + (((((((r165_group % $(8)) * $(4096)) + (r165_row[:, None] * $(128)) + r165_lane[None, :]) / $(128)) % $(32)) * $(1536)) + ((((((r165_group % $(8)) * $(4096)) + (r165_row[:, None] * $(128)) + r165_lane[None, :]) / $(4096)) % $(8)) * $(192)) + ((((r165_group % $(8)) * $(4096)) + (r165_row[:, None] * $(128)) + r165_lane[None, :]) % $(128)) + $(98304)), dtype=tl.bfloat16)
      r165_right = tl.load($(candidate_vto_copy_5_ptr) + (((((((r165_group % $(8)) * $(65536)) + (r165_lane[:, None] * $(512)) + r165_col[None, :]) / $(65536)) % $(8)) * $(131072)) + ((((((r165_group % $(8)) * $(65536)) + (r165_lane[:, None] * $(512)) + r165_col[None, :]) / $(512)) % $(128)) * $(512)) + ((((r165_group % $(8)) * $(65536)) + (r165_lane[:, None] * $(512)) + r165_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r165_value = tl.dot(r165_left, r165_right)
      r165_value = r165_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_12_ptr) + ((r165_group * $(16384)) + (r165_row[:, None] * $(512)) + r165_col[None, :]), r165_value)
    }
    r166_z = tl.arange($(6144))
    r166_value = tl.load($(candidate_vadd_6_ptr) + ((((r166_z / $(64)) % $(12)) * $(512)) + (((r166_z / $(768)) % $(8)) * $(64)) + (r166_z % $(64)) + $(32768)), dtype=tl.bfloat16)
    r166_value = r166_value.to(tl.float32)
    r166_value = tl.broadcast_to(r166_value, [$(6144)])
    tl.store($(candidate_vto_copy_59_ptr) + r166_z, r166_value)
    r167_z = tl.arange($(1536))
    r167_value = tl.load($(candidate_vindex_2_ptr) + ((((r167_z / $(64)) % $(24)) * $(576)) + (r167_z % $(64)) + $(512)), dtype=tl.bfloat16)
    r167_value = r167_value.to(tl.float32)
    r167_value = tl.broadcast_to(r167_value, [$(1536)])
    tl.store($(candidate_vto_copy_60_ptr) + r167_z, r167_value)
    r168_row = tl.arange($(12))
    r168_col = tl.arange($(24))
    r168_lane = tl.arange($(64))
    for r168_group in range($(8)) {
      r168_left = tl.load($(candidate_vto_copy_59_ptr) + (((((((r168_group % $(8)) * $(768)) + (r168_row[:, None] * $(64)) + r168_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r168_group % $(8)) * $(768)) + (r168_row[:, None] * $(64)) + r168_lane[None, :]) / $(64)) % $(12)) * $(64)) + ((((r168_group % $(8)) * $(768)) + (r168_row[:, None] * $(64)) + r168_lane[None, :]) % $(64))), dtype=tl.float32)
      r168_right = tl.load($(candidate_vto_copy_60_ptr) + ((((((r168_group % $(8)) * $(1536)) + (r168_lane[:, None] * $(24)) + r168_col[None, :]) % $(24)) * $(64)) + (((((r168_group % $(8)) * $(1536)) + (r168_lane[:, None] * $(24)) + r168_col[None, :]) / $(24)) % $(64))), dtype=tl.float32)
      r168_value = tl.dot(r168_left, r168_right)
      r168_value = r168_value.to(tl.float32)
      tl.store($(candidate_vbmm_13_ptr) + ((r168_group * $(288)) + (r168_row[:, None] * $(24)) + r168_col[None, :]), r168_value)
    }
    r169_z = tl.arange($(49152))
    r169_value = tl.load($(candidate_vbmm_12_ptr) + ((((r169_z / $(6144)) % $(8)) * $(16384)) + (((r169_z / $(512)) % $(12)) * $(512)) + (r169_z % $(512))), dtype=tl.bfloat16)
    r169_value = r169_value.to(tl.float32)
    r169_value = tl.broadcast_to(r169_value, [$(49152)])
    tl.store($(candidate_vto_copy_61_ptr) + r169_z, r169_value)
    r170_z = tl.arange($(12288))
    r170_value = tl.load($(candidate_vindex_2_ptr) + ((((r170_z / $(512)) % $(24)) * $(576)) + (r170_z % $(512))), dtype=tl.bfloat16)
    r170_value = r170_value.to(tl.float32)
    r170_value = tl.broadcast_to(r170_value, [$(12288)])
    tl.store($(candidate_vto_copy_62_ptr) + r170_z, r170_value)
    r171_row = tl.arange($(96))
    r171_col = tl.arange($(24))
    r171_lane = tl.arange($(512))
    r171_left = tl.load($(candidate_vto_copy_61_ptr) + ((((((r171_row[:, None] * $(512)) + r171_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r171_row[:, None] * $(512)) + r171_lane[None, :]) / $(512)) % $(12)) * $(512)) + (((r171_row[:, None] * $(512)) + r171_lane[None, :]) % $(512))), dtype=tl.float32)
    r171_right = tl.load($(candidate_vto_copy_62_ptr) + (((((r171_lane[:, None] * $(24)) + r171_col[None, :]) % $(24)) * $(512)) + ((((r171_lane[:, None] * $(24)) + r171_col[None, :]) / $(24)) % $(512))), dtype=tl.float32)
    r171_value = tl.dot(r171_left, r171_right)
    r171_value = r171_value.to(tl.float32)
    tl.store($(candidate_vmm_15_ptr) + ((r171_row[:, None] * $(24)) + r171_col[None, :]), r171_value)
    r172_z = tl.arange($(2304))
    r172_value = (tl.load($(candidate_vbmm_13_ptr) + ((((r172_z / $(288)) % $(8)) * $(288)) + (((r172_z / $(24)) % $(12)) * $(24)) + (r172_z % $(24))), dtype=tl.float32) + tl.load($(candidate_vmm_15_ptr) + ((((r172_z / $(288)) % $(8)) * $(288)) + (((r172_z / $(24)) % $(12)) * $(24)) + (r172_z % $(24))), dtype=tl.float32))
    r172_value = r172_value.to(tl.float32)
    r172_value = tl.broadcast_to(r172_value, [$(2304)])
    tl.store($(candidate_vadd_20_ptr) + r172_z, r172_value)
    r173_z = tl.arange($(2304))
    r173_value = (tl.load($(candidate_vadd_20_ptr) + ((((r173_z / $(288)) % $(8)) * $(288)) + (((r173_z / $(24)) % $(12)) * $(24)) + (r173_z % $(24))), dtype=tl.float32) * 0.07216878364870322)
    r173_value = r173_value.to(tl.float32)
    r173_value = tl.broadcast_to(r173_value, [$(2304)])
    tl.store($(candidate_vmul_29_ptr) + r173_z, r173_value)
    r174_z = tl.arange($(12))
    r174_value = (tl.cast(r174_z, tl.int64) + tl.full([], $(24), dtype=tl.int64))
    r174_value = tl.broadcast_to(r174_value, [$(12)])
    tl.store($(candidate_vadd_21_ptr) + r174_z, r174_value)
    r175_z = tl.arange($(12))
    r175_value = (tl.load($(candidate_vadd_21_ptr) + r175_z, dtype=tl.int64) - tl.full([], $(12), dtype=tl.int64))
    r175_value = tl.broadcast_to(r175_value, [$(12)])
    tl.store($(candidate_vsub_15_ptr) + r175_z, r175_value)
    r176_z = tl.arange($(288))
    r176_value = (tl.cast((r176_z % $(24)), tl.int64) > tl.load($(candidate_vsub_15_ptr) + ((r176_z / $(24)) % $(12)), dtype=tl.int64))
    r176_value = tl.broadcast_to(r176_value, [$(288)])
    tl.store($(candidate_vgt_6_ptr) + r176_z, r176_value)
    r177_z = tl.arange($(2304))
    r177_value = tl.where(tl.load($(candidate_vgt_6_ptr) + ((((r177_z / $(24)) % $(12)) * $(24)) + (r177_z % $(24))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_29_ptr) + ((((r177_z / $(288)) % $(8)) * $(288)) + (((r177_z / $(24)) % $(12)) * $(24)) + (r177_z % $(24))), dtype=tl.float32))
    r177_value = r177_value.to(tl.float32)
    r177_value = tl.broadcast_to(r177_value, [$(2304)])
    tl.store($(candidate_vmasked_fill_6_ptr) + r177_z, r177_value)
    r178_z = tl.arange($(2304))
    r178_lane = tl.arange($(24))
    r178_row_values = tl.load($(candidate_vmasked_fill_6_ptr) + ((((r178_z[:, None] / $(288)) % $(8)) * $(288)) + (((r178_z[:, None] / $(24)) % $(12)) * $(24)) + (r178_lane[None, :] % $(24))), dtype=tl.float32)
    r178_maximum = tl.max(r178_row_values, axis=1, keep_dims=true)
    r178_weights = tl.exp(r178_row_values - r178_maximum)
    r178_total = tl.sum(r178_weights, axis=1)
    r178_maximum = tl.max(r178_row_values, axis=1)
    r178_value = tl.exp(tl.load($(candidate_vmasked_fill_6_ptr) + ((((r178_z / $(288)) % $(8)) * $(288)) + (((r178_z / $(24)) % $(12)) * $(24)) + (r178_z % $(24))), dtype=tl.float32) - r178_maximum) / r178_total
    r178_value = r178_value.to(tl.float32)
    r178_value = tl.broadcast_to(r178_value, [$(2304)])
    tl.store($(candidate_vsoftmax_6_ptr) + r178_z, r178_value)
    r179_z = tl.arange($(2304))
    r179_value = tl.load($(candidate_vsoftmax_6_ptr) + ((((r179_z / $(288)) % $(8)) * $(288)) + (((r179_z / $(24)) % $(12)) * $(24)) + (r179_z % $(24))), dtype=tl.float32)
    r179_value = r179_value.to(tl.bfloat16)
    r179_value = tl.broadcast_to(r179_value, [$(2304)])
    tl.store($(candidate_vto_copy_63_ptr) + r179_z, r179_value)
    r180_row = tl.arange($(96))
    r180_col = tl.arange($(512))
    r180_lane = tl.arange($(24))
    r180_left = tl.load($(candidate_vto_copy_63_ptr) + ((((((r180_row[:, None] * $(24)) + r180_lane[None, :]) / $(288)) % $(8)) * $(288)) + (((((r180_row[:, None] * $(24)) + r180_lane[None, :]) / $(24)) % $(12)) * $(24)) + (((r180_row[:, None] * $(24)) + r180_lane[None, :]) % $(24))), dtype=tl.bfloat16)
    r180_right = tl.load($(candidate_vindex_2_ptr) + ((((((r180_lane[:, None] * $(512)) + r180_col[None, :]) / $(512)) % $(24)) * $(576)) + (((r180_lane[:, None] * $(512)) + r180_col[None, :]) % $(512))), dtype=tl.bfloat16)
    r180_value = tl.dot(r180_left, r180_right)
    r180_value = r180_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_16_ptr) + ((r180_row[:, None] * $(512)) + r180_col[None, :]), r180_value)
    r181_z = tl.arange($(2048))
    r181_value = tl.load($(candidate_vadd_6_ptr) + ((((r181_z / $(64)) % $(4)) * $(512)) + (((r181_z / $(256)) % $(8)) * $(64)) + (r181_z % $(64)) + $(38912)), dtype=tl.bfloat16)
    r181_value = r181_value.to(tl.float32)
    r181_value = tl.broadcast_to(r181_value, [$(2048)])
    tl.store($(candidate_vto_copy_64_ptr) + r181_z, r181_value)
    r182_z = tl.arange($(512))
    r182_value = tl.load($(candidate_vindex_2_ptr) + ((((r182_z / $(64)) % $(8)) * $(576)) + (r182_z % $(64)) + $(28160)), dtype=tl.bfloat16)
    r182_value = r182_value.to(tl.float32)
    r182_value = tl.broadcast_to(r182_value, [$(512)])
    tl.store($(candidate_vto_copy_65_ptr) + r182_z, r182_value)
    r183_row = tl.arange($(4))
    r183_col = tl.arange($(8))
    r183_lane = tl.arange($(64))
    for r183_group in range($(8)) {
      r183_left = tl.load($(candidate_vto_copy_64_ptr) + (((((((r183_group % $(8)) * $(256)) + (r183_row[:, None] * $(64)) + r183_lane[None, :]) / $(256)) % $(8)) * $(256)) + ((((((r183_group % $(8)) * $(256)) + (r183_row[:, None] * $(64)) + r183_lane[None, :]) / $(64)) % $(4)) * $(64)) + ((((r183_group % $(8)) * $(256)) + (r183_row[:, None] * $(64)) + r183_lane[None, :]) % $(64))), dtype=tl.float32)
      r183_right = tl.load($(candidate_vto_copy_65_ptr) + ((((((r183_group % $(8)) * $(512)) + (r183_lane[:, None] * $(8)) + r183_col[None, :]) % $(8)) * $(64)) + (((((r183_group % $(8)) * $(512)) + (r183_lane[:, None] * $(8)) + r183_col[None, :]) / $(8)) % $(64))), dtype=tl.float32)
      r183_value = tl.dot(r183_left, r183_right)
      r183_value = r183_value.to(tl.float32)
      tl.store($(candidate_vbmm_14_ptr) + ((r183_group * $(32)) + (r183_row[:, None] * $(8)) + r183_col[None, :]), r183_value)
    }
    r184_z = tl.arange($(16384))
    r184_value = tl.load($(candidate_vbmm_12_ptr) + ((((r184_z / $(2048)) % $(8)) * $(16384)) + (((r184_z / $(512)) % $(4)) * $(512)) + (r184_z % $(512)) + $(6144)), dtype=tl.bfloat16)
    r184_value = r184_value.to(tl.float32)
    r184_value = tl.broadcast_to(r184_value, [$(16384)])
    tl.store($(candidate_vto_copy_66_ptr) + r184_z, r184_value)
    r185_z = tl.arange($(4096))
    r185_value = tl.load($(candidate_vindex_2_ptr) + ((((r185_z / $(512)) % $(8)) * $(576)) + (r185_z % $(512)) + $(27648)), dtype=tl.bfloat16)
    r185_value = r185_value.to(tl.float32)
    r185_value = tl.broadcast_to(r185_value, [$(4096)])
    tl.store($(candidate_vto_copy_67_ptr) + r185_z, r185_value)
    r186_row = tl.arange($(32))
    r186_col = tl.arange($(8))
    r186_lane = tl.arange($(512))
    r186_left = tl.load($(candidate_vto_copy_66_ptr) + ((((((r186_row[:, None] * $(512)) + r186_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + (((((r186_row[:, None] * $(512)) + r186_lane[None, :]) / $(512)) % $(4)) * $(512)) + (((r186_row[:, None] * $(512)) + r186_lane[None, :]) % $(512))), dtype=tl.float32)
    r186_right = tl.load($(candidate_vto_copy_67_ptr) + (((((r186_lane[:, None] * $(8)) + r186_col[None, :]) % $(8)) * $(512)) + ((((r186_lane[:, None] * $(8)) + r186_col[None, :]) / $(8)) % $(512))), dtype=tl.float32)
    r186_value = tl.dot(r186_left, r186_right)
    r186_value = r186_value.to(tl.float32)
    tl.store($(candidate_vmm_17_ptr) + ((r186_row[:, None] * $(8)) + r186_col[None, :]), r186_value)
    r187_z = tl.arange($(256))
    r187_value = (tl.load($(candidate_vbmm_14_ptr) + ((((r187_z / $(32)) % $(8)) * $(32)) + (((r187_z / $(8)) % $(4)) * $(8)) + (r187_z % $(8))), dtype=tl.float32) + tl.load($(candidate_vmm_17_ptr) + ((((r187_z / $(32)) % $(8)) * $(32)) + (((r187_z / $(8)) % $(4)) * $(8)) + (r187_z % $(8))), dtype=tl.float32))
    r187_value = r187_value.to(tl.float32)
    r187_value = tl.broadcast_to(r187_value, [$(256)])
    tl.store($(candidate_vadd_22_ptr) + r187_z, r187_value)
    r188_z = tl.arange($(256))
    r188_value = (tl.load($(candidate_vadd_22_ptr) + ((((r188_z / $(32)) % $(8)) * $(32)) + (((r188_z / $(8)) % $(4)) * $(8)) + (r188_z % $(8))), dtype=tl.float32) * 0.07216878364870322)
    r188_value = r188_value.to(tl.float32)
    r188_value = tl.broadcast_to(r188_value, [$(256)])
    tl.store($(candidate_vmul_30_ptr) + r188_z, r188_value)
    r189_z = tl.arange($(4))
    r189_value = (tl.cast(r189_z, tl.int64) + tl.full([], $(8), dtype=tl.int64))
    r189_value = tl.broadcast_to(r189_value, [$(4)])
    tl.store($(candidate_vadd_23_ptr) + r189_z, r189_value)
    r190_z = tl.arange($(4))
    r190_value = (tl.load($(candidate_vadd_23_ptr) + r190_z, dtype=tl.int64) - tl.full([], $(4), dtype=tl.int64))
    r190_value = tl.broadcast_to(r190_value, [$(4)])
    tl.store($(candidate_vsub_16_ptr) + r190_z, r190_value)
    r191_z = tl.arange($(32))
    r191_value = (tl.cast((r191_z % $(8)), tl.int64) > tl.load($(candidate_vsub_16_ptr) + ((r191_z / $(8)) % $(4)), dtype=tl.int64))
    r191_value = tl.broadcast_to(r191_value, [$(32)])
    tl.store($(candidate_vgt_7_ptr) + r191_z, r191_value)
    r192_z = tl.arange($(256))
    r192_value = tl.where(tl.load($(candidate_vgt_7_ptr) + ((((r192_z / $(8)) % $(4)) * $(8)) + (r192_z % $(8))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_30_ptr) + ((((r192_z / $(32)) % $(8)) * $(32)) + (((r192_z / $(8)) % $(4)) * $(8)) + (r192_z % $(8))), dtype=tl.float32))
    r192_value = r192_value.to(tl.float32)
    r192_value = tl.broadcast_to(r192_value, [$(256)])
    tl.store($(candidate_vmasked_fill_7_ptr) + r192_z, r192_value)
    r193_z = tl.arange($(256))
    r193_lane = tl.arange($(8))
    r193_row_values = tl.load($(candidate_vmasked_fill_7_ptr) + ((((r193_z[:, None] / $(32)) % $(8)) * $(32)) + (((r193_z[:, None] / $(8)) % $(4)) * $(8)) + (r193_lane[None, :] % $(8))), dtype=tl.float32)
    r193_maximum = tl.max(r193_row_values, axis=1, keep_dims=true)
    r193_weights = tl.exp(r193_row_values - r193_maximum)
    r193_total = tl.sum(r193_weights, axis=1)
    r193_maximum = tl.max(r193_row_values, axis=1)
    r193_value = tl.exp(tl.load($(candidate_vmasked_fill_7_ptr) + ((((r193_z / $(32)) % $(8)) * $(32)) + (((r193_z / $(8)) % $(4)) * $(8)) + (r193_z % $(8))), dtype=tl.float32) - r193_maximum) / r193_total
    r193_value = r193_value.to(tl.float32)
    r193_value = tl.broadcast_to(r193_value, [$(256)])
    tl.store($(candidate_vsoftmax_7_ptr) + r193_z, r193_value)
    r194_z = tl.arange($(256))
    r194_value = tl.load($(candidate_vsoftmax_7_ptr) + ((((r194_z / $(32)) % $(8)) * $(32)) + (((r194_z / $(8)) % $(4)) * $(8)) + (r194_z % $(8))), dtype=tl.float32)
    r194_value = r194_value.to(tl.bfloat16)
    r194_value = tl.broadcast_to(r194_value, [$(256)])
    tl.store($(candidate_vto_copy_68_ptr) + r194_z, r194_value)
    r195_row = tl.arange($(32))
    r195_col = tl.arange($(512))
    r195_lane = tl.arange($(8))
    r195_left = tl.load($(candidate_vto_copy_68_ptr) + ((((((r195_row[:, None] * $(8)) + r195_lane[None, :]) / $(32)) % $(8)) * $(32)) + (((((r195_row[:, None] * $(8)) + r195_lane[None, :]) / $(8)) % $(4)) * $(8)) + (((r195_row[:, None] * $(8)) + r195_lane[None, :]) % $(8))), dtype=tl.bfloat16)
    r195_right = tl.load($(candidate_vindex_2_ptr) + ((((((r195_lane[:, None] * $(512)) + r195_col[None, :]) / $(512)) % $(8)) * $(576)) + (((r195_lane[:, None] * $(512)) + r195_col[None, :]) % $(512)) + $(27648)), dtype=tl.bfloat16)
    r195_value = tl.dot(r195_left, r195_right)
    r195_value = r195_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_18_ptr) + ((r195_row[:, None] * $(512)) + r195_col[None, :]), r195_value)
    r196_z = tl.arange($(8192))
    r196_value = tl.load($(candidate_vadd_6_ptr) + ((((r196_z / $(64)) % $(16)) * $(512)) + (((r196_z / $(1024)) % $(8)) * $(64)) + (r196_z % $(64)) + $(40960)), dtype=tl.bfloat16)
    r196_value = r196_value.to(tl.float32)
    r196_value = tl.broadcast_to(r196_value, [$(8192)])
    tl.store($(candidate_vto_copy_69_ptr) + r196_z, r196_value)
    r197_z = tl.arange($(2048))
    r197_value = tl.load($(candidate_vindex_2_ptr) + ((((r197_z / $(64)) % $(32)) * $(576)) + (r197_z % $(64)) + $(37376)), dtype=tl.bfloat16)
    r197_value = r197_value.to(tl.float32)
    r197_value = tl.broadcast_to(r197_value, [$(2048)])
    tl.store($(candidate_vto_copy_70_ptr) + r197_z, r197_value)
    r198_row = tl.arange($(16))
    r198_col = tl.arange($(32))
    r198_lane = tl.arange($(64))
    for r198_group in range($(8)) {
      r198_left = tl.load($(candidate_vto_copy_69_ptr) + (((((((r198_group % $(8)) * $(1024)) + (r198_row[:, None] * $(64)) + r198_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + ((((((r198_group % $(8)) * $(1024)) + (r198_row[:, None] * $(64)) + r198_lane[None, :]) / $(64)) % $(16)) * $(64)) + ((((r198_group % $(8)) * $(1024)) + (r198_row[:, None] * $(64)) + r198_lane[None, :]) % $(64))), dtype=tl.float32)
      r198_right = tl.load($(candidate_vto_copy_70_ptr) + ((((((r198_group % $(8)) * $(2048)) + (r198_lane[:, None] * $(32)) + r198_col[None, :]) % $(32)) * $(64)) + (((((r198_group % $(8)) * $(2048)) + (r198_lane[:, None] * $(32)) + r198_col[None, :]) / $(32)) % $(64))), dtype=tl.float32)
      r198_value = tl.dot(r198_left, r198_right)
      r198_value = r198_value.to(tl.float32)
      tl.store($(candidate_vbmm_15_ptr) + ((r198_group * $(512)) + (r198_row[:, None] * $(32)) + r198_col[None, :]), r198_value)
    }
    r199_z = tl.arange($(65536))
    r199_value = tl.load($(candidate_vbmm_12_ptr) + ((((r199_z / $(8192)) % $(8)) * $(16384)) + (((r199_z / $(512)) % $(16)) * $(512)) + (r199_z % $(512)) + $(8192)), dtype=tl.bfloat16)
    r199_value = r199_value.to(tl.float32)
    r199_value = tl.broadcast_to(r199_value, [$(65536)])
    tl.store($(candidate_vto_copy_71_ptr) + r199_z, r199_value)
    r200_z = tl.arange($(16384))
    r200_value = tl.load($(candidate_vindex_2_ptr) + ((((r200_z / $(512)) % $(32)) * $(576)) + (r200_z % $(512)) + $(36864)), dtype=tl.bfloat16)
    r200_value = r200_value.to(tl.float32)
    r200_value = tl.broadcast_to(r200_value, [$(16384)])
    tl.store($(candidate_vto_copy_72_ptr) + r200_z, r200_value)
    r201_row = tl.arange($(128))
    r201_col = tl.arange($(32))
    r201_lane = tl.arange($(512))
    r201_left = tl.load($(candidate_vto_copy_71_ptr) + ((((((r201_row[:, None] * $(512)) + r201_lane[None, :]) / $(8192)) % $(8)) * $(8192)) + (((((r201_row[:, None] * $(512)) + r201_lane[None, :]) / $(512)) % $(16)) * $(512)) + (((r201_row[:, None] * $(512)) + r201_lane[None, :]) % $(512))), dtype=tl.float32)
    r201_right = tl.load($(candidate_vto_copy_72_ptr) + (((((r201_lane[:, None] * $(32)) + r201_col[None, :]) % $(32)) * $(512)) + ((((r201_lane[:, None] * $(32)) + r201_col[None, :]) / $(32)) % $(512))), dtype=tl.float32)
    r201_value = tl.dot(r201_left, r201_right)
    r201_value = r201_value.to(tl.float32)
    tl.store($(candidate_vmm_19_ptr) + ((r201_row[:, None] * $(32)) + r201_col[None, :]), r201_value)
    r202_z = tl.arange($(4096))
    r202_value = (tl.load($(candidate_vbmm_15_ptr) + ((((r202_z / $(512)) % $(8)) * $(512)) + (((r202_z / $(32)) % $(16)) * $(32)) + (r202_z % $(32))), dtype=tl.float32) + tl.load($(candidate_vmm_19_ptr) + ((((r202_z / $(512)) % $(8)) * $(512)) + (((r202_z / $(32)) % $(16)) * $(32)) + (r202_z % $(32))), dtype=tl.float32))
    r202_value = r202_value.to(tl.float32)
    r202_value = tl.broadcast_to(r202_value, [$(4096)])
    tl.store($(candidate_vadd_24_ptr) + r202_z, r202_value)
    r203_z = tl.arange($(4096))
    r203_value = (tl.load($(candidate_vadd_24_ptr) + ((((r203_z / $(512)) % $(8)) * $(512)) + (((r203_z / $(32)) % $(16)) * $(32)) + (r203_z % $(32))), dtype=tl.float32) * 0.07216878364870322)
    r203_value = r203_value.to(tl.float32)
    r203_value = tl.broadcast_to(r203_value, [$(4096)])
    tl.store($(candidate_vmul_31_ptr) + r203_z, r203_value)
    r204_z = tl.arange($(16))
    r204_value = (tl.cast(r204_z, tl.int64) + tl.full([], $(32), dtype=tl.int64))
    r204_value = tl.broadcast_to(r204_value, [$(16)])
    tl.store($(candidate_vadd_25_ptr) + r204_z, r204_value)
    r205_z = tl.arange($(16))
    r205_value = (tl.load($(candidate_vadd_25_ptr) + r205_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r205_value = tl.broadcast_to(r205_value, [$(16)])
    tl.store($(candidate_vsub_17_ptr) + r205_z, r205_value)
    r206_z = tl.arange($(512))
    r206_value = (tl.cast((r206_z % $(32)), tl.int64) > tl.load($(candidate_vsub_17_ptr) + ((r206_z / $(32)) % $(16)), dtype=tl.int64))
    r206_value = tl.broadcast_to(r206_value, [$(512)])
    tl.store($(candidate_vgt_8_ptr) + r206_z, r206_value)
    r207_z = tl.arange($(4096))
    r207_value = tl.where(tl.load($(candidate_vgt_8_ptr) + ((((r207_z / $(32)) % $(16)) * $(32)) + (r207_z % $(32))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_31_ptr) + ((((r207_z / $(512)) % $(8)) * $(512)) + (((r207_z / $(32)) % $(16)) * $(32)) + (r207_z % $(32))), dtype=tl.float32))
    r207_value = r207_value.to(tl.float32)
    r207_value = tl.broadcast_to(r207_value, [$(4096)])
    tl.store($(candidate_vmasked_fill_8_ptr) + r207_z, r207_value)
    r208_z = tl.arange($(4096))
    r208_lane = tl.arange($(32))
    r208_row_values = tl.load($(candidate_vmasked_fill_8_ptr) + ((((r208_z[:, None] / $(512)) % $(8)) * $(512)) + (((r208_z[:, None] / $(32)) % $(16)) * $(32)) + (r208_lane[None, :] % $(32))), dtype=tl.float32)
    r208_maximum = tl.max(r208_row_values, axis=1, keep_dims=true)
    r208_weights = tl.exp(r208_row_values - r208_maximum)
    r208_total = tl.sum(r208_weights, axis=1)
    r208_maximum = tl.max(r208_row_values, axis=1)
    r208_value = tl.exp(tl.load($(candidate_vmasked_fill_8_ptr) + ((((r208_z / $(512)) % $(8)) * $(512)) + (((r208_z / $(32)) % $(16)) * $(32)) + (r208_z % $(32))), dtype=tl.float32) - r208_maximum) / r208_total
    r208_value = r208_value.to(tl.float32)
    r208_value = tl.broadcast_to(r208_value, [$(4096)])
    tl.store($(candidate_vsoftmax_8_ptr) + r208_z, r208_value)
    r209_z = tl.arange($(4096))
    r209_value = tl.load($(candidate_vsoftmax_8_ptr) + ((((r209_z / $(512)) % $(8)) * $(512)) + (((r209_z / $(32)) % $(16)) * $(32)) + (r209_z % $(32))), dtype=tl.float32)
    r209_value = r209_value.to(tl.bfloat16)
    r209_value = tl.broadcast_to(r209_value, [$(4096)])
    tl.store($(candidate_vto_copy_73_ptr) + r209_z, r209_value)
    r210_row = tl.arange($(128))
    r210_col = tl.arange($(512))
    r210_lane = tl.arange($(32))
    r210_left = tl.load($(candidate_vto_copy_73_ptr) + ((((((r210_row[:, None] * $(32)) + r210_lane[None, :]) / $(512)) % $(8)) * $(512)) + (((((r210_row[:, None] * $(32)) + r210_lane[None, :]) / $(32)) % $(16)) * $(32)) + (((r210_row[:, None] * $(32)) + r210_lane[None, :]) % $(32))), dtype=tl.bfloat16)
    r210_right = tl.load($(candidate_vindex_2_ptr) + ((((((r210_lane[:, None] * $(512)) + r210_col[None, :]) / $(512)) % $(32)) * $(576)) + (((r210_lane[:, None] * $(512)) + r210_col[None, :]) % $(512)) + $(36864)), dtype=tl.bfloat16)
    r210_value = tl.dot(r210_left, r210_right)
    r210_value = r210_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_20_ptr) + ((r210_row[:, None] * $(512)) + r210_col[None, :]), r210_value)
    r211_z = tl.arange($(131072))
    r211_value = tl.where(((r211_z / $(4096)) % $(32)) < $(12), tl.load($(candidate_vmm_16_ptr) + (((((((r211_z / $(512)) % $(8)) * $(12)) + (((r211_z / $(4096)) % $(32)) - $(0))) % $(96)) * $(512)) + (r211_z % $(512))), dtype=tl.bfloat16), tl.where(((r211_z / $(4096)) % $(32)) < $(16), tl.load($(candidate_vmm_18_ptr) + (((((((r211_z / $(512)) % $(8)) * $(4)) + (((r211_z / $(4096)) % $(32)) - $(12))) % $(32)) * $(512)) + (r211_z % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vmm_20_ptr) + (((((((r211_z / $(512)) % $(8)) * $(16)) + (((r211_z / $(4096)) % $(32)) - $(16))) % $(128)) * $(512)) + (r211_z % $(512))), dtype=tl.bfloat16)))
    r211_value = r211_value.to(tl.bfloat16)
    r211_value = tl.broadcast_to(r211_value, [$(131072)])
    tl.store($(candidate_vcat_8_ptr) + r211_z, r211_value)
    r212_row = tl.arange($(32))
    r212_col = tl.arange($(128))
    r212_lane = tl.arange($(512))
    for r212_group in range($(8)) {
      r212_left = tl.load($(candidate_vcat_8_ptr) + (((((((r212_group % $(8)) * $(16384)) + (r212_row[:, None] * $(512)) + r212_lane[None, :]) / $(512)) % $(32)) * $(4096)) + ((((((r212_group % $(8)) * $(16384)) + (r212_row[:, None] * $(512)) + r212_lane[None, :]) / $(16384)) % $(8)) * $(512)) + ((((r212_group % $(8)) * $(16384)) + (r212_row[:, None] * $(512)) + r212_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r212_right = tl.load($(candidate_vto_copy_5_ptr) + (((((((r212_group % $(8)) * $(65536)) + (r212_lane[:, None] * $(128)) + r212_col[None, :]) / $(65536)) % $(8)) * $(131072)) + (((((r212_group % $(8)) * $(65536)) + (r212_lane[:, None] * $(128)) + r212_col[None, :]) % $(128)) * $(512)) + (((((r212_group % $(8)) * $(65536)) + (r212_lane[:, None] * $(128)) + r212_col[None, :]) / $(128)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r212_value = tl.dot(r212_left, r212_right)
      r212_value = r212_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_16_ptr) + ((r212_group * $(4096)) + (r212_row[:, None] * $(128)) + r212_col[None, :]), r212_value)
    }
    r213_row = tl.arange($(32))
    r213_col = tl.arange($(512))
    r213_lane = tl.arange($(128))
    for r213_group in range($(8)) {
      r213_left = tl.load($(candidate_vmm_1_ptr) + (((((((r213_group % $(8)) * $(4096)) + (r213_row[:, None] * $(128)) + r213_lane[None, :]) / $(128)) % $(32)) * $(1536)) + ((((((r213_group % $(8)) * $(4096)) + (r213_row[:, None] * $(128)) + r213_lane[None, :]) / $(4096)) % $(8)) * $(192)) + ((((r213_group % $(8)) * $(4096)) + (r213_row[:, None] * $(128)) + r213_lane[None, :]) % $(128)) + $(147456)), dtype=tl.bfloat16)
      r213_right = tl.load($(candidate_vto_copy_5_ptr) + (((((((r213_group % $(8)) * $(65536)) + (r213_lane[:, None] * $(512)) + r213_col[None, :]) / $(65536)) % $(8)) * $(131072)) + ((((((r213_group % $(8)) * $(65536)) + (r213_lane[:, None] * $(512)) + r213_col[None, :]) / $(512)) % $(128)) * $(512)) + ((((r213_group % $(8)) * $(65536)) + (r213_lane[:, None] * $(512)) + r213_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r213_value = tl.dot(r213_left, r213_right)
      r213_value = r213_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_17_ptr) + ((r213_group * $(16384)) + (r213_row[:, None] * $(512)) + r213_col[None, :]), r213_value)
    }
    r214_z = tl.arange($(6144))
    r214_value = tl.load($(candidate_vadd_6_ptr) + ((((r214_z / $(64)) % $(12)) * $(512)) + (((r214_z / $(768)) % $(8)) * $(64)) + (r214_z % $(64)) + $(49152)), dtype=tl.bfloat16)
    r214_value = r214_value.to(tl.float32)
    r214_value = tl.broadcast_to(r214_value, [$(6144)])
    tl.store($(candidate_vto_copy_74_ptr) + r214_z, r214_value)
    r215_z = tl.arange($(2304))
    r215_value = tl.load($(candidate_vindex_2_ptr) + ((((r215_z / $(64)) % $(36)) * $(576)) + (r215_z % $(64)) + $(512)), dtype=tl.bfloat16)
    r215_value = r215_value.to(tl.float32)
    r215_value = tl.broadcast_to(r215_value, [$(2304)])
    tl.store($(candidate_vto_copy_75_ptr) + r215_z, r215_value)
    r216_row = tl.arange($(12))
    r216_col = tl.arange($(36))
    r216_lane = tl.arange($(64))
    for r216_group in range($(8)) {
      r216_left = tl.load($(candidate_vto_copy_74_ptr) + (((((((r216_group % $(8)) * $(768)) + (r216_row[:, None] * $(64)) + r216_lane[None, :]) / $(768)) % $(8)) * $(768)) + ((((((r216_group % $(8)) * $(768)) + (r216_row[:, None] * $(64)) + r216_lane[None, :]) / $(64)) % $(12)) * $(64)) + ((((r216_group % $(8)) * $(768)) + (r216_row[:, None] * $(64)) + r216_lane[None, :]) % $(64))), dtype=tl.float32)
      r216_right = tl.load($(candidate_vto_copy_75_ptr) + ((((((r216_group % $(8)) * $(2304)) + (r216_lane[:, None] * $(36)) + r216_col[None, :]) % $(36)) * $(64)) + (((((r216_group % $(8)) * $(2304)) + (r216_lane[:, None] * $(36)) + r216_col[None, :]) / $(36)) % $(64))), dtype=tl.float32)
      r216_value = tl.dot(r216_left, r216_right)
      r216_value = r216_value.to(tl.float32)
      tl.store($(candidate_vbmm_18_ptr) + ((r216_group * $(432)) + (r216_row[:, None] * $(36)) + r216_col[None, :]), r216_value)
    }
    r217_z = tl.arange($(49152))
    r217_value = tl.load($(candidate_vbmm_17_ptr) + ((((r217_z / $(6144)) % $(8)) * $(16384)) + (((r217_z / $(512)) % $(12)) * $(512)) + (r217_z % $(512))), dtype=tl.bfloat16)
    r217_value = r217_value.to(tl.float32)
    r217_value = tl.broadcast_to(r217_value, [$(49152)])
    tl.store($(candidate_vto_copy_76_ptr) + r217_z, r217_value)
    r218_z = tl.arange($(18432))
    r218_value = tl.load($(candidate_vindex_2_ptr) + ((((r218_z / $(512)) % $(36)) * $(576)) + (r218_z % $(512))), dtype=tl.bfloat16)
    r218_value = r218_value.to(tl.float32)
    r218_value = tl.broadcast_to(r218_value, [$(18432)])
    tl.store($(candidate_vto_copy_77_ptr) + r218_z, r218_value)
    r219_row = tl.arange($(96))
    r219_col = tl.arange($(36))
    r219_lane = tl.arange($(512))
    r219_left = tl.load($(candidate_vto_copy_76_ptr) + ((((((r219_row[:, None] * $(512)) + r219_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r219_row[:, None] * $(512)) + r219_lane[None, :]) / $(512)) % $(12)) * $(512)) + (((r219_row[:, None] * $(512)) + r219_lane[None, :]) % $(512))), dtype=tl.float32)
    r219_right = tl.load($(candidate_vto_copy_77_ptr) + (((((r219_lane[:, None] * $(36)) + r219_col[None, :]) % $(36)) * $(512)) + ((((r219_lane[:, None] * $(36)) + r219_col[None, :]) / $(36)) % $(512))), dtype=tl.float32)
    r219_value = tl.dot(r219_left, r219_right)
    r219_value = r219_value.to(tl.float32)
    tl.store($(candidate_vmm_21_ptr) + ((r219_row[:, None] * $(36)) + r219_col[None, :]), r219_value)
    r220_z = tl.arange($(3456))
    r220_value = (tl.load($(candidate_vbmm_18_ptr) + ((((r220_z / $(432)) % $(8)) * $(432)) + (((r220_z / $(36)) % $(12)) * $(36)) + (r220_z % $(36))), dtype=tl.float32) + tl.load($(candidate_vmm_21_ptr) + ((((r220_z / $(432)) % $(8)) * $(432)) + (((r220_z / $(36)) % $(12)) * $(36)) + (r220_z % $(36))), dtype=tl.float32))
    r220_value = r220_value.to(tl.float32)
    r220_value = tl.broadcast_to(r220_value, [$(3456)])
    tl.store($(candidate_vadd_26_ptr) + r220_z, r220_value)
    r221_z = tl.arange($(3456))
    r221_value = (tl.load($(candidate_vadd_26_ptr) + ((((r221_z / $(432)) % $(8)) * $(432)) + (((r221_z / $(36)) % $(12)) * $(36)) + (r221_z % $(36))), dtype=tl.float32) * 0.07216878364870322)
    r221_value = r221_value.to(tl.float32)
    r221_value = tl.broadcast_to(r221_value, [$(3456)])
    tl.store($(candidate_vmul_32_ptr) + r221_z, r221_value)
    r222_z = tl.arange($(12))
    r222_value = (tl.cast(r222_z, tl.int64) + tl.full([], $(36), dtype=tl.int64))
    r222_value = tl.broadcast_to(r222_value, [$(12)])
    tl.store($(candidate_vadd_27_ptr) + r222_z, r222_value)
    r223_z = tl.arange($(12))
    r223_value = (tl.load($(candidate_vadd_27_ptr) + r223_z, dtype=tl.int64) - tl.full([], $(12), dtype=tl.int64))
    r223_value = tl.broadcast_to(r223_value, [$(12)])
    tl.store($(candidate_vsub_18_ptr) + r223_z, r223_value)
    r224_z = tl.arange($(432))
    r224_value = (tl.cast((r224_z % $(36)), tl.int64) > tl.load($(candidate_vsub_18_ptr) + ((r224_z / $(36)) % $(12)), dtype=tl.int64))
    r224_value = tl.broadcast_to(r224_value, [$(432)])
    tl.store($(candidate_vgt_9_ptr) + r224_z, r224_value)
    r225_z = tl.arange($(3456))
    r225_value = tl.where(tl.load($(candidate_vgt_9_ptr) + ((((r225_z / $(36)) % $(12)) * $(36)) + (r225_z % $(36))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_32_ptr) + ((((r225_z / $(432)) % $(8)) * $(432)) + (((r225_z / $(36)) % $(12)) * $(36)) + (r225_z % $(36))), dtype=tl.float32))
    r225_value = r225_value.to(tl.float32)
    r225_value = tl.broadcast_to(r225_value, [$(3456)])
    tl.store($(candidate_vmasked_fill_9_ptr) + r225_z, r225_value)
    r226_z = tl.arange($(3456))
    r226_lane = tl.arange($(36))
    r226_row_values = tl.load($(candidate_vmasked_fill_9_ptr) + ((((r226_z[:, None] / $(432)) % $(8)) * $(432)) + (((r226_z[:, None] / $(36)) % $(12)) * $(36)) + (r226_lane[None, :] % $(36))), dtype=tl.float32)
    r226_maximum = tl.max(r226_row_values, axis=1, keep_dims=true)
    r226_weights = tl.exp(r226_row_values - r226_maximum)
    r226_total = tl.sum(r226_weights, axis=1)
    r226_maximum = tl.max(r226_row_values, axis=1)
    r226_value = tl.exp(tl.load($(candidate_vmasked_fill_9_ptr) + ((((r226_z / $(432)) % $(8)) * $(432)) + (((r226_z / $(36)) % $(12)) * $(36)) + (r226_z % $(36))), dtype=tl.float32) - r226_maximum) / r226_total
    r226_value = r226_value.to(tl.float32)
    r226_value = tl.broadcast_to(r226_value, [$(3456)])
    tl.store($(candidate_vsoftmax_9_ptr) + r226_z, r226_value)
    r227_z = tl.arange($(3456))
    r227_value = tl.load($(candidate_vsoftmax_9_ptr) + ((((r227_z / $(432)) % $(8)) * $(432)) + (((r227_z / $(36)) % $(12)) * $(36)) + (r227_z % $(36))), dtype=tl.float32)
    r227_value = r227_value.to(tl.bfloat16)
    r227_value = tl.broadcast_to(r227_value, [$(3456)])
    tl.store($(candidate_vto_copy_78_ptr) + r227_z, r227_value)
    r228_row = tl.arange($(96))
    r228_col = tl.arange($(512))
    r228_lane = tl.arange($(36))
    r228_left = tl.load($(candidate_vto_copy_78_ptr) + ((((((r228_row[:, None] * $(36)) + r228_lane[None, :]) / $(432)) % $(8)) * $(432)) + (((((r228_row[:, None] * $(36)) + r228_lane[None, :]) / $(36)) % $(12)) * $(36)) + (((r228_row[:, None] * $(36)) + r228_lane[None, :]) % $(36))), dtype=tl.bfloat16)
    r228_right = tl.load($(candidate_vindex_2_ptr) + ((((((r228_lane[:, None] * $(512)) + r228_col[None, :]) / $(512)) % $(36)) * $(576)) + (((r228_lane[:, None] * $(512)) + r228_col[None, :]) % $(512))), dtype=tl.bfloat16)
    r228_value = tl.dot(r228_left, r228_right)
    r228_value = r228_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_22_ptr) + ((r228_row[:, None] * $(512)) + r228_col[None, :]), r228_value)
    r229_z = tl.arange($(2048))
    r229_value = tl.load($(candidate_vadd_6_ptr) + ((((r229_z / $(64)) % $(4)) * $(512)) + (((r229_z / $(256)) % $(8)) * $(64)) + (r229_z % $(64)) + $(55296)), dtype=tl.bfloat16)
    r229_value = r229_value.to(tl.float32)
    r229_value = tl.broadcast_to(r229_value, [$(2048)])
    tl.store($(candidate_vto_copy_79_ptr) + r229_z, r229_value)
    r230_z = tl.arange($(768))
    r230_value = tl.load($(candidate_vindex_2_ptr) + ((((r230_z / $(64)) % $(12)) * $(576)) + (r230_z % $(64)) + $(28160)), dtype=tl.bfloat16)
    r230_value = r230_value.to(tl.float32)
    r230_value = tl.broadcast_to(r230_value, [$(768)])
    tl.store($(candidate_vto_copy_80_ptr) + r230_z, r230_value)
    r231_row = tl.arange($(4))
    r231_col = tl.arange($(12))
    r231_lane = tl.arange($(64))
    for r231_group in range($(8)) {
      r231_left = tl.load($(candidate_vto_copy_79_ptr) + (((((((r231_group % $(8)) * $(256)) + (r231_row[:, None] * $(64)) + r231_lane[None, :]) / $(256)) % $(8)) * $(256)) + ((((((r231_group % $(8)) * $(256)) + (r231_row[:, None] * $(64)) + r231_lane[None, :]) / $(64)) % $(4)) * $(64)) + ((((r231_group % $(8)) * $(256)) + (r231_row[:, None] * $(64)) + r231_lane[None, :]) % $(64))), dtype=tl.float32)
      r231_right = tl.load($(candidate_vto_copy_80_ptr) + ((((((r231_group % $(8)) * $(768)) + (r231_lane[:, None] * $(12)) + r231_col[None, :]) % $(12)) * $(64)) + (((((r231_group % $(8)) * $(768)) + (r231_lane[:, None] * $(12)) + r231_col[None, :]) / $(12)) % $(64))), dtype=tl.float32)
      r231_value = tl.dot(r231_left, r231_right)
      r231_value = r231_value.to(tl.float32)
      tl.store($(candidate_vbmm_19_ptr) + ((r231_group * $(48)) + (r231_row[:, None] * $(12)) + r231_col[None, :]), r231_value)
    }
    r232_z = tl.arange($(16384))
    r232_value = tl.load($(candidate_vbmm_17_ptr) + ((((r232_z / $(2048)) % $(8)) * $(16384)) + (((r232_z / $(512)) % $(4)) * $(512)) + (r232_z % $(512)) + $(6144)), dtype=tl.bfloat16)
    r232_value = r232_value.to(tl.float32)
    r232_value = tl.broadcast_to(r232_value, [$(16384)])
    tl.store($(candidate_vto_copy_81_ptr) + r232_z, r232_value)
    r233_z = tl.arange($(6144))
    r233_value = tl.load($(candidate_vindex_2_ptr) + ((((r233_z / $(512)) % $(12)) * $(576)) + (r233_z % $(512)) + $(27648)), dtype=tl.bfloat16)
    r233_value = r233_value.to(tl.float32)
    r233_value = tl.broadcast_to(r233_value, [$(6144)])
    tl.store($(candidate_vto_copy_82_ptr) + r233_z, r233_value)
    r234_row = tl.arange($(32))
    r234_col = tl.arange($(12))
    r234_lane = tl.arange($(512))
    r234_left = tl.load($(candidate_vto_copy_81_ptr) + ((((((r234_row[:, None] * $(512)) + r234_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + (((((r234_row[:, None] * $(512)) + r234_lane[None, :]) / $(512)) % $(4)) * $(512)) + (((r234_row[:, None] * $(512)) + r234_lane[None, :]) % $(512))), dtype=tl.float32)
    r234_right = tl.load($(candidate_vto_copy_82_ptr) + (((((r234_lane[:, None] * $(12)) + r234_col[None, :]) % $(12)) * $(512)) + ((((r234_lane[:, None] * $(12)) + r234_col[None, :]) / $(12)) % $(512))), dtype=tl.float32)
    r234_value = tl.dot(r234_left, r234_right)
    r234_value = r234_value.to(tl.float32)
    tl.store($(candidate_vmm_23_ptr) + ((r234_row[:, None] * $(12)) + r234_col[None, :]), r234_value)
    r235_z = tl.arange($(384))
    r235_value = (tl.load($(candidate_vbmm_19_ptr) + ((((r235_z / $(48)) % $(8)) * $(48)) + (((r235_z / $(12)) % $(4)) * $(12)) + (r235_z % $(12))), dtype=tl.float32) + tl.load($(candidate_vmm_23_ptr) + ((((r235_z / $(48)) % $(8)) * $(48)) + (((r235_z / $(12)) % $(4)) * $(12)) + (r235_z % $(12))), dtype=tl.float32))
    r235_value = r235_value.to(tl.float32)
    r235_value = tl.broadcast_to(r235_value, [$(384)])
    tl.store($(candidate_vadd_28_ptr) + r235_z, r235_value)
    r236_z = tl.arange($(384))
    r236_value = (tl.load($(candidate_vadd_28_ptr) + ((((r236_z / $(48)) % $(8)) * $(48)) + (((r236_z / $(12)) % $(4)) * $(12)) + (r236_z % $(12))), dtype=tl.float32) * 0.07216878364870322)
    r236_value = r236_value.to(tl.float32)
    r236_value = tl.broadcast_to(r236_value, [$(384)])
    tl.store($(candidate_vmul_33_ptr) + r236_z, r236_value)
    r237_z = tl.arange($(4))
    r237_value = (tl.cast(r237_z, tl.int64) + tl.full([], $(12), dtype=tl.int64))
    r237_value = tl.broadcast_to(r237_value, [$(4)])
    tl.store($(candidate_vadd_29_ptr) + r237_z, r237_value)
    r238_z = tl.arange($(4))
    r238_value = (tl.load($(candidate_vadd_29_ptr) + r238_z, dtype=tl.int64) - tl.full([], $(4), dtype=tl.int64))
    r238_value = tl.broadcast_to(r238_value, [$(4)])
    tl.store($(candidate_vsub_19_ptr) + r238_z, r238_value)
    r239_z = tl.arange($(48))
    r239_value = (tl.cast((r239_z % $(12)), tl.int64) > tl.load($(candidate_vsub_19_ptr) + ((r239_z / $(12)) % $(4)), dtype=tl.int64))
    r239_value = tl.broadcast_to(r239_value, [$(48)])
    tl.store($(candidate_vgt_10_ptr) + r239_z, r239_value)
    r240_z = tl.arange($(384))
    r240_value = tl.where(tl.load($(candidate_vgt_10_ptr) + ((((r240_z / $(12)) % $(4)) * $(12)) + (r240_z % $(12))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_33_ptr) + ((((r240_z / $(48)) % $(8)) * $(48)) + (((r240_z / $(12)) % $(4)) * $(12)) + (r240_z % $(12))), dtype=tl.float32))
    r240_value = r240_value.to(tl.float32)
    r240_value = tl.broadcast_to(r240_value, [$(384)])
    tl.store($(candidate_vmasked_fill_10_ptr) + r240_z, r240_value)
    r241_z = tl.arange($(384))
    r241_lane = tl.arange($(12))
    r241_row_values = tl.load($(candidate_vmasked_fill_10_ptr) + ((((r241_z[:, None] / $(48)) % $(8)) * $(48)) + (((r241_z[:, None] / $(12)) % $(4)) * $(12)) + (r241_lane[None, :] % $(12))), dtype=tl.float32)
    r241_maximum = tl.max(r241_row_values, axis=1, keep_dims=true)
    r241_weights = tl.exp(r241_row_values - r241_maximum)
    r241_total = tl.sum(r241_weights, axis=1)
    r241_maximum = tl.max(r241_row_values, axis=1)
    r241_value = tl.exp(tl.load($(candidate_vmasked_fill_10_ptr) + ((((r241_z / $(48)) % $(8)) * $(48)) + (((r241_z / $(12)) % $(4)) * $(12)) + (r241_z % $(12))), dtype=tl.float32) - r241_maximum) / r241_total
    r241_value = r241_value.to(tl.float32)
    r241_value = tl.broadcast_to(r241_value, [$(384)])
    tl.store($(candidate_vsoftmax_10_ptr) + r241_z, r241_value)
    r242_z = tl.arange($(384))
    r242_value = tl.load($(candidate_vsoftmax_10_ptr) + ((((r242_z / $(48)) % $(8)) * $(48)) + (((r242_z / $(12)) % $(4)) * $(12)) + (r242_z % $(12))), dtype=tl.float32)
    r242_value = r242_value.to(tl.bfloat16)
    r242_value = tl.broadcast_to(r242_value, [$(384)])
    tl.store($(candidate_vto_copy_83_ptr) + r242_z, r242_value)
    r243_row = tl.arange($(32))
    r243_col = tl.arange($(512))
    r243_lane = tl.arange($(12))
    r243_left = tl.load($(candidate_vto_copy_83_ptr) + ((((((r243_row[:, None] * $(12)) + r243_lane[None, :]) / $(48)) % $(8)) * $(48)) + (((((r243_row[:, None] * $(12)) + r243_lane[None, :]) / $(12)) % $(4)) * $(12)) + (((r243_row[:, None] * $(12)) + r243_lane[None, :]) % $(12))), dtype=tl.bfloat16)
    r243_right = tl.load($(candidate_vindex_2_ptr) + ((((((r243_lane[:, None] * $(512)) + r243_col[None, :]) / $(512)) % $(12)) * $(576)) + (((r243_lane[:, None] * $(512)) + r243_col[None, :]) % $(512)) + $(27648)), dtype=tl.bfloat16)
    r243_value = tl.dot(r243_left, r243_right)
    r243_value = r243_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_24_ptr) + ((r243_row[:, None] * $(512)) + r243_col[None, :]), r243_value)
    r244_z = tl.arange($(8192))
    r244_value = tl.load($(candidate_vadd_6_ptr) + ((((r244_z / $(64)) % $(16)) * $(512)) + (((r244_z / $(1024)) % $(8)) * $(64)) + (r244_z % $(64)) + $(57344)), dtype=tl.bfloat16)
    r244_value = r244_value.to(tl.float32)
    r244_value = tl.broadcast_to(r244_value, [$(8192)])
    tl.store($(candidate_vto_copy_84_ptr) + r244_z, r244_value)
    r245_z = tl.arange($(3072))
    r245_value = tl.load($(candidate_vindex_2_ptr) + ((((r245_z / $(64)) % $(48)) * $(576)) + (r245_z % $(64)) + $(37376)), dtype=tl.bfloat16)
    r245_value = r245_value.to(tl.float32)
    r245_value = tl.broadcast_to(r245_value, [$(3072)])
    tl.store($(candidate_vto_copy_85_ptr) + r245_z, r245_value)
    r246_row = tl.arange($(16))
    r246_col = tl.arange($(48))
    r246_lane = tl.arange($(64))
    for r246_group in range($(8)) {
      r246_left = tl.load($(candidate_vto_copy_84_ptr) + (((((((r246_group % $(8)) * $(1024)) + (r246_row[:, None] * $(64)) + r246_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + ((((((r246_group % $(8)) * $(1024)) + (r246_row[:, None] * $(64)) + r246_lane[None, :]) / $(64)) % $(16)) * $(64)) + ((((r246_group % $(8)) * $(1024)) + (r246_row[:, None] * $(64)) + r246_lane[None, :]) % $(64))), dtype=tl.float32)
      r246_right = tl.load($(candidate_vto_copy_85_ptr) + ((((((r246_group % $(8)) * $(3072)) + (r246_lane[:, None] * $(48)) + r246_col[None, :]) % $(48)) * $(64)) + (((((r246_group % $(8)) * $(3072)) + (r246_lane[:, None] * $(48)) + r246_col[None, :]) / $(48)) % $(64))), dtype=tl.float32)
      r246_value = tl.dot(r246_left, r246_right)
      r246_value = r246_value.to(tl.float32)
      tl.store($(candidate_vbmm_20_ptr) + ((r246_group * $(768)) + (r246_row[:, None] * $(48)) + r246_col[None, :]), r246_value)
    }
    r247_z = tl.arange($(65536))
    r247_value = tl.load($(candidate_vbmm_17_ptr) + ((((r247_z / $(8192)) % $(8)) * $(16384)) + (((r247_z / $(512)) % $(16)) * $(512)) + (r247_z % $(512)) + $(8192)), dtype=tl.bfloat16)
    r247_value = r247_value.to(tl.float32)
    r247_value = tl.broadcast_to(r247_value, [$(65536)])
    tl.store($(candidate_vto_copy_86_ptr) + r247_z, r247_value)
    r248_z = tl.arange($(24576))
    r248_value = tl.load($(candidate_vindex_2_ptr) + ((((r248_z / $(512)) % $(48)) * $(576)) + (r248_z % $(512)) + $(36864)), dtype=tl.bfloat16)
    r248_value = r248_value.to(tl.float32)
    r248_value = tl.broadcast_to(r248_value, [$(24576)])
    tl.store($(candidate_vto_copy_87_ptr) + r248_z, r248_value)
    r249_row = tl.arange($(128))
    r249_col = tl.arange($(48))
    r249_lane = tl.arange($(512))
    r249_left = tl.load($(candidate_vto_copy_86_ptr) + ((((((r249_row[:, None] * $(512)) + r249_lane[None, :]) / $(8192)) % $(8)) * $(8192)) + (((((r249_row[:, None] * $(512)) + r249_lane[None, :]) / $(512)) % $(16)) * $(512)) + (((r249_row[:, None] * $(512)) + r249_lane[None, :]) % $(512))), dtype=tl.float32)
    r249_right = tl.load($(candidate_vto_copy_87_ptr) + (((((r249_lane[:, None] * $(48)) + r249_col[None, :]) % $(48)) * $(512)) + ((((r249_lane[:, None] * $(48)) + r249_col[None, :]) / $(48)) % $(512))), dtype=tl.float32)
    r249_value = tl.dot(r249_left, r249_right)
    r249_value = r249_value.to(tl.float32)
    tl.store($(candidate_vmm_25_ptr) + ((r249_row[:, None] * $(48)) + r249_col[None, :]), r249_value)
    r250_z = tl.arange($(6144))
    r250_value = (tl.load($(candidate_vbmm_20_ptr) + ((((r250_z / $(768)) % $(8)) * $(768)) + (((r250_z / $(48)) % $(16)) * $(48)) + (r250_z % $(48))), dtype=tl.float32) + tl.load($(candidate_vmm_25_ptr) + ((((r250_z / $(768)) % $(8)) * $(768)) + (((r250_z / $(48)) % $(16)) * $(48)) + (r250_z % $(48))), dtype=tl.float32))
    r250_value = r250_value.to(tl.float32)
    r250_value = tl.broadcast_to(r250_value, [$(6144)])
    tl.store($(candidate_vadd_30_ptr) + r250_z, r250_value)
    r251_z = tl.arange($(6144))
    r251_value = (tl.load($(candidate_vadd_30_ptr) + ((((r251_z / $(768)) % $(8)) * $(768)) + (((r251_z / $(48)) % $(16)) * $(48)) + (r251_z % $(48))), dtype=tl.float32) * 0.07216878364870322)
    r251_value = r251_value.to(tl.float32)
    r251_value = tl.broadcast_to(r251_value, [$(6144)])
    tl.store($(candidate_vmul_34_ptr) + r251_z, r251_value)
    r252_z = tl.arange($(16))
    r252_value = (tl.cast(r252_z, tl.int64) + tl.full([], $(48), dtype=tl.int64))
    r252_value = tl.broadcast_to(r252_value, [$(16)])
    tl.store($(candidate_vadd_31_ptr) + r252_z, r252_value)
    r253_z = tl.arange($(16))
    r253_value = (tl.load($(candidate_vadd_31_ptr) + r253_z, dtype=tl.int64) - tl.full([], $(16), dtype=tl.int64))
    r253_value = tl.broadcast_to(r253_value, [$(16)])
    tl.store($(candidate_vsub_20_ptr) + r253_z, r253_value)
    r254_z = tl.arange($(768))
    r254_value = (tl.cast((r254_z % $(48)), tl.int64) > tl.load($(candidate_vsub_20_ptr) + ((r254_z / $(48)) % $(16)), dtype=tl.int64))
    r254_value = tl.broadcast_to(r254_value, [$(768)])
    tl.store($(candidate_vgt_11_ptr) + r254_z, r254_value)
    r255_z = tl.arange($(6144))
    r255_value = tl.where(tl.load($(candidate_vgt_11_ptr) + ((((r255_z / $(48)) % $(16)) * $(48)) + (r255_z % $(48))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_34_ptr) + ((((r255_z / $(768)) % $(8)) * $(768)) + (((r255_z / $(48)) % $(16)) * $(48)) + (r255_z % $(48))), dtype=tl.float32))
    r255_value = r255_value.to(tl.float32)
    r255_value = tl.broadcast_to(r255_value, [$(6144)])
    tl.store($(candidate_vmasked_fill_11_ptr) + r255_z, r255_value)
    r256_z = tl.arange($(6144))
    r256_lane = tl.arange($(48))
    r256_row_values = tl.load($(candidate_vmasked_fill_11_ptr) + ((((r256_z[:, None] / $(768)) % $(8)) * $(768)) + (((r256_z[:, None] / $(48)) % $(16)) * $(48)) + (r256_lane[None, :] % $(48))), dtype=tl.float32)
    r256_maximum = tl.max(r256_row_values, axis=1, keep_dims=true)
    r256_weights = tl.exp(r256_row_values - r256_maximum)
    r256_total = tl.sum(r256_weights, axis=1)
    r256_maximum = tl.max(r256_row_values, axis=1)
    r256_value = tl.exp(tl.load($(candidate_vmasked_fill_11_ptr) + ((((r256_z / $(768)) % $(8)) * $(768)) + (((r256_z / $(48)) % $(16)) * $(48)) + (r256_z % $(48))), dtype=tl.float32) - r256_maximum) / r256_total
    r256_value = r256_value.to(tl.float32)
    r256_value = tl.broadcast_to(r256_value, [$(6144)])
    tl.store($(candidate_vsoftmax_11_ptr) + r256_z, r256_value)
    r257_z = tl.arange($(6144))
    r257_value = tl.load($(candidate_vsoftmax_11_ptr) + ((((r257_z / $(768)) % $(8)) * $(768)) + (((r257_z / $(48)) % $(16)) * $(48)) + (r257_z % $(48))), dtype=tl.float32)
    r257_value = r257_value.to(tl.bfloat16)
    r257_value = tl.broadcast_to(r257_value, [$(6144)])
    tl.store($(candidate_vto_copy_88_ptr) + r257_z, r257_value)
    r258_row = tl.arange($(128))
    r258_col = tl.arange($(512))
    r258_lane = tl.arange($(48))
    r258_left = tl.load($(candidate_vto_copy_88_ptr) + ((((((r258_row[:, None] * $(48)) + r258_lane[None, :]) / $(768)) % $(8)) * $(768)) + (((((r258_row[:, None] * $(48)) + r258_lane[None, :]) / $(48)) % $(16)) * $(48)) + (((r258_row[:, None] * $(48)) + r258_lane[None, :]) % $(48))), dtype=tl.bfloat16)
    r258_right = tl.load($(candidate_vindex_2_ptr) + ((((((r258_lane[:, None] * $(512)) + r258_col[None, :]) / $(512)) % $(48)) * $(576)) + (((r258_lane[:, None] * $(512)) + r258_col[None, :]) % $(512)) + $(36864)), dtype=tl.bfloat16)
    r258_value = tl.dot(r258_left, r258_right)
    r258_value = r258_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_26_ptr) + ((r258_row[:, None] * $(512)) + r258_col[None, :]), r258_value)
    r259_z = tl.arange($(131072))
    r259_value = tl.where(((r259_z / $(4096)) % $(32)) < $(12), tl.load($(candidate_vmm_22_ptr) + (((((((r259_z / $(512)) % $(8)) * $(12)) + (((r259_z / $(4096)) % $(32)) - $(0))) % $(96)) * $(512)) + (r259_z % $(512))), dtype=tl.bfloat16), tl.where(((r259_z / $(4096)) % $(32)) < $(16), tl.load($(candidate_vmm_24_ptr) + (((((((r259_z / $(512)) % $(8)) * $(4)) + (((r259_z / $(4096)) % $(32)) - $(12))) % $(32)) * $(512)) + (r259_z % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vmm_26_ptr) + (((((((r259_z / $(512)) % $(8)) * $(16)) + (((r259_z / $(4096)) % $(32)) - $(16))) % $(128)) * $(512)) + (r259_z % $(512))), dtype=tl.bfloat16)))
    r259_value = r259_value.to(tl.bfloat16)
    r259_value = tl.broadcast_to(r259_value, [$(131072)])
    tl.store($(candidate_vcat_9_ptr) + r259_z, r259_value)
    r260_row = tl.arange($(32))
    r260_col = tl.arange($(128))
    r260_lane = tl.arange($(512))
    for r260_group in range($(8)) {
      r260_left = tl.load($(candidate_vcat_9_ptr) + (((((((r260_group % $(8)) * $(16384)) + (r260_row[:, None] * $(512)) + r260_lane[None, :]) / $(512)) % $(32)) * $(4096)) + ((((((r260_group % $(8)) * $(16384)) + (r260_row[:, None] * $(512)) + r260_lane[None, :]) / $(16384)) % $(8)) * $(512)) + ((((r260_group % $(8)) * $(16384)) + (r260_row[:, None] * $(512)) + r260_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r260_right = tl.load($(candidate_vto_copy_5_ptr) + (((((((r260_group % $(8)) * $(65536)) + (r260_lane[:, None] * $(128)) + r260_col[None, :]) / $(65536)) % $(8)) * $(131072)) + (((((r260_group % $(8)) * $(65536)) + (r260_lane[:, None] * $(128)) + r260_col[None, :]) % $(128)) * $(512)) + (((((r260_group % $(8)) * $(65536)) + (r260_lane[:, None] * $(128)) + r260_col[None, :]) / $(128)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r260_value = tl.dot(r260_left, r260_right)
      r260_value = r260_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_21_ptr) + ((r260_group * $(4096)) + (r260_row[:, None] * $(128)) + r260_col[None, :]), r260_value)
    }
    r261_z = tl.arange($(131072))
    r261_value = tl.where(((r261_z / $(1024)) % $(128)) < $(32), tl.load($(candidate_vbmm_6_ptr) + ((((r261_z / $(128)) % $(8)) * $(4096)) + (((((r261_z / $(1024)) % $(128)) - $(0)) % $(32)) * $(128)) + (r261_z % $(128))), dtype=tl.bfloat16), tl.where(((r261_z / $(1024)) % $(128)) < $(64), tl.load($(candidate_vbmm_11_ptr) + ((((r261_z / $(128)) % $(8)) * $(4096)) + (((((r261_z / $(1024)) % $(128)) - $(32)) % $(32)) * $(128)) + (r261_z % $(128))), dtype=tl.bfloat16), tl.where(((r261_z / $(1024)) % $(128)) < $(96), tl.load($(candidate_vbmm_16_ptr) + ((((r261_z / $(128)) % $(8)) * $(4096)) + (((((r261_z / $(1024)) % $(128)) - $(64)) % $(32)) * $(128)) + (r261_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vbmm_21_ptr) + ((((r261_z / $(128)) % $(8)) * $(4096)) + (((((r261_z / $(1024)) % $(128)) - $(96)) % $(32)) * $(128)) + (r261_z % $(128))), dtype=tl.bfloat16))))
    r261_value = r261_value.to(tl.bfloat16)
    r261_value = tl.broadcast_to(r261_value, [$(131072)])
    tl.store($(candidate_vcat_10_ptr) + r261_z, r261_value)
    r262_row = tl.arange($(128))
    r262_col = tl.arange($(512))
    r262_lane = tl.arange($(1024))
    r262_left = tl.load($(candidate_vcat_10_ptr) + ((((((r262_row[:, None] * $(1024)) + r262_lane[None, :]) / $(1024)) % $(128)) * $(1024)) + (((((((((r262_row[:, None] * $(1024)) + r262_lane[None, :]) / $(1024)) % $(128)) * $(1024)) + (((r262_row[:, None] * $(1024)) + r262_lane[None, :]) % $(1024))) / $(128)) % $(8)) * $(128)) + (((((((r262_row[:, None] * $(1024)) + r262_lane[None, :]) / $(1024)) % $(128)) * $(1024)) + (((r262_row[:, None] * $(1024)) + r262_lane[None, :]) % $(1024))) % $(128))), dtype=tl.bfloat16)
    r262_right = tl.load($(candidate_vto_copy_6_ptr) + (((r262_col[None, :] % $(512)) * $(1024)) + (r262_lane[:, None] % $(1024))), dtype=tl.bfloat16)
    r262_value = tl.dot(r262_left, r262_right)
    r262_value = r262_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_27_ptr) + ((r262_row[:, None] * $(512)) + r262_col[None, :]), r262_value)
    r263_z = tl.arange($(65536))
    r263_value = tl.load($(candidate_vmm_27_ptr) + (((((((tl.load($(candidate_vsort_index_ptr) + ((r263_z / $(512)) % $(128)), dtype=tl.int64) * $(512)) + (r263_z % $(512))) / $(512)) % $(128)) % $(128)) * $(512)) + ((((tl.load($(candidate_vsort_index_ptr) + ((r263_z / $(512)) % $(128)), dtype=tl.int64) * $(512)) + (r263_z % $(512))) % $(512)) % $(512))), dtype=tl.bfloat16)
    r263_value = r263_value.to(tl.bfloat16)
    r263_value = tl.broadcast_to(r263_value, [$(65536)])
    tl.store($(candidate_vindex_3_ptr) + r263_z, r263_value)
    z = tl.arange($(65536))
    value = tl.load($(candidate_vindex_3_ptr) + ((((z / $(512)) % $(128)) * $(512)) + (z % $(512))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(65536)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_mla_fa4_te_rope_cp_absorb_fwd
