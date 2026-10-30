import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_mla_fa4_absorb_seqused_fwd
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
  let vdiv_ptr : RegionName := "vdiv"
  let vpow_1_ptr : RegionName := "vpow_1"
  let vreciprocal_ptr : RegionName := "vreciprocal"
  let vmul_7_ptr : RegionName := "vmul_7"
  let vmul_8_ptr : RegionName := "vmul_8"
  let vcat_1_ptr : RegionName := "vcat_1"
  let vcos_ptr : RegionName := "vcos"
  let vindex_ptr : RegionName := "vindex"
  let vsin_ptr : RegionName := "vsin"
  let vindex_1_ptr : RegionName := "vindex_1"
  let vto_copy_14_ptr : RegionName := "vto_copy_14"
  let vto_copy_15_ptr : RegionName := "vto_copy_15"
  let vsub_ptr : RegionName := "vsub"
  let vcat_2_ptr : RegionName := "vcat_2"
  let vsub_1_ptr : RegionName := "vsub_1"
  let vne_ptr : RegionName := "vne"
  let vcumsum_1_ptr : RegionName := "vcumsum_1"
  let veq_ptr : RegionName := "veq"
  let vtril_ptr : RegionName := "vtril"
  let vbitwise_and_ptr : RegionName := "vbitwise_and"
  let vwhere_ptr : RegionName := "vwhere"
  let vcat_4_ptr : RegionName := "vcat_4"
  let vmm_ptr : RegionName := "vmm"
  let vto_copy_18_ptr : RegionName := "vto_copy_18"
  let vpow_2_ptr : RegionName := "vpow_2"
  let vmean_ptr : RegionName := "vmean"
  let vadd_2_ptr : RegionName := "vadd_2"
  let vrsqrt_ptr : RegionName := "vrsqrt"
  let vmul_9_ptr : RegionName := "vmul_9"
  let vmul_10_ptr : RegionName := "vmul_10"
  let vto_copy_19_ptr : RegionName := "vto_copy_19"
  let vmm_1_ptr : RegionName := "vmm_1"
  let vmm_2_ptr : RegionName := "vmm_2"
  let vto_copy_20_ptr : RegionName := "vto_copy_20"
  let vpow_3_ptr : RegionName := "vpow_3"
  let vmean_1_ptr : RegionName := "vmean_1"
  let vadd_3_ptr : RegionName := "vadd_3"
  let vrsqrt_1_ptr : RegionName := "vrsqrt_1"
  let vmul_11_ptr : RegionName := "vmul_11"
  let vmul_12_ptr : RegionName := "vmul_12"
  let vto_copy_21_ptr : RegionName := "vto_copy_21"
  let vmm_3_ptr : RegionName := "vmm_3"
  let vmul_13_ptr : RegionName := "vmul_13"
  let vmul_14_ptr : RegionName := "vmul_14"
  let vsub_13_ptr : RegionName := "vsub_13"
  let vmul_15_ptr : RegionName := "vmul_15"
  let vmul_16_ptr : RegionName := "vmul_16"
  let vadd_4_ptr : RegionName := "vadd_4"
  let vcat_5_ptr : RegionName := "vcat_5"
  let vmul_17_ptr : RegionName := "vmul_17"
  let vmul_18_ptr : RegionName := "vmul_18"
  let vsub_14_ptr : RegionName := "vsub_14"
  let vmul_19_ptr : RegionName := "vmul_19"
  let vmul_20_ptr : RegionName := "vmul_20"
  let vadd_5_ptr : RegionName := "vadd_5"
  let vcat_6_ptr : RegionName := "vcat_6"
  let vcat_7_ptr : RegionName := "vcat_7"
  let vcat_8_ptr : RegionName := "vcat_8"
  let vbmm_ptr : RegionName := "vbmm"
  let vmul_21_ptr : RegionName := "vmul_21"
  let vadd_6_ptr : RegionName := "vadd_6"
  let vto_copy_22_ptr : RegionName := "vto_copy_22"
  let vsoftmax_ptr : RegionName := "vsoftmax"
  let vto_copy_23_ptr : RegionName := "vto_copy_23"
  let vbmm_1_ptr : RegionName := "vbmm_1"
  let vmm_4_ptr : RegionName := "vmm_4"
  let vindex_2_ptr : RegionName := "vindex_2"
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
    r16_z = tl.arange($(288))
    r16_value = tl.where(r16_z < $(96), tl.cast((r16_z - $(0)), tl.int64), tl.where(r16_z < $(192), tl.cast((r16_z - $(96)), tl.int64), tl.cast((r16_z - $(192)), tl.int64)))
    r16_value = tl.broadcast_to(r16_value, [$(288)])
    tl.store($(vcat_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(32))
    r17_value = ((tl.cast(tl.toReal(r17_z), tl.float32) * 2.0) / 64.0)
    r17_value = r17_value.to(tl.float32)
    r17_value = tl.broadcast_to(r17_value, [$(32)])
    tl.store($(vdiv_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(32))
    r18_value = tl.extra.cuda.libdevice.pow(10000.0, tl.load($(vdiv_ptr) + r18_z, dtype=tl.float32))
    r18_value = r18_value.to(tl.float32)
    r18_value = tl.broadcast_to(r18_value, [$(32)])
    tl.store($(vpow_1_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(32))
    r19_value = 1.0 / tl.load($(vpow_1_ptr) + r19_z, dtype=tl.float32)
    r19_value = r19_value.to(tl.float32)
    r19_value = tl.broadcast_to(r19_value, [$(32)])
    tl.store($(vreciprocal_ptr) + r19_z, r19_value)
    r20_z = tl.arange($(32))
    r20_value = (tl.load($(vreciprocal_ptr) + r20_z, dtype=tl.float32) * 1.0)
    r20_value = r20_value.to(tl.float32)
    r20_value = tl.broadcast_to(r20_value, [$(32)])
    tl.store($(vmul_7_ptr) + r20_z, r20_value)
    r21_z = tl.arange($(9216))
    r21_value = (tl.cast(tl.toReal(((r21_z / $(32)) % $(288))), tl.float32) * tl.load($(vmul_7_ptr) + (r21_z % $(32)), dtype=tl.float32))
    r21_value = r21_value.to(tl.float32)
    r21_value = tl.broadcast_to(r21_value, [$(9216)])
    tl.store($(vmul_8_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(18432))
    r22_value = tl.where((r22_z % $(64)) < $(32), tl.load($(vmul_8_ptr) + ((((r22_z / $(64)) % $(288)) * $(32)) + (((r22_z % $(64)) - $(0)) % $(32))), dtype=tl.float32), tl.load($(vmul_8_ptr) + ((((r22_z / $(64)) % $(288)) * $(32)) + (((r22_z % $(64)) - $(32)) % $(32))), dtype=tl.float32))
    r22_value = r22_value.to(tl.float32)
    r22_value = tl.broadcast_to(r22_value, [$(18432)])
    tl.store($(vcat_1_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(18432))
    r23_value = tl.cos(tl.load($(vcat_1_ptr) + ((((r23_z / $(64)) % $(288)) * $(64)) + (r23_z % $(64))), dtype=tl.float32))
    r23_value = r23_value.to(tl.float32)
    r23_value = tl.broadcast_to(r23_value, [$(18432)])
    tl.store($(vcos_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(18432))
    r24_value = tl.load($(vcos_ptr) + (((((((tl.load($(vcat_ptr) + ((r24_z / $(64)) % $(288)), dtype=tl.int64) * $(64)) + (r24_z % $(64))) / $(64)) % $(288)) % $(288)) * $(64)) + ((((tl.load($(vcat_ptr) + ((r24_z / $(64)) % $(288)), dtype=tl.int64) * $(64)) + (r24_z % $(64))) % $(64)) % $(64))), dtype=tl.float32)
    r24_value = r24_value.to(tl.float32)
    r24_value = tl.broadcast_to(r24_value, [$(18432)])
    tl.store($(vindex_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(18432))
    r25_value = tl.sin(tl.load($(vcat_1_ptr) + ((((r25_z / $(64)) % $(288)) * $(64)) + (r25_z % $(64))), dtype=tl.float32))
    r25_value = r25_value.to(tl.float32)
    r25_value = tl.broadcast_to(r25_value, [$(18432)])
    tl.store($(vsin_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(18432))
    r26_value = tl.load($(vsin_ptr) + (((((((tl.load($(vcat_ptr) + ((r26_z / $(64)) % $(288)), dtype=tl.int64) * $(64)) + (r26_z % $(64))) / $(64)) % $(288)) % $(288)) * $(64)) + ((((tl.load($(vcat_ptr) + ((r26_z / $(64)) % $(288)), dtype=tl.int64) * $(64)) + (r26_z % $(64))) % $(64)) % $(64))), dtype=tl.float32)
    r26_value = r26_value.to(tl.float32)
    r26_value = tl.broadcast_to(r26_value, [$(18432)])
    tl.store($(vindex_1_ptr) + r26_z, r26_value)
    r27_z = tl.arange($(18432))
    r27_value = tl.load($(vindex_ptr) + ((((r27_z / $(64)) % $(288)) * $(64)) + (r27_z % $(64))), dtype=tl.float32)
    r27_value = r27_value.to(tl.bfloat16)
    r27_value = tl.broadcast_to(r27_value, [$(18432)])
    tl.store($(vto_copy_14_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(18432))
    r28_value = tl.load($(vindex_1_ptr) + ((((r28_z / $(64)) % $(288)) * $(64)) + (r28_z % $(64))), dtype=tl.float32)
    r28_value = r28_value.to(tl.bfloat16)
    r28_value = tl.broadcast_to(r28_value, [$(18432)])
    tl.store($(vto_copy_15_ptr) + r28_z, r28_value)
    r29_z = tl.arange($(1))
    r29_value = (tl.load($(vcat_ptr) + $(0), dtype=tl.int64) - tl.full([], $(1), dtype=tl.int64))
    r29_value = tl.broadcast_to(r29_value, [$(1)])
    tl.store($(vsub_ptr) + r29_z, r29_value)
    r30_z = tl.arange($(289))
    r30_value = tl.where(r30_z < $(1), tl.load($(vsub_ptr) + $(0), dtype=tl.int64), tl.load($(vcat_ptr) + ((r30_z - $(1)) % $(288)), dtype=tl.int64))
    r30_value = tl.broadcast_to(r30_value, [$(289)])
    tl.store($(vcat_2_ptr) + r30_z, r30_value)
    r31_z = tl.arange($(288))
    r31_value = (tl.load($(vcat_2_ptr) + (r31_z + $(1)), dtype=tl.int64) - tl.load($(vcat_2_ptr) + r31_z, dtype=tl.int64))
    r31_value = tl.broadcast_to(r31_value, [$(288)])
    tl.store($(vsub_1_ptr) + r31_z, r31_value)
    r32_z = tl.arange($(288))
    r32_value = (tl.load($(vsub_1_ptr) + r32_z, dtype=tl.int64) != tl.full([], $(1), dtype=tl.int64))
    r32_value = tl.broadcast_to(r32_value, [$(288)])
    tl.store($(vne_ptr) + r32_z, r32_value)
    r33_z = tl.arange($(288))
    r33_value = tl.broadcast_to(tl.full([], $(0), dtype=tl.int64), [$(288)])
    for r33_lane in range($(288)) {
      r33_value = tl.where(r33_lane <= r33_z, (r33_value + tl.where(tl.load($(vne_ptr) + (r33_lane % $(288)), dtype=tl.int1), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))), r33_value)
    }
    r33_value = tl.broadcast_to(r33_value, [$(288)])
    tl.store($(vcumsum_1_ptr) + r33_z, r33_value)
    r34_z = tl.arange($(82944))
    r34_value = (tl.load($(vcumsum_1_ptr) + ((r34_z / $(288)) % $(288)), dtype=tl.int64) == tl.load($(vcumsum_1_ptr) + (r34_z % $(288)), dtype=tl.int64))
    r34_value = tl.broadcast_to(r34_value, [$(82944)])
    tl.store($(veq_ptr) + r34_z, r34_value)
    r35_z = tl.arange($(82944))
    r35_value = tl.where((r35_z % $(288)) <= ((r35_z / $(288)) % $(288)) + $(0), true, false)
    r35_value = tl.broadcast_to(r35_value, [$(82944)])
    tl.store($(vtril_ptr) + r35_z, r35_value)
    r36_z = tl.arange($(82944))
    r36_value = (tl.load($(veq_ptr) + ((((r36_z / $(288)) % $(288)) * $(288)) + (r36_z % $(288))), dtype=tl.int1) & tl.load($(vtril_ptr) + ((((r36_z / $(288)) % $(288)) * $(288)) + (r36_z % $(288))), dtype=tl.int1))
    r36_value = tl.broadcast_to(r36_value, [$(82944)])
    tl.store($(vbitwise_and_ptr) + r36_z, r36_value)
    r37_z = tl.arange($(82944))
    r37_value = tl.where(tl.load($(vbitwise_and_ptr) + ((((r37_z / $(288)) % $(288)) * $(288)) + (r37_z % $(288))), dtype=tl.int1), tl.cast(0.0, tl.bfloat16), tl.cast(-338953138925153547590470800371487866880.0, tl.bfloat16))
    r37_value = r37_value.to(tl.bfloat16)
    r37_value = tl.broadcast_to(r37_value, [$(82944)])
    tl.store($(vwhere_ptr) + r37_z, r37_value)
    r38_z = tl.arange($(96))
    r38_value = tl.where(r38_z < $(32), (tl.full([], $(64), dtype=tl.int64) + tl.cast((r38_z - $(0)), tl.int64)), tl.where(r38_z < $(64), (tl.full([], $(128), dtype=tl.int64) + tl.cast((r38_z - $(32)), tl.int64)), (tl.full([], $(192), dtype=tl.int64) + tl.cast((r38_z - $(64)), tl.int64))))
    r38_value = tl.broadcast_to(r38_value, [$(96)])
    tl.store($(vcat_4_ptr) + r38_z, r38_value)
    r39_row = tl.arange($(288))
    r39_col = tl.arange($(256))
    r39_lane = tl.arange($(512))
    r39_left = tl.load($(input0_ptr) + ((((((r39_row[:, None] * $(512)) + r39_lane[None, :]) / $(512)) % $(288)) * $(512)) + (((r39_row[:, None] * $(512)) + r39_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r39_right = tl.load($(vto_copy_ptr) + (((r39_col[None, :] % $(256)) * $(512)) + (r39_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r39_value = tl.dot(r39_left, r39_right)
    r39_value = r39_value.to(tl.bfloat16)
    tl.store($(vmm_ptr) + ((r39_row[:, None] * $(256)) + r39_col[None, :]), r39_value)
    r40_z = tl.arange($(73728))
    r40_value = tl.load($(vmm_ptr) + ((((r40_z / $(256)) % $(288)) * $(256)) + (r40_z % $(256))), dtype=tl.bfloat16)
    r40_value = r40_value.to(tl.float32)
    r40_value = tl.broadcast_to(r40_value, [$(73728)])
    tl.store($(vto_copy_18_ptr) + r40_z, r40_value)
    r41_z = tl.arange($(73728))
    r41_value = ((tl.load($(vto_copy_18_ptr) + ((((r41_z / $(256)) % $(288)) * $(256)) + (r41_z % $(256))), dtype=tl.float32)) * (tl.load($(vto_copy_18_ptr) + ((((r41_z / $(256)) % $(288)) * $(256)) + (r41_z % $(256))), dtype=tl.float32)))
    r41_value = r41_value.to(tl.float32)
    r41_value = tl.broadcast_to(r41_value, [$(73728)])
    tl.store($(vpow_2_ptr) + r41_z, r41_value)
    r42_z = tl.arange($(288))
    r42_lane = tl.arange($(256))
    r42_value = (tl.sum(tl.load($(vpow_2_ptr) + ((r42_z[:, None] * $(256)) + r42_lane[None, :]), dtype=tl.float32), axis=1) / 256.0)
    r42_value = r42_value.to(tl.float32)
    r42_value = tl.broadcast_to(r42_value, [$(288)])
    tl.store($(vmean_ptr) + r42_z, r42_value)
    r43_z = tl.arange($(288))
    r43_value = (tl.load($(vmean_ptr) + r43_z, dtype=tl.float32) + 0.000001)
    r43_value = r43_value.to(tl.float32)
    r43_value = tl.broadcast_to(r43_value, [$(288)])
    tl.store($(vadd_2_ptr) + r43_z, r43_value)
    r44_z = tl.arange($(288))
    r44_value = tl.rsqrt(tl.load($(vadd_2_ptr) + r44_z, dtype=tl.float32))
    r44_value = r44_value.to(tl.float32)
    r44_value = tl.broadcast_to(r44_value, [$(288)])
    tl.store($(vrsqrt_ptr) + r44_z, r44_value)
    r45_z = tl.arange($(73728))
    r45_value = (tl.load($(vto_copy_18_ptr) + ((((r45_z / $(256)) % $(288)) * $(256)) + (r45_z % $(256))), dtype=tl.float32) * tl.load($(vrsqrt_ptr) + ((r45_z / $(256)) % $(288)), dtype=tl.float32))
    r45_value = r45_value.to(tl.float32)
    r45_value = tl.broadcast_to(r45_value, [$(73728)])
    tl.store($(vmul_9_ptr) + r45_z, r45_value)
    r46_z = tl.arange($(73728))
    r46_value = (tl.cast(tl.load($(vto_copy_1_ptr) + (r46_z % $(256)), dtype=tl.bfloat16), tl.float32) * tl.load($(vmul_9_ptr) + ((((r46_z / $(256)) % $(288)) * $(256)) + (r46_z % $(256))), dtype=tl.float32))
    r46_value = r46_value.to(tl.float32)
    r46_value = tl.broadcast_to(r46_value, [$(73728)])
    tl.store($(vmul_10_ptr) + r46_z, r46_value)
    r47_z = tl.arange($(73728))
    r47_value = tl.load($(vmul_10_ptr) + ((((r47_z / $(256)) % $(288)) * $(256)) + (r47_z % $(256))), dtype=tl.float32)
    r47_value = r47_value.to(tl.bfloat16)
    r47_value = tl.broadcast_to(r47_value, [$(73728)])
    tl.store($(vto_copy_19_ptr) + r47_z, r47_value)
    r48_row = tl.arange($(288))
    r48_col = tl.arange($(1536))
    r48_lane = tl.arange($(256))
    r48_left = tl.load($(vto_copy_19_ptr) + ((((((r48_row[:, None] * $(256)) + r48_lane[None, :]) / $(256)) % $(288)) * $(256)) + (((r48_row[:, None] * $(256)) + r48_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r48_right = tl.load($(vto_copy_2_ptr) + (((r48_col[None, :] % $(1536)) * $(256)) + (r48_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r48_value = tl.dot(r48_left, r48_right)
    r48_value = r48_value.to(tl.bfloat16)
    tl.store($(vmm_1_ptr) + ((r48_row[:, None] * $(1536)) + r48_col[None, :]), r48_value)
    r49_row = tl.arange($(288))
    r49_col = tl.arange($(576))
    r49_lane = tl.arange($(512))
    r49_left = tl.load($(input0_ptr) + ((((((r49_row[:, None] * $(512)) + r49_lane[None, :]) / $(512)) % $(288)) * $(512)) + (((r49_row[:, None] * $(512)) + r49_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r49_right = tl.load($(vto_copy_3_ptr) + (((r49_col[None, :] % $(576)) * $(512)) + (r49_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r49_value = tl.dot(r49_left, r49_right)
    r49_value = r49_value.to(tl.bfloat16)
    tl.store($(vmm_2_ptr) + ((r49_row[:, None] * $(576)) + r49_col[None, :]), r49_value)
    r50_z = tl.arange($(147456))
    r50_value = tl.load($(vmm_2_ptr) + ((((r50_z / $(512)) % $(288)) * $(576)) + (r50_z % $(512))), dtype=tl.bfloat16)
    r50_value = r50_value.to(tl.float32)
    r50_value = tl.broadcast_to(r50_value, [$(147456)])
    tl.store($(vto_copy_20_ptr) + r50_z, r50_value)
    r51_z = tl.arange($(147456))
    r51_value = ((tl.load($(vto_copy_20_ptr) + ((((r51_z / $(512)) % $(288)) * $(512)) + (r51_z % $(512))), dtype=tl.float32)) * (tl.load($(vto_copy_20_ptr) + ((((r51_z / $(512)) % $(288)) * $(512)) + (r51_z % $(512))), dtype=tl.float32)))
    r51_value = r51_value.to(tl.float32)
    r51_value = tl.broadcast_to(r51_value, [$(147456)])
    tl.store($(vpow_3_ptr) + r51_z, r51_value)
    r52_z = tl.arange($(288))
    r52_lane = tl.arange($(512))
    r52_value = (tl.sum(tl.load($(vpow_3_ptr) + ((r52_z[:, None] * $(512)) + r52_lane[None, :]), dtype=tl.float32), axis=1) / 512.0)
    r52_value = r52_value.to(tl.float32)
    r52_value = tl.broadcast_to(r52_value, [$(288)])
    tl.store($(vmean_1_ptr) + r52_z, r52_value)
    r53_z = tl.arange($(288))
    r53_value = (tl.load($(vmean_1_ptr) + r53_z, dtype=tl.float32) + 0.000001)
    r53_value = r53_value.to(tl.float32)
    r53_value = tl.broadcast_to(r53_value, [$(288)])
    tl.store($(vadd_3_ptr) + r53_z, r53_value)
    r54_z = tl.arange($(288))
    r54_value = tl.rsqrt(tl.load($(vadd_3_ptr) + r54_z, dtype=tl.float32))
    r54_value = r54_value.to(tl.float32)
    r54_value = tl.broadcast_to(r54_value, [$(288)])
    tl.store($(vrsqrt_1_ptr) + r54_z, r54_value)
    r55_z = tl.arange($(147456))
    r55_value = (tl.load($(vto_copy_20_ptr) + ((((r55_z / $(512)) % $(288)) * $(512)) + (r55_z % $(512))), dtype=tl.float32) * tl.load($(vrsqrt_1_ptr) + ((r55_z / $(512)) % $(288)), dtype=tl.float32))
    r55_value = r55_value.to(tl.float32)
    r55_value = tl.broadcast_to(r55_value, [$(147456)])
    tl.store($(vmul_11_ptr) + r55_z, r55_value)
    r56_z = tl.arange($(147456))
    r56_value = (tl.cast(tl.load($(vto_copy_4_ptr) + (r56_z % $(512)), dtype=tl.bfloat16), tl.float32) * tl.load($(vmul_11_ptr) + ((((r56_z / $(512)) % $(288)) * $(512)) + (r56_z % $(512))), dtype=tl.float32))
    r56_value = r56_value.to(tl.float32)
    r56_value = tl.broadcast_to(r56_value, [$(147456)])
    tl.store($(vmul_12_ptr) + r56_z, r56_value)
    r57_z = tl.arange($(147456))
    r57_value = tl.load($(vmul_12_ptr) + ((((r57_z / $(512)) % $(288)) * $(512)) + (r57_z % $(512))), dtype=tl.float32)
    r57_value = r57_value.to(tl.bfloat16)
    r57_value = tl.broadcast_to(r57_value, [$(147456)])
    tl.store($(vto_copy_21_ptr) + r57_z, r57_value)
    r58_row = tl.arange($(288))
    r58_col = tl.arange($(2048))
    r58_lane = tl.arange($(512))
    r58_left = tl.load($(vto_copy_21_ptr) + ((((((r58_row[:, None] * $(512)) + r58_lane[None, :]) / $(512)) % $(288)) * $(512)) + (((r58_row[:, None] * $(512)) + r58_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r58_right = tl.load($(vto_copy_5_ptr) + (((r58_col[None, :] % $(2048)) * $(512)) + (r58_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r58_value = tl.dot(r58_left, r58_right)
    r58_value = r58_value.to(tl.bfloat16)
    tl.store($(vmm_3_ptr) + ((r58_row[:, None] * $(2048)) + r58_col[None, :]), r58_value)
    r59_z = tl.arange($(73728))
    r59_value = (tl.load($(vmm_1_ptr) + ((((r59_z / $(32)) % $(288)) * $(1536)) + (((r59_z / $(9216)) % $(8)) * $(192)) + ((r59_z % $(32)) * $(2)) + $(128)), dtype=tl.bfloat16) * tl.load($(vto_copy_14_ptr) + ((((r59_z / $(32)) % $(288)) * $(64)) + (r59_z % $(32))), dtype=tl.bfloat16))
    r59_value = r59_value.to(tl.bfloat16)
    r59_value = tl.broadcast_to(r59_value, [$(73728)])
    tl.store($(vmul_13_ptr) + r59_z, r59_value)
    r60_z = tl.arange($(73728))
    r60_value = (tl.load($(vmm_1_ptr) + ((((r60_z / $(32)) % $(288)) * $(1536)) + (((r60_z / $(9216)) % $(8)) * $(192)) + ((r60_z % $(32)) * $(2)) + $(129)), dtype=tl.bfloat16) * tl.load($(vto_copy_15_ptr) + ((((r60_z / $(32)) % $(288)) * $(64)) + (r60_z % $(32))), dtype=tl.bfloat16))
    r60_value = r60_value.to(tl.bfloat16)
    r60_value = tl.broadcast_to(r60_value, [$(73728)])
    tl.store($(vmul_14_ptr) + r60_z, r60_value)
    r61_z = tl.arange($(73728))
    r61_value = (tl.load($(vmul_13_ptr) + ((((r61_z / $(9216)) % $(8)) * $(9216)) + (((r61_z / $(32)) % $(288)) * $(32)) + (r61_z % $(32))), dtype=tl.bfloat16) - tl.load($(vmul_14_ptr) + ((((r61_z / $(9216)) % $(8)) * $(9216)) + (((r61_z / $(32)) % $(288)) * $(32)) + (r61_z % $(32))), dtype=tl.bfloat16))
    r61_value = r61_value.to(tl.bfloat16)
    r61_value = tl.broadcast_to(r61_value, [$(73728)])
    tl.store($(vsub_13_ptr) + r61_z, r61_value)
    r62_z = tl.arange($(73728))
    r62_value = (tl.load($(vmm_1_ptr) + ((((r62_z / $(32)) % $(288)) * $(1536)) + (((r62_z / $(9216)) % $(8)) * $(192)) + ((r62_z % $(32)) * $(2)) + $(129)), dtype=tl.bfloat16) * tl.load($(vto_copy_14_ptr) + ((((r62_z / $(32)) % $(288)) * $(64)) + (r62_z % $(32))), dtype=tl.bfloat16))
    r62_value = r62_value.to(tl.bfloat16)
    r62_value = tl.broadcast_to(r62_value, [$(73728)])
    tl.store($(vmul_15_ptr) + r62_z, r62_value)
    r63_z = tl.arange($(73728))
    r63_value = (tl.load($(vmm_1_ptr) + ((((r63_z / $(32)) % $(288)) * $(1536)) + (((r63_z / $(9216)) % $(8)) * $(192)) + ((r63_z % $(32)) * $(2)) + $(128)), dtype=tl.bfloat16) * tl.load($(vto_copy_15_ptr) + ((((r63_z / $(32)) % $(288)) * $(64)) + (r63_z % $(32))), dtype=tl.bfloat16))
    r63_value = r63_value.to(tl.bfloat16)
    r63_value = tl.broadcast_to(r63_value, [$(73728)])
    tl.store($(vmul_16_ptr) + r63_z, r63_value)
    r64_z = tl.arange($(73728))
    r64_value = (tl.load($(vmul_15_ptr) + ((((r64_z / $(9216)) % $(8)) * $(9216)) + (((r64_z / $(32)) % $(288)) * $(32)) + (r64_z % $(32))), dtype=tl.bfloat16) + tl.load($(vmul_16_ptr) + ((((r64_z / $(9216)) % $(8)) * $(9216)) + (((r64_z / $(32)) % $(288)) * $(32)) + (r64_z % $(32))), dtype=tl.bfloat16))
    r64_value = r64_value.to(tl.bfloat16)
    r64_value = tl.broadcast_to(r64_value, [$(73728)])
    tl.store($(vadd_4_ptr) + r64_z, r64_value)
    r65_z = tl.arange($(147456))
    r65_value = tl.where((r65_z % $(64)) < $(32), tl.load($(vsub_13_ptr) + ((((r65_z / $(18432)) % $(8)) * $(9216)) + (((r65_z / $(64)) % $(288)) * $(32)) + (((r65_z % $(64)) - $(0)) % $(32))), dtype=tl.bfloat16), tl.load($(vadd_4_ptr) + ((((r65_z / $(18432)) % $(8)) * $(9216)) + (((r65_z / $(64)) % $(288)) * $(32)) + (((r65_z % $(64)) - $(32)) % $(32))), dtype=tl.bfloat16))
    r65_value = r65_value.to(tl.bfloat16)
    r65_value = tl.broadcast_to(r65_value, [$(147456)])
    tl.store($(vcat_5_ptr) + r65_z, r65_value)
    r66_z = tl.arange($(9216))
    r66_value = (tl.load($(vmm_2_ptr) + ((((r66_z / $(32)) % $(288)) * $(576)) + ((r66_z % $(32)) * $(2)) + $(512)), dtype=tl.bfloat16) * tl.load($(vto_copy_14_ptr) + ((((r66_z / $(32)) % $(288)) * $(64)) + (r66_z % $(32))), dtype=tl.bfloat16))
    r66_value = r66_value.to(tl.bfloat16)
    r66_value = tl.broadcast_to(r66_value, [$(9216)])
    tl.store($(vmul_17_ptr) + r66_z, r66_value)
    r67_z = tl.arange($(9216))
    r67_value = (tl.load($(vmm_2_ptr) + ((((r67_z / $(32)) % $(288)) * $(576)) + ((r67_z % $(32)) * $(2)) + $(513)), dtype=tl.bfloat16) * tl.load($(vto_copy_15_ptr) + ((((r67_z / $(32)) % $(288)) * $(64)) + (r67_z % $(32))), dtype=tl.bfloat16))
    r67_value = r67_value.to(tl.bfloat16)
    r67_value = tl.broadcast_to(r67_value, [$(9216)])
    tl.store($(vmul_18_ptr) + r67_z, r67_value)
    r68_z = tl.arange($(9216))
    r68_value = (tl.load($(vmul_17_ptr) + ((((r68_z / $(32)) % $(288)) * $(32)) + (r68_z % $(32))), dtype=tl.bfloat16) - tl.load($(vmul_18_ptr) + ((((r68_z / $(32)) % $(288)) * $(32)) + (r68_z % $(32))), dtype=tl.bfloat16))
    r68_value = r68_value.to(tl.bfloat16)
    r68_value = tl.broadcast_to(r68_value, [$(9216)])
    tl.store($(vsub_14_ptr) + r68_z, r68_value)
    r69_z = tl.arange($(9216))
    r69_value = (tl.load($(vmm_2_ptr) + ((((r69_z / $(32)) % $(288)) * $(576)) + ((r69_z % $(32)) * $(2)) + $(513)), dtype=tl.bfloat16) * tl.load($(vto_copy_14_ptr) + ((((r69_z / $(32)) % $(288)) * $(64)) + (r69_z % $(32))), dtype=tl.bfloat16))
    r69_value = r69_value.to(tl.bfloat16)
    r69_value = tl.broadcast_to(r69_value, [$(9216)])
    tl.store($(vmul_19_ptr) + r69_z, r69_value)
    r70_z = tl.arange($(9216))
    r70_value = (tl.load($(vmm_2_ptr) + ((((r70_z / $(32)) % $(288)) * $(576)) + ((r70_z % $(32)) * $(2)) + $(512)), dtype=tl.bfloat16) * tl.load($(vto_copy_15_ptr) + ((((r70_z / $(32)) % $(288)) * $(64)) + (r70_z % $(32))), dtype=tl.bfloat16))
    r70_value = r70_value.to(tl.bfloat16)
    r70_value = tl.broadcast_to(r70_value, [$(9216)])
    tl.store($(vmul_20_ptr) + r70_z, r70_value)
    r71_z = tl.arange($(9216))
    r71_value = (tl.load($(vmul_19_ptr) + ((((r71_z / $(32)) % $(288)) * $(32)) + (r71_z % $(32))), dtype=tl.bfloat16) + tl.load($(vmul_20_ptr) + ((((r71_z / $(32)) % $(288)) * $(32)) + (r71_z % $(32))), dtype=tl.bfloat16))
    r71_value = r71_value.to(tl.bfloat16)
    r71_value = tl.broadcast_to(r71_value, [$(9216)])
    tl.store($(vadd_5_ptr) + r71_z, r71_value)
    r72_z = tl.arange($(18432))
    r72_value = tl.where((r72_z % $(64)) < $(32), tl.load($(vsub_14_ptr) + ((((r72_z / $(64)) % $(288)) * $(32)) + (((r72_z % $(64)) - $(0)) % $(32))), dtype=tl.bfloat16), tl.load($(vadd_5_ptr) + ((((r72_z / $(64)) % $(288)) * $(32)) + (((r72_z % $(64)) - $(32)) % $(32))), dtype=tl.bfloat16))
    r72_value = r72_value.to(tl.bfloat16)
    r72_value = tl.broadcast_to(r72_value, [$(18432)])
    tl.store($(vcat_6_ptr) + r72_z, r72_value)
    r73_z = tl.arange($(442368))
    r73_value = tl.where((r73_z % $(192)) < $(128), tl.load($(vmm_1_ptr) + ((((((((r73_z / $(192)) % $(288)) * $(1536)) + (((r73_z / $(55296)) % $(8)) * $(192)) + ((r73_z % $(192)) - $(0))) / $(1536)) % $(288)) * $(1536)) + (((((r73_z / $(192)) % $(288)) * $(1536)) + (((r73_z / $(55296)) % $(8)) * $(192)) + ((r73_z % $(192)) - $(0))) % $(1536))), dtype=tl.bfloat16), tl.load($(vcat_5_ptr) + ((((r73_z / $(55296)) % $(8)) * $(18432)) + (((r73_z / $(192)) % $(288)) * $(64)) + (((r73_z % $(192)) - $(128)) % $(64))), dtype=tl.bfloat16))
    r73_value = r73_value.to(tl.bfloat16)
    r73_value = tl.broadcast_to(r73_value, [$(442368)])
    tl.store($(vcat_7_ptr) + r73_z, r73_value)
    r74_z = tl.arange($(442368))
    r74_value = tl.where((r74_z % $(192)) < $(128), tl.load($(vmm_3_ptr) + ((((((((r74_z / $(192)) % $(288)) * $(2048)) + (((r74_z / $(55296)) % $(8)) * $(256)) + ((r74_z % $(192)) - $(0))) / $(2048)) % $(288)) * $(2048)) + (((((r74_z / $(192)) % $(288)) * $(2048)) + (((r74_z / $(55296)) % $(8)) * $(256)) + ((r74_z % $(192)) - $(0))) % $(2048))), dtype=tl.bfloat16), tl.load($(vcat_6_ptr) + ((((r74_z / $(192)) % $(288)) * $(64)) + (((r74_z % $(192)) - $(128)) % $(64))), dtype=tl.bfloat16))
    r74_value = r74_value.to(tl.bfloat16)
    r74_value = tl.broadcast_to(r74_value, [$(442368)])
    tl.store($(vcat_8_ptr) + r74_z, r74_value)
    r75_row = tl.arange($(288))
    r75_col = tl.arange($(288))
    r75_lane = tl.arange($(192))
    for r75_group in range($(8)) {
      r75_left = tl.load($(vcat_7_ptr) + (((((((r75_group % $(8)) * $(55296)) + (r75_row[:, None] * $(192)) + r75_lane[None, :]) / $(55296)) % $(8)) * $(55296)) + ((((((r75_group % $(8)) * $(55296)) + (r75_row[:, None] * $(192)) + r75_lane[None, :]) / $(192)) % $(288)) * $(192)) + ((((r75_group % $(8)) * $(55296)) + (r75_row[:, None] * $(192)) + r75_lane[None, :]) % $(192))), dtype=tl.bfloat16)
      r75_right = tl.load($(vcat_8_ptr) + (((((((r75_group % $(8)) * $(55296)) + (r75_lane[:, None] * $(288)) + r75_col[None, :]) / $(55296)) % $(8)) * $(55296)) + (((((r75_group % $(8)) * $(55296)) + (r75_lane[:, None] * $(288)) + r75_col[None, :]) % $(288)) * $(192)) + (((((r75_group % $(8)) * $(55296)) + (r75_lane[:, None] * $(288)) + r75_col[None, :]) / $(288)) % $(192))), dtype=tl.bfloat16)
      r75_value = tl.dot(r75_left, r75_right)
      r75_value = r75_value.to(tl.bfloat16)
      tl.store($(vbmm_ptr) + ((r75_group * $(82944)) + (r75_row[:, None] * $(288)) + r75_col[None, :]), r75_value)
    }
    r76_z = tl.arange($(663552))
    r76_value = (tl.load($(vbmm_ptr) + ((((r76_z / $(82944)) % $(8)) * $(82944)) + (((r76_z / $(288)) % $(288)) * $(288)) + (r76_z % $(288))), dtype=tl.bfloat16) * tl.cast(0.07216878364870322, tl.bfloat16))
    r76_value = r76_value.to(tl.bfloat16)
    r76_value = tl.broadcast_to(r76_value, [$(663552)])
    tl.store($(vmul_21_ptr) + r76_z, r76_value)
    r77_z = tl.arange($(663552))
    r77_value = (tl.load($(vmul_21_ptr) + ((((r77_z / $(82944)) % $(8)) * $(82944)) + (((r77_z / $(288)) % $(288)) * $(288)) + (r77_z % $(288))), dtype=tl.bfloat16) + tl.load($(vwhere_ptr) + ((((r77_z / $(288)) % $(288)) * $(288)) + (r77_z % $(288))), dtype=tl.bfloat16))
    r77_value = r77_value.to(tl.bfloat16)
    r77_value = tl.broadcast_to(r77_value, [$(663552)])
    tl.store($(vadd_6_ptr) + r77_z, r77_value)
    r78_z = tl.arange($(663552))
    r78_value = tl.load($(vadd_6_ptr) + ((((r78_z / $(82944)) % $(8)) * $(82944)) + (((r78_z / $(288)) % $(288)) * $(288)) + (r78_z % $(288))), dtype=tl.bfloat16)
    r78_value = r78_value.to(tl.float32)
    r78_value = tl.broadcast_to(r78_value, [$(663552)])
    tl.store($(vto_copy_22_ptr) + r78_z, r78_value)
    r79_z = tl.arange($(663552))
    r79_lane = tl.arange($(288))
    r79_row_values = tl.load($(vto_copy_22_ptr) + ((((r79_z[:, None] / $(82944)) % $(8)) * $(82944)) + (((r79_z[:, None] / $(288)) % $(288)) * $(288)) + (r79_lane[None, :] % $(288))), dtype=tl.float32)
    r79_maximum = tl.max(r79_row_values, axis=1, keep_dims=true)
    r79_weights = tl.exp(r79_row_values - r79_maximum)
    r79_total = tl.sum(r79_weights, axis=1)
    r79_maximum = tl.max(r79_row_values, axis=1)
    r79_value = tl.exp(tl.load($(vto_copy_22_ptr) + ((((r79_z / $(82944)) % $(8)) * $(82944)) + (((r79_z / $(288)) % $(288)) * $(288)) + (r79_z % $(288))), dtype=tl.float32) - r79_maximum) / r79_total
    r79_value = r79_value.to(tl.float32)
    r79_value = tl.broadcast_to(r79_value, [$(663552)])
    tl.store($(vsoftmax_ptr) + r79_z, r79_value)
    r80_z = tl.arange($(663552))
    r80_value = tl.load($(vsoftmax_ptr) + ((((r80_z / $(82944)) % $(8)) * $(82944)) + (((r80_z / $(288)) % $(288)) * $(288)) + (r80_z % $(288))), dtype=tl.float32)
    r80_value = r80_value.to(tl.bfloat16)
    r80_value = tl.broadcast_to(r80_value, [$(663552)])
    tl.store($(vto_copy_23_ptr) + r80_z, r80_value)
    r81_row = tl.arange($(288))
    r81_col = tl.arange($(128))
    r81_lane = tl.arange($(288))
    for r81_group in range($(8)) {
      r81_left = tl.load($(vto_copy_23_ptr) + (((((((r81_group % $(8)) * $(82944)) + (r81_row[:, None] * $(288)) + r81_lane[None, :]) / $(82944)) % $(8)) * $(82944)) + ((((((r81_group % $(8)) * $(82944)) + (r81_row[:, None] * $(288)) + r81_lane[None, :]) / $(288)) % $(288)) * $(288)) + ((((r81_group % $(8)) * $(82944)) + (r81_row[:, None] * $(288)) + r81_lane[None, :]) % $(288))), dtype=tl.bfloat16)
      r81_right = tl.load($(vmm_3_ptr) + (((((((r81_group % $(8)) * $(36864)) + (r81_lane[:, None] * $(128)) + r81_col[None, :]) / $(128)) % $(288)) * $(2048)) + ((((((r81_group % $(8)) * $(36864)) + (r81_lane[:, None] * $(128)) + r81_col[None, :]) / $(36864)) % $(8)) * $(256)) + ((((r81_group % $(8)) * $(36864)) + (r81_lane[:, None] * $(128)) + r81_col[None, :]) % $(128)) + $(128)), dtype=tl.bfloat16)
      r81_value = tl.dot(r81_left, r81_right)
      r81_value = r81_value.to(tl.bfloat16)
      tl.store($(vbmm_1_ptr) + ((r81_group * $(36864)) + (r81_row[:, None] * $(128)) + r81_col[None, :]), r81_value)
    }
    r82_row = tl.arange($(288))
    r82_col = tl.arange($(512))
    r82_lane = tl.arange($(1024))
    r82_left = tl.load($(vbmm_1_ptr) + ((((((((((r82_row[:, None] * $(1024)) + r82_lane[None, :]) / $(1024)) % $(288)) * $(1024)) + (((r82_row[:, None] * $(1024)) + r82_lane[None, :]) % $(1024))) / $(128)) % $(8)) * $(36864)) + (((((r82_row[:, None] * $(1024)) + r82_lane[None, :]) / $(1024)) % $(288)) * $(128)) + (((((((r82_row[:, None] * $(1024)) + r82_lane[None, :]) / $(1024)) % $(288)) * $(1024)) + (((r82_row[:, None] * $(1024)) + r82_lane[None, :]) % $(1024))) % $(128))), dtype=tl.bfloat16)
    r82_right = tl.load($(vto_copy_6_ptr) + (((r82_col[None, :] % $(512)) * $(1024)) + (r82_lane[:, None] % $(1024))), dtype=tl.bfloat16)
    r82_value = tl.dot(r82_left, r82_right)
    r82_value = r82_value.to(tl.bfloat16)
    tl.store($(vmm_4_ptr) + ((r82_row[:, None] * $(512)) + r82_col[None, :]), r82_value)
    r83_z = tl.arange($(49152))
    r83_value = tl.load($(vmm_4_ptr) + (((((((tl.load($(vcat_4_ptr) + ((r83_z / $(512)) % $(96)), dtype=tl.int64) * $(512)) + (r83_z % $(512))) / $(512)) % $(288)) % $(288)) * $(512)) + ((((tl.load($(vcat_4_ptr) + ((r83_z / $(512)) % $(96)), dtype=tl.int64) * $(512)) + (r83_z % $(512))) % $(512)) % $(512))), dtype=tl.bfloat16)
    r83_value = r83_value.to(tl.bfloat16)
    r83_value = tl.broadcast_to(r83_value, [$(49152)])
    tl.store($(vindex_2_ptr) + r83_z, r83_value)
    z = tl.arange($(49152))
    value = tl.load($(vindex_2_ptr) + ((((z / $(512)) % $(96)) * $(512)) + (z % $(512))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(49152)])
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
  let vcat_ptr : RegionName := "vcat"
  let vdiv_ptr : RegionName := "vdiv"
  let vpow_1_ptr : RegionName := "vpow_1"
  let vreciprocal_ptr : RegionName := "vreciprocal"
  let vmul_7_ptr : RegionName := "vmul_7"
  let vmul_8_ptr : RegionName := "vmul_8"
  let vcat_1_ptr : RegionName := "vcat_1"
  let vcos_ptr : RegionName := "vcos"
  let vindex_ptr : RegionName := "vindex"
  let vsin_ptr : RegionName := "vsin"
  let vindex_1_ptr : RegionName := "vindex_1"
  let vto_copy_14_ptr : RegionName := "vto_copy_14"
  let vto_copy_15_ptr : RegionName := "vto_copy_15"
  let vcat_4_ptr : RegionName := "vcat_4"
  let vmm_ptr : RegionName := "vmm"
  let vto_copy_18_ptr : RegionName := "vto_copy_18"
  let vpow_2_ptr : RegionName := "vpow_2"
  let vmean_ptr : RegionName := "vmean"
  let vadd_2_ptr : RegionName := "vadd_2"
  let vrsqrt_ptr : RegionName := "vrsqrt"
  let vmul_9_ptr : RegionName := "vmul_9"
  let vmul_10_ptr : RegionName := "vmul_10"
  let vto_copy_19_ptr : RegionName := "vto_copy_19"
  let vmm_1_ptr : RegionName := "vmm_1"
  let vmm_2_ptr : RegionName := "vmm_2"
  let vto_copy_20_ptr : RegionName := "vto_copy_20"
  let vpow_3_ptr : RegionName := "vpow_3"
  let vmean_1_ptr : RegionName := "vmean_1"
  let vadd_3_ptr : RegionName := "vadd_3"
  let vrsqrt_1_ptr : RegionName := "vrsqrt_1"
  let vmul_11_ptr : RegionName := "vmul_11"
  let vmul_12_ptr : RegionName := "vmul_12"
  let vto_copy_21_ptr : RegionName := "vto_copy_21"
  let vmm_3_ptr : RegionName := "vmm_3"
  let vmul_13_ptr : RegionName := "vmul_13"
  let vmul_14_ptr : RegionName := "vmul_14"
  let vsub_13_ptr : RegionName := "vsub_13"
  let vmul_15_ptr : RegionName := "vmul_15"
  let vmul_16_ptr : RegionName := "vmul_16"
  let vadd_4_ptr : RegionName := "vadd_4"
  let vcat_5_ptr : RegionName := "vcat_5"
  let vmul_17_ptr : RegionName := "vmul_17"
  let vmul_18_ptr : RegionName := "vmul_18"
  let vsub_14_ptr : RegionName := "vsub_14"
  let vmul_19_ptr : RegionName := "vmul_19"
  let vmul_20_ptr : RegionName := "vmul_20"
  let vadd_5_ptr : RegionName := "vadd_5"
  let vcat_6_ptr : RegionName := "vcat_6"
  let vcat_7_ptr : RegionName := "vcat_7"
  let vcat_8_ptr : RegionName := "vcat_8"
  let vbmm_ptr : RegionName := "vbmm"
  let vmul_21_ptr : RegionName := "vmul_21"
  let vadd_6_ptr : RegionName := "vadd_6"
  let vto_copy_22_ptr : RegionName := "vto_copy_22"
  let vsoftmax_ptr : RegionName := "vsoftmax"
  let vto_copy_23_ptr : RegionName := "vto_copy_23"
  let vbmm_1_ptr : RegionName := "vbmm_1"
  let vmm_4_ptr : RegionName := "vmm_4"
  let vindex_2_ptr : RegionName := "vindex_2"
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
  let candidate_vdiv_ptr : RegionName := "candidate_vdiv"
  let candidate_vpow_1_ptr : RegionName := "candidate_vpow_1"
  let candidate_vreciprocal_ptr : RegionName := "candidate_vreciprocal"
  let candidate_vmul_7_ptr : RegionName := "candidate_vmul_7"
  let candidate_vmul_8_ptr : RegionName := "candidate_vmul_8"
  let candidate_vcat_1_ptr : RegionName := "candidate_vcat_1"
  let candidate_vcos_ptr : RegionName := "candidate_vcos"
  let candidate_vindex_ptr : RegionName := "candidate_vindex"
  let candidate_vsin_ptr : RegionName := "candidate_vsin"
  let candidate_vindex_1_ptr : RegionName := "candidate_vindex_1"
  let candidate_vto_copy_14_ptr : RegionName := "candidate_vto_copy_14"
  let candidate_vto_copy_15_ptr : RegionName := "candidate_vto_copy_15"
  let candidate_vcat_4_ptr : RegionName := "candidate_vcat_4"
  let candidate_vmm_ptr : RegionName := "candidate_vmm"
  let candidate_vto_copy_18_ptr : RegionName := "candidate_vto_copy_18"
  let candidate_vpow_2_ptr : RegionName := "candidate_vpow_2"
  let candidate_vmean_ptr : RegionName := "candidate_vmean"
  let candidate_vadd_2_ptr : RegionName := "candidate_vadd_2"
  let candidate_vrsqrt_ptr : RegionName := "candidate_vrsqrt"
  let candidate_vmul_9_ptr : RegionName := "candidate_vmul_9"
  let candidate_vmul_10_ptr : RegionName := "candidate_vmul_10"
  let candidate_vto_copy_19_ptr : RegionName := "candidate_vto_copy_19"
  let candidate_vmm_1_ptr : RegionName := "candidate_vmm_1"
  let candidate_vmm_2_ptr : RegionName := "candidate_vmm_2"
  let candidate_vto_copy_20_ptr : RegionName := "candidate_vto_copy_20"
  let candidate_vpow_3_ptr : RegionName := "candidate_vpow_3"
  let candidate_vmean_1_ptr : RegionName := "candidate_vmean_1"
  let candidate_vadd_3_ptr : RegionName := "candidate_vadd_3"
  let candidate_vrsqrt_1_ptr : RegionName := "candidate_vrsqrt_1"
  let candidate_vmul_11_ptr : RegionName := "candidate_vmul_11"
  let candidate_vmul_12_ptr : RegionName := "candidate_vmul_12"
  let candidate_vto_copy_21_ptr : RegionName := "candidate_vto_copy_21"
  let candidate_vmul_13_ptr : RegionName := "candidate_vmul_13"
  let candidate_vmul_14_ptr : RegionName := "candidate_vmul_14"
  let candidate_vsub_13_ptr : RegionName := "candidate_vsub_13"
  let candidate_vmul_15_ptr : RegionName := "candidate_vmul_15"
  let candidate_vmul_16_ptr : RegionName := "candidate_vmul_16"
  let candidate_vadd_4_ptr : RegionName := "candidate_vadd_4"
  let candidate_vcat_5_ptr : RegionName := "candidate_vcat_5"
  let candidate_vmul_17_ptr : RegionName := "candidate_vmul_17"
  let candidate_vmul_18_ptr : RegionName := "candidate_vmul_18"
  let candidate_vsub_14_ptr : RegionName := "candidate_vsub_14"
  let candidate_vmul_19_ptr : RegionName := "candidate_vmul_19"
  let candidate_vmul_20_ptr : RegionName := "candidate_vmul_20"
  let candidate_vadd_5_ptr : RegionName := "candidate_vadd_5"
  let candidate_vcat_6_ptr : RegionName := "candidate_vcat_6"
  let candidate_vcat_7_ptr : RegionName := "candidate_vcat_7"
  let candidate_vbmm_ptr : RegionName := "candidate_vbmm"
  let candidate_vto_copy_22_ptr : RegionName := "candidate_vto_copy_22"
  let candidate_vto_copy_23_ptr : RegionName := "candidate_vto_copy_23"
  let candidate_vbmm_1_ptr : RegionName := "candidate_vbmm_1"
  let candidate_vto_copy_24_ptr : RegionName := "candidate_vto_copy_24"
  let candidate_vto_copy_25_ptr : RegionName := "candidate_vto_copy_25"
  let candidate_vmm_3_ptr : RegionName := "candidate_vmm_3"
  let candidate_vadd_6_ptr : RegionName := "candidate_vadd_6"
  let candidate_vmul_21_ptr : RegionName := "candidate_vmul_21"
  let candidate_vadd_7_ptr : RegionName := "candidate_vadd_7"
  let candidate_vsub_15_ptr : RegionName := "candidate_vsub_15"
  let candidate_vgt_ptr : RegionName := "candidate_vgt"
  let candidate_vmasked_fill_ptr : RegionName := "candidate_vmasked_fill"
  let candidate_vsoftmax_ptr : RegionName := "candidate_vsoftmax"
  let candidate_vto_copy_26_ptr : RegionName := "candidate_vto_copy_26"
  let candidate_vmm_4_ptr : RegionName := "candidate_vmm_4"
  let candidate_vcat_8_ptr : RegionName := "candidate_vcat_8"
  let candidate_vbmm_2_ptr : RegionName := "candidate_vbmm_2"
  let candidate_vbmm_3_ptr : RegionName := "candidate_vbmm_3"
  let candidate_vto_copy_27_ptr : RegionName := "candidate_vto_copy_27"
  let candidate_vto_copy_28_ptr : RegionName := "candidate_vto_copy_28"
  let candidate_vbmm_4_ptr : RegionName := "candidate_vbmm_4"
  let candidate_vto_copy_29_ptr : RegionName := "candidate_vto_copy_29"
  let candidate_vto_copy_30_ptr : RegionName := "candidate_vto_copy_30"
  let candidate_vmm_5_ptr : RegionName := "candidate_vmm_5"
  let candidate_vadd_8_ptr : RegionName := "candidate_vadd_8"
  let candidate_vmul_22_ptr : RegionName := "candidate_vmul_22"
  let candidate_vadd_9_ptr : RegionName := "candidate_vadd_9"
  let candidate_vsub_16_ptr : RegionName := "candidate_vsub_16"
  let candidate_vgt_1_ptr : RegionName := "candidate_vgt_1"
  let candidate_vmasked_fill_1_ptr : RegionName := "candidate_vmasked_fill_1"
  let candidate_vsoftmax_1_ptr : RegionName := "candidate_vsoftmax_1"
  let candidate_vto_copy_31_ptr : RegionName := "candidate_vto_copy_31"
  let candidate_vmm_6_ptr : RegionName := "candidate_vmm_6"
  let candidate_vcat_9_ptr : RegionName := "candidate_vcat_9"
  let candidate_vbmm_5_ptr : RegionName := "candidate_vbmm_5"
  let candidate_vbmm_6_ptr : RegionName := "candidate_vbmm_6"
  let candidate_vto_copy_32_ptr : RegionName := "candidate_vto_copy_32"
  let candidate_vto_copy_33_ptr : RegionName := "candidate_vto_copy_33"
  let candidate_vbmm_7_ptr : RegionName := "candidate_vbmm_7"
  let candidate_vto_copy_34_ptr : RegionName := "candidate_vto_copy_34"
  let candidate_vto_copy_35_ptr : RegionName := "candidate_vto_copy_35"
  let candidate_vmm_7_ptr : RegionName := "candidate_vmm_7"
  let candidate_vadd_10_ptr : RegionName := "candidate_vadd_10"
  let candidate_vmul_23_ptr : RegionName := "candidate_vmul_23"
  let candidate_vadd_11_ptr : RegionName := "candidate_vadd_11"
  let candidate_vsub_17_ptr : RegionName := "candidate_vsub_17"
  let candidate_vgt_2_ptr : RegionName := "candidate_vgt_2"
  let candidate_vmasked_fill_2_ptr : RegionName := "candidate_vmasked_fill_2"
  let candidate_vsoftmax_2_ptr : RegionName := "candidate_vsoftmax_2"
  let candidate_vto_copy_36_ptr : RegionName := "candidate_vto_copy_36"
  let candidate_vmm_8_ptr : RegionName := "candidate_vmm_8"
  let candidate_vcat_10_ptr : RegionName := "candidate_vcat_10"
  let candidate_vbmm_8_ptr : RegionName := "candidate_vbmm_8"
  let candidate_vbmm_9_ptr : RegionName := "candidate_vbmm_9"
  let candidate_vto_copy_37_ptr : RegionName := "candidate_vto_copy_37"
  let candidate_vto_copy_38_ptr : RegionName := "candidate_vto_copy_38"
  let candidate_vbmm_10_ptr : RegionName := "candidate_vbmm_10"
  let candidate_vto_copy_39_ptr : RegionName := "candidate_vto_copy_39"
  let candidate_vto_copy_40_ptr : RegionName := "candidate_vto_copy_40"
  let candidate_vmm_9_ptr : RegionName := "candidate_vmm_9"
  let candidate_vadd_12_ptr : RegionName := "candidate_vadd_12"
  let candidate_vmul_24_ptr : RegionName := "candidate_vmul_24"
  let candidate_vadd_13_ptr : RegionName := "candidate_vadd_13"
  let candidate_vsub_18_ptr : RegionName := "candidate_vsub_18"
  let candidate_vgt_3_ptr : RegionName := "candidate_vgt_3"
  let candidate_vmasked_fill_3_ptr : RegionName := "candidate_vmasked_fill_3"
  let candidate_vsoftmax_3_ptr : RegionName := "candidate_vsoftmax_3"
  let candidate_vto_copy_41_ptr : RegionName := "candidate_vto_copy_41"
  let candidate_vmm_10_ptr : RegionName := "candidate_vmm_10"
  let candidate_vcat_11_ptr : RegionName := "candidate_vcat_11"
  let candidate_vbmm_11_ptr : RegionName := "candidate_vbmm_11"
  let candidate_vbmm_12_ptr : RegionName := "candidate_vbmm_12"
  let candidate_vto_copy_42_ptr : RegionName := "candidate_vto_copy_42"
  let candidate_vto_copy_43_ptr : RegionName := "candidate_vto_copy_43"
  let candidate_vbmm_13_ptr : RegionName := "candidate_vbmm_13"
  let candidate_vto_copy_44_ptr : RegionName := "candidate_vto_copy_44"
  let candidate_vto_copy_45_ptr : RegionName := "candidate_vto_copy_45"
  let candidate_vmm_11_ptr : RegionName := "candidate_vmm_11"
  let candidate_vadd_14_ptr : RegionName := "candidate_vadd_14"
  let candidate_vmul_25_ptr : RegionName := "candidate_vmul_25"
  let candidate_vadd_15_ptr : RegionName := "candidate_vadd_15"
  let candidate_vsub_19_ptr : RegionName := "candidate_vsub_19"
  let candidate_vgt_4_ptr : RegionName := "candidate_vgt_4"
  let candidate_vmasked_fill_4_ptr : RegionName := "candidate_vmasked_fill_4"
  let candidate_vsoftmax_4_ptr : RegionName := "candidate_vsoftmax_4"
  let candidate_vto_copy_46_ptr : RegionName := "candidate_vto_copy_46"
  let candidate_vmm_12_ptr : RegionName := "candidate_vmm_12"
  let candidate_vcat_12_ptr : RegionName := "candidate_vcat_12"
  let candidate_vbmm_14_ptr : RegionName := "candidate_vbmm_14"
  let candidate_vbmm_15_ptr : RegionName := "candidate_vbmm_15"
  let candidate_vto_copy_47_ptr : RegionName := "candidate_vto_copy_47"
  let candidate_vto_copy_48_ptr : RegionName := "candidate_vto_copy_48"
  let candidate_vbmm_16_ptr : RegionName := "candidate_vbmm_16"
  let candidate_vto_copy_49_ptr : RegionName := "candidate_vto_copy_49"
  let candidate_vto_copy_50_ptr : RegionName := "candidate_vto_copy_50"
  let candidate_vmm_13_ptr : RegionName := "candidate_vmm_13"
  let candidate_vadd_16_ptr : RegionName := "candidate_vadd_16"
  let candidate_vmul_26_ptr : RegionName := "candidate_vmul_26"
  let candidate_vadd_17_ptr : RegionName := "candidate_vadd_17"
  let candidate_vsub_20_ptr : RegionName := "candidate_vsub_20"
  let candidate_vgt_5_ptr : RegionName := "candidate_vgt_5"
  let candidate_vmasked_fill_5_ptr : RegionName := "candidate_vmasked_fill_5"
  let candidate_vsoftmax_5_ptr : RegionName := "candidate_vsoftmax_5"
  let candidate_vto_copy_51_ptr : RegionName := "candidate_vto_copy_51"
  let candidate_vmm_14_ptr : RegionName := "candidate_vmm_14"
  let candidate_vcat_13_ptr : RegionName := "candidate_vcat_13"
  let candidate_vbmm_17_ptr : RegionName := "candidate_vbmm_17"
  let candidate_vbmm_18_ptr : RegionName := "candidate_vbmm_18"
  let candidate_vto_copy_52_ptr : RegionName := "candidate_vto_copy_52"
  let candidate_vto_copy_53_ptr : RegionName := "candidate_vto_copy_53"
  let candidate_vbmm_19_ptr : RegionName := "candidate_vbmm_19"
  let candidate_vto_copy_54_ptr : RegionName := "candidate_vto_copy_54"
  let candidate_vto_copy_55_ptr : RegionName := "candidate_vto_copy_55"
  let candidate_vmm_15_ptr : RegionName := "candidate_vmm_15"
  let candidate_vadd_18_ptr : RegionName := "candidate_vadd_18"
  let candidate_vmul_27_ptr : RegionName := "candidate_vmul_27"
  let candidate_vadd_19_ptr : RegionName := "candidate_vadd_19"
  let candidate_vsub_21_ptr : RegionName := "candidate_vsub_21"
  let candidate_vgt_6_ptr : RegionName := "candidate_vgt_6"
  let candidate_vmasked_fill_6_ptr : RegionName := "candidate_vmasked_fill_6"
  let candidate_vsoftmax_6_ptr : RegionName := "candidate_vsoftmax_6"
  let candidate_vto_copy_56_ptr : RegionName := "candidate_vto_copy_56"
  let candidate_vmm_16_ptr : RegionName := "candidate_vmm_16"
  let candidate_vcat_14_ptr : RegionName := "candidate_vcat_14"
  let candidate_vbmm_20_ptr : RegionName := "candidate_vbmm_20"
  let candidate_vbmm_21_ptr : RegionName := "candidate_vbmm_21"
  let candidate_vto_copy_57_ptr : RegionName := "candidate_vto_copy_57"
  let candidate_vto_copy_58_ptr : RegionName := "candidate_vto_copy_58"
  let candidate_vbmm_22_ptr : RegionName := "candidate_vbmm_22"
  let candidate_vto_copy_59_ptr : RegionName := "candidate_vto_copy_59"
  let candidate_vto_copy_60_ptr : RegionName := "candidate_vto_copy_60"
  let candidate_vmm_17_ptr : RegionName := "candidate_vmm_17"
  let candidate_vadd_20_ptr : RegionName := "candidate_vadd_20"
  let candidate_vmul_28_ptr : RegionName := "candidate_vmul_28"
  let candidate_vadd_21_ptr : RegionName := "candidate_vadd_21"
  let candidate_vsub_22_ptr : RegionName := "candidate_vsub_22"
  let candidate_vgt_7_ptr : RegionName := "candidate_vgt_7"
  let candidate_vmasked_fill_7_ptr : RegionName := "candidate_vmasked_fill_7"
  let candidate_vsoftmax_7_ptr : RegionName := "candidate_vsoftmax_7"
  let candidate_vto_copy_61_ptr : RegionName := "candidate_vto_copy_61"
  let candidate_vmm_18_ptr : RegionName := "candidate_vmm_18"
  let candidate_vcat_15_ptr : RegionName := "candidate_vcat_15"
  let candidate_vbmm_23_ptr : RegionName := "candidate_vbmm_23"
  let candidate_vbmm_24_ptr : RegionName := "candidate_vbmm_24"
  let candidate_vto_copy_62_ptr : RegionName := "candidate_vto_copy_62"
  let candidate_vto_copy_63_ptr : RegionName := "candidate_vto_copy_63"
  let candidate_vbmm_25_ptr : RegionName := "candidate_vbmm_25"
  let candidate_vto_copy_64_ptr : RegionName := "candidate_vto_copy_64"
  let candidate_vto_copy_65_ptr : RegionName := "candidate_vto_copy_65"
  let candidate_vmm_19_ptr : RegionName := "candidate_vmm_19"
  let candidate_vadd_22_ptr : RegionName := "candidate_vadd_22"
  let candidate_vmul_29_ptr : RegionName := "candidate_vmul_29"
  let candidate_vadd_23_ptr : RegionName := "candidate_vadd_23"
  let candidate_vsub_23_ptr : RegionName := "candidate_vsub_23"
  let candidate_vgt_8_ptr : RegionName := "candidate_vgt_8"
  let candidate_vmasked_fill_8_ptr : RegionName := "candidate_vmasked_fill_8"
  let candidate_vsoftmax_8_ptr : RegionName := "candidate_vsoftmax_8"
  let candidate_vto_copy_66_ptr : RegionName := "candidate_vto_copy_66"
  let candidate_vmm_20_ptr : RegionName := "candidate_vmm_20"
  let candidate_vcat_16_ptr : RegionName := "candidate_vcat_16"
  let candidate_vbmm_26_ptr : RegionName := "candidate_vbmm_26"
  let candidate_vcat_17_ptr : RegionName := "candidate_vcat_17"
  let candidate_vmm_21_ptr : RegionName := "candidate_vmm_21"
  let candidate_vindex_2_ptr : RegionName := "candidate_vindex_2"
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
    r16_z = tl.arange($(288))
    r16_value = tl.where(r16_z < $(96), tl.cast((r16_z - $(0)), tl.int64), tl.where(r16_z < $(192), tl.cast((r16_z - $(96)), tl.int64), tl.cast((r16_z - $(192)), tl.int64)))
    r16_value = tl.broadcast_to(r16_value, [$(288)])
    tl.store($(candidate_vcat_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(32))
    r17_value = ((tl.cast(tl.toReal(r17_z), tl.float32) * 2.0) / 64.0)
    r17_value = r17_value.to(tl.float32)
    r17_value = tl.broadcast_to(r17_value, [$(32)])
    tl.store($(candidate_vdiv_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(32))
    r18_value = tl.extra.cuda.libdevice.pow(10000.0, tl.load($(candidate_vdiv_ptr) + r18_z, dtype=tl.float32))
    r18_value = r18_value.to(tl.float32)
    r18_value = tl.broadcast_to(r18_value, [$(32)])
    tl.store($(candidate_vpow_1_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(32))
    r19_value = 1.0 / tl.load($(candidate_vpow_1_ptr) + r19_z, dtype=tl.float32)
    r19_value = r19_value.to(tl.float32)
    r19_value = tl.broadcast_to(r19_value, [$(32)])
    tl.store($(candidate_vreciprocal_ptr) + r19_z, r19_value)
    r20_z = tl.arange($(32))
    r20_value = (tl.load($(candidate_vreciprocal_ptr) + r20_z, dtype=tl.float32) * 1.0)
    r20_value = r20_value.to(tl.float32)
    r20_value = tl.broadcast_to(r20_value, [$(32)])
    tl.store($(candidate_vmul_7_ptr) + r20_z, r20_value)
    r21_z = tl.arange($(9216))
    r21_value = (tl.cast(tl.toReal(((r21_z / $(32)) % $(288))), tl.float32) * tl.load($(candidate_vmul_7_ptr) + (r21_z % $(32)), dtype=tl.float32))
    r21_value = r21_value.to(tl.float32)
    r21_value = tl.broadcast_to(r21_value, [$(9216)])
    tl.store($(candidate_vmul_8_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(18432))
    r22_value = tl.where((r22_z % $(64)) < $(32), tl.load($(candidate_vmul_8_ptr) + ((((r22_z / $(64)) % $(288)) * $(32)) + (((r22_z % $(64)) - $(0)) % $(32))), dtype=tl.float32), tl.load($(candidate_vmul_8_ptr) + ((((r22_z / $(64)) % $(288)) * $(32)) + (((r22_z % $(64)) - $(32)) % $(32))), dtype=tl.float32))
    r22_value = r22_value.to(tl.float32)
    r22_value = tl.broadcast_to(r22_value, [$(18432)])
    tl.store($(candidate_vcat_1_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(18432))
    r23_value = tl.cos(tl.load($(candidate_vcat_1_ptr) + ((((r23_z / $(64)) % $(288)) * $(64)) + (r23_z % $(64))), dtype=tl.float32))
    r23_value = r23_value.to(tl.float32)
    r23_value = tl.broadcast_to(r23_value, [$(18432)])
    tl.store($(candidate_vcos_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(18432))
    r24_value = tl.load($(candidate_vcos_ptr) + (((((((tl.load($(candidate_vcat_ptr) + ((r24_z / $(64)) % $(288)), dtype=tl.int64) * $(64)) + (r24_z % $(64))) / $(64)) % $(288)) % $(288)) * $(64)) + ((((tl.load($(candidate_vcat_ptr) + ((r24_z / $(64)) % $(288)), dtype=tl.int64) * $(64)) + (r24_z % $(64))) % $(64)) % $(64))), dtype=tl.float32)
    r24_value = r24_value.to(tl.float32)
    r24_value = tl.broadcast_to(r24_value, [$(18432)])
    tl.store($(candidate_vindex_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(18432))
    r25_value = tl.sin(tl.load($(candidate_vcat_1_ptr) + ((((r25_z / $(64)) % $(288)) * $(64)) + (r25_z % $(64))), dtype=tl.float32))
    r25_value = r25_value.to(tl.float32)
    r25_value = tl.broadcast_to(r25_value, [$(18432)])
    tl.store($(candidate_vsin_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(18432))
    r26_value = tl.load($(candidate_vsin_ptr) + (((((((tl.load($(candidate_vcat_ptr) + ((r26_z / $(64)) % $(288)), dtype=tl.int64) * $(64)) + (r26_z % $(64))) / $(64)) % $(288)) % $(288)) * $(64)) + ((((tl.load($(candidate_vcat_ptr) + ((r26_z / $(64)) % $(288)), dtype=tl.int64) * $(64)) + (r26_z % $(64))) % $(64)) % $(64))), dtype=tl.float32)
    r26_value = r26_value.to(tl.float32)
    r26_value = tl.broadcast_to(r26_value, [$(18432)])
    tl.store($(candidate_vindex_1_ptr) + r26_z, r26_value)
    r27_z = tl.arange($(18432))
    r27_value = tl.load($(candidate_vindex_ptr) + ((((r27_z / $(64)) % $(288)) * $(64)) + (r27_z % $(64))), dtype=tl.float32)
    r27_value = r27_value.to(tl.bfloat16)
    r27_value = tl.broadcast_to(r27_value, [$(18432)])
    tl.store($(candidate_vto_copy_14_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(18432))
    r28_value = tl.load($(candidate_vindex_1_ptr) + ((((r28_z / $(64)) % $(288)) * $(64)) + (r28_z % $(64))), dtype=tl.float32)
    r28_value = r28_value.to(tl.bfloat16)
    r28_value = tl.broadcast_to(r28_value, [$(18432)])
    tl.store($(candidate_vto_copy_15_ptr) + r28_z, r28_value)
    r29_z = tl.arange($(96))
    r29_value = tl.where(r29_z < $(32), (tl.full([], $(64), dtype=tl.int64) + tl.cast((r29_z - $(0)), tl.int64)), tl.where(r29_z < $(64), (tl.full([], $(128), dtype=tl.int64) + tl.cast((r29_z - $(32)), tl.int64)), (tl.full([], $(192), dtype=tl.int64) + tl.cast((r29_z - $(64)), tl.int64))))
    r29_value = tl.broadcast_to(r29_value, [$(96)])
    tl.store($(candidate_vcat_4_ptr) + r29_z, r29_value)
    r30_row = tl.arange($(288))
    r30_col = tl.arange($(256))
    r30_lane = tl.arange($(512))
    r30_left = tl.load($(input0_ptr) + ((((((r30_row[:, None] * $(512)) + r30_lane[None, :]) / $(512)) % $(288)) * $(512)) + (((r30_row[:, None] * $(512)) + r30_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r30_right = tl.load($(candidate_vto_copy_ptr) + (((r30_col[None, :] % $(256)) * $(512)) + (r30_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r30_value = tl.dot(r30_left, r30_right)
    r30_value = r30_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_ptr) + ((r30_row[:, None] * $(256)) + r30_col[None, :]), r30_value)
    r31_z = tl.arange($(73728))
    r31_value = tl.load($(candidate_vmm_ptr) + ((((r31_z / $(256)) % $(288)) * $(256)) + (r31_z % $(256))), dtype=tl.bfloat16)
    r31_value = r31_value.to(tl.float32)
    r31_value = tl.broadcast_to(r31_value, [$(73728)])
    tl.store($(candidate_vto_copy_18_ptr) + r31_z, r31_value)
    r32_z = tl.arange($(73728))
    r32_value = ((tl.load($(candidate_vto_copy_18_ptr) + ((((r32_z / $(256)) % $(288)) * $(256)) + (r32_z % $(256))), dtype=tl.float32)) * (tl.load($(candidate_vto_copy_18_ptr) + ((((r32_z / $(256)) % $(288)) * $(256)) + (r32_z % $(256))), dtype=tl.float32)))
    r32_value = r32_value.to(tl.float32)
    r32_value = tl.broadcast_to(r32_value, [$(73728)])
    tl.store($(candidate_vpow_2_ptr) + r32_z, r32_value)
    r33_z = tl.arange($(288))
    r33_lane = tl.arange($(256))
    r33_value = (tl.sum(tl.load($(candidate_vpow_2_ptr) + ((r33_z[:, None] * $(256)) + r33_lane[None, :]), dtype=tl.float32), axis=1) / 256.0)
    r33_value = r33_value.to(tl.float32)
    r33_value = tl.broadcast_to(r33_value, [$(288)])
    tl.store($(candidate_vmean_ptr) + r33_z, r33_value)
    r34_z = tl.arange($(288))
    r34_value = (tl.load($(candidate_vmean_ptr) + r34_z, dtype=tl.float32) + 0.000001)
    r34_value = r34_value.to(tl.float32)
    r34_value = tl.broadcast_to(r34_value, [$(288)])
    tl.store($(candidate_vadd_2_ptr) + r34_z, r34_value)
    r35_z = tl.arange($(288))
    r35_value = tl.rsqrt(tl.load($(candidate_vadd_2_ptr) + r35_z, dtype=tl.float32))
    r35_value = r35_value.to(tl.float32)
    r35_value = tl.broadcast_to(r35_value, [$(288)])
    tl.store($(candidate_vrsqrt_ptr) + r35_z, r35_value)
    r36_z = tl.arange($(73728))
    r36_value = (tl.load($(candidate_vto_copy_18_ptr) + ((((r36_z / $(256)) % $(288)) * $(256)) + (r36_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_ptr) + ((r36_z / $(256)) % $(288)), dtype=tl.float32))
    r36_value = r36_value.to(tl.float32)
    r36_value = tl.broadcast_to(r36_value, [$(73728)])
    tl.store($(candidate_vmul_9_ptr) + r36_z, r36_value)
    r37_z = tl.arange($(73728))
    r37_value = (tl.cast(tl.load($(candidate_vto_copy_1_ptr) + (r37_z % $(256)), dtype=tl.bfloat16), tl.float32) * tl.load($(candidate_vmul_9_ptr) + ((((r37_z / $(256)) % $(288)) * $(256)) + (r37_z % $(256))), dtype=tl.float32))
    r37_value = r37_value.to(tl.float32)
    r37_value = tl.broadcast_to(r37_value, [$(73728)])
    tl.store($(candidate_vmul_10_ptr) + r37_z, r37_value)
    r38_z = tl.arange($(73728))
    r38_value = tl.load($(candidate_vmul_10_ptr) + ((((r38_z / $(256)) % $(288)) * $(256)) + (r38_z % $(256))), dtype=tl.float32)
    r38_value = r38_value.to(tl.bfloat16)
    r38_value = tl.broadcast_to(r38_value, [$(73728)])
    tl.store($(candidate_vto_copy_19_ptr) + r38_z, r38_value)
    r39_row = tl.arange($(288))
    r39_col = tl.arange($(1536))
    r39_lane = tl.arange($(256))
    r39_left = tl.load($(candidate_vto_copy_19_ptr) + ((((((r39_row[:, None] * $(256)) + r39_lane[None, :]) / $(256)) % $(288)) * $(256)) + (((r39_row[:, None] * $(256)) + r39_lane[None, :]) % $(256))), dtype=tl.bfloat16)
    r39_right = tl.load($(candidate_vto_copy_2_ptr) + (((r39_col[None, :] % $(1536)) * $(256)) + (r39_lane[:, None] % $(256))), dtype=tl.bfloat16)
    r39_value = tl.dot(r39_left, r39_right)
    r39_value = r39_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_1_ptr) + ((r39_row[:, None] * $(1536)) + r39_col[None, :]), r39_value)
    r40_row = tl.arange($(288))
    r40_col = tl.arange($(576))
    r40_lane = tl.arange($(512))
    r40_left = tl.load($(input0_ptr) + ((((((r40_row[:, None] * $(512)) + r40_lane[None, :]) / $(512)) % $(288)) * $(512)) + (((r40_row[:, None] * $(512)) + r40_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r40_right = tl.load($(candidate_vto_copy_3_ptr) + (((r40_col[None, :] % $(576)) * $(512)) + (r40_lane[:, None] % $(512))), dtype=tl.bfloat16)
    r40_value = tl.dot(r40_left, r40_right)
    r40_value = r40_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_2_ptr) + ((r40_row[:, None] * $(576)) + r40_col[None, :]), r40_value)
    r41_z = tl.arange($(147456))
    r41_value = tl.load($(candidate_vmm_2_ptr) + ((((r41_z / $(512)) % $(288)) * $(576)) + (r41_z % $(512))), dtype=tl.bfloat16)
    r41_value = r41_value.to(tl.float32)
    r41_value = tl.broadcast_to(r41_value, [$(147456)])
    tl.store($(candidate_vto_copy_20_ptr) + r41_z, r41_value)
    r42_z = tl.arange($(147456))
    r42_value = ((tl.load($(candidate_vto_copy_20_ptr) + ((((r42_z / $(512)) % $(288)) * $(512)) + (r42_z % $(512))), dtype=tl.float32)) * (tl.load($(candidate_vto_copy_20_ptr) + ((((r42_z / $(512)) % $(288)) * $(512)) + (r42_z % $(512))), dtype=tl.float32)))
    r42_value = r42_value.to(tl.float32)
    r42_value = tl.broadcast_to(r42_value, [$(147456)])
    tl.store($(candidate_vpow_3_ptr) + r42_z, r42_value)
    r43_z = tl.arange($(288))
    r43_lane = tl.arange($(512))
    r43_value = (tl.sum(tl.load($(candidate_vpow_3_ptr) + ((r43_z[:, None] * $(512)) + r43_lane[None, :]), dtype=tl.float32), axis=1) / 512.0)
    r43_value = r43_value.to(tl.float32)
    r43_value = tl.broadcast_to(r43_value, [$(288)])
    tl.store($(candidate_vmean_1_ptr) + r43_z, r43_value)
    r44_z = tl.arange($(288))
    r44_value = (tl.load($(candidate_vmean_1_ptr) + r44_z, dtype=tl.float32) + 0.000001)
    r44_value = r44_value.to(tl.float32)
    r44_value = tl.broadcast_to(r44_value, [$(288)])
    tl.store($(candidate_vadd_3_ptr) + r44_z, r44_value)
    r45_z = tl.arange($(288))
    r45_value = tl.rsqrt(tl.load($(candidate_vadd_3_ptr) + r45_z, dtype=tl.float32))
    r45_value = r45_value.to(tl.float32)
    r45_value = tl.broadcast_to(r45_value, [$(288)])
    tl.store($(candidate_vrsqrt_1_ptr) + r45_z, r45_value)
    r46_z = tl.arange($(147456))
    r46_value = (tl.load($(candidate_vto_copy_20_ptr) + ((((r46_z / $(512)) % $(288)) * $(512)) + (r46_z % $(512))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_1_ptr) + ((r46_z / $(512)) % $(288)), dtype=tl.float32))
    r46_value = r46_value.to(tl.float32)
    r46_value = tl.broadcast_to(r46_value, [$(147456)])
    tl.store($(candidate_vmul_11_ptr) + r46_z, r46_value)
    r47_z = tl.arange($(147456))
    r47_value = (tl.cast(tl.load($(candidate_vto_copy_4_ptr) + (r47_z % $(512)), dtype=tl.bfloat16), tl.float32) * tl.load($(candidate_vmul_11_ptr) + ((((r47_z / $(512)) % $(288)) * $(512)) + (r47_z % $(512))), dtype=tl.float32))
    r47_value = r47_value.to(tl.float32)
    r47_value = tl.broadcast_to(r47_value, [$(147456)])
    tl.store($(candidate_vmul_12_ptr) + r47_z, r47_value)
    r48_z = tl.arange($(147456))
    r48_value = tl.load($(candidate_vmul_12_ptr) + ((((r48_z / $(512)) % $(288)) * $(512)) + (r48_z % $(512))), dtype=tl.float32)
    r48_value = r48_value.to(tl.bfloat16)
    r48_value = tl.broadcast_to(r48_value, [$(147456)])
    tl.store($(candidate_vto_copy_21_ptr) + r48_z, r48_value)
    r49_z = tl.arange($(73728))
    r49_value = (tl.load($(candidate_vmm_1_ptr) + ((((r49_z / $(32)) % $(288)) * $(1536)) + (((r49_z / $(9216)) % $(8)) * $(192)) + ((r49_z % $(32)) * $(2)) + $(128)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_14_ptr) + ((((r49_z / $(32)) % $(288)) * $(64)) + (r49_z % $(32))), dtype=tl.bfloat16))
    r49_value = r49_value.to(tl.bfloat16)
    r49_value = tl.broadcast_to(r49_value, [$(73728)])
    tl.store($(candidate_vmul_13_ptr) + r49_z, r49_value)
    r50_z = tl.arange($(73728))
    r50_value = (tl.load($(candidate_vmm_1_ptr) + ((((r50_z / $(32)) % $(288)) * $(1536)) + (((r50_z / $(9216)) % $(8)) * $(192)) + ((r50_z % $(32)) * $(2)) + $(129)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_15_ptr) + ((((r50_z / $(32)) % $(288)) * $(64)) + (r50_z % $(32))), dtype=tl.bfloat16))
    r50_value = r50_value.to(tl.bfloat16)
    r50_value = tl.broadcast_to(r50_value, [$(73728)])
    tl.store($(candidate_vmul_14_ptr) + r50_z, r50_value)
    r51_z = tl.arange($(73728))
    r51_value = (tl.load($(candidate_vmul_13_ptr) + ((((r51_z / $(9216)) % $(8)) * $(9216)) + (((r51_z / $(32)) % $(288)) * $(32)) + (r51_z % $(32))), dtype=tl.bfloat16) - tl.load($(candidate_vmul_14_ptr) + ((((r51_z / $(9216)) % $(8)) * $(9216)) + (((r51_z / $(32)) % $(288)) * $(32)) + (r51_z % $(32))), dtype=tl.bfloat16))
    r51_value = r51_value.to(tl.bfloat16)
    r51_value = tl.broadcast_to(r51_value, [$(73728)])
    tl.store($(candidate_vsub_13_ptr) + r51_z, r51_value)
    r52_z = tl.arange($(73728))
    r52_value = (tl.load($(candidate_vmm_1_ptr) + ((((r52_z / $(32)) % $(288)) * $(1536)) + (((r52_z / $(9216)) % $(8)) * $(192)) + ((r52_z % $(32)) * $(2)) + $(129)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_14_ptr) + ((((r52_z / $(32)) % $(288)) * $(64)) + (r52_z % $(32))), dtype=tl.bfloat16))
    r52_value = r52_value.to(tl.bfloat16)
    r52_value = tl.broadcast_to(r52_value, [$(73728)])
    tl.store($(candidate_vmul_15_ptr) + r52_z, r52_value)
    r53_z = tl.arange($(73728))
    r53_value = (tl.load($(candidate_vmm_1_ptr) + ((((r53_z / $(32)) % $(288)) * $(1536)) + (((r53_z / $(9216)) % $(8)) * $(192)) + ((r53_z % $(32)) * $(2)) + $(128)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_15_ptr) + ((((r53_z / $(32)) % $(288)) * $(64)) + (r53_z % $(32))), dtype=tl.bfloat16))
    r53_value = r53_value.to(tl.bfloat16)
    r53_value = tl.broadcast_to(r53_value, [$(73728)])
    tl.store($(candidate_vmul_16_ptr) + r53_z, r53_value)
    r54_z = tl.arange($(73728))
    r54_value = (tl.load($(candidate_vmul_15_ptr) + ((((r54_z / $(9216)) % $(8)) * $(9216)) + (((r54_z / $(32)) % $(288)) * $(32)) + (r54_z % $(32))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_16_ptr) + ((((r54_z / $(9216)) % $(8)) * $(9216)) + (((r54_z / $(32)) % $(288)) * $(32)) + (r54_z % $(32))), dtype=tl.bfloat16))
    r54_value = r54_value.to(tl.bfloat16)
    r54_value = tl.broadcast_to(r54_value, [$(73728)])
    tl.store($(candidate_vadd_4_ptr) + r54_z, r54_value)
    r55_z = tl.arange($(147456))
    r55_value = tl.where((r55_z % $(64)) < $(32), tl.load($(candidate_vsub_13_ptr) + ((((r55_z / $(18432)) % $(8)) * $(9216)) + (((r55_z / $(64)) % $(288)) * $(32)) + (((r55_z % $(64)) - $(0)) % $(32))), dtype=tl.bfloat16), tl.load($(candidate_vadd_4_ptr) + ((((r55_z / $(18432)) % $(8)) * $(9216)) + (((r55_z / $(64)) % $(288)) * $(32)) + (((r55_z % $(64)) - $(32)) % $(32))), dtype=tl.bfloat16))
    r55_value = r55_value.to(tl.bfloat16)
    r55_value = tl.broadcast_to(r55_value, [$(147456)])
    tl.store($(candidate_vcat_5_ptr) + r55_z, r55_value)
    r56_z = tl.arange($(9216))
    r56_value = (tl.load($(candidate_vmm_2_ptr) + ((((r56_z / $(32)) % $(288)) * $(576)) + ((r56_z % $(32)) * $(2)) + $(512)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_14_ptr) + ((((r56_z / $(32)) % $(288)) * $(64)) + (r56_z % $(32))), dtype=tl.bfloat16))
    r56_value = r56_value.to(tl.bfloat16)
    r56_value = tl.broadcast_to(r56_value, [$(9216)])
    tl.store($(candidate_vmul_17_ptr) + r56_z, r56_value)
    r57_z = tl.arange($(9216))
    r57_value = (tl.load($(candidate_vmm_2_ptr) + ((((r57_z / $(32)) % $(288)) * $(576)) + ((r57_z % $(32)) * $(2)) + $(513)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_15_ptr) + ((((r57_z / $(32)) % $(288)) * $(64)) + (r57_z % $(32))), dtype=tl.bfloat16))
    r57_value = r57_value.to(tl.bfloat16)
    r57_value = tl.broadcast_to(r57_value, [$(9216)])
    tl.store($(candidate_vmul_18_ptr) + r57_z, r57_value)
    r58_z = tl.arange($(9216))
    r58_value = (tl.load($(candidate_vmul_17_ptr) + ((((r58_z / $(32)) % $(288)) * $(32)) + (r58_z % $(32))), dtype=tl.bfloat16) - tl.load($(candidate_vmul_18_ptr) + ((((r58_z / $(32)) % $(288)) * $(32)) + (r58_z % $(32))), dtype=tl.bfloat16))
    r58_value = r58_value.to(tl.bfloat16)
    r58_value = tl.broadcast_to(r58_value, [$(9216)])
    tl.store($(candidate_vsub_14_ptr) + r58_z, r58_value)
    r59_z = tl.arange($(9216))
    r59_value = (tl.load($(candidate_vmm_2_ptr) + ((((r59_z / $(32)) % $(288)) * $(576)) + ((r59_z % $(32)) * $(2)) + $(513)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_14_ptr) + ((((r59_z / $(32)) % $(288)) * $(64)) + (r59_z % $(32))), dtype=tl.bfloat16))
    r59_value = r59_value.to(tl.bfloat16)
    r59_value = tl.broadcast_to(r59_value, [$(9216)])
    tl.store($(candidate_vmul_19_ptr) + r59_z, r59_value)
    r60_z = tl.arange($(9216))
    r60_value = (tl.load($(candidate_vmm_2_ptr) + ((((r60_z / $(32)) % $(288)) * $(576)) + ((r60_z % $(32)) * $(2)) + $(512)), dtype=tl.bfloat16) * tl.load($(candidate_vto_copy_15_ptr) + ((((r60_z / $(32)) % $(288)) * $(64)) + (r60_z % $(32))), dtype=tl.bfloat16))
    r60_value = r60_value.to(tl.bfloat16)
    r60_value = tl.broadcast_to(r60_value, [$(9216)])
    tl.store($(candidate_vmul_20_ptr) + r60_z, r60_value)
    r61_z = tl.arange($(9216))
    r61_value = (tl.load($(candidate_vmul_19_ptr) + ((((r61_z / $(32)) % $(288)) * $(32)) + (r61_z % $(32))), dtype=tl.bfloat16) + tl.load($(candidate_vmul_20_ptr) + ((((r61_z / $(32)) % $(288)) * $(32)) + (r61_z % $(32))), dtype=tl.bfloat16))
    r61_value = r61_value.to(tl.bfloat16)
    r61_value = tl.broadcast_to(r61_value, [$(9216)])
    tl.store($(candidate_vadd_5_ptr) + r61_z, r61_value)
    r62_z = tl.arange($(18432))
    r62_value = tl.where((r62_z % $(64)) < $(32), tl.load($(candidate_vsub_14_ptr) + ((((r62_z / $(64)) % $(288)) * $(32)) + (((r62_z % $(64)) - $(0)) % $(32))), dtype=tl.bfloat16), tl.load($(candidate_vadd_5_ptr) + ((((r62_z / $(64)) % $(288)) * $(32)) + (((r62_z % $(64)) - $(32)) % $(32))), dtype=tl.bfloat16))
    r62_value = r62_value.to(tl.bfloat16)
    r62_value = tl.broadcast_to(r62_value, [$(18432)])
    tl.store($(candidate_vcat_6_ptr) + r62_z, r62_value)
    r63_z = tl.arange($(165888))
    r63_value = tl.where((r63_z % $(576)) < $(512), tl.load($(candidate_vto_copy_21_ptr) + ((((((((r63_z / $(576)) % $(288)) * $(512)) + ((r63_z % $(576)) - $(0))) / $(512)) % $(288)) * $(512)) + (((((r63_z / $(576)) % $(288)) * $(512)) + ((r63_z % $(576)) - $(0))) % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vcat_6_ptr) + ((((((((r63_z / $(576)) % $(288)) * $(64)) + ((r63_z % $(576)) - $(512))) / $(64)) % $(288)) * $(64)) + (((((r63_z / $(576)) % $(288)) * $(64)) + ((r63_z % $(576)) - $(512))) % $(64))), dtype=tl.bfloat16))
    r63_value = r63_value.to(tl.bfloat16)
    r63_value = tl.broadcast_to(r63_value, [$(165888)])
    tl.store($(candidate_vcat_7_ptr) + r63_z, r63_value)
    r64_row = tl.arange($(32))
    r64_col = tl.arange($(512))
    r64_lane = tl.arange($(128))
    for r64_group in range($(8)) {
      r64_left = tl.load($(candidate_vmm_1_ptr) + (((((((r64_group % $(8)) * $(4096)) + (r64_row[:, None] * $(128)) + r64_lane[None, :]) / $(128)) % $(32)) * $(1536)) + ((((((r64_group % $(8)) * $(4096)) + (r64_row[:, None] * $(128)) + r64_lane[None, :]) / $(4096)) % $(8)) * $(192)) + ((((r64_group % $(8)) * $(4096)) + (r64_row[:, None] * $(128)) + r64_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r64_right = tl.load($(candidate_vto_copy_5_ptr) + (((((((r64_group % $(8)) * $(65536)) + (r64_lane[:, None] * $(512)) + r64_col[None, :]) / $(65536)) % $(8)) * $(131072)) + ((((((r64_group % $(8)) * $(65536)) + (r64_lane[:, None] * $(512)) + r64_col[None, :]) / $(512)) % $(128)) * $(512)) + ((((r64_group % $(8)) * $(65536)) + (r64_lane[:, None] * $(512)) + r64_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r64_value = tl.dot(r64_left, r64_right)
      r64_value = r64_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_ptr) + ((r64_group * $(16384)) + (r64_row[:, None] * $(512)) + r64_col[None, :]), r64_value)
    }
    r65_z = tl.arange($(16384))
    r65_value = tl.load($(candidate_vcat_5_ptr) + ((((r65_z / $(2048)) % $(8)) * $(18432)) + (((r65_z / $(64)) % $(32)) * $(64)) + (r65_z % $(64))), dtype=tl.bfloat16)
    r65_value = r65_value.to(tl.float32)
    r65_value = tl.broadcast_to(r65_value, [$(16384)])
    tl.store($(candidate_vto_copy_22_ptr) + r65_z, r65_value)
    r66_z = tl.arange($(2048))
    r66_value = tl.load($(candidate_vcat_7_ptr) + ((((r66_z / $(64)) % $(32)) * $(576)) + (r66_z % $(64)) + $(512)), dtype=tl.bfloat16)
    r66_value = r66_value.to(tl.float32)
    r66_value = tl.broadcast_to(r66_value, [$(2048)])
    tl.store($(candidate_vto_copy_23_ptr) + r66_z, r66_value)
    r67_row = tl.arange($(32))
    r67_col = tl.arange($(32))
    r67_lane = tl.arange($(64))
    for r67_group in range($(8)) {
      r67_left = tl.load($(candidate_vto_copy_22_ptr) + (((((((r67_group % $(8)) * $(2048)) + (r67_row[:, None] * $(64)) + r67_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r67_group % $(8)) * $(2048)) + (r67_row[:, None] * $(64)) + r67_lane[None, :]) / $(64)) % $(32)) * $(64)) + ((((r67_group % $(8)) * $(2048)) + (r67_row[:, None] * $(64)) + r67_lane[None, :]) % $(64))), dtype=tl.float32)
      r67_right = tl.load($(candidate_vto_copy_23_ptr) + ((((((r67_group % $(8)) * $(2048)) + (r67_lane[:, None] * $(32)) + r67_col[None, :]) % $(32)) * $(64)) + (((((r67_group % $(8)) * $(2048)) + (r67_lane[:, None] * $(32)) + r67_col[None, :]) / $(32)) % $(64))), dtype=tl.float32)
      r67_value = tl.dot(r67_left, r67_right)
      r67_value = r67_value.to(tl.float32)
      tl.store($(candidate_vbmm_1_ptr) + ((r67_group * $(1024)) + (r67_row[:, None] * $(32)) + r67_col[None, :]), r67_value)
    }
    r68_z = tl.arange($(131072))
    r68_value = tl.load($(candidate_vbmm_ptr) + ((((r68_z / $(16384)) % $(8)) * $(16384)) + (((r68_z / $(512)) % $(32)) * $(512)) + (r68_z % $(512))), dtype=tl.bfloat16)
    r68_value = r68_value.to(tl.float32)
    r68_value = tl.broadcast_to(r68_value, [$(131072)])
    tl.store($(candidate_vto_copy_24_ptr) + r68_z, r68_value)
    r69_z = tl.arange($(16384))
    r69_value = tl.load($(candidate_vcat_7_ptr) + ((((r69_z / $(512)) % $(32)) * $(576)) + (r69_z % $(512))), dtype=tl.bfloat16)
    r69_value = r69_value.to(tl.float32)
    r69_value = tl.broadcast_to(r69_value, [$(16384)])
    tl.store($(candidate_vto_copy_25_ptr) + r69_z, r69_value)
    r70_row = tl.arange($(256))
    r70_col = tl.arange($(32))
    r70_lane = tl.arange($(512))
    r70_left = tl.load($(candidate_vto_copy_24_ptr) + ((((((r70_row[:, None] * $(512)) + r70_lane[None, :]) / $(16384)) % $(8)) * $(16384)) + (((((r70_row[:, None] * $(512)) + r70_lane[None, :]) / $(512)) % $(32)) * $(512)) + (((r70_row[:, None] * $(512)) + r70_lane[None, :]) % $(512))), dtype=tl.float32)
    r70_right = tl.load($(candidate_vto_copy_25_ptr) + (((((r70_lane[:, None] * $(32)) + r70_col[None, :]) % $(32)) * $(512)) + ((((r70_lane[:, None] * $(32)) + r70_col[None, :]) / $(32)) % $(512))), dtype=tl.float32)
    r70_value = tl.dot(r70_left, r70_right)
    r70_value = r70_value.to(tl.float32)
    tl.store($(candidate_vmm_3_ptr) + ((r70_row[:, None] * $(32)) + r70_col[None, :]), r70_value)
    r71_z = tl.arange($(8192))
    r71_value = (tl.load($(candidate_vbmm_1_ptr) + ((((r71_z / $(1024)) % $(8)) * $(1024)) + (((r71_z / $(32)) % $(32)) * $(32)) + (r71_z % $(32))), dtype=tl.float32) + tl.load($(candidate_vmm_3_ptr) + ((((r71_z / $(1024)) % $(8)) * $(1024)) + (((r71_z / $(32)) % $(32)) * $(32)) + (r71_z % $(32))), dtype=tl.float32))
    r71_value = r71_value.to(tl.float32)
    r71_value = tl.broadcast_to(r71_value, [$(8192)])
    tl.store($(candidate_vadd_6_ptr) + r71_z, r71_value)
    r72_z = tl.arange($(8192))
    r72_value = (tl.load($(candidate_vadd_6_ptr) + ((((r72_z / $(1024)) % $(8)) * $(1024)) + (((r72_z / $(32)) % $(32)) * $(32)) + (r72_z % $(32))), dtype=tl.float32) * 0.07216878364870322)
    r72_value = r72_value.to(tl.float32)
    r72_value = tl.broadcast_to(r72_value, [$(8192)])
    tl.store($(candidate_vmul_21_ptr) + r72_z, r72_value)
    r73_z = tl.arange($(32))
    r73_value = (tl.cast(r73_z, tl.int64) + tl.full([], $(32), dtype=tl.int64))
    r73_value = tl.broadcast_to(r73_value, [$(32)])
    tl.store($(candidate_vadd_7_ptr) + r73_z, r73_value)
    r74_z = tl.arange($(32))
    r74_value = (tl.load($(candidate_vadd_7_ptr) + r74_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r74_value = tl.broadcast_to(r74_value, [$(32)])
    tl.store($(candidate_vsub_15_ptr) + r74_z, r74_value)
    r75_z = tl.arange($(1024))
    r75_value = (tl.cast((r75_z % $(32)), tl.int64) > tl.load($(candidate_vsub_15_ptr) + ((r75_z / $(32)) % $(32)), dtype=tl.int64))
    r75_value = tl.broadcast_to(r75_value, [$(1024)])
    tl.store($(candidate_vgt_ptr) + r75_z, r75_value)
    r76_z = tl.arange($(8192))
    r76_value = tl.where(tl.load($(candidate_vgt_ptr) + ((((r76_z / $(32)) % $(32)) * $(32)) + (r76_z % $(32))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_21_ptr) + ((((r76_z / $(1024)) % $(8)) * $(1024)) + (((r76_z / $(32)) % $(32)) * $(32)) + (r76_z % $(32))), dtype=tl.float32))
    r76_value = r76_value.to(tl.float32)
    r76_value = tl.broadcast_to(r76_value, [$(8192)])
    tl.store($(candidate_vmasked_fill_ptr) + r76_z, r76_value)
    r77_z = tl.arange($(8192))
    r77_lane = tl.arange($(32))
    r77_row_values = tl.load($(candidate_vmasked_fill_ptr) + ((((r77_z[:, None] / $(1024)) % $(8)) * $(1024)) + (((r77_z[:, None] / $(32)) % $(32)) * $(32)) + (r77_lane[None, :] % $(32))), dtype=tl.float32)
    r77_maximum = tl.max(r77_row_values, axis=1, keep_dims=true)
    r77_weights = tl.exp(r77_row_values - r77_maximum)
    r77_total = tl.sum(r77_weights, axis=1)
    r77_maximum = tl.max(r77_row_values, axis=1)
    r77_value = tl.exp(tl.load($(candidate_vmasked_fill_ptr) + ((((r77_z / $(1024)) % $(8)) * $(1024)) + (((r77_z / $(32)) % $(32)) * $(32)) + (r77_z % $(32))), dtype=tl.float32) - r77_maximum) / r77_total
    r77_value = r77_value.to(tl.float32)
    r77_value = tl.broadcast_to(r77_value, [$(8192)])
    tl.store($(candidate_vsoftmax_ptr) + r77_z, r77_value)
    r78_z = tl.arange($(8192))
    r78_value = tl.load($(candidate_vsoftmax_ptr) + ((((r78_z / $(1024)) % $(8)) * $(1024)) + (((r78_z / $(32)) % $(32)) * $(32)) + (r78_z % $(32))), dtype=tl.float32)
    r78_value = r78_value.to(tl.bfloat16)
    r78_value = tl.broadcast_to(r78_value, [$(8192)])
    tl.store($(candidate_vto_copy_26_ptr) + r78_z, r78_value)
    r79_row = tl.arange($(256))
    r79_col = tl.arange($(512))
    r79_lane = tl.arange($(32))
    r79_left = tl.load($(candidate_vto_copy_26_ptr) + ((((((r79_row[:, None] * $(32)) + r79_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + (((((r79_row[:, None] * $(32)) + r79_lane[None, :]) / $(32)) % $(32)) * $(32)) + (((r79_row[:, None] * $(32)) + r79_lane[None, :]) % $(32))), dtype=tl.bfloat16)
    r79_right = tl.load($(candidate_vcat_7_ptr) + ((((((r79_lane[:, None] * $(512)) + r79_col[None, :]) / $(512)) % $(32)) * $(576)) + (((r79_lane[:, None] * $(512)) + r79_col[None, :]) % $(512))), dtype=tl.bfloat16)
    r79_value = tl.dot(r79_left, r79_right)
    r79_value = r79_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_4_ptr) + ((r79_row[:, None] * $(512)) + r79_col[None, :]), r79_value)
    r80_z = tl.arange($(131072))
    r80_value = tl.load($(candidate_vmm_4_ptr) + (((((((r80_z / $(512)) % $(8)) * $(32)) + (((r80_z / $(4096)) % $(32)) - $(0))) % $(256)) * $(512)) + (r80_z % $(512))), dtype=tl.bfloat16)
    r80_value = r80_value.to(tl.bfloat16)
    r80_value = tl.broadcast_to(r80_value, [$(131072)])
    tl.store($(candidate_vcat_8_ptr) + r80_z, r80_value)
    r81_row = tl.arange($(32))
    r81_col = tl.arange($(128))
    r81_lane = tl.arange($(512))
    for r81_group in range($(8)) {
      r81_left = tl.load($(candidate_vcat_8_ptr) + (((((((r81_group % $(8)) * $(16384)) + (r81_row[:, None] * $(512)) + r81_lane[None, :]) / $(512)) % $(32)) * $(4096)) + ((((((r81_group % $(8)) * $(16384)) + (r81_row[:, None] * $(512)) + r81_lane[None, :]) / $(16384)) % $(8)) * $(512)) + ((((r81_group % $(8)) * $(16384)) + (r81_row[:, None] * $(512)) + r81_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r81_right = tl.load($(candidate_vto_copy_5_ptr) + (((((((r81_group % $(8)) * $(65536)) + (r81_lane[:, None] * $(128)) + r81_col[None, :]) / $(65536)) % $(8)) * $(131072)) + (((((r81_group % $(8)) * $(65536)) + (r81_lane[:, None] * $(128)) + r81_col[None, :]) % $(128)) * $(512)) + (((((r81_group % $(8)) * $(65536)) + (r81_lane[:, None] * $(128)) + r81_col[None, :]) / $(128)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r81_value = tl.dot(r81_left, r81_right)
      r81_value = r81_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_2_ptr) + ((r81_group * $(4096)) + (r81_row[:, None] * $(128)) + r81_col[None, :]), r81_value)
    }
    r82_row = tl.arange($(32))
    r82_col = tl.arange($(512))
    r82_lane = tl.arange($(128))
    for r82_group in range($(8)) {
      r82_left = tl.load($(candidate_vmm_1_ptr) + (((((((r82_group % $(8)) * $(4096)) + (r82_row[:, None] * $(128)) + r82_lane[None, :]) / $(128)) % $(32)) * $(1536)) + ((((((r82_group % $(8)) * $(4096)) + (r82_row[:, None] * $(128)) + r82_lane[None, :]) / $(4096)) % $(8)) * $(192)) + ((((r82_group % $(8)) * $(4096)) + (r82_row[:, None] * $(128)) + r82_lane[None, :]) % $(128)) + $(49152)), dtype=tl.bfloat16)
      r82_right = tl.load($(candidate_vto_copy_5_ptr) + (((((((r82_group % $(8)) * $(65536)) + (r82_lane[:, None] * $(512)) + r82_col[None, :]) / $(65536)) % $(8)) * $(131072)) + ((((((r82_group % $(8)) * $(65536)) + (r82_lane[:, None] * $(512)) + r82_col[None, :]) / $(512)) % $(128)) * $(512)) + ((((r82_group % $(8)) * $(65536)) + (r82_lane[:, None] * $(512)) + r82_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r82_value = tl.dot(r82_left, r82_right)
      r82_value = r82_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_3_ptr) + ((r82_group * $(16384)) + (r82_row[:, None] * $(512)) + r82_col[None, :]), r82_value)
    }
    r83_z = tl.arange($(16384))
    r83_value = tl.load($(candidate_vcat_5_ptr) + ((((r83_z / $(2048)) % $(8)) * $(18432)) + (((r83_z / $(64)) % $(32)) * $(64)) + (r83_z % $(64)) + $(2048)), dtype=tl.bfloat16)
    r83_value = r83_value.to(tl.float32)
    r83_value = tl.broadcast_to(r83_value, [$(16384)])
    tl.store($(candidate_vto_copy_27_ptr) + r83_z, r83_value)
    r84_z = tl.arange($(4096))
    r84_value = tl.load($(candidate_vcat_7_ptr) + ((((r84_z / $(64)) % $(64)) * $(576)) + (r84_z % $(64)) + $(512)), dtype=tl.bfloat16)
    r84_value = r84_value.to(tl.float32)
    r84_value = tl.broadcast_to(r84_value, [$(4096)])
    tl.store($(candidate_vto_copy_28_ptr) + r84_z, r84_value)
    r85_row = tl.arange($(32))
    r85_col = tl.arange($(64))
    r85_lane = tl.arange($(64))
    for r85_group in range($(8)) {
      r85_left = tl.load($(candidate_vto_copy_27_ptr) + (((((((r85_group % $(8)) * $(2048)) + (r85_row[:, None] * $(64)) + r85_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r85_group % $(8)) * $(2048)) + (r85_row[:, None] * $(64)) + r85_lane[None, :]) / $(64)) % $(32)) * $(64)) + ((((r85_group % $(8)) * $(2048)) + (r85_row[:, None] * $(64)) + r85_lane[None, :]) % $(64))), dtype=tl.float32)
      r85_right = tl.load($(candidate_vto_copy_28_ptr) + ((((((r85_group % $(8)) * $(4096)) + (r85_lane[:, None] * $(64)) + r85_col[None, :]) % $(64)) * $(64)) + (((((r85_group % $(8)) * $(4096)) + (r85_lane[:, None] * $(64)) + r85_col[None, :]) / $(64)) % $(64))), dtype=tl.float32)
      r85_value = tl.dot(r85_left, r85_right)
      r85_value = r85_value.to(tl.float32)
      tl.store($(candidate_vbmm_4_ptr) + ((r85_group * $(2048)) + (r85_row[:, None] * $(64)) + r85_col[None, :]), r85_value)
    }
    r86_z = tl.arange($(131072))
    r86_value = tl.load($(candidate_vbmm_3_ptr) + ((((r86_z / $(16384)) % $(8)) * $(16384)) + (((r86_z / $(512)) % $(32)) * $(512)) + (r86_z % $(512))), dtype=tl.bfloat16)
    r86_value = r86_value.to(tl.float32)
    r86_value = tl.broadcast_to(r86_value, [$(131072)])
    tl.store($(candidate_vto_copy_29_ptr) + r86_z, r86_value)
    r87_z = tl.arange($(32768))
    r87_value = tl.load($(candidate_vcat_7_ptr) + ((((r87_z / $(512)) % $(64)) * $(576)) + (r87_z % $(512))), dtype=tl.bfloat16)
    r87_value = r87_value.to(tl.float32)
    r87_value = tl.broadcast_to(r87_value, [$(32768)])
    tl.store($(candidate_vto_copy_30_ptr) + r87_z, r87_value)
    r88_row = tl.arange($(256))
    r88_col = tl.arange($(64))
    r88_lane = tl.arange($(512))
    r88_left = tl.load($(candidate_vto_copy_29_ptr) + ((((((r88_row[:, None] * $(512)) + r88_lane[None, :]) / $(16384)) % $(8)) * $(16384)) + (((((r88_row[:, None] * $(512)) + r88_lane[None, :]) / $(512)) % $(32)) * $(512)) + (((r88_row[:, None] * $(512)) + r88_lane[None, :]) % $(512))), dtype=tl.float32)
    r88_right = tl.load($(candidate_vto_copy_30_ptr) + (((((r88_lane[:, None] * $(64)) + r88_col[None, :]) % $(64)) * $(512)) + ((((r88_lane[:, None] * $(64)) + r88_col[None, :]) / $(64)) % $(512))), dtype=tl.float32)
    r88_value = tl.dot(r88_left, r88_right)
    r88_value = r88_value.to(tl.float32)
    tl.store($(candidate_vmm_5_ptr) + ((r88_row[:, None] * $(64)) + r88_col[None, :]), r88_value)
    r89_z = tl.arange($(16384))
    r89_value = (tl.load($(candidate_vbmm_4_ptr) + ((((r89_z / $(2048)) % $(8)) * $(2048)) + (((r89_z / $(64)) % $(32)) * $(64)) + (r89_z % $(64))), dtype=tl.float32) + tl.load($(candidate_vmm_5_ptr) + ((((r89_z / $(2048)) % $(8)) * $(2048)) + (((r89_z / $(64)) % $(32)) * $(64)) + (r89_z % $(64))), dtype=tl.float32))
    r89_value = r89_value.to(tl.float32)
    r89_value = tl.broadcast_to(r89_value, [$(16384)])
    tl.store($(candidate_vadd_8_ptr) + r89_z, r89_value)
    r90_z = tl.arange($(16384))
    r90_value = (tl.load($(candidate_vadd_8_ptr) + ((((r90_z / $(2048)) % $(8)) * $(2048)) + (((r90_z / $(64)) % $(32)) * $(64)) + (r90_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r90_value = r90_value.to(tl.float32)
    r90_value = tl.broadcast_to(r90_value, [$(16384)])
    tl.store($(candidate_vmul_22_ptr) + r90_z, r90_value)
    r91_z = tl.arange($(32))
    r91_value = (tl.cast(r91_z, tl.int64) + tl.full([], $(64), dtype=tl.int64))
    r91_value = tl.broadcast_to(r91_value, [$(32)])
    tl.store($(candidate_vadd_9_ptr) + r91_z, r91_value)
    r92_z = tl.arange($(32))
    r92_value = (tl.load($(candidate_vadd_9_ptr) + r92_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r92_value = tl.broadcast_to(r92_value, [$(32)])
    tl.store($(candidate_vsub_16_ptr) + r92_z, r92_value)
    r93_z = tl.arange($(2048))
    r93_value = (tl.cast((r93_z % $(64)), tl.int64) > tl.load($(candidate_vsub_16_ptr) + ((r93_z / $(64)) % $(32)), dtype=tl.int64))
    r93_value = tl.broadcast_to(r93_value, [$(2048)])
    tl.store($(candidate_vgt_1_ptr) + r93_z, r93_value)
    r94_z = tl.arange($(16384))
    r94_value = tl.where(tl.load($(candidate_vgt_1_ptr) + ((((r94_z / $(64)) % $(32)) * $(64)) + (r94_z % $(64))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_22_ptr) + ((((r94_z / $(2048)) % $(8)) * $(2048)) + (((r94_z / $(64)) % $(32)) * $(64)) + (r94_z % $(64))), dtype=tl.float32))
    r94_value = r94_value.to(tl.float32)
    r94_value = tl.broadcast_to(r94_value, [$(16384)])
    tl.store($(candidate_vmasked_fill_1_ptr) + r94_z, r94_value)
    r95_z = tl.arange($(16384))
    r95_lane = tl.arange($(64))
    r95_row_values = tl.load($(candidate_vmasked_fill_1_ptr) + ((((r95_z[:, None] / $(2048)) % $(8)) * $(2048)) + (((r95_z[:, None] / $(64)) % $(32)) * $(64)) + (r95_lane[None, :] % $(64))), dtype=tl.float32)
    r95_maximum = tl.max(r95_row_values, axis=1, keep_dims=true)
    r95_weights = tl.exp(r95_row_values - r95_maximum)
    r95_total = tl.sum(r95_weights, axis=1)
    r95_maximum = tl.max(r95_row_values, axis=1)
    r95_value = tl.exp(tl.load($(candidate_vmasked_fill_1_ptr) + ((((r95_z / $(2048)) % $(8)) * $(2048)) + (((r95_z / $(64)) % $(32)) * $(64)) + (r95_z % $(64))), dtype=tl.float32) - r95_maximum) / r95_total
    r95_value = r95_value.to(tl.float32)
    r95_value = tl.broadcast_to(r95_value, [$(16384)])
    tl.store($(candidate_vsoftmax_1_ptr) + r95_z, r95_value)
    r96_z = tl.arange($(16384))
    r96_value = tl.load($(candidate_vsoftmax_1_ptr) + ((((r96_z / $(2048)) % $(8)) * $(2048)) + (((r96_z / $(64)) % $(32)) * $(64)) + (r96_z % $(64))), dtype=tl.float32)
    r96_value = r96_value.to(tl.bfloat16)
    r96_value = tl.broadcast_to(r96_value, [$(16384)])
    tl.store($(candidate_vto_copy_31_ptr) + r96_z, r96_value)
    r97_row = tl.arange($(256))
    r97_col = tl.arange($(512))
    r97_lane = tl.arange($(64))
    r97_left = tl.load($(candidate_vto_copy_31_ptr) + ((((((r97_row[:, None] * $(64)) + r97_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + (((((r97_row[:, None] * $(64)) + r97_lane[None, :]) / $(64)) % $(32)) * $(64)) + (((r97_row[:, None] * $(64)) + r97_lane[None, :]) % $(64))), dtype=tl.bfloat16)
    r97_right = tl.load($(candidate_vcat_7_ptr) + ((((((r97_lane[:, None] * $(512)) + r97_col[None, :]) / $(512)) % $(64)) * $(576)) + (((r97_lane[:, None] * $(512)) + r97_col[None, :]) % $(512))), dtype=tl.bfloat16)
    r97_value = tl.dot(r97_left, r97_right)
    r97_value = r97_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_6_ptr) + ((r97_row[:, None] * $(512)) + r97_col[None, :]), r97_value)
    r98_z = tl.arange($(131072))
    r98_value = tl.load($(candidate_vmm_6_ptr) + (((((((r98_z / $(512)) % $(8)) * $(32)) + (((r98_z / $(4096)) % $(32)) - $(0))) % $(256)) * $(512)) + (r98_z % $(512))), dtype=tl.bfloat16)
    r98_value = r98_value.to(tl.bfloat16)
    r98_value = tl.broadcast_to(r98_value, [$(131072)])
    tl.store($(candidate_vcat_9_ptr) + r98_z, r98_value)
    r99_row = tl.arange($(32))
    r99_col = tl.arange($(128))
    r99_lane = tl.arange($(512))
    for r99_group in range($(8)) {
      r99_left = tl.load($(candidate_vcat_9_ptr) + (((((((r99_group % $(8)) * $(16384)) + (r99_row[:, None] * $(512)) + r99_lane[None, :]) / $(512)) % $(32)) * $(4096)) + ((((((r99_group % $(8)) * $(16384)) + (r99_row[:, None] * $(512)) + r99_lane[None, :]) / $(16384)) % $(8)) * $(512)) + ((((r99_group % $(8)) * $(16384)) + (r99_row[:, None] * $(512)) + r99_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r99_right = tl.load($(candidate_vto_copy_5_ptr) + (((((((r99_group % $(8)) * $(65536)) + (r99_lane[:, None] * $(128)) + r99_col[None, :]) / $(65536)) % $(8)) * $(131072)) + (((((r99_group % $(8)) * $(65536)) + (r99_lane[:, None] * $(128)) + r99_col[None, :]) % $(128)) * $(512)) + (((((r99_group % $(8)) * $(65536)) + (r99_lane[:, None] * $(128)) + r99_col[None, :]) / $(128)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r99_value = tl.dot(r99_left, r99_right)
      r99_value = r99_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_5_ptr) + ((r99_group * $(4096)) + (r99_row[:, None] * $(128)) + r99_col[None, :]), r99_value)
    }
    r100_row = tl.arange($(32))
    r100_col = tl.arange($(512))
    r100_lane = tl.arange($(128))
    for r100_group in range($(8)) {
      r100_left = tl.load($(candidate_vmm_1_ptr) + (((((((r100_group % $(8)) * $(4096)) + (r100_row[:, None] * $(128)) + r100_lane[None, :]) / $(128)) % $(32)) * $(1536)) + ((((((r100_group % $(8)) * $(4096)) + (r100_row[:, None] * $(128)) + r100_lane[None, :]) / $(4096)) % $(8)) * $(192)) + ((((r100_group % $(8)) * $(4096)) + (r100_row[:, None] * $(128)) + r100_lane[None, :]) % $(128)) + $(98304)), dtype=tl.bfloat16)
      r100_right = tl.load($(candidate_vto_copy_5_ptr) + (((((((r100_group % $(8)) * $(65536)) + (r100_lane[:, None] * $(512)) + r100_col[None, :]) / $(65536)) % $(8)) * $(131072)) + ((((((r100_group % $(8)) * $(65536)) + (r100_lane[:, None] * $(512)) + r100_col[None, :]) / $(512)) % $(128)) * $(512)) + ((((r100_group % $(8)) * $(65536)) + (r100_lane[:, None] * $(512)) + r100_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r100_value = tl.dot(r100_left, r100_right)
      r100_value = r100_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_6_ptr) + ((r100_group * $(16384)) + (r100_row[:, None] * $(512)) + r100_col[None, :]), r100_value)
    }
    r101_z = tl.arange($(16384))
    r101_value = tl.load($(candidate_vcat_5_ptr) + ((((r101_z / $(2048)) % $(8)) * $(18432)) + (((r101_z / $(64)) % $(32)) * $(64)) + (r101_z % $(64)) + $(4096)), dtype=tl.bfloat16)
    r101_value = r101_value.to(tl.float32)
    r101_value = tl.broadcast_to(r101_value, [$(16384)])
    tl.store($(candidate_vto_copy_32_ptr) + r101_z, r101_value)
    r102_z = tl.arange($(6144))
    r102_value = tl.load($(candidate_vcat_7_ptr) + ((((r102_z / $(64)) % $(96)) * $(576)) + (r102_z % $(64)) + $(512)), dtype=tl.bfloat16)
    r102_value = r102_value.to(tl.float32)
    r102_value = tl.broadcast_to(r102_value, [$(6144)])
    tl.store($(candidate_vto_copy_33_ptr) + r102_z, r102_value)
    r103_row = tl.arange($(32))
    r103_col = tl.arange($(96))
    r103_lane = tl.arange($(64))
    for r103_group in range($(8)) {
      r103_left = tl.load($(candidate_vto_copy_32_ptr) + (((((((r103_group % $(8)) * $(2048)) + (r103_row[:, None] * $(64)) + r103_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r103_group % $(8)) * $(2048)) + (r103_row[:, None] * $(64)) + r103_lane[None, :]) / $(64)) % $(32)) * $(64)) + ((((r103_group % $(8)) * $(2048)) + (r103_row[:, None] * $(64)) + r103_lane[None, :]) % $(64))), dtype=tl.float32)
      r103_right = tl.load($(candidate_vto_copy_33_ptr) + ((((((r103_group % $(8)) * $(6144)) + (r103_lane[:, None] * $(96)) + r103_col[None, :]) % $(96)) * $(64)) + (((((r103_group % $(8)) * $(6144)) + (r103_lane[:, None] * $(96)) + r103_col[None, :]) / $(96)) % $(64))), dtype=tl.float32)
      r103_value = tl.dot(r103_left, r103_right)
      r103_value = r103_value.to(tl.float32)
      tl.store($(candidate_vbmm_7_ptr) + ((r103_group * $(3072)) + (r103_row[:, None] * $(96)) + r103_col[None, :]), r103_value)
    }
    r104_z = tl.arange($(131072))
    r104_value = tl.load($(candidate_vbmm_6_ptr) + ((((r104_z / $(16384)) % $(8)) * $(16384)) + (((r104_z / $(512)) % $(32)) * $(512)) + (r104_z % $(512))), dtype=tl.bfloat16)
    r104_value = r104_value.to(tl.float32)
    r104_value = tl.broadcast_to(r104_value, [$(131072)])
    tl.store($(candidate_vto_copy_34_ptr) + r104_z, r104_value)
    r105_z = tl.arange($(49152))
    r105_value = tl.load($(candidate_vcat_7_ptr) + ((((r105_z / $(512)) % $(96)) * $(576)) + (r105_z % $(512))), dtype=tl.bfloat16)
    r105_value = r105_value.to(tl.float32)
    r105_value = tl.broadcast_to(r105_value, [$(49152)])
    tl.store($(candidate_vto_copy_35_ptr) + r105_z, r105_value)
    r106_row = tl.arange($(256))
    r106_col = tl.arange($(96))
    r106_lane = tl.arange($(512))
    r106_left = tl.load($(candidate_vto_copy_34_ptr) + ((((((r106_row[:, None] * $(512)) + r106_lane[None, :]) / $(16384)) % $(8)) * $(16384)) + (((((r106_row[:, None] * $(512)) + r106_lane[None, :]) / $(512)) % $(32)) * $(512)) + (((r106_row[:, None] * $(512)) + r106_lane[None, :]) % $(512))), dtype=tl.float32)
    r106_right = tl.load($(candidate_vto_copy_35_ptr) + (((((r106_lane[:, None] * $(96)) + r106_col[None, :]) % $(96)) * $(512)) + ((((r106_lane[:, None] * $(96)) + r106_col[None, :]) / $(96)) % $(512))), dtype=tl.float32)
    r106_value = tl.dot(r106_left, r106_right)
    r106_value = r106_value.to(tl.float32)
    tl.store($(candidate_vmm_7_ptr) + ((r106_row[:, None] * $(96)) + r106_col[None, :]), r106_value)
    r107_z = tl.arange($(24576))
    r107_value = (tl.load($(candidate_vbmm_7_ptr) + ((((r107_z / $(3072)) % $(8)) * $(3072)) + (((r107_z / $(96)) % $(32)) * $(96)) + (r107_z % $(96))), dtype=tl.float32) + tl.load($(candidate_vmm_7_ptr) + ((((r107_z / $(3072)) % $(8)) * $(3072)) + (((r107_z / $(96)) % $(32)) * $(96)) + (r107_z % $(96))), dtype=tl.float32))
    r107_value = r107_value.to(tl.float32)
    r107_value = tl.broadcast_to(r107_value, [$(24576)])
    tl.store($(candidate_vadd_10_ptr) + r107_z, r107_value)
    r108_z = tl.arange($(24576))
    r108_value = (tl.load($(candidate_vadd_10_ptr) + ((((r108_z / $(3072)) % $(8)) * $(3072)) + (((r108_z / $(96)) % $(32)) * $(96)) + (r108_z % $(96))), dtype=tl.float32) * 0.07216878364870322)
    r108_value = r108_value.to(tl.float32)
    r108_value = tl.broadcast_to(r108_value, [$(24576)])
    tl.store($(candidate_vmul_23_ptr) + r108_z, r108_value)
    r109_z = tl.arange($(32))
    r109_value = (tl.cast(r109_z, tl.int64) + tl.full([], $(96), dtype=tl.int64))
    r109_value = tl.broadcast_to(r109_value, [$(32)])
    tl.store($(candidate_vadd_11_ptr) + r109_z, r109_value)
    r110_z = tl.arange($(32))
    r110_value = (tl.load($(candidate_vadd_11_ptr) + r110_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r110_value = tl.broadcast_to(r110_value, [$(32)])
    tl.store($(candidate_vsub_17_ptr) + r110_z, r110_value)
    r111_z = tl.arange($(3072))
    r111_value = (tl.cast((r111_z % $(96)), tl.int64) > tl.load($(candidate_vsub_17_ptr) + ((r111_z / $(96)) % $(32)), dtype=tl.int64))
    r111_value = tl.broadcast_to(r111_value, [$(3072)])
    tl.store($(candidate_vgt_2_ptr) + r111_z, r111_value)
    r112_z = tl.arange($(24576))
    r112_value = tl.where(tl.load($(candidate_vgt_2_ptr) + ((((r112_z / $(96)) % $(32)) * $(96)) + (r112_z % $(96))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_23_ptr) + ((((r112_z / $(3072)) % $(8)) * $(3072)) + (((r112_z / $(96)) % $(32)) * $(96)) + (r112_z % $(96))), dtype=tl.float32))
    r112_value = r112_value.to(tl.float32)
    r112_value = tl.broadcast_to(r112_value, [$(24576)])
    tl.store($(candidate_vmasked_fill_2_ptr) + r112_z, r112_value)
    r113_z = tl.arange($(24576))
    r113_lane = tl.arange($(96))
    r113_row_values = tl.load($(candidate_vmasked_fill_2_ptr) + ((((r113_z[:, None] / $(3072)) % $(8)) * $(3072)) + (((r113_z[:, None] / $(96)) % $(32)) * $(96)) + (r113_lane[None, :] % $(96))), dtype=tl.float32)
    r113_maximum = tl.max(r113_row_values, axis=1, keep_dims=true)
    r113_weights = tl.exp(r113_row_values - r113_maximum)
    r113_total = tl.sum(r113_weights, axis=1)
    r113_maximum = tl.max(r113_row_values, axis=1)
    r113_value = tl.exp(tl.load($(candidate_vmasked_fill_2_ptr) + ((((r113_z / $(3072)) % $(8)) * $(3072)) + (((r113_z / $(96)) % $(32)) * $(96)) + (r113_z % $(96))), dtype=tl.float32) - r113_maximum) / r113_total
    r113_value = r113_value.to(tl.float32)
    r113_value = tl.broadcast_to(r113_value, [$(24576)])
    tl.store($(candidate_vsoftmax_2_ptr) + r113_z, r113_value)
    r114_z = tl.arange($(24576))
    r114_value = tl.load($(candidate_vsoftmax_2_ptr) + ((((r114_z / $(3072)) % $(8)) * $(3072)) + (((r114_z / $(96)) % $(32)) * $(96)) + (r114_z % $(96))), dtype=tl.float32)
    r114_value = r114_value.to(tl.bfloat16)
    r114_value = tl.broadcast_to(r114_value, [$(24576)])
    tl.store($(candidate_vto_copy_36_ptr) + r114_z, r114_value)
    r115_row = tl.arange($(256))
    r115_col = tl.arange($(512))
    r115_lane = tl.arange($(96))
    r115_left = tl.load($(candidate_vto_copy_36_ptr) + ((((((r115_row[:, None] * $(96)) + r115_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + (((((r115_row[:, None] * $(96)) + r115_lane[None, :]) / $(96)) % $(32)) * $(96)) + (((r115_row[:, None] * $(96)) + r115_lane[None, :]) % $(96))), dtype=tl.bfloat16)
    r115_right = tl.load($(candidate_vcat_7_ptr) + ((((((r115_lane[:, None] * $(512)) + r115_col[None, :]) / $(512)) % $(96)) * $(576)) + (((r115_lane[:, None] * $(512)) + r115_col[None, :]) % $(512))), dtype=tl.bfloat16)
    r115_value = tl.dot(r115_left, r115_right)
    r115_value = r115_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_8_ptr) + ((r115_row[:, None] * $(512)) + r115_col[None, :]), r115_value)
    r116_z = tl.arange($(131072))
    r116_value = tl.load($(candidate_vmm_8_ptr) + (((((((r116_z / $(512)) % $(8)) * $(32)) + (((r116_z / $(4096)) % $(32)) - $(0))) % $(256)) * $(512)) + (r116_z % $(512))), dtype=tl.bfloat16)
    r116_value = r116_value.to(tl.bfloat16)
    r116_value = tl.broadcast_to(r116_value, [$(131072)])
    tl.store($(candidate_vcat_10_ptr) + r116_z, r116_value)
    r117_row = tl.arange($(32))
    r117_col = tl.arange($(128))
    r117_lane = tl.arange($(512))
    for r117_group in range($(8)) {
      r117_left = tl.load($(candidate_vcat_10_ptr) + (((((((r117_group % $(8)) * $(16384)) + (r117_row[:, None] * $(512)) + r117_lane[None, :]) / $(512)) % $(32)) * $(4096)) + ((((((r117_group % $(8)) * $(16384)) + (r117_row[:, None] * $(512)) + r117_lane[None, :]) / $(16384)) % $(8)) * $(512)) + ((((r117_group % $(8)) * $(16384)) + (r117_row[:, None] * $(512)) + r117_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r117_right = tl.load($(candidate_vto_copy_5_ptr) + (((((((r117_group % $(8)) * $(65536)) + (r117_lane[:, None] * $(128)) + r117_col[None, :]) / $(65536)) % $(8)) * $(131072)) + (((((r117_group % $(8)) * $(65536)) + (r117_lane[:, None] * $(128)) + r117_col[None, :]) % $(128)) * $(512)) + (((((r117_group % $(8)) * $(65536)) + (r117_lane[:, None] * $(128)) + r117_col[None, :]) / $(128)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r117_value = tl.dot(r117_left, r117_right)
      r117_value = r117_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_8_ptr) + ((r117_group * $(4096)) + (r117_row[:, None] * $(128)) + r117_col[None, :]), r117_value)
    }
    r118_row = tl.arange($(32))
    r118_col = tl.arange($(512))
    r118_lane = tl.arange($(128))
    for r118_group in range($(8)) {
      r118_left = tl.load($(candidate_vmm_1_ptr) + (((((((r118_group % $(8)) * $(4096)) + (r118_row[:, None] * $(128)) + r118_lane[None, :]) / $(128)) % $(32)) * $(1536)) + ((((((r118_group % $(8)) * $(4096)) + (r118_row[:, None] * $(128)) + r118_lane[None, :]) / $(4096)) % $(8)) * $(192)) + ((((r118_group % $(8)) * $(4096)) + (r118_row[:, None] * $(128)) + r118_lane[None, :]) % $(128)) + $(147456)), dtype=tl.bfloat16)
      r118_right = tl.load($(candidate_vto_copy_5_ptr) + (((((((r118_group % $(8)) * $(65536)) + (r118_lane[:, None] * $(512)) + r118_col[None, :]) / $(65536)) % $(8)) * $(131072)) + ((((((r118_group % $(8)) * $(65536)) + (r118_lane[:, None] * $(512)) + r118_col[None, :]) / $(512)) % $(128)) * $(512)) + ((((r118_group % $(8)) * $(65536)) + (r118_lane[:, None] * $(512)) + r118_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r118_value = tl.dot(r118_left, r118_right)
      r118_value = r118_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_9_ptr) + ((r118_group * $(16384)) + (r118_row[:, None] * $(512)) + r118_col[None, :]), r118_value)
    }
    r119_z = tl.arange($(16384))
    r119_value = tl.load($(candidate_vcat_5_ptr) + ((((r119_z / $(2048)) % $(8)) * $(18432)) + (((r119_z / $(64)) % $(32)) * $(64)) + (r119_z % $(64)) + $(6144)), dtype=tl.bfloat16)
    r119_value = r119_value.to(tl.float32)
    r119_value = tl.broadcast_to(r119_value, [$(16384)])
    tl.store($(candidate_vto_copy_37_ptr) + r119_z, r119_value)
    r120_z = tl.arange($(2048))
    r120_value = tl.load($(candidate_vcat_7_ptr) + ((((r120_z / $(64)) % $(32)) * $(576)) + (r120_z % $(64)) + $(55808)), dtype=tl.bfloat16)
    r120_value = r120_value.to(tl.float32)
    r120_value = tl.broadcast_to(r120_value, [$(2048)])
    tl.store($(candidate_vto_copy_38_ptr) + r120_z, r120_value)
    r121_row = tl.arange($(32))
    r121_col = tl.arange($(32))
    r121_lane = tl.arange($(64))
    for r121_group in range($(8)) {
      r121_left = tl.load($(candidate_vto_copy_37_ptr) + (((((((r121_group % $(8)) * $(2048)) + (r121_row[:, None] * $(64)) + r121_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r121_group % $(8)) * $(2048)) + (r121_row[:, None] * $(64)) + r121_lane[None, :]) / $(64)) % $(32)) * $(64)) + ((((r121_group % $(8)) * $(2048)) + (r121_row[:, None] * $(64)) + r121_lane[None, :]) % $(64))), dtype=tl.float32)
      r121_right = tl.load($(candidate_vto_copy_38_ptr) + ((((((r121_group % $(8)) * $(2048)) + (r121_lane[:, None] * $(32)) + r121_col[None, :]) % $(32)) * $(64)) + (((((r121_group % $(8)) * $(2048)) + (r121_lane[:, None] * $(32)) + r121_col[None, :]) / $(32)) % $(64))), dtype=tl.float32)
      r121_value = tl.dot(r121_left, r121_right)
      r121_value = r121_value.to(tl.float32)
      tl.store($(candidate_vbmm_10_ptr) + ((r121_group * $(1024)) + (r121_row[:, None] * $(32)) + r121_col[None, :]), r121_value)
    }
    r122_z = tl.arange($(131072))
    r122_value = tl.load($(candidate_vbmm_9_ptr) + ((((r122_z / $(16384)) % $(8)) * $(16384)) + (((r122_z / $(512)) % $(32)) * $(512)) + (r122_z % $(512))), dtype=tl.bfloat16)
    r122_value = r122_value.to(tl.float32)
    r122_value = tl.broadcast_to(r122_value, [$(131072)])
    tl.store($(candidate_vto_copy_39_ptr) + r122_z, r122_value)
    r123_z = tl.arange($(16384))
    r123_value = tl.load($(candidate_vcat_7_ptr) + ((((r123_z / $(512)) % $(32)) * $(576)) + (r123_z % $(512)) + $(55296)), dtype=tl.bfloat16)
    r123_value = r123_value.to(tl.float32)
    r123_value = tl.broadcast_to(r123_value, [$(16384)])
    tl.store($(candidate_vto_copy_40_ptr) + r123_z, r123_value)
    r124_row = tl.arange($(256))
    r124_col = tl.arange($(32))
    r124_lane = tl.arange($(512))
    r124_left = tl.load($(candidate_vto_copy_39_ptr) + ((((((r124_row[:, None] * $(512)) + r124_lane[None, :]) / $(16384)) % $(8)) * $(16384)) + (((((r124_row[:, None] * $(512)) + r124_lane[None, :]) / $(512)) % $(32)) * $(512)) + (((r124_row[:, None] * $(512)) + r124_lane[None, :]) % $(512))), dtype=tl.float32)
    r124_right = tl.load($(candidate_vto_copy_40_ptr) + (((((r124_lane[:, None] * $(32)) + r124_col[None, :]) % $(32)) * $(512)) + ((((r124_lane[:, None] * $(32)) + r124_col[None, :]) / $(32)) % $(512))), dtype=tl.float32)
    r124_value = tl.dot(r124_left, r124_right)
    r124_value = r124_value.to(tl.float32)
    tl.store($(candidate_vmm_9_ptr) + ((r124_row[:, None] * $(32)) + r124_col[None, :]), r124_value)
    r125_z = tl.arange($(8192))
    r125_value = (tl.load($(candidate_vbmm_10_ptr) + ((((r125_z / $(1024)) % $(8)) * $(1024)) + (((r125_z / $(32)) % $(32)) * $(32)) + (r125_z % $(32))), dtype=tl.float32) + tl.load($(candidate_vmm_9_ptr) + ((((r125_z / $(1024)) % $(8)) * $(1024)) + (((r125_z / $(32)) % $(32)) * $(32)) + (r125_z % $(32))), dtype=tl.float32))
    r125_value = r125_value.to(tl.float32)
    r125_value = tl.broadcast_to(r125_value, [$(8192)])
    tl.store($(candidate_vadd_12_ptr) + r125_z, r125_value)
    r126_z = tl.arange($(8192))
    r126_value = (tl.load($(candidate_vadd_12_ptr) + ((((r126_z / $(1024)) % $(8)) * $(1024)) + (((r126_z / $(32)) % $(32)) * $(32)) + (r126_z % $(32))), dtype=tl.float32) * 0.07216878364870322)
    r126_value = r126_value.to(tl.float32)
    r126_value = tl.broadcast_to(r126_value, [$(8192)])
    tl.store($(candidate_vmul_24_ptr) + r126_z, r126_value)
    r127_z = tl.arange($(32))
    r127_value = (tl.cast(r127_z, tl.int64) + tl.full([], $(32), dtype=tl.int64))
    r127_value = tl.broadcast_to(r127_value, [$(32)])
    tl.store($(candidate_vadd_13_ptr) + r127_z, r127_value)
    r128_z = tl.arange($(32))
    r128_value = (tl.load($(candidate_vadd_13_ptr) + r128_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r128_value = tl.broadcast_to(r128_value, [$(32)])
    tl.store($(candidate_vsub_18_ptr) + r128_z, r128_value)
    r129_z = tl.arange($(1024))
    r129_value = (tl.cast((r129_z % $(32)), tl.int64) > tl.load($(candidate_vsub_18_ptr) + ((r129_z / $(32)) % $(32)), dtype=tl.int64))
    r129_value = tl.broadcast_to(r129_value, [$(1024)])
    tl.store($(candidate_vgt_3_ptr) + r129_z, r129_value)
    r130_z = tl.arange($(8192))
    r130_value = tl.where(tl.load($(candidate_vgt_3_ptr) + ((((r130_z / $(32)) % $(32)) * $(32)) + (r130_z % $(32))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_24_ptr) + ((((r130_z / $(1024)) % $(8)) * $(1024)) + (((r130_z / $(32)) % $(32)) * $(32)) + (r130_z % $(32))), dtype=tl.float32))
    r130_value = r130_value.to(tl.float32)
    r130_value = tl.broadcast_to(r130_value, [$(8192)])
    tl.store($(candidate_vmasked_fill_3_ptr) + r130_z, r130_value)
    r131_z = tl.arange($(8192))
    r131_lane = tl.arange($(32))
    r131_row_values = tl.load($(candidate_vmasked_fill_3_ptr) + ((((r131_z[:, None] / $(1024)) % $(8)) * $(1024)) + (((r131_z[:, None] / $(32)) % $(32)) * $(32)) + (r131_lane[None, :] % $(32))), dtype=tl.float32)
    r131_maximum = tl.max(r131_row_values, axis=1, keep_dims=true)
    r131_weights = tl.exp(r131_row_values - r131_maximum)
    r131_total = tl.sum(r131_weights, axis=1)
    r131_maximum = tl.max(r131_row_values, axis=1)
    r131_value = tl.exp(tl.load($(candidate_vmasked_fill_3_ptr) + ((((r131_z / $(1024)) % $(8)) * $(1024)) + (((r131_z / $(32)) % $(32)) * $(32)) + (r131_z % $(32))), dtype=tl.float32) - r131_maximum) / r131_total
    r131_value = r131_value.to(tl.float32)
    r131_value = tl.broadcast_to(r131_value, [$(8192)])
    tl.store($(candidate_vsoftmax_3_ptr) + r131_z, r131_value)
    r132_z = tl.arange($(8192))
    r132_value = tl.load($(candidate_vsoftmax_3_ptr) + ((((r132_z / $(1024)) % $(8)) * $(1024)) + (((r132_z / $(32)) % $(32)) * $(32)) + (r132_z % $(32))), dtype=tl.float32)
    r132_value = r132_value.to(tl.bfloat16)
    r132_value = tl.broadcast_to(r132_value, [$(8192)])
    tl.store($(candidate_vto_copy_41_ptr) + r132_z, r132_value)
    r133_row = tl.arange($(256))
    r133_col = tl.arange($(512))
    r133_lane = tl.arange($(32))
    r133_left = tl.load($(candidate_vto_copy_41_ptr) + ((((((r133_row[:, None] * $(32)) + r133_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + (((((r133_row[:, None] * $(32)) + r133_lane[None, :]) / $(32)) % $(32)) * $(32)) + (((r133_row[:, None] * $(32)) + r133_lane[None, :]) % $(32))), dtype=tl.bfloat16)
    r133_right = tl.load($(candidate_vcat_7_ptr) + ((((((r133_lane[:, None] * $(512)) + r133_col[None, :]) / $(512)) % $(32)) * $(576)) + (((r133_lane[:, None] * $(512)) + r133_col[None, :]) % $(512)) + $(55296)), dtype=tl.bfloat16)
    r133_value = tl.dot(r133_left, r133_right)
    r133_value = r133_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_10_ptr) + ((r133_row[:, None] * $(512)) + r133_col[None, :]), r133_value)
    r134_z = tl.arange($(131072))
    r134_value = tl.load($(candidate_vmm_10_ptr) + (((((((r134_z / $(512)) % $(8)) * $(32)) + (((r134_z / $(4096)) % $(32)) - $(0))) % $(256)) * $(512)) + (r134_z % $(512))), dtype=tl.bfloat16)
    r134_value = r134_value.to(tl.bfloat16)
    r134_value = tl.broadcast_to(r134_value, [$(131072)])
    tl.store($(candidate_vcat_11_ptr) + r134_z, r134_value)
    r135_row = tl.arange($(32))
    r135_col = tl.arange($(128))
    r135_lane = tl.arange($(512))
    for r135_group in range($(8)) {
      r135_left = tl.load($(candidate_vcat_11_ptr) + (((((((r135_group % $(8)) * $(16384)) + (r135_row[:, None] * $(512)) + r135_lane[None, :]) / $(512)) % $(32)) * $(4096)) + ((((((r135_group % $(8)) * $(16384)) + (r135_row[:, None] * $(512)) + r135_lane[None, :]) / $(16384)) % $(8)) * $(512)) + ((((r135_group % $(8)) * $(16384)) + (r135_row[:, None] * $(512)) + r135_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r135_right = tl.load($(candidate_vto_copy_5_ptr) + (((((((r135_group % $(8)) * $(65536)) + (r135_lane[:, None] * $(128)) + r135_col[None, :]) / $(65536)) % $(8)) * $(131072)) + (((((r135_group % $(8)) * $(65536)) + (r135_lane[:, None] * $(128)) + r135_col[None, :]) % $(128)) * $(512)) + (((((r135_group % $(8)) * $(65536)) + (r135_lane[:, None] * $(128)) + r135_col[None, :]) / $(128)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r135_value = tl.dot(r135_left, r135_right)
      r135_value = r135_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_11_ptr) + ((r135_group * $(4096)) + (r135_row[:, None] * $(128)) + r135_col[None, :]), r135_value)
    }
    r136_row = tl.arange($(32))
    r136_col = tl.arange($(512))
    r136_lane = tl.arange($(128))
    for r136_group in range($(8)) {
      r136_left = tl.load($(candidate_vmm_1_ptr) + (((((((r136_group % $(8)) * $(4096)) + (r136_row[:, None] * $(128)) + r136_lane[None, :]) / $(128)) % $(32)) * $(1536)) + ((((((r136_group % $(8)) * $(4096)) + (r136_row[:, None] * $(128)) + r136_lane[None, :]) / $(4096)) % $(8)) * $(192)) + ((((r136_group % $(8)) * $(4096)) + (r136_row[:, None] * $(128)) + r136_lane[None, :]) % $(128)) + $(196608)), dtype=tl.bfloat16)
      r136_right = tl.load($(candidate_vto_copy_5_ptr) + (((((((r136_group % $(8)) * $(65536)) + (r136_lane[:, None] * $(512)) + r136_col[None, :]) / $(65536)) % $(8)) * $(131072)) + ((((((r136_group % $(8)) * $(65536)) + (r136_lane[:, None] * $(512)) + r136_col[None, :]) / $(512)) % $(128)) * $(512)) + ((((r136_group % $(8)) * $(65536)) + (r136_lane[:, None] * $(512)) + r136_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r136_value = tl.dot(r136_left, r136_right)
      r136_value = r136_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_12_ptr) + ((r136_group * $(16384)) + (r136_row[:, None] * $(512)) + r136_col[None, :]), r136_value)
    }
    r137_z = tl.arange($(16384))
    r137_value = tl.load($(candidate_vcat_5_ptr) + ((((r137_z / $(2048)) % $(8)) * $(18432)) + (((r137_z / $(64)) % $(32)) * $(64)) + (r137_z % $(64)) + $(8192)), dtype=tl.bfloat16)
    r137_value = r137_value.to(tl.float32)
    r137_value = tl.broadcast_to(r137_value, [$(16384)])
    tl.store($(candidate_vto_copy_42_ptr) + r137_z, r137_value)
    r138_z = tl.arange($(4096))
    r138_value = tl.load($(candidate_vcat_7_ptr) + ((((r138_z / $(64)) % $(64)) * $(576)) + (r138_z % $(64)) + $(55808)), dtype=tl.bfloat16)
    r138_value = r138_value.to(tl.float32)
    r138_value = tl.broadcast_to(r138_value, [$(4096)])
    tl.store($(candidate_vto_copy_43_ptr) + r138_z, r138_value)
    r139_row = tl.arange($(32))
    r139_col = tl.arange($(64))
    r139_lane = tl.arange($(64))
    for r139_group in range($(8)) {
      r139_left = tl.load($(candidate_vto_copy_42_ptr) + (((((((r139_group % $(8)) * $(2048)) + (r139_row[:, None] * $(64)) + r139_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r139_group % $(8)) * $(2048)) + (r139_row[:, None] * $(64)) + r139_lane[None, :]) / $(64)) % $(32)) * $(64)) + ((((r139_group % $(8)) * $(2048)) + (r139_row[:, None] * $(64)) + r139_lane[None, :]) % $(64))), dtype=tl.float32)
      r139_right = tl.load($(candidate_vto_copy_43_ptr) + ((((((r139_group % $(8)) * $(4096)) + (r139_lane[:, None] * $(64)) + r139_col[None, :]) % $(64)) * $(64)) + (((((r139_group % $(8)) * $(4096)) + (r139_lane[:, None] * $(64)) + r139_col[None, :]) / $(64)) % $(64))), dtype=tl.float32)
      r139_value = tl.dot(r139_left, r139_right)
      r139_value = r139_value.to(tl.float32)
      tl.store($(candidate_vbmm_13_ptr) + ((r139_group * $(2048)) + (r139_row[:, None] * $(64)) + r139_col[None, :]), r139_value)
    }
    r140_z = tl.arange($(131072))
    r140_value = tl.load($(candidate_vbmm_12_ptr) + ((((r140_z / $(16384)) % $(8)) * $(16384)) + (((r140_z / $(512)) % $(32)) * $(512)) + (r140_z % $(512))), dtype=tl.bfloat16)
    r140_value = r140_value.to(tl.float32)
    r140_value = tl.broadcast_to(r140_value, [$(131072)])
    tl.store($(candidate_vto_copy_44_ptr) + r140_z, r140_value)
    r141_z = tl.arange($(32768))
    r141_value = tl.load($(candidate_vcat_7_ptr) + ((((r141_z / $(512)) % $(64)) * $(576)) + (r141_z % $(512)) + $(55296)), dtype=tl.bfloat16)
    r141_value = r141_value.to(tl.float32)
    r141_value = tl.broadcast_to(r141_value, [$(32768)])
    tl.store($(candidate_vto_copy_45_ptr) + r141_z, r141_value)
    r142_row = tl.arange($(256))
    r142_col = tl.arange($(64))
    r142_lane = tl.arange($(512))
    r142_left = tl.load($(candidate_vto_copy_44_ptr) + ((((((r142_row[:, None] * $(512)) + r142_lane[None, :]) / $(16384)) % $(8)) * $(16384)) + (((((r142_row[:, None] * $(512)) + r142_lane[None, :]) / $(512)) % $(32)) * $(512)) + (((r142_row[:, None] * $(512)) + r142_lane[None, :]) % $(512))), dtype=tl.float32)
    r142_right = tl.load($(candidate_vto_copy_45_ptr) + (((((r142_lane[:, None] * $(64)) + r142_col[None, :]) % $(64)) * $(512)) + ((((r142_lane[:, None] * $(64)) + r142_col[None, :]) / $(64)) % $(512))), dtype=tl.float32)
    r142_value = tl.dot(r142_left, r142_right)
    r142_value = r142_value.to(tl.float32)
    tl.store($(candidate_vmm_11_ptr) + ((r142_row[:, None] * $(64)) + r142_col[None, :]), r142_value)
    r143_z = tl.arange($(16384))
    r143_value = (tl.load($(candidate_vbmm_13_ptr) + ((((r143_z / $(2048)) % $(8)) * $(2048)) + (((r143_z / $(64)) % $(32)) * $(64)) + (r143_z % $(64))), dtype=tl.float32) + tl.load($(candidate_vmm_11_ptr) + ((((r143_z / $(2048)) % $(8)) * $(2048)) + (((r143_z / $(64)) % $(32)) * $(64)) + (r143_z % $(64))), dtype=tl.float32))
    r143_value = r143_value.to(tl.float32)
    r143_value = tl.broadcast_to(r143_value, [$(16384)])
    tl.store($(candidate_vadd_14_ptr) + r143_z, r143_value)
    r144_z = tl.arange($(16384))
    r144_value = (tl.load($(candidate_vadd_14_ptr) + ((((r144_z / $(2048)) % $(8)) * $(2048)) + (((r144_z / $(64)) % $(32)) * $(64)) + (r144_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r144_value = r144_value.to(tl.float32)
    r144_value = tl.broadcast_to(r144_value, [$(16384)])
    tl.store($(candidate_vmul_25_ptr) + r144_z, r144_value)
    r145_z = tl.arange($(32))
    r145_value = (tl.cast(r145_z, tl.int64) + tl.full([], $(64), dtype=tl.int64))
    r145_value = tl.broadcast_to(r145_value, [$(32)])
    tl.store($(candidate_vadd_15_ptr) + r145_z, r145_value)
    r146_z = tl.arange($(32))
    r146_value = (tl.load($(candidate_vadd_15_ptr) + r146_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r146_value = tl.broadcast_to(r146_value, [$(32)])
    tl.store($(candidate_vsub_19_ptr) + r146_z, r146_value)
    r147_z = tl.arange($(2048))
    r147_value = (tl.cast((r147_z % $(64)), tl.int64) > tl.load($(candidate_vsub_19_ptr) + ((r147_z / $(64)) % $(32)), dtype=tl.int64))
    r147_value = tl.broadcast_to(r147_value, [$(2048)])
    tl.store($(candidate_vgt_4_ptr) + r147_z, r147_value)
    r148_z = tl.arange($(16384))
    r148_value = tl.where(tl.load($(candidate_vgt_4_ptr) + ((((r148_z / $(64)) % $(32)) * $(64)) + (r148_z % $(64))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_25_ptr) + ((((r148_z / $(2048)) % $(8)) * $(2048)) + (((r148_z / $(64)) % $(32)) * $(64)) + (r148_z % $(64))), dtype=tl.float32))
    r148_value = r148_value.to(tl.float32)
    r148_value = tl.broadcast_to(r148_value, [$(16384)])
    tl.store($(candidate_vmasked_fill_4_ptr) + r148_z, r148_value)
    r149_z = tl.arange($(16384))
    r149_lane = tl.arange($(64))
    r149_row_values = tl.load($(candidate_vmasked_fill_4_ptr) + ((((r149_z[:, None] / $(2048)) % $(8)) * $(2048)) + (((r149_z[:, None] / $(64)) % $(32)) * $(64)) + (r149_lane[None, :] % $(64))), dtype=tl.float32)
    r149_maximum = tl.max(r149_row_values, axis=1, keep_dims=true)
    r149_weights = tl.exp(r149_row_values - r149_maximum)
    r149_total = tl.sum(r149_weights, axis=1)
    r149_maximum = tl.max(r149_row_values, axis=1)
    r149_value = tl.exp(tl.load($(candidate_vmasked_fill_4_ptr) + ((((r149_z / $(2048)) % $(8)) * $(2048)) + (((r149_z / $(64)) % $(32)) * $(64)) + (r149_z % $(64))), dtype=tl.float32) - r149_maximum) / r149_total
    r149_value = r149_value.to(tl.float32)
    r149_value = tl.broadcast_to(r149_value, [$(16384)])
    tl.store($(candidate_vsoftmax_4_ptr) + r149_z, r149_value)
    r150_z = tl.arange($(16384))
    r150_value = tl.load($(candidate_vsoftmax_4_ptr) + ((((r150_z / $(2048)) % $(8)) * $(2048)) + (((r150_z / $(64)) % $(32)) * $(64)) + (r150_z % $(64))), dtype=tl.float32)
    r150_value = r150_value.to(tl.bfloat16)
    r150_value = tl.broadcast_to(r150_value, [$(16384)])
    tl.store($(candidate_vto_copy_46_ptr) + r150_z, r150_value)
    r151_row = tl.arange($(256))
    r151_col = tl.arange($(512))
    r151_lane = tl.arange($(64))
    r151_left = tl.load($(candidate_vto_copy_46_ptr) + ((((((r151_row[:, None] * $(64)) + r151_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + (((((r151_row[:, None] * $(64)) + r151_lane[None, :]) / $(64)) % $(32)) * $(64)) + (((r151_row[:, None] * $(64)) + r151_lane[None, :]) % $(64))), dtype=tl.bfloat16)
    r151_right = tl.load($(candidate_vcat_7_ptr) + ((((((r151_lane[:, None] * $(512)) + r151_col[None, :]) / $(512)) % $(64)) * $(576)) + (((r151_lane[:, None] * $(512)) + r151_col[None, :]) % $(512)) + $(55296)), dtype=tl.bfloat16)
    r151_value = tl.dot(r151_left, r151_right)
    r151_value = r151_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_12_ptr) + ((r151_row[:, None] * $(512)) + r151_col[None, :]), r151_value)
    r152_z = tl.arange($(131072))
    r152_value = tl.load($(candidate_vmm_12_ptr) + (((((((r152_z / $(512)) % $(8)) * $(32)) + (((r152_z / $(4096)) % $(32)) - $(0))) % $(256)) * $(512)) + (r152_z % $(512))), dtype=tl.bfloat16)
    r152_value = r152_value.to(tl.bfloat16)
    r152_value = tl.broadcast_to(r152_value, [$(131072)])
    tl.store($(candidate_vcat_12_ptr) + r152_z, r152_value)
    r153_row = tl.arange($(32))
    r153_col = tl.arange($(128))
    r153_lane = tl.arange($(512))
    for r153_group in range($(8)) {
      r153_left = tl.load($(candidate_vcat_12_ptr) + (((((((r153_group % $(8)) * $(16384)) + (r153_row[:, None] * $(512)) + r153_lane[None, :]) / $(512)) % $(32)) * $(4096)) + ((((((r153_group % $(8)) * $(16384)) + (r153_row[:, None] * $(512)) + r153_lane[None, :]) / $(16384)) % $(8)) * $(512)) + ((((r153_group % $(8)) * $(16384)) + (r153_row[:, None] * $(512)) + r153_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r153_right = tl.load($(candidate_vto_copy_5_ptr) + (((((((r153_group % $(8)) * $(65536)) + (r153_lane[:, None] * $(128)) + r153_col[None, :]) / $(65536)) % $(8)) * $(131072)) + (((((r153_group % $(8)) * $(65536)) + (r153_lane[:, None] * $(128)) + r153_col[None, :]) % $(128)) * $(512)) + (((((r153_group % $(8)) * $(65536)) + (r153_lane[:, None] * $(128)) + r153_col[None, :]) / $(128)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r153_value = tl.dot(r153_left, r153_right)
      r153_value = r153_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_14_ptr) + ((r153_group * $(4096)) + (r153_row[:, None] * $(128)) + r153_col[None, :]), r153_value)
    }
    r154_row = tl.arange($(32))
    r154_col = tl.arange($(512))
    r154_lane = tl.arange($(128))
    for r154_group in range($(8)) {
      r154_left = tl.load($(candidate_vmm_1_ptr) + (((((((r154_group % $(8)) * $(4096)) + (r154_row[:, None] * $(128)) + r154_lane[None, :]) / $(128)) % $(32)) * $(1536)) + ((((((r154_group % $(8)) * $(4096)) + (r154_row[:, None] * $(128)) + r154_lane[None, :]) / $(4096)) % $(8)) * $(192)) + ((((r154_group % $(8)) * $(4096)) + (r154_row[:, None] * $(128)) + r154_lane[None, :]) % $(128)) + $(245760)), dtype=tl.bfloat16)
      r154_right = tl.load($(candidate_vto_copy_5_ptr) + (((((((r154_group % $(8)) * $(65536)) + (r154_lane[:, None] * $(512)) + r154_col[None, :]) / $(65536)) % $(8)) * $(131072)) + ((((((r154_group % $(8)) * $(65536)) + (r154_lane[:, None] * $(512)) + r154_col[None, :]) / $(512)) % $(128)) * $(512)) + ((((r154_group % $(8)) * $(65536)) + (r154_lane[:, None] * $(512)) + r154_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r154_value = tl.dot(r154_left, r154_right)
      r154_value = r154_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_15_ptr) + ((r154_group * $(16384)) + (r154_row[:, None] * $(512)) + r154_col[None, :]), r154_value)
    }
    r155_z = tl.arange($(16384))
    r155_value = tl.load($(candidate_vcat_5_ptr) + ((((r155_z / $(2048)) % $(8)) * $(18432)) + (((r155_z / $(64)) % $(32)) * $(64)) + (r155_z % $(64)) + $(10240)), dtype=tl.bfloat16)
    r155_value = r155_value.to(tl.float32)
    r155_value = tl.broadcast_to(r155_value, [$(16384)])
    tl.store($(candidate_vto_copy_47_ptr) + r155_z, r155_value)
    r156_z = tl.arange($(6144))
    r156_value = tl.load($(candidate_vcat_7_ptr) + ((((r156_z / $(64)) % $(96)) * $(576)) + (r156_z % $(64)) + $(55808)), dtype=tl.bfloat16)
    r156_value = r156_value.to(tl.float32)
    r156_value = tl.broadcast_to(r156_value, [$(6144)])
    tl.store($(candidate_vto_copy_48_ptr) + r156_z, r156_value)
    r157_row = tl.arange($(32))
    r157_col = tl.arange($(96))
    r157_lane = tl.arange($(64))
    for r157_group in range($(8)) {
      r157_left = tl.load($(candidate_vto_copy_47_ptr) + (((((((r157_group % $(8)) * $(2048)) + (r157_row[:, None] * $(64)) + r157_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r157_group % $(8)) * $(2048)) + (r157_row[:, None] * $(64)) + r157_lane[None, :]) / $(64)) % $(32)) * $(64)) + ((((r157_group % $(8)) * $(2048)) + (r157_row[:, None] * $(64)) + r157_lane[None, :]) % $(64))), dtype=tl.float32)
      r157_right = tl.load($(candidate_vto_copy_48_ptr) + ((((((r157_group % $(8)) * $(6144)) + (r157_lane[:, None] * $(96)) + r157_col[None, :]) % $(96)) * $(64)) + (((((r157_group % $(8)) * $(6144)) + (r157_lane[:, None] * $(96)) + r157_col[None, :]) / $(96)) % $(64))), dtype=tl.float32)
      r157_value = tl.dot(r157_left, r157_right)
      r157_value = r157_value.to(tl.float32)
      tl.store($(candidate_vbmm_16_ptr) + ((r157_group * $(3072)) + (r157_row[:, None] * $(96)) + r157_col[None, :]), r157_value)
    }
    r158_z = tl.arange($(131072))
    r158_value = tl.load($(candidate_vbmm_15_ptr) + ((((r158_z / $(16384)) % $(8)) * $(16384)) + (((r158_z / $(512)) % $(32)) * $(512)) + (r158_z % $(512))), dtype=tl.bfloat16)
    r158_value = r158_value.to(tl.float32)
    r158_value = tl.broadcast_to(r158_value, [$(131072)])
    tl.store($(candidate_vto_copy_49_ptr) + r158_z, r158_value)
    r159_z = tl.arange($(49152))
    r159_value = tl.load($(candidate_vcat_7_ptr) + ((((r159_z / $(512)) % $(96)) * $(576)) + (r159_z % $(512)) + $(55296)), dtype=tl.bfloat16)
    r159_value = r159_value.to(tl.float32)
    r159_value = tl.broadcast_to(r159_value, [$(49152)])
    tl.store($(candidate_vto_copy_50_ptr) + r159_z, r159_value)
    r160_row = tl.arange($(256))
    r160_col = tl.arange($(96))
    r160_lane = tl.arange($(512))
    r160_left = tl.load($(candidate_vto_copy_49_ptr) + ((((((r160_row[:, None] * $(512)) + r160_lane[None, :]) / $(16384)) % $(8)) * $(16384)) + (((((r160_row[:, None] * $(512)) + r160_lane[None, :]) / $(512)) % $(32)) * $(512)) + (((r160_row[:, None] * $(512)) + r160_lane[None, :]) % $(512))), dtype=tl.float32)
    r160_right = tl.load($(candidate_vto_copy_50_ptr) + (((((r160_lane[:, None] * $(96)) + r160_col[None, :]) % $(96)) * $(512)) + ((((r160_lane[:, None] * $(96)) + r160_col[None, :]) / $(96)) % $(512))), dtype=tl.float32)
    r160_value = tl.dot(r160_left, r160_right)
    r160_value = r160_value.to(tl.float32)
    tl.store($(candidate_vmm_13_ptr) + ((r160_row[:, None] * $(96)) + r160_col[None, :]), r160_value)
    r161_z = tl.arange($(24576))
    r161_value = (tl.load($(candidate_vbmm_16_ptr) + ((((r161_z / $(3072)) % $(8)) * $(3072)) + (((r161_z / $(96)) % $(32)) * $(96)) + (r161_z % $(96))), dtype=tl.float32) + tl.load($(candidate_vmm_13_ptr) + ((((r161_z / $(3072)) % $(8)) * $(3072)) + (((r161_z / $(96)) % $(32)) * $(96)) + (r161_z % $(96))), dtype=tl.float32))
    r161_value = r161_value.to(tl.float32)
    r161_value = tl.broadcast_to(r161_value, [$(24576)])
    tl.store($(candidate_vadd_16_ptr) + r161_z, r161_value)
    r162_z = tl.arange($(24576))
    r162_value = (tl.load($(candidate_vadd_16_ptr) + ((((r162_z / $(3072)) % $(8)) * $(3072)) + (((r162_z / $(96)) % $(32)) * $(96)) + (r162_z % $(96))), dtype=tl.float32) * 0.07216878364870322)
    r162_value = r162_value.to(tl.float32)
    r162_value = tl.broadcast_to(r162_value, [$(24576)])
    tl.store($(candidate_vmul_26_ptr) + r162_z, r162_value)
    r163_z = tl.arange($(32))
    r163_value = (tl.cast(r163_z, tl.int64) + tl.full([], $(96), dtype=tl.int64))
    r163_value = tl.broadcast_to(r163_value, [$(32)])
    tl.store($(candidate_vadd_17_ptr) + r163_z, r163_value)
    r164_z = tl.arange($(32))
    r164_value = (tl.load($(candidate_vadd_17_ptr) + r164_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r164_value = tl.broadcast_to(r164_value, [$(32)])
    tl.store($(candidate_vsub_20_ptr) + r164_z, r164_value)
    r165_z = tl.arange($(3072))
    r165_value = (tl.cast((r165_z % $(96)), tl.int64) > tl.load($(candidate_vsub_20_ptr) + ((r165_z / $(96)) % $(32)), dtype=tl.int64))
    r165_value = tl.broadcast_to(r165_value, [$(3072)])
    tl.store($(candidate_vgt_5_ptr) + r165_z, r165_value)
    r166_z = tl.arange($(24576))
    r166_value = tl.where(tl.load($(candidate_vgt_5_ptr) + ((((r166_z / $(96)) % $(32)) * $(96)) + (r166_z % $(96))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_26_ptr) + ((((r166_z / $(3072)) % $(8)) * $(3072)) + (((r166_z / $(96)) % $(32)) * $(96)) + (r166_z % $(96))), dtype=tl.float32))
    r166_value = r166_value.to(tl.float32)
    r166_value = tl.broadcast_to(r166_value, [$(24576)])
    tl.store($(candidate_vmasked_fill_5_ptr) + r166_z, r166_value)
    r167_z = tl.arange($(24576))
    r167_lane = tl.arange($(96))
    r167_row_values = tl.load($(candidate_vmasked_fill_5_ptr) + ((((r167_z[:, None] / $(3072)) % $(8)) * $(3072)) + (((r167_z[:, None] / $(96)) % $(32)) * $(96)) + (r167_lane[None, :] % $(96))), dtype=tl.float32)
    r167_maximum = tl.max(r167_row_values, axis=1, keep_dims=true)
    r167_weights = tl.exp(r167_row_values - r167_maximum)
    r167_total = tl.sum(r167_weights, axis=1)
    r167_maximum = tl.max(r167_row_values, axis=1)
    r167_value = tl.exp(tl.load($(candidate_vmasked_fill_5_ptr) + ((((r167_z / $(3072)) % $(8)) * $(3072)) + (((r167_z / $(96)) % $(32)) * $(96)) + (r167_z % $(96))), dtype=tl.float32) - r167_maximum) / r167_total
    r167_value = r167_value.to(tl.float32)
    r167_value = tl.broadcast_to(r167_value, [$(24576)])
    tl.store($(candidate_vsoftmax_5_ptr) + r167_z, r167_value)
    r168_z = tl.arange($(24576))
    r168_value = tl.load($(candidate_vsoftmax_5_ptr) + ((((r168_z / $(3072)) % $(8)) * $(3072)) + (((r168_z / $(96)) % $(32)) * $(96)) + (r168_z % $(96))), dtype=tl.float32)
    r168_value = r168_value.to(tl.bfloat16)
    r168_value = tl.broadcast_to(r168_value, [$(24576)])
    tl.store($(candidate_vto_copy_51_ptr) + r168_z, r168_value)
    r169_row = tl.arange($(256))
    r169_col = tl.arange($(512))
    r169_lane = tl.arange($(96))
    r169_left = tl.load($(candidate_vto_copy_51_ptr) + ((((((r169_row[:, None] * $(96)) + r169_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + (((((r169_row[:, None] * $(96)) + r169_lane[None, :]) / $(96)) % $(32)) * $(96)) + (((r169_row[:, None] * $(96)) + r169_lane[None, :]) % $(96))), dtype=tl.bfloat16)
    r169_right = tl.load($(candidate_vcat_7_ptr) + ((((((r169_lane[:, None] * $(512)) + r169_col[None, :]) / $(512)) % $(96)) * $(576)) + (((r169_lane[:, None] * $(512)) + r169_col[None, :]) % $(512)) + $(55296)), dtype=tl.bfloat16)
    r169_value = tl.dot(r169_left, r169_right)
    r169_value = r169_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_14_ptr) + ((r169_row[:, None] * $(512)) + r169_col[None, :]), r169_value)
    r170_z = tl.arange($(131072))
    r170_value = tl.load($(candidate_vmm_14_ptr) + (((((((r170_z / $(512)) % $(8)) * $(32)) + (((r170_z / $(4096)) % $(32)) - $(0))) % $(256)) * $(512)) + (r170_z % $(512))), dtype=tl.bfloat16)
    r170_value = r170_value.to(tl.bfloat16)
    r170_value = tl.broadcast_to(r170_value, [$(131072)])
    tl.store($(candidate_vcat_13_ptr) + r170_z, r170_value)
    r171_row = tl.arange($(32))
    r171_col = tl.arange($(128))
    r171_lane = tl.arange($(512))
    for r171_group in range($(8)) {
      r171_left = tl.load($(candidate_vcat_13_ptr) + (((((((r171_group % $(8)) * $(16384)) + (r171_row[:, None] * $(512)) + r171_lane[None, :]) / $(512)) % $(32)) * $(4096)) + ((((((r171_group % $(8)) * $(16384)) + (r171_row[:, None] * $(512)) + r171_lane[None, :]) / $(16384)) % $(8)) * $(512)) + ((((r171_group % $(8)) * $(16384)) + (r171_row[:, None] * $(512)) + r171_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r171_right = tl.load($(candidate_vto_copy_5_ptr) + (((((((r171_group % $(8)) * $(65536)) + (r171_lane[:, None] * $(128)) + r171_col[None, :]) / $(65536)) % $(8)) * $(131072)) + (((((r171_group % $(8)) * $(65536)) + (r171_lane[:, None] * $(128)) + r171_col[None, :]) % $(128)) * $(512)) + (((((r171_group % $(8)) * $(65536)) + (r171_lane[:, None] * $(128)) + r171_col[None, :]) / $(128)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r171_value = tl.dot(r171_left, r171_right)
      r171_value = r171_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_17_ptr) + ((r171_group * $(4096)) + (r171_row[:, None] * $(128)) + r171_col[None, :]), r171_value)
    }
    r172_row = tl.arange($(32))
    r172_col = tl.arange($(512))
    r172_lane = tl.arange($(128))
    for r172_group in range($(8)) {
      r172_left = tl.load($(candidate_vmm_1_ptr) + (((((((r172_group % $(8)) * $(4096)) + (r172_row[:, None] * $(128)) + r172_lane[None, :]) / $(128)) % $(32)) * $(1536)) + ((((((r172_group % $(8)) * $(4096)) + (r172_row[:, None] * $(128)) + r172_lane[None, :]) / $(4096)) % $(8)) * $(192)) + ((((r172_group % $(8)) * $(4096)) + (r172_row[:, None] * $(128)) + r172_lane[None, :]) % $(128)) + $(294912)), dtype=tl.bfloat16)
      r172_right = tl.load($(candidate_vto_copy_5_ptr) + (((((((r172_group % $(8)) * $(65536)) + (r172_lane[:, None] * $(512)) + r172_col[None, :]) / $(65536)) % $(8)) * $(131072)) + ((((((r172_group % $(8)) * $(65536)) + (r172_lane[:, None] * $(512)) + r172_col[None, :]) / $(512)) % $(128)) * $(512)) + ((((r172_group % $(8)) * $(65536)) + (r172_lane[:, None] * $(512)) + r172_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r172_value = tl.dot(r172_left, r172_right)
      r172_value = r172_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_18_ptr) + ((r172_group * $(16384)) + (r172_row[:, None] * $(512)) + r172_col[None, :]), r172_value)
    }
    r173_z = tl.arange($(16384))
    r173_value = tl.load($(candidate_vcat_5_ptr) + ((((r173_z / $(2048)) % $(8)) * $(18432)) + (((r173_z / $(64)) % $(32)) * $(64)) + (r173_z % $(64)) + $(12288)), dtype=tl.bfloat16)
    r173_value = r173_value.to(tl.float32)
    r173_value = tl.broadcast_to(r173_value, [$(16384)])
    tl.store($(candidate_vto_copy_52_ptr) + r173_z, r173_value)
    r174_z = tl.arange($(2048))
    r174_value = tl.load($(candidate_vcat_7_ptr) + ((((r174_z / $(64)) % $(32)) * $(576)) + (r174_z % $(64)) + $(111104)), dtype=tl.bfloat16)
    r174_value = r174_value.to(tl.float32)
    r174_value = tl.broadcast_to(r174_value, [$(2048)])
    tl.store($(candidate_vto_copy_53_ptr) + r174_z, r174_value)
    r175_row = tl.arange($(32))
    r175_col = tl.arange($(32))
    r175_lane = tl.arange($(64))
    for r175_group in range($(8)) {
      r175_left = tl.load($(candidate_vto_copy_52_ptr) + (((((((r175_group % $(8)) * $(2048)) + (r175_row[:, None] * $(64)) + r175_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r175_group % $(8)) * $(2048)) + (r175_row[:, None] * $(64)) + r175_lane[None, :]) / $(64)) % $(32)) * $(64)) + ((((r175_group % $(8)) * $(2048)) + (r175_row[:, None] * $(64)) + r175_lane[None, :]) % $(64))), dtype=tl.float32)
      r175_right = tl.load($(candidate_vto_copy_53_ptr) + ((((((r175_group % $(8)) * $(2048)) + (r175_lane[:, None] * $(32)) + r175_col[None, :]) % $(32)) * $(64)) + (((((r175_group % $(8)) * $(2048)) + (r175_lane[:, None] * $(32)) + r175_col[None, :]) / $(32)) % $(64))), dtype=tl.float32)
      r175_value = tl.dot(r175_left, r175_right)
      r175_value = r175_value.to(tl.float32)
      tl.store($(candidate_vbmm_19_ptr) + ((r175_group * $(1024)) + (r175_row[:, None] * $(32)) + r175_col[None, :]), r175_value)
    }
    r176_z = tl.arange($(131072))
    r176_value = tl.load($(candidate_vbmm_18_ptr) + ((((r176_z / $(16384)) % $(8)) * $(16384)) + (((r176_z / $(512)) % $(32)) * $(512)) + (r176_z % $(512))), dtype=tl.bfloat16)
    r176_value = r176_value.to(tl.float32)
    r176_value = tl.broadcast_to(r176_value, [$(131072)])
    tl.store($(candidate_vto_copy_54_ptr) + r176_z, r176_value)
    r177_z = tl.arange($(16384))
    r177_value = tl.load($(candidate_vcat_7_ptr) + ((((r177_z / $(512)) % $(32)) * $(576)) + (r177_z % $(512)) + $(110592)), dtype=tl.bfloat16)
    r177_value = r177_value.to(tl.float32)
    r177_value = tl.broadcast_to(r177_value, [$(16384)])
    tl.store($(candidate_vto_copy_55_ptr) + r177_z, r177_value)
    r178_row = tl.arange($(256))
    r178_col = tl.arange($(32))
    r178_lane = tl.arange($(512))
    r178_left = tl.load($(candidate_vto_copy_54_ptr) + ((((((r178_row[:, None] * $(512)) + r178_lane[None, :]) / $(16384)) % $(8)) * $(16384)) + (((((r178_row[:, None] * $(512)) + r178_lane[None, :]) / $(512)) % $(32)) * $(512)) + (((r178_row[:, None] * $(512)) + r178_lane[None, :]) % $(512))), dtype=tl.float32)
    r178_right = tl.load($(candidate_vto_copy_55_ptr) + (((((r178_lane[:, None] * $(32)) + r178_col[None, :]) % $(32)) * $(512)) + ((((r178_lane[:, None] * $(32)) + r178_col[None, :]) / $(32)) % $(512))), dtype=tl.float32)
    r178_value = tl.dot(r178_left, r178_right)
    r178_value = r178_value.to(tl.float32)
    tl.store($(candidate_vmm_15_ptr) + ((r178_row[:, None] * $(32)) + r178_col[None, :]), r178_value)
    r179_z = tl.arange($(8192))
    r179_value = (tl.load($(candidate_vbmm_19_ptr) + ((((r179_z / $(1024)) % $(8)) * $(1024)) + (((r179_z / $(32)) % $(32)) * $(32)) + (r179_z % $(32))), dtype=tl.float32) + tl.load($(candidate_vmm_15_ptr) + ((((r179_z / $(1024)) % $(8)) * $(1024)) + (((r179_z / $(32)) % $(32)) * $(32)) + (r179_z % $(32))), dtype=tl.float32))
    r179_value = r179_value.to(tl.float32)
    r179_value = tl.broadcast_to(r179_value, [$(8192)])
    tl.store($(candidate_vadd_18_ptr) + r179_z, r179_value)
    r180_z = tl.arange($(8192))
    r180_value = (tl.load($(candidate_vadd_18_ptr) + ((((r180_z / $(1024)) % $(8)) * $(1024)) + (((r180_z / $(32)) % $(32)) * $(32)) + (r180_z % $(32))), dtype=tl.float32) * 0.07216878364870322)
    r180_value = r180_value.to(tl.float32)
    r180_value = tl.broadcast_to(r180_value, [$(8192)])
    tl.store($(candidate_vmul_27_ptr) + r180_z, r180_value)
    r181_z = tl.arange($(32))
    r181_value = (tl.cast(r181_z, tl.int64) + tl.full([], $(32), dtype=tl.int64))
    r181_value = tl.broadcast_to(r181_value, [$(32)])
    tl.store($(candidate_vadd_19_ptr) + r181_z, r181_value)
    r182_z = tl.arange($(32))
    r182_value = (tl.load($(candidate_vadd_19_ptr) + r182_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r182_value = tl.broadcast_to(r182_value, [$(32)])
    tl.store($(candidate_vsub_21_ptr) + r182_z, r182_value)
    r183_z = tl.arange($(1024))
    r183_value = (tl.cast((r183_z % $(32)), tl.int64) > tl.load($(candidate_vsub_21_ptr) + ((r183_z / $(32)) % $(32)), dtype=tl.int64))
    r183_value = tl.broadcast_to(r183_value, [$(1024)])
    tl.store($(candidate_vgt_6_ptr) + r183_z, r183_value)
    r184_z = tl.arange($(8192))
    r184_value = tl.where(tl.load($(candidate_vgt_6_ptr) + ((((r184_z / $(32)) % $(32)) * $(32)) + (r184_z % $(32))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_27_ptr) + ((((r184_z / $(1024)) % $(8)) * $(1024)) + (((r184_z / $(32)) % $(32)) * $(32)) + (r184_z % $(32))), dtype=tl.float32))
    r184_value = r184_value.to(tl.float32)
    r184_value = tl.broadcast_to(r184_value, [$(8192)])
    tl.store($(candidate_vmasked_fill_6_ptr) + r184_z, r184_value)
    r185_z = tl.arange($(8192))
    r185_lane = tl.arange($(32))
    r185_row_values = tl.load($(candidate_vmasked_fill_6_ptr) + ((((r185_z[:, None] / $(1024)) % $(8)) * $(1024)) + (((r185_z[:, None] / $(32)) % $(32)) * $(32)) + (r185_lane[None, :] % $(32))), dtype=tl.float32)
    r185_maximum = tl.max(r185_row_values, axis=1, keep_dims=true)
    r185_weights = tl.exp(r185_row_values - r185_maximum)
    r185_total = tl.sum(r185_weights, axis=1)
    r185_maximum = tl.max(r185_row_values, axis=1)
    r185_value = tl.exp(tl.load($(candidate_vmasked_fill_6_ptr) + ((((r185_z / $(1024)) % $(8)) * $(1024)) + (((r185_z / $(32)) % $(32)) * $(32)) + (r185_z % $(32))), dtype=tl.float32) - r185_maximum) / r185_total
    r185_value = r185_value.to(tl.float32)
    r185_value = tl.broadcast_to(r185_value, [$(8192)])
    tl.store($(candidate_vsoftmax_6_ptr) + r185_z, r185_value)
    r186_z = tl.arange($(8192))
    r186_value = tl.load($(candidate_vsoftmax_6_ptr) + ((((r186_z / $(1024)) % $(8)) * $(1024)) + (((r186_z / $(32)) % $(32)) * $(32)) + (r186_z % $(32))), dtype=tl.float32)
    r186_value = r186_value.to(tl.bfloat16)
    r186_value = tl.broadcast_to(r186_value, [$(8192)])
    tl.store($(candidate_vto_copy_56_ptr) + r186_z, r186_value)
    r187_row = tl.arange($(256))
    r187_col = tl.arange($(512))
    r187_lane = tl.arange($(32))
    r187_left = tl.load($(candidate_vto_copy_56_ptr) + ((((((r187_row[:, None] * $(32)) + r187_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + (((((r187_row[:, None] * $(32)) + r187_lane[None, :]) / $(32)) % $(32)) * $(32)) + (((r187_row[:, None] * $(32)) + r187_lane[None, :]) % $(32))), dtype=tl.bfloat16)
    r187_right = tl.load($(candidate_vcat_7_ptr) + ((((((r187_lane[:, None] * $(512)) + r187_col[None, :]) / $(512)) % $(32)) * $(576)) + (((r187_lane[:, None] * $(512)) + r187_col[None, :]) % $(512)) + $(110592)), dtype=tl.bfloat16)
    r187_value = tl.dot(r187_left, r187_right)
    r187_value = r187_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_16_ptr) + ((r187_row[:, None] * $(512)) + r187_col[None, :]), r187_value)
    r188_z = tl.arange($(131072))
    r188_value = tl.load($(candidate_vmm_16_ptr) + (((((((r188_z / $(512)) % $(8)) * $(32)) + (((r188_z / $(4096)) % $(32)) - $(0))) % $(256)) * $(512)) + (r188_z % $(512))), dtype=tl.bfloat16)
    r188_value = r188_value.to(tl.bfloat16)
    r188_value = tl.broadcast_to(r188_value, [$(131072)])
    tl.store($(candidate_vcat_14_ptr) + r188_z, r188_value)
    r189_row = tl.arange($(32))
    r189_col = tl.arange($(128))
    r189_lane = tl.arange($(512))
    for r189_group in range($(8)) {
      r189_left = tl.load($(candidate_vcat_14_ptr) + (((((((r189_group % $(8)) * $(16384)) + (r189_row[:, None] * $(512)) + r189_lane[None, :]) / $(512)) % $(32)) * $(4096)) + ((((((r189_group % $(8)) * $(16384)) + (r189_row[:, None] * $(512)) + r189_lane[None, :]) / $(16384)) % $(8)) * $(512)) + ((((r189_group % $(8)) * $(16384)) + (r189_row[:, None] * $(512)) + r189_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r189_right = tl.load($(candidate_vto_copy_5_ptr) + (((((((r189_group % $(8)) * $(65536)) + (r189_lane[:, None] * $(128)) + r189_col[None, :]) / $(65536)) % $(8)) * $(131072)) + (((((r189_group % $(8)) * $(65536)) + (r189_lane[:, None] * $(128)) + r189_col[None, :]) % $(128)) * $(512)) + (((((r189_group % $(8)) * $(65536)) + (r189_lane[:, None] * $(128)) + r189_col[None, :]) / $(128)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r189_value = tl.dot(r189_left, r189_right)
      r189_value = r189_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_20_ptr) + ((r189_group * $(4096)) + (r189_row[:, None] * $(128)) + r189_col[None, :]), r189_value)
    }
    r190_row = tl.arange($(32))
    r190_col = tl.arange($(512))
    r190_lane = tl.arange($(128))
    for r190_group in range($(8)) {
      r190_left = tl.load($(candidate_vmm_1_ptr) + (((((((r190_group % $(8)) * $(4096)) + (r190_row[:, None] * $(128)) + r190_lane[None, :]) / $(128)) % $(32)) * $(1536)) + ((((((r190_group % $(8)) * $(4096)) + (r190_row[:, None] * $(128)) + r190_lane[None, :]) / $(4096)) % $(8)) * $(192)) + ((((r190_group % $(8)) * $(4096)) + (r190_row[:, None] * $(128)) + r190_lane[None, :]) % $(128)) + $(344064)), dtype=tl.bfloat16)
      r190_right = tl.load($(candidate_vto_copy_5_ptr) + (((((((r190_group % $(8)) * $(65536)) + (r190_lane[:, None] * $(512)) + r190_col[None, :]) / $(65536)) % $(8)) * $(131072)) + ((((((r190_group % $(8)) * $(65536)) + (r190_lane[:, None] * $(512)) + r190_col[None, :]) / $(512)) % $(128)) * $(512)) + ((((r190_group % $(8)) * $(65536)) + (r190_lane[:, None] * $(512)) + r190_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r190_value = tl.dot(r190_left, r190_right)
      r190_value = r190_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_21_ptr) + ((r190_group * $(16384)) + (r190_row[:, None] * $(512)) + r190_col[None, :]), r190_value)
    }
    r191_z = tl.arange($(16384))
    r191_value = tl.load($(candidate_vcat_5_ptr) + ((((r191_z / $(2048)) % $(8)) * $(18432)) + (((r191_z / $(64)) % $(32)) * $(64)) + (r191_z % $(64)) + $(14336)), dtype=tl.bfloat16)
    r191_value = r191_value.to(tl.float32)
    r191_value = tl.broadcast_to(r191_value, [$(16384)])
    tl.store($(candidate_vto_copy_57_ptr) + r191_z, r191_value)
    r192_z = tl.arange($(4096))
    r192_value = tl.load($(candidate_vcat_7_ptr) + ((((r192_z / $(64)) % $(64)) * $(576)) + (r192_z % $(64)) + $(111104)), dtype=tl.bfloat16)
    r192_value = r192_value.to(tl.float32)
    r192_value = tl.broadcast_to(r192_value, [$(4096)])
    tl.store($(candidate_vto_copy_58_ptr) + r192_z, r192_value)
    r193_row = tl.arange($(32))
    r193_col = tl.arange($(64))
    r193_lane = tl.arange($(64))
    for r193_group in range($(8)) {
      r193_left = tl.load($(candidate_vto_copy_57_ptr) + (((((((r193_group % $(8)) * $(2048)) + (r193_row[:, None] * $(64)) + r193_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r193_group % $(8)) * $(2048)) + (r193_row[:, None] * $(64)) + r193_lane[None, :]) / $(64)) % $(32)) * $(64)) + ((((r193_group % $(8)) * $(2048)) + (r193_row[:, None] * $(64)) + r193_lane[None, :]) % $(64))), dtype=tl.float32)
      r193_right = tl.load($(candidate_vto_copy_58_ptr) + ((((((r193_group % $(8)) * $(4096)) + (r193_lane[:, None] * $(64)) + r193_col[None, :]) % $(64)) * $(64)) + (((((r193_group % $(8)) * $(4096)) + (r193_lane[:, None] * $(64)) + r193_col[None, :]) / $(64)) % $(64))), dtype=tl.float32)
      r193_value = tl.dot(r193_left, r193_right)
      r193_value = r193_value.to(tl.float32)
      tl.store($(candidate_vbmm_22_ptr) + ((r193_group * $(2048)) + (r193_row[:, None] * $(64)) + r193_col[None, :]), r193_value)
    }
    r194_z = tl.arange($(131072))
    r194_value = tl.load($(candidate_vbmm_21_ptr) + ((((r194_z / $(16384)) % $(8)) * $(16384)) + (((r194_z / $(512)) % $(32)) * $(512)) + (r194_z % $(512))), dtype=tl.bfloat16)
    r194_value = r194_value.to(tl.float32)
    r194_value = tl.broadcast_to(r194_value, [$(131072)])
    tl.store($(candidate_vto_copy_59_ptr) + r194_z, r194_value)
    r195_z = tl.arange($(32768))
    r195_value = tl.load($(candidate_vcat_7_ptr) + ((((r195_z / $(512)) % $(64)) * $(576)) + (r195_z % $(512)) + $(110592)), dtype=tl.bfloat16)
    r195_value = r195_value.to(tl.float32)
    r195_value = tl.broadcast_to(r195_value, [$(32768)])
    tl.store($(candidate_vto_copy_60_ptr) + r195_z, r195_value)
    r196_row = tl.arange($(256))
    r196_col = tl.arange($(64))
    r196_lane = tl.arange($(512))
    r196_left = tl.load($(candidate_vto_copy_59_ptr) + ((((((r196_row[:, None] * $(512)) + r196_lane[None, :]) / $(16384)) % $(8)) * $(16384)) + (((((r196_row[:, None] * $(512)) + r196_lane[None, :]) / $(512)) % $(32)) * $(512)) + (((r196_row[:, None] * $(512)) + r196_lane[None, :]) % $(512))), dtype=tl.float32)
    r196_right = tl.load($(candidate_vto_copy_60_ptr) + (((((r196_lane[:, None] * $(64)) + r196_col[None, :]) % $(64)) * $(512)) + ((((r196_lane[:, None] * $(64)) + r196_col[None, :]) / $(64)) % $(512))), dtype=tl.float32)
    r196_value = tl.dot(r196_left, r196_right)
    r196_value = r196_value.to(tl.float32)
    tl.store($(candidate_vmm_17_ptr) + ((r196_row[:, None] * $(64)) + r196_col[None, :]), r196_value)
    r197_z = tl.arange($(16384))
    r197_value = (tl.load($(candidate_vbmm_22_ptr) + ((((r197_z / $(2048)) % $(8)) * $(2048)) + (((r197_z / $(64)) % $(32)) * $(64)) + (r197_z % $(64))), dtype=tl.float32) + tl.load($(candidate_vmm_17_ptr) + ((((r197_z / $(2048)) % $(8)) * $(2048)) + (((r197_z / $(64)) % $(32)) * $(64)) + (r197_z % $(64))), dtype=tl.float32))
    r197_value = r197_value.to(tl.float32)
    r197_value = tl.broadcast_to(r197_value, [$(16384)])
    tl.store($(candidate_vadd_20_ptr) + r197_z, r197_value)
    r198_z = tl.arange($(16384))
    r198_value = (tl.load($(candidate_vadd_20_ptr) + ((((r198_z / $(2048)) % $(8)) * $(2048)) + (((r198_z / $(64)) % $(32)) * $(64)) + (r198_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r198_value = r198_value.to(tl.float32)
    r198_value = tl.broadcast_to(r198_value, [$(16384)])
    tl.store($(candidate_vmul_28_ptr) + r198_z, r198_value)
    r199_z = tl.arange($(32))
    r199_value = (tl.cast(r199_z, tl.int64) + tl.full([], $(64), dtype=tl.int64))
    r199_value = tl.broadcast_to(r199_value, [$(32)])
    tl.store($(candidate_vadd_21_ptr) + r199_z, r199_value)
    r200_z = tl.arange($(32))
    r200_value = (tl.load($(candidate_vadd_21_ptr) + r200_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r200_value = tl.broadcast_to(r200_value, [$(32)])
    tl.store($(candidate_vsub_22_ptr) + r200_z, r200_value)
    r201_z = tl.arange($(2048))
    r201_value = (tl.cast((r201_z % $(64)), tl.int64) > tl.load($(candidate_vsub_22_ptr) + ((r201_z / $(64)) % $(32)), dtype=tl.int64))
    r201_value = tl.broadcast_to(r201_value, [$(2048)])
    tl.store($(candidate_vgt_7_ptr) + r201_z, r201_value)
    r202_z = tl.arange($(16384))
    r202_value = tl.where(tl.load($(candidate_vgt_7_ptr) + ((((r202_z / $(64)) % $(32)) * $(64)) + (r202_z % $(64))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_28_ptr) + ((((r202_z / $(2048)) % $(8)) * $(2048)) + (((r202_z / $(64)) % $(32)) * $(64)) + (r202_z % $(64))), dtype=tl.float32))
    r202_value = r202_value.to(tl.float32)
    r202_value = tl.broadcast_to(r202_value, [$(16384)])
    tl.store($(candidate_vmasked_fill_7_ptr) + r202_z, r202_value)
    r203_z = tl.arange($(16384))
    r203_lane = tl.arange($(64))
    r203_row_values = tl.load($(candidate_vmasked_fill_7_ptr) + ((((r203_z[:, None] / $(2048)) % $(8)) * $(2048)) + (((r203_z[:, None] / $(64)) % $(32)) * $(64)) + (r203_lane[None, :] % $(64))), dtype=tl.float32)
    r203_maximum = tl.max(r203_row_values, axis=1, keep_dims=true)
    r203_weights = tl.exp(r203_row_values - r203_maximum)
    r203_total = tl.sum(r203_weights, axis=1)
    r203_maximum = tl.max(r203_row_values, axis=1)
    r203_value = tl.exp(tl.load($(candidate_vmasked_fill_7_ptr) + ((((r203_z / $(2048)) % $(8)) * $(2048)) + (((r203_z / $(64)) % $(32)) * $(64)) + (r203_z % $(64))), dtype=tl.float32) - r203_maximum) / r203_total
    r203_value = r203_value.to(tl.float32)
    r203_value = tl.broadcast_to(r203_value, [$(16384)])
    tl.store($(candidate_vsoftmax_7_ptr) + r203_z, r203_value)
    r204_z = tl.arange($(16384))
    r204_value = tl.load($(candidate_vsoftmax_7_ptr) + ((((r204_z / $(2048)) % $(8)) * $(2048)) + (((r204_z / $(64)) % $(32)) * $(64)) + (r204_z % $(64))), dtype=tl.float32)
    r204_value = r204_value.to(tl.bfloat16)
    r204_value = tl.broadcast_to(r204_value, [$(16384)])
    tl.store($(candidate_vto_copy_61_ptr) + r204_z, r204_value)
    r205_row = tl.arange($(256))
    r205_col = tl.arange($(512))
    r205_lane = tl.arange($(64))
    r205_left = tl.load($(candidate_vto_copy_61_ptr) + ((((((r205_row[:, None] * $(64)) + r205_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + (((((r205_row[:, None] * $(64)) + r205_lane[None, :]) / $(64)) % $(32)) * $(64)) + (((r205_row[:, None] * $(64)) + r205_lane[None, :]) % $(64))), dtype=tl.bfloat16)
    r205_right = tl.load($(candidate_vcat_7_ptr) + ((((((r205_lane[:, None] * $(512)) + r205_col[None, :]) / $(512)) % $(64)) * $(576)) + (((r205_lane[:, None] * $(512)) + r205_col[None, :]) % $(512)) + $(110592)), dtype=tl.bfloat16)
    r205_value = tl.dot(r205_left, r205_right)
    r205_value = r205_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_18_ptr) + ((r205_row[:, None] * $(512)) + r205_col[None, :]), r205_value)
    r206_z = tl.arange($(131072))
    r206_value = tl.load($(candidate_vmm_18_ptr) + (((((((r206_z / $(512)) % $(8)) * $(32)) + (((r206_z / $(4096)) % $(32)) - $(0))) % $(256)) * $(512)) + (r206_z % $(512))), dtype=tl.bfloat16)
    r206_value = r206_value.to(tl.bfloat16)
    r206_value = tl.broadcast_to(r206_value, [$(131072)])
    tl.store($(candidate_vcat_15_ptr) + r206_z, r206_value)
    r207_row = tl.arange($(32))
    r207_col = tl.arange($(128))
    r207_lane = tl.arange($(512))
    for r207_group in range($(8)) {
      r207_left = tl.load($(candidate_vcat_15_ptr) + (((((((r207_group % $(8)) * $(16384)) + (r207_row[:, None] * $(512)) + r207_lane[None, :]) / $(512)) % $(32)) * $(4096)) + ((((((r207_group % $(8)) * $(16384)) + (r207_row[:, None] * $(512)) + r207_lane[None, :]) / $(16384)) % $(8)) * $(512)) + ((((r207_group % $(8)) * $(16384)) + (r207_row[:, None] * $(512)) + r207_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r207_right = tl.load($(candidate_vto_copy_5_ptr) + (((((((r207_group % $(8)) * $(65536)) + (r207_lane[:, None] * $(128)) + r207_col[None, :]) / $(65536)) % $(8)) * $(131072)) + (((((r207_group % $(8)) * $(65536)) + (r207_lane[:, None] * $(128)) + r207_col[None, :]) % $(128)) * $(512)) + (((((r207_group % $(8)) * $(65536)) + (r207_lane[:, None] * $(128)) + r207_col[None, :]) / $(128)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r207_value = tl.dot(r207_left, r207_right)
      r207_value = r207_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_23_ptr) + ((r207_group * $(4096)) + (r207_row[:, None] * $(128)) + r207_col[None, :]), r207_value)
    }
    r208_row = tl.arange($(32))
    r208_col = tl.arange($(512))
    r208_lane = tl.arange($(128))
    for r208_group in range($(8)) {
      r208_left = tl.load($(candidate_vmm_1_ptr) + (((((((r208_group % $(8)) * $(4096)) + (r208_row[:, None] * $(128)) + r208_lane[None, :]) / $(128)) % $(32)) * $(1536)) + ((((((r208_group % $(8)) * $(4096)) + (r208_row[:, None] * $(128)) + r208_lane[None, :]) / $(4096)) % $(8)) * $(192)) + ((((r208_group % $(8)) * $(4096)) + (r208_row[:, None] * $(128)) + r208_lane[None, :]) % $(128)) + $(393216)), dtype=tl.bfloat16)
      r208_right = tl.load($(candidate_vto_copy_5_ptr) + (((((((r208_group % $(8)) * $(65536)) + (r208_lane[:, None] * $(512)) + r208_col[None, :]) / $(65536)) % $(8)) * $(131072)) + ((((((r208_group % $(8)) * $(65536)) + (r208_lane[:, None] * $(512)) + r208_col[None, :]) / $(512)) % $(128)) * $(512)) + ((((r208_group % $(8)) * $(65536)) + (r208_lane[:, None] * $(512)) + r208_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r208_value = tl.dot(r208_left, r208_right)
      r208_value = r208_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_24_ptr) + ((r208_group * $(16384)) + (r208_row[:, None] * $(512)) + r208_col[None, :]), r208_value)
    }
    r209_z = tl.arange($(16384))
    r209_value = tl.load($(candidate_vcat_5_ptr) + ((((r209_z / $(2048)) % $(8)) * $(18432)) + (((r209_z / $(64)) % $(32)) * $(64)) + (r209_z % $(64)) + $(16384)), dtype=tl.bfloat16)
    r209_value = r209_value.to(tl.float32)
    r209_value = tl.broadcast_to(r209_value, [$(16384)])
    tl.store($(candidate_vto_copy_62_ptr) + r209_z, r209_value)
    r210_z = tl.arange($(6144))
    r210_value = tl.load($(candidate_vcat_7_ptr) + ((((r210_z / $(64)) % $(96)) * $(576)) + (r210_z % $(64)) + $(111104)), dtype=tl.bfloat16)
    r210_value = r210_value.to(tl.float32)
    r210_value = tl.broadcast_to(r210_value, [$(6144)])
    tl.store($(candidate_vto_copy_63_ptr) + r210_z, r210_value)
    r211_row = tl.arange($(32))
    r211_col = tl.arange($(96))
    r211_lane = tl.arange($(64))
    for r211_group in range($(8)) {
      r211_left = tl.load($(candidate_vto_copy_62_ptr) + (((((((r211_group % $(8)) * $(2048)) + (r211_row[:, None] * $(64)) + r211_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r211_group % $(8)) * $(2048)) + (r211_row[:, None] * $(64)) + r211_lane[None, :]) / $(64)) % $(32)) * $(64)) + ((((r211_group % $(8)) * $(2048)) + (r211_row[:, None] * $(64)) + r211_lane[None, :]) % $(64))), dtype=tl.float32)
      r211_right = tl.load($(candidate_vto_copy_63_ptr) + ((((((r211_group % $(8)) * $(6144)) + (r211_lane[:, None] * $(96)) + r211_col[None, :]) % $(96)) * $(64)) + (((((r211_group % $(8)) * $(6144)) + (r211_lane[:, None] * $(96)) + r211_col[None, :]) / $(96)) % $(64))), dtype=tl.float32)
      r211_value = tl.dot(r211_left, r211_right)
      r211_value = r211_value.to(tl.float32)
      tl.store($(candidate_vbmm_25_ptr) + ((r211_group * $(3072)) + (r211_row[:, None] * $(96)) + r211_col[None, :]), r211_value)
    }
    r212_z = tl.arange($(131072))
    r212_value = tl.load($(candidate_vbmm_24_ptr) + ((((r212_z / $(16384)) % $(8)) * $(16384)) + (((r212_z / $(512)) % $(32)) * $(512)) + (r212_z % $(512))), dtype=tl.bfloat16)
    r212_value = r212_value.to(tl.float32)
    r212_value = tl.broadcast_to(r212_value, [$(131072)])
    tl.store($(candidate_vto_copy_64_ptr) + r212_z, r212_value)
    r213_z = tl.arange($(49152))
    r213_value = tl.load($(candidate_vcat_7_ptr) + ((((r213_z / $(512)) % $(96)) * $(576)) + (r213_z % $(512)) + $(110592)), dtype=tl.bfloat16)
    r213_value = r213_value.to(tl.float32)
    r213_value = tl.broadcast_to(r213_value, [$(49152)])
    tl.store($(candidate_vto_copy_65_ptr) + r213_z, r213_value)
    r214_row = tl.arange($(256))
    r214_col = tl.arange($(96))
    r214_lane = tl.arange($(512))
    r214_left = tl.load($(candidate_vto_copy_64_ptr) + ((((((r214_row[:, None] * $(512)) + r214_lane[None, :]) / $(16384)) % $(8)) * $(16384)) + (((((r214_row[:, None] * $(512)) + r214_lane[None, :]) / $(512)) % $(32)) * $(512)) + (((r214_row[:, None] * $(512)) + r214_lane[None, :]) % $(512))), dtype=tl.float32)
    r214_right = tl.load($(candidate_vto_copy_65_ptr) + (((((r214_lane[:, None] * $(96)) + r214_col[None, :]) % $(96)) * $(512)) + ((((r214_lane[:, None] * $(96)) + r214_col[None, :]) / $(96)) % $(512))), dtype=tl.float32)
    r214_value = tl.dot(r214_left, r214_right)
    r214_value = r214_value.to(tl.float32)
    tl.store($(candidate_vmm_19_ptr) + ((r214_row[:, None] * $(96)) + r214_col[None, :]), r214_value)
    r215_z = tl.arange($(24576))
    r215_value = (tl.load($(candidate_vbmm_25_ptr) + ((((r215_z / $(3072)) % $(8)) * $(3072)) + (((r215_z / $(96)) % $(32)) * $(96)) + (r215_z % $(96))), dtype=tl.float32) + tl.load($(candidate_vmm_19_ptr) + ((((r215_z / $(3072)) % $(8)) * $(3072)) + (((r215_z / $(96)) % $(32)) * $(96)) + (r215_z % $(96))), dtype=tl.float32))
    r215_value = r215_value.to(tl.float32)
    r215_value = tl.broadcast_to(r215_value, [$(24576)])
    tl.store($(candidate_vadd_22_ptr) + r215_z, r215_value)
    r216_z = tl.arange($(24576))
    r216_value = (tl.load($(candidate_vadd_22_ptr) + ((((r216_z / $(3072)) % $(8)) * $(3072)) + (((r216_z / $(96)) % $(32)) * $(96)) + (r216_z % $(96))), dtype=tl.float32) * 0.07216878364870322)
    r216_value = r216_value.to(tl.float32)
    r216_value = tl.broadcast_to(r216_value, [$(24576)])
    tl.store($(candidate_vmul_29_ptr) + r216_z, r216_value)
    r217_z = tl.arange($(32))
    r217_value = (tl.cast(r217_z, tl.int64) + tl.full([], $(96), dtype=tl.int64))
    r217_value = tl.broadcast_to(r217_value, [$(32)])
    tl.store($(candidate_vadd_23_ptr) + r217_z, r217_value)
    r218_z = tl.arange($(32))
    r218_value = (tl.load($(candidate_vadd_23_ptr) + r218_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r218_value = tl.broadcast_to(r218_value, [$(32)])
    tl.store($(candidate_vsub_23_ptr) + r218_z, r218_value)
    r219_z = tl.arange($(3072))
    r219_value = (tl.cast((r219_z % $(96)), tl.int64) > tl.load($(candidate_vsub_23_ptr) + ((r219_z / $(96)) % $(32)), dtype=tl.int64))
    r219_value = tl.broadcast_to(r219_value, [$(3072)])
    tl.store($(candidate_vgt_8_ptr) + r219_z, r219_value)
    r220_z = tl.arange($(24576))
    r220_value = tl.where(tl.load($(candidate_vgt_8_ptr) + ((((r220_z / $(96)) % $(32)) * $(96)) + (r220_z % $(96))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_29_ptr) + ((((r220_z / $(3072)) % $(8)) * $(3072)) + (((r220_z / $(96)) % $(32)) * $(96)) + (r220_z % $(96))), dtype=tl.float32))
    r220_value = r220_value.to(tl.float32)
    r220_value = tl.broadcast_to(r220_value, [$(24576)])
    tl.store($(candidate_vmasked_fill_8_ptr) + r220_z, r220_value)
    r221_z = tl.arange($(24576))
    r221_lane = tl.arange($(96))
    r221_row_values = tl.load($(candidate_vmasked_fill_8_ptr) + ((((r221_z[:, None] / $(3072)) % $(8)) * $(3072)) + (((r221_z[:, None] / $(96)) % $(32)) * $(96)) + (r221_lane[None, :] % $(96))), dtype=tl.float32)
    r221_maximum = tl.max(r221_row_values, axis=1, keep_dims=true)
    r221_weights = tl.exp(r221_row_values - r221_maximum)
    r221_total = tl.sum(r221_weights, axis=1)
    r221_maximum = tl.max(r221_row_values, axis=1)
    r221_value = tl.exp(tl.load($(candidate_vmasked_fill_8_ptr) + ((((r221_z / $(3072)) % $(8)) * $(3072)) + (((r221_z / $(96)) % $(32)) * $(96)) + (r221_z % $(96))), dtype=tl.float32) - r221_maximum) / r221_total
    r221_value = r221_value.to(tl.float32)
    r221_value = tl.broadcast_to(r221_value, [$(24576)])
    tl.store($(candidate_vsoftmax_8_ptr) + r221_z, r221_value)
    r222_z = tl.arange($(24576))
    r222_value = tl.load($(candidate_vsoftmax_8_ptr) + ((((r222_z / $(3072)) % $(8)) * $(3072)) + (((r222_z / $(96)) % $(32)) * $(96)) + (r222_z % $(96))), dtype=tl.float32)
    r222_value = r222_value.to(tl.bfloat16)
    r222_value = tl.broadcast_to(r222_value, [$(24576)])
    tl.store($(candidate_vto_copy_66_ptr) + r222_z, r222_value)
    r223_row = tl.arange($(256))
    r223_col = tl.arange($(512))
    r223_lane = tl.arange($(96))
    r223_left = tl.load($(candidate_vto_copy_66_ptr) + ((((((r223_row[:, None] * $(96)) + r223_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + (((((r223_row[:, None] * $(96)) + r223_lane[None, :]) / $(96)) % $(32)) * $(96)) + (((r223_row[:, None] * $(96)) + r223_lane[None, :]) % $(96))), dtype=tl.bfloat16)
    r223_right = tl.load($(candidate_vcat_7_ptr) + ((((((r223_lane[:, None] * $(512)) + r223_col[None, :]) / $(512)) % $(96)) * $(576)) + (((r223_lane[:, None] * $(512)) + r223_col[None, :]) % $(512)) + $(110592)), dtype=tl.bfloat16)
    r223_value = tl.dot(r223_left, r223_right)
    r223_value = r223_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_20_ptr) + ((r223_row[:, None] * $(512)) + r223_col[None, :]), r223_value)
    r224_z = tl.arange($(131072))
    r224_value = tl.load($(candidate_vmm_20_ptr) + (((((((r224_z / $(512)) % $(8)) * $(32)) + (((r224_z / $(4096)) % $(32)) - $(0))) % $(256)) * $(512)) + (r224_z % $(512))), dtype=tl.bfloat16)
    r224_value = r224_value.to(tl.bfloat16)
    r224_value = tl.broadcast_to(r224_value, [$(131072)])
    tl.store($(candidate_vcat_16_ptr) + r224_z, r224_value)
    r225_row = tl.arange($(32))
    r225_col = tl.arange($(128))
    r225_lane = tl.arange($(512))
    for r225_group in range($(8)) {
      r225_left = tl.load($(candidate_vcat_16_ptr) + (((((((r225_group % $(8)) * $(16384)) + (r225_row[:, None] * $(512)) + r225_lane[None, :]) / $(512)) % $(32)) * $(4096)) + ((((((r225_group % $(8)) * $(16384)) + (r225_row[:, None] * $(512)) + r225_lane[None, :]) / $(16384)) % $(8)) * $(512)) + ((((r225_group % $(8)) * $(16384)) + (r225_row[:, None] * $(512)) + r225_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r225_right = tl.load($(candidate_vto_copy_5_ptr) + (((((((r225_group % $(8)) * $(65536)) + (r225_lane[:, None] * $(128)) + r225_col[None, :]) / $(65536)) % $(8)) * $(131072)) + (((((r225_group % $(8)) * $(65536)) + (r225_lane[:, None] * $(128)) + r225_col[None, :]) % $(128)) * $(512)) + (((((r225_group % $(8)) * $(65536)) + (r225_lane[:, None] * $(128)) + r225_col[None, :]) / $(128)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r225_value = tl.dot(r225_left, r225_right)
      r225_value = r225_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_26_ptr) + ((r225_group * $(4096)) + (r225_row[:, None] * $(128)) + r225_col[None, :]), r225_value)
    }
    r226_z = tl.arange($(294912))
    r226_value = tl.where(((r226_z / $(1024)) % $(288)) < $(32), tl.load($(candidate_vbmm_2_ptr) + ((((r226_z / $(128)) % $(8)) * $(4096)) + (((((r226_z / $(1024)) % $(288)) - $(0)) % $(32)) * $(128)) + (r226_z % $(128))), dtype=tl.bfloat16), tl.where(((r226_z / $(1024)) % $(288)) < $(64), tl.load($(candidate_vbmm_5_ptr) + ((((r226_z / $(128)) % $(8)) * $(4096)) + (((((r226_z / $(1024)) % $(288)) - $(32)) % $(32)) * $(128)) + (r226_z % $(128))), dtype=tl.bfloat16), tl.where(((r226_z / $(1024)) % $(288)) < $(96), tl.load($(candidate_vbmm_8_ptr) + ((((r226_z / $(128)) % $(8)) * $(4096)) + (((((r226_z / $(1024)) % $(288)) - $(64)) % $(32)) * $(128)) + (r226_z % $(128))), dtype=tl.bfloat16), tl.where(((r226_z / $(1024)) % $(288)) < $(128), tl.load($(candidate_vbmm_11_ptr) + ((((r226_z / $(128)) % $(8)) * $(4096)) + (((((r226_z / $(1024)) % $(288)) - $(96)) % $(32)) * $(128)) + (r226_z % $(128))), dtype=tl.bfloat16), tl.where(((r226_z / $(1024)) % $(288)) < $(160), tl.load($(candidate_vbmm_14_ptr) + ((((r226_z / $(128)) % $(8)) * $(4096)) + (((((r226_z / $(1024)) % $(288)) - $(128)) % $(32)) * $(128)) + (r226_z % $(128))), dtype=tl.bfloat16), tl.where(((r226_z / $(1024)) % $(288)) < $(192), tl.load($(candidate_vbmm_17_ptr) + ((((r226_z / $(128)) % $(8)) * $(4096)) + (((((r226_z / $(1024)) % $(288)) - $(160)) % $(32)) * $(128)) + (r226_z % $(128))), dtype=tl.bfloat16), tl.where(((r226_z / $(1024)) % $(288)) < $(224), tl.load($(candidate_vbmm_20_ptr) + ((((r226_z / $(128)) % $(8)) * $(4096)) + (((((r226_z / $(1024)) % $(288)) - $(192)) % $(32)) * $(128)) + (r226_z % $(128))), dtype=tl.bfloat16), tl.where(((r226_z / $(1024)) % $(288)) < $(256), tl.load($(candidate_vbmm_23_ptr) + ((((r226_z / $(128)) % $(8)) * $(4096)) + (((((r226_z / $(1024)) % $(288)) - $(224)) % $(32)) * $(128)) + (r226_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vbmm_26_ptr) + ((((r226_z / $(128)) % $(8)) * $(4096)) + (((((r226_z / $(1024)) % $(288)) - $(256)) % $(32)) * $(128)) + (r226_z % $(128))), dtype=tl.bfloat16)))))))))
    r226_value = r226_value.to(tl.bfloat16)
    r226_value = tl.broadcast_to(r226_value, [$(294912)])
    tl.store($(candidate_vcat_17_ptr) + r226_z, r226_value)
    r227_row = tl.arange($(288))
    r227_col = tl.arange($(512))
    r227_lane = tl.arange($(1024))
    r227_left = tl.load($(candidate_vcat_17_ptr) + ((((((r227_row[:, None] * $(1024)) + r227_lane[None, :]) / $(1024)) % $(288)) * $(1024)) + (((((((((r227_row[:, None] * $(1024)) + r227_lane[None, :]) / $(1024)) % $(288)) * $(1024)) + (((r227_row[:, None] * $(1024)) + r227_lane[None, :]) % $(1024))) / $(128)) % $(8)) * $(128)) + (((((((r227_row[:, None] * $(1024)) + r227_lane[None, :]) / $(1024)) % $(288)) * $(1024)) + (((r227_row[:, None] * $(1024)) + r227_lane[None, :]) % $(1024))) % $(128))), dtype=tl.bfloat16)
    r227_right = tl.load($(candidate_vto_copy_6_ptr) + (((r227_col[None, :] % $(512)) * $(1024)) + (r227_lane[:, None] % $(1024))), dtype=tl.bfloat16)
    r227_value = tl.dot(r227_left, r227_right)
    r227_value = r227_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_21_ptr) + ((r227_row[:, None] * $(512)) + r227_col[None, :]), r227_value)
    r228_z = tl.arange($(49152))
    r228_value = tl.load($(candidate_vmm_21_ptr) + (((((((tl.load($(candidate_vcat_4_ptr) + ((r228_z / $(512)) % $(96)), dtype=tl.int64) * $(512)) + (r228_z % $(512))) / $(512)) % $(288)) % $(288)) * $(512)) + ((((tl.load($(candidate_vcat_4_ptr) + ((r228_z / $(512)) % $(96)), dtype=tl.int64) * $(512)) + (r228_z % $(512))) % $(512)) % $(512))), dtype=tl.bfloat16)
    r228_value = r228_value.to(tl.bfloat16)
    r228_value = tl.broadcast_to(r228_value, [$(49152)])
    tl.store($(candidate_vindex_2_ptr) + r228_z, r228_value)
    z = tl.arange($(49152))
    value = tl.load($(candidate_vindex_2_ptr) + ((((z / $(512)) % $(96)) * $(512)) + (z % $(512))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(49152)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_mla_fa4_absorb_seqused_fwd
