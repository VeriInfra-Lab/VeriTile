import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_mla_fa4_nonabsorb_seqused_fwd
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
  let candidate_vbmm_1_ptr : RegionName := "candidate_vbmm_1"
  let candidate_vcat_8_ptr : RegionName := "candidate_vcat_8"
  let candidate_vcat_9_ptr : RegionName := "candidate_vcat_9"
  let candidate_vto_copy_22_ptr : RegionName := "candidate_vto_copy_22"
  let candidate_vto_copy_23_ptr : RegionName := "candidate_vto_copy_23"
  let candidate_vbmm_2_ptr : RegionName := "candidate_vbmm_2"
  let candidate_vmul_21_ptr : RegionName := "candidate_vmul_21"
  let candidate_vadd_6_ptr : RegionName := "candidate_vadd_6"
  let candidate_vsub_15_ptr : RegionName := "candidate_vsub_15"
  let candidate_vgt_ptr : RegionName := "candidate_vgt"
  let candidate_vmasked_fill_ptr : RegionName := "candidate_vmasked_fill"
  let candidate_vsoftmax_ptr : RegionName := "candidate_vsoftmax"
  let candidate_vto_copy_24_ptr : RegionName := "candidate_vto_copy_24"
  let candidate_vbmm_3_ptr : RegionName := "candidate_vbmm_3"
  let candidate_vcat_10_ptr : RegionName := "candidate_vcat_10"
  let candidate_vbmm_4_ptr : RegionName := "candidate_vbmm_4"
  let candidate_vbmm_5_ptr : RegionName := "candidate_vbmm_5"
  let candidate_vcat_11_ptr : RegionName := "candidate_vcat_11"
  let candidate_vcat_12_ptr : RegionName := "candidate_vcat_12"
  let candidate_vto_copy_25_ptr : RegionName := "candidate_vto_copy_25"
  let candidate_vto_copy_26_ptr : RegionName := "candidate_vto_copy_26"
  let candidate_vbmm_6_ptr : RegionName := "candidate_vbmm_6"
  let candidate_vmul_22_ptr : RegionName := "candidate_vmul_22"
  let candidate_vadd_7_ptr : RegionName := "candidate_vadd_7"
  let candidate_vsub_16_ptr : RegionName := "candidate_vsub_16"
  let candidate_vgt_1_ptr : RegionName := "candidate_vgt_1"
  let candidate_vmasked_fill_1_ptr : RegionName := "candidate_vmasked_fill_1"
  let candidate_vsoftmax_1_ptr : RegionName := "candidate_vsoftmax_1"
  let candidate_vto_copy_27_ptr : RegionName := "candidate_vto_copy_27"
  let candidate_vbmm_7_ptr : RegionName := "candidate_vbmm_7"
  let candidate_vcat_13_ptr : RegionName := "candidate_vcat_13"
  let candidate_vbmm_8_ptr : RegionName := "candidate_vbmm_8"
  let candidate_vbmm_9_ptr : RegionName := "candidate_vbmm_9"
  let candidate_vcat_14_ptr : RegionName := "candidate_vcat_14"
  let candidate_vcat_15_ptr : RegionName := "candidate_vcat_15"
  let candidate_vto_copy_28_ptr : RegionName := "candidate_vto_copy_28"
  let candidate_vto_copy_29_ptr : RegionName := "candidate_vto_copy_29"
  let candidate_vbmm_10_ptr : RegionName := "candidate_vbmm_10"
  let candidate_vmul_23_ptr : RegionName := "candidate_vmul_23"
  let candidate_vadd_8_ptr : RegionName := "candidate_vadd_8"
  let candidate_vsub_17_ptr : RegionName := "candidate_vsub_17"
  let candidate_vgt_2_ptr : RegionName := "candidate_vgt_2"
  let candidate_vmasked_fill_2_ptr : RegionName := "candidate_vmasked_fill_2"
  let candidate_vsoftmax_2_ptr : RegionName := "candidate_vsoftmax_2"
  let candidate_vto_copy_30_ptr : RegionName := "candidate_vto_copy_30"
  let candidate_vbmm_11_ptr : RegionName := "candidate_vbmm_11"
  let candidate_vcat_16_ptr : RegionName := "candidate_vcat_16"
  let candidate_vbmm_12_ptr : RegionName := "candidate_vbmm_12"
  let candidate_vbmm_13_ptr : RegionName := "candidate_vbmm_13"
  let candidate_vcat_17_ptr : RegionName := "candidate_vcat_17"
  let candidate_vcat_18_ptr : RegionName := "candidate_vcat_18"
  let candidate_vto_copy_31_ptr : RegionName := "candidate_vto_copy_31"
  let candidate_vto_copy_32_ptr : RegionName := "candidate_vto_copy_32"
  let candidate_vbmm_14_ptr : RegionName := "candidate_vbmm_14"
  let candidate_vmul_24_ptr : RegionName := "candidate_vmul_24"
  let candidate_vadd_9_ptr : RegionName := "candidate_vadd_9"
  let candidate_vsub_18_ptr : RegionName := "candidate_vsub_18"
  let candidate_vgt_3_ptr : RegionName := "candidate_vgt_3"
  let candidate_vmasked_fill_3_ptr : RegionName := "candidate_vmasked_fill_3"
  let candidate_vsoftmax_3_ptr : RegionName := "candidate_vsoftmax_3"
  let candidate_vto_copy_33_ptr : RegionName := "candidate_vto_copy_33"
  let candidate_vbmm_15_ptr : RegionName := "candidate_vbmm_15"
  let candidate_vcat_19_ptr : RegionName := "candidate_vcat_19"
  let candidate_vbmm_16_ptr : RegionName := "candidate_vbmm_16"
  let candidate_vbmm_17_ptr : RegionName := "candidate_vbmm_17"
  let candidate_vcat_20_ptr : RegionName := "candidate_vcat_20"
  let candidate_vcat_21_ptr : RegionName := "candidate_vcat_21"
  let candidate_vto_copy_34_ptr : RegionName := "candidate_vto_copy_34"
  let candidate_vto_copy_35_ptr : RegionName := "candidate_vto_copy_35"
  let candidate_vbmm_18_ptr : RegionName := "candidate_vbmm_18"
  let candidate_vmul_25_ptr : RegionName := "candidate_vmul_25"
  let candidate_vadd_10_ptr : RegionName := "candidate_vadd_10"
  let candidate_vsub_19_ptr : RegionName := "candidate_vsub_19"
  let candidate_vgt_4_ptr : RegionName := "candidate_vgt_4"
  let candidate_vmasked_fill_4_ptr : RegionName := "candidate_vmasked_fill_4"
  let candidate_vsoftmax_4_ptr : RegionName := "candidate_vsoftmax_4"
  let candidate_vto_copy_36_ptr : RegionName := "candidate_vto_copy_36"
  let candidate_vbmm_19_ptr : RegionName := "candidate_vbmm_19"
  let candidate_vcat_22_ptr : RegionName := "candidate_vcat_22"
  let candidate_vbmm_20_ptr : RegionName := "candidate_vbmm_20"
  let candidate_vbmm_21_ptr : RegionName := "candidate_vbmm_21"
  let candidate_vcat_23_ptr : RegionName := "candidate_vcat_23"
  let candidate_vcat_24_ptr : RegionName := "candidate_vcat_24"
  let candidate_vto_copy_37_ptr : RegionName := "candidate_vto_copy_37"
  let candidate_vto_copy_38_ptr : RegionName := "candidate_vto_copy_38"
  let candidate_vbmm_22_ptr : RegionName := "candidate_vbmm_22"
  let candidate_vmul_26_ptr : RegionName := "candidate_vmul_26"
  let candidate_vadd_11_ptr : RegionName := "candidate_vadd_11"
  let candidate_vsub_20_ptr : RegionName := "candidate_vsub_20"
  let candidate_vgt_5_ptr : RegionName := "candidate_vgt_5"
  let candidate_vmasked_fill_5_ptr : RegionName := "candidate_vmasked_fill_5"
  let candidate_vsoftmax_5_ptr : RegionName := "candidate_vsoftmax_5"
  let candidate_vto_copy_39_ptr : RegionName := "candidate_vto_copy_39"
  let candidate_vbmm_23_ptr : RegionName := "candidate_vbmm_23"
  let candidate_vcat_25_ptr : RegionName := "candidate_vcat_25"
  let candidate_vbmm_24_ptr : RegionName := "candidate_vbmm_24"
  let candidate_vbmm_25_ptr : RegionName := "candidate_vbmm_25"
  let candidate_vcat_26_ptr : RegionName := "candidate_vcat_26"
  let candidate_vcat_27_ptr : RegionName := "candidate_vcat_27"
  let candidate_vto_copy_40_ptr : RegionName := "candidate_vto_copy_40"
  let candidate_vto_copy_41_ptr : RegionName := "candidate_vto_copy_41"
  let candidate_vbmm_26_ptr : RegionName := "candidate_vbmm_26"
  let candidate_vmul_27_ptr : RegionName := "candidate_vmul_27"
  let candidate_vadd_12_ptr : RegionName := "candidate_vadd_12"
  let candidate_vsub_21_ptr : RegionName := "candidate_vsub_21"
  let candidate_vgt_6_ptr : RegionName := "candidate_vgt_6"
  let candidate_vmasked_fill_6_ptr : RegionName := "candidate_vmasked_fill_6"
  let candidate_vsoftmax_6_ptr : RegionName := "candidate_vsoftmax_6"
  let candidate_vto_copy_42_ptr : RegionName := "candidate_vto_copy_42"
  let candidate_vbmm_27_ptr : RegionName := "candidate_vbmm_27"
  let candidate_vcat_28_ptr : RegionName := "candidate_vcat_28"
  let candidate_vbmm_28_ptr : RegionName := "candidate_vbmm_28"
  let candidate_vbmm_29_ptr : RegionName := "candidate_vbmm_29"
  let candidate_vcat_29_ptr : RegionName := "candidate_vcat_29"
  let candidate_vcat_30_ptr : RegionName := "candidate_vcat_30"
  let candidate_vto_copy_43_ptr : RegionName := "candidate_vto_copy_43"
  let candidate_vto_copy_44_ptr : RegionName := "candidate_vto_copy_44"
  let candidate_vbmm_30_ptr : RegionName := "candidate_vbmm_30"
  let candidate_vmul_28_ptr : RegionName := "candidate_vmul_28"
  let candidate_vadd_13_ptr : RegionName := "candidate_vadd_13"
  let candidate_vsub_22_ptr : RegionName := "candidate_vsub_22"
  let candidate_vgt_7_ptr : RegionName := "candidate_vgt_7"
  let candidate_vmasked_fill_7_ptr : RegionName := "candidate_vmasked_fill_7"
  let candidate_vsoftmax_7_ptr : RegionName := "candidate_vsoftmax_7"
  let candidate_vto_copy_45_ptr : RegionName := "candidate_vto_copy_45"
  let candidate_vbmm_31_ptr : RegionName := "candidate_vbmm_31"
  let candidate_vcat_31_ptr : RegionName := "candidate_vcat_31"
  let candidate_vbmm_32_ptr : RegionName := "candidate_vbmm_32"
  let candidate_vbmm_33_ptr : RegionName := "candidate_vbmm_33"
  let candidate_vcat_32_ptr : RegionName := "candidate_vcat_32"
  let candidate_vcat_33_ptr : RegionName := "candidate_vcat_33"
  let candidate_vto_copy_46_ptr : RegionName := "candidate_vto_copy_46"
  let candidate_vto_copy_47_ptr : RegionName := "candidate_vto_copy_47"
  let candidate_vbmm_34_ptr : RegionName := "candidate_vbmm_34"
  let candidate_vmul_29_ptr : RegionName := "candidate_vmul_29"
  let candidate_vadd_14_ptr : RegionName := "candidate_vadd_14"
  let candidate_vsub_23_ptr : RegionName := "candidate_vsub_23"
  let candidate_vgt_8_ptr : RegionName := "candidate_vgt_8"
  let candidate_vmasked_fill_8_ptr : RegionName := "candidate_vmasked_fill_8"
  let candidate_vsoftmax_8_ptr : RegionName := "candidate_vsoftmax_8"
  let candidate_vto_copy_48_ptr : RegionName := "candidate_vto_copy_48"
  let candidate_vbmm_35_ptr : RegionName := "candidate_vbmm_35"
  let candidate_vcat_34_ptr : RegionName := "candidate_vcat_34"
  let candidate_vcat_35_ptr : RegionName := "candidate_vcat_35"
  let candidate_vmm_3_ptr : RegionName := "candidate_vmm_3"
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
    r64_row = tl.arange($(96))
    r64_col = tl.arange($(1024))
    r64_lane = tl.arange($(512))
    for r64_group in range($(1)) {
      r64_left = tl.load($(candidate_vcat_7_ptr) + ((((((r64_row[:, None] * $(512)) + r64_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r64_row[:, None] * $(512)) + r64_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r64_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r64_lane[:, None] * $(1024)) + r64_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r64_lane[:, None] * $(1024)) + r64_col[None, :]) % $(128)) * $(512)) + ((((r64_lane[:, None] * $(1024)) + r64_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r64_value = tl.dot(r64_left, r64_right)
      r64_value = r64_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_ptr) + ((r64_group * $(98304)) + (r64_row[:, None] * $(1024)) + r64_col[None, :]), r64_value)
    }
    r65_row = tl.arange($(96))
    r65_col = tl.arange($(1024))
    r65_lane = tl.arange($(512))
    for r65_group in range($(1)) {
      r65_left = tl.load($(candidate_vcat_7_ptr) + ((((((r65_row[:, None] * $(512)) + r65_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r65_row[:, None] * $(512)) + r65_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r65_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r65_lane[:, None] * $(1024)) + r65_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r65_lane[:, None] * $(1024)) + r65_col[None, :]) % $(128)) * $(512)) + ((((r65_lane[:, None] * $(1024)) + r65_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r65_value = tl.dot(r65_left, r65_right)
      r65_value = r65_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_1_ptr) + ((r65_group * $(98304)) + (r65_row[:, None] * $(1024)) + r65_col[None, :]), r65_value)
    }
    r66_z = tl.arange($(49152))
    r66_value = tl.where((r66_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r66_z / $(1536)) % $(32)) * $(1024)) + (((r66_z / $(192)) % $(8)) * $(128)) + ((r66_z % $(192)) - $(0))) / $(1024)) % $(288)) * $(1536)) + (((((((r66_z / $(1536)) % $(32)) * $(1024)) + (((r66_z / $(192)) % $(8)) * $(128)) + ((r66_z % $(192)) - $(0))) / $(128)) % $(8)) * $(192)) + (((((r66_z / $(1536)) % $(32)) * $(1024)) + (((r66_z / $(192)) % $(8)) * $(128)) + ((r66_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_5_ptr) + ((((((((r66_z / $(1536)) % $(32)) * $(512)) + (((r66_z / $(192)) % $(8)) * $(64)) + ((r66_z % $(192)) - $(128))) / $(64)) % $(8)) * $(18432)) + (((((((r66_z / $(1536)) % $(32)) * $(512)) + (((r66_z / $(192)) % $(8)) * $(64)) + ((r66_z % $(192)) - $(128))) / $(512)) % $(288)) * $(64)) + (((((r66_z / $(1536)) % $(32)) * $(512)) + (((r66_z / $(192)) % $(8)) * $(64)) + ((r66_z % $(192)) - $(128))) % $(64))), dtype=tl.bfloat16))
    r66_value = r66_value.to(tl.bfloat16)
    r66_value = tl.broadcast_to(r66_value, [$(49152)])
    tl.store($(candidate_vcat_8_ptr) + r66_z, r66_value)
    r67_z = tl.arange($(147456))
    r67_value = tl.where((r67_z % $(192)) < $(128), tl.load($(candidate_vbmm_ptr) + ((((((((r67_z / $(1536)) % $(96)) * $(1024)) + (((r67_z / $(192)) % $(8)) * $(128)) + ((r67_z % $(192)) - $(0))) / $(1024)) % $(96)) * $(1024)) + (((((((r67_z / $(1536)) % $(96)) * $(1024)) + (((r67_z / $(192)) % $(8)) * $(128)) + ((r67_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r67_z / $(1536)) % $(96)) * $(1024)) + (((r67_z / $(192)) % $(8)) * $(128)) + ((r67_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_7_ptr) + ((((r67_z / $(1536)) % $(96)) * $(576)) + ((((r67_z % $(192)) - $(128)) + $(512)) % $(576))), dtype=tl.bfloat16))
    r67_value = r67_value.to(tl.bfloat16)
    r67_value = tl.broadcast_to(r67_value, [$(147456)])
    tl.store($(candidate_vcat_9_ptr) + r67_z, r67_value)
    r68_z = tl.arange($(49152))
    r68_value = tl.load($(candidate_vcat_8_ptr) + ((((r68_z / $(192)) % $(32)) * $(1536)) + (((r68_z / $(6144)) % $(8)) * $(192)) + (r68_z % $(192))), dtype=tl.bfloat16)
    r68_value = r68_value.to(tl.float32)
    r68_value = tl.broadcast_to(r68_value, [$(49152)])
    tl.store($(candidate_vto_copy_22_ptr) + r68_z, r68_value)
    r69_z = tl.arange($(49152))
    r69_value = tl.load($(candidate_vcat_9_ptr) + ((((r69_z / $(192)) % $(32)) * $(1536)) + (((r69_z / $(6144)) % $(8)) * $(192)) + (r69_z % $(192))), dtype=tl.bfloat16)
    r69_value = r69_value.to(tl.float32)
    r69_value = tl.broadcast_to(r69_value, [$(49152)])
    tl.store($(candidate_vto_copy_23_ptr) + r69_z, r69_value)
    r70_row = tl.arange($(32))
    r70_col = tl.arange($(32))
    r70_lane = tl.arange($(192))
    for r70_group in range($(8)) {
      r70_left = tl.load($(candidate_vto_copy_22_ptr) + (((((((r70_group % $(8)) * $(6144)) + (r70_row[:, None] * $(192)) + r70_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r70_group % $(8)) * $(6144)) + (r70_row[:, None] * $(192)) + r70_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r70_group % $(8)) * $(6144)) + (r70_row[:, None] * $(192)) + r70_lane[None, :]) % $(192))), dtype=tl.float32)
      r70_right = tl.load($(candidate_vto_copy_23_ptr) + (((((((r70_group % $(8)) * $(6144)) + (r70_lane[:, None] * $(32)) + r70_col[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r70_group % $(8)) * $(6144)) + (r70_lane[:, None] * $(32)) + r70_col[None, :]) % $(32)) * $(192)) + (((((r70_group % $(8)) * $(6144)) + (r70_lane[:, None] * $(32)) + r70_col[None, :]) / $(32)) % $(192))), dtype=tl.float32)
      r70_value = tl.dot(r70_left, r70_right)
      r70_value = r70_value.to(tl.float32)
      tl.store($(candidate_vbmm_2_ptr) + ((r70_group * $(1024)) + (r70_row[:, None] * $(32)) + r70_col[None, :]), r70_value)
    }
    r71_z = tl.arange($(8192))
    r71_value = (tl.load($(candidate_vbmm_2_ptr) + ((((r71_z / $(1024)) % $(8)) * $(1024)) + (((r71_z / $(32)) % $(32)) * $(32)) + (r71_z % $(32))), dtype=tl.float32) * 0.07216878364870322)
    r71_value = r71_value.to(tl.float32)
    r71_value = tl.broadcast_to(r71_value, [$(8192)])
    tl.store($(candidate_vmul_21_ptr) + r71_z, r71_value)
    r72_z = tl.arange($(32))
    r72_value = (tl.cast(r72_z, tl.int64) + tl.full([], $(32), dtype=tl.int64))
    r72_value = tl.broadcast_to(r72_value, [$(32)])
    tl.store($(candidate_vadd_6_ptr) + r72_z, r72_value)
    r73_z = tl.arange($(32))
    r73_value = (tl.load($(candidate_vadd_6_ptr) + r73_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r73_value = tl.broadcast_to(r73_value, [$(32)])
    tl.store($(candidate_vsub_15_ptr) + r73_z, r73_value)
    r74_z = tl.arange($(1024))
    r74_value = (tl.cast((r74_z % $(32)), tl.int64) > tl.load($(candidate_vsub_15_ptr) + ((r74_z / $(32)) % $(32)), dtype=tl.int64))
    r74_value = tl.broadcast_to(r74_value, [$(1024)])
    tl.store($(candidate_vgt_ptr) + r74_z, r74_value)
    r75_z = tl.arange($(8192))
    r75_value = tl.where(tl.load($(candidate_vgt_ptr) + ((((r75_z / $(32)) % $(32)) * $(32)) + (r75_z % $(32))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_21_ptr) + ((((r75_z / $(1024)) % $(8)) * $(1024)) + (((r75_z / $(32)) % $(32)) * $(32)) + (r75_z % $(32))), dtype=tl.float32))
    r75_value = r75_value.to(tl.float32)
    r75_value = tl.broadcast_to(r75_value, [$(8192)])
    tl.store($(candidate_vmasked_fill_ptr) + r75_z, r75_value)
    r76_z = tl.arange($(8192))
    r76_lane = tl.arange($(32))
    r76_row_values = tl.load($(candidate_vmasked_fill_ptr) + ((((r76_z[:, None] / $(1024)) % $(8)) * $(1024)) + (((r76_z[:, None] / $(32)) % $(32)) * $(32)) + (r76_lane[None, :] % $(32))), dtype=tl.float32)
    r76_maximum = tl.max(r76_row_values, axis=1, keep_dims=true)
    r76_weights = tl.exp(r76_row_values - r76_maximum)
    r76_total = tl.sum(r76_weights, axis=1)
    r76_maximum = tl.max(r76_row_values, axis=1)
    r76_value = tl.exp(tl.load($(candidate_vmasked_fill_ptr) + ((((r76_z / $(1024)) % $(8)) * $(1024)) + (((r76_z / $(32)) % $(32)) * $(32)) + (r76_z % $(32))), dtype=tl.float32) - r76_maximum) / r76_total
    r76_value = r76_value.to(tl.float32)
    r76_value = tl.broadcast_to(r76_value, [$(8192)])
    tl.store($(candidate_vsoftmax_ptr) + r76_z, r76_value)
    r77_z = tl.arange($(8192))
    r77_value = tl.load($(candidate_vsoftmax_ptr) + ((((r77_z / $(1024)) % $(8)) * $(1024)) + (((r77_z / $(32)) % $(32)) * $(32)) + (r77_z % $(32))), dtype=tl.float32)
    r77_value = r77_value.to(tl.bfloat16)
    r77_value = tl.broadcast_to(r77_value, [$(8192)])
    tl.store($(candidate_vto_copy_24_ptr) + r77_z, r77_value)
    r78_row = tl.arange($(32))
    r78_col = tl.arange($(128))
    r78_lane = tl.arange($(32))
    for r78_group in range($(8)) {
      r78_left = tl.load($(candidate_vto_copy_24_ptr) + (((((((r78_group % $(8)) * $(1024)) + (r78_row[:, None] * $(32)) + r78_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + ((((((r78_group % $(8)) * $(1024)) + (r78_row[:, None] * $(32)) + r78_lane[None, :]) / $(32)) % $(32)) * $(32)) + ((((r78_group % $(8)) * $(1024)) + (r78_row[:, None] * $(32)) + r78_lane[None, :]) % $(32))), dtype=tl.bfloat16)
      r78_right = tl.load($(candidate_vbmm_1_ptr) + (((((((r78_group % $(8)) * $(4096)) + (r78_lane[:, None] * $(128)) + r78_col[None, :]) / $(128)) % $(32)) * $(1024)) + ((((((r78_group % $(8)) * $(4096)) + (r78_lane[:, None] * $(128)) + r78_col[None, :]) / $(4096)) % $(8)) * $(128)) + ((((r78_group % $(8)) * $(4096)) + (r78_lane[:, None] * $(128)) + r78_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r78_value = tl.dot(r78_left, r78_right)
      r78_value = r78_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_3_ptr) + ((r78_group * $(4096)) + (r78_row[:, None] * $(128)) + r78_col[None, :]), r78_value)
    }
    r79_z = tl.arange($(32768))
    r79_value = tl.load($(candidate_vbmm_3_ptr) + ((((((((r79_z / $(128)) % $(8)) * $(4096)) + ((((r79_z / $(1024)) % $(32)) - $(0)) * $(128)) + (r79_z % $(128))) / $(4096)) % $(8)) * $(4096)) + ((((((r79_z / $(128)) % $(8)) * $(32)) + (((r79_z / $(1024)) % $(32)) - $(0))) % $(32)) * $(128)) + (r79_z % $(128))), dtype=tl.bfloat16)
    r79_value = r79_value.to(tl.bfloat16)
    r79_value = tl.broadcast_to(r79_value, [$(32768)])
    tl.store($(candidate_vcat_10_ptr) + r79_z, r79_value)
    r80_row = tl.arange($(96))
    r80_col = tl.arange($(1024))
    r80_lane = tl.arange($(512))
    for r80_group in range($(1)) {
      r80_left = tl.load($(candidate_vcat_7_ptr) + ((((((r80_row[:, None] * $(512)) + r80_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r80_row[:, None] * $(512)) + r80_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r80_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r80_lane[:, None] * $(1024)) + r80_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r80_lane[:, None] * $(1024)) + r80_col[None, :]) % $(128)) * $(512)) + ((((r80_lane[:, None] * $(1024)) + r80_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r80_value = tl.dot(r80_left, r80_right)
      r80_value = r80_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_4_ptr) + ((r80_group * $(98304)) + (r80_row[:, None] * $(1024)) + r80_col[None, :]), r80_value)
    }
    r81_row = tl.arange($(96))
    r81_col = tl.arange($(1024))
    r81_lane = tl.arange($(512))
    for r81_group in range($(1)) {
      r81_left = tl.load($(candidate_vcat_7_ptr) + ((((((r81_row[:, None] * $(512)) + r81_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r81_row[:, None] * $(512)) + r81_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r81_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r81_lane[:, None] * $(1024)) + r81_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r81_lane[:, None] * $(1024)) + r81_col[None, :]) % $(128)) * $(512)) + ((((r81_lane[:, None] * $(1024)) + r81_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r81_value = tl.dot(r81_left, r81_right)
      r81_value = r81_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_5_ptr) + ((r81_group * $(98304)) + (r81_row[:, None] * $(1024)) + r81_col[None, :]), r81_value)
    }
    r82_z = tl.arange($(49152))
    r82_value = tl.where((r82_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r82_z / $(1536)) % $(32)) * $(1024)) + (((r82_z / $(192)) % $(8)) * $(128)) + ((r82_z % $(192)) - $(0)) + $(32768)) / $(1024)) % $(288)) * $(1536)) + (((((((r82_z / $(1536)) % $(32)) * $(1024)) + (((r82_z / $(192)) % $(8)) * $(128)) + ((r82_z % $(192)) - $(0)) + $(32768)) / $(128)) % $(8)) * $(192)) + (((((r82_z / $(1536)) % $(32)) * $(1024)) + (((r82_z / $(192)) % $(8)) * $(128)) + ((r82_z % $(192)) - $(0)) + $(32768)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_5_ptr) + ((((((((r82_z / $(1536)) % $(32)) * $(512)) + (((r82_z / $(192)) % $(8)) * $(64)) + ((r82_z % $(192)) - $(128)) + $(16384)) / $(64)) % $(8)) * $(18432)) + (((((((r82_z / $(1536)) % $(32)) * $(512)) + (((r82_z / $(192)) % $(8)) * $(64)) + ((r82_z % $(192)) - $(128)) + $(16384)) / $(512)) % $(288)) * $(64)) + (((((r82_z / $(1536)) % $(32)) * $(512)) + (((r82_z / $(192)) % $(8)) * $(64)) + ((r82_z % $(192)) - $(128)) + $(16384)) % $(64))), dtype=tl.bfloat16))
    r82_value = r82_value.to(tl.bfloat16)
    r82_value = tl.broadcast_to(r82_value, [$(49152)])
    tl.store($(candidate_vcat_11_ptr) + r82_z, r82_value)
    r83_z = tl.arange($(147456))
    r83_value = tl.where((r83_z % $(192)) < $(128), tl.load($(candidate_vbmm_4_ptr) + ((((((((r83_z / $(1536)) % $(96)) * $(1024)) + (((r83_z / $(192)) % $(8)) * $(128)) + ((r83_z % $(192)) - $(0))) / $(1024)) % $(96)) * $(1024)) + (((((((r83_z / $(1536)) % $(96)) * $(1024)) + (((r83_z / $(192)) % $(8)) * $(128)) + ((r83_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r83_z / $(1536)) % $(96)) * $(1024)) + (((r83_z / $(192)) % $(8)) * $(128)) + ((r83_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_7_ptr) + ((((r83_z / $(1536)) % $(96)) * $(576)) + ((((r83_z % $(192)) - $(128)) + $(512)) % $(576))), dtype=tl.bfloat16))
    r83_value = r83_value.to(tl.bfloat16)
    r83_value = tl.broadcast_to(r83_value, [$(147456)])
    tl.store($(candidate_vcat_12_ptr) + r83_z, r83_value)
    r84_z = tl.arange($(49152))
    r84_value = tl.load($(candidate_vcat_11_ptr) + ((((r84_z / $(192)) % $(32)) * $(1536)) + (((r84_z / $(6144)) % $(8)) * $(192)) + (r84_z % $(192))), dtype=tl.bfloat16)
    r84_value = r84_value.to(tl.float32)
    r84_value = tl.broadcast_to(r84_value, [$(49152)])
    tl.store($(candidate_vto_copy_25_ptr) + r84_z, r84_value)
    r85_z = tl.arange($(98304))
    r85_value = tl.load($(candidate_vcat_12_ptr) + ((((r85_z / $(192)) % $(64)) * $(1536)) + (((r85_z / $(12288)) % $(8)) * $(192)) + (r85_z % $(192))), dtype=tl.bfloat16)
    r85_value = r85_value.to(tl.float32)
    r85_value = tl.broadcast_to(r85_value, [$(98304)])
    tl.store($(candidate_vto_copy_26_ptr) + r85_z, r85_value)
    r86_row = tl.arange($(32))
    r86_col = tl.arange($(64))
    r86_lane = tl.arange($(192))
    for r86_group in range($(8)) {
      r86_left = tl.load($(candidate_vto_copy_25_ptr) + (((((((r86_group % $(8)) * $(6144)) + (r86_row[:, None] * $(192)) + r86_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r86_group % $(8)) * $(6144)) + (r86_row[:, None] * $(192)) + r86_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r86_group % $(8)) * $(6144)) + (r86_row[:, None] * $(192)) + r86_lane[None, :]) % $(192))), dtype=tl.float32)
      r86_right = tl.load($(candidate_vto_copy_26_ptr) + (((((((r86_group % $(8)) * $(12288)) + (r86_lane[:, None] * $(64)) + r86_col[None, :]) / $(12288)) % $(8)) * $(12288)) + (((((r86_group % $(8)) * $(12288)) + (r86_lane[:, None] * $(64)) + r86_col[None, :]) % $(64)) * $(192)) + (((((r86_group % $(8)) * $(12288)) + (r86_lane[:, None] * $(64)) + r86_col[None, :]) / $(64)) % $(192))), dtype=tl.float32)
      r86_value = tl.dot(r86_left, r86_right)
      r86_value = r86_value.to(tl.float32)
      tl.store($(candidate_vbmm_6_ptr) + ((r86_group * $(2048)) + (r86_row[:, None] * $(64)) + r86_col[None, :]), r86_value)
    }
    r87_z = tl.arange($(16384))
    r87_value = (tl.load($(candidate_vbmm_6_ptr) + ((((r87_z / $(2048)) % $(8)) * $(2048)) + (((r87_z / $(64)) % $(32)) * $(64)) + (r87_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r87_value = r87_value.to(tl.float32)
    r87_value = tl.broadcast_to(r87_value, [$(16384)])
    tl.store($(candidate_vmul_22_ptr) + r87_z, r87_value)
    r88_z = tl.arange($(32))
    r88_value = (tl.cast(r88_z, tl.int64) + tl.full([], $(64), dtype=tl.int64))
    r88_value = tl.broadcast_to(r88_value, [$(32)])
    tl.store($(candidate_vadd_7_ptr) + r88_z, r88_value)
    r89_z = tl.arange($(32))
    r89_value = (tl.load($(candidate_vadd_7_ptr) + r89_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r89_value = tl.broadcast_to(r89_value, [$(32)])
    tl.store($(candidate_vsub_16_ptr) + r89_z, r89_value)
    r90_z = tl.arange($(2048))
    r90_value = (tl.cast((r90_z % $(64)), tl.int64) > tl.load($(candidate_vsub_16_ptr) + ((r90_z / $(64)) % $(32)), dtype=tl.int64))
    r90_value = tl.broadcast_to(r90_value, [$(2048)])
    tl.store($(candidate_vgt_1_ptr) + r90_z, r90_value)
    r91_z = tl.arange($(16384))
    r91_value = tl.where(tl.load($(candidate_vgt_1_ptr) + ((((r91_z / $(64)) % $(32)) * $(64)) + (r91_z % $(64))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_22_ptr) + ((((r91_z / $(2048)) % $(8)) * $(2048)) + (((r91_z / $(64)) % $(32)) * $(64)) + (r91_z % $(64))), dtype=tl.float32))
    r91_value = r91_value.to(tl.float32)
    r91_value = tl.broadcast_to(r91_value, [$(16384)])
    tl.store($(candidate_vmasked_fill_1_ptr) + r91_z, r91_value)
    r92_z = tl.arange($(16384))
    r92_lane = tl.arange($(64))
    r92_row_values = tl.load($(candidate_vmasked_fill_1_ptr) + ((((r92_z[:, None] / $(2048)) % $(8)) * $(2048)) + (((r92_z[:, None] / $(64)) % $(32)) * $(64)) + (r92_lane[None, :] % $(64))), dtype=tl.float32)
    r92_maximum = tl.max(r92_row_values, axis=1, keep_dims=true)
    r92_weights = tl.exp(r92_row_values - r92_maximum)
    r92_total = tl.sum(r92_weights, axis=1)
    r92_maximum = tl.max(r92_row_values, axis=1)
    r92_value = tl.exp(tl.load($(candidate_vmasked_fill_1_ptr) + ((((r92_z / $(2048)) % $(8)) * $(2048)) + (((r92_z / $(64)) % $(32)) * $(64)) + (r92_z % $(64))), dtype=tl.float32) - r92_maximum) / r92_total
    r92_value = r92_value.to(tl.float32)
    r92_value = tl.broadcast_to(r92_value, [$(16384)])
    tl.store($(candidate_vsoftmax_1_ptr) + r92_z, r92_value)
    r93_z = tl.arange($(16384))
    r93_value = tl.load($(candidate_vsoftmax_1_ptr) + ((((r93_z / $(2048)) % $(8)) * $(2048)) + (((r93_z / $(64)) % $(32)) * $(64)) + (r93_z % $(64))), dtype=tl.float32)
    r93_value = r93_value.to(tl.bfloat16)
    r93_value = tl.broadcast_to(r93_value, [$(16384)])
    tl.store($(candidate_vto_copy_27_ptr) + r93_z, r93_value)
    r94_row = tl.arange($(32))
    r94_col = tl.arange($(128))
    r94_lane = tl.arange($(64))
    for r94_group in range($(8)) {
      r94_left = tl.load($(candidate_vto_copy_27_ptr) + (((((((r94_group % $(8)) * $(2048)) + (r94_row[:, None] * $(64)) + r94_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r94_group % $(8)) * $(2048)) + (r94_row[:, None] * $(64)) + r94_lane[None, :]) / $(64)) % $(32)) * $(64)) + ((((r94_group % $(8)) * $(2048)) + (r94_row[:, None] * $(64)) + r94_lane[None, :]) % $(64))), dtype=tl.bfloat16)
      r94_right = tl.load($(candidate_vbmm_5_ptr) + (((((((r94_group % $(8)) * $(8192)) + (r94_lane[:, None] * $(128)) + r94_col[None, :]) / $(128)) % $(64)) * $(1024)) + ((((((r94_group % $(8)) * $(8192)) + (r94_lane[:, None] * $(128)) + r94_col[None, :]) / $(8192)) % $(8)) * $(128)) + ((((r94_group % $(8)) * $(8192)) + (r94_lane[:, None] * $(128)) + r94_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r94_value = tl.dot(r94_left, r94_right)
      r94_value = r94_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_7_ptr) + ((r94_group * $(4096)) + (r94_row[:, None] * $(128)) + r94_col[None, :]), r94_value)
    }
    r95_z = tl.arange($(32768))
    r95_value = tl.load($(candidate_vbmm_7_ptr) + ((((((((r95_z / $(128)) % $(8)) * $(4096)) + ((((r95_z / $(1024)) % $(32)) - $(0)) * $(128)) + (r95_z % $(128))) / $(4096)) % $(8)) * $(4096)) + ((((((r95_z / $(128)) % $(8)) * $(32)) + (((r95_z / $(1024)) % $(32)) - $(0))) % $(32)) * $(128)) + (r95_z % $(128))), dtype=tl.bfloat16)
    r95_value = r95_value.to(tl.bfloat16)
    r95_value = tl.broadcast_to(r95_value, [$(32768)])
    tl.store($(candidate_vcat_13_ptr) + r95_z, r95_value)
    r96_row = tl.arange($(96))
    r96_col = tl.arange($(1024))
    r96_lane = tl.arange($(512))
    for r96_group in range($(1)) {
      r96_left = tl.load($(candidate_vcat_7_ptr) + ((((((r96_row[:, None] * $(512)) + r96_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r96_row[:, None] * $(512)) + r96_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r96_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r96_lane[:, None] * $(1024)) + r96_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r96_lane[:, None] * $(1024)) + r96_col[None, :]) % $(128)) * $(512)) + ((((r96_lane[:, None] * $(1024)) + r96_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r96_value = tl.dot(r96_left, r96_right)
      r96_value = r96_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_8_ptr) + ((r96_group * $(98304)) + (r96_row[:, None] * $(1024)) + r96_col[None, :]), r96_value)
    }
    r97_row = tl.arange($(96))
    r97_col = tl.arange($(1024))
    r97_lane = tl.arange($(512))
    for r97_group in range($(1)) {
      r97_left = tl.load($(candidate_vcat_7_ptr) + ((((((r97_row[:, None] * $(512)) + r97_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r97_row[:, None] * $(512)) + r97_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r97_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r97_lane[:, None] * $(1024)) + r97_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r97_lane[:, None] * $(1024)) + r97_col[None, :]) % $(128)) * $(512)) + ((((r97_lane[:, None] * $(1024)) + r97_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r97_value = tl.dot(r97_left, r97_right)
      r97_value = r97_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_9_ptr) + ((r97_group * $(98304)) + (r97_row[:, None] * $(1024)) + r97_col[None, :]), r97_value)
    }
    r98_z = tl.arange($(49152))
    r98_value = tl.where((r98_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r98_z / $(1536)) % $(32)) * $(1024)) + (((r98_z / $(192)) % $(8)) * $(128)) + ((r98_z % $(192)) - $(0)) + $(65536)) / $(1024)) % $(288)) * $(1536)) + (((((((r98_z / $(1536)) % $(32)) * $(1024)) + (((r98_z / $(192)) % $(8)) * $(128)) + ((r98_z % $(192)) - $(0)) + $(65536)) / $(128)) % $(8)) * $(192)) + (((((r98_z / $(1536)) % $(32)) * $(1024)) + (((r98_z / $(192)) % $(8)) * $(128)) + ((r98_z % $(192)) - $(0)) + $(65536)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_5_ptr) + ((((((((r98_z / $(1536)) % $(32)) * $(512)) + (((r98_z / $(192)) % $(8)) * $(64)) + ((r98_z % $(192)) - $(128)) + $(32768)) / $(64)) % $(8)) * $(18432)) + (((((((r98_z / $(1536)) % $(32)) * $(512)) + (((r98_z / $(192)) % $(8)) * $(64)) + ((r98_z % $(192)) - $(128)) + $(32768)) / $(512)) % $(288)) * $(64)) + (((((r98_z / $(1536)) % $(32)) * $(512)) + (((r98_z / $(192)) % $(8)) * $(64)) + ((r98_z % $(192)) - $(128)) + $(32768)) % $(64))), dtype=tl.bfloat16))
    r98_value = r98_value.to(tl.bfloat16)
    r98_value = tl.broadcast_to(r98_value, [$(49152)])
    tl.store($(candidate_vcat_14_ptr) + r98_z, r98_value)
    r99_z = tl.arange($(147456))
    r99_value = tl.where((r99_z % $(192)) < $(128), tl.load($(candidate_vbmm_8_ptr) + ((((((((r99_z / $(1536)) % $(96)) * $(1024)) + (((r99_z / $(192)) % $(8)) * $(128)) + ((r99_z % $(192)) - $(0))) / $(1024)) % $(96)) * $(1024)) + (((((((r99_z / $(1536)) % $(96)) * $(1024)) + (((r99_z / $(192)) % $(8)) * $(128)) + ((r99_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r99_z / $(1536)) % $(96)) * $(1024)) + (((r99_z / $(192)) % $(8)) * $(128)) + ((r99_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_7_ptr) + ((((r99_z / $(1536)) % $(96)) * $(576)) + ((((r99_z % $(192)) - $(128)) + $(512)) % $(576))), dtype=tl.bfloat16))
    r99_value = r99_value.to(tl.bfloat16)
    r99_value = tl.broadcast_to(r99_value, [$(147456)])
    tl.store($(candidate_vcat_15_ptr) + r99_z, r99_value)
    r100_z = tl.arange($(49152))
    r100_value = tl.load($(candidate_vcat_14_ptr) + ((((r100_z / $(192)) % $(32)) * $(1536)) + (((r100_z / $(6144)) % $(8)) * $(192)) + (r100_z % $(192))), dtype=tl.bfloat16)
    r100_value = r100_value.to(tl.float32)
    r100_value = tl.broadcast_to(r100_value, [$(49152)])
    tl.store($(candidate_vto_copy_28_ptr) + r100_z, r100_value)
    r101_z = tl.arange($(147456))
    r101_value = tl.load($(candidate_vcat_15_ptr) + ((((r101_z / $(192)) % $(96)) * $(1536)) + (((r101_z / $(18432)) % $(8)) * $(192)) + (r101_z % $(192))), dtype=tl.bfloat16)
    r101_value = r101_value.to(tl.float32)
    r101_value = tl.broadcast_to(r101_value, [$(147456)])
    tl.store($(candidate_vto_copy_29_ptr) + r101_z, r101_value)
    r102_row = tl.arange($(32))
    r102_col = tl.arange($(96))
    r102_lane = tl.arange($(192))
    for r102_group in range($(8)) {
      r102_left = tl.load($(candidate_vto_copy_28_ptr) + (((((((r102_group % $(8)) * $(6144)) + (r102_row[:, None] * $(192)) + r102_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r102_group % $(8)) * $(6144)) + (r102_row[:, None] * $(192)) + r102_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r102_group % $(8)) * $(6144)) + (r102_row[:, None] * $(192)) + r102_lane[None, :]) % $(192))), dtype=tl.float32)
      r102_right = tl.load($(candidate_vto_copy_29_ptr) + (((((((r102_group % $(8)) * $(18432)) + (r102_lane[:, None] * $(96)) + r102_col[None, :]) / $(18432)) % $(8)) * $(18432)) + (((((r102_group % $(8)) * $(18432)) + (r102_lane[:, None] * $(96)) + r102_col[None, :]) % $(96)) * $(192)) + (((((r102_group % $(8)) * $(18432)) + (r102_lane[:, None] * $(96)) + r102_col[None, :]) / $(96)) % $(192))), dtype=tl.float32)
      r102_value = tl.dot(r102_left, r102_right)
      r102_value = r102_value.to(tl.float32)
      tl.store($(candidate_vbmm_10_ptr) + ((r102_group * $(3072)) + (r102_row[:, None] * $(96)) + r102_col[None, :]), r102_value)
    }
    r103_z = tl.arange($(24576))
    r103_value = (tl.load($(candidate_vbmm_10_ptr) + ((((r103_z / $(3072)) % $(8)) * $(3072)) + (((r103_z / $(96)) % $(32)) * $(96)) + (r103_z % $(96))), dtype=tl.float32) * 0.07216878364870322)
    r103_value = r103_value.to(tl.float32)
    r103_value = tl.broadcast_to(r103_value, [$(24576)])
    tl.store($(candidate_vmul_23_ptr) + r103_z, r103_value)
    r104_z = tl.arange($(32))
    r104_value = (tl.cast(r104_z, tl.int64) + tl.full([], $(96), dtype=tl.int64))
    r104_value = tl.broadcast_to(r104_value, [$(32)])
    tl.store($(candidate_vadd_8_ptr) + r104_z, r104_value)
    r105_z = tl.arange($(32))
    r105_value = (tl.load($(candidate_vadd_8_ptr) + r105_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r105_value = tl.broadcast_to(r105_value, [$(32)])
    tl.store($(candidate_vsub_17_ptr) + r105_z, r105_value)
    r106_z = tl.arange($(3072))
    r106_value = (tl.cast((r106_z % $(96)), tl.int64) > tl.load($(candidate_vsub_17_ptr) + ((r106_z / $(96)) % $(32)), dtype=tl.int64))
    r106_value = tl.broadcast_to(r106_value, [$(3072)])
    tl.store($(candidate_vgt_2_ptr) + r106_z, r106_value)
    r107_z = tl.arange($(24576))
    r107_value = tl.where(tl.load($(candidate_vgt_2_ptr) + ((((r107_z / $(96)) % $(32)) * $(96)) + (r107_z % $(96))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_23_ptr) + ((((r107_z / $(3072)) % $(8)) * $(3072)) + (((r107_z / $(96)) % $(32)) * $(96)) + (r107_z % $(96))), dtype=tl.float32))
    r107_value = r107_value.to(tl.float32)
    r107_value = tl.broadcast_to(r107_value, [$(24576)])
    tl.store($(candidate_vmasked_fill_2_ptr) + r107_z, r107_value)
    r108_z = tl.arange($(24576))
    r108_lane = tl.arange($(96))
    r108_row_values = tl.load($(candidate_vmasked_fill_2_ptr) + ((((r108_z[:, None] / $(3072)) % $(8)) * $(3072)) + (((r108_z[:, None] / $(96)) % $(32)) * $(96)) + (r108_lane[None, :] % $(96))), dtype=tl.float32)
    r108_maximum = tl.max(r108_row_values, axis=1, keep_dims=true)
    r108_weights = tl.exp(r108_row_values - r108_maximum)
    r108_total = tl.sum(r108_weights, axis=1)
    r108_maximum = tl.max(r108_row_values, axis=1)
    r108_value = tl.exp(tl.load($(candidate_vmasked_fill_2_ptr) + ((((r108_z / $(3072)) % $(8)) * $(3072)) + (((r108_z / $(96)) % $(32)) * $(96)) + (r108_z % $(96))), dtype=tl.float32) - r108_maximum) / r108_total
    r108_value = r108_value.to(tl.float32)
    r108_value = tl.broadcast_to(r108_value, [$(24576)])
    tl.store($(candidate_vsoftmax_2_ptr) + r108_z, r108_value)
    r109_z = tl.arange($(24576))
    r109_value = tl.load($(candidate_vsoftmax_2_ptr) + ((((r109_z / $(3072)) % $(8)) * $(3072)) + (((r109_z / $(96)) % $(32)) * $(96)) + (r109_z % $(96))), dtype=tl.float32)
    r109_value = r109_value.to(tl.bfloat16)
    r109_value = tl.broadcast_to(r109_value, [$(24576)])
    tl.store($(candidate_vto_copy_30_ptr) + r109_z, r109_value)
    r110_row = tl.arange($(32))
    r110_col = tl.arange($(128))
    r110_lane = tl.arange($(96))
    for r110_group in range($(8)) {
      r110_left = tl.load($(candidate_vto_copy_30_ptr) + (((((((r110_group % $(8)) * $(3072)) + (r110_row[:, None] * $(96)) + r110_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r110_group % $(8)) * $(3072)) + (r110_row[:, None] * $(96)) + r110_lane[None, :]) / $(96)) % $(32)) * $(96)) + ((((r110_group % $(8)) * $(3072)) + (r110_row[:, None] * $(96)) + r110_lane[None, :]) % $(96))), dtype=tl.bfloat16)
      r110_right = tl.load($(candidate_vbmm_9_ptr) + (((((((r110_group % $(8)) * $(12288)) + (r110_lane[:, None] * $(128)) + r110_col[None, :]) / $(128)) % $(96)) * $(1024)) + ((((((r110_group % $(8)) * $(12288)) + (r110_lane[:, None] * $(128)) + r110_col[None, :]) / $(12288)) % $(8)) * $(128)) + ((((r110_group % $(8)) * $(12288)) + (r110_lane[:, None] * $(128)) + r110_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r110_value = tl.dot(r110_left, r110_right)
      r110_value = r110_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_11_ptr) + ((r110_group * $(4096)) + (r110_row[:, None] * $(128)) + r110_col[None, :]), r110_value)
    }
    r111_z = tl.arange($(32768))
    r111_value = tl.load($(candidate_vbmm_11_ptr) + ((((((((r111_z / $(128)) % $(8)) * $(4096)) + ((((r111_z / $(1024)) % $(32)) - $(0)) * $(128)) + (r111_z % $(128))) / $(4096)) % $(8)) * $(4096)) + ((((((r111_z / $(128)) % $(8)) * $(32)) + (((r111_z / $(1024)) % $(32)) - $(0))) % $(32)) * $(128)) + (r111_z % $(128))), dtype=tl.bfloat16)
    r111_value = r111_value.to(tl.bfloat16)
    r111_value = tl.broadcast_to(r111_value, [$(32768)])
    tl.store($(candidate_vcat_16_ptr) + r111_z, r111_value)
    r112_row = tl.arange($(96))
    r112_col = tl.arange($(1024))
    r112_lane = tl.arange($(512))
    for r112_group in range($(1)) {
      r112_left = tl.load($(candidate_vcat_7_ptr) + ((((((r112_row[:, None] * $(512)) + r112_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r112_row[:, None] * $(512)) + r112_lane[None, :]) % $(512)) + $(55296)), dtype=tl.bfloat16)
      r112_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r112_lane[:, None] * $(1024)) + r112_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r112_lane[:, None] * $(1024)) + r112_col[None, :]) % $(128)) * $(512)) + ((((r112_lane[:, None] * $(1024)) + r112_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r112_value = tl.dot(r112_left, r112_right)
      r112_value = r112_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_12_ptr) + ((r112_group * $(98304)) + (r112_row[:, None] * $(1024)) + r112_col[None, :]), r112_value)
    }
    r113_row = tl.arange($(96))
    r113_col = tl.arange($(1024))
    r113_lane = tl.arange($(512))
    for r113_group in range($(1)) {
      r113_left = tl.load($(candidate_vcat_7_ptr) + ((((((r113_row[:, None] * $(512)) + r113_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r113_row[:, None] * $(512)) + r113_lane[None, :]) % $(512)) + $(55296)), dtype=tl.bfloat16)
      r113_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r113_lane[:, None] * $(1024)) + r113_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r113_lane[:, None] * $(1024)) + r113_col[None, :]) % $(128)) * $(512)) + ((((r113_lane[:, None] * $(1024)) + r113_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r113_value = tl.dot(r113_left, r113_right)
      r113_value = r113_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_13_ptr) + ((r113_group * $(98304)) + (r113_row[:, None] * $(1024)) + r113_col[None, :]), r113_value)
    }
    r114_z = tl.arange($(49152))
    r114_value = tl.where((r114_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r114_z / $(1536)) % $(32)) * $(1024)) + (((r114_z / $(192)) % $(8)) * $(128)) + ((r114_z % $(192)) - $(0)) + $(98304)) / $(1024)) % $(288)) * $(1536)) + (((((((r114_z / $(1536)) % $(32)) * $(1024)) + (((r114_z / $(192)) % $(8)) * $(128)) + ((r114_z % $(192)) - $(0)) + $(98304)) / $(128)) % $(8)) * $(192)) + (((((r114_z / $(1536)) % $(32)) * $(1024)) + (((r114_z / $(192)) % $(8)) * $(128)) + ((r114_z % $(192)) - $(0)) + $(98304)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_5_ptr) + ((((((((r114_z / $(1536)) % $(32)) * $(512)) + (((r114_z / $(192)) % $(8)) * $(64)) + ((r114_z % $(192)) - $(128)) + $(49152)) / $(64)) % $(8)) * $(18432)) + (((((((r114_z / $(1536)) % $(32)) * $(512)) + (((r114_z / $(192)) % $(8)) * $(64)) + ((r114_z % $(192)) - $(128)) + $(49152)) / $(512)) % $(288)) * $(64)) + (((((r114_z / $(1536)) % $(32)) * $(512)) + (((r114_z / $(192)) % $(8)) * $(64)) + ((r114_z % $(192)) - $(128)) + $(49152)) % $(64))), dtype=tl.bfloat16))
    r114_value = r114_value.to(tl.bfloat16)
    r114_value = tl.broadcast_to(r114_value, [$(49152)])
    tl.store($(candidate_vcat_17_ptr) + r114_z, r114_value)
    r115_z = tl.arange($(147456))
    r115_value = tl.where((r115_z % $(192)) < $(128), tl.load($(candidate_vbmm_12_ptr) + ((((((((r115_z / $(1536)) % $(96)) * $(1024)) + (((r115_z / $(192)) % $(8)) * $(128)) + ((r115_z % $(192)) - $(0))) / $(1024)) % $(96)) * $(1024)) + (((((((r115_z / $(1536)) % $(96)) * $(1024)) + (((r115_z / $(192)) % $(8)) * $(128)) + ((r115_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r115_z / $(1536)) % $(96)) * $(1024)) + (((r115_z / $(192)) % $(8)) * $(128)) + ((r115_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_7_ptr) + ((((r115_z / $(1536)) % $(96)) * $(576)) + ((((r115_z % $(192)) - $(128)) + $(512)) % $(576)) + $(55296)), dtype=tl.bfloat16))
    r115_value = r115_value.to(tl.bfloat16)
    r115_value = tl.broadcast_to(r115_value, [$(147456)])
    tl.store($(candidate_vcat_18_ptr) + r115_z, r115_value)
    r116_z = tl.arange($(49152))
    r116_value = tl.load($(candidate_vcat_17_ptr) + ((((r116_z / $(192)) % $(32)) * $(1536)) + (((r116_z / $(6144)) % $(8)) * $(192)) + (r116_z % $(192))), dtype=tl.bfloat16)
    r116_value = r116_value.to(tl.float32)
    r116_value = tl.broadcast_to(r116_value, [$(49152)])
    tl.store($(candidate_vto_copy_31_ptr) + r116_z, r116_value)
    r117_z = tl.arange($(49152))
    r117_value = tl.load($(candidate_vcat_18_ptr) + ((((r117_z / $(192)) % $(32)) * $(1536)) + (((r117_z / $(6144)) % $(8)) * $(192)) + (r117_z % $(192))), dtype=tl.bfloat16)
    r117_value = r117_value.to(tl.float32)
    r117_value = tl.broadcast_to(r117_value, [$(49152)])
    tl.store($(candidate_vto_copy_32_ptr) + r117_z, r117_value)
    r118_row = tl.arange($(32))
    r118_col = tl.arange($(32))
    r118_lane = tl.arange($(192))
    for r118_group in range($(8)) {
      r118_left = tl.load($(candidate_vto_copy_31_ptr) + (((((((r118_group % $(8)) * $(6144)) + (r118_row[:, None] * $(192)) + r118_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r118_group % $(8)) * $(6144)) + (r118_row[:, None] * $(192)) + r118_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r118_group % $(8)) * $(6144)) + (r118_row[:, None] * $(192)) + r118_lane[None, :]) % $(192))), dtype=tl.float32)
      r118_right = tl.load($(candidate_vto_copy_32_ptr) + (((((((r118_group % $(8)) * $(6144)) + (r118_lane[:, None] * $(32)) + r118_col[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r118_group % $(8)) * $(6144)) + (r118_lane[:, None] * $(32)) + r118_col[None, :]) % $(32)) * $(192)) + (((((r118_group % $(8)) * $(6144)) + (r118_lane[:, None] * $(32)) + r118_col[None, :]) / $(32)) % $(192))), dtype=tl.float32)
      r118_value = tl.dot(r118_left, r118_right)
      r118_value = r118_value.to(tl.float32)
      tl.store($(candidate_vbmm_14_ptr) + ((r118_group * $(1024)) + (r118_row[:, None] * $(32)) + r118_col[None, :]), r118_value)
    }
    r119_z = tl.arange($(8192))
    r119_value = (tl.load($(candidate_vbmm_14_ptr) + ((((r119_z / $(1024)) % $(8)) * $(1024)) + (((r119_z / $(32)) % $(32)) * $(32)) + (r119_z % $(32))), dtype=tl.float32) * 0.07216878364870322)
    r119_value = r119_value.to(tl.float32)
    r119_value = tl.broadcast_to(r119_value, [$(8192)])
    tl.store($(candidate_vmul_24_ptr) + r119_z, r119_value)
    r120_z = tl.arange($(32))
    r120_value = (tl.cast(r120_z, tl.int64) + tl.full([], $(32), dtype=tl.int64))
    r120_value = tl.broadcast_to(r120_value, [$(32)])
    tl.store($(candidate_vadd_9_ptr) + r120_z, r120_value)
    r121_z = tl.arange($(32))
    r121_value = (tl.load($(candidate_vadd_9_ptr) + r121_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r121_value = tl.broadcast_to(r121_value, [$(32)])
    tl.store($(candidate_vsub_18_ptr) + r121_z, r121_value)
    r122_z = tl.arange($(1024))
    r122_value = (tl.cast((r122_z % $(32)), tl.int64) > tl.load($(candidate_vsub_18_ptr) + ((r122_z / $(32)) % $(32)), dtype=tl.int64))
    r122_value = tl.broadcast_to(r122_value, [$(1024)])
    tl.store($(candidate_vgt_3_ptr) + r122_z, r122_value)
    r123_z = tl.arange($(8192))
    r123_value = tl.where(tl.load($(candidate_vgt_3_ptr) + ((((r123_z / $(32)) % $(32)) * $(32)) + (r123_z % $(32))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_24_ptr) + ((((r123_z / $(1024)) % $(8)) * $(1024)) + (((r123_z / $(32)) % $(32)) * $(32)) + (r123_z % $(32))), dtype=tl.float32))
    r123_value = r123_value.to(tl.float32)
    r123_value = tl.broadcast_to(r123_value, [$(8192)])
    tl.store($(candidate_vmasked_fill_3_ptr) + r123_z, r123_value)
    r124_z = tl.arange($(8192))
    r124_lane = tl.arange($(32))
    r124_row_values = tl.load($(candidate_vmasked_fill_3_ptr) + ((((r124_z[:, None] / $(1024)) % $(8)) * $(1024)) + (((r124_z[:, None] / $(32)) % $(32)) * $(32)) + (r124_lane[None, :] % $(32))), dtype=tl.float32)
    r124_maximum = tl.max(r124_row_values, axis=1, keep_dims=true)
    r124_weights = tl.exp(r124_row_values - r124_maximum)
    r124_total = tl.sum(r124_weights, axis=1)
    r124_maximum = tl.max(r124_row_values, axis=1)
    r124_value = tl.exp(tl.load($(candidate_vmasked_fill_3_ptr) + ((((r124_z / $(1024)) % $(8)) * $(1024)) + (((r124_z / $(32)) % $(32)) * $(32)) + (r124_z % $(32))), dtype=tl.float32) - r124_maximum) / r124_total
    r124_value = r124_value.to(tl.float32)
    r124_value = tl.broadcast_to(r124_value, [$(8192)])
    tl.store($(candidate_vsoftmax_3_ptr) + r124_z, r124_value)
    r125_z = tl.arange($(8192))
    r125_value = tl.load($(candidate_vsoftmax_3_ptr) + ((((r125_z / $(1024)) % $(8)) * $(1024)) + (((r125_z / $(32)) % $(32)) * $(32)) + (r125_z % $(32))), dtype=tl.float32)
    r125_value = r125_value.to(tl.bfloat16)
    r125_value = tl.broadcast_to(r125_value, [$(8192)])
    tl.store($(candidate_vto_copy_33_ptr) + r125_z, r125_value)
    r126_row = tl.arange($(32))
    r126_col = tl.arange($(128))
    r126_lane = tl.arange($(32))
    for r126_group in range($(8)) {
      r126_left = tl.load($(candidate_vto_copy_33_ptr) + (((((((r126_group % $(8)) * $(1024)) + (r126_row[:, None] * $(32)) + r126_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + ((((((r126_group % $(8)) * $(1024)) + (r126_row[:, None] * $(32)) + r126_lane[None, :]) / $(32)) % $(32)) * $(32)) + ((((r126_group % $(8)) * $(1024)) + (r126_row[:, None] * $(32)) + r126_lane[None, :]) % $(32))), dtype=tl.bfloat16)
      r126_right = tl.load($(candidate_vbmm_13_ptr) + (((((((r126_group % $(8)) * $(4096)) + (r126_lane[:, None] * $(128)) + r126_col[None, :]) / $(128)) % $(32)) * $(1024)) + ((((((r126_group % $(8)) * $(4096)) + (r126_lane[:, None] * $(128)) + r126_col[None, :]) / $(4096)) % $(8)) * $(128)) + ((((r126_group % $(8)) * $(4096)) + (r126_lane[:, None] * $(128)) + r126_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r126_value = tl.dot(r126_left, r126_right)
      r126_value = r126_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_15_ptr) + ((r126_group * $(4096)) + (r126_row[:, None] * $(128)) + r126_col[None, :]), r126_value)
    }
    r127_z = tl.arange($(32768))
    r127_value = tl.load($(candidate_vbmm_15_ptr) + ((((((((r127_z / $(128)) % $(8)) * $(4096)) + ((((r127_z / $(1024)) % $(32)) - $(0)) * $(128)) + (r127_z % $(128))) / $(4096)) % $(8)) * $(4096)) + ((((((r127_z / $(128)) % $(8)) * $(32)) + (((r127_z / $(1024)) % $(32)) - $(0))) % $(32)) * $(128)) + (r127_z % $(128))), dtype=tl.bfloat16)
    r127_value = r127_value.to(tl.bfloat16)
    r127_value = tl.broadcast_to(r127_value, [$(32768)])
    tl.store($(candidate_vcat_19_ptr) + r127_z, r127_value)
    r128_row = tl.arange($(96))
    r128_col = tl.arange($(1024))
    r128_lane = tl.arange($(512))
    for r128_group in range($(1)) {
      r128_left = tl.load($(candidate_vcat_7_ptr) + ((((((r128_row[:, None] * $(512)) + r128_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r128_row[:, None] * $(512)) + r128_lane[None, :]) % $(512)) + $(55296)), dtype=tl.bfloat16)
      r128_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r128_lane[:, None] * $(1024)) + r128_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r128_lane[:, None] * $(1024)) + r128_col[None, :]) % $(128)) * $(512)) + ((((r128_lane[:, None] * $(1024)) + r128_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r128_value = tl.dot(r128_left, r128_right)
      r128_value = r128_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_16_ptr) + ((r128_group * $(98304)) + (r128_row[:, None] * $(1024)) + r128_col[None, :]), r128_value)
    }
    r129_row = tl.arange($(96))
    r129_col = tl.arange($(1024))
    r129_lane = tl.arange($(512))
    for r129_group in range($(1)) {
      r129_left = tl.load($(candidate_vcat_7_ptr) + ((((((r129_row[:, None] * $(512)) + r129_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r129_row[:, None] * $(512)) + r129_lane[None, :]) % $(512)) + $(55296)), dtype=tl.bfloat16)
      r129_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r129_lane[:, None] * $(1024)) + r129_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r129_lane[:, None] * $(1024)) + r129_col[None, :]) % $(128)) * $(512)) + ((((r129_lane[:, None] * $(1024)) + r129_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r129_value = tl.dot(r129_left, r129_right)
      r129_value = r129_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_17_ptr) + ((r129_group * $(98304)) + (r129_row[:, None] * $(1024)) + r129_col[None, :]), r129_value)
    }
    r130_z = tl.arange($(49152))
    r130_value = tl.where((r130_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r130_z / $(1536)) % $(32)) * $(1024)) + (((r130_z / $(192)) % $(8)) * $(128)) + ((r130_z % $(192)) - $(0)) + $(131072)) / $(1024)) % $(288)) * $(1536)) + (((((((r130_z / $(1536)) % $(32)) * $(1024)) + (((r130_z / $(192)) % $(8)) * $(128)) + ((r130_z % $(192)) - $(0)) + $(131072)) / $(128)) % $(8)) * $(192)) + (((((r130_z / $(1536)) % $(32)) * $(1024)) + (((r130_z / $(192)) % $(8)) * $(128)) + ((r130_z % $(192)) - $(0)) + $(131072)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_5_ptr) + ((((((((r130_z / $(1536)) % $(32)) * $(512)) + (((r130_z / $(192)) % $(8)) * $(64)) + ((r130_z % $(192)) - $(128)) + $(65536)) / $(64)) % $(8)) * $(18432)) + (((((((r130_z / $(1536)) % $(32)) * $(512)) + (((r130_z / $(192)) % $(8)) * $(64)) + ((r130_z % $(192)) - $(128)) + $(65536)) / $(512)) % $(288)) * $(64)) + (((((r130_z / $(1536)) % $(32)) * $(512)) + (((r130_z / $(192)) % $(8)) * $(64)) + ((r130_z % $(192)) - $(128)) + $(65536)) % $(64))), dtype=tl.bfloat16))
    r130_value = r130_value.to(tl.bfloat16)
    r130_value = tl.broadcast_to(r130_value, [$(49152)])
    tl.store($(candidate_vcat_20_ptr) + r130_z, r130_value)
    r131_z = tl.arange($(147456))
    r131_value = tl.where((r131_z % $(192)) < $(128), tl.load($(candidate_vbmm_16_ptr) + ((((((((r131_z / $(1536)) % $(96)) * $(1024)) + (((r131_z / $(192)) % $(8)) * $(128)) + ((r131_z % $(192)) - $(0))) / $(1024)) % $(96)) * $(1024)) + (((((((r131_z / $(1536)) % $(96)) * $(1024)) + (((r131_z / $(192)) % $(8)) * $(128)) + ((r131_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r131_z / $(1536)) % $(96)) * $(1024)) + (((r131_z / $(192)) % $(8)) * $(128)) + ((r131_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_7_ptr) + ((((r131_z / $(1536)) % $(96)) * $(576)) + ((((r131_z % $(192)) - $(128)) + $(512)) % $(576)) + $(55296)), dtype=tl.bfloat16))
    r131_value = r131_value.to(tl.bfloat16)
    r131_value = tl.broadcast_to(r131_value, [$(147456)])
    tl.store($(candidate_vcat_21_ptr) + r131_z, r131_value)
    r132_z = tl.arange($(49152))
    r132_value = tl.load($(candidate_vcat_20_ptr) + ((((r132_z / $(192)) % $(32)) * $(1536)) + (((r132_z / $(6144)) % $(8)) * $(192)) + (r132_z % $(192))), dtype=tl.bfloat16)
    r132_value = r132_value.to(tl.float32)
    r132_value = tl.broadcast_to(r132_value, [$(49152)])
    tl.store($(candidate_vto_copy_34_ptr) + r132_z, r132_value)
    r133_z = tl.arange($(98304))
    r133_value = tl.load($(candidate_vcat_21_ptr) + ((((r133_z / $(192)) % $(64)) * $(1536)) + (((r133_z / $(12288)) % $(8)) * $(192)) + (r133_z % $(192))), dtype=tl.bfloat16)
    r133_value = r133_value.to(tl.float32)
    r133_value = tl.broadcast_to(r133_value, [$(98304)])
    tl.store($(candidate_vto_copy_35_ptr) + r133_z, r133_value)
    r134_row = tl.arange($(32))
    r134_col = tl.arange($(64))
    r134_lane = tl.arange($(192))
    for r134_group in range($(8)) {
      r134_left = tl.load($(candidate_vto_copy_34_ptr) + (((((((r134_group % $(8)) * $(6144)) + (r134_row[:, None] * $(192)) + r134_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r134_group % $(8)) * $(6144)) + (r134_row[:, None] * $(192)) + r134_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r134_group % $(8)) * $(6144)) + (r134_row[:, None] * $(192)) + r134_lane[None, :]) % $(192))), dtype=tl.float32)
      r134_right = tl.load($(candidate_vto_copy_35_ptr) + (((((((r134_group % $(8)) * $(12288)) + (r134_lane[:, None] * $(64)) + r134_col[None, :]) / $(12288)) % $(8)) * $(12288)) + (((((r134_group % $(8)) * $(12288)) + (r134_lane[:, None] * $(64)) + r134_col[None, :]) % $(64)) * $(192)) + (((((r134_group % $(8)) * $(12288)) + (r134_lane[:, None] * $(64)) + r134_col[None, :]) / $(64)) % $(192))), dtype=tl.float32)
      r134_value = tl.dot(r134_left, r134_right)
      r134_value = r134_value.to(tl.float32)
      tl.store($(candidate_vbmm_18_ptr) + ((r134_group * $(2048)) + (r134_row[:, None] * $(64)) + r134_col[None, :]), r134_value)
    }
    r135_z = tl.arange($(16384))
    r135_value = (tl.load($(candidate_vbmm_18_ptr) + ((((r135_z / $(2048)) % $(8)) * $(2048)) + (((r135_z / $(64)) % $(32)) * $(64)) + (r135_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r135_value = r135_value.to(tl.float32)
    r135_value = tl.broadcast_to(r135_value, [$(16384)])
    tl.store($(candidate_vmul_25_ptr) + r135_z, r135_value)
    r136_z = tl.arange($(32))
    r136_value = (tl.cast(r136_z, tl.int64) + tl.full([], $(64), dtype=tl.int64))
    r136_value = tl.broadcast_to(r136_value, [$(32)])
    tl.store($(candidate_vadd_10_ptr) + r136_z, r136_value)
    r137_z = tl.arange($(32))
    r137_value = (tl.load($(candidate_vadd_10_ptr) + r137_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r137_value = tl.broadcast_to(r137_value, [$(32)])
    tl.store($(candidate_vsub_19_ptr) + r137_z, r137_value)
    r138_z = tl.arange($(2048))
    r138_value = (tl.cast((r138_z % $(64)), tl.int64) > tl.load($(candidate_vsub_19_ptr) + ((r138_z / $(64)) % $(32)), dtype=tl.int64))
    r138_value = tl.broadcast_to(r138_value, [$(2048)])
    tl.store($(candidate_vgt_4_ptr) + r138_z, r138_value)
    r139_z = tl.arange($(16384))
    r139_value = tl.where(tl.load($(candidate_vgt_4_ptr) + ((((r139_z / $(64)) % $(32)) * $(64)) + (r139_z % $(64))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_25_ptr) + ((((r139_z / $(2048)) % $(8)) * $(2048)) + (((r139_z / $(64)) % $(32)) * $(64)) + (r139_z % $(64))), dtype=tl.float32))
    r139_value = r139_value.to(tl.float32)
    r139_value = tl.broadcast_to(r139_value, [$(16384)])
    tl.store($(candidate_vmasked_fill_4_ptr) + r139_z, r139_value)
    r140_z = tl.arange($(16384))
    r140_lane = tl.arange($(64))
    r140_row_values = tl.load($(candidate_vmasked_fill_4_ptr) + ((((r140_z[:, None] / $(2048)) % $(8)) * $(2048)) + (((r140_z[:, None] / $(64)) % $(32)) * $(64)) + (r140_lane[None, :] % $(64))), dtype=tl.float32)
    r140_maximum = tl.max(r140_row_values, axis=1, keep_dims=true)
    r140_weights = tl.exp(r140_row_values - r140_maximum)
    r140_total = tl.sum(r140_weights, axis=1)
    r140_maximum = tl.max(r140_row_values, axis=1)
    r140_value = tl.exp(tl.load($(candidate_vmasked_fill_4_ptr) + ((((r140_z / $(2048)) % $(8)) * $(2048)) + (((r140_z / $(64)) % $(32)) * $(64)) + (r140_z % $(64))), dtype=tl.float32) - r140_maximum) / r140_total
    r140_value = r140_value.to(tl.float32)
    r140_value = tl.broadcast_to(r140_value, [$(16384)])
    tl.store($(candidate_vsoftmax_4_ptr) + r140_z, r140_value)
    r141_z = tl.arange($(16384))
    r141_value = tl.load($(candidate_vsoftmax_4_ptr) + ((((r141_z / $(2048)) % $(8)) * $(2048)) + (((r141_z / $(64)) % $(32)) * $(64)) + (r141_z % $(64))), dtype=tl.float32)
    r141_value = r141_value.to(tl.bfloat16)
    r141_value = tl.broadcast_to(r141_value, [$(16384)])
    tl.store($(candidate_vto_copy_36_ptr) + r141_z, r141_value)
    r142_row = tl.arange($(32))
    r142_col = tl.arange($(128))
    r142_lane = tl.arange($(64))
    for r142_group in range($(8)) {
      r142_left = tl.load($(candidate_vto_copy_36_ptr) + (((((((r142_group % $(8)) * $(2048)) + (r142_row[:, None] * $(64)) + r142_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r142_group % $(8)) * $(2048)) + (r142_row[:, None] * $(64)) + r142_lane[None, :]) / $(64)) % $(32)) * $(64)) + ((((r142_group % $(8)) * $(2048)) + (r142_row[:, None] * $(64)) + r142_lane[None, :]) % $(64))), dtype=tl.bfloat16)
      r142_right = tl.load($(candidate_vbmm_17_ptr) + (((((((r142_group % $(8)) * $(8192)) + (r142_lane[:, None] * $(128)) + r142_col[None, :]) / $(128)) % $(64)) * $(1024)) + ((((((r142_group % $(8)) * $(8192)) + (r142_lane[:, None] * $(128)) + r142_col[None, :]) / $(8192)) % $(8)) * $(128)) + ((((r142_group % $(8)) * $(8192)) + (r142_lane[:, None] * $(128)) + r142_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r142_value = tl.dot(r142_left, r142_right)
      r142_value = r142_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_19_ptr) + ((r142_group * $(4096)) + (r142_row[:, None] * $(128)) + r142_col[None, :]), r142_value)
    }
    r143_z = tl.arange($(32768))
    r143_value = tl.load($(candidate_vbmm_19_ptr) + ((((((((r143_z / $(128)) % $(8)) * $(4096)) + ((((r143_z / $(1024)) % $(32)) - $(0)) * $(128)) + (r143_z % $(128))) / $(4096)) % $(8)) * $(4096)) + ((((((r143_z / $(128)) % $(8)) * $(32)) + (((r143_z / $(1024)) % $(32)) - $(0))) % $(32)) * $(128)) + (r143_z % $(128))), dtype=tl.bfloat16)
    r143_value = r143_value.to(tl.bfloat16)
    r143_value = tl.broadcast_to(r143_value, [$(32768)])
    tl.store($(candidate_vcat_22_ptr) + r143_z, r143_value)
    r144_row = tl.arange($(96))
    r144_col = tl.arange($(1024))
    r144_lane = tl.arange($(512))
    for r144_group in range($(1)) {
      r144_left = tl.load($(candidate_vcat_7_ptr) + ((((((r144_row[:, None] * $(512)) + r144_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r144_row[:, None] * $(512)) + r144_lane[None, :]) % $(512)) + $(55296)), dtype=tl.bfloat16)
      r144_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r144_lane[:, None] * $(1024)) + r144_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r144_lane[:, None] * $(1024)) + r144_col[None, :]) % $(128)) * $(512)) + ((((r144_lane[:, None] * $(1024)) + r144_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r144_value = tl.dot(r144_left, r144_right)
      r144_value = r144_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_20_ptr) + ((r144_group * $(98304)) + (r144_row[:, None] * $(1024)) + r144_col[None, :]), r144_value)
    }
    r145_row = tl.arange($(96))
    r145_col = tl.arange($(1024))
    r145_lane = tl.arange($(512))
    for r145_group in range($(1)) {
      r145_left = tl.load($(candidate_vcat_7_ptr) + ((((((r145_row[:, None] * $(512)) + r145_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r145_row[:, None] * $(512)) + r145_lane[None, :]) % $(512)) + $(55296)), dtype=tl.bfloat16)
      r145_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r145_lane[:, None] * $(1024)) + r145_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r145_lane[:, None] * $(1024)) + r145_col[None, :]) % $(128)) * $(512)) + ((((r145_lane[:, None] * $(1024)) + r145_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r145_value = tl.dot(r145_left, r145_right)
      r145_value = r145_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_21_ptr) + ((r145_group * $(98304)) + (r145_row[:, None] * $(1024)) + r145_col[None, :]), r145_value)
    }
    r146_z = tl.arange($(49152))
    r146_value = tl.where((r146_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r146_z / $(1536)) % $(32)) * $(1024)) + (((r146_z / $(192)) % $(8)) * $(128)) + ((r146_z % $(192)) - $(0)) + $(163840)) / $(1024)) % $(288)) * $(1536)) + (((((((r146_z / $(1536)) % $(32)) * $(1024)) + (((r146_z / $(192)) % $(8)) * $(128)) + ((r146_z % $(192)) - $(0)) + $(163840)) / $(128)) % $(8)) * $(192)) + (((((r146_z / $(1536)) % $(32)) * $(1024)) + (((r146_z / $(192)) % $(8)) * $(128)) + ((r146_z % $(192)) - $(0)) + $(163840)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_5_ptr) + ((((((((r146_z / $(1536)) % $(32)) * $(512)) + (((r146_z / $(192)) % $(8)) * $(64)) + ((r146_z % $(192)) - $(128)) + $(81920)) / $(64)) % $(8)) * $(18432)) + (((((((r146_z / $(1536)) % $(32)) * $(512)) + (((r146_z / $(192)) % $(8)) * $(64)) + ((r146_z % $(192)) - $(128)) + $(81920)) / $(512)) % $(288)) * $(64)) + (((((r146_z / $(1536)) % $(32)) * $(512)) + (((r146_z / $(192)) % $(8)) * $(64)) + ((r146_z % $(192)) - $(128)) + $(81920)) % $(64))), dtype=tl.bfloat16))
    r146_value = r146_value.to(tl.bfloat16)
    r146_value = tl.broadcast_to(r146_value, [$(49152)])
    tl.store($(candidate_vcat_23_ptr) + r146_z, r146_value)
    r147_z = tl.arange($(147456))
    r147_value = tl.where((r147_z % $(192)) < $(128), tl.load($(candidate_vbmm_20_ptr) + ((((((((r147_z / $(1536)) % $(96)) * $(1024)) + (((r147_z / $(192)) % $(8)) * $(128)) + ((r147_z % $(192)) - $(0))) / $(1024)) % $(96)) * $(1024)) + (((((((r147_z / $(1536)) % $(96)) * $(1024)) + (((r147_z / $(192)) % $(8)) * $(128)) + ((r147_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r147_z / $(1536)) % $(96)) * $(1024)) + (((r147_z / $(192)) % $(8)) * $(128)) + ((r147_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_7_ptr) + ((((r147_z / $(1536)) % $(96)) * $(576)) + ((((r147_z % $(192)) - $(128)) + $(512)) % $(576)) + $(55296)), dtype=tl.bfloat16))
    r147_value = r147_value.to(tl.bfloat16)
    r147_value = tl.broadcast_to(r147_value, [$(147456)])
    tl.store($(candidate_vcat_24_ptr) + r147_z, r147_value)
    r148_z = tl.arange($(49152))
    r148_value = tl.load($(candidate_vcat_23_ptr) + ((((r148_z / $(192)) % $(32)) * $(1536)) + (((r148_z / $(6144)) % $(8)) * $(192)) + (r148_z % $(192))), dtype=tl.bfloat16)
    r148_value = r148_value.to(tl.float32)
    r148_value = tl.broadcast_to(r148_value, [$(49152)])
    tl.store($(candidate_vto_copy_37_ptr) + r148_z, r148_value)
    r149_z = tl.arange($(147456))
    r149_value = tl.load($(candidate_vcat_24_ptr) + ((((r149_z / $(192)) % $(96)) * $(1536)) + (((r149_z / $(18432)) % $(8)) * $(192)) + (r149_z % $(192))), dtype=tl.bfloat16)
    r149_value = r149_value.to(tl.float32)
    r149_value = tl.broadcast_to(r149_value, [$(147456)])
    tl.store($(candidate_vto_copy_38_ptr) + r149_z, r149_value)
    r150_row = tl.arange($(32))
    r150_col = tl.arange($(96))
    r150_lane = tl.arange($(192))
    for r150_group in range($(8)) {
      r150_left = tl.load($(candidate_vto_copy_37_ptr) + (((((((r150_group % $(8)) * $(6144)) + (r150_row[:, None] * $(192)) + r150_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r150_group % $(8)) * $(6144)) + (r150_row[:, None] * $(192)) + r150_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r150_group % $(8)) * $(6144)) + (r150_row[:, None] * $(192)) + r150_lane[None, :]) % $(192))), dtype=tl.float32)
      r150_right = tl.load($(candidate_vto_copy_38_ptr) + (((((((r150_group % $(8)) * $(18432)) + (r150_lane[:, None] * $(96)) + r150_col[None, :]) / $(18432)) % $(8)) * $(18432)) + (((((r150_group % $(8)) * $(18432)) + (r150_lane[:, None] * $(96)) + r150_col[None, :]) % $(96)) * $(192)) + (((((r150_group % $(8)) * $(18432)) + (r150_lane[:, None] * $(96)) + r150_col[None, :]) / $(96)) % $(192))), dtype=tl.float32)
      r150_value = tl.dot(r150_left, r150_right)
      r150_value = r150_value.to(tl.float32)
      tl.store($(candidate_vbmm_22_ptr) + ((r150_group * $(3072)) + (r150_row[:, None] * $(96)) + r150_col[None, :]), r150_value)
    }
    r151_z = tl.arange($(24576))
    r151_value = (tl.load($(candidate_vbmm_22_ptr) + ((((r151_z / $(3072)) % $(8)) * $(3072)) + (((r151_z / $(96)) % $(32)) * $(96)) + (r151_z % $(96))), dtype=tl.float32) * 0.07216878364870322)
    r151_value = r151_value.to(tl.float32)
    r151_value = tl.broadcast_to(r151_value, [$(24576)])
    tl.store($(candidate_vmul_26_ptr) + r151_z, r151_value)
    r152_z = tl.arange($(32))
    r152_value = (tl.cast(r152_z, tl.int64) + tl.full([], $(96), dtype=tl.int64))
    r152_value = tl.broadcast_to(r152_value, [$(32)])
    tl.store($(candidate_vadd_11_ptr) + r152_z, r152_value)
    r153_z = tl.arange($(32))
    r153_value = (tl.load($(candidate_vadd_11_ptr) + r153_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r153_value = tl.broadcast_to(r153_value, [$(32)])
    tl.store($(candidate_vsub_20_ptr) + r153_z, r153_value)
    r154_z = tl.arange($(3072))
    r154_value = (tl.cast((r154_z % $(96)), tl.int64) > tl.load($(candidate_vsub_20_ptr) + ((r154_z / $(96)) % $(32)), dtype=tl.int64))
    r154_value = tl.broadcast_to(r154_value, [$(3072)])
    tl.store($(candidate_vgt_5_ptr) + r154_z, r154_value)
    r155_z = tl.arange($(24576))
    r155_value = tl.where(tl.load($(candidate_vgt_5_ptr) + ((((r155_z / $(96)) % $(32)) * $(96)) + (r155_z % $(96))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_26_ptr) + ((((r155_z / $(3072)) % $(8)) * $(3072)) + (((r155_z / $(96)) % $(32)) * $(96)) + (r155_z % $(96))), dtype=tl.float32))
    r155_value = r155_value.to(tl.float32)
    r155_value = tl.broadcast_to(r155_value, [$(24576)])
    tl.store($(candidate_vmasked_fill_5_ptr) + r155_z, r155_value)
    r156_z = tl.arange($(24576))
    r156_lane = tl.arange($(96))
    r156_row_values = tl.load($(candidate_vmasked_fill_5_ptr) + ((((r156_z[:, None] / $(3072)) % $(8)) * $(3072)) + (((r156_z[:, None] / $(96)) % $(32)) * $(96)) + (r156_lane[None, :] % $(96))), dtype=tl.float32)
    r156_maximum = tl.max(r156_row_values, axis=1, keep_dims=true)
    r156_weights = tl.exp(r156_row_values - r156_maximum)
    r156_total = tl.sum(r156_weights, axis=1)
    r156_maximum = tl.max(r156_row_values, axis=1)
    r156_value = tl.exp(tl.load($(candidate_vmasked_fill_5_ptr) + ((((r156_z / $(3072)) % $(8)) * $(3072)) + (((r156_z / $(96)) % $(32)) * $(96)) + (r156_z % $(96))), dtype=tl.float32) - r156_maximum) / r156_total
    r156_value = r156_value.to(tl.float32)
    r156_value = tl.broadcast_to(r156_value, [$(24576)])
    tl.store($(candidate_vsoftmax_5_ptr) + r156_z, r156_value)
    r157_z = tl.arange($(24576))
    r157_value = tl.load($(candidate_vsoftmax_5_ptr) + ((((r157_z / $(3072)) % $(8)) * $(3072)) + (((r157_z / $(96)) % $(32)) * $(96)) + (r157_z % $(96))), dtype=tl.float32)
    r157_value = r157_value.to(tl.bfloat16)
    r157_value = tl.broadcast_to(r157_value, [$(24576)])
    tl.store($(candidate_vto_copy_39_ptr) + r157_z, r157_value)
    r158_row = tl.arange($(32))
    r158_col = tl.arange($(128))
    r158_lane = tl.arange($(96))
    for r158_group in range($(8)) {
      r158_left = tl.load($(candidate_vto_copy_39_ptr) + (((((((r158_group % $(8)) * $(3072)) + (r158_row[:, None] * $(96)) + r158_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r158_group % $(8)) * $(3072)) + (r158_row[:, None] * $(96)) + r158_lane[None, :]) / $(96)) % $(32)) * $(96)) + ((((r158_group % $(8)) * $(3072)) + (r158_row[:, None] * $(96)) + r158_lane[None, :]) % $(96))), dtype=tl.bfloat16)
      r158_right = tl.load($(candidate_vbmm_21_ptr) + (((((((r158_group % $(8)) * $(12288)) + (r158_lane[:, None] * $(128)) + r158_col[None, :]) / $(128)) % $(96)) * $(1024)) + ((((((r158_group % $(8)) * $(12288)) + (r158_lane[:, None] * $(128)) + r158_col[None, :]) / $(12288)) % $(8)) * $(128)) + ((((r158_group % $(8)) * $(12288)) + (r158_lane[:, None] * $(128)) + r158_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r158_value = tl.dot(r158_left, r158_right)
      r158_value = r158_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_23_ptr) + ((r158_group * $(4096)) + (r158_row[:, None] * $(128)) + r158_col[None, :]), r158_value)
    }
    r159_z = tl.arange($(32768))
    r159_value = tl.load($(candidate_vbmm_23_ptr) + ((((((((r159_z / $(128)) % $(8)) * $(4096)) + ((((r159_z / $(1024)) % $(32)) - $(0)) * $(128)) + (r159_z % $(128))) / $(4096)) % $(8)) * $(4096)) + ((((((r159_z / $(128)) % $(8)) * $(32)) + (((r159_z / $(1024)) % $(32)) - $(0))) % $(32)) * $(128)) + (r159_z % $(128))), dtype=tl.bfloat16)
    r159_value = r159_value.to(tl.bfloat16)
    r159_value = tl.broadcast_to(r159_value, [$(32768)])
    tl.store($(candidate_vcat_25_ptr) + r159_z, r159_value)
    r160_row = tl.arange($(96))
    r160_col = tl.arange($(1024))
    r160_lane = tl.arange($(512))
    for r160_group in range($(1)) {
      r160_left = tl.load($(candidate_vcat_7_ptr) + ((((((r160_row[:, None] * $(512)) + r160_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r160_row[:, None] * $(512)) + r160_lane[None, :]) % $(512)) + $(110592)), dtype=tl.bfloat16)
      r160_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r160_lane[:, None] * $(1024)) + r160_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r160_lane[:, None] * $(1024)) + r160_col[None, :]) % $(128)) * $(512)) + ((((r160_lane[:, None] * $(1024)) + r160_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r160_value = tl.dot(r160_left, r160_right)
      r160_value = r160_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_24_ptr) + ((r160_group * $(98304)) + (r160_row[:, None] * $(1024)) + r160_col[None, :]), r160_value)
    }
    r161_row = tl.arange($(96))
    r161_col = tl.arange($(1024))
    r161_lane = tl.arange($(512))
    for r161_group in range($(1)) {
      r161_left = tl.load($(candidate_vcat_7_ptr) + ((((((r161_row[:, None] * $(512)) + r161_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r161_row[:, None] * $(512)) + r161_lane[None, :]) % $(512)) + $(110592)), dtype=tl.bfloat16)
      r161_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r161_lane[:, None] * $(1024)) + r161_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r161_lane[:, None] * $(1024)) + r161_col[None, :]) % $(128)) * $(512)) + ((((r161_lane[:, None] * $(1024)) + r161_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r161_value = tl.dot(r161_left, r161_right)
      r161_value = r161_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_25_ptr) + ((r161_group * $(98304)) + (r161_row[:, None] * $(1024)) + r161_col[None, :]), r161_value)
    }
    r162_z = tl.arange($(49152))
    r162_value = tl.where((r162_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r162_z / $(1536)) % $(32)) * $(1024)) + (((r162_z / $(192)) % $(8)) * $(128)) + ((r162_z % $(192)) - $(0)) + $(196608)) / $(1024)) % $(288)) * $(1536)) + (((((((r162_z / $(1536)) % $(32)) * $(1024)) + (((r162_z / $(192)) % $(8)) * $(128)) + ((r162_z % $(192)) - $(0)) + $(196608)) / $(128)) % $(8)) * $(192)) + (((((r162_z / $(1536)) % $(32)) * $(1024)) + (((r162_z / $(192)) % $(8)) * $(128)) + ((r162_z % $(192)) - $(0)) + $(196608)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_5_ptr) + ((((((((r162_z / $(1536)) % $(32)) * $(512)) + (((r162_z / $(192)) % $(8)) * $(64)) + ((r162_z % $(192)) - $(128)) + $(98304)) / $(64)) % $(8)) * $(18432)) + (((((((r162_z / $(1536)) % $(32)) * $(512)) + (((r162_z / $(192)) % $(8)) * $(64)) + ((r162_z % $(192)) - $(128)) + $(98304)) / $(512)) % $(288)) * $(64)) + (((((r162_z / $(1536)) % $(32)) * $(512)) + (((r162_z / $(192)) % $(8)) * $(64)) + ((r162_z % $(192)) - $(128)) + $(98304)) % $(64))), dtype=tl.bfloat16))
    r162_value = r162_value.to(tl.bfloat16)
    r162_value = tl.broadcast_to(r162_value, [$(49152)])
    tl.store($(candidate_vcat_26_ptr) + r162_z, r162_value)
    r163_z = tl.arange($(147456))
    r163_value = tl.where((r163_z % $(192)) < $(128), tl.load($(candidate_vbmm_24_ptr) + ((((((((r163_z / $(1536)) % $(96)) * $(1024)) + (((r163_z / $(192)) % $(8)) * $(128)) + ((r163_z % $(192)) - $(0))) / $(1024)) % $(96)) * $(1024)) + (((((((r163_z / $(1536)) % $(96)) * $(1024)) + (((r163_z / $(192)) % $(8)) * $(128)) + ((r163_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r163_z / $(1536)) % $(96)) * $(1024)) + (((r163_z / $(192)) % $(8)) * $(128)) + ((r163_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_7_ptr) + ((((r163_z / $(1536)) % $(96)) * $(576)) + ((((r163_z % $(192)) - $(128)) + $(512)) % $(576)) + $(110592)), dtype=tl.bfloat16))
    r163_value = r163_value.to(tl.bfloat16)
    r163_value = tl.broadcast_to(r163_value, [$(147456)])
    tl.store($(candidate_vcat_27_ptr) + r163_z, r163_value)
    r164_z = tl.arange($(49152))
    r164_value = tl.load($(candidate_vcat_26_ptr) + ((((r164_z / $(192)) % $(32)) * $(1536)) + (((r164_z / $(6144)) % $(8)) * $(192)) + (r164_z % $(192))), dtype=tl.bfloat16)
    r164_value = r164_value.to(tl.float32)
    r164_value = tl.broadcast_to(r164_value, [$(49152)])
    tl.store($(candidate_vto_copy_40_ptr) + r164_z, r164_value)
    r165_z = tl.arange($(49152))
    r165_value = tl.load($(candidate_vcat_27_ptr) + ((((r165_z / $(192)) % $(32)) * $(1536)) + (((r165_z / $(6144)) % $(8)) * $(192)) + (r165_z % $(192))), dtype=tl.bfloat16)
    r165_value = r165_value.to(tl.float32)
    r165_value = tl.broadcast_to(r165_value, [$(49152)])
    tl.store($(candidate_vto_copy_41_ptr) + r165_z, r165_value)
    r166_row = tl.arange($(32))
    r166_col = tl.arange($(32))
    r166_lane = tl.arange($(192))
    for r166_group in range($(8)) {
      r166_left = tl.load($(candidate_vto_copy_40_ptr) + (((((((r166_group % $(8)) * $(6144)) + (r166_row[:, None] * $(192)) + r166_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r166_group % $(8)) * $(6144)) + (r166_row[:, None] * $(192)) + r166_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r166_group % $(8)) * $(6144)) + (r166_row[:, None] * $(192)) + r166_lane[None, :]) % $(192))), dtype=tl.float32)
      r166_right = tl.load($(candidate_vto_copy_41_ptr) + (((((((r166_group % $(8)) * $(6144)) + (r166_lane[:, None] * $(32)) + r166_col[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r166_group % $(8)) * $(6144)) + (r166_lane[:, None] * $(32)) + r166_col[None, :]) % $(32)) * $(192)) + (((((r166_group % $(8)) * $(6144)) + (r166_lane[:, None] * $(32)) + r166_col[None, :]) / $(32)) % $(192))), dtype=tl.float32)
      r166_value = tl.dot(r166_left, r166_right)
      r166_value = r166_value.to(tl.float32)
      tl.store($(candidate_vbmm_26_ptr) + ((r166_group * $(1024)) + (r166_row[:, None] * $(32)) + r166_col[None, :]), r166_value)
    }
    r167_z = tl.arange($(8192))
    r167_value = (tl.load($(candidate_vbmm_26_ptr) + ((((r167_z / $(1024)) % $(8)) * $(1024)) + (((r167_z / $(32)) % $(32)) * $(32)) + (r167_z % $(32))), dtype=tl.float32) * 0.07216878364870322)
    r167_value = r167_value.to(tl.float32)
    r167_value = tl.broadcast_to(r167_value, [$(8192)])
    tl.store($(candidate_vmul_27_ptr) + r167_z, r167_value)
    r168_z = tl.arange($(32))
    r168_value = (tl.cast(r168_z, tl.int64) + tl.full([], $(32), dtype=tl.int64))
    r168_value = tl.broadcast_to(r168_value, [$(32)])
    tl.store($(candidate_vadd_12_ptr) + r168_z, r168_value)
    r169_z = tl.arange($(32))
    r169_value = (tl.load($(candidate_vadd_12_ptr) + r169_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r169_value = tl.broadcast_to(r169_value, [$(32)])
    tl.store($(candidate_vsub_21_ptr) + r169_z, r169_value)
    r170_z = tl.arange($(1024))
    r170_value = (tl.cast((r170_z % $(32)), tl.int64) > tl.load($(candidate_vsub_21_ptr) + ((r170_z / $(32)) % $(32)), dtype=tl.int64))
    r170_value = tl.broadcast_to(r170_value, [$(1024)])
    tl.store($(candidate_vgt_6_ptr) + r170_z, r170_value)
    r171_z = tl.arange($(8192))
    r171_value = tl.where(tl.load($(candidate_vgt_6_ptr) + ((((r171_z / $(32)) % $(32)) * $(32)) + (r171_z % $(32))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_27_ptr) + ((((r171_z / $(1024)) % $(8)) * $(1024)) + (((r171_z / $(32)) % $(32)) * $(32)) + (r171_z % $(32))), dtype=tl.float32))
    r171_value = r171_value.to(tl.float32)
    r171_value = tl.broadcast_to(r171_value, [$(8192)])
    tl.store($(candidate_vmasked_fill_6_ptr) + r171_z, r171_value)
    r172_z = tl.arange($(8192))
    r172_lane = tl.arange($(32))
    r172_row_values = tl.load($(candidate_vmasked_fill_6_ptr) + ((((r172_z[:, None] / $(1024)) % $(8)) * $(1024)) + (((r172_z[:, None] / $(32)) % $(32)) * $(32)) + (r172_lane[None, :] % $(32))), dtype=tl.float32)
    r172_maximum = tl.max(r172_row_values, axis=1, keep_dims=true)
    r172_weights = tl.exp(r172_row_values - r172_maximum)
    r172_total = tl.sum(r172_weights, axis=1)
    r172_maximum = tl.max(r172_row_values, axis=1)
    r172_value = tl.exp(tl.load($(candidate_vmasked_fill_6_ptr) + ((((r172_z / $(1024)) % $(8)) * $(1024)) + (((r172_z / $(32)) % $(32)) * $(32)) + (r172_z % $(32))), dtype=tl.float32) - r172_maximum) / r172_total
    r172_value = r172_value.to(tl.float32)
    r172_value = tl.broadcast_to(r172_value, [$(8192)])
    tl.store($(candidate_vsoftmax_6_ptr) + r172_z, r172_value)
    r173_z = tl.arange($(8192))
    r173_value = tl.load($(candidate_vsoftmax_6_ptr) + ((((r173_z / $(1024)) % $(8)) * $(1024)) + (((r173_z / $(32)) % $(32)) * $(32)) + (r173_z % $(32))), dtype=tl.float32)
    r173_value = r173_value.to(tl.bfloat16)
    r173_value = tl.broadcast_to(r173_value, [$(8192)])
    tl.store($(candidate_vto_copy_42_ptr) + r173_z, r173_value)
    r174_row = tl.arange($(32))
    r174_col = tl.arange($(128))
    r174_lane = tl.arange($(32))
    for r174_group in range($(8)) {
      r174_left = tl.load($(candidate_vto_copy_42_ptr) + (((((((r174_group % $(8)) * $(1024)) + (r174_row[:, None] * $(32)) + r174_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + ((((((r174_group % $(8)) * $(1024)) + (r174_row[:, None] * $(32)) + r174_lane[None, :]) / $(32)) % $(32)) * $(32)) + ((((r174_group % $(8)) * $(1024)) + (r174_row[:, None] * $(32)) + r174_lane[None, :]) % $(32))), dtype=tl.bfloat16)
      r174_right = tl.load($(candidate_vbmm_25_ptr) + (((((((r174_group % $(8)) * $(4096)) + (r174_lane[:, None] * $(128)) + r174_col[None, :]) / $(128)) % $(32)) * $(1024)) + ((((((r174_group % $(8)) * $(4096)) + (r174_lane[:, None] * $(128)) + r174_col[None, :]) / $(4096)) % $(8)) * $(128)) + ((((r174_group % $(8)) * $(4096)) + (r174_lane[:, None] * $(128)) + r174_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r174_value = tl.dot(r174_left, r174_right)
      r174_value = r174_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_27_ptr) + ((r174_group * $(4096)) + (r174_row[:, None] * $(128)) + r174_col[None, :]), r174_value)
    }
    r175_z = tl.arange($(32768))
    r175_value = tl.load($(candidate_vbmm_27_ptr) + ((((((((r175_z / $(128)) % $(8)) * $(4096)) + ((((r175_z / $(1024)) % $(32)) - $(0)) * $(128)) + (r175_z % $(128))) / $(4096)) % $(8)) * $(4096)) + ((((((r175_z / $(128)) % $(8)) * $(32)) + (((r175_z / $(1024)) % $(32)) - $(0))) % $(32)) * $(128)) + (r175_z % $(128))), dtype=tl.bfloat16)
    r175_value = r175_value.to(tl.bfloat16)
    r175_value = tl.broadcast_to(r175_value, [$(32768)])
    tl.store($(candidate_vcat_28_ptr) + r175_z, r175_value)
    r176_row = tl.arange($(96))
    r176_col = tl.arange($(1024))
    r176_lane = tl.arange($(512))
    for r176_group in range($(1)) {
      r176_left = tl.load($(candidate_vcat_7_ptr) + ((((((r176_row[:, None] * $(512)) + r176_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r176_row[:, None] * $(512)) + r176_lane[None, :]) % $(512)) + $(110592)), dtype=tl.bfloat16)
      r176_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r176_lane[:, None] * $(1024)) + r176_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r176_lane[:, None] * $(1024)) + r176_col[None, :]) % $(128)) * $(512)) + ((((r176_lane[:, None] * $(1024)) + r176_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r176_value = tl.dot(r176_left, r176_right)
      r176_value = r176_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_28_ptr) + ((r176_group * $(98304)) + (r176_row[:, None] * $(1024)) + r176_col[None, :]), r176_value)
    }
    r177_row = tl.arange($(96))
    r177_col = tl.arange($(1024))
    r177_lane = tl.arange($(512))
    for r177_group in range($(1)) {
      r177_left = tl.load($(candidate_vcat_7_ptr) + ((((((r177_row[:, None] * $(512)) + r177_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r177_row[:, None] * $(512)) + r177_lane[None, :]) % $(512)) + $(110592)), dtype=tl.bfloat16)
      r177_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r177_lane[:, None] * $(1024)) + r177_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r177_lane[:, None] * $(1024)) + r177_col[None, :]) % $(128)) * $(512)) + ((((r177_lane[:, None] * $(1024)) + r177_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r177_value = tl.dot(r177_left, r177_right)
      r177_value = r177_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_29_ptr) + ((r177_group * $(98304)) + (r177_row[:, None] * $(1024)) + r177_col[None, :]), r177_value)
    }
    r178_z = tl.arange($(49152))
    r178_value = tl.where((r178_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r178_z / $(1536)) % $(32)) * $(1024)) + (((r178_z / $(192)) % $(8)) * $(128)) + ((r178_z % $(192)) - $(0)) + $(229376)) / $(1024)) % $(288)) * $(1536)) + (((((((r178_z / $(1536)) % $(32)) * $(1024)) + (((r178_z / $(192)) % $(8)) * $(128)) + ((r178_z % $(192)) - $(0)) + $(229376)) / $(128)) % $(8)) * $(192)) + (((((r178_z / $(1536)) % $(32)) * $(1024)) + (((r178_z / $(192)) % $(8)) * $(128)) + ((r178_z % $(192)) - $(0)) + $(229376)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_5_ptr) + ((((((((r178_z / $(1536)) % $(32)) * $(512)) + (((r178_z / $(192)) % $(8)) * $(64)) + ((r178_z % $(192)) - $(128)) + $(114688)) / $(64)) % $(8)) * $(18432)) + (((((((r178_z / $(1536)) % $(32)) * $(512)) + (((r178_z / $(192)) % $(8)) * $(64)) + ((r178_z % $(192)) - $(128)) + $(114688)) / $(512)) % $(288)) * $(64)) + (((((r178_z / $(1536)) % $(32)) * $(512)) + (((r178_z / $(192)) % $(8)) * $(64)) + ((r178_z % $(192)) - $(128)) + $(114688)) % $(64))), dtype=tl.bfloat16))
    r178_value = r178_value.to(tl.bfloat16)
    r178_value = tl.broadcast_to(r178_value, [$(49152)])
    tl.store($(candidate_vcat_29_ptr) + r178_z, r178_value)
    r179_z = tl.arange($(147456))
    r179_value = tl.where((r179_z % $(192)) < $(128), tl.load($(candidate_vbmm_28_ptr) + ((((((((r179_z / $(1536)) % $(96)) * $(1024)) + (((r179_z / $(192)) % $(8)) * $(128)) + ((r179_z % $(192)) - $(0))) / $(1024)) % $(96)) * $(1024)) + (((((((r179_z / $(1536)) % $(96)) * $(1024)) + (((r179_z / $(192)) % $(8)) * $(128)) + ((r179_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r179_z / $(1536)) % $(96)) * $(1024)) + (((r179_z / $(192)) % $(8)) * $(128)) + ((r179_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_7_ptr) + ((((r179_z / $(1536)) % $(96)) * $(576)) + ((((r179_z % $(192)) - $(128)) + $(512)) % $(576)) + $(110592)), dtype=tl.bfloat16))
    r179_value = r179_value.to(tl.bfloat16)
    r179_value = tl.broadcast_to(r179_value, [$(147456)])
    tl.store($(candidate_vcat_30_ptr) + r179_z, r179_value)
    r180_z = tl.arange($(49152))
    r180_value = tl.load($(candidate_vcat_29_ptr) + ((((r180_z / $(192)) % $(32)) * $(1536)) + (((r180_z / $(6144)) % $(8)) * $(192)) + (r180_z % $(192))), dtype=tl.bfloat16)
    r180_value = r180_value.to(tl.float32)
    r180_value = tl.broadcast_to(r180_value, [$(49152)])
    tl.store($(candidate_vto_copy_43_ptr) + r180_z, r180_value)
    r181_z = tl.arange($(98304))
    r181_value = tl.load($(candidate_vcat_30_ptr) + ((((r181_z / $(192)) % $(64)) * $(1536)) + (((r181_z / $(12288)) % $(8)) * $(192)) + (r181_z % $(192))), dtype=tl.bfloat16)
    r181_value = r181_value.to(tl.float32)
    r181_value = tl.broadcast_to(r181_value, [$(98304)])
    tl.store($(candidate_vto_copy_44_ptr) + r181_z, r181_value)
    r182_row = tl.arange($(32))
    r182_col = tl.arange($(64))
    r182_lane = tl.arange($(192))
    for r182_group in range($(8)) {
      r182_left = tl.load($(candidate_vto_copy_43_ptr) + (((((((r182_group % $(8)) * $(6144)) + (r182_row[:, None] * $(192)) + r182_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r182_group % $(8)) * $(6144)) + (r182_row[:, None] * $(192)) + r182_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r182_group % $(8)) * $(6144)) + (r182_row[:, None] * $(192)) + r182_lane[None, :]) % $(192))), dtype=tl.float32)
      r182_right = tl.load($(candidate_vto_copy_44_ptr) + (((((((r182_group % $(8)) * $(12288)) + (r182_lane[:, None] * $(64)) + r182_col[None, :]) / $(12288)) % $(8)) * $(12288)) + (((((r182_group % $(8)) * $(12288)) + (r182_lane[:, None] * $(64)) + r182_col[None, :]) % $(64)) * $(192)) + (((((r182_group % $(8)) * $(12288)) + (r182_lane[:, None] * $(64)) + r182_col[None, :]) / $(64)) % $(192))), dtype=tl.float32)
      r182_value = tl.dot(r182_left, r182_right)
      r182_value = r182_value.to(tl.float32)
      tl.store($(candidate_vbmm_30_ptr) + ((r182_group * $(2048)) + (r182_row[:, None] * $(64)) + r182_col[None, :]), r182_value)
    }
    r183_z = tl.arange($(16384))
    r183_value = (tl.load($(candidate_vbmm_30_ptr) + ((((r183_z / $(2048)) % $(8)) * $(2048)) + (((r183_z / $(64)) % $(32)) * $(64)) + (r183_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r183_value = r183_value.to(tl.float32)
    r183_value = tl.broadcast_to(r183_value, [$(16384)])
    tl.store($(candidate_vmul_28_ptr) + r183_z, r183_value)
    r184_z = tl.arange($(32))
    r184_value = (tl.cast(r184_z, tl.int64) + tl.full([], $(64), dtype=tl.int64))
    r184_value = tl.broadcast_to(r184_value, [$(32)])
    tl.store($(candidate_vadd_13_ptr) + r184_z, r184_value)
    r185_z = tl.arange($(32))
    r185_value = (tl.load($(candidate_vadd_13_ptr) + r185_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r185_value = tl.broadcast_to(r185_value, [$(32)])
    tl.store($(candidate_vsub_22_ptr) + r185_z, r185_value)
    r186_z = tl.arange($(2048))
    r186_value = (tl.cast((r186_z % $(64)), tl.int64) > tl.load($(candidate_vsub_22_ptr) + ((r186_z / $(64)) % $(32)), dtype=tl.int64))
    r186_value = tl.broadcast_to(r186_value, [$(2048)])
    tl.store($(candidate_vgt_7_ptr) + r186_z, r186_value)
    r187_z = tl.arange($(16384))
    r187_value = tl.where(tl.load($(candidate_vgt_7_ptr) + ((((r187_z / $(64)) % $(32)) * $(64)) + (r187_z % $(64))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_28_ptr) + ((((r187_z / $(2048)) % $(8)) * $(2048)) + (((r187_z / $(64)) % $(32)) * $(64)) + (r187_z % $(64))), dtype=tl.float32))
    r187_value = r187_value.to(tl.float32)
    r187_value = tl.broadcast_to(r187_value, [$(16384)])
    tl.store($(candidate_vmasked_fill_7_ptr) + r187_z, r187_value)
    r188_z = tl.arange($(16384))
    r188_lane = tl.arange($(64))
    r188_row_values = tl.load($(candidate_vmasked_fill_7_ptr) + ((((r188_z[:, None] / $(2048)) % $(8)) * $(2048)) + (((r188_z[:, None] / $(64)) % $(32)) * $(64)) + (r188_lane[None, :] % $(64))), dtype=tl.float32)
    r188_maximum = tl.max(r188_row_values, axis=1, keep_dims=true)
    r188_weights = tl.exp(r188_row_values - r188_maximum)
    r188_total = tl.sum(r188_weights, axis=1)
    r188_maximum = tl.max(r188_row_values, axis=1)
    r188_value = tl.exp(tl.load($(candidate_vmasked_fill_7_ptr) + ((((r188_z / $(2048)) % $(8)) * $(2048)) + (((r188_z / $(64)) % $(32)) * $(64)) + (r188_z % $(64))), dtype=tl.float32) - r188_maximum) / r188_total
    r188_value = r188_value.to(tl.float32)
    r188_value = tl.broadcast_to(r188_value, [$(16384)])
    tl.store($(candidate_vsoftmax_7_ptr) + r188_z, r188_value)
    r189_z = tl.arange($(16384))
    r189_value = tl.load($(candidate_vsoftmax_7_ptr) + ((((r189_z / $(2048)) % $(8)) * $(2048)) + (((r189_z / $(64)) % $(32)) * $(64)) + (r189_z % $(64))), dtype=tl.float32)
    r189_value = r189_value.to(tl.bfloat16)
    r189_value = tl.broadcast_to(r189_value, [$(16384)])
    tl.store($(candidate_vto_copy_45_ptr) + r189_z, r189_value)
    r190_row = tl.arange($(32))
    r190_col = tl.arange($(128))
    r190_lane = tl.arange($(64))
    for r190_group in range($(8)) {
      r190_left = tl.load($(candidate_vto_copy_45_ptr) + (((((((r190_group % $(8)) * $(2048)) + (r190_row[:, None] * $(64)) + r190_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r190_group % $(8)) * $(2048)) + (r190_row[:, None] * $(64)) + r190_lane[None, :]) / $(64)) % $(32)) * $(64)) + ((((r190_group % $(8)) * $(2048)) + (r190_row[:, None] * $(64)) + r190_lane[None, :]) % $(64))), dtype=tl.bfloat16)
      r190_right = tl.load($(candidate_vbmm_29_ptr) + (((((((r190_group % $(8)) * $(8192)) + (r190_lane[:, None] * $(128)) + r190_col[None, :]) / $(128)) % $(64)) * $(1024)) + ((((((r190_group % $(8)) * $(8192)) + (r190_lane[:, None] * $(128)) + r190_col[None, :]) / $(8192)) % $(8)) * $(128)) + ((((r190_group % $(8)) * $(8192)) + (r190_lane[:, None] * $(128)) + r190_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r190_value = tl.dot(r190_left, r190_right)
      r190_value = r190_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_31_ptr) + ((r190_group * $(4096)) + (r190_row[:, None] * $(128)) + r190_col[None, :]), r190_value)
    }
    r191_z = tl.arange($(32768))
    r191_value = tl.load($(candidate_vbmm_31_ptr) + ((((((((r191_z / $(128)) % $(8)) * $(4096)) + ((((r191_z / $(1024)) % $(32)) - $(0)) * $(128)) + (r191_z % $(128))) / $(4096)) % $(8)) * $(4096)) + ((((((r191_z / $(128)) % $(8)) * $(32)) + (((r191_z / $(1024)) % $(32)) - $(0))) % $(32)) * $(128)) + (r191_z % $(128))), dtype=tl.bfloat16)
    r191_value = r191_value.to(tl.bfloat16)
    r191_value = tl.broadcast_to(r191_value, [$(32768)])
    tl.store($(candidate_vcat_31_ptr) + r191_z, r191_value)
    r192_row = tl.arange($(96))
    r192_col = tl.arange($(1024))
    r192_lane = tl.arange($(512))
    for r192_group in range($(1)) {
      r192_left = tl.load($(candidate_vcat_7_ptr) + ((((((r192_row[:, None] * $(512)) + r192_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r192_row[:, None] * $(512)) + r192_lane[None, :]) % $(512)) + $(110592)), dtype=tl.bfloat16)
      r192_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r192_lane[:, None] * $(1024)) + r192_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r192_lane[:, None] * $(1024)) + r192_col[None, :]) % $(128)) * $(512)) + ((((r192_lane[:, None] * $(1024)) + r192_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r192_value = tl.dot(r192_left, r192_right)
      r192_value = r192_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_32_ptr) + ((r192_group * $(98304)) + (r192_row[:, None] * $(1024)) + r192_col[None, :]), r192_value)
    }
    r193_row = tl.arange($(96))
    r193_col = tl.arange($(1024))
    r193_lane = tl.arange($(512))
    for r193_group in range($(1)) {
      r193_left = tl.load($(candidate_vcat_7_ptr) + ((((((r193_row[:, None] * $(512)) + r193_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r193_row[:, None] * $(512)) + r193_lane[None, :]) % $(512)) + $(110592)), dtype=tl.bfloat16)
      r193_right = tl.load($(candidate_vto_copy_5_ptr) + ((((((r193_lane[:, None] * $(1024)) + r193_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r193_lane[:, None] * $(1024)) + r193_col[None, :]) % $(128)) * $(512)) + ((((r193_lane[:, None] * $(1024)) + r193_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r193_value = tl.dot(r193_left, r193_right)
      r193_value = r193_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_33_ptr) + ((r193_group * $(98304)) + (r193_row[:, None] * $(1024)) + r193_col[None, :]), r193_value)
    }
    r194_z = tl.arange($(49152))
    r194_value = tl.where((r194_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r194_z / $(1536)) % $(32)) * $(1024)) + (((r194_z / $(192)) % $(8)) * $(128)) + ((r194_z % $(192)) - $(0)) + $(262144)) / $(1024)) % $(288)) * $(1536)) + (((((((r194_z / $(1536)) % $(32)) * $(1024)) + (((r194_z / $(192)) % $(8)) * $(128)) + ((r194_z % $(192)) - $(0)) + $(262144)) / $(128)) % $(8)) * $(192)) + (((((r194_z / $(1536)) % $(32)) * $(1024)) + (((r194_z / $(192)) % $(8)) * $(128)) + ((r194_z % $(192)) - $(0)) + $(262144)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_5_ptr) + ((((((((r194_z / $(1536)) % $(32)) * $(512)) + (((r194_z / $(192)) % $(8)) * $(64)) + ((r194_z % $(192)) - $(128)) + $(131072)) / $(64)) % $(8)) * $(18432)) + (((((((r194_z / $(1536)) % $(32)) * $(512)) + (((r194_z / $(192)) % $(8)) * $(64)) + ((r194_z % $(192)) - $(128)) + $(131072)) / $(512)) % $(288)) * $(64)) + (((((r194_z / $(1536)) % $(32)) * $(512)) + (((r194_z / $(192)) % $(8)) * $(64)) + ((r194_z % $(192)) - $(128)) + $(131072)) % $(64))), dtype=tl.bfloat16))
    r194_value = r194_value.to(tl.bfloat16)
    r194_value = tl.broadcast_to(r194_value, [$(49152)])
    tl.store($(candidate_vcat_32_ptr) + r194_z, r194_value)
    r195_z = tl.arange($(147456))
    r195_value = tl.where((r195_z % $(192)) < $(128), tl.load($(candidate_vbmm_32_ptr) + ((((((((r195_z / $(1536)) % $(96)) * $(1024)) + (((r195_z / $(192)) % $(8)) * $(128)) + ((r195_z % $(192)) - $(0))) / $(1024)) % $(96)) * $(1024)) + (((((((r195_z / $(1536)) % $(96)) * $(1024)) + (((r195_z / $(192)) % $(8)) * $(128)) + ((r195_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r195_z / $(1536)) % $(96)) * $(1024)) + (((r195_z / $(192)) % $(8)) * $(128)) + ((r195_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_7_ptr) + ((((r195_z / $(1536)) % $(96)) * $(576)) + ((((r195_z % $(192)) - $(128)) + $(512)) % $(576)) + $(110592)), dtype=tl.bfloat16))
    r195_value = r195_value.to(tl.bfloat16)
    r195_value = tl.broadcast_to(r195_value, [$(147456)])
    tl.store($(candidate_vcat_33_ptr) + r195_z, r195_value)
    r196_z = tl.arange($(49152))
    r196_value = tl.load($(candidate_vcat_32_ptr) + ((((r196_z / $(192)) % $(32)) * $(1536)) + (((r196_z / $(6144)) % $(8)) * $(192)) + (r196_z % $(192))), dtype=tl.bfloat16)
    r196_value = r196_value.to(tl.float32)
    r196_value = tl.broadcast_to(r196_value, [$(49152)])
    tl.store($(candidate_vto_copy_46_ptr) + r196_z, r196_value)
    r197_z = tl.arange($(147456))
    r197_value = tl.load($(candidate_vcat_33_ptr) + ((((r197_z / $(192)) % $(96)) * $(1536)) + (((r197_z / $(18432)) % $(8)) * $(192)) + (r197_z % $(192))), dtype=tl.bfloat16)
    r197_value = r197_value.to(tl.float32)
    r197_value = tl.broadcast_to(r197_value, [$(147456)])
    tl.store($(candidate_vto_copy_47_ptr) + r197_z, r197_value)
    r198_row = tl.arange($(32))
    r198_col = tl.arange($(96))
    r198_lane = tl.arange($(192))
    for r198_group in range($(8)) {
      r198_left = tl.load($(candidate_vto_copy_46_ptr) + (((((((r198_group % $(8)) * $(6144)) + (r198_row[:, None] * $(192)) + r198_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r198_group % $(8)) * $(6144)) + (r198_row[:, None] * $(192)) + r198_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r198_group % $(8)) * $(6144)) + (r198_row[:, None] * $(192)) + r198_lane[None, :]) % $(192))), dtype=tl.float32)
      r198_right = tl.load($(candidate_vto_copy_47_ptr) + (((((((r198_group % $(8)) * $(18432)) + (r198_lane[:, None] * $(96)) + r198_col[None, :]) / $(18432)) % $(8)) * $(18432)) + (((((r198_group % $(8)) * $(18432)) + (r198_lane[:, None] * $(96)) + r198_col[None, :]) % $(96)) * $(192)) + (((((r198_group % $(8)) * $(18432)) + (r198_lane[:, None] * $(96)) + r198_col[None, :]) / $(96)) % $(192))), dtype=tl.float32)
      r198_value = tl.dot(r198_left, r198_right)
      r198_value = r198_value.to(tl.float32)
      tl.store($(candidate_vbmm_34_ptr) + ((r198_group * $(3072)) + (r198_row[:, None] * $(96)) + r198_col[None, :]), r198_value)
    }
    r199_z = tl.arange($(24576))
    r199_value = (tl.load($(candidate_vbmm_34_ptr) + ((((r199_z / $(3072)) % $(8)) * $(3072)) + (((r199_z / $(96)) % $(32)) * $(96)) + (r199_z % $(96))), dtype=tl.float32) * 0.07216878364870322)
    r199_value = r199_value.to(tl.float32)
    r199_value = tl.broadcast_to(r199_value, [$(24576)])
    tl.store($(candidate_vmul_29_ptr) + r199_z, r199_value)
    r200_z = tl.arange($(32))
    r200_value = (tl.cast(r200_z, tl.int64) + tl.full([], $(96), dtype=tl.int64))
    r200_value = tl.broadcast_to(r200_value, [$(32)])
    tl.store($(candidate_vadd_14_ptr) + r200_z, r200_value)
    r201_z = tl.arange($(32))
    r201_value = (tl.load($(candidate_vadd_14_ptr) + r201_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r201_value = tl.broadcast_to(r201_value, [$(32)])
    tl.store($(candidate_vsub_23_ptr) + r201_z, r201_value)
    r202_z = tl.arange($(3072))
    r202_value = (tl.cast((r202_z % $(96)), tl.int64) > tl.load($(candidate_vsub_23_ptr) + ((r202_z / $(96)) % $(32)), dtype=tl.int64))
    r202_value = tl.broadcast_to(r202_value, [$(3072)])
    tl.store($(candidate_vgt_8_ptr) + r202_z, r202_value)
    r203_z = tl.arange($(24576))
    r203_value = tl.where(tl.load($(candidate_vgt_8_ptr) + ((((r203_z / $(96)) % $(32)) * $(96)) + (r203_z % $(96))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_29_ptr) + ((((r203_z / $(3072)) % $(8)) * $(3072)) + (((r203_z / $(96)) % $(32)) * $(96)) + (r203_z % $(96))), dtype=tl.float32))
    r203_value = r203_value.to(tl.float32)
    r203_value = tl.broadcast_to(r203_value, [$(24576)])
    tl.store($(candidate_vmasked_fill_8_ptr) + r203_z, r203_value)
    r204_z = tl.arange($(24576))
    r204_lane = tl.arange($(96))
    r204_row_values = tl.load($(candidate_vmasked_fill_8_ptr) + ((((r204_z[:, None] / $(3072)) % $(8)) * $(3072)) + (((r204_z[:, None] / $(96)) % $(32)) * $(96)) + (r204_lane[None, :] % $(96))), dtype=tl.float32)
    r204_maximum = tl.max(r204_row_values, axis=1, keep_dims=true)
    r204_weights = tl.exp(r204_row_values - r204_maximum)
    r204_total = tl.sum(r204_weights, axis=1)
    r204_maximum = tl.max(r204_row_values, axis=1)
    r204_value = tl.exp(tl.load($(candidate_vmasked_fill_8_ptr) + ((((r204_z / $(3072)) % $(8)) * $(3072)) + (((r204_z / $(96)) % $(32)) * $(96)) + (r204_z % $(96))), dtype=tl.float32) - r204_maximum) / r204_total
    r204_value = r204_value.to(tl.float32)
    r204_value = tl.broadcast_to(r204_value, [$(24576)])
    tl.store($(candidate_vsoftmax_8_ptr) + r204_z, r204_value)
    r205_z = tl.arange($(24576))
    r205_value = tl.load($(candidate_vsoftmax_8_ptr) + ((((r205_z / $(3072)) % $(8)) * $(3072)) + (((r205_z / $(96)) % $(32)) * $(96)) + (r205_z % $(96))), dtype=tl.float32)
    r205_value = r205_value.to(tl.bfloat16)
    r205_value = tl.broadcast_to(r205_value, [$(24576)])
    tl.store($(candidate_vto_copy_48_ptr) + r205_z, r205_value)
    r206_row = tl.arange($(32))
    r206_col = tl.arange($(128))
    r206_lane = tl.arange($(96))
    for r206_group in range($(8)) {
      r206_left = tl.load($(candidate_vto_copy_48_ptr) + (((((((r206_group % $(8)) * $(3072)) + (r206_row[:, None] * $(96)) + r206_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r206_group % $(8)) * $(3072)) + (r206_row[:, None] * $(96)) + r206_lane[None, :]) / $(96)) % $(32)) * $(96)) + ((((r206_group % $(8)) * $(3072)) + (r206_row[:, None] * $(96)) + r206_lane[None, :]) % $(96))), dtype=tl.bfloat16)
      r206_right = tl.load($(candidate_vbmm_33_ptr) + (((((((r206_group % $(8)) * $(12288)) + (r206_lane[:, None] * $(128)) + r206_col[None, :]) / $(128)) % $(96)) * $(1024)) + ((((((r206_group % $(8)) * $(12288)) + (r206_lane[:, None] * $(128)) + r206_col[None, :]) / $(12288)) % $(8)) * $(128)) + ((((r206_group % $(8)) * $(12288)) + (r206_lane[:, None] * $(128)) + r206_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r206_value = tl.dot(r206_left, r206_right)
      r206_value = r206_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_35_ptr) + ((r206_group * $(4096)) + (r206_row[:, None] * $(128)) + r206_col[None, :]), r206_value)
    }
    r207_z = tl.arange($(32768))
    r207_value = tl.load($(candidate_vbmm_35_ptr) + ((((((((r207_z / $(128)) % $(8)) * $(4096)) + ((((r207_z / $(1024)) % $(32)) - $(0)) * $(128)) + (r207_z % $(128))) / $(4096)) % $(8)) * $(4096)) + ((((((r207_z / $(128)) % $(8)) * $(32)) + (((r207_z / $(1024)) % $(32)) - $(0))) % $(32)) * $(128)) + (r207_z % $(128))), dtype=tl.bfloat16)
    r207_value = r207_value.to(tl.bfloat16)
    r207_value = tl.broadcast_to(r207_value, [$(32768)])
    tl.store($(candidate_vcat_34_ptr) + r207_z, r207_value)
    r208_z = tl.arange($(294912))
    r208_value = tl.where(((r208_z / $(1024)) % $(288)) < $(32), tl.load($(candidate_vcat_10_ptr) + ((((((r208_z / $(1024)) % $(288)) - $(0)) % $(32)) * $(1024)) + (((r208_z / $(128)) % $(8)) * $(128)) + (r208_z % $(128))), dtype=tl.bfloat16), tl.where(((r208_z / $(1024)) % $(288)) < $(64), tl.load($(candidate_vcat_13_ptr) + ((((((r208_z / $(1024)) % $(288)) - $(32)) % $(32)) * $(1024)) + (((r208_z / $(128)) % $(8)) * $(128)) + (r208_z % $(128))), dtype=tl.bfloat16), tl.where(((r208_z / $(1024)) % $(288)) < $(96), tl.load($(candidate_vcat_16_ptr) + ((((((r208_z / $(1024)) % $(288)) - $(64)) % $(32)) * $(1024)) + (((r208_z / $(128)) % $(8)) * $(128)) + (r208_z % $(128))), dtype=tl.bfloat16), tl.where(((r208_z / $(1024)) % $(288)) < $(128), tl.load($(candidate_vcat_19_ptr) + ((((((r208_z / $(1024)) % $(288)) - $(96)) % $(32)) * $(1024)) + (((r208_z / $(128)) % $(8)) * $(128)) + (r208_z % $(128))), dtype=tl.bfloat16), tl.where(((r208_z / $(1024)) % $(288)) < $(160), tl.load($(candidate_vcat_22_ptr) + ((((((r208_z / $(1024)) % $(288)) - $(128)) % $(32)) * $(1024)) + (((r208_z / $(128)) % $(8)) * $(128)) + (r208_z % $(128))), dtype=tl.bfloat16), tl.where(((r208_z / $(1024)) % $(288)) < $(192), tl.load($(candidate_vcat_25_ptr) + ((((((r208_z / $(1024)) % $(288)) - $(160)) % $(32)) * $(1024)) + (((r208_z / $(128)) % $(8)) * $(128)) + (r208_z % $(128))), dtype=tl.bfloat16), tl.where(((r208_z / $(1024)) % $(288)) < $(224), tl.load($(candidate_vcat_28_ptr) + ((((((r208_z / $(1024)) % $(288)) - $(192)) % $(32)) * $(1024)) + (((r208_z / $(128)) % $(8)) * $(128)) + (r208_z % $(128))), dtype=tl.bfloat16), tl.where(((r208_z / $(1024)) % $(288)) < $(256), tl.load($(candidate_vcat_31_ptr) + ((((((r208_z / $(1024)) % $(288)) - $(224)) % $(32)) * $(1024)) + (((r208_z / $(128)) % $(8)) * $(128)) + (r208_z % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_34_ptr) + ((((((r208_z / $(1024)) % $(288)) - $(256)) % $(32)) * $(1024)) + (((r208_z / $(128)) % $(8)) * $(128)) + (r208_z % $(128))), dtype=tl.bfloat16)))))))))
    r208_value = r208_value.to(tl.bfloat16)
    r208_value = tl.broadcast_to(r208_value, [$(294912)])
    tl.store($(candidate_vcat_35_ptr) + r208_z, r208_value)
    r209_row = tl.arange($(288))
    r209_col = tl.arange($(512))
    r209_lane = tl.arange($(1024))
    r209_left = tl.load($(candidate_vcat_35_ptr) + ((((((r209_row[:, None] * $(1024)) + r209_lane[None, :]) / $(1024)) % $(288)) * $(1024)) + (((((((((r209_row[:, None] * $(1024)) + r209_lane[None, :]) / $(1024)) % $(288)) * $(1024)) + (((r209_row[:, None] * $(1024)) + r209_lane[None, :]) % $(1024))) / $(128)) % $(8)) * $(128)) + (((((((r209_row[:, None] * $(1024)) + r209_lane[None, :]) / $(1024)) % $(288)) * $(1024)) + (((r209_row[:, None] * $(1024)) + r209_lane[None, :]) % $(1024))) % $(128))), dtype=tl.bfloat16)
    r209_right = tl.load($(candidate_vto_copy_6_ptr) + (((r209_col[None, :] % $(512)) * $(1024)) + (r209_lane[:, None] % $(1024))), dtype=tl.bfloat16)
    r209_value = tl.dot(r209_left, r209_right)
    r209_value = r209_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_3_ptr) + ((r209_row[:, None] * $(512)) + r209_col[None, :]), r209_value)
    r210_z = tl.arange($(49152))
    r210_value = tl.load($(candidate_vmm_3_ptr) + (((((((tl.load($(candidate_vcat_4_ptr) + ((r210_z / $(512)) % $(96)), dtype=tl.int64) * $(512)) + (r210_z % $(512))) / $(512)) % $(288)) % $(288)) * $(512)) + ((((tl.load($(candidate_vcat_4_ptr) + ((r210_z / $(512)) % $(96)), dtype=tl.int64) * $(512)) + (r210_z % $(512))) % $(512)) % $(512))), dtype=tl.bfloat16)
    r210_value = r210_value.to(tl.bfloat16)
    r210_value = tl.broadcast_to(r210_value, [$(49152)])
    tl.store($(candidate_vindex_2_ptr) + r210_z, r210_value)
    z = tl.arange($(49152))
    value = tl.load($(candidate_vindex_2_ptr) + ((((z / $(512)) % $(96)) * $(512)) + (z % $(512))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(49152)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_mla_fa4_nonabsorb_seqused_fwd
