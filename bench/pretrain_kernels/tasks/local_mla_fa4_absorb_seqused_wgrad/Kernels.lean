import VeriTile.Triton.DSL

/-! Explicit arithmetic lowerings of the two selected paths. Connection to each
selected native backend remains a separate unsolved goal. -/

namespace VeriTile.Bench.PretrainKernels.local_mla_fa4_absorb_seqused_wgrad
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
  let vto_copy_18_ptr : RegionName := "vto_copy_18"
  let vmm_ptr : RegionName := "vmm"
  let vto_copy_19_ptr : RegionName := "vto_copy_19"
  let vpow_2_ptr : RegionName := "vpow_2"
  let vmean_ptr : RegionName := "vmean"
  let vadd_2_ptr : RegionName := "vadd_2"
  let vrsqrt_ptr : RegionName := "vrsqrt"
  let vmul_9_ptr : RegionName := "vmul_9"
  let vmul_10_ptr : RegionName := "vmul_10"
  let vto_copy_20_ptr : RegionName := "vto_copy_20"
  let vmm_1_ptr : RegionName := "vmm_1"
  let vmm_2_ptr : RegionName := "vmm_2"
  let vto_copy_21_ptr : RegionName := "vto_copy_21"
  let vpow_3_ptr : RegionName := "vpow_3"
  let vmean_1_ptr : RegionName := "vmean_1"
  let vadd_3_ptr : RegionName := "vadd_3"
  let vrsqrt_1_ptr : RegionName := "vrsqrt_1"
  let vmul_11_ptr : RegionName := "vmul_11"
  let vmul_12_ptr : RegionName := "vmul_12"
  let vto_copy_22_ptr : RegionName := "vto_copy_22"
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
  let vto_copy_23_ptr : RegionName := "vto_copy_23"
  let vsoftmax_ptr : RegionName := "vsoftmax"
  let vto_copy_24_ptr : RegionName := "vto_copy_24"
  let vindex_put_ptr : RegionName := "vindex_put"
  let vmm_5_ptr : RegionName := "vmm_5"
  let vbmm_2_ptr : RegionName := "vbmm_2"
  let vbmm_3_ptr : RegionName := "vbmm_3"
  let vto_copy_25_ptr : RegionName := "vto_copy_25"
  let vsoftmax_backward_data_ptr : RegionName := "vsoftmax_backward_data"
  let vto_copy_26_ptr : RegionName := "vto_copy_26"
  let vmul_22_ptr : RegionName := "vmul_22"
  let vbmm_4_ptr : RegionName := "vbmm_4"
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
    r15_z = tl.arange($(288))
    r15_value = tl.where(r15_z < $(96), tl.cast((r15_z - $(0)), tl.int64), tl.where(r15_z < $(192), tl.cast((r15_z - $(96)), tl.int64), tl.cast((r15_z - $(192)), tl.int64)))
    r15_value = tl.broadcast_to(r15_value, [$(288)])
    tl.store($(vcat_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(32))
    r16_value = ((tl.cast(tl.toReal(r16_z), tl.float32) * 2.0) / 64.0)
    r16_value = r16_value.to(tl.float32)
    r16_value = tl.broadcast_to(r16_value, [$(32)])
    tl.store($(vdiv_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(32))
    r17_value = tl.extra.cuda.libdevice.pow(10000.0, tl.load($(vdiv_ptr) + r17_z, dtype=tl.float32))
    r17_value = r17_value.to(tl.float32)
    r17_value = tl.broadcast_to(r17_value, [$(32)])
    tl.store($(vpow_1_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(32))
    r18_value = 1.0 / tl.load($(vpow_1_ptr) + r18_z, dtype=tl.float32)
    r18_value = r18_value.to(tl.float32)
    r18_value = tl.broadcast_to(r18_value, [$(32)])
    tl.store($(vreciprocal_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(32))
    r19_value = (tl.load($(vreciprocal_ptr) + r19_z, dtype=tl.float32) * 1.0)
    r19_value = r19_value.to(tl.float32)
    r19_value = tl.broadcast_to(r19_value, [$(32)])
    tl.store($(vmul_7_ptr) + r19_z, r19_value)
    r20_z = tl.arange($(9216))
    r20_value = (tl.cast(tl.toReal(((r20_z / $(32)) % $(288))), tl.float32) * tl.load($(vmul_7_ptr) + (r20_z % $(32)), dtype=tl.float32))
    r20_value = r20_value.to(tl.float32)
    r20_value = tl.broadcast_to(r20_value, [$(9216)])
    tl.store($(vmul_8_ptr) + r20_z, r20_value)
    r21_z = tl.arange($(18432))
    r21_value = tl.where((r21_z % $(64)) < $(32), tl.load($(vmul_8_ptr) + ((((r21_z / $(64)) % $(288)) * $(32)) + (((r21_z % $(64)) - $(0)) % $(32))), dtype=tl.float32), tl.load($(vmul_8_ptr) + ((((r21_z / $(64)) % $(288)) * $(32)) + (((r21_z % $(64)) - $(32)) % $(32))), dtype=tl.float32))
    r21_value = r21_value.to(tl.float32)
    r21_value = tl.broadcast_to(r21_value, [$(18432)])
    tl.store($(vcat_1_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(18432))
    r22_value = tl.cos(tl.load($(vcat_1_ptr) + ((((r22_z / $(64)) % $(288)) * $(64)) + (r22_z % $(64))), dtype=tl.float32))
    r22_value = r22_value.to(tl.float32)
    r22_value = tl.broadcast_to(r22_value, [$(18432)])
    tl.store($(vcos_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(18432))
    r23_value = tl.load($(vcos_ptr) + (((((((tl.load($(vcat_ptr) + ((r23_z / $(64)) % $(288)), dtype=tl.int64) * $(64)) + (r23_z % $(64))) / $(64)) % $(288)) % $(288)) * $(64)) + ((((tl.load($(vcat_ptr) + ((r23_z / $(64)) % $(288)), dtype=tl.int64) * $(64)) + (r23_z % $(64))) % $(64)) % $(64))), dtype=tl.float32)
    r23_value = r23_value.to(tl.float32)
    r23_value = tl.broadcast_to(r23_value, [$(18432)])
    tl.store($(vindex_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(18432))
    r24_value = tl.sin(tl.load($(vcat_1_ptr) + ((((r24_z / $(64)) % $(288)) * $(64)) + (r24_z % $(64))), dtype=tl.float32))
    r24_value = r24_value.to(tl.float32)
    r24_value = tl.broadcast_to(r24_value, [$(18432)])
    tl.store($(vsin_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(18432))
    r25_value = tl.load($(vsin_ptr) + (((((((tl.load($(vcat_ptr) + ((r25_z / $(64)) % $(288)), dtype=tl.int64) * $(64)) + (r25_z % $(64))) / $(64)) % $(288)) % $(288)) * $(64)) + ((((tl.load($(vcat_ptr) + ((r25_z / $(64)) % $(288)), dtype=tl.int64) * $(64)) + (r25_z % $(64))) % $(64)) % $(64))), dtype=tl.float32)
    r25_value = r25_value.to(tl.float32)
    r25_value = tl.broadcast_to(r25_value, [$(18432)])
    tl.store($(vindex_1_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(18432))
    r26_value = tl.load($(vindex_ptr) + ((((r26_z / $(64)) % $(288)) * $(64)) + (r26_z % $(64))), dtype=tl.float32)
    r26_value = r26_value.to(tl.bfloat16)
    r26_value = tl.broadcast_to(r26_value, [$(18432)])
    tl.store($(vto_copy_14_ptr) + r26_z, r26_value)
    r27_z = tl.arange($(18432))
    r27_value = tl.load($(vindex_1_ptr) + ((((r27_z / $(64)) % $(288)) * $(64)) + (r27_z % $(64))), dtype=tl.float32)
    r27_value = r27_value.to(tl.bfloat16)
    r27_value = tl.broadcast_to(r27_value, [$(18432)])
    tl.store($(vto_copy_15_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(1))
    r28_value = (tl.load($(vcat_ptr) + $(0), dtype=tl.int64) - tl.full([], $(1), dtype=tl.int64))
    r28_value = tl.broadcast_to(r28_value, [$(1)])
    tl.store($(vsub_ptr) + r28_z, r28_value)
    r29_z = tl.arange($(289))
    r29_value = tl.where(r29_z < $(1), tl.load($(vsub_ptr) + $(0), dtype=tl.int64), tl.load($(vcat_ptr) + ((r29_z - $(1)) % $(288)), dtype=tl.int64))
    r29_value = tl.broadcast_to(r29_value, [$(289)])
    tl.store($(vcat_2_ptr) + r29_z, r29_value)
    r30_z = tl.arange($(288))
    r30_value = (tl.load($(vcat_2_ptr) + (r30_z + $(1)), dtype=tl.int64) - tl.load($(vcat_2_ptr) + r30_z, dtype=tl.int64))
    r30_value = tl.broadcast_to(r30_value, [$(288)])
    tl.store($(vsub_1_ptr) + r30_z, r30_value)
    r31_z = tl.arange($(288))
    r31_value = (tl.load($(vsub_1_ptr) + r31_z, dtype=tl.int64) != tl.full([], $(1), dtype=tl.int64))
    r31_value = tl.broadcast_to(r31_value, [$(288)])
    tl.store($(vne_ptr) + r31_z, r31_value)
    r32_z = tl.arange($(288))
    r32_value = tl.broadcast_to(tl.full([], $(0), dtype=tl.int64), [$(288)])
    for r32_lane in range($(288)) {
      r32_value = tl.where(r32_lane <= r32_z, (r32_value + tl.where(tl.load($(vne_ptr) + (r32_lane % $(288)), dtype=tl.int1), tl.full([], $(1), dtype=tl.int64), tl.full([], $(0), dtype=tl.int64))), r32_value)
    }
    r32_value = tl.broadcast_to(r32_value, [$(288)])
    tl.store($(vcumsum_1_ptr) + r32_z, r32_value)
    r33_z = tl.arange($(82944))
    r33_value = (tl.load($(vcumsum_1_ptr) + ((r33_z / $(288)) % $(288)), dtype=tl.int64) == tl.load($(vcumsum_1_ptr) + (r33_z % $(288)), dtype=tl.int64))
    r33_value = tl.broadcast_to(r33_value, [$(82944)])
    tl.store($(veq_ptr) + r33_z, r33_value)
    r34_z = tl.arange($(82944))
    r34_value = tl.where((r34_z % $(288)) <= ((r34_z / $(288)) % $(288)) + $(0), true, false)
    r34_value = tl.broadcast_to(r34_value, [$(82944)])
    tl.store($(vtril_ptr) + r34_z, r34_value)
    r35_z = tl.arange($(82944))
    r35_value = (tl.load($(veq_ptr) + ((((r35_z / $(288)) % $(288)) * $(288)) + (r35_z % $(288))), dtype=tl.int1) & tl.load($(vtril_ptr) + ((((r35_z / $(288)) % $(288)) * $(288)) + (r35_z % $(288))), dtype=tl.int1))
    r35_value = tl.broadcast_to(r35_value, [$(82944)])
    tl.store($(vbitwise_and_ptr) + r35_z, r35_value)
    r36_z = tl.arange($(82944))
    r36_value = tl.where(tl.load($(vbitwise_and_ptr) + ((((r36_z / $(288)) % $(288)) * $(288)) + (r36_z % $(288))), dtype=tl.int1), tl.cast(0.0, tl.bfloat16), tl.cast(-338953138925153547590470800371487866880.0, tl.bfloat16))
    r36_value = r36_value.to(tl.bfloat16)
    r36_value = tl.broadcast_to(r36_value, [$(82944)])
    tl.store($(vwhere_ptr) + r36_z, r36_value)
    r37_z = tl.arange($(96))
    r37_value = tl.where(r37_z < $(32), (tl.full([], $(64), dtype=tl.int64) + tl.cast((r37_z - $(0)), tl.int64)), tl.where(r37_z < $(64), (tl.full([], $(128), dtype=tl.int64) + tl.cast((r37_z - $(32)), tl.int64)), (tl.full([], $(192), dtype=tl.int64) + tl.cast((r37_z - $(64)), tl.int64))))
    r37_value = tl.broadcast_to(r37_value, [$(96)])
    tl.store($(vcat_4_ptr) + r37_z, r37_value)
    r38_z = tl.arange($(1048576))
    r38_value = tl.load($(vmul_5_ptr) + ((((r38_z / $(512)) % $(2048)) * $(512)) + (r38_z % $(512))), dtype=tl.float32)
    r38_value = r38_value.to(tl.bfloat16)
    r38_value = tl.broadcast_to(r38_value, [$(1048576)])
    tl.store($(vto_copy_18_ptr) + r38_z, r38_value)
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
    tl.store($(vto_copy_19_ptr) + r40_z, r40_value)
    r41_z = tl.arange($(73728))
    r41_value = ((tl.load($(vto_copy_19_ptr) + ((((r41_z / $(256)) % $(288)) * $(256)) + (r41_z % $(256))), dtype=tl.float32)) * (tl.load($(vto_copy_19_ptr) + ((((r41_z / $(256)) % $(288)) * $(256)) + (r41_z % $(256))), dtype=tl.float32)))
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
    r45_value = (tl.load($(vto_copy_19_ptr) + ((((r45_z / $(256)) % $(288)) * $(256)) + (r45_z % $(256))), dtype=tl.float32) * tl.load($(vrsqrt_ptr) + ((r45_z / $(256)) % $(288)), dtype=tl.float32))
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
    tl.store($(vto_copy_20_ptr) + r47_z, r47_value)
    r48_row = tl.arange($(288))
    r48_col = tl.arange($(1536))
    r48_lane = tl.arange($(256))
    r48_left = tl.load($(vto_copy_20_ptr) + ((((((r48_row[:, None] * $(256)) + r48_lane[None, :]) / $(256)) % $(288)) * $(256)) + (((r48_row[:, None] * $(256)) + r48_lane[None, :]) % $(256))), dtype=tl.bfloat16)
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
    tl.store($(vto_copy_21_ptr) + r50_z, r50_value)
    r51_z = tl.arange($(147456))
    r51_value = ((tl.load($(vto_copy_21_ptr) + ((((r51_z / $(512)) % $(288)) * $(512)) + (r51_z % $(512))), dtype=tl.float32)) * (tl.load($(vto_copy_21_ptr) + ((((r51_z / $(512)) % $(288)) * $(512)) + (r51_z % $(512))), dtype=tl.float32)))
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
    r55_value = (tl.load($(vto_copy_21_ptr) + ((((r55_z / $(512)) % $(288)) * $(512)) + (r55_z % $(512))), dtype=tl.float32) * tl.load($(vrsqrt_1_ptr) + ((r55_z / $(512)) % $(288)), dtype=tl.float32))
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
    tl.store($(vto_copy_22_ptr) + r57_z, r57_value)
    r58_row = tl.arange($(288))
    r58_col = tl.arange($(2048))
    r58_lane = tl.arange($(512))
    r58_left = tl.load($(vto_copy_22_ptr) + ((((((r58_row[:, None] * $(512)) + r58_lane[None, :]) / $(512)) % $(288)) * $(512)) + (((r58_row[:, None] * $(512)) + r58_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r58_right = tl.load($(vto_copy_18_ptr) + (((r58_col[None, :] % $(2048)) * $(512)) + (r58_lane[:, None] % $(512))), dtype=tl.bfloat16)
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
    tl.store($(vto_copy_23_ptr) + r78_z, r78_value)
    r79_z = tl.arange($(663552))
    r79_lane = tl.arange($(288))
    r79_row_values = tl.load($(vto_copy_23_ptr) + ((((r79_z[:, None] / $(82944)) % $(8)) * $(82944)) + (((r79_z[:, None] / $(288)) % $(288)) * $(288)) + (r79_lane[None, :] % $(288))), dtype=tl.float32)
    r79_maximum = tl.max(r79_row_values, axis=1, keep_dims=true)
    r79_weights = tl.exp(r79_row_values - r79_maximum)
    r79_total = tl.sum(r79_weights, axis=1)
    r79_maximum = tl.max(r79_row_values, axis=1)
    r79_value = tl.exp(tl.load($(vto_copy_23_ptr) + ((((r79_z / $(82944)) % $(8)) * $(82944)) + (((r79_z / $(288)) % $(288)) * $(288)) + (r79_z % $(288))), dtype=tl.float32) - r79_maximum) / r79_total
    r79_value = r79_value.to(tl.float32)
    r79_value = tl.broadcast_to(r79_value, [$(663552)])
    tl.store($(vsoftmax_ptr) + r79_z, r79_value)
    r80_z = tl.arange($(663552))
    r80_value = tl.load($(vsoftmax_ptr) + ((((r80_z / $(82944)) % $(8)) * $(82944)) + (((r80_z / $(288)) % $(288)) * $(288)) + (r80_z % $(288))), dtype=tl.float32)
    r80_value = r80_value.to(tl.bfloat16)
    r80_value = tl.broadcast_to(r80_value, [$(663552)])
    tl.store($(vto_copy_24_ptr) + r80_z, r80_value)
    r81_z = tl.arange($(147456))
    r81_value = tl.cast(0.0, tl.bfloat16)
    for r81_lane in range($(96)) {
      r81_value = tl.where(tl.load($(vcat_4_ptr) + (r81_lane % $(96)), dtype=tl.int64) == ((r81_z / $(512)) % $(288)), r81_value + tl.load($(input1_ptr) + (((r81_lane % $(96)) * $(512)) + (r81_z % $(512))), dtype=tl.bfloat16), r81_value)
    }
    r81_value = r81_value.to(tl.bfloat16)
    r81_value = tl.broadcast_to(r81_value, [$(147456)])
    tl.store($(vindex_put_ptr) + r81_z, r81_value)
    r82_row = tl.arange($(288))
    r82_col = tl.arange($(1024))
    r82_lane = tl.arange($(512))
    r82_left = tl.load($(vindex_put_ptr) + ((((((r82_row[:, None] * $(512)) + r82_lane[None, :]) / $(512)) % $(288)) * $(512)) + (((r82_row[:, None] * $(512)) + r82_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r82_right = tl.load($(vto_copy_6_ptr) + (((r82_lane[:, None] % $(512)) * $(1024)) + (r82_col[None, :] % $(1024))), dtype=tl.bfloat16)
    r82_value = tl.dot(r82_left, r82_right)
    r82_value = r82_value.to(tl.bfloat16)
    tl.store($(vmm_5_ptr) + ((r82_row[:, None] * $(1024)) + r82_col[None, :]), r82_value)
    r83_row = tl.arange($(288))
    r83_col = tl.arange($(128))
    r83_lane = tl.arange($(288))
    for r83_group in range($(8)) {
      r83_left = tl.load($(vto_copy_24_ptr) + (((((((r83_group % $(8)) * $(82944)) + (r83_lane[None, :] * $(288)) + r83_row[:, None]) / $(82944)) % $(8)) * $(82944)) + ((((((r83_group % $(8)) * $(82944)) + (r83_lane[None, :] * $(288)) + r83_row[:, None]) / $(288)) % $(288)) * $(288)) + ((((r83_group % $(8)) * $(82944)) + (r83_lane[None, :] * $(288)) + r83_row[:, None]) % $(288))), dtype=tl.bfloat16)
      r83_right = tl.load($(vmm_5_ptr) + (((((((r83_group % $(8)) * $(36864)) + (r83_lane[:, None] * $(128)) + r83_col[None, :]) / $(128)) % $(288)) * $(1024)) + ((((((r83_group % $(8)) * $(36864)) + (r83_lane[:, None] * $(128)) + r83_col[None, :]) / $(36864)) % $(8)) * $(128)) + ((((r83_group % $(8)) * $(36864)) + (r83_lane[:, None] * $(128)) + r83_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r83_value = tl.dot(r83_left, r83_right)
      r83_value = r83_value.to(tl.bfloat16)
      tl.store($(vbmm_2_ptr) + ((r83_group * $(36864)) + (r83_row[:, None] * $(128)) + r83_col[None, :]), r83_value)
    }
    r84_row = tl.arange($(288))
    r84_col = tl.arange($(288))
    r84_lane = tl.arange($(128))
    for r84_group in range($(8)) {
      r84_left = tl.load($(vmm_5_ptr) + (((((((r84_group % $(8)) * $(36864)) + (r84_row[:, None] * $(128)) + r84_lane[None, :]) / $(128)) % $(288)) * $(1024)) + ((((((r84_group % $(8)) * $(36864)) + (r84_row[:, None] * $(128)) + r84_lane[None, :]) / $(36864)) % $(8)) * $(128)) + ((((r84_group % $(8)) * $(36864)) + (r84_row[:, None] * $(128)) + r84_lane[None, :]) % $(128))), dtype=tl.bfloat16)
      r84_right = tl.load($(vmm_3_ptr) + (((((((r84_group % $(8)) * $(36864)) + (r84_col[None, :] * $(128)) + r84_lane[:, None]) / $(128)) % $(288)) * $(2048)) + ((((((r84_group % $(8)) * $(36864)) + (r84_col[None, :] * $(128)) + r84_lane[:, None]) / $(36864)) % $(8)) * $(256)) + ((((r84_group % $(8)) * $(36864)) + (r84_col[None, :] * $(128)) + r84_lane[:, None]) % $(128)) + $(128)), dtype=tl.bfloat16)
      r84_value = tl.dot(r84_left, r84_right)
      r84_value = r84_value.to(tl.bfloat16)
      tl.store($(vbmm_3_ptr) + ((r84_group * $(82944)) + (r84_row[:, None] * $(288)) + r84_col[None, :]), r84_value)
    }
    r85_z = tl.arange($(663552))
    r85_value = tl.load($(vbmm_3_ptr) + ((((r85_z / $(82944)) % $(8)) * $(82944)) + (((r85_z / $(288)) % $(288)) * $(288)) + (r85_z % $(288))), dtype=tl.bfloat16)
    r85_value = r85_value.to(tl.float32)
    r85_value = tl.broadcast_to(r85_value, [$(663552)])
    tl.store($(vto_copy_25_ptr) + r85_z, r85_value)
    r86_z = tl.arange($(663552))
    r86_lane = tl.arange($(288))
    r86_row_values = tl.load($(vsoftmax_ptr) + ((((r86_z[:, None] / $(82944)) % $(8)) * $(82944)) + (((r86_z[:, None] / $(288)) % $(288)) * $(288)) + (r86_lane[None, :] % $(288))), dtype=tl.float32)
    r86_row_grad = tl.load($(vto_copy_25_ptr) + ((((r86_z[:, None] / $(82944)) % $(8)) * $(82944)) + (((r86_z[:, None] / $(288)) % $(288)) * $(288)) + (r86_lane[None, :] % $(288))), dtype=tl.float32)
    r86_weighted = r86_row_grad * r86_row_values
    r86_dot_grad = tl.sum(r86_weighted, axis=1)
    r86_value = (tl.load($(vto_copy_25_ptr) + ((((r86_z / $(82944)) % $(8)) * $(82944)) + (((r86_z / $(288)) % $(288)) * $(288)) + (r86_z % $(288))), dtype=tl.float32) - r86_dot_grad) * tl.load($(vsoftmax_ptr) + ((((r86_z / $(82944)) % $(8)) * $(82944)) + (((r86_z / $(288)) % $(288)) * $(288)) + (r86_z % $(288))), dtype=tl.float32)
    r86_value = r86_value.to(tl.float32)
    r86_value = tl.broadcast_to(r86_value, [$(663552)])
    tl.store($(vsoftmax_backward_data_ptr) + r86_z, r86_value)
    r87_z = tl.arange($(663552))
    r87_value = tl.load($(vsoftmax_backward_data_ptr) + ((((r87_z / $(82944)) % $(8)) * $(82944)) + (((r87_z / $(288)) % $(288)) * $(288)) + (r87_z % $(288))), dtype=tl.float32)
    r87_value = r87_value.to(tl.bfloat16)
    r87_value = tl.broadcast_to(r87_value, [$(663552)])
    tl.store($(vto_copy_26_ptr) + r87_z, r87_value)
    r88_z = tl.arange($(663552))
    r88_value = (tl.load($(vto_copy_26_ptr) + ((((r88_z / $(82944)) % $(8)) * $(82944)) + (((r88_z / $(288)) % $(288)) * $(288)) + (r88_z % $(288))), dtype=tl.bfloat16) * tl.cast(0.07216878364870322, tl.bfloat16))
    r88_value = r88_value.to(tl.bfloat16)
    r88_value = tl.broadcast_to(r88_value, [$(663552)])
    tl.store($(vmul_22_ptr) + r88_z, r88_value)
    r89_row = tl.arange($(192))
    r89_col = tl.arange($(288))
    r89_lane = tl.arange($(288))
    for r89_group in range($(8)) {
      r89_left = tl.load($(vcat_7_ptr) + (((((((r89_group % $(8)) * $(55296)) + (r89_lane[None, :] * $(192)) + r89_row[:, None]) / $(55296)) % $(8)) * $(55296)) + ((((((r89_group % $(8)) * $(55296)) + (r89_lane[None, :] * $(192)) + r89_row[:, None]) / $(192)) % $(288)) * $(192)) + ((((r89_group % $(8)) * $(55296)) + (r89_lane[None, :] * $(192)) + r89_row[:, None]) % $(192))), dtype=tl.bfloat16)
      r89_right = tl.load($(vmul_22_ptr) + (((((((r89_group % $(8)) * $(82944)) + (r89_lane[:, None] * $(288)) + r89_col[None, :]) / $(82944)) % $(8)) * $(82944)) + ((((((r89_group % $(8)) * $(82944)) + (r89_lane[:, None] * $(288)) + r89_col[None, :]) / $(288)) % $(288)) * $(288)) + ((((r89_group % $(8)) * $(82944)) + (r89_lane[:, None] * $(288)) + r89_col[None, :]) % $(288))), dtype=tl.bfloat16)
      r89_value = tl.dot(r89_left, r89_right)
      r89_value = r89_value.to(tl.bfloat16)
      tl.store($(vbmm_4_ptr) + ((r89_group * $(55296)) + (r89_row[:, None] * $(288)) + r89_col[None, :]), r89_value)
    }
    r90_z = tl.arange($(589824))
    r90_value = tl.where((r90_z % $(256)) < $(128), tl.load($(vbmm_4_ptr) + ((((((((r90_z / $(73728)) % $(8)) * $(55296)) + (((r90_z % $(256)) - $(0)) * $(288)) + ((r90_z / $(256)) % $(288))) / $(55296)) % $(8)) * $(55296)) + ((((((r90_z / $(73728)) % $(8)) * $(192)) + ((r90_z % $(256)) - $(0))) % $(192)) * $(288)) + ((r90_z / $(256)) % $(288))), dtype=tl.bfloat16), tl.load($(vbmm_2_ptr) + ((((((((r90_z / $(73728)) % $(8)) * $(36864)) + (((r90_z / $(256)) % $(288)) * $(128)) + ((r90_z % $(256)) - $(128))) / $(36864)) % $(8)) * $(36864)) + (((((((r90_z / $(73728)) % $(8)) * $(36864)) + (((r90_z / $(256)) % $(288)) * $(128)) + ((r90_z % $(256)) - $(128))) / $(128)) % $(288)) * $(128)) + (((((r90_z / $(73728)) % $(8)) * $(36864)) + (((r90_z / $(256)) % $(288)) * $(128)) + ((r90_z % $(256)) - $(128))) % $(128))), dtype=tl.bfloat16))
    r90_value = r90_value.to(tl.bfloat16)
    r90_value = tl.broadcast_to(r90_value, [$(589824)])
    tl.store($(vcat_9_ptr) + r90_z, r90_value)
    r91_row = tl.arange($(2048))
    r91_col = tl.arange($(512))
    r91_lane = tl.arange($(288))
    r91_left = tl.load($(vcat_9_ptr) + ((((((((((r91_lane[None, :] * $(2048)) + r91_row[:, None]) / $(2048)) % $(288)) * $(2048)) + (((r91_lane[None, :] * $(2048)) + r91_row[:, None]) % $(2048))) / $(256)) % $(8)) * $(73728)) + (((((r91_lane[None, :] * $(2048)) + r91_row[:, None]) / $(2048)) % $(288)) * $(256)) + (((((((r91_lane[None, :] * $(2048)) + r91_row[:, None]) / $(2048)) % $(288)) * $(2048)) + (((r91_lane[None, :] * $(2048)) + r91_row[:, None]) % $(2048))) % $(256))), dtype=tl.bfloat16)
    r91_right = tl.load($(vto_copy_22_ptr) + ((((((r91_lane[:, None] * $(512)) + r91_col[None, :]) / $(512)) % $(288)) * $(512)) + (((r91_lane[:, None] * $(512)) + r91_col[None, :]) % $(512))), dtype=tl.bfloat16)
    r91_value = tl.dot(r91_left, r91_right)
    r91_value = r91_value.to(tl.bfloat16)
    tl.store($(vmm_6_ptr) + ((r91_row[:, None] * $(512)) + r91_col[None, :]), r91_value)
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
  let vto_copy_18_ptr : RegionName := "vto_copy_18"
  let vmm_ptr : RegionName := "vmm"
  let vto_copy_19_ptr : RegionName := "vto_copy_19"
  let vpow_2_ptr : RegionName := "vpow_2"
  let vmean_ptr : RegionName := "vmean"
  let vadd_2_ptr : RegionName := "vadd_2"
  let vrsqrt_ptr : RegionName := "vrsqrt"
  let vmul_9_ptr : RegionName := "vmul_9"
  let vmul_10_ptr : RegionName := "vmul_10"
  let vto_copy_20_ptr : RegionName := "vto_copy_20"
  let vmm_1_ptr : RegionName := "vmm_1"
  let vmm_2_ptr : RegionName := "vmm_2"
  let vto_copy_21_ptr : RegionName := "vto_copy_21"
  let vpow_3_ptr : RegionName := "vpow_3"
  let vmean_1_ptr : RegionName := "vmean_1"
  let vadd_3_ptr : RegionName := "vadd_3"
  let vrsqrt_1_ptr : RegionName := "vrsqrt_1"
  let vmul_11_ptr : RegionName := "vmul_11"
  let vmul_12_ptr : RegionName := "vmul_12"
  let vto_copy_22_ptr : RegionName := "vto_copy_22"
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
  let vindex_put_ptr : RegionName := "vindex_put"
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
  let candidate_vto_copy_18_ptr : RegionName := "candidate_vto_copy_18"
  let candidate_vmm_ptr : RegionName := "candidate_vmm"
  let candidate_vto_copy_19_ptr : RegionName := "candidate_vto_copy_19"
  let candidate_vpow_2_ptr : RegionName := "candidate_vpow_2"
  let candidate_vmean_ptr : RegionName := "candidate_vmean"
  let candidate_vadd_2_ptr : RegionName := "candidate_vadd_2"
  let candidate_vrsqrt_ptr : RegionName := "candidate_vrsqrt"
  let candidate_vmul_9_ptr : RegionName := "candidate_vmul_9"
  let candidate_vmul_10_ptr : RegionName := "candidate_vmul_10"
  let candidate_vto_copy_20_ptr : RegionName := "candidate_vto_copy_20"
  let candidate_vmm_1_ptr : RegionName := "candidate_vmm_1"
  let candidate_vmm_2_ptr : RegionName := "candidate_vmm_2"
  let candidate_vto_copy_21_ptr : RegionName := "candidate_vto_copy_21"
  let candidate_vpow_3_ptr : RegionName := "candidate_vpow_3"
  let candidate_vmean_1_ptr : RegionName := "candidate_vmean_1"
  let candidate_vadd_3_ptr : RegionName := "candidate_vadd_3"
  let candidate_vrsqrt_1_ptr : RegionName := "candidate_vrsqrt_1"
  let candidate_vmul_11_ptr : RegionName := "candidate_vmul_11"
  let candidate_vmul_12_ptr : RegionName := "candidate_vmul_12"
  let candidate_vto_copy_22_ptr : RegionName := "candidate_vto_copy_22"
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
  let candidate_vindex_put_ptr : RegionName := "candidate_vindex_put"
  let candidate_vmm_22_ptr : RegionName := "candidate_vmm_22"
  let candidate_vbmm_27_ptr : RegionName := "candidate_vbmm_27"
  let candidate_vbmm_28_ptr : RegionName := "candidate_vbmm_28"
  let candidate_vcat_18_ptr : RegionName := "candidate_vcat_18"
  let candidate_vcat_19_ptr : RegionName := "candidate_vcat_19"
  let candidate_vto_copy_68_ptr : RegionName := "candidate_vto_copy_68"
  let candidate_vto_copy_69_ptr : RegionName := "candidate_vto_copy_69"
  let candidate_vbmm_29_ptr : RegionName := "candidate_vbmm_29"
  let candidate_vmul_30_ptr : RegionName := "candidate_vmul_30"
  let candidate_vadd_24_ptr : RegionName := "candidate_vadd_24"
  let candidate_vsub_24_ptr : RegionName := "candidate_vsub_24"
  let candidate_vgt_9_ptr : RegionName := "candidate_vgt_9"
  let candidate_vmasked_fill_9_ptr : RegionName := "candidate_vmasked_fill_9"
  let candidate_vsoftmax_9_ptr : RegionName := "candidate_vsoftmax_9"
  let candidate_vto_copy_70_ptr : RegionName := "candidate_vto_copy_70"
  let candidate_vbmm_30_ptr : RegionName := "candidate_vbmm_30"
  let candidate_vlogsumexp_9_ptr : RegionName := "candidate_vlogsumexp_9"
  let candidate_vcat_20_ptr : RegionName := "candidate_vcat_20"
  let candidate_vcat_21_ptr : RegionName := "candidate_vcat_21"
  let candidate_vto_copy_71_ptr : RegionName := "candidate_vto_copy_71"
  let candidate_vto_copy_72_ptr : RegionName := "candidate_vto_copy_72"
  let candidate_vto_copy_73_ptr : RegionName := "candidate_vto_copy_73"
  let candidate_vto_copy_74_ptr : RegionName := "candidate_vto_copy_74"
  let candidate_vto_copy_75_ptr : RegionName := "candidate_vto_copy_75"
  let candidate_vbmm_31_ptr : RegionName := "candidate_vbmm_31"
  let candidate_vmul_31_ptr : RegionName := "candidate_vmul_31"
  let candidate_vadd_25_ptr : RegionName := "candidate_vadd_25"
  let candidate_vsub_25_ptr : RegionName := "candidate_vsub_25"
  let candidate_vgt_10_ptr : RegionName := "candidate_vgt_10"
  let candidate_vmasked_fill_10_ptr : RegionName := "candidate_vmasked_fill_10"
  let candidate_vsub_26_ptr : RegionName := "candidate_vsub_26"
  let candidate_vexp_ptr : RegionName := "candidate_vexp"
  let candidate_vmul_32_ptr : RegionName := "candidate_vmul_32"
  let candidate_vsum_1_ptr : RegionName := "candidate_vsum_1"
  let candidate_vbmm_32_ptr : RegionName := "candidate_vbmm_32"
  let candidate_vsub_27_ptr : RegionName := "candidate_vsub_27"
  let candidate_vmul_33_ptr : RegionName := "candidate_vmul_33"
  let candidate_vbmm_34_ptr : RegionName := "candidate_vbmm_34"
  let candidate_vmul_35_ptr : RegionName := "candidate_vmul_35"
  let candidate_vto_copy_77_ptr : RegionName := "candidate_vto_copy_77"
  let candidate_vbmm_35_ptr : RegionName := "candidate_vbmm_35"
  let candidate_vto_copy_78_ptr : RegionName := "candidate_vto_copy_78"
  let candidate_vcat_23_ptr : RegionName := "candidate_vcat_23"
  let candidate_vcat_24_ptr : RegionName := "candidate_vcat_24"
  let candidate_vbmm_38_ptr : RegionName := "candidate_vbmm_38"
  let candidate_vbmm_39_ptr : RegionName := "candidate_vbmm_39"
  let candidate_vcat_25_ptr : RegionName := "candidate_vcat_25"
  let candidate_vbmm_40_ptr : RegionName := "candidate_vbmm_40"
  let candidate_vbmm_41_ptr : RegionName := "candidate_vbmm_41"
  let candidate_vcat_26_ptr : RegionName := "candidate_vcat_26"
  let candidate_vcat_27_ptr : RegionName := "candidate_vcat_27"
  let candidate_vto_copy_79_ptr : RegionName := "candidate_vto_copy_79"
  let candidate_vto_copy_80_ptr : RegionName := "candidate_vto_copy_80"
  let candidate_vbmm_42_ptr : RegionName := "candidate_vbmm_42"
  let candidate_vmul_36_ptr : RegionName := "candidate_vmul_36"
  let candidate_vadd_27_ptr : RegionName := "candidate_vadd_27"
  let candidate_vsub_28_ptr : RegionName := "candidate_vsub_28"
  let candidate_vgt_11_ptr : RegionName := "candidate_vgt_11"
  let candidate_vmasked_fill_11_ptr : RegionName := "candidate_vmasked_fill_11"
  let candidate_vsoftmax_10_ptr : RegionName := "candidate_vsoftmax_10"
  let candidate_vto_copy_81_ptr : RegionName := "candidate_vto_copy_81"
  let candidate_vbmm_43_ptr : RegionName := "candidate_vbmm_43"
  let candidate_vlogsumexp_10_ptr : RegionName := "candidate_vlogsumexp_10"
  let candidate_vcat_28_ptr : RegionName := "candidate_vcat_28"
  let candidate_vcat_29_ptr : RegionName := "candidate_vcat_29"
  let candidate_vto_copy_82_ptr : RegionName := "candidate_vto_copy_82"
  let candidate_vto_copy_83_ptr : RegionName := "candidate_vto_copy_83"
  let candidate_vto_copy_84_ptr : RegionName := "candidate_vto_copy_84"
  let candidate_vto_copy_85_ptr : RegionName := "candidate_vto_copy_85"
  let candidate_vto_copy_86_ptr : RegionName := "candidate_vto_copy_86"
  let candidate_vbmm_44_ptr : RegionName := "candidate_vbmm_44"
  let candidate_vmul_37_ptr : RegionName := "candidate_vmul_37"
  let candidate_vadd_28_ptr : RegionName := "candidate_vadd_28"
  let candidate_vsub_29_ptr : RegionName := "candidate_vsub_29"
  let candidate_vgt_12_ptr : RegionName := "candidate_vgt_12"
  let candidate_vmasked_fill_12_ptr : RegionName := "candidate_vmasked_fill_12"
  let candidate_vsub_30_ptr : RegionName := "candidate_vsub_30"
  let candidate_vexp_1_ptr : RegionName := "candidate_vexp_1"
  let candidate_vmul_38_ptr : RegionName := "candidate_vmul_38"
  let candidate_vsum_3_ptr : RegionName := "candidate_vsum_3"
  let candidate_vbmm_45_ptr : RegionName := "candidate_vbmm_45"
  let candidate_vsub_31_ptr : RegionName := "candidate_vsub_31"
  let candidate_vmul_39_ptr : RegionName := "candidate_vmul_39"
  let candidate_vbmm_47_ptr : RegionName := "candidate_vbmm_47"
  let candidate_vmul_41_ptr : RegionName := "candidate_vmul_41"
  let candidate_vto_copy_88_ptr : RegionName := "candidate_vto_copy_88"
  let candidate_vbmm_48_ptr : RegionName := "candidate_vbmm_48"
  let candidate_vto_copy_89_ptr : RegionName := "candidate_vto_copy_89"
  let candidate_vcat_30_ptr : RegionName := "candidate_vcat_30"
  let candidate_vcat_31_ptr : RegionName := "candidate_vcat_31"
  let candidate_vcat_33_ptr : RegionName := "candidate_vcat_33"
  let candidate_vcat_34_ptr : RegionName := "candidate_vcat_34"
  let candidate_vbmm_51_ptr : RegionName := "candidate_vbmm_51"
  let candidate_vbmm_52_ptr : RegionName := "candidate_vbmm_52"
  let candidate_vcat_35_ptr : RegionName := "candidate_vcat_35"
  let candidate_vadd_30_ptr : RegionName := "candidate_vadd_30"
  let candidate_vbmm_53_ptr : RegionName := "candidate_vbmm_53"
  let candidate_vbmm_54_ptr : RegionName := "candidate_vbmm_54"
  let candidate_vcat_36_ptr : RegionName := "candidate_vcat_36"
  let candidate_vcat_37_ptr : RegionName := "candidate_vcat_37"
  let candidate_vto_copy_90_ptr : RegionName := "candidate_vto_copy_90"
  let candidate_vto_copy_91_ptr : RegionName := "candidate_vto_copy_91"
  let candidate_vbmm_55_ptr : RegionName := "candidate_vbmm_55"
  let candidate_vmul_42_ptr : RegionName := "candidate_vmul_42"
  let candidate_vadd_31_ptr : RegionName := "candidate_vadd_31"
  let candidate_vsub_32_ptr : RegionName := "candidate_vsub_32"
  let candidate_vgt_13_ptr : RegionName := "candidate_vgt_13"
  let candidate_vmasked_fill_13_ptr : RegionName := "candidate_vmasked_fill_13"
  let candidate_vsoftmax_11_ptr : RegionName := "candidate_vsoftmax_11"
  let candidate_vto_copy_92_ptr : RegionName := "candidate_vto_copy_92"
  let candidate_vbmm_56_ptr : RegionName := "candidate_vbmm_56"
  let candidate_vlogsumexp_11_ptr : RegionName := "candidate_vlogsumexp_11"
  let candidate_vcat_38_ptr : RegionName := "candidate_vcat_38"
  let candidate_vcat_39_ptr : RegionName := "candidate_vcat_39"
  let candidate_vto_copy_93_ptr : RegionName := "candidate_vto_copy_93"
  let candidate_vto_copy_94_ptr : RegionName := "candidate_vto_copy_94"
  let candidate_vto_copy_95_ptr : RegionName := "candidate_vto_copy_95"
  let candidate_vto_copy_96_ptr : RegionName := "candidate_vto_copy_96"
  let candidate_vto_copy_97_ptr : RegionName := "candidate_vto_copy_97"
  let candidate_vbmm_57_ptr : RegionName := "candidate_vbmm_57"
  let candidate_vmul_43_ptr : RegionName := "candidate_vmul_43"
  let candidate_vadd_32_ptr : RegionName := "candidate_vadd_32"
  let candidate_vsub_33_ptr : RegionName := "candidate_vsub_33"
  let candidate_vgt_14_ptr : RegionName := "candidate_vgt_14"
  let candidate_vmasked_fill_14_ptr : RegionName := "candidate_vmasked_fill_14"
  let candidate_vsub_34_ptr : RegionName := "candidate_vsub_34"
  let candidate_vexp_2_ptr : RegionName := "candidate_vexp_2"
  let candidate_vmul_44_ptr : RegionName := "candidate_vmul_44"
  let candidate_vsum_5_ptr : RegionName := "candidate_vsum_5"
  let candidate_vbmm_58_ptr : RegionName := "candidate_vbmm_58"
  let candidate_vsub_35_ptr : RegionName := "candidate_vsub_35"
  let candidate_vmul_45_ptr : RegionName := "candidate_vmul_45"
  let candidate_vbmm_60_ptr : RegionName := "candidate_vbmm_60"
  let candidate_vmul_47_ptr : RegionName := "candidate_vmul_47"
  let candidate_vto_copy_99_ptr : RegionName := "candidate_vto_copy_99"
  let candidate_vbmm_61_ptr : RegionName := "candidate_vbmm_61"
  let candidate_vto_copy_100_ptr : RegionName := "candidate_vto_copy_100"
  let candidate_vcat_40_ptr : RegionName := "candidate_vcat_40"
  let candidate_vcat_41_ptr : RegionName := "candidate_vcat_41"
  let candidate_vcat_43_ptr : RegionName := "candidate_vcat_43"
  let candidate_vcat_44_ptr : RegionName := "candidate_vcat_44"
  let candidate_vbmm_64_ptr : RegionName := "candidate_vbmm_64"
  let candidate_vbmm_65_ptr : RegionName := "candidate_vbmm_65"
  let candidate_vcat_45_ptr : RegionName := "candidate_vcat_45"
  let candidate_vadd_34_ptr : RegionName := "candidate_vadd_34"
  let candidate_vbmm_66_ptr : RegionName := "candidate_vbmm_66"
  let candidate_vbmm_67_ptr : RegionName := "candidate_vbmm_67"
  let candidate_vcat_46_ptr : RegionName := "candidate_vcat_46"
  let candidate_vcat_47_ptr : RegionName := "candidate_vcat_47"
  let candidate_vto_copy_101_ptr : RegionName := "candidate_vto_copy_101"
  let candidate_vto_copy_102_ptr : RegionName := "candidate_vto_copy_102"
  let candidate_vbmm_68_ptr : RegionName := "candidate_vbmm_68"
  let candidate_vmul_48_ptr : RegionName := "candidate_vmul_48"
  let candidate_vadd_35_ptr : RegionName := "candidate_vadd_35"
  let candidate_vsub_36_ptr : RegionName := "candidate_vsub_36"
  let candidate_vgt_15_ptr : RegionName := "candidate_vgt_15"
  let candidate_vmasked_fill_15_ptr : RegionName := "candidate_vmasked_fill_15"
  let candidate_vsoftmax_12_ptr : RegionName := "candidate_vsoftmax_12"
  let candidate_vto_copy_103_ptr : RegionName := "candidate_vto_copy_103"
  let candidate_vbmm_69_ptr : RegionName := "candidate_vbmm_69"
  let candidate_vlogsumexp_12_ptr : RegionName := "candidate_vlogsumexp_12"
  let candidate_vcat_48_ptr : RegionName := "candidate_vcat_48"
  let candidate_vcat_49_ptr : RegionName := "candidate_vcat_49"
  let candidate_vto_copy_104_ptr : RegionName := "candidate_vto_copy_104"
  let candidate_vto_copy_105_ptr : RegionName := "candidate_vto_copy_105"
  let candidate_vto_copy_106_ptr : RegionName := "candidate_vto_copy_106"
  let candidate_vto_copy_107_ptr : RegionName := "candidate_vto_copy_107"
  let candidate_vto_copy_108_ptr : RegionName := "candidate_vto_copy_108"
  let candidate_vbmm_70_ptr : RegionName := "candidate_vbmm_70"
  let candidate_vmul_49_ptr : RegionName := "candidate_vmul_49"
  let candidate_vadd_36_ptr : RegionName := "candidate_vadd_36"
  let candidate_vsub_37_ptr : RegionName := "candidate_vsub_37"
  let candidate_vgt_16_ptr : RegionName := "candidate_vgt_16"
  let candidate_vmasked_fill_16_ptr : RegionName := "candidate_vmasked_fill_16"
  let candidate_vsub_38_ptr : RegionName := "candidate_vsub_38"
  let candidate_vexp_3_ptr : RegionName := "candidate_vexp_3"
  let candidate_vmul_50_ptr : RegionName := "candidate_vmul_50"
  let candidate_vsum_7_ptr : RegionName := "candidate_vsum_7"
  let candidate_vbmm_71_ptr : RegionName := "candidate_vbmm_71"
  let candidate_vsub_39_ptr : RegionName := "candidate_vsub_39"
  let candidate_vmul_51_ptr : RegionName := "candidate_vmul_51"
  let candidate_vbmm_73_ptr : RegionName := "candidate_vbmm_73"
  let candidate_vmul_53_ptr : RegionName := "candidate_vmul_53"
  let candidate_vto_copy_110_ptr : RegionName := "candidate_vto_copy_110"
  let candidate_vbmm_74_ptr : RegionName := "candidate_vbmm_74"
  let candidate_vto_copy_111_ptr : RegionName := "candidate_vto_copy_111"
  let candidate_vcat_51_ptr : RegionName := "candidate_vcat_51"
  let candidate_vcat_52_ptr : RegionName := "candidate_vcat_52"
  let candidate_vbmm_77_ptr : RegionName := "candidate_vbmm_77"
  let candidate_vbmm_78_ptr : RegionName := "candidate_vbmm_78"
  let candidate_vcat_53_ptr : RegionName := "candidate_vcat_53"
  let candidate_vadd_38_ptr : RegionName := "candidate_vadd_38"
  let candidate_vbmm_79_ptr : RegionName := "candidate_vbmm_79"
  let candidate_vbmm_80_ptr : RegionName := "candidate_vbmm_80"
  let candidate_vcat_54_ptr : RegionName := "candidate_vcat_54"
  let candidate_vcat_55_ptr : RegionName := "candidate_vcat_55"
  let candidate_vto_copy_112_ptr : RegionName := "candidate_vto_copy_112"
  let candidate_vto_copy_113_ptr : RegionName := "candidate_vto_copy_113"
  let candidate_vbmm_81_ptr : RegionName := "candidate_vbmm_81"
  let candidate_vmul_54_ptr : RegionName := "candidate_vmul_54"
  let candidate_vadd_39_ptr : RegionName := "candidate_vadd_39"
  let candidate_vsub_40_ptr : RegionName := "candidate_vsub_40"
  let candidate_vgt_17_ptr : RegionName := "candidate_vgt_17"
  let candidate_vmasked_fill_17_ptr : RegionName := "candidate_vmasked_fill_17"
  let candidate_vsoftmax_13_ptr : RegionName := "candidate_vsoftmax_13"
  let candidate_vto_copy_114_ptr : RegionName := "candidate_vto_copy_114"
  let candidate_vbmm_82_ptr : RegionName := "candidate_vbmm_82"
  let candidate_vlogsumexp_13_ptr : RegionName := "candidate_vlogsumexp_13"
  let candidate_vcat_56_ptr : RegionName := "candidate_vcat_56"
  let candidate_vcat_57_ptr : RegionName := "candidate_vcat_57"
  let candidate_vto_copy_115_ptr : RegionName := "candidate_vto_copy_115"
  let candidate_vto_copy_116_ptr : RegionName := "candidate_vto_copy_116"
  let candidate_vto_copy_117_ptr : RegionName := "candidate_vto_copy_117"
  let candidate_vto_copy_118_ptr : RegionName := "candidate_vto_copy_118"
  let candidate_vto_copy_119_ptr : RegionName := "candidate_vto_copy_119"
  let candidate_vbmm_83_ptr : RegionName := "candidate_vbmm_83"
  let candidate_vmul_55_ptr : RegionName := "candidate_vmul_55"
  let candidate_vadd_40_ptr : RegionName := "candidate_vadd_40"
  let candidate_vsub_41_ptr : RegionName := "candidate_vsub_41"
  let candidate_vgt_18_ptr : RegionName := "candidate_vgt_18"
  let candidate_vmasked_fill_18_ptr : RegionName := "candidate_vmasked_fill_18"
  let candidate_vsub_42_ptr : RegionName := "candidate_vsub_42"
  let candidate_vexp_4_ptr : RegionName := "candidate_vexp_4"
  let candidate_vmul_56_ptr : RegionName := "candidate_vmul_56"
  let candidate_vsum_9_ptr : RegionName := "candidate_vsum_9"
  let candidate_vbmm_84_ptr : RegionName := "candidate_vbmm_84"
  let candidate_vsub_43_ptr : RegionName := "candidate_vsub_43"
  let candidate_vmul_57_ptr : RegionName := "candidate_vmul_57"
  let candidate_vbmm_86_ptr : RegionName := "candidate_vbmm_86"
  let candidate_vmul_59_ptr : RegionName := "candidate_vmul_59"
  let candidate_vto_copy_121_ptr : RegionName := "candidate_vto_copy_121"
  let candidate_vbmm_87_ptr : RegionName := "candidate_vbmm_87"
  let candidate_vto_copy_122_ptr : RegionName := "candidate_vto_copy_122"
  let candidate_vcat_58_ptr : RegionName := "candidate_vcat_58"
  let candidate_vcat_59_ptr : RegionName := "candidate_vcat_59"
  let candidate_vcat_61_ptr : RegionName := "candidate_vcat_61"
  let candidate_vcat_62_ptr : RegionName := "candidate_vcat_62"
  let candidate_vbmm_90_ptr : RegionName := "candidate_vbmm_90"
  let candidate_vbmm_91_ptr : RegionName := "candidate_vbmm_91"
  let candidate_vcat_63_ptr : RegionName := "candidate_vcat_63"
  let candidate_vadd_42_ptr : RegionName := "candidate_vadd_42"
  let candidate_vbmm_92_ptr : RegionName := "candidate_vbmm_92"
  let candidate_vbmm_93_ptr : RegionName := "candidate_vbmm_93"
  let candidate_vcat_64_ptr : RegionName := "candidate_vcat_64"
  let candidate_vcat_65_ptr : RegionName := "candidate_vcat_65"
  let candidate_vto_copy_123_ptr : RegionName := "candidate_vto_copy_123"
  let candidate_vto_copy_124_ptr : RegionName := "candidate_vto_copy_124"
  let candidate_vbmm_94_ptr : RegionName := "candidate_vbmm_94"
  let candidate_vmul_60_ptr : RegionName := "candidate_vmul_60"
  let candidate_vadd_43_ptr : RegionName := "candidate_vadd_43"
  let candidate_vsub_44_ptr : RegionName := "candidate_vsub_44"
  let candidate_vgt_19_ptr : RegionName := "candidate_vgt_19"
  let candidate_vmasked_fill_19_ptr : RegionName := "candidate_vmasked_fill_19"
  let candidate_vsoftmax_14_ptr : RegionName := "candidate_vsoftmax_14"
  let candidate_vto_copy_125_ptr : RegionName := "candidate_vto_copy_125"
  let candidate_vbmm_95_ptr : RegionName := "candidate_vbmm_95"
  let candidate_vlogsumexp_14_ptr : RegionName := "candidate_vlogsumexp_14"
  let candidate_vcat_66_ptr : RegionName := "candidate_vcat_66"
  let candidate_vcat_67_ptr : RegionName := "candidate_vcat_67"
  let candidate_vto_copy_126_ptr : RegionName := "candidate_vto_copy_126"
  let candidate_vto_copy_127_ptr : RegionName := "candidate_vto_copy_127"
  let candidate_vto_copy_128_ptr : RegionName := "candidate_vto_copy_128"
  let candidate_vto_copy_129_ptr : RegionName := "candidate_vto_copy_129"
  let candidate_vto_copy_130_ptr : RegionName := "candidate_vto_copy_130"
  let candidate_vbmm_96_ptr : RegionName := "candidate_vbmm_96"
  let candidate_vmul_61_ptr : RegionName := "candidate_vmul_61"
  let candidate_vadd_44_ptr : RegionName := "candidate_vadd_44"
  let candidate_vsub_45_ptr : RegionName := "candidate_vsub_45"
  let candidate_vgt_20_ptr : RegionName := "candidate_vgt_20"
  let candidate_vmasked_fill_20_ptr : RegionName := "candidate_vmasked_fill_20"
  let candidate_vsub_46_ptr : RegionName := "candidate_vsub_46"
  let candidate_vexp_5_ptr : RegionName := "candidate_vexp_5"
  let candidate_vmul_62_ptr : RegionName := "candidate_vmul_62"
  let candidate_vsum_11_ptr : RegionName := "candidate_vsum_11"
  let candidate_vbmm_97_ptr : RegionName := "candidate_vbmm_97"
  let candidate_vsub_47_ptr : RegionName := "candidate_vsub_47"
  let candidate_vmul_63_ptr : RegionName := "candidate_vmul_63"
  let candidate_vbmm_99_ptr : RegionName := "candidate_vbmm_99"
  let candidate_vmul_65_ptr : RegionName := "candidate_vmul_65"
  let candidate_vto_copy_132_ptr : RegionName := "candidate_vto_copy_132"
  let candidate_vbmm_100_ptr : RegionName := "candidate_vbmm_100"
  let candidate_vto_copy_133_ptr : RegionName := "candidate_vto_copy_133"
  let candidate_vcat_68_ptr : RegionName := "candidate_vcat_68"
  let candidate_vcat_69_ptr : RegionName := "candidate_vcat_69"
  let candidate_vcat_71_ptr : RegionName := "candidate_vcat_71"
  let candidate_vcat_72_ptr : RegionName := "candidate_vcat_72"
  let candidate_vbmm_103_ptr : RegionName := "candidate_vbmm_103"
  let candidate_vbmm_104_ptr : RegionName := "candidate_vbmm_104"
  let candidate_vcat_73_ptr : RegionName := "candidate_vcat_73"
  let candidate_vadd_46_ptr : RegionName := "candidate_vadd_46"
  let candidate_vbmm_105_ptr : RegionName := "candidate_vbmm_105"
  let candidate_vbmm_106_ptr : RegionName := "candidate_vbmm_106"
  let candidate_vcat_74_ptr : RegionName := "candidate_vcat_74"
  let candidate_vcat_75_ptr : RegionName := "candidate_vcat_75"
  let candidate_vto_copy_134_ptr : RegionName := "candidate_vto_copy_134"
  let candidate_vto_copy_135_ptr : RegionName := "candidate_vto_copy_135"
  let candidate_vbmm_107_ptr : RegionName := "candidate_vbmm_107"
  let candidate_vmul_66_ptr : RegionName := "candidate_vmul_66"
  let candidate_vadd_47_ptr : RegionName := "candidate_vadd_47"
  let candidate_vsub_48_ptr : RegionName := "candidate_vsub_48"
  let candidate_vgt_21_ptr : RegionName := "candidate_vgt_21"
  let candidate_vmasked_fill_21_ptr : RegionName := "candidate_vmasked_fill_21"
  let candidate_vsoftmax_15_ptr : RegionName := "candidate_vsoftmax_15"
  let candidate_vto_copy_136_ptr : RegionName := "candidate_vto_copy_136"
  let candidate_vbmm_108_ptr : RegionName := "candidate_vbmm_108"
  let candidate_vlogsumexp_15_ptr : RegionName := "candidate_vlogsumexp_15"
  let candidate_vcat_76_ptr : RegionName := "candidate_vcat_76"
  let candidate_vcat_77_ptr : RegionName := "candidate_vcat_77"
  let candidate_vto_copy_137_ptr : RegionName := "candidate_vto_copy_137"
  let candidate_vto_copy_138_ptr : RegionName := "candidate_vto_copy_138"
  let candidate_vto_copy_139_ptr : RegionName := "candidate_vto_copy_139"
  let candidate_vto_copy_140_ptr : RegionName := "candidate_vto_copy_140"
  let candidate_vto_copy_141_ptr : RegionName := "candidate_vto_copy_141"
  let candidate_vbmm_109_ptr : RegionName := "candidate_vbmm_109"
  let candidate_vmul_67_ptr : RegionName := "candidate_vmul_67"
  let candidate_vadd_48_ptr : RegionName := "candidate_vadd_48"
  let candidate_vsub_49_ptr : RegionName := "candidate_vsub_49"
  let candidate_vgt_22_ptr : RegionName := "candidate_vgt_22"
  let candidate_vmasked_fill_22_ptr : RegionName := "candidate_vmasked_fill_22"
  let candidate_vsub_50_ptr : RegionName := "candidate_vsub_50"
  let candidate_vexp_6_ptr : RegionName := "candidate_vexp_6"
  let candidate_vmul_68_ptr : RegionName := "candidate_vmul_68"
  let candidate_vsum_13_ptr : RegionName := "candidate_vsum_13"
  let candidate_vbmm_110_ptr : RegionName := "candidate_vbmm_110"
  let candidate_vsub_51_ptr : RegionName := "candidate_vsub_51"
  let candidate_vmul_69_ptr : RegionName := "candidate_vmul_69"
  let candidate_vbmm_112_ptr : RegionName := "candidate_vbmm_112"
  let candidate_vmul_71_ptr : RegionName := "candidate_vmul_71"
  let candidate_vto_copy_143_ptr : RegionName := "candidate_vto_copy_143"
  let candidate_vbmm_113_ptr : RegionName := "candidate_vbmm_113"
  let candidate_vto_copy_144_ptr : RegionName := "candidate_vto_copy_144"
  let candidate_vcat_79_ptr : RegionName := "candidate_vcat_79"
  let candidate_vcat_80_ptr : RegionName := "candidate_vcat_80"
  let candidate_vbmm_116_ptr : RegionName := "candidate_vbmm_116"
  let candidate_vbmm_117_ptr : RegionName := "candidate_vbmm_117"
  let candidate_vcat_81_ptr : RegionName := "candidate_vcat_81"
  let candidate_vadd_50_ptr : RegionName := "candidate_vadd_50"
  let candidate_vbmm_118_ptr : RegionName := "candidate_vbmm_118"
  let candidate_vbmm_119_ptr : RegionName := "candidate_vbmm_119"
  let candidate_vcat_82_ptr : RegionName := "candidate_vcat_82"
  let candidate_vcat_83_ptr : RegionName := "candidate_vcat_83"
  let candidate_vto_copy_145_ptr : RegionName := "candidate_vto_copy_145"
  let candidate_vto_copy_146_ptr : RegionName := "candidate_vto_copy_146"
  let candidate_vbmm_120_ptr : RegionName := "candidate_vbmm_120"
  let candidate_vmul_72_ptr : RegionName := "candidate_vmul_72"
  let candidate_vadd_51_ptr : RegionName := "candidate_vadd_51"
  let candidate_vsub_52_ptr : RegionName := "candidate_vsub_52"
  let candidate_vgt_23_ptr : RegionName := "candidate_vgt_23"
  let candidate_vmasked_fill_23_ptr : RegionName := "candidate_vmasked_fill_23"
  let candidate_vsoftmax_16_ptr : RegionName := "candidate_vsoftmax_16"
  let candidate_vto_copy_147_ptr : RegionName := "candidate_vto_copy_147"
  let candidate_vbmm_121_ptr : RegionName := "candidate_vbmm_121"
  let candidate_vlogsumexp_16_ptr : RegionName := "candidate_vlogsumexp_16"
  let candidate_vcat_84_ptr : RegionName := "candidate_vcat_84"
  let candidate_vcat_85_ptr : RegionName := "candidate_vcat_85"
  let candidate_vto_copy_148_ptr : RegionName := "candidate_vto_copy_148"
  let candidate_vto_copy_149_ptr : RegionName := "candidate_vto_copy_149"
  let candidate_vto_copy_150_ptr : RegionName := "candidate_vto_copy_150"
  let candidate_vto_copy_151_ptr : RegionName := "candidate_vto_copy_151"
  let candidate_vto_copy_152_ptr : RegionName := "candidate_vto_copy_152"
  let candidate_vbmm_122_ptr : RegionName := "candidate_vbmm_122"
  let candidate_vmul_73_ptr : RegionName := "candidate_vmul_73"
  let candidate_vadd_52_ptr : RegionName := "candidate_vadd_52"
  let candidate_vsub_53_ptr : RegionName := "candidate_vsub_53"
  let candidate_vgt_24_ptr : RegionName := "candidate_vgt_24"
  let candidate_vmasked_fill_24_ptr : RegionName := "candidate_vmasked_fill_24"
  let candidate_vsub_54_ptr : RegionName := "candidate_vsub_54"
  let candidate_vexp_7_ptr : RegionName := "candidate_vexp_7"
  let candidate_vmul_74_ptr : RegionName := "candidate_vmul_74"
  let candidate_vsum_15_ptr : RegionName := "candidate_vsum_15"
  let candidate_vbmm_123_ptr : RegionName := "candidate_vbmm_123"
  let candidate_vsub_55_ptr : RegionName := "candidate_vsub_55"
  let candidate_vmul_75_ptr : RegionName := "candidate_vmul_75"
  let candidate_vbmm_125_ptr : RegionName := "candidate_vbmm_125"
  let candidate_vmul_77_ptr : RegionName := "candidate_vmul_77"
  let candidate_vto_copy_154_ptr : RegionName := "candidate_vto_copy_154"
  let candidate_vbmm_126_ptr : RegionName := "candidate_vbmm_126"
  let candidate_vto_copy_155_ptr : RegionName := "candidate_vto_copy_155"
  let candidate_vcat_86_ptr : RegionName := "candidate_vcat_86"
  let candidate_vcat_87_ptr : RegionName := "candidate_vcat_87"
  let candidate_vcat_89_ptr : RegionName := "candidate_vcat_89"
  let candidate_vcat_90_ptr : RegionName := "candidate_vcat_90"
  let candidate_vbmm_129_ptr : RegionName := "candidate_vbmm_129"
  let candidate_vbmm_130_ptr : RegionName := "candidate_vbmm_130"
  let candidate_vcat_91_ptr : RegionName := "candidate_vcat_91"
  let candidate_vadd_54_ptr : RegionName := "candidate_vadd_54"
  let candidate_vbmm_131_ptr : RegionName := "candidate_vbmm_131"
  let candidate_vbmm_132_ptr : RegionName := "candidate_vbmm_132"
  let candidate_vcat_92_ptr : RegionName := "candidate_vcat_92"
  let candidate_vcat_93_ptr : RegionName := "candidate_vcat_93"
  let candidate_vto_copy_156_ptr : RegionName := "candidate_vto_copy_156"
  let candidate_vto_copy_157_ptr : RegionName := "candidate_vto_copy_157"
  let candidate_vbmm_133_ptr : RegionName := "candidate_vbmm_133"
  let candidate_vmul_78_ptr : RegionName := "candidate_vmul_78"
  let candidate_vadd_55_ptr : RegionName := "candidate_vadd_55"
  let candidate_vsub_56_ptr : RegionName := "candidate_vsub_56"
  let candidate_vgt_25_ptr : RegionName := "candidate_vgt_25"
  let candidate_vmasked_fill_25_ptr : RegionName := "candidate_vmasked_fill_25"
  let candidate_vsoftmax_17_ptr : RegionName := "candidate_vsoftmax_17"
  let candidate_vto_copy_158_ptr : RegionName := "candidate_vto_copy_158"
  let candidate_vbmm_134_ptr : RegionName := "candidate_vbmm_134"
  let candidate_vlogsumexp_17_ptr : RegionName := "candidate_vlogsumexp_17"
  let candidate_vcat_94_ptr : RegionName := "candidate_vcat_94"
  let candidate_vcat_95_ptr : RegionName := "candidate_vcat_95"
  let candidate_vto_copy_159_ptr : RegionName := "candidate_vto_copy_159"
  let candidate_vto_copy_160_ptr : RegionName := "candidate_vto_copy_160"
  let candidate_vto_copy_161_ptr : RegionName := "candidate_vto_copy_161"
  let candidate_vto_copy_162_ptr : RegionName := "candidate_vto_copy_162"
  let candidate_vto_copy_163_ptr : RegionName := "candidate_vto_copy_163"
  let candidate_vbmm_135_ptr : RegionName := "candidate_vbmm_135"
  let candidate_vmul_79_ptr : RegionName := "candidate_vmul_79"
  let candidate_vadd_56_ptr : RegionName := "candidate_vadd_56"
  let candidate_vsub_57_ptr : RegionName := "candidate_vsub_57"
  let candidate_vgt_26_ptr : RegionName := "candidate_vgt_26"
  let candidate_vmasked_fill_26_ptr : RegionName := "candidate_vmasked_fill_26"
  let candidate_vsub_58_ptr : RegionName := "candidate_vsub_58"
  let candidate_vexp_8_ptr : RegionName := "candidate_vexp_8"
  let candidate_vmul_80_ptr : RegionName := "candidate_vmul_80"
  let candidate_vsum_17_ptr : RegionName := "candidate_vsum_17"
  let candidate_vbmm_136_ptr : RegionName := "candidate_vbmm_136"
  let candidate_vsub_59_ptr : RegionName := "candidate_vsub_59"
  let candidate_vmul_81_ptr : RegionName := "candidate_vmul_81"
  let candidate_vbmm_138_ptr : RegionName := "candidate_vbmm_138"
  let candidate_vmul_83_ptr : RegionName := "candidate_vmul_83"
  let candidate_vto_copy_165_ptr : RegionName := "candidate_vto_copy_165"
  let candidate_vbmm_139_ptr : RegionName := "candidate_vbmm_139"
  let candidate_vto_copy_166_ptr : RegionName := "candidate_vto_copy_166"
  let candidate_vcat_96_ptr : RegionName := "candidate_vcat_96"
  let candidate_vcat_97_ptr : RegionName := "candidate_vcat_97"
  let candidate_vcat_99_ptr : RegionName := "candidate_vcat_99"
  let candidate_vcat_100_ptr : RegionName := "candidate_vcat_100"
  let candidate_vbmm_142_ptr : RegionName := "candidate_vbmm_142"
  let candidate_vbmm_143_ptr : RegionName := "candidate_vbmm_143"
  let candidate_vcat_101_ptr : RegionName := "candidate_vcat_101"
  let candidate_vadd_58_ptr : RegionName := "candidate_vadd_58"
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
    r15_z = tl.arange($(288))
    r15_value = tl.where(r15_z < $(96), tl.cast((r15_z - $(0)), tl.int64), tl.where(r15_z < $(192), tl.cast((r15_z - $(96)), tl.int64), tl.cast((r15_z - $(192)), tl.int64)))
    r15_value = tl.broadcast_to(r15_value, [$(288)])
    tl.store($(candidate_vcat_ptr) + r15_z, r15_value)
    r16_z = tl.arange($(32))
    r16_value = ((tl.cast(tl.toReal(r16_z), tl.float32) * 2.0) / 64.0)
    r16_value = r16_value.to(tl.float32)
    r16_value = tl.broadcast_to(r16_value, [$(32)])
    tl.store($(candidate_vdiv_ptr) + r16_z, r16_value)
    r17_z = tl.arange($(32))
    r17_value = tl.extra.cuda.libdevice.pow(10000.0, tl.load($(candidate_vdiv_ptr) + r17_z, dtype=tl.float32))
    r17_value = r17_value.to(tl.float32)
    r17_value = tl.broadcast_to(r17_value, [$(32)])
    tl.store($(candidate_vpow_1_ptr) + r17_z, r17_value)
    r18_z = tl.arange($(32))
    r18_value = 1.0 / tl.load($(candidate_vpow_1_ptr) + r18_z, dtype=tl.float32)
    r18_value = r18_value.to(tl.float32)
    r18_value = tl.broadcast_to(r18_value, [$(32)])
    tl.store($(candidate_vreciprocal_ptr) + r18_z, r18_value)
    r19_z = tl.arange($(32))
    r19_value = (tl.load($(candidate_vreciprocal_ptr) + r19_z, dtype=tl.float32) * 1.0)
    r19_value = r19_value.to(tl.float32)
    r19_value = tl.broadcast_to(r19_value, [$(32)])
    tl.store($(candidate_vmul_7_ptr) + r19_z, r19_value)
    r20_z = tl.arange($(9216))
    r20_value = (tl.cast(tl.toReal(((r20_z / $(32)) % $(288))), tl.float32) * tl.load($(candidate_vmul_7_ptr) + (r20_z % $(32)), dtype=tl.float32))
    r20_value = r20_value.to(tl.float32)
    r20_value = tl.broadcast_to(r20_value, [$(9216)])
    tl.store($(candidate_vmul_8_ptr) + r20_z, r20_value)
    r21_z = tl.arange($(18432))
    r21_value = tl.where((r21_z % $(64)) < $(32), tl.load($(candidate_vmul_8_ptr) + ((((r21_z / $(64)) % $(288)) * $(32)) + (((r21_z % $(64)) - $(0)) % $(32))), dtype=tl.float32), tl.load($(candidate_vmul_8_ptr) + ((((r21_z / $(64)) % $(288)) * $(32)) + (((r21_z % $(64)) - $(32)) % $(32))), dtype=tl.float32))
    r21_value = r21_value.to(tl.float32)
    r21_value = tl.broadcast_to(r21_value, [$(18432)])
    tl.store($(candidate_vcat_1_ptr) + r21_z, r21_value)
    r22_z = tl.arange($(18432))
    r22_value = tl.cos(tl.load($(candidate_vcat_1_ptr) + ((((r22_z / $(64)) % $(288)) * $(64)) + (r22_z % $(64))), dtype=tl.float32))
    r22_value = r22_value.to(tl.float32)
    r22_value = tl.broadcast_to(r22_value, [$(18432)])
    tl.store($(candidate_vcos_ptr) + r22_z, r22_value)
    r23_z = tl.arange($(18432))
    r23_value = tl.load($(candidate_vcos_ptr) + (((((((tl.load($(candidate_vcat_ptr) + ((r23_z / $(64)) % $(288)), dtype=tl.int64) * $(64)) + (r23_z % $(64))) / $(64)) % $(288)) % $(288)) * $(64)) + ((((tl.load($(candidate_vcat_ptr) + ((r23_z / $(64)) % $(288)), dtype=tl.int64) * $(64)) + (r23_z % $(64))) % $(64)) % $(64))), dtype=tl.float32)
    r23_value = r23_value.to(tl.float32)
    r23_value = tl.broadcast_to(r23_value, [$(18432)])
    tl.store($(candidate_vindex_ptr) + r23_z, r23_value)
    r24_z = tl.arange($(18432))
    r24_value = tl.sin(tl.load($(candidate_vcat_1_ptr) + ((((r24_z / $(64)) % $(288)) * $(64)) + (r24_z % $(64))), dtype=tl.float32))
    r24_value = r24_value.to(tl.float32)
    r24_value = tl.broadcast_to(r24_value, [$(18432)])
    tl.store($(candidate_vsin_ptr) + r24_z, r24_value)
    r25_z = tl.arange($(18432))
    r25_value = tl.load($(candidate_vsin_ptr) + (((((((tl.load($(candidate_vcat_ptr) + ((r25_z / $(64)) % $(288)), dtype=tl.int64) * $(64)) + (r25_z % $(64))) / $(64)) % $(288)) % $(288)) * $(64)) + ((((tl.load($(candidate_vcat_ptr) + ((r25_z / $(64)) % $(288)), dtype=tl.int64) * $(64)) + (r25_z % $(64))) % $(64)) % $(64))), dtype=tl.float32)
    r25_value = r25_value.to(tl.float32)
    r25_value = tl.broadcast_to(r25_value, [$(18432)])
    tl.store($(candidate_vindex_1_ptr) + r25_z, r25_value)
    r26_z = tl.arange($(18432))
    r26_value = tl.load($(candidate_vindex_ptr) + ((((r26_z / $(64)) % $(288)) * $(64)) + (r26_z % $(64))), dtype=tl.float32)
    r26_value = r26_value.to(tl.bfloat16)
    r26_value = tl.broadcast_to(r26_value, [$(18432)])
    tl.store($(candidate_vto_copy_14_ptr) + r26_z, r26_value)
    r27_z = tl.arange($(18432))
    r27_value = tl.load($(candidate_vindex_1_ptr) + ((((r27_z / $(64)) % $(288)) * $(64)) + (r27_z % $(64))), dtype=tl.float32)
    r27_value = r27_value.to(tl.bfloat16)
    r27_value = tl.broadcast_to(r27_value, [$(18432)])
    tl.store($(candidate_vto_copy_15_ptr) + r27_z, r27_value)
    r28_z = tl.arange($(96))
    r28_value = tl.where(r28_z < $(32), (tl.full([], $(64), dtype=tl.int64) + tl.cast((r28_z - $(0)), tl.int64)), tl.where(r28_z < $(64), (tl.full([], $(128), dtype=tl.int64) + tl.cast((r28_z - $(32)), tl.int64)), (tl.full([], $(192), dtype=tl.int64) + tl.cast((r28_z - $(64)), tl.int64))))
    r28_value = tl.broadcast_to(r28_value, [$(96)])
    tl.store($(candidate_vcat_4_ptr) + r28_z, r28_value)
    r29_z = tl.arange($(1048576))
    r29_value = tl.load($(candidate_vmul_5_ptr) + ((((r29_z / $(512)) % $(2048)) * $(512)) + (r29_z % $(512))), dtype=tl.float32)
    r29_value = r29_value.to(tl.bfloat16)
    r29_value = tl.broadcast_to(r29_value, [$(1048576)])
    tl.store($(candidate_vto_copy_18_ptr) + r29_z, r29_value)
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
    tl.store($(candidate_vto_copy_19_ptr) + r31_z, r31_value)
    r32_z = tl.arange($(73728))
    r32_value = ((tl.load($(candidate_vto_copy_19_ptr) + ((((r32_z / $(256)) % $(288)) * $(256)) + (r32_z % $(256))), dtype=tl.float32)) * (tl.load($(candidate_vto_copy_19_ptr) + ((((r32_z / $(256)) % $(288)) * $(256)) + (r32_z % $(256))), dtype=tl.float32)))
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
    r36_value = (tl.load($(candidate_vto_copy_19_ptr) + ((((r36_z / $(256)) % $(288)) * $(256)) + (r36_z % $(256))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_ptr) + ((r36_z / $(256)) % $(288)), dtype=tl.float32))
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
    tl.store($(candidate_vto_copy_20_ptr) + r38_z, r38_value)
    r39_row = tl.arange($(288))
    r39_col = tl.arange($(1536))
    r39_lane = tl.arange($(256))
    r39_left = tl.load($(candidate_vto_copy_20_ptr) + ((((((r39_row[:, None] * $(256)) + r39_lane[None, :]) / $(256)) % $(288)) * $(256)) + (((r39_row[:, None] * $(256)) + r39_lane[None, :]) % $(256))), dtype=tl.bfloat16)
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
    tl.store($(candidate_vto_copy_21_ptr) + r41_z, r41_value)
    r42_z = tl.arange($(147456))
    r42_value = ((tl.load($(candidate_vto_copy_21_ptr) + ((((r42_z / $(512)) % $(288)) * $(512)) + (r42_z % $(512))), dtype=tl.float32)) * (tl.load($(candidate_vto_copy_21_ptr) + ((((r42_z / $(512)) % $(288)) * $(512)) + (r42_z % $(512))), dtype=tl.float32)))
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
    r46_value = (tl.load($(candidate_vto_copy_21_ptr) + ((((r46_z / $(512)) % $(288)) * $(512)) + (r46_z % $(512))), dtype=tl.float32) * tl.load($(candidate_vrsqrt_1_ptr) + ((r46_z / $(512)) % $(288)), dtype=tl.float32))
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
    tl.store($(candidate_vto_copy_22_ptr) + r48_z, r48_value)
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
    r63_value = tl.where((r63_z % $(576)) < $(512), tl.load($(candidate_vto_copy_22_ptr) + ((((((((r63_z / $(576)) % $(288)) * $(512)) + ((r63_z % $(576)) - $(0))) / $(512)) % $(288)) * $(512)) + (((((r63_z / $(576)) % $(288)) * $(512)) + ((r63_z % $(576)) - $(0))) % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vcat_6_ptr) + ((((((((r63_z / $(576)) % $(288)) * $(64)) + ((r63_z % $(576)) - $(512))) / $(64)) % $(288)) * $(64)) + (((((r63_z / $(576)) % $(288)) * $(64)) + ((r63_z % $(576)) - $(512))) % $(64))), dtype=tl.bfloat16))
    r63_value = r63_value.to(tl.bfloat16)
    r63_value = tl.broadcast_to(r63_value, [$(165888)])
    tl.store($(candidate_vcat_7_ptr) + r63_z, r63_value)
    r64_z = tl.arange($(147456))
    r64_value = tl.cast(0.0, tl.bfloat16)
    for r64_lane in range($(96)) {
      r64_value = tl.where(tl.load($(candidate_vcat_4_ptr) + (r64_lane % $(96)), dtype=tl.int64) == ((r64_z / $(512)) % $(288)), r64_value + tl.load($(input1_ptr) + (((r64_lane % $(96)) * $(512)) + (r64_z % $(512))), dtype=tl.bfloat16), r64_value)
    }
    r64_value = r64_value.to(tl.bfloat16)
    r64_value = tl.broadcast_to(r64_value, [$(147456)])
    tl.store($(candidate_vindex_put_ptr) + r64_z, r64_value)
    r65_row = tl.arange($(288))
    r65_col = tl.arange($(1024))
    r65_lane = tl.arange($(512))
    r65_left = tl.load($(candidate_vindex_put_ptr) + ((((((r65_row[:, None] * $(512)) + r65_lane[None, :]) / $(512)) % $(288)) * $(512)) + (((r65_row[:, None] * $(512)) + r65_lane[None, :]) % $(512))), dtype=tl.bfloat16)
    r65_right = tl.load($(candidate_vto_copy_6_ptr) + (((r65_lane[:, None] % $(512)) * $(1024)) + (r65_col[None, :] % $(1024))), dtype=tl.bfloat16)
    r65_value = tl.dot(r65_left, r65_right)
    r65_value = r65_value.to(tl.bfloat16)
    tl.store($(candidate_vmm_22_ptr) + ((r65_row[:, None] * $(1024)) + r65_col[None, :]), r65_value)
    r66_row = tl.arange($(96))
    r66_col = tl.arange($(1024))
    r66_lane = tl.arange($(512))
    for r66_group in range($(1)) {
      r66_left = tl.load($(candidate_vcat_7_ptr) + ((((((r66_row[:, None] * $(512)) + r66_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r66_row[:, None] * $(512)) + r66_lane[None, :]) % $(512)) + $(110592)), dtype=tl.bfloat16)
      r66_right = tl.load($(candidate_vto_copy_18_ptr) + ((((((r66_lane[:, None] * $(1024)) + r66_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r66_lane[:, None] * $(1024)) + r66_col[None, :]) % $(128)) * $(512)) + ((((r66_lane[:, None] * $(1024)) + r66_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r66_value = tl.dot(r66_left, r66_right)
      r66_value = r66_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_27_ptr) + ((r66_group * $(98304)) + (r66_row[:, None] * $(1024)) + r66_col[None, :]), r66_value)
    }
    r67_row = tl.arange($(96))
    r67_col = tl.arange($(1024))
    r67_lane = tl.arange($(512))
    for r67_group in range($(1)) {
      r67_left = tl.load($(candidate_vcat_7_ptr) + ((((((r67_row[:, None] * $(512)) + r67_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r67_row[:, None] * $(512)) + r67_lane[None, :]) % $(512)) + $(110592)), dtype=tl.bfloat16)
      r67_right = tl.load($(candidate_vto_copy_18_ptr) + ((((((r67_lane[:, None] * $(1024)) + r67_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r67_lane[:, None] * $(1024)) + r67_col[None, :]) % $(128)) * $(512)) + ((((r67_lane[:, None] * $(1024)) + r67_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r67_value = tl.dot(r67_left, r67_right)
      r67_value = r67_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_28_ptr) + ((r67_group * $(98304)) + (r67_row[:, None] * $(1024)) + r67_col[None, :]), r67_value)
    }
    r68_z = tl.arange($(49152))
    r68_value = tl.where((r68_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r68_z / $(1536)) % $(32)) * $(1024)) + (((r68_z / $(192)) % $(8)) * $(128)) + ((r68_z % $(192)) - $(0)) + $(262144)) / $(1024)) % $(288)) * $(1536)) + (((((((r68_z / $(1536)) % $(32)) * $(1024)) + (((r68_z / $(192)) % $(8)) * $(128)) + ((r68_z % $(192)) - $(0)) + $(262144)) / $(128)) % $(8)) * $(192)) + (((((r68_z / $(1536)) % $(32)) * $(1024)) + (((r68_z / $(192)) % $(8)) * $(128)) + ((r68_z % $(192)) - $(0)) + $(262144)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_5_ptr) + ((((((((r68_z / $(1536)) % $(32)) * $(512)) + (((r68_z / $(192)) % $(8)) * $(64)) + ((r68_z % $(192)) - $(128)) + $(131072)) / $(64)) % $(8)) * $(18432)) + (((((((r68_z / $(1536)) % $(32)) * $(512)) + (((r68_z / $(192)) % $(8)) * $(64)) + ((r68_z % $(192)) - $(128)) + $(131072)) / $(512)) % $(288)) * $(64)) + (((((r68_z / $(1536)) % $(32)) * $(512)) + (((r68_z / $(192)) % $(8)) * $(64)) + ((r68_z % $(192)) - $(128)) + $(131072)) % $(64))), dtype=tl.bfloat16))
    r68_value = r68_value.to(tl.bfloat16)
    r68_value = tl.broadcast_to(r68_value, [$(49152)])
    tl.store($(candidate_vcat_18_ptr) + r68_z, r68_value)
    r69_z = tl.arange($(147456))
    r69_value = tl.where((r69_z % $(192)) < $(128), tl.load($(candidate_vbmm_27_ptr) + ((((((((r69_z / $(1536)) % $(96)) * $(1024)) + (((r69_z / $(192)) % $(8)) * $(128)) + ((r69_z % $(192)) - $(0))) / $(1024)) % $(96)) * $(1024)) + (((((((r69_z / $(1536)) % $(96)) * $(1024)) + (((r69_z / $(192)) % $(8)) * $(128)) + ((r69_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r69_z / $(1536)) % $(96)) * $(1024)) + (((r69_z / $(192)) % $(8)) * $(128)) + ((r69_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_7_ptr) + ((((r69_z / $(1536)) % $(96)) * $(576)) + ((((r69_z % $(192)) - $(128)) + $(512)) % $(576)) + $(110592)), dtype=tl.bfloat16))
    r69_value = r69_value.to(tl.bfloat16)
    r69_value = tl.broadcast_to(r69_value, [$(147456)])
    tl.store($(candidate_vcat_19_ptr) + r69_z, r69_value)
    r70_z = tl.arange($(49152))
    r70_value = tl.load($(candidate_vcat_18_ptr) + ((((r70_z / $(192)) % $(32)) * $(1536)) + (((r70_z / $(6144)) % $(8)) * $(192)) + (r70_z % $(192))), dtype=tl.bfloat16)
    r70_value = r70_value.to(tl.float32)
    r70_value = tl.broadcast_to(r70_value, [$(49152)])
    tl.store($(candidate_vto_copy_68_ptr) + r70_z, r70_value)
    r71_z = tl.arange($(147456))
    r71_value = tl.load($(candidate_vcat_19_ptr) + ((((r71_z / $(192)) % $(96)) * $(1536)) + (((r71_z / $(18432)) % $(8)) * $(192)) + (r71_z % $(192))), dtype=tl.bfloat16)
    r71_value = r71_value.to(tl.float32)
    r71_value = tl.broadcast_to(r71_value, [$(147456)])
    tl.store($(candidate_vto_copy_69_ptr) + r71_z, r71_value)
    r72_row = tl.arange($(32))
    r72_col = tl.arange($(96))
    r72_lane = tl.arange($(192))
    for r72_group in range($(8)) {
      r72_left = tl.load($(candidate_vto_copy_68_ptr) + (((((((r72_group % $(8)) * $(6144)) + (r72_row[:, None] * $(192)) + r72_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r72_group % $(8)) * $(6144)) + (r72_row[:, None] * $(192)) + r72_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r72_group % $(8)) * $(6144)) + (r72_row[:, None] * $(192)) + r72_lane[None, :]) % $(192))), dtype=tl.float32)
      r72_right = tl.load($(candidate_vto_copy_69_ptr) + (((((((r72_group % $(8)) * $(18432)) + (r72_lane[:, None] * $(96)) + r72_col[None, :]) / $(18432)) % $(8)) * $(18432)) + (((((r72_group % $(8)) * $(18432)) + (r72_lane[:, None] * $(96)) + r72_col[None, :]) % $(96)) * $(192)) + (((((r72_group % $(8)) * $(18432)) + (r72_lane[:, None] * $(96)) + r72_col[None, :]) / $(96)) % $(192))), dtype=tl.float32)
      r72_value = tl.dot(r72_left, r72_right)
      r72_value = r72_value.to(tl.float32)
      tl.store($(candidate_vbmm_29_ptr) + ((r72_group * $(3072)) + (r72_row[:, None] * $(96)) + r72_col[None, :]), r72_value)
    }
    r73_z = tl.arange($(24576))
    r73_value = (tl.load($(candidate_vbmm_29_ptr) + ((((r73_z / $(3072)) % $(8)) * $(3072)) + (((r73_z / $(96)) % $(32)) * $(96)) + (r73_z % $(96))), dtype=tl.float32) * 0.07216878364870322)
    r73_value = r73_value.to(tl.float32)
    r73_value = tl.broadcast_to(r73_value, [$(24576)])
    tl.store($(candidate_vmul_30_ptr) + r73_z, r73_value)
    r74_z = tl.arange($(32))
    r74_value = (tl.cast(r74_z, tl.int64) + tl.full([], $(96), dtype=tl.int64))
    r74_value = tl.broadcast_to(r74_value, [$(32)])
    tl.store($(candidate_vadd_24_ptr) + r74_z, r74_value)
    r75_z = tl.arange($(32))
    r75_value = (tl.load($(candidate_vadd_24_ptr) + r75_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r75_value = tl.broadcast_to(r75_value, [$(32)])
    tl.store($(candidate_vsub_24_ptr) + r75_z, r75_value)
    r76_z = tl.arange($(3072))
    r76_value = (tl.cast((r76_z % $(96)), tl.int64) > tl.load($(candidate_vsub_24_ptr) + ((r76_z / $(96)) % $(32)), dtype=tl.int64))
    r76_value = tl.broadcast_to(r76_value, [$(3072)])
    tl.store($(candidate_vgt_9_ptr) + r76_z, r76_value)
    r77_z = tl.arange($(24576))
    r77_value = tl.where(tl.load($(candidate_vgt_9_ptr) + ((((r77_z / $(96)) % $(32)) * $(96)) + (r77_z % $(96))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_30_ptr) + ((((r77_z / $(3072)) % $(8)) * $(3072)) + (((r77_z / $(96)) % $(32)) * $(96)) + (r77_z % $(96))), dtype=tl.float32))
    r77_value = r77_value.to(tl.float32)
    r77_value = tl.broadcast_to(r77_value, [$(24576)])
    tl.store($(candidate_vmasked_fill_9_ptr) + r77_z, r77_value)
    r78_z = tl.arange($(24576))
    r78_lane = tl.arange($(96))
    r78_row_values = tl.load($(candidate_vmasked_fill_9_ptr) + ((((r78_z[:, None] / $(3072)) % $(8)) * $(3072)) + (((r78_z[:, None] / $(96)) % $(32)) * $(96)) + (r78_lane[None, :] % $(96))), dtype=tl.float32)
    r78_maximum = tl.max(r78_row_values, axis=1, keep_dims=true)
    r78_weights = tl.exp(r78_row_values - r78_maximum)
    r78_total = tl.sum(r78_weights, axis=1)
    r78_maximum = tl.max(r78_row_values, axis=1)
    r78_value = tl.exp(tl.load($(candidate_vmasked_fill_9_ptr) + ((((r78_z / $(3072)) % $(8)) * $(3072)) + (((r78_z / $(96)) % $(32)) * $(96)) + (r78_z % $(96))), dtype=tl.float32) - r78_maximum) / r78_total
    r78_value = r78_value.to(tl.float32)
    r78_value = tl.broadcast_to(r78_value, [$(24576)])
    tl.store($(candidate_vsoftmax_9_ptr) + r78_z, r78_value)
    r79_z = tl.arange($(24576))
    r79_value = tl.load($(candidate_vsoftmax_9_ptr) + ((((r79_z / $(3072)) % $(8)) * $(3072)) + (((r79_z / $(96)) % $(32)) * $(96)) + (r79_z % $(96))), dtype=tl.float32)
    r79_value = r79_value.to(tl.bfloat16)
    r79_value = tl.broadcast_to(r79_value, [$(24576)])
    tl.store($(candidate_vto_copy_70_ptr) + r79_z, r79_value)
    r80_row = tl.arange($(32))
    r80_col = tl.arange($(128))
    r80_lane = tl.arange($(96))
    for r80_group in range($(8)) {
      r80_left = tl.load($(candidate_vto_copy_70_ptr) + (((((((r80_group % $(8)) * $(3072)) + (r80_row[:, None] * $(96)) + r80_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r80_group % $(8)) * $(3072)) + (r80_row[:, None] * $(96)) + r80_lane[None, :]) / $(96)) % $(32)) * $(96)) + ((((r80_group % $(8)) * $(3072)) + (r80_row[:, None] * $(96)) + r80_lane[None, :]) % $(96))), dtype=tl.bfloat16)
      r80_right = tl.load($(candidate_vbmm_28_ptr) + (((((((r80_group % $(8)) * $(12288)) + (r80_lane[:, None] * $(128)) + r80_col[None, :]) / $(128)) % $(96)) * $(1024)) + ((((((r80_group % $(8)) * $(12288)) + (r80_lane[:, None] * $(128)) + r80_col[None, :]) / $(12288)) % $(8)) * $(128)) + ((((r80_group % $(8)) * $(12288)) + (r80_lane[:, None] * $(128)) + r80_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r80_value = tl.dot(r80_left, r80_right)
      r80_value = r80_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_30_ptr) + ((r80_group * $(4096)) + (r80_row[:, None] * $(128)) + r80_col[None, :]), r80_value)
    }
    r81_z = tl.arange($(256))
    r81_lane = tl.arange($(96))
    r81_row_values = tl.load($(candidate_vmasked_fill_9_ptr) + ((((r81_z[:, None] / $(32)) % $(8)) * $(3072)) + ((r81_z[:, None] % $(32)) * $(96)) + (r81_lane[None, :] % $(96))), dtype=tl.float32)
    r81_maximum = tl.max(r81_row_values, axis=1, keep_dims=true)
    r81_weights = tl.exp(r81_row_values - r81_maximum)
    r81_total = tl.sum(r81_weights, axis=1)
    r81_maximum = tl.max(r81_row_values, axis=1)
    r81_value = r81_maximum + tl.log(r81_total)
    r81_value = r81_value.to(tl.float32)
    r81_value = tl.broadcast_to(r81_value, [$(256)])
    tl.store($(candidate_vlogsumexp_9_ptr) + r81_z, r81_value)
    r82_z = tl.arange($(32768))
    r82_value = tl.load($(candidate_vbmm_30_ptr) + ((((((((r82_z / $(128)) % $(8)) * $(4096)) + ((((r82_z / $(1024)) % $(32)) - $(0)) * $(128)) + (r82_z % $(128))) / $(4096)) % $(8)) * $(4096)) + ((((((r82_z / $(128)) % $(8)) * $(32)) + (((r82_z / $(1024)) % $(32)) - $(0))) % $(32)) * $(128)) + (r82_z % $(128))), dtype=tl.bfloat16)
    r82_value = r82_value.to(tl.bfloat16)
    r82_value = tl.broadcast_to(r82_value, [$(32768)])
    tl.store($(candidate_vcat_20_ptr) + r82_z, r82_value)
    r83_z = tl.arange($(256))
    r83_value = tl.load($(candidate_vlogsumexp_9_ptr) + (((r83_z % $(8)) * $(32)) + ((((r83_z / $(8)) % $(32)) - $(0)) % $(32))), dtype=tl.float32)
    r83_value = r83_value.to(tl.float32)
    r83_value = tl.broadcast_to(r83_value, [$(256)])
    tl.store($(candidate_vcat_21_ptr) + r83_z, r83_value)
    r84_z = tl.arange($(49152))
    r84_value = tl.load($(candidate_vcat_18_ptr) + ((((r84_z / $(192)) % $(32)) * $(1536)) + (((r84_z / $(6144)) % $(8)) * $(192)) + (r84_z % $(192))), dtype=tl.bfloat16)
    r84_value = r84_value.to(tl.float32)
    r84_value = tl.broadcast_to(r84_value, [$(49152)])
    tl.store($(candidate_vto_copy_71_ptr) + r84_z, r84_value)
    r85_z = tl.arange($(147456))
    r85_value = tl.load($(candidate_vcat_19_ptr) + ((((r85_z / $(192)) % $(96)) * $(1536)) + (((r85_z / $(18432)) % $(8)) * $(192)) + (r85_z % $(192))), dtype=tl.bfloat16)
    r85_value = r85_value.to(tl.float32)
    r85_value = tl.broadcast_to(r85_value, [$(147456)])
    tl.store($(candidate_vto_copy_72_ptr) + r85_z, r85_value)
    r86_z = tl.arange($(98304))
    r86_value = tl.load($(candidate_vbmm_28_ptr) + ((((r86_z / $(128)) % $(96)) * $(1024)) + (((r86_z / $(12288)) % $(8)) * $(128)) + (r86_z % $(128))), dtype=tl.bfloat16)
    r86_value = r86_value.to(tl.float32)
    r86_value = tl.broadcast_to(r86_value, [$(98304)])
    tl.store($(candidate_vto_copy_73_ptr) + r86_z, r86_value)
    r87_z = tl.arange($(32768))
    r87_value = tl.load($(candidate_vmm_22_ptr) + ((((r87_z / $(128)) % $(32)) * $(1024)) + (((r87_z / $(4096)) % $(8)) * $(128)) + (r87_z % $(128)) + $(262144)), dtype=tl.bfloat16)
    r87_value = r87_value.to(tl.float32)
    r87_value = tl.broadcast_to(r87_value, [$(32768)])
    tl.store($(candidate_vto_copy_74_ptr) + r87_z, r87_value)
    r88_z = tl.arange($(32768))
    r88_value = tl.load($(candidate_vcat_20_ptr) + ((((r88_z / $(128)) % $(32)) * $(1024)) + (((r88_z / $(4096)) % $(8)) * $(128)) + (r88_z % $(128))), dtype=tl.bfloat16)
    r88_value = r88_value.to(tl.float32)
    r88_value = tl.broadcast_to(r88_value, [$(32768)])
    tl.store($(candidate_vto_copy_75_ptr) + r88_z, r88_value)
    r89_row = tl.arange($(32))
    r89_col = tl.arange($(96))
    r89_lane = tl.arange($(192))
    for r89_group in range($(8)) {
      r89_left = tl.load($(candidate_vto_copy_71_ptr) + (((((((r89_group % $(8)) * $(6144)) + (r89_row[:, None] * $(192)) + r89_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r89_group % $(8)) * $(6144)) + (r89_row[:, None] * $(192)) + r89_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r89_group % $(8)) * $(6144)) + (r89_row[:, None] * $(192)) + r89_lane[None, :]) % $(192))), dtype=tl.float32)
      r89_right = tl.load($(candidate_vto_copy_72_ptr) + (((((((r89_group % $(8)) * $(18432)) + (r89_lane[:, None] * $(96)) + r89_col[None, :]) / $(18432)) % $(8)) * $(18432)) + (((((r89_group % $(8)) * $(18432)) + (r89_lane[:, None] * $(96)) + r89_col[None, :]) % $(96)) * $(192)) + (((((r89_group % $(8)) * $(18432)) + (r89_lane[:, None] * $(96)) + r89_col[None, :]) / $(96)) % $(192))), dtype=tl.float32)
      r89_value = tl.dot(r89_left, r89_right)
      r89_value = r89_value.to(tl.float32)
      tl.store($(candidate_vbmm_31_ptr) + ((r89_group * $(3072)) + (r89_row[:, None] * $(96)) + r89_col[None, :]), r89_value)
    }
    r90_z = tl.arange($(24576))
    r90_value = (tl.load($(candidate_vbmm_31_ptr) + ((((r90_z / $(3072)) % $(8)) * $(3072)) + (((r90_z / $(96)) % $(32)) * $(96)) + (r90_z % $(96))), dtype=tl.float32) * 0.07216878364870322)
    r90_value = r90_value.to(tl.float32)
    r90_value = tl.broadcast_to(r90_value, [$(24576)])
    tl.store($(candidate_vmul_31_ptr) + r90_z, r90_value)
    r91_z = tl.arange($(32))
    r91_value = (tl.cast(r91_z, tl.int64) + tl.full([], $(96), dtype=tl.int64))
    r91_value = tl.broadcast_to(r91_value, [$(32)])
    tl.store($(candidate_vadd_25_ptr) + r91_z, r91_value)
    r92_z = tl.arange($(32))
    r92_value = (tl.load($(candidate_vadd_25_ptr) + r92_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r92_value = tl.broadcast_to(r92_value, [$(32)])
    tl.store($(candidate_vsub_25_ptr) + r92_z, r92_value)
    r93_z = tl.arange($(3072))
    r93_value = (tl.cast((r93_z % $(96)), tl.int64) > tl.load($(candidate_vsub_25_ptr) + ((r93_z / $(96)) % $(32)), dtype=tl.int64))
    r93_value = tl.broadcast_to(r93_value, [$(3072)])
    tl.store($(candidate_vgt_10_ptr) + r93_z, r93_value)
    r94_z = tl.arange($(24576))
    r94_value = tl.where(tl.load($(candidate_vgt_10_ptr) + ((((r94_z / $(96)) % $(32)) * $(96)) + (r94_z % $(96))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_31_ptr) + ((((r94_z / $(3072)) % $(8)) * $(3072)) + (((r94_z / $(96)) % $(32)) * $(96)) + (r94_z % $(96))), dtype=tl.float32))
    r94_value = r94_value.to(tl.float32)
    r94_value = tl.broadcast_to(r94_value, [$(24576)])
    tl.store($(candidate_vmasked_fill_10_ptr) + r94_z, r94_value)
    r95_z = tl.arange($(24576))
    r95_value = (tl.load($(candidate_vmasked_fill_10_ptr) + ((((r95_z / $(3072)) % $(8)) * $(3072)) + (((r95_z / $(96)) % $(32)) * $(96)) + (r95_z % $(96))), dtype=tl.float32) - tl.load($(candidate_vcat_21_ptr) + ((((r95_z / $(96)) % $(32)) * $(8)) + ((r95_z / $(3072)) % $(8))), dtype=tl.float32))
    r95_value = r95_value.to(tl.float32)
    r95_value = tl.broadcast_to(r95_value, [$(24576)])
    tl.store($(candidate_vsub_26_ptr) + r95_z, r95_value)
    r96_z = tl.arange($(24576))
    r96_value = tl.exp(tl.load($(candidate_vsub_26_ptr) + ((((r96_z / $(3072)) % $(8)) * $(3072)) + (((r96_z / $(96)) % $(32)) * $(96)) + (r96_z % $(96))), dtype=tl.float32))
    r96_value = r96_value.to(tl.float32)
    r96_value = tl.broadcast_to(r96_value, [$(24576)])
    tl.store($(candidate_vexp_ptr) + r96_z, r96_value)
    r97_z = tl.arange($(32768))
    r97_value = (tl.load($(candidate_vto_copy_74_ptr) + ((((r97_z / $(4096)) % $(8)) * $(4096)) + (((r97_z / $(128)) % $(32)) * $(128)) + (r97_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_75_ptr) + ((((r97_z / $(4096)) % $(8)) * $(4096)) + (((r97_z / $(128)) % $(32)) * $(128)) + (r97_z % $(128))), dtype=tl.float32))
    r97_value = r97_value.to(tl.float32)
    r97_value = tl.broadcast_to(r97_value, [$(32768)])
    tl.store($(candidate_vmul_32_ptr) + r97_z, r97_value)
    r98_z = tl.arange($(256))
    r98_lane = tl.arange($(128))
    r98_value = tl.sum(tl.load($(candidate_vmul_32_ptr) + ((((r98_z[:, None] / $(32)) % $(8)) * $(4096)) + ((r98_z[:, None] % $(32)) * $(128)) + r98_lane[None, :]), dtype=tl.float32), axis=1)
    r98_value = r98_value.to(tl.float32)
    r98_value = tl.broadcast_to(r98_value, [$(256)])
    tl.store($(candidate_vsum_1_ptr) + r98_z, r98_value)
    r99_row = tl.arange($(32))
    r99_col = tl.arange($(96))
    r99_lane = tl.arange($(128))
    for r99_group in range($(8)) {
      r99_left = tl.load($(candidate_vto_copy_74_ptr) + (((((((r99_group % $(8)) * $(4096)) + (r99_row[:, None] * $(128)) + r99_lane[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r99_group % $(8)) * $(4096)) + (r99_row[:, None] * $(128)) + r99_lane[None, :]) / $(128)) % $(32)) * $(128)) + ((((r99_group % $(8)) * $(4096)) + (r99_row[:, None] * $(128)) + r99_lane[None, :]) % $(128))), dtype=tl.float32)
      r99_right = tl.load($(candidate_vto_copy_73_ptr) + (((((((r99_group % $(8)) * $(12288)) + (r99_lane[:, None] * $(96)) + r99_col[None, :]) / $(12288)) % $(8)) * $(12288)) + (((((r99_group % $(8)) * $(12288)) + (r99_lane[:, None] * $(96)) + r99_col[None, :]) % $(96)) * $(128)) + (((((r99_group % $(8)) * $(12288)) + (r99_lane[:, None] * $(96)) + r99_col[None, :]) / $(96)) % $(128))), dtype=tl.float32)
      r99_value = tl.dot(r99_left, r99_right)
      r99_value = r99_value.to(tl.float32)
      tl.store($(candidate_vbmm_32_ptr) + ((r99_group * $(3072)) + (r99_row[:, None] * $(96)) + r99_col[None, :]), r99_value)
    }
    r100_z = tl.arange($(24576))
    r100_value = (tl.load($(candidate_vbmm_32_ptr) + ((((r100_z / $(3072)) % $(8)) * $(3072)) + (((r100_z / $(96)) % $(32)) * $(96)) + (r100_z % $(96))), dtype=tl.float32) - tl.load($(candidate_vsum_1_ptr) + ((((r100_z / $(3072)) % $(8)) * $(32)) + ((r100_z / $(96)) % $(32))), dtype=tl.float32))
    r100_value = r100_value.to(tl.float32)
    r100_value = tl.broadcast_to(r100_value, [$(24576)])
    tl.store($(candidate_vsub_27_ptr) + r100_z, r100_value)
    r101_z = tl.arange($(24576))
    r101_value = (tl.load($(candidate_vexp_ptr) + ((((r101_z / $(3072)) % $(8)) * $(3072)) + (((r101_z / $(96)) % $(32)) * $(96)) + (r101_z % $(96))), dtype=tl.float32) * tl.load($(candidate_vsub_27_ptr) + ((((r101_z / $(3072)) % $(8)) * $(3072)) + (((r101_z / $(96)) % $(32)) * $(96)) + (r101_z % $(96))), dtype=tl.float32))
    r101_value = r101_value.to(tl.float32)
    r101_value = tl.broadcast_to(r101_value, [$(24576)])
    tl.store($(candidate_vmul_33_ptr) + r101_z, r101_value)
    r102_row = tl.arange($(96))
    r102_col = tl.arange($(192))
    r102_lane = tl.arange($(32))
    for r102_group in range($(8)) {
      r102_left = tl.load($(candidate_vmul_33_ptr) + (((((((r102_group % $(8)) * $(3072)) + (r102_row[:, None] * $(32)) + r102_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + (((((r102_group % $(8)) * $(3072)) + (r102_row[:, None] * $(32)) + r102_lane[None, :]) % $(32)) * $(96)) + (((((r102_group % $(8)) * $(3072)) + (r102_row[:, None] * $(32)) + r102_lane[None, :]) / $(32)) % $(96))), dtype=tl.float32)
      r102_right = tl.load($(candidate_vto_copy_71_ptr) + (((((((r102_group % $(8)) * $(6144)) + (r102_lane[:, None] * $(192)) + r102_col[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r102_group % $(8)) * $(6144)) + (r102_lane[:, None] * $(192)) + r102_col[None, :]) / $(192)) % $(32)) * $(192)) + ((((r102_group % $(8)) * $(6144)) + (r102_lane[:, None] * $(192)) + r102_col[None, :]) % $(192))), dtype=tl.float32)
      r102_value = tl.dot(r102_left, r102_right)
      r102_value = r102_value.to(tl.float32)
      tl.store($(candidate_vbmm_34_ptr) + ((r102_group * $(18432)) + (r102_row[:, None] * $(192)) + r102_col[None, :]), r102_value)
    }
    r103_z = tl.arange($(147456))
    r103_value = (tl.load($(candidate_vbmm_34_ptr) + ((((r103_z / $(18432)) % $(8)) * $(18432)) + (((r103_z / $(192)) % $(96)) * $(192)) + (r103_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r103_value = r103_value.to(tl.float32)
    r103_value = tl.broadcast_to(r103_value, [$(147456)])
    tl.store($(candidate_vmul_35_ptr) + r103_z, r103_value)
    r104_z = tl.arange($(147456))
    r104_value = tl.load($(candidate_vmul_35_ptr) + ((((r104_z / $(192)) % $(8)) * $(18432)) + (((r104_z / $(1536)) % $(96)) * $(192)) + (r104_z % $(192))), dtype=tl.float32)
    r104_value = r104_value.to(tl.bfloat16)
    r104_value = tl.broadcast_to(r104_value, [$(147456)])
    tl.store($(candidate_vto_copy_77_ptr) + r104_z, r104_value)
    r105_row = tl.arange($(96))
    r105_col = tl.arange($(128))
    r105_lane = tl.arange($(32))
    for r105_group in range($(8)) {
      r105_left = tl.load($(candidate_vexp_ptr) + (((((((r105_group % $(8)) * $(3072)) + (r105_row[:, None] * $(32)) + r105_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + (((((r105_group % $(8)) * $(3072)) + (r105_row[:, None] * $(32)) + r105_lane[None, :]) % $(32)) * $(96)) + (((((r105_group % $(8)) * $(3072)) + (r105_row[:, None] * $(32)) + r105_lane[None, :]) / $(32)) % $(96))), dtype=tl.float32)
      r105_right = tl.load($(candidate_vto_copy_74_ptr) + (((((((r105_group % $(8)) * $(4096)) + (r105_lane[:, None] * $(128)) + r105_col[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r105_group % $(8)) * $(4096)) + (r105_lane[:, None] * $(128)) + r105_col[None, :]) / $(128)) % $(32)) * $(128)) + ((((r105_group % $(8)) * $(4096)) + (r105_lane[:, None] * $(128)) + r105_col[None, :]) % $(128))), dtype=tl.float32)
      r105_value = tl.dot(r105_left, r105_right)
      r105_value = r105_value.to(tl.float32)
      tl.store($(candidate_vbmm_35_ptr) + ((r105_group * $(12288)) + (r105_row[:, None] * $(128)) + r105_col[None, :]), r105_value)
    }
    r106_z = tl.arange($(98304))
    r106_value = tl.load($(candidate_vbmm_35_ptr) + ((((r106_z / $(128)) % $(8)) * $(12288)) + (((r106_z / $(1024)) % $(96)) * $(128)) + (r106_z % $(128))), dtype=tl.float32)
    r106_value = r106_value.to(tl.bfloat16)
    r106_value = tl.broadcast_to(r106_value, [$(98304)])
    tl.store($(candidate_vto_copy_78_ptr) + r106_z, r106_value)
    r107_z = tl.arange($(147456))
    r107_value = tl.load($(candidate_vto_copy_77_ptr) + ((((((r107_z / $(1536)) % $(96)) - $(0)) % $(96)) * $(1536)) + (((r107_z / $(192)) % $(8)) * $(192)) + (r107_z % $(192))), dtype=tl.bfloat16)
    r107_value = r107_value.to(tl.bfloat16)
    r107_value = tl.broadcast_to(r107_value, [$(147456)])
    tl.store($(candidate_vcat_23_ptr) + r107_z, r107_value)
    r108_z = tl.arange($(98304))
    r108_value = tl.load($(candidate_vto_copy_78_ptr) + ((((((r108_z / $(1024)) % $(96)) - $(0)) % $(96)) * $(1024)) + (((r108_z / $(128)) % $(8)) * $(128)) + (r108_z % $(128))), dtype=tl.bfloat16)
    r108_value = r108_value.to(tl.bfloat16)
    r108_value = tl.broadcast_to(r108_value, [$(98304)])
    tl.store($(candidate_vcat_24_ptr) + r108_z, r108_value)
    r109_row = tl.arange($(1024))
    r109_col = tl.arange($(512))
    r109_lane = tl.arange($(96))
    for r109_group in range($(1)) {
      r109_left = tl.load($(candidate_vcat_23_ptr) + (((((r109_row[:, None] * $(96)) + r109_lane[None, :]) % $(96)) * $(1536)) + (((((r109_row[:, None] * $(96)) + r109_lane[None, :]) / $(12288)) % $(8)) * $(192)) + ((((r109_row[:, None] * $(96)) + r109_lane[None, :]) / $(96)) % $(128))), dtype=tl.bfloat16)
      r109_right = tl.load($(candidate_vcat_7_ptr) + ((((((r109_lane[:, None] * $(512)) + r109_col[None, :]) / $(512)) % $(96)) * $(576)) + (((r109_lane[:, None] * $(512)) + r109_col[None, :]) % $(512)) + $(110592)), dtype=tl.bfloat16)
      r109_value = tl.dot(r109_left, r109_right)
      r109_value = r109_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_38_ptr) + ((r109_group * $(524288)) + (r109_row[:, None] * $(512)) + r109_col[None, :]), r109_value)
    }
    r110_row = tl.arange($(1024))
    r110_col = tl.arange($(512))
    r110_lane = tl.arange($(96))
    for r110_group in range($(1)) {
      r110_left = tl.load($(candidate_vcat_24_ptr) + (((((r110_row[:, None] * $(96)) + r110_lane[None, :]) % $(96)) * $(1024)) + (((((r110_row[:, None] * $(96)) + r110_lane[None, :]) / $(12288)) % $(8)) * $(128)) + ((((r110_row[:, None] * $(96)) + r110_lane[None, :]) / $(96)) % $(128))), dtype=tl.bfloat16)
      r110_right = tl.load($(candidate_vcat_7_ptr) + ((((((r110_lane[:, None] * $(512)) + r110_col[None, :]) / $(512)) % $(96)) * $(576)) + (((r110_lane[:, None] * $(512)) + r110_col[None, :]) % $(512)) + $(110592)), dtype=tl.bfloat16)
      r110_value = tl.dot(r110_left, r110_right)
      r110_value = r110_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_39_ptr) + ((r110_group * $(524288)) + (r110_row[:, None] * $(512)) + r110_col[None, :]), r110_value)
    }
    r111_z = tl.arange($(1048576))
    r111_value = tl.where(((r111_z / $(512)) % $(256)) < $(128), tl.load($(candidate_vbmm_38_ptr) + ((((((((r111_z / $(131072)) % $(8)) * $(65536)) + ((((r111_z / $(512)) % $(256)) - $(0)) * $(512)) + (r111_z % $(512))) / $(65536)) % $(8)) * $(65536)) + ((((((r111_z / $(131072)) % $(8)) * $(128)) + (((r111_z / $(512)) % $(256)) - $(0))) % $(128)) * $(512)) + (r111_z % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vbmm_39_ptr) + ((((((((r111_z / $(131072)) % $(8)) * $(65536)) + ((((r111_z / $(512)) % $(256)) - $(128)) * $(512)) + (r111_z % $(512))) / $(65536)) % $(8)) * $(65536)) + ((((((r111_z / $(131072)) % $(8)) * $(128)) + (((r111_z / $(512)) % $(256)) - $(128))) % $(128)) * $(512)) + (r111_z % $(512))), dtype=tl.bfloat16))
    r111_value = r111_value.to(tl.bfloat16)
    r111_value = tl.broadcast_to(r111_value, [$(1048576)])
    tl.store($(candidate_vcat_25_ptr) + r111_z, r111_value)
    r112_row = tl.arange($(96))
    r112_col = tl.arange($(1024))
    r112_lane = tl.arange($(512))
    for r112_group in range($(1)) {
      r112_left = tl.load($(candidate_vcat_7_ptr) + ((((((r112_row[:, None] * $(512)) + r112_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r112_row[:, None] * $(512)) + r112_lane[None, :]) % $(512)) + $(110592)), dtype=tl.bfloat16)
      r112_right = tl.load($(candidate_vto_copy_18_ptr) + ((((((r112_lane[:, None] * $(1024)) + r112_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r112_lane[:, None] * $(1024)) + r112_col[None, :]) % $(128)) * $(512)) + ((((r112_lane[:, None] * $(1024)) + r112_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r112_value = tl.dot(r112_left, r112_right)
      r112_value = r112_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_40_ptr) + ((r112_group * $(98304)) + (r112_row[:, None] * $(1024)) + r112_col[None, :]), r112_value)
    }
    r113_row = tl.arange($(96))
    r113_col = tl.arange($(1024))
    r113_lane = tl.arange($(512))
    for r113_group in range($(1)) {
      r113_left = tl.load($(candidate_vcat_7_ptr) + ((((((r113_row[:, None] * $(512)) + r113_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r113_row[:, None] * $(512)) + r113_lane[None, :]) % $(512)) + $(110592)), dtype=tl.bfloat16)
      r113_right = tl.load($(candidate_vto_copy_18_ptr) + ((((((r113_lane[:, None] * $(1024)) + r113_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r113_lane[:, None] * $(1024)) + r113_col[None, :]) % $(128)) * $(512)) + ((((r113_lane[:, None] * $(1024)) + r113_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r113_value = tl.dot(r113_left, r113_right)
      r113_value = r113_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_41_ptr) + ((r113_group * $(98304)) + (r113_row[:, None] * $(1024)) + r113_col[None, :]), r113_value)
    }
    r114_z = tl.arange($(49152))
    r114_value = tl.where((r114_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r114_z / $(1536)) % $(32)) * $(1024)) + (((r114_z / $(192)) % $(8)) * $(128)) + ((r114_z % $(192)) - $(0)) + $(229376)) / $(1024)) % $(288)) * $(1536)) + (((((((r114_z / $(1536)) % $(32)) * $(1024)) + (((r114_z / $(192)) % $(8)) * $(128)) + ((r114_z % $(192)) - $(0)) + $(229376)) / $(128)) % $(8)) * $(192)) + (((((r114_z / $(1536)) % $(32)) * $(1024)) + (((r114_z / $(192)) % $(8)) * $(128)) + ((r114_z % $(192)) - $(0)) + $(229376)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_5_ptr) + ((((((((r114_z / $(1536)) % $(32)) * $(512)) + (((r114_z / $(192)) % $(8)) * $(64)) + ((r114_z % $(192)) - $(128)) + $(114688)) / $(64)) % $(8)) * $(18432)) + (((((((r114_z / $(1536)) % $(32)) * $(512)) + (((r114_z / $(192)) % $(8)) * $(64)) + ((r114_z % $(192)) - $(128)) + $(114688)) / $(512)) % $(288)) * $(64)) + (((((r114_z / $(1536)) % $(32)) * $(512)) + (((r114_z / $(192)) % $(8)) * $(64)) + ((r114_z % $(192)) - $(128)) + $(114688)) % $(64))), dtype=tl.bfloat16))
    r114_value = r114_value.to(tl.bfloat16)
    r114_value = tl.broadcast_to(r114_value, [$(49152)])
    tl.store($(candidate_vcat_26_ptr) + r114_z, r114_value)
    r115_z = tl.arange($(147456))
    r115_value = tl.where((r115_z % $(192)) < $(128), tl.load($(candidate_vbmm_40_ptr) + ((((((((r115_z / $(1536)) % $(96)) * $(1024)) + (((r115_z / $(192)) % $(8)) * $(128)) + ((r115_z % $(192)) - $(0))) / $(1024)) % $(96)) * $(1024)) + (((((((r115_z / $(1536)) % $(96)) * $(1024)) + (((r115_z / $(192)) % $(8)) * $(128)) + ((r115_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r115_z / $(1536)) % $(96)) * $(1024)) + (((r115_z / $(192)) % $(8)) * $(128)) + ((r115_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_7_ptr) + ((((r115_z / $(1536)) % $(96)) * $(576)) + ((((r115_z % $(192)) - $(128)) + $(512)) % $(576)) + $(110592)), dtype=tl.bfloat16))
    r115_value = r115_value.to(tl.bfloat16)
    r115_value = tl.broadcast_to(r115_value, [$(147456)])
    tl.store($(candidate_vcat_27_ptr) + r115_z, r115_value)
    r116_z = tl.arange($(49152))
    r116_value = tl.load($(candidate_vcat_26_ptr) + ((((r116_z / $(192)) % $(32)) * $(1536)) + (((r116_z / $(6144)) % $(8)) * $(192)) + (r116_z % $(192))), dtype=tl.bfloat16)
    r116_value = r116_value.to(tl.float32)
    r116_value = tl.broadcast_to(r116_value, [$(49152)])
    tl.store($(candidate_vto_copy_79_ptr) + r116_z, r116_value)
    r117_z = tl.arange($(98304))
    r117_value = tl.load($(candidate_vcat_27_ptr) + ((((r117_z / $(192)) % $(64)) * $(1536)) + (((r117_z / $(12288)) % $(8)) * $(192)) + (r117_z % $(192))), dtype=tl.bfloat16)
    r117_value = r117_value.to(tl.float32)
    r117_value = tl.broadcast_to(r117_value, [$(98304)])
    tl.store($(candidate_vto_copy_80_ptr) + r117_z, r117_value)
    r118_row = tl.arange($(32))
    r118_col = tl.arange($(64))
    r118_lane = tl.arange($(192))
    for r118_group in range($(8)) {
      r118_left = tl.load($(candidate_vto_copy_79_ptr) + (((((((r118_group % $(8)) * $(6144)) + (r118_row[:, None] * $(192)) + r118_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r118_group % $(8)) * $(6144)) + (r118_row[:, None] * $(192)) + r118_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r118_group % $(8)) * $(6144)) + (r118_row[:, None] * $(192)) + r118_lane[None, :]) % $(192))), dtype=tl.float32)
      r118_right = tl.load($(candidate_vto_copy_80_ptr) + (((((((r118_group % $(8)) * $(12288)) + (r118_lane[:, None] * $(64)) + r118_col[None, :]) / $(12288)) % $(8)) * $(12288)) + (((((r118_group % $(8)) * $(12288)) + (r118_lane[:, None] * $(64)) + r118_col[None, :]) % $(64)) * $(192)) + (((((r118_group % $(8)) * $(12288)) + (r118_lane[:, None] * $(64)) + r118_col[None, :]) / $(64)) % $(192))), dtype=tl.float32)
      r118_value = tl.dot(r118_left, r118_right)
      r118_value = r118_value.to(tl.float32)
      tl.store($(candidate_vbmm_42_ptr) + ((r118_group * $(2048)) + (r118_row[:, None] * $(64)) + r118_col[None, :]), r118_value)
    }
    r119_z = tl.arange($(16384))
    r119_value = (tl.load($(candidate_vbmm_42_ptr) + ((((r119_z / $(2048)) % $(8)) * $(2048)) + (((r119_z / $(64)) % $(32)) * $(64)) + (r119_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r119_value = r119_value.to(tl.float32)
    r119_value = tl.broadcast_to(r119_value, [$(16384)])
    tl.store($(candidate_vmul_36_ptr) + r119_z, r119_value)
    r120_z = tl.arange($(32))
    r120_value = (tl.cast(r120_z, tl.int64) + tl.full([], $(64), dtype=tl.int64))
    r120_value = tl.broadcast_to(r120_value, [$(32)])
    tl.store($(candidate_vadd_27_ptr) + r120_z, r120_value)
    r121_z = tl.arange($(32))
    r121_value = (tl.load($(candidate_vadd_27_ptr) + r121_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r121_value = tl.broadcast_to(r121_value, [$(32)])
    tl.store($(candidate_vsub_28_ptr) + r121_z, r121_value)
    r122_z = tl.arange($(2048))
    r122_value = (tl.cast((r122_z % $(64)), tl.int64) > tl.load($(candidate_vsub_28_ptr) + ((r122_z / $(64)) % $(32)), dtype=tl.int64))
    r122_value = tl.broadcast_to(r122_value, [$(2048)])
    tl.store($(candidate_vgt_11_ptr) + r122_z, r122_value)
    r123_z = tl.arange($(16384))
    r123_value = tl.where(tl.load($(candidate_vgt_11_ptr) + ((((r123_z / $(64)) % $(32)) * $(64)) + (r123_z % $(64))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_36_ptr) + ((((r123_z / $(2048)) % $(8)) * $(2048)) + (((r123_z / $(64)) % $(32)) * $(64)) + (r123_z % $(64))), dtype=tl.float32))
    r123_value = r123_value.to(tl.float32)
    r123_value = tl.broadcast_to(r123_value, [$(16384)])
    tl.store($(candidate_vmasked_fill_11_ptr) + r123_z, r123_value)
    r124_z = tl.arange($(16384))
    r124_lane = tl.arange($(64))
    r124_row_values = tl.load($(candidate_vmasked_fill_11_ptr) + ((((r124_z[:, None] / $(2048)) % $(8)) * $(2048)) + (((r124_z[:, None] / $(64)) % $(32)) * $(64)) + (r124_lane[None, :] % $(64))), dtype=tl.float32)
    r124_maximum = tl.max(r124_row_values, axis=1, keep_dims=true)
    r124_weights = tl.exp(r124_row_values - r124_maximum)
    r124_total = tl.sum(r124_weights, axis=1)
    r124_maximum = tl.max(r124_row_values, axis=1)
    r124_value = tl.exp(tl.load($(candidate_vmasked_fill_11_ptr) + ((((r124_z / $(2048)) % $(8)) * $(2048)) + (((r124_z / $(64)) % $(32)) * $(64)) + (r124_z % $(64))), dtype=tl.float32) - r124_maximum) / r124_total
    r124_value = r124_value.to(tl.float32)
    r124_value = tl.broadcast_to(r124_value, [$(16384)])
    tl.store($(candidate_vsoftmax_10_ptr) + r124_z, r124_value)
    r125_z = tl.arange($(16384))
    r125_value = tl.load($(candidate_vsoftmax_10_ptr) + ((((r125_z / $(2048)) % $(8)) * $(2048)) + (((r125_z / $(64)) % $(32)) * $(64)) + (r125_z % $(64))), dtype=tl.float32)
    r125_value = r125_value.to(tl.bfloat16)
    r125_value = tl.broadcast_to(r125_value, [$(16384)])
    tl.store($(candidate_vto_copy_81_ptr) + r125_z, r125_value)
    r126_row = tl.arange($(32))
    r126_col = tl.arange($(128))
    r126_lane = tl.arange($(64))
    for r126_group in range($(8)) {
      r126_left = tl.load($(candidate_vto_copy_81_ptr) + (((((((r126_group % $(8)) * $(2048)) + (r126_row[:, None] * $(64)) + r126_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r126_group % $(8)) * $(2048)) + (r126_row[:, None] * $(64)) + r126_lane[None, :]) / $(64)) % $(32)) * $(64)) + ((((r126_group % $(8)) * $(2048)) + (r126_row[:, None] * $(64)) + r126_lane[None, :]) % $(64))), dtype=tl.bfloat16)
      r126_right = tl.load($(candidate_vbmm_41_ptr) + (((((((r126_group % $(8)) * $(8192)) + (r126_lane[:, None] * $(128)) + r126_col[None, :]) / $(128)) % $(64)) * $(1024)) + ((((((r126_group % $(8)) * $(8192)) + (r126_lane[:, None] * $(128)) + r126_col[None, :]) / $(8192)) % $(8)) * $(128)) + ((((r126_group % $(8)) * $(8192)) + (r126_lane[:, None] * $(128)) + r126_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r126_value = tl.dot(r126_left, r126_right)
      r126_value = r126_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_43_ptr) + ((r126_group * $(4096)) + (r126_row[:, None] * $(128)) + r126_col[None, :]), r126_value)
    }
    r127_z = tl.arange($(256))
    r127_lane = tl.arange($(64))
    r127_row_values = tl.load($(candidate_vmasked_fill_11_ptr) + ((((r127_z[:, None] / $(32)) % $(8)) * $(2048)) + ((r127_z[:, None] % $(32)) * $(64)) + (r127_lane[None, :] % $(64))), dtype=tl.float32)
    r127_maximum = tl.max(r127_row_values, axis=1, keep_dims=true)
    r127_weights = tl.exp(r127_row_values - r127_maximum)
    r127_total = tl.sum(r127_weights, axis=1)
    r127_maximum = tl.max(r127_row_values, axis=1)
    r127_value = r127_maximum + tl.log(r127_total)
    r127_value = r127_value.to(tl.float32)
    r127_value = tl.broadcast_to(r127_value, [$(256)])
    tl.store($(candidate_vlogsumexp_10_ptr) + r127_z, r127_value)
    r128_z = tl.arange($(32768))
    r128_value = tl.load($(candidate_vbmm_43_ptr) + ((((((((r128_z / $(128)) % $(8)) * $(4096)) + ((((r128_z / $(1024)) % $(32)) - $(0)) * $(128)) + (r128_z % $(128))) / $(4096)) % $(8)) * $(4096)) + ((((((r128_z / $(128)) % $(8)) * $(32)) + (((r128_z / $(1024)) % $(32)) - $(0))) % $(32)) * $(128)) + (r128_z % $(128))), dtype=tl.bfloat16)
    r128_value = r128_value.to(tl.bfloat16)
    r128_value = tl.broadcast_to(r128_value, [$(32768)])
    tl.store($(candidate_vcat_28_ptr) + r128_z, r128_value)
    r129_z = tl.arange($(256))
    r129_value = tl.load($(candidate_vlogsumexp_10_ptr) + (((r129_z % $(8)) * $(32)) + ((((r129_z / $(8)) % $(32)) - $(0)) % $(32))), dtype=tl.float32)
    r129_value = r129_value.to(tl.float32)
    r129_value = tl.broadcast_to(r129_value, [$(256)])
    tl.store($(candidate_vcat_29_ptr) + r129_z, r129_value)
    r130_z = tl.arange($(49152))
    r130_value = tl.load($(candidate_vcat_26_ptr) + ((((r130_z / $(192)) % $(32)) * $(1536)) + (((r130_z / $(6144)) % $(8)) * $(192)) + (r130_z % $(192))), dtype=tl.bfloat16)
    r130_value = r130_value.to(tl.float32)
    r130_value = tl.broadcast_to(r130_value, [$(49152)])
    tl.store($(candidate_vto_copy_82_ptr) + r130_z, r130_value)
    r131_z = tl.arange($(98304))
    r131_value = tl.load($(candidate_vcat_27_ptr) + ((((r131_z / $(192)) % $(64)) * $(1536)) + (((r131_z / $(12288)) % $(8)) * $(192)) + (r131_z % $(192))), dtype=tl.bfloat16)
    r131_value = r131_value.to(tl.float32)
    r131_value = tl.broadcast_to(r131_value, [$(98304)])
    tl.store($(candidate_vto_copy_83_ptr) + r131_z, r131_value)
    r132_z = tl.arange($(65536))
    r132_value = tl.load($(candidate_vbmm_41_ptr) + ((((r132_z / $(128)) % $(64)) * $(1024)) + (((r132_z / $(8192)) % $(8)) * $(128)) + (r132_z % $(128))), dtype=tl.bfloat16)
    r132_value = r132_value.to(tl.float32)
    r132_value = tl.broadcast_to(r132_value, [$(65536)])
    tl.store($(candidate_vto_copy_84_ptr) + r132_z, r132_value)
    r133_z = tl.arange($(32768))
    r133_value = tl.load($(candidate_vmm_22_ptr) + ((((r133_z / $(128)) % $(32)) * $(1024)) + (((r133_z / $(4096)) % $(8)) * $(128)) + (r133_z % $(128)) + $(229376)), dtype=tl.bfloat16)
    r133_value = r133_value.to(tl.float32)
    r133_value = tl.broadcast_to(r133_value, [$(32768)])
    tl.store($(candidate_vto_copy_85_ptr) + r133_z, r133_value)
    r134_z = tl.arange($(32768))
    r134_value = tl.load($(candidate_vcat_28_ptr) + ((((r134_z / $(128)) % $(32)) * $(1024)) + (((r134_z / $(4096)) % $(8)) * $(128)) + (r134_z % $(128))), dtype=tl.bfloat16)
    r134_value = r134_value.to(tl.float32)
    r134_value = tl.broadcast_to(r134_value, [$(32768)])
    tl.store($(candidate_vto_copy_86_ptr) + r134_z, r134_value)
    r135_row = tl.arange($(32))
    r135_col = tl.arange($(64))
    r135_lane = tl.arange($(192))
    for r135_group in range($(8)) {
      r135_left = tl.load($(candidate_vto_copy_82_ptr) + (((((((r135_group % $(8)) * $(6144)) + (r135_row[:, None] * $(192)) + r135_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r135_group % $(8)) * $(6144)) + (r135_row[:, None] * $(192)) + r135_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r135_group % $(8)) * $(6144)) + (r135_row[:, None] * $(192)) + r135_lane[None, :]) % $(192))), dtype=tl.float32)
      r135_right = tl.load($(candidate_vto_copy_83_ptr) + (((((((r135_group % $(8)) * $(12288)) + (r135_lane[:, None] * $(64)) + r135_col[None, :]) / $(12288)) % $(8)) * $(12288)) + (((((r135_group % $(8)) * $(12288)) + (r135_lane[:, None] * $(64)) + r135_col[None, :]) % $(64)) * $(192)) + (((((r135_group % $(8)) * $(12288)) + (r135_lane[:, None] * $(64)) + r135_col[None, :]) / $(64)) % $(192))), dtype=tl.float32)
      r135_value = tl.dot(r135_left, r135_right)
      r135_value = r135_value.to(tl.float32)
      tl.store($(candidate_vbmm_44_ptr) + ((r135_group * $(2048)) + (r135_row[:, None] * $(64)) + r135_col[None, :]), r135_value)
    }
    r136_z = tl.arange($(16384))
    r136_value = (tl.load($(candidate_vbmm_44_ptr) + ((((r136_z / $(2048)) % $(8)) * $(2048)) + (((r136_z / $(64)) % $(32)) * $(64)) + (r136_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r136_value = r136_value.to(tl.float32)
    r136_value = tl.broadcast_to(r136_value, [$(16384)])
    tl.store($(candidate_vmul_37_ptr) + r136_z, r136_value)
    r137_z = tl.arange($(32))
    r137_value = (tl.cast(r137_z, tl.int64) + tl.full([], $(64), dtype=tl.int64))
    r137_value = tl.broadcast_to(r137_value, [$(32)])
    tl.store($(candidate_vadd_28_ptr) + r137_z, r137_value)
    r138_z = tl.arange($(32))
    r138_value = (tl.load($(candidate_vadd_28_ptr) + r138_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r138_value = tl.broadcast_to(r138_value, [$(32)])
    tl.store($(candidate_vsub_29_ptr) + r138_z, r138_value)
    r139_z = tl.arange($(2048))
    r139_value = (tl.cast((r139_z % $(64)), tl.int64) > tl.load($(candidate_vsub_29_ptr) + ((r139_z / $(64)) % $(32)), dtype=tl.int64))
    r139_value = tl.broadcast_to(r139_value, [$(2048)])
    tl.store($(candidate_vgt_12_ptr) + r139_z, r139_value)
    r140_z = tl.arange($(16384))
    r140_value = tl.where(tl.load($(candidate_vgt_12_ptr) + ((((r140_z / $(64)) % $(32)) * $(64)) + (r140_z % $(64))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_37_ptr) + ((((r140_z / $(2048)) % $(8)) * $(2048)) + (((r140_z / $(64)) % $(32)) * $(64)) + (r140_z % $(64))), dtype=tl.float32))
    r140_value = r140_value.to(tl.float32)
    r140_value = tl.broadcast_to(r140_value, [$(16384)])
    tl.store($(candidate_vmasked_fill_12_ptr) + r140_z, r140_value)
    r141_z = tl.arange($(16384))
    r141_value = (tl.load($(candidate_vmasked_fill_12_ptr) + ((((r141_z / $(2048)) % $(8)) * $(2048)) + (((r141_z / $(64)) % $(32)) * $(64)) + (r141_z % $(64))), dtype=tl.float32) - tl.load($(candidate_vcat_29_ptr) + ((((r141_z / $(64)) % $(32)) * $(8)) + ((r141_z / $(2048)) % $(8))), dtype=tl.float32))
    r141_value = r141_value.to(tl.float32)
    r141_value = tl.broadcast_to(r141_value, [$(16384)])
    tl.store($(candidate_vsub_30_ptr) + r141_z, r141_value)
    r142_z = tl.arange($(16384))
    r142_value = tl.exp(tl.load($(candidate_vsub_30_ptr) + ((((r142_z / $(2048)) % $(8)) * $(2048)) + (((r142_z / $(64)) % $(32)) * $(64)) + (r142_z % $(64))), dtype=tl.float32))
    r142_value = r142_value.to(tl.float32)
    r142_value = tl.broadcast_to(r142_value, [$(16384)])
    tl.store($(candidate_vexp_1_ptr) + r142_z, r142_value)
    r143_z = tl.arange($(32768))
    r143_value = (tl.load($(candidate_vto_copy_85_ptr) + ((((r143_z / $(4096)) % $(8)) * $(4096)) + (((r143_z / $(128)) % $(32)) * $(128)) + (r143_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_86_ptr) + ((((r143_z / $(4096)) % $(8)) * $(4096)) + (((r143_z / $(128)) % $(32)) * $(128)) + (r143_z % $(128))), dtype=tl.float32))
    r143_value = r143_value.to(tl.float32)
    r143_value = tl.broadcast_to(r143_value, [$(32768)])
    tl.store($(candidate_vmul_38_ptr) + r143_z, r143_value)
    r144_z = tl.arange($(256))
    r144_lane = tl.arange($(128))
    r144_value = tl.sum(tl.load($(candidate_vmul_38_ptr) + ((((r144_z[:, None] / $(32)) % $(8)) * $(4096)) + ((r144_z[:, None] % $(32)) * $(128)) + r144_lane[None, :]), dtype=tl.float32), axis=1)
    r144_value = r144_value.to(tl.float32)
    r144_value = tl.broadcast_to(r144_value, [$(256)])
    tl.store($(candidate_vsum_3_ptr) + r144_z, r144_value)
    r145_row = tl.arange($(32))
    r145_col = tl.arange($(64))
    r145_lane = tl.arange($(128))
    for r145_group in range($(8)) {
      r145_left = tl.load($(candidate_vto_copy_85_ptr) + (((((((r145_group % $(8)) * $(4096)) + (r145_row[:, None] * $(128)) + r145_lane[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r145_group % $(8)) * $(4096)) + (r145_row[:, None] * $(128)) + r145_lane[None, :]) / $(128)) % $(32)) * $(128)) + ((((r145_group % $(8)) * $(4096)) + (r145_row[:, None] * $(128)) + r145_lane[None, :]) % $(128))), dtype=tl.float32)
      r145_right = tl.load($(candidate_vto_copy_84_ptr) + (((((((r145_group % $(8)) * $(8192)) + (r145_lane[:, None] * $(64)) + r145_col[None, :]) / $(8192)) % $(8)) * $(8192)) + (((((r145_group % $(8)) * $(8192)) + (r145_lane[:, None] * $(64)) + r145_col[None, :]) % $(64)) * $(128)) + (((((r145_group % $(8)) * $(8192)) + (r145_lane[:, None] * $(64)) + r145_col[None, :]) / $(64)) % $(128))), dtype=tl.float32)
      r145_value = tl.dot(r145_left, r145_right)
      r145_value = r145_value.to(tl.float32)
      tl.store($(candidate_vbmm_45_ptr) + ((r145_group * $(2048)) + (r145_row[:, None] * $(64)) + r145_col[None, :]), r145_value)
    }
    r146_z = tl.arange($(16384))
    r146_value = (tl.load($(candidate_vbmm_45_ptr) + ((((r146_z / $(2048)) % $(8)) * $(2048)) + (((r146_z / $(64)) % $(32)) * $(64)) + (r146_z % $(64))), dtype=tl.float32) - tl.load($(candidate_vsum_3_ptr) + ((((r146_z / $(2048)) % $(8)) * $(32)) + ((r146_z / $(64)) % $(32))), dtype=tl.float32))
    r146_value = r146_value.to(tl.float32)
    r146_value = tl.broadcast_to(r146_value, [$(16384)])
    tl.store($(candidate_vsub_31_ptr) + r146_z, r146_value)
    r147_z = tl.arange($(16384))
    r147_value = (tl.load($(candidate_vexp_1_ptr) + ((((r147_z / $(2048)) % $(8)) * $(2048)) + (((r147_z / $(64)) % $(32)) * $(64)) + (r147_z % $(64))), dtype=tl.float32) * tl.load($(candidate_vsub_31_ptr) + ((((r147_z / $(2048)) % $(8)) * $(2048)) + (((r147_z / $(64)) % $(32)) * $(64)) + (r147_z % $(64))), dtype=tl.float32))
    r147_value = r147_value.to(tl.float32)
    r147_value = tl.broadcast_to(r147_value, [$(16384)])
    tl.store($(candidate_vmul_39_ptr) + r147_z, r147_value)
    r148_row = tl.arange($(64))
    r148_col = tl.arange($(192))
    r148_lane = tl.arange($(32))
    for r148_group in range($(8)) {
      r148_left = tl.load($(candidate_vmul_39_ptr) + (((((((r148_group % $(8)) * $(2048)) + (r148_row[:, None] * $(32)) + r148_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + (((((r148_group % $(8)) * $(2048)) + (r148_row[:, None] * $(32)) + r148_lane[None, :]) % $(32)) * $(64)) + (((((r148_group % $(8)) * $(2048)) + (r148_row[:, None] * $(32)) + r148_lane[None, :]) / $(32)) % $(64))), dtype=tl.float32)
      r148_right = tl.load($(candidate_vto_copy_82_ptr) + (((((((r148_group % $(8)) * $(6144)) + (r148_lane[:, None] * $(192)) + r148_col[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r148_group % $(8)) * $(6144)) + (r148_lane[:, None] * $(192)) + r148_col[None, :]) / $(192)) % $(32)) * $(192)) + ((((r148_group % $(8)) * $(6144)) + (r148_lane[:, None] * $(192)) + r148_col[None, :]) % $(192))), dtype=tl.float32)
      r148_value = tl.dot(r148_left, r148_right)
      r148_value = r148_value.to(tl.float32)
      tl.store($(candidate_vbmm_47_ptr) + ((r148_group * $(12288)) + (r148_row[:, None] * $(192)) + r148_col[None, :]), r148_value)
    }
    r149_z = tl.arange($(98304))
    r149_value = (tl.load($(candidate_vbmm_47_ptr) + ((((r149_z / $(12288)) % $(8)) * $(12288)) + (((r149_z / $(192)) % $(64)) * $(192)) + (r149_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r149_value = r149_value.to(tl.float32)
    r149_value = tl.broadcast_to(r149_value, [$(98304)])
    tl.store($(candidate_vmul_41_ptr) + r149_z, r149_value)
    r150_z = tl.arange($(98304))
    r150_value = tl.load($(candidate_vmul_41_ptr) + ((((r150_z / $(192)) % $(8)) * $(12288)) + (((r150_z / $(1536)) % $(64)) * $(192)) + (r150_z % $(192))), dtype=tl.float32)
    r150_value = r150_value.to(tl.bfloat16)
    r150_value = tl.broadcast_to(r150_value, [$(98304)])
    tl.store($(candidate_vto_copy_88_ptr) + r150_z, r150_value)
    r151_row = tl.arange($(64))
    r151_col = tl.arange($(128))
    r151_lane = tl.arange($(32))
    for r151_group in range($(8)) {
      r151_left = tl.load($(candidate_vexp_1_ptr) + (((((((r151_group % $(8)) * $(2048)) + (r151_row[:, None] * $(32)) + r151_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + (((((r151_group % $(8)) * $(2048)) + (r151_row[:, None] * $(32)) + r151_lane[None, :]) % $(32)) * $(64)) + (((((r151_group % $(8)) * $(2048)) + (r151_row[:, None] * $(32)) + r151_lane[None, :]) / $(32)) % $(64))), dtype=tl.float32)
      r151_right = tl.load($(candidate_vto_copy_85_ptr) + (((((((r151_group % $(8)) * $(4096)) + (r151_lane[:, None] * $(128)) + r151_col[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r151_group % $(8)) * $(4096)) + (r151_lane[:, None] * $(128)) + r151_col[None, :]) / $(128)) % $(32)) * $(128)) + ((((r151_group % $(8)) * $(4096)) + (r151_lane[:, None] * $(128)) + r151_col[None, :]) % $(128))), dtype=tl.float32)
      r151_value = tl.dot(r151_left, r151_right)
      r151_value = r151_value.to(tl.float32)
      tl.store($(candidate_vbmm_48_ptr) + ((r151_group * $(8192)) + (r151_row[:, None] * $(128)) + r151_col[None, :]), r151_value)
    }
    r152_z = tl.arange($(65536))
    r152_value = tl.load($(candidate_vbmm_48_ptr) + ((((r152_z / $(128)) % $(8)) * $(8192)) + (((r152_z / $(1024)) % $(64)) * $(128)) + (r152_z % $(128))), dtype=tl.float32)
    r152_value = r152_value.to(tl.bfloat16)
    r152_value = tl.broadcast_to(r152_value, [$(65536)])
    tl.store($(candidate_vto_copy_89_ptr) + r152_z, r152_value)
    r153_z = tl.arange($(147456))
    r153_value = tl.where(((r153_z / $(1536)) % $(96)) < $(64), tl.load($(candidate_vto_copy_88_ptr) + ((((((r153_z / $(1536)) % $(96)) - $(0)) % $(64)) * $(1536)) + (((r153_z / $(192)) % $(8)) * $(192)) + (r153_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r153_value = r153_value.to(tl.bfloat16)
    r153_value = tl.broadcast_to(r153_value, [$(147456)])
    tl.store($(candidate_vcat_30_ptr) + r153_z, r153_value)
    r154_z = tl.arange($(98304))
    r154_value = tl.where(((r154_z / $(1024)) % $(96)) < $(64), tl.load($(candidate_vto_copy_89_ptr) + ((((((r154_z / $(1024)) % $(96)) - $(0)) % $(64)) * $(1024)) + (((r154_z / $(128)) % $(8)) * $(128)) + (r154_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r154_value = r154_value.to(tl.bfloat16)
    r154_value = tl.broadcast_to(r154_value, [$(98304)])
    tl.store($(candidate_vcat_31_ptr) + r154_z, r154_value)
    r155_z = tl.arange($(147456))
    r155_value = tl.load($(candidate_vcat_30_ptr) + ((((((r155_z / $(1536)) % $(96)) - $(0)) % $(96)) * $(1536)) + (((r155_z / $(192)) % $(8)) * $(192)) + (r155_z % $(192))), dtype=tl.bfloat16)
    r155_value = r155_value.to(tl.bfloat16)
    r155_value = tl.broadcast_to(r155_value, [$(147456)])
    tl.store($(candidate_vcat_33_ptr) + r155_z, r155_value)
    r156_z = tl.arange($(98304))
    r156_value = tl.load($(candidate_vcat_31_ptr) + ((((((r156_z / $(1024)) % $(96)) - $(0)) % $(96)) * $(1024)) + (((r156_z / $(128)) % $(8)) * $(128)) + (r156_z % $(128))), dtype=tl.bfloat16)
    r156_value = r156_value.to(tl.bfloat16)
    r156_value = tl.broadcast_to(r156_value, [$(98304)])
    tl.store($(candidate_vcat_34_ptr) + r156_z, r156_value)
    r157_row = tl.arange($(1024))
    r157_col = tl.arange($(512))
    r157_lane = tl.arange($(96))
    for r157_group in range($(1)) {
      r157_left = tl.load($(candidate_vcat_33_ptr) + (((((r157_row[:, None] * $(96)) + r157_lane[None, :]) % $(96)) * $(1536)) + (((((r157_row[:, None] * $(96)) + r157_lane[None, :]) / $(12288)) % $(8)) * $(192)) + ((((r157_row[:, None] * $(96)) + r157_lane[None, :]) / $(96)) % $(128))), dtype=tl.bfloat16)
      r157_right = tl.load($(candidate_vcat_7_ptr) + ((((((r157_lane[:, None] * $(512)) + r157_col[None, :]) / $(512)) % $(96)) * $(576)) + (((r157_lane[:, None] * $(512)) + r157_col[None, :]) % $(512)) + $(110592)), dtype=tl.bfloat16)
      r157_value = tl.dot(r157_left, r157_right)
      r157_value = r157_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_51_ptr) + ((r157_group * $(524288)) + (r157_row[:, None] * $(512)) + r157_col[None, :]), r157_value)
    }
    r158_row = tl.arange($(1024))
    r158_col = tl.arange($(512))
    r158_lane = tl.arange($(96))
    for r158_group in range($(1)) {
      r158_left = tl.load($(candidate_vcat_34_ptr) + (((((r158_row[:, None] * $(96)) + r158_lane[None, :]) % $(96)) * $(1024)) + (((((r158_row[:, None] * $(96)) + r158_lane[None, :]) / $(12288)) % $(8)) * $(128)) + ((((r158_row[:, None] * $(96)) + r158_lane[None, :]) / $(96)) % $(128))), dtype=tl.bfloat16)
      r158_right = tl.load($(candidate_vcat_7_ptr) + ((((((r158_lane[:, None] * $(512)) + r158_col[None, :]) / $(512)) % $(96)) * $(576)) + (((r158_lane[:, None] * $(512)) + r158_col[None, :]) % $(512)) + $(110592)), dtype=tl.bfloat16)
      r158_value = tl.dot(r158_left, r158_right)
      r158_value = r158_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_52_ptr) + ((r158_group * $(524288)) + (r158_row[:, None] * $(512)) + r158_col[None, :]), r158_value)
    }
    r159_z = tl.arange($(1048576))
    r159_value = tl.where(((r159_z / $(512)) % $(256)) < $(128), tl.load($(candidate_vbmm_51_ptr) + ((((((((r159_z / $(131072)) % $(8)) * $(65536)) + ((((r159_z / $(512)) % $(256)) - $(0)) * $(512)) + (r159_z % $(512))) / $(65536)) % $(8)) * $(65536)) + ((((((r159_z / $(131072)) % $(8)) * $(128)) + (((r159_z / $(512)) % $(256)) - $(0))) % $(128)) * $(512)) + (r159_z % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vbmm_52_ptr) + ((((((((r159_z / $(131072)) % $(8)) * $(65536)) + ((((r159_z / $(512)) % $(256)) - $(128)) * $(512)) + (r159_z % $(512))) / $(65536)) % $(8)) * $(65536)) + ((((((r159_z / $(131072)) % $(8)) * $(128)) + (((r159_z / $(512)) % $(256)) - $(128))) % $(128)) * $(512)) + (r159_z % $(512))), dtype=tl.bfloat16))
    r159_value = r159_value.to(tl.bfloat16)
    r159_value = tl.broadcast_to(r159_value, [$(1048576)])
    tl.store($(candidate_vcat_35_ptr) + r159_z, r159_value)
    r160_z = tl.arange($(1048576))
    r160_value = (tl.load($(candidate_vcat_25_ptr) + ((((((((r160_z / $(512)) % $(2048)) * $(512)) + (r160_z % $(512))) / $(131072)) % $(8)) * $(131072)) + ((((r160_z / $(512)) % $(2048)) % $(256)) * $(512)) + (r160_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vcat_35_ptr) + ((((((((r160_z / $(512)) % $(2048)) * $(512)) + (r160_z % $(512))) / $(131072)) % $(8)) * $(131072)) + ((((r160_z / $(512)) % $(2048)) % $(256)) * $(512)) + (r160_z % $(512))), dtype=tl.bfloat16))
    r160_value = r160_value.to(tl.bfloat16)
    r160_value = tl.broadcast_to(r160_value, [$(1048576)])
    tl.store($(candidate_vadd_30_ptr) + r160_z, r160_value)
    r161_row = tl.arange($(96))
    r161_col = tl.arange($(1024))
    r161_lane = tl.arange($(512))
    for r161_group in range($(1)) {
      r161_left = tl.load($(candidate_vcat_7_ptr) + ((((((r161_row[:, None] * $(512)) + r161_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r161_row[:, None] * $(512)) + r161_lane[None, :]) % $(512)) + $(110592)), dtype=tl.bfloat16)
      r161_right = tl.load($(candidate_vto_copy_18_ptr) + ((((((r161_lane[:, None] * $(1024)) + r161_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r161_lane[:, None] * $(1024)) + r161_col[None, :]) % $(128)) * $(512)) + ((((r161_lane[:, None] * $(1024)) + r161_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r161_value = tl.dot(r161_left, r161_right)
      r161_value = r161_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_53_ptr) + ((r161_group * $(98304)) + (r161_row[:, None] * $(1024)) + r161_col[None, :]), r161_value)
    }
    r162_row = tl.arange($(96))
    r162_col = tl.arange($(1024))
    r162_lane = tl.arange($(512))
    for r162_group in range($(1)) {
      r162_left = tl.load($(candidate_vcat_7_ptr) + ((((((r162_row[:, None] * $(512)) + r162_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r162_row[:, None] * $(512)) + r162_lane[None, :]) % $(512)) + $(110592)), dtype=tl.bfloat16)
      r162_right = tl.load($(candidate_vto_copy_18_ptr) + ((((((r162_lane[:, None] * $(1024)) + r162_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r162_lane[:, None] * $(1024)) + r162_col[None, :]) % $(128)) * $(512)) + ((((r162_lane[:, None] * $(1024)) + r162_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r162_value = tl.dot(r162_left, r162_right)
      r162_value = r162_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_54_ptr) + ((r162_group * $(98304)) + (r162_row[:, None] * $(1024)) + r162_col[None, :]), r162_value)
    }
    r163_z = tl.arange($(49152))
    r163_value = tl.where((r163_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r163_z / $(1536)) % $(32)) * $(1024)) + (((r163_z / $(192)) % $(8)) * $(128)) + ((r163_z % $(192)) - $(0)) + $(196608)) / $(1024)) % $(288)) * $(1536)) + (((((((r163_z / $(1536)) % $(32)) * $(1024)) + (((r163_z / $(192)) % $(8)) * $(128)) + ((r163_z % $(192)) - $(0)) + $(196608)) / $(128)) % $(8)) * $(192)) + (((((r163_z / $(1536)) % $(32)) * $(1024)) + (((r163_z / $(192)) % $(8)) * $(128)) + ((r163_z % $(192)) - $(0)) + $(196608)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_5_ptr) + ((((((((r163_z / $(1536)) % $(32)) * $(512)) + (((r163_z / $(192)) % $(8)) * $(64)) + ((r163_z % $(192)) - $(128)) + $(98304)) / $(64)) % $(8)) * $(18432)) + (((((((r163_z / $(1536)) % $(32)) * $(512)) + (((r163_z / $(192)) % $(8)) * $(64)) + ((r163_z % $(192)) - $(128)) + $(98304)) / $(512)) % $(288)) * $(64)) + (((((r163_z / $(1536)) % $(32)) * $(512)) + (((r163_z / $(192)) % $(8)) * $(64)) + ((r163_z % $(192)) - $(128)) + $(98304)) % $(64))), dtype=tl.bfloat16))
    r163_value = r163_value.to(tl.bfloat16)
    r163_value = tl.broadcast_to(r163_value, [$(49152)])
    tl.store($(candidate_vcat_36_ptr) + r163_z, r163_value)
    r164_z = tl.arange($(147456))
    r164_value = tl.where((r164_z % $(192)) < $(128), tl.load($(candidate_vbmm_53_ptr) + ((((((((r164_z / $(1536)) % $(96)) * $(1024)) + (((r164_z / $(192)) % $(8)) * $(128)) + ((r164_z % $(192)) - $(0))) / $(1024)) % $(96)) * $(1024)) + (((((((r164_z / $(1536)) % $(96)) * $(1024)) + (((r164_z / $(192)) % $(8)) * $(128)) + ((r164_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r164_z / $(1536)) % $(96)) * $(1024)) + (((r164_z / $(192)) % $(8)) * $(128)) + ((r164_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_7_ptr) + ((((r164_z / $(1536)) % $(96)) * $(576)) + ((((r164_z % $(192)) - $(128)) + $(512)) % $(576)) + $(110592)), dtype=tl.bfloat16))
    r164_value = r164_value.to(tl.bfloat16)
    r164_value = tl.broadcast_to(r164_value, [$(147456)])
    tl.store($(candidate_vcat_37_ptr) + r164_z, r164_value)
    r165_z = tl.arange($(49152))
    r165_value = tl.load($(candidate_vcat_36_ptr) + ((((r165_z / $(192)) % $(32)) * $(1536)) + (((r165_z / $(6144)) % $(8)) * $(192)) + (r165_z % $(192))), dtype=tl.bfloat16)
    r165_value = r165_value.to(tl.float32)
    r165_value = tl.broadcast_to(r165_value, [$(49152)])
    tl.store($(candidate_vto_copy_90_ptr) + r165_z, r165_value)
    r166_z = tl.arange($(49152))
    r166_value = tl.load($(candidate_vcat_37_ptr) + ((((r166_z / $(192)) % $(32)) * $(1536)) + (((r166_z / $(6144)) % $(8)) * $(192)) + (r166_z % $(192))), dtype=tl.bfloat16)
    r166_value = r166_value.to(tl.float32)
    r166_value = tl.broadcast_to(r166_value, [$(49152)])
    tl.store($(candidate_vto_copy_91_ptr) + r166_z, r166_value)
    r167_row = tl.arange($(32))
    r167_col = tl.arange($(32))
    r167_lane = tl.arange($(192))
    for r167_group in range($(8)) {
      r167_left = tl.load($(candidate_vto_copy_90_ptr) + (((((((r167_group % $(8)) * $(6144)) + (r167_row[:, None] * $(192)) + r167_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r167_group % $(8)) * $(6144)) + (r167_row[:, None] * $(192)) + r167_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r167_group % $(8)) * $(6144)) + (r167_row[:, None] * $(192)) + r167_lane[None, :]) % $(192))), dtype=tl.float32)
      r167_right = tl.load($(candidate_vto_copy_91_ptr) + (((((((r167_group % $(8)) * $(6144)) + (r167_lane[:, None] * $(32)) + r167_col[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r167_group % $(8)) * $(6144)) + (r167_lane[:, None] * $(32)) + r167_col[None, :]) % $(32)) * $(192)) + (((((r167_group % $(8)) * $(6144)) + (r167_lane[:, None] * $(32)) + r167_col[None, :]) / $(32)) % $(192))), dtype=tl.float32)
      r167_value = tl.dot(r167_left, r167_right)
      r167_value = r167_value.to(tl.float32)
      tl.store($(candidate_vbmm_55_ptr) + ((r167_group * $(1024)) + (r167_row[:, None] * $(32)) + r167_col[None, :]), r167_value)
    }
    r168_z = tl.arange($(8192))
    r168_value = (tl.load($(candidate_vbmm_55_ptr) + ((((r168_z / $(1024)) % $(8)) * $(1024)) + (((r168_z / $(32)) % $(32)) * $(32)) + (r168_z % $(32))), dtype=tl.float32) * 0.07216878364870322)
    r168_value = r168_value.to(tl.float32)
    r168_value = tl.broadcast_to(r168_value, [$(8192)])
    tl.store($(candidate_vmul_42_ptr) + r168_z, r168_value)
    r169_z = tl.arange($(32))
    r169_value = (tl.cast(r169_z, tl.int64) + tl.full([], $(32), dtype=tl.int64))
    r169_value = tl.broadcast_to(r169_value, [$(32)])
    tl.store($(candidate_vadd_31_ptr) + r169_z, r169_value)
    r170_z = tl.arange($(32))
    r170_value = (tl.load($(candidate_vadd_31_ptr) + r170_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r170_value = tl.broadcast_to(r170_value, [$(32)])
    tl.store($(candidate_vsub_32_ptr) + r170_z, r170_value)
    r171_z = tl.arange($(1024))
    r171_value = (tl.cast((r171_z % $(32)), tl.int64) > tl.load($(candidate_vsub_32_ptr) + ((r171_z / $(32)) % $(32)), dtype=tl.int64))
    r171_value = tl.broadcast_to(r171_value, [$(1024)])
    tl.store($(candidate_vgt_13_ptr) + r171_z, r171_value)
    r172_z = tl.arange($(8192))
    r172_value = tl.where(tl.load($(candidate_vgt_13_ptr) + ((((r172_z / $(32)) % $(32)) * $(32)) + (r172_z % $(32))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_42_ptr) + ((((r172_z / $(1024)) % $(8)) * $(1024)) + (((r172_z / $(32)) % $(32)) * $(32)) + (r172_z % $(32))), dtype=tl.float32))
    r172_value = r172_value.to(tl.float32)
    r172_value = tl.broadcast_to(r172_value, [$(8192)])
    tl.store($(candidate_vmasked_fill_13_ptr) + r172_z, r172_value)
    r173_z = tl.arange($(8192))
    r173_lane = tl.arange($(32))
    r173_row_values = tl.load($(candidate_vmasked_fill_13_ptr) + ((((r173_z[:, None] / $(1024)) % $(8)) * $(1024)) + (((r173_z[:, None] / $(32)) % $(32)) * $(32)) + (r173_lane[None, :] % $(32))), dtype=tl.float32)
    r173_maximum = tl.max(r173_row_values, axis=1, keep_dims=true)
    r173_weights = tl.exp(r173_row_values - r173_maximum)
    r173_total = tl.sum(r173_weights, axis=1)
    r173_maximum = tl.max(r173_row_values, axis=1)
    r173_value = tl.exp(tl.load($(candidate_vmasked_fill_13_ptr) + ((((r173_z / $(1024)) % $(8)) * $(1024)) + (((r173_z / $(32)) % $(32)) * $(32)) + (r173_z % $(32))), dtype=tl.float32) - r173_maximum) / r173_total
    r173_value = r173_value.to(tl.float32)
    r173_value = tl.broadcast_to(r173_value, [$(8192)])
    tl.store($(candidate_vsoftmax_11_ptr) + r173_z, r173_value)
    r174_z = tl.arange($(8192))
    r174_value = tl.load($(candidate_vsoftmax_11_ptr) + ((((r174_z / $(1024)) % $(8)) * $(1024)) + (((r174_z / $(32)) % $(32)) * $(32)) + (r174_z % $(32))), dtype=tl.float32)
    r174_value = r174_value.to(tl.bfloat16)
    r174_value = tl.broadcast_to(r174_value, [$(8192)])
    tl.store($(candidate_vto_copy_92_ptr) + r174_z, r174_value)
    r175_row = tl.arange($(32))
    r175_col = tl.arange($(128))
    r175_lane = tl.arange($(32))
    for r175_group in range($(8)) {
      r175_left = tl.load($(candidate_vto_copy_92_ptr) + (((((((r175_group % $(8)) * $(1024)) + (r175_row[:, None] * $(32)) + r175_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + ((((((r175_group % $(8)) * $(1024)) + (r175_row[:, None] * $(32)) + r175_lane[None, :]) / $(32)) % $(32)) * $(32)) + ((((r175_group % $(8)) * $(1024)) + (r175_row[:, None] * $(32)) + r175_lane[None, :]) % $(32))), dtype=tl.bfloat16)
      r175_right = tl.load($(candidate_vbmm_54_ptr) + (((((((r175_group % $(8)) * $(4096)) + (r175_lane[:, None] * $(128)) + r175_col[None, :]) / $(128)) % $(32)) * $(1024)) + ((((((r175_group % $(8)) * $(4096)) + (r175_lane[:, None] * $(128)) + r175_col[None, :]) / $(4096)) % $(8)) * $(128)) + ((((r175_group % $(8)) * $(4096)) + (r175_lane[:, None] * $(128)) + r175_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r175_value = tl.dot(r175_left, r175_right)
      r175_value = r175_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_56_ptr) + ((r175_group * $(4096)) + (r175_row[:, None] * $(128)) + r175_col[None, :]), r175_value)
    }
    r176_z = tl.arange($(256))
    r176_lane = tl.arange($(32))
    r176_row_values = tl.load($(candidate_vmasked_fill_13_ptr) + ((((r176_z[:, None] / $(32)) % $(8)) * $(1024)) + ((r176_z[:, None] % $(32)) * $(32)) + (r176_lane[None, :] % $(32))), dtype=tl.float32)
    r176_maximum = tl.max(r176_row_values, axis=1, keep_dims=true)
    r176_weights = tl.exp(r176_row_values - r176_maximum)
    r176_total = tl.sum(r176_weights, axis=1)
    r176_maximum = tl.max(r176_row_values, axis=1)
    r176_value = r176_maximum + tl.log(r176_total)
    r176_value = r176_value.to(tl.float32)
    r176_value = tl.broadcast_to(r176_value, [$(256)])
    tl.store($(candidate_vlogsumexp_11_ptr) + r176_z, r176_value)
    r177_z = tl.arange($(32768))
    r177_value = tl.load($(candidate_vbmm_56_ptr) + ((((((((r177_z / $(128)) % $(8)) * $(4096)) + ((((r177_z / $(1024)) % $(32)) - $(0)) * $(128)) + (r177_z % $(128))) / $(4096)) % $(8)) * $(4096)) + ((((((r177_z / $(128)) % $(8)) * $(32)) + (((r177_z / $(1024)) % $(32)) - $(0))) % $(32)) * $(128)) + (r177_z % $(128))), dtype=tl.bfloat16)
    r177_value = r177_value.to(tl.bfloat16)
    r177_value = tl.broadcast_to(r177_value, [$(32768)])
    tl.store($(candidate_vcat_38_ptr) + r177_z, r177_value)
    r178_z = tl.arange($(256))
    r178_value = tl.load($(candidate_vlogsumexp_11_ptr) + (((r178_z % $(8)) * $(32)) + ((((r178_z / $(8)) % $(32)) - $(0)) % $(32))), dtype=tl.float32)
    r178_value = r178_value.to(tl.float32)
    r178_value = tl.broadcast_to(r178_value, [$(256)])
    tl.store($(candidate_vcat_39_ptr) + r178_z, r178_value)
    r179_z = tl.arange($(49152))
    r179_value = tl.load($(candidate_vcat_36_ptr) + ((((r179_z / $(192)) % $(32)) * $(1536)) + (((r179_z / $(6144)) % $(8)) * $(192)) + (r179_z % $(192))), dtype=tl.bfloat16)
    r179_value = r179_value.to(tl.float32)
    r179_value = tl.broadcast_to(r179_value, [$(49152)])
    tl.store($(candidate_vto_copy_93_ptr) + r179_z, r179_value)
    r180_z = tl.arange($(49152))
    r180_value = tl.load($(candidate_vcat_37_ptr) + ((((r180_z / $(192)) % $(32)) * $(1536)) + (((r180_z / $(6144)) % $(8)) * $(192)) + (r180_z % $(192))), dtype=tl.bfloat16)
    r180_value = r180_value.to(tl.float32)
    r180_value = tl.broadcast_to(r180_value, [$(49152)])
    tl.store($(candidate_vto_copy_94_ptr) + r180_z, r180_value)
    r181_z = tl.arange($(32768))
    r181_value = tl.load($(candidate_vbmm_54_ptr) + ((((r181_z / $(128)) % $(32)) * $(1024)) + (((r181_z / $(4096)) % $(8)) * $(128)) + (r181_z % $(128))), dtype=tl.bfloat16)
    r181_value = r181_value.to(tl.float32)
    r181_value = tl.broadcast_to(r181_value, [$(32768)])
    tl.store($(candidate_vto_copy_95_ptr) + r181_z, r181_value)
    r182_z = tl.arange($(32768))
    r182_value = tl.load($(candidate_vmm_22_ptr) + ((((r182_z / $(128)) % $(32)) * $(1024)) + (((r182_z / $(4096)) % $(8)) * $(128)) + (r182_z % $(128)) + $(196608)), dtype=tl.bfloat16)
    r182_value = r182_value.to(tl.float32)
    r182_value = tl.broadcast_to(r182_value, [$(32768)])
    tl.store($(candidate_vto_copy_96_ptr) + r182_z, r182_value)
    r183_z = tl.arange($(32768))
    r183_value = tl.load($(candidate_vcat_38_ptr) + ((((r183_z / $(128)) % $(32)) * $(1024)) + (((r183_z / $(4096)) % $(8)) * $(128)) + (r183_z % $(128))), dtype=tl.bfloat16)
    r183_value = r183_value.to(tl.float32)
    r183_value = tl.broadcast_to(r183_value, [$(32768)])
    tl.store($(candidate_vto_copy_97_ptr) + r183_z, r183_value)
    r184_row = tl.arange($(32))
    r184_col = tl.arange($(32))
    r184_lane = tl.arange($(192))
    for r184_group in range($(8)) {
      r184_left = tl.load($(candidate_vto_copy_93_ptr) + (((((((r184_group % $(8)) * $(6144)) + (r184_row[:, None] * $(192)) + r184_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r184_group % $(8)) * $(6144)) + (r184_row[:, None] * $(192)) + r184_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r184_group % $(8)) * $(6144)) + (r184_row[:, None] * $(192)) + r184_lane[None, :]) % $(192))), dtype=tl.float32)
      r184_right = tl.load($(candidate_vto_copy_94_ptr) + (((((((r184_group % $(8)) * $(6144)) + (r184_lane[:, None] * $(32)) + r184_col[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r184_group % $(8)) * $(6144)) + (r184_lane[:, None] * $(32)) + r184_col[None, :]) % $(32)) * $(192)) + (((((r184_group % $(8)) * $(6144)) + (r184_lane[:, None] * $(32)) + r184_col[None, :]) / $(32)) % $(192))), dtype=tl.float32)
      r184_value = tl.dot(r184_left, r184_right)
      r184_value = r184_value.to(tl.float32)
      tl.store($(candidate_vbmm_57_ptr) + ((r184_group * $(1024)) + (r184_row[:, None] * $(32)) + r184_col[None, :]), r184_value)
    }
    r185_z = tl.arange($(8192))
    r185_value = (tl.load($(candidate_vbmm_57_ptr) + ((((r185_z / $(1024)) % $(8)) * $(1024)) + (((r185_z / $(32)) % $(32)) * $(32)) + (r185_z % $(32))), dtype=tl.float32) * 0.07216878364870322)
    r185_value = r185_value.to(tl.float32)
    r185_value = tl.broadcast_to(r185_value, [$(8192)])
    tl.store($(candidate_vmul_43_ptr) + r185_z, r185_value)
    r186_z = tl.arange($(32))
    r186_value = (tl.cast(r186_z, tl.int64) + tl.full([], $(32), dtype=tl.int64))
    r186_value = tl.broadcast_to(r186_value, [$(32)])
    tl.store($(candidate_vadd_32_ptr) + r186_z, r186_value)
    r187_z = tl.arange($(32))
    r187_value = (tl.load($(candidate_vadd_32_ptr) + r187_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r187_value = tl.broadcast_to(r187_value, [$(32)])
    tl.store($(candidate_vsub_33_ptr) + r187_z, r187_value)
    r188_z = tl.arange($(1024))
    r188_value = (tl.cast((r188_z % $(32)), tl.int64) > tl.load($(candidate_vsub_33_ptr) + ((r188_z / $(32)) % $(32)), dtype=tl.int64))
    r188_value = tl.broadcast_to(r188_value, [$(1024)])
    tl.store($(candidate_vgt_14_ptr) + r188_z, r188_value)
    r189_z = tl.arange($(8192))
    r189_value = tl.where(tl.load($(candidate_vgt_14_ptr) + ((((r189_z / $(32)) % $(32)) * $(32)) + (r189_z % $(32))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_43_ptr) + ((((r189_z / $(1024)) % $(8)) * $(1024)) + (((r189_z / $(32)) % $(32)) * $(32)) + (r189_z % $(32))), dtype=tl.float32))
    r189_value = r189_value.to(tl.float32)
    r189_value = tl.broadcast_to(r189_value, [$(8192)])
    tl.store($(candidate_vmasked_fill_14_ptr) + r189_z, r189_value)
    r190_z = tl.arange($(8192))
    r190_value = (tl.load($(candidate_vmasked_fill_14_ptr) + ((((r190_z / $(1024)) % $(8)) * $(1024)) + (((r190_z / $(32)) % $(32)) * $(32)) + (r190_z % $(32))), dtype=tl.float32) - tl.load($(candidate_vcat_39_ptr) + ((((r190_z / $(32)) % $(32)) * $(8)) + ((r190_z / $(1024)) % $(8))), dtype=tl.float32))
    r190_value = r190_value.to(tl.float32)
    r190_value = tl.broadcast_to(r190_value, [$(8192)])
    tl.store($(candidate_vsub_34_ptr) + r190_z, r190_value)
    r191_z = tl.arange($(8192))
    r191_value = tl.exp(tl.load($(candidate_vsub_34_ptr) + ((((r191_z / $(1024)) % $(8)) * $(1024)) + (((r191_z / $(32)) % $(32)) * $(32)) + (r191_z % $(32))), dtype=tl.float32))
    r191_value = r191_value.to(tl.float32)
    r191_value = tl.broadcast_to(r191_value, [$(8192)])
    tl.store($(candidate_vexp_2_ptr) + r191_z, r191_value)
    r192_z = tl.arange($(32768))
    r192_value = (tl.load($(candidate_vto_copy_96_ptr) + ((((r192_z / $(4096)) % $(8)) * $(4096)) + (((r192_z / $(128)) % $(32)) * $(128)) + (r192_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_97_ptr) + ((((r192_z / $(4096)) % $(8)) * $(4096)) + (((r192_z / $(128)) % $(32)) * $(128)) + (r192_z % $(128))), dtype=tl.float32))
    r192_value = r192_value.to(tl.float32)
    r192_value = tl.broadcast_to(r192_value, [$(32768)])
    tl.store($(candidate_vmul_44_ptr) + r192_z, r192_value)
    r193_z = tl.arange($(256))
    r193_lane = tl.arange($(128))
    r193_value = tl.sum(tl.load($(candidate_vmul_44_ptr) + ((((r193_z[:, None] / $(32)) % $(8)) * $(4096)) + ((r193_z[:, None] % $(32)) * $(128)) + r193_lane[None, :]), dtype=tl.float32), axis=1)
    r193_value = r193_value.to(tl.float32)
    r193_value = tl.broadcast_to(r193_value, [$(256)])
    tl.store($(candidate_vsum_5_ptr) + r193_z, r193_value)
    r194_row = tl.arange($(32))
    r194_col = tl.arange($(32))
    r194_lane = tl.arange($(128))
    for r194_group in range($(8)) {
      r194_left = tl.load($(candidate_vto_copy_96_ptr) + (((((((r194_group % $(8)) * $(4096)) + (r194_row[:, None] * $(128)) + r194_lane[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r194_group % $(8)) * $(4096)) + (r194_row[:, None] * $(128)) + r194_lane[None, :]) / $(128)) % $(32)) * $(128)) + ((((r194_group % $(8)) * $(4096)) + (r194_row[:, None] * $(128)) + r194_lane[None, :]) % $(128))), dtype=tl.float32)
      r194_right = tl.load($(candidate_vto_copy_95_ptr) + (((((((r194_group % $(8)) * $(4096)) + (r194_lane[:, None] * $(32)) + r194_col[None, :]) / $(4096)) % $(8)) * $(4096)) + (((((r194_group % $(8)) * $(4096)) + (r194_lane[:, None] * $(32)) + r194_col[None, :]) % $(32)) * $(128)) + (((((r194_group % $(8)) * $(4096)) + (r194_lane[:, None] * $(32)) + r194_col[None, :]) / $(32)) % $(128))), dtype=tl.float32)
      r194_value = tl.dot(r194_left, r194_right)
      r194_value = r194_value.to(tl.float32)
      tl.store($(candidate_vbmm_58_ptr) + ((r194_group * $(1024)) + (r194_row[:, None] * $(32)) + r194_col[None, :]), r194_value)
    }
    r195_z = tl.arange($(8192))
    r195_value = (tl.load($(candidate_vbmm_58_ptr) + ((((r195_z / $(1024)) % $(8)) * $(1024)) + (((r195_z / $(32)) % $(32)) * $(32)) + (r195_z % $(32))), dtype=tl.float32) - tl.load($(candidate_vsum_5_ptr) + ((((r195_z / $(1024)) % $(8)) * $(32)) + ((r195_z / $(32)) % $(32))), dtype=tl.float32))
    r195_value = r195_value.to(tl.float32)
    r195_value = tl.broadcast_to(r195_value, [$(8192)])
    tl.store($(candidate_vsub_35_ptr) + r195_z, r195_value)
    r196_z = tl.arange($(8192))
    r196_value = (tl.load($(candidate_vexp_2_ptr) + ((((r196_z / $(1024)) % $(8)) * $(1024)) + (((r196_z / $(32)) % $(32)) * $(32)) + (r196_z % $(32))), dtype=tl.float32) * tl.load($(candidate_vsub_35_ptr) + ((((r196_z / $(1024)) % $(8)) * $(1024)) + (((r196_z / $(32)) % $(32)) * $(32)) + (r196_z % $(32))), dtype=tl.float32))
    r196_value = r196_value.to(tl.float32)
    r196_value = tl.broadcast_to(r196_value, [$(8192)])
    tl.store($(candidate_vmul_45_ptr) + r196_z, r196_value)
    r197_row = tl.arange($(32))
    r197_col = tl.arange($(192))
    r197_lane = tl.arange($(32))
    for r197_group in range($(8)) {
      r197_left = tl.load($(candidate_vmul_45_ptr) + (((((((r197_group % $(8)) * $(1024)) + (r197_row[:, None] * $(32)) + r197_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + (((((r197_group % $(8)) * $(1024)) + (r197_row[:, None] * $(32)) + r197_lane[None, :]) % $(32)) * $(32)) + (((((r197_group % $(8)) * $(1024)) + (r197_row[:, None] * $(32)) + r197_lane[None, :]) / $(32)) % $(32))), dtype=tl.float32)
      r197_right = tl.load($(candidate_vto_copy_93_ptr) + (((((((r197_group % $(8)) * $(6144)) + (r197_lane[:, None] * $(192)) + r197_col[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r197_group % $(8)) * $(6144)) + (r197_lane[:, None] * $(192)) + r197_col[None, :]) / $(192)) % $(32)) * $(192)) + ((((r197_group % $(8)) * $(6144)) + (r197_lane[:, None] * $(192)) + r197_col[None, :]) % $(192))), dtype=tl.float32)
      r197_value = tl.dot(r197_left, r197_right)
      r197_value = r197_value.to(tl.float32)
      tl.store($(candidate_vbmm_60_ptr) + ((r197_group * $(6144)) + (r197_row[:, None] * $(192)) + r197_col[None, :]), r197_value)
    }
    r198_z = tl.arange($(49152))
    r198_value = (tl.load($(candidate_vbmm_60_ptr) + ((((r198_z / $(6144)) % $(8)) * $(6144)) + (((r198_z / $(192)) % $(32)) * $(192)) + (r198_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r198_value = r198_value.to(tl.float32)
    r198_value = tl.broadcast_to(r198_value, [$(49152)])
    tl.store($(candidate_vmul_47_ptr) + r198_z, r198_value)
    r199_z = tl.arange($(49152))
    r199_value = tl.load($(candidate_vmul_47_ptr) + ((((r199_z / $(192)) % $(8)) * $(6144)) + (((r199_z / $(1536)) % $(32)) * $(192)) + (r199_z % $(192))), dtype=tl.float32)
    r199_value = r199_value.to(tl.bfloat16)
    r199_value = tl.broadcast_to(r199_value, [$(49152)])
    tl.store($(candidate_vto_copy_99_ptr) + r199_z, r199_value)
    r200_row = tl.arange($(32))
    r200_col = tl.arange($(128))
    r200_lane = tl.arange($(32))
    for r200_group in range($(8)) {
      r200_left = tl.load($(candidate_vexp_2_ptr) + (((((((r200_group % $(8)) * $(1024)) + (r200_row[:, None] * $(32)) + r200_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + (((((r200_group % $(8)) * $(1024)) + (r200_row[:, None] * $(32)) + r200_lane[None, :]) % $(32)) * $(32)) + (((((r200_group % $(8)) * $(1024)) + (r200_row[:, None] * $(32)) + r200_lane[None, :]) / $(32)) % $(32))), dtype=tl.float32)
      r200_right = tl.load($(candidate_vto_copy_96_ptr) + (((((((r200_group % $(8)) * $(4096)) + (r200_lane[:, None] * $(128)) + r200_col[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r200_group % $(8)) * $(4096)) + (r200_lane[:, None] * $(128)) + r200_col[None, :]) / $(128)) % $(32)) * $(128)) + ((((r200_group % $(8)) * $(4096)) + (r200_lane[:, None] * $(128)) + r200_col[None, :]) % $(128))), dtype=tl.float32)
      r200_value = tl.dot(r200_left, r200_right)
      r200_value = r200_value.to(tl.float32)
      tl.store($(candidate_vbmm_61_ptr) + ((r200_group * $(4096)) + (r200_row[:, None] * $(128)) + r200_col[None, :]), r200_value)
    }
    r201_z = tl.arange($(32768))
    r201_value = tl.load($(candidate_vbmm_61_ptr) + ((((r201_z / $(128)) % $(8)) * $(4096)) + (((r201_z / $(1024)) % $(32)) * $(128)) + (r201_z % $(128))), dtype=tl.float32)
    r201_value = r201_value.to(tl.bfloat16)
    r201_value = tl.broadcast_to(r201_value, [$(32768)])
    tl.store($(candidate_vto_copy_100_ptr) + r201_z, r201_value)
    r202_z = tl.arange($(147456))
    r202_value = tl.where(((r202_z / $(1536)) % $(96)) < $(32), tl.load($(candidate_vto_copy_99_ptr) + ((((((r202_z / $(1536)) % $(96)) - $(0)) % $(32)) * $(1536)) + (((r202_z / $(192)) % $(8)) * $(192)) + (r202_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r202_value = r202_value.to(tl.bfloat16)
    r202_value = tl.broadcast_to(r202_value, [$(147456)])
    tl.store($(candidate_vcat_40_ptr) + r202_z, r202_value)
    r203_z = tl.arange($(98304))
    r203_value = tl.where(((r203_z / $(1024)) % $(96)) < $(32), tl.load($(candidate_vto_copy_100_ptr) + ((((((r203_z / $(1024)) % $(96)) - $(0)) % $(32)) * $(1024)) + (((r203_z / $(128)) % $(8)) * $(128)) + (r203_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r203_value = r203_value.to(tl.bfloat16)
    r203_value = tl.broadcast_to(r203_value, [$(98304)])
    tl.store($(candidate_vcat_41_ptr) + r203_z, r203_value)
    r204_z = tl.arange($(147456))
    r204_value = tl.load($(candidate_vcat_40_ptr) + ((((((r204_z / $(1536)) % $(96)) - $(0)) % $(96)) * $(1536)) + (((r204_z / $(192)) % $(8)) * $(192)) + (r204_z % $(192))), dtype=tl.bfloat16)
    r204_value = r204_value.to(tl.bfloat16)
    r204_value = tl.broadcast_to(r204_value, [$(147456)])
    tl.store($(candidate_vcat_43_ptr) + r204_z, r204_value)
    r205_z = tl.arange($(98304))
    r205_value = tl.load($(candidate_vcat_41_ptr) + ((((((r205_z / $(1024)) % $(96)) - $(0)) % $(96)) * $(1024)) + (((r205_z / $(128)) % $(8)) * $(128)) + (r205_z % $(128))), dtype=tl.bfloat16)
    r205_value = r205_value.to(tl.bfloat16)
    r205_value = tl.broadcast_to(r205_value, [$(98304)])
    tl.store($(candidate_vcat_44_ptr) + r205_z, r205_value)
    r206_row = tl.arange($(1024))
    r206_col = tl.arange($(512))
    r206_lane = tl.arange($(96))
    for r206_group in range($(1)) {
      r206_left = tl.load($(candidate_vcat_43_ptr) + (((((r206_row[:, None] * $(96)) + r206_lane[None, :]) % $(96)) * $(1536)) + (((((r206_row[:, None] * $(96)) + r206_lane[None, :]) / $(12288)) % $(8)) * $(192)) + ((((r206_row[:, None] * $(96)) + r206_lane[None, :]) / $(96)) % $(128))), dtype=tl.bfloat16)
      r206_right = tl.load($(candidate_vcat_7_ptr) + ((((((r206_lane[:, None] * $(512)) + r206_col[None, :]) / $(512)) % $(96)) * $(576)) + (((r206_lane[:, None] * $(512)) + r206_col[None, :]) % $(512)) + $(110592)), dtype=tl.bfloat16)
      r206_value = tl.dot(r206_left, r206_right)
      r206_value = r206_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_64_ptr) + ((r206_group * $(524288)) + (r206_row[:, None] * $(512)) + r206_col[None, :]), r206_value)
    }
    r207_row = tl.arange($(1024))
    r207_col = tl.arange($(512))
    r207_lane = tl.arange($(96))
    for r207_group in range($(1)) {
      r207_left = tl.load($(candidate_vcat_44_ptr) + (((((r207_row[:, None] * $(96)) + r207_lane[None, :]) % $(96)) * $(1024)) + (((((r207_row[:, None] * $(96)) + r207_lane[None, :]) / $(12288)) % $(8)) * $(128)) + ((((r207_row[:, None] * $(96)) + r207_lane[None, :]) / $(96)) % $(128))), dtype=tl.bfloat16)
      r207_right = tl.load($(candidate_vcat_7_ptr) + ((((((r207_lane[:, None] * $(512)) + r207_col[None, :]) / $(512)) % $(96)) * $(576)) + (((r207_lane[:, None] * $(512)) + r207_col[None, :]) % $(512)) + $(110592)), dtype=tl.bfloat16)
      r207_value = tl.dot(r207_left, r207_right)
      r207_value = r207_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_65_ptr) + ((r207_group * $(524288)) + (r207_row[:, None] * $(512)) + r207_col[None, :]), r207_value)
    }
    r208_z = tl.arange($(1048576))
    r208_value = tl.where(((r208_z / $(512)) % $(256)) < $(128), tl.load($(candidate_vbmm_64_ptr) + ((((((((r208_z / $(131072)) % $(8)) * $(65536)) + ((((r208_z / $(512)) % $(256)) - $(0)) * $(512)) + (r208_z % $(512))) / $(65536)) % $(8)) * $(65536)) + ((((((r208_z / $(131072)) % $(8)) * $(128)) + (((r208_z / $(512)) % $(256)) - $(0))) % $(128)) * $(512)) + (r208_z % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vbmm_65_ptr) + ((((((((r208_z / $(131072)) % $(8)) * $(65536)) + ((((r208_z / $(512)) % $(256)) - $(128)) * $(512)) + (r208_z % $(512))) / $(65536)) % $(8)) * $(65536)) + ((((((r208_z / $(131072)) % $(8)) * $(128)) + (((r208_z / $(512)) % $(256)) - $(128))) % $(128)) * $(512)) + (r208_z % $(512))), dtype=tl.bfloat16))
    r208_value = r208_value.to(tl.bfloat16)
    r208_value = tl.broadcast_to(r208_value, [$(1048576)])
    tl.store($(candidate_vcat_45_ptr) + r208_z, r208_value)
    r209_z = tl.arange($(1048576))
    r209_value = (tl.load($(candidate_vadd_30_ptr) + ((((r209_z / $(512)) % $(2048)) * $(512)) + (r209_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vcat_45_ptr) + ((((((((r209_z / $(512)) % $(2048)) * $(512)) + (r209_z % $(512))) / $(131072)) % $(8)) * $(131072)) + ((((r209_z / $(512)) % $(2048)) % $(256)) * $(512)) + (r209_z % $(512))), dtype=tl.bfloat16))
    r209_value = r209_value.to(tl.bfloat16)
    r209_value = tl.broadcast_to(r209_value, [$(1048576)])
    tl.store($(candidate_vadd_34_ptr) + r209_z, r209_value)
    r210_row = tl.arange($(96))
    r210_col = tl.arange($(1024))
    r210_lane = tl.arange($(512))
    for r210_group in range($(1)) {
      r210_left = tl.load($(candidate_vcat_7_ptr) + ((((((r210_row[:, None] * $(512)) + r210_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r210_row[:, None] * $(512)) + r210_lane[None, :]) % $(512)) + $(55296)), dtype=tl.bfloat16)
      r210_right = tl.load($(candidate_vto_copy_18_ptr) + ((((((r210_lane[:, None] * $(1024)) + r210_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r210_lane[:, None] * $(1024)) + r210_col[None, :]) % $(128)) * $(512)) + ((((r210_lane[:, None] * $(1024)) + r210_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r210_value = tl.dot(r210_left, r210_right)
      r210_value = r210_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_66_ptr) + ((r210_group * $(98304)) + (r210_row[:, None] * $(1024)) + r210_col[None, :]), r210_value)
    }
    r211_row = tl.arange($(96))
    r211_col = tl.arange($(1024))
    r211_lane = tl.arange($(512))
    for r211_group in range($(1)) {
      r211_left = tl.load($(candidate_vcat_7_ptr) + ((((((r211_row[:, None] * $(512)) + r211_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r211_row[:, None] * $(512)) + r211_lane[None, :]) % $(512)) + $(55296)), dtype=tl.bfloat16)
      r211_right = tl.load($(candidate_vto_copy_18_ptr) + ((((((r211_lane[:, None] * $(1024)) + r211_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r211_lane[:, None] * $(1024)) + r211_col[None, :]) % $(128)) * $(512)) + ((((r211_lane[:, None] * $(1024)) + r211_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r211_value = tl.dot(r211_left, r211_right)
      r211_value = r211_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_67_ptr) + ((r211_group * $(98304)) + (r211_row[:, None] * $(1024)) + r211_col[None, :]), r211_value)
    }
    r212_z = tl.arange($(49152))
    r212_value = tl.where((r212_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r212_z / $(1536)) % $(32)) * $(1024)) + (((r212_z / $(192)) % $(8)) * $(128)) + ((r212_z % $(192)) - $(0)) + $(163840)) / $(1024)) % $(288)) * $(1536)) + (((((((r212_z / $(1536)) % $(32)) * $(1024)) + (((r212_z / $(192)) % $(8)) * $(128)) + ((r212_z % $(192)) - $(0)) + $(163840)) / $(128)) % $(8)) * $(192)) + (((((r212_z / $(1536)) % $(32)) * $(1024)) + (((r212_z / $(192)) % $(8)) * $(128)) + ((r212_z % $(192)) - $(0)) + $(163840)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_5_ptr) + ((((((((r212_z / $(1536)) % $(32)) * $(512)) + (((r212_z / $(192)) % $(8)) * $(64)) + ((r212_z % $(192)) - $(128)) + $(81920)) / $(64)) % $(8)) * $(18432)) + (((((((r212_z / $(1536)) % $(32)) * $(512)) + (((r212_z / $(192)) % $(8)) * $(64)) + ((r212_z % $(192)) - $(128)) + $(81920)) / $(512)) % $(288)) * $(64)) + (((((r212_z / $(1536)) % $(32)) * $(512)) + (((r212_z / $(192)) % $(8)) * $(64)) + ((r212_z % $(192)) - $(128)) + $(81920)) % $(64))), dtype=tl.bfloat16))
    r212_value = r212_value.to(tl.bfloat16)
    r212_value = tl.broadcast_to(r212_value, [$(49152)])
    tl.store($(candidate_vcat_46_ptr) + r212_z, r212_value)
    r213_z = tl.arange($(147456))
    r213_value = tl.where((r213_z % $(192)) < $(128), tl.load($(candidate_vbmm_66_ptr) + ((((((((r213_z / $(1536)) % $(96)) * $(1024)) + (((r213_z / $(192)) % $(8)) * $(128)) + ((r213_z % $(192)) - $(0))) / $(1024)) % $(96)) * $(1024)) + (((((((r213_z / $(1536)) % $(96)) * $(1024)) + (((r213_z / $(192)) % $(8)) * $(128)) + ((r213_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r213_z / $(1536)) % $(96)) * $(1024)) + (((r213_z / $(192)) % $(8)) * $(128)) + ((r213_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_7_ptr) + ((((r213_z / $(1536)) % $(96)) * $(576)) + ((((r213_z % $(192)) - $(128)) + $(512)) % $(576)) + $(55296)), dtype=tl.bfloat16))
    r213_value = r213_value.to(tl.bfloat16)
    r213_value = tl.broadcast_to(r213_value, [$(147456)])
    tl.store($(candidate_vcat_47_ptr) + r213_z, r213_value)
    r214_z = tl.arange($(49152))
    r214_value = tl.load($(candidate_vcat_46_ptr) + ((((r214_z / $(192)) % $(32)) * $(1536)) + (((r214_z / $(6144)) % $(8)) * $(192)) + (r214_z % $(192))), dtype=tl.bfloat16)
    r214_value = r214_value.to(tl.float32)
    r214_value = tl.broadcast_to(r214_value, [$(49152)])
    tl.store($(candidate_vto_copy_101_ptr) + r214_z, r214_value)
    r215_z = tl.arange($(147456))
    r215_value = tl.load($(candidate_vcat_47_ptr) + ((((r215_z / $(192)) % $(96)) * $(1536)) + (((r215_z / $(18432)) % $(8)) * $(192)) + (r215_z % $(192))), dtype=tl.bfloat16)
    r215_value = r215_value.to(tl.float32)
    r215_value = tl.broadcast_to(r215_value, [$(147456)])
    tl.store($(candidate_vto_copy_102_ptr) + r215_z, r215_value)
    r216_row = tl.arange($(32))
    r216_col = tl.arange($(96))
    r216_lane = tl.arange($(192))
    for r216_group in range($(8)) {
      r216_left = tl.load($(candidate_vto_copy_101_ptr) + (((((((r216_group % $(8)) * $(6144)) + (r216_row[:, None] * $(192)) + r216_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r216_group % $(8)) * $(6144)) + (r216_row[:, None] * $(192)) + r216_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r216_group % $(8)) * $(6144)) + (r216_row[:, None] * $(192)) + r216_lane[None, :]) % $(192))), dtype=tl.float32)
      r216_right = tl.load($(candidate_vto_copy_102_ptr) + (((((((r216_group % $(8)) * $(18432)) + (r216_lane[:, None] * $(96)) + r216_col[None, :]) / $(18432)) % $(8)) * $(18432)) + (((((r216_group % $(8)) * $(18432)) + (r216_lane[:, None] * $(96)) + r216_col[None, :]) % $(96)) * $(192)) + (((((r216_group % $(8)) * $(18432)) + (r216_lane[:, None] * $(96)) + r216_col[None, :]) / $(96)) % $(192))), dtype=tl.float32)
      r216_value = tl.dot(r216_left, r216_right)
      r216_value = r216_value.to(tl.float32)
      tl.store($(candidate_vbmm_68_ptr) + ((r216_group * $(3072)) + (r216_row[:, None] * $(96)) + r216_col[None, :]), r216_value)
    }
    r217_z = tl.arange($(24576))
    r217_value = (tl.load($(candidate_vbmm_68_ptr) + ((((r217_z / $(3072)) % $(8)) * $(3072)) + (((r217_z / $(96)) % $(32)) * $(96)) + (r217_z % $(96))), dtype=tl.float32) * 0.07216878364870322)
    r217_value = r217_value.to(tl.float32)
    r217_value = tl.broadcast_to(r217_value, [$(24576)])
    tl.store($(candidate_vmul_48_ptr) + r217_z, r217_value)
    r218_z = tl.arange($(32))
    r218_value = (tl.cast(r218_z, tl.int64) + tl.full([], $(96), dtype=tl.int64))
    r218_value = tl.broadcast_to(r218_value, [$(32)])
    tl.store($(candidate_vadd_35_ptr) + r218_z, r218_value)
    r219_z = tl.arange($(32))
    r219_value = (tl.load($(candidate_vadd_35_ptr) + r219_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r219_value = tl.broadcast_to(r219_value, [$(32)])
    tl.store($(candidate_vsub_36_ptr) + r219_z, r219_value)
    r220_z = tl.arange($(3072))
    r220_value = (tl.cast((r220_z % $(96)), tl.int64) > tl.load($(candidate_vsub_36_ptr) + ((r220_z / $(96)) % $(32)), dtype=tl.int64))
    r220_value = tl.broadcast_to(r220_value, [$(3072)])
    tl.store($(candidate_vgt_15_ptr) + r220_z, r220_value)
    r221_z = tl.arange($(24576))
    r221_value = tl.where(tl.load($(candidate_vgt_15_ptr) + ((((r221_z / $(96)) % $(32)) * $(96)) + (r221_z % $(96))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_48_ptr) + ((((r221_z / $(3072)) % $(8)) * $(3072)) + (((r221_z / $(96)) % $(32)) * $(96)) + (r221_z % $(96))), dtype=tl.float32))
    r221_value = r221_value.to(tl.float32)
    r221_value = tl.broadcast_to(r221_value, [$(24576)])
    tl.store($(candidate_vmasked_fill_15_ptr) + r221_z, r221_value)
    r222_z = tl.arange($(24576))
    r222_lane = tl.arange($(96))
    r222_row_values = tl.load($(candidate_vmasked_fill_15_ptr) + ((((r222_z[:, None] / $(3072)) % $(8)) * $(3072)) + (((r222_z[:, None] / $(96)) % $(32)) * $(96)) + (r222_lane[None, :] % $(96))), dtype=tl.float32)
    r222_maximum = tl.max(r222_row_values, axis=1, keep_dims=true)
    r222_weights = tl.exp(r222_row_values - r222_maximum)
    r222_total = tl.sum(r222_weights, axis=1)
    r222_maximum = tl.max(r222_row_values, axis=1)
    r222_value = tl.exp(tl.load($(candidate_vmasked_fill_15_ptr) + ((((r222_z / $(3072)) % $(8)) * $(3072)) + (((r222_z / $(96)) % $(32)) * $(96)) + (r222_z % $(96))), dtype=tl.float32) - r222_maximum) / r222_total
    r222_value = r222_value.to(tl.float32)
    r222_value = tl.broadcast_to(r222_value, [$(24576)])
    tl.store($(candidate_vsoftmax_12_ptr) + r222_z, r222_value)
    r223_z = tl.arange($(24576))
    r223_value = tl.load($(candidate_vsoftmax_12_ptr) + ((((r223_z / $(3072)) % $(8)) * $(3072)) + (((r223_z / $(96)) % $(32)) * $(96)) + (r223_z % $(96))), dtype=tl.float32)
    r223_value = r223_value.to(tl.bfloat16)
    r223_value = tl.broadcast_to(r223_value, [$(24576)])
    tl.store($(candidate_vto_copy_103_ptr) + r223_z, r223_value)
    r224_row = tl.arange($(32))
    r224_col = tl.arange($(128))
    r224_lane = tl.arange($(96))
    for r224_group in range($(8)) {
      r224_left = tl.load($(candidate_vto_copy_103_ptr) + (((((((r224_group % $(8)) * $(3072)) + (r224_row[:, None] * $(96)) + r224_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r224_group % $(8)) * $(3072)) + (r224_row[:, None] * $(96)) + r224_lane[None, :]) / $(96)) % $(32)) * $(96)) + ((((r224_group % $(8)) * $(3072)) + (r224_row[:, None] * $(96)) + r224_lane[None, :]) % $(96))), dtype=tl.bfloat16)
      r224_right = tl.load($(candidate_vbmm_67_ptr) + (((((((r224_group % $(8)) * $(12288)) + (r224_lane[:, None] * $(128)) + r224_col[None, :]) / $(128)) % $(96)) * $(1024)) + ((((((r224_group % $(8)) * $(12288)) + (r224_lane[:, None] * $(128)) + r224_col[None, :]) / $(12288)) % $(8)) * $(128)) + ((((r224_group % $(8)) * $(12288)) + (r224_lane[:, None] * $(128)) + r224_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r224_value = tl.dot(r224_left, r224_right)
      r224_value = r224_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_69_ptr) + ((r224_group * $(4096)) + (r224_row[:, None] * $(128)) + r224_col[None, :]), r224_value)
    }
    r225_z = tl.arange($(256))
    r225_lane = tl.arange($(96))
    r225_row_values = tl.load($(candidate_vmasked_fill_15_ptr) + ((((r225_z[:, None] / $(32)) % $(8)) * $(3072)) + ((r225_z[:, None] % $(32)) * $(96)) + (r225_lane[None, :] % $(96))), dtype=tl.float32)
    r225_maximum = tl.max(r225_row_values, axis=1, keep_dims=true)
    r225_weights = tl.exp(r225_row_values - r225_maximum)
    r225_total = tl.sum(r225_weights, axis=1)
    r225_maximum = tl.max(r225_row_values, axis=1)
    r225_value = r225_maximum + tl.log(r225_total)
    r225_value = r225_value.to(tl.float32)
    r225_value = tl.broadcast_to(r225_value, [$(256)])
    tl.store($(candidate_vlogsumexp_12_ptr) + r225_z, r225_value)
    r226_z = tl.arange($(32768))
    r226_value = tl.load($(candidate_vbmm_69_ptr) + ((((((((r226_z / $(128)) % $(8)) * $(4096)) + ((((r226_z / $(1024)) % $(32)) - $(0)) * $(128)) + (r226_z % $(128))) / $(4096)) % $(8)) * $(4096)) + ((((((r226_z / $(128)) % $(8)) * $(32)) + (((r226_z / $(1024)) % $(32)) - $(0))) % $(32)) * $(128)) + (r226_z % $(128))), dtype=tl.bfloat16)
    r226_value = r226_value.to(tl.bfloat16)
    r226_value = tl.broadcast_to(r226_value, [$(32768)])
    tl.store($(candidate_vcat_48_ptr) + r226_z, r226_value)
    r227_z = tl.arange($(256))
    r227_value = tl.load($(candidate_vlogsumexp_12_ptr) + (((r227_z % $(8)) * $(32)) + ((((r227_z / $(8)) % $(32)) - $(0)) % $(32))), dtype=tl.float32)
    r227_value = r227_value.to(tl.float32)
    r227_value = tl.broadcast_to(r227_value, [$(256)])
    tl.store($(candidate_vcat_49_ptr) + r227_z, r227_value)
    r228_z = tl.arange($(49152))
    r228_value = tl.load($(candidate_vcat_46_ptr) + ((((r228_z / $(192)) % $(32)) * $(1536)) + (((r228_z / $(6144)) % $(8)) * $(192)) + (r228_z % $(192))), dtype=tl.bfloat16)
    r228_value = r228_value.to(tl.float32)
    r228_value = tl.broadcast_to(r228_value, [$(49152)])
    tl.store($(candidate_vto_copy_104_ptr) + r228_z, r228_value)
    r229_z = tl.arange($(147456))
    r229_value = tl.load($(candidate_vcat_47_ptr) + ((((r229_z / $(192)) % $(96)) * $(1536)) + (((r229_z / $(18432)) % $(8)) * $(192)) + (r229_z % $(192))), dtype=tl.bfloat16)
    r229_value = r229_value.to(tl.float32)
    r229_value = tl.broadcast_to(r229_value, [$(147456)])
    tl.store($(candidate_vto_copy_105_ptr) + r229_z, r229_value)
    r230_z = tl.arange($(98304))
    r230_value = tl.load($(candidate_vbmm_67_ptr) + ((((r230_z / $(128)) % $(96)) * $(1024)) + (((r230_z / $(12288)) % $(8)) * $(128)) + (r230_z % $(128))), dtype=tl.bfloat16)
    r230_value = r230_value.to(tl.float32)
    r230_value = tl.broadcast_to(r230_value, [$(98304)])
    tl.store($(candidate_vto_copy_106_ptr) + r230_z, r230_value)
    r231_z = tl.arange($(32768))
    r231_value = tl.load($(candidate_vmm_22_ptr) + ((((r231_z / $(128)) % $(32)) * $(1024)) + (((r231_z / $(4096)) % $(8)) * $(128)) + (r231_z % $(128)) + $(163840)), dtype=tl.bfloat16)
    r231_value = r231_value.to(tl.float32)
    r231_value = tl.broadcast_to(r231_value, [$(32768)])
    tl.store($(candidate_vto_copy_107_ptr) + r231_z, r231_value)
    r232_z = tl.arange($(32768))
    r232_value = tl.load($(candidate_vcat_48_ptr) + ((((r232_z / $(128)) % $(32)) * $(1024)) + (((r232_z / $(4096)) % $(8)) * $(128)) + (r232_z % $(128))), dtype=tl.bfloat16)
    r232_value = r232_value.to(tl.float32)
    r232_value = tl.broadcast_to(r232_value, [$(32768)])
    tl.store($(candidate_vto_copy_108_ptr) + r232_z, r232_value)
    r233_row = tl.arange($(32))
    r233_col = tl.arange($(96))
    r233_lane = tl.arange($(192))
    for r233_group in range($(8)) {
      r233_left = tl.load($(candidate_vto_copy_104_ptr) + (((((((r233_group % $(8)) * $(6144)) + (r233_row[:, None] * $(192)) + r233_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r233_group % $(8)) * $(6144)) + (r233_row[:, None] * $(192)) + r233_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r233_group % $(8)) * $(6144)) + (r233_row[:, None] * $(192)) + r233_lane[None, :]) % $(192))), dtype=tl.float32)
      r233_right = tl.load($(candidate_vto_copy_105_ptr) + (((((((r233_group % $(8)) * $(18432)) + (r233_lane[:, None] * $(96)) + r233_col[None, :]) / $(18432)) % $(8)) * $(18432)) + (((((r233_group % $(8)) * $(18432)) + (r233_lane[:, None] * $(96)) + r233_col[None, :]) % $(96)) * $(192)) + (((((r233_group % $(8)) * $(18432)) + (r233_lane[:, None] * $(96)) + r233_col[None, :]) / $(96)) % $(192))), dtype=tl.float32)
      r233_value = tl.dot(r233_left, r233_right)
      r233_value = r233_value.to(tl.float32)
      tl.store($(candidate_vbmm_70_ptr) + ((r233_group * $(3072)) + (r233_row[:, None] * $(96)) + r233_col[None, :]), r233_value)
    }
    r234_z = tl.arange($(24576))
    r234_value = (tl.load($(candidate_vbmm_70_ptr) + ((((r234_z / $(3072)) % $(8)) * $(3072)) + (((r234_z / $(96)) % $(32)) * $(96)) + (r234_z % $(96))), dtype=tl.float32) * 0.07216878364870322)
    r234_value = r234_value.to(tl.float32)
    r234_value = tl.broadcast_to(r234_value, [$(24576)])
    tl.store($(candidate_vmul_49_ptr) + r234_z, r234_value)
    r235_z = tl.arange($(32))
    r235_value = (tl.cast(r235_z, tl.int64) + tl.full([], $(96), dtype=tl.int64))
    r235_value = tl.broadcast_to(r235_value, [$(32)])
    tl.store($(candidate_vadd_36_ptr) + r235_z, r235_value)
    r236_z = tl.arange($(32))
    r236_value = (tl.load($(candidate_vadd_36_ptr) + r236_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r236_value = tl.broadcast_to(r236_value, [$(32)])
    tl.store($(candidate_vsub_37_ptr) + r236_z, r236_value)
    r237_z = tl.arange($(3072))
    r237_value = (tl.cast((r237_z % $(96)), tl.int64) > tl.load($(candidate_vsub_37_ptr) + ((r237_z / $(96)) % $(32)), dtype=tl.int64))
    r237_value = tl.broadcast_to(r237_value, [$(3072)])
    tl.store($(candidate_vgt_16_ptr) + r237_z, r237_value)
    r238_z = tl.arange($(24576))
    r238_value = tl.where(tl.load($(candidate_vgt_16_ptr) + ((((r238_z / $(96)) % $(32)) * $(96)) + (r238_z % $(96))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_49_ptr) + ((((r238_z / $(3072)) % $(8)) * $(3072)) + (((r238_z / $(96)) % $(32)) * $(96)) + (r238_z % $(96))), dtype=tl.float32))
    r238_value = r238_value.to(tl.float32)
    r238_value = tl.broadcast_to(r238_value, [$(24576)])
    tl.store($(candidate_vmasked_fill_16_ptr) + r238_z, r238_value)
    r239_z = tl.arange($(24576))
    r239_value = (tl.load($(candidate_vmasked_fill_16_ptr) + ((((r239_z / $(3072)) % $(8)) * $(3072)) + (((r239_z / $(96)) % $(32)) * $(96)) + (r239_z % $(96))), dtype=tl.float32) - tl.load($(candidate_vcat_49_ptr) + ((((r239_z / $(96)) % $(32)) * $(8)) + ((r239_z / $(3072)) % $(8))), dtype=tl.float32))
    r239_value = r239_value.to(tl.float32)
    r239_value = tl.broadcast_to(r239_value, [$(24576)])
    tl.store($(candidate_vsub_38_ptr) + r239_z, r239_value)
    r240_z = tl.arange($(24576))
    r240_value = tl.exp(tl.load($(candidate_vsub_38_ptr) + ((((r240_z / $(3072)) % $(8)) * $(3072)) + (((r240_z / $(96)) % $(32)) * $(96)) + (r240_z % $(96))), dtype=tl.float32))
    r240_value = r240_value.to(tl.float32)
    r240_value = tl.broadcast_to(r240_value, [$(24576)])
    tl.store($(candidate_vexp_3_ptr) + r240_z, r240_value)
    r241_z = tl.arange($(32768))
    r241_value = (tl.load($(candidate_vto_copy_107_ptr) + ((((r241_z / $(4096)) % $(8)) * $(4096)) + (((r241_z / $(128)) % $(32)) * $(128)) + (r241_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_108_ptr) + ((((r241_z / $(4096)) % $(8)) * $(4096)) + (((r241_z / $(128)) % $(32)) * $(128)) + (r241_z % $(128))), dtype=tl.float32))
    r241_value = r241_value.to(tl.float32)
    r241_value = tl.broadcast_to(r241_value, [$(32768)])
    tl.store($(candidate_vmul_50_ptr) + r241_z, r241_value)
    r242_z = tl.arange($(256))
    r242_lane = tl.arange($(128))
    r242_value = tl.sum(tl.load($(candidate_vmul_50_ptr) + ((((r242_z[:, None] / $(32)) % $(8)) * $(4096)) + ((r242_z[:, None] % $(32)) * $(128)) + r242_lane[None, :]), dtype=tl.float32), axis=1)
    r242_value = r242_value.to(tl.float32)
    r242_value = tl.broadcast_to(r242_value, [$(256)])
    tl.store($(candidate_vsum_7_ptr) + r242_z, r242_value)
    r243_row = tl.arange($(32))
    r243_col = tl.arange($(96))
    r243_lane = tl.arange($(128))
    for r243_group in range($(8)) {
      r243_left = tl.load($(candidate_vto_copy_107_ptr) + (((((((r243_group % $(8)) * $(4096)) + (r243_row[:, None] * $(128)) + r243_lane[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r243_group % $(8)) * $(4096)) + (r243_row[:, None] * $(128)) + r243_lane[None, :]) / $(128)) % $(32)) * $(128)) + ((((r243_group % $(8)) * $(4096)) + (r243_row[:, None] * $(128)) + r243_lane[None, :]) % $(128))), dtype=tl.float32)
      r243_right = tl.load($(candidate_vto_copy_106_ptr) + (((((((r243_group % $(8)) * $(12288)) + (r243_lane[:, None] * $(96)) + r243_col[None, :]) / $(12288)) % $(8)) * $(12288)) + (((((r243_group % $(8)) * $(12288)) + (r243_lane[:, None] * $(96)) + r243_col[None, :]) % $(96)) * $(128)) + (((((r243_group % $(8)) * $(12288)) + (r243_lane[:, None] * $(96)) + r243_col[None, :]) / $(96)) % $(128))), dtype=tl.float32)
      r243_value = tl.dot(r243_left, r243_right)
      r243_value = r243_value.to(tl.float32)
      tl.store($(candidate_vbmm_71_ptr) + ((r243_group * $(3072)) + (r243_row[:, None] * $(96)) + r243_col[None, :]), r243_value)
    }
    r244_z = tl.arange($(24576))
    r244_value = (tl.load($(candidate_vbmm_71_ptr) + ((((r244_z / $(3072)) % $(8)) * $(3072)) + (((r244_z / $(96)) % $(32)) * $(96)) + (r244_z % $(96))), dtype=tl.float32) - tl.load($(candidate_vsum_7_ptr) + ((((r244_z / $(3072)) % $(8)) * $(32)) + ((r244_z / $(96)) % $(32))), dtype=tl.float32))
    r244_value = r244_value.to(tl.float32)
    r244_value = tl.broadcast_to(r244_value, [$(24576)])
    tl.store($(candidate_vsub_39_ptr) + r244_z, r244_value)
    r245_z = tl.arange($(24576))
    r245_value = (tl.load($(candidate_vexp_3_ptr) + ((((r245_z / $(3072)) % $(8)) * $(3072)) + (((r245_z / $(96)) % $(32)) * $(96)) + (r245_z % $(96))), dtype=tl.float32) * tl.load($(candidate_vsub_39_ptr) + ((((r245_z / $(3072)) % $(8)) * $(3072)) + (((r245_z / $(96)) % $(32)) * $(96)) + (r245_z % $(96))), dtype=tl.float32))
    r245_value = r245_value.to(tl.float32)
    r245_value = tl.broadcast_to(r245_value, [$(24576)])
    tl.store($(candidate_vmul_51_ptr) + r245_z, r245_value)
    r246_row = tl.arange($(96))
    r246_col = tl.arange($(192))
    r246_lane = tl.arange($(32))
    for r246_group in range($(8)) {
      r246_left = tl.load($(candidate_vmul_51_ptr) + (((((((r246_group % $(8)) * $(3072)) + (r246_row[:, None] * $(32)) + r246_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + (((((r246_group % $(8)) * $(3072)) + (r246_row[:, None] * $(32)) + r246_lane[None, :]) % $(32)) * $(96)) + (((((r246_group % $(8)) * $(3072)) + (r246_row[:, None] * $(32)) + r246_lane[None, :]) / $(32)) % $(96))), dtype=tl.float32)
      r246_right = tl.load($(candidate_vto_copy_104_ptr) + (((((((r246_group % $(8)) * $(6144)) + (r246_lane[:, None] * $(192)) + r246_col[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r246_group % $(8)) * $(6144)) + (r246_lane[:, None] * $(192)) + r246_col[None, :]) / $(192)) % $(32)) * $(192)) + ((((r246_group % $(8)) * $(6144)) + (r246_lane[:, None] * $(192)) + r246_col[None, :]) % $(192))), dtype=tl.float32)
      r246_value = tl.dot(r246_left, r246_right)
      r246_value = r246_value.to(tl.float32)
      tl.store($(candidate_vbmm_73_ptr) + ((r246_group * $(18432)) + (r246_row[:, None] * $(192)) + r246_col[None, :]), r246_value)
    }
    r247_z = tl.arange($(147456))
    r247_value = (tl.load($(candidate_vbmm_73_ptr) + ((((r247_z / $(18432)) % $(8)) * $(18432)) + (((r247_z / $(192)) % $(96)) * $(192)) + (r247_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r247_value = r247_value.to(tl.float32)
    r247_value = tl.broadcast_to(r247_value, [$(147456)])
    tl.store($(candidate_vmul_53_ptr) + r247_z, r247_value)
    r248_z = tl.arange($(147456))
    r248_value = tl.load($(candidate_vmul_53_ptr) + ((((r248_z / $(192)) % $(8)) * $(18432)) + (((r248_z / $(1536)) % $(96)) * $(192)) + (r248_z % $(192))), dtype=tl.float32)
    r248_value = r248_value.to(tl.bfloat16)
    r248_value = tl.broadcast_to(r248_value, [$(147456)])
    tl.store($(candidate_vto_copy_110_ptr) + r248_z, r248_value)
    r249_row = tl.arange($(96))
    r249_col = tl.arange($(128))
    r249_lane = tl.arange($(32))
    for r249_group in range($(8)) {
      r249_left = tl.load($(candidate_vexp_3_ptr) + (((((((r249_group % $(8)) * $(3072)) + (r249_row[:, None] * $(32)) + r249_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + (((((r249_group % $(8)) * $(3072)) + (r249_row[:, None] * $(32)) + r249_lane[None, :]) % $(32)) * $(96)) + (((((r249_group % $(8)) * $(3072)) + (r249_row[:, None] * $(32)) + r249_lane[None, :]) / $(32)) % $(96))), dtype=tl.float32)
      r249_right = tl.load($(candidate_vto_copy_107_ptr) + (((((((r249_group % $(8)) * $(4096)) + (r249_lane[:, None] * $(128)) + r249_col[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r249_group % $(8)) * $(4096)) + (r249_lane[:, None] * $(128)) + r249_col[None, :]) / $(128)) % $(32)) * $(128)) + ((((r249_group % $(8)) * $(4096)) + (r249_lane[:, None] * $(128)) + r249_col[None, :]) % $(128))), dtype=tl.float32)
      r249_value = tl.dot(r249_left, r249_right)
      r249_value = r249_value.to(tl.float32)
      tl.store($(candidate_vbmm_74_ptr) + ((r249_group * $(12288)) + (r249_row[:, None] * $(128)) + r249_col[None, :]), r249_value)
    }
    r250_z = tl.arange($(98304))
    r250_value = tl.load($(candidate_vbmm_74_ptr) + ((((r250_z / $(128)) % $(8)) * $(12288)) + (((r250_z / $(1024)) % $(96)) * $(128)) + (r250_z % $(128))), dtype=tl.float32)
    r250_value = r250_value.to(tl.bfloat16)
    r250_value = tl.broadcast_to(r250_value, [$(98304)])
    tl.store($(candidate_vto_copy_111_ptr) + r250_z, r250_value)
    r251_z = tl.arange($(147456))
    r251_value = tl.load($(candidate_vto_copy_110_ptr) + ((((((r251_z / $(1536)) % $(96)) - $(0)) % $(96)) * $(1536)) + (((r251_z / $(192)) % $(8)) * $(192)) + (r251_z % $(192))), dtype=tl.bfloat16)
    r251_value = r251_value.to(tl.bfloat16)
    r251_value = tl.broadcast_to(r251_value, [$(147456)])
    tl.store($(candidate_vcat_51_ptr) + r251_z, r251_value)
    r252_z = tl.arange($(98304))
    r252_value = tl.load($(candidate_vto_copy_111_ptr) + ((((((r252_z / $(1024)) % $(96)) - $(0)) % $(96)) * $(1024)) + (((r252_z / $(128)) % $(8)) * $(128)) + (r252_z % $(128))), dtype=tl.bfloat16)
    r252_value = r252_value.to(tl.bfloat16)
    r252_value = tl.broadcast_to(r252_value, [$(98304)])
    tl.store($(candidate_vcat_52_ptr) + r252_z, r252_value)
    r253_row = tl.arange($(1024))
    r253_col = tl.arange($(512))
    r253_lane = tl.arange($(96))
    for r253_group in range($(1)) {
      r253_left = tl.load($(candidate_vcat_51_ptr) + (((((r253_row[:, None] * $(96)) + r253_lane[None, :]) % $(96)) * $(1536)) + (((((r253_row[:, None] * $(96)) + r253_lane[None, :]) / $(12288)) % $(8)) * $(192)) + ((((r253_row[:, None] * $(96)) + r253_lane[None, :]) / $(96)) % $(128))), dtype=tl.bfloat16)
      r253_right = tl.load($(candidate_vcat_7_ptr) + ((((((r253_lane[:, None] * $(512)) + r253_col[None, :]) / $(512)) % $(96)) * $(576)) + (((r253_lane[:, None] * $(512)) + r253_col[None, :]) % $(512)) + $(55296)), dtype=tl.bfloat16)
      r253_value = tl.dot(r253_left, r253_right)
      r253_value = r253_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_77_ptr) + ((r253_group * $(524288)) + (r253_row[:, None] * $(512)) + r253_col[None, :]), r253_value)
    }
    r254_row = tl.arange($(1024))
    r254_col = tl.arange($(512))
    r254_lane = tl.arange($(96))
    for r254_group in range($(1)) {
      r254_left = tl.load($(candidate_vcat_52_ptr) + (((((r254_row[:, None] * $(96)) + r254_lane[None, :]) % $(96)) * $(1024)) + (((((r254_row[:, None] * $(96)) + r254_lane[None, :]) / $(12288)) % $(8)) * $(128)) + ((((r254_row[:, None] * $(96)) + r254_lane[None, :]) / $(96)) % $(128))), dtype=tl.bfloat16)
      r254_right = tl.load($(candidate_vcat_7_ptr) + ((((((r254_lane[:, None] * $(512)) + r254_col[None, :]) / $(512)) % $(96)) * $(576)) + (((r254_lane[:, None] * $(512)) + r254_col[None, :]) % $(512)) + $(55296)), dtype=tl.bfloat16)
      r254_value = tl.dot(r254_left, r254_right)
      r254_value = r254_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_78_ptr) + ((r254_group * $(524288)) + (r254_row[:, None] * $(512)) + r254_col[None, :]), r254_value)
    }
    r255_z = tl.arange($(1048576))
    r255_value = tl.where(((r255_z / $(512)) % $(256)) < $(128), tl.load($(candidate_vbmm_77_ptr) + ((((((((r255_z / $(131072)) % $(8)) * $(65536)) + ((((r255_z / $(512)) % $(256)) - $(0)) * $(512)) + (r255_z % $(512))) / $(65536)) % $(8)) * $(65536)) + ((((((r255_z / $(131072)) % $(8)) * $(128)) + (((r255_z / $(512)) % $(256)) - $(0))) % $(128)) * $(512)) + (r255_z % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vbmm_78_ptr) + ((((((((r255_z / $(131072)) % $(8)) * $(65536)) + ((((r255_z / $(512)) % $(256)) - $(128)) * $(512)) + (r255_z % $(512))) / $(65536)) % $(8)) * $(65536)) + ((((((r255_z / $(131072)) % $(8)) * $(128)) + (((r255_z / $(512)) % $(256)) - $(128))) % $(128)) * $(512)) + (r255_z % $(512))), dtype=tl.bfloat16))
    r255_value = r255_value.to(tl.bfloat16)
    r255_value = tl.broadcast_to(r255_value, [$(1048576)])
    tl.store($(candidate_vcat_53_ptr) + r255_z, r255_value)
    r256_z = tl.arange($(1048576))
    r256_value = (tl.load($(candidate_vadd_34_ptr) + ((((r256_z / $(512)) % $(2048)) * $(512)) + (r256_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vcat_53_ptr) + ((((((((r256_z / $(512)) % $(2048)) * $(512)) + (r256_z % $(512))) / $(131072)) % $(8)) * $(131072)) + ((((r256_z / $(512)) % $(2048)) % $(256)) * $(512)) + (r256_z % $(512))), dtype=tl.bfloat16))
    r256_value = r256_value.to(tl.bfloat16)
    r256_value = tl.broadcast_to(r256_value, [$(1048576)])
    tl.store($(candidate_vadd_38_ptr) + r256_z, r256_value)
    r257_row = tl.arange($(96))
    r257_col = tl.arange($(1024))
    r257_lane = tl.arange($(512))
    for r257_group in range($(1)) {
      r257_left = tl.load($(candidate_vcat_7_ptr) + ((((((r257_row[:, None] * $(512)) + r257_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r257_row[:, None] * $(512)) + r257_lane[None, :]) % $(512)) + $(55296)), dtype=tl.bfloat16)
      r257_right = tl.load($(candidate_vto_copy_18_ptr) + ((((((r257_lane[:, None] * $(1024)) + r257_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r257_lane[:, None] * $(1024)) + r257_col[None, :]) % $(128)) * $(512)) + ((((r257_lane[:, None] * $(1024)) + r257_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r257_value = tl.dot(r257_left, r257_right)
      r257_value = r257_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_79_ptr) + ((r257_group * $(98304)) + (r257_row[:, None] * $(1024)) + r257_col[None, :]), r257_value)
    }
    r258_row = tl.arange($(96))
    r258_col = tl.arange($(1024))
    r258_lane = tl.arange($(512))
    for r258_group in range($(1)) {
      r258_left = tl.load($(candidate_vcat_7_ptr) + ((((((r258_row[:, None] * $(512)) + r258_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r258_row[:, None] * $(512)) + r258_lane[None, :]) % $(512)) + $(55296)), dtype=tl.bfloat16)
      r258_right = tl.load($(candidate_vto_copy_18_ptr) + ((((((r258_lane[:, None] * $(1024)) + r258_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r258_lane[:, None] * $(1024)) + r258_col[None, :]) % $(128)) * $(512)) + ((((r258_lane[:, None] * $(1024)) + r258_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r258_value = tl.dot(r258_left, r258_right)
      r258_value = r258_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_80_ptr) + ((r258_group * $(98304)) + (r258_row[:, None] * $(1024)) + r258_col[None, :]), r258_value)
    }
    r259_z = tl.arange($(49152))
    r259_value = tl.where((r259_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r259_z / $(1536)) % $(32)) * $(1024)) + (((r259_z / $(192)) % $(8)) * $(128)) + ((r259_z % $(192)) - $(0)) + $(131072)) / $(1024)) % $(288)) * $(1536)) + (((((((r259_z / $(1536)) % $(32)) * $(1024)) + (((r259_z / $(192)) % $(8)) * $(128)) + ((r259_z % $(192)) - $(0)) + $(131072)) / $(128)) % $(8)) * $(192)) + (((((r259_z / $(1536)) % $(32)) * $(1024)) + (((r259_z / $(192)) % $(8)) * $(128)) + ((r259_z % $(192)) - $(0)) + $(131072)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_5_ptr) + ((((((((r259_z / $(1536)) % $(32)) * $(512)) + (((r259_z / $(192)) % $(8)) * $(64)) + ((r259_z % $(192)) - $(128)) + $(65536)) / $(64)) % $(8)) * $(18432)) + (((((((r259_z / $(1536)) % $(32)) * $(512)) + (((r259_z / $(192)) % $(8)) * $(64)) + ((r259_z % $(192)) - $(128)) + $(65536)) / $(512)) % $(288)) * $(64)) + (((((r259_z / $(1536)) % $(32)) * $(512)) + (((r259_z / $(192)) % $(8)) * $(64)) + ((r259_z % $(192)) - $(128)) + $(65536)) % $(64))), dtype=tl.bfloat16))
    r259_value = r259_value.to(tl.bfloat16)
    r259_value = tl.broadcast_to(r259_value, [$(49152)])
    tl.store($(candidate_vcat_54_ptr) + r259_z, r259_value)
    r260_z = tl.arange($(147456))
    r260_value = tl.where((r260_z % $(192)) < $(128), tl.load($(candidate_vbmm_79_ptr) + ((((((((r260_z / $(1536)) % $(96)) * $(1024)) + (((r260_z / $(192)) % $(8)) * $(128)) + ((r260_z % $(192)) - $(0))) / $(1024)) % $(96)) * $(1024)) + (((((((r260_z / $(1536)) % $(96)) * $(1024)) + (((r260_z / $(192)) % $(8)) * $(128)) + ((r260_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r260_z / $(1536)) % $(96)) * $(1024)) + (((r260_z / $(192)) % $(8)) * $(128)) + ((r260_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_7_ptr) + ((((r260_z / $(1536)) % $(96)) * $(576)) + ((((r260_z % $(192)) - $(128)) + $(512)) % $(576)) + $(55296)), dtype=tl.bfloat16))
    r260_value = r260_value.to(tl.bfloat16)
    r260_value = tl.broadcast_to(r260_value, [$(147456)])
    tl.store($(candidate_vcat_55_ptr) + r260_z, r260_value)
    r261_z = tl.arange($(49152))
    r261_value = tl.load($(candidate_vcat_54_ptr) + ((((r261_z / $(192)) % $(32)) * $(1536)) + (((r261_z / $(6144)) % $(8)) * $(192)) + (r261_z % $(192))), dtype=tl.bfloat16)
    r261_value = r261_value.to(tl.float32)
    r261_value = tl.broadcast_to(r261_value, [$(49152)])
    tl.store($(candidate_vto_copy_112_ptr) + r261_z, r261_value)
    r262_z = tl.arange($(98304))
    r262_value = tl.load($(candidate_vcat_55_ptr) + ((((r262_z / $(192)) % $(64)) * $(1536)) + (((r262_z / $(12288)) % $(8)) * $(192)) + (r262_z % $(192))), dtype=tl.bfloat16)
    r262_value = r262_value.to(tl.float32)
    r262_value = tl.broadcast_to(r262_value, [$(98304)])
    tl.store($(candidate_vto_copy_113_ptr) + r262_z, r262_value)
    r263_row = tl.arange($(32))
    r263_col = tl.arange($(64))
    r263_lane = tl.arange($(192))
    for r263_group in range($(8)) {
      r263_left = tl.load($(candidate_vto_copy_112_ptr) + (((((((r263_group % $(8)) * $(6144)) + (r263_row[:, None] * $(192)) + r263_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r263_group % $(8)) * $(6144)) + (r263_row[:, None] * $(192)) + r263_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r263_group % $(8)) * $(6144)) + (r263_row[:, None] * $(192)) + r263_lane[None, :]) % $(192))), dtype=tl.float32)
      r263_right = tl.load($(candidate_vto_copy_113_ptr) + (((((((r263_group % $(8)) * $(12288)) + (r263_lane[:, None] * $(64)) + r263_col[None, :]) / $(12288)) % $(8)) * $(12288)) + (((((r263_group % $(8)) * $(12288)) + (r263_lane[:, None] * $(64)) + r263_col[None, :]) % $(64)) * $(192)) + (((((r263_group % $(8)) * $(12288)) + (r263_lane[:, None] * $(64)) + r263_col[None, :]) / $(64)) % $(192))), dtype=tl.float32)
      r263_value = tl.dot(r263_left, r263_right)
      r263_value = r263_value.to(tl.float32)
      tl.store($(candidate_vbmm_81_ptr) + ((r263_group * $(2048)) + (r263_row[:, None] * $(64)) + r263_col[None, :]), r263_value)
    }
    r264_z = tl.arange($(16384))
    r264_value = (tl.load($(candidate_vbmm_81_ptr) + ((((r264_z / $(2048)) % $(8)) * $(2048)) + (((r264_z / $(64)) % $(32)) * $(64)) + (r264_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r264_value = r264_value.to(tl.float32)
    r264_value = tl.broadcast_to(r264_value, [$(16384)])
    tl.store($(candidate_vmul_54_ptr) + r264_z, r264_value)
    r265_z = tl.arange($(32))
    r265_value = (tl.cast(r265_z, tl.int64) + tl.full([], $(64), dtype=tl.int64))
    r265_value = tl.broadcast_to(r265_value, [$(32)])
    tl.store($(candidate_vadd_39_ptr) + r265_z, r265_value)
    r266_z = tl.arange($(32))
    r266_value = (tl.load($(candidate_vadd_39_ptr) + r266_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r266_value = tl.broadcast_to(r266_value, [$(32)])
    tl.store($(candidate_vsub_40_ptr) + r266_z, r266_value)
    r267_z = tl.arange($(2048))
    r267_value = (tl.cast((r267_z % $(64)), tl.int64) > tl.load($(candidate_vsub_40_ptr) + ((r267_z / $(64)) % $(32)), dtype=tl.int64))
    r267_value = tl.broadcast_to(r267_value, [$(2048)])
    tl.store($(candidate_vgt_17_ptr) + r267_z, r267_value)
    r268_z = tl.arange($(16384))
    r268_value = tl.where(tl.load($(candidate_vgt_17_ptr) + ((((r268_z / $(64)) % $(32)) * $(64)) + (r268_z % $(64))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_54_ptr) + ((((r268_z / $(2048)) % $(8)) * $(2048)) + (((r268_z / $(64)) % $(32)) * $(64)) + (r268_z % $(64))), dtype=tl.float32))
    r268_value = r268_value.to(tl.float32)
    r268_value = tl.broadcast_to(r268_value, [$(16384)])
    tl.store($(candidate_vmasked_fill_17_ptr) + r268_z, r268_value)
    r269_z = tl.arange($(16384))
    r269_lane = tl.arange($(64))
    r269_row_values = tl.load($(candidate_vmasked_fill_17_ptr) + ((((r269_z[:, None] / $(2048)) % $(8)) * $(2048)) + (((r269_z[:, None] / $(64)) % $(32)) * $(64)) + (r269_lane[None, :] % $(64))), dtype=tl.float32)
    r269_maximum = tl.max(r269_row_values, axis=1, keep_dims=true)
    r269_weights = tl.exp(r269_row_values - r269_maximum)
    r269_total = tl.sum(r269_weights, axis=1)
    r269_maximum = tl.max(r269_row_values, axis=1)
    r269_value = tl.exp(tl.load($(candidate_vmasked_fill_17_ptr) + ((((r269_z / $(2048)) % $(8)) * $(2048)) + (((r269_z / $(64)) % $(32)) * $(64)) + (r269_z % $(64))), dtype=tl.float32) - r269_maximum) / r269_total
    r269_value = r269_value.to(tl.float32)
    r269_value = tl.broadcast_to(r269_value, [$(16384)])
    tl.store($(candidate_vsoftmax_13_ptr) + r269_z, r269_value)
    r270_z = tl.arange($(16384))
    r270_value = tl.load($(candidate_vsoftmax_13_ptr) + ((((r270_z / $(2048)) % $(8)) * $(2048)) + (((r270_z / $(64)) % $(32)) * $(64)) + (r270_z % $(64))), dtype=tl.float32)
    r270_value = r270_value.to(tl.bfloat16)
    r270_value = tl.broadcast_to(r270_value, [$(16384)])
    tl.store($(candidate_vto_copy_114_ptr) + r270_z, r270_value)
    r271_row = tl.arange($(32))
    r271_col = tl.arange($(128))
    r271_lane = tl.arange($(64))
    for r271_group in range($(8)) {
      r271_left = tl.load($(candidate_vto_copy_114_ptr) + (((((((r271_group % $(8)) * $(2048)) + (r271_row[:, None] * $(64)) + r271_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r271_group % $(8)) * $(2048)) + (r271_row[:, None] * $(64)) + r271_lane[None, :]) / $(64)) % $(32)) * $(64)) + ((((r271_group % $(8)) * $(2048)) + (r271_row[:, None] * $(64)) + r271_lane[None, :]) % $(64))), dtype=tl.bfloat16)
      r271_right = tl.load($(candidate_vbmm_80_ptr) + (((((((r271_group % $(8)) * $(8192)) + (r271_lane[:, None] * $(128)) + r271_col[None, :]) / $(128)) % $(64)) * $(1024)) + ((((((r271_group % $(8)) * $(8192)) + (r271_lane[:, None] * $(128)) + r271_col[None, :]) / $(8192)) % $(8)) * $(128)) + ((((r271_group % $(8)) * $(8192)) + (r271_lane[:, None] * $(128)) + r271_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r271_value = tl.dot(r271_left, r271_right)
      r271_value = r271_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_82_ptr) + ((r271_group * $(4096)) + (r271_row[:, None] * $(128)) + r271_col[None, :]), r271_value)
    }
    r272_z = tl.arange($(256))
    r272_lane = tl.arange($(64))
    r272_row_values = tl.load($(candidate_vmasked_fill_17_ptr) + ((((r272_z[:, None] / $(32)) % $(8)) * $(2048)) + ((r272_z[:, None] % $(32)) * $(64)) + (r272_lane[None, :] % $(64))), dtype=tl.float32)
    r272_maximum = tl.max(r272_row_values, axis=1, keep_dims=true)
    r272_weights = tl.exp(r272_row_values - r272_maximum)
    r272_total = tl.sum(r272_weights, axis=1)
    r272_maximum = tl.max(r272_row_values, axis=1)
    r272_value = r272_maximum + tl.log(r272_total)
    r272_value = r272_value.to(tl.float32)
    r272_value = tl.broadcast_to(r272_value, [$(256)])
    tl.store($(candidate_vlogsumexp_13_ptr) + r272_z, r272_value)
    r273_z = tl.arange($(32768))
    r273_value = tl.load($(candidate_vbmm_82_ptr) + ((((((((r273_z / $(128)) % $(8)) * $(4096)) + ((((r273_z / $(1024)) % $(32)) - $(0)) * $(128)) + (r273_z % $(128))) / $(4096)) % $(8)) * $(4096)) + ((((((r273_z / $(128)) % $(8)) * $(32)) + (((r273_z / $(1024)) % $(32)) - $(0))) % $(32)) * $(128)) + (r273_z % $(128))), dtype=tl.bfloat16)
    r273_value = r273_value.to(tl.bfloat16)
    r273_value = tl.broadcast_to(r273_value, [$(32768)])
    tl.store($(candidate_vcat_56_ptr) + r273_z, r273_value)
    r274_z = tl.arange($(256))
    r274_value = tl.load($(candidate_vlogsumexp_13_ptr) + (((r274_z % $(8)) * $(32)) + ((((r274_z / $(8)) % $(32)) - $(0)) % $(32))), dtype=tl.float32)
    r274_value = r274_value.to(tl.float32)
    r274_value = tl.broadcast_to(r274_value, [$(256)])
    tl.store($(candidate_vcat_57_ptr) + r274_z, r274_value)
    r275_z = tl.arange($(49152))
    r275_value = tl.load($(candidate_vcat_54_ptr) + ((((r275_z / $(192)) % $(32)) * $(1536)) + (((r275_z / $(6144)) % $(8)) * $(192)) + (r275_z % $(192))), dtype=tl.bfloat16)
    r275_value = r275_value.to(tl.float32)
    r275_value = tl.broadcast_to(r275_value, [$(49152)])
    tl.store($(candidate_vto_copy_115_ptr) + r275_z, r275_value)
    r276_z = tl.arange($(98304))
    r276_value = tl.load($(candidate_vcat_55_ptr) + ((((r276_z / $(192)) % $(64)) * $(1536)) + (((r276_z / $(12288)) % $(8)) * $(192)) + (r276_z % $(192))), dtype=tl.bfloat16)
    r276_value = r276_value.to(tl.float32)
    r276_value = tl.broadcast_to(r276_value, [$(98304)])
    tl.store($(candidate_vto_copy_116_ptr) + r276_z, r276_value)
    r277_z = tl.arange($(65536))
    r277_value = tl.load($(candidate_vbmm_80_ptr) + ((((r277_z / $(128)) % $(64)) * $(1024)) + (((r277_z / $(8192)) % $(8)) * $(128)) + (r277_z % $(128))), dtype=tl.bfloat16)
    r277_value = r277_value.to(tl.float32)
    r277_value = tl.broadcast_to(r277_value, [$(65536)])
    tl.store($(candidate_vto_copy_117_ptr) + r277_z, r277_value)
    r278_z = tl.arange($(32768))
    r278_value = tl.load($(candidate_vmm_22_ptr) + ((((r278_z / $(128)) % $(32)) * $(1024)) + (((r278_z / $(4096)) % $(8)) * $(128)) + (r278_z % $(128)) + $(131072)), dtype=tl.bfloat16)
    r278_value = r278_value.to(tl.float32)
    r278_value = tl.broadcast_to(r278_value, [$(32768)])
    tl.store($(candidate_vto_copy_118_ptr) + r278_z, r278_value)
    r279_z = tl.arange($(32768))
    r279_value = tl.load($(candidate_vcat_56_ptr) + ((((r279_z / $(128)) % $(32)) * $(1024)) + (((r279_z / $(4096)) % $(8)) * $(128)) + (r279_z % $(128))), dtype=tl.bfloat16)
    r279_value = r279_value.to(tl.float32)
    r279_value = tl.broadcast_to(r279_value, [$(32768)])
    tl.store($(candidate_vto_copy_119_ptr) + r279_z, r279_value)
    r280_row = tl.arange($(32))
    r280_col = tl.arange($(64))
    r280_lane = tl.arange($(192))
    for r280_group in range($(8)) {
      r280_left = tl.load($(candidate_vto_copy_115_ptr) + (((((((r280_group % $(8)) * $(6144)) + (r280_row[:, None] * $(192)) + r280_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r280_group % $(8)) * $(6144)) + (r280_row[:, None] * $(192)) + r280_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r280_group % $(8)) * $(6144)) + (r280_row[:, None] * $(192)) + r280_lane[None, :]) % $(192))), dtype=tl.float32)
      r280_right = tl.load($(candidate_vto_copy_116_ptr) + (((((((r280_group % $(8)) * $(12288)) + (r280_lane[:, None] * $(64)) + r280_col[None, :]) / $(12288)) % $(8)) * $(12288)) + (((((r280_group % $(8)) * $(12288)) + (r280_lane[:, None] * $(64)) + r280_col[None, :]) % $(64)) * $(192)) + (((((r280_group % $(8)) * $(12288)) + (r280_lane[:, None] * $(64)) + r280_col[None, :]) / $(64)) % $(192))), dtype=tl.float32)
      r280_value = tl.dot(r280_left, r280_right)
      r280_value = r280_value.to(tl.float32)
      tl.store($(candidate_vbmm_83_ptr) + ((r280_group * $(2048)) + (r280_row[:, None] * $(64)) + r280_col[None, :]), r280_value)
    }
    r281_z = tl.arange($(16384))
    r281_value = (tl.load($(candidate_vbmm_83_ptr) + ((((r281_z / $(2048)) % $(8)) * $(2048)) + (((r281_z / $(64)) % $(32)) * $(64)) + (r281_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r281_value = r281_value.to(tl.float32)
    r281_value = tl.broadcast_to(r281_value, [$(16384)])
    tl.store($(candidate_vmul_55_ptr) + r281_z, r281_value)
    r282_z = tl.arange($(32))
    r282_value = (tl.cast(r282_z, tl.int64) + tl.full([], $(64), dtype=tl.int64))
    r282_value = tl.broadcast_to(r282_value, [$(32)])
    tl.store($(candidate_vadd_40_ptr) + r282_z, r282_value)
    r283_z = tl.arange($(32))
    r283_value = (tl.load($(candidate_vadd_40_ptr) + r283_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r283_value = tl.broadcast_to(r283_value, [$(32)])
    tl.store($(candidate_vsub_41_ptr) + r283_z, r283_value)
    r284_z = tl.arange($(2048))
    r284_value = (tl.cast((r284_z % $(64)), tl.int64) > tl.load($(candidate_vsub_41_ptr) + ((r284_z / $(64)) % $(32)), dtype=tl.int64))
    r284_value = tl.broadcast_to(r284_value, [$(2048)])
    tl.store($(candidate_vgt_18_ptr) + r284_z, r284_value)
    r285_z = tl.arange($(16384))
    r285_value = tl.where(tl.load($(candidate_vgt_18_ptr) + ((((r285_z / $(64)) % $(32)) * $(64)) + (r285_z % $(64))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_55_ptr) + ((((r285_z / $(2048)) % $(8)) * $(2048)) + (((r285_z / $(64)) % $(32)) * $(64)) + (r285_z % $(64))), dtype=tl.float32))
    r285_value = r285_value.to(tl.float32)
    r285_value = tl.broadcast_to(r285_value, [$(16384)])
    tl.store($(candidate_vmasked_fill_18_ptr) + r285_z, r285_value)
    r286_z = tl.arange($(16384))
    r286_value = (tl.load($(candidate_vmasked_fill_18_ptr) + ((((r286_z / $(2048)) % $(8)) * $(2048)) + (((r286_z / $(64)) % $(32)) * $(64)) + (r286_z % $(64))), dtype=tl.float32) - tl.load($(candidate_vcat_57_ptr) + ((((r286_z / $(64)) % $(32)) * $(8)) + ((r286_z / $(2048)) % $(8))), dtype=tl.float32))
    r286_value = r286_value.to(tl.float32)
    r286_value = tl.broadcast_to(r286_value, [$(16384)])
    tl.store($(candidate_vsub_42_ptr) + r286_z, r286_value)
    r287_z = tl.arange($(16384))
    r287_value = tl.exp(tl.load($(candidate_vsub_42_ptr) + ((((r287_z / $(2048)) % $(8)) * $(2048)) + (((r287_z / $(64)) % $(32)) * $(64)) + (r287_z % $(64))), dtype=tl.float32))
    r287_value = r287_value.to(tl.float32)
    r287_value = tl.broadcast_to(r287_value, [$(16384)])
    tl.store($(candidate_vexp_4_ptr) + r287_z, r287_value)
    r288_z = tl.arange($(32768))
    r288_value = (tl.load($(candidate_vto_copy_118_ptr) + ((((r288_z / $(4096)) % $(8)) * $(4096)) + (((r288_z / $(128)) % $(32)) * $(128)) + (r288_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_119_ptr) + ((((r288_z / $(4096)) % $(8)) * $(4096)) + (((r288_z / $(128)) % $(32)) * $(128)) + (r288_z % $(128))), dtype=tl.float32))
    r288_value = r288_value.to(tl.float32)
    r288_value = tl.broadcast_to(r288_value, [$(32768)])
    tl.store($(candidate_vmul_56_ptr) + r288_z, r288_value)
    r289_z = tl.arange($(256))
    r289_lane = tl.arange($(128))
    r289_value = tl.sum(tl.load($(candidate_vmul_56_ptr) + ((((r289_z[:, None] / $(32)) % $(8)) * $(4096)) + ((r289_z[:, None] % $(32)) * $(128)) + r289_lane[None, :]), dtype=tl.float32), axis=1)
    r289_value = r289_value.to(tl.float32)
    r289_value = tl.broadcast_to(r289_value, [$(256)])
    tl.store($(candidate_vsum_9_ptr) + r289_z, r289_value)
    r290_row = tl.arange($(32))
    r290_col = tl.arange($(64))
    r290_lane = tl.arange($(128))
    for r290_group in range($(8)) {
      r290_left = tl.load($(candidate_vto_copy_118_ptr) + (((((((r290_group % $(8)) * $(4096)) + (r290_row[:, None] * $(128)) + r290_lane[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r290_group % $(8)) * $(4096)) + (r290_row[:, None] * $(128)) + r290_lane[None, :]) / $(128)) % $(32)) * $(128)) + ((((r290_group % $(8)) * $(4096)) + (r290_row[:, None] * $(128)) + r290_lane[None, :]) % $(128))), dtype=tl.float32)
      r290_right = tl.load($(candidate_vto_copy_117_ptr) + (((((((r290_group % $(8)) * $(8192)) + (r290_lane[:, None] * $(64)) + r290_col[None, :]) / $(8192)) % $(8)) * $(8192)) + (((((r290_group % $(8)) * $(8192)) + (r290_lane[:, None] * $(64)) + r290_col[None, :]) % $(64)) * $(128)) + (((((r290_group % $(8)) * $(8192)) + (r290_lane[:, None] * $(64)) + r290_col[None, :]) / $(64)) % $(128))), dtype=tl.float32)
      r290_value = tl.dot(r290_left, r290_right)
      r290_value = r290_value.to(tl.float32)
      tl.store($(candidate_vbmm_84_ptr) + ((r290_group * $(2048)) + (r290_row[:, None] * $(64)) + r290_col[None, :]), r290_value)
    }
    r291_z = tl.arange($(16384))
    r291_value = (tl.load($(candidate_vbmm_84_ptr) + ((((r291_z / $(2048)) % $(8)) * $(2048)) + (((r291_z / $(64)) % $(32)) * $(64)) + (r291_z % $(64))), dtype=tl.float32) - tl.load($(candidate_vsum_9_ptr) + ((((r291_z / $(2048)) % $(8)) * $(32)) + ((r291_z / $(64)) % $(32))), dtype=tl.float32))
    r291_value = r291_value.to(tl.float32)
    r291_value = tl.broadcast_to(r291_value, [$(16384)])
    tl.store($(candidate_vsub_43_ptr) + r291_z, r291_value)
    r292_z = tl.arange($(16384))
    r292_value = (tl.load($(candidate_vexp_4_ptr) + ((((r292_z / $(2048)) % $(8)) * $(2048)) + (((r292_z / $(64)) % $(32)) * $(64)) + (r292_z % $(64))), dtype=tl.float32) * tl.load($(candidate_vsub_43_ptr) + ((((r292_z / $(2048)) % $(8)) * $(2048)) + (((r292_z / $(64)) % $(32)) * $(64)) + (r292_z % $(64))), dtype=tl.float32))
    r292_value = r292_value.to(tl.float32)
    r292_value = tl.broadcast_to(r292_value, [$(16384)])
    tl.store($(candidate_vmul_57_ptr) + r292_z, r292_value)
    r293_row = tl.arange($(64))
    r293_col = tl.arange($(192))
    r293_lane = tl.arange($(32))
    for r293_group in range($(8)) {
      r293_left = tl.load($(candidate_vmul_57_ptr) + (((((((r293_group % $(8)) * $(2048)) + (r293_row[:, None] * $(32)) + r293_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + (((((r293_group % $(8)) * $(2048)) + (r293_row[:, None] * $(32)) + r293_lane[None, :]) % $(32)) * $(64)) + (((((r293_group % $(8)) * $(2048)) + (r293_row[:, None] * $(32)) + r293_lane[None, :]) / $(32)) % $(64))), dtype=tl.float32)
      r293_right = tl.load($(candidate_vto_copy_115_ptr) + (((((((r293_group % $(8)) * $(6144)) + (r293_lane[:, None] * $(192)) + r293_col[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r293_group % $(8)) * $(6144)) + (r293_lane[:, None] * $(192)) + r293_col[None, :]) / $(192)) % $(32)) * $(192)) + ((((r293_group % $(8)) * $(6144)) + (r293_lane[:, None] * $(192)) + r293_col[None, :]) % $(192))), dtype=tl.float32)
      r293_value = tl.dot(r293_left, r293_right)
      r293_value = r293_value.to(tl.float32)
      tl.store($(candidate_vbmm_86_ptr) + ((r293_group * $(12288)) + (r293_row[:, None] * $(192)) + r293_col[None, :]), r293_value)
    }
    r294_z = tl.arange($(98304))
    r294_value = (tl.load($(candidate_vbmm_86_ptr) + ((((r294_z / $(12288)) % $(8)) * $(12288)) + (((r294_z / $(192)) % $(64)) * $(192)) + (r294_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r294_value = r294_value.to(tl.float32)
    r294_value = tl.broadcast_to(r294_value, [$(98304)])
    tl.store($(candidate_vmul_59_ptr) + r294_z, r294_value)
    r295_z = tl.arange($(98304))
    r295_value = tl.load($(candidate_vmul_59_ptr) + ((((r295_z / $(192)) % $(8)) * $(12288)) + (((r295_z / $(1536)) % $(64)) * $(192)) + (r295_z % $(192))), dtype=tl.float32)
    r295_value = r295_value.to(tl.bfloat16)
    r295_value = tl.broadcast_to(r295_value, [$(98304)])
    tl.store($(candidate_vto_copy_121_ptr) + r295_z, r295_value)
    r296_row = tl.arange($(64))
    r296_col = tl.arange($(128))
    r296_lane = tl.arange($(32))
    for r296_group in range($(8)) {
      r296_left = tl.load($(candidate_vexp_4_ptr) + (((((((r296_group % $(8)) * $(2048)) + (r296_row[:, None] * $(32)) + r296_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + (((((r296_group % $(8)) * $(2048)) + (r296_row[:, None] * $(32)) + r296_lane[None, :]) % $(32)) * $(64)) + (((((r296_group % $(8)) * $(2048)) + (r296_row[:, None] * $(32)) + r296_lane[None, :]) / $(32)) % $(64))), dtype=tl.float32)
      r296_right = tl.load($(candidate_vto_copy_118_ptr) + (((((((r296_group % $(8)) * $(4096)) + (r296_lane[:, None] * $(128)) + r296_col[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r296_group % $(8)) * $(4096)) + (r296_lane[:, None] * $(128)) + r296_col[None, :]) / $(128)) % $(32)) * $(128)) + ((((r296_group % $(8)) * $(4096)) + (r296_lane[:, None] * $(128)) + r296_col[None, :]) % $(128))), dtype=tl.float32)
      r296_value = tl.dot(r296_left, r296_right)
      r296_value = r296_value.to(tl.float32)
      tl.store($(candidate_vbmm_87_ptr) + ((r296_group * $(8192)) + (r296_row[:, None] * $(128)) + r296_col[None, :]), r296_value)
    }
    r297_z = tl.arange($(65536))
    r297_value = tl.load($(candidate_vbmm_87_ptr) + ((((r297_z / $(128)) % $(8)) * $(8192)) + (((r297_z / $(1024)) % $(64)) * $(128)) + (r297_z % $(128))), dtype=tl.float32)
    r297_value = r297_value.to(tl.bfloat16)
    r297_value = tl.broadcast_to(r297_value, [$(65536)])
    tl.store($(candidate_vto_copy_122_ptr) + r297_z, r297_value)
    r298_z = tl.arange($(147456))
    r298_value = tl.where(((r298_z / $(1536)) % $(96)) < $(64), tl.load($(candidate_vto_copy_121_ptr) + ((((((r298_z / $(1536)) % $(96)) - $(0)) % $(64)) * $(1536)) + (((r298_z / $(192)) % $(8)) * $(192)) + (r298_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r298_value = r298_value.to(tl.bfloat16)
    r298_value = tl.broadcast_to(r298_value, [$(147456)])
    tl.store($(candidate_vcat_58_ptr) + r298_z, r298_value)
    r299_z = tl.arange($(98304))
    r299_value = tl.where(((r299_z / $(1024)) % $(96)) < $(64), tl.load($(candidate_vto_copy_122_ptr) + ((((((r299_z / $(1024)) % $(96)) - $(0)) % $(64)) * $(1024)) + (((r299_z / $(128)) % $(8)) * $(128)) + (r299_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r299_value = r299_value.to(tl.bfloat16)
    r299_value = tl.broadcast_to(r299_value, [$(98304)])
    tl.store($(candidate_vcat_59_ptr) + r299_z, r299_value)
    r300_z = tl.arange($(147456))
    r300_value = tl.load($(candidate_vcat_58_ptr) + ((((((r300_z / $(1536)) % $(96)) - $(0)) % $(96)) * $(1536)) + (((r300_z / $(192)) % $(8)) * $(192)) + (r300_z % $(192))), dtype=tl.bfloat16)
    r300_value = r300_value.to(tl.bfloat16)
    r300_value = tl.broadcast_to(r300_value, [$(147456)])
    tl.store($(candidate_vcat_61_ptr) + r300_z, r300_value)
    r301_z = tl.arange($(98304))
    r301_value = tl.load($(candidate_vcat_59_ptr) + ((((((r301_z / $(1024)) % $(96)) - $(0)) % $(96)) * $(1024)) + (((r301_z / $(128)) % $(8)) * $(128)) + (r301_z % $(128))), dtype=tl.bfloat16)
    r301_value = r301_value.to(tl.bfloat16)
    r301_value = tl.broadcast_to(r301_value, [$(98304)])
    tl.store($(candidate_vcat_62_ptr) + r301_z, r301_value)
    r302_row = tl.arange($(1024))
    r302_col = tl.arange($(512))
    r302_lane = tl.arange($(96))
    for r302_group in range($(1)) {
      r302_left = tl.load($(candidate_vcat_61_ptr) + (((((r302_row[:, None] * $(96)) + r302_lane[None, :]) % $(96)) * $(1536)) + (((((r302_row[:, None] * $(96)) + r302_lane[None, :]) / $(12288)) % $(8)) * $(192)) + ((((r302_row[:, None] * $(96)) + r302_lane[None, :]) / $(96)) % $(128))), dtype=tl.bfloat16)
      r302_right = tl.load($(candidate_vcat_7_ptr) + ((((((r302_lane[:, None] * $(512)) + r302_col[None, :]) / $(512)) % $(96)) * $(576)) + (((r302_lane[:, None] * $(512)) + r302_col[None, :]) % $(512)) + $(55296)), dtype=tl.bfloat16)
      r302_value = tl.dot(r302_left, r302_right)
      r302_value = r302_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_90_ptr) + ((r302_group * $(524288)) + (r302_row[:, None] * $(512)) + r302_col[None, :]), r302_value)
    }
    r303_row = tl.arange($(1024))
    r303_col = tl.arange($(512))
    r303_lane = tl.arange($(96))
    for r303_group in range($(1)) {
      r303_left = tl.load($(candidate_vcat_62_ptr) + (((((r303_row[:, None] * $(96)) + r303_lane[None, :]) % $(96)) * $(1024)) + (((((r303_row[:, None] * $(96)) + r303_lane[None, :]) / $(12288)) % $(8)) * $(128)) + ((((r303_row[:, None] * $(96)) + r303_lane[None, :]) / $(96)) % $(128))), dtype=tl.bfloat16)
      r303_right = tl.load($(candidate_vcat_7_ptr) + ((((((r303_lane[:, None] * $(512)) + r303_col[None, :]) / $(512)) % $(96)) * $(576)) + (((r303_lane[:, None] * $(512)) + r303_col[None, :]) % $(512)) + $(55296)), dtype=tl.bfloat16)
      r303_value = tl.dot(r303_left, r303_right)
      r303_value = r303_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_91_ptr) + ((r303_group * $(524288)) + (r303_row[:, None] * $(512)) + r303_col[None, :]), r303_value)
    }
    r304_z = tl.arange($(1048576))
    r304_value = tl.where(((r304_z / $(512)) % $(256)) < $(128), tl.load($(candidate_vbmm_90_ptr) + ((((((((r304_z / $(131072)) % $(8)) * $(65536)) + ((((r304_z / $(512)) % $(256)) - $(0)) * $(512)) + (r304_z % $(512))) / $(65536)) % $(8)) * $(65536)) + ((((((r304_z / $(131072)) % $(8)) * $(128)) + (((r304_z / $(512)) % $(256)) - $(0))) % $(128)) * $(512)) + (r304_z % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vbmm_91_ptr) + ((((((((r304_z / $(131072)) % $(8)) * $(65536)) + ((((r304_z / $(512)) % $(256)) - $(128)) * $(512)) + (r304_z % $(512))) / $(65536)) % $(8)) * $(65536)) + ((((((r304_z / $(131072)) % $(8)) * $(128)) + (((r304_z / $(512)) % $(256)) - $(128))) % $(128)) * $(512)) + (r304_z % $(512))), dtype=tl.bfloat16))
    r304_value = r304_value.to(tl.bfloat16)
    r304_value = tl.broadcast_to(r304_value, [$(1048576)])
    tl.store($(candidate_vcat_63_ptr) + r304_z, r304_value)
    r305_z = tl.arange($(1048576))
    r305_value = (tl.load($(candidate_vadd_38_ptr) + ((((r305_z / $(512)) % $(2048)) * $(512)) + (r305_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vcat_63_ptr) + ((((((((r305_z / $(512)) % $(2048)) * $(512)) + (r305_z % $(512))) / $(131072)) % $(8)) * $(131072)) + ((((r305_z / $(512)) % $(2048)) % $(256)) * $(512)) + (r305_z % $(512))), dtype=tl.bfloat16))
    r305_value = r305_value.to(tl.bfloat16)
    r305_value = tl.broadcast_to(r305_value, [$(1048576)])
    tl.store($(candidate_vadd_42_ptr) + r305_z, r305_value)
    r306_row = tl.arange($(96))
    r306_col = tl.arange($(1024))
    r306_lane = tl.arange($(512))
    for r306_group in range($(1)) {
      r306_left = tl.load($(candidate_vcat_7_ptr) + ((((((r306_row[:, None] * $(512)) + r306_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r306_row[:, None] * $(512)) + r306_lane[None, :]) % $(512)) + $(55296)), dtype=tl.bfloat16)
      r306_right = tl.load($(candidate_vto_copy_18_ptr) + ((((((r306_lane[:, None] * $(1024)) + r306_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r306_lane[:, None] * $(1024)) + r306_col[None, :]) % $(128)) * $(512)) + ((((r306_lane[:, None] * $(1024)) + r306_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r306_value = tl.dot(r306_left, r306_right)
      r306_value = r306_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_92_ptr) + ((r306_group * $(98304)) + (r306_row[:, None] * $(1024)) + r306_col[None, :]), r306_value)
    }
    r307_row = tl.arange($(96))
    r307_col = tl.arange($(1024))
    r307_lane = tl.arange($(512))
    for r307_group in range($(1)) {
      r307_left = tl.load($(candidate_vcat_7_ptr) + ((((((r307_row[:, None] * $(512)) + r307_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r307_row[:, None] * $(512)) + r307_lane[None, :]) % $(512)) + $(55296)), dtype=tl.bfloat16)
      r307_right = tl.load($(candidate_vto_copy_18_ptr) + ((((((r307_lane[:, None] * $(1024)) + r307_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r307_lane[:, None] * $(1024)) + r307_col[None, :]) % $(128)) * $(512)) + ((((r307_lane[:, None] * $(1024)) + r307_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r307_value = tl.dot(r307_left, r307_right)
      r307_value = r307_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_93_ptr) + ((r307_group * $(98304)) + (r307_row[:, None] * $(1024)) + r307_col[None, :]), r307_value)
    }
    r308_z = tl.arange($(49152))
    r308_value = tl.where((r308_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r308_z / $(1536)) % $(32)) * $(1024)) + (((r308_z / $(192)) % $(8)) * $(128)) + ((r308_z % $(192)) - $(0)) + $(98304)) / $(1024)) % $(288)) * $(1536)) + (((((((r308_z / $(1536)) % $(32)) * $(1024)) + (((r308_z / $(192)) % $(8)) * $(128)) + ((r308_z % $(192)) - $(0)) + $(98304)) / $(128)) % $(8)) * $(192)) + (((((r308_z / $(1536)) % $(32)) * $(1024)) + (((r308_z / $(192)) % $(8)) * $(128)) + ((r308_z % $(192)) - $(0)) + $(98304)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_5_ptr) + ((((((((r308_z / $(1536)) % $(32)) * $(512)) + (((r308_z / $(192)) % $(8)) * $(64)) + ((r308_z % $(192)) - $(128)) + $(49152)) / $(64)) % $(8)) * $(18432)) + (((((((r308_z / $(1536)) % $(32)) * $(512)) + (((r308_z / $(192)) % $(8)) * $(64)) + ((r308_z % $(192)) - $(128)) + $(49152)) / $(512)) % $(288)) * $(64)) + (((((r308_z / $(1536)) % $(32)) * $(512)) + (((r308_z / $(192)) % $(8)) * $(64)) + ((r308_z % $(192)) - $(128)) + $(49152)) % $(64))), dtype=tl.bfloat16))
    r308_value = r308_value.to(tl.bfloat16)
    r308_value = tl.broadcast_to(r308_value, [$(49152)])
    tl.store($(candidate_vcat_64_ptr) + r308_z, r308_value)
    r309_z = tl.arange($(147456))
    r309_value = tl.where((r309_z % $(192)) < $(128), tl.load($(candidate_vbmm_92_ptr) + ((((((((r309_z / $(1536)) % $(96)) * $(1024)) + (((r309_z / $(192)) % $(8)) * $(128)) + ((r309_z % $(192)) - $(0))) / $(1024)) % $(96)) * $(1024)) + (((((((r309_z / $(1536)) % $(96)) * $(1024)) + (((r309_z / $(192)) % $(8)) * $(128)) + ((r309_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r309_z / $(1536)) % $(96)) * $(1024)) + (((r309_z / $(192)) % $(8)) * $(128)) + ((r309_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_7_ptr) + ((((r309_z / $(1536)) % $(96)) * $(576)) + ((((r309_z % $(192)) - $(128)) + $(512)) % $(576)) + $(55296)), dtype=tl.bfloat16))
    r309_value = r309_value.to(tl.bfloat16)
    r309_value = tl.broadcast_to(r309_value, [$(147456)])
    tl.store($(candidate_vcat_65_ptr) + r309_z, r309_value)
    r310_z = tl.arange($(49152))
    r310_value = tl.load($(candidate_vcat_64_ptr) + ((((r310_z / $(192)) % $(32)) * $(1536)) + (((r310_z / $(6144)) % $(8)) * $(192)) + (r310_z % $(192))), dtype=tl.bfloat16)
    r310_value = r310_value.to(tl.float32)
    r310_value = tl.broadcast_to(r310_value, [$(49152)])
    tl.store($(candidate_vto_copy_123_ptr) + r310_z, r310_value)
    r311_z = tl.arange($(49152))
    r311_value = tl.load($(candidate_vcat_65_ptr) + ((((r311_z / $(192)) % $(32)) * $(1536)) + (((r311_z / $(6144)) % $(8)) * $(192)) + (r311_z % $(192))), dtype=tl.bfloat16)
    r311_value = r311_value.to(tl.float32)
    r311_value = tl.broadcast_to(r311_value, [$(49152)])
    tl.store($(candidate_vto_copy_124_ptr) + r311_z, r311_value)
    r312_row = tl.arange($(32))
    r312_col = tl.arange($(32))
    r312_lane = tl.arange($(192))
    for r312_group in range($(8)) {
      r312_left = tl.load($(candidate_vto_copy_123_ptr) + (((((((r312_group % $(8)) * $(6144)) + (r312_row[:, None] * $(192)) + r312_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r312_group % $(8)) * $(6144)) + (r312_row[:, None] * $(192)) + r312_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r312_group % $(8)) * $(6144)) + (r312_row[:, None] * $(192)) + r312_lane[None, :]) % $(192))), dtype=tl.float32)
      r312_right = tl.load($(candidate_vto_copy_124_ptr) + (((((((r312_group % $(8)) * $(6144)) + (r312_lane[:, None] * $(32)) + r312_col[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r312_group % $(8)) * $(6144)) + (r312_lane[:, None] * $(32)) + r312_col[None, :]) % $(32)) * $(192)) + (((((r312_group % $(8)) * $(6144)) + (r312_lane[:, None] * $(32)) + r312_col[None, :]) / $(32)) % $(192))), dtype=tl.float32)
      r312_value = tl.dot(r312_left, r312_right)
      r312_value = r312_value.to(tl.float32)
      tl.store($(candidate_vbmm_94_ptr) + ((r312_group * $(1024)) + (r312_row[:, None] * $(32)) + r312_col[None, :]), r312_value)
    }
    r313_z = tl.arange($(8192))
    r313_value = (tl.load($(candidate_vbmm_94_ptr) + ((((r313_z / $(1024)) % $(8)) * $(1024)) + (((r313_z / $(32)) % $(32)) * $(32)) + (r313_z % $(32))), dtype=tl.float32) * 0.07216878364870322)
    r313_value = r313_value.to(tl.float32)
    r313_value = tl.broadcast_to(r313_value, [$(8192)])
    tl.store($(candidate_vmul_60_ptr) + r313_z, r313_value)
    r314_z = tl.arange($(32))
    r314_value = (tl.cast(r314_z, tl.int64) + tl.full([], $(32), dtype=tl.int64))
    r314_value = tl.broadcast_to(r314_value, [$(32)])
    tl.store($(candidate_vadd_43_ptr) + r314_z, r314_value)
    r315_z = tl.arange($(32))
    r315_value = (tl.load($(candidate_vadd_43_ptr) + r315_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r315_value = tl.broadcast_to(r315_value, [$(32)])
    tl.store($(candidate_vsub_44_ptr) + r315_z, r315_value)
    r316_z = tl.arange($(1024))
    r316_value = (tl.cast((r316_z % $(32)), tl.int64) > tl.load($(candidate_vsub_44_ptr) + ((r316_z / $(32)) % $(32)), dtype=tl.int64))
    r316_value = tl.broadcast_to(r316_value, [$(1024)])
    tl.store($(candidate_vgt_19_ptr) + r316_z, r316_value)
    r317_z = tl.arange($(8192))
    r317_value = tl.where(tl.load($(candidate_vgt_19_ptr) + ((((r317_z / $(32)) % $(32)) * $(32)) + (r317_z % $(32))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_60_ptr) + ((((r317_z / $(1024)) % $(8)) * $(1024)) + (((r317_z / $(32)) % $(32)) * $(32)) + (r317_z % $(32))), dtype=tl.float32))
    r317_value = r317_value.to(tl.float32)
    r317_value = tl.broadcast_to(r317_value, [$(8192)])
    tl.store($(candidate_vmasked_fill_19_ptr) + r317_z, r317_value)
    r318_z = tl.arange($(8192))
    r318_lane = tl.arange($(32))
    r318_row_values = tl.load($(candidate_vmasked_fill_19_ptr) + ((((r318_z[:, None] / $(1024)) % $(8)) * $(1024)) + (((r318_z[:, None] / $(32)) % $(32)) * $(32)) + (r318_lane[None, :] % $(32))), dtype=tl.float32)
    r318_maximum = tl.max(r318_row_values, axis=1, keep_dims=true)
    r318_weights = tl.exp(r318_row_values - r318_maximum)
    r318_total = tl.sum(r318_weights, axis=1)
    r318_maximum = tl.max(r318_row_values, axis=1)
    r318_value = tl.exp(tl.load($(candidate_vmasked_fill_19_ptr) + ((((r318_z / $(1024)) % $(8)) * $(1024)) + (((r318_z / $(32)) % $(32)) * $(32)) + (r318_z % $(32))), dtype=tl.float32) - r318_maximum) / r318_total
    r318_value = r318_value.to(tl.float32)
    r318_value = tl.broadcast_to(r318_value, [$(8192)])
    tl.store($(candidate_vsoftmax_14_ptr) + r318_z, r318_value)
    r319_z = tl.arange($(8192))
    r319_value = tl.load($(candidate_vsoftmax_14_ptr) + ((((r319_z / $(1024)) % $(8)) * $(1024)) + (((r319_z / $(32)) % $(32)) * $(32)) + (r319_z % $(32))), dtype=tl.float32)
    r319_value = r319_value.to(tl.bfloat16)
    r319_value = tl.broadcast_to(r319_value, [$(8192)])
    tl.store($(candidate_vto_copy_125_ptr) + r319_z, r319_value)
    r320_row = tl.arange($(32))
    r320_col = tl.arange($(128))
    r320_lane = tl.arange($(32))
    for r320_group in range($(8)) {
      r320_left = tl.load($(candidate_vto_copy_125_ptr) + (((((((r320_group % $(8)) * $(1024)) + (r320_row[:, None] * $(32)) + r320_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + ((((((r320_group % $(8)) * $(1024)) + (r320_row[:, None] * $(32)) + r320_lane[None, :]) / $(32)) % $(32)) * $(32)) + ((((r320_group % $(8)) * $(1024)) + (r320_row[:, None] * $(32)) + r320_lane[None, :]) % $(32))), dtype=tl.bfloat16)
      r320_right = tl.load($(candidate_vbmm_93_ptr) + (((((((r320_group % $(8)) * $(4096)) + (r320_lane[:, None] * $(128)) + r320_col[None, :]) / $(128)) % $(32)) * $(1024)) + ((((((r320_group % $(8)) * $(4096)) + (r320_lane[:, None] * $(128)) + r320_col[None, :]) / $(4096)) % $(8)) * $(128)) + ((((r320_group % $(8)) * $(4096)) + (r320_lane[:, None] * $(128)) + r320_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r320_value = tl.dot(r320_left, r320_right)
      r320_value = r320_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_95_ptr) + ((r320_group * $(4096)) + (r320_row[:, None] * $(128)) + r320_col[None, :]), r320_value)
    }
    r321_z = tl.arange($(256))
    r321_lane = tl.arange($(32))
    r321_row_values = tl.load($(candidate_vmasked_fill_19_ptr) + ((((r321_z[:, None] / $(32)) % $(8)) * $(1024)) + ((r321_z[:, None] % $(32)) * $(32)) + (r321_lane[None, :] % $(32))), dtype=tl.float32)
    r321_maximum = tl.max(r321_row_values, axis=1, keep_dims=true)
    r321_weights = tl.exp(r321_row_values - r321_maximum)
    r321_total = tl.sum(r321_weights, axis=1)
    r321_maximum = tl.max(r321_row_values, axis=1)
    r321_value = r321_maximum + tl.log(r321_total)
    r321_value = r321_value.to(tl.float32)
    r321_value = tl.broadcast_to(r321_value, [$(256)])
    tl.store($(candidate_vlogsumexp_14_ptr) + r321_z, r321_value)
    r322_z = tl.arange($(32768))
    r322_value = tl.load($(candidate_vbmm_95_ptr) + ((((((((r322_z / $(128)) % $(8)) * $(4096)) + ((((r322_z / $(1024)) % $(32)) - $(0)) * $(128)) + (r322_z % $(128))) / $(4096)) % $(8)) * $(4096)) + ((((((r322_z / $(128)) % $(8)) * $(32)) + (((r322_z / $(1024)) % $(32)) - $(0))) % $(32)) * $(128)) + (r322_z % $(128))), dtype=tl.bfloat16)
    r322_value = r322_value.to(tl.bfloat16)
    r322_value = tl.broadcast_to(r322_value, [$(32768)])
    tl.store($(candidate_vcat_66_ptr) + r322_z, r322_value)
    r323_z = tl.arange($(256))
    r323_value = tl.load($(candidate_vlogsumexp_14_ptr) + (((r323_z % $(8)) * $(32)) + ((((r323_z / $(8)) % $(32)) - $(0)) % $(32))), dtype=tl.float32)
    r323_value = r323_value.to(tl.float32)
    r323_value = tl.broadcast_to(r323_value, [$(256)])
    tl.store($(candidate_vcat_67_ptr) + r323_z, r323_value)
    r324_z = tl.arange($(49152))
    r324_value = tl.load($(candidate_vcat_64_ptr) + ((((r324_z / $(192)) % $(32)) * $(1536)) + (((r324_z / $(6144)) % $(8)) * $(192)) + (r324_z % $(192))), dtype=tl.bfloat16)
    r324_value = r324_value.to(tl.float32)
    r324_value = tl.broadcast_to(r324_value, [$(49152)])
    tl.store($(candidate_vto_copy_126_ptr) + r324_z, r324_value)
    r325_z = tl.arange($(49152))
    r325_value = tl.load($(candidate_vcat_65_ptr) + ((((r325_z / $(192)) % $(32)) * $(1536)) + (((r325_z / $(6144)) % $(8)) * $(192)) + (r325_z % $(192))), dtype=tl.bfloat16)
    r325_value = r325_value.to(tl.float32)
    r325_value = tl.broadcast_to(r325_value, [$(49152)])
    tl.store($(candidate_vto_copy_127_ptr) + r325_z, r325_value)
    r326_z = tl.arange($(32768))
    r326_value = tl.load($(candidate_vbmm_93_ptr) + ((((r326_z / $(128)) % $(32)) * $(1024)) + (((r326_z / $(4096)) % $(8)) * $(128)) + (r326_z % $(128))), dtype=tl.bfloat16)
    r326_value = r326_value.to(tl.float32)
    r326_value = tl.broadcast_to(r326_value, [$(32768)])
    tl.store($(candidate_vto_copy_128_ptr) + r326_z, r326_value)
    r327_z = tl.arange($(32768))
    r327_value = tl.load($(candidate_vmm_22_ptr) + ((((r327_z / $(128)) % $(32)) * $(1024)) + (((r327_z / $(4096)) % $(8)) * $(128)) + (r327_z % $(128)) + $(98304)), dtype=tl.bfloat16)
    r327_value = r327_value.to(tl.float32)
    r327_value = tl.broadcast_to(r327_value, [$(32768)])
    tl.store($(candidate_vto_copy_129_ptr) + r327_z, r327_value)
    r328_z = tl.arange($(32768))
    r328_value = tl.load($(candidate_vcat_66_ptr) + ((((r328_z / $(128)) % $(32)) * $(1024)) + (((r328_z / $(4096)) % $(8)) * $(128)) + (r328_z % $(128))), dtype=tl.bfloat16)
    r328_value = r328_value.to(tl.float32)
    r328_value = tl.broadcast_to(r328_value, [$(32768)])
    tl.store($(candidate_vto_copy_130_ptr) + r328_z, r328_value)
    r329_row = tl.arange($(32))
    r329_col = tl.arange($(32))
    r329_lane = tl.arange($(192))
    for r329_group in range($(8)) {
      r329_left = tl.load($(candidate_vto_copy_126_ptr) + (((((((r329_group % $(8)) * $(6144)) + (r329_row[:, None] * $(192)) + r329_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r329_group % $(8)) * $(6144)) + (r329_row[:, None] * $(192)) + r329_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r329_group % $(8)) * $(6144)) + (r329_row[:, None] * $(192)) + r329_lane[None, :]) % $(192))), dtype=tl.float32)
      r329_right = tl.load($(candidate_vto_copy_127_ptr) + (((((((r329_group % $(8)) * $(6144)) + (r329_lane[:, None] * $(32)) + r329_col[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r329_group % $(8)) * $(6144)) + (r329_lane[:, None] * $(32)) + r329_col[None, :]) % $(32)) * $(192)) + (((((r329_group % $(8)) * $(6144)) + (r329_lane[:, None] * $(32)) + r329_col[None, :]) / $(32)) % $(192))), dtype=tl.float32)
      r329_value = tl.dot(r329_left, r329_right)
      r329_value = r329_value.to(tl.float32)
      tl.store($(candidate_vbmm_96_ptr) + ((r329_group * $(1024)) + (r329_row[:, None] * $(32)) + r329_col[None, :]), r329_value)
    }
    r330_z = tl.arange($(8192))
    r330_value = (tl.load($(candidate_vbmm_96_ptr) + ((((r330_z / $(1024)) % $(8)) * $(1024)) + (((r330_z / $(32)) % $(32)) * $(32)) + (r330_z % $(32))), dtype=tl.float32) * 0.07216878364870322)
    r330_value = r330_value.to(tl.float32)
    r330_value = tl.broadcast_to(r330_value, [$(8192)])
    tl.store($(candidate_vmul_61_ptr) + r330_z, r330_value)
    r331_z = tl.arange($(32))
    r331_value = (tl.cast(r331_z, tl.int64) + tl.full([], $(32), dtype=tl.int64))
    r331_value = tl.broadcast_to(r331_value, [$(32)])
    tl.store($(candidate_vadd_44_ptr) + r331_z, r331_value)
    r332_z = tl.arange($(32))
    r332_value = (tl.load($(candidate_vadd_44_ptr) + r332_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r332_value = tl.broadcast_to(r332_value, [$(32)])
    tl.store($(candidate_vsub_45_ptr) + r332_z, r332_value)
    r333_z = tl.arange($(1024))
    r333_value = (tl.cast((r333_z % $(32)), tl.int64) > tl.load($(candidate_vsub_45_ptr) + ((r333_z / $(32)) % $(32)), dtype=tl.int64))
    r333_value = tl.broadcast_to(r333_value, [$(1024)])
    tl.store($(candidate_vgt_20_ptr) + r333_z, r333_value)
    r334_z = tl.arange($(8192))
    r334_value = tl.where(tl.load($(candidate_vgt_20_ptr) + ((((r334_z / $(32)) % $(32)) * $(32)) + (r334_z % $(32))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_61_ptr) + ((((r334_z / $(1024)) % $(8)) * $(1024)) + (((r334_z / $(32)) % $(32)) * $(32)) + (r334_z % $(32))), dtype=tl.float32))
    r334_value = r334_value.to(tl.float32)
    r334_value = tl.broadcast_to(r334_value, [$(8192)])
    tl.store($(candidate_vmasked_fill_20_ptr) + r334_z, r334_value)
    r335_z = tl.arange($(8192))
    r335_value = (tl.load($(candidate_vmasked_fill_20_ptr) + ((((r335_z / $(1024)) % $(8)) * $(1024)) + (((r335_z / $(32)) % $(32)) * $(32)) + (r335_z % $(32))), dtype=tl.float32) - tl.load($(candidate_vcat_67_ptr) + ((((r335_z / $(32)) % $(32)) * $(8)) + ((r335_z / $(1024)) % $(8))), dtype=tl.float32))
    r335_value = r335_value.to(tl.float32)
    r335_value = tl.broadcast_to(r335_value, [$(8192)])
    tl.store($(candidate_vsub_46_ptr) + r335_z, r335_value)
    r336_z = tl.arange($(8192))
    r336_value = tl.exp(tl.load($(candidate_vsub_46_ptr) + ((((r336_z / $(1024)) % $(8)) * $(1024)) + (((r336_z / $(32)) % $(32)) * $(32)) + (r336_z % $(32))), dtype=tl.float32))
    r336_value = r336_value.to(tl.float32)
    r336_value = tl.broadcast_to(r336_value, [$(8192)])
    tl.store($(candidate_vexp_5_ptr) + r336_z, r336_value)
    r337_z = tl.arange($(32768))
    r337_value = (tl.load($(candidate_vto_copy_129_ptr) + ((((r337_z / $(4096)) % $(8)) * $(4096)) + (((r337_z / $(128)) % $(32)) * $(128)) + (r337_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_130_ptr) + ((((r337_z / $(4096)) % $(8)) * $(4096)) + (((r337_z / $(128)) % $(32)) * $(128)) + (r337_z % $(128))), dtype=tl.float32))
    r337_value = r337_value.to(tl.float32)
    r337_value = tl.broadcast_to(r337_value, [$(32768)])
    tl.store($(candidate_vmul_62_ptr) + r337_z, r337_value)
    r338_z = tl.arange($(256))
    r338_lane = tl.arange($(128))
    r338_value = tl.sum(tl.load($(candidate_vmul_62_ptr) + ((((r338_z[:, None] / $(32)) % $(8)) * $(4096)) + ((r338_z[:, None] % $(32)) * $(128)) + r338_lane[None, :]), dtype=tl.float32), axis=1)
    r338_value = r338_value.to(tl.float32)
    r338_value = tl.broadcast_to(r338_value, [$(256)])
    tl.store($(candidate_vsum_11_ptr) + r338_z, r338_value)
    r339_row = tl.arange($(32))
    r339_col = tl.arange($(32))
    r339_lane = tl.arange($(128))
    for r339_group in range($(8)) {
      r339_left = tl.load($(candidate_vto_copy_129_ptr) + (((((((r339_group % $(8)) * $(4096)) + (r339_row[:, None] * $(128)) + r339_lane[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r339_group % $(8)) * $(4096)) + (r339_row[:, None] * $(128)) + r339_lane[None, :]) / $(128)) % $(32)) * $(128)) + ((((r339_group % $(8)) * $(4096)) + (r339_row[:, None] * $(128)) + r339_lane[None, :]) % $(128))), dtype=tl.float32)
      r339_right = tl.load($(candidate_vto_copy_128_ptr) + (((((((r339_group % $(8)) * $(4096)) + (r339_lane[:, None] * $(32)) + r339_col[None, :]) / $(4096)) % $(8)) * $(4096)) + (((((r339_group % $(8)) * $(4096)) + (r339_lane[:, None] * $(32)) + r339_col[None, :]) % $(32)) * $(128)) + (((((r339_group % $(8)) * $(4096)) + (r339_lane[:, None] * $(32)) + r339_col[None, :]) / $(32)) % $(128))), dtype=tl.float32)
      r339_value = tl.dot(r339_left, r339_right)
      r339_value = r339_value.to(tl.float32)
      tl.store($(candidate_vbmm_97_ptr) + ((r339_group * $(1024)) + (r339_row[:, None] * $(32)) + r339_col[None, :]), r339_value)
    }
    r340_z = tl.arange($(8192))
    r340_value = (tl.load($(candidate_vbmm_97_ptr) + ((((r340_z / $(1024)) % $(8)) * $(1024)) + (((r340_z / $(32)) % $(32)) * $(32)) + (r340_z % $(32))), dtype=tl.float32) - tl.load($(candidate_vsum_11_ptr) + ((((r340_z / $(1024)) % $(8)) * $(32)) + ((r340_z / $(32)) % $(32))), dtype=tl.float32))
    r340_value = r340_value.to(tl.float32)
    r340_value = tl.broadcast_to(r340_value, [$(8192)])
    tl.store($(candidate_vsub_47_ptr) + r340_z, r340_value)
    r341_z = tl.arange($(8192))
    r341_value = (tl.load($(candidate_vexp_5_ptr) + ((((r341_z / $(1024)) % $(8)) * $(1024)) + (((r341_z / $(32)) % $(32)) * $(32)) + (r341_z % $(32))), dtype=tl.float32) * tl.load($(candidate_vsub_47_ptr) + ((((r341_z / $(1024)) % $(8)) * $(1024)) + (((r341_z / $(32)) % $(32)) * $(32)) + (r341_z % $(32))), dtype=tl.float32))
    r341_value = r341_value.to(tl.float32)
    r341_value = tl.broadcast_to(r341_value, [$(8192)])
    tl.store($(candidate_vmul_63_ptr) + r341_z, r341_value)
    r342_row = tl.arange($(32))
    r342_col = tl.arange($(192))
    r342_lane = tl.arange($(32))
    for r342_group in range($(8)) {
      r342_left = tl.load($(candidate_vmul_63_ptr) + (((((((r342_group % $(8)) * $(1024)) + (r342_row[:, None] * $(32)) + r342_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + (((((r342_group % $(8)) * $(1024)) + (r342_row[:, None] * $(32)) + r342_lane[None, :]) % $(32)) * $(32)) + (((((r342_group % $(8)) * $(1024)) + (r342_row[:, None] * $(32)) + r342_lane[None, :]) / $(32)) % $(32))), dtype=tl.float32)
      r342_right = tl.load($(candidate_vto_copy_126_ptr) + (((((((r342_group % $(8)) * $(6144)) + (r342_lane[:, None] * $(192)) + r342_col[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r342_group % $(8)) * $(6144)) + (r342_lane[:, None] * $(192)) + r342_col[None, :]) / $(192)) % $(32)) * $(192)) + ((((r342_group % $(8)) * $(6144)) + (r342_lane[:, None] * $(192)) + r342_col[None, :]) % $(192))), dtype=tl.float32)
      r342_value = tl.dot(r342_left, r342_right)
      r342_value = r342_value.to(tl.float32)
      tl.store($(candidate_vbmm_99_ptr) + ((r342_group * $(6144)) + (r342_row[:, None] * $(192)) + r342_col[None, :]), r342_value)
    }
    r343_z = tl.arange($(49152))
    r343_value = (tl.load($(candidate_vbmm_99_ptr) + ((((r343_z / $(6144)) % $(8)) * $(6144)) + (((r343_z / $(192)) % $(32)) * $(192)) + (r343_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r343_value = r343_value.to(tl.float32)
    r343_value = tl.broadcast_to(r343_value, [$(49152)])
    tl.store($(candidate_vmul_65_ptr) + r343_z, r343_value)
    r344_z = tl.arange($(49152))
    r344_value = tl.load($(candidate_vmul_65_ptr) + ((((r344_z / $(192)) % $(8)) * $(6144)) + (((r344_z / $(1536)) % $(32)) * $(192)) + (r344_z % $(192))), dtype=tl.float32)
    r344_value = r344_value.to(tl.bfloat16)
    r344_value = tl.broadcast_to(r344_value, [$(49152)])
    tl.store($(candidate_vto_copy_132_ptr) + r344_z, r344_value)
    r345_row = tl.arange($(32))
    r345_col = tl.arange($(128))
    r345_lane = tl.arange($(32))
    for r345_group in range($(8)) {
      r345_left = tl.load($(candidate_vexp_5_ptr) + (((((((r345_group % $(8)) * $(1024)) + (r345_row[:, None] * $(32)) + r345_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + (((((r345_group % $(8)) * $(1024)) + (r345_row[:, None] * $(32)) + r345_lane[None, :]) % $(32)) * $(32)) + (((((r345_group % $(8)) * $(1024)) + (r345_row[:, None] * $(32)) + r345_lane[None, :]) / $(32)) % $(32))), dtype=tl.float32)
      r345_right = tl.load($(candidate_vto_copy_129_ptr) + (((((((r345_group % $(8)) * $(4096)) + (r345_lane[:, None] * $(128)) + r345_col[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r345_group % $(8)) * $(4096)) + (r345_lane[:, None] * $(128)) + r345_col[None, :]) / $(128)) % $(32)) * $(128)) + ((((r345_group % $(8)) * $(4096)) + (r345_lane[:, None] * $(128)) + r345_col[None, :]) % $(128))), dtype=tl.float32)
      r345_value = tl.dot(r345_left, r345_right)
      r345_value = r345_value.to(tl.float32)
      tl.store($(candidate_vbmm_100_ptr) + ((r345_group * $(4096)) + (r345_row[:, None] * $(128)) + r345_col[None, :]), r345_value)
    }
    r346_z = tl.arange($(32768))
    r346_value = tl.load($(candidate_vbmm_100_ptr) + ((((r346_z / $(128)) % $(8)) * $(4096)) + (((r346_z / $(1024)) % $(32)) * $(128)) + (r346_z % $(128))), dtype=tl.float32)
    r346_value = r346_value.to(tl.bfloat16)
    r346_value = tl.broadcast_to(r346_value, [$(32768)])
    tl.store($(candidate_vto_copy_133_ptr) + r346_z, r346_value)
    r347_z = tl.arange($(147456))
    r347_value = tl.where(((r347_z / $(1536)) % $(96)) < $(32), tl.load($(candidate_vto_copy_132_ptr) + ((((((r347_z / $(1536)) % $(96)) - $(0)) % $(32)) * $(1536)) + (((r347_z / $(192)) % $(8)) * $(192)) + (r347_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r347_value = r347_value.to(tl.bfloat16)
    r347_value = tl.broadcast_to(r347_value, [$(147456)])
    tl.store($(candidate_vcat_68_ptr) + r347_z, r347_value)
    r348_z = tl.arange($(98304))
    r348_value = tl.where(((r348_z / $(1024)) % $(96)) < $(32), tl.load($(candidate_vto_copy_133_ptr) + ((((((r348_z / $(1024)) % $(96)) - $(0)) % $(32)) * $(1024)) + (((r348_z / $(128)) % $(8)) * $(128)) + (r348_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r348_value = r348_value.to(tl.bfloat16)
    r348_value = tl.broadcast_to(r348_value, [$(98304)])
    tl.store($(candidate_vcat_69_ptr) + r348_z, r348_value)
    r349_z = tl.arange($(147456))
    r349_value = tl.load($(candidate_vcat_68_ptr) + ((((((r349_z / $(1536)) % $(96)) - $(0)) % $(96)) * $(1536)) + (((r349_z / $(192)) % $(8)) * $(192)) + (r349_z % $(192))), dtype=tl.bfloat16)
    r349_value = r349_value.to(tl.bfloat16)
    r349_value = tl.broadcast_to(r349_value, [$(147456)])
    tl.store($(candidate_vcat_71_ptr) + r349_z, r349_value)
    r350_z = tl.arange($(98304))
    r350_value = tl.load($(candidate_vcat_69_ptr) + ((((((r350_z / $(1024)) % $(96)) - $(0)) % $(96)) * $(1024)) + (((r350_z / $(128)) % $(8)) * $(128)) + (r350_z % $(128))), dtype=tl.bfloat16)
    r350_value = r350_value.to(tl.bfloat16)
    r350_value = tl.broadcast_to(r350_value, [$(98304)])
    tl.store($(candidate_vcat_72_ptr) + r350_z, r350_value)
    r351_row = tl.arange($(1024))
    r351_col = tl.arange($(512))
    r351_lane = tl.arange($(96))
    for r351_group in range($(1)) {
      r351_left = tl.load($(candidate_vcat_71_ptr) + (((((r351_row[:, None] * $(96)) + r351_lane[None, :]) % $(96)) * $(1536)) + (((((r351_row[:, None] * $(96)) + r351_lane[None, :]) / $(12288)) % $(8)) * $(192)) + ((((r351_row[:, None] * $(96)) + r351_lane[None, :]) / $(96)) % $(128))), dtype=tl.bfloat16)
      r351_right = tl.load($(candidate_vcat_7_ptr) + ((((((r351_lane[:, None] * $(512)) + r351_col[None, :]) / $(512)) % $(96)) * $(576)) + (((r351_lane[:, None] * $(512)) + r351_col[None, :]) % $(512)) + $(55296)), dtype=tl.bfloat16)
      r351_value = tl.dot(r351_left, r351_right)
      r351_value = r351_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_103_ptr) + ((r351_group * $(524288)) + (r351_row[:, None] * $(512)) + r351_col[None, :]), r351_value)
    }
    r352_row = tl.arange($(1024))
    r352_col = tl.arange($(512))
    r352_lane = tl.arange($(96))
    for r352_group in range($(1)) {
      r352_left = tl.load($(candidate_vcat_72_ptr) + (((((r352_row[:, None] * $(96)) + r352_lane[None, :]) % $(96)) * $(1024)) + (((((r352_row[:, None] * $(96)) + r352_lane[None, :]) / $(12288)) % $(8)) * $(128)) + ((((r352_row[:, None] * $(96)) + r352_lane[None, :]) / $(96)) % $(128))), dtype=tl.bfloat16)
      r352_right = tl.load($(candidate_vcat_7_ptr) + ((((((r352_lane[:, None] * $(512)) + r352_col[None, :]) / $(512)) % $(96)) * $(576)) + (((r352_lane[:, None] * $(512)) + r352_col[None, :]) % $(512)) + $(55296)), dtype=tl.bfloat16)
      r352_value = tl.dot(r352_left, r352_right)
      r352_value = r352_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_104_ptr) + ((r352_group * $(524288)) + (r352_row[:, None] * $(512)) + r352_col[None, :]), r352_value)
    }
    r353_z = tl.arange($(1048576))
    r353_value = tl.where(((r353_z / $(512)) % $(256)) < $(128), tl.load($(candidate_vbmm_103_ptr) + ((((((((r353_z / $(131072)) % $(8)) * $(65536)) + ((((r353_z / $(512)) % $(256)) - $(0)) * $(512)) + (r353_z % $(512))) / $(65536)) % $(8)) * $(65536)) + ((((((r353_z / $(131072)) % $(8)) * $(128)) + (((r353_z / $(512)) % $(256)) - $(0))) % $(128)) * $(512)) + (r353_z % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vbmm_104_ptr) + ((((((((r353_z / $(131072)) % $(8)) * $(65536)) + ((((r353_z / $(512)) % $(256)) - $(128)) * $(512)) + (r353_z % $(512))) / $(65536)) % $(8)) * $(65536)) + ((((((r353_z / $(131072)) % $(8)) * $(128)) + (((r353_z / $(512)) % $(256)) - $(128))) % $(128)) * $(512)) + (r353_z % $(512))), dtype=tl.bfloat16))
    r353_value = r353_value.to(tl.bfloat16)
    r353_value = tl.broadcast_to(r353_value, [$(1048576)])
    tl.store($(candidate_vcat_73_ptr) + r353_z, r353_value)
    r354_z = tl.arange($(1048576))
    r354_value = (tl.load($(candidate_vadd_42_ptr) + ((((r354_z / $(512)) % $(2048)) * $(512)) + (r354_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vcat_73_ptr) + ((((((((r354_z / $(512)) % $(2048)) * $(512)) + (r354_z % $(512))) / $(131072)) % $(8)) * $(131072)) + ((((r354_z / $(512)) % $(2048)) % $(256)) * $(512)) + (r354_z % $(512))), dtype=tl.bfloat16))
    r354_value = r354_value.to(tl.bfloat16)
    r354_value = tl.broadcast_to(r354_value, [$(1048576)])
    tl.store($(candidate_vadd_46_ptr) + r354_z, r354_value)
    r355_row = tl.arange($(96))
    r355_col = tl.arange($(1024))
    r355_lane = tl.arange($(512))
    for r355_group in range($(1)) {
      r355_left = tl.load($(candidate_vcat_7_ptr) + ((((((r355_row[:, None] * $(512)) + r355_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r355_row[:, None] * $(512)) + r355_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r355_right = tl.load($(candidate_vto_copy_18_ptr) + ((((((r355_lane[:, None] * $(1024)) + r355_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r355_lane[:, None] * $(1024)) + r355_col[None, :]) % $(128)) * $(512)) + ((((r355_lane[:, None] * $(1024)) + r355_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r355_value = tl.dot(r355_left, r355_right)
      r355_value = r355_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_105_ptr) + ((r355_group * $(98304)) + (r355_row[:, None] * $(1024)) + r355_col[None, :]), r355_value)
    }
    r356_row = tl.arange($(96))
    r356_col = tl.arange($(1024))
    r356_lane = tl.arange($(512))
    for r356_group in range($(1)) {
      r356_left = tl.load($(candidate_vcat_7_ptr) + ((((((r356_row[:, None] * $(512)) + r356_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r356_row[:, None] * $(512)) + r356_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r356_right = tl.load($(candidate_vto_copy_18_ptr) + ((((((r356_lane[:, None] * $(1024)) + r356_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r356_lane[:, None] * $(1024)) + r356_col[None, :]) % $(128)) * $(512)) + ((((r356_lane[:, None] * $(1024)) + r356_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r356_value = tl.dot(r356_left, r356_right)
      r356_value = r356_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_106_ptr) + ((r356_group * $(98304)) + (r356_row[:, None] * $(1024)) + r356_col[None, :]), r356_value)
    }
    r357_z = tl.arange($(49152))
    r357_value = tl.where((r357_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r357_z / $(1536)) % $(32)) * $(1024)) + (((r357_z / $(192)) % $(8)) * $(128)) + ((r357_z % $(192)) - $(0)) + $(65536)) / $(1024)) % $(288)) * $(1536)) + (((((((r357_z / $(1536)) % $(32)) * $(1024)) + (((r357_z / $(192)) % $(8)) * $(128)) + ((r357_z % $(192)) - $(0)) + $(65536)) / $(128)) % $(8)) * $(192)) + (((((r357_z / $(1536)) % $(32)) * $(1024)) + (((r357_z / $(192)) % $(8)) * $(128)) + ((r357_z % $(192)) - $(0)) + $(65536)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_5_ptr) + ((((((((r357_z / $(1536)) % $(32)) * $(512)) + (((r357_z / $(192)) % $(8)) * $(64)) + ((r357_z % $(192)) - $(128)) + $(32768)) / $(64)) % $(8)) * $(18432)) + (((((((r357_z / $(1536)) % $(32)) * $(512)) + (((r357_z / $(192)) % $(8)) * $(64)) + ((r357_z % $(192)) - $(128)) + $(32768)) / $(512)) % $(288)) * $(64)) + (((((r357_z / $(1536)) % $(32)) * $(512)) + (((r357_z / $(192)) % $(8)) * $(64)) + ((r357_z % $(192)) - $(128)) + $(32768)) % $(64))), dtype=tl.bfloat16))
    r357_value = r357_value.to(tl.bfloat16)
    r357_value = tl.broadcast_to(r357_value, [$(49152)])
    tl.store($(candidate_vcat_74_ptr) + r357_z, r357_value)
    r358_z = tl.arange($(147456))
    r358_value = tl.where((r358_z % $(192)) < $(128), tl.load($(candidate_vbmm_105_ptr) + ((((((((r358_z / $(1536)) % $(96)) * $(1024)) + (((r358_z / $(192)) % $(8)) * $(128)) + ((r358_z % $(192)) - $(0))) / $(1024)) % $(96)) * $(1024)) + (((((((r358_z / $(1536)) % $(96)) * $(1024)) + (((r358_z / $(192)) % $(8)) * $(128)) + ((r358_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r358_z / $(1536)) % $(96)) * $(1024)) + (((r358_z / $(192)) % $(8)) * $(128)) + ((r358_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_7_ptr) + ((((r358_z / $(1536)) % $(96)) * $(576)) + ((((r358_z % $(192)) - $(128)) + $(512)) % $(576))), dtype=tl.bfloat16))
    r358_value = r358_value.to(tl.bfloat16)
    r358_value = tl.broadcast_to(r358_value, [$(147456)])
    tl.store($(candidate_vcat_75_ptr) + r358_z, r358_value)
    r359_z = tl.arange($(49152))
    r359_value = tl.load($(candidate_vcat_74_ptr) + ((((r359_z / $(192)) % $(32)) * $(1536)) + (((r359_z / $(6144)) % $(8)) * $(192)) + (r359_z % $(192))), dtype=tl.bfloat16)
    r359_value = r359_value.to(tl.float32)
    r359_value = tl.broadcast_to(r359_value, [$(49152)])
    tl.store($(candidate_vto_copy_134_ptr) + r359_z, r359_value)
    r360_z = tl.arange($(147456))
    r360_value = tl.load($(candidate_vcat_75_ptr) + ((((r360_z / $(192)) % $(96)) * $(1536)) + (((r360_z / $(18432)) % $(8)) * $(192)) + (r360_z % $(192))), dtype=tl.bfloat16)
    r360_value = r360_value.to(tl.float32)
    r360_value = tl.broadcast_to(r360_value, [$(147456)])
    tl.store($(candidate_vto_copy_135_ptr) + r360_z, r360_value)
    r361_row = tl.arange($(32))
    r361_col = tl.arange($(96))
    r361_lane = tl.arange($(192))
    for r361_group in range($(8)) {
      r361_left = tl.load($(candidate_vto_copy_134_ptr) + (((((((r361_group % $(8)) * $(6144)) + (r361_row[:, None] * $(192)) + r361_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r361_group % $(8)) * $(6144)) + (r361_row[:, None] * $(192)) + r361_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r361_group % $(8)) * $(6144)) + (r361_row[:, None] * $(192)) + r361_lane[None, :]) % $(192))), dtype=tl.float32)
      r361_right = tl.load($(candidate_vto_copy_135_ptr) + (((((((r361_group % $(8)) * $(18432)) + (r361_lane[:, None] * $(96)) + r361_col[None, :]) / $(18432)) % $(8)) * $(18432)) + (((((r361_group % $(8)) * $(18432)) + (r361_lane[:, None] * $(96)) + r361_col[None, :]) % $(96)) * $(192)) + (((((r361_group % $(8)) * $(18432)) + (r361_lane[:, None] * $(96)) + r361_col[None, :]) / $(96)) % $(192))), dtype=tl.float32)
      r361_value = tl.dot(r361_left, r361_right)
      r361_value = r361_value.to(tl.float32)
      tl.store($(candidate_vbmm_107_ptr) + ((r361_group * $(3072)) + (r361_row[:, None] * $(96)) + r361_col[None, :]), r361_value)
    }
    r362_z = tl.arange($(24576))
    r362_value = (tl.load($(candidate_vbmm_107_ptr) + ((((r362_z / $(3072)) % $(8)) * $(3072)) + (((r362_z / $(96)) % $(32)) * $(96)) + (r362_z % $(96))), dtype=tl.float32) * 0.07216878364870322)
    r362_value = r362_value.to(tl.float32)
    r362_value = tl.broadcast_to(r362_value, [$(24576)])
    tl.store($(candidate_vmul_66_ptr) + r362_z, r362_value)
    r363_z = tl.arange($(32))
    r363_value = (tl.cast(r363_z, tl.int64) + tl.full([], $(96), dtype=tl.int64))
    r363_value = tl.broadcast_to(r363_value, [$(32)])
    tl.store($(candidate_vadd_47_ptr) + r363_z, r363_value)
    r364_z = tl.arange($(32))
    r364_value = (tl.load($(candidate_vadd_47_ptr) + r364_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r364_value = tl.broadcast_to(r364_value, [$(32)])
    tl.store($(candidate_vsub_48_ptr) + r364_z, r364_value)
    r365_z = tl.arange($(3072))
    r365_value = (tl.cast((r365_z % $(96)), tl.int64) > tl.load($(candidate_vsub_48_ptr) + ((r365_z / $(96)) % $(32)), dtype=tl.int64))
    r365_value = tl.broadcast_to(r365_value, [$(3072)])
    tl.store($(candidate_vgt_21_ptr) + r365_z, r365_value)
    r366_z = tl.arange($(24576))
    r366_value = tl.where(tl.load($(candidate_vgt_21_ptr) + ((((r366_z / $(96)) % $(32)) * $(96)) + (r366_z % $(96))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_66_ptr) + ((((r366_z / $(3072)) % $(8)) * $(3072)) + (((r366_z / $(96)) % $(32)) * $(96)) + (r366_z % $(96))), dtype=tl.float32))
    r366_value = r366_value.to(tl.float32)
    r366_value = tl.broadcast_to(r366_value, [$(24576)])
    tl.store($(candidate_vmasked_fill_21_ptr) + r366_z, r366_value)
    r367_z = tl.arange($(24576))
    r367_lane = tl.arange($(96))
    r367_row_values = tl.load($(candidate_vmasked_fill_21_ptr) + ((((r367_z[:, None] / $(3072)) % $(8)) * $(3072)) + (((r367_z[:, None] / $(96)) % $(32)) * $(96)) + (r367_lane[None, :] % $(96))), dtype=tl.float32)
    r367_maximum = tl.max(r367_row_values, axis=1, keep_dims=true)
    r367_weights = tl.exp(r367_row_values - r367_maximum)
    r367_total = tl.sum(r367_weights, axis=1)
    r367_maximum = tl.max(r367_row_values, axis=1)
    r367_value = tl.exp(tl.load($(candidate_vmasked_fill_21_ptr) + ((((r367_z / $(3072)) % $(8)) * $(3072)) + (((r367_z / $(96)) % $(32)) * $(96)) + (r367_z % $(96))), dtype=tl.float32) - r367_maximum) / r367_total
    r367_value = r367_value.to(tl.float32)
    r367_value = tl.broadcast_to(r367_value, [$(24576)])
    tl.store($(candidate_vsoftmax_15_ptr) + r367_z, r367_value)
    r368_z = tl.arange($(24576))
    r368_value = tl.load($(candidate_vsoftmax_15_ptr) + ((((r368_z / $(3072)) % $(8)) * $(3072)) + (((r368_z / $(96)) % $(32)) * $(96)) + (r368_z % $(96))), dtype=tl.float32)
    r368_value = r368_value.to(tl.bfloat16)
    r368_value = tl.broadcast_to(r368_value, [$(24576)])
    tl.store($(candidate_vto_copy_136_ptr) + r368_z, r368_value)
    r369_row = tl.arange($(32))
    r369_col = tl.arange($(128))
    r369_lane = tl.arange($(96))
    for r369_group in range($(8)) {
      r369_left = tl.load($(candidate_vto_copy_136_ptr) + (((((((r369_group % $(8)) * $(3072)) + (r369_row[:, None] * $(96)) + r369_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + ((((((r369_group % $(8)) * $(3072)) + (r369_row[:, None] * $(96)) + r369_lane[None, :]) / $(96)) % $(32)) * $(96)) + ((((r369_group % $(8)) * $(3072)) + (r369_row[:, None] * $(96)) + r369_lane[None, :]) % $(96))), dtype=tl.bfloat16)
      r369_right = tl.load($(candidate_vbmm_106_ptr) + (((((((r369_group % $(8)) * $(12288)) + (r369_lane[:, None] * $(128)) + r369_col[None, :]) / $(128)) % $(96)) * $(1024)) + ((((((r369_group % $(8)) * $(12288)) + (r369_lane[:, None] * $(128)) + r369_col[None, :]) / $(12288)) % $(8)) * $(128)) + ((((r369_group % $(8)) * $(12288)) + (r369_lane[:, None] * $(128)) + r369_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r369_value = tl.dot(r369_left, r369_right)
      r369_value = r369_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_108_ptr) + ((r369_group * $(4096)) + (r369_row[:, None] * $(128)) + r369_col[None, :]), r369_value)
    }
    r370_z = tl.arange($(256))
    r370_lane = tl.arange($(96))
    r370_row_values = tl.load($(candidate_vmasked_fill_21_ptr) + ((((r370_z[:, None] / $(32)) % $(8)) * $(3072)) + ((r370_z[:, None] % $(32)) * $(96)) + (r370_lane[None, :] % $(96))), dtype=tl.float32)
    r370_maximum = tl.max(r370_row_values, axis=1, keep_dims=true)
    r370_weights = tl.exp(r370_row_values - r370_maximum)
    r370_total = tl.sum(r370_weights, axis=1)
    r370_maximum = tl.max(r370_row_values, axis=1)
    r370_value = r370_maximum + tl.log(r370_total)
    r370_value = r370_value.to(tl.float32)
    r370_value = tl.broadcast_to(r370_value, [$(256)])
    tl.store($(candidate_vlogsumexp_15_ptr) + r370_z, r370_value)
    r371_z = tl.arange($(32768))
    r371_value = tl.load($(candidate_vbmm_108_ptr) + ((((((((r371_z / $(128)) % $(8)) * $(4096)) + ((((r371_z / $(1024)) % $(32)) - $(0)) * $(128)) + (r371_z % $(128))) / $(4096)) % $(8)) * $(4096)) + ((((((r371_z / $(128)) % $(8)) * $(32)) + (((r371_z / $(1024)) % $(32)) - $(0))) % $(32)) * $(128)) + (r371_z % $(128))), dtype=tl.bfloat16)
    r371_value = r371_value.to(tl.bfloat16)
    r371_value = tl.broadcast_to(r371_value, [$(32768)])
    tl.store($(candidate_vcat_76_ptr) + r371_z, r371_value)
    r372_z = tl.arange($(256))
    r372_value = tl.load($(candidate_vlogsumexp_15_ptr) + (((r372_z % $(8)) * $(32)) + ((((r372_z / $(8)) % $(32)) - $(0)) % $(32))), dtype=tl.float32)
    r372_value = r372_value.to(tl.float32)
    r372_value = tl.broadcast_to(r372_value, [$(256)])
    tl.store($(candidate_vcat_77_ptr) + r372_z, r372_value)
    r373_z = tl.arange($(49152))
    r373_value = tl.load($(candidate_vcat_74_ptr) + ((((r373_z / $(192)) % $(32)) * $(1536)) + (((r373_z / $(6144)) % $(8)) * $(192)) + (r373_z % $(192))), dtype=tl.bfloat16)
    r373_value = r373_value.to(tl.float32)
    r373_value = tl.broadcast_to(r373_value, [$(49152)])
    tl.store($(candidate_vto_copy_137_ptr) + r373_z, r373_value)
    r374_z = tl.arange($(147456))
    r374_value = tl.load($(candidate_vcat_75_ptr) + ((((r374_z / $(192)) % $(96)) * $(1536)) + (((r374_z / $(18432)) % $(8)) * $(192)) + (r374_z % $(192))), dtype=tl.bfloat16)
    r374_value = r374_value.to(tl.float32)
    r374_value = tl.broadcast_to(r374_value, [$(147456)])
    tl.store($(candidate_vto_copy_138_ptr) + r374_z, r374_value)
    r375_z = tl.arange($(98304))
    r375_value = tl.load($(candidate_vbmm_106_ptr) + ((((r375_z / $(128)) % $(96)) * $(1024)) + (((r375_z / $(12288)) % $(8)) * $(128)) + (r375_z % $(128))), dtype=tl.bfloat16)
    r375_value = r375_value.to(tl.float32)
    r375_value = tl.broadcast_to(r375_value, [$(98304)])
    tl.store($(candidate_vto_copy_139_ptr) + r375_z, r375_value)
    r376_z = tl.arange($(32768))
    r376_value = tl.load($(candidate_vmm_22_ptr) + ((((r376_z / $(128)) % $(32)) * $(1024)) + (((r376_z / $(4096)) % $(8)) * $(128)) + (r376_z % $(128)) + $(65536)), dtype=tl.bfloat16)
    r376_value = r376_value.to(tl.float32)
    r376_value = tl.broadcast_to(r376_value, [$(32768)])
    tl.store($(candidate_vto_copy_140_ptr) + r376_z, r376_value)
    r377_z = tl.arange($(32768))
    r377_value = tl.load($(candidate_vcat_76_ptr) + ((((r377_z / $(128)) % $(32)) * $(1024)) + (((r377_z / $(4096)) % $(8)) * $(128)) + (r377_z % $(128))), dtype=tl.bfloat16)
    r377_value = r377_value.to(tl.float32)
    r377_value = tl.broadcast_to(r377_value, [$(32768)])
    tl.store($(candidate_vto_copy_141_ptr) + r377_z, r377_value)
    r378_row = tl.arange($(32))
    r378_col = tl.arange($(96))
    r378_lane = tl.arange($(192))
    for r378_group in range($(8)) {
      r378_left = tl.load($(candidate_vto_copy_137_ptr) + (((((((r378_group % $(8)) * $(6144)) + (r378_row[:, None] * $(192)) + r378_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r378_group % $(8)) * $(6144)) + (r378_row[:, None] * $(192)) + r378_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r378_group % $(8)) * $(6144)) + (r378_row[:, None] * $(192)) + r378_lane[None, :]) % $(192))), dtype=tl.float32)
      r378_right = tl.load($(candidate_vto_copy_138_ptr) + (((((((r378_group % $(8)) * $(18432)) + (r378_lane[:, None] * $(96)) + r378_col[None, :]) / $(18432)) % $(8)) * $(18432)) + (((((r378_group % $(8)) * $(18432)) + (r378_lane[:, None] * $(96)) + r378_col[None, :]) % $(96)) * $(192)) + (((((r378_group % $(8)) * $(18432)) + (r378_lane[:, None] * $(96)) + r378_col[None, :]) / $(96)) % $(192))), dtype=tl.float32)
      r378_value = tl.dot(r378_left, r378_right)
      r378_value = r378_value.to(tl.float32)
      tl.store($(candidate_vbmm_109_ptr) + ((r378_group * $(3072)) + (r378_row[:, None] * $(96)) + r378_col[None, :]), r378_value)
    }
    r379_z = tl.arange($(24576))
    r379_value = (tl.load($(candidate_vbmm_109_ptr) + ((((r379_z / $(3072)) % $(8)) * $(3072)) + (((r379_z / $(96)) % $(32)) * $(96)) + (r379_z % $(96))), dtype=tl.float32) * 0.07216878364870322)
    r379_value = r379_value.to(tl.float32)
    r379_value = tl.broadcast_to(r379_value, [$(24576)])
    tl.store($(candidate_vmul_67_ptr) + r379_z, r379_value)
    r380_z = tl.arange($(32))
    r380_value = (tl.cast(r380_z, tl.int64) + tl.full([], $(96), dtype=tl.int64))
    r380_value = tl.broadcast_to(r380_value, [$(32)])
    tl.store($(candidate_vadd_48_ptr) + r380_z, r380_value)
    r381_z = tl.arange($(32))
    r381_value = (tl.load($(candidate_vadd_48_ptr) + r381_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r381_value = tl.broadcast_to(r381_value, [$(32)])
    tl.store($(candidate_vsub_49_ptr) + r381_z, r381_value)
    r382_z = tl.arange($(3072))
    r382_value = (tl.cast((r382_z % $(96)), tl.int64) > tl.load($(candidate_vsub_49_ptr) + ((r382_z / $(96)) % $(32)), dtype=tl.int64))
    r382_value = tl.broadcast_to(r382_value, [$(3072)])
    tl.store($(candidate_vgt_22_ptr) + r382_z, r382_value)
    r383_z = tl.arange($(24576))
    r383_value = tl.where(tl.load($(candidate_vgt_22_ptr) + ((((r383_z / $(96)) % $(32)) * $(96)) + (r383_z % $(96))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_67_ptr) + ((((r383_z / $(3072)) % $(8)) * $(3072)) + (((r383_z / $(96)) % $(32)) * $(96)) + (r383_z % $(96))), dtype=tl.float32))
    r383_value = r383_value.to(tl.float32)
    r383_value = tl.broadcast_to(r383_value, [$(24576)])
    tl.store($(candidate_vmasked_fill_22_ptr) + r383_z, r383_value)
    r384_z = tl.arange($(24576))
    r384_value = (tl.load($(candidate_vmasked_fill_22_ptr) + ((((r384_z / $(3072)) % $(8)) * $(3072)) + (((r384_z / $(96)) % $(32)) * $(96)) + (r384_z % $(96))), dtype=tl.float32) - tl.load($(candidate_vcat_77_ptr) + ((((r384_z / $(96)) % $(32)) * $(8)) + ((r384_z / $(3072)) % $(8))), dtype=tl.float32))
    r384_value = r384_value.to(tl.float32)
    r384_value = tl.broadcast_to(r384_value, [$(24576)])
    tl.store($(candidate_vsub_50_ptr) + r384_z, r384_value)
    r385_z = tl.arange($(24576))
    r385_value = tl.exp(tl.load($(candidate_vsub_50_ptr) + ((((r385_z / $(3072)) % $(8)) * $(3072)) + (((r385_z / $(96)) % $(32)) * $(96)) + (r385_z % $(96))), dtype=tl.float32))
    r385_value = r385_value.to(tl.float32)
    r385_value = tl.broadcast_to(r385_value, [$(24576)])
    tl.store($(candidate_vexp_6_ptr) + r385_z, r385_value)
    r386_z = tl.arange($(32768))
    r386_value = (tl.load($(candidate_vto_copy_140_ptr) + ((((r386_z / $(4096)) % $(8)) * $(4096)) + (((r386_z / $(128)) % $(32)) * $(128)) + (r386_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_141_ptr) + ((((r386_z / $(4096)) % $(8)) * $(4096)) + (((r386_z / $(128)) % $(32)) * $(128)) + (r386_z % $(128))), dtype=tl.float32))
    r386_value = r386_value.to(tl.float32)
    r386_value = tl.broadcast_to(r386_value, [$(32768)])
    tl.store($(candidate_vmul_68_ptr) + r386_z, r386_value)
    r387_z = tl.arange($(256))
    r387_lane = tl.arange($(128))
    r387_value = tl.sum(tl.load($(candidate_vmul_68_ptr) + ((((r387_z[:, None] / $(32)) % $(8)) * $(4096)) + ((r387_z[:, None] % $(32)) * $(128)) + r387_lane[None, :]), dtype=tl.float32), axis=1)
    r387_value = r387_value.to(tl.float32)
    r387_value = tl.broadcast_to(r387_value, [$(256)])
    tl.store($(candidate_vsum_13_ptr) + r387_z, r387_value)
    r388_row = tl.arange($(32))
    r388_col = tl.arange($(96))
    r388_lane = tl.arange($(128))
    for r388_group in range($(8)) {
      r388_left = tl.load($(candidate_vto_copy_140_ptr) + (((((((r388_group % $(8)) * $(4096)) + (r388_row[:, None] * $(128)) + r388_lane[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r388_group % $(8)) * $(4096)) + (r388_row[:, None] * $(128)) + r388_lane[None, :]) / $(128)) % $(32)) * $(128)) + ((((r388_group % $(8)) * $(4096)) + (r388_row[:, None] * $(128)) + r388_lane[None, :]) % $(128))), dtype=tl.float32)
      r388_right = tl.load($(candidate_vto_copy_139_ptr) + (((((((r388_group % $(8)) * $(12288)) + (r388_lane[:, None] * $(96)) + r388_col[None, :]) / $(12288)) % $(8)) * $(12288)) + (((((r388_group % $(8)) * $(12288)) + (r388_lane[:, None] * $(96)) + r388_col[None, :]) % $(96)) * $(128)) + (((((r388_group % $(8)) * $(12288)) + (r388_lane[:, None] * $(96)) + r388_col[None, :]) / $(96)) % $(128))), dtype=tl.float32)
      r388_value = tl.dot(r388_left, r388_right)
      r388_value = r388_value.to(tl.float32)
      tl.store($(candidate_vbmm_110_ptr) + ((r388_group * $(3072)) + (r388_row[:, None] * $(96)) + r388_col[None, :]), r388_value)
    }
    r389_z = tl.arange($(24576))
    r389_value = (tl.load($(candidate_vbmm_110_ptr) + ((((r389_z / $(3072)) % $(8)) * $(3072)) + (((r389_z / $(96)) % $(32)) * $(96)) + (r389_z % $(96))), dtype=tl.float32) - tl.load($(candidate_vsum_13_ptr) + ((((r389_z / $(3072)) % $(8)) * $(32)) + ((r389_z / $(96)) % $(32))), dtype=tl.float32))
    r389_value = r389_value.to(tl.float32)
    r389_value = tl.broadcast_to(r389_value, [$(24576)])
    tl.store($(candidate_vsub_51_ptr) + r389_z, r389_value)
    r390_z = tl.arange($(24576))
    r390_value = (tl.load($(candidate_vexp_6_ptr) + ((((r390_z / $(3072)) % $(8)) * $(3072)) + (((r390_z / $(96)) % $(32)) * $(96)) + (r390_z % $(96))), dtype=tl.float32) * tl.load($(candidate_vsub_51_ptr) + ((((r390_z / $(3072)) % $(8)) * $(3072)) + (((r390_z / $(96)) % $(32)) * $(96)) + (r390_z % $(96))), dtype=tl.float32))
    r390_value = r390_value.to(tl.float32)
    r390_value = tl.broadcast_to(r390_value, [$(24576)])
    tl.store($(candidate_vmul_69_ptr) + r390_z, r390_value)
    r391_row = tl.arange($(96))
    r391_col = tl.arange($(192))
    r391_lane = tl.arange($(32))
    for r391_group in range($(8)) {
      r391_left = tl.load($(candidate_vmul_69_ptr) + (((((((r391_group % $(8)) * $(3072)) + (r391_row[:, None] * $(32)) + r391_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + (((((r391_group % $(8)) * $(3072)) + (r391_row[:, None] * $(32)) + r391_lane[None, :]) % $(32)) * $(96)) + (((((r391_group % $(8)) * $(3072)) + (r391_row[:, None] * $(32)) + r391_lane[None, :]) / $(32)) % $(96))), dtype=tl.float32)
      r391_right = tl.load($(candidate_vto_copy_137_ptr) + (((((((r391_group % $(8)) * $(6144)) + (r391_lane[:, None] * $(192)) + r391_col[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r391_group % $(8)) * $(6144)) + (r391_lane[:, None] * $(192)) + r391_col[None, :]) / $(192)) % $(32)) * $(192)) + ((((r391_group % $(8)) * $(6144)) + (r391_lane[:, None] * $(192)) + r391_col[None, :]) % $(192))), dtype=tl.float32)
      r391_value = tl.dot(r391_left, r391_right)
      r391_value = r391_value.to(tl.float32)
      tl.store($(candidate_vbmm_112_ptr) + ((r391_group * $(18432)) + (r391_row[:, None] * $(192)) + r391_col[None, :]), r391_value)
    }
    r392_z = tl.arange($(147456))
    r392_value = (tl.load($(candidate_vbmm_112_ptr) + ((((r392_z / $(18432)) % $(8)) * $(18432)) + (((r392_z / $(192)) % $(96)) * $(192)) + (r392_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r392_value = r392_value.to(tl.float32)
    r392_value = tl.broadcast_to(r392_value, [$(147456)])
    tl.store($(candidate_vmul_71_ptr) + r392_z, r392_value)
    r393_z = tl.arange($(147456))
    r393_value = tl.load($(candidate_vmul_71_ptr) + ((((r393_z / $(192)) % $(8)) * $(18432)) + (((r393_z / $(1536)) % $(96)) * $(192)) + (r393_z % $(192))), dtype=tl.float32)
    r393_value = r393_value.to(tl.bfloat16)
    r393_value = tl.broadcast_to(r393_value, [$(147456)])
    tl.store($(candidate_vto_copy_143_ptr) + r393_z, r393_value)
    r394_row = tl.arange($(96))
    r394_col = tl.arange($(128))
    r394_lane = tl.arange($(32))
    for r394_group in range($(8)) {
      r394_left = tl.load($(candidate_vexp_6_ptr) + (((((((r394_group % $(8)) * $(3072)) + (r394_row[:, None] * $(32)) + r394_lane[None, :]) / $(3072)) % $(8)) * $(3072)) + (((((r394_group % $(8)) * $(3072)) + (r394_row[:, None] * $(32)) + r394_lane[None, :]) % $(32)) * $(96)) + (((((r394_group % $(8)) * $(3072)) + (r394_row[:, None] * $(32)) + r394_lane[None, :]) / $(32)) % $(96))), dtype=tl.float32)
      r394_right = tl.load($(candidate_vto_copy_140_ptr) + (((((((r394_group % $(8)) * $(4096)) + (r394_lane[:, None] * $(128)) + r394_col[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r394_group % $(8)) * $(4096)) + (r394_lane[:, None] * $(128)) + r394_col[None, :]) / $(128)) % $(32)) * $(128)) + ((((r394_group % $(8)) * $(4096)) + (r394_lane[:, None] * $(128)) + r394_col[None, :]) % $(128))), dtype=tl.float32)
      r394_value = tl.dot(r394_left, r394_right)
      r394_value = r394_value.to(tl.float32)
      tl.store($(candidate_vbmm_113_ptr) + ((r394_group * $(12288)) + (r394_row[:, None] * $(128)) + r394_col[None, :]), r394_value)
    }
    r395_z = tl.arange($(98304))
    r395_value = tl.load($(candidate_vbmm_113_ptr) + ((((r395_z / $(128)) % $(8)) * $(12288)) + (((r395_z / $(1024)) % $(96)) * $(128)) + (r395_z % $(128))), dtype=tl.float32)
    r395_value = r395_value.to(tl.bfloat16)
    r395_value = tl.broadcast_to(r395_value, [$(98304)])
    tl.store($(candidate_vto_copy_144_ptr) + r395_z, r395_value)
    r396_z = tl.arange($(147456))
    r396_value = tl.load($(candidate_vto_copy_143_ptr) + ((((((r396_z / $(1536)) % $(96)) - $(0)) % $(96)) * $(1536)) + (((r396_z / $(192)) % $(8)) * $(192)) + (r396_z % $(192))), dtype=tl.bfloat16)
    r396_value = r396_value.to(tl.bfloat16)
    r396_value = tl.broadcast_to(r396_value, [$(147456)])
    tl.store($(candidate_vcat_79_ptr) + r396_z, r396_value)
    r397_z = tl.arange($(98304))
    r397_value = tl.load($(candidate_vto_copy_144_ptr) + ((((((r397_z / $(1024)) % $(96)) - $(0)) % $(96)) * $(1024)) + (((r397_z / $(128)) % $(8)) * $(128)) + (r397_z % $(128))), dtype=tl.bfloat16)
    r397_value = r397_value.to(tl.bfloat16)
    r397_value = tl.broadcast_to(r397_value, [$(98304)])
    tl.store($(candidate_vcat_80_ptr) + r397_z, r397_value)
    r398_row = tl.arange($(1024))
    r398_col = tl.arange($(512))
    r398_lane = tl.arange($(96))
    for r398_group in range($(1)) {
      r398_left = tl.load($(candidate_vcat_79_ptr) + (((((r398_row[:, None] * $(96)) + r398_lane[None, :]) % $(96)) * $(1536)) + (((((r398_row[:, None] * $(96)) + r398_lane[None, :]) / $(12288)) % $(8)) * $(192)) + ((((r398_row[:, None] * $(96)) + r398_lane[None, :]) / $(96)) % $(128))), dtype=tl.bfloat16)
      r398_right = tl.load($(candidate_vcat_7_ptr) + ((((((r398_lane[:, None] * $(512)) + r398_col[None, :]) / $(512)) % $(96)) * $(576)) + (((r398_lane[:, None] * $(512)) + r398_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r398_value = tl.dot(r398_left, r398_right)
      r398_value = r398_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_116_ptr) + ((r398_group * $(524288)) + (r398_row[:, None] * $(512)) + r398_col[None, :]), r398_value)
    }
    r399_row = tl.arange($(1024))
    r399_col = tl.arange($(512))
    r399_lane = tl.arange($(96))
    for r399_group in range($(1)) {
      r399_left = tl.load($(candidate_vcat_80_ptr) + (((((r399_row[:, None] * $(96)) + r399_lane[None, :]) % $(96)) * $(1024)) + (((((r399_row[:, None] * $(96)) + r399_lane[None, :]) / $(12288)) % $(8)) * $(128)) + ((((r399_row[:, None] * $(96)) + r399_lane[None, :]) / $(96)) % $(128))), dtype=tl.bfloat16)
      r399_right = tl.load($(candidate_vcat_7_ptr) + ((((((r399_lane[:, None] * $(512)) + r399_col[None, :]) / $(512)) % $(96)) * $(576)) + (((r399_lane[:, None] * $(512)) + r399_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r399_value = tl.dot(r399_left, r399_right)
      r399_value = r399_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_117_ptr) + ((r399_group * $(524288)) + (r399_row[:, None] * $(512)) + r399_col[None, :]), r399_value)
    }
    r400_z = tl.arange($(1048576))
    r400_value = tl.where(((r400_z / $(512)) % $(256)) < $(128), tl.load($(candidate_vbmm_116_ptr) + ((((((((r400_z / $(131072)) % $(8)) * $(65536)) + ((((r400_z / $(512)) % $(256)) - $(0)) * $(512)) + (r400_z % $(512))) / $(65536)) % $(8)) * $(65536)) + ((((((r400_z / $(131072)) % $(8)) * $(128)) + (((r400_z / $(512)) % $(256)) - $(0))) % $(128)) * $(512)) + (r400_z % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vbmm_117_ptr) + ((((((((r400_z / $(131072)) % $(8)) * $(65536)) + ((((r400_z / $(512)) % $(256)) - $(128)) * $(512)) + (r400_z % $(512))) / $(65536)) % $(8)) * $(65536)) + ((((((r400_z / $(131072)) % $(8)) * $(128)) + (((r400_z / $(512)) % $(256)) - $(128))) % $(128)) * $(512)) + (r400_z % $(512))), dtype=tl.bfloat16))
    r400_value = r400_value.to(tl.bfloat16)
    r400_value = tl.broadcast_to(r400_value, [$(1048576)])
    tl.store($(candidate_vcat_81_ptr) + r400_z, r400_value)
    r401_z = tl.arange($(1048576))
    r401_value = (tl.load($(candidate_vadd_46_ptr) + ((((r401_z / $(512)) % $(2048)) * $(512)) + (r401_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vcat_81_ptr) + ((((((((r401_z / $(512)) % $(2048)) * $(512)) + (r401_z % $(512))) / $(131072)) % $(8)) * $(131072)) + ((((r401_z / $(512)) % $(2048)) % $(256)) * $(512)) + (r401_z % $(512))), dtype=tl.bfloat16))
    r401_value = r401_value.to(tl.bfloat16)
    r401_value = tl.broadcast_to(r401_value, [$(1048576)])
    tl.store($(candidate_vadd_50_ptr) + r401_z, r401_value)
    r402_row = tl.arange($(96))
    r402_col = tl.arange($(1024))
    r402_lane = tl.arange($(512))
    for r402_group in range($(1)) {
      r402_left = tl.load($(candidate_vcat_7_ptr) + ((((((r402_row[:, None] * $(512)) + r402_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r402_row[:, None] * $(512)) + r402_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r402_right = tl.load($(candidate_vto_copy_18_ptr) + ((((((r402_lane[:, None] * $(1024)) + r402_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r402_lane[:, None] * $(1024)) + r402_col[None, :]) % $(128)) * $(512)) + ((((r402_lane[:, None] * $(1024)) + r402_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r402_value = tl.dot(r402_left, r402_right)
      r402_value = r402_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_118_ptr) + ((r402_group * $(98304)) + (r402_row[:, None] * $(1024)) + r402_col[None, :]), r402_value)
    }
    r403_row = tl.arange($(96))
    r403_col = tl.arange($(1024))
    r403_lane = tl.arange($(512))
    for r403_group in range($(1)) {
      r403_left = tl.load($(candidate_vcat_7_ptr) + ((((((r403_row[:, None] * $(512)) + r403_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r403_row[:, None] * $(512)) + r403_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r403_right = tl.load($(candidate_vto_copy_18_ptr) + ((((((r403_lane[:, None] * $(1024)) + r403_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r403_lane[:, None] * $(1024)) + r403_col[None, :]) % $(128)) * $(512)) + ((((r403_lane[:, None] * $(1024)) + r403_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r403_value = tl.dot(r403_left, r403_right)
      r403_value = r403_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_119_ptr) + ((r403_group * $(98304)) + (r403_row[:, None] * $(1024)) + r403_col[None, :]), r403_value)
    }
    r404_z = tl.arange($(49152))
    r404_value = tl.where((r404_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r404_z / $(1536)) % $(32)) * $(1024)) + (((r404_z / $(192)) % $(8)) * $(128)) + ((r404_z % $(192)) - $(0)) + $(32768)) / $(1024)) % $(288)) * $(1536)) + (((((((r404_z / $(1536)) % $(32)) * $(1024)) + (((r404_z / $(192)) % $(8)) * $(128)) + ((r404_z % $(192)) - $(0)) + $(32768)) / $(128)) % $(8)) * $(192)) + (((((r404_z / $(1536)) % $(32)) * $(1024)) + (((r404_z / $(192)) % $(8)) * $(128)) + ((r404_z % $(192)) - $(0)) + $(32768)) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_5_ptr) + ((((((((r404_z / $(1536)) % $(32)) * $(512)) + (((r404_z / $(192)) % $(8)) * $(64)) + ((r404_z % $(192)) - $(128)) + $(16384)) / $(64)) % $(8)) * $(18432)) + (((((((r404_z / $(1536)) % $(32)) * $(512)) + (((r404_z / $(192)) % $(8)) * $(64)) + ((r404_z % $(192)) - $(128)) + $(16384)) / $(512)) % $(288)) * $(64)) + (((((r404_z / $(1536)) % $(32)) * $(512)) + (((r404_z / $(192)) % $(8)) * $(64)) + ((r404_z % $(192)) - $(128)) + $(16384)) % $(64))), dtype=tl.bfloat16))
    r404_value = r404_value.to(tl.bfloat16)
    r404_value = tl.broadcast_to(r404_value, [$(49152)])
    tl.store($(candidate_vcat_82_ptr) + r404_z, r404_value)
    r405_z = tl.arange($(147456))
    r405_value = tl.where((r405_z % $(192)) < $(128), tl.load($(candidate_vbmm_118_ptr) + ((((((((r405_z / $(1536)) % $(96)) * $(1024)) + (((r405_z / $(192)) % $(8)) * $(128)) + ((r405_z % $(192)) - $(0))) / $(1024)) % $(96)) * $(1024)) + (((((((r405_z / $(1536)) % $(96)) * $(1024)) + (((r405_z / $(192)) % $(8)) * $(128)) + ((r405_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r405_z / $(1536)) % $(96)) * $(1024)) + (((r405_z / $(192)) % $(8)) * $(128)) + ((r405_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_7_ptr) + ((((r405_z / $(1536)) % $(96)) * $(576)) + ((((r405_z % $(192)) - $(128)) + $(512)) % $(576))), dtype=tl.bfloat16))
    r405_value = r405_value.to(tl.bfloat16)
    r405_value = tl.broadcast_to(r405_value, [$(147456)])
    tl.store($(candidate_vcat_83_ptr) + r405_z, r405_value)
    r406_z = tl.arange($(49152))
    r406_value = tl.load($(candidate_vcat_82_ptr) + ((((r406_z / $(192)) % $(32)) * $(1536)) + (((r406_z / $(6144)) % $(8)) * $(192)) + (r406_z % $(192))), dtype=tl.bfloat16)
    r406_value = r406_value.to(tl.float32)
    r406_value = tl.broadcast_to(r406_value, [$(49152)])
    tl.store($(candidate_vto_copy_145_ptr) + r406_z, r406_value)
    r407_z = tl.arange($(98304))
    r407_value = tl.load($(candidate_vcat_83_ptr) + ((((r407_z / $(192)) % $(64)) * $(1536)) + (((r407_z / $(12288)) % $(8)) * $(192)) + (r407_z % $(192))), dtype=tl.bfloat16)
    r407_value = r407_value.to(tl.float32)
    r407_value = tl.broadcast_to(r407_value, [$(98304)])
    tl.store($(candidate_vto_copy_146_ptr) + r407_z, r407_value)
    r408_row = tl.arange($(32))
    r408_col = tl.arange($(64))
    r408_lane = tl.arange($(192))
    for r408_group in range($(8)) {
      r408_left = tl.load($(candidate_vto_copy_145_ptr) + (((((((r408_group % $(8)) * $(6144)) + (r408_row[:, None] * $(192)) + r408_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r408_group % $(8)) * $(6144)) + (r408_row[:, None] * $(192)) + r408_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r408_group % $(8)) * $(6144)) + (r408_row[:, None] * $(192)) + r408_lane[None, :]) % $(192))), dtype=tl.float32)
      r408_right = tl.load($(candidate_vto_copy_146_ptr) + (((((((r408_group % $(8)) * $(12288)) + (r408_lane[:, None] * $(64)) + r408_col[None, :]) / $(12288)) % $(8)) * $(12288)) + (((((r408_group % $(8)) * $(12288)) + (r408_lane[:, None] * $(64)) + r408_col[None, :]) % $(64)) * $(192)) + (((((r408_group % $(8)) * $(12288)) + (r408_lane[:, None] * $(64)) + r408_col[None, :]) / $(64)) % $(192))), dtype=tl.float32)
      r408_value = tl.dot(r408_left, r408_right)
      r408_value = r408_value.to(tl.float32)
      tl.store($(candidate_vbmm_120_ptr) + ((r408_group * $(2048)) + (r408_row[:, None] * $(64)) + r408_col[None, :]), r408_value)
    }
    r409_z = tl.arange($(16384))
    r409_value = (tl.load($(candidate_vbmm_120_ptr) + ((((r409_z / $(2048)) % $(8)) * $(2048)) + (((r409_z / $(64)) % $(32)) * $(64)) + (r409_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r409_value = r409_value.to(tl.float32)
    r409_value = tl.broadcast_to(r409_value, [$(16384)])
    tl.store($(candidate_vmul_72_ptr) + r409_z, r409_value)
    r410_z = tl.arange($(32))
    r410_value = (tl.cast(r410_z, tl.int64) + tl.full([], $(64), dtype=tl.int64))
    r410_value = tl.broadcast_to(r410_value, [$(32)])
    tl.store($(candidate_vadd_51_ptr) + r410_z, r410_value)
    r411_z = tl.arange($(32))
    r411_value = (tl.load($(candidate_vadd_51_ptr) + r411_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r411_value = tl.broadcast_to(r411_value, [$(32)])
    tl.store($(candidate_vsub_52_ptr) + r411_z, r411_value)
    r412_z = tl.arange($(2048))
    r412_value = (tl.cast((r412_z % $(64)), tl.int64) > tl.load($(candidate_vsub_52_ptr) + ((r412_z / $(64)) % $(32)), dtype=tl.int64))
    r412_value = tl.broadcast_to(r412_value, [$(2048)])
    tl.store($(candidate_vgt_23_ptr) + r412_z, r412_value)
    r413_z = tl.arange($(16384))
    r413_value = tl.where(tl.load($(candidate_vgt_23_ptr) + ((((r413_z / $(64)) % $(32)) * $(64)) + (r413_z % $(64))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_72_ptr) + ((((r413_z / $(2048)) % $(8)) * $(2048)) + (((r413_z / $(64)) % $(32)) * $(64)) + (r413_z % $(64))), dtype=tl.float32))
    r413_value = r413_value.to(tl.float32)
    r413_value = tl.broadcast_to(r413_value, [$(16384)])
    tl.store($(candidate_vmasked_fill_23_ptr) + r413_z, r413_value)
    r414_z = tl.arange($(16384))
    r414_lane = tl.arange($(64))
    r414_row_values = tl.load($(candidate_vmasked_fill_23_ptr) + ((((r414_z[:, None] / $(2048)) % $(8)) * $(2048)) + (((r414_z[:, None] / $(64)) % $(32)) * $(64)) + (r414_lane[None, :] % $(64))), dtype=tl.float32)
    r414_maximum = tl.max(r414_row_values, axis=1, keep_dims=true)
    r414_weights = tl.exp(r414_row_values - r414_maximum)
    r414_total = tl.sum(r414_weights, axis=1)
    r414_maximum = tl.max(r414_row_values, axis=1)
    r414_value = tl.exp(tl.load($(candidate_vmasked_fill_23_ptr) + ((((r414_z / $(2048)) % $(8)) * $(2048)) + (((r414_z / $(64)) % $(32)) * $(64)) + (r414_z % $(64))), dtype=tl.float32) - r414_maximum) / r414_total
    r414_value = r414_value.to(tl.float32)
    r414_value = tl.broadcast_to(r414_value, [$(16384)])
    tl.store($(candidate_vsoftmax_16_ptr) + r414_z, r414_value)
    r415_z = tl.arange($(16384))
    r415_value = tl.load($(candidate_vsoftmax_16_ptr) + ((((r415_z / $(2048)) % $(8)) * $(2048)) + (((r415_z / $(64)) % $(32)) * $(64)) + (r415_z % $(64))), dtype=tl.float32)
    r415_value = r415_value.to(tl.bfloat16)
    r415_value = tl.broadcast_to(r415_value, [$(16384)])
    tl.store($(candidate_vto_copy_147_ptr) + r415_z, r415_value)
    r416_row = tl.arange($(32))
    r416_col = tl.arange($(128))
    r416_lane = tl.arange($(64))
    for r416_group in range($(8)) {
      r416_left = tl.load($(candidate_vto_copy_147_ptr) + (((((((r416_group % $(8)) * $(2048)) + (r416_row[:, None] * $(64)) + r416_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + ((((((r416_group % $(8)) * $(2048)) + (r416_row[:, None] * $(64)) + r416_lane[None, :]) / $(64)) % $(32)) * $(64)) + ((((r416_group % $(8)) * $(2048)) + (r416_row[:, None] * $(64)) + r416_lane[None, :]) % $(64))), dtype=tl.bfloat16)
      r416_right = tl.load($(candidate_vbmm_119_ptr) + (((((((r416_group % $(8)) * $(8192)) + (r416_lane[:, None] * $(128)) + r416_col[None, :]) / $(128)) % $(64)) * $(1024)) + ((((((r416_group % $(8)) * $(8192)) + (r416_lane[:, None] * $(128)) + r416_col[None, :]) / $(8192)) % $(8)) * $(128)) + ((((r416_group % $(8)) * $(8192)) + (r416_lane[:, None] * $(128)) + r416_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r416_value = tl.dot(r416_left, r416_right)
      r416_value = r416_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_121_ptr) + ((r416_group * $(4096)) + (r416_row[:, None] * $(128)) + r416_col[None, :]), r416_value)
    }
    r417_z = tl.arange($(256))
    r417_lane = tl.arange($(64))
    r417_row_values = tl.load($(candidate_vmasked_fill_23_ptr) + ((((r417_z[:, None] / $(32)) % $(8)) * $(2048)) + ((r417_z[:, None] % $(32)) * $(64)) + (r417_lane[None, :] % $(64))), dtype=tl.float32)
    r417_maximum = tl.max(r417_row_values, axis=1, keep_dims=true)
    r417_weights = tl.exp(r417_row_values - r417_maximum)
    r417_total = tl.sum(r417_weights, axis=1)
    r417_maximum = tl.max(r417_row_values, axis=1)
    r417_value = r417_maximum + tl.log(r417_total)
    r417_value = r417_value.to(tl.float32)
    r417_value = tl.broadcast_to(r417_value, [$(256)])
    tl.store($(candidate_vlogsumexp_16_ptr) + r417_z, r417_value)
    r418_z = tl.arange($(32768))
    r418_value = tl.load($(candidate_vbmm_121_ptr) + ((((((((r418_z / $(128)) % $(8)) * $(4096)) + ((((r418_z / $(1024)) % $(32)) - $(0)) * $(128)) + (r418_z % $(128))) / $(4096)) % $(8)) * $(4096)) + ((((((r418_z / $(128)) % $(8)) * $(32)) + (((r418_z / $(1024)) % $(32)) - $(0))) % $(32)) * $(128)) + (r418_z % $(128))), dtype=tl.bfloat16)
    r418_value = r418_value.to(tl.bfloat16)
    r418_value = tl.broadcast_to(r418_value, [$(32768)])
    tl.store($(candidate_vcat_84_ptr) + r418_z, r418_value)
    r419_z = tl.arange($(256))
    r419_value = tl.load($(candidate_vlogsumexp_16_ptr) + (((r419_z % $(8)) * $(32)) + ((((r419_z / $(8)) % $(32)) - $(0)) % $(32))), dtype=tl.float32)
    r419_value = r419_value.to(tl.float32)
    r419_value = tl.broadcast_to(r419_value, [$(256)])
    tl.store($(candidate_vcat_85_ptr) + r419_z, r419_value)
    r420_z = tl.arange($(49152))
    r420_value = tl.load($(candidate_vcat_82_ptr) + ((((r420_z / $(192)) % $(32)) * $(1536)) + (((r420_z / $(6144)) % $(8)) * $(192)) + (r420_z % $(192))), dtype=tl.bfloat16)
    r420_value = r420_value.to(tl.float32)
    r420_value = tl.broadcast_to(r420_value, [$(49152)])
    tl.store($(candidate_vto_copy_148_ptr) + r420_z, r420_value)
    r421_z = tl.arange($(98304))
    r421_value = tl.load($(candidate_vcat_83_ptr) + ((((r421_z / $(192)) % $(64)) * $(1536)) + (((r421_z / $(12288)) % $(8)) * $(192)) + (r421_z % $(192))), dtype=tl.bfloat16)
    r421_value = r421_value.to(tl.float32)
    r421_value = tl.broadcast_to(r421_value, [$(98304)])
    tl.store($(candidate_vto_copy_149_ptr) + r421_z, r421_value)
    r422_z = tl.arange($(65536))
    r422_value = tl.load($(candidate_vbmm_119_ptr) + ((((r422_z / $(128)) % $(64)) * $(1024)) + (((r422_z / $(8192)) % $(8)) * $(128)) + (r422_z % $(128))), dtype=tl.bfloat16)
    r422_value = r422_value.to(tl.float32)
    r422_value = tl.broadcast_to(r422_value, [$(65536)])
    tl.store($(candidate_vto_copy_150_ptr) + r422_z, r422_value)
    r423_z = tl.arange($(32768))
    r423_value = tl.load($(candidate_vmm_22_ptr) + ((((r423_z / $(128)) % $(32)) * $(1024)) + (((r423_z / $(4096)) % $(8)) * $(128)) + (r423_z % $(128)) + $(32768)), dtype=tl.bfloat16)
    r423_value = r423_value.to(tl.float32)
    r423_value = tl.broadcast_to(r423_value, [$(32768)])
    tl.store($(candidate_vto_copy_151_ptr) + r423_z, r423_value)
    r424_z = tl.arange($(32768))
    r424_value = tl.load($(candidate_vcat_84_ptr) + ((((r424_z / $(128)) % $(32)) * $(1024)) + (((r424_z / $(4096)) % $(8)) * $(128)) + (r424_z % $(128))), dtype=tl.bfloat16)
    r424_value = r424_value.to(tl.float32)
    r424_value = tl.broadcast_to(r424_value, [$(32768)])
    tl.store($(candidate_vto_copy_152_ptr) + r424_z, r424_value)
    r425_row = tl.arange($(32))
    r425_col = tl.arange($(64))
    r425_lane = tl.arange($(192))
    for r425_group in range($(8)) {
      r425_left = tl.load($(candidate_vto_copy_148_ptr) + (((((((r425_group % $(8)) * $(6144)) + (r425_row[:, None] * $(192)) + r425_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r425_group % $(8)) * $(6144)) + (r425_row[:, None] * $(192)) + r425_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r425_group % $(8)) * $(6144)) + (r425_row[:, None] * $(192)) + r425_lane[None, :]) % $(192))), dtype=tl.float32)
      r425_right = tl.load($(candidate_vto_copy_149_ptr) + (((((((r425_group % $(8)) * $(12288)) + (r425_lane[:, None] * $(64)) + r425_col[None, :]) / $(12288)) % $(8)) * $(12288)) + (((((r425_group % $(8)) * $(12288)) + (r425_lane[:, None] * $(64)) + r425_col[None, :]) % $(64)) * $(192)) + (((((r425_group % $(8)) * $(12288)) + (r425_lane[:, None] * $(64)) + r425_col[None, :]) / $(64)) % $(192))), dtype=tl.float32)
      r425_value = tl.dot(r425_left, r425_right)
      r425_value = r425_value.to(tl.float32)
      tl.store($(candidate_vbmm_122_ptr) + ((r425_group * $(2048)) + (r425_row[:, None] * $(64)) + r425_col[None, :]), r425_value)
    }
    r426_z = tl.arange($(16384))
    r426_value = (tl.load($(candidate_vbmm_122_ptr) + ((((r426_z / $(2048)) % $(8)) * $(2048)) + (((r426_z / $(64)) % $(32)) * $(64)) + (r426_z % $(64))), dtype=tl.float32) * 0.07216878364870322)
    r426_value = r426_value.to(tl.float32)
    r426_value = tl.broadcast_to(r426_value, [$(16384)])
    tl.store($(candidate_vmul_73_ptr) + r426_z, r426_value)
    r427_z = tl.arange($(32))
    r427_value = (tl.cast(r427_z, tl.int64) + tl.full([], $(64), dtype=tl.int64))
    r427_value = tl.broadcast_to(r427_value, [$(32)])
    tl.store($(candidate_vadd_52_ptr) + r427_z, r427_value)
    r428_z = tl.arange($(32))
    r428_value = (tl.load($(candidate_vadd_52_ptr) + r428_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r428_value = tl.broadcast_to(r428_value, [$(32)])
    tl.store($(candidate_vsub_53_ptr) + r428_z, r428_value)
    r429_z = tl.arange($(2048))
    r429_value = (tl.cast((r429_z % $(64)), tl.int64) > tl.load($(candidate_vsub_53_ptr) + ((r429_z / $(64)) % $(32)), dtype=tl.int64))
    r429_value = tl.broadcast_to(r429_value, [$(2048)])
    tl.store($(candidate_vgt_24_ptr) + r429_z, r429_value)
    r430_z = tl.arange($(16384))
    r430_value = tl.where(tl.load($(candidate_vgt_24_ptr) + ((((r430_z / $(64)) % $(32)) * $(64)) + (r430_z % $(64))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_73_ptr) + ((((r430_z / $(2048)) % $(8)) * $(2048)) + (((r430_z / $(64)) % $(32)) * $(64)) + (r430_z % $(64))), dtype=tl.float32))
    r430_value = r430_value.to(tl.float32)
    r430_value = tl.broadcast_to(r430_value, [$(16384)])
    tl.store($(candidate_vmasked_fill_24_ptr) + r430_z, r430_value)
    r431_z = tl.arange($(16384))
    r431_value = (tl.load($(candidate_vmasked_fill_24_ptr) + ((((r431_z / $(2048)) % $(8)) * $(2048)) + (((r431_z / $(64)) % $(32)) * $(64)) + (r431_z % $(64))), dtype=tl.float32) - tl.load($(candidate_vcat_85_ptr) + ((((r431_z / $(64)) % $(32)) * $(8)) + ((r431_z / $(2048)) % $(8))), dtype=tl.float32))
    r431_value = r431_value.to(tl.float32)
    r431_value = tl.broadcast_to(r431_value, [$(16384)])
    tl.store($(candidate_vsub_54_ptr) + r431_z, r431_value)
    r432_z = tl.arange($(16384))
    r432_value = tl.exp(tl.load($(candidate_vsub_54_ptr) + ((((r432_z / $(2048)) % $(8)) * $(2048)) + (((r432_z / $(64)) % $(32)) * $(64)) + (r432_z % $(64))), dtype=tl.float32))
    r432_value = r432_value.to(tl.float32)
    r432_value = tl.broadcast_to(r432_value, [$(16384)])
    tl.store($(candidate_vexp_7_ptr) + r432_z, r432_value)
    r433_z = tl.arange($(32768))
    r433_value = (tl.load($(candidate_vto_copy_151_ptr) + ((((r433_z / $(4096)) % $(8)) * $(4096)) + (((r433_z / $(128)) % $(32)) * $(128)) + (r433_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_152_ptr) + ((((r433_z / $(4096)) % $(8)) * $(4096)) + (((r433_z / $(128)) % $(32)) * $(128)) + (r433_z % $(128))), dtype=tl.float32))
    r433_value = r433_value.to(tl.float32)
    r433_value = tl.broadcast_to(r433_value, [$(32768)])
    tl.store($(candidate_vmul_74_ptr) + r433_z, r433_value)
    r434_z = tl.arange($(256))
    r434_lane = tl.arange($(128))
    r434_value = tl.sum(tl.load($(candidate_vmul_74_ptr) + ((((r434_z[:, None] / $(32)) % $(8)) * $(4096)) + ((r434_z[:, None] % $(32)) * $(128)) + r434_lane[None, :]), dtype=tl.float32), axis=1)
    r434_value = r434_value.to(tl.float32)
    r434_value = tl.broadcast_to(r434_value, [$(256)])
    tl.store($(candidate_vsum_15_ptr) + r434_z, r434_value)
    r435_row = tl.arange($(32))
    r435_col = tl.arange($(64))
    r435_lane = tl.arange($(128))
    for r435_group in range($(8)) {
      r435_left = tl.load($(candidate_vto_copy_151_ptr) + (((((((r435_group % $(8)) * $(4096)) + (r435_row[:, None] * $(128)) + r435_lane[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r435_group % $(8)) * $(4096)) + (r435_row[:, None] * $(128)) + r435_lane[None, :]) / $(128)) % $(32)) * $(128)) + ((((r435_group % $(8)) * $(4096)) + (r435_row[:, None] * $(128)) + r435_lane[None, :]) % $(128))), dtype=tl.float32)
      r435_right = tl.load($(candidate_vto_copy_150_ptr) + (((((((r435_group % $(8)) * $(8192)) + (r435_lane[:, None] * $(64)) + r435_col[None, :]) / $(8192)) % $(8)) * $(8192)) + (((((r435_group % $(8)) * $(8192)) + (r435_lane[:, None] * $(64)) + r435_col[None, :]) % $(64)) * $(128)) + (((((r435_group % $(8)) * $(8192)) + (r435_lane[:, None] * $(64)) + r435_col[None, :]) / $(64)) % $(128))), dtype=tl.float32)
      r435_value = tl.dot(r435_left, r435_right)
      r435_value = r435_value.to(tl.float32)
      tl.store($(candidate_vbmm_123_ptr) + ((r435_group * $(2048)) + (r435_row[:, None] * $(64)) + r435_col[None, :]), r435_value)
    }
    r436_z = tl.arange($(16384))
    r436_value = (tl.load($(candidate_vbmm_123_ptr) + ((((r436_z / $(2048)) % $(8)) * $(2048)) + (((r436_z / $(64)) % $(32)) * $(64)) + (r436_z % $(64))), dtype=tl.float32) - tl.load($(candidate_vsum_15_ptr) + ((((r436_z / $(2048)) % $(8)) * $(32)) + ((r436_z / $(64)) % $(32))), dtype=tl.float32))
    r436_value = r436_value.to(tl.float32)
    r436_value = tl.broadcast_to(r436_value, [$(16384)])
    tl.store($(candidate_vsub_55_ptr) + r436_z, r436_value)
    r437_z = tl.arange($(16384))
    r437_value = (tl.load($(candidate_vexp_7_ptr) + ((((r437_z / $(2048)) % $(8)) * $(2048)) + (((r437_z / $(64)) % $(32)) * $(64)) + (r437_z % $(64))), dtype=tl.float32) * tl.load($(candidate_vsub_55_ptr) + ((((r437_z / $(2048)) % $(8)) * $(2048)) + (((r437_z / $(64)) % $(32)) * $(64)) + (r437_z % $(64))), dtype=tl.float32))
    r437_value = r437_value.to(tl.float32)
    r437_value = tl.broadcast_to(r437_value, [$(16384)])
    tl.store($(candidate_vmul_75_ptr) + r437_z, r437_value)
    r438_row = tl.arange($(64))
    r438_col = tl.arange($(192))
    r438_lane = tl.arange($(32))
    for r438_group in range($(8)) {
      r438_left = tl.load($(candidate_vmul_75_ptr) + (((((((r438_group % $(8)) * $(2048)) + (r438_row[:, None] * $(32)) + r438_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + (((((r438_group % $(8)) * $(2048)) + (r438_row[:, None] * $(32)) + r438_lane[None, :]) % $(32)) * $(64)) + (((((r438_group % $(8)) * $(2048)) + (r438_row[:, None] * $(32)) + r438_lane[None, :]) / $(32)) % $(64))), dtype=tl.float32)
      r438_right = tl.load($(candidate_vto_copy_148_ptr) + (((((((r438_group % $(8)) * $(6144)) + (r438_lane[:, None] * $(192)) + r438_col[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r438_group % $(8)) * $(6144)) + (r438_lane[:, None] * $(192)) + r438_col[None, :]) / $(192)) % $(32)) * $(192)) + ((((r438_group % $(8)) * $(6144)) + (r438_lane[:, None] * $(192)) + r438_col[None, :]) % $(192))), dtype=tl.float32)
      r438_value = tl.dot(r438_left, r438_right)
      r438_value = r438_value.to(tl.float32)
      tl.store($(candidate_vbmm_125_ptr) + ((r438_group * $(12288)) + (r438_row[:, None] * $(192)) + r438_col[None, :]), r438_value)
    }
    r439_z = tl.arange($(98304))
    r439_value = (tl.load($(candidate_vbmm_125_ptr) + ((((r439_z / $(12288)) % $(8)) * $(12288)) + (((r439_z / $(192)) % $(64)) * $(192)) + (r439_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r439_value = r439_value.to(tl.float32)
    r439_value = tl.broadcast_to(r439_value, [$(98304)])
    tl.store($(candidate_vmul_77_ptr) + r439_z, r439_value)
    r440_z = tl.arange($(98304))
    r440_value = tl.load($(candidate_vmul_77_ptr) + ((((r440_z / $(192)) % $(8)) * $(12288)) + (((r440_z / $(1536)) % $(64)) * $(192)) + (r440_z % $(192))), dtype=tl.float32)
    r440_value = r440_value.to(tl.bfloat16)
    r440_value = tl.broadcast_to(r440_value, [$(98304)])
    tl.store($(candidate_vto_copy_154_ptr) + r440_z, r440_value)
    r441_row = tl.arange($(64))
    r441_col = tl.arange($(128))
    r441_lane = tl.arange($(32))
    for r441_group in range($(8)) {
      r441_left = tl.load($(candidate_vexp_7_ptr) + (((((((r441_group % $(8)) * $(2048)) + (r441_row[:, None] * $(32)) + r441_lane[None, :]) / $(2048)) % $(8)) * $(2048)) + (((((r441_group % $(8)) * $(2048)) + (r441_row[:, None] * $(32)) + r441_lane[None, :]) % $(32)) * $(64)) + (((((r441_group % $(8)) * $(2048)) + (r441_row[:, None] * $(32)) + r441_lane[None, :]) / $(32)) % $(64))), dtype=tl.float32)
      r441_right = tl.load($(candidate_vto_copy_151_ptr) + (((((((r441_group % $(8)) * $(4096)) + (r441_lane[:, None] * $(128)) + r441_col[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r441_group % $(8)) * $(4096)) + (r441_lane[:, None] * $(128)) + r441_col[None, :]) / $(128)) % $(32)) * $(128)) + ((((r441_group % $(8)) * $(4096)) + (r441_lane[:, None] * $(128)) + r441_col[None, :]) % $(128))), dtype=tl.float32)
      r441_value = tl.dot(r441_left, r441_right)
      r441_value = r441_value.to(tl.float32)
      tl.store($(candidate_vbmm_126_ptr) + ((r441_group * $(8192)) + (r441_row[:, None] * $(128)) + r441_col[None, :]), r441_value)
    }
    r442_z = tl.arange($(65536))
    r442_value = tl.load($(candidate_vbmm_126_ptr) + ((((r442_z / $(128)) % $(8)) * $(8192)) + (((r442_z / $(1024)) % $(64)) * $(128)) + (r442_z % $(128))), dtype=tl.float32)
    r442_value = r442_value.to(tl.bfloat16)
    r442_value = tl.broadcast_to(r442_value, [$(65536)])
    tl.store($(candidate_vto_copy_155_ptr) + r442_z, r442_value)
    r443_z = tl.arange($(147456))
    r443_value = tl.where(((r443_z / $(1536)) % $(96)) < $(64), tl.load($(candidate_vto_copy_154_ptr) + ((((((r443_z / $(1536)) % $(96)) - $(0)) % $(64)) * $(1536)) + (((r443_z / $(192)) % $(8)) * $(192)) + (r443_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r443_value = r443_value.to(tl.bfloat16)
    r443_value = tl.broadcast_to(r443_value, [$(147456)])
    tl.store($(candidate_vcat_86_ptr) + r443_z, r443_value)
    r444_z = tl.arange($(98304))
    r444_value = tl.where(((r444_z / $(1024)) % $(96)) < $(64), tl.load($(candidate_vto_copy_155_ptr) + ((((((r444_z / $(1024)) % $(96)) - $(0)) % $(64)) * $(1024)) + (((r444_z / $(128)) % $(8)) * $(128)) + (r444_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r444_value = r444_value.to(tl.bfloat16)
    r444_value = tl.broadcast_to(r444_value, [$(98304)])
    tl.store($(candidate_vcat_87_ptr) + r444_z, r444_value)
    r445_z = tl.arange($(147456))
    r445_value = tl.load($(candidate_vcat_86_ptr) + ((((((r445_z / $(1536)) % $(96)) - $(0)) % $(96)) * $(1536)) + (((r445_z / $(192)) % $(8)) * $(192)) + (r445_z % $(192))), dtype=tl.bfloat16)
    r445_value = r445_value.to(tl.bfloat16)
    r445_value = tl.broadcast_to(r445_value, [$(147456)])
    tl.store($(candidate_vcat_89_ptr) + r445_z, r445_value)
    r446_z = tl.arange($(98304))
    r446_value = tl.load($(candidate_vcat_87_ptr) + ((((((r446_z / $(1024)) % $(96)) - $(0)) % $(96)) * $(1024)) + (((r446_z / $(128)) % $(8)) * $(128)) + (r446_z % $(128))), dtype=tl.bfloat16)
    r446_value = r446_value.to(tl.bfloat16)
    r446_value = tl.broadcast_to(r446_value, [$(98304)])
    tl.store($(candidate_vcat_90_ptr) + r446_z, r446_value)
    r447_row = tl.arange($(1024))
    r447_col = tl.arange($(512))
    r447_lane = tl.arange($(96))
    for r447_group in range($(1)) {
      r447_left = tl.load($(candidate_vcat_89_ptr) + (((((r447_row[:, None] * $(96)) + r447_lane[None, :]) % $(96)) * $(1536)) + (((((r447_row[:, None] * $(96)) + r447_lane[None, :]) / $(12288)) % $(8)) * $(192)) + ((((r447_row[:, None] * $(96)) + r447_lane[None, :]) / $(96)) % $(128))), dtype=tl.bfloat16)
      r447_right = tl.load($(candidate_vcat_7_ptr) + ((((((r447_lane[:, None] * $(512)) + r447_col[None, :]) / $(512)) % $(96)) * $(576)) + (((r447_lane[:, None] * $(512)) + r447_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r447_value = tl.dot(r447_left, r447_right)
      r447_value = r447_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_129_ptr) + ((r447_group * $(524288)) + (r447_row[:, None] * $(512)) + r447_col[None, :]), r447_value)
    }
    r448_row = tl.arange($(1024))
    r448_col = tl.arange($(512))
    r448_lane = tl.arange($(96))
    for r448_group in range($(1)) {
      r448_left = tl.load($(candidate_vcat_90_ptr) + (((((r448_row[:, None] * $(96)) + r448_lane[None, :]) % $(96)) * $(1024)) + (((((r448_row[:, None] * $(96)) + r448_lane[None, :]) / $(12288)) % $(8)) * $(128)) + ((((r448_row[:, None] * $(96)) + r448_lane[None, :]) / $(96)) % $(128))), dtype=tl.bfloat16)
      r448_right = tl.load($(candidate_vcat_7_ptr) + ((((((r448_lane[:, None] * $(512)) + r448_col[None, :]) / $(512)) % $(96)) * $(576)) + (((r448_lane[:, None] * $(512)) + r448_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r448_value = tl.dot(r448_left, r448_right)
      r448_value = r448_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_130_ptr) + ((r448_group * $(524288)) + (r448_row[:, None] * $(512)) + r448_col[None, :]), r448_value)
    }
    r449_z = tl.arange($(1048576))
    r449_value = tl.where(((r449_z / $(512)) % $(256)) < $(128), tl.load($(candidate_vbmm_129_ptr) + ((((((((r449_z / $(131072)) % $(8)) * $(65536)) + ((((r449_z / $(512)) % $(256)) - $(0)) * $(512)) + (r449_z % $(512))) / $(65536)) % $(8)) * $(65536)) + ((((((r449_z / $(131072)) % $(8)) * $(128)) + (((r449_z / $(512)) % $(256)) - $(0))) % $(128)) * $(512)) + (r449_z % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vbmm_130_ptr) + ((((((((r449_z / $(131072)) % $(8)) * $(65536)) + ((((r449_z / $(512)) % $(256)) - $(128)) * $(512)) + (r449_z % $(512))) / $(65536)) % $(8)) * $(65536)) + ((((((r449_z / $(131072)) % $(8)) * $(128)) + (((r449_z / $(512)) % $(256)) - $(128))) % $(128)) * $(512)) + (r449_z % $(512))), dtype=tl.bfloat16))
    r449_value = r449_value.to(tl.bfloat16)
    r449_value = tl.broadcast_to(r449_value, [$(1048576)])
    tl.store($(candidate_vcat_91_ptr) + r449_z, r449_value)
    r450_z = tl.arange($(1048576))
    r450_value = (tl.load($(candidate_vadd_50_ptr) + ((((r450_z / $(512)) % $(2048)) * $(512)) + (r450_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vcat_91_ptr) + ((((((((r450_z / $(512)) % $(2048)) * $(512)) + (r450_z % $(512))) / $(131072)) % $(8)) * $(131072)) + ((((r450_z / $(512)) % $(2048)) % $(256)) * $(512)) + (r450_z % $(512))), dtype=tl.bfloat16))
    r450_value = r450_value.to(tl.bfloat16)
    r450_value = tl.broadcast_to(r450_value, [$(1048576)])
    tl.store($(candidate_vadd_54_ptr) + r450_z, r450_value)
    r451_row = tl.arange($(96))
    r451_col = tl.arange($(1024))
    r451_lane = tl.arange($(512))
    for r451_group in range($(1)) {
      r451_left = tl.load($(candidate_vcat_7_ptr) + ((((((r451_row[:, None] * $(512)) + r451_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r451_row[:, None] * $(512)) + r451_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r451_right = tl.load($(candidate_vto_copy_18_ptr) + ((((((r451_lane[:, None] * $(1024)) + r451_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r451_lane[:, None] * $(1024)) + r451_col[None, :]) % $(128)) * $(512)) + ((((r451_lane[:, None] * $(1024)) + r451_col[None, :]) / $(1024)) % $(512))), dtype=tl.bfloat16)
      r451_value = tl.dot(r451_left, r451_right)
      r451_value = r451_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_131_ptr) + ((r451_group * $(98304)) + (r451_row[:, None] * $(1024)) + r451_col[None, :]), r451_value)
    }
    r452_row = tl.arange($(96))
    r452_col = tl.arange($(1024))
    r452_lane = tl.arange($(512))
    for r452_group in range($(1)) {
      r452_left = tl.load($(candidate_vcat_7_ptr) + ((((((r452_row[:, None] * $(512)) + r452_lane[None, :]) / $(512)) % $(96)) * $(576)) + (((r452_row[:, None] * $(512)) + r452_lane[None, :]) % $(512))), dtype=tl.bfloat16)
      r452_right = tl.load($(candidate_vto_copy_18_ptr) + ((((((r452_lane[:, None] * $(1024)) + r452_col[None, :]) / $(128)) % $(8)) * $(131072)) + ((((r452_lane[:, None] * $(1024)) + r452_col[None, :]) % $(128)) * $(512)) + ((((r452_lane[:, None] * $(1024)) + r452_col[None, :]) / $(1024)) % $(512)) + $(65536)), dtype=tl.bfloat16)
      r452_value = tl.dot(r452_left, r452_right)
      r452_value = r452_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_132_ptr) + ((r452_group * $(98304)) + (r452_row[:, None] * $(1024)) + r452_col[None, :]), r452_value)
    }
    r453_z = tl.arange($(49152))
    r453_value = tl.where((r453_z % $(192)) < $(128), tl.load($(candidate_vmm_1_ptr) + ((((((((r453_z / $(1536)) % $(32)) * $(1024)) + (((r453_z / $(192)) % $(8)) * $(128)) + ((r453_z % $(192)) - $(0))) / $(1024)) % $(288)) * $(1536)) + (((((((r453_z / $(1536)) % $(32)) * $(1024)) + (((r453_z / $(192)) % $(8)) * $(128)) + ((r453_z % $(192)) - $(0))) / $(128)) % $(8)) * $(192)) + (((((r453_z / $(1536)) % $(32)) * $(1024)) + (((r453_z / $(192)) % $(8)) * $(128)) + ((r453_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_5_ptr) + ((((((((r453_z / $(1536)) % $(32)) * $(512)) + (((r453_z / $(192)) % $(8)) * $(64)) + ((r453_z % $(192)) - $(128))) / $(64)) % $(8)) * $(18432)) + (((((((r453_z / $(1536)) % $(32)) * $(512)) + (((r453_z / $(192)) % $(8)) * $(64)) + ((r453_z % $(192)) - $(128))) / $(512)) % $(288)) * $(64)) + (((((r453_z / $(1536)) % $(32)) * $(512)) + (((r453_z / $(192)) % $(8)) * $(64)) + ((r453_z % $(192)) - $(128))) % $(64))), dtype=tl.bfloat16))
    r453_value = r453_value.to(tl.bfloat16)
    r453_value = tl.broadcast_to(r453_value, [$(49152)])
    tl.store($(candidate_vcat_92_ptr) + r453_z, r453_value)
    r454_z = tl.arange($(147456))
    r454_value = tl.where((r454_z % $(192)) < $(128), tl.load($(candidate_vbmm_131_ptr) + ((((((((r454_z / $(1536)) % $(96)) * $(1024)) + (((r454_z / $(192)) % $(8)) * $(128)) + ((r454_z % $(192)) - $(0))) / $(1024)) % $(96)) * $(1024)) + (((((((r454_z / $(1536)) % $(96)) * $(1024)) + (((r454_z / $(192)) % $(8)) * $(128)) + ((r454_z % $(192)) - $(0))) / $(128)) % $(8)) * $(128)) + (((((r454_z / $(1536)) % $(96)) * $(1024)) + (((r454_z / $(192)) % $(8)) * $(128)) + ((r454_z % $(192)) - $(0))) % $(128))), dtype=tl.bfloat16), tl.load($(candidate_vcat_7_ptr) + ((((r454_z / $(1536)) % $(96)) * $(576)) + ((((r454_z % $(192)) - $(128)) + $(512)) % $(576))), dtype=tl.bfloat16))
    r454_value = r454_value.to(tl.bfloat16)
    r454_value = tl.broadcast_to(r454_value, [$(147456)])
    tl.store($(candidate_vcat_93_ptr) + r454_z, r454_value)
    r455_z = tl.arange($(49152))
    r455_value = tl.load($(candidate_vcat_92_ptr) + ((((r455_z / $(192)) % $(32)) * $(1536)) + (((r455_z / $(6144)) % $(8)) * $(192)) + (r455_z % $(192))), dtype=tl.bfloat16)
    r455_value = r455_value.to(tl.float32)
    r455_value = tl.broadcast_to(r455_value, [$(49152)])
    tl.store($(candidate_vto_copy_156_ptr) + r455_z, r455_value)
    r456_z = tl.arange($(49152))
    r456_value = tl.load($(candidate_vcat_93_ptr) + ((((r456_z / $(192)) % $(32)) * $(1536)) + (((r456_z / $(6144)) % $(8)) * $(192)) + (r456_z % $(192))), dtype=tl.bfloat16)
    r456_value = r456_value.to(tl.float32)
    r456_value = tl.broadcast_to(r456_value, [$(49152)])
    tl.store($(candidate_vto_copy_157_ptr) + r456_z, r456_value)
    r457_row = tl.arange($(32))
    r457_col = tl.arange($(32))
    r457_lane = tl.arange($(192))
    for r457_group in range($(8)) {
      r457_left = tl.load($(candidate_vto_copy_156_ptr) + (((((((r457_group % $(8)) * $(6144)) + (r457_row[:, None] * $(192)) + r457_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r457_group % $(8)) * $(6144)) + (r457_row[:, None] * $(192)) + r457_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r457_group % $(8)) * $(6144)) + (r457_row[:, None] * $(192)) + r457_lane[None, :]) % $(192))), dtype=tl.float32)
      r457_right = tl.load($(candidate_vto_copy_157_ptr) + (((((((r457_group % $(8)) * $(6144)) + (r457_lane[:, None] * $(32)) + r457_col[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r457_group % $(8)) * $(6144)) + (r457_lane[:, None] * $(32)) + r457_col[None, :]) % $(32)) * $(192)) + (((((r457_group % $(8)) * $(6144)) + (r457_lane[:, None] * $(32)) + r457_col[None, :]) / $(32)) % $(192))), dtype=tl.float32)
      r457_value = tl.dot(r457_left, r457_right)
      r457_value = r457_value.to(tl.float32)
      tl.store($(candidate_vbmm_133_ptr) + ((r457_group * $(1024)) + (r457_row[:, None] * $(32)) + r457_col[None, :]), r457_value)
    }
    r458_z = tl.arange($(8192))
    r458_value = (tl.load($(candidate_vbmm_133_ptr) + ((((r458_z / $(1024)) % $(8)) * $(1024)) + (((r458_z / $(32)) % $(32)) * $(32)) + (r458_z % $(32))), dtype=tl.float32) * 0.07216878364870322)
    r458_value = r458_value.to(tl.float32)
    r458_value = tl.broadcast_to(r458_value, [$(8192)])
    tl.store($(candidate_vmul_78_ptr) + r458_z, r458_value)
    r459_z = tl.arange($(32))
    r459_value = (tl.cast(r459_z, tl.int64) + tl.full([], $(32), dtype=tl.int64))
    r459_value = tl.broadcast_to(r459_value, [$(32)])
    tl.store($(candidate_vadd_55_ptr) + r459_z, r459_value)
    r460_z = tl.arange($(32))
    r460_value = (tl.load($(candidate_vadd_55_ptr) + r460_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r460_value = tl.broadcast_to(r460_value, [$(32)])
    tl.store($(candidate_vsub_56_ptr) + r460_z, r460_value)
    r461_z = tl.arange($(1024))
    r461_value = (tl.cast((r461_z % $(32)), tl.int64) > tl.load($(candidate_vsub_56_ptr) + ((r461_z / $(32)) % $(32)), dtype=tl.int64))
    r461_value = tl.broadcast_to(r461_value, [$(1024)])
    tl.store($(candidate_vgt_25_ptr) + r461_z, r461_value)
    r462_z = tl.arange($(8192))
    r462_value = tl.where(tl.load($(candidate_vgt_25_ptr) + ((((r462_z / $(32)) % $(32)) * $(32)) + (r462_z % $(32))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_78_ptr) + ((((r462_z / $(1024)) % $(8)) * $(1024)) + (((r462_z / $(32)) % $(32)) * $(32)) + (r462_z % $(32))), dtype=tl.float32))
    r462_value = r462_value.to(tl.float32)
    r462_value = tl.broadcast_to(r462_value, [$(8192)])
    tl.store($(candidate_vmasked_fill_25_ptr) + r462_z, r462_value)
    r463_z = tl.arange($(8192))
    r463_lane = tl.arange($(32))
    r463_row_values = tl.load($(candidate_vmasked_fill_25_ptr) + ((((r463_z[:, None] / $(1024)) % $(8)) * $(1024)) + (((r463_z[:, None] / $(32)) % $(32)) * $(32)) + (r463_lane[None, :] % $(32))), dtype=tl.float32)
    r463_maximum = tl.max(r463_row_values, axis=1, keep_dims=true)
    r463_weights = tl.exp(r463_row_values - r463_maximum)
    r463_total = tl.sum(r463_weights, axis=1)
    r463_maximum = tl.max(r463_row_values, axis=1)
    r463_value = tl.exp(tl.load($(candidate_vmasked_fill_25_ptr) + ((((r463_z / $(1024)) % $(8)) * $(1024)) + (((r463_z / $(32)) % $(32)) * $(32)) + (r463_z % $(32))), dtype=tl.float32) - r463_maximum) / r463_total
    r463_value = r463_value.to(tl.float32)
    r463_value = tl.broadcast_to(r463_value, [$(8192)])
    tl.store($(candidate_vsoftmax_17_ptr) + r463_z, r463_value)
    r464_z = tl.arange($(8192))
    r464_value = tl.load($(candidate_vsoftmax_17_ptr) + ((((r464_z / $(1024)) % $(8)) * $(1024)) + (((r464_z / $(32)) % $(32)) * $(32)) + (r464_z % $(32))), dtype=tl.float32)
    r464_value = r464_value.to(tl.bfloat16)
    r464_value = tl.broadcast_to(r464_value, [$(8192)])
    tl.store($(candidate_vto_copy_158_ptr) + r464_z, r464_value)
    r465_row = tl.arange($(32))
    r465_col = tl.arange($(128))
    r465_lane = tl.arange($(32))
    for r465_group in range($(8)) {
      r465_left = tl.load($(candidate_vto_copy_158_ptr) + (((((((r465_group % $(8)) * $(1024)) + (r465_row[:, None] * $(32)) + r465_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + ((((((r465_group % $(8)) * $(1024)) + (r465_row[:, None] * $(32)) + r465_lane[None, :]) / $(32)) % $(32)) * $(32)) + ((((r465_group % $(8)) * $(1024)) + (r465_row[:, None] * $(32)) + r465_lane[None, :]) % $(32))), dtype=tl.bfloat16)
      r465_right = tl.load($(candidate_vbmm_132_ptr) + (((((((r465_group % $(8)) * $(4096)) + (r465_lane[:, None] * $(128)) + r465_col[None, :]) / $(128)) % $(32)) * $(1024)) + ((((((r465_group % $(8)) * $(4096)) + (r465_lane[:, None] * $(128)) + r465_col[None, :]) / $(4096)) % $(8)) * $(128)) + ((((r465_group % $(8)) * $(4096)) + (r465_lane[:, None] * $(128)) + r465_col[None, :]) % $(128))), dtype=tl.bfloat16)
      r465_value = tl.dot(r465_left, r465_right)
      r465_value = r465_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_134_ptr) + ((r465_group * $(4096)) + (r465_row[:, None] * $(128)) + r465_col[None, :]), r465_value)
    }
    r466_z = tl.arange($(256))
    r466_lane = tl.arange($(32))
    r466_row_values = tl.load($(candidate_vmasked_fill_25_ptr) + ((((r466_z[:, None] / $(32)) % $(8)) * $(1024)) + ((r466_z[:, None] % $(32)) * $(32)) + (r466_lane[None, :] % $(32))), dtype=tl.float32)
    r466_maximum = tl.max(r466_row_values, axis=1, keep_dims=true)
    r466_weights = tl.exp(r466_row_values - r466_maximum)
    r466_total = tl.sum(r466_weights, axis=1)
    r466_maximum = tl.max(r466_row_values, axis=1)
    r466_value = r466_maximum + tl.log(r466_total)
    r466_value = r466_value.to(tl.float32)
    r466_value = tl.broadcast_to(r466_value, [$(256)])
    tl.store($(candidate_vlogsumexp_17_ptr) + r466_z, r466_value)
    r467_z = tl.arange($(32768))
    r467_value = tl.load($(candidate_vbmm_134_ptr) + ((((((((r467_z / $(128)) % $(8)) * $(4096)) + ((((r467_z / $(1024)) % $(32)) - $(0)) * $(128)) + (r467_z % $(128))) / $(4096)) % $(8)) * $(4096)) + ((((((r467_z / $(128)) % $(8)) * $(32)) + (((r467_z / $(1024)) % $(32)) - $(0))) % $(32)) * $(128)) + (r467_z % $(128))), dtype=tl.bfloat16)
    r467_value = r467_value.to(tl.bfloat16)
    r467_value = tl.broadcast_to(r467_value, [$(32768)])
    tl.store($(candidate_vcat_94_ptr) + r467_z, r467_value)
    r468_z = tl.arange($(256))
    r468_value = tl.load($(candidate_vlogsumexp_17_ptr) + (((r468_z % $(8)) * $(32)) + ((((r468_z / $(8)) % $(32)) - $(0)) % $(32))), dtype=tl.float32)
    r468_value = r468_value.to(tl.float32)
    r468_value = tl.broadcast_to(r468_value, [$(256)])
    tl.store($(candidate_vcat_95_ptr) + r468_z, r468_value)
    r469_z = tl.arange($(49152))
    r469_value = tl.load($(candidate_vcat_92_ptr) + ((((r469_z / $(192)) % $(32)) * $(1536)) + (((r469_z / $(6144)) % $(8)) * $(192)) + (r469_z % $(192))), dtype=tl.bfloat16)
    r469_value = r469_value.to(tl.float32)
    r469_value = tl.broadcast_to(r469_value, [$(49152)])
    tl.store($(candidate_vto_copy_159_ptr) + r469_z, r469_value)
    r470_z = tl.arange($(49152))
    r470_value = tl.load($(candidate_vcat_93_ptr) + ((((r470_z / $(192)) % $(32)) * $(1536)) + (((r470_z / $(6144)) % $(8)) * $(192)) + (r470_z % $(192))), dtype=tl.bfloat16)
    r470_value = r470_value.to(tl.float32)
    r470_value = tl.broadcast_to(r470_value, [$(49152)])
    tl.store($(candidate_vto_copy_160_ptr) + r470_z, r470_value)
    r471_z = tl.arange($(32768))
    r471_value = tl.load($(candidate_vbmm_132_ptr) + ((((r471_z / $(128)) % $(32)) * $(1024)) + (((r471_z / $(4096)) % $(8)) * $(128)) + (r471_z % $(128))), dtype=tl.bfloat16)
    r471_value = r471_value.to(tl.float32)
    r471_value = tl.broadcast_to(r471_value, [$(32768)])
    tl.store($(candidate_vto_copy_161_ptr) + r471_z, r471_value)
    r472_z = tl.arange($(32768))
    r472_value = tl.load($(candidate_vmm_22_ptr) + ((((r472_z / $(128)) % $(32)) * $(1024)) + (((r472_z / $(4096)) % $(8)) * $(128)) + (r472_z % $(128))), dtype=tl.bfloat16)
    r472_value = r472_value.to(tl.float32)
    r472_value = tl.broadcast_to(r472_value, [$(32768)])
    tl.store($(candidate_vto_copy_162_ptr) + r472_z, r472_value)
    r473_z = tl.arange($(32768))
    r473_value = tl.load($(candidate_vcat_94_ptr) + ((((r473_z / $(128)) % $(32)) * $(1024)) + (((r473_z / $(4096)) % $(8)) * $(128)) + (r473_z % $(128))), dtype=tl.bfloat16)
    r473_value = r473_value.to(tl.float32)
    r473_value = tl.broadcast_to(r473_value, [$(32768)])
    tl.store($(candidate_vto_copy_163_ptr) + r473_z, r473_value)
    r474_row = tl.arange($(32))
    r474_col = tl.arange($(32))
    r474_lane = tl.arange($(192))
    for r474_group in range($(8)) {
      r474_left = tl.load($(candidate_vto_copy_159_ptr) + (((((((r474_group % $(8)) * $(6144)) + (r474_row[:, None] * $(192)) + r474_lane[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r474_group % $(8)) * $(6144)) + (r474_row[:, None] * $(192)) + r474_lane[None, :]) / $(192)) % $(32)) * $(192)) + ((((r474_group % $(8)) * $(6144)) + (r474_row[:, None] * $(192)) + r474_lane[None, :]) % $(192))), dtype=tl.float32)
      r474_right = tl.load($(candidate_vto_copy_160_ptr) + (((((((r474_group % $(8)) * $(6144)) + (r474_lane[:, None] * $(32)) + r474_col[None, :]) / $(6144)) % $(8)) * $(6144)) + (((((r474_group % $(8)) * $(6144)) + (r474_lane[:, None] * $(32)) + r474_col[None, :]) % $(32)) * $(192)) + (((((r474_group % $(8)) * $(6144)) + (r474_lane[:, None] * $(32)) + r474_col[None, :]) / $(32)) % $(192))), dtype=tl.float32)
      r474_value = tl.dot(r474_left, r474_right)
      r474_value = r474_value.to(tl.float32)
      tl.store($(candidate_vbmm_135_ptr) + ((r474_group * $(1024)) + (r474_row[:, None] * $(32)) + r474_col[None, :]), r474_value)
    }
    r475_z = tl.arange($(8192))
    r475_value = (tl.load($(candidate_vbmm_135_ptr) + ((((r475_z / $(1024)) % $(8)) * $(1024)) + (((r475_z / $(32)) % $(32)) * $(32)) + (r475_z % $(32))), dtype=tl.float32) * 0.07216878364870322)
    r475_value = r475_value.to(tl.float32)
    r475_value = tl.broadcast_to(r475_value, [$(8192)])
    tl.store($(candidate_vmul_79_ptr) + r475_z, r475_value)
    r476_z = tl.arange($(32))
    r476_value = (tl.cast(r476_z, tl.int64) + tl.full([], $(32), dtype=tl.int64))
    r476_value = tl.broadcast_to(r476_value, [$(32)])
    tl.store($(candidate_vadd_56_ptr) + r476_z, r476_value)
    r477_z = tl.arange($(32))
    r477_value = (tl.load($(candidate_vadd_56_ptr) + r477_z, dtype=tl.int64) - tl.full([], $(32), dtype=tl.int64))
    r477_value = tl.broadcast_to(r477_value, [$(32)])
    tl.store($(candidate_vsub_57_ptr) + r477_z, r477_value)
    r478_z = tl.arange($(1024))
    r478_value = (tl.cast((r478_z % $(32)), tl.int64) > tl.load($(candidate_vsub_57_ptr) + ((r478_z / $(32)) % $(32)), dtype=tl.int64))
    r478_value = tl.broadcast_to(r478_value, [$(1024)])
    tl.store($(candidate_vgt_26_ptr) + r478_z, r478_value)
    r479_z = tl.arange($(8192))
    r479_value = tl.where(tl.load($(candidate_vgt_26_ptr) + ((((r479_z / $(32)) % $(32)) * $(32)) + (r479_z % $(32))), dtype=tl.int1), -inf, tl.load($(candidate_vmul_79_ptr) + ((((r479_z / $(1024)) % $(8)) * $(1024)) + (((r479_z / $(32)) % $(32)) * $(32)) + (r479_z % $(32))), dtype=tl.float32))
    r479_value = r479_value.to(tl.float32)
    r479_value = tl.broadcast_to(r479_value, [$(8192)])
    tl.store($(candidate_vmasked_fill_26_ptr) + r479_z, r479_value)
    r480_z = tl.arange($(8192))
    r480_value = (tl.load($(candidate_vmasked_fill_26_ptr) + ((((r480_z / $(1024)) % $(8)) * $(1024)) + (((r480_z / $(32)) % $(32)) * $(32)) + (r480_z % $(32))), dtype=tl.float32) - tl.load($(candidate_vcat_95_ptr) + ((((r480_z / $(32)) % $(32)) * $(8)) + ((r480_z / $(1024)) % $(8))), dtype=tl.float32))
    r480_value = r480_value.to(tl.float32)
    r480_value = tl.broadcast_to(r480_value, [$(8192)])
    tl.store($(candidate_vsub_58_ptr) + r480_z, r480_value)
    r481_z = tl.arange($(8192))
    r481_value = tl.exp(tl.load($(candidate_vsub_58_ptr) + ((((r481_z / $(1024)) % $(8)) * $(1024)) + (((r481_z / $(32)) % $(32)) * $(32)) + (r481_z % $(32))), dtype=tl.float32))
    r481_value = r481_value.to(tl.float32)
    r481_value = tl.broadcast_to(r481_value, [$(8192)])
    tl.store($(candidate_vexp_8_ptr) + r481_z, r481_value)
    r482_z = tl.arange($(32768))
    r482_value = (tl.load($(candidate_vto_copy_162_ptr) + ((((r482_z / $(4096)) % $(8)) * $(4096)) + (((r482_z / $(128)) % $(32)) * $(128)) + (r482_z % $(128))), dtype=tl.float32) * tl.load($(candidate_vto_copy_163_ptr) + ((((r482_z / $(4096)) % $(8)) * $(4096)) + (((r482_z / $(128)) % $(32)) * $(128)) + (r482_z % $(128))), dtype=tl.float32))
    r482_value = r482_value.to(tl.float32)
    r482_value = tl.broadcast_to(r482_value, [$(32768)])
    tl.store($(candidate_vmul_80_ptr) + r482_z, r482_value)
    r483_z = tl.arange($(256))
    r483_lane = tl.arange($(128))
    r483_value = tl.sum(tl.load($(candidate_vmul_80_ptr) + ((((r483_z[:, None] / $(32)) % $(8)) * $(4096)) + ((r483_z[:, None] % $(32)) * $(128)) + r483_lane[None, :]), dtype=tl.float32), axis=1)
    r483_value = r483_value.to(tl.float32)
    r483_value = tl.broadcast_to(r483_value, [$(256)])
    tl.store($(candidate_vsum_17_ptr) + r483_z, r483_value)
    r484_row = tl.arange($(32))
    r484_col = tl.arange($(32))
    r484_lane = tl.arange($(128))
    for r484_group in range($(8)) {
      r484_left = tl.load($(candidate_vto_copy_162_ptr) + (((((((r484_group % $(8)) * $(4096)) + (r484_row[:, None] * $(128)) + r484_lane[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r484_group % $(8)) * $(4096)) + (r484_row[:, None] * $(128)) + r484_lane[None, :]) / $(128)) % $(32)) * $(128)) + ((((r484_group % $(8)) * $(4096)) + (r484_row[:, None] * $(128)) + r484_lane[None, :]) % $(128))), dtype=tl.float32)
      r484_right = tl.load($(candidate_vto_copy_161_ptr) + (((((((r484_group % $(8)) * $(4096)) + (r484_lane[:, None] * $(32)) + r484_col[None, :]) / $(4096)) % $(8)) * $(4096)) + (((((r484_group % $(8)) * $(4096)) + (r484_lane[:, None] * $(32)) + r484_col[None, :]) % $(32)) * $(128)) + (((((r484_group % $(8)) * $(4096)) + (r484_lane[:, None] * $(32)) + r484_col[None, :]) / $(32)) % $(128))), dtype=tl.float32)
      r484_value = tl.dot(r484_left, r484_right)
      r484_value = r484_value.to(tl.float32)
      tl.store($(candidate_vbmm_136_ptr) + ((r484_group * $(1024)) + (r484_row[:, None] * $(32)) + r484_col[None, :]), r484_value)
    }
    r485_z = tl.arange($(8192))
    r485_value = (tl.load($(candidate_vbmm_136_ptr) + ((((r485_z / $(1024)) % $(8)) * $(1024)) + (((r485_z / $(32)) % $(32)) * $(32)) + (r485_z % $(32))), dtype=tl.float32) - tl.load($(candidate_vsum_17_ptr) + ((((r485_z / $(1024)) % $(8)) * $(32)) + ((r485_z / $(32)) % $(32))), dtype=tl.float32))
    r485_value = r485_value.to(tl.float32)
    r485_value = tl.broadcast_to(r485_value, [$(8192)])
    tl.store($(candidate_vsub_59_ptr) + r485_z, r485_value)
    r486_z = tl.arange($(8192))
    r486_value = (tl.load($(candidate_vexp_8_ptr) + ((((r486_z / $(1024)) % $(8)) * $(1024)) + (((r486_z / $(32)) % $(32)) * $(32)) + (r486_z % $(32))), dtype=tl.float32) * tl.load($(candidate_vsub_59_ptr) + ((((r486_z / $(1024)) % $(8)) * $(1024)) + (((r486_z / $(32)) % $(32)) * $(32)) + (r486_z % $(32))), dtype=tl.float32))
    r486_value = r486_value.to(tl.float32)
    r486_value = tl.broadcast_to(r486_value, [$(8192)])
    tl.store($(candidate_vmul_81_ptr) + r486_z, r486_value)
    r487_row = tl.arange($(32))
    r487_col = tl.arange($(192))
    r487_lane = tl.arange($(32))
    for r487_group in range($(8)) {
      r487_left = tl.load($(candidate_vmul_81_ptr) + (((((((r487_group % $(8)) * $(1024)) + (r487_row[:, None] * $(32)) + r487_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + (((((r487_group % $(8)) * $(1024)) + (r487_row[:, None] * $(32)) + r487_lane[None, :]) % $(32)) * $(32)) + (((((r487_group % $(8)) * $(1024)) + (r487_row[:, None] * $(32)) + r487_lane[None, :]) / $(32)) % $(32))), dtype=tl.float32)
      r487_right = tl.load($(candidate_vto_copy_159_ptr) + (((((((r487_group % $(8)) * $(6144)) + (r487_lane[:, None] * $(192)) + r487_col[None, :]) / $(6144)) % $(8)) * $(6144)) + ((((((r487_group % $(8)) * $(6144)) + (r487_lane[:, None] * $(192)) + r487_col[None, :]) / $(192)) % $(32)) * $(192)) + ((((r487_group % $(8)) * $(6144)) + (r487_lane[:, None] * $(192)) + r487_col[None, :]) % $(192))), dtype=tl.float32)
      r487_value = tl.dot(r487_left, r487_right)
      r487_value = r487_value.to(tl.float32)
      tl.store($(candidate_vbmm_138_ptr) + ((r487_group * $(6144)) + (r487_row[:, None] * $(192)) + r487_col[None, :]), r487_value)
    }
    r488_z = tl.arange($(49152))
    r488_value = (tl.load($(candidate_vbmm_138_ptr) + ((((r488_z / $(6144)) % $(8)) * $(6144)) + (((r488_z / $(192)) % $(32)) * $(192)) + (r488_z % $(192))), dtype=tl.float32) * 0.07216878364870322)
    r488_value = r488_value.to(tl.float32)
    r488_value = tl.broadcast_to(r488_value, [$(49152)])
    tl.store($(candidate_vmul_83_ptr) + r488_z, r488_value)
    r489_z = tl.arange($(49152))
    r489_value = tl.load($(candidate_vmul_83_ptr) + ((((r489_z / $(192)) % $(8)) * $(6144)) + (((r489_z / $(1536)) % $(32)) * $(192)) + (r489_z % $(192))), dtype=tl.float32)
    r489_value = r489_value.to(tl.bfloat16)
    r489_value = tl.broadcast_to(r489_value, [$(49152)])
    tl.store($(candidate_vto_copy_165_ptr) + r489_z, r489_value)
    r490_row = tl.arange($(32))
    r490_col = tl.arange($(128))
    r490_lane = tl.arange($(32))
    for r490_group in range($(8)) {
      r490_left = tl.load($(candidate_vexp_8_ptr) + (((((((r490_group % $(8)) * $(1024)) + (r490_row[:, None] * $(32)) + r490_lane[None, :]) / $(1024)) % $(8)) * $(1024)) + (((((r490_group % $(8)) * $(1024)) + (r490_row[:, None] * $(32)) + r490_lane[None, :]) % $(32)) * $(32)) + (((((r490_group % $(8)) * $(1024)) + (r490_row[:, None] * $(32)) + r490_lane[None, :]) / $(32)) % $(32))), dtype=tl.float32)
      r490_right = tl.load($(candidate_vto_copy_162_ptr) + (((((((r490_group % $(8)) * $(4096)) + (r490_lane[:, None] * $(128)) + r490_col[None, :]) / $(4096)) % $(8)) * $(4096)) + ((((((r490_group % $(8)) * $(4096)) + (r490_lane[:, None] * $(128)) + r490_col[None, :]) / $(128)) % $(32)) * $(128)) + ((((r490_group % $(8)) * $(4096)) + (r490_lane[:, None] * $(128)) + r490_col[None, :]) % $(128))), dtype=tl.float32)
      r490_value = tl.dot(r490_left, r490_right)
      r490_value = r490_value.to(tl.float32)
      tl.store($(candidate_vbmm_139_ptr) + ((r490_group * $(4096)) + (r490_row[:, None] * $(128)) + r490_col[None, :]), r490_value)
    }
    r491_z = tl.arange($(32768))
    r491_value = tl.load($(candidate_vbmm_139_ptr) + ((((r491_z / $(128)) % $(8)) * $(4096)) + (((r491_z / $(1024)) % $(32)) * $(128)) + (r491_z % $(128))), dtype=tl.float32)
    r491_value = r491_value.to(tl.bfloat16)
    r491_value = tl.broadcast_to(r491_value, [$(32768)])
    tl.store($(candidate_vto_copy_166_ptr) + r491_z, r491_value)
    r492_z = tl.arange($(147456))
    r492_value = tl.where(((r492_z / $(1536)) % $(96)) < $(32), tl.load($(candidate_vto_copy_165_ptr) + ((((((r492_z / $(1536)) % $(96)) - $(0)) % $(32)) * $(1536)) + (((r492_z / $(192)) % $(8)) * $(192)) + (r492_z % $(192))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r492_value = r492_value.to(tl.bfloat16)
    r492_value = tl.broadcast_to(r492_value, [$(147456)])
    tl.store($(candidate_vcat_96_ptr) + r492_z, r492_value)
    r493_z = tl.arange($(98304))
    r493_value = tl.where(((r493_z / $(1024)) % $(96)) < $(32), tl.load($(candidate_vto_copy_166_ptr) + ((((((r493_z / $(1024)) % $(96)) - $(0)) % $(32)) * $(1024)) + (((r493_z / $(128)) % $(8)) * $(128)) + (r493_z % $(128))), dtype=tl.bfloat16), tl.cast(0.0, tl.bfloat16))
    r493_value = r493_value.to(tl.bfloat16)
    r493_value = tl.broadcast_to(r493_value, [$(98304)])
    tl.store($(candidate_vcat_97_ptr) + r493_z, r493_value)
    r494_z = tl.arange($(147456))
    r494_value = tl.load($(candidate_vcat_96_ptr) + ((((((r494_z / $(1536)) % $(96)) - $(0)) % $(96)) * $(1536)) + (((r494_z / $(192)) % $(8)) * $(192)) + (r494_z % $(192))), dtype=tl.bfloat16)
    r494_value = r494_value.to(tl.bfloat16)
    r494_value = tl.broadcast_to(r494_value, [$(147456)])
    tl.store($(candidate_vcat_99_ptr) + r494_z, r494_value)
    r495_z = tl.arange($(98304))
    r495_value = tl.load($(candidate_vcat_97_ptr) + ((((((r495_z / $(1024)) % $(96)) - $(0)) % $(96)) * $(1024)) + (((r495_z / $(128)) % $(8)) * $(128)) + (r495_z % $(128))), dtype=tl.bfloat16)
    r495_value = r495_value.to(tl.bfloat16)
    r495_value = tl.broadcast_to(r495_value, [$(98304)])
    tl.store($(candidate_vcat_100_ptr) + r495_z, r495_value)
    r496_row = tl.arange($(1024))
    r496_col = tl.arange($(512))
    r496_lane = tl.arange($(96))
    for r496_group in range($(1)) {
      r496_left = tl.load($(candidate_vcat_99_ptr) + (((((r496_row[:, None] * $(96)) + r496_lane[None, :]) % $(96)) * $(1536)) + (((((r496_row[:, None] * $(96)) + r496_lane[None, :]) / $(12288)) % $(8)) * $(192)) + ((((r496_row[:, None] * $(96)) + r496_lane[None, :]) / $(96)) % $(128))), dtype=tl.bfloat16)
      r496_right = tl.load($(candidate_vcat_7_ptr) + ((((((r496_lane[:, None] * $(512)) + r496_col[None, :]) / $(512)) % $(96)) * $(576)) + (((r496_lane[:, None] * $(512)) + r496_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r496_value = tl.dot(r496_left, r496_right)
      r496_value = r496_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_142_ptr) + ((r496_group * $(524288)) + (r496_row[:, None] * $(512)) + r496_col[None, :]), r496_value)
    }
    r497_row = tl.arange($(1024))
    r497_col = tl.arange($(512))
    r497_lane = tl.arange($(96))
    for r497_group in range($(1)) {
      r497_left = tl.load($(candidate_vcat_100_ptr) + (((((r497_row[:, None] * $(96)) + r497_lane[None, :]) % $(96)) * $(1024)) + (((((r497_row[:, None] * $(96)) + r497_lane[None, :]) / $(12288)) % $(8)) * $(128)) + ((((r497_row[:, None] * $(96)) + r497_lane[None, :]) / $(96)) % $(128))), dtype=tl.bfloat16)
      r497_right = tl.load($(candidate_vcat_7_ptr) + ((((((r497_lane[:, None] * $(512)) + r497_col[None, :]) / $(512)) % $(96)) * $(576)) + (((r497_lane[:, None] * $(512)) + r497_col[None, :]) % $(512))), dtype=tl.bfloat16)
      r497_value = tl.dot(r497_left, r497_right)
      r497_value = r497_value.to(tl.bfloat16)
      tl.store($(candidate_vbmm_143_ptr) + ((r497_group * $(524288)) + (r497_row[:, None] * $(512)) + r497_col[None, :]), r497_value)
    }
    r498_z = tl.arange($(1048576))
    r498_value = tl.where(((r498_z / $(512)) % $(256)) < $(128), tl.load($(candidate_vbmm_142_ptr) + ((((((((r498_z / $(131072)) % $(8)) * $(65536)) + ((((r498_z / $(512)) % $(256)) - $(0)) * $(512)) + (r498_z % $(512))) / $(65536)) % $(8)) * $(65536)) + ((((((r498_z / $(131072)) % $(8)) * $(128)) + (((r498_z / $(512)) % $(256)) - $(0))) % $(128)) * $(512)) + (r498_z % $(512))), dtype=tl.bfloat16), tl.load($(candidate_vbmm_143_ptr) + ((((((((r498_z / $(131072)) % $(8)) * $(65536)) + ((((r498_z / $(512)) % $(256)) - $(128)) * $(512)) + (r498_z % $(512))) / $(65536)) % $(8)) * $(65536)) + ((((((r498_z / $(131072)) % $(8)) * $(128)) + (((r498_z / $(512)) % $(256)) - $(128))) % $(128)) * $(512)) + (r498_z % $(512))), dtype=tl.bfloat16))
    r498_value = r498_value.to(tl.bfloat16)
    r498_value = tl.broadcast_to(r498_value, [$(1048576)])
    tl.store($(candidate_vcat_101_ptr) + r498_z, r498_value)
    r499_z = tl.arange($(1048576))
    r499_value = (tl.load($(candidate_vadd_54_ptr) + ((((r499_z / $(512)) % $(2048)) * $(512)) + (r499_z % $(512))), dtype=tl.bfloat16) + tl.load($(candidate_vcat_101_ptr) + ((((((((r499_z / $(512)) % $(2048)) * $(512)) + (r499_z % $(512))) / $(131072)) % $(8)) * $(131072)) + ((((r499_z / $(512)) % $(2048)) % $(256)) * $(512)) + (r499_z % $(512))), dtype=tl.bfloat16))
    r499_value = r499_value.to(tl.bfloat16)
    r499_value = tl.broadcast_to(r499_value, [$(1048576)])
    tl.store($(candidate_vadd_58_ptr) + r499_z, r499_value)
    z = tl.arange($(1048576))
    value = tl.load($(candidate_vadd_58_ptr) + ((((z / $(512)) % $(2048)) * $(512)) + (z % $(512))), dtype=tl.bfloat16)
    value = tl.broadcast_to(value, [$(1048576)])
    tl.store($(out_ptr) + z, value)
  }

end Kernels
end
end VeriTile.Bench.PretrainKernels.local_mla_fa4_absorb_seqused_wgrad
